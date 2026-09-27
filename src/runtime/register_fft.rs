//! Register-resident fused kernels for power-of-two lines longer than
//! workgroup storage holds.
//!
//! Invocation `t` keeps `N / workgroup_size` elements of its line in
//! registers through a self-sorting (Stockham) radix schedule. Stage `s` with
//! radix `R` owns units `u = t + m * workgroup_size`; it reads
//! `x[u + q * N / R]`, multiplies by `W_(P*R)^((u % P) * q)` where `P` is the
//! product of the earlier radices, runs `DFT_R`, and produces output `k` at
//! `(u / P) * P * R + k * P + u % P`. The first stage reads the line and the
//! last stage writes it, both coalesced.
//!
//! Between stages the values pass through a workgroup exchange buffer. When
//! the line is longer than the buffer the exchange runs in rounds: in round
//! `r` every invocation writes, then reads, only the elements whose line
//! position lies in `[r * E, (r + 1) * E)`. Every position is written once per
//! exchange, so no invocation reads workgroup memory it has not written.

use std::collections::BTreeSet;

use crate::config::FftDirection;
use crate::runtime::axis_plan::{
    complex_wgsl, radix_root_wgsl, scaled_complex_expr, specialize_complex_wgsl,
    twiddle_lookup_wgsl, wgsl_line_base_fn, AxisPrecision, FusedPow2StageWgslConfig,
};
use crate::runtime::pipeline_cache::RegisterSchedule;

/// Elements each invocation keeps in registers: one radix-16 unit per
/// stage. Longer lines would need more registers per invocation for little
/// gain over two fused passes, and much longer shaders, which DX12's FXC
/// compiler takes seconds to build.
const VALUES_PER_INVOCATION: usize = 16;
/// Largest radix of a register stage.
const MAX_RADIX: usize = 16;
/// Smallest exchange buffer; the bank swizzle permutes 256-element blocks.
const MIN_EXCHANGE_LEN: usize = 256;

/// Workgroup size and schedule of a register-resident kernel for an `f32`
/// power-of-two line of `axis_length`, or `None` when the device cannot run
/// one.
pub(crate) fn register_schedule(
    axis_length: usize,
    precision: AxisPrecision,
    limits: &wgpu::Limits,
) -> Option<(u32, RegisterSchedule)> {
    if precision != AxisPrecision::F32 || !axis_length.is_power_of_two() {
        return None;
    }
    let max_invocations = limits
        .max_compute_invocations_per_workgroup
        .min(limits.max_compute_workgroup_size_x) as usize;
    let workgroup_size = axis_length / VALUES_PER_INVOCATION;
    if workgroup_size == 0 || workgroup_size > max_invocations {
        return None;
    }

    let storage_elements = limits.max_compute_workgroup_storage_size as usize
        / precision.complex_size_bytes() as usize;
    let exchange_len = storage_elements.min(axis_length);
    if exchange_len < MIN_EXCHANGE_LEN {
        return None;
    }
    let exchange_len = 1usize << exchange_len.ilog2();

    let mut radices = Vec::new();
    let mut rest = axis_length;
    while rest.is_multiple_of(MAX_RADIX) {
        radices.push(MAX_RADIX);
        rest /= MAX_RADIX;
    }
    if rest > 1 {
        radices.push(rest);
    }
    if radices
        .iter()
        .any(|&radix| !VALUES_PER_INVOCATION.is_multiple_of(radix))
    {
        return None;
    }
    Some((
        workgroup_size as u32,
        RegisterSchedule {
            radices,
            exchange_len,
        },
    ))
}

/// WGSL of a register-resident kernel; one workgroup transforms one line.
pub(crate) fn generate_register_fft_wgsl(
    config: &FusedPow2StageWgslConfig<'_>,
    schedule: &RegisterSchedule,
) -> String {
    let n = config.axis_length;
    let workgroup = config.workgroup_size as usize;
    let values = n / workgroup;
    let radices = &schedule.radices;
    let exchange = schedule.exchange_len.min(n);
    debug_assert_eq!(config.rank, config.dims.len());
    debug_assert_eq!(n, config.dims[config.axis]);
    debug_assert_eq!(values * workgroup, n);
    debug_assert_eq!(radices.iter().product::<usize>(), n);
    debug_assert!(radices.iter().all(|&radix| values.is_multiple_of(radix)));
    debug_assert!(exchange.is_power_of_two() && n.is_multiple_of(exchange));

    let direction = config.direction;
    let precision = config.precision;
    let mut body = String::new();

    let first = radices[0];
    for m in 0..values / first {
        for q in 0..first {
            let offset = m * workgroup + q * (n / first);
            body.push_str(&format!(
                "  let x0_{}: vec2<f32> = src[base + {} * STRIDE];\n",
                m * first + q,
                thread_offset(offset)
            ));
        }
    }

    let mut previous = 1usize;
    for (stage, &radix) in radices.iter().enumerate() {
        let span = previous * radix;
        if previous > 1 && previous <= workgroup {
            body.push_str(&format!("  let j{stage}: u32 = t & {}u;\n", previous - 1));
        }
        for m in 0..values / radix {
            let inputs = (0..radix)
                .map(|q| {
                    let input = format!("x{stage}_{}", m * radix + q);
                    if previous == 1 || q == 0 {
                        return input;
                    }
                    let j = if previous <= workgroup {
                        format!("j{stage}")
                    } else {
                        format!("({} & {}u)", thread_offset(m * workgroup), previous - 1)
                    };
                    let twiddled = format!("w{stage}_{m}_{q}");
                    body.push_str(&format!(
                        "  let {twiddled}: vec2<f32> = c_mul(twiddle({j} * {}u), {input});\n",
                        q * (n / span)
                    ));
                    twiddled
                })
                .collect::<Vec<_>>();
            let outputs = emit_dft(
                &mut body,
                &format!("f{stage}_{m}"),
                &inputs,
                direction,
                precision,
            );
            for (k, output) in outputs.iter().enumerate() {
                body.push_str(&format!(
                    "  let y{stage}_{}: vec2<f32> = {output};\n",
                    m * radix + k
                ));
            }
        }
        if stage + 1 == radices.len() {
            break;
        }
        emit_exchange(
            &mut body,
            stage,
            ExchangeGeometry {
                n,
                workgroup,
                values,
                radix,
                previous,
                next_radix: radices[stage + 1],
                exchange,
            },
        );
        previous = span;
    }

    let last = radices.len() - 1;
    let radix = radices[last];
    let scale_factor = config.apply_scale.then_some(config.scale_factor);
    for m in 0..values / radix {
        for k in 0..radix {
            let value = scaled_complex_expr(
                &format!("y{last}_{}", m * radix + k),
                scale_factor,
                precision,
            );
            body.push_str(&format!(
                "  dst[base + {} * STRIDE] = {value};\n",
                thread_offset(m * workgroup + k * previous)
            ));
        }
    }

    specialize_complex_wgsl(
        format!(
            r#"struct Params {{
  total: u32,
  baseIndex: u32,
  lineOffset: u32,
  elementBase: u32,
}};

@group(0) @binding(0) var<storage, read> src: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> dst: array<vec2<f32>>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read> axisTwiddles: array<vec2<f32>>;

{complex_wgsl}
{twiddle_lookup_wgsl}

const N: u32 = {n}u;
const STRIDE: u32 = {stride}u;

var<workgroup> exchange: array<vec2<f32>, {exchange}>;

// Spreads the 16 consecutive elements of each 256-element block over all
// banks so strided exchange accesses do not conflict.
fn swizzle(index: u32) -> u32 {{
  return index ^ ((index >> 4u) & 15u);
}}

{line_base_fn}

@compute @workgroup_size({workgroup}, 1, 1)
fn main({entry_params}) {{
  {flat_workgroup_index}
  let firstLine: u32 = params.baseIndex / N;
  let totalLines: u32 = params.total / N;
  if (firstLine >= totalLines) {{
    return;
  }}
  if (wgFlat >= totalLines - firstLine) {{
    return;
  }}
  let base: u32 = line_base(params.lineOffset + firstLine + wgFlat) - params.elementBase;
  let t: u32 = lid.x;
{body}}}
"#,
            complex_wgsl = complex_wgsl(),
            twiddle_lookup_wgsl = twiddle_lookup_wgsl(direction, precision),
            stride = config.stride_complex,
            line_base_fn = wgsl_line_base_fn(config.rank, config.axis, config.dims),
            entry_params = crate::runtime::dispatch::WGSL_FLAT_ENTRY_PARAMS,
            flat_workgroup_index = crate::runtime::dispatch::WGSL_FLAT_WORKGROUP_INDEX,
        ),
        precision,
    )
}

/// `t + offset` as WGSL.
fn thread_offset(offset: usize) -> String {
    if offset == 0 {
        "t".to_owned()
    } else {
        format!("(t + {offset}u)")
    }
}

#[derive(Clone, Copy)]
struct ExchangeGeometry {
    n: usize,
    workgroup: usize,
    values: usize,
    radix: usize,
    previous: usize,
    next_radix: usize,
    exchange: usize,
}

impl ExchangeGeometry {
    /// Line position of output `k` of unit `m` for invocation `t`.
    fn write_position(self, t: usize, m: usize, k: usize) -> usize {
        let u = t + m * self.workgroup;
        (u / self.previous) * self.previous * self.radix + k * self.previous + u % self.previous
    }

    /// Line position of the next stage's input `q` of unit `m`.
    fn read_position(self, t: usize, m: usize, q: usize) -> usize {
        t + m * self.workgroup + q * (self.n / self.next_radix)
    }

    /// How the exchange round of one element depends on the invocation.
    fn round_pattern(self, position: impl Fn(usize) -> usize) -> RoundPattern {
        let rounds = (0..self.workgroup)
            .map(|t| position(t) / self.exchange)
            .collect::<Vec<_>>();
        let base = rounds[0];
        if rounds.iter().all(|&round| round == base) {
            return RoundPattern::Constant(base);
        }
        for shift in 0..self.workgroup.trailing_zeros() {
            if rounds
                .iter()
                .enumerate()
                .all(|(t, &round)| round == base + (t >> shift))
            {
                return RoundPattern::Stepped { shift, base };
            }
        }
        RoundPattern::Varying(rounds.into_iter().collect())
    }
}

/// The exchange round an element uses, as a function of the invocation `t`.
#[derive(Clone, PartialEq, Eq)]
enum RoundPattern {
    /// Every invocation uses this round.
    Constant(usize),
    /// Invocation `t` uses round `base + (t >> shift)`, so one test per round
    /// covers every such element of the invocation.
    Stepped { shift: u32, base: usize },
    /// Any other dependence: each element tests its own position. Holds the
    /// rounds that some invocation uses.
    Varying(BTreeSet<usize>),
}

/// One register's move through the exchange buffer.
struct ExchangeAccess {
    register: usize,
    position: String,
    /// `C` when the position is `t + C`.
    thread_constant: Option<usize>,
    pattern: RoundPattern,
}

impl ExchangeAccess {
    /// The position relative to the round starting at `offset`.
    fn local_position(&self, offset: usize) -> String {
        match self.thread_constant {
            Some(constant) if constant >= offset => thread_offset(constant - offset),
            _ => local_position(&self.position, offset),
        }
    }
}

/// Moves stage `stage`'s outputs `y{stage}_*` to the next stage's inputs
/// `x{stage + 1}_*` through the exchange buffer.
fn emit_exchange(body: &mut String, stage: usize, geometry: ExchangeGeometry) {
    let ExchangeGeometry {
        workgroup,
        values,
        radix,
        previous,
        next_radix,
        exchange,
        ..
    } = geometry;
    let next = stage + 1;

    let mut writes = Vec::new();
    for m in 0..values / radix {
        let u = thread_offset(m * workgroup);
        for k in 0..radix {
            let register = m * radix + k;
            let position = if previous == 1 {
                format!("{u} * {radix}u + {k}u")
            } else {
                format!(
                    "(({u} >> {}u) * {}u) + {}u + ({u} & {}u)",
                    previous.trailing_zeros(),
                    previous * radix,
                    k * previous,
                    previous - 1
                )
            };
            let pattern = geometry.round_pattern(|t| geometry.write_position(t, m, k));
            if matches!(pattern, RoundPattern::Varying(_)) {
                body.push_str(&format!("  let d{stage}_{register}: u32 = {position};\n"));
            }
            writes.push(ExchangeAccess {
                register,
                position,
                thread_constant: None,
                pattern,
            });
        }
    }
    let mut reads = Vec::new();
    for m in 0..values / next_radix {
        for q in 0..next_radix {
            let register = m * next_radix + q;
            let constant = m * workgroup + q * (geometry.n / next_radix);
            let position = thread_offset(constant);
            let pattern = geometry.round_pattern(|t| geometry.read_position(t, m, q));
            if !matches!(pattern, RoundPattern::Constant(_)) {
                body.push_str(&format!("  var x{next}_{register}: vec2<f32>;\n"));
            }
            reads.push(ExchangeAccess {
                register,
                position,
                thread_constant: Some(constant),
                pattern,
            });
        }
    }

    let mask = exchange - 1;
    for round in 0..geometry.n / exchange {
        let offset = round * exchange;
        emit_round(
            body,
            &writes,
            round,
            geometry,
            |access, declare| {
                debug_assert!(!declare);
                format!(
                    "exchange[swizzle({})] = y{stage}_{};",
                    access.local_position(offset),
                    access.register
                )
            },
            |access| {
                format!(
                    "if ((d{stage}_{register} >> {shift}u) == {round}u) {{\n    exchange[swizzle(d{stage}_{register} & {mask}u)] = y{stage}_{register};\n  }}",
                    register = access.register,
                    shift = exchange.trailing_zeros(),
                )
            },
            false,
        );
        body.push_str("  workgroupBarrier();\n");
        emit_round(
            body,
            &reads,
            round,
            geometry,
            |access, declare| {
                format!(
                    "{}x{next}_{}{} = exchange[swizzle({})];",
                    if declare { "let " } else { "" },
                    access.register,
                    if declare { ": vec2<f32>" } else { "" },
                    access.local_position(offset)
                )
            },
            |access| {
                format!(
                    "if (({position} >> {shift}u) == {round}u) {{\n    x{next}_{register} = exchange[swizzle({position} & {mask}u)];\n  }}",
                    position = access.position,
                    register = access.register,
                    shift = exchange.trailing_zeros(),
                )
            },
            true,
        );
        body.push_str("  workgroupBarrier();\n");
    }
}

/// Emits one side of an exchange round: constant-round accesses directly,
/// each stepped group under one invocation test, and varying accesses under
/// their own position test. `access(a, declare)` renders an access known to
/// be in this round; with `declare_constant`, constant-round accesses bind
/// new `let`s instead of assigning `var`s.
fn emit_round(
    body: &mut String,
    accesses: &[ExchangeAccess],
    round: usize,
    geometry: ExchangeGeometry,
    access: impl Fn(&ExchangeAccess, bool) -> String,
    varying: impl Fn(&ExchangeAccess) -> String,
    declare_constant: bool,
) {
    for entry in accesses {
        if entry.pattern == RoundPattern::Constant(round) {
            body.push_str(&format!("  {}\n", access(entry, declare_constant)));
        }
    }
    let mut groups: Vec<((u32, usize), Vec<&ExchangeAccess>)> = Vec::new();
    for entry in accesses {
        if let RoundPattern::Stepped { shift, base } = entry.pattern {
            match groups.iter_mut().find(|(key, _)| *key == (shift, base)) {
                Some((_, members)) => members.push(entry),
                None => groups.push(((shift, base), vec![entry])),
            }
        }
    }
    for ((shift, base), members) in groups {
        let Some(step) = round.checked_sub(base) else {
            continue;
        };
        if step >= geometry.workgroup >> shift {
            continue;
        }
        body.push_str(&format!("  if ((t >> {shift}u) == {step}u) {{\n"));
        for entry in members {
            body.push_str(&format!("    {}\n", access(entry, false)));
        }
        body.push_str("  }\n");
    }
    for entry in accesses {
        if let RoundPattern::Varying(rounds) = &entry.pattern {
            if rounds.contains(&round) {
                body.push_str(&format!("  {}\n", varying(entry)));
            }
        }
    }
}

/// A position known to lie in the round starting at `offset`, relative to it.
fn local_position(position: &str, offset: usize) -> String {
    if offset == 0 {
        position.to_owned()
    } else {
        format!("({position} - {offset}u)")
    }
}

/// Emits a straight-line DFT of `inputs` (WGSL names) and returns the names
/// of its outputs in natural order.
fn emit_dft(
    body: &mut String,
    prefix: &str,
    inputs: &[String],
    direction: FftDirection,
    precision: AxisPrecision,
) -> Vec<String> {
    let radix = inputs.len();
    match radix {
        1 => inputs.to_vec(),
        2 => {
            let (a, b) = (&inputs[0], &inputs[1]);
            body.push_str(&format!(
                "  let {prefix}_0: vec2<f32> = {a} + {b};\n  let {prefix}_1: vec2<f32> = {a} - {b};\n"
            ));
            vec![format!("{prefix}_0"), format!("{prefix}_1")]
        }
        4 => {
            let (x0, x1, x2, x3) = (&inputs[0], &inputs[1], &inputs[2], &inputs[3]);
            let rotated = quarter_turn(&format!("{prefix}_d1"), 1, direction);
            body.push_str(&format!(
                "  let {prefix}_s0: vec2<f32> = {x0} + {x2};
  let {prefix}_d0: vec2<f32> = {x0} - {x2};
  let {prefix}_s1: vec2<f32> = {x1} + {x3};
  let {prefix}_d1: vec2<f32> = {x1} - {x3};
  let {prefix}_r1: vec2<f32> = {rotated};
  let {prefix}_0: vec2<f32> = {prefix}_s0 + {prefix}_s1;
  let {prefix}_1: vec2<f32> = {prefix}_d0 + {prefix}_r1;
  let {prefix}_2: vec2<f32> = {prefix}_s0 - {prefix}_s1;
  let {prefix}_3: vec2<f32> = {prefix}_d0 - {prefix}_r1;
"
            ));
            (0..4).map(|k| format!("{prefix}_{k}")).collect()
        }
        _ => {
            // DFT_R as R1 x R2: n = R2 * n1 + n2 and k = k1 + R1 * k2.
            let r1 = if radix.is_multiple_of(4) { 4 } else { 2 };
            let r2 = radix / r1;
            let columns = (0..r2)
                .map(|n2| {
                    let column = (0..r1)
                        .map(|n1| inputs[r2 * n1 + n2].clone())
                        .collect::<Vec<_>>();
                    emit_dft(
                        body,
                        &format!("{prefix}a{n2}"),
                        &column,
                        direction,
                        precision,
                    )
                    .into_iter()
                    .enumerate()
                    .map(|(k1, value)| {
                        constant_twiddle(
                            body,
                            &format!("{prefix}t{n2}_{k1}"),
                            &value,
                            n2 * k1,
                            radix,
                            direction,
                            precision,
                        )
                    })
                    .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            let mut outputs = vec![String::new(); radix];
            for k1 in 0..r1 {
                let row = columns
                    .iter()
                    .map(|column| column[k1].clone())
                    .collect::<Vec<_>>();
                let row_outputs =
                    emit_dft(body, &format!("{prefix}b{k1}"), &row, direction, precision);
                for (k2, value) in row_outputs.into_iter().enumerate() {
                    outputs[k1 + r1 * k2] = value;
                }
            }
            outputs
        }
    }
}

/// `value * W_radix^exponent` bound to `name`, or `value` itself for `W = 1`.
fn constant_twiddle(
    body: &mut String,
    name: &str,
    value: &str,
    exponent: usize,
    radix: usize,
    direction: FftDirection,
    precision: AxisPrecision,
) -> String {
    let exponent = exponent % radix;
    if exponent == 0 {
        return value.to_owned();
    }
    let product = if (4 * exponent).is_multiple_of(radix) {
        quarter_turn(value, 4 * exponent / radix, direction)
    } else {
        format!(
            "c_mul({value}, {})",
            radix_root_wgsl(radix, exponent, direction, precision)
        )
    };
    body.push_str(&format!("  let {name}: vec2<f32> = {product};\n"));
    name.to_owned()
}

/// `value * W_4^quarters`, where `W_4` is `-i` forward and `i` inverse.
fn quarter_turn(value: &str, quarters: usize, direction: FftDirection) -> String {
    let times_minus_i = format!("vec2<f32>({value}.y, -{value}.x)");
    let times_i = format!("vec2<f32>(-{value}.y, {value}.x)");
    match (quarters % 4, direction) {
        (0, _) => value.to_owned(),
        (2, _) => format!("-{value}"),
        (1, FftDirection::Forward) | (3, FftDirection::Inverse) => times_minus_i,
        _ => times_i,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn limits(invocations: u32, storage: u32) -> wgpu::Limits {
        wgpu::Limits {
            max_compute_invocations_per_workgroup: invocations,
            max_compute_workgroup_size_x: invocations,
            max_compute_workgroup_storage_size: storage,
            ..wgpu::Limits::default()
        }
    }

    #[test]
    fn schedules_keep_sixteen_values_per_invocation() {
        let vulkan = limits(1024, 48 * 1024);
        let (workgroup, schedule) = register_schedule(8192, AxisPrecision::F32, &vulkan).unwrap();
        assert_eq!(workgroup, 512);
        assert_eq!(schedule.radices, [16, 16, 16, 2]);
        assert_eq!(schedule.exchange_len, 4096);

        let (workgroup, schedule) = register_schedule(16384, AxisPrecision::F32, &vulkan).unwrap();
        assert_eq!(workgroup, 1024);
        assert_eq!(schedule.radices, [16, 16, 16, 4]);

        // A 256-invocation device with 16 KiB of workgroup memory.
        let webgpu = limits(256, 16 * 1024);
        let (workgroup, schedule) = register_schedule(4096, AxisPrecision::F32, &webgpu).unwrap();
        assert_eq!(workgroup, 256);
        assert_eq!(schedule.radices, [16, 16, 16]);
        assert_eq!(schedule.exchange_len, 2048);

        assert!(register_schedule(32768, AxisPrecision::F32, &vulkan).is_none());
        assert!(register_schedule(8192, AxisPrecision::F32, &webgpu).is_none());
        assert!(register_schedule(8192, AxisPrecision::F64, &vulkan).is_none());
        assert!(register_schedule(6144, AxisPrecision::F32, &vulkan).is_none());
    }

    /// Runs the schedule on the CPU exactly as the kernel orders it,
    /// including the exchange rounds, and checks it against a direct DFT.
    fn simulate(n: usize, workgroup: usize, radices: &[usize], exchange: usize) -> f64 {
        use crate::math::Complex64;
        let values = n / workgroup;
        let input = (0..n)
            .map(|i| Complex64::new((i as f64 * 0.37).sin(), (i as f64 * 0.11).cos()))
            .collect::<Vec<_>>();
        let root = |numerator: usize, denominator: usize| {
            let angle = -std::f64::consts::TAU * numerator as f64 / denominator as f64;
            Complex64::new(angle.cos(), angle.sin())
        };
        let mul = |a: Complex64, b: Complex64| {
            Complex64::new(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re)
        };
        let add = |a: Complex64, b: Complex64| Complex64::new(a.re + b.re, a.im + b.im);

        let first = radices[0];
        let mut registers = (0..workgroup)
            .map(|t| {
                (0..values)
                    .map(|i| input[t + (i / first) * workgroup + (i % first) * (n / first)])
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let mut previous = 1;
        for (stage, &radix) in radices.iter().enumerate() {
            let span = previous * radix;
            for (t, row) in registers.iter_mut().enumerate() {
                for m in 0..values / radix {
                    let j = (t + m * workgroup) % previous;
                    let inputs = (0..radix)
                        .map(|q| mul(row[m * radix + q], root(j * q, span)))
                        .collect::<Vec<_>>();
                    for k in 0..radix {
                        row[m * radix + k] = (0..radix).fold(Complex64::default(), |sum, q| {
                            add(sum, mul(inputs[q], root(k * q, radix)))
                        });
                    }
                }
            }
            if stage + 1 == radices.len() {
                break;
            }
            let geometry = ExchangeGeometry {
                n,
                workgroup,
                values,
                radix,
                previous,
                next_radix: radices[stage + 1],
                exchange,
            };
            let next_radix = radices[stage + 1];
            let mut next = vec![vec![Complex64::default(); values]; workgroup];
            let mut buffer = vec![None; exchange];
            for round in 0..n / exchange {
                buffer.fill(None);
                for (t, row) in registers.iter().enumerate() {
                    for m in 0..values / radix {
                        for k in 0..radix {
                            let position = geometry.write_position(t, m, k);
                            if position / exchange == round {
                                buffer[position % exchange] = Some(row[m * radix + k]);
                            }
                        }
                    }
                }
                for (t, row) in next.iter_mut().enumerate() {
                    for m in 0..values / next_radix {
                        for q in 0..next_radix {
                            let position = geometry.read_position(t, m, q);
                            if position / exchange == round {
                                row[m * next_radix + q] = buffer[position % exchange]
                                    .expect("every read position is written in its round");
                            }
                        }
                    }
                }
            }
            registers = next;
            previous = span;
        }

        let radix = radices[radices.len() - 1];
        let mut output = vec![Complex64::default(); n];
        for (t, row) in registers.iter().enumerate() {
            for m in 0..values / radix {
                for k in 0..radix {
                    output[t + m * workgroup + k * previous] = row[m * radix + k];
                }
            }
        }
        let mut max_error = 0.0f64;
        for (k, value) in output.iter().enumerate() {
            let expected = (0..n).fold(Complex64::default(), |sum, i| {
                add(sum, mul(input[i], root((i * k) % n, n)))
            });
            max_error = max_error.max((value.re - expected.re).hypot(value.im - expected.im));
        }
        max_error
    }

    #[test]
    fn schedule_with_exchange_rounds_matches_a_direct_dft() {
        assert!(simulate(256, 16, &[16, 16], 256) < 1e-9);
        assert!(simulate(512, 32, &[16, 16, 2], 256) < 1e-9);
        assert!(simulate(1024, 64, &[16, 16, 4], 256) < 1e-9);
        assert!(simulate(2048, 64, &[16, 16, 8], 512) < 1e-9);
    }

    fn wgsl_for(n: usize, invocations: u32, storage: u32, direction: FftDirection) -> String {
        let (workgroup, schedule) =
            register_schedule(n, AxisPrecision::F32, &limits(invocations, storage)).unwrap();
        generate_register_fft_wgsl(
            &FusedPow2StageWgslConfig {
                rank: 1,
                axis: 0,
                dims: &[n],
                axis_length: n,
                stride_complex: 1,
                direction,
                workgroup_size: workgroup,
                apply_scale: true,
                scale_factor: 0.5,
                precision: AxisPrecision::F32,
            },
            &schedule,
        )
    }

    #[test]
    fn kernel_keeps_the_line_in_registers_and_exchanges_in_rounds() {
        let wgsl = wgsl_for(8192, 1024, 48 * 1024, FftDirection::Forward);
        assert!(wgsl.contains("@compute @workgroup_size(512, 1, 1)"));
        assert!(wgsl.contains("var<workgroup> exchange: array<vec2<f32>, 4096>;"));
        // Sixteen loads and stores per invocation, all through registers.
        assert_eq!(wgsl.matches("= src[").count(), 16);
        assert_eq!(wgsl.matches("dst[").count(), 16);
        assert!(wgsl.contains("dst[base + t * STRIDE] = y3_0 * vec2<f32>(0.5, 0.5);"));
        // Three exchanges of two rounds, two barriers each.
        assert_eq!(wgsl.matches("workgroupBarrier();").count(), 12);
        assert!(!wgsl.contains("cos("));
        crate::runtime::assert_workgroup_var_written_before_read(&wgsl, "exchange");

        let inverse = wgsl_for(8192, 1024, 48 * 1024, FftDirection::Inverse);
        assert!(inverse.contains("return vec2<f32>(value.x, -value.y);"));
    }
}
