//! Direct-DFT kernels for short prime axes.
//!
//! For a short prime `p`, Rader's convolution costs more in stages, barriers,
//! and passes than it saves. These kernels compute the DFT directly, using the
//! symmetry of odd lengths: with `h = (p - 1) / 2`, `a_n = x[n] + x[p - n]`,
//! and `b_n = x[n] - x[p - n]`,
//!
//! ```text
//! X[k]     = x[0] + C_k - i S_k
//! X[p - k] = x[0] + C_k + i S_k      (forward; inverse swaps the signs)
//! C_k = sum_{n=1..h} a_n cos(2 pi n k / p),  S_k = sum_{n=1..h} b_n sin(2 pi n k / p)
//! ```
//!
//! so an output pair takes `h` terms, a quarter of the multiply-adds of the
//! plain sum. A workgroup loads several lines and the `p` roots into
//! workgroup memory; each invocation produces one or more consecutive output
//! pairs `X[k]` and `X[p - k]` of one line (`X[0]` counts as the pair
//! `k = 0`), sharing its loads of `a_n` and `b_n` between them (see
//! [`direct_pairs_per_invocation`]).
//! Work per line still grows as `p^2`, so longer primes keep Rader.

use crate::config::FftDirection;
use crate::runtime::axis_plan::{wgsl_line_base_fn, AxisPrecision};
use crate::runtime::pipeline_cache::{FusedPrimeKind, FusedPrimeStageKey};

/// Least lines per workgroup on strided axes, so loads coalesce.
const MIN_STRIDED_LINES: usize = 8;
/// Workgroups a small transform keeps, so its lines spread across the GPU.
const MIN_WORKGROUPS: usize = 256;

/// Whether a direct kernel can transform a line of the prime `axis_length`
/// with `workgroup_size` invocations on a device with `limits`.
pub(crate) fn direct_prime_supported(
    axis_length: usize,
    precision: AxisPrecision,
    workgroup_size: u32,
    limits: &wgpu::Limits,
) -> bool {
    let complex_bytes = precision.complex_size_bytes() as usize;
    precision == AxisPrecision::F32
        && axis_length > 2
        && axis_length % 2 == 1
        && workgroup_size > 0
        && workgroup_size <= limits.max_compute_invocations_per_workgroup
        && workgroup_size <= limits.max_compute_workgroup_size_x
        // One line plus the roots.
        && 2 * axis_length * complex_bytes <= limits.max_compute_workgroup_storage_size as usize
}

/// Invocations a transform keeps before its invocations take several output
/// pairs each.
const MIN_PAIRED_TASKS: usize = 100_000;

/// Output pairs per line, counting `X[0]` as one.
const fn pairs_per_line(axis_length: usize) -> usize {
    axis_length.div_ceil(2)
}

/// Invocations per line.
const fn tasks_per_line(axis_length: usize, pairs_per_invocation: usize) -> usize {
    pairs_per_line(axis_length).div_ceil(pairs_per_invocation)
}

/// Output pairs each invocation of a direct kernel produces. Several pairs
/// read `a_n` and `b_n` once for all of them, cutting the workgroup-memory
/// loads per term, which bound long primes, from three to one and a half
/// with four pairs; but they leave fewer invocations to fill the GPU and
/// hide latency. Four pairs paid off from `N = 97` and two
/// from `N = 61`, each while the transform kept about 100,000 invocations.
pub(crate) fn direct_pairs_per_invocation(axis_length: usize, total_lines: usize) -> usize {
    [(4, 97), (2, 61)]
        .into_iter()
        .find(|&(pairs, min_length)| {
            axis_length >= min_length
                && total_lines.saturating_mul(tasks_per_line(axis_length, pairs))
                    >= MIN_PAIRED_TASKS
        })
        .map_or(1, |(pairs, _)| pairs)
}

/// Lines per workgroup of a direct kernel: enough to give each invocation
/// its output pairs, at least [`MIN_STRIDED_LINES`] on strided axes, and
/// within workgroup storage after the roots. A transform of few lines keeps
/// [`MIN_WORKGROUPS`] workgroups where it has the lines: each line costs
/// `O(N^2)`, so spreading lines over the GPU beats coalescing them.
pub(crate) fn direct_lines_per_workgroup(
    axis_length: usize,
    stride_complex: usize,
    precision: AxisPrecision,
    workgroup_size: u32,
    pairs_per_invocation: usize,
    total_lines: usize,
    max_workgroup_storage_bytes: u64,
) -> u32 {
    let mut lines =
        (workgroup_size as usize / tasks_per_line(axis_length, pairs_per_invocation)).max(1);
    if stride_complex > 1 {
        lines = lines.max(MIN_STRIDED_LINES);
    }
    let complex_bytes = precision.complex_size_bytes() as usize;
    let storage_elements = max_workgroup_storage_bytes as usize / complex_bytes;
    let max_by_storage = (storage_elements.saturating_sub(axis_length) / axis_length).max(1);
    let max_by_fill = (total_lines / MIN_WORKGROUPS).max(1);
    lines.min(max_by_storage).min(max_by_fill) as u32
}

/// WGSL of a direct-DFT kernel for `key` (`key.kind` is
/// [`FusedPrimeKind::Direct`]).
pub(crate) fn generate_direct_prime_wgsl_for_key(key: &FusedPrimeStageKey) -> String {
    debug_assert_eq!(key.kind, FusedPrimeKind::Direct);
    debug_assert_eq!(key.precision, AxisPrecision::F32);
    debug_assert_eq!(key.axis_length % 2, 1);
    let n = key.axis_length;
    let lines = key.lines_per_workgroup as usize;
    // Contiguous lines map consecutive invocations along a line; strided
    // lines map them across neighbouring lines so accesses coalesce.
    let (load_split, task_split) = if key.stride_complex == 1 {
        (
            "let lineSlot: u32 = e / N;\n    let p: u32 = e - lineSlot * N;",
            "let lineSlot: u32 = e / TASKS;\n    let task: u32 = e - lineSlot * TASKS;",
        )
    } else {
        (
            "let lineSlot: u32 = e % LINES;\n    let p: u32 = e / LINES;",
            "let lineSlot: u32 = e % LINES;\n    let task: u32 = e / LINES;",
        )
    };
    // `rotation` is the sine term's contribution to `X[k]`: `-i S` forward,
    // `+i S` inverse; `X[p - k]` takes the opposite.
    let rotation = match key.direction {
        FftDirection::Forward => "vec2<f32>(sine.y, -sine.x)",
        FftDirection::Inverse => "vec2<f32>(-sine.y, sine.x)",
    };
    let scale = |value: &str| {
        if key.apply_scale {
            format!(
                "({value}) * {}",
                key.precision.format_wgsl_scalar(key.scale_factor())
            )
        } else {
            value.to_owned()
        }
    };
    // Pair `i` of a task is `k = task * pairs + i`; its root index
    // `j * k mod N` advances by `k` per term.
    let pairs = key.pairs_per_invocation as usize;
    let mut accumulators = String::new();
    let mut terms = String::new();
    let mut stores = String::new();
    for i in 0..pairs {
        accumulators.push_str(&format!(
            "      let k{i}: u32 = task * {pairs}u + {i}u;\n      var cosine{i}: vec2<f32> = vec2<f32>(0.0, 0.0);\n      var sine{i}: vec2<f32> = vec2<f32>(0.0, 0.0);\n      var index{i}: u32 = 0u;\n"
        ));
        terms.push_str(&format!(
            "        index{i} = index{i} + k{i};\n        if (index{i} >= N) {{\n          index{i} = index{i} - N;\n        }}\n        let angle{i}: vec2<f32> = trig[index{i}];\n        cosine{i} = cosine{i} + pairSum * angle{i}.x;\n        sine{i} = sine{i} + pairDifference * angle{i}.y;\n"
        ));
        let sine = format!("sine{i}");
        let rotation = rotation.replace("sine", &sine);
        stores.push_str(&format!(
            "      {{\n        let even: vec2<f32> = first + cosine{i};\n        if (k{i} == 0u) {{\n          output[base] = {};\n        }} else if (k{i} <= HALF) {{\n          let rotation: vec2<f32> = {rotation};\n          output[base + k{i} * STRIDE] = {};\n          output[base + (N - k{i}) * STRIDE] = {};\n        }}\n      }}\n",
            scale("even"),
            scale("even + rotation"),
            scale("even - rotation"),
        ));
    }
    format!(
        r#"struct Params {{
  lines: u32,
  lineOffset: u32,
  pad0: u32,
  pad1: u32,
}};

@group(0) @binding(0) var<storage, read> input: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec2<f32>>;
@group(0) @binding(4) var<storage, read> axisTwiddles: array<vec2<f32>>;
@group(0) @binding(5) var<uniform> params: Params;

const N: u32 = {n}u;
const HALF: u32 = {half}u;
const TASKS: u32 = {tasks}u;
const STRIDE: u32 = {stride}u;
const LINES: u32 = {lines}u;
const WORKGROUP_SIZE: u32 = {workgroup_size}u;

var<workgroup> values: array<vec2<f32>, {values_len}>;
// (cos, sin) of 2 pi m / N.
var<workgroup> trig: array<vec2<f32>, {n}>;

{line_base_fn}

@compute @workgroup_size({workgroup_size}, 1, 1)
fn main({entry_params}) {{
  {flat_workgroup_index}
  let groupLine: u32 = wgFlat * LINES;
  if (groupLine >= params.lines) {{
    return;
  }}
  let lineCount: u32 = min(LINES, params.lines - groupLine);
  let lineStart: u32 = params.lineOffset + groupLine;

  for (var e: u32 = lid.x; e < LINES * N; e = e + WORKGROUP_SIZE) {{
    {load_split}
    if (lineSlot < lineCount) {{
      values[lineSlot * N + p] = input[line_base(lineStart + lineSlot) + p * STRIDE];
    }}
  }}
  for (var i: u32 = lid.x; i < N; i = i + WORKGROUP_SIZE) {{
    // The table holds W^i = (cos, -sin).
    let root: vec2<f32> = axisTwiddles[i];
    trig[i] = vec2<f32>(root.x, -root.y);
  }}
  workgroupBarrier();

  for (var e: u32 = lid.x; e < LINES * TASKS; e = e + WORKGROUP_SIZE) {{
    {task_split}
    if (lineSlot < lineCount) {{
      let row: u32 = lineSlot * N;
      let first: vec2<f32> = values[row];
{accumulators}      for (var j: u32 = 1u; j <= HALF; j = j + 1u) {{
        let low: vec2<f32> = values[row + j];
        let high: vec2<f32> = values[row + N - j];
        let pairSum: vec2<f32> = low + high;
        let pairDifference: vec2<f32> = low - high;
{terms}      }}
      let base: u32 = line_base(lineStart + lineSlot);
{stores}    }}
  }}
}}
"#,
        half = (n - 1) / 2,
        tasks = tasks_per_line(n, pairs),
        accumulators = accumulators,
        terms = terms,
        stores = stores,
        stride = key.stride_complex,
        workgroup_size = key.workgroup_size,
        values_len = lines * n,
        line_base_fn = wgsl_line_base_fn(key.rank, key.axis, &key.dims),
        entry_params = crate::runtime::dispatch::WGSL_FLAT_ENTRY_PARAMS,
        flat_workgroup_index = crate::runtime::dispatch::WGSL_FLAT_WORKGROUP_INDEX,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(n: usize, dims: &[usize], axis: usize, direction: FftDirection) -> FusedPrimeStageKey {
        let stride = dims[..axis].iter().product::<usize>();
        FusedPrimeStageKey::new(
            FusedPrimeKind::Direct,
            dims.len(),
            axis,
            dims,
            n,
            stride,
            n,
            &[n],
            direction,
            256,
            true,
            0.5,
            AxisPrecision::F32,
        )
        .with_lines_per_workgroup(direct_lines_per_workgroup(
            n,
            stride,
            AxisPrecision::F32,
            256,
            4,
            1 << 20,
            48 * 1024,
        ))
        .with_pairs_per_invocation(4)
    }

    #[test]
    fn line_count_gives_each_invocation_its_output_pairs() {
        // N=17 has 9 output pairs per line (X[0] and eight pairs): three
        // invocations of up to four pairs.
        assert_eq!(
            direct_lines_per_workgroup(17, 1, AxisPrecision::F32, 256, 4, 1 << 20, 48 * 1024),
            85
        );
        assert_eq!(
            direct_lines_per_workgroup(127, 1, AxisPrecision::F32, 256, 4, 1 << 20, 48 * 1024),
            16
        );
        // Strided axes load at least eight neighbouring lines.
        assert_eq!(
            direct_lines_per_workgroup(509, 64, AxisPrecision::F32, 256, 4, 1 << 20, 48 * 1024),
            8
        );
        // One pair per invocation: N=17 has 9 per line.
        assert_eq!(
            direct_lines_per_workgroup(17, 1, AxisPrecision::F32, 256, 1, 1 << 20, 48 * 1024),
            28
        );
        // A transform of few lines keeps enough workgroups.
        assert_eq!(
            direct_lines_per_workgroup(83, 83, AxisPrecision::F32, 256, 1, 83, 48 * 1024),
            1
        );
        assert_eq!(
            direct_lines_per_workgroup(17, 1, AxisPrecision::F32, 256, 1, 17 * 17 * 17, 48 * 1024),
            19
        );
        // Storage keeps room for the roots.
        assert_eq!(
            direct_lines_per_workgroup(1021, 64, AxisPrecision::F32, 256, 1, 1 << 20, 16 * 1024),
            1
        );
    }

    #[test]
    fn long_primes_with_enough_lines_take_several_pairs_per_invocation() {
        assert_eq!(direct_pairs_per_invocation(97, 97 * 97), 4);
        assert_eq!(direct_pairs_per_invocation(89, 89 * 89), 2);
        assert_eq!(direct_pairs_per_invocation(61, 131_072), 2);
        // Too few lines, or too short a prime.
        assert_eq!(direct_pairs_per_invocation(97, 97), 1);
        assert_eq!(direct_pairs_per_invocation(61, 61 * 61), 1);
        assert_eq!(direct_pairs_per_invocation(31, 1 << 20), 1);
    }

    /// Evaluates the kernel's symmetric sums on the CPU for one line.
    fn symmetric_dft(x: &[(f64, f64)], inverse: bool) -> Vec<(f64, f64)> {
        let n = x.len();
        let half = (n - 1) / 2;
        let mut out = vec![(0.0, 0.0); n];
        for k in 0..=half {
            let (mut cos_re, mut cos_im, mut sin_re, mut sin_im) = (0.0, 0.0, 0.0, 0.0);
            for j in 1..=half {
                let angle = std::f64::consts::TAU * ((j * k) % n) as f64 / n as f64;
                let (low, high) = (x[j], x[n - j]);
                cos_re += (low.0 + high.0) * angle.cos();
                cos_im += (low.1 + high.1) * angle.cos();
                sin_re += (low.0 - high.0) * angle.sin();
                sin_im += (low.1 - high.1) * angle.sin();
            }
            let even = (x[0].0 + cos_re, x[0].1 + cos_im);
            let rotation = if inverse {
                (-sin_im, sin_re)
            } else {
                (sin_im, -sin_re)
            };
            if k == 0 {
                out[0] = even;
            } else {
                out[k] = (even.0 + rotation.0, even.1 + rotation.1);
                out[n - k] = (even.0 - rotation.0, even.1 - rotation.1);
            }
        }
        out
    }

    #[test]
    fn symmetric_sums_match_the_plain_dft() {
        for n in [3usize, 17, 31] {
            for inverse in [false, true] {
                let x = (0..n)
                    .map(|i| ((i as f64 * 0.7).sin(), (i as f64 * 0.3).cos()))
                    .collect::<Vec<_>>();
                let sign = if inverse { 1.0 } else { -1.0 };
                for (k, value) in symmetric_dft(&x, inverse).into_iter().enumerate() {
                    let (mut re, mut im) = (0.0, 0.0);
                    for (j, sample) in x.iter().enumerate() {
                        let angle = sign * std::f64::consts::TAU * ((j * k) % n) as f64 / n as f64;
                        re += sample.0 * angle.cos() - sample.1 * angle.sin();
                        im += sample.0 * angle.sin() + sample.1 * angle.cos();
                    }
                    assert!((value.0 - re).abs() < 1e-9 && (value.1 - im).abs() < 1e-9);
                }
            }
        }
    }

    #[test]
    fn kernel_pairs_outputs_from_workgroup_memory() {
        let wgsl = generate_direct_prime_wgsl_for_key(&key(17, &[17], 0, FftDirection::Forward));
        assert!(wgsl.contains("const HALF: u32 = 8u;"));
        assert!(wgsl.contains("const LINES: u32 = 85u;"));
        assert!(wgsl.contains("const TASKS: u32 = 3u;"));
        assert!(wgsl.contains("var<workgroup> values: array<vec2<f32>, 1445>;"));
        assert!(wgsl.contains("let task: u32 = e - lineSlot * TASKS;"));
        // Each invocation's four pairs share the loads of a term's samples.
        assert_eq!(wgsl.matches("values[row + j]").count(), 1);
        assert!(wgsl.contains("let k3: u32 = task * 4u + 3u;"));
        assert!(wgsl.contains("let rotation: vec2<f32> = vec2<f32>(sine3.y, -sine3.x);"));
        assert!(wgsl.contains("output[base + (N - k0) * STRIDE] = (even - rotation) * 0.5;"));
        crate::runtime::assert_workgroup_var_written_before_read(&wgsl, "values");
        crate::runtime::assert_workgroup_var_written_before_read(&wgsl, "trig");

        let strided =
            generate_direct_prime_wgsl_for_key(&key(31, &[4, 31], 1, FftDirection::Inverse));
        assert!(strided.contains("const STRIDE: u32 = 4u;"));
        assert!(strided.contains("let task: u32 = e / LINES;"));
        assert!(strided.contains("vec2<f32>(-sine0.y, sine0.x)"));
    }
}
