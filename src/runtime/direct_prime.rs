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
//! so one invocation produces an output pair from `h` terms, a quarter of the
//! multiply-adds of the plain sum. A workgroup loads several lines and the
//! `p` roots into workgroup memory; invocation `k` of a line produces
//! `X[k]` and `X[p - k]`, and invocation 0 produces `X[0]`. Work per line
//! still grows as `p^2`, so longer primes keep Rader.

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

/// Output pairs per line, counting `X[0]` as one.
const fn tasks_per_line(axis_length: usize) -> usize {
    axis_length.div_ceil(2)
}

/// Lines per workgroup of a direct kernel: enough output pairs to give each
/// invocation one, at least [`MIN_STRIDED_LINES`] on strided axes, and
/// within workgroup storage after the roots. A transform of few lines keeps
/// [`MIN_WORKGROUPS`] workgroups where it has the lines: each line costs
/// `O(N^2)`, so spreading lines over the GPU beats coalescing them.
pub(crate) fn direct_lines_per_workgroup(
    axis_length: usize,
    stride_complex: usize,
    precision: AxisPrecision,
    workgroup_size: u32,
    total_lines: usize,
    max_workgroup_storage_bytes: u64,
) -> u32 {
    let mut lines = (workgroup_size as usize / tasks_per_line(axis_length)).max(1);
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
            "let lineSlot: u32 = e / TASKS;\n    let k: u32 = e - lineSlot * TASKS;",
        )
    } else {
        (
            "let lineSlot: u32 = e % LINES;\n    let p: u32 = e / LINES;",
            "let lineSlot: u32 = e % LINES;\n    let k: u32 = e / LINES;",
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
const TASKS: u32 = HALF + 1u;
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
      var cosine: vec2<f32> = vec2<f32>(0.0, 0.0);
      var sine: vec2<f32> = vec2<f32>(0.0, 0.0);
      var index: u32 = 0u;
      for (var j: u32 = 1u; j <= HALF; j = j + 1u) {{
        index = index + k;
        if (index >= N) {{
          index = index - N;
        }}
        let low: vec2<f32> = values[row + j];
        let high: vec2<f32> = values[row + N - j];
        let angle: vec2<f32> = trig[index];
        cosine = cosine + (low + high) * angle.x;
        sine = sine + (low - high) * angle.y;
      }}
      let base: u32 = line_base(lineStart + lineSlot);
      let even: vec2<f32> = first + cosine;
      if (k == 0u) {{
        output[base] = {scaled_zero};
      }} else {{
        let rotation: vec2<f32> = {rotation};
        output[base + k * STRIDE] = {scaled_low};
        output[base + (N - k) * STRIDE] = {scaled_high};
      }}
    }}
  }}
}}
"#,
        half = (n - 1) / 2,
        stride = key.stride_complex,
        workgroup_size = key.workgroup_size,
        values_len = lines * n,
        scaled_zero = scale("even"),
        scaled_low = scale("even + rotation"),
        scaled_high = scale("even - rotation"),
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
            1 << 20,
            48 * 1024,
        ))
    }

    #[test]
    fn line_count_gives_each_invocation_an_output_pair() {
        // N=17 has 9 output pairs per line (X[0] and eight pairs).
        assert_eq!(
            direct_lines_per_workgroup(17, 1, AxisPrecision::F32, 256, 1 << 20, 48 * 1024),
            28
        );
        assert_eq!(
            direct_lines_per_workgroup(127, 1, AxisPrecision::F32, 256, 1 << 20, 48 * 1024),
            4
        );
        // Strided axes load at least eight neighbouring lines.
        assert_eq!(
            direct_lines_per_workgroup(127, 64, AxisPrecision::F32, 256, 1 << 20, 48 * 1024),
            8
        );
        // A transform of few lines keeps enough workgroups.
        assert_eq!(
            direct_lines_per_workgroup(83, 83, AxisPrecision::F32, 256, 83, 48 * 1024),
            1
        );
        assert_eq!(
            direct_lines_per_workgroup(17, 1, AxisPrecision::F32, 256, 17 * 17 * 17, 48 * 1024),
            19
        );
        // Storage keeps room for the roots.
        assert_eq!(
            direct_lines_per_workgroup(1021, 64, AxisPrecision::F32, 256, 1 << 20, 16 * 1024),
            1
        );
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
        assert!(wgsl.contains("const LINES: u32 = 28u;"));
        assert!(wgsl.contains("var<workgroup> values: array<vec2<f32>, 476>;"));
        assert!(wgsl.contains("let k: u32 = e - lineSlot * TASKS;"));
        assert!(wgsl.contains("let rotation: vec2<f32> = vec2<f32>(sine.y, -sine.x);"));
        assert!(wgsl.contains("output[base + (N - k) * STRIDE] = (even - rotation) * 0.5;"));
        crate::runtime::assert_workgroup_var_written_before_read(&wgsl, "values");
        crate::runtime::assert_workgroup_var_written_before_read(&wgsl, "trig");

        let strided =
            generate_direct_prime_wgsl_for_key(&key(31, &[4, 31], 1, FftDirection::Inverse));
        assert!(strided.contains("const STRIDE: u32 = 4u;"));
        assert!(strided.contains("let k: u32 = e / LINES;"));
        assert!(strided.contains("vec2<f32>(-sine.y, sine.x)"));
    }
}
