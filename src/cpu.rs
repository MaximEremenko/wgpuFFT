//! Host-memory FFT plans for machines without a usable GPU.
//!
//! [`CpuFftPlan`] mirrors [`FftPlan`](crate::FftPlan) on the CPU. It accepts the
//! same [`FftConfig`] (shape, selected axes, batch, direction, normalization,
//! precision) and uses the same buffer layouts: interleaved complex scalars,
//! `[re_hi, re_lo, im_hi, im_lo]` for `Df64`, and the packed
//! `[floor(N0 / 2) + 1, ...]` spectrum for real transforms. Axis 0 is the
//! contiguous axis and batches are contiguous blocks.
//!
//! Transforms run on `rustfft` and `realfft`, which use AVX, SSE, or NEON when
//! the CPU supports them, and large transforms are split across the available
//! CPU threads. `Df64` data is transformed in native `f64` and returned as
//! `hi + lo` pairs. Real transforms support `F32` and `F64`.
//!
//! ```
//! use wgpu_fft::{cpu::CpuFftPlan, FftConfig, Normalization};
//!
//! let plan = CpuFftPlan::c2c(FftConfig::new(4).with_normalization(Normalization::None))?;
//! let input = [1.0f32, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0];
//! let mut output = [0.0f32; 8];
//! plan.execute(&input, &mut output)?;
//!
//! let expected = [10.0f32, 0.0, -2.0, 2.0, -2.0, 0.0, -2.0, -2.0];
//! assert!(output.iter().zip(expected).all(|(a, b)| (a - b).abs() < 1e-5));
//! # Ok::<(), wgpu_fft::FftError>(())
//! ```

use std::sync::Arc;

use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};
use rustfft::num_complex::Complex;
use rustfft::num_traits::Zero;
use rustfft::{Fft, FftNum, FftPlanner};

use crate::config::{FftConfig, FftDirection, FftPrecision};
use crate::error::{FftError, Result};
use crate::plan::FftTransformKind;

/// Work below this many elements per extra thread stays on the calling thread.
const MIN_ELEMENTS_PER_WORKER: usize = 1 << 15;

/// Host-memory FFT plan with the same configuration semantics and buffer
/// layouts as [`FftPlan`](crate::FftPlan).
pub struct CpuFftPlan {
    kind: FftTransformKind,
    config: FftConfig,
    packed_shape: Option<Vec<usize>>,
    total_elements: usize,
    packed_elements: usize,
    kernels: Kernels,
}

enum Kernels {
    F32(TypedKernels<f32>),
    /// Used by both `F64` and `Df64` plans.
    F64(TypedKernels<f64>),
}

struct TypedKernels<T: FftNum> {
    /// Complex transforms in execution order, each with the axis it runs on.
    axes: Vec<(usize, Arc<dyn Fft<T>>)>,
    real: RealKernel<T>,
    scale: T,
}

enum RealKernel<T: FftNum> {
    None,
    Forward(Arc<dyn RealToComplex<T>>),
    Inverse(Arc<dyn ComplexToReal<T>>),
}

impl CpuFftPlan {
    /// Creates a complex-to-complex plan. Every [`FftPrecision`] is supported.
    pub fn c2c(config: FftConfig) -> Result<Self> {
        config.validate()?;
        let direction = config.direction();
        let kernels = match config.precision() {
            FftPrecision::F32 => Kernels::F32(TypedKernels::new(
                config.axes(),
                config.shape(),
                direction,
                RealKernel::None,
                config.scale()?,
            )),
            FftPrecision::F64 | FftPrecision::Df64 => Kernels::F64(TypedKernels::new(
                config.axes(),
                config.shape(),
                direction,
                RealKernel::None,
                config.scale_f64()?,
            )),
        };
        let total_elements = config.total_complex_len()?;
        Ok(Self {
            kind: FftTransformKind::C2c,
            config,
            packed_shape: None,
            total_elements,
            packed_elements: total_elements,
            kernels,
        })
    }

    /// Creates a forward real-to-complex plan over every axis. `F32` and `F64`
    /// are supported.
    pub fn r2c(config: FftConfig) -> Result<Self> {
        Self::real(config, FftTransformKind::R2c)
    }

    /// Creates an inverse complex-to-real plan over every axis. `F32` and `F64`
    /// are supported.
    pub fn c2r(config: FftConfig) -> Result<Self> {
        Self::real(config, FftTransformKind::C2r)
    }

    fn real(config: FftConfig, kind: FftTransformKind) -> Result<Self> {
        config.validate()?;
        let (transform, expected) = match kind {
            FftTransformKind::R2c => ("r2c", FftDirection::Forward),
            _ => ("c2r", FftDirection::Inverse),
        };
        if config.direction() != expected {
            return Err(FftError::InvalidRealTransformDirection {
                transform,
                expected: direction_name(expected),
                actual: direction_name(config.direction()),
            });
        }
        let full_axes = (0..config.shape().len()).collect::<Vec<_>>();
        if config.axes() != full_axes.as_slice() {
            return Err(FftError::UnsupportedRealAxes {
                expected: full_axes,
                actual: config.axes().to_vec(),
            });
        }

        let mut packed_shape = config.shape().to_vec();
        packed_shape[0] = packed_shape[0] / 2 + 1;
        // Axis 0 is the real axis; the others are complex transforms of the
        // packed spectrum.
        let complex_axes = &full_axes[1..];
        let axis_len = config.shape()[0];
        let kernels = match config.precision() {
            FftPrecision::F32 => {
                let mut planner = RealFftPlanner::<f32>::new();
                let real = match kind {
                    FftTransformKind::R2c => {
                        RealKernel::Forward(planner.plan_fft_forward(axis_len))
                    }
                    _ => RealKernel::Inverse(planner.plan_fft_inverse(axis_len)),
                };
                Kernels::F32(TypedKernels::new(
                    complex_axes,
                    &packed_shape,
                    expected,
                    real,
                    config.scale()?,
                ))
            }
            FftPrecision::F64 => {
                let mut planner = RealFftPlanner::<f64>::new();
                let real = match kind {
                    FftTransformKind::R2c => {
                        RealKernel::Forward(planner.plan_fft_forward(axis_len))
                    }
                    _ => RealKernel::Inverse(planner.plan_fft_inverse(axis_len)),
                };
                Kernels::F64(TypedKernels::new(
                    complex_axes,
                    &packed_shape,
                    expected,
                    real,
                    config.scale_f64()?,
                ))
            }
            FftPrecision::Df64 => {
                return Err(FftError::PrecisionUnsupported {
                    requested: FftPrecision::Df64,
                    route: transform,
                    reason: "cpu-real-transforms-support-f32-and-f64",
                })
            }
        };
        let total_elements = config.total_complex_len()?;
        let packed_elements = packed_shape.iter().product::<usize>() * config.batch();
        Ok(Self {
            kind,
            config,
            packed_shape: Some(packed_shape),
            total_elements,
            packed_elements,
            kernels,
        })
    }

    /// Returns the transform kind.
    pub fn kind(&self) -> FftTransformKind {
        self.kind
    }

    /// Returns the configuration the plan was created with.
    pub fn config(&self) -> &FftConfig {
        &self.config
    }

    /// Returns the packed complex shape of a real transform.
    pub fn packed_shape(&self) -> Option<&[usize]> {
        self.packed_shape.as_deref()
    }

    /// Number of scalar words the input slice must hold.
    pub fn required_input_len(&self) -> usize {
        match self.kind {
            FftTransformKind::C2c => self.total_elements * self.words_per_complex(),
            FftTransformKind::R2c => self.total_elements,
            FftTransformKind::C2r => self.packed_elements * 2,
        }
    }

    /// Number of scalar words the output slice must hold.
    pub fn required_output_len(&self) -> usize {
        match self.kind {
            FftTransformKind::C2c => self.total_elements * self.words_per_complex(),
            FftTransformKind::R2c => self.packed_elements * 2,
            FftTransformKind::C2r => self.total_elements,
        }
    }

    fn words_per_complex(&self) -> usize {
        match self.config.precision() {
            FftPrecision::Df64 => 4,
            FftPrecision::F32 | FftPrecision::F64 => 2,
        }
    }

    /// Executes an `F32` or `Df64` plan on `f32` words.
    ///
    /// `F64` plans must use [`Self::execute_f64`].
    pub fn execute(&self, input: &[f32], output: &mut [f32]) -> Result<()> {
        match (&self.kernels, self.config.precision()) {
            (Kernels::F32(kernels), _) => {
                self.check_lengths(input.len(), output.len())?;
                kernels.run(self.kind, self.config.shape(), self.packed(), input, output);
                Ok(())
            }
            (Kernels::F64(kernels), FftPrecision::Df64) => {
                self.check_lengths(input.len(), output.len())?;
                kernels.run_df64_c2c(self.config.shape(), input, output);
                Ok(())
            }
            (Kernels::F64(_), requested) => Err(FftError::PrecisionUnsupported {
                requested,
                route: "cpu",
                reason: "f64-plans-execute-with-execute_f64",
            }),
        }
    }

    /// Executes an `F64` plan on `f64` words.
    pub fn execute_f64(&self, input: &[f64], output: &mut [f64]) -> Result<()> {
        match (&self.kernels, self.config.precision()) {
            (Kernels::F64(kernels), FftPrecision::F64) => {
                self.check_lengths(input.len(), output.len())?;
                kernels.run(self.kind, self.config.shape(), self.packed(), input, output);
                Ok(())
            }
            (_, requested) => Err(FftError::PrecisionUnsupported {
                requested,
                route: "cpu",
                reason: "execute_f64-requires-an-f64-plan",
            }),
        }
    }

    fn packed(&self) -> &[usize] {
        self.packed_shape.as_deref().unwrap_or(&[])
    }

    fn check_lengths(&self, input: usize, output: usize) -> Result<()> {
        for (buffer, expected, actual) in [
            ("input", self.required_input_len(), input),
            ("output", self.required_output_len(), output),
        ] {
            if actual != expected {
                return Err(FftError::HostBufferLength {
                    buffer,
                    expected,
                    actual,
                });
            }
        }
        Ok(())
    }
}

impl<T: FftNum + bytemuck::Pod> TypedKernels<T> {
    fn new(
        axes: &[usize],
        shape: &[usize],
        direction: FftDirection,
        real: RealKernel<T>,
        scale: T,
    ) -> Self {
        let direction = match direction {
            FftDirection::Forward => rustfft::FftDirection::Forward,
            FftDirection::Inverse => rustfft::FftDirection::Inverse,
        };
        let mut planner = FftPlanner::<T>::new();
        let axes = axes
            .iter()
            .map(|&axis| (axis, planner.plan_fft(shape[axis], direction)))
            .collect();
        Self { axes, real, scale }
    }

    /// Runs a plan whose host words have the kernel's scalar type.
    fn run(
        &self,
        kind: FftTransformKind,
        shape: &[usize],
        packed: &[usize],
        input: &[T],
        output: &mut [T],
    ) {
        match kind {
            FftTransformKind::C2c => {
                output.copy_from_slice(input);
                let data: &mut [Complex<T>] = bytemuck::cast_slice_mut(output);
                self.transform_axes(data, shape);
                scale_all(data, self.scale);
            }
            FftTransformKind::R2c => {
                let data: &mut [Complex<T>] = bytemuck::cast_slice_mut(output);
                self.real_forward(input, data, shape[0], packed[0]);
                self.transform_axes(data, packed);
                scale_all(data, self.scale);
            }
            FftTransformKind::C2r => {
                let mut spectrum = bytemuck::cast_slice::<T, Complex<T>>(input).to_vec();
                self.transform_axes(&mut spectrum, packed);
                self.real_inverse(&mut spectrum, output, shape[0], packed[0]);
                scale_all(output, self.scale);
            }
        }
    }

    fn transform_axes(&self, data: &mut [Complex<T>], shape: &[usize]) {
        let mut work = Vec::new();
        for (axis, fft) in &self.axes {
            transform_axis(data, shape, *axis, fft.as_ref(), &mut work);
        }
    }

    fn real_forward(&self, input: &[T], output: &mut [Complex<T>], len: usize, packed_len: usize) {
        let RealKernel::Forward(fft) = &self.real else {
            unreachable!("r2c plans own a forward real kernel");
        };
        for_each_chunk_group(output, packed_len, |first_line, spectra| {
            let mut line = vec![T::zero(); len];
            let mut scratch = vec![Complex::zero(); fft.get_scratch_len()];
            for (index, spectrum) in spectra.chunks_exact_mut(packed_len).enumerate() {
                let start = (first_line + index) * len;
                line.copy_from_slice(&input[start..start + len]);
                fft.process_with_scratch(&mut line, spectrum, &mut scratch)
                    .expect("real line lengths match the plan");
            }
        });
    }

    fn real_inverse(
        &self,
        spectrum: &mut [Complex<T>],
        output: &mut [T],
        len: usize,
        packed_len: usize,
    ) {
        let RealKernel::Inverse(fft) = &self.real else {
            unreachable!("c2r plans own an inverse real kernel");
        };
        for_each_chunk_group_zip(spectrum, packed_len, output, len, |spectra, lines| {
            let mut scratch = vec![Complex::zero(); fft.get_scratch_len()];
            for (bins, line) in spectra
                .chunks_exact_mut(packed_len)
                .zip(lines.chunks_exact_mut(len))
            {
                // The DC and Nyquist bins of a real signal are real; like the
                // GPU route, ignore any imaginary part the caller left there.
                bins[0].im = T::zero();
                if len.is_multiple_of(2) {
                    bins[packed_len - 1].im = T::zero();
                }
                fft.process_with_scratch(bins, line, &mut scratch)
                    .expect("real line lengths match the plan");
            }
        });
    }
}

impl TypedKernels<f64> {
    /// Runs a `Df64` C2C plan: `[re_hi, re_lo, im_hi, im_lo]` words are
    /// transformed in `f64` and split back into `hi + lo` pairs.
    fn run_df64_c2c(&self, shape: &[usize], input: &[f32], output: &mut [f32]) {
        let mut data = input
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&[re_hi, re_lo, im_hi, im_lo]| {
                Complex::new(
                    f64::from(re_hi) + f64::from(re_lo),
                    f64::from(im_hi) + f64::from(im_lo),
                )
            })
            .collect::<Vec<_>>();
        self.transform_axes(&mut data, shape);
        scale_all(&mut data, self.scale);
        for (words, value) in output.as_chunks_mut::<4>().0.iter_mut().zip(&data) {
            let (re_hi, re_lo) = split_df64(value.re);
            let (im_hi, im_lo) = split_df64(value.im);
            *words = [re_hi, re_lo, im_hi, im_lo];
        }
    }
}

fn split_df64(value: f64) -> (f32, f32) {
    let hi = value as f32;
    (hi, (value - f64::from(hi)) as f32)
}

fn direction_name(direction: FftDirection) -> &'static str {
    match direction {
        FftDirection::Forward => "forward",
        FftDirection::Inverse => "inverse",
    }
}

/// Transforms every line of `axis`, using `work` to make strided lines
/// contiguous.
fn transform_axis<T: FftNum>(
    data: &mut [Complex<T>],
    shape: &[usize],
    axis: usize,
    fft: &dyn Fft<T>,
    work: &mut Vec<Complex<T>>,
) {
    let len = shape[axis];
    if len == 1 {
        return;
    }
    // Elements between consecutive values of one line.
    let inner = shape[..axis].iter().product::<usize>();
    if inner == 1 {
        process_lines(data, len, fft);
        return;
    }

    // View the data as [outer][len][inner] and transpose each block to
    // [outer][inner][len] so every line is contiguous.
    let block = len * inner;
    work.resize(data.len(), Complex::zero());
    let source: &[Complex<T>] = data;
    for_each_chunk_group(work, len, |first_line, lines| {
        for (index, line) in lines.chunks_exact_mut(len).enumerate() {
            let line_index = first_line + index;
            let base = (line_index / inner) * block + line_index % inner;
            for (n, value) in line.iter_mut().enumerate() {
                *value = source[base + n * inner];
            }
        }
    });
    process_lines(work, len, fft);
    let lines: &[Complex<T>] = work;
    for_each_chunk_group(data, inner, |first_row, rows| {
        for (index, row) in rows.chunks_exact_mut(inner).enumerate() {
            let row_index = first_row + index;
            let (outer, n) = (row_index / len, row_index % len);
            for (j, value) in row.iter_mut().enumerate() {
                *value = lines[(outer * inner + j) * len + n];
            }
        }
    });
}

fn process_lines<T: FftNum>(data: &mut [Complex<T>], len: usize, fft: &dyn Fft<T>) {
    for_each_chunk_group(data, len, |_, lines| {
        let mut scratch = vec![Complex::zero(); fft.get_inplace_scratch_len()];
        fft.process_with_scratch(lines, &mut scratch);
    });
}

fn scale_all<V, T>(data: &mut [V], scale: T)
where
    V: Copy + Send + std::ops::Mul<T, Output = V>,
    T: FftNum,
{
    if scale != T::one() {
        for_each_chunk_group(data, 1, |_, values| {
            for value in values {
                *value = *value * scale;
            }
        });
    }
}

fn worker_count(elements: usize) -> usize {
    let available = std::thread::available_parallelism().map_or(1, usize::from);
    available.min(elements / MIN_ELEMENTS_PER_WORKER).max(1)
}

/// Runs `task` over groups of whole `chunk_len` chunks, passing the index of
/// each group's first chunk. Large inputs are split across scoped threads.
fn for_each_chunk_group<T: Send>(
    data: &mut [T],
    chunk_len: usize,
    task: impl Fn(usize, &mut [T]) + Sync,
) {
    let chunks = data.len() / chunk_len;
    let workers = worker_count(data.len()).min(chunks);
    if workers <= 1 {
        task(0, data);
        return;
    }
    let per_worker = chunks.div_ceil(workers);
    std::thread::scope(|scope| {
        for (worker, group) in data.chunks_mut(per_worker * chunk_len).enumerate() {
            let task = &task;
            scope.spawn(move || task(worker * per_worker, group));
        }
    });
}

/// Like [`for_each_chunk_group`], over two slices with matching chunk counts.
fn for_each_chunk_group_zip<A: Send, B: Send>(
    first: &mut [A],
    first_chunk: usize,
    second: &mut [B],
    second_chunk: usize,
    task: impl Fn(&mut [A], &mut [B]) + Sync,
) {
    let chunks = first.len() / first_chunk;
    let workers = worker_count(first.len() + second.len()).min(chunks);
    if workers <= 1 {
        task(first, second);
        return;
    }
    let per_worker = chunks.div_ceil(workers);
    std::thread::scope(|scope| {
        for (a, b) in first
            .chunks_mut(per_worker * first_chunk)
            .zip(second.chunks_mut(per_worker * second_chunk))
        {
            let task = &task;
            scope.spawn(move || task(a, b));
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Normalization;
    use crate::math::{
        from_interleaved_f32, reference_c2c_nd, reference_c2c_nd_f64,
        reference_c2r_from_packed_interleaved, reference_r2c_packed_interleaved,
        to_interleaved_f32, Complex64,
    };

    fn signal(len: usize, seed: f32) -> Vec<f32> {
        (0..len)
            .map(|i| ((i as f32 * 0.37 + seed).sin() + (i % 7) as f32 * 0.125) * 0.5)
            .collect()
    }

    fn assert_close_f32(actual: &[f32], expected: &[f32], label: &str) {
        assert_eq!(actual.len(), expected.len(), "{label}: length");
        let peak = expected.iter().fold(1.0f32, |m, v| m.max(v.abs()));
        let worst = actual
            .iter()
            .zip(expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            worst <= 2e-4 * peak,
            "{label}: max error {worst} (peak {peak})"
        );
    }

    fn assert_close_f64(actual: &[f64], expected: &[f64], tolerance: f64, label: &str) {
        assert_eq!(actual.len(), expected.len(), "{label}: length");
        let peak = expected.iter().fold(1.0f64, |m, v| m.max(v.abs()));
        let worst = actual
            .iter()
            .zip(expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);
        assert!(
            worst <= tolerance * peak,
            "{label}: max error {worst} (peak {peak})"
        );
    }

    fn configs() -> Vec<FftConfig> {
        let mut configs = Vec::new();
        for len in [1, 2, 3, 4, 5, 7, 8, 12, 16, 17, 29, 64, 100, 243] {
            configs.push(FftConfig::new(len).with_normalization(Normalization::None));
        }
        configs.push(FftConfig::inverse(96));
        configs.push(FftConfig::new(60).with_normalization(Normalization::Orthogonal));
        configs.push(FftConfig::new(30).with_normalization(Normalization::Forward));
        configs.push(FftConfig::new_nd([4, 6]).with_batch(3));
        configs.push(FftConfig::new_nd([8, 3, 5]).with_axes([2, 0]).with_batch(2));
        configs.push(FftConfig::inverse_nd([6, 5, 4]).with_axes([1]));
        configs.push(FftConfig::new_nd([5, 1, 7]).with_axes([0, 2]));
        configs
    }

    #[test]
    fn c2c_f32_matches_reference() {
        for config in configs() {
            let input = signal(config.total_complex_len().unwrap() * 2, 0.3);
            let plan = CpuFftPlan::c2c(config.clone()).unwrap();
            let mut output = vec![0.0f32; plan.required_output_len()];
            plan.execute(&input, &mut output).unwrap();
            let expected = to_interleaved_f32(
                &reference_c2c_nd(&from_interleaved_f32(&input), &config).unwrap(),
            );
            assert_close_f32(&output, &expected, &format!("{config:?}"));
        }
    }

    #[test]
    fn c2c_f64_and_df64_match_the_f64_reference() {
        for config in configs() {
            let values = signal(config.total_complex_len().unwrap() * 2, 1.1)
                .into_iter()
                .map(f64::from)
                .collect::<Vec<_>>();
            let reference = reference_c2c_nd_f64(
                &values
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|&[re, im]| Complex64::new(re, im))
                    .collect::<Vec<_>>(),
                &config,
            )
            .unwrap()
            .into_iter()
            .flat_map(|value| [value.re, value.im])
            .collect::<Vec<_>>();

            let plan = CpuFftPlan::c2c(config.clone().with_precision(FftPrecision::F64)).unwrap();
            let mut output = vec![0.0f64; plan.required_output_len()];
            plan.execute_f64(&values, &mut output).unwrap();
            assert_close_f64(&output, &reference, 1e-12, &format!("f64 {config:?}"));

            let df64_input = values
                .iter()
                .flat_map(|&value| {
                    let (hi, lo) = split_df64(value);
                    [hi, lo]
                })
                .collect::<Vec<_>>();
            let plan = CpuFftPlan::c2c(config.clone().with_precision(FftPrecision::Df64)).unwrap();
            let mut output = vec![0.0f32; plan.required_output_len()];
            plan.execute(&df64_input, &mut output).unwrap();
            let joined = output
                .as_chunks::<2>()
                .0
                .iter()
                .map(|&[hi, lo]| f64::from(hi) + f64::from(lo))
                .collect::<Vec<_>>();
            assert_close_f64(&joined, &reference, 1e-12, &format!("df64 {config:?}"));
        }
    }

    fn real_configs() -> Vec<FftConfig> {
        vec![
            FftConfig::new(1),
            FftConfig::new(2),
            FftConfig::new(9),
            FftConfig::new(16).with_batch(3),
            FftConfig::new_nd([6, 5]),
            FftConfig::new_nd([7, 4]).with_batch(2),
            FftConfig::new_nd([8, 3, 2]),
        ]
    }

    #[test]
    fn r2c_matches_packed_reference_in_f32_and_f64() {
        for config in real_configs() {
            let real = signal(config.total_complex_len().unwrap(), 0.7);
            let expected = reference_r2c_packed_interleaved(&real, &config).unwrap();

            let plan = CpuFftPlan::r2c(config.clone()).unwrap();
            let mut packed = vec![0.0f32; plan.required_output_len()];
            plan.execute(&real, &mut packed).unwrap();
            assert_close_f32(&packed, &expected, &format!("r2c {config:?}"));

            let plan = CpuFftPlan::r2c(config.clone().with_precision(FftPrecision::F64)).unwrap();
            let real64 = real.iter().copied().map(f64::from).collect::<Vec<_>>();
            let mut packed64 = vec![0.0f64; plan.required_output_len()];
            plan.execute_f64(&real64, &mut packed64).unwrap();
            let expected64 = expected.iter().copied().map(f64::from).collect::<Vec<_>>();
            assert_close_f64(&packed64, &expected64, 2e-4, &format!("r2c f64 {config:?}"));
        }
    }

    #[test]
    fn c2r_matches_reference_and_round_trips() {
        for config in real_configs() {
            let real = signal(config.total_complex_len().unwrap(), 2.3);
            let forward = CpuFftPlan::r2c(config.clone()).unwrap();
            let mut packed = vec![0.0f32; forward.required_output_len()];
            forward.execute(&real, &mut packed).unwrap();
            // Perturb the self-conjugate bins' imaginary parts: C2R ignores them.
            packed[1] += 0.25;

            let inverse_config = config.clone().with_direction(FftDirection::Inverse);
            let inverse = CpuFftPlan::c2r(inverse_config.clone()).unwrap();
            let mut restored = vec![0.0f32; inverse.required_output_len()];
            inverse.execute(&packed, &mut restored).unwrap();

            let expected = reference_c2r_from_packed_interleaved(&packed, &inverse_config).unwrap();
            assert_close_f32(&restored, &expected, &format!("c2r {config:?}"));
            assert_close_f32(&restored, &real, &format!("round trip {config:?}"));
        }
    }

    #[test]
    fn large_transforms_split_across_threads_stay_exact_on_a_pure_tone() {
        // A single complex exponential transforms to one scaled delta, which
        // checks the contiguous and transposed (strided) axis paths at sizes
        // large enough to run on several threads.
        let shape = [256usize, 128, 8];
        let (k0, k1, k2) = (5usize, 17usize, 3usize);
        let total: usize = shape.iter().product();
        let mut input = vec![0.0f64; total * 2];
        for i2 in 0..shape[2] {
            for i1 in 0..shape[1] {
                for i0 in 0..shape[0] {
                    let phase = std::f64::consts::TAU
                        * ((k0 * i0) as f64 / shape[0] as f64
                            + (k1 * i1) as f64 / shape[1] as f64
                            + (k2 * i2) as f64 / shape[2] as f64);
                    let index = (i2 * shape[1] + i1) * shape[0] + i0;
                    input[2 * index] = phase.cos();
                    input[2 * index + 1] = phase.sin();
                }
            }
        }
        let plan = CpuFftPlan::c2c(
            FftConfig::new_nd(shape)
                .with_precision(FftPrecision::F64)
                .with_normalization(Normalization::Forward),
        )
        .unwrap();
        let mut output = vec![0.0f64; total * 2];
        plan.execute_f64(&input, &mut output).unwrap();
        let peak = (k2 * shape[1] + k1) * shape[0] + k0;
        for (index, pair) in output.as_chunks::<2>().0.iter().enumerate() {
            let expected = if index == peak { 1.0 } else { 0.0 };
            assert!(
                (pair[0] - expected).abs() < 1e-9 && pair[1].abs() < 1e-9,
                "bin {index}: {pair:?}"
            );
        }
    }

    #[test]
    fn plans_report_lengths_and_reject_mismatches() {
        let plan = CpuFftPlan::c2c(FftConfig::new(8).with_batch(2)).unwrap();
        assert_eq!(plan.kind(), FftTransformKind::C2c);
        assert_eq!(plan.required_input_len(), 32);
        let mut output = vec![0.0f32; 32];
        assert_eq!(
            plan.execute(&[0.0; 30], &mut output),
            Err(FftError::HostBufferLength {
                buffer: "input",
                expected: 32,
                actual: 30
            })
        );
        assert!(matches!(
            plan.execute_f64(&[0.0; 32], &mut [0.0; 32]),
            Err(FftError::PrecisionUnsupported { .. })
        ));

        let df64 = CpuFftPlan::c2c(FftConfig::new(8).with_precision(FftPrecision::Df64)).unwrap();
        assert_eq!(df64.required_input_len(), 32);

        let r2c = CpuFftPlan::r2c(FftConfig::new_nd([10, 3])).unwrap();
        assert_eq!(r2c.packed_shape(), Some(&[6, 3][..]));
        assert_eq!(
            (r2c.required_input_len(), r2c.required_output_len()),
            (30, 36)
        );

        assert!(matches!(
            CpuFftPlan::r2c(FftConfig::inverse(8)),
            Err(FftError::InvalidRealTransformDirection { .. })
        ));
        assert!(matches!(
            CpuFftPlan::r2c(FftConfig::new_nd([4, 4]).with_axes([1])),
            Err(FftError::UnsupportedRealAxes { .. })
        ));
        assert!(matches!(
            CpuFftPlan::c2r(FftConfig::inverse(8).with_precision(FftPrecision::Df64)),
            Err(FftError::PrecisionUnsupported { .. })
        ));
        assert_eq!(
            CpuFftPlan::c2c(FftConfig::new(0)).err(),
            Some(FftError::ZeroLength)
        );
    }
}
