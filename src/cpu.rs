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

use std::sync::{Arc, Mutex};

use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};
use rustfft::num_complex::Complex;
use rustfft::num_traits::Zero;
use rustfft::{Fft, FftNum, FftPlanner};

use crate::config::{FftConfig, FftDirection, FftPrecision};
use crate::error::{FftError, Result};
use crate::plan::FftTransformKind;

/// Work below this many elements per extra thread stays on the calling thread.
const MIN_ELEMENTS_PER_WORKER: usize = 1 << 15;
/// Contiguous lines at least this long, in batches of fewer than
/// `FOUR_STEP_MAX_LINES` lines, run as a parallel four-step decomposition.
/// The choice depends on the problem alone, so results do not depend on the
/// machine's thread count.
const FOUR_STEP_MIN_LEN: usize = 1 << 15;
const FOUR_STEP_MAX_LINES: usize = 16;
/// Smallest factor of a four-step decomposition worth its extra passes.
const FOUR_STEP_MIN_FACTOR: usize = 16;

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
    /// Complex transforms in execution order.
    axes: Vec<AxisKernel<T>>,
    real: RealKernel<T>,
    scale: T,
}

/// The transform of one axis.
struct AxisKernel<T: FftNum> {
    axis: usize,
    fft: Arc<dyn Fft<T>>,
    /// A parallel decomposition of long contiguous lines.
    four_step: Option<FourStep<T>>,
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

    /// Executes an `F32` or `Df64` complex-to-complex plan in place on `f32`
    /// words, which hold the input and receive the output.
    ///
    /// `F64` plans must use [`Self::execute_in_place_f64`]; real transforms
    /// change the element count and cannot run in place.
    pub fn execute_in_place(&self, data: &mut [f32]) -> Result<()> {
        self.check_in_place(data.len())?;
        match (&self.kernels, self.config.precision()) {
            (Kernels::F32(kernels), _) => {
                kernels.run_in_place(self.config.shape(), data);
                Ok(())
            }
            (Kernels::F64(kernels), FftPrecision::Df64) => {
                let input = data.to_vec();
                kernels.run_df64_c2c(self.config.shape(), &input, data);
                Ok(())
            }
            (Kernels::F64(_), requested) => Err(FftError::PrecisionUnsupported {
                requested,
                route: "cpu",
                reason: "f64-plans-execute-with-execute_in_place_f64",
            }),
        }
    }

    /// Executes an `F64` complex-to-complex plan in place on `f64` words.
    pub fn execute_in_place_f64(&self, data: &mut [f64]) -> Result<()> {
        self.check_in_place(data.len())?;
        match (&self.kernels, self.config.precision()) {
            (Kernels::F64(kernels), FftPrecision::F64) => {
                kernels.run_in_place(self.config.shape(), data);
                Ok(())
            }
            (_, requested) => Err(FftError::PrecisionUnsupported {
                requested,
                route: "cpu",
                reason: "execute_in_place_f64-requires-an-f64-plan",
            }),
        }
    }

    fn check_in_place(&self, len: usize) -> Result<()> {
        if self.kind != FftTransformKind::C2c {
            return Err(FftError::InPlaceUnsupported {
                route: "cpu",
                reason: "real-transforms-change-the-element-count",
            });
        }
        self.check_lengths(len, len)
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
            .map(|&axis| {
                let len = shape[axis];
                let contiguous = shape[..axis].iter().product::<usize>() == 1;
                AxisKernel {
                    axis,
                    fft: planner.plan_fft(len, direction),
                    four_step: contiguous
                        .then(|| FourStep::new(len, direction, &mut planner))
                        .flatten(),
                }
            })
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
                copy_parallel(input, output);
                self.run_in_place(shape, output);
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

    /// Runs a complex-to-complex plan on `data` in place.
    fn run_in_place(&self, shape: &[usize], data: &mut [T]) {
        let data: &mut [Complex<T>] = bytemuck::cast_slice_mut(data);
        self.transform_axes(data, shape);
        scale_all(data, self.scale);
    }

    fn transform_axes(&self, data: &mut [Complex<T>], shape: &[usize]) {
        for kernel in &self.axes {
            let len = shape[kernel.axis];
            match &kernel.four_step {
                // Too few lines to keep threads busy: split each line.
                Some(four_step) if data.len() / len < FOUR_STEP_MAX_LINES => {
                    for line in data.chunks_exact_mut(len) {
                        four_step.run(line);
                    }
                }
                _ => transform_axis(data, shape, kernel.axis, kernel.fft.as_ref()),
            }
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

/// Transforms every line of `axis` in place.
fn transform_axis<T: FftNum>(
    data: &mut [Complex<T>],
    shape: &[usize],
    axis: usize,
    fft: &dyn Fft<T>,
) {
    let len = shape[axis];
    if len == 1 {
        return;
    }
    // Elements between consecutive values of one line.
    let inner = shape[..axis].iter().product::<usize>();
    if inner == 1 {
        process_lines(data, len, fft);
    } else {
        transform_strided(data, len, inner, fft, None);
    }
}

/// Called with the column index and the transformed line of every column
/// of a strided transform.
type ColumnHook<'a, T> = &'a (dyn Fn(usize, &mut [Complex<T>]) + Sync);

/// Transforms, in place, the lines of `data` viewed as `[outer][len][inner]`
/// along its middle dimension.
///
/// Tiles of neighbouring columns are gathered into a contiguous buffer,
/// transformed together, and written back, so no transposed copy of the
/// data is needed. With enough blocks, each thread takes whole blocks;
/// otherwise each thread takes a range of columns of every row.
fn transform_strided<T: FftNum>(
    data: &mut [Complex<T>],
    len: usize,
    inner: usize,
    fft: &dyn Fft<T>,
    hook: Option<ColumnHook<'_, T>>,
) {
    let block = len * inner;
    let outer = data.len() / block;
    let workers = worker_count(data.len());
    if workers <= 1 || outer >= workers {
        for_each_chunk_group(data, block, |_, blocks| {
            let mut tile = Tile::new(len, fft);
            for block in blocks.chunks_exact_mut(block) {
                let mut rows = block.chunks_exact_mut(inner).collect::<Vec<_>>();
                tile.transform_columns(&mut rows, 0, hook);
            }
        });
        return;
    }
    let tile_width = tile_width::<T>();
    let width = inner.div_ceil(workers).div_ceil(tile_width) * tile_width;
    let parts = inner.div_ceil(width);
    let mut columns = (0..parts)
        .map(|_| Vec::with_capacity(outer * len))
        .collect::<Vec<_>>();
    for row in data.chunks_exact_mut(inner) {
        let mut rest = row;
        for part in &mut columns {
            let count = width.min(rest.len());
            let (head, tail) = std::mem::take(&mut rest).split_at_mut(count);
            part.push(head);
            rest = tail;
        }
    }
    std::thread::scope(|scope| {
        for (part, mut rows) in columns.into_iter().enumerate() {
            scope.spawn(move || {
                let mut tile = Tile::new(len, fft);
                for block_rows in rows.chunks_exact_mut(len) {
                    tile.transform_columns(block_rows, part * width, hook);
                }
            });
        }
    });
}

/// Complex values per row that one tile gathers: 256 bytes, so every row
/// contributes whole cache lines.
fn tile_width<T>() -> usize {
    (256 / std::mem::size_of::<Complex<T>>()).max(1)
}

/// A thread's buffers for transforming strided lines through tiles.
struct Tile<'a, T: FftNum> {
    fft: &'a dyn Fft<T>,
    len: usize,
    lines: Vec<Complex<T>>,
    scratch: Vec<Complex<T>>,
}

impl<'a, T: FftNum> Tile<'a, T> {
    fn new(len: usize, fft: &'a dyn Fft<T>) -> Self {
        Self {
            fft,
            len,
            lines: vec![Complex::zero(); tile_width::<T>() * len],
            scratch: vec![Complex::zero(); fft.get_inplace_scratch_len()],
        }
    }

    /// Transforms every column of `rows` in place, one line per column over
    /// the `len` rows; `first` is the index of the first column, for `hook`.
    fn transform_columns(
        &mut self,
        rows: &mut [&mut [Complex<T>]],
        first: usize,
        hook: Option<ColumnHook<'_, T>>,
    ) {
        let len = self.len;
        let width = rows.first().map_or(0, |row| row.len());
        let mut column = 0;
        while column < width {
            let count = tile_width::<T>().min(width - column);
            let lines = &mut self.lines[..count * len];
            for (n, row) in rows.iter().enumerate() {
                for (t, &value) in row[column..column + count].iter().enumerate() {
                    lines[t * len + n] = value;
                }
            }
            self.fft.process_with_scratch(lines, &mut self.scratch);
            if let Some(hook) = hook {
                for (t, line) in lines.chunks_exact_mut(len).enumerate() {
                    hook(first + column + t, line);
                }
            }
            for (n, row) in rows.iter_mut().enumerate() {
                for (t, value) in row[column..column + count].iter_mut().enumerate() {
                    *value = lines[t * len + n];
                }
            }
            column += count;
        }
    }
}

/// A long line of `rows * columns` points transformed as a matrix, row
/// `n1` holding points `n1 * columns..(n1 + 1) * columns`: a transform of
/// every column, twiddles, a transform of every row, and a transpose. Each
/// step runs on every thread.
struct FourStep<T: FftNum> {
    len: usize,
    rows: usize,
    columns: usize,
    /// Transforms a column: `rows` points.
    column_fft: Arc<dyn Fft<T>>,
    /// Transforms a row: `columns` points.
    row_fft: Arc<dyn Fft<T>>,
    /// The twiddle `w^e` is `coarse[e >> shift] * fine[e & mask]`.
    coarse: Vec<Complex<f64>>,
    fine: Vec<Complex<f64>>,
    shift: u32,
    /// The transpose buffer, kept between executions.
    work: Mutex<Vec<Complex<T>>>,
}

impl<T: FftNum> FourStep<T> {
    /// Splits `len` into its two factors closest to its square root, if
    /// neither is small.
    fn new(
        len: usize,
        direction: rustfft::FftDirection,
        planner: &mut FftPlanner<T>,
    ) -> Option<Self> {
        if len < FOUR_STEP_MIN_LEN {
            return None;
        }
        let mut rows = (len as f64).sqrt() as usize;
        while rows > 1 && !len.is_multiple_of(rows) {
            rows -= 1;
        }
        if rows < FOUR_STEP_MIN_FACTOR {
            return None;
        }
        let columns = len / rows;
        let sign = match direction {
            rustfft::FftDirection::Forward => -1.0,
            rustfft::FftDirection::Inverse => 1.0,
        };
        let root = |exponent: usize| {
            let angle = sign * std::f64::consts::TAU * exponent as f64 / len as f64;
            Complex::new(angle.cos(), angle.sin())
        };
        let shift = (usize::BITS - len.leading_zeros()).div_ceil(2);
        let fine = (0..1usize << shift).map(root).collect();
        let coarse = (0..=(len - 1) >> shift)
            .map(|high| root(high << shift))
            .collect();
        Some(Self {
            len,
            rows,
            columns,
            column_fft: planner.plan_fft(rows, direction),
            row_fft: planner.plan_fft(columns, direction),
            coarse,
            fine,
            shift,
            work: Mutex::new(Vec::new()),
        })
    }

    /// Transforms one line in place.
    fn run(&self, line: &mut [Complex<T>]) {
        let (rows, columns) = (self.rows, self.columns);
        // Column n2 holds Y[k1] after its transform; scale it by w^(n2 * k1).
        let twiddle = |column: usize, values: &mut [Complex<T>]| {
            let mut exponent = 0;
            for value in values {
                let w = self.coarse[exponent >> self.shift]
                    * self.fine[exponent & ((1 << self.shift) - 1)];
                let w = Complex::new(
                    T::from_f64(w.re).expect("twiddles are finite"),
                    T::from_f64(w.im).expect("twiddles are finite"),
                );
                *value = *value * w;
                exponent += column;
                if exponent >= self.len {
                    exponent -= self.len;
                }
            }
        };
        transform_strided(
            line,
            rows,
            columns,
            self.column_fft.as_ref(),
            Some(&twiddle),
        );
        process_lines(line, columns, self.row_fft.as_ref());
        // Row k1 holds X[k1 + rows * k2] at column k2: transpose.
        let mut work = self
            .work
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if work.len() != self.len {
            *work = vec![Complex::zero(); self.len];
        }
        transpose(line, rows, columns, &mut work);
        copy_parallel(&work, line);
    }
}

/// Writes `source`, viewed as `[rows][columns]`, into `target` as
/// `[columns][rows]`, in tiles that keep both sides in cache.
fn transpose<T: Copy + Send + Sync>(source: &[T], rows: usize, columns: usize, target: &mut [T]) {
    const TILE: usize = 32;
    for_each_chunk_group(target, rows, |first, chunk| {
        let count = chunk.len() / rows;
        for start in (0..count).step_by(TILE) {
            let tile = TILE.min(count - start);
            for row in 0..rows {
                let values = &source[row * columns + first + start..][..tile];
                for (offset, &value) in values.iter().enumerate() {
                    chunk[(start + offset) * rows + row] = value;
                }
            }
        }
    });
}

/// Copies `input` into `output` on several threads for large inputs.
fn copy_parallel<T: Copy + Send + Sync>(input: &[T], output: &mut [T]) {
    for_each_chunk_group(output, 1, |first, chunk| {
        chunk.copy_from_slice(&input[first..first + chunk.len()]);
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

    /// Transforms every selected axis line by line with rustfft, as a
    /// reference for the tiled, split, and four-step paths.
    fn line_by_line(values: &[Complex<f64>], config: &FftConfig) -> Vec<Complex<f64>> {
        let shape = config.shape();
        let direction = match config.direction() {
            FftDirection::Forward => rustfft::FftDirection::Forward,
            FftDirection::Inverse => rustfft::FftDirection::Inverse,
        };
        let mut data = values.to_vec();
        let total = shape.iter().product::<usize>();
        let mut planner = FftPlanner::<f64>::new();
        for &axis in config.axes() {
            let len = shape[axis];
            let inner = shape[..axis].iter().product::<usize>();
            let fft = planner.plan_fft(len, direction);
            let mut line = vec![Complex::zero(); len];
            for transform in data.chunks_exact_mut(total) {
                for outer in 0..total / (len * inner) {
                    for column in 0..inner {
                        let base = outer * len * inner + column;
                        for (n, value) in line.iter_mut().enumerate() {
                            *value = transform[base + n * inner];
                        }
                        fft.process(&mut line);
                        for (n, value) in line.iter().enumerate() {
                            transform[base + n * inner] = *value;
                        }
                    }
                }
            }
        }
        let scale = config.scale_f64().unwrap();
        data.iter().map(|value| value * scale).collect()
    }

    fn large_configs() -> Vec<FftConfig> {
        vec![
            // Few blocks: threads split the columns of every row.
            FftConfig::new_nd([512, 400]),
            FftConfig::inverse_nd([300, 700]),
            // Blocks per thread on the middle axis, columns on the last.
            FftConfig::new_nd([64, 48, 40]).with_batch(3),
            FftConfig::new_nd([40, 64, 48]).with_axes([2, 0]),
            // Long lines: the four-step decomposition.
            FftConfig::new(100_000),
            FftConfig::inverse(98_304).with_batch(2),
            FftConfig::new_nd([70_000, 3]).with_axes([0]),
            // A prime length keeps one transform per line.
            FftConfig::new(65_537),
        ]
    }

    #[test]
    fn tiled_split_and_four_step_paths_match_a_line_by_line_transform() {
        for config in large_configs() {
            let total = config.total_complex_len().unwrap();
            let values = (0..total)
                .map(|index| {
                    let x = index as f64;
                    Complex::new((x * 0.37).sin() + 0.25, (x * 0.11).cos() - 0.5)
                })
                .collect::<Vec<_>>();
            let expected = line_by_line(&values, &config);
            let peak = expected.iter().fold(0.0f64, |m, v| m.max(v.norm()));

            let plan = CpuFftPlan::c2c(config.clone().with_precision(FftPrecision::F64)).unwrap();
            let input = values.iter().flat_map(|v| [v.re, v.im]).collect::<Vec<_>>();
            let mut output = vec![0.0f64; input.len()];
            plan.execute_f64(&input, &mut output).unwrap();
            let mut in_place = input.clone();
            plan.execute_in_place_f64(&mut in_place).unwrap();
            assert_eq!(output, in_place, "{config:?}: in place differs");
            let worst = output
                .as_chunks::<2>()
                .0
                .iter()
                .zip(&expected)
                .map(|(&[re, im], e)| (Complex::new(re, im) - e).norm())
                .fold(0.0f64, f64::max);
            assert!(worst <= 1e-12 * peak, "f64 {config:?}: {worst} of {peak}");

            let plan = CpuFftPlan::c2c(config.clone()).unwrap();
            let input = input.iter().map(|&v| v as f32).collect::<Vec<_>>();
            let mut output = vec![0.0f32; input.len()];
            plan.execute(&input, &mut output).unwrap();
            let worst = output
                .as_chunks::<2>()
                .0
                .iter()
                .zip(&expected)
                .map(|(&[re, im], e)| (Complex::new(f64::from(re), f64::from(im)) - e).norm())
                .fold(0.0f64, f64::max);
            assert!(worst <= 2e-5 * peak, "f32 {config:?}: {worst} of {peak}");
        }
    }

    #[test]
    fn real_plans_cannot_run_in_place() {
        let plan = CpuFftPlan::r2c(FftConfig::new(8)).unwrap();
        assert!(matches!(
            plan.execute_in_place(&mut [0.0; 8]),
            Err(FftError::InPlaceUnsupported { .. })
        ));
        let plan = CpuFftPlan::c2c(FftConfig::new(8)).unwrap();
        assert!(matches!(
            plan.execute_in_place(&mut [0.0; 12]),
            Err(FftError::HostBufferLength { .. })
        ));
        assert!(matches!(
            plan.execute_in_place_f64(&mut [0.0; 16]),
            Err(FftError::PrecisionUnsupported { .. })
        ));
    }
}
