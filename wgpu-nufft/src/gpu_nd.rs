//! Rank-generic (runtime-dimensional) type-2 GPU execution.
//!
//! This is adoption phase 1 of the ND NUFFT design (vault note "ND NUFFT Math
//! Design", section 11): dimension appears only as generated WGSL, so one
//! implementation serves every rank the config validates. Plan routing keeps
//! the hand-tuned 1D/2D/3D paths for d <= 3 and selects this module for
//! d >= 4. All three precisions are supported: F32, native F64 (requires
//! `wgpu::Features::SHADER_F64`, gated in plan.rs), and portable Df64.
//!
//! Conventions match the fixed-rank modules exactly: interleaved point-major
//! coordinates in FINUFFT's `[-3*pi, 3*pi]` radian interval, axis-0-fast mode
//! and fine-grid linearization, batch-major fine grids, and sub-cell position
//! folding in emulated df64 arithmetic even for f32 data. F64/Df64 evaluate
//! the ES kernel through the host-fitted Horner table (GLSL.std.450
//! transcendentals are 32-bit only), matching the fixed-rank shaders.

use std::fmt::Write as _;
use std::num::NonZeroU64;

use wgpu::util::DeviceExt;
use wgpu_fft::math::DoubleFloat;
use wgpu_fft::{BufferView, FftConfig, FftDirection, FftPlan, FftPrecision, Normalization};

use crate::config::{ModeOrder, NufftConfig, NufftSign};
use crate::direct::mode_for_storage_index;
use crate::error::{NufftError, Result};
use crate::gpu::max_supported_workgroup_size;
use crate::gpu_dispatch::split_workgroups;
#[cfg(feature = "gpu-profiling")]
use crate::gpu_profile::{GpuProfileQueryWriter, NufftGpuProfileLayout};
use crate::kernel::EsKernel;

const WORKGROUP_SIZE: u32 = 64;
const VECTOR_TILE: usize = 4;

pub(crate) struct Type2GpuPlanNd {
    fft: FftPlan,
    amplitudes: wgpu::Buffer,
    fine_input: wgpu::Buffer,
    fine_output: wgpu::Buffer,
    predeconvolution_pipeline: wgpu::ComputePipeline,
    predeconvolution_layout: wgpu::BindGroupLayout,
    interpolation_pipeline: wgpu::ComputePipeline,
    interpolation_layout: wgpu::BindGroupLayout,
    predeconvolution_dispatch: (u32, u32, u32),
    max_workgroups_per_dimension: u32,
    batch_capacity: usize,
    mode_count: usize,
    dimensions: usize,
    precision: FftPrecision,
    max_storage_binding_bytes: u64,
}

impl Type2GpuPlanNd {
    pub(crate) fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        config: &NufftConfig,
        kernel: EsKernel,
        fine_shape: &[usize],
        kernel_fourier_coefficients: &[Vec<f64>],
    ) -> Result<Self> {
        let dimensions = config.dimensions();
        debug_assert_eq!(fine_shape.len(), dimensions);
        debug_assert_eq!(kernel_fourier_coefficients.len(), dimensions);
        let precision = config.precision();
        // F32 and Df64 fold integer support indices through an f32 high word;
        // native f64 keeps adjacent cells distinct across the signed i32 range.
        let maximum_signed_length = match precision {
            FftPrecision::F64 => i32::MAX as usize - kernel.width(),
            FftPrecision::F32 | FftPrecision::Df64 => (1usize << f32::MANTISSA_DIGITS)
                .saturating_sub(kernel.width())
                .min(i32::MAX as usize - kernel.width()),
        };
        for (&length, &modes) in fine_shape.iter().zip(config.n_modes()) {
            if length % 2 != 0 {
                return Err(NufftError::FineGridMustBeEven { length });
            }
            debug_assert!(
                length >= modes,
                "select_fine_grid_size returned a fine axis below the mode count"
            );
            if length > maximum_signed_length {
                return Err(NufftError::GpuFineGridIndexUnsupported {
                    length,
                    maximum: maximum_signed_length,
                });
            }
            u32::try_from(length).map_err(|_| NufftError::LengthOverflow {
                context: "rank-generic type-2 fine-grid axis length",
            })?;
        }
        let limits = device.limits();
        let maximum_workgroup_size = max_supported_workgroup_size(&limits);
        if maximum_workgroup_size < WORKGROUP_SIZE {
            return Err(NufftError::GpuWorkgroupSizeUnsupported {
                requested: WORKGROUP_SIZE,
                maximum: maximum_workgroup_size,
            });
        }

        let batch_capacity = config.batch();
        let mode_count = config.mode_count()?;
        let fine_product = fine_shape.iter().try_fold(1usize, |product, &length| {
            product
                .checked_mul(length)
                .ok_or(NufftError::LengthOverflow {
                    context: "rank-generic type-2 fine-grid element count",
                })
        })?;
        let fine_element_count = checked_product(
            "rank-generic type-2 batched fine-grid element count",
            fine_product,
            batch_capacity,
        )?;
        let fine_element_count_u32 =
            u32::try_from(fine_element_count).map_err(|_| NufftError::LengthOverflow {
                context: "rank-generic type-2 batched fine-grid shader index space",
            })?;
        let fine_bytes = checked_buffer_size(
            "rank-generic type-2 fine grid",
            fine_element_count,
            precision.complex_size_bytes(),
        )?;
        let amplitude_bytes = checked_buffer_size(
            "rank-generic type-2 deconvolution amplitudes",
            mode_count,
            precision.scalar_size_bytes(),
        )?;
        let max_storage_binding_bytes = limits.max_storage_buffer_binding_size;
        validate_binding_limit(
            "rank-generic type-2 fine grid",
            fine_bytes,
            max_storage_binding_bytes,
        )?;
        validate_binding_limit(
            "rank-generic type-2 deconvolution amplitudes",
            amplitude_bytes,
            max_storage_binding_bytes,
        )?;
        if fine_bytes > limits.max_buffer_size {
            return Err(NufftError::GpuBufferBindingTooLarge {
                buffer: "rank-generic type-2 fine grid",
                required_bytes: fine_bytes,
                limit_bytes: limits.max_buffer_size,
            });
        }

        let amplitudes = mode_amplitude_bytes_nd(config, kernel_fourier_coefficients, precision)?;
        let amplitude_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("wgpu_nufft.type2_nd.deconvolution_amplitudes"),
            contents: &amplitudes,
            usage: wgpu::BufferUsages::STORAGE,
        });
        let fine_usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST;
        let fine_input = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("wgpu_nufft.type2_nd.fine_input"),
            size: fine_bytes,
            usage: fine_usage,
            mapped_at_creation: false,
        });
        let fine_output = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("wgpu_nufft.type2_nd.fine_output"),
            size: fine_bytes,
            usage: fine_usage,
            mapped_at_creation: false,
        });

        let fft_direction = match config.sign() {
            NufftSign::Positive => FftDirection::Inverse,
            NufftSign::Negative => FftDirection::Forward,
        };
        let fft_config = FftConfig::new_nd(fine_shape.to_vec())
            .with_direction(fft_direction)
            .with_normalization(Normalization::None)
            .with_batch(batch_capacity)
            .with_precision(precision);
        let fft = FftPlan::c2c(device, queue, fft_config).map_err(|source| {
            NufftError::FftShapeUnsupported {
                stage: "rank-generic type-2 oversampled-grid C2C plan",
                source,
            }
        })?;

        let predeconvolution_pipeline = create_compute_pipeline(
            device,
            "wgpu_nufft.type2_nd.predeconvolution",
            &generate_predeconvolution_wgsl_for_precision(config, fine_shape),
        );
        let predeconvolution_layout = predeconvolution_pipeline.get_bind_group_layout(0);
        let interpolation_pipeline = create_compute_pipeline(
            device,
            "wgpu_nufft.type2_nd.interpolation",
            &generate_interpolation_wgsl_for_precision(precision, kernel, fine_shape),
        );
        let interpolation_layout = interpolation_pipeline.get_bind_group_layout(0);

        let workgroups = fine_element_count_u32.div_ceil(WORKGROUP_SIZE);
        let max_workgroups_per_dimension = limits.max_compute_workgroups_per_dimension;
        let predeconvolution_dispatch = split_workgroups(workgroups, max_workgroups_per_dimension)?;

        Ok(Self {
            fft,
            amplitudes: amplitude_buffer,
            fine_input,
            fine_output,
            predeconvolution_pipeline,
            predeconvolution_layout,
            interpolation_pipeline,
            interpolation_layout,
            predeconvolution_dispatch,
            max_workgroups_per_dimension,
            batch_capacity,
            mode_count,
            dimensions,
            precision,
            max_storage_binding_bytes,
        })
    }

    #[allow(dead_code)]
    pub(crate) fn encode(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        point_count: usize,
        points: &wgpu::Buffer,
        coefficients: &wgpu::Buffer,
        output: &wgpu::Buffer,
    ) -> Result<()> {
        self.encode_batch(
            device,
            encoder,
            self.batch_capacity,
            point_count,
            points,
            coefficients,
            output,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_batch(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        active_batch: usize,
        point_count: usize,
        points: &wgpu::Buffer,
        coefficients: &wgpu::Buffer,
        output: &wgpu::Buffer,
    ) -> Result<()> {
        self.encode_impl(
            device,
            encoder,
            active_batch,
            point_count,
            points,
            coefficients,
            output,
            #[cfg(feature = "gpu-profiling")]
            GpuProfileQueryWriter::disabled(),
        )
    }

    #[cfg(feature = "gpu-profiling")]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_profiled(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        point_count: usize,
        points: &wgpu::Buffer,
        coefficients: &wgpu::Buffer,
        output: &wgpu::Buffer,
        query_set: &wgpu::QuerySet,
        first_query: u32,
    ) -> Result<NufftGpuProfileLayout> {
        if point_count == 0 {
            return Err(NufftError::GpuExecutionUnavailable {
                kind: "rank-generic type-2 stage profiling",
                reason: "at least one point is required",
            });
        }
        let layout =
            NufftGpuProfileLayout::type2(first_query).map_err(|_| NufftError::LengthOverflow {
                context: "rank-generic type-2 stage-profile query range",
            })?;
        self.encode_impl(
            device,
            encoder,
            self.batch_capacity,
            point_count,
            points,
            coefficients,
            output,
            GpuProfileQueryWriter::enabled(query_set, &layout),
        )?;
        Ok(layout)
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_impl(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        active_batch: usize,
        point_count: usize,
        points: &wgpu::Buffer,
        coefficients: &wgpu::Buffer,
        output: &wgpu::Buffer,
        #[cfg(feature = "gpu-profiling")] profile: GpuProfileQueryWriter<'_>,
    ) -> Result<()> {
        if active_batch == 0 || active_batch > self.batch_capacity {
            return Err(NufftError::InvalidBatch {
                actual: active_batch,
                maximum: self.batch_capacity,
            });
        }
        if point_count == 0 {
            return Ok(());
        }
        u32::try_from(point_count).map_err(|_| NufftError::LengthOverflow {
            context: "rank-generic type-2 GPU point count",
        })?;
        let coordinate_count = checked_product(
            "rank-generic type-2 coordinate scalar count",
            point_count,
            self.dimensions,
        )?;
        let point_bytes = checked_buffer_size(
            "rank-generic type-2 point buffer",
            coordinate_count,
            self.precision.scalar_size_bytes(),
        )?;
        let coefficient_elements = checked_product(
            "rank-generic type-2 batched Fourier coefficient count",
            active_batch,
            self.mode_count,
        )?;
        let output_elements = checked_product(
            "rank-generic type-2 batched output element count",
            active_batch,
            point_count,
        )?;
        let coefficient_bytes = checked_buffer_size(
            "rank-generic type-2 Fourier coefficient buffer",
            coefficient_elements,
            self.precision.complex_size_bytes(),
        )?;
        let output_bytes = checked_buffer_size(
            "rank-generic type-2 output buffer",
            output_elements,
            self.precision.complex_size_bytes(),
        )?;
        validate_external_storage_buffer(
            "rank-generic type-2 point",
            points,
            point_bytes,
            self.max_storage_binding_bytes,
        )?;
        validate_external_storage_buffer(
            "rank-generic type-2 Fourier coefficient",
            coefficients,
            coefficient_bytes,
            self.max_storage_binding_bytes,
        )?;
        validate_external_storage_buffer(
            "rank-generic type-2 output",
            output,
            output_bytes,
            self.max_storage_binding_bytes,
        )?;
        if points == output {
            return Err(NufftError::GpuBufferAliasUnsupported {
                first: "rank-generic type-2 point",
                second: "rank-generic type-2 output",
            });
        }
        let vector_tiles = active_batch.div_ceil(VECTOR_TILE);
        let interpolation_elements = checked_product(
            "rank-generic type-2 batched interpolation work item count",
            point_count,
            vector_tiles,
        )?;
        let interpolation_elements_u32 =
            u32::try_from(interpolation_elements).map_err(|_| NufftError::LengthOverflow {
                context: "rank-generic type-2 batched interpolation shader index space",
            })?;
        let interpolation_dispatch = split_workgroups(
            interpolation_elements_u32.div_ceil(WORKGROUP_SIZE),
            self.max_workgroups_per_dimension,
        )?;

        let predeconvolution_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("wgpu_nufft.type2_nd.predeconvolution.bind_group"),
            layout: &self.predeconvolution_layout,
            entries: &[
                binding_entry(0, coefficients, coefficient_bytes),
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.amplitudes.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.fine_input.as_entire_binding(),
                },
            ],
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("wgpu_nufft.type2_nd.predeconvolution.pass"),
                timestamp_writes: {
                    #[cfg(feature = "gpu-profiling")]
                    {
                        profile.timestamp_writes(Some(0), Some(1))
                    }
                    #[cfg(not(feature = "gpu-profiling"))]
                    {
                        None
                    }
                },
            });
            pass.set_pipeline(&self.predeconvolution_pipeline);
            pass.set_bind_group(0, &predeconvolution_bind_group, &[]);
            pass.dispatch_workgroups(
                self.predeconvolution_dispatch.0,
                self.predeconvolution_dispatch.1,
                self.predeconvolution_dispatch.2,
            );
        }

        self.fft
            .execute_views(
                device,
                encoder,
                BufferView::whole(&self.fine_input),
                BufferView::whole(&self.fine_output),
            )
            .map_err(|source| NufftError::FftExecutionFailed {
                stage: "rank-generic type-2 oversampled-grid C2C transform",
                source,
            })?;

        let interpolation_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("wgpu_nufft.type2_nd.interpolation.bind_group"),
            layout: &self.interpolation_layout,
            entries: &[
                binding_entry(0, points, point_bytes),
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.fine_output.as_entire_binding(),
                },
                binding_entry(2, output, output_bytes),
            ],
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("wgpu_nufft.type2_nd.interpolation.pass"),
                timestamp_writes: {
                    #[cfg(feature = "gpu-profiling")]
                    {
                        profile.timestamp_writes(Some(2), Some(3))
                    }
                    #[cfg(not(feature = "gpu-profiling"))]
                    {
                        None
                    }
                },
            });
            pass.set_pipeline(&self.interpolation_pipeline);
            pass.set_bind_group(0, &interpolation_bind_group, &[]);
            pass.dispatch_workgroups(
                interpolation_dispatch.0,
                interpolation_dispatch.1,
                interpolation_dispatch.2,
            );
        }
        Ok(())
    }

    #[cfg(feature = "gpu-profiling")]
    pub(crate) fn fft_diagnostics(&self) -> wgpu_fft::FftDiagnostics {
        self.fft.diagnostics()
    }
}

/// Product amplitude table over the full mode box, axis-0-fast linearization:
/// `amplitude[flat] = prod_i 1 / phi_hat_i(|k_i|)`.
pub(crate) fn mode_amplitudes_nd(
    config: &NufftConfig,
    coefficients: &[Vec<f64>],
) -> Result<Vec<f32>> {
    let n_modes = config.n_modes();
    let mode_count = config.mode_count()?;
    let mut amplitudes = Vec::new();
    amplitudes
        .try_reserve_exact(mode_count)
        .map_err(|_| NufftError::HostAllocationFailed {
            buffer: "rank-generic type-2 deconvolution amplitudes",
            elements: mode_count,
        })?;
    for flat in 0..mode_count {
        let mut remainder = flat;
        let mut amplitude = 1.0f64;
        for (axis, &length) in n_modes.iter().enumerate() {
            let index = remainder % length;
            remainder /= length;
            let mode = mode_for_storage_index(config.mode_order(), index, length);
            let coefficient = coefficients[axis][mode.unsigned_abs() as usize];
            if !coefficient.is_finite() || coefficient == 0.0 {
                return Err(NufftError::InvalidKernelFourierCoefficient {
                    mode,
                    value: coefficient,
                });
            }
            amplitude /= coefficient;
        }
        let amplitude = amplitude as f32;
        if !amplitude.is_finite() {
            return Err(NufftError::InvalidKernelFourierCoefficient {
                mode: 0,
                value: f64::from(amplitude),
            });
        }
        amplitudes.push(amplitude);
    }
    Ok(amplitudes)
}

/// Full-accuracy variant of [`mode_amplitudes_nd`] for the F64/Df64 paths.
pub(crate) fn mode_amplitudes_nd_f64(
    config: &NufftConfig,
    coefficients: &[Vec<f64>],
) -> Result<Vec<f64>> {
    let n_modes = config.n_modes();
    let mode_count = config.mode_count()?;
    let mut amplitudes = Vec::new();
    amplitudes
        .try_reserve_exact(mode_count)
        .map_err(|_| NufftError::HostAllocationFailed {
            buffer: "rank-generic type-2 deconvolution amplitudes",
            elements: mode_count,
        })?;
    for flat in 0..mode_count {
        let mut remainder = flat;
        let mut amplitude = 1.0f64;
        for (axis, &length) in n_modes.iter().enumerate() {
            let index = remainder % length;
            remainder /= length;
            let mode = mode_for_storage_index(config.mode_order(), index, length);
            let coefficient = coefficients[axis][mode.unsigned_abs() as usize];
            if !coefficient.is_finite() || coefficient == 0.0 {
                return Err(NufftError::InvalidKernelFourierCoefficient {
                    mode,
                    value: coefficient,
                });
            }
            amplitude /= coefficient;
        }
        if !amplitude.is_finite() {
            return Err(NufftError::InvalidKernelFourierCoefficient {
                mode: 0,
                value: amplitude,
            });
        }
        amplitudes.push(amplitude);
    }
    Ok(amplitudes)
}

/// Uploadable amplitude bytes in each precision's storage layout: `f32`,
/// `f64`, or df64 hi/lo pairs.
pub(crate) fn mode_amplitude_bytes_nd(
    config: &NufftConfig,
    coefficients: &[Vec<f64>],
    precision: FftPrecision,
) -> Result<Vec<u8>> {
    match precision {
        FftPrecision::F64 => {
            Ok(bytemuck::cast_slice(&mode_amplitudes_nd_f64(config, coefficients)?).to_vec())
        }
        FftPrecision::Df64 => {
            let values = mode_amplitudes_nd_f64(config, coefficients)?
                .into_iter()
                .map(DoubleFloat::from_f64)
                .collect::<Vec<_>>();
            Ok(bytemuck::cast_slice(&values).to_vec())
        }
        FftPrecision::F32 => {
            Ok(bytemuck::cast_slice(&mode_amplitudes_nd(config, coefficients)?).to_vec())
        }
    }
}

pub(crate) fn create_compute_pipeline(
    device: &wgpu::Device,
    label: &str,
    source: &str,
) -> wgpu::ComputePipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: None,
        module: &shader,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    })
}

fn generate_predeconvolution_wgsl_for_precision(
    config: &NufftConfig,
    fine_shape: &[usize],
) -> String {
    match config.precision() {
        FftPrecision::F64 => generate_predeconvolution_wgsl_f64(config, fine_shape),
        FftPrecision::Df64 => generate_predeconvolution_wgsl_df64(config, fine_shape),
        FftPrecision::F32 => generate_predeconvolution_wgsl(config, fine_shape),
    }
}

/// Native-f64 lowering of the f32 shader: WGSL abstract-float literals
/// concretize to `f64` after the type substitution, matching `gpu_3d.rs`.
fn generate_predeconvolution_wgsl_f64(config: &NufftConfig, fine_shape: &[usize]) -> String {
    generate_predeconvolution_wgsl(config, fine_shape)
        .replace("vec2<f32>", "vec2<f64>")
        .replace("array<f32>", "array<f64>")
}

fn generate_predeconvolution_wgsl_df64(config: &NufftConfig, fine_shape: &[usize]) -> String {
    let source = generate_predeconvolution_wgsl(config, fine_shape)
        .replace("array<vec2<f32>>", "array<vec4<f32>>")
        .replace("array<f32>", "array<vec2<f32>>")
        .replace(
            "fine_grid[index] = vec2<f32>(0.0, 0.0);",
            "fine_grid[index] = vec4<f32>(0.0, 0.0, 0.0, 0.0);",
        )
        .replace(
            "fine_grid[index] = coefficients[coefficient_index] * amplitudes[source_index];",
            "let amplitude = amplitudes[source_index];\n    \
             fine_grid[index] = df64_complex_scale(\n        \
             coefficients[coefficient_index],\n        \
             Df64(amplitude.x, amplitude.y),\n    );",
        );
    format!("{}\n{source}", wgpu_fft::kernels::DF64_WGSL)
}

fn generate_predeconvolution_wgsl(config: &NufftConfig, fine_shape: &[usize]) -> String {
    let dimensions = config.dimensions();
    let n_modes = config.n_modes();
    let batch_capacity = config.batch();
    let fine_product: usize = fine_shape.iter().product();
    let total_fine_count = fine_product * batch_capacity;
    let mode_count: usize = n_modes.iter().product();
    let centered = matches!(config.mode_order(), ModeOrder::Centered);

    let mut source = String::new();
    let _ = writeln!(source, "const WORKGROUP_SIZE: u32 = {WORKGROUP_SIZE}u;");
    let _ = writeln!(source, "const FINE_PRODUCT: u32 = {fine_product}u;");
    let _ = writeln!(source, "const TOTAL_FINE_COUNT: u32 = {total_fine_count}u;");
    let _ = writeln!(source, "const MODE_COUNT: u32 = {mode_count}u;");
    let _ = writeln!(source, "const CENTERED_ORDER: bool = {centered};");
    for axis in 0..dimensions {
        let modes = n_modes[axis];
        let _ = writeln!(source, "const FINE{axis}: u32 = {}u;", fine_shape[axis]);
        let _ = writeln!(source, "const MODE{axis}: u32 = {modes}u;");
        let _ = writeln!(source, "const HALF{axis}: u32 = {}u;", modes / 2);
        let _ = writeln!(source, "const NONNEG{axis}: u32 = {}u;", modes.div_ceil(2));
    }
    source.push_str(
        "\n@group(0) @binding(0) var<storage, read> coefficients: array<vec2<f32>>;\n\
         @group(0) @binding(1) var<storage, read> amplitudes: array<f32>;\n\
         @group(0) @binding(2) var<storage, read_write> fine_grid: array<vec2<f32>>;\n\n",
    );
    source.push_str(
        "fn map_axis(fine_index: u32, mode_count: u32, fine_length: u32, half: u32, nonnegative: u32) -> i32 {\n\
         \x20   if (fine_index < nonnegative) {\n\
         \x20       return i32(select(fine_index, half + fine_index, CENTERED_ORDER));\n\
         \x20   }\n\
         \x20   if (fine_index >= fine_length - half) {\n\
         \x20       let magnitude = fine_length - fine_index;\n\
         \x20       return i32(select(mode_count - magnitude, half - magnitude, CENTERED_ORDER));\n\
         \x20   }\n\
         \x20   return -1;\n}\n\n",
    );
    source.push_str(
        "@compute @workgroup_size(WORKGROUP_SIZE)\n\
         fn main(\n\
         \x20   @builtin(local_invocation_id) lid: vec3<u32>,\n\
         \x20   @builtin(workgroup_id) wid: vec3<u32>,\n\
         \x20   @builtin(num_workgroups) nwg: vec3<u32>,\n\
         ) {\n\
         \x20   let wg_flat = (wid.z * nwg.y + wid.y) * nwg.x + wid.x;\n\
         \x20   if (wg_flat > (TOTAL_FINE_COUNT - 1u) / WORKGROUP_SIZE) { return; }\n\
         \x20   let index = wg_flat * WORKGROUP_SIZE + lid.x;\n\
         \x20   if (index >= TOTAL_FINE_COUNT) { return; }\n\n\
         \x20   let batch_index = index / FINE_PRODUCT;\n\
         \x20   let grid_index = index - batch_index * FINE_PRODUCT;\n\
         \x20   let active_batch = arrayLength(&coefficients) / MODE_COUNT;\n\
         \x20   if (batch_index >= active_batch) {\n\
         \x20       fine_grid[index] = vec2<f32>(0.0, 0.0);\n\
         \x20       return;\n\
         \x20   }\n\n\
         \x20   var remainder = grid_index;\n",
    );
    for axis in 0..dimensions {
        let _ = writeln!(
            source,
            "    let fine_index{axis} = remainder % FINE{axis}; remainder = remainder / FINE{axis};\n    \
             let mapped{axis} = map_axis(fine_index{axis}, MODE{axis}, FINE{axis}, HALF{axis}, NONNEG{axis});"
        );
    }
    let unmapped = (0..dimensions)
        .map(|axis| format!("mapped{axis} < 0"))
        .collect::<Vec<_>>()
        .join(" || ");
    let _ = writeln!(
        source,
        "    if ({unmapped}) {{\n        fine_grid[index] = vec2<f32>(0.0, 0.0);\n        return;\n    }}"
    );
    // axis-0-fast fold: source = m0 + MODE0 * (m1 + MODE1 * (...))
    let mut fold = format!("u32(mapped{})", dimensions - 1);
    for axis in (0..dimensions.saturating_sub(1)).rev() {
        fold = format!("u32(mapped{axis}) + MODE{axis} * ({fold})");
    }
    let _ = writeln!(source, "    let source_index = {fold};");
    source.push_str(
        "    let coefficient_index = batch_index * MODE_COUNT + source_index;\n\
         \x20   fine_grid[index] = coefficients[coefficient_index] * amplitudes[source_index];\n}\n",
    );
    source
}

fn generate_interpolation_wgsl_for_precision(
    precision: FftPrecision,
    kernel: EsKernel,
    fine_shape: &[usize],
) -> String {
    match precision {
        FftPrecision::F64 => generate_interpolation_wgsl_f64(kernel, fine_shape),
        FftPrecision::Df64 => generate_interpolation_wgsl_df64(kernel, fine_shape),
        FftPrecision::F32 => generate_interpolation_wgsl(kernel, fine_shape),
    }
}

fn generate_interpolation_wgsl(kernel: EsKernel, fine_shape: &[usize]) -> String {
    let dimensions = fine_shape.len();
    let width = kernel.width();
    let fine_product: usize = fine_shape.iter().product();

    let mut source = String::new();
    let _ = writeln!(source, "const WORKGROUP_SIZE: u32 = {WORKGROUP_SIZE}u;");
    let _ = writeln!(source, "const DIMS: u32 = {dimensions}u;");
    let _ = writeln!(source, "const VECTOR_TILE: u32 = {VECTOR_TILE}u;");
    let _ = writeln!(source, "const WIDTH: u32 = {width}u;");
    let _ = writeln!(source, "const WIDTH_F32: f32 = {width}.0;");
    let _ = writeln!(
        source,
        "const HALF_WIDTH: f32 = {};",
        format_wgsl_f32(kernel.half_width() as f32)
    );
    let _ = writeln!(
        source,
        "const BETA: f32 = {};",
        format_wgsl_f32(kernel.beta() as f32)
    );
    let _ = writeln!(source, "const FINE_PRODUCT: u32 = {fine_product}u;");
    let mut stride = 1usize;
    for (axis, &length) in fine_shape.iter().enumerate() {
        let position_scale = length as f64 / std::f64::consts::TAU;
        let position_scale_hi = position_scale as f32;
        let position_scale_lo = (position_scale - f64::from(position_scale_hi)) as f32;
        let _ = writeln!(source, "const FINE_LENGTH{axis}: u32 = {length}u;");
        let _ = writeln!(source, "const FINE_LENGTH_I32_{axis}: i32 = {length}i;");
        let _ = writeln!(
            source,
            "const FINE_LENGTH_F32_{axis}: f32 = {};",
            format_wgsl_f32(length as f32)
        );
        let _ = writeln!(
            source,
            "const POSITION_SCALE_HI{axis}: f32 = {};",
            format_wgsl_f32(position_scale_hi)
        );
        let _ = writeln!(
            source,
            "const POSITION_SCALE_LO{axis}: f32 = {};",
            format_wgsl_f32(position_scale_lo)
        );
        let _ = writeln!(
            source,
            "const GRID_ORIGIN{axis}: f32 = {};",
            format_wgsl_f32((length / 2) as f32)
        );
        let _ = writeln!(source, "const FINE_STRIDE{axis}: u32 = {stride}u;");
        stride *= length;
    }
    source.push_str(
        "\n@group(0) @binding(0) var<storage, read> points: array<f32>;\n\
         @group(0) @binding(1) var<storage, read> fine_grid: array<vec2<f32>>;\n\
         @group(0) @binding(2) var<storage, read_write> output_values: array<vec2<f32>>;\n\n",
    );
    source.push_str(
        "fn es_weight(distance: f32) -> f32 {\n\
         \x20   let scaled = 2.0 * abs(distance) / WIDTH_F32;\n\
         \x20   let squared = scaled * scaled;\n\
         \x20   if (squared >= 1.0) { return 0.0; }\n\
         \x20   return exp(BETA * (sqrt(max(0.0, 1.0 - squared)) - 1.0));\n}\n\n\
         fn wrap_index(index: i32, limit: i32) -> u32 {\n\
         \x20   var wrapped = index;\n\
         \x20   if (wrapped < 0) { wrapped = wrapped + limit; }\n\
         \x20   if (wrapped >= limit) { wrapped = wrapped - limit; }\n\
         \x20   return u32(wrapped);\n}\n\n\
         fn ceil_df64_to_i32(value: Df64) -> i32 {\n\
         \x20   let base = floor(value.hi);\n\
         \x20   let remainder = df64_sub(value, Df64(base, 0.0));\n\
         \x20   let has_positive_remainder = remainder.hi > 0.0 ||\n\
         \x20       (remainder.hi == 0.0 && remainder.lo > 0.0);\n\
         \x20   return i32(base) + select(0, 1, has_positive_remainder);\n}\n\n\
         fn position_is_negative(value: Df64) -> bool {\n\
         \x20   return value.hi < 0.0 || (value.hi == 0.0 && value.lo < 0.0);\n}\n\n\
         fn position_at_least(value: Df64, limit: f32) -> bool {\n\
         \x20   return value.hi > limit || (value.hi == limit && value.lo >= 0.0);\n}\n\n\
         fn fold_position(point: f32, scale: vec2<f32>, origin: f32, limit: f32) -> Df64 {\n\
         \x20   let scaled = df64_mul(Df64(point, 0.0), Df64(scale.x, scale.y));\n\
         \x20   var position = df64_add(scaled, Df64(origin, 0.0));\n\
         \x20   if (position_is_negative(position)) { position = df64_add(position, Df64(limit, 0.0)); }\n\
         \x20   if (position_is_negative(position)) { position = df64_add(position, Df64(limit, 0.0)); }\n\
         \x20   if (position_at_least(position, limit)) { position = df64_sub(position, Df64(limit, 0.0)); }\n\
         \x20   if (position_at_least(position, limit)) { position = df64_sub(position, Df64(limit, 0.0)); }\n\
         \x20   return position;\n}\n\n",
    );
    source.push_str(
        "@compute @workgroup_size(WORKGROUP_SIZE)\n\
         fn main(\n\
         \x20   @builtin(local_invocation_id) lid: vec3<u32>,\n\
         \x20   @builtin(workgroup_id) wid: vec3<u32>,\n\
         \x20   @builtin(num_workgroups) nwg: vec3<u32>,\n\
         ) {\n\
         \x20   let point_count = arrayLength(&points) / DIMS;\n\
         \x20   let active_batch = arrayLength(&output_values) / point_count;\n\
         \x20   let vector_tile_count = (active_batch + VECTOR_TILE - 1u) / VECTOR_TILE;\n\
         \x20   let total = point_count * vector_tile_count;\n\
         \x20   let wg_flat = (wid.z * nwg.y + wid.y) * nwg.x + wid.x;\n\
         \x20   if (wg_flat > (total - 1u) / WORKGROUP_SIZE) { return; }\n\
         \x20   let work_index = wg_flat * WORKGROUP_SIZE + lid.x;\n\
         \x20   if (work_index >= total) { return; }\n\n\
         \x20   let vector_tile_index = work_index / point_count;\n\
         \x20   let point_index = work_index - vector_tile_index * point_count;\n\
         \x20   let batch_start = vector_tile_index * VECTOR_TILE;\n\
         \x20   let point_base = point_index * DIMS;\n\n",
    );
    for axis in 0..dimensions {
        let _ = writeln!(
            source,
            "    let position{axis} = fold_position(points[point_base + {axis}u], \
             vec2<f32>(POSITION_SCALE_HI{axis}, POSITION_SCALE_LO{axis}), GRID_ORIGIN{axis}, FINE_LENGTH_F32_{axis});\n    \
             let start{axis} = ceil_df64_to_i32(df64_sub(position{axis}, Df64(HALF_WIDTH, 0.0)));\n    \
             var fine_indices{axis}: array<u32, WIDTH>;\n    \
             var weights{axis}: array<f32, WIDTH>;\n    \
             for (var offset = 0u; offset < WIDTH; offset = offset + 1u) {{\n        \
             let unwrapped = start{axis} + i32(offset);\n        \
             let distance = df64_sub(Df64(f32(unwrapped), 0.0), position{axis});\n        \
             fine_indices{axis}[offset] = wrap_index(unwrapped, FINE_LENGTH_I32_{axis});\n        \
             weights{axis}[offset] = es_weight(distance.hi + distance.lo);\n    }}"
        );
    }
    source.push_str(
        "\n    var sums: array<vec2<f32>, VECTOR_TILE>;\n\
         \x20   for (var vector = 0u; vector < VECTOR_TILE; vector = vector + 1u) {\n\
         \x20       sums[vector] = vec2<f32>(0.0, 0.0);\n\
         \x20   }\n",
    );
    // nested accumulation loops, highest axis outermost, partial products hoisted
    for axis in (0..dimensions).rev() {
        let indent = "    ".repeat(dimensions - axis);
        let _ = writeln!(
            source,
            "{indent}for (var o{axis} = 0u; o{axis} < WIDTH; o{axis} = o{axis} + 1u) {{"
        );
        let inner = "    ".repeat(dimensions - axis + 1);
        if axis == dimensions - 1 {
            let _ = writeln!(source, "{inner}let w{axis} = weights{axis}[o{axis}];");
            let _ = writeln!(
                source,
                "{inner}let c{axis} = fine_indices{axis}[o{axis}] * FINE_STRIDE{axis};"
            );
        } else {
            let upper = axis + 1;
            let _ = writeln!(
                source,
                "{inner}let w{axis} = w{upper} * weights{axis}[o{axis}];"
            );
            let _ = writeln!(
                source,
                "{inner}let c{axis} = c{upper} + fine_indices{axis}[o{axis}] * FINE_STRIDE{axis};"
            );
        }
    }
    let deepest = "    ".repeat(dimensions + 1);
    let _ = writeln!(
        source,
        "{deepest}for (var vector = 0u; vector < VECTOR_TILE; vector = vector + 1u) {{\n\
         {deepest}    let batch_index = batch_start + vector;\n\
         {deepest}    if (batch_index < active_batch) {{\n\
         {deepest}        let fine_index = batch_index * FINE_PRODUCT + c0;\n\
         {deepest}        sums[vector] = sums[vector] + fine_grid[fine_index] * w0;\n\
         {deepest}    }}\n\
         {deepest}}}"
    );
    for axis in 0..dimensions {
        let indent = "    ".repeat(dimensions - axis);
        let _ = writeln!(source, "{indent}}}");
    }
    source.push_str(
        "\n    for (var vector = 0u; vector < VECTOR_TILE; vector = vector + 1u) {\n\
         \x20       let batch_index = batch_start + vector;\n\
         \x20       if (batch_index < active_batch) {\n\
         \x20           output_values[batch_index * point_count + point_index] = sums[vector];\n\
         \x20       }\n\
         \x20   }\n}\n",
    );
    format!("{}\n{source}", wgpu_fft::kernels::DF64_WGSL)
}

fn generate_interpolation_wgsl_f64(kernel: EsKernel, fine_shape: &[usize]) -> String {
    let dimensions = fine_shape.len();
    let table = kernel.horner_table();
    let width = table.width();
    let coefficient_count = table.coefficient_count();
    let coefficient_total = table.coefficients().len();
    let coefficients = table
        .coefficients()
        .iter()
        .map(|&value| format_wgsl_f64(value))
        .collect::<Vec<_>>()
        .join(", ");
    let fine_product: usize = fine_shape.iter().product();

    let mut source = String::new();
    let _ = writeln!(source, "const WORKGROUP_SIZE: u32 = {WORKGROUP_SIZE}u;");
    let _ = writeln!(source, "const DIMS: u32 = {dimensions}u;");
    let _ = writeln!(source, "const VECTOR_TILE: u32 = {VECTOR_TILE}u;");
    let _ = writeln!(source, "const WIDTH: u32 = {width}u;");
    let _ = writeln!(
        source,
        "const HALF_WIDTH: f64 = {};",
        format_wgsl_f64(kernel.half_width())
    );
    let _ = writeln!(
        source,
        "const HORNER_COEFFICIENT_COUNT: u32 = {coefficient_count}u;"
    );
    let _ = writeln!(
        source,
        "const HORNER_COEFFICIENTS: array<f64, {coefficient_total}> =\n    \
         array<f64, {coefficient_total}>({coefficients});"
    );
    let _ = writeln!(source, "const FINE_PRODUCT: u32 = {fine_product}u;");
    let mut stride = 1usize;
    for (axis, &length) in fine_shape.iter().enumerate() {
        let _ = writeln!(source, "const FINE_LENGTH{axis}: u32 = {length}u;");
        let _ = writeln!(source, "const FINE_LENGTH_I32_{axis}: i32 = {length}i;");
        let _ = writeln!(
            source,
            "const FINE_LENGTH_F64_{axis}: f64 = {};",
            format_wgsl_f64(length as f64)
        );
        let _ = writeln!(
            source,
            "const POSITION_SCALE{axis}: f64 = {};",
            format_wgsl_f64(length as f64 / std::f64::consts::TAU)
        );
        let _ = writeln!(
            source,
            "const GRID_ORIGIN{axis}: f64 = {};",
            format_wgsl_f64((length / 2) as f64)
        );
        let _ = writeln!(source, "const FINE_STRIDE{axis}: u32 = {stride}u;");
        stride *= length;
    }
    source.push_str(
        "\n@group(0) @binding(0) var<storage, read> points: array<f64>;\n\
         @group(0) @binding(1) var<storage, read> fine_grid: array<vec2<f64>>;\n\
         @group(0) @binding(2) var<storage, read_write> output_values: array<vec2<f64>>;\n\n",
    );
    source.push_str(
        "fn ceil_f64_to_i32(value: f64) -> i32 {\n\
         \x20   let truncated = i32(value);\n\
         \x20   return truncated + select(0, 1, value > f64(truncated));\n}\n\n\
         fn es_weight(distance: f64) -> f64 {\n\
         \x20   if (abs(distance) >= HALF_WIDTH) { return 0.0lf; }\n\
         \x20   var panel = ceil_f64_to_i32(distance + HALF_WIDTH) - 1;\n\
         \x20   panel = clamp(panel, 0, i32(WIDTH) - 1);\n\
         \x20   let local = 2.0lf * (distance - f64(panel)) + f64(WIDTH - 1u);\n\
         \x20   var value = 0.0lf;\n\
         \x20   for (var coefficient = 0u; coefficient < HORNER_COEFFICIENT_COUNT; coefficient = coefficient + 1u) {\n\
         \x20       value = value * local + HORNER_COEFFICIENTS[coefficient * WIDTH + u32(panel)];\n\
         \x20   }\n\
         \x20   return value;\n}\n\n\
         fn wrap_index(index: i32, limit: i32) -> u32 {\n\
         \x20   var wrapped = index;\n\
         \x20   if (wrapped < 0) { wrapped = wrapped + limit; }\n\
         \x20   if (wrapped >= limit) { wrapped = wrapped - limit; }\n\
         \x20   return u32(wrapped);\n}\n\n\
         fn fold_position(point: f64, fine_length: f64, scale: f64, origin: f64) -> f64 {\n\
         \x20   var position = point * scale + origin;\n\
         \x20   if (position < 0.0lf) { position = position + fine_length; }\n\
         \x20   if (position < 0.0lf) { position = position + fine_length; }\n\
         \x20   if (position >= fine_length) { position = position - fine_length; }\n\
         \x20   if (position >= fine_length) { position = position - fine_length; }\n\
         \x20   return position;\n}\n\n",
    );
    source.push_str(
        "@compute @workgroup_size(WORKGROUP_SIZE)\n\
         fn main(\n\
         \x20   @builtin(local_invocation_id) lid: vec3<u32>,\n\
         \x20   @builtin(workgroup_id) wid: vec3<u32>,\n\
         \x20   @builtin(num_workgroups) nwg: vec3<u32>,\n\
         ) {\n\
         \x20   let point_count = arrayLength(&points) / DIMS;\n\
         \x20   let active_batch = arrayLength(&output_values) / point_count;\n\
         \x20   let vector_tile_count = (active_batch + VECTOR_TILE - 1u) / VECTOR_TILE;\n\
         \x20   let total = point_count * vector_tile_count;\n\
         \x20   let wg_flat = (wid.z * nwg.y + wid.y) * nwg.x + wid.x;\n\
         \x20   if (wg_flat > (total - 1u) / WORKGROUP_SIZE) { return; }\n\
         \x20   let work_index = wg_flat * WORKGROUP_SIZE + lid.x;\n\
         \x20   if (work_index >= total) { return; }\n\n\
         \x20   let vector_tile_index = work_index / point_count;\n\
         \x20   let point_index = work_index - vector_tile_index * point_count;\n\
         \x20   let batch_start = vector_tile_index * VECTOR_TILE;\n\
         \x20   let point_base = point_index * DIMS;\n\n",
    );
    for axis in 0..dimensions {
        let _ = writeln!(
            source,
            "    let position{axis} = fold_position(points[point_base + {axis}u], \
             FINE_LENGTH_F64_{axis}, POSITION_SCALE{axis}, GRID_ORIGIN{axis});\n    \
             let start{axis} = ceil_f64_to_i32(position{axis} - HALF_WIDTH);\n    \
             var fine_indices{axis}: array<u32, WIDTH>;\n    \
             var weights{axis}: array<f64, WIDTH>;\n    \
             for (var offset = 0u; offset < WIDTH; offset = offset + 1u) {{\n        \
             let unwrapped = start{axis} + i32(offset);\n        \
             fine_indices{axis}[offset] = wrap_index(unwrapped, FINE_LENGTH_I32_{axis});\n        \
             weights{axis}[offset] = es_weight(f64(unwrapped) - position{axis});\n    }}"
        );
    }
    source.push_str(
        "\n    var sums: array<vec2<f64>, VECTOR_TILE>;\n\
         \x20   for (var vector = 0u; vector < VECTOR_TILE; vector = vector + 1u) {\n\
         \x20       sums[vector] = vec2<f64>(0.0lf, 0.0lf);\n\
         \x20   }\n",
    );
    for axis in (0..dimensions).rev() {
        let indent = "    ".repeat(dimensions - axis);
        let _ = writeln!(
            source,
            "{indent}for (var o{axis} = 0u; o{axis} < WIDTH; o{axis} = o{axis} + 1u) {{"
        );
        let inner = "    ".repeat(dimensions - axis + 1);
        if axis == dimensions - 1 {
            let _ = writeln!(source, "{inner}let w{axis} = weights{axis}[o{axis}];");
            let _ = writeln!(
                source,
                "{inner}let c{axis} = fine_indices{axis}[o{axis}] * FINE_STRIDE{axis};"
            );
        } else {
            let upper = axis + 1;
            let _ = writeln!(
                source,
                "{inner}let w{axis} = w{upper} * weights{axis}[o{axis}];"
            );
            let _ = writeln!(
                source,
                "{inner}let c{axis} = c{upper} + fine_indices{axis}[o{axis}] * FINE_STRIDE{axis};"
            );
        }
    }
    let deepest = "    ".repeat(dimensions + 1);
    let _ = writeln!(
        source,
        "{deepest}for (var vector = 0u; vector < VECTOR_TILE; vector = vector + 1u) {{\n\
         {deepest}    let batch_index = batch_start + vector;\n\
         {deepest}    if (batch_index < active_batch) {{\n\
         {deepest}        let fine_index = batch_index * FINE_PRODUCT + c0;\n\
         {deepest}        sums[vector] = sums[vector] + fine_grid[fine_index] * w0;\n\
         {deepest}    }}\n\
         {deepest}}}"
    );
    for axis in 0..dimensions {
        let indent = "    ".repeat(dimensions - axis);
        let _ = writeln!(source, "{indent}}}");
    }
    source.push_str(
        "\n    for (var vector = 0u; vector < VECTOR_TILE; vector = vector + 1u) {\n\
         \x20       let batch_index = batch_start + vector;\n\
         \x20       if (batch_index < active_batch) {\n\
         \x20           output_values[batch_index * point_count + point_index] = sums[vector];\n\
         \x20       }\n\
         \x20   }\n}\n",
    );
    source
}

fn generate_interpolation_wgsl_df64(kernel: EsKernel, fine_shape: &[usize]) -> String {
    let dimensions = fine_shape.len();
    let table = kernel.horner_table();
    let width = table.width();
    let coefficient_count = table.coefficient_count();
    let coefficient_total = table.coefficients().len();
    let coefficients = table
        .coefficients()
        .iter()
        .map(|&value| format_wgsl_df64_pair(value))
        .collect::<Vec<_>>()
        .join(", ");
    let fine_product: usize = fine_shape.iter().product();

    let mut source = String::new();
    let _ = writeln!(source, "const WORKGROUP_SIZE: u32 = {WORKGROUP_SIZE}u;");
    let _ = writeln!(source, "const DIMS: u32 = {dimensions}u;");
    let _ = writeln!(source, "const VECTOR_TILE: u32 = {VECTOR_TILE}u;");
    let _ = writeln!(source, "const WIDTH: u32 = {width}u;");
    let _ = writeln!(
        source,
        "const HALF_WIDTH: vec2<f32> = {};",
        format_wgsl_df64_pair(kernel.half_width())
    );
    let _ = writeln!(
        source,
        "const HORNER_COEFFICIENT_COUNT: u32 = {coefficient_count}u;"
    );
    let _ = writeln!(
        source,
        "const HORNER_COEFFICIENTS: array<vec2<f32>, {coefficient_total}> =\n    \
         array<vec2<f32>, {coefficient_total}>({coefficients});"
    );
    let _ = writeln!(source, "const FINE_PRODUCT: u32 = {fine_product}u;");
    let mut stride = 1usize;
    for (axis, &length) in fine_shape.iter().enumerate() {
        let _ = writeln!(source, "const FINE_LENGTH{axis}: u32 = {length}u;");
        let _ = writeln!(source, "const FINE_LENGTH_I32_{axis}: i32 = {length}i;");
        let _ = writeln!(
            source,
            "const FINE_DF64_{axis}: vec2<f32> = {};",
            format_wgsl_df64_pair(length as f64)
        );
        let _ = writeln!(
            source,
            "const POSITION_SCALE{axis}: vec2<f32> = {};",
            format_wgsl_df64_pair(length as f64 / std::f64::consts::TAU)
        );
        let _ = writeln!(
            source,
            "const GRID_ORIGIN{axis}: vec2<f32> = {};",
            format_wgsl_df64_pair((length / 2) as f64)
        );
        let _ = writeln!(source, "const FINE_STRIDE{axis}: u32 = {stride}u;");
        stride *= length;
    }
    source.push_str(
        "\n@group(0) @binding(0) var<storage, read> points: array<vec2<f32>>;\n\
         @group(0) @binding(1) var<storage, read> fine_grid: array<vec4<f32>>;\n\
         @group(0) @binding(2) var<storage, read_write> output_values: array<vec4<f32>>;\n\n",
    );
    source.push_str(
        "fn load_df64(value: vec2<f32>) -> Df64 {\n\
         \x20   return Df64(value.x, value.y);\n}\n\n\
         fn store_df64(value: Df64) -> vec2<f32> {\n\
         \x20   return vec2<f32>(value.hi, value.lo);\n}\n\n\
         fn df64_less(a: Df64, b: Df64) -> bool {\n\
         \x20   let difference = df64_sub(a, b);\n\
         \x20   return difference.hi < 0.0 || (difference.hi == 0.0 && difference.lo < 0.0);\n}\n\n\
         fn df64_at_least(a: Df64, b: Df64) -> bool {\n\
         \x20   return !df64_less(a, b);\n}\n\n\
         fn df64_abs_value(value: Df64) -> Df64 {\n\
         \x20   if (df64_less(value, Df64(0.0, 0.0))) {\n\
         \x20       return df64_neg(value);\n\
         \x20   }\n\
         \x20   return value;\n}\n\n\
         fn ceil_df64_to_i32(value: Df64) -> i32 {\n\
         \x20   let base = floor(value.hi);\n\
         \x20   let remainder = df64_sub(value, Df64(base, 0.0));\n\
         \x20   let has_positive_remainder = remainder.hi > 0.0 ||\n\
         \x20       (remainder.hi == 0.0 && remainder.lo > 0.0);\n\
         \x20   return i32(base) + select(0, 1, has_positive_remainder);\n}\n\n\
         fn es_weight(distance: Df64) -> Df64 {\n\
         \x20   let half_width = load_df64(HALF_WIDTH);\n\
         \x20   if (df64_at_least(df64_abs_value(distance), half_width)) {\n\
         \x20       return Df64(0.0, 0.0);\n\
         \x20   }\n\
         \x20   var panel = ceil_df64_to_i32(df64_add(distance, half_width)) - 1;\n\
         \x20   panel = clamp(panel, 0, i32(WIDTH) - 1);\n\
         \x20   let local = df64_add(\n\
         \x20       df64_mul(Df64(2.0, 0.0), df64_sub(distance, Df64(f32(panel), 0.0))),\n\
         \x20       Df64(f32(WIDTH - 1u), 0.0),\n\
         \x20   );\n\
         \x20   var value = Df64(0.0, 0.0);\n\
         \x20   for (var coefficient = 0u; coefficient < HORNER_COEFFICIENT_COUNT; coefficient = coefficient + 1u) {\n\
         \x20       let packed = HORNER_COEFFICIENTS[coefficient * WIDTH + u32(panel)];\n\
         \x20       value = df64_add(df64_mul(value, local), load_df64(packed));\n\
         \x20   }\n\
         \x20   return value;\n}\n\n\
         fn wrap_index(index: i32, limit: i32) -> u32 {\n\
         \x20   var wrapped = index;\n\
         \x20   if (wrapped < 0) { wrapped = wrapped + limit; }\n\
         \x20   if (wrapped >= limit) { wrapped = wrapped - limit; }\n\
         \x20   return u32(wrapped);\n}\n\n\
         fn fold_position(\n\
         \x20   point: vec2<f32>,\n\
         \x20   fine_length_value: vec2<f32>,\n\
         \x20   scale: vec2<f32>,\n\
         \x20   origin: vec2<f32>,\n\
         ) -> Df64 {\n\
         \x20   var position = df64_add(\n\
         \x20       df64_mul(load_df64(point), load_df64(scale)),\n\
         \x20       load_df64(origin),\n\
         \x20   );\n\
         \x20   let fine_length = load_df64(fine_length_value);\n\
         \x20   if (df64_less(position, Df64(0.0, 0.0))) {\n\
         \x20       position = df64_add(position, fine_length);\n\
         \x20   }\n\
         \x20   if (df64_less(position, Df64(0.0, 0.0))) {\n\
         \x20       position = df64_add(position, fine_length);\n\
         \x20   }\n\
         \x20   if (df64_at_least(position, fine_length)) {\n\
         \x20       position = df64_sub(position, fine_length);\n\
         \x20   }\n\
         \x20   if (df64_at_least(position, fine_length)) {\n\
         \x20       position = df64_sub(position, fine_length);\n\
         \x20   }\n\
         \x20   return position;\n}\n\n",
    );
    source.push_str(
        "@compute @workgroup_size(WORKGROUP_SIZE)\n\
         fn main(\n\
         \x20   @builtin(local_invocation_id) lid: vec3<u32>,\n\
         \x20   @builtin(workgroup_id) wid: vec3<u32>,\n\
         \x20   @builtin(num_workgroups) nwg: vec3<u32>,\n\
         ) {\n\
         \x20   let point_count = arrayLength(&points) / DIMS;\n\
         \x20   let active_batch = arrayLength(&output_values) / point_count;\n\
         \x20   let vector_tile_count = (active_batch + VECTOR_TILE - 1u) / VECTOR_TILE;\n\
         \x20   let total = point_count * vector_tile_count;\n\
         \x20   let wg_flat = (wid.z * nwg.y + wid.y) * nwg.x + wid.x;\n\
         \x20   if (wg_flat > (total - 1u) / WORKGROUP_SIZE) { return; }\n\
         \x20   let work_index = wg_flat * WORKGROUP_SIZE + lid.x;\n\
         \x20   if (work_index >= total) { return; }\n\n\
         \x20   let vector_tile_index = work_index / point_count;\n\
         \x20   let point_index = work_index - vector_tile_index * point_count;\n\
         \x20   let batch_start = vector_tile_index * VECTOR_TILE;\n\
         \x20   let point_base = point_index * DIMS;\n\n",
    );
    for axis in 0..dimensions {
        let _ = writeln!(
            source,
            "    let position{axis} = fold_position(points[point_base + {axis}u], \
             FINE_DF64_{axis}, POSITION_SCALE{axis}, GRID_ORIGIN{axis});\n    \
             let start{axis} = ceil_df64_to_i32(df64_sub(position{axis}, load_df64(HALF_WIDTH)));\n    \
             var fine_indices{axis}: array<u32, WIDTH>;\n    \
             var weights{axis}: array<vec2<f32>, WIDTH>;\n    \
             for (var offset = 0u; offset < WIDTH; offset = offset + 1u) {{\n        \
             let unwrapped = start{axis} + i32(offset);\n        \
             let distance = df64_sub(Df64(f32(unwrapped), 0.0), position{axis});\n        \
             fine_indices{axis}[offset] = wrap_index(unwrapped, FINE_LENGTH_I32_{axis});\n        \
             weights{axis}[offset] = store_df64(es_weight(distance));\n    }}"
        );
    }
    source.push_str(
        "\n    var sums: array<vec4<f32>, VECTOR_TILE>;\n\
         \x20   for (var vector = 0u; vector < VECTOR_TILE; vector = vector + 1u) {\n\
         \x20       sums[vector] = vec4<f32>(0.0, 0.0, 0.0, 0.0);\n\
         \x20   }\n",
    );
    for axis in (0..dimensions).rev() {
        let indent = "    ".repeat(dimensions - axis);
        let _ = writeln!(
            source,
            "{indent}for (var o{axis} = 0u; o{axis} < WIDTH; o{axis} = o{axis} + 1u) {{"
        );
        let inner = "    ".repeat(dimensions - axis + 1);
        if axis == dimensions - 1 {
            let _ = writeln!(
                source,
                "{inner}let w{axis} = load_df64(weights{axis}[o{axis}]);"
            );
            let _ = writeln!(
                source,
                "{inner}let c{axis} = fine_indices{axis}[o{axis}] * FINE_STRIDE{axis};"
            );
        } else {
            let upper = axis + 1;
            let _ = writeln!(
                source,
                "{inner}let w{axis} = df64_mul(load_df64(weights{axis}[o{axis}]), w{upper});"
            );
            let _ = writeln!(
                source,
                "{inner}let c{axis} = c{upper} + fine_indices{axis}[o{axis}] * FINE_STRIDE{axis};"
            );
        }
    }
    let deepest = "    ".repeat(dimensions + 1);
    let _ = writeln!(
        source,
        "{deepest}for (var vector = 0u; vector < VECTOR_TILE; vector = vector + 1u) {{\n\
         {deepest}    let batch_index = batch_start + vector;\n\
         {deepest}    if (batch_index < active_batch) {{\n\
         {deepest}        let fine_index = batch_index * FINE_PRODUCT + c0;\n\
         {deepest}        sums[vector] = df64_complex_add(\n\
         {deepest}            sums[vector],\n\
         {deepest}            df64_complex_scale(fine_grid[fine_index], w0),\n\
         {deepest}        );\n\
         {deepest}    }}\n\
         {deepest}}}"
    );
    for axis in 0..dimensions {
        let indent = "    ".repeat(dimensions - axis);
        let _ = writeln!(source, "{indent}}}");
    }
    source.push_str(
        "\n    for (var vector = 0u; vector < VECTOR_TILE; vector = vector + 1u) {\n\
         \x20       let batch_index = batch_start + vector;\n\
         \x20       if (batch_index < active_batch) {\n\
         \x20           output_values[batch_index * point_count + point_index] = sums[vector];\n\
         \x20       }\n\
         \x20   }\n}\n",
    );
    format!("{}\n{source}", wgpu_fft::kernels::DF64_WGSL)
}

pub(crate) fn format_wgsl_f32(value: f32) -> String {
    debug_assert!(value.is_finite());
    let mut formatted = value.to_string();
    if !formatted.contains('.') && !formatted.contains('e') && !formatted.contains('E') {
        formatted.push_str(".0");
    }
    formatted
}

pub(crate) fn format_wgsl_f64(value: f64) -> String {
    debug_assert!(value.is_finite());
    let mut formatted = value.to_string();
    if !formatted.contains('.') && !formatted.contains('e') && !formatted.contains('E') {
        formatted.push_str(".0");
    }
    formatted.push_str("lf");
    formatted
}

pub(crate) fn format_wgsl_df64_pair(value: f64) -> String {
    let value = DoubleFloat::from_f64(value);
    format!(
        "vec2<f32>({}, {})",
        format_wgsl_f32(value.hi),
        format_wgsl_f32(value.lo)
    )
}

pub(crate) fn checked_buffer_size(
    context: &'static str,
    elements: usize,
    bytes_per_element: u64,
) -> Result<u64> {
    u64::try_from(elements)
        .ok()
        .and_then(|count| count.checked_mul(bytes_per_element))
        .ok_or(NufftError::LengthOverflow { context })
}

pub(crate) fn checked_product(context: &'static str, left: usize, right: usize) -> Result<usize> {
    left.checked_mul(right)
        .ok_or(NufftError::LengthOverflow { context })
}

pub(crate) fn validate_binding_limit(buffer: &'static str, bytes: u64, limit: u64) -> Result<()> {
    if bytes > limit {
        Err(NufftError::GpuBufferBindingTooLarge {
            buffer,
            required_bytes: bytes,
            limit_bytes: limit,
        })
    } else {
        Ok(())
    }
}

pub(crate) fn validate_external_storage_buffer(
    label: &'static str,
    buffer: &wgpu::Buffer,
    required_bytes: u64,
    max_storage_binding_bytes: u64,
) -> Result<()> {
    if !buffer.usage().contains(wgpu::BufferUsages::STORAGE) {
        return Err(NufftError::GpuBufferMissingUsage {
            buffer: label,
            required_usage: "STORAGE",
        });
    }
    if buffer.size() < required_bytes {
        return Err(NufftError::GpuBufferTooSmall {
            buffer: label,
            required_bytes,
            actual_bytes: buffer.size(),
        });
    }
    validate_binding_limit(label, required_bytes, max_storage_binding_bytes)
}

pub(crate) fn binding_entry<'a>(
    binding: u32,
    buffer: &'a wgpu::Buffer,
    size: u64,
) -> wgpu::BindGroupEntry<'a> {
    wgpu::BindGroupEntry {
        binding,
        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer,
            offset: 0,
            size: NonZeroU64::new(size),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nd_predeconvolution_shader_maps_axes_and_zeros_holes() {
        let config = NufftConfig::new([5, 4, 3, 3], 1.0e-3);
        let fine_shape = [12, 10, 8, 8];
        let source = generate_predeconvolution_wgsl(&config, &fine_shape);
        assert!(source.contains("fine_grid[index] = vec2<f32>(0.0, 0.0);"));
        assert!(source.contains("const MODE_COUNT: u32 = 180u;"));
        assert!(source.contains("const FINE3: u32 = 8u;"));
        assert!(source.contains("let mapped3 = map_axis(fine_index3"));
        assert!(source.contains(
            "let source_index = u32(mapped0) + MODE0 * (u32(mapped1) + MODE1 * (u32(mapped2) + MODE2 * (u32(mapped3))));"
        ));
    }

    #[test]
    fn nd_interpolation_shader_uses_df64_folding_and_nested_gather() {
        let kernel = EsKernel::for_tolerance(1.0e-3, 2.0).unwrap();
        let source = generate_interpolation_wgsl(kernel, &[12, 10, 8, 8]);
        assert!(source.contains("fn ceil_df64_to_i32(value: Df64) -> i32"));
        assert!(source.contains("let start3 = ceil_df64_to_i32("));
        assert!(source.contains("const FINE_STRIDE3: u32 = 960u;"));
        assert!(source.contains("let fine_index = batch_index * FINE_PRODUCT + c0;"));
        assert!(source.contains("let w0 = w1 * weights0[o0];"));
    }

    #[test]
    fn nd_predeconvolution_f64_lowering_replaces_types_only() {
        let config =
            NufftConfig::new([5, 4, 3, 3], 1.0e-3).with_precision(wgpu_fft::FftPrecision::F64);
        let source = generate_predeconvolution_wgsl_for_precision(&config, &[12, 10, 8, 8]);
        assert!(source.contains("array<vec2<f64>>"));
        assert!(source.contains("array<f64>"));
        assert!(source.contains("fine_grid[index] = vec2<f64>(0.0, 0.0);"));
        assert!(!source.contains("vec2<f32>"));
        assert!(!source.contains("enable f64"));
    }

    #[test]
    fn nd_predeconvolution_df64_lowering_uses_complex_scale() {
        let config =
            NufftConfig::new([5, 4, 3, 3], 1.0e-3).with_precision(wgpu_fft::FftPrecision::Df64);
        let source = generate_predeconvolution_wgsl_for_precision(&config, &[12, 10, 8, 8]);
        assert!(source.contains("array<vec4<f32>>"));
        assert!(source.contains("var<storage, read> amplitudes: array<vec2<f32>>;"));
        assert!(source.contains("fine_grid[index] = vec4<f32>(0.0, 0.0, 0.0, 0.0);"));
        assert!(source.contains("df64_complex_scale"));
        assert!(source.contains("struct Df64"));
    }

    #[test]
    fn nd_interpolation_f64_uses_horner_and_native_folding() {
        let kernel = EsKernel::for_tolerance(1.0e-6, 2.0).unwrap();
        let source = generate_interpolation_wgsl_f64(kernel, &[12, 10, 8, 8]);
        assert!(source.contains("const HALF_WIDTH: f64"));
        assert!(source.contains("HORNER_COEFFICIENTS: array<f64,"));
        assert!(source.contains("fn fold_position(point: f64"));
        assert!(source.contains("let start3 = ceil_f64_to_i32(position3 - HALF_WIDTH);"));
        assert!(source.contains("let w0 = w1 * weights0[o0];"));
        assert!(source.contains("array<vec2<f64>>"));
        assert!(!source.contains("exp("));
        assert!(!source.contains("enable f64"));
    }

    #[test]
    fn nd_interpolation_df64_uses_horner_and_complex_accumulation() {
        let kernel = EsKernel::for_tolerance(1.0e-6, 2.0).unwrap();
        let source = generate_interpolation_wgsl_df64(kernel, &[12, 10, 8, 8]);
        assert!(source.contains("HORNER_COEFFICIENTS: array<vec2<f32>,"));
        assert!(source.contains("weights3[offset] = store_df64(es_weight(distance));"));
        assert!(source.contains("let w0 = df64_mul(load_df64(weights0[o0]), w1);"));
        assert!(source.contains("df64_complex_scale(fine_grid[fine_index], w0)"));
        assert!(source.contains("struct Df64"));
        assert!(!source.contains("exp("));
    }

    #[test]
    fn nd_amplitudes_f64_match_f32_products() {
        let config = NufftConfig::new([3, 2], 1.0e-3);
        let coefficients = vec![vec![2.0, 4.0], vec![5.0, 8.0]];
        let single = mode_amplitudes_nd(&config, &coefficients).unwrap();
        let double = mode_amplitudes_nd_f64(&config, &coefficients).unwrap();
        assert_eq!(single.len(), double.len());
        for (&narrow, &wide) in single.iter().zip(&double) {
            assert!((f64::from(narrow) - wide).abs() <= f64::from(f32::EPSILON) * 4.0);
        }
    }

    #[test]
    fn nd_amplitudes_are_per_axis_products() {
        let config = NufftConfig::new([3, 2], 1.0e-3);
        let coefficients = vec![vec![2.0, 4.0], vec![5.0, 8.0]];
        let amplitudes = mode_amplitudes_nd(&config, &coefficients).unwrap();
        assert_eq!(amplitudes.len(), 6);
        // centered order, axis-0-fast: k0 in {-1,0,1}, k1 in {-1,0}
        // amplitude(flat = i0 + 3*i1) = 1/(c0[|k0|] * c1[|k1|])
        let expected = [
            1.0 / (4.0 * 8.0), // k0=-1, k1=-1
            1.0 / (2.0 * 8.0), // k0=0,  k1=-1
            1.0 / (4.0 * 8.0), // k0=1,  k1=-1
            1.0 / (4.0 * 5.0), // k0=-1, k1=0
            1.0 / (2.0 * 5.0), // k0=0,  k1=0
            1.0 / (4.0 * 5.0), // k0=1,  k1=0
        ];
        for (actual, expected) in amplitudes.iter().zip(expected) {
            assert!((actual - expected as f32).abs() <= f32::EPSILON * 4.0);
        }
    }
}
