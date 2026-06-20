use std::num::NonZeroU64;

use wgpu::util::DeviceExt;
use wgpu_fft::math::DoubleFloat;
use wgpu_fft::{BufferView, FftConfig, FftDirection, FftPlan, FftPrecision, Normalization};

use crate::config::{ModeOrder, NufftConfig, NufftSign};
use crate::direct::mode_for_storage_index;
use crate::error::{NufftError, Result};
use crate::gpu_dispatch::split_workgroups;
#[cfg(feature = "gpu-profiling")]
use crate::gpu_profile::{GpuProfileQueryWriter, NufftGpuProfileLayout};
use crate::kernel::EsKernel;

const WORKGROUP_SIZE: u32 = 64;
const VECTOR_TILE: usize = 4;
const DIMENSIONS: usize = 2;

/// Device-specific state for a two-dimensional type-2 NUFFT.
///
/// Both mode and fine-grid buffers use axis-0-fast linearization. Point
/// coordinates are caller-owned AoS pairs `[x0, y0, x1, y1, ...]`.
pub(crate) struct Type2GpuPlan2d {
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
    precision: FftPrecision,
    max_storage_binding_bytes: u64,
}

impl Type2GpuPlan2d {
    pub(crate) fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        config: &NufftConfig,
        kernel: EsKernel,
        fine_shape: &[usize],
        kernel_fourier_coefficients: &[Vec<f64>],
    ) -> Result<Self> {
        if config.dimensions() != DIMENSIONS {
            return Err(NufftError::GpuDimensionsUnsupported {
                kind: "type-2",
                actual: config.dimensions(),
                supported: DIMENSIONS,
            });
        }
        if fine_shape.len() != DIMENSIONS {
            return Err(NufftError::GpuDimensionsUnsupported {
                kind: "type-2 fine grid",
                actual: fine_shape.len(),
                supported: DIMENSIONS,
            });
        }
        if kernel_fourier_coefficients.len() != DIMENSIONS {
            return Err(NufftError::InputLength {
                input: "type-2 per-axis kernel Fourier coefficient tables",
                expected: DIMENSIONS,
                actual: kernel_fourier_coefficients.len(),
            });
        }

        let precision = config.precision();
        let fft_direction = match config.sign() {
            NufftSign::Positive => FftDirection::Inverse,
            NufftSign::Negative => FftDirection::Forward,
        };
        let fft_config = FftConfig::new_nd(fine_shape.to_vec())
            .with_direction(fft_direction)
            .with_normalization(Normalization::None)
            .with_batch(config.batch())
            .with_precision(precision);
        fft_config
            .validate()
            .map_err(|source| NufftError::FftShapeUnsupported {
                stage: "type-2 oversampled-grid C2C plan",
                source,
            })?;

        // F32 and df64 convert integer support indices through an f32 high
        // word. Native f64 retains adjacent cells throughout the signed index
        // range.
        let maximum_signed_length = match precision {
            FftPrecision::F64 => i32::MAX as usize - kernel.width(),
            FftPrecision::F32 | FftPrecision::Df64 => (1usize << f32::MANTISSA_DIGITS)
                .saturating_sub(kernel.width())
                .min(i32::MAX as usize - kernel.width()),
        };
        for (axis, &length) in fine_shape.iter().enumerate() {
            if !length.is_multiple_of(2) {
                return Err(NufftError::FineGridMustBeEven { length });
            }
            if length < config.n_modes()[axis] {
                let input = match axis {
                    0 => "type-2 fine-grid axis-0 length",
                    _ => "type-2 fine-grid axis-1 length",
                };
                return Err(NufftError::InputLength {
                    input,
                    expected: config.n_modes()[axis],
                    actual: length,
                });
            }
            if length > maximum_signed_length {
                return Err(NufftError::GpuFineGridIndexUnsupported {
                    length,
                    maximum: maximum_signed_length,
                });
            }
            u32::try_from(length).map_err(|_| NufftError::LengthOverflow {
                context: "type-2 fine-grid axis length",
            })?;
        }

        let limits = device.limits();
        let maximum_workgroup_size = maximum_supported_workgroup_size(&limits);
        if maximum_workgroup_size < WORKGROUP_SIZE {
            return Err(NufftError::GpuWorkgroupSizeUnsupported {
                requested: WORKGROUP_SIZE,
                maximum: maximum_workgroup_size,
            });
        }

        let batch_capacity = config.batch();
        let mode_count = config.mode_count()?;
        let fine_count = checked_product(
            "type-2 two-dimensional fine-grid element count",
            fine_shape[0],
            fine_shape[1],
        )?;
        let fine_element_count = checked_product(
            "type-2 two-dimensional batched fine-grid element count",
            fine_count,
            batch_capacity,
        )?;
        let fine_element_count_u32 =
            u32::try_from(fine_element_count).map_err(|_| NufftError::LengthOverflow {
                context: "type-2 two-dimensional batched fine-grid shader index space",
            })?;
        u32::try_from(mode_count).map_err(|_| NufftError::LengthOverflow {
            context: "type-2 two-dimensional mode shader index space",
        })?;

        let fine_bytes = Self::complex_buffer_size_bytes_for_precision(
            "type-2 fine grid",
            fine_element_count,
            precision,
        )?;
        let amplitude_bytes = checked_buffer_size(
            "type-2 deconvolution amplitudes",
            mode_count,
            precision.scalar_size_bytes(),
        )?;
        let max_storage_binding_bytes = limits.max_storage_buffer_binding_size;
        validate_internal_buffer_limits(
            "type-2 fine grid",
            fine_bytes,
            max_storage_binding_bytes,
            limits.max_buffer_size,
        )?;
        validate_internal_buffer_limits(
            "type-2 deconvolution amplitudes",
            amplitude_bytes,
            max_storage_binding_bytes,
            limits.max_buffer_size,
        )?;

        let amplitude_buffer = match precision {
            FftPrecision::F64 => {
                let amplitudes =
                    mode_amplitudes_2d_f64(config, fine_shape, kernel_fourier_coefficients)?;
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("wgpu_nufft.type2_2d.deconvolution_amplitudes"),
                    contents: bytemuck::cast_slice(&amplitudes),
                    usage: wgpu::BufferUsages::STORAGE,
                })
            }
            FftPrecision::F32 => {
                let amplitudes =
                    mode_amplitudes_2d(config, fine_shape, kernel_fourier_coefficients)?;
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("wgpu_nufft.type2_2d.deconvolution_amplitudes"),
                    contents: bytemuck::cast_slice(&amplitudes),
                    usage: wgpu::BufferUsages::STORAGE,
                })
            }
            FftPrecision::Df64 => {
                let amplitudes =
                    mode_amplitudes_2d_df64(config, fine_shape, kernel_fourier_coefficients)?;
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("wgpu_nufft.type2_2d.deconvolution_amplitudes"),
                    contents: bytemuck::cast_slice(&amplitudes),
                    usage: wgpu::BufferUsages::STORAGE,
                })
            }
        };
        let fine_usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST;
        let fine_input = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("wgpu_nufft.type2_2d.fine_input"),
            size: fine_bytes,
            usage: fine_usage,
            mapped_at_creation: false,
        });
        let fine_output = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("wgpu_nufft.type2_2d.fine_output"),
            size: fine_bytes,
            usage: fine_usage,
            mapped_at_creation: false,
        });

        let fft = FftPlan::c2c(device, queue, fft_config).map_err(|source| {
            NufftError::FftShapeUnsupported {
                stage: "type-2 oversampled-grid C2C plan",
                source,
            }
        })?;

        let predeconvolution_pipeline = create_compute_pipeline(
            device,
            "wgpu_nufft.type2_2d.predeconvolution",
            &generate_predeconvolution_wgsl(config, fine_shape),
        );
        let predeconvolution_layout = predeconvolution_pipeline.get_bind_group_layout(0);
        let interpolation_pipeline = create_compute_pipeline(
            device,
            "wgpu_nufft.type2_2d.interpolation",
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
            precision,
            max_storage_binding_bytes,
        })
    }

    pub(crate) fn point_buffer_size_bytes_for_precision(
        point_count: usize,
        precision: FftPrecision,
    ) -> Result<u64> {
        let coordinate_count =
            point_count
                .checked_mul(DIMENSIONS)
                .ok_or(NufftError::LengthOverflow {
                    context: "type-2 two-dimensional point-coordinate count",
                })?;
        checked_buffer_size(
            "type-2 point buffer",
            coordinate_count,
            precision.scalar_size_bytes(),
        )
    }

    pub(crate) fn complex_buffer_size_bytes_for_precision(
        buffer: &'static str,
        element_count: usize,
        precision: FftPrecision,
    ) -> Result<u64> {
        checked_buffer_size(buffer, element_count, precision.complex_size_bytes())
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
                kind: "type-2 stage profiling",
                reason: "at least one point is required",
            });
        }
        let layout =
            NufftGpuProfileLayout::type2(first_query).map_err(|_| NufftError::LengthOverflow {
                context: "type-2 stage-profile query range",
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
            context: "type-2 GPU point count",
        })?;
        u32::try_from(
            point_count
                .checked_mul(DIMENSIONS)
                .ok_or(NufftError::LengthOverflow {
                    context: "type-2 two-dimensional point-coordinate count",
                })?,
        )
        .map_err(|_| NufftError::LengthOverflow {
            context: "type-2 two-dimensional point-coordinate shader index space",
        })?;
        let point_bytes = Self::point_buffer_size_bytes_for_precision(point_count, self.precision)?;
        let coefficient_elements = checked_product(
            "type-2 two-dimensional batched Fourier coefficient count",
            active_batch,
            self.mode_count,
        )?;
        let output_elements = checked_product(
            "type-2 two-dimensional batched output element count",
            active_batch,
            point_count,
        )?;
        let coefficient_bytes = Self::complex_buffer_size_bytes_for_precision(
            "type-2 Fourier coefficient buffer",
            coefficient_elements,
            self.precision,
        )?;
        let output_bytes = Self::complex_buffer_size_bytes_for_precision(
            "type-2 output buffer",
            output_elements,
            self.precision,
        )?;
        validate_external_storage_buffer(
            "type-2 point",
            points,
            point_bytes,
            self.max_storage_binding_bytes,
        )?;
        validate_external_storage_buffer(
            "type-2 Fourier coefficient",
            coefficients,
            coefficient_bytes,
            self.max_storage_binding_bytes,
        )?;
        validate_external_storage_buffer(
            "type-2 output",
            output,
            output_bytes,
            self.max_storage_binding_bytes,
        )?;
        if points == output {
            return Err(NufftError::GpuBufferAliasUnsupported {
                first: "type-2 point",
                second: "type-2 output",
            });
        }
        let vector_tiles = active_batch.div_ceil(VECTOR_TILE);
        let interpolation_elements = checked_product(
            "type-2 two-dimensional batched interpolation work item count",
            point_count,
            vector_tiles,
        )?;
        let interpolation_elements_u32 =
            u32::try_from(interpolation_elements).map_err(|_| NufftError::LengthOverflow {
                context: "type-2 two-dimensional batched interpolation shader index space",
            })?;
        let interpolation_dispatch = split_workgroups(
            interpolation_elements_u32.div_ceil(WORKGROUP_SIZE),
            self.max_workgroups_per_dimension,
        )?;

        let predeconvolution_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("wgpu_nufft.type2_2d.predeconvolution.bind_group"),
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
                label: Some("wgpu_nufft.type2_2d.predeconvolution.pass"),
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
                stage: "type-2 two-dimensional oversampled-grid C2C transform",
                source,
            })?;

        let interpolation_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("wgpu_nufft.type2_2d.interpolation.bind_group"),
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
                label: Some("wgpu_nufft.type2_2d.interpolation.pass"),
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

fn mode_amplitudes_2d(
    config: &NufftConfig,
    fine_shape: &[usize],
    coefficients: &[Vec<f64>],
) -> Result<Vec<f32>> {
    for axis in 0..DIMENSIONS {
        let expected = fine_shape[axis] / 2 + 1;
        if coefficients[axis].len() != expected {
            let input = match axis {
                0 => "axis-0 centered kernel Fourier coefficients",
                _ => "axis-1 centered kernel Fourier coefficients",
            };
            return Err(NufftError::InputLength {
                input,
                expected,
                actual: coefficients[axis].len(),
            });
        }
    }

    let mode_shape = config.n_modes();
    for axis in 0..DIMENSIONS {
        let required = mode_shape[axis] / 2 + 1;
        if coefficients[axis].len() < required {
            let input = match axis {
                0 => "axis-0 centered kernel Fourier coefficients",
                _ => "axis-1 centered kernel Fourier coefficients",
            };
            return Err(NufftError::InputLength {
                input,
                expected: required,
                actual: coefficients[axis].len(),
            });
        }
    }
    let mode_count = config.mode_count()?;
    let mut amplitudes = Vec::new();
    amplitudes
        .try_reserve_exact(mode_count)
        .map_err(|_| NufftError::HostAllocationFailed {
            buffer: "type-2 two-dimensional deconvolution amplitudes",
            elements: mode_count,
        })?;

    for index1 in 0..mode_shape[1] {
        let mode1 = mode_for_storage_index(config.mode_order(), index1, mode_shape[1]);
        let coefficient1 = coefficients[1][mode1.unsigned_abs() as usize];
        validate_kernel_coefficient(mode1, coefficient1)?;
        for index0 in 0..mode_shape[0] {
            let mode0 = mode_for_storage_index(config.mode_order(), index0, mode_shape[0]);
            let coefficient0 = coefficients[0][mode0.unsigned_abs() as usize];
            validate_kernel_coefficient(mode0, coefficient0)?;
            let denominator = coefficient0 * coefficient1;
            let amplitude = (1.0 / denominator) as f32;
            if !denominator.is_finite() || denominator == 0.0 || !amplitude.is_finite() {
                return Err(NufftError::InvalidKernelFourierCoefficient {
                    mode: mode0,
                    value: denominator,
                });
            }
            amplitudes.push(amplitude);
        }
    }
    Ok(amplitudes)
}

fn mode_amplitudes_2d_f64(
    config: &NufftConfig,
    fine_shape: &[usize],
    coefficients: &[Vec<f64>],
) -> Result<Vec<f64>> {
    for axis in 0..DIMENSIONS {
        let expected = fine_shape[axis] / 2 + 1;
        if coefficients[axis].len() != expected {
            let input = match axis {
                0 => "axis-0 centered kernel Fourier coefficients",
                _ => "axis-1 centered kernel Fourier coefficients",
            };
            return Err(NufftError::InputLength {
                input,
                expected,
                actual: coefficients[axis].len(),
            });
        }
    }

    let mode_shape = config.n_modes();
    for axis in 0..DIMENSIONS {
        let required = mode_shape[axis] / 2 + 1;
        if coefficients[axis].len() < required {
            let input = match axis {
                0 => "axis-0 centered kernel Fourier coefficients",
                _ => "axis-1 centered kernel Fourier coefficients",
            };
            return Err(NufftError::InputLength {
                input,
                expected: required,
                actual: coefficients[axis].len(),
            });
        }
    }
    let mode_count = config.mode_count()?;
    let mut amplitudes = Vec::new();
    amplitudes
        .try_reserve_exact(mode_count)
        .map_err(|_| NufftError::HostAllocationFailed {
            buffer: "type-2 two-dimensional deconvolution amplitudes",
            elements: mode_count,
        })?;

    for index1 in 0..mode_shape[1] {
        let mode1 = mode_for_storage_index(config.mode_order(), index1, mode_shape[1]);
        let coefficient1 = coefficients[1][mode1.unsigned_abs() as usize];
        validate_kernel_coefficient(mode1, coefficient1)?;
        for index0 in 0..mode_shape[0] {
            let mode0 = mode_for_storage_index(config.mode_order(), index0, mode_shape[0]);
            let coefficient0 = coefficients[0][mode0.unsigned_abs() as usize];
            validate_kernel_coefficient(mode0, coefficient0)?;
            let denominator = coefficient0 * coefficient1;
            let amplitude = 1.0 / denominator;
            if !denominator.is_finite() || denominator == 0.0 || !amplitude.is_finite() {
                return Err(NufftError::InvalidKernelFourierCoefficient {
                    mode: mode0,
                    value: denominator,
                });
            }
            amplitudes.push(amplitude);
        }
    }
    Ok(amplitudes)
}

fn mode_amplitudes_2d_df64(
    config: &NufftConfig,
    fine_shape: &[usize],
    coefficients: &[Vec<f64>],
) -> Result<Vec<DoubleFloat>> {
    mode_amplitudes_2d_f64(config, fine_shape, coefficients)
        .map(|amplitudes| amplitudes.into_iter().map(DoubleFloat::from_f64).collect())
}

fn validate_kernel_coefficient(mode: i64, coefficient: f64) -> Result<()> {
    if !coefficient.is_finite() || coefficient == 0.0 {
        Err(NufftError::InvalidKernelFourierCoefficient {
            mode,
            value: coefficient,
        })
    } else {
        Ok(())
    }
}

fn create_compute_pipeline(
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

fn generate_predeconvolution_wgsl(config: &NufftConfig, fine_shape: &[usize]) -> String {
    match config.precision() {
        FftPrecision::F64 => return generate_predeconvolution_wgsl_f64(config, fine_shape),
        FftPrecision::Df64 => return generate_predeconvolution_wgsl_df64(config, fine_shape),
        FftPrecision::F32 => {}
    }
    let modes0 = config.n_modes()[0];
    let modes1 = config.n_modes()[1];
    let fine0 = fine_shape[0];
    let fine1 = fine_shape[1];
    let fine_total = fine0 * fine1;
    let mode_total = modes0 * modes1;
    let total = fine_total * config.batch();
    let half0 = modes0 / 2;
    let half1 = modes1 / 2;
    let nonnegative0 = modes0.div_ceil(2);
    let nonnegative1 = modes1.div_ceil(2);
    let centered = matches!(config.mode_order(), ModeOrder::Centered);
    format!(
        r#"const WORKGROUP_SIZE: u32 = {WORKGROUP_SIZE}u;
const MODE0: u32 = {modes0}u;
const MODE1: u32 = {modes1}u;
const MODE_TOTAL: u32 = {mode_total}u;
const FINE0: u32 = {fine0}u;
const FINE1: u32 = {fine1}u;
const FINE_TOTAL: u32 = {fine_total}u;
const TOTAL_FINE_COUNT: u32 = {total}u;
const HALF0: u32 = {half0}u;
const HALF1: u32 = {half1}u;
const NONNEGATIVE0: u32 = {nonnegative0}u;
const NONNEGATIVE1: u32 = {nonnegative1}u;
const CENTERED_ORDER: bool = {centered};

@group(0) @binding(0) var<storage, read> coefficients: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read> amplitudes: array<f32>;
@group(0) @binding(2) var<storage, read_write> fine_grid: array<vec2<f32>>;

fn map_axis(
    index: u32,
    mode_count: u32,
    fine_length: u32,
    half: u32,
    nonnegative_count: u32,
) -> vec2<u32> {{
    if (index < nonnegative_count) {{
        return vec2<u32>(select(index, half + index, CENTERED_ORDER), 1u);
    }}
    if (index >= fine_length - half) {{
        let magnitude = fine_length - index;
        return vec2<u32>(
            select(mode_count - magnitude, half - magnitude, CENTERED_ORDER),
            1u,
        );
    }}
    return vec2<u32>(0u, 0u);
}}

@compute @workgroup_size({WORKGROUP_SIZE})
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {{
    let wg_flat = (wid.z * nwg.y + wid.y) * nwg.x + wid.x;
    if (wg_flat > (TOTAL_FINE_COUNT - 1u) / WORKGROUP_SIZE) {{ return; }}
    let index = wg_flat * WORKGROUP_SIZE + lid.x;
    if (index >= TOTAL_FINE_COUNT) {{ return; }}

    let batch_index = index / FINE_TOTAL;
    let grid_index = index - batch_index * FINE_TOTAL;
    let active_batch = arrayLength(&coefficients) / MODE_TOTAL;
    if (batch_index >= active_batch) {{
        fine_grid[index] = vec2<f32>(0.0, 0.0);
        return;
    }}

    let fine_index0 = grid_index % FINE0;
    let fine_index1 = grid_index / FINE0;
    let source0 = map_axis(fine_index0, MODE0, FINE0, HALF0, NONNEGATIVE0);
    let source1 = map_axis(fine_index1, MODE1, FINE1, HALF1, NONNEGATIVE1);
    if (source0.y != 0u && source1.y != 0u) {{
        let source_index = source0.x + MODE0 * source1.x;
        let coefficient_index = batch_index * MODE_TOTAL + source_index;
        fine_grid[index] = coefficients[coefficient_index] * amplitudes[source_index];
    }} else {{
        fine_grid[index] = vec2<f32>(0.0, 0.0);
    }}
}}
"#,
    )
}

fn generate_predeconvolution_wgsl_df64(config: &NufftConfig, fine_shape: &[usize]) -> String {
    let source = generate_predeconvolution_wgsl(
        &config.clone().with_precision(FftPrecision::F32),
        fine_shape,
    )
    .replace(
        "var<storage, read> coefficients: array<vec2<f32>>;",
        "var<storage, read> coefficients: array<vec4<f32>>;",
    )
    .replace(
        "var<storage, read> amplitudes: array<f32>;",
        "var<storage, read> amplitudes: array<vec2<f32>>;",
    )
    .replace(
        "var<storage, read_write> fine_grid: array<vec2<f32>>;",
        "var<storage, read_write> fine_grid: array<vec4<f32>>;",
    )
    .replace(
        "fine_grid[index] = vec2<f32>(0.0, 0.0);",
        "fine_grid[index] = vec4<f32>(0.0, 0.0, 0.0, 0.0);",
    )
    .replace(
        "fine_grid[index] = coefficients[coefficient_index] * amplitudes[source_index];",
        "let amplitude = amplitudes[source_index];\n        fine_grid[index] = df64_complex_scale(coefficients[coefficient_index], Df64(amplitude.x, amplitude.y));",
    );
    format!("{}\n{source}", wgpu_fft::kernels::DF64_WGSL)
}

fn generate_predeconvolution_wgsl_f64(config: &NufftConfig, fine_shape: &[usize]) -> String {
    let modes0 = config.n_modes()[0];
    let modes1 = config.n_modes()[1];
    let fine0 = fine_shape[0];
    let fine1 = fine_shape[1];
    let fine_total = fine0 * fine1;
    let mode_total = modes0 * modes1;
    let total = fine_total * config.batch();
    let half0 = modes0 / 2;
    let half1 = modes1 / 2;
    let nonnegative0 = modes0.div_ceil(2);
    let nonnegative1 = modes1.div_ceil(2);
    let centered = matches!(config.mode_order(), ModeOrder::Centered);
    format!(
        r#"const WORKGROUP_SIZE: u32 = {WORKGROUP_SIZE}u;
const MODE0: u32 = {modes0}u;
const MODE1: u32 = {modes1}u;
const MODE_TOTAL: u32 = {mode_total}u;
const FINE0: u32 = {fine0}u;
const FINE1: u32 = {fine1}u;
const FINE_TOTAL: u32 = {fine_total}u;
const TOTAL_FINE_COUNT: u32 = {total}u;
const HALF0: u32 = {half0}u;
const HALF1: u32 = {half1}u;
const NONNEGATIVE0: u32 = {nonnegative0}u;
const NONNEGATIVE1: u32 = {nonnegative1}u;
const CENTERED_ORDER: bool = {centered};

@group(0) @binding(0) var<storage, read> coefficients: array<vec2<f64>>;
@group(0) @binding(1) var<storage, read> amplitudes: array<f64>;
@group(0) @binding(2) var<storage, read_write> fine_grid: array<vec2<f64>>;

fn map_axis(
    index: u32,
    mode_count: u32,
    fine_length: u32,
    half: u32,
    nonnegative_count: u32,
) -> vec2<u32> {{
    if (index < nonnegative_count) {{
        return vec2<u32>(select(index, half + index, CENTERED_ORDER), 1u);
    }}
    if (index >= fine_length - half) {{
        let magnitude = fine_length - index;
        return vec2<u32>(
            select(mode_count - magnitude, half - magnitude, CENTERED_ORDER),
            1u,
        );
    }}
    return vec2<u32>(0u, 0u);
}}

@compute @workgroup_size({WORKGROUP_SIZE})
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {{
    let wg_flat = (wid.z * nwg.y + wid.y) * nwg.x + wid.x;
    if (wg_flat > (TOTAL_FINE_COUNT - 1u) / WORKGROUP_SIZE) {{ return; }}
    let index = wg_flat * WORKGROUP_SIZE + lid.x;
    if (index >= TOTAL_FINE_COUNT) {{ return; }}

    let batch_index = index / FINE_TOTAL;
    let grid_index = index - batch_index * FINE_TOTAL;
    let active_batch = arrayLength(&coefficients) / MODE_TOTAL;
    if (batch_index >= active_batch) {{
        fine_grid[index] = vec2<f64>(0.0lf, 0.0lf);
        return;
    }}

    let fine_index0 = grid_index % FINE0;
    let fine_index1 = grid_index / FINE0;
    let source0 = map_axis(fine_index0, MODE0, FINE0, HALF0, NONNEGATIVE0);
    let source1 = map_axis(fine_index1, MODE1, FINE1, HALF1, NONNEGATIVE1);
    if (source0.y != 0u && source1.y != 0u) {{
        let source_index = source0.x + MODE0 * source1.x;
        let coefficient_index = batch_index * MODE_TOTAL + source_index;
        fine_grid[index] = coefficients[coefficient_index] * amplitudes[source_index];
    }} else {{
        fine_grid[index] = vec2<f64>(0.0lf, 0.0lf);
    }}
}}
"#,
    )
}

fn generate_interpolation_wgsl_for_precision(
    precision: FftPrecision,
    kernel: EsKernel,
    fine_shape: &[usize],
) -> String {
    match precision {
        FftPrecision::F64 => generate_interpolation_wgsl_f64(kernel, fine_shape),
        FftPrecision::F32 => generate_interpolation_wgsl(kernel, fine_shape),
        FftPrecision::Df64 => generate_interpolation_wgsl_df64(kernel, fine_shape),
    }
}

fn generate_interpolation_wgsl(kernel: EsKernel, fine_shape: &[usize]) -> String {
    let width = kernel.width();
    let beta = format_wgsl_f32(kernel.beta() as f32);
    let half_width = format_wgsl_f32(kernel.half_width() as f32);
    let fine0 = fine_shape[0];
    let fine1 = fine_shape[1];
    let fine_total = fine0 * fine1;
    let fine0_f32 = format_wgsl_f32(fine0 as f32);
    let fine1_f32 = format_wgsl_f32(fine1 as f32);
    let scale0 = split_f64(fine0 as f64 / std::f64::consts::TAU);
    let scale1 = split_f64(fine1 as f64 / std::f64::consts::TAU);
    let scale0_hi = format_wgsl_f32(scale0.0);
    let scale0_lo = format_wgsl_f32(scale0.1);
    let scale1_hi = format_wgsl_f32(scale1.0);
    let scale1_lo = format_wgsl_f32(scale1.1);
    let origin0 = format_wgsl_f32((fine0 / 2) as f32);
    let origin1 = format_wgsl_f32((fine1 / 2) as f32);
    let entry = format!(
        r#"const WORKGROUP_SIZE: u32 = {WORKGROUP_SIZE}u;
const FINE0: u32 = {fine0}u;
const FINE1: u32 = {fine1}u;
const FINE_TOTAL: u32 = {fine_total}u;
const FINE0_I32: i32 = {fine0}i;
const FINE1_I32: i32 = {fine1}i;
const FINE0_F32: f32 = {fine0_f32};
const FINE1_F32: f32 = {fine1_f32};
const VECTOR_TILE: u32 = {VECTOR_TILE}u;
const WIDTH: u32 = {width}u;
const WIDTH_F32: f32 = {width}.0;
const HALF_WIDTH: f32 = {half_width};
const BETA: f32 = {beta};
const POSITION_SCALE0_HI: f32 = {scale0_hi};
const POSITION_SCALE0_LO: f32 = {scale0_lo};
const POSITION_SCALE1_HI: f32 = {scale1_hi};
const POSITION_SCALE1_LO: f32 = {scale1_lo};
const GRID_ORIGIN0: f32 = {origin0};
const GRID_ORIGIN1: f32 = {origin1};

@group(0) @binding(0) var<storage, read> points: array<f32>;
@group(0) @binding(1) var<storage, read> fine_grid: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> output_values: array<vec2<f32>>;

fn es_weight(distance: f32) -> f32 {{
    let scaled = 2.0 * abs(distance) / WIDTH_F32;
    let squared = scaled * scaled;
    if (squared >= 1.0) {{ return 0.0; }}
    return exp(BETA * (sqrt(max(0.0, 1.0 - squared)) - 1.0));
}}

fn wrap_index(index: i32, length: i32) -> u32 {{
    var wrapped = index;
    if (wrapped < 0) {{ wrapped = wrapped + length; }}
    if (wrapped >= length) {{ wrapped = wrapped - length; }}
    return u32(wrapped);
}}

fn ceil_df64_to_i32(value: Df64) -> i32 {{
    let base = floor(value.hi);
    let remainder = df64_sub(value, Df64(base, 0.0));
    let has_positive_remainder = remainder.hi > 0.0 ||
        (remainder.hi == 0.0 && remainder.lo > 0.0);
    return i32(base) + select(0, 1, has_positive_remainder);
}}

fn position_is_negative(value: Df64) -> bool {{
    return value.hi < 0.0 || (value.hi == 0.0 && value.lo < 0.0);
}}

fn position_at_least_grid(value: Df64, fine_length: f32) -> bool {{
    return value.hi > fine_length ||
        (value.hi == fine_length && value.lo >= 0.0);
}}

fn fold_position(
    point: f32,
    fine_length: f32,
    scale_hi: f32,
    scale_lo: f32,
    origin: f32,
) -> Df64 {{
    let scaled = df64_mul(Df64(point, 0.0), Df64(scale_hi, scale_lo));
    var position = df64_add(scaled, Df64(origin, 0.0));
    if (position_is_negative(position)) {{
        position = df64_add(position, Df64(fine_length, 0.0));
    }}
    if (position_is_negative(position)) {{
        position = df64_add(position, Df64(fine_length, 0.0));
    }}
    if (position_at_least_grid(position, fine_length)) {{
        position = df64_sub(position, Df64(fine_length, 0.0));
    }}
    if (position_at_least_grid(position, fine_length)) {{
        position = df64_sub(position, Df64(fine_length, 0.0));
    }}
    return position;
}}

@compute @workgroup_size({WORKGROUP_SIZE})
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {{
    let point_count = arrayLength(&points) / 2u;
    let active_batch = arrayLength(&output_values) / point_count;
    let vector_tile_count = (active_batch + VECTOR_TILE - 1u) / VECTOR_TILE;
    let total = point_count * vector_tile_count;
    let wg_flat = (wid.z * nwg.y + wid.y) * nwg.x + wid.x;
    if (wg_flat > (total - 1u) / WORKGROUP_SIZE) {{ return; }}
    let work_index = wg_flat * WORKGROUP_SIZE + lid.x;
    if (work_index >= total) {{ return; }}

    let vector_tile_index = work_index / point_count;
    let point_index = work_index - vector_tile_index * point_count;
    let batch_start = vector_tile_index * VECTOR_TILE;

    let point_base = 2u * point_index;
    let position0 = fold_position(
        points[point_base],
        FINE0_F32,
        POSITION_SCALE0_HI,
        POSITION_SCALE0_LO,
        GRID_ORIGIN0,
    );
    let position1 = fold_position(
        points[point_base + 1u],
        FINE1_F32,
        POSITION_SCALE1_HI,
        POSITION_SCALE1_LO,
        GRID_ORIGIN1,
    );
    let start0 = ceil_df64_to_i32(df64_sub(position0, Df64(HALF_WIDTH, 0.0)));
    let start1 = ceil_df64_to_i32(df64_sub(position1, Df64(HALF_WIDTH, 0.0)));
    var fine_indices0: array<u32, WIDTH>;
    var weights0: array<f32, WIDTH>;
    for (var offset0 = 0u; offset0 < WIDTH; offset0 = offset0 + 1u) {{
        let unwrapped0 = start0 + i32(offset0);
        let distance0 = df64_sub(Df64(f32(unwrapped0), 0.0), position0);
        fine_indices0[offset0] = wrap_index(unwrapped0, FINE0_I32);
        weights0[offset0] = es_weight(distance0.hi + distance0.lo);
    }}
    var sums: array<vec2<f32>, VECTOR_TILE>;
    for (var vector = 0u; vector < VECTOR_TILE; vector = vector + 1u) {{
        sums[vector] = vec2<f32>(0.0, 0.0);
    }}
    for (var offset1 = 0u; offset1 < WIDTH; offset1 = offset1 + 1u) {{
        let unwrapped1 = start1 + i32(offset1);
        let distance1 = df64_sub(Df64(f32(unwrapped1), 0.0), position1);
        let weight1 = es_weight(distance1.hi + distance1.lo);
        let fine_index1 = wrap_index(unwrapped1, FINE1_I32);
        for (var offset0 = 0u; offset0 < WIDTH; offset0 = offset0 + 1u) {{
            let weight0 = weights0[offset0];
            let fine_index0 = fine_indices0[offset0];
            let grid_index = fine_index0 + FINE0 * fine_index1;
            for (var vector = 0u; vector < VECTOR_TILE; vector = vector + 1u) {{
                let batch_index = batch_start + vector;
                if (batch_index < active_batch) {{
                    let fine_index = batch_index * FINE_TOTAL + grid_index;
                    sums[vector] = sums[vector] + fine_grid[fine_index] * (weight0 * weight1);
                }}
            }}
        }}
    }}
    for (var vector = 0u; vector < VECTOR_TILE; vector = vector + 1u) {{
        let batch_index = batch_start + vector;
        if (batch_index < active_batch) {{
            output_values[batch_index * point_count + point_index] = sums[vector];
        }}
    }}
}}
"#,
    );
    format!("{}\n{entry}", wgpu_fft::kernels::DF64_WGSL)
}

fn generate_interpolation_wgsl_df64(kernel: EsKernel, fine_shape: &[usize]) -> String {
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
    let fine0 = fine_shape[0];
    let fine1 = fine_shape[1];
    let fine_total = fine0 * fine1;
    let entry = format!(
        r#"const WORKGROUP_SIZE: u32 = {WORKGROUP_SIZE}u;
const FINE0: u32 = {fine0}u;
const FINE1: u32 = {fine1}u;
const FINE_TOTAL: u32 = {fine_total}u;
const FINE0_I32: i32 = {fine0}i;
const FINE1_I32: i32 = {fine1}i;
const FINE0_DF64: vec2<f32> = {fine0_df64};
const FINE1_DF64: vec2<f32> = {fine1_df64};
const VECTOR_TILE: u32 = {VECTOR_TILE}u;
const WIDTH: u32 = {width}u;
const HALF_WIDTH: vec2<f32> = {half_width};
const HORNER_COEFFICIENT_COUNT: u32 = {coefficient_count}u;
const HORNER_COEFFICIENTS: array<vec2<f32>, {coefficient_total}> = array<vec2<f32>, {coefficient_total}>({coefficients});
const POSITION_SCALE0: vec2<f32> = {scale0};
const POSITION_SCALE1: vec2<f32> = {scale1};
const GRID_ORIGIN0: vec2<f32> = {origin0};
const GRID_ORIGIN1: vec2<f32> = {origin1};

@group(0) @binding(0) var<storage, read> points: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read> fine_grid: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> output_values: array<vec4<f32>>;

fn load_df64(value: vec2<f32>) -> Df64 {{
    return Df64(value.x, value.y);
}}

fn store_df64(value: Df64) -> vec2<f32> {{
    return vec2<f32>(value.hi, value.lo);
}}

fn df64_less(a: Df64, b: Df64) -> bool {{
    let difference = df64_sub(a, b);
    return difference.hi < 0.0 || (difference.hi == 0.0 && difference.lo < 0.0);
}}

fn df64_at_least(a: Df64, b: Df64) -> bool {{
    return !df64_less(a, b);
}}

fn df64_abs_value(value: Df64) -> Df64 {{
    if (df64_less(value, Df64(0.0, 0.0))) {{
        return df64_neg(value);
    }}
    return value;
}}

fn ceil_df64_to_i32(value: Df64) -> i32 {{
    let base = floor(value.hi);
    let remainder = df64_sub(value, Df64(base, 0.0));
    let has_positive_remainder = remainder.hi > 0.0 ||
        (remainder.hi == 0.0 && remainder.lo > 0.0);
    return i32(base) + select(0, 1, has_positive_remainder);
}}

fn es_weight(distance: Df64) -> Df64 {{
    let half_width = load_df64(HALF_WIDTH);
    if (df64_at_least(df64_abs_value(distance), half_width)) {{
        return Df64(0.0, 0.0);
    }}
    var panel = ceil_df64_to_i32(df64_add(distance, half_width)) - 1;
    panel = clamp(panel, 0, i32(WIDTH) - 1);
    let local = df64_add(
        df64_mul(Df64(2.0, 0.0), df64_sub(distance, Df64(f32(panel), 0.0))),
        Df64(f32(WIDTH - 1u), 0.0),
    );
    var value = Df64(0.0, 0.0);
    for (var coefficient = 0u; coefficient < HORNER_COEFFICIENT_COUNT; coefficient = coefficient + 1u) {{
        let packed = HORNER_COEFFICIENTS[coefficient * WIDTH + u32(panel)];
        value = df64_add(df64_mul(value, local), load_df64(packed));
    }}
    return value;
}}

fn wrap_index(index: i32, length: i32) -> u32 {{
    var wrapped = index;
    if (wrapped < 0) {{ wrapped = wrapped + length; }}
    if (wrapped >= length) {{ wrapped = wrapped - length; }}
    return u32(wrapped);
}}

fn fold_position(
    point: vec2<f32>,
    fine_length_value: vec2<f32>,
    scale: vec2<f32>,
    origin: vec2<f32>,
) -> Df64 {{
    var position = df64_add(
        df64_mul(load_df64(point), load_df64(scale)),
        load_df64(origin),
    );
    let fine_length = load_df64(fine_length_value);
    if (df64_less(position, Df64(0.0, 0.0))) {{
        position = df64_add(position, fine_length);
    }}
    if (df64_less(position, Df64(0.0, 0.0))) {{
        position = df64_add(position, fine_length);
    }}
    if (df64_at_least(position, fine_length)) {{
        position = df64_sub(position, fine_length);
    }}
    if (df64_at_least(position, fine_length)) {{
        position = df64_sub(position, fine_length);
    }}
    return position;
}}

@compute @workgroup_size({WORKGROUP_SIZE})
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {{
    let point_count = arrayLength(&points) / 2u;
    let active_batch = arrayLength(&output_values) / point_count;
    let vector_tile_count = (active_batch + VECTOR_TILE - 1u) / VECTOR_TILE;
    let total = point_count * vector_tile_count;
    let wg_flat = (wid.z * nwg.y + wid.y) * nwg.x + wid.x;
    if (wg_flat > (total - 1u) / WORKGROUP_SIZE) {{ return; }}
    let work_index = wg_flat * WORKGROUP_SIZE + lid.x;
    if (work_index >= total) {{ return; }}

    let vector_tile_index = work_index / point_count;
    let point_index = work_index - vector_tile_index * point_count;
    let batch_start = vector_tile_index * VECTOR_TILE;
    let point_base = 2u * point_index;
    let position0 = fold_position(
        points[point_base], FINE0_DF64, POSITION_SCALE0, GRID_ORIGIN0,
    );
    let position1 = fold_position(
        points[point_base + 1u], FINE1_DF64, POSITION_SCALE1, GRID_ORIGIN1,
    );
    let start0 = ceil_df64_to_i32(df64_sub(position0, load_df64(HALF_WIDTH)));
    let start1 = ceil_df64_to_i32(df64_sub(position1, load_df64(HALF_WIDTH)));
    var fine_indices0: array<u32, WIDTH>;
    var weights0: array<vec2<f32>, WIDTH>;
    for (var offset0 = 0u; offset0 < WIDTH; offset0 = offset0 + 1u) {{
        let unwrapped0 = start0 + i32(offset0);
        let distance0 = df64_sub(Df64(f32(unwrapped0), 0.0), position0);
        fine_indices0[offset0] = wrap_index(unwrapped0, FINE0_I32);
        weights0[offset0] = store_df64(es_weight(distance0));
    }}
    var sums: array<vec4<f32>, VECTOR_TILE>;
    for (var vector = 0u; vector < VECTOR_TILE; vector = vector + 1u) {{
        sums[vector] = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }}
    for (var offset1 = 0u; offset1 < WIDTH; offset1 = offset1 + 1u) {{
        let unwrapped1 = start1 + i32(offset1);
        let distance1 = df64_sub(Df64(f32(unwrapped1), 0.0), position1);
        let weight1 = es_weight(distance1);
        let fine_index1 = wrap_index(unwrapped1, FINE1_I32);
        for (var offset0 = 0u; offset0 < WIDTH; offset0 = offset0 + 1u) {{
            let weight = df64_mul(load_df64(weights0[offset0]), weight1);
            let grid_index = fine_indices0[offset0] + FINE0 * fine_index1;
            for (var vector = 0u; vector < VECTOR_TILE; vector = vector + 1u) {{
                let batch_index = batch_start + vector;
                if (batch_index < active_batch) {{
                    let fine_index = batch_index * FINE_TOTAL + grid_index;
                    sums[vector] = df64_complex_add(
                        sums[vector],
                        df64_complex_scale(fine_grid[fine_index], weight),
                    );
                }}
            }}
        }}
    }}
    for (var vector = 0u; vector < VECTOR_TILE; vector = vector + 1u) {{
        let batch_index = batch_start + vector;
        if (batch_index < active_batch) {{
            output_values[batch_index * point_count + point_index] = sums[vector];
        }}
    }}
}}
"#,
        fine0_df64 = format_wgsl_df64_pair(fine0 as f64),
        fine1_df64 = format_wgsl_df64_pair(fine1 as f64),
        half_width = format_wgsl_df64_pair(kernel.half_width()),
        scale0 = format_wgsl_df64_pair(fine0 as f64 / std::f64::consts::TAU),
        scale1 = format_wgsl_df64_pair(fine1 as f64 / std::f64::consts::TAU),
        origin0 = format_wgsl_df64_pair((fine0 / 2) as f64),
        origin1 = format_wgsl_df64_pair((fine1 / 2) as f64),
    );
    format!("{}\n{entry}", wgpu_fft::kernels::DF64_WGSL)
}

fn generate_interpolation_wgsl_f64(kernel: EsKernel, fine_shape: &[usize]) -> String {
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
    let half_width = format_wgsl_f64(kernel.half_width());
    let fine0 = fine_shape[0];
    let fine1 = fine_shape[1];
    let fine_total = fine0 * fine1;
    let fine0_f64 = format_wgsl_f64(fine0 as f64);
    let fine1_f64 = format_wgsl_f64(fine1 as f64);
    let scale0 = format_wgsl_f64(fine0 as f64 / std::f64::consts::TAU);
    let scale1 = format_wgsl_f64(fine1 as f64 / std::f64::consts::TAU);
    let origin0 = format_wgsl_f64((fine0 / 2) as f64);
    let origin1 = format_wgsl_f64((fine1 / 2) as f64);
    format!(
        r#"const WORKGROUP_SIZE: u32 = {WORKGROUP_SIZE}u;
const FINE0: u32 = {fine0}u;
const FINE1: u32 = {fine1}u;
const FINE_TOTAL: u32 = {fine_total}u;
const FINE0_I32: i32 = {fine0}i;
const FINE1_I32: i32 = {fine1}i;
const FINE0_F64: f64 = {fine0_f64};
const FINE1_F64: f64 = {fine1_f64};
const VECTOR_TILE: u32 = {VECTOR_TILE}u;
const WIDTH: u32 = {width}u;
const HALF_WIDTH: f64 = {half_width};
const HORNER_COEFFICIENT_COUNT: u32 = {coefficient_count}u;
const HORNER_COEFFICIENTS: array<f64, {coefficient_total}> = array<f64, {coefficient_total}>({coefficients});
const POSITION_SCALE0: f64 = {scale0};
const POSITION_SCALE1: f64 = {scale1};
const GRID_ORIGIN0: f64 = {origin0};
const GRID_ORIGIN1: f64 = {origin1};

@group(0) @binding(0) var<storage, read> points: array<f64>;
@group(0) @binding(1) var<storage, read> fine_grid: array<vec2<f64>>;
@group(0) @binding(2) var<storage, read_write> output_values: array<vec2<f64>>;

fn ceil_f64_to_i32(value: f64) -> i32 {{
    let truncated = i32(value);
    return truncated + select(0, 1, value > f64(truncated));
}}

fn es_weight(distance: f64) -> f64 {{
    if (abs(distance) >= HALF_WIDTH) {{ return 0.0lf; }}
    var panel = ceil_f64_to_i32(distance + HALF_WIDTH) - 1;
    panel = clamp(panel, 0, i32(WIDTH) - 1);
    let local = 2.0lf * (distance - f64(panel)) + f64(WIDTH - 1u);
    var value = 0.0lf;
    for (var coefficient = 0u; coefficient < HORNER_COEFFICIENT_COUNT; coefficient = coefficient + 1u) {{
        value = value * local + HORNER_COEFFICIENTS[coefficient * WIDTH + u32(panel)];
    }}
    return value;
}}

fn wrap_index(index: i32, length: i32) -> u32 {{
    var wrapped = index;
    if (wrapped < 0) {{ wrapped = wrapped + length; }}
    if (wrapped >= length) {{ wrapped = wrapped - length; }}
    return u32(wrapped);
}}

fn fold_position(point: f64, fine_length: f64, scale: f64, origin: f64) -> f64 {{
    var position = point * scale + origin;
    if (position < 0.0lf) {{ position = position + fine_length; }}
    if (position < 0.0lf) {{ position = position + fine_length; }}
    if (position >= fine_length) {{ position = position - fine_length; }}
    if (position >= fine_length) {{ position = position - fine_length; }}
    return position;
}}

@compute @workgroup_size({WORKGROUP_SIZE})
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {{
    let point_count = arrayLength(&points) / 2u;
    let active_batch = arrayLength(&output_values) / point_count;
    let vector_tile_count = (active_batch + VECTOR_TILE - 1u) / VECTOR_TILE;
    let total = point_count * vector_tile_count;
    let wg_flat = (wid.z * nwg.y + wid.y) * nwg.x + wid.x;
    if (wg_flat > (total - 1u) / WORKGROUP_SIZE) {{ return; }}
    let work_index = wg_flat * WORKGROUP_SIZE + lid.x;
    if (work_index >= total) {{ return; }}

    let vector_tile_index = work_index / point_count;
    let point_index = work_index - vector_tile_index * point_count;
    let batch_start = vector_tile_index * VECTOR_TILE;

    let point_base = 2u * point_index;
    let position0 = fold_position(
        points[point_base],
        FINE0_F64,
        POSITION_SCALE0,
        GRID_ORIGIN0,
    );
    let position1 = fold_position(
        points[point_base + 1u],
        FINE1_F64,
        POSITION_SCALE1,
        GRID_ORIGIN1,
    );
    let start0 = ceil_f64_to_i32(position0 - HALF_WIDTH);
    let start1 = ceil_f64_to_i32(position1 - HALF_WIDTH);
    var fine_indices0: array<u32, WIDTH>;
    var weights0: array<f64, WIDTH>;
    for (var offset0 = 0u; offset0 < WIDTH; offset0 = offset0 + 1u) {{
        let unwrapped0 = start0 + i32(offset0);
        fine_indices0[offset0] = wrap_index(unwrapped0, FINE0_I32);
        weights0[offset0] = es_weight(f64(unwrapped0) - position0);
    }}
    var sums: array<vec2<f64>, VECTOR_TILE>;
    for (var vector = 0u; vector < VECTOR_TILE; vector = vector + 1u) {{
        sums[vector] = vec2<f64>(0.0lf, 0.0lf);
    }}
    for (var offset1 = 0u; offset1 < WIDTH; offset1 = offset1 + 1u) {{
        let unwrapped1 = start1 + i32(offset1);
        let weight1 = es_weight(f64(unwrapped1) - position1);
        let fine_index1 = wrap_index(unwrapped1, FINE1_I32);
        for (var offset0 = 0u; offset0 < WIDTH; offset0 = offset0 + 1u) {{
            let grid_index = fine_indices0[offset0] + FINE0 * fine_index1;
            let weight = weights0[offset0] * weight1;
            for (var vector = 0u; vector < VECTOR_TILE; vector = vector + 1u) {{
                let batch_index = batch_start + vector;
                if (batch_index < active_batch) {{
                    let fine_index = batch_index * FINE_TOTAL + grid_index;
                    sums[vector] = sums[vector] + fine_grid[fine_index] * weight;
                }}
            }}
        }}
    }}
    for (var vector = 0u; vector < VECTOR_TILE; vector = vector + 1u) {{
        let batch_index = batch_start + vector;
        if (batch_index < active_batch) {{
            output_values[batch_index * point_count + point_index] = sums[vector];
        }}
    }}
}}
"#,
    )
}

fn split_f64(value: f64) -> (f32, f32) {
    let hi = value as f32;
    (hi, (value - f64::from(hi)) as f32)
}

fn format_wgsl_f32(value: f32) -> String {
    debug_assert!(value.is_finite());
    let mut formatted = value.to_string();
    if !formatted.contains('.') && !formatted.contains('e') && !formatted.contains('E') {
        formatted.push_str(".0");
    }
    formatted
}

fn format_wgsl_f64(value: f64) -> String {
    debug_assert!(value.is_finite());
    let mut formatted = value.to_string();
    if !formatted.contains('.') && !formatted.contains('e') && !formatted.contains('E') {
        formatted.push_str(".0");
    }
    formatted.push_str("lf");
    formatted
}

fn format_wgsl_df64_pair(value: f64) -> String {
    let value = DoubleFloat::from_f64(value);
    format!(
        "vec2<f32>({}, {})",
        format_wgsl_f32(value.hi),
        format_wgsl_f32(value.lo)
    )
}

fn maximum_supported_workgroup_size(limits: &wgpu::Limits) -> u32 {
    limits
        .max_compute_invocations_per_workgroup
        .min(limits.max_compute_workgroup_size_x)
}

fn checked_product(context: &'static str, left: usize, right: usize) -> Result<usize> {
    left.checked_mul(right)
        .ok_or(NufftError::LengthOverflow { context })
}

fn checked_buffer_size(
    context: &'static str,
    elements: usize,
    bytes_per_element: u64,
) -> Result<u64> {
    u64::try_from(elements)
        .ok()
        .and_then(|count| count.checked_mul(bytes_per_element))
        .ok_or(NufftError::LengthOverflow { context })
}

fn validate_binding_limit(buffer: &'static str, bytes: u64, limit: u64) -> Result<()> {
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

fn validate_internal_buffer_limits(
    buffer: &'static str,
    bytes: u64,
    max_storage_binding_bytes: u64,
    max_buffer_bytes: u64,
) -> Result<()> {
    validate_binding_limit(buffer, bytes, max_storage_binding_bytes)?;
    validate_binding_limit(buffer, bytes, max_buffer_bytes)
}

fn validate_external_storage_buffer(
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

fn binding_entry<'a>(
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
    fn tensor_amplitudes_are_axis0_fast_for_both_mode_orders() {
        let fine_shape = [8, 10];
        let coefficients = vec![
            vec![2.0, 4.0, 8.0, 16.0, 32.0],
            vec![5.0, 10.0, 20.0, 40.0, 80.0, 160.0],
        ];
        for order in [ModeOrder::Centered, ModeOrder::Fft] {
            let config = NufftConfig::new([3, 2], 1.0e-6).with_mode_order(order);
            let amplitudes = mode_amplitudes_2d(&config, &fine_shape, &coefficients).unwrap();
            let mut expected = Vec::new();
            for index1 in 0..2 {
                let mode1 = mode_for_storage_index(order, index1, 2);
                for index0 in 0..3 {
                    let mode0 = mode_for_storage_index(order, index0, 3);
                    expected.push(
                        (1.0 / (coefficients[0][mode0.unsigned_abs() as usize]
                            * coefficients[1][mode1.unsigned_abs() as usize]))
                            as f32,
                    );
                }
            }
            assert_eq!(amplitudes, expected);
        }
    }

    #[test]
    fn predeconvolution_shader_uses_axis0_fast_tensor_mapping() {
        let config = NufftConfig::new([5, 4], 1.0e-6).with_mode_order(ModeOrder::Centered);
        let source = generate_predeconvolution_wgsl(&config, &[16, 18]);
        assert!(source.contains("let fine_index0 = grid_index % FINE0;"));
        assert!(source.contains("let fine_index1 = grid_index / FINE0;"));
        assert!(source.contains("let source_index = source0.x + MODE0 * source1.x;"));
        assert!(source.contains("fine_grid[index] = vec2<f32>(0.0, 0.0);"));
        assert!(source.contains("const MODE0: u32 = 5u;"));
        assert!(source.contains("const MODE1: u32 = 4u;"));
    }

    #[test]
    fn interpolation_shader_reads_aos_points_and_tensor_product_support() {
        let kernel = EsKernel::for_tolerance(1.0e-6, 2.0).unwrap();
        let source = generate_interpolation_wgsl(kernel, &[256, 1024]);
        assert!(source.contains("let point_base = 2u * point_index;"));
        assert!(source.contains("points[point_base]"));
        assert!(source.contains("points[point_base + 1u]"));
        assert!(source.contains("let grid_index = fine_index0 + FINE0 * fine_index1;"));
        assert!(source.contains("let fine_index = batch_index * FINE_TOTAL + grid_index;"));
        assert!(source.contains("fine_grid[fine_index] * (weight0 * weight1)"));
        assert!(source.contains("for (var offset1 = 0u; offset1 < WIDTH;"));
        assert!(source.contains("for (var offset0 = 0u; offset0 < WIDTH;"));
    }

    #[test]
    fn interpolation_shader_preserves_low_words_on_both_axes() {
        let kernel = EsKernel::for_tolerance(1.0e-6, 2.0).unwrap();
        let source = generate_interpolation_wgsl(kernel, &[65_536, 131_072]);
        assert!(source.contains("fn ceil_df64_to_i32(value: Df64) -> i32"));
        assert!(source.contains("POSITION_SCALE0_LO"));
        assert!(source.contains("POSITION_SCALE1_LO"));
        assert!(source.contains("let remainder = df64_sub(value, Df64(base, 0.0));"));
        assert!(!source.contains("ceil(shifted.hi + shifted.lo)"));
    }

    #[test]
    fn batched_shaders_use_transform_major_storage_and_reuse_point_weights() {
        let config = NufftConfig::new([5, 4], 1.0e-6).with_batch(5);
        let predeconvolution = generate_predeconvolution_wgsl(&config, &[16, 18]);
        assert!(predeconvolution.contains("const TOTAL_FINE_COUNT: u32 = 1440u;"));
        assert!(predeconvolution.contains("let batch_index = index / FINE_TOTAL;"));
        assert!(predeconvolution
            .contains("let coefficient_index = batch_index * MODE_TOTAL + source_index;"));

        let kernel = EsKernel::for_tolerance(1.0e-6, 2.0).unwrap();
        let interpolation = generate_interpolation_wgsl(kernel, &[16, 18]);
        assert!(interpolation.contains("const VECTOR_TILE: u32 = 4u;"));
        assert!(interpolation.contains("let point_count = arrayLength(&points) / 2u;"));
        assert!(interpolation.contains("let batch_start = vector_tile_index * VECTOR_TILE;"));
        assert!(interpolation
            .contains("output_values[batch_index * point_count + point_index] = sums[vector];"));
    }

    #[test]
    fn native_f64_shaders_use_horner_without_transcendentals() {
        let config = NufftConfig::new([5, 4], 1.0e-6).with_precision(FftPrecision::F64);
        let kernel = EsKernel::for_tolerance(1.0e-6, 2.0).unwrap();
        let sources = [
            generate_predeconvolution_wgsl(&config, &[16, 18]),
            generate_interpolation_wgsl_for_precision(FftPrecision::F64, kernel, &[16, 18]),
        ];
        for source in sources {
            assert!(source.contains("array<vec2<f64>>"));
            assert!(source.contains("array<f64>"));
            for forbidden in ["exp(", "log(", "pow(", "sin(", "cos("] {
                assert!(
                    !source.contains(forbidden),
                    "found {forbidden} in:\n{source}"
                );
            }
        }
    }

    #[test]
    fn portable_df64_shaders_use_split_horner_and_complex_arithmetic() {
        let config = NufftConfig::new([5, 4], 1.0e-6).with_precision(FftPrecision::Df64);
        let kernel = EsKernel::for_tolerance(1.0e-6, 2.0).unwrap();
        let sources = [
            generate_predeconvolution_wgsl(&config, &[16, 18]),
            generate_interpolation_wgsl_for_precision(FftPrecision::Df64, kernel, &[16, 18]),
        ];
        for source in sources {
            assert!(source.contains("array<vec4<f32>>"));
            assert!(source.contains("df64_complex_scale"));
            assert!(source.contains("struct Df64"));
            for forbidden in ["exp(", "log(", "pow(", "sin(", "cos("] {
                assert!(
                    !source.contains(forbidden),
                    "found {forbidden} in:\n{source}"
                );
            }
        }
    }

    #[test]
    fn point_buffer_size_accounts_for_two_coordinates() {
        assert_eq!(
            Type2GpuPlan2d::point_buffer_size_bytes_for_precision(7, FftPrecision::F32).unwrap(),
            56
        );
        assert!(matches!(
            Type2GpuPlan2d::point_buffer_size_bytes_for_precision(usize::MAX, FftPrecision::F32),
            Err(NufftError::LengthOverflow {
                context: "type-2 two-dimensional point-coordinate count"
            })
        ));
    }
}
