use std::num::NonZeroU64;

use wgpu::util::DeviceExt;
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
const DIMENSIONS: usize = 3;

/// Device-specific state for a three-dimensional type-2 NUFFT.
///
/// Mode and fine-grid buffers use axis-0-fast linearization. Caller-owned
/// point coordinates are point-major triples `[x0, y0, z0, x1, y1, z1, ...]`.
pub(crate) struct Type2GpuPlan3d {
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

impl Type2GpuPlan3d {
    pub(crate) fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        config: &NufftConfig,
        kernel: EsKernel,
        fine_shape: &[usize],
        kernel_fourier_coefficients: &[Vec<f64>],
    ) -> Result<Self> {
        if config.precision() == FftPrecision::Df64 {
            return Err(NufftError::PrecisionUnsupported {
                requested: FftPrecision::Df64,
                stage: "three-dimensional type-2 GPU plan",
                reason: "portable df64 NUFFT kernels are not implemented",
            });
        }
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
                stage: "type-2 three-dimensional oversampled-grid C2C plan",
                source,
            })?;

        // The legacy f32 path converts every support coordinate through an
        // exact f32 integer. Native f64 retains adjacent cells throughout the
        // signed WGSL index range.
        let maximum_signed_length = match precision {
            FftPrecision::F64 => i32::MAX as usize - kernel.width(),
            FftPrecision::F32 => (1usize << f32::MANTISSA_DIGITS)
                .saturating_sub(kernel.width())
                .min(i32::MAX as usize - kernel.width()),
            FftPrecision::Df64 => unreachable!("df64 was rejected above"),
        };
        for (axis, &length) in fine_shape.iter().enumerate() {
            if !length.is_multiple_of(2) {
                return Err(NufftError::FineGridMustBeEven { length });
            }
            if length < config.n_modes()[axis] {
                return Err(NufftError::InputLength {
                    input: fine_axis_label(axis),
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
                context: "type-2 three-dimensional fine-grid axis length",
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
        let fine_count = checked_product3(
            "type-2 three-dimensional fine-grid element count",
            fine_shape[0],
            fine_shape[1],
            fine_shape[2],
        )?;
        let fine_element_count = checked_product(
            "type-2 three-dimensional batched fine-grid element count",
            fine_count,
            batch_capacity,
        )?;
        let fine_element_count_u32 =
            u32::try_from(fine_element_count).map_err(|_| NufftError::LengthOverflow {
                context: "type-2 three-dimensional batched fine-grid shader index space",
            })?;
        u32::try_from(mode_count).map_err(|_| NufftError::LengthOverflow {
            context: "type-2 three-dimensional mode shader index space",
        })?;

        let fine_bytes = Self::complex_buffer_size_bytes_for_precision(
            "type-2 3D fine grid",
            fine_element_count,
            precision,
        )?;
        let amplitude_bytes = checked_buffer_size(
            "type-2 3D deconvolution amplitudes",
            mode_count,
            precision.scalar_size_bytes(),
        )?;
        let max_storage_binding_bytes = limits.max_storage_buffer_binding_size;
        validate_internal_buffer_limits(
            "type-2 3D fine grid",
            fine_bytes,
            max_storage_binding_bytes,
            limits.max_buffer_size,
        )?;
        validate_internal_buffer_limits(
            "type-2 3D deconvolution amplitudes",
            amplitude_bytes,
            max_storage_binding_bytes,
            limits.max_buffer_size,
        )?;

        let amplitude_buffer = match precision {
            FftPrecision::F64 => {
                let amplitudes =
                    mode_amplitudes_3d_f64(config, fine_shape, kernel_fourier_coefficients)?;
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("wgpu_nufft.type2_3d.deconvolution_amplitudes"),
                    contents: bytemuck::cast_slice(&amplitudes),
                    usage: wgpu::BufferUsages::STORAGE,
                })
            }
            FftPrecision::F32 => {
                let amplitudes =
                    mode_amplitudes_3d(config, fine_shape, kernel_fourier_coefficients)?;
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("wgpu_nufft.type2_3d.deconvolution_amplitudes"),
                    contents: bytemuck::cast_slice(&amplitudes),
                    usage: wgpu::BufferUsages::STORAGE,
                })
            }
            FftPrecision::Df64 => unreachable!("df64 was rejected above"),
        };
        let fine_usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST;
        let fine_input = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("wgpu_nufft.type2_3d.fine_input"),
            size: fine_bytes,
            usage: fine_usage,
            mapped_at_creation: false,
        });
        let fine_output = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("wgpu_nufft.type2_3d.fine_output"),
            size: fine_bytes,
            usage: fine_usage,
            mapped_at_creation: false,
        });

        let fft = FftPlan::c2c(device, queue, fft_config).map_err(|source| {
            NufftError::FftShapeUnsupported {
                stage: "type-2 three-dimensional oversampled-grid C2C plan",
                source,
            }
        })?;

        let predeconvolution_pipeline = create_compute_pipeline(
            device,
            "wgpu_nufft.type2_3d.predeconvolution",
            &generate_predeconvolution_wgsl(config, fine_shape),
        );
        let predeconvolution_layout = predeconvolution_pipeline.get_bind_group_layout(0);
        let interpolation_pipeline = create_compute_pipeline(
            device,
            "wgpu_nufft.type2_3d.interpolation",
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
                    context: "type-2 three-dimensional point-coordinate count",
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
                kind: "type-2 3D stage profiling",
                reason: "at least one point is required",
            });
        }
        let layout =
            NufftGpuProfileLayout::type2(first_query).map_err(|_| NufftError::LengthOverflow {
                context: "type-2 3D stage-profile query range",
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
                    context: "type-2 three-dimensional point-coordinate count",
                })?,
        )
        .map_err(|_| NufftError::LengthOverflow {
            context: "type-2 three-dimensional point-coordinate shader index space",
        })?;
        let point_bytes = Self::point_buffer_size_bytes_for_precision(point_count, self.precision)?;
        let coefficient_elements = checked_product(
            "type-2 three-dimensional batched Fourier coefficient count",
            active_batch,
            self.mode_count,
        )?;
        let output_elements = checked_product(
            "type-2 three-dimensional batched output element count",
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
            "type-2 three-dimensional batched interpolation work item count",
            point_count,
            vector_tiles,
        )?;
        let interpolation_elements_u32 =
            u32::try_from(interpolation_elements).map_err(|_| NufftError::LengthOverflow {
                context: "type-2 three-dimensional batched interpolation shader index space",
            })?;
        let interpolation_dispatch = split_workgroups(
            interpolation_elements_u32.div_ceil(WORKGROUP_SIZE),
            self.max_workgroups_per_dimension,
        )?;

        let predeconvolution_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("wgpu_nufft.type2_3d.predeconvolution.bind_group"),
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
                label: Some("wgpu_nufft.type2_3d.predeconvolution.pass"),
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
                stage: "type-2 three-dimensional oversampled-grid C2C transform",
                source,
            })?;

        let interpolation_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("wgpu_nufft.type2_3d.interpolation.bind_group"),
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
                label: Some("wgpu_nufft.type2_3d.interpolation.pass"),
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

fn fine_axis_label(axis: usize) -> &'static str {
    match axis {
        0 => "type-2 fine-grid axis-0 length",
        1 => "type-2 fine-grid axis-1 length",
        _ => "type-2 fine-grid axis-2 length",
    }
}

fn coefficient_axis_label(axis: usize) -> &'static str {
    match axis {
        0 => "axis-0 centered kernel Fourier coefficients",
        1 => "axis-1 centered kernel Fourier coefficients",
        _ => "axis-2 centered kernel Fourier coefficients",
    }
}

fn mode_amplitudes_3d(
    config: &NufftConfig,
    fine_shape: &[usize],
    coefficients: &[Vec<f64>],
) -> Result<Vec<f32>> {
    for axis in 0..DIMENSIONS {
        let expected = fine_shape[axis] / 2 + 1;
        if coefficients[axis].len() != expected {
            return Err(NufftError::InputLength {
                input: coefficient_axis_label(axis),
                expected,
                actual: coefficients[axis].len(),
            });
        }
        let required = config.n_modes()[axis] / 2 + 1;
        if coefficients[axis].len() < required {
            return Err(NufftError::InputLength {
                input: coefficient_axis_label(axis),
                expected: required,
                actual: coefficients[axis].len(),
            });
        }
    }

    let mode_shape = config.n_modes();
    let mode_count = config.mode_count()?;
    let mut amplitudes = Vec::new();
    amplitudes
        .try_reserve_exact(mode_count)
        .map_err(|_| NufftError::HostAllocationFailed {
            buffer: "type-2 three-dimensional deconvolution amplitudes",
            elements: mode_count,
        })?;

    for index2 in 0..mode_shape[2] {
        let mode2 = mode_for_storage_index(config.mode_order(), index2, mode_shape[2]);
        let coefficient2 = coefficients[2][mode2.unsigned_abs() as usize];
        validate_kernel_coefficient(mode2, coefficient2)?;
        for index1 in 0..mode_shape[1] {
            let mode1 = mode_for_storage_index(config.mode_order(), index1, mode_shape[1]);
            let coefficient1 = coefficients[1][mode1.unsigned_abs() as usize];
            validate_kernel_coefficient(mode1, coefficient1)?;
            for index0 in 0..mode_shape[0] {
                let mode0 = mode_for_storage_index(config.mode_order(), index0, mode_shape[0]);
                let coefficient0 = coefficients[0][mode0.unsigned_abs() as usize];
                validate_kernel_coefficient(mode0, coefficient0)?;
                let denominator = coefficient0 * coefficient1 * coefficient2;
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
    }
    Ok(amplitudes)
}

fn mode_amplitudes_3d_f64(
    config: &NufftConfig,
    fine_shape: &[usize],
    coefficients: &[Vec<f64>],
) -> Result<Vec<f64>> {
    for axis in 0..DIMENSIONS {
        let expected = fine_shape[axis] / 2 + 1;
        if coefficients[axis].len() != expected {
            return Err(NufftError::InputLength {
                input: coefficient_axis_label(axis),
                expected,
                actual: coefficients[axis].len(),
            });
        }
        let required = config.n_modes()[axis] / 2 + 1;
        if coefficients[axis].len() < required {
            return Err(NufftError::InputLength {
                input: coefficient_axis_label(axis),
                expected: required,
                actual: coefficients[axis].len(),
            });
        }
    }

    let mode_shape = config.n_modes();
    let mode_count = config.mode_count()?;
    let mut amplitudes = Vec::new();
    amplitudes
        .try_reserve_exact(mode_count)
        .map_err(|_| NufftError::HostAllocationFailed {
            buffer: "type-2 three-dimensional deconvolution amplitudes",
            elements: mode_count,
        })?;

    for index2 in 0..mode_shape[2] {
        let mode2 = mode_for_storage_index(config.mode_order(), index2, mode_shape[2]);
        let coefficient2 = coefficients[2][mode2.unsigned_abs() as usize];
        validate_kernel_coefficient(mode2, coefficient2)?;
        for index1 in 0..mode_shape[1] {
            let mode1 = mode_for_storage_index(config.mode_order(), index1, mode_shape[1]);
            let coefficient1 = coefficients[1][mode1.unsigned_abs() as usize];
            validate_kernel_coefficient(mode1, coefficient1)?;
            for index0 in 0..mode_shape[0] {
                let mode0 = mode_for_storage_index(config.mode_order(), index0, mode_shape[0]);
                let coefficient0 = coefficients[0][mode0.unsigned_abs() as usize];
                validate_kernel_coefficient(mode0, coefficient0)?;
                let denominator = coefficient0 * coefficient1 * coefficient2;
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
    }
    Ok(amplitudes)
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
        FftPrecision::Df64 => panic!("df64 type-2 WGSL is generated by the df64 path"),
        FftPrecision::F32 => {}
    }
    let modes0 = config.n_modes()[0];
    let modes1 = config.n_modes()[1];
    let modes2 = config.n_modes()[2];
    let fine0 = fine_shape[0];
    let fine1 = fine_shape[1];
    let fine2 = fine_shape[2];
    let fine01 = fine0 * fine1;
    let fine_total = fine01 * fine2;
    let mode_total = modes0 * modes1 * modes2;
    let total = fine_total * config.batch();
    let half0 = modes0 / 2;
    let half1 = modes1 / 2;
    let half2 = modes2 / 2;
    let nonnegative0 = modes0.div_ceil(2);
    let nonnegative1 = modes1.div_ceil(2);
    let nonnegative2 = modes2.div_ceil(2);
    let centered = matches!(config.mode_order(), ModeOrder::Centered);
    format!(
        r#"const WORKGROUP_SIZE: u32 = {WORKGROUP_SIZE}u;
const MODE0: u32 = {modes0}u;
const MODE1: u32 = {modes1}u;
const MODE2: u32 = {modes2}u;
const MODE_TOTAL: u32 = {mode_total}u;
const FINE0: u32 = {fine0}u;
const FINE1: u32 = {fine1}u;
const FINE2: u32 = {fine2}u;
const FINE01: u32 = {fine01}u;
const FINE_TOTAL: u32 = {fine_total}u;
const TOTAL_FINE_COUNT: u32 = {total}u;
const HALF0: u32 = {half0}u;
const HALF1: u32 = {half1}u;
const HALF2: u32 = {half2}u;
const NONNEGATIVE0: u32 = {nonnegative0}u;
const NONNEGATIVE1: u32 = {nonnegative1}u;
const NONNEGATIVE2: u32 = {nonnegative2}u;
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
    let fine_index1 = (grid_index / FINE0) % FINE1;
    let fine_index2 = grid_index / FINE01;
    let source0 = map_axis(fine_index0, MODE0, FINE0, HALF0, NONNEGATIVE0);
    let source1 = map_axis(fine_index1, MODE1, FINE1, HALF1, NONNEGATIVE1);
    let source2 = map_axis(fine_index2, MODE2, FINE2, HALF2, NONNEGATIVE2);
    if (source0.y != 0u && source1.y != 0u && source2.y != 0u) {{
        let source_index = source0.x + MODE0 * (source1.x + MODE1 * source2.x);
        let coefficient_index = batch_index * MODE_TOTAL + source_index;
        fine_grid[index] = coefficients[coefficient_index] * amplitudes[source_index];
    }} else {{
        fine_grid[index] = vec2<f32>(0.0, 0.0);
    }}
}}
"#,
    )
}

fn generate_predeconvolution_wgsl_f64(config: &NufftConfig, fine_shape: &[usize]) -> String {
    generate_predeconvolution_wgsl(
        &config.clone().with_precision(FftPrecision::F32),
        fine_shape,
    )
    .replace("vec2<f32>", "vec2<f64>")
    .replace("array<f32>", "array<f64>")
}

fn generate_interpolation_wgsl_for_precision(
    precision: FftPrecision,
    kernel: EsKernel,
    fine_shape: &[usize],
) -> String {
    match precision {
        FftPrecision::F64 => generate_interpolation_wgsl_f64(kernel, fine_shape),
        FftPrecision::F32 => generate_interpolation_wgsl(kernel, fine_shape),
        FftPrecision::Df64 => panic!("df64 type-2 WGSL is generated by the df64 path"),
    }
}

fn generate_interpolation_wgsl(kernel: EsKernel, fine_shape: &[usize]) -> String {
    let width = kernel.width();
    let beta = format_wgsl_f32(kernel.beta() as f32);
    let half_width = format_wgsl_f32(kernel.half_width() as f32);
    let fine0 = fine_shape[0];
    let fine1 = fine_shape[1];
    let fine2 = fine_shape[2];
    let fine_total = fine0 * fine1 * fine2;
    let fine0_f32 = format_wgsl_f32(fine0 as f32);
    let fine1_f32 = format_wgsl_f32(fine1 as f32);
    let fine2_f32 = format_wgsl_f32(fine2 as f32);
    let scale0 = split_f64(fine0 as f64 / std::f64::consts::TAU);
    let scale1 = split_f64(fine1 as f64 / std::f64::consts::TAU);
    let scale2 = split_f64(fine2 as f64 / std::f64::consts::TAU);
    let entry = format!(
        r#"const WORKGROUP_SIZE: u32 = {WORKGROUP_SIZE}u;
const FINE0: u32 = {fine0}u;
const FINE1: u32 = {fine1}u;
const FINE2: u32 = {fine2}u;
const FINE_TOTAL: u32 = {fine_total}u;
const FINE0_I32: i32 = {fine0}i;
const FINE1_I32: i32 = {fine1}i;
const FINE2_I32: i32 = {fine2}i;
const FINE0_F32: f32 = {fine0_f32};
const FINE1_F32: f32 = {fine1_f32};
const FINE2_F32: f32 = {fine2_f32};
const VECTOR_TILE: u32 = {VECTOR_TILE}u;
const WIDTH: u32 = {width}u;
const WIDTH_F32: f32 = {width}.0;
const HALF_WIDTH: f32 = {half_width};
const BETA: f32 = {beta};
const POSITION_SCALE0_HI: f32 = {scale0_hi};
const POSITION_SCALE0_LO: f32 = {scale0_lo};
const POSITION_SCALE1_HI: f32 = {scale1_hi};
const POSITION_SCALE1_LO: f32 = {scale1_lo};
const POSITION_SCALE2_HI: f32 = {scale2_hi};
const POSITION_SCALE2_LO: f32 = {scale2_lo};
const GRID_ORIGIN0: f32 = {origin0};
const GRID_ORIGIN1: f32 = {origin1};
const GRID_ORIGIN2: f32 = {origin2};

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
    let point_count = arrayLength(&points) / 3u;
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

    let point_base = 3u * point_index;
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
    let position2 = fold_position(
        points[point_base + 2u],
        FINE2_F32,
        POSITION_SCALE2_HI,
        POSITION_SCALE2_LO,
        GRID_ORIGIN2,
    );
    let start0 = ceil_df64_to_i32(df64_sub(position0, Df64(HALF_WIDTH, 0.0)));
    let start1 = ceil_df64_to_i32(df64_sub(position1, Df64(HALF_WIDTH, 0.0)));
    let start2 = ceil_df64_to_i32(df64_sub(position2, Df64(HALF_WIDTH, 0.0)));
    var fine_indices0: array<u32, WIDTH>;
    var weights0: array<f32, WIDTH>;
    var fine_indices1: array<u32, WIDTH>;
    var weights1: array<f32, WIDTH>;
    for (var offset0 = 0u; offset0 < WIDTH; offset0 = offset0 + 1u) {{
        let unwrapped0 = start0 + i32(offset0);
        let distance0 = df64_sub(Df64(f32(unwrapped0), 0.0), position0);
        fine_indices0[offset0] = wrap_index(unwrapped0, FINE0_I32);
        weights0[offset0] = es_weight(distance0.hi + distance0.lo);
    }}
    for (var offset1 = 0u; offset1 < WIDTH; offset1 = offset1 + 1u) {{
        let unwrapped1 = start1 + i32(offset1);
        let distance1 = df64_sub(Df64(f32(unwrapped1), 0.0), position1);
        fine_indices1[offset1] = wrap_index(unwrapped1, FINE1_I32);
        weights1[offset1] = es_weight(distance1.hi + distance1.lo);
    }}
    var sums: array<vec2<f32>, VECTOR_TILE>;
    for (var vector = 0u; vector < VECTOR_TILE; vector = vector + 1u) {{
        sums[vector] = vec2<f32>(0.0, 0.0);
    }}
    for (var offset2 = 0u; offset2 < WIDTH; offset2 = offset2 + 1u) {{
        let unwrapped2 = start2 + i32(offset2);
        let distance2 = df64_sub(Df64(f32(unwrapped2), 0.0), position2);
        let weight2 = es_weight(distance2.hi + distance2.lo);
        let fine_index2 = wrap_index(unwrapped2, FINE2_I32);
        for (var offset1 = 0u; offset1 < WIDTH; offset1 = offset1 + 1u) {{
            let weight12 = weights1[offset1] * weight2;
            let fine_index1 = fine_indices1[offset1];
            for (var offset0 = 0u; offset0 < WIDTH; offset0 = offset0 + 1u) {{
                let fine_index0 = fine_indices0[offset0];
                let grid_index = fine_index0 + FINE0 * (fine_index1 + FINE1 * fine_index2);
                for (var vector = 0u; vector < VECTOR_TILE; vector = vector + 1u) {{
                    let batch_index = batch_start + vector;
                    if (batch_index < active_batch) {{
                        let fine_index = batch_index * FINE_TOTAL + grid_index;
                        sums[vector] = sums[vector] + fine_grid[fine_index] *
                            (weights0[offset0] * weight12);
                    }}
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
        scale0_hi = format_wgsl_f32(scale0.0),
        scale0_lo = format_wgsl_f32(scale0.1),
        scale1_hi = format_wgsl_f32(scale1.0),
        scale1_lo = format_wgsl_f32(scale1.1),
        scale2_hi = format_wgsl_f32(scale2.0),
        scale2_lo = format_wgsl_f32(scale2.1),
        origin0 = format_wgsl_f32((fine0 / 2) as f32),
        origin1 = format_wgsl_f32((fine1 / 2) as f32),
        origin2 = format_wgsl_f32((fine2 / 2) as f32),
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
    let fine2 = fine_shape[2];
    let fine_total = fine0 * fine1 * fine2;
    let fine0_f64 = format_wgsl_f64(fine0 as f64);
    let fine1_f64 = format_wgsl_f64(fine1 as f64);
    let fine2_f64 = format_wgsl_f64(fine2 as f64);
    let scale0 = format_wgsl_f64(fine0 as f64 / std::f64::consts::TAU);
    let scale1 = format_wgsl_f64(fine1 as f64 / std::f64::consts::TAU);
    let scale2 = format_wgsl_f64(fine2 as f64 / std::f64::consts::TAU);
    let origin0 = format_wgsl_f64((fine0 / 2) as f64);
    let origin1 = format_wgsl_f64((fine1 / 2) as f64);
    let origin2 = format_wgsl_f64((fine2 / 2) as f64);
    format!(
        r#"const WORKGROUP_SIZE: u32 = {WORKGROUP_SIZE}u;
const FINE0: u32 = {fine0}u;
const FINE1: u32 = {fine1}u;
const FINE2: u32 = {fine2}u;
const FINE_TOTAL: u32 = {fine_total}u;
const FINE0_I32: i32 = {fine0}i;
const FINE1_I32: i32 = {fine1}i;
const FINE2_I32: i32 = {fine2}i;
const FINE0_F64: f64 = {fine0_f64};
const FINE1_F64: f64 = {fine1_f64};
const FINE2_F64: f64 = {fine2_f64};
const VECTOR_TILE: u32 = {VECTOR_TILE}u;
const WIDTH: u32 = {width}u;
const HALF_WIDTH: f64 = {half_width};
const HORNER_COEFFICIENT_COUNT: u32 = {coefficient_count}u;
const HORNER_COEFFICIENTS: array<f64, {coefficient_total}> = array<f64, {coefficient_total}>({coefficients});
const POSITION_SCALE0: f64 = {scale0};
const POSITION_SCALE1: f64 = {scale1};
const POSITION_SCALE2: f64 = {scale2};
const GRID_ORIGIN0: f64 = {origin0};
const GRID_ORIGIN1: f64 = {origin1};
const GRID_ORIGIN2: f64 = {origin2};

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
    let point_count = arrayLength(&points) / 3u;
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

    let point_base = 3u * point_index;
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
    let position2 = fold_position(
        points[point_base + 2u],
        FINE2_F64,
        POSITION_SCALE2,
        GRID_ORIGIN2,
    );
    let start0 = ceil_f64_to_i32(position0 - HALF_WIDTH);
    let start1 = ceil_f64_to_i32(position1 - HALF_WIDTH);
    let start2 = ceil_f64_to_i32(position2 - HALF_WIDTH);
    var fine_indices0: array<u32, WIDTH>;
    var weights0: array<f64, WIDTH>;
    var fine_indices1: array<u32, WIDTH>;
    var weights1: array<f64, WIDTH>;
    for (var offset0 = 0u; offset0 < WIDTH; offset0 = offset0 + 1u) {{
        let unwrapped0 = start0 + i32(offset0);
        fine_indices0[offset0] = wrap_index(unwrapped0, FINE0_I32);
        weights0[offset0] = es_weight(f64(unwrapped0) - position0);
    }}
    for (var offset1 = 0u; offset1 < WIDTH; offset1 = offset1 + 1u) {{
        let unwrapped1 = start1 + i32(offset1);
        fine_indices1[offset1] = wrap_index(unwrapped1, FINE1_I32);
        weights1[offset1] = es_weight(f64(unwrapped1) - position1);
    }}
    var sums: array<vec2<f64>, VECTOR_TILE>;
    for (var vector = 0u; vector < VECTOR_TILE; vector = vector + 1u) {{
        sums[vector] = vec2<f64>(0.0lf, 0.0lf);
    }}
    for (var offset2 = 0u; offset2 < WIDTH; offset2 = offset2 + 1u) {{
        let unwrapped2 = start2 + i32(offset2);
        let weight2 = es_weight(f64(unwrapped2) - position2);
        let fine_index2 = wrap_index(unwrapped2, FINE2_I32);
        for (var offset1 = 0u; offset1 < WIDTH; offset1 = offset1 + 1u) {{
            let weight12 = weights1[offset1] * weight2;
            let fine_index1 = fine_indices1[offset1];
            for (var offset0 = 0u; offset0 < WIDTH; offset0 = offset0 + 1u) {{
                let grid_index = fine_indices0[offset0] + FINE0 * (fine_index1 + FINE1 * fine_index2);
                let weight = weights0[offset0] * weight12;
                for (var vector = 0u; vector < VECTOR_TILE; vector = vector + 1u) {{
                    let batch_index = batch_start + vector;
                    if (batch_index < active_batch) {{
                        let fine_index = batch_index * FINE_TOTAL + grid_index;
                        sums[vector] = sums[vector] + fine_grid[fine_index] * weight;
                    }}
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

fn maximum_supported_workgroup_size(limits: &wgpu::Limits) -> u32 {
    limits
        .max_compute_invocations_per_workgroup
        .min(limits.max_compute_workgroup_size_x)
}

fn checked_product3(
    context: &'static str,
    first: usize,
    second: usize,
    third: usize,
) -> Result<usize> {
    first
        .checked_mul(second)
        .and_then(|value| value.checked_mul(third))
        .ok_or(NufftError::LengthOverflow { context })
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
    binding_limit: u64,
    buffer_limit: u64,
) -> Result<()> {
    validate_binding_limit(buffer, bytes, binding_limit)?;
    if bytes > buffer_limit {
        return Err(NufftError::GpuBufferTooSmall {
            buffer,
            required_bytes: bytes,
            actual_bytes: buffer_limit,
        });
    }
    Ok(())
}

fn validate_external_storage_buffer(
    label: &'static str,
    buffer: &wgpu::Buffer,
    required_bytes: u64,
    binding_limit: u64,
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
    validate_binding_limit(label, required_bytes, binding_limit)
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
    fn predeconvolution_shader_uses_axis_zero_fast_three_dimensional_mapping() {
        let config = NufftConfig::new([6, 8, 10], 1.0e-6);
        let source = generate_predeconvolution_wgsl(&config, &[16, 20, 24]);
        assert!(source.contains("let fine_index0 = grid_index % FINE0;"));
        assert!(source.contains("let fine_index1 = (grid_index / FINE0) % FINE1;"));
        assert!(source.contains("let fine_index2 = grid_index / FINE01;"));
        assert!(source
            .contains("let source_index = source0.x + MODE0 * (source1.x + MODE1 * source2.x);"));
        assert!(source.contains("source2.y != 0u"));
        assert!(source.contains("fine_grid[index] = vec2<f32>(0.0, 0.0);"));
    }

    #[test]
    fn interpolation_shader_reads_xyz_aos_and_hoists_two_axes() {
        let kernel = EsKernel::for_tolerance(1.0e-6, 2.0).unwrap();
        let source = generate_interpolation_wgsl(kernel, &[16, 20, 24]);
        assert!(source.contains("let point_base = 3u * point_index;"));
        assert!(source.contains("points[point_base + 2u]"));
        assert!(source.contains("var fine_indices0: array<u32, WIDTH>;"));
        assert!(source.contains("var weights0: array<f32, WIDTH>;"));
        assert!(source.contains("var fine_indices1: array<u32, WIDTH>;"));
        assert!(source.contains("var weights1: array<f32, WIDTH>;"));
        assert!(source.contains("for (var offset2 = 0u; offset2 < WIDTH;"));
        assert!(source.contains(
            "let grid_index = fine_index0 + FINE0 * (fine_index1 + FINE1 * fine_index2);"
        ));
        assert!(source.contains("let fine_index = batch_index * FINE_TOTAL + grid_index;"));
        assert!(source.contains("weights0[offset0] * weight12"));
    }

    #[test]
    fn interpolation_shader_preserves_low_words_on_all_three_axes() {
        let kernel = EsKernel::for_tolerance(1.0e-6, 2.0).unwrap();
        let source = generate_interpolation_wgsl(kernel, &[65_536, 98_304, 131_072]);
        for axis in 0..DIMENSIONS {
            assert!(source.contains(&format!("const POSITION_SCALE{axis}_LO: f32 =")));
            assert!(source.contains(&format!("POSITION_SCALE{axis}_LO,")));
        }
        assert!(source.contains("ceil_df64_to_i32(df64_sub(position2"));
    }

    #[test]
    fn batched_shaders_use_transform_major_storage_and_reuse_xyz_weights() {
        let config = NufftConfig::new([6, 8, 10], 1.0e-6).with_batch(5);
        let predeconvolution = generate_predeconvolution_wgsl(&config, &[16, 20, 24]);
        assert!(predeconvolution.contains("const TOTAL_FINE_COUNT: u32 = 38400u;"));
        assert!(predeconvolution.contains("let batch_index = index / FINE_TOTAL;"));
        assert!(predeconvolution
            .contains("let coefficient_index = batch_index * MODE_TOTAL + source_index;"));

        let kernel = EsKernel::for_tolerance(1.0e-6, 2.0).unwrap();
        let interpolation = generate_interpolation_wgsl(kernel, &[16, 20, 24]);
        assert!(interpolation.contains("const VECTOR_TILE: u32 = 4u;"));
        assert!(interpolation.contains("let point_count = arrayLength(&points) / 3u;"));
        assert!(interpolation.contains("let batch_start = vector_tile_index * VECTOR_TILE;"));
        assert!(interpolation
            .contains("output_values[batch_index * point_count + point_index] = sums[vector];"));
    }

    #[test]
    fn native_f64_shaders_use_horner_without_transcendentals() {
        let config = NufftConfig::new([6, 8, 10], 1.0e-6).with_precision(FftPrecision::F64);
        let kernel = EsKernel::for_tolerance(1.0e-6, 2.0).unwrap();
        let sources = [
            generate_predeconvolution_wgsl(&config, &[16, 20, 24]),
            generate_interpolation_wgsl_for_precision(FftPrecision::F64, kernel, &[16, 20, 24]),
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
    fn amplitude_table_is_axis_zero_fast_tensor_product() {
        let config = NufftConfig::new([2, 2, 2], 1.0e-6).with_mode_order(ModeOrder::Fft);
        let fine_shape = [4, 4, 4];
        let coefficients = vec![
            vec![2.0, 3.0, 5.0],
            vec![7.0, 11.0, 13.0],
            vec![17.0, 19.0, 23.0],
        ];
        let amplitudes = mode_amplitudes_3d(&config, &fine_shape, &coefficients).unwrap();
        assert_eq!(amplitudes.len(), 8);
        let expected = [
            1.0 / (2.0 * 7.0 * 17.0),
            1.0 / (3.0 * 7.0 * 17.0),
            1.0 / (2.0 * 11.0 * 17.0),
            1.0 / (3.0 * 11.0 * 17.0),
            1.0 / (2.0 * 7.0 * 19.0),
            1.0 / (3.0 * 7.0 * 19.0),
            1.0 / (2.0 * 11.0 * 19.0),
            1.0 / (3.0 * 11.0 * 19.0),
        ];
        for (actual, expected) in amplitudes.into_iter().zip(expected) {
            assert_eq!(actual, expected as f32);
        }
    }

    #[test]
    fn three_dimensional_size_arithmetic_is_checked() {
        assert_eq!(checked_product3("test", 2, 3, 5).unwrap(), 30);
        assert!(matches!(
            checked_product3("test", usize::MAX, 2, 2),
            Err(NufftError::LengthOverflow { context: "test" })
        ));
        assert_eq!(
            Type2GpuPlan3d::point_buffer_size_bytes_for_precision(7, FftPrecision::F32).unwrap(),
            84
        );
        assert!(matches!(
            Type2GpuPlan3d::point_buffer_size_bytes_for_precision(usize::MAX, FftPrecision::F32),
            Err(NufftError::LengthOverflow {
                context: "type-2 three-dimensional point-coordinate count"
            })
        ));
    }
}
