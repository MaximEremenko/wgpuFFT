use std::num::NonZeroU64;
use std::sync::{Mutex, MutexGuard};

use crate::config::{ModeOrder, NufftConfig};
use crate::error::{NufftError, Result};
use crate::gpu::max_supported_workgroup_size;
use crate::gpu_dispatch::split_workgroups;
use crate::gpu_type1::Type1GpuPlan;
use crate::gpu_type1_2d::{Type1Gather2d, Type1GpuPlan2d};
use crate::gpu_type1_3d::{Type1Gather3d, Type1GpuPlan3d};
use crate::gpu_type1_nd::Type1GpuPlanNd;
use crate::kernel::{EsHornerTable, EsKernel};
use crate::plan::NufftPlan;
use crate::type3::NufftType3Plan;
use wgpu_fft::FftPrecision;

const WORKGROUP_SIZE: u32 = 64;
const PHASE_REDUCTION_STEPS: u32 = 8;
const PHASE_FINAL_CORRECTIONS: usize = 4;
const MAX_GPU_PHASE_MAGNITUDE: f64 = f32::MAX as f64 * 0.5;
// The three-word pi/2 reduction below is verified through this range. Beyond
// it, reliable native-f64 phase accuracy needs a wider Payne-Hanek table.
const MAX_NATIVE_F64_PHASE_MAGNITUDE: f64 = 1.0e6;
// The portable two-f32 reducer loses absolute phase accuracy as the unreduced
// phase grows. Exact host emulation over quadrant boundaries stays below 1e-11
// through this conservative power-of-two limit (the 2048 boundary is marginal).
const MAX_DF64_PHASE_MAGNITUDE: f64 = 1024.0;

/// GPU-resident composition of the outer spread and inner type-2 plan.
///
/// The caller supplies original source/target coordinates and complex strengths.
/// Rescaled coordinates, phased strengths, correction factors, and interpolated
/// values live in grow-only plan scratch, so repeated execution does not allocate
/// GPU buffers. Executions on one plan must retain queue order.
pub(crate) struct GpuType3Plan {
    dimensions: usize,
    batch_capacity: usize,
    precision: FftPrecision,
    raw_spread: RawSpreadPlan,
    inner_type2: NufftPlan,
    source_pipeline: wgpu::ComputePipeline,
    source_layout: wgpu::BindGroupLayout,
    target_pipeline: wgpu::ComputePipeline,
    target_layout: wgpu::BindGroupLayout,
    final_pipeline: wgpu::ComputePipeline,
    final_layout: wgpu::BindGroupLayout,
    scratch: Mutex<Type3ScratchBuffers>,
    max_workgroups_per_dimension: u32,
    max_storage_binding_bytes: u64,
    max_buffer_bytes: u64,
}

impl GpuType3Plan {
    pub(crate) fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        metadata: &NufftType3Plan,
    ) -> Result<Self> {
        validate_gpu_metadata(metadata)?;
        let dimensions = metadata.config().dimensions();
        let precision = metadata.config().precision();
        let limits = device.limits();
        let maximum_workgroup_size = max_supported_workgroup_size(&limits);
        if maximum_workgroup_size < WORKGROUP_SIZE {
            return Err(NufftError::GpuWorkgroupSizeUnsupported {
                requested: WORKGROUP_SIZE,
                maximum: maximum_workgroup_size,
            });
        }

        let outer_shape = metadata.outer_grid_shape();
        let outer_modes = outer_shape
            .iter()
            .map(|&length| {
                if !length.is_multiple_of(2) {
                    return Err(NufftError::FineGridMustBeEven { length });
                }
                Ok(length / 2)
            })
            .collect::<Result<Vec<_>>>()?;
        let outer_config = NufftConfig::new(outer_modes, metadata.config().eps())
            .with_sign(metadata.config().sign())
            .with_mode_order(ModeOrder::Centered)
            .with_sigma(metadata.config().sigma())
            .with_batch(metadata.config().batch())
            .with_precision(precision);
        let kernel = metadata.kernel();
        let outer_coefficients = if precision != FftPrecision::F32 {
            let horner = kernel.horner_table();
            outer_shape
                .iter()
                .map(|&length| kernel.centered_fourier_coefficients_horner(length, &horner))
                .collect::<Result<Vec<_>>>()?
        } else {
            outer_shape
                .iter()
                .map(|&length| kernel.centered_fourier_coefficients(length))
                .collect::<Result<Vec<_>>>()?
        };
        let raw_spread = match dimensions {
            1 => RawSpreadPlan::OneD(Type1GpuPlan::new(
                device,
                queue,
                &outer_config,
                kernel,
                outer_shape[0],
                &outer_coefficients[0],
            )?),
            2 => RawSpreadPlan::TwoD(Type1GpuPlan2d::new(
                device,
                queue,
                &outer_config,
                kernel,
                [outer_shape[0], outer_shape[1]],
                [
                    outer_coefficients[0].as_slice(),
                    outer_coefficients[1].as_slice(),
                ],
                Type1Gather2d::Tiled16,
            )?),
            3 => RawSpreadPlan::ThreeD(Type1GpuPlan3d::new(
                device,
                queue,
                &outer_config,
                kernel,
                [outer_shape[0], outer_shape[1], outer_shape[2]],
                [
                    outer_coefficients[0].as_slice(),
                    outer_coefficients[1].as_slice(),
                    outer_coefficients[2].as_slice(),
                ],
                Type1Gather3d::Tiled8x8x4,
            )?),
            _ => RawSpreadPlan::Nd(Type1GpuPlanNd::new(
                device,
                queue,
                &outer_config,
                kernel,
                &outer_shape,
                &outer_coefficients,
            )?),
        };
        if raw_spread.fine_grid_element_count() != metadata.outer_grid_count() {
            return Err(NufftError::LengthOverflow {
                context: "type-3 outer raw-spread grid element count",
            });
        }

        // The type-3 outer grid is the centered coefficient grid of a
        // complete inner type-2 NUFFT. It is not the inner plan's fine grid.
        let inner_config = NufftConfig::new(outer_shape, metadata.config().eps())
            .with_sign(metadata.config().sign())
            .with_mode_order(ModeOrder::Centered)
            .with_sigma(metadata.config().sigma())
            .with_batch(metadata.config().batch())
            .with_precision(precision);
        let inner_type2 = NufftPlan::type2_gpu(device, queue, inner_config)?;

        let source_pipeline = create_compute_pipeline(
            device,
            "wgpu_nufft.type3.source_rescale_prephase",
            &generate_source_wgsl(metadata)?,
        );
        let source_layout = source_pipeline.get_bind_group_layout(0);
        let target_pipeline = create_compute_pipeline(
            device,
            "wgpu_nufft.type3.target_rescale_correction",
            &generate_target_wgsl(metadata)?,
        );
        let target_layout = target_pipeline.get_bind_group_layout(0);
        let final_pipeline = create_compute_pipeline(
            device,
            "wgpu_nufft.type3.final_correction",
            &generate_final_wgsl(precision)?,
        );
        let final_layout = final_pipeline.get_bind_group_layout(0);

        Ok(Self {
            dimensions,
            batch_capacity: metadata.config().batch(),
            precision,
            raw_spread,
            inner_type2,
            source_pipeline,
            source_layout,
            target_pipeline,
            target_layout,
            final_pipeline,
            final_layout,
            scratch: Mutex::new(Type3ScratchBuffers::new(device, precision)),
            max_workgroups_per_dimension: limits.max_compute_workgroups_per_dimension,
            max_storage_binding_bytes: limits.max_storage_buffer_binding_size,
            max_buffer_bytes: limits.max_buffer_size,
        })
    }

    pub(crate) const fn dimensions(&self) -> usize {
        self.dimensions
    }

    pub(crate) fn required_source_point_buffer_size_bytes(
        &self,
        source_count: usize,
    ) -> Result<u64> {
        coordinate_buffer_size(
            "type-3 source point",
            source_count,
            self.dimensions,
            self.precision,
        )
    }

    #[allow(dead_code)]
    pub(crate) fn required_source_strength_buffer_size_bytes(
        &self,
        source_count: usize,
    ) -> Result<u64> {
        self.required_source_strength_buffer_size_bytes_for_batch(source_count, self.batch_capacity)
    }

    pub(crate) fn required_target_point_buffer_size_bytes(
        &self,
        target_count: usize,
    ) -> Result<u64> {
        coordinate_buffer_size(
            "type-3 target point",
            target_count,
            self.dimensions,
            self.precision,
        )
    }

    #[allow(dead_code)]
    pub(crate) fn required_output_buffer_size_bytes(&self, target_count: usize) -> Result<u64> {
        self.required_output_buffer_size_bytes_for_batch(target_count, self.batch_capacity)
    }

    fn required_source_strength_buffer_size_bytes_for_batch(
        &self,
        source_count: usize,
        active_batch: usize,
    ) -> Result<u64> {
        let count = source_count
            .checked_mul(active_batch)
            .ok_or(NufftError::LengthOverflow {
                context: "type-3 batched source strength count",
            })?;
        complex_buffer_size("type-3 source strength", count, self.precision)
    }

    fn required_output_buffer_size_bytes_for_batch(
        &self,
        target_count: usize,
        active_batch: usize,
    ) -> Result<u64> {
        let count = target_count
            .checked_mul(active_batch)
            .ok_or(NufftError::LengthOverflow {
                context: "type-3 batched output count",
            })?;
        complex_buffer_size("type-3 output", count, self.precision)
    }

    #[allow(clippy::too_many_arguments)]
    #[allow(dead_code)]
    pub(crate) fn encode(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        source_count: usize,
        source_points: &wgpu::Buffer,
        strengths: &wgpu::Buffer,
        target_count: usize,
        target_points: &wgpu::Buffer,
        output: &wgpu::Buffer,
    ) -> Result<()> {
        self.encode_batch(
            device,
            encoder,
            self.batch_capacity,
            source_count,
            source_points,
            strengths,
            target_count,
            target_points,
            output,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_batch(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        active_batch: usize,
        source_count: usize,
        source_points: &wgpu::Buffer,
        strengths: &wgpu::Buffer,
        target_count: usize,
        target_points: &wgpu::Buffer,
        output: &wgpu::Buffer,
    ) -> Result<()> {
        if active_batch == 0 || active_batch > self.batch_capacity {
            return Err(NufftError::InvalidBatch {
                actual: active_batch,
                maximum: self.batch_capacity,
            });
        }
        if target_count == 0 {
            return Ok(());
        }

        let source_point_bytes = self.required_source_point_buffer_size_bytes(source_count)?;
        let strength_bytes =
            self.required_source_strength_buffer_size_bytes_for_batch(source_count, active_batch)?;
        let target_point_bytes = self.required_target_point_buffer_size_bytes(target_count)?;
        let target_factor_bytes =
            complex_buffer_size("type-3 target factor", target_count, self.precision)?;
        let output_bytes =
            self.required_output_buffer_size_bytes_for_batch(target_count, active_batch)?;
        if source_count != 0 {
            validate_external_storage_buffer(
                "type-3 source point",
                source_points,
                source_point_bytes,
                self.max_storage_binding_bytes,
            )?;
            validate_external_storage_buffer(
                "type-3 source strength",
                strengths,
                strength_bytes,
                self.max_storage_binding_bytes,
            )?;
        }
        validate_external_storage_buffer(
            "type-3 target point",
            target_points,
            target_point_bytes,
            self.max_storage_binding_bytes,
        )?;
        validate_external_storage_buffer(
            "type-3 output",
            output,
            output_bytes,
            self.max_storage_binding_bytes,
        )?;

        let target_count_u32 =
            u32::try_from(target_count).map_err(|_| NufftError::LengthOverflow {
                context: "type-3 target count shader index space",
            })?;
        u32::try_from(source_count.checked_mul(self.dimensions).ok_or(
            NufftError::LengthOverflow {
                context: "type-3 source coordinate shader index space",
            },
        )?)
        .map_err(|_| NufftError::LengthOverflow {
            context: "type-3 source coordinate shader index space",
        })?;
        u32::try_from(target_count.checked_mul(self.dimensions).ok_or(
            NufftError::LengthOverflow {
                context: "type-3 target coordinate shader index space",
            },
        )?)
        .map_err(|_| NufftError::LengthOverflow {
            context: "type-3 target coordinate shader index space",
        })?;

        let mut scratch = self.lock_scratch();
        scratch.ensure(
            device,
            source_point_bytes,
            strength_bytes,
            target_point_bytes,
            target_factor_bytes,
            output_bytes,
            self.max_storage_binding_bytes,
            self.max_buffer_bytes,
        )?;

        if source_count != 0 {
            let source_values = source_count
                .checked_mul(active_batch)
                .and_then(|value| u32::try_from(value).ok())
                .ok_or(NufftError::LengthOverflow {
                    context: "type-3 batched source dispatch",
                })?;
            let source_dispatch =
                dispatch_for_elements(source_values, self.max_workgroups_per_dimension)?;
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("wgpu_nufft.type3.source_rescale_prephase.bind_group"),
                layout: &self.source_layout,
                entries: &[
                    binding_entry(0, source_points, source_point_bytes),
                    binding_entry(1, strengths, strength_bytes),
                    binding_entry(2, &scratch.rescaled_sources, source_point_bytes),
                    binding_entry(3, &scratch.prephased_strengths, strength_bytes),
                ],
            });
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("wgpu_nufft.type3.source_rescale_prephase.pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.source_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(source_dispatch.0, source_dispatch.1, source_dispatch.2);
        }

        self.raw_spread.encode_spread_batch(
            device,
            encoder,
            active_batch,
            source_count,
            &scratch.rescaled_sources,
            &scratch.prephased_strengths,
        )?;

        let target_dispatch =
            dispatch_for_elements(target_count_u32, self.max_workgroups_per_dimension)?;
        let target_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("wgpu_nufft.type3.target_rescale_correction.bind_group"),
            layout: &self.target_layout,
            entries: &[
                binding_entry(0, target_points, target_point_bytes),
                binding_entry(1, &scratch.rescaled_targets, target_point_bytes),
                binding_entry(2, &scratch.target_factors, target_factor_bytes),
            ],
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("wgpu_nufft.type3.target_rescale_correction.pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.target_pipeline);
            pass.set_bind_group(0, &target_bind_group, &[]);
            pass.dispatch_workgroups(target_dispatch.0, target_dispatch.1, target_dispatch.2);
        }

        self.inner_type2.encode_type2_gpu_batch(
            device,
            encoder,
            target_count,
            active_batch,
            &scratch.rescaled_targets,
            self.raw_spread.fine_grid_buffer(),
            &scratch.interpolated,
        )?;

        let final_values = target_count
            .checked_mul(active_batch)
            .and_then(|value| u32::try_from(value).ok())
            .ok_or(NufftError::LengthOverflow {
                context: "type-3 batched final-correction dispatch",
            })?;
        let final_dispatch =
            dispatch_for_elements(final_values, self.max_workgroups_per_dimension)?;
        let final_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("wgpu_nufft.type3.final_correction.bind_group"),
            layout: &self.final_layout,
            entries: &[
                binding_entry(0, &scratch.interpolated, output_bytes),
                binding_entry(1, &scratch.target_factors, target_factor_bytes),
                binding_entry(2, output, output_bytes),
            ],
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("wgpu_nufft.type3.final_correction.pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.final_pipeline);
            pass.set_bind_group(0, &final_bind_group, &[]);
            pass.dispatch_workgroups(final_dispatch.0, final_dispatch.1, final_dispatch.2);
        }
        Ok(())
    }

    fn lock_scratch(&self) -> MutexGuard<'_, Type3ScratchBuffers> {
        self.scratch
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

enum RawSpreadPlan {
    OneD(Type1GpuPlan),
    TwoD(Type1GpuPlan2d),
    ThreeD(Type1GpuPlan3d),
    Nd(Type1GpuPlanNd),
}

impl RawSpreadPlan {
    #[allow(dead_code)]
    fn encode_spread(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        point_count: usize,
        points: &wgpu::Buffer,
        strengths: &wgpu::Buffer,
    ) -> Result<()> {
        self.encode_spread_batch(device, encoder, 1, point_count, points, strengths)
    }

    fn encode_spread_batch(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        active_batch: usize,
        point_count: usize,
        points: &wgpu::Buffer,
        strengths: &wgpu::Buffer,
    ) -> Result<()> {
        match self {
            Self::OneD(plan) => plan.encode_spread_batch(
                device,
                encoder,
                active_batch,
                point_count,
                points,
                strengths,
            ),
            Self::TwoD(plan) => plan.encode_spread_batch(
                device,
                encoder,
                active_batch,
                point_count,
                points,
                strengths,
            ),
            Self::ThreeD(plan) => plan.encode_spread_batch(
                device,
                encoder,
                active_batch,
                point_count,
                points,
                strengths,
            ),
            Self::Nd(plan) => plan.encode_spread_batch(
                device,
                encoder,
                active_batch,
                point_count,
                points,
                strengths,
            ),
        }
    }

    fn fine_grid_buffer(&self) -> &wgpu::Buffer {
        match self {
            Self::OneD(plan) => plan.fine_grid_buffer(),
            Self::TwoD(plan) => plan.fine_grid_buffer(),
            Self::ThreeD(plan) => plan.fine_grid_buffer(),
            Self::Nd(plan) => plan.fine_grid_buffer(),
        }
    }

    fn fine_grid_element_count(&self) -> usize {
        match self {
            Self::OneD(plan) => plan.fine_grid_element_count(),
            Self::TwoD(plan) => plan.fine_grid_element_count(),
            Self::ThreeD(plan) => plan.fine_grid_element_count(),
            Self::Nd(plan) => plan.fine_grid_element_count(),
        }
    }
}

struct Type3ScratchBuffers {
    rescaled_sources: wgpu::Buffer,
    rescaled_source_capacity: u64,
    prephased_strengths: wgpu::Buffer,
    prephased_strength_capacity: u64,
    rescaled_targets: wgpu::Buffer,
    rescaled_target_capacity: u64,
    target_factors: wgpu::Buffer,
    target_factor_capacity: u64,
    interpolated: wgpu::Buffer,
    interpolated_capacity: u64,
}

impl Type3ScratchBuffers {
    fn new(device: &wgpu::Device, precision: FftPrecision) -> Self {
        let scalar_bytes = precision.scalar_size_bytes();
        let complex_bytes = precision.complex_size_bytes();
        Self {
            rescaled_sources: create_scratch_buffer(
                device,
                "wgpu_nufft.type3.rescaled_sources",
                scalar_bytes,
            ),
            rescaled_source_capacity: scalar_bytes,
            prephased_strengths: create_scratch_buffer(
                device,
                "wgpu_nufft.type3.prephased_strengths",
                complex_bytes,
            ),
            prephased_strength_capacity: complex_bytes,
            rescaled_targets: create_scratch_buffer(
                device,
                "wgpu_nufft.type3.rescaled_targets",
                scalar_bytes,
            ),
            rescaled_target_capacity: scalar_bytes,
            target_factors: create_scratch_buffer(
                device,
                "wgpu_nufft.type3.target_factors",
                complex_bytes,
            ),
            target_factor_capacity: complex_bytes,
            interpolated: create_scratch_buffer(
                device,
                "wgpu_nufft.type3.interpolated",
                complex_bytes,
            ),
            interpolated_capacity: complex_bytes,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn ensure(
        &mut self,
        device: &wgpu::Device,
        source_point_bytes: u64,
        strength_bytes: u64,
        target_point_bytes: u64,
        target_factor_bytes: u64,
        target_value_bytes: u64,
        max_storage_binding_bytes: u64,
        max_buffer_bytes: u64,
    ) -> Result<()> {
        for (label, bytes) in [
            ("type-3 rescaled source scratch", source_point_bytes),
            ("type-3 prephased strength scratch", strength_bytes),
            ("type-3 rescaled target scratch", target_point_bytes),
            ("type-3 target factor scratch", target_factor_bytes),
            ("type-3 interpolated value scratch", target_value_bytes),
        ] {
            validate_internal_buffer_limits(
                label,
                bytes,
                max_storage_binding_bytes,
                max_buffer_bytes,
            )?;
        }

        grow_scratch_buffer(
            device,
            "wgpu_nufft.type3.rescaled_sources",
            source_point_bytes,
            &mut self.rescaled_sources,
            &mut self.rescaled_source_capacity,
        );
        grow_scratch_buffer(
            device,
            "wgpu_nufft.type3.prephased_strengths",
            strength_bytes,
            &mut self.prephased_strengths,
            &mut self.prephased_strength_capacity,
        );
        grow_scratch_buffer(
            device,
            "wgpu_nufft.type3.rescaled_targets",
            target_point_bytes,
            &mut self.rescaled_targets,
            &mut self.rescaled_target_capacity,
        );
        grow_scratch_buffer(
            device,
            "wgpu_nufft.type3.target_factors",
            target_factor_bytes,
            &mut self.target_factors,
            &mut self.target_factor_capacity,
        );
        grow_scratch_buffer(
            device,
            "wgpu_nufft.type3.interpolated",
            target_value_bytes,
            &mut self.interpolated,
            &mut self.interpolated_capacity,
        );
        Ok(())
    }
}

fn validate_gpu_metadata(metadata: &NufftType3Plan) -> Result<()> {
    metadata.config().validate()?;
    let precision = metadata.config().precision();
    for (axis, (&source, &target)) in metadata
        .config()
        .source_bounds()
        .iter()
        .zip(metadata.config().target_bounds())
        .enumerate()
    {
        for (quantity, value) in [
            ("source interval lower endpoint", source.lower()),
            ("source interval upper endpoint", source.upper()),
            ("target interval lower endpoint", target.lower()),
            ("target interval upper endpoint", target.upper()),
        ] {
            validate_shader_constant(precision, axis, quantity, value)?;
        }
    }
    for (axis, metadata_axis) in metadata.axes().iter().copied().enumerate() {
        for (quantity, value) in [
            ("source center", metadata_axis.source_center()),
            ("target center", metadata_axis.target_center()),
            ("inverse source scale", 1.0 / metadata_axis.source_scale()),
            (
                "target rescaling factor",
                metadata_axis.grid_spacing() * metadata_axis.source_scale(),
            ),
        ] {
            validate_shader_constant(precision, axis, quantity, value)?;
        }
    }

    let source_phase_bound = metadata
        .axes()
        .iter()
        .zip(metadata.config().source_bounds())
        .map(|(axis, interval)| {
            interval.lower().abs().max(interval.upper().abs()) * axis.target_center().abs()
        })
        .sum::<f64>();
    validate_phase_bound(precision, "source pre-phase", source_phase_bound)?;
    let target_phase_bound = metadata
        .axes()
        .iter()
        .zip(metadata.config().target_bounds())
        .map(|(axis, interval)| {
            (interval.lower() - axis.target_center())
                .abs()
                .max((interval.upper() - axis.target_center()).abs())
                * axis.source_center().abs()
        })
        .sum::<f64>();
    validate_phase_bound(precision, "target post-phase", target_phase_bound)?;
    Ok(())
}

fn validate_shader_constant(
    precision: FftPrecision,
    axis: usize,
    quantity: &'static str,
    value: f64,
) -> Result<()> {
    if precision == FftPrecision::F64 {
        if value.is_finite() {
            return Ok(());
        }
        return Err(NufftError::Type3RescalingUnsupported {
            axis,
            quantity,
            value,
            reason: "the value is not representable as a finite f64 shader constant",
        });
    }
    validate_split_constant(axis, quantity, value)
}

fn validate_split_constant(axis: usize, quantity: &'static str, value: f64) -> Result<()> {
    let high = value as f32;
    let low = (value - f64::from(high)) as f32;
    if value.is_finite() && high.is_finite() && low.is_finite() {
        Ok(())
    } else {
        Err(NufftError::Type3RescalingUnsupported {
            axis,
            quantity,
            value,
            reason: "the value is not representable as a finite f32 hi/lo pair",
        })
    }
}

fn validate_phase_bound(precision: FftPrecision, quantity: &'static str, value: f64) -> Result<()> {
    let maximum = match precision {
        FftPrecision::F32 => MAX_GPU_PHASE_MAGNITUDE,
        FftPrecision::F64 => MAX_NATIVE_F64_PHASE_MAGNITUDE,
        FftPrecision::Df64 => MAX_DF64_PHASE_MAGNITUDE,
    };
    if value.is_finite() && value <= maximum {
        Ok(())
    } else {
        Err(NufftError::Type3RescalingUnsupported {
            axis: 0,
            quantity,
            value,
            reason: match precision {
                FftPrecision::F32 => {
                    "the conservative phase bound exceeds the portable f32 exponent range"
                }
                FftPrecision::F64 => {
                    "the conservative phase bound exceeds the native-f64 three-word reduction range"
                }
                FftPrecision::Df64 => {
                    "the conservative phase bound exceeds the portable-df64 accuracy range"
                }
            },
        })
    }
}

fn generate_source_wgsl(metadata: &NufftType3Plan) -> Result<String> {
    match metadata.config().precision() {
        FftPrecision::F32 => generate_source_wgsl_f32(metadata),
        FftPrecision::F64 => generate_source_wgsl_f64(metadata),
        FftPrecision::Df64 => generate_source_wgsl_df64(metadata),
    }
}

fn generate_source_wgsl_f32(metadata: &NufftType3Plan) -> Result<String> {
    let dimensions = metadata.config().dimensions();
    let sign = metadata.config().sign().isign() as f64;
    let mut constants = String::new();
    let mut body = String::new();
    for (axis, values) in metadata.axes().iter().copied().enumerate() {
        constants.push_str(&format_dd_const(
            &format!("SOURCE_CENTER_{axis}"),
            values.source_center(),
        )?);
        constants.push_str(&format_dd_const(
            &format!("SOURCE_INV_SCALE_{axis}"),
            1.0 / values.source_scale(),
        )?);
        constants.push_str(&format_dd_const(
            &format!("TARGET_CENTER_{axis}"),
            values.target_center(),
        )?);
        body.push_str(&format!(
            r#"
    let source_{axis} = source_points[coordinate_offset + {axis}u];
    let centered_{axis} = df64_sub(Df64(source_{axis}, 0.0), SOURCE_CENTER_{axis});
    let rescaled_{axis} = df64_mul(centered_{axis}, SOURCE_INV_SCALE_{axis});
    if (value_index < source_count) {{
        rescaled_sources[coordinate_offset + {axis}u] = rescaled_{axis}.hi + rescaled_{axis}.lo;
    }}
    phase = df64_add(phase, df64_mul(Df64(source_{axis}, 0.0), TARGET_CENTER_{axis}));
"#,
        ));
    }
    constants.push_str(&format_dd_const("ISIGN", sign)?);
    Ok(format!(
        r#"{df64}
const WORKGROUP_SIZE: u32 = {WORKGROUP_SIZE}u;
const DIMENSIONS: u32 = {dimensions}u;
{constants}
{phase_reduction}

@group(0) @binding(0) var<storage, read> source_points: array<f32>;
@group(0) @binding(1) var<storage, read> strengths: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> rescaled_sources: array<f32>;
@group(0) @binding(3) var<storage, read_write> prephased_strengths: array<vec2<f32>>;

@compute @workgroup_size({WORKGROUP_SIZE})
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {{
    let workgroup_flat = (wid.z * nwg.y + wid.y) * nwg.x + wid.x;
    let value_index = workgroup_flat * WORKGROUP_SIZE + lid.x;
    if (value_index >= arrayLength(&strengths)) {{ return; }}
    let source_count = arrayLength(&rescaled_sources) / DIMENSIONS;
    let source_index = value_index % source_count;
    let coordinate_offset = source_index * DIMENSIONS;
    var phase = Df64(0.0, 0.0);
{body}
    let angle = reduce_phase_df64(df64_mul(phase, ISIGN));
    let sine = sin(angle);
    let cosine = cos(angle);
    let value = strengths[value_index];
    prephased_strengths[value_index] = vec2<f32>(
        value.x * cosine - value.y * sine,
        value.x * sine + value.y * cosine,
    );
}}
"#,
        df64 = wgpu_fft::kernels::DF64_WGSL,
        phase_reduction = phase_reduction_wgsl()?,
    ))
}

fn generate_source_wgsl_f64(metadata: &NufftType3Plan) -> Result<String> {
    let dimensions = metadata.config().dimensions();
    let sign = metadata.config().sign().isign() as f64;
    let mut constants = String::new();
    let mut body = String::new();
    for (axis, values) in metadata.axes().iter().copied().enumerate() {
        constants.push_str(&format_f64_const(
            &format!("SOURCE_CENTER_{axis}"),
            values.source_center(),
        )?);
        constants.push_str(&format_f64_const(
            &format!("SOURCE_INV_SCALE_{axis}"),
            1.0 / values.source_scale(),
        )?);
        constants.push_str(&format_f64_const(
            &format!("TARGET_CENTER_{axis}"),
            values.target_center(),
        )?);
        body.push_str(&format!(
            r#"
    let source_{axis} = source_points[coordinate_offset + {axis}u];
    let centered_{axis} = source_{axis} - SOURCE_CENTER_{axis};
    let rescaled_{axis} = centered_{axis} * SOURCE_INV_SCALE_{axis};
    if (value_index < source_count) {{
        rescaled_sources[coordinate_offset + {axis}u] = rescaled_{axis};
    }}
    phase = phase + source_{axis} * TARGET_CENTER_{axis};
"#,
        ));
    }
    constants.push_str(&format_f64_const("ISIGN", sign)?);
    Ok(format!(
        r#"const WORKGROUP_SIZE: u32 = {WORKGROUP_SIZE}u;
const DIMENSIONS: u32 = {dimensions}u;
{constants}
{sincos}

@group(0) @binding(0) var<storage, read> source_points: array<f64>;
@group(0) @binding(1) var<storage, read> strengths: array<vec2<f64>>;
@group(0) @binding(2) var<storage, read_write> rescaled_sources: array<f64>;
@group(0) @binding(3) var<storage, read_write> prephased_strengths: array<vec2<f64>>;

@compute @workgroup_size({WORKGROUP_SIZE})
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {{
    let workgroup_flat = (wid.z * nwg.y + wid.y) * nwg.x + wid.x;
    let value_index = workgroup_flat * WORKGROUP_SIZE + lid.x;
    if (value_index >= arrayLength(&strengths)) {{ return; }}
    let source_count = arrayLength(&rescaled_sources) / DIMENSIONS;
    let source_index = value_index % source_count;
    let coordinate_offset = source_index * DIMENSIONS;
    var phase = 0.0lf;
{body}
    let factor = sincos_f64(phase * ISIGN);
    let value = strengths[value_index];
    prephased_strengths[value_index] = vec2<f64>(
        value.x * factor.x - value.y * factor.y,
        value.x * factor.y + value.y * factor.x,
    );
}}
"#,
        sincos = sincos_f64_wgsl(),
    ))
}

fn generate_source_wgsl_df64(metadata: &NufftType3Plan) -> Result<String> {
    let dimensions = metadata.config().dimensions();
    let sign = metadata.config().sign().isign() as f64;
    let mut constants = String::new();
    let mut body = String::new();
    for (axis, values) in metadata.axes().iter().copied().enumerate() {
        constants.push_str(&format_dd_const(
            &format!("SOURCE_CENTER_{axis}"),
            values.source_center(),
        )?);
        constants.push_str(&format_dd_const(
            &format!("SOURCE_INV_SCALE_{axis}"),
            1.0 / values.source_scale(),
        )?);
        constants.push_str(&format_dd_const(
            &format!("TARGET_CENTER_{axis}"),
            values.target_center(),
        )?);
        body.push_str(&format!(
            r#"
    let source_words_{axis} = source_points[coordinate_offset + {axis}u];
    let source_{axis} = Df64(source_words_{axis}.x, source_words_{axis}.y);
    let centered_{axis} = df64_sub(source_{axis}, SOURCE_CENTER_{axis});
    let rescaled_{axis} = df64_mul(centered_{axis}, SOURCE_INV_SCALE_{axis});
    if (value_index < source_count) {{
        rescaled_sources[coordinate_offset + {axis}u] =
            vec2<f32>(rescaled_{axis}.hi, rescaled_{axis}.lo);
    }}
    phase = df64_add(phase, df64_mul(source_{axis}, TARGET_CENTER_{axis}));
"#,
        ));
    }
    constants.push_str(&format_dd_const("ISIGN", sign)?);
    Ok(format!(
        r#"{df64}
const WORKGROUP_SIZE: u32 = {WORKGROUP_SIZE}u;
const DIMENSIONS: u32 = {dimensions}u;
{constants}
{sincos}

@group(0) @binding(0) var<storage, read> source_points: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read> strengths: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> rescaled_sources: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read_write> prephased_strengths: array<vec4<f32>>;

@compute @workgroup_size({WORKGROUP_SIZE})
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {{
    let workgroup_flat = (wid.z * nwg.y + wid.y) * nwg.x + wid.x;
    let value_index = workgroup_flat * WORKGROUP_SIZE + lid.x;
    if (value_index >= arrayLength(&strengths)) {{ return; }}
    let source_count = arrayLength(&rescaled_sources) / DIMENSIONS;
    let source_index = value_index % source_count;
    let coordinate_offset = source_index * DIMENSIONS;
    var phase = Df64(0.0, 0.0);
{body}
    let factor = sincos_df64(df64_mul(phase, ISIGN));
    prephased_strengths[value_index] =
        df64_complex_mul(strengths[value_index], factor);
}}
"#,
        df64 = wgpu_fft::kernels::DF64_WGSL,
        sincos = sincos_df64_wgsl()?,
    ))
}

fn generate_target_wgsl(metadata: &NufftType3Plan) -> Result<String> {
    match metadata.config().precision() {
        FftPrecision::F32 => generate_target_wgsl_f32(metadata),
        FftPrecision::F64 => generate_target_wgsl_f64(metadata),
        FftPrecision::Df64 => generate_target_wgsl_df64(metadata),
    }
}

fn generate_target_wgsl_f32(metadata: &NufftType3Plan) -> Result<String> {
    let dimensions = metadata.config().dimensions();
    let sign = metadata.config().sign().isign() as f64;
    let mut constants = String::new();
    let mut body = String::new();
    for (axis, values) in metadata.axes().iter().copied().enumerate() {
        constants.push_str(&format_dd_const(
            &format!("SOURCE_CENTER_{axis}"),
            values.source_center(),
        )?);
        constants.push_str(&format_dd_const(
            &format!("TARGET_CENTER_{axis}"),
            values.target_center(),
        )?);
        constants.push_str(&format_dd_const(
            &format!("TARGET_SCALE_{axis}"),
            values.grid_spacing() * values.source_scale(),
        )?);
        body.push_str(&format!(
            r#"
    let target_{axis} = target_points[coordinate_offset + {axis}u];
    let centered_{axis} = df64_sub(Df64(target_{axis}, 0.0), TARGET_CENTER_{axis});
    let rescaled_{axis} = df64_mul(centered_{axis}, TARGET_SCALE_{axis});
    rescaled_targets[coordinate_offset + {axis}u] = rescaled_{axis}.hi + rescaled_{axis}.lo;
    phase = df64_add(phase, df64_mul(centered_{axis}, SOURCE_CENTER_{axis}));
    phi_hat = df64_mul(phi_hat, outer_kernel_ft(rescaled_{axis}));
"#,
        ));
    }
    constants.push_str(&format_dd_const("ISIGN", sign)?);
    let quadrature = quadrature_constants_wgsl(metadata.kernel())?;
    Ok(format!(
        r#"{df64}
const WORKGROUP_SIZE: u32 = {WORKGROUP_SIZE}u;
const DIMENSIONS: u32 = {dimensions}u;
{constants}
{quadrature}
{phase_reduction}

fn outer_kernel_ft(frequency: Df64) -> Df64 {{
    var sum = Df64(0.0, 0.0);
    for (var node = 0u; node < KERNEL_QUADRATURE_COUNT; node = node + 1u) {{
        let angle = df64_mul(frequency, KERNEL_QUADRATURE_NODES[node]);
        let cosine = cos(angle.hi + angle.lo);
        sum = df64_add(
            sum,
            df64_mul(KERNEL_QUADRATURE_WEIGHTS[node], Df64(cosine, 0.0)),
        );
    }}
    return sum;
}}

@group(0) @binding(0) var<storage, read> target_points: array<f32>;
@group(0) @binding(1) var<storage, read_write> rescaled_targets: array<f32>;
@group(0) @binding(2) var<storage, read_write> target_factors: array<vec2<f32>>;

@compute @workgroup_size({WORKGROUP_SIZE})
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {{
    let workgroup_flat = (wid.z * nwg.y + wid.y) * nwg.x + wid.x;
    let target_index = workgroup_flat * WORKGROUP_SIZE + lid.x;
    if (target_index >= arrayLength(&target_factors)) {{ return; }}
    let coordinate_offset = target_index * DIMENSIONS;
    var phase = Df64(0.0, 0.0);
    var phi_hat = Df64(1.0, 0.0);
{body}
    let angle = reduce_phase_df64(df64_mul(phase, ISIGN));
    let inverse_phi_hat = 1.0 / (phi_hat.hi + phi_hat.lo);
    target_factors[target_index] = inverse_phi_hat * vec2<f32>(cos(angle), sin(angle));
}}
"#,
        df64 = wgpu_fft::kernels::DF64_WGSL,
        phase_reduction = phase_reduction_wgsl()?,
    ))
}

fn generate_target_wgsl_f64(metadata: &NufftType3Plan) -> Result<String> {
    let dimensions = metadata.config().dimensions();
    let sign = metadata.config().sign().isign() as f64;
    let mut constants = String::new();
    let mut body = String::new();
    for (axis, values) in metadata.axes().iter().copied().enumerate() {
        constants.push_str(&format_f64_const(
            &format!("SOURCE_CENTER_{axis}"),
            values.source_center(),
        )?);
        constants.push_str(&format_f64_const(
            &format!("TARGET_CENTER_{axis}"),
            values.target_center(),
        )?);
        constants.push_str(&format_f64_const(
            &format!("TARGET_SCALE_{axis}"),
            values.grid_spacing() * values.source_scale(),
        )?);
        body.push_str(&format!(
            r#"
    let target_{axis} = target_points[coordinate_offset + {axis}u];
    let centered_{axis} = target_{axis} - TARGET_CENTER_{axis};
    let rescaled_{axis} = centered_{axis} * TARGET_SCALE_{axis};
    rescaled_targets[coordinate_offset + {axis}u] = rescaled_{axis};
    phase = phase + centered_{axis} * SOURCE_CENTER_{axis};
    phi_hat = phi_hat * outer_kernel_ft(rescaled_{axis});
"#,
        ));
    }
    constants.push_str(&format_f64_const("ISIGN", sign)?);
    let horner = metadata.kernel().horner_table();
    let quadrature = quadrature_constants_wgsl_f64(metadata.kernel(), &horner)?;
    Ok(format!(
        r#"const WORKGROUP_SIZE: u32 = {WORKGROUP_SIZE}u;
const DIMENSIONS: u32 = {dimensions}u;
{constants}
{quadrature}
{sincos}

fn outer_kernel_ft(frequency: f64) -> f64 {{
    var sum = 0.0lf;
    for (var node = 0u; node < KERNEL_QUADRATURE_COUNT; node = node + 1u) {{
        let angle = frequency * KERNEL_QUADRATURE_NODES[node];
        let cosine = sincos_f64(angle).x;
        sum = sum + KERNEL_QUADRATURE_WEIGHTS[node] * cosine;
    }}
    return sum;
}}

@group(0) @binding(0) var<storage, read> target_points: array<f64>;
@group(0) @binding(1) var<storage, read_write> rescaled_targets: array<f64>;
@group(0) @binding(2) var<storage, read_write> target_factors: array<vec2<f64>>;

@compute @workgroup_size({WORKGROUP_SIZE})
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {{
    let workgroup_flat = (wid.z * nwg.y + wid.y) * nwg.x + wid.x;
    let target_index = workgroup_flat * WORKGROUP_SIZE + lid.x;
    if (target_index >= arrayLength(&target_factors)) {{ return; }}
    let coordinate_offset = target_index * DIMENSIONS;
    var phase = 0.0lf;
    var phi_hat = 1.0lf;
{body}
    let inverse_phi_hat = 1.0lf / phi_hat;
    target_factors[target_index] = inverse_phi_hat * sincos_f64(phase * ISIGN);
}}
"#,
        sincos = sincos_f64_wgsl(),
    ))
}

fn generate_target_wgsl_df64(metadata: &NufftType3Plan) -> Result<String> {
    let dimensions = metadata.config().dimensions();
    let sign = metadata.config().sign().isign() as f64;
    let mut constants = String::new();
    let mut body = String::new();
    for (axis, values) in metadata.axes().iter().copied().enumerate() {
        constants.push_str(&format_dd_const(
            &format!("SOURCE_CENTER_{axis}"),
            values.source_center(),
        )?);
        constants.push_str(&format_dd_const(
            &format!("TARGET_CENTER_{axis}"),
            values.target_center(),
        )?);
        constants.push_str(&format_dd_const(
            &format!("TARGET_SCALE_{axis}"),
            values.grid_spacing() * values.source_scale(),
        )?);
        body.push_str(&format!(
            r#"
    let target_words_{axis} = target_points[coordinate_offset + {axis}u];
    let target_{axis} = Df64(target_words_{axis}.x, target_words_{axis}.y);
    let centered_{axis} = df64_sub(target_{axis}, TARGET_CENTER_{axis});
    let rescaled_{axis} = df64_mul(centered_{axis}, TARGET_SCALE_{axis});
    rescaled_targets[coordinate_offset + {axis}u] =
        vec2<f32>(rescaled_{axis}.hi, rescaled_{axis}.lo);
    phase = df64_add(phase, df64_mul(centered_{axis}, SOURCE_CENTER_{axis}));
    phi_hat = df64_mul(phi_hat, outer_kernel_ft(rescaled_{axis}));
"#,
        ));
    }
    constants.push_str(&format_dd_const("ISIGN", sign)?);
    let horner = metadata.kernel().horner_table();
    let quadrature = quadrature_constants_wgsl_df64(metadata.kernel(), &horner)?;
    Ok(format!(
        r#"{df64}
const WORKGROUP_SIZE: u32 = {WORKGROUP_SIZE}u;
const DIMENSIONS: u32 = {dimensions}u;
{constants}
{quadrature}
{sincos}
{reciprocal}

fn outer_kernel_ft(frequency: Df64) -> Df64 {{
    var sum = Df64(0.0, 0.0);
    for (var node = 0u; node < KERNEL_QUADRATURE_COUNT; node = node + 1u) {{
        let angle = df64_mul(frequency, KERNEL_QUADRATURE_NODES[node]);
        let cosine = df64_complex_real(sincos_df64_small(angle));
        sum = df64_add(
            sum,
            df64_mul(KERNEL_QUADRATURE_WEIGHTS[node], cosine),
        );
    }}
    return sum;
}}

@group(0) @binding(0) var<storage, read> target_points: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> rescaled_targets: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> target_factors: array<vec4<f32>>;

@compute @workgroup_size({WORKGROUP_SIZE})
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {{
    let workgroup_flat = (wid.z * nwg.y + wid.y) * nwg.x + wid.x;
    let target_index = workgroup_flat * WORKGROUP_SIZE + lid.x;
    if (target_index >= arrayLength(&target_factors)) {{ return; }}
    let coordinate_offset = target_index * DIMENSIONS;
    var phase = Df64(0.0, 0.0);
    var phi_hat = Df64(1.0, 0.0);
{body}
    let inverse_phi_hat = df64_reciprocal(phi_hat);
    target_factors[target_index] = df64_complex_scale(
        sincos_df64(df64_mul(phase, ISIGN)),
        inverse_phi_hat,
    );
}}
"#,
        df64 = wgpu_fft::kernels::DF64_WGSL,
        sincos = sincos_df64_wgsl()?,
        reciprocal = df64_reciprocal_wgsl(),
    ))
}

fn generate_final_wgsl(precision: FftPrecision) -> Result<String> {
    match precision {
        FftPrecision::F32 => Ok(generate_final_wgsl_f32()),
        FftPrecision::F64 => Ok(generate_final_wgsl_f64()),
        FftPrecision::Df64 => Ok(generate_final_wgsl_df64()),
    }
}

fn generate_final_wgsl_f32() -> String {
    format!(
        r#"const WORKGROUP_SIZE: u32 = {WORKGROUP_SIZE}u;

@group(0) @binding(0) var<storage, read> interpolated: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read> target_factors: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> output_values: array<vec2<f32>>;

@compute @workgroup_size({WORKGROUP_SIZE})
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {{
    let workgroup_flat = (wid.z * nwg.y + wid.y) * nwg.x + wid.x;
    let index = workgroup_flat * WORKGROUP_SIZE + lid.x;
    if (index >= arrayLength(&output_values)) {{ return; }}
    let value = interpolated[index];
    let factor = target_factors[index % arrayLength(&target_factors)];
    output_values[index] = vec2<f32>(
        value.x * factor.x - value.y * factor.y,
        value.x * factor.y + value.y * factor.x,
    );
}}
"#,
    )
}

fn generate_final_wgsl_f64() -> String {
    format!(
        r#"const WORKGROUP_SIZE: u32 = {WORKGROUP_SIZE}u;

@group(0) @binding(0) var<storage, read> interpolated: array<vec2<f64>>;
@group(0) @binding(1) var<storage, read> target_factors: array<vec2<f64>>;
@group(0) @binding(2) var<storage, read_write> output_values: array<vec2<f64>>;

@compute @workgroup_size({WORKGROUP_SIZE})
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {{
    let workgroup_flat = (wid.z * nwg.y + wid.y) * nwg.x + wid.x;
    let index = workgroup_flat * WORKGROUP_SIZE + lid.x;
    if (index >= arrayLength(&output_values)) {{ return; }}
    let value = interpolated[index];
    let factor = target_factors[index % arrayLength(&target_factors)];
    output_values[index] = vec2<f64>(
        value.x * factor.x - value.y * factor.y,
        value.x * factor.y + value.y * factor.x,
    );
}}
"#,
    )
}

fn generate_final_wgsl_df64() -> String {
    format!(
        r#"{df64}
const WORKGROUP_SIZE: u32 = {WORKGROUP_SIZE}u;

@group(0) @binding(0) var<storage, read> interpolated: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> target_factors: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> output_values: array<vec4<f32>>;

@compute @workgroup_size({WORKGROUP_SIZE})
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {{
    let workgroup_flat = (wid.z * nwg.y + wid.y) * nwg.x + wid.x;
    let index = workgroup_flat * WORKGROUP_SIZE + lid.x;
    if (index >= arrayLength(&output_values)) {{ return; }}
    let factor = target_factors[index % arrayLength(&target_factors)];
    output_values[index] = df64_complex_mul(interpolated[index], factor);
}}
"#,
        df64 = wgpu_fft::kernels::DF64_WGSL,
    )
}

/// Arithmetic-only native-f64 sine/cosine. The phase first undergoes a
/// split-constant reduction and is then mapped by its nearest quadrant into
/// [-pi/4, pi/4]. Degree-17/16 odd/even polynomials keep the approximation
/// error below native-f64 rounding noise on that interval.
fn sincos_f64_wgsl() -> String {
    let mut sine_horner = format!(
        "    var sine_polynomial = {};\n",
        format_wgsl_f64(taylor_coefficient(17))
    );
    for degree in (3..=15).rev().step_by(2) {
        sine_horner.push_str(&format!(
            "    sine_polynomial = sine_polynomial * squared + {};\n",
            format_wgsl_f64(taylor_coefficient(degree))
        ));
    }
    let mut cosine_horner = format!(
        "    var cosine_polynomial = {};\n",
        format_wgsl_f64(taylor_coefficient(16))
    );
    for degree in (2..=14).rev().step_by(2) {
        cosine_horner.push_str(&format!(
            "    cosine_polynomial = cosine_polynomial * squared + {};\n",
            format_wgsl_f64(taylor_coefficient(degree))
        ));
    }
    format!(
        r#"const PHASE_HALF_PI_1: f64 = {half_pi_1};
const PHASE_HALF_PI_2: f64 = {half_pi_2};
const PHASE_HALF_PI_3: f64 = {half_pi_3};
const PHASE_INV_HALF_PI: f64 = {inverse_half_pi};

fn sincos_f64(value: f64) -> vec2<f64> {{
    let scaled_quadrant = value * PHASE_INV_HALF_PI;
    var quadrant = 0i;
    if (scaled_quadrant >= 0.0lf) {{
        quadrant = i32(scaled_quadrant + 0.5lf);
    }} else {{
        quadrant = i32(scaled_quadrant - 0.5lf);
    }}
    let quadrant_f64 = f64(quadrant);
    let angle = ((value - quadrant_f64 * PHASE_HALF_PI_1) -
        quadrant_f64 * PHASE_HALF_PI_2) - quadrant_f64 * PHASE_HALF_PI_3;
    let squared = angle * angle;
{sine_horner}    let sine = angle + angle * squared * sine_polynomial;
{cosine_horner}    let cosine = 1.0lf + squared * cosine_polynomial;

    let lane = quadrant & 3i;
    if (lane == 0i) {{ return vec2<f64>(cosine, sine); }}
    if (lane == 1i) {{ return vec2<f64>(-sine, cosine); }}
    if (lane == 2i) {{ return vec2<f64>(-cosine, -sine); }}
    return vec2<f64>(sine, -cosine);
}}
"#,
        half_pi_1 = format_wgsl_f64(1.570_796_326_734_125_6),
        half_pi_2 = format_wgsl_f64(6.077_100_506_303_966e-11),
        half_pi_3 = format_wgsl_f64(2.022_266_248_711_166_5e-21),
        inverse_half_pi = format_wgsl_f64(2.0 / std::f64::consts::PI),
    )
}

/// Portable double-float sine/cosine using only the error-controlled dd
/// primitives, integer quadrant selection, and Horner polynomials. A bounded
/// iterative tau reduction brings the validated phase range down to where the
/// nearest pi/2 quadrant is an exact small integer.
fn sincos_df64_wgsl() -> Result<String> {
    let mut sine_horner = format!(
        "    var sine_polynomial = {};\n",
        format_dd_value(taylor_coefficient(17))?
    );
    for degree in (3..=15).rev().step_by(2) {
        sine_horner.push_str(&format!(
            "    sine_polynomial = df64_add(\n        df64_mul(sine_polynomial, squared),\n        {},\n    );\n",
            format_dd_value(taylor_coefficient(degree))?
        ));
    }
    let mut cosine_horner = format!(
        "    var cosine_polynomial = {};\n",
        format_dd_value(taylor_coefficient(16))?
    );
    for degree in (2..=14).rev().step_by(2) {
        cosine_horner.push_str(&format!(
            "    cosine_polynomial = df64_add(\n        df64_mul(cosine_polynomial, squared),\n        {},\n    );\n",
            format_dd_value(taylor_coefficient(degree))?
        ));
    }
    Ok(format!(
        r#"const DF64_PHASE_TAU: Df64 = {tau};
const DF64_PHASE_INV_TAU: Df64 = {inverse_tau};
const DF64_PHASE_HALF_PI_1: Df64 = {half_pi_1};
const DF64_PHASE_HALF_PI_2: Df64 = {half_pi_2};
const DF64_PHASE_HALF_PI_3: Df64 = {half_pi_3};
const DF64_PHASE_INV_HALF_PI: Df64 = {inverse_half_pi};
const DF64_PHASE_REDUCTION_STEPS: u32 = {PHASE_REDUCTION_STEPS}u;

fn df64_less_than_zero(value: Df64) -> bool {{
    return value.hi < 0.0 || (value.hi == 0.0 && value.lo < 0.0);
}}

fn reduce_phase_df64_full(value: Df64) -> Df64 {{
    var reduced = value;
    for (var step = 0u; step < DF64_PHASE_REDUCTION_STEPS; step = step + 1u) {{
        let quotient = df64_mul(reduced, DF64_PHASE_INV_TAU);
        let integer_quotient = trunc(quotient.hi);
        reduced = df64_sub(
            reduced,
            df64_mul(Df64(integer_quotient, 0.0), DF64_PHASE_TAU),
        );
    }}
{corrections}    return reduced;
}}

fn sincos_df64_reduced(reduced: Df64) -> vec4<f32> {{
    let scaled_quadrant = df64_mul(reduced, DF64_PHASE_INV_HALF_PI);
    var shifted_quadrant = Df64(0.0, 0.0);
    if (df64_less_than_zero(scaled_quadrant)) {{
        shifted_quadrant = df64_sub(scaled_quadrant, Df64(0.5, 0.0));
    }} else {{
        shifted_quadrant = df64_add(scaled_quadrant, Df64(0.5, 0.0));
    }}
    let quadrant = i32(trunc(shifted_quadrant.hi));
    let quadrant_df64 = Df64(f32(quadrant), 0.0);
    var angle = df64_sub(
        reduced,
        df64_mul(quadrant_df64, DF64_PHASE_HALF_PI_1),
    );
    angle = df64_sub(
        angle,
        df64_mul(quadrant_df64, DF64_PHASE_HALF_PI_2),
    );
    angle = df64_sub(
        angle,
        df64_mul(quadrant_df64, DF64_PHASE_HALF_PI_3),
    );
    let squared = df64_mul(angle, angle);
{sine_horner}    let sine_tail = df64_mul(
        df64_mul(angle, squared),
        sine_polynomial,
    );
    let sine = df64_add(angle, sine_tail);
{cosine_horner}    let cosine = df64_add(
        Df64(1.0, 0.0),
        df64_mul(squared, cosine_polynomial),
    );

    let lane = quadrant & 3i;
    if (lane == 0i) {{ return df64_complex_pack(cosine, sine); }}
    if (lane == 1i) {{ return df64_complex_pack(df64_neg(sine), cosine); }}
    if (lane == 2i) {{
        return df64_complex_pack(df64_neg(cosine), df64_neg(sine));
    }}
    return df64_complex_pack(sine, df64_neg(cosine));
}}

fn sincos_df64_small(value: Df64) -> vec4<f32> {{
    return sincos_df64_reduced(value);
}}

fn sincos_df64(value: Df64) -> vec4<f32> {{
    return sincos_df64_reduced(reduce_phase_df64_full(value));
}}
"#,
        tau = format_dd_value(std::f64::consts::TAU)?,
        inverse_tau = format_dd_value(1.0 / std::f64::consts::TAU)?,
        half_pi_1 = format_dd_value(1.570_796_326_734_125_6)?,
        half_pi_2 = format_dd_value(6.077_100_506_303_966e-11)?,
        half_pi_3 = format_dd_value(2.022_266_248_711_166_5e-21)?,
        inverse_half_pi = format_dd_value(2.0 / std::f64::consts::PI)?,
        corrections = (0..PHASE_FINAL_CORRECTIONS)
            .map(|_| {
                r#"    if (reduced.hi > DF64_PHASE_TAU.hi ||
        (reduced.hi == DF64_PHASE_TAU.hi && reduced.lo >= DF64_PHASE_TAU.lo)) {
        reduced = df64_sub(reduced, DF64_PHASE_TAU);
    }
    if (reduced.hi < -DF64_PHASE_TAU.hi ||
        (reduced.hi == -DF64_PHASE_TAU.hi && reduced.lo <= -DF64_PHASE_TAU.lo)) {
        reduced = df64_add(reduced, DF64_PHASE_TAU);
    }
"#
            })
            .collect::<String>(),
    ))
}

fn df64_reciprocal_wgsl() -> &'static str {
    r#"fn df64_reciprocal(value: Df64) -> Df64 {
    var estimate = Df64(1.0 / value.hi, 0.0);
    for (var iteration = 0u; iteration < 2u; iteration = iteration + 1u) {
        let correction = df64_sub(
            Df64(2.0, 0.0),
            df64_mul(value, estimate),
        );
        estimate = df64_mul(estimate, correction);
    }
    return estimate;
}
"#
}

fn taylor_coefficient(degree: usize) -> f64 {
    let sign = if (degree / 2).is_multiple_of(2) {
        1.0
    } else {
        -1.0
    };
    sign / (1..=degree).map(|factor| factor as f64).product::<f64>()
}

fn phase_reduction_wgsl() -> Result<String> {
    let tau = format_dd_value(std::f64::consts::TAU)?;
    let inverse_tau = format_dd_value(1.0 / std::f64::consts::TAU)?;
    Ok(format!(
        r#"const PHASE_TAU: Df64 = {tau};
const PHASE_INV_TAU: Df64 = {inverse_tau};
const PHASE_REDUCTION_STEPS: u32 = {PHASE_REDUCTION_STEPS}u;

fn reduce_phase_df64(value: Df64) -> f32 {{
    var reduced = value;
    for (var step = 0u; step < PHASE_REDUCTION_STEPS; step = step + 1u) {{
        let quotient = df64_mul(reduced, PHASE_INV_TAU);
        let integer_quotient = trunc(quotient.hi);
        reduced = df64_sub(
            reduced,
            df64_mul(Df64(integer_quotient, 0.0), PHASE_TAU),
        );
    }}
{corrections}
    return reduced.hi + reduced.lo;
}}
"#,
        corrections = (0..PHASE_FINAL_CORRECTIONS)
            .map(|_| {
                r#"    if (reduced.hi > PHASE_TAU.hi ||
        (reduced.hi == PHASE_TAU.hi && reduced.lo >= PHASE_TAU.lo)) {
        reduced = df64_sub(reduced, PHASE_TAU);
    }
    if (reduced.hi < -PHASE_TAU.hi ||
        (reduced.hi == -PHASE_TAU.hi && reduced.lo <= -PHASE_TAU.lo)) {
        reduced = df64_add(reduced, PHASE_TAU);
    }
"#
            })
            .collect::<String>(),
    ))
}

fn quadrature_constants_wgsl(kernel: EsKernel) -> Result<String> {
    let count = kernel.width() + 2;
    let (nodes, weights) = gauss_legendre(count * 2);
    let half_width = kernel.half_width();
    let mut node_values = Vec::with_capacity(count);
    let mut weight_values = Vec::with_capacity(count);
    for (&node, &weight) in nodes[count..].iter().zip(&weights[count..]) {
        let distance = half_width * node;
        node_values.push(format_dd_value(distance)?);
        weight_values.push(format_dd_value(
            2.0 * half_width * weight * kernel.evaluate(distance),
        )?);
    }
    Ok(format!(
        r#"const KERNEL_QUADRATURE_COUNT: u32 = {count}u;
const KERNEL_QUADRATURE_NODES: array<Df64, {count}> = array<Df64, {count}>(
    {nodes}
);
const KERNEL_QUADRATURE_WEIGHTS: array<Df64, {count}> = array<Df64, {count}>(
    {weights}
);
"#,
        nodes = node_values.join(",\n    "),
        weights = weight_values.join(",\n    "),
    ))
}

fn quadrature_constants_wgsl_f64(kernel: EsKernel, horner: &EsHornerTable) -> Result<String> {
    let count = kernel.width() + 2;
    let (nodes, weights) = gauss_legendre(count * 2);
    let half_width = kernel.half_width();
    let mut node_values = Vec::with_capacity(count);
    let mut weight_values = Vec::with_capacity(count);
    for (&node, &weight) in nodes[count..].iter().zip(&weights[count..]) {
        let distance = half_width * node;
        node_values.push(format_wgsl_f64(distance));
        weight_values.push(format_wgsl_f64(
            2.0 * half_width * weight * horner.evaluate(distance),
        ));
    }
    Ok(format!(
        r#"const KERNEL_QUADRATURE_COUNT: u32 = {count}u;
const KERNEL_QUADRATURE_NODES: array<f64, {count}> = array<f64, {count}>(
    {nodes}
);
const KERNEL_QUADRATURE_WEIGHTS: array<f64, {count}> = array<f64, {count}>(
    {weights}
);
"#,
        nodes = node_values.join(",\n    "),
        weights = weight_values.join(",\n    "),
    ))
}

fn quadrature_constants_wgsl_df64(kernel: EsKernel, horner: &EsHornerTable) -> Result<String> {
    let count = kernel.width() + 2;
    let (nodes, weights) = gauss_legendre(count * 2);
    let half_width = kernel.half_width();
    let mut node_values = Vec::with_capacity(count);
    let mut weight_values = Vec::with_capacity(count);
    for (&node, &weight) in nodes[count..].iter().zip(&weights[count..]) {
        let distance = half_width * node;
        node_values.push(format_dd_value(distance)?);
        weight_values.push(format_dd_value(
            2.0 * half_width * weight * horner.evaluate(distance),
        )?);
    }
    Ok(format!(
        r#"const KERNEL_QUADRATURE_COUNT: u32 = {count}u;
const KERNEL_QUADRATURE_NODES: array<Df64, {count}> = array<Df64, {count}>(
    {nodes}
);
const KERNEL_QUADRATURE_WEIGHTS: array<Df64, {count}> = array<Df64, {count}>(
    {weights}
);
"#,
        nodes = node_values.join(",\n    "),
        weights = weight_values.join(",\n    "),
    ))
}

fn format_dd_const(name: &str, value: f64) -> Result<String> {
    Ok(format!(
        "const {name}: Df64 = {};\n",
        format_dd_value(value)?
    ))
}

fn format_f64_const(name: &str, value: f64) -> Result<String> {
    if !value.is_finite() {
        return Err(NufftError::Type3RescalingUnsupported {
            axis: 0,
            quantity: "shader constant",
            value,
            reason: "the value is not representable as a finite f64 shader constant",
        });
    }
    Ok(format!("const {name}: f64 = {};\n", format_wgsl_f64(value)))
}

fn format_dd_value(value: f64) -> Result<String> {
    let high = value as f32;
    let low = (value - f64::from(high)) as f32;
    if !value.is_finite() || !high.is_finite() || !low.is_finite() {
        return Err(NufftError::Type3RescalingUnsupported {
            axis: 0,
            quantity: "shader constant",
            value,
            reason: "the value is not representable as a finite f32 hi/lo pair",
        });
    }
    Ok(format!("Df64({}, {})", format_f32(high), format_f32(low)))
}

fn format_f32(value: f32) -> String {
    if value == 0.0 {
        if value.is_sign_negative() {
            "-0.0".to_owned()
        } else {
            "0.0".to_owned()
        }
    } else {
        format!("{value:.9e}")
    }
}

fn format_wgsl_f64(value: f64) -> String {
    assert!(value.is_finite(), "WGSL f64 constants must be finite");
    let mut formatted = value.to_string();
    if !formatted.contains('.') && !formatted.contains('e') && !formatted.contains('E') {
        formatted.push_str(".0");
    }
    formatted.push_str("lf");
    formatted
}

fn gauss_legendre(order: usize) -> (Vec<f64>, Vec<f64>) {
    let mut nodes = vec![0.0; order];
    let mut weights = vec![0.0; order];
    for index in 0..order.div_ceil(2) {
        let mut root = (std::f64::consts::PI * (index as f64 + 0.75) / (order as f64 + 0.5)).cos();
        loop {
            let (polynomial, previous) = legendre_pair(order, root);
            let derivative = order as f64 * (root * polynomial - previous) / (root * root - 1.0);
            let next = root - polynomial / derivative;
            if (next - root).abs() <= 4.0 * f64::EPSILON {
                root = next;
                break;
            }
            root = next;
        }
        let (polynomial, previous) = legendre_pair(order, root);
        let derivative = order as f64 * (root * polynomial - previous) / (root * root - 1.0);
        let weight = 2.0 / ((1.0 - root * root) * derivative * derivative);
        nodes[index] = -root;
        nodes[order - 1 - index] = root;
        weights[index] = weight;
        weights[order - 1 - index] = weight;
    }
    (nodes, weights)
}

fn legendre_pair(order: usize, x: f64) -> (f64, f64) {
    let mut previous = 1.0;
    if order == 0 {
        return (previous, 0.0);
    }
    let mut current = x;
    for degree in 2..=order {
        let next = ((2 * degree - 1) as f64 * x * current - (degree - 1) as f64 * previous)
            / degree as f64;
        previous = current;
        current = next;
    }
    (current, previous)
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

fn create_scratch_buffer(device: &wgpu::Device, label: &str, size: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    })
}

fn grow_scratch_buffer(
    device: &wgpu::Device,
    label: &str,
    required_bytes: u64,
    buffer: &mut wgpu::Buffer,
    capacity: &mut u64,
) {
    if required_bytes > *capacity {
        *buffer = create_scratch_buffer(device, label, required_bytes);
        *capacity = required_bytes;
    }
}

fn coordinate_buffer_size(
    context: &'static str,
    point_count: usize,
    dimensions: usize,
    precision: FftPrecision,
) -> Result<u64> {
    let elements = point_count
        .checked_mul(dimensions)
        .ok_or(NufftError::LengthOverflow { context })?;
    checked_buffer_size(context, elements, precision.scalar_size_bytes())
}

fn complex_buffer_size(
    context: &'static str,
    element_count: usize,
    precision: FftPrecision,
) -> Result<u64> {
    checked_buffer_size(context, element_count, precision.complex_size_bytes())
}

fn checked_buffer_size(
    context: &'static str,
    element_count: usize,
    element_bytes: u64,
) -> Result<u64> {
    u64::try_from(element_count)
        .ok()
        .and_then(|count| count.checked_mul(element_bytes))
        .ok_or(NufftError::LengthOverflow { context })
}

fn validate_external_storage_buffer(
    label: &'static str,
    buffer: &wgpu::Buffer,
    required_bytes: u64,
    max_storage_binding_bytes: u64,
) -> Result<()> {
    if buffer.size() < required_bytes {
        return Err(NufftError::GpuBufferTooSmall {
            buffer: label,
            required_bytes,
            actual_bytes: buffer.size(),
        });
    }
    if !buffer.usage().contains(wgpu::BufferUsages::STORAGE) {
        return Err(NufftError::GpuBufferMissingUsage {
            buffer: label,
            required_usage: "STORAGE",
        });
    }
    if required_bytes > max_storage_binding_bytes {
        return Err(NufftError::GpuBufferBindingTooLarge {
            buffer: label,
            required_bytes,
            limit_bytes: max_storage_binding_bytes,
        });
    }
    Ok(())
}

fn validate_internal_buffer_limits(
    label: &'static str,
    required_bytes: u64,
    max_storage_binding_bytes: u64,
    max_buffer_bytes: u64,
) -> Result<()> {
    if required_bytes > max_storage_binding_bytes {
        return Err(NufftError::GpuBufferBindingTooLarge {
            buffer: label,
            required_bytes,
            limit_bytes: max_storage_binding_bytes,
        });
    }
    if required_bytes > max_buffer_bytes {
        return Err(NufftError::GpuBufferTooSmall {
            buffer: label,
            required_bytes,
            actual_bytes: max_buffer_bytes,
        });
    }
    Ok(())
}

fn dispatch_for_elements(elements: u32, max_per_dimension: u32) -> Result<(u32, u32, u32)> {
    split_workgroups(elements.div_ceil(WORKGROUP_SIZE), max_per_dimension)
}

fn binding_entry(binding: u32, buffer: &wgpu::Buffer, size: u64) -> wgpu::BindGroupEntry<'_> {
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
#[cfg_attr(target_arch = "wasm32", allow(dead_code))]
mod tests {
    use super::*;
    use crate::type3::{es_kernel_fourier_transform, NufftInterval, NufftType3Config};
    use std::mem::ManuallyDrop;
    use std::sync::mpsc;
    use wgpu::util::DeviceExt;
    use wgpu_fft::math::{ComplexDoubleFloat, DoubleFloat};

    fn metadata(dimensions: usize) -> NufftType3Plan {
        metadata_with_precision(dimensions, FftPrecision::F32)
    }

    fn metadata_with_precision(dimensions: usize, precision: FftPrecision) -> NufftType3Plan {
        NufftType3Plan::new(
            NufftType3Config::new(
                (0..dimensions)
                    .map(|axis| NufftInterval::new(-1.5 - axis as f64, 2.0 + axis as f64))
                    .collect::<Vec<_>>(),
                (0..dimensions)
                    .map(|axis| NufftInterval::new(-3.0 - axis as f64, 4.0 + axis as f64))
                    .collect::<Vec<_>>(),
                1.0e-6,
            )
            .with_precision(precision),
        )
        .unwrap()
    }

    #[test]
    fn generated_shaders_cover_every_axis_and_use_df64_phase_reduction() {
        for dimensions in 1..=4 {
            let metadata = metadata(dimensions);
            let source = generate_source_wgsl(&metadata).unwrap();
            let target = generate_target_wgsl(&metadata).unwrap();
            for axis in 0..dimensions {
                assert!(source.contains(&format!("SOURCE_INV_SCALE_{axis}")));
                assert!(target.contains(&format!("TARGET_SCALE_{axis}")));
                assert!(target.contains(&format!("rescaled_{axis}")));
            }
            for shader in [&source, &target] {
                assert!(shader.contains(wgpu_fft::kernels::DF64_WGSL));
                assert!(shader.contains("step < PHASE_REDUCTION_STEPS"));
                assert!(shader.contains("const PHASE_REDUCTION_STEPS: u32 = 8u"));
                assert!(shader.contains("let integer_quotient = trunc(quotient.hi);"));
                assert!(shader.contains("df64_mul(Df64(integer_quotient, 0.0), PHASE_TAU)"));
            }
        }
    }

    #[test]
    fn native_f64_type3_shaders_are_transcendental_free() {
        for dimensions in 1..=4 {
            let metadata = metadata_with_precision(dimensions, FftPrecision::F64);
            let source = generate_source_wgsl(&metadata).unwrap();
            let target = generate_target_wgsl(&metadata).unwrap();
            let final_correction = generate_final_wgsl(FftPrecision::F64).unwrap();
            assert!(source.contains("array<f64>"));
            assert!(source.contains("array<vec2<f64>>"));
            assert!(target.contains("fn outer_kernel_ft(frequency: f64) -> f64"));
            assert!(target.contains("sincos_f64(angle).x"));
            assert!(final_correction.contains("array<vec2<f64>>"));
            for shader in [&source, &target, &final_correction] {
                assert!(!shader.contains("enable f64"));
                for forbidden in ["exp(", "log(", "pow(", "sin(", "cos("] {
                    assert!(
                        !shader.contains(forbidden),
                        "native-f64 shader unexpectedly contains {forbidden}:\n{shader}",
                    );
                }
            }
        }
    }

    #[test]
    fn portable_df64_type3_shaders_use_dd_horner_without_transcendentals() {
        for dimensions in 1..=4 {
            let metadata = metadata_with_precision(dimensions, FftPrecision::Df64);
            let source = generate_source_wgsl(&metadata).unwrap();
            let target = generate_target_wgsl(&metadata).unwrap();
            let final_correction = generate_final_wgsl(FftPrecision::Df64).unwrap();
            assert!(source.contains("source_points: array<vec2<f32>>"));
            assert!(source.contains("strengths: array<vec4<f32>>"));
            assert!(source.contains("sincos_df64"));
            assert!(source.contains("reduce_phase_df64_full"));
            assert!(target.contains("fn outer_kernel_ft(frequency: Df64) -> Df64"));
            assert!(target.contains("KERNEL_QUADRATURE_NODES: array<Df64"));
            assert!(target.contains("KERNEL_QUADRATURE_WEIGHTS: array<Df64"));
            assert!(target.contains("for (var node = 0u; node < KERNEL_QUADRATURE_COUNT"));
            // Quadrature angles have a tight host-proved bound, so keep their
            // path free of the nested tau-reduction loop that optimized DX12
            // miscompiled with the outer node loop.
            assert!(target.contains("sincos_df64_small(angle)"));
            assert!(target.contains("fn sincos_df64_small(value: Df64)"));
            assert!(target.contains("return sincos_df64_reduced(value);"));
            assert!(target.contains("return sincos_df64_reduced(reduce_phase_df64_full(value));"));
            assert!(target.contains("df64_reciprocal(phi_hat)"));
            assert!(final_correction.contains("df64_complex_mul"));
            for shader in [&source, &target, &final_correction] {
                assert!(!shader.contains("enable f64"));
                assert!(!shader.contains("array<f64>"));
                assert!(!shader.contains("vec2<f64>"));
                for forbidden in ["exp(", "log(", "pow(", "sin(", "cos("] {
                    assert!(
                        !shader.contains(forbidden),
                        "df64 shader unexpectedly contains {forbidden}:\n{shader}",
                    );
                }
            }
        }
    }

    #[test]
    fn df64_quadrature_angles_fit_the_direct_quadrant_reducer() {
        for (eps, sigma) in [(1.0e-15, 2.0), (1.0e-2, 1.125), (1.0e-6, 3.0)] {
            let metadata = NufftType3Plan::new(
                NufftType3Config::new(
                    [NufftInterval::new(-1.7, 2.3)],
                    [NufftInterval::new(-3.7, 4.1)],
                    eps,
                )
                .with_sigma(sigma)
                .with_precision(FftPrecision::Df64),
            )
            .unwrap();
            let axis = metadata.axes()[0];
            let frequency_bound =
                axis.grid_spacing() * axis.source_scale() * axis.target_half_width();
            let planned_bound = std::f64::consts::PI / sigma;
            assert!(
                frequency_bound <= planned_bound * (1.0 + 8.0 * f64::EPSILON),
                "rescaled target bound {frequency_bound} exceeds pi/sigma {planned_bound}"
            );
            let angle_bound = frequency_bound * metadata.kernel().half_width();
            assert!(angle_bound < 8.0 * std::f64::consts::PI);
            assert!(angle_bound * (2.0 / std::f64::consts::PI) < 16.0);
        }
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn gpu_df64_type3_source_and_target_arithmetic_stages_are_finite() {
        if std::env::var_os("WGPU_FFT_RUN_GPU_TESTS").is_none() {
            eprintln!("skipping GPU test; set WGPU_FFT_RUN_GPU_TESTS=1 to run it");
            return;
        }
        pollster::block_on(run_df64_type3_stage_probe());
    }

    async fn run_df64_type3_stage_probe() {
        let config = NufftType3Config::new(
            [NufftInterval::new(-0.55, 0.95)],
            [NufftInterval::new(-2.05, 0.75)],
            1.0e-8,
        )
        .with_sign(crate::config::NufftSign::Negative)
        .with_batch(2)
        .with_precision(FftPrecision::Df64);
        let metadata = NufftType3Plan::new(config).unwrap();
        validate_gpu_metadata(&metadata).unwrap();

        let Some(context) = wgpu_fft::device::request_default_device().await else {
            eprintln!("skipping GPU test; no suitable adapter was found");
            return;
        };
        let context = ManuallyDrop::new(context);
        let info = context.adapter.get_info();
        let (device, queue) = context
            .adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("wgpu_nufft.type3_df64_stage_probe.device"),
                required_features: wgpu::Features::empty(),
                required_limits: context.adapter.limits(),
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            })
            .await
            .unwrap();
        let device_and_queue = ManuallyDrop::new((device, queue));
        let device = &device_and_queue.0;
        let queue = &device_and_queue.1;
        eprintln!(
            "TYPE3_DF64_STAGE_PROBE adapter={:?} backend={:?} driver={:?} {}",
            info.name, info.backend, info.driver, info.driver_info
        );

        let source_values = [-0.55 + 1.0e-10, 0.95 - 1.0e-10, 0.2, -0.125];
        let target_values = [-2.05 + 1.0e-10, 0.75 - 1.0e-10, -0.65, 0.375];
        let source_points = source_values
            .iter()
            .copied()
            .map(DoubleFloat::from_f64)
            .collect::<Vec<_>>();
        let target_points = target_values
            .iter()
            .copied()
            .map(DoubleFloat::from_f64)
            .collect::<Vec<_>>();
        let strengths = (0..2)
            .flat_map(|batch| {
                (0..source_values.len()).map(move |index| {
                    let re = 0.17 + 0.11 * index as f64 + 0.07 * batch as f64;
                    let im = -0.23 + 0.09 * index as f64 - 0.05 * batch as f64;
                    ComplexDoubleFloat::from_f64(re, im)
                })
            })
            .collect::<Vec<_>>();
        let source_points_buffer = storage_buffer_init(
            device,
            "wgpu_nufft.type3_df64_probe.source_points",
            &source_points,
        );
        let strengths_buffer =
            storage_buffer_init(device, "wgpu_nufft.type3_df64_probe.strengths", &strengths);
        let target_points_buffer = storage_buffer_init(
            device,
            "wgpu_nufft.type3_df64_probe.target_points",
            &target_points,
        );
        let rescaled_sources = storage_output_buffer::<DoubleFloat>(
            device,
            "wgpu_nufft.type3_df64_probe.rescaled_sources",
            source_values.len(),
        );
        let prephased_strengths = storage_output_buffer::<ComplexDoubleFloat>(
            device,
            "wgpu_nufft.type3_df64_probe.prephased_strengths",
            strengths.len(),
        );
        let rescaled_targets = storage_output_buffer::<DoubleFloat>(
            device,
            "wgpu_nufft.type3_df64_probe.rescaled_targets",
            target_values.len(),
        );
        let target_factors = storage_output_buffer::<ComplexDoubleFloat>(
            device,
            "wgpu_nufft.type3_df64_probe.target_factors",
            target_values.len(),
        );

        let source_pipeline = create_compute_pipeline(
            device,
            "wgpu_nufft.type3_df64_probe.source",
            &generate_source_wgsl(&metadata).unwrap(),
        );
        let source_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("wgpu_nufft.type3_df64_probe.source.bind_group"),
            layout: &source_pipeline.get_bind_group_layout(0),
            entries: &[
                entire_binding_entry(0, &source_points_buffer),
                entire_binding_entry(1, &strengths_buffer),
                entire_binding_entry(2, &rescaled_sources),
                entire_binding_entry(3, &prephased_strengths),
            ],
        });
        let target_pipeline = create_compute_pipeline(
            device,
            "wgpu_nufft.type3_df64_probe.target",
            &generate_target_wgsl(&metadata).unwrap(),
        );
        let target_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("wgpu_nufft.type3_df64_probe.target.bind_group"),
            layout: &target_pipeline.get_bind_group_layout(0),
            entries: &[
                entire_binding_entry(0, &target_points_buffer),
                entire_binding_entry(1, &rescaled_targets),
                entire_binding_entry(2, &target_factors),
            ],
        });

        let source_readback = readback_buffer(device, rescaled_sources.size(), "source");
        let prephase_readback = readback_buffer(device, prephased_strengths.size(), "prephase");
        let target_readback = readback_buffer(device, rescaled_targets.size(), "target");
        let factor_readback = readback_buffer(device, target_factors.size(), "factor");
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("wgpu_nufft.type3_df64_probe.encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("wgpu_nufft.type3_df64_probe.source.pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&source_pipeline);
            pass.set_bind_group(0, &source_bind_group, &[]);
            pass.dispatch_workgroups(
                strengths.len().div_ceil(WORKGROUP_SIZE as usize) as u32,
                1,
                1,
            );
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("wgpu_nufft.type3_df64_probe.target.pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&target_pipeline);
            pass.set_bind_group(0, &target_bind_group, &[]);
            pass.dispatch_workgroups(
                target_values.len().div_ceil(WORKGROUP_SIZE as usize) as u32,
                1,
                1,
            );
        }
        for (source, destination) in [
            (&rescaled_sources, &source_readback),
            (&prephased_strengths, &prephase_readback),
            (&rescaled_targets, &target_readback),
            (&target_factors, &factor_readback),
        ] {
            encoder.copy_buffer_to_buffer(source, 0, destination, 0, source.size());
        }
        let submission = queue.submit([encoder.finish()]);
        let readbacks = [
            &source_readback,
            &prephase_readback,
            &target_readback,
            &factor_readback,
        ];
        let mut receivers = Vec::with_capacity(readbacks.len());
        for buffer in readbacks {
            let (sender, receiver) = mpsc::channel();
            buffer
                .slice(..)
                .map_async(wgpu::MapMode::Read, move |result| {
                    sender.send(result).unwrap();
                });
            receivers.push(receiver);
        }
        device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(submission),
                timeout: None,
            })
            .unwrap();
        for receiver in receivers {
            receiver.recv().unwrap().unwrap();
        }
        let actual_sources = mapped_values::<DoubleFloat>(&source_readback);
        let actual_prephase = mapped_values::<ComplexDoubleFloat>(&prephase_readback);
        let actual_targets = mapped_values::<DoubleFloat>(&target_readback);
        let actual_factors = mapped_values::<ComplexDoubleFloat>(&factor_readback);

        let axis = metadata.axes()[0];
        let sign = metadata.config().sign().isign() as f64;
        for (index, (&input, actual)) in source_values.iter().zip(&actual_sources).enumerate() {
            let value = actual.to_f64();
            eprintln!("TYPE3_DF64_STAGE source[{index}]={value:.17e}");
            assert!(value.is_finite(), "rescaled source {index} is non-finite");
            assert!((value - axis.rescale_source(input)).abs() <= 2.0e-13);
        }
        for (index, actual) in actual_prephase.iter().copied().enumerate() {
            let source_index = index % source_values.len();
            let phase = sign * source_values[source_index] * axis.target_center();
            let input = strengths[index];
            let expected_re = input.re().to_f64() * phase.cos() - input.im().to_f64() * phase.sin();
            let expected_im = input.re().to_f64() * phase.sin() + input.im().to_f64() * phase.cos();
            let re = actual.re().to_f64();
            let im = actual.im().to_f64();
            eprintln!("TYPE3_DF64_STAGE prephase[{index}]=({re:.17e},{im:.17e})");
            assert!(
                re.is_finite() && im.is_finite(),
                "prephase {index} is non-finite"
            );
            assert!((re - expected_re).abs() <= 2.0e-12);
            assert!((im - expected_im).abs() <= 2.0e-12);
        }
        let table = metadata.kernel().horner_table();
        for (index, (&input, (rescaled, factor))) in target_values
            .iter()
            .zip(actual_targets.iter().zip(&actual_factors))
            .enumerate()
        {
            let frequency = rescaled.to_f64();
            let re = factor.re().to_f64();
            let im = factor.im().to_f64();
            eprintln!(
                "TYPE3_DF64_STAGE target[{index}]={frequency:.17e} factor=({re:.17e},{im:.17e})"
            );
            assert!(
                frequency.is_finite(),
                "rescaled target {index} is non-finite"
            );
            assert!(
                re.is_finite() && im.is_finite(),
                "target factor {index} is non-finite"
            );
            assert!((frequency - axis.rescale_target(input)).abs() <= 2.0e-13);
            let phi_hat = horner_kernel_fourier_transform(metadata.kernel(), &table, frequency);
            let phase = sign * (input - axis.target_center()) * axis.source_center();
            assert!((re - phase.cos() / phi_hat).abs() <= 2.0e-11);
            assert!((im - phase.sin() / phi_hat).abs() <= 2.0e-11);
        }
    }

    fn storage_buffer_init<T: bytemuck::Pod>(
        device: &wgpu::Device,
        label: &'static str,
        values: &[T],
    ) -> wgpu::Buffer {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::cast_slice(values),
            usage: wgpu::BufferUsages::STORAGE,
        })
    }

    fn storage_output_buffer<T>(
        device: &wgpu::Device,
        label: &'static str,
        count: usize,
    ) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: (count * std::mem::size_of::<T>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    }

    fn entire_binding_entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
        wgpu::BindGroupEntry {
            binding,
            resource: buffer.as_entire_binding(),
        }
    }

    fn readback_buffer(device: &wgpu::Device, size: u64, stage: &str) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("wgpu_nufft.type3_df64_probe.{stage}.readback")),
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        })
    }

    fn mapped_values<T: bytemuck::Pod>(buffer: &wgpu::Buffer) -> Vec<T> {
        let mapped = buffer.slice(..).get_mapped_range();
        let values = bytemuck::cast_slice(&mapped).to_vec();
        drop(mapped);
        buffer.unmap();
        values
    }

    fn horner_kernel_fourier_transform(
        kernel: EsKernel,
        table: &EsHornerTable,
        frequency: f64,
    ) -> f64 {
        let count = kernel.width() + 2;
        let (nodes, weights) = gauss_legendre(count * 2);
        nodes[count..]
            .iter()
            .zip(&weights[count..])
            .map(|(&node, &weight)| {
                let distance = kernel.half_width() * node;
                2.0 * kernel.half_width()
                    * weight
                    * table.evaluate(distance)
                    * (frequency * distance).cos()
            })
            .sum()
    }

    fn sincos_f64_host(value: f64) -> (f64, f64) {
        let scaled_quadrant = value * (2.0 / std::f64::consts::PI);
        let quadrant = if scaled_quadrant >= 0.0 {
            (scaled_quadrant + 0.5) as i32
        } else {
            (scaled_quadrant - 0.5) as i32
        };
        let quadrant_f64 = f64::from(quadrant);
        let angle = ((value - quadrant_f64 * 1.570_796_326_734_125_6)
            - quadrant_f64 * 6.077_100_506_303_966e-11)
            - quadrant_f64 * 2.022_266_248_711_166_5e-21;
        let squared = angle * angle;
        let mut sine_polynomial = taylor_coefficient(17);
        for degree in (3..=15).rev().step_by(2) {
            sine_polynomial = sine_polynomial * squared + taylor_coefficient(degree);
        }
        let sine = angle + angle * squared * sine_polynomial;
        let mut cosine_polynomial = taylor_coefficient(16);
        for degree in (2..=14).rev().step_by(2) {
            cosine_polynomial = cosine_polynomial * squared + taylor_coefficient(degree);
        }
        let cosine = 1.0 + squared * cosine_polynomial;
        match quadrant & 3 {
            0 => (sine, cosine),
            1 => (cosine, -sine),
            2 => (-sine, -cosine),
            _ => (-cosine, sine),
        }
    }

    #[test]
    fn native_f64_sincos_matches_host_transcendentals_through_validated_bound() {
        for value in [
            -MAX_NATIVE_F64_PHASE_MAGNITUDE,
            -123_456.789_012_345,
            -31.25,
            -std::f64::consts::PI,
            -0.25,
            0.0,
            0.25,
            std::f64::consts::PI,
            31.25,
            123_456.789_012_345,
            MAX_NATIVE_F64_PHASE_MAGNITUDE,
        ] {
            let (sine, cosine) = sincos_f64_host(value);
            assert!(
                (sine - value.sin()).abs() <= 5.0e-16,
                "value={value:e} sine={sine:e} expected={:e}",
                value.sin()
            );
            assert!(
                (cosine - value.cos()).abs() <= 5.0e-16,
                "value={value:e} cosine={cosine:e} expected={:e}",
                value.cos()
            );
        }
        assert!(
            validate_phase_bound(FftPrecision::F64, "test", MAX_NATIVE_F64_PHASE_MAGNITUDE).is_ok()
        );
        assert!(validate_phase_bound(
            FftPrecision::F64,
            "test",
            MAX_NATIVE_F64_PHASE_MAGNITUDE.next_up()
        )
        .is_err());
    }

    #[test]
    fn gpu_quadrature_constants_match_host_continuous_kernel_transform() {
        let kernel = EsKernel::for_tolerance(1.0e-6, 2.0).unwrap();
        let count = kernel.width() + 2;
        let (nodes, weights) = gauss_legendre(count * 2);
        for frequency in [-1.7, -0.25, 0.0, 0.9, 2.1] {
            let actual = nodes[count..]
                .iter()
                .zip(&weights[count..])
                .map(|(&node, &weight)| {
                    let distance = kernel.half_width() * node;
                    2.0 * kernel.half_width()
                        * weight
                        * kernel.evaluate(distance)
                        * (frequency * distance).cos()
                })
                .sum::<f64>();
            let expected = es_kernel_fourier_transform(kernel, frequency).unwrap();
            assert!((actual - expected).abs() <= 2.0e-14 * expected.abs().max(1.0));
        }
    }

    fn reduce_phase_df64_host(value: f64) -> DoubleFloat {
        let tau = DoubleFloat::from_f64(std::f64::consts::TAU);
        let inverse_tau = DoubleFloat::from_f64(1.0 / std::f64::consts::TAU);
        let mut reduced = DoubleFloat::from_f64(value);
        for _ in 0..PHASE_REDUCTION_STEPS {
            let quotient = reduced.mul_df(inverse_tau);
            let integer_quotient = quotient.hi.trunc();
            reduced = reduced.sub_df(DoubleFloat::new(integer_quotient, 0.0).mul_df(tau));
        }
        for _ in 0..PHASE_FINAL_CORRECTIONS {
            if reduced.to_f64() >= std::f64::consts::TAU {
                reduced = reduced.sub_df(tau);
            }
            if reduced.to_f64() <= -std::f64::consts::TAU {
                reduced = reduced.add_df(tau);
            }
        }
        reduced
    }

    fn negate_df64(value: DoubleFloat) -> DoubleFloat {
        DoubleFloat::new(-value.hi, -value.lo)
    }

    fn sincos_df64_host(value: f64) -> (f64, f64) {
        let reduced = reduce_phase_df64_host(value);
        let scaled_quadrant = reduced.mul_df(DoubleFloat::from_f64(2.0 / std::f64::consts::PI));
        let shifted_quadrant = if scaled_quadrant.hi < 0.0
            || (scaled_quadrant.hi == 0.0 && scaled_quadrant.lo < 0.0)
        {
            scaled_quadrant.sub_df(DoubleFloat::from_f64(0.5))
        } else {
            scaled_quadrant.add_df(DoubleFloat::from_f64(0.5))
        };
        let quadrant = shifted_quadrant.hi.trunc() as i32;
        let quadrant_df64 = DoubleFloat::new(quadrant as f32, 0.0);
        let mut angle =
            reduced.sub_df(quadrant_df64.mul_df(DoubleFloat::from_f64(1.570_796_326_734_125_6)));
        angle =
            angle.sub_df(quadrant_df64.mul_df(DoubleFloat::from_f64(6.077_100_506_303_966e-11)));
        angle =
            angle.sub_df(quadrant_df64.mul_df(DoubleFloat::from_f64(2.022_266_248_711_166_5e-21)));
        let squared = angle.mul_df(angle);
        let mut sine_polynomial = DoubleFloat::from_f64(taylor_coefficient(17));
        for degree in (3..=15).rev().step_by(2) {
            sine_polynomial = sine_polynomial
                .mul_df(squared)
                .add_df(DoubleFloat::from_f64(taylor_coefficient(degree)));
        }
        let sine = angle.add_df(angle.mul_df(squared).mul_df(sine_polynomial));
        let mut cosine_polynomial = DoubleFloat::from_f64(taylor_coefficient(16));
        for degree in (2..=14).rev().step_by(2) {
            cosine_polynomial = cosine_polynomial
                .mul_df(squared)
                .add_df(DoubleFloat::from_f64(taylor_coefficient(degree)));
        }
        let cosine = DoubleFloat::from_f64(1.0).add_df(squared.mul_df(cosine_polynomial));
        let (sine, cosine) = match quadrant & 3 {
            0 => (sine, cosine),
            1 => (cosine, negate_df64(sine)),
            2 => (negate_df64(sine), negate_df64(cosine)),
            _ => (negate_df64(cosine), sine),
        };
        (sine.to_f64(), cosine.to_f64())
    }

    #[test]
    fn legacy_f32_phase_reduction_remains_bounded_across_its_exponent_range() {
        for value in [
            0.0,
            7.0,
            -31.25,
            1.0e6,
            -1.0e12,
            2.0f64.powi(40) * 1.234_567_890_123,
            f32::MAX as f64 * 0.25,
            -f32::MAX as f64 * 0.25,
        ] {
            let reduced = reduce_phase_df64_host(value).to_f64();
            assert!(reduced.is_finite());
            assert!(reduced.abs() < std::f64::consts::TAU);
            if value.abs() <= 2.0f64.powi(40) {
                let expected = value.rem_euclid(std::f64::consts::TAU);
                let actual = reduced.rem_euclid(std::f64::consts::TAU);
                // The f32 route uses df64 only as an internal reduction aid;
                // its absolute modulo error still grows with the unreduced
                // phase even though the reduction itself remains bounded.
                let tolerance = 2.0e-4 + 4.0 * value.abs() * 2.0f64.powi(-48);
                assert!(
                    (actual - expected).abs() <= tolerance,
                    "value={value:e} reduced={reduced:e} actual={actual:e} expected={expected:e} error={:e}",
                    (actual - expected).abs(),
                );
            }
        }
    }

    #[test]
    fn df64_phase_accuracy_gate_matches_the_verified_portable_bound() {
        assert!(validate_phase_bound(FftPrecision::Df64, "test", MAX_DF64_PHASE_MAGNITUDE).is_ok());
        assert!(validate_phase_bound(
            FftPrecision::Df64,
            "test",
            MAX_DF64_PHASE_MAGNITUDE.next_up()
        )
        .is_err());

        let mut phases = (-4096..=4096)
            .map(|index| MAX_DF64_PHASE_MAGNITUDE * f64::from(index) / 4096.0)
            .collect::<Vec<_>>();
        let offsets = [
            0.0,
            -2.0f64.powi(-45),
            2.0f64.powi(-45),
            -2.0f64.powi(-35),
            2.0f64.powi(-35),
            -1.0e-10,
            1.0e-10,
        ];
        for quadrant in -652..=652 {
            for offset in offsets {
                for quadrant_offset in [0.0, 0.5] {
                    let value = (f64::from(quadrant) + quadrant_offset)
                        * std::f64::consts::FRAC_PI_2
                        + offset;
                    if value.abs() <= MAX_DF64_PHASE_MAGNITUDE {
                        phases.push(value);
                    }
                }
            }
        }
        let mut worst_error = 0.0f64;
        for value in phases {
            let (sine, cosine) = sincos_df64_host(value);
            let error = (sine - value.sin()).abs().max((cosine - value.cos()).abs());
            worst_error = worst_error.max(error);
            assert!(
                error <= 1.0e-11,
                "value={value:e} error={error:e} sine={sine:e} cosine={cosine:e}"
            );
        }
        eprintln!("df64 phase boundary canary worst absolute error={worst_error:.9e}");
    }

    #[test]
    fn dimension_aware_buffer_sizes_are_checked() {
        assert_eq!(
            coordinate_buffer_size("test", 7, 3, FftPrecision::F32).unwrap(),
            84
        );
        assert_eq!(
            coordinate_buffer_size("test", 7, 3, FftPrecision::F64).unwrap(),
            168
        );
        assert_eq!(
            coordinate_buffer_size("test", 7, 3, FftPrecision::Df64).unwrap(),
            168
        );
        assert_eq!(
            complex_buffer_size("test", 7, FftPrecision::F32).unwrap(),
            56
        );
        assert_eq!(
            complex_buffer_size("test", 7, FftPrecision::F64).unwrap(),
            112
        );
        assert_eq!(
            complex_buffer_size("test", 7, FftPrecision::Df64).unwrap(),
            112
        );
        assert!(coordinate_buffer_size("test", usize::MAX, 3, FftPrecision::F64).is_err());
    }

    #[test]
    fn metadata_validation_rejects_a_df64_phase_outside_the_accuracy_range() {
        let plan = NufftType3Plan::new(
            NufftType3Config::new(
                [NufftInterval::new(1.0e4, 1.0e4)],
                [NufftInterval::new(1.0e3, 1.0e3)],
                1.0e-6,
            )
            .with_precision(FftPrecision::Df64),
        )
        .unwrap();
        assert!(matches!(
            validate_gpu_metadata(&plan),
            Err(NufftError::Type3RescalingUnsupported {
                quantity: "source pre-phase",
                reason: "the conservative phase bound exceeds the portable-df64 accuracy range",
                ..
            })
        ));
    }
}
