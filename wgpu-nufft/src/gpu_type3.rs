use std::num::NonZeroU64;
use std::sync::{Mutex, MutexGuard};

use crate::config::{ModeOrder, NufftConfig};
use crate::error::{NufftError, Result};
use crate::gpu::max_supported_workgroup_size;
use crate::gpu_dispatch::split_workgroups;
use crate::gpu_type1::Type1GpuPlan;
use crate::gpu_type1_2d::{Type1Gather2d, Type1GpuPlan2d};
use crate::gpu_type1_3d::{Type1Gather3d, Type1GpuPlan3d};
use crate::kernel::EsKernel;
use crate::plan::NufftPlan;
use crate::type3::NufftType3Plan;

const WORKGROUP_SIZE: u32 = 64;
const F32_BYTES: u64 = 4;
const COMPLEX_F32_BYTES: u64 = 8;
const PHASE_REDUCTION_STEPS: u32 = 8;
const PHASE_FINAL_CORRECTIONS: usize = 4;
const MAX_GPU_PHASE_MAGNITUDE: f64 = f32::MAX as f64 * 0.5;

/// GPU-resident composition of the outer spread and inner type-2 plan.
///
/// The caller supplies original source/target coordinates and complex strengths.
/// Rescaled coordinates, phased strengths, correction factors, and interpolated
/// values live in grow-only plan scratch, so repeated execution does not allocate
/// GPU buffers. Executions on one plan must retain queue order.
pub(crate) struct GpuType3Plan {
    dimensions: usize,
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
            .with_sigma(metadata.config().sigma());
        let kernel = metadata.kernel();
        let outer_coefficients = outer_shape
            .iter()
            .map(|&length| kernel.centered_fourier_coefficients(length))
            .collect::<Result<Vec<_>>>()?;
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
            _ => {
                return Err(NufftError::GpuDimensionsUnsupported {
                    kind: "type-3",
                    actual: dimensions,
                    supported: 3,
                });
            }
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
            .with_sigma(metadata.config().sigma());
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
            &generate_final_wgsl(),
        );
        let final_layout = final_pipeline.get_bind_group_layout(0);

        Ok(Self {
            dimensions,
            raw_spread,
            inner_type2,
            source_pipeline,
            source_layout,
            target_pipeline,
            target_layout,
            final_pipeline,
            final_layout,
            scratch: Mutex::new(Type3ScratchBuffers::new(device)),
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
        coordinate_buffer_size("type-3 source point", source_count, self.dimensions)
    }

    pub(crate) fn required_source_strength_buffer_size_bytes(
        &self,
        source_count: usize,
    ) -> Result<u64> {
        complex_buffer_size("type-3 source strength", source_count)
    }

    pub(crate) fn required_target_point_buffer_size_bytes(
        &self,
        target_count: usize,
    ) -> Result<u64> {
        coordinate_buffer_size("type-3 target point", target_count, self.dimensions)
    }

    pub(crate) fn required_output_buffer_size_bytes(&self, target_count: usize) -> Result<u64> {
        complex_buffer_size("type-3 output", target_count)
    }

    #[allow(clippy::too_many_arguments)]
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
        if target_count == 0 {
            return Ok(());
        }

        let source_point_bytes = self.required_source_point_buffer_size_bytes(source_count)?;
        let strength_bytes = self.required_source_strength_buffer_size_bytes(source_count)?;
        let target_point_bytes = self.required_target_point_buffer_size_bytes(target_count)?;
        let output_bytes = self.required_output_buffer_size_bytes(target_count)?;
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

        let source_count_u32 =
            u32::try_from(source_count).map_err(|_| NufftError::LengthOverflow {
                context: "type-3 source count shader index space",
            })?;
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
            output_bytes,
            self.max_storage_binding_bytes,
            self.max_buffer_bytes,
        )?;

        if source_count != 0 {
            let source_dispatch =
                dispatch_for_elements(source_count_u32, self.max_workgroups_per_dimension)?;
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

        self.raw_spread.encode_spread(
            device,
            encoder,
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
                binding_entry(2, &scratch.target_factors, output_bytes),
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

        self.inner_type2.encode_type2_gpu(
            device,
            encoder,
            target_count,
            &scratch.rescaled_targets,
            self.raw_spread.fine_grid_buffer(),
            &scratch.interpolated,
        )?;

        let final_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("wgpu_nufft.type3.final_correction.bind_group"),
            layout: &self.final_layout,
            entries: &[
                binding_entry(0, &scratch.interpolated, output_bytes),
                binding_entry(1, &scratch.target_factors, output_bytes),
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
            pass.dispatch_workgroups(target_dispatch.0, target_dispatch.1, target_dispatch.2);
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
}

impl RawSpreadPlan {
    fn encode_spread(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        point_count: usize,
        points: &wgpu::Buffer,
        strengths: &wgpu::Buffer,
    ) -> Result<()> {
        match self {
            Self::OneD(plan) => plan.encode_spread(device, encoder, point_count, points, strengths),
            Self::TwoD(plan) => plan.encode_spread(device, encoder, point_count, points, strengths),
            Self::ThreeD(plan) => {
                plan.encode_spread(device, encoder, point_count, points, strengths)
            }
        }
    }

    fn fine_grid_buffer(&self) -> &wgpu::Buffer {
        match self {
            Self::OneD(plan) => plan.fine_grid_buffer(),
            Self::TwoD(plan) => plan.fine_grid_buffer(),
            Self::ThreeD(plan) => plan.fine_grid_buffer(),
        }
    }

    fn fine_grid_element_count(&self) -> usize {
        match self {
            Self::OneD(plan) => plan.fine_grid_element_count(),
            Self::TwoD(plan) => plan.fine_grid_element_count(),
            Self::ThreeD(plan) => plan.fine_grid_element_count(),
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
    fn new(device: &wgpu::Device) -> Self {
        Self {
            rescaled_sources: create_scratch_buffer(
                device,
                "wgpu_nufft.type3.rescaled_sources",
                F32_BYTES,
            ),
            rescaled_source_capacity: F32_BYTES,
            prephased_strengths: create_scratch_buffer(
                device,
                "wgpu_nufft.type3.prephased_strengths",
                COMPLEX_F32_BYTES,
            ),
            prephased_strength_capacity: COMPLEX_F32_BYTES,
            rescaled_targets: create_scratch_buffer(
                device,
                "wgpu_nufft.type3.rescaled_targets",
                F32_BYTES,
            ),
            rescaled_target_capacity: F32_BYTES,
            target_factors: create_scratch_buffer(
                device,
                "wgpu_nufft.type3.target_factors",
                COMPLEX_F32_BYTES,
            ),
            target_factor_capacity: COMPLEX_F32_BYTES,
            interpolated: create_scratch_buffer(
                device,
                "wgpu_nufft.type3.interpolated",
                COMPLEX_F32_BYTES,
            ),
            interpolated_capacity: COMPLEX_F32_BYTES,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn ensure(
        &mut self,
        device: &wgpu::Device,
        source_point_bytes: u64,
        strength_bytes: u64,
        target_point_bytes: u64,
        target_value_bytes: u64,
        max_storage_binding_bytes: u64,
        max_buffer_bytes: u64,
    ) -> Result<()> {
        for (label, bytes) in [
            ("type-3 rescaled source scratch", source_point_bytes),
            ("type-3 prephased strength scratch", strength_bytes),
            ("type-3 rescaled target scratch", target_point_bytes),
            ("type-3 target factor scratch", target_value_bytes),
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
            target_value_bytes,
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
            validate_split_constant(axis, quantity, value)?;
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
            validate_split_constant(axis, quantity, value)?;
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
    validate_phase_bound("source pre-phase", source_phase_bound)?;
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
    validate_phase_bound("target post-phase", target_phase_bound)?;
    Ok(())
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

fn validate_phase_bound(quantity: &'static str, value: f64) -> Result<()> {
    if value.is_finite() && value <= MAX_GPU_PHASE_MAGNITUDE {
        Ok(())
    } else {
        Err(NufftError::Type3RescalingUnsupported {
            axis: 0,
            quantity,
            value,
            reason: "the conservative phase bound exceeds the portable df64 exponent range",
        })
    }
}

fn generate_source_wgsl(metadata: &NufftType3Plan) -> Result<String> {
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
    rescaled_sources[coordinate_offset + {axis}u] = rescaled_{axis}.hi + rescaled_{axis}.lo;
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
    let source_index = workgroup_flat * WORKGROUP_SIZE + lid.x;
    if (source_index >= arrayLength(&strengths)) {{ return; }}
    let coordinate_offset = source_index * DIMENSIONS;
    var phase = Df64(0.0, 0.0);
{body}
    let angle = reduce_phase_df64(df64_mul(phase, ISIGN));
    let sine = sin(angle);
    let cosine = cos(angle);
    let value = strengths[source_index];
    prephased_strengths[source_index] = vec2<f32>(
        value.x * cosine - value.y * sine,
        value.x * sine + value.y * cosine,
    );
}}
"#,
        df64 = wgpu_fft::kernels::DF64_WGSL,
        phase_reduction = phase_reduction_wgsl()?,
    ))
}

fn generate_target_wgsl(metadata: &NufftType3Plan) -> Result<String> {
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

fn generate_final_wgsl() -> String {
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
    let factor = target_factors[index];
    output_values[index] = vec2<f32>(
        value.x * factor.x - value.y * factor.y,
        value.x * factor.y + value.y * factor.x,
    );
}}
"#,
    )
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

fn format_dd_const(name: &str, value: f64) -> Result<String> {
    Ok(format!(
        "const {name}: Df64 = {};\n",
        format_dd_value(value)?
    ))
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
) -> Result<u64> {
    let elements = point_count
        .checked_mul(dimensions)
        .ok_or(NufftError::LengthOverflow { context })?;
    checked_buffer_size(context, elements, F32_BYTES)
}

fn complex_buffer_size(context: &'static str, element_count: usize) -> Result<u64> {
    checked_buffer_size(context, element_count, COMPLEX_F32_BYTES)
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
mod tests {
    use super::*;
    use crate::type3::{es_kernel_fourier_transform, NufftInterval, NufftType3Config};
    use wgpu_fft::math::DoubleFloat;

    fn metadata(dimensions: usize) -> NufftType3Plan {
        NufftType3Plan::new(NufftType3Config::new(
            (0..dimensions)
                .map(|axis| NufftInterval::new(-1.5 - axis as f64, 2.0 + axis as f64))
                .collect::<Vec<_>>(),
            (0..dimensions)
                .map(|axis| NufftInterval::new(-3.0 - axis as f64, 4.0 + axis as f64))
                .collect::<Vec<_>>(),
            1.0e-6,
        ))
        .unwrap()
    }

    #[test]
    fn generated_shaders_cover_every_axis_and_use_df64_phase_reduction() {
        for dimensions in 1..=3 {
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

    fn reduce_phase_host(value: f64) -> f64 {
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
        reduced.to_f64()
    }

    #[test]
    fn fixed_phase_reduction_covers_df64_exponent_range() {
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
            let reduced = reduce_phase_host(value);
            assert!(reduced.is_finite());
            assert!(reduced.abs() < std::f64::consts::TAU);
            if value.abs() <= 2.0f64.powi(40) {
                let expected = value.rem_euclid(std::f64::consts::TAU);
                let actual = reduced.rem_euclid(std::f64::consts::TAU);
                // A df64 phase carries roughly 48 significant bits. Its
                // absolute modulo error therefore grows with the unreduced
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
    fn dimension_aware_buffer_sizes_are_checked() {
        assert_eq!(coordinate_buffer_size("test", 7, 3).unwrap(), 84);
        assert_eq!(complex_buffer_size("test", 7).unwrap(), 56);
        assert!(coordinate_buffer_size("test", usize::MAX, 3).is_err());
    }

    #[test]
    fn metadata_validation_rejects_a_phase_outside_df64_exponent_range() {
        let plan = NufftType3Plan::new(NufftType3Config::new(
            [NufftInterval::new(-1.0e30, 1.0e30)],
            [NufftInterval::new(1.0e20, 1.0e20)],
            1.0e-6,
        ));
        // Grid selection generally rejects this first. If a future grid policy
        // permits it, the GPU-specific phase guard must still reject it.
        if let Ok(plan) = plan {
            assert!(matches!(
                validate_gpu_metadata(&plan),
                Err(NufftError::Type3RescalingUnsupported { .. })
            ));
        }
    }
}
