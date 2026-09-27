use std::collections::HashMap;
use std::sync::Arc;

use bytemuck::{Pod, Zeroable};

use crate::config::{FftConfig, FftDirection, FftPrecision, Normalization};
use crate::device::device_supports_precision;
use crate::error::{FftError, Result};
use crate::runtime::buffer_view::BufferView;
use crate::runtime::dispatch::{max_workgroups_per_dimension, split_workgroups};
use crate::runtime::large_graph::ElementFormat;
use crate::runtime::pipeline_cache::{
    with_device_pipeline_cache, ComputePipelineCacheKey, FusedPow2StageKey, FusedSmoothStageKey,
    PipelineLayoutCacheKey, StockhamStageKey,
};
use crate::runtime::recorder::CommandRecorder;
use crate::runtime::twiddle::create_twiddle_lut_buffer_for_len_with_precision;
#[cfg(test)]
use crate::runtime::twiddle::twiddle_lut_f32;
use crate::runtime::window_scheduler::WindowScheduler;

#[cfg(test)]
const DEFAULT_WORKGROUP_SIZE: u32 = 64;
#[cfg(test)]
const DEFAULT_FUSED_WORKGROUP_SIZE: u32 = 256;
#[cfg(test)]
const WORKGROUP_SIZE: u32 = DEFAULT_WORKGROUP_SIZE;
#[cfg(test)]
const FUSED_POW2_WORKGROUP_SIZE: u32 = DEFAULT_FUSED_WORKGROUP_SIZE;
#[cfg(test)]
const FUSED_SMOOTH_WORKGROUP_SIZE: u32 = DEFAULT_FUSED_WORKGROUP_SIZE;

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct AxisParams {
    total: u32,
    base_index: u32,
    line_offset: u32,
    element_base: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AxisLayout {
    Interleaved,
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum AxisPrecision {
    F32,
    F64,
    Df64,
}

impl AxisPrecision {
    pub(crate) const fn complex_size_bytes(self) -> u64 {
        match self {
            Self::F32 => 8,
            Self::F64 | Self::Df64 => 16,
        }
    }

    pub(crate) const fn element_format(self) -> ElementFormat {
        match self {
            Self::F32 => ElementFormat::ComplexF32,
            Self::F64 => ElementFormat::ComplexF64,
            Self::Df64 => ElementFormat::ComplexDf64,
        }
    }

    pub(crate) const fn as_fft_precision(self) -> FftPrecision {
        match self {
            Self::F32 => FftPrecision::F32,
            Self::F64 => FftPrecision::F64,
            Self::Df64 => FftPrecision::Df64,
        }
    }

    pub(crate) const fn as_str(self) -> &'static str {
        self.as_fft_precision().as_str()
    }

    pub(crate) fn wgsl_scalar_type(self) -> &'static str {
        match self {
            Self::F32 | Self::Df64 => "f32",
            Self::F64 => "f64",
        }
    }

    pub(crate) fn wgsl_complex_type(self) -> &'static str {
        match self {
            Self::F32 => "vec2<f32>",
            Self::F64 => "vec2<f64>",
            Self::Df64 => "vec4<f32>",
        }
    }

    pub(crate) fn format_wgsl_scalar(self, value: f64) -> String {
        match self {
            Self::F32 => format_wgsl_f32(value as f32),
            Self::F64 => format_wgsl_f64(value),
            Self::Df64 => panic!("df64 scalars require an explicit hi/lo pair"),
        }
    }

    pub(crate) fn specialize_wgsl(self, source: String) -> String {
        match self {
            Self::F32 => source,
            Self::F64 => source.replace("vec2<f32>", "vec2<f64>"),
            Self::Df64 => source.replace("vec2<f32>", "vec4<f32>"),
        }
    }

    pub(crate) fn format_wgsl_complex(self, re: f64, im: f64) -> String {
        match self {
            Self::F32 | Self::F64 => format!(
                "{}({}, {})",
                self.wgsl_complex_type(),
                self.format_wgsl_scalar(re),
                self.format_wgsl_scalar(im)
            ),
            Self::Df64 => {
                let re = crate::math::DoubleFloat::from_f64(re);
                let im = crate::math::DoubleFloat::from_f64(im);
                format!(
                    "vec4<f32>({}, {}, {}, {})",
                    format_wgsl_f32(re.hi),
                    format_wgsl_f32(re.lo),
                    format_wgsl_f32(im.hi),
                    format_wgsl_f32(im.lo)
                )
            }
        }
    }
}

impl From<FftPrecision> for AxisPrecision {
    fn from(precision: FftPrecision) -> Self {
        match precision {
            FftPrecision::F32 => Self::F32,
            FftPrecision::F64 => Self::F64,
            FftPrecision::Df64 => Self::Df64,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct AxisPlanConfig {
    pub(crate) shape: Vec<usize>,
    pub(crate) axes: Vec<usize>,
    pub(crate) batch: usize,
    pub(crate) direction: FftDirection,
    pub(crate) normalization: Normalization,
    pub(crate) scale_override_bits: Option<u32>,
    pub(crate) layout: AxisLayout,
    pub(crate) precision: AxisPrecision,
    pub(crate) workgroup_size: u32,
    pub(crate) fused_workgroup_size: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AxisStageKind {
    Stockham { radix: usize, ns: usize },
    FusedPow2 { axis_length: usize },
    FusedSmooth { axis_length: usize },
}

impl AxisStageKind {
    fn detail(self) -> String {
        match self {
            Self::Stockham { radix, ns } => format!("stockham.radix{radix}.ns{ns}"),
            Self::FusedPow2 { axis_length } => format!("fused_pow2.n{axis_length}"),
            Self::FusedSmooth { axis_length } => format!("fused_smooth.n{axis_length}"),
        }
    }
}

pub(crate) struct AxisStage {
    pub(crate) axis: usize,
    pub(crate) kind: AxisStageKind,
    pub(crate) stride_complex: usize,
    pub(crate) apply_scale: bool,
    pub(crate) pipeline_key: ComputePipelineCacheKey,
    twiddle_lut_index: usize,
    workgroups_x: u32,
    pipeline: wgpu::ComputePipeline,
}

struct AxisTwiddleLut {
    axis_length: usize,
    precision: AxisPrecision,
    buffer: Arc<wgpu::Buffer>,
}

#[derive(Default)]
pub(crate) struct AxisTwiddleLutPool {
    buffers: HashMap<(AxisPrecision, usize), Arc<wgpu::Buffer>>,
}

impl AxisTwiddleLutPool {
    pub(crate) fn storage_bytes(&self) -> u64 {
        self.buffers
            .values()
            .fold(0u64, |bytes, buffer| bytes.saturating_add(buffer.size()))
    }
}

pub(crate) struct AxisPlan {
    config: AxisPlanConfig,
    factors: Vec<Vec<usize>>,
    stages: Vec<AxisStage>,
    bind_group_layout: wgpu::BindGroupLayout,
    params_buffer: wgpu::Buffer,
    temp_buffer: Option<wgpu::Buffer>,
    twiddle_luts: Vec<AxisTwiddleLut>,
    required_buffer_size_bytes: u64,
    workspace_size_bytes: u64,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct StockhamStageWgslConfig<'a> {
    pub(crate) rank: usize,
    pub(crate) axis: usize,
    pub(crate) dims: &'a [usize],
    pub(crate) axis_length: usize,
    pub(crate) stride_complex: usize,
    pub(crate) radix: usize,
    pub(crate) ns: usize,
    pub(crate) direction: FftDirection,
    pub(crate) workgroup_size: u32,
    pub(crate) apply_scale: bool,
    pub(crate) scale_factor: f64,
    pub(crate) precision: AxisPrecision,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct FusedPow2StageWgslConfig<'a> {
    pub(crate) rank: usize,
    pub(crate) axis: usize,
    pub(crate) dims: &'a [usize],
    pub(crate) axis_length: usize,
    pub(crate) stride_complex: usize,
    pub(crate) direction: FftDirection,
    pub(crate) workgroup_size: u32,
    pub(crate) apply_scale: bool,
    pub(crate) scale_factor: f64,
    pub(crate) precision: AxisPrecision,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct FusedSmoothStageWgslConfig<'a> {
    pub(crate) rank: usize,
    pub(crate) axis: usize,
    pub(crate) dims: &'a [usize],
    pub(crate) axis_length: usize,
    pub(crate) stride_complex: usize,
    pub(crate) factors: &'a [usize],
    pub(crate) direction: FftDirection,
    pub(crate) workgroup_size: u32,
    pub(crate) apply_scale: bool,
    pub(crate) scale_factor: f64,
    pub(crate) precision: AxisPrecision,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BufferSlot {
    Input,
    Output,
    Temp,
}

fn axis_plan_buffer_flow_error(reason: &'static str) -> FftError {
    FftError::LargeGraphStageUnsupported {
        stage: "axis-plan-buffer-flow",
        reason,
    }
}

fn axis_plan_workspace_error() -> FftError {
    FftError::LargeGraphStageUnsupported {
        stage: "axis-plan-workspace",
        reason: "multi-stage AxisPlan requires temp storage",
    }
}

fn next_stage_destination(src_slot: BufferSlot) -> Result<BufferSlot> {
    match src_slot {
        BufferSlot::Output => Ok(BufferSlot::Temp),
        BufferSlot::Temp => Ok(BufferSlot::Output),
        BufferSlot::Input => Err(axis_plan_buffer_flow_error(
            "mixed-radix stage buffer flow attempted to use input as destination",
        )),
    }
}

impl AxisPlanConfig {
    pub(crate) fn from_c2c_config(config: &FftConfig) -> Self {
        Self {
            shape: config.shape().to_vec(),
            axes: config.axes().to_vec(),
            batch: config.batch(),
            direction: config.direction(),
            normalization: config.normalization(),
            scale_override_bits: None,
            layout: AxisLayout::Interleaved,
            precision: config.precision().into(),
            workgroup_size: config.tuning().workgroup_size(),
            fused_workgroup_size: config.tuning().fused_workgroup_size(),
        }
    }

    pub(crate) fn validate(&self) -> Result<()> {
        if self.shape.is_empty() || self.shape.iter().any(|&len| len == 0) {
            return Err(FftError::ZeroLength);
        }

        if self.batch == 0 {
            return Err(FftError::ZeroBatch);
        }

        if self.axes.is_empty() {
            return Err(FftError::EmptyAxes);
        }

        match (self.layout, self.precision) {
            (
                AxisLayout::Interleaved,
                AxisPrecision::F32 | AxisPrecision::F64 | AxisPrecision::Df64,
            ) => {}
        }

        if self.workgroup_size == 0 || !self.workgroup_size.is_power_of_two() {
            return Err(FftError::LargeGraphStageUnsupported {
                stage: "axis-plan-tuning",
                reason: "staged workgroup size must be a nonzero power of two",
            });
        }
        if self.fused_workgroup_size == 0 {
            return Err(FftError::LargeGraphStageUnsupported {
                stage: "axis-plan-tuning",
                reason: "fused workgroup size must be nonzero",
            });
        }

        let rank = self.shape.len();
        let mut seen_axes = vec![false; rank];
        for &axis in &self.axes {
            if axis >= rank {
                return Err(FftError::InvalidAxis { axis, rank });
            }
            if seen_axes[axis] {
                return Err(FftError::DuplicateAxis { axis });
            }
            seen_axes[axis] = true;

            let axis_len = self.shape[axis];
            if axis_len < 2 {
                return Err(FftError::UnsupportedLength { len: axis_len });
            }
            crate::runtime::factor_supported_length(axis_len)?;
        }

        self.total_complex()?;
        Ok(())
    }

    pub(crate) fn total_complex(&self) -> Result<usize> {
        let mut total = 1usize;
        for &dim in &self.shape {
            total = total
                .checked_mul(dim)
                .ok_or(FftError::LengthTooLarge { len: usize::MAX })?;
        }
        total = total
            .checked_mul(self.batch)
            .ok_or(FftError::LengthTooLarge { len: usize::MAX })?;

        if total > u32::MAX as usize {
            return Err(FftError::LengthTooLarge { len: total });
        }

        Ok(total)
    }

    pub(crate) fn scale(&self) -> Result<f64> {
        if let Some(bits) = self.scale_override_bits {
            return Ok(f64::from(f32::from_bits(bits)));
        }

        let total = product(&self.shape);
        match self.precision {
            AxisPrecision::F32 => {
                // Preserve the established f32 normalization bits for existing
                // plans; widening happens only after the f32 arithmetic.
                let total = total as f32;
                let scale = match (self.direction, self.normalization) {
                    (_, Normalization::None) => 1.0,
                    (FftDirection::Forward, Normalization::Forward) => 1.0 / total,
                    (FftDirection::Inverse, Normalization::Inverse) => 1.0 / total,
                    (_, Normalization::Orthogonal) => 1.0 / total.sqrt(),
                    _ => 1.0,
                };
                Ok(f64::from(scale))
            }
            AxisPrecision::F64 | AxisPrecision::Df64 => {
                let total = total as f64;
                let scale = match (self.direction, self.normalization) {
                    (_, Normalization::None) => 1.0,
                    (FftDirection::Forward, Normalization::Forward) => 1.0 / total,
                    (FftDirection::Inverse, Normalization::Inverse) => 1.0 / total,
                    (_, Normalization::Orthogonal) => 1.0 / total.sqrt(),
                    _ => 1.0,
                };
                Ok(scale)
            }
        }
    }

    #[cfg(test)]
    pub(crate) fn stockham_workspace_size_bytes(&self) -> Result<u64> {
        self.validate()?;
        let stage_count = self.stockham_stage_count()?;
        Ok(workspace_size_bytes_for_stage_count_with_precision(
            stage_count,
            self.total_complex()?,
            self.precision,
        ))
    }

    #[cfg(test)]
    fn stockham_stage_count(&self) -> Result<usize> {
        let mut count = 0usize;
        for &axis in &self.axes {
            count += crate::runtime::factor_supported_length(self.shape[axis])?.len();
        }
        Ok(count)
    }
}

impl AxisPlan {
    pub(crate) fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        config: AxisPlanConfig,
    ) -> Result<Self> {
        let mut twiddle_lut_pool = AxisTwiddleLutPool::default();
        Self::new_with_twiddle_lut_pool(device, queue, config, &mut twiddle_lut_pool)
    }

    pub(crate) fn new_with_twiddle_lut_pool(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        config: AxisPlanConfig,
        twiddle_lut_pool: &mut AxisTwiddleLutPool,
    ) -> Result<Self> {
        config.validate()?;
        validate_1d_workgroup_size(
            config.workgroup_size,
            &device.limits(),
            "axis-plan-staged-workgroup",
        )?;
        if !device_supports_precision(device, config.precision.as_fft_precision()) {
            return Err(FftError::PrecisionUnsupported {
                requested: config.precision.as_fft_precision(),
                route: "mixed-radix",
                reason: "device-missing-shader-f64",
            });
        }

        let total_complex = config.total_complex()?;
        let total_complex_u32 = total_complex as u32;
        let scale = config.scale()?;
        let apply_any_scale = scale != 1.0;

        let bind_group_layout = with_device_pipeline_cache(device, |cache| {
            cache.get_bind_group_layout(
                device,
                match config.precision {
                    AxisPrecision::F32 => PipelineLayoutCacheKey::AxisPlanInterleavedF32Lut,
                    AxisPrecision::F64 => PipelineLayoutCacheKey::AxisPlanInterleavedF64Lut,
                    AxisPrecision::Df64 => PipelineLayoutCacheKey::AxisPlanInterleavedDf64Lut,
                },
            )
        });

        let mut factors = Vec::with_capacity(config.axes.len());
        let mut stages = Vec::new();
        let mut twiddle_luts = Vec::<AxisTwiddleLut>::new();

        for (axis_index, &axis) in config.axes.iter().enumerate() {
            let axis_len = config.shape[axis];
            let axis_factors = crate::runtime::factor_supported_length(axis_len)?;
            let stride_complex = stride_for_axis(&config.shape, axis);
            let final_axis = axis_index + 1 == config.axes.len();
            let twiddle_lut_index = if let Some(index) = twiddle_luts
                .iter()
                .position(|lut| lut.axis_length == axis_len && lut.precision == config.precision)
            {
                index
            } else {
                let pool_key = (config.precision, axis_len);
                let buffer = if let Some(buffer) = twiddle_lut_pool.buffers.get(&pool_key) {
                    Arc::clone(buffer)
                } else {
                    let buffer = Arc::new(create_twiddle_lut_buffer_for_len_with_precision(
                        device,
                        queue,
                        "wgpu_fft.axis_plan.twiddle_lut",
                        axis_len,
                        config.precision.as_fft_precision(),
                    )?);
                    twiddle_lut_pool
                        .buffers
                        .insert(pool_key, Arc::clone(&buffer));
                    buffer
                };
                let index = twiddle_luts.len();
                twiddle_luts.push(AxisTwiddleLut {
                    axis_length: axis_len,
                    precision: config.precision,
                    buffer,
                });
                index
            };

            if fused_pow2_supported(
                axis_len,
                config.precision,
                config.fused_workgroup_size,
                &device.limits(),
            ) {
                let apply_scale = apply_any_scale && final_axis;
                let total_lines = total_complex / axis_len;
                let lines_per_workgroup = fused_lines_per_workgroup(
                    axis_len,
                    stride_complex,
                    config.precision,
                    total_lines,
                    u64::from(device.limits().max_compute_workgroup_storage_size),
                );
                let shader_key = FusedPow2StageKey::new(
                    config.shape.len(),
                    axis,
                    &config.shape,
                    axis_len,
                    stride_complex,
                    config.direction,
                    config.fused_workgroup_size,
                    apply_scale,
                    scale,
                    config.precision,
                )
                .with_lines_per_workgroup(lines_per_workgroup);
                let pipeline_key = ComputePipelineCacheKey::fused_pow2_stage(shader_key.clone());

                let shader_label =
                    format!("wgpu_fft.axis_plan.fused_pow2.axis{axis}.n{axis_len}.shader");
                let pipeline_label =
                    format!("wgpu_fft.axis_plan.fused_pow2.axis{axis}.n{axis_len}.pipeline");
                let pipeline = with_device_pipeline_cache(device, |cache| {
                    cache.get_compute_pipeline(
                        device,
                        &pipeline_key,
                        &pipeline_label,
                        &shader_label,
                        || generate_fused_pow2_stage_wgsl_for_key(&shader_key),
                    )
                });

                stages.push(AxisStage {
                    axis,
                    kind: AxisStageKind::FusedPow2 {
                        axis_length: axis_len,
                    },
                    stride_complex,
                    apply_scale,
                    pipeline_key,
                    twiddle_lut_index,
                    workgroups_x: (total_lines as u32).div_ceil(lines_per_workgroup),
                    pipeline,
                });
            } else if fused_smooth_supported(
                axis_len,
                &axis_factors,
                config.precision,
                config.fused_workgroup_size,
                &device.limits(),
            ) {
                let apply_scale = apply_any_scale && final_axis;
                let total_lines = total_complex / axis_len;
                let lines_per_workgroup = fused_lines_per_workgroup(
                    axis_len,
                    stride_complex,
                    config.precision,
                    total_lines,
                    u64::from(device.limits().max_compute_workgroup_storage_size),
                );
                let shader_key = FusedSmoothStageKey::new(
                    config.shape.len(),
                    axis,
                    &config.shape,
                    axis_len,
                    stride_complex,
                    &axis_factors,
                    config.direction,
                    config.fused_workgroup_size,
                    apply_scale,
                    scale,
                    config.precision,
                )
                .with_lines_per_workgroup(lines_per_workgroup);
                let pipeline_key = ComputePipelineCacheKey::fused_smooth_stage(shader_key.clone());

                let shader_label =
                    format!("wgpu_fft.axis_plan.fused_smooth.axis{axis}.n{axis_len}.shader");
                let pipeline_label =
                    format!("wgpu_fft.axis_plan.fused_smooth.axis{axis}.n{axis_len}.pipeline");
                let pipeline = with_device_pipeline_cache(device, |cache| {
                    cache.get_compute_pipeline(
                        device,
                        &pipeline_key,
                        &pipeline_label,
                        &shader_label,
                        || generate_fused_smooth_stage_wgsl_for_key(&shader_key),
                    )
                });

                stages.push(AxisStage {
                    axis,
                    kind: AxisStageKind::FusedSmooth {
                        axis_length: axis_len,
                    },
                    stride_complex,
                    apply_scale,
                    pipeline_key,
                    twiddle_lut_index,
                    workgroups_x: (total_lines as u32).div_ceil(lines_per_workgroup),
                    pipeline,
                });
            } else {
                let mut ns = 1usize;
                for (stage_index, &radix) in axis_factors.iter().enumerate() {
                    ns *= radix;
                    let apply_scale =
                        apply_any_scale && final_axis && stage_index + 1 == axis_factors.len();
                    let shader_key = StockhamStageKey::new(
                        config.shape.len(),
                        axis,
                        &config.shape,
                        axis_len,
                        stride_complex,
                        radix,
                        ns,
                        config.direction,
                        config.workgroup_size,
                        apply_scale,
                        scale,
                        config.precision,
                    );
                    let pipeline_key = ComputePipelineCacheKey::stockham_stage(shader_key.clone());

                    let shader_label = format!(
                        "wgpu_fft.axis_plan.stockham.axis{axis}.radix{radix}.ns{ns}.shader"
                    );
                    let pipeline_label = format!(
                        "wgpu_fft.axis_plan.stockham.axis{axis}.radix{radix}.ns{ns}.pipeline"
                    );
                    let pipeline = with_device_pipeline_cache(device, |cache| {
                        cache.get_compute_pipeline(
                            device,
                            &pipeline_key,
                            &pipeline_label,
                            &shader_label,
                            || generate_stockham_radix_stage_wgsl_for_key(&shader_key),
                        )
                    });

                    stages.push(AxisStage {
                        axis,
                        kind: AxisStageKind::Stockham { radix, ns },
                        stride_complex,
                        apply_scale,
                        pipeline_key,
                        twiddle_lut_index,
                        workgroups_x: (total_complex_u32 / radix as u32)
                            .div_ceil(config.workgroup_size),
                        pipeline,
                    });
                }
            }

            factors.push(axis_factors);
        }

        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("wgpu_fft.axis_plan.params"),
            size: std::mem::size_of::<AxisParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let params = AxisParams {
            total: total_complex_u32,
            base_index: 0,
            line_offset: 0,
            element_base: 0,
        };
        queue.write_buffer(&params_buffer, 0, bytemuck::bytes_of(&params));

        let required_buffer_size_bytes =
            total_complex as u64 * config.precision.complex_size_bytes();
        let workspace_size_bytes = workspace_size_bytes_for_stage_count_with_precision(
            stages.len(),
            total_complex,
            config.precision,
        );
        let temp_buffer = if workspace_size_bytes > 0 {
            Some(create_axis_temp_buffer(
                device,
                "wgpu_fft.axis_plan.temp",
                workspace_size_bytes,
            )?)
        } else {
            None
        };

        Ok(Self {
            config,
            factors,
            stages,
            bind_group_layout,
            params_buffer,
            temp_buffer,
            twiddle_luts,
            required_buffer_size_bytes,
            workspace_size_bytes,
        })
    }

    pub(crate) fn factors(&self) -> &[Vec<usize>] {
        &self.factors
    }

    pub(crate) fn workspace_size_bytes(&self) -> u64 {
        self.workspace_size_bytes
    }

    pub(crate) fn twiddle_lut_storage_bytes(&self) -> u64 {
        self.twiddle_luts
            .iter()
            .fold(0u64, |bytes, lut| bytes.saturating_add(lut.buffer.size()))
    }

    pub(crate) fn graph_stage_kinds(&self) -> Vec<AxisStageKind> {
        self.stages.iter().map(|stage| stage.kind).collect()
    }

    pub(crate) fn execute(
        &self,
        device: &wgpu::Device,
        encoder: &mut CommandRecorder<'_>,
        input: &wgpu::Buffer,
        output: &wgpu::Buffer,
    ) -> Result<()> {
        self.execute_views(
            device,
            encoder,
            BufferView::whole(input),
            BufferView::whole(output),
        )
    }

    pub(crate) fn execute_views(
        &self,
        device: &wgpu::Device,
        encoder: &mut CommandRecorder<'_>,
        input: BufferView<'_>,
        output: BufferView<'_>,
    ) -> Result<()> {
        let input = input.prefix(self.required_buffer_size_bytes)?;
        let output = output.prefix(self.required_buffer_size_bytes)?;
        self.execute_impl(device, encoder, input, output, None)?;
        Ok(())
    }

    pub(crate) fn execute_views_with_workspace(
        &self,
        device: &wgpu::Device,
        encoder: &mut CommandRecorder<'_>,
        input: BufferView<'_>,
        output: BufferView<'_>,
        workspace: BufferView<'_>,
    ) -> Result<()> {
        let input = input.prefix(self.required_buffer_size_bytes)?;
        let output = output.prefix(self.required_buffer_size_bytes)?;
        let workspace = self.validate_workspace_view(workspace)?;
        self.execute_impl(device, encoder, input, output, Some(workspace))?;
        Ok(())
    }

    fn execute_impl(
        &self,
        device: &wgpu::Device,
        encoder: &mut CommandRecorder<'_>,
        input: BufferView<'_>,
        output: BufferView<'_>,
        workspace: Option<BufferView<'_>>,
    ) -> Result<()> {
        debug_assert!(
            !self.stages.is_empty(),
            "AxisPlan execution requires at least one FFT stage"
        );
        debug_assert_eq!(self.config.layout, AxisLayout::Interleaved);

        if self.stages.is_empty() {
            return Ok(());
        }

        let scheduler = WindowScheduler::for_device(device);
        let max_workgroups_per_dimension = max_workgroups_per_dimension(device);
        let mut src_slot = BufferSlot::Input;
        let mut dst_slot = if self.stages.len() % 2 == 1 {
            BufferSlot::Output
        } else {
            BufferSlot::Temp
        };

        for (stage_index, stage) in self.stages.iter().enumerate() {
            let stage_detail = stage.kind.detail();
            let src =
                self.resolve_buffer(src_slot, input.clone(), output.clone(), workspace.clone())?;
            let dst =
                self.resolve_buffer(dst_slot, input.clone(), output.clone(), workspace.clone())?;
            let element_format = self.config.precision.element_format();
            let src_resource = scheduler.storage_binding_resource(&src, element_format)?;
            let dst_resource = scheduler.storage_binding_resource(&dst, element_format)?;
            let twiddle_lut = &self.twiddle_luts[stage.twiddle_lut_index];
            let twiddle_view = BufferView::whole(twiddle_lut.buffer.as_ref());
            let twiddle_resource =
                scheduler.storage_binding_resource(&twiddle_view, element_format)?;

            let bind_group_label = format!(
                "wgpu_fft.axis_plan.bind_group.axis{}.{}.stride{}.scale{}.cache{}",
                stage.axis,
                stage_detail,
                stage.stride_complex,
                stage.apply_scale,
                stage.pipeline_key.stable_key()
            );
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&bind_group_label),
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: src_resource,
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: dst_resource,
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.params_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: twiddle_resource,
                    },
                ],
            });

            {
                let pass = encoder.pass();
                pass.set_pipeline(&stage.pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                let (x, y, z) = split_workgroups(stage.workgroups_x, max_workgroups_per_dimension)?;
                pass.dispatch_workgroups(x, y, z);
            }

            if stage_index + 1 < self.stages.len() {
                src_slot = dst_slot;
                dst_slot = next_stage_destination(src_slot)?;
            }
        }
        Ok(())
    }

    fn validate_workspace_view<'a>(&self, workspace: BufferView<'a>) -> Result<BufferView<'a>> {
        if workspace.size() < self.workspace_size_bytes {
            Err(FftError::WorkspaceTooSmall {
                required: self.workspace_size_bytes,
                actual: workspace.size(),
            })
        } else {
            workspace.prefix(self.workspace_size_bytes)
        }
    }

    fn resolve_buffer<'a>(
        &'a self,
        slot: BufferSlot,
        input: BufferView<'a>,
        output: BufferView<'a>,
        workspace: Option<BufferView<'a>>,
    ) -> Result<BufferView<'a>> {
        match slot {
            BufferSlot::Input => Ok(input),
            BufferSlot::Output => Ok(output),
            BufferSlot::Temp => {
                if let Some(workspace) = workspace {
                    Ok(workspace)
                } else if let Some(buffer) = self.temp_buffer.as_ref() {
                    Ok(BufferView::whole(buffer))
                } else {
                    Err(axis_plan_workspace_error())
                }
            }
        }
    }
}

fn workspace_size_bytes_for_stage_count_with_precision(
    stage_count: usize,
    total_complex: usize,
    precision: AxisPrecision,
) -> u64 {
    if stage_count > 1 {
        total_complex as u64 * precision.complex_size_bytes()
    } else {
        0
    }
}

fn create_axis_temp_buffer(
    device: &wgpu::Device,
    label: &'static str,
    size: u64,
) -> Result<wgpu::Buffer> {
    let max_buffer_size = device.limits().max_buffer_size;
    if size > max_buffer_size {
        return Err(FftError::HelperBufferTooLarge {
            helper_buffer: label,
            requested_bytes: size,
            max_buffer_size,
        });
    }
    Ok(device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    }))
}

fn validate_1d_workgroup_size(
    workgroup_size: u32,
    limits: &wgpu::Limits,
    stage: &'static str,
) -> Result<()> {
    if workgroup_size > limits.max_compute_invocations_per_workgroup
        || workgroup_size > limits.max_compute_workgroup_size_x
    {
        return Err(FftError::LargeGraphStageUnsupported {
            stage,
            reason: "configured workgroup size exceeds the active device compute limits",
        });
    }
    Ok(())
}

fn fused_pow2_supported(
    axis_length: usize,
    precision: AxisPrecision,
    workgroup_size: u32,
    limits: &wgpu::Limits,
) -> bool {
    fused_pow2_supported_by_limits(
        axis_length,
        precision,
        workgroup_size,
        u64::from(limits.max_compute_workgroup_storage_size),
        limits.max_compute_invocations_per_workgroup,
        limits.max_compute_workgroup_size_x,
    )
}

fn fused_pow2_supported_by_limits(
    axis_length: usize,
    precision: AxisPrecision,
    workgroup_size: u32,
    max_workgroup_storage_bytes: u64,
    max_invocations_per_workgroup: u32,
    max_workgroup_size_x: u32,
) -> bool {
    if axis_length < 2 || !axis_length.is_power_of_two() {
        return false;
    }
    let Some(scratch_bytes) = axis_length.checked_mul(precision.complex_size_bytes() as usize)
    else {
        return false;
    };
    scratch_bytes as u64 <= max_workgroup_storage_bytes
        && workgroup_size > 0
        && workgroup_size <= max_invocations_per_workgroup
        && workgroup_size <= max_workgroup_size_x
}

fn fused_smooth_supported(
    axis_length: usize,
    factors: &[usize],
    precision: AxisPrecision,
    workgroup_size: u32,
    limits: &wgpu::Limits,
) -> bool {
    fused_smooth_supported_by_limits(
        axis_length,
        factors,
        precision,
        workgroup_size,
        u64::from(limits.max_compute_workgroup_storage_size),
        limits.max_compute_invocations_per_workgroup,
        limits.max_compute_workgroup_size_x,
    )
}

fn fused_smooth_supported_by_limits(
    axis_length: usize,
    factors: &[usize],
    precision: AxisPrecision,
    workgroup_size: u32,
    max_workgroup_storage_bytes: u64,
    max_invocations_per_workgroup: u32,
    max_workgroup_size_x: u32,
) -> bool {
    if axis_length < 2
        || axis_length.is_power_of_two()
        || factors.len() < 2
        || factors.iter().product::<usize>() != axis_length
    {
        return false;
    }
    let Some(scratch_bytes) = axis_length.checked_mul(precision.complex_size_bytes() as usize)
    else {
        return false;
    };
    scratch_bytes as u64 <= max_workgroup_storage_bytes
        && workgroup_size > 0
        && workgroup_size <= max_invocations_per_workgroup
        && workgroup_size <= max_workgroup_size_x
}

pub(crate) fn generate_fused_pow2_stage_wgsl(config: &FusedPow2StageWgslConfig<'_>) -> String {
    debug_assert_eq!(config.rank, config.dims.len());
    debug_assert_eq!(config.axis_length, config.dims[config.axis]);
    debug_assert!(config.axis_length.is_power_of_two());
    debug_assert!(config.workgroup_size > 0);

    let maybe_scale = if config.apply_scale {
        let value = scaled_complex_expr("value", Some(config.scale_factor), config.precision);
        format!("    value = {value};\n")
    } else {
        String::new()
    };
    let line_base_fn = wgsl_line_base_fn(config.rank, config.axis, config.dims);
    let factors = crate::runtime::factor_supported_length(config.axis_length)
        .expect("a power-of-two FFT length must have a supported radix schedule");
    debug_assert!(factors.iter().all(|&radix| matches!(radix, 2 | 4 | 8)));
    let mut previous = 1usize;
    let mut radix_stages = String::new();
    for radix in factors {
        radix_stages.push_str(&generate_fused_radix_stage_wgsl(
            config.axis_length,
            radix,
            previous,
            config.direction,
            config.precision,
        ));
        previous *= radix;
    }
    debug_assert_eq!(previous, config.axis_length);

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
const LOG_N: u32 = {log_n}u;
const STRIDE: u32 = {stride}u;
const WORKGROUP_SIZE: u32 = {workgroup_size}u;

var<workgroup> scratch: array<vec2<f32>, {n}>;

fn bit_reverse(value: u32) -> u32 {{
  var x: u32 = value;
  var reversed: u32 = 0u;
  for (var bit: u32 = 0u; bit < LOG_N; bit = bit + 1u) {{
    reversed = (reversed << 1u) | (x & 1u);
    x = x >> 1u;
  }}
  return reversed;
}}

{line_base_fn}

@compute @workgroup_size({workgroup_size}, 1, 1)
fn main({entry_params}) {{
  {flat_workgroup_index}
  let firstLine: u32 = params.baseIndex / N;
  let totalLines: u32 = params.total / N;
  if (firstLine >= totalLines) {{
    return;
  }}
  let activeLines: u32 = totalLines - firstLine;
  if (wgFlat >= activeLines) {{
    return;
  }}

  let lineLocal: u32 = firstLine + wgFlat;
  let line: u32 = params.lineOffset + lineLocal;
  let baseLineGlobal: u32 = line_base(line);

  for (var p: u32 = lid.x; p < N; p = p + WORKGROUP_SIZE) {{
    let srcIdxGlobal: u32 = baseLineGlobal + p * STRIDE;
    let srcIdx: u32 = srcIdxGlobal - params.elementBase;
    scratch[bit_reverse(p)] = src[srcIdx];
  }}
  workgroupBarrier();

{radix_stages}
  for (var p: u32 = lid.x; p < N; p = p + WORKGROUP_SIZE) {{
    var value: vec2<f32> = scratch[p];
{maybe_scale}    let dstIdxGlobal: u32 = baseLineGlobal + p * STRIDE;
    let dstIdx: u32 = dstIdxGlobal - params.elementBase;
    dst[dstIdx] = value;
  }}
}}
"#,
            complex_wgsl = complex_wgsl(),
            n = config.axis_length,
            log_n = config.axis_length.ilog2(),
            stride = config.stride_complex,
            workgroup_size = config.workgroup_size,
            twiddle_lookup_wgsl = twiddle_lookup_wgsl(config.direction, config.precision),
            line_base_fn = line_base_fn,
            entry_params = crate::runtime::dispatch::WGSL_FLAT_ENTRY_PARAMS,
            flat_workgroup_index = crate::runtime::dispatch::WGSL_FLAT_WORKGROUP_INDEX,
            radix_stages = radix_stages,
            maybe_scale = maybe_scale,
        ),
        config.precision,
    )
}

/// Fused power-of-two kernel that transforms `lines` lines per workgroup.
///
/// Short lines leave most invocations of a one-line workgroup idle, and on
/// strided axes one line per workgroup makes neighbouring invocations read
/// addresses a whole stride apart. Here every radix stage spreads the units of
/// all lines over the invocations, and strided axes load element-major so
/// neighbouring invocations read neighbouring lines.
pub(crate) fn generate_fused_pow2_multiline_stage_wgsl(
    config: &FusedPow2StageWgslConfig<'_>,
    lines: usize,
) -> String {
    debug_assert_eq!(config.rank, config.dims.len());
    debug_assert_eq!(config.axis_length, config.dims[config.axis]);
    debug_assert!(config.axis_length.is_power_of_two() && config.axis_length >= 2);
    debug_assert!(lines > 1 && config.workgroup_size > 0);

    let maybe_scale = if config.apply_scale {
        let value = scaled_complex_expr("value", Some(config.scale_factor), config.precision);
        format!("      value = {value};\n")
    } else {
        String::new()
    };
    let line_base_fn = wgsl_line_base_fn(config.rank, config.axis, config.dims);
    let factors = crate::runtime::factor_supported_length(config.axis_length)
        .expect("a power-of-two FFT length must have a supported radix schedule");
    let mut previous = 1usize;
    let mut radix_stages = String::new();
    for radix in factors {
        radix_stages.push_str(&generate_fused_radix_stage_multiline_wgsl(
            config.axis_length,
            radix,
            previous,
            config.direction,
            config.precision,
        ));
        previous *= radix;
    }
    // Contiguous lines load line-major; strided lines load element-major so
    // neighbouring invocations touch neighbouring lines.
    let split = if config.stride_complex == 1 {
        "let lineSlot: u32 = e / N;\n    let p: u32 = e - lineSlot * N;"
    } else {
        "let lineSlot: u32 = e % LINES;\n    let p: u32 = e / LINES;"
    };

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
const LOG_N: u32 = {log_n}u;
const STRIDE: u32 = {stride}u;
const WORKGROUP_SIZE: u32 = {workgroup_size}u;
const LINES: u32 = {lines}u;

var<workgroup> scratch: array<vec2<f32>, {scratch_len}>;

{line_base_fn}

@compute @workgroup_size({workgroup_size}, 1, 1)
fn main({entry_params}) {{
  {flat_workgroup_index}
  let firstLine: u32 = params.baseIndex / N;
  let totalLines: u32 = params.total / N;
  if (firstLine >= totalLines) {{
    return;
  }}
  let activeLines: u32 = totalLines - firstLine;
  let groupLine: u32 = wgFlat * LINES;
  if (groupLine >= activeLines) {{
    return;
  }}
  let lineCount: u32 = min(LINES, activeLines - groupLine);
  let lineStart: u32 = params.lineOffset + firstLine + groupLine;

  for (var e: u32 = lid.x; e < LINES * N; e = e + WORKGROUP_SIZE) {{
    {split}
    if (lineSlot < lineCount) {{
      let srcIdx: u32 = line_base(lineStart + lineSlot) + p * STRIDE - params.elementBase;
      scratch[lineSlot * N + (reverseBits(p) >> (32u - LOG_N))] = src[srcIdx];
    }}
  }}
  workgroupBarrier();

{radix_stages}
  for (var e: u32 = lid.x; e < LINES * N; e = e + WORKGROUP_SIZE) {{
    {split}
    if (lineSlot < lineCount) {{
      var value: vec2<f32> = scratch[lineSlot * N + p];
{maybe_scale}      let dstIdx: u32 = line_base(lineStart + lineSlot) + p * STRIDE - params.elementBase;
      dst[dstIdx] = value;
    }}
  }}
}}
"#,
            complex_wgsl = complex_wgsl(),
            n = config.axis_length,
            log_n = config.axis_length.ilog2(),
            stride = config.stride_complex,
            workgroup_size = config.workgroup_size,
            lines = lines,
            scratch_len = config.axis_length * lines,
            twiddle_lookup_wgsl = twiddle_lookup_wgsl(config.direction, config.precision),
            line_base_fn = line_base_fn,
            entry_params = crate::runtime::dispatch::WGSL_FLAT_ENTRY_PARAMS,
            flat_workgroup_index = crate::runtime::dispatch::WGSL_FLAT_WORKGROUP_INDEX,
            radix_stages = radix_stages,
            maybe_scale = maybe_scale,
            split = split,
        ),
        config.precision,
    )
}

fn generate_fused_radix_stage_wgsl(
    axis_length: usize,
    radix: usize,
    previous: usize,
    direction: FftDirection,
    precision: AxisPrecision,
) -> String {
    format!(
        r#"  {{
    const RADIX: u32 = {radix}u;
    const PREVIOUS: u32 = {previous}u;
    const UNIT_COUNT: u32 = {unit_count}u;
    for (var unit: u32 = lid.x; unit < UNIT_COUNT; unit = unit + WORKGROUP_SIZE) {{
      let block: u32 = unit / PREVIOUS;
      let j: u32 = unit - block * PREVIOUS;
      let base: u32 = block * (RADIX * PREVIOUS) + j;
{body}    }}
    workgroupBarrier();
  }}
"#,
        unit_count = axis_length / radix,
        body = fused_radix_stage_body_wgsl(axis_length, radix, previous, direction, precision),
    )
}

/// One radix stage of the multi-line fused kernel: butterfly units of every
/// line in the workgroup share the invocations, and absent lines of a partial
/// last workgroup are skipped.
fn generate_fused_radix_stage_multiline_wgsl(
    axis_length: usize,
    radix: usize,
    previous: usize,
    direction: FftDirection,
    precision: AxisPrecision,
) -> String {
    format!(
        r#"  {{
    const RADIX: u32 = {radix}u;
    const PREVIOUS: u32 = {previous}u;
    const UNIT_COUNT: u32 = {unit_count}u;
    for (var slot: u32 = lid.x; slot < LINES * UNIT_COUNT; slot = slot + WORKGROUP_SIZE) {{
      let lineSlot: u32 = slot / UNIT_COUNT;
      if (lineSlot < lineCount) {{
      let unit: u32 = slot - lineSlot * UNIT_COUNT;
      let block: u32 = unit / PREVIOUS;
      let j: u32 = unit - block * PREVIOUS;
      let base: u32 = lineSlot * N + block * (RADIX * PREVIOUS) + j;
{body}      }}
    }}
    workgroupBarrier();
  }}
"#,
        unit_count = axis_length / radix,
        body = fused_radix_stage_body_wgsl(axis_length, radix, previous, direction, precision),
    )
}

/// Loads, twiddles, butterflies, and stores of one fused radix-2/4/8 unit.
fn fused_radix_stage_body_wgsl(
    axis_length: usize,
    radix: usize,
    previous: usize,
    direction: FftDirection,
    precision: AxisPrecision,
) -> String {
    debug_assert!(matches!(radix, 2 | 4 | 8));
    debug_assert_eq!(axis_length % radix, 0);
    let mut values = String::new();
    let mut writes = String::new();
    for q in 0..radix {
        values.push_str(&format!(
            "      var v{q}: vec2<f32> = scratch[base + {q}u * PREVIOUS];\n"
        ));
        writes.push_str(&format!("      scratch[base + {q}u * PREVIOUS] = v{q};\n"));
    }

    let step = axis_length / (radix * previous);
    let root4 = radix_root_wgsl(4, 1, direction, precision);
    let root8_1 = radix_root_wgsl(8, 1, direction, precision);
    let root8_3 = radix_root_wgsl(8, 3, direction, precision);
    let twiddles = match radix {
        8 => {
            format!(
                r#"      let z8: vec2<f32> = twiddle(j * {step}u);
      let z4: vec2<f32> = twiddle(j * {step2}u);
      let z2: vec2<f32> = twiddle(j * {step4}u);
      let root4: vec2<f32> = {root4};
      let root8_1: vec2<f32> = {root8_1};
      let root8_3: vec2<f32> = {root8_3};
      let z4_1: vec2<f32> = c_mul(z4, root4);
      let z8_1: vec2<f32> = c_mul(z8, root8_1);
      let z8_2: vec2<f32> = c_mul(z8, root4);
      let z8_3: vec2<f32> = c_mul(z8, root8_3);
"#,
                step2 = step * 2,
                step4 = step * 4,
            )
        }
        4 => {
            format!(
                r#"      let z4: vec2<f32> = twiddle(j * {step}u);
      let z2: vec2<f32> = twiddle(j * {step2}u);
      let root4: vec2<f32> = {root4};
      let z4_1: vec2<f32> = c_mul(z4, root4);
"#,
                step2 = step * 2,
            )
        }
        2 => {
            format!(
                r#"      let z2: vec2<f32> = twiddle(j * {step}u);
"#
            )
        }
        _ => unreachable!(),
    };

    let mut butterflies = String::new();
    match radix {
        8 => {
            append_fused_butterfly(&mut butterflies, 0, 1, "z2", "w2_0");
            append_fused_butterfly(&mut butterflies, 2, 3, "z2", "w2_1");
            append_fused_butterfly(&mut butterflies, 4, 5, "z2", "w2_2");
            append_fused_butterfly(&mut butterflies, 6, 7, "z2", "w2_3");
            append_fused_butterfly(&mut butterflies, 0, 2, "z4", "w4_0");
            append_fused_butterfly(&mut butterflies, 1, 3, "z4_1", "w4_1");
            append_fused_butterfly(&mut butterflies, 4, 6, "z4", "w4_2");
            append_fused_butterfly(&mut butterflies, 5, 7, "z4_1", "w4_3");
            append_fused_butterfly(&mut butterflies, 0, 4, "z8", "w8_0");
            append_fused_butterfly(&mut butterflies, 1, 5, "z8_1", "w8_1");
            append_fused_butterfly(&mut butterflies, 2, 6, "z8_2", "w8_2");
            append_fused_butterfly(&mut butterflies, 3, 7, "z8_3", "w8_3");
        }
        4 => {
            append_fused_butterfly(&mut butterflies, 0, 1, "z2", "w2_0");
            append_fused_butterfly(&mut butterflies, 2, 3, "z2", "w2_1");
            append_fused_butterfly(&mut butterflies, 0, 2, "z4", "w4_0");
            append_fused_butterfly(&mut butterflies, 1, 3, "z4_1", "w4_1");
        }
        2 => append_fused_butterfly(&mut butterflies, 0, 1, "z2", "w2_0"),
        _ => unreachable!(),
    }

    format!("{values}{twiddles}{butterflies}{writes}")
}

fn append_fused_butterfly(
    output: &mut String,
    lower: usize,
    upper: usize,
    twiddle: &str,
    suffix: &str,
) {
    output.push_str(&format!(
        "      let a_{suffix}: vec2<f32> = v{lower};\n      let t_{suffix}: vec2<f32> = c_mul({twiddle}, v{upper});\n      v{lower} = c_add(a_{suffix}, t_{suffix});\n      v{upper} = c_sub(a_{suffix}, t_{suffix});\n"
    ));
}

pub(crate) fn generate_fused_pow2_stage_wgsl_for_key(key: &FusedPow2StageKey) -> String {
    let config = FusedPow2StageWgslConfig {
        rank: key.rank,
        axis: key.axis,
        dims: &key.dims,
        axis_length: key.axis_length,
        stride_complex: key.stride_complex,
        direction: key.direction,
        workgroup_size: key.workgroup_size,
        apply_scale: key.apply_scale,
        scale_factor: key.scale_factor(),
        precision: key.precision,
    };
    if key.lines_per_workgroup > 1 {
        generate_fused_pow2_multiline_stage_wgsl(&config, key.lines_per_workgroup as usize)
    } else {
        generate_fused_pow2_stage_wgsl(&config)
    }
}

/// Lines per workgroup for a fused kernel on an axis of `axis_length`.
///
/// Targets about 2048 elements per workgroup so radix stages keep 256
/// invocations busy; strided axes take at least 8 adjacent lines so their
/// loads coalesce. Workgroup storage caps the result, and small transforms
/// keep at least 256 workgroups so their lines spread across the GPU instead
/// of queueing on a few compute units.
fn fused_lines_per_workgroup(
    axis_length: usize,
    stride_complex: usize,
    precision: AxisPrecision,
    total_lines: usize,
    max_workgroup_storage_bytes: u64,
) -> u32 {
    const TARGET_ELEMENTS: usize = 2048;
    const MIN_STRIDED_LINES: usize = 8;
    const MIN_WORKGROUPS: usize = 256;
    let mut lines = (TARGET_ELEMENTS / axis_length).max(1);
    if stride_complex > 1 {
        lines = lines.max(MIN_STRIDED_LINES);
    }
    let line_bytes = axis_length * precision.complex_size_bytes() as usize;
    let max_by_storage = (max_workgroup_storage_bytes as usize / line_bytes).max(1);
    let max_by_fill = (total_lines / MIN_WORKGROUPS).max(1);
    lines.min(max_by_storage).min(max_by_fill) as u32
}

pub(crate) fn generate_fused_smooth_stage_wgsl(config: &FusedSmoothStageWgslConfig<'_>) -> String {
    debug_assert_eq!(config.rank, config.dims.len());
    debug_assert_eq!(config.axis_length, config.dims[config.axis]);
    debug_assert!(!config.axis_length.is_power_of_two());
    debug_assert_eq!(config.factors.iter().product::<usize>(), config.axis_length);
    debug_assert!(config.workgroup_size > 0);

    let scale_factor = config.apply_scale.then_some(config.scale_factor);
    let line_base_fn = wgsl_line_base_fn(config.rank, config.axis, config.dims);
    let line_slot_count = config.axis_length.div_ceil(config.workgroup_size as usize);

    let mut ns = 1usize;
    let mut radix_stages = String::new();
    let final_stage_index = config.factors.len() - 1;
    for (stage_index, &radix) in config.factors.iter().enumerate() {
        ns *= radix;
        if stage_index == final_stage_index {
            radix_stages.push_str(&generate_fused_smooth_final_stage_wgsl(
                config.axis_length,
                config.stride_complex,
                radix,
                ns,
                scale_factor,
                config.direction,
                config.workgroup_size,
                config.precision,
            ));
        } else {
            radix_stages.push_str(&generate_fused_smooth_intermediate_stage_wgsl(
                config.axis_length,
                radix,
                ns,
                config.direction,
                config.workgroup_size,
                config.precision,
            ));
        }
    }
    debug_assert_eq!(ns, config.axis_length);

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
const WORKGROUP_SIZE: u32 = {workgroup_size}u;
const LINE_SLOT_COUNT: u32 = {line_slot_count}u;

var<workgroup> scratch: array<vec2<f32>, {n}>;

{line_base_fn}

@compute @workgroup_size({workgroup_size}, 1, 1)
fn main({entry_params}) {{
  {flat_workgroup_index}
  let firstLine: u32 = params.baseIndex / N;
  let totalLines: u32 = params.total / N;
  if (firstLine >= totalLines) {{
    return;
  }}
  let activeLines: u32 = totalLines - firstLine;
  if (wgFlat >= activeLines) {{
    return;
  }}

  let lineLocal: u32 = firstLine + wgFlat;
  let line: u32 = params.lineOffset + lineLocal;
  let baseLineGlobal: u32 = line_base(line);

  for (var slot: u32 = 0u; slot < LINE_SLOT_COUNT; slot = slot + 1u) {{
    let p: u32 = lid.x + slot * WORKGROUP_SIZE;
    if (p < N) {{
      let srcIdxGlobal: u32 = baseLineGlobal + p * STRIDE;
      let srcIdx: u32 = srcIdxGlobal - params.elementBase;
      scratch[p] = src[srcIdx];
    }}
  }}
  workgroupBarrier();

{radix_stages}}}
"#,
            complex_wgsl = complex_wgsl(),
            n = config.axis_length,
            stride = config.stride_complex,
            workgroup_size = config.workgroup_size,
            line_slot_count = line_slot_count,
            twiddle_lookup_wgsl = twiddle_lookup_wgsl(config.direction, config.precision),
            line_base_fn = line_base_fn,
            entry_params = crate::runtime::dispatch::WGSL_FLAT_ENTRY_PARAMS,
            flat_workgroup_index = crate::runtime::dispatch::WGSL_FLAT_WORKGROUP_INDEX,
            radix_stages = radix_stages,
        ),
        config.precision,
    )
}

/// Generates an in-place mixed-radix FFT fragment over an existing
/// `var<workgroup>` array. The caller must declare `N`, `WORKGROUP_SIZE`, the
/// scratch array, the complex helpers, and the named twiddle lookup function.
/// Every radix stage writes its results back to scratch and places uniform
/// workgroup barriers around that writeback.
pub(crate) fn generate_fused_scratch_fft_stages_wgsl(
    axis_length: usize,
    factors: &[usize],
    direction: FftDirection,
    workgroup_size: u32,
    scratch_name: &str,
    twiddle_fn_name: &str,
    precision: AxisPrecision,
) -> String {
    debug_assert!(axis_length >= 2);
    debug_assert!(!factors.is_empty());
    debug_assert!(factors
        .iter()
        .all(|radix| crate::runtime::SUPPORTED_RADICES.contains(radix)));
    debug_assert_eq!(factors.iter().product::<usize>(), axis_length);
    debug_assert!(workgroup_size > 0);
    debug_assert!(!scratch_name.is_empty());
    debug_assert!(!twiddle_fn_name.is_empty());

    let mut ns = 1usize;
    let mut stages = String::new();
    for &radix in factors {
        ns *= radix;
        stages.push_str(&generate_in_place_smooth_fft_stage_wgsl(
            axis_length,
            radix,
            ns,
            direction,
            workgroup_size,
            scratch_name,
            twiddle_fn_name,
            precision,
        ));
    }
    debug_assert_eq!(ns, axis_length);
    stages
}

fn generate_fused_smooth_intermediate_stage_wgsl(
    axis_length: usize,
    radix: usize,
    ns: usize,
    direction: FftDirection,
    workgroup_size: u32,
    precision: AxisPrecision,
) -> String {
    generate_in_place_smooth_fft_stage_wgsl(
        axis_length,
        radix,
        ns,
        direction,
        workgroup_size,
        "scratch",
        "twiddle",
        precision,
    )
}

#[allow(clippy::too_many_arguments)]
fn generate_in_place_smooth_fft_stage_wgsl(
    axis_length: usize,
    radix: usize,
    ns: usize,
    direction: FftDirection,
    workgroup_size: u32,
    scratch_name: &str,
    twiddle_fn_name: &str,
    precision: AxisPrecision,
) -> String {
    debug_assert_eq!(ns % radix, 0);
    debug_assert_eq!(axis_length % radix, 0);
    let ns_div_r = ns / radix;
    let n_div_r = axis_length / radix;
    let n_div_ns = axis_length / ns;
    let unit_count = axis_length / radix;
    let unit_slot_count = unit_count.div_ceil(workgroup_size as usize);
    let mut computes = String::new();
    let mut writes = String::new();

    for slot in 0..unit_slot_count {
        let mut stage_outputs = String::new();
        for output in 0..radix {
            let zero = if precision == AxisPrecision::Df64 {
                "vec4<f32>(0.0, 0.0, 0.0, 0.0)"
            } else {
                "vec2<f32>(0.0, 0.0)"
            };
            stage_outputs.push_str(&format!(
                "    var stageOut_{slot}_{output}: vec2<f32> = {zero};\n"
            ));
        }
        let butterfly = generate_fused_smooth_butterfly_math_wgsl(
            radix,
            ns_div_r,
            n_div_r,
            n_div_ns,
            direction,
            slot,
            scratch_name,
            twiddle_fn_name,
            precision,
        );
        computes.push_str(&format!(
            r#"    let unit_{slot}: u32 = lid.x + {slot}u * WORKGROUP_SIZE;
    let block_{slot}: u32 = unit_{slot} / {ns_div_r}u;
    let j_{slot}: u32 = unit_{slot} - block_{slot} * {ns_div_r}u;
{stage_outputs}    if (unit_{slot} < {unit_count}u) {{
      let base_{slot}: u32 = block_{slot} * {ns_div_r}u + j_{slot};
{butterfly}    }}
"#
        ));
        let mut slot_writes = String::new();
        for output in 0..radix {
            slot_writes.push_str(&format!(
                "      {scratch_name}[block_{slot} * {ns}u + {output}u * {ns_div_r}u + j_{slot}] = stageOut_{slot}_{output};\n"
            ));
        }
        writes.push_str(&format!(
            "    if (unit_{slot} < {unit_count}u) {{\n{slot_writes}    }}\n"
        ));
    }

    specialize_complex_wgsl(
        format!(
            r#"  {{ // fused smooth radix-{radix} butterflies
{computes}    workgroupBarrier();
{writes}    workgroupBarrier();
  }}
"#,
            radix = radix,
            computes = computes,
            writes = writes,
        ),
        precision,
    )
}

fn generate_fused_smooth_final_stage_wgsl(
    axis_length: usize,
    stride_complex: usize,
    radix: usize,
    ns: usize,
    scale_factor: Option<f64>,
    direction: FftDirection,
    workgroup_size: u32,
    precision: AxisPrecision,
) -> String {
    debug_assert_eq!(ns, axis_length);
    debug_assert_eq!(ns % radix, 0);
    let ns_div_r = ns / radix;
    let n_div_r = axis_length / radix;
    let n_div_ns = axis_length / ns;
    let unit_count = axis_length / radix;
    let unit_slot_count = unit_count.div_ceil(workgroup_size as usize);
    let mut slots = String::new();

    for slot in 0..unit_slot_count {
        let mut stage_outputs = String::new();
        for output in 0..radix {
            let zero = if precision == AxisPrecision::Df64 {
                "vec4<f32>(0.0, 0.0, 0.0, 0.0)"
            } else {
                "vec2<f32>(0.0, 0.0)"
            };
            stage_outputs.push_str(&format!(
                "      var stageOut_{slot}_{output}: vec2<f32> = {zero};\n"
            ));
        }
        let butterfly = generate_fused_smooth_butterfly_math_wgsl(
            radix, ns_div_r, n_div_r, n_div_ns, direction, slot, "scratch", "twiddle", precision,
        );
        let mut stores = String::new();
        for output in 0..radix {
            let value = format!("stageOut_{slot}_{output}");
            let value = scaled_complex_expr(&value, scale_factor, precision);
            stores.push_str(&format!(
                r#"      let value_{slot}_{output}: vec2<f32> = {value};
      let p_{slot}_{output}: u32 = block_{slot} * {ns}u + {output}u * {ns_div_r}u + j_{slot};
      let dstIdxGlobal_{slot}_{output}: u32 = baseLineGlobal + p_{slot}_{output} * {stride}u;
      let dstIdx_{slot}_{output}: u32 = dstIdxGlobal_{slot}_{output} - params.elementBase;
      dst[dstIdx_{slot}_{output}] = value_{slot}_{output};
"#,
                stride = stride_complex,
            ));
        }
        slots.push_str(&format!(
            r#"    let unit_{slot}: u32 = lid.x + {slot}u * WORKGROUP_SIZE;
    if (unit_{slot} < {unit_count}u) {{
      let block_{slot}: u32 = unit_{slot} / {ns_div_r}u;
      let j_{slot}: u32 = unit_{slot} - block_{slot} * {ns_div_r}u;
      let base_{slot}: u32 = block_{slot} * {ns_div_r}u + j_{slot};
{stage_outputs}{butterfly}{stores}    }}
"#
        ));
    }

    specialize_complex_wgsl(
        format!("  {{ // fused smooth radix-{radix} butterflies\n{slots}\n  }}\n"),
        precision,
    )
}

#[allow(clippy::too_many_arguments)]
fn generate_fused_smooth_butterfly_math_wgsl(
    radix: usize,
    ns_div_r: usize,
    n_div_r: usize,
    n_div_ns: usize,
    direction: FftDirection,
    slot: usize,
    scratch_name: &str,
    twiddle_fn_name: &str,
    precision: AxisPrecision,
) -> String {
    // Workgroup-scratch form of the same unit-centric Stockham factorization
    // used by the multi-pass generator.
    let mut shader = String::new();
    shader.push_str(&format!(
        "      let x_{slot}_0: vec2<f32> = {scratch_name}[base_{slot}];\n"
    ));
    for q in 1..radix {
        if ns_div_r == 1 {
            shader.push_str(&format!(
                "      let x_{slot}_{q}: vec2<f32> = {scratch_name}[base_{slot} + {q}u * {n_div_r}u];\n"
            ));
        } else {
            shader.push_str(&format!(
                "      let x_{slot}_{q}: vec2<f32> = c_mul({twiddle_fn_name}(j_{slot} * {}u), {scratch_name}[base_{slot} + {q}u * {n_div_r}u]);\n",
                q * n_div_ns,
            ));
        }
    }

    for output in 0..radix {
        shader.push_str(&format!(
            "      var out_{slot}_{output}: vec2<f32> = x_{slot}_0;\n"
        ));
        for q in 1..radix {
            let power = (output * q) % radix;
            if power == 0 {
                shader.push_str(&format!(
                    "      out_{slot}_{output} = c_add(out_{slot}_{output}, x_{slot}_{q});\n"
                ));
            } else {
                let root = radix_root_wgsl(radix, power, direction, precision);
                shader.push_str(&format!(
                    "      out_{slot}_{output} = c_add(out_{slot}_{output}, c_mul({root}, x_{slot}_{q}));\n"
                ));
            }
        }
        shader.push_str(&format!(
            "      stageOut_{slot}_{output} = out_{slot}_{output};\n"
        ));
    }
    shader
}

pub(crate) fn generate_fused_smooth_stage_wgsl_for_key(key: &FusedSmoothStageKey) -> String {
    let config = FusedSmoothStageWgslConfig {
        rank: key.rank,
        axis: key.axis,
        dims: &key.dims,
        axis_length: key.axis_length,
        stride_complex: key.stride_complex,
        factors: &key.factors,
        direction: key.direction,
        workgroup_size: key.workgroup_size,
        apply_scale: key.apply_scale,
        scale_factor: key.scale_factor(),
        precision: key.precision,
    };
    if key.lines_per_workgroup > 1 {
        generate_fused_smooth_multiline_stage_wgsl(&config, key.lines_per_workgroup as usize)
    } else {
        generate_fused_smooth_stage_wgsl(&config)
    }
}

/// Fused smooth-radix kernel that transforms `lines` lines per workgroup; see
/// [`generate_fused_pow2_multiline_stage_wgsl`]. Every stage stays in
/// workgroup memory and a final store pass writes the lines out, so strided
/// axes store element-major as well as load that way.
pub(crate) fn generate_fused_smooth_multiline_stage_wgsl(
    config: &FusedSmoothStageWgslConfig<'_>,
    lines: usize,
) -> String {
    debug_assert_eq!(config.rank, config.dims.len());
    debug_assert_eq!(config.axis_length, config.dims[config.axis]);
    debug_assert_eq!(config.factors.iter().product::<usize>(), config.axis_length);
    debug_assert!(lines > 1 && config.workgroup_size > 0);

    let maybe_scale = if config.apply_scale {
        let value = scaled_complex_expr("value", Some(config.scale_factor), config.precision);
        format!("      value = {value};\n")
    } else {
        String::new()
    };
    let line_base_fn = wgsl_line_base_fn(config.rank, config.axis, config.dims);
    let mut ns = 1usize;
    let mut radix_stages = String::new();
    for &radix in config.factors {
        ns *= radix;
        radix_stages.push_str(&generate_in_place_smooth_fft_stage_multiline_wgsl(
            config.axis_length,
            radix,
            ns,
            config.direction,
            config.workgroup_size,
            lines,
            config.precision,
        ));
    }
    debug_assert_eq!(ns, config.axis_length);
    let split = if config.stride_complex == 1 {
        "let lineSlot: u32 = e / N;\n    let p: u32 = e - lineSlot * N;"
    } else {
        "let lineSlot: u32 = e % LINES;\n    let p: u32 = e / LINES;"
    };

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
const WORKGROUP_SIZE: u32 = {workgroup_size}u;
const LINES: u32 = {lines}u;

var<workgroup> scratch: array<vec2<f32>, {scratch_len}>;

{line_base_fn}

@compute @workgroup_size({workgroup_size}, 1, 1)
fn main({entry_params}) {{
  {flat_workgroup_index}
  let firstLine: u32 = params.baseIndex / N;
  let totalLines: u32 = params.total / N;
  if (firstLine >= totalLines) {{
    return;
  }}
  let activeLines: u32 = totalLines - firstLine;
  let groupLine: u32 = wgFlat * LINES;
  if (groupLine >= activeLines) {{
    return;
  }}
  let lineCount: u32 = min(LINES, activeLines - groupLine);
  let lineStart: u32 = params.lineOffset + firstLine + groupLine;

  for (var e: u32 = lid.x; e < LINES * N; e = e + WORKGROUP_SIZE) {{
    {split}
    if (lineSlot < lineCount) {{
      let srcIdx: u32 = line_base(lineStart + lineSlot) + p * STRIDE - params.elementBase;
      scratch[lineSlot * N + p] = src[srcIdx];
    }}
  }}
  workgroupBarrier();

{radix_stages}
  for (var e: u32 = lid.x; e < LINES * N; e = e + WORKGROUP_SIZE) {{
    {split}
    if (lineSlot < lineCount) {{
      var value: vec2<f32> = scratch[lineSlot * N + p];
{maybe_scale}      let dstIdx: u32 = line_base(lineStart + lineSlot) + p * STRIDE - params.elementBase;
      dst[dstIdx] = value;
    }}
  }}
}}
"#,
            complex_wgsl = complex_wgsl(),
            n = config.axis_length,
            stride = config.stride_complex,
            workgroup_size = config.workgroup_size,
            lines = lines,
            scratch_len = config.axis_length * lines,
            twiddle_lookup_wgsl = twiddle_lookup_wgsl(config.direction, config.precision),
            line_base_fn = line_base_fn,
            entry_params = crate::runtime::dispatch::WGSL_FLAT_ENTRY_PARAMS,
            flat_workgroup_index = crate::runtime::dispatch::WGSL_FLAT_WORKGROUP_INDEX,
            radix_stages = radix_stages,
            maybe_scale = maybe_scale,
            split = split,
        ),
        config.precision,
    )
}

/// One in-place smooth radix stage over `lines` lines of `axis_length`
/// elements in `scratch`, with each line's units on consecutive invocations.
#[allow(clippy::too_many_arguments)]
fn generate_in_place_smooth_fft_stage_multiline_wgsl(
    axis_length: usize,
    radix: usize,
    ns: usize,
    direction: FftDirection,
    workgroup_size: u32,
    lines: usize,
    precision: AxisPrecision,
) -> String {
    debug_assert_eq!(ns % radix, 0);
    debug_assert_eq!(axis_length % radix, 0);
    let ns_div_r = ns / radix;
    let n_div_r = axis_length / radix;
    let n_div_ns = axis_length / ns;
    let unit_count = axis_length / radix;
    let unit_slot_count = (unit_count * lines).div_ceil(workgroup_size as usize);
    let zero = if precision == AxisPrecision::Df64 {
        "vec4<f32>(0.0, 0.0, 0.0, 0.0)"
    } else {
        "vec2<f32>(0.0, 0.0)"
    };
    let mut computes = String::new();
    let mut writes = String::new();

    for slot in 0..unit_slot_count {
        let mut stage_outputs = String::new();
        let mut slot_writes = String::new();
        for output in 0..radix {
            stage_outputs.push_str(&format!(
                "    var stageOut_{slot}_{output}: vec2<f32> = {zero};\n"
            ));
            slot_writes.push_str(&format!(
                "      scratch[lineOffset_{slot} + block_{slot} * {ns}u + {output}u * {ns_div_r}u + j_{slot}] = stageOut_{slot}_{output};\n"
            ));
        }
        let butterfly = generate_fused_smooth_butterfly_math_wgsl(
            radix, ns_div_r, n_div_r, n_div_ns, direction, slot, "scratch", "twiddle", precision,
        );
        computes.push_str(&format!(
            r#"    let slotUnit_{slot}: u32 = lid.x + {slot}u * WORKGROUP_SIZE;
    let lineSlot_{slot}: u32 = slotUnit_{slot} / {unit_count}u;
    let unit_{slot}: u32 = slotUnit_{slot} - lineSlot_{slot} * {unit_count}u;
    let block_{slot}: u32 = unit_{slot} / {ns_div_r}u;
    let j_{slot}: u32 = unit_{slot} - block_{slot} * {ns_div_r}u;
    let lineOffset_{slot}: u32 = lineSlot_{slot} * N;
{stage_outputs}    if (lineSlot_{slot} < lineCount) {{
      let base_{slot}: u32 = lineOffset_{slot} + block_{slot} * {ns_div_r}u + j_{slot};
{butterfly}    }}
"#
        ));
        writes.push_str(&format!(
            "    if (lineSlot_{slot} < lineCount) {{\n{slot_writes}    }}\n"
        ));
    }

    specialize_complex_wgsl(
        format!(
            r#"  {{ // fused smooth radix-{radix} butterflies, {lines} lines
{computes}    workgroupBarrier();
{writes}    workgroupBarrier();
  }}
"#
        ),
        precision,
    )
}

pub(crate) fn generate_stockham_radix_stage_wgsl(config: &StockhamStageWgslConfig<'_>) -> String {
    debug_assert_eq!(config.rank, config.dims.len());
    debug_assert_eq!(config.axis_length, config.dims[config.axis]);
    debug_assert_eq!(config.ns % config.radix, 0);
    debug_assert_eq!(config.axis_length % config.radix, 0);

    let scale_factor = config.apply_scale.then_some(config.scale_factor);
    let complex_wgsl = complex_wgsl();
    let line_base_fn = wgsl_line_base_fn(config.rank, config.axis, config.dims);
    let ns_div_r = config.ns / config.radix;
    let n_div_r = config.axis_length / config.radix;
    let n_div_ns = config.axis_length / config.ns;

    // One unit owns all radix outputs for a (block, j) butterfly. Splitting
    // W_NS^((t*NS/R+j)*q) into W_N^(j*q*N/NS) * W_R^(t*q) lets us load each
    // input and exact external LUT power once. Since j < NS/R and q < R,
    // j*q < NS, so the generated u32 LUT index is both reduced and < N.
    let mut inputs = String::new();
    for q in 0..config.radix {
        inputs.push_str(&format!(
            "  let srcIdxGlobal_{q}: u32 = baseLineGlobal + (base + {q}u * N_DIV_R) * STRIDE;\n  let srcIdx_{q}: u32 = srcIdxGlobal_{q} - params.elementBase;\n"
        ));
        if q == 0 || ns_div_r == 1 {
            inputs.push_str(&format!("  let x_{q}: vec2<f32> = src[srcIdx_{q}];\n"));
        } else {
            inputs.push_str(&format!(
                "  let x_{q}: vec2<f32> = c_mul(twiddle(j * {}u), src[srcIdx_{q}]);\n",
                q * n_div_ns,
            ));
        }
    }

    let mut outputs = String::new();
    for output in 0..config.radix {
        outputs.push_str(&format!("  var out_{output}: vec2<f32> = x_0;\n"));
        for q in 1..config.radix {
            let power = (output * q) % config.radix;
            if power == 0 {
                outputs.push_str(&format!("  out_{output} = c_add(out_{output}, x_{q});\n"));
            } else {
                let root = radix_root_wgsl(config.radix, power, config.direction, config.precision);
                outputs.push_str(&format!(
                    "  out_{output} = c_add(out_{output}, c_mul({root}, x_{q}));\n"
                ));
            }
        }
        let value = scaled_complex_expr(&format!("out_{output}"), scale_factor, config.precision);
        outputs.push_str(&format!(
            "  let value_{output}: vec2<f32> = {value};\n  let p_{output}: u32 = block * NS + {output}u * NS_DIV_R + j;\n  let dstIdxGlobal_{output}: u32 = baseLineGlobal + p_{output} * STRIDE;\n  let dstIdx_{output}: u32 = dstIdxGlobal_{output} - params.elementBase;\n  dst[dstIdx_{output}] = value_{output};\n"
        ));
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
const RADIX: u32 = {radix}u;
const NS: u32 = {ns}u;
const NS_DIV_R: u32 = {ns_div_r}u;
const N_DIV_R: u32 = {n_div_r}u;
const N_DIV_NS: u32 = {n_div_ns}u;
const UNITS_PER_LINE: u32 = N_DIV_R;
const STRIDE: u32 = {stride}u;

{line_base_fn}

@compute @workgroup_size({workgroup_size}, 1, 1)
fn main({entry_params}) {{
  {flat_workgroup_index}
  let firstUnit: u32 = params.baseIndex / RADIX;
  let totalUnits: u32 = params.total / RADIX;
  if (wgFlat > totalUnits / {workgroup_size}u) {{
    return;
  }}
  let unit: u32 = firstUnit + wgFlat * {workgroup_size}u + lid.x;
  if (unit >= totalUnits) {{
    return;
  }}

  let lineLocal: u32 = unit / UNITS_PER_LINE;
  let line: u32 = params.lineOffset + lineLocal;
  let unitInLine: u32 = unit - lineLocal * UNITS_PER_LINE;
  let baseLineGlobal: u32 = line_base(line);

  let block: u32 = unitInLine / NS_DIV_R;
  let j: u32 = unitInLine - block * NS_DIV_R;
  let base: u32 = block * NS_DIV_R + j;

{inputs}{outputs}
}}
"#,
            complex_wgsl = complex_wgsl,
            n = config.axis_length,
            radix = config.radix,
            ns = config.ns,
            ns_div_r = ns_div_r,
            n_div_r = n_div_r,
            n_div_ns = n_div_ns,
            stride = config.stride_complex,
            twiddle_lookup_wgsl = twiddle_lookup_wgsl(config.direction, config.precision),
            line_base_fn = line_base_fn,
            workgroup_size = config.workgroup_size,
            entry_params = crate::runtime::dispatch::WGSL_FLAT_ENTRY_PARAMS,
            flat_workgroup_index = crate::runtime::dispatch::WGSL_FLAT_WORKGROUP_INDEX,
            inputs = inputs,
            outputs = outputs,
        ),
        config.precision,
    )
}

pub(crate) fn generate_stockham_radix_stage_wgsl_for_key(key: &StockhamStageKey) -> String {
    generate_stockham_radix_stage_wgsl(&StockhamStageWgslConfig {
        rank: key.rank,
        axis: key.axis,
        dims: &key.dims,
        axis_length: key.axis_length,
        stride_complex: key.stride_complex,
        radix: key.radix,
        ns: key.ns,
        direction: key.direction,
        workgroup_size: key.workgroup_size,
        apply_scale: key.apply_scale,
        scale_factor: key.scale_factor(),
        precision: key.precision,
    })
}

fn complex_wgsl() -> &'static str {
    r#"fn c_add(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
  return a + b;
}

fn c_sub(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
  return a - b;
}

fn c_mul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
  return vec2<f32>(
    a.x * b.x - a.y * b.y,
    a.x * b.y + a.y * b.x
  );
}"#
}

fn twiddle_lookup_wgsl(direction: FftDirection, precision: AxisPrecision) -> String {
    if precision == AxisPrecision::Df64 {
        return match direction {
            FftDirection::Forward => r#"fn twiddle(index: u32) -> vec4<f32> {
  return axisTwiddles[index];
}"#
            .to_owned(),
            FftDirection::Inverse => r#"fn twiddle(index: u32) -> vec4<f32> {
  let value: vec4<f32> = axisTwiddles[index];
  return vec4<f32>(value.x, value.y, -value.z, -value.w);
}"#
            .to_owned(),
        };
    }

    match direction {
        FftDirection::Forward => r#"fn twiddle(index: u32) -> vec2<f32> {
  return axisTwiddles[index];
}"#
        .to_owned(),
        FftDirection::Inverse => r#"fn twiddle(index: u32) -> vec2<f32> {
  let value: vec2<f32> = axisTwiddles[index];
  return vec2<f32>(value.x, -value.y);
}"#
        .to_owned(),
    }
}

fn radix_root_wgsl(
    radix: usize,
    power: usize,
    direction: FftDirection,
    precision: AxisPrecision,
) -> String {
    debug_assert!(radix > 0);
    debug_assert!(power < radix);
    let sign = match direction {
        FftDirection::Forward => -1.0f64,
        FftDirection::Inverse => 1.0f64,
    };
    let angle = sign * std::f64::consts::TAU * power as f64 / radix as f64;
    let (sin, cos) = angle.sin_cos();
    precision.format_wgsl_complex(cos, sin)
}

fn specialize_complex_wgsl(source: String, precision: AxisPrecision) -> String {
    if precision != AxisPrecision::Df64 {
        return precision.specialize_wgsl(source);
    }

    let is_full_shader = source.contains(complex_wgsl());
    let source = source
        .replace(complex_wgsl(), df64_complex_aliases_wgsl())
        .replace("vec2<f32>", "vec4<f32>");
    if is_full_shader {
        format!("{}\n{source}", crate::kernels::DF64_WGSL)
    } else {
        source
    }
}

fn df64_complex_aliases_wgsl() -> &'static str {
    r#"fn c_add(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
  return df64_complex_add(a, b);
}

fn c_sub(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
  return df64_complex_sub(a, b);
}

fn c_mul(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
  return df64_complex_mul(a, b);
}"#
}

fn scaled_complex_expr(value: &str, scale_factor: Option<f64>, precision: AxisPrecision) -> String {
    let Some(scale_factor) = scale_factor else {
        return value.to_owned();
    };
    match precision {
        AxisPrecision::F32 | AxisPrecision::F64 => {
            let scale = precision.format_wgsl_scalar(scale_factor);
            format!("{value} * vec2<f32>({scale}, {scale})")
        }
        AxisPrecision::Df64 => {
            let scale = crate::math::DoubleFloat::from_f64(scale_factor);
            format!(
                "df64_complex_scale({value}, Df64({}, {}))",
                format_wgsl_f32(scale.hi),
                format_wgsl_f32(scale.lo)
            )
        }
    }
}

fn wgsl_line_base_fn(rank: usize, axis: usize, dims: &[usize]) -> String {
    assert!(rank >= 1, "rank must be at least one");
    assert_eq!(rank, dims.len(), "dims length must match rank");
    assert!(axis < rank, "axis must be in range");

    let n_total = product(dims);
    let lines_per_batch = dims
        .iter()
        .enumerate()
        .filter_map(|(index, &dim)| (index != axis).then_some(dim))
        .product::<usize>();

    let mut strides = vec![1usize; rank];
    for index in 1..rank {
        strides[index] = strides[index - 1] * dims[index - 1];
    }

    let mut decode = String::new();
    let mut rem_name = String::from("rem");
    let mut rem_init = true;
    for dim_index in 0..rank {
        if dim_index == axis {
            continue;
        }

        if rem_init {
            decode.push_str("  var rem: u32 = line - b * lines_per_batch;\n");
            rem_init = false;
        }

        let coord_name = format!("c{dim_index}");
        let next_rem = format!("{rem_name}_{dim_index}");
        decode.push_str(&format!(
            "  let {coord_name}: u32 = {rem_name} % {dim}u;\n",
            dim = dims[dim_index]
        ));
        decode.push_str(&format!(
            "  base = base + {coord_name} * {stride}u;\n",
            stride = strides[dim_index]
        ));
        decode.push_str(&format!(
            "  var {next_rem}: u32 = {rem_name} / {dim}u;\n",
            dim = dims[dim_index]
        ));
        rem_name = next_rem;
    }

    if decode.is_empty() {
        decode.push_str("  // axis-only line (rank=1): no non-axis coordinates\n");
    }

    format!(
        r#"fn line_base(line: u32) -> u32 {{
  let lines_per_batch: u32 = {lines_per_batch}u;
  let b: u32 = line / lines_per_batch;
  var base: u32 = b * {n_total}u;
{decode}  return base;
}}"#,
        lines_per_batch = lines_per_batch,
        n_total = n_total,
        decode = decode,
    )
}

fn stride_for_axis(dims: &[usize], axis: usize) -> usize {
    dims.iter().take(axis).product()
}

fn product(values: &[usize]) -> usize {
    values.iter().product()
}

fn format_wgsl_f32(value: f32) -> String {
    crate::runtime::nd_wgsl::format_wgsl_f32_roundtrip(value)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::{reference_c2c_nd, Complex32};

    fn wgsl_for(radix: usize, axis_length: usize, ns: usize) -> String {
        wgsl_for_precision(radix, axis_length, ns, AxisPrecision::F32)
    }

    fn wgsl_for_precision(
        radix: usize,
        axis_length: usize,
        ns: usize,
        precision: AxisPrecision,
    ) -> String {
        let dims = [axis_length];
        generate_stockham_radix_stage_wgsl(&StockhamStageWgslConfig {
            rank: 1,
            axis: 0,
            dims: &dims,
            axis_length,
            stride_complex: 1,
            radix,
            ns,
            direction: FftDirection::Forward,
            workgroup_size: WORKGROUP_SIZE,
            apply_scale: false,
            scale_factor: 1.0,
            precision,
        })
    }

    fn fused_wgsl_for(axis_length: usize, direction: FftDirection) -> String {
        fused_wgsl_for_precision(axis_length, direction, AxisPrecision::F32)
    }

    fn fused_wgsl_for_precision(
        axis_length: usize,
        direction: FftDirection,
        precision: AxisPrecision,
    ) -> String {
        let dims = [axis_length];
        generate_fused_pow2_stage_wgsl(&FusedPow2StageWgslConfig {
            rank: 1,
            axis: 0,
            dims: &dims,
            axis_length,
            stride_complex: 1,
            direction,
            workgroup_size: FUSED_POW2_WORKGROUP_SIZE,
            apply_scale: false,
            scale_factor: 1.0,
            precision,
        })
    }

    fn fused_smooth_wgsl_for(axis_length: usize, direction: FftDirection) -> String {
        fused_smooth_wgsl_for_precision(axis_length, direction, AxisPrecision::F32)
    }

    fn fused_smooth_wgsl_for_precision(
        axis_length: usize,
        direction: FftDirection,
        precision: AxisPrecision,
    ) -> String {
        let dims = [axis_length];
        let factors = crate::runtime::factor_supported_length(axis_length).unwrap();
        generate_fused_smooth_stage_wgsl(&FusedSmoothStageWgslConfig {
            rank: 1,
            axis: 0,
            dims: &dims,
            axis_length,
            stride_complex: 1,
            factors: &factors,
            direction,
            workgroup_size: FUSED_SMOOTH_WORKGROUP_SIZE,
            apply_scale: false,
            scale_factor: 1.0,
            precision,
        })
    }

    fn simulate_fused_smooth_1d(
        input: &[Complex32],
        factors: &[usize],
        direction: FftDirection,
    ) -> Vec<Complex32> {
        let n = input.len();
        let twiddles = twiddle_lut_f32(n);
        let mut scratch = input.to_vec();
        let mut ns = 1usize;
        for &radix in factors {
            ns *= radix;
            let ns_div_r = ns / radix;
            let n_div_r = n / radix;
            let n_div_ns = n / ns;
            let mut roots = twiddle_lut_f32(radix);
            if direction == FftDirection::Inverse {
                for root in &mut roots {
                    root.im = -root.im;
                }
            }
            let mut stage = vec![Complex32::default(); n];
            for unit in 0..n_div_r {
                let block = unit / ns_div_r;
                let j = unit - block * ns_div_r;
                let base = block * ns_div_r + j;
                let mut inputs = Vec::with_capacity(radix);
                for q in 0..radix {
                    let mut value = scratch[base + q * n_div_r];
                    if q != 0 && ns_div_r != 1 {
                        let mut w = twiddles[j * q * n_div_ns];
                        if direction == FftDirection::Inverse {
                            w.im = -w.im;
                        }
                        value = Complex32::new(
                            w.re * value.re - w.im * value.im,
                            w.re * value.im + w.im * value.re,
                        );
                    }
                    inputs.push(value);
                }
                for output in 0..radix {
                    let mut out = inputs[0];
                    for (q, &value) in inputs.iter().enumerate().skip(1) {
                        let power = (output * q) % radix;
                        if power == 0 {
                            out.re += value.re;
                            out.im += value.im;
                        } else {
                            let root = roots[power];
                            out.re += root.re * value.re - root.im * value.im;
                            out.im += root.re * value.im + root.im * value.re;
                        }
                    }
                    stage[block * ns + output * ns_div_r + j] = out;
                }
            }
            scratch = stage;
        }
        scratch
    }

    #[test]
    fn generated_wgsl_contains_radix_three_constants() {
        let wgsl = wgsl_for(3, 15, 3);
        assert!(wgsl.contains("const N: u32 = 15u;"));
        assert!(wgsl.contains("const RADIX: u32 = 3u;"));
        assert!(wgsl.contains("const NS: u32 = 3u;"));
        assert!(wgsl.contains("const NS_DIV_R: u32 = 1u;"));
        assert!(wgsl.contains("const N_DIV_R: u32 = 5u;"));
        assert!(wgsl.contains("const STRIDE: u32 = 1u;"));
        assert!(wgsl.contains("const N_DIV_NS: u32 = 5u;"));
        assert!(wgsl.contains("@binding(3) var<storage, read> axisTwiddles"));
    }

    #[test]
    fn invalid_axis_plan_buffer_flow_returns_stage_error() {
        assert_eq!(
            next_stage_destination(BufferSlot::Input),
            Err(FftError::LargeGraphStageUnsupported {
                stage: "axis-plan-buffer-flow",
                reason: "mixed-radix stage buffer flow attempted to use input as destination",
            })
        );
        assert_eq!(
            axis_plan_workspace_error(),
            FftError::LargeGraphStageUnsupported {
                stage: "axis-plan-workspace",
                reason: "multi-stage AxisPlan requires temp storage",
            }
        );
    }

    #[test]
    fn generated_wgsl_contains_radix_five_constants() {
        let wgsl = wgsl_for(5, 15, 5);
        assert!(wgsl.contains("const N: u32 = 15u;"));
        assert!(wgsl.contains("const RADIX: u32 = 5u;"));
        assert!(wgsl.contains("const NS: u32 = 5u;"));
        assert!(wgsl.contains("const NS_DIV_R: u32 = 1u;"));
        assert!(wgsl.contains("const N_DIV_R: u32 = 3u;"));
    }

    #[test]
    fn generated_wgsl_contains_radix_eight_constants() {
        let wgsl = wgsl_for(8, 8, 8);
        assert!(wgsl.contains("const N: u32 = 8u;"));
        assert!(wgsl.contains("const RADIX: u32 = 8u;"));
        assert!(wgsl.contains("const NS: u32 = 8u;"));
        assert!(wgsl.contains("const NS_DIV_R: u32 = 1u;"));
        assert!(wgsl.contains("const N_DIV_R: u32 = 1u;"));
    }

    #[test]
    fn radix_root_literals_are_f64_generated_then_rounded_once() {
        for radix in [2, 3, 4, 5, 7, 8, 11, 13] {
            for power in 0..radix {
                for direction in [FftDirection::Forward, FftDirection::Inverse] {
                    let sign = if direction == FftDirection::Forward {
                        -1.0
                    } else {
                        1.0
                    };
                    let angle = sign * std::f64::consts::TAU * power as f64 / radix as f64;
                    let (sin, cos) = angle.sin_cos();
                    assert_eq!(
                        radix_root_wgsl(radix, power, direction, AxisPrecision::F32),
                        format!(
                            "vec2<f32>({}, {})",
                            format_wgsl_f32(cos as f32),
                            format_wgsl_f32(sin as f32)
                        )
                    );
                }
            }
        }
    }

    #[test]
    fn axis_generators_use_host_luts_without_shader_trigonometry() {
        let sources = [
            wgsl_for(13, 143, 13),
            fused_wgsl_for(4096, FftDirection::Forward),
            fused_smooth_wgsl_for(3000, FftDirection::Inverse),
        ];
        for wgsl in sources {
            assert!(wgsl.contains("@binding(3) var<storage, read> axisTwiddles"));
            assert!(wgsl.contains("fn twiddle(index: u32)"));
            assert!(!wgsl.contains("cos("));
            assert!(!wgsl.contains("sin("));
            assert!(!wgsl.contains("cis("));
        }
    }

    #[test]
    fn native_f64_axis_generators_share_the_lut_path_and_use_f64_literals() {
        let sources = [
            wgsl_for_precision(13, 143, 13, AxisPrecision::F64),
            fused_wgsl_for_precision(2048, FftDirection::Forward, AxisPrecision::F64),
            fused_smooth_wgsl_for_precision(3000, FftDirection::Inverse, AxisPrecision::F64),
        ];
        for wgsl in sources {
            assert!(wgsl.contains("array<vec2<f64>>"));
            assert!(
                wgsl.contains("var<workgroup> scratch: array<vec2<f64>")
                    || wgsl.contains("fn c_mul(a: vec2<f64>")
            );
            assert!(wgsl.contains("lf"));
            assert!(!wgsl.contains("vec2<f32>"));
            assert!(!wgsl.contains("sin("));
            assert!(!wgsl.contains("cos("));
        }
    }

    #[test]
    fn df64_axis_generators_use_split_luts_and_pure_f32_arithmetic() {
        let sources = [
            wgsl_for_precision(13, 143, 13, AxisPrecision::Df64),
            fused_wgsl_for_precision(2048, FftDirection::Forward, AxisPrecision::Df64),
            fused_smooth_wgsl_for_precision(3000, FftDirection::Inverse, AxisPrecision::Df64),
        ];
        for wgsl in sources {
            assert!(wgsl.starts_with(crate::kernels::DF64_WGSL));
            assert!(wgsl.contains("array<vec4<f32>>"));
            assert!(wgsl.contains("fn c_mul(a: vec4<f32>"));
            assert!(wgsl.contains("df64_complex_mul(a, b)"));
            assert!(!wgsl.contains("vec2<f64>"));
            assert!(!wgsl.contains("enable f64"));
            assert!(!wgsl.contains("lf"));
            assert!(!wgsl.contains("sin("));
            assert!(!wgsl.contains("cos("));
        }
    }

    #[test]
    fn df64_radix_constants_and_scale_preserve_host_f64_low_words() {
        let angle = -std::f64::consts::TAU / 13.0;
        let (sin, cos) = angle.sin_cos();
        let re = crate::math::DoubleFloat::from_f64(cos);
        let im = crate::math::DoubleFloat::from_f64(sin);
        let expected_root = format!(
            "vec4<f32>({}, {}, {}, {})",
            format_wgsl_f32(re.hi),
            format_wgsl_f32(re.lo),
            format_wgsl_f32(im.hi),
            format_wgsl_f32(im.lo)
        );
        assert_eq!(
            radix_root_wgsl(13, 1, FftDirection::Forward, AxisPrecision::Df64),
            expected_root
        );
        assert_ne!(re.lo, 0.0);
        assert_ne!(im.lo, 0.0);

        let dims = [3];
        let scale = 1.0f64 / 3.0;
        let split_scale = crate::math::DoubleFloat::from_f64(scale);
        let wgsl = generate_stockham_radix_stage_wgsl(&StockhamStageWgslConfig {
            rank: 1,
            axis: 0,
            dims: &dims,
            axis_length: 3,
            stride_complex: 1,
            radix: 3,
            ns: 3,
            direction: FftDirection::Inverse,
            workgroup_size: WORKGROUP_SIZE,
            apply_scale: true,
            scale_factor: scale,
            precision: AxisPrecision::Df64,
        });
        assert!(wgsl.contains(&format!(
            "df64_complex_scale(out_0, Df64({}, {}))",
            format_wgsl_f32(split_scale.hi),
            format_wgsl_f32(split_scale.lo)
        )));
        assert_ne!(split_scale.lo, 0.0);
    }

    #[test]
    fn df64_reusable_scratch_stages_are_fragments_with_vec4_values() {
        let wgsl = generate_fused_scratch_fft_stages_wgsl(
            40,
            &[8, 5],
            FftDirection::Inverse,
            32,
            "sharedData",
            "lookupInverseRoot",
            AxisPrecision::Df64,
        );
        assert!(wgsl.contains("var stageOut_0_0: vec4<f32>"));
        assert!(wgsl.contains("vec4<f32>(0.0, 0.0, 0.0, 0.0)"));
        assert!(wgsl.contains("lookupInverseRoot(j_0 *"));
        assert!(!wgsl.contains("struct Df64"));
        assert!(!wgsl.contains("vec2<f32>"));
        assert!(!wgsl.contains("vec2<f64>"));
    }

    #[test]
    fn native_f64_radix_constants_and_scale_are_not_rounded_through_f32() {
        let root = radix_root_wgsl(13, 1, FftDirection::Forward, AxisPrecision::F64);
        let angle = -std::f64::consts::TAU / 13.0;
        let (sin, cos) = angle.sin_cos();
        assert_eq!(
            root,
            format!(
                "vec2<f64>({}, {})",
                format_wgsl_f64(cos),
                format_wgsl_f64(sin)
            )
        );
        assert_ne!(
            format_wgsl_f64(cos),
            format!("{}lf", format_wgsl_f32(cos as f32))
        );

        let dims = [3];
        let scale = 1.0f64 / 3.0;
        let wgsl = generate_stockham_radix_stage_wgsl(&StockhamStageWgslConfig {
            rank: 1,
            axis: 0,
            dims: &dims,
            axis_length: 3,
            stride_complex: 1,
            radix: 3,
            ns: 3,
            direction: FftDirection::Inverse,
            workgroup_size: WORKGROUP_SIZE,
            apply_scale: true,
            scale_factor: scale,
            precision: AxisPrecision::F64,
        });
        assert!(wgsl.contains(&format_wgsl_f64(scale)));
        assert!(!wgsl.contains(&format_wgsl_f32(scale as f32)));
    }

    #[test]
    fn generated_stockham_loads_each_exact_external_power_without_a_chain() {
        let wgsl = wgsl_for(11, 143, 143);
        for q in 1..11 {
            assert!(wgsl.contains(&format!("twiddle(j * {q}u)")));
        }
        assert!(wgsl.contains("let x_10: vec2<f32>"));
        assert!(wgsl.contains("let value_10: vec2<f32>"));
        assert!(!wgsl.contains("w = c_mul(w"));
        assert!(!wgsl.contains("p_in_block"));
    }

    #[test]
    fn generated_wgsl_can_be_recreated_from_stockham_cache_key() {
        let dims = [4, 3];
        let config = StockhamStageWgslConfig {
            rank: 2,
            axis: 1,
            dims: &dims,
            axis_length: 3,
            stride_complex: 4,
            radix: 3,
            ns: 3,
            direction: FftDirection::Inverse,
            workgroup_size: WORKGROUP_SIZE,
            apply_scale: true,
            scale_factor: 1.0 / 12.0,
            precision: AxisPrecision::F32,
        };
        let key = StockhamStageKey::new(
            config.rank,
            config.axis,
            config.dims,
            config.axis_length,
            config.stride_complex,
            config.radix,
            config.ns,
            config.direction,
            config.workgroup_size,
            config.apply_scale,
            config.scale_factor,
            config.precision,
        );

        assert_eq!(
            generate_stockham_radix_stage_wgsl_for_key(&key),
            generate_stockham_radix_stage_wgsl(&config)
        );
    }

    #[test]
    fn df64_axis_wgsl_can_be_recreated_from_typed_cache_keys() {
        let stockham_dims = [4, 3];
        let stockham = StockhamStageWgslConfig {
            rank: 2,
            axis: 1,
            dims: &stockham_dims,
            axis_length: 3,
            stride_complex: 4,
            radix: 3,
            ns: 3,
            direction: FftDirection::Inverse,
            workgroup_size: WORKGROUP_SIZE,
            apply_scale: true,
            scale_factor: 1.0 / 12.0,
            precision: AxisPrecision::Df64,
        };
        let stockham_key = StockhamStageKey::new(
            stockham.rank,
            stockham.axis,
            stockham.dims,
            stockham.axis_length,
            stockham.stride_complex,
            stockham.radix,
            stockham.ns,
            stockham.direction,
            stockham.workgroup_size,
            stockham.apply_scale,
            stockham.scale_factor,
            stockham.precision,
        );
        assert_eq!(
            generate_stockham_radix_stage_wgsl_for_key(&stockham_key),
            generate_stockham_radix_stage_wgsl(&stockham)
        );

        let fused_dims = [3, 256, 5];
        let fused = FusedPow2StageWgslConfig {
            rank: 3,
            axis: 1,
            dims: &fused_dims,
            axis_length: 256,
            stride_complex: 3,
            direction: FftDirection::Inverse,
            workgroup_size: FUSED_POW2_WORKGROUP_SIZE,
            apply_scale: true,
            scale_factor: 1.0 / 3840.0,
            precision: AxisPrecision::Df64,
        };
        let fused_key = FusedPow2StageKey::new(
            fused.rank,
            fused.axis,
            fused.dims,
            fused.axis_length,
            fused.stride_complex,
            fused.direction,
            fused.workgroup_size,
            fused.apply_scale,
            fused.scale_factor,
            fused.precision,
        );
        assert_eq!(
            generate_fused_pow2_stage_wgsl_for_key(&fused_key),
            generate_fused_pow2_stage_wgsl(&fused)
        );

        let smooth_dims = [4, 1001, 3];
        let smooth_factors = crate::runtime::factor_supported_length(1001).unwrap();
        let smooth = FusedSmoothStageWgslConfig {
            rank: 3,
            axis: 1,
            dims: &smooth_dims,
            axis_length: 1001,
            stride_complex: 4,
            factors: &smooth_factors,
            direction: FftDirection::Inverse,
            workgroup_size: FUSED_SMOOTH_WORKGROUP_SIZE,
            apply_scale: true,
            scale_factor: 1.0 / 12012.0,
            precision: AxisPrecision::Df64,
        };
        let smooth_key = FusedSmoothStageKey::new(
            smooth.rank,
            smooth.axis,
            smooth.dims,
            smooth.axis_length,
            smooth.stride_complex,
            smooth.factors,
            smooth.direction,
            smooth.workgroup_size,
            smooth.apply_scale,
            smooth.scale_factor,
            smooth.precision,
        );
        assert_eq!(
            generate_fused_smooth_stage_wgsl_for_key(&smooth_key),
            generate_fused_smooth_stage_wgsl(&smooth)
        );
    }

    #[test]
    fn fused_pow2_gate_matches_storage_and_workgroup_boundaries() {
        for length in [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048] {
            assert!(fused_pow2_supported_by_limits(
                length,
                AxisPrecision::F32,
                DEFAULT_FUSED_WORKGROUP_SIZE,
                16 * 1024,
                256,
                256
            ));
        }
        for (length, storage, invocations, size_x, expected) in [
            (4096, 16 * 1024, 256, 256, false),
            (4096, 48 * 1024, 256, 256, true),
            (1, 48 * 1024, 256, 256, false),
            (12, 48 * 1024, 256, 256, false),
            (8192, 48 * 1024, 256, 256, false),
            (2048, 16 * 1024, 255, 256, false),
            (2048, 16 * 1024, 256, 255, false),
        ] {
            assert_eq!(
                fused_pow2_supported_by_limits(
                    length,
                    AxisPrecision::F32,
                    DEFAULT_FUSED_WORKGROUP_SIZE,
                    storage,
                    invocations,
                    size_x,
                ),
                expected
            );
        }
        assert!(fused_pow2_supported_by_limits(
            2048,
            AxisPrecision::F64,
            DEFAULT_FUSED_WORKGROUP_SIZE,
            48 * 1024,
            256,
            256
        ));
        assert!(!fused_pow2_supported_by_limits(
            4096,
            AxisPrecision::F64,
            DEFAULT_FUSED_WORKGROUP_SIZE,
            48 * 1024,
            256,
            256
        ));
        assert!(fused_pow2_supported_by_limits(
            2048,
            AxisPrecision::Df64,
            DEFAULT_FUSED_WORKGROUP_SIZE,
            48 * 1024,
            256,
            256
        ));
        assert!(!fused_pow2_supported_by_limits(
            4096,
            AxisPrecision::Df64,
            DEFAULT_FUSED_WORKGROUP_SIZE,
            48 * 1024,
            256,
            256
        ));
        assert!(fused_pow2_supported_by_limits(
            2048,
            AxisPrecision::F32,
            128,
            16 * 1024,
            128,
            128
        ));
        assert!(!fused_pow2_supported_by_limits(
            2048,
            AxisPrecision::F32,
            128,
            16 * 1024,
            127,
            128
        ));
    }

    #[test]
    fn fused_smooth_gate_matches_factorization_and_compute_boundaries() {
        let factors_2048 = crate::runtime::factor_supported_length(2048).unwrap();
        let factors_13 = crate::runtime::factor_supported_length(13).unwrap();
        let factors_2187 = crate::runtime::factor_supported_length(2187).unwrap();
        let factors_3000 = crate::runtime::factor_supported_length(3000).unwrap();

        assert!(!fused_smooth_supported_by_limits(
            2048,
            &factors_2048,
            AxisPrecision::F32,
            DEFAULT_FUSED_WORKGROUP_SIZE,
            48 * 1024,
            256,
            256
        ));
        assert!(!fused_smooth_supported_by_limits(
            13,
            &factors_13,
            AxisPrecision::F32,
            DEFAULT_FUSED_WORKGROUP_SIZE,
            48 * 1024,
            256,
            256
        ));
        assert!(!fused_smooth_supported_by_limits(
            2187,
            &factors_2187,
            AxisPrecision::F32,
            DEFAULT_FUSED_WORKGROUP_SIZE,
            16 * 1024,
            256,
            256
        ));
        assert!(fused_smooth_supported_by_limits(
            2187,
            &factors_2187,
            AxisPrecision::F32,
            DEFAULT_FUSED_WORKGROUP_SIZE,
            48 * 1024,
            256,
            256
        ));
        assert!(fused_smooth_supported_by_limits(
            3000,
            &factors_3000,
            AxisPrecision::F32,
            DEFAULT_FUSED_WORKGROUP_SIZE,
            48 * 1024,
            256,
            256
        ));
        assert!(!fused_smooth_supported_by_limits(
            3000,
            &factors_3000,
            AxisPrecision::F32,
            DEFAULT_FUSED_WORKGROUP_SIZE,
            48 * 1024,
            255,
            256
        ));
        assert!(!fused_smooth_supported_by_limits(
            3000,
            &factors_3000,
            AxisPrecision::F32,
            DEFAULT_FUSED_WORKGROUP_SIZE,
            48 * 1024,
            256,
            255
        ));
        assert!(!fused_smooth_supported_by_limits(
            3000,
            &[8, 5, 5, 3],
            AxisPrecision::F32,
            DEFAULT_FUSED_WORKGROUP_SIZE,
            48 * 1024,
            256,
            256
        ));
        assert!(fused_smooth_supported_by_limits(
            3000,
            &factors_3000,
            AxisPrecision::F64,
            DEFAULT_FUSED_WORKGROUP_SIZE,
            48 * 1024,
            256,
            256
        ));
        assert!(!fused_smooth_supported_by_limits(
            3000,
            &factors_3000,
            AxisPrecision::F64,
            DEFAULT_FUSED_WORKGROUP_SIZE,
            47_999,
            256,
            256
        ));
        assert!(fused_smooth_supported_by_limits(
            3000,
            &factors_3000,
            AxisPrecision::Df64,
            DEFAULT_FUSED_WORKGROUP_SIZE,
            48 * 1024,
            256,
            256
        ));
        assert!(!fused_smooth_supported_by_limits(
            3000,
            &factors_3000,
            AxisPrecision::Df64,
            DEFAULT_FUSED_WORKGROUP_SIZE,
            47_999,
            256,
            256
        ));
    }

    fn multiline_wgsl_for(dims: &[usize], axis: usize, lines: usize) -> String {
        generate_fused_pow2_multiline_stage_wgsl(
            &FusedPow2StageWgslConfig {
                rank: dims.len(),
                axis,
                dims,
                axis_length: dims[axis],
                stride_complex: stride_for_axis(dims, axis),
                direction: FftDirection::Forward,
                workgroup_size: FUSED_POW2_WORKGROUP_SIZE,
                apply_scale: true,
                scale_factor: 0.25,
                precision: AxisPrecision::F32,
            },
            lines,
        )
    }

    #[test]
    fn multiline_fused_pow2_packs_lines_and_orders_loads_by_axis() {
        let contiguous = multiline_wgsl_for(&[64, 64], 0, 32);
        assert!(contiguous.contains("const LINES: u32 = 32u;"));
        assert!(contiguous.contains("var<workgroup> scratch: array<vec2<f32>, 2048>;"));
        assert!(contiguous.contains("let lineSlot: u32 = e / N;"));
        assert!(contiguous.contains("scratch[lineSlot * N + (reverseBits(p) >> (32u - LOG_N))]"));
        assert!(contiguous.contains("let lineCount: u32 = min(LINES, activeLines - groupLine);"));
        assert!(contiguous.contains("value = value * vec2<f32>(0.25, 0.25);"));
        crate::runtime::assert_workgroup_var_written_before_read(&contiguous, "scratch");

        // Strided axes load element-major so neighbouring invocations read
        // neighbouring lines.
        let strided = multiline_wgsl_for(&[64, 64], 1, 32);
        assert!(strided.contains("const STRIDE: u32 = 64u;"));
        assert!(strided.contains("let lineSlot: u32 = e % LINES;"));
        crate::runtime::assert_workgroup_var_written_before_read(&strided, "scratch");
    }

    #[test]
    fn multiline_fused_smooth_keeps_stages_in_workgroup_memory() {
        let dims = [60usize, 48];
        let factors = crate::runtime::factor_supported_length(48).unwrap();
        let wgsl = generate_fused_smooth_multiline_stage_wgsl(
            &FusedSmoothStageWgslConfig {
                rank: 2,
                axis: 1,
                dims: &dims,
                axis_length: 48,
                stride_complex: 60,
                factors: &factors,
                direction: FftDirection::Inverse,
                workgroup_size: FUSED_POW2_WORKGROUP_SIZE,
                apply_scale: false,
                scale_factor: 1.0,
                precision: AxisPrecision::F32,
            },
            16,
        );
        assert!(wgsl.contains("const LINES: u32 = 16u;"));
        assert!(wgsl.contains("var<workgroup> scratch: array<vec2<f32>, 768>;"));
        assert!(wgsl.contains("let lineSlot: u32 = e % LINES;"));
        assert!(wgsl.contains("if (lineSlot_0 < lineCount) {"));
        // The last stage writes workgroup memory; only the store pass writes dst.
        assert_eq!(wgsl.matches("dst[").count(), 1);
        crate::runtime::assert_workgroup_var_written_before_read(&wgsl, "scratch");
    }

    #[test]
    fn fused_lines_per_workgroup_fills_invocations_within_storage() {
        let storage = 48 * 1024;
        assert_eq!(
            fused_lines_per_workgroup(64, 1, AxisPrecision::F32, 1 << 20, storage),
            32
        );
        assert_eq!(
            fused_lines_per_workgroup(1024, 1, AxisPrecision::F32, 1 << 20, storage),
            2
        );
        assert_eq!(
            fused_lines_per_workgroup(4096, 1, AxisPrecision::F32, 1 << 20, storage),
            1
        );
        // Strided axes want at least 8 lines, capped by workgroup storage.
        assert_eq!(
            fused_lines_per_workgroup(512, 64, AxisPrecision::F32, 1 << 20, storage),
            8
        );
        assert_eq!(
            fused_lines_per_workgroup(1024, 64, AxisPrecision::F32, 1 << 20, storage),
            6
        );
        assert_eq!(
            fused_lines_per_workgroup(1024, 64, AxisPrecision::F64, 1 << 20, storage),
            3
        );
        // Small transforms keep enough workgroups to spread across the GPU.
        assert_eq!(
            fused_lines_per_workgroup(64, 1, AxisPrecision::F32, 64, storage),
            1
        );
        assert_eq!(
            fused_lines_per_workgroup(64, 64, AxisPrecision::F32, 4096, storage),
            16
        );

        let key = FusedPow2StageKey::new(
            1,
            0,
            &[64],
            64,
            1,
            FftDirection::Forward,
            256,
            false,
            1.0,
            AxisPrecision::F32,
        );
        assert!(!key.stable_key().contains(":lines="));
        let key = key.with_lines_per_workgroup(32);
        assert!(key.stable_key().ends_with(":lines=32"));
    }

    #[test]
    fn fused_pow2_generator_uses_one_shared_line_and_radix_schedule() {
        let wgsl = fused_wgsl_for(4096, FftDirection::Forward);
        assert!(wgsl.contains("@compute @workgroup_size(256, 1, 1)"));
        assert!(wgsl.contains("var<workgroup> scratch: array<vec2<f32>, 4096>;"));
        assert!(wgsl.contains("scratch[bit_reverse(p)] = src[srcIdx];"));
        crate::runtime::assert_workgroup_var_written_before_read(&wgsl, "scratch");
        assert!(wgsl.contains("dst[dstIdx] = value;"));
        assert!(wgsl.contains("@binding(3) var<storage, read> axisTwiddles"));
        assert!(wgsl.contains("let z8: vec2<f32> = twiddle(j *"));
        assert!(!wgsl.contains("cos("));
        assert!(!wgsl.contains("sin("));
        assert!(!wgsl.contains("cis("));
        assert_eq!(wgsl.matches("const RADIX: u32 = 8u;").count(), 4);
        assert_eq!(wgsl.matches("workgroupBarrier();").count(), 5);
        assert!(wgsl.contains("let wgFlat: u32 ="));
        assert!(wgsl.contains("if (wgFlat >= activeLines)"));
    }

    #[test]
    fn fused_pow2_generator_uses_radix_two_and_four_remainders() {
        let n1024 = fused_wgsl_for(1024, FftDirection::Inverse);
        assert_eq!(n1024.matches("const RADIX: u32 = 8u;").count(), 3);
        assert_eq!(n1024.matches("const RADIX: u32 = 2u;").count(), 1);
        assert!(n1024.contains("return vec2<f32>(value.x, -value.y);"));

        let n2048 = fused_wgsl_for(2048, FftDirection::Forward);
        assert_eq!(n2048.matches("const RADIX: u32 = 8u;").count(), 3);
        assert_eq!(n2048.matches("const RADIX: u32 = 4u;").count(), 1);
    }

    #[test]
    fn generated_wgsl_can_be_recreated_from_fused_cache_key() {
        let dims = [3, 256, 5];
        let config = FusedPow2StageWgslConfig {
            rank: 3,
            axis: 1,
            dims: &dims,
            axis_length: 256,
            stride_complex: 3,
            direction: FftDirection::Inverse,
            workgroup_size: FUSED_POW2_WORKGROUP_SIZE,
            apply_scale: true,
            scale_factor: 1.0 / 3840.0,
            precision: AxisPrecision::F32,
        };
        let key = FusedPow2StageKey::new(
            config.rank,
            config.axis,
            config.dims,
            config.axis_length,
            config.stride_complex,
            config.direction,
            config.workgroup_size,
            config.apply_scale,
            config.scale_factor,
            config.precision,
        );
        assert_eq!(
            generate_fused_pow2_stage_wgsl_for_key(&key),
            generate_fused_pow2_stage_wgsl(&config)
        );
    }

    #[test]
    fn reusable_in_place_smooth_fft_uses_named_scratch_and_twiddles_for_pow2_schedule() {
        let factors = [8, 8];
        let wgsl = generate_fused_scratch_fft_stages_wgsl(
            64,
            &factors,
            FftDirection::Inverse,
            FUSED_SMOOTH_WORKGROUP_SIZE,
            "sharedData",
            "lookupInverseRoot",
            AxisPrecision::F32,
        );

        assert_eq!(wgsl.matches("fused smooth radix-8 butterflies").count(), 2);
        assert_eq!(wgsl.matches("workgroupBarrier();").count(), 4);
        assert!(wgsl.contains("sharedData[base_0 + 7u * 8u]"));
        assert!(wgsl.contains("sharedData[block_0 * 64u + 7u * 8u + j_0]"));
        assert!(wgsl.contains("lookupInverseRoot(j_0 * 7u)"));
        assert!(!wgsl.contains("scratch["));
        assert!(!wgsl.contains("twiddle("));
    }

    #[test]
    fn reusable_in_place_smooth_fft_accepts_every_supported_radix() {
        for &radix in crate::runtime::SUPPORTED_RADICES {
            let factors = [radix, 2];
            let wgsl = generate_fused_scratch_fft_stages_wgsl(
                radix * 2,
                &factors,
                FftDirection::Forward,
                32,
                "values",
                "root",
                AxisPrecision::F32,
            );

            assert!(wgsl.contains(&format!("fused smooth radix-{radix} butterflies")));
            assert_eq!(wgsl.matches("workgroupBarrier();").count(), 4);
        }
    }

    #[test]
    fn fused_smooth_generator_uses_guarded_named_slots_and_uniform_barriers() {
        let wgsl = fused_smooth_wgsl_for(3000, FftDirection::Forward);
        assert!(wgsl.contains("@compute @workgroup_size(256, 1, 1)"));
        assert!(wgsl.contains("var<workgroup> scratch: array<vec2<f32>, 3000>;"));
        assert!(wgsl.contains("const LINE_SLOT_COUNT: u32 = 12u;"));
        assert!(wgsl.contains("scratch[p] = src[srcIdx];"));
        crate::runtime::assert_workgroup_var_written_before_read(&wgsl, "scratch");
        assert!(wgsl.contains("var stageOut_1_7: vec2<f32>"));
        assert!(wgsl.contains("if (unit_1 < 375u)"));
        assert!(wgsl.contains("scratch[block_1 * 8u + 7u * 1u + j_1] = stageOut_1_7;"));
        assert!(wgsl.contains("dst[dstIdx_3_2] = value_3_2;"));
        assert!(!wgsl.contains("bit_reverse"));
        assert_eq!(wgsl.matches("workgroupBarrier();").count(), 9);
    }

    #[test]
    fn fused_smooth_generator_covers_large_odd_radices() {
        let wgsl = fused_smooth_wgsl_for(1001, FftDirection::Inverse);
        assert!(wgsl.contains("fused smooth radix-13 butterflies"));
        assert!(wgsl.contains("fused smooth radix-11 butterflies"));
        assert!(wgsl.contains("fused smooth radix-7 butterflies"));
        assert!(wgsl.contains("return vec2<f32>(value.x, -value.y);"));
        assert!(!wgsl.contains("cos("));
        assert!(!wgsl.contains("sin("));
        assert!(!wgsl.contains("cis("));
        assert_eq!(wgsl.matches("workgroupBarrier();").count(), 5);
    }

    #[test]
    fn fused_smooth_stockham_math_matches_cpu_reference_in_both_directions() {
        for length in [3, 5, 7, 11, 13, 24, 143, 315] {
            let factors = crate::runtime::factor_supported_length(length).unwrap();
            let input = (0..length)
                .map(|index| {
                    let x = index as f32 + 1.0;
                    Complex32::new((x * 0.17).sin(), (x * 0.11).cos())
                })
                .collect::<Vec<_>>();
            for direction in [FftDirection::Forward, FftDirection::Inverse] {
                let config = match direction {
                    FftDirection::Forward => FftConfig::new(length),
                    FftDirection::Inverse => FftConfig::inverse(length),
                }
                .with_normalization(Normalization::None);
                let expected = reference_c2c_nd(&input, &config).unwrap();
                let actual = simulate_fused_smooth_1d(&input, &factors, direction);
                for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
                    let tolerance = 2.0e-2 + 2.0e-5 * expected.re.abs().max(expected.im.abs());
                    assert!(
                        actual.abs_diff(*expected) <= tolerance,
                        "N={length} direction={direction:?} index={index}: actual={actual:?} expected={expected:?} tolerance={tolerance}"
                    );
                }
            }
        }
    }

    #[test]
    fn generated_wgsl_can_be_recreated_from_fused_smooth_cache_key() {
        let dims = [4, 1001, 3];
        let factors = crate::runtime::factor_supported_length(1001).unwrap();
        let config = FusedSmoothStageWgslConfig {
            rank: 3,
            axis: 1,
            dims: &dims,
            axis_length: 1001,
            stride_complex: 4,
            factors: &factors,
            direction: FftDirection::Inverse,
            workgroup_size: FUSED_SMOOTH_WORKGROUP_SIZE,
            apply_scale: true,
            scale_factor: 1.0 / 12012.0,
            precision: AxisPrecision::F32,
        };
        let key = FusedSmoothStageKey::new(
            config.rank,
            config.axis,
            config.dims,
            config.axis_length,
            config.stride_complex,
            config.factors,
            config.direction,
            config.workgroup_size,
            config.apply_scale,
            config.scale_factor,
            config.precision,
        );
        assert_eq!(
            generate_fused_smooth_stage_wgsl_for_key(&key),
            generate_fused_smooth_stage_wgsl(&config)
        );
    }

    #[test]
    fn stockham_workspace_is_zero_for_one_stage_and_full_buffer_for_multi_stage() {
        let one_stage = AxisPlanConfig::from_c2c_config(&FftConfig::new(8));
        assert_eq!(one_stage.stockham_workspace_size_bytes().unwrap(), 0);

        let multi_stage = AxisPlanConfig::from_c2c_config(&FftConfig::new(16));
        assert_eq!(multi_stage.stockham_workspace_size_bytes().unwrap(), 16 * 8);

        let nd_batched = AxisPlanConfig::from_c2c_config(&FftConfig::new_nd([4, 3]).with_batch(2));
        assert_eq!(nd_batched.stockham_workspace_size_bytes().unwrap(), 24 * 8);

        let f64 =
            AxisPlanConfig::from_c2c_config(&FftConfig::new(16).with_precision(FftPrecision::F64));
        assert_eq!(f64.precision, AxisPrecision::F64);
        assert_eq!(f64.stockham_workspace_size_bytes().unwrap(), 16 * 16);
        assert_eq!(f64.precision.element_format(), ElementFormat::ComplexF64);
    }

    #[test]
    fn normalization_preserves_f32_bits_and_uses_native_f64_arithmetic() {
        const LEN: usize = 17_279_405;

        let f32 = AxisPlanConfig::from_c2c_config(&FftConfig::inverse(LEN));
        let f32_scale = f32.scale().unwrap();
        assert_eq!((f32_scale as f32).to_bits(), 0x3378_8f57);

        let f64 = AxisPlanConfig::from_c2c_config(
            &FftConfig::inverse(LEN).with_precision(FftPrecision::F64),
        );
        let f64_scale = f64.scale().unwrap();
        assert_eq!(f64_scale.to_bits(), (1.0f64 / LEN as f64).to_bits());
        assert_ne!(f64_scale, f32_scale);
    }
}
