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
    axis_plan_layout, with_device_pipeline_cache, ComputePipelineCacheKey, FusedPow2StageKey,
    FusedSmoothStageKey, PipelineLayoutCacheKey, ShaderCacheKey, SplitPass, StockhamStageKey,
};
use crate::runtime::recorder::CommandRecorder;
use crate::runtime::register_fft::{
    generate_register_fft_wgsl, register_schedule, register_schedule_for_lines,
};
use crate::runtime::twiddle::create_twiddle_lut_buffer_for_len_with_precision;
#[cfg(test)]
use crate::runtime::twiddle::twiddle_lut_f32;
use crate::runtime::window_scheduler::WindowScheduler;
use crate::tuning::DEFAULT_FUSED_WORKGROUP_SIZE;

#[cfg(test)]
const DEFAULT_WORKGROUP_SIZE: u32 = 64;
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
    pub(crate) long_axes: LongAxisRoute,
    /// Whether leading axes whose slabs fit one workgroup may run as one
    /// small-volume stage (`FftTuning::fuse_small_volumes`).
    pub(crate) small_volumes: bool,
}

/// How an axis too long for one fused workgroup runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LongAxisRoute {
    /// One Stockham pass per radix.
    Stockham,
    /// A register-resident fused kernel where the device runs one, else
    /// Stockham. Windowed large routes use this: split passes transform a
    /// transposed view of each line, which the window geometry does not
    /// describe.
    Registers,
    /// A register-resident fused kernel, else two fused passes, else
    /// Stockham.
    Fused,
}

impl LongAxisRoute {
    /// The route for `FftTuning::fuse_long_axes`.
    pub(crate) const fn new(fuse: bool) -> Self {
        if fuse {
            Self::Fused
        } else {
            Self::Stockham
        }
    }

    /// The route for `FftTuning::fuse_long_axes` in a windowed large route.
    pub(crate) const fn windowed(fuse: bool) -> Self {
        if fuse {
            Self::Registers
        } else {
            Self::Stockham
        }
    }

    const fn allows_registers(self) -> bool {
        !matches!(self, Self::Stockham)
    }

    const fn allows_split(self) -> bool {
        matches!(self, Self::Fused)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AxisStageKind {
    Stockham {
        radix: usize,
        ns: usize,
    },
    FusedPow2 {
        axis_length: usize,
    },
    FusedSmooth {
        axis_length: usize,
    },
    /// The leading axes of slabs of `elements` points in one kernel (see
    /// `runtime::small_volume`).
    SmallVolume {
        elements: usize,
    },
}

impl AxisStageKind {
    fn detail(self) -> String {
        match self {
            Self::Stockham { radix, ns } => format!("stockham.radix{radix}.ns{ns}"),
            Self::FusedPow2 { axis_length } => format!("fused_pow2.n{axis_length}"),
            Self::FusedSmooth { axis_length } => format!("fused_smooth.n{axis_length}"),
            Self::SmallVolume { elements } => format!("small_volume.e{elements}"),
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
    /// Set when every stage after the first runs in place on the output.
    in_place_bind_group_layout: Option<wgpu::BindGroupLayout>,
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
            long_axes: LongAxisRoute::new(config.tuning().fuse_long_axes()),
            small_volumes: config.tuning().fuse_small_volumes(),
        }
    }

    pub(crate) fn validate(&self) -> Result<()> {
        if self.shape.is_empty() || self.shape.contains(&0) {
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
        let mut lut_index = |length: usize| -> Result<usize> {
            if let Some(index) = twiddle_luts
                .iter()
                .position(|lut| lut.axis_length == length && lut.precision == config.precision)
            {
                return Ok(index);
            }
            let pool_key = (config.precision, length);
            let buffer = if let Some(buffer) = twiddle_lut_pool.buffers.get(&pool_key) {
                Arc::clone(buffer)
            } else {
                let buffer = Arc::new(create_twiddle_lut_buffer_for_len_with_precision(
                    device,
                    queue,
                    "wgpu_fft.axis_plan.twiddle_lut",
                    length,
                    config.precision.as_fft_precision(),
                )?);
                twiddle_lut_pool
                    .buffers
                    .insert(pool_key, Arc::clone(&buffer));
                buffer
            };
            twiddle_luts.push(AxisTwiddleLut {
                axis_length: length,
                precision: config.precision,
                buffer,
            });
            Ok(twiddle_luts.len() - 1)
        };

        // Leading axes whose slabs fit one workgroup run as one stage.
        let slab = (config.small_volumes
            && config.precision == AxisPrecision::F32
            && config.layout == AxisLayout::Interleaved)
            .then(|| {
                crate::runtime::small_volume::leading_slab(
                    &config.shape,
                    &config.axes,
                    config.direction,
                    config.fused_workgroup_size,
                    apply_any_scale,
                    scale,
                    &device.limits(),
                )
            })
            .flatten();
        let slab_axes = slab.as_ref().map_or(0, |(axes, _)| *axes);
        if let Some((_, key)) = slab {
            let elements = key.dims.iter().product::<usize>();
            let twiddle_lut_index = lut_index(key.twiddle_length)?;
            let pipeline_key = ComputePipelineCacheKey::small_volume(key.clone());
            let pipeline = with_device_pipeline_cache(device, |cache| {
                cache.get_compute_pipeline(
                    device,
                    &pipeline_key,
                    "wgpu_fft.axis_plan.small_volume.pipeline",
                    "wgpu_fft.axis_plan.small_volume.shader",
                    || crate::runtime::small_volume::generate_small_volume_wgsl_for_key(&key),
                )
            });
            stages.push(AxisStage {
                axis: 0,
                kind: AxisStageKind::SmallVolume { elements },
                stride_complex: 1,
                apply_scale: key.apply_scale,
                pipeline_key,
                twiddle_lut_index,
                workgroups_x: (total_complex / elements) as u32,
                pipeline,
            });
            for &axis in &config.axes[..slab_axes] {
                factors.push(crate::runtime::factor_supported_length(config.shape[axis])?);
            }
        }

        for (axis_index, &axis) in config.axes.iter().enumerate().skip(slab_axes) {
            let axis_len = config.shape[axis];
            let axis_factors = crate::runtime::factor_supported_length(axis_len)?;
            let stride_complex = stride_for_axis(&config.shape, axis);
            let final_axis = axis_index + 1 == config.axes.len();
            let twiddle_lut_index = lut_index(axis_len)?;

            let register_lines = register_lines_plan(
                axis_len,
                stride_complex,
                total_complex / axis_len,
                &config,
                &device.limits(),
            );
            if let Some((lines_per_workgroup, workgroup_size, schedule)) = register_lines {
                let apply_scale = apply_any_scale && final_axis;
                let total_lines = total_complex / axis_len;
                let shader_key = FusedPow2StageKey::new(
                    config.shape.len(),
                    axis,
                    &config.shape,
                    axis_len,
                    stride_complex,
                    config.direction,
                    workgroup_size,
                    apply_scale,
                    scale,
                    config.precision,
                )
                .with_lines_per_workgroup(lines_per_workgroup)
                .with_registers(schedule);
                let pipeline_key = ComputePipelineCacheKey::fused_pow2_stage(shader_key.clone());
                let shader_label =
                    format!("wgpu_fft.axis_plan.register_pow2.axis{axis}.n{axis_len}.shader");
                let pipeline_label =
                    format!("wgpu_fft.axis_plan.register_pow2.axis{axis}.n{axis_len}.pipeline");
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
            } else if fused_pow2_supported(
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
                &fused_smooth_factors(axis_len, &axis_factors),
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
                let smooth_key = |factors: &[usize]| {
                    FusedSmoothStageKey::new(
                        config.shape.len(),
                        axis,
                        &config.shape,
                        axis_len,
                        stride_complex,
                        factors,
                        config.direction,
                        config.fused_workgroup_size,
                        apply_scale,
                        scale,
                        config.precision,
                    )
                    .with_lines_per_workgroup(lines_per_workgroup)
                };
                let schedule = fused_smooth_factors(axis_len, &axis_factors);
                // A schedule whose first stage needs padded indices keeps
                // the multi-pass factors where the padding does not fit.
                let shader_key = if fused_smooth_pads_indices(&schedule, stride_complex != 1) {
                    let padded = smooth_key(&schedule).with_padded_indices();
                    if padded.supported_by_device_limits(&device.limits()) {
                        padded
                    } else {
                        smooth_key(&axis_factors)
                    }
                } else {
                    smooth_key(&schedule)
                };
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
            } else if let Some((workgroup_size, schedule)) = (stride_complex == 1
                && config.long_axes.allows_registers())
            .then(|| register_schedule(axis_len, config.precision, &device.limits()))
            .flatten()
            {
                // Contiguous lines too long for workgroup memory stay in
                // registers and run as one fused pass.
                let apply_scale = apply_any_scale && final_axis;
                let total_lines = total_complex / axis_len;
                let shader_key = FusedPow2StageKey::new(
                    config.shape.len(),
                    axis,
                    &config.shape,
                    axis_len,
                    stride_complex,
                    config.direction,
                    workgroup_size,
                    apply_scale,
                    scale,
                    config.precision,
                )
                .with_registers(schedule);
                let pipeline_key = ComputePipelineCacheKey::fused_pow2_stage(shader_key.clone());
                let shader_label =
                    format!("wgpu_fft.axis_plan.register_pow2.axis{axis}.n{axis_len}.shader");
                let pipeline_label =
                    format!("wgpu_fft.axis_plan.register_pow2.axis{axis}.n{axis_len}.pipeline");
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
                    workgroups_x: total_lines as u32,
                    pipeline,
                });
            } else if let Some((n1, n2)) = config
                .long_axes
                .allows_split()
                .then(|| {
                    long_axis_split(
                        axis_len,
                        config.precision,
                        config.fused_workgroup_size,
                        &device.limits(),
                    )
                })
                .flatten()
            {
                // View the axis as n = n2 + n2_len * n1 (see `SplitPass`): the
                // first pass transforms n1 and applies the split twiddle, the
                // second transforms n2 and stores transposed.
                let lower = &config.shape[..axis];
                let upper = &config.shape[axis + 1..];
                let split_dims = [lower, &[n2, n1][..], upper].concat();
                let out_dims = [lower, &[n1, n2][..], upper].concat();
                let row_stride_lines = lower.iter().product::<usize>();
                let passes = [
                    (
                        axis + 1,
                        n1,
                        SplitPass {
                            full_length: axis_len,
                            twiddle_scale: n2,
                            row_twiddle: Some((n2, row_stride_lines)),
                            output: None,
                        },
                        false,
                    ),
                    (
                        axis,
                        n2,
                        SplitPass {
                            full_length: axis_len,
                            twiddle_scale: n1,
                            row_twiddle: None,
                            output: Some((out_dims, axis + 1)),
                        },
                        apply_any_scale && final_axis,
                    ),
                ];
                for (pass_axis, pass_len, split_pass, apply_scale) in passes {
                    let pass_stride = stride_for_axis(&split_dims, pass_axis);
                    let total_lines = total_complex / pass_len;
                    let lines_per_workgroup = fused_lines_per_workgroup(
                        pass_len,
                        pass_stride,
                        config.precision,
                        total_lines,
                        u64::from(device.limits().max_compute_workgroup_storage_size),
                    );
                    let (kind, pipeline_key, shader_source): (_, _, Box<dyn FnOnce() -> String>) =
                        if pass_len.is_power_of_two() {
                            let key = FusedPow2StageKey::new(
                                split_dims.len(),
                                pass_axis,
                                &split_dims,
                                pass_len,
                                pass_stride,
                                config.direction,
                                config.fused_workgroup_size,
                                apply_scale,
                                scale,
                                config.precision,
                            )
                            .with_lines_per_workgroup(lines_per_workgroup)
                            .with_split_pass(split_pass);
                            (
                                AxisStageKind::FusedPow2 {
                                    axis_length: pass_len,
                                },
                                ComputePipelineCacheKey::fused_pow2_stage(key.clone()),
                                Box::new(move || generate_fused_pow2_stage_wgsl_for_key(&key)),
                            )
                        } else {
                            let key = FusedSmoothStageKey::new(
                                split_dims.len(),
                                pass_axis,
                                &split_dims,
                                pass_len,
                                pass_stride,
                                &crate::runtime::factor_supported_length(pass_len)?,
                                config.direction,
                                config.fused_workgroup_size,
                                apply_scale,
                                scale,
                                config.precision,
                            )
                            .with_lines_per_workgroup(lines_per_workgroup)
                            .with_split_pass(split_pass);
                            (
                                AxisStageKind::FusedSmooth {
                                    axis_length: pass_len,
                                },
                                ComputePipelineCacheKey::fused_smooth_stage(key.clone()),
                                Box::new(move || generate_fused_smooth_stage_wgsl_for_key(&key)),
                            )
                        };
                    let shader_label = format!(
                        "wgpu_fft.axis_plan.split.axis{axis}.n{axis_len}.pass{pass_len}.shader"
                    );
                    let pipeline_label = format!(
                        "wgpu_fft.axis_plan.split.axis{axis}.n{axis_len}.pass{pass_len}.pipeline"
                    );
                    let pipeline = with_device_pipeline_cache(device, |cache| {
                        cache.get_compute_pipeline(
                            device,
                            &pipeline_key,
                            &pipeline_label,
                            &shader_label,
                            shader_source,
                        )
                    });
                    stages.push(AxisStage {
                        axis,
                        kind,
                        stride_complex: pass_stride,
                        apply_scale,
                        pipeline_key,
                        twiddle_lut_index,
                        workgroups_x: (total_lines as u32).div_ceil(lines_per_workgroup),
                        pipeline,
                    });
                }
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
        // Fused kernels store each line where they read it, so after the
        // first stage they can transform the output in place: the working
        // set halves and no workspace is needed.
        let in_place = stages.len() > 1
            && stages
                .iter()
                .skip(1)
                .all(|stage| stage.pipeline_key.supports_in_place());
        let in_place_bind_group_layout = if in_place {
            for stage in stages.iter_mut().skip(1) {
                let pipeline_key = stage.pipeline_key.in_place();
                let label = format!(
                    "wgpu_fft.axis_plan.in_place.axis{}.{}",
                    stage.axis,
                    stage.kind.detail()
                );
                stage.pipeline = with_device_pipeline_cache(device, |cache| {
                    cache.get_compute_pipeline(
                        device,
                        &pipeline_key,
                        &format!("{label}.pipeline"),
                        &format!("{label}.shader"),
                        || fused_axis_stage_wgsl(&pipeline_key),
                    )
                });
                stage.pipeline_key = pipeline_key;
            }
            Some(with_device_pipeline_cache(device, |cache| {
                cache.get_bind_group_layout(device, axis_plan_layout(config.precision, true))
            }))
        } else {
            None
        };
        let workspace_size_bytes = if in_place {
            0
        } else {
            workspace_size_bytes_for_stage_count_with_precision(
                stages.len(),
                total_complex,
                config.precision,
            )
        };
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
            in_place_bind_group_layout,
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
        let in_place = self.in_place_bind_group_layout.is_some();
        let mut src_slot = BufferSlot::Input;
        let mut dst_slot = if in_place || self.stages.len() % 2 == 1 {
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
            let bind_group = match (&self.in_place_bind_group_layout, stage_index) {
                (Some(layout), 1..) => device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(&bind_group_label),
                    layout,
                    entries: &[
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
                }),
                _ => device.create_bind_group(&wgpu::BindGroupDescriptor {
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
                }),
            };

            {
                let pass = encoder.pass();
                pass.set_pipeline(&stage.pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                let (x, y, z) = split_workgroups(stage.workgroups_x, max_workgroups_per_dimension)?;
                pass.dispatch_workgroups(x, y, z);
            }

            if stage_index + 1 < self.stages.len() {
                src_slot = dst_slot;
                dst_slot = if in_place {
                    BufferSlot::Output
                } else {
                    next_stage_destination(src_slot)?
                };
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
    split: Option<&SplitPass>,
) -> String {
    debug_assert_eq!(config.rank, config.dims.len());
    debug_assert_eq!(config.axis_length, config.dims[config.axis]);
    debug_assert!(config.axis_length.is_power_of_two() && config.axis_length >= 2);
    debug_assert!(lines >= 1 && config.workgroup_size > 0);

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
    let store = multiline_store(split, config.stride_complex);
    let element_major = config.stride_complex != 1 || store.stride_out != 1;
    let store_split = multiline_split_wgsl(store.stride_out);

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
const STRIDE_OUT: u32 = {stride_out}u;
const WORKGROUP_SIZE: u32 = {workgroup_size}u;
const LINES: u32 = {lines}u;
const LINE_STRIDE: u32 = {line_stride}u;

var<workgroup> scratch: array<vec2<f32>, {scratch_len}>;

{line_base_fn}

{line_base_out_fn}

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

{load_block}  workgroupBarrier();

{radix_stages}
  for (var e: u32 = lid.x; e < LINES * N; e = e + WORKGROUP_SIZE) {{
    {store_split}
    if (lineSlot < lineCount) {{
      var value: vec2<f32> = scratch[lineSlot * LINE_STRIDE + p];
{row_twiddle}{maybe_scale}      let dstIdx: u32 = line_base_out(lineStart + lineSlot) + p * STRIDE_OUT - params.elementBase;
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
            scratch_len = multiline_line_stride(config.axis_length, lines, element_major) * lines,
            line_stride = multiline_line_stride(config.axis_length, lines, element_major),
            twiddle_lookup_wgsl =
                multiline_twiddle_lookup_wgsl(config.direction, config.precision, split),
            line_base_fn = line_base_fn,
            entry_params = crate::runtime::dispatch::WGSL_FLAT_ENTRY_PARAMS,
            flat_workgroup_index = crate::runtime::dispatch::WGSL_FLAT_WORKGROUP_INDEX,
            radix_stages = radix_stages,
            maybe_scale = maybe_scale,
            store_split = store_split,
            stride_out = store.stride_out,
            line_base_out_fn = store.line_base_out_fn,
            row_twiddle = store.row_twiddle,
            load_block = multiline_load_wgsl(
                config.axis_length,
                lines,
                config.workgroup_size as usize,
                config.stride_complex != 1,
                "(reverseBits(P) >> (32u - LOG_N))",
            ),
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
      let base: u32 = lineSlot * LINE_STRIDE + block * (RADIX * PREVIOUS) + j;
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
    let source = if let Some(registers) = &key.registers {
        generate_register_fft_wgsl(&config, registers, key.lines_per_workgroup as usize)
    } else if key.lines_per_workgroup > 1 || key.split_pass.is_some() {
        generate_fused_pow2_multiline_stage_wgsl(
            &config,
            key.lines_per_workgroup as usize,
            key.split_pass.as_ref(),
        )
    } else {
        generate_fused_pow2_stage_wgsl(&config)
    };
    if key.in_place {
        in_place_wgsl(&source)
    } else {
        source
    }
}

/// WGSL of a fused axis kernel from its pipeline key.
fn fused_axis_stage_wgsl(key: &ComputePipelineCacheKey) -> String {
    match &key.shader {
        ShaderCacheKey::FusedPow2Stage(shader) => generate_fused_pow2_stage_wgsl_for_key(shader),
        ShaderCacheKey::FusedSmoothStage(shader) => {
            generate_fused_smooth_stage_wgsl_for_key(shader)
        }
        _ => unreachable!("only fused axis kernels run in place"),
    }
}

/// Rebinds an out-of-place axis kernel to read its lines from `dst`: the
/// `src` binding goes away and every `src` load reads `dst`.
fn in_place_wgsl(source: &str) -> String {
    let source = source
        .lines()
        .filter(|line| !line.starts_with("@group(0) @binding(0) var<storage, read> src:"))
        .collect::<Vec<_>>()
        .join("\n");
    debug_assert!(!source.contains("var<storage, read> src"));
    source.replace("src[", "dst[") + "\n"
}

/// Splits an axis too long for one fused workgroup into `n1 * n2`, where each
/// factor fits a fused kernel, preferring the most balanced split. Returns
/// `None` when no such split exists.
fn long_axis_split(
    axis_length: usize,
    precision: AxisPrecision,
    fused_workgroup_size: u32,
    limits: &wgpu::Limits,
) -> Option<(usize, usize)> {
    let fused = |len: usize| {
        fused_pow2_supported(len, precision, fused_workgroup_size, limits)
            || crate::runtime::factor_supported_length(len).is_ok_and(|factors| {
                fused_smooth_supported(len, &factors, precision, fused_workgroup_size, limits)
            })
    };
    let mut split = None;
    let mut n1 = 2;
    while n1 * n1 <= axis_length {
        if axis_length.is_multiple_of(n1) && fused(n1) && fused(axis_length / n1) {
            split = Some((n1, axis_length / n1));
        }
        n1 += 1;
    }
    split
}

/// Longest line [`register_lines_plan`] runs in registers on any axis.
const MAX_SHORT_REGISTER_LENGTH: usize = 512;
/// Shortest contiguous line run in registers; 32-point lines measured faster
/// in the workgroup-memory kernel.
const MIN_CONTIGUOUS_REGISTER_LENGTH: usize = 64;
/// Shortest strided line run in registers.
const MIN_STRIDED_REGISTER_LENGTH: usize = 16;
/// Invocations a workgroup of short register lines aims for.
const SHORT_REGISTER_INVOCATIONS: usize = 128;
/// Fewest lines a workgroup of short strided lines takes, so each of its
/// loads spans at least 128 contiguous bytes.
const MIN_SHORT_STRIDED_LINES: usize = 16;
/// Workgroups short register lines keep when the transform is small, so its
/// lines spread over the GPU.
const MIN_REGISTER_WORKGROUPS: usize = 64;
/// Shortest contiguous line longer than [`MAX_SHORT_REGISTER_LENGTH`] run in
/// registers although workgroup memory holds it.
const MIN_LONG_CONTIGUOUS_REGISTER_LENGTH: usize = 1024;
/// Invocations a workgroup of long contiguous register lines aims for.
const LONG_CONTIGUOUS_REGISTER_INVOCATIONS: usize = 256;

/// Lines per workgroup, workgroup size, and schedule of a register-resident
/// kernel for a power-of-two axis, when it is the faster kernel.
///
/// A register kernel loads its lines straight into registers and passes
/// them through workgroup memory once per radix-16 stage; the
/// workgroup-memory kernel scatters them into workgroup memory and passes
/// through it once per radix-8 stage. Registers win for:
///
/// - lines of at most [`MAX_SHORT_REGISTER_LENGTH`] points, contiguous from
///   [`MIN_CONTIGUOUS_REGISTER_LENGTH`] and strided from
///   [`MIN_STRIDED_REGISTER_LENGTH`];
/// - longer contiguous lines, from [`MIN_LONG_CONTIGUOUS_REGISTER_LENGTH`];
/// - longer strided lines when registers interleave more of them than
///   workgroup memory could, up to eight, so neighbouring invocations load
///   neighbouring lines.
///
/// A fused workgroup size other than the default keeps the first two in the
/// workgroup-memory kernel, which honours it; the third follows
/// `fuse_long_axes`.
fn register_lines_plan(
    axis_length: usize,
    stride_complex: usize,
    total_lines: usize,
    config: &AxisPlanConfig,
    limits: &wgpu::Limits,
) -> Option<(u32, u32, crate::runtime::pipeline_cache::RegisterSchedule)> {
    let default_fused = config.fused_workgroup_size == DEFAULT_FUSED_WORKGROUP_SIZE;
    if axis_length <= MAX_SHORT_REGISTER_LENGTH {
        if !default_fused {
            return None;
        }
        return short_register_lines_plan(axis_length, stride_complex, total_lines, config, limits);
    }
    let line_invocations =
        axis_length / crate::runtime::register_fft::REGISTER_VALUES_PER_INVOCATION;
    let fused = fused_pow2_supported(
        axis_length,
        config.precision,
        config.fused_workgroup_size,
        limits,
    );
    if stride_complex == 1 {
        if !default_fused || axis_length < MIN_LONG_CONTIGUOUS_REGISTER_LENGTH || !fused {
            return None;
        }
        // No more lines than there are, rounded up to a power of two.
        let lines = (LONG_CONTIGUOUS_REGISTER_INVOCATIONS / line_invocations)
            .max(1)
            .min(total_lines.max(1).next_power_of_two());
        return largest_register_schedule(axis_length, lines, 1, config.precision, limits);
    }
    if !config.long_axes.allows_registers() {
        return None;
    }
    const MAX_LINES: usize = 8;
    let shared_lines = if fused {
        fused_lines_per_workgroup(
            axis_length,
            stride_complex,
            config.precision,
            total_lines,
            u64::from(limits.max_compute_workgroup_storage_size),
        ) as usize
    } else {
        0
    };
    largest_register_schedule(
        axis_length,
        MAX_LINES,
        (shared_lines + 1).max(2),
        config.precision,
        limits,
    )
}

/// [`register_lines_plan`] for lines of at most [`MAX_SHORT_REGISTER_LENGTH`]
/// points: about [`SHORT_REGISTER_INVOCATIONS`] invocations per workgroup,
/// strided axes taking at least [`MIN_SHORT_STRIDED_LINES`] lines, and whole
/// lines in the exchange buffer, whose rounds would add barriers.
fn short_register_lines_plan(
    axis_length: usize,
    stride_complex: usize,
    total_lines: usize,
    config: &AxisPlanConfig,
    limits: &wgpu::Limits,
) -> Option<(u32, u32, crate::runtime::pipeline_cache::RegisterSchedule)> {
    let min_length = if stride_complex == 1 {
        MIN_CONTIGUOUS_REGISTER_LENGTH
    } else {
        MIN_STRIDED_REGISTER_LENGTH
    };
    if axis_length < min_length {
        return None;
    }
    let line_invocations =
        axis_length / crate::runtime::register_fft::REGISTER_VALUES_PER_INVOCATION;
    let mut lines = (SHORT_REGISTER_INVOCATIONS / line_invocations).max(1);
    if stride_complex != 1 {
        lines = lines.max(MIN_SHORT_STRIDED_LINES);
    }
    lines = lines.min(total_lines.max(1).next_power_of_two());
    while lines > 1 && total_lines / lines < MIN_REGISTER_WORKGROUPS {
        lines /= 2;
    }
    let line_bytes = axis_length * config.precision.complex_size_bytes() as usize;
    while lines > 1 && lines * line_bytes > limits.max_compute_workgroup_storage_size as usize {
        lines /= 2;
    }
    largest_register_schedule(axis_length, lines, 1, config.precision, limits)
}

/// The register schedule for the most lines per workgroup from `lines` down
/// to `min_lines`, halving, that the device can run.
fn largest_register_schedule(
    axis_length: usize,
    mut lines: usize,
    min_lines: usize,
    precision: AxisPrecision,
    limits: &wgpu::Limits,
) -> Option<(u32, u32, crate::runtime::pipeline_cache::RegisterSchedule)> {
    while lines >= min_lines.max(1) {
        if let Some((workgroup_size, schedule)) =
            register_schedule_for_lines(axis_length, lines, precision, limits)
        {
            return Some((lines as u32, workgroup_size, schedule));
        }
        lines /= 2;
    }
    None
}

/// Radices of fused smooth kernels, largest first. A composite radix runs as
/// one butterfly in registers (see [`emit_small_dft_wgsl`]): more work per
/// invocation, but fewer stages, barriers, and trips through workgroup
/// memory than its prime factors as separate stages.
const FUSED_SMOOTH_RADICES: &[usize] = &[16, 15, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2];

/// Primes above 13 that fused Rader convolutions also run as radices, each
/// as one straight-line butterfly (see [`emit_small_dft_wgsl`]) of about
/// `p^2` real multiplications, so a cyclic convolution of `N - 1` points can
/// replace a zero-padded one about twice as long.
pub(crate) const FUSED_PRIME_RADICES: &[usize] = &[17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61];

/// Radix schedule of a fused smooth kernel for `axis_length`, whose
/// multi-pass factorization is `factors`: the fewest stages; among those,
/// the most radix-16 stages, whose butterflies need the fewest
/// multiplications, then the most balanced radices (the largest smallest
/// radix); largest radix first, since the first stage needs no twiddles.
/// Keeps `factors` unless that saves stages, and when it would leave a
/// single stage. A grid search over schedules favoured these;
/// schedules with as many stages as `factors` measured no better.
pub(crate) fn fused_smooth_factors(axis_length: usize, factors: &[usize]) -> Vec<usize> {
    fn rank(schedule: &[usize]) -> (usize, std::cmp::Reverse<usize>, std::cmp::Reverse<usize>) {
        let sixteens = schedule.iter().filter(|&&radix| radix == 16).count();
        let smallest = schedule.iter().copied().min().unwrap_or(0);
        (
            schedule.len(),
            std::cmp::Reverse(sixteens),
            std::cmp::Reverse(smallest),
        )
    }
    // Non-increasing radix sequences, each schedule once.
    fn search(rest: usize, from: usize, current: &mut Vec<usize>, best: &mut Option<Vec<usize>>) {
        if rest == 1 {
            if best.as_ref().is_none_or(|best| {
                rank(current) < rank(best) || (rank(current) == rank(best) && *current > *best)
            }) {
                *best = Some(current.clone());
            }
            return;
        }
        if best
            .as_ref()
            .is_some_and(|best| current.len() >= best.len())
        {
            return;
        }
        for (index, &radix) in FUSED_SMOOTH_RADICES.iter().enumerate().skip(from) {
            if rest.is_multiple_of(radix) {
                current.push(radix);
                search(rest / radix, index, current, best);
                current.pop();
            }
        }
    }
    let mut best = None;
    search(axis_length, 0, &mut Vec::new(), &mut best);
    match best {
        Some(schedule) if schedule.len() >= 2 && schedule.len() < factors.len() => schedule,
        _ => factors.to_vec(),
    }
}

/// Elements of workgroup memory a padded fused smooth kernel needs for
/// `elements` unpadded ones (see [`pad_workgroup_indices`]).
pub(crate) const fn padded_workgroup_len(elements: usize) -> usize {
    elements + elements.saturating_sub(1) / 16
}

/// Whether a fused smooth kernel pads its workgroup-memory indices: when it
/// keeps its lines line-major and starts with a radix that is a multiple of
/// 8. Its first stage writes each invocation's outputs at a stride of that
/// radix, which puts a warp's writes into a few banks; one element of
/// padding per 16 spreads them. Kernels that interleave lines, or start with
/// a radix whose stride spreads anyway, measured no gain.
pub(crate) fn fused_smooth_pads_indices(factors: &[usize], element_major: bool) -> bool {
    !element_major && factors.first().is_some_and(|radix| radix.is_multiple_of(8))
}

/// Rewrites every `scratch[index]` of a fused smooth kernel to
/// `scratch[padded(index)]`, with `padded(i) = i + i / 16`, and enlarges the
/// declaration to match.
pub(crate) fn pad_workgroup_indices(source: &str) -> String {
    let mut out = String::with_capacity(source.len() + 256);
    let mut rest = source;
    while let Some(at) = rest.find("scratch[") {
        out.push_str(&rest[..at]);
        let after = &rest[at + "scratch[".len()..];
        let mut depth = 1usize;
        let end = after
            .char_indices()
            .find_map(|(position, character)| {
                match character {
                    '[' => depth += 1,
                    ']' => {
                        depth -= 1;
                        if depth == 0 {
                            return Some(position);
                        }
                    }
                    _ => {}
                }
                None
            })
            .expect("balanced workgroup-memory index");
        out.push_str(&format!("scratch[padded({})]", &after[..end]));
        rest = &after[end + 1..];
    }
    out.push_str(rest);
    let declaration = "var<workgroup> scratch: array<";
    let at = out.find(declaration).expect("a scratch declaration");
    let tail = &out[at + declaration.len()..];
    let comma = tail.find(", ").expect("a scratch element type");
    let close = tail.find(">;").expect("a scratch length");
    let length: usize = tail[comma + 2..close]
        .trim()
        .parse()
        .expect("a literal scratch length");
    let old = format!("{declaration}{}", &tail[..close + 2]);
    let new = format!(
        "fn padded(index: u32) -> u32 {{\n  return index + (index >> 4u);\n}}\n\n{declaration}{}, {}>;",
        &tail[..comma],
        padded_workgroup_len(length)
    );
    out.replacen(&old, &new, 1)
}

/// Lines per workgroup for a fused kernel on an axis of `axis_length`.
///
/// Targets about 2048 elements per workgroup so radix stages keep 256
/// invocations busy; strided axes take at least 8 adjacent lines so their
/// loads coalesce. Workgroup storage caps the result, and small transforms
/// keep at least 256 workgroups so their lines spread across the GPU instead
/// of queueing on a few compute units. Strided axes round down to a power of
/// two: the row a workgroup reads then fills whole 32-byte sectors, and each
/// invocation keeps one line for all its loads.
pub(crate) fn fused_lines_per_workgroup(
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
    // Padded as for strided axes; contiguous ones never approach this cap.
    let line_bytes = (axis_length + 1) * precision.complex_size_bytes() as usize;
    let max_by_storage = (max_workgroup_storage_bytes as usize / line_bytes).max(1);
    let max_by_fill = (total_lines / MIN_WORKGROUPS).max(1);
    let lines = lines.min(max_by_storage).min(max_by_fill);
    if stride_complex > 1 {
        1 << lines.ilog2()
    } else {
        lines as u32
    }
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
        .all(|radix| FUSED_SMOOTH_RADICES.contains(radix) || FUSED_PRIME_RADICES.contains(radix)));
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
            &|index| format!("{scratch_name}[{index}]"),
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

/// First radix stage of a multi-line fused smooth kernel, reading its
/// inputs straight from global memory instead of from a copy of the lines in
/// workgroup memory. It needs no twiddles, and nothing in workgroup memory is
/// read before it writes there, so it writes its outputs as soon as they are
/// ready and ends with one barrier. Strided (`element_major`) lines assign
/// neighbouring invocations to neighbouring lines, so their reads coalesce.
fn generate_fused_smooth_first_stage_multiline_wgsl(
    axis_length: usize,
    radix: usize,
    direction: FftDirection,
    workgroup_size: u32,
    lines: usize,
    element_major: bool,
    precision: AxisPrecision,
) -> String {
    debug_assert_eq!(axis_length % radix, 0);
    let unit_count = axis_length / radix;
    let slot_count = (unit_count * lines).div_ceil(workgroup_size as usize);
    let mut slots = String::new();
    for slot in 0..slot_count {
        let split = if element_major {
            format!(
                "    let lineSlot_{slot}: u32 = slotUnit_{slot} % LINES;\n    let unit_{slot}: u32 = slotUnit_{slot} / LINES;\n"
            )
        } else {
            format!(
                "    let lineSlot_{slot}: u32 = slotUnit_{slot} / {unit_count}u;\n    let unit_{slot}: u32 = slotUnit_{slot} - lineSlot_{slot} * {unit_count}u;\n"
            )
        };
        let butterfly = generate_fused_smooth_butterfly_math_wgsl(
            radix,
            1,
            unit_count,
            unit_count,
            direction,
            slot,
            &|index| format!("src[first_{slot} + ({index}) * STRIDE]"),
            "twiddle",
            precision,
        );
        let mut outputs = String::new();
        let mut writes = String::new();
        for output in 0..radix {
            outputs.push_str(&format!("      var stageOut_{slot}_{output}: vec2<f32>;\n"));
            writes.push_str(&format!(
                "      scratch[lineSlot_{slot} * LINE_STRIDE + unit_{slot} * {radix}u + {output}u] = stageOut_{slot}_{output};\n"
            ));
        }
        slots.push_str(&format!(
            "    let slotUnit_{slot}: u32 = lid.x + {slot}u * WORKGROUP_SIZE;\n{split}    if (lineSlot_{slot} < lineCount && unit_{slot} < {unit_count}u) {{\n      let first_{slot}: u32 = line_base(lineStart + lineSlot_{slot}) - params.elementBase;\n      let base_{slot}: u32 = unit_{slot};\n{outputs}{butterfly}{writes}    }}\n"
        ));
    }
    specialize_complex_wgsl(
        format!(
            "  {{ // fused smooth radix-{radix} butterflies from global memory, {lines} lines\n{slots}    workgroupBarrier();\n  }}\n"
        ),
        precision,
    )
}

#[allow(clippy::too_many_arguments)]
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
            radix,
            ns_div_r,
            n_div_r,
            n_div_ns,
            direction,
            slot,
            &|index| format!("scratch[{index}]"),
            "twiddle",
            precision,
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
pub(crate) fn generate_fused_smooth_butterfly_math_wgsl(
    radix: usize,
    ns_div_r: usize,
    n_div_r: usize,
    n_div_ns: usize,
    direction: FftDirection,
    slot: usize,
    load: &dyn Fn(&str) -> String,
    twiddle_fn_name: &str,
    precision: AxisPrecision,
) -> String {
    // Workgroup-scratch form of the same unit-centric Stockham factorization
    // used by the multi-pass generator; `load` renders the read of a line
    // position.
    let mut shader = String::new();
    shader.push_str(&format!(
        "      let x_{slot}_0: vec2<f32> = {};\n",
        load(&format!("base_{slot}"))
    ));
    let twiddles = if ns_div_r == 1 {
        Vec::new()
    } else {
        emit_split_twiddles(&mut shader, &format!("tw_{slot}"), radix, "      ", &|q| {
            format!("{twiddle_fn_name}(j_{slot} * {}u)", q * n_div_ns)
        })
    };
    for q in 1..radix {
        let input = load(&format!("base_{slot} + {q}u * {n_div_r}u"));
        let value = match twiddles.get(q) {
            Some(twiddle) => format!("c_mul({twiddle}, {input})"),
            None => input,
        };
        shader.push_str(&format!("      let x_{slot}_{q}: vec2<f32> = {value};\n"));
    }

    let inputs = (0..radix)
        .map(|q| format!("x_{slot}_{q}"))
        .collect::<Vec<_>>();
    let outputs = emit_small_dft_wgsl(
        &mut shader,
        &format!("dft_{slot}"),
        &inputs,
        direction,
        precision,
    );
    for (output, value) in outputs.iter().enumerate() {
        shader.push_str(&format!("      stageOut_{slot}_{output} = {value};\n"));
    }
    shader
}

/// Binds the stage twiddles `W^(j q)` for `q` in `1..radix`, where
/// `twiddle(q)` renders the table read of `W^(j q)`, and returns their names
/// indexed by `q` (index 0 unused). With `B` the smallest integer whose
/// square reaches `radix`, `q = B a + b` and `W^(j q) = W^(j B a) W^(j b)`:
/// about `2 sqrt(radix)` reads instead of `radix - 1` (6 instead of 15 for
/// radix 16), each other twiddle one product of two exact table values.
/// The table reads are scattered across the table, so they cost more than
/// the products.
pub(crate) fn emit_split_twiddles(
    body: &mut String,
    prefix: &str,
    radix: usize,
    indent: &str,
    twiddle: &dyn Fn(usize) -> String,
) -> Vec<String> {
    let mut names = vec![String::new(); radix];
    if radix <= 3 {
        for (q, name) in names.iter_mut().enumerate().skip(1) {
            *name = format!("{prefix}_{q}");
            body.push_str(&format!(
                "{indent}let {name}: vec2<f32> = {};\n",
                twiddle(q)
            ));
        }
        return names;
    }
    let base = (2..).find(|base| base * base >= radix).unwrap_or(radix);
    for (b, name) in names.iter_mut().enumerate().take(base).skip(1) {
        *name = format!("{prefix}_{b}");
        body.push_str(&format!(
            "{indent}let {name}: vec2<f32> = {};\n",
            twiddle(b)
        ));
    }
    for a in 1..=(radix - 1) / base {
        let q = a * base;
        names[q] = format!("{prefix}_{q}");
        body.push_str(&format!(
            "{indent}let {}: vec2<f32> = {};\n",
            names[q],
            twiddle(q)
        ));
    }
    for q in 1..radix {
        let (a, b) = (q / base, q % base);
        if a > 0 && b > 0 {
            names[q] = format!("{prefix}_{q}");
            body.push_str(&format!(
                "{indent}let {}: vec2<f32> = c_mul({}, {});\n",
                names[q],
                names[a * base],
                names[b]
            ));
        }
    }
    names
}

/// Appends statements computing `DFT_R` of the complex WGSL values `inputs`
/// and returns the names of its outputs in order. Radix 2 and 4 run as
/// butterflies whose rotations by `-i` are component swaps. Odd primes pair
/// `x[k]` with `x[R - k]`: output `m` and `R - m` share the cosine sum of the
/// pair sums and the sine sum of the pair differences, so the DFT takes
/// `(R - 1)^2` real multiplications instead of `(R - 1)^2` complex ones.
/// Composite radices split as `R = R1 * R2` with `R2` their smallest prime
/// factor: `R2` DFTs of `R1` points, twiddles `W_R^(n2 * k1)`, then `R1` DFTs
/// of `R2` points.
fn emit_small_dft_wgsl(
    body: &mut String,
    prefix: &str,
    inputs: &[String],
    direction: FftDirection,
    precision: AxisPrecision,
) -> Vec<String> {
    let bind = |body: &mut String, name: String, value: String| {
        body.push_str(&format!("      let {name}: vec2<f32> = {value};\n"));
        name
    };
    let radix = inputs.len();
    match radix {
        1 => inputs.to_vec(),
        2 => vec![
            bind(
                body,
                format!("{prefix}_0"),
                format!("c_add({}, {})", inputs[0], inputs[1]),
            ),
            bind(
                body,
                format!("{prefix}_1"),
                format!("c_sub({}, {})", inputs[0], inputs[1]),
            ),
        ],
        4 => {
            let (x0, x1, x2, x3) = (&inputs[0], &inputs[1], &inputs[2], &inputs[3]);
            let s0 = bind(body, format!("{prefix}_s0"), format!("c_add({x0}, {x2})"));
            let d0 = bind(body, format!("{prefix}_d0"), format!("c_sub({x0}, {x2})"));
            let s1 = bind(body, format!("{prefix}_s1"), format!("c_add({x1}, {x3})"));
            let d1 = bind(body, format!("{prefix}_d1"), format!("c_sub({x1}, {x3})"));
            let r1 = bind(
                body,
                format!("{prefix}_r1"),
                quarter_turn_wgsl(&d1, direction, precision),
            );
            vec![
                bind(body, format!("{prefix}_0"), format!("c_add({s0}, {s1})")),
                bind(body, format!("{prefix}_1"), format!("c_add({d0}, {r1})")),
                bind(body, format!("{prefix}_2"), format!("c_sub({s0}, {s1})")),
                bind(body, format!("{prefix}_3"), format!("c_sub({d0}, {r1})")),
            ]
        }
        _ if smallest_prime_factor(radix) < radix => {
            // n = R2 * n1 + n2 and k = k1 + R1 * k2: a DFT_R1 over n1 for
            // each n2, twiddled by W_R^(n2 * k1), then a DFT_R2 over n2.
            let r2 = smallest_prime_factor(radix);
            let r1 = radix / r2;
            let columns = (0..r2)
                .map(|n2| {
                    let column = (0..r1)
                        .map(|n1| inputs[r2 * n1 + n2].clone())
                        .collect::<Vec<_>>();
                    emit_small_dft_wgsl(
                        body,
                        &format!("{prefix}c{n2}"),
                        &column,
                        direction,
                        precision,
                    )
                })
                .collect::<Vec<_>>();
            let mut outputs = vec![String::new(); radix];
            for k1 in 0..r1 {
                let row = (0..r2)
                    .map(|n2| {
                        let exponent = (n2 * k1) % radix;
                        let value = &columns[n2][k1];
                        if exponent == 0 {
                            value.clone()
                        } else if (4 * exponent).is_multiple_of(radix) {
                            bind(
                                body,
                                format!("{prefix}t{n2}_{k1}"),
                                rotate_quarters_wgsl(
                                    value,
                                    4 * exponent / radix,
                                    direction,
                                    precision,
                                ),
                            )
                        } else {
                            bind(
                                body,
                                format!("{prefix}t{n2}_{k1}"),
                                format!(
                                    "c_mul({value}, {})",
                                    radix_root_wgsl(radix, exponent, direction, precision)
                                ),
                            )
                        }
                    })
                    .collect::<Vec<_>>();
                let row_outputs = emit_small_dft_wgsl(
                    body,
                    &format!("{prefix}r{k1}"),
                    &row,
                    direction,
                    precision,
                );
                for (k2, value) in row_outputs.into_iter().enumerate() {
                    outputs[k1 + r1 * k2] = value;
                }
            }
            outputs
        }
        _ => {
            debug_assert!(radix % 2 == 1, "unsupported small DFT radix {radix}");
            let half = (radix - 1) / 2;
            let mut sums = Vec::with_capacity(half);
            let mut differences = Vec::with_capacity(half);
            for k in 1..=half {
                let (a, b) = (&inputs[k], &inputs[radix - k]);
                sums.push(bind(
                    body,
                    format!("{prefix}_s{k}"),
                    format!("c_add({a}, {b})"),
                ));
                differences.push(bind(
                    body,
                    format!("{prefix}_d{k}"),
                    format!("c_sub({a}, {b})"),
                ));
            }
            let mut outputs = vec![String::new(); radix];
            let total = sums.iter().fold(inputs[0].clone(), |sum, value| {
                format!("c_add({sum}, {value})")
            });
            outputs[0] = bind(body, format!("{prefix}_0"), total);
            for m in 1..=half {
                let angle =
                    |k: usize| std::f64::consts::TAU * ((k * m) % radix) as f64 / radix as f64;
                let cosine = sums
                    .iter()
                    .enumerate()
                    .fold(inputs[0].clone(), |sum, (i, value)| {
                        let term = scaled_complex_expr(value, Some(angle(i + 1).cos()), precision);
                        format!("c_add({sum}, {term})")
                    });
                let sine = differences
                    .iter()
                    .enumerate()
                    .map(|(i, value)| {
                        scaled_complex_expr(value, Some(angle(i + 1).sin()), precision)
                    })
                    .reduce(|sum, term| format!("c_add({sum}, {term})"))
                    .expect("odd radices above 1 have pairs");
                let cosine = bind(body, format!("{prefix}_a{m}"), cosine);
                let sine = bind(body, format!("{prefix}_b{m}"), sine);
                // Forward: X[m] = A - iB and X[R - m] = A + iB; inverse flips
                // the rotation.
                let rotated = bind(
                    body,
                    format!("{prefix}_r{m}"),
                    quarter_turn_wgsl(&sine, direction, precision),
                );
                outputs[m] = bind(
                    body,
                    format!("{prefix}_{m}"),
                    format!("c_add({cosine}, {rotated})"),
                );
                outputs[radix - m] = bind(
                    body,
                    format!("{prefix}_{}", radix - m),
                    format!("c_sub({cosine}, {rotated})"),
                );
            }
            outputs
        }
    }
}

/// `value` times `W_4^quarters` of `direction`: a component swap and sign
/// changes, or a negation.
fn rotate_quarters_wgsl(
    value: &str,
    quarters: usize,
    direction: FftDirection,
    precision: AxisPrecision,
) -> String {
    match quarters % 4 {
        0 => value.to_owned(),
        1 => quarter_turn_wgsl(value, direction, precision),
        2 => format!("-{value}"),
        _ => quarter_turn_wgsl(value, direction.opposite(), precision),
    }
}

/// The smallest prime factor of `n` (at least 2).
fn smallest_prime_factor(n: usize) -> usize {
    (2..n).find(|factor| n.is_multiple_of(*factor)).unwrap_or(n)
}

/// `value` times `-i` for a forward transform and `i` for an inverse one:
/// a swap of the real and imaginary parts and one sign.
fn quarter_turn_wgsl(value: &str, direction: FftDirection, precision: AxisPrecision) -> String {
    match (precision, direction) {
        (AxisPrecision::Df64, FftDirection::Forward) => {
            format!("vec4<f32>({value}.z, {value}.w, -{value}.x, -{value}.y)")
        }
        (AxisPrecision::Df64, FftDirection::Inverse) => {
            format!("vec4<f32>(-{value}.z, -{value}.w, {value}.x, {value}.y)")
        }
        (_, FftDirection::Forward) => format!("vec2<f32>({value}.y, -{value}.x)"),
        (_, FftDirection::Inverse) => format!("vec2<f32>(-{value}.y, {value}.x)"),
    }
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
    let source = if key.lines_per_workgroup > 1 || key.split_pass.is_some() {
        generate_fused_smooth_multiline_stage_wgsl(
            &config,
            key.lines_per_workgroup as usize,
            key.split_pass.as_ref(),
        )
    } else {
        generate_fused_smooth_stage_wgsl(&config)
    };
    let source = if key.padded_indices {
        pad_workgroup_indices(&source)
    } else {
        source
    };
    if key.in_place {
        in_place_wgsl(&source)
    } else {
        source
    }
}

/// Fused smooth-radix kernel that transforms `lines` lines per workgroup; see
/// [`generate_fused_pow2_multiline_stage_wgsl`]. Every stage stays in
/// workgroup memory and a final store pass writes the lines out, so strided
/// axes store element-major as well as load that way.
pub(crate) fn generate_fused_smooth_multiline_stage_wgsl(
    config: &FusedSmoothStageWgslConfig<'_>,
    lines: usize,
    split: Option<&SplitPass>,
) -> String {
    debug_assert_eq!(config.rank, config.dims.len());
    debug_assert_eq!(config.axis_length, config.dims[config.axis]);
    debug_assert_eq!(config.factors.iter().product::<usize>(), config.axis_length);
    debug_assert!(lines >= 1 && config.workgroup_size > 0);

    let maybe_scale = if config.apply_scale {
        let value = scaled_complex_expr("value", Some(config.scale_factor), config.precision);
        format!("      value = {value};\n")
    } else {
        String::new()
    };
    let line_base_fn = wgsl_line_base_fn(config.rank, config.axis, config.dims);
    let mut ns = 1usize;
    let mut radix_stages = String::new();
    for (stage_index, &radix) in config.factors.iter().enumerate() {
        ns *= radix;
        if stage_index == 0 {
            radix_stages.push_str(&generate_fused_smooth_first_stage_multiline_wgsl(
                config.axis_length,
                radix,
                config.direction,
                config.workgroup_size,
                lines,
                config.stride_complex != 1,
                config.precision,
            ));
        } else {
            radix_stages.push_str(&generate_in_place_smooth_fft_stage_multiline_wgsl(
                config.axis_length,
                radix,
                ns,
                config.direction,
                config.workgroup_size,
                lines,
                "twiddle",
                config.precision,
            ));
        }
    }
    debug_assert_eq!(ns, config.axis_length);
    let store = multiline_store(split, config.stride_complex);
    let element_major = config.stride_complex != 1 || store.stride_out != 1;
    let store_split = multiline_split_wgsl(store.stride_out);

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
const STRIDE_OUT: u32 = {stride_out}u;
const WORKGROUP_SIZE: u32 = {workgroup_size}u;
const LINES: u32 = {lines}u;
const LINE_STRIDE: u32 = {line_stride}u;

var<workgroup> scratch: array<vec2<f32>, {scratch_len}>;

{line_base_fn}

{line_base_out_fn}

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


{radix_stages}
  for (var e: u32 = lid.x; e < LINES * N; e = e + WORKGROUP_SIZE) {{
    {store_split}
    if (lineSlot < lineCount) {{
      var value: vec2<f32> = scratch[lineSlot * LINE_STRIDE + p];
{row_twiddle}{maybe_scale}      let dstIdx: u32 = line_base_out(lineStart + lineSlot) + p * STRIDE_OUT - params.elementBase;
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
            scratch_len = multiline_line_stride(config.axis_length, lines, element_major) * lines,
            line_stride = multiline_line_stride(config.axis_length, lines, element_major),
            twiddle_lookup_wgsl =
                multiline_twiddle_lookup_wgsl(config.direction, config.precision, split),
            line_base_fn = line_base_fn,
            entry_params = crate::runtime::dispatch::WGSL_FLAT_ENTRY_PARAMS,
            flat_workgroup_index = crate::runtime::dispatch::WGSL_FLAT_WORKGROUP_INDEX,
            radix_stages = radix_stages,
            maybe_scale = maybe_scale,
            store_split = store_split,
            stride_out = store.stride_out,
            line_base_out_fn = store.line_base_out_fn,
            row_twiddle = store.row_twiddle,
        ),
        config.precision,
    )
}

/// One in-place smooth radix stage over `lines` lines of `axis_length`
/// elements in `scratch`, with each line's units on consecutive invocations.
#[allow(clippy::too_many_arguments)]
pub(crate) fn generate_in_place_smooth_fft_stage_multiline_wgsl(
    axis_length: usize,
    radix: usize,
    ns: usize,
    direction: FftDirection,
    workgroup_size: u32,
    lines: usize,
    twiddle_fn_name: &str,
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
            radix,
            ns_div_r,
            n_div_r,
            n_div_ns,
            direction,
            slot,
            &|index| format!("scratch[{index}]"),
            twiddle_fn_name,
            precision,
        );
        computes.push_str(&format!(
            r#"    let slotUnit_{slot}: u32 = lid.x + {slot}u * WORKGROUP_SIZE;
    let lineSlot_{slot}: u32 = slotUnit_{slot} / {unit_count}u;
    let unit_{slot}: u32 = slotUnit_{slot} - lineSlot_{slot} * {unit_count}u;
    let block_{slot}: u32 = unit_{slot} / {ns_div_r}u;
    let j_{slot}: u32 = unit_{slot} - block_{slot} * {ns_div_r}u;
    let lineOffset_{slot}: u32 = lineSlot_{slot} * LINE_STRIDE;
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

pub(crate) fn complex_wgsl() -> &'static str {
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

pub(crate) fn twiddle_lookup_wgsl(direction: FftDirection, precision: AxisPrecision) -> String {
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

pub(crate) fn radix_root_wgsl(
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

/// Workgroup-memory elements between the lines of a multi-line kernel. When
/// neighbouring invocations touch neighbouring lines (`element_major`), one
/// element of padding keeps those lines in different banks instead of all in
/// one. Line-major kernels gain nothing from it and slow down.
pub(crate) const fn multiline_line_stride(
    axis_length: usize,
    lines: usize,
    element_major: bool,
) -> usize {
    if lines > 1 && element_major {
        axis_length + 1
    } else {
        axis_length
    }
}

/// Whether a multi-line kernel loads or stores element-major: a strided
/// axis, or a split pass storing into a strided output axis.
pub(crate) fn multiline_element_major(stride_complex: usize, split: Option<&SplitPass>) -> bool {
    let stride_out = match split.and_then(|split| split.output.as_ref()) {
        Some((dims, axis)) => stride_for_axis(dims, *axis),
        None => stride_complex,
    };
    stride_complex != 1 || stride_out != 1
}

/// Global loads of a multi-line kernel into workgroup memory, as straight-line
/// code: every invocation issues all of its loads before storing any, so their
/// latencies overlap. A loop would serialize them, since naga bounds every loop
/// and that keeps drivers from unrolling it. `offset` is the element's offset
/// within its line of workgroup memory, in terms of its index `P`.
///
/// Lines past `lineCount` in a partial last workgroup read a valid line
/// instead and skip the store.
fn multiline_load_wgsl(
    axis_length: usize,
    lines: usize,
    workgroup_size: usize,
    element_major: bool,
    offset: &str,
) -> String {
    let total = axis_length * lines;
    let count = total.div_ceil(workgroup_size);
    let mut loads = String::new();
    let mut stores = String::new();
    // Element-major lines with a line count dividing the workgroup keep one
    // line per invocation: its base address is computed once.
    let hoisted = element_major && workgroup_size.is_multiple_of(lines);
    if hoisted {
        loads.push_str("  let loadSlot: u32 = lid.x % LINES;\n  let loadBase: u32 = line_base(lineStart + min(loadSlot, lineCount - 1u)) - params.elementBase;\n  let loadP: u32 = lid.x / LINES;\n");
    }
    for k in 0..count {
        let tail = (k + 1) * workgroup_size > total;
        let (slot, p) = if hoisted {
            let p = format!("loadP + {}u", k * (workgroup_size / lines));
            ("loadSlot".to_owned(), p)
        } else {
            loads.push_str(&format!(
                "  let e_{k}: u32 = lid.x + {}u;\n",
                k * workgroup_size
            ));
            if element_major {
                loads.push_str(&format!(
                    "  let slot_{k}: u32 = e_{k} % LINES;\n  let p_{k}: u32 = e_{k} / LINES;\n"
                ));
            } else {
                loads.push_str(&format!(
                    "  let slot_{k}: u32 = e_{k} / N;\n  let p_{k}: u32 = e_{k} - slot_{k} * N;\n"
                ));
            }
            (format!("slot_{k}"), format!("p_{k}"))
        };
        let clamped_p = if tail {
            format!("min({p}, N - 1u)")
        } else {
            p.clone()
        };
        let base = if hoisted {
            "loadBase".to_owned()
        } else {
            format!("line_base(lineStart + min({slot}, lineCount - 1u)) - params.elementBase")
        };
        loads.push_str(&format!(
            "  let loaded_{k}: vec2<f32> = src[{base} + ({clamped_p}) * STRIDE];\n"
        ));
        let guard = if tail {
            format!("{slot} < lineCount && {p} < N")
        } else {
            format!("{slot} < lineCount")
        };
        stores.push_str(&format!(
            "  if ({guard}) {{\n    scratch[{slot} * LINE_STRIDE + {}] = loaded_{k};\n  }}\n",
            offset.replace('P', &format!("({p})"))
        ));
    }
    loads + &stores
}

/// Maps a flat element index `e` to `(lineSlot, p)`: line-major for
/// contiguous lines, element-major for strided lines so neighbouring
/// invocations touch neighbouring lines.
fn multiline_split_wgsl(stride: usize) -> &'static str {
    if stride == 1 {
        "let lineSlot: u32 = e / N;\n    let p: u32 = e - lineSlot * N;"
    } else {
        "let lineSlot: u32 = e % LINES;\n    let p: u32 = e / LINES;"
    }
}

/// Twiddle lookup for a multi-line kernel. A split-axis pass binds the full
/// axis's table, so `twiddle` scales its own indices and `twiddle_full` reads
/// the table directly for the split twiddle.
fn multiline_twiddle_lookup_wgsl(
    direction: FftDirection,
    precision: AxisPrecision,
    split: Option<&SplitPass>,
) -> String {
    let lookup = twiddle_lookup_wgsl(direction, precision);
    match split {
        None => lookup,
        Some(split) => format!(
            "{}\n\nfn twiddle(index: u32) -> vec2<f32> {{\n  return twiddle_full(index * {}u);\n}}",
            lookup.replace("fn twiddle(", "fn twiddle_full("),
            split.twiddle_scale
        ),
    }
}

/// Store-side WGSL of a multi-line kernel: the split twiddle of a first pass
/// and the transposed output addressing of a second pass.
struct MultilineStore {
    line_base_out_fn: String,
    stride_out: usize,
    row_twiddle: String,
}

fn multiline_store(split: Option<&SplitPass>, stride_in: usize) -> MultilineStore {
    let (line_base_out_fn, stride_out) = match split.and_then(|split| split.output.as_ref()) {
        Some((dims, axis)) => (
            wgsl_line_base_fn(dims.len(), *axis, dims)
                .replace("fn line_base(", "fn line_base_out("),
            stride_for_axis(dims, *axis),
        ),
        None => (
            "fn line_base_out(line: u32) -> u32 {\n  return line_base(line);\n}".to_owned(),
            stride_in,
        ),
    };
    let row_twiddle = match split.and_then(|split| split.row_twiddle) {
        Some((rows, row_stride_lines)) => format!(
            "      let row: u32 = ((lineStart + lineSlot) / {row_stride_lines}u) % {rows}u;\n      value = c_mul(value, twiddle_full(row * p));\n"
        ),
        None => String::new(),
    };
    MultilineStore {
        line_base_out_fn,
        stride_out,
        row_twiddle,
    }
}

pub(crate) fn specialize_complex_wgsl(source: String, precision: AxisPrecision) -> String {
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

pub(crate) fn scaled_complex_expr(
    value: &str,
    scale_factor: Option<f64>,
    precision: AxisPrecision,
) -> String {
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

pub(crate) fn wgsl_line_base_fn(rank: usize, axis: usize, dims: &[usize]) -> String {
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
            None,
        )
    }

    #[test]
    fn multiline_fused_pow2_packs_lines_and_orders_loads_by_axis() {
        let contiguous = multiline_wgsl_for(&[64, 64], 0, 32);
        assert!(contiguous.contains("const LINES: u32 = 32u;"));
        assert!(contiguous.contains("var<workgroup> scratch: array<vec2<f32>, 2048>;"));
        assert!(contiguous.contains("let lineSlot: u32 = e / N;"));
        // Every load is issued before the first store to workgroup memory.
        assert!(contiguous.contains("let slot_0: u32 = e_0 / N;"));
        assert!(contiguous.contains(
            "scratch[slot_7 * LINE_STRIDE + (reverseBits((p_7)) >> (32u - LOG_N))] = loaded_7;"
        ));
        assert!(
            contiguous.find("loaded_7: vec2<f32> = src[").unwrap()
                < contiguous.find("] = loaded_0;").unwrap()
        );
        assert!(contiguous.contains("const LINE_STRIDE: u32 = 64u;"));
        assert!(contiguous.contains("let lineCount: u32 = min(LINES, activeLines - groupLine);"));
        assert!(contiguous.contains("value = value * vec2<f32>(0.25, 0.25);"));
        crate::runtime::assert_workgroup_var_written_before_read(&contiguous, "scratch");

        // Strided axes load element-major so neighbouring invocations read
        // neighbouring lines.
        let strided = multiline_wgsl_for(&[64, 64], 1, 32);
        assert!(strided.contains("const STRIDE: u32 = 64u;"));
        assert!(strided.contains("let lineSlot: u32 = e % LINES;"));
        // Each invocation loads one line: its base is computed once.
        assert!(strided.contains("let loadSlot: u32 = lid.x % LINES;"));
        assert!(
            strided.contains("let loaded_7: vec2<f32> = src[loadBase + (loadP + 56u) * STRIDE];")
        );
        // One padding element per line keeps neighbouring lines in
        // different banks.
        assert!(strided.contains("const LINE_STRIDE: u32 = 65u;"));
        assert!(strided.contains("var<workgroup> scratch: array<vec2<f32>, 2080>;"));
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
            None,
        );
        assert!(wgsl.contains("const LINES: u32 = 16u;"));
        assert!(wgsl.contains("var<workgroup> scratch: array<vec2<f32>, 784>;"));
        assert!(wgsl.contains("let lineSlot: u32 = e % LINES;"));
        assert!(wgsl.contains("if (lineSlot_0 < lineCount) {"));
        // The first stage reads the strided lines from global memory,
        // neighbouring invocations on neighbouring lines.
        assert!(wgsl.contains("let lineSlot_0: u32 = slotUnit_0 % LINES;"));
        assert!(wgsl.contains(
            "let first_0: u32 = line_base(lineStart + lineSlot_0) - params.elementBase;"
        ));
        assert!(!wgsl.contains("= src[srcIdx]"));
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
        // Each line takes one padding element, and strided counts round down
        // to a power of two.
        assert_eq!(
            fused_lines_per_workgroup(1024, 64, AxisPrecision::F32, 1 << 20, storage),
            4
        );
        assert_eq!(
            fused_lines_per_workgroup(1080, 1920, AxisPrecision::F32, 1920, storage),
            4
        );
        assert_eq!(
            fused_lines_per_workgroup(1024, 64, AxisPrecision::F64, 1 << 20, storage),
            2
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
    fn long_axis_split_prefers_balanced_fused_factors() {
        let limits = wgpu::Limits {
            max_compute_workgroup_storage_size: 32 * 1024,
            ..wgpu::Limits::default()
        };
        let split =
            |len, precision| long_axis_split(len, precision, DEFAULT_FUSED_WORKGROUP_SIZE, &limits);
        assert_eq!(split(8192, AxisPrecision::F32), Some((64, 128)));
        assert_eq!(split(12288, AxisPrecision::F32), Some((96, 128)));
        assert_eq!(split(1 << 20, AxisPrecision::F32), Some((1024, 1024)));
        assert_eq!(split(3000, AxisPrecision::F32), Some((50, 60)));
        assert_eq!(split(4096, AxisPrecision::Df64), Some((64, 64)));
        // Every factor must fit one workgroup, and primes have no split.
        assert_eq!(split(1 << 26, AxisPrecision::F32), None);
        assert_eq!(split(8191, AxisPrecision::F32), None);
        assert_eq!(split(2 * 4099, AxisPrecision::F32), None);
    }

    fn split_pow2_wgsl(
        dims: &[usize],
        axis: usize,
        lines: usize,
        apply_scale: bool,
        split: &SplitPass,
    ) -> String {
        generate_fused_pow2_multiline_stage_wgsl(
            &FusedPow2StageWgslConfig {
                rank: dims.len(),
                axis,
                dims,
                axis_length: dims[axis],
                stride_complex: stride_for_axis(dims, axis),
                direction: FftDirection::Forward,
                workgroup_size: FUSED_POW2_WORKGROUP_SIZE,
                apply_scale,
                scale_factor: 0.5,
                precision: AxisPrecision::F32,
            },
            lines,
            Some(split),
        )
    }

    #[test]
    fn split_passes_twiddle_rows_and_store_transposed() {
        // N = 8192 as n = n2 + 128 * n1, viewed as dims [128, 64].
        let dims = [128usize, 64];
        let first = split_pow2_wgsl(
            &dims,
            1,
            8,
            false,
            &SplitPass {
                full_length: 8192,
                twiddle_scale: 128,
                row_twiddle: Some((128, 1)),
                output: None,
            },
        );
        assert!(first.contains("fn twiddle_full(index: u32)"));
        assert!(first.contains("return twiddle_full(index * 128u);"));
        assert!(first.contains("const STRIDE: u32 = 128u;"));
        assert!(first.contains("const STRIDE_OUT: u32 = 128u;"));
        assert!(first.contains("let row: u32 = ((lineStart + lineSlot) / 1u) % 128u;"));
        assert!(first.contains("value = c_mul(value, twiddle_full(row * p));"));
        crate::runtime::assert_workgroup_var_written_before_read(&first, "scratch");

        // The second pass reads contiguous n2 lines and writes them as the
        // strided axis of [64, 128], after scaling.
        let second = split_pow2_wgsl(
            &dims,
            0,
            1,
            true,
            &SplitPass {
                full_length: 8192,
                twiddle_scale: 64,
                row_twiddle: None,
                output: Some((vec![64, 128], 1)),
            },
        );
        assert!(second.contains("return twiddle_full(index * 64u);"));
        assert!(!second.contains("let row: u32"));
        assert!(second.contains("const STRIDE: u32 = 1u;"));
        assert!(second.contains("const STRIDE_OUT: u32 = 64u;"));
        assert!(second.contains("fn line_base_out(line: u32) -> u32 {"));
        assert!(second.contains("value = value * vec2<f32>(0.5, 0.5);"));
        assert!(second.contains("line_base_out(lineStart + lineSlot) + p * STRIDE_OUT"));
        crate::runtime::assert_workgroup_var_written_before_read(&second, "scratch");
    }

    #[test]
    fn split_passes_have_distinct_cache_keys() {
        let key = FusedPow2StageKey::new(
            2,
            1,
            &[128, 64],
            64,
            128,
            FftDirection::Forward,
            256,
            false,
            1.0,
            AxisPrecision::F32,
        )
        .with_lines_per_workgroup(8);
        let first = key.clone().with_split_pass(SplitPass {
            full_length: 8192,
            twiddle_scale: 128,
            row_twiddle: Some((128, 1)),
            output: None,
        });
        let second = key.clone().with_split_pass(SplitPass {
            full_length: 8192,
            twiddle_scale: 128,
            row_twiddle: None,
            output: Some((vec![64, 128], 1)),
        });
        assert!(!key.stable_key().contains(":split="));
        assert!(first
            .stable_key()
            .contains(":split=n8192.scale128.rows128.rowstride1"));
        assert!(second
            .stable_key()
            .contains(":split=n8192.scale128.out64x128.axis1"));
        assert_ne!(first.stable_key(), second.stable_key());
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
        // Radix 8 reads W^j, W^2j, W^3j, and W^6j; W^7j = W^6j W^j.
        assert!(wgsl.contains("lookupInverseRoot(j_0 * 6u)"));
        assert!(!wgsl.contains("lookupInverseRoot(j_0 * 7u)"));
        assert!(wgsl.contains("c_mul(tw_0_6, tw_0_1)"));
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
    fn fused_smooth_schedules_take_few_balanced_stages() {
        let schedule = |n: usize| {
            fused_smooth_factors(n, &crate::runtime::factor_supported_length(n).unwrap())
        };
        assert_eq!(schedule(1920), [16, 12, 10]);
        assert_eq!(schedule(1080), [12, 10, 9]);
        assert_eq!(schedule(720), [16, 9, 5]);
        assert_eq!(schedule(1280), [16, 16, 5]);
        assert_eq!(schedule(1000), [10, 10, 10]);
        // A single composite stage falls back to the multi-pass factors.
        assert_eq!(schedule(12), [4, 3]);
        assert_eq!(schedule(15), [5, 3]);
    }

    #[test]
    fn padded_smooth_kernels_spread_every_workgroup_index() {
        let wgsl = pad_workgroup_indices(
            "var<workgroup> scratch: array<vec2<f32>, 1280>;
let a = scratch[i * 16u + scratch_len[j]];
scratch[k] = a;
",
        );
        assert!(wgsl.contains("fn padded(index: u32) -> u32 {"));
        assert!(wgsl.contains("var<workgroup> scratch: array<vec2<f32>, 1359>;"));
        assert!(wgsl.contains("scratch[padded(i * 16u + scratch_len[j])]"));
        assert!(wgsl.contains("scratch[padded(k)] = a;"));
        assert_eq!(padded_workgroup_len(1280), 1359);
        assert!(fused_smooth_pads_indices(&[16, 16, 5], false));
        assert!(!fused_smooth_pads_indices(&[16, 16, 5], true));
        assert!(!fused_smooth_pads_indices(&[10, 10, 10], false));
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
