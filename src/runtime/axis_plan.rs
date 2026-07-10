use bytemuck::{Pod, Zeroable};

use crate::config::{FftConfig, FftDirection, Normalization};
use crate::error::{FftError, Result};
use crate::runtime::buffer_view::BufferView;
use crate::runtime::dispatch::{max_workgroups_per_dimension, split_workgroups};
use crate::runtime::large_graph::ElementFormat;
use crate::runtime::pipeline_cache::{
    with_device_pipeline_cache, ComputePipelineCacheKey, FusedPow2StageKey, PipelineLayoutCacheKey,
    StockhamStageKey,
};
use crate::runtime::window_scheduler::WindowScheduler;

const WORKGROUP_SIZE: u32 = 64;
const FUSED_POW2_WORKGROUP_SIZE: u32 = 256;
const COMPLEX_F32_BYTES: u64 = 8;

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AxisPrecision {
    F32,
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
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AxisStageKind {
    Stockham { radix: usize, ns: usize },
    FusedPow2 { axis_length: usize },
}

impl AxisStageKind {
    fn detail(self) -> String {
        match self {
            Self::Stockham { radix, ns } => format!("stockham.radix{radix}.ns{ns}"),
            Self::FusedPow2 { axis_length } => format!("fused_pow2.n{axis_length}"),
        }
    }
}

pub(crate) struct AxisStage {
    pub(crate) axis: usize,
    pub(crate) kind: AxisStageKind,
    pub(crate) stride_complex: usize,
    pub(crate) apply_scale: bool,
    pub(crate) pipeline_key: ComputePipelineCacheKey,
    workgroups_x: u32,
    pipeline: wgpu::ComputePipeline,
}

pub(crate) struct AxisPlan {
    config: AxisPlanConfig,
    factors: Vec<Vec<usize>>,
    stages: Vec<AxisStage>,
    bind_group_layout: wgpu::BindGroupLayout,
    params_buffer: wgpu::Buffer,
    temp_buffer: Option<wgpu::Buffer>,
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
    pub(crate) scale_factor: f32,
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
    pub(crate) scale_factor: f32,
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
            precision: AxisPrecision::F32,
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
            (AxisLayout::Interleaved, AxisPrecision::F32) => {}
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

    pub(crate) fn scale(&self) -> Result<f32> {
        if let Some(bits) = self.scale_override_bits {
            return Ok(f32::from_bits(bits));
        }

        let total = product(&self.shape) as f32;
        let scale = match (self.direction, self.normalization) {
            (_, Normalization::None) => 1.0,
            (FftDirection::Forward, Normalization::Forward) => 1.0 / total,
            (FftDirection::Inverse, Normalization::Inverse) => 1.0 / total,
            (_, Normalization::Orthogonal) => 1.0 / total.sqrt(),
            _ => 1.0,
        };
        Ok(scale)
    }

    #[cfg(test)]
    pub(crate) fn planned_workspace_size_bytes(&self) -> Result<u64> {
        self.validate()?;
        let stage_count = self.stage_count()?;
        Ok(workspace_size_bytes_for_stage_count(
            stage_count,
            self.total_complex()?,
        ))
    }

    #[cfg(test)]
    fn stage_count(&self) -> Result<usize> {
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
        config.validate()?;

        let total_complex = config.total_complex()?;
        let total_complex_u32 = total_complex as u32;
        let scale = config.scale()?;
        let apply_any_scale = (scale - 1.0).abs() > f32::EPSILON;

        let bind_group_layout = with_device_pipeline_cache(device, |cache| {
            cache.get_bind_group_layout(device, PipelineLayoutCacheKey::AxisPlanInterleavedF32)
        });

        let mut factors = Vec::with_capacity(config.axes.len());
        let mut stages = Vec::new();

        for (axis_index, &axis) in config.axes.iter().enumerate() {
            let axis_len = config.shape[axis];
            let axis_factors = crate::runtime::factor_supported_length(axis_len)?;
            let stride_complex = stride_for_axis(&config.shape, axis);
            let final_axis = axis_index + 1 == config.axes.len();

            if fused_pow2_supported(axis_len, &device.limits()) {
                let apply_scale = apply_any_scale && final_axis;
                let shader_key = FusedPow2StageKey::new(
                    config.shape.len(),
                    axis,
                    &config.shape,
                    axis_len,
                    stride_complex,
                    config.direction,
                    FUSED_POW2_WORKGROUP_SIZE,
                    apply_scale,
                    scale,
                );
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
                    workgroups_x: total_complex_u32 / axis_len as u32,
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
                        WORKGROUP_SIZE,
                        apply_scale,
                        scale,
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
                        workgroups_x: total_complex_u32.div_ceil(WORKGROUP_SIZE),
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

        let required_buffer_size_bytes = total_complex as u64 * COMPLEX_F32_BYTES;
        let workspace_size_bytes =
            workspace_size_bytes_for_stage_count(stages.len(), total_complex);
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

    pub(crate) fn graph_stage_kinds(&self) -> Vec<AxisStageKind> {
        self.stages.iter().map(|stage| stage.kind).collect()
    }

    pub(crate) fn execute(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
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
        encoder: &mut wgpu::CommandEncoder,
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
        encoder: &mut wgpu::CommandEncoder,
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
        encoder: &mut wgpu::CommandEncoder,
        input: BufferView<'_>,
        output: BufferView<'_>,
        workspace: Option<BufferView<'_>>,
    ) -> Result<()> {
        debug_assert!(
            !self.stages.is_empty(),
            "AxisPlan execution requires at least one FFT stage"
        );
        debug_assert_eq!(self.config.layout, AxisLayout::Interleaved);
        debug_assert_eq!(self.config.precision, AxisPrecision::F32);

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
            let src_resource =
                scheduler.storage_binding_resource(&src, ElementFormat::ComplexF32)?;
            let dst_resource =
                scheduler.storage_binding_resource(&dst, ElementFormat::ComplexF32)?;

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
                ],
            });

            let pass_label = format!(
                "wgpu_fft.axis_plan.pass.axis{}.{}.stride{}.scale{}.cache{}",
                stage.axis,
                stage_detail,
                stage.stride_complex,
                stage.apply_scale,
                stage.pipeline_key.stable_key()
            );
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some(&pass_label),
                    timestamp_writes: None,
                });
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

pub(crate) fn workspace_size_bytes_for_stage_count(
    stage_count: usize,
    total_complex: usize,
) -> u64 {
    if stage_count > 1 {
        total_complex as u64 * COMPLEX_F32_BYTES
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

fn fused_pow2_supported(axis_length: usize, limits: &wgpu::Limits) -> bool {
    fused_pow2_supported_by_limits(
        axis_length,
        u64::from(limits.max_compute_workgroup_storage_size),
        limits.max_compute_invocations_per_workgroup,
        limits.max_compute_workgroup_size_x,
    )
}

fn fused_pow2_supported_by_limits(
    axis_length: usize,
    max_workgroup_storage_bytes: u64,
    max_invocations_per_workgroup: u32,
    max_workgroup_size_x: u32,
) -> bool {
    if axis_length < 2 || !axis_length.is_power_of_two() {
        return false;
    }
    let Some(scratch_bytes) = axis_length.checked_mul(COMPLEX_F32_BYTES as usize) else {
        return false;
    };
    scratch_bytes as u64 <= max_workgroup_storage_bytes
        && FUSED_POW2_WORKGROUP_SIZE <= max_invocations_per_workgroup
        && FUSED_POW2_WORKGROUP_SIZE <= max_workgroup_size_x
}

pub(crate) fn generate_fused_pow2_stage_wgsl(config: &FusedPow2StageWgslConfig<'_>) -> String {
    debug_assert_eq!(config.rank, config.dims.len());
    debug_assert_eq!(config.axis_length, config.dims[config.axis]);
    debug_assert!(config.axis_length.is_power_of_two());
    debug_assert_eq!(config.workgroup_size, FUSED_POW2_WORKGROUP_SIZE);

    let sign = match config.direction {
        FftDirection::Forward => "-1.0",
        FftDirection::Inverse => "1.0",
    };
    let maybe_scale = if config.apply_scale {
        let scale = format_wgsl_f32(config.scale_factor);
        format!("    value = value * vec2<f32>({scale}, {scale});\n")
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
        ));
        previous *= radix;
    }
    debug_assert_eq!(previous, config.axis_length);

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

{complex_wgsl}

const N: u32 = {n}u;
const LOG_N: u32 = {log_n}u;
const STRIDE: u32 = {stride}u;
const WORKGROUP_SIZE: u32 = {workgroup_size}u;
const SIGN: f32 = {sign};
const SQRT_HALF: f32 = 0.7071067811865475244;

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
        sign = sign,
        line_base_fn = line_base_fn,
        entry_params = crate::runtime::dispatch::WGSL_FLAT_ENTRY_PARAMS,
        flat_workgroup_index = crate::runtime::dispatch::WGSL_FLAT_WORKGROUP_INDEX,
        radix_stages = radix_stages,
        maybe_scale = maybe_scale,
    )
}

fn generate_fused_radix_stage_wgsl(axis_length: usize, radix: usize, previous: usize) -> String {
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

    let twiddles = match radix {
        8 => {
            r#"      let z8: vec2<f32> = cis(SIGN * (2.0 * PI) * (f32(j) / f32(RADIX * PREVIOUS)));
      let z4: vec2<f32> = c_mul(z8, z8);
      let z2: vec2<f32> = c_mul(z4, z4);
      let root4: vec2<f32> = vec2<f32>(0.0, SIGN);
      let root8_1: vec2<f32> = vec2<f32>(SQRT_HALF, SIGN * SQRT_HALF);
      let root8_3: vec2<f32> = vec2<f32>(-SQRT_HALF, SIGN * SQRT_HALF);
      let z4_1: vec2<f32> = c_mul(z4, root4);
      let z8_1: vec2<f32> = c_mul(z8, root8_1);
      let z8_2: vec2<f32> = c_mul(z8, root4);
      let z8_3: vec2<f32> = c_mul(z8, root8_3);
"#
        }
        4 => {
            r#"      let z4: vec2<f32> = cis(SIGN * (2.0 * PI) * (f32(j) / f32(RADIX * PREVIOUS)));
      let z2: vec2<f32> = c_mul(z4, z4);
      let root4: vec2<f32> = vec2<f32>(0.0, SIGN);
      let z4_1: vec2<f32> = c_mul(z4, root4);
"#
        }
        2 => {
            r#"      let z2: vec2<f32> = cis(SIGN * (2.0 * PI) * (f32(j) / f32(RADIX * PREVIOUS)));
"#
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

    format!(
        r#"  {{
    const RADIX: u32 = {radix}u;
    const PREVIOUS: u32 = {previous}u;
    const UNIT_COUNT: u32 = {unit_count}u;
    for (var unit: u32 = lid.x; unit < UNIT_COUNT; unit = unit + WORKGROUP_SIZE) {{
      let block: u32 = unit / PREVIOUS;
      let j: u32 = unit - block * PREVIOUS;
      let base: u32 = block * (RADIX * PREVIOUS) + j;
{values}{twiddles}{butterflies}{writes}    }}
    workgroupBarrier();
  }}
"#,
        radix = radix,
        previous = previous,
        unit_count = axis_length / radix,
        values = values,
        twiddles = twiddles,
        butterflies = butterflies,
        writes = writes,
    )
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
    generate_fused_pow2_stage_wgsl(&FusedPow2StageWgslConfig {
        rank: key.rank,
        axis: key.axis,
        dims: &key.dims,
        axis_length: key.axis_length,
        stride_complex: key.stride_complex,
        direction: key.direction,
        workgroup_size: key.workgroup_size,
        apply_scale: key.apply_scale,
        scale_factor: key.scale_factor(),
    })
}

pub(crate) fn generate_stockham_radix_stage_wgsl(config: &StockhamStageWgslConfig<'_>) -> String {
    debug_assert_eq!(config.rank, config.dims.len());
    debug_assert_eq!(config.axis_length, config.dims[config.axis]);
    debug_assert_eq!(config.ns % config.radix, 0);
    debug_assert_eq!(config.axis_length % config.radix, 0);

    let sign = match config.direction {
        FftDirection::Forward => "-1.0",
        FftDirection::Inverse => "1.0",
    };
    let maybe_scale = if config.apply_scale {
        let scale = format_wgsl_f32(config.scale_factor);
        format!("  out = out * vec2<f32>({scale}, {scale});\n")
    } else {
        String::new()
    };
    let complex_wgsl = complex_wgsl();
    let line_base_fn = wgsl_line_base_fn(config.rank, config.axis, config.dims);

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

{complex_wgsl}

const N: u32 = {n}u;
const RADIX: u32 = {radix}u;
const NS: u32 = {ns}u;
const NS_DIV_R: u32 = {ns_div_r}u;
const N_DIV_R: u32 = {n_div_r}u;
const STRIDE: u32 = {stride}u;
const SIGN: f32 = {sign};

{line_base_fn}

@compute @workgroup_size({workgroup_size}, 1, 1)
fn main({entry_params}) {{
  {flat_workgroup_index}
  if (wgFlat > params.total / {workgroup_size}u) {{
    return;
  }}
  let idx: u32 = params.baseIndex + wgFlat * {workgroup_size}u + lid.x;
  if (idx >= params.total) {{
    return;
  }}

  let lineLocal: u32 = idx / N;
  let line: u32 = params.lineOffset + lineLocal;
  let p: u32 = idx - lineLocal * N;
  let baseLineGlobal: u32 = line_base(line);

  let block: u32 = p / NS;
  let p_in_block: u32 = p - block * NS;
  let offset: u32 = p - (p / NS_DIV_R) * NS_DIV_R;
  let base: u32 = block * NS_DIV_R + offset;

  let r: u32 = p_in_block;
  let angle: f32 = SIGN * (2.0 * PI) * (f32(r) / f32(NS));
  let w1: vec2<f32> = cis(angle);

  var w: vec2<f32> = vec2<f32>(1.0, 0.0);
  var out: vec2<f32> = vec2<f32>(0.0, 0.0);

  for (var q: u32 = 0u; q < RADIX; q = q + 1u) {{
    let srcIdxGlobal: u32 = baseLineGlobal + (base + q * N_DIV_R) * STRIDE;
    let srcIdx: u32 = srcIdxGlobal - params.elementBase;
    let x: vec2<f32> = src[srcIdx];
    out = c_add(out, c_mul(w, x));
    w = c_mul(w, w1);
  }}
{maybe_scale}  let dstIdxGlobal: u32 = baseLineGlobal + p * STRIDE;
  let dstIdx: u32 = dstIdxGlobal - params.elementBase;
  dst[dstIdx] = out;
}}
"#,
        complex_wgsl = complex_wgsl,
        n = config.axis_length,
        radix = config.radix,
        ns = config.ns,
        ns_div_r = config.ns / config.radix,
        n_div_r = config.axis_length / config.radix,
        stride = config.stride_complex,
        sign = sign,
        line_base_fn = line_base_fn,
        workgroup_size = config.workgroup_size,
        entry_params = crate::runtime::dispatch::WGSL_FLAT_ENTRY_PARAMS,
        flat_workgroup_index = crate::runtime::dispatch::WGSL_FLAT_WORKGROUP_INDEX,
        maybe_scale = maybe_scale,
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
    })
}

fn complex_wgsl() -> &'static str {
    r#"const PI: f32 = 3.1415926535897932384626433832795;

fn c_add(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
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
}

fn cis(angle: f32) -> vec2<f32> {
  return vec2<f32>(cos(angle), sin(angle));
}"#
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
    assert!(value.is_finite(), "WGSL f32 constants must be finite");

    let mut formatted = format!("{value:.9}");
    while formatted.contains('.') && formatted.ends_with('0') {
        formatted.pop();
    }
    if formatted.ends_with('.') {
        formatted.push('0');
    }
    if !formatted.contains('.') {
        formatted.push_str(".0");
    }
    formatted
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wgsl_for(radix: usize, axis_length: usize, ns: usize) -> String {
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
        })
    }

    fn fused_wgsl_for(axis_length: usize, direction: FftDirection) -> String {
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
        })
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
        assert!(wgsl.contains("const SIGN: f32 = -1.0;"));
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
        );

        assert_eq!(
            generate_stockham_radix_stage_wgsl_for_key(&key),
            generate_stockham_radix_stage_wgsl(&config)
        );
    }

    #[test]
    fn fused_pow2_gate_matches_storage_and_workgroup_boundaries() {
        for length in [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048] {
            assert!(fused_pow2_supported_by_limits(length, 16 * 1024, 256, 256));
        }
        assert!(!fused_pow2_supported_by_limits(4096, 16 * 1024, 256, 256));
        assert!(fused_pow2_supported_by_limits(4096, 48 * 1024, 256, 256));
        assert!(!fused_pow2_supported_by_limits(1, 48 * 1024, 256, 256));
        assert!(!fused_pow2_supported_by_limits(12, 48 * 1024, 256, 256));
        assert!(!fused_pow2_supported_by_limits(8192, 48 * 1024, 256, 256));
        assert!(!fused_pow2_supported_by_limits(2048, 16 * 1024, 255, 256));
        assert!(!fused_pow2_supported_by_limits(2048, 16 * 1024, 256, 255));
    }

    #[test]
    fn fused_pow2_generator_uses_one_shared_line_and_radix_schedule() {
        let wgsl = fused_wgsl_for(4096, FftDirection::Forward);
        assert!(wgsl.contains("@compute @workgroup_size(256, 1, 1)"));
        assert!(wgsl.contains("var<workgroup> scratch: array<vec2<f32>, 4096>;"));
        assert!(wgsl.contains("scratch[bit_reverse(p)] = src[srcIdx];"));
        assert!(wgsl.contains("dst[dstIdx] = value;"));
        assert!(wgsl.contains("const SIGN: f32 = -1.0;"));
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
        assert!(n1024.contains("const SIGN: f32 = 1.0;"));

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
        );
        assert_eq!(
            generate_fused_pow2_stage_wgsl_for_key(&key),
            generate_fused_pow2_stage_wgsl(&config)
        );
    }

    #[test]
    fn planned_workspace_is_zero_for_one_stage_and_full_buffer_for_multi_stage() {
        let one_stage = AxisPlanConfig::from_c2c_config(&FftConfig::new(8));
        assert_eq!(one_stage.planned_workspace_size_bytes().unwrap(), 0);

        let multi_stage = AxisPlanConfig::from_c2c_config(&FftConfig::new(16));
        assert_eq!(multi_stage.planned_workspace_size_bytes().unwrap(), 16 * 8);

        let nd_batched = AxisPlanConfig::from_c2c_config(&FftConfig::new_nd([4, 3]).with_batch(2));
        assert_eq!(nd_batched.planned_workspace_size_bytes().unwrap(), 24 * 8);
    }
}
