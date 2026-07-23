use bytemuck::{Pod, Zeroable};

use crate::config::{FftConfig, FftDirection, FftPrecision};
use crate::device::device_supports_precision;
use crate::error::{FftError, Result};
use crate::runtime::axis_policy::{resolve_axis_kinds_for_config, AxisKind};
use crate::runtime::buffer_view::BufferView;
use crate::runtime::c2c::{C2cPlan, C2cRoute};
use crate::runtime::dispatch::{max_workgroups_per_dimension, split_workgroups};
use crate::runtime::large_chunk::LargeChunkPlan;
use crate::runtime::large_graph::{
    ElementFormat, LargeExecutionGraph, LargeStage, LogicalBufferId, LogicalRange,
    StageRequirements,
};
use crate::runtime::large_policy::{
    resolve_large_routing_policy, LargePolicyLimits, LargeRouteMode, LargeRoutingPolicy,
    LargeRoutingPolicyInput,
};
use crate::runtime::logical_io::{
    BoundLogicalIo, FftEndpointFormat, FftLogicalLayout, FftLogicalView,
};
use crate::runtime::pipeline_cache::{
    with_device_pipeline_cache, ComputePipelineCacheKey, PipelineLayoutCacheKey, RealKernelKind,
    RealStageKey,
};
use crate::runtime::recorder::CommandRecorder;
use crate::runtime::stage_executor::StageExecutor;
use crate::runtime::window_scheduler::WindowScheduler;
use crate::tuning::{FftLargeRoute, FftTuningErrorKind};

// Generator/key tests use the default to keep cache-key compatibility explicit;
// executable real helpers always receive FftTuning::workgroup_size().
#[cfg(test)]
const DEFAULT_WORKGROUP_SIZE: u32 = 64;
const F32_BYTES: u64 = 4;
const COMPLEX_F32_BYTES: u64 = 8;

fn validate_real_precision(
    device: &wgpu::Device,
    config: &FftConfig,
    route: &'static str,
) -> Result<()> {
    match config.precision() {
        FftPrecision::F32 => Ok(()),
        FftPrecision::F64 if !device_supports_precision(device, FftPrecision::F64) => {
            Err(FftError::PrecisionUnsupported {
                requested: FftPrecision::F64,
                route,
                reason: "device-missing-shader-f64",
            })
        }
        FftPrecision::F64 => Err(FftError::PrecisionUnsupported {
            requested: FftPrecision::F64,
            route,
            reason: "real-f64-not-implemented",
        }),
        FftPrecision::Df64 => Err(FftError::PrecisionUnsupported {
            requested: FftPrecision::Df64,
            route,
            reason: "real-df64-not-implemented",
        }),
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct RealParams {
    value: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct RealWindowParams {
    total: u32,
    input_base: u32,
    output_base: u32,
    input_logical_start: u32,
    output_logical_start: u32,
    batch: u32,
    _pad1: u32,
    _pad2: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct RealStridedCopyParams {
    total_elements: u32,
    logical_per_batch: u32,
    element_offset: u32,
    element_stride: u32,
    batch_stride: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RealTransform {
    R2c,
    C2r,
}

impl RealTransform {
    const fn as_str(self) -> &'static str {
        match self {
            Self::R2c => "r2c",
            Self::C2r => "c2r",
        }
    }
}

pub struct R2cPlan {
    config: FftConfig,
    packed_shape: Vec<usize>,
    sizes: RealPlanSizes,
    axis_kinds: Vec<AxisKind>,
    large_routing_policy: LargeRoutingPolicy,
    execution: R2cExecution,
}

struct R2cNormalPlan {
    c2c: C2cPlan,
    real_to_complex: RealKernel,
    pack: RealKernel,
    full_input_buffer: wgpu::Buffer,
    full_output_buffer: wgpu::Buffer,
}

struct R2cLargeChunkPlan {
    chunk: RealLargeChunkPlan,
    child: Box<R2cPlan>,
    input_stage: wgpu::Buffer,
    output_stage: wgpu::Buffer,
}

enum R2cExecution {
    Normal(R2cNormalPlan),
    LargeChunk(R2cLargeChunkPlan),
    LargeDecomposition(R2cNormalPlan),
}

pub struct C2rPlan {
    config: FftConfig,
    packed_shape: Vec<usize>,
    sizes: RealPlanSizes,
    axis_kinds: Vec<AxisKind>,
    large_routing_policy: LargeRoutingPolicy,
    execution: C2rExecution,
}

struct C2rNormalPlan {
    c2c: C2cPlan,
    unpack: RealKernel,
    complex_to_real: RealKernel,
    full_input_buffer: wgpu::Buffer,
    full_output_buffer: wgpu::Buffer,
}

struct C2rLargeChunkPlan {
    chunk: RealLargeChunkPlan,
    child: Box<C2rPlan>,
    input_stage: wgpu::Buffer,
    output_stage: wgpu::Buffer,
}

enum C2rExecution {
    Normal(C2rNormalPlan),
    LargeChunk(C2rLargeChunkPlan),
    LargeDecomposition(C2rNormalPlan),
}

struct RealKernel {
    kind: RealKernelKind,
    pipeline_key: ComputePipelineCacheKey,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    params_buffer: wgpu::Buffer,
    workgroups_x: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct RealPlanSizes {
    real_bytes: u64,
    packed_bytes: u64,
    full_complex_bytes: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct RealLargeChunkPlan {
    batch_count: u64,
    chunk_batch_count: u64,
    real_bytes_per_batch: u64,
    packed_bytes_per_batch: u64,
    full_complex_bytes_per_batch: u64,
    real_staging_size_bytes: u64,
    packed_staging_size_bytes: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct RealLargeChunkRange {
    batch_start: u64,
    batch_count: u64,
    real_offset: u64,
    real_size: u64,
    packed_offset: u64,
    packed_size: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RealEndpointOrder {
    RealToPacked,
    PackedToReal,
}

fn policy_limits(policy: &LargeRoutingPolicy) -> LargePolicyLimits {
    LargePolicyLimits {
        max_storage_buffer_binding_size: policy.max_bind_bytes,
        max_buffer_size: policy.max_buffer_size,
    }
}

fn effective_real_policy_limits(
    device: &wgpu::Device,
    config: &FftConfig,
    explicit: Option<LargePolicyLimits>,
) -> LargePolicyLimits {
    effective_real_policy_limits_for_device_limits(&device.limits(), config, explicit)
}

fn effective_real_policy_limits_for_device_limits(
    device_limits: &wgpu::Limits,
    config: &FftConfig,
    explicit: Option<LargePolicyLimits>,
) -> LargePolicyLimits {
    let tuned = LargePolicyLimits::effective_with_overrides(
        device_limits,
        config.tuning().max_storage_buffer_binding_size(),
        config.tuning().max_buffer_size(),
    );
    explicit.map_or(tuned, |limits| limits.componentwise_min(tuned))
}

fn validate_real_tuning(device: &wgpu::Device, config: &FftConfig) -> Result<()> {
    config.tuning().validate_for_device(&device.limits())?;
    validate_real_route_tuning(config)
}

fn validate_real_route_tuning(config: &FftConfig) -> Result<()> {
    if config.tuning().large_route() != FftLargeRoute::Auto {
        return Err(FftError::InvalidTuning {
            kind: FftTuningErrorKind::UnsupportedForTransform,
            field: "large_route",
            value: config.tuning().large_route().as_str().to_owned(),
            reason: "forced large-route selection is not implemented for real transforms",
        });
    }
    Ok(())
}

fn graph_requirements(limits: LargePolicyLimits, scratch_bytes: u64) -> Result<StageRequirements> {
    StageRequirements::new(
        limits.max_storage_buffer_binding_size,
        limits.max_buffer_size,
        1,
        4,
        scratch_bytes,
    )
}

fn graph_requirements_covering(
    limits: LargePolicyLimits,
    range_bytes: u64,
    scratch_bytes: u64,
) -> Result<StageRequirements> {
    graph_requirements(
        LargePolicyLimits {
            max_storage_buffer_binding_size: limits.max_storage_buffer_binding_size,
            max_buffer_size: limits.max_buffer_size.max(range_bytes),
        },
        scratch_bytes,
    )
}

fn real_range(
    buffer: LogicalBufferId,
    offset_bytes: u64,
    size_bytes: u64,
    format: ElementFormat,
) -> Result<LogicalRange> {
    LogicalRange::new(buffer, offset_bytes, size_bytes, format)
}

fn build_real_normal_graph(
    label: &'static str,
    first_kernel_label: &'static str,
    final_kernel_label: &'static str,
    child_graph: &LargeExecutionGraph,
    order: RealEndpointOrder,
    sizes: &RealPlanSizes,
    limits: LargePolicyLimits,
    windowed_kernels: bool,
) -> Result<LargeExecutionGraph> {
    let mut graph = LargeExecutionGraph::new(label);
    let (input_bytes, input_format, output_bytes, output_format) = match order {
        RealEndpointOrder::RealToPacked => (
            sizes.real_bytes,
            ElementFormat::RealF32,
            sizes.packed_bytes,
            ElementFormat::PackedComplexF32,
        ),
        RealEndpointOrder::PackedToReal => (
            sizes.packed_bytes,
            ElementFormat::PackedComplexF32,
            sizes.real_bytes,
            ElementFormat::RealF32,
        ),
    };
    graph.push_stage(
        LargeStage::HostWindow {
            label: "logical-input",
            range: real_range(LogicalBufferId::Input, 0, input_bytes, input_format)?,
        },
        graph_requirements(limits, 0)?,
    )?;
    let first_input = real_range(LogicalBufferId::Input, 0, input_bytes, input_format)?;
    let first_output = real_range(
        LogicalBufferId::Stage(0),
        0,
        sizes.full_complex_bytes,
        ElementFormat::ComplexF32,
    )?;
    let first_work_items = input_bytes / input_format.bytes_per_element();
    let first_stage = if windowed_kernels {
        LargeStage::WindowedKernel {
            label: first_kernel_label,
            input: first_input,
            output: first_output,
            work_items: first_work_items,
        }
    } else {
        LargeStage::Kernel {
            label: first_kernel_label,
            input: first_input,
            output: first_output,
            work_items: first_work_items,
        }
    };
    graph.push_stage(
        first_stage,
        graph_requirements(limits, sizes.full_complex_bytes)?,
    )?;
    let child_input = real_range(
        LogicalBufferId::Stage(0),
        0,
        sizes.full_complex_bytes,
        ElementFormat::ComplexF32,
    )?;
    let child_output = real_range(
        LogicalBufferId::Stage(1),
        0,
        sizes.full_complex_bytes,
        ElementFormat::ComplexF32,
    )?;
    append_child_graph(
        &mut graph,
        child_graph,
        child_input,
        child_output,
        limits,
        2,
    )?;
    let final_input = real_range(
        LogicalBufferId::Stage(1),
        0,
        sizes.full_complex_bytes,
        ElementFormat::ComplexF32,
    )?;
    let final_output = real_range(LogicalBufferId::Output, 0, output_bytes, output_format)?;
    let final_work_items = output_bytes / output_format.bytes_per_element();
    let final_stage = if windowed_kernels {
        LargeStage::WindowedKernel {
            label: final_kernel_label,
            input: final_input,
            output: final_output,
            work_items: final_work_items,
        }
    } else {
        LargeStage::Kernel {
            label: final_kernel_label,
            input: final_input,
            output: final_output,
            work_items: final_work_items,
        }
    };
    graph.push_stage(
        final_stage,
        graph_requirements(limits, sizes.full_complex_bytes)?,
    )?;
    graph.push_stage(
        LargeStage::HostWindow {
            label: "logical-output",
            range: real_range(LogicalBufferId::Output, 0, output_bytes, output_format)?,
        },
        graph_requirements(limits, 0)?,
    )?;
    Ok(graph)
}

fn append_child_graph(
    graph: &mut LargeExecutionGraph,
    child_graph: &LargeExecutionGraph,
    child_input: LogicalRange,
    child_output: LogicalRange,
    limits: LargePolicyLimits,
    stage_index_base: u32,
) -> Result<()> {
    for stage in child_graph.stages() {
        let mapped = remap_child_stage(stage, child_input, child_output, stage_index_base)?;
        graph.push_stage(
            mapped,
            graph_requirements_covering(
                limits,
                max_stage_range_bytes(stage),
                stage_scratch_bytes(stage),
            )?,
        )?;
    }
    Ok(())
}

fn remap_child_stage(
    stage: &LargeStage,
    child_input: LogicalRange,
    child_output: LogicalRange,
    stage_index_base: u32,
) -> Result<LargeStage> {
    Ok(match *stage {
        LargeStage::Copy { label, src, dst } => LargeStage::Copy {
            label,
            src: remap_child_range(src, child_input, child_output, stage_index_base)?,
            dst: remap_child_range(dst, child_input, child_output, stage_index_base)?,
        },
        LargeStage::GatherScatter {
            label,
            src,
            dst,
            stride_elements,
        } => LargeStage::GatherScatter {
            label,
            src: remap_child_range(src, child_input, child_output, stage_index_base)?,
            dst: remap_child_range(dst, child_input, child_output, stage_index_base)?,
            stride_elements,
        },
        LargeStage::HelperWindow { label, range } => LargeStage::HelperWindow {
            label,
            range: remap_child_range(range, child_input, child_output, stage_index_base)?,
        },
        LargeStage::WindowedHelper { label, range } => LargeStage::WindowedHelper {
            label,
            range: remap_child_range(range, child_input, child_output, stage_index_base)?,
        },
        LargeStage::Kernel {
            label,
            input,
            output,
            work_items,
        } => LargeStage::Kernel {
            label,
            input: remap_child_range(input, child_input, child_output, stage_index_base)?,
            output: remap_child_range(output, child_input, child_output, stage_index_base)?,
            work_items,
        },
        LargeStage::WindowedKernel {
            label,
            input,
            output,
            work_items,
        } => LargeStage::WindowedKernel {
            label,
            input: remap_child_range(input, child_input, child_output, stage_index_base)?,
            output: remap_child_range(output, child_input, child_output, stage_index_base)?,
            work_items,
        },
        LargeStage::TwiddleTranspose {
            label,
            input,
            output,
            work_items,
        } => LargeStage::TwiddleTranspose {
            label,
            input: remap_child_range(input, child_input, child_output, stage_index_base)?,
            output: remap_child_range(output, child_input, child_output, stage_index_base)?,
            work_items,
        },
        LargeStage::StripeTranspose {
            label,
            input,
            output,
            work_items,
        } => LargeStage::StripeTranspose {
            label,
            input: remap_child_range(input, child_input, child_output, stage_index_base)?,
            output: remap_child_range(output, child_input, child_output, stage_index_base)?,
            work_items,
        },
        LargeStage::Permutation {
            label,
            input,
            output,
            work_items,
        } => LargeStage::Permutation {
            label,
            input: remap_child_range(input, child_input, child_output, stage_index_base)?,
            output: remap_child_range(output, child_input, child_output, stage_index_base)?,
            work_items,
        },
        LargeStage::Scale {
            label,
            range,
            work_items,
        } => LargeStage::Scale {
            label,
            range: remap_child_range(range, child_input, child_output, stage_index_base)?,
            work_items,
        },
        LargeStage::HostWindow { label, range } => LargeStage::HostWindow {
            label,
            range: remap_child_range(range, child_input, child_output, stage_index_base)?,
        },
    })
}

fn remap_child_range(
    range: LogicalRange,
    child_input: LogicalRange,
    child_output: LogicalRange,
    stage_index_base: u32,
) -> Result<LogicalRange> {
    let (buffer, base_offset) = match range.buffer {
        LogicalBufferId::Input => (child_input.buffer, child_input.offset_bytes),
        LogicalBufferId::Output => (child_output.buffer, child_output.offset_bytes),
        LogicalBufferId::Temp(index) => (LogicalBufferId::Temp(index), 0),
        LogicalBufferId::Stage(index) => (
            LogicalBufferId::Stage(
                index
                    .checked_add(stage_index_base)
                    .ok_or(FftError::LengthTooLarge { len: usize::MAX })?,
            ),
            0,
        ),
    };
    let offset_bytes = base_offset
        .checked_add(range.offset_bytes)
        .ok_or(FftError::LengthTooLarge { len: usize::MAX })?;
    real_range(buffer, offset_bytes, range.size_bytes, range.format)
}

fn max_stage_range_bytes(stage: &LargeStage) -> u64 {
    stage
        .ranges()
        .into_iter()
        .map(|range| range.size_bytes)
        .max()
        .unwrap_or(0)
}

fn stage_scratch_bytes(stage: &LargeStage) -> u64 {
    match stage {
        LargeStage::HelperWindow { range, .. }
        | LargeStage::WindowedHelper { range, .. }
        | LargeStage::Scale { range, .. }
        | LargeStage::HostWindow { range, .. } => range.size_bytes,
        LargeStage::Kernel { input, output, .. }
        | LargeStage::WindowedKernel { input, output, .. }
        | LargeStage::TwiddleTranspose { input, output, .. }
        | LargeStage::StripeTranspose { input, output, .. }
        | LargeStage::Permutation { input, output, .. } => input.size_bytes.max(output.size_bytes),
        LargeStage::Copy { src, dst, .. } | LargeStage::GatherScatter { src, dst, .. } => {
            src.size_bytes.max(dst.size_bytes)
        }
    }
}

fn build_r2c_chunk_graph(
    chunk: RealLargeChunkPlan,
    child_graph: &LargeExecutionGraph,
    limits: LargePolicyLimits,
) -> Result<LargeExecutionGraph> {
    let mut graph = LargeExecutionGraph::new("r2c-large-chunk");
    for range in chunk.ranges() {
        let range = range?;
        graph.push_stage(
            LargeStage::Copy {
                label: "r2c-large-chunk-copy-input",
                src: real_range(
                    LogicalBufferId::Input,
                    range.real_offset,
                    range.real_size,
                    ElementFormat::RealF32,
                )?,
                dst: real_range(
                    LogicalBufferId::Stage(0),
                    0,
                    range.real_size,
                    ElementFormat::RealF32,
                )?,
            },
            graph_requirements(limits, range.real_size)?,
        )?;
        append_child_graph(
            &mut graph,
            child_graph,
            real_range(
                LogicalBufferId::Stage(0),
                0,
                chunk.real_staging_size_bytes,
                ElementFormat::RealF32,
            )?,
            real_range(
                LogicalBufferId::Stage(1),
                0,
                chunk.packed_staging_size_bytes,
                ElementFormat::PackedComplexF32,
            )?,
            limits,
            2,
        )?;
        graph.push_stage(
            LargeStage::Copy {
                label: "r2c-large-chunk-copy-output",
                src: real_range(
                    LogicalBufferId::Stage(1),
                    0,
                    range.packed_size,
                    ElementFormat::PackedComplexF32,
                )?,
                dst: real_range(
                    LogicalBufferId::Output,
                    range.packed_offset,
                    range.packed_size,
                    ElementFormat::PackedComplexF32,
                )?,
            },
            graph_requirements(limits, range.packed_size)?,
        )?;
    }
    Ok(graph)
}

fn build_c2r_chunk_graph(
    chunk: RealLargeChunkPlan,
    child_graph: &LargeExecutionGraph,
    limits: LargePolicyLimits,
) -> Result<LargeExecutionGraph> {
    let mut graph = LargeExecutionGraph::new("c2r-large-chunk");
    for range in chunk.ranges() {
        let range = range?;
        graph.push_stage(
            LargeStage::Copy {
                label: "c2r-large-chunk-copy-input",
                src: real_range(
                    LogicalBufferId::Input,
                    range.packed_offset,
                    range.packed_size,
                    ElementFormat::PackedComplexF32,
                )?,
                dst: real_range(
                    LogicalBufferId::Stage(0),
                    0,
                    range.packed_size,
                    ElementFormat::PackedComplexF32,
                )?,
            },
            graph_requirements(limits, range.packed_size)?,
        )?;
        append_child_graph(
            &mut graph,
            child_graph,
            real_range(
                LogicalBufferId::Stage(0),
                0,
                chunk.packed_staging_size_bytes,
                ElementFormat::PackedComplexF32,
            )?,
            real_range(
                LogicalBufferId::Stage(1),
                0,
                chunk.real_staging_size_bytes,
                ElementFormat::RealF32,
            )?,
            limits,
            2,
        )?;
        graph.push_stage(
            LargeStage::Copy {
                label: "c2r-large-chunk-copy-output",
                src: real_range(
                    LogicalBufferId::Stage(1),
                    0,
                    range.real_size,
                    ElementFormat::RealF32,
                )?,
                dst: real_range(
                    LogicalBufferId::Output,
                    range.real_offset,
                    range.real_size,
                    ElementFormat::RealF32,
                )?,
            },
            graph_requirements(limits, range.real_size)?,
        )?;
    }
    Ok(graph)
}

impl R2cPlan {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, config: FftConfig) -> Result<Self> {
        Self::new_with_large_policy_limits(device, queue, config, None)
    }

    pub(crate) fn new_with_large_policy_limits_for_testing(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        config: FftConfig,
        limits: LargePolicyLimits,
    ) -> Result<Self> {
        let tuning = config
            .tuning()
            .clone()
            .with_max_storage_buffer_binding_size(limits.max_storage_buffer_binding_size)
            .with_max_buffer_size(limits.max_buffer_size);
        Self::new_with_large_policy_limits(device, queue, config.with_tuning(tuning), None)
    }

    fn new_with_large_policy_limits(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        config: FftConfig,
        policy_limits: Option<LargePolicyLimits>,
    ) -> Result<Self> {
        validate_real_config(&config, RealTransform::R2c)?;
        validate_real_tuning(device, &config)?;
        validate_real_precision(device, &config, "r2c")?;
        let packed_shape = packed_shape_for(config.shape())?;
        let sizes = RealPlanSizes::new(&config, &packed_shape)?;
        let axis_kinds = resolve_axis_kinds_for_config(&config)?;
        let mut large_routing_policy = resolve_real_large_routing_policy(
            device,
            &config,
            &packed_shape,
            &sizes,
            policy_limits,
        )?;
        let execution = match large_routing_policy.route_mode() {
            LargeRouteMode::Normal => R2cExecution::Normal(R2cNormalPlan::new(
                device,
                queue,
                &config,
                &packed_shape,
                &sizes,
                policy_limits,
            )?),
            LargeRouteMode::LargeChunk if config.batch() == 1 => {
                let limits = effective_real_policy_limits(device, &config, policy_limits);
                let plan = R2cNormalPlan::new(
                    device,
                    queue,
                    &config,
                    &packed_shape,
                    &sizes,
                    Some(limits),
                )?;
                let diagnostics = plan.c2c.large_routing_policy().diagnostics();
                large_routing_policy = large_routing_policy
                    .with_route_mode(LargeRouteMode::LargeChunk)
                    .with_execution_kind(diagnostics.execution_kind)
                    .with_diagnostics(
                        diagnostics.selected_axis,
                        diagnostics.factor_splits,
                        diagnostics.staging_bytes,
                        diagnostics.unsupported_reason,
                    );
                R2cExecution::LargeDecomposition(plan)
            }
            LargeRouteMode::LargeChunk => {
                let limits = effective_real_policy_limits(device, &config, policy_limits);
                match R2cLargeChunkPlan::new(device, queue, &config, &packed_shape, &sizes, limits)
                {
                    Ok(plan) => R2cExecution::LargeChunk(plan),
                    Err(_err) => {
                        let plan = R2cNormalPlan::new(
                            device,
                            queue,
                            &config,
                            &packed_shape,
                            &sizes,
                            Some(limits),
                        )?;
                        let diagnostics = plan.c2c.large_routing_policy().diagnostics();
                        large_routing_policy = large_routing_policy
                            .with_route_mode(LargeRouteMode::LargeChunk)
                            .with_execution_kind(diagnostics.execution_kind)
                            .with_diagnostics(
                                diagnostics.selected_axis,
                                diagnostics.factor_splits,
                                diagnostics.staging_bytes,
                                diagnostics.unsupported_reason,
                            );
                        R2cExecution::LargeDecomposition(plan)
                    }
                }
            }
            LargeRouteMode::LargeOutOfCore => {
                let limits = effective_real_policy_limits(device, &config, policy_limits);
                let plan = R2cNormalPlan::new(
                    device,
                    queue,
                    &config,
                    &packed_shape,
                    &sizes,
                    Some(limits),
                )?;
                let diagnostics = plan.c2c.large_routing_policy().diagnostics();
                large_routing_policy = large_routing_policy
                    .with_route_mode(LargeRouteMode::LargeChunk)
                    .with_execution_kind(diagnostics.execution_kind)
                    .with_diagnostics(
                        diagnostics.selected_axis,
                        diagnostics.factor_splits,
                        diagnostics.staging_bytes,
                        diagnostics.unsupported_reason,
                    );
                R2cExecution::LargeDecomposition(plan)
            }
        };

        Ok(Self {
            config,
            packed_shape,
            sizes,
            axis_kinds,
            large_routing_policy,
            execution,
        })
    }

    pub fn config(&self) -> FftConfig {
        self.config.clone()
    }

    pub fn packed_shape(&self) -> &[usize] {
        &self.packed_shape
    }

    pub fn factors(&self) -> &[usize] {
        match &self.execution {
            R2cExecution::Normal(plan) => plan.c2c.factors(),
            R2cExecution::LargeChunk(plan) => plan.child.factors(),
            R2cExecution::LargeDecomposition(plan) => plan.c2c.factors(),
        }
    }

    pub fn axis_factors(&self) -> &[Vec<usize>] {
        match &self.execution {
            R2cExecution::Normal(plan) => plan.c2c.axis_factors(),
            R2cExecution::LargeChunk(plan) => plan.child.axis_factors(),
            R2cExecution::LargeDecomposition(plan) => plan.c2c.axis_factors(),
        }
    }

    pub fn axis_kinds(&self) -> &[AxisKind] {
        &self.axis_kinds
    }

    pub fn route(&self) -> C2cRoute {
        match &self.execution {
            R2cExecution::Normal(plan) => plan.c2c.route(),
            R2cExecution::LargeChunk(plan) => plan.child.route(),
            R2cExecution::LargeDecomposition(plan) => plan.c2c.route(),
        }
    }

    pub fn large_routing_policy(&self) -> &LargeRoutingPolicy {
        &self.large_routing_policy
    }

    pub fn workspace_size_bytes(&self) -> u64 {
        0
    }

    pub(crate) fn twiddle_lut_storage_bytes(&self) -> u64 {
        match &self.execution {
            R2cExecution::Normal(plan) | R2cExecution::LargeDecomposition(plan) => {
                plan.c2c.twiddle_lut_storage_bytes()
            }
            R2cExecution::LargeChunk(plan) => plan.child.twiddle_lut_storage_bytes(),
        }
    }

    pub fn required_input_buffer_size_bytes(&self) -> u64 {
        self.sizes.real_bytes
    }

    pub fn required_output_buffer_size_bytes(&self) -> u64 {
        self.sizes.packed_bytes
    }

    pub fn required_buffer_size_bytes(&self) -> u64 {
        self.sizes.real_bytes.max(self.sizes.packed_bytes)
    }

    pub(crate) fn execution_graph(&self) -> Result<LargeExecutionGraph> {
        match &self.execution {
            R2cExecution::Normal(plan) => {
                let child_graph = plan.c2c.execution_graph()?;
                build_real_normal_graph(
                    "r2c-normal",
                    "r2c-real-to-complex",
                    "r2c-pack",
                    &child_graph,
                    RealEndpointOrder::RealToPacked,
                    &self.sizes,
                    policy_limits(&self.large_routing_policy),
                    false,
                )
            }
            R2cExecution::LargeDecomposition(plan) => {
                let child_graph = plan.c2c.execution_graph()?;
                build_real_normal_graph(
                    "r2c-large-decomposition",
                    "r2c-windowed-real-to-complex",
                    "r2c-windowed-pack",
                    &child_graph,
                    RealEndpointOrder::RealToPacked,
                    &self.sizes,
                    policy_limits(&self.large_routing_policy),
                    true,
                )
            }
            R2cExecution::LargeChunk(plan) => {
                let child_graph = plan.child.execution_graph()?;
                build_r2c_chunk_graph(
                    plan.chunk,
                    &child_graph,
                    policy_limits(&self.large_routing_policy),
                )
            }
        }
    }

    pub fn execute_views(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        input: BufferView<'_>,
        output: BufferView<'_>,
    ) -> Result<()> {
        self.execute_views_recorded(device, &mut CommandRecorder::new(encoder), input, output)
    }

    /// [`Self::execute_views`] for plans nested in another execution, which
    /// share the caller's compute pass.
    pub(crate) fn execute_views_recorded(
        &self,
        device: &wgpu::Device,
        encoder: &mut CommandRecorder<'_>,
        input: BufferView<'_>,
        output: BufferView<'_>,
    ) -> Result<()> {
        self.validate_execution_graph(device)?;
        match &self.execution {
            R2cExecution::Normal(plan) => {
                plan.execute_views(device, encoder, &self.sizes, input, output)
            }
            R2cExecution::LargeChunk(plan) => plan.execute_views(device, encoder, input, output),
            R2cExecution::LargeDecomposition(plan) => {
                plan.execute_views_large_decomposition(device, encoder, &self.sizes, input, output)
            }
        }
    }

    pub fn execute_views_with_workspace(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        input: BufferView<'_>,
        output: BufferView<'_>,
        _workspace: BufferView<'_>,
    ) -> Result<()> {
        self.execute_views(device, encoder, input, output)
    }

    pub fn execute_logical_views(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        input: FftLogicalView<'_>,
        output: FftLogicalView<'_>,
    ) -> Result<()> {
        execute_r2c_logical_views(
            device,
            &mut CommandRecorder::new(encoder),
            self,
            input,
            output,
        )
    }

    fn validate_execution_graph(&self, device: &wgpu::Device) -> Result<()> {
        let graph = self.execution_graph()?;
        let scheduler = WindowScheduler::for_device(device);
        StageExecutor::new(&scheduler).validate_graph(&graph)
    }
}

impl C2rPlan {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, config: FftConfig) -> Result<Self> {
        Self::new_with_large_policy_limits(device, queue, config, None)
    }

    pub(crate) fn new_with_large_policy_limits_for_testing(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        config: FftConfig,
        limits: LargePolicyLimits,
    ) -> Result<Self> {
        let tuning = config
            .tuning()
            .clone()
            .with_max_storage_buffer_binding_size(limits.max_storage_buffer_binding_size)
            .with_max_buffer_size(limits.max_buffer_size);
        Self::new_with_large_policy_limits(device, queue, config.with_tuning(tuning), None)
    }

    fn new_with_large_policy_limits(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        config: FftConfig,
        policy_limits: Option<LargePolicyLimits>,
    ) -> Result<Self> {
        validate_real_config(&config, RealTransform::C2r)?;
        validate_real_tuning(device, &config)?;
        validate_real_precision(device, &config, "c2r")?;
        let packed_shape = packed_shape_for(config.shape())?;
        let sizes = RealPlanSizes::new(&config, &packed_shape)?;
        let axis_kinds = resolve_axis_kinds_for_config(&config)?;
        let mut large_routing_policy = resolve_real_large_routing_policy(
            device,
            &config,
            &packed_shape,
            &sizes,
            policy_limits,
        )?;
        let execution = match large_routing_policy.route_mode() {
            LargeRouteMode::Normal => C2rExecution::Normal(C2rNormalPlan::new(
                device,
                queue,
                &config,
                &packed_shape,
                &sizes,
                policy_limits,
            )?),
            LargeRouteMode::LargeChunk if config.batch() == 1 => {
                let limits = effective_real_policy_limits(device, &config, policy_limits);
                let plan = C2rNormalPlan::new(
                    device,
                    queue,
                    &config,
                    &packed_shape,
                    &sizes,
                    Some(limits),
                )?;
                let diagnostics = plan.c2c.large_routing_policy().diagnostics();
                large_routing_policy = large_routing_policy
                    .with_route_mode(LargeRouteMode::LargeChunk)
                    .with_execution_kind(diagnostics.execution_kind)
                    .with_diagnostics(
                        diagnostics.selected_axis,
                        diagnostics.factor_splits,
                        diagnostics.staging_bytes,
                        diagnostics.unsupported_reason,
                    );
                C2rExecution::LargeDecomposition(plan)
            }
            LargeRouteMode::LargeChunk => {
                let limits = effective_real_policy_limits(device, &config, policy_limits);
                match C2rLargeChunkPlan::new(device, queue, &config, &packed_shape, &sizes, limits)
                {
                    Ok(plan) => C2rExecution::LargeChunk(plan),
                    Err(_err) => {
                        let plan = C2rNormalPlan::new(
                            device,
                            queue,
                            &config,
                            &packed_shape,
                            &sizes,
                            Some(limits),
                        )?;
                        let diagnostics = plan.c2c.large_routing_policy().diagnostics();
                        large_routing_policy = large_routing_policy
                            .with_execution_kind(diagnostics.execution_kind)
                            .with_diagnostics(
                                diagnostics.selected_axis,
                                diagnostics.factor_splits,
                                diagnostics.staging_bytes,
                                diagnostics.unsupported_reason,
                            );
                        C2rExecution::LargeDecomposition(plan)
                    }
                }
            }
            LargeRouteMode::LargeOutOfCore => {
                let limits = effective_real_policy_limits(device, &config, policy_limits);
                let plan = C2rNormalPlan::new(
                    device,
                    queue,
                    &config,
                    &packed_shape,
                    &sizes,
                    Some(limits),
                )?;
                let diagnostics = plan.c2c.large_routing_policy().diagnostics();
                large_routing_policy = large_routing_policy
                    .with_route_mode(LargeRouteMode::LargeChunk)
                    .with_execution_kind(diagnostics.execution_kind)
                    .with_diagnostics(
                        diagnostics.selected_axis,
                        diagnostics.factor_splits,
                        diagnostics.staging_bytes,
                        diagnostics.unsupported_reason,
                    );
                C2rExecution::LargeDecomposition(plan)
            }
        };

        Ok(Self {
            config,
            packed_shape,
            sizes,
            axis_kinds,
            large_routing_policy,
            execution,
        })
    }

    pub fn config(&self) -> FftConfig {
        self.config.clone()
    }

    pub fn packed_shape(&self) -> &[usize] {
        &self.packed_shape
    }

    pub fn factors(&self) -> &[usize] {
        match &self.execution {
            C2rExecution::Normal(plan) => plan.c2c.factors(),
            C2rExecution::LargeChunk(plan) => plan.child.factors(),
            C2rExecution::LargeDecomposition(plan) => plan.c2c.factors(),
        }
    }

    pub fn axis_factors(&self) -> &[Vec<usize>] {
        match &self.execution {
            C2rExecution::Normal(plan) => plan.c2c.axis_factors(),
            C2rExecution::LargeChunk(plan) => plan.child.axis_factors(),
            C2rExecution::LargeDecomposition(plan) => plan.c2c.axis_factors(),
        }
    }

    pub fn axis_kinds(&self) -> &[AxisKind] {
        &self.axis_kinds
    }

    pub fn route(&self) -> C2cRoute {
        match &self.execution {
            C2rExecution::Normal(plan) => plan.c2c.route(),
            C2rExecution::LargeChunk(plan) => plan.child.route(),
            C2rExecution::LargeDecomposition(plan) => plan.c2c.route(),
        }
    }

    pub fn large_routing_policy(&self) -> &LargeRoutingPolicy {
        &self.large_routing_policy
    }

    pub fn workspace_size_bytes(&self) -> u64 {
        0
    }

    pub(crate) fn twiddle_lut_storage_bytes(&self) -> u64 {
        match &self.execution {
            C2rExecution::Normal(plan) | C2rExecution::LargeDecomposition(plan) => {
                plan.c2c.twiddle_lut_storage_bytes()
            }
            C2rExecution::LargeChunk(plan) => plan.child.twiddle_lut_storage_bytes(),
        }
    }

    pub fn required_input_buffer_size_bytes(&self) -> u64 {
        self.sizes.packed_bytes
    }

    pub fn required_output_buffer_size_bytes(&self) -> u64 {
        self.sizes.real_bytes
    }

    pub fn required_buffer_size_bytes(&self) -> u64 {
        self.sizes.real_bytes.max(self.sizes.packed_bytes)
    }

    pub(crate) fn execution_graph(&self) -> Result<LargeExecutionGraph> {
        match &self.execution {
            C2rExecution::Normal(plan) => {
                let child_graph = plan.c2c.execution_graph()?;
                build_real_normal_graph(
                    "c2r-normal",
                    "c2r-unpack",
                    "c2r-complex-to-real",
                    &child_graph,
                    RealEndpointOrder::PackedToReal,
                    &self.sizes,
                    policy_limits(&self.large_routing_policy),
                    false,
                )
            }
            C2rExecution::LargeDecomposition(plan) => {
                let child_graph = plan.c2c.execution_graph()?;
                build_real_normal_graph(
                    "c2r-large-decomposition",
                    "c2r-windowed-unpack",
                    "c2r-windowed-complex-to-real",
                    &child_graph,
                    RealEndpointOrder::PackedToReal,
                    &self.sizes,
                    policy_limits(&self.large_routing_policy),
                    true,
                )
            }
            C2rExecution::LargeChunk(plan) => {
                let child_graph = plan.child.execution_graph()?;
                build_c2r_chunk_graph(
                    plan.chunk,
                    &child_graph,
                    policy_limits(&self.large_routing_policy),
                )
            }
        }
    }

    pub fn execute_views(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        input: BufferView<'_>,
        output: BufferView<'_>,
    ) -> Result<()> {
        self.execute_views_recorded(device, &mut CommandRecorder::new(encoder), input, output)
    }

    /// [`Self::execute_views`] for plans nested in another execution, which
    /// share the caller's compute pass.
    pub(crate) fn execute_views_recorded(
        &self,
        device: &wgpu::Device,
        encoder: &mut CommandRecorder<'_>,
        input: BufferView<'_>,
        output: BufferView<'_>,
    ) -> Result<()> {
        self.validate_execution_graph(device)?;
        match &self.execution {
            C2rExecution::Normal(plan) => {
                plan.execute_views(device, encoder, &self.sizes, input, output)
            }
            C2rExecution::LargeChunk(plan) => plan.execute_views(device, encoder, input, output),
            C2rExecution::LargeDecomposition(plan) => {
                plan.execute_views_large_decomposition(device, encoder, &self.sizes, input, output)
            }
        }
    }

    pub fn execute_views_with_workspace(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        input: BufferView<'_>,
        output: BufferView<'_>,
        _workspace: BufferView<'_>,
    ) -> Result<()> {
        self.execute_views(device, encoder, input, output)
    }

    pub fn execute_logical_views(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        input: FftLogicalView<'_>,
        output: FftLogicalView<'_>,
    ) -> Result<()> {
        execute_c2r_logical_views(
            device,
            &mut CommandRecorder::new(encoder),
            self,
            input,
            output,
        )
    }

    fn validate_execution_graph(&self, device: &wgpu::Device) -> Result<()> {
        let graph = self.execution_graph()?;
        let scheduler = WindowScheduler::for_device(device);
        StageExecutor::new(&scheduler).validate_graph(&graph)
    }
}

fn execute_r2c_logical_views(
    device: &wgpu::Device,
    encoder: &mut CommandRecorder<'_>,
    plan: &R2cPlan,
    input: FftLogicalView<'_>,
    output: FftLogicalView<'_>,
) -> Result<()> {
    let real_per_batch = plan.config.logical_complex_len()? as u64;
    let packed_per_batch = packed_total_complex_len(&plan.packed_shape, 1)? as u64;
    let scheduler = WindowScheduler::for_device(device);
    let input = scheduler.bind_logical_io(
        input,
        FftEndpointFormat::RealF32,
        real_per_batch,
        plan.config.batch() as u64,
    )?;
    let output = scheduler.bind_logical_io(
        output,
        FftEndpointFormat::PackedComplexF32,
        packed_per_batch,
        plan.config.batch() as u64,
    )?;
    execute_real_logical_views(
        device,
        encoder,
        &input,
        &output,
        plan.sizes.real_bytes,
        plan.sizes.packed_bytes,
        plan.config.tuning().workgroup_size(),
        |device, encoder, input, output| {
            plan.execute_views_recorded(device, encoder, input, output)
        },
    )
}

fn execute_c2r_logical_views(
    device: &wgpu::Device,
    encoder: &mut CommandRecorder<'_>,
    plan: &C2rPlan,
    input: FftLogicalView<'_>,
    output: FftLogicalView<'_>,
) -> Result<()> {
    let packed_per_batch = packed_total_complex_len(&plan.packed_shape, 1)? as u64;
    let real_per_batch = plan.config.logical_complex_len()? as u64;
    let scheduler = WindowScheduler::for_device(device);
    let input = scheduler.bind_logical_io(
        input,
        FftEndpointFormat::PackedComplexF32,
        packed_per_batch,
        plan.config.batch() as u64,
    )?;
    let output = scheduler.bind_logical_io(
        output,
        FftEndpointFormat::RealF32,
        real_per_batch,
        plan.config.batch() as u64,
    )?;
    execute_real_logical_views(
        device,
        encoder,
        &input,
        &output,
        plan.sizes.packed_bytes,
        plan.sizes.real_bytes,
        plan.config.tuning().workgroup_size(),
        |device, encoder, input, output| {
            plan.execute_views_recorded(device, encoder, input, output)
        },
    )
}

fn execute_real_logical_views(
    device: &wgpu::Device,
    encoder: &mut CommandRecorder<'_>,
    input: &BoundLogicalIo<'_>,
    output: &BoundLogicalIo<'_>,
    required_input_bytes: u64,
    required_output_bytes: u64,
    workgroup_size: u32,
    execute: impl for<'i, 'o> FnOnce(
        &wgpu::Device,
        &mut CommandRecorder<'_>,
        BufferView<'i>,
        BufferView<'o>,
    ) -> Result<()>,
) -> Result<()> {
    let input_physical_stage = if !input.is_contiguous() && input.is_segmented() {
        let buffer = create_internal_buffer(
            device,
            "wgpu_fft.real.logical.segmented_strided_input_physical_stage",
            input.physical_span_bytes,
            wgpu::BufferUsages::COPY_DST,
        )?;
        copy_view_range_to_buffer(
            device,
            encoder,
            &input.view,
            0,
            &buffer,
            0,
            input.physical_span_bytes,
        )?;
        Some(buffer)
    } else {
        None
    };
    let input_logical_stage = if input.is_contiguous() {
        None
    } else {
        let buffer = create_internal_buffer(
            device,
            "wgpu_fft.real.logical.strided_input_stage",
            required_input_bytes,
            wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        )?;
        let strided_source = if let Some(buffer) = input_physical_stage.as_ref() {
            BufferView::whole(buffer).prefix(input.physical_span_bytes)?
        } else {
            input.view_prefix(input.physical_span_bytes)?
        };
        dispatch_real_strided_copy(
            device,
            encoder,
            strided_pack_kind(input.format)?,
            input.format,
            &strided_source,
            &BufferView::whole(&buffer).prefix(required_input_bytes)?,
            input.layout,
            input.logical_elements_per_batch,
            input.batch,
            workgroup_size,
        )?;
        Some(buffer)
    };

    let output_logical_stage = if output.is_contiguous() {
        None
    } else {
        Some(create_internal_buffer(
            device,
            "wgpu_fft.real.logical.strided_output_stage",
            required_output_bytes,
            wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        )?)
    };
    let output_physical_stage = if !output.is_contiguous() && output.is_segmented() {
        Some(create_internal_buffer(
            device,
            "wgpu_fft.real.logical.segmented_strided_output_physical_stage",
            output.physical_span_bytes,
            wgpu::BufferUsages::COPY_SRC,
        )?)
    } else {
        None
    };

    let exec_input = if let Some(buffer) = input_logical_stage.as_ref() {
        BufferView::whole(buffer).prefix(required_input_bytes)?
    } else {
        input.view_prefix(required_input_bytes)?
    };
    let exec_output = if let Some(buffer) = output_logical_stage.as_ref() {
        BufferView::whole(buffer).prefix(required_output_bytes)?
    } else {
        output.view_prefix(required_output_bytes)?
    };

    execute(device, encoder, exec_input, exec_output)?;

    if let Some(buffer) = output_logical_stage.as_ref() {
        let strided_target = if let Some(physical) = output_physical_stage.as_ref() {
            BufferView::whole(physical).prefix(output.physical_span_bytes)?
        } else {
            output.view_prefix(output.physical_span_bytes)?
        };
        dispatch_real_strided_copy(
            device,
            encoder,
            strided_unpack_kind(output.format)?,
            output.format,
            &BufferView::whole(buffer).prefix(required_output_bytes)?,
            &strided_target,
            output.layout,
            output.logical_elements_per_batch,
            output.batch,
            workgroup_size,
        )?;
        if let Some(physical) = output_physical_stage.as_ref() {
            copy_buffer_to_view_range(
                device,
                encoder,
                physical,
                0,
                &output.view,
                0,
                output.physical_span_bytes,
            )?;
        }
    }

    Ok(())
}

fn strided_pack_kind(format: FftEndpointFormat) -> Result<RealKernelKind> {
    Ok(match format {
        FftEndpointFormat::RealF32 => RealKernelKind::PackRealStrided,
        FftEndpointFormat::ComplexF32 | FftEndpointFormat::PackedComplexF32 => {
            RealKernelKind::PackComplexStrided
        }
        FftEndpointFormat::ComplexF64 => {
            return Err(FftError::PrecisionUnsupported {
                requested: crate::config::FftPrecision::F64,
                route: "real",
                reason: "real-f64-not-implemented",
            });
        }
        FftEndpointFormat::ComplexDf64 => {
            return Err(FftError::PrecisionUnsupported {
                requested: crate::config::FftPrecision::Df64,
                route: "real",
                reason: "real-df64-not-implemented",
            });
        }
    })
}

fn strided_unpack_kind(format: FftEndpointFormat) -> Result<RealKernelKind> {
    Ok(match format {
        FftEndpointFormat::RealF32 => RealKernelKind::UnpackRealStrided,
        FftEndpointFormat::ComplexF32 | FftEndpointFormat::PackedComplexF32 => {
            RealKernelKind::UnpackComplexStrided
        }
        FftEndpointFormat::ComplexF64 => {
            return Err(FftError::PrecisionUnsupported {
                requested: crate::config::FftPrecision::F64,
                route: "real",
                reason: "real-f64-not-implemented",
            });
        }
        FftEndpointFormat::ComplexDf64 => {
            return Err(FftError::PrecisionUnsupported {
                requested: crate::config::FftPrecision::Df64,
                route: "real",
                reason: "real-df64-not-implemented",
            });
        }
    })
}

fn real_strided_kind_matches_format(kind: RealKernelKind, format: FftEndpointFormat) -> bool {
    matches!(
        (kind, format),
        (
            RealKernelKind::PackRealStrided | RealKernelKind::UnpackRealStrided,
            FftEndpointFormat::RealF32,
        ) | (
            RealKernelKind::PackComplexStrided | RealKernelKind::UnpackComplexStrided,
            FftEndpointFormat::ComplexF32 | FftEndpointFormat::PackedComplexF32,
        )
    )
}

fn validate_real_strided_kind(kind: RealKernelKind, format: FftEndpointFormat) -> Result<()> {
    if real_strided_kind_matches_format(kind, format) {
        Ok(())
    } else {
        Err(FftError::LargeGraphStageUnsupported {
            stage: "real-strided-kernel-kind",
            reason: "real strided copy kernel kind does not match endpoint format",
        })
    }
}

fn real_shader_key_error(reason: &'static str) -> FftError {
    FftError::LargeGraphStageUnsupported {
        stage: "real-shader-key",
        reason,
    }
}

fn validate_real_stage_key(key: &RealStageKey) -> Result<()> {
    if key.rank != key.dims.len() {
        return Err(real_shader_key_error(
            "real shader key rank does not match dimensions",
        ));
    }
    if key.workgroup_size == 0 {
        return Err(real_shader_key_error(
            "real shader key workgroup size must be non-zero",
        ));
    }
    match key.kind {
        RealKernelKind::PackR2c
        | RealKernelKind::PackR2cWindowed
        | RealKernelKind::UnpackC2r
        | RealKernelKind::UnpackC2rWindowed => {
            if key.dims.is_empty() {
                return Err(real_shader_key_error(
                    "real pack/unpack shader key requires a non-empty shape",
                ));
            }
            if key.dims.iter().any(|&dim| dim == 0) {
                return Err(real_shader_key_error(
                    "real pack/unpack shader key dimensions must be non-zero",
                ));
            }
        }
        RealKernelKind::PackRealStrided
        | RealKernelKind::UnpackRealStrided
        | RealKernelKind::PackComplexStrided
        | RealKernelKind::UnpackComplexStrided => {
            if !key.dims.is_empty() {
                return Err(real_shader_key_error(
                    "real strided shader key must not carry transform dimensions",
                ));
            }
        }
        RealKernelKind::RealToComplex
        | RealKernelKind::RealToComplexWindowed
        | RealKernelKind::ComplexToReal
        | RealKernelKind::ComplexToRealWindowed => {}
    }
    Ok(())
}

impl R2cNormalPlan {
    fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        config: &FftConfig,
        packed_shape: &[usize],
        sizes: &RealPlanSizes,
        policy_limits: Option<LargePolicyLimits>,
    ) -> Result<Self> {
        if let Some(limits) = policy_limits {
            if sizes.full_complex_bytes > limits.max_buffer_size {
                return Err(FftError::LargeChunkUnsupported {
                    reason: "real large decomposition requires a full-complex staging buffer in V1",
                    bytes_per_batch: sizes.full_complex_bytes,
                    max_bind_bytes: limits.max_storage_buffer_binding_size,
                });
            }
        }
        let c2c = if let Some(limits) = policy_limits {
            C2cPlan::new_with_large_policy_limits_for_testing(
                device,
                queue,
                config.clone(),
                limits,
            )?
        } else {
            C2cPlan::new(device, queue, config.clone())?
        };
        let full_input_buffer = create_internal_buffer(
            device,
            "wgpu_fft.r2c.full_input",
            sizes.full_complex_bytes,
            wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        )?;
        let full_output_buffer = create_internal_buffer(
            device,
            "wgpu_fft.r2c.full_output",
            sizes.full_complex_bytes,
            wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        )?;
        let real_to_complex = RealKernel::new(
            device,
            queue,
            RealKernelKind::RealToComplex,
            config.shape(),
            config.total_complex_len_u32()?,
            config.total_complex_len_u32()?,
            config.tuning().workgroup_size(),
            "wgpu_fft.r2c.real_to_complex",
        )?;
        let pack = RealKernel::new(
            device,
            queue,
            RealKernelKind::PackR2c,
            config.shape(),
            config.batch() as u32,
            packed_total_complex_len(packed_shape, config.batch())? as u32,
            config.tuning().workgroup_size(),
            "wgpu_fft.r2c.pack",
        )?;

        Ok(Self {
            c2c,
            real_to_complex,
            pack,
            full_input_buffer,
            full_output_buffer,
        })
    }

    fn execute_views(
        &self,
        device: &wgpu::Device,
        encoder: &mut CommandRecorder<'_>,
        sizes: &RealPlanSizes,
        input: BufferView<'_>,
        output: BufferView<'_>,
    ) -> Result<()> {
        let input = input.prefix(sizes.real_bytes)?;
        let output = output.prefix(sizes.packed_bytes)?;

        let input_stage = if input.is_single_segment() {
            None
        } else {
            let buffer = create_internal_buffer(
                device,
                "wgpu_fft.r2c.segmented_real_input_stage",
                sizes.real_bytes,
                wgpu::BufferUsages::COPY_DST,
            )?;
            copy_view_range_to_buffer(device, encoder, &input, 0, &buffer, 0, sizes.real_bytes)?;
            Some(buffer)
        };
        let output_stage = if output.is_single_segment() {
            None
        } else {
            Some(create_internal_buffer(
                device,
                "wgpu_fft.r2c.segmented_packed_output_stage",
                sizes.packed_bytes,
                wgpu::BufferUsages::COPY_SRC,
            )?)
        };

        let real_input = if let Some(buffer) = input_stage.as_ref() {
            BufferView::whole(buffer).prefix(sizes.real_bytes)?
        } else {
            input.clone()
        };
        let packed_output = if let Some(buffer) = output_stage.as_ref() {
            BufferView::whole(buffer).prefix(sizes.packed_bytes)?
        } else {
            output.clone()
        };

        let full_input =
            BufferView::whole(&self.full_input_buffer).prefix(sizes.full_complex_bytes)?;
        let full_output =
            BufferView::whole(&self.full_output_buffer).prefix(sizes.full_complex_bytes)?;
        self.real_to_complex
            .execute(device, encoder, real_input, full_input.clone())?;
        self.c2c
            .execute_views_recorded(device, encoder, full_input, full_output.clone())?;
        self.pack
            .execute(device, encoder, full_output, packed_output)?;

        if let Some(buffer) = output_stage.as_ref() {
            copy_buffer_to_view_range(device, encoder, buffer, 0, &output, 0, sizes.packed_bytes)?;
        }
        Ok(())
    }

    fn execute_views_large_decomposition(
        &self,
        device: &wgpu::Device,
        encoder: &mut CommandRecorder<'_>,
        sizes: &RealPlanSizes,
        input: BufferView<'_>,
        output: BufferView<'_>,
    ) -> Result<()> {
        let input = input.prefix(sizes.real_bytes)?;
        let output = output.prefix(sizes.packed_bytes)?;
        let scheduler = WindowScheduler::for_device(device);
        let executor = StageExecutor::new(&scheduler);

        let input_stage = if input.is_single_segment() {
            None
        } else {
            let buffer = create_internal_buffer(
                device,
                "wgpu_fft.r2c.large_decompose.segmented_real_input_stage",
                sizes.real_bytes,
                wgpu::BufferUsages::COPY_DST,
            )?;
            executor.copy_view_range_to_buffer(encoder, &input, 0, &buffer, 0, sizes.real_bytes)?;
            Some(buffer)
        };
        let output_stage = if output.is_single_segment() {
            None
        } else {
            Some(create_internal_buffer(
                device,
                "wgpu_fft.r2c.large_decompose.segmented_packed_output_stage",
                sizes.packed_bytes,
                wgpu::BufferUsages::COPY_SRC,
            )?)
        };

        let real_input = if let Some(buffer) = input_stage.as_ref() {
            BufferView::whole(buffer).prefix(sizes.real_bytes)?
        } else {
            input.clone()
        };
        let packed_output = if let Some(buffer) = output_stage.as_ref() {
            BufferView::whole(buffer).prefix(sizes.packed_bytes)?
        } else {
            output.clone()
        };

        let full_input =
            BufferView::whole(&self.full_input_buffer).prefix(sizes.full_complex_bytes)?;
        let full_output =
            BufferView::whole(&self.full_output_buffer).prefix(sizes.full_complex_bytes)?;
        let config = self.c2c.config();
        let workgroup_size = config.tuning().workgroup_size();
        dispatch_real_to_complex_windowed(
            device,
            encoder,
            real_input,
            full_input.clone(),
            workgroup_size,
        )?;
        self.c2c
            .execute_views_recorded(device, encoder, full_input, full_output.clone())?;
        dispatch_pack_r2c_windowed(
            device,
            encoder,
            config.shape(),
            config.batch(),
            full_output,
            packed_output,
            workgroup_size,
        )?;

        if let Some(buffer) = output_stage.as_ref() {
            executor.copy_buffer_to_view_range(
                encoder,
                buffer,
                0,
                &output,
                0,
                sizes.packed_bytes,
            )?;
        }
        Ok(())
    }
}

impl R2cLargeChunkPlan {
    fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        config: &FftConfig,
        _packed_shape: &[usize],
        sizes: &RealPlanSizes,
        limits: LargePolicyLimits,
    ) -> Result<Self> {
        let chunk = RealLargeChunkPlan::new_with_max_batches(
            sizes,
            config.batch() as u64,
            limits,
            config.tuning().large_chunk_max_batches(),
        )?;
        let child_tuning = config
            .tuning()
            .clone()
            .with_large_route(FftLargeRoute::Auto);
        let child_config = config
            .clone()
            .with_batch(chunk.chunk_batch_count as usize)
            .with_tuning(child_tuning);
        let child = Box::new(R2cPlan::new_with_large_policy_limits(
            device,
            queue,
            child_config,
            Some(LargePolicyLimits {
                max_storage_buffer_binding_size: limits.max_storage_buffer_binding_size,
                max_buffer_size: limits.max_buffer_size,
            }),
        )?);
        if child.large_routing_policy().route_mode() != LargeRouteMode::Normal {
            return Err(FftError::LargeChunkUnsupported {
                reason: "real large-chunk child plan did not fit normal binding limits",
                bytes_per_batch: chunk.full_complex_bytes_per_batch,
                max_bind_bytes: limits.max_storage_buffer_binding_size,
            });
        }
        let input_stage = create_internal_buffer(
            device,
            "wgpu_fft.r2c.large_chunk.real_input_stage",
            chunk.real_staging_size_bytes,
            wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        )?;
        let output_stage = create_internal_buffer(
            device,
            "wgpu_fft.r2c.large_chunk.packed_output_stage",
            chunk.packed_staging_size_bytes,
            wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        )?;
        Ok(Self {
            chunk,
            child,
            input_stage,
            output_stage,
        })
    }

    fn execute_views(
        &self,
        device: &wgpu::Device,
        encoder: &mut CommandRecorder<'_>,
        input: BufferView<'_>,
        output: BufferView<'_>,
    ) -> Result<()> {
        let input = input.prefix(self.chunk.real_total_bytes()?)?;
        let output = output.prefix(self.chunk.packed_total_bytes()?)?;
        let child_input =
            BufferView::whole(&self.input_stage).prefix(self.chunk.real_staging_size_bytes)?;
        let child_output =
            BufferView::whole(&self.output_stage).prefix(self.chunk.packed_staging_size_bytes)?;

        for range in self.chunk.ranges() {
            let range = range?;
            copy_view_range_to_buffer(
                device,
                encoder,
                &input,
                range.real_offset,
                &self.input_stage,
                0,
                range.real_size,
            )?;
            self.child.execute_views_recorded(
                device,
                encoder,
                child_input.clone(),
                child_output.clone(),
            )?;
            copy_buffer_to_view_range(
                device,
                encoder,
                &self.output_stage,
                0,
                &output,
                range.packed_offset,
                range.packed_size,
            )?;
        }
        Ok(())
    }
}

impl C2rNormalPlan {
    fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        config: &FftConfig,
        _packed_shape: &[usize],
        sizes: &RealPlanSizes,
        policy_limits: Option<LargePolicyLimits>,
    ) -> Result<Self> {
        if let Some(limits) = policy_limits {
            if sizes.full_complex_bytes > limits.max_buffer_size {
                return Err(FftError::LargeChunkUnsupported {
                    reason: "real large decomposition requires a full-complex staging buffer in V1",
                    bytes_per_batch: sizes.full_complex_bytes,
                    max_bind_bytes: limits.max_storage_buffer_binding_size,
                });
            }
        }
        let c2c = if let Some(limits) = policy_limits {
            C2cPlan::new_with_large_policy_limits_for_testing(
                device,
                queue,
                config.clone(),
                limits,
            )?
        } else {
            C2cPlan::new(device, queue, config.clone())?
        };
        let full_input_buffer = create_internal_buffer(
            device,
            "wgpu_fft.c2r.full_input",
            sizes.full_complex_bytes,
            wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        )?;
        let full_output_buffer = create_internal_buffer(
            device,
            "wgpu_fft.c2r.full_output",
            sizes.full_complex_bytes,
            wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        )?;
        let unpack = RealKernel::new(
            device,
            queue,
            RealKernelKind::UnpackC2r,
            config.shape(),
            config.batch() as u32,
            config.total_complex_len_u32()?,
            config.tuning().workgroup_size(),
            "wgpu_fft.c2r.unpack",
        )?;
        let complex_to_real = RealKernel::new(
            device,
            queue,
            RealKernelKind::ComplexToReal,
            config.shape(),
            config.total_complex_len_u32()?,
            config.total_complex_len_u32()?,
            config.tuning().workgroup_size(),
            "wgpu_fft.c2r.complex_to_real",
        )?;

        Ok(Self {
            c2c,
            unpack,
            complex_to_real,
            full_input_buffer,
            full_output_buffer,
        })
    }

    fn execute_views(
        &self,
        device: &wgpu::Device,
        encoder: &mut CommandRecorder<'_>,
        sizes: &RealPlanSizes,
        input: BufferView<'_>,
        output: BufferView<'_>,
    ) -> Result<()> {
        let input = input.prefix(sizes.packed_bytes)?;
        let output = output.prefix(sizes.real_bytes)?;

        let input_stage = if input.is_single_segment() {
            None
        } else {
            let buffer = create_internal_buffer(
                device,
                "wgpu_fft.c2r.segmented_packed_input_stage",
                sizes.packed_bytes,
                wgpu::BufferUsages::COPY_DST,
            )?;
            copy_view_range_to_buffer(device, encoder, &input, 0, &buffer, 0, sizes.packed_bytes)?;
            Some(buffer)
        };
        let output_stage = if output.is_single_segment() {
            None
        } else {
            Some(create_internal_buffer(
                device,
                "wgpu_fft.c2r.segmented_real_output_stage",
                sizes.real_bytes,
                wgpu::BufferUsages::COPY_SRC,
            )?)
        };

        let packed_input = if let Some(buffer) = input_stage.as_ref() {
            BufferView::whole(buffer).prefix(sizes.packed_bytes)?
        } else {
            input.clone()
        };
        let real_output = if let Some(buffer) = output_stage.as_ref() {
            BufferView::whole(buffer).prefix(sizes.real_bytes)?
        } else {
            output.clone()
        };

        let full_input =
            BufferView::whole(&self.full_input_buffer).prefix(sizes.full_complex_bytes)?;
        let full_output =
            BufferView::whole(&self.full_output_buffer).prefix(sizes.full_complex_bytes)?;
        self.unpack
            .execute(device, encoder, packed_input, full_input.clone())?;
        self.c2c
            .execute_views_recorded(device, encoder, full_input, full_output.clone())?;
        self.complex_to_real
            .execute(device, encoder, full_output, real_output)?;

        if let Some(buffer) = output_stage.as_ref() {
            copy_buffer_to_view_range(device, encoder, buffer, 0, &output, 0, sizes.real_bytes)?;
        }
        Ok(())
    }

    fn execute_views_large_decomposition(
        &self,
        device: &wgpu::Device,
        encoder: &mut CommandRecorder<'_>,
        sizes: &RealPlanSizes,
        input: BufferView<'_>,
        output: BufferView<'_>,
    ) -> Result<()> {
        let input = input.prefix(sizes.packed_bytes)?;
        let output = output.prefix(sizes.real_bytes)?;
        let scheduler = WindowScheduler::for_device(device);
        let executor = StageExecutor::new(&scheduler);

        let input_stage = if input.is_single_segment() {
            None
        } else {
            let buffer = create_internal_buffer(
                device,
                "wgpu_fft.c2r.large_decompose.segmented_packed_input_stage",
                sizes.packed_bytes,
                wgpu::BufferUsages::COPY_DST,
            )?;
            executor.copy_view_range_to_buffer(
                encoder,
                &input,
                0,
                &buffer,
                0,
                sizes.packed_bytes,
            )?;
            Some(buffer)
        };
        let output_stage = if output.is_single_segment() {
            None
        } else {
            Some(create_internal_buffer(
                device,
                "wgpu_fft.c2r.large_decompose.segmented_real_output_stage",
                sizes.real_bytes,
                wgpu::BufferUsages::COPY_SRC,
            )?)
        };

        let packed_input = if let Some(buffer) = input_stage.as_ref() {
            BufferView::whole(buffer).prefix(sizes.packed_bytes)?
        } else {
            input.clone()
        };
        let real_output = if let Some(buffer) = output_stage.as_ref() {
            BufferView::whole(buffer).prefix(sizes.real_bytes)?
        } else {
            output.clone()
        };

        let full_input =
            BufferView::whole(&self.full_input_buffer).prefix(sizes.full_complex_bytes)?;
        let full_output =
            BufferView::whole(&self.full_output_buffer).prefix(sizes.full_complex_bytes)?;
        let config = self.c2c.config();
        dispatch_unpack_c2r_windowed(
            device,
            encoder,
            config.shape(),
            config.batch(),
            packed_input,
            full_input.clone(),
            config.tuning().workgroup_size(),
        )?;
        self.c2c
            .execute_views_recorded(device, encoder, full_input, full_output.clone())?;
        dispatch_complex_to_real_windowed(
            device,
            encoder,
            full_output,
            real_output,
            config.tuning().workgroup_size(),
        )?;

        if let Some(buffer) = output_stage.as_ref() {
            executor.copy_buffer_to_view_range(encoder, buffer, 0, &output, 0, sizes.real_bytes)?;
        }
        Ok(())
    }
}

impl C2rLargeChunkPlan {
    fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        config: &FftConfig,
        _packed_shape: &[usize],
        sizes: &RealPlanSizes,
        limits: LargePolicyLimits,
    ) -> Result<Self> {
        let chunk = RealLargeChunkPlan::new_with_max_batches(
            sizes,
            config.batch() as u64,
            limits,
            config.tuning().large_chunk_max_batches(),
        )?;
        let child_tuning = config
            .tuning()
            .clone()
            .with_large_route(FftLargeRoute::Auto);
        let child_config = config
            .clone()
            .with_batch(chunk.chunk_batch_count as usize)
            .with_tuning(child_tuning);
        let child = Box::new(C2rPlan::new_with_large_policy_limits(
            device,
            queue,
            child_config,
            Some(LargePolicyLimits {
                max_storage_buffer_binding_size: limits.max_storage_buffer_binding_size,
                max_buffer_size: limits.max_buffer_size,
            }),
        )?);
        if child.large_routing_policy().route_mode() != LargeRouteMode::Normal {
            return Err(FftError::LargeChunkUnsupported {
                reason: "real large-chunk child plan did not fit normal binding limits",
                bytes_per_batch: chunk.full_complex_bytes_per_batch,
                max_bind_bytes: limits.max_storage_buffer_binding_size,
            });
        }
        let input_stage = create_internal_buffer(
            device,
            "wgpu_fft.c2r.large_chunk.packed_input_stage",
            chunk.packed_staging_size_bytes,
            wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        )?;
        let output_stage = create_internal_buffer(
            device,
            "wgpu_fft.c2r.large_chunk.real_output_stage",
            chunk.real_staging_size_bytes,
            wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        )?;
        Ok(Self {
            chunk,
            child,
            input_stage,
            output_stage,
        })
    }

    fn execute_views(
        &self,
        device: &wgpu::Device,
        encoder: &mut CommandRecorder<'_>,
        input: BufferView<'_>,
        output: BufferView<'_>,
    ) -> Result<()> {
        let input = input.prefix(self.chunk.packed_total_bytes()?)?;
        let output = output.prefix(self.chunk.real_total_bytes()?)?;
        let child_input =
            BufferView::whole(&self.input_stage).prefix(self.chunk.packed_staging_size_bytes)?;
        let child_output =
            BufferView::whole(&self.output_stage).prefix(self.chunk.real_staging_size_bytes)?;

        for range in self.chunk.ranges() {
            let range = range?;
            copy_view_range_to_buffer(
                device,
                encoder,
                &input,
                range.packed_offset,
                &self.input_stage,
                0,
                range.packed_size,
            )?;
            self.child.execute_views_recorded(
                device,
                encoder,
                child_input.clone(),
                child_output.clone(),
            )?;
            copy_buffer_to_view_range(
                device,
                encoder,
                &self.output_stage,
                0,
                &output,
                range.real_offset,
                range.real_size,
            )?;
        }
        Ok(())
    }
}

impl RealKernel {
    fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        kind: RealKernelKind,
        shape: &[usize],
        param_value: u32,
        work_items: u32,
        workgroup_size: u32,
        label: &'static str,
    ) -> Result<Self> {
        let shader_key = RealStageKey::new(kind, shape, workgroup_size);
        validate_real_stage_key(&shader_key)?;
        let pipeline_key = ComputePipelineCacheKey::real_stage(shader_key.clone());
        let bind_group_layout = with_device_pipeline_cache(device, |cache| {
            cache.get_bind_group_layout(device, PipelineLayoutCacheKey::RealBinaryF32)
        });
        let pipeline = with_device_pipeline_cache(device, |cache| {
            cache.get_compute_pipeline(
                device,
                &pipeline_key,
                &format!("{label}.pipeline"),
                &format!("{label}.shader"),
                || generate_real_wgsl_for_key(&shader_key),
            )
        });
        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("{label}.params")),
            size: std::mem::size_of::<RealParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(
            &params_buffer,
            0,
            bytemuck::bytes_of(&RealParams {
                value: param_value,
                _pad0: 0,
                _pad1: 0,
                _pad2: 0,
            }),
        );

        Ok(Self {
            kind,
            pipeline_key,
            pipeline,
            bind_group_layout,
            params_buffer,
            workgroups_x: work_items.div_ceil(workgroup_size),
        })
    }

    fn execute(
        &self,
        device: &wgpu::Device,
        encoder: &mut CommandRecorder<'_>,
        input: BufferView<'_>,
        output: BufferView<'_>,
    ) -> Result<()> {
        let scheduler = WindowScheduler::for_device(device);
        let (input_format, output_format) = real_kernel_storage_formats(self.kind);
        let input_resource = scheduler.storage_binding_resource(&input, input_format)?;
        let output_resource = scheduler.storage_binding_resource(&output, output_format)?;
        let bind_group_label = format!(
            "wgpu_fft.real.bind_group.cache{}",
            self.pipeline_key.stable_key()
        );
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&bind_group_label),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_resource,
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output_resource,
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.params_buffer.as_entire_binding(),
                },
            ],
        });

        let pass = encoder.pass();
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let (x, y, z) = split_workgroups(self.workgroups_x, max_workgroups_per_dimension(device))?;
        pass.dispatch_workgroups(x, y, z);
        Ok(())
    }
}

fn real_kernel_storage_formats(kind: RealKernelKind) -> (ElementFormat, ElementFormat) {
    match kind {
        RealKernelKind::RealToComplex | RealKernelKind::RealToComplexWindowed => {
            (ElementFormat::RealF32, ElementFormat::ComplexF32)
        }
        RealKernelKind::ComplexToReal | RealKernelKind::ComplexToRealWindowed => {
            (ElementFormat::ComplexF32, ElementFormat::RealF32)
        }
        RealKernelKind::PackR2c
        | RealKernelKind::PackR2cWindowed
        | RealKernelKind::UnpackC2r
        | RealKernelKind::UnpackC2rWindowed
        | RealKernelKind::PackComplexStrided
        | RealKernelKind::UnpackComplexStrided => {
            (ElementFormat::ComplexF32, ElementFormat::ComplexF32)
        }
        RealKernelKind::PackRealStrided | RealKernelKind::UnpackRealStrided => {
            (ElementFormat::RealF32, ElementFormat::RealF32)
        }
    }
}

fn dispatch_real_to_complex_windowed(
    device: &wgpu::Device,
    encoder: &mut CommandRecorder<'_>,
    input: BufferView<'_>,
    output: BufferView<'_>,
    workgroup_size: u32,
) -> Result<()> {
    dispatch_linear_real_windowed(
        device,
        encoder,
        RealKernelKind::RealToComplexWindowed,
        &[],
        input,
        output,
        ElementFormat::RealF32,
        ElementFormat::ComplexF32,
        workgroup_size,
    )
}

fn dispatch_complex_to_real_windowed(
    device: &wgpu::Device,
    encoder: &mut CommandRecorder<'_>,
    input: BufferView<'_>,
    output: BufferView<'_>,
    workgroup_size: u32,
) -> Result<()> {
    dispatch_linear_real_windowed(
        device,
        encoder,
        RealKernelKind::ComplexToRealWindowed,
        &[],
        input,
        output,
        ElementFormat::ComplexF32,
        ElementFormat::RealF32,
        workgroup_size,
    )
}

fn dispatch_linear_real_windowed(
    device: &wgpu::Device,
    encoder: &mut CommandRecorder<'_>,
    kind: RealKernelKind,
    shape: &[usize],
    input: BufferView<'_>,
    output: BufferView<'_>,
    input_format: ElementFormat,
    output_format: ElementFormat,
    workgroup_size: u32,
) -> Result<()> {
    let input_count = input.size() / input_format.bytes_per_element();
    let output_count = output.size() / output_format.bytes_per_element();
    let total = input_count.min(output_count);
    let mut start = 0u64;
    while start < total {
        let count = choose_real_window_count(
            device,
            &input,
            start,
            input_format,
            &output,
            start,
            output_format,
            total - start,
        )?;
        let scheduler = WindowScheduler::for_device(device);
        let (input_binding, input_base) =
            scheduler.bind_element_window(&input, start, count, input_format)?;
        let (output_binding, output_base) =
            scheduler.bind_element_window(&output, start, count, output_format)?;
        dispatch_real_windowed_kernel(
            device,
            encoder,
            kind,
            shape,
            &input_binding,
            &output_binding,
            RealWindowParams {
                total: u64_to_u32(count)?,
                input_base,
                output_base,
                input_logical_start: u64_to_u32(start)?,
                output_logical_start: u64_to_u32(start)?,
                batch: 1,
                _pad1: 0,
                _pad2: 0,
            },
            workgroup_size,
        )?;
        start += count;
    }
    Ok(())
}

fn dispatch_pack_r2c_windowed(
    device: &wgpu::Device,
    encoder: &mut CommandRecorder<'_>,
    shape: &[usize],
    batch: usize,
    input: BufferView<'_>,
    output: BufferView<'_>,
    workgroup_size: u32,
) -> Result<()> {
    let packed_shape = packed_shape_for(shape)?;
    let full_n0 = shape[0] as u64;
    let packed_n0 = packed_shape[0] as u64;
    let rows = checked_product(&shape[1..])?
        .checked_mul(batch)
        .ok_or(FftError::LengthTooLarge { len: usize::MAX })? as u64;

    for row in 0..rows {
        let full_row_base = row
            .checked_mul(full_n0)
            .ok_or(FftError::LengthTooLarge { len: usize::MAX })?;
        let packed_row_base = row
            .checked_mul(packed_n0)
            .ok_or(FftError::LengthTooLarge { len: usize::MAX })?;
        let mut x = 0u64;
        while x < packed_n0 {
            let full_start = full_row_base + x;
            let packed_start = packed_row_base + x;
            let count = choose_real_window_count(
                device,
                &input,
                full_start,
                ElementFormat::ComplexF32,
                &output,
                packed_start,
                ElementFormat::PackedComplexF32,
                packed_n0 - x,
            )?;
            dispatch_real_windowed_pair(
                device,
                encoder,
                RealKernelKind::PackR2cWindowed,
                shape,
                &input,
                full_start,
                ElementFormat::ComplexF32,
                &output,
                packed_start,
                ElementFormat::PackedComplexF32,
                count,
                batch,
                workgroup_size,
            )?;
            x += count;
        }
    }
    Ok(())
}

fn dispatch_unpack_c2r_windowed(
    device: &wgpu::Device,
    encoder: &mut CommandRecorder<'_>,
    shape: &[usize],
    batch: usize,
    input: BufferView<'_>,
    output: BufferView<'_>,
    workgroup_size: u32,
) -> Result<()> {
    let packed_shape = packed_shape_for(shape)?;
    let full_n0 = shape[0] as u64;
    let packed_n0 = packed_shape[0] as u64;
    let rows_per_batch = checked_product(&shape[1..])? as u64;
    let rows = rows_per_batch
        .checked_mul(batch as u64)
        .ok_or(FftError::LengthTooLarge { len: usize::MAX })?;

    for row in 0..rows {
        let full_row_base = row
            .checked_mul(full_n0)
            .ok_or(FftError::LengthTooLarge { len: usize::MAX })?;
        let mut x = 0u64;
        while x < full_n0 {
            let boundary = if x < packed_n0 { packed_n0 } else { full_n0 };
            let remaining = boundary - x;
            let packed_row_base = if x < packed_n0 {
                row.checked_mul(packed_n0)
                    .ok_or(FftError::LengthTooLarge { len: usize::MAX })?
            } else {
                mirrored_packed_row_base(shape, &packed_shape, row, rows_per_batch)?
            };
            let output_start = full_row_base + x;
            let input_start = if x < packed_n0 {
                packed_row_base + x
            } else {
                packed_row_base + full_n0 - (x + remaining) + 1
            };
            let count = choose_real_window_count(
                device,
                &input,
                input_start,
                ElementFormat::PackedComplexF32,
                &output,
                output_start,
                ElementFormat::ComplexF32,
                remaining,
            )?;
            let input_start = if x < packed_n0 {
                packed_row_base + x
            } else {
                packed_row_base + full_n0 - (x + count) + 1
            };
            dispatch_real_windowed_pair(
                device,
                encoder,
                RealKernelKind::UnpackC2rWindowed,
                shape,
                &input,
                input_start,
                ElementFormat::PackedComplexF32,
                &output,
                output_start,
                ElementFormat::ComplexF32,
                count,
                batch,
                workgroup_size,
            )?;
            x += count;
        }
    }
    Ok(())
}

fn dispatch_real_windowed_pair(
    device: &wgpu::Device,
    encoder: &mut CommandRecorder<'_>,
    kind: RealKernelKind,
    shape: &[usize],
    input: &BufferView<'_>,
    input_start: u64,
    input_format: ElementFormat,
    output: &BufferView<'_>,
    output_start: u64,
    output_format: ElementFormat,
    count: u64,
    batch: usize,
    workgroup_size: u32,
) -> Result<()> {
    let scheduler = WindowScheduler::for_device(device);
    let (input_binding, input_base) =
        scheduler.bind_element_window(input, input_start, count, input_format)?;
    let (output_binding, output_base) =
        scheduler.bind_element_window(output, output_start, count, output_format)?;
    dispatch_real_windowed_kernel(
        device,
        encoder,
        kind,
        shape,
        &input_binding,
        &output_binding,
        RealWindowParams {
            total: u64_to_u32(count)?,
            input_base,
            output_base,
            input_logical_start: u64_to_u32(input_start)?,
            output_logical_start: u64_to_u32(output_start)?,
            batch: batch
                .try_into()
                .map_err(|_| FftError::LengthTooLarge { len: usize::MAX })?,
            _pad1: 0,
            _pad2: 0,
        },
        workgroup_size,
    )
}

fn choose_real_window_count(
    device: &wgpu::Device,
    input: &BufferView<'_>,
    input_start: u64,
    input_format: ElementFormat,
    output: &BufferView<'_>,
    output_start: u64,
    output_format: ElementFormat,
    remaining: u64,
) -> Result<u64> {
    let scheduler = WindowScheduler::for_device(device);
    let mut count = remaining.min(u64::from(u32::MAX));
    while count > 0 {
        if scheduler
            .storage_window_fits(input, input_start, count, input_format)
            .unwrap_or(false)
            && scheduler
                .storage_window_fits(output, output_start, count, output_format)
                .unwrap_or(false)
        {
            return Ok(count);
        }
        count /= 2;
    }
    Err(FftError::WindowScheduleUnsupported {
        reason: "real helper window cannot fit storage binding limits",
        requested_bytes: remaining
            .checked_mul(
                input_format
                    .bytes_per_element()
                    .max(output_format.bytes_per_element()),
            )
            .ok_or(FftError::LengthTooLarge { len: usize::MAX })?,
        max_bind_bytes: device.limits().max_storage_buffer_binding_size,
    })
}

fn dispatch_real_windowed_kernel(
    device: &wgpu::Device,
    encoder: &mut CommandRecorder<'_>,
    kind: RealKernelKind,
    shape: &[usize],
    input: &BufferView<'_>,
    output: &BufferView<'_>,
    params: RealWindowParams,
    workgroup_size: u32,
) -> Result<()> {
    let scheduler = WindowScheduler::for_device(device);
    let (input_format, output_format) = real_kernel_storage_formats(kind);
    let input_resource = scheduler.storage_binding_resource(input, input_format)?;
    let output_resource = scheduler.storage_binding_resource(output, output_format)?;
    let shader_key = RealStageKey::new(kind, shape, workgroup_size);
    validate_real_stage_key(&shader_key)?;
    let pipeline_key = ComputePipelineCacheKey::real_stage(shader_key.clone());
    let bind_group_layout = with_device_pipeline_cache(device, |cache| {
        cache.get_bind_group_layout(device, PipelineLayoutCacheKey::RealBinaryF32)
    });
    let pipeline = with_device_pipeline_cache(device, |cache| {
        cache.get_compute_pipeline(
            device,
            &pipeline_key,
            &format!(
                "wgpu_fft.real.windowed.pipeline.{}",
                pipeline_key.stable_key()
            ),
            &format!("wgpu_fft.real.windowed.shader.{}", shader_key.stable_key()),
            || generate_real_wgsl_for_key(&shader_key),
        )
    });
    let params_buffer = create_real_window_params_buffer(device, params);
    let bind_group_label = format!(
        "wgpu_fft.real.windowed.bind_group.{}",
        pipeline_key.stable_key()
    );
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(&bind_group_label),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input_resource,
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output_resource,
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: params_buffer.as_entire_binding(),
            },
        ],
    });
    let pass = encoder.pass();
    pass.set_pipeline(&pipeline);
    pass.set_bind_group(0, &bind_group, &[]);
    let (x, y, z) = split_workgroups(
        params.total.div_ceil(workgroup_size),
        max_workgroups_per_dimension(device),
    )?;
    pass.dispatch_workgroups(x, y, z);
    Ok(())
}

fn dispatch_real_strided_copy(
    device: &wgpu::Device,
    encoder: &mut CommandRecorder<'_>,
    kind: RealKernelKind,
    format: FftEndpointFormat,
    input: &BufferView<'_>,
    output: &BufferView<'_>,
    layout: FftLogicalLayout,
    logical_per_batch: u64,
    batch: u64,
    workgroup_size: u32,
) -> Result<()> {
    validate_real_strided_kind(kind, format)?;
    let scheduler = WindowScheduler::for_device(device);
    let (input_format, output_format) = real_kernel_storage_formats(kind);
    let input_resource = scheduler.storage_binding_resource(input, input_format)?;
    let output_resource = scheduler.storage_binding_resource(output, output_format)?;
    let total_elements = logical_per_batch
        .checked_mul(batch)
        .ok_or(FftError::LengthTooLarge { len: usize::MAX })?;
    let shader_key = RealStageKey::new(kind, &[], workgroup_size);
    validate_real_stage_key(&shader_key)?;
    let pipeline_key = ComputePipelineCacheKey::real_stage(shader_key.clone());
    let bind_group_layout = with_device_pipeline_cache(device, |cache| {
        cache.get_bind_group_layout(device, PipelineLayoutCacheKey::RealBinaryF32)
    });
    let pipeline = with_device_pipeline_cache(device, |cache| {
        cache.get_compute_pipeline(
            device,
            &pipeline_key,
            &format!(
                "wgpu_fft.real.strided.pipeline.{}",
                pipeline_key.stable_key()
            ),
            &format!("wgpu_fft.real.strided.shader.{}", shader_key.stable_key()),
            || generate_real_wgsl_for_key(&shader_key),
        )
    });
    let params = RealStridedCopyParams {
        total_elements: u64_to_u32(total_elements)?,
        logical_per_batch: u64_to_u32(logical_per_batch)?,
        element_offset: u64_to_u32(layout.element_offset)?,
        element_stride: u64_to_u32(layout.element_stride)?,
        batch_stride: u64_to_u32(layout.resolved_batch_stride(logical_per_batch)?)?,
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
    };
    let params_buffer = create_real_strided_params_buffer(device, params);
    let bind_group_label = format!(
        "wgpu_fft.real.strided.bind_group.{}",
        pipeline_key.stable_key()
    );
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(&bind_group_label),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input_resource,
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output_resource,
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: params_buffer.as_entire_binding(),
            },
        ],
    });
    let pass = encoder.pass();
    pass.set_pipeline(&pipeline);
    pass.set_bind_group(0, &bind_group, &[]);
    let (x, y, z) = split_workgroups(
        params.total_elements.div_ceil(workgroup_size),
        max_workgroups_per_dimension(device),
    )?;
    pass.dispatch_workgroups(x, y, z);
    Ok(())
}

fn create_real_strided_params_buffer(
    device: &wgpu::Device,
    params: RealStridedCopyParams,
) -> wgpu::Buffer {
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.real.strided.params"),
        size: std::mem::size_of::<RealStridedCopyParams>() as u64,
        usage: wgpu::BufferUsages::UNIFORM,
        mapped_at_creation: true,
    });
    {
        let mut mapped = buffer
            .slice(..)
            .get_mapped_range_mut()
            .expect("uniform buffer is mapped at creation");
        mapped.copy_from_slice(bytemuck::bytes_of(&params));
    }
    buffer.unmap();
    buffer
}

fn create_real_window_params_buffer(
    device: &wgpu::Device,
    params: RealWindowParams,
) -> wgpu::Buffer {
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.real.windowed.params"),
        size: std::mem::size_of::<RealWindowParams>() as u64,
        usage: wgpu::BufferUsages::UNIFORM,
        mapped_at_creation: true,
    });
    {
        let mut mapped = buffer
            .slice(..)
            .get_mapped_range_mut()
            .expect("uniform buffer is mapped at creation");
        mapped.copy_from_slice(bytemuck::bytes_of(&params));
    }
    buffer.unmap();
    buffer
}

fn mirrored_packed_row_base(
    shape: &[usize],
    packed_shape: &[usize],
    row: u64,
    rows_per_batch: u64,
) -> Result<u64> {
    let batch = row / rows_per_batch;
    let mut rem = row - batch * rows_per_batch;
    let mut mirrored_row = 0u64;
    let mut stride = 1u64;
    for &dim in &shape[1..] {
        let dim = dim as u64;
        let coord = rem % dim;
        rem /= dim;
        let mirrored = if coord == 0 { 0 } else { dim - coord };
        mirrored_row = mirrored_row
            .checked_add(
                mirrored
                    .checked_mul(stride)
                    .ok_or(FftError::LengthTooLarge { len: usize::MAX })?,
            )
            .ok_or(FftError::LengthTooLarge { len: usize::MAX })?;
        stride = stride
            .checked_mul(dim)
            .ok_or(FftError::LengthTooLarge { len: usize::MAX })?;
    }
    let packed_rows_per_batch = checked_product(&packed_shape[1..])? as u64;
    batch
        .checked_mul(packed_rows_per_batch)
        .and_then(|base| base.checked_add(mirrored_row))
        .and_then(|packed_row| packed_row.checked_mul(packed_shape[0] as u64))
        .ok_or(FftError::LengthTooLarge { len: usize::MAX })
}

fn u64_to_u32(value: u64) -> Result<u32> {
    value
        .try_into()
        .map_err(|_| FftError::BufferLayoutTooLarge {
            value,
            limit: u64::from(u32::MAX),
        })
}

impl RealPlanSizes {
    fn new(config: &FftConfig, packed_shape: &[usize]) -> Result<Self> {
        let total_real = config.total_complex_len()? as u64;
        let packed_total = packed_total_complex_len(packed_shape, config.batch())?;
        Ok(Self {
            real_bytes: total_real
                .checked_mul(F32_BYTES)
                .ok_or(FftError::LengthTooLarge { len: usize::MAX })?,
            packed_bytes: (packed_total as u64)
                .checked_mul(COMPLEX_F32_BYTES)
                .ok_or(FftError::LengthTooLarge { len: usize::MAX })?,
            full_complex_bytes: total_real
                .checked_mul(COMPLEX_F32_BYTES)
                .ok_or(FftError::LengthTooLarge { len: usize::MAX })?,
        })
    }

    fn per_batch(&self, batch_count: u64) -> Result<RealPlanSizes> {
        if batch_count == 0 {
            return Err(FftError::ZeroBatch);
        }
        Ok(RealPlanSizes {
            real_bytes: self.real_bytes / batch_count,
            packed_bytes: self.packed_bytes / batch_count,
            full_complex_bytes: self.full_complex_bytes / batch_count,
        })
    }
}

impl RealLargeChunkPlan {
    #[cfg(test)]
    fn new(sizes: &RealPlanSizes, batch_count: u64, limits: LargePolicyLimits) -> Result<Self> {
        Self::new_with_max_batches(sizes, batch_count, limits, None)
    }

    fn new_with_max_batches(
        sizes: &RealPlanSizes,
        batch_count: u64,
        limits: LargePolicyLimits,
        max_batches: Option<usize>,
    ) -> Result<Self> {
        let per_batch = sizes.per_batch(batch_count)?;
        let bytes_per_batch = per_batch
            .real_bytes
            .max(per_batch.packed_bytes)
            .max(per_batch.full_complex_bytes);
        let chunk = LargeChunkPlan::new_with_max_batches(
            bytes_per_batch,
            batch_count,
            limits,
            max_batches,
        )?;
        let chunk_batch_count = chunk.chunk_batch_count();
        let real_staging_size_bytes = checked_mul_for_real_chunk(
            per_batch.real_bytes,
            chunk_batch_count,
            limits.max_storage_buffer_binding_size,
        )?;
        let packed_staging_size_bytes = checked_mul_for_real_chunk(
            per_batch.packed_bytes,
            chunk_batch_count,
            limits.max_storage_buffer_binding_size,
        )?;

        Ok(Self {
            batch_count,
            chunk_batch_count,
            real_bytes_per_batch: per_batch.real_bytes,
            packed_bytes_per_batch: per_batch.packed_bytes,
            full_complex_bytes_per_batch: per_batch.full_complex_bytes,
            real_staging_size_bytes,
            packed_staging_size_bytes,
        })
    }

    fn ranges(self) -> RealLargeChunkRanges {
        RealLargeChunkRanges {
            plan: self,
            next_batch: 0,
        }
    }

    fn real_total_bytes(self) -> Result<u64> {
        checked_mul_for_real_chunk(
            self.real_bytes_per_batch,
            self.batch_count,
            self.real_bytes_per_batch,
        )
    }

    fn packed_total_bytes(self) -> Result<u64> {
        checked_mul_for_real_chunk(
            self.packed_bytes_per_batch,
            self.batch_count,
            self.packed_bytes_per_batch,
        )
    }
}

struct RealLargeChunkRanges {
    plan: RealLargeChunkPlan,
    next_batch: u64,
}

impl Iterator for RealLargeChunkRanges {
    type Item = Result<RealLargeChunkRange>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next_batch >= self.plan.batch_count {
            return None;
        }
        let batch_start = self.next_batch;
        let batch_count = (self.plan.batch_count - batch_start).min(self.plan.chunk_batch_count);
        self.next_batch += batch_count;

        let real_offset = match real_chunk_range_mul(
            batch_start,
            self.plan.real_bytes_per_batch,
            "real chunk range offset overflowed u64",
        ) {
            Ok(real_offset) => real_offset,
            Err(error) => return Some(Err(error)),
        };
        let real_size = match real_chunk_range_mul(
            batch_count,
            self.plan.real_bytes_per_batch,
            "real chunk range size overflowed u64",
        ) {
            Ok(real_size) => real_size,
            Err(error) => return Some(Err(error)),
        };
        let packed_offset = match real_chunk_range_mul(
            batch_start,
            self.plan.packed_bytes_per_batch,
            "packed chunk range offset overflowed u64",
        ) {
            Ok(packed_offset) => packed_offset,
            Err(error) => return Some(Err(error)),
        };
        let packed_size = match real_chunk_range_mul(
            batch_count,
            self.plan.packed_bytes_per_batch,
            "packed chunk range size overflowed u64",
        ) {
            Ok(packed_size) => packed_size,
            Err(error) => return Some(Err(error)),
        };

        Some(Ok(RealLargeChunkRange {
            batch_start,
            batch_count,
            real_offset,
            real_size,
            packed_offset,
            packed_size,
        }))
    }
}

fn real_chunk_range_mul(value: u64, bytes_per_batch: u64, reason: &'static str) -> Result<u64> {
    value
        .checked_mul(bytes_per_batch)
        .ok_or(FftError::LargeChunkUnsupported {
            reason,
            bytes_per_batch,
            max_bind_bytes: u64::MAX,
        })
}

fn checked_mul_for_real_chunk(a: u64, b: u64, max_bind_bytes: u64) -> Result<u64> {
    a.checked_mul(b).ok_or(FftError::LargeChunkUnsupported {
        reason: "real chunk size overflowed u64",
        bytes_per_batch: a,
        max_bind_bytes,
    })
}

fn validate_real_config(config: &FftConfig, transform: RealTransform) -> Result<()> {
    config.validate()?;
    let expected_direction = match transform {
        RealTransform::R2c => FftDirection::Forward,
        RealTransform::C2r => FftDirection::Inverse,
    };
    if config.direction() != expected_direction {
        return Err(FftError::InvalidRealTransformDirection {
            transform: transform.as_str(),
            expected: direction_str(expected_direction),
            actual: direction_str(config.direction()),
        });
    }

    let expected_axes = (0..config.shape().len()).collect::<Vec<_>>();
    if config.axes() != expected_axes.as_slice() {
        return Err(FftError::UnsupportedRealAxes {
            expected: expected_axes,
            actual: config.axes().to_vec(),
        });
    }

    Ok(())
}

fn direction_str(direction: FftDirection) -> &'static str {
    match direction {
        FftDirection::Forward => "forward",
        FftDirection::Inverse => "inverse",
    }
}

fn packed_shape_for(shape: &[usize]) -> Result<Vec<usize>> {
    if shape.is_empty() || shape[0] == 0 {
        return Err(FftError::ZeroLength);
    }
    let mut packed = shape.to_vec();
    packed[0] = shape[0] / 2 + 1;
    Ok(packed)
}

fn checked_product(values: &[usize]) -> Result<usize> {
    let mut product = 1usize;
    for &value in values {
        product = product
            .checked_mul(value)
            .ok_or(FftError::LengthTooLarge { len: usize::MAX })?;
    }
    Ok(product)
}

fn packed_total_complex_len(packed_shape: &[usize], batch: usize) -> Result<usize> {
    checked_product(packed_shape)?
        .checked_mul(batch)
        .ok_or(FftError::LengthTooLarge { len: usize::MAX })
}

fn resolve_real_large_routing_policy(
    device: &wgpu::Device,
    config: &FftConfig,
    packed_shape: &[usize],
    sizes: &RealPlanSizes,
    policy_limits: Option<LargePolicyLimits>,
) -> Result<LargeRoutingPolicy> {
    let axis_kinds = resolve_axis_kinds_for_config(config)?;
    let line_bytes = [
        (config.shape()[0] as u64)
            .checked_mul(F32_BYTES)
            .ok_or(FftError::LengthTooLarge { len: usize::MAX })?,
        (config.shape()[0] as u64)
            .checked_mul(COMPLEX_F32_BYTES)
            .ok_or(FftError::LengthTooLarge { len: usize::MAX })?,
        (packed_shape[0] as u64)
            .checked_mul(COMPLEX_F32_BYTES)
            .ok_or(FftError::LengthTooLarge { len: usize::MAX })?,
    ];
    let required_bindings = [
        sizes.full_complex_bytes,
        sizes.packed_bytes,
        sizes.real_bytes,
        sizes.real_bytes.max(sizes.packed_bytes),
    ];
    let per_batch = sizes.per_batch(config.batch() as u64)?;
    let bytes_per_batch = per_batch
        .real_bytes
        .max(per_batch.packed_bytes)
        .max(per_batch.full_complex_bytes);
    let limits = effective_real_policy_limits(device, config, policy_limits);

    resolve_large_routing_policy(LargeRoutingPolicyInput {
        limits,
        required_binding_bytes: &required_bindings,
        line_bytes: &line_bytes,
        axis_kinds: Some(&axis_kinds),
        axis_lengths: Some(config.shape()),
        allow_non_mixed_bounded_slicing: true,
        allow_out_of_core: config.shape().len() >= 2,
        rank: config.shape().len(),
        bytes_per_batch: Some(bytes_per_batch),
        ..LargeRoutingPolicyInput::new(limits, &[])
    })
}

fn create_internal_buffer(
    device: &wgpu::Device,
    label: &'static str,
    size: u64,
    extra_usage: wgpu::BufferUsages,
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
        usage: wgpu::BufferUsages::STORAGE | extra_usage,
        mapped_at_creation: false,
    }))
}

fn copy_view_range_to_buffer(
    device: &wgpu::Device,
    encoder: &mut CommandRecorder<'_>,
    view: &BufferView<'_>,
    view_offset: u64,
    dst: &wgpu::Buffer,
    dst_offset: u64,
    size: u64,
) -> Result<()> {
    let scheduler = WindowScheduler::for_device(device);
    StageExecutor::new(&scheduler).copy_view_range_to_buffer(
        encoder,
        view,
        view_offset,
        dst,
        dst_offset,
        size,
    )
}

fn copy_buffer_to_view_range(
    device: &wgpu::Device,
    encoder: &mut CommandRecorder<'_>,
    src: &wgpu::Buffer,
    src_offset: u64,
    view: &BufferView<'_>,
    view_offset: u64,
    size: u64,
) -> Result<()> {
    let scheduler = WindowScheduler::for_device(device);
    StageExecutor::new(&scheduler).copy_buffer_to_view_range(
        encoder,
        src,
        src_offset,
        view,
        view_offset,
        size,
    )
}

pub(crate) fn generate_real_wgsl_for_key(key: &RealStageKey) -> String {
    match key.kind {
        RealKernelKind::RealToComplex => generate_real_to_complex_wgsl(key.workgroup_size),
        RealKernelKind::RealToComplexWindowed => {
            generate_real_to_complex_windowed_wgsl(key.workgroup_size)
        }
        RealKernelKind::PackR2c => generate_pack_r2c_wgsl(&key.dims, key.workgroup_size),
        RealKernelKind::PackR2cWindowed => {
            generate_pack_r2c_windowed_wgsl(&key.dims, key.workgroup_size)
        }
        RealKernelKind::UnpackC2r => generate_unpack_c2r_wgsl(&key.dims, key.workgroup_size),
        RealKernelKind::UnpackC2rWindowed => {
            generate_unpack_c2r_windowed_wgsl(&key.dims, key.workgroup_size)
        }
        RealKernelKind::ComplexToReal => generate_complex_to_real_wgsl(key.workgroup_size),
        RealKernelKind::ComplexToRealWindowed => {
            generate_complex_to_real_windowed_wgsl(key.workgroup_size)
        }
        RealKernelKind::PackRealStrided | RealKernelKind::UnpackRealStrided => {
            generate_real_strided_wgsl(key.kind, key.workgroup_size)
        }
        RealKernelKind::PackComplexStrided | RealKernelKind::UnpackComplexStrided => {
            generate_complex_strided_wgsl(key.kind, key.workgroup_size)
        }
    }
}

fn generate_real_strided_wgsl(kind: RealKernelKind, workgroup_size: u32) -> String {
    let assignment = match kind {
        RealKernelKind::PackRealStrided => "output[i] = input[physical_index];",
        RealKernelKind::UnpackRealStrided => "output[physical_index] = input[i];",
        _ => unreachable!("real strided generator called with non-real kind"),
    };
    generate_strided_copy_wgsl("f32", assignment, workgroup_size)
}

fn generate_complex_strided_wgsl(kind: RealKernelKind, workgroup_size: u32) -> String {
    let assignment = match kind {
        RealKernelKind::PackComplexStrided => "output[i] = input[physical_index];",
        RealKernelKind::UnpackComplexStrided => "output[physical_index] = input[i];",
        _ => unreachable!("complex strided generator called with non-complex kind"),
    };
    generate_strided_copy_wgsl("vec2<f32>", assignment, workgroup_size)
}

fn generate_strided_copy_wgsl(element_type: &str, assignment: &str, workgroup_size: u32) -> String {
    format!(
        r#"
struct Params {{
  total_elements: u32,
  logical_per_batch: u32,
  element_offset: u32,
  element_stride: u32,
  batch_stride: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
}}

@group(0) @binding(0) var<storage, read> input: array<{element_type}>;
@group(0) @binding(1) var<storage, read_write> output: array<{element_type}>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size({workgroup_size}, 1, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {{
  let wgFlat: u32 = (wid.z * nwg.y + wid.y) * nwg.x + wid.x;
  if (wgFlat > params.total_elements / {workgroup_size}u) {{ return; }}
  let i: u32 = wgFlat * {workgroup_size}u + lid.x;
  if (i >= params.total_elements) {{ return; }}
  let batch: u32 = i / params.logical_per_batch;
  let element: u32 = i - batch * params.logical_per_batch;
  let physical_index: u32 = params.element_offset + batch * params.batch_stride + element * params.element_stride;
  {assignment}
}}
"#
    )
}

fn generate_real_to_complex_wgsl(workgroup_size: u32) -> String {
    format!(
        r#"
struct Params {{
  total: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
}}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<vec2<f32>>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size({workgroup_size}, 1, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {{
  let wgFlat: u32 = (wid.z * nwg.y + wid.y) * nwg.x + wid.x;
  if (wgFlat > params.total / {workgroup_size}u) {{ return; }}
  let i: u32 = wgFlat * {workgroup_size}u + lid.x;
  if (i >= params.total) {{ return; }}
  output[i] = vec2<f32>(input[i], 0.0);
}}
"#
    )
}

fn generate_complex_to_real_wgsl(workgroup_size: u32) -> String {
    format!(
        r#"
struct Params {{
  total: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
}}

@group(0) @binding(0) var<storage, read> input: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size({workgroup_size}, 1, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {{
  let wgFlat: u32 = (wid.z * nwg.y + wid.y) * nwg.x + wid.x;
  if (wgFlat > params.total / {workgroup_size}u) {{ return; }}
  let i: u32 = wgFlat * {workgroup_size}u + lid.x;
  if (i >= params.total) {{ return; }}
  output[i] = input[i].x;
}}
"#
    )
}

fn generate_real_to_complex_windowed_wgsl(workgroup_size: u32) -> String {
    format!(
        r#"
struct Params {{
  total: u32,
  input_base: u32,
  output_base: u32,
  input_logical_start: u32,
  output_logical_start: u32,
  batch: u32,
  _pad1: u32,
  _pad2: u32,
}}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<vec2<f32>>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size({workgroup_size}, 1, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {{
  let wgFlat: u32 = (wid.z * nwg.y + wid.y) * nwg.x + wid.x;
  if (wgFlat > params.total / {workgroup_size}u) {{ return; }}
  let i: u32 = wgFlat * {workgroup_size}u + lid.x;
  if (i >= params.total) {{ return; }}
  output[params.output_base + i] = vec2<f32>(input[params.input_base + i], 0.0);
}}
"#
    )
}

fn generate_complex_to_real_windowed_wgsl(workgroup_size: u32) -> String {
    format!(
        r#"
struct Params {{
  total: u32,
  input_base: u32,
  output_base: u32,
  input_logical_start: u32,
  output_logical_start: u32,
  batch: u32,
  _pad1: u32,
  _pad2: u32,
}}

@group(0) @binding(0) var<storage, read> input: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size({workgroup_size}, 1, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {{
  let wgFlat: u32 = (wid.z * nwg.y + wid.y) * nwg.x + wid.x;
  if (wgFlat > params.total / {workgroup_size}u) {{ return; }}
  let i: u32 = wgFlat * {workgroup_size}u + lid.x;
  if (i >= params.total) {{ return; }}
  output[params.output_base + i] = input[params.input_base + i].x;
}}
"#
    )
}

fn generate_pack_r2c_wgsl(shape: &[usize], workgroup_size: u32) -> String {
    let out_shape = packed_shape_for(shape).expect("validated real shape must pack");
    let in_strides = row_major_strides(shape);
    let in_total = product(shape);
    let out_total = product(&out_shape);
    let decoded = decode_coords_wgsl("rem", &out_shape, "c");

    let mut in_index_body = format!("  var inIndex: u32 = b * {in_total}u;\n");
    for (dim, coord) in decoded.coords.iter().enumerate() {
        if in_strides[dim] == 1 {
            in_index_body.push_str(&format!("  inIndex = inIndex + {coord};\n"));
        } else {
            in_index_body.push_str(&format!(
                "  inIndex = inIndex + {coord} * {}u;\n",
                in_strides[dim]
            ));
        }
    }

    format!(
        r#"
struct Params {{
  batch: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
}}

@group(0) @binding(0) var<storage, read> input: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec2<f32>>;
@group(0) @binding(2) var<uniform> params: Params;

const OUT_TOTAL_PER_BATCH: u32 = {out_total}u;

@compute @workgroup_size({workgroup_size}, 1, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {{
  let wgFlat: u32 = (wid.z * nwg.y + wid.y) * nwg.x + wid.x;
  let totalOut: u32 = OUT_TOTAL_PER_BATCH * params.batch;
  if (wgFlat > totalOut / {workgroup_size}u) {{ return; }}
  let i: u32 = wgFlat * {workgroup_size}u + lid.x;
  if (i >= totalOut) {{ return; }}
  let b: u32 = i / OUT_TOTAL_PER_BATCH;
  let rem: u32 = i - b * OUT_TOTAL_PER_BATCH;
{decode_code}
{in_index_body}
  output[i] = input[inIndex];
}}
"#,
        decode_code = decoded.code
    )
}

fn generate_unpack_c2r_wgsl(shape: &[usize], workgroup_size: u32) -> String {
    let in_shape = packed_shape_for(shape).expect("validated real shape must pack");
    let in_strides = row_major_strides(&in_shape);
    let full_total = product(shape);
    let in_total = product(&in_shape);
    let nx = shape[0];
    let in_nx = in_shape[0];
    let even = nx % 2 == 0;
    let decoded = decode_coords_wgsl("rem", shape, "c");

    let mut mirror_coords_code = String::new();
    let mut coord_for_in_index = Vec::with_capacity(shape.len());
    coord_for_in_index.push(String::from("xPacked"));
    for dim in 1..shape.len() {
        let coord = &decoded.coords[dim];
        let mirror = format!("c{dim}m");
        let packed = format!("c{dim}p");
        mirror_coords_code.push_str(&format!(
            "  let {mirror}: u32 = select(0u, {}u - {coord}, {coord} != 0u);\n",
            shape[dim]
        ));
        mirror_coords_code.push_str(&format!(
            "  let {packed}: u32 = select({coord}, {mirror}, x >= IN_NX);\n"
        ));
        coord_for_in_index.push(packed);
    }

    let mut in_index_body = format!("  var inIndex: u32 = b * {in_total}u;\n");
    for (dim, coord) in coord_for_in_index.iter().enumerate() {
        if in_strides[dim] == 1 {
            in_index_body.push_str(&format!("  inIndex = inIndex + {coord};\n"));
        } else {
            in_index_body.push_str(&format!(
                "  inIndex = inIndex + {coord} * {}u;\n",
                in_strides[dim]
            ));
        }
    }

    let mut self_conj_expr = String::from("(x == 0u || (EVEN_NX && x == (NX / 2u)))");
    for dim in 1..shape.len() {
        let coord = &decoded.coords[dim];
        if shape[dim] % 2 == 0 {
            self_conj_expr.push_str(&format!(
                " && ({coord} == 0u || {coord} == {}u)",
                shape[dim] / 2
            ));
        } else {
            self_conj_expr.push_str(&format!(" && ({coord} == 0u)"));
        }
    }

    format!(
        r#"
struct Params {{
  batch: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
}}

@group(0) @binding(0) var<storage, read> input: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec2<f32>>;
@group(0) @binding(2) var<uniform> params: Params;

const NX: u32 = {nx}u;
const IN_NX: u32 = {in_nx}u;
const EVEN_NX: bool = {even};
const OUT_TOTAL_PER_BATCH: u32 = {full_total}u;

fn conj(v: vec2<f32>) -> vec2<f32> {{ return vec2<f32>(v.x, -v.y); }}

@compute @workgroup_size({workgroup_size}, 1, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {{
  let wgFlat: u32 = (wid.z * nwg.y + wid.y) * nwg.x + wid.x;
  let totalOut: u32 = OUT_TOTAL_PER_BATCH * params.batch;
  if (wgFlat > totalOut / {workgroup_size}u) {{ return; }}
  let i: u32 = wgFlat * {workgroup_size}u + lid.x;
  if (i >= totalOut) {{ return; }}
  let b: u32 = i / OUT_TOTAL_PER_BATCH;
  let rem: u32 = i - b * OUT_TOTAL_PER_BATCH;
{decode_code}

  var v: vec2<f32>;
  let x: u32 = {x_coord};
  let xPacked: u32 = select(x, NX - x, x >= IN_NX);
{mirror_coords_code}
{in_index_body}
  v = input[inIndex];
  if (x >= IN_NX) {{ v = conj(v); }}
  if ({self_conj_expr}) {{
    v = vec2<f32>(v.x, 0.0);
  }}
  output[i] = v;
}}
"#,
        even = if even { "true" } else { "false" },
        decode_code = decoded.code,
        x_coord = decoded.coords[0]
    )
}

fn generate_pack_r2c_windowed_wgsl(shape: &[usize], workgroup_size: u32) -> String {
    let out_shape = packed_shape_for(shape).expect("validated real shape must pack");
    let in_strides = row_major_strides(shape);
    let in_total = product(shape);
    let out_total = product(&out_shape);
    let decoded = decode_coords_wgsl("rem", &out_shape, "c");

    let mut in_index_body = format!("  var inIndex: u32 = b * {in_total}u;\n");
    for (dim, coord) in decoded.coords.iter().enumerate() {
        if in_strides[dim] == 1 {
            in_index_body.push_str(&format!("  inIndex = inIndex + {coord};\n"));
        } else {
            in_index_body.push_str(&format!(
                "  inIndex = inIndex + {coord} * {}u;\n",
                in_strides[dim]
            ));
        }
    }

    format!(
        r#"
struct Params {{
  total: u32,
  input_base: u32,
  output_base: u32,
  input_logical_start: u32,
  output_logical_start: u32,
  batch: u32,
  _pad1: u32,
  _pad2: u32,
}}

@group(0) @binding(0) var<storage, read> input: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec2<f32>>;
@group(0) @binding(2) var<uniform> params: Params;

const OUT_TOTAL_PER_BATCH: u32 = {out_total}u;

@compute @workgroup_size({workgroup_size}, 1, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {{
  let wgFlat: u32 = (wid.z * nwg.y + wid.y) * nwg.x + wid.x;
  if (wgFlat > params.total / {workgroup_size}u) {{ return; }}
  let i: u32 = wgFlat * {workgroup_size}u + lid.x;
  if (i >= params.total) {{ return; }}
  let globalOut: u32 = params.output_logical_start + i;
  let b: u32 = globalOut / OUT_TOTAL_PER_BATCH;
  let rem: u32 = globalOut - b * OUT_TOTAL_PER_BATCH;
{decode_code}
{in_index_body}
  output[params.output_base + i] = input[params.input_base + (inIndex - params.input_logical_start)];
}}
"#,
        decode_code = decoded.code
    )
}

fn generate_unpack_c2r_windowed_wgsl(shape: &[usize], workgroup_size: u32) -> String {
    let in_shape = packed_shape_for(shape).expect("validated real shape must pack");
    let in_strides = row_major_strides(&in_shape);
    let full_total = product(shape);
    let in_total = product(&in_shape);
    let nx = shape[0];
    let in_nx = in_shape[0];
    let even = nx % 2 == 0;
    let decoded = decode_coords_wgsl("rem", shape, "c");

    let mut mirror_coords_code = String::new();
    let mut coord_for_in_index = Vec::with_capacity(shape.len());
    coord_for_in_index.push(String::from("xPacked"));
    for dim in 1..shape.len() {
        let coord = &decoded.coords[dim];
        let mirror = format!("c{dim}m");
        let packed = format!("c{dim}p");
        mirror_coords_code.push_str(&format!(
            "  let {mirror}: u32 = select(0u, {}u - {coord}, {coord} != 0u);\n",
            shape[dim]
        ));
        mirror_coords_code.push_str(&format!(
            "  let {packed}: u32 = select({coord}, {mirror}, x >= IN_NX);\n"
        ));
        coord_for_in_index.push(packed);
    }

    let mut in_index_body = format!("  var inIndex: u32 = b * {in_total}u;\n");
    for (dim, coord) in coord_for_in_index.iter().enumerate() {
        if in_strides[dim] == 1 {
            in_index_body.push_str(&format!("  inIndex = inIndex + {coord};\n"));
        } else {
            in_index_body.push_str(&format!(
                "  inIndex = inIndex + {coord} * {}u;\n",
                in_strides[dim]
            ));
        }
    }

    let mut self_conj_expr = String::from("(x == 0u || (EVEN_NX && x == (NX / 2u)))");
    for dim in 1..shape.len() {
        let coord = &decoded.coords[dim];
        if shape[dim] % 2 == 0 {
            self_conj_expr.push_str(&format!(
                " && ({coord} == 0u || {coord} == {}u)",
                shape[dim] / 2
            ));
        } else {
            self_conj_expr.push_str(&format!(" && ({coord} == 0u)"));
        }
    }

    format!(
        r#"
struct Params {{
  total: u32,
  input_base: u32,
  output_base: u32,
  input_logical_start: u32,
  output_logical_start: u32,
  batch: u32,
  _pad1: u32,
  _pad2: u32,
}}

@group(0) @binding(0) var<storage, read> input: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec2<f32>>;
@group(0) @binding(2) var<uniform> params: Params;

const NX: u32 = {nx}u;
const IN_NX: u32 = {in_nx}u;
const EVEN_NX: bool = {even};
const OUT_TOTAL_PER_BATCH: u32 = {full_total}u;

fn conj(v: vec2<f32>) -> vec2<f32> {{ return vec2<f32>(v.x, -v.y); }}

@compute @workgroup_size({workgroup_size}, 1, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {{
  let wgFlat: u32 = (wid.z * nwg.y + wid.y) * nwg.x + wid.x;
  if (wgFlat > params.total / {workgroup_size}u) {{ return; }}
  let i: u32 = wgFlat * {workgroup_size}u + lid.x;
  if (i >= params.total) {{ return; }}
  let globalOut: u32 = params.output_logical_start + i;
  let b: u32 = globalOut / OUT_TOTAL_PER_BATCH;
  let rem: u32 = globalOut - b * OUT_TOTAL_PER_BATCH;
{decode_code}

  var v: vec2<f32>;
  let x: u32 = {x_coord};
  let xPacked: u32 = select(x, NX - x, x >= IN_NX);
{mirror_coords_code}
{in_index_body}
  v = input[params.input_base + (inIndex - params.input_logical_start)];
  if (x >= IN_NX) {{ v = conj(v); }}
  if ({self_conj_expr}) {{
    v = vec2<f32>(v.x, 0.0);
  }}
  output[params.output_base + i] = v;
}}
"#,
        even = if even { "true" } else { "false" },
        decode_code = decoded.code,
        x_coord = decoded.coords[0]
    )
}

struct DecodedCoords {
    code: String,
    coords: Vec<String>,
}

fn decode_coords_wgsl(index_name: &str, dims: &[usize], coord_prefix: &str) -> DecodedCoords {
    let mut rem = index_name.to_owned();
    let mut coords = Vec::with_capacity(dims.len());
    let mut code = String::new();
    for dim in 0..dims.len() {
        let coord = format!("{coord_prefix}{dim}");
        coords.push(coord.clone());
        code.push_str(&format!("  let {coord}: u32 = {rem} % {}u;\n", dims[dim]));
        if dim + 1 < dims.len() {
            let next_rem = format!("{coord_prefix}rem{dim}");
            code.push_str(&format!(
                "  let {next_rem}: u32 = {rem} / {}u;\n",
                dims[dim]
            ));
            rem = next_rem;
        }
    }
    DecodedCoords { code, coords }
}

fn row_major_strides(dims: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; dims.len()];
    for index in 1..dims.len() {
        strides[index] = strides[index - 1] * dims[index - 1];
    }
    strides
}

fn product(values: &[usize]) -> usize {
    values.iter().product()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Normalization;
    use crate::tuning::FftTuning;

    fn test_child_c2c_graph(
        sizes: &RealPlanSizes,
        limits: LargePolicyLimits,
    ) -> LargeExecutionGraph {
        let mut child = LargeExecutionGraph::new("child-c2c");
        let input = real_range(
            LogicalBufferId::Input,
            0,
            sizes.full_complex_bytes,
            ElementFormat::ComplexF32,
        )
        .unwrap();
        let temp = real_range(
            LogicalBufferId::Temp(0),
            0,
            sizes.full_complex_bytes,
            ElementFormat::ComplexF32,
        )
        .unwrap();
        let output = real_range(
            LogicalBufferId::Output,
            0,
            sizes.full_complex_bytes,
            ElementFormat::ComplexF32,
        )
        .unwrap();
        child
            .push_stage(
                LargeStage::HelperWindow {
                    label: "mixed-radix-workspace",
                    range: temp,
                },
                graph_requirements(limits, sizes.full_complex_bytes).unwrap(),
            )
            .unwrap();
        child
            .push_stage(
                LargeStage::Kernel {
                    label: "mixed-radix-stockham-stage",
                    input,
                    output: temp,
                    work_items: sizes.full_complex_bytes / COMPLEX_F32_BYTES,
                },
                graph_requirements(limits, 0).unwrap(),
            )
            .unwrap();
        child
            .push_stage(
                LargeStage::Kernel {
                    label: "mixed-radix-stockham-stage",
                    input: temp,
                    output,
                    work_items: sizes.full_complex_bytes / COMPLEX_F32_BYTES,
                },
                graph_requirements(limits, 0).unwrap(),
            )
            .unwrap();
        child
    }

    #[test]
    fn computes_packed_shape_for_odd_and_even_axis_zero() {
        assert_eq!(packed_shape_for(&[16]).unwrap(), vec![9]);
        assert_eq!(packed_shape_for(&[17, 4]).unwrap(), vec![9, 4]);
    }

    #[test]
    fn computes_real_plan_byte_sizes() {
        let config = FftConfig::new_nd([17, 4])
            .with_batch(2)
            .with_normalization(Normalization::None);
        let packed = packed_shape_for(config.shape()).unwrap();
        let sizes = RealPlanSizes::new(&config, &packed).unwrap();
        assert_eq!(sizes.real_bytes, 17 * 4 * 2 * 4);
        assert_eq!(sizes.full_complex_bytes, 17 * 4 * 2 * 8);
        assert_eq!(sizes.packed_bytes, 9 * 4 * 2 * 8);
    }

    #[test]
    fn df64_real_endpoint_helpers_return_structured_precision_errors() {
        for error in [
            strided_pack_kind(FftEndpointFormat::ComplexDf64).unwrap_err(),
            strided_unpack_kind(FftEndpointFormat::ComplexDf64).unwrap_err(),
        ] {
            assert_eq!(
                error,
                FftError::PrecisionUnsupported {
                    requested: FftPrecision::Df64,
                    route: "real",
                    reason: "real-df64-not-implemented",
                }
            );
        }
    }

    #[test]
    fn plans_real_large_chunk_batch_ranges() {
        let config = FftConfig::new(16)
            .with_batch(5)
            .with_normalization(Normalization::None);
        let packed = packed_shape_for(config.shape()).unwrap();
        let sizes = RealPlanSizes::new(&config, &packed).unwrap();
        let plan = RealLargeChunkPlan::new(
            &sizes,
            config.batch() as u64,
            LargePolicyLimits {
                max_storage_buffer_binding_size: 16 * COMPLEX_F32_BYTES * 2,
                max_buffer_size: 1 << 20,
            },
        )
        .unwrap();

        assert_eq!(plan.chunk_batch_count, 2);
        assert_eq!(plan.real_staging_size_bytes, 16 * F32_BYTES * 2);
        assert_eq!(plan.packed_staging_size_bytes, 9 * COMPLEX_F32_BYTES * 2);
        let ranges = plan.ranges().collect::<Result<Vec<_>>>().unwrap();
        assert_eq!(
            ranges,
            [
                RealLargeChunkRange {
                    batch_start: 0,
                    batch_count: 2,
                    real_offset: 0,
                    real_size: 16 * F32_BYTES * 2,
                    packed_offset: 0,
                    packed_size: 9 * COMPLEX_F32_BYTES * 2,
                },
                RealLargeChunkRange {
                    batch_start: 2,
                    batch_count: 2,
                    real_offset: 16 * F32_BYTES * 2,
                    real_size: 16 * F32_BYTES * 2,
                    packed_offset: 9 * COMPLEX_F32_BYTES * 2,
                    packed_size: 9 * COMPLEX_F32_BYTES * 2,
                },
                RealLargeChunkRange {
                    batch_start: 4,
                    batch_count: 1,
                    real_offset: 16 * F32_BYTES * 4,
                    real_size: 16 * F32_BYTES,
                    packed_offset: 9 * COMPLEX_F32_BYTES * 4,
                    packed_size: 9 * COMPLEX_F32_BYTES,
                },
            ]
        );

        let capped = RealLargeChunkPlan::new_with_max_batches(
            &sizes,
            config.batch() as u64,
            LargePolicyLimits {
                max_storage_buffer_binding_size: 16 * COMPLEX_F32_BYTES * 2,
                max_buffer_size: 1 << 20,
            },
            Some(1),
        )
        .unwrap();
        assert_eq!(capped.chunk_batch_count, 1);
    }

    #[test]
    fn real_chunk_range_overflow_returns_route_error() {
        let plan = RealLargeChunkPlan {
            batch_count: 3,
            chunk_batch_count: 1,
            real_bytes_per_batch: u64::MAX,
            packed_bytes_per_batch: 4,
            full_complex_bytes_per_batch: u64::MAX,
            real_staging_size_bytes: u64::MAX,
            packed_staging_size_bytes: 4,
        };
        let mut ranges = plan.ranges();

        assert!(ranges.next().unwrap().is_ok());
        assert!(ranges.next().unwrap().is_ok());
        assert_eq!(
            ranges.next().unwrap(),
            Err(FftError::LargeChunkUnsupported {
                reason: "real chunk range offset overflowed u64",
                bytes_per_batch: u64::MAX,
                max_bind_bytes: u64::MAX,
            })
        );
    }

    #[test]
    fn real_routes_build_non_empty_stage_graphs() {
        let limits = LargePolicyLimits {
            max_storage_buffer_binding_size: 4096,
            max_buffer_size: 4096,
        };
        let config = FftConfig::new(16).with_normalization(Normalization::None);
        let packed = packed_shape_for(config.shape()).unwrap();
        let sizes = RealPlanSizes::new(&config, &packed).unwrap();

        let child_graph = test_child_c2c_graph(&sizes, limits);
        let r2c = build_real_normal_graph(
            "r2c-normal",
            "r2c-real-to-complex",
            "r2c-pack",
            &child_graph,
            RealEndpointOrder::RealToPacked,
            &sizes,
            limits,
            false,
        )
        .unwrap();
        assert!(r2c
            .stages()
            .iter()
            .any(|stage| stage.label() == "mixed-radix-stockham-stage"));

        let c2r = build_real_normal_graph(
            "c2r-normal",
            "c2r-unpack",
            "c2r-complex-to-real",
            &child_graph,
            RealEndpointOrder::PackedToReal,
            &sizes,
            limits,
            false,
        )
        .unwrap();
        assert!(c2r
            .stages()
            .iter()
            .any(|stage| stage.label() == "mixed-radix-stockham-stage"));

        let chunk_config = FftConfig::new(16)
            .with_batch(2)
            .with_normalization(Normalization::None);
        let chunk_packed = packed_shape_for(chunk_config.shape()).unwrap();
        let chunk_sizes = RealPlanSizes::new(&chunk_config, &chunk_packed).unwrap();
        let chunk =
            RealLargeChunkPlan::new(&chunk_sizes, chunk_config.batch() as u64, limits).unwrap();
        let child_config = FftConfig::new(16)
            .with_batch(chunk.chunk_batch_count as usize)
            .with_normalization(Normalization::None);
        let child_packed = packed_shape_for(child_config.shape()).unwrap();
        let child_sizes = RealPlanSizes::new(&child_config, &child_packed).unwrap();
        let child = build_real_normal_graph(
            "r2c-normal",
            "r2c-real-to-complex",
            "r2c-pack",
            &test_child_c2c_graph(&child_sizes, limits),
            RealEndpointOrder::RealToPacked,
            &child_sizes,
            limits,
            false,
        )
        .unwrap();
        let graph = build_r2c_chunk_graph(chunk, &child, limits).unwrap();
        assert!(!graph
            .stages()
            .iter()
            .any(|stage| stage.label() == "r2c-large-chunk-child"));
        assert!(graph
            .stages()
            .iter()
            .any(|stage| stage.label() == "r2c-real-to-complex"));
    }

    #[test]
    fn real_normal_graph_can_inline_child_c2c_route_stages() {
        let limits = LargePolicyLimits {
            max_storage_buffer_binding_size: 4096,
            max_buffer_size: 4096,
        };
        let config = FftConfig::new(16).with_normalization(Normalization::None);
        let packed = packed_shape_for(config.shape()).unwrap();
        let sizes = RealPlanSizes::new(&config, &packed).unwrap();

        let child = test_child_c2c_graph(&sizes, limits);

        let graph = build_real_normal_graph(
            "r2c-normal",
            "r2c-real-to-complex",
            "r2c-pack",
            &child,
            RealEndpointOrder::RealToPacked,
            &sizes,
            limits,
            false,
        )
        .unwrap();

        assert!(!graph
            .stages()
            .iter()
            .any(|stage| stage.label() == "r2c-child-c2c"));
        assert!(graph
            .stages()
            .iter()
            .any(|stage| stage.label() == "mixed-radix-workspace"));
        assert_eq!(
            graph
                .stages()
                .iter()
                .filter(|stage| stage.label() == "mixed-radix-stockham-stage")
                .count(),
            2
        );

        let scheduler = WindowScheduler::new(crate::runtime::window_scheduler::SchedulerLimits {
            max_storage_buffer_binding_size: 64,
            max_buffer_size: 4096,
            storage_alignment: 1,
            copy_alignment: 4,
        });
        let blockers = StageExecutor::new(&scheduler).graph_blockers("r2c", &graph);
        assert!(blockers.iter().any(|blocker| {
            blocker.kind == crate::diagnostics::FftBlockerKind::DeviceLimit
                && blocker.route.as_deref() == Some("mixed-radix")
                && blocker.stage.as_deref() == Some("mixed-radix-workspace")
        }));
    }

    #[test]
    fn real_large_chunk_graph_inlines_child_real_route_stages() {
        let limits = LargePolicyLimits {
            max_storage_buffer_binding_size: 256,
            max_buffer_size: 4096,
        };
        let config = FftConfig::new(16)
            .with_batch(5)
            .with_normalization(Normalization::None);
        let packed = packed_shape_for(config.shape()).unwrap();
        let sizes = RealPlanSizes::new(&config, &packed).unwrap();
        let chunk = RealLargeChunkPlan::new(&sizes, config.batch() as u64, limits).unwrap();
        let child_config = FftConfig::new(16)
            .with_batch(chunk.chunk_batch_count as usize)
            .with_normalization(Normalization::None);
        let child_packed = packed_shape_for(child_config.shape()).unwrap();
        let child_sizes = RealPlanSizes::new(&child_config, &child_packed).unwrap();
        let child_c2c = test_child_c2c_graph(&child_sizes, limits);
        let child_r2c = build_real_normal_graph(
            "r2c-normal",
            "r2c-real-to-complex",
            "r2c-pack",
            &child_c2c,
            RealEndpointOrder::RealToPacked,
            &child_sizes,
            limits,
            false,
        )
        .unwrap();

        let graph = build_r2c_chunk_graph(chunk, &child_r2c, limits).unwrap();

        assert!(!graph
            .stages()
            .iter()
            .any(|stage| stage.label() == "r2c-large-chunk-child"));
        assert_eq!(
            graph
                .stages()
                .iter()
                .filter(|stage| stage.label() == "r2c-real-to-complex")
                .count(),
            3
        );
        assert_eq!(
            graph
                .stages()
                .iter()
                .filter(|stage| stage.label() == "mixed-radix-stockham-stage")
                .count(),
            6
        );
        assert!(graph.stages().iter().any(|stage| {
            matches!(
                stage,
                LargeStage::Kernel {
                    label: "r2c-real-to-complex",
                    output: LogicalRange {
                        buffer: LogicalBufferId::Stage(2),
                        ..
                    },
                    ..
                }
            )
        }));

        let child_c2r = build_real_normal_graph(
            "c2r-normal",
            "c2r-unpack",
            "c2r-complex-to-real",
            &test_child_c2c_graph(&child_sizes, limits),
            RealEndpointOrder::PackedToReal,
            &child_sizes,
            limits,
            false,
        )
        .unwrap();
        let graph = build_c2r_chunk_graph(chunk, &child_c2r, limits).unwrap();
        assert!(!graph
            .stages()
            .iter()
            .any(|stage| stage.label() == "c2r-large-chunk-child"));
        assert_eq!(
            graph
                .stages()
                .iter()
                .filter(|stage| stage.label() == "c2r-complex-to-real")
                .count(),
            3
        );
    }

    #[test]
    fn rejects_real_large_chunk_when_one_batch_exceeds_limit() {
        let config = FftConfig::new(16)
            .with_batch(2)
            .with_normalization(Normalization::None);
        let packed = packed_shape_for(config.shape()).unwrap();
        let sizes = RealPlanSizes::new(&config, &packed).unwrap();
        assert_eq!(
            RealLargeChunkPlan::new(
                &sizes,
                config.batch() as u64,
                LargePolicyLimits {
                    max_storage_buffer_binding_size: 16 * COMPLEX_F32_BYTES - 4,
                    max_buffer_size: 1 << 20,
                },
            )
            .unwrap_err(),
            FftError::LargeChunkUnsupported {
                reason: "one batch exceeds the maximum storage-buffer binding size",
                bytes_per_batch: 16 * COMPLEX_F32_BYTES,
                max_bind_bytes: 16 * COMPLEX_F32_BYTES - 4,
            }
        );
    }

    #[test]
    fn validates_real_direction_and_axes() {
        assert_eq!(
            validate_real_config(&FftConfig::inverse(16), RealTransform::R2c).unwrap_err(),
            FftError::InvalidRealTransformDirection {
                transform: "r2c",
                expected: "forward",
                actual: "inverse",
            }
        );
        assert_eq!(
            validate_real_config(&FftConfig::new(16), RealTransform::C2r).unwrap_err(),
            FftError::InvalidRealTransformDirection {
                transform: "c2r",
                expected: "inverse",
                actual: "forward",
            }
        );
        assert_eq!(
            validate_real_config(
                &FftConfig::new_nd([4, 3]).with_axes([1]),
                RealTransform::R2c,
            )
            .unwrap_err(),
            FftError::UnsupportedRealAxes {
                expected: vec![0, 1],
                actual: vec![1],
            }
        );
    }

    #[test]
    fn real_policy_limits_honor_tuning_and_explicit_caps() {
        let mut device_limits = wgpu::Limits::default();
        device_limits.max_storage_buffer_binding_size = 8_192;
        device_limits.max_buffer_size = 16_384;
        let config = FftConfig::new(16).with_tuning(
            FftTuning::default()
                .with_max_storage_buffer_binding_size(4_096u64)
                .with_max_buffer_size(12_288u64),
        );

        assert_eq!(
            effective_real_policy_limits_for_device_limits(&device_limits, &config, None),
            LargePolicyLimits {
                max_storage_buffer_binding_size: 4_096,
                max_buffer_size: 12_288,
            }
        );
        assert_eq!(
            effective_real_policy_limits_for_device_limits(
                &device_limits,
                &config,
                Some(LargePolicyLimits {
                    max_storage_buffer_binding_size: 2_048,
                    max_buffer_size: 8_192,
                }),
            ),
            LargePolicyLimits {
                max_storage_buffer_binding_size: 2_048,
                max_buffer_size: 8_192,
            }
        );
    }

    #[test]
    fn real_transforms_reject_forced_large_routes_structurally() {
        assert!(validate_real_route_tuning(&FftConfig::new(16)).is_ok());
        for route in [
            FftLargeRoute::ForceChunk,
            FftLargeRoute::ForceFourStep,
            FftLargeRoute::ForceSegmented,
        ] {
            let config =
                FftConfig::new(16).with_tuning(FftTuning::default().with_large_route(route));
            assert!(matches!(
                validate_real_route_tuning(&config),
                Err(FftError::InvalidTuning {
                    kind: FftTuningErrorKind::UnsupportedForTransform,
                    field: "large_route",
                    ..
                })
            ));
        }
    }

    #[test]
    fn real_strided_copy_rejects_kind_format_mismatch() {
        assert_eq!(
            validate_real_strided_kind(
                RealKernelKind::PackComplexStrided,
                FftEndpointFormat::RealF32
            ),
            Err(FftError::LargeGraphStageUnsupported {
                stage: "real-strided-kernel-kind",
                reason: "real strided copy kernel kind does not match endpoint format",
            })
        );
        assert_eq!(
            validate_real_strided_kind(
                RealKernelKind::UnpackRealStrided,
                FftEndpointFormat::PackedComplexF32,
            ),
            Err(FftError::LargeGraphStageUnsupported {
                stage: "real-strided-kernel-kind",
                reason: "real strided copy kernel kind does not match endpoint format",
            })
        );
        assert!(validate_real_strided_kind(
            RealKernelKind::PackRealStrided,
            FftEndpointFormat::RealF32,
        )
        .is_ok());
        assert!(validate_real_strided_kind(
            RealKernelKind::UnpackComplexStrided,
            FftEndpointFormat::PackedComplexF32,
        )
        .is_ok());
    }

    #[test]
    fn real_shader_key_validation_rejects_invalid_shapes() {
        let mut rank_mismatch =
            RealStageKey::new(RealKernelKind::PackR2c, &[16], DEFAULT_WORKGROUP_SIZE);
        rank_mismatch.rank = 2;
        assert_eq!(
            validate_real_stage_key(&rank_mismatch),
            Err(FftError::LargeGraphStageUnsupported {
                stage: "real-shader-key",
                reason: "real shader key rank does not match dimensions",
            })
        );
        assert_eq!(
            validate_real_stage_key(&RealStageKey::new(
                RealKernelKind::PackR2c,
                &[],
                DEFAULT_WORKGROUP_SIZE
            )),
            Err(FftError::LargeGraphStageUnsupported {
                stage: "real-shader-key",
                reason: "real pack/unpack shader key requires a non-empty shape",
            })
        );
        assert_eq!(
            validate_real_stage_key(&RealStageKey::new(
                RealKernelKind::UnpackC2r,
                &[16, 0],
                DEFAULT_WORKGROUP_SIZE,
            )),
            Err(FftError::LargeGraphStageUnsupported {
                stage: "real-shader-key",
                reason: "real pack/unpack shader key dimensions must be non-zero",
            })
        );
        assert_eq!(
            validate_real_stage_key(&RealStageKey::new(
                RealKernelKind::PackRealStrided,
                &[16],
                DEFAULT_WORKGROUP_SIZE,
            )),
            Err(FftError::LargeGraphStageUnsupported {
                stage: "real-shader-key",
                reason: "real strided shader key must not carry transform dimensions",
            })
        );
        assert!(validate_real_stage_key(&RealStageKey::new(
            RealKernelKind::PackR2c,
            &[16],
            DEFAULT_WORKGROUP_SIZE,
        ))
        .is_ok());
        assert!(validate_real_stage_key(&RealStageKey::new(
            RealKernelKind::PackRealStrided,
            &[],
            DEFAULT_WORKGROUP_SIZE,
        ))
        .is_ok());
    }

    #[test]
    fn generated_real_wgsl_contains_expected_shape_constants() {
        let pack = generate_real_wgsl_for_key(&RealStageKey::new(
            RealKernelKind::PackR2c,
            &[17, 4],
            DEFAULT_WORKGROUP_SIZE,
        ));
        assert!(pack.contains("const OUT_TOTAL_PER_BATCH: u32 = 36u;"));

        let unpack = generate_real_wgsl_for_key(&RealStageKey::new(
            RealKernelKind::UnpackC2r,
            &[17, 4],
            DEFAULT_WORKGROUP_SIZE,
        ));
        assert!(unpack.contains("const NX: u32 = 17u;"));
        assert!(unpack.contains("const IN_NX: u32 = 9u;"));
    }

    #[test]
    fn real_shader_keys_and_wgsl_include_tuned_workgroup_size() {
        let default_key =
            RealStageKey::new(RealKernelKind::RealToComplex, &[16], DEFAULT_WORKGROUP_SIZE);
        assert!(default_key.stable_key().ends_with("workgroup=64"));

        let tuned_key = RealStageKey::new(RealKernelKind::RealToComplex, &[16], 128);
        assert!(tuned_key.stable_key().ends_with("workgroup=128"));
        assert!(generate_real_wgsl_for_key(&tuned_key).contains("@workgroup_size(128, 1, 1)"));
    }
}
