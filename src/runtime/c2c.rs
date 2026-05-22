use bytemuck::{Pod, Zeroable};

use crate::config::{FftConfig, FftDirection, FftPrecision, Normalization};
use crate::device::device_supports_precision;
use crate::error::{FftError, Result};
use crate::math::{to_interleaved_f32, DoubleFloat};
use crate::runtime::axis_plan::{
    AxisLayout, AxisPlan, AxisPlanConfig, AxisPrecision, AxisStageKind, AxisTwiddleLutPool,
};
use crate::runtime::axis_policy::{resolve_axis_kinds_for_axes, AxisKind};
use crate::runtime::bluestein_axis::{
    bluestein_bfft, bluestein_chirp, bluestein_convolution_length,
    fused_bluestein_supported_by_limits, BluesteinAxis, BluesteinAxisConfig,
};
use crate::runtime::buffer_view::{BufferLayout, BufferView, FftIoView};
use crate::runtime::dispatch::{max_workgroups_per_dimension, split_workgroups};
use crate::runtime::four_step::FourStepC2cPlan;
use crate::runtime::large_bridge::{plan_large_bridge, LargeBridgePlan, LargeBridgeRoute};
use crate::runtime::large_chunk::LargeChunkPlan;
use crate::runtime::large_graph::{
    ElementFormat, HelperBufferRange, LargeExecutionGraph, LargeExecutionPlan, LargeStage,
    LogicalBufferId, LogicalRange, StageRequirements,
};
use crate::runtime::large_policy::{
    line_bytes_for_axis_len, non_mixed_axis_window_supported,
    resolve_large_routing_policy_with_complex_element_bytes, LargeExecutionKind, LargeFactorSplit,
    LargePolicyLimits, LargeRouteMode, LargeRoutingPolicy, LargeRoutingPolicyInput,
};
use crate::runtime::logical_io::{FftEndpointFormat, FftLogicalView};
use crate::runtime::nd_wgsl::{format_wgsl_f32, stride_for_axis, wgsl_line_base_fn};
use crate::runtime::pipeline_cache::{
    with_device_pipeline_cache, BridgeKernelKind, BridgeStageKey, C2cSmoothKernelKind,
    C2cSmoothStageKey, C2cStridedKernelKind, C2cStridedStageKey, ComputePipelineCacheKey,
    PipelineLayoutCacheKey, ShaderCacheKey,
};
use crate::runtime::rader_axis::{
    fused_rader_supported_by_limits, rader_bfft, rader_convolution_length, rader_permutation,
    RaderAxis, RaderAxisConfig,
};
use crate::runtime::segmented_volume::{
    validate_segmented_burst_depth, SegmentedVolumeC2cPlan, DEFAULT_SEGMENTED_BURST_DEPTH,
};
use crate::runtime::smooth_decompose::{
    MixedAxisStep, SmoothAxisStep, SmoothDecompositionPlan, SmoothDecompositionStep,
};
use crate::runtime::stage_executor::StageExecutor;
#[cfg(test)]
use crate::runtime::twiddle::twiddle_lut_f32;
use crate::runtime::twiddle::{
    create_twiddle_lut_buffer, create_twiddle_lut_buffer_for_len_with_precision,
    two_level_twiddle_lut_f32,
};
use crate::runtime::window_scheduler::{strided_span_elements, WindowScheduler};

const WORKGROUP_SIZE: u32 = 64;
const COMPLEX_F32_BYTES: u64 = 8;

fn validate_c2c_precision_feature(device: &wgpu::Device, config: &FftConfig) -> Result<()> {
    if !device_supports_precision(device, config.precision()) {
        return Err(FftError::PrecisionUnsupported {
            requested: config.precision(),
            route: "c2c",
            reason: "device-missing-shader-f64",
        });
    }
    Ok(())
}

fn extended_precision_route_error(
    config: &FftConfig,
    axis_kinds: &[AxisKind],
    limits: LargePolicyLimits,
    route: C2cRoute,
    compute_limits: &wgpu::Limits,
) -> Result<()> {
    let precision = config.precision();
    if precision == FftPrecision::F32 {
        return Ok(());
    }
    if precision == FftPrecision::Df64
        && !matches!(route, C2cRoute::DirectDft | C2cRoute::MixedRadix)
    {
        let reason = match route {
            C2cRoute::Rader => "rader-df64-not-implemented",
            C2cRoute::Bluestein => "bluestein-df64-not-implemented",
            C2cRoute::AxisSequence => "axis-sequence-prime-df64-not-implemented",
            C2cRoute::DirectDft | C2cRoute::MixedRadix => unreachable!(),
        };
        return Err(FftError::PrecisionUnsupported {
            requested: precision,
            route: route.as_str(),
            reason,
        });
    }
    let required_bytes = config.required_buffer_size_bytes()?;
    let bytes_per_batch = bytes_per_batch(config)?;
    let prime_helper_bytes =
        extended_precision_prime_max_binding_bytes(config, axis_kinds, compute_limits)?;
    let normal_binding_bytes = required_bytes.max(prime_helper_bytes);
    let needs_large_mode = normal_binding_bytes > limits.max_storage_buffer_binding_size
        || normal_binding_bytes > limits.max_buffer_size;
    let needs_out_of_core = needs_large_mode
        && config.shape().len() >= 2
        && bytes_per_batch > limits.max_storage_buffer_binding_size;
    let four_step_eligible =
        needs_out_of_core && lightweight_four_step_eligible(config, axis_kinds, limits)?;
    if !needs_large_mode {
        return Ok(());
    }
    let (route, reason) = if !four_step_eligible {
        (
            "large-chunk",
            match precision {
                FftPrecision::F64 => "large-chunk-f64-not-implemented",
                FftPrecision::Df64 => "large-chunk-df64-not-implemented",
                FftPrecision::F32 => unreachable!(),
            },
        )
    } else if required_bytes > limits.max_buffer_size
        && axis_kinds.iter().all(|kind| *kind == AxisKind::Mixed)
    {
        (
            "segmented-full-volume",
            match precision {
                FftPrecision::F64 => "segmented-volume-f64-not-implemented",
                FftPrecision::Df64 => "segmented-volume-df64-not-implemented",
                FftPrecision::F32 => unreachable!(),
            },
        )
    } else if required_bytes > limits.max_buffer_size {
        (
            "large-chunk",
            match precision {
                FftPrecision::F64 => "large-chunk-f64-not-implemented",
                FftPrecision::Df64 => "large-chunk-df64-not-implemented",
                FftPrecision::F32 => unreachable!(),
            },
        )
    } else {
        (
            "out-of-core-four-step",
            match precision {
                FftPrecision::F64 => "four-step-f64-not-implemented",
                FftPrecision::Df64 => "four-step-df64-not-implemented",
                FftPrecision::F32 => unreachable!(),
            },
        )
    };
    Err(FftError::PrecisionUnsupported {
        requested: precision,
        route,
        reason,
    })
}

fn extended_precision_prime_max_binding_bytes(
    config: &FftConfig,
    axis_kinds: &[AxisKind],
    compute_limits: &wgpu::Limits,
) -> Result<u64> {
    let total_complex = config.total_complex_len_u32()? as usize;
    let precision = config.precision();
    debug_assert_ne!(precision, FftPrecision::F32);
    let axis_precision = precision.into();
    let complex_bytes = precision.complex_size_bytes();
    config
        .axes()
        .iter()
        .copied()
        .zip(axis_kinds.iter().copied())
        .try_fold(0u64, |maximum, (axis, kind)| {
            let n = config.shape()[axis];
            let m = match kind {
                AxisKind::Mixed => return Ok(maximum),
                AxisKind::Rader => rader_convolution_length(n)?,
                AxisKind::Bluestein => bluestein_convolution_length(n)?,
            };
            let fused = match kind {
                AxisKind::Mixed => false,
                AxisKind::Rader => fused_rader_supported_by_limits(
                    m,
                    axis_precision,
                    u64::from(compute_limits.max_compute_workgroup_storage_size),
                    compute_limits.max_compute_invocations_per_workgroup,
                    compute_limits.max_compute_workgroup_size_x,
                ),
                AxisKind::Bluestein => fused_bluestein_supported_by_limits(
                    m,
                    axis_precision,
                    u64::from(compute_limits.max_compute_workgroup_storage_size),
                    compute_limits.max_compute_invocations_per_workgroup,
                    compute_limits.max_compute_workgroup_size_x,
                ),
            };
            let helper_complex = if fused {
                m
            } else {
                total_complex
                    .checked_div(n)
                    .and_then(|lines| lines.checked_mul(m))
                    .ok_or(FftError::LengthTooLarge { len: total_complex })?
            };
            let helper_bytes = u64::try_from(helper_complex)
                .ok()
                .and_then(|value| value.checked_mul(complex_bytes))
                .ok_or(FftError::LengthTooLarge {
                    len: helper_complex,
                })?;
            Ok(maximum.max(helper_bytes))
        })
}

fn lightweight_four_step_eligible(
    config: &FftConfig,
    axis_kinds: &[AxisKind],
    limits: LargePolicyLimits,
) -> Result<bool> {
    let axis_lengths = config
        .axes()
        .iter()
        .map(|&axis| config.shape()[axis])
        .collect::<Vec<_>>();
    let complex_element_bytes = config.precision().complex_size_bytes();
    let line_bytes = axis_lengths
        .iter()
        .map(|&len| line_bytes_for_axis_len(len, complex_element_bytes))
        .collect::<Result<Vec<_>>>()?;
    if !four_step_route_shape_supported(
        config.shape().len(),
        config.axes().len(),
        axis_kinds,
        &line_bytes,
        limits.max_storage_buffer_binding_size,
    ) {
        return Ok(false);
    }

    Ok(axis_kinds.iter().zip(axis_lengths).zip(line_bytes).all(
        |((&kind, axis_len), line_bytes)| match kind {
            AxisKind::Mixed => line_bytes <= limits.max_buffer_size,
            AxisKind::Rader | AxisKind::Bluestein => non_mixed_axis_window_supported(
                kind,
                axis_len,
                line_bytes,
                limits.max_storage_buffer_binding_size,
                limits.max_buffer_size,
            ),
        },
    ))
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct DirectParams {
    len: u32,
    inverse: u32,
    scale: f32,
    _pad: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct DirectParamsF64 {
    len: u32,
    inverse: u32,
    scale: f64,
    _pad: [u32; 2],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct DirectParamsDf64 {
    len: u32,
    inverse: u32,
    scale_hi: f32,
    scale_lo: f32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct StridedCopyParams {
    total_complex: u32,
    logical_per_batch: u32,
    element_offset: u32,
    element_stride: u32,
    batch_stride: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct SmoothTwiddleParams {
    total_complex: u32,
    chunk_inner: u32,
    chunk_outer: u32,
    outer: u32,
    n1_start: u32,
    k1_start: u32,
    total_len: u32,
    inverse: u32,
    scale: f32,
    inner: u32,
    lut_shift: u32,
    lut_mask: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct SmoothAxisLineCopyParams {
    total_complex: u32,
    input_base: u32,
    output_base: u32,
    stride: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct SmoothAxisChunkCopyParams {
    total_complex: u32,
    input_base: u32,
    output_base: u32,
    stride: u32,
    inner: u32,
    outer: u32,
    chunk_inner: u32,
    chunk_outer: u32,
    offset_start: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct BridgeKernelParams {
    line_count: u32,
    line_offset: u32,
    t_offset: u32,
    t_count: u32,
    input_base: u32,
    output_base: u32,
    aux0_base: u32,
    aux1_base: u32,
    aux2_base: u32,
    aux3_base: u32,
    stride: u32,
    _pad0: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C2cRoute {
    DirectDft,
    MixedRadix,
    Rader,
    Bluestein,
    AxisSequence,
}

impl C2cRoute {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::DirectDft => "direct-dft",
            Self::MixedRadix => "mixed-radix",
            Self::Rader => "rader",
            Self::Bluestein => "bluestein",
            Self::AxisSequence => "axis-sequence",
        }
    }

    pub(crate) const fn graph_label(self) -> &'static str {
        match self {
            Self::DirectDft => "direct-dft-kernel",
            Self::MixedRadix => "mixed-radix-stockham",
            Self::Rader => "rader-axis",
            Self::Bluestein => "bluestein-axis",
            Self::AxisSequence => "axis-sequence",
        }
    }
}

pub struct C2cPlan {
    config: FftConfig,
    factors: Vec<usize>,
    axis_factors: Vec<Vec<usize>>,
    axis_kinds: Vec<AxisKind>,
    large_routing_policy: LargeRoutingPolicy,
    route: C2cRoute,
    execution: C2cExecution,
}

enum C2cExecution {
    Normal(C2cRouteImpl),
    LargeChunk(LargeChunkC2cPlan),
    SmoothDecomposition(SmoothDecompositionC2cPlan),
    LargeBridge(LargeBridgeC2cPlan),
    LargeAxisSequence(LargeAxisSequenceC2cPlan),
    FourStep(FourStepC2cPlan),
    SegmentedVolume(SegmentedVolumeC2cPlan),
}

enum C2cRouteImpl {
    DirectDft(DirectDftPlan),
    MixedRadix(AxisPlan),
    Rader(RaderAxis),
    Bluestein(BluesteinAxis),
    AxisSequence(AxisSequencePlan),
}

struct LargeChunkC2cPlan {
    child: Box<C2cPlan>,
    plan: LargeChunkPlan,
    graph_plan: LargeExecutionPlan,
    input_stage: wgpu::Buffer,
    output_stage: wgpu::Buffer,
}

struct LargeAxisSequenceC2cPlan {
    children: Vec<C2cPlan>,
    graph_plan: LargeExecutionPlan,
    temp_buffer: Option<wgpu::Buffer>,
    required_buffer_size_bytes: u64,
}

pub(crate) struct WindowedPrimeBridge {
    plan: LargeBridgeC2cPlan,
    required_bytes: u64,
    graph: LargeExecutionGraph,
    factor_splits: Vec<LargeFactorSplit>,
}

enum LargeBridgeC2cPlan {
    Rader(RaderBridgeC2cPlan),
    Bluestein(BluesteinBridgeC2cPlan),
}

struct RaderBridgeC2cPlan {
    shape: Vec<usize>,
    axis: usize,
    n: usize,
    l: usize,
    m: usize,
    lines: u64,
    stride_complex: u64,
    limits: LargePolicyLimits,
    child_forward: Box<C2cPlan>,
    child_inverse: Box<C2cPlan>,
    perm_buffer: wgpu::Buffer,
    bfft_buffer: wgpu::Buffer,
    sum_buffer: wgpu::Buffer,
    x0_buffer: wgpu::Buffer,
    work_buffer: wgpu::Buffer,
    fft_buffer: wgpu::Buffer,
    keys: BridgePipelineKeys,
}

struct BluesteinBridgeC2cPlan {
    shape: Vec<usize>,
    axis: usize,
    n: usize,
    m: usize,
    lines: u64,
    stride_complex: u64,
    limits: LargePolicyLimits,
    child_forward: Box<C2cPlan>,
    child_inverse: Box<C2cPlan>,
    chirp_buffer: wgpu::Buffer,
    bfft_buffer: wgpu::Buffer,
    work_buffer: wgpu::Buffer,
    fft_buffer: wgpu::Buffer,
    keys: BridgePipelineKeys,
}

struct BridgePipelineKeys {
    rader_sum_init: Option<ComputePipelineCacheKey>,
    rader_sum_accumulate: Option<ComputePipelineCacheKey>,
    rader_pack: Option<ComputePipelineCacheKey>,
    rader_mul: Option<ComputePipelineCacheKey>,
    rader_write_y0: Option<ComputePipelineCacheKey>,
    rader_post: Option<ComputePipelineCacheKey>,
    bluestein_pack: Option<ComputePipelineCacheKey>,
    bluestein_mul: Option<ComputePipelineCacheKey>,
    bluestein_post: Option<ComputePipelineCacheKey>,
}

struct SmoothDecompositionC2cPlan {
    plan: SmoothDecompositionPlan,
    graph_plan: LargeExecutionPlan,
    steps: Vec<SmoothExecutionStep>,
    temp_buffer: Option<wgpu::Buffer>,
    axis_twiddle_lut_storage_bytes: u64,
}

enum SmoothExecutionStep {
    Mixed(MixedAxisExecution),
    Smooth(SmoothAxisExecution),
}

struct MixedAxisExecution {
    step: MixedAxisStep,
    plan: AxisPlan,
    line_input: wgpu::Buffer,
    line_output: wgpu::Buffer,
}

struct SmoothAxisExecution {
    step: SmoothAxisStep,
    phase1: SmoothPhaseExecution,
    phase2: SmoothPhaseExecution,
    phase1_input: wgpu::Buffer,
    phase1_output: wgpu::Buffer,
    phase2_input: wgpu::Buffer,
    phase2_output: wgpu::Buffer,
    twiddle_coarse: wgpu::Buffer,
    twiddle_fine: wgpu::Buffer,
    twiddle_shift: u32,
    twiddle_mask: u32,
    scale: f32,
    inverse: bool,
}

enum SmoothPhaseExecution {
    Axis(AxisPlan),
    C2c(Box<C2cPlan>),
}

enum SmoothGraphStep {
    Mixed {
        step: SmoothMixedGraphStep,
        stage_kinds: Vec<AxisStageKind>,
        workspace_bytes: u64,
    },
    Smooth {
        step: SmoothAxisGraphStep,
        phase1: SmoothPhaseGraph,
        phase2: SmoothPhaseGraph,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SmoothMixedGraphStep {
    stride: u64,
    line_bytes: u64,
}

impl From<MixedAxisStep> for SmoothMixedGraphStep {
    fn from(step: MixedAxisStep) -> Self {
        Self {
            stride: step.stride(),
            line_bytes: step.line_bytes(),
        }
    }
}

impl SmoothMixedGraphStep {
    const fn stride(self) -> u64 {
        self.stride
    }

    const fn line_bytes(self) -> u64 {
        self.line_bytes
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SmoothAxisGraphStep {
    stride: u64,
    chunk_inner: u64,
    chunk_outer: u64,
    phase1_chunk_bytes: u64,
    phase2_chunk_bytes: u64,
}

impl From<SmoothAxisStep> for SmoothAxisGraphStep {
    fn from(step: SmoothAxisStep) -> Self {
        Self {
            stride: step.stride(),
            chunk_inner: step.chunk_inner(),
            chunk_outer: step.chunk_outer(),
            phase1_chunk_bytes: step.phase1_chunk_bytes(),
            phase2_chunk_bytes: step.phase2_chunk_bytes(),
        }
    }
}

impl SmoothAxisGraphStep {
    const fn stride(self) -> u64 {
        self.stride
    }

    const fn chunk_inner(self) -> u64 {
        self.chunk_inner
    }

    const fn chunk_outer(self) -> u64 {
        self.chunk_outer
    }

    const fn phase1_chunk_bytes(self) -> u64 {
        self.phase1_chunk_bytes
    }

    const fn phase2_chunk_bytes(self) -> u64 {
        self.phase2_chunk_bytes
    }
}

enum SmoothPhaseGraph {
    Axis {
        stage_kinds: Vec<AxisStageKind>,
        workspace_bytes: u64,
    },
    C2c(LargeExecutionGraph),
}

#[derive(Debug, Clone, Copy)]
struct SmoothChunkCopyRequest {
    base: u64,
    stride: u64,
    inner: u64,
    outer: u64,
    chunk_inner: u64,
    chunk_outer: u64,
    offset_start: u64,
}

struct DirectDftPlan {
    precision: AxisPrecision,
    pipeline_key: ComputePipelineCacheKey,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    params_buffer: wgpu::Buffer,
    twiddle_buffer: wgpu::Buffer,
    workgroups_x: u32,
}

fn route_impl_twiddle_lut_storage_bytes(route_impl: &C2cRouteImpl) -> u64 {
    match route_impl {
        C2cRouteImpl::DirectDft(plan) => plan.twiddle_buffer.size(),
        C2cRouteImpl::MixedRadix(plan) => plan.twiddle_lut_storage_bytes(),
        C2cRouteImpl::Rader(plan) => plan.twiddle_lut_storage_bytes(),
        C2cRouteImpl::Bluestein(plan) => plan.twiddle_lut_storage_bytes(),
        C2cRouteImpl::AxisSequence(plan) => {
            plan.steps
                .iter()
                .fold(plan.axis_twiddle_lut_storage_bytes, |bytes, step| {
                    bytes.saturating_add(match step {
                        AxisStep::Mixed(_) => 0,
                        AxisStep::Rader(plan) => plan.twiddle_lut_storage_bytes(),
                        AxisStep::Bluestein(plan) => plan.twiddle_lut_storage_bytes(),
                    })
                })
        }
    }
}

fn smooth_phase_twiddle_lut_storage_bytes(phase: &SmoothPhaseExecution) -> u64 {
    match phase {
        SmoothPhaseExecution::Axis(_) => 0,
        SmoothPhaseExecution::C2c(plan) => plan.twiddle_lut_storage_bytes(),
    }
}

fn smooth_decomposition_twiddle_lut_storage_bytes(plan: &SmoothDecompositionC2cPlan) -> u64 {
    plan.steps
        .iter()
        .fold(plan.axis_twiddle_lut_storage_bytes, |bytes, step| {
            bytes.saturating_add(match step {
                SmoothExecutionStep::Mixed(_) => 0,
                SmoothExecutionStep::Smooth(step) => {
                    smooth_phase_twiddle_lut_storage_bytes(&step.phase1)
                        .saturating_add(smooth_phase_twiddle_lut_storage_bytes(&step.phase2))
                        .saturating_add(step.twiddle_coarse.size())
                        .saturating_add(step.twiddle_fine.size())
                }
            })
        })
}

fn large_bridge_twiddle_lut_storage_bytes(plan: &LargeBridgeC2cPlan) -> u64 {
    match plan {
        LargeBridgeC2cPlan::Rader(plan) => plan
            .child_forward
            .twiddle_lut_storage_bytes()
            .saturating_add(plan.child_inverse.twiddle_lut_storage_bytes()),
        LargeBridgeC2cPlan::Bluestein(plan) => plan
            .child_forward
            .twiddle_lut_storage_bytes()
            .saturating_add(plan.child_inverse.twiddle_lut_storage_bytes()),
    }
}

struct C2cIoLayout<'a> {
    view: BufferView<'a>,
    layout: BufferLayout,
    contiguous: bool,
    physical_span_bytes: u64,
}

fn missing_segmented_strided_output_stage_error() -> FftError {
    FftError::LargeGraphStageUnsupported {
        stage: "c2c-logical-output-stage",
        reason: "segmented+strided output requires a physical staging buffer",
    }
}

impl C2cPlan {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, config: FftConfig) -> Result<Self> {
        Self::new_with_large_policy_limits(device, queue, config, None)
    }

    #[doc(hidden)]
    pub fn new_with_large_policy_limits_for_testing(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        config: FftConfig,
        limits: LargePolicyLimits,
    ) -> Result<Self> {
        Self::new_with_large_policy_limits(device, queue, config, Some(limits))
    }

    #[doc(hidden)]
    pub fn new_with_large_policy_limits_and_burst_depth_for_testing(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        config: FftConfig,
        limits: LargePolicyLimits,
        burst_depth: usize,
    ) -> Result<Self> {
        Self::new_with_large_policy_limits_and_burst_depth(
            device,
            queue,
            config,
            Some(limits),
            burst_depth,
        )
    }

    pub(crate) fn new_with_large_policy_limits(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        config: FftConfig,
        policy_limits: Option<LargePolicyLimits>,
    ) -> Result<Self> {
        Self::new_with_large_policy_limits_and_burst_depth(
            device,
            queue,
            config,
            policy_limits,
            DEFAULT_SEGMENTED_BURST_DEPTH,
        )
    }

    fn new_with_large_policy_limits_and_burst_depth(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        config: FftConfig,
        policy_limits: Option<LargePolicyLimits>,
        segmented_burst_depth: usize,
    ) -> Result<Self> {
        validate_segmented_burst_depth(segmented_burst_depth)?;
        config.validate()?;
        validate_c2c_precision_feature(device, &config)?;
        let device_policy_limits = LargePolicyLimits::from(&device.limits());
        let effective_policy_limits = policy_limits
            .unwrap_or(device_policy_limits)
            .componentwise_min(device_policy_limits);
        let policy_limits = Some(effective_policy_limits);
        let len = config.total_complex_len_u32()?;
        let route = select_route(&config);
        let axis_kinds = resolve_axis_kinds_for_axes(config.shape(), config.axes())?;
        extended_precision_route_error(
            &config,
            &axis_kinds,
            effective_policy_limits,
            route,
            &device.limits(),
        )?;
        let mut large_routing_policy =
            resolve_c2c_large_routing_policy(device, &config, &axis_kinds, policy_limits)?;

        let execution = match large_routing_policy.route_mode() {
            LargeRouteMode::Normal => C2cExecution::Normal(build_route_impl(
                device,
                queue,
                &config,
                &axis_kinds,
                route,
                len,
            )?),
            LargeRouteMode::LargeChunk => {
                let bytes_per_batch = bytes_per_batch(&config)?;
                let limits =
                    policy_limits.unwrap_or_else(|| LargePolicyLimits::from(&device.limits()));
                let chunk_plan =
                    match LargeChunkPlan::new(bytes_per_batch, config.batch() as u64, limits) {
                        Ok(plan) => plan,
                        Err(err) => {
                            if let Some(bridge_route) = large_bridge_route_for(route) {
                                let bridge_plan = plan_large_bridge(&config, bridge_route, limits)?;
                                large_routing_policy = large_routing_policy
                                    .with_execution_kind(match bridge_route {
                                        LargeBridgeRoute::Rader => LargeExecutionKind::RaderBridge,
                                        LargeBridgeRoute::Bluestein => {
                                            LargeExecutionKind::BluesteinBridge
                                        }
                                    })
                                    .with_diagnostics(
                                        Some(bridge_plan.axis()),
                                        bridge_plan.factor_splits(),
                                        bridge_plan.staging_bytes().to_vec(),
                                        None,
                                    );
                                let bridge = LargeBridgeC2cPlan::new(
                                    device,
                                    queue,
                                    &config,
                                    bridge_plan,
                                    limits,
                                )?;
                                let axis_factors = axis_kinds
                                    .iter()
                                    .enumerate()
                                    .map(|(index, kind)| match kind {
                                        AxisKind::Mixed => crate::runtime::factor_supported_length(
                                            config.shape()[config.axes()[index]],
                                        ),
                                        AxisKind::Rader | AxisKind::Bluestein => Ok(Vec::new()),
                                    })
                                    .collect::<Result<Vec<_>>>()?;
                                let factors = axis_factors.first().cloned().unwrap_or_default();
                                return Ok(Self {
                                    config: config.clone(),
                                    factors,
                                    axis_factors,
                                    axis_kinds,
                                    large_routing_policy,
                                    route,
                                    execution: C2cExecution::LargeBridge(bridge),
                                });
                            }

                            if route == C2cRoute::AxisSequence {
                                let sequence =
                                    LargeAxisSequenceC2cPlan::new(device, queue, &config, limits)?;
                                large_routing_policy = large_routing_policy
                                    .with_execution_kind(LargeExecutionKind::AxisDecomposition)
                                    .with_diagnostics(
                                        None,
                                        sequence.factor_splits(),
                                        sequence.staging_bytes(),
                                        None,
                                    );
                                let axis_factors =
                                    axis_factors_for_axis_kinds(&config, &axis_kinds)?;
                                let factors = axis_factors.first().cloned().unwrap_or_default();
                                return Ok(Self {
                                    config: config.clone(),
                                    factors,
                                    axis_factors,
                                    axis_kinds,
                                    large_routing_policy,
                                    route,
                                    execution: C2cExecution::LargeAxisSequence(sequence),
                                });
                            }

                            if !axis_kinds.iter().all(|kind| *kind == AxisKind::Mixed) {
                                return Err(err);
                            }

                            let smooth_plan = SmoothDecompositionPlan::new(&config, limits)?;
                            large_routing_policy = large_routing_policy
                                .with_execution_kind(smooth_plan.execution_kind())
                                .with_diagnostics(
                                    smooth_plan.selected_axis(),
                                    smooth_plan.factor_splits(),
                                    smooth_plan.staging_bytes(),
                                    None,
                                );
                            let smooth = SmoothDecompositionC2cPlan::new(
                                device,
                                queue,
                                &config,
                                smooth_plan,
                                limits,
                            )?;
                            return Ok(Self {
                                config: config.clone(),
                                factors: crate::runtime::factor_supported_length(
                                    config.shape()[config.axes()[0]],
                                )?,
                                axis_factors: config
                                    .axes()
                                    .iter()
                                    .map(|&axis| {
                                        crate::runtime::factor_supported_length(
                                            config.shape()[axis],
                                        )
                                    })
                                    .collect::<Result<Vec<_>>>()?,
                                axis_kinds,
                                large_routing_policy,
                                route,
                                execution: C2cExecution::SmoothDecomposition(smooth),
                            });
                        }
                    };
                large_routing_policy = large_routing_policy.with_diagnostics(
                    None,
                    Vec::new(),
                    vec![chunk_plan.staging_size_bytes()],
                    None,
                );
                let child_batch = usize::try_from(chunk_plan.chunk_batch_count())
                    .map_err(|_| FftError::LengthTooLarge { len: usize::MAX })?;
                let child_config = config.clone().with_batch(child_batch);
                let child = Box::new(Self::new_with_large_policy_limits(
                    device,
                    queue,
                    child_config,
                    Some(limits),
                )?);
                debug_assert_eq!(
                    child.large_routing_policy().route_mode(),
                    LargeRouteMode::Normal
                );
                let input_stage = create_view_staging_buffer(
                    device,
                    "wgpu_fft.c2c.large_chunk.input_stage",
                    chunk_plan.staging_size_bytes(),
                    wgpu::BufferUsages::COPY_DST,
                )?;
                let output_stage = create_view_staging_buffer(
                    device,
                    "wgpu_fft.c2c.large_chunk.output_stage",
                    chunk_plan.staging_size_bytes(),
                    wgpu::BufferUsages::COPY_SRC,
                )?;
                let child_graph = child.execution_graph()?;
                let graph_plan = build_large_chunk_c2c_graph(
                    chunk_plan,
                    &child_graph,
                    limits,
                    device.limits().min_storage_buffer_offset_alignment,
                )?;
                C2cExecution::LargeChunk(LargeChunkC2cPlan {
                    child,
                    plan: chunk_plan,
                    graph_plan,
                    input_stage,
                    output_stage,
                })
            }
            LargeRouteMode::LargeOutOfCore => {
                let limits =
                    policy_limits.unwrap_or_else(|| LargePolicyLimits::from(&device.limits()));
                if config.required_buffer_size_bytes()? > limits.max_buffer_size {
                    if let Some((index, kind)) = axis_kinds
                        .iter()
                        .enumerate()
                        .find(|(_, kind)| **kind != AxisKind::Mixed)
                    {
                        let axis = config.axes()[index];
                        return Err(FftError::UnsupportedAxisKind {
                            axis,
                            len: config.shape()[axis],
                            kind: kind.as_str(),
                        });
                    }
                    let plan = SegmentedVolumeC2cPlan::new(
                        device,
                        queue,
                        &config,
                        limits,
                        segmented_burst_depth,
                    )?;
                    large_routing_policy = large_routing_policy
                        .with_execution_kind(LargeExecutionKind::SegmentedFullVolume)
                        .with_diagnostics(None, plan.factor_splits(), plan.staging_bytes(), None);
                    C2cExecution::SegmentedVolume(plan)
                } else {
                    let plan = FourStepC2cPlan::new(device, queue, &config, limits)?;
                    large_routing_policy = large_routing_policy
                        .with_execution_kind(LargeExecutionKind::OutOfCoreFourStep)
                        .with_diagnostics(None, plan.factor_splits(), plan.staging_bytes(), None);
                    C2cExecution::FourStep(plan)
                }
            }
        };
        let axis_factors = match &execution {
            C2cExecution::Normal(route_impl) => axis_factors_for_route_impl(&config, route_impl)?,
            C2cExecution::LargeChunk(plan) => plan.child.axis_factors().to_vec(),
            C2cExecution::LargeAxisSequence(plan) => plan.axis_factors(),
            C2cExecution::FourStep(_) | C2cExecution::SegmentedVolume(_) => config
                .axes()
                .iter()
                .zip(&axis_kinds)
                .map(|(&axis, kind)| match kind {
                    AxisKind::Mixed => {
                        crate::runtime::factor_supported_length(config.shape()[axis])
                    }
                    AxisKind::Rader | AxisKind::Bluestein => Ok(Vec::new()),
                })
                .collect::<Result<Vec<_>>>()?,
            C2cExecution::SmoothDecomposition(_) | C2cExecution::LargeBridge(_) => config
                .axes()
                .iter()
                .map(|&axis| crate::runtime::factor_supported_length(config.shape()[axis]))
                .collect::<Result<Vec<_>>>()?,
        };
        let factors = axis_factors.first().cloned().unwrap_or_default();

        Ok(Self {
            config,
            factors,
            axis_factors,
            axis_kinds,
            large_routing_policy,
            route,
            execution,
        })
    }

    pub fn config(&self) -> FftConfig {
        self.config.clone()
    }

    pub fn factors(&self) -> &[usize] {
        &self.factors
    }

    pub fn axis_factors(&self) -> &[Vec<usize>] {
        &self.axis_factors
    }

    pub fn axis_kinds(&self) -> &[AxisKind] {
        &self.axis_kinds
    }

    pub fn route(&self) -> C2cRoute {
        self.route
    }

    pub fn large_routing_policy(&self) -> &LargeRoutingPolicy {
        &self.large_routing_policy
    }

    pub fn workspace_size_bytes(&self) -> u64 {
        match &self.execution {
            C2cExecution::Normal(route_impl) => match route_impl {
                C2cRouteImpl::DirectDft(plan) => plan.workspace_size_bytes(),
                C2cRouteImpl::MixedRadix(plan) => plan.workspace_size_bytes(),
                C2cRouteImpl::Rader(plan) => plan.workspace_size_bytes(),
                C2cRouteImpl::Bluestein(plan) => plan.workspace_size_bytes(),
                C2cRouteImpl::AxisSequence(plan) => plan.workspace_size_bytes(),
            },
            C2cExecution::LargeChunk(_) => 0,
            C2cExecution::SmoothDecomposition(_) => 0,
            C2cExecution::LargeBridge(_) => 0,
            C2cExecution::LargeAxisSequence(_) => 0,
            C2cExecution::FourStep(_) | C2cExecution::SegmentedVolume(_) => 0,
        }
    }

    pub(crate) fn twiddle_lut_storage_bytes(&self) -> u64 {
        match &self.execution {
            C2cExecution::Normal(route_impl) => route_impl_twiddle_lut_storage_bytes(route_impl),
            C2cExecution::LargeChunk(plan) => plan.child.twiddle_lut_storage_bytes(),
            C2cExecution::SmoothDecomposition(plan) => {
                smooth_decomposition_twiddle_lut_storage_bytes(plan)
            }
            C2cExecution::LargeBridge(plan) => large_bridge_twiddle_lut_storage_bytes(plan),
            C2cExecution::LargeAxisSequence(plan) => {
                plan.children.iter().fold(0u64, |bytes, child| {
                    bytes.saturating_add(child.twiddle_lut_storage_bytes())
                })
            }
            C2cExecution::FourStep(plan) => plan.twiddle_lut_storage_bytes(),
            C2cExecution::SegmentedVolume(plan) => plan.twiddle_lut_storage_bytes(),
        }
    }

    pub fn required_buffer_size_bytes(&self) -> u64 {
        self.config
            .required_buffer_size_bytes()
            .expect("validated config must have a buffer size")
    }

    pub(crate) fn execution_graph(&self) -> Result<LargeExecutionGraph> {
        match &self.execution {
            C2cExecution::Normal(route_impl) => build_normal_c2c_graph_for_impl(
                route_impl,
                self.required_buffer_size_bytes(),
                self.workspace_size_bytes(),
                self.complex_element_format(),
                policy_limits(&self.large_routing_policy),
            ),
            C2cExecution::LargeChunk(plan) => Ok(plan.graph_plan.graph().clone()),
            C2cExecution::SmoothDecomposition(plan) => Ok(plan.graph_plan.graph().clone()),
            C2cExecution::LargeBridge(plan) => {
                plan.execution_graph(self.required_buffer_size_bytes())
            }
            C2cExecution::LargeAxisSequence(plan) => Ok(plan.graph_plan.graph().clone()),
            C2cExecution::FourStep(plan) => Ok(plan.graph_plan().graph().clone()),
            C2cExecution::SegmentedVolume(plan) => Ok(plan.graph_plan().graph().clone()),
        }
    }

    pub fn execute(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        input: &wgpu::Buffer,
        output: &wgpu::Buffer,
    ) {
        self.execute_views(
            device,
            encoder,
            BufferView::whole(input),
            BufferView::whole(output),
        )
        .expect("whole-buffer C2C execution should satisfy buffer view validation");
    }

    pub fn execute_with_workspace(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        input: &wgpu::Buffer,
        output: &wgpu::Buffer,
        workspace: &wgpu::Buffer,
    ) -> Result<()> {
        self.execute_views_with_workspace(
            device,
            encoder,
            BufferView::whole(input),
            BufferView::whole(output),
            BufferView::whole(workspace),
        )
    }

    pub fn execute_views(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        input: BufferView<'_>,
        output: BufferView<'_>,
    ) -> Result<()> {
        let input = self.validate_io_view(device, input)?;
        let output = self.validate_io_view(device, output)?;
        self.execute_views_impl(device, encoder, input, output, None)
    }

    pub fn execute_views_with_workspace(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        input: BufferView<'_>,
        output: BufferView<'_>,
        workspace: BufferView<'_>,
    ) -> Result<()> {
        if self.is_large_route() {
            return Err(FftError::LargeRouteWorkspaceUnsupported {
                route_mode: self.large_routing_policy.route_mode().as_str(),
            });
        }
        let input = self.validate_io_view(device, input)?;
        let output = self.validate_io_view(device, output)?;
        let workspace = self.validate_workspace_view(device, workspace)?;
        self.execute_views_impl(device, encoder, input, output, Some(workspace))
    }

    pub fn execute_io_views(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        input: FftIoView<'_>,
        output: FftIoView<'_>,
    ) -> Result<()> {
        let input = self.validate_io_layout(device, input)?;
        let output = self.validate_io_layout(device, output)?;
        if input.contiguous && output.contiguous {
            return self.execute_views(device, encoder, input.view, output.view);
        }
        self.execute_io_views_impl(device, encoder, input, output, None)
    }

    pub fn execute_io_views_with_workspace(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        input: FftIoView<'_>,
        output: FftIoView<'_>,
        workspace: BufferView<'_>,
    ) -> Result<()> {
        if self.is_large_route() {
            return Err(FftError::LargeRouteWorkspaceUnsupported {
                route_mode: self.large_routing_policy.route_mode().as_str(),
            });
        }
        let input = self.validate_io_layout(device, input)?;
        let output = self.validate_io_layout(device, output)?;
        let workspace = self.validate_workspace_view(device, workspace)?;
        self.execute_io_views_impl(device, encoder, input, output, Some(workspace))
    }

    pub fn execute_logical_views(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        input: FftLogicalView<'_>,
        output: FftLogicalView<'_>,
    ) -> Result<()> {
        let logical_per_batch = self.config.logical_complex_len()? as u64;
        let batch = self.config.batch() as u64;
        let scheduler = WindowScheduler::for_device(device);
        let endpoint_format = self.complex_endpoint_format();
        let input = scheduler.bind_logical_io(input, endpoint_format, logical_per_batch, batch)?;
        let output =
            scheduler.bind_logical_io(output, endpoint_format, logical_per_batch, batch)?;
        self.execute_io_views(
            device,
            encoder,
            input.into_c2c_io_view()?,
            output.into_c2c_io_view()?,
        )
    }

    pub fn execute_logical_views_with_workspace(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        input: FftLogicalView<'_>,
        output: FftLogicalView<'_>,
        workspace: BufferView<'_>,
    ) -> Result<()> {
        let logical_per_batch = self.config.logical_complex_len()? as u64;
        let batch = self.config.batch() as u64;
        let scheduler = WindowScheduler::for_device(device);
        let endpoint_format = self.complex_endpoint_format();
        let input = scheduler.bind_logical_io(input, endpoint_format, logical_per_batch, batch)?;
        let output =
            scheduler.bind_logical_io(output, endpoint_format, logical_per_batch, batch)?;
        self.execute_io_views_with_workspace(
            device,
            encoder,
            input.into_c2c_io_view()?,
            output.into_c2c_io_view()?,
            workspace,
        )
    }

    fn execute_views_impl(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        input: BufferView<'_>,
        output: BufferView<'_>,
        workspace: Option<BufferView<'_>>,
    ) -> Result<()> {
        self.validate_execution_graph(device)?;
        if workspace.is_some() && self.is_large_route() {
            return Err(FftError::LargeRouteWorkspaceUnsupported {
                route_mode: self.large_routing_policy.route_mode().as_str(),
            });
        }
        if let C2cExecution::FourStep(plan) = &self.execution {
            return plan.execute_views(device, encoder, input, output);
        }
        if let C2cExecution::SegmentedVolume(plan) = &self.execution {
            return plan.execute_views(device, encoder, input, output);
        }
        if let C2cExecution::LargeChunk(plan) = &self.execution {
            return plan.execute_views(device, encoder, input, output);
        }
        if let C2cExecution::SmoothDecomposition(plan) = &self.execution {
            return plan.execute_views(device, encoder, input, output);
        }
        if let C2cExecution::LargeBridge(plan) = &self.execution {
            return plan.execute_views(
                device,
                encoder,
                input,
                output,
                self.required_buffer_size_bytes(),
            );
        }
        if let C2cExecution::LargeAxisSequence(plan) = &self.execution {
            return plan.execute_views(device, encoder, input, output);
        }

        let required = self.required_buffer_size_bytes();
        let input_stage = if input.is_single_segment() {
            None
        } else {
            let buffer = create_view_staging_buffer(
                device,
                "wgpu_fft.c2c.segmented_input_stage",
                required,
                wgpu::BufferUsages::COPY_DST,
            )?;
            copy_view_to_buffer(device, encoder, &input, &buffer, 0, required)?;
            Some(buffer)
        };
        let output_stage = if output.is_single_segment() {
            None
        } else {
            Some(create_view_staging_buffer(
                device,
                "wgpu_fft.c2c.segmented_output_stage",
                required,
                wgpu::BufferUsages::COPY_SRC,
            )?)
        };

        let exec_input = if let Some(buffer) = input_stage.as_ref() {
            BufferView::whole(buffer).prefix(required)?
        } else {
            input.clone()
        };
        let exec_output = if let Some(buffer) = output_stage.as_ref() {
            BufferView::whole(buffer).prefix(required)?
        } else {
            output.clone()
        };

        let element_format = self.complex_element_format();
        validate_exact_storage_view(device, &exec_input, element_format)?;
        validate_exact_storage_view(device, &exec_output, element_format)?;
        self.execute_route_views(device, encoder, exec_input, exec_output, workspace)?;

        if let Some(buffer) = output_stage.as_ref() {
            copy_buffer_to_view(device, encoder, buffer, 0, &output, required)?;
        }

        Ok(())
    }

    fn execute_io_views_impl(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        input: C2cIoLayout<'_>,
        output: C2cIoLayout<'_>,
        workspace: Option<BufferView<'_>>,
    ) -> Result<()> {
        if self.config.precision() == FftPrecision::Df64
            && (!input.contiguous || !output.contiguous)
        {
            return Err(FftError::PrecisionUnsupported {
                requested: FftPrecision::Df64,
                route: "c2c-strided",
                reason: "strided-df64-not-implemented",
            });
        }
        if matches!(&self.execution, C2cExecution::SegmentedVolume(_)) {
            return Err(FftError::LargeGraphStageUnsupported {
                stage: "segmented-volume-logical-io",
                reason: "segmented full-volume execution requires a single zero-offset contiguous endpoint buffer",
            });
        }
        if matches!(&self.execution, C2cExecution::FourStep(_)) {
            return Err(FftError::LargeGraphStageUnsupported {
                stage: "four-step-logical-io",
                reason: "four-step execution does not yet support strided logical I/O",
            });
        }
        let required = self.required_buffer_size_bytes();
        let input_stage = if input.contiguous {
            if input.view.is_single_segment() {
                None
            } else {
                let buffer = create_view_staging_buffer(
                    device,
                    "wgpu_fft.c2c.layout.segmented_input_stage",
                    required,
                    wgpu::BufferUsages::COPY_DST,
                )?;
                copy_view_to_buffer(device, encoder, &input.view, &buffer, 0, required)?;
                Some(buffer)
            }
        } else {
            let physical_stage = if input.view.is_single_segment() {
                None
            } else {
                let buffer = create_view_staging_buffer(
                    device,
                    "wgpu_fft.c2c.layout.segmented_strided_input_physical_stage",
                    input.physical_span_bytes,
                    wgpu::BufferUsages::COPY_DST,
                )?;
                copy_view_to_buffer(
                    device,
                    encoder,
                    &input.view,
                    &buffer,
                    0,
                    input.physical_span_bytes,
                )?;
                Some(buffer)
            };
            let buffer = create_view_staging_buffer(
                device,
                "wgpu_fft.c2c.layout.strided_input_stage",
                required,
                wgpu::BufferUsages::COPY_SRC,
            )?;
            let strided_source = if let Some(buffer) = physical_stage.as_ref() {
                BufferView::whole(buffer).prefix(input.physical_span_bytes)?
            } else {
                input.view.clone()
            };
            dispatch_c2c_strided_copy(
                device,
                encoder,
                C2cStridedKernelKind::Pack,
                &strided_source,
                &BufferView::whole(&buffer).prefix(required)?,
                input.layout,
                self.logical_complex_len_u32()?,
                self.config.batch() as u32,
                self.config.precision().into(),
            )?;
            Some(buffer)
        };

        let output_stage = if output.contiguous {
            if output.view.is_single_segment() {
                None
            } else {
                Some(create_view_staging_buffer(
                    device,
                    "wgpu_fft.c2c.layout.segmented_output_stage",
                    required,
                    wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                )?)
            }
        } else {
            Some(create_view_staging_buffer(
                device,
                "wgpu_fft.c2c.layout.strided_output_stage",
                required,
                wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            )?)
        };
        let output_physical_stage = if !output.contiguous && !output.view.is_single_segment() {
            Some(create_view_staging_buffer(
                device,
                "wgpu_fft.c2c.layout.segmented_strided_output_physical_stage",
                output.physical_span_bytes,
                wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            )?)
        } else {
            None
        };

        let exec_input = if let Some(buffer) = input_stage.as_ref() {
            BufferView::whole(buffer).prefix(required)?
        } else {
            input.view.clone()
        };
        let exec_output = if let Some(buffer) = output_stage.as_ref() {
            BufferView::whole(buffer).prefix(required)?
        } else {
            output.view.clone()
        };

        let element_format = self.complex_element_format();
        validate_exact_storage_view(device, &exec_input, element_format)?;
        validate_exact_storage_view(device, &exec_output, element_format)?;
        self.execute_views_impl(device, encoder, exec_input, exec_output, workspace)?;

        if let Some(buffer) = output_stage.as_ref() {
            if output.contiguous {
                copy_buffer_to_view(device, encoder, buffer, 0, &output.view, required)?;
            } else {
                let strided_target = if output.view.is_single_segment() {
                    output.view.clone()
                } else if let Some(buffer) = output_physical_stage.as_ref() {
                    BufferView::whole(buffer).prefix(output.physical_span_bytes)?
                } else {
                    return Err(missing_segmented_strided_output_stage_error());
                };
                dispatch_c2c_strided_copy(
                    device,
                    encoder,
                    C2cStridedKernelKind::Unpack,
                    &BufferView::whole(buffer).prefix(required)?,
                    &strided_target,
                    output.layout,
                    self.logical_complex_len_u32()?,
                    self.config.batch() as u32,
                    self.config.precision().into(),
                )?;
                if let Some(physical_stage) = output_physical_stage.as_ref() {
                    copy_buffer_to_view(
                        device,
                        encoder,
                        physical_stage,
                        0,
                        &output.view,
                        output.physical_span_bytes,
                    )?;
                }
            }
        }

        Ok(())
    }

    fn execute_route_views(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        input: BufferView<'_>,
        output: BufferView<'_>,
        workspace: Option<BufferView<'_>>,
    ) -> Result<()> {
        match &self.execution {
            C2cExecution::Normal(route_impl) => match route_impl {
                C2cRouteImpl::DirectDft(plan) => plan.execute_views(device, encoder, input, output),
                C2cRouteImpl::MixedRadix(plan) => {
                    if let Some(workspace) = workspace {
                        plan.execute_views_with_workspace(device, encoder, input, output, workspace)
                    } else {
                        plan.execute_views(device, encoder, input, output)
                    }
                }
                C2cRouteImpl::Rader(plan) => plan.execute_views(device, encoder, input, output),
                C2cRouteImpl::Bluestein(plan) => plan.execute_views(device, encoder, input, output),
                C2cRouteImpl::AxisSequence(plan) => {
                    if let Some(workspace) = workspace {
                        plan.execute_views_with_workspace(device, encoder, input, output, workspace)
                    } else {
                        plan.execute_views(device, encoder, input, output)
                    }
                }
            },
            C2cExecution::LargeChunk(_) | C2cExecution::SmoothDecomposition(_) => {
                Err(FftError::LargeRouteWorkspaceUnsupported {
                    route_mode: self.large_routing_policy.route_mode().as_str(),
                })
            }
            C2cExecution::LargeBridge(_)
            | C2cExecution::LargeAxisSequence(_)
            | C2cExecution::FourStep(_)
            | C2cExecution::SegmentedVolume(_) => Err(FftError::LargeRouteWorkspaceUnsupported {
                route_mode: self.large_routing_policy.route_mode().as_str(),
            }),
        }
    }

    fn validate_io_view<'a>(
        &self,
        _device: &wgpu::Device,
        view: BufferView<'a>,
    ) -> Result<BufferView<'a>> {
        view.prefix(self.required_buffer_size_bytes())
    }

    fn validate_io_layout<'a>(
        &self,
        device: &wgpu::Device,
        io: FftIoView<'a>,
    ) -> Result<C2cIoLayout<'a>> {
        let (view, layout) = io.into_parts();
        let logical_per_batch = self.config.logical_complex_len()? as u64;
        let batch = self.config.batch() as u64;
        let span_complex = layout.required_complex_span(logical_per_batch, batch)?;
        let span_bytes = span_complex
            .checked_mul(self.config.precision().complex_size_bytes())
            .ok_or(FftError::LengthTooLarge { len: usize::MAX })?;
        if span_bytes > view.size() {
            return Err(FftError::BufferLayoutOutOfBounds {
                required_bytes: span_bytes,
                actual_bytes: view.size(),
            });
        }
        let contiguous = layout.is_contiguous_for(logical_per_batch, batch)?;
        let view = view.prefix(if contiguous {
            self.required_buffer_size_bytes()
        } else {
            span_bytes
        })?;
        if !contiguous && view.is_single_segment() {
            validate_exact_storage_view(device, &view, self.complex_element_format())?;
        }
        Ok(C2cIoLayout {
            view,
            layout,
            contiguous,
            physical_span_bytes: span_bytes,
        })
    }

    fn logical_complex_len_u32(&self) -> Result<u32> {
        self.config
            .logical_complex_len()?
            .try_into()
            .map_err(|_| FftError::LengthTooLarge { len: usize::MAX })
    }

    fn is_large_route(&self) -> bool {
        matches!(
            &self.execution,
            C2cExecution::LargeChunk(_)
                | C2cExecution::SmoothDecomposition(_)
                | C2cExecution::LargeBridge(_)
                | C2cExecution::LargeAxisSequence(_)
                | C2cExecution::FourStep(_)
                | C2cExecution::SegmentedVolume(_)
        )
    }

    fn validate_workspace_view<'a>(
        &self,
        device: &wgpu::Device,
        view: BufferView<'a>,
    ) -> Result<BufferView<'a>> {
        let required = self.workspace_size_bytes();
        if required == 0 {
            return view.prefix(0);
        }
        if !view.is_single_segment() {
            return Err(FftError::SegmentedWorkspaceUnsupported);
        }
        if view.size() < required {
            return Err(FftError::WorkspaceTooSmall {
                required,
                actual: view.size(),
            });
        }
        let view = view.prefix(required)?;
        if required > 0 {
            validate_exact_storage_view(device, &view, self.complex_element_format())?;
        }
        Ok(view)
    }

    fn validate_execution_graph(&self, device: &wgpu::Device) -> Result<()> {
        let graph = self.execution_graph()?;
        let scheduler = WindowScheduler::for_device(device);
        StageExecutor::new(&scheduler).validate_graph(&graph)
    }

    fn complex_element_format(&self) -> ElementFormat {
        match self.config.precision() {
            FftPrecision::F32 => ElementFormat::ComplexF32,
            FftPrecision::F64 => ElementFormat::ComplexF64,
            FftPrecision::Df64 => ElementFormat::ComplexDf64,
        }
    }

    fn complex_endpoint_format(&self) -> FftEndpointFormat {
        match self.config.precision() {
            FftPrecision::F32 => FftEndpointFormat::ComplexF32,
            FftPrecision::F64 => FftEndpointFormat::ComplexF64,
            FftPrecision::Df64 => FftEndpointFormat::ComplexDf64,
        }
    }
}

impl LargeChunkC2cPlan {
    fn execute_views(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        input: BufferView<'_>,
        output: BufferView<'_>,
    ) -> Result<()> {
        let scheduler = WindowScheduler::for_device(device);
        let executor = StageExecutor::new(&scheduler);
        executor.validate_graph(self.graph_plan.graph())?;
        for range in self.plan.ranges() {
            let range = range?;
            debug_assert!(range.batch_start < self.plan.batch_count());
            debug_assert!(range.batch_count <= self.plan.chunk_batch_count());
            debug_assert_eq!(
                range.byte_size,
                range.batch_count * self.plan.bytes_per_batch()
            );
            executor.copy_view_range_to_buffer(
                encoder,
                &input,
                range.byte_offset,
                &self.input_stage,
                0,
                range.byte_size,
            )?;
            self.child.execute_views(
                device,
                encoder,
                BufferView::whole(&self.input_stage).prefix(self.plan.staging_size_bytes())?,
                BufferView::whole(&self.output_stage).prefix(self.plan.staging_size_bytes())?,
            )?;
            executor.copy_buffer_to_view_range(
                encoder,
                &self.output_stage,
                0,
                &output,
                range.byte_offset,
                range.byte_size,
            )?;
        }
        Ok(())
    }
}

impl LargeAxisSequenceC2cPlan {
    fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        config: &FftConfig,
        limits: LargePolicyLimits,
    ) -> Result<Self> {
        let required_buffer_size_bytes = config.required_buffer_size_bytes()?;
        let mut children = Vec::with_capacity(config.axes().len());
        for (axis_index, &axis) in config.axes().iter().enumerate() {
            let final_axis = axis_index + 1 == config.axes().len();
            let child_config = config
                .clone()
                .with_axes([axis])
                .with_normalization(if final_axis {
                    config.normalization()
                } else {
                    Normalization::None
                });
            children.push(C2cPlan::new_with_large_policy_limits(
                device,
                queue,
                child_config,
                Some(limits),
            )?);
        }

        let temp_buffer = (children.len() > 1)
            .then(|| {
                create_view_staging_buffer(
                    device,
                    "wgpu_fft.c2c.large_axis_sequence.temp",
                    required_buffer_size_bytes,
                    wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                )
            })
            .transpose()?;
        let child_graphs = children
            .iter()
            .map(C2cPlan::execution_graph)
            .collect::<Result<Vec<_>>>()?;
        let graph_plan = build_large_axis_sequence_c2c_graph(
            &child_graphs,
            required_buffer_size_bytes,
            device.limits().min_storage_buffer_offset_alignment,
            limits,
        )?;

        Ok(Self {
            children,
            graph_plan,
            temp_buffer,
            required_buffer_size_bytes,
        })
    }

    fn axis_factors(&self) -> Vec<Vec<usize>> {
        self.children
            .iter()
            .map(|child| child.axis_factors().first().cloned().unwrap_or_default())
            .collect()
    }

    fn factor_splits(&self) -> Vec<crate::runtime::large_policy::LargeFactorSplit> {
        self.children
            .iter()
            .flat_map(|child| child.large_routing_policy().factor_splits.clone())
            .collect()
    }

    fn staging_bytes(&self) -> Vec<u64> {
        let mut bytes = Vec::new();
        if self.temp_buffer.is_some() {
            bytes.push(self.required_buffer_size_bytes);
        }
        for child in &self.children {
            bytes.extend(child.large_routing_policy().staging_bytes.iter().copied());
        }
        bytes.sort_unstable();
        bytes.dedup();
        bytes
    }

    fn execute_views(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        input: BufferView<'_>,
        output: BufferView<'_>,
    ) -> Result<()> {
        let scheduler = WindowScheduler::for_device(device);
        let executor = StageExecutor::new(&scheduler);
        executor.validate_graph(self.graph_plan.graph())?;
        let input = input.prefix(self.required_buffer_size_bytes)?;
        let output = output.prefix(self.required_buffer_size_bytes)?;
        let input_stage = if view_has_single_storage_range(&input) {
            None
        } else {
            let buffer = create_view_staging_buffer(
                device,
                "wgpu_fft.c2c.large_axis_sequence.segmented_input_stage",
                self.required_buffer_size_bytes,
                wgpu::BufferUsages::COPY_DST,
            )?;
            executor.copy_view_range_to_buffer(
                encoder,
                &input,
                0,
                &buffer,
                0,
                self.required_buffer_size_bytes,
            )?;
            Some(buffer)
        };
        let output_stage = if view_has_single_storage_range(&output) {
            None
        } else {
            Some(create_view_staging_buffer(
                device,
                "wgpu_fft.c2c.large_axis_sequence.segmented_output_stage",
                self.required_buffer_size_bytes,
                wgpu::BufferUsages::COPY_SRC,
            )?)
        };
        let exec_input = if let Some(buffer) = input_stage.as_ref() {
            BufferView::whole(buffer).prefix(self.required_buffer_size_bytes)?
        } else {
            input.clone()
        };
        let exec_output = if let Some(buffer) = output_stage.as_ref() {
            BufferView::whole(buffer).prefix(self.required_buffer_size_bytes)?
        } else {
            output.clone()
        };

        let mut src_slot = SequenceBufferSlot::Input;
        let mut dst_slot = if self.children.len() % 2 == 1 {
            SequenceBufferSlot::Output
        } else {
            SequenceBufferSlot::Temp
        };
        for (step_index, child) in self.children.iter().enumerate() {
            let src = self.resolve_buffer(src_slot, exec_input.clone(), exec_output.clone())?;
            let dst = self.resolve_buffer(dst_slot, exec_input.clone(), exec_output.clone())?;
            child.execute_views(device, encoder, src, dst)?;

            if step_index + 1 < self.children.len() {
                src_slot = dst_slot;
                dst_slot = next_sequence_destination(
                    "large-axis-sequence-buffer-flow",
                    src_slot,
                    "large axis sequence buffer flow attempted to use input as destination",
                )?;
            }
        }

        if let Some(buffer) = output_stage.as_ref() {
            executor.copy_buffer_to_view_range(
                encoder,
                buffer,
                0,
                &output,
                0,
                self.required_buffer_size_bytes,
            )?;
        }
        Ok(())
    }

    fn resolve_buffer<'a>(
        &'a self,
        slot: SequenceBufferSlot,
        input: BufferView<'a>,
        output: BufferView<'a>,
    ) -> Result<BufferView<'a>> {
        match slot {
            SequenceBufferSlot::Input => Ok(input),
            SequenceBufferSlot::Output => Ok(output),
            SequenceBufferSlot::Temp => {
                if let Some(buffer) = self.temp_buffer.as_ref() {
                    Ok(BufferView::whole(buffer))
                } else {
                    Err(sequence_temp_storage_error(
                        "large-axis-sequence-workspace",
                        "multi-step large AxisSequence requires temp storage",
                    ))
                }
            }
        }
    }
}

impl LargeBridgeC2cPlan {
    fn execution_graph(&self, required_bytes: u64) -> Result<LargeExecutionGraph> {
        match self {
            Self::Rader(plan) => plan.execution_graph(required_bytes),
            Self::Bluestein(plan) => plan.execution_graph(required_bytes),
        }
    }

    fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        config: &FftConfig,
        plan: LargeBridgePlan,
        limits: LargePolicyLimits,
    ) -> Result<Self> {
        match plan.route() {
            LargeBridgeRoute::Rader => Ok(Self::Rader(RaderBridgeC2cPlan::new(
                device, queue, config, plan, limits,
            )?)),
            LargeBridgeRoute::Bluestein => Ok(Self::Bluestein(BluesteinBridgeC2cPlan::new(
                device, queue, config, plan, limits,
            )?)),
        }
    }

    fn execute_views(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        input: BufferView<'_>,
        output: BufferView<'_>,
        required_bytes: u64,
    ) -> Result<()> {
        let scheduler = WindowScheduler::for_device(device);
        let executor = StageExecutor::new(&scheduler);
        let input = input.prefix(required_bytes)?;
        let output = output.prefix(required_bytes)?;
        let input_stage = if input.is_single_segment() {
            None
        } else {
            let buffer = create_view_staging_buffer(
                device,
                "wgpu_fft.c2c.large_bridge.segmented_input_stage",
                required_bytes,
                wgpu::BufferUsages::COPY_DST,
            )?;
            executor.copy_view_range_to_buffer(encoder, &input, 0, &buffer, 0, required_bytes)?;
            Some(buffer)
        };
        let output_stage = if output.is_single_segment() {
            None
        } else {
            Some(create_view_staging_buffer(
                device,
                "wgpu_fft.c2c.large_bridge.segmented_output_stage",
                required_bytes,
                wgpu::BufferUsages::COPY_SRC,
            )?)
        };
        let exec_input = if let Some(buffer) = input_stage.as_ref() {
            BufferView::whole(buffer).prefix(required_bytes)?
        } else {
            input.clone()
        };
        let exec_output = if let Some(buffer) = output_stage.as_ref() {
            BufferView::whole(buffer).prefix(required_bytes)?
        } else {
            output.clone()
        };

        match self {
            Self::Rader(plan) => plan.execute_views(device, encoder, exec_input, exec_output)?,
            Self::Bluestein(plan) => {
                plan.execute_views(device, encoder, exec_input, exec_output)?
            }
        }
        if let Some(buffer) = output_stage.as_ref() {
            executor.copy_buffer_to_view_range(encoder, buffer, 0, &output, 0, required_bytes)?;
        }
        Ok(())
    }
}

impl WindowedPrimeBridge {
    pub(crate) fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        config: &FftConfig,
        route: LargeBridgeRoute,
        limits: LargePolicyLimits,
    ) -> Result<Self> {
        let required_bytes = config.required_buffer_size_bytes()?;
        let bridge_plan = plan_large_bridge(config, route, limits)?;
        let factor_splits = bridge_plan.factor_splits();
        let plan = LargeBridgeC2cPlan::new(device, queue, config, bridge_plan, limits)?;
        let graph = plan.execution_graph(required_bytes)?;
        Ok(Self {
            plan,
            required_bytes,
            graph,
            factor_splits,
        })
    }

    pub(crate) fn execute_views(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        input: BufferView<'_>,
        output: BufferView<'_>,
    ) -> Result<()> {
        self.plan
            .execute_views(device, encoder, input, output, self.required_bytes)
    }

    pub(crate) fn execution_graph(&self) -> &LargeExecutionGraph {
        &self.graph
    }

    pub(crate) fn factor_splits(&self) -> &[LargeFactorSplit] {
        &self.factor_splits
    }

    pub(crate) fn twiddle_lut_storage_bytes(&self) -> u64 {
        large_bridge_twiddle_lut_storage_bytes(&self.plan)
    }
}

impl RaderBridgeC2cPlan {
    fn execution_graph(&self, required_bytes: u64) -> Result<LargeExecutionGraph> {
        let child_forward = self.child_forward.execution_graph()?;
        let child_inverse = self.child_inverse.execution_graph()?;
        build_rader_bridge_c2c_graph(
            required_bytes,
            [
                HelperBufferRange {
                    label: "rader-permutation-helper",
                    index: 0,
                    size_bytes: self.perm_buffer.size(),
                    format: ElementFormat::U32,
                },
                HelperBufferRange {
                    label: "rader-bfft-helper",
                    index: 1,
                    size_bytes: self.bfft_buffer.size(),
                    format: ElementFormat::ComplexF32,
                },
                HelperBufferRange {
                    label: "rader-sum-helper",
                    index: 2,
                    size_bytes: self.sum_buffer.size(),
                    format: ElementFormat::ComplexF32,
                },
                HelperBufferRange {
                    label: "rader-x0-helper",
                    index: 3,
                    size_bytes: self.x0_buffer.size(),
                    format: ElementFormat::ComplexF32,
                },
                HelperBufferRange {
                    label: "rader-work-helper",
                    index: 4,
                    size_bytes: self.work_buffer.size(),
                    format: ElementFormat::ComplexF32,
                },
                HelperBufferRange {
                    label: "rader-fft-helper",
                    index: 5,
                    size_bytes: self.fft_buffer.size(),
                    format: ElementFormat::ComplexF32,
                },
            ],
            &child_forward,
            &child_inverse,
            self.limits,
        )
    }
}

impl BluesteinBridgeC2cPlan {
    fn execution_graph(&self, required_bytes: u64) -> Result<LargeExecutionGraph> {
        let child_forward = self.child_forward.execution_graph()?;
        let child_inverse = self.child_inverse.execution_graph()?;
        build_bluestein_bridge_c2c_graph(
            required_bytes,
            [
                HelperBufferRange {
                    label: "bluestein-chirp-helper",
                    index: 0,
                    size_bytes: self.chirp_buffer.size(),
                    format: ElementFormat::ComplexF32,
                },
                HelperBufferRange {
                    label: "bluestein-bfft-helper",
                    index: 1,
                    size_bytes: self.bfft_buffer.size(),
                    format: ElementFormat::ComplexF32,
                },
                HelperBufferRange {
                    label: "bluestein-work-helper",
                    index: 2,
                    size_bytes: self.work_buffer.size(),
                    format: ElementFormat::ComplexF32,
                },
                HelperBufferRange {
                    label: "bluestein-fft-helper",
                    index: 3,
                    size_bytes: self.fft_buffer.size(),
                    format: ElementFormat::ComplexF32,
                },
            ],
            &child_forward,
            &child_inverse,
            self.limits,
        )
    }
}

fn build_rader_bridge_c2c_graph(
    required_bytes: u64,
    helpers: [HelperBufferRange; 6],
    child_forward: &LargeExecutionGraph,
    child_inverse: &LargeExecutionGraph,
    limits: LargePolicyLimits,
) -> Result<LargeExecutionGraph> {
    let mut graph = LargeExecutionGraph::new("c2c-rader-bridge");
    let io_req = graph_requirements(limits, 1, 0)?;
    graph.push_stage(
        LargeStage::HostWindow {
            label: "logical-input",
            range: c2c_range(LogicalBufferId::Input, 0, required_bytes)?,
        },
        io_req,
    )?;
    push_bridge_helper_windows(&mut graph, helpers, limits)?;

    let [perm, _bfft, sum, x0, work, fft] = helpers;
    let input = c2c_range(LogicalBufferId::Input, 0, required_bytes)?;
    let output = c2c_range(LogicalBufferId::Output, 0, required_bytes)?;
    let sum_range = helper_range_from_info(sum)?;
    let x0_range = helper_range_from_info(x0)?;
    let work_range = helper_range_from_info(work)?;
    let fft_range = helper_range_from_info(fft)?;

    push_bridge_windowed_kernel(
        &mut graph,
        "rader-bridge-sum-init",
        sum_range,
        x0_range,
        1,
        limits,
    )?;
    push_bridge_windowed_kernel(
        &mut graph,
        "rader-bridge-sum-accumulate",
        input,
        sum_range,
        work_items_for_bytes(required_bytes, ElementFormat::ComplexF32),
        limits,
    )?;
    push_bridge_windowed_kernel(
        &mut graph,
        "rader-bridge-pack",
        input,
        work_range,
        work_items_for_bytes(work.size_bytes, ElementFormat::ComplexF32),
        limits,
    )?;
    append_child_c2c_graph_with_bases(
        &mut graph,
        child_forward,
        work_range,
        fft_range,
        limits,
        1,
        16,
        0,
    )?;
    push_bridge_windowed_kernel(
        &mut graph,
        "rader-bridge-mul",
        fft_range,
        fft_range,
        work_items_for_bytes(fft.size_bytes, ElementFormat::ComplexF32),
        limits,
    )?;
    append_child_c2c_graph_with_bases(
        &mut graph,
        child_inverse,
        fft_range,
        work_range,
        limits,
        1,
        32,
        16,
    )?;
    push_bridge_windowed_kernel(
        &mut graph,
        "rader-bridge-write-y0",
        sum_range,
        output,
        1,
        limits,
    )?;
    push_bridge_windowed_kernel(
        &mut graph,
        "rader-bridge-post",
        work_range,
        output,
        work_items_for_bytes(
            work.size_bytes.max(perm.size_bytes),
            ElementFormat::ComplexF32,
        ),
        limits,
    )?;
    graph.push_stage(
        LargeStage::HostWindow {
            label: "logical-output",
            range: output,
        },
        io_req,
    )?;
    Ok(graph)
}

fn build_bluestein_bridge_c2c_graph(
    required_bytes: u64,
    helpers: [HelperBufferRange; 4],
    child_forward: &LargeExecutionGraph,
    child_inverse: &LargeExecutionGraph,
    limits: LargePolicyLimits,
) -> Result<LargeExecutionGraph> {
    let mut graph = LargeExecutionGraph::new("c2c-bluestein-bridge");
    let io_req = graph_requirements(limits, 1, 0)?;
    graph.push_stage(
        LargeStage::HostWindow {
            label: "logical-input",
            range: c2c_range(LogicalBufferId::Input, 0, required_bytes)?,
        },
        io_req,
    )?;
    push_bridge_helper_windows(&mut graph, helpers, limits)?;

    let [chirp, _bfft, work, fft] = helpers;
    let input = c2c_range(LogicalBufferId::Input, 0, required_bytes)?;
    let output = c2c_range(LogicalBufferId::Output, 0, required_bytes)?;
    let work_range = helper_range_from_info(work)?;
    let fft_range = helper_range_from_info(fft)?;

    push_bridge_windowed_kernel(
        &mut graph,
        "bluestein-bridge-pack",
        input,
        work_range,
        work_items_for_bytes(work.size_bytes, ElementFormat::ComplexF32),
        limits,
    )?;
    append_child_c2c_graph_with_bases(
        &mut graph,
        child_forward,
        work_range,
        fft_range,
        limits,
        1,
        16,
        0,
    )?;
    push_bridge_windowed_kernel(
        &mut graph,
        "bluestein-bridge-mul",
        fft_range,
        fft_range,
        work_items_for_bytes(fft.size_bytes, ElementFormat::ComplexF32),
        limits,
    )?;
    append_child_c2c_graph_with_bases(
        &mut graph,
        child_inverse,
        fft_range,
        work_range,
        limits,
        1,
        32,
        16,
    )?;
    push_bridge_windowed_kernel(
        &mut graph,
        "bluestein-bridge-post",
        work_range,
        output,
        work_items_for_bytes(
            work.size_bytes.max(chirp.size_bytes),
            ElementFormat::ComplexF32,
        ),
        limits,
    )?;
    graph.push_stage(
        LargeStage::HostWindow {
            label: "logical-output",
            range: output,
        },
        io_req,
    )?;
    Ok(graph)
}

fn push_bridge_helper_windows<I>(
    graph: &mut LargeExecutionGraph,
    helpers: I,
    limits: LargePolicyLimits,
) -> Result<()>
where
    I: IntoIterator<Item = HelperBufferRange>,
{
    for helper in helpers {
        let window_bytes = helper_window_size(helper, limits);
        graph.push_stage(
            LargeStage::HelperWindow {
                label: helper.label,
                range: helper_range(helper.index, 0, window_bytes, helper.format)?,
            },
            graph_requirements_covering(limits, 1, helper.size_bytes, helper.size_bytes)?,
        )?;
    }
    Ok(())
}

fn helper_window_size(helper: HelperBufferRange, limits: LargePolicyLimits) -> u64 {
    let element_bytes = helper.format.bytes_per_element();
    let max_window = limits
        .max_storage_buffer_binding_size
        .min(helper.size_bytes);
    let aligned = max_window - (max_window % element_bytes);
    aligned.max(element_bytes).min(helper.size_bytes)
}

fn push_bridge_windowed_kernel(
    graph: &mut LargeExecutionGraph,
    label: &'static str,
    input: LogicalRange,
    output: LogicalRange,
    work_items: u64,
    limits: LargePolicyLimits,
) -> Result<()> {
    graph.push_stage(
        LargeStage::WindowedKernel {
            label,
            input,
            output,
            work_items,
        },
        graph_requirements_covering(
            limits,
            1,
            input.size_bytes.max(output.size_bytes),
            input.size_bytes.max(output.size_bytes),
        )?,
    )
}

impl RaderBridgeC2cPlan {
    fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        config: &FftConfig,
        plan: LargeBridgePlan,
        limits: LargePolicyLimits,
    ) -> Result<Self> {
        let axis = plan.axis();
        let n = usize_from_u64(plan.axis_len())?;
        let m = usize_from_u64(plan.convolution_len())?;
        let l = n - 1;
        let lines = plan.line_count();
        let stride_complex = stride_for_axis(config.shape(), axis) as u64;
        let scale = config.scale()?;
        let keys = BridgePipelineKeys::rader(
            config.shape().len(),
            axis,
            config.shape(),
            n,
            stride_complex as usize,
            m,
            (scale - 1.0).abs() > f32::EPSILON,
            scale,
        );

        let perm = rader_permutation(n)?;
        let bfft = rader_bfft(n, m, config.direction(), &perm)?;
        let perm_buffer = storage_buffer_with_data(
            device,
            queue,
            "wgpu_fft.c2c.bridge.rader.perm",
            bytemuck::cast_slice(&perm),
        )?;
        let bfft_values = to_interleaved_f32(&bfft);
        let bfft_buffer = storage_buffer_with_data(
            device,
            queue,
            "wgpu_fft.c2c.bridge.rader.bfft",
            bytemuck::cast_slice(&bfft_values),
        )?;
        let sum_buffer = create_view_staging_buffer(
            device,
            "wgpu_fft.c2c.bridge.rader.sum",
            COMPLEX_F32_BYTES,
            wgpu::BufferUsages::empty(),
        )?;
        let x0_buffer = create_view_staging_buffer(
            device,
            "wgpu_fft.c2c.bridge.rader.x0",
            COMPLEX_F32_BYTES,
            wgpu::BufferUsages::empty(),
        )?;
        let work_bytes = checked_mul_u64(m as u64, COMPLEX_F32_BYTES)?;
        let work_buffer = create_view_staging_buffer(
            device,
            "wgpu_fft.c2c.bridge.rader.work",
            work_bytes,
            wgpu::BufferUsages::empty(),
        )?;
        let fft_buffer = create_view_staging_buffer(
            device,
            "wgpu_fft.c2c.bridge.rader.fft",
            work_bytes,
            wgpu::BufferUsages::empty(),
        )?;
        let child_forward = Box::new(C2cPlan::new_with_large_policy_limits(
            device,
            queue,
            FftConfig::new(m)
                .with_direction(FftDirection::Forward)
                .with_normalization(Normalization::None),
            Some(limits),
        )?);
        let child_inverse = Box::new(C2cPlan::new_with_large_policy_limits(
            device,
            queue,
            FftConfig::new(m)
                .with_direction(FftDirection::Inverse)
                .with_normalization(Normalization::Inverse),
            Some(limits),
        )?);

        Ok(Self {
            shape: config.shape().to_vec(),
            axis,
            n,
            l,
            m,
            lines,
            stride_complex,
            limits,
            child_forward,
            child_inverse,
            perm_buffer,
            bfft_buffer,
            sum_buffer,
            x0_buffer,
            work_buffer,
            fft_buffer,
            keys,
        })
    }

    fn execute_views(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        input: BufferView<'_>,
        output: BufferView<'_>,
    ) -> Result<()> {
        ensure_single_storage_bridge_view(&input, "large Rader bridge input")?;
        ensure_single_storage_bridge_view(&output, "large Rader bridge output")?;
        for line in 0..self.lines {
            self.dispatch_sum_init(device, encoder)?;
            self.dispatch_sum_accumulate(device, encoder, &input, line)?;
            self.dispatch_pack(device, encoder, &input, line)?;
            self.child_forward.execute_views(
                device,
                encoder,
                BufferView::whole(&self.work_buffer).prefix(self.work_bytes())?,
                BufferView::whole(&self.fft_buffer).prefix(self.work_bytes())?,
            )?;
            self.dispatch_mul(device, encoder)?;
            self.child_inverse.execute_views(
                device,
                encoder,
                BufferView::whole(&self.fft_buffer).prefix(self.work_bytes())?,
                BufferView::whole(&self.work_buffer).prefix(self.work_bytes())?,
            )?;
            self.dispatch_write_y0(device, encoder, &output, line)?;
            self.dispatch_post(device, encoder, &output, line)?;
        }
        Ok(())
    }

    fn work_bytes(&self) -> u64 {
        self.m as u64 * COMPLEX_F32_BYTES
    }

    fn max_complex_window(&self) -> u64 {
        (self.limits.max_storage_buffer_binding_size / COMPLEX_F32_BYTES).max(1)
    }

    fn max_u32_window(&self) -> u64 {
        (self.limits.max_storage_buffer_binding_size / 4).max(1)
    }

    fn dispatch_sum_init(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
    ) -> Result<()> {
        let scheduler = WindowScheduler::for_device(device);
        let sum = BufferView::whole(&self.sum_buffer).prefix(COMPLEX_F32_BYTES)?;
        let x0 = BufferView::whole(&self.x0_buffer).prefix(COMPLEX_F32_BYTES)?;
        dispatch_bridge_kernel(
            device,
            encoder,
            "rader",
            bridge_pipeline_key(
                self.keys.rader_sum_init.as_ref(),
                "rader",
                "Rader bridge sum-init pipeline key is missing",
            )?,
            &[
                bridge_entry(&scheduler, 0, &sum, ElementFormat::ComplexF32)?,
                bridge_entry(&scheduler, 1, &x0, ElementFormat::ComplexF32)?,
            ],
            2,
            BridgeKernelParams {
                line_count: 1,
                line_offset: 0,
                t_offset: 0,
                t_count: 0,
                input_base: 0,
                output_base: 0,
                aux0_base: 0,
                aux1_base: 0,
                aux2_base: 0,
                aux3_base: 0,
                stride: 1,
                _pad0: 0,
            },
            1,
        )
    }

    fn dispatch_sum_accumulate(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        input: &BufferView<'_>,
        line: u64,
    ) -> Result<()> {
        let scheduler = WindowScheduler::for_device(device);
        let base = line_base_complex(&self.shape, self.axis, line)?;
        let mut t_offset = 0u64;
        while t_offset < self.n as u64 {
            let t_count = (self.n as u64 - t_offset)
                .min(self.max_complex_window().min(u64::from(WORKGROUP_SIZE)));
            let input_first =
                checked_add_u64(base, checked_mul_u64(t_offset, self.stride_complex)?)?;
            let input_span = strided_span_elements(t_count, self.stride_complex)?;
            let (input_window, input_base) =
                bind_complex_window(device, input, input_first, input_span)?;
            let sum = BufferView::whole(&self.sum_buffer).prefix(COMPLEX_F32_BYTES)?;
            let x0 = BufferView::whole(&self.x0_buffer).prefix(COMPLEX_F32_BYTES)?;
            dispatch_bridge_kernel(
                device,
                encoder,
                "rader",
                bridge_pipeline_key(
                    self.keys.rader_sum_accumulate.as_ref(),
                    "rader",
                    "Rader bridge sum-accumulate pipeline key is missing",
                )?,
                &[
                    bridge_entry(&scheduler, 0, &input_window, ElementFormat::ComplexF32)?,
                    bridge_entry(&scheduler, 1, &sum, ElementFormat::ComplexF32)?,
                    bridge_entry(&scheduler, 2, &x0, ElementFormat::ComplexF32)?,
                ],
                3,
                BridgeKernelParams {
                    line_count: 1,
                    line_offset: u64_to_u32(line)?,
                    t_offset: u64_to_u32(t_offset)?,
                    t_count: u64_to_u32(t_count)?,
                    input_base,
                    output_base: 0,
                    aux0_base: 0,
                    aux1_base: 0,
                    aux2_base: 0,
                    aux3_base: 0,
                    stride: u64_to_u32(self.stride_complex)?,
                    _pad0: 0,
                },
                1,
            )?;
            t_offset += t_count;
        }
        Ok(())
    }

    fn dispatch_pack(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        input: &BufferView<'_>,
        line: u64,
    ) -> Result<()> {
        let scheduler = WindowScheduler::for_device(device);
        let base = line_base_complex(&self.shape, self.axis, line)?;
        let mut t_offset = 0u64;
        while t_offset < self.m as u64 {
            let mut t_count = (self.m as u64 - t_offset).min(self.max_complex_window());
            if t_offset < self.l as u64 {
                t_count = t_count
                    .min(self.l as u64 - t_offset)
                    .min(self.max_u32_window());
            }
            let work = BufferView::whole(&self.work_buffer);
            let (work_window, work_base) = bind_complex_window(device, &work, t_offset, t_count)?;
            let input_span = strided_span_elements(self.n as u64, self.stride_complex)?;
            let (input_window, input_base) = bind_complex_window(device, input, base, input_span)?;
            let (perm_window, perm_base, perm_logical_start) = if t_offset < self.l as u64 {
                let perm_start = self.l as u64 - t_offset - t_count;
                let (window, base) =
                    bind_u32_window(device, &self.perm_buffer, perm_start, t_count)?;
                (window, base, perm_start)
            } else {
                let (window, base) = bind_u32_window(device, &self.perm_buffer, 0, 1)?;
                (window, base, 0)
            };
            dispatch_bridge_kernel(
                device,
                encoder,
                "rader",
                bridge_pipeline_key(
                    self.keys.rader_pack.as_ref(),
                    "rader",
                    "Rader bridge pack pipeline key is missing",
                )?,
                &[
                    bridge_entry(&scheduler, 0, &input_window, ElementFormat::ComplexF32)?,
                    bridge_entry(&scheduler, 1, &work_window, ElementFormat::ComplexF32)?,
                    bridge_entry(&scheduler, 2, &perm_window, ElementFormat::U32)?,
                ],
                3,
                BridgeKernelParams {
                    line_count: 1,
                    line_offset: u64_to_u32(line)?,
                    t_offset: u64_to_u32(t_offset)?,
                    t_count: u64_to_u32(t_count)?,
                    input_base,
                    output_base: work_base,
                    aux0_base: perm_base,
                    aux1_base: u64_to_u32(perm_logical_start)?,
                    aux2_base: 0,
                    aux3_base: 0,
                    stride: u64_to_u32(self.stride_complex)?,
                    _pad0: 0,
                },
                u64_to_u32(t_count)?,
            )?;
            t_offset += t_count;
        }
        Ok(())
    }

    fn dispatch_mul(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
    ) -> Result<()> {
        dispatch_bridge_mul_windows(
            device,
            encoder,
            "rader",
            bridge_pipeline_key(
                self.keys.rader_mul.as_ref(),
                "rader",
                "Rader bridge multiply pipeline key is missing",
            )?,
            &self.fft_buffer,
            &self.bfft_buffer,
            self.m as u64,
            self.max_complex_window(),
        )
    }

    fn dispatch_write_y0(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        output: &BufferView<'_>,
        line: u64,
    ) -> Result<()> {
        let scheduler = WindowScheduler::for_device(device);
        let base = line_base_complex(&self.shape, self.axis, line)?;
        let (output_window, output_base) = bind_complex_window(device, output, base, 1)?;
        let sum = BufferView::whole(&self.sum_buffer).prefix(COMPLEX_F32_BYTES)?;
        dispatch_bridge_kernel(
            device,
            encoder,
            "rader",
            bridge_pipeline_key(
                self.keys.rader_write_y0.as_ref(),
                "rader",
                "Rader bridge y0 pipeline key is missing",
            )?,
            &[
                bridge_entry(&scheduler, 0, &sum, ElementFormat::ComplexF32)?,
                bridge_entry(&scheduler, 1, &output_window, ElementFormat::ComplexF32)?,
            ],
            2,
            BridgeKernelParams {
                line_count: 1,
                line_offset: u64_to_u32(line)?,
                t_offset: 0,
                t_count: 1,
                input_base: 0,
                output_base,
                aux0_base: 0,
                aux1_base: 0,
                aux2_base: 0,
                aux3_base: 0,
                stride: u64_to_u32(self.stride_complex)?,
                _pad0: 0,
            },
            1,
        )
    }

    fn dispatch_post(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        output: &BufferView<'_>,
        line: u64,
    ) -> Result<()> {
        let scheduler = WindowScheduler::for_device(device);
        let base = line_base_complex(&self.shape, self.axis, line)?;
        let mut t_offset = 0u64;
        while t_offset < self.l as u64 {
            let t_count = (self.l as u64 - t_offset)
                .min(self.max_complex_window())
                .min(self.max_u32_window());
            let work = BufferView::whole(&self.work_buffer);
            let (conv_low, conv_low_base) = bind_complex_window(device, &work, t_offset, t_count)?;
            let wrap_offset = t_offset + self.l as u64;
            let wrap_count = if wrap_offset < self.m as u64 {
                t_count.min(self.m as u64 - wrap_offset)
            } else {
                1
            };
            let (conv_high, conv_high_base) = bind_complex_window(
                device,
                &work,
                wrap_offset.min((self.m - 1) as u64),
                wrap_count,
            )?;
            let x0 = BufferView::whole(&self.x0_buffer).prefix(COMPLEX_F32_BYTES)?;
            let (perm_window, perm_base) =
                bind_u32_window(device, &self.perm_buffer, t_offset, t_count)?;
            let output_span = strided_span_elements(self.n as u64, self.stride_complex)?;
            let (output_window, output_base) =
                bind_complex_window(device, output, base, output_span)?;
            dispatch_bridge_kernel(
                device,
                encoder,
                "rader",
                bridge_pipeline_key(
                    self.keys.rader_post.as_ref(),
                    "rader",
                    "Rader bridge post pipeline key is missing",
                )?,
                &[
                    bridge_entry(&scheduler, 0, &conv_low, ElementFormat::ComplexF32)?,
                    bridge_entry(&scheduler, 1, &conv_high, ElementFormat::ComplexF32)?,
                    bridge_entry(&scheduler, 2, &x0, ElementFormat::ComplexF32)?,
                    bridge_entry(&scheduler, 3, &perm_window, ElementFormat::U32)?,
                    bridge_entry(&scheduler, 4, &output_window, ElementFormat::ComplexF32)?,
                ],
                5,
                BridgeKernelParams {
                    line_count: 1,
                    line_offset: u64_to_u32(line)?,
                    t_offset: u64_to_u32(t_offset)?,
                    t_count: u64_to_u32(t_count)?,
                    input_base: conv_low_base,
                    output_base,
                    aux0_base: conv_high_base,
                    aux1_base: 0,
                    aux2_base: perm_base,
                    aux3_base: u64_to_u32(t_offset)?,
                    stride: u64_to_u32(self.stride_complex)?,
                    _pad0: 0,
                },
                u64_to_u32(t_count)?,
            )?;
            t_offset += t_count;
        }
        Ok(())
    }
}

impl BluesteinBridgeC2cPlan {
    fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        config: &FftConfig,
        plan: LargeBridgePlan,
        limits: LargePolicyLimits,
    ) -> Result<Self> {
        let axis = plan.axis();
        let n = usize_from_u64(plan.axis_len())?;
        let m = usize_from_u64(plan.convolution_len())?;
        let lines = plan.line_count();
        let stride_complex = stride_for_axis(config.shape(), axis) as u64;
        let scale = config.scale()?;
        let keys = BridgePipelineKeys::bluestein(
            config.shape().len(),
            axis,
            config.shape(),
            n,
            stride_complex as usize,
            m,
            (scale - 1.0).abs() > f32::EPSILON,
            scale,
        );

        let chirp = bluestein_chirp(n, config.direction());
        let bfft = bluestein_bfft(n, m, config.direction())?;
        let chirp_values = to_interleaved_f32(&chirp);
        let bfft_values = to_interleaved_f32(&bfft);
        let chirp_buffer = storage_buffer_with_data(
            device,
            queue,
            "wgpu_fft.c2c.bridge.bluestein.chirp",
            bytemuck::cast_slice(&chirp_values),
        )?;
        let bfft_buffer = storage_buffer_with_data(
            device,
            queue,
            "wgpu_fft.c2c.bridge.bluestein.bfft",
            bytemuck::cast_slice(&bfft_values),
        )?;
        let work_bytes = checked_mul_u64(m as u64, COMPLEX_F32_BYTES)?;
        let work_buffer = create_view_staging_buffer(
            device,
            "wgpu_fft.c2c.bridge.bluestein.work",
            work_bytes,
            wgpu::BufferUsages::empty(),
        )?;
        let fft_buffer = create_view_staging_buffer(
            device,
            "wgpu_fft.c2c.bridge.bluestein.fft",
            work_bytes,
            wgpu::BufferUsages::empty(),
        )?;
        let child_forward = Box::new(C2cPlan::new_with_large_policy_limits(
            device,
            queue,
            FftConfig::new(m)
                .with_direction(FftDirection::Forward)
                .with_normalization(Normalization::None),
            Some(limits),
        )?);
        let child_inverse = Box::new(C2cPlan::new_with_large_policy_limits(
            device,
            queue,
            FftConfig::new(m)
                .with_direction(FftDirection::Inverse)
                .with_normalization(Normalization::Inverse),
            Some(limits),
        )?);

        Ok(Self {
            shape: config.shape().to_vec(),
            axis,
            n,
            m,
            lines,
            stride_complex,
            limits,
            child_forward,
            child_inverse,
            chirp_buffer,
            bfft_buffer,
            work_buffer,
            fft_buffer,
            keys,
        })
    }

    fn execute_views(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        input: BufferView<'_>,
        output: BufferView<'_>,
    ) -> Result<()> {
        ensure_single_storage_bridge_view(&input, "large Bluestein bridge input")?;
        ensure_single_storage_bridge_view(&output, "large Bluestein bridge output")?;
        for line in 0..self.lines {
            self.dispatch_pack(device, encoder, &input, line)?;
            self.child_forward.execute_views(
                device,
                encoder,
                BufferView::whole(&self.work_buffer).prefix(self.work_bytes())?,
                BufferView::whole(&self.fft_buffer).prefix(self.work_bytes())?,
            )?;
            self.dispatch_mul(device, encoder)?;
            self.child_inverse.execute_views(
                device,
                encoder,
                BufferView::whole(&self.fft_buffer).prefix(self.work_bytes())?,
                BufferView::whole(&self.work_buffer).prefix(self.work_bytes())?,
            )?;
            self.dispatch_post(device, encoder, &output, line)?;
        }
        Ok(())
    }

    fn work_bytes(&self) -> u64 {
        self.m as u64 * COMPLEX_F32_BYTES
    }

    fn max_complex_window(&self) -> u64 {
        (self.limits.max_storage_buffer_binding_size / COMPLEX_F32_BYTES).max(1)
    }

    fn dispatch_pack(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        input: &BufferView<'_>,
        line: u64,
    ) -> Result<()> {
        let scheduler = WindowScheduler::for_device(device);
        let base = line_base_complex(&self.shape, self.axis, line)?;
        let mut t_offset = 0u64;
        while t_offset < self.m as u64 {
            let t_count = (self.m as u64 - t_offset).min(self.max_complex_window());
            let work = BufferView::whole(&self.work_buffer);
            let (work_window, work_base) = bind_complex_window(device, &work, t_offset, t_count)?;
            let active_count = t_count.min((self.n as u64).saturating_sub(t_offset));
            let input_first = checked_add_u64(
                base,
                checked_mul_u64(t_offset.min((self.n - 1) as u64), self.stride_complex)?,
            )?;
            let input_span = strided_span_elements(active_count.max(1), self.stride_complex)?;
            let (input_window, input_base) =
                bind_complex_window(device, input, input_first, input_span)?;
            let chirp_start = t_offset.min((self.n - 1) as u64);
            let chirp_count = active_count.max(1);
            let chirp = BufferView::whole(&self.chirp_buffer);
            let (chirp_window, chirp_base) =
                bind_complex_window(device, &chirp, chirp_start, chirp_count)?;
            dispatch_bridge_kernel(
                device,
                encoder,
                "bluestein",
                bridge_pipeline_key(
                    self.keys.bluestein_pack.as_ref(),
                    "bluestein",
                    "Bluestein bridge pack pipeline key is missing",
                )?,
                &[
                    bridge_entry(&scheduler, 0, &input_window, ElementFormat::ComplexF32)?,
                    bridge_entry(&scheduler, 1, &work_window, ElementFormat::ComplexF32)?,
                    bridge_entry(&scheduler, 2, &chirp_window, ElementFormat::ComplexF32)?,
                ],
                3,
                BridgeKernelParams {
                    line_count: 1,
                    line_offset: u64_to_u32(line)?,
                    t_offset: u64_to_u32(t_offset)?,
                    t_count: u64_to_u32(t_count)?,
                    input_base,
                    output_base: work_base,
                    aux0_base: chirp_base,
                    aux1_base: u64_to_u32(chirp_start)?,
                    aux2_base: 0,
                    aux3_base: 0,
                    stride: u64_to_u32(self.stride_complex)?,
                    _pad0: 0,
                },
                u64_to_u32(t_count)?,
            )?;
            t_offset += t_count;
        }
        Ok(())
    }

    fn dispatch_mul(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
    ) -> Result<()> {
        dispatch_bridge_mul_windows(
            device,
            encoder,
            "bluestein",
            bridge_pipeline_key(
                self.keys.bluestein_mul.as_ref(),
                "bluestein",
                "Bluestein bridge multiply pipeline key is missing",
            )?,
            &self.fft_buffer,
            &self.bfft_buffer,
            self.m as u64,
            self.max_complex_window(),
        )
    }

    fn dispatch_post(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        output: &BufferView<'_>,
        line: u64,
    ) -> Result<()> {
        let scheduler = WindowScheduler::for_device(device);
        let base = line_base_complex(&self.shape, self.axis, line)?;
        let mut t_offset = 0u64;
        while t_offset < self.n as u64 {
            let t_count = (self.n as u64 - t_offset).min(self.max_complex_window());
            let work = BufferView::whole(&self.work_buffer);
            let (work_window, work_base) = bind_complex_window(device, &work, t_offset, t_count)?;
            let chirp = BufferView::whole(&self.chirp_buffer);
            let (chirp_window, chirp_base) =
                bind_complex_window(device, &chirp, t_offset, t_count)?;
            let output_first =
                checked_add_u64(base, checked_mul_u64(t_offset, self.stride_complex)?)?;
            let output_span = strided_span_elements(t_count, self.stride_complex)?;
            let (output_window, output_base) =
                bind_complex_window(device, output, output_first, output_span)?;
            dispatch_bridge_kernel(
                device,
                encoder,
                "bluestein",
                bridge_pipeline_key(
                    self.keys.bluestein_post.as_ref(),
                    "bluestein",
                    "Bluestein bridge post pipeline key is missing",
                )?,
                &[
                    bridge_entry(&scheduler, 0, &work_window, ElementFormat::ComplexF32)?,
                    bridge_entry(&scheduler, 1, &chirp_window, ElementFormat::ComplexF32)?,
                    bridge_entry(&scheduler, 2, &output_window, ElementFormat::ComplexF32)?,
                ],
                3,
                BridgeKernelParams {
                    line_count: 1,
                    line_offset: u64_to_u32(line)?,
                    t_offset: u64_to_u32(t_offset)?,
                    t_count: u64_to_u32(t_count)?,
                    input_base: work_base,
                    output_base,
                    aux0_base: chirp_base,
                    aux1_base: 0,
                    aux2_base: 0,
                    aux3_base: 0,
                    stride: u64_to_u32(self.stride_complex)?,
                    _pad0: 0,
                },
                u64_to_u32(t_count)?,
            )?;
            t_offset += t_count;
        }
        Ok(())
    }
}

impl BridgePipelineKeys {
    #[allow(clippy::too_many_arguments)]
    fn rader(
        rank: usize,
        axis: usize,
        dims: &[usize],
        n: usize,
        stride_complex: usize,
        m: usize,
        apply_scale: bool,
        scale: f32,
    ) -> Self {
        let key = |kind| {
            ComputePipelineCacheKey::bridge_stage(BridgeStageKey::new(
                kind,
                rank,
                axis,
                dims,
                n,
                stride_complex,
                m,
                WORKGROUP_SIZE,
                apply_scale,
                scale,
            ))
        };
        Self {
            rader_sum_init: Some(key(BridgeKernelKind::RaderSumInit)),
            rader_sum_accumulate: Some(key(BridgeKernelKind::RaderSumAccumulate)),
            rader_pack: Some(key(BridgeKernelKind::RaderPack)),
            rader_mul: Some(key(BridgeKernelKind::RaderMul)),
            rader_write_y0: Some(key(BridgeKernelKind::RaderWriteY0)),
            rader_post: Some(key(BridgeKernelKind::RaderPost)),
            bluestein_pack: None,
            bluestein_mul: None,
            bluestein_post: None,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn bluestein(
        rank: usize,
        axis: usize,
        dims: &[usize],
        n: usize,
        stride_complex: usize,
        m: usize,
        apply_scale: bool,
        scale: f32,
    ) -> Self {
        let key = |kind| {
            ComputePipelineCacheKey::bridge_stage(BridgeStageKey::new(
                kind,
                rank,
                axis,
                dims,
                n,
                stride_complex,
                m,
                WORKGROUP_SIZE,
                apply_scale,
                scale,
            ))
        };
        Self {
            rader_sum_init: None,
            rader_sum_accumulate: None,
            rader_pack: None,
            rader_mul: None,
            rader_write_y0: None,
            rader_post: None,
            bluestein_pack: Some(key(BridgeKernelKind::BluesteinPack)),
            bluestein_mul: Some(key(BridgeKernelKind::BluesteinMul)),
            bluestein_post: Some(key(BridgeKernelKind::BluesteinPost)),
        }
    }
}

impl SmoothPhaseExecution {
    fn execute_views(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        input: BufferView<'_>,
        output: BufferView<'_>,
    ) -> Result<()> {
        match self {
            Self::Axis(plan) => plan.execute_views(device, encoder, input, output),
            Self::C2c(plan) => plan.execute_views(device, encoder, input, output),
        }
    }

    fn graph(&self) -> Result<SmoothPhaseGraph> {
        Ok(match self {
            Self::Axis(plan) => SmoothPhaseGraph::Axis {
                stage_kinds: plan.graph_stage_kinds(),
                workspace_bytes: plan.workspace_size_bytes(),
            },
            Self::C2c(plan) => SmoothPhaseGraph::C2c(plan.execution_graph()?),
        })
    }
}

fn build_smooth_phase_execution(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    config: AxisPlanConfig,
    required_bytes: u64,
    limits: LargePolicyLimits,
    twiddle_lut_pool: &mut AxisTwiddleLutPool,
) -> Result<SmoothPhaseExecution> {
    if required_bytes <= limits.max_storage_buffer_binding_size {
        return Ok(SmoothPhaseExecution::Axis(
            AxisPlan::new_with_twiddle_lut_pool(device, queue, config, twiddle_lut_pool)?,
        ));
    }
    let c2c_config = FftConfig::new_nd(config.shape.clone())
        .with_axes(config.axes.clone())
        .with_direction(config.direction)
        .with_normalization(config.normalization)
        .with_batch(config.batch);
    Ok(SmoothPhaseExecution::C2c(Box::new(
        C2cPlan::new_with_large_policy_limits(device, queue, c2c_config, Some(limits))?,
    )))
}

impl SmoothDecompositionC2cPlan {
    fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        config: &FftConfig,
        plan: SmoothDecompositionPlan,
        limits: LargePolicyLimits,
    ) -> Result<Self> {
        let mut steps = Vec::with_capacity(plan.steps().len());
        let mut twiddle_lut_pool = AxisTwiddleLutPool::default();
        for (step_index, step) in plan.steps().iter().copied().enumerate() {
            let final_axis = step_index + 1 == plan.steps().len();
            match step {
                SmoothDecompositionStep::MixedAxis(step) => {
                    let axis_plan = AxisPlan::new_with_twiddle_lut_pool(
                        device,
                        queue,
                        AxisPlanConfig {
                            shape: vec![step.len() as usize],
                            axes: vec![0],
                            batch: 1,
                            direction: config.direction(),
                            normalization: if final_axis {
                                config.normalization()
                            } else {
                                Normalization::None
                            },
                            scale_override_bits: if final_axis {
                                Some(config.scale()?.to_bits())
                            } else {
                                None
                            },
                            layout: AxisLayout::Interleaved,
                            precision: AxisPrecision::F32,
                        },
                        &mut twiddle_lut_pool,
                    )?;
                    let line_input = create_view_staging_buffer(
                        device,
                        "wgpu_fft.c2c.decompose.mixed.line_input",
                        step.line_bytes(),
                        wgpu::BufferUsages::COPY_DST,
                    )?;
                    let line_output = create_view_staging_buffer(
                        device,
                        "wgpu_fft.c2c.decompose.mixed.line_output",
                        step.line_bytes(),
                        wgpu::BufferUsages::COPY_SRC,
                    )?;
                    steps.push(SmoothExecutionStep::Mixed(MixedAxisExecution {
                        step,
                        plan: axis_plan,
                        line_input,
                        line_output,
                    }));
                }
                SmoothDecompositionStep::SmoothAxis(step) => {
                    let phase1_config = AxisPlanConfig {
                        shape: vec![step.chunk_inner() as usize, step.outer() as usize],
                        axes: vec![1],
                        batch: 1,
                        direction: config.direction(),
                        normalization: Normalization::None,
                        scale_override_bits: None,
                        layout: AxisLayout::Interleaved,
                        precision: AxisPrecision::F32,
                    };
                    let phase1 = build_smooth_phase_execution(
                        device,
                        queue,
                        phase1_config,
                        step.phase1_chunk_bytes(),
                        limits,
                        &mut twiddle_lut_pool,
                    )?;
                    let phase2_config = AxisPlanConfig {
                        shape: vec![step.chunk_outer() as usize, step.inner() as usize],
                        axes: vec![1],
                        batch: 1,
                        direction: config.direction(),
                        normalization: Normalization::None,
                        scale_override_bits: None,
                        layout: AxisLayout::Interleaved,
                        precision: AxisPrecision::F32,
                    };
                    let phase2 = build_smooth_phase_execution(
                        device,
                        queue,
                        phase2_config,
                        step.phase2_chunk_bytes(),
                        limits,
                        &mut twiddle_lut_pool,
                    )?;
                    let phase1_input = create_view_staging_buffer(
                        device,
                        "wgpu_fft.c2c.smooth.phase1_input",
                        step.phase1_chunk_bytes(),
                        wgpu::BufferUsages::COPY_DST,
                    )?;
                    let phase1_output = create_view_staging_buffer(
                        device,
                        "wgpu_fft.c2c.smooth.phase1_output",
                        step.phase1_chunk_bytes(),
                        wgpu::BufferUsages::empty(),
                    )?;
                    let phase2_input = create_view_staging_buffer(
                        device,
                        "wgpu_fft.c2c.smooth.phase2_input",
                        step.phase2_chunk_bytes(),
                        wgpu::BufferUsages::empty(),
                    )?;
                    let phase2_output = create_view_staging_buffer(
                        device,
                        "wgpu_fft.c2c.smooth.phase2_output",
                        step.phase2_chunk_bytes(),
                        wgpu::BufferUsages::COPY_SRC,
                    )?;
                    let twiddle_lut = two_level_twiddle_lut_f32(step.len() as usize);
                    let twiddle_coarse = create_twiddle_lut_buffer(
                        device,
                        queue,
                        "wgpu_fft.c2c.smooth.twiddle_coarse",
                        &twiddle_lut.coarse,
                    )?;
                    let twiddle_fine = create_twiddle_lut_buffer(
                        device,
                        queue,
                        "wgpu_fft.c2c.smooth.twiddle_fine",
                        &twiddle_lut.fine,
                    )?;
                    steps.push(SmoothExecutionStep::Smooth(SmoothAxisExecution {
                        step,
                        phase1,
                        phase2,
                        phase1_input,
                        phase1_output,
                        phase2_input,
                        phase2_output,
                        twiddle_coarse,
                        twiddle_fine,
                        twiddle_shift: twiddle_lut.shift,
                        twiddle_mask: twiddle_lut.mask,
                        scale: if final_axis { config.scale()? } else { 1.0 },
                        inverse: config.direction() == FftDirection::Inverse,
                    }));
                }
            }
        }
        let temp_buffer = (plan.temp_buffer_size_bytes() > 0)
            .then(|| {
                create_view_staging_buffer(
                    device,
                    "wgpu_fft.c2c.decompose.sequence_temp",
                    plan.temp_buffer_size_bytes(),
                    wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                )
            })
            .transpose()?;
        let graph_steps = smooth_graph_steps(&steps)?;
        let graph_plan = build_smooth_c2c_graph(
            &graph_steps,
            limits,
            device.limits().min_storage_buffer_offset_alignment,
        )?;
        let axis_twiddle_lut_storage_bytes = twiddle_lut_pool.storage_bytes();
        Ok(Self {
            plan,
            graph_plan,
            steps,
            temp_buffer,
            axis_twiddle_lut_storage_bytes,
        })
    }

    fn execute_views(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        input: BufferView<'_>,
        output: BufferView<'_>,
    ) -> Result<()> {
        let scheduler = WindowScheduler::for_device(device);
        let executor = StageExecutor::new(&scheduler);
        executor.validate_graph(self.graph_plan.graph())?;
        let input = input.prefix(self.plan.required_buffer_size_bytes())?;
        let output = output.prefix(self.plan.required_buffer_size_bytes())?;
        let input_stage = if view_has_single_storage_range(&input) {
            None
        } else {
            let buffer = create_view_staging_buffer(
                device,
                "wgpu_fft.c2c.decompose.segmented_input_stage",
                self.plan.required_buffer_size_bytes(),
                wgpu::BufferUsages::COPY_DST,
            )?;
            executor.copy_view_range_to_buffer(
                encoder,
                &input,
                0,
                &buffer,
                0,
                self.plan.required_buffer_size_bytes(),
            )?;
            Some(buffer)
        };
        let output_stage = if view_has_single_storage_range(&output) {
            None
        } else {
            Some(create_view_staging_buffer(
                device,
                "wgpu_fft.c2c.decompose.segmented_output_stage",
                self.plan.required_buffer_size_bytes(),
                wgpu::BufferUsages::COPY_SRC,
            )?)
        };
        let exec_input = if let Some(buffer) = input_stage.as_ref() {
            BufferView::whole(buffer).prefix(self.plan.required_buffer_size_bytes())?
        } else {
            input.clone()
        };
        let exec_output = if let Some(buffer) = output_stage.as_ref() {
            BufferView::whole(buffer).prefix(self.plan.required_buffer_size_bytes())?
        } else {
            output.clone()
        };

        let mut src_slot = SequenceBufferSlot::Input;
        let mut dst_slot = if self.steps.len() % 2 == 1 {
            SequenceBufferSlot::Output
        } else {
            SequenceBufferSlot::Temp
        };

        for (step_index, step) in self.steps.iter().enumerate() {
            let src = self.resolve_buffer(src_slot, exec_input.clone(), exec_output.clone())?;
            let dst = self.resolve_buffer(dst_slot, exec_input.clone(), exec_output.clone())?;
            match step {
                SmoothExecutionStep::Mixed(step) => {
                    step.execute(device, encoder, self.plan.shape(), src, dst)?;
                }
                SmoothExecutionStep::Smooth(step) => {
                    step.execute(device, encoder, self.plan.shape(), src, dst)?;
                }
            }

            if step_index + 1 < self.steps.len() {
                src_slot = dst_slot;
                dst_slot = next_sequence_destination(
                    "smooth-decomposition-buffer-flow",
                    src_slot,
                    "smooth decomposition buffer flow attempted to use input as destination",
                )?;
            }
        }
        if let Some(buffer) = output_stage.as_ref() {
            executor.copy_buffer_to_view_range(
                encoder,
                buffer,
                0,
                &output,
                0,
                self.plan.required_buffer_size_bytes(),
            )?;
        }
        Ok(())
    }

    fn resolve_buffer<'a>(
        &'a self,
        slot: SequenceBufferSlot,
        input: BufferView<'a>,
        output: BufferView<'a>,
    ) -> Result<BufferView<'a>> {
        match slot {
            SequenceBufferSlot::Input => Ok(input),
            SequenceBufferSlot::Output => Ok(output),
            SequenceBufferSlot::Temp => {
                if let Some(buffer) = self.temp_buffer.as_ref() {
                    Ok(BufferView::whole(buffer))
                } else {
                    Err(sequence_temp_storage_error(
                        "smooth-decomposition-workspace",
                        "multi-step smooth decomposition requires temp storage",
                    ))
                }
            }
        }
    }
}

impl MixedAxisExecution {
    fn execute(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        shape: &[usize],
        input: BufferView<'_>,
        output: BufferView<'_>,
    ) -> Result<()> {
        for line in 0..self.step.line_count() {
            let base = line_base_complex(shape, self.step.axis(), line)?;
            dispatch_c2c_axis_line_copy_blocks(
                device,
                encoder,
                C2cSmoothKernelKind::GatherAxisLine,
                &input,
                &BufferView::whole(&self.line_input).prefix(self.step.line_bytes())?,
                base,
                self.step.stride(),
                self.step.len(),
            )?;
            self.plan.execute_views(
                device,
                encoder,
                BufferView::whole(&self.line_input).prefix(self.step.line_bytes())?,
                BufferView::whole(&self.line_output).prefix(self.step.line_bytes())?,
            )?;
            dispatch_c2c_axis_line_copy_blocks(
                device,
                encoder,
                C2cSmoothKernelKind::ScatterAxisLine,
                &BufferView::whole(&self.line_output).prefix(self.step.line_bytes())?,
                &output,
                base,
                self.step.stride(),
                self.step.len(),
            )?;
        }
        Ok(())
    }
}

impl SmoothAxisExecution {
    fn execute(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        shape: &[usize],
        input: BufferView<'_>,
        output: BufferView<'_>,
    ) -> Result<()> {
        let inner = self.step.inner();
        let outer = self.step.outer();
        let chunk_inner = self.step.chunk_inner();
        let chunk_outer = self.step.chunk_outer();

        for line in 0..self.step.line_count() {
            let base = line_base_complex(shape, self.step.axis(), line)?;
            for k1_start in (0..outer).step_by(chunk_outer as usize) {
                for n1_start in (0..inner).step_by(chunk_inner as usize) {
                    dispatch_c2c_smooth_chunk_copy_blocks(
                        device,
                        encoder,
                        C2cSmoothKernelKind::GatherSmoothPhase1,
                        &input,
                        &BufferView::whole(&self.phase1_input)
                            .prefix(self.step.phase1_chunk_bytes())?,
                        SmoothChunkCopyRequest {
                            base,
                            stride: self.step.stride(),
                            inner,
                            outer,
                            chunk_inner,
                            chunk_outer,
                            offset_start: n1_start,
                        },
                    )?;
                    self.phase1.execute_views(
                        device,
                        encoder,
                        BufferView::whole(&self.phase1_input)
                            .prefix(self.step.phase1_chunk_bytes())?,
                        BufferView::whole(&self.phase1_output)
                            .prefix(self.step.phase1_chunk_bytes())?,
                    )?;
                    dispatch_c2c_smooth_twiddle_transpose(
                        device,
                        encoder,
                        &BufferView::whole(&self.phase1_output)
                            .prefix(self.step.phase1_chunk_bytes())?,
                        &BufferView::whole(&self.phase2_input)
                            .prefix(self.step.phase2_chunk_bytes())?,
                        &self.twiddle_coarse,
                        &self.twiddle_fine,
                        SmoothTwiddleParams {
                            total_complex: (chunk_inner * chunk_outer) as u32,
                            chunk_inner: chunk_inner as u32,
                            chunk_outer: chunk_outer as u32,
                            outer: outer as u32,
                            n1_start: n1_start as u32,
                            k1_start: k1_start as u32,
                            total_len: self.step.len() as u32,
                            inverse: u32::from(self.inverse),
                            scale: self.scale,
                            inner: inner as u32,
                            lut_shift: self.twiddle_shift,
                            lut_mask: self.twiddle_mask,
                        },
                    )?;
                }
                self.phase2.execute_views(
                    device,
                    encoder,
                    BufferView::whole(&self.phase2_input).prefix(self.step.phase2_chunk_bytes())?,
                    BufferView::whole(&self.phase2_output)
                        .prefix(self.step.phase2_chunk_bytes())?,
                )?;
                dispatch_c2c_smooth_chunk_copy_blocks(
                    device,
                    encoder,
                    C2cSmoothKernelKind::ScatterSmoothPhase2,
                    &BufferView::whole(&self.phase2_output)
                        .prefix(self.step.phase2_chunk_bytes())?,
                    &output,
                    SmoothChunkCopyRequest {
                        base,
                        stride: self.step.stride(),
                        inner,
                        outer,
                        chunk_inner,
                        chunk_outer,
                        offset_start: k1_start,
                    },
                )?;
            }
        }
        Ok(())
    }
}

fn build_route_impl(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    config: &FftConfig,
    axis_kinds: &[AxisKind],
    route: C2cRoute,
    len: u32,
) -> Result<C2cRouteImpl> {
    Ok(match route {
        C2cRoute::DirectDft => {
            C2cRouteImpl::DirectDft(DirectDftPlan::new(device, queue, config, len)?)
        }
        C2cRoute::MixedRadix => C2cRouteImpl::MixedRadix(AxisPlan::new(
            device,
            queue,
            AxisPlanConfig::from_c2c_config(config),
        )?),
        C2cRoute::Rader => C2cRouteImpl::Rader(RaderAxis::new(
            device,
            queue,
            rader_config_for_axis(config, config.axes()[0], true),
        )?),
        C2cRoute::Bluestein => C2cRouteImpl::Bluestein(BluesteinAxis::new(
            device,
            queue,
            bluestein_config_for_axis(config, config.axes()[0], true),
        )?),
        C2cRoute::AxisSequence => {
            C2cRouteImpl::AxisSequence(AxisSequencePlan::new(device, queue, config, axis_kinds)?)
        }
    })
}

fn axis_factors_for_route_impl(
    config: &FftConfig,
    route_impl: &C2cRouteImpl,
) -> Result<Vec<Vec<usize>>> {
    match route_impl {
        C2cRouteImpl::DirectDft(_) => config
            .axes()
            .iter()
            .map(|&axis| crate::runtime::factor_supported_length(config.shape()[axis]))
            .collect::<Result<Vec<_>>>(),
        C2cRouteImpl::MixedRadix(plan) => Ok(plan.factors().to_vec()),
        C2cRouteImpl::Rader(_) => Ok(config.axes().iter().map(|_| Vec::new()).collect()),
        C2cRouteImpl::Bluestein(_) => Ok(config.axes().iter().map(|_| Vec::new()).collect()),
        C2cRouteImpl::AxisSequence(plan) => Ok(plan.axis_factors().to_vec()),
    }
}

fn axis_factors_for_axis_kinds(
    config: &FftConfig,
    axis_kinds: &[AxisKind],
) -> Result<Vec<Vec<usize>>> {
    config
        .axes()
        .iter()
        .zip(axis_kinds)
        .map(|(&axis, kind)| match kind {
            AxisKind::Mixed => crate::runtime::factor_supported_length(config.shape()[axis]),
            AxisKind::Rader | AxisKind::Bluestein => Ok(Vec::new()),
        })
        .collect()
}

fn large_bridge_route_for(route: C2cRoute) -> Option<LargeBridgeRoute> {
    match route {
        C2cRoute::Rader => Some(LargeBridgeRoute::Rader),
        C2cRoute::Bluestein => Some(LargeBridgeRoute::Bluestein),
        _ => None,
    }
}

fn bytes_per_batch(config: &FftConfig) -> Result<u64> {
    (config.logical_complex_len()? as u64)
        .checked_mul(config.precision().complex_size_bytes())
        .ok_or(FftError::LengthTooLarge { len: usize::MAX })
}

fn graph_requirements(
    limits: LargePolicyLimits,
    storage_alignment: u32,
    scratch_bytes: u64,
) -> Result<StageRequirements> {
    StageRequirements::new(
        limits.max_storage_buffer_binding_size,
        limits.max_buffer_size,
        u64::from(storage_alignment.max(1)),
        4,
        scratch_bytes,
    )
}

fn policy_limits(policy: &LargeRoutingPolicy) -> LargePolicyLimits {
    LargePolicyLimits {
        max_storage_buffer_binding_size: policy.max_bind_bytes,
        max_buffer_size: policy.max_buffer_size,
    }
}

fn c2c_range(buffer: LogicalBufferId, offset_bytes: u64, size_bytes: u64) -> Result<LogicalRange> {
    LogicalRange::new(buffer, offset_bytes, size_bytes, ElementFormat::ComplexF32)
}

fn c2c_range_with_format(
    buffer: LogicalBufferId,
    offset_bytes: u64,
    size_bytes: u64,
    format: ElementFormat,
) -> Result<LogicalRange> {
    LogicalRange::new(buffer, offset_bytes, size_bytes, format)
}

fn build_normal_c2c_graph_for_impl(
    route_impl: &C2cRouteImpl,
    required_bytes: u64,
    workspace_bytes: u64,
    element_format: ElementFormat,
    limits: LargePolicyLimits,
) -> Result<LargeExecutionGraph> {
    match route_impl {
        C2cRouteImpl::DirectDft(_) => {
            build_direct_dft_c2c_graph_with_format(required_bytes, element_format, limits)
        }
        C2cRouteImpl::MixedRadix(plan) => build_axis_plan_c2c_graph_with_kinds(
            "c2c-mixed-radix-normal",
            "mixed-radix-stockham-stage",
            "fused-pow2-workgroup-stage",
            "fused-smooth-workgroup-stage",
            "mixed-radix-workspace",
            &plan.graph_stage_kinds(),
            required_bytes,
            workspace_bytes,
            element_format,
            limits,
        ),
        C2cRouteImpl::Rader(plan) => build_normal_rader_c2c_graph(
            plan.graph_helper_buffers(),
            ConvolutionGraphFfts::rader(plan),
            required_bytes,
            element_format,
            limits,
        ),
        C2cRouteImpl::Bluestein(plan) => build_normal_bluestein_c2c_graph(
            plan.graph_helper_buffers(),
            ConvolutionGraphFfts::bluestein(plan),
            required_bytes,
            element_format,
            limits,
        ),
        C2cRouteImpl::AxisSequence(plan) => build_axis_sequence_c2c_graph(
            plan,
            required_bytes,
            workspace_bytes,
            element_format,
            limits,
        ),
    }
}

fn helper_range(
    index: u32,
    offset_bytes: u64,
    size_bytes: u64,
    format: ElementFormat,
) -> Result<LogicalRange> {
    LogicalRange::new(
        LogicalBufferId::Temp(index),
        offset_bytes,
        size_bytes,
        format,
    )
}

fn helper_range_from_info(helper: HelperBufferRange) -> Result<LogicalRange> {
    helper_range_from_info_with_base(helper, 0)
}

fn helper_range_from_info_with_base(
    helper: HelperBufferRange,
    index_base: u32,
) -> Result<LogicalRange> {
    helper_range(
        index_base + helper.index,
        0,
        helper.size_bytes,
        helper.format,
    )
}

fn stage_range(slot: SequenceBufferSlot, size_bytes: u64) -> Result<LogicalRange> {
    stage_range_with_format(slot, size_bytes, ElementFormat::ComplexF32)
}

fn stage_range_with_format(
    slot: SequenceBufferSlot,
    size_bytes: u64,
    format: ElementFormat,
) -> Result<LogicalRange> {
    let buffer = match slot {
        SequenceBufferSlot::Input => LogicalBufferId::Input,
        SequenceBufferSlot::Output => LogicalBufferId::Output,
        SequenceBufferSlot::Temp => LogicalBufferId::Temp(0),
    };
    c2c_range_with_format(buffer, 0, size_bytes, format)
}

fn stage_range_with_temp(
    slot: SequenceBufferSlot,
    input: LogicalRange,
    output: LogicalRange,
    temp: LogicalRange,
) -> LogicalRange {
    match slot {
        SequenceBufferSlot::Input => input,
        SequenceBufferSlot::Output => output,
        SequenceBufferSlot::Temp => temp,
    }
}

fn push_helper_windows<I>(
    graph: &mut LargeExecutionGraph,
    helpers: I,
    limits: LargePolicyLimits,
) -> Result<()>
where
    I: IntoIterator<Item = HelperBufferRange>,
{
    push_helper_windows_with_base(graph, helpers, 0, limits)
}

fn push_helper_windows_with_base<I>(
    graph: &mut LargeExecutionGraph,
    helpers: I,
    index_base: u32,
    limits: LargePolicyLimits,
) -> Result<()>
where
    I: IntoIterator<Item = HelperBufferRange>,
{
    for helper in helpers {
        graph.push_stage(
            LargeStage::HelperWindow {
                label: helper.label,
                range: helper_range_from_info_with_base(helper, index_base)?,
            },
            graph_requirements_covering(limits, 1, helper.size_bytes, helper.size_bytes)?,
        )?;
    }
    Ok(())
}

fn graph_requirements_covering(
    limits: LargePolicyLimits,
    storage_alignment: u32,
    range_bytes: u64,
    scratch_bytes: u64,
) -> Result<StageRequirements> {
    graph_requirements(
        LargePolicyLimits {
            max_storage_buffer_binding_size: limits.max_storage_buffer_binding_size,
            max_buffer_size: limits.max_buffer_size.max(range_bytes),
        },
        storage_alignment,
        scratch_bytes,
    )
}

fn work_items_for_bytes(bytes: u64, format: ElementFormat) -> u64 {
    (bytes / format.bytes_per_element()).max(1)
}

#[derive(Debug, Clone)]
struct ConvolutionGraphFfts {
    fused: bool,
    forward_stage_kinds: Vec<AxisStageKind>,
    forward_workspace_bytes: u64,
    inverse_stage_kinds: Vec<AxisStageKind>,
    inverse_workspace_bytes: u64,
}

impl ConvolutionGraphFfts {
    fn rader(plan: &RaderAxis) -> Self {
        Self {
            fused: plan.graph_is_fused(),
            forward_stage_kinds: plan.graph_forward_fft_stage_kinds(),
            forward_workspace_bytes: plan.graph_forward_fft_workspace_bytes(),
            inverse_stage_kinds: plan.graph_inverse_fft_stage_kinds(),
            inverse_workspace_bytes: plan.graph_inverse_fft_workspace_bytes(),
        }
    }

    fn bluestein(plan: &BluesteinAxis) -> Self {
        Self {
            fused: plan.graph_is_fused(),
            forward_stage_kinds: plan.graph_forward_fft_stage_kinds(),
            forward_workspace_bytes: plan.graph_forward_fft_workspace_bytes(),
            inverse_stage_kinds: plan.graph_inverse_fft_stage_kinds(),
            inverse_workspace_bytes: plan.graph_inverse_fft_workspace_bytes(),
        }
    }
}

#[derive(Debug, Clone)]
enum AxisSequenceGraphStep {
    Mixed {
        stage_kinds: Vec<AxisStageKind>,
        workspace_bytes: u64,
    },
    Rader {
        helpers: Vec<HelperBufferRange>,
        convolution: ConvolutionGraphFfts,
    },
    Bluestein {
        helpers: Vec<HelperBufferRange>,
        convolution: ConvolutionGraphFfts,
    },
}

#[cfg(test)]
fn build_direct_dft_c2c_graph(
    required_bytes: u64,
    limits: LargePolicyLimits,
) -> Result<LargeExecutionGraph> {
    build_direct_dft_c2c_graph_with_format(required_bytes, ElementFormat::ComplexF32, limits)
}

fn build_direct_dft_c2c_graph_with_format(
    required_bytes: u64,
    element_format: ElementFormat,
    limits: LargePolicyLimits,
) -> Result<LargeExecutionGraph> {
    let mut graph = LargeExecutionGraph::new("c2c-direct-dft-normal");
    graph.push_stage(
        LargeStage::HostWindow {
            label: "logical-input",
            range: c2c_range_with_format(
                LogicalBufferId::Input,
                0,
                required_bytes,
                element_format,
            )?,
        },
        graph_requirements_covering(limits, 1, required_bytes, 0)?,
    )?;
    graph.push_stage(
        LargeStage::Kernel {
            label: C2cRoute::DirectDft.graph_label(),
            input: c2c_range_with_format(
                LogicalBufferId::Input,
                0,
                required_bytes,
                element_format,
            )?,
            output: c2c_range_with_format(
                LogicalBufferId::Output,
                0,
                required_bytes,
                element_format,
            )?,
            work_items: work_items_for_bytes(required_bytes, element_format),
        },
        graph_requirements_covering(limits, 1, required_bytes, 0)?,
    )?;
    graph.push_stage(
        LargeStage::HostWindow {
            label: "logical-output",
            range: c2c_range_with_format(
                LogicalBufferId::Output,
                0,
                required_bytes,
                element_format,
            )?,
        },
        graph_requirements_covering(limits, 1, required_bytes, 0)?,
    )?;
    Ok(graph)
}

#[cfg(test)]
fn build_axis_plan_c2c_graph(
    graph_label: &'static str,
    kernel_label: &'static str,
    workspace_label: &'static str,
    stage_count: usize,
    required_bytes: u64,
    workspace_bytes: u64,
    limits: LargePolicyLimits,
) -> Result<LargeExecutionGraph> {
    let stage_labels = vec![kernel_label; stage_count];
    build_axis_plan_c2c_graph_with_labels(
        graph_label,
        &stage_labels,
        workspace_label,
        required_bytes,
        workspace_bytes,
        limits,
    )
}

#[allow(clippy::too_many_arguments)]
fn build_axis_plan_c2c_graph_with_kinds(
    graph_label: &'static str,
    stockham_label: &'static str,
    fused_pow2_label: &'static str,
    fused_smooth_label: &'static str,
    workspace_label: &'static str,
    stage_kinds: &[AxisStageKind],
    required_bytes: u64,
    workspace_bytes: u64,
    element_format: ElementFormat,
    limits: LargePolicyLimits,
) -> Result<LargeExecutionGraph> {
    let stage_labels = stage_kinds
        .iter()
        .map(|kind| match kind {
            AxisStageKind::Stockham { .. } => stockham_label,
            AxisStageKind::FusedPow2 { .. } => fused_pow2_label,
            AxisStageKind::FusedSmooth { .. } => fused_smooth_label,
        })
        .collect::<Vec<_>>();
    build_axis_plan_c2c_graph_with_labels_and_format(
        graph_label,
        &stage_labels,
        workspace_label,
        required_bytes,
        workspace_bytes,
        element_format,
        limits,
    )
}

#[cfg(test)]
fn build_axis_plan_c2c_graph_with_labels(
    graph_label: &'static str,
    stage_labels: &[&'static str],
    workspace_label: &'static str,
    required_bytes: u64,
    workspace_bytes: u64,
    limits: LargePolicyLimits,
) -> Result<LargeExecutionGraph> {
    build_axis_plan_c2c_graph_with_labels_and_format(
        graph_label,
        stage_labels,
        workspace_label,
        required_bytes,
        workspace_bytes,
        ElementFormat::ComplexF32,
        limits,
    )
}

fn build_axis_plan_c2c_graph_with_labels_and_format(
    graph_label: &'static str,
    stage_labels: &[&'static str],
    workspace_label: &'static str,
    required_bytes: u64,
    workspace_bytes: u64,
    element_format: ElementFormat,
    limits: LargePolicyLimits,
) -> Result<LargeExecutionGraph> {
    let mut graph = LargeExecutionGraph::new(graph_label);
    graph.push_stage(
        LargeStage::HostWindow {
            label: "logical-input",
            range: c2c_range_with_format(
                LogicalBufferId::Input,
                0,
                required_bytes,
                element_format,
            )?,
        },
        graph_requirements_covering(limits, 1, required_bytes, 0)?,
    )?;
    add_axis_plan_kernel_stages_with_labels(
        &mut graph,
        stage_labels,
        workspace_label,
        c2c_range_with_format(LogicalBufferId::Input, 0, required_bytes, element_format)?,
        c2c_range_with_format(LogicalBufferId::Output, 0, required_bytes, element_format)?,
        0,
        workspace_bytes,
        limits,
    )?;

    graph.push_stage(
        LargeStage::HostWindow {
            label: "logical-output",
            range: c2c_range_with_format(
                LogicalBufferId::Output,
                0,
                required_bytes,
                element_format,
            )?,
        },
        graph_requirements_covering(limits, 1, required_bytes, 0)?,
    )?;
    Ok(graph)
}

#[allow(clippy::too_many_arguments)]
fn add_axis_plan_kernel_stages_with_kinds(
    graph: &mut LargeExecutionGraph,
    stockham_label: &'static str,
    fused_pow2_label: &'static str,
    fused_smooth_label: &'static str,
    workspace_label: &'static str,
    stage_kinds: &[AxisStageKind],
    input: LogicalRange,
    output: LogicalRange,
    temp_index: u32,
    workspace_bytes: u64,
    limits: LargePolicyLimits,
) -> Result<()> {
    let stage_labels = stage_kinds
        .iter()
        .map(|kind| match kind {
            AxisStageKind::Stockham { .. } => stockham_label,
            AxisStageKind::FusedPow2 { .. } => fused_pow2_label,
            AxisStageKind::FusedSmooth { .. } => fused_smooth_label,
        })
        .collect::<Vec<_>>();
    add_axis_plan_kernel_stages_with_labels(
        graph,
        &stage_labels,
        workspace_label,
        input,
        output,
        temp_index,
        workspace_bytes,
        limits,
    )
}

#[allow(clippy::too_many_arguments)]
fn add_axis_plan_kernel_stages_with_labels(
    graph: &mut LargeExecutionGraph,
    stage_labels: &[&'static str],
    workspace_label: &'static str,
    input: LogicalRange,
    output: LogicalRange,
    temp_index: u32,
    workspace_bytes: u64,
    limits: LargePolicyLimits,
) -> Result<()> {
    let element_format = input.format;
    let temp = c2c_range_with_format(
        LogicalBufferId::Temp(temp_index),
        0,
        workspace_bytes.max(element_format.bytes_per_element()),
        element_format,
    )?;
    if workspace_bytes > 0 {
        graph.push_stage(
            LargeStage::HelperWindow {
                label: workspace_label,
                range: c2c_range_with_format(
                    LogicalBufferId::Temp(temp_index),
                    0,
                    workspace_bytes,
                    element_format,
                )?,
            },
            graph_requirements_covering(limits, 1, workspace_bytes, workspace_bytes)?,
        )?;
    }

    let stage_count = stage_labels.len();
    let mut src_slot = SequenceBufferSlot::Input;
    let mut dst_slot = if stage_count % 2 == 1 {
        SequenceBufferSlot::Output
    } else {
        SequenceBufferSlot::Temp
    };
    for (stage_index, &stage_label) in stage_labels.iter().enumerate() {
        graph.push_stage(
            LargeStage::Kernel {
                label: stage_label,
                input: stage_range_with_temp(src_slot, input, output, temp),
                output: stage_range_with_temp(dst_slot, input, output, temp),
                work_items: work_items_for_bytes(input.size_bytes, element_format),
            },
            graph_requirements_covering(limits, 1, input.size_bytes.max(output.size_bytes), 0)?,
        )?;
        if stage_index + 1 < stage_count {
            src_slot = dst_slot;
            dst_slot = next_sequence_destination(
                "axis-plan-buffer-flow",
                src_slot,
                "mixed-radix stage buffer flow attempted to use input as destination",
            )?;
        }
    }
    Ok(())
}

fn build_normal_rader_c2c_graph(
    helpers: Vec<HelperBufferRange>,
    convolution: ConvolutionGraphFfts,
    required_bytes: u64,
    element_format: ElementFormat,
    limits: LargePolicyLimits,
) -> Result<LargeExecutionGraph> {
    let mut graph = LargeExecutionGraph::new("c2c-rader-normal");
    graph.push_stage(
        LargeStage::HostWindow {
            label: "logical-input",
            range: c2c_range_with_format(
                LogicalBufferId::Input,
                0,
                required_bytes,
                element_format,
            )?,
        },
        graph_requirements_covering(limits, 1, required_bytes, 0)?,
    )?;
    push_helper_windows(&mut graph, helpers.iter().copied(), limits)?;

    let input = c2c_range_with_format(LogicalBufferId::Input, 0, required_bytes, element_format)?;
    let output = c2c_range_with_format(LogicalBufferId::Output, 0, required_bytes, element_format)?;
    if convolution.fused {
        graph.push_stage(
            LargeStage::Kernel {
                label: "rader-fused-workgroup-stage",
                input,
                output,
                work_items: work_items_for_bytes(required_bytes, element_format),
            },
            graph_requirements_covering(limits, 1, required_bytes, 0)?,
        )?;
        graph.push_stage(
            LargeStage::HostWindow {
                label: "logical-output",
                range: output,
            },
            graph_requirements_covering(limits, 1, required_bytes, 0)?,
        )?;
        return Ok(graph);
    }

    let [perm, _bfft, sum, x0, work, fft] = helpers.as_slice() else {
        return Err(FftError::LargeGraphStageUnsupported {
            stage: "rader-normal-helpers",
            reason: "non-fused Rader graph requires six helper buffers",
        });
    };
    let (perm, sum, x0, work, fft) = (*perm, *sum, *x0, *work, *fft);
    let work_range = helper_range_from_info(work)?;
    let fft_range = helper_range_from_info(fft)?;

    graph.push_stage(
        LargeStage::Kernel {
            label: "rader-sum",
            input,
            output: helper_range_from_info(sum)?,
            work_items: work_items_for_bytes(required_bytes, element_format),
        },
        graph_requirements_covering(
            limits,
            1,
            required_bytes.max(sum.size_bytes),
            sum.size_bytes.max(x0.size_bytes),
        )?,
    )?;
    graph.push_stage(
        LargeStage::Kernel {
            label: "rader-pack",
            input,
            output: work_range,
            work_items: work_items_for_bytes(work.size_bytes, element_format),
        },
        graph_requirements_covering(
            limits,
            1,
            required_bytes.max(work.size_bytes),
            work.size_bytes.max(perm.size_bytes),
        )?,
    )?;
    add_axis_plan_kernel_stages_with_kinds(
        &mut graph,
        "rader-forward-stockham-stage",
        "rader-forward-fused-pow2-stage",
        "rader-forward-fused-smooth-stage",
        "rader-forward-workspace",
        &convolution.forward_stage_kinds,
        work_range,
        fft_range,
        8,
        convolution.forward_workspace_bytes,
        limits,
    )?;
    graph.push_stage(
        LargeStage::Kernel {
            label: "rader-mul",
            input: fft_range,
            output: fft_range,
            work_items: work_items_for_bytes(fft.size_bytes, element_format),
        },
        graph_requirements_covering(limits, 1, fft.size_bytes, fft.size_bytes)?,
    )?;
    add_axis_plan_kernel_stages_with_kinds(
        &mut graph,
        "rader-inverse-stockham-stage",
        "rader-inverse-fused-pow2-stage",
        "rader-inverse-fused-smooth-stage",
        "rader-inverse-workspace",
        &convolution.inverse_stage_kinds,
        fft_range,
        work_range,
        9,
        convolution.inverse_workspace_bytes,
        limits,
    )?;
    graph.push_stage(
        LargeStage::Kernel {
            label: "rader-write-y0",
            input: helper_range_from_info(sum)?,
            output,
            work_items: work_items_for_bytes(sum.size_bytes, element_format),
        },
        graph_requirements_covering(
            limits,
            1,
            required_bytes.max(sum.size_bytes),
            sum.size_bytes,
        )?,
    )?;
    graph.push_stage(
        LargeStage::Kernel {
            label: "rader-post",
            input: work_range,
            output,
            work_items: work_items_for_bytes(required_bytes, element_format),
        },
        graph_requirements_covering(
            limits,
            1,
            required_bytes.max(work.size_bytes),
            work.size_bytes.max(x0.size_bytes),
        )?,
    )?;
    graph.push_stage(
        LargeStage::HostWindow {
            label: "logical-output",
            range: output,
        },
        graph_requirements_covering(limits, 1, required_bytes, 0)?,
    )?;
    Ok(graph)
}

fn build_normal_bluestein_c2c_graph(
    helpers: Vec<HelperBufferRange>,
    convolution: ConvolutionGraphFfts,
    required_bytes: u64,
    element_format: ElementFormat,
    limits: LargePolicyLimits,
) -> Result<LargeExecutionGraph> {
    let mut graph = LargeExecutionGraph::new("c2c-bluestein-normal");
    graph.push_stage(
        LargeStage::HostWindow {
            label: "logical-input",
            range: c2c_range_with_format(
                LogicalBufferId::Input,
                0,
                required_bytes,
                element_format,
            )?,
        },
        graph_requirements_covering(limits, 1, required_bytes, 0)?,
    )?;
    push_helper_windows(&mut graph, helpers.iter().copied(), limits)?;

    let input = c2c_range_with_format(LogicalBufferId::Input, 0, required_bytes, element_format)?;
    let output = c2c_range_with_format(LogicalBufferId::Output, 0, required_bytes, element_format)?;
    if convolution.fused {
        graph.push_stage(
            LargeStage::Kernel {
                label: "bluestein-fused-workgroup-stage",
                input,
                output,
                work_items: work_items_for_bytes(required_bytes, element_format),
            },
            graph_requirements_covering(limits, 1, required_bytes, 0)?,
        )?;
        graph.push_stage(
            LargeStage::HostWindow {
                label: "logical-output",
                range: output,
            },
            graph_requirements_covering(limits, 1, required_bytes, 0)?,
        )?;
        return Ok(graph);
    }

    let [chirp, _bfft, work, fft] = helpers.as_slice() else {
        return Err(FftError::LargeGraphStageUnsupported {
            stage: "bluestein-normal-helpers",
            reason: "non-fused Bluestein graph requires four helper buffers",
        });
    };
    let (chirp, work, fft) = (*chirp, *work, *fft);
    let work_range = helper_range_from_info(work)?;
    let fft_range = helper_range_from_info(fft)?;

    graph.push_stage(
        LargeStage::Kernel {
            label: "bluestein-pack",
            input,
            output: work_range,
            work_items: work_items_for_bytes(work.size_bytes, element_format),
        },
        graph_requirements_covering(
            limits,
            1,
            required_bytes.max(work.size_bytes),
            work.size_bytes.max(chirp.size_bytes),
        )?,
    )?;
    add_axis_plan_kernel_stages_with_kinds(
        &mut graph,
        "bluestein-forward-stockham-stage",
        "bluestein-forward-fused-pow2-stage",
        "bluestein-forward-fused-smooth-stage",
        "bluestein-forward-workspace",
        &convolution.forward_stage_kinds,
        work_range,
        fft_range,
        8,
        convolution.forward_workspace_bytes,
        limits,
    )?;
    graph.push_stage(
        LargeStage::Kernel {
            label: "bluestein-mul",
            input: fft_range,
            output: fft_range,
            work_items: work_items_for_bytes(fft.size_bytes, element_format),
        },
        graph_requirements_covering(limits, 1, fft.size_bytes, fft.size_bytes)?,
    )?;
    add_axis_plan_kernel_stages_with_kinds(
        &mut graph,
        "bluestein-inverse-stockham-stage",
        "bluestein-inverse-fused-pow2-stage",
        "bluestein-inverse-fused-smooth-stage",
        "bluestein-inverse-workspace",
        &convolution.inverse_stage_kinds,
        fft_range,
        work_range,
        9,
        convolution.inverse_workspace_bytes,
        limits,
    )?;
    graph.push_stage(
        LargeStage::Kernel {
            label: "bluestein-post",
            input: work_range,
            output,
            work_items: work_items_for_bytes(required_bytes, element_format),
        },
        graph_requirements_covering(
            limits,
            1,
            required_bytes.max(work.size_bytes),
            work.size_bytes.max(chirp.size_bytes),
        )?,
    )?;
    graph.push_stage(
        LargeStage::HostWindow {
            label: "logical-output",
            range: output,
        },
        graph_requirements_covering(limits, 1, required_bytes, 0)?,
    )?;
    Ok(graph)
}

fn build_axis_sequence_c2c_graph(
    plan: &AxisSequencePlan,
    required_bytes: u64,
    workspace_bytes: u64,
    element_format: ElementFormat,
    limits: LargePolicyLimits,
) -> Result<LargeExecutionGraph> {
    build_axis_sequence_c2c_graph_from_steps(
        &plan.graph_steps(),
        required_bytes,
        workspace_bytes,
        element_format,
        limits,
    )
}

fn build_axis_sequence_c2c_graph_from_steps(
    steps: &[AxisSequenceGraphStep],
    required_bytes: u64,
    workspace_bytes: u64,
    element_format: ElementFormat,
    limits: LargePolicyLimits,
) -> Result<LargeExecutionGraph> {
    let mut graph = LargeExecutionGraph::new("c2c-axis-sequence-normal");
    graph.push_stage(
        LargeStage::HostWindow {
            label: "logical-input",
            range: c2c_range_with_format(
                LogicalBufferId::Input,
                0,
                required_bytes,
                element_format,
            )?,
        },
        graph_requirements_covering(limits, 1, required_bytes, 0)?,
    )?;
    if workspace_bytes > 0 {
        graph.push_stage(
            LargeStage::HelperWindow {
                label: "axis-sequence-workspace",
                range: c2c_range_with_format(
                    LogicalBufferId::Temp(0),
                    0,
                    workspace_bytes,
                    element_format,
                )?,
            },
            graph_requirements_covering(limits, 1, workspace_bytes, workspace_bytes)?,
        )?;
    }

    let mut src_slot = SequenceBufferSlot::Input;
    let mut dst_slot = if steps.len() % 2 == 1 {
        SequenceBufferSlot::Output
    } else {
        SequenceBufferSlot::Temp
    };
    for (step_index, step) in steps.iter().cloned().enumerate() {
        let input = stage_range_with_format(src_slot, required_bytes, element_format)?;
        let output = stage_range_with_format(dst_slot, required_bytes, element_format)?;
        let helper_base = 16 + step_index as u32 * 16;
        match step {
            AxisSequenceGraphStep::Mixed {
                stage_kinds,
                workspace_bytes,
            } => add_axis_plan_kernel_stages_with_kinds(
                &mut graph,
                "axis-sequence-mixed-stockham-stage",
                "axis-sequence-mixed-fused-pow2-stage",
                "axis-sequence-mixed-fused-smooth-stage",
                "axis-sequence-mixed-workspace",
                &stage_kinds,
                input,
                output,
                helper_base,
                workspace_bytes,
                limits,
            )?,
            AxisSequenceGraphStep::Rader {
                helpers,
                convolution,
            } => add_rader_c2c_stages(
                &mut graph,
                input,
                output,
                helpers,
                convolution,
                helper_base,
                required_bytes,
                limits,
            )?,
            AxisSequenceGraphStep::Bluestein {
                helpers,
                convolution,
            } => add_bluestein_c2c_stages(
                &mut graph,
                input,
                output,
                helpers,
                convolution,
                helper_base,
                required_bytes,
                limits,
            )?,
        }

        if step_index + 1 < steps.len() {
            src_slot = dst_slot;
            dst_slot = next_sequence_destination(
                "axis-sequence-buffer-flow",
                src_slot,
                "axis sequence buffer flow attempted to use input as destination",
            )?;
        }
    }

    graph.push_stage(
        LargeStage::HostWindow {
            label: "logical-output",
            range: c2c_range_with_format(
                LogicalBufferId::Output,
                0,
                required_bytes,
                element_format,
            )?,
        },
        graph_requirements_covering(limits, 1, required_bytes, 0)?,
    )?;
    Ok(graph)
}

fn add_rader_c2c_stages(
    graph: &mut LargeExecutionGraph,
    input: LogicalRange,
    output: LogicalRange,
    helpers: Vec<HelperBufferRange>,
    convolution: ConvolutionGraphFfts,
    helper_index_base: u32,
    required_bytes: u64,
    limits: LargePolicyLimits,
) -> Result<()> {
    push_helper_windows_with_base(graph, helpers.iter().copied(), helper_index_base, limits)?;
    let element_format = input.format;

    if convolution.fused {
        graph.push_stage(
            LargeStage::Kernel {
                label: "rader-fused-workgroup-stage",
                input,
                output,
                work_items: work_items_for_bytes(required_bytes, element_format),
            },
            graph_requirements_covering(limits, 1, required_bytes, 0)?,
        )?;
        return Ok(());
    }

    let [perm, _bfft, sum, x0, work, fft] = helpers.as_slice() else {
        return Err(FftError::LargeGraphStageUnsupported {
            stage: "rader-axis-sequence-helpers",
            reason: "non-fused Rader graph requires six helper buffers",
        });
    };
    let (perm, sum, x0, work, fft) = (*perm, *sum, *x0, *work, *fft);
    let work_range = helper_range_from_info_with_base(work, helper_index_base)?;
    let fft_range = helper_range_from_info_with_base(fft, helper_index_base)?;

    graph.push_stage(
        LargeStage::Kernel {
            label: "rader-sum",
            input,
            output: helper_range_from_info_with_base(sum, helper_index_base)?,
            work_items: work_items_for_bytes(required_bytes, element_format),
        },
        graph_requirements_covering(
            limits,
            1,
            required_bytes.max(sum.size_bytes),
            sum.size_bytes.max(x0.size_bytes),
        )?,
    )?;
    graph.push_stage(
        LargeStage::Kernel {
            label: "rader-pack",
            input,
            output: work_range,
            work_items: work_items_for_bytes(work.size_bytes, element_format),
        },
        graph_requirements_covering(
            limits,
            1,
            required_bytes.max(work.size_bytes),
            work.size_bytes.max(perm.size_bytes),
        )?,
    )?;
    add_axis_plan_kernel_stages_with_kinds(
        graph,
        "rader-forward-stockham-stage",
        "rader-forward-fused-pow2-stage",
        "rader-forward-fused-smooth-stage",
        "rader-forward-workspace",
        &convolution.forward_stage_kinds,
        work_range,
        fft_range,
        helper_index_base + 8,
        convolution.forward_workspace_bytes,
        limits,
    )?;
    graph.push_stage(
        LargeStage::Kernel {
            label: "rader-mul",
            input: fft_range,
            output: fft_range,
            work_items: work_items_for_bytes(fft.size_bytes, element_format),
        },
        graph_requirements_covering(limits, 1, fft.size_bytes, fft.size_bytes)?,
    )?;
    add_axis_plan_kernel_stages_with_kinds(
        graph,
        "rader-inverse-stockham-stage",
        "rader-inverse-fused-pow2-stage",
        "rader-inverse-fused-smooth-stage",
        "rader-inverse-workspace",
        &convolution.inverse_stage_kinds,
        fft_range,
        work_range,
        helper_index_base + 9,
        convolution.inverse_workspace_bytes,
        limits,
    )?;
    graph.push_stage(
        LargeStage::Kernel {
            label: "rader-write-y0",
            input: helper_range_from_info_with_base(sum, helper_index_base)?,
            output,
            work_items: work_items_for_bytes(sum.size_bytes, element_format),
        },
        graph_requirements_covering(
            limits,
            1,
            required_bytes.max(sum.size_bytes),
            sum.size_bytes,
        )?,
    )?;
    graph.push_stage(
        LargeStage::Kernel {
            label: "rader-post",
            input: work_range,
            output,
            work_items: work_items_for_bytes(required_bytes, element_format),
        },
        graph_requirements_covering(
            limits,
            1,
            required_bytes.max(work.size_bytes),
            work.size_bytes.max(x0.size_bytes),
        )?,
    )?;
    Ok(())
}

fn add_bluestein_c2c_stages(
    graph: &mut LargeExecutionGraph,
    input: LogicalRange,
    output: LogicalRange,
    helpers: Vec<HelperBufferRange>,
    convolution: ConvolutionGraphFfts,
    helper_index_base: u32,
    required_bytes: u64,
    limits: LargePolicyLimits,
) -> Result<()> {
    push_helper_windows_with_base(graph, helpers.iter().copied(), helper_index_base, limits)?;
    let element_format = input.format;

    if convolution.fused {
        graph.push_stage(
            LargeStage::Kernel {
                label: "bluestein-fused-workgroup-stage",
                input,
                output,
                work_items: work_items_for_bytes(required_bytes, element_format),
            },
            graph_requirements_covering(limits, 1, required_bytes, 0)?,
        )?;
        return Ok(());
    }

    let [chirp, _bfft, work, fft] = helpers.as_slice() else {
        return Err(FftError::LargeGraphStageUnsupported {
            stage: "bluestein-axis-sequence-helpers",
            reason: "non-fused Bluestein graph requires four helper buffers",
        });
    };
    let (chirp, work, fft) = (*chirp, *work, *fft);
    let work_range = helper_range_from_info_with_base(work, helper_index_base)?;
    let fft_range = helper_range_from_info_with_base(fft, helper_index_base)?;

    graph.push_stage(
        LargeStage::Kernel {
            label: "bluestein-pack",
            input,
            output: work_range,
            work_items: work_items_for_bytes(work.size_bytes, element_format),
        },
        graph_requirements_covering(
            limits,
            1,
            required_bytes.max(work.size_bytes),
            work.size_bytes.max(chirp.size_bytes),
        )?,
    )?;
    add_axis_plan_kernel_stages_with_kinds(
        graph,
        "bluestein-forward-stockham-stage",
        "bluestein-forward-fused-pow2-stage",
        "bluestein-forward-fused-smooth-stage",
        "bluestein-forward-workspace",
        &convolution.forward_stage_kinds,
        work_range,
        fft_range,
        helper_index_base + 8,
        convolution.forward_workspace_bytes,
        limits,
    )?;
    graph.push_stage(
        LargeStage::Kernel {
            label: "bluestein-mul",
            input: fft_range,
            output: fft_range,
            work_items: work_items_for_bytes(fft.size_bytes, element_format),
        },
        graph_requirements_covering(limits, 1, fft.size_bytes, fft.size_bytes)?,
    )?;
    add_axis_plan_kernel_stages_with_kinds(
        graph,
        "bluestein-inverse-stockham-stage",
        "bluestein-inverse-fused-pow2-stage",
        "bluestein-inverse-fused-smooth-stage",
        "bluestein-inverse-workspace",
        &convolution.inverse_stage_kinds,
        fft_range,
        work_range,
        helper_index_base + 9,
        convolution.inverse_workspace_bytes,
        limits,
    )?;
    graph.push_stage(
        LargeStage::Kernel {
            label: "bluestein-post",
            input: work_range,
            output,
            work_items: work_items_for_bytes(required_bytes, element_format),
        },
        graph_requirements_covering(
            limits,
            1,
            required_bytes.max(work.size_bytes),
            work.size_bytes.max(chirp.size_bytes),
        )?,
    )?;
    Ok(())
}

fn build_large_chunk_c2c_graph(
    plan: LargeChunkPlan,
    child_graph: &LargeExecutionGraph,
    limits: LargePolicyLimits,
    storage_alignment: u32,
) -> Result<LargeExecutionPlan> {
    let mut graph = LargeExecutionGraph::new("c2c-large-chunk");
    for range in plan.ranges() {
        let range = range?;
        let input = c2c_range(LogicalBufferId::Input, range.byte_offset, range.byte_size)?;
        let input_stage = c2c_range(LogicalBufferId::Stage(0), 0, range.byte_size)?;
        graph.push_stage(
            LargeStage::Copy {
                label: "large-chunk-copy-input",
                src: input,
                dst: input_stage,
            },
            graph_requirements(limits, storage_alignment, range.byte_size)?,
        )?;

        append_child_c2c_graph(
            &mut graph,
            child_graph,
            c2c_range(LogicalBufferId::Stage(0), 0, plan.staging_size_bytes())?,
            c2c_range(LogicalBufferId::Stage(1), 0, plan.staging_size_bytes())?,
            limits,
            storage_alignment,
            0,
            2,
        )?;

        let output_stage = c2c_range(LogicalBufferId::Stage(1), 0, range.byte_size)?;
        let output = c2c_range(LogicalBufferId::Output, range.byte_offset, range.byte_size)?;
        graph.push_stage(
            LargeStage::Copy {
                label: "large-chunk-copy-output",
                src: output_stage,
                dst: output,
            },
            graph_requirements(limits, storage_alignment, range.byte_size)?,
        )?;
    }
    Ok(LargeExecutionPlan::new(graph))
}

fn build_large_axis_sequence_c2c_graph(
    child_graphs: &[LargeExecutionGraph],
    required_bytes: u64,
    storage_alignment: u32,
    limits: LargePolicyLimits,
) -> Result<LargeExecutionPlan> {
    let mut graph = LargeExecutionGraph::new("c2c-large-axis-sequence");
    graph.push_stage(
        LargeStage::HostWindow {
            label: "logical-input",
            range: c2c_range(LogicalBufferId::Input, 0, required_bytes)?,
        },
        graph_requirements_covering(limits, storage_alignment, required_bytes, 0)?,
    )?;
    if child_graphs.len() > 1 {
        graph.push_stage(
            LargeStage::HelperWindow {
                label: "large-axis-sequence-workspace",
                range: c2c_range(LogicalBufferId::Temp(0), 0, required_bytes)?,
            },
            graph_requirements_covering(limits, storage_alignment, required_bytes, required_bytes)?,
        )?;
    }

    let mut src_slot = SequenceBufferSlot::Input;
    let mut dst_slot = if child_graphs.len() % 2 == 1 {
        SequenceBufferSlot::Output
    } else {
        SequenceBufferSlot::Temp
    };
    for (step_index, child_graph) in child_graphs.iter().enumerate() {
        let input = stage_range(src_slot, required_bytes)?;
        let output = stage_range(dst_slot, required_bytes)?;
        append_child_c2c_graph_with_bases(
            &mut graph,
            child_graph,
            input,
            output,
            limits,
            storage_alignment,
            large_axis_sequence_temp_base(step_index)?,
            large_axis_sequence_stage_base(step_index)?,
        )?;

        if step_index + 1 < child_graphs.len() {
            src_slot = dst_slot;
            dst_slot = next_sequence_destination(
                "large-axis-sequence-buffer-flow",
                src_slot,
                "large axis sequence buffer flow attempted to use input as destination",
            )?;
        }
    }

    graph.push_stage(
        LargeStage::HostWindow {
            label: "logical-output",
            range: c2c_range(LogicalBufferId::Output, 0, required_bytes)?,
        },
        graph_requirements_covering(limits, storage_alignment, required_bytes, 0)?,
    )?;
    Ok(LargeExecutionPlan::new(graph))
}

fn append_child_c2c_graph(
    graph: &mut LargeExecutionGraph,
    child_graph: &LargeExecutionGraph,
    child_input: LogicalRange,
    child_output: LogicalRange,
    limits: LargePolicyLimits,
    storage_alignment: u32,
    temp_index_base: u32,
    stage_index_base: u32,
) -> Result<()> {
    append_child_c2c_graph_with_bases(
        graph,
        child_graph,
        child_input,
        child_output,
        limits,
        storage_alignment,
        temp_index_base,
        stage_index_base,
    )
}

fn large_axis_sequence_temp_base(step_index: usize) -> Result<u32> {
    let index =
        u32::try_from(step_index).map_err(|_| FftError::LengthTooLarge { len: usize::MAX })?;
    128u32
        .checked_add(
            index
                .checked_mul(64)
                .ok_or(FftError::LengthTooLarge { len: usize::MAX })?,
        )
        .ok_or(FftError::LengthTooLarge { len: usize::MAX })
}

fn large_axis_sequence_stage_base(step_index: usize) -> Result<u32> {
    let index =
        u32::try_from(step_index).map_err(|_| FftError::LengthTooLarge { len: usize::MAX })?;
    32u32
        .checked_add(
            index
                .checked_mul(32)
                .ok_or(FftError::LengthTooLarge { len: usize::MAX })?,
        )
        .ok_or(FftError::LengthTooLarge { len: usize::MAX })
}

#[allow(clippy::too_many_arguments)]
fn append_child_c2c_graph_with_bases(
    graph: &mut LargeExecutionGraph,
    child_graph: &LargeExecutionGraph,
    child_input: LogicalRange,
    child_output: LogicalRange,
    limits: LargePolicyLimits,
    storage_alignment: u32,
    temp_index_base: u32,
    stage_index_base: u32,
) -> Result<()> {
    for stage in child_graph.stages() {
        let mapped = remap_child_c2c_stage(
            stage,
            child_input,
            child_output,
            temp_index_base,
            stage_index_base,
        )?;
        graph.push_stage(
            mapped,
            graph_requirements_covering(
                limits,
                storage_alignment,
                max_stage_range_bytes(stage),
                stage_scratch_bytes(stage),
            )?,
        )?;
    }
    Ok(())
}

fn remap_child_c2c_stage(
    stage: &LargeStage,
    child_input: LogicalRange,
    child_output: LogicalRange,
    temp_index_base: u32,
    stage_index_base: u32,
) -> Result<LargeStage> {
    Ok(match *stage {
        LargeStage::Copy { label, src, dst } => LargeStage::Copy {
            label,
            src: remap_child_c2c_range(
                src,
                child_input,
                child_output,
                temp_index_base,
                stage_index_base,
            )?,
            dst: remap_child_c2c_range(
                dst,
                child_input,
                child_output,
                temp_index_base,
                stage_index_base,
            )?,
        },
        LargeStage::GatherScatter {
            label,
            src,
            dst,
            stride_elements,
        } => LargeStage::GatherScatter {
            label,
            src: remap_child_c2c_range(
                src,
                child_input,
                child_output,
                temp_index_base,
                stage_index_base,
            )?,
            dst: remap_child_c2c_range(
                dst,
                child_input,
                child_output,
                temp_index_base,
                stage_index_base,
            )?,
            stride_elements,
        },
        LargeStage::HelperWindow { label, range } => LargeStage::HelperWindow {
            label,
            range: remap_child_c2c_range(
                range,
                child_input,
                child_output,
                temp_index_base,
                stage_index_base,
            )?,
        },
        LargeStage::WindowedHelper { label, range } => LargeStage::WindowedHelper {
            label,
            range: remap_child_c2c_range(
                range,
                child_input,
                child_output,
                temp_index_base,
                stage_index_base,
            )?,
        },
        LargeStage::Kernel {
            label,
            input,
            output,
            work_items,
        } => LargeStage::Kernel {
            label,
            input: remap_child_c2c_range(
                input,
                child_input,
                child_output,
                temp_index_base,
                stage_index_base,
            )?,
            output: remap_child_c2c_range(
                output,
                child_input,
                child_output,
                temp_index_base,
                stage_index_base,
            )?,
            work_items,
        },
        LargeStage::WindowedKernel {
            label,
            input,
            output,
            work_items,
        } => LargeStage::WindowedKernel {
            label,
            input: remap_child_c2c_range(
                input,
                child_input,
                child_output,
                temp_index_base,
                stage_index_base,
            )?,
            output: remap_child_c2c_range(
                output,
                child_input,
                child_output,
                temp_index_base,
                stage_index_base,
            )?,
            work_items,
        },
        LargeStage::TwiddleTranspose {
            label,
            input,
            output,
            work_items,
        } => LargeStage::TwiddleTranspose {
            label,
            input: remap_child_c2c_range(
                input,
                child_input,
                child_output,
                temp_index_base,
                stage_index_base,
            )?,
            output: remap_child_c2c_range(
                output,
                child_input,
                child_output,
                temp_index_base,
                stage_index_base,
            )?,
            work_items,
        },
        LargeStage::StripeTranspose {
            label,
            input,
            output,
            work_items,
        } => LargeStage::StripeTranspose {
            label,
            input: remap_child_c2c_range(
                input,
                child_input,
                child_output,
                temp_index_base,
                stage_index_base,
            )?,
            output: remap_child_c2c_range(
                output,
                child_input,
                child_output,
                temp_index_base,
                stage_index_base,
            )?,
            work_items,
        },
        LargeStage::Permutation {
            label,
            input,
            output,
            work_items,
        } => LargeStage::Permutation {
            label,
            input: remap_child_c2c_range(
                input,
                child_input,
                child_output,
                temp_index_base,
                stage_index_base,
            )?,
            output: remap_child_c2c_range(
                output,
                child_input,
                child_output,
                temp_index_base,
                stage_index_base,
            )?,
            work_items,
        },
        LargeStage::Scale {
            label,
            range,
            work_items,
        } => LargeStage::Scale {
            label,
            range: remap_child_c2c_range(
                range,
                child_input,
                child_output,
                temp_index_base,
                stage_index_base,
            )?,
            work_items,
        },
        LargeStage::HostWindow { label, range } => LargeStage::HostWindow {
            label,
            range: remap_child_c2c_range(
                range,
                child_input,
                child_output,
                temp_index_base,
                stage_index_base,
            )?,
        },
    })
}

fn remap_child_c2c_range(
    range: LogicalRange,
    child_input: LogicalRange,
    child_output: LogicalRange,
    temp_index_base: u32,
    stage_index_base: u32,
) -> Result<LogicalRange> {
    let (buffer, base_offset) = match range.buffer {
        LogicalBufferId::Input => (child_input.buffer, child_input.offset_bytes),
        LogicalBufferId::Output => (child_output.buffer, child_output.offset_bytes),
        LogicalBufferId::Temp(index) => (
            LogicalBufferId::Temp(
                index
                    .checked_add(temp_index_base)
                    .ok_or(FftError::LengthTooLarge { len: usize::MAX })?,
            ),
            0,
        ),
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
    LogicalRange::new(buffer, offset_bytes, range.size_bytes, range.format)
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

fn smooth_graph_steps(steps: &[SmoothExecutionStep]) -> Result<Vec<SmoothGraphStep>> {
    steps
        .iter()
        .map(|step| match step {
            SmoothExecutionStep::Mixed(step) => Ok(SmoothGraphStep::Mixed {
                step: step.step.into(),
                stage_kinds: step.plan.graph_stage_kinds(),
                workspace_bytes: step.plan.workspace_size_bytes(),
            }),
            SmoothExecutionStep::Smooth(step) => Ok(SmoothGraphStep::Smooth {
                step: step.step.into(),
                phase1: step.phase1.graph()?,
                phase2: step.phase2.graph()?,
            }),
        })
        .collect()
}

fn build_smooth_c2c_graph(
    steps: &[SmoothGraphStep],
    limits: LargePolicyLimits,
    storage_alignment: u32,
) -> Result<LargeExecutionPlan> {
    let mut graph = LargeExecutionGraph::new("c2c-smooth-decomposition");
    for (index, step) in steps.iter().enumerate() {
        let input_buffer = if index == 0 {
            LogicalBufferId::Input
        } else {
            LogicalBufferId::Temp((index - 1) as u32)
        };
        let output_buffer = if index + 1 == steps.len() {
            LogicalBufferId::Output
        } else {
            LogicalBufferId::Temp(index as u32)
        };
        match step {
            SmoothGraphStep::Mixed {
                step,
                stage_kinds,
                workspace_bytes,
            } => {
                let step = *step;
                let input = c2c_range(input_buffer, 0, step.line_bytes())?;
                let line_input = c2c_range(LogicalBufferId::Stage(0), 0, step.line_bytes())?;
                graph.push_stage(
                    LargeStage::GatherScatter {
                        label: "mixed-axis-gather-line",
                        src: input,
                        dst: line_input,
                        stride_elements: step.stride(),
                    },
                    graph_requirements(limits, storage_alignment, step.line_bytes())?,
                )?;

                let line_output = c2c_range(LogicalBufferId::Stage(1), 0, step.line_bytes())?;
                add_axis_plan_kernel_stages_with_kinds(
                    &mut graph,
                    "mixed-axis-stockham-stage",
                    "mixed-axis-fused-pow2-stage",
                    "mixed-axis-fused-smooth-stage",
                    "mixed-axis-workspace",
                    stage_kinds,
                    line_input,
                    line_output,
                    smooth_graph_temp_base(index, 0)?,
                    *workspace_bytes,
                    limits,
                )?;

                let output = c2c_range(output_buffer, 0, step.line_bytes())?;
                graph.push_stage(
                    LargeStage::GatherScatter {
                        label: "mixed-axis-scatter-line",
                        src: line_output,
                        dst: output,
                        stride_elements: step.stride(),
                    },
                    graph_requirements(limits, storage_alignment, step.line_bytes())?,
                )?;
            }
            SmoothGraphStep::Smooth {
                step,
                phase1,
                phase2,
            } => {
                let step = *step;
                let phase1_input =
                    c2c_range(LogicalBufferId::Stage(0), 0, step.phase1_chunk_bytes())?;
                graph.push_stage(
                    LargeStage::GatherScatter {
                        label: "smooth-axis-gather-phase1",
                        src: c2c_range(input_buffer, 0, step.phase1_chunk_bytes())?,
                        dst: phase1_input,
                        stride_elements: step.stride(),
                    },
                    graph_requirements(limits, storage_alignment, step.phase1_chunk_bytes())?,
                )?;

                let phase1_output =
                    c2c_range(LogicalBufferId::Stage(1), 0, step.phase1_chunk_bytes())?;
                append_smooth_phase_graph(
                    &mut graph,
                    phase1,
                    "smooth-axis-phase1-stockham-stage",
                    "smooth-axis-phase1-fused-pow2-stage",
                    "smooth-axis-phase1-fused-smooth-stage",
                    "smooth-axis-phase1-workspace",
                    phase1_input,
                    phase1_output,
                    limits,
                    storage_alignment,
                    smooth_graph_temp_base(index, 0)?,
                    smooth_graph_stage_base(index, 0)?,
                )?;

                let phase2_input =
                    c2c_range(LogicalBufferId::Stage(2), 0, step.phase2_chunk_bytes())?;
                graph.push_stage(
                    LargeStage::TwiddleTranspose {
                        label: "smooth-axis-twiddle-transpose",
                        input: phase1_output,
                        output: phase2_input,
                        work_items: step.chunk_inner() * step.chunk_outer(),
                    },
                    graph_requirements(limits, storage_alignment, step.phase2_chunk_bytes())?,
                )?;

                let phase2_output =
                    c2c_range(LogicalBufferId::Stage(3), 0, step.phase2_chunk_bytes())?;
                append_smooth_phase_graph(
                    &mut graph,
                    phase2,
                    "smooth-axis-phase2-stockham-stage",
                    "smooth-axis-phase2-fused-pow2-stage",
                    "smooth-axis-phase2-fused-smooth-stage",
                    "smooth-axis-phase2-workspace",
                    phase2_input,
                    phase2_output,
                    limits,
                    storage_alignment,
                    smooth_graph_temp_base(index, 16)?,
                    smooth_graph_stage_base(index, 16)?,
                )?;

                graph.push_stage(
                    LargeStage::GatherScatter {
                        label: "smooth-axis-scatter-phase2",
                        src: phase2_output,
                        dst: c2c_range(output_buffer, 0, step.phase2_chunk_bytes())?,
                        stride_elements: step.stride(),
                    },
                    graph_requirements(limits, storage_alignment, step.phase2_chunk_bytes())?,
                )?;
            }
        }
    }
    Ok(LargeExecutionPlan::new(graph))
}

#[allow(clippy::too_many_arguments)]
fn append_smooth_phase_graph(
    graph: &mut LargeExecutionGraph,
    phase: &SmoothPhaseGraph,
    kernel_label: &'static str,
    fused_pow2_label: &'static str,
    fused_smooth_label: &'static str,
    workspace_label: &'static str,
    input: LogicalRange,
    output: LogicalRange,
    limits: LargePolicyLimits,
    storage_alignment: u32,
    temp_index_base: u32,
    stage_index_base: u32,
) -> Result<()> {
    match phase {
        SmoothPhaseGraph::Axis {
            stage_kinds,
            workspace_bytes,
        } => add_axis_plan_kernel_stages_with_kinds(
            graph,
            kernel_label,
            fused_pow2_label,
            fused_smooth_label,
            workspace_label,
            stage_kinds,
            input,
            output,
            temp_index_base,
            *workspace_bytes,
            limits,
        ),
        SmoothPhaseGraph::C2c(child_graph) => append_child_c2c_graph_with_bases(
            graph,
            child_graph,
            input,
            output,
            limits,
            storage_alignment,
            temp_index_base,
            stage_index_base,
        ),
    }
}

fn smooth_graph_temp_base(step_index: usize, phase_offset: u32) -> Result<u32> {
    let index =
        u32::try_from(step_index).map_err(|_| FftError::LengthTooLarge { len: usize::MAX })?;
    64u32
        .checked_add(
            index
                .checked_mul(32)
                .ok_or(FftError::LengthTooLarge { len: usize::MAX })?,
        )
        .and_then(|base| base.checked_add(phase_offset))
        .ok_or(FftError::LengthTooLarge { len: usize::MAX })
}

fn smooth_graph_stage_base(step_index: usize, phase_offset: u32) -> Result<u32> {
    let index =
        u32::try_from(step_index).map_err(|_| FftError::LengthTooLarge { len: usize::MAX })?;
    16u32
        .checked_add(
            index
                .checked_mul(32)
                .ok_or(FftError::LengthTooLarge { len: usize::MAX })?,
        )
        .and_then(|base| base.checked_add(phase_offset))
        .ok_or(FftError::LengthTooLarge { len: usize::MAX })
}

fn line_base_complex(shape: &[usize], axis: usize, line: u64) -> Result<u64> {
    let axis_len = shape.get(axis).copied().ok_or(FftError::InvalidAxis {
        axis,
        rank: shape.len(),
    })? as u64;
    let total = shape_product_u64(shape)?;
    let lines_per_batch = total.checked_div(axis_len).ok_or(FftError::ZeroLength)?;
    let batch = line / lines_per_batch;
    let mut rem = line - batch * lines_per_batch;
    let mut base = batch
        .checked_mul(total)
        .ok_or(FftError::LengthTooLarge { len: usize::MAX })?;
    let mut stride = 1u64;
    for (dim_index, &dim) in shape.iter().enumerate() {
        if dim_index == axis {
            stride = stride
                .checked_mul(dim as u64)
                .ok_or(FftError::LengthTooLarge { len: usize::MAX })?;
            continue;
        }
        let dim = dim as u64;
        let coord = rem % dim;
        rem /= dim;
        base = base
            .checked_add(
                coord
                    .checked_mul(stride)
                    .ok_or(FftError::LengthTooLarge { len: usize::MAX })?,
            )
            .ok_or(FftError::LengthTooLarge { len: usize::MAX })?;
        stride = stride
            .checked_mul(dim)
            .ok_or(FftError::LengthTooLarge { len: usize::MAX })?;
    }
    Ok(base)
}

fn shape_product_u64(shape: &[usize]) -> Result<u64> {
    let mut total = 1u64;
    for &dim in shape {
        total = total
            .checked_mul(dim as u64)
            .ok_or(FftError::LengthTooLarge { len: usize::MAX })?;
    }
    Ok(total)
}

fn resolve_c2c_large_routing_policy(
    device: &wgpu::Device,
    config: &FftConfig,
    axis_kinds: &[AxisKind],
    policy_limits: Option<LargePolicyLimits>,
) -> Result<LargeRoutingPolicy> {
    let required = config.required_buffer_size_bytes()?;
    let required_bindings = [required, required];
    let axis_lengths = config
        .axes()
        .iter()
        .map(|&axis| config.shape()[axis])
        .collect::<Vec<_>>();
    let line_bytes = axis_lengths
        .iter()
        .map(|&len| line_bytes_for_axis_len(len, config.precision().complex_size_bytes()))
        .collect::<Result<Vec<_>>>()?;
    let bytes_per_batch = bytes_per_batch(config)?;
    let limits = policy_limits.unwrap_or_else(|| LargePolicyLimits::from(&device.limits()));
    let four_step_supported =
        four_step_route_shape_supported(
            config.shape().len(),
            config.axes().len(),
            axis_kinds,
            &line_bytes,
            limits.max_storage_buffer_binding_size,
        ) && four_step_axis_resources_supported(config, axis_kinds, &line_bytes, limits);

    resolve_large_routing_policy_with_complex_element_bytes(
        LargeRoutingPolicyInput {
            limits,
            required_binding_bytes: &required_bindings,
            line_bytes: &line_bytes,
            axis_kinds: Some(axis_kinds),
            axis_lengths: Some(&axis_lengths),
            allow_non_mixed_bounded_slicing: true,
            allow_out_of_core: four_step_supported,
            rank: config.shape().len(),
            bytes_per_batch: Some(bytes_per_batch),
            ..LargeRoutingPolicyInput::new(limits, &[])
        },
        config.precision().complex_size_bytes(),
    )
}

fn four_step_axis_resources_supported(
    config: &FftConfig,
    axis_kinds: &[AxisKind],
    line_bytes: &[u64],
    limits: LargePolicyLimits,
) -> bool {
    config.axes().iter().zip(axis_kinds).zip(line_bytes).all(
        |((&axis, &kind), &bytes)| match kind {
            AxisKind::Mixed => true,
            AxisKind::Rader | AxisKind::Bluestein => {
                let route =
                    if kind == AxisKind::Rader && bytes > limits.max_storage_buffer_binding_size {
                        LargeBridgeRoute::Bluestein
                    } else if kind == AxisKind::Rader {
                        LargeBridgeRoute::Rader
                    } else {
                        LargeBridgeRoute::Bluestein
                    };
                let line_config = FftConfig::new(config.shape()[axis])
                    .with_direction(config.direction())
                    .with_normalization(Normalization::None);
                plan_large_bridge(&line_config, route, limits).is_ok()
            }
        },
    )
}

fn four_step_route_shape_supported(
    rank: usize,
    selected_axis_count: usize,
    axis_kinds: &[AxisKind],
    line_bytes: &[u64],
    max_bind_bytes: u64,
) -> bool {
    rank >= 2
        && selected_axis_count >= 2
        && axis_kinds.len() == selected_axis_count
        && line_bytes.len() == selected_axis_count
        && axis_kinds
            .iter()
            .zip(line_bytes)
            .all(|(&kind, &bytes)| kind != AxisKind::Mixed || bytes <= max_bind_bytes)
}

fn create_view_staging_buffer(
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

fn copy_view_to_buffer(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    view: &BufferView<'_>,
    dst: &wgpu::Buffer,
    dst_offset: u64,
    size: u64,
) -> Result<()> {
    copy_view_range_to_buffer(device, encoder, view, 0, dst, dst_offset, size)
}

fn copy_view_range_to_buffer(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
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

fn copy_buffer_to_view(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    src: &wgpu::Buffer,
    src_offset: u64,
    view: &BufferView<'_>,
    size: u64,
) -> Result<()> {
    copy_buffer_to_view_range(device, encoder, src, src_offset, view, 0, size)
}

fn copy_buffer_to_view_range(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
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

fn view_has_single_storage_range(view: &BufferView<'_>) -> bool {
    match view.ranges(0, view.size()) {
        Ok(ranges) if ranges.len() == 1 => ranges[0]
            .buffer
            .usage()
            .contains(wgpu::BufferUsages::STORAGE),
        _ => false,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AxisLineCopyDirection {
    Gather,
    Scatter,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SmoothChunkCopyDirection {
    GatherPhase1,
    ScatterPhase2,
}

fn invalid_c2c_smooth_stage(stage: &'static str, reason: &'static str) -> FftError {
    FftError::LargeGraphStageUnsupported { stage, reason }
}

fn axis_line_copy_direction(kind: C2cSmoothKernelKind) -> Result<AxisLineCopyDirection> {
    match kind {
        C2cSmoothKernelKind::GatherAxisLine => Ok(AxisLineCopyDirection::Gather),
        C2cSmoothKernelKind::ScatterAxisLine => Ok(AxisLineCopyDirection::Scatter),
        _ => Err(invalid_c2c_smooth_stage(
            "c2c-smooth-axis-line-copy",
            "axis-line copy called with non-axis kernel",
        )),
    }
}

fn axis_line_copy_ranges(
    kind: C2cSmoothKernelKind,
    base: u64,
    stride: u64,
    done: u64,
    count: u64,
) -> Result<(u64, u64, u64, u64)> {
    match axis_line_copy_direction(kind)? {
        AxisLineCopyDirection::Gather => Ok((
            checked_add_u64(base, checked_mul_u64(done, stride)?)?,
            strided_span_complex(count, stride)?,
            done,
            count,
        )),
        AxisLineCopyDirection::Scatter => Ok((
            done,
            count,
            checked_add_u64(base, checked_mul_u64(done, stride)?)?,
            strided_span_complex(count, stride)?,
        )),
    }
}

fn smooth_chunk_copy_direction(kind: C2cSmoothKernelKind) -> Result<SmoothChunkCopyDirection> {
    match kind {
        C2cSmoothKernelKind::GatherSmoothPhase1 => Ok(SmoothChunkCopyDirection::GatherPhase1),
        C2cSmoothKernelKind::ScatterSmoothPhase2 => Ok(SmoothChunkCopyDirection::ScatterPhase2),
        _ => Err(invalid_c2c_smooth_stage(
            "c2c-smooth-chunk-copy",
            "smooth chunk copy called with non-chunk kernel",
        )),
    }
}

fn dispatch_c2c_axis_line_copy_blocks(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    kind: C2cSmoothKernelKind,
    input: &BufferView<'_>,
    output: &BufferView<'_>,
    base: u64,
    stride: u64,
    len: u64,
) -> Result<()> {
    let mut done = 0u64;
    while done < len {
        let count =
            choose_axis_line_copy_count(device, kind, input, output, base, stride, len, done)?;
        let (input_first, input_span, output_first, output_span) =
            axis_line_copy_ranges(kind, base, stride, done, count)?;
        let (input_binding, input_base) =
            bind_complex_window(device, input, input_first, input_span)?;
        let (output_binding, output_base) =
            bind_complex_window(device, output, output_first, output_span)?;
        dispatch_c2c_smooth_axis_line_copy(
            device,
            encoder,
            kind,
            &input_binding,
            &output_binding,
            SmoothAxisLineCopyParams {
                total_complex: u64_to_u32(count)?,
                input_base,
                output_base,
                stride: u64_to_u32(stride)?,
            },
        )?;
        done += count;
    }
    Ok(())
}

fn choose_axis_line_copy_count(
    device: &wgpu::Device,
    kind: C2cSmoothKernelKind,
    input: &BufferView<'_>,
    output: &BufferView<'_>,
    base: u64,
    stride: u64,
    len: u64,
    done: u64,
) -> Result<u64> {
    let remaining = len - done;
    let mut count = remaining.min(u64::from(u32::MAX));
    while count > 0 {
        let (input_first, input_span, output_first, output_span) =
            axis_line_copy_ranges(kind, base, stride, done, count)?;
        if storage_window_fits(device, input, input_first, input_span)?
            && storage_window_fits(device, output, output_first, output_span)?
        {
            return Ok(count);
        }
        count /= 2;
    }
    Err(FftError::LargeChunkUnsupported {
        reason: "C2C decomposition axis-line copy block does not fit the storage binding limit",
        bytes_per_batch: strided_span_complex(1, stride)? * COMPLEX_F32_BYTES,
        max_bind_bytes: device.limits().max_storage_buffer_binding_size,
    })
}

fn dispatch_c2c_smooth_chunk_copy_blocks(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    kind: C2cSmoothKernelKind,
    input: &BufferView<'_>,
    output: &BufferView<'_>,
    request: SmoothChunkCopyRequest,
) -> Result<()> {
    match smooth_chunk_copy_direction(kind)? {
        SmoothChunkCopyDirection::GatherPhase1 => {
            let mut n2_start = 0u64;
            while n2_start < request.outer {
                let n2_count =
                    choose_smooth_gather_n2_count(device, input, output, request, n2_start)?;
                let input_first = checked_add_u64(
                    request.base,
                    checked_mul_u64(
                        checked_add_u64(
                            checked_mul_u64(n2_start, request.inner)?,
                            request.offset_start,
                        )?,
                        request.stride,
                    )?,
                )?;
                let input_span = smooth_gather_input_span(
                    n2_count,
                    request.inner,
                    request.chunk_inner,
                    request.stride,
                )?;
                let output_first = checked_mul_u64(n2_start, request.chunk_inner)?;
                let output_span = checked_mul_u64(n2_count, request.chunk_inner)?;
                let (input_binding, input_base) =
                    bind_complex_window(device, input, input_first, input_span)?;
                let (output_binding, output_base) =
                    bind_complex_window(device, output, output_first, output_span)?;
                dispatch_c2c_smooth_axis_chunk_copy(
                    device,
                    encoder,
                    kind,
                    &input_binding,
                    &output_binding,
                    SmoothAxisChunkCopyParams {
                        total_complex: u64_to_u32(output_span)?,
                        input_base,
                        output_base,
                        stride: u64_to_u32(request.stride)?,
                        inner: u64_to_u32(request.inner)?,
                        outer: u64_to_u32(request.outer)?,
                        chunk_inner: u64_to_u32(request.chunk_inner)?,
                        chunk_outer: u64_to_u32(request.chunk_outer)?,
                        offset_start: u64_to_u32(request.offset_start)?,
                        _pad0: 0,
                        _pad1: 0,
                        _pad2: 0,
                    },
                )?;
                n2_start += n2_count;
            }
        }
        SmoothChunkCopyDirection::ScatterPhase2 => {
            let mut k2_start = 0u64;
            while k2_start < request.inner {
                let k2_count =
                    choose_smooth_scatter_k2_count(device, input, output, request, k2_start)?;
                let input_first = checked_mul_u64(k2_start, request.chunk_outer)?;
                let input_span = checked_mul_u64(k2_count, request.chunk_outer)?;
                let output_first = checked_add_u64(
                    request.base,
                    checked_mul_u64(
                        checked_add_u64(
                            checked_mul_u64(k2_start, request.outer)?,
                            request.offset_start,
                        )?,
                        request.stride,
                    )?,
                )?;
                let output_span = smooth_scatter_output_span(
                    k2_count,
                    request.outer,
                    request.chunk_outer,
                    request.stride,
                )?;
                let (input_binding, input_base) =
                    bind_complex_window(device, input, input_first, input_span)?;
                let (output_binding, output_base) =
                    bind_complex_window(device, output, output_first, output_span)?;
                dispatch_c2c_smooth_axis_chunk_copy(
                    device,
                    encoder,
                    kind,
                    &input_binding,
                    &output_binding,
                    SmoothAxisChunkCopyParams {
                        total_complex: u64_to_u32(input_span)?,
                        input_base,
                        output_base,
                        stride: u64_to_u32(request.stride)?,
                        inner: u64_to_u32(request.inner)?,
                        outer: u64_to_u32(request.outer)?,
                        chunk_inner: u64_to_u32(request.chunk_inner)?,
                        chunk_outer: u64_to_u32(request.chunk_outer)?,
                        offset_start: u64_to_u32(request.offset_start)?,
                        _pad0: 0,
                        _pad1: 0,
                        _pad2: 0,
                    },
                )?;
                k2_start += k2_count;
            }
        }
    }
    Ok(())
}

fn choose_smooth_gather_n2_count(
    device: &wgpu::Device,
    input: &BufferView<'_>,
    output: &BufferView<'_>,
    request: SmoothChunkCopyRequest,
    n2_start: u64,
) -> Result<u64> {
    let remaining = request.outer - n2_start;
    let mut count = remaining.min(u64::from(u32::MAX));
    while count > 0 {
        let input_first = checked_add_u64(
            request.base,
            checked_mul_u64(
                checked_add_u64(
                    checked_mul_u64(n2_start, request.inner)?,
                    request.offset_start,
                )?,
                request.stride,
            )?,
        )?;
        let input_span =
            smooth_gather_input_span(count, request.inner, request.chunk_inner, request.stride)?;
        let output_first = checked_mul_u64(n2_start, request.chunk_inner)?;
        let output_span = checked_mul_u64(count, request.chunk_inner)?;
        if storage_window_fits(device, input, input_first, input_span)?
            && storage_window_fits(device, output, output_first, output_span)?
        {
            return Ok(count);
        }
        count /= 2;
    }
    Err(FftError::LargeChunkUnsupported {
        reason: "C2C smooth phase1 gather block does not fit the storage binding limit",
        bytes_per_batch: request.chunk_inner * COMPLEX_F32_BYTES,
        max_bind_bytes: device.limits().max_storage_buffer_binding_size,
    })
}

fn choose_smooth_scatter_k2_count(
    device: &wgpu::Device,
    input: &BufferView<'_>,
    output: &BufferView<'_>,
    request: SmoothChunkCopyRequest,
    k2_start: u64,
) -> Result<u64> {
    let remaining = request.inner - k2_start;
    let mut count = remaining.min(u64::from(u32::MAX));
    while count > 0 {
        let input_first = checked_mul_u64(k2_start, request.chunk_outer)?;
        let input_span = checked_mul_u64(count, request.chunk_outer)?;
        let output_first = checked_add_u64(
            request.base,
            checked_mul_u64(
                checked_add_u64(
                    checked_mul_u64(k2_start, request.outer)?,
                    request.offset_start,
                )?,
                request.stride,
            )?,
        )?;
        let output_span =
            smooth_scatter_output_span(count, request.outer, request.chunk_outer, request.stride)?;
        if storage_window_fits(device, input, input_first, input_span)?
            && storage_window_fits(device, output, output_first, output_span)?
        {
            return Ok(count);
        }
        count /= 2;
    }
    Err(FftError::LargeChunkUnsupported {
        reason: "C2C smooth phase2 scatter block does not fit the storage binding limit",
        bytes_per_batch: request.chunk_outer * COMPLEX_F32_BYTES,
        max_bind_bytes: device.limits().max_storage_buffer_binding_size,
    })
}

fn storage_window_fits(
    device: &wgpu::Device,
    view: &BufferView<'_>,
    first_complex: u64,
    span_complex: u64,
) -> Result<bool> {
    WindowScheduler::for_device(device).storage_window_fits(
        view,
        first_complex,
        span_complex,
        ElementFormat::ComplexF32,
    )
}

fn bind_complex_window<'a>(
    device: &wgpu::Device,
    view: &BufferView<'a>,
    first_complex: u64,
    span_complex: u64,
) -> Result<(BufferView<'a>, u32)> {
    WindowScheduler::for_device(device).bind_element_window(
        view,
        first_complex,
        span_complex,
        ElementFormat::ComplexF32,
    )
}

fn bind_u32_window<'a>(
    device: &wgpu::Device,
    buffer: &'a wgpu::Buffer,
    first_u32: u64,
    span_u32: u64,
) -> Result<(BufferView<'a>, u32)> {
    WindowScheduler::for_device(device).bind_element_window(
        &BufferView::whole(buffer),
        first_u32,
        span_u32,
        ElementFormat::U32,
    )
}

fn validate_exact_storage_view(
    device: &wgpu::Device,
    view: &BufferView<'_>,
    format: ElementFormat,
) -> Result<()> {
    WindowScheduler::for_device(device)
        .storage_binding_resource(view, format)
        .map(|_| ())
}

fn ensure_single_storage_bridge_view(view: &BufferView<'_>, usage: &'static str) -> Result<()> {
    if !view.is_single_segment() {
        return Err(FftError::SegmentedBufferViewUnsupported { usage });
    }
    if !view.buffer().usage().contains(wgpu::BufferUsages::STORAGE) {
        return Err(FftError::BufferViewMissingUsage { usage: "STORAGE" });
    }
    Ok(())
}

fn storage_buffer_with_data(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &'static str,
    bytes: &[u8],
) -> Result<wgpu::Buffer> {
    let buffer = create_view_staging_buffer(
        device,
        label,
        bytes.len() as u64,
        wgpu::BufferUsages::COPY_DST,
    )?;
    queue.write_buffer(&buffer, 0, bytes);
    Ok(buffer)
}

fn bridge_entry<'a>(
    scheduler: &WindowScheduler,
    binding: u32,
    view: &'a BufferView<'a>,
    format: ElementFormat,
) -> Result<wgpu::BindGroupEntry<'a>> {
    Ok(wgpu::BindGroupEntry {
        binding,
        resource: scheduler.storage_binding_resource(view, format)?,
    })
}

fn bridge_pipeline_key<'a>(
    key: Option<&'a ComputePipelineCacheKey>,
    route: &'static str,
    reason: &'static str,
) -> Result<&'a ComputePipelineCacheKey> {
    key.ok_or(FftError::LargeBridgeUnsupported { route, reason })
}

fn dispatch_bridge_mul_windows(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    route: &'static str,
    pipeline_key: &ComputePipelineCacheKey,
    work_buffer: &wgpu::Buffer,
    helper_buffer: &wgpu::Buffer,
    total_complex: u64,
    max_complex_window: u64,
) -> Result<()> {
    let scheduler = WindowScheduler::for_device(device);
    let mut t_offset = 0u64;
    while t_offset < total_complex {
        let t_count = (total_complex - t_offset).min(max_complex_window);
        let work = BufferView::whole(work_buffer);
        let helper = BufferView::whole(helper_buffer);
        let (work_window, work_base) = bind_complex_window(device, &work, t_offset, t_count)?;
        let (helper_window, helper_base) = bind_complex_window(device, &helper, t_offset, t_count)?;
        dispatch_bridge_kernel(
            device,
            encoder,
            route,
            pipeline_key,
            &[
                bridge_entry(&scheduler, 0, &work_window, ElementFormat::ComplexF32)?,
                bridge_entry(&scheduler, 1, &helper_window, ElementFormat::ComplexF32)?,
            ],
            2,
            BridgeKernelParams {
                line_count: 1,
                line_offset: 0,
                t_offset: u64_to_u32(t_offset)?,
                t_count: u64_to_u32(t_count)?,
                input_base: work_base,
                output_base: work_base,
                aux0_base: helper_base,
                aux1_base: 0,
                aux2_base: 0,
                aux3_base: 0,
                stride: 1,
                _pad0: 0,
            },
            u64_to_u32(t_count)?,
        )?;
        t_offset += t_count;
    }
    Ok(())
}

fn dispatch_bridge_kernel(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    route: &'static str,
    pipeline_key: &ComputePipelineCacheKey,
    storage_entries: &[wgpu::BindGroupEntry<'_>],
    uniform_binding: u32,
    params: BridgeKernelParams,
    work_items: u32,
) -> Result<()> {
    let bridge_shader_key = match &pipeline_key.shader {
        ShaderCacheKey::BridgeStage(key) => key,
        _ => {
            return Err(FftError::LargeBridgeUnsupported {
                route,
                reason: "large bridge pipeline key has a non-bridge shader stage",
            });
        }
    };
    let bind_group_layout = with_device_pipeline_cache(device, |cache| {
        cache.get_bind_group_layout(device, pipeline_key.layout)
    });
    let pipeline = with_device_pipeline_cache(device, |cache| {
        cache.get_compute_pipeline(
            device,
            pipeline_key,
            &format!("wgpu_fft.c2c.bridge.pipeline.{}", pipeline_key.stable_key()),
            &format!(
                "wgpu_fft.c2c.bridge.shader.{}",
                pipeline_key.shader.stable_key()
            ),
            || generate_bridge_wgsl_for_key(bridge_shader_key),
        )
    });
    let params_buffer = create_smooth_params_buffer(
        device,
        "wgpu_fft.c2c.bridge.params",
        bytemuck::bytes_of(&params),
    );
    let mut entries = storage_entries.to_vec();
    entries.push(wgpu::BindGroupEntry {
        binding: uniform_binding,
        resource: params_buffer.as_entire_binding(),
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("wgpu_fft.c2c.bridge.bind_group"),
        layout: &bind_group_layout,
        entries: &entries,
    });
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("wgpu_fft.c2c.bridge.pass"),
        timestamp_writes: None,
    });
    pass.set_pipeline(&pipeline);
    pass.set_bind_group(0, &bind_group, &[]);
    let (x, y, z) = split_workgroups(
        work_items.div_ceil(WORKGROUP_SIZE),
        max_workgroups_per_dimension(device),
    )?;
    pass.dispatch_workgroups(x, y, z);
    Ok(())
}

fn strided_span_complex(count: u64, stride: u64) -> Result<u64> {
    strided_span_elements(count, stride)
}

fn smooth_gather_input_span(
    n2_count: u64,
    inner: u64,
    chunk_inner: u64,
    stride: u64,
) -> Result<u64> {
    if n2_count == 0 {
        return Ok(0);
    }
    checked_add_u64(
        checked_mul_u64(
            checked_add_u64(checked_mul_u64(n2_count - 1, inner)?, chunk_inner - 1)?,
            stride,
        )?,
        1,
    )
}

fn smooth_scatter_output_span(
    k2_count: u64,
    outer: u64,
    chunk_outer: u64,
    stride: u64,
) -> Result<u64> {
    if k2_count == 0 {
        return Ok(0);
    }
    checked_add_u64(
        checked_mul_u64(
            checked_add_u64(checked_mul_u64(k2_count - 1, outer)?, chunk_outer - 1)?,
            stride,
        )?,
        1,
    )
}

fn checked_add_u64(a: u64, b: u64) -> Result<u64> {
    a.checked_add(b)
        .ok_or(FftError::LengthTooLarge { len: usize::MAX })
}

fn checked_mul_u64(a: u64, b: u64) -> Result<u64> {
    a.checked_mul(b)
        .ok_or(FftError::LengthTooLarge { len: usize::MAX })
}

fn dispatch_c2c_smooth_axis_line_copy(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    kind: C2cSmoothKernelKind,
    input: &BufferView<'_>,
    output: &BufferView<'_>,
    params: SmoothAxisLineCopyParams,
) -> Result<()> {
    dispatch_c2c_smooth_copy_pipeline(
        device,
        encoder,
        kind,
        input,
        output,
        bytemuck::bytes_of(&params),
        params.total_complex,
    )
}

fn dispatch_c2c_smooth_axis_chunk_copy(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    kind: C2cSmoothKernelKind,
    input: &BufferView<'_>,
    output: &BufferView<'_>,
    params: SmoothAxisChunkCopyParams,
) -> Result<()> {
    dispatch_c2c_smooth_copy_pipeline(
        device,
        encoder,
        kind,
        input,
        output,
        bytemuck::bytes_of(&params),
        params.total_complex,
    )
}

fn dispatch_c2c_smooth_copy_pipeline(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    kind: C2cSmoothKernelKind,
    input: &BufferView<'_>,
    output: &BufferView<'_>,
    params_bytes: &[u8],
    work_items: u32,
) -> Result<()> {
    let scheduler = WindowScheduler::for_device(device);
    let input_resource = scheduler.storage_binding_resource(input, ElementFormat::ComplexF32)?;
    let output_resource = scheduler.storage_binding_resource(output, ElementFormat::ComplexF32)?;
    let shader_key = C2cSmoothStageKey::new(kind, WORKGROUP_SIZE);
    let pipeline_key = ComputePipelineCacheKey::c2c_smooth_stage(shader_key.clone());
    let bind_group_layout = with_device_pipeline_cache(device, |cache| {
        cache.get_bind_group_layout(device, PipelineLayoutCacheKey::C2cSmoothBinaryF32)
    });
    let pipeline = with_device_pipeline_cache(device, |cache| {
        cache.get_compute_pipeline(
            device,
            &pipeline_key,
            &format!("wgpu_fft.c2c.smooth.pipeline.{}", pipeline_key.stable_key()),
            &format!("wgpu_fft.c2c.smooth.shader.{}", shader_key.stable_key()),
            || generate_c2c_smooth_wgsl_for_key(&shader_key),
        )
    });
    let params_buffer =
        create_smooth_params_buffer(device, "wgpu_fft.c2c.smooth.copy_params", params_bytes);
    let bind_group_label = format!(
        "wgpu_fft.c2c.smooth.bind_group.{}",
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

    let pass_label = format!("wgpu_fft.c2c.smooth.pass.{}", pipeline_key.stable_key());
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some(&pass_label),
        timestamp_writes: None,
    });
    pass.set_pipeline(&pipeline);
    pass.set_bind_group(0, &bind_group, &[]);
    let (x, y, z) = split_workgroups(
        work_items.div_ceil(WORKGROUP_SIZE),
        max_workgroups_per_dimension(device),
    )?;
    pass.dispatch_workgroups(x, y, z);
    Ok(())
}

fn dispatch_c2c_smooth_twiddle_transpose(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    input: &BufferView<'_>,
    output: &BufferView<'_>,
    twiddle_coarse: &wgpu::Buffer,
    twiddle_fine: &wgpu::Buffer,
    params: SmoothTwiddleParams,
) -> Result<()> {
    let scheduler = WindowScheduler::for_device(device);
    let input_resource = scheduler.storage_binding_resource(input, ElementFormat::ComplexF32)?;
    let output_resource = scheduler.storage_binding_resource(output, ElementFormat::ComplexF32)?;
    let coarse_view = BufferView::whole(twiddle_coarse);
    let fine_view = BufferView::whole(twiddle_fine);
    let coarse_resource =
        scheduler.storage_binding_resource(&coarse_view, ElementFormat::ComplexF32)?;
    let fine_resource =
        scheduler.storage_binding_resource(&fine_view, ElementFormat::ComplexF32)?;

    let shader_key = C2cSmoothStageKey::new(C2cSmoothKernelKind::TwiddleTranspose, WORKGROUP_SIZE);
    let pipeline_key = ComputePipelineCacheKey::c2c_smooth_stage(shader_key.clone());
    let bind_group_layout = with_device_pipeline_cache(device, |cache| {
        cache.get_bind_group_layout(device, pipeline_key.layout)
    });
    let pipeline = with_device_pipeline_cache(device, |cache| {
        cache.get_compute_pipeline(
            device,
            &pipeline_key,
            &format!("wgpu_fft.c2c.smooth.pipeline.{}", pipeline_key.stable_key()),
            &format!("wgpu_fft.c2c.smooth.shader.{}", shader_key.stable_key()),
            || generate_c2c_smooth_wgsl_for_key(&shader_key),
        )
    });
    let params_buffer = create_smooth_twiddle_params_buffer(device, params);
    let bind_group_label = format!(
        "wgpu_fft.c2c.smooth.bind_group.{}",
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
            wgpu::BindGroupEntry {
                binding: 3,
                resource: coarse_resource,
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: fine_resource,
            },
        ],
    });

    let pass_label = format!("wgpu_fft.c2c.smooth.pass.{}", pipeline_key.stable_key());
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some(&pass_label),
        timestamp_writes: None,
    });
    pass.set_pipeline(&pipeline);
    pass.set_bind_group(0, &bind_group, &[]);
    let (x, y, z) = split_workgroups(
        params.total_complex.div_ceil(WORKGROUP_SIZE),
        max_workgroups_per_dimension(device),
    )?;
    pass.dispatch_workgroups(x, y, z);
    Ok(())
}

fn create_smooth_twiddle_params_buffer(
    device: &wgpu::Device,
    params: SmoothTwiddleParams,
) -> wgpu::Buffer {
    create_smooth_params_buffer(
        device,
        "wgpu_fft.c2c.smooth.twiddle_params",
        bytemuck::bytes_of(&params),
    )
}

fn create_smooth_params_buffer(
    device: &wgpu::Device,
    label: &'static str,
    params_bytes: &[u8],
) -> wgpu::Buffer {
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: params_bytes.len() as u64,
        usage: wgpu::BufferUsages::UNIFORM,
        mapped_at_creation: true,
    });
    {
        let mut mapped = buffer.slice(..).get_mapped_range_mut();
        mapped.copy_from_slice(params_bytes);
    }
    buffer.unmap();
    buffer
}

fn dispatch_c2c_strided_copy(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    kind: C2cStridedKernelKind,
    input: &BufferView<'_>,
    output: &BufferView<'_>,
    layout: BufferLayout,
    logical_per_batch: u32,
    batch: u32,
    precision: AxisPrecision,
) -> Result<()> {
    let scheduler = WindowScheduler::for_device(device);
    let element_format = precision.element_format();
    let input_resource = scheduler.storage_binding_resource(input, element_format)?;
    let output_resource = scheduler.storage_binding_resource(output, element_format)?;

    let shader_key = C2cStridedStageKey::new(kind, WORKGROUP_SIZE, precision);
    let pipeline_key = ComputePipelineCacheKey::c2c_strided_stage(shader_key.clone());
    let bind_group_layout = with_device_pipeline_cache(device, |cache| {
        cache.get_bind_group_layout(device, pipeline_key.layout)
    });
    let pipeline = with_device_pipeline_cache(device, |cache| {
        cache.get_compute_pipeline(
            device,
            &pipeline_key,
            &format!(
                "wgpu_fft.c2c.strided.pipeline.{}",
                pipeline_key.stable_key()
            ),
            &format!("wgpu_fft.c2c.strided.shader.{}", shader_key.stable_key()),
            || generate_c2c_strided_wgsl_for_key(&shader_key),
        )
    });
    let total_complex = logical_per_batch
        .checked_mul(batch)
        .ok_or(FftError::LengthTooLarge { len: usize::MAX })?;
    let params = StridedCopyParams {
        total_complex,
        logical_per_batch,
        element_offset: u64_to_u32(layout.element_offset)?,
        element_stride: u64_to_u32(layout.element_stride)?,
        batch_stride: u64_to_u32(layout.resolved_batch_stride(u64::from(logical_per_batch))?)?,
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
    };
    let params_buffer = create_strided_params_buffer(device, params);
    let bind_group_label = format!(
        "wgpu_fft.c2c.strided.bind_group.{}",
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

    let pass_label = format!("wgpu_fft.c2c.strided.pass.{}", pipeline_key.stable_key());
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some(&pass_label),
        timestamp_writes: None,
    });
    pass.set_pipeline(&pipeline);
    pass.set_bind_group(0, &bind_group, &[]);
    let (x, y, z) = split_workgroups(
        total_complex.div_ceil(WORKGROUP_SIZE),
        max_workgroups_per_dimension(device),
    )?;
    pass.dispatch_workgroups(x, y, z);
    Ok(())
}

fn create_strided_params_buffer(device: &wgpu::Device, params: StridedCopyParams) -> wgpu::Buffer {
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.c2c.strided.params"),
        size: std::mem::size_of::<StridedCopyParams>() as u64,
        usage: wgpu::BufferUsages::UNIFORM,
        mapped_at_creation: true,
    });
    {
        let mut mapped = buffer.slice(..).get_mapped_range_mut();
        mapped.copy_from_slice(bytemuck::bytes_of(&params));
    }
    buffer.unmap();
    buffer
}

fn u64_to_u32(value: u64) -> Result<u32> {
    value
        .try_into()
        .map_err(|_| FftError::BufferLayoutTooLarge {
            value,
            limit: u64::from(u32::MAX),
        })
}

fn usize_from_u64(value: u64) -> Result<usize> {
    value
        .try_into()
        .map_err(|_| FftError::LengthTooLarge { len: usize::MAX })
}

pub(crate) fn generate_bridge_wgsl_for_key(key: &BridgeStageKey) -> String {
    let line_base_fn = wgsl_line_base_fn(key.rank, key.axis, &key.dims);
    let scale = format_wgsl_f32(key.scale_factor());
    let body = match key.kind {
        BridgeKernelKind::RaderSumInit => {
            r#"
  if (i >= params.line_count) { return; }
  sum_all[params.aux0_base + i] = vec2<f32>(0.0, 0.0);
  x0[params.aux1_base + i] = vec2<f32>(0.0, 0.0);
"#
        }
        BridgeKernelKind::RaderSumAccumulate => {
            r#"
  let line_local: u32 = wgFlat;
  if (line_local >= params.line_count) { return; }
  let base: u32 = line_base(params.line_offset + line_local);
  var acc: vec2<f32> = vec2<f32>(0.0, 0.0);
  var t: u32 = lid.x;
  loop {
    if (t >= params.t_count) { break; }
    let logical_t: u32 = params.t_offset + t;
    let value: vec2<f32> = input[params.input_base + t * params.stride];
    acc = acc + value;
    if (logical_t == 0u) {
      x0[params.aux1_base + line_local] = value;
    }
    t = t + WORKGROUP_SIZE;
  }
  scratch[lid.x] = acc;
  workgroupBarrier();
  var stride_reduce: u32 = WORKGROUP_SIZE / 2u;
  loop {
    if (stride_reduce == 0u) { break; }
    if (lid.x < stride_reduce) {
      scratch[lid.x] = scratch[lid.x] + scratch[lid.x + stride_reduce];
    }
    workgroupBarrier();
    stride_reduce = stride_reduce / 2u;
  }
  if (lid.x == 0u) {
    sum_all[params.aux0_base + line_local] = sum_all[params.aux0_base + line_local] + scratch[0];
  }
"#
        }
        BridgeKernelKind::RaderPack => {
            r#"
  if (i >= params.t_count) { return; }
  let t: u32 = params.t_offset + i;
  if (t >= L) {
    work[params.output_base + i] = vec2<f32>(0.0, 0.0);
    return;
  }
  let perm_index: u32 = (L - 1u) - t;
  let source_index: u32 = perm[params.aux0_base + perm_index - params.aux1_base];
  work[params.output_base + i] = input[params.input_base + source_index * params.stride];
"#
        }
        BridgeKernelKind::RaderMul | BridgeKernelKind::BluesteinMul => {
            r#"
  if (i >= params.t_count) { return; }
  work[params.output_base + i] = c_mul(work[params.input_base + i], helper[params.aux0_base + i]);
"#
        }
        BridgeKernelKind::RaderWriteY0 => {
            r#"
  if (i >= params.line_count) { return; }
  output[params.output_base] = sum_all[params.input_base + i] * vec2<f32>(SCALE, SCALE);
"#
        }
        BridgeKernelKind::RaderPost => {
            r#"
  if (i >= params.t_count) { return; }
  let t: u32 = params.t_offset + i;
  var value: vec2<f32> = conv_low[params.input_base + i];
  let wrap: u32 = t + L;
  if (wrap < M) {
    value = value + conv_high[params.aux0_base + i];
  }
  let output_index: u32 = perm[params.aux2_base + t - params.aux3_base];
  output[params.output_base + output_index * params.stride] =
    (x0[params.aux1_base] + value) * vec2<f32>(SCALE, SCALE);
"#
        }
        BridgeKernelKind::BluesteinPack => {
            r#"
  if (i >= params.t_count) { return; }
  let t: u32 = params.t_offset + i;
  if (t >= N) {
    work[params.output_base + i] = vec2<f32>(0.0, 0.0);
    return;
  }
  let chirp_index: u32 = params.aux0_base + t - params.aux1_base;
  work[params.output_base + i] = c_mul(input[params.input_base + i * params.stride], chirp[chirp_index]);
"#
        }
        BridgeKernelKind::BluesteinPost => {
            r#"
  if (i >= params.t_count) { return; }
  output[params.output_base + i * params.stride] =
    c_mul(conv[params.input_base + i], chirp[params.aux0_base + i]) * vec2<f32>(SCALE, SCALE);
"#
        }
    };
    let index_prologue = match key.kind {
        BridgeKernelKind::RaderSumAccumulate => format!(
            "{}\n  if (wgFlat >= params.line_count) {{ return; }}",
            crate::runtime::dispatch::WGSL_FLAT_WORKGROUP_INDEX
        ),
        BridgeKernelKind::RaderSumInit | BridgeKernelKind::RaderWriteY0 => {
            crate::runtime::dispatch::wgsl_flat_index_stmts(
                "i",
                "params.line_count",
                key.workgroup_size,
            )
        }
        BridgeKernelKind::RaderPack
        | BridgeKernelKind::RaderMul
        | BridgeKernelKind::RaderPost
        | BridgeKernelKind::BluesteinPack
        | BridgeKernelKind::BluesteinMul
        | BridgeKernelKind::BluesteinPost => crate::runtime::dispatch::wgsl_flat_index_stmts(
            "i",
            "params.t_count",
            key.workgroup_size,
        ),
    };
    let bindings = match key.kind {
        BridgeKernelKind::RaderSumInit => {
            r#"@group(0) @binding(0) var<storage, read_write> sum_all: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> x0: array<vec2<f32>>;
@group(0) @binding(2) var<uniform> params: Params;"#
        }
        BridgeKernelKind::RaderSumAccumulate => {
            r#"@group(0) @binding(0) var<storage, read> input: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> sum_all: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> x0: array<vec2<f32>>;
@group(0) @binding(3) var<uniform> params: Params;"#
        }
        BridgeKernelKind::RaderPack => {
            r#"@group(0) @binding(0) var<storage, read> input: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> work: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> perm: array<u32>;
@group(0) @binding(3) var<uniform> params: Params;"#
        }
        BridgeKernelKind::RaderMul | BridgeKernelKind::BluesteinMul => {
            r#"@group(0) @binding(0) var<storage, read_write> work: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read> helper: array<vec2<f32>>;
@group(0) @binding(2) var<uniform> params: Params;"#
        }
        BridgeKernelKind::RaderWriteY0 => {
            r#"@group(0) @binding(0) var<storage, read> sum_all: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec2<f32>>;
@group(0) @binding(2) var<uniform> params: Params;"#
        }
        BridgeKernelKind::RaderPost => {
            r#"@group(0) @binding(0) var<storage, read> conv_low: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read> conv_high: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> x0: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read> perm: array<u32>;
@group(0) @binding(4) var<storage, read_write> output: array<vec2<f32>>;
@group(0) @binding(5) var<uniform> params: Params;"#
        }
        BridgeKernelKind::BluesteinPack => {
            r#"@group(0) @binding(0) var<storage, read> input: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> work: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> chirp: array<vec2<f32>>;
@group(0) @binding(3) var<uniform> params: Params;"#
        }
        BridgeKernelKind::BluesteinPost => {
            r#"@group(0) @binding(0) var<storage, read> conv: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read> chirp: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> output: array<vec2<f32>>;
@group(0) @binding(3) var<uniform> params: Params;"#
        }
    };
    format!(
        r#"
struct Params {{
  line_count: u32,
  line_offset: u32,
  t_offset: u32,
  t_count: u32,
  input_base: u32,
  output_base: u32,
  aux0_base: u32,
  aux1_base: u32,
  aux2_base: u32,
  aux3_base: u32,
  stride: u32,
  pad0: u32,
}}

{bindings}

const N: u32 = {n}u;
const L: u32 = {l}u;
const M: u32 = {m}u;
const SCALE: f32 = {scale};
const WORKGROUP_SIZE: u32 = {workgroup_size}u;

{line_base_fn}

fn c_mul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {{
  return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}}

var<workgroup> scratch: array<vec2<f32>, {workgroup_size}>;

@compute @workgroup_size({workgroup_size}, 1, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {{
  {index_prologue}
  {body}
}}
"#,
        bindings = bindings,
        n = key.axis_length,
        l = key.axis_length.saturating_sub(1),
        m = key.convolution_length,
        scale = scale,
        workgroup_size = key.workgroup_size,
        line_base_fn = line_base_fn,
        index_prologue = index_prologue,
        body = body,
    )
}

pub(crate) fn generate_c2c_smooth_wgsl_for_key(key: &C2cSmoothStageKey) -> String {
    match key.kind {
        C2cSmoothKernelKind::TwiddleTranspose => {
            format!(
                r#"
struct Params {{
  total_complex: u32,
  chunk_inner: u32,
  chunk_outer: u32,
  outer: u32,
  n1_start: u32,
  k1_start: u32,
  total_len: u32,
  inverse: u32,
  scale: f32,
  inner: u32,
  lut_shift: u32,
  lut_mask: u32,
}}

@group(0) @binding(0) var<storage, read> input: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec2<f32>>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read> twiddle_coarse: array<vec2<f32>>;
@group(0) @binding(4) var<storage, read> twiddle_fine: array<vec2<f32>>;

fn c_mul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {{
  return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}}

@compute @workgroup_size({workgroup_size}, 1, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {{
  let wgFlat: u32 = (wid.z * nwg.y + wid.y) * nwg.x + wid.x;
  if (wgFlat > params.total_complex / {workgroup_size}u) {{ return; }}
  let i: u32 = wgFlat * {workgroup_size}u + lid.x;
  if (i >= params.total_complex) {{ return; }}
  let local_k1: u32 = i % params.chunk_outer;
  let local_n1: u32 = i / params.chunk_outer;
  let k1: u32 = params.k1_start + local_k1;
  let n1: u32 = params.n1_start + local_n1;
  // The decomposition keeps n1 and k1 inside complementary factors of
  // total_len, so this product is reduced, less than total_len, and u32-safe.
  let exponent: u32 = n1 * k1;
  let coarse: vec2<f32> = twiddle_coarse[exponent >> params.lut_shift];
  let fine: vec2<f32> = twiddle_fine[exponent & params.lut_mask];
  let forward_twiddle: vec2<f32> = c_mul(coarse, fine);
  let twiddle: vec2<f32> = select(
    forward_twiddle,
    vec2<f32>(forward_twiddle.x, -forward_twiddle.y),
    params.inverse != 0u,
  );
  let input_index: u32 = local_n1 + params.chunk_inner * k1;
  let value: vec2<f32> = c_mul(input[input_index], twiddle) * vec2<f32>(params.scale, params.scale);
  let out_index: u32 = local_k1 + params.chunk_outer * n1;
  output[out_index] = value;
}}
"#,
                workgroup_size = key.workgroup_size,
            )
        }
        C2cSmoothKernelKind::GatherAxisLine | C2cSmoothKernelKind::ScatterAxisLine => {
            let assignment = match key.kind {
                C2cSmoothKernelKind::GatherAxisLine => {
                    "output[params.output_base + i] = input[params.input_base + i * params.stride];"
                }
                C2cSmoothKernelKind::ScatterAxisLine => {
                    "output[params.output_base + i * params.stride] = input[params.input_base + i];"
                }
                _ => unreachable!(),
            };
            format!(
                r#"
struct Params {{
  total_complex: u32,
  input_base: u32,
  output_base: u32,
  stride: u32,
}}

@group(0) @binding(0) var<storage, read> input: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec2<f32>>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size({workgroup_size}, 1, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {{
  let wgFlat: u32 = (wid.z * nwg.y + wid.y) * nwg.x + wid.x;
  if (wgFlat > params.total_complex / {workgroup_size}u) {{ return; }}
  let i: u32 = wgFlat * {workgroup_size}u + lid.x;
  if (i >= params.total_complex) {{ return; }}
  {assignment}
}}
"#,
                workgroup_size = key.workgroup_size,
                assignment = assignment,
            )
        }
        C2cSmoothKernelKind::GatherSmoothPhase1 | C2cSmoothKernelKind::ScatterSmoothPhase2 => {
            let body = match key.kind {
                C2cSmoothKernelKind::GatherSmoothPhase1 => {
                    r#"
  let local_n1: u32 = i % params.chunk_inner;
  let n2: u32 = i / params.chunk_inner;
  let input_index: u32 = params.input_base + (n2 * params.inner + local_n1) * params.stride;
  let output_index: u32 = params.output_base + n2 * params.chunk_inner + local_n1;
  output[output_index] = input[input_index];
"#
                }
                C2cSmoothKernelKind::ScatterSmoothPhase2 => {
                    r#"
  let local_k1: u32 = i % params.chunk_outer;
  let k2: u32 = i / params.chunk_outer;
  let input_index: u32 = params.input_base + k2 * params.chunk_outer + local_k1;
  let output_index: u32 = params.output_base + (k2 * params.outer + local_k1) * params.stride;
  output[output_index] = input[input_index];
"#
                }
                _ => unreachable!(),
            };
            format!(
                r#"
struct Params {{
  total_complex: u32,
  input_base: u32,
  output_base: u32,
  stride: u32,
  inner: u32,
  outer: u32,
  chunk_inner: u32,
  chunk_outer: u32,
  offset_start: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
}}

@group(0) @binding(0) var<storage, read> input: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec2<f32>>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size({workgroup_size}, 1, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {{
  let wgFlat: u32 = (wid.z * nwg.y + wid.y) * nwg.x + wid.x;
  if (wgFlat > params.total_complex / {workgroup_size}u) {{ return; }}
  let i: u32 = wgFlat * {workgroup_size}u + lid.x;
  if (i >= params.total_complex) {{ return; }}
  {body}
}}
"#,
                workgroup_size = key.workgroup_size,
                body = body,
            )
        }
    }
}

pub(crate) fn generate_c2c_strided_wgsl_for_key(key: &C2cStridedStageKey) -> String {
    let assignment = match key.kind {
        C2cStridedKernelKind::Pack => "output[i] = input[physical_index];",
        C2cStridedKernelKind::Unpack => "output[physical_index] = input[i];",
    };
    let source = format!(
        r#"
struct Params {{
  total_complex: u32,
  logical_per_batch: u32,
  element_offset: u32,
  element_stride: u32,
  batch_stride: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
}}

@group(0) @binding(0) var<storage, read> input: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec2<f32>>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size({workgroup_size}, 1, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {{
  let wgFlat: u32 = (wid.z * nwg.y + wid.y) * nwg.x + wid.x;
  if (wgFlat > params.total_complex / {workgroup_size}u) {{ return; }}
  let i: u32 = wgFlat * {workgroup_size}u + lid.x;
  if (i >= params.total_complex) {{ return; }}
  let batch: u32 = i / params.logical_per_batch;
  let element: u32 = i - batch * params.logical_per_batch;
  let physical_index: u32 = params.element_offset + batch * params.batch_stride + element * params.element_stride;
  {assignment}
}}
"#,
        workgroup_size = key.workgroup_size,
    );
    match key.precision {
        AxisPrecision::F32 => source,
        AxisPrecision::F64 => source.replace("vec2<f32>", "vec2<f64>"),
        AxisPrecision::Df64 => {
            unreachable!("df64 strided copies are added by the strided-I/O phase")
        }
    }
}

pub fn select_route(config: &FftConfig) -> C2cRoute {
    if config.total_complex_len().ok() == Some(1) {
        return C2cRoute::DirectDft;
    }

    if let Ok(axis_kinds) = resolve_axis_kinds_for_axes(config.shape(), config.axes()) {
        if axis_kinds.iter().all(|kind| *kind == AxisKind::Mixed) {
            return C2cRoute::MixedRadix;
        }
        if axis_kinds.len() == 1 && axis_kinds[0] == AxisKind::Rader {
            return C2cRoute::Rader;
        }
        if axis_kinds.len() == 1 && axis_kinds[0] == AxisKind::Bluestein {
            return C2cRoute::Bluestein;
        }
        return C2cRoute::AxisSequence;
    }

    C2cRoute::DirectDft
}

fn axis_plan_config_for_axis(config: &FftConfig, axis: usize, final_axis: bool) -> AxisPlanConfig {
    AxisPlanConfig {
        shape: config.shape().to_vec(),
        axes: vec![axis],
        batch: config.batch(),
        direction: config.direction(),
        normalization: if final_axis {
            config.normalization()
        } else {
            Normalization::None
        },
        scale_override_bits: None,
        layout: crate::runtime::axis_plan::AxisLayout::Interleaved,
        precision: config.precision().into(),
    }
}

fn rader_config_for_axis(config: &FftConfig, axis: usize, final_axis: bool) -> RaderAxisConfig {
    RaderAxisConfig {
        shape: config.shape().to_vec(),
        axis,
        batch: config.batch(),
        direction: config.direction(),
        normalization: if final_axis {
            config.normalization()
        } else {
            Normalization::None
        },
        precision: config.precision().into(),
    }
}

fn bluestein_config_for_axis(
    config: &FftConfig,
    axis: usize,
    final_axis: bool,
) -> BluesteinAxisConfig {
    BluesteinAxisConfig {
        shape: config.shape().to_vec(),
        axis,
        batch: config.batch(),
        direction: config.direction(),
        normalization: if final_axis {
            config.normalization()
        } else {
            Normalization::None
        },
        precision: config.precision().into(),
    }
}

enum AxisStep {
    Mixed(AxisPlan),
    Rader(RaderAxis),
    Bluestein(BluesteinAxis),
}

struct AxisSequencePlan {
    steps: Vec<AxisStep>,
    axis_factors: Vec<Vec<usize>>,
    temp_buffer: Option<wgpu::Buffer>,
    workspace_size_bytes: u64,
    required_buffer_size_bytes: u64,
    axis_twiddle_lut_storage_bytes: u64,
}

impl AxisSequencePlan {
    fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        config: &FftConfig,
        axis_kinds: &[AxisKind],
    ) -> Result<Self> {
        let required_buffer_size_bytes = config.required_buffer_size_bytes()?;
        let mut steps = Vec::with_capacity(config.axes().len());
        let mut axis_factors = Vec::with_capacity(config.axes().len());
        let mut twiddle_lut_pool = AxisTwiddleLutPool::default();

        for (axis_index, (&axis, &kind)) in config.axes().iter().zip(axis_kinds).enumerate() {
            let final_axis = axis_index + 1 == config.axes().len();
            match kind {
                AxisKind::Mixed => {
                    let plan = AxisPlan::new_with_twiddle_lut_pool(
                        device,
                        queue,
                        axis_plan_config_for_axis(config, axis, final_axis),
                        &mut twiddle_lut_pool,
                    )?;
                    axis_factors.push(plan.factors().first().cloned().unwrap_or_default());
                    steps.push(AxisStep::Mixed(plan));
                }
                AxisKind::Rader => {
                    steps.push(AxisStep::Rader(RaderAxis::new(
                        device,
                        queue,
                        rader_config_for_axis(config, axis, final_axis),
                    )?));
                    axis_factors.push(Vec::new());
                }
                AxisKind::Bluestein => {
                    steps.push(AxisStep::Bluestein(BluesteinAxis::new(
                        device,
                        queue,
                        bluestein_config_for_axis(config, axis, final_axis),
                    )?));
                    axis_factors.push(Vec::new());
                }
            }
        }

        let workspace_size_bytes = if steps.len() > 1 {
            required_buffer_size_bytes
        } else {
            0
        };
        let temp_buffer = if workspace_size_bytes > 0 {
            Some(create_view_staging_buffer(
                device,
                "wgpu_fft.axis_sequence.temp",
                workspace_size_bytes,
                wgpu::BufferUsages::empty(),
            )?)
        } else {
            None
        };

        let axis_twiddle_lut_storage_bytes = twiddle_lut_pool.storage_bytes();
        Ok(Self {
            steps,
            axis_factors,
            temp_buffer,
            workspace_size_bytes,
            required_buffer_size_bytes,
            axis_twiddle_lut_storage_bytes,
        })
    }

    fn axis_factors(&self) -> &[Vec<usize>] {
        &self.axis_factors
    }

    fn workspace_size_bytes(&self) -> u64 {
        self.workspace_size_bytes
    }

    fn graph_steps(&self) -> Vec<AxisSequenceGraphStep> {
        self.steps
            .iter()
            .map(|step| match step {
                AxisStep::Mixed(plan) => AxisSequenceGraphStep::Mixed {
                    stage_kinds: plan.graph_stage_kinds(),
                    workspace_bytes: plan.workspace_size_bytes(),
                },
                AxisStep::Rader(plan) => AxisSequenceGraphStep::Rader {
                    helpers: plan.graph_helper_buffers(),
                    convolution: ConvolutionGraphFfts::rader(plan),
                },
                AxisStep::Bluestein(plan) => AxisSequenceGraphStep::Bluestein {
                    helpers: plan.graph_helper_buffers(),
                    convolution: ConvolutionGraphFfts::bluestein(plan),
                },
            })
            .collect()
    }

    fn execute_views(
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

    fn execute_views_with_workspace(
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
        let mut src_slot = SequenceBufferSlot::Input;
        let mut dst_slot = if self.steps.len() % 2 == 1 {
            SequenceBufferSlot::Output
        } else {
            SequenceBufferSlot::Temp
        };

        for (step_index, step) in self.steps.iter().enumerate() {
            let src =
                self.resolve_buffer(src_slot, input.clone(), output.clone(), workspace.clone())?;
            let dst =
                self.resolve_buffer(dst_slot, input.clone(), output.clone(), workspace.clone())?;
            step.execute_views(device, encoder, src, dst)?;

            if step_index + 1 < self.steps.len() {
                src_slot = dst_slot;
                dst_slot = next_sequence_destination(
                    "axis-sequence-buffer-flow",
                    src_slot,
                    "axis sequence buffer flow attempted to use input as destination",
                )?;
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
        slot: SequenceBufferSlot,
        input: BufferView<'a>,
        output: BufferView<'a>,
        workspace: Option<BufferView<'a>>,
    ) -> Result<BufferView<'a>> {
        match slot {
            SequenceBufferSlot::Input => Ok(input),
            SequenceBufferSlot::Output => Ok(output),
            SequenceBufferSlot::Temp => {
                if let Some(workspace) = workspace {
                    Ok(workspace)
                } else if let Some(buffer) = self.temp_buffer.as_ref() {
                    Ok(BufferView::whole(buffer))
                } else {
                    Err(sequence_temp_storage_error(
                        "axis-sequence-workspace",
                        "multi-step AxisSequencePlan requires temp storage",
                    ))
                }
            }
        }
    }
}

impl AxisStep {
    fn execute_views(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        input: BufferView<'_>,
        output: BufferView<'_>,
    ) -> Result<()> {
        match self {
            Self::Mixed(plan) => plan.execute_views(device, encoder, input, output),
            Self::Rader(plan) => plan.execute_views(device, encoder, input, output),
            Self::Bluestein(plan) => plan.execute_views(device, encoder, input, output),
        }
    }
}

pub(crate) fn generate_direct_dft_wgsl(precision: AxisPrecision) -> String {
    let source = crate::kernels::C2C_DFT_WGSL.to_owned();
    match precision {
        AxisPrecision::F32 => source,
        AxisPrecision::F64 => source
            .replace("scale: f32", "scale: f64")
            .replace("vec2<f32>", "vec2<f64>"),
        AxisPrecision::Df64 => format!(
            "{}\n{}",
            crate::kernels::DF64_WGSL,
            crate::kernels::C2C_DFT_DF64_WGSL
        ),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SequenceBufferSlot {
    Input,
    Output,
    Temp,
}

fn sequence_buffer_flow_error(stage: &'static str, reason: &'static str) -> FftError {
    FftError::LargeGraphStageUnsupported { stage, reason }
}

fn sequence_temp_storage_error(stage: &'static str, reason: &'static str) -> FftError {
    FftError::LargeGraphStageUnsupported { stage, reason }
}

fn next_sequence_destination(
    stage: &'static str,
    src_slot: SequenceBufferSlot,
    input_destination_reason: &'static str,
) -> Result<SequenceBufferSlot> {
    match src_slot {
        SequenceBufferSlot::Output => Ok(SequenceBufferSlot::Temp),
        SequenceBufferSlot::Temp => Ok(SequenceBufferSlot::Output),
        SequenceBufferSlot::Input => {
            Err(sequence_buffer_flow_error(stage, input_destination_reason))
        }
    }
}

impl DirectDftPlan {
    fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        config: &FftConfig,
        len: u32,
    ) -> Result<Self> {
        let precision: AxisPrecision = config.precision().into();
        let pipeline_key = ComputePipelineCacheKey::direct_dft_c2c(precision);
        let inverse = u32::from(config.direction() == FftDirection::Inverse);
        let params_bytes = match precision {
            AxisPrecision::F32 => bytemuck::bytes_of(&DirectParams {
                len,
                inverse,
                scale: config.scale()?,
                _pad: 0,
            })
            .to_vec(),
            AxisPrecision::F64 => bytemuck::bytes_of(&DirectParamsF64 {
                len,
                inverse,
                scale: config.scale_f64()?,
                _pad: [0; 2],
            })
            .to_vec(),
            AxisPrecision::Df64 => {
                let scale = DoubleFloat::from_f64(config.scale_f64()?);
                bytemuck::bytes_of(&DirectParamsDf64 {
                    len,
                    inverse,
                    scale_hi: scale.hi,
                    scale_lo: scale.lo,
                })
                .to_vec()
            }
        };

        let bind_group_layout = with_device_pipeline_cache(device, |cache| {
            cache.get_bind_group_layout(device, pipeline_key.layout)
        });
        let pipeline = with_device_pipeline_cache(device, |cache| {
            cache.get_compute_pipeline(
                device,
                &pipeline_key,
                "wgpu_fft.c2c_dft.pipeline",
                "wgpu_fft.c2c_dft.shader",
                || generate_direct_dft_wgsl(precision),
            )
        });

        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("wgpu_fft.c2c_dft.params"),
            size: params_bytes.len() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&params_buffer, 0, &params_bytes);
        let twiddle_buffer = create_twiddle_lut_buffer_for_len_with_precision(
            device,
            queue,
            "wgpu_fft.c2c_dft.twiddle_lut",
            len as usize,
            config.precision(),
        )?;

        let workgroups_x = len.div_ceil(WORKGROUP_SIZE);

        Ok(Self {
            precision,
            pipeline_key,
            pipeline,
            bind_group_layout,
            params_buffer,
            twiddle_buffer,
            workgroups_x,
        })
    }

    fn workspace_size_bytes(&self) -> u64 {
        0
    }

    fn execute_views(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        input: BufferView<'_>,
        output: BufferView<'_>,
    ) -> Result<()> {
        let scheduler = WindowScheduler::for_device(device);
        let element_format = self.precision.element_format();
        let input_resource = scheduler.storage_binding_resource(&input, element_format)?;
        let output_resource = scheduler.storage_binding_resource(&output, element_format)?;
        let twiddle_view = BufferView::whole(&self.twiddle_buffer);
        let twiddle_resource = scheduler.storage_binding_resource(&twiddle_view, element_format)?;
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("wgpu_fft.c2c_dft.bind_group"),
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
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: twiddle_resource,
                },
            ],
        });

        let pass_label = format!(
            "wgpu_fft.c2c_dft.pass.cache{}",
            self.pipeline_key.stable_key()
        );
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(&pass_label),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let (x, y, z) = split_workgroups(self.workgroups_x, max_workgroups_per_dimension(device))?;
        pass.dispatch_workgroups(x, y, z);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn c2c_policy_byte_math_follows_precision() {
        let f32 = FftConfig::new_nd([8, 4]).with_batch(3);
        let f64 = f32.clone().with_precision(FftPrecision::F64);
        assert_eq!(bytes_per_batch(&f32).unwrap(), 8 * 4 * 8);
        assert_eq!(bytes_per_batch(&f64).unwrap(), 8 * 4 * 16);
        assert_eq!(f32.required_buffer_size_bytes().unwrap(), 8 * 4 * 3 * 8);
        assert_eq!(f64.required_buffer_size_bytes().unwrap(), 8 * 4 * 3 * 16);
    }

    #[test]
    fn phase_c_f64_boundary_allows_normal_routes_and_gates_deferred_resources() {
        let compute_limits = wgpu::Limits::default();
        let normal = FftConfig::new(8).with_precision(FftPrecision::F64);
        let normal_kinds = resolve_axis_kinds_for_axes(normal.shape(), normal.axes()).unwrap();
        assert_eq!(
            extended_precision_route_error(
                &normal,
                &normal_kinds,
                LargePolicyLimits {
                    max_storage_buffer_binding_size: 128,
                    max_buffer_size: 4096,
                },
                select_route(&normal),
                &compute_limits,
            ),
            Ok(())
        );

        let error_for = |config: FftConfig, max_buffer_size| {
            let route = select_route(&config);
            let axis_kinds = resolve_axis_kinds_for_axes(config.shape(), config.axes()).unwrap();
            extended_precision_route_error(
                &config,
                &axis_kinds,
                LargePolicyLimits {
                    max_storage_buffer_binding_size: 128,
                    max_buffer_size,
                },
                route,
                &compute_limits,
            )
            .unwrap_err()
        };

        let prime = FftConfig::new(17).with_precision(FftPrecision::F64);
        let prime_kinds = resolve_axis_kinds_for_axes(prime.shape(), prime.axes()).unwrap();
        assert_eq!(
            extended_precision_route_error(
                &prime,
                &prime_kinds,
                LargePolicyLimits {
                    max_storage_buffer_binding_size: 4096,
                    max_buffer_size: 4096,
                },
                select_route(&prime),
                &compute_limits,
            ),
            Ok(())
        );
        for normal_prime in [
            FftConfig::new(34).with_precision(FftPrecision::F64),
            FftConfig::new_nd([2, 17]).with_precision(FftPrecision::F64),
        ] {
            let kinds =
                resolve_axis_kinds_for_axes(normal_prime.shape(), normal_prime.axes()).unwrap();
            assert_eq!(
                extended_precision_route_error(
                    &normal_prime,
                    &kinds,
                    LargePolicyLimits {
                        max_storage_buffer_binding_size: 4096,
                        max_buffer_size: 4096,
                    },
                    select_route(&normal_prime),
                    &compute_limits,
                ),
                Ok(())
            );
        }

        for (helper_limited, binding_limit) in [
            (
                FftConfig::new(17)
                    .with_batch(9)
                    .with_precision(FftPrecision::F64),
                4096,
            ),
            (
                FftConfig::new(34)
                    .with_batch(4)
                    .with_precision(FftPrecision::F64),
                4096,
            ),
            (
                FftConfig::new_nd([2, 17]).with_precision(FftPrecision::F64),
                800,
            ),
        ] {
            let route = select_route(&helper_limited);
            let kinds =
                resolve_axis_kinds_for_axes(helper_limited.shape(), helper_limited.axes()).unwrap();
            assert!(matches!(
                extended_precision_route_error(
                    &helper_limited,
                    &kinds,
                    LargePolicyLimits {
                        max_storage_buffer_binding_size: binding_limit,
                        max_buffer_size: 16 * 1024,
                    },
                    route,
                    &compute_limits,
                ),
                Err(FftError::PrecisionUnsupported {
                    route: "large-chunk",
                    reason: "large-chunk-f64-not-implemented",
                    ..
                })
            ));
        }
        assert!(matches!(
            error_for(
                FftConfig::new(8)
                    .with_batch(2)
                    .with_precision(FftPrecision::F64),
                4096,
            ),
            FftError::PrecisionUnsupported {
                route: "large-chunk",
                reason: "large-chunk-f64-not-implemented",
                ..
            }
        ));

        assert!(matches!(
            error_for(
                FftConfig::new_nd([8, 8])
                    .with_axes([0])
                    .with_precision(FftPrecision::F64),
                4096,
            ),
            FftError::PrecisionUnsupported {
                route: "large-chunk",
                reason: "large-chunk-f64-not-implemented",
                ..
            }
        ));
        assert!(matches!(
            error_for(
                FftConfig::new_nd([16, 8]).with_precision(FftPrecision::F64),
                4096,
            ),
            FftError::PrecisionUnsupported {
                route: "large-chunk",
                reason: "large-chunk-f64-not-implemented",
                ..
            }
        ));
        assert!(matches!(
            error_for(
                FftConfig::new_nd([8, 8]).with_precision(FftPrecision::F64),
                4096,
            ),
            FftError::PrecisionUnsupported {
                route: "out-of-core-four-step",
                reason: "four-step-f64-not-implemented",
                ..
            }
        ));
        assert!(matches!(
            error_for(
                FftConfig::new_nd([8, 8]).with_precision(FftPrecision::F64),
                512,
            ),
            FftError::PrecisionUnsupported {
                route: "segmented-full-volume",
                reason: "segmented-volume-f64-not-implemented",
                ..
            }
        ));

        let forced_large = FftConfig::new_nd([17, 5]).with_precision(FftPrecision::F64);
        assert!(matches!(
            error_for(forced_large.clone(), 2048),
            FftError::PrecisionUnsupported {
                route: "out-of-core-four-step",
                reason: "four-step-f64-not-implemented",
                ..
            }
        ));
        assert!(matches!(
            error_for(forced_large, 300),
            FftError::PrecisionUnsupported {
                route: "large-chunk",
                reason: "large-chunk-f64-not-implemented",
                ..
            }
        ));
    }

    #[test]
    fn phase_b_df64_boundary_allows_normal_mixed_only_and_gates_deferred_routes() {
        let compute_limits = wgpu::Limits::default();
        let limits = LargePolicyLimits {
            max_storage_buffer_binding_size: 4096,
            max_buffer_size: 4096,
        };
        for config in [
            FftConfig::new(1).with_precision(FftPrecision::Df64),
            FftConfig::new(24).with_precision(FftPrecision::Df64),
            FftConfig::new_nd([8, 15]).with_precision(FftPrecision::Df64),
        ] {
            let kinds = resolve_axis_kinds_for_axes(config.shape(), config.axes()).unwrap();
            assert_eq!(
                extended_precision_route_error(
                    &config,
                    &kinds,
                    limits,
                    select_route(&config),
                    &compute_limits,
                ),
                Ok(())
            );
        }

        let error_for = |config: FftConfig, limits: LargePolicyLimits| {
            let kinds = resolve_axis_kinds_for_axes(config.shape(), config.axes()).unwrap();
            extended_precision_route_error(
                &config,
                &kinds,
                limits,
                select_route(&config),
                &compute_limits,
            )
            .unwrap_err()
        };
        assert!(matches!(
            error_for(
                FftConfig::new(17).with_precision(FftPrecision::Df64),
                limits,
            ),
            FftError::PrecisionUnsupported {
                requested: FftPrecision::Df64,
                route: "rader",
                reason: "rader-df64-not-implemented",
            }
        ));
        assert!(matches!(
            error_for(
                FftConfig::new(34).with_precision(FftPrecision::Df64),
                limits,
            ),
            FftError::PrecisionUnsupported {
                requested: FftPrecision::Df64,
                route: "bluestein",
                reason: "bluestein-df64-not-implemented",
            }
        ));
        assert!(matches!(
            error_for(
                FftConfig::new_nd([2, 17]).with_precision(FftPrecision::Df64),
                limits,
            ),
            FftError::PrecisionUnsupported {
                requested: FftPrecision::Df64,
                route: "axis-sequence",
                reason: "axis-sequence-prime-df64-not-implemented",
            }
        ));

        let small_binding_limits = LargePolicyLimits {
            max_storage_buffer_binding_size: 128,
            max_buffer_size: 4096,
        };
        assert!(matches!(
            error_for(
                FftConfig::new(8)
                    .with_batch(2)
                    .with_precision(FftPrecision::Df64),
                small_binding_limits,
            ),
            FftError::PrecisionUnsupported {
                requested: FftPrecision::Df64,
                route: "large-chunk",
                reason: "large-chunk-df64-not-implemented",
            }
        ));
        assert!(matches!(
            error_for(
                FftConfig::new_nd([8, 8]).with_precision(FftPrecision::Df64),
                small_binding_limits,
            ),
            FftError::PrecisionUnsupported {
                requested: FftPrecision::Df64,
                route: "out-of-core-four-step",
                reason: "four-step-df64-not-implemented",
            }
        ));
    }

    #[test]
    fn four_step_route_requires_rank_and_two_selected_axes() {
        let kinds = [AxisKind::Rader, AxisKind::Mixed];
        let line_bytes = [272, 40];
        assert!(four_step_route_shape_supported(
            2,
            2,
            &kinds,
            &line_bytes,
            256
        ));
        assert!(!four_step_route_shape_supported(
            1,
            2,
            &kinds,
            &line_bytes,
            256
        ));
        assert!(!four_step_route_shape_supported(
            2,
            1,
            &kinds[..1],
            &line_bytes[..1],
            256
        ));

        // Non-mixed lines may use a bounded bridge, while the current mixed
        // window executor still requires a complete line binding.
        assert!(!four_step_route_shape_supported(
            2,
            2,
            &[AxisKind::Mixed, AxisKind::Rader],
            &line_bytes,
            256
        ));

        let config = FftConfig::new_nd([17, 4]);
        let kinds = [AxisKind::Rader, AxisKind::Mixed];
        let line_bytes = [136, 32];
        assert!(!four_step_axis_resources_supported(
            &config,
            &kinds,
            &line_bytes,
            LargePolicyLimits {
                max_storage_buffer_binding_size: 64,
                max_buffer_size: 1 << 20,
            },
        ));
        assert!(four_step_axis_resources_supported(
            &config,
            &kinds,
            &line_bytes,
            LargePolicyLimits {
                max_storage_buffer_binding_size: 128,
                max_buffer_size: 1 << 20,
            },
        ));
    }

    #[test]
    fn generated_twiddle_kernels_use_host_luts_without_shader_trig() {
        let direct = crate::kernels::C2C_DFT_WGSL;
        assert!(direct.contains("@group(0) @binding(3)"));
        assert!(direct.contains("twiddle_lut[twiddle_index]"));
        assert!(!direct.contains("sin("));
        assert!(!direct.contains("cos("));

        let smooth = generate_c2c_smooth_wgsl_for_key(&C2cSmoothStageKey::new(
            C2cSmoothKernelKind::TwiddleTranspose,
            WORKGROUP_SIZE,
        ));
        assert!(smooth.contains("@group(0) @binding(3)"));
        assert!(smooth.contains("@group(0) @binding(4)"));
        assert!(smooth.contains("exponent >> params.lut_shift"));
        assert!(!smooth.contains("sin("));
        assert!(!smooth.contains("cos("));
    }

    #[test]
    fn direct_dft_and_strided_copy_sources_follow_precision() {
        let direct_f32 = generate_direct_dft_wgsl(AxisPrecision::F32);
        let direct_f64 = generate_direct_dft_wgsl(AxisPrecision::F64);
        let direct_df64 = generate_direct_dft_wgsl(AxisPrecision::Df64);
        assert_eq!(direct_f32, crate::kernels::C2C_DFT_WGSL);
        assert!(direct_f64.contains("scale: f64"));
        assert!(direct_f64.contains("array<vec2<f64>>"));
        assert!(!direct_f64.contains("vec2<f32>"));
        assert!(direct_df64.contains("array<vec4<f32>>"));
        assert!(direct_df64.contains("df64_complex_mul"));
        assert!(direct_df64.contains("df64_complex_scale(sum, scale)"));
        assert!(!direct_df64.contains("vec2<f64>"));
        assert_eq!(std::mem::size_of::<DirectParams>(), 16);
        assert_eq!(std::mem::size_of::<DirectParamsF64>(), 24);
        assert_eq!(std::mem::size_of::<DirectParamsDf64>(), 16);

        let f64_key = C2cStridedStageKey::new(
            C2cStridedKernelKind::Pack,
            WORKGROUP_SIZE,
            AxisPrecision::F64,
        );
        let strided_f64 = generate_c2c_strided_wgsl_for_key(&f64_key);
        assert!(strided_f64.contains("array<vec2<f64>>"));
        assert!(!strided_f64.contains("vec2<f32>"));
    }

    #[test]
    fn direct_dft_lut_recurrence_matches_exact_modular_products() {
        for len in [1u32, 2, 17, 4096, u32::MAX] {
            for k in [0, len / 3, len / 2, len - 1] {
                let mut index = 0u32;
                for n in 0..len.min(64) {
                    let expected = ((u128::from(k) * u128::from(n)) % u128::from(len)) as u32;
                    assert_eq!(index, expected, "len={len} k={k} n={n}");
                    if index >= len - k {
                        index -= len - k;
                    } else {
                        index += k;
                    }
                }
            }
        }
    }

    #[test]
    fn direct_dft_lut_loop_matches_reference_in_both_directions() {
        for len in [2usize, 3, 5, 17] {
            let input = (0..len)
                .map(|index| {
                    let x = index as f32 + 1.0;
                    crate::math::Complex32::new(x * 0.25 - 0.5, x * -0.125 + 0.75)
                })
                .collect::<Vec<_>>();
            let lut = twiddle_lut_f32(len);
            for direction in [FftDirection::Forward, FftDirection::Inverse] {
                let mut actual = Vec::with_capacity(len);
                for k in 0..len {
                    let mut sum = crate::math::Complex32::default();
                    let mut twiddle_index = 0usize;
                    for &value in &input {
                        let mut twiddle = lut[twiddle_index];
                        if direction == FftDirection::Inverse {
                            twiddle.im = -twiddle.im;
                        }
                        sum.re += value.re * twiddle.re - value.im * twiddle.im;
                        sum.im += value.re * twiddle.im + value.im * twiddle.re;
                        if twiddle_index >= len - k {
                            twiddle_index -= len - k;
                        } else {
                            twiddle_index += k;
                        }
                    }
                    actual.push(sum);
                }
                let config = match direction {
                    FftDirection::Forward => FftConfig::new(len),
                    FftDirection::Inverse => FftConfig::inverse(len),
                }
                .with_normalization(Normalization::None);
                let input_f64 = input
                    .iter()
                    .map(|value| {
                        crate::math::Complex64::new(f64::from(value.re), f64::from(value.im))
                    })
                    .collect::<Vec<_>>();
                let expected = crate::math::reference_c2c_nd_f64(&input_f64, &config).unwrap();
                for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
                    let error = (f64::from(actual.re) - expected.re)
                        .abs()
                        .max((f64::from(actual.im) - expected.im).abs());
                    assert!(
                        error < 2.0e-6 * len as f64,
                        "N={len} direction={direction:?} index={index}: actual={actual:?} expected={expected:?} error={error}"
                    );
                }
            }
        }
    }

    #[test]
    fn invalid_smooth_copy_kernel_kinds_return_stage_errors() {
        assert_eq!(
            axis_line_copy_direction(C2cSmoothKernelKind::TwiddleTranspose),
            Err(FftError::LargeGraphStageUnsupported {
                stage: "c2c-smooth-axis-line-copy",
                reason: "axis-line copy called with non-axis kernel",
            })
        );
        assert_eq!(
            smooth_chunk_copy_direction(C2cSmoothKernelKind::GatherAxisLine),
            Err(FftError::LargeGraphStageUnsupported {
                stage: "c2c-smooth-chunk-copy",
                reason: "smooth chunk copy called with non-chunk kernel",
            })
        );
    }

    #[test]
    fn segmented_strided_output_missing_stage_returns_layout_error() {
        assert_eq!(
            missing_segmented_strided_output_stage_error(),
            FftError::LargeGraphStageUnsupported {
                stage: "c2c-logical-output-stage",
                reason: "segmented+strided output requires a physical staging buffer",
            }
        );
    }

    #[test]
    fn invalid_c2c_sequence_buffer_flow_returns_stage_errors() {
        assert_eq!(
            next_sequence_destination(
                "axis-plan-buffer-flow",
                SequenceBufferSlot::Input,
                "mixed-radix stage buffer flow attempted to use input as destination",
            ),
            Err(FftError::LargeGraphStageUnsupported {
                stage: "axis-plan-buffer-flow",
                reason: "mixed-radix stage buffer flow attempted to use input as destination",
            })
        );
        assert_eq!(
            next_sequence_destination(
                "axis-sequence-buffer-flow",
                SequenceBufferSlot::Input,
                "axis sequence buffer flow attempted to use input as destination",
            ),
            Err(FftError::LargeGraphStageUnsupported {
                stage: "axis-sequence-buffer-flow",
                reason: "axis sequence buffer flow attempted to use input as destination",
            })
        );
        assert_eq!(
            next_sequence_destination(
                "large-axis-sequence-buffer-flow",
                SequenceBufferSlot::Input,
                "large axis sequence buffer flow attempted to use input as destination",
            ),
            Err(FftError::LargeGraphStageUnsupported {
                stage: "large-axis-sequence-buffer-flow",
                reason: "large axis sequence buffer flow attempted to use input as destination",
            })
        );
        assert_eq!(
            sequence_temp_storage_error(
                "axis-sequence-workspace",
                "multi-step AxisSequencePlan requires temp storage",
            ),
            FftError::LargeGraphStageUnsupported {
                stage: "axis-sequence-workspace",
                reason: "multi-step AxisSequencePlan requires temp storage",
            }
        );
        assert_eq!(
            sequence_temp_storage_error(
                "smooth-decomposition-workspace",
                "multi-step smooth decomposition requires temp storage",
            ),
            FftError::LargeGraphStageUnsupported {
                stage: "smooth-decomposition-workspace",
                reason: "multi-step smooth decomposition requires temp storage",
            }
        );
    }

    #[test]
    fn selects_mixed_radix_for_factorable_lengths_above_one() {
        for len in [2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 15, 16, 21] {
            assert_eq!(select_route(&FftConfig::new(len)), C2cRoute::MixedRadix);
        }
    }

    #[test]
    fn selects_direct_dft_for_len_one_and_bluestein_for_unsupported_composites() {
        assert_eq!(select_route(&FftConfig::new(1)), C2cRoute::DirectDft);
        for len in [34, 38, 46, 58] {
            assert_eq!(select_route(&FftConfig::new(len)), C2cRoute::Bluestein);
        }
    }

    #[test]
    fn selects_rader_for_1d_prime_lengths() {
        for len in [17, 29] {
            assert_eq!(select_route(&FftConfig::new(len)), C2cRoute::Rader);
            assert_eq!(
                select_route(&FftConfig::new(len).with_batch(2)),
                C2cRoute::Rader
            );
        }
    }

    #[test]
    fn selects_mixed_radix_for_nd_factorable_axes() {
        let all_axes = FftConfig::new_nd([2, 3]);
        assert_eq!(select_route(&all_axes), C2cRoute::MixedRadix);

        let axis_subset = FftConfig::new_nd([4, 3]).with_axes([1]);
        assert_eq!(select_route(&axis_subset), C2cRoute::MixedRadix);
    }

    #[test]
    fn selects_single_axis_rader_and_bluestein_inside_nd_shapes() {
        assert_eq!(
            select_route(&FftConfig::new_nd([8, 17]).with_axes([1])),
            C2cRoute::Rader
        );
        assert_eq!(
            select_route(&FftConfig::new_nd([8, 34]).with_axes([1])),
            C2cRoute::Bluestein
        );
    }

    #[test]
    fn selects_axis_sequence_for_mixed_algorithm_axes() {
        assert_eq!(
            select_route(&FftConfig::new_nd([17, 4])),
            C2cRoute::AxisSequence
        );
        assert_eq!(
            select_route(&FftConfig::new_nd([4, 17])),
            C2cRoute::AxisSequence
        );
        assert_eq!(
            select_route(&FftConfig::new_nd([4, 34])),
            C2cRoute::AxisSequence
        );
        assert_eq!(
            select_route(&FftConfig::new_nd([17, 34])),
            C2cRoute::AxisSequence
        );
    }

    #[test]
    fn normal_direct_graph_exposes_kernel_stage() {
        let limits = LargePolicyLimits {
            max_storage_buffer_binding_size: 4096,
            max_buffer_size: 4096,
        };
        let graph = build_direct_dft_c2c_graph(8, limits).unwrap();

        assert_eq!(graph.stages().len(), 3);
        assert!(graph.stages().iter().any(|stage| {
            stage.label() == C2cRoute::DirectDft.graph_label()
                && stage.kind() == crate::runtime::large_graph::LargeStageKind::Kernel
        }));
    }

    #[test]
    fn normal_mixed_radix_graph_exposes_stockham_stages_and_workspace() {
        let limits = LargePolicyLimits {
            max_storage_buffer_binding_size: 4096,
            max_buffer_size: 4096,
        };
        let graph = build_axis_plan_c2c_graph(
            "c2c-mixed-radix-normal",
            "mixed-radix-stockham-stage",
            "mixed-radix-workspace",
            3,
            128,
            128,
            limits,
        )
        .unwrap();

        assert_eq!(graph.stages().len(), 6);
        assert_eq!(
            graph
                .stages()
                .iter()
                .filter(|stage| stage.label() == "mixed-radix-stockham-stage")
                .count(),
            3
        );
        assert!(graph.stages().iter().any(|stage| {
            stage.label() == "mixed-radix-workspace"
                && stage.kind() == crate::runtime::large_graph::LargeStageKind::HelperWindow
        }));

        let scheduler = crate::runtime::window_scheduler::WindowScheduler::new(
            crate::runtime::window_scheduler::SchedulerLimits {
                max_storage_buffer_binding_size: 64,
                max_buffer_size: 4096,
                storage_alignment: 1,
                copy_alignment: 4,
            },
        );
        let blockers = crate::runtime::stage_executor::StageExecutor::new(&scheduler)
            .graph_blockers("mixed-radix", &graph);
        assert!(blockers.iter().any(|blocker| {
            blocker.kind == crate::diagnostics::FftBlockerKind::DeviceLimit
                && blocker.route.as_deref() == Some("mixed-radix")
                && blocker.stage.as_deref() == Some("mixed-radix-workspace")
        }));
    }

    #[test]
    fn normal_axis_sequence_graph_exposes_each_step_and_helpers() {
        let limits = LargePolicyLimits {
            max_storage_buffer_binding_size: 4096,
            max_buffer_size: 4096,
        };
        let rader_helpers = [
            helper("rader-permutation-helper", 0, 64, ElementFormat::U32),
            helper("rader-bfft-helper", 1, 128, ElementFormat::ComplexF32),
            helper("rader-sum-helper", 2, 32, ElementFormat::ComplexF32),
            helper("rader-x0-helper", 3, 32, ElementFormat::ComplexF32),
            helper("rader-work-helper", 4, 128, ElementFormat::ComplexF32),
            helper("rader-fft-helper", 5, 128, ElementFormat::ComplexF32),
        ];
        let bluestein_helpers = [
            helper("bluestein-chirp-helper", 0, 128, ElementFormat::ComplexF32),
            helper("bluestein-bfft-helper", 1, 128, ElementFormat::ComplexF32),
            helper("bluestein-work-helper", 2, 128, ElementFormat::ComplexF32),
            helper("bluestein-fft-helper", 3, 128, ElementFormat::ComplexF32),
        ];
        let graph = build_axis_sequence_c2c_graph_from_steps(
            &[
                AxisSequenceGraphStep::Mixed {
                    stage_kinds: vec![
                        AxisStageKind::Stockham { radix: 8, ns: 8 },
                        AxisStageKind::Stockham { radix: 2, ns: 16 },
                    ],
                    workspace_bytes: 128,
                },
                AxisSequenceGraphStep::Rader {
                    helpers: rader_helpers.to_vec(),
                    convolution: test_convolution_ffts(),
                },
                AxisSequenceGraphStep::Bluestein {
                    helpers: bluestein_helpers.to_vec(),
                    convolution: test_convolution_ffts(),
                },
            ],
            128,
            128,
            ElementFormat::ComplexF32,
            limits,
        )
        .unwrap();

        for label in [
            "axis-sequence-workspace",
            "axis-sequence-mixed-workspace",
            "axis-sequence-mixed-stockham-stage",
            "rader-work-helper",
            "rader-post",
            "bluestein-work-helper",
            "bluestein-post",
            "logical-output",
        ] {
            assert!(graph.stages().iter().any(|stage| stage.label() == label));
        }
        assert_eq!(
            graph
                .stages()
                .iter()
                .filter(|stage| stage.label() == "axis-sequence-mixed-stockham-stage")
                .count(),
            2
        );

        let scheduler = crate::runtime::window_scheduler::WindowScheduler::new(
            crate::runtime::window_scheduler::SchedulerLimits {
                max_storage_buffer_binding_size: 64,
                max_buffer_size: 4096,
                storage_alignment: 1,
                copy_alignment: 4,
            },
        );
        let blockers = crate::runtime::stage_executor::StageExecutor::new(&scheduler)
            .graph_blockers("axis-sequence", &graph);
        assert!(blockers.iter().any(|blocker| {
            blocker.kind == crate::diagnostics::FftBlockerKind::DeviceLimit
                && blocker.route.as_deref() == Some("axis-sequence")
                && blocker.stage.as_deref() == Some("axis-sequence-workspace")
        }));
        assert!(blockers.iter().any(|blocker| {
            blocker.kind == crate::diagnostics::FftBlockerKind::DeviceLimit
                && blocker.route.as_deref() == Some("rader")
                && blocker.stage.as_deref() == Some("rader-work-helper")
        }));
    }

    #[test]
    fn large_chunk_graph_inlines_child_route_stages() {
        let limits = LargePolicyLimits {
            max_storage_buffer_binding_size: 256,
            max_buffer_size: 4096,
        };
        let plan = LargeChunkPlan::new(128, 5, limits).unwrap();
        let child_graph = build_axis_plan_c2c_graph(
            "c2c-mixed-radix-normal",
            "mixed-radix-stockham-stage",
            "mixed-radix-workspace",
            2,
            plan.staging_size_bytes(),
            plan.staging_size_bytes(),
            limits,
        )
        .unwrap();
        let graph = build_large_chunk_c2c_graph(plan, &child_graph, limits, 1)
            .unwrap()
            .graph()
            .clone();

        assert!(!graph
            .stages()
            .iter()
            .any(|stage| stage.label() == "large-chunk-child-c2c"));
        assert_eq!(
            graph
                .stages()
                .iter()
                .filter(|stage| stage.label() == "mixed-radix-workspace")
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

        let scheduler = crate::runtime::window_scheduler::WindowScheduler::new(
            crate::runtime::window_scheduler::SchedulerLimits {
                max_storage_buffer_binding_size: 64,
                max_buffer_size: 4096,
                storage_alignment: 1,
                copy_alignment: 4,
            },
        );
        let blockers = crate::runtime::stage_executor::StageExecutor::new(&scheduler)
            .graph_blockers("large-chunk", &graph);
        assert!(blockers.iter().any(|blocker| {
            blocker.kind == crate::diagnostics::FftBlockerKind::DeviceLimit
                && blocker.route.as_deref() == Some("mixed-radix")
                && blocker.stage.as_deref() == Some("mixed-radix-workspace")
        }));
    }

    #[test]
    fn large_axis_sequence_graph_inlines_child_route_stages() {
        let limits = LargePolicyLimits {
            max_storage_buffer_binding_size: 64,
            max_buffer_size: 4096,
        };
        let convolution_child = test_child_c2c_graph(256, limits);
        let rader_helpers = [
            helper("rader-permutation-helper", 0, 64, ElementFormat::U32),
            helper("rader-bfft-helper", 1, 256, ElementFormat::ComplexF32),
            helper("rader-sum-helper", 2, 8, ElementFormat::ComplexF32),
            helper("rader-x0-helper", 3, 8, ElementFormat::ComplexF32),
            helper("rader-work-helper", 4, 256, ElementFormat::ComplexF32),
            helper("rader-fft-helper", 5, 256, ElementFormat::ComplexF32),
        ];
        let rader_bridge = build_rader_bridge_c2c_graph(
            544,
            rader_helpers,
            &convolution_child,
            &convolution_child,
            limits,
        )
        .unwrap();
        let mixed_child = test_child_c2c_graph(544, limits);
        let graph =
            build_large_axis_sequence_c2c_graph(&[rader_bridge, mixed_child], 544, 1, limits)
                .unwrap()
                .graph()
                .clone();

        assert!(graph
            .stages()
            .iter()
            .any(|stage| stage.label() == "large-axis-sequence-workspace"));
        assert!(graph
            .stages()
            .iter()
            .any(|stage| stage.label() == "rader-bridge-pack"));
        assert!(graph
            .stages()
            .iter()
            .any(|stage| stage.label() == "mixed-radix-stockham-stage"));
        assert!(!graph
            .stages()
            .iter()
            .any(|stage| stage.label() == "axis-sequence-child-c2c"));

        let scheduler = crate::runtime::window_scheduler::WindowScheduler::new(
            crate::runtime::window_scheduler::SchedulerLimits {
                max_storage_buffer_binding_size: 64,
                max_buffer_size: 4096,
                storage_alignment: 1,
                copy_alignment: 4,
            },
        );
        let blockers = crate::runtime::stage_executor::StageExecutor::new(&scheduler)
            .graph_blockers("axis-sequence", &graph);
        assert!(blockers.iter().any(|blocker| {
            blocker.kind == crate::diagnostics::FftBlockerKind::DeviceLimit
                && blocker.route.as_deref() == Some("axis-sequence")
                && blocker.stage.as_deref() == Some("large-axis-sequence-workspace")
        }));
    }

    fn helper(
        label: &'static str,
        index: u32,
        size_bytes: u64,
        format: ElementFormat,
    ) -> HelperBufferRange {
        HelperBufferRange {
            label,
            index,
            size_bytes,
            format,
        }
    }

    fn test_convolution_ffts() -> ConvolutionGraphFfts {
        ConvolutionGraphFfts {
            fused: false,
            forward_stage_kinds: vec![
                AxisStageKind::Stockham { radix: 8, ns: 8 },
                AxisStageKind::Stockham { radix: 2, ns: 16 },
            ],
            forward_workspace_bytes: 128,
            inverse_stage_kinds: vec![
                AxisStageKind::Stockham { radix: 8, ns: 8 },
                AxisStageKind::Stockham { radix: 2, ns: 16 },
            ],
            inverse_workspace_bytes: 128,
        }
    }

    fn test_fused_convolution() -> ConvolutionGraphFfts {
        ConvolutionGraphFfts {
            fused: true,
            forward_stage_kinds: Vec::new(),
            forward_workspace_bytes: 0,
            inverse_stage_kinds: Vec::new(),
            inverse_workspace_bytes: 0,
        }
    }

    fn test_child_c2c_graph(required_bytes: u64, limits: LargePolicyLimits) -> LargeExecutionGraph {
        build_axis_plan_c2c_graph(
            "c2c-mixed-radix-normal",
            "mixed-radix-stockham-stage",
            "mixed-radix-workspace",
            2,
            required_bytes,
            required_bytes,
            limits,
        )
        .unwrap()
    }

    #[test]
    fn smooth_decomposition_graph_inlines_axis_and_c2c_phase_stages() {
        let limits = LargePolicyLimits {
            max_storage_buffer_binding_size: 64,
            max_buffer_size: 4096,
        };
        let mixed = SmoothMixedGraphStep {
            stride: 16,
            line_bytes: 128,
        };
        let smooth = SmoothAxisGraphStep {
            stride: 1,
            chunk_inner: 8,
            chunk_outer: 16,
            phase1_chunk_bytes: 128,
            phase2_chunk_bytes: 128,
        };
        let child = test_child_c2c_graph(128, limits);
        let graph = build_smooth_c2c_graph(
            &[
                SmoothGraphStep::Mixed {
                    step: mixed,
                    stage_kinds: vec![
                        AxisStageKind::Stockham { radix: 8, ns: 8 },
                        AxisStageKind::Stockham { radix: 2, ns: 16 },
                    ],
                    workspace_bytes: 128,
                },
                SmoothGraphStep::Smooth {
                    step: smooth,
                    phase1: SmoothPhaseGraph::Axis {
                        stage_kinds: vec![
                            AxisStageKind::Stockham { radix: 8, ns: 8 },
                            AxisStageKind::Stockham { radix: 2, ns: 16 },
                        ],
                        workspace_bytes: 128,
                    },
                    phase2: SmoothPhaseGraph::C2c(child),
                },
                SmoothGraphStep::Smooth {
                    step: smooth,
                    phase1: SmoothPhaseGraph::Axis {
                        stage_kinds: vec![AxisStageKind::FusedSmooth { axis_length: 15 }],
                        workspace_bytes: 0,
                    },
                    phase2: SmoothPhaseGraph::Axis {
                        stage_kinds: vec![AxisStageKind::FusedPow2 { axis_length: 16 }],
                        workspace_bytes: 0,
                    },
                },
            ],
            limits,
            1,
        )
        .unwrap()
        .graph()
        .clone();

        for obsolete in [
            "mixed-axis-child-c2c",
            "smooth-axis-child-phase1",
            "smooth-axis-child-phase2",
        ] {
            assert!(!graph.stages().iter().any(|stage| stage.label() == obsolete));
        }
        assert_eq!(
            graph
                .stages()
                .iter()
                .filter(|stage| stage.label() == "mixed-axis-stockham-stage")
                .count(),
            2
        );
        assert!(graph
            .stages()
            .iter()
            .any(|stage| { stage.label() == "smooth-axis-phase1-fused-smooth-stage" }));
        assert!(graph
            .stages()
            .iter()
            .any(|stage| { stage.label() == "smooth-axis-phase2-fused-pow2-stage" }));
        assert_eq!(
            graph
                .stages()
                .iter()
                .filter(|stage| stage.label() == "smooth-axis-phase1-stockham-stage")
                .count(),
            2
        );
        assert_eq!(
            graph
                .stages()
                .iter()
                .filter(|stage| stage.label() == "mixed-radix-stockham-stage")
                .count(),
            2
        );
        assert!(graph.stages().iter().any(|stage| {
            matches!(
                stage,
                LargeStage::HelperWindow {
                    label: "mixed-axis-workspace",
                    range: LogicalRange {
                        buffer: LogicalBufferId::Temp(64),
                        ..
                    },
                }
            )
        }));
        assert!(graph.stages().iter().any(|stage| {
            matches!(
                stage,
                LargeStage::HelperWindow {
                    label: "smooth-axis-phase1-workspace",
                    range: LogicalRange {
                        buffer: LogicalBufferId::Temp(96),
                        ..
                    },
                }
            )
        }));
        assert!(graph.stages().iter().any(|stage| {
            matches!(
                stage,
                LargeStage::HelperWindow {
                    label: "mixed-radix-workspace",
                    range: LogicalRange {
                        buffer: LogicalBufferId::Temp(112),
                        ..
                    },
                }
            )
        }));

        let scheduler = crate::runtime::window_scheduler::WindowScheduler::new(
            crate::runtime::window_scheduler::SchedulerLimits {
                max_storage_buffer_binding_size: 64,
                max_buffer_size: 4096,
                storage_alignment: 1,
                copy_alignment: 4,
            },
        );
        let blockers = crate::runtime::stage_executor::StageExecutor::new(&scheduler)
            .graph_blockers("smooth-decomposition", &graph);
        assert!(blockers.iter().any(|blocker| {
            blocker.kind == crate::diagnostics::FftBlockerKind::DeviceLimit
                && blocker.route.as_deref() == Some("smooth-decomposition")
                && blocker.stage.as_deref() == Some("mixed-axis-workspace")
        }));
        assert!(blockers.iter().any(|blocker| {
            blocker.kind == crate::diagnostics::FftBlockerKind::DeviceLimit
                && blocker.route.as_deref() == Some("mixed-radix")
                && blocker.stage.as_deref() == Some("mixed-radix-workspace")
        }));
    }

    #[test]
    fn large_bridge_graph_inlines_child_convolution_and_bridge_kernels() {
        let limits = LargePolicyLimits {
            max_storage_buffer_binding_size: 64,
            max_buffer_size: 4096,
        };
        let child = test_child_c2c_graph(256, limits);
        let rader_helpers = [
            helper("rader-permutation-helper", 0, 64, ElementFormat::U32),
            helper("rader-bfft-helper", 1, 256, ElementFormat::ComplexF32),
            helper("rader-sum-helper", 2, 8, ElementFormat::ComplexF32),
            helper("rader-x0-helper", 3, 8, ElementFormat::ComplexF32),
            helper("rader-work-helper", 4, 256, ElementFormat::ComplexF32),
            helper("rader-fft-helper", 5, 256, ElementFormat::ComplexF32),
        ];
        let graph =
            build_rader_bridge_c2c_graph(136, rader_helpers, &child, &child, limits).unwrap();

        assert!(!graph
            .stages()
            .iter()
            .any(|stage| stage.label() == "rader-bridge-child-convolution"));
        for label in [
            "rader-bridge-sum-init",
            "rader-bridge-sum-accumulate",
            "rader-bridge-pack",
            "rader-bridge-mul",
            "rader-bridge-write-y0",
            "rader-bridge-post",
        ] {
            assert!(graph.stages().iter().any(|stage| stage.label() == label));
        }
        assert_eq!(
            graph
                .stages()
                .iter()
                .filter(|stage| stage.label() == "mixed-radix-stockham-stage")
                .count(),
            4
        );
        assert!(graph.stages().iter().any(|stage| {
            matches!(
                stage,
                LargeStage::HelperWindow {
                    label: "rader-work-helper",
                    range: LogicalRange {
                        buffer: LogicalBufferId::Temp(4),
                        size_bytes: 64,
                        ..
                    },
                }
            )
        }));
        assert!(graph.stages().iter().any(|stage| {
            matches!(
                stage,
                LargeStage::HelperWindow {
                    label: "mixed-radix-workspace",
                    range: LogicalRange {
                        buffer: LogicalBufferId::Temp(16),
                        ..
                    },
                }
            )
        }));
        assert!(graph.stages().iter().any(|stage| {
            matches!(
                stage,
                LargeStage::HelperWindow {
                    label: "mixed-radix-workspace",
                    range: LogicalRange {
                        buffer: LogicalBufferId::Temp(32),
                        ..
                    },
                }
            )
        }));

        let bluestein_helpers = [
            helper("bluestein-chirp-helper", 0, 256, ElementFormat::ComplexF32),
            helper("bluestein-bfft-helper", 1, 256, ElementFormat::ComplexF32),
            helper("bluestein-work-helper", 2, 256, ElementFormat::ComplexF32),
            helper("bluestein-fft-helper", 3, 256, ElementFormat::ComplexF32),
        ];
        let graph =
            build_bluestein_bridge_c2c_graph(272, bluestein_helpers, &child, &child, limits)
                .unwrap();
        assert!(!graph
            .stages()
            .iter()
            .any(|stage| stage.label() == "bluestein-bridge-child-convolution"));
        for label in [
            "bluestein-bridge-pack",
            "bluestein-bridge-mul",
            "bluestein-bridge-post",
        ] {
            assert!(graph.stages().iter().any(|stage| stage.label() == label));
        }
        assert!(graph.stages().iter().any(|stage| {
            matches!(
                stage,
                LargeStage::HelperWindow {
                    label: "bluestein-chirp-helper",
                    range: LogicalRange { size_bytes: 64, .. },
                }
            )
        }));
    }

    #[test]
    fn normal_rader_graph_exposes_helper_and_kernel_stages() {
        let limits = LargePolicyLimits {
            max_storage_buffer_binding_size: 4096,
            max_buffer_size: 4096,
        };
        let helpers = [
            helper("rader-permutation-helper", 0, 64, ElementFormat::U32),
            helper("rader-bfft-helper", 1, 128, ElementFormat::ComplexF32),
            helper("rader-sum-helper", 2, 32, ElementFormat::ComplexF32),
            helper("rader-x0-helper", 3, 32, ElementFormat::ComplexF32),
            helper("rader-work-helper", 4, 128, ElementFormat::ComplexF32),
            helper("rader-fft-helper", 5, 128, ElementFormat::ComplexF32),
        ];
        let graph = build_normal_rader_c2c_graph(
            helpers.to_vec(),
            test_convolution_ffts(),
            128,
            ElementFormat::ComplexF32,
            limits,
        )
        .unwrap();

        assert_eq!(graph.stages().len(), 19);
        for label in [
            "logical-input",
            "rader-permutation-helper",
            "rader-sum",
            "rader-pack",
            "rader-forward-workspace",
            "rader-forward-stockham-stage",
            "rader-mul",
            "rader-inverse-workspace",
            "rader-inverse-stockham-stage",
            "rader-write-y0",
            "rader-post",
            "logical-output",
        ] {
            assert!(graph.stages().iter().any(|stage| stage.label() == label));
        }

        let scheduler = crate::runtime::window_scheduler::WindowScheduler::new(
            crate::runtime::window_scheduler::SchedulerLimits {
                max_storage_buffer_binding_size: 64,
                max_buffer_size: 4096,
                storage_alignment: 1,
                copy_alignment: 4,
            },
        );
        let blockers = crate::runtime::stage_executor::StageExecutor::new(&scheduler)
            .graph_blockers("rader", &graph);
        assert!(blockers.iter().any(|blocker| {
            blocker.kind == crate::diagnostics::FftBlockerKind::DeviceLimit
                && blocker.route.as_deref() == Some("rader")
                && blocker.stage.as_deref() == Some("rader-work-helper")
        }));
        assert!(blockers.iter().any(|blocker| {
            blocker.kind == crate::diagnostics::FftBlockerKind::DeviceLimit
                && blocker.route.as_deref() == Some("rader")
                && blocker.stage.as_deref() == Some("rader-pack")
        }));
    }

    #[test]
    fn fused_prime_graphs_expose_one_kernel_and_only_common_helpers() {
        let limits = LargePolicyLimits {
            max_storage_buffer_binding_size: 4096,
            max_buffer_size: 4096,
        };
        let rader = build_normal_rader_c2c_graph(
            vec![
                helper("rader-permutation-helper", 0, 64, ElementFormat::U32),
                helper("rader-bfft-helper", 1, 128, ElementFormat::ComplexF32),
            ],
            test_fused_convolution(),
            128,
            ElementFormat::ComplexF32,
            limits,
        )
        .unwrap();
        let bluestein = build_normal_bluestein_c2c_graph(
            vec![
                helper("bluestein-chirp-helper", 0, 128, ElementFormat::ComplexF32),
                helper("bluestein-bfft-helper", 1, 128, ElementFormat::ComplexF32),
            ],
            test_fused_convolution(),
            128,
            ElementFormat::ComplexF32,
            limits,
        )
        .unwrap();

        for (graph, label) in [
            (&rader, "rader-fused-workgroup-stage"),
            (&bluestein, "bluestein-fused-workgroup-stage"),
        ] {
            let kernels = graph
                .stages()
                .iter()
                .filter(|stage| matches!(stage, LargeStage::Kernel { .. }))
                .collect::<Vec<_>>();
            assert_eq!(kernels.len(), 1);
            assert_eq!(kernels[0].label(), label);
            assert_eq!(graph.stages().first().unwrap().label(), "logical-input");
            assert_eq!(graph.stages().last().unwrap().label(), "logical-output");
            assert!(!graph.stages().iter().any(|stage| {
                matches!(
                    stage.label(),
                    "rader-work-helper"
                        | "rader-fft-helper"
                        | "bluestein-work-helper"
                        | "bluestein-fft-helper"
                )
            }));
        }
    }

    #[test]
    fn normal_bluestein_graph_exposes_helper_and_kernel_stages() {
        let limits = LargePolicyLimits {
            max_storage_buffer_binding_size: 4096,
            max_buffer_size: 4096,
        };
        let helpers = [
            helper("bluestein-chirp-helper", 0, 128, ElementFormat::ComplexF32),
            helper("bluestein-bfft-helper", 1, 128, ElementFormat::ComplexF32),
            helper("bluestein-work-helper", 2, 128, ElementFormat::ComplexF32),
            helper("bluestein-fft-helper", 3, 128, ElementFormat::ComplexF32),
        ];
        let graph = build_normal_bluestein_c2c_graph(
            helpers.to_vec(),
            test_convolution_ffts(),
            128,
            ElementFormat::ComplexF32,
            limits,
        )
        .unwrap();

        assert_eq!(graph.stages().len(), 15);
        for label in [
            "logical-input",
            "bluestein-chirp-helper",
            "bluestein-pack",
            "bluestein-forward-workspace",
            "bluestein-forward-stockham-stage",
            "bluestein-mul",
            "bluestein-inverse-workspace",
            "bluestein-inverse-stockham-stage",
            "bluestein-post",
            "logical-output",
        ] {
            assert!(graph.stages().iter().any(|stage| stage.label() == label));
        }
    }
}
