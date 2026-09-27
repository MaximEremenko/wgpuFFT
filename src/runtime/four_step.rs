use std::collections::HashMap;

use bytemuck::{Pod, Zeroable};

use crate::config::{FftConfig, Normalization};
use crate::error::{FftError, Result};
use crate::runtime::axis_plan::{
    AxisLayout, AxisPlan, AxisPlanConfig, AxisPrecision, AxisStageKind, AxisTwiddleLutPool,
    LongAxisRoute,
};
use crate::runtime::axis_policy::{resolve_axis_kinds_for_config, AxisKind};
use crate::runtime::buffer_view::BufferView;
use crate::runtime::c2c::{C2cPlan, WindowedPrimeBridge};
use crate::runtime::dispatch::{max_workgroups_per_dimension, split_workgroups};
use crate::runtime::large_bridge::{plan_large_bridge, LargeBridgeRoute};
use crate::runtime::large_graph::{
    ElementFormat, LargeExecutionGraph, LargeExecutionPlan, LargeStage, LargeStageKind,
    LogicalBufferId, LogicalRange, StageRequirements,
};
use crate::runtime::large_policy::{
    plan_out_of_core_windows_for_independent_buffers, LargeFactorSplit, LargePolicyLimits,
    OutOfCoreAxisWindowPolicyInput, OutOfCorePlanInput, OutOfCoreWindow,
};
use crate::runtime::pipeline_cache::{
    with_device_pipeline_cache, ComputePipelineCacheKey, FourStepKernelKind, FourStepStageKey,
};
use crate::runtime::recorder::CommandRecorder;
use crate::runtime::stage_executor::StageExecutor;
use crate::runtime::window_scheduler::{SchedulerLimits, WindowScheduler};

const COMPLEX_F32_BYTES: u64 = 8;
const TRANSPOSE_TILE: u32 = 16;
const TRANSPOSE_WORKGROUP_SIZE: u32 = TRANSPOSE_TILE * TRANSPOSE_TILE;
const SCALE_WORKGROUP_SIZE: u32 = 64;

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct StripeTransposeParams {
    width: u32,
    height: u32,
    _pad0: u32,
    _pad1: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct ScaleParams {
    total_complex: u32,
    base_element: u32,
    scale: f32,
    _pad: u32,
}

pub(crate) struct FourStepC2cPlan {
    required_buffer_size_bytes: u64,
    limits: LargePolicyLimits,
    axes: Vec<FourStepAxisStep>,
    scale: Option<ScaleWindowPlan>,
    scratch: wgpu::Buffer,
    axis_stage_input: wgpu::Buffer,
    axis_stage_output: wgpu::Buffer,
    permutation_input: Option<wgpu::Buffer>,
    permutation_output: Option<wgpu::Buffer>,
    transpose_pipeline: Option<wgpu::ComputePipeline>,
    transpose_bind_group_layout: Option<wgpu::BindGroupLayout>,
    scale_pipeline: Option<wgpu::ComputePipeline>,
    scale_bind_group_layout: Option<wgpu::BindGroupLayout>,
    graph_plan: LargeExecutionPlan,
    staging_bytes: Vec<u64>,
    factor_splits: Vec<LargeFactorSplit>,
    twiddle_lut_storage_bytes: u64,
}

struct FourStepAxisStep {
    axis: usize,
    fft: AxisWindowPlan,
    to_front: Option<TiledTransposePlan>,
    from_front: Option<TiledTransposePlan>,
}

struct AxisWindowPlan {
    windows: Vec<AxisWindowDispatch>,
    plans: Vec<AxisWindowExecutor>,
    graph_stages: Vec<AxisWindowGraphStage>,
    max_window_bytes: u64,
    helper_bytes: Vec<u64>,
    factor_splits: Vec<LargeFactorSplit>,
    twiddle_lut_storage_bytes: u64,
}

enum AxisWindowExecutor {
    Mixed(AxisPlan),
    Prime(Box<C2cPlan>),
    Bridge(Box<WindowedPrimeBridge>),
}

#[derive(Clone, Copy)]
enum AxisWindowGraphStage {
    Mixed(AxisStageKind),
    Rader,
    Bluestein,
    BluesteinFallback,
}

struct AxisWindowDispatch {
    window: OutOfCoreWindow,
    plan_index: usize,
}

struct TiledTransposePlan {
    nx: usize,
    ny: usize,
    repetitions: usize,
    max_tile_width: usize,
    max_tile_height: usize,
    max_tile_bytes: u64,
    params_by_shape: HashMap<(usize, usize), wgpu::Buffer>,
}

#[derive(Clone, Copy)]
struct FourStepGraphAxis<'a> {
    axis: usize,
    stages: &'a [AxisWindowGraphStage],
    helper_bytes: &'a [u64],
}

struct ScaleWindowPlan {
    windows: Vec<ScaleWindowDispatch>,
    max_window_bytes: u64,
}

struct ScaleWindowDispatch {
    window: OutOfCoreWindow,
    params_buffer: wgpu::Buffer,
}

impl FourStepC2cPlan {
    pub(crate) fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        config: &FftConfig,
        limits: LargePolicyLimits,
    ) -> Result<Self> {
        let rank = config.shape().len();
        if rank < 2 {
            return Err(FftError::LargeGraphStageUnsupported {
                stage: "four-step-plan",
                reason: "four-step execution requires rank >= 2",
            });
        }

        let device_limits = device.limits();
        let effective_limits = effective_scheduler_limits(limits, &device_limits);
        let planning_limits = LargePolicyLimits {
            max_storage_buffer_binding_size: effective_limits.max_storage_buffer_binding_size,
            max_buffer_size: effective_limits.max_buffer_size,
        };
        let required_buffer_size_bytes = config.required_buffer_size_bytes()?;
        if required_buffer_size_bytes > effective_limits.max_buffer_size {
            return Err(FftError::HelperBufferTooLarge {
                helper_buffer: "four-step-transpose-scratch",
                requested_bytes: required_buffer_size_bytes,
                max_buffer_size: effective_limits.max_buffer_size,
            });
        }

        let total_complex = required_buffer_size_bytes / COMPLEX_F32_BYTES;
        let storage_alignment = effective_limits.storage_alignment;
        let mut twiddle_lut_pool = AxisTwiddleLutPool::default();
        let per_batch_complex = config.shape().iter().try_fold(1usize, |total, &dim| {
            total
                .checked_mul(dim)
                .ok_or(FftError::LengthTooLarge { len: usize::MAX })
        })?;
        let axis_kinds = resolve_axis_kinds_for_config(config)?;
        let mut axes = Vec::with_capacity(config.axes().len());
        for (&axis, &axis_kind) in config.axes().iter().zip(&axis_kinds) {
            let axis_len = config.shape()[axis];
            let lines_total = per_batch_complex
                .checked_div(axis_len)
                .and_then(|lines| lines.checked_mul(config.batch()))
                .ok_or(FftError::LengthTooLarge { len: usize::MAX })?;
            let fft = AxisWindowPlan::new(
                device,
                queue,
                axis_len,
                lines_total,
                axis,
                axis_kind,
                config,
                planning_limits,
                storage_alignment,
                &mut twiddle_lut_pool,
            )?;
            let (to_front, from_front) = if axis == 0 {
                (None, None)
            } else {
                let prefix = config.shape()[..axis]
                    .iter()
                    .try_fold(1usize, |total, &dim| {
                        total
                            .checked_mul(dim)
                            .ok_or(FftError::LengthTooLarge { len: usize::MAX })
                    })?;
                let suffix =
                    config.shape()[axis + 1..]
                        .iter()
                        .try_fold(1usize, |total, &dim| {
                            total
                                .checked_mul(dim)
                                .ok_or(FftError::LengthTooLarge { len: usize::MAX })
                        })?;
                let repetitions = suffix
                    .checked_mul(config.batch())
                    .ok_or(FftError::LengthTooLarge { len: usize::MAX })?;
                (
                    Some(TiledTransposePlan::new(
                        device,
                        prefix,
                        axis_len,
                        repetitions,
                        effective_limits,
                    )?),
                    Some(TiledTransposePlan::new(
                        device,
                        axis_len,
                        prefix,
                        repetitions,
                        effective_limits,
                    )?),
                )
            };
            axes.push(FourStepAxisStep {
                axis,
                fft,
                to_front,
                from_front,
            });
        }

        let scale_value = config.scale()?;
        let scale = if (scale_value - 1.0).abs() > f32::EPSILON {
            Some(ScaleWindowPlan::new(
                device,
                config,
                total_complex,
                scale_value,
                planning_limits,
                storage_alignment,
            )?)
        } else {
            None
        };

        let axis_stage_bytes = axes
            .iter()
            .map(|step| step.fft.max_window_bytes)
            .max()
            .ok_or(FftError::ZeroLength)?
            .max(scale.as_ref().map_or(0, |plan| plan.max_window_bytes));
        let permutation_bytes = axes
            .iter()
            .flat_map(|step| [step.to_front.as_ref(), step.from_front.as_ref()])
            .flatten()
            .map(|plan| plan.max_tile_bytes)
            .max();

        let scratch = create_four_step_buffer(
            device,
            "wgpu_fft.four_step.transpose_scratch",
            required_buffer_size_bytes,
            effective_limits.max_buffer_size,
        )?;
        let axis_stage_input = create_four_step_buffer(
            device,
            "wgpu_fft.four_step.axis_stage_input",
            axis_stage_bytes,
            effective_limits.max_buffer_size,
        )?;
        let axis_stage_output = create_four_step_buffer(
            device,
            "wgpu_fft.four_step.axis_stage_output",
            axis_stage_bytes,
            effective_limits.max_buffer_size,
        )?;
        let (permutation_input, permutation_output) = if let Some(bytes) = permutation_bytes {
            let input_label = if rank == 2 {
                "wgpu_fft.four_step.stripe_input"
            } else {
                "wgpu_fft.four_step.permutation_input"
            };
            let output_label = if rank == 2 {
                "wgpu_fft.four_step.stripe_output"
            } else {
                "wgpu_fft.four_step.permutation_output"
            };
            (
                Some(create_four_step_buffer(
                    device,
                    input_label,
                    bytes,
                    effective_limits.max_buffer_size,
                )?),
                Some(create_four_step_buffer(
                    device,
                    output_label,
                    bytes,
                    effective_limits.max_buffer_size,
                )?),
            )
        } else {
            (None, None)
        };

        let (transpose_pipeline, transpose_bind_group_layout) = if permutation_bytes.is_some() {
            let transpose_key = ComputePipelineCacheKey::four_step_stage(FourStepStageKey::new(
                FourStepKernelKind::StripeTranspose,
                TRANSPOSE_WORKGROUP_SIZE,
            ));
            let layout = with_device_pipeline_cache(device, |cache| {
                cache.get_bind_group_layout(device, transpose_key.layout)
            });
            let pipeline = with_device_pipeline_cache(device, |cache| {
                cache.get_compute_pipeline(
                    device,
                    &transpose_key,
                    "wgpu_fft.four_step.stripe_transpose.pipeline",
                    "wgpu_fft.four_step.stripe_transpose.shader",
                    || {
                        generate_four_step_wgsl_for_key(match &transpose_key.shader {
                            crate::runtime::pipeline_cache::ShaderCacheKey::FourStepStage(key) => {
                                key
                            }
                            _ => unreachable!(),
                        })
                    },
                )
            });
            (Some(pipeline), Some(layout))
        } else {
            (None, None)
        };

        let (scale_pipeline, scale_bind_group_layout) = if scale.is_some() {
            let scale_key = ComputePipelineCacheKey::four_step_stage(FourStepStageKey::new(
                FourStepKernelKind::Scale,
                SCALE_WORKGROUP_SIZE,
            ));
            let layout = with_device_pipeline_cache(device, |cache| {
                cache.get_bind_group_layout(device, scale_key.layout)
            });
            let pipeline = with_device_pipeline_cache(device, |cache| {
                cache.get_compute_pipeline(
                    device,
                    &scale_key,
                    "wgpu_fft.four_step.scale.pipeline",
                    "wgpu_fft.four_step.scale.shader",
                    || {
                        generate_four_step_wgsl_for_key(match &scale_key.shader {
                            crate::runtime::pipeline_cache::ShaderCacheKey::FourStepStage(key) => {
                                key
                            }
                            _ => unreachable!(),
                        })
                    },
                )
            });
            (Some(pipeline), Some(layout))
        } else {
            (None, None)
        };

        let graph_axes = axes
            .iter()
            .map(|step| FourStepGraphAxis {
                axis: step.axis,
                stages: &step.fft.graph_stages,
                helper_bytes: &step.fft.helper_bytes,
            })
            .collect::<Vec<_>>();
        let graph_plan = build_four_step_graph(
            required_buffer_size_bytes,
            total_complex,
            rank,
            &graph_axes,
            scale.is_some(),
            axis_stage_bytes,
            permutation_bytes,
            planning_limits,
            storage_alignment,
        )?;
        let mut staging_bytes = vec![
            required_buffer_size_bytes,
            axis_stage_bytes,
            axis_stage_bytes,
        ];
        if let Some(bytes) = permutation_bytes {
            staging_bytes.extend([bytes, bytes]);
        }
        for step in &axes {
            staging_bytes.extend(step.fft.helper_bytes.iter().copied());
        }
        let factor_splits = axes
            .iter()
            .flat_map(|step| step.fft.factor_splits.iter().cloned())
            .collect::<Vec<_>>();
        let child_twiddle_lut_storage_bytes = axes.iter().fold(0u64, |bytes, step| {
            bytes.saturating_add(step.fft.twiddle_lut_storage_bytes)
        });

        Ok(Self {
            required_buffer_size_bytes,
            limits,
            axes,
            scale,
            scratch,
            axis_stage_input,
            axis_stage_output,
            permutation_input,
            permutation_output,
            transpose_pipeline,
            transpose_bind_group_layout,
            scale_pipeline,
            scale_bind_group_layout,
            graph_plan,
            staging_bytes,
            factor_splits,
            twiddle_lut_storage_bytes: twiddle_lut_pool
                .storage_bytes()
                .saturating_add(child_twiddle_lut_storage_bytes),
        })
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
        let scratch = BufferView::whole(&self.scratch).prefix(self.required_buffer_size_bytes)?;
        let scheduler =
            WindowScheduler::new(effective_scheduler_limits(self.limits, &device.limits()));
        let executor = StageExecutor::new(&scheduler);
        executor.validate_graph(self.graph_plan.graph())?;
        validate_copy_capable_view(&input, wgpu::BufferUsages::COPY_SRC, "COPY_SRC")?;
        validate_copy_capable_view(
            &output,
            wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            "COPY_SRC|COPY_DST",
        )?;

        for (step_index, step) in self.axes.iter().enumerate() {
            let source = if step_index == 0 {
                &input
            } else if step_index % 2 == 1 {
                &scratch
            } else {
                &output
            };
            let result = if step_index % 2 == 0 {
                &scratch
            } else {
                &output
            };
            let intermediate = if step_index % 2 == 0 {
                &output
            } else {
                &scratch
            };

            if step.axis == 0 {
                step.fft.execute(
                    device,
                    encoder,
                    &scheduler,
                    &executor,
                    source,
                    result,
                    &self.axis_stage_input,
                    &self.axis_stage_output,
                )?;
            } else {
                let tile_input = self
                    .permutation_input
                    .as_ref()
                    .expect("permutation staging exists for a non-front axis");
                let tile_output = self
                    .permutation_output
                    .as_ref()
                    .expect("permutation staging exists for a non-front axis");
                let transpose_layout = self
                    .transpose_bind_group_layout
                    .as_ref()
                    .expect("transpose layout exists for a non-front axis");
                let transpose_pipeline = self
                    .transpose_pipeline
                    .as_ref()
                    .expect("transpose pipeline exists for a non-front axis");
                step.to_front
                    .as_ref()
                    .expect("to-front plan exists for a non-front axis")
                    .execute(
                        device,
                        encoder,
                        &scheduler,
                        &executor,
                        source,
                        result,
                        tile_input,
                        tile_output,
                        transpose_layout,
                        transpose_pipeline,
                    )?;
                step.fft.execute(
                    device,
                    encoder,
                    &scheduler,
                    &executor,
                    result,
                    intermediate,
                    &self.axis_stage_input,
                    &self.axis_stage_output,
                )?;
                step.from_front
                    .as_ref()
                    .expect("from-front plan exists for a non-front axis")
                    .execute(
                        device,
                        encoder,
                        &scheduler,
                        &executor,
                        intermediate,
                        result,
                        tile_input,
                        tile_output,
                        transpose_layout,
                        transpose_pipeline,
                    )?;
            }
        }

        let final_data = if self.axes.len() % 2 == 0 {
            &output
        } else {
            &scratch
        };
        if let Some(scale) = self.scale.as_ref() {
            scale.execute(
                device,
                encoder,
                &scheduler,
                &executor,
                final_data,
                &self.axis_stage_input,
                self.scale_bind_group_layout
                    .as_ref()
                    .expect("scale layout exists with scale plan"),
                self.scale_pipeline
                    .as_ref()
                    .expect("scale pipeline exists with scale plan"),
            )?;
        }
        if self.axes.len() % 2 == 1 {
            executor.copy_buffer_to_view_range(
                encoder,
                &self.scratch,
                0,
                &output,
                0,
                self.required_buffer_size_bytes,
            )?;
        }

        Ok(())
    }

    pub(crate) fn graph_plan(&self) -> &LargeExecutionPlan {
        &self.graph_plan
    }

    pub(crate) fn staging_bytes(&self) -> Vec<u64> {
        self.staging_bytes.clone()
    }

    pub(crate) fn factor_splits(&self) -> Vec<LargeFactorSplit> {
        self.factor_splits.clone()
    }

    pub(crate) fn twiddle_lut_storage_bytes(&self) -> u64 {
        self.twiddle_lut_storage_bytes
    }
}

#[allow(clippy::too_many_arguments)]
fn configured_axis_window_policy_input(
    config: &FftConfig,
    axis_len: usize,
    line_bytes: u64,
    lines_total: usize,
    max_bind_bytes: u64,
    axis_kind: AxisKind,
    storage_align: u64,
) -> OutOfCoreAxisWindowPolicyInput {
    OutOfCoreAxisWindowPolicyInput {
        axis_len,
        line_bytes,
        lines_total,
        max_bind_bytes,
        axis_kind,
        storage_align,
        swap_to_2_stage_4_step: config.tuning().swap_to_2_stage_4_step(),
        swap_to_3_stage_4_step: config.tuning().swap_to_3_stage_4_step(),
        grouped_batch: config.tuning().grouped_batch(),
        // The four-step executor does not yet own a multi-window burst ring.
        out_of_core_burst_windows: 1,
    }
}

impl AxisWindowPlan {
    #[allow(clippy::too_many_arguments)]
    fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        axis_len: usize,
        lines_total: usize,
        axis: usize,
        axis_kind: AxisKind,
        config: &FftConfig,
        limits: LargePolicyLimits,
        storage_alignment: u64,
        twiddle_lut_pool: &mut AxisTwiddleLutPool,
    ) -> Result<Self> {
        let line_bytes = (axis_len as u64)
            .checked_mul(COMPLEX_F32_BYTES)
            .ok_or(FftError::LengthTooLarge { len: usize::MAX })?;
        if line_bytes > limits.max_buffer_size {
            return Err(FftError::HelperBufferTooLarge {
                helper_buffer: "four-step-axis-line-stage",
                requested_bytes: line_bytes,
                max_buffer_size: limits.max_buffer_size,
            });
        }

        let child_tuning = match axis_kind {
            AxisKind::Mixed => config.tuning().for_internal_child(),
            AxisKind::Rader => config
                .tuning()
                .for_internal_child()
                .with_force_rader_axes([0]),
            AxisKind::Bluestein => config
                .tuning()
                .for_internal_child()
                .with_force_bluestein_axes([0]),
        };
        let line_config = FftConfig::new(axis_len)
            .with_direction(config.direction())
            .with_normalization(Normalization::None)
            .with_tuning(child_tuning);
        let (bridge_route, bridge_plan, use_bridge, window_max_bind) = match axis_kind {
            AxisKind::Mixed => (None, None, false, limits.max_storage_buffer_binding_size),
            AxisKind::Rader | AxisKind::Bluestein => {
                let route = match axis_kind {
                    AxisKind::Rader if line_bytes > limits.max_storage_buffer_binding_size => {
                        LargeBridgeRoute::Bluestein
                    }
                    AxisKind::Rader => LargeBridgeRoute::Rader,
                    AxisKind::Bluestein => LargeBridgeRoute::Bluestein,
                    AxisKind::Mixed => unreachable!(),
                };
                let plan = plan_large_bridge(&line_config, route, limits)?;
                let convolution_line_bytes = plan
                    .convolution_len()
                    .checked_mul(COMPLEX_F32_BYTES)
                    .ok_or(FftError::LengthTooLarge { len: usize::MAX })?;
                let bridge = line_bytes > limits.max_storage_buffer_binding_size
                    || convolution_line_bytes > limits.max_storage_buffer_binding_size;
                let max_bind = if bridge {
                    if line_bytes <= limits.max_storage_buffer_binding_size {
                        line_bytes
                    } else {
                        limits.max_storage_buffer_binding_size
                    }
                } else {
                    let max_lines = (limits.max_storage_buffer_binding_size / line_bytes)
                        .min(limits.max_storage_buffer_binding_size / convolution_line_bytes)
                        .max(1);
                    line_bytes
                        .checked_mul(max_lines)
                        .ok_or(FftError::LengthTooLarge { len: usize::MAX })?
                };
                (Some(route), Some(plan), bridge, max_bind)
            }
        };
        let window_plan = plan_out_of_core_windows_for_independent_buffers(OutOfCorePlanInput {
            axis_window: configured_axis_window_policy_input(
                config,
                axis_len,
                line_bytes,
                lines_total,
                window_max_bind,
                axis_kind,
                storage_alignment,
            ),
            max_buffer_size: limits.max_buffer_size,
        })?;

        let mut plans = Vec::new();
        let mut plan_indices = HashMap::<usize, usize>::new();
        let mut windows = Vec::with_capacity(window_plan.upload_windows.len());
        let mut graph_stages = Vec::new();
        let mut helper_bytes = Vec::new();
        let mut twiddle_lut_storage_bytes = 0u64;
        let mut executor_factor_splits = None;
        for window in window_plan.upload_windows {
            let plan_index = if let Some(&index) = plan_indices.get(&window.line_count) {
                index
            } else {
                let index = plans.len();
                let plan = match axis_kind {
                    AxisKind::Mixed => {
                        AxisWindowExecutor::Mixed(AxisPlan::new_with_twiddle_lut_pool(
                            device,
                            queue,
                            AxisPlanConfig {
                                shape: vec![axis_len],
                                axes: vec![0],
                                batch: window.line_count,
                                direction: config.direction(),
                                normalization: Normalization::None,
                                scale_override_bits: Some(1.0f32.to_bits()),
                                layout: AxisLayout::Interleaved,
                                precision: AxisPrecision::F32,
                                workgroup_size: config.tuning().workgroup_size(),
                                fused_workgroup_size: config.tuning().fused_workgroup_size(),
                                long_axes: LongAxisRoute::windowed(
                                    config.tuning().fuse_long_axes(),
                                ),
                            },
                            twiddle_lut_pool,
                        )?)
                    }
                    AxisKind::Rader | AxisKind::Bluestein if use_bridge => {
                        debug_assert_eq!(window.line_count, 1);
                        AxisWindowExecutor::Bridge(Box::new(WindowedPrimeBridge::new(
                            device,
                            queue,
                            &line_config,
                            bridge_route.expect("a prime bridge has a route"),
                            limits,
                        )?))
                    }
                    AxisKind::Rader | AxisKind::Bluestein => {
                        let child_config = line_config.clone().with_batch(window.line_count);
                        AxisWindowExecutor::Prime(Box::new(C2cPlan::new_with_large_policy_limits(
                            device,
                            queue,
                            child_config,
                            Some(limits),
                        )?))
                    }
                };
                let metadata = axis_window_executor_metadata(&plan, bridge_route, axis_kind)?;
                if let AxisWindowExecutor::Bridge(plan) = &plan {
                    executor_factor_splits = Some(plan.factor_splits().to_vec());
                }
                if graph_stages.is_empty() {
                    graph_stages = metadata.graph_stages;
                }
                helper_bytes.extend(metadata.helper_bytes);
                twiddle_lut_storage_bytes =
                    twiddle_lut_storage_bytes.saturating_add(metadata.twiddle_lut_storage_bytes);
                plans.push(plan);
                plan_indices.insert(window.line_count, index);
                index
            };
            windows.push(AxisWindowDispatch { window, plan_index });
        }
        let max_window_bytes = windows
            .iter()
            .map(|dispatch| dispatch.window.byte_size)
            .max()
            .ok_or(FftError::ZeroLength)?;

        let factor_splits = match axis_kind {
            AxisKind::Mixed => vec![LargeFactorSplit {
                axis: Some(axis),
                len: axis_len as u64,
                factors: crate::runtime::factor_supported_length(axis_len)?
                    .into_iter()
                    .map(|factor| factor as u64)
                    .collect(),
            }],
            AxisKind::Rader | AxisKind::Bluestein => executor_factor_splits
                .unwrap_or_else(|| {
                    bridge_plan
                        .expect("a non-mixed axis has bridge metadata")
                        .factor_splits()
                })
                .into_iter()
                .map(|mut split| {
                    if split.axis.is_some() {
                        split.axis = Some(axis);
                    }
                    split
                })
                .collect(),
        };

        Ok(Self {
            windows,
            plans,
            graph_stages,
            max_window_bytes,
            helper_bytes,
            factor_splits,
            twiddle_lut_storage_bytes,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn execute(
        &self,
        device: &wgpu::Device,
        encoder: &mut CommandRecorder<'_>,
        scheduler: &WindowScheduler,
        executor: &StageExecutor<'_>,
        input: &BufferView<'_>,
        output: &BufferView<'_>,
        stage_input: &wgpu::Buffer,
        stage_output: &wgpu::Buffer,
    ) -> Result<()> {
        for dispatch in &self.windows {
            let first_element = dispatch.window.byte_offset / COMPLEX_F32_BYTES;
            let span_elements = dispatch.window.byte_size / COMPLEX_F32_BYTES;
            let direct_input = scheduler.bind_element_window(
                input,
                first_element,
                span_elements,
                ElementFormat::ComplexF32,
            );
            let direct_output = scheduler.bind_element_window(
                output,
                first_element,
                span_elements,
                ElementFormat::ComplexF32,
            );
            let plan = &self.plans[dispatch.plan_index];

            match (direct_input, direct_output) {
                (Ok((input_window, 0)), Ok((output_window, 0))) => {
                    plan.execute_views(device, encoder, input_window, output_window)?;
                }
                _ => {
                    executor.copy_view_range_to_buffer(
                        encoder,
                        input,
                        dispatch.window.byte_offset,
                        stage_input,
                        0,
                        dispatch.window.byte_size,
                    )?;
                    plan.execute_views(
                        device,
                        encoder,
                        BufferView::whole(stage_input).prefix(dispatch.window.byte_size)?,
                        BufferView::whole(stage_output).prefix(dispatch.window.byte_size)?,
                    )?;
                    executor.copy_buffer_to_view_range(
                        encoder,
                        stage_output,
                        0,
                        output,
                        dispatch.window.byte_offset,
                        dispatch.window.byte_size,
                    )?;
                }
            }
        }
        Ok(())
    }
}

struct AxisWindowExecutorMetadata {
    graph_stages: Vec<AxisWindowGraphStage>,
    helper_bytes: Vec<u64>,
    twiddle_lut_storage_bytes: u64,
}

impl AxisWindowExecutor {
    fn execute_views(
        &self,
        device: &wgpu::Device,
        encoder: &mut CommandRecorder<'_>,
        input: BufferView<'_>,
        output: BufferView<'_>,
    ) -> Result<()> {
        match self {
            Self::Mixed(plan) => plan.execute_views(device, encoder, input, output),
            Self::Prime(plan) => plan.execute_views_recorded(device, encoder, input, output),
            Self::Bridge(plan) => plan.execute_views(device, encoder, input, output),
        }
    }
}

fn axis_window_executor_metadata(
    plan: &AxisWindowExecutor,
    bridge_route: Option<LargeBridgeRoute>,
    axis_kind: AxisKind,
) -> Result<AxisWindowExecutorMetadata> {
    match plan {
        AxisWindowExecutor::Mixed(plan) => Ok(AxisWindowExecutorMetadata {
            graph_stages: plan
                .graph_stage_kinds()
                .into_iter()
                .map(AxisWindowGraphStage::Mixed)
                .collect(),
            helper_bytes: [plan.workspace_size_bytes()]
                .into_iter()
                .filter(|&bytes| bytes > 0)
                .collect(),
            twiddle_lut_storage_bytes: 0,
        }),
        AxisWindowExecutor::Prime(plan) => {
            let graph = plan.execution_graph()?;
            Ok(AxisWindowExecutorMetadata {
                graph_stages: prime_graph_stages(
                    operational_stage_count(&graph),
                    bridge_route,
                    axis_kind,
                )?,
                helper_bytes: owned_graph_helper_bytes(&graph),
                twiddle_lut_storage_bytes: plan.twiddle_lut_storage_bytes(),
            })
        }
        AxisWindowExecutor::Bridge(plan) => Ok(AxisWindowExecutorMetadata {
            graph_stages: prime_graph_stages(
                operational_stage_count(plan.execution_graph()),
                bridge_route,
                axis_kind,
            )?,
            helper_bytes: owned_graph_helper_bytes(plan.execution_graph()),
            twiddle_lut_storage_bytes: plan.twiddle_lut_storage_bytes(),
        }),
    }
}

fn operational_stage_count(graph: &LargeExecutionGraph) -> usize {
    graph
        .stages()
        .iter()
        .filter(|stage| {
            matches!(
                stage.kind(),
                LargeStageKind::Copy
                    | LargeStageKind::GatherScatter
                    | LargeStageKind::Kernel
                    | LargeStageKind::WindowedKernel
                    | LargeStageKind::TwiddleTranspose
                    | LargeStageKind::StripeTranspose
                    | LargeStageKind::Permutation
                    | LargeStageKind::Scale
            )
        })
        .count()
}

fn owned_graph_helper_bytes(graph: &LargeExecutionGraph) -> Vec<u64> {
    graph
        .stages()
        .iter()
        .filter(|stage| {
            matches!(
                stage.kind(),
                LargeStageKind::HelperWindow | LargeStageKind::WindowedHelper
            )
        })
        .flat_map(LargeStage::ranges)
        .map(|range| range.size_bytes)
        .collect()
}

fn prime_graph_stages(
    count: usize,
    bridge_route: Option<LargeBridgeRoute>,
    axis_kind: AxisKind,
) -> Result<Vec<AxisWindowGraphStage>> {
    if count == 0 {
        return Err(FftError::LargeGraphStageUnsupported {
            stage: "four-step-prime-window",
            reason: "prime window child graph contains no executable stages",
        });
    }
    let stage = match bridge_route {
        Some(LargeBridgeRoute::Rader) => AxisWindowGraphStage::Rader,
        Some(LargeBridgeRoute::Bluestein) if axis_kind == AxisKind::Rader => {
            AxisWindowGraphStage::BluesteinFallback
        }
        Some(LargeBridgeRoute::Bluestein) => AxisWindowGraphStage::Bluestein,
        None => match axis_kind {
            AxisKind::Rader => AxisWindowGraphStage::Rader,
            AxisKind::Bluestein => AxisWindowGraphStage::Bluestein,
            AxisKind::Mixed => {
                return Err(FftError::LargeGraphStageUnsupported {
                    stage: "four-step-prime-window",
                    reason: "mixed-radix window is missing its stage metadata",
                });
            }
        },
    };
    Ok(vec![stage; count])
}

impl TiledTransposePlan {
    fn new(
        device: &wgpu::Device,
        nx: usize,
        ny: usize,
        repetitions: usize,
        limits: SchedulerLimits,
    ) -> Result<Self> {
        if device.limits().max_compute_invocations_per_workgroup < TRANSPOSE_WORKGROUP_SIZE
            || device.limits().max_compute_workgroup_size_x < TRANSPOSE_TILE
            || device.limits().max_compute_workgroup_size_y < TRANSPOSE_TILE
        {
            return Err(FftError::LargeGraphStageUnsupported {
                stage: "four-step-stripe-transpose",
                reason: "device cannot run the portable 16x16 transpose workgroup",
            });
        }
        let max_tile_elements = limits
            .max_storage_buffer_binding_size
            .min(limits.max_buffer_size)
            / COMPLEX_F32_BYTES;
        if max_tile_elements == 0 {
            return Err(FftError::WindowScheduleUnsupported {
                reason: "four-step transpose cannot fit one complex value",
                requested_bytes: COMPLEX_F32_BYTES,
                max_bind_bytes: limits.max_storage_buffer_binding_size,
            });
        }
        let max_tile_elements = usize::try_from(max_tile_elements)
            .map_err(|_| FftError::LengthTooLarge { len: usize::MAX })?;
        let (max_tile_width, max_tile_height) = choose_transpose_tile(nx, ny, max_tile_elements);
        let max_tile_bytes = (max_tile_width as u64)
            .checked_mul(max_tile_height as u64)
            .and_then(|elements| elements.checked_mul(COMPLEX_F32_BYTES))
            .ok_or(FftError::LengthTooLarge { len: usize::MAX })?;
        let mut params_by_shape = HashMap::new();
        let tail_width = nx % max_tile_width;
        let tail_height = ny % max_tile_height;
        for width in [max_tile_width, tail_width] {
            for height in [max_tile_height, tail_height] {
                if width == 0 || height == 0 || params_by_shape.contains_key(&(width, height)) {
                    continue;
                }
                let params = StripeTransposeParams {
                    width: u32::try_from(width)
                        .map_err(|_| FftError::LengthTooLarge { len: width })?,
                    height: u32::try_from(height)
                        .map_err(|_| FftError::LengthTooLarge { len: height })?,
                    _pad0: 0,
                    _pad1: 0,
                };
                params_by_shape.insert(
                    (width, height),
                    create_uniform_buffer(
                        device,
                        "wgpu_fft.four_step.tiled_transpose.params",
                        bytemuck::bytes_of(&params),
                    ),
                );
            }
        }

        Ok(Self {
            nx,
            ny,
            repetitions,
            max_tile_width,
            max_tile_height,
            max_tile_bytes,
            params_by_shape,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn execute(
        &self,
        device: &wgpu::Device,
        encoder: &mut CommandRecorder<'_>,
        scheduler: &WindowScheduler,
        executor: &StageExecutor<'_>,
        input: &BufferView<'_>,
        output: &BufferView<'_>,
        tile_input: &wgpu::Buffer,
        tile_output: &wgpu::Buffer,
        bind_group_layout: &wgpu::BindGroupLayout,
        pipeline: &wgpu::ComputePipeline,
    ) -> Result<()> {
        let per_matrix_elements = (self.nx as u64)
            .checked_mul(self.ny as u64)
            .ok_or(FftError::LengthTooLarge { len: usize::MAX })?;
        for repetition in 0..self.repetitions {
            let matrix_base = (repetition as u64)
                .checked_mul(per_matrix_elements)
                .ok_or(FftError::LengthTooLarge { len: usize::MAX })?;
            for y0 in (0..self.ny).step_by(self.max_tile_height) {
                let height = self.max_tile_height.min(self.ny - y0);
                for x0 in (0..self.nx).step_by(self.max_tile_width) {
                    let width = self.max_tile_width.min(self.nx - x0);
                    let tile_elements = (width as u64)
                        .checked_mul(height as u64)
                        .ok_or(FftError::LengthTooLarge { len: usize::MAX })?;
                    let tile_bytes = tile_elements
                        .checked_mul(COMPLEX_F32_BYTES)
                        .ok_or(FftError::LengthTooLarge { len: usize::MAX })?;
                    if x0 == 0 && width == self.nx {
                        let source_element = matrix_base
                            .checked_add((y0 as u64) * self.nx as u64)
                            .ok_or(FftError::LengthTooLarge { len: usize::MAX })?;
                        executor.copy_view_range_to_buffer(
                            encoder,
                            input,
                            source_element * COMPLEX_F32_BYTES,
                            tile_input,
                            0,
                            tile_bytes,
                        )?;
                    } else {
                        for local_y in 0..height {
                            let source_element = matrix_base
                                .checked_add(((y0 + local_y) as u64) * self.nx as u64)
                                .and_then(|value| value.checked_add(x0 as u64))
                                .ok_or(FftError::LengthTooLarge { len: usize::MAX })?;
                            executor.copy_view_range_to_buffer(
                                encoder,
                                input,
                                source_element * COMPLEX_F32_BYTES,
                                tile_input,
                                (local_y as u64) * (width as u64) * COMPLEX_F32_BYTES,
                                (width as u64) * COMPLEX_F32_BYTES,
                            )?;
                        }
                    }

                    let tile_input_view = BufferView::whole(tile_input).prefix(tile_bytes)?;
                    let tile_output_view = BufferView::whole(tile_output).prefix(tile_bytes)?;
                    let input_resource = scheduler
                        .storage_binding_resource(&tile_input_view, ElementFormat::ComplexF32)?;
                    let output_resource = scheduler
                        .storage_binding_resource(&tile_output_view, ElementFormat::ComplexF32)?;
                    let params_buffer = &self.params_by_shape[&(width, height)];
                    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("wgpu_fft.four_step.tiled_transpose.bind_group"),
                        layout: bind_group_layout,
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
                    pass.set_pipeline(pipeline);
                    pass.set_bind_group(0, &bind_group, &[]);
                    let tiles_x = (width as u32).div_ceil(TRANSPOSE_TILE);
                    let tiles_y = (height as u32).div_ceil(TRANSPOSE_TILE);
                    let workgroups = tiles_x.checked_mul(tiles_y).ok_or(
                        FftError::DispatchWorkgroupsUnsupported {
                            workgroups: u32::MAX,
                            max_per_dimension: max_workgroups_per_dimension(device),
                        },
                    )?;
                    let (x, y, z) =
                        split_workgroups(workgroups, max_workgroups_per_dimension(device))?;
                    pass.dispatch_workgroups(x, y, z);

                    if y0 == 0 && height == self.ny {
                        let destination_element = matrix_base
                            .checked_add((x0 as u64) * self.ny as u64)
                            .ok_or(FftError::LengthTooLarge { len: usize::MAX })?;
                        executor.copy_buffer_to_view_range(
                            encoder,
                            tile_output,
                            0,
                            output,
                            destination_element * COMPLEX_F32_BYTES,
                            tile_bytes,
                        )?;
                    } else {
                        for local_x in 0..width {
                            let destination_element = matrix_base
                                .checked_add(((x0 + local_x) as u64) * self.ny as u64)
                                .and_then(|value| value.checked_add(y0 as u64))
                                .ok_or(FftError::LengthTooLarge { len: usize::MAX })?;
                            executor.copy_buffer_to_view_range(
                                encoder,
                                tile_output,
                                (local_x as u64) * (height as u64) * COMPLEX_F32_BYTES,
                                output,
                                destination_element * COMPLEX_F32_BYTES,
                                (height as u64) * COMPLEX_F32_BYTES,
                            )?;
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

fn choose_transpose_tile(nx: usize, ny: usize, max_elements: usize) -> (usize, usize) {
    debug_assert!(nx > 0 && ny > 0 && max_elements > 0);
    let mut candidates = Vec::with_capacity(2);
    if ny <= max_elements {
        candidates.push((nx.min(max_elements / ny).max(1), ny));
    }
    if nx <= max_elements {
        candidates.push((nx, ny.min(max_elements / nx).max(1)));
    }
    if let Some(&(width, height)) = candidates.iter().min_by_key(|&&(width, height)| {
        let x_tiles = nx.div_ceil(width) as u128;
        let y_tiles = ny.div_ceil(height) as u128;
        let gather_per_tile = if width == nx { 1 } else { height as u128 };
        let scatter_per_tile = if height == ny { 1 } else { width as u128 };
        (
            x_tiles * y_tiles * (gather_per_tile + scatter_per_tile),
            std::cmp::Reverse((width as u128) * (height as u128)),
        )
    }) {
        return (width, height);
    }

    let mut width = 1usize;
    while width <= max_elements / width {
        width += 1;
    }
    width = width.saturating_sub(1).max(1).min(nx);
    let height = ny.min(max_elements / width).max(1);
    (width, height)
}

impl ScaleWindowPlan {
    fn new(
        device: &wgpu::Device,
        config: &FftConfig,
        total_complex: u64,
        scale: f32,
        limits: LargePolicyLimits,
        storage_alignment: u64,
    ) -> Result<Self> {
        let lines_total = usize::try_from(total_complex)
            .map_err(|_| FftError::LengthTooLarge { len: usize::MAX })?;
        let window_plan = plan_out_of_core_windows_for_independent_buffers(OutOfCorePlanInput {
            axis_window: configured_axis_window_policy_input(
                config,
                1,
                COMPLEX_F32_BYTES,
                lines_total,
                limits.max_storage_buffer_binding_size,
                AxisKind::Mixed,
                storage_alignment,
            ),
            max_buffer_size: limits.max_buffer_size,
        })?;
        let mut windows = Vec::with_capacity(window_plan.upload_windows.len());
        for window in window_plan.upload_windows {
            let total = u32::try_from(window.line_count).map_err(|_| FftError::LengthTooLarge {
                len: window.line_count,
            })?;
            let params = ScaleParams {
                total_complex: total,
                base_element: 0,
                scale,
                _pad: 0,
            };
            windows.push(ScaleWindowDispatch {
                window,
                params_buffer: create_uniform_buffer(
                    device,
                    "wgpu_fft.four_step.scale.params",
                    bytemuck::bytes_of(&params),
                ),
            });
        }
        let max_window_bytes = windows
            .iter()
            .map(|dispatch| dispatch.window.byte_size)
            .max()
            .ok_or(FftError::ZeroLength)?;
        Ok(Self {
            windows,
            max_window_bytes,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn execute(
        &self,
        device: &wgpu::Device,
        encoder: &mut CommandRecorder<'_>,
        scheduler: &WindowScheduler,
        executor: &StageExecutor<'_>,
        data: &BufferView<'_>,
        stage: &wgpu::Buffer,
        bind_group_layout: &wgpu::BindGroupLayout,
        pipeline: &wgpu::ComputePipeline,
    ) -> Result<()> {
        for dispatch in &self.windows {
            let first_element = dispatch.window.byte_offset / COMPLEX_F32_BYTES;
            let span_elements = dispatch.window.byte_size / COMPLEX_F32_BYTES;
            let direct = scheduler.bind_element_window(
                data,
                first_element,
                span_elements,
                ElementFormat::ComplexF32,
            );
            let (window, copied) = match direct {
                Ok((window, 0)) => (window, false),
                _ => {
                    executor.copy_view_range_to_buffer(
                        encoder,
                        data,
                        dispatch.window.byte_offset,
                        stage,
                        0,
                        dispatch.window.byte_size,
                    )?;
                    (
                        BufferView::whole(stage).prefix(dispatch.window.byte_size)?,
                        true,
                    )
                }
            };
            let resource =
                scheduler.storage_binding_resource(&window, ElementFormat::ComplexF32)?;
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("wgpu_fft.four_step.scale.bind_group"),
                layout: bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource,
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: dispatch.params_buffer.as_entire_binding(),
                    },
                ],
            });
            let pass = encoder.pass();
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let workgroups = u32::try_from(dispatch.window.line_count)
                .map_err(|_| FftError::LengthTooLarge { len: usize::MAX })?
                .div_ceil(SCALE_WORKGROUP_SIZE);
            let (x, y, z) = split_workgroups(workgroups, max_workgroups_per_dimension(device))?;
            pass.dispatch_workgroups(x, y, z);
            if copied {
                executor.copy_buffer_to_view_range(
                    encoder,
                    stage,
                    0,
                    data,
                    dispatch.window.byte_offset,
                    dispatch.window.byte_size,
                )?;
            }
        }
        Ok(())
    }
}

pub(crate) fn effective_scheduler_limits(
    stored: LargePolicyLimits,
    device: &wgpu::Limits,
) -> SchedulerLimits {
    let max_buffer_size = stored.max_buffer_size.min(device.max_buffer_size);
    SchedulerLimits {
        max_storage_buffer_binding_size: stored
            .max_storage_buffer_binding_size
            .min(device.max_storage_buffer_binding_size)
            .min(max_buffer_size),
        max_buffer_size,
        storage_alignment: u64::from(device.min_storage_buffer_offset_alignment.max(1)),
        copy_alignment: 4,
    }
}

fn create_four_step_buffer(
    device: &wgpu::Device,
    label: &'static str,
    size: u64,
    max_buffer_size: u64,
) -> Result<wgpu::Buffer> {
    if size == 0 {
        return Err(FftError::ZeroLength);
    }
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
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    }))
}

fn validate_copy_capable_view(
    view: &BufferView<'_>,
    required: wgpu::BufferUsages,
    usage_name: &'static str,
) -> Result<()> {
    if view
        .ranges(0, view.size())?
        .iter()
        .all(|range| range.buffer.usage().contains(required))
    {
        Ok(())
    } else {
        Err(FftError::BufferViewMissingUsage { usage: usage_name })
    }
}

fn create_uniform_buffer(device: &wgpu::Device, label: &'static str, bytes: &[u8]) -> wgpu::Buffer {
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: bytes.len() as u64,
        usage: wgpu::BufferUsages::UNIFORM,
        mapped_at_creation: true,
    });
    {
        let mut mapped = buffer
            .slice(..)
            .get_mapped_range_mut()
            .expect("uniform buffer is mapped at creation");
        mapped.copy_from_slice(bytes);
    }
    buffer.unmap();
    buffer
}

#[allow(clippy::too_many_arguments)]
fn build_four_step_graph(
    required_bytes: u64,
    total_complex: u64,
    rank: usize,
    axes: &[FourStepGraphAxis<'_>],
    has_scale: bool,
    axis_stage_bytes: u64,
    permutation_bytes: Option<u64>,
    limits: LargePolicyLimits,
    storage_alignment: u64,
) -> Result<LargeExecutionPlan> {
    let requirements = |scratch_bytes| {
        StageRequirements::new(
            limits.max_storage_buffer_binding_size,
            limits.max_buffer_size,
            storage_alignment,
            4,
            scratch_bytes,
        )
    };
    let range = |buffer| LogicalRange::new(buffer, 0, required_bytes, ElementFormat::ComplexF32);
    let mut graph = LargeExecutionGraph::new(if rank == 2 {
        "c2c-four-step-rank2"
    } else {
        "c2c-four-step-rank-nd"
    });
    graph.push_stage(
        LargeStage::WindowedHelper {
            label: "four-step-transpose-scratch",
            range: range(LogicalBufferId::Temp(0))?,
        },
        requirements(required_bytes)?,
    )?;
    for (label, index) in [
        ("four-step-axis-stage-input", 0),
        ("four-step-axis-stage-output", 1),
    ] {
        graph.push_stage(
            LargeStage::WindowedHelper {
                label,
                range: LogicalRange::new(
                    LogicalBufferId::Stage(index),
                    0,
                    axis_stage_bytes,
                    ElementFormat::ComplexF32,
                )?,
            },
            requirements(axis_stage_bytes)?,
        )?;
    }
    let mut workspace_stage_index = 4u32;
    for graph_axis in axes {
        for (index, &helper_bytes) in graph_axis.helper_bytes.iter().enumerate() {
            graph.push_stage(
                LargeStage::WindowedHelper {
                    label: four_step_axis_workspace_label(graph_axis.axis, index),
                    range: LogicalRange::new(
                        LogicalBufferId::Stage(workspace_stage_index),
                        0,
                        helper_bytes,
                        ElementFormat::ComplexF32,
                    )?,
                },
                requirements(helper_bytes)?,
            )?;
            workspace_stage_index = workspace_stage_index
                .checked_add(1)
                .ok_or(FftError::LengthTooLarge { len: usize::MAX })?;
        }
    }
    if let Some(bytes) = permutation_bytes {
        let labels = if rank == 2 {
            ["four-step-stripe-input", "four-step-stripe-output"]
        } else {
            [
                "four-step-permutation-input",
                "four-step-permutation-output",
            ]
        };
        for (label, index) in labels.into_iter().zip([2, 3]) {
            graph.push_stage(
                LargeStage::WindowedHelper {
                    label,
                    range: LogicalRange::new(
                        LogicalBufferId::Stage(index),
                        0,
                        bytes,
                        ElementFormat::ComplexF32,
                    )?,
                },
                requirements(bytes)?,
            )?;
        }
    }

    for (step_index, graph_axis) in axes.iter().enumerate() {
        let source_buffer = if step_index == 0 {
            LogicalBufferId::Input
        } else if step_index % 2 == 1 {
            LogicalBufferId::Temp(0)
        } else {
            LogicalBufferId::Output
        };
        let result_buffer = if step_index % 2 == 0 {
            LogicalBufferId::Temp(0)
        } else {
            LogicalBufferId::Output
        };
        let intermediate_buffer = if step_index % 2 == 0 {
            LogicalBufferId::Output
        } else {
            LogicalBufferId::Temp(0)
        };

        let fft_input = if graph_axis.axis == 0 {
            source_buffer
        } else {
            let bytes = permutation_bytes.ok_or(FftError::LargeGraphStageUnsupported {
                stage: "four-step-permutation",
                reason: "non-front axis is missing permutation staging",
            })?;
            push_four_step_permutation_stage(
                &mut graph,
                rank,
                graph_axis.axis,
                true,
                range(source_buffer)?,
                range(result_buffer)?,
                total_complex,
                requirements(bytes)?,
            )?;
            result_buffer
        };
        let fft_output = if graph_axis.axis == 0 {
            result_buffer
        } else {
            intermediate_buffer
        };
        for &kind in graph_axis.stages {
            graph.push_stage(
                LargeStage::WindowedKernel {
                    label: four_step_axis_stage_label(graph_axis.axis, kind),
                    input: range(fft_input)?,
                    output: range(fft_output)?,
                    work_items: total_complex,
                },
                requirements(axis_stage_bytes)?,
            )?;
        }
        if graph_axis.axis != 0 {
            let bytes = permutation_bytes.expect("checked above for a non-front axis");
            push_four_step_permutation_stage(
                &mut graph,
                rank,
                graph_axis.axis,
                false,
                range(intermediate_buffer)?,
                range(result_buffer)?,
                total_complex,
                requirements(bytes)?,
            )?;
        }
    }
    let final_buffer = if axes.len() % 2 == 0 {
        LogicalBufferId::Output
    } else {
        LogicalBufferId::Temp(0)
    };
    if has_scale {
        graph.push_stage(
            LargeStage::Scale {
                label: "four-step-scale",
                range: range(final_buffer)?,
                work_items: total_complex,
            },
            requirements(axis_stage_bytes)?,
        )?;
    }
    if axes.len() % 2 == 1 {
        graph.push_stage(
            LargeStage::Copy {
                label: "four-step-final-copy",
                src: range(LogicalBufferId::Temp(0))?,
                dst: range(LogicalBufferId::Output)?,
            },
            requirements(0)?,
        )?;
    }
    Ok(LargeExecutionPlan::new(graph))
}

#[allow(clippy::too_many_arguments)]
fn push_four_step_permutation_stage(
    graph: &mut LargeExecutionGraph,
    rank: usize,
    axis: usize,
    to_front: bool,
    input: LogicalRange,
    output: LogicalRange,
    work_items: u64,
    requirements: StageRequirements,
) -> Result<()> {
    if rank == 2 && axis == 1 {
        graph.push_stage(
            LargeStage::StripeTranspose {
                label: if to_front {
                    "four-step-stripe-transpose-forward"
                } else {
                    "four-step-stripe-transpose-back"
                },
                input,
                output,
                work_items,
            },
            requirements,
        )
    } else {
        graph.push_stage(
            LargeStage::Permutation {
                label: four_step_permutation_label(axis, to_front),
                input,
                output,
                work_items,
            },
            requirements,
        )
    }
}

fn four_step_axis_stage_label(axis: usize, kind: AxisWindowGraphStage) -> &'static str {
    match (axis, kind) {
        (0, AxisWindowGraphStage::Mixed(AxisStageKind::Stockham { .. })) => {
            "four-step-axis0-windowed-stockham-stage"
        }
        (0, AxisWindowGraphStage::Mixed(AxisStageKind::FusedPow2 { .. })) => {
            "four-step-axis0-windowed-fused-pow2"
        }
        (0, AxisWindowGraphStage::Mixed(AxisStageKind::FusedSmooth { .. })) => {
            "four-step-axis0-windowed-fused-smooth"
        }
        (1, AxisWindowGraphStage::Mixed(AxisStageKind::Stockham { .. })) => {
            "four-step-axis1-windowed-stockham-stage"
        }
        (1, AxisWindowGraphStage::Mixed(AxisStageKind::FusedPow2 { .. })) => {
            "four-step-axis1-windowed-fused-pow2"
        }
        (1, AxisWindowGraphStage::Mixed(AxisStageKind::FusedSmooth { .. })) => {
            "four-step-axis1-windowed-fused-smooth"
        }
        (2, AxisWindowGraphStage::Mixed(AxisStageKind::Stockham { .. })) => {
            "four-step-axis2-windowed-stockham-stage"
        }
        (2, AxisWindowGraphStage::Mixed(AxisStageKind::FusedPow2 { .. })) => {
            "four-step-axis2-windowed-fused-pow2"
        }
        (2, AxisWindowGraphStage::Mixed(AxisStageKind::FusedSmooth { .. })) => {
            "four-step-axis2-windowed-fused-smooth"
        }
        (3, AxisWindowGraphStage::Mixed(AxisStageKind::Stockham { .. })) => {
            "four-step-axis3-windowed-stockham-stage"
        }
        (3, AxisWindowGraphStage::Mixed(AxisStageKind::FusedPow2 { .. })) => {
            "four-step-axis3-windowed-fused-pow2"
        }
        (3, AxisWindowGraphStage::Mixed(AxisStageKind::FusedSmooth { .. })) => {
            "four-step-axis3-windowed-fused-smooth"
        }
        (0, AxisWindowGraphStage::Rader) => "four-step-axis0-windowed-rader",
        (1, AxisWindowGraphStage::Rader) => "four-step-axis1-windowed-rader",
        (2, AxisWindowGraphStage::Rader) => "four-step-axis2-windowed-rader",
        (3, AxisWindowGraphStage::Rader) => "four-step-axis3-windowed-rader",
        (0, AxisWindowGraphStage::Bluestein) => "four-step-axis0-windowed-bluestein",
        (1, AxisWindowGraphStage::Bluestein) => "four-step-axis1-windowed-bluestein",
        (2, AxisWindowGraphStage::Bluestein) => "four-step-axis2-windowed-bluestein",
        (3, AxisWindowGraphStage::Bluestein) => "four-step-axis3-windowed-bluestein",
        (0, AxisWindowGraphStage::BluesteinFallback) => {
            "four-step-axis0-windowed-bluestein-fallback"
        }
        (1, AxisWindowGraphStage::BluesteinFallback) => {
            "four-step-axis1-windowed-bluestein-fallback"
        }
        (2, AxisWindowGraphStage::BluesteinFallback) => {
            "four-step-axis2-windowed-bluestein-fallback"
        }
        (3, AxisWindowGraphStage::BluesteinFallback) => {
            "four-step-axis3-windowed-bluestein-fallback"
        }
        _ => "four-step-axis-windowed-stage",
    }
}

fn four_step_permutation_label(axis: usize, to_front: bool) -> &'static str {
    match (axis, to_front) {
        (1, true) => "four-step-axis1-permute-to-front",
        (1, false) => "four-step-axis1-permute-from-front",
        (2, true) => "four-step-axis2-permute-to-front",
        (2, false) => "four-step-axis2-permute-from-front",
        (3, true) => "four-step-axis3-permute-to-front",
        (3, false) => "four-step-axis3-permute-from-front",
        (_, true) => "four-step-axis-permute-to-front",
        (_, false) => "four-step-axis-permute-from-front",
    }
}

fn four_step_axis_workspace_label(axis: usize, index: usize) -> &'static str {
    match (axis, index) {
        (0, 0) => "four-step-axis0-child-workspace-main",
        (0, 1) => "four-step-axis0-child-workspace-tail",
        (1, 0) => "four-step-axis1-child-workspace-main",
        (1, 1) => "four-step-axis1-child-workspace-tail",
        (2, 0) => "four-step-axis2-child-workspace-main",
        (2, 1) => "four-step-axis2-child-workspace-tail",
        (3, 0) => "four-step-axis3-child-workspace-main",
        (3, 1) => "four-step-axis3-child-workspace-tail",
        (0, _) => "four-step-axis0-child-workspace-extra",
        (1, _) => "four-step-axis1-child-workspace-extra",
        (2, _) => "four-step-axis2-child-workspace-extra",
        (3, _) => "four-step-axis3-child-workspace-extra",
        _ => "four-step-axis-child-workspace",
    }
}

pub(crate) fn generate_four_step_wgsl_for_key(key: &FourStepStageKey) -> String {
    match key.kind {
        FourStepKernelKind::StripeTranspose => {
            debug_assert_eq!(key.workgroup_size, TRANSPOSE_WORKGROUP_SIZE);
            format!(
                r#"
struct Params {{
  width: u32,
  height: u32,
  pad0: u32,
  pad1: u32,
}}

@group(0) @binding(0) var<storage, read> input: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec2<f32>>;
@group(0) @binding(2) var<uniform> params: Params;

const TILE: u32 = {tile}u;
var<workgroup> tile_data: array<vec2<f32>, {tile_storage}>;

@compute @workgroup_size({tile}, {tile}, 1)
fn main(
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(num_workgroups) nwg: vec3<u32>,
) {{
  let wg_flat: u32 = (wid.z * nwg.y + wid.y) * nwg.x + wid.x;
  let tiles_x: u32 = (params.width + TILE - 1u) / TILE;
  let tiles_y: u32 = (params.height + TILE - 1u) / TILE;
  if (wg_flat >= tiles_x * tiles_y) {{ return; }}
  let tile_x: u32 = wg_flat % tiles_x;
  let tile_y: u32 = wg_flat / tiles_x;
  let source_x: u32 = tile_x * TILE + lid.x;
  let source_y: u32 = tile_y * TILE + lid.y;
  if (source_x < params.width && source_y < params.height) {{
    tile_data[lid.y * (TILE + 1u) + lid.x] = input[source_y * params.width + source_x];
  }}
  workgroupBarrier();
  let output_x: u32 = tile_x * TILE + lid.y;
  let output_y: u32 = tile_y * TILE + lid.x;
  if (output_x < params.width && output_y < params.height) {{
    output[output_x * params.height + output_y] = tile_data[lid.x * (TILE + 1u) + lid.y];
  }}
}}
"#,
                tile = TRANSPOSE_TILE,
                tile_storage = TRANSPOSE_TILE * (TRANSPOSE_TILE + 1),
            )
        }
        FourStepKernelKind::Scale => format!(
            r#"
struct Params {{
  total_complex: u32,
  base_element: u32,
  scale: f32,
  pad: u32,
}}

@group(0) @binding(0) var<storage, read_write> data: array<vec2<f32>>;
@group(0) @binding(1) var<uniform> params: Params;

@compute @workgroup_size({workgroup_size}, 1, 1)
fn main(
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(num_workgroups) nwg: vec3<u32>,
) {{
  let wg_flat: u32 = (wid.z * nwg.y + wid.y) * nwg.x + wid.x;
  if (wg_flat > params.total_complex / {workgroup_size}u) {{ return; }}
  let i: u32 = wg_flat * {workgroup_size}u + lid.x;
  if (i >= params.total_complex) {{ return; }}
  let index: u32 = params.base_element + i;
  data[index] = data[index] * vec2<f32>(params.scale, params.scale);
}}
"#,
            workgroup_size = key.workgroup_size,
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tuning::FftTuning;

    #[test]
    fn configured_window_policy_uses_public_tuning_and_preserves_defaults() {
        let default_config = FftConfig::new_nd([8, 8]);
        let default = configured_axis_window_policy_input(
            &default_config,
            4096,
            264,
            1024,
            65536,
            AxisKind::Bluestein,
            256,
        );
        assert_eq!(default.swap_to_2_stage_4_step, 0);
        assert_eq!(default.swap_to_3_stage_4_step, 0);
        assert_eq!(default.grouped_batch, None);
        assert_eq!(default.out_of_core_burst_windows, 1);

        let tuned_config = FftConfig::new_nd([8, 8]).with_tuning(
            FftTuning::new()
                .with_swap_to_2_stage_4_step(1024)
                .with_swap_to_3_stage_4_step(4096)
                .with_grouped_batch(Some(8)),
        );
        let tuned = configured_axis_window_policy_input(
            &tuned_config,
            4096,
            264,
            1024,
            65536,
            AxisKind::Bluestein,
            256,
        );
        assert_eq!(tuned.swap_to_2_stage_4_step, 1024);
        assert_eq!(tuned.swap_to_3_stage_4_step, 4096);
        assert_eq!(tuned.grouped_batch, Some(8));
        let resolved =
            crate::runtime::large_policy::resolve_out_of_core_axis_window_policy(tuned).unwrap();
        assert_eq!(resolved.num_axis_uploads, 3);
        assert_eq!(resolved.grouped_batch, Some(8));
    }

    #[test]
    fn effective_limits_take_componentwise_minimum() {
        let mut device = wgpu::Limits::defaults();
        device.max_storage_buffer_binding_size = 4096;
        device.max_buffer_size = 8192;
        device.min_storage_buffer_offset_alignment = 256;
        let limits = effective_scheduler_limits(
            LargePolicyLimits {
                max_storage_buffer_binding_size: 512,
                max_buffer_size: 16384,
            },
            &device,
        );
        assert_eq!(limits.max_storage_buffer_binding_size, 512);
        assert_eq!(limits.max_buffer_size, 8192);
        assert_eq!(limits.storage_alignment, 256);
    }

    #[test]
    fn effective_limits_cap_bindings_to_the_effective_buffer_size() {
        let mut device = wgpu::Limits::defaults();
        device.max_storage_buffer_binding_size = 4096;
        device.max_buffer_size = 8192;
        let limits = effective_scheduler_limits(
            LargePolicyLimits {
                max_storage_buffer_binding_size: 1024,
                max_buffer_size: 256,
            },
            &device,
        );
        assert_eq!(limits.max_storage_buffer_binding_size, 256);
        assert_eq!(limits.max_buffer_size, 256);
    }

    #[test]
    fn generated_transpose_keeps_barrier_outside_element_guards() {
        let wgsl = generate_four_step_wgsl_for_key(&FourStepStageKey::new(
            FourStepKernelKind::StripeTranspose,
            TRANSPOSE_WORKGROUP_SIZE,
        ));
        let load_guard = wgsl.find("if (source_x <").unwrap();
        let barrier = wgsl.find("workgroupBarrier();").unwrap();
        let store_guard = wgsl.find("if (output_x <").unwrap();
        assert!(load_guard < barrier && barrier < store_guard);
        crate::runtime::assert_workgroup_var_written_before_read(&wgsl, "tile_data");
    }

    #[test]
    fn graph_reports_logical_passes_and_every_owned_workspace() {
        let axis0_stages = [AxisWindowGraphStage::Mixed(AxisStageKind::FusedSmooth {
            axis_length: 15,
        })];
        let axis1_stages = [
            AxisWindowGraphStage::Mixed(AxisStageKind::Stockham { radix: 8, ns: 8 }),
            AxisWindowGraphStage::Mixed(AxisStageKind::Stockham { radix: 2, ns: 16 }),
        ];
        let axes = [
            FourStepGraphAxis {
                axis: 0,
                stages: &axis0_stages,
                helper_bytes: &[128],
            },
            FourStepGraphAxis {
                axis: 1,
                stages: &axis1_stages,
                helper_bytes: &[1024, 64],
            },
        ];
        let graph = build_four_step_graph(
            2048,
            256,
            2,
            &axes,
            true,
            256,
            Some(512),
            LargePolicyLimits {
                max_storage_buffer_binding_size: 512,
                max_buffer_size: 4096,
            },
            32,
        )
        .unwrap();
        let stages = graph.graph().stages();
        for label in [
            "four-step-transpose-scratch",
            "four-step-axis-stage-input",
            "four-step-axis-stage-output",
            "four-step-stripe-input",
            "four-step-stripe-output",
            "four-step-axis0-child-workspace-main",
            "four-step-axis1-child-workspace-main",
            "four-step-axis1-child-workspace-tail",
        ] {
            assert!(stages.iter().any(|stage| {
                stage.label() == label && matches!(stage, LargeStage::WindowedHelper { .. })
            }));
        }
        assert_eq!(
            stages
                .iter()
                .filter(|stage| {
                    matches!(
                        stage,
                        LargeStage::WindowedKernel { .. }
                            | LargeStage::StripeTranspose { .. }
                            | LargeStage::Scale { .. }
                    )
                })
                .count(),
            6
        );
    }

    #[test]
    fn transpose_tiles_split_both_dimensions_within_the_binding_budget() {
        assert_eq!(choose_transpose_tile(15, 14, 32), (15, 2));
        assert_eq!(
            choose_transpose_tile(256, 1_048_576, 268_435_455),
            (256, 1_048_575)
        );
        let (width, height) = choose_transpose_tile(4096, 1_048_576, 32);
        assert!(width > 0 && height > 0);
        assert!(width * height <= 32);
        assert!(width < 4096 && height < 1_048_576);
    }

    #[test]
    fn rank3_graph_reports_permutations_and_final_copy() {
        let stages = [AxisWindowGraphStage::Mixed(AxisStageKind::FusedSmooth {
            axis_length: 9,
        })];
        let axes = [
            FourStepGraphAxis {
                axis: 0,
                stages: &stages,
                helper_bytes: &[],
            },
            FourStepGraphAxis {
                axis: 1,
                stages: &stages,
                helper_bytes: &[],
            },
            FourStepGraphAxis {
                axis: 2,
                stages: &stages,
                helper_bytes: &[],
            },
        ];
        let graph = build_four_step_graph(
            2520,
            315,
            3,
            &axes,
            false,
            256,
            Some(256),
            LargePolicyLimits {
                max_storage_buffer_binding_size: 256,
                max_buffer_size: 4096,
            },
            32,
        )
        .unwrap();
        let stages = graph.graph().stages();
        assert_eq!(
            stages
                .iter()
                .filter(|stage| matches!(stage, LargeStage::Permutation { .. }))
                .count(),
            4
        );
        assert!(stages.iter().any(|stage| {
            stage.label() == "four-step-final-copy" && matches!(stage, LargeStage::Copy { .. })
        }));
    }
}
