use std::collections::HashMap;

use bytemuck::{Pod, Zeroable};

use crate::config::{FftConfig, Normalization};
use crate::error::{FftError, Result};
use crate::runtime::axis_plan::{
    AxisLayout, AxisPlan, AxisPlanConfig, AxisPrecision, AxisTwiddleLutPool,
};
use crate::runtime::buffer_view::{BufferSegment, BufferView};
use crate::runtime::dispatch::{max_workgroups_per_dimension, split_workgroups};
use crate::runtime::four_step::{effective_scheduler_limits, generate_four_step_wgsl_for_key};
use crate::runtime::large_graph::{
    ElementFormat, LargeExecutionGraph, LargeExecutionPlan, LargeStage, LogicalBufferId,
    LogicalRange, StageRequirements,
};
use crate::runtime::large_policy::{LargeFactorSplit, LargePolicyLimits};
use crate::runtime::pipeline_cache::{
    with_device_pipeline_cache, ComputePipelineCacheKey, FourStepKernelKind, FourStepStageKey,
    ShaderCacheKey,
};
use crate::runtime::recorder::CommandRecorder;
use crate::runtime::stage_executor::StageExecutor;
use crate::runtime::window_scheduler::{SchedulerLimits, WindowScheduler};

const COMPLEX_F32_BYTES: u64 = 8;
const TRANSPOSE_TILE: u32 = 16;
const TRANSPOSE_WORKGROUP_SIZE: u32 = TRANSPOSE_TILE * TRANSPOSE_TILE;
const SCALE_WORKGROUP_SIZE: u32 = 64;
const MIN_SEGMENTED_BURST_DEPTH: usize = 1;
const MAX_SEGMENTED_BURST_DEPTH: usize = 3;

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct TransposeParams {
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

/// GPU-resident execution for one logical volume split across physical buffers.
///
/// The typed schedule is the execution authority. `diagnostic_graph` is a
/// bounded physical-stage summary only; it deliberately never represents the
/// full segmented volume as one `LogicalRange`.
pub(crate) struct SegmentedVolumeC2cPlan {
    required_bytes: u64,
    stored_limits: LargePolicyLimits,
    arena: SegmentedArena,
    schedule: SegmentedVolumeSchedule,
    burst_ring: Vec<BurstStagePair>,
    row_plans: Vec<AxisPlan>,
    transpose_pipeline: wgpu::ComputePipeline,
    transpose_bind_group_layout: wgpu::BindGroupLayout,
    scale_pipeline: Option<wgpu::ComputePipeline>,
    scale_bind_group_layout: Option<wgpu::BindGroupLayout>,
    diagnostic_graph: LargeExecutionPlan,
    staging_bytes: Vec<u64>,
    factor_splits: Vec<LargeFactorSplit>,
    twiddle_lut_storage_bytes: u64,
}

struct SegmentedArena {
    buffers: Vec<wgpu::Buffer>,
    segments: Vec<SegmentedArenaSegment>,
    total_bytes: u64,
}

struct BurstStagePair {
    stage_a: wgpu::Buffer,
    stage_b: wgpu::Buffer,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SegmentedArenaSegment {
    logical_offset: u64,
    size_bytes: u64,
}

struct SegmentedVolumeSchedule {
    stages: Vec<SegmentedVolumeStage>,
}

enum SegmentedVolumeStage {
    Upload,
    FrontRows(FrontRowBurstPlan),
    SlabAxis(SlabAxisPlan),
    Scale(SegmentedScalePlan),
    Download,
}

struct FrontRowBurstPlan {
    axis: usize,
    max_window_bytes: u64,
    windows: Vec<FrontRowWindow>,
}

struct FrontRowWindow {
    byte_offset: u64,
    byte_size: u64,
    plan_index: usize,
}

struct SlabAxisPlan {
    axis: usize,
    axis_len: usize,
    prefix: usize,
    prefix_chunk: usize,
    chunks_per_repetition: usize,
    dispatch_count: usize,
    max_slab_bytes: u64,
    variants: Vec<SlabVariant>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SlabDispatch {
    matrix_base: u64,
    prefix_start: usize,
    prefix_count: usize,
    slab_bytes: u64,
    variant_index: usize,
}

impl SlabDispatch {
    const EMPTY: Self = Self {
        matrix_base: 0,
        prefix_start: 0,
        prefix_count: 0,
        slab_bytes: 0,
        variant_index: 0,
    };
}

struct SlabVariant {
    prefix_count: usize,
    plan_index: usize,
    to_front_params_buffer: wgpu::Buffer,
    from_front_params_buffer: wgpu::Buffer,
}

struct SegmentedScalePlan {
    max_chunk_bytes: u64,
    dispatches: Vec<ScaleDispatch>,
}

struct ScaleDispatch {
    segment_index: usize,
    offset_bytes: u64,
    size_bytes: u64,
    params_buffer: wgpu::Buffer,
}

pub(crate) fn validate_segmented_burst_depth(burst_depth: usize) -> Result<()> {
    if (MIN_SEGMENTED_BURST_DEPTH..=MAX_SEGMENTED_BURST_DEPTH).contains(&burst_depth) {
        Ok(())
    } else {
        Err(FftError::LargeGraphStageUnsupported {
            stage: "segmented-volume-burst-ring",
            reason: "segmented full-volume burst depth must be in 1..=3",
        })
    }
}

impl SegmentedVolumeC2cPlan {
    pub(crate) fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        config: &FftConfig,
        stored_limits: LargePolicyLimits,
        burst_depth: usize,
    ) -> Result<Self> {
        validate_segmented_burst_depth(burst_depth)?;
        if config.shape().len() < 2 || config.axes().len() < 2 {
            return Err(FftError::LargeGraphStageUnsupported {
                stage: "segmented-volume-plan",
                reason: "segmented full-volume execution requires rank >= 2 and at least two selected axes",
            });
        }

        let scheduler_limits = effective_scheduler_limits(stored_limits, &device.limits());
        let planning_limits = LargePolicyLimits {
            max_storage_buffer_binding_size: scheduler_limits.max_storage_buffer_binding_size,
            max_buffer_size: scheduler_limits.max_buffer_size,
        };
        let required_bytes = config.required_buffer_size_bytes()?;
        if required_bytes <= planning_limits.max_buffer_size {
            return Err(FftError::LargeGraphStageUnsupported {
                stage: "segmented-volume-plan",
                reason: "segmented full-volume execution requires a volume above maxBufferSize",
            });
        }

        for &axis in config.axes() {
            crate::runtime::factor_supported_length(config.shape()[axis])?;
        }

        let arena = SegmentedArena::new(device, required_bytes, scheduler_limits)?;
        let bind_stage_bytes = align_down(
            scheduler_limits
                .max_storage_buffer_binding_size
                .min(scheduler_limits.max_buffer_size),
            COMPLEX_F32_BYTES,
        );
        if bind_stage_bytes < COMPLEX_F32_BYTES {
            return Err(FftError::WindowScheduleUnsupported {
                reason: "segmented full-volume execution cannot fit one complex value",
                requested_bytes: COMPLEX_F32_BYTES,
                max_bind_bytes: scheduler_limits.max_storage_buffer_binding_size,
            });
        }

        let total_complex = config.total_complex_len()?;
        let mut twiddle_lut_pool = AxisTwiddleLutPool::default();
        let mut row_plans = Vec::new();
        let mut row_plan_indices = HashMap::<(usize, usize), usize>::new();
        let mut stages = vec![SegmentedVolumeStage::Upload];
        let mut max_stage_bytes = 0u64;

        for &axis in config.axes() {
            let axis_len = config.shape()[axis];
            if axis == 0 {
                let plan = build_front_row_plan(
                    device,
                    queue,
                    config,
                    axis,
                    axis_len,
                    total_complex / axis_len,
                    bind_stage_bytes,
                    &mut row_plans,
                    &mut row_plan_indices,
                    &mut twiddle_lut_pool,
                )?;
                max_stage_bytes = max_stage_bytes.max(plan.max_window_bytes);
                stages.push(SegmentedVolumeStage::FrontRows(plan));
            } else {
                let plan = build_slab_axis_plan(
                    device,
                    queue,
                    config,
                    axis,
                    bind_stage_bytes,
                    &mut row_plans,
                    &mut row_plan_indices,
                    &mut twiddle_lut_pool,
                )?;
                max_stage_bytes = max_stage_bytes.max(plan.max_slab_bytes);
                stages.push(SegmentedVolumeStage::SlabAxis(plan));
            }
        }

        let scale_value = config.scale()?;
        if (scale_value - 1.0).abs() > f32::EPSILON {
            let scale = SegmentedScalePlan::new(device, &arena, scale_value, scheduler_limits)?;
            max_stage_bytes = max_stage_bytes.max(scale.max_chunk_bytes);
            stages.push(SegmentedVolumeStage::Scale(scale));
        }
        stages.push(SegmentedVolumeStage::Download);

        if max_stage_bytes == 0 {
            return Err(FftError::ZeroLength);
        }
        let mut burst_ring = Vec::with_capacity(burst_depth);
        for _ in 0..burst_depth {
            burst_ring.push(BurstStagePair {
                stage_a: create_segmented_buffer(
                    device,
                    "wgpu_fft.segmented_volume.burst_stage_a",
                    max_stage_bytes,
                    planning_limits.max_buffer_size,
                )?,
                stage_b: create_segmented_buffer(
                    device,
                    "wgpu_fft.segmented_volume.burst_stage_b",
                    max_stage_bytes,
                    planning_limits.max_buffer_size,
                )?,
            });
        }

        let transpose_key = ComputePipelineCacheKey::four_step_stage(FourStepStageKey::new(
            FourStepKernelKind::StripeTranspose,
            TRANSPOSE_WORKGROUP_SIZE,
        ));
        let transpose_bind_group_layout = with_device_pipeline_cache(device, |cache| {
            cache.get_bind_group_layout(device, transpose_key.layout)
        });
        let transpose_pipeline = with_device_pipeline_cache(device, |cache| {
            cache.get_compute_pipeline(
                device,
                &transpose_key,
                "wgpu_fft.segmented_volume.transpose.pipeline",
                "wgpu_fft.segmented_volume.transpose.shader",
                || match &transpose_key.shader {
                    ShaderCacheKey::FourStepStage(key) => generate_four_step_wgsl_for_key(key),
                    _ => unreachable!(),
                },
            )
        });

        let has_scale = stages
            .iter()
            .any(|stage| matches!(stage, SegmentedVolumeStage::Scale(_)));
        let (scale_pipeline, scale_bind_group_layout) = if has_scale {
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
                    "wgpu_fft.segmented_volume.scale.pipeline",
                    "wgpu_fft.segmented_volume.scale.shader",
                    || match &scale_key.shader {
                        ShaderCacheKey::FourStepStage(key) => generate_four_step_wgsl_for_key(key),
                        _ => unreachable!(),
                    },
                )
            });
            (Some(pipeline), Some(layout))
        } else {
            (None, None)
        };

        let schedule = SegmentedVolumeSchedule { stages };
        let diagnostic_graph = build_diagnostic_graph(
            &arena,
            &schedule,
            &row_plans,
            max_stage_bytes,
            burst_depth,
            total_complex as u64,
            planning_limits,
            scheduler_limits.storage_alignment,
        )?;
        let mut staging_bytes = arena.segment_sizes();
        staging_bytes.extend(std::iter::repeat_n(max_stage_bytes, 2 * burst_depth));
        staging_bytes.extend(
            row_plans
                .iter()
                .map(AxisPlan::workspace_size_bytes)
                .filter(|&bytes| bytes > 0),
        );
        let factor_splits = config
            .axes()
            .iter()
            .map(|&axis| {
                Ok(LargeFactorSplit {
                    axis: Some(axis),
                    len: config.shape()[axis] as u64,
                    factors: crate::runtime::factor_supported_length(config.shape()[axis])?
                        .into_iter()
                        .map(|factor| factor as u64)
                        .collect(),
                })
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(Self {
            required_bytes,
            stored_limits,
            arena,
            schedule,
            burst_ring,
            row_plans,
            transpose_pipeline,
            transpose_bind_group_layout,
            scale_pipeline,
            scale_bind_group_layout,
            diagnostic_graph,
            staging_bytes,
            factor_splits,
            twiddle_lut_storage_bytes: twiddle_lut_pool.storage_bytes(),
        })
    }

    pub(crate) fn execute_views(
        &self,
        device: &wgpu::Device,
        encoder: &mut CommandRecorder<'_>,
        input: BufferView<'_>,
        output: BufferView<'_>,
    ) -> Result<()> {
        validate_whole_buffer_endpoint(&input)?;
        validate_whole_buffer_endpoint(&output)?;
        validate_endpoint_usage(&input, wgpu::BufferUsages::COPY_SRC, "COPY_SRC")?;
        validate_endpoint_usage(&output, wgpu::BufferUsages::COPY_DST, "COPY_DST")?;
        let input = input.prefix(self.required_bytes)?;
        let output = output.prefix(self.required_bytes)?;

        let scheduler = WindowScheduler::new(effective_scheduler_limits(
            self.stored_limits,
            &device.limits(),
        ));
        let executor = StageExecutor::new(&scheduler);
        executor.validate_graph(self.diagnostic_graph.graph())?;
        let arena_view = self.arena.view()?;

        for stage in &self.schedule.stages {
            match stage {
                SegmentedVolumeStage::Upload => self.upload(encoder, &executor, &input)?,
                SegmentedVolumeStage::FrontRows(plan) => plan.execute(
                    device,
                    encoder,
                    &executor,
                    &arena_view,
                    &self.burst_ring,
                    &self.row_plans,
                )?,
                SegmentedVolumeStage::SlabAxis(plan) => plan.execute(
                    device,
                    encoder,
                    &scheduler,
                    &executor,
                    &arena_view,
                    &self.burst_ring,
                    &self.row_plans,
                    &self.transpose_bind_group_layout,
                    &self.transpose_pipeline,
                )?,
                SegmentedVolumeStage::Scale(plan) => plan.execute(
                    device,
                    encoder,
                    &scheduler,
                    &self.arena,
                    self.scale_bind_group_layout
                        .as_ref()
                        .expect("scale layout exists with scale schedule"),
                    self.scale_pipeline
                        .as_ref()
                        .expect("scale pipeline exists with scale schedule"),
                )?,
                SegmentedVolumeStage::Download => self.download(encoder, &executor, &output)?,
            }
        }
        Ok(())
    }

    pub(crate) fn graph_plan(&self) -> &LargeExecutionPlan {
        &self.diagnostic_graph
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

    fn upload(
        &self,
        encoder: &mut CommandRecorder<'_>,
        executor: &StageExecutor<'_>,
        input: &BufferView<'_>,
    ) -> Result<()> {
        for (index, segment) in self.arena.segments.iter().enumerate() {
            executor.copy_view_range_to_buffer(
                encoder,
                input,
                segment.logical_offset,
                &self.arena.buffers[index],
                0,
                segment.size_bytes,
            )?;
        }
        Ok(())
    }

    fn download(
        &self,
        encoder: &mut CommandRecorder<'_>,
        executor: &StageExecutor<'_>,
        output: &BufferView<'_>,
    ) -> Result<()> {
        for (index, segment) in self.arena.segments.iter().enumerate() {
            executor.copy_buffer_to_view_range(
                encoder,
                &self.arena.buffers[index],
                0,
                output,
                segment.logical_offset,
                segment.size_bytes,
            )?;
        }
        Ok(())
    }
}

impl SegmentedArena {
    fn new(device: &wgpu::Device, total_bytes: u64, limits: SchedulerLimits) -> Result<Self> {
        let segment_bytes = align_down(limits.max_buffer_size, COMPLEX_F32_BYTES);
        if segment_bytes < COMPLEX_F32_BYTES {
            return Err(FftError::WindowScheduleUnsupported {
                reason: "segmented arena cannot fit one complex value",
                requested_bytes: COMPLEX_F32_BYTES,
                max_bind_bytes: limits.max_buffer_size,
            });
        }

        let mut buffers = Vec::new();
        let mut segments = Vec::new();
        let mut logical_offset = 0u64;
        while logical_offset < total_bytes {
            let size_bytes = segment_bytes.min(total_bytes - logical_offset);
            buffers.push(create_segmented_buffer(
                device,
                "wgpu_fft.segmented_volume.arena",
                size_bytes,
                limits.max_buffer_size,
            )?);
            segments.push(SegmentedArenaSegment {
                logical_offset,
                size_bytes,
            });
            logical_offset = logical_offset
                .checked_add(size_bytes)
                .ok_or(FftError::LengthTooLarge { len: usize::MAX })?;
        }

        Ok(Self {
            buffers,
            segments,
            total_bytes,
        })
    }

    fn view(&self) -> Result<BufferView<'_>> {
        let segments = self
            .buffers
            .iter()
            .zip(&self.segments)
            .map(|(buffer, segment)| BufferSegment::new(buffer, 0, segment.size_bytes))
            .collect::<Vec<_>>();
        BufferView::from_segments(&segments, 0, self.total_bytes)
    }

    fn segment_sizes(&self) -> Vec<u64> {
        self.segments
            .iter()
            .map(|segment| segment.size_bytes)
            .collect()
    }
}

impl FrontRowBurstPlan {
    fn execute(
        &self,
        device: &wgpu::Device,
        encoder: &mut CommandRecorder<'_>,
        executor: &StageExecutor<'_>,
        arena: &BufferView<'_>,
        burst_ring: &[BurstStagePair],
        row_plans: &[AxisPlan],
    ) -> Result<()> {
        let _ = self.axis;
        debug_assert!(!burst_ring.is_empty());
        for burst in self.windows.chunks(burst_ring.len()) {
            for (window, stages) in burst.iter().zip(burst_ring) {
                executor.copy_view_range_to_buffer(
                    encoder,
                    arena,
                    window.byte_offset,
                    &stages.stage_a,
                    0,
                    window.byte_size,
                )?;
            }
            for (window, stages) in burst.iter().zip(burst_ring) {
                row_plans[window.plan_index].execute_views(
                    device,
                    encoder,
                    BufferView::whole(&stages.stage_a).prefix(window.byte_size)?,
                    BufferView::whole(&stages.stage_b).prefix(window.byte_size)?,
                )?;
            }
            for (window, stages) in burst.iter().zip(burst_ring) {
                executor.copy_buffer_to_view_range(
                    encoder,
                    &stages.stage_b,
                    0,
                    arena,
                    window.byte_offset,
                    window.byte_size,
                )?;
            }
        }
        Ok(())
    }
}

impl SlabAxisPlan {
    #[allow(clippy::too_many_arguments)]
    fn execute(
        &self,
        device: &wgpu::Device,
        encoder: &mut CommandRecorder<'_>,
        scheduler: &WindowScheduler,
        executor: &StageExecutor<'_>,
        arena: &BufferView<'_>,
        burst_ring: &[BurstStagePair],
        row_plans: &[AxisPlan],
        transpose_layout: &wgpu::BindGroupLayout,
        transpose_pipeline: &wgpu::ComputePipeline,
    ) -> Result<()> {
        let _ = self.axis;
        debug_assert!(!burst_ring.is_empty());
        for burst_start in (0..self.dispatch_count).step_by(burst_ring.len()) {
            let burst_len = burst_ring.len().min(self.dispatch_count - burst_start);
            let mut burst = [SlabDispatch::EMPTY; MAX_SEGMENTED_BURST_DEPTH];
            for (slot, dispatch) in burst.iter_mut().take(burst_len).enumerate() {
                *dispatch = self.dispatch_for_index(burst_start + slot)?;
            }
            let burst = &burst[..burst_len];
            for (dispatch, stages) in burst.iter().zip(burst_ring) {
                self.gather_dispatch(encoder, executor, arena, dispatch, &stages.stage_a)?;
            }
            for (dispatch, stages) in burst.iter().zip(burst_ring) {
                let variant = &self.variants[dispatch.variant_index];
                dispatch_transpose(
                    device,
                    encoder,
                    scheduler,
                    &stages.stage_a,
                    &stages.stage_b,
                    dispatch.slab_bytes,
                    &variant.to_front_params_buffer,
                    dispatch.prefix_count,
                    self.axis_len,
                    transpose_layout,
                    transpose_pipeline,
                )?;
            }
            for (dispatch, stages) in burst.iter().zip(burst_ring) {
                let variant = &self.variants[dispatch.variant_index];
                row_plans[variant.plan_index].execute_views(
                    device,
                    encoder,
                    BufferView::whole(&stages.stage_b).prefix(dispatch.slab_bytes)?,
                    BufferView::whole(&stages.stage_a).prefix(dispatch.slab_bytes)?,
                )?;
            }
            for (dispatch, stages) in burst.iter().zip(burst_ring) {
                let variant = &self.variants[dispatch.variant_index];
                dispatch_transpose(
                    device,
                    encoder,
                    scheduler,
                    &stages.stage_a,
                    &stages.stage_b,
                    dispatch.slab_bytes,
                    &variant.from_front_params_buffer,
                    self.axis_len,
                    dispatch.prefix_count,
                    transpose_layout,
                    transpose_pipeline,
                )?;
            }
            for (dispatch, stages) in burst.iter().zip(burst_ring) {
                self.scatter_dispatch(encoder, executor, arena, dispatch, &stages.stage_b)?;
            }
        }
        Ok(())
    }

    fn dispatch_for_index(&self, index: usize) -> Result<SlabDispatch> {
        debug_assert!(index < self.dispatch_count);
        let (matrix_base, prefix_start, prefix_count, slab_bytes) = slab_dispatch_geometry(
            index,
            self.axis_len,
            self.prefix,
            self.prefix_chunk,
            self.chunks_per_repetition,
        )?;
        let variant_index = self
            .variants
            .iter()
            .position(|variant| variant.prefix_count == prefix_count)
            .expect("slab schedule contains main and tail variants");
        Ok(SlabDispatch {
            matrix_base,
            prefix_start,
            prefix_count,
            slab_bytes,
            variant_index,
        })
    }

    fn gather_dispatch(
        &self,
        encoder: &mut CommandRecorder<'_>,
        executor: &StageExecutor<'_>,
        arena: &BufferView<'_>,
        dispatch: &SlabDispatch,
        stage_a: &wgpu::Buffer,
    ) -> Result<()> {
        if dispatch.prefix_count == self.prefix {
            executor.copy_view_range_to_buffer(
                encoder,
                arena,
                dispatch.matrix_base * COMPLEX_F32_BYTES,
                stage_a,
                0,
                dispatch.slab_bytes,
            )?;
        } else {
            let row_bytes = (dispatch.prefix_count as u64) * COMPLEX_F32_BYTES;
            for axis_element in 0..self.axis_len {
                let source_element = dispatch
                    .matrix_base
                    .checked_add((axis_element as u64) * self.prefix as u64)
                    .and_then(|value| value.checked_add(dispatch.prefix_start as u64))
                    .ok_or(FftError::LengthTooLarge { len: usize::MAX })?;
                executor.copy_view_range_to_buffer(
                    encoder,
                    arena,
                    source_element * COMPLEX_F32_BYTES,
                    stage_a,
                    (axis_element as u64) * row_bytes,
                    row_bytes,
                )?;
            }
        }
        Ok(())
    }

    fn scatter_dispatch(
        &self,
        encoder: &mut CommandRecorder<'_>,
        executor: &StageExecutor<'_>,
        arena: &BufferView<'_>,
        dispatch: &SlabDispatch,
        stage_b: &wgpu::Buffer,
    ) -> Result<()> {
        if dispatch.prefix_count == self.prefix {
            executor.copy_buffer_to_view_range(
                encoder,
                stage_b,
                0,
                arena,
                dispatch.matrix_base * COMPLEX_F32_BYTES,
                dispatch.slab_bytes,
            )?;
        } else {
            let row_bytes = (dispatch.prefix_count as u64) * COMPLEX_F32_BYTES;
            for axis_element in 0..self.axis_len {
                let destination_element = dispatch
                    .matrix_base
                    .checked_add((axis_element as u64) * self.prefix as u64)
                    .and_then(|value| value.checked_add(dispatch.prefix_start as u64))
                    .ok_or(FftError::LengthTooLarge { len: usize::MAX })?;
                executor.copy_buffer_to_view_range(
                    encoder,
                    stage_b,
                    (axis_element as u64) * row_bytes,
                    arena,
                    destination_element * COMPLEX_F32_BYTES,
                    row_bytes,
                )?;
            }
        }
        Ok(())
    }
}

fn slab_dispatch_geometry(
    index: usize,
    axis_len: usize,
    prefix: usize,
    prefix_chunk: usize,
    chunks_per_repetition: usize,
) -> Result<(u64, usize, usize, u64)> {
    let repetition = index / chunks_per_repetition;
    let chunk = index % chunks_per_repetition;
    let prefix_start = chunk
        .checked_mul(prefix_chunk)
        .ok_or(FftError::LengthTooLarge { len: usize::MAX })?;
    let prefix_count = prefix_chunk.min(prefix - prefix_start);
    let matrix_elements = (prefix as u64)
        .checked_mul(axis_len as u64)
        .ok_or(FftError::LengthTooLarge { len: usize::MAX })?;
    let matrix_base = (repetition as u64)
        .checked_mul(matrix_elements)
        .ok_or(FftError::LengthTooLarge { len: usize::MAX })?;
    let slab_bytes = (prefix_count as u64)
        .checked_mul(axis_len as u64)
        .and_then(|elements| elements.checked_mul(COMPLEX_F32_BYTES))
        .ok_or(FftError::LengthTooLarge { len: usize::MAX })?;
    Ok((matrix_base, prefix_start, prefix_count, slab_bytes))
}

impl SegmentedScalePlan {
    fn new(
        device: &wgpu::Device,
        arena: &SegmentedArena,
        scale: f32,
        limits: SchedulerLimits,
    ) -> Result<Self> {
        let max_segment_bytes = arena
            .segments
            .iter()
            .map(|segment| segment.size_bytes)
            .max()
            .ok_or(FftError::ZeroLength)?;
        let max_chunk_bytes = segmented_scale_chunk_bytes(max_segment_bytes, limits)?;

        let mut dispatches = Vec::new();
        for (segment_index, segment) in arena.segments.iter().enumerate() {
            let mut offset_bytes = 0u64;
            while offset_bytes < segment.size_bytes {
                let size_bytes = max_chunk_bytes.min(segment.size_bytes - offset_bytes);
                let total_complex = u32::try_from(size_bytes / COMPLEX_F32_BYTES)
                    .map_err(|_| FftError::LengthTooLarge { len: usize::MAX })?;
                let params = ScaleParams {
                    total_complex,
                    base_element: 0,
                    scale,
                    _pad: 0,
                };
                dispatches.push(ScaleDispatch {
                    segment_index,
                    offset_bytes,
                    size_bytes,
                    params_buffer: create_uniform_buffer(
                        device,
                        "wgpu_fft.segmented_volume.scale.params",
                        bytemuck::bytes_of(&params),
                    ),
                });
                offset_bytes = offset_bytes
                    .checked_add(size_bytes)
                    .ok_or(FftError::LengthTooLarge { len: usize::MAX })?;
            }
        }
        Ok(Self {
            max_chunk_bytes,
            dispatches,
        })
    }

    fn execute(
        &self,
        device: &wgpu::Device,
        encoder: &mut CommandRecorder<'_>,
        scheduler: &WindowScheduler,
        arena: &SegmentedArena,
        bind_group_layout: &wgpu::BindGroupLayout,
        pipeline: &wgpu::ComputePipeline,
    ) -> Result<()> {
        for dispatch in &self.dispatches {
            let view = BufferView::new(
                &arena.buffers[dispatch.segment_index],
                dispatch.offset_bytes,
                dispatch.size_bytes,
            )?;
            let resource = scheduler.storage_binding_resource(&view, ElementFormat::ComplexF32)?;
            let params_buffer = &dispatch.params_buffer;
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("wgpu_fft.segmented_volume.scale.bind_group"),
                layout: bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource,
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: params_buffer.as_entire_binding(),
                    },
                ],
            });
            let pass = encoder.pass();
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let workgroups = u32::try_from(dispatch.size_bytes / COMPLEX_F32_BYTES)
                .map_err(|_| FftError::LengthTooLarge { len: usize::MAX })?
                .div_ceil(SCALE_WORKGROUP_SIZE);
            let (x, y, z) = split_workgroups(workgroups, max_workgroups_per_dimension(device))?;
            pass.dispatch_workgroups(x, y, z);
        }
        Ok(())
    }
}

fn segmented_scale_chunk_bytes(max_segment_bytes: u64, limits: SchedulerLimits) -> Result<u64> {
    let binding_alignment = lcm(limits.storage_alignment.max(1), COMPLEX_F32_BYTES)?;
    let bind_bytes = align_down(
        limits
            .max_storage_buffer_binding_size
            .min(limits.max_buffer_size),
        COMPLEX_F32_BYTES,
    );
    let aligned_chunk_bytes = align_down(bind_bytes, binding_alignment);
    let max_chunk_bytes = if aligned_chunk_bytes >= COMPLEX_F32_BYTES {
        aligned_chunk_bytes
    } else if max_segment_bytes <= bind_bytes {
        // Binding alignment applies to offsets, not sizes. When every
        // physical segment fits one binding, its only chunk starts at the
        // naturally aligned offset zero even if the cap is smaller than
        // the device's dynamic-offset alignment.
        bind_bytes
    } else {
        0
    };
    if max_chunk_bytes < COMPLEX_F32_BYTES {
        return Err(FftError::WindowScheduleUnsupported {
            reason: "segmented scale cannot cover the arena with aligned storage windows",
            requested_bytes: COMPLEX_F32_BYTES,
            max_bind_bytes: limits.max_storage_buffer_binding_size,
        });
    }
    Ok(max_chunk_bytes)
}

#[allow(clippy::too_many_arguments)]
fn build_front_row_plan(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    config: &FftConfig,
    axis: usize,
    axis_len: usize,
    lines_total: usize,
    bind_stage_bytes: u64,
    row_plans: &mut Vec<AxisPlan>,
    row_plan_indices: &mut HashMap<(usize, usize), usize>,
    twiddle_lut_pool: &mut AxisTwiddleLutPool,
) -> Result<FrontRowBurstPlan> {
    let line_bytes = (axis_len as u64)
        .checked_mul(COMPLEX_F32_BYTES)
        .ok_or(FftError::LengthTooLarge { len: usize::MAX })?;
    let max_lines = usize::try_from(bind_stage_bytes / line_bytes)
        .map_err(|_| FftError::LengthTooLarge { len: usize::MAX })?;
    if max_lines == 0 {
        return Err(FftError::WindowScheduleUnsupported {
            reason: "segmented front-axis row does not fit one staging window",
            requested_bytes: line_bytes,
            max_bind_bytes: bind_stage_bytes,
        });
    }

    let lines_per_window = max_lines.min(lines_total);
    let mut windows = Vec::new();
    let mut start_line = 0usize;
    while start_line < lines_total {
        let line_count = lines_per_window.min(lines_total - start_line);
        let byte_offset = (start_line as u64)
            .checked_mul(line_bytes)
            .ok_or(FftError::LengthTooLarge { len: usize::MAX })?;
        let byte_size = (line_count as u64)
            .checked_mul(line_bytes)
            .ok_or(FftError::LengthTooLarge { len: usize::MAX })?;
        let plan_index = row_axis_plan_index(
            device,
            queue,
            config,
            axis_len,
            line_count,
            row_plans,
            row_plan_indices,
            twiddle_lut_pool,
        )?;
        windows.push(FrontRowWindow {
            byte_offset,
            byte_size,
            plan_index,
        });
        start_line += line_count;
    }
    let max_window_bytes = windows
        .iter()
        .map(|window| window.byte_size)
        .max()
        .ok_or(FftError::ZeroLength)?;
    Ok(FrontRowBurstPlan {
        axis,
        max_window_bytes,
        windows,
    })
}

#[allow(clippy::too_many_arguments)]
fn build_slab_axis_plan(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    config: &FftConfig,
    axis: usize,
    bind_stage_bytes: u64,
    row_plans: &mut Vec<AxisPlan>,
    row_plan_indices: &mut HashMap<(usize, usize), usize>,
    twiddle_lut_pool: &mut AxisTwiddleLutPool,
) -> Result<SlabAxisPlan> {
    let axis_len = config.shape()[axis];
    let prefix = checked_product(&config.shape()[..axis])?;
    let repetitions = checked_product(&config.shape()[axis + 1..])?
        .checked_mul(config.batch())
        .ok_or(FftError::LengthTooLarge { len: usize::MAX })?;
    let line_bytes = (axis_len as u64)
        .checked_mul(COMPLEX_F32_BYTES)
        .ok_or(FftError::LengthTooLarge { len: usize::MAX })?;
    let max_prefix = usize::try_from(bind_stage_bytes / line_bytes)
        .map_err(|_| FftError::LengthTooLarge { len: usize::MAX })?;
    if max_prefix == 0 {
        return Err(FftError::WindowScheduleUnsupported {
            reason: "segmented non-front axis does not fit one slab line",
            requested_bytes: line_bytes,
            max_bind_bytes: bind_stage_bytes,
        });
    }
    let prefix_chunk = max_prefix.min(prefix);
    let tail = prefix % prefix_chunk;
    let mut counts = vec![prefix_chunk];
    if tail > 0 && tail != prefix_chunk {
        counts.push(tail);
    }
    let mut variants = Vec::with_capacity(counts.len());
    for prefix_count in counts {
        let plan_index = row_axis_plan_index(
            device,
            queue,
            config,
            axis_len,
            prefix_count,
            row_plans,
            row_plan_indices,
            twiddle_lut_pool,
        )?;
        variants.push(SlabVariant {
            prefix_count,
            plan_index,
            to_front_params_buffer: create_transpose_params_buffer(device, prefix_count, axis_len)?,
            from_front_params_buffer: create_transpose_params_buffer(
                device,
                axis_len,
                prefix_count,
            )?,
        });
    }
    let max_slab_bytes = (prefix_chunk as u64)
        .checked_mul(line_bytes)
        .ok_or(FftError::LengthTooLarge { len: usize::MAX })?;
    let chunks_per_repetition = prefix.div_ceil(prefix_chunk);
    let dispatch_count = repetitions
        .checked_mul(chunks_per_repetition)
        .ok_or(FftError::LengthTooLarge { len: usize::MAX })?;
    Ok(SlabAxisPlan {
        axis,
        axis_len,
        prefix,
        prefix_chunk,
        chunks_per_repetition,
        dispatch_count,
        max_slab_bytes,
        variants,
    })
}

#[allow(clippy::too_many_arguments)]
fn row_axis_plan_index(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    config: &FftConfig,
    axis_len: usize,
    line_count: usize,
    row_plans: &mut Vec<AxisPlan>,
    row_plan_indices: &mut HashMap<(usize, usize), usize>,
    twiddle_lut_pool: &mut AxisTwiddleLutPool,
) -> Result<usize> {
    if let Some(&index) = row_plan_indices.get(&(axis_len, line_count)) {
        return Ok(index);
    }
    let index = row_plans.len();
    row_plans.push(AxisPlan::new_with_twiddle_lut_pool(
        device,
        queue,
        AxisPlanConfig {
            shape: vec![axis_len],
            axes: vec![0],
            batch: line_count,
            direction: config.direction(),
            normalization: Normalization::None,
            scale_override_bits: Some(1.0f32.to_bits()),
            layout: AxisLayout::Interleaved,
            precision: AxisPrecision::F32,
            workgroup_size: config.tuning().workgroup_size(),
            fused_workgroup_size: config.tuning().fused_workgroup_size(),
        },
        twiddle_lut_pool,
    )?);
    row_plan_indices.insert((axis_len, line_count), index);
    Ok(index)
}

#[allow(clippy::too_many_arguments)]
fn dispatch_transpose(
    device: &wgpu::Device,
    encoder: &mut CommandRecorder<'_>,
    scheduler: &WindowScheduler,
    input: &wgpu::Buffer,
    output: &wgpu::Buffer,
    size_bytes: u64,
    params_buffer: &wgpu::Buffer,
    width: usize,
    height: usize,
    bind_group_layout: &wgpu::BindGroupLayout,
    pipeline: &wgpu::ComputePipeline,
) -> Result<()> {
    let input_view = BufferView::whole(input).prefix(size_bytes)?;
    let output_view = BufferView::whole(output).prefix(size_bytes)?;
    let input_resource =
        scheduler.storage_binding_resource(&input_view, ElementFormat::ComplexF32)?;
    let output_resource =
        scheduler.storage_binding_resource(&output_view, ElementFormat::ComplexF32)?;
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("wgpu_fft.segmented_volume.transpose.bind_group"),
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
    let tiles_x = u32::try_from(width)
        .map_err(|_| FftError::LengthTooLarge { len: width })?
        .div_ceil(TRANSPOSE_TILE);
    let tiles_y = u32::try_from(height)
        .map_err(|_| FftError::LengthTooLarge { len: height })?
        .div_ceil(TRANSPOSE_TILE);
    let workgroups =
        tiles_x
            .checked_mul(tiles_y)
            .ok_or(FftError::DispatchWorkgroupsUnsupported {
                workgroups: u32::MAX,
                max_per_dimension: max_workgroups_per_dimension(device),
            })?;
    let (x, y, z) = split_workgroups(workgroups, max_workgroups_per_dimension(device))?;
    let pass = encoder.pass();
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, &bind_group, &[]);
    pass.dispatch_workgroups(x, y, z);
    Ok(())
}

fn create_transpose_params_buffer(
    device: &wgpu::Device,
    width: usize,
    height: usize,
) -> Result<wgpu::Buffer> {
    let params = TransposeParams {
        width: u32::try_from(width).map_err(|_| FftError::LengthTooLarge { len: width })?,
        height: u32::try_from(height).map_err(|_| FftError::LengthTooLarge { len: height })?,
        _pad0: 0,
        _pad1: 0,
    };
    Ok(create_uniform_buffer(
        device,
        "wgpu_fft.segmented_volume.transpose.params",
        bytemuck::bytes_of(&params),
    ))
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

fn create_segmented_buffer(
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

fn validate_whole_buffer_endpoint(view: &BufferView<'_>) -> Result<()> {
    if view.covers_whole_buffers() {
        Ok(())
    } else {
        Err(FftError::LargeRouteLayoutUnsupported {
            route_mode: "large-out-of-core",
            layout: "segmented full-volume execution requires zero-offset whole-buffer endpoints",
        })
    }
}

fn validate_endpoint_usage(
    view: &BufferView<'_>,
    required: wgpu::BufferUsages,
    usage: &'static str,
) -> Result<()> {
    if view
        .segments()
        .iter()
        .all(|segment| segment.buffer.usage().contains(required))
    {
        Ok(())
    } else {
        Err(FftError::BufferViewMissingUsage { usage })
    }
}

#[allow(clippy::too_many_arguments)]
fn build_diagnostic_graph(
    arena: &SegmentedArena,
    schedule: &SegmentedVolumeSchedule,
    row_plans: &[AxisPlan],
    stage_bytes: u64,
    burst_depth: usize,
    total_complex: u64,
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
    let mut graph = LargeExecutionGraph::new("c2c-segmented-full-volume");
    let mut helper_index = 0u32;
    for segment in &arena.segments {
        graph.push_stage(
            LargeStage::WindowedHelper {
                label: "segmented-volume-arena",
                range: LogicalRange::new(
                    LogicalBufferId::Stage(helper_index),
                    0,
                    segment.size_bytes,
                    ElementFormat::ComplexF32,
                )?,
            },
            requirements(segment.size_bytes)?,
        )?;
        helper_index = helper_index
            .checked_add(1)
            .ok_or(FftError::LengthTooLarge { len: usize::MAX })?;
    }
    // Logical traffic stages use the first physical ring pair. The remaining
    // pairs are parallel inventory, not additional traffic-equivalent passes.
    let stage_a_index = helper_index;
    let stage_b_index = helper_index
        .checked_add(1)
        .ok_or(FftError::LengthTooLarge { len: usize::MAX })?;
    for _ in 0..burst_depth {
        for label in [
            "segmented-volume-burst-stage-a",
            "segmented-volume-burst-stage-b",
        ] {
            graph.push_stage(
                LargeStage::WindowedHelper {
                    label,
                    range: LogicalRange::new(
                        LogicalBufferId::Stage(helper_index),
                        0,
                        stage_bytes,
                        ElementFormat::ComplexF32,
                    )?,
                },
                requirements(stage_bytes)?,
            )?;
            helper_index = helper_index
                .checked_add(1)
                .ok_or(FftError::LengthTooLarge { len: usize::MAX })?;
        }
    }
    for plan in row_plans {
        let workspace_bytes = plan.workspace_size_bytes();
        if workspace_bytes == 0 {
            continue;
        }
        graph.push_stage(
            LargeStage::WindowedHelper {
                label: "segmented-volume-axis-workspace",
                range: LogicalRange::new(
                    LogicalBufferId::Stage(helper_index),
                    0,
                    workspace_bytes,
                    ElementFormat::ComplexF32,
                )?,
            },
            requirements(workspace_bytes)?,
        )?;
        helper_index = helper_index
            .checked_add(1)
            .ok_or(FftError::LengthTooLarge { len: usize::MAX })?;
    }

    let arena_window_bytes = arena
        .segments
        .iter()
        .map(|segment| segment.size_bytes)
        .max()
        .ok_or(FftError::ZeroLength)?;
    graph.push_stage(
        LargeStage::Copy {
            label: "segmented-volume-upload",
            src: diagnostic_range(LogicalBufferId::Input, arena_window_bytes)?,
            dst: diagnostic_range(LogicalBufferId::Stage(0), arena_window_bytes)?,
        },
        requirements(0)?,
    )?;

    for stage in &schedule.stages {
        match stage {
            SegmentedVolumeStage::FrontRows(plan) => {
                let bytes = plan.max_window_bytes;
                push_diagnostic_copy(
                    &mut graph,
                    "segmented-volume-axis0-gather",
                    LogicalBufferId::Temp(0),
                    LogicalBufferId::Stage(stage_a_index),
                    bytes,
                    requirements(bytes)?,
                )?;
                let child = &row_plans[plan.windows[0].plan_index];
                push_diagnostic_child_stages(
                    &mut graph,
                    child.graph_stage_kinds().len(),
                    "segmented-volume-axis0-fft",
                    stage_a_index,
                    stage_b_index,
                    bytes,
                    total_complex,
                    requirements(bytes)?,
                )?;
                push_diagnostic_copy(
                    &mut graph,
                    "segmented-volume-axis0-scatter",
                    LogicalBufferId::Stage(stage_b_index),
                    LogicalBufferId::Temp(0),
                    bytes,
                    requirements(bytes)?,
                )?;
            }
            SegmentedVolumeStage::SlabAxis(plan) => {
                let bytes = plan.max_slab_bytes;
                graph.push_stage(
                    LargeStage::GatherScatter {
                        label: "segmented-volume-axis-slab-gather",
                        src: diagnostic_range(LogicalBufferId::Temp(0), bytes)?,
                        dst: diagnostic_range(LogicalBufferId::Stage(stage_a_index), bytes)?,
                        stride_elements: total_complex,
                    },
                    requirements(bytes)?,
                )?;
                push_diagnostic_transpose(
                    &mut graph,
                    "segmented-volume-axis-slab-transpose-to-front",
                    stage_a_index,
                    stage_b_index,
                    bytes,
                    total_complex,
                    requirements(bytes)?,
                )?;
                let child = &row_plans[plan.variants[0].plan_index];
                push_diagnostic_child_stages(
                    &mut graph,
                    child.graph_stage_kinds().len(),
                    "segmented-volume-axis-slab-fft",
                    stage_b_index,
                    stage_a_index,
                    bytes,
                    total_complex,
                    requirements(bytes)?,
                )?;
                push_diagnostic_transpose(
                    &mut graph,
                    "segmented-volume-axis-slab-transpose-back",
                    stage_a_index,
                    stage_b_index,
                    bytes,
                    total_complex,
                    requirements(bytes)?,
                )?;
                graph.push_stage(
                    LargeStage::GatherScatter {
                        label: "segmented-volume-axis-slab-scatter",
                        src: diagnostic_range(LogicalBufferId::Stage(stage_b_index), bytes)?,
                        dst: diagnostic_range(LogicalBufferId::Temp(0), bytes)?,
                        stride_elements: total_complex,
                    },
                    requirements(bytes)?,
                )?;
            }
            SegmentedVolumeStage::Scale(plan) => {
                graph.push_stage(
                    LargeStage::Scale {
                        label: "segmented-volume-scale",
                        range: diagnostic_range(LogicalBufferId::Temp(0), plan.max_chunk_bytes)?,
                        work_items: total_complex,
                    },
                    requirements(plan.max_chunk_bytes)?,
                )?;
            }
            SegmentedVolumeStage::Upload | SegmentedVolumeStage::Download => {}
        }
    }

    graph.push_stage(
        LargeStage::Copy {
            label: "segmented-volume-download",
            src: diagnostic_range(LogicalBufferId::Stage(0), arena_window_bytes)?,
            dst: diagnostic_range(LogicalBufferId::Output, arena_window_bytes)?,
        },
        requirements(0)?,
    )?;
    Ok(LargeExecutionPlan::new(graph))
}

fn diagnostic_range(buffer: LogicalBufferId, bytes: u64) -> Result<LogicalRange> {
    LogicalRange::new(buffer, 0, bytes, ElementFormat::ComplexF32)
}

fn push_diagnostic_copy(
    graph: &mut LargeExecutionGraph,
    label: &'static str,
    src: LogicalBufferId,
    dst: LogicalBufferId,
    bytes: u64,
    requirements: StageRequirements,
) -> Result<()> {
    graph.push_stage(
        LargeStage::Copy {
            label,
            src: diagnostic_range(src, bytes)?,
            dst: diagnostic_range(dst, bytes)?,
        },
        requirements,
    )
}

#[allow(clippy::too_many_arguments)]
fn push_diagnostic_child_stages(
    graph: &mut LargeExecutionGraph,
    count: usize,
    label: &'static str,
    input_index: u32,
    output_index: u32,
    bytes: u64,
    work_items: u64,
    requirements: StageRequirements,
) -> Result<()> {
    for _ in 0..count {
        graph.push_stage(
            LargeStage::WindowedKernel {
                label,
                input: diagnostic_range(LogicalBufferId::Stage(input_index), bytes)?,
                output: diagnostic_range(LogicalBufferId::Stage(output_index), bytes)?,
                work_items,
            },
            requirements,
        )?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn push_diagnostic_transpose(
    graph: &mut LargeExecutionGraph,
    label: &'static str,
    input_index: u32,
    output_index: u32,
    bytes: u64,
    work_items: u64,
    requirements: StageRequirements,
) -> Result<()> {
    graph.push_stage(
        LargeStage::StripeTranspose {
            label,
            input: diagnostic_range(LogicalBufferId::Stage(input_index), bytes)?,
            output: diagnostic_range(LogicalBufferId::Stage(output_index), bytes)?,
            work_items,
        },
        requirements,
    )
}

fn checked_product(values: &[usize]) -> Result<usize> {
    values.iter().try_fold(1usize, |product, &value| {
        product
            .checked_mul(value)
            .ok_or(FftError::LengthTooLarge { len: usize::MAX })
    })
}

fn align_down(value: u64, alignment: u64) -> u64 {
    value - value % alignment.max(1)
}

fn lcm(a: u64, b: u64) -> Result<u64> {
    let gcd = gcd(a, b);
    (a / gcd)
        .checked_mul(b)
        .ok_or(FftError::LengthTooLarge { len: usize::MAX })
}

fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let next = a % b;
        a = b;
        b = next;
    }
    a.max(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slab_indexing_restores_dimension_zero_fast_layout() {
        let prefix = 5u64;
        let axis_len = 7u64;
        let prefix_start = 2u64;
        let prefix_count = 3u64;
        for k in 0..axis_len {
            for p in 0..prefix_count {
                let logical = k * prefix + prefix_start + p;
                let slab_a = k * prefix_count + p;
                let slab_b = p * axis_len + k;
                assert_eq!(slab_a / prefix_count, k);
                assert_eq!(slab_b / axis_len, p);
                assert_eq!(logical, k * prefix + prefix_start + slab_b / axis_len);
            }
        }
    }

    #[test]
    fn exact_segment_capacity_preserves_forced_limit_layout() {
        let total = 102_400u64;
        let cap = align_down(32_768, COMPLEX_F32_BYTES);
        let mut sizes = Vec::new();
        let mut offset = 0u64;
        while offset < total {
            let size = cap.min(total - offset);
            sizes.push(size);
            offset += size;
        }
        assert_eq!(sizes, [32_768, 32_768, 32_768, 4_096]);
    }

    #[test]
    fn scale_uses_one_offset_zero_binding_when_cap_is_below_offset_alignment() {
        let limits = SchedulerLimits {
            max_storage_buffer_binding_size: 128,
            max_buffer_size: 128,
            storage_alignment: 256,
            copy_alignment: 4,
        };
        assert_eq!(segmented_scale_chunk_bytes(128, limits).unwrap(), 128);
    }

    #[test]
    fn burst_depth_accepts_only_the_supported_ring_range() {
        for depth in 1..=3 {
            validate_segmented_burst_depth(depth).unwrap();
        }
        for depth in [0, 4, usize::MAX] {
            assert!(matches!(
                validate_segmented_burst_depth(depth),
                Err(FftError::LargeGraphStageUnsupported {
                    stage: "segmented-volume-burst-ring",
                    ..
                })
            ));
        }
    }

    #[test]
    fn slab_dispatch_geometry_preserves_tail_and_repetition_boundaries() {
        let axis_len = 7usize;
        let prefix = 10usize;
        let prefix_chunk = 4usize;
        let chunks_per_repetition = prefix.div_ceil(prefix_chunk);
        let expected = [
            (0, 0, 4, 224),
            (0, 4, 4, 224),
            (0, 8, 2, 112),
            (70, 0, 4, 224),
            (70, 4, 4, 224),
            (70, 8, 2, 112),
        ];
        for (index, expected) in expected.into_iter().enumerate() {
            assert_eq!(
                slab_dispatch_geometry(
                    index,
                    axis_len,
                    prefix,
                    prefix_chunk,
                    chunks_per_repetition,
                )
                .unwrap(),
                expected
            );
        }
    }
}
