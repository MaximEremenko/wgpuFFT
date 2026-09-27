use crate::diagnostics::{stage_route_for_label, FftBlocker, FftBlockerKind};
use crate::error::{FftError, Result};
use crate::runtime::buffer_view::{BufferRange, BufferView};
use crate::runtime::large_graph::{LargeExecutionGraph, LargeStage, LogicalRange};
use crate::runtime::recorder::CommandRecorder;
use crate::runtime::window_scheduler::{SchedulerLimits, WindowScheduler};

pub(crate) struct StageExecutor<'a> {
    scheduler: &'a WindowScheduler,
}

impl<'a> StageExecutor<'a> {
    pub(crate) fn new(scheduler: &'a WindowScheduler) -> Self {
        Self { scheduler }
    }

    pub(crate) fn validate_graph(&self, graph: &LargeExecutionGraph) -> Result<()> {
        for stage in graph.stages() {
            if stage.label().is_empty() {
                return Err(FftError::LargeGraphStageUnsupported {
                    stage: "unknown",
                    reason: "stage label must not be empty",
                });
            }
            if matches!(stage.work_items(), Some(0)) {
                return Err(FftError::LargeGraphStageUnsupported {
                    stage: stage.label(),
                    reason: "stage work item count must be non-zero",
                });
            }
            self.validate_stage_ranges(stage)?;
        }
        Ok(())
    }

    pub(crate) fn graph_blockers(
        &self,
        route: impl Into<String>,
        graph: &LargeExecutionGraph,
    ) -> Vec<FftBlocker> {
        let route = route.into();
        let limits = self.scheduler.limits();
        let mut blockers = Vec::new();
        for stage in graph.stages() {
            let stage_route = stage_route_for_label(stage.label(), &route);
            blockers.extend(stage_blockers(stage, &stage_route, limits));
        }
        blockers
    }

    fn validate_stage_ranges(&self, stage: &LargeStage) -> Result<()> {
        let limits = self.scheduler.limits();
        for range in stage.ranges() {
            let _ = (range.buffer.kind_label(), range.buffer.index());
            if range.element_count() == 0 {
                return Err(FftError::LargeGraphStageUnsupported {
                    stage: stage.label(),
                    reason: "stage logical range must not be empty",
                });
            }
            if range.size_bytes > limits.max_buffer_size {
                return Err(FftError::WindowScheduleUnsupported {
                    reason: "stage logical range exceeds active max buffer size",
                    requested_bytes: range.size_bytes,
                    max_bind_bytes: limits.max_buffer_size,
                });
            }
        }
        for range in stage.storage_ranges() {
            validate_storage_range(
                stage.label(),
                range,
                limits.storage_alignment,
                limits.max_storage_buffer_binding_size,
            )?;
        }
        for range in stage.copy_ranges() {
            if range.offset_bytes % limits.copy_alignment != 0
                || range.size_bytes % limits.copy_alignment != 0
            {
                return Err(FftError::BufferViewCopyUnaligned {
                    offset: range.offset_bytes,
                    size: range.size_bytes,
                    alignment: limits.copy_alignment,
                });
            }
        }
        Ok(())
    }

    pub(crate) fn copy_view_range_to_buffer(
        &self,
        encoder: &mut CommandRecorder<'_>,
        view: &BufferView<'_>,
        view_offset: u64,
        dst: &wgpu::Buffer,
        dst_offset: u64,
        size: u64,
    ) -> Result<()> {
        let ranges = self
            .scheduler
            .validate_copy_ranges(view, view_offset, size)?;
        validate_copy_usage(&ranges, wgpu::BufferUsages::COPY_SRC, "COPY_SRC")?;
        let mut cursor = dst_offset;
        for range in ranges {
            encoder.encoder().copy_buffer_to_buffer(
                range.buffer,
                range.offset_bytes,
                dst,
                cursor,
                range.size_bytes,
            );
            cursor += range.size_bytes;
        }
        Ok(())
    }

    pub(crate) fn copy_buffer_to_view_range(
        &self,
        encoder: &mut CommandRecorder<'_>,
        src: &wgpu::Buffer,
        src_offset: u64,
        view: &BufferView<'_>,
        view_offset: u64,
        size: u64,
    ) -> Result<()> {
        let ranges = self
            .scheduler
            .validate_copy_ranges(view, view_offset, size)?;
        validate_copy_usage(&ranges, wgpu::BufferUsages::COPY_DST, "COPY_DST")?;
        let mut cursor = src_offset;
        for range in ranges {
            encoder.encoder().copy_buffer_to_buffer(
                src,
                cursor,
                range.buffer,
                range.offset_bytes,
                range.size_bytes,
            );
            cursor += range.size_bytes;
        }
        Ok(())
    }
}

fn stage_blockers(stage: &LargeStage, route: &str, limits: SchedulerLimits) -> Vec<FftBlocker> {
    let mut blockers = Vec::new();
    if stage.label().is_empty() {
        blockers.push(
            FftBlocker::new(FftBlockerKind::Unsupported, "stage label must not be empty")
                .with_route(route.to_owned())
                .with_stage("unknown")
                .with_layout("stage graph"),
        );
    }
    if matches!(stage.work_items(), Some(0)) {
        blockers.push(
            FftBlocker::new(
                FftBlockerKind::Unsupported,
                "stage work item count must be non-zero",
            )
            .with_route(route.to_owned())
            .with_stage(stage.label())
            .with_layout("stage graph")
            .with_required_bytes(1)
            .with_actual_bytes(0),
        );
    }
    for range in stage.ranges() {
        let _ = (range.buffer.kind_label(), range.buffer.index());
        if range.element_count() == 0 {
            blockers.push(with_stage_helper_context(
                FftBlocker::new(
                    FftBlockerKind::Unsupported,
                    "stage logical range must not be empty",
                )
                .with_route(route.to_owned())
                .with_stage(stage.label())
                .with_layout(stage_range_layout(stage))
                .with_required_bytes(1)
                .with_actual_bytes(0),
                stage,
            ));
        }
        if range.size_bytes > limits.max_buffer_size {
            blockers.push(with_stage_helper_context(
                FftBlocker::new(
                    FftBlockerKind::DeviceLimit,
                    "stage logical range exceeds active max buffer size",
                )
                .with_route(route.to_owned())
                .with_stage(stage.label())
                .with_layout(stage_range_layout(stage))
                .with_required_bytes(range.size_bytes)
                .with_limit_bytes(limits.max_buffer_size),
                stage,
            ));
        }
    }
    for range in stage.storage_ranges() {
        if let Some(binding_bytes) = storage_binding_bytes(range, limits.storage_alignment.max(1)) {
            if binding_bytes > limits.max_storage_buffer_binding_size {
                blockers.push(with_stage_helper_context(
                    FftBlocker::new(
                        FftBlockerKind::DeviceLimit,
                        "stage storage binding window exceeds active device limits",
                    )
                    .with_route(route.to_owned())
                    .with_stage(stage.label())
                    .with_layout("storage binding")
                    .with_required_bytes(binding_bytes)
                    .with_limit_bytes(limits.max_storage_buffer_binding_size),
                    stage,
                ));
            }
        } else {
            blockers.push(with_stage_helper_context(
                FftBlocker::new(
                    FftBlockerKind::Validation,
                    "stage storage binding window size overflowed",
                )
                .with_route(route.to_owned())
                .with_stage(stage.label())
                .with_layout("storage binding"),
                stage,
            ));
        }
    }
    for range in stage.copy_ranges() {
        if range.offset_bytes % limits.copy_alignment != 0
            || range.size_bytes % limits.copy_alignment != 0
        {
            blockers.push(
                FftBlocker::new(
                    FftBlockerKind::Alignment,
                    "stage copy range is not aligned to active copy alignment",
                )
                .with_route(route.to_owned())
                .with_stage(stage.label())
                .with_layout("copy range")
                .with_required_bytes(limits.copy_alignment)
                .with_actual_bytes(range.offset_bytes.saturating_add(range.size_bytes)),
            );
        }
    }
    blockers
}

fn with_stage_helper_context(blocker: FftBlocker, stage: &LargeStage) -> FftBlocker {
    match stage {
        LargeStage::HelperWindow { label, .. } | LargeStage::WindowedHelper { label, .. } => {
            blocker.with_helper_buffer(*label)
        }
        _ => blocker,
    }
}

fn stage_range_layout(stage: &LargeStage) -> &'static str {
    match stage {
        LargeStage::Copy { .. } => "copy range",
        LargeStage::GatherScatter { .. } => "gather/scatter range",
        LargeStage::HelperWindow { .. } | LargeStage::WindowedHelper { .. } => "helper buffer",
        LargeStage::HostWindow { .. } => "host window",
        LargeStage::Kernel { .. }
        | LargeStage::WindowedKernel { .. }
        | LargeStage::TwiddleTranspose { .. }
        | LargeStage::StripeTranspose { .. }
        | LargeStage::Permutation { .. }
        | LargeStage::Scale { .. } => "logical range",
    }
}

fn storage_binding_bytes(range: LogicalRange, storage_alignment: u64) -> Option<u64> {
    let leading_bytes = range.offset_bytes % storage_alignment.max(1);
    leading_bytes.checked_add(range.size_bytes)
}

fn validate_storage_range(
    _stage: &'static str,
    range: LogicalRange,
    storage_alignment: u64,
    max_binding_bytes: u64,
) -> Result<()> {
    let binding_bytes = storage_binding_bytes(range, storage_alignment)
        .ok_or(FftError::LengthTooLarge { len: usize::MAX })?;
    if binding_bytes > max_binding_bytes {
        return Err(FftError::WindowScheduleUnsupported {
            reason: "stage storage binding window exceeds active device limits",
            requested_bytes: binding_bytes,
            max_bind_bytes: max_binding_bytes,
        });
    }
    Ok(())
}

fn validate_copy_usage(
    ranges: &[BufferRange<'_>],
    required: wgpu::BufferUsages,
    usage_name: &'static str,
) -> Result<()> {
    if ranges
        .iter()
        .all(|range| range.buffer.usage().contains(required))
    {
        Ok(())
    } else {
        Err(FftError::BufferViewMissingUsage { usage: usage_name })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::large_graph::{
        ElementFormat, LargeExecutionGraph, LargeStage, LogicalBufferId, LogicalRange,
        StageRequirements,
    };
    use crate::runtime::window_scheduler::SchedulerLimits;

    fn scheduler(max_storage_buffer_binding_size: u64) -> WindowScheduler {
        WindowScheduler::new(SchedulerLimits {
            max_storage_buffer_binding_size,
            max_buffer_size: 1024,
            storage_alignment: 256,
            copy_alignment: 4,
        })
    }

    fn requirements(max_binding_bytes: u64) -> StageRequirements {
        StageRequirements::new(max_binding_bytes, 1024, 256, 4, 0).unwrap()
    }

    #[test]
    fn validates_empty_graph() {
        let scheduler = scheduler(256);
        let executor = StageExecutor::new(&scheduler);
        let graph = LargeExecutionGraph::new("empty");
        executor.validate_graph(&graph).unwrap();
    }

    #[test]
    fn rejects_stage_that_exceeds_active_scheduler_limits() {
        let build_requirements = requirements(256);
        let scheduler = scheduler(64);
        let executor = StageExecutor::new(&scheduler);
        let mut graph = LargeExecutionGraph::new("active-limit");
        let input =
            LogicalRange::new(LogicalBufferId::Input, 0, 128, ElementFormat::ComplexF32).unwrap();
        let output =
            LogicalRange::new(LogicalBufferId::Output, 0, 128, ElementFormat::ComplexF32).unwrap();
        graph
            .push_stage(
                LargeStage::Kernel {
                    label: "kernel",
                    input,
                    output,
                    work_items: 16,
                },
                build_requirements,
            )
            .unwrap();

        assert!(matches!(
            executor.validate_graph(&graph),
            Err(FftError::WindowScheduleUnsupported { .. })
        ));
        let blockers = executor.graph_blockers("c2c-test", &graph);
        assert_eq!(blockers.len(), 2);
        assert!(blockers.iter().all(|blocker| {
            blocker.kind == FftBlockerKind::DeviceLimit
                && blocker.route.as_deref() == Some("c2c-test")
                && blocker.stage.as_deref() == Some("kernel")
                && blocker.layout.as_deref() == Some("storage binding")
                && blocker.required_bytes == Some(128)
                && blocker.limit_bytes == Some(64)
        }));
    }

    #[test]
    fn helper_window_blockers_identify_helper_buffer() {
        let build_requirements = requirements(256);
        let scheduler = scheduler(64);
        let executor = StageExecutor::new(&scheduler);
        let mut graph = LargeExecutionGraph::new("helper-limit");
        let range =
            LogicalRange::new(LogicalBufferId::Temp(7), 0, 128, ElementFormat::ComplexF32).unwrap();
        graph
            .push_stage(
                LargeStage::HelperWindow {
                    label: "rader-helper-window",
                    range,
                },
                build_requirements,
            )
            .unwrap();

        let blockers = executor.graph_blockers("rader", &graph);
        assert_eq!(blockers.len(), 1);
        let blocker = &blockers[0];
        assert_eq!(blocker.kind, FftBlockerKind::DeviceLimit);
        assert_eq!(blocker.route.as_deref(), Some("rader"));
        assert_eq!(blocker.stage.as_deref(), Some("rader-helper-window"));
        assert_eq!(blocker.layout.as_deref(), Some("storage binding"));
        assert_eq!(
            blocker.helper_buffer.as_deref(),
            Some("rader-helper-window")
        );
        assert_eq!(blocker.required_bytes, Some(128));
        assert_eq!(blocker.limit_bytes, Some(64));
    }

    #[test]
    fn permutation_ranges_are_windowed_below_max_buffer() {
        let build_requirements = requirements(256);
        let scheduler = scheduler(64);
        let executor = StageExecutor::new(&scheduler);
        let mut graph = LargeExecutionGraph::new("permutation-window");
        let input =
            LogicalRange::new(LogicalBufferId::Input, 0, 128, ElementFormat::ComplexF32).unwrap();
        let output =
            LogicalRange::new(LogicalBufferId::Output, 0, 128, ElementFormat::ComplexF32).unwrap();
        graph
            .push_stage(
                LargeStage::Permutation {
                    label: "four-step-permute-axis-to-front",
                    input,
                    output,
                    work_items: 16,
                },
                build_requirements,
            )
            .unwrap();

        executor.validate_graph(&graph).unwrap();
        assert!(executor
            .graph_blockers("large-out-of-core", &graph)
            .is_empty());
    }

    #[test]
    fn graph_blockers_infer_embedded_stage_routes() {
        let build_requirements = requirements(256);
        let scheduler = WindowScheduler::new(SchedulerLimits {
            max_storage_buffer_binding_size: 64,
            max_buffer_size: 256,
            storage_alignment: 256,
            copy_alignment: 4,
        });
        let executor = StageExecutor::new(&scheduler);
        let mut graph = LargeExecutionGraph::new("embedded-routes");
        let rader_range =
            LogicalRange::new(LogicalBufferId::Temp(7), 0, 128, ElementFormat::ComplexF32).unwrap();
        let logical_range =
            LogicalRange::new(LogicalBufferId::Output, 0, 512, ElementFormat::ComplexF32).unwrap();
        graph
            .push_stage(
                LargeStage::HelperWindow {
                    label: "rader-work-helper",
                    range: rader_range,
                },
                build_requirements,
            )
            .unwrap();
        graph
            .push_stage(
                LargeStage::HostWindow {
                    label: "logical-output",
                    range: logical_range,
                },
                StageRequirements::new(4096, 1024, 256, 4, 0).unwrap(),
            )
            .unwrap();

        let blockers = executor.graph_blockers("axis-sequence", &graph);
        assert!(blockers.iter().any(|blocker| {
            blocker.kind == FftBlockerKind::DeviceLimit
                && blocker.route.as_deref() == Some("rader")
                && blocker.stage.as_deref() == Some("rader-work-helper")
                && blocker.helper_buffer.as_deref() == Some("rader-work-helper")
        }));
        assert!(blockers.iter().any(|blocker| {
            blocker.kind == FftBlockerKind::DeviceLimit
                && blocker.route.as_deref() == Some("axis-sequence")
                && blocker.stage.as_deref() == Some("logical-output")
                && blocker.layout.as_deref() == Some("host window")
        }));
    }

    #[test]
    fn rejects_empty_labeled_stage() {
        let scheduler = scheduler(256);
        let executor = StageExecutor::new(&scheduler);
        let mut graph = LargeExecutionGraph::new("bad-label");
        let input =
            LogicalRange::new(LogicalBufferId::Input, 0, 128, ElementFormat::ComplexF32).unwrap();
        let output =
            LogicalRange::new(LogicalBufferId::Output, 0, 128, ElementFormat::ComplexF32).unwrap();
        graph
            .push_stage(
                LargeStage::Copy {
                    label: "",
                    src: input,
                    dst: output,
                },
                requirements(256),
            )
            .unwrap();

        assert!(matches!(
            executor.validate_graph(&graph),
            Err(FftError::LargeGraphStageUnsupported { .. })
        ));
        let blockers = executor.graph_blockers("c2c-test", &graph);
        assert_eq!(blockers.len(), 1);
        let blocker = &blockers[0];
        assert_eq!(blocker.kind, FftBlockerKind::Unsupported);
        assert_eq!(blocker.route.as_deref(), Some("c2c-test"));
        assert_eq!(blocker.stage.as_deref(), Some("unknown"));
        assert_eq!(blocker.layout.as_deref(), Some("stage graph"));
    }

    #[test]
    fn rejects_active_copy_alignment_mismatch() {
        let scheduler = WindowScheduler::new(SchedulerLimits {
            max_storage_buffer_binding_size: 256,
            max_buffer_size: 1024,
            storage_alignment: 256,
            copy_alignment: 16,
        });
        let executor = StageExecutor::new(&scheduler);
        let mut graph = LargeExecutionGraph::new("copy-align");
        let src =
            LogicalRange::new(LogicalBufferId::Input, 0, 128, ElementFormat::ComplexF32).unwrap();
        let dst =
            LogicalRange::new(LogicalBufferId::Output, 8, 128, ElementFormat::ComplexF32).unwrap();
        graph
            .push_stage(
                LargeStage::Copy {
                    label: "copy",
                    src,
                    dst,
                },
                requirements(256),
            )
            .unwrap();

        assert!(matches!(
            executor.validate_graph(&graph),
            Err(FftError::BufferViewCopyUnaligned { .. })
        ));
    }

    #[test]
    fn rejects_active_copy_range_that_exceeds_max_buffer_size() {
        let scheduler = WindowScheduler::new(SchedulerLimits {
            max_storage_buffer_binding_size: 4096,
            max_buffer_size: 256,
            storage_alignment: 256,
            copy_alignment: 4,
        });
        let executor = StageExecutor::new(&scheduler);
        let mut graph = LargeExecutionGraph::new("copy-max-buffer");
        let src =
            LogicalRange::new(LogicalBufferId::Input, 0, 512, ElementFormat::ComplexF32).unwrap();
        let dst =
            LogicalRange::new(LogicalBufferId::Output, 0, 512, ElementFormat::ComplexF32).unwrap();
        graph
            .push_stage(
                LargeStage::Copy {
                    label: "copy",
                    src,
                    dst,
                },
                StageRequirements::new(4096, 1024, 256, 4, 0).unwrap(),
            )
            .unwrap();

        assert!(matches!(
            executor.validate_graph(&graph),
            Err(FftError::WindowScheduleUnsupported {
                reason: "stage logical range exceeds active max buffer size",
                requested_bytes: 512,
                max_bind_bytes: 256,
            })
        ));
        let blockers = executor.graph_blockers("c2c-test", &graph);
        assert_eq!(blockers.len(), 2);
        assert!(blockers.iter().all(|blocker| {
            blocker.kind == FftBlockerKind::DeviceLimit
                && blocker.route.as_deref() == Some("c2c-test")
                && blocker.stage.as_deref() == Some("copy")
                && blocker.layout.as_deref() == Some("copy range")
                && blocker.required_bytes == Some(512)
                && blocker.limit_bytes == Some(256)
        }));
    }

    #[test]
    fn host_window_blockers_identify_host_window_layout() {
        let scheduler = WindowScheduler::new(SchedulerLimits {
            max_storage_buffer_binding_size: 4096,
            max_buffer_size: 256,
            storage_alignment: 256,
            copy_alignment: 4,
        });
        let executor = StageExecutor::new(&scheduler);
        let mut graph = LargeExecutionGraph::new("host-window-max-buffer");
        let range = LogicalRange::new(LogicalBufferId::Stage(3), 0, 512, ElementFormat::ComplexF32)
            .unwrap();
        graph
            .push_stage(
                LargeStage::HostWindow {
                    label: "out-of-core-host-window",
                    range,
                },
                StageRequirements::new(4096, 1024, 256, 4, 0).unwrap(),
            )
            .unwrap();

        let blockers = executor.graph_blockers("large-out-of-core", &graph);
        assert_eq!(blockers.len(), 1);
        let blocker = &blockers[0];
        assert_eq!(blocker.kind, FftBlockerKind::DeviceLimit);
        assert_eq!(blocker.route.as_deref(), Some("large-out-of-core"));
        assert_eq!(blocker.stage.as_deref(), Some("out-of-core-host-window"));
        assert_eq!(blocker.layout.as_deref(), Some("host window"));
        assert_eq!(blocker.required_bytes, Some(512));
        assert_eq!(blocker.limit_bytes, Some(256));
    }
}
