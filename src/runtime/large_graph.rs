use crate::error::{FftError, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ElementFormat {
    ComplexF32,
    RealF32,
    PackedComplexF32,
    U32,
}

impl ElementFormat {
    pub(crate) const fn bytes_per_element(self) -> u64 {
        match self {
            Self::ComplexF32 | Self::PackedComplexF32 => 8,
            Self::RealF32 | Self::U32 => 4,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum LogicalBufferId {
    Input,
    Output,
    Temp(u32),
    Stage(u32),
}

impl LogicalBufferId {
    pub(crate) const fn kind_label(self) -> &'static str {
        match self {
            Self::Input => "input",
            Self::Output => "output",
            Self::Temp(_) => "temp",
            Self::Stage(_) => "stage",
        }
    }

    pub(crate) const fn index(self) -> Option<u32> {
        match self {
            Self::Input | Self::Output => None,
            Self::Temp(index) | Self::Stage(index) => Some(index),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct LogicalRange {
    pub(crate) buffer: LogicalBufferId,
    pub(crate) offset_bytes: u64,
    pub(crate) size_bytes: u64,
    pub(crate) format: ElementFormat,
}

impl LogicalRange {
    pub(crate) fn new(
        buffer: LogicalBufferId,
        offset_bytes: u64,
        size_bytes: u64,
        format: ElementFormat,
    ) -> Result<Self> {
        if offset_bytes % format.bytes_per_element() != 0
            || size_bytes % format.bytes_per_element() != 0
        {
            return Err(FftError::WindowScheduleUnsupported {
                reason: "logical range is not aligned to its element format",
                requested_bytes: size_bytes,
                max_bind_bytes: 0,
            });
        }
        Ok(Self {
            buffer,
            offset_bytes,
            size_bytes,
            format,
        })
    }

    pub(crate) fn element_count(self) -> u64 {
        self.size_bytes / self.format.bytes_per_element()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct HelperBufferRange {
    pub(crate) label: &'static str,
    pub(crate) index: u32,
    pub(crate) size_bytes: u64,
    pub(crate) format: ElementFormat,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LargeStageKind {
    Copy,
    GatherScatter,
    HelperWindow,
    WindowedHelper,
    Kernel,
    WindowedKernel,
    TwiddleTranspose,
    StripeTranspose,
    Scale,
    HostWindow,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum LargeStage {
    Copy {
        label: &'static str,
        src: LogicalRange,
        dst: LogicalRange,
    },
    GatherScatter {
        label: &'static str,
        src: LogicalRange,
        dst: LogicalRange,
        stride_elements: u64,
    },
    HelperWindow {
        label: &'static str,
        range: LogicalRange,
    },
    WindowedHelper {
        label: &'static str,
        range: LogicalRange,
    },
    Kernel {
        label: &'static str,
        input: LogicalRange,
        output: LogicalRange,
        work_items: u64,
    },
    WindowedKernel {
        label: &'static str,
        input: LogicalRange,
        output: LogicalRange,
        work_items: u64,
    },
    TwiddleTranspose {
        label: &'static str,
        input: LogicalRange,
        output: LogicalRange,
        work_items: u64,
    },
    StripeTranspose {
        label: &'static str,
        input: LogicalRange,
        output: LogicalRange,
        work_items: u64,
    },
    Scale {
        label: &'static str,
        range: LogicalRange,
        work_items: u64,
    },
    HostWindow {
        label: &'static str,
        range: LogicalRange,
    },
}

impl LargeStage {
    pub(crate) fn kind(&self) -> LargeStageKind {
        match self {
            Self::Copy { .. } => LargeStageKind::Copy,
            Self::GatherScatter { .. } => LargeStageKind::GatherScatter,
            Self::HelperWindow { .. } => LargeStageKind::HelperWindow,
            Self::WindowedHelper { .. } => LargeStageKind::WindowedHelper,
            Self::Kernel { .. } => LargeStageKind::Kernel,
            Self::WindowedKernel { .. } => LargeStageKind::WindowedKernel,
            Self::TwiddleTranspose { .. } => LargeStageKind::TwiddleTranspose,
            Self::StripeTranspose { .. } => LargeStageKind::StripeTranspose,
            Self::Scale { .. } => LargeStageKind::Scale,
            Self::HostWindow { .. } => LargeStageKind::HostWindow,
        }
    }

    pub(crate) fn label(&self) -> &'static str {
        match self {
            Self::Copy { label, .. }
            | Self::GatherScatter { label, .. }
            | Self::HelperWindow { label, .. }
            | Self::WindowedHelper { label, .. }
            | Self::Kernel { label, .. }
            | Self::WindowedKernel { label, .. }
            | Self::TwiddleTranspose { label, .. }
            | Self::StripeTranspose { label, .. }
            | Self::Scale { label, .. }
            | Self::HostWindow { label, .. } => label,
        }
    }

    pub(crate) fn ranges(&self) -> Vec<LogicalRange> {
        match self {
            Self::Copy { src, dst, .. } | Self::GatherScatter { src, dst, .. } => {
                vec![*src, *dst]
            }
            Self::HelperWindow { range, .. }
            | Self::WindowedHelper { range, .. }
            | Self::Scale { range, .. } => vec![*range],
            Self::Kernel { input, output, .. }
            | Self::WindowedKernel { input, output, .. }
            | Self::TwiddleTranspose { input, output, .. }
            | Self::StripeTranspose { input, output, .. } => vec![*input, *output],
            Self::HostWindow { range, .. } => vec![*range],
        }
    }

    pub(crate) fn storage_ranges(&self) -> Vec<LogicalRange> {
        match self {
            Self::Copy { .. }
            | Self::GatherScatter { .. }
            | Self::WindowedHelper { .. }
            | Self::WindowedKernel { .. }
            | Self::StripeTranspose { .. }
            | Self::Scale { .. }
            | Self::HostWindow { .. } => Vec::new(),
            Self::HelperWindow { range, .. } => vec![*range],
            Self::Kernel { input, output, .. } | Self::TwiddleTranspose { input, output, .. } => {
                vec![*input, *output]
            }
        }
    }

    pub(crate) fn copy_ranges(&self) -> Vec<LogicalRange> {
        match self {
            Self::Copy { src, dst, .. } => vec![*src, *dst],
            _ => Vec::new(),
        }
    }

    pub(crate) fn work_items(&self) -> Option<u64> {
        match self {
            Self::Kernel { work_items, .. } | Self::TwiddleTranspose { work_items, .. } => {
                Some(*work_items)
            }
            Self::WindowedKernel { work_items, .. } => Some(*work_items),
            Self::StripeTranspose { work_items, .. } | Self::Scale { work_items, .. } => {
                Some(*work_items)
            }
            Self::GatherScatter {
                stride_elements, ..
            } => Some(*stride_elements),
            Self::Copy { .. }
            | Self::HelperWindow { .. }
            | Self::WindowedHelper { .. }
            | Self::HostWindow { .. } => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct StageRequirements {
    pub(crate) max_binding_bytes: u64,
    pub(crate) max_buffer_size: u64,
    pub(crate) storage_alignment: u64,
    pub(crate) copy_alignment: u64,
    pub(crate) scratch_bytes: u64,
}

impl StageRequirements {
    pub(crate) fn new(
        max_binding_bytes: u64,
        max_buffer_size: u64,
        storage_alignment: u64,
        copy_alignment: u64,
        scratch_bytes: u64,
    ) -> Result<Self> {
        if max_binding_bytes == 0 || max_buffer_size == 0 {
            return Err(FftError::WindowScheduleUnsupported {
                reason: "large graph requires non-zero device limits",
                requested_bytes: max_binding_bytes.max(max_buffer_size),
                max_bind_bytes: max_binding_bytes,
            });
        }
        Ok(Self {
            max_binding_bytes,
            max_buffer_size,
            storage_alignment: storage_alignment.max(1),
            copy_alignment: copy_alignment.max(1),
            scratch_bytes,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct LargeExecutionGraph {
    label: &'static str,
    stages: Vec<LargeStage>,
    requirements: Vec<StageRequirements>,
    scratch_bytes: Vec<u64>,
}

impl LargeExecutionGraph {
    pub(crate) fn new(label: &'static str) -> Self {
        Self {
            label,
            stages: Vec::new(),
            requirements: Vec::new(),
            scratch_bytes: Vec::new(),
        }
    }

    #[cfg(test)]
    pub(crate) fn label(&self) -> &'static str {
        self.label
    }

    pub(crate) fn push_stage(
        &mut self,
        stage: LargeStage,
        requirements: StageRequirements,
    ) -> Result<()> {
        validate_stage(&stage, requirements)?;
        if requirements.scratch_bytes > 0 {
            self.scratch_bytes.push(requirements.scratch_bytes);
        }
        self.stages.push(stage);
        self.requirements.push(requirements);
        Ok(())
    }

    pub(crate) fn stages(&self) -> &[LargeStage] {
        &self.stages
    }

    #[cfg(test)]
    pub(crate) fn scratch_bytes(&self) -> &[u64] {
        &self.scratch_bytes
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct LargeExecutionPlan {
    graph: LargeExecutionGraph,
}

impl LargeExecutionPlan {
    pub(crate) fn new(graph: LargeExecutionGraph) -> Self {
        Self { graph }
    }

    pub(crate) fn graph(&self) -> &LargeExecutionGraph {
        &self.graph
    }
}

fn validate_stage(stage: &LargeStage, requirements: StageRequirements) -> Result<()> {
    if matches!(stage.work_items(), Some(0)) {
        return Err(FftError::LargeGraphStageUnsupported {
            stage: stage.label(),
            reason: "stage work item count must be non-zero",
        });
    }

    for range in stage.ranges() {
        let _ = (range.buffer.kind_label(), range.buffer.index());
        if range.element_count() == 0 {
            return Err(FftError::LargeGraphStageUnsupported {
                stage: stage.label(),
                reason: "stage logical range must not be empty",
            });
        }
        if range.size_bytes > requirements.max_buffer_size {
            return Err(FftError::WindowScheduleUnsupported {
                reason: "large graph logical range exceeds max buffer size",
                requested_bytes: range.size_bytes,
                max_bind_bytes: requirements.max_buffer_size,
            });
        }
    }

    for range in stage.copy_ranges() {
        if range.offset_bytes % requirements.copy_alignment != 0
            || range.size_bytes % requirements.copy_alignment != 0
        {
            return Err(FftError::BufferViewCopyUnaligned {
                offset: range.offset_bytes,
                size: range.size_bytes,
                alignment: requirements.copy_alignment,
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req() -> StageRequirements {
        StageRequirements::new(256, 1024, 256, 4, 128).unwrap()
    }

    #[test]
    fn graph_records_stage_kinds_and_scratch() {
        let mut graph = LargeExecutionGraph::new("test");
        let src =
            LogicalRange::new(LogicalBufferId::Input, 0, 128, ElementFormat::ComplexF32).unwrap();
        let dst = LogicalRange::new(LogicalBufferId::Stage(0), 0, 128, ElementFormat::ComplexF32)
            .unwrap();

        graph
            .push_stage(
                LargeStage::Copy {
                    label: "copy",
                    src,
                    dst,
                },
                req(),
            )
            .unwrap();

        assert_eq!(graph.label(), "test");
        assert_eq!(graph.stages()[0].kind(), LargeStageKind::Copy);
        assert_eq!(graph.stages()[0].label(), "copy");
        assert_eq!(graph.scratch_bytes(), &[128]);
    }

    #[test]
    fn graph_rejects_range_larger_than_max_buffer() {
        let src =
            LogicalRange::new(LogicalBufferId::Input, 0, 2048, ElementFormat::ComplexF32).unwrap();
        let dst =
            LogicalRange::new(LogicalBufferId::Output, 0, 2048, ElementFormat::ComplexF32).unwrap();
        let mut graph = LargeExecutionGraph::new("too-large");
        assert!(matches!(
            graph.push_stage(
                LargeStage::Copy {
                    label: "copy",
                    src,
                    dst,
                },
                req(),
            ),
            Err(FftError::WindowScheduleUnsupported {
                reason: "large graph logical range exceeds max buffer size",
                requested_bytes: 2048,
                max_bind_bytes: 1024,
            })
        ));
    }
}
