use std::fmt;
use std::ops::Range;

const TYPE1_QUERY_COUNT: u32 = 8;
const TYPE2_QUERY_COUNT: u32 = 4;

/// A measured stage in a GPU-resident NUFFT execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum NufftGpuStage {
    BinClearCount,
    ScanAndTerminal,
    Scatter,
    Sort,
    GatherSpread,
    FineGridFft,
    Deconvolution,
    Predeconvolution,
    Interpolation,
}

impl NufftGpuStage {
    /// Stable, human-readable stage label used by diagnostic output.
    pub const fn label(self) -> &'static str {
        match self {
            Self::BinClearCount => "bin-clear-count",
            Self::ScanAndTerminal => "scan-and-terminal",
            Self::Scatter => "scatter",
            Self::Sort => "sort",
            Self::GatherSpread => "gather-spread",
            Self::FineGridFft => "fine-grid-fft",
            Self::Deconvolution => "deconvolution",
            Self::Predeconvolution => "predeconvolution",
            Self::Interpolation => "interpolation",
        }
    }
}

impl fmt::Display for NufftGpuStage {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.label())
    }
}

/// The two absolute timestamp-query indices bounding one measured stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NufftGpuStageQuery {
    stage: NufftGpuStage,
    start_query: u32,
    end_query: u32,
}

impl NufftGpuStageQuery {
    pub const fn stage(&self) -> NufftGpuStage {
        self.stage
    }

    pub const fn start_query(&self) -> u32 {
        self.start_query
    }

    pub const fn end_query(&self) -> u32 {
        self.end_query
    }
}

/// Error returned when an absolute query range cannot be represented by `u32`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct NufftGpuProfileLayoutError {
    first_query: u32,
    query_count: u32,
}

impl fmt::Display for NufftGpuProfileLayoutError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "timestamp query range starting at {} with {} entries exceeds u32",
            self.first_query, self.query_count
        )
    }
}

impl std::error::Error for NufftGpuProfileLayoutError {}

/// Absolute query allocation and stage boundaries for one profiled execution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NufftGpuProfileLayout {
    query_range: Range<u32>,
    stages: Vec<NufftGpuStageQuery>,
}

impl NufftGpuProfileLayout {
    /// Builds the eight-query layout for a type-1 execution.
    pub(crate) fn type1(first_query: u32) -> Result<Self, NufftGpuProfileLayoutError> {
        Self::new(
            first_query,
            TYPE1_QUERY_COUNT,
            &[
                (NufftGpuStage::BinClearCount, 0, 1),
                (NufftGpuStage::ScanAndTerminal, 1, 2),
                (NufftGpuStage::Scatter, 2, 3),
                (NufftGpuStage::Sort, 3, 4),
                (NufftGpuStage::GatherSpread, 4, 5),
                (NufftGpuStage::FineGridFft, 5, 6),
                (NufftGpuStage::Deconvolution, 6, 7),
            ],
        )
    }

    /// Builds the four-query layout for a type-2 execution.
    pub(crate) fn type2(first_query: u32) -> Result<Self, NufftGpuProfileLayoutError> {
        Self::new(
            first_query,
            TYPE2_QUERY_COUNT,
            &[
                (NufftGpuStage::Predeconvolution, 0, 1),
                (NufftGpuStage::FineGridFft, 1, 2),
                (NufftGpuStage::Interpolation, 2, 3),
            ],
        )
    }

    pub fn query_range(&self) -> Range<u32> {
        self.query_range.clone()
    }

    pub const fn query_count(&self) -> u32 {
        self.query_range.end - self.query_range.start
    }

    pub fn stages(&self) -> &[NufftGpuStageQuery] {
        &self.stages
    }

    pub fn stage(&self, stage: NufftGpuStage) -> Option<&NufftGpuStageQuery> {
        self.stages.iter().find(|query| query.stage == stage)
    }

    fn new(
        first_query: u32,
        query_count: u32,
        stage_offsets: &[(NufftGpuStage, u32, u32)],
    ) -> Result<Self, NufftGpuProfileLayoutError> {
        let error = NufftGpuProfileLayoutError {
            first_query,
            query_count,
        };
        let query_end = first_query.checked_add(query_count).ok_or(error)?;
        let mut stages = Vec::with_capacity(stage_offsets.len());
        for &(stage, start_offset, end_offset) in stage_offsets {
            let start_query = first_query.checked_add(start_offset).ok_or(error)?;
            let end_query = first_query.checked_add(end_offset).ok_or(error)?;
            debug_assert!(start_query < query_end);
            debug_assert!(end_query < query_end);
            stages.push(NufftGpuStageQuery {
                stage,
                start_query,
                end_query,
            });
        }
        Ok(Self {
            query_range: first_query..query_end,
            stages,
        })
    }

    fn absolute_query(&self, offset: u32) -> Option<u32> {
        self.query_range
            .start
            .checked_add(offset)
            .filter(|&query| query < self.query_range.end)
    }
}

/// Optional timestamp-query writes used only by profiled encode paths.
#[derive(Clone, Copy)]
pub(crate) enum GpuProfileQueryWriter<'a> {
    Disabled,
    Enabled {
        query_set: &'a wgpu::QuerySet,
        layout: &'a NufftGpuProfileLayout,
    },
}

impl<'a> GpuProfileQueryWriter<'a> {
    pub(crate) const fn disabled() -> Self {
        Self::Disabled
    }

    pub(crate) const fn enabled(
        query_set: &'a wgpu::QuerySet,
        layout: &'a NufftGpuProfileLayout,
    ) -> Self {
        Self::Enabled { query_set, layout }
    }

    /// Builds pass timestamp writes from offsets relative to the layout's first query.
    pub(crate) fn timestamp_writes(
        &self,
        beginning_offset: Option<u32>,
        end_offset: Option<u32>,
    ) -> Option<wgpu::ComputePassTimestampWrites<'a>> {
        let Self::Enabled { query_set, layout } = self else {
            return None;
        };
        assert!(
            beginning_offset.is_some() || end_offset.is_some(),
            "a profiled compute pass must write at least one timestamp"
        );
        let absolute = |offset: u32| {
            layout
                .absolute_query(offset)
                .expect("profile timestamp offset must lie inside its query layout")
        };
        Some(wgpu::ComputePassTimestampWrites {
            query_set,
            beginning_of_pass_write_index: beginning_offset.map(absolute),
            end_of_pass_write_index: end_offset.map(absolute),
        })
    }

    /// Encodes an empty pass whose end timestamp precedes type-1 buffer clears.
    pub(crate) fn encode_type1_start_marker(&self, encoder: &mut wgpu::CommandEncoder) {
        let Some(timestamp_writes) = self.timestamp_writes(None, Some(0)) else {
            return;
        };
        if let Self::Enabled { layout, .. } = self {
            debug_assert_eq!(
                layout.stages.first().map(NufftGpuStageQuery::stage),
                Some(NufftGpuStage::BinClearCount)
            );
        }
        let _pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("wgpu_nufft.profile.type1.start_marker"),
            timestamp_writes: Some(timestamp_writes),
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn type1_layout_maps_all_seven_stages() {
        let layout = NufftGpuProfileLayout::type1(10).unwrap();
        assert_eq!(layout.query_range(), 10..18);
        assert_eq!(layout.query_count(), 8);
        let expected = [
            (NufftGpuStage::BinClearCount, 10, 11),
            (NufftGpuStage::ScanAndTerminal, 11, 12),
            (NufftGpuStage::Scatter, 12, 13),
            (NufftGpuStage::Sort, 13, 14),
            (NufftGpuStage::GatherSpread, 14, 15),
            (NufftGpuStage::FineGridFft, 15, 16),
            (NufftGpuStage::Deconvolution, 16, 17),
        ];
        let actual = layout
            .stages()
            .iter()
            .map(|query| (query.stage(), query.start_query(), query.end_query()))
            .collect::<Vec<_>>();
        assert_eq!(actual, expected);
    }

    #[test]
    fn type2_layout_maps_all_three_stages() {
        let layout = NufftGpuProfileLayout::type2(20).unwrap();
        assert_eq!(layout.query_range(), 20..24);
        assert_eq!(layout.query_count(), 4);
        let expected = [
            (NufftGpuStage::Predeconvolution, 20, 21),
            (NufftGpuStage::FineGridFft, 21, 22),
            (NufftGpuStage::Interpolation, 22, 23),
        ];
        let actual = layout
            .stages()
            .iter()
            .map(|query| (query.stage(), query.start_query(), query.end_query()))
            .collect::<Vec<_>>();
        assert_eq!(actual, expected);
        assert_eq!(
            layout.stage(NufftGpuStage::FineGridFft),
            Some(&NufftGpuStageQuery {
                stage: NufftGpuStage::FineGridFft,
                start_query: 21,
                end_query: 22,
            })
        );
    }

    #[test]
    fn layouts_reject_query_range_overflow() {
        let type1_first = u32::MAX - TYPE1_QUERY_COUNT + 1;
        let type2_first = u32::MAX - TYPE2_QUERY_COUNT + 1;
        assert_eq!(
            NufftGpuProfileLayout::type1(type1_first),
            Err(NufftGpuProfileLayoutError {
                first_query: type1_first,
                query_count: TYPE1_QUERY_COUNT,
            })
        );
        assert_eq!(
            NufftGpuProfileLayout::type2(type2_first),
            Err(NufftGpuProfileLayoutError {
                first_query: type2_first,
                query_count: TYPE2_QUERY_COUNT,
            })
        );
    }

    #[test]
    fn layouts_accept_largest_representable_ranges() {
        let type1 = NufftGpuProfileLayout::type1(u32::MAX - TYPE1_QUERY_COUNT).unwrap();
        let type2 = NufftGpuProfileLayout::type2(u32::MAX - TYPE2_QUERY_COUNT).unwrap();
        assert_eq!(type1.query_range().end, u32::MAX);
        assert_eq!(type2.query_range().end, u32::MAX);
    }
}
