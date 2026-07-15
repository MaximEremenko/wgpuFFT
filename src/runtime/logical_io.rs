use crate::error::{FftError, Result};
use crate::runtime::buffer_view::{BufferLayout, BufferRange, BufferView, FftIoView};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FftEndpointFormat {
    ComplexF32,
    ComplexF64,
    ComplexDf64,
    RealF32,
    PackedComplexF32,
}

impl FftEndpointFormat {
    pub const fn bytes_per_element(self) -> u64 {
        match self {
            Self::ComplexF32 | Self::PackedComplexF32 => 8,
            Self::ComplexF64 | Self::ComplexDf64 => 16,
            Self::RealF32 => 4,
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ComplexF32 => "complex-f32",
            Self::ComplexF64 => "complex-f64",
            Self::ComplexDf64 => "complex-df64",
            Self::RealF32 => "real-f32",
            Self::PackedComplexF32 => "packed-complex-f32",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FftLogicalLayout {
    pub element_offset: u64,
    pub element_stride: u64,
    pub batch_stride: Option<u64>,
}

#[derive(Debug, Clone)]
pub struct FftLogicalView<'a> {
    view: BufferView<'a>,
    layout: FftLogicalLayout,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BoundLogicalIoKind {
    Contiguous,
    Offset,
    Segmented,
    Strided,
    SegmentedStrided,
}

impl BoundLogicalIoKind {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Contiguous => "contiguous",
            Self::Offset => "offset",
            Self::Segmented => "segmented",
            Self::Strided => "strided",
            Self::SegmentedStrided => "segmented-strided",
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct BoundLogicalIo<'a> {
    pub(crate) view: BufferView<'a>,
    pub(crate) layout: FftLogicalLayout,
    pub(crate) format: FftEndpointFormat,
    pub(crate) kind: BoundLogicalIoKind,
    pub(crate) logical_elements_per_batch: u64,
    pub(crate) batch: u64,
    pub(crate) logical_bytes: u64,
    pub(crate) physical_span_bytes: u64,
    pub(crate) physical_ranges: Vec<BufferRange<'a>>,
    pub(crate) storage_aligned: bool,
    pub(crate) copy_aligned: bool,
    pub(crate) covers_whole_buffers: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct LogicalIoShape {
    kind: BoundLogicalIoKind,
    logical_bytes: u64,
    physical_span_bytes: u64,
    required_view_bytes: u64,
}

impl FftLogicalLayout {
    pub fn contiguous() -> Self {
        Self {
            element_offset: 0,
            element_stride: 1,
            batch_stride: None,
        }
    }

    pub fn new(element_offset: u64, element_stride: u64) -> Result<Self> {
        let layout = Self {
            element_offset,
            element_stride,
            batch_stride: None,
        };
        layout.validate()?;
        Ok(layout)
    }

    pub fn with_batch_stride(mut self, batch_stride: u64) -> Self {
        self.batch_stride = Some(batch_stride);
        self
    }

    pub(crate) fn validate(&self) -> Result<()> {
        if self.element_stride == 0 {
            Err(FftError::BufferLayoutZeroStride)
        } else {
            Ok(())
        }
    }

    pub(crate) fn required_element_span(&self, logical_per_batch: u64, batch: u64) -> Result<u64> {
        self.validate()?;
        if logical_per_batch == 0 || batch == 0 {
            return Ok(0);
        }
        let per_batch_span = if logical_per_batch == 0 {
            0
        } else {
            checked_add(checked_mul(self.element_stride, logical_per_batch - 1)?, 1)?
        };
        let batch_stride = self.batch_stride.unwrap_or(per_batch_span);
        if batch_stride < per_batch_span {
            return Err(FftError::BufferLayoutBatchStrideTooSmall {
                required: per_batch_span,
                actual: batch_stride,
            });
        }
        let batch_offset = checked_mul(batch - 1, batch_stride)?;
        checked_add(
            self.element_offset,
            checked_add(batch_offset, per_batch_span)?,
        )
    }

    pub(crate) fn resolved_batch_stride(&self, logical_per_batch: u64) -> Result<u64> {
        self.validate()?;
        let per_batch_span = if logical_per_batch == 0 {
            0
        } else {
            checked_add(checked_mul(self.element_stride, logical_per_batch - 1)?, 1)?
        };
        Ok(self.batch_stride.unwrap_or(per_batch_span))
    }

    pub(crate) fn is_contiguous_for(&self, logical_per_batch: u64, batch: u64) -> Result<bool> {
        self.validate()?;
        if self.element_offset != 0 || self.element_stride != 1 {
            return Ok(false);
        }
        let Some(batch_stride) = self.batch_stride else {
            return Ok(true);
        };
        if batch <= 1 {
            Ok(true)
        } else {
            Ok(batch_stride == logical_per_batch)
        }
    }

    pub(crate) fn as_buffer_layout(self) -> BufferLayout {
        let mut layout = BufferLayout::new(self.element_offset, self.element_stride)
            .expect("validated logical layout must convert to BufferLayout");
        if let Some(batch_stride) = self.batch_stride {
            layout = layout.with_batch_stride(batch_stride);
        }
        layout
    }
}

impl<'a> FftLogicalView<'a> {
    pub fn new(view: BufferView<'a>, layout: FftLogicalLayout) -> Result<Self> {
        layout.validate()?;
        Ok(Self { view, layout })
    }

    pub fn contiguous(view: BufferView<'a>) -> Self {
        Self {
            view,
            layout: FftLogicalLayout::contiguous(),
        }
    }

    pub fn from_io_view(io: FftIoView<'a>) -> Self {
        let layout = io.layout();
        Self {
            view: io.view().clone(),
            layout: FftLogicalLayout {
                element_offset: layout.element_offset,
                element_stride: layout.element_stride,
                batch_stride: layout.batch_stride,
            },
        }
    }

    pub fn from_c2c_io_view(io: FftIoView<'a>) -> Self {
        Self::from_io_view(io)
    }

    pub fn view(&self) -> &BufferView<'a> {
        &self.view
    }

    pub fn layout(&self) -> FftLogicalLayout {
        self.layout
    }

    pub(crate) fn into_parts(self) -> (BufferView<'a>, FftLogicalLayout) {
        (self.view, self.layout)
    }
}

impl<'a> BoundLogicalIo<'a> {
    pub(crate) fn bind(
        logical: FftLogicalView<'a>,
        format: FftEndpointFormat,
        logical_elements_per_batch: u64,
        batch: u64,
        storage_alignment: u64,
    ) -> Result<Self> {
        let (view, layout) = logical.into_parts();
        let covers_whole_buffers = view.covers_whole_buffers();
        let segmented = !view.is_single_segment();
        let single_segment_offset = if segmented { None } else { Some(view.offset()) };
        let shape = logical_io_shape(
            layout,
            format,
            logical_elements_per_batch,
            batch,
            view.size(),
            segmented,
            single_segment_offset,
        )?;
        let view = view.prefix(shape.required_view_bytes)?;
        let physical_ranges = view.ranges(0, shape.physical_span_bytes)?;
        let storage_alignment = storage_alignment.max(1);
        let storage_aligned = physical_ranges
            .iter()
            .all(|range| range.offset_bytes % storage_alignment == 0);
        let copy_aligned = physical_ranges
            .iter()
            .all(|range| range.offset_bytes % 4 == 0 && range.size_bytes % 4 == 0);

        Ok(Self {
            view,
            layout,
            format,
            kind: shape.kind,
            logical_elements_per_batch,
            batch,
            logical_bytes: shape.logical_bytes,
            physical_span_bytes: shape.physical_span_bytes,
            physical_ranges,
            storage_aligned,
            copy_aligned,
            covers_whole_buffers,
        })
    }

    pub(crate) fn is_contiguous(&self) -> bool {
        matches!(
            self.kind,
            BoundLogicalIoKind::Contiguous
                | BoundLogicalIoKind::Offset
                | BoundLogicalIoKind::Segmented
        )
    }

    pub(crate) fn is_segmented(&self) -> bool {
        matches!(
            self.kind,
            BoundLogicalIoKind::Segmented | BoundLogicalIoKind::SegmentedStrided
        )
    }

    pub(crate) fn into_c2c_io_view(self) -> Result<FftIoView<'a>> {
        FftIoView::new(self.view, self.layout.as_buffer_layout())
    }

    pub(crate) fn view_prefix(&self, bytes: u64) -> Result<BufferView<'a>> {
        self.view.clone().prefix(bytes)
    }

    pub(crate) fn layout_kind(&self) -> &'static str {
        self.kind.as_str()
    }
}

fn logical_io_shape(
    layout: FftLogicalLayout,
    format: FftEndpointFormat,
    logical_elements_per_batch: u64,
    batch: u64,
    view_size: u64,
    segmented: bool,
    single_segment_offset: Option<u64>,
) -> Result<LogicalIoShape> {
    layout.validate()?;
    let logical_elements = checked_mul(logical_elements_per_batch, batch)?;
    let logical_bytes = checked_mul(logical_elements, format.bytes_per_element())?;
    let physical_elements = layout.required_element_span(logical_elements_per_batch, batch)?;
    let physical_span_bytes = checked_mul(physical_elements, format.bytes_per_element())?;
    if physical_span_bytes > view_size {
        return Err(FftError::BufferLayoutOutOfBounds {
            required_bytes: physical_span_bytes,
            actual_bytes: view_size,
        });
    }
    let contiguous = layout.is_contiguous_for(logical_elements_per_batch, batch)?;
    let kind = match (contiguous, segmented) {
        (true, false) if single_segment_offset.unwrap_or(0) == 0 => BoundLogicalIoKind::Contiguous,
        (true, false) => BoundLogicalIoKind::Offset,
        (true, true) => BoundLogicalIoKind::Segmented,
        (false, false) => BoundLogicalIoKind::Strided,
        (false, true) => BoundLogicalIoKind::SegmentedStrided,
    };
    Ok(LogicalIoShape {
        kind,
        logical_bytes,
        physical_span_bytes,
        required_view_bytes: if contiguous {
            logical_bytes
        } else {
            physical_span_bytes
        },
    })
}

fn checked_add(a: u64, b: u64) -> Result<u64> {
    a.checked_add(b).ok_or(FftError::BufferLayoutTooLarge {
        value: u64::MAX,
        limit: u64::MAX - 1,
    })
}

fn checked_mul(a: u64, b: u64) -> Result<u64> {
    a.checked_mul(b).ok_or(FftError::BufferLayoutTooLarge {
        value: u64::MAX,
        limit: u64::MAX - 1,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn logical_layout_matches_c2c_span_rules() {
        let layout = FftLogicalLayout::new(2, 3).unwrap().with_batch_stride(64);
        assert_eq!(layout.required_element_span(16, 2).unwrap(), 112);
        assert!(!layout.is_contiguous_for(16, 2).unwrap());
    }

    #[test]
    fn endpoint_formats_expose_element_sizes() {
        assert_eq!(FftEndpointFormat::RealF32.bytes_per_element(), 4);
        assert_eq!(FftEndpointFormat::ComplexF32.bytes_per_element(), 8);
        assert_eq!(FftEndpointFormat::ComplexF64.bytes_per_element(), 16);
        assert_eq!(FftEndpointFormat::ComplexDf64.bytes_per_element(), 16);
        assert_eq!(FftEndpointFormat::ComplexDf64.as_str(), "complex-df64");
        assert_eq!(
            FftEndpointFormat::PackedComplexF32.as_str(),
            "packed-complex-f32"
        );
    }

    #[test]
    fn logical_io_shape_classifies_c2c_layout_variants() {
        assert_logical_io_shape(
            FftEndpointFormat::ComplexF32,
            FftLogicalLayout::contiguous(),
            8,
            2,
            128,
            false,
            Some(0),
            BoundLogicalIoKind::Contiguous,
            128,
            128,
            128,
        );
        assert_logical_io_shape(
            FftEndpointFormat::ComplexF32,
            FftLogicalLayout::contiguous(),
            8,
            2,
            192,
            false,
            Some(64),
            BoundLogicalIoKind::Offset,
            128,
            128,
            128,
        );
        assert_logical_io_shape(
            FftEndpointFormat::ComplexF32,
            FftLogicalLayout::new(1, 2).unwrap().with_batch_stride(20),
            5,
            3,
            400,
            false,
            Some(0),
            BoundLogicalIoKind::Strided,
            120,
            400,
            400,
        );
    }

    #[test]
    fn logical_io_shape_classifies_real_and_packed_segmented_layouts() {
        assert_logical_io_shape(
            FftEndpointFormat::RealF32,
            FftLogicalLayout::contiguous(),
            8,
            2,
            64,
            true,
            None,
            BoundLogicalIoKind::Segmented,
            64,
            64,
            64,
        );
        assert_logical_io_shape(
            FftEndpointFormat::PackedComplexF32,
            FftLogicalLayout::new(3, 4).unwrap().with_batch_stride(24),
            6,
            2,
            384,
            true,
            None,
            BoundLogicalIoKind::SegmentedStrided,
            96,
            384,
            384,
        );
    }

    #[test]
    fn logical_io_shape_reports_format_aware_span_bounds() {
        assert_eq!(
            logical_io_shape(
                FftLogicalLayout::new(3, 4).unwrap().with_batch_stride(24),
                FftEndpointFormat::RealF32,
                6,
                2,
                191,
                true,
                None,
            )
            .unwrap_err(),
            FftError::BufferLayoutOutOfBounds {
                required_bytes: 192,
                actual_bytes: 191,
            }
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn assert_logical_io_shape(
        format: FftEndpointFormat,
        layout: FftLogicalLayout,
        logical_elements_per_batch: u64,
        batch: u64,
        view_size: u64,
        segmented: bool,
        single_segment_offset: Option<u64>,
        expected_kind: BoundLogicalIoKind,
        expected_logical_bytes: u64,
        expected_physical_span_bytes: u64,
        expected_required_view_bytes: u64,
    ) {
        let shape = logical_io_shape(
            layout,
            format,
            logical_elements_per_batch,
            batch,
            view_size,
            segmented,
            single_segment_offset,
        )
        .unwrap();
        assert_eq!(shape.kind, expected_kind);
        assert_eq!(shape.logical_bytes, expected_logical_bytes);
        assert_eq!(shape.physical_span_bytes, expected_physical_span_bytes);
        assert_eq!(shape.required_view_bytes, expected_required_view_bytes);
    }
}
