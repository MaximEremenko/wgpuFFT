use std::collections::HashSet;

use crate::error::{FftError, Result};

const COPY_BUFFER_ALIGNMENT: u64 = 4;

/// One physical segment of a logical FFT buffer view.
#[derive(Debug, Clone, Copy)]
pub struct BufferSegment<'a> {
    pub buffer: &'a wgpu::Buffer,
    pub offset_bytes: u64,
    pub size_bytes: u64,
}

impl<'a> BufferSegment<'a> {
    pub fn new(buffer: &'a wgpu::Buffer, offset_bytes: u64, size_bytes: u64) -> Self {
        Self {
            buffer,
            offset_bytes,
            size_bytes,
        }
    }
}

/// A logical byte range used as FFT input, output, or workspace.
#[derive(Debug, Clone)]
pub struct BufferView<'a> {
    segments: Vec<BufferSegment<'a>>,
    logical_byte_offset: u64,
    length_bytes: u64,
}

#[derive(Debug, Clone, Copy)]
pub struct BufferRange<'a> {
    pub buffer: &'a wgpu::Buffer,
    pub offset_bytes: u64,
    pub size_bytes: u64,
}

/// Logical FFT buffer layout expressed in endpoint element units.
///
/// For C2C endpoints one element is an interleaved complex pair in the plan's
/// configured precision. For real endpoints one element is a scalar `f32`, and
/// for packed real-spectrum endpoints one element is an interleaved complex
/// `f32` pair.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BufferLayout {
    pub element_offset: u64,
    pub element_stride: u64,
    pub batch_stride: Option<u64>,
}

/// A buffer view plus an optional logical layout.
#[derive(Debug, Clone)]
pub struct FftIoView<'a> {
    view: BufferView<'a>,
    layout: BufferLayout,
}

impl BufferLayout {
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

    pub(crate) fn required_complex_span(&self, logical_per_batch: u64, batch: u64) -> Result<u64> {
        self.validate()?;
        if logical_per_batch == 0 || batch == 0 {
            return Ok(0);
        }

        let per_batch_span = per_batch_span(self.element_stride, logical_per_batch)?;
        let batch_stride = self.batch_stride.unwrap_or(per_batch_span);
        if batch_stride < per_batch_span {
            return Err(FftError::BufferLayoutBatchStrideTooSmall {
                required: per_batch_span,
                actual: batch_stride,
            });
        }

        let batch_offset = checked_mul(batch - 1, batch_stride)?;
        let last_batch_end = checked_add(batch_offset, per_batch_span)?;
        checked_add(self.element_offset, last_batch_end)
    }

    pub(crate) fn resolved_batch_stride(&self, logical_per_batch: u64) -> Result<u64> {
        self.validate()?;
        let per_batch_span = per_batch_span(self.element_stride, logical_per_batch)?;
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
}

impl<'a> FftIoView<'a> {
    pub fn new(view: BufferView<'a>, layout: BufferLayout) -> Result<Self> {
        layout.validate()?;
        Ok(Self { view, layout })
    }

    pub fn contiguous(view: BufferView<'a>) -> Self {
        Self {
            view,
            layout: BufferLayout::contiguous(),
        }
    }

    pub fn view(&self) -> &BufferView<'a> {
        &self.view
    }

    pub fn layout(&self) -> BufferLayout {
        self.layout
    }

    pub(crate) fn into_parts(self) -> (BufferView<'a>, BufferLayout) {
        (self.view, self.layout)
    }
}

impl<'a> BufferView<'a> {
    pub fn new(buffer: &'a wgpu::Buffer, offset: u64, size: u64) -> Result<Self> {
        validate_range(offset, size, buffer.size())?;

        Ok(Self {
            segments: vec![BufferSegment::new(buffer, offset, size)],
            logical_byte_offset: 0,
            length_bytes: size,
        })
    }

    pub fn whole(buffer: &'a wgpu::Buffer) -> Self {
        Self {
            segments: vec![BufferSegment::new(buffer, 0, buffer.size())],
            logical_byte_offset: 0,
            length_bytes: buffer.size(),
        }
    }

    pub fn from_segments(
        segments: &[BufferSegment<'a>],
        logical_byte_offset: u64,
        length_bytes: u64,
    ) -> Result<Self> {
        validate_segments(segments, logical_byte_offset, length_bytes)?;
        Ok(Self {
            segments: segments.to_vec(),
            logical_byte_offset,
            length_bytes,
        })
    }

    /// Returns the buffer for single-segment views.
    ///
    /// Panics for segmented views; use `segments()` or `ranges()` when
    /// `is_single_segment()` is false.
    pub fn buffer(&self) -> &'a wgpu::Buffer {
        self.single_range()
            .expect("BufferView::buffer requires a single physical segment")
            .buffer
    }

    /// Returns the physical byte offset for single-segment views.
    ///
    /// Panics for segmented views; use `segments()` or `ranges()` when
    /// `is_single_segment()` is false.
    pub fn offset(&self) -> u64 {
        self.single_range()
            .expect("BufferView::offset requires a single physical segment")
            .offset_bytes
    }

    pub fn size(&self) -> u64 {
        self.length_bytes
    }

    pub fn logical_byte_offset(&self) -> u64 {
        self.logical_byte_offset
    }

    pub fn segments(&self) -> &[BufferSegment<'a>] {
        &self.segments
    }

    pub fn is_single_segment(&self) -> bool {
        self.single_range().is_some()
    }

    /// Returns whether this logical view is made from complete physical buffers.
    ///
    /// The segmented full-volume executor uses this stricter shape for its
    /// caller-owned endpoints: multiple buffers are supported, but partial
    /// buffers and logical windows into their concatenation remain rejected.
    pub(crate) fn covers_whole_buffers(&self) -> bool {
        if self.logical_byte_offset != 0 {
            return false;
        }
        // Buffers hash by identity; their interior state does not affect it.
        #[allow(clippy::mutable_key_type)]
        let mut unique_buffers = HashSet::with_capacity(self.segments.len());
        let Some(total_bytes) = self.segments.iter().try_fold(0u64, |total, segment| {
            if segment.offset_bytes != 0
                || segment.size_bytes != segment.buffer.size()
                || !unique_buffers.insert(segment.buffer)
            {
                None
            } else {
                total.checked_add(segment.size_bytes)
            }
        }) else {
            return false;
        };
        total_bytes == self.length_bytes
    }

    pub fn ranges(
        &self,
        relative_start_bytes: u64,
        size_bytes: u64,
    ) -> Result<Vec<BufferRange<'a>>> {
        iter_ranges(
            &self.segments,
            self.logical_byte_offset,
            self.length_bytes,
            relative_start_bytes,
            size_bytes,
        )
    }

    pub(crate) fn copy_ranges(
        &self,
        relative_start_bytes: u64,
        size_bytes: u64,
    ) -> Result<Vec<BufferRange<'a>>> {
        let ranges = self.ranges(relative_start_bytes, size_bytes)?;
        for range in &ranges {
            validate_copy_alignment(range.offset_bytes, range.size_bytes)?;
        }
        Ok(ranges)
    }

    pub(crate) fn prefix(self, required_size: u64) -> Result<Self> {
        self.validate_min_size(required_size)?;
        Ok(Self {
            segments: self.segments,
            logical_byte_offset: self.logical_byte_offset,
            length_bytes: required_size,
        })
    }

    pub(crate) fn validate_min_size(&self, required_size: u64) -> Result<()> {
        if self.length_bytes < required_size {
            Err(FftError::BufferViewTooSmall {
                required: required_size,
                actual: self.length_bytes,
            })
        } else {
            Ok(())
        }
    }

    fn single_range(&self) -> Option<BufferRange<'a>> {
        let ranges = self.ranges(0, self.length_bytes).ok()?;
        if ranges.len() == 1 {
            Some(ranges[0])
        } else {
            None
        }
    }
}

fn per_batch_span(element_stride: u64, logical_per_batch: u64) -> Result<u64> {
    if logical_per_batch == 0 {
        Ok(0)
    } else {
        checked_add(checked_mul(element_stride, logical_per_batch - 1)?, 1)
    }
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

fn validate_range(offset: u64, size: u64, buffer_size: u64) -> Result<()> {
    let end = offset
        .checked_add(size)
        .ok_or(FftError::BufferViewOutOfBounds {
            offset,
            size,
            buffer_size,
        })?;
    if end > buffer_size {
        Err(FftError::BufferViewOutOfBounds {
            offset,
            size,
            buffer_size,
        })
    } else {
        Ok(())
    }
}

fn validate_segments(
    segments: &[BufferSegment<'_>],
    logical_byte_offset: u64,
    length_bytes: u64,
) -> Result<()> {
    if segments.is_empty() {
        return Err(FftError::BufferViewEmptySegments);
    }

    let mut total = 0u64;
    for (index, segment) in segments.iter().enumerate() {
        if segment.size_bytes == 0 {
            return Err(FftError::BufferSegmentZeroSize { index });
        }
        validate_segment_range(
            index,
            segment.offset_bytes,
            segment.size_bytes,
            segment.buffer.size(),
        )?;
        total = total
            .checked_add(segment.size_bytes)
            .ok_or(FftError::LengthTooLarge { len: usize::MAX })?;
    }

    let end = logical_byte_offset.checked_add(length_bytes).ok_or(
        FftError::BufferViewWindowOutOfRange {
            offset: logical_byte_offset,
            size: length_bytes,
            length: total,
        },
    )?;
    if end > total {
        return Err(FftError::BufferViewWindowOutOfRange {
            offset: logical_byte_offset,
            size: length_bytes,
            length: total,
        });
    }

    Ok(())
}

fn validate_segment_range(index: usize, offset: u64, size: u64, buffer_size: u64) -> Result<()> {
    let end = offset
        .checked_add(size)
        .ok_or(FftError::BufferSegmentOutOfBounds {
            index,
            offset,
            size,
            buffer_size,
        })?;
    if end > buffer_size {
        Err(FftError::BufferSegmentOutOfBounds {
            index,
            offset,
            size,
            buffer_size,
        })
    } else {
        Ok(())
    }
}

fn iter_ranges<'a>(
    segments: &[BufferSegment<'a>],
    logical_byte_offset: u64,
    length_bytes: u64,
    relative_start_bytes: u64,
    size_bytes: u64,
) -> Result<Vec<BufferRange<'a>>> {
    let relative_end = relative_start_bytes.checked_add(size_bytes).ok_or(
        FftError::BufferViewWindowOutOfRange {
            offset: relative_start_bytes,
            size: size_bytes,
            length: length_bytes,
        },
    )?;
    if relative_end > length_bytes {
        return Err(FftError::BufferViewWindowOutOfRange {
            offset: relative_start_bytes,
            size: size_bytes,
            length: length_bytes,
        });
    }
    if size_bytes == 0 {
        return Ok(Vec::new());
    }

    let mut remaining = size_bytes;
    let mut cursor = logical_byte_offset
        .checked_add(relative_start_bytes)
        .ok_or(FftError::BufferViewWindowOutOfRange {
            offset: relative_start_bytes,
            size: size_bytes,
            length: length_bytes,
        })?;
    let mut logical_position = 0u64;
    let mut ranges = Vec::new();

    for segment in segments {
        let segment_start = logical_position;
        let segment_end = segment_start
            .checked_add(segment.size_bytes)
            .ok_or(FftError::LengthTooLarge { len: usize::MAX })?;
        if cursor >= segment_end {
            logical_position = segment_end;
            continue;
        }

        let within = cursor.saturating_sub(segment_start);
        let take = remaining.min(segment.size_bytes - within);
        ranges.push(BufferRange {
            buffer: segment.buffer,
            offset_bytes: segment.offset_bytes + within,
            size_bytes: take,
        });
        remaining -= take;
        cursor += take;
        logical_position = segment_end;
        if remaining == 0 {
            break;
        }
    }

    if remaining != 0 {
        return Err(FftError::BufferViewWindowOutOfRange {
            offset: relative_start_bytes,
            size: size_bytes,
            length: length_bytes,
        });
    }

    Ok(ranges)
}

fn validate_copy_alignment(offset: u64, size: u64) -> Result<()> {
    if offset.is_multiple_of(COPY_BUFFER_ALIGNMENT) && size.is_multiple_of(COPY_BUFFER_ALIGNMENT) {
        Ok(())
    } else {
        Err(FftError::BufferViewCopyUnaligned {
            offset,
            size,
            alignment: COPY_BUFFER_ALIGNMENT,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_byte_ranges() {
        assert_eq!(validate_range(16, 32, 64), Ok(()));
        assert_eq!(
            validate_range(48, 24, 64).unwrap_err(),
            FftError::BufferViewOutOfBounds {
                offset: 48,
                size: 24,
                buffer_size: 64,
            }
        );
    }

    #[test]
    fn validates_minimum_size() {
        let fake_buffer_size = 128;
        validate_range(32, 64, fake_buffer_size).unwrap();
        let err = FftError::BufferViewTooSmall {
            required: 96,
            actual: 64,
        };
        assert_eq!(
            err.to_string(),
            "FFT buffer view is too small: required 96 bytes, got 64 bytes"
        );
    }

    #[test]
    fn rejects_empty_segments() {
        assert_eq!(
            validate_segments(&[], 0, 0),
            Err(FftError::BufferViewEmptySegments)
        );
    }

    #[test]
    fn rejects_zero_size_segments() {
        assert_eq!(
            validate_segment_shape(1, 8, 0, 64),
            Err(FftError::BufferSegmentZeroSize { index: 1 })
        );
    }

    #[test]
    fn rejects_out_of_bounds_segments() {
        assert_eq!(
            validate_segment_range(0, 48, 24, 64).unwrap_err(),
            FftError::BufferSegmentOutOfBounds {
                index: 0,
                offset: 48,
                size: 24,
                buffer_size: 64,
            }
        );
    }

    #[test]
    fn rejects_out_of_range_windows() {
        assert_eq!(
            validate_logical_window(48, 24, 64).unwrap_err(),
            FftError::BufferViewWindowOutOfRange {
                offset: 48,
                size: 24,
                length: 64,
            }
        );
    }

    #[test]
    fn rejects_unaligned_copy_ranges() {
        assert_eq!(
            validate_copy_alignment(2, 16).unwrap_err(),
            FftError::BufferViewCopyUnaligned {
                offset: 2,
                size: 16,
                alignment: COPY_BUFFER_ALIGNMENT,
            }
        );
        assert_eq!(
            validate_copy_alignment(4, 6).unwrap_err(),
            FftError::BufferViewCopyUnaligned {
                offset: 4,
                size: 6,
                alignment: COPY_BUFFER_ALIGNMENT,
            }
        );
    }

    #[test]
    fn validates_c2c_layout_spans() {
        let contiguous = BufferLayout::contiguous();
        assert!(contiguous.is_contiguous_for(16, 2).unwrap());
        assert_eq!(contiguous.required_complex_span(16, 2).unwrap(), 32);

        let strided = BufferLayout::new(2, 3).unwrap().with_batch_stride(64);
        assert!(!strided.is_contiguous_for(16, 2).unwrap());
        assert_eq!(strided.required_complex_span(16, 2).unwrap(), 112);
    }

    #[test]
    fn rejects_invalid_c2c_layouts() {
        assert_eq!(
            BufferLayout::new(0, 0).unwrap_err(),
            FftError::BufferLayoutZeroStride
        );
        assert_eq!(
            BufferLayout::new(0, 3)
                .unwrap()
                .with_batch_stride(4)
                .required_complex_span(16, 2)
                .unwrap_err(),
            FftError::BufferLayoutBatchStrideTooSmall {
                required: 46,
                actual: 4,
            }
        );
    }

    fn validate_segment_shape(
        index: usize,
        offset: u64,
        size: u64,
        buffer_size: u64,
    ) -> Result<()> {
        if size == 0 {
            return Err(FftError::BufferSegmentZeroSize { index });
        }
        validate_segment_range(index, offset, size, buffer_size)
    }

    fn validate_logical_window(offset: u64, size: u64, length: u64) -> Result<()> {
        if offset.checked_add(size).is_none_or(|end| end > length) {
            Err(FftError::BufferViewWindowOutOfRange {
                offset,
                size,
                length,
            })
        } else {
            Ok(())
        }
    }
}
