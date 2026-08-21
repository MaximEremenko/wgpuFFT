use crate::error::{FftError, Result};
use crate::runtime::buffer_view::{BufferRange, BufferView};
use crate::runtime::large_graph::ElementFormat;
use crate::runtime::logical_io::{BoundLogicalIo, FftEndpointFormat, FftLogicalView};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SchedulerLimits {
    pub(crate) max_storage_buffer_binding_size: u64,
    pub(crate) max_buffer_size: u64,
    pub(crate) storage_alignment: u64,
    pub(crate) copy_alignment: u64,
}

impl SchedulerLimits {
    pub(crate) fn from_device(device: &wgpu::Device) -> Self {
        let limits = device.limits();
        Self {
            max_storage_buffer_binding_size: limits.max_storage_buffer_binding_size,
            max_buffer_size: limits.max_buffer_size,
            storage_alignment: u64::from(limits.min_storage_buffer_offset_alignment.max(1)),
            copy_alignment: 4,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct BindWindow<'a> {
    pub(crate) buffer: &'a wgpu::Buffer,
    pub(crate) binding_offset: u64,
    pub(crate) binding_size: u64,
    pub(crate) base_element_in_binding: u32,
}

#[derive(Debug, Clone)]
pub(crate) struct WindowSchedule<'a> {
    windows: Vec<BindWindow<'a>>,
}

impl<'a> WindowSchedule<'a> {
    pub(crate) fn new(windows: Vec<BindWindow<'a>>) -> Self {
        Self { windows }
    }
}

pub(crate) struct WindowScheduler {
    limits: SchedulerLimits,
}

impl WindowScheduler {
    pub(crate) fn new(limits: SchedulerLimits) -> Self {
        Self { limits }
    }

    pub(crate) fn for_device(device: &wgpu::Device) -> Self {
        Self::new(SchedulerLimits::from_device(device))
    }

    pub(crate) fn limits(&self) -> SchedulerLimits {
        self.limits
    }

    pub(crate) fn bind_logical_io<'a>(
        &self,
        logical: FftLogicalView<'a>,
        format: FftEndpointFormat,
        logical_elements_per_batch: u64,
        batch: u64,
    ) -> Result<BoundLogicalIo<'a>> {
        BoundLogicalIo::bind(
            logical,
            format,
            logical_elements_per_batch,
            batch,
            self.limits.storage_alignment,
        )
    }

    pub(crate) fn bind_element_window<'a>(
        &self,
        view: &BufferView<'a>,
        first_element: u64,
        span_elements: u64,
        format: ElementFormat,
    ) -> Result<(BufferView<'a>, u32)> {
        let mut windows = self
            .schedule_element_window(view, first_element, span_elements, format)?
            .windows;
        let binding = windows.pop().ok_or(FftError::WindowScheduleUnsupported {
            reason: "storage binding schedule did not produce a window",
            requested_bytes: 0,
            max_bind_bytes: self.limits.max_storage_buffer_binding_size,
        })?;
        let binding_view =
            BufferView::new(binding.buffer, binding.binding_offset, binding.binding_size)?;
        Ok((binding_view, binding.base_element_in_binding))
    }

    pub(crate) fn storage_binding_resource<'a>(
        &self,
        view: &BufferView<'a>,
        format: ElementFormat,
    ) -> Result<wgpu::BindingResource<'a>> {
        let element_count = exact_element_count(view.size(), format)?;
        let range = self.single_element_range(view, 0, element_count, format)?;
        if range.offset_bytes % self.limits.storage_alignment != 0 {
            return Err(FftError::BufferViewOffsetUnaligned {
                offset: range.offset_bytes,
                alignment: self.limits.storage_alignment,
            });
        }
        let binding = self.bind_window_for_range(range, format)?;
        if binding.base_element_in_binding != 0 {
            return Err(FftError::BufferViewOffsetUnaligned {
                offset: range.offset_bytes,
                alignment: self.limits.storage_alignment,
            });
        }
        Ok(wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer: binding.buffer,
            offset: binding.binding_offset,
            size: wgpu::BufferSize::new(binding.binding_size),
        }))
    }

    pub(crate) fn schedule_element_window<'a>(
        &self,
        view: &BufferView<'a>,
        first_element: u64,
        span_elements: u64,
        format: ElementFormat,
    ) -> Result<WindowSchedule<'a>> {
        let range = self.single_element_range(view, first_element, span_elements, format)?;
        Ok(WindowSchedule::new(vec![
            self.bind_window_for_range(range, format)?
        ]))
    }

    pub(crate) fn storage_window_fits(
        &self,
        view: &BufferView<'_>,
        first_element: u64,
        span_elements: u64,
        format: ElementFormat,
    ) -> Result<bool> {
        let range = self.single_element_range(view, first_element, span_elements, format)?;
        let binding = self.bind_window_for_range(range, format)?;
        Ok(binding.binding_size <= self.limits.max_storage_buffer_binding_size)
    }

    pub(crate) fn validate_copy_ranges<'a>(
        &self,
        view: &BufferView<'a>,
        offset_bytes: u64,
        size_bytes: u64,
    ) -> Result<Vec<BufferRange<'a>>> {
        let ranges = view.copy_ranges(offset_bytes, size_bytes)?;
        for range in &ranges {
            if range.size_bytes > self.limits.max_buffer_size {
                return Err(FftError::WindowScheduleUnsupported {
                    reason: "copy range exceeds max buffer size",
                    requested_bytes: range.size_bytes,
                    max_bind_bytes: self.limits.max_buffer_size,
                });
            }
            if range.offset_bytes % self.limits.copy_alignment != 0
                || range.size_bytes % self.limits.copy_alignment != 0
            {
                return Err(FftError::BufferViewCopyUnaligned {
                    offset: range.offset_bytes,
                    size: range.size_bytes,
                    alignment: self.limits.copy_alignment,
                });
            }
        }
        Ok(ranges)
    }

    fn single_element_range<'a>(
        &self,
        view: &BufferView<'a>,
        first_element: u64,
        span_elements: u64,
        format: ElementFormat,
    ) -> Result<BufferRange<'a>> {
        let element_bytes = format.bytes_per_element();
        let offset = first_element
            .checked_mul(element_bytes)
            .ok_or(FftError::LengthTooLarge { len: usize::MAX })?;
        let size = span_elements
            .checked_mul(element_bytes)
            .ok_or(FftError::LengthTooLarge { len: usize::MAX })?;
        let ranges = view.ranges(offset, size)?;
        if ranges.len() != 1 {
            return Err(FftError::SegmentedBufferViewUnsupported {
                usage: "large graph storage window",
            });
        }
        Ok(ranges[0])
    }

    fn bind_window_for_range<'a>(
        &self,
        range: BufferRange<'a>,
        format: ElementFormat,
    ) -> Result<BindWindow<'a>> {
        if !range.buffer.usage().contains(wgpu::BufferUsages::STORAGE) {
            return Err(FftError::BufferViewMissingUsage { usage: "STORAGE" });
        }
        let binding_offset = align_down(range.offset_bytes, self.limits.storage_alignment);
        let leading_bytes = range
            .offset_bytes
            .checked_sub(binding_offset)
            .ok_or(FftError::LengthTooLarge { len: usize::MAX })?;
        let element_bytes = format.bytes_per_element();
        if leading_bytes % element_bytes != 0 {
            return Err(FftError::BufferViewOffsetUnaligned {
                offset: range.offset_bytes,
                alignment: element_bytes,
            });
        }
        let binding_size = leading_bytes
            .checked_add(range.size_bytes)
            .ok_or(FftError::LengthTooLarge { len: usize::MAX })?;
        if binding_size > self.limits.max_storage_buffer_binding_size {
            return Err(FftError::WindowScheduleUnsupported {
                reason: "storage binding window exceeds device limits",
                requested_bytes: binding_size,
                max_bind_bytes: self.limits.max_storage_buffer_binding_size,
            });
        }
        let base_element_in_binding = (leading_bytes / element_bytes).try_into().map_err(|_| {
            FftError::BufferLayoutTooLarge {
                value: leading_bytes / element_bytes,
                limit: u64::from(u32::MAX),
            }
        })?;
        Ok(BindWindow {
            buffer: range.buffer,
            binding_offset,
            binding_size,
            base_element_in_binding,
        })
    }
}

pub(crate) fn strided_span_elements(count: u64, stride: u64) -> Result<u64> {
    if count == 0 {
        return Ok(0);
    }
    (count - 1)
        .checked_mul(stride)
        .and_then(|value| value.checked_add(1))
        .ok_or(FftError::LengthTooLarge { len: usize::MAX })
}

fn exact_element_count(size_bytes: u64, format: ElementFormat) -> Result<u64> {
    let element_bytes = format.bytes_per_element();
    if !size_bytes.is_multiple_of(element_bytes) {
        return Err(FftError::BufferViewCopyUnaligned {
            offset: 0,
            size: size_bytes,
            alignment: element_bytes,
        });
    }
    Ok(size_bytes / element_bytes)
}

#[cfg(test)]
pub(crate) fn c2c_strided_physical_span_bytes(
    layout: crate::runtime::buffer_view::BufferLayout,
    logical_per_batch: u64,
    batch: u64,
) -> Result<u64> {
    layout
        .required_complex_span(logical_per_batch, batch)?
        .checked_mul(8)
        .ok_or(FftError::LengthTooLarge { len: usize::MAX })
}

fn align_down(value: u64, alignment: u64) -> u64 {
    value - value % alignment
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::buffer_view::BufferLayout;

    fn scheduler() -> WindowScheduler {
        WindowScheduler::new(SchedulerLimits {
            max_storage_buffer_binding_size: 256,
            max_buffer_size: 1024,
            storage_alignment: 256,
            copy_alignment: 4,
        })
    }

    #[test]
    fn strided_span_counts_last_addressed_element() {
        assert_eq!(strided_span_elements(0, 4).unwrap(), 0);
        assert_eq!(strided_span_elements(5, 3).unwrap(), 13);
    }

    #[test]
    fn c2c_layout_span_rejects_too_small_batch_stride() {
        let layout = BufferLayout::new(0, 2).unwrap().with_batch_stride(3);
        assert!(matches!(
            c2c_strided_physical_span_bytes(layout, 4, 2),
            Err(FftError::BufferLayoutBatchStrideTooSmall { .. })
        ));
    }

    #[test]
    fn scheduler_limits_are_preserved() {
        let limits = scheduler().limits();
        assert_eq!(limits.max_storage_buffer_binding_size, 256);
        assert_eq!(limits.storage_alignment, 256);
    }

    #[test]
    fn route_modules_do_not_construct_raw_storage_bindings() {
        for (path, source) in route_source_files() {
            assert!(
                !source.contains("BindingResource::Buffer"),
                "{path} constructs a raw buffer binding instead of WindowScheduler"
            );
            assert!(
                !source.contains("BufferBinding {"),
                "{path} constructs a raw buffer binding instead of WindowScheduler"
            );
        }
    }

    #[test]
    fn route_entire_buffer_bindings_are_uniform_params_only() {
        for (path, source) in route_source_files() {
            for (line_index, line) in source.lines().enumerate() {
                if line.contains(".as_entire_binding()") {
                    assert!(
                        line.contains("params_buffer") || line.contains("uniform_buffer"),
                        "{path}:{} uses as_entire_binding outside a params/uniform binding: {line}",
                        line_index + 1
                    );
                }
            }
        }
    }

    #[test]
    fn route_modules_do_not_issue_raw_copy_commands() {
        for (path, source) in route_source_files() {
            if path == "runtime/stage_executor.rs" {
                continue;
            }
            assert!(
                !source.contains(".copy_buffer_to_buffer("),
                "{path} issues a raw copy command instead of using StageExecutor copy windows"
            );
        }
    }

    fn route_source_files() -> [(&'static str, &'static str); 13] {
        [
            ("runtime/axis_plan.rs", include_str!("axis_plan.rs")),
            (
                "runtime/bluestein_axis.rs",
                include_str!("bluestein_axis.rs"),
            ),
            ("runtime/c2c.rs", include_str!("c2c.rs")),
            ("runtime/four_step.rs", include_str!("four_step.rs")),
            ("runtime/large_bridge.rs", include_str!("large_bridge.rs")),
            ("runtime/large_chunk.rs", include_str!("large_chunk.rs")),
            ("runtime/large_graph.rs", include_str!("large_graph.rs")),
            ("runtime/rader_axis.rs", include_str!("rader_axis.rs")),
            ("runtime/real.rs", include_str!("real.rs")),
            (
                "runtime/segmented_volume.rs",
                include_str!("segmented_volume.rs"),
            ),
            (
                "runtime/smooth_decompose.rs",
                include_str!("smooth_decompose.rs"),
            ),
            (
                "runtime/stage_executor.rs",
                include_str!("stage_executor.rs"),
            ),
            ("plan.rs", include_str!("../plan.rs")),
        ]
    }
}
