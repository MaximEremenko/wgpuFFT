use std::sync::Mutex;

/// Plan-owned scratch for type-1 point binning.
///
/// The bin arrays are fixed by the fine-grid shape. The sorted-index buffer is
/// grown on demand because the point count is supplied at execution time. A
/// cloned `wgpu::Buffer` keeps an older allocation alive when a command buffer
/// has already captured it and a later encode grows the cache.
pub(crate) struct Type1ScratchBuffers {
    pub(crate) bin_counts: wgpu::Buffer,
    pub(crate) bin_cursors: wgpu::Buffer,
    pub(crate) bin_offsets: wgpu::Buffer,
    device: wgpu::Device,
    sorted_indices: Mutex<GrowableBuffer>,
    label_prefix: &'static str,
}

struct GrowableBuffer {
    buffer: Option<wgpu::Buffer>,
    capacity_bytes: u64,
}

impl Type1ScratchBuffers {
    pub(crate) fn new(
        device: &wgpu::Device,
        label_prefix: &'static str,
        count_bytes: u64,
        offset_bytes: u64,
    ) -> Self {
        let atomic_usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let count_label = format!("{label_prefix}.bin_counts");
        let cursor_label = format!("{label_prefix}.bin_cursors");
        let offset_label = format!("{label_prefix}.bin_offsets");
        Self {
            bin_counts: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&count_label),
                size: count_bytes,
                usage: atomic_usage,
                mapped_at_creation: false,
            }),
            bin_cursors: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&cursor_label),
                size: count_bytes,
                usage: atomic_usage,
                mapped_at_creation: false,
            }),
            bin_offsets: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&offset_label),
                size: offset_bytes,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            }),
            device: device.clone(),
            sorted_indices: Mutex::new(GrowableBuffer {
                buffer: None,
                capacity_bytes: 0,
            }),
            label_prefix,
        }
    }

    pub(crate) fn sorted_indices(&self, required_bytes: u64) -> wgpu::Buffer {
        debug_assert!(required_bytes > 0);
        let mut cached = self
            .sorted_indices
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if cached.capacity_bytes < required_bytes {
            let label = format!("{}.sorted_point_indices", self.label_prefix);
            cached.buffer = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&label),
                size: required_bytes,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            }));
            cached.capacity_bytes = required_bytes;
        }
        cached
            .buffer
            .as_ref()
            .expect("a positive scratch request must allocate a buffer")
            .clone()
    }
}
