/// Records one FFT execution, batching consecutive dispatches into a single
/// compute pass.
///
/// On an RTX 5090, opening a compute pass per dispatch costs about 5 µs of GPU
/// time on Vulkan and 3 µs on DX12, against about 1 µs per dispatch inside a
/// shared pass. wgpu still orders dependent dispatches within one pass, so
/// every stage sees the previous stage's writes. Copies end the open pass.
pub(crate) struct CommandRecorder<'a> {
    encoder: &'a mut wgpu::CommandEncoder,
    pass: Option<wgpu::ComputePass<'static>>,
}

impl<'a> CommandRecorder<'a> {
    pub(crate) fn new(encoder: &'a mut wgpu::CommandEncoder) -> Self {
        Self {
            encoder,
            pass: None,
        }
    }

    /// Returns the shared compute pass, opening it on first use.
    pub(crate) fn pass(&mut self) -> &mut wgpu::ComputePass<'static> {
        let encoder = &mut *self.encoder;
        self.pass.get_or_insert_with(|| {
            encoder
                .begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("wgpu_fft.pass"),
                    timestamp_writes: None,
                })
                .forget_lifetime()
        })
    }

    /// Returns the command encoder for copies, ending any open compute pass.
    pub(crate) fn encoder(&mut self) -> &mut wgpu::CommandEncoder {
        self.pass = None;
        self.encoder
    }
}
