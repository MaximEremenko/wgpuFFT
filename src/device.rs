/// Minimal native `wgpu` context helper for examples and opt-in integration tests.
pub struct GpuContext {
    pub instance: wgpu::Instance,
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

pub async fn request_default_device() -> Option<GpuContext> {
    let instance = wgpu::Instance::new(default_instance_descriptor());
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        })
        .await
        .ok()?;

    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("wgpu_fft.device"),
            required_features: wgpu::Features::empty(),
            required_limits: adapter.limits(),
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: wgpu::MemoryHints::MemoryUsage,
            trace: wgpu::Trace::Off,
        })
        .await
        .ok()?;

    Some(GpuContext {
        instance,
        adapter,
        device,
        queue,
    })
}

fn default_instance_descriptor() -> wgpu::InstanceDescriptor {
    let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
    descriptor.backends = default_native_backends();
    descriptor.with_env()
}

fn default_native_backends() -> wgpu::Backends {
    wgpu::Backends::VULKAN | wgpu::Backends::METAL | wgpu::Backends::DX12
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_native_backends_exclude_gl() {
        assert!(!default_native_backends().contains(wgpu::Backends::GL));
    }
}
