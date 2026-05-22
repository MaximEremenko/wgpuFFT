use crate::config::FftPrecision;

/// Minimal native `wgpu` context helper for examples and opt-in integration tests.
pub struct GpuContext {
    pub instance: wgpu::Instance,
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

impl GpuContext {
    pub fn supports_precision(&self, precision: FftPrecision) -> bool {
        device_supports_precision(&self.device, precision)
    }
}

/// Returns whether the features enabled on `device` support `precision`.
pub fn device_supports_precision(device: &wgpu::Device, precision: FftPrecision) -> bool {
    match precision {
        FftPrecision::F32 | FftPrecision::Df64 => true,
        FftPrecision::F64 => device.features().contains(wgpu::Features::SHADER_F64),
    }
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

    let required_features = precision_features_supported_by_adapter(adapter.features());
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("wgpu_fft.device"),
            required_features,
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

fn precision_features_supported_by_adapter(features: wgpu::Features) -> wgpu::Features {
    features & wgpu::Features::SHADER_F64
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

    #[test]
    fn default_device_only_enables_supported_precision_features() {
        assert_eq!(
            precision_features_supported_by_adapter(wgpu::Features::empty()),
            wgpu::Features::empty()
        );
        assert_eq!(
            precision_features_supported_by_adapter(wgpu::Features::SHADER_F64),
            wgpu::Features::SHADER_F64
        );
    }
}
