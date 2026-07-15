use crate::config::FftPrecision;

/// Minimal platform `wgpu` context helper for examples and opt-in integration tests.
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

/// `WGPU_FFT_FORCE_FALLBACK=1` routes the whole stack onto the platform's
/// software adapter (WARP on Windows, lavapipe on Linux): the same WGSL
/// pipelines execute on the CPU. Native builds only; ignored in wasm.
fn force_fallback_from_env() -> bool {
    #[cfg(not(target_arch = "wasm32"))]
    {
        matches!(
            std::env::var("WGPU_FFT_FORCE_FALLBACK").as_deref(),
            Ok("1") | Ok("true") | Ok("yes")
        )
    }
    #[cfg(target_arch = "wasm32")]
    {
        false
    }
}

pub async fn request_default_device() -> Option<GpuContext> {
    let instance = wgpu::Instance::new(default_instance_descriptor());
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: force_fallback_from_env(),
            apply_limit_buckets: false,
            compatible_surface: None,
        })
        .await
        .ok()?;

    let required_features = precision_features_supported_by_adapter(adapter.features());
    let maximum_descriptor = wgpu::DeviceDescriptor {
        label: Some("wgpu_fft.device"),
        required_features,
        required_limits: adapter.limits(),
        experimental_features: wgpu::ExperimentalFeatures::disabled(),
        memory_hints: wgpu::MemoryHints::MemoryUsage,
        trace: wgpu::Trace::Off,
    };
    let maximum_request = adapter.request_device(&maximum_descriptor).await;
    #[cfg(not(target_arch = "wasm32"))]
    let (device, queue) = maximum_request.ok()?;
    #[cfg(target_arch = "wasm32")]
    let (device, queue) = match maximum_request {
        Ok(device) => device,
        Err(_) => adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("wgpu_fft.browser_default_device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                memory_hints: wgpu::MemoryHints::MemoryUsage,
                trace: wgpu::Trace::Off,
            })
            .await
            .ok()?,
    };

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
    descriptor.backends = default_backends();
    descriptor.with_env()
}

#[cfg(not(target_arch = "wasm32"))]
fn default_backends() -> wgpu::Backends {
    wgpu::Backends::VULKAN | wgpu::Backends::METAL | wgpu::Backends::DX12
}

#[cfg(target_arch = "wasm32")]
fn default_backends() -> wgpu::Backends {
    wgpu::Backends::BROWSER_WEBGPU
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn default_native_backends_exclude_gl() {
        assert!(!default_backends().contains(wgpu::Backends::GL));
    }

    #[test]
    #[cfg(target_arch = "wasm32")]
    fn default_browser_backend_uses_webgpu() {
        assert_eq!(default_backends(), wgpu::Backends::BROWSER_WEBGPU);
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
