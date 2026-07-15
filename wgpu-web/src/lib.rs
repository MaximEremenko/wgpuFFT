//! Minimal browser-facing `wasm-bindgen` surface for `wgpu-fft`.
//!
//! The wrapper deliberately deals in raw byte arrays so JavaScript callers can
//! choose the storage representation required by `F32`, `Df64`, or a future
//! precision without an extra host-side conversion layer.

use std::rc::Rc;

use futures_channel::oneshot;
use wasm_bindgen::prelude::*;
use wgpu::util::DeviceExt;
use wgpu_fft::df64_canary::{validate_df64_invariants, DF64_CANARY_WORD_COUNT};
use wgpu_fft::{
    clear_thread_local_pipeline_cache, export_pipeline_cache_snapshot,
    import_pipeline_cache_snapshot, C2cRoute, FftConfig, FftDirection, FftPlan, FftPrecision,
    Normalization, PipelineCacheSnapshot,
};

struct Runtime {
    _instance: wgpu::Instance,
    device: wgpu::Device,
    queue: wgpu::Queue,
    adapter_name: String,
    adapter_vendor: u32,
    adapter_device: u32,
    adapter_device_type: String,
    adapter_driver: String,
    adapter_driver_info: String,
    backend: String,
}

impl Drop for Runtime {
    fn drop(&mut self) {
        clear_thread_local_pipeline_cache(&self.device);
    }
}

/// Browser scalar precision requested for a C2C plan.
#[wasm_bindgen]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WebFftPrecision {
    F32,
    Df64,
    F64,
}

/// Browser C2C direction.
#[wasm_bindgen]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WebFftDirection {
    Forward,
    Inverse,
}

/// Browser C2C normalization policy.
#[wasm_bindgen]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WebFftNormalization {
    None,
    Forward,
    Inverse,
    Orthogonal,
}

/// Initialized WebGPU context shared by plans and buffers.
#[wasm_bindgen]
pub struct WgpuFft {
    runtime: Rc<Runtime>,
    df64_available: bool,
    df64_canary_words: usize,
    df64_canary_error: Option<String>,
}

/// Reusable browser C2C plan.
#[wasm_bindgen]
pub struct WgpuFftPlan {
    runtime: Rc<Runtime>,
    plan: FftPlan,
}

/// GPU-resident caller-owned byte buffer.
#[wasm_bindgen]
pub struct WgpuFftBuffer {
    runtime: Rc<Runtime>,
    buffer: wgpu::Buffer,
    size: u64,
}

async fn initialize() -> Result<WgpuFft, JsValue> {
    let mut instance_descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
    instance_descriptor.backends = wgpu::Backends::BROWSER_WEBGPU;
    let instance = wgpu::Instance::new(instance_descriptor);
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        })
        .await
        .map_err(|error| js_error(format!("WebGPU adapter request failed: {error}")))?;
    let info = adapter.get_info();
    let maximum_descriptor = wgpu::DeviceDescriptor {
        label: Some("wgpu_web.device"),
        required_features: wgpu::Features::empty(),
        required_limits: adapter.limits(),
        experimental_features: wgpu::ExperimentalFeatures::disabled(),
        memory_hints: wgpu::MemoryHints::MemoryUsage,
        trace: wgpu::Trace::Off,
    };
    let maximum_request = adapter.request_device(&maximum_descriptor).await;
    let (device, queue) = match maximum_request {
        Ok(device) => device,
        Err(maximum_error) => adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("wgpu_web.default_device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                memory_hints: wgpu::MemoryHints::MemoryUsage,
                trace: wgpu::Trace::Off,
            })
            .await
            .map_err(|default_error| {
                js_error(format!(
                    "WebGPU device request failed at adapter limits ({maximum_error}) and defaults ({default_error})"
                ))
            })?,
    };

    let (df64_available, df64_canary_words, df64_canary_error) =
        match validate_df64_invariants(&device, &queue).await {
            Ok(report) if report.exact_words == DF64_CANARY_WORD_COUNT => {
                (true, report.exact_words, None)
            }
            Ok(report) => (
                false,
                report.exact_words,
                Some(format!(
                    "df64 canary returned {} exact words instead of {}",
                    report.exact_words, DF64_CANARY_WORD_COUNT
                )),
            ),
            Err(error) => (false, 0, Some(error.to_string())),
        };

    Ok(WgpuFft {
        runtime: Rc::new(Runtime {
            _instance: instance,
            device,
            queue,
            adapter_name: info.name,
            adapter_vendor: info.vendor,
            adapter_device: info.device,
            adapter_device_type: format!("{:?}", info.device_type),
            adapter_driver: info.driver,
            adapter_driver_info: info.driver_info,
            backend: format!("{:?}", info.backend),
        }),
        df64_available,
        df64_canary_words,
        df64_canary_error,
    })
}

#[wasm_bindgen]
impl WgpuFft {
    /// Acquires browser WebGPU and runs all 96 df64 invariant words before
    /// returning a usable context. A failed canary disables only `Df64`; `F32`
    /// remains available and the failure text is exposed for diagnostics.
    #[wasm_bindgen(js_name = init)]
    pub async fn init() -> Result<WgpuFft, JsValue> {
        initialize().await
    }

    /// Adapter name reported by the browser.
    #[wasm_bindgen(getter, js_name = adapterName)]
    pub fn adapter_name(&self) -> String {
        self.runtime.adapter_name.clone()
    }

    #[wasm_bindgen(getter, js_name = adapterVendor)]
    pub fn adapter_vendor(&self) -> u32 {
        self.runtime.adapter_vendor
    }

    #[wasm_bindgen(getter, js_name = adapterDevice)]
    pub fn adapter_device(&self) -> u32 {
        self.runtime.adapter_device
    }

    #[wasm_bindgen(getter, js_name = adapterDeviceType)]
    pub fn adapter_device_type(&self) -> String {
        self.runtime.adapter_device_type.clone()
    }

    #[wasm_bindgen(getter, js_name = adapterDriver)]
    pub fn adapter_driver(&self) -> String {
        self.runtime.adapter_driver.clone()
    }

    #[wasm_bindgen(getter, js_name = adapterDriverInfo)]
    pub fn adapter_driver_info(&self) -> String {
        self.runtime.adapter_driver_info.clone()
    }

    /// Active wgpu backend, expected to be `BrowserWebGpu` in a browser.
    #[wasm_bindgen(getter)]
    pub fn backend(&self) -> String {
        self.runtime.backend.clone()
    }

    #[wasm_bindgen(getter, js_name = maxBufferSize)]
    pub fn max_buffer_size(&self) -> f64 {
        self.runtime.device.limits().max_buffer_size as f64
    }

    #[wasm_bindgen(getter, js_name = maxStorageBufferBindingSize)]
    pub fn max_storage_buffer_binding_size(&self) -> f64 {
        self.runtime.device.limits().max_storage_buffer_binding_size as f64
    }

    #[wasm_bindgen(getter, js_name = maxComputeWorkgroupStorageSize)]
    pub fn max_compute_workgroup_storage_size(&self) -> u32 {
        self.runtime
            .device
            .limits()
            .max_compute_workgroup_storage_size
    }

    /// Whether the browser's Tint/backend compiler passed the exact df64
    /// arithmetic invariant suite.
    #[wasm_bindgen(getter, js_name = df64Available)]
    pub fn df64_available(&self) -> bool {
        self.df64_available
    }

    /// Exact canary words observed on success (currently 96).
    #[wasm_bindgen(getter, js_name = df64CanaryWords)]
    pub fn df64_canary_words(&self) -> u32 {
        self.df64_canary_words as u32
    }

    /// Canary failure text when `df64Available` is false.
    #[wasm_bindgen(getter, js_name = df64CanaryError)]
    pub fn df64_canary_error(&self) -> Option<String> {
        self.df64_canary_error.clone()
    }

    /// Builds a reusable 1D C2C plan. Native `F64` is deliberately forwarded
    /// to `wgpu-fft`, which returns its structured browser unsupported error.
    #[wasm_bindgen(js_name = createPlan)]
    pub async fn plan_c2c(
        &self,
        len: u32,
        batch: u32,
        direction: WebFftDirection,
        precision: WebFftPrecision,
        normalization: WebFftNormalization,
    ) -> Result<WgpuFftPlan, JsValue> {
        if precision == WebFftPrecision::Df64 && !self.df64_available {
            let reason = self
                .df64_canary_error
                .as_deref()
                .unwrap_or("the 96-word browser invariant suite did not pass");
            return Err(js_error(format!(
                "Df64 is disabled for this browser/compiler: {reason}"
            )));
        }
        let config = FftConfig::new(len as usize)
            .with_batch(batch as usize)
            .with_direction(direction.into())
            .with_precision(precision.into())
            .with_normalization(normalization.into());
        let plan = FftPlan::c2c_checked(&self.runtime.device, &self.runtime.queue, config)
            .await
            .map_err(|error| js_error(format!("C2C plan creation failed: {error}")))?;
        Ok(WgpuFftPlan {
            runtime: Rc::clone(&self.runtime),
            plan,
        })
    }

    /// Uploads f32 storage words from a JavaScript `Float32Array`. F32 complex
    /// inputs use two words per value; Df64 inputs use four.
    pub fn upload(&self, words: &[f32]) -> Result<WgpuFftBuffer, JsValue> {
        let bytes = bytemuck::cast_slice(words);
        validate_byte_len(
            bytes.len() as u64,
            self.runtime.device.limits().max_buffer_size,
        )?;
        let buffer = self
            .runtime
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("wgpu_web.upload"),
                contents: bytes,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            });
        Ok(WgpuFftBuffer {
            runtime: Rc::clone(&self.runtime),
            buffer,
            size: bytes.len() as u64,
        })
    }

    /// Allocates an uninitialized caller-owned GPU buffer for transform output.
    #[wasm_bindgen(js_name = createBuffer)]
    pub fn create_buffer(&self, byte_len: u32) -> Result<WgpuFftBuffer, JsValue> {
        let size = u64::from(byte_len);
        validate_byte_len(size, self.runtime.device.limits().max_buffer_size)?;
        let buffer = self.runtime.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("wgpu_web.buffer"),
            size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Ok(WgpuFftBuffer {
            runtime: Rc::clone(&self.runtime),
            buffer,
            size,
        })
    }

    /// Downloads a GPU buffer through a temporary map-readable staging buffer.
    pub async fn download(&self, source: &WgpuFftBuffer) -> Result<Vec<u8>, JsValue> {
        ensure_same_runtime(&self.runtime, &source.runtime, "download")?;
        let readback = self.runtime.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("wgpu_web.download"),
            size: source.size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder =
            self.runtime
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("wgpu_web.download.encoder"),
                });
        encoder.copy_buffer_to_buffer(&source.buffer, 0, &readback, 0, source.size);
        self.runtime.queue.submit([encoder.finish()]);

        let (sender, receiver) = oneshot::channel();
        readback
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                let _ = sender.send(result);
            });
        receiver
            .await
            .map_err(|_| js_error("download mapping callback was dropped"))?
            .map_err(|error| js_error(format!("download mapping failed: {error}")))?;
        let mapped = readback.slice(..).get_mapped_range();
        let bytes = mapped.to_vec();
        drop(mapped);
        readback.unmap();
        Ok(bytes)
    }

    /// Serializes the current source/pipeline prewarm cache for browser
    /// persistence. This is not a driver-binary cache.
    #[wasm_bindgen(js_name = exportSnapshot)]
    pub fn export_snapshot(&self) -> Result<String, JsValue> {
        export_pipeline_cache_snapshot(&self.runtime.device)
            .to_json()
            .map_err(|error| js_error(format!("pipeline snapshot export failed: {error}")))
    }

    /// Validates and imports a source/pipeline prewarm cache, returning the
    /// normalized snapshot JSON accepted by `wgpu-fft`.
    #[wasm_bindgen(js_name = importSnapshot)]
    pub async fn import_snapshot(&self, json: &str) -> Result<String, JsValue> {
        let snapshot = PipelineCacheSnapshot::from_json(json)
            .map_err(|error| js_error(format!("pipeline snapshot parse failed: {error}")))?;
        let out_of_memory = self
            .runtime
            .device
            .push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let internal = self
            .runtime
            .device
            .push_error_scope(wgpu::ErrorFilter::Internal);
        let validation = self
            .runtime
            .device
            .push_error_scope(wgpu::ErrorFilter::Validation);
        let imported = import_pipeline_cache_snapshot(&self.runtime.device, &snapshot);
        let validation_pop = validation.pop();
        let internal_pop = internal.pop();
        let out_of_memory_pop = out_of_memory.pop();
        let validation_error = validation_pop.await;
        let internal_error = internal_pop.await;
        let out_of_memory_error = out_of_memory_pop.await;
        let scoped_error = validation_error
            .map(|error| ("validation", error))
            .or_else(|| internal_error.map(|error| ("internal", error)))
            .or_else(|| out_of_memory_error.map(|error| ("out-of-memory", error)));
        if let Some((kind, error)) = scoped_error {
            clear_thread_local_pipeline_cache(&self.runtime.device);
            return Err(js_error(format!(
                "pipeline snapshot import failed during {kind}: {error}"
            )));
        }
        imported
            .to_json()
            .map_err(|error| js_error(format!("pipeline snapshot import failed: {error}")))
    }
}

#[wasm_bindgen]
impl WgpuFftPlan {
    #[wasm_bindgen(getter, js_name = inputBytes)]
    pub fn input_bytes(&self) -> f64 {
        self.plan.required_input_buffer_size_bytes() as f64
    }

    #[wasm_bindgen(getter, js_name = outputBytes)]
    pub fn output_bytes(&self) -> f64 {
        self.plan.required_output_buffer_size_bytes() as f64
    }

    #[wasm_bindgen(getter, js_name = workspaceBytes)]
    pub fn workspace_bytes(&self) -> f64 {
        self.plan.workspace_size_bytes() as f64
    }

    #[wasm_bindgen(getter)]
    pub fn route(&self) -> String {
        route_name(self.plan.route()).to_owned()
    }

    /// Encodes, submits, and asynchronously waits for queue completion. Reusing
    /// an existing output buffer makes this span suitable for browser timing:
    /// plan creation, upload, and allocation stay outside the awaited region.
    pub async fn execute(
        &self,
        input: &WgpuFftBuffer,
        output: &WgpuFftBuffer,
    ) -> Result<(), JsValue> {
        ensure_same_runtime(&self.runtime, &input.runtime, "FFT input")?;
        ensure_same_runtime(&self.runtime, &output.runtime, "FFT output")?;
        if input.buffer == output.buffer {
            return Err(js_error(
                "out-of-place FFT input and output must be distinct buffers",
            ));
        }
        if input.size < self.plan.required_input_buffer_size_bytes() {
            return Err(js_error(format!(
                "FFT input has {} bytes but the plan requires {}",
                input.size,
                self.plan.required_input_buffer_size_bytes()
            )));
        }
        if output.size < self.plan.required_output_buffer_size_bytes() {
            return Err(js_error(format!(
                "FFT output has {} bytes but the plan requires {}",
                output.size,
                self.plan.required_output_buffer_size_bytes()
            )));
        }
        let mut encoder =
            self.runtime
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("wgpu_web.fft.encoder"),
                });
        self.plan
            .execute_checked(
                &self.runtime.device,
                &mut encoder,
                &input.buffer,
                &output.buffer,
            )
            .map_err(|error| js_error(format!("FFT execution failed: {error}")))?;
        self.runtime.queue.submit([encoder.finish()]);
        let (sender, receiver) = oneshot::channel();
        self.runtime.queue.on_submitted_work_done(move || {
            let _ = sender.send(());
        });
        receiver
            .await
            .map_err(|_| js_error("queue completion callback was dropped"))?;
        Ok(())
    }
}

#[wasm_bindgen]
impl WgpuFftBuffer {
    #[wasm_bindgen(getter, js_name = byteLength)]
    pub fn byte_length(&self) -> f64 {
        self.size as f64
    }
}

impl From<WebFftPrecision> for FftPrecision {
    fn from(value: WebFftPrecision) -> Self {
        match value {
            WebFftPrecision::F32 => Self::F32,
            WebFftPrecision::Df64 => Self::Df64,
            WebFftPrecision::F64 => Self::F64,
        }
    }
}

impl From<WebFftDirection> for FftDirection {
    fn from(value: WebFftDirection) -> Self {
        match value {
            WebFftDirection::Forward => Self::Forward,
            WebFftDirection::Inverse => Self::Inverse,
        }
    }
}

impl From<WebFftNormalization> for Normalization {
    fn from(value: WebFftNormalization) -> Self {
        match value {
            WebFftNormalization::None => Self::None,
            WebFftNormalization::Forward => Self::Forward,
            WebFftNormalization::Inverse => Self::Inverse,
            WebFftNormalization::Orthogonal => Self::Orthogonal,
        }
    }
}

fn validate_byte_len(size: u64, max_buffer_size: u64) -> Result<(), JsValue> {
    if size == 0 {
        return Err(js_error("GPU buffers must contain at least one byte"));
    }
    if !size.is_multiple_of(wgpu::COPY_BUFFER_ALIGNMENT) {
        return Err(js_error(format!(
            "GPU byte length {size} is not aligned to {} bytes",
            wgpu::COPY_BUFFER_ALIGNMENT
        )));
    }
    if size > max_buffer_size {
        return Err(js_error(format!(
            "GPU byte length {size} exceeds the device maxBufferSize of {max_buffer_size}"
        )));
    }
    Ok(())
}

fn ensure_same_runtime(
    expected: &Rc<Runtime>,
    actual: &Rc<Runtime>,
    role: &str,
) -> Result<(), JsValue> {
    if Rc::ptr_eq(expected, actual) {
        Ok(())
    } else {
        Err(js_error(format!(
            "{role} belongs to a different WebGPU device"
        )))
    }
}

fn route_name(route: C2cRoute) -> &'static str {
    match route {
        C2cRoute::DirectDft => "direct-dft",
        C2cRoute::MixedRadix => "mixed-radix",
        C2cRoute::Rader => "rader",
        C2cRoute::Bluestein => "bluestein",
        C2cRoute::AxisSequence => "axis-sequence",
    }
}

fn js_error(message: impl AsRef<str>) -> JsValue {
    js_sys::Error::new(message.as_ref()).into()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn browser_enums_map_to_core_configuration_values() {
        assert_eq!(FftPrecision::from(WebFftPrecision::F32), FftPrecision::F32);
        assert_eq!(
            FftPrecision::from(WebFftPrecision::Df64),
            FftPrecision::Df64
        );
        assert_eq!(FftPrecision::from(WebFftPrecision::F64), FftPrecision::F64);
        assert_eq!(
            FftDirection::from(WebFftDirection::Inverse),
            FftDirection::Inverse
        );
        assert_eq!(
            Normalization::from(WebFftNormalization::Orthogonal),
            Normalization::Orthogonal
        );
    }
}
