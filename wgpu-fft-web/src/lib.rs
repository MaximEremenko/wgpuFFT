//! Browser-facing `wasm-bindgen` surface for `wgpu-fft`: N-dimensional C2C
//! plans on WebGPU, and `cpuFft`, the host-memory `CpuFftPlan` in `f64`, for
//! pages without WebGPU.
//!
//! GPU buffers hold raw words so JavaScript callers choose the storage their
//! precision needs: `F32` complex values are interleaved `re, im`; `Df64`
//! values are `re_hi, re_lo, im_hi, im_lo`. Axis 0 of a shape varies fastest.
//! The structure follows wgpuNUFFT's `wgpu-web`.

use std::rc::Rc;

use futures_channel::oneshot;
use wasm_bindgen::prelude::*;
use wgpu::util::DeviceExt;
use wgpu_fft::math::DoubleFloat;
use wgpu_fft::{
    clear_thread_local_pipeline_cache, validate_df64_invariants, C2cRoute, CpuFftPlan, FftConfig,
    FftDirection, FftPlan, FftPrecision, Normalization, DF64_CANARY_WORD_COUNT,
};

struct Runtime {
    _instance: wgpu::Instance,
    device: wgpu::Device,
    queue: wgpu::Queue,
    adapter_name: String,
    adapter_vendor_name: String,
    adapter_architecture: String,
    backend: String,
}

// wgpu's WebGPU backend leaves dropped objects to the JavaScript garbage
// collector, which cannot see GPU memory pressure. Destroying them releases
// that memory as soon as JavaScript calls `free()`.

impl Drop for Runtime {
    fn drop(&mut self) {
        clear_thread_local_pipeline_cache(&self.device);
        // Every plan and buffer holds this runtime, so nothing can use the
        // device any more.
        self.device.destroy();
    }
}

impl Drop for WgpuFftBuffer {
    fn drop(&mut self) {
        self.buffer.destroy();
    }
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console, js_name = error)]
    fn console_error(message: &str);
}

/// The browser's `GPUAdapterInfo` vendor and architecture, which wgpu does
/// not keep: its adapter name is only the description, which browsers
/// usually withhold.
#[derive(Default)]
struct BrowserAdapterInfo {
    vendor: String,
    architecture: String,
}

/// Asks the browser for the adapter that `initialize` requests, with the same
/// options, and reads its info. Empty when the browser does not say. Works
/// on pages and in workers.
#[cfg(target_arch = "wasm32")]
async fn browser_adapter_info(force_fallback: bool) -> BrowserAdapterInfo {
    async fn query(force_fallback: bool) -> Result<BrowserAdapterInfo, JsValue> {
        let get =
            |target: &JsValue, key: &str| js_sys::Reflect::get(target, &JsValue::from_str(key));
        let gpu = get(&get(&js_sys::global(), "navigator")?, "gpu")?;
        let request_adapter = get(&gpu, "requestAdapter")?.dyn_into::<js_sys::Function>()?;
        let options = js_sys::Object::new();
        js_sys::Reflect::set(
            &options,
            &"powerPreference".into(),
            &"high-performance".into(),
        )?;
        js_sys::Reflect::set(
            &options,
            &"forceFallbackAdapter".into(),
            &JsValue::from_bool(force_fallback),
        )?;
        let request = request_adapter
            .call1(&gpu, &options.into())?
            .dyn_into::<js_sys::Promise>()?;
        let adapter = wasm_bindgen_futures::JsFuture::from(request).await?;
        let info = get(&adapter, "info")?;
        let text = |key| get(&info, key).map(|value| value.as_string().unwrap_or_default());
        Ok(BrowserAdapterInfo {
            vendor: text("vendor")?,
            architecture: text("architecture")?,
        })
    }
    query(force_fallback).await.unwrap_or_default()
}

#[cfg(not(target_arch = "wasm32"))]
async fn browser_adapter_info(_force_fallback: bool) -> BrowserAdapterInfo {
    BrowserAdapterInfo::default()
}

/// Sends Rust panic messages to `console.error`; the browser otherwise reports
/// only an `unreachable` trap. A previously installed hook still runs.
#[cfg(target_arch = "wasm32")]
fn install_panic_hook() {
    static INSTALL: std::sync::Once = std::sync::Once::new();
    INSTALL.call_once(|| {
        let previous = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            console_error(&info.to_string());
            previous(info);
        }));
    });
}

/// Scalar precision of a plan.
#[wasm_bindgen]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WebFftPrecision {
    F32,
    Df64,
    F64,
}

/// Transform direction; the forward kernel is `exp(-2 pi i jk / n)`.
#[wasm_bindgen]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WebFftDirection {
    Forward,
    Inverse,
}

/// Where the `1 / n` scaling goes.
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

/// Reusable N-dimensional C2C plan.
#[wasm_bindgen]
pub struct WgpuFftPlan {
    runtime: Rc<Runtime>,
    plan: FftPlan,
}

/// GPU-resident caller-owned byte buffer. `free()` releases its GPU memory
/// immediately.
#[wasm_bindgen]
pub struct WgpuFftBuffer {
    runtime: Rc<Runtime>,
    buffer: wgpu::Buffer,
    size: u64,
}

async fn initialize(
    request_adapter_maximums: bool,
    force_fallback: bool,
) -> Result<WgpuFft, JsValue> {
    #[cfg(target_arch = "wasm32")]
    install_panic_hook();
    let mut instance_descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
    instance_descriptor.backends = wgpu::Backends::BROWSER_WEBGPU;
    let instance = wgpu::Instance::new(instance_descriptor);
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: force_fallback,
            apply_limit_buckets: false,
            compatible_surface: None,
        })
        .await
        .map_err(|error| js_error(format!("WebGPU adapter request failed: {error}")))?;
    let info = adapter.get_info();
    let browser_info = browser_adapter_info(force_fallback).await;
    let adapter_name = if info.name.is_empty() {
        [
            browser_info.vendor.as_str(),
            browser_info.architecture.as_str(),
        ]
        .into_iter()
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
    } else {
        info.name
    };
    let default_descriptor = || wgpu::DeviceDescriptor {
        label: Some("wgpu_fft_web.default_device"),
        required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits::default(),
        experimental_features: wgpu::ExperimentalFeatures::disabled(),
        memory_hints: wgpu::MemoryHints::MemoryUsage,
        trace: wgpu::Trace::Off,
    };
    let (device, queue) = if request_adapter_maximums {
        let maximum_descriptor = wgpu::DeviceDescriptor {
            label: Some("wgpu_fft_web.device"),
            required_features: wgpu::Features::empty(),
            required_limits: adapter.limits(),
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: wgpu::MemoryHints::MemoryUsage,
            trace: wgpu::Trace::Off,
        };
        match adapter.request_device(&maximum_descriptor).await {
            Ok(device) => device,
            Err(maximum_error) => adapter
                .request_device(&default_descriptor())
                .await
                .map_err(|default_error| {
                    js_error(format!(
                        "WebGPU device request failed at adapter limits ({maximum_error}) and defaults ({default_error})"
                    ))
                })?,
        }
    } else {
        adapter
            .request_device(&default_descriptor())
            .await
            .map_err(|error| {
                js_error(format!(
                    "WebGPU default-limit device request failed: {error}"
                ))
            })?
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
            adapter_name,
            adapter_vendor_name: browser_info.vendor,
            adapter_architecture: browser_info.architecture,
            backend: format!("{:?}", info.backend),
        }),
        df64_available,
        df64_canary_words,
        df64_canary_error,
    })
}

#[wasm_bindgen]
impl WgpuFft {
    /// Acquires browser WebGPU at the adapter's maximum limits (falling back
    /// to the defaults) and runs the 96-word df64 invariant suite. A failed
    /// canary disables only `Df64`; `F32` remains available.
    #[wasm_bindgen(js_name = init)]
    pub async fn init() -> Result<WgpuFft, JsValue> {
        initialize(true, false).await
    }

    /// Initializes on the browser's software fallback adapter (such as
    /// SwiftShader), where no hardware adapter is available.
    #[wasm_bindgen(js_name = initFallback)]
    pub async fn init_fallback() -> Result<WgpuFft, JsValue> {
        initialize(true, true).await
    }

    /// Acquires a featureless device at the WebGPU default limits.
    #[wasm_bindgen(js_name = initWithDefaultLimits)]
    pub async fn init_with_default_limits() -> Result<WgpuFft, JsValue> {
        initialize(false, false).await
    }

    /// The adapter's description or, as browsers usually withhold it, its
    /// vendor and architecture.
    #[wasm_bindgen(getter, js_name = adapterName)]
    pub fn adapter_name(&self) -> String {
        self.runtime.adapter_name.clone()
    }

    #[wasm_bindgen(getter, js_name = adapterVendorName)]
    pub fn adapter_vendor_name(&self) -> String {
        self.runtime.adapter_vendor_name.clone()
    }

    #[wasm_bindgen(getter, js_name = adapterArchitecture)]
    pub fn adapter_architecture(&self) -> String {
        self.runtime.adapter_architecture.clone()
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

    #[wasm_bindgen(getter, js_name = df64Available)]
    pub fn df64_available(&self) -> bool {
        self.df64_available
    }

    #[wasm_bindgen(getter, js_name = df64CanaryWords)]
    pub fn df64_canary_words(&self) -> u32 {
        self.df64_canary_words as u32
    }

    #[wasm_bindgen(getter, js_name = df64CanaryError)]
    pub fn df64_canary_error(&self) -> Option<String> {
        self.df64_canary_error.clone()
    }

    /// Builds a reusable C2C plan over every axis of `shape` (axis 0 varies
    /// fastest). Native `F64` is forwarded to `wgpu-fft`, which reports that
    /// browsers have no 64-bit float shaders.
    #[wasm_bindgen(js_name = createPlan)]
    pub async fn create_plan(
        &self,
        shape: Vec<u32>,
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
        let config = plan_config(&shape, batch, direction, precision, normalization)?;
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
                label: Some("wgpu_fft_web.upload"),
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

    /// Splits every `f64` into an `f32` hi/lo pair and uploads the words, so
    /// interleaved complex `[re, im]` becomes df64 `[re_hi, re_lo, im_hi, im_lo]`.
    #[wasm_bindgen(js_name = uploadDf64)]
    pub fn upload_df64(&self, values: &[f64]) -> Result<WgpuFftBuffer, JsValue> {
        let mut words = Vec::new();
        words
            .try_reserve_exact(values.len().saturating_mul(2))
            .map_err(|_| js_error("host allocation failed while splitting df64 upload"))?;
        for &value in values {
            let split = DoubleFloat::from_f64(value);
            words.extend_from_slice(&[split.hi, split.lo]);
        }
        self.upload(&words)
    }

    /// Allocates an uninitialized caller-owned GPU buffer for transform output.
    #[wasm_bindgen(js_name = createBuffer)]
    pub fn create_buffer(&self, byte_len: f64) -> Result<WgpuFftBuffer, JsValue> {
        if !(byte_len.is_finite() && byte_len >= 0.0 && byte_len.fract() == 0.0) {
            return Err(js_error("byte length must be a non-negative integer"));
        }
        let size = byte_len as u64;
        validate_byte_len(size, self.runtime.device.limits().max_buffer_size)?;
        let buffer = self.runtime.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("wgpu_fft_web.buffer"),
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
            label: Some("wgpu_fft_web.download"),
            size: source.size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder =
            self.runtime
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("wgpu_fft_web.download.encoder"),
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
        let mapped = readback
            .slice(..)
            .get_mapped_range()
            .map_err(|error| js_error(format!("download mapped range failed: {error}")))?;
        let bytes = mapped.to_vec();
        drop(mapped);
        readback.unmap();
        readback.destroy();
        Ok(bytes)
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

    /// Encodes, submits, and waits for queue completion.
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
        for (role, buffer, required) in [
            ("input", input, self.plan.required_input_buffer_size_bytes()),
            (
                "output",
                output,
                self.plan.required_output_buffer_size_bytes(),
            ),
        ] {
            if buffer.size < required {
                return Err(js_error(format!(
                    "FFT {role} has {} bytes but the plan requires {required}",
                    buffer.size
                )));
            }
        }
        let mut encoder =
            self.runtime
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("wgpu_fft_web.fft.encoder"),
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
        wait_for_queue(&self.runtime.queue).await
    }
}

#[wasm_bindgen]
impl WgpuFftBuffer {
    #[wasm_bindgen(getter, js_name = byteLength)]
    pub fn byte_length(&self) -> f64 {
        self.size as f64
    }
}

/// C2C transform in host memory with `CpuFftPlan` in `f64`: `data` holds
/// interleaved `re, im` pairs over `shape` (axis 0 fastest). Needs no WebGPU.
#[wasm_bindgen(js_name = cpuFft)]
pub fn cpu_fft(
    shape: Vec<u32>,
    data: &[f64],
    direction: WebFftDirection,
    normalization: WebFftNormalization,
) -> Result<Vec<f64>, JsValue> {
    let config = plan_config(&shape, 1, direction, WebFftPrecision::F64, normalization)?;
    let plan = CpuFftPlan::c2c(config)
        .map_err(|error| js_error(format!("CPU FFT plan creation failed: {error}")))?;
    let mut output = vec![0.0; plan.required_output_len()];
    plan.execute_f64(data, &mut output)
        .map_err(|error| js_error(format!("CPU FFT failed: {error}")))?;
    Ok(output)
}

fn plan_config(
    shape: &[u32],
    batch: u32,
    direction: WebFftDirection,
    precision: WebFftPrecision,
    normalization: WebFftNormalization,
) -> Result<FftConfig, JsValue> {
    if shape.is_empty() || shape.contains(&0) {
        return Err(js_error(
            "the FFT shape needs at least one axis, all non-zero",
        ));
    }
    Ok(
        FftConfig::new_nd(shape.iter().map(|&n| n as usize).collect::<Vec<_>>())
            .with_batch(batch.max(1) as usize)
            .with_direction(direction.into())
            .with_precision(precision.into())
            .with_normalization(normalization.into()),
    )
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

fn wait_for_queue(queue: &wgpu::Queue) -> impl std::future::Future<Output = Result<(), JsValue>> {
    let (sender, receiver) = oneshot::channel();
    queue.on_submitted_work_done(move || {
        let _ = sender.send(());
    });
    async move {
        receiver
            .await
            .map_err(|_| js_error("queue completion callback was dropped"))
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

#[cfg(target_arch = "wasm32")]
fn js_error(message: impl AsRef<str>) -> JsValue {
    js_sys::Error::new(message.as_ref()).into()
}

// JavaScript errors need the wasm32 host; native tests see the message.
#[cfg(not(target_arch = "wasm32"))]
fn js_error(message: impl AsRef<str>) -> JsValue {
    panic!("{}", message.as_ref())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn browser_enums_map_to_core_configuration_values() {
        assert_eq!(
            FftPrecision::from(WebFftPrecision::Df64),
            FftPrecision::Df64
        );
        assert_eq!(
            FftDirection::from(WebFftDirection::Inverse),
            FftDirection::Inverse
        );
        assert_eq!(
            Normalization::from(WebFftNormalization::Orthogonal),
            Normalization::Orthogonal
        );
    }

    // cpuFft against a direct DFT on a 3 x 2 x 5 grid (axis 0 fastest).
    #[test]
    fn cpu_fft_matches_a_direct_dft() {
        let shape = [3usize, 2, 5];
        let n = shape.iter().product::<usize>();
        let data: Vec<f64> = (0..2 * n)
            .map(|i| ((i * 7919) % 23) as f64 - 11.0)
            .collect();
        let out = cpu_fft(
            shape.iter().map(|&x| x as u32).collect(),
            &data,
            WebFftDirection::Forward,
            WebFftNormalization::None,
        )
        .unwrap();
        let index = |a: usize, b: usize, c: usize| (c * shape[1] + b) * shape[0] + a;
        for k2 in 0..shape[2] {
            for k1 in 0..shape[1] {
                for k0 in 0..shape[0] {
                    let (mut re, mut im) = (0.0, 0.0);
                    for j2 in 0..shape[2] {
                        for j1 in 0..shape[1] {
                            for j0 in 0..shape[0] {
                                let phase = -2.0
                                    * std::f64::consts::PI
                                    * ((k0 * j0) as f64 / shape[0] as f64
                                        + (k1 * j1) as f64 / shape[1] as f64
                                        + (k2 * j2) as f64 / shape[2] as f64);
                                let i = index(j0, j1, j2);
                                let (x, y) = (data[2 * i], data[2 * i + 1]);
                                re += x * phase.cos() - y * phase.sin();
                                im += x * phase.sin() + y * phase.cos();
                            }
                        }
                    }
                    let o = index(k0, k1, k2);
                    assert!((out[2 * o] - re).abs() < 1e-9, "re at {k0},{k1},{k2}");
                    assert!((out[2 * o + 1] - im).abs() < 1e-9, "im at {k0},{k1},{k2}");
                }
            }
        }
    }
}
