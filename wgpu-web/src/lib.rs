//! Minimal browser-facing `wasm-bindgen` surface for `wgpu-fft` and `wgpu-nufft`.
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
use wgpu_nufft::{
    DoubleFloat, ModeOrder, NufftConfig, NufftInterval, NufftKind, NufftPlan, NufftType3Config,
    NufftType3Plan,
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

/// Browser uniform-mode ordering for type-1 and type-2 NUFFTs.
#[wasm_bindgen]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WebNufftModeOrder {
    Centered,
    Fft,
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

/// Reusable browser type-1 or type-2 NUFFT plan with a fixed point count.
#[wasm_bindgen]
pub struct WgpuNufftPlan {
    runtime: Rc<Runtime>,
    plan: NufftPlan,
    point_count: usize,
    batch_capacity: usize,
    point_bytes: u64,
    input_bytes: u64,
    output_bytes: u64,
}

/// Reusable browser type-3 NUFFT plan with fixed source and target counts.
#[wasm_bindgen]
pub struct WgpuNufftType3Plan {
    runtime: Rc<Runtime>,
    plan: NufftType3Plan,
    source_count: usize,
    target_count: usize,
    batch_capacity: usize,
    source_point_bytes: u64,
    target_point_bytes: u64,
    strength_bytes: u64,
    output_bytes: u64,
}

/// GPU-resident caller-owned byte buffer.
#[wasm_bindgen]
pub struct WgpuFftBuffer {
    runtime: Rc<Runtime>,
    buffer: wgpu::Buffer,
    size: u64,
}

async fn initialize(request_adapter_maximums: bool, force_fallback: bool) -> Result<WgpuFft, JsValue> {
    let mut instance_descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
    instance_descriptor.backends = wgpu::Backends::BROWSER_WEBGPU;
    let instance = wgpu::Instance::new(instance_descriptor);
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: force_fallback,
            compatible_surface: None,
        })
        .await
        .map_err(|error| js_error(format!("WebGPU adapter request failed: {error}")))?;
    let info = adapter.get_info();
    let default_descriptor = || wgpu::DeviceDescriptor {
        label: Some("wgpu_web.default_device"),
        required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits::default(),
        experimental_features: wgpu::ExperimentalFeatures::disabled(),
        memory_hints: wgpu::MemoryHints::MemoryUsage,
        trace: wgpu::Trace::Off,
    };
    let (device, queue) = if request_adapter_maximums {
        let maximum_descriptor = wgpu::DeviceDescriptor {
            label: Some("wgpu_web.device"),
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
        initialize(true, false).await
    }

    /// Acquires a featureless browser device at the WebGPU default limits and
    /// runs the same 96-word df64 invariant suite as [`Self::init`].
    /// Initializes on the browser's software fallback adapter (CPU execution
    /// of the same WGSL pipelines, e.g. SwiftShader). Useful where no hardware
    /// WebGPU adapter is available or for deterministic CPU runs.
    #[wasm_bindgen(js_name = initFallback)]
    pub async fn init_fallback() -> Result<WgpuFft, JsValue> {
        initialize(true, true).await
    }

    #[wasm_bindgen(js_name = initWithDefaultLimits)]
    pub async fn init_with_default_limits() -> Result<WgpuFft, JsValue> {
        initialize(false, false).await
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

    #[wasm_bindgen(getter, js_name = maxComputeInvocationsPerWorkgroup)]
    pub fn max_compute_invocations_per_workgroup(&self) -> u32 {
        self.runtime
            .device
            .limits()
            .max_compute_invocations_per_workgroup
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

    /// Builds a reusable one-, two-, or three-dimensional type-1 NUFFT plan.
    ///
    /// `nModes` is copied before this Rust future first awaits GPU work. Because
    /// JavaScript starts the future asynchronously, callers must retain the
    /// typed array unchanged until the returned Promise settles. The plan fixes
    /// point count and batch capacity so every byte size is then synchronous.
    #[wasm_bindgen(js_name = createNufftType1Plan)]
    #[allow(clippy::too_many_arguments)]
    pub async fn create_nufft_type1_plan(
        &self,
        n_modes: js_sys::Uint32Array,
        point_count: u32,
        batch: u32,
        eps: f64,
        isign: i32,
        mode_order: WebNufftModeOrder,
        sigma: f64,
        precision: WebFftPrecision,
    ) -> Result<WgpuNufftPlan, JsValue> {
        let n_modes = n_modes
            .to_vec()
            .into_iter()
            .map(|length| length as usize)
            .collect::<Vec<_>>();
        let point_count = point_count as usize;
        let batch_capacity = batch as usize;
        let precision = self.checked_nufft_precision(precision)?;
        let config = NufftConfig::new(n_modes, eps)
            .with_isign(isign)
            .with_mode_order(mode_order.into())
            .with_sigma(sigma)
            .with_batch(batch_capacity)
            .with_precision(precision);

        let scopes = WebGpuErrorScopes::push(&self.runtime.device);
        let plan_result = NufftPlan::type1_gpu(&self.runtime.device, &self.runtime.queue, config);
        if let Some((kind, message)) = scopes.pop_error().await {
            clear_thread_local_pipeline_cache(&self.runtime.device);
            return Err(js_error(format!(
                "type-1 NUFFT plan creation failed during {kind}: {message}"
            )));
        }
        let plan = plan_result
            .map_err(|error| js_error(format!("type-1 NUFFT plan creation failed: {error}")))?;
        WgpuNufftPlan::from_plan(Rc::clone(&self.runtime), plan, point_count, batch_capacity)
    }

    /// Builds a reusable one-, two-, or three-dimensional type-2 NUFFT plan.
    #[wasm_bindgen(js_name = createNufftType2Plan)]
    #[allow(clippy::too_many_arguments)]
    pub async fn create_nufft_type2_plan(
        &self,
        n_modes: js_sys::Uint32Array,
        point_count: u32,
        batch: u32,
        eps: f64,
        isign: i32,
        mode_order: WebNufftModeOrder,
        sigma: f64,
        precision: WebFftPrecision,
    ) -> Result<WgpuNufftPlan, JsValue> {
        let n_modes = n_modes
            .to_vec()
            .into_iter()
            .map(|length| length as usize)
            .collect::<Vec<_>>();
        let point_count = point_count as usize;
        let batch_capacity = batch as usize;
        let precision = self.checked_nufft_precision(precision)?;
        let config = NufftConfig::new(n_modes, eps)
            .with_isign(isign)
            .with_mode_order(mode_order.into())
            .with_sigma(sigma)
            .with_batch(batch_capacity)
            .with_precision(precision);

        let scopes = WebGpuErrorScopes::push(&self.runtime.device);
        let plan_result = NufftPlan::type2_gpu(&self.runtime.device, &self.runtime.queue, config);
        if let Some((kind, message)) = scopes.pop_error().await {
            clear_thread_local_pipeline_cache(&self.runtime.device);
            return Err(js_error(format!(
                "type-2 NUFFT plan creation failed during {kind}: {message}"
            )));
        }
        let plan = plan_result
            .map_err(|error| js_error(format!("type-2 NUFFT plan creation failed: {error}")))?;
        WgpuNufftPlan::from_plan(Rc::clone(&self.runtime), plan, point_count, batch_capacity)
    }

    /// Builds a reusable one-, two-, or three-dimensional type-3 NUFFT plan.
    ///
    /// Bounds are flattened `[lower0, upper0, lower1, upper1, ...]` arrays.
    /// They are copied before this Rust future first awaits GPU work, but JS
    /// callers must retain them unchanged until the returned Promise settles.
    #[wasm_bindgen(js_name = createNufftType3Plan)]
    #[allow(clippy::too_many_arguments)]
    pub async fn create_nufft_type3_plan(
        &self,
        source_bounds: js_sys::Float64Array,
        target_bounds: js_sys::Float64Array,
        source_count: u32,
        target_count: u32,
        batch: u32,
        eps: f64,
        isign: i32,
        sigma: f64,
        precision: WebFftPrecision,
    ) -> Result<WgpuNufftType3Plan, JsValue> {
        let source_bounds = parse_intervals(source_bounds.to_vec(), "source")?;
        let target_bounds = parse_intervals(target_bounds.to_vec(), "target")?;
        let source_count = source_count as usize;
        let target_count = target_count as usize;
        let batch_capacity = batch as usize;
        let precision = self.checked_nufft_precision(precision)?;
        let config = NufftType3Config::new(source_bounds, target_bounds, eps)
            .with_isign(isign)
            .with_sigma(sigma)
            .with_batch(batch_capacity)
            .with_precision(precision);

        let scopes = WebGpuErrorScopes::push(&self.runtime.device);
        let plan_result =
            NufftType3Plan::new_gpu(&self.runtime.device, &self.runtime.queue, config);
        if let Some((kind, message)) = scopes.pop_error().await {
            clear_thread_local_pipeline_cache(&self.runtime.device);
            return Err(js_error(format!(
                "type-3 NUFFT plan creation failed during {kind}: {message}"
            )));
        }
        let plan = plan_result
            .map_err(|error| js_error(format!("type-3 NUFFT plan creation failed: {error}")))?;
        WgpuNufftType3Plan::from_plan(
            Rc::clone(&self.runtime),
            plan,
            source_count,
            target_count,
            batch_capacity,
        )
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

    /// Splits every JavaScript `f64` scalar into an `f32` hi/lo pair and
    /// uploads the resulting storage words. Interleaved complex `[re, im]`
    /// input therefore becomes df64 `[re_hi, re_lo, im_hi, im_lo]` storage.
    #[wasm_bindgen(js_name = uploadDf64)]
    pub fn upload_df64(&self, values: js_sys::Float64Array) -> Result<WgpuFftBuffer, JsValue> {
        let values = values.to_vec();
        let capacity = values
            .len()
            .checked_mul(2)
            .ok_or_else(|| js_error("df64 upload word count overflows usize"))?;
        let mut words = Vec::new();
        words
            .try_reserve_exact(capacity)
            .map_err(|_| js_error("host allocation failed while splitting df64 upload"))?;
        for value in values {
            let split = DoubleFloat::from_f64(value);
            words.extend_from_slice(&[split.hi, split.lo]);
        }
        self.upload(&words)
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
        let scopes = WebGpuErrorScopes::push(&self.runtime.device);
        let imported = import_pipeline_cache_snapshot(&self.runtime.device, &snapshot);
        if let Some((kind, message)) = scopes.pop_error().await {
            clear_thread_local_pipeline_cache(&self.runtime.device);
            return Err(js_error(format!(
                "pipeline snapshot import failed during {kind}: {message}"
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
        wait_for_queue(&self.runtime.queue).await
    }
}

impl WgpuNufftPlan {
    fn from_plan(
        runtime: Rc<Runtime>,
        plan: NufftPlan,
        point_count: usize,
        batch_capacity: usize,
    ) -> Result<Self, JsValue> {
        let point_bytes = plan
            .required_point_buffer_size_bytes(point_count)
            .map_err(|error| js_error(format!("NUFFT point size failed: {error}")))?;
        let (input_bytes, output_bytes) = match plan.kind() {
            NufftKind::Type1 => (
                plan.required_type1_strength_buffer_size_bytes_for_batch(
                    point_count,
                    batch_capacity,
                )
                .map_err(|error| js_error(format!("type-1 input size failed: {error}")))?,
                plan.required_type1_output_buffer_size_bytes_for_batch(batch_capacity)
                    .map_err(|error| js_error(format!("type-1 output size failed: {error}")))?,
            ),
            NufftKind::Type2 => (
                plan.required_type2_coefficient_buffer_size_bytes_for_batch(batch_capacity)
                    .map_err(|error| js_error(format!("type-2 input size failed: {error}")))?,
                plan.required_type2_output_buffer_size_bytes_for_batch(point_count, batch_capacity)
                    .map_err(|error| js_error(format!("type-2 output size failed: {error}")))?,
            ),
            _ => return Err(js_error("unsupported NUFFT plan kind")),
        };
        Ok(Self {
            runtime,
            plan,
            point_count,
            batch_capacity,
            point_bytes,
            input_bytes,
            output_bytes,
        })
    }
}

#[wasm_bindgen]
impl WgpuNufftPlan {
    #[wasm_bindgen(getter)]
    pub fn kind(&self) -> String {
        match self.plan.kind() {
            NufftKind::Type1 => String::from("type-1"),
            NufftKind::Type2 => String::from("type-2"),
            _ => String::from("unsupported"),
        }
    }

    #[wasm_bindgen(getter)]
    pub fn dimensions(&self) -> u32 {
        self.plan.config().dimensions() as u32
    }

    #[wasm_bindgen(getter, js_name = pointCount)]
    pub fn point_count(&self) -> u32 {
        self.point_count as u32
    }

    #[wasm_bindgen(getter, js_name = batchCapacity)]
    pub fn batch_capacity(&self) -> u32 {
        self.batch_capacity as u32
    }

    #[wasm_bindgen(getter, js_name = pointBytes)]
    pub fn point_bytes(&self) -> f64 {
        self.point_bytes as f64
    }

    /// Full-capacity type-1 strength or type-2 coefficient buffer bytes.
    #[wasm_bindgen(getter, js_name = inputBytes)]
    pub fn input_bytes(&self) -> f64 {
        self.input_bytes as f64
    }

    /// Full-capacity result buffer bytes.
    #[wasm_bindgen(getter, js_name = outputBytes)]
    pub fn output_bytes(&self) -> f64 {
        self.output_bytes as f64
    }

    /// Executes `activeBatch` transforms sharing this plan's point buffer.
    /// Input and output vectors are transform-major. Device-resident point
    /// coordinates cannot be inspected here; callers must supply finite values
    /// inside the `[-3*pi, 3*pi]` contract.
    pub async fn execute(
        &self,
        points: &WgpuFftBuffer,
        input: &WgpuFftBuffer,
        output: &WgpuFftBuffer,
        #[wasm_bindgen(js_name = activeBatch)] active_batch: u32,
    ) -> Result<(), JsValue> {
        let active_batch = active_batch as usize;
        validate_buffer(
            &self.runtime,
            points,
            self.point_bytes,
            "NUFFT point buffer",
        )?;
        let (input_bytes, output_bytes) = match self.plan.kind() {
            NufftKind::Type1 => (
                self.plan
                    .required_type1_strength_buffer_size_bytes_for_batch(
                        self.point_count,
                        active_batch,
                    )
                    .map_err(|error| js_error(format!("type-1 input size failed: {error}")))?,
                self.plan
                    .required_type1_output_buffer_size_bytes_for_batch(active_batch)
                    .map_err(|error| js_error(format!("type-1 output size failed: {error}")))?,
            ),
            NufftKind::Type2 => (
                self.plan
                    .required_type2_coefficient_buffer_size_bytes_for_batch(active_batch)
                    .map_err(|error| js_error(format!("type-2 input size failed: {error}")))?,
                self.plan
                    .required_type2_output_buffer_size_bytes_for_batch(
                        self.point_count,
                        active_batch,
                    )
                    .map_err(|error| js_error(format!("type-2 output size failed: {error}")))?,
            ),
            _ => return Err(js_error("unsupported NUFFT plan kind")),
        };
        validate_buffer(&self.runtime, input, input_bytes, "NUFFT input buffer")?;
        validate_buffer(&self.runtime, output, output_bytes, "NUFFT output buffer")?;
        ensure_distinct(
            &[("points", points), ("input", input), ("output", output)],
            "NUFFT execution",
        )?;

        let scopes = WebGpuErrorScopes::push(&self.runtime.device);
        let mut encoder =
            self.runtime
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("wgpu_web.nufft.encoder"),
                });
        let encode_result = match self.plan.kind() {
            NufftKind::Type1 => self.plan.encode_type1_gpu_batch(
                &self.runtime.device,
                &mut encoder,
                self.point_count,
                active_batch,
                &points.buffer,
                &input.buffer,
                &output.buffer,
            ),
            NufftKind::Type2 => self.plan.encode_type2_gpu_batch(
                &self.runtime.device,
                &mut encoder,
                self.point_count,
                active_batch,
                &points.buffer,
                &input.buffer,
                &output.buffer,
            ),
            _ => return Err(js_error("unsupported NUFFT plan kind")),
        };
        let queue_future = if encode_result.is_ok() {
            self.runtime.queue.submit([encoder.finish()]);
            Some(wait_for_queue(&self.runtime.queue))
        } else {
            None
        };
        if let Some((kind, message)) = scopes.pop_error().await {
            return Err(js_error(format!(
                "NUFFT execution failed during {kind}: {message}"
            )));
        }
        encode_result.map_err(|error| js_error(format!("NUFFT execution failed: {error}")))?;
        if let Some(queue_future) = queue_future {
            queue_future.await?;
        }
        Ok(())
    }
}

impl WgpuNufftType3Plan {
    fn from_plan(
        runtime: Rc<Runtime>,
        plan: NufftType3Plan,
        source_count: usize,
        target_count: usize,
        batch_capacity: usize,
    ) -> Result<Self, JsValue> {
        let source_point_bytes = plan
            .required_source_point_buffer_size_bytes(source_count)
            .map_err(|error| js_error(format!("type-3 source point size failed: {error}")))?;
        let target_point_bytes = plan
            .required_target_point_buffer_size_bytes(target_count)
            .map_err(|error| js_error(format!("type-3 target point size failed: {error}")))?;
        let strength_bytes = plan
            .required_strength_buffer_size_bytes_for_batch(source_count, batch_capacity)
            .map_err(|error| js_error(format!("type-3 strength size failed: {error}")))?;
        let output_bytes = plan
            .required_output_buffer_size_bytes_for_batch(target_count, batch_capacity)
            .map_err(|error| js_error(format!("type-3 output size failed: {error}")))?;
        Ok(Self {
            runtime,
            plan,
            source_count,
            target_count,
            batch_capacity,
            source_point_bytes,
            target_point_bytes,
            strength_bytes,
            output_bytes,
        })
    }
}

#[wasm_bindgen]
impl WgpuNufftType3Plan {
    #[wasm_bindgen(getter)]
    pub fn dimensions(&self) -> u32 {
        self.plan.config().dimensions() as u32
    }

    #[wasm_bindgen(getter, js_name = sourceCount)]
    pub fn source_count(&self) -> u32 {
        self.source_count as u32
    }

    #[wasm_bindgen(getter, js_name = targetCount)]
    pub fn target_count(&self) -> u32 {
        self.target_count as u32
    }

    #[wasm_bindgen(getter, js_name = batchCapacity)]
    pub fn batch_capacity(&self) -> u32 {
        self.batch_capacity as u32
    }

    #[wasm_bindgen(getter, js_name = sourcePointBytes)]
    pub fn source_point_bytes(&self) -> f64 {
        self.source_point_bytes as f64
    }

    #[wasm_bindgen(getter, js_name = targetPointBytes)]
    pub fn target_point_bytes(&self) -> f64 {
        self.target_point_bytes as f64
    }

    #[wasm_bindgen(getter, js_name = strengthBytes)]
    pub fn strength_bytes(&self) -> f64 {
        self.strength_bytes as f64
    }

    #[wasm_bindgen(getter, js_name = outputBytes)]
    pub fn output_bytes(&self) -> f64 {
        self.output_bytes as f64
    }

    /// Executes `activeBatch` type-3 transforms sharing both point sets.
    /// Device-resident coordinates cannot be inspected here; callers must keep
    /// every finite source/target coordinate inside its plan-time interval.
    #[allow(clippy::too_many_arguments)]
    pub async fn execute(
        &self,
        source_points: &WgpuFftBuffer,
        strengths: &WgpuFftBuffer,
        target_points: &WgpuFftBuffer,
        output: &WgpuFftBuffer,
        #[wasm_bindgen(js_name = activeBatch)] active_batch: u32,
    ) -> Result<(), JsValue> {
        let active_batch = active_batch as usize;
        let strength_bytes = self
            .plan
            .required_strength_buffer_size_bytes_for_batch(self.source_count, active_batch)
            .map_err(|error| js_error(format!("type-3 strength size failed: {error}")))?;
        let output_bytes = self
            .plan
            .required_output_buffer_size_bytes_for_batch(self.target_count, active_batch)
            .map_err(|error| js_error(format!("type-3 output size failed: {error}")))?;
        validate_buffer(
            &self.runtime,
            source_points,
            self.source_point_bytes,
            "type-3 source point buffer",
        )?;
        validate_buffer(
            &self.runtime,
            strengths,
            strength_bytes,
            "type-3 strength buffer",
        )?;
        validate_buffer(
            &self.runtime,
            target_points,
            self.target_point_bytes,
            "type-3 target point buffer",
        )?;
        validate_buffer(&self.runtime, output, output_bytes, "type-3 output buffer")?;
        ensure_distinct(
            &[
                ("source points", source_points),
                ("strengths", strengths),
                ("target points", target_points),
                ("output", output),
            ],
            "type-3 NUFFT execution",
        )?;

        let scopes = WebGpuErrorScopes::push(&self.runtime.device);
        let mut encoder =
            self.runtime
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("wgpu_web.nufft_type3.encoder"),
                });
        let encode_result = self.plan.encode_gpu_batch(
            &self.runtime.device,
            &mut encoder,
            active_batch,
            self.source_count,
            &source_points.buffer,
            &strengths.buffer,
            self.target_count,
            &target_points.buffer,
            &output.buffer,
        );
        let queue_future = if encode_result.is_ok() {
            self.runtime.queue.submit([encoder.finish()]);
            Some(wait_for_queue(&self.runtime.queue))
        } else {
            None
        };
        if let Some((kind, message)) = scopes.pop_error().await {
            return Err(js_error(format!(
                "type-3 NUFFT execution failed during {kind}: {message}"
            )));
        }
        encode_result
            .map_err(|error| js_error(format!("type-3 NUFFT execution failed: {error}")))?;
        if let Some(queue_future) = queue_future {
            queue_future.await?;
        }
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

impl From<WebNufftModeOrder> for ModeOrder {
    fn from(value: WebNufftModeOrder) -> Self {
        match value {
            WebNufftModeOrder::Centered => Self::Centered,
            WebNufftModeOrder::Fft => Self::Fft,
        }
    }
}

impl WgpuFft {
    fn checked_nufft_precision(&self, precision: WebFftPrecision) -> Result<FftPrecision, JsValue> {
        if precision == WebFftPrecision::Df64 && !self.df64_available {
            let reason = self
                .df64_canary_error
                .as_deref()
                .unwrap_or("the 96-word browser invariant suite did not pass");
            return Err(js_error(format!(
                "Df64 is disabled for this browser/compiler: {reason}"
            )));
        }
        Ok(precision.into())
    }
}

struct WebGpuErrorScopes {
    out_of_memory: wgpu::ErrorScopeGuard,
    internal: wgpu::ErrorScopeGuard,
    validation: wgpu::ErrorScopeGuard,
}

impl WebGpuErrorScopes {
    fn push(device: &wgpu::Device) -> Self {
        let out_of_memory = device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let internal = device.push_error_scope(wgpu::ErrorFilter::Internal);
        let validation = device.push_error_scope(wgpu::ErrorFilter::Validation);
        Self {
            out_of_memory,
            internal,
            validation,
        }
    }

    fn pop_error(self) -> impl std::future::Future<Output = Option<(&'static str, String)>> {
        let validation_pop = self.validation.pop();
        let internal_pop = self.internal.pop();
        let out_of_memory_pop = self.out_of_memory.pop();
        async move {
            let validation_error = validation_pop.await;
            let internal_error = internal_pop.await;
            let out_of_memory_error = out_of_memory_pop.await;
            if let Some(error) = validation_error {
                Some(("validation", error.to_string()))
            } else if let Some(error) = internal_error {
                Some(("internal", error.to_string()))
            } else {
                out_of_memory_error.map(|error| ("out-of-memory", error.to_string()))
            }
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

fn parse_intervals(values: Vec<f64>, set: &'static str) -> Result<Vec<NufftInterval>, JsValue> {
    if values.is_empty() || !values.len().is_multiple_of(2) {
        return Err(js_error(format!(
            "{set} bounds must contain one [lower, upper] pair per dimension"
        )));
    }
    Ok(values
        .chunks_exact(2)
        .map(|bounds| NufftInterval::new(bounds[0], bounds[1]))
        .collect())
}

fn validate_buffer(
    runtime: &Rc<Runtime>,
    buffer: &WgpuFftBuffer,
    required_bytes: u64,
    role: &str,
) -> Result<(), JsValue> {
    ensure_same_runtime(runtime, &buffer.runtime, role)?;
    if buffer.size < required_bytes {
        return Err(js_error(format!(
            "{role} has {} bytes but requires at least {required_bytes}",
            buffer.size
        )));
    }
    Ok(())
}

fn ensure_distinct(buffers: &[(&str, &WgpuFftBuffer)], operation: &str) -> Result<(), JsValue> {
    for left in 0..buffers.len() {
        for right in (left + 1)..buffers.len() {
            if buffers[left].1.buffer == buffers[right].1.buffer {
                return Err(js_error(format!(
                    "{operation} requires distinct {} and {} buffers",
                    buffers[left].0, buffers[right].0
                )));
            }
        }
    }
    Ok(())
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
