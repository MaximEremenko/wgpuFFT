#![cfg(target_arch = "wasm32")]

use futures_channel::oneshot;
use wasm_bindgen_test::*;
use wgpu_fft::math::{
    reference_c2c_nd_f64, reference_c2r_from_packed_interleaved, reference_r2c_packed_interleaved,
    Complex64,
};
use wgpu_fft::{
    validate_df64_invariants, C2cRoute, FftConfig, FftError, FftPlan, FftPrecision, FftTuning,
    Normalization, DF64_CANARY_CASE_COUNT, DF64_CANARY_WORD_COUNT,
};

wasm_bindgen_test_configure!(run_in_browser);

struct BrowserDefaultContext {
    _instance: wgpu::Instance,
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
}

#[wasm_bindgen_test(async)]
async fn browser_default_limits_correctness_matrix() {
    let context = request_browser_default_device().await;
    let info = context.adapter.get_info();
    assert_eq!(info.backend, wgpu::Backend::BrowserWebGpu);

    let requested_limits = wgpu::Limits::default();
    let actual_limits = context.device.limits();
    assert_eq!(
        actual_limits.max_compute_workgroup_storage_size,
        requested_limits.max_compute_workgroup_storage_size,
        "the browser matrix must run at WebGPU's default workgroup-storage limit"
    );
    assert_eq!(
        actual_limits.max_compute_invocations_per_workgroup,
        requested_limits.max_compute_invocations_per_workgroup,
        "the browser matrix must run at WebGPU's default invocation limit"
    );
    assert_eq!(
        actual_limits.max_storage_buffer_binding_size,
        requested_limits.max_storage_buffer_binding_size,
        "the browser matrix must run at WebGPU's default storage-binding limit"
    );
    assert_eq!(
        actual_limits.max_buffer_size, requested_limits.max_buffer_size,
        "the browser matrix must run at WebGPU's default buffer limit"
    );
    assert_eq!(context.device.features(), wgpu::Features::empty());
    console_log!(
        "browser-default adapter={:?} backend={:?} limits={{max_bind:{}, max_buffer:{}, workgroup_storage:{}, invocations:{}}}",
        info.name,
        info.backend,
        actual_limits.max_storage_buffer_binding_size,
        actual_limits.max_buffer_size,
        actual_limits.max_compute_workgroup_storage_size,
        actual_limits.max_compute_invocations_per_workgroup,
    );

    let canary = validate_df64_invariants(&context.device, &context.queue)
        .await
        .expect("Tint must preserve all df64 error-free-transform invariants");
    assert_eq!(canary.cases, DF64_CANARY_CASE_COUNT);
    assert_eq!(canary.exact_words, DF64_CANARY_WORD_COUNT);
    console_log!(
        "df64 Tint canary passed: cases={} exact_words={}",
        canary.cases,
        canary.exact_words
    );

    for (label, config, expected_route) in [
        (
            "smooth-forward-330",
            FftConfig::new(330).with_normalization(Normalization::None),
            C2cRoute::MixedRadix,
        ),
        (
            "smooth-inverse-330",
            FftConfig::inverse(330),
            C2cRoute::MixedRadix,
        ),
        (
            "rader-forward-101",
            FftConfig::new(101).with_normalization(Normalization::None),
            C2cRoute::Rader,
        ),
        (
            "rader-inverse-101",
            FftConfig::inverse(101),
            C2cRoute::Rader,
        ),
        (
            "bluestein-forward-85",
            FftConfig::new(85).with_normalization(Normalization::None),
            C2cRoute::Bluestein,
        ),
        (
            "bluestein-inverse-85",
            FftConfig::inverse(85),
            C2cRoute::Bluestein,
        ),
        // Multi-line, register, and in-place kernels at the default limits.
        (
            "registers-2d-128x96",
            FftConfig::new_nd([128, 96]).with_normalization(Normalization::None),
            C2cRoute::MixedRadix,
        ),
        (
            "registers-3d-inverse-32x32x32",
            FftConfig::inverse_nd([32, 32, 32]),
            C2cRoute::MixedRadix,
        ),
        (
            "smooth-3d-60x48x10",
            FftConfig::new_nd([60, 48, 10]).with_normalization(Normalization::None),
            C2cRoute::MixedRadix,
        ),
        // A composite-radix schedule with padded workgroup indices that still
        // fits 16 KiB.
        (
            "smooth-padded-1920",
            FftConfig::new(1920).with_normalization(Normalization::None),
            C2cRoute::MixedRadix,
        ),
        // Primes: the direct kernel on a small 2D transform, and a Rader
        // axis whose convolution runs as a short register Bluestein one.
        (
            "direct-2d-31x31",
            FftConfig::new_nd([31, 31]).with_normalization(Normalization::None),
            C2cRoute::AxisSequence,
        ),
        (
            "rader-register-bluestein-inverse-179",
            FftConfig::inverse(179).with_batch(3),
            C2cRoute::Rader,
        ),
        // A cyclic Rader convolution with a radix-23 stage (1380 = 60 * 23).
        (
            "rader-medium-prime-1381",
            FftConfig::new(1381)
                .with_batch(2)
                .with_normalization(Normalization::None),
            C2cRoute::Rader,
        ),
    ] {
        run_c2c_case(&context, label, config, expected_route).await;
    }

    assert_browser_fused_boundary(&context).await;
    run_real_roundtrip_routes(&context).await;
    assert_browser_rejects_native_f64(&context).await;
}

async fn request_browser_default_device() -> BrowserDefaultContext {
    let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
    descriptor.backends = wgpu::Backends::BROWSER_WEBGPU;
    let instance = wgpu::Instance::new(descriptor);
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            apply_limit_buckets: false,
            compatible_surface: None,
        })
        .await
        .expect("Chrome must expose a WebGPU adapter");
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("wgpu_fft.browser_matrix.default_device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: wgpu::MemoryHints::MemoryUsage,
            trace: wgpu::Trace::Off,
        })
        .await
        .expect("Chrome must grant the WebGPU default limits");

    BrowserDefaultContext {
        _instance: instance,
        adapter,
        device,
        queue,
    }
}

async fn run_c2c_case(
    context: &BrowserDefaultContext,
    label: &str,
    config: FftConfig,
    expected_route: C2cRoute,
) {
    let input = complex_signal(config.total_complex_len().unwrap());
    // An f64 reference: an f32 one drifts by about 3e-4 of the peak at 1920
    // points, more than the transforms under test.
    let expected = reference_c2c_f64_as_f32(&input, &config);
    let plan = FftPlan::c2c_checked(&context.device, &context.queue, config)
        .await
        .unwrap_or_else(|error| panic!("{label}: failed to create C2C plan: {error}"));
    assert_eq!(plan.route(), expected_route, "{label}: unexpected route");
    let actual = execute_f32(context, &plan, &input, label).await;
    assert_fft_accuracy(&actual, &expected, label);
}

async fn assert_browser_fused_boundary(context: &BrowserDefaultContext) {
    let fused_config = FftConfig::new(2048).with_normalization(Normalization::None);
    let fused = FftPlan::c2c_checked(&context.device, &context.queue, fused_config.clone())
        .await
        .expect("N=2048 must fit the browser-default 16-KiB f32 fused gate");
    let fused_kernels = kernel_labels(&fused);
    assert_eq!(
        fused_kernels,
        ["fused-pow2-workgroup-stage"],
        "N=2048 must use exactly one fused f32 kernel at browser defaults"
    );
    assert_eq!(fused.workspace_size_bytes(), 0);
    let fused_input = complex_signal(2048);
    let fused_expected = reference_c2c_f64_as_f32(&fused_input, &fused_config);
    let fused_actual = execute_f32(context, &fused, &fused_input, "fused-boundary-2048").await;
    assert_fft_accuracy(&fused_actual, &fused_expected, "fused-boundary-2048");

    // N=4096 outgrows 16 KiB of workgroup memory: it runs as one
    // register-resident kernel, exchanging through workgroup memory in rounds.
    let long_config = FftConfig::new(4096).with_normalization(Normalization::None);
    let long = FftPlan::c2c_checked(&context.device, &context.queue, long_config.clone())
        .await
        .expect("N=4096 must plan at browser defaults");
    assert_eq!(
        kernel_labels(&long),
        ["fused-pow2-workgroup-stage"],
        "N=4096 must run as one register-resident kernel at browser defaults"
    );
    assert_eq!(long.workspace_size_bytes(), 0);
    let long_input = complex_signal(4096);
    let long_expected = reference_c2c_f64_as_f32(&long_input, &long_config);
    let long_actual = execute_f32(context, &long, &long_input, "register-long-4096").await;
    assert_fft_accuracy(&long_actual, &long_expected, "register-long-4096");

    // Without long-axis fusion it keeps the multipass Stockham fallback.
    let multipass_config = FftConfig::new(4096)
        .with_normalization(Normalization::None)
        .with_tuning(FftTuning::default().with_fuse_long_axes(false));
    let multipass = FftPlan::c2c_checked(&context.device, &context.queue, multipass_config.clone())
        .await
        .expect("N=4096 must retain the multipass fallback at browser defaults");
    let multipass_kernels = kernel_labels(&multipass);
    assert!(
        !multipass_kernels
            .iter()
            .any(|label| label == "fused-pow2-workgroup-stage"),
        "N=4096 must not cross the browser-default 16-KiB f32 fused gate: {multipass_kernels:?}"
    );
    assert!(
        multipass_kernels
            .iter()
            .all(|label| label == "mixed-radix-stockham-stage"),
        "N=4096 must use only Stockham fallback kernels: {multipass_kernels:?}"
    );
    assert!(multipass_kernels.len() > 1);
    assert!(multipass.workspace_size_bytes() > 0);
    let multipass_input = complex_signal(4096);
    let multipass_expected = reference_c2c_f64_as_f32(&multipass_input, &multipass_config);
    let multipass_actual = execute_f32(
        context,
        &multipass,
        &multipass_input,
        "multipass-boundary-4096",
    )
    .await;
    assert_fft_accuracy(
        &multipass_actual,
        &multipass_expected,
        "multipass-boundary-4096",
    );
}

async fn run_real_roundtrip_routes(context: &BrowserDefaultContext) {
    let forward = FftConfig::new(34).with_normalization(Normalization::None);
    let real = real_signal(34);
    let expected_packed = reference_r2c_packed_interleaved(&real, &forward).unwrap();
    let r2c =
        FftPlan::r2c(&context.device, &context.queue, forward).expect("browser-default R2C plan");
    assert_eq!(r2c.route(), C2cRoute::Bluestein);
    let actual_packed = execute_f32(context, &r2c, &real, "r2c-bluestein-34").await;
    assert_close(&actual_packed, &expected_packed, "r2c-bluestein-34");

    let inverse = FftConfig::inverse(34);
    let expected_real = reference_c2r_from_packed_interleaved(&expected_packed, &inverse).unwrap();
    let c2r =
        FftPlan::c2r(&context.device, &context.queue, inverse).expect("browser-default C2R plan");
    assert_eq!(c2r.route(), C2cRoute::Bluestein);
    let actual_real = execute_f32(context, &c2r, &expected_packed, "c2r-bluestein-34").await;
    assert_close(&actual_real, &expected_real, "c2r-bluestein-34");
}

async fn assert_browser_rejects_native_f64(context: &BrowserDefaultContext) {
    let result = FftPlan::c2c_checked(
        &context.device,
        &context.queue,
        FftConfig::new(8).with_precision(FftPrecision::F64),
    )
    .await;
    let error = match result {
        Ok(_) => panic!("native f64 must not be enabled by a browser-default device"),
        Err(error) => error,
    };
    assert_eq!(
        error,
        FftError::PrecisionUnsupported {
            requested: FftPrecision::F64,
            route: "c2c",
            reason: "device-missing-shader-f64",
        }
    );
}

fn kernel_labels(plan: &FftPlan) -> Vec<String> {
    plan.diagnostics()
        .stages()
        .iter()
        .filter(|stage| stage.kind == "kernel")
        .map(|stage| stage.label.clone())
        .collect()
}

async fn execute_f32(
    context: &BrowserDefaultContext,
    plan: &FftPlan,
    input: &[f32],
    label: &str,
) -> Vec<f32> {
    let input_bytes = plan.required_input_buffer_size_bytes();
    let output_bytes = plan.required_output_buffer_size_bytes();
    assert_eq!(
        input_bytes,
        std::mem::size_of_val(input) as u64,
        "{label}: input size"
    );

    let input_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.browser_matrix.input"),
        size: input_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    context
        .queue
        .write_buffer(&input_buffer, 0, bytemuck::cast_slice(input));
    let output_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.browser_matrix.output"),
        size: output_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.browser_matrix.readback"),
        size: output_bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("wgpu_fft.browser_matrix.encoder"),
        });
    plan.execute_checked(&context.device, &mut encoder, &input_buffer, &output_buffer)
        .unwrap_or_else(|error| panic!("{label}: failed to encode: {error}"));
    encoder.copy_buffer_to_buffer(&output_buffer, 0, &readback, 0, output_bytes);
    context.queue.submit([encoder.finish()]);

    let (sender, receiver) = oneshot::channel();
    readback
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
    receiver
        .await
        .expect("browser map callback must run")
        .unwrap_or_else(|error| panic!("{label}: readback mapping failed: {error}"));
    let mapped = readback
        .slice(..)
        .get_mapped_range()
        .expect("browser readback range must be mapped");
    bytemuck::cast_slice::<u8, f32>(&mapped).to_vec()
}

fn complex_signal(complex_len: usize) -> Vec<f32> {
    (0..complex_len)
        .flat_map(|index| {
            let x = index as f32 + 1.0;
            [
                (x * 0.017).sin() * 0.7 + (x * 0.031).cos() * 0.2,
                (x * 0.023).cos() * 0.5 - (x * 0.011).sin() * 0.3,
            ]
        })
        .collect()
}

fn real_signal(len: usize) -> Vec<f32> {
    (0..len)
        .map(|index| {
            let x = index as f32 + 0.5;
            (x * 0.071).sin() * 0.6 + (x * 0.019).cos() * 0.35
        })
        .collect()
}

fn reference_c2c_f64_as_f32(input: &[f32], config: &FftConfig) -> Vec<f32> {
    let values = input
        .chunks_exact(2)
        .map(|pair| Complex64::new(f64::from(pair[0]), f64::from(pair[1])))
        .collect::<Vec<_>>();
    reference_c2c_nd_f64(&values, config)
        .unwrap()
        .into_iter()
        .flat_map(|value| [value.re as f32, value.im as f32])
        .collect()
}

fn assert_close(actual: &[f32], expected: &[f32], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        let tolerance = 2.0e-2 + 2.0e-5 * expected.abs();
        assert!(
            actual.is_finite() && (actual - expected).abs() <= tolerance,
            "{label}: index {index}: actual={actual}, expected={expected}, tolerance={tolerance}"
        );
    }
}

fn assert_fft_accuracy(actual: &[f32], expected: &[f32], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");
    let mut squared_error = 0.0f64;
    let mut squared_reference = 0.0f64;
    let mut max_abs_error = 0.0f64;
    let mut max_abs_reference = 0.0f64;
    for (&actual, &expected) in actual.iter().zip(expected) {
        assert!(actual.is_finite(), "{label}: nonfinite GPU output");
        let error = f64::from(actual) - f64::from(expected);
        squared_error += error * error;
        squared_reference += f64::from(expected) * f64::from(expected);
        max_abs_error = max_abs_error.max(error.abs());
        max_abs_reference = max_abs_reference.max(f64::from(expected).abs());
    }
    let relative_l2 = (squared_error / squared_reference.max(f64::MIN_POSITIVE)).sqrt();
    let max_relative = max_abs_error / max_abs_reference.max(f64::MIN_POSITIVE);
    console_log!(
        "browser FFT accuracy label={} relative_l2={:.9e} max_relative={:.9e} max_abs={:.9e}",
        label,
        relative_l2,
        max_relative,
        max_abs_error
    );
    assert!(
        relative_l2 <= 2.0e-5 && max_relative <= 1.0e-3,
        "{label}: relative_l2={relative_l2:.9e}, max_relative={max_relative:.9e}, max_abs={max_abs_error:.9e}"
    );
}
