#![cfg(not(target_arch = "wasm32"))]

//! Opt-in backend canaries for the portable double-float error-free transforms.

use std::mem::ManuallyDrop;

use wgpu_fft::{validate_df64_invariants, DF64_CANARY_CASE_COUNT, DF64_CANARY_WORD_COUNT};

#[test]
fn gpu_df64_error_free_transform_canaries() {
    if std::env::var_os("WGPU_FFT_RUN_GPU_TESTS").is_none() {
        eprintln!("skipping GPU test; set WGPU_FFT_RUN_GPU_TESTS=1 to run it");
        return;
    }

    pollster::block_on(run_canaries());
}

async fn run_canaries() {
    let Some(context) = wgpu_fft::device::request_default_device().await else {
        eprintln!("skipping GPU test; no suitable wgpu adapter was found");
        return;
    };
    // Native wgpu teardown can stall on Windows after an assertion panic. Keep
    // all native objects alive until process exit, matching the other GPU suites.
    let context = ManuallyDrop::new(context);
    let adapter_info = context.adapter.get_info();
    eprintln!(
        "df64 canary adapter: name={:?}, backend={:?}, device_type={:?}, driver={:?}",
        adapter_info.name, adapter_info.backend, adapter_info.device_type, adapter_info.driver
    );

    // Request a featureless device deliberately: these kernels are pure f32 and
    // their support contract must not accidentally depend on SHADER_F64.
    let (device, queue) = context
        .adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("wgpu_fft.test.df64_canary.featureless_device"),
            required_features: wgpu::Features::empty(),
            required_limits: context.adapter.limits(),
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: wgpu::MemoryHints::MemoryUsage,
            trace: wgpu::Trace::Off,
        })
        .await
        .expect("df64 canaries require only core f32 shader support");
    let featureless = ManuallyDrop::new((device, queue));
    assert!(featureless.0.features().is_empty());

    let report = validate_df64_invariants(&featureless.0, &featureless.1)
        .await
        .expect("df64 backend invariants must hold exactly");
    assert_eq!(report.cases, DF64_CANARY_CASE_COUNT);
    assert_eq!(report.exact_words, DF64_CANARY_WORD_COUNT);
    eprintln!(
        "df64 canaries passed: backend={:?}, cases={}, exact_words={}",
        adapter_info.backend, report.cases, report.exact_words
    );
}
