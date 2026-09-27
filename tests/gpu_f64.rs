#![cfg(not(target_arch = "wasm32"))]

//! Opt-in native-f64 GPU correctness, routing, and capability coverage.

use std::mem::ManuallyDrop;
use std::sync::mpsc;

use wgpu_fft::math::{from_interleaved_f64, reference_c2c_nd_f64, Complex64};
use wgpu_fft::{
    export_pipeline_cache_snapshot, import_pipeline_cache_snapshot, BufferLayout, BufferView,
    C2cRoute, FftConfig, FftError, FftIoView, FftPlan, FftPrecision, LargePolicyLimits,
    Normalization,
};

const F64_RMS_LIMIT: f64 = 1.0e-13;
const FUSED_POW2_LABEL: &str = "fused-pow2-workgroup-stage";
const FUSED_SMOOTH_LABEL: &str = "fused-smooth-workgroup-stage";
const RADER_FUSED_LABEL: &str = "rader-fused-workgroup-stage";
const BLUESTEIN_FUSED_LABEL: &str = "bluestein-fused-workgroup-stage";
const STOCKHAM_LABEL: &str = "mixed-radix-stockham-stage";

#[test]
fn gpu_native_f64_c2c_and_structured_gates() {
    if std::env::var_os("WGPU_FFT_RUN_GPU_TESTS").is_none() {
        eprintln!("skipping GPU test; set WGPU_FFT_RUN_GPU_TESTS=1 to run it");
        return;
    }

    pollster::block_on(run_f64_cases());
}

async fn run_f64_cases() {
    let Some(context) = wgpu_fft::device::request_default_device().await else {
        eprintln!("skipping GPU test; no suitable wgpu adapter was found");
        return;
    };
    // Native wgpu teardown can stall on Windows after an assertion panic. Keep
    // every device used by this opt-in integration binary alive until exit.
    let context = ManuallyDrop::new(context);
    let adapter_info = context.adapter.get_info();
    let adapter_supports_f64 = context
        .adapter
        .features()
        .contains(wgpu::Features::SHADER_F64);
    eprintln!(
        "adapter: {adapter_info:?}; SHADER_F64 available={adapter_supports_f64}; enabled={}",
        context
            .device
            .features()
            .contains(wgpu::Features::SHADER_F64)
    );

    assert_eq!(
        context.supports_precision(FftPrecision::F64),
        adapter_supports_f64,
        "the default device must enable SHADER_F64 exactly when the adapter exposes it"
    );
    verify_missing_feature_gate(&context).await;

    if !adapter_supports_f64 {
        eprintln!("skipping native-f64 execution cases: adapter does not expose wgpu SHADER_F64");
        return;
    }

    verify_deferred_route_gates(&context.device, &context.queue);
    run_small_normal_cases(&context.device, &context.queue);
    verify_precision_cache_coexistence(&context.device, &context.queue);
    run_strided_normal_case(&context.device, &context.queue);

    let low_storage = request_low_storage_f64_device(&context).await;
    run_fused_and_multipass_cases(&context, low_storage.as_ref());
    run_prime_cases(&context, low_storage.as_ref());
}

async fn verify_missing_feature_gate(context: &wgpu_fft::device::GpuContext) {
    if !context
        .adapter
        .features()
        .contains(wgpu::Features::SHADER_F64)
    {
        assert_precision_unsupported(
            FftPlan::c2c(
                &context.device,
                &context.queue,
                FftConfig::new(8).with_precision(FftPrecision::F64),
            ),
            "c2c",
            "device-missing-shader-f64",
        );
        return;
    }

    let (device, queue) = context
        .adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("wgpu_fft.test.f64_feature_disabled_device"),
            required_features: wgpu::Features::empty(),
            required_limits: context.adapter.limits(),
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: wgpu::MemoryHints::MemoryUsage,
            trace: wgpu::Trace::Off,
        })
        .await
        .expect("an adapter exposing SHADER_F64 should also allow a feature-disabled device");
    let no_feature_device = ManuallyDrop::new((device, queue));
    assert!(!no_feature_device
        .0
        .features()
        .contains(wgpu::Features::SHADER_F64));
    assert_precision_unsupported(
        FftPlan::c2c(
            &no_feature_device.0,
            &no_feature_device.1,
            FftConfig::new(8).with_precision(FftPrecision::F64),
        ),
        "c2c",
        "device-missing-shader-f64",
    );
}

fn verify_deferred_route_gates(device: &wgpu::Device, queue: &wgpu::Queue) {
    assert_precision_unsupported(
        FftPlan::r2c(
            device,
            queue,
            FftConfig::new(16).with_precision(FftPrecision::F64),
        ),
        "r2c",
        "real-f64-not-implemented",
    );
    assert_precision_unsupported(
        FftPlan::c2r(
            device,
            queue,
            FftConfig::inverse(16).with_precision(FftPrecision::F64),
        ),
        "c2r",
        "real-f64-not-implemented",
    );

    assert_precision_unsupported(
        FftPlan::c2c_with_large_policy_limits_for_testing(
            device,
            queue,
            FftConfig::new(8)
                .with_batch(2)
                .with_precision(FftPrecision::F64),
            LargePolicyLimits {
                max_storage_buffer_binding_size: 128,
                max_buffer_size: 4096,
            },
        ),
        "large-chunk",
        "large-chunk-f64-not-implemented",
    );
    assert_precision_unsupported(
        FftPlan::c2c_with_large_policy_limits_for_testing(
            device,
            queue,
            FftConfig::new_nd([8, 8]).with_precision(FftPrecision::F64),
            LargePolicyLimits {
                max_storage_buffer_binding_size: 128,
                max_buffer_size: 4096,
            },
        ),
        "out-of-core-four-step",
        "four-step-f64-not-implemented",
    );
    assert_precision_unsupported(
        FftPlan::c2c_with_large_policy_limits_for_testing(
            device,
            queue,
            FftConfig::new_nd([8, 8]).with_precision(FftPrecision::F64),
            LargePolicyLimits {
                max_storage_buffer_binding_size: 128,
                max_buffer_size: 512,
            },
        ),
        "segmented-full-volume",
        "segmented-volume-f64-not-implemented",
    );
}

fn run_small_normal_cases(device: &wgpu::Device, queue: &wgpu::Queue) {
    for (label, config, expected_route) in [
        (
            "direct-forward-n1",
            FftConfig::new(1)
                .with_normalization(Normalization::None)
                .with_precision(FftPrecision::F64),
            C2cRoute::DirectDft,
        ),
        (
            "direct-inverse-n1",
            FftConfig::inverse(1).with_precision(FftPrecision::F64),
            C2cRoute::DirectDft,
        ),
        (
            "mixed-forward-n60",
            FftConfig::new(60)
                .with_normalization(Normalization::None)
                .with_precision(FftPrecision::F64),
            C2cRoute::MixedRadix,
        ),
        (
            "mixed-inverse-n60",
            FftConfig::inverse(60).with_precision(FftPrecision::F64),
            C2cRoute::MixedRadix,
        ),
        (
            "batched-forward-n45-b3",
            FftConfig::new(45)
                .with_batch(3)
                .with_normalization(Normalization::None)
                .with_precision(FftPrecision::F64),
            C2cRoute::MixedRadix,
        ),
        (
            "batched-inverse-n45-b3",
            FftConfig::inverse(45)
                .with_batch(3)
                .with_precision(FftPrecision::F64),
            C2cRoute::MixedRadix,
        ),
        (
            "nd-forward-8x15-b2",
            FftConfig::new_nd([8, 15])
                .with_batch(2)
                .with_normalization(Normalization::None)
                .with_precision(FftPrecision::F64),
            C2cRoute::MixedRadix,
        ),
        (
            "nd-inverse-8x15-b2",
            FftConfig::inverse_nd([8, 15])
                .with_batch(2)
                .with_precision(FftPrecision::F64),
            C2cRoute::MixedRadix,
        ),
    ] {
        let (_, plan) = execute_reference_case(device, queue, label, config);
        assert_eq!(
            plan.route(),
            expected_route,
            "{label}: unexpected C2C route"
        );
    }
}

fn verify_precision_cache_coexistence(device: &wgpu::Device, queue: &wgpu::Queue) {
    let _f32 = FftPlan::c2c(
        device,
        queue,
        FftConfig::new(64).with_normalization(Normalization::None),
    )
    .unwrap();
    let _f64 = FftPlan::c2c(
        device,
        queue,
        FftConfig::new(64)
            .with_normalization(Normalization::None)
            .with_precision(FftPrecision::F64),
    )
    .unwrap();

    let snapshot = export_pipeline_cache_snapshot(device);
    let f32_key = snapshot
        .pipeline_keys()
        .iter()
        .find(|key| {
            key.contains(":n=64:")
                && key.contains("precision=f32")
                && key.contains("layout=axis-plan/interleaved-f32-lut")
        })
        .unwrap_or_else(|| panic!("missing typed f32 N=64 cache key"));
    let f64_key = snapshot
        .pipeline_keys()
        .iter()
        .find(|key| {
            key.contains(":n=64:")
                && key.contains("precision=f64")
                && key.contains("layout=axis-plan/interleaved-f64-lut")
        })
        .unwrap_or_else(|| panic!("missing typed f64 N=64 cache key"));
    assert_ne!(f32_key, f64_key);

    let imported = import_pipeline_cache_snapshot(device, &snapshot);
    assert!(imported.pipeline_keys().contains(f32_key));
    assert!(imported.pipeline_keys().contains(f64_key));
}

fn run_strided_normal_case(device: &wgpu::Device, queue: &wgpu::Queue) {
    let config = FftConfig::new(60)
        .with_batch(2)
        .with_normalization(Normalization::None)
        .with_precision(FftPrecision::F64);
    let logical_per_batch = config.logical_complex_len().unwrap() as u64;
    let batch = config.batch() as u64;
    let input_layout = strided_layout(logical_per_batch, 3, 2, 11);
    let output_layout = strided_layout(logical_per_batch, 5, 3, 13);
    let input = test_signal(config.total_complex_len().unwrap());
    let expected = reference_c2c_nd_f64(&input, &config).unwrap();
    let physical_input = scatter_strided_complex(&input, input_layout, logical_per_batch, batch);
    let input_bytes = std::mem::size_of_val(physical_input.as_slice()) as u64;
    let output_bytes = layout_span_complex(output_layout, logical_per_batch, batch) * 16;

    let input_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.f64.strided_input"),
        size: input_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.f64.strided_output"),
        size: output_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.f64.strided_readback"),
        size: output_bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    queue.write_buffer(&input_buffer, 0, bytemuck::cast_slice(&physical_input));

    let plan = FftPlan::c2c(device, queue, config).unwrap();
    assert_eq!(plan.route(), C2cRoute::MixedRadix);
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("wgpu_fft.test.f64.strided_encoder"),
    });
    plan.execute_io_views(
        device,
        &mut encoder,
        FftIoView::new(BufferView::whole(&input_buffer), input_layout).unwrap(),
        FftIoView::new(BufferView::whole(&output_buffer), output_layout).unwrap(),
    )
    .unwrap();
    encoder.copy_buffer_to_buffer(&output_buffer, 0, &readback, 0, output_bytes);
    queue.submit([encoder.finish()]);

    let physical_output = read_f64_buffer(device, &readback);
    let actual = gather_strided_f64(&physical_output, output_layout, logical_per_batch, batch);
    assert_f64_accuracy("strided-batch-stride-forward-n60-b2", &actual, &expected);
}

async fn request_low_storage_f64_device(
    context: &wgpu_fft::device::GpuContext,
) -> Option<ManuallyDrop<(wgpu::Device, wgpu::Queue)>> {
    const LOW_STORAGE_BYTES: u32 = 16 * 1024;
    if context.adapter.limits().max_compute_workgroup_storage_size < LOW_STORAGE_BYTES {
        eprintln!(
            "skipping forced-storage equivalence: adapter exposes less than 16 KiB workgroup storage"
        );
        return None;
    }

    let mut low_limits = context.adapter.limits();
    low_limits.max_compute_workgroup_storage_size = LOW_STORAGE_BYTES;
    let (device, queue) = context
        .adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("wgpu_fft.test.f64_low_workgroup_storage_device"),
            required_features: wgpu::Features::SHADER_F64,
            required_limits: low_limits,
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: wgpu::MemoryHints::MemoryUsage,
            trace: wgpu::Trace::Off,
        })
        .await
        .expect("the f64 adapter should allow a second device with a 16 KiB storage limit");
    assert_eq!(
        device.limits().max_compute_workgroup_storage_size,
        LOW_STORAGE_BYTES
    );
    assert!(device.features().contains(wgpu::Features::SHADER_F64));
    Some(ManuallyDrop::new((device, queue)))
}

fn run_fused_and_multipass_cases(
    context: &wgpu_fft::device::GpuContext,
    low_storage: Option<&ManuallyDrop<(wgpu::Device, wgpu::Queue)>>,
) {
    let config_2048 = FftConfig::new(2048)
        .with_normalization(Normalization::None)
        .with_precision(FftPrecision::F64);
    let input_2048 = test_signal(config_2048.total_complex_len().unwrap());
    let expected_2048 = reference_c2c_nd_f64(&input_2048, &config_2048).unwrap();
    let (native_2048, plan_2048) = execute_c2c_f64(
        &context.device,
        &context.queue,
        config_2048.clone(),
        &input_2048,
        "pow2-forward-n2048-native",
    );
    assert_f64_accuracy("pow2-forward-n2048-native", &native_2048, &expected_2048);
    assert_pow2_stage_selection(
        &plan_2048,
        2048,
        context.device.limits().max_compute_workgroup_storage_size,
        "pow2-forward-n2048-native",
    );

    if context.device.limits().max_compute_workgroup_storage_size >= 2048 * 16 {
        assert_fused_pow2_plan(&plan_2048, "pow2-forward-n2048-native");
    } else {
        eprintln!(
            "N=2048 is multipass on this device because native f64 scratch exceeds its storage limit"
        );
    }

    if let Some(low_storage) = low_storage {
        // Stockham coverage: keep the long axis unsplit on the low-storage device.
        let unsplit_tuning = config_2048.tuning().clone().with_fuse_long_axes(false);
        let (multipass_2048, multipass_plan) = execute_c2c_f64(
            &low_storage.0,
            &low_storage.1,
            config_2048.with_tuning(unsplit_tuning),
            &input_2048,
            "pow2-forward-n2048-forced-multipass",
        );
        assert_stockham_plan(&multipass_plan, 2048, "pow2-forward-n2048-forced-multipass");
        assert_f64_accuracy(
            "pow2-forward-n2048-forced-multipass",
            &multipass_2048,
            &expected_2048,
        );
        assert_f64_accuracy(
            "pow2-forward-n2048-fused-vs-multipass",
            &multipass_2048,
            &from_interleaved_f64(&native_2048),
        );
    }

    let config_4096 = FftConfig::inverse(4096).with_precision(FftPrecision::F64);
    let input_4096 = test_signal(config_4096.total_complex_len().unwrap());
    let expected_4096 = reference_c2c_nd_f64(&input_4096, &config_4096).unwrap();
    let (actual_4096, plan_4096) = execute_c2c_f64(
        &context.device,
        &context.queue,
        config_4096,
        &input_4096,
        "pow2-inverse-n4096-native",
    );
    assert_f64_accuracy("pow2-inverse-n4096-native", &actual_4096, &expected_4096);
    assert_pow2_stage_selection(
        &plan_4096,
        4096,
        context.device.limits().max_compute_workgroup_storage_size,
        "pow2-inverse-n4096-native",
    );

    let config_3000 = FftConfig::new(3000)
        .with_normalization(Normalization::None)
        .with_precision(FftPrecision::F64);
    let input_3000 = test_signal(config_3000.total_complex_len().unwrap());
    let expected_3000 = reference_c2c_nd_f64(&input_3000, &config_3000).unwrap();
    let (actual_3000, plan_3000) = execute_c2c_f64(
        &context.device,
        &context.queue,
        config_3000,
        &input_3000,
        "smooth-forward-n3000-native",
    );
    assert_f64_accuracy("smooth-forward-n3000-native", &actual_3000, &expected_3000);
    assert_smooth_stage_selection(
        &plan_3000,
        3000,
        context.device.limits().max_compute_workgroup_storage_size,
        "smooth-forward-n3000-native",
    );
}

fn run_prime_cases(
    context: &wgpu_fft::device::GpuContext,
    low_storage: Option<&ManuallyDrop<(wgpu::Device, wgpu::Queue)>>,
) {
    let storage_limit = context.device.limits().max_compute_workgroup_storage_size as usize;

    // Rader N=509 uses M=1024: 16*M bytes of f64 scratch plus one
    // complex-f64 reduction slot. It fuses at 48 KiB but not at 16 KiB.
    for inverse in [false, true] {
        let label = format!("rader-n509-inverse={inverse}");
        let config = prime_config_1d(509, 1, inverse);
        let (_, plan) =
            execute_prime_reference_case(&context.device, &context.queue, &label, config);
        if storage_limit >= 16 * 1024 + 16 {
            assert_fused_prime_plan(
                &plan,
                C2cRoute::Rader,
                RADER_FUSED_LABEL,
                &["rader-permutation-helper", "rader-bfft-helper"],
                &label,
            );
        } else {
            assert_rader_multipass_plan(&plan, &label);
        }
    }

    // Bluestein N=515 uses M=1029: just over 16 KiB in f64, while
    // remaining comfortably inside the RTX 5090 48 KiB limit.
    for inverse in [false, true] {
        let label = format!("bluestein-n515-inverse={inverse}");
        let config = prime_config_1d(515, 1, inverse);
        let (_, plan) =
            execute_prime_reference_case(&context.device, &context.queue, &label, config);
        if storage_limit >= 1029 * 16 {
            assert_fused_prime_plan(
                &plan,
                C2cRoute::Bluestein,
                BLUESTEIN_FUSED_LABEL,
                &["bluestein-chirp-helper", "bluestein-bfft-helper"],
                &label,
            );
        } else {
            assert_bluestein_multipass_plan(&plan, &label);
        }
    }

    // These convolution sizes fit the f32 fused gate but exceed 48 KiB
    // after native-f64 doubles each scratch element.
    let (_, rader_multipass) = execute_prime_reference_case(
        &context.device,
        &context.queue,
        "rader-n1601-native-storage-fallback",
        prime_config_1d(1601, 1, false),
    );
    if storage_limit < 3200 * 16 + 16 {
        assert_rader_multipass_plan(&rader_multipass, "rader-n1601-native-storage-fallback");
        assert!(
            kernel_labels(&rader_multipass)
                .iter()
                .filter(|stage| stage.starts_with("rader-forward-") && stage.ends_with("-stage"))
                .count()
                >= 2,
            "Rader N=1601 should also run its child FFTs in several passes at this storage limit"
        );
    } else {
        assert_fused_prime_plan(
            &rader_multipass,
            C2cRoute::Rader,
            RADER_FUSED_LABEL,
            &["rader-permutation-helper", "rader-bfft-helper"],
            "rader-n1601-native-storage-fallback",
        );
    }

    let (_, bluestein_multipass) = execute_prime_reference_case(
        &context.device,
        &context.queue,
        "bluestein-n1544-native-storage-fallback",
        prime_config_1d(1544, 1, false),
    );
    if storage_limit < 3087 * 16 {
        assert_bluestein_multipass_plan(
            &bluestein_multipass,
            "bluestein-n1544-native-storage-fallback",
        );
        assert!(
            kernel_labels(&bluestein_multipass)
                .iter()
                .filter(|stage| stage.starts_with("bluestein-forward-") && stage.ends_with("-stage"))
                .count()
                >= 2,
            "Bluestein N=1544 should also run its child FFTs in several passes at this storage limit"
        );
    } else {
        assert_fused_prime_plan(
            &bluestein_multipass,
            C2cRoute::Bluestein,
            BLUESTEIN_FUSED_LABEL,
            &["bluestein-chirp-helper", "bluestein-bfft-helper"],
            "bluestein-n1544-native-storage-fallback",
        );
    }

    for (shape, prime_route, prime_stage, prime_helpers) in [
        (
            [2, 509],
            C2cRoute::Rader,
            RADER_FUSED_LABEL,
            &["rader-permutation-helper", "rader-bfft-helper"][..],
        ),
        (
            [2, 515],
            C2cRoute::Bluestein,
            BLUESTEIN_FUSED_LABEL,
            &["bluestein-chirp-helper", "bluestein-bfft-helper"][..],
        ),
    ] {
        for inverse in [false, true] {
            let label = format!("axis-sequence-{shape:?}-batch2-inverse={inverse}");
            let config = if inverse {
                FftConfig::inverse_nd(shape)
            } else {
                FftConfig::new_nd(shape).with_normalization(Normalization::None)
            }
            .with_batch(2)
            .with_precision(FftPrecision::F64);
            let (_, plan) =
                execute_prime_reference_case(&context.device, &context.queue, &label, config);
            assert_axis_sequence_prime_plan(&plan, prime_route, prime_stage, prime_helpers, &label);
        }
    }

    if let Some(low_storage) = low_storage {
        if storage_limit >= 16 * 1024 + 16 {
            compare_fused_prime_with_low_storage(
                context,
                low_storage,
                509,
                C2cRoute::Rader,
                RADER_FUSED_LABEL,
                "rader-n509-fused-vs-16k",
            );
        }
        if storage_limit >= 1029 * 16 {
            compare_fused_prime_with_low_storage(
                context,
                low_storage,
                515,
                C2cRoute::Bluestein,
                BLUESTEIN_FUSED_LABEL,
                "bluestein-n515-fused-vs-16k",
            );
        }
    }
}

fn prime_config_1d(length: usize, batch: usize, inverse: bool) -> FftConfig {
    if inverse {
        FftConfig::inverse(length)
    } else {
        FftConfig::new(length).with_normalization(Normalization::None)
    }
    .with_batch(batch)
    .with_precision(FftPrecision::F64)
}

fn execute_prime_reference_case(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &str,
    config: FftConfig,
) -> (Vec<f64>, FftPlan) {
    let input = test_signal(config.total_complex_len().unwrap());
    let expected = reference_c2c_nd_f64(&input, &config).unwrap();
    let (actual, plan) = execute_c2c_f64(device, queue, config, &input, label);
    assert_f64_accuracy(label, &actual, &expected);
    (actual, plan)
}

fn compare_fused_prime_with_low_storage(
    context: &wgpu_fft::device::GpuContext,
    low_storage: &ManuallyDrop<(wgpu::Device, wgpu::Queue)>,
    length: usize,
    route: C2cRoute,
    fused_stage: &str,
    label: &str,
) {
    let config = prime_config_1d(length, 1, false);
    let input = test_signal(length);
    let expected = reference_c2c_nd_f64(&input, &config).unwrap();
    let (fused, fused_plan) = execute_c2c_f64(
        &context.device,
        &context.queue,
        config.clone(),
        &input,
        label,
    );
    let (multipass, multipass_plan) =
        execute_c2c_f64(&low_storage.0, &low_storage.1, config, &input, label);
    let helpers = match route {
        C2cRoute::Rader => &["rader-permutation-helper", "rader-bfft-helper"][..],
        C2cRoute::Bluestein => &["bluestein-chirp-helper", "bluestein-bfft-helper"][..],
        _ => unreachable!("prime comparison route"),
    };
    assert_fused_prime_plan(&fused_plan, route, fused_stage, helpers, label);
    match route {
        C2cRoute::Rader => assert_rader_multipass_plan(&multipass_plan, label),
        C2cRoute::Bluestein => assert_bluestein_multipass_plan(&multipass_plan, label),
        _ => unreachable!("prime comparison route"),
    }
    assert_f64_accuracy(&format!("{label}-fused"), &fused, &expected);
    assert_f64_accuracy(&format!("{label}-multipass"), &multipass, &expected);
    assert_f64_accuracy(
        &format!("{label}-equivalence"),
        &multipass,
        &from_interleaved_f64(&fused),
    );
}

fn execute_reference_case(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &str,
    config: FftConfig,
) -> (Vec<f64>, FftPlan) {
    let input = test_signal(config.total_complex_len().unwrap());
    let expected = reference_c2c_nd_f64(&input, &config).unwrap();
    let (actual, plan) = execute_c2c_f64(device, queue, config, &input, label);
    assert_f64_accuracy(label, &actual, &expected);
    (actual, plan)
}

fn execute_c2c_f64(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    config: FftConfig,
    input: &[Complex64],
    label: &str,
) -> (Vec<f64>, FftPlan) {
    let plan = FftPlan::c2c(device, queue, config)
        .unwrap_or_else(|error| panic!("{label}: native-f64 plan creation failed: {error}"));
    let byte_len = std::mem::size_of_val(input) as u64;
    assert_eq!(
        plan.required_input_buffer_size_bytes(),
        byte_len,
        "{label}: f64 input byte size"
    );
    assert_eq!(
        plan.required_output_buffer_size_bytes(),
        byte_len,
        "{label}: f64 output byte size"
    );

    let input_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.f64.input"),
        size: byte_len,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.f64.output"),
        size: byte_len,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.f64.readback"),
        size: byte_len,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    queue.write_buffer(&input_buffer, 0, bytemuck::cast_slice(input));

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("wgpu_fft.test.f64.encoder"),
    });
    plan.execute_checked(device, &mut encoder, &input_buffer, &output_buffer)
        .unwrap_or_else(|error| panic!("{label}: native-f64 execution failed: {error}"));
    encoder.copy_buffer_to_buffer(&output_buffer, 0, &readback, 0, byte_len);
    queue.submit([encoder.finish()]);

    let values = read_f64_buffer(device, &readback);
    (values, plan)
}

fn read_f64_buffer(device: &wgpu::Device, readback: &wgpu::Buffer) -> Vec<f64> {
    let slice = readback.slice(..);
    let (sender, receiver) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        sender.send(result).unwrap();
    });
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    receiver.recv().unwrap().unwrap();
    let mapped = slice
        .get_mapped_range()
        .expect("readback buffer should be mapped");
    let values = bytemuck::cast_slice(&mapped).to_vec();
    drop(mapped);
    readback.unmap();
    values
}

fn test_signal(complex_len: usize) -> Vec<Complex64> {
    (0..complex_len)
        .map(|index| {
            let x = index as f64 + 1.0;
            Complex64::new(
                (x * 0.017).sin() * 0.7 + (x * 0.031).cos() * 0.2,
                (x * 0.023).cos() * 0.5 - (x * 0.011).sin() * 0.3,
            )
        })
        .collect()
}

fn strided_layout(
    logical_per_batch: u64,
    element_offset: u64,
    element_stride: u64,
    batch_gap: u64,
) -> BufferLayout {
    let per_batch_span = element_stride * (logical_per_batch - 1) + 1;
    BufferLayout::new(element_offset, element_stride)
        .unwrap()
        .with_batch_stride(per_batch_span + batch_gap)
}

fn layout_span_complex(layout: BufferLayout, logical_per_batch: u64, batch: u64) -> u64 {
    let per_batch_span = layout.element_stride * (logical_per_batch - 1) + 1;
    let batch_stride = layout.batch_stride.unwrap_or(per_batch_span);
    layout.element_offset + (batch - 1) * batch_stride + per_batch_span
}

fn physical_complex_index(
    layout: BufferLayout,
    logical_per_batch: u64,
    logical_index: u64,
) -> usize {
    let per_batch_span = layout.element_stride * (logical_per_batch - 1) + 1;
    let batch_stride = layout.batch_stride.unwrap_or(per_batch_span);
    let batch = logical_index / logical_per_batch;
    let element = logical_index - batch * logical_per_batch;
    (layout.element_offset + batch * batch_stride + element * layout.element_stride) as usize
}

fn scatter_strided_complex(
    logical: &[Complex64],
    layout: BufferLayout,
    logical_per_batch: u64,
    batch: u64,
) -> Vec<Complex64> {
    let mut physical =
        vec![Complex64::default(); layout_span_complex(layout, logical_per_batch, batch) as usize];
    for (logical_index, value) in logical.iter().copied().enumerate() {
        let physical_index =
            physical_complex_index(layout, logical_per_batch, logical_index as u64);
        physical[physical_index] = value;
    }
    physical
}

fn gather_strided_f64(
    physical: &[f64],
    layout: BufferLayout,
    logical_per_batch: u64,
    batch: u64,
) -> Vec<f64> {
    let logical_complex = logical_per_batch * batch;
    let mut logical = vec![0.0; logical_complex as usize * 2];
    for logical_index in 0..logical_complex {
        let physical_index = physical_complex_index(layout, logical_per_batch, logical_index);
        logical[logical_index as usize * 2] = physical[physical_index * 2];
        logical[logical_index as usize * 2 + 1] = physical[physical_index * 2 + 1];
    }
    logical
}

fn assert_f64_accuracy(label: &str, actual: &[f64], expected: &[Complex64]) {
    assert_eq!(actual.len(), expected.len() * 2, "{label}: output length");
    let mut max_abs = 0.0f64;
    let mut error_energy = 0.0f64;
    let mut reference_energy = 0.0f64;
    let mut reference_max = 0.0f64;

    for (pair, expected) in actual.chunks_exact(2).zip(expected) {
        let error = (pair[0] - expected.re).hypot(pair[1] - expected.im);
        let magnitude = expected.re.hypot(expected.im);
        max_abs = max_abs.max(error);
        error_energy += error * error;
        reference_energy += magnitude * magnitude;
        reference_max = reference_max.max(magnitude);
    }

    let rms_relative = (error_energy / reference_energy.max(f64::MIN_POSITIVE)).sqrt();
    let max_relative = max_abs / reference_max.max(f64::MIN_POSITIVE);
    eprintln!(
        "F64_ACCURACY label={label:?} max_relative={max_relative:.9e} rms_relative={rms_relative:.9e}"
    );
    assert!(
        rms_relative.is_finite() && rms_relative <= F64_RMS_LIMIT,
        "{label}: RMS relative error {rms_relative:.9e} exceeds {F64_RMS_LIMIT:.1e}"
    );
}

fn assert_pow2_stage_selection(plan: &FftPlan, len: usize, storage_limit: u32, label: &str) {
    if len * 16 <= storage_limit as usize {
        assert_fused_pow2_plan(plan, label);
    } else {
        assert_split_plan(plan, len, FUSED_POW2_LABEL, label);
    }
}

fn assert_smooth_stage_selection(plan: &FftPlan, len: usize, storage_limit: u32, label: &str) {
    if len * 16 <= storage_limit as usize {
        let kernels = kernel_labels(plan);
        assert_eq!(
            kernels,
            vec![FUSED_SMOOTH_LABEL.to_owned()],
            "{label}: expected one native-f64 fused smooth stage"
        );
        assert_eq!(plan.workspace_size_bytes(), 0, "{label}: fused workspace");
    } else {
        assert_split_plan(plan, len, FUSED_SMOOTH_LABEL, label);
    }
}

fn assert_fused_pow2_plan(plan: &FftPlan, label: &str) {
    let kernels = kernel_labels(plan);
    assert_eq!(
        kernels,
        vec![FUSED_POW2_LABEL.to_owned()],
        "{label}: expected one native-f64 fused pow2 stage"
    );
    assert_eq!(plan.workspace_size_bytes(), 0, "{label}: fused workspace");
}

/// Axes too long for one workgroup run as two fused passes (`N = N1 * N2`).
fn assert_split_plan(plan: &FftPlan, len: usize, fused_label: &str, label: &str) {
    assert_eq!(
        kernel_labels(plan),
        vec![fused_label; 2],
        "{label}: N={len} should split into two fused passes"
    );
    assert_eq!(
        plan.workspace_size_bytes(),
        (len * 16) as u64,
        "{label}: split workspace"
    );
}

fn assert_stockham_plan(plan: &FftPlan, len: usize, label: &str) {
    let kernels = kernel_labels(plan);
    assert_eq!(
        kernels.len(),
        plan.factors().len(),
        "{label}: N={len} should expose one kernel per Stockham factor"
    );
    assert!(
        kernels.iter().all(|stage| stage == STOCKHAM_LABEL),
        "{label}: expected only native-f64 Stockham stages, got {kernels:?}"
    );
    assert_eq!(
        plan.workspace_size_bytes(),
        (len * 16) as u64,
        "{label}: native-f64 multipass workspace"
    );
}

fn assert_fused_prime_plan(
    plan: &FftPlan,
    route: C2cRoute,
    fused_stage: &str,
    expected_helpers: &[&str],
    label: &str,
) {
    assert_eq!(plan.route(), route, "{label}: prime route");
    assert_eq!(
        kernel_labels(plan),
        vec![fused_stage.to_owned()],
        "{label}: fused prime kernel inventory"
    );
    assert_eq!(
        helper_labels(plan),
        expected_helpers
            .iter()
            .map(|helper| (*helper).to_owned())
            .collect::<Vec<_>>(),
        "{label}: fused prime helper inventory"
    );
    assert_eq!(plan.workspace_size_bytes(), 0, "{label}: plan workspace");
    assert!(
        plan.diagnostics().blockers().is_empty(),
        "{label}: blockers"
    );
    match route {
        C2cRoute::Rader => assert_helper_format(plan, "rader-bfft-helper", "complex-f64", label),
        C2cRoute::Bluestein => {
            assert_helper_format(plan, "bluestein-chirp-helper", "complex-f64", label);
            assert_helper_format(plan, "bluestein-bfft-helper", "complex-f64", label);
        }
        _ => unreachable!("fused prime route"),
    }
}

fn assert_rader_multipass_plan(plan: &FftPlan, label: &str) {
    assert_eq!(plan.route(), C2cRoute::Rader, "{label}: Rader route");
    let kernels = kernel_labels(plan);
    assert!(
        !kernels.iter().any(|stage| stage == RADER_FUSED_LABEL),
        "{label}: whole-pipeline Rader fusion should be disabled: {kernels:?}"
    );
    for expected in [
        "rader-sum",
        "rader-pack",
        "rader-mul",
        "rader-write-y0",
        "rader-post",
    ] {
        assert!(
            kernels.iter().any(|stage| stage == expected),
            "{label}: missing {expected}: {kernels:?}"
        );
    }
    let helpers = helper_labels(plan);
    for expected in [
        "rader-permutation-helper",
        "rader-bfft-helper",
        "rader-sum-helper",
        "rader-x0-helper",
        "rader-work-helper",
        "rader-fft-helper",
    ] {
        assert!(
            helpers.iter().any(|helper| helper == expected),
            "{label}: missing {expected}: {helpers:?}"
        );
    }
    assert_eq!(plan.workspace_size_bytes(), 0, "{label}: plan workspace");
    assert!(
        plan.diagnostics().blockers().is_empty(),
        "{label}: blockers"
    );
    for helper in [
        "rader-bfft-helper",
        "rader-sum-helper",
        "rader-x0-helper",
        "rader-work-helper",
        "rader-fft-helper",
    ] {
        assert_helper_format(plan, helper, "complex-f64", label);
    }
}

fn assert_bluestein_multipass_plan(plan: &FftPlan, label: &str) {
    assert_eq!(
        plan.route(),
        C2cRoute::Bluestein,
        "{label}: Bluestein route"
    );
    let kernels = kernel_labels(plan);
    assert!(
        !kernels.iter().any(|stage| stage == BLUESTEIN_FUSED_LABEL),
        "{label}: whole-pipeline Bluestein fusion should be disabled: {kernels:?}"
    );
    for expected in ["bluestein-pack", "bluestein-mul", "bluestein-post"] {
        assert!(
            kernels.iter().any(|stage| stage == expected),
            "{label}: missing {expected}: {kernels:?}"
        );
    }
    let helpers = helper_labels(plan);
    for expected in [
        "bluestein-chirp-helper",
        "bluestein-bfft-helper",
        "bluestein-work-helper",
        "bluestein-fft-helper",
    ] {
        assert!(
            helpers.iter().any(|helper| helper == expected),
            "{label}: missing {expected}: {helpers:?}"
        );
        assert_helper_format(plan, expected, "complex-f64", label);
    }
    assert_eq!(plan.workspace_size_bytes(), 0, "{label}: plan workspace");
    assert!(
        plan.diagnostics().blockers().is_empty(),
        "{label}: blockers"
    );
}

fn assert_axis_sequence_prime_plan(
    plan: &FftPlan,
    prime_route: C2cRoute,
    prime_stage: &str,
    prime_helpers: &[&str],
    label: &str,
) {
    assert_eq!(
        plan.route(),
        C2cRoute::AxisSequence,
        "{label}: AxisSequence route"
    );
    assert!(matches!(prime_route, C2cRoute::Rader | C2cRoute::Bluestein));
    assert_eq!(
        kernel_labels(plan),
        vec![
            "axis-sequence-mixed-fused-pow2-stage".to_owned(),
            prime_stage.to_owned(),
        ],
        "{label}: AxisSequence kernel order"
    );
    let mut expected_helpers = vec!["axis-sequence-workspace".to_owned()];
    expected_helpers.extend(prime_helpers.iter().map(|helper| (*helper).to_owned()));
    assert_eq!(
        helper_labels(plan),
        expected_helpers,
        "{label}: AxisSequence helper inventory"
    );
    assert_eq!(
        plan.workspace_size_bytes(),
        plan.required_input_buffer_size_bytes(),
        "{label}: AxisSequence workspace must hold one complete f64 volume"
    );
    assert!(
        plan.diagnostics().blockers().is_empty(),
        "{label}: blockers"
    );
    assert_helper_format(plan, "axis-sequence-workspace", "complex-f64", label);
    match prime_route {
        C2cRoute::Rader => assert_helper_format(plan, "rader-bfft-helper", "complex-f64", label),
        C2cRoute::Bluestein => {
            assert_helper_format(plan, "bluestein-chirp-helper", "complex-f64", label);
            assert_helper_format(plan, "bluestein-bfft-helper", "complex-f64", label);
        }
        _ => unreachable!("AxisSequence prime route"),
    }
}

fn assert_helper_format(plan: &FftPlan, helper: &str, format: &str, label: &str) {
    let role = format!("helper:{helper}");
    let diagnostics = plan.diagnostics();
    let requirement = diagnostics
        .buffer_requirements()
        .iter()
        .find(|requirement| requirement.role == role)
        .unwrap_or_else(|| panic!("{label}: missing buffer requirement {role}"));
    assert_eq!(requirement.format, format, "{label}: {role} format");
}

fn kernel_labels(plan: &FftPlan) -> Vec<String> {
    plan.diagnostics()
        .stages()
        .iter()
        .filter(|stage| stage.kind == "kernel")
        .map(|stage| stage.label.clone())
        .collect()
}

fn helper_labels(plan: &FftPlan) -> Vec<String> {
    plan.diagnostics()
        .stages()
        .iter()
        .filter(|stage| stage.kind == "helper-buffer-window")
        .map(|stage| stage.label.clone())
        .collect()
}

fn assert_precision_unsupported(
    result: Result<FftPlan, FftError>,
    expected_route: &'static str,
    expected_reason: &'static str,
) {
    let error = match result {
        Ok(_) => panic!(
            "expected PrecisionUnsupported for route={expected_route} reason={expected_reason}"
        ),
        Err(error) => error,
    };
    assert_eq!(
        error,
        FftError::PrecisionUnsupported {
            requested: FftPrecision::F64,
            route: expected_route,
            reason: expected_reason,
        }
    );
}
