//! Opt-in portable-df64 normal C2C correctness and routing coverage.

use std::mem::ManuallyDrop;
use std::sync::mpsc;

use wgpu_fft::math::{reference_c2c_nd_f64, Complex64, ComplexDoubleFloat};
use wgpu_fft::{C2cRoute, FftConfig, FftError, FftPlan, FftPrecision, Normalization};

const DF64_RMS_LIMIT: f64 = 1.0e-11;
const FUSED_POW2_LABEL: &str = "fused-pow2-workgroup-stage";
const FUSED_SMOOTH_LABEL: &str = "fused-smooth-workgroup-stage";
const STOCKHAM_LABEL: &str = "mixed-radix-stockham-stage";

#[test]
fn gpu_portable_df64_normal_c2c() {
    if std::env::var_os("WGPU_FFT_RUN_GPU_TESTS").is_none() {
        eprintln!("skipping GPU test; set WGPU_FFT_RUN_GPU_TESTS=1 to run it");
        return;
    }
    pollster::block_on(run_df64_cases());
}

async fn run_df64_cases() {
    let Some(context) = wgpu_fft::device::request_default_device().await else {
        eprintln!("skipping GPU test; no suitable wgpu adapter was found");
        return;
    };
    // Native wgpu teardown can stall on Windows after an assertion panic.
    let context = ManuallyDrop::new(context);
    let info = context.adapter.get_info();
    eprintln!(
        "df64 adapter: name={:?}, backend={:?}, device_type={:?}, driver={:?}",
        info.name, info.backend, info.device_type, info.driver
    );

    let primary = request_featureless_device(
        &context.adapter,
        context.adapter.limits(),
        "wgpu_fft.test.df64_featureless_device",
    )
    .await;
    assert!(primary.0.features().is_empty());
    verify_structured_phase_b_gates(&primary.0, &primary.1);
    if info.backend == wgpu::Backend::Dx12 {
        eprintln!(
            "df64 DX12 coverage: exact arithmetic canaries plus a representative fused-pow2 C2C case; exhaustive topology coverage runs on Vulkan"
        );
        run_dx12_normal_cases(&primary.0, &primary.1);
        return;
    }
    run_small_normal_cases(&primary.0, &primary.1);
    run_fused_and_multipass_cases(&context.adapter, &primary).await;
}

async fn request_featureless_device(
    adapter: &wgpu::Adapter,
    limits: wgpu::Limits,
    label: &'static str,
) -> ManuallyDrop<(wgpu::Device, wgpu::Queue)> {
    let pair = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some(label),
            required_features: wgpu::Features::empty(),
            required_limits: limits,
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: wgpu::MemoryHints::MemoryUsage,
            trace: wgpu::Trace::Off,
        })
        .await
        .expect("portable df64 requires no optional device features");
    ManuallyDrop::new(pair)
}

fn verify_structured_phase_b_gates(device: &wgpu::Device, queue: &wgpu::Queue) {
    for result in [
        FftPlan::r2c(
            device,
            queue,
            FftConfig::new(16).with_precision(FftPrecision::Df64),
        ),
        FftPlan::c2c(
            device,
            queue,
            FftConfig::new(17).with_precision(FftPrecision::Df64),
        ),
    ] {
        assert!(matches!(
            result,
            Err(FftError::PrecisionUnsupported {
                requested: FftPrecision::Df64,
                ..
            })
        ));
    }
}

fn run_small_normal_cases(device: &wgpu::Device, queue: &wgpu::Queue) {
    for (label, config, expected_route) in [
        (
            "direct-forward-n1",
            FftConfig::new(1)
                .with_normalization(Normalization::None)
                .with_precision(FftPrecision::Df64),
            C2cRoute::DirectDft,
        ),
        (
            "direct-inverse-n1",
            FftConfig::inverse(1).with_precision(FftPrecision::Df64),
            C2cRoute::DirectDft,
        ),
        (
            "mixed-forward-n60",
            FftConfig::new(60)
                .with_normalization(Normalization::None)
                .with_precision(FftPrecision::Df64),
            C2cRoute::MixedRadix,
        ),
        (
            "mixed-inverse-n60",
            FftConfig::inverse(60).with_precision(FftPrecision::Df64),
            C2cRoute::MixedRadix,
        ),
        (
            "batched-forward-n45-b3",
            FftConfig::new(45)
                .with_batch(3)
                .with_normalization(Normalization::None)
                .with_precision(FftPrecision::Df64),
            C2cRoute::MixedRadix,
        ),
        (
            "nd-inverse-8x15-b2",
            FftConfig::inverse_nd([8, 15])
                .with_batch(2)
                .with_precision(FftPrecision::Df64),
            C2cRoute::MixedRadix,
        ),
    ] {
        let (_, plan) = execute_reference_case(device, queue, label, config);
        assert_eq!(plan.route(), expected_route, "{label}: route");
    }
}

fn run_dx12_normal_cases(device: &wgpu::Device, queue: &wgpu::Queue) {
    let label = "dx12-fused-pow2-inverse-n2";
    let (_, plan) = execute_reference_case(
        device,
        queue,
        label,
        FftConfig::inverse(2).with_precision(FftPrecision::Df64),
    );
    assert_eq!(plan.route(), C2cRoute::MixedRadix, "{label}: route");
    assert_eq!(kernel_labels(&plan), vec![FUSED_POW2_LABEL]);
}

async fn run_fused_and_multipass_cases(
    adapter: &wgpu::Adapter,
    primary: &ManuallyDrop<(wgpu::Device, wgpu::Queue)>,
) {
    let storage_limit = primary.0.limits().max_compute_workgroup_storage_size;
    let config_2048 = FftConfig::new(2048)
        .with_normalization(Normalization::None)
        .with_precision(FftPrecision::Df64);
    let input_2048 = test_signal(2048);
    let expected_2048 = sampled_reference_1d(&input_2048, &config_2048, 64);
    let (fused, fused_plan) = execute_c2c_df64(
        &primary.0,
        &primary.1,
        config_2048.clone(),
        &input_2048,
        "pow2-forward-n2048",
    );
    assert_sampled_df64_accuracy("pow2-forward-n2048", &fused, &expected_2048);
    if storage_limit >= 2048 * 16 {
        assert_eq!(kernel_labels(&fused_plan), vec![FUSED_POW2_LABEL]);
        assert_eq!(fused_plan.workspace_size_bytes(), 0);
    }

    if adapter.limits().max_compute_workgroup_storage_size >= 16 * 1024 {
        let mut low_limits = adapter.limits();
        low_limits.max_compute_workgroup_storage_size = 16 * 1024;
        let low = request_featureless_device(
            adapter,
            low_limits,
            "wgpu_fft.test.df64_low_storage_device",
        )
        .await;
        let (multipass, multipass_plan) = execute_c2c_df64(
            &low.0,
            &low.1,
            config_2048,
            &input_2048,
            "pow2-forward-n2048-forced-multipass",
        );
        assert_stockham_plan(&multipass_plan, 2048, "forced-multipass");
        assert_sampled_df64_accuracy("pow2-forward-n2048-forced", &multipass, &expected_2048);
        let fused_reference = fused
            .iter()
            .map(|value| Complex64::new(value.re().to_f64(), value.im().to_f64()))
            .collect::<Vec<_>>();
        assert_df64_accuracy("pow2-fused-vs-multipass", &multipass, &fused_reference);
    }

    let config_4096 = FftConfig::inverse(4096).with_precision(FftPrecision::Df64);
    let input_4096 = test_signal(4096);
    let expected_4096 = sampled_reference_1d(&input_4096, &config_4096, 64);
    let (actual_4096, plan_4096) = execute_c2c_df64(
        &primary.0,
        &primary.1,
        config_4096,
        &input_4096,
        "pow2-inverse-n4096",
    );
    assert_sampled_df64_accuracy("pow2-inverse-n4096", &actual_4096, &expected_4096);
    assert_stockham_plan(&plan_4096, 4096, "pow2-inverse-n4096");

    let config_3000 = FftConfig::new(3000)
        .with_normalization(Normalization::None)
        .with_precision(FftPrecision::Df64);
    let input_3000 = test_signal(3000);
    let expected_3000 = sampled_reference_1d(&input_3000, &config_3000, 64);
    let (actual_3000, smooth_plan) = execute_c2c_df64(
        &primary.0,
        &primary.1,
        config_3000,
        &input_3000,
        "smooth-forward-n3000",
    );
    assert_sampled_df64_accuracy("smooth-forward-n3000", &actual_3000, &expected_3000);
    if storage_limit >= 3000 * 16 {
        assert_eq!(kernel_labels(&smooth_plan), vec![FUSED_SMOOTH_LABEL]);
        assert_eq!(smooth_plan.workspace_size_bytes(), 0);
    } else {
        assert_stockham_plan(&smooth_plan, 3000, "smooth-forward-n3000");
    }
}

fn execute_reference_case(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &str,
    config: FftConfig,
) -> (Vec<ComplexDoubleFloat>, FftPlan) {
    let input = test_signal(config.total_complex_len().unwrap());
    let expected = reference_c2c_nd_f64(&input, &config).unwrap();
    let (actual, plan) = execute_c2c_df64(device, queue, config, &input, label);
    assert_df64_accuracy(label, &actual, &expected);
    (actual, plan)
}

fn execute_c2c_df64(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    config: FftConfig,
    input: &[Complex64],
    label: &str,
) -> (Vec<ComplexDoubleFloat>, FftPlan) {
    let plan = FftPlan::c2c(device, queue, config)
        .unwrap_or_else(|error| panic!("{label}: df64 plan creation failed: {error}"));
    let input = input
        .iter()
        .map(|value| ComplexDoubleFloat::from_f64(value.re, value.im))
        .collect::<Vec<_>>();
    let byte_len = std::mem::size_of_val(input.as_slice()) as u64;
    assert_eq!(plan.required_input_buffer_size_bytes(), byte_len);
    assert_eq!(plan.required_output_buffer_size_bytes(), byte_len);

    let input_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.df64.input"),
        size: byte_len,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.df64.output"),
        size: byte_len,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.df64.readback"),
        size: byte_len,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    queue.write_buffer(&input_buffer, 0, bytemuck::cast_slice(&input));

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("wgpu_fft.test.df64.encoder"),
    });
    plan.execute_checked(device, &mut encoder, &input_buffer, &output_buffer)
        .unwrap_or_else(|error| panic!("{label}: df64 execution failed: {error}"));
    encoder.copy_buffer_to_buffer(&output_buffer, 0, &readback, 0, byte_len);
    queue.submit([encoder.finish()]);

    (read_df64(device, &readback), plan)
}

fn read_df64(device: &wgpu::Device, readback: &wgpu::Buffer) -> Vec<ComplexDoubleFloat> {
    let slice = readback.slice(..);
    let (sender, receiver) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        sender.send(result).unwrap();
    });
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    receiver.recv().unwrap().unwrap();
    let mapped = slice.get_mapped_range();
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

fn assert_df64_accuracy(label: &str, actual: &[ComplexDoubleFloat], expected: &[Complex64]) {
    assert_eq!(actual.len(), expected.len(), "{label}: output length");
    let mut max_abs = 0.0f64;
    let mut error_energy = 0.0f64;
    let mut reference_energy = 0.0f64;
    let mut reference_max = 0.0f64;
    for (actual, expected) in actual.iter().zip(expected) {
        let error = (actual.re().to_f64() - expected.re).hypot(actual.im().to_f64() - expected.im);
        let magnitude = expected.re.hypot(expected.im);
        max_abs = max_abs.max(error);
        error_energy += error * error;
        reference_energy += magnitude * magnitude;
        reference_max = reference_max.max(magnitude);
    }
    let rms_relative = (error_energy / reference_energy.max(f64::MIN_POSITIVE)).sqrt();
    let max_relative = max_abs / reference_max.max(f64::MIN_POSITIVE);
    eprintln!(
        "DF64_ACCURACY label={label:?} max_relative={max_relative:.9e} rms_relative={rms_relative:.9e}"
    );
    assert!(
        rms_relative.is_finite() && rms_relative <= DF64_RMS_LIMIT,
        "{label}: RMS relative error {rms_relative:.9e} exceeds {DF64_RMS_LIMIT:.1e}"
    );
}

fn sampled_reference_1d(
    input: &[Complex64],
    config: &FftConfig,
    sample_count: usize,
) -> Vec<(usize, Complex64)> {
    assert_eq!(config.shape(), [input.len()]);
    assert_eq!(config.batch(), 1);
    let len = input.len();
    let step = len.div_ceil(sample_count).max(1);
    let sign = if config.direction() == wgpu_fft::FftDirection::Forward {
        -1.0
    } else {
        1.0
    };
    let scale = config.scale_f64().unwrap();
    (0..len)
        .step_by(step)
        .map(|k| {
            let mut sum = Complex64::default();
            for (n, value) in input.iter().enumerate() {
                let exponent = ((k as u128 * n as u128) % len as u128) as f64;
                let angle = sign * std::f64::consts::TAU * exponent / len as f64;
                let (sin, cos) = angle.sin_cos();
                sum.re += value.re * cos - value.im * sin;
                sum.im += value.re * sin + value.im * cos;
            }
            (k, Complex64::new(sum.re * scale, sum.im * scale))
        })
        .collect()
}

fn assert_sampled_df64_accuracy(
    label: &str,
    actual: &[ComplexDoubleFloat],
    expected: &[(usize, Complex64)],
) {
    let actual = expected
        .iter()
        .map(|&(index, _)| actual[index])
        .collect::<Vec<_>>();
    let expected = expected.iter().map(|&(_, value)| value).collect::<Vec<_>>();
    assert_df64_accuracy(label, &actual, &expected);
}

fn assert_stockham_plan(plan: &FftPlan, len: usize, label: &str) {
    let kernels = kernel_labels(plan);
    assert_eq!(kernels.len(), plan.factors().len(), "{label}: N={len}");
    assert!(
        kernels.iter().all(|stage| stage == STOCKHAM_LABEL),
        "{label}: expected Stockham stages, got {kernels:?}"
    );
    assert_eq!(plan.workspace_size_bytes(), (len * 16) as u64);
}

fn kernel_labels(plan: &FftPlan) -> Vec<String> {
    plan.diagnostics()
        .stages()
        .iter()
        .filter(|stage| stage.kind == "kernel")
        .map(|stage| stage.label.clone())
        .collect()
}
