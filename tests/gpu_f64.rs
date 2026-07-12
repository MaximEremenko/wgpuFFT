//! Opt-in native-f64 GPU correctness, routing, and capability coverage.

use std::mem::ManuallyDrop;
use std::sync::mpsc;

use wgpu_fft::math::{from_interleaved_f64, reference_c2c_nd_f64, Complex64};
use wgpu_fft::{
    BufferLayout, BufferView, C2cRoute, FftConfig, FftError, FftIoView, FftPlan, FftPrecision,
    LargePolicyLimits, Normalization,
};

const F64_RMS_LIMIT: f64 = 1.0e-13;
const FUSED_POW2_LABEL: &str = "fused-pow2-workgroup-stage";
const FUSED_SMOOTH_LABEL: &str = "fused-smooth-workgroup-stage";
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
    run_strided_normal_case(&context.device, &context.queue);

    let low_storage = request_low_storage_f64_device(&context).await;
    run_fused_and_multipass_cases(&context, low_storage.as_ref());
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
        let (multipass_2048, multipass_plan) = execute_c2c_f64(
            &low_storage.0,
            &low_storage.1,
            config_2048,
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
        assert_stockham_plan(plan, len, label);
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
        assert_stockham_plan(plan, len, label);
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

fn kernel_labels(plan: &FftPlan) -> Vec<String> {
    plan.diagnostics()
        .stages()
        .iter()
        .filter(|stage| stage.kind == "kernel")
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
