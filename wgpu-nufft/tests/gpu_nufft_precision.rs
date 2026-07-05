#![cfg(not(target_arch = "wasm32"))]

//! Opt-in native-f64 NUFFT correctness, layout, and capability coverage.

use std::f64::consts::PI;
use std::mem::ManuallyDrop;
use std::sync::mpsc;

use wgpu::util::DeviceExt;
use wgpu_nufft::{
    reference_type1_f64, reference_type2_f64, reference_type3_f64, Complex64, FftPrecision,
    ModeOrder, NufftConfig, NufftError, NufftInterval, NufftPlan, NufftSign, NufftType3Config,
    NufftType3Plan,
};

const EPS: f64 = 1.0e-8;
const ORACLE_TOLERANCE: f64 = 20.0 * EPS;
const ADJOINT_TOLERANCE: f64 = 40.0 * EPS;
const HIGH_ACCURACY_EPS: f64 = 1.0e-12;
const HIGH_ACCURACY_TOLERANCE: f64 = 40.0 * HIGH_ACCURACY_EPS;
const BATCH: usize = 2;

#[test]
fn gpu_native_f64_nufft_matches_direct_oracles_and_adjoint_identities() {
    if std::env::var_os("WGPU_FFT_RUN_GPU_TESTS").is_none() {
        eprintln!("skipping GPU test; set WGPU_FFT_RUN_GPU_TESTS=1 to run it");
        return;
    }
    pollster::block_on(run_native_f64_cases());
}

struct VulkanContext {
    _instance: wgpu::Instance,
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
}

async fn run_native_f64_cases() {
    let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
    descriptor.backends = wgpu::Backends::VULKAN;
    let instance = wgpu::Instance::new(descriptor);
    let adapters = instance
        .enumerate_adapters(wgpu::Backends::VULKAN)
        .await
        .into_iter()
        .collect::<Vec<_>>();
    let adapter = adapters
        .into_iter()
        .max_by_key(|candidate| {
            (
                usize::from(candidate.features().contains(wgpu::Features::SHADER_F64)),
                usize::from(candidate.get_info().device_type == wgpu::DeviceType::DiscreteGpu),
            )
        })
        .unwrap_or_else(|| panic!("GPU tests were requested but no Vulkan adapter was found"));
    let info = adapter.get_info();
    assert_eq!(info.backend, wgpu::Backend::Vulkan);
    let adapter_supports_f64 = adapter.features().contains(wgpu::Features::SHADER_F64);
    eprintln!(
        "gpu_nufft_precision adapter: {} backend={:?} driver={} {} SHADER_F64_available={adapter_supports_f64}",
        info.name, info.backend, info.driver, info.driver_info
    );

    let required_features = adapter.features() & wgpu::Features::SHADER_F64;
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("wgpu_nufft.precision.f64.device"),
            required_features,
            required_limits: adapter.limits(),
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
        })
        .await
        .expect("requesting the selected Vulkan adapter must succeed");
    let context = ManuallyDrop::new(VulkanContext {
        _instance: instance,
        adapter,
        device,
        queue,
    });
    eprintln!(
        "gpu_nufft_precision enabled_features={:?}",
        context.device.features()
    );

    verify_feature_disabled_device_gate(&context).await;
    if !adapter_supports_f64 {
        eprintln!("skipping native-f64 execution cases: Vulkan adapter does not expose SHADER_F64");
        return;
    }
    assert!(
        context
            .device
            .features()
            .contains(wgpu::Features::SHADER_F64),
        "SHADER_F64 must be enabled when the adapter exposes it"
    );

    let validation = context
        .device
        .push_error_scope(wgpu::ErrorFilter::Validation);
    for (shape, sign) in [
        (vec![17], NufftSign::Positive),
        (vec![17], NufftSign::Negative),
        (vec![5, 6], NufftSign::Positive),
        (vec![5, 6], NufftSign::Negative),
        (vec![4, 5, 3], NufftSign::Positive),
        (vec![4, 5, 3], NufftSign::Negative),
    ] {
        validate_type1_and_type2(&context.device, &context.queue, &shape, sign);
    }
    for dimensions in 1..=4 {
        validate_type12_adjoint(&context.device, &context.queue, dimensions);
        for sign in [NufftSign::Positive, NufftSign::Negative] {
            validate_type3_oracle(&context.device, &context.queue, dimensions, sign);
        }
        validate_type3_adjoint(&context.device, &context.queue, dimensions);
    }
    validate_fft_mode_order(&context.device, &context.queue);
    validate_active_batch_shrink(&context.device, &context.queue);
    validate_high_accuracy_1d(&context.device, &context.queue);
    if let Some(error) = validation.pop().await {
        panic!("native-f64 NUFFT validation scope captured an unexpected error: {error}");
    }
}

fn validate_fft_mode_order(device: &wgpu::Device, queue: &wgpu::Queue) {
    let shape = [12, 5];
    let point_count = 10;
    let mode_count = shape.iter().product::<usize>();
    let points = test_points(2, point_count, 0x510e_527f_ade6_82d1);
    let config = NufftConfig::new(shape, EPS)
        .with_sign(NufftSign::Negative)
        .with_mode_order(ModeOrder::Fft)
        .with_batch(BATCH)
        .with_precision(FftPrecision::F64);

    let coefficients = test_values(BATCH, mode_count, 0.37);
    let type2 = NufftPlan::type2_gpu(device, queue, config.clone()).unwrap();
    let actual = execute_type2(device, queue, &type2, &points, &coefficients);
    let reference = reference_type2_f64(&config, &points, &coefficients).unwrap();
    assert_oracle_error("type2-fft-order", 2, config.sign(), &actual, &reference);

    let strengths = test_values(BATCH, point_count, -0.49);
    let type1 = NufftPlan::type1_gpu(device, queue, config.clone()).unwrap();
    let actual = execute_type1(device, queue, &type1, &points, &strengths);
    let reference = reference_type1_f64(&config, &points, &strengths).unwrap();
    assert_oracle_error("type1-fft-order", 2, config.sign(), &actual, &reference);
}

fn validate_active_batch_shrink(device: &wgpu::Device, queue: &wgpu::Queue) {
    const CAPACITY: usize = 4;
    const SHRUNK_BATCH: usize = 2;
    let shape = [11, 7];
    let mode_count = shape.iter().product::<usize>();
    let config = NufftConfig::new(shape, EPS)
        .with_sign(NufftSign::Positive)
        .with_mode_order(ModeOrder::Centered)
        .with_batch(CAPACITY)
        .with_precision(FftPrecision::F64);
    let type1 = NufftPlan::type1_gpu(device, queue, config.clone()).unwrap();
    let type2 = NufftPlan::type2_gpu(device, queue, config.clone()).unwrap();

    let first_point_count = 15;
    let first_points = test_points(2, first_point_count, 0x1f83_d9ab_fb41_bd6b);
    let first_strengths = test_values(CAPACITY, first_point_count, 0.61);
    let first_coefficients = test_values(CAPACITY, mode_count, -0.57);
    let first_type1 = execute_type1_active(
        device,
        queue,
        &type1,
        CAPACITY,
        &first_points,
        &first_strengths,
    );
    let first_type1_reference =
        reference_type1_f64(&config, &first_points, &first_strengths).unwrap();
    assert_oracle_error_for_batch(
        "type1-active-batch-capacity",
        2,
        config.sign(),
        &first_type1,
        &first_type1_reference,
        CAPACITY,
    );
    let first_type2 = execute_type2_active(
        device,
        queue,
        &type2,
        CAPACITY,
        &first_points,
        &first_coefficients,
    );
    let first_type2_reference =
        reference_type2_f64(&config, &first_points, &first_coefficients).unwrap();
    assert_oracle_error_for_batch(
        "type2-active-batch-capacity",
        2,
        config.sign(),
        &first_type2,
        &first_type2_reference,
        CAPACITY,
    );

    let shrunk_config = config.clone().with_batch(SHRUNK_BATCH);
    let second_point_count = 7;
    let second_points = test_points(2, second_point_count, 0x5be0_cd19_137e_2179);
    let second_strengths = test_values(SHRUNK_BATCH, second_point_count, -0.73);
    let second_coefficients = test_values(SHRUNK_BATCH, mode_count, 0.79);
    let second_type1 = execute_type1_active(
        device,
        queue,
        &type1,
        SHRUNK_BATCH,
        &second_points,
        &second_strengths,
    );
    let second_type1_reference =
        reference_type1_f64(&shrunk_config, &second_points, &second_strengths).unwrap();
    assert_oracle_error_for_batch(
        "type1-active-batch-shrunk",
        2,
        shrunk_config.sign(),
        &second_type1,
        &second_type1_reference,
        SHRUNK_BATCH,
    );
    let second_type2 = execute_type2_active(
        device,
        queue,
        &type2,
        SHRUNK_BATCH,
        &second_points,
        &second_coefficients,
    );
    let second_type2_reference =
        reference_type2_f64(&shrunk_config, &second_points, &second_coefficients).unwrap();
    assert_oracle_error_for_batch(
        "type2-active-batch-shrunk",
        2,
        shrunk_config.sign(),
        &second_type2,
        &second_type2_reference,
        SHRUNK_BATCH,
    );
}

fn validate_high_accuracy_1d(device: &wgpu::Device, queue: &wgpu::Queue) {
    let shape = [17];
    let point_count = 13;
    let points = test_points(1, point_count, 0x3c6e_f372_fe94_f82b);
    let (source_bounds, target_bounds) = type3_bounds(1);
    let source = interval_points(&source_bounds, 9, 0xbb67_ae85_84ca_a73b);
    let target = interval_points(&target_bounds, 8, 0xa54f_f53a_5f1d_36f1);
    for sign in [NufftSign::Positive, NufftSign::Negative] {
        let config = NufftConfig::new(shape, HIGH_ACCURACY_EPS)
            .with_sign(sign)
            .with_mode_order(ModeOrder::Centered)
            .with_batch(BATCH)
            .with_precision(FftPrecision::F64);
        let coefficients = test_values(BATCH, shape[0], 0.71);
        let type2 = NufftPlan::type2_gpu(device, queue, config.clone()).unwrap();
        let actual = execute_type2(device, queue, &type2, &points, &coefficients);
        let reference = reference_type2_f64(&config, &points, &coefficients).unwrap();
        assert_oracle_error_with_tolerance(
            "type2-high-accuracy",
            1,
            sign,
            &actual,
            &reference,
            HIGH_ACCURACY_EPS,
            HIGH_ACCURACY_TOLERANCE,
            BATCH,
        );

        let strengths = test_values(BATCH, point_count, -0.67);
        let type1 = NufftPlan::type1_gpu(device, queue, config.clone()).unwrap();
        let actual = execute_type1(device, queue, &type1, &points, &strengths);
        let reference = reference_type1_f64(&config, &points, &strengths).unwrap();
        assert_oracle_error_with_tolerance(
            "type1-high-accuracy",
            1,
            sign,
            &actual,
            &reference,
            HIGH_ACCURACY_EPS,
            HIGH_ACCURACY_TOLERANCE,
            BATCH,
        );

        let type3_config = NufftType3Config::new(
            source_bounds.clone(),
            target_bounds.clone(),
            HIGH_ACCURACY_EPS,
        )
        .with_sign(sign)
        .with_batch(BATCH)
        .with_precision(FftPrecision::F64);
        let type3 = NufftType3Plan::new_gpu(device, queue, type3_config.clone()).unwrap();
        let strengths = test_values(BATCH, 9, 0.83);
        let actual = execute_type3(device, queue, &type3, &source, &strengths, &target);
        let reference = reference_type3_f64(&type3_config, &source, &target, &strengths).unwrap();
        assert_oracle_error_with_tolerance(
            "type3-high-accuracy",
            1,
            sign,
            &actual,
            &reference,
            HIGH_ACCURACY_EPS,
            HIGH_ACCURACY_TOLERANCE,
            BATCH,
        );
    }
}

async fn verify_feature_disabled_device_gate(context: &VulkanContext) {
    let (device, queue) = context
        .adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("wgpu_nufft.precision.featureless.device"),
            required_features: wgpu::Features::empty(),
            required_limits: context.adapter.limits(),
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: wgpu::MemoryHints::MemoryUsage,
            trace: wgpu::Trace::Off,
        })
        .await
        .expect("requesting a feature-disabled Vulkan device must succeed");
    let featureless = ManuallyDrop::new((device, queue));
    assert!(!featureless
        .0
        .features()
        .contains(wgpu::Features::SHADER_F64));

    let config = NufftConfig::new([8], EPS).with_precision(FftPrecision::F64);
    assert_precision_unsupported(
        NufftPlan::type1_gpu(&featureless.0, &featureless.1, config.clone()),
        "type-1 GPU plan",
    );
    assert_precision_unsupported(
        NufftPlan::type2_gpu(&featureless.0, &featureless.1, config),
        "type-2 GPU plan",
    );
    let type3_config = NufftType3Config::new(
        [NufftInterval::new(-1.0, 1.0)],
        [NufftInterval::new(-2.0, 2.0)],
        EPS,
    )
    .with_precision(FftPrecision::F64);
    assert_precision_unsupported(
        NufftType3Plan::new_gpu(&featureless.0, &featureless.1, type3_config),
        "type-3 GPU plan",
    );
}

fn assert_precision_unsupported<T>(result: Result<T, NufftError>, expected_stage: &'static str) {
    match result {
        Err(NufftError::PrecisionUnsupported {
            requested,
            stage,
            reason,
        }) => {
            assert_eq!(requested, FftPrecision::F64);
            assert_eq!(stage, expected_stage);
            assert!(
                reason.contains("SHADER_F64"),
                "precision gate should name SHADER_F64, got {reason:?}"
            );
        }
        Err(other) => panic!("expected PrecisionUnsupported, got {other:?}"),
        Ok(_) => panic!("native-f64 plan unexpectedly succeeded without SHADER_F64"),
    }
}

fn validate_type1_and_type2(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    shape: &[usize],
    sign: NufftSign,
) {
    let dimensions = shape.len();
    let point_count = 11;
    let mode_count = shape.iter().product::<usize>();
    let points = test_points(dimensions, point_count, 0x9e37_79b9 ^ dimensions as u64);
    let config = NufftConfig::new(shape.to_vec(), EPS)
        .with_sign(sign)
        .with_mode_order(ModeOrder::Centered)
        .with_batch(BATCH)
        .with_precision(FftPrecision::F64);

    let coefficients = test_values(BATCH, mode_count, 0.19);
    let type2 = NufftPlan::type2_gpu(device, queue, config.clone()).unwrap();
    assert_f64_type2_layout(&type2, point_count);
    let type2_actual = execute_type2(device, queue, &type2, &points, &coefficients);
    let type2_reference = reference_type2_f64(&config, &points, &coefficients).unwrap();
    assert_oracle_error("type2", dimensions, sign, &type2_actual, &type2_reference);

    let strengths = test_values(BATCH, point_count, -0.31);
    let type1 = NufftPlan::type1_gpu(device, queue, config.clone()).unwrap();
    assert_f64_type1_layout(&type1, point_count);
    let type1_actual = execute_type1(device, queue, &type1, &points, &strengths);
    let type1_reference = reference_type1_f64(&config, &points, &strengths).unwrap();
    assert_oracle_error("type1", dimensions, sign, &type1_actual, &type1_reference);
}

fn validate_type12_adjoint(device: &wgpu::Device, queue: &wgpu::Queue, dimensions: usize) {
    let shape = match dimensions {
        1 => vec![15],
        2 => vec![5, 7],
        3 => vec![3, 4, 5],
        4 => vec![3, 4, 3, 4],
        _ => unreachable!(),
    };
    let point_count = 9;
    let mode_count = shape.iter().product::<usize>();
    let points = test_points(dimensions, point_count, 0xa511_e9b3 ^ dimensions as u64);
    let type2_config = NufftConfig::new(shape.clone(), EPS)
        .with_sign(NufftSign::Positive)
        .with_batch(BATCH)
        .with_precision(FftPrecision::F64);
    let type1_config = NufftConfig::new(shape, EPS)
        .with_sign(NufftSign::Negative)
        .with_batch(BATCH)
        .with_precision(FftPrecision::F64);
    let type2 = NufftPlan::type2_gpu(device, queue, type2_config).unwrap();
    let type1 = NufftPlan::type1_gpu(device, queue, type1_config).unwrap();
    let modes = test_values(BATCH, mode_count, 0.43);
    let point_values = test_values(BATCH, point_count, -0.37);
    let av = execute_type2(device, queue, &type2, &points, &modes);
    let a_star_u = execute_type1(device, queue, &type1, &points, &point_values);
    for transform in 0..BATCH {
        let point_range = transform * point_count..(transform + 1) * point_count;
        let mode_range = transform * mode_count..(transform + 1) * mode_count;
        assert_adjoint(
            "type12",
            dimensions,
            transform,
            hermitian_inner(&point_values[point_range.clone()], &av[point_range]),
            hermitian_inner(&a_star_u[mode_range.clone()], &modes[mode_range]),
        );
    }
}

fn validate_type3_oracle(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    dimensions: usize,
    sign: NufftSign,
) {
    let (source_bounds, target_bounds) = type3_bounds(dimensions);
    let source = interval_points(&source_bounds, 8, 0xd1b5_4a32 ^ dimensions as u64);
    let target = interval_points(&target_bounds, 7, 0x94d0_49bb ^ dimensions as u64);
    let strengths = test_values(BATCH, 8, 0.27);
    let config = NufftType3Config::new(source_bounds, target_bounds, EPS)
        .with_sign(sign)
        .with_batch(BATCH)
        .with_precision(FftPrecision::F64);
    let plan = NufftType3Plan::new_gpu(device, queue, config.clone()).unwrap();
    assert_f64_type3_layout(&plan, 8, 7);
    let actual = execute_type3(device, queue, &plan, &source, &strengths, &target);
    let reference = reference_type3_f64(&config, &source, &target, &strengths).unwrap();
    assert_oracle_error("type3", dimensions, sign, &actual, &reference);
}

fn validate_type3_adjoint(device: &wgpu::Device, queue: &wgpu::Queue, dimensions: usize) {
    let (source_bounds, target_bounds) = type3_bounds(dimensions);
    let source_count = 7;
    let target_count = 6;
    let source = interval_points(
        &source_bounds,
        source_count,
        0x2545_f491 ^ dimensions as u64,
    );
    let target = interval_points(
        &target_bounds,
        target_count,
        0x6a09_e667 ^ dimensions as u64,
    );
    let forward = NufftType3Plan::new_gpu(
        device,
        queue,
        NufftType3Config::new(source_bounds.clone(), target_bounds.clone(), EPS)
            .with_sign(NufftSign::Positive)
            .with_batch(BATCH)
            .with_precision(FftPrecision::F64),
    )
    .unwrap();
    let adjoint = NufftType3Plan::new_gpu(
        device,
        queue,
        NufftType3Config::new(target_bounds, source_bounds, EPS)
            .with_sign(NufftSign::Negative)
            .with_batch(BATCH)
            .with_precision(FftPrecision::F64),
    )
    .unwrap();
    let source_values = test_values(BATCH, source_count, 0.59);
    let target_values = test_values(BATCH, target_count, -0.23);
    let av = execute_type3(device, queue, &forward, &source, &source_values, &target);
    let a_star_u = execute_type3(device, queue, &adjoint, &target, &target_values, &source);
    for transform in 0..BATCH {
        let source_range = transform * source_count..(transform + 1) * source_count;
        let target_range = transform * target_count..(transform + 1) * target_count;
        assert_adjoint(
            "type3",
            dimensions,
            transform,
            hermitian_inner(&target_values[target_range.clone()], &av[target_range]),
            hermitian_inner(
                &a_star_u[source_range.clone()],
                &source_values[source_range],
            ),
        );
    }
}

fn assert_f64_type1_layout(plan: &NufftPlan, point_count: usize) {
    assert_eq!(plan.config().precision(), FftPrecision::F64);
    assert_eq!(
        plan.required_point_buffer_size_bytes(point_count).unwrap(),
        (point_count * plan.config().dimensions() * size_of::<f64>()) as u64
    );
    assert_eq!(
        plan.required_type1_strength_buffer_size_bytes_for_batch(point_count, BATCH)
            .unwrap(),
        (point_count * BATCH * size_of::<Complex64>()) as u64
    );
    assert_eq!(
        plan.required_type1_output_buffer_size_bytes_for_batch(BATCH)
            .unwrap(),
        (plan.config().mode_count().unwrap() * BATCH * size_of::<Complex64>()) as u64
    );
}

fn assert_f64_type2_layout(plan: &NufftPlan, point_count: usize) {
    assert_eq!(plan.config().precision(), FftPrecision::F64);
    assert_eq!(
        plan.required_point_buffer_size_bytes(point_count).unwrap(),
        (point_count * plan.config().dimensions() * size_of::<f64>()) as u64
    );
    assert_eq!(
        plan.required_type2_coefficient_buffer_size_bytes_for_batch(BATCH)
            .unwrap(),
        (plan.config().mode_count().unwrap() * BATCH * size_of::<Complex64>()) as u64
    );
    assert_eq!(
        plan.required_type2_output_buffer_size_bytes_for_batch(point_count, BATCH)
            .unwrap(),
        (point_count * BATCH * size_of::<Complex64>()) as u64
    );
}

fn assert_f64_type3_layout(plan: &NufftType3Plan, source_count: usize, target_count: usize) {
    let dimensions = plan.config().dimensions();
    assert_eq!(plan.config().precision(), FftPrecision::F64);
    assert_eq!(
        plan.required_source_point_buffer_size_bytes(source_count)
            .unwrap(),
        (source_count * dimensions * size_of::<f64>()) as u64
    );
    assert_eq!(
        plan.required_target_point_buffer_size_bytes(target_count)
            .unwrap(),
        (target_count * dimensions * size_of::<f64>()) as u64
    );
    assert_eq!(
        plan.required_strength_buffer_size_bytes_for_batch(source_count, BATCH)
            .unwrap(),
        (source_count * BATCH * size_of::<Complex64>()) as u64
    );
    assert_eq!(
        plan.required_output_buffer_size_bytes_for_batch(target_count, BATCH)
            .unwrap(),
        (target_count * BATCH * size_of::<Complex64>()) as u64
    );
}

fn execute_type1(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    plan: &NufftPlan,
    points: &[f64],
    strengths: &[Complex64],
) -> Vec<Complex64> {
    execute_type1_active(
        device,
        queue,
        plan,
        plan.config().batch(),
        points,
        strengths,
    )
}

fn execute_type1_active(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    plan: &NufftPlan,
    active_batch: usize,
    points: &[f64],
    strengths: &[Complex64],
) -> Vec<Complex64> {
    let point_count = points.len() / plan.config().dimensions();
    assert_eq!(strengths.len(), active_batch * point_count);
    let points = storage_buffer_init(device, "precision.type1.points", points);
    let strengths = storage_buffer_init(device, "precision.type1.strengths", strengths);
    let output_bytes = plan
        .required_type1_output_buffer_size_bytes_for_batch(active_batch)
        .unwrap();
    execute_and_read(device, queue, output_bytes, |encoder, output| {
        plan.encode_type1_gpu_batch(
            device,
            encoder,
            point_count,
            active_batch,
            &points,
            &strengths,
            output,
        )
        .unwrap();
    })
}

fn execute_type2(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    plan: &NufftPlan,
    points: &[f64],
    coefficients: &[Complex64],
) -> Vec<Complex64> {
    execute_type2_active(
        device,
        queue,
        plan,
        plan.config().batch(),
        points,
        coefficients,
    )
}

fn execute_type2_active(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    plan: &NufftPlan,
    active_batch: usize,
    points: &[f64],
    coefficients: &[Complex64],
) -> Vec<Complex64> {
    let point_count = points.len() / plan.config().dimensions();
    assert_eq!(
        coefficients.len(),
        active_batch * plan.config().mode_count().unwrap()
    );
    let points = storage_buffer_init(device, "precision.type2.points", points);
    let coefficients = storage_buffer_init(device, "precision.type2.coefficients", coefficients);
    let output_bytes = plan
        .required_type2_output_buffer_size_bytes_for_batch(point_count, active_batch)
        .unwrap();
    execute_and_read(device, queue, output_bytes, |encoder, output| {
        plan.encode_type2_gpu_batch(
            device,
            encoder,
            point_count,
            active_batch,
            &points,
            &coefficients,
            output,
        )
        .unwrap();
    })
}

fn execute_type3(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    plan: &NufftType3Plan,
    source: &[f64],
    strengths: &[Complex64],
    target: &[f64],
) -> Vec<Complex64> {
    let dimensions = plan.config().dimensions();
    let source_count = source.len() / dimensions;
    let target_count = target.len() / dimensions;
    let source = storage_buffer_init(device, "precision.type3.source", source);
    let strengths = storage_buffer_init(device, "precision.type3.strengths", strengths);
    let target = storage_buffer_init(device, "precision.type3.target", target);
    let output_bytes = plan
        .required_output_buffer_size_bytes(target_count)
        .unwrap();
    execute_and_read(device, queue, output_bytes, |encoder, output| {
        plan.encode_gpu(
            device,
            encoder,
            source_count,
            &source,
            &strengths,
            target_count,
            &target,
            output,
        )
        .unwrap();
    })
}

fn storage_buffer_init<T: bytemuck::Pod>(
    device: &wgpu::Device,
    label: &'static str,
    values: &[T],
) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(values),
        usage: wgpu::BufferUsages::STORAGE,
    })
}

fn execute_and_read(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    output_bytes: u64,
    encode: impl FnOnce(&mut wgpu::CommandEncoder, &wgpu::Buffer),
) -> Vec<Complex64> {
    let output = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_nufft.precision.output"),
        size: output_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_nufft.precision.readback"),
        size: output_bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("wgpu_nufft.precision.encoder"),
    });
    encode(&mut encoder, &output);
    encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, output_bytes);
    let submission = queue.submit([encoder.finish()]);
    let slice = readback.slice(..);
    let (sender, receiver) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        sender.send(result).unwrap();
    });
    device
        .poll(wgpu::PollType::Wait {
            submission_index: Some(submission),
            timeout: None,
        })
        .unwrap();
    receiver.recv().unwrap().unwrap();
    let mapped = slice.get_mapped_range();
    let result = bytemuck::cast_slice::<u8, Complex64>(&mapped).to_vec();
    drop(mapped);
    readback.unmap();
    result
}

fn test_points(dimensions: usize, count: usize, mut state: u64) -> Vec<f64> {
    let mut points = Vec::with_capacity(dimensions * count);
    for point in 0..count {
        for axis in 0..dimensions {
            let value = match point {
                0 => -PI + (axis as f64 + 1.0) * 1.0e-10,
                1 => PI - (axis as f64 + 1.0) * 1.0e-10,
                2 | 3 => 0.375 - axis as f64 * 0.125,
                4 => -0.25 + axis as f64 * 1.0e-11,
                _ => {
                    let unit = next_unit(&mut state);
                    (unit * 2.0 - 1.0) * (PI - 1.0e-8)
                }
            };
            points.push(value);
        }
    }
    points
}

fn type3_bounds(dimensions: usize) -> (Vec<NufftInterval>, Vec<NufftInterval>) {
    let source = (0..dimensions)
        .map(|axis| {
            let center = 0.2 + axis as f64 * 0.17;
            let half_width = 0.75 + axis as f64 * 0.09;
            NufftInterval::new(center - half_width, center + half_width)
        })
        .collect();
    let target = (0..dimensions)
        .map(|axis| {
            let center = -0.65 + axis as f64 * 0.23;
            let half_width = 1.4 + axis as f64 * 0.21;
            NufftInterval::new(center - half_width, center + half_width)
        })
        .collect();
    (source, target)
}

fn interval_points(bounds: &[NufftInterval], count: usize, mut state: u64) -> Vec<f64> {
    let mut points = Vec::with_capacity(bounds.len() * count);
    for point in 0..count {
        for (axis, bound) in bounds.iter().enumerate() {
            let span = bound.upper() - bound.lower();
            let value = match point {
                0 => bound.lower() + span * 1.0e-10,
                1 => bound.upper() - span * 1.0e-10,
                2 | 3 => (bound.lower() + bound.upper()) * 0.5 + axis as f64 * 1.0e-12,
                _ => bound.lower() + span * (0.05 + 0.9 * next_unit(&mut state)),
            };
            points.push(value);
        }
    }
    points
}

fn next_unit(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    ((*state >> 11) as f64) * (1.0 / ((1u64 << 53) as f64))
}

fn test_values(batch: usize, count: usize, bias: f64) -> Vec<Complex64> {
    (0..batch)
        .flat_map(|transform| {
            (0..count).map(move |index| {
                let x = index as f64 + 1.0;
                let vector_bias = bias + transform as f64 * 0.117;
                Complex64::new(
                    (x * 0.31 + vector_bias).sin() * 0.7,
                    (x * 0.23 - vector_bias).cos() * 0.5,
                )
            })
        })
        .collect()
}

fn assert_oracle_error(
    kind: &str,
    dimensions: usize,
    sign: NufftSign,
    actual: &[Complex64],
    reference: &[Complex64],
) {
    assert_oracle_error_for_batch(kind, dimensions, sign, actual, reference, BATCH);
}

fn assert_oracle_error_for_batch(
    kind: &str,
    dimensions: usize,
    sign: NufftSign,
    actual: &[Complex64],
    reference: &[Complex64],
    batch: usize,
) {
    assert_oracle_error_with_tolerance(
        kind,
        dimensions,
        sign,
        actual,
        reference,
        EPS,
        ORACLE_TOLERANCE,
        batch,
    );
}

#[allow(clippy::too_many_arguments)]
fn assert_oracle_error_with_tolerance(
    kind: &str,
    dimensions: usize,
    sign: NufftSign,
    actual: &[Complex64],
    reference: &[Complex64],
    eps: f64,
    tolerance: f64,
    batch: usize,
) {
    let error = relative_l2(actual, reference);
    eprintln!(
        "NUFFT_F64_ACCURACY kind={kind} dimensions={dimensions} sign={sign:?} batch={batch} eps={eps:.1e} relative_l2={error:.9e}"
    );
    assert!(
        error <= tolerance,
        "native-f64 {kind} {dimensions}D {sign:?} relative L2 {error} exceeds {tolerance}"
    );
}

fn assert_adjoint(
    kind: &str,
    dimensions: usize,
    transform: usize,
    left: Complex64,
    right: Complex64,
) {
    let residual = Complex64::new(left.re - right.re, left.im - right.im);
    let scale = (complex_abs(left) + complex_abs(right)).max(f64::MIN_POSITIVE);
    let relative = complex_abs(residual) / scale;
    eprintln!(
        "NUFFT_F64_ADJOINT kind={kind} dimensions={dimensions} transform={transform} relative_residual={relative:.9e}"
    );
    assert!(
        relative <= ADJOINT_TOLERANCE,
        "native-f64 {kind} {dimensions}D vector {transform} adjoint residual {relative} exceeds {ADJOINT_TOLERANCE}"
    );
}

fn relative_l2(actual: &[Complex64], reference: &[Complex64]) -> f64 {
    assert_eq!(actual.len(), reference.len());
    let error_squared = actual
        .iter()
        .zip(reference)
        .map(|(actual, reference)| {
            let re = actual.re - reference.re;
            let im = actual.im - reference.im;
            re * re + im * im
        })
        .sum::<f64>();
    let reference_squared = reference
        .iter()
        .map(|value| value.re * value.re + value.im * value.im)
        .sum::<f64>();
    error_squared.sqrt() / reference_squared.sqrt().max(f64::MIN_POSITIVE)
}

fn hermitian_inner(left: &[Complex64], right: &[Complex64]) -> Complex64 {
    assert_eq!(left.len(), right.len());
    left.iter()
        .zip(right)
        .fold(Complex64::default(), |mut sum, (left, right)| {
            sum.re += left.re * right.re + left.im * right.im;
            sum.im += left.re * right.im - left.im * right.re;
            sum
        })
}

fn complex_abs(value: Complex64) -> f64 {
    value.re.hypot(value.im)
}
