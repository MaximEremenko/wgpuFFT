#![cfg(not(target_arch = "wasm32"))]

//! Opt-in portable double-float NUFFT correctness and backend coverage.

use std::f64::consts::PI;
use std::mem::ManuallyDrop;
use std::sync::mpsc;

use wgpu::util::DeviceExt;
use wgpu_nufft::{
    reference_type1_f64, reference_type2_f64, reference_type3_f64, Complex64, ComplexDoubleFloat,
    DoubleFloat, FftPrecision, ModeOrder, NufftConfig, NufftInterval, NufftPlan, NufftSign,
    NufftType3Config, NufftType3Plan,
};

const EPS: f64 = 1.0e-8;
const ORACLE_TOLERANCE: f64 = 24.0 * EPS;
const ADJOINT_TOLERANCE: f64 = 48.0 * EPS;
const HIGH_ACCURACY_EPS: f64 = 1.0e-11;
const HIGH_ACCURACY_TOLERANCE: f64 = 5.0e-11;
const BATCH: usize = 2;

#[test]
fn gpu_portable_df64_nufft_matches_direct_oracles_and_adjoint_identities() {
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
    let context = ManuallyDrop::new(context);
    let info = context.adapter.get_info();
    eprintln!(
        "gpu_nufft_df64 adapter: name={:?} backend={:?} device_type={:?} driver={:?} driver_info={:?}",
        info.name, info.backend, info.device_type, info.driver, info.driver_info
    );

    let (device, queue) = context
        .adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("wgpu_nufft.df64.featureless.device"),
            required_features: wgpu::Features::empty(),
            required_limits: context.adapter.limits(),
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
        })
        .await
        .expect("portable df64 requires no optional device features");
    let primary = ManuallyDrop::new((device, queue));
    assert!(primary.0.features().is_empty());

    let validation = primary.0.push_error_scope(wgpu::ErrorFilter::Validation);
    if info.backend == wgpu::Backend::Dx12 {
        eprintln!(
            "gpu_nufft_df64 DX12 coverage: representative batched type-1/type-2/type-3 oracle cases; exhaustive dimensional, adjoint, reuse, and high-accuracy coverage runs on Vulkan"
        );
        validate_type1_and_type2(&primary.0, &primary.1, &[17], NufftSign::Positive);
        validate_type3_oracle(&primary.0, &primary.1, 1, NufftSign::Negative);
    } else {
        for (shape, sign) in [
            (vec![17], NufftSign::Positive),
            (vec![17], NufftSign::Negative),
            (vec![5, 6], NufftSign::Positive),
            (vec![5, 6], NufftSign::Negative),
            (vec![4, 5, 3], NufftSign::Positive),
            (vec![4, 5, 3], NufftSign::Negative),
        ] {
            validate_type1_and_type2(&primary.0, &primary.1, &shape, sign);
        }
        for dimensions in 1..=3 {
            validate_type12_adjoint(&primary.0, &primary.1, dimensions);
            for sign in [NufftSign::Positive, NufftSign::Negative] {
                validate_type3_oracle(&primary.0, &primary.1, dimensions, sign);
            }
            validate_type3_adjoint(&primary.0, &primary.1, dimensions);
        }
        validate_fft_mode_order(&primary.0, &primary.1);
        validate_active_batch_shrink(&primary.0, &primary.1);
        validate_high_accuracy_1d(&primary.0, &primary.1);
    }
    if let Some(error) = validation.pop().await {
        panic!("portable-df64 NUFFT validation scope captured an unexpected error: {error}");
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
        .with_precision(FftPrecision::Df64);

    let coefficients = test_values(BATCH, mode_count, 0.37);
    let type2 = NufftPlan::type2_gpu(device, queue, config.clone()).unwrap();
    let actual = execute_type2(device, queue, &type2, BATCH, &points, &coefficients);
    let reference = reference_type2_f64(&config, &points, &coefficients).unwrap();
    assert_oracle_error(
        "type2-fft-order",
        2,
        config.sign(),
        &actual,
        &reference,
        EPS,
        BATCH,
    );

    let strengths = test_values(BATCH, point_count, -0.49);
    let type1 = NufftPlan::type1_gpu(device, queue, config.clone()).unwrap();
    let actual = execute_type1(device, queue, &type1, BATCH, &points, &strengths);
    let reference = reference_type1_f64(&config, &points, &strengths).unwrap();
    assert_oracle_error(
        "type1-fft-order",
        2,
        config.sign(),
        &actual,
        &reference,
        EPS,
        BATCH,
    );
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
        .with_batch(BATCH)
        .with_precision(FftPrecision::Df64);

    let coefficients = test_values(BATCH, mode_count, 0.19);
    let type2 = NufftPlan::type2_gpu(device, queue, config.clone()).unwrap();
    assert_df64_type2_layout(&type2, point_count);
    let actual = execute_type2(device, queue, &type2, BATCH, &points, &coefficients);
    let reference = reference_type2_f64(&config, &points, &coefficients).unwrap();
    assert_oracle_error("type2", dimensions, sign, &actual, &reference, EPS, BATCH);

    let strengths = test_values(BATCH, point_count, -0.31);
    let type1 = NufftPlan::type1_gpu(device, queue, config.clone()).unwrap();
    assert_df64_type1_layout(&type1, point_count);
    let actual = execute_type1(device, queue, &type1, BATCH, &points, &strengths);
    let reference = reference_type1_f64(&config, &points, &strengths).unwrap();
    assert_oracle_error("type1", dimensions, sign, &actual, &reference, EPS, BATCH);
}

fn validate_type12_adjoint(device: &wgpu::Device, queue: &wgpu::Queue, dimensions: usize) {
    let shape = match dimensions {
        1 => vec![15],
        2 => vec![5, 7],
        3 => vec![3, 4, 5],
        _ => unreachable!(),
    };
    let point_count = 9;
    let mode_count = shape.iter().product::<usize>();
    let points = test_points(dimensions, point_count, 0xa511_e9b3 ^ dimensions as u64);
    let type2 = NufftPlan::type2_gpu(
        device,
        queue,
        NufftConfig::new(shape.clone(), EPS)
            .with_sign(NufftSign::Positive)
            .with_batch(BATCH)
            .with_precision(FftPrecision::Df64),
    )
    .unwrap();
    let type1 = NufftPlan::type1_gpu(
        device,
        queue,
        NufftConfig::new(shape, EPS)
            .with_sign(NufftSign::Negative)
            .with_batch(BATCH)
            .with_precision(FftPrecision::Df64),
    )
    .unwrap();
    let modes = test_values(BATCH, mode_count, 0.43);
    let point_values = test_values(BATCH, point_count, -0.37);
    let av = execute_type2(device, queue, &type2, BATCH, &points, &modes);
    let a_star_u = execute_type1(device, queue, &type1, BATCH, &points, &point_values);
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
        .with_precision(FftPrecision::Df64);
    let plan = NufftType3Plan::new_gpu(device, queue, config.clone()).unwrap();
    assert_df64_type3_layout(&plan, 8, 7);
    let actual = execute_type3(device, queue, &plan, &source, &strengths, &target);
    let reference = reference_type3_f64(&config, &source, &target, &strengths).unwrap();
    assert_oracle_error("type3", dimensions, sign, &actual, &reference, EPS, BATCH);
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
            .with_precision(FftPrecision::Df64),
    )
    .unwrap();
    let adjoint = NufftType3Plan::new_gpu(
        device,
        queue,
        NufftType3Config::new(target_bounds, source_bounds, EPS)
            .with_sign(NufftSign::Negative)
            .with_batch(BATCH)
            .with_precision(FftPrecision::Df64),
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

fn validate_active_batch_shrink(device: &wgpu::Device, queue: &wgpu::Queue) {
    const CAPACITY: usize = 4;
    let shape = [11, 7];
    let mode_count = shape.iter().product::<usize>();
    let config = NufftConfig::new(shape, EPS)
        .with_batch(CAPACITY)
        .with_precision(FftPrecision::Df64);
    let type1 = NufftPlan::type1_gpu(device, queue, config.clone()).unwrap();
    let type2 = NufftPlan::type2_gpu(device, queue, config.clone()).unwrap();

    let first_points = test_points(2, 15, 0x1f83_d9ab_fb41_bd6b);
    let first_strengths = test_values(CAPACITY, 15, 0.61);
    let first_coefficients = test_values(CAPACITY, mode_count, -0.57);
    let first = execute_type1(
        device,
        queue,
        &type1,
        CAPACITY,
        &first_points,
        &first_strengths,
    );
    let reference = reference_type1_f64(&config, &first_points, &first_strengths).unwrap();
    assert_oracle_error(
        "type1-capacity",
        2,
        config.sign(),
        &first,
        &reference,
        EPS,
        CAPACITY,
    );
    let first = execute_type2(
        device,
        queue,
        &type2,
        CAPACITY,
        &first_points,
        &first_coefficients,
    );
    let reference = reference_type2_f64(&config, &first_points, &first_coefficients).unwrap();
    assert_oracle_error(
        "type2-capacity",
        2,
        config.sign(),
        &first,
        &reference,
        EPS,
        CAPACITY,
    );

    let active_batch = 2;
    let shrunk_config = config.clone().with_batch(active_batch);
    let points = test_points(2, 7, 0x5be0_cd19_137e_2179);
    let strengths = test_values(active_batch, 7, -0.73);
    let coefficients = test_values(active_batch, mode_count, 0.79);
    let actual = execute_type1(device, queue, &type1, active_batch, &points, &strengths);
    let reference = reference_type1_f64(&shrunk_config, &points, &strengths).unwrap();
    assert_oracle_error(
        "type1-shrunk",
        2,
        config.sign(),
        &actual,
        &reference,
        EPS,
        active_batch,
    );
    let actual = execute_type2(device, queue, &type2, active_batch, &points, &coefficients);
    let reference = reference_type2_f64(&shrunk_config, &points, &coefficients).unwrap();
    assert_oracle_error(
        "type2-shrunk",
        2,
        config.sign(),
        &actual,
        &reference,
        EPS,
        active_batch,
    );
}

fn validate_high_accuracy_1d(device: &wgpu::Device, queue: &wgpu::Queue) {
    let shape = [17];
    let points = test_points(1, 13, 0x3c6e_f372_fe94_f82b);
    let config = NufftConfig::new(shape, HIGH_ACCURACY_EPS)
        .with_sign(NufftSign::Negative)
        .with_batch(BATCH)
        .with_precision(FftPrecision::Df64);
    let coefficients = test_values(BATCH, shape[0], 0.71);
    let type2 = NufftPlan::type2_gpu(device, queue, config.clone()).unwrap();
    let actual = execute_type2(device, queue, &type2, BATCH, &points, &coefficients);
    let reference = reference_type2_f64(&config, &points, &coefficients).unwrap();
    assert_high_accuracy("type2", &actual, &reference);

    let strengths = test_values(BATCH, points.len(), -0.67);
    let type1 = NufftPlan::type1_gpu(device, queue, config.clone()).unwrap();
    let actual = execute_type1(device, queue, &type1, BATCH, &points, &strengths);
    let reference = reference_type1_f64(&config, &points, &strengths).unwrap();
    assert_high_accuracy("type1", &actual, &reference);

    let (source_bounds, target_bounds) = type3_bounds(1);
    let source = interval_points(&source_bounds, 9, 0xbb67_ae85_84ca_a73b);
    let target = interval_points(&target_bounds, 8, 0xa54f_f53a_5f1d_36f1);
    let config = NufftType3Config::new(source_bounds, target_bounds, HIGH_ACCURACY_EPS)
        .with_sign(NufftSign::Negative)
        .with_batch(BATCH)
        .with_precision(FftPrecision::Df64);
    let plan = NufftType3Plan::new_gpu(device, queue, config.clone()).unwrap();
    let strengths = test_values(BATCH, 9, 0.83);
    let actual = execute_type3(device, queue, &plan, &source, &strengths, &target);
    let reference = reference_type3_f64(&config, &source, &target, &strengths).unwrap();
    assert_high_accuracy("type3", &actual, &reference);
}

fn assert_df64_type1_layout(plan: &NufftPlan, point_count: usize) {
    assert_eq!(plan.config().precision(), FftPrecision::Df64);
    assert_eq!(
        plan.required_point_buffer_size_bytes(point_count).unwrap(),
        (point_count * plan.config().dimensions() * size_of::<DoubleFloat>()) as u64
    );
    assert_eq!(
        plan.required_type1_strength_buffer_size_bytes_for_batch(point_count, BATCH)
            .unwrap(),
        (point_count * BATCH * size_of::<ComplexDoubleFloat>()) as u64
    );
}

fn assert_df64_type2_layout(plan: &NufftPlan, point_count: usize) {
    assert_eq!(plan.config().precision(), FftPrecision::Df64);
    assert_eq!(
        plan.required_point_buffer_size_bytes(point_count).unwrap(),
        (point_count * plan.config().dimensions() * size_of::<DoubleFloat>()) as u64
    );
    assert_eq!(
        plan.required_type2_output_buffer_size_bytes_for_batch(point_count, BATCH)
            .unwrap(),
        (point_count * BATCH * size_of::<ComplexDoubleFloat>()) as u64
    );
}

fn assert_df64_type3_layout(plan: &NufftType3Plan, source_count: usize, target_count: usize) {
    let dimensions = plan.config().dimensions();
    assert_eq!(plan.config().precision(), FftPrecision::Df64);
    assert_eq!(
        plan.required_source_point_buffer_size_bytes(source_count)
            .unwrap(),
        (source_count * dimensions * size_of::<DoubleFloat>()) as u64
    );
    assert_eq!(
        plan.required_output_buffer_size_bytes_for_batch(target_count, BATCH)
            .unwrap(),
        (target_count * BATCH * size_of::<ComplexDoubleFloat>()) as u64
    );
}

fn execute_type1(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    plan: &NufftPlan,
    active_batch: usize,
    points: &[f64],
    strengths: &[Complex64],
) -> Vec<Complex64> {
    let point_count = points.len() / plan.config().dimensions();
    let points = pack_scalar_buffer(device, "df64.type1.points", points);
    let strengths = pack_complex_buffer(device, "df64.type1.strengths", strengths);
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
    active_batch: usize,
    points: &[f64],
    coefficients: &[Complex64],
) -> Vec<Complex64> {
    let point_count = points.len() / plan.config().dimensions();
    let points = pack_scalar_buffer(device, "df64.type2.points", points);
    let coefficients = pack_complex_buffer(device, "df64.type2.coefficients", coefficients);
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
    let source = pack_scalar_buffer(device, "df64.type3.source", source);
    let strengths = pack_complex_buffer(device, "df64.type3.strengths", strengths);
    let target = pack_scalar_buffer(device, "df64.type3.target", target);
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

fn pack_scalar_buffer(device: &wgpu::Device, label: &str, values: &[f64]) -> wgpu::Buffer {
    let packed = values
        .iter()
        .copied()
        .map(DoubleFloat::from_f64)
        .collect::<Vec<_>>();
    storage_buffer_init(device, label, &packed)
}

fn pack_complex_buffer(device: &wgpu::Device, label: &str, values: &[Complex64]) -> wgpu::Buffer {
    let packed = values
        .iter()
        .map(|value| ComplexDoubleFloat::from_f64(value.re, value.im))
        .collect::<Vec<_>>();
    storage_buffer_init(device, label, &packed)
}

fn storage_buffer_init<T: bytemuck::Pod>(
    device: &wgpu::Device,
    label: &str,
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
        label: Some("wgpu_nufft.df64.output"),
        size: output_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_nufft.df64.readback"),
        size: output_bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("wgpu_nufft.df64.encoder"),
    });
    encode(&mut encoder, &output);
    encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, output_bytes);
    let submission = queue.submit([encoder.finish()]);
    let slice = readback.slice(..);
    let (sender, receiver) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        sender.send(result).unwrap()
    });
    device
        .poll(wgpu::PollType::Wait {
            submission_index: Some(submission),
            timeout: None,
        })
        .unwrap();
    receiver.recv().unwrap().unwrap();
    let mapped = slice.get_mapped_range();
    let result = bytemuck::cast_slice::<u8, ComplexDoubleFloat>(&mapped)
        .iter()
        .map(|value| Complex64::new(value.re().to_f64(), value.im().to_f64()))
        .collect();
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
                _ => (next_unit(&mut state) * 2.0 - 1.0) * (PI - 1.0e-8),
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
    eps: f64,
    batch: usize,
) {
    let error = relative_l2(actual, reference);
    eprintln!(
        "NUFFT_DF64_ACCURACY kind={kind} dimensions={dimensions} sign={sign:?} batch={batch} eps={eps:.1e} relative_l2={error:.9e}"
    );
    assert!(
        error <= ORACLE_TOLERANCE,
        "df64 {kind} {dimensions}D {sign:?} relative L2 {error} exceeds {ORACLE_TOLERANCE}"
    );
}

fn assert_high_accuracy(kind: &str, actual: &[Complex64], reference: &[Complex64]) {
    let error = relative_l2(actual, reference);
    eprintln!(
        "NUFFT_DF64_HIGH_ACCURACY kind={kind} eps={HIGH_ACCURACY_EPS:.1e} relative_l2={error:.9e}"
    );
    assert!(
        error <= HIGH_ACCURACY_TOLERANCE,
        "df64 {kind} high-accuracy relative L2 {error} exceeds {HIGH_ACCURACY_TOLERANCE}"
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
        "NUFFT_DF64_ADJOINT kind={kind} dimensions={dimensions} transform={transform} relative_residual={relative:.9e}"
    );
    assert!(
        relative <= ADJOINT_TOLERANCE,
        "df64 {kind} {dimensions}D vector {transform} adjoint residual {relative} exceeds {ADJOINT_TOLERANCE}"
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
