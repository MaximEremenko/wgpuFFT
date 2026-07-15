//! Opt-in GPU validation of FINUFFT-style `ntransf` batching.

use std::sync::mpsc;

use wgpu::util::DeviceExt;
use wgpu_nufft::{
    reference_type1_f64, reference_type2_f64, reference_type3_f64, Complex64, ModeOrder,
    NufftConfig, NufftInterval, NufftPlan, NufftSign, NufftType3Config, NufftType3Plan,
};

const EPS: f64 = 1.0e-5;
const TYPE12_TOLERANCE: f64 = 80.0 * EPS;
const TYPE3_TOLERANCE: f64 = 120.0 * EPS;
const ADJOINT_TOLERANCE: f64 = 4.0e-5;

#[test]
fn gpu_many_vector_batching_matches_single_plans_and_oracles() {
    if std::env::var_os("WGPU_FFT_RUN_GPU_TESTS").is_none() {
        eprintln!("skipping GPU test; set WGPU_FFT_RUN_GPU_TESTS=1 to run it");
        return;
    }
    pollster::block_on(run_gpu_batch_cases());
}

async fn run_gpu_batch_cases() {
    let Some(context) = wgpu_fft::device::request_default_device().await else {
        panic!("WGPU_FFT_RUN_GPU_TESTS was set but no suitable adapter was found");
    };
    let info = context.adapter.get_info();
    eprintln!(
        "gpu_nufft_batch adapter: {} backend={:?} driver={} {}",
        info.name, info.backend, info.driver, info.driver_info
    );

    for (shape, sign) in [
        (vec![17], NufftSign::Positive),
        (vec![17], NufftSign::Negative),
        (vec![9, 10], NufftSign::Positive),
        (vec![5, 6, 7], NufftSign::Negative),
    ] {
        // Five crosses the four-vector shader tile and leaves a partial tail.
        validate_type1_and_type2_batch(&context.device, &context.queue, &shape, sign, 5);
    }
    validate_per_vector_adjoint(&context.device, &context.queue);
    validate_active_batch_and_stale_scratch(&context.device, &context.queue);
    for dimensions in 1..=3 {
        validate_type3_batch(&context.device, &context.queue, dimensions);
    }
    validate_type3_per_vector_adjoint(&context.device, &context.queue);
    validate_type3_active_batch_and_stale_scratch(&context.device, &context.queue);

    #[cfg(windows)]
    std::mem::forget(context);
}

fn validate_type1_and_type2_batch(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    shape: &[usize],
    sign: NufftSign,
    batch: usize,
) {
    let dimensions = shape.len();
    let point_count = 12;
    let points = test_points(dimensions, point_count, 0x9e37_79b9 ^ dimensions as u32);
    let mode_count = shape.iter().product::<usize>();
    let config = NufftConfig::new(shape.to_vec(), EPS)
        .with_sign(sign)
        .with_mode_order(ModeOrder::Centered)
        .with_batch(batch);
    assert_eq!(config.batch(), batch);

    let coefficients = test_values(batch, mode_count, 0.13);
    let type2_plan = NufftPlan::type2_gpu(device, queue, config.clone()).unwrap();
    let batched_type2 =
        execute_type2_batch(device, queue, &type2_plan, batch, &points, &coefficients);
    let single_config = NufftConfig::new(shape.to_vec(), EPS)
        .with_sign(sign)
        .with_mode_order(ModeOrder::Centered);
    let single_type2_plan = NufftPlan::type2_gpu(device, queue, single_config.clone()).unwrap();
    let mut loop_type2 = Vec::with_capacity(batched_type2.len());
    for transform in 0..batch {
        let start = transform * mode_count * 2;
        let end = start + mode_count * 2;
        loop_type2.extend(execute_type2_batch(
            device,
            queue,
            &single_type2_plan,
            1,
            &points,
            &coefficients[start..end],
        ));
    }
    assert_bit_identical(
        "type-2 batched versus loop-of-single",
        &batched_type2,
        &loop_type2,
    );
    let type2_reference = reference_type2_f64(
        &config,
        &f32_to_f64(&points),
        &interleaved_to_complex64(&coefficients),
    )
    .unwrap();
    let type2_error = relative_l2(&interleaved_to_complex64(&batched_type2), &type2_reference);
    eprintln!(
        "NUFFT_BATCH_ACCURACY kind=type2 dimensions={dimensions} sign={sign:?} batch={batch} relative_l2={type2_error:.9e}"
    );
    assert!(
        type2_error <= TYPE12_TOLERANCE,
        "batched type-2 {dimensions}D error {type2_error} exceeds {TYPE12_TOLERANCE}"
    );

    let strengths = test_values(batch, point_count, -0.27);
    let type1_plan = NufftPlan::type1_gpu(device, queue, config.clone()).unwrap();
    let batched_type1 = execute_type1_batch(device, queue, &type1_plan, batch, &points, &strengths);
    let single_type1_plan = NufftPlan::type1_gpu(device, queue, single_config).unwrap();
    let mut loop_type1 = Vec::with_capacity(batched_type1.len());
    for transform in 0..batch {
        let start = transform * point_count * 2;
        let end = start + point_count * 2;
        loop_type1.extend(execute_type1_batch(
            device,
            queue,
            &single_type1_plan,
            1,
            &points,
            &strengths[start..end],
        ));
    }
    assert_bit_identical(
        "type-1 batched versus loop-of-single",
        &batched_type1,
        &loop_type1,
    );
    let type1_reference = reference_type1_f64(
        &config,
        &f32_to_f64(&points),
        &interleaved_to_complex64(&strengths),
    )
    .unwrap();
    let type1_error = relative_l2(&interleaved_to_complex64(&batched_type1), &type1_reference);
    eprintln!(
        "NUFFT_BATCH_ACCURACY kind=type1 dimensions={dimensions} sign={sign:?} batch={batch} relative_l2={type1_error:.9e}"
    );
    assert!(
        type1_error <= TYPE12_TOLERANCE,
        "batched type-1 {dimensions}D error {type1_error} exceeds {TYPE12_TOLERANCE}"
    );
}

fn validate_per_vector_adjoint(device: &wgpu::Device, queue: &wgpu::Queue) {
    const BATCH: usize = 3;
    const MODE_COUNT: usize = 24;
    const POINT_COUNT: usize = 15;
    let points = test_points(1, POINT_COUNT, 0xa511_e9b3);
    let type2_config = NufftConfig::new([MODE_COUNT], 1.0e-6)
        .with_sign(NufftSign::Positive)
        .with_batch(BATCH);
    let type1_config = NufftConfig::new([MODE_COUNT], 1.0e-6)
        .with_sign(NufftSign::Negative)
        .with_batch(BATCH);
    let type2_plan = NufftPlan::type2_gpu(device, queue, type2_config).unwrap();
    let type1_plan = NufftPlan::type1_gpu(device, queue, type1_config).unwrap();
    let modes = test_values(BATCH, MODE_COUNT, 0.41);
    let point_values = test_values(BATCH, POINT_COUNT, -0.33);
    let av = interleaved_to_complex64(&execute_type2_batch(
        device,
        queue,
        &type2_plan,
        BATCH,
        &points,
        &modes,
    ));
    let a_star_u = interleaved_to_complex64(&execute_type1_batch(
        device,
        queue,
        &type1_plan,
        BATCH,
        &points,
        &point_values,
    ));
    let u = interleaved_to_complex64(&point_values);
    let v = interleaved_to_complex64(&modes);
    for transform in 0..BATCH {
        let point_range = transform * POINT_COUNT..(transform + 1) * POINT_COUNT;
        let mode_range = transform * MODE_COUNT..(transform + 1) * MODE_COUNT;
        let left = hermitian_inner(&u[point_range.clone()], &av[point_range]);
        let right = hermitian_inner(&a_star_u[mode_range.clone()], &v[mode_range]);
        let residual = complex_abs(Complex64::new(left.re - right.re, left.im - right.im));
        let scale = (complex_abs(left) + complex_abs(right)).max(f64::MIN_POSITIVE);
        let relative = residual / scale;
        eprintln!("NUFFT_BATCH_ADJOINT transform={transform} relative_residual={relative:.9e}");
        assert!(
            relative <= ADJOINT_TOLERANCE,
            "batch vector {transform} adjoint residual {relative} exceeds {ADJOINT_TOLERANCE}"
        );
    }
}

fn validate_active_batch_and_stale_scratch(device: &wgpu::Device, queue: &wgpu::Queue) {
    const CAPACITY: usize = 4;
    let config = NufftConfig::new([13, 9], EPS).with_batch(CAPACITY);
    let type1_plan = NufftPlan::type1_gpu(device, queue, config.clone()).unwrap();
    let type2_plan = NufftPlan::type2_gpu(device, queue, config.clone()).unwrap();
    let mode_count = config.mode_count().unwrap();

    let large_points = test_points(2, 23, 0xc2b2_ae35);
    let large_strengths = test_values(CAPACITY, 23, 0.73);
    let large_coefficients = test_values(CAPACITY, mode_count, -0.61);
    let _ = execute_type1_batch(
        device,
        queue,
        &type1_plan,
        CAPACITY,
        &large_points,
        &large_strengths,
    );
    let _ = execute_type2_batch(
        device,
        queue,
        &type2_plan,
        CAPACITY,
        &large_points,
        &large_coefficients,
    );

    const ACTIVE: usize = 2;
    let small_points = test_points(2, 7, 0x27d4_eb2f);
    let small_strengths = test_values(ACTIVE, 7, -0.19);
    let small_coefficients = test_values(ACTIVE, mode_count, 0.29);
    let actual_type1 = execute_type1_batch(
        device,
        queue,
        &type1_plan,
        ACTIVE,
        &small_points,
        &small_strengths,
    );
    let actual_type2 = execute_type2_batch(
        device,
        queue,
        &type2_plan,
        ACTIVE,
        &small_points,
        &small_coefficients,
    );

    let active_config = NufftConfig::new([13, 9], EPS).with_batch(ACTIVE);
    let type1_reference = reference_type1_f64(
        &active_config,
        &f32_to_f64(&small_points),
        &interleaved_to_complex64(&small_strengths),
    )
    .unwrap();
    let type2_reference = reference_type2_f64(
        &active_config,
        &f32_to_f64(&small_points),
        &interleaved_to_complex64(&small_coefficients),
    )
    .unwrap();
    let type1_error = relative_l2(&interleaved_to_complex64(&actual_type1), &type1_reference);
    let type2_error = relative_l2(&interleaved_to_complex64(&actual_type2), &type2_reference);
    assert!(type1_error <= TYPE12_TOLERANCE);
    assert!(type2_error <= TYPE12_TOLERANCE);

    let fresh_type1 = NufftPlan::type1_gpu(device, queue, active_config.clone()).unwrap();
    let fresh_type2 = NufftPlan::type2_gpu(device, queue, active_config).unwrap();
    assert_bit_identical(
        "type-1 capacity-four grow/shrink scratch reuse",
        &actual_type1,
        &execute_type1_batch(
            device,
            queue,
            &fresh_type1,
            ACTIVE,
            &small_points,
            &small_strengths,
        ),
    );
    assert_bit_identical(
        "type-2 capacity-four grow/shrink scratch reuse",
        &actual_type2,
        &execute_type2_batch(
            device,
            queue,
            &fresh_type2,
            ACTIVE,
            &small_points,
            &small_coefficients,
        ),
    );
}

fn validate_type3_batch(device: &wgpu::Device, queue: &wgpu::Queue, dimensions: usize) {
    // Five crosses the four-vector shader tile and leaves a partial tail.
    const BATCH: usize = 5;
    const SOURCE_COUNT: usize = 8;
    const TARGET_COUNT: usize = 7;
    let source_bounds = (0..dimensions)
        .map(|axis| {
            let center = 0.15 * axis as f64 - 0.2;
            NufftInterval::new(center - 0.9, center + 0.9)
        })
        .collect::<Vec<_>>();
    let target_bounds = (0..dimensions)
        .map(|axis| {
            let center = -0.3 * axis as f64 + 0.45;
            NufftInterval::new(center - 1.7, center + 1.7)
        })
        .collect::<Vec<_>>();
    let config = NufftType3Config::new(source_bounds.clone(), target_bounds.clone(), EPS)
        .with_sign(NufftSign::Positive)
        .with_batch(BATCH);
    let plan = NufftType3Plan::new_gpu(device, queue, config.clone()).unwrap();
    let source = interval_points(&source_bounds, SOURCE_COUNT, 0x1656_67b1);
    let target = interval_points(&target_bounds, TARGET_COUNT, 0xd3a2_646c);
    let strengths = test_values(BATCH, SOURCE_COUNT, dimensions as f32 * 0.17);
    let batched = execute_type3_batch(device, queue, &plan, BATCH, &source, &strengths, &target);

    let single_config =
        NufftType3Config::new(source_bounds, target_bounds, EPS).with_sign(NufftSign::Positive);
    let single_plan = NufftType3Plan::new_gpu(device, queue, single_config).unwrap();
    let mut looped = Vec::with_capacity(batched.len());
    for transform in 0..BATCH {
        let start = transform * SOURCE_COUNT * 2;
        let end = start + SOURCE_COUNT * 2;
        looped.extend(execute_type3_batch(
            device,
            queue,
            &single_plan,
            1,
            &source,
            &strengths[start..end],
            &target,
        ));
    }
    assert_bit_identical("type-3 batched versus loop-of-single", &batched, &looped);
    let reference = reference_type3_f64(
        &config,
        &f32_to_f64(&source),
        &f32_to_f64(&target),
        &interleaved_to_complex64(&strengths),
    )
    .unwrap();
    let error = relative_l2(&interleaved_to_complex64(&batched), &reference);
    eprintln!(
        "NUFFT_BATCH_ACCURACY kind=type3 dimensions={dimensions} batch={BATCH} relative_l2={error:.9e}"
    );
    assert!(
        error <= TYPE3_TOLERANCE,
        "batched type-3 {dimensions}D error {error} exceeds {TYPE3_TOLERANCE}"
    );
}

fn validate_type3_per_vector_adjoint(device: &wgpu::Device, queue: &wgpu::Queue) {
    const BATCH: usize = 3;
    const SOURCE_COUNT: usize = 7;
    const TARGET_COUNT: usize = 6;

    for dimensions in 1..=3 {
        let source_bounds = (0..dimensions)
            .map(|axis| {
                let center = 0.11 * axis as f64 - 0.18;
                NufftInterval::new(center - 0.85, center + 0.85)
            })
            .collect::<Vec<_>>();
        let target_bounds = (0..dimensions)
            .map(|axis| {
                let center = -0.23 * axis as f64 + 0.31;
                NufftInterval::new(center - 1.55, center + 1.55)
            })
            .collect::<Vec<_>>();
        let source = interval_points(&source_bounds, SOURCE_COUNT, 0x85eb_ca6b);
        let target = interval_points(&target_bounds, TARGET_COUNT, 0xc2b2_ae35);

        let forward = NufftType3Plan::new_gpu(
            device,
            queue,
            NufftType3Config::new(source_bounds.clone(), target_bounds.clone(), 1.0e-6)
                .with_sign(NufftSign::Positive)
                .with_batch(BATCH),
        )
        .unwrap();
        let adjoint = NufftType3Plan::new_gpu(
            device,
            queue,
            NufftType3Config::new(target_bounds, source_bounds, 1.0e-6)
                .with_sign(NufftSign::Negative)
                .with_batch(BATCH),
        )
        .unwrap();
        let v = test_values(BATCH, SOURCE_COUNT, 0.37);
        let u = test_values(BATCH, TARGET_COUNT, -0.43);
        let av = interleaved_to_complex64(&execute_type3_batch(
            device, queue, &forward, BATCH, &source, &v, &target,
        ));
        let a_star_u = interleaved_to_complex64(&execute_type3_batch(
            device, queue, &adjoint, BATCH, &target, &u, &source,
        ));
        let u = interleaved_to_complex64(&u);
        let v = interleaved_to_complex64(&v);

        for transform in 0..BATCH {
            let target_range = transform * TARGET_COUNT..(transform + 1) * TARGET_COUNT;
            let source_range = transform * SOURCE_COUNT..(transform + 1) * SOURCE_COUNT;
            let left = hermitian_inner(&u[target_range.clone()], &av[target_range]);
            let right = hermitian_inner(&a_star_u[source_range.clone()], &v[source_range]);
            let residual = complex_abs(Complex64::new(left.re - right.re, left.im - right.im));
            let scale = (complex_abs(left) + complex_abs(right)).max(f64::MIN_POSITIVE);
            let relative = residual / scale;
            eprintln!(
                "NUFFT_BATCH_TYPE3_ADJOINT dimensions={dimensions} transform={transform} relative_residual={relative:.9e}"
            );
            assert!(
                relative <= ADJOINT_TOLERANCE,
                "type-3 {dimensions}D batch vector {transform} adjoint residual {relative} exceeds {ADJOINT_TOLERANCE}"
            );
        }
    }
}

fn validate_type3_active_batch_and_stale_scratch(device: &wgpu::Device, queue: &wgpu::Queue) {
    const CAPACITY: usize = 4;
    const ACTIVE: usize = 2;
    let source_bounds = vec![NufftInterval::new(-1.1, 0.9), NufftInterval::new(-0.7, 1.3)];
    let target_bounds = vec![NufftInterval::new(-2.0, 1.6), NufftInterval::new(-1.4, 2.2)];
    let capacity_config = NufftType3Config::new(source_bounds.clone(), target_bounds.clone(), EPS)
        .with_sign(NufftSign::Negative)
        .with_batch(CAPACITY);
    let plan = NufftType3Plan::new_gpu(device, queue, capacity_config).unwrap();

    let large_source = interval_points(&source_bounds, 17, 0x27d4_eb2f);
    let large_target = interval_points(&target_bounds, 15, 0x1656_67b1);
    let large_strengths = test_values(CAPACITY, 17, 0.61);
    let _ = execute_type3_batch(
        device,
        queue,
        &plan,
        CAPACITY,
        &large_source,
        &large_strengths,
        &large_target,
    );

    let small_source = interval_points(&source_bounds, 6, 0xd3a2_646c);
    let small_target = interval_points(&target_bounds, 5, 0x9e37_79b9);
    let small_strengths = test_values(ACTIVE, 6, -0.29);
    let actual = execute_type3_batch(
        device,
        queue,
        &plan,
        ACTIVE,
        &small_source,
        &small_strengths,
        &small_target,
    );

    let active_config = NufftType3Config::new(source_bounds, target_bounds, EPS)
        .with_sign(NufftSign::Negative)
        .with_batch(ACTIVE);
    let fresh = NufftType3Plan::new_gpu(device, queue, active_config.clone()).unwrap();
    let expected = execute_type3_batch(
        device,
        queue,
        &fresh,
        ACTIVE,
        &small_source,
        &small_strengths,
        &small_target,
    );
    assert_bit_identical(
        "type-3 capacity-four grow/shrink scratch reuse",
        &actual,
        &expected,
    );

    let oracle = reference_type3_f64(
        &active_config,
        &f32_to_f64(&small_source),
        &f32_to_f64(&small_target),
        &interleaved_to_complex64(&small_strengths),
    )
    .unwrap();
    let error = relative_l2(&interleaved_to_complex64(&actual), &oracle);
    assert!(
        error <= TYPE3_TOLERANCE,
        "type-3 grow/shrink error {error} exceeds {TYPE3_TOLERANCE}"
    );
}

fn execute_type1_batch(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    plan: &NufftPlan,
    active_batch: usize,
    points: &[f32],
    strengths: &[f32],
) -> Vec<f32> {
    let dimensions = plan.config().dimensions();
    let point_count = points.len() / dimensions;
    assert_eq!(strengths.len(), active_batch * point_count * 2);
    let point_buffer = storage_buffer_init(device, "batch.type1.points", points);
    let strength_buffer = storage_buffer_init(device, "batch.type1.strengths", strengths);
    let output_bytes = plan
        .required_type1_output_buffer_size_bytes_for_batch(active_batch)
        .unwrap();
    execute_and_read(device, queue, output_bytes, |encoder, output| {
        plan.encode_type1_gpu_batch(
            device,
            encoder,
            point_count,
            active_batch,
            &point_buffer,
            &strength_buffer,
            output,
        )
        .unwrap();
    })
}

fn execute_type2_batch(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    plan: &NufftPlan,
    active_batch: usize,
    points: &[f32],
    coefficients: &[f32],
) -> Vec<f32> {
    let dimensions = plan.config().dimensions();
    let point_count = points.len() / dimensions;
    assert_eq!(
        coefficients.len(),
        active_batch * plan.config().mode_count().unwrap() * 2
    );
    let point_buffer = storage_buffer_init(device, "batch.type2.points", points);
    let coefficient_buffer = storage_buffer_init(device, "batch.type2.coefficients", coefficients);
    let output_bytes = plan
        .required_type2_output_buffer_size_bytes_for_batch(point_count, active_batch)
        .unwrap();
    execute_and_read(device, queue, output_bytes, |encoder, output| {
        plan.encode_type2_gpu_batch(
            device,
            encoder,
            point_count,
            active_batch,
            &point_buffer,
            &coefficient_buffer,
            output,
        )
        .unwrap();
    })
}

fn execute_type3_batch(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    plan: &NufftType3Plan,
    active_batch: usize,
    source: &[f32],
    strengths: &[f32],
    target: &[f32],
) -> Vec<f32> {
    let dimensions = plan.config().dimensions();
    let source_count = source.len() / dimensions;
    let target_count = target.len() / dimensions;
    assert_eq!(strengths.len(), active_batch * source_count * 2);
    let source_buffer = storage_buffer_init(device, "batch.type3.source", source);
    let strength_buffer = storage_buffer_init(device, "batch.type3.strengths", strengths);
    let target_buffer = storage_buffer_init(device, "batch.type3.target", target);
    let output_bytes = plan
        .required_output_buffer_size_bytes_for_batch(target_count, active_batch)
        .unwrap();
    execute_and_read(device, queue, output_bytes, |encoder, output| {
        plan.encode_gpu_batch(
            device,
            encoder,
            active_batch,
            source_count,
            &source_buffer,
            &strength_buffer,
            target_count,
            &target_buffer,
            output,
        )
        .unwrap();
    })
}

fn storage_buffer_init(device: &wgpu::Device, label: &'static str, values: &[f32]) -> wgpu::Buffer {
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
) -> Vec<f32> {
    let output = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_nufft.batch.output"),
        size: output_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_nufft.batch.readback"),
        size: output_bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("wgpu_nufft.batch.encoder"),
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
    let values = bytemuck::cast_slice::<u8, f32>(&mapped).to_vec();
    drop(mapped);
    readback.unmap();
    values
}

fn test_points(dimensions: usize, count: usize, mut state: u32) -> Vec<f32> {
    let mut points = Vec::with_capacity(dimensions * count);
    for point in 0..count {
        for axis in 0..dimensions {
            let value = match point {
                0 => -std::f32::consts::PI + (axis as f32 + 1.0) * 1.0e-5,
                1 | 2 => 0.375 - axis as f32 * 0.125,
                3 => 0.25 + axis as f32 * 1.0e-5,
                4 => 0.25 + (axis as f32 + 1.0) * 2.0e-5,
                _ => {
                    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    let unit = (state >> 8) as f32 / ((1u32 << 24) - 1) as f32;
                    (unit * 2.0 - 1.0) * (std::f32::consts::PI - 1.0e-4)
                }
            };
            points.push(value);
        }
    }
    points
}

fn interval_points(bounds: &[NufftInterval], count: usize, mut state: u32) -> Vec<f32> {
    let mut points = Vec::with_capacity(bounds.len() * count);
    for point in 0..count {
        for &bound in bounds {
            let lower_f64 = bound.lower();
            let upper_f64 = bound.upper();
            let span_f64 = upper_f64 - lower_f64;
            let lower = lower_f64 as f32;
            let upper = upper_f64 as f32;
            let value = match point {
                // Stay strictly inside the f64 contract: casting a decimal
                // endpoint such as -1.1 to f32 can round just outside it.
                0 => (lower_f64 + span_f64 * 1.0e-4) as f32,
                1 => (upper_f64 - span_f64 * 1.0e-4) as f32,
                2 | 3 => (lower + upper) * 0.5,
                _ => {
                    state = state.wrapping_mul(22_695_477).wrapping_add(1);
                    let unit = (state >> 8) as f32 / ((1u32 << 24) - 1) as f32;
                    lower + (upper - lower) * (0.1 + 0.8 * unit)
                }
            };
            points.push(value);
        }
    }
    points
}

fn test_values(batch: usize, count: usize, bias: f32) -> Vec<f32> {
    (0..batch)
        .flat_map(|transform| {
            (0..count).flat_map(move |index| {
                let x = index as f32 + 1.0;
                let vector_bias = bias + transform as f32 * 0.117;
                [
                    (x * 0.31 + vector_bias).sin() * 0.7,
                    (x * 0.23 - vector_bias).cos() * 0.5,
                ]
            })
        })
        .collect()
}

fn assert_bit_identical(label: &str, actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(
            actual.to_bits(),
            expected.to_bits(),
            "{label}: float {index} differs: {actual:?} versus {expected:?}"
        );
    }
}

fn f32_to_f64(values: &[f32]) -> Vec<f64> {
    values.iter().map(|&value| f64::from(value)).collect()
}

fn interleaved_to_complex64(values: &[f32]) -> Vec<Complex64> {
    values
        .chunks_exact(2)
        .map(|pair| Complex64::new(f64::from(pair[0]), f64::from(pair[1])))
        .collect()
}

fn relative_l2(actual: &[Complex64], reference: &[Complex64]) -> f64 {
    let numerator = actual
        .iter()
        .zip(reference)
        .map(|(actual, reference)| {
            let re = actual.re - reference.re;
            let im = actual.im - reference.im;
            re * re + im * im
        })
        .sum::<f64>()
        .sqrt();
    let denominator = reference
        .iter()
        .map(|value| value.re * value.re + value.im * value.im)
        .sum::<f64>()
        .sqrt()
        .max(f64::MIN_POSITIVE);
    numerator / denominator
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
