//! Opt-in 1D GPU validation against the direct f64 NDFT oracle.

use std::f64::consts::PI;
use std::sync::mpsc;

use wgpu::util::DeviceExt;
use wgpu_nufft::{
    reference_type1_f64, reference_type2_f64, Complex64, ModeOrder, NufftConfig, NufftError,
    NufftPlan, NufftSign,
};

const FULL_VECTOR_TOLERANCE_FACTOR: f64 = 4.0;
// Keep every small, deliberately hostile subset below the stricter factor
// of eight while the complete vector retains the Phase-A factor of four.
const ADVERSARIAL_TOLERANCE_FACTOR: f64 = 8.0;
const ADJOINT_TOLERANCE: f64 = 2.0e-5;

#[test]
fn gpu_type1_and_type2_match_oracle_and_are_adjoint() {
    if std::env::var_os("WGPU_FFT_RUN_GPU_TESTS").is_none() {
        eprintln!("skipping GPU test; set WGPU_FFT_RUN_GPU_TESTS=1 to run it");
        return;
    }
    pollster::block_on(run_gpu_nufft_cases());
}

async fn run_gpu_nufft_cases() {
    let Some(context) = wgpu_fft::device::request_default_device().await else {
        panic!("WGPU_FFT_RUN_GPU_TESTS was set but no suitable adapter was found");
    };
    let info = context.adapter.get_info();
    eprintln!(
        "gpu_nufft adapter: {} backend={:?} driver={} {}",
        info.name, info.backend, info.driver, info.driver_info
    );
    let (points, classes) = adversarial_points();
    for eps in [1.0e-2, 1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6] {
        for sign in [NufftSign::Positive, NufftSign::Negative] {
            for order in [ModeOrder::Centered, ModeOrder::Fft] {
                let config = NufftConfig::new([32], eps)
                    .with_sign(sign)
                    .with_mode_order(order);
                let coefficients = test_coefficients(32, sign, order);
                let validation_scope = context
                    .device
                    .push_error_scope(wgpu::ErrorFilter::Validation);
                let plan =
                    NufftPlan::type2_gpu(&context.device, &context.queue, config.clone()).unwrap();
                assert!(plan.is_gpu_ready());
                let actual = execute_type2(
                    &context.device,
                    &context.queue,
                    &plan,
                    &points,
                    &coefficients,
                );
                if let Some(error) = validation_scope.pop().await {
                    panic!("GPU validation scope captured an unexpected error: {error}");
                }
                let point_f64 = points.iter().map(|&x| f64::from(x)).collect::<Vec<_>>();
                let coefficient_f64 = coefficients
                    .chunks_exact(2)
                    .map(|pair| Complex64::new(f64::from(pair[0]), f64::from(pair[1])))
                    .collect::<Vec<_>>();
                let reference = reference_type2_f64(&config, &point_f64, &coefficient_f64).unwrap();
                let actual_complex = interleaved_to_complex64(&actual);
                let full_error = relative_l2(&actual_complex, &reference);
                eprintln!(
                    "NUFFT_ACCURACY eps={eps:.0e} sign={sign:?} order={order:?} class=all relative_l2={full_error:.9e}"
                );
                assert!(
                    full_error <= FULL_VECTOR_TOLERANCE_FACTOR * eps,
                    "eps={eps} sign={sign:?} order={order:?}: full relative l2 {full_error} exceeds {}",
                    FULL_VECTOR_TOLERANCE_FACTOR * eps
                );
                for class in &classes {
                    let error = relative_l2(
                        &actual_complex[class.start..class.end],
                        &reference[class.start..class.end],
                    );
                    eprintln!(
                        "NUFFT_ACCURACY eps={eps:.0e} sign={sign:?} order={order:?} class={} relative_l2={error:.9e}",
                        class.label
                    );
                    assert!(
                        error <= ADVERSARIAL_TOLERANCE_FACTOR * eps,
                        "eps={eps} sign={sign:?} order={order:?} class={}: adversarial relative l2 {error} exceeds {}",
                        class.label,
                        ADVERSARIAL_TOLERANCE_FACTOR * eps
                    );
                }
                let duplicate = classes
                    .iter()
                    .find(|class| class.label == "duplicates")
                    .unwrap();
                let first = &actual[duplicate.start * 2..duplicate.start * 2 + 2];
                for index in duplicate.start + 1..duplicate.start + 4 {
                    assert_eq!(
                        first,
                        &actual[index * 2..index * 2 + 2],
                        "identical points must produce bit-identical gathers"
                    );
                }

                if eps == 1.0e-6 && sign == NufftSign::Positive && order == ModeOrder::Centered {
                    let zero_coefficients = vec![0.0f32; 64];
                    let reused = execute_type2(
                        &context.device,
                        &context.queue,
                        &plan,
                        &points[..5],
                        &zero_coefficients,
                    );
                    assert!(
                        reused.iter().all(|&value| value == 0.0),
                        "a reused plan must clear every stale fine-grid bin"
                    );
                }
            }
        }
    }

    let odd_config = NufftConfig::new([17], 1.0e-6).with_sign(NufftSign::Negative);
    let odd_coefficients = test_coefficients(17, NufftSign::Negative, ModeOrder::Centered);
    let odd_plan =
        NufftPlan::type2_gpu(&context.device, &context.queue, odd_config.clone()).unwrap();
    assert_eq!(odd_plan.fine_grid_shape(), [36]);
    let odd_points = [-2.3f32, -0.5, 0.25, 1.75];
    let odd_actual = execute_type2(
        &context.device,
        &context.queue,
        &odd_plan,
        &odd_points,
        &odd_coefficients,
    );
    let odd_reference = reference_type2_f64(
        &odd_config,
        &odd_points.iter().map(|&x| f64::from(x)).collect::<Vec<_>>(),
        &odd_coefficients
            .chunks_exact(2)
            .map(|pair| Complex64::new(f64::from(pair[0]), f64::from(pair[1])))
            .collect::<Vec<_>>(),
    )
    .unwrap();
    assert!(relative_l2(&interleaved_to_complex64(&odd_actual), &odd_reference) <= 4.0e-6);

    run_gpu_type1_cases(&context.device, &context.queue).await;
    validate_type1_scratch_grow_then_shrink(&context.device, &context.queue);
    validate_gpu_adjoint_consistency(&context.device, &context.queue);
    validate_structured_gpu_errors(&context.device, &context.queue);

    #[cfg(windows)]
    std::mem::forget(context);
}

async fn run_gpu_type1_cases(device: &wgpu::Device, queue: &wgpu::Queue) {
    let (points, classes) = adversarial_points();
    for eps in [1.0e-2, 1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6] {
        for sign in [NufftSign::Positive, NufftSign::Negative] {
            for order in [ModeOrder::Centered, ModeOrder::Fft] {
                let config = NufftConfig::new([32], eps)
                    .with_sign(sign)
                    .with_mode_order(order);
                let strengths = test_coefficients(points.len(), sign, order);
                let validation_scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
                let plan = NufftPlan::type1_gpu(device, queue, config.clone()).unwrap();
                assert!(plan.is_gpu_ready());

                assert_type1_matches_oracle(
                    device,
                    queue,
                    &plan,
                    &points,
                    &strengths,
                    "all",
                    FULL_VECTOR_TOLERANCE_FACTOR * eps,
                );
                for class in &classes {
                    assert_type1_matches_oracle(
                        device,
                        queue,
                        &plan,
                        &points[class.start..class.end],
                        &strengths[class.start * 2..class.end * 2],
                        class.label,
                        ADVERSARIAL_TOLERANCE_FACTOR * eps,
                    );
                }

                if eps == 1.0e-6 && sign == NufftSign::Positive && order == ModeOrder::Centered {
                    let first = execute_type1(device, queue, &plan, &points, &strengths);
                    let second = execute_type1(device, queue, &plan, &points, &strengths);
                    assert_eq!(
                        first
                            .iter()
                            .map(|value| value.to_bits())
                            .collect::<Vec<_>>(),
                        second
                            .iter()
                            .map(|value| value.to_bits())
                            .collect::<Vec<_>>(),
                        "stable per-bin ordering must make repeated type-1 executions bit-identical"
                    );
                    let zero = execute_type1(device, queue, &plan, &[], &[]);
                    assert!(
                        zero.iter().all(|&value| value == 0.0),
                        "a zero-point type-1 execution must overwrite stale output with zeros"
                    );

                    let random = classes
                        .iter()
                        .find(|class| class.label == "random")
                        .unwrap();
                    assert_type1_matches_oracle(
                        device,
                        queue,
                        &plan,
                        &points[random.start..random.end],
                        &strengths[random.start * 2..random.end * 2],
                        "random-after-zero",
                        ADVERSARIAL_TOLERANCE_FACTOR * eps,
                    );
                }

                if let Some(error) = validation_scope.pop().await {
                    panic!("type-1 GPU validation scope captured an unexpected error: {error}");
                }
            }
        }
    }

    let odd_config = NufftConfig::new([17], 1.0e-6)
        .with_sign(NufftSign::Negative)
        .with_mode_order(ModeOrder::Fft);
    let odd_points = [-2.3f32, -0.5, 0.25, 1.75];
    let odd_strengths = test_coefficients(odd_points.len(), NufftSign::Negative, ModeOrder::Fft);
    let odd_plan = NufftPlan::type1_gpu(device, queue, odd_config.clone()).unwrap();
    assert_eq!(odd_plan.fine_grid_shape(), [36]);
    assert_type1_matches_oracle(
        device,
        queue,
        &odd_plan,
        &odd_points,
        &odd_strengths,
        "odd-modes",
        4.0e-6,
    );
}

fn validate_type1_scratch_grow_then_shrink(device: &wgpu::Device, queue: &wgpu::Queue) {
    const EPS: f64 = 1.0e-6;
    const LARGE_POINT_COUNT: usize = 257;

    let config = NufftConfig::new([64], EPS)
        .with_sign(NufftSign::Negative)
        .with_mode_order(ModeOrder::Fft);
    let plan = NufftPlan::type1_gpu(device, queue, config.clone()).unwrap();

    // The first execution grows the cached point-index scratch and leaves a
    // deliberately large, broadly distributed signal in every reusable stage.
    // The clustered tail also poisons a high-occupancy bin.
    let mut large_points = Vec::with_capacity(LARGE_POINT_COUNT);
    let mut large_strengths = Vec::with_capacity(LARGE_POINT_COUNT * 2);
    let pi = PI as f32;
    for index in 0..LARGE_POINT_COUNT {
        let point = if index >= 224 {
            1.375 + (index - 224) as f32 * 1.0e-5
        } else {
            let permuted = (index * 97) % LARGE_POINT_COUNT;
            -pi + 2.0 * pi * (permuted as f32 + 0.25) / LARGE_POINT_COUNT as f32
        };
        let phase = index as f32 * 0.173;
        let magnitude = 32.0 + (index % 11) as f32;
        large_points.push(point);
        large_strengths.extend([magnitude * phase.cos(), -magnitude * phase.sin()]);
    }

    // The second execution is much smaller, occupies a different set of bins,
    // includes a duplicate and wrap-boundary points, and has low amplitudes so
    // any leakage from the first execution is conspicuous.
    let three_pi = (3.0 * PI) as f32;
    let three_pi_inside = f32::from_bits(three_pi.to_bits() - 1);
    let small_points = [
        -three_pi_inside,
        -2.75,
        -0.8125,
        -0.8125,
        0.125,
        2.9375,
        three_pi_inside,
    ];
    let small_strengths = [
        0.25, -0.5, -1.25, 0.75, 2.0, -1.0, -0.125, 0.375, 0.625, 1.5, -0.75, -0.25, 1.125, -1.75,
    ];
    assert!(large_points.len() > small_points.len());

    let large_point_f64 = large_points
        .iter()
        .map(|&point| f64::from(point))
        .collect::<Vec<_>>();
    let small_point_f64 = small_points
        .iter()
        .map(|&point| f64::from(point))
        .collect::<Vec<_>>();
    let large_reference = reference_type1_f64(
        &config,
        &large_point_f64,
        &interleaved_to_complex64(&large_strengths),
    )
    .unwrap();
    let small_reference = reference_type1_f64(
        &config,
        &small_point_f64,
        &interleaved_to_complex64(&small_strengths),
    )
    .unwrap();

    // `execute_type1` waits for the exact returned submission before the next
    // execution, making the grow-then-shrink ordering explicit.
    let large_actual = interleaved_to_complex64(&execute_type1(
        device,
        queue,
        &plan,
        &large_points,
        &large_strengths,
    ));
    let small_actual = interleaved_to_complex64(&execute_type1(
        device,
        queue,
        &plan,
        &small_points,
        &small_strengths,
    ));

    for (label, actual, reference) in [
        ("scratch-grow-large", &large_actual, &large_reference),
        ("scratch-shrink-stale", &small_actual, &small_reference),
    ] {
        let error = relative_l2(actual, reference);
        eprintln!("NUFFT_ACCURACY kind=type1 class={label} relative_l2={error:.9e}");
        assert!(
            error <= ADVERSARIAL_TOLERANCE_FACTOR * EPS,
            "type-1 class={label}: relative l2 {error} exceeds {}",
            ADVERSARIAL_TOLERANCE_FACTOR * EPS
        );
    }
}

fn assert_type1_matches_oracle(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    plan: &NufftPlan,
    points: &[f32],
    strengths: &[f32],
    class: &str,
    tolerance: f64,
) {
    let config = plan.config();
    let actual = interleaved_to_complex64(&execute_type1(device, queue, plan, points, strengths));
    let point_f64 = points.iter().map(|&x| f64::from(x)).collect::<Vec<_>>();
    let strength_f64 = interleaved_to_complex64(strengths);
    let reference = reference_type1_f64(config, &point_f64, &strength_f64).unwrap();
    let error = relative_l2(&actual, &reference);
    eprintln!(
        "NUFFT_ACCURACY kind=type1 eps={:.0e} sign={:?} order={:?} class={class} relative_l2={error:.9e}",
        config.eps(),
        config.sign(),
        config.mode_order(),
    );
    assert!(
        error <= tolerance,
        "type1 eps={} sign={:?} order={:?} class={class}: relative l2 {error} exceeds {tolerance}",
        config.eps(),
        config.sign(),
        config.mode_order(),
    );
}

fn validate_gpu_adjoint_consistency(device: &wgpu::Device, queue: &wgpu::Queue) {
    let (points, _) = adversarial_points();
    for sign in [NufftSign::Positive, NufftSign::Negative] {
        for order in [ModeOrder::Centered, ModeOrder::Fft] {
            let type2_config = NufftConfig::new([32], 1.0e-6)
                .with_sign(sign)
                .with_mode_order(order);
            let type1_config = NufftConfig::new([32], 1.0e-6)
                .with_sign(opposite_sign(sign))
                .with_mode_order(order);
            let type2_plan = NufftPlan::type2_gpu(device, queue, type2_config.clone()).unwrap();
            let type1_plan = NufftPlan::type1_gpu(device, queue, type1_config.clone()).unwrap();
            let modes = test_coefficients(32, sign, order);
            let point_values = test_coefficients(points.len(), opposite_sign(sign), order);

            let av = interleaved_to_complex64(&execute_type2(
                device,
                queue,
                &type2_plan,
                &points,
                &modes,
            ));
            let a_star_u = interleaved_to_complex64(&execute_type1(
                device,
                queue,
                &type1_plan,
                &points,
                &point_values,
            ));
            let u = interleaved_to_complex64(&point_values);
            let v = interleaved_to_complex64(&modes);
            let left = hermitian_inner(&u, &av);
            let right = hermitian_inner(&a_star_u, &v);
            let residual = complex_abs(Complex64::new(left.re - right.re, left.im - right.im));
            let scale = (l2_norm(&u) * l2_norm(&av) + l2_norm(&a_star_u) * l2_norm(&v))
                .max(f64::MIN_POSITIVE);
            let relative = residual / scale;
            eprintln!(
                "NUFFT_ADJOINT sign={sign:?} order={order:?} relative_residual={relative:.9e}"
            );
            assert!(
                relative <= ADJOINT_TOLERANCE,
                "type2 sign={sign:?} / opposite-sign type1 order={order:?}: adjoint residual {relative} exceeds {ADJOINT_TOLERANCE}"
            );
        }
    }
}

fn opposite_sign(sign: NufftSign) -> NufftSign {
    match sign {
        NufftSign::Positive => NufftSign::Negative,
        NufftSign::Negative => NufftSign::Positive,
    }
}

struct PointClass {
    label: &'static str,
    start: usize,
    end: usize,
}

fn adversarial_points() -> (Vec<f32>, Vec<PointClass>) {
    let three_pi_inside = f32::from_bits(((3.0 * PI) as f32).to_bits() - 1);
    let sets: [(&str, Vec<f32>); 4] = [
        ("random", seeded_random_points(16, three_pi_inside)),
        (
            "clustered",
            (0..9).map(|index| 0.25 + index as f32 * 1.0e-5).collect(),
        ),
        (
            "boundary",
            vec![
                -three_pi_inside,
                -(PI as f32),
                -(PI as f32) + 1.0e-6,
                (PI as f32) - 1.0e-6,
                PI as f32,
                three_pi_inside,
            ],
        ),
        (
            "duplicates",
            vec![0.375, 0.375, 0.375, 0.375, -1.125, -1.125, -1.125],
        ),
    ];
    let mut points = Vec::new();
    let mut classes = Vec::new();
    for (label, values) in sets {
        let start = points.len();
        points.extend(values);
        classes.push(PointClass {
            label,
            start,
            end: points.len(),
        });
    }
    (points, classes)
}

fn seeded_random_points(count: usize, maximum_magnitude: f32) -> Vec<f32> {
    let mut state = 0x6d2b_79f5u32;
    (0..count)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            let unit = state as f32 / u32::MAX as f32;
            (2.0 * unit - 1.0) * maximum_magnitude
        })
        .collect()
}

fn test_coefficients(count: usize, sign: NufftSign, order: ModeOrder) -> Vec<f32> {
    let sign_bias = if sign == NufftSign::Positive {
        0.17
    } else {
        -0.11
    };
    let order_bias = if order == ModeOrder::Centered {
        0.07
    } else {
        0.19
    };
    (0..count)
        .flat_map(|index| {
            let x = index as f32 + 1.0;
            [
                (x * 0.31 + sign_bias).sin() * 0.7 + order_bias,
                (x * 0.23 - order_bias).cos() * 0.5 - sign_bias,
            ]
        })
        .collect()
}

fn execute_type2(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    plan: &NufftPlan,
    points: &[f32],
    coefficients: &[f32],
) -> Vec<f32> {
    let point_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("wgpu_nufft.test.points"),
        contents: bytemuck::cast_slice(points),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let coefficient_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("wgpu_nufft.test.coefficients"),
        contents: bytemuck::cast_slice(coefficients),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let output_bytes = NufftPlan::required_type2_output_buffer_size_bytes(points.len()).unwrap();
    let output = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_nufft.test.output"),
        size: output_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_nufft.test.readback"),
        size: output_bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("wgpu_nufft.test.type2.encoder"),
    });
    plan.encode_type2_gpu(
        device,
        &mut encoder,
        points.len(),
        &point_buffer,
        &coefficient_buffer,
        &output,
    )
    .unwrap();
    encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, output_bytes);
    queue.submit([encoder.finish()]);

    let slice = readback.slice(..);
    let (sender, receiver) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        sender.send(result).unwrap();
    });
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    receiver.recv().unwrap().unwrap();
    let mapped = slice.get_mapped_range();
    let result = bytemuck::cast_slice::<u8, f32>(&mapped).to_vec();
    drop(mapped);
    readback.unmap();
    result
}

fn execute_type1(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    plan: &NufftPlan,
    points: &[f32],
    strengths: &[f32],
) -> Vec<f32> {
    assert_eq!(strengths.len(), points.len() * 2);
    let point_count = points.len();
    let point_bytes = NufftPlan::required_type1_point_buffer_size_bytes(point_count).unwrap();
    let strength_bytes = NufftPlan::required_type1_strength_buffer_size_bytes(point_count).unwrap();
    let output_bytes = plan.required_type1_output_buffer_size_bytes().unwrap();
    assert_eq!(point_bytes, point_count as u64 * 4);
    assert_eq!(strength_bytes, point_count as u64 * 8);

    // Poisoned tails prove that execution uses `point_count` and exact binding
    // prefixes instead of treating each caller buffer's entire allocation as input.
    let mut uploaded_points = points.to_vec();
    uploaded_points.extend([0.0, 0.5, -0.5]);
    let mut uploaded_strengths = strengths.to_vec();
    uploaded_strengths.extend([8192.0, -4096.0, -2048.0, 1024.0, 512.0, -256.0]);
    let point_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("wgpu_nufft.test.type1.points"),
        contents: bytemuck::cast_slice(&uploaded_points),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let strength_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("wgpu_nufft.test.type1.strengths"),
        contents: bytemuck::cast_slice(&uploaded_strengths),
        usage: wgpu::BufferUsages::STORAGE,
    });

    const OUTPUT_SENTINEL: f32 = 12_345.5;
    const OUTPUT_TAIL_FLOATS: usize = 4;
    let output_float_count = usize::try_from(output_bytes / 4).unwrap();
    let output_initial = vec![OUTPUT_SENTINEL; output_float_count + OUTPUT_TAIL_FLOATS];
    let output = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("wgpu_nufft.test.type1.output"),
        contents: bytemuck::cast_slice(&output_initial),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });
    let readback_bytes = u64::try_from(output_initial.len()).unwrap() * 4;
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_nufft.test.type1.readback"),
        size: readback_bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("wgpu_nufft.test.type1.encoder"),
    });
    plan.encode_type1_gpu(
        device,
        &mut encoder,
        point_count,
        &point_buffer,
        &strength_buffer,
        &output,
    )
    .unwrap();
    encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, readback_bytes);
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
    let values = bytemuck::cast_slice::<u8, f32>(&mapped);
    assert!(
        values[output_float_count..]
            .iter()
            .all(|&value| value.to_bits() == OUTPUT_SENTINEL.to_bits()),
        "type-1 output writes must stay inside the exact advertised prefix"
    );
    let result = values[..output_float_count].to_vec();
    drop(mapped);
    readback.unmap();
    result
}

fn validate_structured_gpu_errors(device: &wgpu::Device, queue: &wgpu::Queue) {
    assert!(matches!(
        NufftPlan::type2_gpu(device, queue, NufftConfig::new([4, 4, 4], 1.0e-6)),
        Err(NufftError::GpuDimensionsUnsupported {
            actual: 3,
            supported: 2,
            ..
        })
    ));
    assert!(matches!(
        NufftPlan::type1_gpu(device, queue, NufftConfig::new([4, 4, 4], 1.0e-6)),
        Err(NufftError::GpuDimensionsUnsupported {
            actual: 3,
            supported: 2,
            ..
        })
    ));
    let config = NufftConfig::new([8], 1.0e-6);
    let host_only = NufftPlan::type2(config.clone()).unwrap();
    let plan = NufftPlan::type2_gpu(device, queue, config).unwrap();
    let type1_config = NufftConfig::new([8], 1.0e-6);
    let type1_host_only = NufftPlan::type1(type1_config.clone()).unwrap();
    let type1_plan = NufftPlan::type1_gpu(device, queue, type1_config).unwrap();
    let storage = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_nufft.test.validation.storage"),
        size: 64,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let too_small = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_nufft.test.validation.too_small"),
        size: 8,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let missing_usage = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_nufft.test.validation.missing_usage"),
        size: 16,
        usage: wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("wgpu_nufft.test.validation.encoder"),
    });
    assert!(matches!(
        host_only.encode_type2_gpu(device, &mut encoder, 1, &storage, &storage, &storage),
        Err(NufftError::GpuExecutionUnavailable { .. })
    ));
    assert!(matches!(
        plan.encode_type2_gpu(device, &mut encoder, 1, &storage, &too_small, &storage),
        Err(NufftError::GpuBufferTooSmall {
            buffer: "type-2 Fourier coefficient",
            ..
        })
    ));
    assert!(matches!(
        plan.encode_type2_gpu(device, &mut encoder, 1, &missing_usage, &storage, &storage),
        Err(NufftError::GpuBufferMissingUsage {
            buffer: "type-2 point",
            ..
        })
    ));
    assert!(matches!(
        plan.encode_type2_gpu(device, &mut encoder, 2, &storage, &storage, &storage),
        Err(NufftError::GpuBufferAliasUnsupported { .. })
    ));
    plan.encode_type2_gpu(
        device,
        &mut encoder,
        0,
        &missing_usage,
        &too_small,
        &missing_usage,
    )
    .unwrap();

    assert!(matches!(
        type1_host_only.encode_type1_gpu(device, &mut encoder, 1, &storage, &storage, &storage,),
        Err(NufftError::GpuExecutionUnavailable { .. })
    ));
    assert!(matches!(
        plan.encode_type1_gpu(device, &mut encoder, 1, &storage, &storage, &storage,),
        Err(NufftError::GpuExecutionUnavailable { .. })
    ));
    assert!(matches!(
        type1_plan.encode_type1_gpu(device, &mut encoder, 3, &too_small, &storage, &storage,),
        Err(NufftError::GpuBufferTooSmall { .. })
    ));
    assert!(matches!(
        type1_plan.encode_type1_gpu(device, &mut encoder, 2, &storage, &too_small, &storage,),
        Err(NufftError::GpuBufferTooSmall { .. })
    ));
    assert!(matches!(
        type1_plan.encode_type1_gpu(device, &mut encoder, 1, &storage, &storage, &too_small,),
        Err(NufftError::GpuBufferTooSmall { .. })
    ));
    assert!(matches!(
        type1_plan.encode_type1_gpu(device, &mut encoder, 1, &missing_usage, &storage, &storage,),
        Err(NufftError::GpuBufferMissingUsage { .. })
    ));
    assert!(matches!(
        type1_plan.encode_type1_gpu(device, &mut encoder, 1, &storage, &missing_usage, &storage,),
        Err(NufftError::GpuBufferMissingUsage { .. })
    ));
    assert!(matches!(
        type1_plan.encode_type1_gpu(device, &mut encoder, 1, &storage, &storage, &missing_usage,),
        Err(NufftError::GpuBufferMissingUsage { .. })
    ));
    assert!(matches!(
        type1_plan.encode_type1_gpu(
            device,
            &mut encoder,
            usize::MAX,
            &storage,
            &storage,
            &storage,
        ),
        Err(NufftError::LengthOverflow { .. })
    ));
    type1_plan
        .encode_type1_gpu(
            device,
            &mut encoder,
            0,
            &missing_usage,
            &too_small,
            &storage,
        )
        .unwrap();
}

fn interleaved_to_complex64(values: &[f32]) -> Vec<Complex64> {
    values
        .chunks_exact(2)
        .map(|pair| Complex64::new(f64::from(pair[0]), f64::from(pair[1])))
        .collect()
}

fn relative_l2(actual: &[Complex64], reference: &[Complex64]) -> f64 {
    let error = actual
        .iter()
        .zip(reference)
        .map(|(actual, reference)| {
            (actual.re - reference.re).powi(2) + (actual.im - reference.im).powi(2)
        })
        .sum::<f64>()
        .sqrt();
    let norm = reference
        .iter()
        .map(|value| value.re * value.re + value.im * value.im)
        .sum::<f64>()
        .sqrt();
    error / norm.max(f64::MIN_POSITIVE)
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

fn l2_norm(values: &[Complex64]) -> f64 {
    values
        .iter()
        .map(|value| value.re * value.re + value.im * value.im)
        .sum::<f64>()
        .sqrt()
}

fn complex_abs(value: Complex64) -> f64 {
    value.re.hypot(value.im)
}
