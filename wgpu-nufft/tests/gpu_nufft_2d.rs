//! Opt-in 2D GPU validation against the direct f64 NDFT oracle.

use std::f64::consts::PI;
use std::sync::mpsc;

use wgpu::util::DeviceExt;
use wgpu_nufft::{
    reference_type1_f64, reference_type2_f64, Complex64, ModeOrder, NufftConfig, NufftPlan,
    NufftSign,
};

const DIMENSIONS: usize = 2;
const FINUFFT_CUDA_FLOAT_TOLERANCE_FACTOR: f64 = 20.0;
const ADJOINT_TOLERANCE: f64 = 2.0e-5;

#[test]
fn gpu_2d_type1_and_type2_match_oracle_and_are_adjoint() {
    if std::env::var_os("WGPU_FFT_RUN_GPU_TESTS").is_none() {
        eprintln!("skipping GPU test; set WGPU_FFT_RUN_GPU_TESTS=1 to run it");
        return;
    }
    pollster::block_on(run_gpu_2d_cases());
}

async fn run_gpu_2d_cases() {
    let Some(context) = wgpu_fft::device::request_default_device().await else {
        panic!("WGPU_FFT_RUN_GPU_TESTS was set but no suitable adapter was found");
    };
    let info = context.adapter.get_info();
    eprintln!(
        "gpu_nufft_2d adapter: {} backend={:?} driver={} {}",
        info.name, info.backend, info.driver, info.driver_info
    );

    validate_accuracy_matrix(&context.device, &context.queue);
    validate_type2_duplicate_outputs(&context.device, &context.queue);
    validate_type1_repeat_determinism(&context.device, &context.queue);
    validate_type1_scratch_grow_then_shrink(&context.device, &context.queue);
    validate_adjoint_consistency(&context.device, &context.queue);
    validate_non_square_case(&context.device, &context.queue);
    #[cfg(feature = "type1-2d-tile-prototype")]
    {
        validate_tiled_equivalence_and_oracle(&context.device, &context.queue);
        validate_tiled_tolerance_sweep(&context.device, &context.queue);
        validate_tiled_adjoint_consistency(&context.device, &context.queue);
        validate_tiled_scratch_grow_then_shrink(&context.device, &context.queue);
    }

    // Keep the native Windows teardown workaround used by the other GPU suites.
    std::mem::forget(context);
}

fn validate_accuracy_matrix(device: &wgpu::Device, queue: &wgpu::Queue) {
    let mode_shape = [12, 10];
    let classes = adversarial_point_classes();
    let all_points = classes
        .iter()
        .flat_map(|class| class.points.iter().copied())
        .collect::<Vec<_>>();
    let all_points_f64 = all_points
        .iter()
        .map(|&value| f64::from(value))
        .collect::<Vec<_>>();
    let mode_count = mode_shape.iter().product();

    for eps in [1.0e-2, 1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6] {
        let tolerance = FINUFFT_CUDA_FLOAT_TOLERANCE_FACTOR * eps;
        for sign in [NufftSign::Positive, NufftSign::Negative] {
            for order in [ModeOrder::Centered, ModeOrder::Fft] {
                let config = NufftConfig::new(mode_shape, eps)
                    .with_sign(sign)
                    .with_mode_order(order);
                let type2_plan = NufftPlan::type2_gpu(device, queue, config.clone()).unwrap();
                let type1_plan = NufftPlan::type1_gpu(device, queue, config.clone()).unwrap();
                let coefficients = test_values(mode_count, sign, order);

                let actual_type2 = interleaved_to_complex64(&execute_type2(
                    device,
                    queue,
                    &type2_plan,
                    &all_points,
                    &coefficients,
                ));
                let reference_type2 = reference_type2_f64(
                    &config,
                    &all_points_f64,
                    &interleaved_to_complex64(&coefficients),
                )
                .unwrap();
                let mut point_start = 0;
                for class in &classes {
                    let point_count = class.point_count();
                    let point_end = point_start + point_count;
                    let error = relative_l2(
                        &actual_type2[point_start..point_end],
                        &reference_type2[point_start..point_end],
                    );
                    eprintln!(
                        "NUFFT_2D_ACCURACY kind=type2 shape=12x10 eps={eps:.0e} sign={sign:?} order={order:?} class={} relative_l2={error:.9e}",
                        class.label
                    );
                    assert!(
                        error <= tolerance,
                        "2D type2 eps={eps} sign={sign:?} order={order:?} class={}: relative l2 {error} exceeds {tolerance}",
                        class.label
                    );
                    point_start = point_end;
                }

                for class in &classes {
                    let strengths = test_values(class.point_count(), opposite_sign(sign), order);
                    let actual_type1 = interleaved_to_complex64(&execute_type1(
                        device,
                        queue,
                        &type1_plan,
                        &class.points,
                        &strengths,
                    ));
                    let reference_type1 = reference_type1_f64(
                        &config,
                        &points_f64(&class.points),
                        &interleaved_to_complex64(&strengths),
                    )
                    .unwrap();
                    let error = relative_l2(&actual_type1, &reference_type1);
                    eprintln!(
                        "NUFFT_2D_ACCURACY kind=type1 shape=12x10 eps={eps:.0e} sign={sign:?} order={order:?} class={} relative_l2={error:.9e}",
                        class.label
                    );
                    assert!(
                        error <= tolerance,
                        "2D type1 eps={eps} sign={sign:?} order={order:?} class={}: relative l2 {error} exceeds {tolerance}",
                        class.label
                    );
                }
            }
        }
    }
}

fn validate_type2_duplicate_outputs(device: &wgpu::Device, queue: &wgpu::Queue) {
    let config = NufftConfig::new([12, 10], 1.0e-6)
        .with_sign(NufftSign::Negative)
        .with_mode_order(ModeOrder::Fft);
    let plan = NufftPlan::type2_gpu(device, queue, config).unwrap();
    let points = [
        0.375f32, -1.125, 0.375, -1.125, 0.375, -1.125, -2.25, 1.75, -2.25, 1.75,
    ];
    let coefficients = test_values(120, NufftSign::Negative, ModeOrder::Fft);
    let actual = execute_type2(device, queue, &plan, &points, &coefficients);
    for point in 1..3 {
        assert_complex_bits_equal(&actual, 0, point, "first duplicate group");
    }
    assert_complex_bits_equal(&actual, 3, 4, "second duplicate group");
}

fn validate_type1_repeat_determinism(device: &wgpu::Device, queue: &wgpu::Queue) {
    let config = NufftConfig::new([12, 10], 1.0e-6)
        .with_sign(NufftSign::Positive)
        .with_mode_order(ModeOrder::Centered);
    let plan = NufftPlan::type1_gpu(device, queue, config).unwrap();
    let class = adversarial_point_classes()
        .into_iter()
        .find(|class| class.label == "clustered")
        .unwrap();
    let strengths = test_values(
        class.point_count(),
        NufftSign::Positive,
        ModeOrder::Centered,
    );
    let first = execute_type1(device, queue, &plan, &class.points, &strengths);
    let second = execute_type1(device, queue, &plan, &class.points, &strengths);
    assert_eq!(
        first
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        second
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        "2D type1 must be bitwise deterministic across repeated executions"
    );
}

fn validate_type1_scratch_grow_then_shrink(device: &wgpu::Device, queue: &wgpu::Queue) {
    let eps = 1.0e-6;
    let config = NufftConfig::new([12, 10], eps)
        .with_sign(NufftSign::Negative)
        .with_mode_order(ModeOrder::Fft);
    let plan = NufftPlan::type1_gpu(device, queue, config.clone()).unwrap();

    // Cross both a point-dispatch boundary and the growable scratch capacity
    // before reusing the same plan for a much smaller, unrelated point set.
    let large_points = seeded_random_points(257, 0xc851_4a79);
    let mut large_strengths = test_values(257, NufftSign::Positive, ModeOrder::Centered);
    for value in &mut large_strengths {
        *value *= 64.0;
    }

    let three_pi_inside = f32::from_bits(((3.0 * PI) as f32).to_bits() - 1);
    let small_points = vec![
        0.0,
        0.0,
        0.0,
        0.0,
        PI as f32,
        -(PI as f32),
        -three_pi_inside,
        three_pi_inside,
        0.3125,
        -0.6875,
        -1.75,
        2.25,
        2.875,
        -2.625,
    ];
    let small_strengths = vec![
        0.125, -0.25, -0.375, 0.5, 0.625, -0.75, -0.875, 1.0, 1.125, -1.25, -1.375, 1.5, 1.625,
        -1.75,
    ];
    assert!(large_points.len() > small_points.len());

    for (label, points, strengths) in [
        ("large", large_points.as_slice(), large_strengths.as_slice()),
        ("small", small_points.as_slice(), small_strengths.as_slice()),
    ] {
        let actual =
            interleaved_to_complex64(&execute_type1(device, queue, &plan, points, strengths));
        let reference = reference_type1_f64(
            &config,
            &points_f64(points),
            &interleaved_to_complex64(strengths),
        )
        .unwrap();
        let error = relative_l2(&actual, &reference);
        let tolerance = FINUFFT_CUDA_FLOAT_TOLERANCE_FACTOR * eps;
        eprintln!(
            "NUFFT_2D_SCRATCH_REUSE phase={label} M={} relative_l2={error:.9e}",
            points.len() / DIMENSIONS
        );
        assert!(
            error <= tolerance,
            "2D type1 scratch reuse {label} execution: relative l2 {error} exceeds {tolerance}"
        );
    }
}

fn validate_adjoint_consistency(device: &wgpu::Device, queue: &wgpu::Queue) {
    let points = seeded_random_points(13, 0x51a9_7e2d);
    let mode_shape = [12, 10];
    let mode_count = mode_shape.iter().product();
    for sign in [NufftSign::Positive, NufftSign::Negative] {
        for order in [ModeOrder::Centered, ModeOrder::Fft] {
            let type2_config = NufftConfig::new(mode_shape, 1.0e-6)
                .with_sign(sign)
                .with_mode_order(order);
            let type1_config = NufftConfig::new(mode_shape, 1.0e-6)
                .with_sign(opposite_sign(sign))
                .with_mode_order(order);
            let type2_plan = NufftPlan::type2_gpu(device, queue, type2_config).unwrap();
            let type1_plan = NufftPlan::type1_gpu(device, queue, type1_config).unwrap();
            let modes = test_values(mode_count, sign, order);
            let point_values = test_values(points.len() / DIMENSIONS, opposite_sign(sign), order);

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
                "NUFFT_2D_ADJOINT shape=12x10 sign={sign:?} order={order:?} relative_residual={relative:.9e}"
            );
            assert!(
                relative <= ADJOINT_TOLERANCE,
                "2D type2 sign={sign:?} / opposite-sign type1 order={order:?}: adjoint residual {relative} exceeds {ADJOINT_TOLERANCE}"
            );
        }
    }
}

fn validate_non_square_case(device: &wgpu::Device, queue: &wgpu::Queue) {
    let mode_shape = [256, 1024];
    let eps = 1.0e-5;
    let tolerance = FINUFFT_CUDA_FLOAT_TOLERANCE_FACTOR * eps;
    let config = NufftConfig::new(mode_shape, eps)
        .with_sign(NufftSign::Negative)
        .with_mode_order(ModeOrder::Centered);
    let points = [-2.75f32, 0.125, 0.375, -1.5, 2.25, 2.875];
    let coefficients = test_values(
        mode_shape.iter().product(),
        config.sign(),
        config.mode_order(),
    );
    let type2_plan = NufftPlan::type2_gpu(device, queue, config.clone()).unwrap();
    let actual_type2 = interleaved_to_complex64(&execute_type2(
        device,
        queue,
        &type2_plan,
        &points,
        &coefficients,
    ));
    let reference_type2 = reference_type2_f64(
        &config,
        &points_f64(&points),
        &interleaved_to_complex64(&coefficients),
    )
    .unwrap();
    let type2_error = relative_l2(&actual_type2, &reference_type2);
    eprintln!(
        "NUFFT_2D_ACCURACY kind=type2 shape=256x1024 eps={eps:.0e} class=sparse relative_l2={type2_error:.9e}"
    );
    assert!(type2_error <= tolerance);

    let strengths = test_values(
        points.len() / DIMENSIONS,
        opposite_sign(config.sign()),
        config.mode_order(),
    );
    #[cfg(feature = "type1-2d-tile-prototype")]
    let type1_plan =
        NufftPlan::type1_gpu_with_global_2d_gather_for_testing(device, queue, config.clone())
            .unwrap();
    #[cfg(not(feature = "type1-2d-tile-prototype"))]
    let type1_plan = NufftPlan::type1_gpu(device, queue, config.clone()).unwrap();
    let global_type1 = execute_type1(device, queue, &type1_plan, &points, &strengths);
    let actual_type1 = interleaved_to_complex64(&global_type1);
    let reference_type1 = reference_type1_f64(
        &config,
        &points_f64(&points),
        &interleaved_to_complex64(&strengths),
    )
    .unwrap();
    let type1_error = relative_l2(&actual_type1, &reference_type1);
    eprintln!(
        "NUFFT_2D_ACCURACY kind=type1 shape=256x1024 eps={eps:.0e} class=sparse relative_l2={type1_error:.9e}"
    );
    assert!(type1_error <= tolerance);

    #[cfg(feature = "type1-2d-tile-prototype")]
    {
        let tiled_plan =
            NufftPlan::type1_gpu_with_tiled_2d_gather_for_testing(device, queue, config.clone())
                .unwrap();
        let tiled_type1 = execute_type1(device, queue, &tiled_plan, &points, &strengths);
        assert_f32_bits_equal(
            &tiled_type1,
            &global_type1,
            "non-square 256x1024 tiled/global type-1",
        );
        let tiled_error = relative_l2(&interleaved_to_complex64(&tiled_type1), &reference_type1);
        eprintln!(
            "NUFFT_2D_TILED_ACCURACY kind=type1 shape=256x1024 eps={eps:.0e} class=sparse relative_l2={tiled_error:.9e}"
        );
        assert!(tiled_error <= tolerance);
    }
}

#[cfg(feature = "type1-2d-tile-prototype")]
fn validate_tiled_equivalence_and_oracle(device: &wgpu::Device, queue: &wgpu::Queue) {
    let mode_shape = [17, 19];
    let eps = 1.0e-6;
    let tolerance = FINUFFT_CUDA_FLOAT_TOLERANCE_FACTOR * eps;
    let classes = tiled_point_classes();

    for sign in [NufftSign::Positive, NufftSign::Negative] {
        for order in [ModeOrder::Centered, ModeOrder::Fft] {
            let config = NufftConfig::new(mode_shape, eps)
                .with_sign(sign)
                .with_mode_order(order);
            let global = NufftPlan::type1_gpu_with_global_2d_gather_for_testing(
                device,
                queue,
                config.clone(),
            )
            .unwrap();
            let tiled = NufftPlan::type1_gpu_with_tiled_2d_gather_for_testing(
                device,
                queue,
                config.clone(),
            )
            .unwrap();

            for class in &classes {
                let strengths = test_values(class.point_count(), opposite_sign(sign), order);
                let global_output =
                    execute_type1(device, queue, &global, &class.points, &strengths);
                let tiled_output = execute_type1(device, queue, &tiled, &class.points, &strengths);
                assert_f32_bits_equal(
                    &tiled_output,
                    &global_output,
                    &format!(
                        "17x19 tiled/global type-1 sign={sign:?} order={order:?} class={}",
                        class.label
                    ),
                );

                let reference = reference_type1_f64(
                    &config,
                    &points_f64(&class.points),
                    &interleaved_to_complex64(&strengths),
                )
                .unwrap();
                let error = relative_l2(&interleaved_to_complex64(&tiled_output), &reference);
                eprintln!(
                    "NUFFT_2D_TILED_ACCURACY kind=type1 shape=17x19 eps={eps:.0e} sign={sign:?} order={order:?} class={} relative_l2={error:.9e}",
                    class.label
                );
                assert!(
                    error <= tolerance,
                    "17x19 tiled type1 sign={sign:?} order={order:?} class={}: relative l2 {error} exceeds {tolerance}",
                    class.label
                );
            }
        }
    }
}

#[cfg(feature = "type1-2d-tile-prototype")]
fn validate_tiled_tolerance_sweep(device: &wgpu::Device, queue: &wgpu::Queue) {
    let mode_shape = [17, 19];
    let classes = adversarial_point_classes()
        .into_iter()
        .filter(|class| matches!(class.label, "clustered" | "boundary"))
        .collect::<Vec<_>>();

    for eps in [1.0e-2, 1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6] {
        let tolerance = FINUFFT_CUDA_FLOAT_TOLERANCE_FACTOR * eps;
        for sign in [NufftSign::Positive, NufftSign::Negative] {
            let config = NufftConfig::new(mode_shape, eps)
                .with_sign(sign)
                .with_mode_order(ModeOrder::Centered);
            let tiled = NufftPlan::type1_gpu_with_tiled_2d_gather_for_testing(
                device,
                queue,
                config.clone(),
            )
            .unwrap();
            for class in &classes {
                let strengths = test_values(
                    class.point_count(),
                    opposite_sign(sign),
                    ModeOrder::Centered,
                );
                let actual = interleaved_to_complex64(&execute_type1(
                    device,
                    queue,
                    &tiled,
                    &class.points,
                    &strengths,
                ));
                let reference = reference_type1_f64(
                    &config,
                    &points_f64(&class.points),
                    &interleaved_to_complex64(&strengths),
                )
                .unwrap();
                let error = relative_l2(&actual, &reference);
                eprintln!(
                    "NUFFT_2D_TILED_SWEEP shape=17x19 eps={eps:.0e} sign={sign:?} class={} relative_l2={error:.9e}",
                    class.label
                );
                assert!(
                    error <= tolerance,
                    "17x19 tiled sweep eps={eps} sign={sign:?} class={}: relative l2 {error} exceeds {tolerance}",
                    class.label
                );
            }
        }
    }
}

#[cfg(feature = "type1-2d-tile-prototype")]
fn validate_tiled_adjoint_consistency(device: &wgpu::Device, queue: &wgpu::Queue) {
    let points = seeded_random_points(13, 0x27c4_9b8d);
    let mode_shape = [17, 19];
    let mode_count = mode_shape.iter().product();
    for sign in [NufftSign::Positive, NufftSign::Negative] {
        for order in [ModeOrder::Centered, ModeOrder::Fft] {
            let type2_config = NufftConfig::new(mode_shape, 1.0e-6)
                .with_sign(sign)
                .with_mode_order(order);
            let type1_config = NufftConfig::new(mode_shape, 1.0e-6)
                .with_sign(opposite_sign(sign))
                .with_mode_order(order);
            let type2_plan = NufftPlan::type2_gpu(device, queue, type2_config).unwrap();
            let type1_plan =
                NufftPlan::type1_gpu_with_tiled_2d_gather_for_testing(device, queue, type1_config)
                    .unwrap();
            let modes = test_values(mode_count, sign, order);
            let point_values = test_values(points.len() / DIMENSIONS, opposite_sign(sign), order);

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
                "NUFFT_2D_TILED_ADJOINT shape=17x19 sign={sign:?} order={order:?} relative_residual={relative:.9e}"
            );
            assert!(
                relative <= ADJOINT_TOLERANCE,
                "17x19 tiled type2 sign={sign:?} / opposite-sign type1 order={order:?}: adjoint residual {relative} exceeds {ADJOINT_TOLERANCE}"
            );
        }
    }
}

#[cfg(feature = "type1-2d-tile-prototype")]
fn validate_tiled_scratch_grow_then_shrink(device: &wgpu::Device, queue: &wgpu::Queue) {
    let eps = 1.0e-6;
    let config = NufftConfig::new([17, 19], eps)
        .with_sign(NufftSign::Negative)
        .with_mode_order(ModeOrder::Fft);
    let global =
        NufftPlan::type1_gpu_with_global_2d_gather_for_testing(device, queue, config.clone())
            .unwrap();
    let tiled =
        NufftPlan::type1_gpu_with_tiled_2d_gather_for_testing(device, queue, config.clone())
            .unwrap();

    // All 513 points occupy one bin. This exceeds the tiled point-cache capacity
    // and therefore forces multiple uniform cache batches before the smaller run.
    let large_points = dense_same_bin_points(513);
    let large_strengths = test_values(513, NufftSign::Positive, ModeOrder::Centered);
    let small_points = vec![
        0.0,
        0.0,
        PI as f32,
        -(PI as f32),
        -2.75,
        2.5,
        0.3125,
        -0.6875,
        -1.75,
        2.25,
        2.875,
        -2.625,
        -0.125,
        0.75,
    ];
    let small_strengths = test_values(7, NufftSign::Positive, ModeOrder::Centered);

    let global_large = execute_type1(device, queue, &global, &large_points, &large_strengths);
    let global_small = execute_type1(device, queue, &global, &small_points, &small_strengths);

    for (label, points, strengths, expected) in [
        (
            "dense-multibatch",
            large_points.as_slice(),
            large_strengths.as_slice(),
            global_large.as_slice(),
        ),
        (
            "small-after-grow",
            small_points.as_slice(),
            small_strengths.as_slice(),
            global_small.as_slice(),
        ),
    ] {
        let actual = execute_type1(device, queue, &tiled, points, strengths);
        assert_f32_bits_equal(
            &actual,
            expected,
            &format!("17x19 tiled scratch reuse {label}"),
        );
        let reference = reference_type1_f64(
            &config,
            &points_f64(points),
            &interleaved_to_complex64(strengths),
        )
        .unwrap();
        let error = relative_l2(&interleaved_to_complex64(&actual), &reference);
        let tolerance = FINUFFT_CUDA_FLOAT_TOLERANCE_FACTOR * eps;
        eprintln!(
            "NUFFT_2D_TILED_SCRATCH_REUSE phase={label} M={} relative_l2={error:.9e}",
            points.len() / DIMENSIONS
        );
        assert!(
            error <= tolerance,
            "17x19 tiled scratch reuse {label}: relative l2 {error} exceeds {tolerance}"
        );
    }
}

#[cfg(feature = "type1-2d-tile-prototype")]
fn tiled_point_classes() -> Vec<PointClass> {
    let mut classes = adversarial_point_classes();
    classes[0] = PointClass {
        label: "seeded",
        points: seeded_random_points(257, 0xa67f_4d31),
    };
    classes.push(PointClass {
        label: "dense-same-bin-multibatch",
        points: dense_same_bin_points(513),
    });
    classes
}

#[cfg(feature = "type1-2d-tile-prototype")]
fn dense_same_bin_points(point_count: usize) -> Vec<f32> {
    (0..point_count)
        .flat_map(|_| [0.25f32, -0.375f32])
        .collect()
}

struct PointClass {
    label: &'static str,
    points: Vec<f32>,
}

impl PointClass {
    fn point_count(&self) -> usize {
        self.points.len() / DIMENSIONS
    }
}

fn adversarial_point_classes() -> Vec<PointClass> {
    let three_pi_inside = f32::from_bits(((3.0 * PI) as f32).to_bits() - 1);
    vec![
        PointClass {
            label: "random",
            points: seeded_random_points(12, 0x6d2b_79f5),
        },
        PointClass {
            label: "clustered",
            points: (0..8)
                .flat_map(|index| [0.25 + index as f32 * 1.0e-5, -0.375 + index as f32 * 7.0e-6])
                .collect(),
        },
        PointClass {
            label: "boundary",
            points: vec![
                -three_pi_inside,
                three_pi_inside,
                -(PI as f32),
                PI as f32,
                -(PI as f32) + 1.0e-6,
                (PI as f32) - 1.0e-6,
                0.0,
                -three_pi_inside,
                three_pi_inside,
                0.0,
            ],
        },
        PointClass {
            label: "duplicates",
            points: vec![
                0.375, -1.125, 0.375, -1.125, 0.375, -1.125, 0.375, -1.125, -2.25, 1.75, -2.25,
                1.75,
            ],
        },
    ]
}

fn seeded_random_points(point_count: usize, mut state: u32) -> Vec<f32> {
    let maximum = f32::from_bits(((3.0 * PI) as f32).to_bits() - 1);
    (0..point_count * DIMENSIONS)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            let unit = state as f32 / u32::MAX as f32;
            (2.0 * unit - 1.0) * maximum
        })
        .collect()
}

fn test_values(count: usize, sign: NufftSign, order: ModeOrder) -> Vec<f32> {
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
            let value = index as f32 + 1.0;
            [
                (value * 0.31 + sign_bias).sin() * 0.7 + order_bias,
                (value * 0.23 - order_bias).cos() * 0.5 - sign_bias,
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
    assert!(points.len().is_multiple_of(DIMENSIONS));
    let point_count = points.len() / DIMENSIONS;
    let point_bytes = plan.required_point_buffer_size_bytes(point_count).unwrap();
    assert_eq!(point_bytes, points.len() as u64 * 4);
    let coefficient_bytes = plan.required_type2_coefficient_buffer_size_bytes().unwrap();
    assert_eq!(coefficient_bytes, coefficients.len() as u64 * 4);
    let output_bytes = NufftPlan::required_type2_output_buffer_size_bytes(point_count).unwrap();

    let point_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("wgpu_nufft.test2d.type2.points"),
        contents: bytemuck::cast_slice(points),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let coefficient_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("wgpu_nufft.test2d.type2.coefficients"),
        contents: bytemuck::cast_slice(coefficients),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let output = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_nufft.test2d.type2.output"),
        size: output_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback = readback_buffer(device, output_bytes, "wgpu_nufft.test2d.type2.readback");
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("wgpu_nufft.test2d.type2.encoder"),
    });
    plan.encode_type2_gpu(
        device,
        &mut encoder,
        point_count,
        &point_buffer,
        &coefficient_buffer,
        &output,
    )
    .unwrap();
    encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, output_bytes);
    queue.submit([encoder.finish()]);
    read_f32_buffer(device, &readback)
}

fn execute_type1(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    plan: &NufftPlan,
    points: &[f32],
    strengths: &[f32],
) -> Vec<f32> {
    assert!(points.len().is_multiple_of(DIMENSIONS));
    let point_count = points.len() / DIMENSIONS;
    assert_eq!(strengths.len(), point_count * 2);
    let point_bytes = plan.required_point_buffer_size_bytes(point_count).unwrap();
    assert_eq!(point_bytes, points.len() as u64 * 4);
    let output_bytes = plan.required_type1_output_buffer_size_bytes().unwrap();

    let point_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("wgpu_nufft.test2d.type1.points"),
        contents: bytemuck::cast_slice(points),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let strength_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("wgpu_nufft.test2d.type1.strengths"),
        contents: bytemuck::cast_slice(strengths),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let output = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_nufft.test2d.type1.output"),
        size: output_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback = readback_buffer(device, output_bytes, "wgpu_nufft.test2d.type1.readback");
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("wgpu_nufft.test2d.type1.encoder"),
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
    encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, output_bytes);
    queue.submit([encoder.finish()]);
    read_f32_buffer(device, &readback)
}

fn readback_buffer(device: &wgpu::Device, size: u64, label: &str) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    })
}

fn read_f32_buffer(device: &wgpu::Device, buffer: &wgpu::Buffer) -> Vec<f32> {
    let slice = buffer.slice(..);
    let (sender, receiver) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        sender.send(result).unwrap();
    });
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    receiver.recv().unwrap().unwrap();
    let mapped = slice.get_mapped_range();
    let result = bytemuck::cast_slice::<u8, f32>(&mapped).to_vec();
    drop(mapped);
    buffer.unmap();
    result
}

fn points_f64(points: &[f32]) -> Vec<f64> {
    points.iter().map(|&value| f64::from(value)).collect()
}

fn interleaved_to_complex64(values: &[f32]) -> Vec<Complex64> {
    values
        .chunks_exact(2)
        .map(|pair| Complex64::new(f64::from(pair[0]), f64::from(pair[1])))
        .collect()
}

fn relative_l2(actual: &[Complex64], reference: &[Complex64]) -> f64 {
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

fn opposite_sign(sign: NufftSign) -> NufftSign {
    match sign {
        NufftSign::Positive => NufftSign::Negative,
        NufftSign::Negative => NufftSign::Positive,
    }
}

fn assert_complex_bits_equal(values: &[f32], first: usize, second: usize, context: &str) {
    let first = &values[first * 2..first * 2 + 2];
    let second = &values[second * 2..second * 2 + 2];
    assert_eq!(first[0].to_bits(), second[0].to_bits(), "{context} real");
    assert_eq!(first[1].to_bits(), second[1].to_bits(), "{context} imag");
}

#[cfg(feature = "type1-2d-tile-prototype")]
fn assert_f32_bits_equal(actual: &[f32], expected: &[f32], context: &str) {
    assert_eq!(actual.len(), expected.len(), "{context} length");
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(
            actual.to_bits(),
            expected.to_bits(),
            "{context} word {index}: actual={actual:?} expected={expected:?}"
        );
    }
}
