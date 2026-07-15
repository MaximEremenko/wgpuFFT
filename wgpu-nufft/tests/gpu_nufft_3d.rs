use std::f64::consts::PI;
use std::sync::mpsc;
use std::time::Duration;

use wgpu::util::DeviceExt;
use wgpu_nufft::{reference_type2_f64, Complex64, ModeOrder, NufftConfig, NufftPlan, NufftSign};

const DIMENSIONS: usize = 3;
const FINUFFT_FLOAT_TOLERANCE_FACTOR_3D: f64 = 32.0;

#[test]
fn gpu_3d_type2_matches_oracle_and_reuses_its_plan() {
    if std::env::var_os("WGPU_FFT_RUN_GPU_TESTS").is_none() {
        eprintln!("skipping GPU test; set WGPU_FFT_RUN_GPU_TESTS=1 to run it");
        return;
    }
    pollster::block_on(run_gpu_3d_type2_cases());
}

async fn run_gpu_3d_type2_cases() {
    let Some(context) = wgpu_fft::device::request_default_device().await else {
        panic!("WGPU_FFT_RUN_GPU_TESTS was set but no suitable adapter was found");
    };
    let info = context.adapter.get_info();
    eprintln!(
        "gpu_nufft_3d adapter: {} backend={:?} driver={} {}",
        info.name, info.backend, info.driver, info.driver_info
    );

    validate_type2_oracle_matrix(&context.device, &context.queue);
    validate_non_cubic_type2(&context.device, &context.queue);
    validate_type2_back_to_back_grow_then_shrink(&context.device, &context.queue);
    if std::env::var_os("WGPU_NUFFT_RUN_LARGE_GPU_TESTS").is_some() {
        #[cfg(not(feature = "gpu-profiling"))]
        panic!(
            "WGPU_NUFFT_RUN_LARGE_GPU_TESTS requires --features gpu-profiling so the selected FFT route is verified"
        );
        #[cfg(feature = "gpu-profiling")]
        validate_large_256_cubed_type2(&context.device, &context.queue);
    } else {
        eprintln!("skipping 256^3 NUFFT case; set WGPU_NUFFT_RUN_LARGE_GPU_TESTS=1 to run it");
    }

    #[cfg(windows)]
    std::mem::forget(context);
}

fn validate_type2_oracle_matrix(device: &wgpu::Device, queue: &wgpu::Queue) {
    let mode_shape = [8, 10, 12];
    let classes = point_classes();
    for eps in [1.0e-2, 1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6] {
        let tolerance = FINUFFT_FLOAT_TOLERANCE_FACTOR_3D * eps;
        for sign in [NufftSign::Positive, NufftSign::Negative] {
            for order in [ModeOrder::Centered, ModeOrder::Fft] {
                let config = NufftConfig::new(mode_shape, eps)
                    .with_sign(sign)
                    .with_mode_order(order);
                let plan = NufftPlan::type2_gpu(device, queue, config.clone()).unwrap();
                assert_eq!(plan.fine_grid_shape().len(), DIMENSIONS);
                let coefficients = test_values(config.mode_count().unwrap(), sign, order);
                for class in &classes {
                    let actual = interleaved_to_complex64(&execute_type2(
                        device,
                        queue,
                        &plan,
                        &class.points,
                        &coefficients,
                    ));
                    let reference = reference_type2_f64(
                        &config,
                        &points_f64(&class.points),
                        &interleaved_to_complex64(&coefficients),
                    )
                    .unwrap();
                    let error = relative_l2(&actual, &reference);
                    eprintln!(
                        "NUFFT_3D_ACCURACY kind=type2 shape=8x10x12 eps={eps:.0e} sign={sign:?} order={order:?} class={} relative_l2={error:.9e}",
                        class.label
                    );
                    assert!(
                        error <= tolerance,
                        "3D type2 eps={eps} sign={sign:?} order={order:?} class={}: relative l2 {error} exceeds {tolerance}",
                        class.label
                    );
                }
            }
        }
    }
}

fn validate_non_cubic_type2(device: &wgpu::Device, queue: &wgpu::Queue) {
    let mode_shape = [32, 64, 128];
    let points = vec![
        -PI as f32, 0.25, PI as f32, -0.375, 2.75, -2.5, 1.125, -1.75, 0.0,
    ];
    for sign in [NufftSign::Positive, NufftSign::Negative] {
        let config = NufftConfig::new(mode_shape, 1.0e-5).with_sign(sign);
        let plan = NufftPlan::type2_gpu(device, queue, config.clone()).unwrap();
        assert_eq!(plan.fine_grid_shape(), [64, 128, 256]);
        let coefficients = test_values(config.mode_count().unwrap(), sign, ModeOrder::Centered);
        let actual =
            interleaved_to_complex64(&execute_type2(device, queue, &plan, &points, &coefficients));
        let reference = reference_type2_f64(
            &config,
            &points_f64(&points),
            &interleaved_to_complex64(&coefficients),
        )
        .unwrap();
        let error = relative_l2(&actual, &reference);
        eprintln!(
            "NUFFT_3D_ACCURACY kind=type2 shape=32x64x128 eps=1e-5 sign={sign:?} class=non-cubic relative_l2={error:.9e}"
        );
        assert!(error <= FINUFFT_FLOAT_TOLERANCE_FACTOR_3D * 1.0e-5);
    }
}

fn validate_type2_back_to_back_grow_then_shrink(device: &wgpu::Device, queue: &wgpu::Queue) {
    let config = NufftConfig::new([9, 10, 11], 1.0e-6)
        .with_sign(NufftSign::Negative)
        .with_mode_order(ModeOrder::Fft);
    let plan = NufftPlan::type2_gpu(device, queue, config.clone()).unwrap();
    let coefficients = test_values(
        config.mode_count().unwrap(),
        config.sign(),
        config.mode_order(),
    );
    let large_points = seeded_random_points(257, 0x51a7_3d01);
    let small_points = vec![
        0.0,
        0.0,
        0.0,
        PI as f32,
        -(PI as f32),
        0.5,
        -2.75,
        2.5,
        -0.125,
        1.25,
        -1.5,
        2.0,
        -0.375,
        0.625,
        -0.875,
        2.875,
        -2.625,
        0.125,
        -1.0,
        1.75,
        -2.25,
    ];
    for (label, points) in [
        ("large", large_points.as_slice()),
        ("small-after-large", small_points.as_slice()),
    ] {
        let actual =
            interleaved_to_complex64(&execute_type2(device, queue, &plan, points, &coefficients));
        let reference = reference_type2_f64(
            &config,
            &points_f64(points),
            &interleaved_to_complex64(&coefficients),
        )
        .unwrap();
        let error = relative_l2(&actual, &reference);
        eprintln!(
            "NUFFT_3D_SCRATCH_REUSE kind=type2 phase={label} M={} relative_l2={error:.9e}",
            points.len() / DIMENSIONS
        );
        assert!(error <= FINUFFT_FLOAT_TOLERANCE_FACTOR_3D * config.eps());
    }
}

#[cfg(feature = "gpu-profiling")]
fn validate_large_256_cubed_type2(device: &wgpu::Device, queue: &wgpu::Queue) {
    let mode_shape = [256, 256, 256];
    let config = NufftConfig::new(mode_shape, 1.0e-6)
        .with_sign(NufftSign::Positive)
        .with_mode_order(ModeOrder::Fft);
    let plan = NufftPlan::type2_gpu(device, queue, config.clone()).unwrap();
    assert_eq!(plan.fine_grid_shape(), [512, 512, 512]);

    let diagnostics = plan.gpu_fft_diagnostics().unwrap();
    eprintln!(
        "NUFFT_3D_LARGE_ROUTE modes=256x256x256 fine=512x512x512 route={:?} stages={} workspace={:?}",
        diagnostics.route(),
        diagnostics.stages().len(),
        diagnostics.buffer_requirements(),
    );
    assert_eq!(diagnostics.route().route, "mixed-radix");
    assert_eq!(
        diagnostics.route().large_route_mode.as_deref(),
        Some("normal")
    );
    assert_eq!(
        diagnostics.route().execution_kind.as_deref(),
        Some("normal")
    );

    let mode_count = config.mode_count().unwrap();
    let mut coefficients = vec![0.0f32; mode_count * 2];
    let storage = 1usize + mode_shape[0] * (2usize + mode_shape[1] * 3usize);
    let coefficient = Complex64::new(0.75, -0.5);
    coefficients[2 * storage] = coefficient.re as f32;
    coefficients[2 * storage + 1] = coefficient.im as f32;
    let points = vec![-2.75f32, 0.5, 1.125, 0.0, 0.0, 0.0, 2.875, -2.625, 0.125];
    let actual =
        interleaved_to_complex64(&execute_type2(device, queue, &plan, &points, &coefficients));
    let expected = points
        .chunks_exact(DIMENSIONS)
        .map(|point| {
            let phase = f64::from(point[0]) + 2.0 * f64::from(point[1]) + 3.0 * f64::from(point[2]);
            rotate(coefficient, phase)
        })
        .collect::<Vec<_>>();
    let error = relative_l2(&actual, &expected);
    eprintln!(
        "NUFFT_3D_LARGE_ACCURACY modes=256x256x256 fine=512x512x512 M=3 relative_l2={error:.9e}"
    );
    assert!(error <= 2.0e-5);
}

struct PointClass {
    label: &'static str,
    points: Vec<f32>,
}

fn point_classes() -> Vec<PointClass> {
    vec![
        PointClass {
            label: "random",
            points: seeded_random_points(13, 0x3d7a_11c5),
        },
        PointClass {
            label: "clustered",
            points: (0..11)
                .flat_map(|index| {
                    [
                        0.25 + index as f32 * 1.0e-5,
                        -0.375 + index as f32 * 7.0e-6,
                        0.125 - index as f32 * 9.0e-6,
                    ]
                })
                .collect(),
        },
        PointClass {
            label: "boundary",
            points: vec![
                -PI as f32,
                -PI as f32,
                -PI as f32,
                PI as f32,
                PI as f32,
                PI as f32,
                -2.999 * PI as f32,
                2.999 * PI as f32,
                0.0,
                0.0,
                -2.999 * PI as f32,
                2.999 * PI as f32,
                PI as f32,
                -(PI as f32),
                0.5,
                -0.125,
                0.25,
                -0.75,
                2.875,
                -2.625,
                2.375,
            ],
        },
    ]
}

fn seeded_random_points(point_count: usize, mut state: u32) -> Vec<f32> {
    (0..point_count * DIMENSIONS)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let unit = (state >> 8) as f32 * (1.0 / 16_777_216.0);
            -PI as f32 + 2.0 * PI as f32 * unit
        })
        .collect()
}

fn test_values(count: usize, sign: NufftSign, order: ModeOrder) -> Vec<f32> {
    let sign_bias = if sign == NufftSign::Positive {
        0.17
    } else {
        -0.23
    };
    let order_bias = if order == ModeOrder::Centered {
        0.11
    } else {
        -0.19
    };
    (0..count)
        .flat_map(|index| {
            let x = index as f32 + 0.5;
            [
                (0.071 * x + sign_bias).cos() + 0.001 * order_bias,
                0.5 * (0.047 * x + order_bias).sin() - 0.001 * sign_bias,
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
    let points_buffer = storage_buffer_init(device, "wgpu_nufft.3d.test.points", points);
    let coefficients_buffer =
        storage_buffer_init(device, "wgpu_nufft.3d.test.coefficients", coefficients);
    let output_bytes = NufftPlan::required_type2_output_buffer_size_bytes(point_count).unwrap();
    let output = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_nufft.3d.test.output"),
        size: output_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_nufft.3d.test.readback"),
        size: output_bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("wgpu_nufft.3d.test.encoder"),
    });
    plan.encode_type2_gpu(
        device,
        &mut encoder,
        point_count,
        &points_buffer,
        &coefficients_buffer,
        &output,
    )
    .unwrap();
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
            timeout: Some(Duration::from_secs(240)),
        })
        .unwrap();
    receiver.recv().unwrap().unwrap();
    let mapped = slice.get_mapped_range();
    let values = bytemuck::cast_slice::<u8, f32>(&mapped).to_vec();
    drop(mapped);
    readback.unmap();
    values
}

fn storage_buffer_init(device: &wgpu::Device, label: &str, values: &[f32]) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(values),
        usage: wgpu::BufferUsages::STORAGE,
    })
}

fn points_f64(points: &[f32]) -> Vec<f64> {
    points.iter().map(|&value| f64::from(value)).collect()
}

fn interleaved_to_complex64(values: &[f32]) -> Vec<Complex64> {
    values
        .chunks_exact(2)
        .map(|value| Complex64::new(f64::from(value[0]), f64::from(value[1])))
        .collect()
}

fn relative_l2(actual: &[Complex64], reference: &[Complex64]) -> f64 {
    assert_eq!(actual.len(), reference.len());
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
    if norm == 0.0 {
        error
    } else {
        error / norm
    }
}

#[cfg(feature = "gpu-profiling")]
fn rotate(value: Complex64, phase: f64) -> Complex64 {
    let (sin, cos) = phase.sin_cos();
    Complex64::new(
        value.re * cos - value.im * sin,
        value.re * sin + value.im * cos,
    )
}
