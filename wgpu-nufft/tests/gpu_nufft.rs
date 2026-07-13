//! Opt-in 1D type-2 GPU validation against the direct f64 NDFT oracle.

use std::f64::consts::PI;
use std::sync::mpsc;

use wgpu::util::DeviceExt;
use wgpu_nufft::{
    reference_type2_f64, Complex64, ModeOrder, NufftConfig, NufftError, NufftPlan, NufftSign,
};

const FULL_VECTOR_TOLERANCE_FACTOR: f64 = 4.0;
// Pinned FINUFFT 704cbfee accepts 10x requested tolerance in its CPU float
// regression and 20x for the CUDA sigma=2 float matrix. Keep every small,
// deliberately hostile subset below the stricter factor of eight while the
// complete vector retains the Phase-A factor of four.
const ADVERSARIAL_TOLERANCE_FACTOR: f64 = 8.0;

#[test]
fn gpu_type2_matches_direct_oracle() {
    if std::env::var_os("WGPU_FFT_RUN_GPU_TESTS").is_none() {
        eprintln!("skipping GPU test; set WGPU_FFT_RUN_GPU_TESTS=1 to run it");
        return;
    }
    pollster::block_on(run_gpu_type2_cases());
}

async fn run_gpu_type2_cases() {
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

    validate_structured_gpu_errors(&context.device, &context.queue);

    #[cfg(windows)]
    std::mem::forget(context);
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

fn validate_structured_gpu_errors(device: &wgpu::Device, queue: &wgpu::Queue) {
    assert!(matches!(
        NufftPlan::type2_gpu(device, queue, NufftConfig::new([4, 4], 1.0e-6)),
        Err(NufftError::GpuDimensionsUnsupported { actual: 2, .. })
    ));
    let config = NufftConfig::new([8], 1.0e-6);
    let host_only = NufftPlan::type2(config.clone()).unwrap();
    let plan = NufftPlan::type2_gpu(device, queue, config).unwrap();
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
