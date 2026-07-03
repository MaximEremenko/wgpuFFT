#![cfg(not(target_arch = "wasm32"))]

//! Opt-in rank-generic (d >= 4) type-2 GPU validation against the direct f64
//! NDFT oracle. Adoption phase 1 of the ND NUFFT design: F32 precision,
//! sigma per config, both signs and mode orders, batching, and coordinates
//! exercising the full [-3*pi, 3*pi] folding interval.

use std::f64::consts::PI;
use std::sync::mpsc;

use wgpu::util::DeviceExt;
use wgpu_nufft::{reference_type2_f64, Complex64, ModeOrder, NufftConfig, NufftPlan, NufftSign};

#[test]
fn gpu_type2_nd_matches_oracle() {
    if std::env::var_os("WGPU_FFT_RUN_GPU_TESTS").is_none() {
        eprintln!("skipping GPU test; set WGPU_FFT_RUN_GPU_TESTS=1 to run it");
        return;
    }
    pollster::block_on(run_cases());
}

async fn run_cases() {
    let Some(context) = wgpu_fft::device::request_default_device().await else {
        panic!("WGPU_FFT_RUN_GPU_TESTS was set but no suitable adapter was found");
    };
    let info = context.adapter.get_info();
    eprintln!(
        "gpu_nufft_nd adapter: {} backend={:?} driver={}",
        info.name, info.backend, info.driver
    );

    // (label, n_modes, eps, sigma, sign, order, batch, tolerance_factor)
    #[allow(clippy::type_complexity)]
    let cases: [(&str, Vec<usize>, f64, f64, NufftSign, ModeOrder, usize, f64); 6] = [
        (
            "4d eps=1e-3 sigma=2 negative centered",
            vec![6, 5, 4, 4],
            1.0e-3,
            2.0,
            NufftSign::Negative,
            ModeOrder::Centered,
            1,
            8.0,
        ),
        (
            "4d eps=1e-3 sigma=2 positive fft-order",
            vec![6, 5, 4, 4],
            1.0e-3,
            2.0,
            NufftSign::Positive,
            ModeOrder::Fft,
            1,
            8.0,
        ),
        (
            "4d eps=1e-3 sigma=1.25 negative centered",
            vec![6, 5, 4, 4],
            1.0e-3,
            1.25,
            NufftSign::Negative,
            ModeOrder::Centered,
            1,
            8.0,
        ),
        (
            "4d eps=1e-6 sigma=2 negative centered",
            vec![8, 7, 6, 5],
            1.0e-6,
            2.0,
            NufftSign::Negative,
            ModeOrder::Centered,
            1,
            16.0,
        ),
        (
            "4d eps=1e-3 sigma=2 batch=3",
            vec![5, 4, 4, 3],
            1.0e-3,
            2.0,
            NufftSign::Negative,
            ModeOrder::Centered,
            3,
            8.0,
        ),
        (
            "5d eps=1e-3 sigma=2 negative centered",
            vec![4, 4, 3, 3, 3],
            1.0e-3,
            2.0,
            NufftSign::Negative,
            ModeOrder::Centered,
            1,
            8.0,
        ),
    ];

    for (label, n_modes, eps, sigma, sign, order, batch, tolerance_factor) in cases {
        run_case(
            &context,
            label,
            &n_modes,
            eps,
            sigma,
            sign,
            order,
            batch,
            tolerance_factor,
        )
        .await;
    }

    #[cfg(windows)]
    std::mem::forget(context);
}

#[allow(clippy::too_many_arguments)]
async fn run_case(
    context: &wgpu_fft::device::GpuContext,
    label: &str,
    n_modes: &[usize],
    eps: f64,
    sigma: f64,
    sign: NufftSign,
    order: ModeOrder,
    batch: usize,
    tolerance_factor: f64,
) {
    let dims = n_modes.len();
    let mode_count: usize = n_modes.iter().product();
    let config = NufftConfig::new(n_modes.to_vec(), eps)
        .with_sigma(sigma)
        .with_sign(sign)
        .with_mode_order(order)
        .with_batch(batch);

    // deterministic coordinates spanning the folding interval, incl. values
    // beyond one period to exercise the double fold
    let point_count = 37;
    let mut state = 0x1234_5678_9abc_def0u64;
    let mut next = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 11) as f64) / ((1u64 << 53) as f64)
    };
    let mut points = Vec::with_capacity(point_count * dims);
    for point in 0..point_count {
        for _ in 0..dims {
            let unit = next();
            let coordinate = if point == 0 {
                2.5 * PI // beyond one period: double-fold coverage
            } else if point == 1 {
                -2.75 * PI
            } else {
                (unit * 2.0 - 1.0) * PI
            };
            points.push(coordinate as f32);
        }
    }
    let coefficient_count = mode_count * batch;
    let mut coefficients = Vec::with_capacity(coefficient_count * 2);
    for _ in 0..coefficient_count {
        coefficients.push((next() * 2.0 - 1.0) as f32);
        coefficients.push((next() * 2.0 - 1.0) as f32);
    }

    let validation_scope = {
        let scope = context
            .device
            .push_error_scope(wgpu::ErrorFilter::Validation);
        let plan = NufftPlan::type2_gpu(&context.device, &context.queue, config.clone())
            .unwrap_or_else(|error| panic!("{label}: plan creation failed: {error}"));
        let actual = execute_type2(
            &context.device,
            &context.queue,
            &plan,
            point_count,
            batch,
            &points,
            &coefficients,
        );
        let reference = reference_type2_f64(
            &config,
            &points.iter().map(|&x| f64::from(x)).collect::<Vec<_>>(),
            &coefficients
                .chunks_exact(2)
                .map(|pair| Complex64::new(f64::from(pair[0]), f64::from(pair[1])))
                .collect::<Vec<_>>(),
        )
        .unwrap();
        let actual_complex = actual
            .chunks_exact(2)
            .map(|pair| Complex64::new(f64::from(pair[0]), f64::from(pair[1])))
            .collect::<Vec<_>>();
        let error = relative_l2(&actual_complex, &reference);
        eprintln!("NUFFT_ND_ACCURACY {label}: relative_l2={error:.9e}");
        assert!(
            error <= tolerance_factor * eps,
            "{label}: relative l2 {error} exceeds {} * {eps}",
            tolerance_factor
        );
        scope
    };
    if let Some(error) = validation_scope.pop().await {
        panic!("{label}: validation error: {error}");
    }
}

fn execute_type2(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    plan: &NufftPlan,
    point_count: usize,
    batch: usize,
    points: &[f32],
    coefficients: &[f32],
) -> Vec<f32> {
    let point_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("wgpu_nufft.test_nd.points"),
        contents: bytemuck::cast_slice(points),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let coefficient_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("wgpu_nufft.test_nd.coefficients"),
        contents: bytemuck::cast_slice(coefficients),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let output_bytes = (point_count * batch * 8) as u64;
    let output = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_nufft.test_nd.output"),
        size: output_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_nufft.test_nd.readback"),
        size: output_bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("wgpu_nufft.test_nd.encoder"),
    });
    plan.encode_type2_gpu_batch(
        device,
        &mut encoder,
        point_count,
        batch,
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

fn relative_l2(actual: &[Complex64], reference: &[Complex64]) -> f64 {
    assert_eq!(actual.len(), reference.len());
    let mut difference = 0.0f64;
    let mut norm = 0.0f64;
    for (lhs, rhs) in actual.iter().zip(reference) {
        let dre = lhs.re - rhs.re;
        let dim = lhs.im - rhs.im;
        difference += dre * dre + dim * dim;
        norm += rhs.re * rhs.re + rhs.im * rhs.im;
    }
    (difference / norm.max(f64::MIN_POSITIVE)).sqrt()
}
