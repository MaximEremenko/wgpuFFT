#![cfg(not(target_arch = "wasm32"))]

//! Opt-in rank-generic (d >= 4) GPU validation against the direct f64 NDFT
//! oracle: F32/F64/Df64 precisions, sigma per config, both signs and mode
//! orders, batching, and coordinates exercising the full FINUFFT
//! [-3*pi, 3*pi] folding interval. The F64 cases run only when the adapter
//! exposes SHADER_F64; Df64 needs no device feature.

use std::f64::consts::PI;
use std::sync::mpsc;

use wgpu::util::DeviceExt;
use wgpu_nufft::{
    reference_type1_f64, reference_type2_f64, Complex64, ComplexDoubleFloat, DoubleFloat,
    FftPrecision, ModeOrder, NufftConfig, NufftPlan, NufftSign,
};

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

    adjoint_case(&context).await;

    high_precision_case(
        &context,
        "4d eps=1e-8 df64",
        &[10, 9, 8, 8],
        1.0e-8,
        FftPrecision::Df64,
        48.0,
    )
    .await;
    if context
        .device
        .features()
        .contains(wgpu::Features::SHADER_F64)
    {
        high_precision_case(
            &context,
            "4d eps=1e-8 f64",
            &[10, 9, 8, 8],
            1.0e-8,
            FftPrecision::F64,
            40.0,
        )
        .await;
    } else {
        eprintln!("skipping ND F64 case; the adapter lacks SHADER_F64");
    }

    #[cfg(windows)]
    std::mem::forget(context);
}

/// Runs both transform types in a high-precision configuration against the
/// f64 oracle. Sub-f32 relative error is the pass criterion: an f32 pipeline
/// cannot reach these thresholds, so agreement proves the F64/Df64 shaders
/// carry the extra precision end to end.
async fn high_precision_case(
    context: &wgpu_fft::device::GpuContext,
    label: &str,
    n_modes: &[usize],
    eps: f64,
    precision: FftPrecision,
    tolerance_factor: f64,
) {
    let dims = n_modes.len();
    let mode_count: usize = n_modes.iter().product();
    let config = NufftConfig::new(n_modes.to_vec(), eps)
        .with_sign(NufftSign::Negative)
        .with_precision(precision);

    let point_count = 37;
    let mut state = 0x0dd5_ba11_5eed_f00du64;
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
                2.5 * PI
            } else if point == 1 {
                -2.75 * PI
            } else {
                (unit * 2.0 - 1.0) * PI
            };
            points.push(coordinate);
        }
    }
    let coefficients: Vec<Complex64> = (0..mode_count)
        .map(|_| Complex64::new(next() * 2.0 - 1.0, next() * 2.0 - 1.0))
        .collect();
    let strengths: Vec<Complex64> = (0..point_count)
        .map(|_| Complex64::new(next() * 2.0 - 1.0, next() * 2.0 - 1.0))
        .collect();

    let (point_bytes, coefficient_bytes, strength_bytes) = match precision {
        FftPrecision::F64 => (
            bytemuck::cast_slice::<f64, u8>(&points).to_vec(),
            bytemuck::cast_slice::<Complex64, u8>(&coefficients).to_vec(),
            bytemuck::cast_slice::<Complex64, u8>(&strengths).to_vec(),
        ),
        FftPrecision::Df64 => {
            let packed_points: Vec<DoubleFloat> =
                points.iter().map(|&x| DoubleFloat::from_f64(x)).collect();
            let packed_coefficients: Vec<ComplexDoubleFloat> = coefficients
                .iter()
                .map(|value| ComplexDoubleFloat::from_f64(value.re, value.im))
                .collect();
            let packed_strengths: Vec<ComplexDoubleFloat> = strengths
                .iter()
                .map(|value| ComplexDoubleFloat::from_f64(value.re, value.im))
                .collect();
            (
                bytemuck::cast_slice::<DoubleFloat, u8>(&packed_points).to_vec(),
                bytemuck::cast_slice::<ComplexDoubleFloat, u8>(&packed_coefficients).to_vec(),
                bytemuck::cast_slice::<ComplexDoubleFloat, u8>(&packed_strengths).to_vec(),
            )
        }
        FftPrecision::F32 => unreachable!("high-precision ND cases cover F64/Df64 only"),
    };

    let scope = context
        .device
        .push_error_scope(wgpu::ErrorFilter::Validation);
    let plan_t2 = NufftPlan::type2_gpu(&context.device, &context.queue, config.clone())
        .unwrap_or_else(|error| panic!("{label}: type-2 plan creation failed: {error}"));
    let t2_bytes = execute_type2_bytes(
        &context.device,
        &context.queue,
        &plan_t2,
        point_count,
        1,
        &point_bytes,
        &coefficient_bytes,
        16,
    );
    let plan_t1 = NufftPlan::type1_gpu(&context.device, &context.queue, config.clone())
        .unwrap_or_else(|error| panic!("{label}: type-1 plan creation failed: {error}"));
    let t1_bytes = execute_type1_bytes(
        &context.device,
        &context.queue,
        &plan_t1,
        point_count,
        1,
        mode_count,
        &point_bytes,
        &strength_bytes,
        16,
    );
    if let Some(error) = scope.pop().await {
        panic!("{label}: validation error: {error}");
    }

    let unpack = |bytes: &[u8]| -> Vec<Complex64> {
        match precision {
            FftPrecision::F64 => bytemuck::cast_slice::<u8, Complex64>(bytes).to_vec(),
            FftPrecision::Df64 => bytemuck::cast_slice::<u8, ComplexDoubleFloat>(bytes)
                .iter()
                .map(|value| Complex64::new(value.re().to_f64(), value.im().to_f64()))
                .collect(),
            FftPrecision::F32 => unreachable!(),
        }
    };
    let t2_actual = unpack(&t2_bytes);
    let t2_reference = reference_type2_f64(&config, &points, &coefficients).unwrap();
    let t2_error = relative_l2(&t2_actual, &t2_reference);
    eprintln!("NUFFT_ND_ACCURACY {label} type-2: relative_l2={t2_error:.9e}");
    assert!(
        t2_error <= tolerance_factor * eps,
        "{label}: type-2 relative l2 {t2_error} exceeds {tolerance_factor} * {eps}"
    );
    let t1_actual = unpack(&t1_bytes);
    let t1_reference = reference_type1_f64(&config, &points, &strengths).unwrap();
    let t1_error = relative_l2(&t1_actual, &t1_reference);
    eprintln!("NUFFT_ND_ACCURACY {label} type-1: relative_l2={t1_error:.9e}");
    assert!(
        t1_error <= tolerance_factor * eps,
        "{label}: type-1 relative l2 {t1_error} exceeds {tolerance_factor} * {eps}"
    );
}

/// <y, A x> == <A^H y, x>: type-2 with one sign is adjoint to type-1 with the
/// opposite sign under complex inner products.
async fn adjoint_case(context: &wgpu_fft::device::GpuContext) {
    let n_modes = vec![6usize, 5, 4, 4];
    let eps = 1.0e-6;
    let config_t2 = NufftConfig::new(n_modes.clone(), eps).with_sign(NufftSign::Negative);
    let config_t1 = NufftConfig::new(n_modes.clone(), eps).with_sign(NufftSign::Positive);
    let mode_count: usize = n_modes.iter().product();
    let dims = n_modes.len();
    let point_count = 29;

    let mut state = 0xfeed_beef_dead_c0deu64;
    let mut next = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 11) as f64) / ((1u64 << 53) as f64)
    };
    let points: Vec<f32> = (0..point_count * dims)
        .map(|_| ((next() * 2.0 - 1.0) * PI) as f32)
        .collect();
    let fhat: Vec<f32> = (0..mode_count * 2)
        .map(|_| (next() * 2.0 - 1.0) as f32)
        .collect();
    let y: Vec<f32> = (0..point_count * 2)
        .map(|_| (next() * 2.0 - 1.0) as f32)
        .collect();

    let scope = context
        .device
        .push_error_scope(wgpu::ErrorFilter::Validation);
    let plan_t2 = NufftPlan::type2_gpu(&context.device, &context.queue, config_t2).unwrap();
    let plan_t1 = NufftPlan::type1_gpu(&context.device, &context.queue, config_t1).unwrap();
    let t2_out = execute_type2(
        &context.device,
        &context.queue,
        &plan_t2,
        point_count,
        1,
        &points,
        &fhat,
    );
    let t1_out = execute_type1(
        &context.device,
        &context.queue,
        &plan_t1,
        point_count,
        1,
        mode_count,
        &points,
        &y,
    );
    if let Some(error) = scope.pop().await {
        panic!("adjoint case: validation error: {error}");
    }

    let vdot = |a: &[f32], b: &[f32]| -> (f64, f64) {
        let mut re = 0.0f64;
        let mut im = 0.0f64;
        for (pa, pb) in a.chunks_exact(2).zip(b.chunks_exact(2)) {
            let (ar, ai) = (f64::from(pa[0]), -f64::from(pa[1]));
            let (br, bi) = (f64::from(pb[0]), f64::from(pb[1]));
            re += ar * br - ai * bi;
            im += ar * bi + ai * br;
        }
        (re, im)
    };
    let lhs = vdot(&y, &t2_out);
    let rhs = vdot(&t1_out, &fhat);
    let difference = ((lhs.0 - rhs.0).powi(2) + (lhs.1 - rhs.1).powi(2)).sqrt();
    let scale = (lhs.0 * lhs.0 + lhs.1 * lhs.1)
        .sqrt()
        .max(f64::MIN_POSITIVE);
    let residual = difference / scale;
    eprintln!("NUFFT_ND_ADJOINT 4d: residual={residual:.9e}");
    assert!(
        residual <= 2.0e-5,
        "4d adjoint residual {residual} exceeds 2.0e-5"
    );
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
        eprintln!("NUFFT_ND_ACCURACY {label} type-2: relative_l2={error:.9e}");
        assert!(
            error <= tolerance_factor * eps,
            "{label}: type-2 relative l2 {error} exceeds {} * {eps}",
            tolerance_factor
        );

        // type-1 on the same points: strengths are transform-major complex
        let strengths: Vec<f32> = (0..point_count * batch * 2)
            .map(|index| ((index as f64 * 0.37).sin() * 0.9) as f32)
            .collect();
        let plan_t1 = NufftPlan::type1_gpu(&context.device, &context.queue, config.clone())
            .unwrap_or_else(|error| panic!("{label}: type-1 plan creation failed: {error}"));
        let t1_actual = execute_type1(
            &context.device,
            &context.queue,
            &plan_t1,
            point_count,
            batch,
            mode_count,
            &points,
            &strengths,
        );
        let t1_reference = reference_type1_f64(
            &config,
            &points.iter().map(|&x| f64::from(x)).collect::<Vec<_>>(),
            &strengths
                .chunks_exact(2)
                .map(|pair| Complex64::new(f64::from(pair[0]), f64::from(pair[1])))
                .collect::<Vec<_>>(),
        )
        .unwrap();
        let t1_actual_complex = t1_actual
            .chunks_exact(2)
            .map(|pair| Complex64::new(f64::from(pair[0]), f64::from(pair[1])))
            .collect::<Vec<_>>();
        let t1_error = relative_l2(&t1_actual_complex, &t1_reference);
        eprintln!("NUFFT_ND_ACCURACY {label} type-1: relative_l2={t1_error:.9e}");
        assert!(
            t1_error <= tolerance_factor * eps,
            "{label}: type-1 relative l2 {t1_error} exceeds {} * {eps}",
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

#[allow(clippy::too_many_arguments)]
fn execute_type1(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    plan: &NufftPlan,
    point_count: usize,
    batch: usize,
    mode_count: usize,
    points: &[f32],
    strengths: &[f32],
) -> Vec<f32> {
    let point_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("wgpu_nufft.test_nd.t1_points"),
        contents: bytemuck::cast_slice(points),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let strength_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("wgpu_nufft.test_nd.t1_strengths"),
        contents: bytemuck::cast_slice(strengths),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let output_bytes = (mode_count * batch * 8) as u64;
    let output = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_nufft.test_nd.t1_output"),
        size: output_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_nufft.test_nd.t1_readback"),
        size: output_bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("wgpu_nufft.test_nd.t1_encoder"),
    });
    plan.encode_type1_gpu_batch(
        device,
        &mut encoder,
        point_count,
        batch,
        &point_buffer,
        &strength_buffer,
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

#[allow(clippy::too_many_arguments)]
fn execute_type2_bytes(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    plan: &NufftPlan,
    point_count: usize,
    batch: usize,
    points: &[u8],
    coefficients: &[u8],
    complex_bytes: usize,
) -> Vec<u8> {
    let point_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("wgpu_nufft.test_nd.hp_points"),
        contents: points,
        usage: wgpu::BufferUsages::STORAGE,
    });
    let coefficient_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("wgpu_nufft.test_nd.hp_coefficients"),
        contents: coefficients,
        usage: wgpu::BufferUsages::STORAGE,
    });
    let output_bytes = (point_count * batch * complex_bytes) as u64;
    let output = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_nufft.test_nd.hp_output"),
        size: output_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_nufft.test_nd.hp_readback"),
        size: output_bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("wgpu_nufft.test_nd.hp_encoder"),
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
    let result = mapped.to_vec();
    drop(mapped);
    readback.unmap();
    result
}

#[allow(clippy::too_many_arguments)]
fn execute_type1_bytes(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    plan: &NufftPlan,
    point_count: usize,
    batch: usize,
    mode_count: usize,
    points: &[u8],
    strengths: &[u8],
    complex_bytes: usize,
) -> Vec<u8> {
    let point_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("wgpu_nufft.test_nd.hp_t1_points"),
        contents: points,
        usage: wgpu::BufferUsages::STORAGE,
    });
    let strength_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("wgpu_nufft.test_nd.hp_t1_strengths"),
        contents: strengths,
        usage: wgpu::BufferUsages::STORAGE,
    });
    let output_bytes = (mode_count * batch * complex_bytes) as u64;
    let output = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_nufft.test_nd.hp_t1_output"),
        size: output_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_nufft.test_nd.hp_t1_readback"),
        size: output_bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("wgpu_nufft.test_nd.hp_t1_encoder"),
    });
    plan.encode_type1_gpu_batch(
        device,
        &mut encoder,
        point_count,
        batch,
        &point_buffer,
        &strength_buffer,
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
    let result = mapped.to_vec();
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
