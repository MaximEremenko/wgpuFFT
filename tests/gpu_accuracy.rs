//! Opt-in GPU accuracy measurements against an f64 host reference.

use std::sync::mpsc;

use wgpu_fft::math::{reference_c2c_nd_f64, Complex64};
use wgpu_fft::{FftConfig, FftPlan, Normalization};

#[test]
fn gpu_accuracy_against_f64_reference() {
    if std::env::var_os("WGPU_FFT_RUN_GPU_TESTS").is_none() {
        eprintln!("skipping GPU test; set WGPU_FFT_RUN_GPU_TESTS=1 to run it");
        return;
    }

    pollster::block_on(run_accuracy_cases());
}

async fn run_accuracy_cases() {
    let Some(context) = wgpu_fft::device::request_default_device().await else {
        eprintln!("skipping GPU test; no suitable wgpu adapter was found");
        return;
    };
    eprintln!("adapter: {:?}", context.adapter.get_info());

    for (label, config) in [
        (
            "pow2-4096",
            FftConfig::new(4096).with_normalization(Normalization::None),
        ),
        (
            "smooth-3000",
            FftConfig::new(3000).with_normalization(Normalization::None),
        ),
        (
            "rader-2999",
            FftConfig::new(2999).with_normalization(Normalization::None),
        ),
        (
            "bluestein-3256",
            FftConfig::new(3256).with_normalization(Normalization::None),
        ),
        (
            "batched-nd-12x25x2",
            FftConfig::new_nd([12, 25])
                .with_batch(2)
                .with_normalization(Normalization::None),
        ),
        (
            "smooth-3000-inverse",
            FftConfig::inverse(3000).with_normalization(Normalization::None),
        ),
        (
            "rader-2999-inverse",
            FftConfig::inverse(2999).with_normalization(Normalization::None),
        ),
        (
            "bluestein-3256-inverse",
            FftConfig::inverse(3256).with_normalization(Normalization::None),
        ),
    ] {
        let input = test_signal(config.total_complex_len().unwrap());
        let reference = reference_c2c_f64(&input, &config);
        let actual = execute_c2c(&context.device, &context.queue, config, &input);
        let metrics = accuracy_metrics(&actual, &reference);
        eprintln!(
            "ACCURACY label={label} max_relative={:.9e} rms_relative={:.9e} max_abs={:.9e} rms_abs={:.9e} reference_max={:.9e}",
            metrics.max_relative,
            metrics.rms_relative,
            metrics.max_abs,
            metrics.rms_abs,
            metrics.reference_max,
        );
        assert!(
            metrics.max_relative.is_finite() && metrics.max_relative < 5.0e-7,
            "{label}: max relative error is unexpectedly large: {}",
            metrics.max_relative
        );
        assert!(
            metrics.rms_relative.is_finite() && metrics.rms_relative < 5.0e-7,
            "{label}: RMS relative error is unexpectedly large: {}",
            metrics.rms_relative
        );
    }

    #[cfg(windows)]
    std::mem::forget(context);
}

fn test_signal(complex_len: usize) -> Vec<f32> {
    (0..complex_len)
        .flat_map(|index| {
            let x = index as f64 + 1.0;
            [
                ((x * 0.017).sin() * 0.7 + (x * 0.031).cos() * 0.2) as f32,
                ((x * 0.023).cos() * 0.5 - (x * 0.011).sin() * 0.3) as f32,
            ]
        })
        .collect()
}

fn reference_c2c_f64(input: &[f32], config: &FftConfig) -> Vec<Complex64> {
    let total = config.total_complex_len().unwrap();
    assert_eq!(input.len(), total * 2);
    let values = input
        .chunks_exact(2)
        .map(|pair| Complex64::new(f64::from(pair[0]), f64::from(pair[1])))
        .collect::<Vec<_>>();
    reference_c2c_nd_f64(&values, config).unwrap()
}

struct AccuracyMetrics {
    max_relative: f64,
    rms_relative: f64,
    max_abs: f64,
    rms_abs: f64,
    reference_max: f64,
}

fn accuracy_metrics(actual: &[f32], reference: &[Complex64]) -> AccuracyMetrics {
    assert_eq!(actual.len(), reference.len() * 2);
    let mut max_abs = 0.0f64;
    let mut error_energy = 0.0f64;
    let mut reference_energy = 0.0f64;
    let mut reference_max = 0.0f64;

    for (pair, expected) in actual.chunks_exact(2).zip(reference) {
        let dr = f64::from(pair[0]) - expected.re;
        let di = f64::from(pair[1]) - expected.im;
        let error = dr.hypot(di);
        let magnitude = expected.re.hypot(expected.im);
        max_abs = max_abs.max(error);
        reference_max = reference_max.max(magnitude);
        error_energy += error * error;
        reference_energy += magnitude * magnitude;
    }

    let count = reference.len() as f64;
    AccuracyMetrics {
        max_relative: max_abs / reference_max.max(f64::MIN_POSITIVE),
        rms_relative: (error_energy / reference_energy.max(f64::MIN_POSITIVE)).sqrt(),
        max_abs,
        rms_abs: (error_energy / count).sqrt(),
        reference_max,
    }
}

fn execute_c2c(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    config: FftConfig,
    input: &[f32],
) -> Vec<f32> {
    let plan = FftPlan::c2c(device, queue, config).unwrap();
    let byte_len = std::mem::size_of_val(input) as u64;
    let input_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.accuracy.input"),
        size: byte_len,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.accuracy.output"),
        size: byte_len,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.accuracy.readback"),
        size: byte_len,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    queue.write_buffer(&input_buffer, 0, bytemuck::cast_slice(input));

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("wgpu_fft.test.accuracy.encoder"),
    });
    plan.execute_checked(device, &mut encoder, &input_buffer, &output_buffer)
        .unwrap();
    encoder.copy_buffer_to_buffer(&output_buffer, 0, &readback, 0, byte_len);
    queue.submit([encoder.finish()]);

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
