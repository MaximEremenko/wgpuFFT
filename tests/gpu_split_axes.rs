#![cfg(not(target_arch = "wasm32"))]

//! Opt-in GPU checks for axes too long for one fused workgroup, which run as
//! two fused passes (`N = N1 * N2`), against the f64 CPU backend.

use std::mem::ManuallyDrop;
use std::sync::mpsc;

use wgpu_fft::{CpuFftPlan, FftConfig, FftPlan, FftPrecision, Normalization};

const FUSED_POW2_LABEL: &str = "fused-pow2-workgroup-stage";
const FUSED_SMOOTH_LABEL: &str = "fused-smooth-workgroup-stage";
const STOCKHAM_LABEL: &str = "mixed-radix-stockham-stage";

#[test]
fn long_axes_split_into_two_fused_passes() {
    if std::env::var_os("WGPU_FFT_RUN_GPU_TESTS").is_none() {
        eprintln!("skipping GPU test; set WGPU_FFT_RUN_GPU_TESTS=1 to run it");
        return;
    }

    pollster::block_on(run_split_cases());
}

async fn run_split_cases() {
    let Some(context) = wgpu_fft::device::request_default_device().await else {
        eprintln!("skipping GPU test; no suitable wgpu adapter was found");
        return;
    };
    // Dropping a device can hang on Windows; leak it so failures stay fast.
    let context = ManuallyDrop::new(context);
    let device = &context.device;
    let queue = &context.queue;

    for (label, config) in [
        (
            "n8192",
            FftConfig::new(8192).with_normalization(Normalization::None),
        ),
        ("n12288-inverse", FftConfig::inverse(12288)),
        (
            "n16384-batch2",
            FftConfig::new(16384)
                .with_batch(2)
                .with_normalization(Normalization::Forward),
        ),
        (
            "8192x2-axis0-batch2",
            FftConfig::new_nd([8192, 2])
                .with_axes([0])
                .with_batch(2)
                .with_normalization(Normalization::None),
        ),
        (
            "3x8192-axis1",
            FftConfig::new_nd([3, 8192])
                .with_axes([1])
                .with_normalization(Normalization::None),
        ),
        (
            "3x10000x2-axis1",
            FftConfig::new_nd([3, 10000, 2])
                .with_axes([1])
                .with_normalization(Normalization::Orthogonal),
        ),
        ("8192x4-inverse", FftConfig::inverse_nd([8192, 4])),
    ] {
        let total = config.total_complex_len().unwrap();
        let input = test_signal(total);
        let reference = cpu_reference(&input, &config);

        let (split, split_plan) = execute_c2c(device, queue, config.clone(), &input);
        let kernels = kernel_labels(&split_plan);
        assert!(
            kernels
                .iter()
                .all(|kernel| kernel == FUSED_POW2_LABEL || kernel == FUSED_SMOOTH_LABEL),
            "{label}: split plans run only fused kernels, got {kernels:?}"
        );
        let (max_relative, rms_relative) = relative_errors(&split, &reference);
        eprintln!(
            "SPLIT label={label} max_relative={max_relative:.3e} rms_relative={rms_relative:.3e}"
        );
        assert!(
            max_relative < 2.0e-6 && rms_relative < 5.0e-7,
            "{label}: split errors max {max_relative:.3e}, rms {rms_relative:.3e}"
        );

        // The unsplit Stockham route must agree to the same accuracy.
        let unsplit = config.tuning().clone().with_split_long_axes(false);
        let (stockham, stockham_plan) =
            execute_c2c(device, queue, config.clone().with_tuning(unsplit), &input);
        assert!(
            kernel_labels(&stockham_plan)
                .iter()
                .any(|kernel| kernel == STOCKHAM_LABEL),
            "{label}: disabling the split keeps Stockham stages"
        );
        let (max_relative, rms_relative) = relative_errors(&stockham, &reference);
        assert!(
            max_relative < 2.0e-6 && rms_relative < 5.0e-7,
            "{label}: Stockham errors max {max_relative:.3e}, rms {rms_relative:.3e}"
        );
    }
}

fn kernel_labels(plan: &FftPlan) -> Vec<String> {
    plan.diagnostics()
        .stages()
        .iter()
        .filter(|stage| stage.kind == "kernel")
        .map(|stage| stage.label.to_string())
        .collect()
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

fn cpu_reference(input: &[f32], config: &FftConfig) -> Vec<f64> {
    let plan = CpuFftPlan::c2c(config.clone().with_precision(FftPrecision::F64)).unwrap();
    let input = input.iter().copied().map(f64::from).collect::<Vec<_>>();
    let mut output = vec![0.0f64; plan.required_output_len()];
    plan.execute_f64(&input, &mut output).unwrap();
    output
}

fn relative_errors(actual: &[f32], reference: &[f64]) -> (f64, f64) {
    assert_eq!(actual.len(), reference.len());
    let (mut max_error, mut peak, mut error_energy, mut energy) = (0.0f64, 0.0f64, 0.0, 0.0);
    for (pair, expected) in actual.chunks_exact(2).zip(reference.chunks_exact(2)) {
        let error = (f64::from(pair[0]) - expected[0]).hypot(f64::from(pair[1]) - expected[1]);
        let magnitude = expected[0].hypot(expected[1]);
        max_error = max_error.max(error);
        peak = peak.max(magnitude);
        error_energy += error * error;
        energy += magnitude * magnitude;
    }
    (max_error / peak, (error_energy / energy).sqrt())
}

fn execute_c2c(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    config: FftConfig,
    input: &[f32],
) -> (Vec<f32>, FftPlan) {
    let plan = FftPlan::c2c(device, queue, config).unwrap();
    let byte_len = std::mem::size_of_val(input) as u64;
    let buffer = |label: &str, usage: wgpu::BufferUsages| {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: byte_len,
            usage,
            mapped_at_creation: false,
        })
    };
    let input_buffer = buffer(
        "wgpu_fft.test.split.input",
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    );
    let output_buffer = buffer(
        "wgpu_fft.test.split.output",
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    );
    let readback = buffer(
        "wgpu_fft.test.split.readback",
        wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
    );
    queue.write_buffer(&input_buffer, 0, bytemuck::cast_slice(input));

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    plan.execute_checked(device, &mut encoder, &input_buffer, &output_buffer)
        .unwrap();
    encoder.copy_buffer_to_buffer(&output_buffer, 0, &readback, 0, byte_len);
    queue.submit([encoder.finish()]);

    let slice = readback.slice(..);
    let (sender, receiver) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        sender.send(result).unwrap();
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("device polling should succeed");
    receiver.recv().unwrap().unwrap();
    let output = bytemuck::cast_slice(&slice.get_mapped_range().unwrap()).to_vec();
    readback.unmap();
    (output, plan)
}
