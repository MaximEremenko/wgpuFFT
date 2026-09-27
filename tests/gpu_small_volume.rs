#![cfg(not(target_arch = "wasm32"))]

//! Correctness and routing of small volumes transformed whole in one
//! workgroup: every axis in one kernel.

use std::sync::mpsc;

use wgpu_fft::math::{reference_c2c_nd_f64, Complex64};
use wgpu_fft::{C2cRoute, FftConfig, FftPlan, FftTuning, Normalization};

const SMALL_VOLUME_LABEL: &str = "small-volume-stage";

#[test]
fn small_volumes_match_the_f64_reference() {
    if std::env::var_os("WGPU_FFT_RUN_GPU_TESTS").is_none() {
        eprintln!("skipping GPU test; set WGPU_FFT_RUN_GPU_TESTS=1 to run it");
        return;
    }
    pollster::block_on(run_small_volume_cases());
}

async fn run_small_volume_cases() {
    let Some(context) = wgpu_fft::device::request_default_device().await else {
        eprintln!("skipping GPU test; no suitable wgpu adapter was found");
        return;
    };
    eprintln!("adapter: {:?}", context.adapter.get_info());
    let storage = u64::from(context.device.limits().max_compute_workgroup_storage_size);

    // (label, config, route, workgroup storage the kernel needs)
    let cases = [
        (
            "64x64",
            FftConfig::new_nd([64, 64]).with_normalization(Normalization::None),
            C2cRoute::MixedRadix,
            32_768u64,
        ),
        (
            "inverse 64x64 batch 3",
            FftConfig::inverse_nd([64, 64]).with_batch(3),
            C2cRoute::MixedRadix,
            32_768,
        ),
        (
            "16x16x16",
            FftConfig::new_nd([16, 16, 16]).with_normalization(Normalization::None),
            C2cRoute::MixedRadix,
            32_768,
        ),
        (
            "8x9x10 orthogonal",
            FftConfig::new_nd([8, 9, 10]).with_normalization(Normalization::Orthogonal),
            C2cRoute::MixedRadix,
            5_760,
        ),
        (
            "31x31",
            FftConfig::new_nd([31, 31]).with_normalization(Normalization::None),
            C2cRoute::AxisSequence,
            8_184,
        ),
        (
            "inverse 17x17 batch 5",
            FftConfig::inverse_nd([17, 17]).with_batch(5),
            C2cRoute::AxisSequence,
            2_584,
        ),
        (
            "2x101 axes in reverse",
            FftConfig::new_nd([2, 101])
                .with_axes([1, 0])
                .with_normalization(Normalization::None),
            C2cRoute::AxisSequence,
            2_424,
        ),
        (
            "51x8x3 (51 = 3 * 17)",
            FftConfig::new_nd([51, 8, 3]).with_normalization(Normalization::Forward),
            C2cRoute::AxisSequence,
            10_200,
        ),
    ];
    for (label, config, route, needed) in cases {
        let input = test_signal(config.total_complex_len().unwrap());
        let values = input
            .chunks_exact(2)
            .map(|pair| Complex64::new(f64::from(pair[0]), f64::from(pair[1])))
            .collect::<Vec<_>>();
        let expected = reference_c2c_nd_f64(&values, &config).unwrap();
        let (actual, plan) = execute_c2c(&context.device, &context.queue, config, &input);
        assert_eq!(plan.route(), route, "{label}");
        let kernels = kernel_labels(&plan);
        if needed <= storage {
            assert_eq!(kernels, [SMALL_VOLUME_LABEL], "{label}");
            assert_eq!(plan.workspace_size_bytes(), 0, "{label}");
        } else {
            assert!(
                !kernels.iter().any(|kernel| kernel == SMALL_VOLUME_LABEL),
                "{label}"
            );
        }
        let (max_relative, rms_relative) = relative_error_metrics(&actual, &expected);
        eprintln!(
            "SMALL_VOLUME_ACCURACY label={label:?} max_relative={max_relative:.9e} rms_relative={rms_relative:.9e}"
        );
        assert!(
            max_relative < 5.0e-7 && rms_relative < 5.0e-7,
            "{label}: max/rms relative error={max_relative}/{rms_relative}"
        );
    }

    // Larger volumes run their leading axes over slabs in one kernel, then
    // the remaining axes in place.
    for (label, config, slab_bytes) in [
        (
            "slabs 32x16 of 32x16x16",
            FftConfig::new_nd([32, 16, 16]).with_normalization(Normalization::None),
            4_096u64,
        ),
        (
            "slabs 64x64 of inverse 64x64x32 batch 2",
            FftConfig::inverse_nd([64, 64, 32]).with_batch(2),
            32_768,
        ),
        (
            "slabs 16x16 of 16x16x256",
            FftConfig::new_nd([16, 16, 256]).with_normalization(Normalization::Orthogonal),
            2_048,
        ),
        (
            "slabs 24x20 of 24x20x10",
            FftConfig::new_nd([24, 20, 10]).with_normalization(Normalization::None),
            3_840,
        ),
        (
            "slabs 8x8x8 of 8x8x8x16",
            FftConfig::new_nd([8, 8, 8, 16]).with_normalization(Normalization::None),
            4_096,
        ),
    ] {
        let input = test_signal(config.total_complex_len().unwrap());
        let values = input
            .chunks_exact(2)
            .map(|pair| Complex64::new(f64::from(pair[0]), f64::from(pair[1])))
            .collect::<Vec<_>>();
        let expected = reference_c2c_nd_f64(&values, &config).unwrap();
        let (actual, plan) = execute_c2c(&context.device, &context.queue, config, &input);
        assert_eq!(plan.route(), C2cRoute::MixedRadix, "{label}");
        let kernels = kernel_labels(&plan);
        if slab_bytes <= storage {
            assert_eq!(
                kernels.first().map(String::as_str),
                Some(SMALL_VOLUME_LABEL),
                "{label}"
            );
            assert_eq!(kernels.len(), 2, "{label}: {kernels:?}");
            assert_eq!(plan.workspace_size_bytes(), 0, "{label}");
        }
        let (max_relative, rms_relative) = relative_error_metrics(&actual, &expected);
        eprintln!(
            "SMALL_VOLUME_ACCURACY label={label:?} max_relative={max_relative:.9e} rms_relative={rms_relative:.9e}"
        );
        assert!(
            max_relative < 5.0e-7 && rms_relative < 5.0e-7,
            "{label}: max/rms relative error={max_relative}/{rms_relative}"
        );
    }

    // Direct axes too costly for one workgroup, tuning that asks for Rader
    // or per-axis kernels, and a partial set of axes keep one kernel per axis.
    for (label, config) in [
        ("47x47", FftConfig::new_nd([47, 47])),
        (
            "31x31 with direct kernels disabled",
            FftConfig::new_nd([31, 31]).with_tuning(FftTuning::default().with_direct_max_prime(0)),
        ),
        (
            "64x64 with small volumes disabled",
            FftConfig::new_nd([64, 64])
                .with_tuning(FftTuning::default().with_fuse_small_volumes(false)),
        ),
        (
            "64x64 axis 1 only",
            FftConfig::new_nd([64, 64]).with_axes([1]),
        ),
    ] {
        let input = test_signal(config.total_complex_len().unwrap());
        let (_, plan) = execute_c2c(&context.device, &context.queue, config, &input);
        assert!(
            !kernel_labels(&plan)
                .iter()
                .any(|kernel| kernel == SMALL_VOLUME_LABEL),
            "{label}"
        );
    }

    #[cfg(windows)]
    std::mem::forget(context);
}

fn kernel_labels(plan: &FftPlan) -> Vec<String> {
    plan.diagnostics()
        .stages()
        .iter()
        .filter(|stage| stage.kind == "kernel")
        .map(|stage| stage.label.clone())
        .collect()
}

fn execute_c2c(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    config: FftConfig,
    input: &[f32],
) -> (Vec<f32>, FftPlan) {
    let plan = FftPlan::c2c(device, queue, config).unwrap();
    let byte_len = std::mem::size_of_val(input) as u64;
    assert_eq!(plan.required_buffer_size_bytes(), byte_len);
    let input_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.small_volume.input"),
        size: byte_len,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.small_volume.output"),
        size: byte_len,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.small_volume.readback"),
        size: byte_len,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    queue.write_buffer(&input_buffer, 0, bytemuck::cast_slice(input));
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("wgpu_fft.test.small_volume.encoder"),
    });
    plan.execute_checked(device, &mut encoder, &input_buffer, &output_buffer)
        .unwrap();
    encoder.copy_buffer_to_buffer(&output_buffer, 0, &readback, 0, byte_len);
    queue.submit([encoder.finish()]);
    (read_f32(device, &readback), plan)
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

fn relative_error_metrics(actual: &[f32], expected: &[Complex64]) -> (f64, f64) {
    let mut max_error = 0.0f64;
    let mut max_reference = 0.0f64;
    let mut error_energy = 0.0f64;
    let mut reference_energy = 0.0f64;
    for (pair, expected) in actual.chunks_exact(2).zip(expected) {
        let dr = f64::from(pair[0]) - expected.re;
        let di = f64::from(pair[1]) - expected.im;
        let error2 = dr * dr + di * di;
        let reference2 = expected.re * expected.re + expected.im * expected.im;
        max_error = max_error.max(error2.sqrt());
        max_reference = max_reference.max(reference2.sqrt());
        error_energy += error2;
        reference_energy += reference2;
    }
    (
        max_error / max_reference.max(f64::MIN_POSITIVE),
        (error_energy / reference_energy.max(f64::MIN_POSITIVE)).sqrt(),
    )
}

fn read_f32(device: &wgpu::Device, buffer: &wgpu::Buffer) -> Vec<f32> {
    let slice = buffer.slice(..);
    let (sender, receiver) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        sender.send(result).expect("map result receiver is alive");
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("device polling should succeed");
    receiver
        .recv()
        .expect("map callback should send a result")
        .expect("readback buffer should map");
    let mapped = slice
        .get_mapped_range()
        .expect("readback buffer should be mapped");
    let values = bytemuck::cast_slice(&mapped).to_vec();
    drop(mapped);
    buffer.unmap();
    values
}
