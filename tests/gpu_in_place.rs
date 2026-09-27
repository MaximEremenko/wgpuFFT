#![cfg(not(target_arch = "wasm32"))]

//! In-place execution: every route transforms one buffer as it transforms
//! two, alone and paired with its inverse in one recorder, and plans that
//! copy their input first say so.

use std::sync::mpsc;

use wgpu_fft::{
    C2cRoute, FftConfig, FftDirection, FftError, FftPlan, FftRecorder, FftTuning, Normalization,
};

#[test]
fn in_place_execution_matches_out_of_place() {
    if std::env::var_os("WGPU_FFT_RUN_GPU_TESTS").is_none() {
        eprintln!("skipping GPU test; set WGPU_FFT_RUN_GPU_TESTS=1 to run it");
        return;
    }
    pollster::block_on(run_in_place_cases());
}

async fn run_in_place_cases() {
    let Some(context) = wgpu_fft::device::request_default_device().await else {
        eprintln!("skipping GPU test; no suitable wgpu adapter was found");
        return;
    };
    eprintln!("adapter: {:?}", context.adapter.get_info());
    let device = &context.device;
    let queue = &context.queue;

    let multi_pass = FftTuning::default()
        .with_fused_min_convolution_length(1 << 20)
        .with_direct_max_prime(0);
    let cases = [
        (
            "512x256 fused",
            FftConfig::new_nd([512, 256]),
            C2cRoute::MixedRadix,
        ),
        (
            "720x480 fused smooth",
            FftConfig::new_nd([720, 480]),
            C2cRoute::MixedRadix,
        ),
        (
            "64x64 small volume",
            FftConfig::new_nd([64, 64]),
            C2cRoute::MixedRadix,
        ),
        (
            "13 one Stockham pass",
            FftConfig::new(13),
            C2cRoute::MixedRadix,
        ),
        (
            "inverse 11 batch 100",
            FftConfig::inverse(11).with_batch(100),
            C2cRoute::MixedRadix,
        ),
        (
            "3x8192 Stockham, then fused",
            FftConfig::new_nd([3, 8192]),
            C2cRoute::MixedRadix,
        ),
        (
            "256x256x3 fused, then Stockham",
            FftConfig::new_nd([256, 256, 3]),
            C2cRoute::MixedRadix,
        ),
        (
            "65536 split passes",
            FftConfig::new(65536),
            C2cRoute::MixedRadix,
        ),
        ("4241 fused Rader", FftConfig::new(4241), C2cRoute::Rader),
        (
            "inverse 2003 batch 64",
            FftConfig::inverse(2003).with_batch(64),
            C2cRoute::Rader,
        ),
        (
            "47 batch 8 direct",
            FftConfig::new(47).with_batch(8),
            C2cRoute::Rader,
        ),
        (
            "17 batch 3 multi-pass Rader",
            FftConfig::new(17)
                .with_batch(3)
                .with_tuning(multi_pass.clone()),
            C2cRoute::Rader,
        ),
        (
            "1523 register Bluestein",
            FftConfig::new(1523),
            C2cRoute::Rader,
        ),
        (
            "65537 Bluestein",
            FftConfig::new(65537),
            C2cRoute::Bluestein,
        ),
        (
            "101 multi-pass Bluestein",
            FftConfig::new(101).with_tuning(multi_pass.clone().with_force_bluestein_axes([0])),
            C2cRoute::Bluestein,
        ),
        (
            "47x47 direct axes",
            FftConfig::new_nd([47, 47]),
            C2cRoute::AxisSequence,
        ),
        (
            "179x179 register Bluestein axes",
            FftConfig::new_nd([179, 179]),
            C2cRoute::AxisSequence,
        ),
        (
            "31x31x31 three prime axes",
            FftConfig::new_nd([31, 31, 31]),
            C2cRoute::AxisSequence,
        ),
        (
            "inverse 97x97x97",
            FftConfig::inverse_nd([97, 97, 97]),
            C2cRoute::AxisSequence,
        ),
        (
            "100x47 smooth and prime axes",
            FftConfig::new_nd([100, 47]),
            C2cRoute::AxisSequence,
        ),
        (
            "65536x17 split passes, then Rader",
            FftConfig::new_nd([65536, 17]),
            C2cRoute::AxisSequence,
        ),
        (
            "65536x3x17 split passes, Stockham, Rader",
            FftConfig::new_nd([65536, 3, 17]),
            C2cRoute::AxisSequence,
        ),
    ];
    for (label, config, route) in cases {
        let plan = FftPlan::c2c(device, queue, config.clone()).unwrap();
        assert_eq!(plan.route(), route, "{label}");
        assert!(plan.supports_in_place(), "{label}");
        let bytes = plan.required_buffer_size_bytes();
        let input = test_signal(bytes as usize / 4);

        let expected = execute_out_of_place(device, queue, &plan, &input, bytes);
        let buffer = io_buffer(device, queue, &input, bytes);
        let mut encoder = device.create_command_encoder(&Default::default());
        plan.execute_in_place(device, &mut encoder, &buffer)
            .unwrap();
        queue.submit([encoder.finish()]);
        let actual = read_back(device, queue, &buffer, bytes);
        let difference = relative_difference(&actual, &expected);
        eprintln!("IN_PLACE label={label:?} relative_difference={difference:.3e}");
        assert!(difference <= 1.0e-6, "{label}: {difference}");

        // The transform and its inverse in place in one shared pass return
        // the input.
        let opposite = match config.direction() {
            FftDirection::Forward => FftDirection::Inverse,
            FftDirection::Inverse => FftDirection::Forward,
        };
        let inverse = FftPlan::c2c(device, queue, config.with_direction(opposite)).unwrap();
        assert!(inverse.supports_in_place(), "{label}");
        let buffer = io_buffer(device, queue, &input, bytes);
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut recorder = FftRecorder::new(&mut encoder);
            plan.record_in_place(device, &mut recorder, &buffer)
                .unwrap();
            inverse
                .record_in_place(device, &mut recorder, &buffer)
                .unwrap();
        }
        queue.submit([encoder.finish()]);
        let roundtrip = read_back(device, queue, &buffer, bytes);
        let error = relative_difference(&roundtrip, &input);
        eprintln!("IN_PLACE_ROUNDTRIP label={label:?} relative_error={error:.3e}");
        // Bluestein over 65537 points round-trips within 3e-5.
        assert!(error <= 1.0e-4, "{label} roundtrip: {error}");
    }

    // R2C and C2R plans copy their input into a buffer they keep, which
    // needs COPY_SRC on the caller's buffer.
    let r2c = FftPlan::r2c(device, queue, FftConfig::new_nd([60, 64])).unwrap();
    let c2r = FftPlan::c2r(
        device,
        queue,
        FftConfig::inverse_nd([60, 64]).with_normalization(Normalization::Inverse),
    )
    .unwrap();
    for (label, plan) in [("r2c 60x64", &r2c), ("c2r 60x64", &c2r)] {
        assert!(!plan.supports_in_place(), "{label}");
        let input_bytes = plan.required_input_buffer_size_bytes();
        let output_bytes = plan.required_output_buffer_size_bytes();
        let bytes = input_bytes.max(output_bytes);
        let input = test_signal(input_bytes as usize / 4);

        let without_copy_src = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("wgpu_fft.test.in_place.without_copy_src"),
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        assert!(
            matches!(
                plan.execute_in_place(device, &mut encoder, &without_copy_src),
                Err(FftError::BufferViewMissingUsage { usage: "COPY_SRC" })
            ),
            "{label}"
        );
        drop(encoder);

        let input_buffer = io_buffer(device, queue, &input, input_bytes);
        let output_buffer = io_buffer(device, queue, &[], output_bytes);
        let mut encoder = device.create_command_encoder(&Default::default());
        plan.execute_checked(device, &mut encoder, &input_buffer, &output_buffer)
            .unwrap();
        queue.submit([encoder.finish()]);
        let expected = read_back(device, queue, &output_buffer, output_bytes);

        let buffer = io_buffer(device, queue, &input, bytes);
        // Twice, through the buffer the plan keeps for its input.
        for _ in 0..2 {
            queue.write_buffer(&buffer, 0, bytemuck::cast_slice(&input));
            let mut encoder = device.create_command_encoder(&Default::default());
            plan.execute_in_place(device, &mut encoder, &buffer)
                .unwrap();
            queue.submit([encoder.finish()]);
            let actual = read_back(device, queue, &buffer, output_bytes);
            assert_eq!(actual, expected, "{label}");
        }
    }

    #[cfg(windows)]
    std::mem::forget(context);
}

fn execute_out_of_place(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    plan: &FftPlan,
    input: &[f32],
    bytes: u64,
) -> Vec<f32> {
    let input_buffer = io_buffer(device, queue, input, bytes);
    let output_buffer = io_buffer(device, queue, &[], bytes);
    let mut encoder = device.create_command_encoder(&Default::default());
    plan.execute_checked(device, &mut encoder, &input_buffer, &output_buffer)
        .unwrap();
    queue.submit([encoder.finish()]);
    read_back(device, queue, &output_buffer, bytes)
}

fn io_buffer(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    contents: &[f32],
    bytes: u64,
) -> wgpu::Buffer {
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.in_place.buffer"),
        size: bytes,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    if !contents.is_empty() {
        queue.write_buffer(&buffer, 0, bytemuck::cast_slice(contents));
    }
    buffer
}

fn read_back(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    buffer: &wgpu::Buffer,
    bytes: u64,
) -> Vec<f32> {
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.in_place.readback"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(buffer, 0, &readback, 0, bytes);
    queue.submit([encoder.finish()]);
    let slice = readback.slice(..);
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
    readback.unmap();
    values
}

fn test_signal(len: usize) -> Vec<f32> {
    (0..len)
        .map(|index| {
            let x = index as f64 + 1.0;
            ((x * 0.017).sin() * 0.7 + (x * 0.031).cos() * 0.2) as f32
        })
        .collect()
}

/// Largest difference relative to the largest expected magnitude.
fn relative_difference(actual: &[f32], expected: &[f32]) -> f64 {
    assert_eq!(actual.len(), expected.len());
    let mut difference = 0.0f64;
    let mut magnitude = 0.0f64;
    for (&actual, &expected) in actual.iter().zip(expected) {
        difference = difference.max((f64::from(actual) - f64::from(expected)).abs());
        magnitude = magnitude.max(f64::from(expected).abs());
    }
    difference / magnitude.max(f64::MIN_POSITIVE)
}
