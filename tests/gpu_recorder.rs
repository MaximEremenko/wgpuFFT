#![cfg(not(target_arch = "wasm32"))]

//! Opt-in GPU checks that executions recorded through one `FftRecorder`,
//! sharing a compute pass with each other and with the caller's own work,
//! produce exactly what separate executions do.

use std::mem::ManuallyDrop;
use std::sync::mpsc;

use wgpu_fft::{FftConfig, FftPlan, FftRecorder, LargePolicyLimits, Normalization};

#[test]
fn recorded_executions_match_separate_executions() {
    if std::env::var_os("WGPU_FFT_RUN_GPU_TESTS").is_none() {
        eprintln!("skipping GPU test; set WGPU_FFT_RUN_GPU_TESTS=1 to run it");
        return;
    }
    pollster::block_on(run_cases());
}

async fn run_cases() {
    let Some(context) = wgpu_fft::device::request_default_device().await else {
        eprintln!("skipping GPU test; no suitable wgpu adapter was found");
        return;
    };
    // Dropping a device can hang on Windows; leak it so failures stay fast.
    let context = ManuallyDrop::new(context);
    let device = &context.device;
    let queue = &context.queue;

    // C2C forward and inverse, with a 3D plan in between.
    let config = FftConfig::new_nd([64, 32]).with_normalization(Normalization::None);
    let forward = FftPlan::c2c(device, queue, config.clone()).unwrap();
    let inverse = FftPlan::c2c(device, queue, FftConfig::inverse_nd([64, 32])).unwrap();
    let cube = FftPlan::c2c(device, queue, FftConfig::new_nd([8, 8, 32])).unwrap();
    compare(
        device,
        queue,
        "c2c",
        &[&forward, &cube, &inverse],
        64 * 32 * 2,
    );

    // R2C then C2R through the packed spectrum.
    let real_config = FftConfig::new_nd([30, 12]).with_normalization(Normalization::None);
    let r2c = FftPlan::r2c(device, queue, real_config).unwrap();
    let c2r = FftPlan::c2r(device, queue, FftConfig::inverse_nd([30, 12])).unwrap();
    compare_real(device, queue, &r2c, &c2r, 30 * 12);

    // An out-of-core four-step plan copies between its kernels, so the
    // recorder must end and reopen its pass mid-execution.
    let windowed_config = FftConfig::new_nd([15, 14])
        .with_batch(3)
        .with_normalization(Normalization::None);
    let windowed = FftPlan::c2c_with_large_policy_limits_for_testing(
        device,
        queue,
        windowed_config.clone(),
        LargePolicyLimits {
            max_storage_buffer_binding_size: 256,
            max_buffer_size: windowed_config.required_buffer_size_bytes().unwrap(),
        },
    )
    .unwrap();
    assert!(
        windowed
            .diagnostics()
            .stages()
            .iter()
            .any(|stage| stage.kind == "copy" || stage.kind == "stripe-transpose"),
        "the windowed plan should need encoder-level work between kernels"
    );
    compare(
        device,
        queue,
        "windowed",
        &[&windowed, &windowed],
        15 * 14 * 3 * 2,
    );

    // A caller kernel in the shared pass and a caller copy between executions.
    compare_caller_work(device, queue, &forward, &inverse, 64 * 32 * 2);
}

/// Runs `plans` in sequence, ping-ponging two buffers, once with separate
/// executions and once through one recorder, and requires identical output.
fn compare(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &str,
    plans: &[&FftPlan],
    floats: usize,
) {
    let input = test_signal(floats);
    let run = |shared: bool| {
        let a = storage(device, floats);
        let b = storage(device, floats);
        queue.write_buffer(&a, 0, bytemuck::cast_slice(&input));
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        let mut buffers = [&a, &b];
        if shared {
            let mut recorder = FftRecorder::new(&mut encoder);
            for plan in plans {
                plan.record(device, &mut recorder, buffers[0], buffers[1])
                    .unwrap();
                buffers.swap(0, 1);
            }
        } else {
            for plan in plans {
                plan.execute_checked(device, &mut encoder, buffers[0], buffers[1])
                    .unwrap();
                buffers.swap(0, 1);
            }
        }
        read_back(device, queue, encoder, buffers[0], floats)
    };
    let separate = run(false);
    let shared = run(true);
    assert!(
        separate.iter().all(|value| value.is_finite()),
        "{label}: separate executions produced non-finite values"
    );
    assert_eq!(separate, shared, "{label}: recorded output differs");
}

fn compare_real(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    r2c: &FftPlan,
    c2r: &FftPlan,
    real_len: usize,
) {
    let input = test_signal(real_len);
    let spectrum_floats = (r2c.required_output_buffer_size_bytes() / 4) as usize;
    let run = |shared: bool| {
        let real = storage(device, real_len);
        let spectrum = storage(device, spectrum_floats);
        let back = storage(device, real_len);
        queue.write_buffer(&real, 0, bytemuck::cast_slice(&input));
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        if shared {
            let mut recorder = FftRecorder::new(&mut encoder);
            r2c.record(device, &mut recorder, &real, &spectrum).unwrap();
            c2r.record(device, &mut recorder, &spectrum, &back).unwrap();
        } else {
            r2c.execute_checked(device, &mut encoder, &real, &spectrum)
                .unwrap();
            c2r.execute_checked(device, &mut encoder, &spectrum, &back)
                .unwrap();
        }
        read_back(device, queue, encoder, &back, real_len)
    };
    let separate = run(false);
    let shared = run(true);
    // C2R with Inverse normalization undoes R2C without normalization.
    for (value, original) in separate.iter().zip(&input) {
        assert!((value - original).abs() < 1e-4, "r2c/c2r round trip");
    }
    assert_eq!(separate, shared, "real: recorded output differs");
}

const SCALE_WGSL: &str = "
@group(0) @binding(0) var<storage, read_write> data: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x < arrayLength(&data)) {
        data[id.x] = data[id.x] * 0.25 + 1.0;
    }
}
";

/// Runs a forward transform, a caller kernel over its output, a caller copy,
/// and the inverse, once with the caller's work in its own pass and once
/// through the recorder's `compute_pass` and `encoder`, and requires
/// identical output.
fn compare_caller_work(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    forward: &FftPlan,
    inverse: &FftPlan,
    floats: usize,
) {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("wgpu_fft.test.recorder.scale"),
        source: wgpu::ShaderSource::Wgsl(SCALE_WGSL.into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("wgpu_fft.test.recorder.scale"),
        layout: None,
        module: &shader,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });
    let input = test_signal(floats);
    let bytes = (floats * 4) as u64;
    let run = |shared: bool| {
        let a = storage(device, floats);
        let b = storage(device, floats);
        let c = storage(device, floats);
        queue.write_buffer(&a, 0, bytemuck::cast_slice(&input));
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("wgpu_fft.test.recorder.scale"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: b.as_entire_binding(),
            }],
        });
        let scale = |pass: &mut wgpu::ComputePass<'_>| {
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups((floats as u32).div_ceil(64), 1, 1);
        };
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        if shared {
            let mut recorder = FftRecorder::new(&mut encoder);
            forward.record(device, &mut recorder, &a, &b).unwrap();
            scale(recorder.compute_pass());
            recorder
                .encoder()
                .copy_buffer_to_buffer(&b, 0, &c, 0, bytes);
            inverse.record(device, &mut recorder, &c, &a).unwrap();
        } else {
            forward
                .execute_checked(device, &mut encoder, &a, &b)
                .unwrap();
            scale(&mut encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default()));
            encoder.copy_buffer_to_buffer(&b, 0, &c, 0, bytes);
            inverse
                .execute_checked(device, &mut encoder, &c, &a)
                .unwrap();
        }
        read_back(device, queue, encoder, &a, floats)
    };
    let separate = run(false);
    let shared = run(true);
    // The kernel adds 1 + i to every bin, which the normalized inverse turns
    // into 1 + i at the first element.
    for (index, (value, original)) in separate.iter().zip(&input).enumerate() {
        let expected = original * 0.25 + if index < 2 { 1.0 } else { 0.0 };
        assert!(
            (value - expected).abs() < 1e-4,
            "caller work: element {index} is {value}, expected {expected}"
        );
    }
    assert_eq!(separate, shared, "caller work: recorded output differs");
}

fn storage(device: &wgpu::Device, floats: usize) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.recorder.buffer"),
        size: (floats * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

fn read_back(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    mut encoder: wgpu::CommandEncoder,
    buffer: &wgpu::Buffer,
    floats: usize,
) -> Vec<f32> {
    let bytes = (floats * 4) as u64;
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.recorder.readback"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    encoder.copy_buffer_to_buffer(buffer, 0, &readback, 0, bytes);
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
    let values = bytemuck::cast_slice(&slice.get_mapped_range().unwrap()).to_vec();
    readback.unmap();
    values
}

fn test_signal(len: usize) -> Vec<f32> {
    (0..len)
        .map(|index| {
            let x = index as f32 + 1.0;
            (x * 0.017).sin() * 0.7 + (x * 0.031).cos() * 0.2
        })
        .collect()
}
