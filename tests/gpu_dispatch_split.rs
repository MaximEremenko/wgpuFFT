#![cfg(not(target_arch = "wasm32"))]

//! Regression coverage for dispatches that exceed
//! `max_compute_workgroups_per_dimension` (65535 on most devices). Before the
//! 3D dispatch-grid split these cases issued invalid `dispatch_workgroups`
//! calls once a pass covered more than `limit * WORKGROUP_SIZE` work items,
//! and fused one-workgroup-per-line kernels overflowed at just `limit + 1`
//! lines.

use std::sync::mpsc;

use wgpu_fft::{C2cRoute, FftConfig, FftPlan};

const WORKGROUP_SIZE: u64 = 64;

#[test]
fn oversized_dispatches_split_across_grid_dimensions() {
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
    eprintln!("adapter: {:?}", context.adapter.get_info());

    // The fused N=2 kernel dispatches one workgroup per line. 65_537 lines
    // produce one padded workgroup in the balanced 32_769 x 2 grid.
    roundtrip_c2c(
        &context,
        &[2],
        65_537,
        C2cRoute::MixedRadix,
        65_537,
        Some("fused-pow2-workgroup-stage"),
    );
    // N=67 uses a small fitting Rader convolution (M=132). The fused prime
    // kernel dispatches exactly one workgroup per line.
    roundtrip_c2c(
        &context,
        &[67],
        65_537,
        C2cRoute::Rader,
        65_537,
        Some("rader-fused-workgroup-stage"),
    );
    // N=68 routes through a small fitting Bluestein convolution (M=135) and
    // likewise dispatches exactly one fused workgroup per line.
    roundtrip_c2c(
        &context,
        &[68],
        65_537,
        C2cRoute::Bluestein,
        65_537,
        Some("bluestein-fused-workgroup-stage"),
    );
    // Retain coverage for the legacy multipass kernels as well. N=17 stays
    // below the fused-prime floor, and its sum pass has one workgroup per line.
    roundtrip_c2c(&context, &[17], 65_537, C2cRoute::Rader, 65_537, None);
    // N=34 likewise remains a tiny Bluestein fallback; its flattened M=70
    // pack/multiply dispatch exceeds the one-dimensional limit.
    roundtrip_c2c(&context, &[34], 62_601, C2cRoute::Bluestein, 65_536, None);
    // Real conversion kernels share the same flat dispatch path. N=256 keeps
    // their element dispatch oversized without creating millions of mostly
    // idle fused child-FFT workgroups.
    roundtrip_real(&context, 256, 16_385);

    #[cfg(windows)]
    {
        eprintln!("retaining GPU context to avoid native backend teardown hang");
        std::mem::forget(context);
    }
}

fn assert_case_exceeds_limit(context: &wgpu_fft::device::GpuContext, workgroups: u64, label: &str) {
    let limit = u64::from(context.device.limits().max_compute_workgroups_per_dimension);
    assert!(
        workgroups > limit,
        "{label}: case covers {workgroups} workgroups, which no longer exceeds \
         the device limit {limit}; grow the case so the split path stays covered"
    );
}

fn roundtrip_c2c(
    context: &wgpu_fft::device::GpuContext,
    shape: &[usize],
    batch: usize,
    expected_route: C2cRoute,
    oversized_dispatch_workgroups: u64,
    expected_fused_stage: Option<&str>,
) {
    let label = format!("c2c shape={shape:?} batch={batch}");
    let total: usize = shape.iter().product::<usize>() * batch;
    assert_case_exceeds_limit(context, oversized_dispatch_workgroups, &label);
    eprintln!("running {label} ({oversized_dispatch_workgroups} workgroups)");

    let input = test_signal(total * 2);
    let byte_len = (input.len() * std::mem::size_of::<f32>()) as u64;

    let input_buffer = storage_buffer(context, byte_len, wgpu::BufferUsages::COPY_DST, &label);
    let mid_buffer = storage_buffer(context, byte_len, wgpu::BufferUsages::empty(), &label);
    let output_buffer = storage_buffer(context, byte_len, wgpu::BufferUsages::COPY_SRC, &label);
    let readback_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.dispatch_split.readback"),
        size: byte_len,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    context
        .queue
        .write_buffer(&input_buffer, 0, bytemuck::cast_slice(&input));

    let forward = FftPlan::c2c(
        &context.device,
        &context.queue,
        FftConfig::new_nd(shape.to_vec()).with_batch(batch),
    )
    .unwrap();
    let inverse = FftPlan::c2c(
        &context.device,
        &context.queue,
        FftConfig::inverse_nd(shape.to_vec()).with_batch(batch),
    )
    .unwrap();
    assert_eq!(forward.route(), expected_route, "{label}");
    assert_eq!(inverse.route(), expected_route, "{label}");
    if let Some(expected_fused_stage) = expected_fused_stage {
        assert_single_fused_stage(&forward, expected_fused_stage, &label);
        assert_single_fused_stage(&inverse, expected_fused_stage, &label);
    }

    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("wgpu_fft.test.dispatch_split.encoder"),
        });
    forward
        .execute_checked(&context.device, &mut encoder, &input_buffer, &mid_buffer)
        .unwrap();
    inverse
        .execute_checked(&context.device, &mut encoder, &mid_buffer, &output_buffer)
        .unwrap();
    encoder.copy_buffer_to_buffer(&output_buffer, 0, &readback_buffer, 0, byte_len);
    context.queue.submit([encoder.finish()]);

    let actual = read_f32(context, &readback_buffer);
    assert_roundtrip(&actual, &input, &label);
    eprintln!("passed {label}");
}

fn assert_single_fused_stage(plan: &FftPlan, expected_label: &str, case_label: &str) {
    let diagnostics = plan.diagnostics();
    let kernels = diagnostics
        .stages()
        .iter()
        .filter(|stage| stage.kind == "kernel")
        .collect::<Vec<_>>();
    assert_eq!(
        kernels.len(),
        1,
        "{case_label}: expected one fused kernel stage, got {kernels:?}"
    );
    assert_eq!(
        kernels[0].label, expected_label,
        "{case_label}: expected fused stage"
    );
}

fn roundtrip_real(context: &wgpu_fft::device::GpuContext, len: usize, batch: usize) {
    let label = format!("real len={len} batch={batch}");
    let total = len * batch;
    assert_case_exceeds_limit(context, (total as u64).div_ceil(WORKGROUP_SIZE), &label);
    eprintln!("running {label}");

    let input = test_signal(total);
    let real_bytes = (input.len() * std::mem::size_of::<f32>()) as u64;

    let r2c = FftPlan::r2c(
        &context.device,
        &context.queue,
        FftConfig::new(len).with_batch(batch),
    )
    .unwrap();
    let c2r = FftPlan::c2r(
        &context.device,
        &context.queue,
        FftConfig::inverse(len).with_batch(batch),
    )
    .unwrap();
    let packed_bytes = r2c.required_output_buffer_size_bytes();

    let input_buffer = storage_buffer(context, real_bytes, wgpu::BufferUsages::COPY_DST, &label);
    let packed_buffer = storage_buffer(context, packed_bytes, wgpu::BufferUsages::empty(), &label);
    let output_buffer = storage_buffer(context, real_bytes, wgpu::BufferUsages::COPY_SRC, &label);
    let readback_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.dispatch_split.real_readback"),
        size: real_bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    context
        .queue
        .write_buffer(&input_buffer, 0, bytemuck::cast_slice(&input));

    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("wgpu_fft.test.dispatch_split.real_encoder"),
        });
    r2c.execute_checked(&context.device, &mut encoder, &input_buffer, &packed_buffer)
        .unwrap();
    c2r.execute_checked(
        &context.device,
        &mut encoder,
        &packed_buffer,
        &output_buffer,
    )
    .unwrap();
    encoder.copy_buffer_to_buffer(&output_buffer, 0, &readback_buffer, 0, real_bytes);
    context.queue.submit([encoder.finish()]);

    let actual = read_f32(context, &readback_buffer);
    assert_roundtrip(&actual, &input, &label);
    eprintln!("passed {label}");
}

fn storage_buffer(
    context: &wgpu_fft::device::GpuContext,
    size: u64,
    extra: wgpu::BufferUsages,
    label: &str,
) -> wgpu::Buffer {
    context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: wgpu::BufferUsages::STORAGE | extra,
        mapped_at_creation: false,
    })
}

fn test_signal(len: usize) -> Vec<f32> {
    // Deterministic full-range values; index corruption anywhere in a split
    // dispatch shows up as a large elementwise mismatch.
    let mut state = 0x243F_6A88u32;
    (0..len)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 8) as f32 / 16_777_216.0 - 1.0
        })
        .collect()
}

fn assert_roundtrip(actual: &[f32], expected: &[f32], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}");
    let mut worst = 0.0f32;
    let mut worst_index = 0usize;
    for (index, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            a.is_finite(),
            "{label}: non-finite roundtrip output {a} at element {index}"
        );
        let diff = (a - e).abs();
        if diff > worst {
            worst = diff;
            worst_index = index;
        }
    }
    assert!(
        worst <= 2.0e-3,
        "{label}: worst roundtrip error {worst} at element {worst_index} \
         (actual {}, expected {})",
        actual[worst_index],
        expected[worst_index]
    );
}

fn read_f32(context: &wgpu_fft::device::GpuContext, readback_buffer: &wgpu::Buffer) -> Vec<f32> {
    let slice = readback_buffer.slice(..);
    let (sender, receiver) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        sender.send(result).expect("map result receiver is alive");
    });
    context
        .device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("device polling should succeed");
    receiver
        .recv()
        .expect("map callback should send a result")
        .expect("readback buffer should map");

    let mapped = slice
        .get_mapped_range()
        .expect("readback buffer should be mapped");
    let actual = bytemuck::cast_slice(&mapped).to_vec();
    drop(mapped);
    readback_buffer.unmap();
    actual
}
