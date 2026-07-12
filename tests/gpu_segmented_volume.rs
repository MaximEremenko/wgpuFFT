//! Opt-in GPU coverage for the internally segmented full-volume C2C route.
//!
//! Caller-owned input and output buffers intentionally remain contiguous. Only
//! the plan-owned arena is split into physical buffers by the forced test limit.

use std::sync::mpsc;

use wgpu_fft::math::{reference_c2c_nd_f64, Complex64};
use wgpu_fft::{
    BufferSegment, BufferView, FftConfig, FftDirection, FftError, FftPlan, LargeExecutionKind,
    LargePolicyLimits, LargeRouteMode, Normalization,
};

const MODERATE_MAX_BIND_BYTES: u64 = 2_048;
const MODERATE_MAX_BUFFER_BYTES: u64 = 32_768;
const MODERATE_VOLUME_BYTES: u64 = 102_400;
const MODERATE_SEGMENT_BYTES: [u64; 4] = [32_768, 32_768, 32_768, 4_096];
const ARENA_ROLE: &str = "helper:segmented-volume-arena";

const LARGE_N0: usize = 4_096;
const LARGE_N1: usize = 8_000;
const LARGE_VOLUME_BYTES: u64 = 262_144_000;
const LARGE_MAX_BIND_BYTES: u64 = 16 * 1024 * 1024;
const LARGE_MAX_BUFFER_BYTES: u64 = 64 * 1024 * 1024;
const LARGE_SEGMENT_BYTES: [u64; 4] = [
    64 * 1024 * 1024,
    64 * 1024 * 1024,
    64 * 1024 * 1024,
    60_817_408,
];

#[cfg(target_pointer_width = "64")]
#[test]
fn segmented_volume_rejects_u32_index_space_before_device_use() {
    let config = FftConfig::new_nd([65_536, 65_536]);
    assert_eq!(
        config.validate(),
        Err(FftError::LengthTooLarge {
            len: u32::MAX as usize + 1,
        })
    );
}

#[test]
fn segmented_volume_gpu_matches_baseline_and_f64_reference() {
    if std::env::var_os("WGPU_FFT_RUN_GPU_TESTS").is_none() {
        eprintln!("skipping GPU test; set WGPU_FFT_RUN_GPU_TESTS=1 to run it");
        return;
    }
    pollster::block_on(run_gpu_cases());
}

async fn run_gpu_cases() {
    let Some(context) = wgpu_fft::device::request_default_device().await else {
        eprintln!("skipping GPU test; no suitable wgpu adapter was found");
        return;
    };
    let adapter_info = context.adapter.get_info();
    let limits = context.device.limits();
    eprintln!("adapter: {adapter_info:?}");
    eprintln!(
        "backend={:?} maxStorageBufferBindingSize={} maxBufferSize={} storageAlignment={}",
        adapter_info.backend,
        limits.max_storage_buffer_binding_size,
        limits.max_buffer_size,
        limits.min_storage_buffer_offset_alignment,
    );

    assert!(
        limits.max_storage_buffer_binding_size >= MODERATE_MAX_BIND_BYTES,
        "adapter cannot exercise the forced 2 KiB binding limit"
    );
    assert!(
        limits.max_buffer_size >= MODERATE_VOLUME_BYTES,
        "adapter cannot hold the contiguous moderate-case endpoints"
    );

    for (label, shape, batch) in [
        ("rank2", vec![100, 128], 1),
        ("rank3", vec![16, 25, 32], 1),
        ("batched-rank3", vec![8, 10, 16], 10),
    ] {
        for config in [
            FftConfig::new_nd(shape.clone())
                .with_batch(batch)
                .with_normalization(Normalization::None),
            FftConfig::inverse_nd(shape.clone()).with_batch(batch),
        ] {
            run_moderate_case(&context, label, config);
        }
    }

    run_cross_normalized_limit_case(&context);
    assert_non_mixed_axes_are_rejected(&context);
    run_large_sampled_case(&context);

    #[cfg(windows)]
    std::mem::forget(context);
}

fn run_cross_normalized_limit_case(context: &wgpu_fft::device::GpuContext) {
    const VOLUME_BYTES: u64 = 768;
    const BUFFER_CAP: u64 = 256;
    let config = FftConfig::new_nd([8, 12]).with_normalization(Normalization::None);
    let input = test_signal(config.total_complex_len().unwrap());
    let expected = reference_f64(&input, &config);
    let baseline = FftPlan::c2c(&context.device, &context.queue, config.clone()).unwrap();
    let segmented = FftPlan::c2c_with_large_policy_limits_for_testing(
        &context.device,
        &context.queue,
        config.clone(),
        LargePolicyLimits {
            max_storage_buffer_binding_size: 1_024,
            max_buffer_size: BUFFER_CAP,
        },
    )
    .unwrap();
    assert_segmented_plan(
        &segmented,
        &config,
        BUFFER_CAP,
        BUFFER_CAP,
        VOLUME_BYTES,
        &[BUFFER_CAP, BUFFER_CAP, BUFFER_CAP],
    );
    let baseline_output = execute_plan(context, &baseline, &input, "cross-limit-baseline");
    let segmented_output = execute_plan(context, &segmented, &input, "cross-limit-segmented");
    assert_close_f32(
        &segmented_output,
        &baseline_output,
        "cross-normalized maxBuffer versus baseline",
    );
    assert_matches_reference(
        &segmented_output,
        &expected,
        "cross-normalized maxBuffer versus f64",
    );

    let usages =
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;
    let offset_input = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.segmented_volume.offset_input"),
        size: VOLUME_BYTES + 8,
        usage: usages,
        mapped_at_creation: false,
    });
    let output = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.segmented_volume.diagnostics_output"),
        size: VOLUME_BYTES,
        usage: usages,
        mapped_at_creation: false,
    });
    let offset_diagnostics = segmented.diagnostics_for_views(
        &context.device,
        BufferView::new(&offset_input, 8, VOLUME_BYTES).unwrap(),
        BufferView::whole(&output),
    );
    assert!(offset_diagnostics
        .blockers()
        .iter()
        .any(|blocker| { blocker.stage.as_deref() == Some("input-segmented-volume-endpoint") }));

    let segment_a = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.segmented_volume.segment_a"),
        size: VOLUME_BYTES / 2,
        usage: usages,
        mapped_at_creation: false,
    });
    let segment_b = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.segmented_volume.segment_b"),
        size: VOLUME_BYTES / 2,
        usage: usages,
        mapped_at_creation: false,
    });
    let segmented_input = BufferView::from_segments(
        &[
            BufferSegment::new(&segment_a, 0, VOLUME_BYTES / 2),
            BufferSegment::new(&segment_b, 0, VOLUME_BYTES / 2),
        ],
        0,
        VOLUME_BYTES,
    )
    .unwrap();
    let segmented_diagnostics = segmented.diagnostics_for_views(
        &context.device,
        segmented_input,
        BufferView::whole(&output),
    );
    assert!(segmented_diagnostics
        .blockers()
        .iter()
        .any(|blocker| { blocker.stage.as_deref() == Some("input-segmented-volume-endpoint") }));
}

fn run_moderate_case(context: &wgpu_fft::device::GpuContext, label: &str, config: FftConfig) {
    assert_eq!(
        config.required_buffer_size_bytes().unwrap(),
        MODERATE_VOLUME_BYTES,
        "the regression case must allocate exactly four forced arena segments"
    );
    let direction = config.direction();
    let case_label = format!("{label}-{direction:?}");
    eprintln!("segmented moderate start: {case_label}");

    let input = test_signal(config.total_complex_len().unwrap());
    let expected = reference_f64(&input, &config);
    let baseline = FftPlan::c2c(&context.device, &context.queue, config.clone()).unwrap();
    let segmented = FftPlan::c2c_with_large_policy_limits_for_testing(
        &context.device,
        &context.queue,
        config.clone(),
        LargePolicyLimits {
            max_storage_buffer_binding_size: MODERATE_MAX_BIND_BYTES,
            max_buffer_size: MODERATE_MAX_BUFFER_BYTES,
        },
    )
    .unwrap();

    assert_segmented_plan(
        &segmented,
        &config,
        MODERATE_MAX_BIND_BYTES,
        MODERATE_MAX_BUFFER_BYTES,
        MODERATE_VOLUME_BYTES,
        &MODERATE_SEGMENT_BYTES,
    );

    let baseline_output = execute_plan(context, &baseline, &input, "segmented-baseline");
    let segmented_output = execute_plan(context, &segmented, &input, "segmented-forced");
    assert_close_f32(
        &segmented_output,
        &baseline_output,
        &format!("{case_label}: segmented versus baseline"),
    );
    assert_matches_reference(
        &segmented_output,
        &expected,
        &format!("{case_label}: segmented versus f64"),
    );
    eprintln!("segmented moderate passed: {case_label}");
}

fn assert_segmented_plan(
    plan: &FftPlan,
    config: &FftConfig,
    max_bind_bytes: u64,
    max_buffer_bytes: u64,
    full_volume_bytes: u64,
    expected_segment_bytes: &[u64],
) {
    let policy = plan.large_routing_policy();
    assert_eq!(policy.route_mode(), LargeRouteMode::LargeOutOfCore);
    assert_eq!(
        policy.execution_kind(),
        LargeExecutionKind::SegmentedFullVolume
    );
    assert_eq!(policy.max_bind_bytes, max_bind_bytes);
    assert_eq!(policy.max_buffer_size, max_buffer_bytes);
    assert_eq!(plan.workspace_size_bytes(), 0);

    let diagnostics = plan.diagnostics();
    assert_eq!(
        diagnostics.route().execution_kind.as_deref(),
        Some("segmented-full-volume")
    );
    let arena_segments = diagnostics
        .buffer_requirements()
        .iter()
        .filter(|requirement| requirement.role == ARENA_ROLE)
        .map(|requirement| requirement.required_bytes)
        .collect::<Vec<_>>();
    assert_eq!(arena_segments, expected_segment_bytes);
    assert_eq!(arena_segments.iter().sum::<u64>(), full_volume_bytes);
    assert!(arena_segments
        .iter()
        .all(|&segment_bytes| segment_bytes <= max_buffer_bytes));
    assert!(
        !diagnostics.buffer_requirements().iter().any(|requirement| {
            requirement.role.starts_with("helper:")
                && requirement.required_bytes == full_volume_bytes
        }),
        "segmented execution must not retain a full-volume helper allocation"
    );

    let scale_stage_count = diagnostics
        .stages()
        .iter()
        .filter(|stage| stage.kind == "scale")
        .count();
    match config.direction() {
        FftDirection::Forward => assert_eq!(scale_stage_count, 0),
        FftDirection::Inverse => assert_eq!(scale_stage_count, 1),
    }
}

fn assert_non_mixed_axes_are_rejected(context: &wgpu_fft::device::GpuContext) {
    for (label, config, expected_len, expected_kind) in [
        (
            "rader",
            FftConfig::new_nd([17, 256]).with_normalization(Normalization::None),
            17,
            "rader",
        ),
        (
            "bluestein",
            FftConfig::new_nd([34, 128]).with_normalization(Normalization::None),
            34,
            "bluestein",
        ),
    ] {
        assert_eq!(config.required_buffer_size_bytes().unwrap(), 34_816);
        let error = match FftPlan::c2c_with_large_policy_limits_for_testing(
            &context.device,
            &context.queue,
            config,
            LargePolicyLimits {
                max_storage_buffer_binding_size: MODERATE_MAX_BIND_BYTES,
                max_buffer_size: MODERATE_MAX_BUFFER_BYTES,
            },
        ) {
            Ok(_) => panic!("{label}: segmented Phase A unexpectedly accepted a non-mixed axis"),
            Err(error) => error,
        };
        assert_eq!(
            error,
            FftError::UnsupportedAxisKind {
                axis: 0,
                len: expected_len,
                kind: expected_kind,
            },
            "{label}: rejection must preserve the unsupported axis"
        );
    }
}

fn run_large_sampled_case(context: &wgpu_fft::device::GpuContext) {
    let limits = context.device.limits();
    if limits.max_storage_buffer_binding_size < LARGE_MAX_BIND_BYTES
        || limits.max_buffer_size < LARGE_VOLUME_BYTES
    {
        eprintln!(
            "skipping 250 MiB segmented case: maxBind={} maxBuffer={}",
            limits.max_storage_buffer_binding_size, limits.max_buffer_size
        );
        return;
    }

    let config = FftConfig::new_nd([LARGE_N0, LARGE_N1]).with_normalization(Normalization::None);
    assert_eq!(
        config.required_buffer_size_bytes().unwrap(),
        LARGE_VOLUME_BYTES
    );
    eprintln!("segmented 250 MiB start: shape=[{LARGE_N0}, {LARGE_N1}] bytes={LARGE_VOLUME_BYTES}");
    let plan = FftPlan::c2c_with_large_policy_limits_for_testing(
        &context.device,
        &context.queue,
        config.clone(),
        LargePolicyLimits {
            max_storage_buffer_binding_size: LARGE_MAX_BIND_BYTES,
            max_buffer_size: LARGE_MAX_BUFFER_BYTES,
        },
    )
    .unwrap();
    assert_segmented_plan(
        &plan,
        &config,
        LARGE_MAX_BIND_BYTES,
        LARGE_MAX_BUFFER_BYTES,
        LARGE_VOLUME_BYTES,
        &LARGE_SEGMENT_BYTES,
    );

    let usages =
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;
    let input = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.segmented_volume.large.input"),
        size: LARGE_VOLUME_BYTES,
        usage: usages,
        mapped_at_creation: false,
    });
    let output = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.segmented_volume.large.output"),
        size: LARGE_VOLUME_BYTES,
        usage: usages,
        mapped_at_creation: false,
    });
    let mut clear_encoder =
        context
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("wgpu_fft.test.segmented_volume.large.clear"),
            });
    clear_encoder.clear_buffer(&input, 0, None);
    context.queue.submit([clear_encoder.finish()]);
    context
        .device
        .poll(wgpu::PollType::wait_indefinitely())
        .unwrap();

    let impulses = [
        (3usize, 5usize, Complex64::new(0.75, -0.25)),
        (LARGE_N0 - 7, LARGE_N1 - 11, Complex64::new(-0.4, 0.6)),
    ];
    for &(x, y, value) in &impulses {
        let offset = ((y as u64) * LARGE_N0 as u64 + x as u64) * 8;
        context.queue.write_buffer(
            &input,
            offset,
            bytemuck::cast_slice(&[value.re as f32, value.im as f32]),
        );
    }

    let sampled_k1 = [0usize, 1usize, LARGE_N1 - 1];
    let line_bytes = LARGE_N0 as u64 * 8;
    let readback = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.segmented_volume.large.readback"),
        size: line_bytes * sampled_k1.len() as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let input_view = wgpu_fft::BufferView::whole(&input);
    let output_view = wgpu_fft::BufferView::whole(&output);
    let view_diagnostics = plan.diagnostics_for_views(&context.device, input_view, output_view);
    assert!(
        view_diagnostics.blockers().is_empty(),
        "whole contiguous endpoints must be valid against the real device limits: {:?}",
        view_diagnostics.blockers()
    );

    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("wgpu_fft.test.segmented_volume.large.encoder"),
        });
    plan.execute_checked(&context.device, &mut encoder, &input, &output)
        .unwrap();
    for (sample_index, &k1) in sampled_k1.iter().enumerate() {
        encoder.copy_buffer_to_buffer(
            &output,
            k1 as u64 * line_bytes,
            &readback,
            sample_index as u64 * line_bytes,
            line_bytes,
        );
    }
    context.queue.submit([encoder.finish()]);
    let actual = read_f32(&context.device, &readback);
    for (sample_index, &k1) in sampled_k1.iter().enumerate() {
        let line = &actual[sample_index * LARGE_N0 * 2..(sample_index + 1) * LARGE_N0 * 2];
        for (k0, pair) in line.chunks_exact(2).enumerate() {
            let expected = impulses
                .iter()
                .fold(Complex64::new(0.0, 0.0), |sum, &(x, y, value)| {
                    let angle = -std::f64::consts::TAU
                        * ((k0 * x) as f64 / LARGE_N0 as f64 + (k1 * y) as f64 / LARGE_N1 as f64);
                    let twiddle = Complex64::new(angle.cos(), angle.sin());
                    Complex64::new(
                        sum.re + value.re * twiddle.re - value.im * twiddle.im,
                        sum.im + value.re * twiddle.im + value.im * twiddle.re,
                    )
                });
            let error = (f64::from(pair[0]) - expected.re).hypot(f64::from(pair[1]) - expected.im);
            assert!(
                error.is_finite() && error < 1.0e-3,
                "250 MiB sample k0={k0} k1={k1}: actual=({}, {}), expected=({}, {}), error={error}",
                pair[0],
                pair[1],
                expected.re,
                expected.im,
            );
        }
    }
    eprintln!(
        "segmented 250 MiB passed: shape=[{LARGE_N0}, {LARGE_N1}] sampled_lines={sampled_k1:?}"
    );
}

fn execute_plan(
    context: &wgpu_fft::device::GpuContext,
    plan: &FftPlan,
    input: &[f32],
    label: &str,
) -> Vec<f32> {
    let byte_len = std::mem::size_of_val(input) as u64;
    let usages =
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;
    let input_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.segmented_volume.input"),
        size: byte_len,
        usage: usages,
        mapped_at_creation: false,
    });
    let output_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.segmented_volume.output"),
        size: byte_len,
        usage: usages,
        mapped_at_creation: false,
    });
    let readback = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.segmented_volume.readback"),
        size: byte_len,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    context
        .queue
        .write_buffer(&input_buffer, 0, bytemuck::cast_slice(input));

    let view_diagnostics = plan.diagnostics_for_views(
        &context.device,
        wgpu_fft::BufferView::whole(&input_buffer),
        wgpu_fft::BufferView::whole(&output_buffer),
    );
    assert!(
        view_diagnostics.blockers().is_empty(),
        "{label}: whole contiguous endpoints must be valid against real device limits: {:?}",
        view_diagnostics.blockers()
    );

    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("wgpu_fft.test.segmented_volume.encoder"),
        });
    plan.execute_checked(&context.device, &mut encoder, &input_buffer, &output_buffer)
        .unwrap();
    encoder.copy_buffer_to_buffer(&output_buffer, 0, &readback, 0, byte_len);
    context.queue.submit([encoder.finish()]);
    read_f32(&context.device, &readback)
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

fn reference_f64(input: &[f32], config: &FftConfig) -> Vec<Complex64> {
    let values = input
        .chunks_exact(2)
        .map(|pair| Complex64::new(f64::from(pair[0]), f64::from(pair[1])))
        .collect::<Vec<_>>();
    reference_c2c_nd_f64(&values, config).unwrap()
}

fn assert_close_f32(actual: &[f32], expected: &[f32], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}");
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        let tolerance = 5.0e-3 + 3.0e-5 * expected.abs();
        assert!(
            actual.is_finite() && (actual - expected).abs() <= tolerance,
            "{label}: scalar {index}: actual={actual}, expected={expected}, tolerance={tolerance}"
        );
    }
}

fn assert_matches_reference(actual: &[f32], expected: &[Complex64], label: &str) {
    assert_eq!(actual.len(), expected.len() * 2, "{label}");
    for (index, (pair, expected)) in actual.chunks_exact(2).zip(expected).enumerate() {
        let error = (f64::from(pair[0]) - expected.re).hypot(f64::from(pair[1]) - expected.im);
        let tolerance = 1.0e-2 + 3.0e-5 * expected.re.hypot(expected.im);
        assert!(
            error.is_finite() && error <= tolerance,
            "{label}: complex {index}: actual=({}, {}), expected=({}, {}), error={error}, tolerance={tolerance}",
            pair[0], pair[1], expected.re, expected.im,
        );
    }
}

fn read_f32(device: &wgpu::Device, buffer: &wgpu::Buffer) -> Vec<f32> {
    let slice = buffer.slice(..);
    let (sender, receiver) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        sender.send(result).expect("map result receiver is alive");
    });
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    receiver.recv().unwrap().unwrap();
    let mapped = slice.get_mapped_range();
    let values = bytemuck::cast_slice(&mapped).to_vec();
    drop(mapped);
    buffer.unmap();
    values
}
