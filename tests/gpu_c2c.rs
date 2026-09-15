#![cfg(not(target_arch = "wasm32"))]

use std::sync::mpsc;

use wgpu_fft::math::{from_interleaved_f32, reference_c2c_nd, to_interleaved_f32};
use wgpu_fft::{
    clear_thread_local_pipeline_cache, export_pipeline_cache_snapshot,
    import_pipeline_cache_snapshot, BufferLayout, BufferSegment, BufferView, C2cRoute,
    FftBlockerKind, FftConfig, FftDeviceLimits, FftError, FftIoView, FftLogicalLayout,
    FftLogicalView, FftPlan, FftTuning, LargeExecutionKind, LargePolicyLimits, LargeRouteMode,
    Normalization, PIPELINE_CACHE_SNAPSHOT_SCHEMA, PIPELINE_CACHE_SNAPSHOT_VERSION,
};

fn c2c_route_name(route: C2cRoute) -> &'static str {
    match route {
        C2cRoute::DirectDft => "direct-dft",
        C2cRoute::MixedRadix => "mixed-radix",
        C2cRoute::Rader => "rader",
        C2cRoute::Bluestein => "bluestein",
        C2cRoute::AxisSequence => "axis-sequence",
    }
}

#[test]
fn c2c_gpu_matches_cpu_reference_for_small_size() {
    if std::env::var_os("WGPU_FFT_RUN_GPU_TESTS").is_none() {
        eprintln!("skipping GPU test; set WGPU_FFT_RUN_GPU_TESTS=1 to run it");
        return;
    }

    pollster::block_on(run_gpu_case());
}

async fn run_gpu_case() {
    let Some(context) = wgpu_fft::device::request_default_device().await else {
        eprintln!("skipping GPU test; no suitable wgpu adapter was found");
        return;
    };
    trace_adapter(&context);

    for config in [
        FftConfig::new(1).with_normalization(Normalization::None),
        FftConfig::inverse(1),
    ] {
        run_one_case(
            &context,
            input_for_config(&config),
            config,
            C2cRoute::DirectDft,
        );
    }

    for len in [2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 15, 16, 21] {
        run_one_case(
            &context,
            input_for_len(len),
            FftConfig::new(len).with_normalization(Normalization::None),
            C2cRoute::MixedRadix,
        );
    }

    for len in [8, 12, 15] {
        run_one_case(
            &context,
            input_for_len(len),
            FftConfig::inverse(len),
            C2cRoute::MixedRadix,
        );
    }

    for len in [17, 29] {
        run_one_case(
            &context,
            input_for_len(len),
            FftConfig::new(len).with_normalization(Normalization::None),
            C2cRoute::Rader,
        );
    }

    for len in [17, 29] {
        run_one_case(
            &context,
            input_for_len(len),
            FftConfig::inverse(len),
            C2cRoute::Rader,
        );
    }

    run_one_case(
        &context,
        input_for_config(&FftConfig::new(17).with_batch(2)),
        FftConfig::new(17)
            .with_batch(2)
            .with_normalization(Normalization::None),
        C2cRoute::Rader,
    );

    for config in [
        FftConfig::new_nd([17, 4])
            .with_tuning(per_axis_tuning())
            .with_axes([0])
            .with_normalization(Normalization::None),
        FftConfig::new_nd([4, 17])
            .with_axes([1])
            .with_normalization(Normalization::None),
        FftConfig::new_nd([4, 17])
            .with_axes([1])
            .with_batch(2)
            .with_normalization(Normalization::None),
    ] {
        run_one_case(&context, input_for_config(&config), config, C2cRoute::Rader);
    }

    for config in [
        FftConfig::new_nd([17, 4])
            .with_tuning(per_axis_tuning())
            .with_normalization(Normalization::None),
        FftConfig::new_nd([4, 17]).with_normalization(Normalization::None),
        FftConfig::new_nd([17, 3, 2]).with_normalization(Normalization::None),
        FftConfig::new_nd([17, 4])
            .with_tuning(per_axis_tuning())
            .with_batch(2)
            .with_normalization(Normalization::None),
    ] {
        run_one_case(
            &context,
            input_for_config(&config),
            config,
            C2cRoute::AxisSequence,
        );
    }

    for len in [34, 38, 46, 58] {
        run_one_case(
            &context,
            input_for_len(len),
            FftConfig::new(len).with_normalization(Normalization::None),
            C2cRoute::Bluestein,
        );
    }

    for config in [
        FftConfig::new_nd([34, 4])
            .with_axes([0])
            .with_normalization(Normalization::None),
        FftConfig::new_nd([4, 34])
            .with_axes([1])
            .with_normalization(Normalization::None),
    ] {
        run_one_case(
            &context,
            input_for_config(&config),
            config,
            C2cRoute::Bluestein,
        );
    }

    for config in [
        FftConfig::new_nd([4, 34]).with_normalization(Normalization::None),
        FftConfig::new_nd([17, 34]).with_normalization(Normalization::None),
    ] {
        run_one_case(
            &context,
            input_for_config(&config),
            config,
            C2cRoute::AxisSequence,
        );
    }

    for config in [
        FftConfig::new_nd([2, 3]).with_normalization(Normalization::None),
        FftConfig::new_nd([4, 3])
            .with_tuning(per_axis_tuning())
            .with_axes([0])
            .with_normalization(Normalization::None),
        FftConfig::new_nd([4, 3])
            .with_tuning(per_axis_tuning())
            .with_axes([1])
            .with_normalization(Normalization::None),
        FftConfig::new_nd([4, 3, 2]).with_normalization(Normalization::None),
        FftConfig::new_nd([2, 3])
            .with_batch(2)
            .with_normalization(Normalization::None),
    ] {
        run_one_case(
            &context,
            input_for_config(&config),
            config,
            C2cRoute::MixedRadix,
        );
    }

    for config in [
        FftConfig::inverse_nd([2, 3]),
        FftConfig::inverse_nd([4, 3]).with_axes([0]),
        FftConfig::inverse_nd([2, 3]).with_batch(2),
    ] {
        run_one_case(
            &context,
            input_for_config(&config),
            config,
            C2cRoute::MixedRadix,
        );
    }

    assert_workspace_behavior(&context);
    assert_pipeline_cache_snapshot_behavior(&context);
    assert_pipeline_cache_clear_behavior(&context);
    assert_view_validation_behavior(&context);
    run_one_case_with_caller_workspace(
        &context,
        FftConfig::new_nd([4, 3])
            .with_tuning(per_axis_tuning())
            .with_normalization(Normalization::None),
    );
    run_offset_view_cases(&context);
    run_segmented_view_cases(&context);
    run_one_case_with_offset_workspace(
        &context,
        FftConfig::new_nd([4, 3])
            .with_tuning(per_axis_tuning())
            .with_normalization(Normalization::None),
    );
    run_one_case_with_strided_logical_workspace(&context);
    assert_segmented_workspace_rejected(&context);
    run_strided_view_cases(&context);
    assert_strided_pipeline_cache_snapshot_behavior(&context);
    run_large_chunk_cases(&context);
    run_smooth_decomposition_cases(&context);
    run_large_bridge_cases(&context);
    run_large_axis_sequence_cases(&context);
    assert_smooth_decomposition_pipeline_cache_snapshot_behavior(&context);
    assert_large_bridge_pipeline_cache_snapshot_behavior(&context);
    trace_gpu_step("finish c2c gpu test");

    #[cfg(windows)]
    {
        trace_gpu_step("forget gpu context to avoid native backend teardown hang");
        std::mem::forget(context);
    }
}

fn assert_pipeline_cache_snapshot_behavior(context: &wgpu_fft::device::GpuContext) {
    let snapshot = export_pipeline_cache_snapshot(&context.device);
    assert_eq!(snapshot.schema(), PIPELINE_CACHE_SNAPSHOT_SCHEMA);
    assert_eq!(snapshot.version(), PIPELINE_CACHE_SNAPSHOT_VERSION);
    assert!(!snapshot.shader_codes().is_empty());
    assert!(!snapshot.pipeline_keys().is_empty());
    assert!(snapshot
        .pipeline_keys()
        .iter()
        .any(|key| key.contains("fused-pow2")));
    assert!(snapshot
        .pipeline_keys()
        .iter()
        .any(|key| key.contains("fused-smooth")));
    // Short prime axes run direct DFT kernels instead of Rader's.
    assert!(snapshot
        .pipeline_keys()
        .iter()
        .any(|key| key.contains("rader") || key.contains("fused-prime:direct")));
    assert!(snapshot
        .shader_codes()
        .iter()
        .any(|code| code.contains("const N: u32 = 17u;")));

    let imported = import_pipeline_cache_snapshot(&context.device, &snapshot);
    assert_eq!(imported.schema(), PIPELINE_CACHE_SNAPSHOT_SCHEMA);
    assert_eq!(imported.version(), PIPELINE_CACHE_SNAPSHOT_VERSION);
    assert_eq!(imported.shader_codes(), snapshot.shader_codes());
    assert_eq!(imported.pipeline_keys(), snapshot.pipeline_keys());
}

fn assert_pipeline_cache_clear_behavior(context: &wgpu_fft::device::GpuContext) {
    assert!(clear_thread_local_pipeline_cache(&context.device));
    assert!(!clear_thread_local_pipeline_cache(&context.device));

    let plan = FftPlan::c2c(
        &context.device,
        &context.queue,
        FftConfig::new(8).with_normalization(Normalization::None),
    )
    .unwrap();
    assert!(clear_thread_local_pipeline_cache(&context.device));

    let required = plan.required_buffer_size_bytes();
    let input = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.cache_clear_input"),
        size: required,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let output = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.cache_clear_output"),
        size: required,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    context.queue.write_buffer(
        &input,
        0,
        bytemuck::cast_slice(&input_for_total_complex_len(8)),
    );
    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("wgpu_fft.test.cache_clear_existing_plan"),
        });
    plan.execute_checked(&context.device, &mut encoder, &input, &output)
        .unwrap();
    let submission = context.queue.submit([encoder.finish()]);
    context
        .device
        .poll(wgpu::PollType::Wait {
            submission_index: Some(submission),
            timeout: Some(std::time::Duration::from_secs(30)),
        })
        .unwrap();

    let empty_snapshot = export_pipeline_cache_snapshot(&context.device);
    assert!(empty_snapshot.shader_codes().is_empty());
    assert!(empty_snapshot.pipeline_keys().is_empty());
    assert!(clear_thread_local_pipeline_cache(&context.device));
    assert!(!clear_thread_local_pipeline_cache(&context.device));
}

fn assert_strided_pipeline_cache_snapshot_behavior(context: &wgpu_fft::device::GpuContext) {
    let snapshot = export_pipeline_cache_snapshot(&context.device);
    assert!(snapshot
        .pipeline_keys()
        .iter()
        .any(|key| key.contains("pack-c2c-strided")));
    assert!(snapshot
        .pipeline_keys()
        .iter()
        .any(|key| key.contains("unpack-c2c-strided")));
}

fn assert_smooth_decomposition_pipeline_cache_snapshot_behavior(
    context: &wgpu_fft::device::GpuContext,
) {
    let snapshot = export_pipeline_cache_snapshot(&context.device);
    for expected in [
        "gather-axis-line",
        "scatter-axis-line",
        "gather-smooth-phase1",
        "scatter-smooth-phase2",
        "twiddle-transpose",
    ] {
        assert!(
            snapshot
                .pipeline_keys()
                .iter()
                .any(|key| key.contains(expected)),
            "missing C2C smooth decomposition pipeline key containing {expected}"
        );
    }
}

fn assert_large_bridge_pipeline_cache_snapshot_behavior(context: &wgpu_fft::device::GpuContext) {
    let snapshot = export_pipeline_cache_snapshot(&context.device);
    for expected in [
        "rader-sum-init",
        "rader-sum-accumulate",
        "rader-pack-windowed",
        "rader-mul-windowed",
        "rader-write-y0-windowed",
        "rader-post-windowed",
        "bluestein-pack-windowed",
        "bluestein-mul-windowed",
        "bluestein-post-windowed",
    ] {
        assert!(
            snapshot
                .pipeline_keys()
                .iter()
                .any(|key| key.contains(expected)),
            "missing large bridge pipeline key containing {expected}"
        );
    }
}

fn input_for_len(len: usize) -> Vec<f32> {
    input_for_total_complex_len(len)
}

fn input_for_config(config: &FftConfig) -> Vec<f32> {
    input_for_total_complex_len(config.total_complex_len().unwrap())
}

fn input_for_total_complex_len(len: usize) -> Vec<f32> {
    (0..len)
        .flat_map(|i| {
            let x = i as f32;
            [
                (x * 0.37).sin() * 0.75 + (x * 0.11).cos() * 0.25,
                (x * 0.23).cos() * 0.5 - (x * 0.19).sin() * 0.35,
            ]
        })
        .collect()
}

fn assert_workspace_behavior(context: &wgpu_fft::device::GpuContext) {
    let one_stage = FftPlan::c2c(&context.device, &context.queue, FftConfig::new(8)).unwrap();
    assert_eq!(one_stage.route(), C2cRoute::MixedRadix);
    assert_eq!(one_stage.workspace_size_bytes(), 0);
    let zero_workspace_input = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.zero_workspace_input"),
        size: one_stage.required_buffer_size_bytes(),
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let zero_workspace_output = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.zero_workspace_output"),
        size: one_stage.required_buffer_size_bytes(),
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let unused_workspace_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.zero_unused_segmented_workspace"),
        size: 16,
        usage: wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let unused_workspace_segments = split_buffer_segments(&unused_workspace_buffer, 16, 2);
    let unused_workspace = BufferView::from_segments(&unused_workspace_segments, 0, 16).unwrap();
    let mut zero_workspace_encoder =
        context
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("wgpu_fft.test.zero_workspace_encoder"),
            });
    one_stage
        .execute_views_with_workspace(
            &context.device,
            &mut zero_workspace_encoder,
            BufferView::whole(&zero_workspace_input),
            BufferView::whole(&zero_workspace_output),
            unused_workspace,
        )
        .unwrap();

    let multi_stage = FftPlan::c2c(
        &context.device,
        &context.queue,
        FftConfig::new_nd([4, 3]).with_tuning(per_axis_tuning()),
    )
    .unwrap();
    assert_eq!(multi_stage.route(), C2cRoute::MixedRadix);
    assert_eq!(multi_stage.workspace_size_bytes(), 12 * 2 * 4);

    let nd_batched = FftPlan::c2c(
        &context.device,
        &context.queue,
        FftConfig::new_nd([4, 3])
            .with_tuning(per_axis_tuning())
            .with_batch(2),
    )
    .unwrap();
    assert_eq!(nd_batched.route(), C2cRoute::MixedRadix);
    assert_eq!(nd_batched.workspace_size_bytes(), 24 * 2 * 4);
    assert_eq!(nd_batched.required_buffer_size_bytes(), 24 * 2 * 4);

    let sequence = FftPlan::c2c(
        &context.device,
        &context.queue,
        FftConfig::new_nd([17, 4])
            .with_tuning(per_axis_tuning())
            .with_normalization(Normalization::None),
    )
    .unwrap();
    assert_eq!(sequence.route(), C2cRoute::AxisSequence);
    assert_eq!(sequence.workspace_size_bytes(), 68 * 2 * 4);
    assert_eq!(sequence.required_buffer_size_bytes(), 68 * 2 * 4);

    let too_small_workspace = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.too_small_workspace"),
        size: multi_stage.workspace_size_bytes() - 4,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let input_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.workspace_check_input"),
        size: multi_stage.required_buffer_size_bytes(),
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let output_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.workspace_check_output"),
        size: multi_stage.required_buffer_size_bytes(),
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("wgpu_fft.test.workspace_check_encoder"),
        });
    assert_eq!(
        multi_stage.execute_with_workspace(
            &context.device,
            &mut encoder,
            &input_buffer,
            &output_buffer,
            &too_small_workspace,
        ),
        Err(FftError::WorkspaceTooSmall {
            required: multi_stage.workspace_size_bytes(),
            actual: multi_stage.workspace_size_bytes() - 4,
        })
    );
}

fn run_one_case_with_caller_workspace(context: &wgpu_fft::device::GpuContext, config: FftConfig) {
    trace_gpu_step(&format!("start caller workspace config={config:?}"));
    let input = input_for_config(&config);
    let expected = reference_c2c_nd(&from_interleaved_f32(&input), &config).unwrap();
    let expected = to_interleaved_f32(&expected);
    let byte_len = (input.len() * std::mem::size_of::<f32>()) as u64;

    let input_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.workspace_input"),
        size: byte_len,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    context
        .queue
        .write_buffer(&input_buffer, 0, bytemuck::cast_slice(&input));

    let output_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.workspace_output"),
        size: byte_len,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.workspace_readback"),
        size: byte_len,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let plan = FftPlan::c2c(&context.device, &context.queue, config).unwrap();
    assert_eq!(plan.route(), C2cRoute::MixedRadix);
    assert!(plan.workspace_size_bytes() > 0);
    let workspace = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.caller_workspace"),
        size: plan.workspace_size_bytes(),
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("wgpu_fft.test.workspace_encoder"),
        });
    plan.execute_with_workspace(
        &context.device,
        &mut encoder,
        &input_buffer,
        &output_buffer,
        &workspace,
    )
    .unwrap();
    encoder.copy_buffer_to_buffer(&output_buffer, 0, &readback_buffer, 0, byte_len);
    context.queue.submit([encoder.finish()]);
    let label = String::from("caller workspace");
    trace_gpu_step(&format!("submitted {label}"));

    let actual = read_interleaved_f32(context, &readback_buffer, &label);
    assert_close(&actual, &expected, &label);
    trace_gpu_step(&format!("finish {label}"));
}

fn run_offset_view_cases(context: &wgpu_fft::device::GpuContext) {
    for (config, route) in [
        (
            FftConfig::new(16).with_normalization(Normalization::None),
            C2cRoute::MixedRadix,
        ),
        (
            FftConfig::new(17).with_normalization(Normalization::None),
            C2cRoute::Rader,
        ),
        (
            FftConfig::new(34).with_normalization(Normalization::None),
            C2cRoute::Bluestein,
        ),
        (
            FftConfig::new_nd([17, 4])
                .with_tuning(per_axis_tuning())
                .with_normalization(Normalization::None),
            C2cRoute::AxisSequence,
        ),
    ] {
        run_one_case_with_offset_views(context, config, route);
    }
}

fn run_one_case_with_offset_views(
    context: &wgpu_fft::device::GpuContext,
    config: FftConfig,
    expected_route: C2cRoute,
) {
    trace_gpu_step(&format!(
        "start offset views config={config:?} route={expected_route:?}"
    ));
    let input = input_for_config(&config);
    let expected = reference_c2c_nd(&from_interleaved_f32(&input), &config).unwrap();
    let expected = to_interleaved_f32(&expected);
    let byte_len = (input.len() * std::mem::size_of::<f32>()) as u64;
    let offset = aligned_test_offset(context);
    let backing_size = offset + byte_len + offset;

    let input_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.offset_input"),
        size: backing_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    context
        .queue
        .write_buffer(&input_buffer, offset, bytemuck::cast_slice(&input));

    let output_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.offset_output"),
        size: backing_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.offset_readback"),
        size: byte_len,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let plan = FftPlan::c2c(&context.device, &context.queue, config).unwrap();
    assert_eq!(plan.route(), expected_route);
    assert_eq!(
        plan.large_routing_policy().route_mode(),
        LargeRouteMode::Normal
    );
    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("wgpu_fft.test.offset_encoder"),
        });
    plan.execute_views(
        &context.device,
        &mut encoder,
        BufferView::new(&input_buffer, offset, byte_len).unwrap(),
        BufferView::new(&output_buffer, offset, byte_len).unwrap(),
    )
    .unwrap();
    encoder.copy_buffer_to_buffer(&output_buffer, offset, &readback_buffer, 0, byte_len);
    context.queue.submit([encoder.finish()]);

    let label = format!("offset views {:?}", plan.config());
    trace_gpu_step(&format!("submitted {label}"));
    let actual = read_interleaved_f32(context, &readback_buffer, &label);
    assert_close(&actual, &expected, &label);
    trace_gpu_step(&format!("finish {label}"));
}

fn run_one_case_with_offset_workspace(context: &wgpu_fft::device::GpuContext, config: FftConfig) {
    trace_gpu_step(&format!("start offset caller workspace config={config:?}"));
    let input = input_for_config(&config);
    let expected = reference_c2c_nd(&from_interleaved_f32(&input), &config).unwrap();
    let expected = to_interleaved_f32(&expected);
    let byte_len = (input.len() * std::mem::size_of::<f32>()) as u64;
    let offset = aligned_test_offset(context);
    let backing_size = offset + byte_len + offset;

    let input_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.offset_workspace_input"),
        size: backing_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    context
        .queue
        .write_buffer(&input_buffer, offset, bytemuck::cast_slice(&input));

    let output_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.offset_workspace_output"),
        size: backing_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.offset_workspace_readback"),
        size: byte_len,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let plan = FftPlan::c2c(&context.device, &context.queue, config).unwrap();
    assert_eq!(plan.route(), C2cRoute::MixedRadix);
    assert!(plan.workspace_size_bytes() > 0);
    let workspace_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.offset_workspace"),
        size: offset + plan.workspace_size_bytes() + offset,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("wgpu_fft.test.offset_workspace_encoder"),
        });
    plan.execute_views_with_workspace(
        &context.device,
        &mut encoder,
        BufferView::new(&input_buffer, offset, byte_len).unwrap(),
        BufferView::new(&output_buffer, offset, byte_len).unwrap(),
        BufferView::new(&workspace_buffer, offset, plan.workspace_size_bytes()).unwrap(),
    )
    .unwrap();
    encoder.copy_buffer_to_buffer(&output_buffer, offset, &readback_buffer, 0, byte_len);
    context.queue.submit([encoder.finish()]);
    let label = String::from("offset caller workspace");
    trace_gpu_step(&format!("submitted {label}"));

    let actual = read_interleaved_f32(context, &readback_buffer, &label);
    assert_close(&actual, &expected, &label);
    trace_gpu_step(&format!("finish {label}"));
}

/// Tuning for tiny ND shapes that exercise per-axis plans (their stages and
/// workspaces) rather than the single small-volume kernel.
fn per_axis_tuning() -> FftTuning {
    FftTuning::default().with_fuse_small_volumes(false)
}

fn run_one_case_with_strided_logical_workspace(context: &wgpu_fft::device::GpuContext) {
    let config = FftConfig::new_nd([4, 3])
        .with_tuning(per_axis_tuning())
        .with_normalization(Normalization::None);
    trace_gpu_step(&format!(
        "start strided logical workspace config={config:?}"
    ));
    let input = input_for_config(&config);
    let expected = reference_c2c_nd(&from_interleaved_f32(&input), &config).unwrap();
    let expected = to_interleaved_f32(&expected);
    let logical_per_batch = config.logical_complex_len().unwrap() as u64;
    let batch = config.batch() as u64;
    let input_layout = strided_test_layout(logical_per_batch, 3);
    let output_layout = strided_test_layout(logical_per_batch, 5);
    let input_span_bytes = layout_span_complex(input_layout, logical_per_batch, batch) * 8;
    let output_span_bytes = layout_span_complex(output_layout, logical_per_batch, batch) * 8;
    let physical_input =
        scatter_strided_interleaved(&input, input_layout, logical_per_batch, batch);

    let input_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.strided_workspace_input"),
        size: input_span_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    context
        .queue
        .write_buffer(&input_buffer, 0, bytemuck::cast_slice(&physical_input));
    let output_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.strided_workspace_output"),
        size: output_span_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.strided_workspace_readback"),
        size: output_span_bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let plan = FftPlan::c2c(&context.device, &context.queue, config).unwrap();
    assert_eq!(plan.route(), C2cRoute::MixedRadix);
    assert!(plan.workspace_size_bytes() > 0);
    let workspace_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.strided_logical_workspace"),
        size: plan.workspace_size_bytes(),
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    let input_view = BufferView::whole(&input_buffer);
    let output_view = BufferView::whole(&output_buffer);
    let input_logical =
        FftLogicalView::new(input_view.clone(), logical_layout_from_c2c(input_layout)).unwrap();
    let output_logical =
        FftLogicalView::new(output_view.clone(), logical_layout_from_c2c(output_layout)).unwrap();
    let workspace_view = BufferView::whole(&workspace_buffer);
    let logical_diagnostics = plan.diagnostics_for_logical_views_with_workspace(
        &context.device,
        &input_logical,
        &output_logical,
        workspace_view.clone(),
    );
    assert!(logical_diagnostics.blockers().is_empty());
    assert!(logical_diagnostics
        .stages()
        .iter()
        .any(|stage| stage.kind == "strided-pack"));
    assert!(logical_diagnostics
        .stages()
        .iter()
        .any(|stage| stage.kind == "strided-unpack"));

    let io_diagnostics = plan.diagnostics_for_io_views_with_workspace(
        &context.device,
        FftIoView::new(input_view, input_layout).unwrap(),
        FftIoView::new(output_view, output_layout).unwrap(),
        workspace_view.clone(),
    );
    assert!(io_diagnostics.blockers().is_empty());
    assert!(io_diagnostics
        .stages()
        .iter()
        .any(|stage| stage.kind == "strided-pack"));
    assert!(io_diagnostics
        .stages()
        .iter()
        .any(|stage| stage.kind == "strided-unpack"));

    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("wgpu_fft.test.strided_workspace_encoder"),
        });
    plan.execute_logical_views_with_workspace(
        &context.device,
        &mut encoder,
        input_logical,
        output_logical,
        workspace_view,
    )
    .unwrap();
    encoder.copy_buffer_to_buffer(&output_buffer, 0, &readback_buffer, 0, output_span_bytes);
    context.queue.submit([encoder.finish()]);

    let label = "strided logical caller workspace";
    trace_gpu_step(&format!("submitted {label}"));
    let actual_physical = read_interleaved_f32(context, &readback_buffer, label);
    let actual =
        gather_strided_interleaved(&actual_physical, output_layout, logical_per_batch, batch);
    assert_close(&actual, &expected, label);
    trace_gpu_step(&format!("finish {label}"));
}

fn run_segmented_view_cases(context: &wgpu_fft::device::GpuContext) {
    for (config, route) in [
        (
            FftConfig::new(16).with_normalization(Normalization::None),
            C2cRoute::MixedRadix,
        ),
        (
            FftConfig::new(17).with_normalization(Normalization::None),
            C2cRoute::Rader,
        ),
        (
            FftConfig::new(34).with_normalization(Normalization::None),
            C2cRoute::Bluestein,
        ),
        (
            FftConfig::new_nd([17, 4])
                .with_tuning(per_axis_tuning())
                .with_normalization(Normalization::None),
            C2cRoute::AxisSequence,
        ),
    ] {
        run_one_case_with_segmented_views(context, config, route, true, true);
    }

    run_one_case_with_segmented_views(
        context,
        FftConfig::new(16).with_normalization(Normalization::None),
        C2cRoute::MixedRadix,
        true,
        false,
    );
    run_one_case_with_segmented_views(
        context,
        FftConfig::new(16).with_normalization(Normalization::None),
        C2cRoute::MixedRadix,
        false,
        true,
    );
}

#[derive(Debug, Clone, Copy)]
enum StridedMode {
    Both,
    InputOnly,
    OutputOnly,
    OffsetBoth,
}

fn run_strided_view_cases(context: &wgpu_fft::device::GpuContext) {
    for (config, route) in [
        (
            FftConfig::new(16).with_normalization(Normalization::None),
            C2cRoute::MixedRadix,
        ),
        (
            FftConfig::new(17).with_normalization(Normalization::None),
            C2cRoute::Rader,
        ),
        (
            FftConfig::new(34).with_normalization(Normalization::None),
            C2cRoute::Bluestein,
        ),
        (
            FftConfig::new_nd([17, 4])
                .with_tuning(per_axis_tuning())
                .with_normalization(Normalization::None),
            C2cRoute::AxisSequence,
        ),
        (
            FftConfig::new(16)
                .with_batch(2)
                .with_normalization(Normalization::None),
            C2cRoute::MixedRadix,
        ),
    ] {
        run_one_case_with_strided_layout(context, config, route, StridedMode::Both);
    }

    for mode in [
        StridedMode::InputOnly,
        StridedMode::OutputOnly,
        StridedMode::OffsetBoth,
    ] {
        run_one_case_with_strided_layout(
            context,
            FftConfig::new(16).with_normalization(Normalization::None),
            C2cRoute::MixedRadix,
            mode,
        );
    }
    run_one_case_with_segmented_strided_layout(context);
}

fn run_one_case_with_strided_layout(
    context: &wgpu_fft::device::GpuContext,
    config: FftConfig,
    expected_route: C2cRoute,
    mode: StridedMode,
) {
    trace_gpu_step(&format!(
        "start strided config={config:?} route={expected_route:?} mode={mode:?}"
    ));
    let input = input_for_config(&config);
    let expected = reference_c2c_nd(&from_interleaved_f32(&input), &config).unwrap();
    let expected = to_interleaved_f32(&expected);
    let logical_per_batch = config.logical_complex_len().unwrap() as u64;
    let batch = config.batch() as u64;
    let strided_input = matches!(
        mode,
        StridedMode::Both | StridedMode::InputOnly | StridedMode::OffsetBoth
    );
    let strided_output = matches!(
        mode,
        StridedMode::Both | StridedMode::OutputOnly | StridedMode::OffsetBoth
    );
    let offset = if matches!(mode, StridedMode::OffsetBoth) {
        aligned_test_offset(context)
    } else {
        0
    };

    let input_layout = if strided_input {
        strided_test_layout(logical_per_batch, 3)
    } else {
        BufferLayout::contiguous()
    };
    let input_span_complex = if strided_input {
        layout_span_complex(input_layout, logical_per_batch, batch)
    } else {
        input.len() as u64 / 2
    };
    let input_span_bytes = input_span_complex * 8;
    let input_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.strided_input"),
        size: offset + input_span_bytes + offset,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    if strided_input {
        let physical_input =
            scatter_strided_interleaved(&input, input_layout, logical_per_batch, batch);
        context
            .queue
            .write_buffer(&input_buffer, offset, bytemuck::cast_slice(&physical_input));
    } else {
        context
            .queue
            .write_buffer(&input_buffer, offset, bytemuck::cast_slice(&input));
    }

    let output_layout = if strided_output {
        strided_test_layout(logical_per_batch, 5)
    } else {
        BufferLayout::contiguous()
    };
    let output_span_complex = if strided_output {
        layout_span_complex(output_layout, logical_per_batch, batch)
    } else {
        expected.len() as u64 / 2
    };
    let output_span_bytes = output_span_complex * 8;
    let output_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.strided_output"),
        size: offset + output_span_bytes + offset,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.strided_readback"),
        size: output_span_bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let plan = FftPlan::c2c(&context.device, &context.queue, config).unwrap();
    assert_eq!(plan.route(), expected_route);
    let input_view = BufferView::new(&input_buffer, offset, input_span_bytes).unwrap();
    let output_view = BufferView::new(&output_buffer, offset, output_span_bytes).unwrap();
    let input_io = if strided_input {
        FftIoView::new(input_view, input_layout).unwrap()
    } else {
        FftIoView::contiguous(input_view)
    };
    let output_io = if strided_output {
        FftIoView::new(output_view, output_layout).unwrap()
    } else {
        FftIoView::contiguous(output_view)
    };
    let io_diagnostics =
        plan.diagnostics_for_io_views(&context.device, input_io.clone(), output_io.clone());
    if strided_input {
        assert!(io_diagnostics
            .stages()
            .iter()
            .any(|stage| stage.kind == "strided-pack"));
    }
    if strided_output {
        assert!(io_diagnostics
            .stages()
            .iter()
            .any(|stage| stage.kind == "strided-unpack"));
    }
    let forced_io_limits = FftDeviceLimits {
        max_storage_buffer_binding_size: 32,
        max_buffer_size: 32,
        min_storage_buffer_offset_alignment: 1,
    };
    let forced_io_diagnostics = plan.diagnostics_for_io_views_with_limits(
        forced_io_limits,
        input_io.clone(),
        output_io.clone(),
    );
    assert_eq!(
        forced_io_diagnostics.device_limits(),
        Some(forced_io_limits)
    );
    if strided_input {
        assert!(forced_io_diagnostics.blockers().iter().any(|blocker| {
            blocker.kind == FftBlockerKind::HelperBuffer
                && blocker.route.as_deref() == Some(c2c_route_name(expected_route))
                && blocker.stage.as_deref() == Some("input-strided-pack")
                && blocker.helper_buffer.as_deref() == Some("input-logical-stage")
        }));
    }
    if strided_output {
        assert!(forced_io_diagnostics.blockers().iter().any(|blocker| {
            blocker.kind == FftBlockerKind::HelperBuffer
                && blocker.route.as_deref() == Some(c2c_route_name(expected_route))
                && blocker.stage.as_deref() == Some("output-strided-unpack")
                && blocker.helper_buffer.as_deref() == Some("output-logical-stage")
        }));
    }

    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("wgpu_fft.test.strided_encoder"),
        });
    plan.execute_io_views(&context.device, &mut encoder, input_io, output_io)
        .unwrap();
    encoder.copy_buffer_to_buffer(
        &output_buffer,
        offset,
        &readback_buffer,
        0,
        output_span_bytes,
    );
    context.queue.submit([encoder.finish()]);

    let label = format!("strided {:?} {:?}", plan.config(), mode);
    trace_gpu_step(&format!("submitted {label}"));
    let actual_physical = read_interleaved_f32(context, &readback_buffer, &label);
    let actual = if strided_output {
        gather_strided_interleaved(&actual_physical, output_layout, logical_per_batch, batch)
    } else {
        actual_physical
    };
    assert_close(&actual, &expected, &label);
    trace_gpu_step(&format!("finish {label}"));
}

fn run_one_case_with_segmented_strided_layout(context: &wgpu_fft::device::GpuContext) {
    let config = FftConfig::new(16).with_normalization(Normalization::None);
    trace_gpu_step(&format!("start segmented+strided config={config:?}"));
    let input = input_for_config(&config);
    let expected = reference_c2c_nd(&from_interleaved_f32(&input), &config).unwrap();
    let expected = to_interleaved_f32(&expected);
    let logical_per_batch = config.logical_complex_len().unwrap() as u64;
    let input_layout = strided_test_layout(logical_per_batch, 3);
    let output_layout = strided_test_layout(logical_per_batch, 5);
    let input_span_bytes = layout_span_complex(input_layout, logical_per_batch, 1) * 8;
    let output_span_bytes = layout_span_complex(output_layout, logical_per_batch, 1) * 8;
    let physical_input = scatter_strided_interleaved(&input, input_layout, logical_per_batch, 1);

    let input_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.segmented_strided_logical_input"),
        size: input_span_bytes,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    context
        .queue
        .write_buffer(&input_buffer, 0, bytemuck::cast_slice(&physical_input));
    let output_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.segmented_strided_logical_output"),
        size: output_span_bytes,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let readback_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.segmented_strided_logical_readback"),
        size: output_span_bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let input_segments = split_buffer_segments(&input_buffer, input_span_bytes, 4);
    let output_segments = split_buffer_segments(&output_buffer, output_span_bytes, 4);
    let input_view = BufferView::from_segments(&input_segments, 0, input_span_bytes).unwrap();
    let output_view = BufferView::from_segments(&output_segments, 0, output_span_bytes).unwrap();
    let plan = FftPlan::c2c(&context.device, &context.queue, config).unwrap();
    assert_eq!(plan.route(), C2cRoute::MixedRadix);
    assert!(!plan.diagnostics().stages().is_empty());
    let input_logical =
        FftLogicalView::new(input_view, logical_layout_from_c2c(input_layout)).unwrap();
    let output_logical =
        FftLogicalView::new(output_view, logical_layout_from_c2c(output_layout)).unwrap();
    let logical_diagnostics =
        plan.diagnostics_for_logical_views(&context.device, &input_logical, &output_logical);
    assert!(logical_diagnostics
        .stages()
        .iter()
        .any(|stage| stage.kind == "strided-pack"));
    assert!(logical_diagnostics
        .stages()
        .iter()
        .any(|stage| stage.kind == "strided-unpack"));

    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("wgpu_fft.test.segmented_strided_logical_encoder"),
        });
    plan.execute_logical_views(&context.device, &mut encoder, input_logical, output_logical)
        .unwrap();
    encoder.copy_buffer_to_buffer(&output_buffer, 0, &readback_buffer, 0, output_span_bytes);
    context.queue.submit([encoder.finish()]);

    let label = "segmented+strided logical c2c";
    trace_gpu_step(&format!("submitted {label}"));
    let actual_physical = read_interleaved_f32(context, &readback_buffer, label);
    let actual = gather_strided_interleaved(&actual_physical, output_layout, logical_per_batch, 1);
    assert_close(&actual, &expected, label);
    trace_gpu_step(&format!("finish {label}"));
}

#[derive(Debug, Clone, Copy)]
enum LargeChunkViewMode {
    Direct,
    Offset,
    Segmented,
    IoContiguous,
    Strided,
}

fn run_large_chunk_cases(context: &wgpu_fft::device::GpuContext) {
    for (config, route) in [
        (
            FftConfig::new(16)
                .with_batch(5)
                .with_normalization(Normalization::None),
            C2cRoute::MixedRadix,
        ),
        (
            FftConfig::new(17)
                .with_batch(5)
                .with_normalization(Normalization::None),
            C2cRoute::Rader,
        ),
        (
            FftConfig::new(34)
                .with_batch(5)
                .with_normalization(Normalization::None),
            C2cRoute::Bluestein,
        ),
        (
            FftConfig::new_nd([17, 4])
                .with_tuning(per_axis_tuning())
                .with_batch(5)
                .with_normalization(Normalization::None),
            C2cRoute::AxisSequence,
        ),
    ] {
        run_one_case_with_large_chunk(context, config, route, LargeChunkViewMode::Direct);
    }

    for mode in [
        LargeChunkViewMode::Offset,
        LargeChunkViewMode::Segmented,
        LargeChunkViewMode::IoContiguous,
        LargeChunkViewMode::Strided,
    ] {
        run_one_case_with_large_chunk(
            context,
            FftConfig::new(16)
                .with_batch(5)
                .with_normalization(Normalization::None),
            C2cRoute::MixedRadix,
            mode,
        );
    }

    assert_large_chunk_validation_behavior(context);
}

fn run_one_case_with_large_chunk(
    context: &wgpu_fft::device::GpuContext,
    config: FftConfig,
    expected_route: C2cRoute,
    mode: LargeChunkViewMode,
) {
    trace_gpu_step(&format!(
        "start large-chunk config={config:?} route={expected_route:?} mode={mode:?}"
    ));
    let input = input_for_config(&config);
    let expected = reference_c2c_nd(&from_interleaved_f32(&input), &config).unwrap();
    let expected = to_interleaved_f32(&expected);
    let byte_len = (input.len() * std::mem::size_of::<f32>()) as u64;
    let batch = config.batch() as u64;
    let logical_per_batch = config.logical_complex_len().unwrap() as u64;
    let input_layout = strided_test_layout(logical_per_batch, 3);
    let output_layout = strided_test_layout(logical_per_batch, 5);
    let strided_input_bytes = layout_span_complex(input_layout, logical_per_batch, batch) * 8;
    let strided_output_bytes = layout_span_complex(output_layout, logical_per_batch, batch) * 8;
    let input_view_bytes = if matches!(mode, LargeChunkViewMode::Strided) {
        strided_input_bytes
    } else {
        byte_len
    };
    let output_view_bytes = if matches!(mode, LargeChunkViewMode::Strided) {
        strided_output_bytes
    } else {
        byte_len
    };
    let offset = if matches!(mode, LargeChunkViewMode::Offset) {
        aligned_test_offset(context)
    } else {
        0
    };
    let input_backing_size = offset + input_view_bytes + offset;
    let output_backing_size = offset + output_view_bytes + offset;

    let input_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.large_chunk_input"),
        size: input_backing_size,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let physical_input = if matches!(mode, LargeChunkViewMode::Strided) {
        scatter_strided_interleaved(&input, input_layout, logical_per_batch, batch)
    } else {
        input.clone()
    };
    context
        .queue
        .write_buffer(&input_buffer, offset, bytemuck::cast_slice(&physical_input));

    let output_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.large_chunk_output"),
        size: output_backing_size,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let readback_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.large_chunk_readback"),
        size: output_view_bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let limits = large_chunk_test_limits(&config, 2);
    let plan = FftPlan::c2c_with_large_policy_limits_for_testing(
        &context.device,
        &context.queue,
        config,
        limits,
    )
    .unwrap();
    assert_eq!(plan.route(), expected_route);
    assert_eq!(
        plan.large_routing_policy().route_mode(),
        LargeRouteMode::LargeChunk
    );
    assert_eq!(plan.workspace_size_bytes(), 0);

    let input_view = if matches!(mode, LargeChunkViewMode::Segmented) {
        BufferView::from_segments(
            &split_buffer_segments(&input_buffer, byte_len, 4),
            0,
            byte_len,
        )
        .unwrap()
    } else {
        BufferView::new(&input_buffer, offset, input_view_bytes).unwrap()
    };
    let output_view = if matches!(mode, LargeChunkViewMode::Segmented) {
        BufferView::from_segments(
            &split_buffer_segments(&output_buffer, byte_len, 4),
            0,
            byte_len,
        )
        .unwrap()
    } else {
        BufferView::new(&output_buffer, offset, output_view_bytes).unwrap()
    };

    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("wgpu_fft.test.large_chunk_encoder"),
        });
    if matches!(mode, LargeChunkViewMode::IoContiguous) {
        plan.execute_io_views(
            &context.device,
            &mut encoder,
            FftIoView::contiguous(input_view),
            FftIoView::contiguous(output_view),
        )
        .unwrap();
    } else if matches!(mode, LargeChunkViewMode::Strided) {
        let input_logical =
            FftLogicalView::new(input_view, logical_layout_from_c2c(input_layout)).unwrap();
        let output_logical =
            FftLogicalView::new(output_view, logical_layout_from_c2c(output_layout)).unwrap();
        let diagnostics =
            plan.diagnostics_for_logical_views(&context.device, &input_logical, &output_logical);
        assert!(diagnostics.blockers().is_empty());
        assert!(diagnostics
            .stages()
            .iter()
            .any(|stage| stage.kind == "strided-pack"));
        assert!(diagnostics
            .stages()
            .iter()
            .any(|stage| stage.kind == "strided-unpack"));
        plan.execute_logical_views(&context.device, &mut encoder, input_logical, output_logical)
            .unwrap();
    } else {
        plan.execute_views(&context.device, &mut encoder, input_view, output_view)
            .unwrap();
    }
    encoder.copy_buffer_to_buffer(
        &output_buffer,
        offset,
        &readback_buffer,
        0,
        output_view_bytes,
    );
    context.queue.submit([encoder.finish()]);

    let label = format!("large-chunk {:?} {:?}", plan.config(), mode);
    trace_gpu_step(&format!("submitted {label}"));
    let actual_physical = read_interleaved_f32(context, &readback_buffer, &label);
    let actual = if matches!(mode, LargeChunkViewMode::Strided) {
        gather_strided_interleaved(&actual_physical, output_layout, logical_per_batch, batch)
    } else {
        actual_physical
    };
    assert_close(&actual, &expected, &label);
    trace_gpu_step(&format!("finish {label}"));
}

fn assert_large_chunk_validation_behavior(context: &wgpu_fft::device::GpuContext) {
    trace_gpu_step("start large-chunk validation behavior");
    let config = FftConfig::new(16)
        .with_batch(5)
        .with_normalization(Normalization::None);
    let limits = large_chunk_test_limits(&config, 2);
    let plan = FftPlan::c2c_with_large_policy_limits_for_testing(
        &context.device,
        &context.queue,
        config.clone(),
        limits,
    )
    .unwrap();
    let required = plan.required_buffer_size_bytes();
    let input_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.large_chunk_validation_input"),
        size: required,
        usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let output_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.large_chunk_validation_output"),
        size: required,
        usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let workspace = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.large_chunk_workspace"),
        size: 4,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("wgpu_fft.test.large_chunk_validation_encoder"),
        });
    assert_eq!(
        plan.execute_views_with_workspace(
            &context.device,
            &mut encoder,
            BufferView::whole(&input_buffer),
            BufferView::whole(&output_buffer),
            BufferView::whole(&workspace),
        ),
        Err(FftError::LargeRouteWorkspaceUnsupported {
            route_mode: "large-chunk",
        })
    );
    let assert_large_workspace_blocker = |diagnostics: wgpu_fft::FftDiagnostics| {
        assert!(diagnostics.blockers().iter().any(|blocker| {
            blocker.kind == FftBlockerKind::Workspace
                && blocker.route.as_deref() == Some("large-chunk")
                && blocker.stage.as_deref() == Some("workspace")
                && blocker.helper_buffer.as_deref() == Some("workspace")
        }));
    };
    assert_large_workspace_blocker(plan.diagnostics_for_views_with_workspace(
        &context.device,
        BufferView::whole(&input_buffer),
        BufferView::whole(&output_buffer),
        BufferView::whole(&workspace),
    ));
    assert_large_workspace_blocker(plan.diagnostics_for_io_views_with_workspace(
        &context.device,
        FftIoView::contiguous(BufferView::whole(&input_buffer)),
        FftIoView::contiguous(BufferView::whole(&output_buffer)),
        BufferView::whole(&workspace),
    ));
    assert_large_workspace_blocker(plan.diagnostics_for_logical_views_with_workspace(
        &context.device,
        &FftLogicalView::contiguous(BufferView::whole(&input_buffer)),
        &FftLogicalView::contiguous(BufferView::whole(&output_buffer)),
        BufferView::whole(&workspace),
    ));

    let span_bytes = layout_span_complex(BufferLayout::new(0, 2).unwrap(), 16, 5) * 8;
    let strided_input = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.large_chunk_strided_input"),
        size: span_bytes,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let strided_output = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.large_chunk_strided_output"),
        size: required,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let diagnostics = plan.diagnostics_for_logical_views(
        &context.device,
        &FftLogicalView::new(
            BufferView::whole(&strided_input),
            FftLogicalLayout::new(0, 2).unwrap(),
        )
        .unwrap(),
        &FftLogicalView::contiguous(BufferView::whole(&strided_output)),
    );
    assert!(diagnostics
        .stages()
        .iter()
        .any(|stage| stage.kind == "strided-pack"));
    assert!(!diagnostics
        .blockers()
        .iter()
        .any(|blocker| blocker.layout.as_deref() == Some("strided")));

    let smooth_limits = LargePolicyLimits {
        max_storage_buffer_binding_size: 64,
        max_buffer_size: 1 << 30,
    };
    let smooth_plan = FftPlan::c2c_with_large_policy_limits_for_testing(
        &context.device,
        &context.queue,
        config,
        smooth_limits,
    )
    .unwrap();
    assert_eq!(
        smooth_plan.large_routing_policy().execution_kind(),
        LargeExecutionKind::Smooth1dDecomposition
    );
    assert_eq!(
        smooth_plan
            .large_routing_policy()
            .diagnostics()
            .selected_axis,
        Some(0)
    );
    trace_gpu_step("finish large-chunk validation behavior");
}

fn run_smooth_decomposition_cases(context: &wgpu_fft::device::GpuContext) {
    for (config, expected_kind, abs_tolerance) in [
        (
            FftConfig::new(64).with_normalization(Normalization::None),
            LargeExecutionKind::Smooth1dDecomposition,
            1.0e-2,
        ),
        (
            FftConfig::new(64)
                .with_batch(2)
                .with_normalization(Normalization::None),
            LargeExecutionKind::Smooth1dDecomposition,
            5.0e-2,
        ),
        (
            FftConfig::inverse(64),
            LargeExecutionKind::Smooth1dDecomposition,
            1.0e-2,
        ),
        (
            FftConfig::new(64).with_normalization(Normalization::Forward),
            LargeExecutionKind::Smooth1dDecomposition,
            1.0e-2,
        ),
        (
            FftConfig::new_nd([64, 2]).with_normalization(Normalization::None),
            LargeExecutionKind::AxisDecomposition,
            1.0e-2,
        ),
        (
            FftConfig::new_nd([2, 64])
                .with_axes([1])
                .with_normalization(Normalization::None),
            LargeExecutionKind::AxisDecomposition,
            1.0e-2,
        ),
        (
            FftConfig::new_nd([16, 16]).with_normalization(Normalization::None),
            LargeExecutionKind::OutOfCoreFourStep,
            1.0e-2,
        ),
        (
            FftConfig::new_nd([64, 64]).with_normalization(Normalization::None),
            LargeExecutionKind::AxisDecomposition,
            5.0e-2,
        ),
    ] {
        run_one_case_with_smooth_decomposition(
            context,
            config,
            false,
            expected_kind,
            abs_tolerance,
        );
    }
    run_one_case_with_smooth_decomposition(
        context,
        FftConfig::new(64).with_normalization(Normalization::None),
        true,
        LargeExecutionKind::Smooth1dDecomposition,
        1.0e-2,
    );
    run_one_case_with_recursive_smooth_decomposition(context);
    run_one_case_with_smooth_decomposition_strided_io(context);
    run_one_case_with_smooth_decomposition_segmented_io(context);
    assert_smooth_decomposition_batched_diagnostics(context);
    assert_smooth_decomposition_multi_axis_diagnostics(context);
}

fn run_one_case_with_smooth_decomposition(
    context: &wgpu_fft::device::GpuContext,
    config: FftConfig,
    offset_views: bool,
    expected_kind: LargeExecutionKind,
    abs_tolerance: f32,
) {
    trace_gpu_step(&format!(
        "start smooth-decomposition config={config:?} offset_views={offset_views}"
    ));
    let input = input_for_config(&config);
    let expected = reference_c2c_nd(&from_interleaved_f32(&input), &config).unwrap();
    let expected = to_interleaved_f32(&expected);
    let byte_len = (input.len() * std::mem::size_of::<f32>()) as u64;
    let offset = if offset_views {
        aligned_test_offset(context)
    } else {
        0
    };
    let backing_size = offset + byte_len + offset;
    let input_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.smooth_decomposition_input"),
        size: backing_size,
        usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    context
        .queue
        .write_buffer(&input_buffer, offset, bytemuck::cast_slice(&input));
    let output_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.smooth_decomposition_output"),
        size: backing_size,
        usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let readback_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.smooth_decomposition_readback"),
        size: byte_len,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let plan = FftPlan::c2c_with_large_policy_limits_for_testing(
        &context.device,
        &context.queue,
        config,
        smooth_decomposition_test_limits(byte_len),
    )
    .unwrap();
    assert_eq!(plan.route(), C2cRoute::MixedRadix);
    let expected_route_mode = if expected_kind == LargeExecutionKind::OutOfCoreFourStep {
        LargeRouteMode::LargeOutOfCore
    } else {
        LargeRouteMode::LargeChunk
    };
    assert_eq!(
        plan.large_routing_policy().route_mode(),
        expected_route_mode
    );
    assert_eq!(plan.large_routing_policy().execution_kind(), expected_kind);

    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("wgpu_fft.test.smooth_decomposition_encoder"),
        });
    plan.execute_views(
        &context.device,
        &mut encoder,
        BufferView::new(&input_buffer, offset, byte_len).unwrap(),
        BufferView::new(&output_buffer, offset, byte_len).unwrap(),
    )
    .unwrap();
    encoder.copy_buffer_to_buffer(&output_buffer, offset, &readback_buffer, 0, byte_len);
    context.queue.submit([encoder.finish()]);

    let label = format!("smooth-decomposition {:?}", plan.config());
    trace_gpu_step(&format!("submitted {label}"));
    let actual = read_interleaved_f32(context, &readback_buffer, &label);
    assert_close_with_abs_tolerance(&actual, &expected, &label, abs_tolerance);
    trace_gpu_step(&format!("finish {label}"));
}

fn run_one_case_with_recursive_smooth_decomposition(context: &wgpu_fft::device::GpuContext) {
    trace_gpu_step("start recursive smooth-decomposition");
    let config = FftConfig::new(1024).with_normalization(Normalization::None);
    let input = input_for_config(&config);
    let expected = reference_c2c_nd(&from_interleaved_f32(&input), &config).unwrap();
    let expected = to_interleaved_f32(&expected);
    let byte_len = (input.len() * std::mem::size_of::<f32>()) as u64;
    let input_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.recursive_smooth_decomposition_input"),
        size: byte_len,
        usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    context
        .queue
        .write_buffer(&input_buffer, 0, bytemuck::cast_slice(&input));
    let output_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.recursive_smooth_decomposition_output"),
        size: byte_len,
        usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let readback_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.recursive_smooth_decomposition_readback"),
        size: byte_len,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let plan = FftPlan::c2c_with_large_policy_limits_for_testing(
        &context.device,
        &context.queue,
        config,
        LargePolicyLimits {
            max_storage_buffer_binding_size: 128,
            max_buffer_size: byte_len,
        },
    )
    .unwrap();
    assert_eq!(
        plan.large_routing_policy().execution_kind(),
        LargeExecutionKind::Smooth1dDecomposition
    );
    assert_eq!(
        plan.large_routing_policy().diagnostics().selected_axis,
        Some(0)
    );

    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("wgpu_fft.test.recursive_smooth_decomposition_encoder"),
        });
    plan.execute_views(
        &context.device,
        &mut encoder,
        BufferView::whole(&input_buffer),
        BufferView::whole(&output_buffer),
    )
    .unwrap();
    encoder.copy_buffer_to_buffer(&output_buffer, 0, &readback_buffer, 0, byte_len);
    context.queue.submit([encoder.finish()]);

    let label = "recursive smooth-decomposition";
    trace_gpu_step(&format!("submitted {label}"));
    let actual = read_interleaved_f32(context, &readback_buffer, label);
    assert_close_with_abs_tolerance(&actual, &expected, label, 5.0e-2);
    trace_gpu_step(&format!("finish {label}"));
}

fn run_one_case_with_smooth_decomposition_segmented_io(context: &wgpu_fft::device::GpuContext) {
    trace_gpu_step("start smooth-decomposition segmented io");
    let config = FftConfig::new(64).with_normalization(Normalization::None);
    let input = input_for_config(&config);
    let expected = reference_c2c_nd(&from_interleaved_f32(&input), &config).unwrap();
    let expected = to_interleaved_f32(&expected);
    let byte_len = config.required_buffer_size_bytes().unwrap();
    let plan = FftPlan::c2c_with_large_policy_limits_for_testing(
        &context.device,
        &context.queue,
        config,
        smooth_decomposition_test_limits(byte_len),
    )
    .unwrap();
    let input_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.smooth_decomposition_segmented_input"),
        size: byte_len,
        usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    context
        .queue
        .write_buffer(&input_buffer, 0, bytemuck::cast_slice(&input));
    let output_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.smooth_decomposition_segmented_output"),
        size: byte_len,
        usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let readback_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.smooth_decomposition_segmented_readback"),
        size: byte_len,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("wgpu_fft.test.smooth_decomposition_validation_encoder"),
        });
    plan.execute_views(
        &context.device,
        &mut encoder,
        BufferView::from_segments(
            &split_buffer_segments(&input_buffer, byte_len, 2),
            0,
            byte_len,
        )
        .unwrap(),
        BufferView::from_segments(
            &split_buffer_segments(&output_buffer, byte_len, 2),
            0,
            byte_len,
        )
        .unwrap(),
    )
    .unwrap();
    encoder.copy_buffer_to_buffer(&output_buffer, 0, &readback_buffer, 0, byte_len);
    context.queue.submit([encoder.finish()]);

    let label = "smooth-decomposition segmented io";
    trace_gpu_step(&format!("submitted {label}"));
    let actual = read_interleaved_f32(context, &readback_buffer, label);
    assert_close(&actual, &expected, label);
    trace_gpu_step(&format!("finish {label}"));
}

fn run_one_case_with_smooth_decomposition_strided_io(context: &wgpu_fft::device::GpuContext) {
    trace_gpu_step("start smooth-decomposition strided io");
    let config = FftConfig::new(64).with_normalization(Normalization::None);
    let input = input_for_config(&config);
    let expected = reference_c2c_nd(&from_interleaved_f32(&input), &config).unwrap();
    let expected = to_interleaved_f32(&expected);
    let logical_per_batch = config.logical_complex_len().unwrap() as u64;
    let input_layout = strided_test_layout(logical_per_batch, 3);
    let output_layout = strided_test_layout(logical_per_batch, 5);
    let input_span_bytes = layout_span_complex(input_layout, logical_per_batch, 1) * 8;
    let output_span_bytes = layout_span_complex(output_layout, logical_per_batch, 1) * 8;
    let input_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.smooth_decomposition_strided_input"),
        size: input_span_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let physical_input = scatter_strided_interleaved(&input, input_layout, logical_per_batch, 1);
    context
        .queue
        .write_buffer(&input_buffer, 0, bytemuck::cast_slice(&physical_input));
    let output_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.smooth_decomposition_strided_output"),
        size: output_span_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.smooth_decomposition_strided_readback"),
        size: output_span_bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let plan = FftPlan::c2c_with_large_policy_limits_for_testing(
        &context.device,
        &context.queue,
        config,
        smooth_decomposition_test_limits((input.len() * std::mem::size_of::<f32>()) as u64),
    )
    .unwrap();
    assert_eq!(
        plan.large_routing_policy().execution_kind(),
        LargeExecutionKind::Smooth1dDecomposition
    );

    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("wgpu_fft.test.smooth_decomposition_strided_encoder"),
        });
    plan.execute_io_views(
        &context.device,
        &mut encoder,
        FftIoView::new(BufferView::whole(&input_buffer), input_layout).unwrap(),
        FftIoView::new(BufferView::whole(&output_buffer), output_layout).unwrap(),
    )
    .unwrap();
    encoder.copy_buffer_to_buffer(&output_buffer, 0, &readback_buffer, 0, output_span_bytes);
    context.queue.submit([encoder.finish()]);

    let label = "smooth-decomposition strided io";
    trace_gpu_step(&format!("submitted {label}"));
    let actual_physical = read_interleaved_f32(context, &readback_buffer, label);
    let actual = gather_strided_interleaved(&actual_physical, output_layout, logical_per_batch, 1);
    assert_close(&actual, &expected, label);
    trace_gpu_step(&format!("finish {label}"));
}

fn assert_smooth_decomposition_multi_axis_diagnostics(context: &wgpu_fft::device::GpuContext) {
    trace_gpu_step("start smooth-decomposition multi-axis diagnostics");
    let byte_len = FftConfig::new_nd([64, 64])
        .with_normalization(Normalization::None)
        .required_buffer_size_bytes()
        .unwrap();
    let plan = FftPlan::c2c_with_large_policy_limits_for_testing(
        &context.device,
        &context.queue,
        FftConfig::new_nd([64, 64]).with_normalization(Normalization::None),
        smooth_decomposition_test_limits(byte_len),
    )
    .unwrap();
    assert_eq!(
        plan.large_routing_policy().execution_kind(),
        LargeExecutionKind::AxisDecomposition
    );
    let route_diagnostics = plan.large_routing_policy().diagnostics();
    assert_eq!(route_diagnostics.factor_splits.len(), 2);
    assert!(route_diagnostics
        .factor_splits
        .iter()
        .all(|split| split.factors.len() == 2));
    assert!(route_diagnostics.staging_bytes.contains(&byte_len));
    let diagnostics = plan.diagnostics_for_device(&context.device);
    assert!(diagnostics.blockers().is_empty());
    let smooth_stage_count = diagnostics
        .stages()
        .iter()
        .filter(|stage| stage.label.contains("smooth-axis"))
        .count();
    assert!(smooth_stage_count >= 2);
    trace_gpu_step("finish smooth-decomposition multi-axis diagnostics");
}

fn assert_smooth_decomposition_batched_diagnostics(context: &wgpu_fft::device::GpuContext) {
    trace_gpu_step("start smooth-decomposition batched diagnostics");
    let config = FftConfig::new(64)
        .with_batch(2)
        .with_normalization(Normalization::None);
    let byte_len = config.required_buffer_size_bytes().unwrap();
    let plan = FftPlan::c2c_with_large_policy_limits_for_testing(
        &context.device,
        &context.queue,
        config,
        smooth_decomposition_test_limits(byte_len),
    )
    .unwrap();
    assert_eq!(
        plan.large_routing_policy().execution_kind(),
        LargeExecutionKind::Smooth1dDecomposition
    );
    let route_diagnostics = plan.large_routing_policy().diagnostics();
    assert_eq!(route_diagnostics.selected_axis, Some(0));
    assert_eq!(route_diagnostics.factor_splits.len(), 1);
    assert!(!route_diagnostics.staging_bytes.contains(&byte_len));
    let diagnostics = plan.diagnostics_for_device(&context.device);
    assert!(diagnostics.blockers().is_empty());
    assert!(diagnostics
        .stages()
        .iter()
        .any(|stage| stage.label.contains("smooth-axis-gather-phase1")));
    trace_gpu_step("finish smooth-decomposition batched diagnostics");
}

fn run_large_bridge_cases(context: &wgpu_fft::device::GpuContext) {
    for (config, route, kind) in [
        (
            FftConfig::new(17).with_normalization(Normalization::None),
            C2cRoute::Rader,
            LargeExecutionKind::RaderBridge,
        ),
        (
            FftConfig::new(34).with_normalization(Normalization::None),
            C2cRoute::Bluestein,
            LargeExecutionKind::BluesteinBridge,
        ),
    ] {
        for mode in [
            LargeBridgeViewMode::Direct,
            LargeBridgeViewMode::Offset,
            LargeBridgeViewMode::Segmented,
            LargeBridgeViewMode::Strided,
            LargeBridgeViewMode::SegmentedStrided,
        ] {
            run_one_case_with_large_bridge(context, config.clone(), route, kind, mode);
        }
    }

    for (config, route, kind) in [
        (
            FftConfig::new(17)
                .with_batch(2)
                .with_normalization(Normalization::None),
            C2cRoute::Rader,
            LargeExecutionKind::RaderBridge,
        ),
        (
            FftConfig::new(34)
                .with_batch(2)
                .with_normalization(Normalization::None),
            C2cRoute::Bluestein,
            LargeExecutionKind::BluesteinBridge,
        ),
    ] {
        run_one_case_with_large_bridge(context, config, route, kind, LargeBridgeViewMode::Direct);
    }

    for (config, route, kind) in [
        (
            FftConfig::inverse(17),
            C2cRoute::Rader,
            LargeExecutionKind::RaderBridge,
        ),
        (
            FftConfig::inverse(34),
            C2cRoute::Bluestein,
            LargeExecutionKind::BluesteinBridge,
        ),
    ] {
        run_one_case_with_large_bridge(context, config, route, kind, LargeBridgeViewMode::Direct);
    }
}

fn run_large_axis_sequence_cases(context: &wgpu_fft::device::GpuContext) {
    for config in [
        FftConfig::new_nd([17, 4])
            .with_tuning(per_axis_tuning())
            .with_normalization(Normalization::None),
        FftConfig::new_nd([34, 16]).with_normalization(Normalization::None),
        FftConfig::inverse_nd([17, 4]),
        FftConfig::inverse_nd([17, 4]).with_batch(2),
        FftConfig::inverse_nd([34, 16]),
    ] {
        trace_gpu_step(&format!("start large-axis-sequence config={config:?}"));
        let input = input_for_config(&config);
        let expected = reference_c2c_nd(&from_interleaved_f32(&input), &config).unwrap();
        let expected = to_interleaved_f32(&expected);
        let byte_len = (input.len() * std::mem::size_of::<f32>()) as u64;
        let input_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("wgpu_fft.test.large_axis_sequence_input"),
            size: byte_len,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        context
            .queue
            .write_buffer(&input_buffer, 0, bytemuck::cast_slice(&input));
        let output_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("wgpu_fft.test.large_axis_sequence_output"),
            size: byte_len,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let readback_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("wgpu_fft.test.large_axis_sequence_readback"),
            size: byte_len,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let plan = FftPlan::c2c_with_large_policy_limits_for_testing(
            &context.device,
            &context.queue,
            config,
            LargePolicyLimits {
                max_storage_buffer_binding_size: 64,
                max_buffer_size: 1 << 20,
            },
        )
        .unwrap();
        assert_eq!(plan.route(), C2cRoute::AxisSequence);
        assert_eq!(
            plan.large_routing_policy().route_mode(),
            LargeRouteMode::LargeChunk
        );
        assert_eq!(
            plan.large_routing_policy().execution_kind(),
            LargeExecutionKind::AxisDecomposition
        );
        let diagnostics = plan.diagnostics_for_device(&context.device);
        assert!(diagnostics.blockers().is_empty());
        assert!(diagnostics
            .stages()
            .iter()
            .any(|stage| stage.label == "large-axis-sequence-workspace"
                && stage.route == "axis-sequence"));
        assert!(diagnostics.stages().iter().any(|stage| {
            (stage.label == "rader-bridge-pack" && stage.route == "rader-bridge")
                || (stage.label == "bluestein-bridge-pack" && stage.route == "bluestein-bridge")
        }));
        let forced_limits = FftDeviceLimits {
            max_storage_buffer_binding_size: 32,
            max_buffer_size: 1 << 20,
            min_storage_buffer_offset_alignment: 1,
        };
        let forced_diagnostics = plan.diagnostics_for_limits(forced_limits);
        assert_eq!(forced_diagnostics.device_limits(), Some(forced_limits));
        assert!(forced_diagnostics.blockers().iter().any(|blocker| {
            blocker.kind == FftBlockerKind::DeviceLimit
                && blocker.route.as_deref() == Some("axis-sequence")
                && blocker.stage.as_deref() == Some("large-axis-sequence-workspace")
        }));
        assert!(forced_diagnostics.blockers().iter().any(|blocker| {
            blocker.kind == FftBlockerKind::DeviceLimit
                && matches!(blocker.route.as_deref(), Some("rader" | "bluestein"))
                && blocker
                    .stage
                    .as_deref()
                    .is_some_and(|stage| stage.ends_with("-work-helper"))
        }));

        let mut encoder = context
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("wgpu_fft.test.large_axis_sequence_encoder"),
            });
        plan.execute_views(
            &context.device,
            &mut encoder,
            BufferView::whole(&input_buffer),
            BufferView::whole(&output_buffer),
        )
        .unwrap();
        encoder.copy_buffer_to_buffer(&output_buffer, 0, &readback_buffer, 0, byte_len);
        context.queue.submit([encoder.finish()]);

        let label = format!("large-axis-sequence {:?}", plan.config());
        trace_gpu_step(&format!("submitted {label}"));
        let actual = read_interleaved_f32(context, &readback_buffer, &label);
        assert_close(&actual, &expected, &label);
        trace_gpu_step(&format!("finish {label}"));
    }
}

#[derive(Debug, Clone, Copy)]
enum LargeBridgeViewMode {
    Direct,
    Offset,
    Segmented,
    Strided,
    SegmentedStrided,
}

fn run_one_case_with_large_bridge(
    context: &wgpu_fft::device::GpuContext,
    config: FftConfig,
    expected_route: C2cRoute,
    expected_kind: LargeExecutionKind,
    mode: LargeBridgeViewMode,
) {
    trace_gpu_step(&format!(
        "start large-bridge config={config:?} route={expected_route:?} mode={mode:?}"
    ));
    let input = input_for_config(&config);
    let expected = reference_c2c_nd(&from_interleaved_f32(&input), &config).unwrap();
    let expected = to_interleaved_f32(&expected);
    let byte_len = (input.len() * std::mem::size_of::<f32>()) as u64;
    let logical_per_batch = config.logical_complex_len().unwrap() as u64;
    let batch = config.batch() as u64;
    let input_layout = strided_test_layout(logical_per_batch, 3);
    let output_layout = strided_test_layout(logical_per_batch, 5);
    let strided_input_bytes = layout_span_complex(input_layout, logical_per_batch, batch) * 8;
    let strided_output_bytes = layout_span_complex(output_layout, logical_per_batch, batch) * 8;
    let strided = matches!(
        mode,
        LargeBridgeViewMode::Strided | LargeBridgeViewMode::SegmentedStrided
    );
    let input_view_bytes = if strided {
        strided_input_bytes
    } else {
        byte_len
    };
    let output_view_bytes = if strided {
        strided_output_bytes
    } else {
        byte_len
    };
    let offset = if matches!(mode, LargeBridgeViewMode::Offset) {
        aligned_test_offset(context)
    } else {
        0
    };
    let input_backing_size = offset + input_view_bytes + offset;
    let output_backing_size = offset + output_view_bytes + offset;
    let input_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.large_bridge_input"),
        size: input_backing_size,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let physical_input = if strided {
        scatter_strided_interleaved(&input, input_layout, logical_per_batch, batch)
    } else {
        input.clone()
    };
    context
        .queue
        .write_buffer(&input_buffer, offset, bytemuck::cast_slice(&physical_input));
    let output_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.large_bridge_output"),
        size: output_backing_size,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let readback_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.large_bridge_readback"),
        size: output_view_bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let plan = FftPlan::c2c_with_large_policy_limits_for_testing(
        &context.device,
        &context.queue,
        config,
        LargePolicyLimits {
            max_storage_buffer_binding_size: 64,
            max_buffer_size: 1 << 20,
        },
    )
    .unwrap();
    assert_eq!(plan.route(), expected_route);
    assert_eq!(
        plan.large_routing_policy().route_mode(),
        LargeRouteMode::LargeChunk
    );
    assert_eq!(plan.large_routing_policy().execution_kind(), expected_kind);

    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("wgpu_fft.test.large_bridge_encoder"),
        });
    let input_segments = split_buffer_segments(&input_buffer, input_view_bytes, 4);
    let output_segments = split_buffer_segments(&output_buffer, output_view_bytes, 4);
    let segmented = matches!(
        mode,
        LargeBridgeViewMode::Segmented | LargeBridgeViewMode::SegmentedStrided
    );
    let input_view = if segmented {
        BufferView::from_segments(&input_segments, 0, input_view_bytes).unwrap()
    } else {
        BufferView::new(&input_buffer, offset, input_view_bytes).unwrap()
    };
    let output_view = if segmented {
        BufferView::from_segments(&output_segments, 0, output_view_bytes).unwrap()
    } else {
        BufferView::new(&output_buffer, offset, output_view_bytes).unwrap()
    };
    if strided {
        let input_logical =
            FftLogicalView::new(input_view, logical_layout_from_c2c(input_layout)).unwrap();
        let output_logical =
            FftLogicalView::new(output_view, logical_layout_from_c2c(output_layout)).unwrap();
        let diagnostics =
            plan.diagnostics_for_logical_views(&context.device, &input_logical, &output_logical);
        assert!(diagnostics.blockers().is_empty());
        assert!(diagnostics
            .stages()
            .iter()
            .any(|stage| stage.kind == "strided-pack"));
        assert!(diagnostics
            .stages()
            .iter()
            .any(|stage| stage.kind == "strided-unpack"));
        if segmented {
            assert!(diagnostics
                .stages()
                .iter()
                .any(|stage| stage.label == "input-segmented-copy-window"));
            assert!(diagnostics
                .stages()
                .iter()
                .any(|stage| stage.label == "output-segmented-copy-window"));
        }
        plan.execute_logical_views(&context.device, &mut encoder, input_logical, output_logical)
            .unwrap();
    } else {
        let diagnostics =
            plan.diagnostics_for_views(&context.device, input_view.clone(), output_view.clone());
        assert!(diagnostics.blockers().is_empty());
        if segmented {
            assert!(diagnostics
                .stages()
                .iter()
                .any(|stage| stage.label == "input-segmented-copy-window"));
            assert!(diagnostics
                .stages()
                .iter()
                .any(|stage| stage.label == "output-segmented-copy-window"));
        }
        plan.execute_views(&context.device, &mut encoder, input_view, output_view)
            .unwrap();
    }
    encoder.copy_buffer_to_buffer(
        &output_buffer,
        offset,
        &readback_buffer,
        0,
        output_view_bytes,
    );
    context.queue.submit([encoder.finish()]);

    let label = format!("large-bridge {:?} {:?}", plan.config(), mode);
    trace_gpu_step(&format!("submitted {label}"));
    let actual_physical = read_interleaved_f32(context, &readback_buffer, &label);
    let actual = if strided {
        gather_strided_interleaved(&actual_physical, output_layout, logical_per_batch, batch)
    } else {
        actual_physical
    };
    assert_close(&actual, &expected, &label);
    trace_gpu_step(&format!("finish {label}"));
}

fn large_chunk_test_limits(config: &FftConfig, chunk_batches: u64) -> LargePolicyLimits {
    let bytes_per_batch = config.logical_complex_len().unwrap() as u64 * 8;
    LargePolicyLimits {
        max_storage_buffer_binding_size: bytes_per_batch * chunk_batches,
        max_buffer_size: 1 << 30,
    }
}

fn smooth_decomposition_test_limits(byte_len: u64) -> LargePolicyLimits {
    LargePolicyLimits {
        max_storage_buffer_binding_size: 256,
        max_buffer_size: byte_len,
    }
}

fn run_one_case_with_segmented_views(
    context: &wgpu_fft::device::GpuContext,
    config: FftConfig,
    expected_route: C2cRoute,
    segmented_input: bool,
    segmented_output: bool,
) {
    trace_gpu_step(&format!(
        "start segmented config={config:?} route={expected_route:?} input={segmented_input} output={segmented_output}"
    ));
    let input = input_for_config(&config);
    let expected = reference_c2c_nd(&from_interleaved_f32(&input), &config).unwrap();
    let expected = to_interleaved_f32(&expected);
    let byte_len = (input.len() * std::mem::size_of::<f32>()) as u64;

    let input_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.segmented_input"),
        size: byte_len,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    context
        .queue
        .write_buffer(&input_buffer, 0, bytemuck::cast_slice(&input));

    let output_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.segmented_output"),
        size: byte_len,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let readback_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.segmented_readback"),
        size: byte_len,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let plan = FftPlan::c2c(&context.device, &context.queue, config).unwrap();
    assert_eq!(plan.route(), expected_route);
    assert_eq!(
        plan.large_routing_policy().route_mode(),
        LargeRouteMode::Normal
    );

    let input_view = if segmented_input {
        let segments = split_buffer_segments(&input_buffer, byte_len, 4);
        let view = BufferView::from_segments(&segments, 0, byte_len).unwrap();
        assert!(!view.is_single_segment());
        view
    } else {
        BufferView::whole(&input_buffer)
    };
    let output_view = if segmented_output {
        let segments = split_buffer_segments(&output_buffer, byte_len, 4);
        let view = BufferView::from_segments(&segments, 0, byte_len).unwrap();
        assert!(!view.is_single_segment());
        view
    } else {
        BufferView::whole(&output_buffer)
    };

    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("wgpu_fft.test.segmented_encoder"),
        });
    plan.execute_views(&context.device, &mut encoder, input_view, output_view)
        .unwrap();
    encoder.copy_buffer_to_buffer(&output_buffer, 0, &readback_buffer, 0, byte_len);
    context.queue.submit([encoder.finish()]);

    let label = format!(
        "segmented input={segmented_input} output={segmented_output} {:?}",
        plan.config()
    );
    trace_gpu_step(&format!("submitted {label}"));
    let actual = read_interleaved_f32(context, &readback_buffer, &label);
    assert_close(&actual, &expected, &label);
    trace_gpu_step(&format!("finish {label}"));
}

fn assert_segmented_workspace_rejected(context: &wgpu_fft::device::GpuContext) {
    let plan = FftPlan::c2c(
        &context.device,
        &context.queue,
        FftConfig::new_nd([4, 3])
            .with_tuning(per_axis_tuning())
            .with_normalization(Normalization::None),
    )
    .unwrap();
    assert_eq!(plan.route(), C2cRoute::MixedRadix);
    assert!(plan.workspace_size_bytes() > 0);

    let input_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.segmented_workspace_input"),
        size: plan.required_buffer_size_bytes(),
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let output_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.segmented_workspace_output"),
        size: plan.required_buffer_size_bytes(),
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let workspace_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.segmented_workspace"),
        size: plan.workspace_size_bytes(),
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let workspace_segments =
        split_buffer_segments(&workspace_buffer, plan.workspace_size_bytes(), 2);
    let workspace_view =
        BufferView::from_segments(&workspace_segments, 0, plan.workspace_size_bytes()).unwrap();
    let workspace_diagnostics = plan.diagnostics_for_views_with_workspace(
        &context.device,
        BufferView::whole(&input_buffer),
        BufferView::whole(&output_buffer),
        workspace_view.clone(),
    );
    assert!(workspace_diagnostics.blockers().iter().any(|blocker| {
        blocker.kind == FftBlockerKind::Workspace
            && blocker.route.as_deref() == Some("mixed-radix")
            && blocker.stage.as_deref() == Some("workspace")
            && blocker.helper_buffer.as_deref() == Some("workspace")
            && blocker.layout.as_deref() == Some("segmented")
    }));

    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("wgpu_fft.test.segmented_workspace_encoder"),
        });
    assert_eq!(
        plan.execute_views_with_workspace(
            &context.device,
            &mut encoder,
            BufferView::whole(&input_buffer),
            BufferView::whole(&output_buffer),
            workspace_view,
        ),
        Err(FftError::SegmentedWorkspaceUnsupported)
    );
}

fn run_one_case(
    context: &wgpu_fft::device::GpuContext,
    input: Vec<f32>,
    config: FftConfig,
    expected_route: C2cRoute,
) {
    trace_gpu_step(&format!(
        "start c2c config={config:?} route={expected_route:?}"
    ));
    let expected = reference_c2c_nd(&from_interleaved_f32(&input), &config).unwrap();
    let expected = to_interleaved_f32(&expected);
    let byte_len = (input.len() * std::mem::size_of::<f32>()) as u64;

    let input_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.input"),
        size: byte_len,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    context
        .queue
        .write_buffer(&input_buffer, 0, bytemuck::cast_slice(&input));

    let output_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.output"),
        size: byte_len,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.readback"),
        size: byte_len,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let plan = FftPlan::c2c(&context.device, &context.queue, config).unwrap();
    assert_eq!(plan.route(), expected_route);
    assert_eq!(
        plan.large_routing_policy().route_mode(),
        LargeRouteMode::Normal
    );
    assert_plan_diagnostics_graph(&context.device, &plan);
    let twiddle_lut_bytes = plan
        .diagnostics()
        .buffer_requirements()
        .iter()
        .find(|requirement| requirement.role == "helper:twiddle-luts-total")
        .unwrap()
        .required_bytes;
    if expected_route == C2cRoute::DirectDft {
        assert_eq!(twiddle_lut_bytes, 8);
    }
    if expected_route == C2cRoute::Rader && plan.config().shape()[plan.config().axes()[0]] == 17 {
        // A direct DFT of N=17 reads one table of W_17^i.
        assert_eq!(twiddle_lut_bytes, 17 * 8);
    }
    assert_eq!(plan.axis_factors().len(), plan.config().axes().len());
    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("wgpu_fft.test.encoder"),
        });
    plan.execute_checked(&context.device, &mut encoder, &input_buffer, &output_buffer)
        .unwrap();
    encoder.copy_buffer_to_buffer(&output_buffer, 0, &readback_buffer, 0, byte_len);
    context.queue.submit([encoder.finish()]);

    let label = format!("{:?}", plan.config());
    trace_gpu_step(&format!("submitted {label}"));
    let actual = read_interleaved_f32(context, &readback_buffer, &label);
    assert_close(&actual, &expected, &label);
    trace_gpu_step(&format!("finish {label}"));
}

fn assert_view_validation_behavior(context: &wgpu_fft::device::GpuContext) {
    let creation_error = FftPlan::c2c_with_diagnostics(
        &context.device,
        &context.queue,
        FftConfig::new(8).with_axes([1]),
    )
    .err()
    .expect("invalid axis should fail plan construction");
    assert_eq!(
        creation_error.error(),
        &FftError::InvalidAxis { axis: 1, rank: 1 }
    );
    assert_eq!(creation_error.diagnostics().route().transform, "c2c");
    assert!(creation_error
        .diagnostics()
        .blockers()
        .iter()
        .any(|blocker| blocker.stage.as_deref() == Some("axis-policy")));

    let plan = FftPlan::c2c(&context.device, &context.queue, FftConfig::new(8)).unwrap();
    let required = plan.required_buffer_size_bytes();

    let too_small_input = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.too_small_input_view"),
        size: required - 8,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let valid_output = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.valid_output_view"),
        size: required,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("wgpu_fft.test.view_validation_encoder"),
        });
    assert_eq!(
        plan.execute_views(
            &context.device,
            &mut encoder,
            BufferView::whole(&too_small_input),
            BufferView::whole(&valid_output),
        ),
        Err(FftError::BufferViewTooSmall {
            required,
            actual: required - 8,
        })
    );
    let too_small_whole_buffer_error = plan
        .execute_checked(
            &context.device,
            &mut encoder,
            &too_small_input,
            &valid_output,
        )
        .unwrap_err();
    assert_eq!(
        too_small_whole_buffer_error.error(),
        &FftError::BufferViewTooSmall {
            required,
            actual: required - 8,
        }
    );
    assert!(too_small_whole_buffer_error
        .diagnostics()
        .blockers()
        .iter()
        .any(|blocker| {
            blocker.kind == FftBlockerKind::Validation
                && blocker.route.as_deref() == Some("mixed-radix")
                && blocker.required_bytes == Some(required)
                && blocker.actual_bytes == Some(required - 8)
        }));
    assert!(too_small_whole_buffer_error
        .diagnostics()
        .blockers()
        .iter()
        .any(|blocker| {
            blocker.kind == FftBlockerKind::Validation
                && blocker.route.as_deref() == Some("mixed-radix")
                && blocker.stage.as_deref() == Some("input-logical-view")
                && blocker.layout.as_deref() == Some("contiguous")
                && blocker.required_bytes == Some(required)
                && blocker.actual_bytes == Some(required - 8)
        }));
    let too_small_diagnostic_error = plan
        .execute_views_with_diagnostics(
            &context.device,
            &mut encoder,
            BufferView::whole(&too_small_input),
            BufferView::whole(&valid_output),
        )
        .unwrap_err();
    assert_eq!(
        too_small_diagnostic_error.error(),
        &FftError::BufferViewTooSmall {
            required,
            actual: required - 8,
        }
    );
    assert!(too_small_diagnostic_error
        .diagnostics()
        .blockers()
        .iter()
        .any(|blocker| {
            blocker.kind == FftBlockerKind::Validation
                && blocker.route.as_deref() == Some("mixed-radix")
                && blocker.required_bytes == Some(required)
                && blocker.actual_bytes == Some(required - 8)
        }));
    let input_size_blockers = too_small_diagnostic_error
        .diagnostics()
        .blockers()
        .iter()
        .filter(|blocker| {
            blocker.kind == FftBlockerKind::Validation
                && blocker.route.as_deref() == Some("mixed-radix")
                && blocker.stage.as_deref() == Some("input-logical-view")
                && blocker.layout.as_deref() == Some("contiguous")
                && blocker.required_bytes == Some(required)
                && blocker.actual_bytes == Some(required - 8)
        })
        .count();
    assert_eq!(input_size_blockers, 1);

    let valid_input = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.valid_input_view"),
        size: required,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let too_small_output = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.too_small_output_view"),
        size: required - 8,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    assert_eq!(
        plan.execute_views(
            &context.device,
            &mut encoder,
            BufferView::whole(&valid_input),
            BufferView::whole(&too_small_output),
        ),
        Err(FftError::BufferViewTooSmall {
            required,
            actual: required - 8,
        })
    );
    let too_small_output_error = plan
        .execute_views_with_diagnostics(
            &context.device,
            &mut encoder,
            BufferView::whole(&valid_input),
            BufferView::whole(&too_small_output),
        )
        .unwrap_err();
    assert_eq!(
        too_small_output_error.error(),
        &FftError::BufferViewTooSmall {
            required,
            actual: required - 8,
        }
    );
    assert!(too_small_output_error
        .diagnostics()
        .blockers()
        .iter()
        .any(|blocker| {
            blocker.kind == FftBlockerKind::Validation
                && blocker.route.as_deref() == Some("mixed-radix")
                && blocker.stage.as_deref() == Some("output-logical-view")
                && blocker.layout.as_deref() == Some("contiguous")
                && blocker.required_bytes == Some(required)
                && blocker.actual_bytes == Some(required - 8)
        }));
    let strided_layout = FftLogicalLayout::new(1, 2).unwrap();
    let logical_elements = plan.config().logical_complex_len().unwrap() as u64;
    let strided_required_bytes = (strided_layout.element_offset
        + strided_layout.element_stride * (logical_elements - 1)
        + 1)
        * 2
        * std::mem::size_of::<f32>() as u64;
    let too_small_strided_input = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.too_small_strided_input_view"),
        size: strided_required_bytes - 8,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let too_small_strided_output = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.too_small_strided_output_view"),
        size: strided_required_bytes - 8,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let strided_diagnostics = plan.diagnostics_for_logical_views(
        &context.device,
        &FftLogicalView::new(BufferView::whole(&too_small_strided_input), strided_layout).unwrap(),
        &FftLogicalView::new(BufferView::whole(&too_small_strided_output), strided_layout).unwrap(),
    );
    for stage in ["input-logical-view", "output-logical-view"] {
        assert!(
            strided_diagnostics.blockers().iter().any(|blocker| {
                blocker.kind == FftBlockerKind::Layout
                    && blocker.route.as_deref() == Some("mixed-radix")
                    && blocker.stage.as_deref() == Some(stage)
                    && blocker.layout.as_deref() == Some("strided")
                    && blocker.required_bytes == Some(strided_required_bytes)
                    && blocker.actual_bytes == Some(strided_required_bytes - 8)
            }),
            "expected {stage} strided layout-size blocker"
        );
    }
    let no_storage_output_for_mixed_diagnostics =
        context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("wgpu_fft.test.no_storage_output_with_bad_input_view"),
            size: required,
            usage: wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
    let mixed_endpoint_diagnostics = plan.diagnostics_for_logical_views(
        &context.device,
        &FftLogicalView::new(BufferView::whole(&too_small_strided_input), strided_layout).unwrap(),
        &FftLogicalView::contiguous(BufferView::whole(&no_storage_output_for_mixed_diagnostics)),
    );
    assert!(mixed_endpoint_diagnostics.blockers().iter().any(|blocker| {
        blocker.kind == FftBlockerKind::Layout
            && blocker.route.as_deref() == Some("mixed-radix")
            && blocker.stage.as_deref() == Some("input-logical-view")
            && blocker.layout.as_deref() == Some("strided")
    }));
    assert!(mixed_endpoint_diagnostics.blockers().iter().any(|blocker| {
        blocker.kind == FftBlockerKind::BufferUsage
            && blocker.route.as_deref() == Some("mixed-radix")
            && blocker.stage.as_deref() == Some("output-storage-window")
            && blocker.layout.as_deref() == Some("contiguous")
    }));

    assert_eq!(
        BufferView::new(&valid_input, required, 8).unwrap_err(),
        FftError::BufferViewOutOfBounds {
            offset: required,
            size: 8,
            buffer_size: required,
        }
    );

    let storage_alignment =
        u64::from(context.device.limits().min_storage_buffer_offset_alignment).max(1);
    if storage_alignment > 1 {
        let misaligned_offset = storage_alignment / 2;
        let unaligned_input = context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("wgpu_fft.test.unaligned_logical_input"),
            size: required + misaligned_offset,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let unaligned_output = context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("wgpu_fft.test.unaligned_logical_output"),
            size: required + misaligned_offset,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let diagnostics = plan.diagnostics_for_logical_views(
            &context.device,
            &FftLogicalView::contiguous(
                BufferView::new(&unaligned_input, misaligned_offset, required).unwrap(),
            ),
            &FftLogicalView::contiguous(
                BufferView::new(&unaligned_output, misaligned_offset, required).unwrap(),
            ),
        );
        for stage in ["input-storage-window", "output-storage-window"] {
            assert!(
                diagnostics.blockers().iter().any(|blocker| {
                    blocker.kind == FftBlockerKind::Alignment
                        && blocker.route.as_deref() == Some("mixed-radix")
                        && blocker.stage.as_deref() == Some(stage)
                        && blocker.layout.as_deref() == Some("offset")
                        && blocker.required_bytes == Some(storage_alignment)
                        && blocker.actual_bytes == Some(misaligned_offset)
                }),
                "expected {stage} alignment blocker"
            );
        }
    }

    let no_storage_input = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.no_storage_input_view"),
        size: required,
        usage: wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let usage_diagnostics = plan.diagnostics_for_logical_views(
        &context.device,
        &FftLogicalView::contiguous(BufferView::whole(&no_storage_input)),
        &FftLogicalView::contiguous(BufferView::whole(&valid_output)),
    );
    assert!(usage_diagnostics.blockers().iter().any(|blocker| {
        blocker.kind == FftBlockerKind::BufferUsage
            && blocker.stage.as_deref() == Some("input-storage-window")
    }));
    let view_usage_diagnostics = plan.diagnostics_for_views(
        &context.device,
        BufferView::whole(&no_storage_input),
        BufferView::whole(&valid_output),
    );
    assert!(view_usage_diagnostics.blockers().iter().any(|blocker| {
        blocker.kind == FftBlockerKind::BufferUsage
            && blocker.route.as_deref() == Some("mixed-radix")
            && blocker.stage.as_deref() == Some("input-storage-window")
            && blocker.layout.as_deref() == Some("contiguous")
    }));
    let forced_view_limits = FftDeviceLimits {
        max_storage_buffer_binding_size: required - 8,
        max_buffer_size: required,
        min_storage_buffer_offset_alignment: 1,
    };
    let forced_view_diagnostics = plan.diagnostics_for_views_with_limits(
        forced_view_limits,
        BufferView::whole(&valid_input),
        BufferView::whole(&valid_output),
    );
    assert_eq!(
        forced_view_diagnostics.device_limits(),
        Some(forced_view_limits)
    );
    assert!(forced_view_diagnostics.blockers().iter().any(|blocker| {
        blocker.kind == FftBlockerKind::DeviceLimit
            && blocker.route.as_deref() == Some("mixed-radix")
            && blocker.stage.as_deref() == Some("input-storage-window")
            && blocker.layout.as_deref() == Some("contiguous")
            && blocker.required_bytes == Some(required)
            && blocker.limit_bytes == Some(required - 8)
    }));
    assert_eq!(
        plan.execute_views(
            &context.device,
            &mut encoder,
            BufferView::whole(&no_storage_input),
            BufferView::whole(&valid_output),
        ),
        Err(FftError::BufferViewMissingUsage { usage: "STORAGE" })
    );
    let usage_diagnostic_error = plan
        .execute_logical_views_with_diagnostics(
            &context.device,
            &mut encoder,
            FftLogicalView::contiguous(BufferView::whole(&no_storage_input)),
            FftLogicalView::contiguous(BufferView::whole(&valid_output)),
        )
        .unwrap_err();
    assert_eq!(
        usage_diagnostic_error.error(),
        &FftError::BufferViewMissingUsage { usage: "STORAGE" }
    );
    assert!(usage_diagnostic_error
        .diagnostics()
        .blockers()
        .iter()
        .any(|blocker| {
            blocker.kind == FftBlockerKind::BufferUsage
                && blocker.route.as_deref() == Some("mixed-radix")
                && blocker.stage.as_deref() == Some("input-storage-window")
        }));

    let empty_segments: [BufferSegment<'_>; 0] = [];
    assert_eq!(
        BufferView::from_segments(&empty_segments, 0, 0).unwrap_err(),
        FftError::BufferViewEmptySegments
    );

    let zero_segments = [BufferSegment::new(&valid_input, 0, 0)];
    assert_eq!(
        BufferView::from_segments(&zero_segments, 0, 0).unwrap_err(),
        FftError::BufferSegmentZeroSize { index: 0 }
    );

    let out_of_bounds_segments = [BufferSegment::new(&valid_input, required - 4, 8)];
    assert_eq!(
        BufferView::from_segments(&out_of_bounds_segments, 0, 8).unwrap_err(),
        FftError::BufferSegmentOutOfBounds {
            index: 0,
            offset: required - 4,
            size: 8,
            buffer_size: required,
        }
    );

    let split_segments = split_buffer_segments(&valid_input, required, 2);
    assert_eq!(
        BufferView::from_segments(&split_segments, required - 4, 8).unwrap_err(),
        FftError::BufferViewWindowOutOfRange {
            offset: required - 4,
            size: 8,
            length: required,
        }
    );
    let split_view = BufferView::from_segments(&split_segments, 0, required).unwrap();
    let segmented_usage_diagnostics = plan.diagnostics_for_logical_views(
        &context.device,
        &FftLogicalView::contiguous(split_view.clone()),
        &FftLogicalView::contiguous(BufferView::whole(&valid_output)),
    );
    assert!(segmented_usage_diagnostics
        .blockers()
        .iter()
        .any(|blocker| {
            blocker.kind == FftBlockerKind::BufferUsage
                && blocker.stage.as_deref() == Some("input-segmented-copy-window")
                && blocker.layout.as_deref() == Some("segmented")
        }));
    assert_eq!(split_view.logical_byte_offset(), 0);
    assert_eq!(split_view.segments().len(), 2);
    assert!(!split_view.is_single_segment());
    let ranges = split_view.ranges(16, 32).unwrap();
    assert_eq!(ranges.len(), 2);
    assert_eq!(ranges[0].offset_bytes, 16);
    assert_eq!(ranges[0].size_bytes, 16);
    assert_eq!(ranges[1].offset_bytes, 32);
    assert_eq!(ranges[1].size_bytes, 16);

    let copy_input = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.unaligned_copy_input_view"),
        size: required,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let unaligned_segments = [
        BufferSegment::new(&copy_input, 0, 6),
        BufferSegment::new(&copy_input, 6, required - 6),
    ];
    let copy_output = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.unaligned_copy_output_view"),
        size: required,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let unaligned_output_segments = [
        BufferSegment::new(&copy_output, 0, 6),
        BufferSegment::new(&copy_output, 6, required - 6),
    ];
    let copy_alignment_diagnostics = plan.diagnostics_for_logical_views(
        &context.device,
        &FftLogicalView::contiguous(
            BufferView::from_segments(&unaligned_segments, 0, required).unwrap(),
        ),
        &FftLogicalView::contiguous(
            BufferView::from_segments(&unaligned_output_segments, 0, required).unwrap(),
        ),
    );
    for stage in [
        "input-segmented-copy-window",
        "output-segmented-copy-window",
    ] {
        assert!(
            copy_alignment_diagnostics.blockers().iter().any(|blocker| {
                blocker.kind == FftBlockerKind::Alignment
                    && blocker.route.as_deref() == Some("mixed-radix")
                    && blocker.stage.as_deref() == Some(stage)
                    && blocker.layout.as_deref() == Some("segmented")
                    && blocker.required_bytes == Some(4)
                    && blocker.actual_bytes == Some(6)
                    && blocker.reason.contains("offset 0")
                    && blocker.reason.contains("size 6")
            }),
            "expected {stage} copy-alignment blocker"
        );
    }
    assert_eq!(
        plan.execute_views(
            &context.device,
            &mut encoder,
            BufferView::from_segments(&unaligned_segments, 0, required).unwrap(),
            BufferView::whole(&valid_output),
        ),
        Err(FftError::BufferViewCopyUnaligned {
            offset: 0,
            size: 6,
            alignment: 4,
        })
    );
    assert_eq!(
        plan.execute_views(
            &context.device,
            &mut encoder,
            split_view,
            BufferView::whole(&valid_output),
        ),
        Err(FftError::BufferViewMissingUsage { usage: "COPY_SRC" })
    );

    let zero_stride_layout = BufferLayout {
        element_offset: 0,
        element_stride: 0,
        batch_stride: None,
    };
    assert_eq!(
        FftIoView::new(BufferView::whole(&valid_input), zero_stride_layout).unwrap_err(),
        FftError::BufferLayoutZeroStride
    );
    let out_of_bounds_layout = BufferLayout::new(8, 2).unwrap();
    assert_eq!(
        plan.execute_io_views(
            &context.device,
            &mut encoder,
            FftIoView::new(BufferView::whole(&valid_input), out_of_bounds_layout).unwrap(),
            FftIoView::contiguous(BufferView::whole(&valid_output)),
        ),
        Err(FftError::BufferLayoutOutOfBounds {
            required_bytes: 184,
            actual_bytes: required,
        })
    );
    let segmented_strided_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.segmented_strided_input"),
        size: 120,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let segmented_strided_view = BufferView::from_segments(
        &split_buffer_segments(&segmented_strided_buffer, 120, 2),
        0,
        120,
    )
    .unwrap();
    assert_eq!(
        plan.execute_io_views(
            &context.device,
            &mut encoder,
            FftIoView::new(segmented_strided_view, BufferLayout::new(0, 2).unwrap()).unwrap(),
            FftIoView::contiguous(BufferView::whole(&valid_output)),
        ),
        Err(FftError::BufferViewMissingUsage { usage: "COPY_SRC" })
    );

    let batch_plan = FftPlan::c2c(
        &context.device,
        &context.queue,
        FftConfig::new(8)
            .with_batch(2)
            .with_normalization(Normalization::None),
    )
    .unwrap();
    let batch_input = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.strided_batch_input"),
        size: 512,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let batch_output = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.strided_batch_output"),
        size: batch_plan.required_buffer_size_bytes(),
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    assert_eq!(
        batch_plan.execute_io_views(
            &context.device,
            &mut encoder,
            FftIoView::new(
                BufferView::whole(&batch_input),
                BufferLayout::new(0, 3).unwrap().with_batch_stride(4),
            )
            .unwrap(),
            FftIoView::contiguous(BufferView::whole(&batch_output)),
        ),
        Err(FftError::BufferLayoutBatchStrideTooSmall {
            required: 22,
            actual: 4,
        })
    );

    let r2c_plan = FftPlan::r2c(&context.device, &context.queue, FftConfig::new(8)).unwrap();
    r2c_plan
        .execute_io_views(
            &context.device,
            &mut encoder,
            FftIoView::contiguous(BufferView::whole(&valid_input)),
            FftIoView::contiguous(BufferView::whole(&valid_output)),
        )
        .unwrap();
    let c2r_plan = FftPlan::c2r(&context.device, &context.queue, FftConfig::inverse(8)).unwrap();
    c2r_plan
        .execute_io_views(
            &context.device,
            &mut encoder,
            FftIoView::contiguous(BufferView::whole(&valid_input)),
            FftIoView::contiguous(BufferView::whole(&valid_output)),
        )
        .unwrap();

    let workspace_plan = FftPlan::c2c(
        &context.device,
        &context.queue,
        FftConfig::new_nd([4, 3]).with_tuning(per_axis_tuning()),
    )
    .unwrap();
    let workspace_required = workspace_plan.workspace_size_bytes();
    let valid_input = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.workspace_view_valid_input"),
        size: workspace_plan.required_buffer_size_bytes(),
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let too_small_workspace_input = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.workspace_view_too_small_input"),
        size: workspace_plan.required_buffer_size_bytes() - 8,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let valid_output = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.workspace_view_valid_output"),
        size: workspace_plan.required_buffer_size_bytes(),
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let too_small_workspace = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.too_small_workspace_view"),
        size: workspace_required - 8,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    assert_eq!(
        workspace_plan.execute_views_with_workspace(
            &context.device,
            &mut encoder,
            BufferView::whole(&valid_input),
            BufferView::whole(&valid_output),
            BufferView::whole(&too_small_workspace),
        ),
        Err(FftError::WorkspaceTooSmall {
            required: workspace_required,
            actual: workspace_required - 8,
        })
    );
    let too_small_workspace_diagnostics = workspace_plan.diagnostics_for_views_with_workspace(
        &context.device,
        BufferView::whole(&valid_input),
        BufferView::whole(&valid_output),
        BufferView::whole(&too_small_workspace),
    );
    assert!(too_small_workspace_diagnostics
        .blockers()
        .iter()
        .any(|blocker| {
            blocker.kind == FftBlockerKind::Workspace
                && blocker.route.as_deref() == Some("mixed-radix")
                && blocker.stage.as_deref() == Some("workspace")
                && blocker.helper_buffer.as_deref() == Some("workspace")
                && blocker.required_bytes == Some(workspace_required)
                && blocker.actual_bytes == Some(workspace_required - 8)
        }));
    let workspace_whole_buffer_error = workspace_plan
        .execute_checked_with_workspace(
            &context.device,
            &mut encoder,
            &valid_input,
            &valid_output,
            &too_small_workspace,
        )
        .unwrap_err();
    assert_eq!(
        workspace_whole_buffer_error.error(),
        &FftError::WorkspaceTooSmall {
            required: workspace_required,
            actual: workspace_required - 8,
        }
    );
    assert!(workspace_whole_buffer_error
        .diagnostics()
        .blockers()
        .iter()
        .any(|blocker| {
            blocker.kind == FftBlockerKind::Workspace
                && blocker.route.as_deref() == Some("mixed-radix")
                && blocker.required_bytes == Some(workspace_required)
                && blocker.actual_bytes == Some(workspace_required - 8)
        }));
    let no_storage_workspace = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.no_storage_workspace_view"),
        size: workspace_required,
        usage: wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let workspace_usage_error = workspace_plan
        .execute_with_workspace_diagnostics(
            &context.device,
            &mut encoder,
            &valid_input,
            &valid_output,
            &no_storage_workspace,
        )
        .unwrap_err();
    assert_eq!(
        workspace_usage_error.error(),
        &FftError::BufferViewMissingUsage { usage: "STORAGE" }
    );
    assert!(workspace_usage_error
        .diagnostics()
        .blockers()
        .iter()
        .any(|blocker| {
            blocker.kind == FftBlockerKind::BufferUsage
                && blocker.route.as_deref() == Some("mixed-radix")
                && blocker.stage.as_deref() == Some("workspace")
                && blocker.helper_buffer.as_deref() == Some("workspace")
                && blocker.required_bytes == Some(workspace_required)
                && blocker.actual_bytes == Some(workspace_required)
        }));
    let workspace_usage_diagnostics = workspace_plan.diagnostics_for_workspace(
        &context.device,
        &valid_input,
        &valid_output,
        &no_storage_workspace,
    );
    assert!(workspace_usage_diagnostics
        .blockers()
        .iter()
        .any(|blocker| {
            blocker.kind == FftBlockerKind::BufferUsage
                && blocker.route.as_deref() == Some("mixed-radix")
                && blocker.stage.as_deref() == Some("workspace")
                && blocker.helper_buffer.as_deref() == Some("workspace")
                && blocker.required_bytes == Some(workspace_required)
                && blocker.actual_bytes == Some(workspace_required)
        }));
    let valid_workspace = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.valid_workspace_view"),
        size: workspace_required,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let forced_workspace_limits = FftDeviceLimits {
        max_storage_buffer_binding_size: workspace_required - 8,
        max_buffer_size: workspace_required,
        min_storage_buffer_offset_alignment: 1,
    };
    let forced_workspace_diagnostics = workspace_plan.diagnostics_for_workspace_with_limits(
        forced_workspace_limits,
        &valid_input,
        &valid_output,
        &valid_workspace,
    );
    assert_eq!(
        forced_workspace_diagnostics.device_limits(),
        Some(forced_workspace_limits)
    );
    assert!(forced_workspace_diagnostics
        .blockers()
        .iter()
        .any(|blocker| {
            blocker.kind == FftBlockerKind::DeviceLimit
                && blocker.route.as_deref() == Some("mixed-radix")
                && blocker.stage.as_deref() == Some("workspace")
                && blocker.helper_buffer.as_deref() == Some("workspace")
                && blocker.required_bytes == Some(workspace_required)
                && blocker.limit_bytes == Some(workspace_required - 8)
        }));
    let forced_workspace_buffer_limits = FftDeviceLimits {
        max_storage_buffer_binding_size: workspace_required,
        max_buffer_size: workspace_required - 8,
        min_storage_buffer_offset_alignment: 1,
    };
    let forced_workspace_buffer_diagnostics = workspace_plan.diagnostics_for_workspace_with_limits(
        forced_workspace_buffer_limits,
        &valid_input,
        &valid_output,
        &valid_workspace,
    );
    assert!(forced_workspace_buffer_diagnostics
        .blockers()
        .iter()
        .any(|blocker| {
            blocker.kind == FftBlockerKind::HelperBuffer
                && blocker.route.as_deref() == Some("mixed-radix")
                && blocker.stage.as_deref() == Some("workspace")
                && blocker.helper_buffer.as_deref() == Some("workspace")
                && blocker.required_bytes == Some(workspace_required)
                && blocker.limit_bytes == Some(workspace_required - 8)
        }));
    let workspace_diagnostic_error = workspace_plan
        .execute_views_with_workspace_diagnostics(
            &context.device,
            &mut encoder,
            BufferView::whole(&valid_input),
            BufferView::whole(&valid_output),
            BufferView::whole(&too_small_workspace),
        )
        .unwrap_err();
    assert_eq!(
        workspace_diagnostic_error.error(),
        &FftError::WorkspaceTooSmall {
            required: workspace_required,
            actual: workspace_required - 8,
        }
    );
    assert!(workspace_diagnostic_error
        .diagnostics()
        .buffer_requirements()
        .iter()
        .any(|requirement| requirement.role == "workspace"
            && requirement.required_bytes == workspace_required));
    assert!(workspace_diagnostic_error
        .diagnostics()
        .blockers()
        .iter()
        .any(|blocker| {
            blocker.kind == FftBlockerKind::Workspace
                && blocker.route.as_deref() == Some("mixed-radix")
                && blocker.required_bytes == Some(workspace_required)
                && blocker.actual_bytes == Some(workspace_required - 8)
        }));
    let workspace_size_blockers = workspace_diagnostic_error
        .diagnostics()
        .blockers()
        .iter()
        .filter(|blocker| {
            blocker.kind == FftBlockerKind::Workspace
                && blocker.route.as_deref() == Some("mixed-radix")
                && blocker.stage.as_deref() == Some("workspace")
                && blocker.helper_buffer.as_deref() == Some("workspace")
                && blocker.required_bytes == Some(workspace_required)
                && blocker.actual_bytes == Some(workspace_required - 8)
        })
        .count();
    assert_eq!(workspace_size_blockers, 1);
    let input_and_workspace_diagnostic_error = workspace_plan
        .execute_views_with_workspace_diagnostics(
            &context.device,
            &mut encoder,
            BufferView::whole(&too_small_workspace_input),
            BufferView::whole(&valid_output),
            BufferView::whole(&too_small_workspace),
        )
        .unwrap_err();
    assert_eq!(
        input_and_workspace_diagnostic_error.error(),
        &FftError::BufferViewTooSmall {
            required: workspace_plan.required_input_buffer_size_bytes(),
            actual: workspace_plan.required_input_buffer_size_bytes() - 8,
        }
    );
    assert!(input_and_workspace_diagnostic_error
        .diagnostics()
        .blockers()
        .iter()
        .any(|blocker| {
            blocker.kind == FftBlockerKind::Layout
                && blocker.route.as_deref() == Some("mixed-radix")
                && blocker.stage.as_deref() == Some("input-logical-view")
                && blocker.required_bytes == Some(workspace_plan.required_input_buffer_size_bytes())
                && blocker.actual_bytes
                    == Some(workspace_plan.required_input_buffer_size_bytes() - 8)
        }));
    assert!(input_and_workspace_diagnostic_error
        .diagnostics()
        .blockers()
        .iter()
        .any(|blocker| {
            blocker.kind == FftBlockerKind::Workspace
                && blocker.route.as_deref() == Some("mixed-radix")
                && blocker.stage.as_deref() == Some("workspace")
                && blocker.helper_buffer.as_deref() == Some("workspace")
                && blocker.required_bytes == Some(workspace_required)
                && blocker.actual_bytes == Some(workspace_required - 8)
        }));

    let alignment = u64::from(context.device.limits().min_storage_buffer_offset_alignment);
    if alignment > 1 {
        let misaligned_offset = alignment / 2;
        let misaligned_input = context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("wgpu_fft.test.misaligned_input_view"),
            size: misaligned_offset + required,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let valid_output = context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("wgpu_fft.test.misaligned_valid_output"),
            size: required,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        assert_eq!(
            plan.execute_views(
                &context.device,
                &mut encoder,
                BufferView::new(&misaligned_input, misaligned_offset, required).unwrap(),
                BufferView::whole(&valid_output),
            ),
            Err(FftError::BufferViewOffsetUnaligned {
                offset: misaligned_offset,
                alignment,
            })
        );
    }
}

fn aligned_test_offset(context: &wgpu_fft::device::GpuContext) -> u64 {
    u64::from(context.device.limits().min_storage_buffer_offset_alignment).max(256)
}

fn split_buffer_segments(
    buffer: &wgpu::Buffer,
    total_bytes: u64,
    segment_count: usize,
) -> Vec<BufferSegment<'_>> {
    assert_eq!(total_bytes % 4, 0);
    let total_words = total_bytes / 4;
    let segment_count = segment_count.min(total_words as usize).max(1);
    let mut segments = Vec::with_capacity(segment_count);
    let mut offset_words = 0u64;

    for index in 0..segment_count {
        let remaining_words = total_words - offset_words;
        let remaining_segments = (segment_count - index) as u64;
        let size_words = remaining_words / remaining_segments;
        let offset_bytes = offset_words * 4;
        let size_bytes = if index + 1 == segment_count {
            total_bytes - offset_bytes
        } else {
            size_words * 4
        };
        segments.push(BufferSegment::new(buffer, offset_bytes, size_bytes));
        offset_words += size_bytes / 4;
    }

    segments
}

fn strided_test_layout(logical_per_batch: u64, element_offset: u64) -> BufferLayout {
    let per_batch_span = if logical_per_batch == 0 {
        0
    } else {
        2 * (logical_per_batch - 1) + 1
    };
    BufferLayout::new(element_offset, 2)
        .unwrap()
        .with_batch_stride(per_batch_span + 7)
}

fn logical_layout_from_c2c(layout: BufferLayout) -> FftLogicalLayout {
    let mut logical = FftLogicalLayout::new(layout.element_offset, layout.element_stride).unwrap();
    if let Some(batch_stride) = layout.batch_stride {
        logical = logical.with_batch_stride(batch_stride);
    }
    logical
}

fn assert_plan_diagnostics_graph(device: &wgpu::Device, plan: &FftPlan) {
    let diagnostics = plan.diagnostics();
    assert!(!diagnostics.stages().is_empty());
    assert!(diagnostics
        .stages()
        .iter()
        .any(|stage| matches!(stage.kind.as_str(), "kernel" | "windowed-kernel" | "copy")));
    assert!(diagnostics.buffer_requirements().len() >= 2);
    assert!(diagnostics.buffer_requirements().iter().any(|requirement| {
        requirement.role == "helper:twiddle-luts-total"
            && requirement.required_bytes > 0
            && requirement.format == "complex-f32"
    }));
    for stage in diagnostics
        .stages()
        .iter()
        .filter(|stage| stage.kind == "helper-buffer-window")
    {
        assert!(
            diagnostics.buffer_requirements().iter().any(|requirement| {
                requirement.role == format!("helper:{}", stage.label)
                    && stage
                        .required_bytes
                        .is_some_and(|bytes| requirement.required_bytes >= bytes)
            }),
            "expected helper buffer requirement for {}",
            stage.label
        );
    }

    let device_diagnostics = plan.diagnostics_for_device(device);
    assert!(!device_diagnostics.stages().is_empty());
    assert!(device_diagnostics.blockers().is_empty());
    assert!(device_diagnostics.device_limits().is_some());

    let forced_limits = FftDeviceLimits {
        max_storage_buffer_binding_size: 8,
        max_buffer_size: 8,
        min_storage_buffer_offset_alignment: 1,
    };
    let forced_diagnostics = plan.diagnostics_for_limits(forced_limits);
    assert_eq!(forced_diagnostics.device_limits(), Some(forced_limits));
    assert_eq!(
        forced_diagnostics.stages().len(),
        diagnostics.stages().len()
    );
    let exceeds_forced_limits = diagnostics
        .buffer_requirements()
        .iter()
        .any(|requirement| requirement.required_bytes > 8);
    let has_forced_limit_blocker = forced_diagnostics.blockers().iter().any(|blocker| {
        blocker.route.is_some()
            && blocker.stage.is_some()
            && blocker.required_bytes.is_some()
            && blocker.limit_bytes.is_some()
            && matches!(
                blocker.kind,
                FftBlockerKind::DeviceLimit | FftBlockerKind::HelperBuffer
            )
    });
    assert_eq!(
        has_forced_limit_blocker,
        exceeds_forced_limits,
        "unexpected forced-limit blocker state for {:?}",
        plan.config()
    );
}

fn layout_span_complex(layout: BufferLayout, logical_per_batch: u64, batch: u64) -> u64 {
    if logical_per_batch == 0 || batch == 0 {
        return 0;
    }
    let per_batch_span = layout.element_stride * (logical_per_batch - 1) + 1;
    let batch_stride = layout.batch_stride.unwrap_or(per_batch_span);
    layout.element_offset + (batch - 1) * batch_stride + per_batch_span
}

fn physical_complex_index(
    layout: BufferLayout,
    logical_per_batch: u64,
    logical_index: u64,
) -> usize {
    let per_batch_span = layout.element_stride * (logical_per_batch - 1) + 1;
    let batch_stride = layout.batch_stride.unwrap_or(per_batch_span);
    let batch = logical_index / logical_per_batch;
    let element = logical_index - batch * logical_per_batch;
    (layout.element_offset + batch * batch_stride + element * layout.element_stride) as usize
}

fn scatter_strided_interleaved(
    logical: &[f32],
    layout: BufferLayout,
    logical_per_batch: u64,
    batch: u64,
) -> Vec<f32> {
    let span = layout_span_complex(layout, logical_per_batch, batch) as usize;
    let mut physical = vec![0.0; span * 2];
    for logical_index in 0..(logical.len() / 2) {
        let physical_index =
            physical_complex_index(layout, logical_per_batch, logical_index as u64);
        physical[physical_index * 2] = logical[logical_index * 2];
        physical[physical_index * 2 + 1] = logical[logical_index * 2 + 1];
    }
    physical
}

fn gather_strided_interleaved(
    physical: &[f32],
    layout: BufferLayout,
    logical_per_batch: u64,
    batch: u64,
) -> Vec<f32> {
    let total = (logical_per_batch * batch) as usize;
    let mut logical = vec![0.0; total * 2];
    for logical_index in 0..total {
        let physical_index =
            physical_complex_index(layout, logical_per_batch, logical_index as u64);
        logical[logical_index * 2] = physical[physical_index * 2];
        logical[logical_index * 2 + 1] = physical[physical_index * 2 + 1];
    }
    logical
}

fn read_interleaved_f32(
    context: &wgpu_fft::device::GpuContext,
    readback_buffer: &wgpu::Buffer,
    label: &str,
) -> Vec<f32> {
    let slice = readback_buffer.slice(..);
    let (sender, receiver) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        sender.send(result).expect("map result receiver is alive");
    });
    trace_gpu_step(&format!("poll readback start {label}"));
    context
        .device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("device polling should succeed");
    trace_gpu_step(&format!("poll readback complete {label}"));
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

fn trace_adapter(context: &wgpu_fft::device::GpuContext) {
    if std::env::var_os("WGPU_FFT_TRACE_GPU_TESTS").is_none() {
        return;
    }
    let info = context.adapter.get_info();
    eprintln!(
        "gpu-test adapter.name={} vendor={:#x} device={:#x} type={:?} backend={:?}",
        info.name, info.vendor, info.device, info.device_type, info.backend
    );
}

fn trace_gpu_step(message: &str) {
    if std::env::var_os("WGPU_FFT_TRACE_GPU_TESTS").is_some() {
        eprintln!("gpu-test: {message}");
    }
}

fn assert_close(actual: &[f32], expected: &[f32], label: &str) {
    assert_close_with_abs_tolerance(actual, expected, label, 1.0e-2);
}

fn assert_close_with_abs_tolerance(
    actual: &[f32],
    expected: &[f32],
    label: &str,
    abs_tolerance: f32,
) {
    const ABS_TOLERANCE: f32 = 1.0e-2;
    const REL_TOLERANCE: f32 = 1.0e-6;
    for (index, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
        let tolerance = abs_tolerance.max(ABS_TOLERANCE) + REL_TOLERANCE * expected.abs();
        assert!(
            (actual - expected).abs() < tolerance,
            "{label}: index {index}: actual={actual}, expected={expected}, tolerance={tolerance}"
        );
    }
}
