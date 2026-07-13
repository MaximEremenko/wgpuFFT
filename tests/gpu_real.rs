use std::sync::mpsc;

use wgpu_fft::math::{reference_c2r_from_packed_interleaved, reference_r2c_packed_interleaved};
use wgpu_fft::{
    export_pipeline_cache_snapshot, BufferLayout, BufferSegment, BufferView, C2cRoute,
    FftBlockerKind, FftConfig, FftDeviceLimits, FftError, FftIoView, FftLogicalLayout,
    FftLogicalView, FftPlan, FftTransformKind, LargePolicyLimits, LargeRouteMode, Normalization,
};

#[test]
fn real_gpu_matches_cpu_reference() {
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
            FftConfig::new_nd([17, 4]).with_normalization(Normalization::None),
            C2cRoute::AxisSequence,
        ),
        (
            FftConfig::new(16)
                .with_batch(2)
                .with_normalization(Normalization::None),
            C2cRoute::MixedRadix,
        ),
    ] {
        run_r2c_case(&context, config.clone(), route, ViewMode::Direct);
        run_c2r_case(
            &context,
            inverse_config_for(&config),
            route,
            ViewMode::Direct,
        );
    }

    run_r2c_case(
        &context,
        FftConfig::new(16).with_normalization(Normalization::None),
        C2cRoute::MixedRadix,
        ViewMode::Offset,
    );
    run_c2r_case(
        &context,
        FftConfig::inverse(16),
        C2cRoute::MixedRadix,
        ViewMode::Offset,
    );
    for mode in [
        ViewMode::SegmentedBoth,
        ViewMode::SegmentedInput,
        ViewMode::SegmentedOutput,
    ] {
        run_r2c_case(
            &context,
            FftConfig::new(16).with_normalization(Normalization::None),
            C2cRoute::MixedRadix,
            mode,
        );
        run_c2r_case(&context, FftConfig::inverse(16), C2cRoute::MixedRadix, mode);
    }
    run_r2c_strided_logical_case(&context);
    run_c2r_strided_logical_case(&context);
    run_r2c_strided_io_view_case(&context);
    run_c2r_strided_io_view_case(&context);
    run_r2c_segmented_strided_logical_case(
        &context,
        FftConfig::new(16).with_normalization(Normalization::None),
        C2cRoute::MixedRadix,
        None,
        LargeRouteMode::Normal,
    );
    run_c2r_segmented_strided_logical_case(
        &context,
        FftConfig::inverse(16),
        C2cRoute::MixedRadix,
        None,
        LargeRouteMode::Normal,
    );
    run_real_four_step_child_case(&context);

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
                .with_batch(5)
                .with_normalization(Normalization::None),
            C2cRoute::AxisSequence,
        ),
    ] {
        run_r2c_large_chunk_case(&context, config.clone(), route, ViewMode::Direct);
        run_c2r_large_chunk_case(
            &context,
            inverse_config_for(&config),
            route,
            ViewMode::Direct,
        );
    }
    for mode in [
        ViewMode::Offset,
        ViewMode::SegmentedBoth,
        ViewMode::SegmentedInput,
        ViewMode::SegmentedOutput,
        ViewMode::Strided,
    ] {
        let config = FftConfig::new(16)
            .with_batch(5)
            .with_normalization(Normalization::None);
        run_r2c_large_chunk_case(&context, config.clone(), C2cRoute::MixedRadix, mode);
        run_c2r_large_chunk_case(
            &context,
            inverse_config_for(&config),
            C2cRoute::MixedRadix,
            mode,
        );
    }

    for mode in [
        ViewMode::Direct,
        ViewMode::Offset,
        ViewMode::SegmentedBoth,
        ViewMode::Strided,
    ] {
        let config = FftConfig::new(64).with_normalization(Normalization::None);
        run_r2c_large_single_decomposition_case(
            &context,
            config.clone(),
            C2cRoute::MixedRadix,
            mode,
        );
        run_c2r_large_single_decomposition_case(
            &context,
            inverse_config_for(&config),
            C2cRoute::MixedRadix,
            mode,
        );
    }
    let odd_config = FftConfig::new(63).with_normalization(Normalization::None);
    run_r2c_large_single_decomposition_case(
        &context,
        odd_config.clone(),
        C2cRoute::MixedRadix,
        ViewMode::Direct,
    );
    run_c2r_large_single_decomposition_case(
        &context,
        inverse_config_for(&odd_config),
        C2cRoute::MixedRadix,
        ViewMode::Direct,
    );
    let batched_large_config = FftConfig::new(64)
        .with_batch(2)
        .with_normalization(Normalization::None);
    run_r2c_large_single_decomposition_case(
        &context,
        batched_large_config.clone(),
        C2cRoute::MixedRadix,
        ViewMode::Direct,
    );
    run_c2r_large_single_decomposition_case(
        &context,
        inverse_config_for(&batched_large_config),
        C2cRoute::MixedRadix,
        ViewMode::Direct,
    );
    for config in [
        FftConfig::new_nd([17, 64])
            .with_batch(2)
            .with_normalization(Normalization::None),
        FftConfig::new_nd([34, 64])
            .with_batch(2)
            .with_normalization(Normalization::None),
    ] {
        run_r2c_large_single_decomposition_case(
            &context,
            config.clone(),
            C2cRoute::AxisSequence,
            ViewMode::Direct,
        );
        run_c2r_large_single_decomposition_case(
            &context,
            inverse_config_for(&config),
            C2cRoute::AxisSequence,
            ViewMode::Direct,
        );
    }
    let segmented_strided_large_config = FftConfig::new_nd([17, 64])
        .with_batch(2)
        .with_normalization(Normalization::None);
    run_r2c_segmented_strided_logical_case(
        &context,
        segmented_strided_large_config.clone(),
        C2cRoute::AxisSequence,
        Some(real_single_decomposition_test_limits(
            &segmented_strided_large_config,
        )),
        LargeRouteMode::LargeChunk,
    );
    run_c2r_segmented_strided_logical_case(
        &context,
        inverse_config_for(&segmented_strided_large_config),
        C2cRoute::AxisSequence,
        Some(real_single_decomposition_test_limits(
            &segmented_strided_large_config,
        )),
        LargeRouteMode::LargeChunk,
    );

    trace_gpu_step("start real logical execution diagnostics");
    assert_real_logical_execution_diagnostics(&context);
    trace_gpu_step("finish real logical execution diagnostics");
    trace_gpu_step("start real validation behavior");
    assert_real_validation_behavior(&context);
    trace_gpu_step("finish real validation behavior");
    trace_gpu_step("start real pipeline cache snapshot behavior");
    assert_real_pipeline_cache_snapshot_behavior(&context);
    trace_gpu_step("finish real pipeline cache snapshot behavior");
    trace_gpu_step("finish real gpu test");

    #[cfg(windows)]
    {
        trace_gpu_step("forget gpu context to avoid native backend teardown hang");
        std::mem::forget(context);
    }
}

fn run_real_four_step_child_case(context: &wgpu_fft::device::GpuContext) {
    let forward = FftConfig::new_nd([15, 14]).with_normalization(Normalization::None);
    let inverse = FftConfig::inverse_nd([15, 14]);
    let limits = LargePolicyLimits {
        max_storage_buffer_binding_size: 256,
        max_buffer_size: forward.required_buffer_size_bytes().unwrap(),
    };
    let real = real_input_for_config(&forward);
    let packed = reference_r2c_packed_interleaved(&real, &forward).unwrap();

    let r2c = FftPlan::r2c_with_large_policy_limits_for_testing(
        &context.device,
        &context.queue,
        forward,
        limits,
    )
    .unwrap();
    assert_eq!(
        r2c.large_routing_policy().execution_kind(),
        wgpu_fft::LargeExecutionKind::OutOfCoreFourStep
    );
    let real_input = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.real.four_step.r2c_input"),
        size: r2c.required_input_buffer_size_bytes(),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let packed_output = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.real.four_step.r2c_output"),
        size: r2c.required_output_buffer_size_bytes(),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    context
        .queue
        .write_buffer(&real_input, 0, bytemuck::cast_slice(&real));
    assert!(r2c
        .diagnostics_for_views(
            &context.device,
            BufferView::whole(&real_input),
            BufferView::whole(&packed_output),
        )
        .blockers()
        .is_empty());
    let r2c_readback = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.real.four_step.r2c_readback"),
        size: r2c.required_output_buffer_size_bytes(),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("wgpu_fft.test.real.four_step.r2c_encoder"),
        });
    r2c.execute_checked(&context.device, &mut encoder, &real_input, &packed_output)
        .unwrap();
    encoder.copy_buffer_to_buffer(
        &packed_output,
        0,
        &r2c_readback,
        0,
        r2c.required_output_buffer_size_bytes(),
    );
    context.queue.submit([encoder.finish()]);
    let actual_packed = read_f32(context, &r2c_readback, "r2c four-step child");
    assert_close(&actual_packed, &packed, "r2c four-step child");

    let c2r = FftPlan::c2r_with_large_policy_limits_for_testing(
        &context.device,
        &context.queue,
        inverse.clone(),
        limits,
    )
    .unwrap();
    assert_eq!(
        c2r.large_routing_policy().execution_kind(),
        wgpu_fft::LargeExecutionKind::OutOfCoreFourStep
    );
    let packed_input = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.real.four_step.c2r_input"),
        size: c2r.required_input_buffer_size_bytes(),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let real_output = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.real.four_step.c2r_output"),
        size: c2r.required_output_buffer_size_bytes(),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    context
        .queue
        .write_buffer(&packed_input, 0, bytemuck::cast_slice(&packed));
    assert!(c2r
        .diagnostics_for_views(
            &context.device,
            BufferView::whole(&packed_input),
            BufferView::whole(&real_output),
        )
        .blockers()
        .is_empty());
    let c2r_readback = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.real.four_step.c2r_readback"),
        size: c2r.required_output_buffer_size_bytes(),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("wgpu_fft.test.real.four_step.c2r_encoder"),
        });
    c2r.execute_checked(&context.device, &mut encoder, &packed_input, &real_output)
        .unwrap();
    encoder.copy_buffer_to_buffer(
        &real_output,
        0,
        &c2r_readback,
        0,
        c2r.required_output_buffer_size_bytes(),
    );
    context.queue.submit([encoder.finish()]);
    let actual_real = read_f32(context, &c2r_readback, "c2r four-step child");
    let expected_real = reference_c2r_from_packed_interleaved(&packed, &inverse).unwrap();
    assert_close(&actual_real, &expected_real, "c2r four-step child");
}

#[derive(Debug, Clone, Copy)]
enum ViewMode {
    Direct,
    Offset,
    SegmentedBoth,
    SegmentedInput,
    SegmentedOutput,
    Strided,
}

fn run_r2c_case(
    context: &wgpu_fft::device::GpuContext,
    config: FftConfig,
    expected_route: C2cRoute,
    mode: ViewMode,
) {
    trace_gpu_step(&format!(
        "start r2c config={config:?} route={expected_route:?} mode={mode:?}"
    ));
    let input = real_input_for_config(&config);
    let expected = reference_r2c_packed_interleaved(&input, &config).unwrap();

    let plan = FftPlan::r2c(&context.device, &context.queue, config).unwrap();
    assert_eq!(plan.kind(), FftTransformKind::R2c);
    assert_eq!(plan.route(), expected_route);
    assert_eq!(
        plan.large_routing_policy().route_mode(),
        LargeRouteMode::Normal
    );
    assert_plan_diagnostics_graph(&context.device, &plan);
    assert_eq!(
        plan.required_input_buffer_size_bytes(),
        (input.len() * std::mem::size_of::<f32>()) as u64
    );
    assert_eq!(
        plan.required_output_buffer_size_bytes(),
        (expected.len() * std::mem::size_of::<f32>()) as u64
    );
    assert_eq!(
        plan.packed_shape().unwrap()[0],
        plan.config().shape()[0] / 2 + 1
    );

    let input_bytes = plan.required_input_buffer_size_bytes();
    let output_bytes = plan.required_output_buffer_size_bytes();
    let offset = if matches!(mode, ViewMode::Offset) {
        aligned_test_offset(context)
    } else {
        0
    };
    let input_size = offset + input_bytes + offset;
    let output_size = offset + output_bytes + offset;

    let input_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.r2c.input"),
        size: input_size,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    context
        .queue
        .write_buffer(&input_buffer, offset, bytemuck::cast_slice(&input));

    let output_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.r2c.output"),
        size: output_size,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let readback_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.r2c.readback"),
        size: output_bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let input_view = match mode {
        ViewMode::SegmentedBoth | ViewMode::SegmentedInput => {
            let segments = split_buffer_segments(&input_buffer, input_bytes, 4);
            BufferView::from_segments(&segments, 0, input_bytes).unwrap()
        }
        _ => BufferView::new(&input_buffer, offset, input_bytes).unwrap(),
    };
    let output_view = match mode {
        ViewMode::SegmentedBoth | ViewMode::SegmentedOutput => {
            let segments = split_buffer_segments(&output_buffer, output_bytes, 4);
            BufferView::from_segments(&segments, 0, output_bytes).unwrap()
        }
        _ => BufferView::new(&output_buffer, offset, output_bytes).unwrap(),
    };

    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("wgpu_fft.test.r2c.encoder"),
        });
    plan.execute_views(&context.device, &mut encoder, input_view, output_view)
        .unwrap();
    encoder.copy_buffer_to_buffer(&output_buffer, offset, &readback_buffer, 0, output_bytes);
    context.queue.submit([encoder.finish()]);
    trace_gpu_step(&format!(
        "submitted r2c config={:?} mode={mode:?}",
        plan.config()
    ));

    let label = format!("r2c {:?} {:?}", plan.config(), mode);
    let actual = read_f32(context, &readback_buffer, &label);
    assert_close(&actual, &expected, &label);
    trace_gpu_step(&format!("finish {label}"));
}

fn run_c2r_case(
    context: &wgpu_fft::device::GpuContext,
    config: FftConfig,
    expected_route: C2cRoute,
    mode: ViewMode,
) {
    trace_gpu_step(&format!(
        "start c2r config={config:?} route={expected_route:?} mode={mode:?}"
    ));
    let real = real_input_for_config(
        &FftConfig::new_nd(config.shape().to_vec()).with_batch(config.batch()),
    );
    let forward = FftConfig::new_nd(config.shape().to_vec())
        .with_batch(config.batch())
        .with_normalization(Normalization::None);
    let packed = reference_r2c_packed_interleaved(&real, &forward).unwrap();
    let expected = reference_c2r_from_packed_interleaved(&packed, &config).unwrap();

    let plan = FftPlan::c2r(&context.device, &context.queue, config).unwrap();
    assert_eq!(plan.kind(), FftTransformKind::C2r);
    assert_eq!(plan.route(), expected_route);
    assert_eq!(
        plan.large_routing_policy().route_mode(),
        LargeRouteMode::Normal
    );
    assert_plan_diagnostics_graph(&context.device, &plan);

    let input_bytes = plan.required_input_buffer_size_bytes();
    let output_bytes = plan.required_output_buffer_size_bytes();
    let offset = if matches!(mode, ViewMode::Offset) {
        aligned_test_offset(context)
    } else {
        0
    };
    let input_size = offset + input_bytes + offset;
    let output_size = offset + output_bytes + offset;

    let input_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.c2r.input"),
        size: input_size,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    context
        .queue
        .write_buffer(&input_buffer, offset, bytemuck::cast_slice(&packed));

    let output_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.c2r.output"),
        size: output_size,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let readback_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.c2r.readback"),
        size: output_bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let input_view = match mode {
        ViewMode::SegmentedBoth | ViewMode::SegmentedInput => {
            let segments = split_buffer_segments(&input_buffer, input_bytes, 4);
            BufferView::from_segments(&segments, 0, input_bytes).unwrap()
        }
        _ => BufferView::new(&input_buffer, offset, input_bytes).unwrap(),
    };
    let output_view = match mode {
        ViewMode::SegmentedBoth | ViewMode::SegmentedOutput => {
            let segments = split_buffer_segments(&output_buffer, output_bytes, 4);
            BufferView::from_segments(&segments, 0, output_bytes).unwrap()
        }
        _ => BufferView::new(&output_buffer, offset, output_bytes).unwrap(),
    };

    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("wgpu_fft.test.c2r.encoder"),
        });
    plan.execute_views(&context.device, &mut encoder, input_view, output_view)
        .unwrap();
    encoder.copy_buffer_to_buffer(&output_buffer, offset, &readback_buffer, 0, output_bytes);
    context.queue.submit([encoder.finish()]);
    trace_gpu_step(&format!(
        "submitted c2r config={:?} mode={mode:?}",
        plan.config()
    ));

    let label = format!("c2r {:?} {:?}", plan.config(), mode);
    let actual = read_f32(context, &readback_buffer, &label);
    assert_close(&actual, &expected, &label);
    trace_gpu_step(&format!("finish {label}"));
}

fn run_r2c_strided_logical_case(context: &wgpu_fft::device::GpuContext) {
    let config = FftConfig::new(16).with_normalization(Normalization::None);
    trace_gpu_step(&format!("start r2c logical strided config={config:?}"));
    let input = real_input_for_config(&config);
    let expected = reference_r2c_packed_interleaved(&input, &config).unwrap();
    let logical_per_batch = config.logical_complex_len().unwrap() as u64;
    let packed_per_batch = (config.shape()[0] / 2 + 1) as u64;
    let input_layout = strided_layout(logical_per_batch, 3);
    let output_layout = strided_layout(packed_per_batch, 5);
    let input_span_bytes = layout_span(input_layout, logical_per_batch, 1) * 4;
    let output_span_bytes = layout_span(output_layout, packed_per_batch, 1) * 8;
    let physical_input = scatter_strided_scalar(&input, input_layout, logical_per_batch, 1);

    let input_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.r2c_logical_strided_input"),
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
        label: Some("wgpu_fft.test.r2c_logical_strided_output"),
        size: output_span_bytes,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let readback_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.r2c_logical_strided_readback"),
        size: output_span_bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let output_segments = split_buffer_segments(&output_buffer, output_span_bytes, 4);
    let plan = FftPlan::r2c(&context.device, &context.queue, config).unwrap();
    assert_eq!(plan.route(), C2cRoute::MixedRadix);
    assert!(!plan.diagnostics().stages().is_empty());
    let input_logical =
        FftLogicalView::new(BufferView::whole(&input_buffer), input_layout).unwrap();
    let output_logical = FftLogicalView::new(
        BufferView::from_segments(&output_segments, 0, output_span_bytes).unwrap(),
        output_layout,
    )
    .unwrap();
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
    let forced_limits = FftDeviceLimits {
        max_storage_buffer_binding_size: 32,
        max_buffer_size: 32,
        min_storage_buffer_offset_alignment: 1,
    };
    let forced_diagnostics = plan.diagnostics_for_logical_views_with_limits(
        forced_limits,
        &input_logical,
        &output_logical,
    );
    assert_eq!(forced_diagnostics.device_limits(), Some(forced_limits));
    assert!(forced_diagnostics.blockers().iter().any(|blocker| {
        blocker.kind == FftBlockerKind::HelperBuffer
            && blocker.route.as_deref() == Some("mixed-radix")
            && blocker.stage.as_deref() == Some("input-strided-pack")
            && blocker.helper_buffer.as_deref() == Some("input-logical-stage")
    }));
    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("wgpu_fft.test.r2c_logical_strided_encoder"),
        });
    plan.execute_logical_views(&context.device, &mut encoder, input_logical, output_logical)
        .unwrap();
    encoder.copy_buffer_to_buffer(&output_buffer, 0, &readback_buffer, 0, output_span_bytes);
    context.queue.submit([encoder.finish()]);

    let label = "r2c logical strided";
    let actual_physical = read_f32(context, &readback_buffer, label);
    let actual = gather_strided_complex(&actual_physical, output_layout, packed_per_batch, 1);
    assert_close(&actual, &expected, label);
    trace_gpu_step(&format!("finish {label}"));
}

fn run_c2r_strided_logical_case(context: &wgpu_fft::device::GpuContext) {
    let config = FftConfig::inverse(16);
    trace_gpu_step(&format!("start c2r logical strided config={config:?}"));
    let real = real_input_for_config(&FftConfig::new(16));
    let packed = reference_r2c_packed_interleaved(
        &real,
        &FftConfig::new(16).with_normalization(Normalization::None),
    )
    .unwrap();
    let expected = reference_c2r_from_packed_interleaved(&packed, &config).unwrap();
    let logical_per_batch = config.logical_complex_len().unwrap() as u64;
    let packed_per_batch = (config.shape()[0] / 2 + 1) as u64;
    let input_layout = strided_layout(packed_per_batch, 3);
    let output_layout = strided_layout(logical_per_batch, 5);
    let input_span_bytes = layout_span(input_layout, packed_per_batch, 1) * 8;
    let output_span_bytes = layout_span(output_layout, logical_per_batch, 1) * 4;
    let physical_input = scatter_strided_complex(&packed, input_layout, packed_per_batch, 1);

    let input_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.c2r_logical_strided_input"),
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
        label: Some("wgpu_fft.test.c2r_logical_strided_output"),
        size: output_span_bytes,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let readback_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.c2r_logical_strided_readback"),
        size: output_span_bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let input_segments = split_buffer_segments(&input_buffer, input_span_bytes, 4);
    let plan = FftPlan::c2r(&context.device, &context.queue, config).unwrap();
    assert_eq!(plan.route(), C2cRoute::MixedRadix);
    let input_logical = FftLogicalView::new(
        BufferView::from_segments(&input_segments, 0, input_span_bytes).unwrap(),
        input_layout,
    )
    .unwrap();
    let output_logical =
        FftLogicalView::new(BufferView::whole(&output_buffer), output_layout).unwrap();
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
            label: Some("wgpu_fft.test.c2r_logical_strided_encoder"),
        });
    plan.execute_logical_views(&context.device, &mut encoder, input_logical, output_logical)
        .unwrap();
    encoder.copy_buffer_to_buffer(&output_buffer, 0, &readback_buffer, 0, output_span_bytes);
    context.queue.submit([encoder.finish()]);

    let label = "c2r logical strided";
    let actual_physical = read_f32(context, &readback_buffer, label);
    let actual = gather_strided_scalar(&actual_physical, output_layout, logical_per_batch, 1);
    assert_close(&actual, &expected, label);
    trace_gpu_step(&format!("finish {label}"));
}

fn run_r2c_segmented_strided_logical_case(
    context: &wgpu_fft::device::GpuContext,
    config: FftConfig,
    expected_route: C2cRoute,
    limits: Option<LargePolicyLimits>,
    expected_route_mode: LargeRouteMode,
) {
    trace_gpu_step(&format!(
        "start r2c segmented+strided logical config={config:?} route={expected_route:?}"
    ));
    let input = real_input_for_config(&config);
    let expected = reference_r2c_packed_interleaved(&input, &config).unwrap();
    let plan = if let Some(limits) = limits {
        FftPlan::r2c_with_large_policy_limits_for_testing(
            &context.device,
            &context.queue,
            config,
            limits,
        )
        .unwrap()
    } else {
        FftPlan::r2c(&context.device, &context.queue, config).unwrap()
    };
    assert_eq!(plan.kind(), FftTransformKind::R2c);
    assert_eq!(plan.route(), expected_route);
    assert_eq!(
        plan.large_routing_policy().route_mode(),
        expected_route_mode
    );

    let input_bytes = plan.required_input_buffer_size_bytes();
    let output_bytes = plan.required_output_buffer_size_bytes();
    let batch = plan.config().batch() as u64;
    let real_per_batch = input_bytes / batch / std::mem::size_of::<f32>() as u64;
    let packed_per_batch = output_bytes / batch / (2 * std::mem::size_of::<f32>() as u64);
    let input_layout = strided_layout(real_per_batch, 3);
    let output_layout = strided_layout(packed_per_batch, 5);
    let input_span_bytes =
        layout_span(input_layout, real_per_batch, batch) * std::mem::size_of::<f32>() as u64;
    let output_span_bytes =
        layout_span(output_layout, packed_per_batch, batch) * 2 * std::mem::size_of::<f32>() as u64;
    let physical_input = scatter_strided_scalar(&input, input_layout, real_per_batch, batch);

    let input_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.r2c_segmented_strided_input"),
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
        label: Some("wgpu_fft.test.r2c_segmented_strided_output"),
        size: output_span_bytes,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let readback_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.r2c_segmented_strided_readback"),
        size: output_span_bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let input_segments = split_buffer_segments(&input_buffer, input_span_bytes, 4);
    let output_segments = split_buffer_segments(&output_buffer, output_span_bytes, 4);
    let input_logical = FftLogicalView::new(
        BufferView::from_segments(&input_segments, 0, input_span_bytes).unwrap(),
        input_layout,
    )
    .unwrap();
    let output_logical = FftLogicalView::new(
        BufferView::from_segments(&output_segments, 0, output_span_bytes).unwrap(),
        output_layout,
    )
    .unwrap();
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

    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("wgpu_fft.test.r2c_segmented_strided_encoder"),
        });
    plan.execute_logical_views(&context.device, &mut encoder, input_logical, output_logical)
        .unwrap();
    encoder.copy_buffer_to_buffer(&output_buffer, 0, &readback_buffer, 0, output_span_bytes);
    context.queue.submit([encoder.finish()]);

    let label = format!("r2c segmented+strided logical {:?}", plan.config());
    let actual_physical = read_f32(context, &readback_buffer, &label);
    let actual = gather_strided_complex(&actual_physical, output_layout, packed_per_batch, batch);
    assert_close(&actual, &expected, &label);
    trace_gpu_step(&format!("finish {label}"));
}

fn run_c2r_segmented_strided_logical_case(
    context: &wgpu_fft::device::GpuContext,
    config: FftConfig,
    expected_route: C2cRoute,
    limits: Option<LargePolicyLimits>,
    expected_route_mode: LargeRouteMode,
) {
    trace_gpu_step(&format!(
        "start c2r segmented+strided logical config={config:?} route={expected_route:?}"
    ));
    let real = real_input_for_config(
        &FftConfig::new_nd(config.shape().to_vec()).with_batch(config.batch()),
    );
    let forward = FftConfig::new_nd(config.shape().to_vec())
        .with_batch(config.batch())
        .with_normalization(Normalization::None);
    let packed = reference_r2c_packed_interleaved(&real, &forward).unwrap();
    let expected = reference_c2r_from_packed_interleaved(&packed, &config).unwrap();
    let plan = if let Some(limits) = limits {
        FftPlan::c2r_with_large_policy_limits_for_testing(
            &context.device,
            &context.queue,
            config,
            limits,
        )
        .unwrap()
    } else {
        FftPlan::c2r(&context.device, &context.queue, config).unwrap()
    };
    assert_eq!(plan.kind(), FftTransformKind::C2r);
    assert_eq!(plan.route(), expected_route);
    assert_eq!(
        plan.large_routing_policy().route_mode(),
        expected_route_mode
    );

    let input_bytes = plan.required_input_buffer_size_bytes();
    let output_bytes = plan.required_output_buffer_size_bytes();
    let batch = plan.config().batch() as u64;
    let packed_per_batch = input_bytes / batch / (2 * std::mem::size_of::<f32>() as u64);
    let real_per_batch = output_bytes / batch / std::mem::size_of::<f32>() as u64;
    let input_layout = strided_layout(packed_per_batch, 3);
    let output_layout = strided_layout(real_per_batch, 5);
    let input_span_bytes =
        layout_span(input_layout, packed_per_batch, batch) * 2 * std::mem::size_of::<f32>() as u64;
    let output_span_bytes =
        layout_span(output_layout, real_per_batch, batch) * std::mem::size_of::<f32>() as u64;
    let physical_input = scatter_strided_complex(&packed, input_layout, packed_per_batch, batch);

    let input_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.c2r_segmented_strided_input"),
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
        label: Some("wgpu_fft.test.c2r_segmented_strided_output"),
        size: output_span_bytes,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let readback_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.c2r_segmented_strided_readback"),
        size: output_span_bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let input_segments = split_buffer_segments(&input_buffer, input_span_bytes, 4);
    let output_segments = split_buffer_segments(&output_buffer, output_span_bytes, 4);
    let input_logical = FftLogicalView::new(
        BufferView::from_segments(&input_segments, 0, input_span_bytes).unwrap(),
        input_layout,
    )
    .unwrap();
    let output_logical = FftLogicalView::new(
        BufferView::from_segments(&output_segments, 0, output_span_bytes).unwrap(),
        output_layout,
    )
    .unwrap();
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

    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("wgpu_fft.test.c2r_segmented_strided_encoder"),
        });
    plan.execute_logical_views(&context.device, &mut encoder, input_logical, output_logical)
        .unwrap();
    encoder.copy_buffer_to_buffer(&output_buffer, 0, &readback_buffer, 0, output_span_bytes);
    context.queue.submit([encoder.finish()]);

    let label = format!("c2r segmented+strided logical {:?}", plan.config());
    let actual_physical = read_f32(context, &readback_buffer, &label);
    let actual = gather_strided_scalar(&actual_physical, output_layout, real_per_batch, batch);
    assert_close(&actual, &expected, &label);
    trace_gpu_step(&format!("finish {label}"));
}

fn assert_real_logical_execution_diagnostics(context: &wgpu_fft::device::GpuContext) {
    assert_r2c_segmented_strided_usage_execution_diagnostics(context);
    assert_c2r_segmented_strided_usage_execution_diagnostics(context);
}

fn assert_r2c_segmented_strided_usage_execution_diagnostics(
    context: &wgpu_fft::device::GpuContext,
) {
    let config = FftConfig::new(16).with_normalization(Normalization::None);
    let input = real_input_for_config(&config);
    let plan = FftPlan::r2c(&context.device, &context.queue, config).unwrap();
    let input_bytes = plan.required_input_buffer_size_bytes();
    let output_bytes = plan.required_output_buffer_size_bytes();
    let real_per_batch = input_bytes / std::mem::size_of::<f32>() as u64;
    let packed_per_batch = output_bytes / (2 * std::mem::size_of::<f32>() as u64);
    let input_layout = strided_layout(real_per_batch, 3);
    let output_layout = strided_layout(packed_per_batch, 5);
    let input_span_bytes =
        layout_span(input_layout, real_per_batch, 1) * std::mem::size_of::<f32>() as u64;
    let output_span_bytes =
        layout_span(output_layout, packed_per_batch, 1) * 2 * std::mem::size_of::<f32>() as u64;
    let physical_input = scatter_strided_scalar(&input, input_layout, real_per_batch, 1);

    let input_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.r2c_segmented_strided_missing_copy_src_input"),
        size: input_span_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    context
        .queue
        .write_buffer(&input_buffer, 0, bytemuck::cast_slice(&physical_input));
    let output_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.r2c_segmented_strided_missing_copy_dst_output"),
        size: output_span_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let input_segments = split_buffer_segments(&input_buffer, input_span_bytes, 4);
    let output_segments = split_buffer_segments(&output_buffer, output_span_bytes, 4);
    let input_logical = FftLogicalView::new(
        BufferView::from_segments(&input_segments, 0, input_span_bytes).unwrap(),
        input_layout,
    )
    .unwrap();
    let output_logical = FftLogicalView::new(
        BufferView::from_segments(&output_segments, 0, output_span_bytes).unwrap(),
        output_layout,
    )
    .unwrap();

    let diagnostics =
        plan.diagnostics_for_logical_views(&context.device, &input_logical, &output_logical);
    assert_real_segmented_strided_usage_diagnostics(&diagnostics, "r2c");

    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("wgpu_fft.test.r2c_segmented_strided_diagnostics_encoder"),
        });
    let error = plan
        .execute_logical_views_with_diagnostics(
            &context.device,
            &mut encoder,
            input_logical,
            output_logical,
        )
        .unwrap_err();
    assert_eq!(
        error.error(),
        &FftError::BufferViewMissingUsage { usage: "COPY_SRC" }
    );
    assert_real_segmented_strided_usage_diagnostics(error.diagnostics(), "r2c");
}

fn assert_c2r_segmented_strided_usage_execution_diagnostics(
    context: &wgpu_fft::device::GpuContext,
) {
    let config = FftConfig::inverse(16);
    let real = real_input_for_config(&FftConfig::new(16));
    let packed = reference_r2c_packed_interleaved(
        &real,
        &FftConfig::new(16).with_normalization(Normalization::None),
    )
    .unwrap();
    let plan = FftPlan::c2r(&context.device, &context.queue, config).unwrap();
    let input_bytes = plan.required_input_buffer_size_bytes();
    let output_bytes = plan.required_output_buffer_size_bytes();
    let packed_per_batch = input_bytes / (2 * std::mem::size_of::<f32>() as u64);
    let real_per_batch = output_bytes / std::mem::size_of::<f32>() as u64;
    let input_layout = strided_layout(packed_per_batch, 3);
    let output_layout = strided_layout(real_per_batch, 5);
    let input_span_bytes =
        layout_span(input_layout, packed_per_batch, 1) * 2 * std::mem::size_of::<f32>() as u64;
    let output_span_bytes =
        layout_span(output_layout, real_per_batch, 1) * std::mem::size_of::<f32>() as u64;
    let physical_input = scatter_strided_complex(&packed, input_layout, packed_per_batch, 1);

    let input_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.c2r_segmented_strided_missing_copy_src_input"),
        size: input_span_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    context
        .queue
        .write_buffer(&input_buffer, 0, bytemuck::cast_slice(&physical_input));
    let output_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.c2r_segmented_strided_missing_copy_dst_output"),
        size: output_span_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let input_segments = split_buffer_segments(&input_buffer, input_span_bytes, 4);
    let output_segments = split_buffer_segments(&output_buffer, output_span_bytes, 4);
    let input_logical = FftLogicalView::new(
        BufferView::from_segments(&input_segments, 0, input_span_bytes).unwrap(),
        input_layout,
    )
    .unwrap();
    let output_logical = FftLogicalView::new(
        BufferView::from_segments(&output_segments, 0, output_span_bytes).unwrap(),
        output_layout,
    )
    .unwrap();

    let diagnostics =
        plan.diagnostics_for_logical_views(&context.device, &input_logical, &output_logical);
    assert_real_segmented_strided_usage_diagnostics(&diagnostics, "c2r");

    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("wgpu_fft.test.c2r_segmented_strided_diagnostics_encoder"),
        });
    let error = plan
        .execute_logical_views_with_diagnostics(
            &context.device,
            &mut encoder,
            input_logical,
            output_logical,
        )
        .unwrap_err();
    assert_eq!(
        error.error(),
        &FftError::BufferViewMissingUsage { usage: "COPY_SRC" }
    );
    assert_real_segmented_strided_usage_diagnostics(error.diagnostics(), "c2r");
}

fn assert_real_segmented_strided_usage_diagnostics(
    diagnostics: &wgpu_fft::FftDiagnostics,
    transform: &'static str,
) {
    assert_eq!(diagnostics.route().transform, transform);
    assert_eq!(diagnostics.route().route, "mixed-radix");
    assert!(diagnostics
        .stages()
        .iter()
        .any(|stage| stage.label == "input-segmented-copy-window"));
    assert!(diagnostics
        .stages()
        .iter()
        .any(|stage| stage.label == "output-segmented-copy-window"));
    assert!(diagnostics
        .stages()
        .iter()
        .any(|stage| stage.kind == "strided-pack"));
    assert!(diagnostics
        .stages()
        .iter()
        .any(|stage| stage.kind == "strided-unpack"));
    assert!(diagnostics
        .stages()
        .iter()
        .any(|stage| stage.label.starts_with(transform) && stage.route == transform));
    assert!(diagnostics.stages().iter().any(|stage| {
        matches!(
            stage.label.as_str(),
            "mixed-radix-stockham-stage"
                | "fused-pow2-workgroup-stage"
                | "fused-smooth-workgroup-stage"
        ) && stage.route == "mixed-radix"
    }));
    assert!(diagnostics.blockers().iter().any(|blocker| {
        blocker.kind == FftBlockerKind::BufferUsage
            && blocker.route.as_deref() == Some("mixed-radix")
            && blocker.stage.as_deref() == Some("input-segmented-copy-window")
            && blocker.layout.as_deref() == Some("segmented-strided")
            && blocker.reason.contains("COPY_SRC")
    }));
    assert!(diagnostics.blockers().iter().any(|blocker| {
        blocker.kind == FftBlockerKind::BufferUsage
            && blocker.route.as_deref() == Some("mixed-radix")
            && blocker.stage.as_deref() == Some("output-segmented-copy-window")
            && blocker.layout.as_deref() == Some("segmented-strided")
            && blocker.reason.contains("COPY_DST")
    }));
}

fn run_r2c_strided_io_view_case(context: &wgpu_fft::device::GpuContext) {
    let config = FftConfig::new(16).with_normalization(Normalization::None);
    trace_gpu_step(&format!("start r2c FftIoView strided config={config:?}"));
    let input = real_input_for_config(&config);
    let expected = reference_r2c_packed_interleaved(&input, &config).unwrap();
    let logical_per_batch = config.logical_complex_len().unwrap() as u64;
    let packed_per_batch = (config.shape()[0] / 2 + 1) as u64;
    let input_layout = strided_layout(logical_per_batch, 4);
    let output_layout = strided_layout(packed_per_batch, 6);
    let input_span_bytes = layout_span(input_layout, logical_per_batch, 1) * 4;
    let output_span_bytes = layout_span(output_layout, packed_per_batch, 1) * 8;
    let physical_input = scatter_strided_scalar(&input, input_layout, logical_per_batch, 1);

    let input_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.r2c_io_strided_input"),
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
        label: Some("wgpu_fft.test.r2c_io_strided_output"),
        size: output_span_bytes,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let readback_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.r2c_io_strided_readback"),
        size: output_span_bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let plan = FftPlan::r2c(&context.device, &context.queue, config).unwrap();
    let input_io = FftIoView::new(
        BufferView::whole(&input_buffer),
        buffer_layout_from_logical(input_layout),
    )
    .unwrap();
    let output_io = FftIoView::new(
        BufferView::whole(&output_buffer),
        buffer_layout_from_logical(output_layout),
    )
    .unwrap();
    let io_diagnostics =
        plan.diagnostics_for_io_views(&context.device, input_io.clone(), output_io.clone());
    assert!(io_diagnostics
        .stages()
        .iter()
        .any(|stage| stage.kind == "strided-pack"));
    assert!(io_diagnostics
        .stages()
        .iter()
        .any(|stage| stage.kind == "strided-unpack"));
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
    assert!(forced_io_diagnostics.blockers().iter().any(|blocker| {
        blocker.kind == FftBlockerKind::HelperBuffer
            && blocker.route.as_deref() == Some("mixed-radix")
            && blocker.stage.as_deref() == Some("input-strided-pack")
            && blocker.helper_buffer.as_deref() == Some("input-logical-stage")
    }));
    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("wgpu_fft.test.r2c_io_strided_encoder"),
        });
    plan.execute_io_views(&context.device, &mut encoder, input_io, output_io)
        .unwrap();
    encoder.copy_buffer_to_buffer(&output_buffer, 0, &readback_buffer, 0, output_span_bytes);
    context.queue.submit([encoder.finish()]);

    let label = "r2c FftIoView strided";
    let actual_physical = read_f32(context, &readback_buffer, label);
    let actual = gather_strided_complex(&actual_physical, output_layout, packed_per_batch, 1);
    assert_close(&actual, &expected, label);
    trace_gpu_step(&format!("finish {label}"));
}

fn run_c2r_strided_io_view_case(context: &wgpu_fft::device::GpuContext) {
    let config = FftConfig::inverse(16);
    trace_gpu_step(&format!("start c2r FftIoView strided config={config:?}"));
    let real = real_input_for_config(&FftConfig::new(16));
    let packed = reference_r2c_packed_interleaved(
        &real,
        &FftConfig::new(16).with_normalization(Normalization::None),
    )
    .unwrap();
    let expected = reference_c2r_from_packed_interleaved(&packed, &config).unwrap();
    let logical_per_batch = config.logical_complex_len().unwrap() as u64;
    let packed_per_batch = (config.shape()[0] / 2 + 1) as u64;
    let input_layout = strided_layout(packed_per_batch, 4);
    let output_layout = strided_layout(logical_per_batch, 6);
    let input_span_bytes = layout_span(input_layout, packed_per_batch, 1) * 8;
    let output_span_bytes = layout_span(output_layout, logical_per_batch, 1) * 4;
    let physical_input = scatter_strided_complex(&packed, input_layout, packed_per_batch, 1);

    let input_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.c2r_io_strided_input"),
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
        label: Some("wgpu_fft.test.c2r_io_strided_output"),
        size: output_span_bytes,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let readback_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.c2r_io_strided_readback"),
        size: output_span_bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let plan = FftPlan::c2r(&context.device, &context.queue, config).unwrap();
    let input_io = FftIoView::new(
        BufferView::whole(&input_buffer),
        buffer_layout_from_logical(input_layout),
    )
    .unwrap();
    let output_io = FftIoView::new(
        BufferView::whole(&output_buffer),
        buffer_layout_from_logical(output_layout),
    )
    .unwrap();
    let io_diagnostics =
        plan.diagnostics_for_io_views(&context.device, input_io.clone(), output_io.clone());
    assert!(io_diagnostics
        .stages()
        .iter()
        .any(|stage| stage.kind == "strided-pack"));
    assert!(io_diagnostics
        .stages()
        .iter()
        .any(|stage| stage.kind == "strided-unpack"));
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
    assert!(forced_io_diagnostics.blockers().iter().any(|blocker| {
        blocker.kind == FftBlockerKind::HelperBuffer
            && blocker.route.as_deref() == Some("mixed-radix")
            && blocker.stage.as_deref() == Some("output-strided-unpack")
            && blocker.helper_buffer.as_deref() == Some("output-logical-stage")
    }));
    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("wgpu_fft.test.c2r_io_strided_encoder"),
        });
    plan.execute_io_views(&context.device, &mut encoder, input_io, output_io)
        .unwrap();
    encoder.copy_buffer_to_buffer(&output_buffer, 0, &readback_buffer, 0, output_span_bytes);
    context.queue.submit([encoder.finish()]);

    let label = "c2r FftIoView strided";
    let actual_physical = read_f32(context, &readback_buffer, label);
    let actual = gather_strided_scalar(&actual_physical, output_layout, logical_per_batch, 1);
    assert_close(&actual, &expected, label);
    trace_gpu_step(&format!("finish {label}"));
}

fn run_r2c_large_chunk_case(
    context: &wgpu_fft::device::GpuContext,
    config: FftConfig,
    expected_route: C2cRoute,
    mode: ViewMode,
) {
    trace_gpu_step(&format!(
        "start r2c large-chunk config={config:?} route={expected_route:?} mode={mode:?}"
    ));
    let input = real_input_for_config(&config);
    let expected = reference_r2c_packed_interleaved(&input, &config).unwrap();
    // Public max-buffer caps also apply to the child FFT's route-owned
    // helpers. Leave four batches of capacity for Bluestein so its one-line
    // convolution helper fits while the five-batch endpoint still chunks.
    let chunk_batches = if expected_route == C2cRoute::Bluestein {
        4
    } else {
        2
    };
    let limits = real_large_chunk_test_limits(&config, chunk_batches);
    let plan = FftPlan::r2c_with_large_policy_limits_for_testing(
        &context.device,
        &context.queue,
        config,
        limits,
    )
    .unwrap();
    assert_eq!(plan.kind(), FftTransformKind::R2c);
    assert_eq!(plan.route(), expected_route);
    assert_eq!(
        plan.large_routing_policy().route_mode(),
        LargeRouteMode::LargeChunk
    );
    assert_eq!(plan.workspace_size_bytes(), 0);

    let input_bytes = plan.required_input_buffer_size_bytes();
    let output_bytes = plan.required_output_buffer_size_bytes();
    let batch = plan.config().batch() as u64;
    let real_per_batch = input_bytes / batch / std::mem::size_of::<f32>() as u64;
    let packed_per_batch = output_bytes / batch / (2 * std::mem::size_of::<f32>() as u64);
    let input_layout = strided_layout(real_per_batch, 3);
    let output_layout = strided_layout(packed_per_batch, 5);
    let input_view_bytes = if matches!(mode, ViewMode::Strided) {
        layout_span(input_layout, real_per_batch, batch) * std::mem::size_of::<f32>() as u64
    } else {
        input_bytes
    };
    let output_view_bytes = if matches!(mode, ViewMode::Strided) {
        layout_span(output_layout, packed_per_batch, batch) * 2 * std::mem::size_of::<f32>() as u64
    } else {
        output_bytes
    };
    let offset = if matches!(mode, ViewMode::Offset) {
        aligned_test_offset(context)
    } else {
        0
    };
    let input_size = offset + input_view_bytes + offset;
    let output_size = offset + output_view_bytes + offset;

    let input_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.r2c_large_chunk.input"),
        size: input_size,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let physical_input = if matches!(mode, ViewMode::Strided) {
        scatter_strided_scalar(&input, input_layout, real_per_batch, batch)
    } else {
        input.clone()
    };
    context
        .queue
        .write_buffer(&input_buffer, offset, bytemuck::cast_slice(&physical_input));
    let output_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.r2c_large_chunk.output"),
        size: output_size,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let readback_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.r2c_large_chunk.readback"),
        size: output_view_bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let input_view = match mode {
        ViewMode::SegmentedBoth | ViewMode::SegmentedInput => {
            let segments = split_buffer_segments(&input_buffer, input_bytes, 4);
            BufferView::from_segments(&segments, 0, input_bytes).unwrap()
        }
        _ => BufferView::new(&input_buffer, offset, input_view_bytes).unwrap(),
    };
    let output_view = match mode {
        ViewMode::SegmentedBoth | ViewMode::SegmentedOutput => {
            let segments = split_buffer_segments(&output_buffer, output_bytes, 4);
            BufferView::from_segments(&segments, 0, output_bytes).unwrap()
        }
        _ => BufferView::new(&output_buffer, offset, output_view_bytes).unwrap(),
    };

    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("wgpu_fft.test.r2c_large_chunk.encoder"),
        });
    if matches!(mode, ViewMode::Strided) {
        let input_logical = FftLogicalView::new(input_view, input_layout).unwrap();
        let output_logical = FftLogicalView::new(output_view, output_layout).unwrap();
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

    let label = format!("r2c large-chunk {:?} {:?}", plan.config(), mode);
    let actual_physical = read_f32(context, &readback_buffer, &label);
    let actual = if matches!(mode, ViewMode::Strided) {
        gather_strided_complex(&actual_physical, output_layout, packed_per_batch, batch)
    } else {
        actual_physical
    };
    assert_close(&actual, &expected, &label);
    trace_gpu_step(&format!("finish {label}"));
}

fn run_c2r_large_chunk_case(
    context: &wgpu_fft::device::GpuContext,
    config: FftConfig,
    expected_route: C2cRoute,
    mode: ViewMode,
) {
    trace_gpu_step(&format!(
        "start c2r large-chunk config={config:?} route={expected_route:?} mode={mode:?}"
    ));
    let real = real_input_for_config(
        &FftConfig::new_nd(config.shape().to_vec()).with_batch(config.batch()),
    );
    let forward = FftConfig::new_nd(config.shape().to_vec())
        .with_batch(config.batch())
        .with_normalization(Normalization::None);
    let packed = reference_r2c_packed_interleaved(&real, &forward).unwrap();
    let expected = reference_c2r_from_packed_interleaved(&packed, &config).unwrap();

    let chunk_batches = if expected_route == C2cRoute::Bluestein {
        4
    } else {
        2
    };
    let limits = real_large_chunk_test_limits(&config, chunk_batches);
    let plan = FftPlan::c2r_with_large_policy_limits_for_testing(
        &context.device,
        &context.queue,
        config,
        limits,
    )
    .unwrap();
    assert_eq!(plan.kind(), FftTransformKind::C2r);
    assert_eq!(plan.route(), expected_route);
    assert_eq!(
        plan.large_routing_policy().route_mode(),
        LargeRouteMode::LargeChunk
    );
    assert_eq!(plan.workspace_size_bytes(), 0);

    let input_bytes = plan.required_input_buffer_size_bytes();
    let output_bytes = plan.required_output_buffer_size_bytes();
    let batch = plan.config().batch() as u64;
    let packed_per_batch = input_bytes / batch / (2 * std::mem::size_of::<f32>() as u64);
    let real_per_batch = output_bytes / batch / std::mem::size_of::<f32>() as u64;
    let input_layout = strided_layout(packed_per_batch, 3);
    let output_layout = strided_layout(real_per_batch, 5);
    let input_view_bytes = if matches!(mode, ViewMode::Strided) {
        layout_span(input_layout, packed_per_batch, batch) * 2 * std::mem::size_of::<f32>() as u64
    } else {
        input_bytes
    };
    let output_view_bytes = if matches!(mode, ViewMode::Strided) {
        layout_span(output_layout, real_per_batch, batch) * std::mem::size_of::<f32>() as u64
    } else {
        output_bytes
    };
    let offset = if matches!(mode, ViewMode::Offset) {
        aligned_test_offset(context)
    } else {
        0
    };
    let input_size = offset + input_view_bytes + offset;
    let output_size = offset + output_view_bytes + offset;

    let input_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.c2r_large_chunk.input"),
        size: input_size,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let physical_input = if matches!(mode, ViewMode::Strided) {
        scatter_strided_complex(&packed, input_layout, packed_per_batch, batch)
    } else {
        packed.clone()
    };
    context
        .queue
        .write_buffer(&input_buffer, offset, bytemuck::cast_slice(&physical_input));
    let output_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.c2r_large_chunk.output"),
        size: output_size,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let readback_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.c2r_large_chunk.readback"),
        size: output_view_bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let input_view = match mode {
        ViewMode::SegmentedBoth | ViewMode::SegmentedInput => {
            let segments = split_buffer_segments(&input_buffer, input_bytes, 4);
            BufferView::from_segments(&segments, 0, input_bytes).unwrap()
        }
        _ => BufferView::new(&input_buffer, offset, input_view_bytes).unwrap(),
    };
    let output_view = match mode {
        ViewMode::SegmentedBoth | ViewMode::SegmentedOutput => {
            let segments = split_buffer_segments(&output_buffer, output_bytes, 4);
            BufferView::from_segments(&segments, 0, output_bytes).unwrap()
        }
        _ => BufferView::new(&output_buffer, offset, output_view_bytes).unwrap(),
    };

    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("wgpu_fft.test.c2r_large_chunk.encoder"),
        });
    if matches!(mode, ViewMode::Strided) {
        let input_logical = FftLogicalView::new(input_view, input_layout).unwrap();
        let output_logical = FftLogicalView::new(output_view, output_layout).unwrap();
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

    let label = format!("c2r large-chunk {:?} {:?}", plan.config(), mode);
    let actual_physical = read_f32(context, &readback_buffer, &label);
    let actual = if matches!(mode, ViewMode::Strided) {
        gather_strided_scalar(&actual_physical, output_layout, real_per_batch, batch)
    } else {
        actual_physical
    };
    assert_close(&actual, &expected, &label);
    trace_gpu_step(&format!("finish {label}"));
}

fn run_r2c_large_single_decomposition_case(
    context: &wgpu_fft::device::GpuContext,
    config: FftConfig,
    expected_route: C2cRoute,
    mode: ViewMode,
) {
    trace_gpu_step(&format!(
        "start r2c single large-decomposition config={config:?} route={expected_route:?} mode={mode:?}"
    ));
    let input = real_input_for_config(&config);
    let expected = reference_r2c_packed_interleaved(&input, &config).unwrap();
    let limits = real_single_decomposition_test_limits(&config);
    let plan = FftPlan::r2c_with_large_policy_limits_for_testing(
        &context.device,
        &context.queue,
        config,
        limits,
    )
    .unwrap();
    assert_eq!(plan.kind(), FftTransformKind::R2c);
    assert_eq!(plan.route(), expected_route);
    assert_eq!(
        plan.large_routing_policy().route_mode(),
        LargeRouteMode::LargeChunk
    );
    let expected_execution_kind = if expected_route == C2cRoute::AxisSequence {
        wgpu_fft::LargeExecutionKind::AxisDecomposition
    } else {
        wgpu_fft::LargeExecutionKind::Smooth1dDecomposition
    };
    assert_eq!(
        plan.large_routing_policy().execution_kind(),
        expected_execution_kind
    );
    let route_diagnostics = plan.large_routing_policy().diagnostics();
    if expected_execution_kind == wgpu_fft::LargeExecutionKind::Smooth1dDecomposition {
        assert_eq!(route_diagnostics.selected_axis, Some(0));
        assert_eq!(route_diagnostics.factor_splits.len(), 1);
    } else {
        assert!(route_diagnostics.factor_splits.len() >= 2);
    }
    let device_diagnostics = plan.diagnostics_for_device(&context.device);
    assert!(device_diagnostics.blockers().is_empty());
    if expected_execution_kind == wgpu_fft::LargeExecutionKind::AxisDecomposition {
        assert!(device_diagnostics
            .stages()
            .iter()
            .any(|stage| stage.label == "large-axis-sequence-workspace"));
    }

    let input_bytes = plan.required_input_buffer_size_bytes();
    let output_bytes = plan.required_output_buffer_size_bytes();
    let batch = plan.config().batch() as u64;
    let real_per_batch = input_bytes / batch / std::mem::size_of::<f32>() as u64;
    let packed_per_batch = output_bytes / batch / (2 * std::mem::size_of::<f32>() as u64);
    let input_layout = strided_layout(real_per_batch, 3);
    let output_layout = strided_layout(packed_per_batch, 5);
    let input_view_bytes = if matches!(mode, ViewMode::Strided) {
        layout_span(input_layout, real_per_batch, batch) * std::mem::size_of::<f32>() as u64
    } else {
        input_bytes
    };
    let output_view_bytes = if matches!(mode, ViewMode::Strided) {
        layout_span(output_layout, packed_per_batch, batch) * 2 * std::mem::size_of::<f32>() as u64
    } else {
        output_bytes
    };
    let offset = if matches!(mode, ViewMode::Offset) {
        aligned_test_offset(context)
    } else {
        0
    };
    let input_size = offset + input_view_bytes + offset;
    let output_size = offset + output_view_bytes + offset;
    let output_read_offset = if matches!(mode, ViewMode::Offset) {
        offset
    } else {
        0
    };

    let input_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.r2c_single_large.input"),
        size: input_size,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let physical_input = if matches!(mode, ViewMode::Strided) {
        scatter_strided_scalar(&input, input_layout, real_per_batch, batch)
    } else {
        input.clone()
    };
    context
        .queue
        .write_buffer(&input_buffer, offset, bytemuck::cast_slice(&physical_input));
    let output_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.r2c_single_large.output"),
        size: output_size,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let readback_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.r2c_single_large.readback"),
        size: output_view_bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let input_view = if matches!(mode, ViewMode::SegmentedBoth | ViewMode::SegmentedInput) {
        let segments = split_buffer_segments(&input_buffer, input_bytes, 4);
        BufferView::from_segments(&segments, 0, input_bytes).unwrap()
    } else {
        BufferView::new(&input_buffer, offset, input_view_bytes).unwrap()
    };
    let output_view = if matches!(mode, ViewMode::SegmentedBoth | ViewMode::SegmentedOutput) {
        let segments = split_buffer_segments(&output_buffer, output_bytes, 4);
        BufferView::from_segments(&segments, 0, output_bytes).unwrap()
    } else {
        BufferView::new(&output_buffer, offset, output_view_bytes).unwrap()
    };

    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("wgpu_fft.test.r2c_single_large.encoder"),
        });
    if matches!(mode, ViewMode::Strided) {
        let input_logical = FftLogicalView::new(input_view, input_layout).unwrap();
        let output_logical = FftLogicalView::new(output_view, output_layout).unwrap();
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
        output_read_offset,
        &readback_buffer,
        0,
        output_view_bytes,
    );
    context.queue.submit([encoder.finish()]);

    let label = format!(
        "r2c single large-decomposition {:?} {:?}",
        plan.config(),
        mode
    );
    let actual_physical = read_f32(context, &readback_buffer, &label);
    let actual = if matches!(mode, ViewMode::Strided) {
        gather_strided_complex(&actual_physical, output_layout, packed_per_batch, batch)
    } else {
        actual_physical
    };
    assert_close(&actual, &expected, &label);
    trace_gpu_step(&format!("finish {label}"));
}

fn run_c2r_large_single_decomposition_case(
    context: &wgpu_fft::device::GpuContext,
    config: FftConfig,
    expected_route: C2cRoute,
    mode: ViewMode,
) {
    trace_gpu_step(&format!(
        "start c2r single large-decomposition config={config:?} route={expected_route:?} mode={mode:?}"
    ));
    let real = real_input_for_config(
        &FftConfig::new_nd(config.shape().to_vec()).with_batch(config.batch()),
    );
    let forward = FftConfig::new_nd(config.shape().to_vec())
        .with_batch(config.batch())
        .with_normalization(Normalization::None);
    let packed = reference_r2c_packed_interleaved(&real, &forward).unwrap();
    let expected = reference_c2r_from_packed_interleaved(&packed, &config).unwrap();
    let limits = real_single_decomposition_test_limits(&config);
    let plan = FftPlan::c2r_with_large_policy_limits_for_testing(
        &context.device,
        &context.queue,
        config,
        limits,
    )
    .unwrap();
    assert_eq!(plan.kind(), FftTransformKind::C2r);
    assert_eq!(plan.route(), expected_route);
    assert_eq!(
        plan.large_routing_policy().route_mode(),
        LargeRouteMode::LargeChunk
    );
    let expected_execution_kind = if expected_route == C2cRoute::AxisSequence {
        wgpu_fft::LargeExecutionKind::AxisDecomposition
    } else {
        wgpu_fft::LargeExecutionKind::Smooth1dDecomposition
    };
    assert_eq!(
        plan.large_routing_policy().execution_kind(),
        expected_execution_kind
    );
    let route_diagnostics = plan.large_routing_policy().diagnostics();
    if expected_execution_kind == wgpu_fft::LargeExecutionKind::Smooth1dDecomposition {
        assert_eq!(route_diagnostics.selected_axis, Some(0));
        assert_eq!(route_diagnostics.factor_splits.len(), 1);
    } else {
        assert!(route_diagnostics.factor_splits.len() >= 2);
    }
    let device_diagnostics = plan.diagnostics_for_device(&context.device);
    assert!(device_diagnostics.blockers().is_empty());
    if expected_execution_kind == wgpu_fft::LargeExecutionKind::AxisDecomposition {
        assert!(device_diagnostics
            .stages()
            .iter()
            .any(|stage| stage.label == "large-axis-sequence-workspace"));
    }

    let input_bytes = plan.required_input_buffer_size_bytes();
    let output_bytes = plan.required_output_buffer_size_bytes();
    let batch = plan.config().batch() as u64;
    let packed_per_batch = input_bytes / batch / (2 * std::mem::size_of::<f32>() as u64);
    let real_per_batch = output_bytes / batch / std::mem::size_of::<f32>() as u64;
    let input_layout = strided_layout(packed_per_batch, 3);
    let output_layout = strided_layout(real_per_batch, 5);
    let input_view_bytes = if matches!(mode, ViewMode::Strided) {
        layout_span(input_layout, packed_per_batch, batch) * 2 * std::mem::size_of::<f32>() as u64
    } else {
        input_bytes
    };
    let output_view_bytes = if matches!(mode, ViewMode::Strided) {
        layout_span(output_layout, real_per_batch, batch) * std::mem::size_of::<f32>() as u64
    } else {
        output_bytes
    };
    let offset = if matches!(mode, ViewMode::Offset) {
        aligned_test_offset(context)
    } else {
        0
    };
    let input_size = offset + input_view_bytes + offset;
    let output_size = offset + output_view_bytes + offset;
    let output_read_offset = if matches!(mode, ViewMode::Offset) {
        offset
    } else {
        0
    };

    let input_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.c2r_single_large.input"),
        size: input_size,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let physical_input = if matches!(mode, ViewMode::Strided) {
        scatter_strided_complex(&packed, input_layout, packed_per_batch, batch)
    } else {
        packed.clone()
    };
    context
        .queue
        .write_buffer(&input_buffer, offset, bytemuck::cast_slice(&physical_input));
    let output_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.c2r_single_large.output"),
        size: output_size,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let readback_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.c2r_single_large.readback"),
        size: output_view_bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let input_view = if matches!(mode, ViewMode::SegmentedBoth | ViewMode::SegmentedInput) {
        let segments = split_buffer_segments(&input_buffer, input_bytes, 4);
        BufferView::from_segments(&segments, 0, input_bytes).unwrap()
    } else {
        BufferView::new(&input_buffer, offset, input_view_bytes).unwrap()
    };
    let output_view = if matches!(mode, ViewMode::SegmentedBoth | ViewMode::SegmentedOutput) {
        let segments = split_buffer_segments(&output_buffer, output_bytes, 4);
        BufferView::from_segments(&segments, 0, output_bytes).unwrap()
    } else {
        BufferView::new(&output_buffer, offset, output_view_bytes).unwrap()
    };

    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("wgpu_fft.test.c2r_single_large.encoder"),
        });
    if matches!(mode, ViewMode::Strided) {
        let input_logical = FftLogicalView::new(input_view, input_layout).unwrap();
        let output_logical = FftLogicalView::new(output_view, output_layout).unwrap();
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
        output_read_offset,
        &readback_buffer,
        0,
        output_view_bytes,
    );
    context.queue.submit([encoder.finish()]);

    let label = format!(
        "c2r single large-decomposition {:?} {:?}",
        plan.config(),
        mode
    );
    let actual_physical = read_f32(context, &readback_buffer, &label);
    let actual = if matches!(mode, ViewMode::Strided) {
        gather_strided_scalar(&actual_physical, output_layout, real_per_batch, batch)
    } else {
        actual_physical
    };
    assert_close_with_abs_tolerance(&actual, &expected, &label, 5.0e-2);
    trace_gpu_step(&format!("finish {label}"));
}

fn assert_real_validation_behavior(context: &wgpu_fft::device::GpuContext) {
    let r2c_creation_error =
        FftPlan::r2c_with_diagnostics(&context.device, &context.queue, FftConfig::inverse(16))
            .err()
            .expect("invalid R2C direction should fail plan construction");
    assert_eq!(
        r2c_creation_error.error(),
        &FftError::InvalidRealTransformDirection {
            transform: "r2c",
            expected: "forward",
            actual: "inverse",
        }
    );
    assert_eq!(r2c_creation_error.diagnostics().route().transform, "r2c");
    assert!(r2c_creation_error
        .diagnostics()
        .blockers()
        .iter()
        .any(|blocker| blocker.stage.as_deref() == Some("direction")));

    let c2r_creation_error =
        FftPlan::c2r_with_diagnostics(&context.device, &context.queue, FftConfig::new(16))
            .err()
            .expect("invalid C2R direction should fail plan construction");
    assert_eq!(
        c2r_creation_error.error(),
        &FftError::InvalidRealTransformDirection {
            transform: "c2r",
            expected: "inverse",
            actual: "forward",
        }
    );
    assert_eq!(c2r_creation_error.diagnostics().route().transform, "c2r");

    assert_eq!(
        FftPlan::r2c(&context.device, &context.queue, FftConfig::inverse(16))
            .err()
            .unwrap(),
        FftError::InvalidRealTransformDirection {
            transform: "r2c",
            expected: "forward",
            actual: "inverse",
        }
    );
    assert_eq!(
        FftPlan::c2r(&context.device, &context.queue, FftConfig::new(16))
            .err()
            .unwrap(),
        FftError::InvalidRealTransformDirection {
            transform: "c2r",
            expected: "inverse",
            actual: "forward",
        }
    );
    assert_eq!(
        FftPlan::r2c(
            &context.device,
            &context.queue,
            FftConfig::new_nd([4, 3]).with_axes([1]),
        )
        .err()
        .unwrap(),
        FftError::UnsupportedRealAxes {
            expected: vec![0, 1],
            actual: vec![1],
        }
    );

    let plan = FftPlan::r2c(
        &context.device,
        &context.queue,
        FftConfig::new(16).with_normalization(Normalization::None),
    )
    .unwrap();
    let input = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.real.too_small_input"),
        size: plan.required_input_buffer_size_bytes() - 4,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let output = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.real.valid_output"),
        size: plan.required_output_buffer_size_bytes(),
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let workspace = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.real.workspace"),
        size: 64,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("wgpu_fft.test.real.validation_encoder"),
        });
    assert_eq!(
        plan.execute_views(
            &context.device,
            &mut encoder,
            BufferView::whole(&input),
            BufferView::whole(&output),
        ),
        Err(FftError::BufferViewTooSmall {
            required: plan.required_input_buffer_size_bytes(),
            actual: plan.required_input_buffer_size_bytes() - 4,
        })
    );
    assert_eq!(
        plan.execute_views_with_workspace(
            &context.device,
            &mut encoder,
            BufferView::whole(&input),
            BufferView::whole(&output),
            BufferView::whole(&workspace),
        ),
        Err(FftError::BufferViewTooSmall {
            required: plan.required_input_buffer_size_bytes(),
            actual: plan.required_input_buffer_size_bytes() - 4,
        })
    );
    let r2c_size_error = plan
        .execute_views_with_diagnostics(
            &context.device,
            &mut encoder,
            BufferView::whole(&input),
            BufferView::whole(&output),
        )
        .unwrap_err();
    assert_eq!(
        r2c_size_error.error(),
        &FftError::BufferViewTooSmall {
            required: plan.required_input_buffer_size_bytes(),
            actual: plan.required_input_buffer_size_bytes() - 4,
        }
    );
    assert_eq!(r2c_size_error.diagnostics().route().transform, "r2c");
    assert!(r2c_size_error
        .diagnostics()
        .blockers()
        .iter()
        .any(|blocker| {
            blocker.kind == FftBlockerKind::Validation
                && blocker.route.as_deref() == Some("mixed-radix")
                && blocker.stage.as_deref() == Some("input-logical-view")
                && blocker.layout.as_deref() == Some("contiguous")
                && blocker.required_bytes == Some(plan.required_input_buffer_size_bytes())
                && blocker.actual_bytes == Some(plan.required_input_buffer_size_bytes() - 4)
        }));
    let strided_layout = FftLogicalLayout::new(1, 2).unwrap();
    let real_elements = plan.required_input_buffer_size_bytes() / std::mem::size_of::<f32>() as u64;
    let packed_elements =
        plan.required_output_buffer_size_bytes() / (2 * std::mem::size_of::<f32>() as u64);
    let strided_input_required =
        (strided_layout.element_offset + strided_layout.element_stride * (real_elements - 1) + 1)
            * std::mem::size_of::<f32>() as u64;
    let strided_output_required =
        (strided_layout.element_offset + strided_layout.element_stride * (packed_elements - 1) + 1)
            * 2
            * std::mem::size_of::<f32>() as u64;
    let too_small_strided_input = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.real.too_small_strided_input"),
        size: strided_input_required - 4,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let too_small_strided_output = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.real.too_small_strided_output"),
        size: strided_output_required - 8,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let strided_diagnostics = plan.diagnostics_for_logical_views(
        &context.device,
        &FftLogicalView::new(BufferView::whole(&too_small_strided_input), strided_layout).unwrap(),
        &FftLogicalView::new(BufferView::whole(&too_small_strided_output), strided_layout).unwrap(),
    );
    assert!(strided_diagnostics.blockers().iter().any(|blocker| {
        blocker.kind == FftBlockerKind::Layout
            && blocker.route.as_deref() == Some("mixed-radix")
            && blocker.stage.as_deref() == Some("input-logical-view")
            && blocker.layout.as_deref() == Some("strided")
            && blocker.required_bytes == Some(strided_input_required)
            && blocker.actual_bytes == Some(strided_input_required - 4)
    }));
    assert!(strided_diagnostics.blockers().iter().any(|blocker| {
        blocker.kind == FftBlockerKind::Layout
            && blocker.route.as_deref() == Some("mixed-radix")
            && blocker.stage.as_deref() == Some("output-logical-view")
            && blocker.layout.as_deref() == Some("strided")
            && blocker.required_bytes == Some(strided_output_required)
            && blocker.actual_bytes == Some(strided_output_required - 8)
    }));
    let no_storage_output_for_mixed_diagnostics =
        context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("wgpu_fft.test.real.no_storage_output_with_bad_input"),
            size: plan.required_output_buffer_size_bytes(),
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
    let storage_alignment =
        u64::from(context.device.limits().min_storage_buffer_offset_alignment).max(1);
    if storage_alignment > 1 {
        let misaligned_offset = storage_alignment / 2;
        let unaligned_input = context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("wgpu_fft.test.real.unaligned_logical_input"),
            size: plan.required_input_buffer_size_bytes() + misaligned_offset,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let unaligned_output = context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("wgpu_fft.test.real.unaligned_logical_output"),
            size: plan.required_output_buffer_size_bytes() + misaligned_offset,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let alignment_diagnostics = plan.diagnostics_for_logical_views(
            &context.device,
            &FftLogicalView::contiguous(
                BufferView::new(
                    &unaligned_input,
                    misaligned_offset,
                    plan.required_input_buffer_size_bytes(),
                )
                .unwrap(),
            ),
            &FftLogicalView::contiguous(
                BufferView::new(
                    &unaligned_output,
                    misaligned_offset,
                    plan.required_output_buffer_size_bytes(),
                )
                .unwrap(),
            ),
        );
        for stage in ["input-storage-window", "output-storage-window"] {
            assert!(
                alignment_diagnostics.blockers().iter().any(|blocker| {
                    blocker.kind == FftBlockerKind::Alignment
                        && blocker.route.as_deref() == Some("mixed-radix")
                        && blocker.stage.as_deref() == Some(stage)
                        && blocker.layout.as_deref() == Some("offset")
                        && blocker.required_bytes == Some(storage_alignment)
                        && blocker.actual_bytes == Some(misaligned_offset)
                }),
                "expected real {stage} alignment blocker"
            );
        }
    }
    let copy_input = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.real.unaligned_copy_input"),
        size: plan.required_input_buffer_size_bytes(),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let copy_output = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.real.unaligned_copy_output"),
        size: plan.required_output_buffer_size_bytes(),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let input_segments = [
        BufferSegment::new(&copy_input, 0, 6),
        BufferSegment::new(&copy_input, 6, plan.required_input_buffer_size_bytes() - 6),
    ];
    let output_segments = [
        BufferSegment::new(&copy_output, 0, 6),
        BufferSegment::new(
            &copy_output,
            6,
            plan.required_output_buffer_size_bytes() - 6,
        ),
    ];
    let copy_alignment_diagnostics = plan.diagnostics_for_logical_views(
        &context.device,
        &FftLogicalView::contiguous(
            BufferView::from_segments(&input_segments, 0, plan.required_input_buffer_size_bytes())
                .unwrap(),
        ),
        &FftLogicalView::contiguous(
            BufferView::from_segments(
                &output_segments,
                0,
                plan.required_output_buffer_size_bytes(),
            )
            .unwrap(),
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
            "expected real {stage} copy-alignment blocker"
        );
    }

    let valid_input = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.real.workspace_noop_input"),
        size: plan.required_input_buffer_size_bytes(),
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let valid_output = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.real.workspace_noop_output"),
        size: plan.required_output_buffer_size_bytes(),
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let unused_workspace = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.real.workspace_noop_unused"),
        size: 16,
        usage: wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let unused_segments = split_buffer_segments(&unused_workspace, 16, 2);
    let workspace_diagnostics = plan.diagnostics_for_views_with_workspace(
        &context.device,
        BufferView::whole(&valid_input),
        BufferView::whole(&valid_output),
        BufferView::from_segments(&unused_segments, 0, 16).unwrap(),
    );
    assert!(!workspace_diagnostics.blockers().iter().any(|blocker| {
        blocker.stage.as_deref() == Some("workspace")
            || blocker.helper_buffer.as_deref() == Some("workspace")
    }));
    let logical_workspace_diagnostics = plan.diagnostics_for_logical_views_with_workspace(
        &context.device,
        &FftLogicalView::contiguous(BufferView::whole(&valid_input)),
        &FftLogicalView::contiguous(BufferView::whole(&valid_output)),
        BufferView::from_segments(&unused_segments, 0, 16).unwrap(),
    );
    assert!(!logical_workspace_diagnostics
        .blockers()
        .iter()
        .any(|blocker| {
            blocker.stage.as_deref() == Some("workspace")
                || blocker.helper_buffer.as_deref() == Some("workspace")
        }));
    let mut workspace_encoder =
        context
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("wgpu_fft.test.real.workspace_noop_encoder"),
            });
    plan.execute_views_with_workspace(
        &context.device,
        &mut workspace_encoder,
        BufferView::whole(&valid_input),
        BufferView::whole(&valid_output),
        BufferView::from_segments(&unused_segments, 0, 16).unwrap(),
    )
    .unwrap();
    plan.execute_logical_views_with_workspace(
        &context.device,
        &mut workspace_encoder,
        FftLogicalView::contiguous(BufferView::whole(&valid_input)),
        FftLogicalView::contiguous(BufferView::whole(&valid_output)),
        BufferView::from_segments(&unused_segments, 0, 16).unwrap(),
    )
    .unwrap();

    let c2r_plan = FftPlan::c2r(&context.device, &context.queue, FftConfig::inverse(16)).unwrap();
    let c2r_input = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.real.c2r_workspace_noop_input"),
        size: c2r_plan.required_input_buffer_size_bytes(),
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let c2r_output = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.real.c2r_workspace_noop_output"),
        size: c2r_plan.required_output_buffer_size_bytes(),
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let mut c2r_workspace_encoder =
        context
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("wgpu_fft.test.real.c2r_workspace_noop_encoder"),
            });
    c2r_plan
        .execute_views_with_workspace(
            &context.device,
            &mut c2r_workspace_encoder,
            BufferView::whole(&c2r_input),
            BufferView::whole(&c2r_output),
            BufferView::from_segments(&unused_segments, 0, 16).unwrap(),
        )
        .unwrap();
    c2r_plan
        .execute_logical_views_with_workspace(
            &context.device,
            &mut c2r_workspace_encoder,
            FftLogicalView::contiguous(BufferView::whole(&c2r_input)),
            FftLogicalView::contiguous(BufferView::whole(&c2r_output)),
            BufferView::from_segments(&unused_segments, 0, 16).unwrap(),
        )
        .unwrap();
    let c2r_too_small_output = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.real.c2r_too_small_output"),
        size: c2r_plan.required_output_buffer_size_bytes() - 4,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let c2r_size_error = c2r_plan
        .execute_views_with_diagnostics(
            &context.device,
            &mut c2r_workspace_encoder,
            BufferView::whole(&c2r_input),
            BufferView::whole(&c2r_too_small_output),
        )
        .unwrap_err();
    assert_eq!(
        c2r_size_error.error(),
        &FftError::BufferViewTooSmall {
            required: c2r_plan.required_output_buffer_size_bytes(),
            actual: c2r_plan.required_output_buffer_size_bytes() - 4,
        }
    );
    assert_eq!(c2r_size_error.diagnostics().route().transform, "c2r");
    assert!(c2r_size_error
        .diagnostics()
        .blockers()
        .iter()
        .any(|blocker| {
            blocker.kind == FftBlockerKind::Validation
                && blocker.route.as_deref() == Some("mixed-radix")
                && blocker.stage.as_deref() == Some("output-logical-view")
                && blocker.layout.as_deref() == Some("contiguous")
                && blocker.required_bytes == Some(c2r_plan.required_output_buffer_size_bytes())
                && blocker.actual_bytes == Some(c2r_plan.required_output_buffer_size_bytes() - 4)
        }));
}

fn assert_real_pipeline_cache_snapshot_behavior(context: &wgpu_fft::device::GpuContext) {
    let snapshot = export_pipeline_cache_snapshot(&context.device);
    assert!(snapshot
        .pipeline_keys()
        .iter()
        .any(|key| key.contains("real-to-complex")));
    assert!(snapshot
        .pipeline_keys()
        .iter()
        .any(|key| key.contains("pack-r2c")));
    assert!(snapshot
        .pipeline_keys()
        .iter()
        .any(|key| key.contains("unpack-c2r")));
    assert!(snapshot
        .pipeline_keys()
        .iter()
        .any(|key| key.contains("complex-to-real")));
    assert!(snapshot
        .pipeline_keys()
        .iter()
        .any(|key| key.contains("real-to-complex-windowed")));
    assert!(snapshot
        .pipeline_keys()
        .iter()
        .any(|key| key.contains("pack-r2c-windowed")));
    assert!(snapshot
        .pipeline_keys()
        .iter()
        .any(|key| key.contains("unpack-c2r-windowed")));
    assert!(snapshot
        .pipeline_keys()
        .iter()
        .any(|key| key.contains("complex-to-real-windowed")));
}

fn inverse_config_for(config: &FftConfig) -> FftConfig {
    FftConfig::inverse_nd(config.shape().to_vec()).with_batch(config.batch())
}

fn real_input_for_config(config: &FftConfig) -> Vec<f32> {
    (0..config.total_complex_len().unwrap())
        .map(|i| {
            let x = i as f32;
            (x * 0.31).sin() * 0.6 + (x * 0.17).cos() * 0.4
        })
        .collect()
}

fn real_large_chunk_test_limits(config: &FftConfig, chunk_batches: u64) -> LargePolicyLimits {
    let logical_len = config.logical_complex_len().unwrap() as u64;
    let packed_len = (config.shape()[0] / 2 + 1) as u64
        * config.shape()[1..]
            .iter()
            .copied()
            .map(|value| value as u64)
            .product::<u64>();
    let real_bytes_per_batch = logical_len * std::mem::size_of::<f32>() as u64;
    let full_complex_bytes_per_batch = logical_len * 2 * std::mem::size_of::<f32>() as u64;
    let packed_bytes_per_batch = packed_len * 2 * std::mem::size_of::<f32>() as u64;
    let bytes_per_batch = real_bytes_per_batch
        .max(full_complex_bytes_per_batch)
        .max(packed_bytes_per_batch);
    let max_storage_buffer_binding_size = bytes_per_batch * chunk_batches;
    assert!(
        bytes_per_batch * config.batch() as u64 > max_storage_buffer_binding_size,
        "forced large-chunk limits must make the full batched plan exceed one binding"
    );
    LargePolicyLimits {
        max_storage_buffer_binding_size,
        max_buffer_size: max_storage_buffer_binding_size,
    }
}

fn real_single_decomposition_test_limits(config: &FftConfig) -> LargePolicyLimits {
    let total_len = config.total_complex_len().unwrap() as u64;
    let full_complex_bytes = total_len * 2 * std::mem::size_of::<f32>() as u64;
    LargePolicyLimits {
        max_storage_buffer_binding_size: 256,
        max_buffer_size: full_complex_bytes,
    }
}

fn aligned_test_offset(context: &wgpu_fft::device::GpuContext) -> u64 {
    u64::from(context.device.limits().min_storage_buffer_offset_alignment).max(256)
}

fn strided_layout(logical_per_batch: u64, element_offset: u64) -> FftLogicalLayout {
    let per_batch_span = if logical_per_batch == 0 {
        0
    } else {
        2 * (logical_per_batch - 1) + 1
    };
    FftLogicalLayout::new(element_offset, 2)
        .unwrap()
        .with_batch_stride(per_batch_span + 7)
}

fn layout_span(layout: FftLogicalLayout, logical_per_batch: u64, batch: u64) -> u64 {
    if logical_per_batch == 0 || batch == 0 {
        return 0;
    }
    let per_batch_span = layout.element_stride * (logical_per_batch - 1) + 1;
    let batch_stride = layout.batch_stride.unwrap_or(per_batch_span);
    layout.element_offset + (batch - 1) * batch_stride + per_batch_span
}

fn buffer_layout_from_logical(layout: FftLogicalLayout) -> BufferLayout {
    let mut buffer_layout = BufferLayout::new(layout.element_offset, layout.element_stride)
        .expect("test logical layout should convert to BufferLayout");
    if let Some(batch_stride) = layout.batch_stride {
        buffer_layout = buffer_layout.with_batch_stride(batch_stride);
    }
    buffer_layout
}

fn physical_index(layout: FftLogicalLayout, logical_per_batch: u64, logical_index: u64) -> usize {
    let per_batch_span = layout.element_stride * (logical_per_batch - 1) + 1;
    let batch_stride = layout.batch_stride.unwrap_or(per_batch_span);
    let batch = logical_index / logical_per_batch;
    let element = logical_index - batch * logical_per_batch;
    (layout.element_offset + batch * batch_stride + element * layout.element_stride) as usize
}

fn scatter_strided_scalar(
    logical: &[f32],
    layout: FftLogicalLayout,
    logical_per_batch: u64,
    batch: u64,
) -> Vec<f32> {
    let span = layout_span(layout, logical_per_batch, batch) as usize;
    let mut physical = vec![0.0; span];
    for (logical_index, value) in logical.iter().enumerate() {
        let physical_index = physical_index(layout, logical_per_batch, logical_index as u64);
        physical[physical_index] = *value;
    }
    physical
}

fn gather_strided_scalar(
    physical: &[f32],
    layout: FftLogicalLayout,
    logical_per_batch: u64,
    batch: u64,
) -> Vec<f32> {
    let total = (logical_per_batch * batch) as usize;
    let mut logical = vec![0.0; total];
    for logical_index in 0..total {
        let physical_index = physical_index(layout, logical_per_batch, logical_index as u64);
        logical[logical_index] = physical[physical_index];
    }
    logical
}

fn scatter_strided_complex(
    logical: &[f32],
    layout: FftLogicalLayout,
    logical_per_batch: u64,
    batch: u64,
) -> Vec<f32> {
    let span = layout_span(layout, logical_per_batch, batch) as usize;
    let mut physical = vec![0.0; span * 2];
    for logical_index in 0..(logical.len() / 2) {
        let physical_index = physical_index(layout, logical_per_batch, logical_index as u64);
        physical[physical_index * 2] = logical[logical_index * 2];
        physical[physical_index * 2 + 1] = logical[logical_index * 2 + 1];
    }
    physical
}

fn gather_strided_complex(
    physical: &[f32],
    layout: FftLogicalLayout,
    logical_per_batch: u64,
    batch: u64,
) -> Vec<f32> {
    let total = (logical_per_batch * batch) as usize;
    let mut logical = vec![0.0; total * 2];
    for logical_index in 0..total {
        let physical_index = physical_index(layout, logical_per_batch, logical_index as u64);
        logical[logical_index * 2] = physical[physical_index * 2];
        logical[logical_index * 2 + 1] = physical[physical_index * 2 + 1];
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
    assert!(
        forced_diagnostics.blockers().iter().any(|blocker| {
            blocker.route.is_some()
                && blocker.stage.is_some()
                && blocker.required_bytes.is_some()
                && blocker.limit_bytes.is_some()
                && matches!(
                    blocker.kind,
                    FftBlockerKind::DeviceLimit | FftBlockerKind::HelperBuffer
                )
        }),
        "expected forced-limit graph blocker for {:?}",
        plan.config()
    );
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

fn read_f32(
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

    let mapped = slice.get_mapped_range();
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
    const ABS_TOLERANCE: f32 = 1.0e-2;
    const REL_TOLERANCE: f32 = 1.0e-6;
    assert_close_with_tolerances(actual, expected, label, ABS_TOLERANCE, REL_TOLERANCE);
}

fn assert_close_with_abs_tolerance(
    actual: &[f32],
    expected: &[f32],
    label: &str,
    abs_tolerance: f32,
) {
    assert_close_with_tolerances(actual, expected, label, abs_tolerance, 1.0e-6);
}

fn assert_close_with_tolerances(
    actual: &[f32],
    expected: &[f32],
    label: &str,
    abs_tolerance: f32,
    rel_tolerance: f32,
) {
    assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");
    for (index, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
        let tolerance = abs_tolerance + rel_tolerance * expected.abs();
        assert!(
            (actual - expected).abs() < tolerance,
            "{label}: index {index}: actual={actual}, expected={expected}, tolerance={tolerance}"
        );
    }
}
