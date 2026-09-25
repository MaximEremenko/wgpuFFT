#![cfg(not(target_arch = "wasm32"))]

//! GPU correctness and routing coverage for the GPU-resident four-step C2C route.

use std::sync::mpsc;

use wgpu_fft::math::{reference_c2c_nd_f64, Complex64};
use wgpu_fft::{
    AxisKind, BufferLayout, BufferSegment, BufferView, FftConfig, FftDeviceLimits, FftError,
    FftIoView, FftPlan, LargeExecutionKind, LargePolicyLimits, LargeRouteMode, Normalization,
};

#[test]
fn four_step_gpu_matches_normal_route_and_f64_reference() {
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
    #[cfg(windows)]
    let context = std::mem::ManuallyDrop::new(context);
    eprintln!("adapter: {:?}", context.adapter.get_info());
    eprintln!(
        "limits: maxStorageBufferBindingSize={} maxBufferSize={} storageAlignment={}",
        context.device.limits().max_storage_buffer_binding_size,
        context.device.limits().max_buffer_size,
        context.device.limits().min_storage_buffer_offset_alignment,
    );

    for config in [
        FftConfig::new_nd([15, 14])
            .with_batch(3)
            .with_normalization(Normalization::None),
        FftConfig::inverse_nd([15, 14]).with_batch(3),
    ] {
        run_forced_equivalence(&context, config);
    }
    for config in [
        FftConfig::new_nd([5, 7, 9])
            .with_batch(3)
            .with_normalization(Normalization::None),
        FftConfig::inverse_nd([5, 7, 9]).with_batch(3),
    ] {
        run_forced_rank_nd_equivalence(&context, config, true);
    }
    for config in [
        FftConfig::new_nd([3, 5, 7, 11])
            .with_axes([3, 1])
            .with_batch(2)
            .with_normalization(Normalization::None),
        FftConfig::inverse_nd([3, 5, 7, 11])
            .with_axes([3, 1])
            .with_batch(2),
    ] {
        run_forced_rank_nd_equivalence(&context, config, false);
    }
    for config in [
        FftConfig::new_nd([5, 17, 7])
            .with_batch(3)
            .with_normalization(Normalization::None),
        FftConfig::inverse_nd([5, 17, 7]).with_batch(3),
    ] {
        run_forced_non_mixed_equivalence(&context, config, 256, 1, AxisKind::Rader, "rader");
    }
    for config in [
        FftConfig::new_nd([5, 7, 34])
            .with_batch(2)
            .with_normalization(Normalization::None),
        FftConfig::inverse_nd([5, 7, 34]).with_batch(2),
    ] {
        run_forced_non_mixed_equivalence(
            &context,
            config,
            1024,
            2,
            AxisKind::Bluestein,
            "bluestein",
        );
    }
    for config in [
        FftConfig::new_nd([2, 101, 3]).with_normalization(Normalization::None),
        FftConfig::inverse_nd([2, 101, 3]),
    ] {
        run_forced_non_mixed_equivalence(
            &context,
            config,
            512,
            1,
            AxisKind::Rader,
            "bluestein-fallback",
        );
    }
    assert_strided_io_is_explicitly_deferred(&context);
    assert_full_volume_above_max_buffer_selects_segmented(&context);
    run_real_oversized_sampled_case(&context);
    run_real_oversized_rank3_sampled_case(&context);
    run_real_oversized_prime_sampled_case(&context);
}

fn run_forced_rank_nd_equivalence(
    context: &wgpu_fft::device::GpuContext,
    config: FftConfig,
    exercise_views: bool,
) {
    let input = test_signal(config.total_complex_len().unwrap());
    let expected = reference_f64(&input, &config);
    let baseline = FftPlan::c2c(&context.device, &context.queue, config.clone()).unwrap();
    let forced = FftPlan::c2c_with_large_policy_limits_for_testing(
        &context.device,
        &context.queue,
        config.clone(),
        LargePolicyLimits {
            max_storage_buffer_binding_size: 256,
            max_buffer_size: config.required_buffer_size_bytes().unwrap(),
        },
    )
    .unwrap();
    assert_eq!(
        forced.large_routing_policy().route_mode(),
        LargeRouteMode::LargeOutOfCore
    );
    assert_eq!(
        forced.large_routing_policy().execution_kind(),
        LargeExecutionKind::OutOfCoreFourStep
    );
    let expected_permutations = config.axes().iter().filter(|&&axis| axis != 0).count() * 2;
    assert_eq!(
        forced
            .diagnostics()
            .stages()
            .iter()
            .filter(|stage| stage.kind == "permutation")
            .count(),
        expected_permutations,
        "rank-N diagnostics must expose both permutations for every nonzero axis"
    );
    assert!(forced
        .diagnostics()
        .stages()
        .iter()
        .any(|stage| stage.kind == "windowed-kernel"));
    if config.axes().contains(&3) {
        assert!(forced
            .diagnostics()
            .stages()
            .iter()
            .any(|stage| stage.label.starts_with("four-step-axis3-windowed-")));
    }

    let baseline_output = execute_plan(context, &baseline, &input, "rank-N-baseline");
    let forced_output = execute_plan(context, &forced, &input, "rank-N-four-step");
    assert_close_f32(
        &forced_output,
        &baseline_output,
        "rank-N forced versus baseline",
    );
    assert_matches_reference(&forced_output, &expected, "rank-N forced versus f64");
    if exercise_views {
        let offset_output = execute_plan_with_offset(context, &forced, &input);
        assert_matches_reference(&offset_output, &expected, "rank-N offset versus f64");
        let segmented_output = execute_plan_with_segments(context, &forced, &input);
        assert_matches_reference(&segmented_output, &expected, "rank-N segmented versus f64");
    }
}

fn run_forced_non_mixed_equivalence(
    context: &wgpu_fft::device::GpuContext,
    config: FftConfig,
    max_bind_bytes: u64,
    non_mixed_axis: usize,
    expected_axis_kind: AxisKind,
    effective_kind: &str,
) {
    let input = test_signal(config.total_complex_len().unwrap());
    let expected = reference_f64(&input, &config);
    let baseline = FftPlan::c2c(&context.device, &context.queue, config.clone()).unwrap();
    let forced_max_buffer = config.required_buffer_size_bytes().unwrap().max(64 * 1024);
    let forced = FftPlan::c2c_with_large_policy_limits_for_testing(
        &context.device,
        &context.queue,
        config.clone(),
        LargePolicyLimits {
            max_storage_buffer_binding_size: max_bind_bytes,
            max_buffer_size: forced_max_buffer,
        },
    )
    .unwrap();

    let kind_index = config
        .axes()
        .iter()
        .position(|&axis| axis == non_mixed_axis)
        .expect("the non-mixed axis is transformed");
    assert_eq!(forced.axis_kinds()[kind_index], expected_axis_kind);
    assert_eq!(
        forced.large_routing_policy().route_mode(),
        LargeRouteMode::LargeOutOfCore
    );
    assert_eq!(
        forced.large_routing_policy().execution_kind(),
        LargeExecutionKind::OutOfCoreFourStep
    );
    assert_eq!(
        forced.large_routing_policy().max_bind_bytes,
        max_bind_bytes,
        "the encode-time route must preserve the forced binding limit"
    );
    assert_eq!(
        forced.large_routing_policy().axis_supported,
        Some(vec![true; config.axes().len()])
    );
    assert!(forced
        .large_routing_policy()
        .attempted_routes()
        .contains(&"out-of-core-four-step"));
    assert!(forced
        .large_routing_policy()
        .reason_codes()
        .contains(&"out-of-core-eligible"));

    let diagnostics = forced.diagnostics();
    let expected_stage_label = format!("four-step-axis{non_mixed_axis}-windowed-{effective_kind}");
    assert!(diagnostics
        .stages()
        .iter()
        .any(|stage| { stage.label == expected_stage_label && stage.kind == "windowed-kernel" }));
    if effective_kind == "bluestein-fallback" {
        let child_helpers = diagnostics
            .buffer_requirements()
            .iter()
            .filter(|requirement| {
                requirement.role.starts_with(&format!(
                    "helper:four-step-axis{non_mixed_axis}-child-workspace"
                ))
            })
            .count();
        assert!(
            child_helpers >= 4,
            "Bluestein fallback diagnostics must preserve distinct chirp, bfft, work, and fft helpers"
        );
    }
    if non_mixed_axis > 0 {
        for direction in ["to-front", "from-front"] {
            let label = format!("four-step-axis{non_mixed_axis}-permute-{direction}");
            assert!(diagnostics
                .stages()
                .iter()
                .any(|stage| stage.label == label && stage.kind == "permutation"));
        }
    }
    let route_diagnostics = forced.large_routing_policy().diagnostics();
    let axis_len = config.shape()[non_mixed_axis] as u64;
    let axis_split = route_diagnostics
        .factor_splits
        .iter()
        .find(|split| split.axis == Some(non_mixed_axis) && split.len == axis_len)
        .expect("non-mixed axis diagnostics must identify its convolution length");
    assert_eq!(axis_split.factors.len(), 1);
    let convolution_len = axis_split.factors[0];
    let minimum_convolution_len = if effective_kind == "rader" {
        2 * axis_len - 3
    } else {
        2 * axis_len - 1
    };
    assert!(convolution_len >= minimum_convolution_len);
    let convolution_split = route_diagnostics
        .factor_splits
        .iter()
        .find(|split| split.axis.is_none() && split.len == convolution_len)
        .expect("non-mixed axis diagnostics must expose the convolution radix split");
    assert_eq!(
        convolution_split.factors.iter().product::<u64>(),
        convolution_len
    );

    let baseline_output = execute_plan(context, &baseline, &input, "non-mixed-baseline");
    let forced_output = execute_plan(context, &forced, &input, effective_kind);
    assert_close_f32(
        &forced_output,
        &baseline_output,
        "non-mixed forced versus baseline",
    );
    assert_matches_reference(&forced_output, &expected, "non-mixed forced versus f64");
}

fn run_forced_equivalence(context: &wgpu_fft::device::GpuContext, config: FftConfig) {
    let input = test_signal(config.total_complex_len().unwrap());
    let expected = reference_f64(&input, &config);
    let baseline = FftPlan::c2c(&context.device, &context.queue, config.clone()).unwrap();
    let forced = FftPlan::c2c_with_large_policy_limits_for_testing(
        &context.device,
        &context.queue,
        config.clone(),
        LargePolicyLimits {
            max_storage_buffer_binding_size: 256,
            max_buffer_size: config.required_buffer_size_bytes().unwrap(),
        },
    )
    .unwrap();
    assert_eq!(
        forced.large_routing_policy().route_mode(),
        LargeRouteMode::LargeOutOfCore
    );
    assert_eq!(
        forced.large_routing_policy().execution_kind(),
        LargeExecutionKind::OutOfCoreFourStep
    );
    assert!(forced
        .large_routing_policy()
        .attempted_routes()
        .contains(&"out-of-core-four-step"));
    assert_eq!(forced.workspace_size_bytes(), 0);
    let diagnostics = forced.diagnostics();
    for (label, kind) in [
        ("four-step-axis0-windowed-fused-smooth", "windowed-kernel"),
        ("four-step-stripe-transpose-forward", "stripe-transpose"),
        ("four-step-axis1-windowed-fused-smooth", "windowed-kernel"),
        ("four-step-stripe-transpose-back", "stripe-transpose"),
    ] {
        assert!(diagnostics
            .stages()
            .iter()
            .any(|stage| stage.label == label && stage.kind == kind));
    }
    assert_eq!(
        diagnostics
            .stages()
            .iter()
            .any(|stage| stage.label == "four-step-scale" && stage.kind == "scale"),
        (config.scale().unwrap() - 1.0).abs() > f32::EPSILON,
    );
    assert!(diagnostics.buffer_requirements().iter().any(|requirement| {
        requirement.role == "helper:four-step-transpose-scratch"
            && requirement.required_bytes == config.required_buffer_size_bytes().unwrap()
    }));

    let baseline_output = execute_plan(context, &baseline, &input, "baseline");
    let forced_output = execute_plan(context, &forced, &input, "forced-four-step");
    assert_close_f32(&forced_output, &baseline_output, "forced versus baseline");
    assert_matches_reference(&forced_output, &expected, "forced versus f64");
    let offset_output = execute_plan_with_offset(context, &forced, &input);
    assert_matches_reference(&offset_output, &expected, "offset four-step versus f64");
    let segmented_output = execute_plan_with_segments(context, &forced, &input);
    assert_matches_reference(
        &segmented_output,
        &expected,
        "segmented four-step versus f64",
    );
    let prefixed_output = execute_plan_with_unused_segments(context, &forced, &input);
    assert_matches_reference(
        &prefixed_output,
        &expected,
        "unused-segment four-step versus f64",
    );
    assert_workspace_is_deferred(context, &forced, &input);
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
        label: Some("wgpu_fft.test.four_step.input"),
        size: byte_len,
        usage: usages,
        mapped_at_creation: false,
    });
    let output_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.four_step.output"),
        size: byte_len,
        usage: usages,
        mapped_at_creation: false,
    });
    let readback = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.four_step.readback"),
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
        "{label}: unexpected diagnostics blockers: {:?}",
        view_diagnostics.blockers()
    );
    if plan.large_routing_policy().execution_kind() == LargeExecutionKind::OutOfCoreFourStep {
        let policy = plan.large_routing_policy();
        let forced_diagnostics = plan.diagnostics_for_views_with_limits(
            FftDeviceLimits {
                max_storage_buffer_binding_size: policy.max_bind_bytes,
                max_buffer_size: policy.max_buffer_size,
                min_storage_buffer_offset_alignment: u64::from(
                    context.device.limits().min_storage_buffer_offset_alignment,
                ),
            },
            wgpu_fft::BufferView::whole(&input_buffer),
            wgpu_fft::BufferView::whole(&output_buffer),
        );
        assert!(
            forced_diagnostics.blockers().is_empty(),
            "{label}: unexpected forced-limit diagnostics blockers: {:?}",
            forced_diagnostics.blockers()
        );
    }

    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("wgpu_fft.test.four_step.encoder"),
        });
    plan.execute_checked(&context.device, &mut encoder, &input_buffer, &output_buffer)
        .unwrap();
    encoder.copy_buffer_to_buffer(&output_buffer, 0, &readback, 0, byte_len);
    context.queue.submit([encoder.finish()]);
    let values = read_f32(&context.device, &readback);
    eprintln!("four-step case completed: {label}");
    values
}

fn execute_plan_with_offset(
    context: &wgpu_fft::device::GpuContext,
    plan: &FftPlan,
    input: &[f32],
) -> Vec<f32> {
    let byte_len = std::mem::size_of_val(input) as u64;
    let offset = u64::from(
        context
            .device
            .limits()
            .min_storage_buffer_offset_alignment
            .max(256),
    );
    let backing_size = offset + byte_len + offset;
    let usages =
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;
    let input_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.four_step.offset_input"),
        size: backing_size,
        usage: usages,
        mapped_at_creation: false,
    });
    let output_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.four_step.offset_output"),
        size: backing_size,
        usage: usages,
        mapped_at_creation: false,
    });
    let readback = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.four_step.offset_readback"),
        size: byte_len,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    context
        .queue
        .write_buffer(&input_buffer, offset, bytemuck::cast_slice(input));
    let input_view = BufferView::new(&input_buffer, offset, byte_len).unwrap();
    let output_view = BufferView::new(&output_buffer, offset, byte_len).unwrap();
    assert!(plan
        .diagnostics_for_views(&context.device, input_view.clone(), output_view.clone())
        .blockers()
        .is_empty());

    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("wgpu_fft.test.four_step.offset_encoder"),
        });
    plan.execute_views(&context.device, &mut encoder, input_view, output_view)
        .unwrap();
    encoder.copy_buffer_to_buffer(&output_buffer, offset, &readback, 0, byte_len);
    context.queue.submit([encoder.finish()]);
    read_f32(&context.device, &readback)
}

fn execute_plan_with_segments(
    context: &wgpu_fft::device::GpuContext,
    plan: &FftPlan,
    input: &[f32],
) -> Vec<f32> {
    let byte_len = std::mem::size_of_val(input) as u64;
    let split = ((byte_len / 2) / 8) * 8;
    let tail = byte_len - split;
    let usages =
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;
    let create = |label, size| {
        context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage: usages,
            mapped_at_creation: false,
        })
    };
    let input_a = create("wgpu_fft.test.four_step.segmented_input_a", split);
    let input_b = create("wgpu_fft.test.four_step.segmented_input_b", tail);
    let output_a = create("wgpu_fft.test.four_step.segmented_output_a", split);
    let output_b = create("wgpu_fft.test.four_step.segmented_output_b", tail);
    let input_bytes = bytemuck::cast_slice(input);
    context
        .queue
        .write_buffer(&input_a, 0, &input_bytes[..usize::try_from(split).unwrap()]);
    context
        .queue
        .write_buffer(&input_b, 0, &input_bytes[usize::try_from(split).unwrap()..]);
    let input_segments = [
        BufferSegment::new(&input_a, 0, split),
        BufferSegment::new(&input_b, 0, tail),
    ];
    let output_segments = [
        BufferSegment::new(&output_a, 0, split),
        BufferSegment::new(&output_b, 0, tail),
    ];
    let input_view = BufferView::from_segments(&input_segments, 0, byte_len).unwrap();
    let output_view = BufferView::from_segments(&output_segments, 0, byte_len).unwrap();
    assert!(plan
        .diagnostics_for_views(&context.device, input_view.clone(), output_view.clone())
        .blockers()
        .is_empty());
    let readback = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.four_step.segmented_readback"),
        size: byte_len,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("wgpu_fft.test.four_step.segmented_encoder"),
        });
    plan.execute_views(&context.device, &mut encoder, input_view, output_view)
        .unwrap();
    encoder.copy_buffer_to_buffer(&output_a, 0, &readback, 0, split);
    encoder.copy_buffer_to_buffer(&output_b, 0, &readback, split, tail);
    context.queue.submit([encoder.finish()]);
    read_f32(&context.device, &readback)
}

fn execute_plan_with_unused_segments(
    context: &wgpu_fft::device::GpuContext,
    plan: &FftPlan,
    input: &[f32],
) -> Vec<f32> {
    let byte_len = std::mem::size_of_val(input) as u64;
    let unused_bytes = 8;
    let endpoint_usages =
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;
    let create = |label, size, usage| {
        context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage,
            mapped_at_creation: false,
        })
    };
    let unused_input = create(
        "wgpu_fft.test.four_step.unused_input_segment",
        unused_bytes,
        wgpu::BufferUsages::STORAGE,
    );
    let input_buffer = create(
        "wgpu_fft.test.four_step.prefixed_input",
        byte_len,
        endpoint_usages,
    );
    let output_buffer = create(
        "wgpu_fft.test.four_step.prefixed_output",
        byte_len,
        endpoint_usages,
    );
    let unused_output = create(
        "wgpu_fft.test.four_step.unused_output_segment",
        unused_bytes,
        wgpu::BufferUsages::STORAGE,
    );
    context
        .queue
        .write_buffer(&input_buffer, 0, bytemuck::cast_slice(input));

    let input_segments = [
        BufferSegment::new(&unused_input, 0, unused_bytes),
        BufferSegment::new(&input_buffer, 0, byte_len),
    ];
    let output_segments = [
        BufferSegment::new(&output_buffer, 0, byte_len),
        BufferSegment::new(&unused_output, 0, unused_bytes),
    ];
    let input_view = BufferView::from_segments(&input_segments, unused_bytes, byte_len).unwrap();
    let output_view = BufferView::from_segments(&output_segments, 0, byte_len).unwrap();
    assert!(plan
        .diagnostics_for_views(&context.device, input_view.clone(), output_view.clone())
        .blockers()
        .is_empty());

    let readback = create(
        "wgpu_fft.test.four_step.prefixed_readback",
        byte_len,
        wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
    );
    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("wgpu_fft.test.four_step.prefixed_encoder"),
        });
    plan.execute_views(&context.device, &mut encoder, input_view, output_view)
        .unwrap();
    encoder.copy_buffer_to_buffer(&output_buffer, 0, &readback, 0, byte_len);
    context.queue.submit([encoder.finish()]);
    read_f32(&context.device, &readback)
}

fn assert_workspace_is_deferred(
    context: &wgpu_fft::device::GpuContext,
    plan: &FftPlan,
    input: &[f32],
) {
    let byte_len = std::mem::size_of_val(input) as u64;
    let usages =
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;
    let create = |label| {
        context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: byte_len,
            usage: usages,
            mapped_at_creation: false,
        })
    };
    let input_buffer = create("wgpu_fft.test.four_step.workspace_input");
    let output_buffer = create("wgpu_fft.test.four_step.workspace_output");
    let workspace = create("wgpu_fft.test.four_step.workspace");
    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("wgpu_fft.test.four_step.workspace_encoder"),
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
            route_mode: "large-out-of-core",
        })
    );
}

fn assert_strided_io_is_explicitly_deferred(context: &wgpu_fft::device::GpuContext) {
    let config = FftConfig::new_nd([15, 14]).with_normalization(Normalization::None);
    let plan = FftPlan::c2c_with_large_policy_limits_for_testing(
        &context.device,
        &context.queue,
        config.clone(),
        LargePolicyLimits {
            max_storage_buffer_binding_size: 256,
            max_buffer_size: config.required_buffer_size_bytes().unwrap() * 2,
        },
    )
    .unwrap();
    let logical = config.logical_complex_len().unwrap() as u64;
    let physical_bytes = (logical * 2 - 1) * 8;
    let usages =
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;
    let input = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.four_step.strided_input"),
        size: physical_bytes,
        usage: usages,
        mapped_at_creation: false,
    });
    let output = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.four_step.strided_output"),
        size: config.required_buffer_size_bytes().unwrap(),
        usage: usages,
        mapped_at_creation: false,
    });
    let input_view =
        FftIoView::new(BufferView::whole(&input), BufferLayout::new(0, 2).unwrap()).unwrap();
    let output_view = FftIoView::contiguous(BufferView::whole(&output));
    let diagnostics =
        plan.diagnostics_for_io_views(&context.device, input_view.clone(), output_view.clone());
    assert!(diagnostics.blockers().iter().any(|blocker| {
        blocker.kind == wgpu_fft::FftBlockerKind::Unsupported
            && blocker.route.as_deref() == Some("large-out-of-core")
    }));
    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("wgpu_fft.test.four_step.strided_encoder"),
        });
    assert!(matches!(
        plan.execute_io_views(&context.device, &mut encoder, input_view, output_view),
        Err(FftError::LargeGraphStageUnsupported {
            stage: "four-step-logical-io",
            ..
        })
    ));
}

fn assert_full_volume_above_max_buffer_selects_segmented(context: &wgpu_fft::device::GpuContext) {
    let config = FftConfig::new_nd([15, 14]).with_normalization(Normalization::None);
    let required = config.required_buffer_size_bytes().unwrap();
    let plan = FftPlan::c2c_with_large_policy_limits_for_testing(
        &context.device,
        &context.queue,
        config,
        LargePolicyLimits {
            max_storage_buffer_binding_size: 256,
            max_buffer_size: required - 8,
        },
    )
    .unwrap();
    assert_eq!(
        plan.large_routing_policy().execution_kind(),
        LargeExecutionKind::SegmentedFullVolume
    );
    let diagnostics = plan.diagnostics();
    assert_eq!(
        diagnostics.route().large_route_mode.as_deref(),
        Some("large-out-of-core")
    );
    assert_eq!(
        diagnostics.route().execution_kind.as_deref(),
        Some("segmented-full-volume")
    );
    let arena_bytes = diagnostics
        .buffer_requirements()
        .iter()
        .filter(|requirement| requirement.role == "helper:segmented-volume-arena")
        .map(|requirement| requirement.required_bytes)
        .collect::<Vec<_>>();
    assert_eq!(arena_bytes, [required - 8, 8]);
}

fn run_real_oversized_sampled_case(context: &wgpu_fft::device::GpuContext) {
    let Some((n0, n1, byte_len)) = real_oversized_shape(&context.device.limits()) else {
        eprintln!(
            "skipping real oversized four-step case: adapter limits do not admit a safe rank-2 shape"
        );
        return;
    };
    eprintln!(
        "real oversized four-step start: shape=[{n0}, {n1}] bytes={byte_len} maxBind={}",
        context.device.limits().max_storage_buffer_binding_size
    );
    let config = FftConfig::new_nd([n0, n1]).with_normalization(Normalization::None);
    let plan = FftPlan::c2c(&context.device, &context.queue, config).unwrap();
    assert_eq!(
        plan.large_routing_policy().route_mode(),
        LargeRouteMode::LargeOutOfCore
    );
    assert_eq!(
        plan.large_routing_policy().execution_kind(),
        LargeExecutionKind::OutOfCoreFourStep
    );
    assert!(plan
        .diagnostics()
        .buffer_requirements()
        .iter()
        .any(|requirement| {
            requirement.role == "helper:four-step-transpose-scratch"
                && requirement.required_bytes == byte_len
        }));

    let usages =
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;
    let input = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.four_step.real_oversized_input"),
        size: byte_len,
        usage: usages,
        mapped_at_creation: false,
    });
    let output = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.four_step.real_oversized_output"),
        size: byte_len,
        usage: usages,
        mapped_at_creation: false,
    });
    let mut clear_encoder =
        context
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("wgpu_fft.test.four_step.real_oversized_clear"),
            });
    clear_encoder.clear_buffer(&input, 0, None);
    context.queue.submit([clear_encoder.finish()]);
    context
        .device
        .poll(wgpu::PollType::wait_indefinitely())
        .unwrap();

    let impulses = [
        (3usize, 5usize, Complex64::new(0.75, -0.25)),
        (n0 - 7, n1 - 11, Complex64::new(-0.4, 0.6)),
    ];
    for &(x, y, value) in &impulses {
        let offset = ((y as u64) * (n0 as u64) + x as u64) * 8;
        context.queue.write_buffer(
            &input,
            offset,
            bytemuck::cast_slice(&[value.re as f32, value.im as f32]),
        );
    }

    let sampled_k1 = [0usize, 1usize, n1 - 1];
    let line_bytes = (n0 as u64) * 8;
    let readback_size = line_bytes * sampled_k1.len() as u64;
    let readback = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.four_step.real_oversized_readback"),
        size: readback_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("wgpu_fft.test.four_step.real_oversized_encoder"),
        });
    plan.execute_checked(&context.device, &mut encoder, &input, &output)
        .unwrap();
    for (sample_index, &k1) in sampled_k1.iter().enumerate() {
        encoder.copy_buffer_to_buffer(
            &output,
            (k1 as u64) * line_bytes,
            &readback,
            (sample_index as u64) * line_bytes,
            line_bytes,
        );
    }
    context.queue.submit([encoder.finish()]);
    let actual = read_f32(&context.device, &readback);
    for (sample_index, &k1) in sampled_k1.iter().enumerate() {
        let line = &actual[sample_index * n0 * 2..(sample_index + 1) * n0 * 2];
        for (k0, pair) in line.as_chunks::<2>().0.iter().enumerate() {
            let expected = impulses
                .iter()
                .fold(Complex64::new(0.0, 0.0), |sum, impulse| {
                    let (x, y, value) = *impulse;
                    let angle = -std::f64::consts::TAU
                        * ((k0 * x) as f64 / n0 as f64 + (k1 * y) as f64 / n1 as f64);
                    let twiddle = Complex64::new(angle.cos(), angle.sin());
                    Complex64::new(
                        sum.re + value.re * twiddle.re - value.im * twiddle.im,
                        sum.im + value.re * twiddle.im + value.im * twiddle.re,
                    )
                });
            let error = (f64::from(pair[0]) - expected.re).hypot(f64::from(pair[1]) - expected.im);
            assert!(
                error.is_finite() && error < 5.0e-4,
                "real oversized sample k0={k0} k1={k1}: actual=({}, {}), expected=({}, {}), error={error}",
                pair[0], pair[1], expected.re, expected.im,
            );
        }
    }
    eprintln!("real oversized four-step passed: shape=[{n0}, {n1}] sampled_lines={sampled_k1:?}");
}

fn run_real_oversized_rank3_sampled_case(context: &wgpu_fft::device::GpuContext) {
    let Some((n0, n1, n2, byte_len)) = real_oversized_rank3_shape(&context.device.limits()) else {
        eprintln!(
            "skipping real oversized rank-3 case: adapter limits do not admit a safe smooth shape"
        );
        return;
    };
    eprintln!(
        "real oversized rank-3 start: shape=[{n0}, {n1}, {n2}] bytes={byte_len} maxBind={}",
        context.device.limits().max_storage_buffer_binding_size
    );
    let config = FftConfig::new_nd([n0, n1, n2]).with_normalization(Normalization::None);
    let plan = FftPlan::c2c(&context.device, &context.queue, config).unwrap();
    assert_eq!(
        plan.large_routing_policy().route_mode(),
        LargeRouteMode::LargeOutOfCore
    );
    assert_eq!(
        plan.large_routing_policy().execution_kind(),
        LargeExecutionKind::OutOfCoreFourStep
    );
    assert_eq!(
        plan.diagnostics()
            .stages()
            .iter()
            .filter(|stage| stage.kind == "permutation")
            .count(),
        4
    );

    let usages =
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;
    let input = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.four_step.rank3_oversized_input"),
        size: byte_len,
        usage: usages,
        mapped_at_creation: false,
    });
    let output = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.four_step.rank3_oversized_output"),
        size: byte_len,
        usage: usages,
        mapped_at_creation: false,
    });
    let mut clear_encoder =
        context
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("wgpu_fft.test.four_step.rank3_oversized_clear"),
            });
    clear_encoder.clear_buffer(&input, 0, None);
    context.queue.submit([clear_encoder.finish()]);
    context
        .device
        .poll(wgpu::PollType::wait_indefinitely())
        .unwrap();

    let impulses = [
        (3usize, 5usize, 7usize, Complex64::new(0.75, -0.25)),
        (n0 - 7, n1 - 11, n2 - 13, Complex64::new(-0.4, 0.6)),
    ];
    for &(x, y, z, value) in &impulses {
        let offset = (((z as u64) * (n1 as u64) + y as u64) * (n0 as u64) + x as u64) * 8;
        context.queue.write_buffer(
            &input,
            offset,
            bytemuck::cast_slice(&[value.re as f32, value.im as f32]),
        );
    }

    let sampled_lines = [(0usize, 0usize), (1, 2), (n1 - 1, n2 - 1)];
    let line_bytes = (n0 as u64) * 8;
    let readback = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.four_step.rank3_oversized_readback"),
        size: line_bytes * sampled_lines.len() as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("wgpu_fft.test.four_step.rank3_oversized_encoder"),
        });
    plan.execute_checked(&context.device, &mut encoder, &input, &output)
        .unwrap();
    for (sample_index, &(k1, k2)) in sampled_lines.iter().enumerate() {
        let source_offset = ((k2 as u64) * (n1 as u64) + k1 as u64) * line_bytes;
        encoder.copy_buffer_to_buffer(
            &output,
            source_offset,
            &readback,
            (sample_index as u64) * line_bytes,
            line_bytes,
        );
    }
    context.queue.submit([encoder.finish()]);
    let actual = read_f32(&context.device, &readback);
    for (sample_index, &(k1, k2)) in sampled_lines.iter().enumerate() {
        let line = &actual[sample_index * n0 * 2..(sample_index + 1) * n0 * 2];
        for (k0, pair) in line.as_chunks::<2>().0.iter().enumerate() {
            let expected =
                impulses
                    .iter()
                    .fold(Complex64::new(0.0, 0.0), |sum, &(x, y, z, value)| {
                        let angle = -std::f64::consts::TAU
                            * ((k0 * x) as f64 / n0 as f64
                                + (k1 * y) as f64 / n1 as f64
                                + (k2 * z) as f64 / n2 as f64);
                        let twiddle = Complex64::new(angle.cos(), angle.sin());
                        Complex64::new(
                            sum.re + value.re * twiddle.re - value.im * twiddle.im,
                            sum.im + value.re * twiddle.im + value.im * twiddle.re,
                        )
                    });
            let error = (f64::from(pair[0]) - expected.re).hypot(f64::from(pair[1]) - expected.im);
            assert!(
                error.is_finite() && error < 7.5e-4,
                "rank-3 sample k0={k0} k1={k1} k2={k2}: actual=({}, {}), expected=({}, {}), error={error}",
                pair[0], pair[1], expected.re, expected.im,
            );
        }
    }
    eprintln!(
        "real oversized rank-3 passed: shape=[{n0}, {n1}, {n2}] sampled_lines={sampled_lines:?}"
    );
}

fn run_real_oversized_prime_sampled_case(context: &wgpu_fft::device::GpuContext) {
    let Some((n0, n1, byte_len)) = real_oversized_prime_shape(&context.device.limits()) else {
        eprintln!(
            "skipping real oversized prime case: adapter limits do not admit a safe Rader-containing shape"
        );
        return;
    };
    eprintln!(
        "real oversized prime start: shape=[{n0}, {n1}] bytes={byte_len} maxBind={}",
        context.device.limits().max_storage_buffer_binding_size
    );
    let config = FftConfig::new_nd([n0, n1]).with_normalization(Normalization::None);
    let plan = FftPlan::c2c(&context.device, &context.queue, config).unwrap();
    assert_eq!(plan.axis_kinds(), [AxisKind::Rader, AxisKind::Mixed]);
    assert_eq!(
        plan.large_routing_policy().route_mode(),
        LargeRouteMode::LargeOutOfCore
    );
    assert_eq!(
        plan.large_routing_policy().execution_kind(),
        LargeExecutionKind::OutOfCoreFourStep
    );
    assert!(plan.diagnostics().stages().iter().any(|stage| {
        stage.label == "four-step-axis0-windowed-rader" && stage.kind == "windowed-kernel"
    }));
    let route_diagnostics = plan.large_routing_policy().diagnostics();
    assert!(route_diagnostics
        .factor_splits
        .iter()
        .any(|split| split.axis == Some(0) && split.len == n0 as u64));

    let usages =
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;
    let input = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.four_step.prime_oversized_input"),
        size: byte_len,
        usage: usages,
        mapped_at_creation: false,
    });
    let output = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.four_step.prime_oversized_output"),
        size: byte_len,
        usage: usages,
        mapped_at_creation: false,
    });
    let mut clear_encoder =
        context
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("wgpu_fft.test.four_step.prime_oversized_clear"),
            });
    clear_encoder.clear_buffer(&input, 0, None);
    context.queue.submit([clear_encoder.finish()]);
    context
        .device
        .poll(wgpu::PollType::wait_indefinitely())
        .unwrap();

    let impulses = [
        (3usize, 5usize, Complex64::new(0.75, -0.25)),
        (n0 - 7, n1 - 11, Complex64::new(-0.4, 0.6)),
    ];
    for &(x, y, value) in &impulses {
        let offset = ((y as u64) * (n0 as u64) + x as u64) * 8;
        context.queue.write_buffer(
            &input,
            offset,
            bytemuck::cast_slice(&[value.re as f32, value.im as f32]),
        );
    }

    let sampled_k1 = [0usize, 1usize, n1 - 1];
    let line_bytes = (n0 as u64) * 8;
    let readback = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.four_step.prime_oversized_readback"),
        size: line_bytes * sampled_k1.len() as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("wgpu_fft.test.four_step.prime_oversized_encoder"),
        });
    plan.execute_checked(&context.device, &mut encoder, &input, &output)
        .unwrap();
    for (sample_index, &k1) in sampled_k1.iter().enumerate() {
        encoder.copy_buffer_to_buffer(
            &output,
            (k1 as u64) * line_bytes,
            &readback,
            (sample_index as u64) * line_bytes,
            line_bytes,
        );
    }
    context.queue.submit([encoder.finish()]);
    let actual = read_f32(&context.device, &readback);
    for (sample_index, &k1) in sampled_k1.iter().enumerate() {
        let line = &actual[sample_index * n0 * 2..(sample_index + 1) * n0 * 2];
        for (k0, pair) in line.as_chunks::<2>().0.iter().enumerate() {
            let expected = impulses
                .iter()
                .fold(Complex64::new(0.0, 0.0), |sum, &(x, y, value)| {
                    let angle = -std::f64::consts::TAU
                        * ((k0 * x) as f64 / n0 as f64 + (k1 * y) as f64 / n1 as f64);
                    let twiddle = Complex64::new(angle.cos(), angle.sin());
                    Complex64::new(
                        sum.re + value.re * twiddle.re - value.im * twiddle.im,
                        sum.im + value.re * twiddle.im + value.im * twiddle.re,
                    )
                });
            let error = (f64::from(pair[0]) - expected.re).hypot(f64::from(pair[1]) - expected.im);
            assert!(
                error.is_finite() && error < 1.0e-3,
                "prime oversized sample k0={k0} k1={k1}: actual=({}, {}), expected=({}, {}), error={error}",
                pair[0], pair[1], expected.re, expected.im,
            );
        }
    }
    eprintln!("real oversized prime passed: shape=[{n0}, {n1}] sampled_lines={sampled_k1:?}");
}

fn real_oversized_prime_shape(limits: &wgpu::Limits) -> Option<(usize, usize, u64)> {
    let max_bind = limits.max_storage_buffer_binding_size;
    let max_buffer = limits.max_buffer_size;
    if max_bind < 32 || max_bind >= max_buffer {
        return None;
    }
    let n0 = 1009usize;
    let line_bytes = (n0 as u64).checked_mul(8)?;
    if line_bytes > max_bind {
        return None;
    }
    let first_n1 = usize::try_from(max_bind / line_bytes)
        .ok()?
        .checked_add(1)?;
    let n1 = (first_n1..first_n1.checked_add(100_000)?).find(|&candidate| is_smooth(candidate))?;
    let byte_len = line_bytes.checked_mul(n1 as u64)?;
    let total_complex = (n0 as u64).checked_mul(n1 as u64)?;
    if byte_len <= max_bind
        || byte_len > max_buffer
        || (n1 as u64) * 8 > max_bind
        || total_complex > u64::from(u32::MAX)
        || byte_len > 3 * 1024 * 1024 * 1024
    {
        return None;
    }
    Some((n0, n1, byte_len))
}

fn real_oversized_rank3_shape(limits: &wgpu::Limits) -> Option<(usize, usize, usize, u64)> {
    let max_bind = limits.max_storage_buffer_binding_size;
    let max_buffer = limits.max_buffer_size;
    if max_bind < 32 || max_bind >= max_buffer {
        return None;
    }
    let n0 = 4096usize;
    let n1 = 256usize;
    let plane_bytes = (n0 as u64).checked_mul(n1 as u64)?.checked_mul(8)?;
    if (n0 as u64) * 8 > max_bind || (n1 as u64) * 8 > max_bind {
        return None;
    }
    let first_n2 = usize::try_from(max_bind / plane_bytes)
        .ok()?
        .checked_add(1)?;
    let n2 = (first_n2..first_n2.checked_add(100_000)?).find(|&candidate| is_smooth(candidate))?;
    let byte_len = plane_bytes.checked_mul(n2 as u64)?;
    let total_complex = (n0 as u64).checked_mul(n1 as u64)?.checked_mul(n2 as u64)?;
    if n2 <= 13
        || byte_len <= max_bind
        || byte_len > max_buffer
        || (n2 as u64) * 8 > max_bind
        || total_complex > u64::from(u32::MAX)
        || byte_len > 3 * 1024 * 1024 * 1024
    {
        return None;
    }
    Some((n0, n1, n2, byte_len))
}

fn real_oversized_shape(limits: &wgpu::Limits) -> Option<(usize, usize, u64)> {
    let max_bind = limits.max_storage_buffer_binding_size;
    let max_buffer = limits.max_buffer_size;
    if max_bind < 32 || max_bind >= max_buffer {
        return None;
    }
    let n0 = 4096usize;
    let n0_line_bytes = (n0 as u64) * 8;
    if n0_line_bytes > max_bind {
        return None;
    }
    let first_n1 = usize::try_from(max_bind / n0_line_bytes)
        .ok()?
        .checked_add(1)?;
    let n1 = (first_n1..first_n1.checked_add(100_000)?).find(|&candidate| is_smooth(candidate))?;
    let byte_len = (n0 as u64).checked_mul(n1 as u64)?.checked_mul(8)?;
    let total_complex = (n0 as u64).checked_mul(n1 as u64)?;
    if byte_len <= max_bind
        || byte_len > max_buffer
        || (n1 as u64) * 8 > max_bind
        || total_complex > u64::from(u32::MAX)
        || byte_len > 3 * 1024 * 1024 * 1024
    {
        return None;
    }
    Some((n0, n1, byte_len))
}

fn is_smooth(mut value: usize) -> bool {
    for factor in [2usize, 3, 5, 7, 11, 13] {
        while value.is_multiple_of(factor) {
            value /= factor;
        }
    }
    value == 1
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
        .as_chunks::<2>()
        .0
        .iter()
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
    for (index, (pair, expected)) in actual.as_chunks::<2>().0.iter().zip(expected).enumerate() {
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
    let mapped = slice
        .get_mapped_range()
        .expect("readback buffer should be mapped");
    let values = bytemuck::cast_slice(&mapped).to_vec();
    drop(mapped);
    buffer.unmap();
    values
}
