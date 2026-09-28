#![cfg(not(target_arch = "wasm32"))]

//! Opt-in GPU coverage for the public per-plan tuning surface.

use std::sync::mpsc;

use wgpu_fft::math::{reference_c2c_nd_f64, Complex64};
use wgpu_fft::{
    export_pipeline_cache_snapshot, AxisKind, C2cRoute, FftBlockerKind, FftConfig, FftError,
    FftLargeRoute, FftPlan, FftTuning, FftTuningErrorKind, LargeExecutionKind, LargeRouteMode,
    Normalization,
};

#[test]
fn gpu_public_tuning_controls_are_live_and_correct() {
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
    let adapter_info = context.adapter.get_info();
    let limits = context.device.limits();
    eprintln!("adapter: {adapter_info:?}");
    eprintln!(
        "backend={:?} maxInvocations={} maxWorkgroupX={} maxStorage={} maxBinding={} maxBuffer={}",
        adapter_info.backend,
        limits.max_compute_invocations_per_workgroup,
        limits.max_compute_workgroup_size_x,
        limits.max_compute_workgroup_storage_size,
        limits.max_storage_buffer_binding_size,
        limits.max_buffer_size,
    );

    assert_default_equivalence(&context);
    assert_forced_bluestein_correctness(&context);
    assert_forced_rader_correctness(&context);
    assert_workgroup_sweep(&context);
    assert_public_large_limit_routes(&context);
    assert_segmented_burst_depths(&context);
    assert_structured_tuning_errors(&context);

    #[cfg(windows)]
    std::mem::forget(context);
}

fn assert_default_equivalence(context: &wgpu_fft::device::GpuContext) {
    let implicit_config = FftConfig::new_nd([16, 15]).with_normalization(Normalization::None);
    let explicit_config = implicit_config.clone().with_tuning(FftTuning::default());
    let implicit = FftPlan::c2c(&context.device, &context.queue, implicit_config.clone()).unwrap();
    let explicit = FftPlan::c2c(&context.device, &context.queue, explicit_config).unwrap();

    assert_eq!(implicit.config(), implicit_config);
    assert_eq!(implicit.route(), explicit.route());
    assert_eq!(
        implicit.large_routing_policy(),
        explicit.large_routing_policy()
    );
    assert_eq!(implicit.diagnostics(), explicit.diagnostics());
    assert_eq!(
        implicit.diagnostics().active_tuning().requested(),
        &FftTuning::default()
    );
}

fn assert_forced_bluestein_correctness(context: &wgpu_fft::device::GpuContext) {
    let config = FftConfig::new(17)
        .with_normalization(Normalization::None)
        .with_tuning(FftTuning::default().with_force_bluestein_axes([0]));
    let input = test_signal(config.total_complex_len().unwrap());
    let expected = reference_f64(&input, &config);
    let (actual, plan) = execute_config(context, config, &input, "forced-bluestein");

    assert_eq!(plan.route(), C2cRoute::Bluestein);
    assert_eq!(plan.axis_kinds(), &[AxisKind::Bluestein]);
    assert_eq!(
        plan.diagnostics()
            .active_tuning()
            .effective()
            .force_bluestein_axes(),
        &[0]
    );
    assert_matches_reference(&actual, &expected, "forced Bluestein prime");
}

fn assert_forced_rader_correctness(context: &wgpu_fft::device::GpuContext) {
    let automatic_tuning = FftTuning::default().with_rader_max_prime(13);
    let automatic = FftPlan::c2c(
        &context.device,
        &context.queue,
        FftConfig::new(17).with_tuning(automatic_tuning.clone()),
    )
    .unwrap();
    assert_eq!(automatic.route(), C2cRoute::Bluestein);

    let config = FftConfig::new(17)
        .with_normalization(Normalization::None)
        .with_tuning(automatic_tuning.with_force_rader_axes([0]));
    let input = test_signal(config.total_complex_len().unwrap());
    let expected = reference_f64(&input, &config);
    let (actual, plan) = execute_config(context, config, &input, "forced-rader");

    assert_eq!(plan.route(), C2cRoute::Rader);
    assert_eq!(plan.axis_kinds(), &[AxisKind::Rader]);
    assert_matches_reference(&actual, &expected, "forced Rader prime");
}

fn assert_workgroup_sweep(context: &wgpu_fft::device::GpuContext) {
    assert!(
        context
            .device
            .limits()
            .max_compute_invocations_per_workgroup
            >= 256
            && context.device.limits().max_compute_workgroup_size_x >= 256,
        "adapter cannot exercise the requested 32/64/128/256 workgroup sweep"
    );

    for (label, len, expected_stage) in [
        ("staged-radix13", 13usize, "mixed-radix-stockham-stage"),
        ("fused-pow2", 256usize, "fused-pow2-workgroup-stage"),
    ] {
        let baseline_config = FftConfig::new(len).with_normalization(Normalization::None);
        let input = test_signal(len);
        let expected = reference_f64(&input, &baseline_config);
        let (baseline, _) = execute_config(
            context,
            baseline_config,
            &input,
            &format!("{label}-baseline"),
        );

        for workgroup_size in [32u32, 64, 128, 256] {
            let tuning = FftTuning::default()
                .with_workgroup_size(workgroup_size)
                .with_fused_workgroup_size(workgroup_size);
            let config = FftConfig::new(len)
                .with_normalization(Normalization::None)
                .with_tuning(tuning);
            let case_label = format!("{label}-wg{workgroup_size}");
            let (actual, plan) = execute_config(context, config, &input, &case_label);
            let diagnostics = plan.diagnostics();

            assert!(
                diagnostics
                    .stages()
                    .iter()
                    .any(|stage| stage.label == expected_stage),
                "{case_label}: expected stage {expected_stage}: {:?}",
                diagnostics.stages()
            );
            assert_eq!(
                diagnostics.active_tuning().effective().workgroup_size(),
                workgroup_size
            );
            assert_eq!(
                diagnostics
                    .active_tuning()
                    .effective()
                    .fused_workgroup_size(),
                workgroup_size
            );
            assert_outputs_close(&actual, &baseline, &case_label);
            assert_matches_reference(&actual, &expected, &case_label);
        }
    }

    let snapshot = export_pipeline_cache_snapshot(&context.device);
    for workgroup_size in [32u32, 64, 128, 256] {
        let workgroup_fragment = format!("workgroup={workgroup_size}");
        assert!(
            snapshot.pipeline_keys().iter().any(|key| {
                key.contains("shader:v3:stockham")
                    && key.contains("n=13")
                    && key.contains(&workgroup_fragment)
            }),
            "missing staged pipeline cache key for workgroup size {workgroup_size}"
        );
        // The default fused workgroup size lets the line run in a register
        // kernel, which sizes its own workgroup.
        if workgroup_size != FftTuning::default().fused_workgroup_size() {
            assert!(
                snapshot.pipeline_keys().iter().any(|key| {
                    key.contains("shader:v3:fused-pow2")
                        && key.contains("n=256")
                        && key.contains(&workgroup_fragment)
                        && !key.contains(":registers=")
                }),
                "missing fused pipeline cache key for workgroup size {workgroup_size}"
            );
        }
    }
    assert!(
        snapshot.pipeline_keys().iter().any(|key| {
            key.contains("shader:v3:fused-pow2")
                && key.contains("n=256")
                && key.contains(":registers=")
        }),
        "missing register pipeline cache key for the default fused workgroup size"
    );
}

fn assert_public_large_limit_routes(context: &wgpu_fft::device::GpuContext) {
    let four_step_config = FftConfig::new_nd([15, 14])
        .with_normalization(Normalization::None)
        .with_tuning(
            FftTuning::default()
                .with_max_storage_buffer_binding_size(512u64)
                .with_max_buffer_size(4_096u64),
        );
    let four_step = FftPlan::c2c(&context.device, &context.queue, four_step_config).unwrap();
    assert_eq!(
        four_step.large_routing_policy().route_mode(),
        LargeRouteMode::LargeOutOfCore
    );
    assert_eq!(
        four_step.large_routing_policy().execution_kind(),
        LargeExecutionKind::OutOfCoreFourStep
    );
    assert_eq!(
        four_step
            .diagnostics()
            .active_tuning()
            .effective()
            .max_storage_buffer_binding_size(),
        Some(512)
    );

    let segmented = build_public_segmented_plan(context, 2);
    assert_eq!(
        segmented.large_routing_policy().route_mode(),
        LargeRouteMode::LargeOutOfCore
    );
    assert_eq!(
        segmented.large_routing_policy().execution_kind(),
        LargeExecutionKind::SegmentedFullVolume
    );
    let diagnostics = segmented.diagnostics();
    let effective = diagnostics.active_tuning().effective();
    assert_eq!(effective.max_storage_buffer_binding_size(), Some(256));
    assert_eq!(effective.max_buffer_size(), Some(256));
}

fn assert_segmented_burst_depths(context: &wgpu_fft::device::GpuContext) {
    for depth in 1usize..=3 {
        let plan = build_public_segmented_plan(context, depth);
        assert_eq!(plan.config().tuning().segmented_burst_depth(), depth);
        let diagnostics = plan.diagnostics();
        assert_eq!(
            diagnostics
                .active_tuning()
                .requested()
                .segmented_burst_depth(),
            depth
        );
        assert_eq!(
            diagnostics
                .active_tuning()
                .effective()
                .segmented_burst_depth(),
            depth
        );
        assert_eq!(
            plan.large_routing_policy().execution_kind(),
            LargeExecutionKind::SegmentedFullVolume
        );
    }
}

fn build_public_segmented_plan(
    context: &wgpu_fft::device::GpuContext,
    burst_depth: usize,
) -> FftPlan {
    let config = FftConfig::new_nd([8, 12])
        .with_normalization(Normalization::None)
        .with_tuning(
            FftTuning::default()
                .with_max_storage_buffer_binding_size(1_024u64)
                .with_max_buffer_size(256u64)
                .with_segmented_burst_depth(burst_depth),
        );
    FftPlan::c2c(&context.device, &context.queue, config).unwrap()
}

fn assert_structured_tuning_errors(context: &wgpu_fft::device::GpuContext) {
    assert!(matches!(
        FftConfig::new(16)
            .with_tuning(FftTuning::default().with_workgroup_size(0))
            .validate(),
        Err(FftError::InvalidTuning {
            kind: FftTuningErrorKind::InvalidValue,
            field: "workgroup_size",
            ..
        })
    ));

    let limits = context.device.limits();
    let device_ceiling = limits
        .max_compute_invocations_per_workgroup
        .min(limits.max_compute_workgroup_size_x);
    let invalid_workgroup_size = if device_ceiling.is_power_of_two() {
        device_ceiling.checked_mul(2).unwrap()
    } else {
        device_ceiling.next_power_of_two()
    };
    let device_tuning = FftTuning::default().with_workgroup_size(invalid_workgroup_size);
    let device_error = match FftPlan::c2c_with_diagnostics(
        &context.device,
        &context.queue,
        FftConfig::new(16).with_tuning(device_tuning.clone()),
    ) {
        Ok(_) => panic!("oversized workgroup tuning unexpectedly built a plan"),
        Err(error) => error,
    };
    assert!(matches!(
        device_error.error(),
        FftError::InvalidTuning {
            kind: FftTuningErrorKind::DeviceLimit,
            field: "workgroup_size",
            ..
        }
    ));
    assert_eq!(
        device_error.diagnostics().active_tuning().requested(),
        &device_tuning
    );
    assert_eq!(
        device_error.diagnostics().blockers()[0].kind,
        FftBlockerKind::DeviceLimit
    );

    let forced_tuning = FftTuning::default().with_large_route(FftLargeRoute::ForceChunk);
    let forced_error = match FftPlan::c2c_with_diagnostics(
        &context.device,
        &context.queue,
        FftConfig::new(16).with_tuning(forced_tuning.clone()),
    ) {
        Ok(_) => panic!("infeasible forced chunk route unexpectedly built a plan"),
        Err(error) => error,
    };
    assert!(matches!(
        forced_error.error(),
        FftError::InvalidTuning {
            kind: FftTuningErrorKind::RouteInfeasible,
            field: "large_route",
            value,
            ..
        } if value == "force-chunk"
    ));
    assert_eq!(
        forced_error.diagnostics().active_tuning().requested(),
        &forced_tuning
    );
    assert_eq!(
        forced_error.diagnostics().blockers()[0].kind,
        FftBlockerKind::Route
    );
}

fn execute_config(
    context: &wgpu_fft::device::GpuContext,
    config: FftConfig,
    input: &[f32],
    label: &str,
) -> (Vec<f32>, FftPlan) {
    let plan = FftPlan::c2c(&context.device, &context.queue, config).unwrap();
    let byte_len = std::mem::size_of_val(input) as u64;
    let input_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(&format!("wgpu_fft.test.tuning.{label}.input")),
        size: byte_len,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let output_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(&format!("wgpu_fft.test.tuning.{label}.output")),
        size: byte_len,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(&format!("wgpu_fft.test.tuning.{label}.readback")),
        size: byte_len,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    context
        .queue
        .write_buffer(&input_buffer, 0, bytemuck::cast_slice(input));

    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some(&format!("wgpu_fft.test.tuning.{label}.encoder")),
        });
    plan.execute_checked(&context.device, &mut encoder, &input_buffer, &output_buffer)
        .unwrap();
    encoder.copy_buffer_to_buffer(&output_buffer, 0, &readback, 0, byte_len);
    context.queue.submit([encoder.finish()]);

    let slice = readback.slice(..);
    let (sender, receiver) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        sender.send(result).unwrap();
    });
    context
        .device
        .poll(wgpu::PollType::wait_indefinitely())
        .unwrap();
    receiver.recv().unwrap().unwrap();
    let mapped = slice
        .get_mapped_range()
        .expect("readback buffer should be mapped");
    let values = bytemuck::cast_slice(&mapped).to_vec();
    drop(mapped);
    readback.unmap();
    (values, plan)
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

fn assert_matches_reference(actual: &[f32], expected: &[Complex64], label: &str) {
    assert_eq!(actual.len(), expected.len() * 2);
    let mut error_energy = 0.0f64;
    let mut reference_energy = 0.0f64;
    let mut max_error = 0.0f64;
    let mut max_reference = 0.0f64;
    for (pair, expected) in actual.as_chunks::<2>().0.iter().zip(expected) {
        let error = (f64::from(pair[0]) - expected.re).hypot(f64::from(pair[1]) - expected.im);
        let magnitude = expected.re.hypot(expected.im);
        error_energy += error * error;
        reference_energy += magnitude * magnitude;
        max_error = max_error.max(error);
        max_reference = max_reference.max(magnitude);
    }
    let rms_relative = (error_energy / reference_energy.max(f64::MIN_POSITIVE)).sqrt();
    let max_relative = max_error / max_reference.max(f64::MIN_POSITIVE);
    assert!(
        rms_relative < 5.0e-6 && max_relative < 2.0e-5,
        "{label}: rms_relative={rms_relative:e} max_relative={max_relative:e}"
    );
}

fn assert_outputs_close(actual: &[f32], expected: &[f32], label: &str) {
    assert_eq!(actual.len(), expected.len());
    let mut error_energy = 0.0f64;
    let mut expected_energy = 0.0f64;
    for (&actual, &expected) in actual.iter().zip(expected) {
        let error = f64::from(actual) - f64::from(expected);
        error_energy += error * error;
        expected_energy += f64::from(expected) * f64::from(expected);
    }
    let rms_relative = (error_energy / expected_energy.max(f64::MIN_POSITIVE)).sqrt();
    assert!(
        rms_relative < 5.0e-6,
        "{label}: workgroup sweep differs from baseline: {rms_relative:e}"
    );
}
