#![cfg(not(target_arch = "wasm32"))]

//! Opt-in portable-df64 normal C2C correctness and routing coverage.

use std::mem::ManuallyDrop;
use std::sync::mpsc;

use wgpu_fft::math::{reference_c2c_nd_f64, Complex64, ComplexDoubleFloat};
use wgpu_fft::{
    BufferView, C2cRoute, FftConfig, FftError, FftLogicalLayout, FftLogicalView, FftPlan,
    FftPrecision, Normalization,
};

const DF64_RMS_LIMIT: f64 = 1.0e-11;
const FUSED_POW2_LABEL: &str = "fused-pow2-workgroup-stage";
const FUSED_SMOOTH_LABEL: &str = "fused-smooth-workgroup-stage";
const STOCKHAM_LABEL: &str = "mixed-radix-stockham-stage";

#[test]
fn gpu_portable_df64_normal_c2c() {
    if std::env::var_os("WGPU_FFT_RUN_GPU_TESTS").is_none() {
        eprintln!("skipping GPU test; set WGPU_FFT_RUN_GPU_TESTS=1 to run it");
        return;
    }
    pollster::block_on(run_df64_cases());
}

#[test]
fn gpu_portable_df64_strided_c2c() {
    if std::env::var_os("WGPU_FFT_RUN_GPU_TESTS").is_none() {
        eprintln!("skipping GPU test; set WGPU_FFT_RUN_GPU_TESTS=1 to run it");
        return;
    }
    pollster::block_on(run_df64_strided_cases());
}

async fn run_df64_strided_cases() {
    let Some(context) = wgpu_fft::device::request_default_device().await else {
        eprintln!("skipping GPU test; no suitable wgpu adapter was found");
        return;
    };
    let context = ManuallyDrop::new(context);
    let info = context.adapter.get_info();
    eprintln!(
        "df64 strided adapter: name={:?}, backend={:?}, device_type={:?}, driver={:?}",
        info.name, info.backend, info.device_type, info.driver
    );
    let featureless = request_featureless_device(
        &context.adapter,
        context.adapter.limits(),
        "wgpu_fft.test.df64_strided_featureless_device",
    )
    .await;
    assert!(featureless.0.features().is_empty());
    if info.backend == wgpu::Backend::Dx12 {
        eprintln!(
            "df64 DX12 strided coverage: representative batched N=2 layout; exhaustive strided ND coverage runs on Vulkan"
        );
        run_strided_normal_case(
            &featureless.0,
            &featureless.1,
            "dx12-strided-inverse-n2-b2",
            FftConfig::inverse(2)
                .with_batch(2)
                .with_precision(FftPrecision::Df64),
            (3, 2, 5),
            (2, 3, 7),
        );
        return;
    }
    run_strided_normal_cases(&featureless.0, &featureless.1);
}

async fn run_df64_cases() {
    let Some(context) = wgpu_fft::device::request_default_device().await else {
        eprintln!("skipping GPU test; no suitable wgpu adapter was found");
        return;
    };
    // Native wgpu teardown can stall on Windows after an assertion panic.
    let context = ManuallyDrop::new(context);
    let info = context.adapter.get_info();
    eprintln!(
        "df64 adapter: name={:?}, backend={:?}, device_type={:?}, driver={:?}",
        info.name, info.backend, info.device_type, info.driver
    );

    let primary = request_featureless_device(
        &context.adapter,
        context.adapter.limits(),
        "wgpu_fft.test.df64_featureless_device",
    )
    .await;
    assert!(primary.0.features().is_empty());
    verify_structured_deferred_gates(&primary.0, &primary.1);
    if info.backend == wgpu::Backend::Dx12 {
        eprintln!(
            "df64 DX12 coverage: exact arithmetic canaries plus a representative fused-pow2 C2C case; exhaustive topology coverage runs on Vulkan"
        );
        run_dx12_normal_cases(&primary.0, &primary.1);
        return;
    }
    run_small_normal_cases(&primary.0, &primary.1);
    run_fused_and_multipass_cases(&context.adapter, &primary).await;
}

async fn request_featureless_device(
    adapter: &wgpu::Adapter,
    limits: wgpu::Limits,
    label: &'static str,
) -> ManuallyDrop<(wgpu::Device, wgpu::Queue)> {
    let pair = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some(label),
            required_features: wgpu::Features::empty(),
            required_limits: limits,
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: wgpu::MemoryHints::MemoryUsage,
            trace: wgpu::Trace::Off,
        })
        .await
        .expect("portable df64 requires no optional device features");
    ManuallyDrop::new(pair)
}

fn verify_structured_deferred_gates(device: &wgpu::Device, queue: &wgpu::Queue) {
    assert!(matches!(
        FftPlan::r2c(
            device,
            queue,
            FftConfig::new(16).with_precision(FftPrecision::Df64),
        ),
        Err(FftError::PrecisionUnsupported {
            requested: FftPrecision::Df64,
            ..
        })
    ));
}

fn run_small_normal_cases(device: &wgpu::Device, queue: &wgpu::Queue) {
    for (label, config, expected_route) in [
        (
            "direct-forward-n1",
            FftConfig::new(1)
                .with_normalization(Normalization::None)
                .with_precision(FftPrecision::Df64),
            C2cRoute::DirectDft,
        ),
        (
            "direct-inverse-n1",
            FftConfig::inverse(1).with_precision(FftPrecision::Df64),
            C2cRoute::DirectDft,
        ),
        (
            "mixed-forward-n60",
            FftConfig::new(60)
                .with_normalization(Normalization::None)
                .with_precision(FftPrecision::Df64),
            C2cRoute::MixedRadix,
        ),
        (
            "mixed-inverse-n60",
            FftConfig::inverse(60).with_precision(FftPrecision::Df64),
            C2cRoute::MixedRadix,
        ),
        (
            "batched-forward-n45-b3",
            FftConfig::new(45)
                .with_batch(3)
                .with_normalization(Normalization::None)
                .with_precision(FftPrecision::Df64),
            C2cRoute::MixedRadix,
        ),
        (
            "nd-inverse-8x15-b2",
            FftConfig::inverse_nd([8, 15])
                .with_batch(2)
                .with_precision(FftPrecision::Df64),
            C2cRoute::MixedRadix,
        ),
        (
            "rader-forward-n17-staged",
            FftConfig::new(17)
                .with_normalization(Normalization::None)
                .with_precision(FftPrecision::Df64),
            C2cRoute::Rader,
        ),
        (
            "bluestein-inverse-n34-staged",
            FftConfig::inverse(34).with_precision(FftPrecision::Df64),
            C2cRoute::Bluestein,
        ),
        (
            "rader-inverse-n67-fused",
            FftConfig::inverse(67).with_precision(FftPrecision::Df64),
            C2cRoute::Rader,
        ),
        (
            "bluestein-forward-n68-fused",
            FftConfig::new(68)
                .with_normalization(Normalization::None)
                .with_precision(FftPrecision::Df64),
            C2cRoute::Bluestein,
        ),
        (
            "axis-sequence-inverse-3x17-b2",
            FftConfig::inverse_nd([3, 17])
                .with_batch(2)
                .with_precision(FftPrecision::Df64),
            C2cRoute::AxisSequence,
        ),
    ] {
        let (_, plan) = execute_reference_case(device, queue, label, config);
        assert_eq!(plan.route(), expected_route, "{label}: route");
        let stage_labels = kernel_labels(&plan);
        if label.ends_with("-fused") {
            let expected = match expected_route {
                C2cRoute::Rader => "rader-fused-workgroup-stage",
                C2cRoute::Bluestein => "bluestein-fused-workgroup-stage",
                _ => unreachable!(),
            };
            assert_eq!(
                stage_labels,
                vec![expected.to_owned()],
                "{label}: fused graph"
            );
            assert_eq!(plan.workspace_size_bytes(), 0, "{label}: workspace");
        } else if label.ends_with("-staged") {
            assert!(
                stage_labels
                    .iter()
                    .all(|stage| !stage.contains("fused-workgroup")),
                "{label}: expected staged graph, got {stage_labels:?}"
            );
        }
    }
}

fn run_strided_normal_cases(device: &wgpu::Device, queue: &wgpu::Queue) {
    for (label, config, input_layout_args, output_layout_args) in [
        (
            "strided-forward-nd-8x15-b2",
            FftConfig::new_nd([8, 15])
                .with_batch(2)
                .with_normalization(Normalization::None)
                .with_precision(FftPrecision::Df64),
            (3, 2, 11),
            (5, 3, 13),
        ),
        (
            "strided-inverse-nd-9x10-b2",
            FftConfig::inverse_nd([9, 10])
                .with_batch(2)
                .with_precision(FftPrecision::Df64),
            (7, 3, 17),
            (2, 4, 19),
        ),
    ] {
        run_strided_normal_case(
            device,
            queue,
            label,
            config,
            input_layout_args,
            output_layout_args,
        );
    }
}

fn run_strided_normal_case(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &str,
    config: FftConfig,
    input_layout_args: (u64, u64, u64),
    output_layout_args: (u64, u64, u64),
) {
    let logical_per_batch = config.logical_complex_len().unwrap() as u64;
    let batch = config.batch() as u64;
    let input_layout = df64_strided_layout(logical_per_batch, input_layout_args);
    let output_layout = df64_strided_layout(logical_per_batch, output_layout_args);
    let input = test_signal(config.total_complex_len().unwrap());
    let expected = reference_c2c_nd_f64(&input, &config).unwrap();
    let sentinel = df64_padding_sentinel();
    let physical_input =
        scatter_strided_df64(&input, input_layout, logical_per_batch, batch, sentinel);
    let output_len = df64_layout_span(output_layout, logical_per_batch, batch) as usize;
    let physical_output_initial = vec![sentinel; output_len];
    let input_bytes = std::mem::size_of_val(physical_input.as_slice()) as u64;
    let output_bytes = std::mem::size_of_val(physical_output_initial.as_slice()) as u64;
    assert_eq!(input_bytes % 16, 0, "{label}: input alignment");
    assert_eq!(output_bytes % 16, 0, "{label}: output alignment");

    let input_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.df64.strided_input"),
        size: input_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.df64.strided_output"),
        size: output_bytes,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.df64.strided_readback"),
        size: output_bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    queue.write_buffer(&input_buffer, 0, bytemuck::cast_slice(&physical_input));
    queue.write_buffer(
        &output_buffer,
        0,
        bytemuck::cast_slice(&physical_output_initial),
    );

    let plan = FftPlan::c2c(device, queue, config)
        .unwrap_or_else(|error| panic!("{label}: df64 strided plan failed: {error}"));
    assert_eq!(plan.route(), C2cRoute::MixedRadix, "{label}: route");
    assert_eq!(
        plan.required_input_buffer_size_bytes(),
        logical_per_batch * batch * 16,
        "{label}: logical byte size"
    );
    let input_view = FftLogicalView::new(BufferView::whole(&input_buffer), input_layout).unwrap();
    let output_view =
        FftLogicalView::new(BufferView::whole(&output_buffer), output_layout).unwrap();
    let diagnostics = plan.diagnostics_for_logical_views(device, &input_view, &output_view);
    assert!(diagnostics.blockers().is_empty(), "{label}: diagnostics");
    assert!(diagnostics
        .stages()
        .iter()
        .any(|stage| stage.kind == "strided-pack"));
    assert!(diagnostics
        .stages()
        .iter()
        .any(|stage| stage.kind == "strided-unpack"));

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("wgpu_fft.test.df64.strided_encoder"),
    });
    plan.execute_logical_views(device, &mut encoder, input_view, output_view)
        .unwrap_or_else(|error| panic!("{label}: df64 strided execution failed: {error}"));
    encoder.copy_buffer_to_buffer(&output_buffer, 0, &readback, 0, output_bytes);
    queue.submit([encoder.finish()]);

    let physical_output = read_df64(device, &readback);
    let actual = gather_strided_df64(&physical_output, output_layout, logical_per_batch, batch);
    assert_df64_accuracy(label, &actual, &expected);
    assert_df64_padding_preserved(
        label,
        &physical_output,
        output_layout,
        logical_per_batch,
        batch,
        sentinel,
    );
}

fn run_dx12_normal_cases(device: &wgpu::Device, queue: &wgpu::Queue) {
    let pow2_label = "dx12-fused-pow2-inverse-n2";
    let (_, pow2_plan) = execute_reference_case(
        device,
        queue,
        pow2_label,
        FftConfig::inverse(2).with_precision(FftPrecision::Df64),
    );
    assert_eq!(
        pow2_plan.route(),
        C2cRoute::MixedRadix,
        "{pow2_label}: route"
    );
    assert_eq!(kernel_labels(&pow2_plan), vec![FUSED_POW2_LABEL]);

    let prime_label = "dx12-rader-forward-n17-staged";
    let (_, prime_plan) = execute_reference_case(
        device,
        queue,
        prime_label,
        FftConfig::new(17)
            .with_normalization(Normalization::None)
            .with_precision(FftPrecision::Df64),
    );
    assert_eq!(prime_plan.route(), C2cRoute::Rader, "{prime_label}: route");
    assert!(
        kernel_labels(&prime_plan)
            .iter()
            .all(|stage| !stage.contains("fused-workgroup")),
        "{prime_label}: expected staged Rader graph"
    );
}

async fn run_fused_and_multipass_cases(
    adapter: &wgpu::Adapter,
    primary: &ManuallyDrop<(wgpu::Device, wgpu::Queue)>,
) {
    let storage_limit = primary.0.limits().max_compute_workgroup_storage_size;
    let config_2048 = FftConfig::new(2048)
        .with_normalization(Normalization::None)
        .with_precision(FftPrecision::Df64);
    let input_2048 = test_signal(2048);
    let expected_2048 = sampled_reference_1d(&input_2048, &config_2048, 64);
    let (fused, fused_plan) = execute_c2c_df64(
        &primary.0,
        &primary.1,
        config_2048.clone(),
        &input_2048,
        "pow2-forward-n2048",
    );
    assert_sampled_df64_accuracy("pow2-forward-n2048", &fused, &expected_2048);
    if storage_limit >= 2048 * 16 {
        assert_eq!(kernel_labels(&fused_plan), vec![FUSED_POW2_LABEL]);
        assert_eq!(fused_plan.workspace_size_bytes(), 0);
    }

    if adapter.limits().max_compute_workgroup_storage_size >= 16 * 1024 {
        let mut low_limits = adapter.limits();
        low_limits.max_compute_workgroup_storage_size = 16 * 1024;
        let low = request_featureless_device(
            adapter,
            low_limits,
            "wgpu_fft.test.df64_low_storage_device",
        )
        .await;
        let (multipass, multipass_plan) = execute_c2c_df64(
            &low.0,
            &low.1,
            config_2048,
            &input_2048,
            "pow2-forward-n2048-forced-multipass",
        );
        assert_stockham_plan(&multipass_plan, 2048, "forced-multipass");
        assert_sampled_df64_accuracy("pow2-forward-n2048-forced", &multipass, &expected_2048);
        let fused_reference = fused
            .iter()
            .map(|value| Complex64::new(value.re().to_f64(), value.im().to_f64()))
            .collect::<Vec<_>>();
        assert_df64_accuracy("pow2-fused-vs-multipass", &multipass, &fused_reference);
    }

    let config_4096 = FftConfig::inverse(4096).with_precision(FftPrecision::Df64);
    let input_4096 = test_signal(4096);
    let expected_4096 = sampled_reference_1d(&input_4096, &config_4096, 64);
    let (actual_4096, plan_4096) = execute_c2c_df64(
        &primary.0,
        &primary.1,
        config_4096,
        &input_4096,
        "pow2-inverse-n4096",
    );
    assert_sampled_df64_accuracy("pow2-inverse-n4096", &actual_4096, &expected_4096);
    assert_stockham_plan(&plan_4096, 4096, "pow2-inverse-n4096");

    let config_3000 = FftConfig::new(3000)
        .with_normalization(Normalization::None)
        .with_precision(FftPrecision::Df64);
    let input_3000 = test_signal(3000);
    let expected_3000 = sampled_reference_1d(&input_3000, &config_3000, 64);
    let (actual_3000, smooth_plan) = execute_c2c_df64(
        &primary.0,
        &primary.1,
        config_3000,
        &input_3000,
        "smooth-forward-n3000",
    );
    assert_sampled_df64_accuracy("smooth-forward-n3000", &actual_3000, &expected_3000);
    if storage_limit >= 3000 * 16 {
        assert_eq!(kernel_labels(&smooth_plan), vec![FUSED_SMOOTH_LABEL]);
        assert_eq!(smooth_plan.workspace_size_bytes(), 0);
    } else {
        assert_stockham_plan(&smooth_plan, 3000, "smooth-forward-n3000");
    }
}

fn execute_reference_case(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &str,
    config: FftConfig,
) -> (Vec<ComplexDoubleFloat>, FftPlan) {
    let input = test_signal(config.total_complex_len().unwrap());
    let expected = reference_c2c_nd_f64(&input, &config).unwrap();
    let (actual, plan) = execute_c2c_df64(device, queue, config, &input, label);
    assert_df64_accuracy(label, &actual, &expected);
    (actual, plan)
}

fn execute_c2c_df64(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    config: FftConfig,
    input: &[Complex64],
    label: &str,
) -> (Vec<ComplexDoubleFloat>, FftPlan) {
    let plan = FftPlan::c2c(device, queue, config)
        .unwrap_or_else(|error| panic!("{label}: df64 plan creation failed: {error}"));
    let input = input
        .iter()
        .map(|value| ComplexDoubleFloat::from_f64(value.re, value.im))
        .collect::<Vec<_>>();
    let byte_len = std::mem::size_of_val(input.as_slice()) as u64;
    assert_eq!(plan.required_input_buffer_size_bytes(), byte_len);
    assert_eq!(plan.required_output_buffer_size_bytes(), byte_len);

    let input_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.df64.input"),
        size: byte_len,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.df64.output"),
        size: byte_len,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.df64.readback"),
        size: byte_len,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    queue.write_buffer(&input_buffer, 0, bytemuck::cast_slice(&input));

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("wgpu_fft.test.df64.encoder"),
    });
    plan.execute_checked(device, &mut encoder, &input_buffer, &output_buffer)
        .unwrap_or_else(|error| panic!("{label}: df64 execution failed: {error}"));
    encoder.copy_buffer_to_buffer(&output_buffer, 0, &readback, 0, byte_len);
    queue.submit([encoder.finish()]);

    (read_df64(device, &readback), plan)
}

fn read_df64(device: &wgpu::Device, readback: &wgpu::Buffer) -> Vec<ComplexDoubleFloat> {
    let slice = readback.slice(..);
    let (sender, receiver) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        sender.send(result).unwrap();
    });
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    receiver.recv().unwrap().unwrap();
    let mapped = slice.get_mapped_range();
    let values = bytemuck::cast_slice(&mapped).to_vec();
    drop(mapped);
    readback.unmap();
    values
}

fn test_signal(complex_len: usize) -> Vec<Complex64> {
    (0..complex_len)
        .map(|index| {
            let x = index as f64 + 1.0;
            Complex64::new(
                (x * 0.017).sin() * 0.7 + (x * 0.031).cos() * 0.2,
                (x * 0.023).cos() * 0.5 - (x * 0.011).sin() * 0.3,
            )
        })
        .collect()
}

fn df64_strided_layout(
    logical_per_batch: u64,
    (element_offset, element_stride, batch_gap): (u64, u64, u64),
) -> FftLogicalLayout {
    let per_batch_span = element_stride * (logical_per_batch - 1) + 1;
    FftLogicalLayout::new(element_offset, element_stride)
        .unwrap()
        .with_batch_stride(per_batch_span + batch_gap)
}

fn df64_layout_span(layout: FftLogicalLayout, logical_per_batch: u64, batch: u64) -> u64 {
    let per_batch_span = layout.element_stride * (logical_per_batch - 1) + 1;
    let batch_stride = layout.batch_stride.unwrap_or(per_batch_span);
    layout.element_offset + (batch - 1) * batch_stride + per_batch_span
}

fn df64_physical_index(
    layout: FftLogicalLayout,
    logical_per_batch: u64,
    logical_index: u64,
) -> usize {
    let per_batch_span = layout.element_stride * (logical_per_batch - 1) + 1;
    let batch_stride = layout.batch_stride.unwrap_or(per_batch_span);
    let batch = logical_index / logical_per_batch;
    let element = logical_index - batch * logical_per_batch;
    (layout.element_offset + batch * batch_stride + element * layout.element_stride) as usize
}

fn df64_padding_sentinel() -> ComplexDoubleFloat {
    ComplexDoubleFloat {
        re_hi: f32::from_bits(0x461c_4000),
        re_lo: f32::from_bits(0xba83_126f),
        im_hi: f32::from_bits(0xc5f6_e000),
        im_lo: f32::from_bits(0x3980_0001),
    }
}

fn scatter_strided_df64(
    logical: &[Complex64],
    layout: FftLogicalLayout,
    logical_per_batch: u64,
    batch: u64,
    sentinel: ComplexDoubleFloat,
) -> Vec<ComplexDoubleFloat> {
    let mut physical = vec![sentinel; df64_layout_span(layout, logical_per_batch, batch) as usize];
    for (logical_index, value) in logical.iter().enumerate() {
        let physical_index = df64_physical_index(layout, logical_per_batch, logical_index as u64);
        physical[physical_index] = ComplexDoubleFloat::from_f64(value.re, value.im);
    }
    physical
}

fn gather_strided_df64(
    physical: &[ComplexDoubleFloat],
    layout: FftLogicalLayout,
    logical_per_batch: u64,
    batch: u64,
) -> Vec<ComplexDoubleFloat> {
    (0..logical_per_batch * batch)
        .map(|logical_index| {
            physical[df64_physical_index(layout, logical_per_batch, logical_index)]
        })
        .collect()
}

fn assert_df64_padding_preserved(
    label: &str,
    physical: &[ComplexDoubleFloat],
    layout: FftLogicalLayout,
    logical_per_batch: u64,
    batch: u64,
    sentinel: ComplexDoubleFloat,
) {
    let mut logical_positions = vec![false; physical.len()];
    for logical_index in 0..logical_per_batch * batch {
        logical_positions[df64_physical_index(layout, logical_per_batch, logical_index)] = true;
    }
    for (index, (value, is_logical)) in physical.iter().zip(logical_positions).enumerate() {
        if !is_logical {
            assert_eq!(
                bytemuck::bytes_of(value),
                bytemuck::bytes_of(&sentinel),
                "{label}: output padding word changed at complex index {index}"
            );
        }
    }
}

fn assert_df64_accuracy(label: &str, actual: &[ComplexDoubleFloat], expected: &[Complex64]) {
    assert_eq!(actual.len(), expected.len(), "{label}: output length");
    let mut max_abs = 0.0f64;
    let mut error_energy = 0.0f64;
    let mut reference_energy = 0.0f64;
    let mut reference_max = 0.0f64;
    for (actual, expected) in actual.iter().zip(expected) {
        let error = (actual.re().to_f64() - expected.re).hypot(actual.im().to_f64() - expected.im);
        let magnitude = expected.re.hypot(expected.im);
        max_abs = max_abs.max(error);
        error_energy += error * error;
        reference_energy += magnitude * magnitude;
        reference_max = reference_max.max(magnitude);
    }
    let rms_relative = (error_energy / reference_energy.max(f64::MIN_POSITIVE)).sqrt();
    let max_relative = max_abs / reference_max.max(f64::MIN_POSITIVE);
    eprintln!(
        "DF64_ACCURACY label={label:?} max_relative={max_relative:.9e} rms_relative={rms_relative:.9e}"
    );
    assert!(
        rms_relative.is_finite() && rms_relative <= DF64_RMS_LIMIT,
        "{label}: RMS relative error {rms_relative:.9e} exceeds {DF64_RMS_LIMIT:.1e}"
    );
}

fn sampled_reference_1d(
    input: &[Complex64],
    config: &FftConfig,
    sample_count: usize,
) -> Vec<(usize, Complex64)> {
    assert_eq!(config.shape(), [input.len()]);
    assert_eq!(config.batch(), 1);
    let len = input.len();
    let step = len.div_ceil(sample_count).max(1);
    let sign = if config.direction() == wgpu_fft::FftDirection::Forward {
        -1.0
    } else {
        1.0
    };
    let scale = config.scale_f64().unwrap();
    (0..len)
        .step_by(step)
        .map(|k| {
            let mut sum = Complex64::default();
            for (n, value) in input.iter().enumerate() {
                let exponent = ((k as u128 * n as u128) % len as u128) as f64;
                let angle = sign * std::f64::consts::TAU * exponent / len as f64;
                let (sin, cos) = angle.sin_cos();
                sum.re += value.re * cos - value.im * sin;
                sum.im += value.re * sin + value.im * cos;
            }
            (k, Complex64::new(sum.re * scale, sum.im * scale))
        })
        .collect()
}

fn assert_sampled_df64_accuracy(
    label: &str,
    actual: &[ComplexDoubleFloat],
    expected: &[(usize, Complex64)],
) {
    let actual = expected
        .iter()
        .map(|&(index, _)| actual[index])
        .collect::<Vec<_>>();
    let expected = expected.iter().map(|&(_, value)| value).collect::<Vec<_>>();
    assert_df64_accuracy(label, &actual, &expected);
}

fn assert_stockham_plan(plan: &FftPlan, len: usize, label: &str) {
    let kernels = kernel_labels(plan);
    assert_eq!(kernels.len(), plan.factors().len(), "{label}: N={len}");
    assert!(
        kernels.iter().all(|stage| stage == STOCKHAM_LABEL),
        "{label}: expected Stockham stages, got {kernels:?}"
    );
    assert_eq!(plan.workspace_size_bytes(), (len * 16) as u64);
}

fn kernel_labels(plan: &FftPlan) -> Vec<String> {
    plan.diagnostics()
        .stages()
        .iter()
        .filter(|stage| stage.kind == "kernel")
        .map(|stage| stage.label.clone())
        .collect()
}
