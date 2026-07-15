#![cfg(feature = "gpu-profiling")]
#![cfg(not(target_arch = "wasm32"))]

//! Opt-in validation for the timestamped NUFFT diagnostic paths.

use std::f32::consts::{PI, TAU};
use std::sync::mpsc;

use wgpu::util::DeviceExt;
use wgpu_nufft::{
    ModeOrder, NufftConfig, NufftGpuProfileLayout, NufftGpuStage, NufftPlan, NufftSign,
};

const MODE_COUNT: usize = 256;
const POINT_COUNT: usize = 1_024;
const MODE_SHAPE_2D: [usize; 2] = [32, 48];
const MODE_COUNT_2D: usize = MODE_SHAPE_2D[0] * MODE_SHAPE_2D[1];
const POINT_COUNT_2D: usize = 1_021;

const TYPE1_STAGES: &[NufftGpuStage] = &[
    NufftGpuStage::BinClearCount,
    NufftGpuStage::ScanAndTerminal,
    NufftGpuStage::Scatter,
    NufftGpuStage::Sort,
    NufftGpuStage::GatherSpread,
    NufftGpuStage::FineGridFft,
    NufftGpuStage::Deconvolution,
];
const TYPE1_LABELS: &[&str] = &[
    "bin-clear-count",
    "scan-and-terminal",
    "scatter",
    "sort",
    "gather-spread",
    "fine-grid-fft",
    "deconvolution",
];
const TYPE2_STAGES: &[NufftGpuStage] = &[
    NufftGpuStage::Predeconvolution,
    NufftGpuStage::FineGridFft,
    NufftGpuStage::Interpolation,
];
const TYPE2_LABELS: &[&str] = &["predeconvolution", "fine-grid-fft", "interpolation"];

#[test]
fn profiled_gpu_paths_match_ordinary_execution_and_write_timestamps() {
    if std::env::var("WGPU_FFT_RUN_GPU_TESTS").ok().as_deref() != Some("1") {
        eprintln!("skipping GPU stage-profile test; set WGPU_FFT_RUN_GPU_TESTS=1 to run it");
        return;
    }
    pollster::block_on(run_gpu_stage_profile_test());
}

async fn run_gpu_stage_profile_test() {
    let mut instance_descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
    instance_descriptor.backends = wgpu::Backends::VULKAN;
    let instance = wgpu::Instance::new(instance_descriptor);
    let adapters = instance
        .enumerate_adapters(wgpu::Backends::VULKAN)
        .await
        .into_iter()
        .collect::<Vec<_>>();
    let adapter = adapters
        .into_iter()
        .find(|adapter| adapter.get_info().device_type == wgpu::DeviceType::DiscreteGpu)
        .unwrap_or_else(|| {
            panic!("WGPU_FFT_RUN_GPU_TESTS=1 but no discrete Vulkan adapter was found")
        });
    let info = adapter.get_info();
    assert_eq!(
        info.backend,
        wgpu::Backend::Vulkan,
        "stage profiling must run through Vulkan"
    );
    eprintln!(
        "gpu_stage_profile adapter: {} backend={:?} driver={} {}",
        info.name, info.backend, info.driver, info.driver_info
    );

    if !adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
        eprintln!(
            "skipping GPU stage-profile test: adapter {} backend={:?} does not support TIMESTAMP_QUERY",
            info.name, info.backend
        );
        #[cfg(windows)]
        std::mem::forget((adapter, instance));
        return;
    }

    let adapter_limits = adapter.limits();
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("wgpu_nufft.stage_profile.test.device"),
            required_features: wgpu::Features::TIMESTAMP_QUERY,
            required_limits: adapter_limits,
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
        })
        .await
        .expect("requesting a Vulkan device with TIMESTAMP_QUERY must succeed");
    assert!(
        device.features().contains(wgpu::Features::TIMESTAMP_QUERY),
        "the requested timestamp-query feature must be enabled on the device"
    );
    let period_ns = f64::from(queue.get_timestamp_period());
    assert!(
        period_ns.is_finite() && period_ns > 0.0,
        "timestamp period must be finite and positive, got {period_ns}"
    );
    eprintln!("gpu_stage_profile timestamp_period_ns={period_ns}");

    let validation_scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let points = test_points();
    run_type1_case(&device, &queue, &points, period_ns);
    run_type2_case(&device, &queue, &points, period_ns);
    let points_2d = test_points_2d();
    run_type1_case_2d(&device, &queue, &points_2d, period_ns);
    run_type2_case_2d(&device, &queue, &points_2d, period_ns);
    if let Some(error) = validation_scope.pop().await {
        panic!("GPU stage-profile validation scope captured an unexpected error: {error}");
    }

    // Native wgpu teardown can stall on Windows after compute-heavy tests.
    #[cfg(windows)]
    std::mem::forget((queue, device, adapter, instance));
}

fn run_type1_case(device: &wgpu::Device, queue: &wgpu::Queue, points: &[f32], period_ns: f64) {
    let config = test_config();
    let plan = NufftPlan::type1_gpu(device, queue, config).unwrap();
    assert_eq!(plan.gpu_profile_query_count(), 8);
    let strengths = test_complex_values(points.len(), 0.19, 0.31);
    let point_buffer = create_storage_buffer(device, "type1.points", points);
    let strength_buffer = create_storage_buffer(device, "type1.strengths", &strengths);
    let output_bytes = plan.required_type1_output_buffer_size_bytes().unwrap();
    let ordinary_output = create_output_buffer(device, "type1.ordinary_output", output_bytes);
    let profiled_output = create_output_buffer(device, "type1.profiled_output", output_bytes);
    let query_count = plan.gpu_profile_query_count();
    let query_set = create_timestamp_query_set(device, "type1", query_count);

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("wgpu_nufft.stage_profile.type1.encoder"),
    });
    plan.encode_type1_gpu(
        device,
        &mut encoder,
        points.len(),
        &point_buffer,
        &strength_buffer,
        &ordinary_output,
    )
    .unwrap();
    let layout = plan
        .encode_type1_gpu_profiled(
            device,
            &mut encoder,
            points.len(),
            &point_buffer,
            &strength_buffer,
            &profiled_output,
            &query_set,
            0,
        )
        .unwrap();
    assert_eq!(layout.query_count(), query_count);
    finish_case(
        device,
        queue,
        encoder,
        "type-1",
        &ordinary_output,
        &profiled_output,
        output_bytes,
        &query_set,
        &layout,
        TYPE1_STAGES,
        TYPE1_LABELS,
        period_ns,
    );
}

fn run_type2_case(device: &wgpu::Device, queue: &wgpu::Queue, points: &[f32], period_ns: f64) {
    let config = test_config();
    let plan = NufftPlan::type2_gpu(device, queue, config).unwrap();
    assert_eq!(plan.gpu_profile_query_count(), 4);
    let coefficients = test_complex_values(MODE_COUNT, 0.23, 0.37);
    let point_buffer = create_storage_buffer(device, "type2.points", points);
    let coefficient_buffer = create_storage_buffer(device, "type2.coefficients", &coefficients);
    let output_bytes = NufftPlan::required_type2_output_buffer_size_bytes(points.len()).unwrap();
    let ordinary_output = create_output_buffer(device, "type2.ordinary_output", output_bytes);
    let profiled_output = create_output_buffer(device, "type2.profiled_output", output_bytes);
    let query_count = plan.gpu_profile_query_count();
    let query_set = create_timestamp_query_set(device, "type2", query_count);

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("wgpu_nufft.stage_profile.type2.encoder"),
    });
    plan.encode_type2_gpu(
        device,
        &mut encoder,
        points.len(),
        &point_buffer,
        &coefficient_buffer,
        &ordinary_output,
    )
    .unwrap();
    let layout = plan
        .encode_type2_gpu_profiled(
            device,
            &mut encoder,
            points.len(),
            &point_buffer,
            &coefficient_buffer,
            &profiled_output,
            &query_set,
            0,
        )
        .unwrap();
    assert_eq!(layout.query_count(), query_count);
    finish_case(
        device,
        queue,
        encoder,
        "type-2",
        &ordinary_output,
        &profiled_output,
        output_bytes,
        &query_set,
        &layout,
        TYPE2_STAGES,
        TYPE2_LABELS,
        period_ns,
    );
}

fn run_type1_case_2d(device: &wgpu::Device, queue: &wgpu::Queue, points: &[f32], period_ns: f64) {
    const FIRST_QUERY: u32 = 3;

    assert_eq!(points.len(), POINT_COUNT_2D * MODE_SHAPE_2D.len());
    let config = test_config_2d();
    #[cfg(feature = "type1-2d-tile-prototype")]
    let plan =
        NufftPlan::type1_gpu_with_tiled_2d_gather_for_testing(device, queue, config).unwrap();
    #[cfg(not(feature = "type1-2d-tile-prototype"))]
    let plan = NufftPlan::type1_gpu(device, queue, config).unwrap();
    assert_eq!(plan.gpu_profile_query_count(), 8);
    let strengths = test_complex_values(POINT_COUNT_2D, 0.17, 0.29);
    let point_buffer = create_storage_buffer(device, "type1_2d.points", points);
    let strength_buffer = create_storage_buffer(device, "type1_2d.strengths", &strengths);
    let output_bytes = plan.required_type1_output_buffer_size_bytes().unwrap();
    let ordinary_output = create_output_buffer(device, "type1_2d.ordinary_output", output_bytes);
    let profiled_output = create_output_buffer(device, "type1_2d.profiled_output", output_bytes);
    let query_count = plan.gpu_profile_query_count();
    let query_set = create_timestamp_query_set(
        device,
        "type1_2d",
        FIRST_QUERY.checked_add(query_count).unwrap(),
    );

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("wgpu_nufft.stage_profile.type1_2d.encoder"),
    });
    plan.encode_type1_gpu(
        device,
        &mut encoder,
        POINT_COUNT_2D,
        &point_buffer,
        &strength_buffer,
        &ordinary_output,
    )
    .unwrap();
    let layout = plan
        .encode_type1_gpu_profiled(
            device,
            &mut encoder,
            POINT_COUNT_2D,
            &point_buffer,
            &strength_buffer,
            &profiled_output,
            &query_set,
            FIRST_QUERY,
        )
        .unwrap();
    assert_eq!(layout.query_count(), query_count);
    assert_eq!(layout.query_range(), FIRST_QUERY..FIRST_QUERY + query_count);
    finish_case(
        device,
        queue,
        encoder,
        "type-1-2d",
        &ordinary_output,
        &profiled_output,
        output_bytes,
        &query_set,
        &layout,
        TYPE1_STAGES,
        TYPE1_LABELS,
        period_ns,
    );
}

fn run_type2_case_2d(device: &wgpu::Device, queue: &wgpu::Queue, points: &[f32], period_ns: f64) {
    const FIRST_QUERY: u32 = 5;

    assert_eq!(points.len(), POINT_COUNT_2D * MODE_SHAPE_2D.len());
    let config = test_config_2d();
    let plan = NufftPlan::type2_gpu(device, queue, config).unwrap();
    assert_eq!(plan.gpu_profile_query_count(), 4);
    let coefficients = test_complex_values(MODE_COUNT_2D, 0.21, 0.35);
    let point_buffer = create_storage_buffer(device, "type2_2d.points", points);
    let coefficient_buffer = create_storage_buffer(device, "type2_2d.coefficients", &coefficients);
    let output_bytes = NufftPlan::required_type2_output_buffer_size_bytes(POINT_COUNT_2D).unwrap();
    let ordinary_output = create_output_buffer(device, "type2_2d.ordinary_output", output_bytes);
    let profiled_output = create_output_buffer(device, "type2_2d.profiled_output", output_bytes);
    let query_count = plan.gpu_profile_query_count();
    let query_set = create_timestamp_query_set(
        device,
        "type2_2d",
        FIRST_QUERY.checked_add(query_count).unwrap(),
    );

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("wgpu_nufft.stage_profile.type2_2d.encoder"),
    });
    plan.encode_type2_gpu(
        device,
        &mut encoder,
        POINT_COUNT_2D,
        &point_buffer,
        &coefficient_buffer,
        &ordinary_output,
    )
    .unwrap();
    let layout = plan
        .encode_type2_gpu_profiled(
            device,
            &mut encoder,
            POINT_COUNT_2D,
            &point_buffer,
            &coefficient_buffer,
            &profiled_output,
            &query_set,
            FIRST_QUERY,
        )
        .unwrap();
    assert_eq!(layout.query_count(), query_count);
    assert_eq!(layout.query_range(), FIRST_QUERY..FIRST_QUERY + query_count);
    finish_case(
        device,
        queue,
        encoder,
        "type-2-2d",
        &ordinary_output,
        &profiled_output,
        output_bytes,
        &query_set,
        &layout,
        TYPE2_STAGES,
        TYPE2_LABELS,
        period_ns,
    );
}

#[allow(clippy::too_many_arguments)]
fn finish_case(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    mut encoder: wgpu::CommandEncoder,
    kind: &str,
    ordinary_output: &wgpu::Buffer,
    profiled_output: &wgpu::Buffer,
    output_bytes: u64,
    query_set: &wgpu::QuerySet,
    layout: &NufftGpuProfileLayout,
    expected_stages: &[NufftGpuStage],
    expected_labels: &[&str],
    period_ns: f64,
) {
    let first_query = layout.query_range().start;
    assert_eq!(
        layout.query_range(),
        first_query..first_query + layout.query_count()
    );
    let stages = layout
        .stages()
        .iter()
        .map(|query| query.stage())
        .collect::<Vec<_>>();
    let labels = layout
        .stages()
        .iter()
        .map(|query| query.stage().label())
        .collect::<Vec<_>>();
    assert_eq!(stages, expected_stages);
    assert_eq!(labels, expected_labels);

    let output_readback_bytes = output_bytes.checked_mul(2).unwrap();
    let output_readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_nufft.stage_profile.output_readback"),
        size: output_readback_bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    encoder.copy_buffer_to_buffer(ordinary_output, 0, &output_readback, 0, output_bytes);
    encoder.copy_buffer_to_buffer(
        profiled_output,
        0,
        &output_readback,
        output_bytes,
        output_bytes,
    );

    let timestamp_bytes = u64::from(layout.query_count()) * u64::from(wgpu::QUERY_SIZE);
    let timestamp_resolve = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_nufft.stage_profile.timestamp_resolve"),
        size: timestamp_bytes,
        usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let timestamp_readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_nufft.stage_profile.timestamp_readback"),
        size: timestamp_bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    encoder.resolve_query_set(query_set, layout.query_range(), &timestamp_resolve, 0);
    encoder.copy_buffer_to_buffer(
        &timestamp_resolve,
        0,
        &timestamp_readback,
        0,
        timestamp_bytes,
    );
    queue.submit([encoder.finish()]);

    let output = read_buffer(device, &output_readback, output_readback_bytes);
    let output_word_count = usize::try_from(output_bytes / 4).unwrap();
    let output_words = output
        .chunks_exact(4)
        .map(|bytes| u32::from_le_bytes(bytes.try_into().unwrap()))
        .collect::<Vec<_>>();
    assert_eq!(output_words.len(), output_word_count * 2);
    assert_eq!(
        &output_words[..output_word_count],
        &output_words[output_word_count..],
        "{kind} profiled execution must be bit-identical to ordinary execution"
    );

    let timestamp_data = read_buffer(device, &timestamp_readback, timestamp_bytes);
    let timestamps = timestamp_data
        .chunks_exact(8)
        .map(|bytes| u64::from_le_bytes(bytes.try_into().unwrap()))
        .collect::<Vec<_>>();
    assert_eq!(
        timestamps.len(),
        usize::try_from(layout.query_count()).unwrap()
    );
    for stage in layout.stages() {
        let start_index = usize::try_from(stage.start_query() - first_query).unwrap();
        let end_index = usize::try_from(stage.end_query() - first_query).unwrap();
        let start = timestamps[start_index];
        let end = timestamps[end_index];
        assert!(
            end >= start,
            "{kind} stage {} timestamp went backwards: {start} -> {end}",
            stage.stage()
        );
        let delta = end - start;
        assert!(
            delta > 0,
            "{kind} stage {} must have a positive timestamp delta: {start} -> {end}; resolved timestamps={timestamps:?}",
            stage.stage(),
        );
        eprintln!(
            "GPU_STAGE_PROFILE kind={kind} stage={} ticks={delta} ms={:.9}",
            stage.stage(),
            delta as f64 * period_ns / 1.0e6
        );
    }
}

fn test_config() -> NufftConfig {
    NufftConfig::new([MODE_COUNT], 1.0e-6)
        .with_sign(NufftSign::Positive)
        .with_mode_order(ModeOrder::Centered)
}

fn test_config_2d() -> NufftConfig {
    NufftConfig::new(MODE_SHAPE_2D, 1.0e-6)
        .with_sign(NufftSign::Positive)
        .with_mode_order(ModeOrder::Centered)
}

fn test_points() -> Vec<f32> {
    (0..POINT_COUNT)
        .map(|index| {
            let fraction = ((index * 37) % 1_021) as f32 / 1_021.0;
            fraction * TAU - PI
        })
        .collect()
}

fn test_points_2d() -> Vec<f32> {
    let mut points = Vec::with_capacity(POINT_COUNT_2D * MODE_SHAPE_2D.len());
    for index in 0..POINT_COUNT_2D {
        let x_fraction = ((index * 37) % 1_019) as f32 / 1_019.0;
        let y_fraction = ((index * 53 + 7) % 1_021) as f32 / 1_021.0;
        points.push(x_fraction * TAU - PI);
        points.push(y_fraction * TAU - PI);
    }
    points
}

fn test_complex_values(count: usize, real_scale: f32, imaginary_scale: f32) -> Vec<f32> {
    (0..count)
        .flat_map(|index| {
            let x = index as f32 + 1.0;
            [
                (x * real_scale).sin() * 0.75,
                (x * imaginary_scale).cos() * 0.5,
            ]
        })
        .collect()
}

fn create_storage_buffer(
    device: &wgpu::Device,
    label_suffix: &str,
    contents: &[f32],
) -> wgpu::Buffer {
    let label = format!("wgpu_nufft.stage_profile.{label_suffix}");
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(&label),
        contents: bytemuck::cast_slice(contents),
        usage: wgpu::BufferUsages::STORAGE,
    })
}

fn create_output_buffer(device: &wgpu::Device, label_suffix: &str, size: u64) -> wgpu::Buffer {
    let label = format!("wgpu_nufft.stage_profile.{label_suffix}");
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(&label),
        size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}

fn create_timestamp_query_set(
    device: &wgpu::Device,
    label_suffix: &str,
    count: u32,
) -> wgpu::QuerySet {
    let label = format!("wgpu_nufft.stage_profile.{label_suffix}.queries");
    device.create_query_set(&wgpu::QuerySetDescriptor {
        label: Some(&label),
        ty: wgpu::QueryType::Timestamp,
        count,
    })
}

fn read_buffer(device: &wgpu::Device, buffer: &wgpu::Buffer, size: u64) -> Vec<u8> {
    let slice = buffer.slice(0..size);
    let (sender, receiver) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        sender.send(result).unwrap();
    });
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    receiver.recv().unwrap().unwrap();
    let mapped = slice.get_mapped_range();
    let result = mapped.to_vec();
    drop(mapped);
    buffer.unmap();
    result
}
