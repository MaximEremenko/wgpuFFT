#![cfg(target_arch = "wasm32")]

use futures_channel::oneshot;
use wasm_bindgen_test::*;
use wgpu_fft::{
    BufferSegment, BufferView, FftConfig, FftPlan, LargeExecutionKind, LargeRouteMode,
    Normalization,
};

wasm_bindgen_test_configure!(run_in_browser);

const COMPLEX_BYTES: u64 = 8;
const FOUR_STEP_SHAPE: [usize; 2] = [4_096, 4_116];
const SEGMENTED_SHAPE: [usize; 2] = [4_096, 8_232];

struct BrowserContext {
    instance: wgpu::Instance,
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
}

#[wasm_bindgen_test(async)]
async fn browser_default_limits_execute_natural_large_routes() {
    if option_env!("WGPU_FFT_RUN_BROWSER_LARGE_TESTS").is_none() {
        console_log!("skipping ~1 GiB browser large-route test; run web\\run_browser_tests.cmd");
        return;
    }
    let context = request_browser_default_device().await;
    let info = context.adapter.get_info();
    assert_eq!(info.backend, wgpu::Backend::BrowserWebGpu);

    let defaults = wgpu::Limits::default();
    let limits = context.device.limits();
    assert_eq!(
        limits.max_storage_buffer_binding_size, defaults.max_storage_buffer_binding_size,
        "test device must retain the WebGPU default storage-binding cap"
    );
    assert_eq!(
        limits.max_buffer_size, defaults.max_buffer_size,
        "test device must retain the WebGPU default per-buffer cap"
    );
    assert_eq!(
        limits.max_compute_workgroup_storage_size, defaults.max_compute_workgroup_storage_size,
        "test device must retain the WebGPU default workgroup-storage cap"
    );

    run_four_step_impulse_case(&context).await;
    run_segmented_impulse_case(&context).await;

    console_log!(
        "browser large routes passed: four_step_shape={:?} segmented_shape={:?}",
        FOUR_STEP_SHAPE,
        SEGMENTED_SHAPE
    );

    // Keep the instance alive through both asynchronous readbacks.
    let _ = &context.instance;
}

async fn request_browser_default_device() -> BrowserContext {
    let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
    descriptor.backends = wgpu::Backends::BROWSER_WEBGPU;
    let instance = wgpu::Instance::new(descriptor);
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            apply_limit_buckets: false,
            compatible_surface: None,
        })
        .await
        .expect("browser must expose a WebGPU adapter");
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("wgpu_fft.browser_default_limits.device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: wgpu::MemoryHints::MemoryUsage,
            trace: wgpu::Trace::Off,
        })
        .await
        .expect("browser default-limit device request must succeed");
    BrowserContext {
        instance,
        adapter,
        device,
        queue,
    }
}

async fn run_four_step_impulse_case(context: &BrowserContext) {
    let config = FftConfig::new_nd(FOUR_STEP_SHAPE).with_normalization(Normalization::None);
    let byte_len = config.required_buffer_size_bytes().unwrap();
    let limits = context.device.limits();
    assert!(byte_len > limits.max_storage_buffer_binding_size);
    assert!(byte_len <= limits.max_buffer_size);

    let plan = FftPlan::c2c_checked(&context.device, &context.queue, config)
        .await
        .expect("natural browser-default four-step plan");
    assert_eq!(
        plan.large_routing_policy().route_mode(),
        LargeRouteMode::LargeOutOfCore
    );
    assert_eq!(
        plan.large_routing_policy().execution_kind(),
        LargeExecutionKind::OutOfCoreFourStep
    );
    assert_eq!(
        plan.large_routing_policy().max_bind_bytes,
        limits.max_storage_buffer_binding_size
    );

    let usages = endpoint_usages();
    let input = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.browser.four_step.input"),
        size: byte_len,
        usage: usages,
        mapped_at_creation: false,
    });
    let output = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.browser.four_step.output"),
        size: byte_len,
        usage: usages,
        mapped_at_creation: false,
    });
    clear_buffers(context, [&input], "wgpu_fft.browser.four_step.clear");
    context
        .queue
        .write_buffer(&input, 0, bytemuck::cast_slice(&[1.0f32, 0.0]));

    let sample_indices = [
        0u64,
        1,
        FOUR_STEP_SHAPE[0] as u64 - 1,
        FOUR_STEP_SHAPE[0] as u64,
        byte_len / COMPLEX_BYTES / 2,
        byte_len / COMPLEX_BYTES - 1,
    ];
    let actual = execute_and_read_samples(
        context,
        &plan,
        BufferView::whole(&input),
        BufferView::whole(&output),
        &sample_indices,
        "wgpu_fft.browser.four_step",
    )
    .await;
    assert_unit_impulse_samples(&actual, &sample_indices, "four-step");
}

async fn run_segmented_impulse_case(context: &BrowserContext) {
    let config = FftConfig::new_nd(SEGMENTED_SHAPE).with_normalization(Normalization::None);
    let byte_len = config.required_buffer_size_bytes().unwrap();
    let limits = context.device.limits();
    assert!(byte_len > limits.max_buffer_size);

    let plan = FftPlan::c2c_checked(&context.device, &context.queue, config)
        .await
        .expect("natural browser-default segmented plan");
    assert_eq!(
        plan.large_routing_policy().route_mode(),
        LargeRouteMode::LargeOutOfCore
    );
    assert_eq!(
        plan.large_routing_policy().execution_kind(),
        LargeExecutionKind::SegmentedFullVolume
    );
    assert_eq!(
        plan.large_routing_policy().max_buffer_size,
        limits.max_buffer_size
    );

    let input_buffers = create_segment_buffers(
        &context.device,
        byte_len,
        limits.max_storage_buffer_binding_size,
        "wgpu_fft.browser.segmented.input",
    );
    let output_buffers = create_segment_buffers(
        &context.device,
        byte_len,
        limits.max_storage_buffer_binding_size,
        "wgpu_fft.browser.segmented.output",
    );
    clear_buffers(
        context,
        input_buffers.iter(),
        "wgpu_fft.browser.segmented.clear",
    );
    context
        .queue
        .write_buffer(&input_buffers[0], 0, bytemuck::cast_slice(&[1.0f32, 0.0]));

    let input_view = whole_segmented_view(&input_buffers, byte_len);
    let output_view = whole_segmented_view(&output_buffers, byte_len);
    let view_diagnostics =
        plan.diagnostics_for_views(&context.device, input_view.clone(), output_view.clone());
    assert!(
        view_diagnostics.blockers().is_empty(),
        "whole physical-buffer segments must be valid: {:?}",
        view_diagnostics.blockers()
    );

    let boundary_element = limits.max_storage_buffer_binding_size / COMPLEX_BYTES;
    let sample_indices = [
        0u64,
        1,
        SEGMENTED_SHAPE[0] as u64 - 1,
        SEGMENTED_SHAPE[0] as u64,
        boundary_element - 1,
        boundary_element,
        2 * boundary_element - 1,
        2 * boundary_element,
        byte_len / COMPLEX_BYTES - 1,
    ];
    let actual = execute_and_read_samples(
        context,
        &plan,
        input_view,
        output_view,
        &sample_indices,
        "wgpu_fft.browser.segmented",
    )
    .await;
    assert_unit_impulse_samples(&actual, &sample_indices, "segmented");
}

fn endpoint_usages() -> wgpu::BufferUsages {
    wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST
}

fn create_segment_buffers(
    device: &wgpu::Device,
    total_bytes: u64,
    segment_cap: u64,
    label: &'static str,
) -> Vec<wgpu::Buffer> {
    let mut buffers = Vec::new();
    let mut remaining = total_bytes;
    while remaining > 0 {
        let size = remaining.min(segment_cap);
        buffers.push(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage: endpoint_usages(),
            mapped_at_creation: false,
        }));
        remaining -= size;
    }
    buffers
}

fn whole_segmented_view<'a>(buffers: &'a [wgpu::Buffer], byte_len: u64) -> BufferView<'a> {
    let segments = buffers
        .iter()
        .map(|buffer| BufferSegment::new(buffer, 0, buffer.size()))
        .collect::<Vec<_>>();
    BufferView::from_segments(&segments, 0, byte_len).unwrap()
}

fn clear_buffers<'a>(
    context: &BrowserContext,
    buffers: impl IntoIterator<Item = &'a wgpu::Buffer>,
    label: &'static str,
) {
    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) });
    for buffer in buffers {
        encoder.clear_buffer(buffer, 0, None);
    }
    context.queue.submit([encoder.finish()]);
}

async fn execute_and_read_samples(
    context: &BrowserContext,
    plan: &FftPlan,
    input: BufferView<'_>,
    output: BufferView<'_>,
    sample_indices: &[u64],
    label: &'static str,
) -> Vec<f32> {
    let readback = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: sample_indices.len() as u64 * COMPLEX_BYTES,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) });
    plan.execute_views(&context.device, &mut encoder, input, output.clone())
        .expect("encode natural large route");
    for (sample, &index) in sample_indices.iter().enumerate() {
        let ranges = output.ranges(index * COMPLEX_BYTES, COMPLEX_BYTES).unwrap();
        assert_eq!(ranges.len(), 1, "complex samples must not cross a segment");
        encoder.copy_buffer_to_buffer(
            ranges[0].buffer,
            ranges[0].offset_bytes,
            &readback,
            sample as u64 * COMPLEX_BYTES,
            COMPLEX_BYTES,
        );
    }
    context.queue.submit([encoder.finish()]);

    let (sender, receiver) = oneshot::channel();
    readback
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
    receiver
        .await
        .expect("browser map callback must run")
        .expect("browser sample readback must map");
    let mapped = readback
        .slice(..)
        .get_mapped_range()
        .expect("browser readback range must be mapped");
    bytemuck::cast_slice::<u8, f32>(&mapped).to_vec()
}

fn assert_unit_impulse_samples(actual: &[f32], sample_indices: &[u64], route: &str) {
    for (sample, (&index, pair)) in sample_indices
        .iter()
        .zip(actual.as_chunks::<2>().0)
        .enumerate()
    {
        let error = (pair[0] - 1.0).hypot(pair[1]);
        assert!(
            error.is_finite() && error <= 2.0e-3,
            "{route} sample {sample} at logical index {index}: actual=({}, {}), expected=(1, 0), error={error}",
            pair[0],
            pair[1]
        );
    }
}
