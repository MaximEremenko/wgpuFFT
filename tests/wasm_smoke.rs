#![cfg(target_arch = "wasm32")]

use futures_channel::oneshot;
use wasm_bindgen_test::*;
use wgpu_fft::{FftConfig, FftPlan, Normalization};

wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test(async)]
async fn browser_webgpu_executes_small_c2c() {
    let context = wgpu_fft::device::request_default_device()
        .await
        .expect("browser must expose a WebGPU adapter");
    let info = context.adapter.get_info();
    assert_eq!(info.backend, wgpu::Backend::BrowserWebGpu);

    let input = [1.0f32, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0];
    let expected = [10.0f32, 0.0, -2.0, 2.0, -2.0, 0.0, -2.0, -2.0];
    let byte_len = std::mem::size_of_val(&input) as u64;
    let input_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.wasm_smoke.input"),
        size: byte_len,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    context
        .queue
        .write_buffer(&input_buffer, 0, bytemuck::cast_slice(&input));
    let output_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.wasm_smoke.output"),
        size: byte_len,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.wasm_smoke.readback"),
        size: byte_len,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let plan = FftPlan::c2c_checked(
        &context.device,
        &context.queue,
        FftConfig::new(4).with_normalization(Normalization::None),
    )
    .await
    .expect("small browser C2C plan");
    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("wgpu_fft.wasm_smoke.encoder"),
        });
    plan.execute_checked(&context.device, &mut encoder, &input_buffer, &output_buffer)
        .expect("encode browser C2C");
    encoder.copy_buffer_to_buffer(&output_buffer, 0, &readback, 0, byte_len);
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
        .expect("browser readback mapping must succeed");

    let mapped = readback.slice(..).get_mapped_range();
    let actual = bytemuck::cast_slice::<u8, f32>(&mapped);
    for (index, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
        assert!(
            (actual - expected).abs() <= 1.0e-5,
            "element {index}: actual={actual}, expected={expected}"
        );
    }
}
