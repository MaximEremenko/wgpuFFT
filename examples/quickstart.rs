//! Forward FFT of an impulse on the default GPU, falling back to the CPU
//! backend when no adapter is available.
//!
//! Run with `cargo run --example quickstart`.
#![cfg_attr(target_arch = "wasm32", allow(dead_code, unused_imports))]

use wgpu_fft::{FftConfig, FftPlan};

#[cfg(target_arch = "wasm32")]
fn main() {}

#[cfg(not(target_arch = "wasm32"))]
fn main() {
    let n = 8;
    // Interleaved complex f32 values: [re0, im0, re1, im1, ...].
    let mut input = vec![0.0f32; 2 * n];
    input[2] = 1.0; // impulse at index 1

    let output = match pollster::block_on(wgpu_fft::device::request_default_device()) {
        Some(gpu) => {
            println!("running on {}", gpu.adapter.get_info().name);
            run_on_gpu(&gpu, n, &input)
        }
        None => {
            println!("no GPU adapter found; running on the CPU");
            run_on_cpu(n, &input)
        }
    };

    // The DFT of an impulse at index 1 is exp(-2*pi*i*k/n).
    for (k, value) in output.chunks_exact(2).enumerate() {
        println!("X[{k}] = {:+.6} {:+.6}i", value[0], value[1]);
    }
}

fn run_on_gpu(gpu: &wgpu_fft::device::GpuContext, n: usize, input: &[f32]) -> Vec<f32> {
    let plan = FftPlan::c2c(&gpu.device, &gpu.queue, FftConfig::new(n)).expect("create plan");
    let size = plan.required_buffer_size_bytes();
    let buffer = |label: &str, usage: wgpu::BufferUsages| {
        gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage,
            mapped_at_creation: false,
        })
    };
    let source = buffer(
        "input",
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    );
    let spectrum = buffer(
        "output",
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    );
    let readback = buffer(
        "readback",
        wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
    );
    gpu.queue
        .write_buffer(&source, 0, bytemuck::cast_slice(input));

    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    plan.execute_checked(&gpu.device, &mut encoder, &source, &spectrum)
        .expect("buffers match the plan");
    encoder.copy_buffer_to_buffer(&spectrum, 0, &readback, 0, size);
    gpu.queue.submit([encoder.finish()]);

    let slice = readback.slice(..);
    slice.map_async(wgpu::MapMode::Read, |result| result.expect("map readback"));
    gpu.device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("wait for the GPU");
    let output = bytemuck::cast_slice(&slice.get_mapped_range().expect("mapped range")).to_vec();
    readback.unmap();
    output
}

#[cfg(feature = "cpu")]
fn run_on_cpu(n: usize, input: &[f32]) -> Vec<f32> {
    let plan = wgpu_fft::CpuFftPlan::c2c(FftConfig::new(n)).expect("create plan");
    let mut output = vec![0.0f32; plan.required_output_len()];
    plan.execute(input, &mut output)
        .expect("slices match the plan");
    output
}

#[cfg(not(feature = "cpu"))]
fn run_on_cpu(_n: usize, _input: &[f32]) -> Vec<f32> {
    panic!("no GPU adapter found; enable the `cpu` feature for the CPU fallback");
}
