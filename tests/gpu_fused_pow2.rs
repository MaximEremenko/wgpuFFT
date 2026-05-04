//! Focused correctness and routing coverage for the single-workgroup power-of-two kernel.

use std::f64::consts::PI;
use std::sync::mpsc;

use wgpu_fft::math::{from_interleaved_f32, reference_c2c_nd, to_interleaved_f32};
use wgpu_fft::{
    export_pipeline_cache_snapshot, import_pipeline_cache_snapshot, C2cRoute, FftConfig, FftPlan,
    Normalization,
};

const FUSED_LABEL: &str = "fused-pow2-workgroup-stage";
const STOCKHAM_LABEL: &str = "mixed-radix-stockham-stage";

#[test]
fn fused_pow2_gpu_matches_cpu_and_multipass() {
    if std::env::var_os("WGPU_FFT_RUN_GPU_TESTS").is_none() {
        eprintln!("skipping GPU test; set WGPU_FFT_RUN_GPU_TESTS=1 to run it");
        return;
    }

    pollster::block_on(run_fused_cases());
}

async fn run_fused_cases() {
    let Some(context) = wgpu_fft::device::request_default_device().await else {
        eprintln!("skipping GPU test; no suitable wgpu adapter was found");
        return;
    };
    eprintln!("adapter: {:?}", context.adapter.get_info());

    let storage_limit = context.device.limits().max_compute_workgroup_storage_size as usize;
    for length in [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096] {
        if length * 8 > storage_limit {
            eprintln!(
                "skipping fused N={length}: {}-byte line exceeds {}-byte workgroup storage limit",
                length * 8,
                storage_limit
            );
            continue;
        }
        for inverse in [false, true] {
            let config = if inverse {
                FftConfig::inverse(length)
            } else {
                FftConfig::new(length).with_normalization(Normalization::None)
            };
            let input = test_signal(length);
            let expected = cpu_fft_pow2(&input, inverse, inverse);
            let (actual, plan) = execute_c2c(&context.device, &context.queue, config, &input);
            assert_fused_plan(&plan, 1, 0);
            assert_close(&actual, &expected, &format!("N={length} inverse={inverse}"));
        }
    }

    for (config, expected_fused, expected_stockham) in [
        (
            FftConfig::new(256)
                .with_batch(3)
                .with_normalization(Normalization::None),
            1,
            0,
        ),
        (FftConfig::inverse(256).with_batch(3), 1, 0),
        (
            FftConfig::new_nd([8, 16])
                .with_batch(2)
                .with_normalization(Normalization::None),
            2,
            0,
        ),
        (FftConfig::inverse_nd([8, 16]).with_batch(2), 2, 0),
        (
            FftConfig::new_nd([3, 256, 5])
                .with_axes([1])
                .with_batch(2)
                .with_normalization(Normalization::None),
            1,
            0,
        ),
        (
            FftConfig::inverse_nd([3, 256, 5])
                .with_axes([1])
                .with_batch(2),
            1,
            0,
        ),
        (
            FftConfig::new_nd([256, 12]).with_normalization(Normalization::None),
            1,
            2,
        ),
        (FftConfig::inverse_nd([256, 12]), 1, 2),
    ] {
        let input = test_signal(config.logical_complex_len().unwrap() * config.batch());
        let expected = reference_c2c_nd(&from_interleaved_f32(&input), &config).unwrap();
        let expected = to_interleaved_f32(&expected);
        let (actual, plan) = execute_c2c(&context.device, &context.queue, config.clone(), &input);
        assert_stage_labels(&plan, expected_fused, expected_stockham);
        assert_close(&actual, &expected, &format!("batched/ND {config:?}"));
    }

    if storage_limit >= 4096 * 8 {
        compare_fused_and_multipass_4096(&context).await;
    } else {
        eprintln!("skipping N=4096 fused/multipass comparison: fused route is unavailable");
    }

    #[cfg(windows)]
    {
        std::mem::forget(context);
    }
}

async fn compare_fused_and_multipass_4096(context: &wgpu_fft::device::GpuContext) {
    let mut low_limits = context.adapter.limits();
    low_limits.max_compute_workgroup_storage_size = 16 * 1024;
    let (low_device, low_queue) = context
        .adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("wgpu_fft.test.low_workgroup_storage_device"),
            required_features: wgpu::Features::empty(),
            required_limits: low_limits,
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: wgpu::MemoryHints::MemoryUsage,
            trace: wgpu::Trace::Off,
        })
        .await
        .expect("the adapter should allow a second device with a 16 KiB workgroup-storage limit");
    assert_eq!(
        low_device.limits().max_compute_workgroup_storage_size,
        16 * 1024
    );

    let full_snapshot = export_pipeline_cache_snapshot(&context.device);
    assert!(full_snapshot
        .pipeline_keys()
        .iter()
        .any(|key| key.contains("fused-pow2") && key.contains("n=4096")));
    let low_snapshot = import_pipeline_cache_snapshot(&low_device, &full_snapshot);
    assert!(!low_snapshot
        .pipeline_keys()
        .iter()
        .any(|key| key.contains("fused-pow2") && key.contains("n=4096")));

    let input = test_signal(4096);
    for inverse in [false, true] {
        let config = if inverse {
            FftConfig::inverse(4096)
        } else {
            FftConfig::new(4096).with_normalization(Normalization::None)
        };
        let expected = cpu_fft_pow2(&input, inverse, inverse);
        let (fused, fused_plan) =
            execute_c2c(&context.device, &context.queue, config.clone(), &input);
        let (multipass, multipass_plan) = execute_c2c(&low_device, &low_queue, config, &input);

        assert_fused_plan(&fused_plan, 1, 0);
        assert_stage_labels(&multipass_plan, 0, 4);
        assert_eq!(multipass_plan.workspace_size_bytes(), 4096 * 8);
        assert_close(
            &fused,
            &multipass,
            &format!("N=4096 fused versus multipass inverse={inverse}"),
        );
        assert_close(
            &fused,
            &expected,
            &format!("N=4096 fused versus CPU inverse={inverse}"),
        );
    }

    #[cfg(windows)]
    {
        std::mem::forget(low_queue);
        std::mem::forget(low_device);
    }
}

fn execute_c2c(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    config: FftConfig,
    input: &[f32],
) -> (Vec<f32>, FftPlan) {
    let plan = FftPlan::c2c(device, queue, config).unwrap();
    assert_eq!(plan.route(), C2cRoute::MixedRadix);
    let byte_len = std::mem::size_of_val(input) as u64;
    assert_eq!(plan.required_buffer_size_bytes(), byte_len);

    let input_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.fused_pow2.input"),
        size: byte_len,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.fused_pow2.output"),
        size: byte_len,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.fused_pow2.readback"),
        size: byte_len,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    queue.write_buffer(&input_buffer, 0, bytemuck::cast_slice(input));

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("wgpu_fft.test.fused_pow2.encoder"),
    });
    plan.execute_checked(device, &mut encoder, &input_buffer, &output_buffer)
        .unwrap();
    encoder.copy_buffer_to_buffer(&output_buffer, 0, &readback_buffer, 0, byte_len);
    queue.submit([encoder.finish()]);

    let actual = read_f32(device, &readback_buffer);
    (actual, plan)
}

fn assert_fused_plan(plan: &FftPlan, fused_stages: usize, workspace_bytes: u64) {
    assert_stage_labels(plan, fused_stages, 0);
    assert_eq!(plan.workspace_size_bytes(), workspace_bytes);
}

fn assert_stage_labels(plan: &FftPlan, fused_stages: usize, stockham_stages: usize) {
    let diagnostics = plan.diagnostics();
    assert!(diagnostics.blockers().is_empty());
    let kernels = diagnostics
        .stages()
        .iter()
        .filter(|stage| stage.kind == "kernel")
        .collect::<Vec<_>>();
    assert_eq!(
        kernels
            .iter()
            .filter(|stage| stage.label == FUSED_LABEL)
            .count(),
        fused_stages,
        "unexpected fused-stage diagnostics: {kernels:?}"
    );
    assert_eq!(
        kernels
            .iter()
            .filter(|stage| stage.label == STOCKHAM_LABEL)
            .count(),
        stockham_stages,
        "unexpected Stockham-stage diagnostics: {kernels:?}"
    );
    assert_eq!(
        kernels.len(),
        fused_stages + stockham_stages,
        "unexpected extra kernel diagnostics: {kernels:?}"
    );
}

fn test_signal(complex_len: usize) -> Vec<f32> {
    (0..complex_len)
        .flat_map(|index| {
            let x = index as f32 + 1.0;
            [
                (x * 0.017).sin() * 0.7 + (x * 0.031).cos() * 0.2,
                (x * 0.023).cos() * 0.5 - (x * 0.011).sin() * 0.3,
            ]
        })
        .collect()
}

fn cpu_fft_pow2(input: &[f32], inverse: bool, normalize: bool) -> Vec<f32> {
    let n = input.len() / 2;
    assert!(n.is_power_of_two());
    let mut values = input
        .chunks_exact(2)
        .map(|value| (f64::from(value[0]), f64::from(value[1])))
        .collect::<Vec<_>>();

    let mut reversed = 0usize;
    for index in 1..n {
        let mut bit = n >> 1;
        while reversed & bit != 0 {
            reversed ^= bit;
            bit >>= 1;
        }
        reversed ^= bit;
        if index < reversed {
            values.swap(index, reversed);
        }
    }

    let sign = if inverse { 1.0 } else { -1.0 };
    let mut width = 2usize;
    while width <= n {
        let angle = sign * 2.0 * PI / width as f64;
        let root = (angle.cos(), angle.sin());
        for base in (0..n).step_by(width) {
            let mut twiddle = (1.0, 0.0);
            for offset in 0..width / 2 {
                let lower = values[base + offset];
                let upper = values[base + offset + width / 2];
                let product = complex_mul(twiddle, upper);
                values[base + offset] = (lower.0 + product.0, lower.1 + product.1);
                values[base + offset + width / 2] = (lower.0 - product.0, lower.1 - product.1);
                twiddle = complex_mul(twiddle, root);
            }
        }
        width *= 2;
    }

    let scale = if normalize { 1.0 / n as f64 } else { 1.0 };
    values
        .into_iter()
        .flat_map(|value| [(value.0 * scale) as f32, (value.1 * scale) as f32])
        .collect()
}

fn complex_mul(a: (f64, f64), b: (f64, f64)) -> (f64, f64) {
    (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
}

fn assert_close(actual: &[f32], expected: &[f32], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}");
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        let tolerance = 2.0e-2 + 2.0e-5 * expected.abs();
        assert!(
            actual.is_finite() && (actual - expected).abs() <= tolerance,
            "{label}: index {index}: actual={actual}, expected={expected}, tolerance={tolerance}"
        );
    }
}

fn read_f32(device: &wgpu::Device, buffer: &wgpu::Buffer) -> Vec<f32> {
    let slice = buffer.slice(..);
    let (sender, receiver) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        sender.send(result).expect("map result receiver is alive");
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("device polling should succeed");
    receiver
        .recv()
        .expect("map callback should send a result")
        .expect("readback buffer should map");
    let mapped = slice.get_mapped_range();
    let values = bytemuck::cast_slice(&mapped).to_vec();
    drop(mapped);
    buffer.unmap();
    values
}
