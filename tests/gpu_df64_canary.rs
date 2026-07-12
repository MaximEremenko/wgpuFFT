//! Opt-in backend canaries for the portable double-float error-free transforms.

use std::mem::ManuallyDrop;
use std::sync::mpsc;

use wgpu::util::DeviceExt;
use wgpu_fft::math::{
    quick_two_sum_f32, split_f32, two_prod_f32, two_sum_f32, ComplexDoubleFloat, DoubleFloat,
};

const WORDS_PER_CASE: usize = 22;

#[test]
fn gpu_df64_error_free_transform_canaries() {
    if std::env::var_os("WGPU_FFT_RUN_GPU_TESTS").is_none() {
        eprintln!("skipping GPU test; set WGPU_FFT_RUN_GPU_TESTS=1 to run it");
        return;
    }

    pollster::block_on(run_canaries());
}

async fn run_canaries() {
    let Some(context) = wgpu_fft::device::request_default_device().await else {
        eprintln!("skipping GPU test; no suitable wgpu adapter was found");
        return;
    };
    // Native wgpu teardown can stall on Windows after an assertion panic. Keep
    // all native objects alive until process exit, matching the other GPU suites.
    let context = ManuallyDrop::new(context);
    let adapter_info = context.adapter.get_info();
    eprintln!(
        "df64 canary adapter: name={:?}, backend={:?}, device_type={:?}, driver={:?}",
        adapter_info.name, adapter_info.backend, adapter_info.device_type, adapter_info.driver
    );

    // Request a featureless device deliberately: these kernels are pure f32 and
    // their support contract must not accidentally depend on SHADER_F64.
    let (device, queue) = context
        .adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("wgpu_fft.test.df64_canary.featureless_device"),
            required_features: wgpu::Features::empty(),
            required_limits: context.adapter.limits(),
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: wgpu::MemoryHints::MemoryUsage,
            trace: wgpu::Trace::Off,
        })
        .await
        .expect("df64 canaries require only core f32 shader support");
    let featureless = ManuallyDrop::new((device, queue));
    assert!(featureless.0.features().is_empty());

    let cases = canary_inputs();
    let expected = cases
        .iter()
        .flat_map(|case| expected_words(*case))
        .collect::<Vec<_>>();
    assert_canaries_are_adversarial(&expected);

    let input_buffer = featureless
        .0
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("wgpu_fft.test.df64_canary.input"),
            contents: bytemuck::cast_slice(&cases),
            usage: wgpu::BufferUsages::STORAGE,
        });
    let output_bytes = (expected.len() * std::mem::size_of::<u32>()) as u64;
    let output_buffer = featureless.0.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.df64_canary.output"),
        size: output_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback = featureless.0.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.test.df64_canary.readback"),
        size: output_bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let shader_source = format!("{}\n{}", wgpu_fft::kernels::DF64_WGSL, CANARY_ENTRY_WGSL);
    let shader = featureless
        .0
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("wgpu_fft.test.df64_canary.shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });
    let pipeline = featureless
        .0
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("wgpu_fft.test.df64_canary.pipeline"),
            layout: None,
            module: &shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
    let bind_group = featureless.0.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("wgpu_fft.test.df64_canary.bind_group"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output_buffer.as_entire_binding(),
            },
        ],
    });

    let mut encoder = featureless
        .0
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("wgpu_fft.test.df64_canary.encoder"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("wgpu_fft.test.df64_canary.pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(cases.len() as u32, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&output_buffer, 0, &readback, 0, output_bytes);
    featureless.1.submit(Some(encoder.finish()));

    let actual = read_u32(&featureless.0, &readback);
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
        assert_eq!(
            actual, expected,
            "df64 backend invariant failed at case {}, word {}: actual=0x{actual:08x}, expected=0x{expected:08x}; WGSL contraction/reassociation may have broken an error-free transform",
            index / WORDS_PER_CASE,
            index % WORDS_PER_CASE,
        );
    }
    eprintln!(
        "df64 canaries passed: backend={:?}, cases={}, exact_words={}",
        adapter_info.backend,
        cases.len(),
        expected.len()
    );
}

fn canary_inputs() -> Vec<[[f32; 4]; 2]> {
    vec![
        [
            [16_777_216.0, 1.0, 16_777_216.0, 4097.0],
            [-16_777_216.0, 1.0, 1.0, 4097.0],
        ],
        [
            [1.0, f32::from_bits(0x3380_0000), 1.000_000_1, 1.000_000_1],
            [-1.0, f32::from_bits(0x3300_0000), -1.0, 0.999_999_9],
        ],
        [
            [1.0e36, 1.0e28, 1.0e20, f32::from_bits(0x7b40_97ce)],
            [1.0e-30, 1.0e-37, 1.0, f32::from_bits(0x0da2_4260)],
        ],
        [
            [-12_345.125, 0.000_122_070_31, -33_554_432.0, -17.25],
            [0.031_257_63, -1.0e-10, 3.0, 3.000_000_2],
        ],
    ]
}

fn expected_words(input: [[f32; 4]; 2]) -> [u32; WORDS_PER_CASE] {
    let a = input[0];
    let b = input[1];
    let (sum_hi, sum_lo) = two_sum_f32(a[0], b[0]);
    let (quick_hi, quick_lo) = quick_two_sum_f32(a[2], b[2]);
    let (split_hi, split_lo) = split_f32(a[3]);
    let (prod_hi, prod_lo) = two_prod_f32(a[3], b[3]);
    let a_dd = DoubleFloat::new(a[0], a[1]);
    let b_dd = DoubleFloat::new(b[0], b[1]);
    let add = a_dd.add_df(b_dd);
    let mul = a_dd.mul_df(b_dd);
    let a_complex = ComplexDoubleFloat::new(a_dd, DoubleFloat::new(a[2], a[3]));
    let b_complex = ComplexDoubleFloat::new(b_dd, DoubleFloat::new(b[2], b[3]));
    let complex_add = a_complex.add_df(b_complex);
    let complex_mul = a_complex.mul_df(b_complex);
    let (edge_prod_hi, edge_prod_lo) = two_prod_f32(f32::MAX, f32::MIN_POSITIVE);
    [
        sum_hi.to_bits(),
        sum_lo.to_bits(),
        quick_hi.to_bits(),
        quick_lo.to_bits(),
        split_hi.to_bits(),
        split_lo.to_bits(),
        prod_hi.to_bits(),
        prod_lo.to_bits(),
        add.hi.to_bits(),
        add.lo.to_bits(),
        mul.hi.to_bits(),
        mul.lo.to_bits(),
        complex_add.re_hi.to_bits(),
        complex_add.re_lo.to_bits(),
        complex_add.im_hi.to_bits(),
        complex_add.im_lo.to_bits(),
        complex_mul.re_hi.to_bits(),
        complex_mul.re_lo.to_bits(),
        complex_mul.im_hi.to_bits(),
        complex_mul.im_lo.to_bits(),
        edge_prod_hi.to_bits(),
        edge_prod_lo.to_bits(),
    ]
}

fn assert_canaries_are_adversarial(expected: &[u32]) {
    let first = &expected[..WORDS_PER_CASE];
    assert_eq!(first[2], 16_777_216.0f32.to_bits());
    assert_eq!(first[3], 1.0f32.to_bits());
    assert_eq!(first[4], 4096.0f32.to_bits());
    assert_eq!(first[5], 1.0f32.to_bits());
    assert_eq!(first[6], 16_785_408.0f32.to_bits());
    assert_eq!(first[7], 1.0f32.to_bits());
    assert_eq!(first[8], 2.0f32.to_bits());
    assert_eq!(first[9], 0.0f32.to_bits());
    assert!(
        expected
            .chunks_exact(WORDS_PER_CASE)
            .all(|words| words[7] != 0.0f32.to_bits()),
        "each two_prod canary must require a nonzero error word"
    );
    let edge = two_prod_f32(f32::MAX, f32::MIN_POSITIVE);
    for words in expected.chunks_exact(WORDS_PER_CASE) {
        assert_eq!(words[20], edge.0.to_bits());
        assert_eq!(words[21], edge.1.to_bits());
        assert!(f32::from_bits(words[20]).is_finite());
        assert!(f32::from_bits(words[21]).is_finite());
    }
}

fn read_u32(device: &wgpu::Device, buffer: &wgpu::Buffer) -> Vec<u32> {
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

const CANARY_ENTRY_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> inputs: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> outputs: array<u32>;

fn store_df64(base: u32, value: Df64) {
    outputs[base] = bitcast<u32>(value.hi);
    outputs[base + 1u] = bitcast<u32>(value.lo);
}

fn store_complex(base: u32, value: vec4<f32>) {
    outputs[base] = bitcast<u32>(value.x);
    outputs[base + 1u] = bitcast<u32>(value.y);
    outputs[base + 2u] = bitcast<u32>(value.z);
    outputs[base + 3u] = bitcast<u32>(value.w);
}

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let case_index = gid.x;
    let a = inputs[case_index * 2u];
    let b = inputs[case_index * 2u + 1u];
    let base = case_index * 22u;
    store_df64(base, df64_two_sum(a.x, b.x));
    store_df64(base + 2u, df64_quick_two_sum(a.z, b.z));
    store_df64(base + 4u, df64_split(a.w));
    store_df64(base + 6u, df64_two_prod(a.w, b.w));
    let a_dd = Df64(a.x, a.y);
    let b_dd = Df64(b.x, b.y);
    store_df64(base + 8u, df64_add(a_dd, b_dd));
    store_df64(base + 10u, df64_mul(a_dd, b_dd));
    store_complex(base + 12u, df64_complex_add(a, b));
    store_complex(base + 16u, df64_complex_mul(a, b));
    store_df64(base + 20u, df64_two_prod(
        bitcast<f32>(0x7f7fffffu),
        bitcast<f32>(0x00800000u),
    ));
}
"#;
