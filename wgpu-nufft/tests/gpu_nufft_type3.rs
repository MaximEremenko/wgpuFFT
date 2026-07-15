//! Opt-in 1D/2D/3D GPU type-3 validation against the direct f64 NDFT.

use std::sync::mpsc;

use wgpu::util::DeviceExt;
use wgpu_nufft::{
    reference_type3_f64, Complex64, NufftInterval, NufftSign, NufftType3Config, NufftType3Plan,
};

const FINUFFT_FLOAT_TOLERANCE_FACTOR: f64 = 100.0;
const ADJOINT_TOLERANCE: f64 = 2.0e-4;

#[test]
fn gpu_type3_matches_oracles_reuses_scratch_and_is_adjoint() {
    if std::env::var_os("WGPU_FFT_RUN_GPU_TESTS").is_none() {
        eprintln!("skipping GPU test; set WGPU_FFT_RUN_GPU_TESTS=1 to run it");
        return;
    }
    pollster::block_on(run_gpu_type3_cases());
}

async fn run_gpu_type3_cases() {
    let Some(context) = wgpu_fft::device::request_default_device().await else {
        panic!("WGPU_FFT_RUN_GPU_TESTS was set but no suitable adapter was found");
    };
    let info = context.adapter.get_info();
    eprintln!(
        "gpu_nufft_type3 adapter: {} backend={:?} driver={} {}",
        info.name, info.backend, info.driver, info.driver_info
    );

    validate_accuracy_matrix(&context.device, &context.queue);
    validate_scratch_grow_then_shrink(&context.device, &context.queue);
    validate_zero_source_and_target_counts(&context.device, &context.queue);
    validate_adjoint_consistency(&context.device, &context.queue);
    validate_repeat_determinism(&context.device, &context.queue);

    #[cfg(windows)]
    std::mem::forget(context);
}

fn validate_accuracy_matrix(device: &wgpu::Device, queue: &wgpu::Queue) {
    for dimensions in 1..=3 {
        let (source_bounds, target_bounds) = planned_bounds(dimensions);
        for eps in [1.0e-2, 1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6] {
            for sign in [NufftSign::Positive, NufftSign::Negative] {
                let config =
                    NufftType3Config::new(source_bounds.clone(), target_bounds.clone(), eps)
                        .with_sign(sign);
                let plan = NufftType3Plan::new_gpu(device, queue, config.clone()).unwrap();
                assert!(plan.is_gpu_ready());
                assert_eq!(plan.config().dimensions(), dimensions);
                for class in [
                    PointClass::Random,
                    PointClass::Clustered,
                    PointClass::Boundary,
                    PointClass::Duplicates,
                ] {
                    let source_count = class.source_count();
                    let target_count = class.target_count();
                    let source = class.coordinates(
                        &source_bounds,
                        source_count,
                        0x31a7_7e09 ^ dimensions as u32,
                    );
                    let target = class.coordinates(
                        &target_bounds,
                        target_count,
                        0x85eb_ca6b ^ dimensions as u32,
                    );
                    let strengths = test_values(source_count, sign);
                    let actual = interleaved_to_complex64(&execute_type3(
                        device, queue, &plan, &source, &strengths, &target,
                    ));
                    let reference = reference_type3_f64(
                        &config,
                        &f32_to_f64(&source),
                        &f32_to_f64(&target),
                        &interleaved_to_complex64(&strengths),
                    )
                    .unwrap();
                    let error = relative_l2(&actual, &reference);
                    let tolerance = FINUFFT_FLOAT_TOLERANCE_FACTOR * eps;
                    eprintln!(
                        "NUFFT_TYPE3_ACCURACY dimensions={dimensions} eps={eps:.0e} sign={sign:?} class={} M={source_count} K={target_count} outer_grid={:?} relative_l2={error:.9e}",
                        class.label(),
                        plan.outer_grid_shape(),
                    );
                    assert!(
                        error <= tolerance,
                        "type3 dimensions={dimensions} eps={eps} sign={sign:?} class={}: relative L2 {error} exceeds {tolerance}",
                        class.label(),
                    );
                }
            }
        }
    }
}

fn validate_scratch_grow_then_shrink(device: &wgpu::Device, queue: &wgpu::Queue) {
    let dimensions = 3;
    let (source_bounds, target_bounds) = planned_bounds(dimensions);
    let config = NufftType3Config::new(source_bounds.clone(), target_bounds.clone(), 1.0e-6)
        .with_sign(NufftSign::Negative);
    let plan = NufftType3Plan::new_gpu(device, queue, config.clone()).unwrap();

    for (label, class, source_count, target_count) in [
        ("large", PointClass::Random, 257, 193),
        ("small-after-large", PointClass::Clustered, 7, 5),
    ] {
        let source = class.coordinates(&source_bounds, source_count, 0x9e37_79b9);
        let target = class.coordinates(&target_bounds, target_count, 0x243f_6a88);
        let strengths = test_values(source_count, NufftSign::Negative);
        let actual = interleaved_to_complex64(&execute_type3(
            device, queue, &plan, &source, &strengths, &target,
        ));
        let reference = reference_type3_f64(
            &config,
            &f32_to_f64(&source),
            &f32_to_f64(&target),
            &interleaved_to_complex64(&strengths),
        )
        .unwrap();
        let error = relative_l2(&actual, &reference);
        eprintln!(
            "NUFFT_TYPE3_SCRATCH_REUSE phase={label} M={source_count} K={target_count} relative_l2={error:.9e}"
        );
        assert!(error <= 1.0e-4);
    }
}

fn validate_zero_source_and_target_counts(device: &wgpu::Device, queue: &wgpu::Queue) {
    let (source_bounds, target_bounds) = planned_bounds(1);
    let config = NufftType3Config::new(source_bounds, target_bounds.clone(), 1.0e-6);
    let plan = NufftType3Plan::new_gpu(device, queue, config).unwrap();
    let target = PointClass::Boundary.coordinates(&target_bounds, 5, 0x6a09_e667);
    let target_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("wgpu_nufft.type3.test.zero.target"),
        contents: bytemuck::cast_slice(&target),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let dummy = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_nufft.type3.test.zero.dummy"),
        size: 8,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let output_bytes = plan.required_output_buffer_size_bytes(5).unwrap();
    let output = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_nufft.type3.test.zero.output"),
        size: output_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_nufft.type3.test.zero.readback"),
        size: output_bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("wgpu_nufft.type3.test.zero.encoder"),
    });
    plan.encode_gpu(
        device,
        &mut encoder,
        0,
        &dummy,
        &dummy,
        5,
        &target_buffer,
        &output,
    )
    .unwrap();
    encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, output_bytes);
    queue.submit([encoder.finish()]);
    assert!(
        read_f32_buffer(device, &readback)
            .iter()
            .all(|&value| value == 0.0),
        "zero-source type-3 output must overwrite stale scratch with zeros"
    );

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("wgpu_nufft.type3.test.zero_target.encoder"),
    });
    plan.encode_gpu(device, &mut encoder, 0, &dummy, &dummy, 0, &dummy, &dummy)
        .unwrap();
    queue.submit([encoder.finish()]);
}

fn validate_adjoint_consistency(device: &wgpu::Device, queue: &wgpu::Queue) {
    for dimensions in 1..=3 {
        let (source_bounds, target_bounds) = planned_bounds(dimensions);
        let source =
            PointClass::Random.coordinates(&source_bounds, 9, 0xa341_316c ^ dimensions as u32);
        let target =
            PointClass::Boundary.coordinates(&target_bounds, 11, 0xc801_3ea4 ^ dimensions as u32);
        let v = test_values(9, NufftSign::Positive);
        let u = test_values(11, NufftSign::Negative);
        for sign in [NufftSign::Positive, NufftSign::Negative] {
            let config =
                NufftType3Config::new(source_bounds.clone(), target_bounds.clone(), 1.0e-6)
                    .with_sign(sign);
            let adjoint_config =
                NufftType3Config::new(target_bounds.clone(), source_bounds.clone(), 1.0e-6)
                    .with_sign(opposite_sign(sign));
            let plan = NufftType3Plan::new_gpu(device, queue, config).unwrap();
            let adjoint_plan = NufftType3Plan::new_gpu(device, queue, adjoint_config).unwrap();
            let av = interleaved_to_complex64(&execute_type3(
                device, queue, &plan, &source, &v, &target,
            ));
            let astar_u = interleaved_to_complex64(&execute_type3(
                device,
                queue,
                &adjoint_plan,
                &target,
                &u,
                &source,
            ));
            let lhs = hermitian_inner(&interleaved_to_complex64(&u), &av);
            let rhs = hermitian_inner(&astar_u, &interleaved_to_complex64(&v));
            let residual = complex_abs(Complex64::new(lhs.re - rhs.re, lhs.im - rhs.im))
                / complex_abs(lhs).max(complex_abs(rhs)).max(1.0);
            eprintln!(
                "NUFFT_TYPE3_ADJOINT dimensions={dimensions} sign={sign:?} relative_residual={residual:.9e}"
            );
            assert!(residual <= ADJOINT_TOLERANCE);
        }
    }
}

fn validate_repeat_determinism(device: &wgpu::Device, queue: &wgpu::Queue) {
    let (source_bounds, target_bounds) = planned_bounds(3);
    let config = NufftType3Config::new(source_bounds.clone(), target_bounds.clone(), 1.0e-6);
    let plan = NufftType3Plan::new_gpu(device, queue, config).unwrap();
    let source = PointClass::Clustered.coordinates(&source_bounds, 47, 0xdead_beef);
    let target = PointClass::Duplicates.coordinates(&target_bounds, 31, 0x5a17_90c3);
    let strengths = test_values(47, NufftSign::Positive);
    let first = execute_type3(device, queue, &plan, &source, &strengths, &target);
    let second = execute_type3(device, queue, &plan, &source, &strengths, &target);
    assert_eq!(
        f32_bits(&first),
        f32_bits(&second),
        "type-3 repeated execution must be deterministic"
    );
}

#[derive(Debug, Clone, Copy)]
enum PointClass {
    Random,
    Clustered,
    Boundary,
    Duplicates,
}

impl PointClass {
    fn label(self) -> &'static str {
        match self {
            Self::Random => "random",
            Self::Clustered => "clustered",
            Self::Boundary => "boundary",
            Self::Duplicates => "duplicates",
        }
    }

    fn source_count(self) -> usize {
        match self {
            Self::Random => 11,
            Self::Clustered => 9,
            Self::Boundary => 7,
            Self::Duplicates => 8,
        }
    }

    fn target_count(self) -> usize {
        match self {
            Self::Random => 13,
            Self::Clustered => 8,
            Self::Boundary => 9,
            Self::Duplicates => 6,
        }
    }

    fn coordinates(self, bounds: &[NufftInterval], count: usize, mut state: u32) -> Vec<f32> {
        let mut coordinates = Vec::with_capacity(count * bounds.len());
        for point in 0..count {
            for (axis, bound) in bounds.iter().enumerate() {
                let lower = bound.lower() as f32;
                let upper = bound.upper() as f32;
                let center = 0.5 * (lower + upper);
                let half_width = 0.5 * (upper - lower);
                let coordinate = match self {
                    Self::Random => {
                        state ^= state << 13;
                        state ^= state >> 17;
                        state ^= state << 5;
                        let unit = state as f32 / u32::MAX as f32;
                        center + (2.0 * unit - 1.0) * (0.95 * half_width)
                    }
                    Self::Clustered => {
                        center + (point as f32 - count as f32 * 0.5) * 1.0e-5 + axis as f32 * 3.0e-6
                    }
                    Self::Boundary => {
                        if (point + axis).is_multiple_of(2) {
                            lower
                        } else {
                            upper
                        }
                    }
                    Self::Duplicates => center + (axis as f32 + 1.0) * 0.17 * half_width,
                };
                coordinates.push(coordinate);
            }
        }
        coordinates
    }
}

fn planned_bounds(dimensions: usize) -> (Vec<NufftInterval>, Vec<NufftInterval>) {
    let source_centers = [0.35f32, -0.45, 0.7];
    let source_half_widths = [1.0f32, 1.2, 0.8];
    let target_centers = [0.8f32, -1.1, 0.55];
    let target_half_widths = [2.2f32, 1.7, 2.5];
    let source = (0..dimensions)
        .map(|axis| {
            NufftInterval::new(
                f64::from(source_centers[axis] - source_half_widths[axis]),
                f64::from(source_centers[axis] + source_half_widths[axis]),
            )
        })
        .collect();
    let target = (0..dimensions)
        .map(|axis| {
            NufftInterval::new(
                f64::from(target_centers[axis] - target_half_widths[axis]),
                f64::from(target_centers[axis] + target_half_widths[axis]),
            )
        })
        .collect();
    (source, target)
}

fn execute_type3(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    plan: &NufftType3Plan,
    source: &[f32],
    strengths: &[f32],
    target: &[f32],
) -> Vec<f32> {
    let dimensions = plan.config().dimensions();
    assert!(source.len().is_multiple_of(dimensions));
    assert!(target.len().is_multiple_of(dimensions));
    let source_count = source.len() / dimensions;
    let target_count = target.len() / dimensions;
    assert_eq!(strengths.len(), source_count * 2);

    let source_bytes = plan
        .required_source_point_buffer_size_bytes(source_count)
        .unwrap();
    let target_bytes = plan
        .required_target_point_buffer_size_bytes(target_count)
        .unwrap();
    let strength_bytes = plan
        .required_strength_buffer_size_bytes(source_count)
        .unwrap();
    let output_bytes = plan
        .required_output_buffer_size_bytes(target_count)
        .unwrap();
    assert_eq!(source_bytes, source.len() as u64 * 4);
    assert_eq!(target_bytes, target.len() as u64 * 4);
    assert_eq!(strength_bytes, strengths.len() as u64 * 4);

    let source_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("wgpu_nufft.type3.test.source"),
        contents: bytemuck::cast_slice(source),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let strength_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("wgpu_nufft.type3.test.strengths"),
        contents: bytemuck::cast_slice(strengths),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let target_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("wgpu_nufft.type3.test.target"),
        contents: bytemuck::cast_slice(target),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let output = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_nufft.type3.test.output"),
        size: output_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_nufft.type3.test.readback"),
        size: output_bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("wgpu_nufft.type3.test.encoder"),
    });
    plan.encode_gpu(
        device,
        &mut encoder,
        source_count,
        &source_buffer,
        &strength_buffer,
        target_count,
        &target_buffer,
        &output,
    )
    .unwrap();
    encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, output_bytes);
    queue.submit([encoder.finish()]);
    read_f32_buffer(device, &readback)
}

fn read_f32_buffer(device: &wgpu::Device, buffer: &wgpu::Buffer) -> Vec<f32> {
    let slice = buffer.slice(..);
    let (sender, receiver) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        sender.send(result).unwrap();
    });
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    receiver.recv().unwrap().unwrap();
    let mapped = slice.get_mapped_range();
    let result = bytemuck::cast_slice::<u8, f32>(&mapped).to_vec();
    drop(mapped);
    buffer.unmap();
    result
}

fn test_values(count: usize, sign: NufftSign) -> Vec<f32> {
    let bias = if sign == NufftSign::Positive {
        0.17
    } else {
        -0.11
    };
    (0..count)
        .flat_map(|index| {
            let value = index as f32 + 1.0;
            [
                (value * 0.31 + bias).sin() * 0.7,
                (value * 0.23 - bias).cos() * 0.5,
            ]
        })
        .collect()
}

fn f32_to_f64(values: &[f32]) -> Vec<f64> {
    values.iter().map(|&value| f64::from(value)).collect()
}

fn interleaved_to_complex64(values: &[f32]) -> Vec<Complex64> {
    values
        .chunks_exact(2)
        .map(|pair| Complex64::new(f64::from(pair[0]), f64::from(pair[1])))
        .collect()
}

fn relative_l2(actual: &[Complex64], reference: &[Complex64]) -> f64 {
    let error_squared = actual
        .iter()
        .zip(reference)
        .map(|(actual, reference)| {
            let re = actual.re - reference.re;
            let im = actual.im - reference.im;
            re * re + im * im
        })
        .sum::<f64>();
    let reference_squared = reference
        .iter()
        .map(|value| value.re * value.re + value.im * value.im)
        .sum::<f64>();
    error_squared.sqrt() / reference_squared.sqrt().max(f64::MIN_POSITIVE)
}

fn hermitian_inner(left: &[Complex64], right: &[Complex64]) -> Complex64 {
    left.iter()
        .zip(right)
        .fold(Complex64::default(), |mut sum, (left, right)| {
            sum.re += left.re * right.re + left.im * right.im;
            sum.im += left.re * right.im - left.im * right.re;
            sum
        })
}

fn complex_abs(value: Complex64) -> f64 {
    value.re.hypot(value.im)
}

fn opposite_sign(sign: NufftSign) -> NufftSign {
    match sign {
        NufftSign::Positive => NufftSign::Negative,
        NufftSign::Negative => NufftSign::Positive,
    }
}

fn f32_bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}
