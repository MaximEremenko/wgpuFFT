#![cfg(target_arch = "wasm32")]

//! Browser-default WebGPU validation for the complete NUFFT transform surface.

use std::f64::consts::PI;

use futures_channel::oneshot;
use wasm_bindgen_test::*;
use wgpu::util::DeviceExt;
use wgpu_fft::{
    validate_df64_invariants, FftPrecision, DF64_CANARY_CASE_COUNT, DF64_CANARY_WORD_COUNT,
};
use wgpu_nufft::{
    reference_type1_f64, reference_type2_f64, reference_type3_f64, Complex64, ComplexDoubleFloat,
    DoubleFloat, NufftConfig, NufftError, NufftInterval, NufftPlan, NufftSign, NufftType3Config,
    NufftType3Plan,
};

wasm_bindgen_test_configure!(run_in_browser);

const BATCH: usize = 2;
const TYPE12_POINT_COUNT: usize = 7;
const TYPE3_SOURCE_COUNT: usize = 7;
const TYPE3_TARGET_COUNT: usize = 6;
const F32_EPS: f64 = 1.0e-5;
const DF64_EPS: f64 = 1.0e-8;

struct BrowserDefaultContext {
    _instance: wgpu::Instance,
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
}

struct PackedScalars {
    bytes: Vec<u8>,
    canonical: Vec<f64>,
}

struct PackedComplex {
    bytes: Vec<u8>,
    canonical: Vec<Complex64>,
}

#[wasm_bindgen_test(async)]
async fn browser_default_nufft_surface_matches_direct_f64_oracles() {
    let context = request_browser_default_device().await;
    assert_browser_defaults(&context);

    let canary = validate_df64_invariants(&context.device, &context.queue)
        .await
        .expect("Chrome/Tint must preserve the portable-df64 invariants");
    assert_eq!(canary.cases, DF64_CANARY_CASE_COUNT);
    assert_eq!(canary.exact_words, DF64_CANARY_WORD_COUNT);
    console_log!(
        "browser NUFFT df64 canary passed: cases={} exact_words={}",
        canary.cases,
        canary.exact_words
    );

    let out_of_memory = context
        .device
        .push_error_scope(wgpu::ErrorFilter::OutOfMemory);
    let internal = context.device.push_error_scope(wgpu::ErrorFilter::Internal);
    let validation = context
        .device
        .push_error_scope(wgpu::ErrorFilter::Validation);

    for precision in [FftPrecision::F32, FftPrecision::Df64] {
        for (shape, sign) in [
            (vec![17], NufftSign::Positive),
            (vec![8, 12], NufftSign::Negative),
            (vec![4, 6, 8], NufftSign::Positive),
        ] {
            run_type12_case(&context, shape, precision, sign).await;
        }

        for (dimensions, sign) in [
            (1, NufftSign::Negative),
            (2, NufftSign::Positive),
            (3, NufftSign::Negative),
        ] {
            run_type3_case(&context, dimensions, precision, sign).await;
        }
    }

    run_df64_type3_phase_boundary_canary(&context).await;
    assert_df64_type3_phase_above_boundary_is_structured(&context);

    // Pop every nested scope before awaiting any one result. WebGPU requires
    // strict LIFO initiation; awaiting between pops can let unrelated async
    // device work race with a still-active outer scope.
    let validation_pop = validation.pop();
    let internal_pop = internal.pop();
    let out_of_memory_pop = out_of_memory.pop();
    let validation_error = validation_pop.await;
    let internal_error = internal_pop.await;
    let out_of_memory_error = out_of_memory_pop.await;
    assert!(
        validation_error.is_none() && internal_error.is_none() && out_of_memory_error.is_none(),
        "browser NUFFT error scopes: validation={validation_error:?} internal={internal_error:?} out_of_memory={out_of_memory_error:?}"
    );
}

async fn request_browser_default_device() -> BrowserDefaultContext {
    let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
    descriptor.backends = wgpu::Backends::BROWSER_WEBGPU;
    let instance = wgpu::Instance::new(descriptor);
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        })
        .await
        .expect("Chrome must expose a WebGPU adapter");
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("wgpu_nufft.browser.default_device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: wgpu::MemoryHints::MemoryUsage,
            trace: wgpu::Trace::Off,
        })
        .await
        .expect("Chrome must grant WebGPU default limits");
    BrowserDefaultContext {
        _instance: instance,
        adapter,
        device,
        queue,
    }
}

fn assert_browser_defaults(context: &BrowserDefaultContext) {
    let info = context.adapter.get_info();
    assert_eq!(info.backend, wgpu::Backend::BrowserWebGpu);
    assert_eq!(context.device.features(), wgpu::Features::empty());
    let expected = wgpu::Limits::default();
    let actual = context.device.limits();
    assert_eq!(
        actual.max_storage_buffer_binding_size,
        expected.max_storage_buffer_binding_size
    );
    assert_eq!(actual.max_buffer_size, expected.max_buffer_size);
    assert_eq!(
        actual.max_compute_workgroup_storage_size,
        expected.max_compute_workgroup_storage_size
    );
    assert_eq!(
        actual.max_compute_invocations_per_workgroup,
        expected.max_compute_invocations_per_workgroup
    );
    console_log!(
        "browser NUFFT adapter={:?} backend={:?} limits={{max_bind:{}, max_buffer:{}, workgroup_storage:{}, invocations:{}}}",
        info.name,
        info.backend,
        actual.max_storage_buffer_binding_size,
        actual.max_buffer_size,
        actual.max_compute_workgroup_storage_size,
        actual.max_compute_invocations_per_workgroup,
    );
}

async fn run_type12_case(
    context: &BrowserDefaultContext,
    shape: Vec<usize>,
    precision: FftPrecision,
    sign: NufftSign,
) {
    let dimensions = shape.len();
    let mode_count = shape.iter().product::<usize>();
    let eps = requested_eps(precision);
    let config = NufftConfig::new(shape.clone(), eps)
        .with_sign(sign)
        .with_batch(BATCH)
        .with_precision(precision);
    let points = pack_scalars(&type12_points(dimensions), precision);
    let strengths = pack_complex(&complex_values(BATCH, TYPE12_POINT_COUNT, 0.37), precision);
    let coefficients = pack_complex(&complex_values(BATCH, mode_count, -0.29), precision);
    let point_buffer = storage_buffer(
        &context.device,
        "wgpu_nufft.browser.type12.points",
        &points.bytes,
    );

    let type1 = NufftPlan::type1_gpu(&context.device, &context.queue, config.clone())
        .expect("browser type-1 plan");
    let strength_buffer = storage_buffer(
        &context.device,
        "wgpu_nufft.browser.type1.strengths",
        &strengths.bytes,
    );
    let type1_bytes = type1
        .required_type1_output_buffer_size_bytes_for_batch(BATCH)
        .unwrap();
    let actual_type1 = execute_and_read_complex(
        context,
        precision,
        type1_bytes,
        "wgpu_nufft.browser.type1",
        |encoder, output| {
            type1
                .encode_type1_gpu_batch(
                    &context.device,
                    encoder,
                    TYPE12_POINT_COUNT,
                    BATCH,
                    &point_buffer,
                    &strength_buffer,
                    output,
                )
                .unwrap();
        },
    )
    .await;
    let reference_type1 =
        reference_type1_f64(&config, &points.canonical, &strengths.canonical).unwrap();
    assert_oracle(
        "type1",
        dimensions,
        precision,
        sign,
        &actual_type1,
        &reference_type1,
    );

    let type2 = NufftPlan::type2_gpu(&context.device, &context.queue, config.clone())
        .expect("browser type-2 plan");
    let coefficient_buffer = storage_buffer(
        &context.device,
        "wgpu_nufft.browser.type2.coefficients",
        &coefficients.bytes,
    );
    let type2_bytes = type2
        .required_type2_output_buffer_size_bytes_for_batch(TYPE12_POINT_COUNT, BATCH)
        .unwrap();
    let actual_type2 = execute_and_read_complex(
        context,
        precision,
        type2_bytes,
        "wgpu_nufft.browser.type2",
        |encoder, output| {
            type2
                .encode_type2_gpu_batch(
                    &context.device,
                    encoder,
                    TYPE12_POINT_COUNT,
                    BATCH,
                    &point_buffer,
                    &coefficient_buffer,
                    output,
                )
                .unwrap();
        },
    )
    .await;
    let reference_type2 =
        reference_type2_f64(&config, &points.canonical, &coefficients.canonical).unwrap();
    assert_oracle(
        "type2",
        dimensions,
        precision,
        sign,
        &actual_type2,
        &reference_type2,
    );
}

async fn run_type3_case(
    context: &BrowserDefaultContext,
    dimensions: usize,
    precision: FftPrecision,
    sign: NufftSign,
) {
    let (source_bounds, target_bounds) = type3_bounds(dimensions);
    let source = pack_scalars(
        &interval_points(&source_bounds, TYPE3_SOURCE_COUNT),
        precision,
    );
    let target = pack_scalars(
        &interval_points(&target_bounds, TYPE3_TARGET_COUNT),
        precision,
    );
    let strengths = pack_complex(&complex_values(BATCH, TYPE3_SOURCE_COUNT, 0.53), precision);
    let eps = requested_eps(precision);
    let config = NufftType3Config::new(source_bounds, target_bounds, eps)
        .with_sign(sign)
        .with_batch(BATCH)
        .with_precision(precision);
    let plan = NufftType3Plan::new_gpu(&context.device, &context.queue, config.clone())
        .expect("browser type-3 plan");
    let source_buffer = storage_buffer(
        &context.device,
        "wgpu_nufft.browser.type3.source",
        &source.bytes,
    );
    let target_buffer = storage_buffer(
        &context.device,
        "wgpu_nufft.browser.type3.target",
        &target.bytes,
    );
    let strength_buffer = storage_buffer(
        &context.device,
        "wgpu_nufft.browser.type3.strengths",
        &strengths.bytes,
    );
    let output_bytes = plan
        .required_output_buffer_size_bytes_for_batch(TYPE3_TARGET_COUNT, BATCH)
        .unwrap();
    let actual = execute_and_read_complex(
        context,
        precision,
        output_bytes,
        "wgpu_nufft.browser.type3",
        |encoder, output| {
            plan.encode_gpu_batch(
                &context.device,
                encoder,
                BATCH,
                TYPE3_SOURCE_COUNT,
                &source_buffer,
                &strength_buffer,
                TYPE3_TARGET_COUNT,
                &target_buffer,
                output,
            )
            .unwrap();
        },
    )
    .await;
    let reference = reference_type3_f64(
        &config,
        &source.canonical,
        &target.canonical,
        &strengths.canonical,
    )
    .unwrap();
    assert_oracle("type3", dimensions, precision, sign, &actual, &reference);
}

async fn run_df64_type3_phase_boundary_canary(context: &BrowserDefaultContext) {
    let source_bounds = vec![NufftInterval::new(1.0, 1.0)];
    let target_bounds = vec![NufftInterval::new(1023.75, 1024.25)];
    let config = NufftType3Config::new(source_bounds, target_bounds, DF64_EPS)
        .with_sign(NufftSign::Negative)
        .with_precision(FftPrecision::Df64);
    let plan = NufftType3Plan::new_gpu(&context.device, &context.queue, config.clone())
        .expect("the exact portable-df64 phase boundary must be accepted");
    let source = pack_scalars(&[1.0, 1.0], FftPrecision::Df64);
    let target = pack_scalars(&[1023.75, 1024.0, 1024.25], FftPrecision::Df64);
    let strengths = pack_complex(
        &[Complex64::new(0.75, -0.25), Complex64::new(-0.125, 0.625)],
        FftPrecision::Df64,
    );
    let source_buffer = storage_buffer(
        &context.device,
        "wgpu_nufft.browser.phase.source",
        &source.bytes,
    );
    let target_buffer = storage_buffer(
        &context.device,
        "wgpu_nufft.browser.phase.target",
        &target.bytes,
    );
    let strength_buffer = storage_buffer(
        &context.device,
        "wgpu_nufft.browser.phase.strengths",
        &strengths.bytes,
    );
    let output_bytes = plan.required_output_buffer_size_bytes(3).unwrap();
    let actual = execute_and_read_complex(
        context,
        FftPrecision::Df64,
        output_bytes,
        "wgpu_nufft.browser.phase",
        |encoder, output| {
            plan.encode_gpu(
                &context.device,
                encoder,
                2,
                &source_buffer,
                &strength_buffer,
                3,
                &target_buffer,
                output,
            )
            .unwrap();
        },
    )
    .await;
    let reference = reference_type3_f64(
        &config,
        &source.canonical,
        &target.canonical,
        &strengths.canonical,
    )
    .unwrap();
    assert_oracle(
        "type3-df64-phase-boundary",
        1,
        FftPrecision::Df64,
        NufftSign::Negative,
        &actual,
        &reference,
    );
}

fn assert_df64_type3_phase_above_boundary_is_structured(context: &BrowserDefaultContext) {
    let config = NufftType3Config::new(
        [NufftInterval::new(1.0, 1.0)],
        [NufftInterval::new(1024.75, 1025.25)],
        DF64_EPS,
    )
    .with_precision(FftPrecision::Df64);
    let error = NufftType3Plan::new_gpu(&context.device, &context.queue, config)
        .expect_err("a portable-df64 phase bound above 1024 must be rejected");
    assert_eq!(
        error,
        NufftError::Type3RescalingUnsupported {
            axis: 0,
            quantity: "source pre-phase",
            value: 1025.0,
            reason: "the conservative phase bound exceeds the portable-df64 accuracy range",
        }
    );
}

async fn execute_and_read_complex(
    context: &BrowserDefaultContext,
    precision: FftPrecision,
    output_bytes: u64,
    label: &str,
    encode: impl FnOnce(&mut wgpu::CommandEncoder, &wgpu::Buffer),
) -> Vec<Complex64> {
    let output = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: output_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_nufft.browser.readback"),
        size: output_bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) });
    encode(&mut encoder, &output);
    encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, output_bytes);
    context.queue.submit([encoder.finish()]);

    let (sender, receiver) = oneshot::channel();
    readback
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
    receiver
        .await
        .expect("browser readback callback must run")
        .expect("browser readback mapping must succeed");
    let mapped = readback.slice(..).get_mapped_range();
    let values = match precision {
        FftPrecision::F32 => bytemuck::cast_slice::<u8, f32>(&mapped)
            .chunks_exact(2)
            .map(|pair| Complex64::new(f64::from(pair[0]), f64::from(pair[1])))
            .collect(),
        FftPrecision::Df64 => bytemuck::cast_slice::<u8, ComplexDoubleFloat>(&mapped)
            .iter()
            .map(|value| Complex64::new(value.re().to_f64(), value.im().to_f64()))
            .collect(),
        FftPrecision::F64 => unreachable!("browser NUFFT matrix never requests native f64"),
    };
    drop(mapped);
    readback.unmap();
    values
}

fn storage_buffer(device: &wgpu::Device, label: &str, bytes: &[u8]) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytes,
        usage: wgpu::BufferUsages::STORAGE,
    })
}

fn pack_scalars(values: &[f64], precision: FftPrecision) -> PackedScalars {
    match precision {
        FftPrecision::F32 => {
            let packed = values.iter().map(|&value| value as f32).collect::<Vec<_>>();
            PackedScalars {
                bytes: bytemuck::cast_slice(&packed).to_vec(),
                canonical: packed.iter().map(|&value| f64::from(value)).collect(),
            }
        }
        FftPrecision::Df64 => {
            let packed = values
                .iter()
                .copied()
                .map(DoubleFloat::from_f64)
                .collect::<Vec<_>>();
            PackedScalars {
                bytes: bytemuck::cast_slice(&packed).to_vec(),
                canonical: packed.iter().map(|value| value.to_f64()).collect(),
            }
        }
        FftPrecision::F64 => unreachable!("browser NUFFT matrix never requests native f64"),
    }
}

fn pack_complex(values: &[Complex64], precision: FftPrecision) -> PackedComplex {
    match precision {
        FftPrecision::F32 => {
            let packed = values
                .iter()
                .flat_map(|value| [value.re as f32, value.im as f32])
                .collect::<Vec<_>>();
            PackedComplex {
                bytes: bytemuck::cast_slice(&packed).to_vec(),
                canonical: packed
                    .chunks_exact(2)
                    .map(|pair| Complex64::new(f64::from(pair[0]), f64::from(pair[1])))
                    .collect(),
            }
        }
        FftPrecision::Df64 => {
            let packed = values
                .iter()
                .map(|value| ComplexDoubleFloat::from_f64(value.re, value.im))
                .collect::<Vec<_>>();
            PackedComplex {
                bytes: bytemuck::cast_slice(&packed).to_vec(),
                canonical: packed
                    .iter()
                    .map(|value| Complex64::new(value.re().to_f64(), value.im().to_f64()))
                    .collect(),
            }
        }
        FftPrecision::F64 => unreachable!("browser NUFFT matrix never requests native f64"),
    }
}

fn type12_points(dimensions: usize) -> Vec<f64> {
    let mut points = Vec::with_capacity(TYPE12_POINT_COUNT * dimensions);
    for point in 0..TYPE12_POINT_COUNT {
        for axis in 0..dimensions {
            let axis = axis as f64;
            let value = match point {
                0 => -PI + (axis + 1.0) * 1.0e-6,
                1 => PI - (axis + 1.0) * 1.0e-6,
                2 | 3 => 0.375 - 0.125 * axis,
                4 => -0.2 + axis * 0.07,
                5 => -2.4 + axis * 0.11,
                _ => 2.2 - axis * 0.09,
            };
            points.push(value);
        }
    }
    points
}

fn type3_bounds(dimensions: usize) -> (Vec<NufftInterval>, Vec<NufftInterval>) {
    // Binary-exact endpoints keep the adversarial boundary samples inside the
    // declared intervals after the F32 input path rounds them for upload.
    let source_centers = [0.375, -0.5, 0.75];
    let source_half_widths = [1.0, 1.25, 0.75];
    let target_centers = [0.75, -1.0, 0.5];
    let target_half_widths = [2.25, 1.75, 2.5];
    let source = (0..dimensions)
        .map(|axis| {
            NufftInterval::new(
                source_centers[axis] - source_half_widths[axis],
                source_centers[axis] + source_half_widths[axis],
            )
        })
        .collect();
    let target = (0..dimensions)
        .map(|axis| {
            NufftInterval::new(
                target_centers[axis] - target_half_widths[axis],
                target_centers[axis] + target_half_widths[axis],
            )
        })
        .collect();
    (source, target)
}

fn interval_points(bounds: &[NufftInterval], count: usize) -> Vec<f64> {
    let mut points = Vec::with_capacity(count * bounds.len());
    for point in 0..count {
        for (axis, bound) in bounds.iter().enumerate() {
            let center = 0.5 * (bound.lower() + bound.upper());
            let half_width = 0.5 * (bound.upper() - bound.lower());
            let value = match point {
                0 => bound.lower(),
                1 => bound.upper(),
                2 | 3 => center,
                4 => center + (axis as f64 + 1.0) * 1.0e-5,
                5 => center - 0.47 * half_width,
                _ => center + 0.61 * half_width,
            };
            points.push(value);
        }
    }
    points
}

fn complex_values(batch: usize, count: usize, bias: f64) -> Vec<Complex64> {
    (0..batch)
        .flat_map(|transform| {
            (0..count).map(move |index| {
                let x = index as f64 + 1.0;
                let offset = bias + transform as f64 * 0.117;
                Complex64::new(
                    (x * 0.31 + offset).sin() * 0.7,
                    (x * 0.23 - offset).cos() * 0.5,
                )
            })
        })
        .collect()
}

fn requested_eps(precision: FftPrecision) -> f64 {
    match precision {
        FftPrecision::F32 => F32_EPS,
        FftPrecision::Df64 => DF64_EPS,
        FftPrecision::F64 => unreachable!("browser NUFFT matrix never requests native f64"),
    }
}

fn oracle_tolerance(kind: &str, dimensions: usize, precision: FftPrecision) -> f64 {
    match precision {
        FftPrecision::F32 if matches!(kind, "type1" | "type2") => {
            let factor = match dimensions {
                1 => 4.0,
                2 => 20.0,
                3 => 32.0,
                _ => unreachable!("the browser matrix covers one through three dimensions"),
            };
            factor * F32_EPS
        }
        FftPrecision::F32 => 100.0 * F32_EPS,
        FftPrecision::Df64 => 24.0 * DF64_EPS,
        FftPrecision::F64 => unreachable!("browser NUFFT matrix never requests native f64"),
    }
}

fn assert_oracle(
    kind: &str,
    dimensions: usize,
    precision: FftPrecision,
    sign: NufftSign,
    actual: &[Complex64],
    reference: &[Complex64],
) {
    assert_eq!(actual.len(), reference.len());
    assert!(actual
        .iter()
        .all(|value| value.re.is_finite() && value.im.is_finite()));
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
    let relative_l2 = error_squared.sqrt() / reference_squared.sqrt().max(f64::MIN_POSITIVE);
    let tolerance = oracle_tolerance(kind, dimensions, precision);
    console_log!(
        "browser NUFFT accuracy kind={} dimensions={} precision={:?} sign={:?} relative_l2={:.9e} tolerance={:.9e}",
        kind,
        dimensions,
        precision,
        sign,
        relative_l2,
        tolerance,
    );
    assert!(
        relative_l2 <= tolerance,
        "{kind} {dimensions}D {precision:?} {sign:?}: relative L2 {relative_l2} exceeds {tolerance}"
    );
}
