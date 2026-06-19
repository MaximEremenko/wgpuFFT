use std::error::Error;
use std::fmt;
use std::io;
use std::time::{Duration, Instant};

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;
use wgpu_nufft::{
    Complex64, ComplexDoubleFloat, DoubleFloat, FftPrecision, ModeOrder, NufftConfig, NufftError,
    NufftInterval, NufftPlan, NufftSign, NufftType3Config, NufftType3Plan, DEFAULT_EPS,
    DEFAULT_SIGMA,
};

type BenchResult<T> = Result<T, Box<dyn Error>>;

const DEFAULT_RUNS: usize = 2;
const DEFAULT_SAMPLES: usize = 3;
const DEFAULT_WAIT_TIMEOUT: Duration = Duration::from_secs(120);
const TYPE3_SOURCE_COUNT: usize = 65_536;
const TYPE3_TARGET_COUNT: usize = 65_536;
const TYPE3_SOURCE_CENTER: f64 = std::f64::consts::PI / 4.0;
const TYPE3_SOURCE_HALFWIDTH: f64 = std::f64::consts::PI;
const TYPE3_TARGET_CENTER: f64 = 4.0;
const TYPE3_TARGET_HALFWIDTH: f64 = 16.0;
const DATA_SEED: u32 = 0x4E55_4646;
const COORDINATE_SEEDS: [u32; 3] = [0xA341_316C, 0xB7E1_5162, 0x9E37_79B9];
const TARGET_SEEDS: [u32; 3] = [0x243F_6A88, 0x85A3_08D3, 0x1319_8A2E];
const COMPLEX_REAL_SEED: u32 = 0xC801_3EA4;
const COMPLEX_IMAG_SEED: u32 = 0xAD90_777D;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Complex32 {
    re: f32,
    im: f32,
}

#[derive(Debug)]
struct Options {
    adapter_selector: Option<String>,
    precisions: Vec<PrecisionChoice>,
    cases: Vec<CaseChoice>,
    kinds: Vec<TransformKind>,
    runs: usize,
    samples: usize,
    wait_timeout: Duration,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PrecisionChoice {
    F32,
    F64,
    Df64,
}

impl PrecisionChoice {
    const ALL: [Self; 3] = [Self::F32, Self::F64, Self::Df64];

    const fn precision(self) -> FftPrecision {
        match self {
            Self::F32 => FftPrecision::F32,
            Self::F64 => FftPrecision::F64,
            Self::Df64 => FftPrecision::Df64,
        }
    }

    const fn name(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F64 => "f64",
            Self::Df64 => "df64",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CaseChoice {
    OneD,
    TwoD,
    ThreeD,
    Type3,
}

impl CaseChoice {
    const ALL: [Self; 4] = [Self::OneD, Self::TwoD, Self::ThreeD, Self::Type3];

    const fn name(self) -> &'static str {
        match self {
            Self::OneD => "1d",
            Self::TwoD => "2d",
            Self::ThreeD => "3d",
            Self::Type3 => "type3",
        }
    }

    fn mode_shape(self) -> Option<Vec<usize>> {
        match self {
            Self::OneD => Some(vec![262_144]),
            Self::TwoD => Some(vec![512, 512]),
            Self::ThreeD => Some(vec![64, 64, 64]),
            Self::Type3 => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TransformKind {
    Type1,
    Type2,
}

impl TransformKind {
    const ALL: [Self; 2] = [Self::Type1, Self::Type2];

    const fn name(self) -> &'static str {
        match self {
            Self::Type1 => "type-1",
            Self::Type2 => "type-2",
        }
    }

    fn host_plan(self, config: NufftConfig) -> wgpu_nufft::Result<NufftPlan> {
        match self {
            Self::Type1 => NufftPlan::type1(config),
            Self::Type2 => NufftPlan::type2(config),
        }
    }

    fn gpu_plan(
        self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        config: NufftConfig,
    ) -> wgpu_nufft::Result<NufftPlan> {
        match self {
            Self::Type1 => NufftPlan::type1_gpu(device, queue, config),
            Self::Type2 => NufftPlan::type2_gpu(device, queue, config),
        }
    }

    fn encode(
        self,
        plan: &NufftPlan,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        point_count: usize,
        buffers: &NormalBuffers,
    ) -> wgpu_nufft::Result<()> {
        match self {
            Self::Type1 => plan.encode_type1_gpu(
                device,
                encoder,
                point_count,
                &buffers.points,
                &buffers.input,
                &buffers.output,
            ),
            Self::Type2 => plan.encode_type2_gpu(
                device,
                encoder,
                point_count,
                &buffers.points,
                &buffers.input,
                &buffers.output,
            ),
        }
    }
}

struct NormalBuffers {
    points: wgpu::Buffer,
    input: wgpu::Buffer,
    output: wgpu::Buffer,
}

struct Type3Buffers {
    source_points: wgpu::Buffer,
    target_points: wgpu::Buffer,
    strengths: wgpu::Buffer,
    output: wgpu::Buffer,
}

#[derive(Debug)]
struct Statistics {
    raw_ms: Vec<f64>,
    run_means_ms: Vec<f64>,
    mean_ms: f64,
    stderr_ms: Option<f64>,
    minimum_ms: f64,
    plan_create_ms: Vec<f64>,
}

fn main() {
    if let Err(error) = pollster::block_on(run()) {
        eprintln!("wgpu-nufft precision benchmark failed: {error}");
        let mut source = error.source();
        while let Some(cause) = source {
            eprintln!("  caused by: {cause}");
            source = cause.source();
        }
        std::process::exit(1);
    }
}

async fn run() -> BenchResult<()> {
    let options = parse_options()?;
    println!("wgpu-nufft precision benchmark");
    println!(
        "configuration: precision={} cases={} kinds={} runs={} samples_per_run={} eps={} sigma={} seed={DATA_SEED:#010x}",
        join_tokens(&options.precisions, PrecisionChoice::name),
        join_tokens(&options.cases, CaseChoice::name),
        join_tokens(&options.kinds, TransformKind::name),
        options.runs,
        options.samples,
        DEFAULT_EPS,
        DEFAULT_SIGMA,
    );
    println!(
        "method: caller-owned inputs and outputs remain GPU-resident; plans are recreated for every run; each plan receives one untimed warmup followed by individually submitted samples"
    );
    println!(
        "timed scope: wall clock starts immediately before queue.submit and ends after device.poll(Wait); plan creation, command encoding, buffer upload, and readback are excluded; stderr is computed across plan-recreated run means"
    );
    println!(
        "precision ABI: f32 coordinates/complex values use f32/vec2<f32>; f64 uses f64/Complex64; df64 uses DoubleFloat/ComplexDoubleFloat (two-f32 words per scalar)"
    );

    let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
    descriptor.backends = wgpu::Backends::VULKAN;
    let instance = wgpu::Instance::new(descriptor);
    let adapter = select_vulkan_adapter(&instance, options.adapter_selector.as_deref()).await?;
    let adapter_info = adapter.get_info();
    let adapter_features = adapter.features();
    println!("adapter info:\n{adapter_info:#?}");
    println!("adapter features: {adapter_features:?}");
    println!("adapter limits:\n{:#?}", adapter.limits());

    let wants_f64 = options.precisions.contains(&PrecisionChoice::F64);
    let enables_f64 = wants_f64 && adapter_features.contains(wgpu::Features::SHADER_F64);
    if wants_f64 && !enables_f64 {
        println!(
            "NOTICE precision=f64 unavailable: Vulkan adapter does not expose SHADER_F64; f64 rows will be skipped"
        );
    }
    let required_features = if enables_f64 {
        wgpu::Features::SHADER_F64
    } else {
        wgpu::Features::empty()
    };
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("wgpu_nufft.precision_bench.device"),
            required_features,
            required_limits: adapter.limits(),
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
        })
        .await
        .map_err(|error| contextual_error("requesting precision benchmark device", error))?;
    println!("requested device features: {:?}", device.features());
    println!("device limits:\n{:#?}", device.limits());

    let result = run_selected_cases(&device, &queue, &options, enables_f64).await;
    #[cfg(windows)]
    std::mem::forget((queue, device, adapter, instance));
    #[cfg(not(windows))]
    drop((queue, device, adapter, instance));
    result
}

async fn run_selected_cases(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    options: &Options,
    enables_f64: bool,
) -> BenchResult<()> {
    for &precision_choice in &options.precisions {
        if precision_choice == PrecisionChoice::F64 && !enables_f64 {
            for &case in &options.cases {
                println!(
                    "SKIP precision=f64 case={} reason=adapter_missing_SHADER_F64",
                    case.name()
                );
            }
            continue;
        }
        for &case in &options.cases {
            if let Some(mode_shape) = case.mode_shape() {
                for &kind in &options.kinds {
                    run_normal_case(
                        device,
                        queue,
                        options,
                        precision_choice,
                        case,
                        kind,
                        &mode_shape,
                    )
                    .await?;
                }
            } else {
                run_type3_case(device, queue, options, precision_choice).await?;
            }
            device
                .poll(wgpu::PollType::Poll)
                .map_err(|error| contextual_error("polling after precision case", error))?;
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
async fn run_normal_case(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    options: &Options,
    precision_choice: PrecisionChoice,
    case: CaseChoice,
    kind: TransformKind,
    mode_shape: &[usize],
) -> BenchResult<()> {
    let precision = precision_choice.precision();
    let point_count = checked_product(mode_shape, "normal-case point count")?;
    let config = normal_config(mode_shape, precision);
    let metadata = kind.host_plan(config.clone())?;
    let buffers = create_normal_buffers(device, &metadata, kind, point_count, precision)?;
    println!(
        "\n=== kind={} precision={} case={} dimensions={} modes={} N={} M={} fine_grid={} ===",
        kind.name(),
        precision_choice.name(),
        case.name(),
        mode_shape.len(),
        shape_token(mode_shape),
        point_count,
        point_count,
        shape_token(metadata.fine_grid_shape()),
    );

    let statistics = match benchmark_normal(
        device,
        queue,
        options,
        precision_choice,
        kind,
        point_count,
        config,
        &buffers,
    )
    .await?
    {
        Some(statistics) => statistics,
        None => return Ok(()),
    };
    println!(
        "RESULT kind={} precision={} case={} dimensions={} modes={} N={} M={} fine_grid={} eps={} sigma={} runs={} samples_per_run={} raw_submit_wait_ms={:?} run_mean_submit_wait_ms={:?} avg_submit_wait_ms={:.6} stderr_submit_wait_ms={} stderr_basis=plan-recreated-run-means min_submit_wait_ms={:.6} plan_create_ms={:?} timing_scope=submit-through-device-poll setup_excluded=plan,encode,upload,readback gpu_resident=true",
        kind.name(),
        precision_choice.name(),
        case.name(),
        mode_shape.len(),
        shape_token(mode_shape),
        point_count,
        point_count,
        shape_token(metadata.fine_grid_shape()),
        DEFAULT_EPS,
        DEFAULT_SIGMA,
        options.runs,
        options.samples,
        statistics.raw_ms,
        statistics.run_means_ms,
        statistics.mean_ms,
        format_optional(statistics.stderr_ms),
        statistics.minimum_ms,
        statistics.plan_create_ms,
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
async fn benchmark_normal(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    options: &Options,
    precision_choice: PrecisionChoice,
    kind: TransformKind,
    point_count: usize,
    config: NufftConfig,
    buffers: &NormalBuffers,
) -> BenchResult<Option<Statistics>> {
    let mut raw_ms = Vec::with_capacity(options.runs * options.samples);
    let mut run_means_ms = Vec::with_capacity(options.runs);
    let mut plan_create_ms = Vec::with_capacity(options.runs);
    for run_index in 0..options.runs {
        let scopes = push_gpu_error_scopes(device);
        let plan_start = Instant::now();
        let plan_result = kind.gpu_plan(device, queue, config.clone());
        let creation_ms = plan_start.elapsed().as_secs_f64() * 1_000.0;
        pop_gpu_error_scopes(scopes, "creating normal precision benchmark plan").await?;
        let plan = match plan_result {
            Ok(plan) => plan,
            Err(error @ NufftError::PrecisionUnsupported { .. }) => {
                println!(
                    "SKIP kind={} precision={} reason={error}",
                    kind.name(),
                    precision_choice.name()
                );
                return Ok(None);
            }
            Err(error) => return Err(error.into()),
        };
        plan_create_ms.push(creation_ms);

        let warmup_context = format!(
            "{} {} run {} warmup",
            kind.name(),
            precision_choice.name(),
            run_index + 1
        );
        let warmup_ms = execute_normal_once(
            device,
            queue,
            &plan,
            kind,
            point_count,
            buffers,
            options.wait_timeout,
            &warmup_context,
        )
        .await?;
        println!(
            "run {}/{} plan_create_ms={creation_ms:.6} warmup_submit_wait_ms={warmup_ms:.6}",
            run_index + 1,
            options.runs,
        );

        let mut this_run = Vec::with_capacity(options.samples);
        for sample_index in 0..options.samples {
            let context = format!(
                "{} {} run {} sample {}",
                kind.name(),
                precision_choice.name(),
                run_index + 1,
                sample_index + 1
            );
            let elapsed_ms = execute_normal_once(
                device,
                queue,
                &plan,
                kind,
                point_count,
                buffers,
                options.wait_timeout,
                &context,
            )
            .await?;
            println!(
                "run {}/{} sample {}/{} submit_wait_ms={elapsed_ms:.6}",
                run_index + 1,
                options.runs,
                sample_index + 1,
                options.samples,
            );
            raw_ms.push(elapsed_ms);
            this_run.push(elapsed_ms);
        }
        run_means_ms.push(mean(&this_run)?);
    }
    Ok(Some(build_statistics(
        raw_ms,
        run_means_ms,
        plan_create_ms,
    )?))
}

#[allow(clippy::too_many_arguments)]
async fn execute_normal_once(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    plan: &NufftPlan,
    kind: TransformKind,
    point_count: usize,
    buffers: &NormalBuffers,
    wait_timeout: Duration,
    context: &str,
) -> BenchResult<f64> {
    let scopes = push_gpu_error_scopes(device);
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("wgpu_nufft.precision_bench.normal.encoder"),
    });
    if let Err(error) = kind.encode(plan, device, &mut encoder, point_count, buffers) {
        pop_gpu_error_scopes(scopes, &format!("encoding {context}")).await?;
        return Err(error.into());
    }
    let command_buffer = encoder.finish();
    let start = Instant::now();
    let submission = queue.submit([command_buffer]);
    let wait_result = wait_for_submission(device, submission, wait_timeout, context);
    let elapsed_ms = start.elapsed().as_secs_f64() * 1_000.0;
    let scope_result = pop_gpu_error_scopes(scopes, &format!("executing {context}")).await;
    wait_result?;
    scope_result?;
    validate_elapsed(elapsed_ms, context)
}

async fn run_type3_case(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    options: &Options,
    precision_choice: PrecisionChoice,
) -> BenchResult<()> {
    let precision = precision_choice.precision();
    let config = type3_config(precision);
    let metadata = NufftType3Plan::new(config.clone())?;
    let buffers = create_type3_buffers(device, &metadata, precision)?;
    println!(
        "\n=== kind=type-3 precision={} case=type3 dimensions=1 M={} K={} outer_grid={} ===",
        precision_choice.name(),
        TYPE3_SOURCE_COUNT,
        TYPE3_TARGET_COUNT,
        shape_token(&metadata.outer_grid_shape()),
    );

    let statistics =
        match benchmark_type3(device, queue, options, precision_choice, config, &buffers).await? {
            Some(statistics) => statistics,
            None => return Ok(()),
        };
    println!(
        "RESULT kind=type-3 precision={} case=type3 dimensions=1 M={} K={} outer_grid={} eps={} sigma={} runs={} samples_per_run={} raw_submit_wait_ms={:?} run_mean_submit_wait_ms={:?} avg_submit_wait_ms={:.6} stderr_submit_wait_ms={} stderr_basis=plan-recreated-run-means min_submit_wait_ms={:.6} plan_create_ms={:?} timing_scope=submit-through-device-poll setup_excluded=plan,encode,upload,readback gpu_resident=true source_bounds={} target_bounds={}",
        precision_choice.name(),
        TYPE3_SOURCE_COUNT,
        TYPE3_TARGET_COUNT,
        shape_token(&metadata.outer_grid_shape()),
        DEFAULT_EPS,
        DEFAULT_SIGMA,
        options.runs,
        options.samples,
        statistics.raw_ms,
        statistics.run_means_ms,
        statistics.mean_ms,
        format_optional(statistics.stderr_ms),
        statistics.minimum_ms,
        statistics.plan_create_ms,
        interval_token(metadata.config().source_bounds()),
        interval_token(metadata.config().target_bounds()),
    );
    Ok(())
}

async fn benchmark_type3(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    options: &Options,
    precision_choice: PrecisionChoice,
    config: NufftType3Config,
    buffers: &Type3Buffers,
) -> BenchResult<Option<Statistics>> {
    let mut raw_ms = Vec::with_capacity(options.runs * options.samples);
    let mut run_means_ms = Vec::with_capacity(options.runs);
    let mut plan_create_ms = Vec::with_capacity(options.runs);
    for run_index in 0..options.runs {
        let scopes = push_gpu_error_scopes(device);
        let plan_start = Instant::now();
        let plan_result = NufftType3Plan::new_gpu(device, queue, config.clone());
        let creation_ms = plan_start.elapsed().as_secs_f64() * 1_000.0;
        pop_gpu_error_scopes(scopes, "creating type-3 precision benchmark plan").await?;
        let plan = match plan_result {
            Ok(plan) => plan,
            Err(error @ NufftError::PrecisionUnsupported { .. }) => {
                println!(
                    "SKIP kind=type-3 precision={} reason={error}",
                    precision_choice.name()
                );
                return Ok(None);
            }
            Err(error) => return Err(error.into()),
        };
        plan_create_ms.push(creation_ms);

        let warmup_context = format!(
            "type-3 {} run {} warmup",
            precision_choice.name(),
            run_index + 1
        );
        let warmup_ms = execute_type3_once(
            device,
            queue,
            &plan,
            buffers,
            options.wait_timeout,
            &warmup_context,
        )
        .await?;
        println!(
            "run {}/{} plan_create_ms={creation_ms:.6} warmup_submit_wait_ms={warmup_ms:.6}",
            run_index + 1,
            options.runs,
        );

        let mut this_run = Vec::with_capacity(options.samples);
        for sample_index in 0..options.samples {
            let context = format!(
                "type-3 {} run {} sample {}",
                precision_choice.name(),
                run_index + 1,
                sample_index + 1
            );
            let elapsed_ms = execute_type3_once(
                device,
                queue,
                &plan,
                buffers,
                options.wait_timeout,
                &context,
            )
            .await?;
            println!(
                "run {}/{} sample {}/{} submit_wait_ms={elapsed_ms:.6}",
                run_index + 1,
                options.runs,
                sample_index + 1,
                options.samples,
            );
            raw_ms.push(elapsed_ms);
            this_run.push(elapsed_ms);
        }
        run_means_ms.push(mean(&this_run)?);
    }
    Ok(Some(build_statistics(
        raw_ms,
        run_means_ms,
        plan_create_ms,
    )?))
}

async fn execute_type3_once(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    plan: &NufftType3Plan,
    buffers: &Type3Buffers,
    wait_timeout: Duration,
    context: &str,
) -> BenchResult<f64> {
    let scopes = push_gpu_error_scopes(device);
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("wgpu_nufft.precision_bench.type3.encoder"),
    });
    if let Err(error) = plan.encode_gpu(
        device,
        &mut encoder,
        TYPE3_SOURCE_COUNT,
        &buffers.source_points,
        &buffers.strengths,
        TYPE3_TARGET_COUNT,
        &buffers.target_points,
        &buffers.output,
    ) {
        pop_gpu_error_scopes(scopes, &format!("encoding {context}")).await?;
        return Err(error.into());
    }
    let command_buffer = encoder.finish();
    let start = Instant::now();
    let submission = queue.submit([command_buffer]);
    let wait_result = wait_for_submission(device, submission, wait_timeout, context);
    let elapsed_ms = start.elapsed().as_secs_f64() * 1_000.0;
    let scope_result = pop_gpu_error_scopes(scopes, &format!("executing {context}")).await;
    wait_result?;
    scope_result?;
    validate_elapsed(elapsed_ms, context)
}

fn normal_config(mode_shape: &[usize], precision: FftPrecision) -> NufftConfig {
    NufftConfig::new(mode_shape.to_vec(), DEFAULT_EPS)
        .with_sigma(DEFAULT_SIGMA)
        .with_sign(NufftSign::Positive)
        .with_mode_order(ModeOrder::Centered)
        .with_precision(precision)
}

fn type3_config(precision: FftPrecision) -> NufftType3Config {
    let source = NufftInterval::new(
        TYPE3_SOURCE_CENTER - TYPE3_SOURCE_HALFWIDTH,
        TYPE3_SOURCE_CENTER + TYPE3_SOURCE_HALFWIDTH,
    );
    let target = NufftInterval::new(
        TYPE3_TARGET_CENTER - TYPE3_TARGET_HALFWIDTH,
        TYPE3_TARGET_CENTER + TYPE3_TARGET_HALFWIDTH,
    );
    NufftType3Config::new([source], [target], DEFAULT_EPS)
        .with_sigma(DEFAULT_SIGMA)
        .with_sign(NufftSign::Positive)
        .with_precision(precision)
}

fn create_normal_buffers(
    device: &wgpu::Device,
    metadata: &NufftPlan,
    kind: TransformKind,
    point_count: usize,
    precision: FftPrecision,
) -> BenchResult<NormalBuffers> {
    let dimensions = metadata.config().dimensions();
    let coordinates = generate_coordinates(
        dimensions,
        point_count,
        &COORDINATE_SEEDS,
        0.0,
        std::f64::consts::PI,
    )?;
    let input_count = match kind {
        TransformKind::Type1 => point_count,
        TransformKind::Type2 => metadata.config().mode_count()?,
    };
    let complex = generate_complex(input_count);
    let point_bytes = encode_coordinates(&coordinates, precision);
    let input_bytes = encode_complex(&complex, precision);
    let output_size = match kind {
        TransformKind::Type1 => metadata.required_type1_output_buffer_size_bytes()?,
        TransformKind::Type2 => {
            metadata.required_type2_output_buffer_size_bytes_for_batch(point_count, 1)?
        }
    };
    let expected_points = metadata.required_point_buffer_size_bytes(point_count)?;
    let expected_input = match kind {
        TransformKind::Type1 => {
            metadata.required_type1_strength_buffer_size_bytes_for_batch(point_count, 1)?
        }
        TransformKind::Type2 => metadata.required_type2_coefficient_buffer_size_bytes()?,
    };
    validate_encoded_size("normal point buffer", point_bytes.len(), expected_points)?;
    validate_encoded_size("normal input buffer", input_bytes.len(), expected_input)?;

    let points =
        create_initialized_buffer(device, "wgpu_nufft.precision_bench.points", &point_bytes);
    let input = create_initialized_buffer(device, "wgpu_nufft.precision_bench.input", &input_bytes);
    let output = create_output_buffer(device, "wgpu_nufft.precision_bench.output", output_size);
    Ok(NormalBuffers {
        points,
        input,
        output,
    })
}

fn create_type3_buffers(
    device: &wgpu::Device,
    metadata: &NufftType3Plan,
    precision: FftPrecision,
) -> BenchResult<Type3Buffers> {
    let source_points = generate_coordinates(
        1,
        TYPE3_SOURCE_COUNT,
        &COORDINATE_SEEDS,
        TYPE3_SOURCE_CENTER,
        TYPE3_SOURCE_HALFWIDTH,
    )?;
    let target_points = generate_coordinates(
        1,
        TYPE3_TARGET_COUNT,
        &TARGET_SEEDS,
        TYPE3_TARGET_CENTER,
        TYPE3_TARGET_HALFWIDTH,
    )?;
    let strengths = generate_complex(TYPE3_SOURCE_COUNT);
    let source_bytes = encode_coordinates(&source_points, precision);
    let target_bytes = encode_coordinates(&target_points, precision);
    let strength_bytes = encode_complex(&strengths, precision);
    validate_encoded_size(
        "type-3 source point buffer",
        source_bytes.len(),
        metadata.required_source_point_buffer_size_bytes(TYPE3_SOURCE_COUNT)?,
    )?;
    validate_encoded_size(
        "type-3 target point buffer",
        target_bytes.len(),
        metadata.required_target_point_buffer_size_bytes(TYPE3_TARGET_COUNT)?,
    )?;
    validate_encoded_size(
        "type-3 strength buffer",
        strength_bytes.len(),
        metadata.required_strength_buffer_size_bytes(TYPE3_SOURCE_COUNT)?,
    )?;
    let output_size = metadata.required_output_buffer_size_bytes(TYPE3_TARGET_COUNT)?;
    Ok(Type3Buffers {
        source_points: create_initialized_buffer(
            device,
            "wgpu_nufft.precision_bench.type3.source_points",
            &source_bytes,
        ),
        target_points: create_initialized_buffer(
            device,
            "wgpu_nufft.precision_bench.type3.target_points",
            &target_bytes,
        ),
        strengths: create_initialized_buffer(
            device,
            "wgpu_nufft.precision_bench.type3.strengths",
            &strength_bytes,
        ),
        output: create_output_buffer(
            device,
            "wgpu_nufft.precision_bench.type3.output",
            output_size,
        ),
    })
}

fn create_initialized_buffer(device: &wgpu::Device, label: &str, bytes: &[u8]) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytes,
        usage: wgpu::BufferUsages::STORAGE,
    })
}

fn create_output_buffer(device: &wgpu::Device, label: &str, size: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    })
}

fn encode_coordinates(values: &[f64], precision: FftPrecision) -> Vec<u8> {
    match precision {
        FftPrecision::F32 => {
            let packed = values.iter().map(|&value| value as f32).collect::<Vec<_>>();
            bytemuck::cast_slice(&packed).to_vec()
        }
        FftPrecision::F64 => bytemuck::cast_slice(values).to_vec(),
        FftPrecision::Df64 => {
            let packed = values
                .iter()
                .map(|&value| DoubleFloat::from_f64(value))
                .collect::<Vec<_>>();
            bytemuck::cast_slice(&packed).to_vec()
        }
    }
}

fn encode_complex(values: &[(f64, f64)], precision: FftPrecision) -> Vec<u8> {
    match precision {
        FftPrecision::F32 => {
            let packed = values
                .iter()
                .map(|&(re, im)| Complex32 {
                    re: re as f32,
                    im: im as f32,
                })
                .collect::<Vec<_>>();
            bytemuck::cast_slice(&packed).to_vec()
        }
        FftPrecision::F64 => {
            let packed = values
                .iter()
                .map(|&(re, im)| Complex64::new(re, im))
                .collect::<Vec<_>>();
            bytemuck::cast_slice(&packed).to_vec()
        }
        FftPrecision::Df64 => {
            let packed = values
                .iter()
                .map(|&(re, im)| ComplexDoubleFloat::from_f64(re, im))
                .collect::<Vec<_>>();
            bytemuck::cast_slice(&packed).to_vec()
        }
    }
}

fn generate_coordinates(
    dimensions: usize,
    point_count: usize,
    seeds: &[u32; 3],
    center: f64,
    halfwidth: f64,
) -> BenchResult<Vec<f64>> {
    if dimensions == 0 || dimensions > seeds.len() {
        return Err(input_error(format!(
            "coordinate generation supports 1..={} dimensions, got {dimensions}",
            seeds.len()
        )));
    }
    let capacity = point_count
        .checked_mul(dimensions)
        .ok_or_else(|| input_error("coordinate count overflow"))?;
    let mut streams = seeds[..dimensions]
        .iter()
        .map(|&seed| Lcg32::new(DATA_SEED ^ seed))
        .collect::<Vec<_>>();
    let mut result = Vec::with_capacity(capacity);
    for _ in 0..point_count {
        for stream in &mut streams {
            let unit = stream.next_unit_interval();
            result.push(center + (2.0 * unit - 1.0) * halfwidth);
        }
    }
    Ok(result)
}

fn generate_complex(count: usize) -> Vec<(f64, f64)> {
    let mut real = Lcg32::new(DATA_SEED ^ COMPLEX_REAL_SEED);
    let mut imag = Lcg32::new(DATA_SEED ^ COMPLEX_IMAG_SEED);
    (0..count)
        .map(|_| {
            (
                2.0 * real.next_unit_interval() - 1.0,
                2.0 * imag.next_unit_interval() - 1.0,
            )
        })
        .collect()
}

struct Lcg32 {
    state: u32,
}

impl Lcg32 {
    const fn new(seed: u32) -> Self {
        Self { state: seed }
    }

    fn next_unit_interval(&mut self) -> f64 {
        self.state = self
            .state
            .wrapping_mul(1_664_525)
            .wrapping_add(1_013_904_223);
        f64::from(self.state >> 8) * (1.0 / 16_777_216.0)
    }
}

async fn select_vulkan_adapter(
    instance: &wgpu::Instance,
    selector: Option<&str>,
) -> BenchResult<wgpu::Adapter> {
    let mut adapters = instance
        .enumerate_adapters(wgpu::Backends::VULKAN)
        .await
        .into_iter()
        .map(|adapter| {
            let info = adapter.get_info();
            (adapter, info)
        })
        .collect::<Vec<_>>();
    if adapters.is_empty() {
        return Err(input_error("no Vulkan adapters were found"));
    }
    println!("enumerated Vulkan adapters:");
    for (index, (_, info)) in adapters.iter().enumerate() {
        println!(
            "  index={index} name={:?} type={:?} backend={:?}",
            info.name, info.device_type, info.backend
        );
    }
    let selected = if let Some(selector) = selector {
        if let Ok(index) = selector.parse::<usize>() {
            if index >= adapters.len() {
                return Err(input_error(format!(
                    "adapter index {index} is outside 0..{}",
                    adapters.len()
                )));
            }
            index
        } else {
            let needle = selector.to_ascii_lowercase();
            let matches = adapters
                .iter()
                .enumerate()
                .filter_map(|(index, (_, info))| {
                    info.name
                        .to_ascii_lowercase()
                        .contains(&needle)
                        .then_some(index)
                })
                .collect::<Vec<_>>();
            match matches.as_slice() {
                [index] => *index,
                [] => {
                    return Err(input_error(format!(
                        "adapter selector {selector:?} matched no Vulkan adapter"
                    )));
                }
                _ => {
                    return Err(input_error(format!(
                        "adapter selector {selector:?} is ambiguous: {matches:?}"
                    )));
                }
            }
        }
    } else {
        adapters
            .iter()
            .position(|(_, info)| info.device_type == wgpu::DeviceType::DiscreteGpu)
            .or_else(|| {
                adapters
                    .iter()
                    .position(|(_, info)| info.device_type == wgpu::DeviceType::IntegratedGpu)
            })
            .ok_or_else(|| input_error("no Vulkan hardware adapter was found"))?
    };
    let (adapter, info) = adapters.swap_remove(selected);
    println!("selected adapter index={selected} name={:?}", info.name);
    Ok(adapter)
}

fn parse_options() -> BenchResult<Options> {
    let mut adapter_selector = None;
    let mut precisions = Vec::new();
    let mut cases = Vec::new();
    let mut kinds = Vec::new();
    let mut runs = DEFAULT_RUNS;
    let mut samples = DEFAULT_SAMPLES;
    let mut wait_timeout = DEFAULT_WAIT_TIMEOUT;
    let mut arguments = std::env::args().skip(1).filter(|arg| arg != "--bench");
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--precision" => {
                let value = next_value(&mut arguments, "--precision")?;
                append_precisions(&mut precisions, &value)?;
            }
            "--case" => {
                let value = next_value(&mut arguments, "--case")?;
                append_cases(&mut cases, &value)?;
            }
            "--kind" => {
                let value = next_value(&mut arguments, "--kind")?;
                append_kinds(&mut kinds, &value)?;
            }
            "--adapter" => {
                adapter_selector = Some(next_value(&mut arguments, "--adapter")?);
            }
            "--runs" => {
                runs = parse_positive_usize(&next_value(&mut arguments, "--runs")?, "runs")?;
            }
            "--samples" => {
                samples =
                    parse_positive_usize(&next_value(&mut arguments, "--samples")?, "samples")?;
            }
            "--wait-timeout-secs" => {
                let seconds = next_value(&mut arguments, "--wait-timeout-secs")?
                    .parse::<u64>()
                    .map_err(|_| input_error("wait timeout must be a positive integer"))?;
                if seconds == 0 {
                    return Err(input_error("wait timeout must be greater than zero"));
                }
                wait_timeout = Duration::from_secs(seconds);
            }
            "--help" | "-h" => {
                print_usage();
                std::process::exit(0);
            }
            _ => return Err(input_error(format!("unknown argument {argument:?}"))),
        }
    }
    if precisions.is_empty() {
        precisions.extend(PrecisionChoice::ALL);
    }
    if cases.is_empty() {
        cases.extend(CaseChoice::ALL);
    }
    if kinds.is_empty() {
        kinds.extend(TransformKind::ALL);
    }
    Ok(Options {
        adapter_selector,
        precisions,
        cases,
        kinds,
        runs,
        samples,
        wait_timeout,
    })
}

fn append_precisions(destination: &mut Vec<PrecisionChoice>, value: &str) -> BenchResult<()> {
    for token in value.split(',') {
        match token.to_ascii_lowercase().as_str() {
            "f32" => push_unique(destination, PrecisionChoice::F32),
            "f64" => push_unique(destination, PrecisionChoice::F64),
            "df64" => push_unique(destination, PrecisionChoice::Df64),
            "all" => {
                for precision in PrecisionChoice::ALL {
                    push_unique(destination, precision);
                }
            }
            _ => {
                return Err(input_error(format!(
                    "unknown precision {token:?}; expected f32, f64, df64, or all"
                )));
            }
        }
    }
    Ok(())
}

fn append_cases(destination: &mut Vec<CaseChoice>, value: &str) -> BenchResult<()> {
    for token in value.split(',') {
        match token.to_ascii_lowercase().as_str() {
            "1d" => push_unique(destination, CaseChoice::OneD),
            "2d" => push_unique(destination, CaseChoice::TwoD),
            "3d" => push_unique(destination, CaseChoice::ThreeD),
            "type3" | "type-3" => push_unique(destination, CaseChoice::Type3),
            "all" => {
                for case in CaseChoice::ALL {
                    push_unique(destination, case);
                }
            }
            _ => {
                return Err(input_error(format!(
                    "unknown case {token:?}; expected 1d, 2d, 3d, type3, or all"
                )));
            }
        }
    }
    Ok(())
}

fn append_kinds(destination: &mut Vec<TransformKind>, value: &str) -> BenchResult<()> {
    for token in value.split(',') {
        match token.to_ascii_lowercase().as_str() {
            "type1" | "type-1" => push_unique(destination, TransformKind::Type1),
            "type2" | "type-2" => push_unique(destination, TransformKind::Type2),
            "both" | "all" => {
                for kind in TransformKind::ALL {
                    push_unique(destination, kind);
                }
            }
            _ => {
                return Err(input_error(format!(
                    "unknown kind {token:?}; expected type1, type2, or both"
                )));
            }
        }
    }
    Ok(())
}

fn push_unique<T: PartialEq>(destination: &mut Vec<T>, value: T) {
    if !destination.contains(&value) {
        destination.push(value);
    }
}

fn print_usage() {
    println!(
        r#"Usage:
  cargo bench -p wgpu-nufft --bench nufft_precision_bench -- [options]

Options:
  --precision <f32|f64|df64|all>  Precision(s), comma-separated or repeated (default all).
  --case <1d|2d|3d|type3|all>     Case(s), comma-separated or repeated (default all).
  --kind <type1|type2|both>       Normal-transform kind(s) (default both).
  --adapter <index-or-name>       Select a Vulkan hardware adapter.
  --runs <count>                  Plan recreations per row (default 2).
  --samples <count>               Timed submissions per plan (default 3).
  --wait-timeout-secs <secs>      Per-submission timeout (default 120).
  --help                          Show this help.

Default normal cases use N=M: 1D 262144, 2D 512x512, and 3D 64^3.
The moderate 1D type-3 case uses M=K=65536. One untimed warmup is run per
recreated plan. Timings cover queue.submit through device.poll(Wait) only;
planning, encoding, upload, and readback are excluded. Df64 rows are reported
as structured skips until the portable NUFFT kernels are enabled."#
    );
}

fn push_gpu_error_scopes(
    device: &wgpu::Device,
) -> (
    wgpu::ErrorScopeGuard,
    wgpu::ErrorScopeGuard,
    wgpu::ErrorScopeGuard,
) {
    let out_of_memory = device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
    let internal = device.push_error_scope(wgpu::ErrorFilter::Internal);
    let validation = device.push_error_scope(wgpu::ErrorFilter::Validation);
    (out_of_memory, internal, validation)
}

async fn pop_gpu_error_scopes(
    (out_of_memory, internal, validation): (
        wgpu::ErrorScopeGuard,
        wgpu::ErrorScopeGuard,
        wgpu::ErrorScopeGuard,
    ),
    context: &str,
) -> BenchResult<()> {
    let validation_error = validation.pop().await;
    let internal_error = internal.pop().await;
    let out_of_memory_error = out_of_memory.pop().await;
    if let Some(error) = validation_error {
        return Err(contextual_error(
            format!("{context}: wgpu validation error"),
            error,
        ));
    }
    if let Some(error) = internal_error {
        return Err(contextual_error(
            format!("{context}: wgpu internal error"),
            error,
        ));
    }
    if let Some(error) = out_of_memory_error {
        return Err(contextual_error(
            format!("{context}: wgpu out-of-memory error"),
            error,
        ));
    }
    Ok(())
}

fn wait_for_submission(
    device: &wgpu::Device,
    submission: wgpu::SubmissionIndex,
    timeout: Duration,
    context: &str,
) -> BenchResult<()> {
    device
        .poll(wgpu::PollType::Wait {
            submission_index: Some(submission),
            timeout: Some(timeout),
        })
        .map(|_| ())
        .map_err(|error| contextual_error(format!("waiting for {context}"), error))
}

fn build_statistics(
    raw_ms: Vec<f64>,
    run_means_ms: Vec<f64>,
    plan_create_ms: Vec<f64>,
) -> BenchResult<Statistics> {
    let mean_ms = mean(&run_means_ms)?;
    let minimum_ms = raw_ms.iter().copied().fold(f64::INFINITY, f64::min);
    let stderr_ms = standard_error(&run_means_ms)?;
    Ok(Statistics {
        raw_ms,
        run_means_ms,
        mean_ms,
        stderr_ms,
        minimum_ms,
        plan_create_ms,
    })
}

fn mean(values: &[f64]) -> BenchResult<f64> {
    if values.is_empty() || values.iter().any(|value| !value.is_finite()) {
        return Err(input_error("statistics require finite, nonempty samples"));
    }
    Ok(values.iter().sum::<f64>() / values.len() as f64)
}

fn standard_error(values: &[f64]) -> BenchResult<Option<f64>> {
    let average = mean(values)?;
    if values.len() < 2 {
        return Ok(None);
    }
    let variance = values
        .iter()
        .map(|value| (value - average).powi(2))
        .sum::<f64>()
        / (values.len() - 1) as f64;
    Ok(Some(variance.sqrt() / (values.len() as f64).sqrt()))
}

fn validate_elapsed(elapsed_ms: f64, context: &str) -> BenchResult<f64> {
    if elapsed_ms.is_finite() && elapsed_ms > 0.0 {
        Ok(elapsed_ms)
    } else {
        Err(input_error(format!(
            "invalid submit-to-wait duration for {context}: {elapsed_ms} ms"
        )))
    }
}

fn validate_encoded_size(context: &str, actual: usize, expected: u64) -> BenchResult<()> {
    if u64::try_from(actual).ok() == Some(expected) {
        Ok(())
    } else {
        Err(input_error(format!(
            "{context} ABI mismatch: encoded {actual} bytes, plan requires {expected}"
        )))
    }
}

fn checked_product(values: &[usize], context: &'static str) -> BenchResult<usize> {
    values.iter().try_fold(1usize, |product, &value| {
        product
            .checked_mul(value)
            .ok_or_else(|| input_error(format!("{context} overflow")))
    })
}

fn shape_token(shape: &[usize]) -> String {
    shape
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join("x")
}

fn interval_token(intervals: &[NufftInterval]) -> String {
    intervals
        .iter()
        .map(|interval| format!("{:.9}:{:.9}", interval.lower(), interval.upper()))
        .collect::<Vec<_>>()
        .join("x")
}

fn join_tokens<T: Copy>(values: &[T], name: impl Fn(T) -> &'static str) -> String {
    values
        .iter()
        .copied()
        .map(name)
        .collect::<Vec<_>>()
        .join(",")
}

fn format_optional(value: Option<f64>) -> String {
    value.map_or_else(|| "undefined".to_owned(), |value| format!("{value:.6}"))
}

fn next_value(arguments: &mut impl Iterator<Item = String>, option: &str) -> BenchResult<String> {
    arguments
        .next()
        .ok_or_else(|| input_error(format!("{option} requires a value")))
}

fn parse_positive_usize(value: &str, label: &str) -> BenchResult<usize> {
    let parsed = value
        .parse::<usize>()
        .map_err(|_| input_error(format!("{label} must be a positive integer")))?;
    if parsed == 0 {
        return Err(input_error(format!("{label} must be greater than zero")));
    }
    Ok(parsed)
}

fn input_error(message: impl Into<String>) -> Box<dyn Error> {
    io::Error::new(io::ErrorKind::InvalidInput, message.into()).into()
}

fn contextual_error(context: impl Into<String>, source: impl Error + 'static) -> Box<dyn Error> {
    Box::new(ContextError {
        context: context.into(),
        source: Box::new(source),
    })
}

#[derive(Debug)]
struct ContextError {
    context: String,
    source: Box<dyn Error>,
}

impl fmt::Display for ContextError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}", self.context)
    }
}

impl Error for ContextError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        Some(self.source.as_ref())
    }
}
