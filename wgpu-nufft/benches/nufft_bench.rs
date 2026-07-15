#![cfg_attr(target_arch = "wasm32", allow(dead_code, unused_imports))]

#[cfg(target_arch = "wasm32")]
fn main() {}

use std::error::Error;
use std::fmt;
use std::io;
use std::time::{Duration, Instant};

use wgpu::util::DeviceExt;
use wgpu_nufft::{
    ModeOrder, NufftConfig, NufftInterval, NufftPlan, NufftSign, NufftType3Config, NufftType3Plan,
    DEFAULT_EPS, DEFAULT_SIGMA,
};

type BenchResult<T> = Result<T, Box<dyn Error>>;

const DEFAULT_CASE_SIZES: [usize; 2] = [262_144, 1_048_576];
const DEFAULT_3D_CASE_EDGES: [usize; 2] = [64, 128];
const DEFAULT_TYPE3_DIMENSIONS: [usize; 3] = [1, 2, 3];
const DEFAULT_TYPE3_SOURCE_COUNT: usize = 65_536;
const DEFAULT_TYPE3_TARGET_COUNT: usize = 65_536;
const DEFAULT_TYPE3_SOURCE_HALFWIDTH: f64 = std::f64::consts::PI;
const DEFAULT_TYPE3_TARGET_HALFWIDTH: f64 = 16.0;
const DEFAULT_TYPE3_SOURCE_CENTER: f64 = std::f64::consts::PI / 4.0;
const DEFAULT_TYPE3_TARGET_CENTER: f64 = 4.0;
const DEFAULT_RUNS: usize = 3;
const DEFAULT_SAMPLES: usize = 10;
const TYPE2_TRANSFORMS_PER_SAMPLE: usize = 32;
const DEFAULT_WAIT_TIMEOUT: Duration = Duration::from_secs(120);
const DATA_SEED: u32 = 0x4E55_4646;
const POINT_SEED_MASK: u32 = 0xA341_316C;
const STRENGTH_REAL_SEED_MASK: u32 = 0xC801_3EA4;
const STRENGTH_IMAG_SEED_MASK: u32 = 0xAD90_777D;
const MODE_REAL_SEED_MASK: u32 = 0x7E95_761E;
const MODE_IMAG_SEED_MASK: u32 = 0x6C8E_9CF5;
const POINT_Y_SEED_MASK: u32 = 0xB7E1_5162;
const POINT_Z_SEED_MASK: u32 = 0x9E37_79B9;
const SOURCE_POINT_SEED_MASKS: [u32; 3] = [POINT_SEED_MASK, POINT_Y_SEED_MASK, POINT_Z_SEED_MASK];
const TYPE3_TARGET_SEED_MASKS: [u32; 3] = [0x243F_6A88, 0x85A3_08D3, 0x1319_8A2E];

#[derive(Debug)]
struct Options {
    adapter_selector: Option<String>,
    runs: usize,
    samples: usize,
    wait_timeout: Duration,
    run_1d: bool,
    run_3d: bool,
    run_type3: bool,
}

#[derive(Debug, Clone)]
struct Type12Case {
    mode_shape: Vec<usize>,
    point_count: usize,
}

impl Type12Case {
    fn dimensions(&self) -> usize {
        self.mode_shape.len()
    }

    fn mode_count(&self) -> BenchResult<usize> {
        checked_product(&self.mode_shape, "benchmark mode count")
    }

    fn shape_token(&self) -> String {
        shape_token(&self.mode_shape)
    }
}

#[derive(Debug, Clone)]
struct Type3Case {
    dimensions: usize,
    source_count: usize,
    target_count: usize,
    config: NufftType3Config,
}

impl Type3Case {
    fn shape_token(&self) -> String {
        format!("M{}-K{}", self.source_count, self.target_count)
    }
}

#[derive(Debug, Clone, Copy)]
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

    const fn transforms_per_sample(self, dimensions: usize) -> usize {
        if dimensions > 1 {
            1
        } else {
            match self {
                Self::Type1 => 1,
                Self::Type2 => TYPE2_TRANSFORMS_PER_SAMPLE,
            }
        }
    }

    fn create_plan(
        self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        config: NufftConfig,
    ) -> BenchResult<NufftPlan> {
        match self {
            Self::Type1 => Ok(NufftPlan::type1_gpu(device, queue, config)?),
            Self::Type2 => Ok(NufftPlan::type2_gpu(device, queue, config)?),
        }
    }

    fn encode(
        self,
        plan: &NufftPlan,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        point_count: usize,
        buffers: &CaseBuffers,
    ) -> BenchResult<()> {
        match self {
            Self::Type1 => plan.encode_type1_gpu(
                device,
                encoder,
                point_count,
                &buffers.points,
                &buffers.strengths,
                &buffers.type1_output,
            )?,
            Self::Type2 => plan.encode_type2_gpu(
                device,
                encoder,
                point_count,
                &buffers.points,
                &buffers.modes,
                &buffers.type2_output,
            )?,
        }
        Ok(())
    }
}

struct CaseBuffers {
    points: wgpu::Buffer,
    strengths: wgpu::Buffer,
    modes: wgpu::Buffer,
    type1_output: wgpu::Buffer,
    type2_output: wgpu::Buffer,
}

struct Type3CaseBuffers {
    source_points: wgpu::Buffer,
    target_points: wgpu::Buffer,
    strengths: wgpu::Buffer,
    output: wgpu::Buffer,
}

struct TransformExecution<'a> {
    device: &'a wgpu::Device,
    queue: &'a wgpu::Queue,
    plan: &'a NufftPlan,
    buffers: &'a CaseBuffers,
    point_count: usize,
    kind: TransformKind,
    wait_timeout: Duration,
}

struct Type3Execution<'a> {
    device: &'a wgpu::Device,
    queue: &'a wgpu::Queue,
    plan: &'a NufftType3Plan,
    buffers: &'a Type3CaseBuffers,
    source_count: usize,
    target_count: usize,
    wait_timeout: Duration,
}

#[derive(Debug)]
struct Statistics {
    mean: f64,
    stderr: Option<f64>,
    minimum: f64,
}

#[cfg(not(target_arch = "wasm32"))]
fn main() {
    if let Err(error) = pollster::block_on(run()) {
        eprintln!("wgpu-nufft benchmark failed: {error}");
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
    if options.run_1d && !options.run_3d && !options.run_type3 {
        println!("wgpu-nufft 1D GPU benchmark");
        println!(
            "configuration: sizes={:?} point_count_equals_mode_count=true runs={} samples_per_run={} seed={:#010x} data_streams=field-specific-xor eps={} sigma={} sign=positive mode_order=centered adapter_selector={} wait_timeout_secs={}",
            DEFAULT_CASE_SIZES,
            options.runs,
            options.samples,
            DATA_SEED,
            DEFAULT_EPS,
            DEFAULT_SIGMA,
            options
                .adapter_selector
                .as_deref()
                .unwrap_or("auto-single-hardware"),
            options.wait_timeout.as_secs(),
        );
        println!(
            "method: reusable plan; caller buffers initialized once; one untimed warmup; type-1 uses one transform per timed submit, while type-2 batches {TYPE2_TRANSFORMS_PER_SAMPLE} transforms per timed submit to amortize queue-wait jitter; headline wall clock starts immediately before queue.submit and ends after device.poll(Wait), then is divided per transform"
        );
    } else {
        println!("wgpu-nufft GPU benchmark");
        println!(
            "configuration: suites={} 1d_sizes={:?} 3d_edges={:?} type3_dimensions={:?} type3_M={} type3_K={} type3_source_center={:.9} type3_source_halfwidth={:.9} type3_target_center={:.9} type3_target_halfwidth={:.9} point_count_equals_mode_count=true runs={} samples_per_run={} seed={:#010x} data_streams=field-specific-xor eps={} sigma={} sign=positive mode_order=centered adapter_selector={} wait_timeout_secs={}",
            selected_suites_token(&options),
            DEFAULT_CASE_SIZES,
            DEFAULT_3D_CASE_EDGES,
            DEFAULT_TYPE3_DIMENSIONS,
            DEFAULT_TYPE3_SOURCE_COUNT,
            DEFAULT_TYPE3_TARGET_COUNT,
            DEFAULT_TYPE3_SOURCE_CENTER,
            DEFAULT_TYPE3_SOURCE_HALFWIDTH,
            DEFAULT_TYPE3_TARGET_CENTER,
            DEFAULT_TYPE3_TARGET_HALFWIDTH,
            options.runs,
            options.samples,
            DATA_SEED,
            DEFAULT_EPS,
            DEFAULT_SIGMA,
            options
                .adapter_selector
                .as_deref()
                .unwrap_or("auto-single-hardware"),
            options.wait_timeout.as_secs(),
        );
        println!(
            "method: reusable plan recreated for each run; caller buffers initialized once; one untimed warmup; 1D type-2 batches {TYPE2_TRANSFORMS_PER_SAMPLE} transforms per submit while all multidimensional and type-3 cases use one; headline wall clock starts immediately before queue.submit and ends after device.poll(Wait), then is divided per transform"
        );
    }
    println!(
        "scope: plan creation, command encoding, upload, and readback excluded from headline; host command-encoding time is reported separately; no output readback is performed"
    );
    println!(
        "comparison caveat: these are f32 GPU execution measurements; any FINUFFT CPU results are contextual CPU-versus-GPU data, not an apples-to-apples device comparison"
    );

    let mut instance_descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
    instance_descriptor.backends = wgpu::Backends::VULKAN;
    let instance = wgpu::Instance::new(instance_descriptor);
    let adapter = select_vulkan_adapter(&instance, options.adapter_selector.as_deref()).await?;
    let adapter_info = adapter.get_info();
    if adapter_info.backend != wgpu::Backend::Vulkan {
        return Err(input_error(format!(
            "requested Vulkan-only execution but selected backend {:?}",
            adapter_info.backend
        )));
    }
    let adapter_limits = adapter.limits();
    println!("adapter info:\n{adapter_info:#?}");
    println!("adapter limits:\n{adapter_limits:#?}");
    println!(
        "ADAPTER_RESULT backend={:?} name={:?} vendor={:#x} device={:#x} device_type={:?} driver={:?} driver_info={:?}",
        adapter_info.backend,
        adapter_info.name,
        adapter_info.vendor,
        adapter_info.device,
        adapter_info.device_type,
        adapter_info.driver,
        adapter_info.driver_info,
    );

    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("wgpu_nufft.bench.device"),
            required_features: wgpu::Features::empty(),
            required_limits: adapter_limits,
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
        })
        .await
        .map_err(|error| contextual_error("requesting benchmark device", error))?;
    println!("device limits:\n{:#?}", device.limits());

    let result = run_selected_cases(&device, &queue, &options).await;

    // Native wgpu teardown can stall on Windows after large benchmark runs.
    // The process exits immediately after this function returns.
    #[cfg(windows)]
    std::mem::forget((queue, device, adapter, instance));
    #[cfg(not(windows))]
    drop((queue, device, adapter, instance));
    result
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
            "  index={} name={:?} vendor={:#x} device={:#x} type={:?} driver={:?}",
            index, info.name, info.vendor, info.device, info.device_type, info.driver
        );
    }

    let selected_position = if let Some(selector) = selector {
        if let Some(index) = selector
            .parse::<usize>()
            .ok()
            .filter(|&index| index < adapters.len())
        {
            index
        } else {
            let selector_lower = selector.to_ascii_lowercase();
            let matches = adapters
                .iter()
                .enumerate()
                .filter_map(|(index, (_, info))| {
                    info.name
                        .to_ascii_lowercase()
                        .contains(&selector_lower)
                        .then_some(index)
                })
                .collect::<Vec<_>>();
            match matches.as_slice() {
                [index] => *index,
                [] => {
                    return Err(input_error(format!(
                        "adapter selector {selector:?} did not match any Vulkan adapter"
                    )))
                }
                _ => {
                    return Err(input_error(format!(
                        "adapter selector {selector:?} is ambiguous; matched indices {matches:?}"
                    )))
                }
            }
        }
    } else {
        let hardware = adapters
            .iter()
            .enumerate()
            .filter_map(|(index, (_, info))| is_hardware_adapter(info).then_some(index))
            .collect::<Vec<_>>();
        match hardware.as_slice() {
            [index] => *index,
            [] => return Err(input_error("no native Vulkan hardware adapter was found")),
            _ => {
                return Err(input_error(format!(
                    "multiple native Vulkan hardware adapters were found at indices {hardware:?}; select one with --adapter <index-or-name>"
                )))
            }
        }
    };

    let (adapter, info) = adapters.swap_remove(selected_position);
    if !is_hardware_adapter(&info) {
        return Err(input_error(format!(
            "refusing non-hardware Vulkan adapter {:?} ({:?})",
            info.name, info.device_type
        )));
    }
    println!(
        "selected Vulkan adapter index={} name={:?} vendor={:#x} device={:#x} type={:?}",
        selected_position, info.name, info.vendor, info.device, info.device_type
    );
    Ok(adapter)
}

fn is_hardware_adapter(info: &wgpu::AdapterInfo) -> bool {
    matches!(
        info.device_type,
        wgpu::DeviceType::DiscreteGpu | wgpu::DeviceType::IntegratedGpu
    )
}

async fn run_selected_cases(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    options: &Options,
) -> BenchResult<()> {
    if options.run_1d {
        let cases = DEFAULT_CASE_SIZES.map(|size| Type12Case {
            mode_shape: vec![size],
            point_count: size,
        });
        run_type12_cases(device, queue, options, &cases).await?;
    }
    if options.run_3d {
        let cases = DEFAULT_3D_CASE_EDGES.map(|edge| {
            let count = edge * edge * edge;
            Type12Case {
                mode_shape: vec![edge; 3],
                point_count: count,
            }
        });
        run_type12_cases(device, queue, options, &cases).await?;
    }
    if options.run_type3 {
        run_type3_cases(device, queue, options).await?;
    }
    Ok(())
}

async fn run_type12_cases(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    options: &Options,
    cases: &[Type12Case],
) -> BenchResult<()> {
    for case in cases {
        let config = benchmark_config(&case.mode_shape);
        let buffers = create_case_buffers(device, case)?;
        let mode_count = case.mode_count()?;
        let point_bytes = checked_buffer_bytes(
            case.point_count
                .checked_mul(case.dimensions())
                .ok_or_else(|| input_error("benchmark point-coordinate count overflow"))?,
            4,
            "benchmark point-coordinate buffer",
        )?;
        let complex_bytes =
            checked_buffer_bytes(case.point_count, 8, "benchmark complex input buffer")?;
        println!(
            "\n=== case dimensions={} mode_shape={} N={} M={} input_points_bytes={} input_complex_bytes={} ===",
            case.dimensions(),
            case.shape_token(),
            mode_count,
            case.point_count,
            point_bytes,
            complex_bytes,
        );

        for kind in TransformKind::ALL {
            run_kind(device, queue, &buffers, &config, case, kind, options).await?;
        }

        device
            .poll(wgpu::PollType::Poll)
            .map_err(|error| contextual_error("polling after benchmark case", error))?;
    }
    Ok(())
}

async fn run_kind(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    buffers: &CaseBuffers,
    config: &NufftConfig,
    case: &Type12Case,
    kind: TransformKind,
    options: &Options,
) -> BenchResult<()> {
    let mut submit_wait_samples = Vec::with_capacity(options.runs * options.samples);
    let mut encoding_samples = Vec::with_capacity(options.runs * options.samples);
    let mut run_submit_wait_means = Vec::with_capacity(options.runs);
    let mut run_encoding_means = Vec::with_capacity(options.runs);
    let mut plan_creation_samples = Vec::with_capacity(options.runs);
    let mut fine_grid_shape = None;

    let transforms_per_sample = kind.transforms_per_sample(case.dimensions());
    println!(
        "\n{}: starting transforms_per_sample={transforms_per_sample}",
        kind.name()
    );
    for run_index in 0..options.runs {
        let plan_scopes = push_gpu_error_scopes(device);
        let plan_start = Instant::now();
        let plan = kind.create_plan(device, queue, config.clone());
        let plan_creation_ms = plan_start.elapsed().as_secs_f64() * 1_000.0;
        pop_gpu_error_scopes(plan_scopes, "creating NUFFT benchmark plan").await?;
        let plan = plan?;
        plan_creation_samples.push(plan_creation_ms);
        let this_fine_grid_shape = plan.fine_grid_shape().to_vec();
        if let Some(previous) = fine_grid_shape.as_ref() {
            if previous != &this_fine_grid_shape {
                return Err(input_error(format!(
                    "fine-grid shape changed between plan recreations: {previous:?} versus {this_fine_grid_shape:?}"
                )));
            }
        } else {
            fine_grid_shape = Some(this_fine_grid_shape);
        }

        let execution = TransformExecution {
            device,
            queue,
            plan: &plan,
            buffers,
            point_count: case.point_count,
            kind,
            wait_timeout: options.wait_timeout,
        };
        let (warmup_encoding_ms, warmup_submit_wait_ms) =
            execution.execute_once("warmup", 1).await?;
        println!(
            "run {}/{} plan_create_ms={:.6} warmup: encode_ms={:.6} submit_wait_ms={:.6}",
            run_index + 1,
            options.runs,
            plan_creation_ms,
            warmup_encoding_ms,
            warmup_submit_wait_ms,
        );

        let mut run_submit_wait = Vec::with_capacity(options.samples);
        let mut run_encoding = Vec::with_capacity(options.samples);
        for sample_index in 0..options.samples {
            let context = format!(
                "{} N={} run={} sample={}",
                kind.name(),
                case.mode_count()?,
                run_index + 1,
                sample_index + 1
            );
            let (encoding_ms, submit_wait_ms) = execution
                .execute_once(&context, transforms_per_sample)
                .await?;
            println!(
                "run {}/{} sample {}/{}: encode_ms={:.6} submit_wait_ms={:.6}",
                run_index + 1,
                options.runs,
                sample_index + 1,
                options.samples,
                encoding_ms,
                submit_wait_ms,
            );
            encoding_samples.push(encoding_ms);
            submit_wait_samples.push(submit_wait_ms);
            run_encoding.push(encoding_ms);
            run_submit_wait.push(submit_wait_ms);
        }
        let run_submit_stats = statistics(&run_submit_wait)?;
        let run_encoding_stats = statistics(&run_encoding)?;
        run_submit_wait_means.push(run_submit_stats.mean);
        run_encoding_means.push(run_encoding_stats.mean);
        println!(
            "run {}/{} summary: avg_submit_wait_ms={:.6} min_submit_wait_ms={:.6} avg_encode_ms={:.6} min_encode_ms={:.6}",
            run_index + 1,
            options.runs,
            run_submit_stats.mean,
            run_submit_stats.minimum,
            run_encoding_stats.mean,
            run_encoding_stats.minimum,
        );
    }

    let submit_wait_stats = statistics(&run_submit_wait_means)?;
    let raw_submit_wait_stats = statistics(&submit_wait_samples)?;
    let encoding_stats = statistics(&run_encoding_means)?;
    let raw_encoding_stats = statistics(&encoding_samples)?;
    let plan_creation_stats = statistics(&plan_creation_samples)?;
    let stderr = submit_wait_stats
        .stderr
        .map_or_else(|| "undefined".to_owned(), |value| format!("{value:.6}"));
    let encoding_stderr = encoding_stats
        .stderr
        .map_or_else(|| "undefined".to_owned(), |value| format!("{value:.6}"));
    let million_points_per_second = case.point_count as f64 / (submit_wait_stats.mean * 1_000.0);
    let fine_grid_shape = fine_grid_shape.expect("at least one validated run");
    let fine_grid_points = checked_product(&fine_grid_shape, "benchmark fine-grid point count")?;
    let mode_count = case.mode_count()?;

    println!(
        "RESULT kind={} N={} M={} fine_grid_length={} eps={} sigma={} sign=positive mode_order=centered precision=f32 runs={} samples_per_run={} transforms_per_sample={} total_samples={} raw_submit_wait_ms_per_transform={:?} run_avg_submit_wait_ms_per_transform={:?} avg_submit_wait_ms_per_transform={:.6} stderr_submit_wait_ms_per_transform={} stderr_basis=plan-recreated-run-means min_submit_wait_ms_per_transform={:.6} million_points_per_second={:.6} raw_encode_ms_per_transform={:?} run_avg_encode_ms_per_transform={:?} avg_encode_ms_per_transform={:.6} stderr_encode_ms_per_transform={} min_encode_ms_per_transform={:.6} raw_plan_create_ms={:?} avg_plan_create_ms={:.6} stderr_plan_create_ms={} min_plan_create_ms={:.6} timing_scope=submit-through-device-poll-divided-per-transform setup_excluded=plan,encode,upload,readback dimensions={} mode_shape={} fine_grid_shape={} fine_grid_points={}",
        kind.name(),
        mode_count,
        case.point_count,
        fine_grid_points,
        config.eps(),
        config.sigma(),
        options.runs,
        options.samples,
        transforms_per_sample,
        submit_wait_samples.len(),
        submit_wait_samples,
        run_submit_wait_means,
        submit_wait_stats.mean,
        stderr,
        raw_submit_wait_stats.minimum,
        million_points_per_second,
        encoding_samples,
        run_encoding_means,
        encoding_stats.mean,
        encoding_stderr,
        raw_encoding_stats.minimum,
        plan_creation_samples,
        plan_creation_stats.mean,
        plan_creation_stats
            .stderr
            .map_or_else(|| "undefined".to_owned(), |value| format!("{value:.6}")),
        plan_creation_stats.minimum,
        case.dimensions(),
        case.shape_token(),
        shape_token(&fine_grid_shape),
        fine_grid_points,
    );
    Ok(())
}

async fn run_type3_cases(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    options: &Options,
) -> BenchResult<()> {
    let mut cases = Vec::with_capacity(DEFAULT_TYPE3_DIMENSIONS.len());
    for &dimensions in &DEFAULT_TYPE3_DIMENSIONS {
        let case = create_type3_case(dimensions)?;
        let host_metadata = NufftType3Plan::new(case.config.clone())?;
        println!(
            "\n=== type-3 case dimensions={} logical_shape={} M={} K={} source_bounds={} target_bounds={} outer_grid_shape={} ===",
            dimensions,
            case.shape_token(),
            case.source_count,
            case.target_count,
            interval_token(case.config.source_bounds()),
            interval_token(case.config.target_bounds()),
            shape_token(&host_metadata.outer_grid_shape()),
        );
        cases.push(case);
    }
    run_type3_gpu_cases(device, queue, options, &cases).await
}

async fn run_type3_gpu_cases(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    options: &Options,
    cases: &[Type3Case],
) -> BenchResult<()> {
    for case in cases {
        let buffers = create_type3_case_buffers(device, case)?;
        run_type3_kind(device, queue, &buffers, case, options).await?;
        device
            .poll(wgpu::PollType::Poll)
            .map_err(|error| contextual_error("polling after type-3 benchmark case", error))?;
    }
    Ok(())
}

async fn run_type3_kind(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    buffers: &Type3CaseBuffers,
    case: &Type3Case,
    options: &Options,
) -> BenchResult<()> {
    let mut submit_wait_samples = Vec::with_capacity(options.runs * options.samples);
    let mut encoding_samples = Vec::with_capacity(options.runs * options.samples);
    let mut run_submit_wait_means = Vec::with_capacity(options.runs);
    let mut run_encoding_means = Vec::with_capacity(options.runs);
    let mut plan_creation_samples = Vec::with_capacity(options.runs);
    let mut outer_grid_shape = None;

    println!("\ntype-3: starting transforms_per_sample=1");
    for run_index in 0..options.runs {
        let plan_scopes = push_gpu_error_scopes(device);
        let plan_start = Instant::now();
        let plan = NufftType3Plan::new_gpu(device, queue, case.config.clone());
        let plan_creation_ms = plan_start.elapsed().as_secs_f64() * 1_000.0;
        pop_gpu_error_scopes(plan_scopes, "creating type-3 NUFFT benchmark plan").await?;
        let plan = plan?;
        if !plan.is_gpu_ready() {
            return Err(input_error(
                "NufftType3Plan::new_gpu returned a plan without GPU execution resources",
            ));
        }
        validate_type3_buffer_sizes(&plan, case)?;
        let this_outer_grid_shape = plan.outer_grid_shape();
        if let Some(previous) = outer_grid_shape.as_ref() {
            if previous != &this_outer_grid_shape {
                return Err(input_error(format!(
                    "type-3 outer-grid shape changed between plan recreations: {previous:?} versus {this_outer_grid_shape:?}"
                )));
            }
        } else {
            outer_grid_shape = Some(this_outer_grid_shape);
        }
        plan_creation_samples.push(plan_creation_ms);

        let execution = Type3Execution {
            device,
            queue,
            plan: &plan,
            buffers,
            source_count: case.source_count,
            target_count: case.target_count,
            wait_timeout: options.wait_timeout,
        };
        let (warmup_encoding_ms, warmup_submit_wait_ms) =
            execution.execute_once("type-3 warmup").await?;
        println!(
            "run {}/{} plan_create_ms={:.6} warmup: encode_ms={:.6} submit_wait_ms={:.6}",
            run_index + 1,
            options.runs,
            plan_creation_ms,
            warmup_encoding_ms,
            warmup_submit_wait_ms,
        );

        let mut run_submit_wait = Vec::with_capacity(options.samples);
        let mut run_encoding = Vec::with_capacity(options.samples);
        for sample_index in 0..options.samples {
            let context = format!(
                "type-3 dimensions={} M={} K={} run={} sample={}",
                case.dimensions,
                case.source_count,
                case.target_count,
                run_index + 1,
                sample_index + 1,
            );
            let (encoding_ms, submit_wait_ms) = execution.execute_once(&context).await?;
            println!(
                "run {}/{} sample {}/{}: encode_ms={:.6} submit_wait_ms={:.6}",
                run_index + 1,
                options.runs,
                sample_index + 1,
                options.samples,
                encoding_ms,
                submit_wait_ms,
            );
            encoding_samples.push(encoding_ms);
            submit_wait_samples.push(submit_wait_ms);
            run_encoding.push(encoding_ms);
            run_submit_wait.push(submit_wait_ms);
        }
        let run_submit_stats = statistics(&run_submit_wait)?;
        let run_encoding_stats = statistics(&run_encoding)?;
        run_submit_wait_means.push(run_submit_stats.mean);
        run_encoding_means.push(run_encoding_stats.mean);
        println!(
            "run {}/{} summary: avg_submit_wait_ms={:.6} min_submit_wait_ms={:.6} avg_encode_ms={:.6} min_encode_ms={:.6}",
            run_index + 1,
            options.runs,
            run_submit_stats.mean,
            run_submit_stats.minimum,
            run_encoding_stats.mean,
            run_encoding_stats.minimum,
        );
    }

    let submit_wait_stats = statistics(&run_submit_wait_means)?;
    let raw_submit_wait_stats = statistics(&submit_wait_samples)?;
    let encoding_stats = statistics(&run_encoding_means)?;
    let raw_encoding_stats = statistics(&encoding_samples)?;
    let plan_creation_stats = statistics(&plan_creation_samples)?;
    let stderr = format_optional_stat(submit_wait_stats.stderr);
    let encoding_stderr = format_optional_stat(encoding_stats.stderr);
    let throughput =
        (case.source_count + case.target_count) as f64 / (submit_wait_stats.mean * 1_000.0);
    let outer_grid_shape = outer_grid_shape.expect("at least one validated run");

    println!(
        "RESULT kind=type-3 dimensions={} logical_shape={} M={} K={} outer_grid_shape={} eps={} sigma={} sign=positive precision=f32 runs={} samples_per_run={} transforms_per_sample=1 total_samples={} raw_submit_wait_ms_per_transform={:?} run_avg_submit_wait_ms_per_transform={:?} avg_submit_wait_ms_per_transform={:.6} stderr_submit_wait_ms_per_transform={} stderr_basis=plan-recreated-run-means min_submit_wait_ms_per_transform={:.6} million_source_plus_target_points_per_second={:.6} raw_encode_ms_per_transform={:?} run_avg_encode_ms_per_transform={:?} avg_encode_ms_per_transform={:.6} stderr_encode_ms_per_transform={} min_encode_ms_per_transform={:.6} raw_plan_create_ms={:?} avg_plan_create_ms={:.6} stderr_plan_create_ms={} min_plan_create_ms={:.6} timing_scope=submit-through-device-poll setup_excluded=plan,encode,upload,readback cufinufft_comparison_span=setpts-plus-execute source_bounds={} target_bounds={}",
        case.dimensions,
        case.shape_token(),
        case.source_count,
        case.target_count,
        shape_token(&outer_grid_shape),
        case.config.eps(),
        case.config.sigma(),
        options.runs,
        options.samples,
        submit_wait_samples.len(),
        submit_wait_samples,
        run_submit_wait_means,
        submit_wait_stats.mean,
        stderr,
        raw_submit_wait_stats.minimum,
        throughput,
        encoding_samples,
        run_encoding_means,
        encoding_stats.mean,
        encoding_stderr,
        raw_encoding_stats.minimum,
        plan_creation_samples,
        plan_creation_stats.mean,
        format_optional_stat(plan_creation_stats.stderr),
        plan_creation_stats.minimum,
        interval_token(case.config.source_bounds()),
        interval_token(case.config.target_bounds()),
    );
    Ok(())
}

impl TransformExecution<'_> {
    async fn execute_once(&self, context: &str, transforms: usize) -> BenchResult<(f64, f64)> {
        if transforms == 0 {
            return Err(input_error("timed transform count must be positive"));
        }
        let scopes = push_gpu_error_scopes(self.device);
        let encode_start = Instant::now();
        let command_buffer = (|| -> BenchResult<wgpu::CommandBuffer> {
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("wgpu_nufft.bench.encoder"),
                });
            for _ in 0..transforms {
                self.kind.encode(
                    self.plan,
                    self.device,
                    &mut encoder,
                    self.point_count,
                    self.buffers,
                )?;
            }
            Ok(encoder.finish())
        })();
        let encode_ms = encode_start.elapsed().as_secs_f64() * 1_000.0 / transforms as f64;
        let command_buffer = match command_buffer {
            Ok(command_buffer) => command_buffer,
            Err(error) => {
                pop_gpu_error_scopes(scopes, &format!("recording {context}")).await?;
                return Err(error);
            }
        };

        let submit_start = Instant::now();
        let submission = self.queue.submit([command_buffer]);
        let wait_result = wait_for_submission(
            self.device,
            submission,
            &format!("waiting for {context}"),
            self.wait_timeout,
        );
        let submit_wait_ms = submit_start.elapsed().as_secs_f64() * 1_000.0 / transforms as f64;
        let scope_result = pop_gpu_error_scopes(scopes, &format!("executing {context}")).await;
        wait_result?;
        scope_result?;

        if !encode_ms.is_finite() || encode_ms < 0.0 {
            return Err(input_error(format!(
                "invalid command-encoding duration for {context}: {encode_ms} ms"
            )));
        }
        if !submit_wait_ms.is_finite() || submit_wait_ms <= 0.0 {
            return Err(input_error(format!(
                "invalid submit-to-wait duration for {context}: {submit_wait_ms} ms"
            )));
        }
        Ok((encode_ms, submit_wait_ms))
    }
}

impl Type3Execution<'_> {
    async fn execute_once(&self, context: &str) -> BenchResult<(f64, f64)> {
        let scopes = push_gpu_error_scopes(self.device);
        let encode_start = Instant::now();
        let command_buffer = (|| -> BenchResult<wgpu::CommandBuffer> {
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("wgpu_nufft.bench.type3.encoder"),
                });
            self.plan.encode_gpu(
                self.device,
                &mut encoder,
                self.source_count,
                &self.buffers.source_points,
                &self.buffers.strengths,
                self.target_count,
                &self.buffers.target_points,
                &self.buffers.output,
            )?;
            Ok(encoder.finish())
        })();
        let encode_ms = encode_start.elapsed().as_secs_f64() * 1_000.0;
        let command_buffer = match command_buffer {
            Ok(command_buffer) => command_buffer,
            Err(error) => {
                pop_gpu_error_scopes(scopes, &format!("recording {context}")).await?;
                return Err(error);
            }
        };

        let submit_start = Instant::now();
        let submission = self.queue.submit([command_buffer]);
        let wait_result = wait_for_submission(
            self.device,
            submission,
            &format!("waiting for {context}"),
            self.wait_timeout,
        );
        let submit_wait_ms = submit_start.elapsed().as_secs_f64() * 1_000.0;
        let scope_result = pop_gpu_error_scopes(scopes, &format!("executing {context}")).await;
        wait_result?;
        scope_result?;

        if !encode_ms.is_finite() || encode_ms < 0.0 {
            return Err(input_error(format!(
                "invalid command-encoding duration for {context}: {encode_ms} ms"
            )));
        }
        if !submit_wait_ms.is_finite() || submit_wait_ms <= 0.0 {
            return Err(input_error(format!(
                "invalid submit-to-wait duration for {context}: {submit_wait_ms} ms"
            )));
        }
        Ok((encode_ms, submit_wait_ms))
    }
}

fn create_type3_case(dimensions: usize) -> BenchResult<Type3Case> {
    // cuFINUFFT derives its type-3 bounds from the f32 point arrays passed to
    // setpts. Use the same rounded, pinned endpoints for the reusable Rust
    // plan's explicit bounds contract.
    let source_interval =
        pinned_type3_interval(DEFAULT_TYPE3_SOURCE_HALFWIDTH, DEFAULT_TYPE3_SOURCE_CENTER);
    let target_interval =
        pinned_type3_interval(DEFAULT_TYPE3_TARGET_HALFWIDTH, DEFAULT_TYPE3_TARGET_CENTER);
    let source_bounds = vec![source_interval; dimensions];
    let target_bounds = vec![target_interval; dimensions];
    let config = NufftType3Config::new(source_bounds, target_bounds, DEFAULT_EPS)
        .with_sigma(DEFAULT_SIGMA)
        .with_sign(NufftSign::Positive);
    config.validate()?;
    Ok(Type3Case {
        dimensions,
        source_count: DEFAULT_TYPE3_SOURCE_COUNT,
        target_count: DEFAULT_TYPE3_TARGET_COUNT,
        config,
    })
}

fn pinned_type3_interval(halfwidth: f64, center: f64) -> NufftInterval {
    NufftInterval::new(
        f64::from((center - halfwidth) as f32),
        f64::from((center + halfwidth) as f32),
    )
}

fn create_type3_case_buffers(
    device: &wgpu::Device,
    case: &Type3Case,
) -> BenchResult<Type3CaseBuffers> {
    let (source_points, target_points, strengths) = generate_type3_case_data(case)?;
    let storage_usage = wgpu::BufferUsages::STORAGE;
    let source_points = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("wgpu_nufft.bench.type3.source_points"),
        contents: bytemuck::cast_slice(&source_points),
        usage: storage_usage,
    });
    let target_points = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("wgpu_nufft.bench.type3.target_points"),
        contents: bytemuck::cast_slice(&target_points),
        usage: storage_usage,
    });
    let strengths = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("wgpu_nufft.bench.type3.strengths"),
        contents: bytemuck::cast_slice(&strengths),
        usage: storage_usage,
    });
    let output = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_nufft.bench.type3.output"),
        size: checked_buffer_bytes(case.target_count, 8, "type-3 benchmark output buffer")?,
        usage: storage_usage,
        mapped_at_creation: false,
    });
    Ok(Type3CaseBuffers {
        source_points,
        target_points,
        strengths,
        output,
    })
}

fn generate_type3_case_data(case: &Type3Case) -> BenchResult<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    let source_coordinate_count = case
        .source_count
        .checked_mul(case.dimensions)
        .ok_or_else(|| input_error("type-3 source-coordinate count overflow"))?;
    let target_coordinate_count = case
        .target_count
        .checked_mul(case.dimensions)
        .ok_or_else(|| input_error("type-3 target-coordinate count overflow"))?;
    let strength_value_count = case
        .source_count
        .checked_mul(2)
        .ok_or_else(|| input_error("type-3 strength value count overflow"))?;
    let source_points = generate_type3_coordinates(
        case.dimensions,
        case.source_count,
        &SOURCE_POINT_SEED_MASKS,
        DEFAULT_TYPE3_SOURCE_HALFWIDTH,
        DEFAULT_TYPE3_SOURCE_CENTER,
    )?;
    let target_points = generate_type3_coordinates(
        case.dimensions,
        case.target_count,
        &TYPE3_TARGET_SEED_MASKS,
        DEFAULT_TYPE3_TARGET_HALFWIDTH,
        DEFAULT_TYPE3_TARGET_CENTER,
    )?;
    debug_assert_eq!(source_points.len(), source_coordinate_count);
    debug_assert_eq!(target_points.len(), target_coordinate_count);
    let mut strength_real_rng = Lcg32::new(DATA_SEED ^ STRENGTH_REAL_SEED_MASK);
    let mut strength_imag_rng = Lcg32::new(DATA_SEED ^ STRENGTH_IMAG_SEED_MASK);
    let mut strengths = Vec::with_capacity(strength_value_count);
    for _ in 0..case.source_count {
        strengths.push(-1.0 + 2.0 * strength_real_rng.next_unit_interval_f32());
        strengths.push(-1.0 + 2.0 * strength_imag_rng.next_unit_interval_f32());
    }
    Ok((source_points, target_points, strengths))
}

fn validate_type3_buffer_sizes(plan: &NufftType3Plan, case: &Type3Case) -> BenchResult<()> {
    let expected_source_points = checked_buffer_bytes(
        case.source_count
            .checked_mul(case.dimensions)
            .ok_or_else(|| input_error("type-3 source-coordinate size overflow"))?,
        4,
        "type-3 source-coordinate buffer",
    )?;
    let expected_target_points = checked_buffer_bytes(
        case.target_count
            .checked_mul(case.dimensions)
            .ok_or_else(|| input_error("type-3 target-coordinate size overflow"))?,
        4,
        "type-3 target-coordinate buffer",
    )?;
    let expected_strengths = checked_buffer_bytes(case.source_count, 8, "type-3 strength buffer")?;
    let expected_output = checked_buffer_bytes(case.target_count, 8, "type-3 output buffer")?;
    let reported = [
        plan.required_source_point_buffer_size_bytes(case.source_count)?,
        plan.required_target_point_buffer_size_bytes(case.target_count)?,
        plan.required_strength_buffer_size_bytes(case.source_count)?,
        plan.required_output_buffer_size_bytes(case.target_count)?,
    ];
    let expected = [
        expected_source_points,
        expected_target_points,
        expected_strengths,
        expected_output,
    ];
    if reported != expected {
        return Err(input_error(format!(
            "type-3 GPU buffer-size API mismatch: reported={reported:?} expected={expected:?}"
        )));
    }
    Ok(())
}

fn benchmark_config(mode_shape: &[usize]) -> NufftConfig {
    NufftConfig::new(mode_shape.to_vec(), DEFAULT_EPS)
        .with_sigma(DEFAULT_SIGMA)
        .with_sign(NufftSign::Positive)
        .with_mode_order(ModeOrder::Centered)
}

fn create_case_buffers(device: &wgpu::Device, case: &Type12Case) -> BenchResult<CaseBuffers> {
    let mode_count = case.mode_count()?;
    let (points, strengths, modes) =
        generate_case_data(case.dimensions(), case.point_count, mode_count)?;
    let storage_usage = wgpu::BufferUsages::STORAGE;
    let points = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("wgpu_nufft.bench.points"),
        contents: bytemuck::cast_slice(&points),
        usage: storage_usage,
    });
    let strengths = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("wgpu_nufft.bench.strengths"),
        contents: bytemuck::cast_slice(&strengths),
        usage: storage_usage,
    });
    let modes = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("wgpu_nufft.bench.modes"),
        contents: bytemuck::cast_slice(&modes),
        usage: storage_usage,
    });
    let type1_output_bytes = checked_buffer_bytes(mode_count, 8, "benchmark type-1 output buffer")?;
    let type2_output_bytes =
        checked_buffer_bytes(case.point_count, 8, "benchmark type-2 output buffer")?;
    let type1_output = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_nufft.bench.type1_output"),
        size: type1_output_bytes,
        usage: storage_usage,
        mapped_at_creation: false,
    });
    let type2_output = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_nufft.bench.type2_output"),
        size: type2_output_bytes,
        usage: storage_usage,
        mapped_at_creation: false,
    });
    Ok(CaseBuffers {
        points,
        strengths,
        modes,
        type1_output,
        type2_output,
    })
}

fn generate_case_data(
    dimensions: usize,
    point_count: usize,
    mode_count: usize,
) -> BenchResult<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    let point_coordinates = point_count
        .checked_mul(dimensions)
        .ok_or_else(|| input_error("benchmark point-coordinate count overflow"))?;
    let strength_values = point_count
        .checked_mul(2)
        .ok_or_else(|| input_error("benchmark strength value count overflow"))?;
    let mode_values = mode_count
        .checked_mul(2)
        .ok_or_else(|| input_error("benchmark mode value count overflow"))?;
    let mut strength_real_rng = Lcg32::new(DATA_SEED ^ STRENGTH_REAL_SEED_MASK);
    let mut strength_imag_rng = Lcg32::new(DATA_SEED ^ STRENGTH_IMAG_SEED_MASK);
    let mut mode_real_rng = Lcg32::new(DATA_SEED ^ MODE_REAL_SEED_MASK);
    let mut mode_imag_rng = Lcg32::new(DATA_SEED ^ MODE_IMAG_SEED_MASK);
    let points = generate_type12_coordinates(dimensions, point_count)?;
    debug_assert_eq!(points.len(), point_coordinates);
    let mut strengths = Vec::with_capacity(strength_values);
    for _ in 0..point_count {
        strengths.push(-1.0 + 2.0 * strength_real_rng.next_unit_interval_f32());
        strengths.push(-1.0 + 2.0 * strength_imag_rng.next_unit_interval_f32());
    }
    let mut modes = Vec::with_capacity(mode_values);
    for _ in 0..mode_count {
        modes.push(-1.0 + 2.0 * mode_real_rng.next_unit_interval_f32());
        modes.push(-1.0 + 2.0 * mode_imag_rng.next_unit_interval_f32());
    }
    Ok((points, strengths, modes))
}

fn generate_type12_coordinates(dimensions: usize, point_count: usize) -> BenchResult<Vec<f32>> {
    if dimensions == 0 || dimensions > SOURCE_POINT_SEED_MASKS.len() {
        return Err(input_error(format!(
            "benchmark point generation supports 1..={} dimensions, got {dimensions}",
            SOURCE_POINT_SEED_MASKS.len()
        )));
    }
    let coordinate_count = point_count
        .checked_mul(dimensions)
        .ok_or_else(|| input_error("benchmark point-coordinate count overflow"))?;
    let mut streams = SOURCE_POINT_SEED_MASKS[..dimensions]
        .iter()
        .map(|&mask| Lcg32::new(DATA_SEED ^ mask))
        .collect::<Vec<_>>();
    let mut coordinates = Vec::with_capacity(coordinate_count);
    for _ in 0..point_count {
        for stream in &mut streams {
            let unit = stream.next_unit_interval_f32();
            coordinates.push(unit * std::f32::consts::TAU - std::f32::consts::PI);
        }
    }
    Ok(coordinates)
}

fn generate_type3_coordinates(
    dimensions: usize,
    point_count: usize,
    seed_masks: &[u32; 3],
    halfwidth: f64,
    center: f64,
) -> BenchResult<Vec<f32>> {
    if dimensions == 0 || dimensions > seed_masks.len() {
        return Err(input_error(format!(
            "type-3 point generation supports 1..={} dimensions, got {dimensions}",
            seed_masks.len()
        )));
    }
    let coordinate_count = point_count
        .checked_mul(dimensions)
        .ok_or_else(|| input_error("type-3 point-coordinate count overflow"))?;
    let mut streams = seed_masks[..dimensions]
        .iter()
        .map(|&mask| Lcg32::new(DATA_SEED ^ mask))
        .collect::<Vec<_>>();
    let lower = (center - halfwidth) as f32;
    let upper = (center + halfwidth) as f32;
    let mut coordinates = Vec::with_capacity(coordinate_count);
    for point in 0..point_count {
        for stream in &mut streams {
            let unit = stream.next_unit_interval_f32();
            let coordinate = match point {
                0 => lower,
                1 => upper,
                _ => (center + (f64::from(unit) * 2.0 - 1.0) * halfwidth) as f32,
            };
            coordinates.push(coordinate);
        }
    }
    Ok(coordinates)
}

struct Lcg32 {
    state: u32,
}

impl Lcg32 {
    const fn new(seed: u32) -> Self {
        Self { state: seed }
    }

    fn next_unit_interval_f32(&mut self) -> f32 {
        self.state = self
            .state
            .wrapping_mul(1_664_525)
            .wrapping_add(1_013_904_223);
        (self.state >> 8) as f32 * (1.0 / 16_777_216.0)
    }
}

fn statistics(samples: &[f64]) -> BenchResult<Statistics> {
    if samples.is_empty() || samples.iter().any(|sample| !sample.is_finite()) {
        return Err(input_error("statistics require finite, nonempty samples"));
    }
    let mean = samples.iter().sum::<f64>() / samples.len() as f64;
    let minimum = samples.iter().copied().fold(f64::INFINITY, f64::min);
    let stderr = if samples.len() > 1 {
        let squared_deviations = samples
            .iter()
            .map(|sample| (sample - mean).powi(2))
            .sum::<f64>();
        let sample_variance = squared_deviations / (samples.len() - 1) as f64;
        Some(sample_variance.sqrt() / (samples.len() as f64).sqrt())
    } else {
        None
    };
    Ok(Statistics {
        mean,
        stderr,
        minimum,
    })
}

fn parse_options() -> BenchResult<Options> {
    let mut adapter_selector = None;
    let mut runs = DEFAULT_RUNS;
    let mut samples = DEFAULT_SAMPLES;
    let mut wait_timeout = DEFAULT_WAIT_TIMEOUT;
    let mut explicit_suite = false;
    let mut run_1d = false;
    let mut run_3d = false;
    let mut run_type3 = false;
    // Cargo invokes harness-free benchmark binaries with an implicit
    // `--bench`; it is not part of this harness's CLI.
    let mut arguments = std::env::args().skip(1).filter(|arg| arg != "--bench");
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--1d" => {
                explicit_suite = true;
                run_1d = true;
            }
            "--3d" => {
                explicit_suite = true;
                run_3d = true;
            }
            "--type3" => {
                explicit_suite = true;
                run_type3 = true;
            }
            "--all" => {
                explicit_suite = true;
                run_1d = true;
                run_3d = true;
                run_type3 = true;
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
    if !explicit_suite {
        run_1d = true;
    }
    Ok(Options {
        adapter_selector,
        runs,
        samples,
        wait_timeout,
        run_1d,
        run_3d,
        run_type3,
    })
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

fn print_usage() {
    println!(
        r#"Usage:
  cargo bench -p wgpu-nufft --bench nufft_bench -- [options]

Options:
  --1d                       Run established 1D type-1/type-2 cases (default).
  --3d                       Run 3D type-1/type-2 cases at 64^3 and 128^3.
  --type3                    Run one type-3 case in each of 1D, 2D, and 3D.
  --all                      Run all suites (equivalent to --1d --3d --type3).
  --adapter <index-or-name>   Select a Vulkan hardware adapter.
  --runs <count>              Plan recreations per case (default: 3).
  --samples <count>           Timed submissions per plan (default: 10).
  --wait-timeout-secs <secs>  Per-submission timeout (default: 120).
  --help                      Show this help.

The default 1D cases use N=M=262144 and 1048576. The 3D cases use
N=M=64^3 and 128^3. Type-3 uses M=K=65536 in each dimension, source
center pi/4 with halfwidth pi, and target center 4 with halfwidth 16.
All cases use f32, eps=1e-6, sigma=2, and positive sign;
type-1/type-2 cases use centered mode order."#
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
    let validation_pop = validation.pop();
    let internal_pop = internal.pop();
    let out_of_memory_pop = out_of_memory.pop();
    let validation_error = validation_pop.await;
    let internal_error = internal_pop.await;
    let out_of_memory_error = out_of_memory_pop.await;
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
    submission_index: wgpu::SubmissionIndex,
    context: &str,
    timeout: Duration,
) -> BenchResult<()> {
    device
        .poll(wgpu::PollType::Wait {
            submission_index: Some(submission_index),
            timeout: Some(timeout),
        })
        .map(|_| ())
        .map_err(|error| {
            contextual_error(
                format!("{context} (submission wait timeout {timeout:?})"),
                error,
            )
        })
}

fn selected_suites_token(options: &Options) -> String {
    [
        (options.run_1d, "1d"),
        (options.run_3d, "3d"),
        (options.run_type3, "type3"),
    ]
    .into_iter()
    .filter_map(|(selected, name)| selected.then_some(name))
    .collect::<Vec<_>>()
    .join(",")
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

fn checked_product(values: &[usize], context: &'static str) -> BenchResult<usize> {
    values.iter().try_fold(1usize, |product, &value| {
        product
            .checked_mul(value)
            .ok_or_else(|| input_error(format!("{context} overflow")))
    })
}

fn checked_buffer_bytes(
    elements: usize,
    bytes_per_element: usize,
    context: &'static str,
) -> BenchResult<u64> {
    elements
        .checked_mul(bytes_per_element)
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or_else(|| input_error(format!("{context} size overflow")))
}

fn format_optional_stat(value: Option<f64>) -> String {
    value.map_or_else(|| "undefined".to_owned(), |value| format!("{value:.6}"))
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
        formatter.write_str(&self.context)
    }
}

impl Error for ContextError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        Some(self.source.as_ref())
    }
}
