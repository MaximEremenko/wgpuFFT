use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;
use std::io;
use std::time::{Duration, Instant};

use wgpu_fft::{
    clear_thread_local_pipeline_cache, FftConfig, FftDiagnostics, FftPlan, FftPrecision,
    LargePolicyLimits, Normalization,
};

type BenchResult<T> = Result<T, Box<dyn Error>>;

const DEFAULT_RUNS: usize = 3;
const DEFAULT_ITER_CAP: u64 = 1000;
const DEFAULT_WAIT_TIMEOUT_SECS: u64 = 120;
const ITER_TRAFFIC_BUDGET_BYTES: u64 = 3 * 4096 * 1024 * 1024;
const TARGET_COMPLEX_ELEMENTS: usize = 1 << 27;
const INITIALIZATION_SEED_BYTES: u64 = 16 * 1024 * 1024;
const DEFAULT_SEGMENTED_BURST_DEPTH: usize = 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PrecisionMode {
    F32,
    F64,
    Df64,
    Both,
    Df64F64,
}

impl PrecisionMode {
    fn parse(value: &str) -> Option<Self> {
        match value.to_ascii_lowercase().as_str() {
            "f32" => Some(Self::F32),
            "f64" => Some(Self::F64),
            "df64" => Some(Self::Df64),
            "both" => Some(Self::Both),
            "df64-f64" | "df64_f64" | "df64f64" => Some(Self::Df64F64),
            _ => None,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F64 => "f64",
            Self::Df64 => "df64",
            Self::Both => "both",
            Self::Df64F64 => "df64-f64",
        }
    }

    fn precisions(self) -> Vec<FftPrecision> {
        match self {
            Self::F32 => vec![FftPrecision::F32],
            Self::F64 => vec![FftPrecision::F64],
            Self::Df64 => vec![FftPrecision::Df64],
            Self::Both => vec![FftPrecision::F32, FftPrecision::F64],
            Self::Df64F64 => vec![FftPrecision::Df64, FftPrecision::F64],
        }
    }

    fn needs_f64(self) -> bool {
        matches!(self, Self::F64 | Self::Both | Self::Df64F64)
    }

    fn single_precision(self) -> Option<FftPrecision> {
        match self {
            Self::F32 => Some(FftPrecision::F32),
            Self::F64 => Some(FftPrecision::F64),
            Self::Df64 => Some(FftPrecision::Df64),
            Self::Both | Self::Df64F64 => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Suite {
    Smoke,
    Sample0,
    Sample1000,
    Sample3,
    Sample7,
    All,
    Custom,
}

impl Suite {
    fn parse(value: &str) -> Option<Self> {
        match value.to_ascii_lowercase().as_str() {
            "smoke" => Some(Self::Smoke),
            "sample0" | "sample_0" | "0" => Some(Self::Sample0),
            "sample1000" | "sample_1000" | "1000" => Some(Self::Sample1000),
            "sample3" | "sample_3" | "3" => Some(Self::Sample3),
            "sample7" | "sample_7" | "7" => Some(Self::Sample7),
            "all" => Some(Self::All),
            "shape" | "custom" => Some(Self::Custom),
            _ => None,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Smoke => "smoke",
            Self::Sample0 => "sample0",
            Self::Sample1000 => "sample1000",
            Self::Sample3 => "sample3",
            Self::Sample7 => "sample7",
            Self::All => "all",
            Self::Custom => "custom",
        }
    }
}

#[derive(Debug)]
struct Options {
    suite: Suite,
    precision: PrecisionMode,
    custom_shape: Option<Vec<usize>>,
    custom_batch: usize,
    adapter_selector: Option<String>,
    runs: usize,
    iter_cap: u64,
    max_cases: Option<usize>,
    wait_timeout: Duration,
    plan_max_bind_bytes: Option<u64>,
    compare_max_buffer_bytes: Option<CompareMaxBufferBytes>,
    segmented_burst_depth: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CompareMaxBufferBytes {
    unsharded: u64,
    sharded: u64,
}

#[derive(Debug, Clone)]
struct BenchCase {
    suite: &'static str,
    label: String,
    shape: Vec<usize>,
    batch: usize,
    report: bool,
}

#[derive(Debug)]
struct CaseResult {
    precision: FftPrecision,
    route: String,
    execution_kind: String,
    axis_kinds: String,
    graph_stage_count: u64,
    traffic_equivalent_pass_count: u64,
    pass_count_method: String,
    buffer_size: u64,
    external_io_bytes: u64,
    initialization_seed_bytes: u64,
    plan_workspace_requirement_bytes: u64,
    diagnostic_helper_requirement_bytes: u64,
    diagnostic_helper_requirements: String,
    arena_segment_bytes: Vec<u64>,
    segmented_burst_depth: usize,
    num_iter: u64,
    run_pair_ms: Vec<f64>,
}

#[derive(Debug)]
struct CompareVariantResult {
    label: &'static str,
    requested_max_buffer_bytes: u64,
    effective_max_bind_bytes: u64,
    effective_max_buffer_bytes: u64,
    result: CaseResult,
}

#[derive(Debug)]
struct CompareCaseResult {
    unsharded: CompareVariantResult,
    sharded: CompareVariantResult,
}

#[derive(Debug)]
struct PlanPairMetadata {
    route: String,
    execution_kind: String,
    axis_kinds: String,
    graph_stage_count: u64,
    traffic_equivalent_pass_count: u64,
    pass_count_method: String,
    plan_workspace_requirement_bytes: u64,
    diagnostic_helper_requirement_bytes: u64,
    diagnostic_helper_requirements: String,
    arena_segment_bytes: Vec<u64>,
    segmented_burst_depth: usize,
}

#[derive(Debug, Default)]
struct VariantAccumulator {
    route: Option<String>,
    execution_kind: Option<String>,
    axis_kinds: Option<String>,
    graph_stage_count: Option<u64>,
    traffic_equivalent_pass_count: Option<u64>,
    pass_count_method: Option<String>,
    plan_workspace_requirement_bytes: Option<u64>,
    diagnostic_helper_requirement_bytes: Option<u64>,
    diagnostic_helper_requirements: Option<String>,
    arena_segment_bytes: Option<Vec<u64>>,
    segmented_burst_depth: Option<usize>,
    run_pair_ms: Vec<f64>,
}

#[derive(Debug)]
struct Statistics {
    mean_ms: f64,
    stderr_ms: Option<f64>,
    vkfft_population_spread_ms: f64,
}

fn main() {
    if let Err(error) = pollster::block_on(run()) {
        eprintln!("wgpuFFT benchmark failed: {error}");
        let mut source = error.source();
        while let Some(cause) = source {
            eprintln!("  caused by: {cause}");
            source = cause.source();
        }
        std::process::exit(1);
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
            "  index={} name={:?} vendor={:#x} device={:#x} type={:?} driver={:?}",
            index, info.name, info.vendor, info.device, info.device_type, info.driver
        );
    }

    let selected_position = if let Some(selector) = selector {
        let numeric_index = selector
            .parse::<usize>()
            .ok()
            .filter(|&index| index < adapters.len());
        if let Some(index) = numeric_index {
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
            .filter_map(|(index, (_, info))| is_native_hardware_adapter(info).then_some(index))
            .collect::<Vec<_>>();
        match hardware.as_slice() {
            [index] => *index,
            [] => {
                return Err(input_error(
                    "no native discrete or integrated Vulkan GPU was found",
                ))
            }
            _ => {
                return Err(input_error(format!(
                    "multiple native Vulkan GPUs were found at indices {hardware:?}; select one with --adapter <index-or-name>"
                )))
            }
        }
    };

    let (adapter, info) = adapters.swap_remove(selected_position);
    if !is_native_hardware_adapter(&info) {
        return Err(input_error(format!(
            "refusing non-native-hardware Vulkan adapter {:?} ({:?})",
            info.name, info.device_type
        )));
    }
    println!(
        "selected Vulkan adapter index={} name={:?} vendor={:#x} device={:#x} type={:?}",
        selected_position, info.name, info.vendor, info.device, info.device_type
    );
    Ok(adapter)
}

fn is_native_hardware_adapter(info: &wgpu::AdapterInfo) -> bool {
    matches!(
        info.device_type,
        wgpu::DeviceType::DiscreteGpu | wgpu::DeviceType::IntegratedGpu
    )
}

async fn run() -> BenchResult<()> {
    let options = parse_options()?;
    let cases = build_cases(&options)?;
    if cases.is_empty() {
        return Err(input_error("the selected suite contains no cases"));
    }

    println!("wgpuFFT VkFFT-parity benchmark harness");
    println!(
        "configuration: suite={} precision={} adapter_selector={} runs={} iter_cap={} max_reported_cases_per_suite={} traffic_budget_bytes={} wait_timeout_secs={}",
        options.suite.name(),
        options.precision.name(),
        options.adapter_selector.as_deref().unwrap_or("auto-single-hardware"),
        options.runs,
        options.iter_cap,
        options
            .max_cases
            .map_or_else(|| "unlimited".to_owned(), |value| value.to_string()),
        ITER_TRAFFIC_BUDGET_BYTES,
        options.wait_timeout.as_secs(),
    );
    println!(
        "method: FFT+iFFT pairs, one command encoder, one submit, wall-clock submit-through-device-poll"
    );
    println!(
        "comparability: wgpuFFT mode=out-of-place; VkFFT reference samples mode=in-place; both directions normalization=none"
    );
    println!(
        "iteration budget note: normalized 3*4096 MiB for every suite as requested; upstream VkFFT sample3/sample7 use 4096 MiB"
    );
    if options.suite == Suite::Custom {
        println!("CUSTOM SHAPE SMOKE MODE: this case is outside the upstream VkFFT grids");
    }
    if let (Some(max_bind_bytes), Some(max_buffer_bytes)) = (
        options.plan_max_bind_bytes,
        options.compare_max_buffer_bytes,
    ) {
        println!(
            "CUSTOM SHAPE SEGMENT-CAP COMPARISON: plan_max_bind_bytes={} unsharded_max_buffer_bytes={} sharded_max_buffer_bytes={} segmented_burst_depth={}",
            max_bind_bytes,
            max_buffer_bytes.unsharded,
            max_buffer_bytes.sharded,
            options.segmented_burst_depth,
        );
        println!(
            "comparison method: recreate and time both variants in every run; variant order alternates by run"
        );
    }
    if options.runs != DEFAULT_RUNS
        || options.iter_cap != DEFAULT_ITER_CAP
        || options.max_cases.is_some()
    {
        println!("SMOKE OVERRIDES ACTIVE: results are not the default full VkFFT-comparable run");
    }

    let mut instance_descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
    instance_descriptor.backends = wgpu::Backends::VULKAN;
    let instance = wgpu::Instance::new(instance_descriptor);
    let adapter = select_vulkan_adapter(&instance, options.adapter_selector.as_deref()).await?;

    let adapter_info = adapter.get_info();
    if adapter_info.backend != wgpu::Backend::Vulkan {
        return Err(input_error(format!(
            "requested a Vulkan-only instance but selected backend {:?}",
            adapter_info.backend
        )));
    }
    let adapter_limits = adapter.limits();
    let adapter_features = adapter.features();
    println!("adapter info:\n{adapter_info:#?}");
    println!("adapter limits:\n{adapter_limits:#?}");

    let required_features = if options.precision.needs_f64() {
        if !adapter_features.contains(wgpu::Features::SHADER_F64) {
            return Err(input_error(format!(
                "precision {} requires wgpu SHADER_F64, but Vulkan adapter {:?} does not expose it",
                options.precision.name(),
                adapter_info.name,
            )));
        }
        wgpu::Features::SHADER_F64
    } else {
        wgpu::Features::empty()
    };
    println!(
        "precision feature selection: mode={} adapter_shader_f64={} requested_shader_f64={}",
        options.precision.name(),
        adapter_features.contains(wgpu::Features::SHADER_F64),
        required_features.contains(wgpu::Features::SHADER_F64),
    );

    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("wgpu_fft.bench.device"),
            required_features,
            required_limits: adapter_limits,
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
        })
        .await
        .map_err(|error| contextual_error("requesting the benchmark device", error))?;
    println!("device limits:\n{:#?}", device.limits());

    let benchmark_result = run_cases(&device, &queue, &cases, &options).await;

    // Keep native objects alive until all success or failure cleanup is done.
    // The existing native GPU tests use the same Windows workaround for a
    // backend teardown hang; the process exits immediately after `run`.
    #[cfg(windows)]
    std::mem::forget((queue, device, adapter, instance));
    #[cfg(not(windows))]
    drop((queue, device, adapter, instance));
    benchmark_result
}

async fn run_cases(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    cases: &[BenchCase],
    options: &Options,
) -> BenchResult<()> {
    let mut previous_suite = None;
    let mut suite_scores = BTreeMap::<(&'static str, &'static str), Vec<f64>>::new();
    for (case_index, case) in cases.iter().enumerate() {
        if previous_suite != Some(case.suite) {
            println!("\n=== suite {} ===", case.suite);
            previous_suite = Some(case.suite);
        }

        let role = if case.report { "case" } else { "warmup" };
        println!(
            "\n[{}/{}] {} {}: shape={:?} batch={}",
            case_index + 1,
            cases.len(),
            role,
            case.label,
            case.shape,
            case.batch,
        );

        if options.compare_max_buffer_bytes.is_some() {
            let precision = options.precision.single_precision().ok_or_else(|| {
                input_error("precision=both is unavailable for segment-cap comparison mode")
            })?;
            let result = run_compare_case(device, queue, case, options, precision).await;
            device
                .poll(wgpu::PollType::Poll)
                .map_err(|error| contextual_error("polling after a comparison case", error))?;
            let cache_cleared = clear_thread_local_pipeline_cache(device);
            println!("pipeline_cache_cleared_after_case={cache_cleared}");
            device
                .poll(wgpu::PollType::Poll)
                .map_err(|error| contextual_error("reclaiming cleared cache resources", error))?;
            let result = result.map_err(|error| {
                contextual_boxed_error(
                    format!(
                        "running segment-cap comparison {} shape={:?} batch={}",
                        case.label, case.shape, case.batch
                    ),
                    error,
                )
            })?;
            print_compare_result(case, options, &result)?;
            continue;
        }

        let mut completed_results = Vec::new();
        for precision in options.precision.precisions() {
            println!("precision={} starting case", precision.as_str());
            let result = run_case(device, queue, case, options, precision).await;
            device
                .poll(wgpu::PollType::Poll)
                .map_err(|error| contextual_error("polling after a benchmark case", error))?;
            let cache_cleared = clear_thread_local_pipeline_cache(device);
            println!(
                "precision={} pipeline_cache_cleared_after_case={cache_cleared}",
                precision.as_str()
            );
            device
                .poll(wgpu::PollType::Poll)
                .map_err(|error| contextual_error("reclaiming cleared cache resources", error))?;

            let result = result.map_err(|error| {
                contextual_boxed_error(
                    format!(
                        "running suite {} case {} precision={} shape={:?} batch={}",
                        case.suite,
                        case.label,
                        precision.as_str(),
                        case.shape,
                        case.batch
                    ),
                    error,
                )
            })?;

            if case.report {
                let score = print_case_result(case, options, &result)?;
                suite_scores
                    .entry((case.suite, precision.as_str()))
                    .or_default()
                    .push(score);
            } else {
                println!(
                    "warmup {} precision={} complete; timing intentionally excluded from reported results",
                    case.label,
                    precision.as_str(),
                );
            }
            completed_results.push(result);
        }

        if case.report && options.precision == PrecisionMode::Both {
            let f32_result = completed_results
                .iter()
                .find(|result| result.precision == FftPrecision::F32)
                .ok_or_else(|| input_error("precision comparison produced no f32 result"))?;
            let f64_result = completed_results
                .iter()
                .find(|result| result.precision == FftPrecision::F64)
                .ok_or_else(|| input_error("precision comparison produced no f64 result"))?;
            let f32_statistics = statistics(&f32_result.run_pair_ms)?;
            let f64_statistics = statistics(&f64_result.run_pair_ms)?;
            let ratio = f64_statistics.mean_ms / f32_statistics.mean_ms;
            println!(
                "PRECISION_COMPARE suite={} label={} shape={:?} batch={} f32_logical_buffer_bytes={} f64_logical_buffer_bytes={} f32_num_iter={} f64_num_iter={} f32_avg_pair_ms={:.9} f64_avg_pair_ms={:.9} f64_over_f32_avg_pair_time_ratio={:.9} precision_order=f32-then-f64 same_process=true same_adapter=true same_device=true",
                case.suite,
                case.label,
                case.shape,
                case.batch,
                f32_result.buffer_size,
                f64_result.buffer_size,
                f32_result.num_iter,
                f64_result.num_iter,
                f32_statistics.mean_ms,
                f64_statistics.mean_ms,
                ratio,
            );
        }
        if case.report && options.precision == PrecisionMode::Df64F64 {
            let df64_result = completed_results
                .iter()
                .find(|result| result.precision == FftPrecision::Df64)
                .ok_or_else(|| input_error("df64-f64 comparison produced no df64 result"))?;
            let f64_result = completed_results
                .iter()
                .find(|result| result.precision == FftPrecision::F64)
                .ok_or_else(|| input_error("df64-f64 comparison produced no native-f64 result"))?;
            if df64_result.buffer_size != f64_result.buffer_size
                || df64_result.external_io_bytes != f64_result.external_io_bytes
                || df64_result.num_iter != f64_result.num_iter
            {
                return Err(input_error(format!(
                    "df64-f64 topology mismatch: df64 buffer/io/iterations={}/{}/{} versus f64={}/{}/{}",
                    df64_result.buffer_size,
                    df64_result.external_io_bytes,
                    df64_result.num_iter,
                    f64_result.buffer_size,
                    f64_result.external_io_bytes,
                    f64_result.num_iter,
                )));
            }
            let df64_statistics = statistics(&df64_result.run_pair_ms)?;
            let f64_statistics = statistics(&f64_result.run_pair_ms)?;
            let ratio = df64_statistics.mean_ms / f64_statistics.mean_ms;
            println!(
                "DF64_F64_COMPARE suite={} label={} shape={:?} batch={} logical_buffer_bytes={} external_io_bytes={} num_iter={} df64_avg_pair_ms={:.9} f64_avg_pair_ms={:.9} df64_over_f64_avg_pair_time_ratio={:.9} precision_order=df64-then-f64 same_process=true same_adapter=true same_device=true identical_topology=true",
                case.suite,
                case.label,
                case.shape,
                case.batch,
                df64_result.buffer_size,
                df64_result.external_io_bytes,
                df64_result.num_iter,
                df64_statistics.mean_ms,
                f64_statistics.mean_ms,
                ratio,
            );
        }
    }

    let default_methodology = options.runs == DEFAULT_RUNS
        && options.iter_cap == DEFAULT_ITER_CAP
        && options.max_cases.is_none();
    for ((suite, precision), scores) in suite_scores {
        let mean_score = scores.iter().sum::<f64>() / scores.len() as f64;
        let source_exact_iteration_budget = matches!(suite, "sample0" | "sample1000");
        let upstream_suite =
            source_exact_iteration_budget || matches!(suite, "sample3" | "sample7");
        let normalized_methodology = default_methodology && upstream_suite;
        println!(
            "SUITE_RESULT suite={} precision={} reported_cases={} benchmark_score_mean_KiB_per_ms={:.3} vkfft_integer_score={} normalized_default_methodology={} source_exact_iteration_budget={}",
            suite,
            precision,
            scores.len(),
            mean_score,
            mean_score as u64,
            normalized_methodology,
            source_exact_iteration_budget,
        );
    }
    Ok(())
}

async fn run_case(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    case: &BenchCase,
    options: &Options,
    precision: FftPrecision,
) -> BenchResult<CaseResult> {
    let logical_elements = checked_case_elements(case)?;
    let expected_buffer_size = u64::try_from(logical_elements)
        .map_err(|_| input_error("logical element count does not fit u64"))?
        .checked_mul(precision.complex_size_bytes())
        .ok_or_else(|| input_error("logical buffer size overflow"))?;
    benchmark_config(case, false, precision).validate()?;
    benchmark_config(case, true, precision).validate()?;
    let max_buffer_size = device.limits().max_buffer_size;
    if expected_buffer_size > max_buffer_size {
        return Err(input_error(format!(
            "logical buffer requires {expected_buffer_size} bytes but device max_buffer_size is {max_buffer_size}"
        )));
    }
    let external_io_bytes = expected_buffer_size
        .checked_mul(2)
        .ok_or_else(|| input_error("out-of-place I/O allocation size overflow"))?;
    let initialization_seed_bytes = expected_buffer_size.min(INITIALIZATION_SEED_BYTES);
    let num_iter = (ITER_TRAFFIC_BUDGET_BYTES / expected_buffer_size)
        .clamp(1, DEFAULT_ITER_CAP)
        .min(options.iter_cap);

    println!(
        "precision={} complex_element_bytes={} logical_elements={} logical_buffer_bytes={} logical_buffer_MiB={:.3} external_io_bytes={} initialization_seed_bytes={} num_iter={}",
        precision.as_str(),
        precision.complex_size_bytes(),
        logical_elements,
        expected_buffer_size,
        expected_buffer_size as f64 / 1024_f64.powi(2),
        external_io_bytes,
        initialization_seed_bytes,
        num_iter,
    );

    let usage =
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;
    let error_scopes = push_gpu_error_scopes(device);
    let buffer_a = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.bench.buffer_a"),
        size: expected_buffer_size,
        usage,
        mapped_at_creation: false,
    });
    let buffer_b = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.bench.buffer_b"),
        size: expected_buffer_size,
        usage,
        mapped_at_creation: false,
    });
    let initialization_seed = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.bench.initialization_seed"),
        size: initialization_seed_bytes,
        usage: wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: true,
    });
    pop_gpu_error_scopes(error_scopes, "allocating benchmark data buffers").await?;
    fill_initialization_seed(&initialization_seed, precision)?;

    let mut route = None;
    let mut execution_kind = None;
    let mut axis_kinds = None;
    let mut graph_stage_count = None;
    let mut traffic_equivalent_pass_count = None;
    let mut pass_count_method = None;
    let mut plan_workspace_requirement_bytes = None;
    let mut diagnostic_helper_requirement_total = None;
    let mut diagnostic_helper_requirements = None;
    let mut arena_segment_layout = None;
    let mut segmented_burst_depth = None;
    let mut run_pair_ms = Vec::with_capacity(options.runs);

    for run_index in 0..options.runs {
        println!(
            "run {}/{}: recreating forward and inverse plans",
            run_index + 1,
            options.runs
        );
        initialize_buffers(
            device,
            queue,
            &initialization_seed,
            &buffer_a,
            &buffer_b,
            expected_buffer_size,
            options.wait_timeout,
        )?;

        let error_scopes = push_gpu_error_scopes(device);
        let plans = (|| -> BenchResult<(FftPlan, FftPlan)> {
            let forward = FftPlan::c2c_with_diagnostics(
                device,
                queue,
                benchmark_config(case, false, precision),
            )?;
            let inverse = FftPlan::c2c_with_diagnostics(
                device,
                queue,
                benchmark_config(case, true, precision),
            )?;
            Ok((forward, inverse))
        })();
        pop_gpu_error_scopes(error_scopes, "creating forward and inverse FFT plans").await?;
        let (forward, inverse) = plans?;

        let forward_size = forward.required_buffer_size_bytes();
        let inverse_size = inverse.required_buffer_size_bytes();
        if forward_size != expected_buffer_size || inverse_size != expected_buffer_size {
            return Err(input_error(format!(
                "plan buffer-size mismatch: expected {expected_buffer_size}, forward {forward_size}, inverse {inverse_size}"
            )));
        }

        let forward_diagnostics = forward.diagnostics();
        let inverse_diagnostics = inverse.diagnostics();
        let this_plan_workspace_requirement_bytes = forward
            .workspace_size_bytes()
            .checked_add(inverse.workspace_size_bytes())
            .ok_or_else(|| input_error("combined plan workspace requirement overflow"))?;
        let this_diagnostic_helper_requirement_bytes =
            diagnostic_helper_requirement_bytes(&forward_diagnostics)?
                .checked_add(diagnostic_helper_requirement_bytes(&inverse_diagnostics)?)
                .ok_or_else(|| input_error("combined plan helper requirement overflow"))?;
        let this_diagnostic_helper_requirements =
            combined_helper_requirement_inventory(&forward_diagnostics, &inverse_diagnostics)?;
        let this_arena_segment_bytes = arena_segment_bytes(&forward_diagnostics);
        let this_segmented_burst_depth =
            segmented_burst_depth_from_diagnostics(&forward_diagnostics)?;
        let inverse_segmented_burst_depth =
            segmented_burst_depth_from_diagnostics(&inverse_diagnostics)?;
        if this_segmented_burst_depth != inverse_segmented_burst_depth {
            return Err(input_error(format!(
                "forward/inverse segmented burst-depth mismatch: {this_segmented_burst_depth} versus {inverse_segmented_burst_depth}"
            )));
        }
        let (forward_graph_stages, forward_passes, forward_pass_method) =
            compute_pass_count(&forward_diagnostics)?;
        let (inverse_graph_stages, inverse_passes, inverse_pass_method) =
            compute_pass_count(&inverse_diagnostics)?;
        if forward_graph_stages != inverse_graph_stages {
            return Err(input_error(format!(
                "forward/inverse graph-stage mismatch: {forward_graph_stages} versus {inverse_graph_stages}"
            )));
        }
        if forward_passes != inverse_passes {
            return Err(input_error(format!(
                "forward/inverse compute-pass mismatch: {forward_passes} versus {inverse_passes}"
            )));
        }
        if forward_pass_method != inverse_pass_method {
            return Err(input_error(format!(
                "forward/inverse pass-count method mismatch: {forward_pass_method} versus {inverse_pass_method}"
            )));
        }

        let this_route = format!("{:?}", forward.route());
        let inverse_route = format!("{:?}", inverse.route());
        if this_route != inverse_route {
            return Err(input_error(format!(
                "forward/inverse route mismatch: {this_route} versus {inverse_route}"
            )));
        }
        let this_execution_kind = forward_diagnostics
            .route()
            .execution_kind
            .clone()
            .unwrap_or_else(|| "unknown".to_owned());
        let inverse_execution_kind = inverse_diagnostics
            .route()
            .execution_kind
            .clone()
            .unwrap_or_else(|| "unknown".to_owned());
        if this_execution_kind != inverse_execution_kind {
            return Err(input_error(format!(
                "forward/inverse execution-kind mismatch: {this_execution_kind} versus {inverse_execution_kind}"
            )));
        }
        let this_axis_kinds = format!("{:?}", forward.axis_kinds());
        let inverse_axis_kinds = format!("{:?}", inverse.axis_kinds());
        if this_axis_kinds != inverse_axis_kinds {
            return Err(input_error(format!(
                "forward/inverse axis-kind mismatch: {this_axis_kinds} versus {inverse_axis_kinds}"
            )));
        }

        ensure_consistent(&mut route, this_route, "route")?;
        ensure_consistent(&mut execution_kind, this_execution_kind, "execution kind")?;
        ensure_consistent(&mut axis_kinds, this_axis_kinds, "axis kinds")?;
        ensure_consistent(
            &mut graph_stage_count,
            forward_graph_stages,
            "graph stage count",
        )?;
        ensure_consistent(
            &mut traffic_equivalent_pass_count,
            forward_passes,
            "traffic-equivalent pass count",
        )?;
        ensure_consistent(
            &mut pass_count_method,
            forward_pass_method,
            "compute pass-count method",
        )?;
        ensure_consistent(
            &mut plan_workspace_requirement_bytes,
            this_plan_workspace_requirement_bytes,
            "combined plan workspace requirement bytes",
        )?;
        ensure_consistent(
            &mut diagnostic_helper_requirement_total,
            this_diagnostic_helper_requirement_bytes,
            "combined diagnostic plan helper bytes",
        )?;
        ensure_consistent(
            &mut diagnostic_helper_requirements,
            this_diagnostic_helper_requirements,
            "diagnostic helper requirement inventory",
        )?;
        ensure_consistent_debug(
            &mut arena_segment_layout,
            this_arena_segment_bytes,
            "segmented arena layout",
        )?;
        ensure_consistent(
            &mut segmented_burst_depth,
            this_segmented_burst_depth,
            "segmented burst depth",
        )?;

        // Plan construction queues parameter and LUT uploads. Flush and wait for
        // them before the timed submission so setup traffic is excluded.
        wait_for_submission(
            device,
            queue.submit([]),
            "flushing plan uploads before timing",
            options.wait_timeout,
        )?;

        let error_scopes = push_gpu_error_scopes(device);
        let command_buffer = (|| -> BenchResult<wgpu::CommandBuffer> {
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("wgpu_fft.bench.pairs"),
            });
            for iteration in 0..num_iter {
                forward
                    .execute_checked(device, &mut encoder, &buffer_a, &buffer_b)
                    .map_err(|error| {
                        contextual_error(
                            format!("recording forward FFT for iteration {iteration}"),
                            error,
                        )
                    })?;
                inverse
                    .execute_checked(device, &mut encoder, &buffer_b, &buffer_a)
                    .map_err(|error| {
                        contextual_error(
                            format!("recording inverse FFT for iteration {iteration}"),
                            error,
                        )
                    })?;
            }
            Ok(encoder.finish())
        })();
        pop_gpu_error_scopes(error_scopes, "recording the timed FFT command buffer").await?;
        let command_buffer = command_buffer?;

        println!(
            "run {}/{}: submitting {} precision={} FFT+iFFT pairs to {:?} via Vulkan",
            run_index + 1,
            options.runs,
            num_iter,
            precision.as_str(),
            forward.route(),
        );
        let submit_error_scopes = push_gpu_error_scopes(device);
        let start = Instant::now();
        let submission = queue.submit([command_buffer]);
        let wait_result = wait_for_submission(
            device,
            submission,
            "waiting for timed FFT submission",
            options.wait_timeout,
        );
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        let submit_scope_result = pop_gpu_error_scopes(
            submit_error_scopes,
            "submitting and executing timed FFT work",
        )
        .await;
        wait_result?;
        submit_scope_result?;
        let pair_ms = elapsed_ms / num_iter as f64;
        if !pair_ms.is_finite() || pair_ms <= 0.0 {
            return Err(input_error(format!(
                "invalid elapsed time: total {elapsed_ms} ms, per pair {pair_ms} ms"
            )));
        }
        if case.report {
            println!(
                "run {}/{} complete: total_ms={:.6} pair_ms={:.6}",
                run_index + 1,
                options.runs,
                elapsed_ms,
                pair_ms,
            );
        } else {
            println!("warmup run {}/{} complete", run_index + 1, options.runs);
        }
        run_pair_ms.push(pair_ms);

        drop((forward, inverse));
        device
            .poll(wgpu::PollType::Poll)
            .map_err(|error| contextual_error("reclaiming plan resources between runs", error))?;
    }

    Ok(CaseResult {
        precision,
        route: route.ok_or_else(|| input_error("benchmark produced no route"))?,
        execution_kind: execution_kind
            .ok_or_else(|| input_error("benchmark produced no execution kind"))?,
        axis_kinds: axis_kinds.ok_or_else(|| input_error("benchmark produced no axis kinds"))?,
        graph_stage_count: graph_stage_count
            .ok_or_else(|| input_error("benchmark produced no graph stage count"))?,
        traffic_equivalent_pass_count: traffic_equivalent_pass_count
            .ok_or_else(|| input_error("benchmark produced no traffic-equivalent pass count"))?,
        pass_count_method: pass_count_method
            .ok_or_else(|| input_error("benchmark produced no pass-count method"))?,
        buffer_size: expected_buffer_size,
        external_io_bytes,
        initialization_seed_bytes,
        plan_workspace_requirement_bytes: plan_workspace_requirement_bytes
            .ok_or_else(|| input_error("benchmark produced no plan workspace requirement"))?,
        diagnostic_helper_requirement_bytes: diagnostic_helper_requirement_total
            .ok_or_else(|| input_error("benchmark produced no plan helper requirement"))?,
        diagnostic_helper_requirements: diagnostic_helper_requirements
            .ok_or_else(|| input_error("benchmark produced no plan helper inventory"))?,
        arena_segment_bytes: arena_segment_layout
            .ok_or_else(|| input_error("benchmark produced no segmented arena layout"))?,
        segmented_burst_depth: segmented_burst_depth
            .ok_or_else(|| input_error("benchmark produced no segmented burst depth"))?,
        num_iter,
        run_pair_ms,
    })
}

impl VariantAccumulator {
    fn record(&mut self, metadata: PlanPairMetadata, pair_ms: f64) -> BenchResult<()> {
        ensure_consistent(&mut self.route, metadata.route, "route")?;
        ensure_consistent(
            &mut self.execution_kind,
            metadata.execution_kind,
            "execution kind",
        )?;
        ensure_consistent(&mut self.axis_kinds, metadata.axis_kinds, "axis kinds")?;
        ensure_consistent(
            &mut self.graph_stage_count,
            metadata.graph_stage_count,
            "graph stage count",
        )?;
        ensure_consistent(
            &mut self.traffic_equivalent_pass_count,
            metadata.traffic_equivalent_pass_count,
            "traffic-equivalent pass count",
        )?;
        ensure_consistent(
            &mut self.pass_count_method,
            metadata.pass_count_method,
            "pass-count method",
        )?;
        ensure_consistent(
            &mut self.plan_workspace_requirement_bytes,
            metadata.plan_workspace_requirement_bytes,
            "combined plan workspace requirement bytes",
        )?;
        ensure_consistent(
            &mut self.diagnostic_helper_requirement_bytes,
            metadata.diagnostic_helper_requirement_bytes,
            "combined diagnostic plan helper bytes",
        )?;
        ensure_consistent(
            &mut self.diagnostic_helper_requirements,
            metadata.diagnostic_helper_requirements,
            "diagnostic helper requirement inventory",
        )?;
        ensure_consistent_debug(
            &mut self.arena_segment_bytes,
            metadata.arena_segment_bytes,
            "segmented arena layout",
        )?;
        ensure_consistent(
            &mut self.segmented_burst_depth,
            metadata.segmented_burst_depth,
            "segmented burst depth",
        )?;
        self.run_pair_ms.push(pair_ms);
        Ok(())
    }

    fn finish(
        self,
        precision: FftPrecision,
        buffer_size: u64,
        external_io_bytes: u64,
        initialization_seed_bytes: u64,
        num_iter: u64,
    ) -> BenchResult<CaseResult> {
        Ok(CaseResult {
            precision,
            route: self
                .route
                .ok_or_else(|| input_error("comparison variant produced no route"))?,
            execution_kind: self
                .execution_kind
                .ok_or_else(|| input_error("comparison variant produced no execution kind"))?,
            axis_kinds: self
                .axis_kinds
                .ok_or_else(|| input_error("comparison variant produced no axis kinds"))?,
            graph_stage_count: self
                .graph_stage_count
                .ok_or_else(|| input_error("comparison variant produced no graph stage count"))?,
            traffic_equivalent_pass_count: self.traffic_equivalent_pass_count.ok_or_else(|| {
                input_error("comparison variant produced no traffic-equivalent pass count")
            })?,
            pass_count_method: self
                .pass_count_method
                .ok_or_else(|| input_error("comparison variant produced no pass-count method"))?,
            buffer_size,
            external_io_bytes,
            initialization_seed_bytes,
            plan_workspace_requirement_bytes: self.plan_workspace_requirement_bytes.ok_or_else(
                || input_error("comparison variant produced no workspace requirement"),
            )?,
            diagnostic_helper_requirement_bytes: self
                .diagnostic_helper_requirement_bytes
                .ok_or_else(|| input_error("comparison variant produced no helper requirement"))?,
            diagnostic_helper_requirements: self.diagnostic_helper_requirements.ok_or_else(
                || input_error("comparison variant produced no helper requirement inventory"),
            )?,
            arena_segment_bytes: self
                .arena_segment_bytes
                .ok_or_else(|| input_error("comparison variant produced no arena layout"))?,
            segmented_burst_depth: self.segmented_burst_depth.ok_or_else(|| {
                input_error("comparison variant produced no segmented burst depth")
            })?,
            num_iter,
            run_pair_ms: self.run_pair_ms,
        })
    }
}

async fn run_compare_case(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    case: &BenchCase,
    options: &Options,
    precision: FftPrecision,
) -> BenchResult<CompareCaseResult> {
    let requested_max_bind_bytes = options
        .plan_max_bind_bytes
        .ok_or_else(|| input_error("comparison mode is missing --plan-max-bind-bytes"))?;
    let caps = options
        .compare_max_buffer_bytes
        .ok_or_else(|| input_error("comparison mode is missing --compare-max-buffer-bytes"))?;
    let logical_elements = checked_case_elements(case)?;
    let expected_buffer_size = u64::try_from(logical_elements)
        .map_err(|_| input_error("logical element count does not fit u64"))?
        .checked_mul(precision.complex_size_bytes())
        .ok_or_else(|| input_error("logical buffer size overflow"))?;
    benchmark_config(case, false, precision).validate()?;
    benchmark_config(case, true, precision).validate()?;

    let device_limits = device.limits();
    let real_max_buffer_size = device_limits.max_buffer_size;
    if expected_buffer_size > real_max_buffer_size {
        return Err(input_error(format!(
            "comparison endpoints require {expected_buffer_size} bytes but the real device max_buffer_size is {real_max_buffer_size}"
        )));
    }
    let effective_unsharded_cap = caps.unsharded.min(real_max_buffer_size);
    let effective_sharded_cap = caps.sharded.min(real_max_buffer_size);
    let device_max_bind = device_limits.max_storage_buffer_binding_size;
    let effective_unsharded_bind = requested_max_bind_bytes
        .min(device_max_bind)
        .min(effective_unsharded_cap);
    let effective_sharded_bind = requested_max_bind_bytes
        .min(device_max_bind)
        .min(effective_sharded_cap);
    if effective_unsharded_bind != effective_sharded_bind {
        return Err(input_error(format!(
            "comparison must isolate maxBufferSize while keeping one effective binding cap: unsharded {effective_unsharded_bind}, sharded {effective_sharded_bind}; choose --plan-max-bind-bytes no larger than both effective buffer caps"
        )));
    }
    if effective_unsharded_bind >= expected_buffer_size {
        return Err(input_error(format!(
            "comparison requires a plan binding cap below the logical buffer size: effective bind cap {effective_unsharded_bind}, buffer {expected_buffer_size}"
        )));
    }
    if effective_unsharded_cap < expected_buffer_size {
        return Err(input_error(format!(
            "unsharded max-buffer cap must cover the logical buffer: effective cap {effective_unsharded_cap}, buffer {expected_buffer_size}"
        )));
    }
    if effective_sharded_cap >= expected_buffer_size {
        return Err(input_error(format!(
            "sharded max-buffer cap must be below the logical buffer: effective cap {effective_sharded_cap}, buffer {expected_buffer_size}"
        )));
    }

    let external_io_bytes = expected_buffer_size
        .checked_mul(2)
        .ok_or_else(|| input_error("out-of-place I/O allocation size overflow"))?;
    let initialization_seed_bytes = expected_buffer_size.min(INITIALIZATION_SEED_BYTES);
    let num_iter = (ITER_TRAFFIC_BUDGET_BYTES / expected_buffer_size)
        .clamp(1, DEFAULT_ITER_CAP)
        .min(options.iter_cap);
    println!(
        "SEGMENT_CAP_VARIANTS precision={} complex_element_bytes={} logical_elements={} logical_buffer_bytes={} real_device_max_buffer_bytes={} requested_plan_max_bind_bytes={} unsharded_effective_max_bind_bytes={} sharded_effective_max_bind_bytes={} unsharded_requested_max_buffer_bytes={} unsharded_effective_max_buffer_bytes={} sharded_requested_max_buffer_bytes={} sharded_effective_max_buffer_bytes={} num_iter={}",
        precision.as_str(),
        precision.complex_size_bytes(),
        logical_elements,
        expected_buffer_size,
        real_max_buffer_size,
        requested_max_bind_bytes,
        effective_unsharded_bind,
        effective_sharded_bind,
        caps.unsharded,
        effective_unsharded_cap,
        caps.sharded,
        effective_sharded_cap,
        num_iter,
    );

    let usage =
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;
    let error_scopes = push_gpu_error_scopes(device);
    let buffer_a = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.bench.compare.buffer_a"),
        size: expected_buffer_size,
        usage,
        mapped_at_creation: false,
    });
    let buffer_b = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.bench.compare.buffer_b"),
        size: expected_buffer_size,
        usage,
        mapped_at_creation: false,
    });
    let initialization_seed = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.bench.compare.initialization_seed"),
        size: initialization_seed_bytes,
        usage: wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: true,
    });
    pop_gpu_error_scopes(error_scopes, "allocating comparison data buffers").await?;
    fill_initialization_seed(&initialization_seed, precision)?;

    let mut unsharded = VariantAccumulator::default();
    let mut sharded = VariantAccumulator::default();
    for run_index in 0..options.runs {
        let order = if run_index % 2 == 0 {
            [("unsharded", caps.unsharded), ("sharded", caps.sharded)]
        } else {
            [("sharded", caps.sharded), ("unsharded", caps.unsharded)]
        };
        println!(
            "comparison run {}/{} variant_order={},{}",
            run_index + 1,
            options.runs,
            order[0].0,
            order[1].0
        );
        for (variant_label, max_buffer_bytes) in order {
            initialize_buffers(
                device,
                queue,
                &initialization_seed,
                &buffer_a,
                &buffer_b,
                expected_buffer_size,
                options.wait_timeout,
            )?;
            let limits = LargePolicyLimits {
                max_storage_buffer_binding_size: requested_max_bind_bytes,
                max_buffer_size: max_buffer_bytes,
            };
            let error_scopes = push_gpu_error_scopes(device);
            let plans = (|| -> BenchResult<(FftPlan, FftPlan)> {
                let create = |inverse| {
                    let config = benchmark_config(case, inverse, precision);
                    if variant_label == "sharded" {
                        FftPlan::c2c_with_large_policy_limits_and_burst_depth_for_testing(
                            device,
                            queue,
                            config,
                            limits,
                            options.segmented_burst_depth,
                        )
                    } else {
                        FftPlan::c2c_with_large_policy_limits_for_testing(
                            device, queue, config, limits,
                        )
                    }
                };
                let forward = create(false)?;
                let inverse = create(true)?;
                Ok((forward, inverse))
            })();
            pop_gpu_error_scopes(
                error_scopes,
                &format!("creating {variant_label} forward and inverse FFT plans"),
            )
            .await?;
            let (forward, inverse) = plans?;
            let metadata = inspect_plan_pair(&forward, &inverse, expected_buffer_size)?;

            wait_for_submission(
                device,
                queue.submit([]),
                &format!("flushing {variant_label} plan uploads before timing"),
                options.wait_timeout,
            )?;

            let error_scopes = push_gpu_error_scopes(device);
            let command_buffer = (|| -> BenchResult<wgpu::CommandBuffer> {
                let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("wgpu_fft.bench.compare.pairs"),
                });
                for iteration in 0..num_iter {
                    forward
                        .execute_checked(device, &mut encoder, &buffer_a, &buffer_b)
                        .map_err(|error| {
                            contextual_error(
                                format!(
                                    "recording {variant_label} forward FFT for iteration {iteration}"
                                ),
                                error,
                            )
                        })?;
                    inverse
                        .execute_checked(device, &mut encoder, &buffer_b, &buffer_a)
                        .map_err(|error| {
                            contextual_error(
                                format!(
                                    "recording {variant_label} inverse FFT for iteration {iteration}"
                                ),
                                error,
                            )
                        })?;
                }
                Ok(encoder.finish())
            })();
            pop_gpu_error_scopes(
                error_scopes,
                &format!("recording the timed {variant_label} FFT command buffer"),
            )
            .await?;
            let command_buffer = command_buffer?;

            let submit_error_scopes = push_gpu_error_scopes(device);
            let start = Instant::now();
            let submission = queue.submit([command_buffer]);
            let wait_result = wait_for_submission(
                device,
                submission,
                &format!("waiting for timed {variant_label} FFT submission"),
                options.wait_timeout,
            );
            let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
            let submit_scope_result = pop_gpu_error_scopes(
                submit_error_scopes,
                &format!("submitting and executing timed {variant_label} FFT work"),
            )
            .await;
            wait_result?;
            submit_scope_result?;
            let pair_ms = elapsed_ms / num_iter as f64;
            if !pair_ms.is_finite() || pair_ms <= 0.0 {
                return Err(input_error(format!(
                    "invalid {variant_label} elapsed time: total {elapsed_ms} ms, per pair {pair_ms} ms"
                )));
            }
            println!(
                "comparison run {}/{} variant={} total_ms={:.6} pair_ms={:.6} route={} execution_kind={}",
                run_index + 1,
                options.runs,
                variant_label,
                elapsed_ms,
                pair_ms,
                metadata.route,
                metadata.execution_kind,
            );
            match variant_label {
                "unsharded" => unsharded.record(metadata, pair_ms)?,
                "sharded" => sharded.record(metadata, pair_ms)?,
                _ => return Err(input_error("internal comparison variant label error")),
            }

            drop((forward, inverse));
            device.poll(wgpu::PollType::Poll).map_err(|error| {
                contextual_error(
                    format!("reclaiming {variant_label} plan resources between runs"),
                    error,
                )
            })?;
        }
    }

    let result = CompareCaseResult {
        unsharded: CompareVariantResult {
            label: "unsharded",
            requested_max_buffer_bytes: caps.unsharded,
            effective_max_bind_bytes: effective_unsharded_bind,
            effective_max_buffer_bytes: effective_unsharded_cap,
            result: unsharded.finish(
                precision,
                expected_buffer_size,
                external_io_bytes,
                initialization_seed_bytes,
                num_iter,
            )?,
        },
        sharded: CompareVariantResult {
            label: "sharded",
            requested_max_buffer_bytes: caps.sharded,
            effective_max_bind_bytes: effective_sharded_bind,
            effective_max_buffer_bytes: effective_sharded_cap,
            result: sharded.finish(
                precision,
                expected_buffer_size,
                external_io_bytes,
                initialization_seed_bytes,
                num_iter,
            )?,
        },
    };
    if result.unsharded.result.execution_kind != "out-of-core-four-step" {
        return Err(input_error(format!(
            "unsharded comparison variant selected unexpected execution kind {}; expected out-of-core-four-step",
            result.unsharded.result.execution_kind
        )));
    }
    if result.sharded.result.execution_kind != "segmented-full-volume" {
        return Err(input_error(format!(
            "sharded comparison variant selected unexpected execution kind {}; expected segmented-full-volume",
            result.sharded.result.execution_kind
        )));
    }
    if result.unsharded.result.segmented_burst_depth != 0 {
        return Err(input_error(format!(
            "unsharded comparison variant unexpectedly reported burst depth {}",
            result.unsharded.result.segmented_burst_depth
        )));
    }
    if result.sharded.result.segmented_burst_depth != options.segmented_burst_depth {
        return Err(input_error(format!(
            "sharded comparison variant reported burst depth {}, expected {}",
            result.sharded.result.segmented_burst_depth, options.segmented_burst_depth
        )));
    }
    if !result.unsharded.result.arena_segment_bytes.is_empty() {
        return Err(input_error(format!(
            "unsharded comparison variant unexpectedly reported segmented arena buffers {:?}",
            result.unsharded.result.arena_segment_bytes
        )));
    }
    if result.sharded.result.arena_segment_bytes.len() < 2 {
        return Err(input_error(format!(
            "sharded comparison variant reported fewer than two arena segments: {:?}",
            result.sharded.result.arena_segment_bytes
        )));
    }
    let sharded_arena_bytes = result
        .sharded
        .result
        .arena_segment_bytes
        .iter()
        .try_fold(0u64, |total, &bytes| total.checked_add(bytes))
        .ok_or_else(|| input_error("sharded arena segment-byte total overflow"))?;
    if sharded_arena_bytes != expected_buffer_size
        || result
            .sharded
            .result
            .arena_segment_bytes
            .iter()
            .any(|&bytes| bytes > effective_sharded_cap)
    {
        return Err(input_error(format!(
            "sharded arena layout {:?} does not exactly cover {} bytes within cap {}",
            result.sharded.result.arena_segment_bytes, expected_buffer_size, effective_sharded_cap
        )));
    }
    Ok(result)
}

fn inspect_plan_pair(
    forward: &FftPlan,
    inverse: &FftPlan,
    expected_buffer_size: u64,
) -> BenchResult<PlanPairMetadata> {
    let forward_size = forward.required_buffer_size_bytes();
    let inverse_size = inverse.required_buffer_size_bytes();
    if forward_size != expected_buffer_size || inverse_size != expected_buffer_size {
        return Err(input_error(format!(
            "plan buffer-size mismatch: expected {expected_buffer_size}, forward {forward_size}, inverse {inverse_size}"
        )));
    }
    let forward_diagnostics = forward.diagnostics();
    let inverse_diagnostics = inverse.diagnostics();
    let route = format!("{:?}", forward.route());
    let inverse_route = format!("{:?}", inverse.route());
    if route != inverse_route {
        return Err(input_error(format!(
            "forward/inverse route mismatch: {route} versus {inverse_route}"
        )));
    }
    let execution_kind = forward_diagnostics
        .route()
        .execution_kind
        .clone()
        .unwrap_or_else(|| "unknown".to_owned());
    let inverse_execution_kind = inverse_diagnostics
        .route()
        .execution_kind
        .clone()
        .unwrap_or_else(|| "unknown".to_owned());
    if execution_kind != inverse_execution_kind {
        return Err(input_error(format!(
            "forward/inverse execution-kind mismatch: {execution_kind} versus {inverse_execution_kind}"
        )));
    }
    let axis_kinds = format!("{:?}", forward.axis_kinds());
    let inverse_axis_kinds = format!("{:?}", inverse.axis_kinds());
    if axis_kinds != inverse_axis_kinds {
        return Err(input_error(format!(
            "forward/inverse axis-kind mismatch: {axis_kinds} versus {inverse_axis_kinds}"
        )));
    }
    let (forward_graph_stages, forward_passes, forward_pass_method) =
        compute_pass_count(&forward_diagnostics)?;
    let (inverse_graph_stages, inverse_passes, inverse_pass_method) =
        compute_pass_count(&inverse_diagnostics)?;
    if forward_graph_stages != inverse_graph_stages
        || forward_passes != inverse_passes
        || forward_pass_method != inverse_pass_method
    {
        return Err(input_error(format!(
            "forward/inverse pass metadata mismatch: graph {forward_graph_stages}/{inverse_graph_stages}, traffic {forward_passes}/{inverse_passes}, method {forward_pass_method}/{inverse_pass_method}"
        )));
    }
    let plan_workspace_requirement_bytes = forward
        .workspace_size_bytes()
        .checked_add(inverse.workspace_size_bytes())
        .ok_or_else(|| input_error("combined plan workspace requirement overflow"))?;
    let diagnostic_helper_requirement_bytes =
        diagnostic_helper_requirement_bytes(&forward_diagnostics)?
            .checked_add(diagnostic_helper_requirement_bytes(&inverse_diagnostics)?)
            .ok_or_else(|| input_error("combined plan helper requirement overflow"))?;
    let forward_arena_segments = arena_segment_bytes(&forward_diagnostics);
    let inverse_arena_segments = arena_segment_bytes(&inverse_diagnostics);
    if forward_arena_segments != inverse_arena_segments {
        return Err(input_error(format!(
            "forward/inverse segmented arena mismatch: {forward_arena_segments:?} versus {inverse_arena_segments:?}"
        )));
    }
    let forward_burst_depth = segmented_burst_depth_from_diagnostics(&forward_diagnostics)?;
    let inverse_burst_depth = segmented_burst_depth_from_diagnostics(&inverse_diagnostics)?;
    if forward_burst_depth != inverse_burst_depth {
        return Err(input_error(format!(
            "forward/inverse segmented burst-depth mismatch: {forward_burst_depth} versus {inverse_burst_depth}"
        )));
    }
    Ok(PlanPairMetadata {
        route,
        execution_kind,
        axis_kinds,
        graph_stage_count: forward_graph_stages,
        traffic_equivalent_pass_count: forward_passes,
        pass_count_method: forward_pass_method,
        plan_workspace_requirement_bytes,
        diagnostic_helper_requirement_bytes,
        diagnostic_helper_requirements: combined_helper_requirement_inventory(
            &forward_diagnostics,
            &inverse_diagnostics,
        )?,
        arena_segment_bytes: forward_arena_segments,
        segmented_burst_depth: forward_burst_depth,
    })
}

fn print_compare_result(
    case: &BenchCase,
    options: &Options,
    result: &CompareCaseResult,
) -> BenchResult<()> {
    for variant in [&result.unsharded, &result.sharded] {
        let statistics = statistics(&variant.result.run_pair_ms)?;
        let traffic_multiplier = variant
            .result
            .traffic_equivalent_pass_count
            .checked_mul(4)
            .ok_or_else(|| input_error("comparison traffic multiplier overflow"))?;
        let seconds_per_pair = statistics.mean_ms / 1000.0;
        let traffic_bytes_per_pair = variant.result.buffer_size as f64 * traffic_multiplier as f64;
        let bandwidth_gib_s = traffic_bytes_per_pair / seconds_per_pair / 1024_f64.powi(3);
        println!(
            "COMPARE_VARIANT suite={} label={} precision={} variant={} shape={:?} batch={} segmented_burst_depth={} effective_max_bind_bytes={} requested_max_buffer_bytes={} effective_max_buffer_bytes={} logical_buffer_bytes={} runs={} num_iter={} raw_pair_ms={:?} avg_pair_ms={:.6} stderr_ms={} stderr_defined={} route={} execution_kind={} graph_stages_per_fft={} traffic_equivalent_passes_per_fft={} pass_count_method={} traffic_multiplier_per_pair={} effective_traffic_bandwidth_GiB_s={:.3} plan_workspace_requirement_bytes={} combined_diagnostic_helper_requirement_bytes={} helper_requirements={} arena_segment_count={} arena_segment_bytes={:?}",
            case.suite,
            case.label,
            variant.result.precision.as_str(),
            variant.label,
            case.shape,
            case.batch,
            variant.result.segmented_burst_depth,
            variant.effective_max_bind_bytes,
            variant.requested_max_buffer_bytes,
            variant.effective_max_buffer_bytes,
            variant.result.buffer_size,
            options.runs,
            variant.result.num_iter,
            variant.result.run_pair_ms,
            statistics.mean_ms,
            statistics
                .stderr_ms
                .map_or_else(|| "NA".to_owned(), |value| format!("{value:.6}")),
            statistics.stderr_ms.is_some(),
            variant.result.route,
            variant.result.execution_kind,
            variant.result.graph_stage_count,
            variant.result.traffic_equivalent_pass_count,
            variant.result.pass_count_method,
            traffic_multiplier,
            bandwidth_gib_s,
            variant.result.plan_workspace_requirement_bytes,
            variant.result.diagnostic_helper_requirement_bytes,
            variant.result.diagnostic_helper_requirements,
            variant.result.arena_segment_bytes.len(),
            variant.result.arena_segment_bytes,
        );
    }
    let unsharded_statistics = statistics(&result.unsharded.result.run_pair_ms)?;
    let sharded_statistics = statistics(&result.sharded.result.run_pair_ms)?;
    let sharded_over_unsharded = sharded_statistics.mean_ms / unsharded_statistics.mean_ms;
    println!(
        "COMPARE_RESULT suite={} label={} precision={} shape={:?} batch={} segmented_burst_depth={} unsharded_avg_pair_ms={:.6} sharded_avg_pair_ms={:.6} sharded_over_unsharded_ratio={:.6} sharding_overhead_percent={:.3} variant_order=alternated-by-run",
        case.suite,
        case.label,
        result.unsharded.result.precision.as_str(),
        case.shape,
        case.batch,
        result.sharded.result.segmented_burst_depth,
        unsharded_statistics.mean_ms,
        sharded_statistics.mean_ms,
        sharded_over_unsharded,
        (sharded_over_unsharded - 1.0) * 100.0,
    );
    Ok(())
}

fn initialize_buffers(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    initialization_seed: &wgpu::Buffer,
    buffer_a: &wgpu::Buffer,
    buffer_b: &wgpu::Buffer,
    buffer_size: u64,
    wait_timeout: Duration,
) -> BenchResult<()> {
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("wgpu_fft.bench.initialize"),
    });
    encoder.clear_buffer(buffer_b, 0, None);
    let seed_size = initialization_seed.size();
    let mut offset = 0u64;
    while offset < buffer_size {
        let copy_size = seed_size.min(buffer_size - offset);
        encoder.copy_buffer_to_buffer(initialization_seed, 0, buffer_a, offset, copy_size);
        offset += copy_size;
    }
    wait_for_submission(
        device,
        queue.submit([encoder.finish()]),
        "initializing benchmark buffers with deterministic nonzero data",
        wait_timeout,
    )
}

fn fill_initialization_seed(seed: &wgpu::Buffer, precision: FftPrecision) -> BenchResult<()> {
    let mut mapped = seed.slice(..).get_mapped_range_mut();
    match precision {
        FftPrecision::F32 => {
            let value_count = mapped.len() / std::mem::size_of::<f32>();
            let mut values = vec![0.0f32; value_count];
            for (index, value) in values.iter_mut().enumerate() {
                let numerator = ((index as u64).wrapping_mul(17) % 251 + 1) as f32;
                let magnitude = numerator / 251.0;
                *value = if index % 2 == 0 {
                    magnitude
                } else {
                    -magnitude
                };
            }
            mapped.copy_from_slice(bytemuck::cast_slice(&values));
        }
        FftPrecision::F64 => {
            let value_count = mapped.len() / std::mem::size_of::<f64>();
            let mut values = vec![0.0f64; value_count];
            for (index, value) in values.iter_mut().enumerate() {
                let numerator = ((index as u64).wrapping_mul(17) % 251 + 1) as f64;
                let magnitude = numerator / 251.0;
                *value = if index % 2 == 0 {
                    magnitude
                } else {
                    -magnitude
                };
            }
            mapped.copy_from_slice(bytemuck::cast_slice(&values));
        }
        FftPrecision::Df64 => {
            let complex_count = mapped.len() / FftPrecision::Df64.complex_size_bytes() as usize;
            let mut values = vec![[0.0f32; 4]; complex_count];
            for (index, value) in values.iter_mut().enumerate() {
                // Match the native-f64 scalar sequence exactly so df64-f64
                // comparison mode differs only in arithmetic representation.
                let re_index = (index as u64).wrapping_mul(2);
                let im_index = re_index + 1;
                let re = (re_index.wrapping_mul(17) % 251 + 1) as f64 / 251.0;
                let im = -((im_index.wrapping_mul(17) % 251 + 1) as f64 / 251.0);
                let re_hi = re as f32;
                let im_hi = im as f32;
                *value = [
                    re_hi,
                    (re - f64::from(re_hi)) as f32,
                    im_hi,
                    (im - f64::from(im_hi)) as f32,
                ];
            }
            mapped.copy_from_slice(bytemuck::cast_slice(&values));
        }
    }
    drop(mapped);
    seed.unmap();
    Ok(())
}

fn benchmark_config(case: &BenchCase, inverse: bool, precision: FftPrecision) -> FftConfig {
    let config = if inverse {
        FftConfig::inverse_nd(case.shape.clone())
    } else {
        FftConfig::new_nd(case.shape.clone())
    };
    config
        .with_batch(case.batch)
        .with_normalization(Normalization::None)
        .with_precision(precision)
}

fn diagnostic_helper_requirement_bytes(diagnostics: &FftDiagnostics) -> BenchResult<u64> {
    diagnostics
        .buffer_requirements()
        .iter()
        .filter(|requirement| requirement.role.starts_with("helper:"))
        .try_fold(0u64, |total, requirement| {
            total
                .checked_add(requirement.required_bytes)
                .ok_or_else(|| input_error("diagnostic helper requirement total overflow"))
        })
}

fn combined_helper_requirement_inventory(
    forward: &FftDiagnostics,
    inverse: &FftDiagnostics,
) -> BenchResult<String> {
    let mut requirements = BTreeMap::<(String, String), u64>::new();
    for diagnostics in [forward, inverse] {
        for requirement in diagnostics
            .buffer_requirements()
            .iter()
            .filter(|requirement| requirement.role.starts_with("helper:"))
        {
            let key = (requirement.role.clone(), requirement.format.clone());
            let current = requirements.get(&key).copied().unwrap_or(0);
            requirements.insert(
                key,
                current
                    .checked_add(requirement.required_bytes)
                    .ok_or_else(|| input_error("diagnostic helper inventory overflow"))?,
            );
        }
    }
    Ok(format!(
        "[{}]",
        requirements
            .into_iter()
            .map(|((role, format), bytes)| format!("{role}:{format}:{bytes}"))
            .collect::<Vec<_>>()
            .join(",")
    ))
}

fn arena_segment_bytes(diagnostics: &FftDiagnostics) -> Vec<u64> {
    diagnostics
        .buffer_requirements()
        .iter()
        .filter(|requirement| requirement.role == "helper:segmented-volume-arena")
        .map(|requirement| requirement.required_bytes)
        .collect()
}

fn segmented_burst_depth_from_diagnostics(diagnostics: &FftDiagnostics) -> BenchResult<usize> {
    let stage_a = diagnostics
        .buffer_requirements()
        .iter()
        .filter(|requirement| requirement.role == "helper:segmented-volume-burst-stage-a")
        .count();
    let stage_b = diagnostics
        .buffer_requirements()
        .iter()
        .filter(|requirement| requirement.role == "helper:segmented-volume-burst-stage-b")
        .count();
    if stage_a != stage_b {
        return Err(input_error(format!(
            "segmented burst-ring helper mismatch: stage-a={stage_a}, stage-b={stage_b}"
        )));
    }
    Ok(stage_a)
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
                format!("{context} (submission wait timeout {:?})", timeout),
                error,
            )
        })
}

fn compute_pass_count(diagnostics: &FftDiagnostics) -> BenchResult<(u64, u64, String)> {
    if !diagnostics.blockers().is_empty() {
        return Err(input_error(format!(
            "plan diagnostics contain {} blocker(s)",
            diagnostics.blockers().len()
        )));
    }

    let mut graph_count = diagnostics
        .stages()
        .iter()
        .filter(|stage| {
            matches!(
                stage.kind.as_str(),
                "kernel"
                    | "windowed-kernel"
                    | "gather-scatter"
                    | "twiddle-transpose"
                    | "permutation"
                    | "stripe-transpose"
                    | "scale"
            ) || (stage.kind == "copy"
                && (stage.label == "four-step-final-copy"
                    || stage.label.starts_with("segmented-volume-")))
        })
        .count();
    let mut traffic_count: usize = diagnostics
        .stages()
        .iter()
        .map(|stage| match stage.kind.as_str() {
            // Windowed transposes and generic axis permutations gather into
            // compact storage, run one kernel, then scatter back: three
            // full-volume read/write traffic passes per logical graph stage.
            "stripe-transpose" if stage.label.starts_with("segmented-volume-") => 1,
            "stripe-transpose" | "permutation" => 3usize,
            "copy" if stage.label == "four-step-final-copy" => 1,
            "copy" if stage.label.starts_with("segmented-volume-") => 1,
            "kernel" | "windowed-kernel" | "gather-scatter" | "twiddle-transpose" | "scale" => 1,
            _ => 0,
        })
        .sum();
    if graph_count == 0 || traffic_count == 0 {
        return Err(input_error(
            "diagnostics reported no FFT axis-pass graph stages",
        ));
    }

    let execution_kind = diagnostics
        .route()
        .execution_kind
        .as_deref()
        .unwrap_or("unknown");
    let method = if execution_kind == "batch-chunk" {
        let chunk_count = diagnostics
            .stages()
            .iter()
            .filter(|stage| stage.label == "large-chunk-copy-input")
            .count();
        if chunk_count == 0 || graph_count % chunk_count != 0 || traffic_count % chunk_count != 0 {
            return Err(input_error(format!(
                "cannot collapse batch-chunk graph: {graph_count} graph stages and {traffic_count} traffic-equivalent passes across {chunk_count} chunks"
            )));
        }
        graph_count /= chunk_count;
        traffic_count /= chunk_count;
        "estimated-batch-chunk-collapsed"
    } else if execution_kind == "normal" {
        "exact-normal-graph"
    } else if execution_kind == "out-of-core-four-step" {
        if has_expanded_prime_window_stages(diagnostics) {
            // A non-fused prime window expands its Rader/Bluestein child graph
            // over convolution length M, so counting each child stage as one
            // logical N-volume pass is an estimate rather than exact traffic.
            "estimated-four-step-prime-stage-equivalent"
        } else {
            "exact-four-step-traffic-equivalent"
        }
    } else if execution_kind == "segmented-full-volume" {
        "exact-segmented-four-step-traffic-equivalent"
    } else {
        "estimated-graph"
    };
    let graph_count = u64::try_from(graph_count)
        .map_err(|_| input_error("graph stage count does not fit u64"))?;
    let traffic_count = u64::try_from(traffic_count)
        .map_err(|_| input_error("traffic-equivalent pass count does not fit u64"))?;
    Ok((graph_count, traffic_count, method.to_owned()))
}

fn has_expanded_prime_window_stages(diagnostics: &FftDiagnostics) -> bool {
    diagnostics.stages().iter().any(|stage| {
        let prime_window = stage.label.starts_with("four-step-axis")
            && (stage.label.contains("-windowed-rader")
                || stage.label.contains("-windowed-bluestein"));
        prime_window
            && diagnostics
                .stages()
                .iter()
                .filter(|candidate| candidate.label == stage.label)
                .count()
                > 1
    })
}

fn ensure_consistent<T>(slot: &mut Option<T>, value: T, name: &str) -> BenchResult<()>
where
    T: PartialEq + fmt::Display,
{
    if let Some(previous) = slot.as_ref() {
        if previous != &value {
            return Err(input_error(format!(
                "{name} changed across recreated runs: {previous} versus {value}"
            )));
        }
    } else {
        *slot = Some(value);
    }
    Ok(())
}

fn ensure_consistent_debug<T>(slot: &mut Option<T>, value: T, name: &str) -> BenchResult<()>
where
    T: PartialEq + fmt::Debug,
{
    if let Some(previous) = slot.as_ref() {
        if previous != &value {
            return Err(input_error(format!(
                "{name} changed across recreated runs: {previous:?} versus {value:?}"
            )));
        }
    } else {
        *slot = Some(value);
    }
    Ok(())
}

fn print_case_result(case: &BenchCase, options: &Options, result: &CaseResult) -> BenchResult<f64> {
    let statistics = statistics(&result.run_pair_ms)?;
    let traffic_multiplier = result
        .traffic_equivalent_pass_count
        .checked_mul(4)
        .ok_or_else(|| input_error("traffic multiplier overflow"))?;
    let traffic_bytes_per_pair = result.buffer_size as f64 * traffic_multiplier as f64;
    let harness_owned_buffer_allocation_bytes = result
        .external_io_bytes
        .checked_add(result.initialization_seed_bytes)
        .ok_or_else(|| input_error("harness-owned buffer allocation size overflow"))?;
    let seconds_per_pair = statistics.mean_ms / 1000.0;
    let score = (result.buffer_size as f64 / 1024.0) / statistics.mean_ms;
    let bandwidth_gib_s = traffic_bytes_per_pair / seconds_per_pair / 1024_f64.powi(3);
    let bandwidth_gb_s = traffic_bytes_per_pair / seconds_per_pair / 1_000_000_000.0;

    println!(
        "RESULT suite={} label={} precision={} complex_element_bytes={} shape={:?} batch={} logical_vkfft_style_buffer_bytes={} logical_buffer_MiB={:.3} mode=out-of-place reference_mode=VkFFT-in-place external_io_allocation_bytes={} retained_initialization_seed_allocation_bytes={} harness_owned_buffer_allocation_bytes={} plan_workspace_requirement_bytes={} partial_diagnostic_helper_requirement_bytes={} memory_note=not-total-vram;diagnostic-requirements-are-not-allocations;excludes-unreported-plan-stage-temp-command-pipeline-cache-driver-resources runs={} num_iter={} avg_pair_ms={:.6} stderr_ms={} stderr_defined={} vkfft_population_spread_ms={:.6} score_KiB_per_ms={:.3} diagnostic_axis_passes_per_fft={} diagnostic_traffic_equivalent_passes_per_fft={} pass_count_method={} estimated_axis_traffic_multiplier_per_pair={} bandwidth_model=4x-diagnostic-traffic-equivalent-passes estimated_axis_traffic_bandwidth_GiB_s={:.3} estimated_axis_traffic_bandwidth_GB_s={:.3} route={} axis_kinds={}",
        case.suite,
        case.label,
        result.precision.as_str(),
        result.precision.complex_size_bytes(),
        case.shape,
        case.batch,
        result.buffer_size,
        result.buffer_size as f64 / 1024_f64.powi(2),
        result.external_io_bytes,
        result.initialization_seed_bytes,
        harness_owned_buffer_allocation_bytes,
        result.plan_workspace_requirement_bytes,
        result.diagnostic_helper_requirement_bytes,
        options.runs,
        result.num_iter,
        statistics.mean_ms,
        statistics
            .stderr_ms
            .map_or_else(|| "NA".to_owned(), |value| format!("{value:.6}")),
        statistics.stderr_ms.is_some(),
        statistics.vkfft_population_spread_ms,
        score,
        result.graph_stage_count,
        result.traffic_equivalent_pass_count,
        result.pass_count_method,
        traffic_multiplier,
        bandwidth_gib_s,
        bandwidth_gb_s,
        result.route,
        result.axis_kinds,
    );
    Ok(score)
}

fn statistics(samples: &[f64]) -> BenchResult<Statistics> {
    if samples.is_empty() {
        return Err(input_error("cannot calculate statistics without samples"));
    }
    if samples.iter().any(|value| !value.is_finite()) {
        return Err(input_error(
            "cannot calculate statistics from non-finite samples",
        ));
    }
    let sample_count = samples.len() as f64;
    let mean_ms = samples.iter().sum::<f64>() / sample_count;
    let squared_deviations = samples
        .iter()
        .map(|value| {
            let deviation = value - mean_ms;
            deviation * deviation
        })
        .sum::<f64>();
    let stderr_ms = if samples.len() > 1 {
        Some((squared_deviations / (sample_count * (sample_count - 1.0))).sqrt())
    } else {
        None
    };
    let vkfft_population_spread_ms = (squared_deviations / sample_count).sqrt();
    Ok(Statistics {
        mean_ms,
        stderr_ms,
        vkfft_population_spread_ms,
    })
}

fn parse_options() -> BenchResult<Options> {
    // Cargo invokes harness-free benchmark binaries with an implicit
    // `--bench`; it is not part of this harness's CLI.
    let mut args = std::env::args().skip(1).filter(|arg| arg != "--bench");
    let Some(suite_arg) = args.next() else {
        print_usage();
        return Err(input_error("missing suite name"));
    };
    if matches!(suite_arg.as_str(), "-h" | "--help" | "help") {
        print_usage();
        std::process::exit(0);
    }
    let suite = Suite::parse(&suite_arg).ok_or_else(|| {
        input_error(format!(
            "unknown suite {suite_arg:?}; expected smoke, sample0, sample1000, sample3, sample7, all, or shape"
        ))
    })?;
    let custom_shape = if suite == Suite::Custom {
        let shape_arg = args.next().ok_or_else(|| {
            input_error("custom shape mode requires dimensions such as 1024x1024")
        })?;
        if matches!(shape_arg.as_str(), "-h" | "--help" | "help") {
            print_usage();
            std::process::exit(0);
        }
        Some(parse_shape(&shape_arg)?)
    } else {
        None
    };

    let mut options = Options {
        suite,
        precision: PrecisionMode::F32,
        custom_shape,
        custom_batch: 1,
        adapter_selector: None,
        runs: DEFAULT_RUNS,
        iter_cap: DEFAULT_ITER_CAP,
        max_cases: None,
        wait_timeout: Duration::from_secs(DEFAULT_WAIT_TIMEOUT_SECS),
        plan_max_bind_bytes: None,
        compare_max_buffer_bytes: None,
        segmented_burst_depth: DEFAULT_SEGMENTED_BURST_DEPTH,
    };
    let mut segmented_burst_depth_was_set = false;
    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--precision" => {
                let value = next_value(&mut args, "--precision")?;
                options.precision = PrecisionMode::parse(&value).ok_or_else(|| {
                    input_error(format!(
                        "unknown --precision value {value:?}; expected f32, f64, df64, both, or df64-f64"
                    ))
                })?;
            }
            "--runs" => {
                options.runs = parse_positive::<usize>(next_value(&mut args, "--runs")?, "--runs")?;
            }
            "--iter-cap" => {
                options.iter_cap =
                    parse_positive::<u64>(next_value(&mut args, "--iter-cap")?, "--iter-cap")?;
            }
            "--max-cases" => {
                options.max_cases = Some(parse_positive::<usize>(
                    next_value(&mut args, "--max-cases")?,
                    "--max-cases",
                )?);
            }
            "--adapter" => {
                let selector = next_value(&mut args, "--adapter")?;
                let selector = selector.trim();
                if selector.is_empty() {
                    return Err(input_error("--adapter must not be empty"));
                }
                options.adapter_selector = Some(selector.to_owned());
            }
            "--wait-timeout-secs" => {
                let seconds = parse_positive::<u64>(
                    next_value(&mut args, "--wait-timeout-secs")?,
                    "--wait-timeout-secs",
                )?;
                options.wait_timeout = Duration::from_secs(seconds);
            }
            "--batch" => {
                if suite != Suite::Custom {
                    return Err(input_error("--batch is valid only in custom shape mode"));
                }
                options.custom_batch =
                    parse_positive::<usize>(next_value(&mut args, "--batch")?, "--batch")?;
            }
            "--plan-max-bind-bytes" => {
                if suite != Suite::Custom {
                    return Err(input_error(
                        "--plan-max-bind-bytes is valid only in custom shape mode",
                    ));
                }
                options.plan_max_bind_bytes = Some(parse_positive::<u64>(
                    next_value(&mut args, "--plan-max-bind-bytes")?,
                    "--plan-max-bind-bytes",
                )?);
            }
            "--compare-max-buffer-bytes" => {
                if suite != Suite::Custom {
                    return Err(input_error(
                        "--compare-max-buffer-bytes is valid only in custom shape mode",
                    ));
                }
                options.compare_max_buffer_bytes = Some(parse_compare_max_buffer_bytes(
                    next_value(&mut args, "--compare-max-buffer-bytes")?,
                )?);
            }
            "--segmented-burst-depth" => {
                if suite != Suite::Custom {
                    return Err(input_error(
                        "--segmented-burst-depth is valid only in custom shape mode",
                    ));
                }
                options.segmented_burst_depth = parse_positive::<usize>(
                    next_value(&mut args, "--segmented-burst-depth")?,
                    "--segmented-burst-depth",
                )?;
                segmented_burst_depth_was_set = true;
                if options.segmented_burst_depth > 3 {
                    return Err(input_error(
                        "--segmented-burst-depth must be in the range 1..=3",
                    ));
                }
            }
            "-h" | "--help" => {
                print_usage();
                std::process::exit(0);
            }
            _ => return Err(input_error(format!("unknown argument {argument:?}"))),
        }
    }
    if options.plan_max_bind_bytes.is_some() != options.compare_max_buffer_bytes.is_some() {
        return Err(input_error(
            "--plan-max-bind-bytes and --compare-max-buffer-bytes must be provided together",
        ));
    }
    if segmented_burst_depth_was_set && options.compare_max_buffer_bytes.is_none() {
        return Err(input_error(
            "--segmented-burst-depth requires the segment-cap comparison flags",
        ));
    }
    if options.precision.single_precision().is_none() && options.compare_max_buffer_bytes.is_some()
    {
        return Err(input_error(
            "multi-precision modes are not supported with the segment-cap comparison flags; run each precision separately",
        ));
    }
    Ok(options)
}

fn next_value(args: &mut impl Iterator<Item = String>, option: &str) -> BenchResult<String> {
    args.next()
        .ok_or_else(|| input_error(format!("{option} requires a value")))
}

fn parse_positive<T>(value: String, option: &str) -> BenchResult<T>
where
    T: std::str::FromStr + PartialEq + Default,
    T::Err: Error + 'static,
{
    let parsed = value
        .parse::<T>()
        .map_err(|error| contextual_error(format!("parsing {option} value {value:?}"), error))?;
    if parsed == T::default() {
        return Err(input_error(format!("{option} must be greater than zero")));
    }
    Ok(parsed)
}

fn parse_shape(value: &str) -> BenchResult<Vec<usize>> {
    let parts = value.split(['x', 'X', ',']).collect::<Vec<_>>();
    if parts.is_empty() || parts.iter().any(|part| part.is_empty()) {
        return Err(input_error(format!(
            "invalid custom shape {value:?}; use dimensions such as 1024x1024"
        )));
    }
    let mut shape = Vec::with_capacity(parts.len());
    for part in parts {
        let dimension = part.parse::<usize>().map_err(|error| {
            contextual_error(format!("parsing shape dimension {part:?}"), error)
        })?;
        if dimension == 0 {
            return Err(input_error(
                "custom shape dimensions must be greater than zero",
            ));
        }
        shape.push(dimension);
    }
    Ok(shape)
}

fn parse_compare_max_buffer_bytes(value: String) -> BenchResult<CompareMaxBufferBytes> {
    let mut caps = value.split(',').map(str::trim);
    let unsharded = caps
        .next()
        .ok_or_else(|| {
            input_error("--compare-max-buffer-bytes requires unsharded,sharded byte caps")
        })?
        .parse::<u64>()
        .map_err(|error| {
            contextual_error(
                format!("parsing unsharded max-buffer cap from {value:?}"),
                error,
            )
        })?;
    let sharded = caps
        .next()
        .ok_or_else(|| {
            input_error("--compare-max-buffer-bytes requires unsharded,sharded byte caps")
        })?
        .parse::<u64>()
        .map_err(|error| {
            contextual_error(
                format!("parsing sharded max-buffer cap from {value:?}"),
                error,
            )
        })?;
    if caps.next().is_some() || unsharded == 0 || sharded == 0 {
        return Err(input_error(
            "--compare-max-buffer-bytes requires exactly two positive comma-separated u64 values",
        ));
    }
    Ok(CompareMaxBufferBytes { unsharded, sharded })
}

fn print_usage() {
    eprintln!(
        r#"Usage:
  cargo bench --bench fft_bench -- <smoke|sample0|sample1000|sample3|sample7|all> [--precision f32|f64|df64|both|df64-f64] [--adapter INDEX_OR_NAME] [--runs N] [--iter-cap N] [--max-cases N] [--wait-timeout-secs N]
  cargo bench --bench fft_bench -- shape <N[xN...]> [--batch N] [--precision f32|f64|df64|both|df64-f64] [--adapter INDEX_OR_NAME] [--runs N] [--iter-cap N] [--wait-timeout-secs N]
  cargo bench --bench fft_bench -- shape <N[xN...]> [--batch N] --plan-max-bind-bytes BYTES --compare-max-buffer-bytes UNSHARDED_BYTES,SHARDED_BYTES [--segmented-burst-depth 1|2|3] [--precision f32|f64|df64] [--adapter INDEX_OR_NAME] [--runs N] [--iter-cap N] [--wait-timeout-secs N]

Defaults:
  precision=f32, runs=3, iter-cap=1000, submission-wait-timeout=120 seconds.
  With one hardware Vulkan GPU it is selected automatically; multiple GPUs require --adapter.
  Precision f64/both/df64-f64 requests SHADER_F64; df64 alone requests no optional features.
  `both` reports f64/f32 timing; `df64-f64` reports the df64/native-f64 ratio with identical 16-byte-complex topology.
  Segment-cap comparison recreates both variants per run and alternates their order.
  Segmented burst depth defaults to 2 and is valid only in comparison mode; multi-precision modes are rejected there.

Examples:
  cargo bench --bench fft_bench -- smoke --adapter "RTX 5090" --runs 1 --iter-cap 2
  cargo bench --bench fft_bench -- sample0 --adapter "RTX 5090"
  cargo bench --bench fft_bench -- sample1000 --adapter 0 --runs 1 --iter-cap 1 --max-cases 2
  cargo bench --bench fft_bench -- shape 1024x1024 --batch 2 --adapter "RTX 5090" --runs 1 --iter-cap 1
  cargo bench --bench fft_bench -- shape 4096 --batch 16384 --precision both --adapter "RTX 5090" --runs 2 --iter-cap 200
  cargo bench --bench fft_bench -- shape 4096 --batch 16384 --precision df64-f64 --adapter "RTX 5090" --runs 2 --iter-cap 200
  cargo bench --bench fft_bench -- shape 2048 --batch 32768 --precision df64-f64 --adapter "RTX 5090" --runs 2 --iter-cap 200
  cargo bench --bench fft_bench -- shape 320x320x320 --plan-max-bind-bytes 16777216 --compare-max-buffer-bytes 1073741824,67108864 --segmented-burst-depth 2 --adapter "RTX 5090" --runs 2 --iter-cap 1"#
    );
}

fn build_cases(options: &Options) -> BenchResult<Vec<BenchCase>> {
    match options.suite {
        Suite::Smoke => Ok(vec![one_dimensional_case(
            "smoke",
            "N=64-batch=1024".to_owned(),
            64,
            1024,
            true,
        )?]),
        Suite::Sample0 => limited_cases(sample0_cases()?, options.max_cases),
        Suite::Sample1000 => limited_cases(sample1000_cases()?, options.max_cases),
        Suite::Sample3 => limited_cases(sample3_cases(), options.max_cases),
        Suite::Sample7 => limited_cases(sample7_cases(), options.max_cases),
        Suite::All => {
            let mut cases = Vec::new();
            cases.extend(limited_cases(sample0_cases()?, options.max_cases)?);
            cases.extend(limited_cases(sample1000_cases()?, options.max_cases)?);
            cases.extend(limited_cases(sample3_cases(), options.max_cases)?);
            cases.extend(limited_cases(sample7_cases(), options.max_cases)?);
            Ok(cases)
        }
        Suite::Custom => {
            let shape = options
                .custom_shape
                .clone()
                .ok_or_else(|| input_error("custom suite is missing its shape"))?;
            Ok(vec![BenchCase {
                suite: "custom",
                label: format!("shape={}", shape_label(&shape)),
                shape,
                batch: options.custom_batch,
                report: true,
            }])
        }
    }
}

fn limited_cases(
    cases: Vec<BenchCase>,
    max_reported_cases: Option<usize>,
) -> BenchResult<Vec<BenchCase>> {
    let Some(max_reported_cases) = max_reported_cases else {
        return Ok(cases);
    };
    let mut reported = 0usize;
    Ok(cases
        .into_iter()
        .filter(|case| {
            if !case.report {
                true
            } else if reported < max_reported_cases {
                reported += 1;
                true
            } else {
                false
            }
        })
        .collect())
}

fn sample0_cases() -> BenchResult<Vec<BenchCase>> {
    let mut cases = Vec::with_capacity(26);
    cases.push(one_dimensional_case(
        "sample0",
        "upstream-unreported-warmup-N4096".to_owned(),
        4096,
        TARGET_COMPLEX_ELEMENTS / 4096,
        false,
    )?);
    for exponent in 1u32..=25 {
        let n = 4usize
            .checked_shl(exponent)
            .ok_or_else(|| input_error(format!("sample0 size overflow at exponent {exponent}")))?;
        let batch = (TARGET_COMPLEX_ELEMENTS / n).max(1);
        cases.push(one_dimensional_case(
            "sample0",
            format!("n={exponent}-N={n}"),
            n,
            batch,
            true,
        )?);
    }
    Ok(cases)
}

fn sample1000_cases() -> BenchResult<Vec<BenchCase>> {
    let mut cases = Vec::with_capacity(4096);
    let warmup_batch = floor_power_of_two(TARGET_COMPLEX_ELEMENTS / 4096)?;
    cases.push(one_dimensional_case(
        "sample1000",
        "upstream-unreported-warmup-N4096".to_owned(),
        4096,
        warmup_batch,
        false,
    )?);
    for n in 2usize..=4096 {
        let quotient = TARGET_COMPLEX_ELEMENTS / n;
        let batch = floor_power_of_two(quotient)?;
        cases.push(one_dimensional_case(
            "sample1000",
            format!("N={n}"),
            n,
            batch,
            true,
        )?);
    }
    Ok(cases)
}

fn one_dimensional_case(
    suite: &'static str,
    label: String,
    n: usize,
    batch: usize,
    report: bool,
) -> BenchResult<BenchCase> {
    if n == 0 || batch == 0 {
        return Err(input_error(format!(
            "invalid generated case N={n}, batch={batch}"
        )));
    }
    Ok(BenchCase {
        suite,
        label,
        shape: vec![n],
        batch,
        report,
    })
}

fn floor_power_of_two(value: usize) -> BenchResult<usize> {
    if value == 0 {
        return Err(input_error("cannot find a power of two at or below zero"));
    }
    1usize
        .checked_shl(value.ilog2())
        .ok_or_else(|| input_error(format!("power-of-two overflow for {value}")))
}

fn sample3_cases() -> Vec<BenchCase> {
    SAMPLE3_DIMENSIONS
        .iter()
        .enumerate()
        .map(|(index, dimensions)| BenchCase {
            suite: "sample3",
            label: if index == 0 {
                "upstream-unreported-warmup-1024x1024".to_owned()
            } else {
                format!("system-{index}-{}", shape_label(dimensions))
            },
            shape: dimensions.to_vec(),
            batch: 1,
            report: index != 0,
        })
        .collect()
}

fn sample7_cases() -> Vec<BenchCase> {
    SAMPLE7_DIMENSIONS
        .iter()
        .enumerate()
        .map(|(index, dimensions)| BenchCase {
            suite: "sample7",
            label: if index == 0 {
                "upstream-unreported-warmup-1024x1024".to_owned()
            } else {
                format!("system-{index}-{}", shape_label(dimensions))
            },
            shape: dimensions.to_vec(),
            batch: 1,
            report: index != 0,
        })
        .collect()
}

fn checked_case_elements(case: &BenchCase) -> BenchResult<usize> {
    let per_transform = case.shape.iter().try_fold(1usize, |product, &dimension| {
        product
            .checked_mul(dimension)
            .ok_or_else(|| input_error(format!("shape product overflow for {:?}", case.shape)))
    })?;
    per_transform.checked_mul(case.batch).ok_or_else(|| {
        input_error(format!(
            "batched element-count overflow for shape {:?}, batch {}",
            case.shape, case.batch
        ))
    })
}

fn shape_label(shape: &[usize]) -> String {
    shape
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join("x")
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

fn contextual_boxed_error(context: impl Into<String>, source: Box<dyn Error>) -> Box<dyn Error> {
    Box::new(ContextError {
        context: context.into(),
        source,
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

const SAMPLE3_DIMENSIONS: [&[usize]; 39] = [
    &[1024, 1024],
    &[720, 480],
    &[1280, 720],
    &[1920, 1080],
    &[2560, 1440],
    &[3840, 2160],
    &[7680, 4320],
    &[64, 64],
    &[128, 64],
    &[128, 128],
    &[256, 128],
    &[256, 256],
    &[512, 256],
    &[512, 512],
    &[1024, 512],
    &[1024, 1024],
    &[2048, 1024],
    &[2048, 2048],
    &[4096, 2048],
    &[4096, 4096],
    &[8192, 4096],
    &[8192, 8192],
    &[16384, 8192],
    &[16, 16, 16],
    &[32, 16, 16],
    &[32, 32, 16],
    &[32, 32, 32],
    &[64, 32, 32],
    &[64, 64, 32],
    &[64, 64, 64],
    &[128, 64, 64],
    &[128, 128, 64],
    &[128, 128, 128],
    &[256, 128, 128],
    &[256, 256, 128],
    &[256, 256, 256],
    &[512, 256, 256],
    &[512, 512, 256],
    &[512, 512, 512],
];

const SAMPLE7_DIMENSIONS: [&[usize]; 54] = [
    &[1024, 1024],
    &[17, 17],
    &[19, 19],
    &[23, 23],
    &[29, 29],
    &[31, 31],
    &[37, 37],
    &[41, 41],
    &[43, 43],
    &[47, 47],
    &[53, 53],
    &[59, 59],
    &[61, 61],
    &[67, 67],
    &[71, 71],
    &[73, 73],
    &[79, 79],
    &[83, 83],
    &[89, 89],
    &[97, 97],
    &[17, 17, 17],
    &[19, 19, 19],
    &[23, 23, 23],
    &[29, 29, 29],
    &[31, 31, 31],
    &[37, 37, 37],
    &[41, 41, 41],
    &[43, 43, 43],
    &[47, 47, 47],
    &[53, 53, 53],
    &[59, 59, 59],
    &[61, 61, 61],
    &[67, 67, 67],
    &[71, 71, 71],
    &[73, 73, 73],
    &[79, 79, 79],
    &[83, 83, 83],
    &[89, 89, 89],
    &[97, 97, 97],
    &[179, 179],
    &[283, 283],
    &[419, 419],
    &[547, 547],
    &[661, 661],
    &[811, 811],
    &[947, 947],
    &[1087, 1087],
    &[1229, 1229],
    &[1381, 1381],
    &[1523, 1523],
    &[2909, 2909],
    &[4241, 4241],
    &[6841, 6841],
    &[7727, 7727],
];
