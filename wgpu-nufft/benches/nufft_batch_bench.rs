use std::error::Error;
use std::fmt;
use std::io;
use std::time::{Duration, Instant};

use wgpu::util::DeviceExt;
use wgpu_nufft::{ModeOrder, NufftConfig, NufftPlan, NufftSign, DEFAULT_EPS, DEFAULT_SIGMA};

type BenchResult<T> = Result<T, Box<dyn Error>>;

const DEFAULT_RUNS: usize = 2;
const DEFAULT_SAMPLES: usize = 2;
const DEFAULT_BATCHES: [usize; 3] = [1, 4, 16];
const DEFAULT_WAIT_TIMEOUT: Duration = Duration::from_secs(120);
const DATA_SEED: u32 = 0x4E55_4646;
const POINT_SEED_MASKS: [u32; 3] = [0xA341_316C, 0xB7E1_5162, 0x9E37_79B9];
const STRENGTH_REAL_SEED_MASK: u32 = 0xC801_3EA4;
const STRENGTH_IMAG_SEED_MASK: u32 = 0xAD90_777D;
const MODE_REAL_SEED_MASK: u32 = 0x7E95_761E;
const MODE_IMAG_SEED_MASK: u32 = 0x6C8E_9CF5;

#[derive(Debug, Clone, PartialEq, Eq)]
struct Case {
    mode_shape: Vec<usize>,
}

impl Case {
    fn mode_count(&self) -> BenchResult<usize> {
        checked_product(&self.mode_shape, "mode count")
    }

    fn dimensions(&self) -> usize {
        self.mode_shape.len()
    }

    fn token(&self) -> String {
        shape_token(&self.mode_shape)
    }
}

#[derive(Debug)]
struct Options {
    adapter_selector: Option<String>,
    cases: Vec<Case>,
    batches: Vec<usize>,
    kinds: Vec<TransformKind>,
    runs: usize,
    samples: usize,
    wait_timeout: Duration,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TransformKind {
    Type1,
    Type2,
}

impl TransformKind {
    const fn name(self) -> &'static str {
        match self {
            Self::Type1 => "type-1",
            Self::Type2 => "type-2",
        }
    }

    fn create_plan(
        self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        config: NufftConfig,
    ) -> BenchResult<NufftPlan> {
        Ok(match self {
            Self::Type1 => NufftPlan::type1_gpu(device, queue, config)?,
            Self::Type2 => NufftPlan::type2_gpu(device, queue, config)?,
        })
    }
}

struct NativeBuffers {
    strengths: wgpu::Buffer,
    modes: wgpu::Buffer,
    type1_output: wgpu::Buffer,
    type2_output: wgpu::Buffer,
}

struct SingleBuffers {
    strengths: wgpu::Buffer,
    modes: wgpu::Buffer,
    type1_output: wgpu::Buffer,
    type2_output: wgpu::Buffer,
}

struct CaseBuffers {
    points: wgpu::Buffer,
    native: NativeBuffers,
    singles: Vec<SingleBuffers>,
}

#[derive(Debug)]
struct MethodStatistics {
    raw_ms: Vec<f64>,
    run_means_ms: Vec<f64>,
    mean_ms: f64,
    stderr_ms: f64,
    minimum_ms: f64,
}

#[derive(Debug)]
struct PairStatistics {
    native: MethodStatistics,
    looped: MethodStatistics,
    native_plan_ms: Vec<f64>,
    looped_plan_ms: Vec<f64>,
    fine_grid_shape: Vec<usize>,
}

fn main() {
    if let Err(error) = pollster::block_on(run()) {
        eprintln!("wgpu-nufft batch benchmark failed: {error}");
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
    println!("wgpu-nufft native batching benchmark");
    println!(
        "configuration: cases={} ntr={:?} kinds={} runs={} samples_per_run={} eps={} sigma={} seed={DATA_SEED:#010x}",
        options
            .cases
            .iter()
            .map(Case::token)
            .collect::<Vec<_>>()
            .join(","),
        options.batches,
        options
            .kinds
            .iter()
            .map(|kind| kind.name())
            .collect::<Vec<_>>()
            .join(","),
        options.runs,
        options.samples,
        DEFAULT_EPS,
        DEFAULT_SIGMA,
    );
    println!(
        "method: same point set and distinct transform-major vectors; native uses one with_batch(ntr) encode; loop uses ntr sequential encodes through one reusable batch-1 plan; one command buffer and one submit per measured batch; one untimed warmup; plans recreated per run"
    );
    println!(
        "scope: wall clock starts immediately before queue.submit and ends after device.poll(Wait); plan creation, command encoding, uploads, and readback are excluded; batch-total and per-vector times are reported"
    );

    let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
    descriptor.backends = wgpu::Backends::VULKAN;
    let instance = wgpu::Instance::new(descriptor);
    let adapter = select_vulkan_adapter(&instance, options.adapter_selector.as_deref()).await?;
    let adapter_info = adapter.get_info();
    println!("adapter info:\n{adapter_info:#?}");
    println!("backend={:?}", adapter_info.backend);
    let adapter_limits = adapter.limits();
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("wgpu_nufft.batch_bench.device"),
            required_features: wgpu::Features::empty(),
            required_limits: adapter_limits,
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
        })
        .await
        .map_err(|error| contextual_error("requesting batch benchmark device", error))?;
    println!("device limits:\n{:#?}", device.limits());

    let result = run_cases(&device, &queue, &options).await;
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
        let hardware = adapters
            .iter()
            .enumerate()
            .filter_map(|(index, (_, info))| {
                matches!(
                    info.device_type,
                    wgpu::DeviceType::DiscreteGpu | wgpu::DeviceType::IntegratedGpu
                )
                .then_some(index)
            })
            .collect::<Vec<_>>();
        match hardware.as_slice() {
            [index] => *index,
            [] => return Err(input_error("no Vulkan hardware adapter was found")),
            _ => {
                return Err(input_error(format!(
                    "multiple Vulkan hardware adapters found at {hardware:?}; use --adapter"
                )));
            }
        }
    };
    let (adapter, info) = adapters.swap_remove(selected);
    if !matches!(
        info.device_type,
        wgpu::DeviceType::DiscreteGpu | wgpu::DeviceType::IntegratedGpu
    ) {
        return Err(input_error(format!(
            "refusing non-hardware Vulkan adapter {:?} ({:?})",
            info.name, info.device_type
        )));
    }
    println!(
        "selected Vulkan adapter index={selected} name={:?} backend={:?}",
        info.name, info.backend
    );
    Ok(adapter)
}

async fn run_cases(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    options: &Options,
) -> BenchResult<()> {
    for case in &options.cases {
        let point_count = case.mode_count()?;
        println!(
            "\n=== case dimensions={} shape={} N=M={} ===",
            case.dimensions(),
            case.token(),
            point_count
        );
        for &batch in &options.batches {
            let buffers = create_case_buffers(device, case, batch)?;
            for &kind in &options.kinds {
                let stats = benchmark_pair(
                    device,
                    queue,
                    case,
                    point_count,
                    batch,
                    kind,
                    &buffers,
                    options,
                )
                .await?;
                print_result(case, point_count, batch, kind, &stats, options)?;
            }
            device
                .poll(wgpu::PollType::Poll)
                .map_err(|error| contextual_error("polling after batch case", error))?;
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
async fn benchmark_pair(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    case: &Case,
    point_count: usize,
    batch: usize,
    kind: TransformKind,
    buffers: &CaseBuffers,
    options: &Options,
) -> BenchResult<PairStatistics> {
    let mut native_raw = Vec::with_capacity(options.runs * options.samples);
    let mut looped_raw = Vec::with_capacity(options.runs * options.samples);
    let mut native_run_means = Vec::with_capacity(options.runs);
    let mut looped_run_means = Vec::with_capacity(options.runs);
    let mut native_plan_ms = Vec::with_capacity(options.runs);
    let mut looped_plan_ms = Vec::with_capacity(options.runs);
    let mut fine_grid_shape = None;

    println!("\n{} ntr={batch}: starting native-vs-loop", kind.name());
    for run in 0..options.runs {
        let native_config = benchmark_config(&case.mode_shape).with_batch(batch);
        let native_plan_start = Instant::now();
        let native_plan = kind.create_plan(device, queue, native_config)?;
        native_plan_ms.push(native_plan_start.elapsed().as_secs_f64() * 1_000.0);
        let looped_plan_start = Instant::now();
        let looped_plan = kind.create_plan(device, queue, benchmark_config(&case.mode_shape))?;
        looped_plan_ms.push(looped_plan_start.elapsed().as_secs_f64() * 1_000.0);
        let this_fine_grid_shape = native_plan.fine_grid_shape().to_vec();
        if let Some(previous) = fine_grid_shape.as_ref() {
            if previous != &this_fine_grid_shape {
                return Err(input_error(
                    "fine-grid shape changed between plan recreations",
                ));
            }
        } else {
            fine_grid_shape = Some(this_fine_grid_shape);
        }

        submit_native(
            device,
            queue,
            &native_plan,
            buffers,
            point_count,
            batch,
            kind,
            options.wait_timeout,
            "native warmup",
        )?;
        submit_looped(
            device,
            queue,
            &looped_plan,
            buffers,
            point_count,
            kind,
            options.wait_timeout,
            "loop warmup",
        )?;

        let mut native_this_run = Vec::with_capacity(options.samples);
        let mut looped_this_run = Vec::with_capacity(options.samples);
        for sample in 0..options.samples {
            let native_context = format!(
                "{} native ntr={batch} run={} sample={}",
                kind.name(),
                run + 1,
                sample + 1
            );
            let loop_context = format!(
                "{} loop ntr={batch} run={} sample={}",
                kind.name(),
                run + 1,
                sample + 1
            );
            let (native_ms, looped_ms) = if (run + sample) % 2 == 0 {
                let native_ms = submit_native(
                    device,
                    queue,
                    &native_plan,
                    buffers,
                    point_count,
                    batch,
                    kind,
                    options.wait_timeout,
                    &native_context,
                )?;
                let looped_ms = submit_looped(
                    device,
                    queue,
                    &looped_plan,
                    buffers,
                    point_count,
                    kind,
                    options.wait_timeout,
                    &loop_context,
                )?;
                (native_ms, looped_ms)
            } else {
                let looped_ms = submit_looped(
                    device,
                    queue,
                    &looped_plan,
                    buffers,
                    point_count,
                    kind,
                    options.wait_timeout,
                    &loop_context,
                )?;
                let native_ms = submit_native(
                    device,
                    queue,
                    &native_plan,
                    buffers,
                    point_count,
                    batch,
                    kind,
                    options.wait_timeout,
                    &native_context,
                )?;
                (native_ms, looped_ms)
            };
            println!(
                "run {}/{} sample {}/{} native_total_ms={native_ms:.6} native_ms_per_vector={:.6} loop_total_ms={looped_ms:.6} loop_ms_per_vector={:.6} speedup={:.4}",
                run + 1,
                options.runs,
                sample + 1,
                options.samples,
                native_ms / batch as f64,
                looped_ms / batch as f64,
                looped_ms / native_ms,
            );
            native_raw.push(native_ms);
            looped_raw.push(looped_ms);
            native_this_run.push(native_ms);
            looped_this_run.push(looped_ms);
        }
        native_run_means.push(mean(&native_this_run)?);
        looped_run_means.push(mean(&looped_this_run)?);
    }

    Ok(PairStatistics {
        native: method_statistics(native_raw, native_run_means)?,
        looped: method_statistics(looped_raw, looped_run_means)?,
        native_plan_ms,
        looped_plan_ms,
        fine_grid_shape: fine_grid_shape.expect("validated runs are nonempty"),
    })
}

#[allow(clippy::too_many_arguments)]
fn submit_native(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    plan: &NufftPlan,
    buffers: &CaseBuffers,
    point_count: usize,
    batch: usize,
    kind: TransformKind,
    timeout: Duration,
    context: &str,
) -> BenchResult<f64> {
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("wgpu_nufft.batch_bench.native_encoder"),
    });
    match kind {
        TransformKind::Type1 => plan.encode_type1_gpu_batch(
            device,
            &mut encoder,
            point_count,
            batch,
            &buffers.points,
            &buffers.native.strengths,
            &buffers.native.type1_output,
        )?,
        TransformKind::Type2 => plan.encode_type2_gpu_batch(
            device,
            &mut encoder,
            point_count,
            batch,
            &buffers.points,
            &buffers.native.modes,
            &buffers.native.type2_output,
        )?,
    }
    submit_and_wait(device, queue, encoder.finish(), timeout, context)
}

#[allow(clippy::too_many_arguments)]
fn submit_looped(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    plan: &NufftPlan,
    buffers: &CaseBuffers,
    point_count: usize,
    kind: TransformKind,
    timeout: Duration,
    context: &str,
) -> BenchResult<f64> {
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("wgpu_nufft.batch_bench.loop_encoder"),
    });
    for single in &buffers.singles {
        match kind {
            TransformKind::Type1 => plan.encode_type1_gpu(
                device,
                &mut encoder,
                point_count,
                &buffers.points,
                &single.strengths,
                &single.type1_output,
            )?,
            TransformKind::Type2 => plan.encode_type2_gpu(
                device,
                &mut encoder,
                point_count,
                &buffers.points,
                &single.modes,
                &single.type2_output,
            )?,
        }
    }
    submit_and_wait(device, queue, encoder.finish(), timeout, context)
}

fn submit_and_wait(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    command_buffer: wgpu::CommandBuffer,
    timeout: Duration,
    context: &str,
) -> BenchResult<f64> {
    let start = Instant::now();
    let submission = queue.submit([command_buffer]);
    device
        .poll(wgpu::PollType::Wait {
            submission_index: Some(submission),
            timeout: Some(timeout),
        })
        .map_err(|error| contextual_error(format!("waiting for {context}"), error))?;
    let elapsed_ms = start.elapsed().as_secs_f64() * 1_000.0;
    if !elapsed_ms.is_finite() || elapsed_ms <= 0.0 {
        return Err(input_error(format!(
            "invalid submit-to-wait duration for {context}: {elapsed_ms} ms"
        )));
    }
    Ok(elapsed_ms)
}

fn print_result(
    case: &Case,
    point_count: usize,
    batch: usize,
    kind: TransformKind,
    stats: &PairStatistics,
    options: &Options,
) -> BenchResult<()> {
    let speedup = stats.looped.mean_ms / stats.native.mean_ms;
    let fine_points = checked_product(&stats.fine_grid_shape, "fine-grid point count")?;
    println!(
        "RESULT kind={} dimensions={} mode_shape={} N_total={} M={} ntr={} eps={} sigma={} runs={} samples_per_run={} fine_grid_shape={} fine_grid_points={} native_raw_total_ms={:?} native_run_means_total_ms={:?} native_avg_total_ms={:.6} native_stderr_total_ms={:.6} native_min_total_ms={:.6} native_avg_ms_per_vector={:.6} loop_raw_total_ms={:?} loop_run_means_total_ms={:?} loop_avg_total_ms={:.6} loop_stderr_total_ms={:.6} loop_min_total_ms={:.6} loop_avg_ms_per_vector={:.6} native_vs_loop_speedup={:.6} native_plan_ms={:?} loop_plan_ms={:?} timing_scope=single-submit-through-device-poll setup_excluded=plan,encode,upload,readback layout=transform-major shared_points=true precision=f32",
        kind.name(),
        case.dimensions(),
        case.token(),
        point_count,
        point_count,
        batch,
        DEFAULT_EPS,
        DEFAULT_SIGMA,
        options.runs,
        options.samples,
        shape_token(&stats.fine_grid_shape),
        fine_points,
        stats.native.raw_ms,
        stats.native.run_means_ms,
        stats.native.mean_ms,
        stats.native.stderr_ms,
        stats.native.minimum_ms,
        stats.native.mean_ms / batch as f64,
        stats.looped.raw_ms,
        stats.looped.run_means_ms,
        stats.looped.mean_ms,
        stats.looped.stderr_ms,
        stats.looped.minimum_ms,
        stats.looped.mean_ms / batch as f64,
        speedup,
        stats.native_plan_ms,
        stats.looped_plan_ms,
    );
    Ok(())
}

fn benchmark_config(shape: &[usize]) -> NufftConfig {
    NufftConfig::new(shape.to_vec(), DEFAULT_EPS)
        .with_sigma(DEFAULT_SIGMA)
        .with_sign(NufftSign::Positive)
        .with_mode_order(ModeOrder::Centered)
}

fn create_case_buffers(
    device: &wgpu::Device,
    case: &Case,
    batch: usize,
) -> BenchResult<CaseBuffers> {
    let point_count = case.mode_count()?;
    let (points, base_strengths, base_modes) = generate_case_data(case, point_count)?;
    let mut all_strengths = Vec::with_capacity(base_strengths.len() * batch);
    let mut all_modes = Vec::with_capacity(base_modes.len() * batch);
    let mut single_host = Vec::with_capacity(batch);
    for vector in 0..batch {
        let strengths = transform_vector(&base_strengths, vector);
        let modes = transform_vector(&base_modes, vector);
        all_strengths.extend_from_slice(&strengths);
        all_modes.extend_from_slice(&modes);
        single_host.push((strengths, modes));
    }
    let usage = wgpu::BufferUsages::STORAGE;
    let points = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("wgpu_nufft.batch_bench.points"),
        contents: bytemuck::cast_slice(&points),
        usage,
    });
    let native = NativeBuffers {
        strengths: create_storage_buffer_init(
            device,
            "wgpu_nufft.batch_bench.native_strengths",
            &all_strengths,
        ),
        modes: create_storage_buffer_init(
            device,
            "wgpu_nufft.batch_bench.native_modes",
            &all_modes,
        ),
        type1_output: create_storage_buffer(
            device,
            "wgpu_nufft.batch_bench.native_type1_output",
            checked_complex_bytes(point_count, batch, "native type-1 output")?,
        ),
        type2_output: create_storage_buffer(
            device,
            "wgpu_nufft.batch_bench.native_type2_output",
            checked_complex_bytes(point_count, batch, "native type-2 output")?,
        ),
    };
    let mut singles = Vec::with_capacity(batch);
    for (strengths, modes) in single_host {
        singles.push(SingleBuffers {
            strengths: create_storage_buffer_init(
                device,
                "wgpu_nufft.batch_bench.single_strengths",
                &strengths,
            ),
            modes: create_storage_buffer_init(
                device,
                "wgpu_nufft.batch_bench.single_modes",
                &modes,
            ),
            type1_output: create_storage_buffer(
                device,
                "wgpu_nufft.batch_bench.single_type1_output",
                checked_complex_bytes(point_count, 1, "single type-1 output")?,
            ),
            type2_output: create_storage_buffer(
                device,
                "wgpu_nufft.batch_bench.single_type2_output",
                checked_complex_bytes(point_count, 1, "single type-2 output")?,
            ),
        });
    }
    Ok(CaseBuffers {
        points,
        native,
        singles,
    })
}

fn create_storage_buffer_init(
    device: &wgpu::Device,
    label: &'static str,
    values: &[f32],
) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(values),
        usage: wgpu::BufferUsages::STORAGE,
    })
}

fn create_storage_buffer(device: &wgpu::Device, label: &'static str, size: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    })
}

fn generate_case_data(
    case: &Case,
    point_count: usize,
) -> BenchResult<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    let coordinate_count = point_count
        .checked_mul(case.dimensions())
        .ok_or_else(|| input_error("point-coordinate count overflow"))?;
    let mut point_rngs = POINT_SEED_MASKS[..case.dimensions()]
        .iter()
        .map(|mask| Lcg32::new(DATA_SEED ^ mask))
        .collect::<Vec<_>>();
    let mut points = Vec::with_capacity(coordinate_count);
    for _ in 0..point_count {
        for rng in &mut point_rngs {
            points
                .push(rng.next_unit_interval_f32() * std::f32::consts::TAU - std::f32::consts::PI);
        }
    }
    let strengths = generate_complex_values(
        point_count,
        DATA_SEED ^ STRENGTH_REAL_SEED_MASK,
        DATA_SEED ^ STRENGTH_IMAG_SEED_MASK,
    );
    let modes = generate_complex_values(
        case.mode_count()?,
        DATA_SEED ^ MODE_REAL_SEED_MASK,
        DATA_SEED ^ MODE_IMAG_SEED_MASK,
    );
    Ok((points, strengths, modes))
}

fn generate_complex_values(count: usize, real_seed: u32, imag_seed: u32) -> Vec<f32> {
    let mut real = Lcg32::new(real_seed);
    let mut imag = Lcg32::new(imag_seed);
    let mut values = Vec::with_capacity(count * 2);
    for _ in 0..count {
        values.push(-1.0 + 2.0 * real.next_unit_interval_f32());
        values.push(-1.0 + 2.0 * imag.next_unit_interval_f32());
    }
    values
}

fn transform_vector(base: &[f32], vector: usize) -> Vec<f32> {
    let factor_re = 1.0 + vector as f32 * (1.0 / 32.0);
    let factor_im = -(vector as f32) * (1.0 / 64.0);
    let mut values = Vec::with_capacity(base.len());
    for pair in base.chunks_exact(2) {
        values.push(pair[0] * factor_re - pair[1] * factor_im);
        values.push(pair[0] * factor_im + pair[1] * factor_re);
    }
    values
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

fn method_statistics(raw_ms: Vec<f64>, run_means_ms: Vec<f64>) -> BenchResult<MethodStatistics> {
    Ok(MethodStatistics {
        mean_ms: mean(&run_means_ms)?,
        stderr_ms: stderr(&run_means_ms)?,
        minimum_ms: raw_ms.iter().copied().fold(f64::INFINITY, f64::min),
        raw_ms,
        run_means_ms,
    })
}

fn mean(values: &[f64]) -> BenchResult<f64> {
    if values.is_empty() || values.iter().any(|value| !value.is_finite()) {
        return Err(input_error("statistics require finite, nonempty samples"));
    }
    Ok(values.iter().sum::<f64>() / values.len() as f64)
}

fn stderr(values: &[f64]) -> BenchResult<f64> {
    if values.len() < 2 {
        return Ok(0.0);
    }
    let average = mean(values)?;
    let variance = values
        .iter()
        .map(|value| (value - average).powi(2))
        .sum::<f64>()
        / (values.len() - 1) as f64;
    Ok(variance.sqrt() / (values.len() as f64).sqrt())
}

fn parse_options() -> BenchResult<Options> {
    let mut adapter_selector = None;
    let mut cases = Vec::new();
    let mut batches = Vec::new();
    let mut kinds = Vec::new();
    let mut runs = DEFAULT_RUNS;
    let mut samples = DEFAULT_SAMPLES;
    let mut wait_timeout = DEFAULT_WAIT_TIMEOUT;
    let mut arguments = std::env::args().skip(1).filter(|arg| arg != "--bench");
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--1d" => cases.push(Case {
                mode_shape: vec![262_144],
            }),
            "--2d" => cases.push(Case {
                mode_shape: vec![512, 512],
            }),
            "--3d" => cases.push(Case {
                mode_shape: vec![64, 64, 64],
            }),
            "--all" => {
                cases.push(Case {
                    mode_shape: vec![262_144],
                });
                cases.push(Case {
                    mode_shape: vec![512, 512],
                });
                cases.push(Case {
                    mode_shape: vec![64, 64, 64],
                });
            }
            "--case" => cases.push(parse_case(&next_value(&mut arguments, "--case")?)?),
            "--ntr" => batches.push(parse_positive_usize(
                &next_value(&mut arguments, "--ntr")?,
                "ntr",
            )?),
            "--type1" => kinds.push(TransformKind::Type1),
            "--type2" => kinds.push(TransformKind::Type2),
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
    if cases.is_empty() {
        cases = vec![
            Case {
                mode_shape: vec![262_144],
            },
            Case {
                mode_shape: vec![512, 512],
            },
            Case {
                mode_shape: vec![64, 64, 64],
            },
        ];
    }
    if batches.is_empty() {
        batches.extend(DEFAULT_BATCHES);
    }
    cases.dedup();
    batches.sort_unstable();
    batches.dedup();
    if kinds.is_empty() {
        kinds = vec![TransformKind::Type1, TransformKind::Type2];
    }
    kinds.dedup();
    Ok(Options {
        adapter_selector,
        cases,
        batches,
        kinds,
        runs,
        samples,
        wait_timeout,
    })
}

fn parse_case(value: &str) -> BenchResult<Case> {
    let mode_shape = value
        .split(['x', 'X'])
        .map(|field| parse_positive_usize(field, "case dimension"))
        .collect::<BenchResult<Vec<_>>>()?;
    if !(1..=3).contains(&mode_shape.len()) {
        return Err(input_error("--case requires one to three dimensions"));
    }
    Ok(Case { mode_shape })
}

fn print_usage() {
    println!(
        r#"Usage:
  cargo bench -p wgpu-nufft --bench nufft_batch_bench -- [options]

Options:
  --1d | --2d | --3d       Select default 262144, 512x512, or 64x64x64 case.
  --all                     Select all three default cases (also the default).
  --case <N[xN[xN]]>        Add a custom 1D, 2D, or 3D N=M case.
  --ntr <count>             Add a transform count (repeatable; default 1,4,16).
  --type1 | --type2         Select transform kind (default both).
  --adapter <index-or-name> Select a Vulkan hardware adapter.
  --runs <count>            Plan recreations per case (default 2).
  --samples <count>         Timed submissions per plan (default 2).
  --wait-timeout-secs <n>   Submission timeout (default 120).
  --help                    Show this help.

Inputs and outputs are f32 complex, transform-major, and GPU-resident. Plan
creation, encoding, upload, and readback are outside the timed submit-to-wait
span. Each native-batch sample and loop-of-single sample uses one submission."#
    );
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

fn checked_product(values: &[usize], context: &'static str) -> BenchResult<usize> {
    values.iter().try_fold(1usize, |product, &value| {
        product
            .checked_mul(value)
            .ok_or_else(|| input_error(format!("{context} overflow")))
    })
}

fn checked_complex_bytes(elements: usize, batch: usize, context: &'static str) -> BenchResult<u64> {
    elements
        .checked_mul(batch)
        .and_then(|count| count.checked_mul(8))
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or_else(|| input_error(format!("{context} size overflow")))
}

fn shape_token(shape: &[usize]) -> String {
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
