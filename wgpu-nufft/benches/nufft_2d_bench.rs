use std::error::Error;
use std::fmt;
use std::io;
use std::time::{Duration, Instant};

use wgpu::util::DeviceExt;
use wgpu_nufft::{ModeOrder, NufftConfig, NufftPlan, NufftSign, DEFAULT_EPS, DEFAULT_SIGMA};

type BenchResult<T> = Result<T, Box<dyn Error>>;

const CASE_SHAPES: [[usize; 2]; 2] = [[512, 512], [1_024, 1_024]];
const DEFAULT_RUNS: usize = 3;
const DEFAULT_SAMPLES: usize = 10;
const TYPE2_TRANSFORMS_PER_SAMPLE: usize = 32;
const DEFAULT_WAIT_TIMEOUT: Duration = Duration::from_secs(120);
const DATA_SEED: u32 = 0x4E55_4646;
const POINT_X_SEED_MASK: u32 = 0xA341_316C;
const POINT_Y_SEED_MASK: u32 = 0xB7E1_5162;
const STRENGTH_REAL_SEED_MASK: u32 = 0xC801_3EA4;
const STRENGTH_IMAG_SEED_MASK: u32 = 0xAD90_777D;
const MODE_REAL_SEED_MASK: u32 = 0x7E95_761E;
const MODE_IMAG_SEED_MASK: u32 = 0x6C8E_9CF5;

#[derive(Debug)]
struct Options {
    adapter_selector: Option<String>,
    runs: usize,
    samples: usize,
    wait_timeout: Duration,
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

    const fn transforms_per_sample(self) -> usize {
        match self {
            Self::Type1 => 1,
            Self::Type2 => TYPE2_TRANSFORMS_PER_SAMPLE,
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

struct TransformExecution<'a> {
    device: &'a wgpu::Device,
    queue: &'a wgpu::Queue,
    plan: &'a NufftPlan,
    buffers: &'a CaseBuffers,
    point_count: usize,
    kind: TransformKind,
    wait_timeout: Duration,
}

#[derive(Debug)]
struct Statistics {
    mean: f64,
    stderr: Option<f64>,
    minimum: f64,
}

fn main() {
    if let Err(error) = pollster::block_on(run()) {
        eprintln!("wgpu-nufft 2D benchmark failed: {error}");
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
    println!("wgpu-nufft 2D GPU benchmark");
    println!(
        "configuration: shapes={CASE_SHAPES:?} point_count_equals_total_mode_count=true axis_zero_fastest=true point_layout=point-major-[x,y] runs={} samples_per_run={} seed={DATA_SEED:#010x} point_y_seed_mask={POINT_Y_SEED_MASK:#010x} eps={} sigma={} sign=positive mode_order=centered adapter_selector={} wait_timeout_secs={}",
        options.runs,
        options.samples,
        DEFAULT_EPS,
        DEFAULT_SIGMA,
        options
            .adapter_selector
            .as_deref()
            .unwrap_or("auto-single-hardware"),
        options.wait_timeout.as_secs(),
    );
    println!(
        "method: reusable plan recreated for every run; caller buffers initialized once; one untimed warmup; type-1 uses one transform per timed submit, while type-2 batches {TYPE2_TRANSFORMS_PER_SAMPLE} transforms per timed submit; wall clock starts immediately before queue.submit and ends after device.poll(Wait), then is divided per transform"
    );
    println!(
        "scope: plan creation, command encoding, upload, and readback excluded from headline; host command-encoding time and plan creation are reported separately; all transform inputs and outputs remain GPU-resident"
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

    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("wgpu_nufft.2d_bench.device"),
            required_features: wgpu::Features::empty(),
            required_limits: adapter_limits,
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
        })
        .await
        .map_err(|error| contextual_error("requesting 2D benchmark device", error))?;
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
                    )));
                }
                _ => {
                    return Err(input_error(format!(
                        "adapter selector {selector:?} is ambiguous; matched indices {matches:?}"
                    )));
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
                )));
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

async fn run_cases(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    options: &Options,
) -> BenchResult<()> {
    for shape in CASE_SHAPES {
        let mode_count = shape[0]
            .checked_mul(shape[1])
            .ok_or_else(|| input_error("2D benchmark mode-count overflow"))?;
        let config = benchmark_config(shape);
        let buffers = create_case_buffers(device, mode_count)?;
        let point_bytes = mode_count
            .checked_mul(2)
            .and_then(|values| values.checked_mul(size_of::<f32>()))
            .ok_or_else(|| input_error("2D benchmark point-buffer size overflow"))?;
        println!(
            "\n=== case dimensions=2 N0={} N1={} N_total={} M={} input_points_bytes={} input_complex_bytes={} ===",
            shape[0],
            shape[1],
            mode_count,
            mode_count,
            point_bytes,
            NufftPlan::required_type1_strength_buffer_size_bytes(mode_count)?,
        );

        for kind in TransformKind::ALL {
            run_kind(
                device, queue, &buffers, &config, shape, mode_count, kind, options,
            )
            .await?;
        }
        device
            .poll(wgpu::PollType::Poll)
            .map_err(|error| contextual_error("polling after 2D benchmark case", error))?;
    }
    Ok(())
}

async fn run_kind(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    buffers: &CaseBuffers,
    config: &NufftConfig,
    shape: [usize; 2],
    point_count: usize,
    kind: TransformKind,
    options: &Options,
) -> BenchResult<()> {
    let mut submit_wait_samples = Vec::with_capacity(options.runs * options.samples);
    let mut encoding_samples = Vec::with_capacity(options.runs * options.samples);
    let mut run_submit_wait_means = Vec::with_capacity(options.runs);
    let mut run_encoding_means = Vec::with_capacity(options.runs);
    let mut plan_creation_samples = Vec::with_capacity(options.runs);
    let mut fine_shape: Option<Vec<usize>> = None;
    let transforms_per_sample = kind.transforms_per_sample();

    println!(
        "\n{}: starting transforms_per_sample={transforms_per_sample}",
        kind.name()
    );
    for run_index in 0..options.runs {
        let plan_scopes = push_gpu_error_scopes(device);
        let plan_start = Instant::now();
        let plan = kind.create_plan(device, queue, config.clone());
        let plan_creation_ms = plan_start.elapsed().as_secs_f64() * 1_000.0;
        pop_gpu_error_scopes(plan_scopes, "creating 2D NUFFT benchmark plan").await?;
        let plan = plan?;
        plan_creation_samples.push(plan_creation_ms);
        let this_fine_shape = plan.fine_grid_shape().to_vec();
        if this_fine_shape.len() != 2 {
            return Err(input_error(format!(
                "2D plan reported unexpected fine-grid shape {this_fine_shape:?}"
            )));
        }
        if let Some(previous) = &fine_shape {
            if previous != &this_fine_shape {
                return Err(input_error(format!(
                    "fine-grid shape changed between plan recreations: {previous:?} versus {this_fine_shape:?}"
                )));
            }
        } else {
            fine_shape = Some(this_fine_shape);
        }

        let execution = TransformExecution {
            device,
            queue,
            plan: &plan,
            buffers,
            point_count,
            kind,
            wait_timeout: options.wait_timeout,
        };
        let (warmup_encode_ms, warmup_submit_wait_ms) =
            execution.execute_once("2D warmup", 1).await?;
        println!(
            "run {}/{} plan_create_ms={:.6} warmup: encode_ms={:.6} submit_wait_ms={:.6}",
            run_index + 1,
            options.runs,
            plan_creation_ms,
            warmup_encode_ms,
            warmup_submit_wait_ms,
        );

        let mut run_submit_wait = Vec::with_capacity(options.samples);
        let mut run_encoding = Vec::with_capacity(options.samples);
        for sample_index in 0..options.samples {
            let context = format!(
                "{} {}x{} run={} sample={}",
                kind.name(),
                shape[0],
                shape[1],
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
    let fine_shape = fine_shape.expect("at least one validated run");
    let million_points_per_second = point_count as f64 / (submit_wait_stats.mean * 1_000.0);
    println!(
        "RESULT kind={} dimensions=2 N0={} N1={} N_total={} M={} fine_grid_shape={}x{} eps={} sigma={} sign=positive mode_order=centered precision=f32 runs={} samples_per_run={} transforms_per_sample={} total_samples={} raw_submit_wait_ms_per_transform={:?} run_avg_submit_wait_ms_per_transform={:?} avg_submit_wait_ms_per_transform={:.6} stderr_submit_wait_ms_per_transform={} stderr_basis=plan-recreated-run-means min_submit_wait_ms_per_transform={:.6} million_points_per_second={:.6} raw_encode_ms_per_transform={:?} run_avg_encode_ms_per_transform={:?} avg_encode_ms_per_transform={:.6} stderr_encode_ms_per_transform={} min_encode_ms_per_transform={:.6} raw_plan_create_ms={:?} avg_plan_create_ms={:.6} stderr_plan_create_ms={} min_plan_create_ms={:.6} timing_scope=submit-through-device-poll-divided-per-transform setup_excluded=plan,encode,upload,readback",
        kind.name(),
        shape[0],
        shape[1],
        point_count,
        point_count,
        fine_shape[0],
        fine_shape[1],
        config.eps(),
        config.sigma(),
        options.runs,
        options.samples,
        transforms_per_sample,
        submit_wait_samples.len(),
        submit_wait_samples,
        run_submit_wait_means,
        submit_wait_stats.mean,
        format_optional(submit_wait_stats.stderr),
        raw_submit_wait_stats.minimum,
        million_points_per_second,
        encoding_samples,
        run_encoding_means,
        encoding_stats.mean,
        format_optional(encoding_stats.stderr),
        raw_encoding_stats.minimum,
        plan_creation_samples,
        plan_creation_stats.mean,
        format_optional(plan_creation_stats.stderr),
        plan_creation_stats.minimum,
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
                    label: Some("wgpu_nufft.2d_bench.encoder"),
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

fn benchmark_config(shape: [usize; 2]) -> NufftConfig {
    NufftConfig::new(shape, DEFAULT_EPS)
        .with_sigma(DEFAULT_SIGMA)
        .with_sign(NufftSign::Positive)
        .with_mode_order(ModeOrder::Centered)
}

fn create_case_buffers(device: &wgpu::Device, count: usize) -> BenchResult<CaseBuffers> {
    let (points, strengths, modes) = generate_case_data(count)?;
    let storage_usage = wgpu::BufferUsages::STORAGE;
    let points = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("wgpu_nufft.2d_bench.points"),
        contents: bytemuck::cast_slice(&points),
        usage: storage_usage,
    });
    let strengths = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("wgpu_nufft.2d_bench.strengths"),
        contents: bytemuck::cast_slice(&strengths),
        usage: storage_usage,
    });
    let modes = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("wgpu_nufft.2d_bench.modes"),
        contents: bytemuck::cast_slice(&modes),
        usage: storage_usage,
    });
    let complex_bytes = NufftPlan::required_type1_strength_buffer_size_bytes(count)?;
    let type1_output = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_nufft.2d_bench.type1_output"),
        size: complex_bytes,
        usage: storage_usage,
        mapped_at_creation: false,
    });
    let type2_output = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_nufft.2d_bench.type2_output"),
        size: complex_bytes,
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

fn generate_case_data(count: usize) -> BenchResult<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    let complex_values = count
        .checked_mul(2)
        .ok_or_else(|| input_error("2D benchmark complex value count overflow"))?;
    let mut point_x_rng = Lcg32::new(DATA_SEED ^ POINT_X_SEED_MASK);
    let mut point_y_rng = Lcg32::new(DATA_SEED ^ POINT_Y_SEED_MASK);
    let mut strength_real_rng = Lcg32::new(DATA_SEED ^ STRENGTH_REAL_SEED_MASK);
    let mut strength_imag_rng = Lcg32::new(DATA_SEED ^ STRENGTH_IMAG_SEED_MASK);
    let mut mode_real_rng = Lcg32::new(DATA_SEED ^ MODE_REAL_SEED_MASK);
    let mut mode_imag_rng = Lcg32::new(DATA_SEED ^ MODE_IMAG_SEED_MASK);

    let mut points = Vec::with_capacity(complex_values);
    for _ in 0..count {
        points.push(
            -std::f32::consts::PI + std::f32::consts::TAU * point_x_rng.next_unit_interval_f32(),
        );
        points.push(
            -std::f32::consts::PI + std::f32::consts::TAU * point_y_rng.next_unit_interval_f32(),
        );
    }
    let mut strengths = Vec::with_capacity(complex_values);
    let mut modes = Vec::with_capacity(complex_values);
    for _ in 0..count {
        strengths.push(-1.0 + 2.0 * strength_real_rng.next_unit_interval_f32());
        strengths.push(-1.0 + 2.0 * strength_imag_rng.next_unit_interval_f32());
        modes.push(-1.0 + 2.0 * mode_real_rng.next_unit_interval_f32());
        modes.push(-1.0 + 2.0 * mode_imag_rng.next_unit_interval_f32());
    }
    Ok((points, strengths, modes))
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

fn format_optional(value: Option<f64>) -> String {
    value.map_or_else(|| "undefined".to_owned(), |value| format!("{value:.6}"))
}

fn parse_options() -> BenchResult<Options> {
    let mut adapter_selector = None;
    let mut runs = DEFAULT_RUNS;
    let mut samples = DEFAULT_SAMPLES;
    let mut wait_timeout = DEFAULT_WAIT_TIMEOUT;
    let mut arguments = std::env::args().skip(1).filter(|arg| arg != "--bench");
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--adapter" => adapter_selector = Some(next_value(&mut arguments, "--adapter")?),
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
    Ok(Options {
        adapter_selector,
        runs,
        samples,
        wait_timeout,
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
  cargo bench -p wgpu-nufft --bench nufft_2d_bench -- [options]

Options:
  --adapter <index-or-name>   Select a Vulkan hardware adapter.
  --runs <count>              Plan recreations per case (default: 3).
  --samples <count>           Timed submissions per plan (default: 10).
  --wait-timeout-secs <secs>  Per-submission timeout (default: 120).
  --help                      Show this help.

Fixed cases are 512x512 and 1024x1024 with M=N0*N1, f32, eps=1e-6,
sigma=2, positive sign, centered mode order, and axis zero fastest."#
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
