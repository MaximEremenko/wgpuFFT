use std::error::Error;
use std::fmt;
use std::io;
use std::sync::mpsc;
use std::time::{Duration, Instant};

use wgpu::util::DeviceExt;
use wgpu_nufft::{
    ModeOrder, NufftConfig, NufftGpuProfileLayout, NufftGpuStage, NufftPlan, NufftSign,
    DEFAULT_EPS, DEFAULT_SIGMA,
};

type BenchResult<T> = Result<T, Box<dyn Error>>;

const CASE_SIZES: [usize; 2] = [262_144, 1_048_576];
const DEFAULT_RUNS: usize = 3;
const DEFAULT_SAMPLES: usize = 10;
const TYPE2_TRANSFORMS_PER_SAMPLE: usize = 32;
const WAIT_TIMEOUT: Duration = Duration::from_secs(120);
const MAX_QUERY_COUNT: u32 = 4_096;
const QUERY_SIZE_BYTES: u64 = size_of::<u64>() as u64;

const DATA_SEED: u32 = 0x4E55_4646;
const POINT_SEED_MASK: u32 = 0xA341_316C;
const STRENGTH_REAL_SEED_MASK: u32 = 0xC801_3EA4;
const STRENGTH_IMAG_SEED_MASK: u32 = 0xAD90_777D;
const MODE_REAL_SEED_MASK: u32 = 0x7E95_761E;
const MODE_IMAG_SEED_MASK: u32 = 0x6C8E_9CF5;

#[derive(Debug)]
struct Options {
    adapter_selector: Option<String>,
    runs: usize,
    samples: usize,
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

    fn encode_profiled(
        self,
        plan: &NufftPlan,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        point_count: usize,
        buffers: &CaseBuffers,
        query_set: &wgpu::QuerySet,
        first_query: u32,
    ) -> BenchResult<NufftGpuProfileLayout> {
        Ok(match self {
            Self::Type1 => plan.encode_type1_gpu_profiled(
                device,
                encoder,
                point_count,
                &buffers.points,
                &buffers.strengths,
                &buffers.type1_output,
                query_set,
                first_query,
            )?,
            Self::Type2 => plan.encode_type2_gpu_profiled(
                device,
                encoder,
                point_count,
                &buffers.points,
                &buffers.modes,
                &buffers.type2_output,
                query_set,
                first_query,
            )?,
        })
    }
}

struct CaseBuffers {
    points: wgpu::Buffer,
    strengths: wgpu::Buffer,
    modes: wgpu::Buffer,
    type1_output: wgpu::Buffer,
    type2_output: wgpu::Buffer,
}

struct ProfileResources {
    query_set: wgpu::QuerySet,
    resolve: wgpu::Buffer,
    readback: wgpu::Buffer,
    query_count: u32,
    query_bytes: u64,
}

#[derive(Debug)]
struct ProfileSample {
    stages: Vec<(NufftGpuStage, f64)>,
    envelope_ms: f64,
    host: HostTiming,
}

#[derive(Debug, Clone, Copy)]
struct HostTiming {
    submit_call_ms: f64,
    poll_wait_ms: f64,
    submit_wait_total_ms: f64,
}

#[derive(Debug)]
struct StageSeries {
    stage: NufftGpuStage,
    raw_sample_means_ms: Vec<f64>,
    run_means_ms: Vec<f64>,
}

#[derive(Debug, Default)]
struct TimingSeries {
    raw_sample_means_ms: Vec<f64>,
    run_means_ms: Vec<f64>,
}

#[derive(Debug)]
struct Statistics {
    mean: f64,
    stderr: Option<f64>,
    minimum: f64,
}

fn main() {
    if let Err(error) = pollster::block_on(run()) {
        eprintln!("wgpu-nufft stage profiler failed: {error}");
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
    println!("wgpu-nufft 1D GPU per-stage timestamp profile");
    println!(
        "configuration: sizes={CASE_SIZES:?} N_equals_M=true runs={} samples_per_run={} warmups_per_run=1 type1_transforms_per_sample=1 type2_transforms_per_sample={} seed={DATA_SEED:#010x} eps={} sigma={} sign=positive mode_order=centered adapter_selector={}",
        options.runs,
        options.samples,
        TYPE2_TRANSFORMS_PER_SAMPLE,
        DEFAULT_EPS,
        DEFAULT_SIGMA,
        options
            .adapter_selector
            .as_deref()
            .unwrap_or("auto-single-hardware"),
    );
    println!(
        "method: GPU timestamp-query stage intervals only; query resolve, copy, map, command encoding, plan creation, upload, and readback are outside every decoded interval; this diagnostic harness reports no submit-to-wait headline timing"
    );
    println!(
        "profiled host spans: command-buffer finish is excluded; queue.submit call and the subsequent exact-submission device.poll call are timed separately per transform, and submit-plus-poll-wait is their sum; map registration is between and excluded; these spans include the profiling resolve/copy workload and are not the uninstrumented headline"
    );
    println!(
        "warmup: one excluded submission per recreated plan uses the exact profiled transform count, timestamp writes, resolve, copy, map, and decode path"
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
    let adapter_features = adapter.features();
    println!("adapter info:\n{adapter_info:#?}");
    println!("adapter features: {adapter_features:?}");
    if !adapter_features.contains(wgpu::Features::TIMESTAMP_QUERY) {
        return Err(input_error(format!(
            "adapter {:?} does not support TIMESTAMP_QUERY",
            adapter_info.name
        )));
    }
    let adapter_limits = adapter.limits();
    println!("adapter limits:\n{adapter_limits:#?}");

    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("wgpu_nufft.stage_profile.device"),
            required_features: wgpu::Features::TIMESTAMP_QUERY,
            required_limits: adapter_limits,
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
        })
        .await
        .map_err(|error| contextual_error("requesting timestamp-query device", error))?;
    let timestamp_period_ns = queue.get_timestamp_period();
    if !timestamp_period_ns.is_finite() || timestamp_period_ns <= 0.0 {
        return Err(input_error(format!(
            "invalid queue timestamp period {timestamp_period_ns:?} ns"
        )));
    }
    println!("device features: {:?}", device.features());
    println!("device limits:\n{:#?}", device.limits());
    println!("timestamp_period_ns={timestamp_period_ns:.9}");

    let result = run_cases(&device, &queue, timestamp_period_ns, &options).await;

    // Large native wgpu runs may stall while tearing down on Windows. The
    // process exits immediately after this function returns.
    #[cfg(windows)]
    std::mem::forget((queue, device, adapter, instance));
    #[cfg(not(windows))]
    drop((queue, device, adapter, instance));
    result
}

async fn run_cases(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    timestamp_period_ns: f32,
    options: &Options,
) -> BenchResult<()> {
    for &size in &CASE_SIZES {
        let config = benchmark_config(size);
        let buffers = create_case_buffers(device, size)?;
        println!("\n=== case N={size} M={size} ===");
        for kind in TransformKind::ALL {
            run_kind(
                device,
                queue,
                &buffers,
                &config,
                size,
                kind,
                timestamp_period_ns,
                options,
            )
            .await?;
        }
        device
            .poll(wgpu::PollType::Poll)
            .map_err(|error| contextual_error("polling after profile case", error))?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
async fn run_kind(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    buffers: &CaseBuffers,
    config: &NufftConfig,
    point_count: usize,
    kind: TransformKind,
    timestamp_period_ns: f32,
    options: &Options,
) -> BenchResult<()> {
    let transforms = kind.transforms_per_sample();
    println!(
        "\n{}: starting transforms_per_sample={transforms}",
        kind.name()
    );

    let mut stage_series: Vec<StageSeries> = Vec::new();
    let mut envelope_raw_ms = Vec::with_capacity(options.runs * options.samples);
    let mut envelope_run_means_ms = Vec::with_capacity(options.runs);
    let mut submit_call = TimingSeries::default();
    let mut poll_wait = TimingSeries::default();
    let mut submit_wait_total = TimingSeries::default();
    let mut expected_fft_diagnostics = None;

    for run_index in 0..options.runs {
        let plan_start = Instant::now();
        let plan = kind.create_plan(device, queue, config.clone())?;
        let plan_create_ms = plan_start.elapsed().as_secs_f64() * 1_000.0;
        let fine_length = plan.fine_grid_shape()[0];
        let diagnostics = plan.gpu_fft_diagnostics().ok_or_else(|| {
            input_error(format!(
                "{} plan did not expose embedded FFT diagnostics",
                kind.name()
            ))
        })?;
        if let Some(expected) = expected_fft_diagnostics.as_ref() {
            if expected != &diagnostics {
                return Err(input_error(format!(
                    "{} FFT diagnostics changed between plan recreations",
                    kind.name()
                )));
            }
        } else {
            print_fft_diagnostics(kind, point_count, fine_length, &diagnostics);
            expected_fft_diagnostics = Some(diagnostics.clone());
        }

        let queries_per_transform = plan.gpu_profile_query_count();
        let profile_resources =
            create_profile_resources(device, kind, queries_per_transform, transforms)?;
        let _excluded_warmup = execute_profile_sample(
            device,
            queue,
            &plan,
            buffers,
            point_count,
            kind,
            transforms,
            queries_per_transform,
            &profile_resources,
            timestamp_period_ns,
        )
        .await?;
        println!(
            "run {}/{} plan_create_ms={plan_create_ms:.6} fine_grid_length={fine_length} queries_per_transform={queries_per_transform} total_queries={}",
            run_index + 1,
            options.runs,
            profile_resources.query_count,
        );

        let mut run_stage_samples: Vec<Vec<f64>> = Vec::new();
        let mut run_envelope_samples = Vec::with_capacity(options.samples);
        let mut run_submit_call_samples = Vec::with_capacity(options.samples);
        let mut run_poll_wait_samples = Vec::with_capacity(options.samples);
        let mut run_submit_wait_total_samples = Vec::with_capacity(options.samples);
        for sample_index in 0..options.samples {
            let sample = execute_profile_sample(
                device,
                queue,
                &plan,
                buffers,
                point_count,
                kind,
                transforms,
                queries_per_transform,
                &profile_resources,
                timestamp_period_ns,
            )
            .await?;

            if stage_series.is_empty() {
                stage_series = sample
                    .stages
                    .iter()
                    .map(|&(stage, _)| StageSeries {
                        stage,
                        raw_sample_means_ms: Vec::with_capacity(options.runs * options.samples),
                        run_means_ms: Vec::with_capacity(options.runs),
                    })
                    .collect();
            }
            validate_stage_order(&stage_series, &sample.stages, kind)?;
            if run_stage_samples.is_empty() {
                run_stage_samples = (0..sample.stages.len())
                    .map(|_| Vec::with_capacity(options.samples))
                    .collect();
            }

            for (stage_index, &(_, milliseconds)) in sample.stages.iter().enumerate() {
                stage_series[stage_index]
                    .raw_sample_means_ms
                    .push(milliseconds);
                run_stage_samples[stage_index].push(milliseconds);
            }
            envelope_raw_ms.push(sample.envelope_ms);
            run_envelope_samples.push(sample.envelope_ms);
            submit_call
                .raw_sample_means_ms
                .push(sample.host.submit_call_ms);
            poll_wait.raw_sample_means_ms.push(sample.host.poll_wait_ms);
            submit_wait_total
                .raw_sample_means_ms
                .push(sample.host.submit_wait_total_ms);
            run_submit_call_samples.push(sample.host.submit_call_ms);
            run_poll_wait_samples.push(sample.host.poll_wait_ms);
            run_submit_wait_total_samples.push(sample.host.submit_wait_total_ms);
            println!(
                "PROFILE_SAMPLE kind={} N={} M={} run={} sample={} transforms={} envelope_ms={:.9} host_submit_call_ms_per_transform={:.9} host_poll_wait_ms_per_transform={:.9} host_submit_wait_total_ms_per_transform={:.9} stages={}",
                kind.name(),
                point_count,
                point_count,
                run_index + 1,
                sample_index + 1,
                transforms,
                sample.envelope_ms,
                sample.host.submit_call_ms,
                sample.host.poll_wait_ms,
                sample.host.submit_wait_total_ms,
                format_stage_values(&sample.stages),
            );
        }

        let envelope_run_mean = statistics(&run_envelope_samples)?.mean;
        envelope_run_means_ms.push(envelope_run_mean);
        let submit_call_run_mean = statistics(&run_submit_call_samples)?.mean;
        let poll_wait_run_mean = statistics(&run_poll_wait_samples)?.mean;
        let submit_wait_total_run_mean = statistics(&run_submit_wait_total_samples)?.mean;
        submit_call.run_means_ms.push(submit_call_run_mean);
        poll_wait.run_means_ms.push(poll_wait_run_mean);
        submit_wait_total
            .run_means_ms
            .push(submit_wait_total_run_mean);
        let mut run_stage_summary = Vec::with_capacity(stage_series.len());
        for (series, samples) in stage_series.iter_mut().zip(&run_stage_samples) {
            let mean = statistics(samples)?.mean;
            series.run_means_ms.push(mean);
            run_stage_summary.push((series.stage, mean));
        }
        println!(
            "PROFILE_RUN kind={} N={} M={} run={} envelope_mean_ms={:.9} host_submit_call_mean_ms_per_transform={:.9} host_poll_wait_mean_ms_per_transform={:.9} host_submit_wait_total_mean_ms_per_transform={:.9} stages={}",
            kind.name(),
            point_count,
            point_count,
            run_index + 1,
            envelope_run_mean,
            submit_call_run_mean,
            poll_wait_run_mean,
            submit_wait_total_run_mean,
            format_stage_values(&run_stage_summary),
        );
    }

    print_final_results(
        kind,
        point_count,
        options,
        transforms,
        timestamp_period_ns,
        &stage_series,
        &envelope_raw_ms,
        &envelope_run_means_ms,
        &submit_call,
        &poll_wait,
        &submit_wait_total,
    )?;
    Ok(())
}

fn create_profile_resources(
    device: &wgpu::Device,
    kind: TransformKind,
    queries_per_transform: u32,
    transforms: usize,
) -> BenchResult<ProfileResources> {
    let transforms_u32 = u32::try_from(transforms)
        .map_err(|_| input_error("profile transform count exceeds u32"))?;
    let query_count = queries_per_transform
        .checked_mul(transforms_u32)
        .ok_or_else(|| input_error("profile query count overflow"))?;
    if query_count == 0 || query_count > MAX_QUERY_COUNT {
        return Err(input_error(format!(
            "{} profile requires {query_count} timestamp queries; allowed range is 1..={MAX_QUERY_COUNT}",
            kind.name()
        )));
    }
    let query_bytes = u64::from(query_count)
        .checked_mul(QUERY_SIZE_BYTES)
        .ok_or_else(|| input_error("profile query buffer size overflow"))?;
    let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
        label: Some("wgpu_nufft.stage_profile.query_set"),
        ty: wgpu::QueryType::Timestamp,
        count: query_count,
    });
    let resolve = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_nufft.stage_profile.resolve"),
        size: query_bytes,
        usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_nufft.stage_profile.readback"),
        size: query_bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    Ok(ProfileResources {
        query_set,
        resolve,
        readback,
        query_count,
        query_bytes,
    })
}

#[allow(clippy::too_many_arguments)]
async fn execute_profile_sample(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    plan: &NufftPlan,
    buffers: &CaseBuffers,
    point_count: usize,
    kind: TransformKind,
    transforms: usize,
    queries_per_transform: u32,
    resources: &ProfileResources,
    timestamp_period_ns: f32,
) -> BenchResult<ProfileSample> {
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("wgpu_nufft.stage_profile.encoder"),
    });
    let mut layouts = Vec::with_capacity(transforms);
    for transform_index in 0..transforms {
        let transform_index = u32::try_from(transform_index)
            .map_err(|_| input_error("profile transform index exceeds u32"))?;
        let first_query = transform_index
            .checked_mul(queries_per_transform)
            .ok_or_else(|| input_error("profile first-query index overflow"))?;
        let layout = kind.encode_profiled(
            plan,
            device,
            &mut encoder,
            point_count,
            buffers,
            &resources.query_set,
            first_query,
        )?;
        let expected_end = first_query
            .checked_add(queries_per_transform)
            .ok_or_else(|| input_error("profile query-range overflow"))?;
        if layout.query_range() != (first_query..expected_end) {
            return Err(input_error(format!(
                "{} returned unexpected query range {:?}, expected {:?}",
                kind.name(),
                layout.query_range(),
                first_query..expected_end
            )));
        }
        layouts.push(layout);
    }
    encoder.resolve_query_set(
        &resources.query_set,
        0..resources.query_count,
        &resources.resolve,
        0,
    );
    encoder.copy_buffer_to_buffer(
        &resources.resolve,
        0,
        &resources.readback,
        0,
        resources.query_bytes,
    );

    let command_buffer = encoder.finish();
    let submit_call_start = Instant::now();
    let submission = queue.submit([command_buffer]);
    let submit_call_ms = submit_call_start.elapsed().as_secs_f64() * 1_000.0 / transforms as f64;
    let slice = resources.readback.slice(0..resources.query_bytes);
    let (sender, receiver) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    let poll_wait_start = Instant::now();
    wait_for_submission(device, submission, "waiting for profiled submission")?;
    let poll_wait_ms = poll_wait_start.elapsed().as_secs_f64() * 1_000.0 / transforms as f64;
    let submit_wait_total_ms = submit_call_ms + poll_wait_ms;
    let host = HostTiming {
        submit_call_ms: validate_host_duration("queue.submit call", submit_call_ms)?,
        poll_wait_ms: validate_host_duration("device.poll wait", poll_wait_ms)?,
        submit_wait_total_ms: validate_host_duration(
            "queue.submit plus device.poll wait",
            submit_wait_total_ms,
        )?,
    };
    receiver
        .recv()
        .map_err(|error| contextual_error("receiving profile map result", error))?
        .map_err(|error| contextual_error("mapping profile query readback", error))?;
    let mapped = slice.get_mapped_range();
    let timestamps = bytemuck::cast_slice::<u8, u64>(&mapped).to_vec();
    drop(mapped);
    resources.readback.unmap();
    if timestamps.len() != resources.query_count as usize {
        return Err(input_error(format!(
            "query readback contained {} timestamps, expected {}",
            timestamps.len(),
            resources.query_count
        )));
    }
    decode_profile_samples(&layouts, &timestamps, timestamp_period_ns, host)
}

fn decode_profile_samples(
    layouts: &[NufftGpuProfileLayout],
    timestamps: &[u64],
    timestamp_period_ns: f32,
    host: HostTiming,
) -> BenchResult<ProfileSample> {
    if layouts.is_empty() {
        return Err(input_error("profile decode requires at least one layout"));
    }
    let expected_stages = layouts[0]
        .stages()
        .iter()
        .map(|query| query.stage())
        .collect::<Vec<_>>();
    if expected_stages.is_empty() {
        return Err(input_error("profile layout has no measured stages"));
    }
    let mut stage_sums_ms = vec![0.0; expected_stages.len()];
    let mut envelope_sum_ms = 0.0;
    for layout in layouts {
        let actual_stages = layout
            .stages()
            .iter()
            .map(|query| query.stage())
            .collect::<Vec<_>>();
        if actual_stages != expected_stages {
            return Err(input_error(
                "profile stage order changed between transforms",
            ));
        }
        for (stage_index, query) in layout.stages().iter().enumerate() {
            stage_sums_ms[stage_index] += timestamp_delta_ms(
                timestamps,
                query.start_query(),
                query.end_query(),
                timestamp_period_ns,
                query.stage().label(),
            )?;
        }
        let first = layout.stages().first().expect("validated nonempty stages");
        let last = layout.stages().last().expect("validated nonempty stages");
        envelope_sum_ms += timestamp_delta_ms(
            timestamps,
            first.start_query(),
            last.end_query(),
            timestamp_period_ns,
            "pipeline-envelope",
        )?;
    }
    let divisor = layouts.len() as f64;
    let stages = expected_stages
        .into_iter()
        .zip(stage_sums_ms.into_iter().map(|value| value / divisor))
        .collect();
    Ok(ProfileSample {
        stages,
        envelope_ms: envelope_sum_ms / divisor,
        host,
    })
}

fn validate_host_duration(label: &str, milliseconds: f64) -> BenchResult<f64> {
    if !milliseconds.is_finite() || milliseconds < 0.0 {
        return Err(input_error(format!(
            "invalid profiled host {label} duration {milliseconds} ms"
        )));
    }
    Ok(milliseconds)
}

fn timestamp_delta_ms(
    timestamps: &[u64],
    start_query: u32,
    end_query: u32,
    timestamp_period_ns: f32,
    label: &str,
) -> BenchResult<f64> {
    let start = *timestamps
        .get(start_query as usize)
        .ok_or_else(|| input_error(format!("missing {label} start query {start_query}")))?;
    let end = *timestamps
        .get(end_query as usize)
        .ok_or_else(|| input_error(format!("missing {label} end query {end_query}")))?;
    if end < start {
        return Err(input_error(format!(
            "{label} timestamp decreased: start={start} end={end}"
        )));
    }
    let milliseconds = (end - start) as f64 * f64::from(timestamp_period_ns) / 1_000_000.0;
    if !milliseconds.is_finite() || milliseconds < 0.0 {
        return Err(input_error(format!(
            "invalid decoded {label} duration {milliseconds} ms"
        )));
    }
    Ok(milliseconds)
}

fn print_fft_diagnostics(
    kind: TransformKind,
    point_count: usize,
    fine_length: usize,
    diagnostics: &wgpu_fft::FftDiagnostics,
) {
    println!(
        "FFT_DIAGNOSTICS kind={} N={} M={} fine_grid_length={} route={:?} stage_count={} blockers={} workspace_requirements={:?}",
        kind.name(),
        point_count,
        point_count,
        fine_length,
        diagnostics.route(),
        diagnostics.stages().len(),
        diagnostics.blockers().len(),
        diagnostics.buffer_requirements(),
    );
    for (stage_index, stage) in diagnostics.stages().iter().enumerate() {
        println!(
            "FFT_STAGE kind={} N={} index={} label={} stage_kind={} route={} required_bytes={:?}",
            kind.name(),
            point_count,
            stage_index,
            stage.label,
            stage.kind,
            stage.route,
            stage.required_bytes,
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn print_final_results(
    kind: TransformKind,
    point_count: usize,
    options: &Options,
    transforms: usize,
    timestamp_period_ns: f32,
    stage_series: &[StageSeries],
    envelope_raw_ms: &[f64],
    envelope_run_means_ms: &[f64],
    submit_call: &TimingSeries,
    poll_wait: &TimingSeries,
    submit_wait_total: &TimingSeries,
) -> BenchResult<()> {
    let envelope_stats = statistics(envelope_run_means_ms)?;
    let raw_envelope_stats = statistics(envelope_raw_ms)?;
    let component_stats = stage_series
        .iter()
        .map(|series| Ok((series.stage, statistics(&series.run_means_ms)?)))
        .collect::<BenchResult<Vec<_>>>()?;
    let component_sum_ms = component_stats
        .iter()
        .map(|(_, stats)| stats.mean)
        .sum::<f64>();
    let residual_ms = envelope_stats.mean - component_sum_ms;
    let residual_percent = if envelope_stats.mean == 0.0 {
        0.0
    } else {
        residual_ms / envelope_stats.mean * 100.0
    };
    println!(
        "PROFILE_ENVELOPE kind={} N={} M={} runs={} samples_per_run={} transforms_per_sample={} timestamp_period_ns={:.9} raw_sample_mean_ms={:?} run_mean_ms={:?} avg_ms={:.9} stderr_ms={} stderr_basis=plan-recreated-run-means min_raw_ms={:.9} component_sum_ms={:.9} residual_ms={:.9} residual_percent={:.6}",
        kind.name(),
        point_count,
        point_count,
        options.runs,
        options.samples,
        transforms,
        timestamp_period_ns,
        envelope_raw_ms,
        envelope_run_means_ms,
        envelope_stats.mean,
        format_optional(envelope_stats.stderr),
        raw_envelope_stats.minimum,
        component_sum_ms,
        residual_ms,
        residual_percent,
    );
    for (series, (_, stats)) in stage_series.iter().zip(&component_stats) {
        let raw_stats = statistics(&series.raw_sample_means_ms)?;
        let percent = if envelope_stats.mean == 0.0 {
            0.0
        } else {
            stats.mean / envelope_stats.mean * 100.0
        };
        println!(
            "PROFILE_STAGE kind={} N={} M={} stage={} runs={} samples_per_run={} transforms_per_sample={} raw_sample_mean_ms={:?} run_mean_ms={:?} avg_ms={:.9} stderr_ms={} stderr_basis=plan-recreated-run-means min_raw_ms={:.9} percent_of_pipeline_envelope={:.6}",
            kind.name(),
            point_count,
            point_count,
            series.stage.label(),
            options.runs,
            options.samples,
            transforms,
            series.raw_sample_means_ms,
            series.run_means_ms,
            stats.mean,
            format_optional(stats.stderr),
            raw_stats.minimum,
            percent,
        );
    }
    print_host_timing_result(
        kind,
        point_count,
        options,
        transforms,
        "queue-submit-call",
        submit_call,
    )?;
    print_host_timing_result(
        kind,
        point_count,
        options,
        transforms,
        "device-poll-wait",
        poll_wait,
    )?;
    print_host_timing_result(
        kind,
        point_count,
        options,
        transforms,
        "submit-plus-poll-wait",
        submit_wait_total,
    )?;
    Ok(())
}

fn print_host_timing_result(
    kind: TransformKind,
    point_count: usize,
    options: &Options,
    transforms: usize,
    span: &str,
    series: &TimingSeries,
) -> BenchResult<()> {
    let stats = statistics(&series.run_means_ms)?;
    let raw_stats = statistics(&series.raw_sample_means_ms)?;
    println!(
        "PROFILE_HOST kind={} N={} M={} span={} runs={} samples_per_run={} transforms_per_sample={} raw_sample_ms_per_transform={:?} run_mean_ms_per_transform={:?} avg_ms_per_transform={:.9} stderr_ms_per_transform={} stderr_basis=plan-recreated-run-means min_raw_ms_per_transform={:.9} scope=profiled-submission-with-query-resolve-copy-readback",
        kind.name(),
        point_count,
        point_count,
        span,
        options.runs,
        options.samples,
        transforms,
        series.raw_sample_means_ms,
        series.run_means_ms,
        stats.mean,
        format_optional(stats.stderr),
        raw_stats.minimum,
    );
    Ok(())
}

fn validate_stage_order(
    expected: &[StageSeries],
    actual: &[(NufftGpuStage, f64)],
    kind: TransformKind,
) -> BenchResult<()> {
    let expected = expected
        .iter()
        .map(|series| series.stage)
        .collect::<Vec<_>>();
    let actual = actual.iter().map(|&(stage, _)| stage).collect::<Vec<_>>();
    if expected != actual {
        return Err(input_error(format!(
            "{} stage order changed: expected {expected:?}, actual {actual:?}",
            kind.name()
        )));
    }
    Ok(())
}

fn format_stage_values(values: &[(NufftGpuStage, f64)]) -> String {
    values
        .iter()
        .map(|(stage, value)| format!("{}={value:.9}", stage.label()))
        .collect::<Vec<_>>()
        .join(",")
}

fn format_optional(value: Option<f64>) -> String {
    value.map_or_else(|| "undefined".to_owned(), |value| format!("{value:.9}"))
}

fn benchmark_config(size: usize) -> NufftConfig {
    NufftConfig::new([size], DEFAULT_EPS)
        .with_sigma(DEFAULT_SIGMA)
        .with_sign(NufftSign::Positive)
        .with_mode_order(ModeOrder::Centered)
}

fn create_case_buffers(device: &wgpu::Device, size: usize) -> BenchResult<CaseBuffers> {
    let (points, strengths, modes) = generate_case_data(size)?;
    let usage = wgpu::BufferUsages::STORAGE;
    let points = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("wgpu_nufft.stage_profile.points"),
        contents: bytemuck::cast_slice(&points),
        usage,
    });
    let strengths = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("wgpu_nufft.stage_profile.strengths"),
        contents: bytemuck::cast_slice(&strengths),
        usage,
    });
    let modes = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("wgpu_nufft.stage_profile.modes"),
        contents: bytemuck::cast_slice(&modes),
        usage,
    });
    let complex_bytes = NufftPlan::required_type1_strength_buffer_size_bytes(size)?;
    let type1_output = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_nufft.stage_profile.type1_output"),
        size: complex_bytes,
        usage,
        mapped_at_creation: false,
    });
    let type2_output = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_nufft.stage_profile.type2_output"),
        size: complex_bytes,
        usage,
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

fn generate_case_data(size: usize) -> BenchResult<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    let complex_values = size
        .checked_mul(2)
        .ok_or_else(|| input_error("benchmark complex value count overflow"))?;
    let mut point_rng = Lcg32::new(DATA_SEED ^ POINT_SEED_MASK);
    let mut strength_real_rng = Lcg32::new(DATA_SEED ^ STRENGTH_REAL_SEED_MASK);
    let mut strength_imag_rng = Lcg32::new(DATA_SEED ^ STRENGTH_IMAG_SEED_MASK);
    let mut mode_real_rng = Lcg32::new(DATA_SEED ^ MODE_REAL_SEED_MASK);
    let mut mode_imag_rng = Lcg32::new(DATA_SEED ^ MODE_IMAG_SEED_MASK);
    let mut points = Vec::with_capacity(size);
    for _ in 0..size {
        points.push(
            -std::f32::consts::PI + std::f32::consts::TAU * point_rng.next_unit_interval_f32(),
        );
    }
    let mut strengths = Vec::with_capacity(complex_values);
    for _ in 0..size {
        strengths.push(-1.0 + 2.0 * strength_real_rng.next_unit_interval_f32());
        strengths.push(-1.0 + 2.0 * strength_imag_rng.next_unit_interval_f32());
    }
    let mut modes = Vec::with_capacity(complex_values);
    for _ in 0..size {
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

fn wait_for_submission(
    device: &wgpu::Device,
    submission: wgpu::SubmissionIndex,
    context: &str,
) -> BenchResult<()> {
    device
        .poll(wgpu::PollType::Wait {
            submission_index: Some(submission),
            timeout: Some(WAIT_TIMEOUT),
        })
        .map_err(|error| contextual_error(context, error))?;
    Ok(())
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
            "  index={index} name={:?} vendor={:#x} device={:#x} type={:?} driver={:?}",
            info.name, info.vendor, info.device, info.device_type, info.driver
        );
    }

    let selected_position = if let Some(selector) = selector {
        if let Ok(index) = selector.parse::<usize>() {
            if index >= adapters.len() {
                return Err(input_error(format!(
                    "adapter index {index} is out of range for {} Vulkan adapters",
                    adapters.len()
                )));
            }
            index
        } else {
            let selector = selector.to_ascii_lowercase();
            let matches = adapters
                .iter()
                .enumerate()
                .filter_map(|(index, (_, info))| {
                    info.name
                        .to_ascii_lowercase()
                        .contains(&selector)
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
        "selected Vulkan adapter index={selected_position} name={:?} vendor={:#x} device={:#x} type={:?}",
        info.name, info.vendor, info.device, info.device_type
    );
    Ok(adapter)
}

fn is_hardware_adapter(info: &wgpu::AdapterInfo) -> bool {
    matches!(
        info.device_type,
        wgpu::DeviceType::DiscreteGpu | wgpu::DeviceType::IntegratedGpu
    )
}

fn parse_options() -> BenchResult<Options> {
    let mut adapter_selector = None;
    let mut runs = DEFAULT_RUNS;
    let mut samples = DEFAULT_SAMPLES;
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
  cargo bench -p wgpu-nufft --features gpu-profiling --bench nufft_stage_profile -- [options]

Options:
  --adapter <index-or-name>  Select a Vulkan hardware adapter.
  --runs <count>             Plan recreations per case (default: 3).
  --samples <count>          Timestamp-query submissions per plan (default: 10).
  --help                     Show this help.

Fixed cases use N=M=262144 and 1048576, f32, eps=1e-6, sigma=2,
positive sign, and centered mode order."#
    );
}

#[derive(Debug)]
struct ContextError<E> {
    context: String,
    source: E,
}

impl<E: fmt::Display> fmt::Display for ContextError<E> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}: {}", self.context, self.source)
    }
}

impl<E: Error + 'static> Error for ContextError<E> {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        Some(&self.source)
    }
}

fn contextual_error<E: Error + 'static>(context: impl Into<String>, source: E) -> Box<dyn Error> {
    Box::new(ContextError {
        context: context.into(),
        source,
    })
}

fn input_error(message: impl Into<String>) -> Box<dyn Error> {
    Box::new(io::Error::new(io::ErrorKind::InvalidInput, message.into()))
}
