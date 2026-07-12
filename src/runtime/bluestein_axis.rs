use bytemuck::{Pod, Zeroable};

use crate::config::{FftDirection, Normalization};
use crate::error::{FftError, Result};
use crate::math::{reference_c2c_nd_f64, Complex32, Complex64};
use crate::runtime::axis_plan::{
    generate_fused_scratch_fft_stages_wgsl, AxisLayout, AxisPlan, AxisPlanConfig, AxisPrecision,
    AxisStageKind, AxisTwiddleLutPool,
};
use crate::runtime::axis_policy::next_smooth_at_least;
use crate::runtime::buffer_view::BufferView;
use crate::runtime::dispatch::{max_workgroups_per_dimension, split_workgroups};
use crate::runtime::large_graph::{ElementFormat, HelperBufferRange};
use crate::runtime::nd_wgsl::{
    format_wgsl_f32, lines_per_batch, product, stride_for_axis, wgsl_line_base_fn,
};
use crate::runtime::pipeline_cache::{
    with_device_pipeline_cache, BluesteinKernelKind, BluesteinStageKey, ComputePipelineCacheKey,
    FusedPrimeKind, FusedPrimeStageKey, ShaderCacheKey,
};
use crate::runtime::twiddle::create_twiddle_lut_buffer_for_len_with_precision;
use crate::runtime::window_scheduler::WindowScheduler;

const WORKGROUP_SIZE: u32 = 64;
const FUSED_WORKGROUP_SIZE: u32 = 256;
// Avoid a 256-lane whole-pipeline shader when the convolution is too small to
// keep even half of the workgroup useful; the staged path is already cheap.
const FUSED_MIN_CONVOLUTION_LENGTH: usize = 128;

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct BluesteinLinesParams {
    lines: u32,
    line_offset: u32,
    _pad0: u32,
    _pad1: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct BluesteinTotalParams {
    total: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct BluesteinAxisConfig {
    pub(crate) shape: Vec<usize>,
    pub(crate) axis: usize,
    pub(crate) batch: usize,
    pub(crate) direction: FftDirection,
    pub(crate) normalization: Normalization,
    pub(crate) precision: AxisPrecision,
}

pub(crate) struct BluesteinAxis {
    n: usize,
    m: usize,
    lines: u32,
    precision: AxisPrecision,
    lines_params_buffer: wgpu::Buffer,
    chirp_buffer: wgpu::Buffer,
    bfft_buffer: wgpu::Buffer,
    execution: BluesteinExecution,
}

enum BluesteinExecution {
    Fused(FusedBluesteinExecution),
    MultiPass(Box<MultiPassBluesteinExecution>),
}

struct FusedBluesteinExecution {
    workgroups: u32,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    twiddle_buffer: wgpu::Buffer,
}

struct MultiPassBluesteinExecution {
    workgroups_work: u32,
    workgroups_output: u32,
    pack_pipeline: wgpu::ComputePipeline,
    pack_bind_group_layout: wgpu::BindGroupLayout,
    mul_pipeline: wgpu::ComputePipeline,
    mul_bind_group_layout: wgpu::BindGroupLayout,
    post_pipeline: wgpu::ComputePipeline,
    post_bind_group_layout: wgpu::BindGroupLayout,
    total_params_buffer: wgpu::Buffer,
    work_buffer: wgpu::Buffer,
    fft_buffer: wgpu::Buffer,
    work_fft_forward: AxisPlan,
    work_fft_inverse: AxisPlan,
}

impl BluesteinAxisConfig {
    pub(crate) fn validate(&self) -> Result<()> {
        if self.shape.is_empty() || self.shape.iter().any(|&len| len == 0) {
            return Err(FftError::ZeroLength);
        }
        if self.axis >= self.shape.len() {
            return Err(FftError::InvalidAxis {
                axis: self.axis,
                rank: self.shape.len(),
            });
        }
        if self.batch == 0 {
            return Err(FftError::ZeroBatch);
        }
        if self.shape[self.axis] < 2 {
            return Err(FftError::UnsupportedLength {
                len: self.shape[self.axis],
            });
        }
        total_complex(&self.shape, self.batch)?;
        Ok(())
    }

    fn scale(&self) -> Result<f64> {
        Ok(match self.precision {
            AxisPrecision::F32 => {
                let total = product(&self.shape) as f32;
                f64::from(match (self.direction, self.normalization) {
                    (_, Normalization::None) => 1.0,
                    (FftDirection::Forward, Normalization::Forward) => 1.0 / total,
                    (FftDirection::Inverse, Normalization::Inverse) => 1.0 / total,
                    (_, Normalization::Orthogonal) => 1.0 / total.sqrt(),
                    _ => 1.0,
                })
            }
            AxisPrecision::F64 => {
                let total = product(&self.shape) as f64;
                match (self.direction, self.normalization) {
                    (_, Normalization::None) => 1.0,
                    (FftDirection::Forward, Normalization::Forward) => 1.0 / total,
                    (FftDirection::Inverse, Normalization::Inverse) => 1.0 / total,
                    (_, Normalization::Orthogonal) => 1.0 / total.sqrt(),
                    _ => 1.0,
                }
            }
            AxisPrecision::Df64 => {
                unreachable!("df64 Bluestein plans are rejected before scale generation")
            }
        })
    }
}

impl BluesteinAxis {
    pub(crate) fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        config: BluesteinAxisConfig,
    ) -> Result<Self> {
        config.validate()?;
        if config.precision == AxisPrecision::Df64 {
            return Err(FftError::PrecisionUnsupported {
                requested: crate::config::FftPrecision::Df64,
                route: "bluestein",
                reason: "bluestein-df64-not-implemented",
            });
        }

        let n = config.shape[config.axis];
        let m = bluestein_convolution_length(n)?;
        let factors = crate::runtime::factor_supported_length(m)?;

        let lines = checked_mul(config.batch, lines_per_batch(&config.shape, config.axis))?;
        let lines_u32 = lines as u32;
        let stride_complex = stride_for_axis(&config.shape, config.axis);
        let scale = config.scale()?;
        let apply_scale = scale != 1.0;

        let chirp = bluestein_chirp_f64(n, config.direction);
        let bfft = bluestein_bfft_f64(n, m, config.direction)?;
        let complex_bytes = config.precision.complex_size_bytes();

        let lines_params_buffer =
            uniform_buffer::<BluesteinLinesParams>(device, "wgpu_fft.bluestein.lines_params");
        queue.write_buffer(
            &lines_params_buffer,
            0,
            bytemuck::bytes_of(&BluesteinLinesParams {
                lines: lines_u32,
                line_offset: 0,
                _pad0: 0,
                _pad1: 0,
            }),
        );

        let chirp_buffer = storage_buffer(
            device,
            "wgpu_fft.bluestein.chirp",
            n as u64 * complex_bytes,
            wgpu::BufferUsages::COPY_DST,
        )?;
        match config.precision {
            AxisPrecision::F32 => {
                let values = chirp
                    .iter()
                    .copied()
                    .map(round_complex64)
                    .collect::<Vec<_>>();
                queue.write_buffer(&chirp_buffer, 0, bytemuck::cast_slice(&values));
            }
            AxisPrecision::F64 => {
                queue.write_buffer(&chirp_buffer, 0, bytemuck::cast_slice(&chirp));
            }
            AxisPrecision::Df64 => {
                unreachable!("df64 Bluestein plans are rejected before chirp upload")
            }
        }

        let bfft_buffer = storage_buffer(
            device,
            "wgpu_fft.bluestein.bfft",
            m as u64 * complex_bytes,
            wgpu::BufferUsages::COPY_DST,
        )?;
        match config.precision {
            AxisPrecision::F32 => {
                let values = bfft
                    .iter()
                    .copied()
                    .map(round_complex64)
                    .collect::<Vec<_>>();
                queue.write_buffer(&bfft_buffer, 0, bytemuck::cast_slice(&values));
            }
            AxisPrecision::F64 => {
                queue.write_buffer(&bfft_buffer, 0, bytemuck::cast_slice(&bfft));
            }
            AxisPrecision::Df64 => {
                unreachable!("df64 Bluestein plans are rejected before bfft upload")
            }
        }

        let execution = if fused_bluestein_supported(m, config.precision, &device.limits()) {
            let twiddle_buffer = create_twiddle_lut_buffer_for_len_with_precision(
                device,
                queue,
                "wgpu_fft.bluestein.fused.twiddle_lut",
                m,
                config.precision.as_fft_precision(),
            )?;
            let shader_key = FusedPrimeStageKey::new(
                FusedPrimeKind::Bluestein,
                config.shape.len(),
                config.axis,
                &config.shape,
                n,
                stride_complex,
                m,
                &factors,
                config.direction,
                FUSED_WORKGROUP_SIZE,
                apply_scale,
                scale,
                config.precision,
            );
            let pipeline_key = ComputePipelineCacheKey::fused_prime_stage(shader_key.clone());
            let bind_group_layout = with_device_pipeline_cache(device, |cache| {
                cache.get_bind_group_layout(device, pipeline_key.layout)
            });
            let pipeline = cached_fused_bluestein_pipeline(device, &pipeline_key, &shader_key);
            BluesteinExecution::Fused(FusedBluesteinExecution {
                workgroups: lines_u32,
                pipeline,
                bind_group_layout,
                twiddle_buffer,
            })
        } else {
            let work_complex = checked_mul(lines, m)?;
            if work_complex > u32::MAX as usize {
                return Err(FftError::LengthTooLarge { len: work_complex });
            }
            let total_work_u32 = work_complex as u32;
            let total_params_buffer =
                uniform_buffer::<BluesteinTotalParams>(device, "wgpu_fft.bluestein.total_params");
            queue.write_buffer(
                &total_params_buffer,
                0,
                bytemuck::bytes_of(&BluesteinTotalParams {
                    total: total_work_u32,
                    _pad0: 0,
                    _pad1: 0,
                    _pad2: 0,
                }),
            );
            let work_bytes = work_complex as u64 * complex_bytes;
            let work_buffer = storage_buffer(
                device,
                "wgpu_fft.bluestein.work",
                work_bytes,
                wgpu::BufferUsages::empty(),
            )?;
            let fft_buffer = storage_buffer(
                device,
                "wgpu_fft.bluestein.fft",
                work_bytes,
                wgpu::BufferUsages::empty(),
            )?;

            let mut twiddle_lut_pool = AxisTwiddleLutPool::default();
            let work_fft_forward = AxisPlan::new_with_twiddle_lut_pool(
                device,
                queue,
                AxisPlanConfig {
                    shape: vec![m],
                    axes: vec![0],
                    batch: lines,
                    direction: FftDirection::Forward,
                    normalization: Normalization::None,
                    scale_override_bits: None,
                    layout: AxisLayout::Interleaved,
                    precision: config.precision,
                },
                &mut twiddle_lut_pool,
            )?;
            let work_fft_inverse = AxisPlan::new_with_twiddle_lut_pool(
                device,
                queue,
                AxisPlanConfig {
                    shape: vec![m],
                    axes: vec![0],
                    batch: lines,
                    direction: FftDirection::Inverse,
                    normalization: Normalization::Inverse,
                    scale_override_bits: None,
                    layout: AxisLayout::Interleaved,
                    precision: config.precision,
                },
                &mut twiddle_lut_pool,
            )?;

            let stage_key = |kind, stage_applies_scale| {
                BluesteinStageKey::new(
                    kind,
                    config.shape.len(),
                    config.axis,
                    &config.shape,
                    n,
                    stride_complex,
                    m,
                    WORKGROUP_SIZE,
                    stage_applies_scale,
                    scale,
                    config.precision,
                )
            };
            let pack_key = ComputePipelineCacheKey::bluestein_stage(stage_key(
                BluesteinKernelKind::Pack,
                false,
            ));
            let mul_key = ComputePipelineCacheKey::bluestein_stage(stage_key(
                BluesteinKernelKind::Mul,
                false,
            ));
            let post_key = ComputePipelineCacheKey::bluestein_stage(stage_key(
                BluesteinKernelKind::Post,
                apply_scale,
            ));
            let pack_bind_group_layout = cached_layout(device, pack_key.layout);
            let mul_bind_group_layout = cached_layout(device, mul_key.layout);
            let post_bind_group_layout = cached_layout(device, post_key.layout);
            let pack_pipeline = cached_bluestein_pipeline(device, &pack_key)?;
            let mul_pipeline = cached_bluestein_pipeline(device, &mul_key)?;
            let post_pipeline = cached_bluestein_pipeline(device, &post_key)?;

            BluesteinExecution::MultiPass(Box::new(MultiPassBluesteinExecution {
                workgroups_work: total_work_u32.div_ceil(WORKGROUP_SIZE),
                workgroups_output: (lines_u32 * n as u32).div_ceil(WORKGROUP_SIZE),
                pack_pipeline,
                pack_bind_group_layout,
                mul_pipeline,
                mul_bind_group_layout,
                post_pipeline,
                post_bind_group_layout,
                total_params_buffer,
                work_buffer,
                fft_buffer,
                work_fft_forward,
                work_fft_inverse,
            }))
        };

        Ok(Self {
            n,
            m,
            lines: lines_u32,
            precision: config.precision,
            lines_params_buffer,
            chirp_buffer,
            bfft_buffer,
            execution,
        })
    }

    pub(crate) fn workspace_size_bytes(&self) -> u64 {
        0
    }

    pub(crate) fn twiddle_lut_storage_bytes(&self) -> u64 {
        match &self.execution {
            BluesteinExecution::Fused(execution) => execution.twiddle_buffer.size(),
            BluesteinExecution::MultiPass(execution) => {
                let bytes = execution.work_fft_forward.twiddle_lut_storage_bytes();
                debug_assert_eq!(
                    bytes,
                    execution.work_fft_inverse.twiddle_lut_storage_bytes()
                );
                bytes
            }
        }
    }

    pub(crate) fn graph_is_fused(&self) -> bool {
        matches!(self.execution, BluesteinExecution::Fused(_))
    }

    pub(crate) fn graph_forward_fft_stage_kinds(&self) -> Vec<AxisStageKind> {
        match &self.execution {
            BluesteinExecution::Fused(_) => Vec::new(),
            BluesteinExecution::MultiPass(execution) => {
                execution.work_fft_forward.graph_stage_kinds()
            }
        }
    }

    pub(crate) fn graph_forward_fft_workspace_bytes(&self) -> u64 {
        match &self.execution {
            BluesteinExecution::Fused(_) => 0,
            BluesteinExecution::MultiPass(execution) => {
                execution.work_fft_forward.workspace_size_bytes()
            }
        }
    }

    pub(crate) fn graph_inverse_fft_stage_kinds(&self) -> Vec<AxisStageKind> {
        match &self.execution {
            BluesteinExecution::Fused(_) => Vec::new(),
            BluesteinExecution::MultiPass(execution) => {
                execution.work_fft_inverse.graph_stage_kinds()
            }
        }
    }

    pub(crate) fn graph_inverse_fft_workspace_bytes(&self) -> u64 {
        match &self.execution {
            BluesteinExecution::Fused(_) => 0,
            BluesteinExecution::MultiPass(execution) => {
                execution.work_fft_inverse.workspace_size_bytes()
            }
        }
    }

    pub(crate) fn graph_helper_buffers(&self) -> Vec<HelperBufferRange> {
        let element_format = self.precision.element_format();
        let mut helpers = vec![
            HelperBufferRange {
                label: "bluestein-chirp-helper",
                index: 0,
                size_bytes: self.chirp_buffer.size(),
                format: element_format,
            },
            HelperBufferRange {
                label: "bluestein-bfft-helper",
                index: 1,
                size_bytes: self.bfft_buffer.size(),
                format: element_format,
            },
        ];
        if let BluesteinExecution::MultiPass(execution) = &self.execution {
            helpers.extend([
                HelperBufferRange {
                    label: "bluestein-work-helper",
                    index: 2,
                    size_bytes: execution.work_buffer.size(),
                    format: element_format,
                },
                HelperBufferRange {
                    label: "bluestein-fft-helper",
                    index: 3,
                    size_bytes: execution.fft_buffer.size(),
                    format: element_format,
                },
            ]);
        }
        helpers
    }

    pub(crate) fn execute_views(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        input: BufferView<'_>,
        output: BufferView<'_>,
    ) -> Result<()> {
        debug_assert!(self.n > 1);
        debug_assert!(self.m >= 2 * self.n - 1);
        debug_assert!(self.lines > 0);

        match &self.execution {
            BluesteinExecution::Fused(execution) => {
                self.dispatch_fused(device, encoder, input, output, execution)?;
            }
            BluesteinExecution::MultiPass(execution) => {
                self.dispatch_pack(device, encoder, input, execution)?;
                execution.work_fft_forward.execute(
                    device,
                    encoder,
                    &execution.work_buffer,
                    &execution.fft_buffer,
                )?;
                self.dispatch_mul(device, encoder, execution)?;
                execution.work_fft_inverse.execute(
                    device,
                    encoder,
                    &execution.fft_buffer,
                    &execution.work_buffer,
                )?;
                self.dispatch_post(device, encoder, output, execution)?;
            }
        }
        Ok(())
    }

    fn dispatch_fused(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        input: BufferView<'_>,
        output: BufferView<'_>,
        execution: &FusedBluesteinExecution,
    ) -> Result<()> {
        let scheduler = WindowScheduler::for_device(device);
        let element_format = self.precision.element_format();
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("wgpu_fft.bluestein.fused.bind_group"),
            layout: &execution.bind_group_layout,
            entries: &[
                bind_view_entry(&scheduler, 0, input, element_format)?,
                bind_view_entry(&scheduler, 1, output, element_format)?,
                bind_storage_entry(&scheduler, 2, &self.chirp_buffer, element_format)?,
                bind_storage_entry(&scheduler, 3, &self.bfft_buffer, element_format)?,
                bind_storage_entry(&scheduler, 4, &execution.twiddle_buffer, element_format)?,
                bind_uniform_entry(5, &self.lines_params_buffer),
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("wgpu_fft.bluestein.fused.pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&execution.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let (x, y, z) =
            split_workgroups(execution.workgroups, max_workgroups_per_dimension(device))?;
        pass.dispatch_workgroups(x, y, z);
        Ok(())
    }

    fn dispatch_pack(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        input: BufferView<'_>,
        execution: &MultiPassBluesteinExecution,
    ) -> Result<()> {
        let scheduler = WindowScheduler::for_device(device);
        let element_format = self.precision.element_format();
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("wgpu_fft.bluestein.pack.bind_group"),
            layout: &execution.pack_bind_group_layout,
            entries: &[
                bind_view_entry(&scheduler, 0, input, element_format)?,
                bind_storage_entry(&scheduler, 1, &execution.work_buffer, element_format)?,
                bind_storage_entry(&scheduler, 2, &self.chirp_buffer, element_format)?,
                bind_uniform_entry(3, &self.lines_params_buffer),
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("wgpu_fft.bluestein.pack.pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&execution.pack_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let (x, y, z) = split_workgroups(
            execution.workgroups_work,
            max_workgroups_per_dimension(device),
        )?;
        pass.dispatch_workgroups(x, y, z);
        Ok(())
    }

    fn dispatch_mul(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        execution: &MultiPassBluesteinExecution,
    ) -> Result<()> {
        let scheduler = WindowScheduler::for_device(device);
        let element_format = self.precision.element_format();
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("wgpu_fft.bluestein.mul.bind_group"),
            layout: &execution.mul_bind_group_layout,
            entries: &[
                bind_storage_entry(&scheduler, 0, &execution.fft_buffer, element_format)?,
                bind_storage_entry(&scheduler, 1, &self.bfft_buffer, element_format)?,
                bind_uniform_entry(2, &execution.total_params_buffer),
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("wgpu_fft.bluestein.mul.pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&execution.mul_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let (x, y, z) = split_workgroups(
            execution.workgroups_work,
            max_workgroups_per_dimension(device),
        )?;
        pass.dispatch_workgroups(x, y, z);
        Ok(())
    }

    fn dispatch_post(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        output: BufferView<'_>,
        execution: &MultiPassBluesteinExecution,
    ) -> Result<()> {
        let scheduler = WindowScheduler::for_device(device);
        let element_format = self.precision.element_format();
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("wgpu_fft.bluestein.post.bind_group"),
            layout: &execution.post_bind_group_layout,
            entries: &[
                bind_storage_entry(&scheduler, 0, &execution.work_buffer, element_format)?,
                bind_storage_entry(&scheduler, 1, &self.chirp_buffer, element_format)?,
                bind_view_entry(&scheduler, 2, output, element_format)?,
                bind_uniform_entry(3, &self.lines_params_buffer),
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("wgpu_fft.bluestein.post.pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&execution.post_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let (x, y, z) = split_workgroups(
            execution.workgroups_output,
            max_workgroups_per_dimension(device),
        )?;
        pass.dispatch_workgroups(x, y, z);
        Ok(())
    }
}

pub(crate) fn bluestein_chirp(n: usize, direction: FftDirection) -> Vec<Complex32> {
    bluestein_chirp_f64(n, direction)
        .into_iter()
        .map(round_complex64)
        .collect()
}

pub(crate) fn bluestein_chirp_f64(n: usize, direction: FftDirection) -> Vec<Complex64> {
    let sign = transform_sign(direction);
    (0..n).map(|i| bluestein_phase(n, i, sign)).collect()
}

pub(crate) fn bluestein_bfft(
    n: usize,
    m: usize,
    direction: FftDirection,
) -> Result<Vec<Complex32>> {
    Ok(bluestein_bfft_f64(n, m, direction)?
        .into_iter()
        .map(round_complex64)
        .collect())
}

pub(crate) fn bluestein_bfft_f64(
    n: usize,
    m: usize,
    direction: FftDirection,
) -> Result<Vec<Complex64>> {
    let sign = transform_sign(direction);
    let mut values = vec![Complex64::default(); m];
    values[0] = Complex64::new(1.0, 0.0);
    for i in 1..n {
        let value = bluestein_phase(n, i, -sign);
        values[i] = value;
        values[m - i] = value;
    }

    let config = crate::config::FftConfig::new(m).with_normalization(Normalization::None);
    reference_c2c_nd_f64(&values, &config)
}

fn bluestein_phase(n: usize, i: usize, sign: f64) -> Complex64 {
    debug_assert!(n > 0);
    debug_assert!(i < n);
    let modulus = 2 * n as u128;
    let i = i as u128;
    let square_mod = (i * i) % modulus;
    let angle = sign * std::f64::consts::PI * square_mod as f64 / n as f64;
    let (sin, cos) = angle.sin_cos();
    Complex64::new(cos, sin)
}

fn round_complex64(value: Complex64) -> Complex32 {
    Complex32::new(value.re as f32, value.im as f32)
}

fn transform_sign(direction: FftDirection) -> f64 {
    match direction {
        FftDirection::Forward => -1.0,
        FftDirection::Inverse => 1.0,
    }
}

fn fused_bluestein_supported(m: usize, precision: AxisPrecision, limits: &wgpu::Limits) -> bool {
    fused_bluestein_supported_by_limits(
        m,
        precision,
        u64::from(limits.max_compute_workgroup_storage_size),
        limits.max_compute_invocations_per_workgroup,
        limits.max_compute_workgroup_size_x,
    )
}

pub(crate) fn fused_bluestein_supported_by_limits(
    m: usize,
    precision: AxisPrecision,
    max_workgroup_storage_bytes: u64,
    max_invocations_per_workgroup: u32,
    max_workgroup_size_x: u32,
) -> bool {
    if m < FUSED_MIN_CONVOLUTION_LENGTH {
        return false;
    }
    let Some(scratch_bytes) = m.checked_mul(precision.complex_size_bytes() as usize) else {
        return false;
    };
    scratch_bytes as u64 <= max_workgroup_storage_bytes
        && FUSED_WORKGROUP_SIZE <= max_invocations_per_workgroup
        && FUSED_WORKGROUP_SIZE <= max_workgroup_size_x
}

pub(crate) fn bluestein_convolution_length(n: usize) -> Result<usize> {
    let min_conv = n
        .checked_mul(2)
        .and_then(|value| value.checked_sub(1))
        .ok_or(FftError::LengthTooLarge { len: n })?;
    Ok(next_smooth_at_least(min_conv))
}

fn checked_mul(left: usize, right: usize) -> Result<usize> {
    left.checked_mul(right)
        .ok_or(FftError::LengthTooLarge { len: usize::MAX })
}

fn total_complex(shape: &[usize], batch: usize) -> Result<usize> {
    let total = product(shape)
        .checked_mul(batch)
        .ok_or(FftError::LengthTooLarge { len: usize::MAX })?;
    if total > u32::MAX as usize {
        Err(FftError::LengthTooLarge { len: total })
    } else {
        Ok(total)
    }
}

fn uniform_buffer<T>(device: &wgpu::Device, label: &'static str) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: std::mem::size_of::<T>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

fn storage_buffer(
    device: &wgpu::Device,
    label: &'static str,
    size: u64,
    extra_usage: wgpu::BufferUsages,
) -> Result<wgpu::Buffer> {
    let max_buffer_size = device.limits().max_buffer_size;
    if size > max_buffer_size {
        return Err(FftError::HelperBufferTooLarge {
            helper_buffer: label,
            requested_bytes: size,
            max_buffer_size,
        });
    }
    Ok(device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: wgpu::BufferUsages::STORAGE | extra_usage,
        mapped_at_creation: false,
    }))
}

fn cached_fused_bluestein_pipeline(
    device: &wgpu::Device,
    key: &ComputePipelineCacheKey,
    shader_key: &FusedPrimeStageKey,
) -> wgpu::ComputePipeline {
    let stable_key = key.stable_key();
    let pipeline_label = format!("wgpu_fft.bluestein.fused.pipeline.{stable_key}");
    let shader_label = format!("wgpu_fft.bluestein.fused.shader.{stable_key}");
    with_device_pipeline_cache(device, |cache| {
        cache.get_compute_pipeline(device, key, &pipeline_label, &shader_label, || {
            generate_fused_bluestein_wgsl_for_key(shader_key)
        })
    })
}

fn cached_layout(
    device: &wgpu::Device,
    key: crate::runtime::pipeline_cache::PipelineLayoutCacheKey,
) -> wgpu::BindGroupLayout {
    with_device_pipeline_cache(device, |cache| cache.get_bind_group_layout(device, key))
}

fn bluestein_stage_key(key: &ComputePipelineCacheKey) -> Result<&BluesteinStageKey> {
    match &key.shader {
        ShaderCacheKey::BluesteinStage(stage) => Ok(stage),
        _ => Err(FftError::LargeGraphStageUnsupported {
            stage: "bluestein-pipeline-key",
            reason: "Bluestein pipeline key has a non-Bluestein shader stage",
        }),
    }
}

fn cached_bluestein_pipeline(
    device: &wgpu::Device,
    key: &ComputePipelineCacheKey,
) -> Result<wgpu::ComputePipeline> {
    let stage_key = bluestein_stage_key(key)?;
    let stable_key = key.stable_key();
    let pipeline_label = format!("wgpu_fft.bluestein.pipeline.{stable_key}");
    let shader_label = format!("wgpu_fft.bluestein.shader.{stable_key}");
    Ok(with_device_pipeline_cache(device, |cache| {
        cache.get_compute_pipeline(device, key, &pipeline_label, &shader_label, || {
            generate_bluestein_wgsl_for_key(stage_key)
        })
    }))
}

fn bind_uniform_entry<'a>(
    binding: u32,
    uniform_buffer: &'a wgpu::Buffer,
) -> wgpu::BindGroupEntry<'a> {
    wgpu::BindGroupEntry {
        binding,
        resource: uniform_buffer.as_entire_binding(),
    }
}

fn bind_storage_entry<'a>(
    scheduler: &WindowScheduler,
    binding: u32,
    buffer: &'a wgpu::Buffer,
    format: ElementFormat,
) -> Result<wgpu::BindGroupEntry<'a>> {
    let view = BufferView::whole(buffer);
    Ok(wgpu::BindGroupEntry {
        binding,
        resource: scheduler.storage_binding_resource(&view, format)?,
    })
}

fn bind_view_entry<'a>(
    scheduler: &WindowScheduler,
    binding: u32,
    view: BufferView<'a>,
    format: ElementFormat,
) -> Result<wgpu::BindGroupEntry<'a>> {
    Ok(wgpu::BindGroupEntry {
        binding,
        resource: scheduler.storage_binding_resource(&view, format)?,
    })
}

fn complex_mul_wgsl() -> &'static str {
    r#"fn c_mul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
  return vec2<f32>(
    a.x * b.x - a.y * b.y,
    a.x * b.y + a.y * b.x
  );
}"#
}

pub(crate) fn generate_fused_bluestein_wgsl_for_key(key: &FusedPrimeStageKey) -> String {
    debug_assert_eq!(key.kind, FusedPrimeKind::Bluestein);
    let n = key.axis_length;
    let m = key.convolution_length;
    let m_slot_count = m.div_ceil(key.workgroup_size as usize);
    let n_slot_count = n.div_ceil(key.workgroup_size as usize);
    let line_base_fn = wgsl_line_base_fn(key.rank, key.axis, &key.dims);
    let scale = key.precision.format_wgsl_scalar(key.scale_factor());
    let inverse_m = match key.precision {
        AxisPrecision::F32 => key
            .precision
            .format_wgsl_scalar(f64::from(1.0f32 / m as f32)),
        AxisPrecision::F64 => key.precision.format_wgsl_scalar(1.0 / m as f64),
        AxisPrecision::Df64 => {
            unreachable!("df64 fused Bluestein shaders are not implemented in Phase B")
        }
    };
    let forward_stages = generate_fused_scratch_fft_stages_wgsl(
        m,
        &key.factors,
        FftDirection::Forward,
        key.workgroup_size,
        "scratch",
        "twiddle_forward",
        key.precision,
    );
    let inverse_stages = generate_fused_scratch_fft_stages_wgsl(
        m,
        &key.factors,
        FftDirection::Inverse,
        key.workgroup_size,
        "scratch",
        "twiddle_inverse",
        key.precision,
    );

    key.precision.specialize_wgsl(format!(
        r#"struct Params {{
  lines: u32,
  lineOffset: u32,
  pad0: u32,
  pad1: u32,
}};

@group(0) @binding(0) var<storage, read> input: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> chirp: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read> bfft: array<vec2<f32>>;
@group(0) @binding(4) var<storage, read> axisTwiddles: array<vec2<f32>>;
@group(0) @binding(5) var<uniform> params: Params;

fn c_add(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {{
  return a + b;
}}

fn c_sub(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {{
  return a - b;
}}

fn c_mul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {{
  return vec2<f32>(
    a.x * b.x - a.y * b.y,
    a.x * b.y + a.y * b.x
  );
}}

fn twiddle_forward(index: u32) -> vec2<f32> {{
  return axisTwiddles[index];
}}

fn twiddle_inverse(index: u32) -> vec2<f32> {{
  let value: vec2<f32> = axisTwiddles[index];
  return vec2<f32>(value.x, -value.y);
}}

const N: u32 = {n}u;
const M: u32 = {m}u;
const STRIDE: u32 = {stride}u;
const WORKGROUP_SIZE: u32 = {workgroup_size}u;
const M_SLOT_COUNT: u32 = {m_slot_count}u;
const N_SLOT_COUNT: u32 = {n_slot_count}u;
const INVERSE_M: {scalar} = {inverse_m};
const SCALE: {scalar} = {scale};

var<workgroup> scratch: array<vec2<f32>, {m}>;

{line_base_fn}

@compute @workgroup_size({workgroup_size}, 1, 1)
fn main({entry_params}) {{
  {flat_workgroup_index}
  let lineLocal: u32 = wgFlat;
  if (lineLocal >= params.lines) {{
    return;
  }}

  let base: u32 = line_base(params.lineOffset + lineLocal);
  for (var slot: u32 = 0u; slot < M_SLOT_COUNT; slot = slot + 1u) {{
    let t: u32 = lid.x + slot * WORKGROUP_SIZE;
    if (t < M) {{
      if (t < N) {{
        scratch[t] = c_mul(input[base + t * STRIDE], chirp[t]);
      }} else {{
        scratch[t] = vec2<f32>(0.0, 0.0);
      }}
    }}
  }}
  workgroupBarrier();

{forward_stages}
  for (var slot: u32 = 0u; slot < M_SLOT_COUNT; slot = slot + 1u) {{
    let t: u32 = lid.x + slot * WORKGROUP_SIZE;
    if (t < M) {{
      scratch[t] = c_mul(scratch[t], bfft[t]);
    }}
  }}
  workgroupBarrier();

{inverse_stages}
  for (var slot: u32 = 0u; slot < N_SLOT_COUNT; slot = slot + 1u) {{
    let t: u32 = lid.x + slot * WORKGROUP_SIZE;
    if (t < N) {{
      let convolution: vec2<f32> = scratch[t] * vec2<f32>(INVERSE_M, INVERSE_M);
      let value: vec2<f32> = c_mul(convolution, chirp[t]);
      output[base + t * STRIDE] = value * vec2<f32>(SCALE, SCALE);
    }}
  }}
}}
"#,
        stride = key.stride_complex,
        workgroup_size = key.workgroup_size,
        entry_params = crate::runtime::dispatch::WGSL_FLAT_ENTRY_PARAMS,
        flat_workgroup_index = crate::runtime::dispatch::WGSL_FLAT_WORKGROUP_INDEX,
        scalar = key.precision.wgsl_scalar_type(),
    ))
}

pub(crate) fn generate_bluestein_wgsl_for_key(key: &BluesteinStageKey) -> String {
    match key.kind {
        BluesteinKernelKind::Pack => generate_bluestein_pack_wgsl(key),
        BluesteinKernelKind::Mul => generate_bluestein_mul_wgsl(key),
        BluesteinKernelKind::Post => generate_bluestein_post_wgsl(key),
    }
}

fn generate_bluestein_pack_wgsl(key: &BluesteinStageKey) -> String {
    let line_base_fn = wgsl_line_base_fn(key.rank, key.axis, &key.dims);
    let complex_mul = complex_mul_wgsl();
    key.precision.specialize_wgsl(format!(
        r#"struct Params {{
  lines: u32,
  lineOffset: u32,
  pad0: u32,
  pad1: u32,
}};

@group(0) @binding(0) var<storage, read> input: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> work: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> chirp: array<vec2<f32>>;
@group(0) @binding(3) var<uniform> params: Params;

const N: u32 = {n}u;
const M: u32 = {m}u;
const STRIDE: u32 = {stride}u;

{complex_mul}

{line_base_fn}

@compute @workgroup_size({workgroup_size}, 1, 1)
fn main({entry_params}) {{
  {flat_index}

  let lineLocal: u32 = i / M;
  let t: u32 = i - lineLocal * M;
  let dst: u32 = lineLocal * M + t;
  if (t >= N) {{
    work[dst] = vec2<f32>(0.0, 0.0);
    return;
  }}

  let base: u32 = line_base(params.lineOffset + lineLocal);
  work[dst] = c_mul(input[base + t * STRIDE], chirp[t]);
}}
"#,
        n = key.axis_length,
        m = key.convolution_length,
        stride = key.stride_complex,
        complex_mul = complex_mul,
        line_base_fn = line_base_fn,
        workgroup_size = WORKGROUP_SIZE,
        entry_params = crate::runtime::dispatch::WGSL_FLAT_ENTRY_PARAMS,
        flat_index = crate::runtime::dispatch::wgsl_flat_index_stmts(
            "i",
            "params.lines * M",
            WORKGROUP_SIZE,
        ),
    ))
}

fn generate_bluestein_mul_wgsl(key: &BluesteinStageKey) -> String {
    let complex_mul = complex_mul_wgsl();
    key.precision.specialize_wgsl(format!(
        r#"struct Params {{
  total: u32,
  pad0: u32,
  pad1: u32,
  pad2: u32,
}};

@group(0) @binding(0) var<storage, read_write> work: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read> bfft: array<vec2<f32>>;
@group(0) @binding(2) var<uniform> params: Params;

const M: u32 = {m}u;

{complex_mul}

@compute @workgroup_size({workgroup_size}, 1, 1)
fn main({entry_params}) {{
  {flat_index}
  let t: u32 = i - (i / M) * M;
  work[i] = c_mul(work[i], bfft[t]);
}}
"#,
        m = key.convolution_length,
        complex_mul = complex_mul,
        workgroup_size = WORKGROUP_SIZE,
        entry_params = crate::runtime::dispatch::WGSL_FLAT_ENTRY_PARAMS,
        flat_index =
            crate::runtime::dispatch::wgsl_flat_index_stmts("i", "params.total", WORKGROUP_SIZE),
    ))
}

fn generate_bluestein_post_wgsl(key: &BluesteinStageKey) -> String {
    let line_base_fn = wgsl_line_base_fn(key.rank, key.axis, &key.dims);
    let complex_mul = complex_mul_wgsl();
    let scale = format_staged_scalar(key.precision, key.scale_factor());
    key.precision.specialize_wgsl(format!(
        r#"struct Params {{
  lines: u32,
  lineOffset: u32,
  pad0: u32,
  pad1: u32,
}};

@group(0) @binding(0) var<storage, read> conv: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read> chirp: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> output: array<vec2<f32>>;
@group(0) @binding(3) var<uniform> params: Params;

const N: u32 = {n}u;
const M: u32 = {m}u;
const STRIDE: u32 = {stride}u;
const SCALE: {scalar} = {scale};

{complex_mul}

{line_base_fn}

@compute @workgroup_size({workgroup_size}, 1, 1)
fn main({entry_params}) {{
  {flat_index}

  let lineLocal: u32 = i / N;
  let t: u32 = i - lineLocal * N;
  let baseWork: u32 = lineLocal * M;
  let baseOutput: u32 = line_base(params.lineOffset + lineLocal);
  let value: vec2<f32> = c_mul(conv[baseWork + t], chirp[t]);
  output[baseOutput + t * STRIDE] = value * vec2<f32>(SCALE, SCALE);
}}
"#,
        n = key.axis_length,
        m = key.convolution_length,
        stride = key.stride_complex,
        scale = scale,
        complex_mul = complex_mul,
        line_base_fn = line_base_fn,
        workgroup_size = WORKGROUP_SIZE,
        entry_params = crate::runtime::dispatch::WGSL_FLAT_ENTRY_PARAMS,
        flat_index = crate::runtime::dispatch::wgsl_flat_index_stmts(
            "i",
            "params.lines * N",
            WORKGROUP_SIZE,
        ),
        scalar = key.precision.wgsl_scalar_type(),
    ))
}

fn format_staged_scalar(precision: AxisPrecision, value: f64) -> String {
    match precision {
        AxisPrecision::F32 => format_wgsl_f32(value as f32),
        AxisPrecision::F64 => precision.format_wgsl_scalar(value),
        AxisPrecision::Df64 => {
            unreachable!("df64 Bluestein shader constants are not implemented in Phase B")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fused_bluestein_gate_respects_storage_lane_limits_and_tiny_floor() {
        assert!(fused_bluestein_supported_by_limits(
            4056,
            AxisPrecision::F32,
            48 * 1024,
            256,
            256
        ));
        assert!(!fused_bluestein_supported_by_limits(
            4056,
            AxisPrecision::F32,
            16 * 1024,
            256,
            256
        ));
        assert!(fused_bluestein_supported_by_limits(
            2016,
            AxisPrecision::F32,
            16 * 1024,
            256,
            256
        ));
        assert!(!fused_bluestein_supported_by_limits(
            6561,
            AxisPrecision::F32,
            48 * 1024,
            256,
            256
        ));
        assert!(!fused_bluestein_supported_by_limits(
            70,
            AxisPrecision::F32,
            48 * 1024,
            256,
            256
        ));
        assert!(!fused_bluestein_supported_by_limits(
            4056,
            AxisPrecision::F32,
            48 * 1024,
            256,
            255
        ));
    }

    #[test]
    fn fused_bluestein_wgsl_inlines_both_ffts_and_chirp_steps() {
        let factors = crate::runtime::factor_supported_length(441).unwrap();
        let key = FusedPrimeStageKey::new(
            FusedPrimeKind::Bluestein,
            2,
            0,
            &[221, 3],
            221,
            1,
            441,
            &factors,
            FftDirection::Inverse,
            FUSED_WORKGROUP_SIZE,
            true,
            1.0 / 663.0,
            AxisPrecision::F32,
        );
        let wgsl = generate_fused_bluestein_wgsl_for_key(&key);
        assert!(wgsl.contains("var<workgroup> scratch: array<vec2<f32>, 441>"));
        assert!(wgsl.contains("c_mul(input[base + t * STRIDE], chirp[t])"));
        assert!(wgsl.contains("scratch[t] = c_mul(scratch[t], bfft[t])"));
        assert!(wgsl.contains("c_mul(convolution, chirp[t])"));
        assert!(wgsl.contains("fn twiddle_forward"));
        assert!(wgsl.contains("fn twiddle_inverse"));
        assert!(wgsl.contains("const INVERSE_M: f32"));
        assert!(wgsl.contains("let wgFlat: u32"));
        assert!(!wgsl.contains("sin("));
        assert!(!wgsl.contains("cos("));
    }

    #[test]
    fn bluestein_chirp_has_unit_first_value_and_conjugate_kernel() {
        let chirp = bluestein_chirp(14, FftDirection::Forward);
        assert!((chirp[0].re - 1.0).abs() < 1.0e-6);
        assert!(chirp[0].im.abs() < 1.0e-6);

        let bfft_input = round_complex64(bluestein_phase(
            14,
            1,
            -transform_sign(FftDirection::Forward),
        ));
        assert!((bfft_input.re - chirp[1].re).abs() < 1.0e-6);
        assert!((bfft_input.im + chirp[1].im).abs() < 1.0e-6);
    }

    #[test]
    fn bluestein_chirp_reduces_large_square_before_f64_phase() {
        const N: usize = 1_000_003;
        const I: usize = 999_999;
        const SQUARE_MOD_2N: u128 = 1_000_019;

        let actual = bluestein_phase(N, I, transform_sign(FftDirection::Forward));
        let expected_angle = -std::f64::consts::PI * SQUARE_MOD_2N as f64 / N as f64;
        let (expected_sin, expected_cos) = expected_angle.sin_cos();
        assert_eq!(actual.re.to_bits(), expected_cos.to_bits());
        assert_eq!(actual.im.to_bits(), expected_sin.to_bits());
        let inverse = bluestein_phase(N, I, transform_sign(FftDirection::Inverse));
        assert_eq!(inverse.re.to_bits(), actual.re.to_bits());
        assert_eq!(inverse.im.to_bits(), (-actual.im).to_bits());

        let old_square = I as u64 * I as u64;
        let old_angle = -std::f32::consts::PI * old_square as f32 / N as f32;
        let (old_sin, old_cos) = old_angle.sin_cos();
        let old_error =
            ((old_cos as f64 - actual.re).powi(2) + (old_sin as f64 - actual.im).powi(2)).sqrt();
        assert!(
            old_error > 0.25,
            "legacy unreduced f32 phase unexpectedly close: error={old_error}"
        );
    }

    #[test]
    fn bluestein_bfft_is_f64_generated_then_rounded_once() {
        let n = 14;
        let m = 27;
        for direction in [FftDirection::Forward, FftDirection::Inverse] {
            let actual = bluestein_bfft(n, m, direction).unwrap();

            let mut kernel = vec![Complex64::default(); m];
            kernel[0] = Complex64::new(1.0, 0.0);
            for i in 1..n {
                let value = bluestein_phase(n, i, -transform_sign(direction));
                kernel[i] = value;
                kernel[m - i] = value;
            }
            let expected = reference_c2c_nd_f64(
                &kernel,
                &crate::config::FftConfig::new(m).with_normalization(Normalization::None),
            )
            .unwrap();

            for (actual, expected) in actual.iter().zip(expected) {
                assert_eq!(actual.re.to_bits(), (expected.re as f32).to_bits());
                assert_eq!(actual.im.to_bits(), (expected.im as f32).to_bits());
            }
        }
    }

    #[test]
    fn generated_bluestein_wgsl_contains_nd_constants() {
        let key = BluesteinStageKey::new(
            BluesteinKernelKind::Pack,
            2,
            1,
            &[4, 14],
            14,
            4,
            27,
            WORKGROUP_SIZE,
            false,
            1.0,
            AxisPrecision::F32,
        );
        let wgsl = generate_bluestein_pack_wgsl(&key);
        assert!(wgsl.contains("const N: u32 = 14u;"));
        assert!(wgsl.contains("const M: u32 = 27u;"));
        assert!(wgsl.contains("const STRIDE: u32 = 4u;"));
        assert!(wgsl.contains("let lines_per_batch: u32 = 4u;"));
    }
}
