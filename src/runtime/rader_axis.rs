use bytemuck::{Pod, Zeroable};

use crate::config::{FftDirection, Normalization};
use crate::error::{FftError, Result};
use crate::math::{fft_f64, Complex32, Complex64, ComplexDoubleFloat, DoubleFloat};
use crate::runtime::axis_plan::{
    fused_lines_per_workgroup, fused_smooth_factors, fused_smooth_pads_indices,
    generate_fused_scratch_fft_stages_wgsl, generate_in_place_smooth_fft_stage_multiline_wgsl,
    multiline_line_stride, AxisLayout, AxisPlan, AxisPlanConfig, AxisPrecision, AxisStageKind,
    AxisTwiddleLutPool, LongAxisRoute, FUSED_PRIME_RADICES,
};
use crate::runtime::axis_policy::{
    is_prime, mod_pow, next_power_of_two_at_least, next_smooth_at_least, primitive_root_prime,
};
use crate::runtime::bluestein_axis::{
    register_bluestein_plan, short_register_convolution, BluesteinAxis, BluesteinAxisConfig,
};
use crate::runtime::buffer_view::BufferView;
use crate::runtime::direct_prime::{
    direct_lines_per_workgroup, direct_pairs_per_invocation, direct_prime_supported,
    generate_direct_prime_wgsl_for_key,
};
use crate::runtime::dispatch::{max_workgroups_per_dimension, split_workgroups};
use crate::runtime::large_graph::{ElementFormat, HelperBufferRange};
use crate::runtime::nd_wgsl::{
    format_wgsl_f32, format_wgsl_f32_roundtrip, lines_per_batch, product, stride_for_axis,
    wgsl_line_base_fn,
};
use crate::runtime::pipeline_cache::{
    with_device_pipeline_cache, ComputePipelineCacheKey, FusedPrimeKind, FusedPrimeStageKey,
    PipelineLayoutCacheKey, RaderKernelKind, RaderStageKey, ShaderCacheKey,
};
use crate::runtime::recorder::CommandRecorder;
use crate::runtime::twiddle::create_twiddle_lut_buffer_for_len_with_precision;
use crate::runtime::window_scheduler::WindowScheduler;

// At smaller convolution lengths the 256-lane fused kernel is mostly idle and
// the existing staged path is already inexpensive.
#[cfg(test)]
const DEFAULT_WORKGROUP_SIZE: u32 = 64;
#[cfg(test)]
const DEFAULT_FUSED_WORKGROUP_SIZE: u32 = 256;
#[cfg(test)]
const DEFAULT_FUSED_MIN_CONVOLUTION_LENGTH: usize = 128;
#[cfg(test)]
const WORKGROUP_SIZE: u32 = DEFAULT_WORKGROUP_SIZE;
#[cfg(test)]
const FUSED_WORKGROUP_SIZE: u32 = DEFAULT_FUSED_WORKGROUP_SIZE;

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct RaderLinesParams {
    lines: u32,
    line_offset: u32,
    _pad0: u32,
    _pad1: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct RaderTotalParams {
    total: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RaderAxisConfig {
    pub(crate) shape: Vec<usize>,
    pub(crate) axis: usize,
    pub(crate) batch: usize,
    pub(crate) direction: FftDirection,
    pub(crate) normalization: Normalization,
    pub(crate) precision: AxisPrecision,
    pub(crate) workgroup_size: u32,
    pub(crate) fused_workgroup_size: u32,
    pub(crate) fused_min_convolution_length: usize,
    pub(crate) fuse_long_axes: bool,
    pub(crate) direct_max_prime: usize,
    /// Whether a prime whose convolution does not fit one workgroup may run
    /// Bluestein's register-resident kernel; false when Rader is forced.
    pub(crate) bluestein_fallback: bool,
}

pub(crate) struct RaderAxis {
    n: usize,
    m: usize,
    lines: u32,
    precision: AxisPrecision,
    lines_params_buffer: wgpu::Buffer,
    perm_buffer: wgpu::Buffer,
    bfft_buffer: wgpu::Buffer,
    execution: RaderExecution,
}

enum RaderExecution {
    /// A direct DFT of a short prime (see `runtime::direct_prime`).
    Direct(FusedRaderExecution),
    Fused(FusedRaderExecution),
    /// Bluestein's register-resident kernel, for primes whose Rader
    /// convolution fits no workgroup: one coalesced pass instead of the
    /// multi-pass pipeline.
    Bluestein(Box<BluesteinAxis>),
    MultiPass(Box<MultiPassRaderExecution>),
}

struct FusedRaderExecution {
    workgroups: u32,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    twiddle_buffer: wgpu::Buffer,
}

struct MultiPassRaderExecution {
    workgroups_sum: u32,
    workgroups_lines: u32,
    workgroups_work: u32,
    workgroups_tail: u32,
    sum_pipeline: wgpu::ComputePipeline,
    sum_bind_group_layout: wgpu::BindGroupLayout,
    pack_pipeline: wgpu::ComputePipeline,
    pack_bind_group_layout: wgpu::BindGroupLayout,
    mul_pipeline: wgpu::ComputePipeline,
    mul_bind_group_layout: wgpu::BindGroupLayout,
    write_y0_pipeline: wgpu::ComputePipeline,
    write_y0_bind_group_layout: wgpu::BindGroupLayout,
    post_pipeline: wgpu::ComputePipeline,
    post_bind_group_layout: wgpu::BindGroupLayout,
    total_params_buffer: wgpu::Buffer,
    sum_buffer: wgpu::Buffer,
    x0_buffer: wgpu::Buffer,
    work_buffer: wgpu::Buffer,
    fft_buffer: wgpu::Buffer,
    work_fft_forward: AxisPlan,
    work_fft_inverse: AxisPlan,
}

impl RaderAxisConfig {
    pub(crate) fn validate(&self) -> Result<()> {
        if self.shape.is_empty() || self.shape.contains(&0) {
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
        if self.workgroup_size == 0 || !self.workgroup_size.is_power_of_two() {
            return Err(FftError::LargeGraphStageUnsupported {
                stage: "rader-tuning",
                reason: "staged workgroup size must be a nonzero power of two",
            });
        }
        if self.fused_workgroup_size == 0 {
            return Err(FftError::LargeGraphStageUnsupported {
                stage: "rader-tuning",
                reason: "fused workgroup size must be nonzero",
            });
        }
        let n = self.shape[self.axis];
        if !is_prime(n) {
            return Err(FftError::UnsupportedLength { len: n });
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
            AxisPrecision::F64 | AxisPrecision::Df64 => {
                let total = product(&self.shape) as f64;
                match (self.direction, self.normalization) {
                    (_, Normalization::None) => 1.0,
                    (FftDirection::Forward, Normalization::Forward) => 1.0 / total,
                    (FftDirection::Inverse, Normalization::Inverse) => 1.0 / total,
                    (_, Normalization::Orthogonal) => 1.0 / total.sqrt(),
                    _ => 1.0,
                }
            }
        })
    }
}

impl RaderAxis {
    pub(crate) fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        config: RaderAxisConfig,
    ) -> Result<Self> {
        config.validate()?;
        let n = config.shape[config.axis];
        let l = n - 1;
        let (m, medium_prime_schedule) = rader_axis_convolution(
            n,
            config.precision,
            config.fused_workgroup_size,
            config.fused_min_convolution_length,
            &device.limits(),
        )?;
        let factors = match medium_prime_schedule {
            Some(schedule) => schedule,
            None => crate::runtime::factor_supported_length(m)?,
        };

        let lines = checked_mul(config.batch, lines_per_batch(&config.shape, config.axis))?;
        let lines_u32 = lines as u32;
        let stride_complex = stride_for_axis(&config.shape, config.axis);
        let scale = config.scale()?;
        let apply_scale = scale != 1.0;
        let perm = rader_permutation(n)?;
        let bfft = rader_bfft_f64(n, m, config.direction, &perm)?;
        let complex_bytes = config.precision.complex_size_bytes();
        validate_rader_staged_workgroup(&config, &device.limits())?;

        let lines_params_buffer =
            uniform_buffer::<RaderLinesParams>(device, "wgpu_fft.rader.lines_params");
        queue.write_buffer(
            &lines_params_buffer,
            0,
            bytemuck::bytes_of(&RaderLinesParams {
                lines: lines_u32,
                line_offset: 0,
                _pad0: 0,
                _pad1: 0,
            }),
        );

        // The fused kernel also reads, after the permutation, each natural
        // element's slot in the convolution input.
        let perm_table = rader_permutation_table(&perm);
        let perm_buffer = storage_buffer(
            device,
            "wgpu_fft.rader.perm",
            (perm_table.len() * std::mem::size_of::<u32>()) as u64,
            wgpu::BufferUsages::COPY_DST,
        )?;
        queue.write_buffer(&perm_buffer, 0, bytemuck::cast_slice(&perm_table));

        let bfft_buffer = storage_buffer(
            device,
            "wgpu_fft.rader.bfft",
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
                let values = bfft
                    .iter()
                    .map(|value| ComplexDoubleFloat::from_f64(value.re, value.im))
                    .collect::<Vec<_>>();
                queue.write_buffer(&bfft_buffer, 0, bytemuck::cast_slice(&values));
            }
        }

        // A cyclic convolution in the fused Rader kernel, several lines per
        // workgroup, beats the direct kernel's O(n^2) sums; the direct kernel
        // keeps the primes whose convolution would be linear.
        let cyclic_fused = m == n - 1
            && fused_rader_supported(
                n,
                m,
                config.precision,
                config.fused_workgroup_size,
                config.fused_min_convolution_length,
                &device.limits(),
            );
        let execution = if n <= config.direct_max_prime
            && !cyclic_fused
            && direct_prime_supported(
                n,
                config.precision,
                config.fused_workgroup_size,
                &device.limits(),
            ) {
            let twiddle_buffer = create_twiddle_lut_buffer_for_len_with_precision(
                device,
                queue,
                "wgpu_fft.rader.direct.twiddle_lut",
                n,
                config.precision.as_fft_precision(),
            )?;
            let pairs_per_invocation = direct_pairs_per_invocation(n, lines_u32 as usize);
            let lines_per_workgroup = direct_lines_per_workgroup(
                n,
                stride_complex,
                config.precision,
                config.fused_workgroup_size,
                pairs_per_invocation,
                lines_u32 as usize,
                u64::from(device.limits().max_compute_workgroup_storage_size),
            );
            let shader_key = FusedPrimeStageKey::new(
                FusedPrimeKind::Direct,
                config.shape.len(),
                config.axis,
                &config.shape,
                n,
                stride_complex,
                n,
                &[n],
                config.direction,
                config.fused_workgroup_size,
                apply_scale,
                scale,
                config.precision,
            )
            .with_lines_per_workgroup(lines_per_workgroup)
            .with_pairs_per_invocation(pairs_per_invocation as u32);
            let pipeline_key = ComputePipelineCacheKey::fused_prime_stage(shader_key.clone());
            let bind_group_layout = cached_layout(device, pipeline_key.layout);
            let pipeline = cached_fused_pipeline(device, &pipeline_key, &shader_key);
            RaderExecution::Direct(FusedRaderExecution {
                workgroups: lines_u32.div_ceil(lines_per_workgroup),
                pipeline,
                bind_group_layout,
                twiddle_buffer,
            })
        } else if config.bluestein_fallback
            && register_bluestein_plan(
                n,
                config.precision,
                config.fused_workgroup_size,
                config.fused_min_convolution_length,
                config.fuse_long_axes,
                &device.limits(),
            )
            .is_some_and(|(register_length, _, _)| {
                rader_prefers_register_bluestein(
                    n,
                    m,
                    register_length,
                    fused_rader_supported(
                        n,
                        m,
                        config.precision,
                        config.fused_workgroup_size,
                        config.fused_min_convolution_length,
                        &device.limits(),
                    ),
                )
            })
        {
            RaderExecution::Bluestein(Box::new(BluesteinAxis::new(
                device,
                queue,
                BluesteinAxisConfig {
                    shape: config.shape.clone(),
                    axis: config.axis,
                    batch: config.batch,
                    direction: config.direction,
                    normalization: config.normalization,
                    precision: config.precision,
                    workgroup_size: config.workgroup_size,
                    fused_workgroup_size: config.fused_workgroup_size,
                    fused_min_convolution_length: config.fused_min_convolution_length,
                    fuse_long_axes: config.fuse_long_axes,
                },
            )?))
        } else if fused_rader_supported(
            n,
            m,
            config.precision,
            config.fused_workgroup_size,
            config.fused_min_convolution_length,
            &device.limits(),
        ) {
            let twiddle_buffer = create_twiddle_lut_buffer_for_len_with_precision(
                device,
                queue,
                "wgpu_fft.rader.fused.twiddle_lut",
                m,
                config.precision.as_fft_precision(),
            )?;
            let rader_key = |factors: &[usize]| {
                FusedPrimeStageKey::new(
                    FusedPrimeKind::Rader,
                    config.shape.len(),
                    config.axis,
                    &config.shape,
                    n,
                    stride_complex,
                    m,
                    factors,
                    config.direction,
                    config.fused_workgroup_size,
                    apply_scale,
                    scale,
                    config.precision,
                )
            };
            // Several lines per workgroup keep its invocations busy on short
            // convolutions, as in the fused axis kernels. Fewer than
            // MIN_RADER_LINES measured slower than one line per workgroup.
            let lines_per_workgroup = fused_lines_per_workgroup(
                m,
                stride_complex,
                config.precision,
                lines_u32 as usize,
                u64::from(device.limits().max_compute_workgroup_storage_size),
            )
            .min(config.fused_workgroup_size);
            let lines_per_workgroup = if lines_per_workgroup < MIN_RADER_LINES {
                1
            } else {
                lines_per_workgroup
            };
            let element_major = lines_per_workgroup > 1 && stride_complex != 1;
            let rader_key = |factors: &[usize]| {
                rader_key(factors).with_lines_per_workgroup(lines_per_workgroup)
            };
            // The convolution runs the fused smooth schedule; one whose first
            // stage needs padded indices keeps the multi-pass factors where
            // the padding does not fit.
            let schedule = fused_smooth_factors(m, &factors);
            let shader_key = if fused_smooth_pads_indices(&schedule, element_major) {
                let padded = rader_key(&schedule).with_padded_indices();
                if padded.supported_by_device_limits(&device.limits()) {
                    padded
                } else {
                    rader_key(&factors)
                }
            } else {
                rader_key(&schedule)
            };
            // Storage that holds one line may not hold several.
            let shader_key = if shader_key.supported_by_device_limits(&device.limits()) {
                shader_key
            } else {
                shader_key.with_lines_per_workgroup(1)
            };
            // One strided line per workgroup loads one element per sector:
            // serial lines load and store together.
            let serial_lines = serial_rader_lines(m, lines);
            let shader_key = if shader_key.lines_per_workgroup == 1
                && stride_complex != 1
                && config.precision == AxisPrecision::F32
                && serial_lines > 1
            {
                let serial = shader_key.clone().with_serial_lines(serial_lines as u32);
                if serial.supported_by_device_limits(&device.limits()) {
                    serial
                } else {
                    shader_key
                }
            } else {
                shader_key
            };
            let lines_per_workgroup = shader_key.lines_per_workgroup.max(shader_key.serial_lines);
            let pipeline_key = ComputePipelineCacheKey::fused_prime_stage(shader_key.clone());
            let bind_group_layout = cached_layout(device, pipeline_key.layout);
            let pipeline = cached_fused_pipeline(device, &pipeline_key, &shader_key);
            RaderExecution::Fused(FusedRaderExecution {
                workgroups: lines_u32.div_ceil(lines_per_workgroup),
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
                uniform_buffer::<RaderTotalParams>(device, "wgpu_fft.rader.total_params");
            queue.write_buffer(
                &total_params_buffer,
                0,
                bytemuck::bytes_of(&RaderTotalParams {
                    total: total_work_u32,
                    _pad0: 0,
                    _pad1: 0,
                    _pad2: 0,
                }),
            );

            let line_bytes = lines as u64 * complex_bytes;
            let work_bytes = work_complex as u64 * complex_bytes;
            let sum_buffer = storage_buffer(
                device,
                "wgpu_fft.rader.sum",
                line_bytes,
                wgpu::BufferUsages::empty(),
            )?;
            let x0_buffer = storage_buffer(
                device,
                "wgpu_fft.rader.x0",
                line_bytes,
                wgpu::BufferUsages::empty(),
            )?;
            let work_buffer = storage_buffer(
                device,
                "wgpu_fft.rader.work",
                work_bytes,
                wgpu::BufferUsages::empty(),
            )?;
            let fft_buffer = storage_buffer(
                device,
                "wgpu_fft.rader.fft",
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
                    workgroup_size: config.workgroup_size,
                    fused_workgroup_size: config.fused_workgroup_size,
                    long_axes: LongAxisRoute::new(config.fuse_long_axes),
                    small_volumes: false,
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
                    workgroup_size: config.workgroup_size,
                    fused_workgroup_size: config.fused_workgroup_size,
                    long_axes: LongAxisRoute::new(config.fuse_long_axes),
                    small_volumes: false,
                },
                &mut twiddle_lut_pool,
            )?;

            let keys = RaderPipelineKeys::new(
                config.shape.len(),
                config.axis,
                &config.shape,
                n,
                stride_complex,
                m,
                apply_scale,
                scale,
                config.precision,
                config.workgroup_size,
            );
            let sum_bind_group_layout = cached_layout(device, keys.sum.layout);
            let pack_bind_group_layout = cached_layout(device, keys.pack.layout);
            let mul_bind_group_layout = cached_layout(device, keys.mul.layout);
            let write_y0_bind_group_layout = cached_layout(device, keys.write_y0.layout);
            let post_bind_group_layout = cached_layout(device, keys.post.layout);
            let sum_pipeline = cached_pipeline(device, &keys.sum)?;
            let pack_pipeline = cached_pipeline(device, &keys.pack)?;
            let mul_pipeline = cached_pipeline(device, &keys.mul)?;
            let write_y0_pipeline = cached_pipeline(device, &keys.write_y0)?;
            let post_pipeline = cached_pipeline(device, &keys.post)?;

            RaderExecution::MultiPass(Box::new(MultiPassRaderExecution {
                workgroups_sum: lines_u32,
                workgroups_lines: lines_u32.div_ceil(config.workgroup_size),
                workgroups_work: total_work_u32.div_ceil(config.workgroup_size),
                workgroups_tail: (lines_u32 * l as u32).div_ceil(config.workgroup_size),
                sum_pipeline,
                sum_bind_group_layout,
                pack_pipeline,
                pack_bind_group_layout,
                mul_pipeline,
                mul_bind_group_layout,
                write_y0_pipeline,
                write_y0_bind_group_layout,
                post_pipeline,
                post_bind_group_layout,
                total_params_buffer,
                sum_buffer,
                x0_buffer,
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
            perm_buffer,
            bfft_buffer,
            execution,
        })
    }

    pub(crate) fn workspace_size_bytes(&self) -> u64 {
        0
    }

    pub(crate) fn twiddle_lut_storage_bytes(&self) -> u64 {
        match &self.execution {
            RaderExecution::Direct(execution) | RaderExecution::Fused(execution) => {
                execution.twiddle_buffer.size()
            }
            RaderExecution::Bluestein(plan) => plan.twiddle_lut_storage_bytes(),
            RaderExecution::MultiPass(execution) => {
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
        matches!(
            self.execution,
            RaderExecution::Direct(_) | RaderExecution::Fused(_) | RaderExecution::Bluestein(_)
        )
    }

    /// Graph label of the single kernel of a fused execution.
    pub(crate) fn graph_fused_label(&self) -> &'static str {
        match &self.execution {
            RaderExecution::Direct(_) => "rader-direct-dft-stage",
            RaderExecution::Bluestein(plan) => plan.graph_fused_label(),
            _ => "rader-fused-workgroup-stage",
        }
    }

    pub(crate) fn graph_forward_fft_stage_kinds(&self) -> Vec<AxisStageKind> {
        match &self.execution {
            RaderExecution::Direct(_) | RaderExecution::Fused(_) | RaderExecution::Bluestein(_) => {
                Vec::new()
            }
            RaderExecution::MultiPass(execution) => execution.work_fft_forward.graph_stage_kinds(),
        }
    }

    pub(crate) fn graph_forward_fft_workspace_bytes(&self) -> u64 {
        match &self.execution {
            RaderExecution::Direct(_) | RaderExecution::Fused(_) | RaderExecution::Bluestein(_) => {
                0
            }
            RaderExecution::MultiPass(execution) => {
                execution.work_fft_forward.workspace_size_bytes()
            }
        }
    }

    pub(crate) fn graph_inverse_fft_stage_kinds(&self) -> Vec<AxisStageKind> {
        match &self.execution {
            RaderExecution::Direct(_) | RaderExecution::Fused(_) | RaderExecution::Bluestein(_) => {
                Vec::new()
            }
            RaderExecution::MultiPass(execution) => execution.work_fft_inverse.graph_stage_kinds(),
        }
    }

    pub(crate) fn graph_inverse_fft_workspace_bytes(&self) -> u64 {
        match &self.execution {
            RaderExecution::Direct(_) | RaderExecution::Fused(_) | RaderExecution::Bluestein(_) => {
                0
            }
            RaderExecution::MultiPass(execution) => {
                execution.work_fft_inverse.workspace_size_bytes()
            }
        }
    }

    pub(crate) fn graph_helper_buffers(&self) -> Vec<HelperBufferRange> {
        if let RaderExecution::Bluestein(plan) = &self.execution {
            return plan.graph_helper_buffers();
        }
        let element_format = self.precision.element_format();
        let mut helpers = vec![
            HelperBufferRange {
                label: "rader-permutation-helper",
                index: 0,
                size_bytes: self.perm_buffer.size(),
                format: ElementFormat::U32,
            },
            HelperBufferRange {
                label: "rader-bfft-helper",
                index: 1,
                size_bytes: self.bfft_buffer.size(),
                format: element_format,
            },
        ];
        if let RaderExecution::MultiPass(execution) = &self.execution {
            helpers.extend([
                HelperBufferRange {
                    label: "rader-sum-helper",
                    index: 2,
                    size_bytes: execution.sum_buffer.size(),
                    format: element_format,
                },
                HelperBufferRange {
                    label: "rader-x0-helper",
                    index: 3,
                    size_bytes: execution.x0_buffer.size(),
                    format: element_format,
                },
                HelperBufferRange {
                    label: "rader-work-helper",
                    index: 4,
                    size_bytes: execution.work_buffer.size(),
                    format: element_format,
                },
                HelperBufferRange {
                    label: "rader-fft-helper",
                    index: 5,
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
        encoder: &mut CommandRecorder<'_>,
        input: BufferView<'_>,
        output: BufferView<'_>,
    ) -> Result<()> {
        debug_assert!(self.n > 1);
        // A cyclic convolution over `n - 1` points, or a linear one.
        debug_assert!(self.m == self.n - 1 || self.m >= 2 * (self.n - 1) - 1);
        debug_assert!(self.lines > 0);

        match &self.execution {
            RaderExecution::Direct(execution) | RaderExecution::Fused(execution) => {
                self.dispatch_fused(device, encoder, input, output, execution)?;
            }
            RaderExecution::Bluestein(plan) => {
                plan.execute_views(device, encoder, input, output)?
            }
            RaderExecution::MultiPass(execution) => {
                self.dispatch_sum(device, encoder, input.clone(), execution)?;
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
                self.dispatch_write_y0(device, encoder, output.clone(), execution)?;
                self.dispatch_post(device, encoder, output, execution)?;
            }
        }
        Ok(())
    }

    fn dispatch_fused(
        &self,
        device: &wgpu::Device,
        encoder: &mut CommandRecorder<'_>,
        input: BufferView<'_>,
        output: BufferView<'_>,
        execution: &FusedRaderExecution,
    ) -> Result<()> {
        let scheduler = WindowScheduler::for_device(device);
        let element_format = self.precision.element_format();
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("wgpu_fft.rader.fused.bind_group"),
            layout: &execution.bind_group_layout,
            entries: &[
                bind_view_entry(&scheduler, 0, input, element_format)?,
                bind_view_entry(&scheduler, 1, output, element_format)?,
                bind_storage_entry(&scheduler, 2, &self.perm_buffer, ElementFormat::U32)?,
                bind_storage_entry(&scheduler, 3, &self.bfft_buffer, element_format)?,
                bind_storage_entry(&scheduler, 4, &execution.twiddle_buffer, element_format)?,
                bind_uniform_entry(5, &self.lines_params_buffer),
            ],
        });
        let pass = encoder.pass();
        pass.set_pipeline(&execution.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let (x, y, z) =
            split_workgroups(execution.workgroups, max_workgroups_per_dimension(device))?;
        pass.dispatch_workgroups(x, y, z);
        Ok(())
    }

    fn dispatch_sum(
        &self,
        device: &wgpu::Device,
        encoder: &mut CommandRecorder<'_>,
        input: BufferView<'_>,
        execution: &MultiPassRaderExecution,
    ) -> Result<()> {
        let scheduler = WindowScheduler::for_device(device);
        let element_format = self.precision.element_format();
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("wgpu_fft.rader.sum.bind_group"),
            layout: &execution.sum_bind_group_layout,
            entries: &[
                bind_view_entry(&scheduler, 0, input, element_format)?,
                bind_storage_entry(&scheduler, 1, &execution.sum_buffer, element_format)?,
                bind_storage_entry(&scheduler, 2, &execution.x0_buffer, element_format)?,
                bind_uniform_entry(3, &self.lines_params_buffer),
            ],
        });
        let pass = encoder.pass();
        pass.set_pipeline(&execution.sum_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let (x, y, z) = split_workgroups(
            execution.workgroups_sum,
            max_workgroups_per_dimension(device),
        )?;
        pass.dispatch_workgroups(x, y, z);
        Ok(())
    }

    fn dispatch_pack(
        &self,
        device: &wgpu::Device,
        encoder: &mut CommandRecorder<'_>,
        input: BufferView<'_>,
        execution: &MultiPassRaderExecution,
    ) -> Result<()> {
        let scheduler = WindowScheduler::for_device(device);
        let element_format = self.precision.element_format();
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("wgpu_fft.rader.pack.bind_group"),
            layout: &execution.pack_bind_group_layout,
            entries: &[
                bind_view_entry(&scheduler, 0, input, element_format)?,
                bind_storage_entry(&scheduler, 1, &execution.work_buffer, element_format)?,
                bind_storage_entry(&scheduler, 2, &self.perm_buffer, ElementFormat::U32)?,
                bind_uniform_entry(3, &self.lines_params_buffer),
            ],
        });
        let pass = encoder.pass();
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
        encoder: &mut CommandRecorder<'_>,
        execution: &MultiPassRaderExecution,
    ) -> Result<()> {
        let scheduler = WindowScheduler::for_device(device);
        let element_format = self.precision.element_format();
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("wgpu_fft.rader.mul.bind_group"),
            layout: &execution.mul_bind_group_layout,
            entries: &[
                bind_storage_entry(&scheduler, 0, &execution.fft_buffer, element_format)?,
                bind_storage_entry(&scheduler, 1, &self.bfft_buffer, element_format)?,
                bind_uniform_entry(2, &execution.total_params_buffer),
            ],
        });
        let pass = encoder.pass();
        pass.set_pipeline(&execution.mul_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let (x, y, z) = split_workgroups(
            execution.workgroups_work,
            max_workgroups_per_dimension(device),
        )?;
        pass.dispatch_workgroups(x, y, z);
        Ok(())
    }

    fn dispatch_write_y0(
        &self,
        device: &wgpu::Device,
        encoder: &mut CommandRecorder<'_>,
        output: BufferView<'_>,
        execution: &MultiPassRaderExecution,
    ) -> Result<()> {
        let scheduler = WindowScheduler::for_device(device);
        let element_format = self.precision.element_format();
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("wgpu_fft.rader.write_y0.bind_group"),
            layout: &execution.write_y0_bind_group_layout,
            entries: &[
                bind_storage_entry(&scheduler, 0, &execution.sum_buffer, element_format)?,
                bind_view_entry(&scheduler, 1, output, element_format)?,
                bind_uniform_entry(2, &self.lines_params_buffer),
            ],
        });
        let pass = encoder.pass();
        pass.set_pipeline(&execution.write_y0_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let (x, y, z) = split_workgroups(
            execution.workgroups_lines,
            max_workgroups_per_dimension(device),
        )?;
        pass.dispatch_workgroups(x, y, z);
        Ok(())
    }

    fn dispatch_post(
        &self,
        device: &wgpu::Device,
        encoder: &mut CommandRecorder<'_>,
        output: BufferView<'_>,
        execution: &MultiPassRaderExecution,
    ) -> Result<()> {
        let scheduler = WindowScheduler::for_device(device);
        let element_format = self.precision.element_format();
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("wgpu_fft.rader.post.bind_group"),
            layout: &execution.post_bind_group_layout,
            entries: &[
                bind_storage_entry(&scheduler, 0, &execution.work_buffer, element_format)?,
                bind_storage_entry(&scheduler, 1, &execution.x0_buffer, element_format)?,
                bind_storage_entry(&scheduler, 2, &self.perm_buffer, ElementFormat::U32)?,
                bind_view_entry(&scheduler, 3, output, element_format)?,
                bind_uniform_entry(4, &self.lines_params_buffer),
            ],
        });
        let pass = encoder.pass();
        pass.set_pipeline(&execution.post_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let (x, y, z) = split_workgroups(
            execution.workgroups_tail,
            max_workgroups_per_dimension(device),
        )?;
        pass.dispatch_workgroups(x, y, z);
        Ok(())
    }
}

struct RaderPipelineKeys {
    sum: ComputePipelineCacheKey,
    pack: ComputePipelineCacheKey,
    mul: ComputePipelineCacheKey,
    write_y0: ComputePipelineCacheKey,
    post: ComputePipelineCacheKey,
}

impl RaderPipelineKeys {
    #[allow(clippy::too_many_arguments)]
    fn new(
        rank: usize,
        axis: usize,
        shape: &[usize],
        n: usize,
        stride_complex: usize,
        m: usize,
        apply_scale: bool,
        scale: f64,
        precision: AxisPrecision,
        workgroup_size: u32,
    ) -> Self {
        let key = |kind| {
            RaderStageKey::new(
                kind,
                rank,
                axis,
                shape,
                n,
                stride_complex,
                m,
                workgroup_size,
                apply_scale,
                scale,
                precision,
            )
        };
        Self {
            sum: ComputePipelineCacheKey::rader_stage(key(RaderKernelKind::Sum)),
            pack: ComputePipelineCacheKey::rader_stage(key(RaderKernelKind::Pack)),
            mul: ComputePipelineCacheKey::rader_stage(key(RaderKernelKind::Mul)),
            write_y0: ComputePipelineCacheKey::rader_stage(key(RaderKernelKind::WriteY0)),
            post: ComputePipelineCacheKey::rader_stage(key(RaderKernelKind::Post)),
        }
    }
}

fn cached_layout(device: &wgpu::Device, key: PipelineLayoutCacheKey) -> wgpu::BindGroupLayout {
    with_device_pipeline_cache(device, |cache| cache.get_bind_group_layout(device, key))
}

fn rader_stage_key(key: &ComputePipelineCacheKey) -> Result<&RaderStageKey> {
    match &key.shader {
        ShaderCacheKey::RaderStage(stage) => Ok(stage),
        _ => Err(FftError::LargeGraphStageUnsupported {
            stage: "rader-pipeline-key",
            reason: "Rader pipeline key has a non-Rader shader stage",
        }),
    }
}

fn cached_pipeline(
    device: &wgpu::Device,
    key: &ComputePipelineCacheKey,
) -> Result<wgpu::ComputePipeline> {
    let stage_key = rader_stage_key(key)?;
    let stable_key = key.stable_key();
    let pipeline_label = format!("wgpu_fft.rader.pipeline.{stable_key}");
    let shader_label = format!("wgpu_fft.rader.shader.{stable_key}");
    Ok(with_device_pipeline_cache(device, |cache| {
        cache.get_compute_pipeline(device, key, &pipeline_label, &shader_label, || {
            generate_rader_wgsl_for_key(stage_key)
        })
    }))
}

fn cached_fused_pipeline(
    device: &wgpu::Device,
    key: &ComputePipelineCacheKey,
    shader_key: &FusedPrimeStageKey,
) -> wgpu::ComputePipeline {
    let stable_key = key.stable_key();
    let pipeline_label = format!("wgpu_fft.rader.fused.pipeline.{stable_key}");
    let shader_label = format!("wgpu_fft.rader.fused.shader.{stable_key}");
    with_device_pipeline_cache(device, |cache| {
        cache.get_compute_pipeline(device, key, &pipeline_label, &shader_label, || {
            if shader_key.kind == FusedPrimeKind::Direct {
                generate_direct_prime_wgsl_for_key(shader_key)
            } else {
                generate_fused_rader_wgsl_for_key(shader_key)
            }
        })
    })
}

fn validate_rader_staged_workgroup(config: &RaderAxisConfig, limits: &wgpu::Limits) -> Result<()> {
    if config.workgroup_size > limits.max_compute_invocations_per_workgroup
        || config.workgroup_size > limits.max_compute_workgroup_size_x
    {
        return Err(FftError::LargeGraphStageUnsupported {
            stage: "rader-staged-workgroup",
            reason: "configured workgroup size exceeds the active device compute limits",
        });
    }
    let scratch_bytes = u64::from(config.workgroup_size)
        .checked_mul(config.precision.complex_size_bytes())
        .ok_or(FftError::LengthTooLarge {
            len: config.workgroup_size as usize,
        })?;
    if scratch_bytes > u64::from(limits.max_compute_workgroup_storage_size) {
        return Err(FftError::LargeGraphStageUnsupported {
            stage: "rader-staged-workgroup",
            reason: "configured Rader sum scratch exceeds workgroup storage",
        });
    }
    Ok(())
}

fn fused_rader_supported(
    n: usize,
    m: usize,
    precision: AxisPrecision,
    workgroup_size: u32,
    min_convolution_length: usize,
    limits: &wgpu::Limits,
) -> bool {
    fused_rader_supported_by_limits(
        n,
        m,
        precision,
        workgroup_size,
        min_convolution_length,
        u64::from(limits.max_compute_workgroup_storage_size),
        limits.max_compute_invocations_per_workgroup,
        limits.max_compute_workgroup_size_x,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn fused_rader_supported_by_limits(
    n: usize,
    m: usize,
    precision: AxisPrecision,
    workgroup_size: u32,
    min_convolution_length: usize,
    max_workgroup_storage_bytes: u64,
    max_invocations_per_workgroup: u32,
    max_workgroup_size_x: u32,
) -> bool {
    // The floor applies to the prime's linear convolution length, so a cyclic
    // convolution fuses exactly when its zero-padded form would.
    let linear_length = n.saturating_sub(1).saturating_mul(2).saturating_sub(1);
    if linear_length.max(m) < min_convolution_length {
        return false;
    }
    let Some(workgroup_bytes) = m.checked_mul(precision.complex_size_bytes() as usize) else {
        return false;
    };
    workgroup_bytes as u64 <= max_workgroup_storage_bytes
        && workgroup_size > 0
        && workgroup_size <= max_invocations_per_workgroup
        && workgroup_size <= max_workgroup_size_x
}

/// Fewest butterflies of the largest medium prime radix (see
/// [`FUSED_PRIME_RADICES`]) per line for a Rader axis to take the cyclic
/// medium-prime convolution: each butterfly is one invocation's long
/// straight-line sum, so with fewer of them most invocations wait. On an
/// RTX 5090 lines with at least 36 ran 5% to 55% faster than the zero-padded
/// or register Bluestein convolution (N = 4241 three times as fast), while
/// shorter ones ranged from 40% faster to twice as slow.
const MIN_MEDIUM_PRIME_BUTTERFLIES: usize = 36;

/// Fused schedule of the cyclic convolution of the prime `n` when `n - 1`
/// has prime factors from [`FUSED_PRIME_RADICES`] besides radices up to 13:
/// those primes' stages, largest first, then the fused smooth schedule of
/// the rest. `None` when `n - 1` is 13-smooth, has a larger prime factor, or
/// holds fewer than [`MIN_MEDIUM_PRIME_BUTTERFLIES`] butterflies of its
/// largest medium prime.
pub(crate) fn medium_prime_rader_schedule(n: usize) -> Option<Vec<usize>> {
    let l = n.checked_sub(1)?;
    if l < 2 || crate::runtime::factor_supported_length(l).is_ok() {
        return None;
    }
    let mut rest = l;
    let mut schedule = Vec::new();
    for &prime in FUSED_PRIME_RADICES.iter().rev() {
        while rest.is_multiple_of(prime) {
            schedule.push(prime);
            rest /= prime;
        }
    }
    let largest = *schedule.first()?;
    if l / largest < MIN_MEDIUM_PRIME_BUTTERFLIES {
        return None;
    }
    if rest > 1 {
        let factors = crate::runtime::factor_supported_length(rest).ok()?;
        schedule.extend(fused_smooth_factors(rest, &factors));
    }
    Some(schedule)
}

/// Convolution length of the Rader axis of the prime `n`, and its fused
/// schedule when it is a cyclic medium-prime convolution (see
/// [`medium_prime_rader_schedule`]), which needs an `f32` fused kernel the
/// device fits; otherwise [`rader_convolution_length`].
pub(crate) fn rader_axis_convolution(
    n: usize,
    precision: AxisPrecision,
    fused_workgroup_size: u32,
    fused_min_convolution_length: usize,
    limits: &wgpu::Limits,
) -> Result<(usize, Option<Vec<usize>>)> {
    if precision == AxisPrecision::F32 {
        if let Some(schedule) = medium_prime_rader_schedule(n).filter(|_| {
            fused_rader_supported(
                n,
                n - 1,
                precision,
                fused_workgroup_size,
                fused_min_convolution_length,
                limits,
            )
        }) {
            return Ok((n - 1, Some(schedule)));
        }
    }
    Ok((rader_convolution_length(n)?, None))
}

/// Length of Rader's convolution for the prime `n`: the cyclic length
/// `n - 1` itself when an FFT of that length is supported, else a zero-padded
/// length of at least `2 (n - 1) - 1` whose linear convolution folds back.
/// Whether a Rader axis of the prime `n`, whose convolution takes
/// `rader_length` points, runs as a register Bluestein convolution of
/// `register_length` points instead: when its own fused convolution does
/// not fit, or when that convolution is linear (`n - 1` is not smooth) and
/// the register one is short (see [`short_register_convolution`]).
pub(crate) fn rader_prefers_register_bluestein(
    n: usize,
    rader_length: usize,
    register_length: usize,
    rader_fused: bool,
) -> bool {
    !rader_fused
        || (rader_length != n - 1 && short_register_convolution(register_length, rader_length))
}

pub(crate) fn rader_convolution_length(n: usize) -> Result<usize> {
    let l = n
        .checked_sub(1)
        .ok_or(FftError::UnsupportedLength { len: n })?;
    if l >= 2 && crate::runtime::factor_supported_length(l).is_ok() {
        return Ok(l);
    }
    let min_conv = l
        .checked_mul(2)
        .and_then(|value| value.checked_sub(1))
        .ok_or(FftError::LengthTooLarge { len: n })?;
    let mut m = next_smooth_at_least(min_conv);
    if crate::runtime::factor_supported_length(m).is_err() {
        m = next_power_of_two_at_least(min_conv);
    }
    Ok(m)
}

/// `perm` followed by the destination table: entry `L + q` is the slot of
/// the reversed convolution input that `x[q + 1]` fills, where slot `t` holds
/// `x[perm[L - 1 - t]]`.
pub(crate) fn rader_permutation_table(perm: &[u32]) -> Vec<u32> {
    let l = perm.len();
    let mut table = perm.to_vec();
    table.resize(2 * l, 0);
    for (k, &natural) in perm.iter().enumerate() {
        table[l + natural as usize - 1] = (l - 1 - k) as u32;
    }
    table
}

pub(crate) fn rader_permutation(n: usize) -> Result<Vec<u32>> {
    let root = primitive_root_prime(n).ok_or(FftError::UnsupportedLength { len: n })?;
    Ok((0..n - 1).map(|k| mod_pow(root, k + 1, n) as u32).collect())
}

pub(crate) fn rader_bfft(
    n: usize,
    m: usize,
    direction: FftDirection,
    perm: &[u32],
) -> Result<Vec<Complex32>> {
    Ok(rader_bfft_f64(n, m, direction, perm)?
        .into_iter()
        .map(round_complex64)
        .collect())
}

pub(crate) fn rader_bfft_f64(
    n: usize,
    m: usize,
    direction: FftDirection,
    perm: &[u32],
) -> Result<Vec<Complex64>> {
    let sign: f64 = match direction {
        FftDirection::Forward => 1.0,
        FftDirection::Inverse => -1.0,
    };
    let mut values = vec![Complex64::default(); m];
    for (k, &index) in perm.iter().enumerate() {
        let angle = sign * (-std::f64::consts::TAU * index as f64 / n as f64);
        let (sin, cos) = angle.sin_cos();
        values[k] = Complex64::new(cos, sin);
    }

    Ok(fft_f64(&values, FftDirection::Forward))
}

fn round_complex64(value: Complex64) -> Complex32 {
    Complex32::new(value.re as f32, value.im as f32)
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

/// WGSL of a fused Rader kernel: one workgroup transforms one line.
///
/// Every invocation loads the line in natural order, coalesced, and scatters
/// each element to its place in the permuted convolution input using the
/// destination table after the permutation in `perm`. The forward FFT's
/// bin 0 is the sum of `x[1..N)`, so `X[0] = x[0] + A[0]` needs no reduction.
/// Results return to natural order in workgroup memory before a coalesced
/// store.
pub(crate) fn generate_fused_rader_wgsl_for_key(key: &FusedPrimeStageKey) -> String {
    debug_assert_eq!(key.kind, FusedPrimeKind::Rader);
    if key.serial_lines > 1 {
        let source = generate_fused_rader_serial_wgsl(key);
        return if key.padded_indices {
            crate::runtime::axis_plan::pad_workgroup_indices(&source)
        } else {
            source
        };
    }
    if key.lines_per_workgroup > 1 {
        let source = generate_fused_rader_multiline_wgsl(key);
        return if key.padded_indices {
            crate::runtime::axis_plan::pad_workgroup_indices(&source)
        } else {
            source
        };
    }
    let n = key.axis_length;
    let l = n - 1;
    let m = key.convolution_length;
    let m_slot_count = m.div_ceil(key.workgroup_size as usize);
    let l_slot_count = l.div_ceil(key.workgroup_size as usize);
    let line_base_fn = wgsl_line_base_fn(key.rank, key.axis, &key.dims);
    let scale = match key.precision {
        AxisPrecision::Df64 => format_df64(key.scale_factor()),
        _ => key.precision.format_wgsl_scalar(key.scale_factor()),
    };
    let inverse_m = match key.precision {
        AxisPrecision::F32 => key
            .precision
            .format_wgsl_scalar(f64::from(1.0f32 / m as f32)),
        AxisPrecision::F64 => key.precision.format_wgsl_scalar(1.0 / m as f64),
        AxisPrecision::Df64 => format_df64(1.0 / m as f64),
    };
    let scale_ref = if key.precision == AxisPrecision::Df64 {
        scale.as_str()
    } else {
        "SCALE"
    };
    let inverse_m_ref = if key.precision == AxisPrecision::Df64 {
        inverse_m.as_str()
    } else {
        "INVERSE_M"
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

    let zero = complex_zero(key.precision);
    let twiddle_inverse_value = match key.precision {
        AxisPrecision::Df64 => "vec4<f32>(value.x, value.y, -value.z, -value.w)",
        _ => "vec2<f32>(value.x, -value.y)",
    };
    // Parenthesized: the f32 and f64 scale multiplies its operand's last term.
    let scaled_first = complex_scale_expr(
        key.precision,
        &format!("({})", complex_add_expr(key.precision, "x0", "scratch[0]")),
        scale_ref,
    );
    let scaled_convolution = complex_scale_expr(key.precision, "scratch[t]", inverse_m_ref);
    let scaled_wrap = complex_scale_expr(key.precision, "scratch[wrap]", inverse_m_ref);
    let wrap_add = complex_add_expr(key.precision, "convolution", &scaled_wrap);
    let x0_add = complex_add_expr(key.precision, "x0", "convolution");
    let scaled_result = complex_scale_expr(key.precision, "scratch[q]", scale_ref);

    // Each invocation holds its outputs while others still read the
    // convolution, then writes them back in natural order.
    let mut collect = String::new();
    let mut scatter = String::new();
    for slot in 0..l_slot_count {
        collect.push_str(&format!(
            r#"  var value{slot}: vec2<f32> = {zero};
  {{
    let t: u32 = lid.x + {slot}u * WORKGROUP_SIZE;
    if (t < L) {{
      var convolution: vec2<f32> = {scaled_convolution};
      let wrap: u32 = t + L;
      if (wrap < M) {{
        convolution = {wrap_add};
      }}
      value{slot} = {x0_add};
    }}
  }}
"#
        ));
        scatter.push_str(&format!(
            "  if (lid.x + {slot}u * WORKGROUP_SIZE < L) {{\n    scratch[perm[lid.x + {slot}u * WORKGROUP_SIZE] - 1u] = value{slot};\n  }}\n"
        ));
    }

    let source = specialize_rader_wgsl(
        format!(
            r#"struct Params {{
  lines: u32,
  lineOffset: u32,
  pad0: u32,
  pad1: u32,
}};

@group(0) @binding(0) var<storage, read> input: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec2<f32>>;
// perm[t] is the natural index of convolution output t; perm[L + q] is the
// convolution input slot of x[q + 1].
@group(0) @binding(2) var<storage, read> perm: array<u32>;
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
  return {twiddle_inverse_value};
}}

const N: u32 = {n}u;
const L: u32 = {l}u;
const M: u32 = {m}u;
const STRIDE: u32 = {stride}u;
const WORKGROUP_SIZE: u32 = {workgroup_size}u;
const M_SLOT_COUNT: u32 = {m_slot_count}u;
const L_SLOT_COUNT: u32 = {l_slot_count}u;
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
  let x0: vec2<f32> = input[base];
  for (var slot: u32 = 0u; slot < L_SLOT_COUNT; slot = slot + 1u) {{
    let q: u32 = lid.x + slot * WORKGROUP_SIZE;
    if (q < L) {{
      scratch[perm[L + q]] = input[base + (q + 1u) * STRIDE];
    }}
  }}
  for (var slot: u32 = 0u; slot < M_SLOT_COUNT; slot = slot + 1u) {{
    let t: u32 = lid.x + slot * WORKGROUP_SIZE;
    if (t >= L && t < M) {{
      scratch[t] = {zero};
    }}
  }}
  workgroupBarrier();

{forward_stages}
  // Bin 0 of the forward FFT is the sum of x[1..N).
  if (lid.x == 0u) {{
    output[base] = {scaled_first};
  }}
  for (var slot: u32 = 0u; slot < M_SLOT_COUNT; slot = slot + 1u) {{
    let t: u32 = lid.x + slot * WORKGROUP_SIZE;
    if (t < M) {{
      scratch[t] = c_mul(scratch[t], bfft[t]);
    }}
  }}
  workgroupBarrier();

{inverse_stages}
{collect}  workgroupBarrier();
{scatter}  workgroupBarrier();
  for (var slot: u32 = 0u; slot < L_SLOT_COUNT; slot = slot + 1u) {{
    let q: u32 = lid.x + slot * WORKGROUP_SIZE;
    if (q < L) {{
      output[base + (q + 1u) * STRIDE] = {scaled_result};
    }}
  }}
}}
"#,
            stride = key.stride_complex,
            workgroup_size = key.workgroup_size,
            entry_params = crate::runtime::dispatch::WGSL_FLAT_ENTRY_PARAMS,
            flat_workgroup_index = crate::runtime::dispatch::WGSL_FLAT_WORKGROUP_INDEX,
            scalar = staged_scalar_type(key.precision),
        ),
        key.precision,
    );
    if key.padded_indices {
        crate::runtime::axis_plan::pad_workgroup_indices(&source)
    } else {
        source
    }
}

/// Workgroups a strided fused Rader axis keeps when it takes serial lines:
/// with 64, an axis of 256 lines ran 70% slower.
const MIN_SERIAL_RADER_WORKGROUPS: usize = 256;

/// Strided lines a one-line fused Rader kernel of `m` points loads and
/// stores together (see [`generate_fused_rader_serial_wgsl`]) when `lines`
/// lines leave [`MIN_SERIAL_RADER_WORKGROUPS`] workgroups; 1 otherwise. Each
/// line's convolution waits for the previous one, so shorter convolutions
/// take fewer: on an RTX 5090 two lines of 810 points ran 13% faster and
/// four 4% slower, while four lines of 2178 and 2457 points ran 8% and 13%
/// faster than one.
fn serial_rader_lines(m: usize, lines: usize) -> usize {
    let serial = if m >= 2048 { 4 } else { 2 };
    if lines >= serial * MIN_SERIAL_RADER_WORKGROUPS {
        serial
    } else {
        1
    }
}

/// WGSL of a fused Rader kernel for strided lines too long for several in
/// workgroup memory: a workgroup loads `key.serial_lines` neighbouring lines
/// element-major, so each row of the load spans them, convolves them one
/// after another in one line of workgroup memory, and stores them together.
/// Invocation `lid` holds elements `lid / LINES + s * LANES` of line
/// `lid % LINES` in registers.
fn generate_fused_rader_serial_wgsl(key: &FusedPrimeStageKey) -> String {
    debug_assert_eq!(key.precision, AxisPrecision::F32);
    let n = key.axis_length;
    let l = n - 1;
    let m = key.convolution_length;
    let lines = key.serial_lines as usize;
    let workgroup_size = key.workgroup_size as usize;
    debug_assert!(lines > 1 && workgroup_size.is_multiple_of(lines));
    let lanes = workgroup_size / lines;
    let slots = n.div_ceil(lanes);
    let m_slot_count = m.div_ceil(workgroup_size);
    let l_slot_count = l.div_ceil(workgroup_size);
    let line_base_fn = wgsl_line_base_fn(key.rank, key.axis, &key.dims);
    let scale = key.precision.format_wgsl_scalar(key.scale_factor());
    let inverse_m = key
        .precision
        .format_wgsl_scalar(f64::from(1.0f32 / m as f32));
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

    let mut loads = String::new();
    let mut scatter = String::new();
    let mut gather = String::new();
    let mut stores = String::new();
    for s in 0..slots {
        let q = format!("(lane + {}u)", s * lanes);
        loads.push_str(&format!(
            "  var v{s}: vec2<f32> = vec2<f32>(0.0, 0.0);\n  if ({q} < N) {{\n    v{s} = input[base + {q} * STRIDE];\n  }}\n"
        ));
        scatter.push_str(&format!(
            "      if ({q} >= 1u && {q} < N) {{\n        scratch[perm[L + {q} - 1u]] = v{s};\n      }}\n"
        ));
        gather.push_str(&format!(
            "      if ({q} >= 1u && {q} < N) {{\n        v{s} = scratch[{q} - 1u] * SCALE;\n      }}\n"
        ));
        stores.push_str(&format!(
            "    if ({q} < N) {{\n      output[base + {q} * STRIDE] = v{s};\n    }}\n"
        ));
    }
    let mut collect = String::new();
    let mut collect_scatter = String::new();
    for slot in 0..l_slot_count {
        collect.push_str(&format!(
            r#"    var value{slot}: vec2<f32> = vec2<f32>(0.0, 0.0);
    {{
      let t: u32 = lid.x + {slot}u * WORKGROUP_SIZE;
      if (t < L) {{
        var convolution: vec2<f32> = scratch[t] * INVERSE_M;
        let wrap: u32 = t + L;
        if (wrap < M) {{
          convolution = convolution + scratch[wrap] * INVERSE_M;
        }}
        value{slot} = x0 + convolution;
      }}
    }}
"#
        ));
        collect_scatter.push_str(&format!(
            "    if (lid.x + {slot}u * WORKGROUP_SIZE < L) {{\n      scratch[perm[lid.x + {slot}u * WORKGROUP_SIZE] - 1u] = value{slot};\n    }}\n"
        ));
    }

    format!(
        r#"struct Params {{
  lines: u32,
  lineOffset: u32,
  pad0: u32,
  pad1: u32,
}};

@group(0) @binding(0) var<storage, read> input: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec2<f32>>;
// perm[t] is the natural index of convolution output t; perm[L + q] is the
// convolution input slot of x[q + 1].
@group(0) @binding(2) var<storage, read> perm: array<u32>;
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
const L: u32 = {l}u;
const M: u32 = {m}u;
const STRIDE: u32 = {stride}u;
const WORKGROUP_SIZE: u32 = {workgroup_size}u;
const LINES: u32 = {lines}u;
const LANES: u32 = {lanes}u;
const M_SLOT_COUNT: u32 = {m_slot_count}u;
const INVERSE_M: f32 = {inverse_m};
const SCALE: f32 = {scale};

var<workgroup> scratch: array<vec2<f32>, {m}>;
// Each line's x[0], and its output bin 0.
var<workgroup> firsts: array<vec2<f32>, {lines}>;
var<workgroup> dcs: array<vec2<f32>, {lines}>;

{line_base_fn}

@compute @workgroup_size({workgroup_size}, 1, 1)
fn main({entry_params}) {{
  {flat_workgroup_index}
  let groupLine: u32 = wgFlat * LINES;
  if (groupLine >= params.lines) {{
    return;
  }}
  let lineCount: u32 = min(LINES, params.lines - groupLine);
  let lineInGroup: u32 = lid.x % LINES;
  let lane: u32 = lid.x / LINES;
  let base: u32 = line_base(params.lineOffset + groupLine + min(lineInGroup, lineCount - 1u));
{loads}  if (lane == 0u) {{
    firsts[lineInGroup] = v0;
  }}
  for (var line: u32 = 0u; line < lineCount; line = line + 1u) {{
    if (lineInGroup == line) {{
{scatter}    }}
    for (var slot: u32 = 0u; slot < M_SLOT_COUNT; slot = slot + 1u) {{
      let t: u32 = lid.x + slot * WORKGROUP_SIZE;
      if (t >= L && t < M) {{
        scratch[t] = vec2<f32>(0.0, 0.0);
      }}
    }}
    workgroupBarrier();
    let x0: vec2<f32> = firsts[line];

{forward_stages}
    // Bin 0 of the forward FFT is the sum of x[1..N).
    if (lid.x == 0u) {{
      dcs[line] = (x0 + scratch[0]) * SCALE;
    }}
    for (var slot: u32 = 0u; slot < M_SLOT_COUNT; slot = slot + 1u) {{
      let t: u32 = lid.x + slot * WORKGROUP_SIZE;
      if (t < M) {{
        scratch[t] = c_mul(scratch[t], bfft[t]);
      }}
    }}
    workgroupBarrier();

{inverse_stages}
{collect}    workgroupBarrier();
{collect_scatter}    workgroupBarrier();
    if (lineInGroup == line) {{
{gather}    }}
    workgroupBarrier();
  }}
  if (lane == 0u) {{
    v0 = dcs[lineInGroup];
  }}
  if (lineInGroup < lineCount) {{
{stores}  }}
}}
"#,
        stride = key.stride_complex,
        entry_params = crate::runtime::dispatch::WGSL_FLAT_ENTRY_PARAMS,
        flat_workgroup_index = crate::runtime::dispatch::WGSL_FLAT_WORKGROUP_INDEX,
    )
}

/// Fewest lines a multi-line fused Rader kernel takes per workgroup: with
/// two, the 1008-point convolution of N=1009 measured 24% slower than one
/// line per workgroup, while 4 to 16 lines of convolutions up to 126 points
/// measured 17% to 41% faster (RTX 5090).
const MIN_RADER_LINES: u32 = 4;

/// WGSL of a fused Rader kernel over `key.lines_per_workgroup` lines per
/// workgroup: the single-line kernel's steps with every line's
/// convolution in workgroup memory, `LINE_STRIDE` apart, and each line's
/// `x[0]` in `firsts`. Strided lines load and store element-major, so
/// neighbouring invocations touch neighbouring lines.
fn generate_fused_rader_multiline_wgsl(key: &FusedPrimeStageKey) -> String {
    let n = key.axis_length;
    let l = n - 1;
    let m = key.convolution_length;
    let lines = key.lines_per_workgroup as usize;
    let workgroup_size = key.workgroup_size as usize;
    debug_assert!(lines > 1 && lines <= workgroup_size);
    let element_major = key.stride_complex != 1;
    let line_stride = multiline_line_stride(m, lines, element_major);
    let line_base_fn = wgsl_line_base_fn(key.rank, key.axis, &key.dims);
    let scale = match key.precision {
        AxisPrecision::Df64 => format_df64(key.scale_factor()),
        _ => key.precision.format_wgsl_scalar(key.scale_factor()),
    };
    let inverse_m = match key.precision {
        AxisPrecision::F32 => key
            .precision
            .format_wgsl_scalar(f64::from(1.0f32 / m as f32)),
        AxisPrecision::F64 => key.precision.format_wgsl_scalar(1.0 / m as f64),
        AxisPrecision::Df64 => format_df64(1.0 / m as f64),
    };
    let scale_ref = if key.precision == AxisPrecision::Df64 {
        scale.as_str()
    } else {
        "SCALE"
    };
    let inverse_m_ref = if key.precision == AxisPrecision::Df64 {
        inverse_m.as_str()
    } else {
        "INVERSE_M"
    };
    let stages = |direction: FftDirection, twiddle: &str| {
        let mut ns = 1usize;
        let mut stages = String::new();
        for &radix in &key.factors {
            ns *= radix;
            stages.push_str(&generate_in_place_smooth_fft_stage_multiline_wgsl(
                m,
                radix,
                ns,
                direction,
                key.workgroup_size,
                lines,
                twiddle,
                key.precision,
            ));
        }
        stages
    };
    let forward_stages = stages(FftDirection::Forward, "twiddle_forward");
    let inverse_stages = stages(FftDirection::Inverse, "twiddle_inverse");

    let zero = complex_zero(key.precision);
    let twiddle_inverse_value = match key.precision {
        AxisPrecision::Df64 => "vec4<f32>(value.x, value.y, -value.z, -value.w)",
        _ => "vec2<f32>(value.x, -value.y)",
    };
    // Invocation `e` takes line `lineSlot`, input `q` of the permuted load
    // and natural-order store.
    let split = if element_major {
        "let lineSlot: u32 = e % LINES;\n    let q: u32 = e / LINES;"
    } else {
        "let lineSlot: u32 = e / L;\n    let q: u32 = e - lineSlot * L;"
    };
    let zero_fill = if m > l {
        format!(
            r#"  for (var e: u32 = lid.x; e < LINES * (M - L); e = e + WORKGROUP_SIZE) {{
    let lineSlot: u32 = e / (M - L);
    let t: u32 = L + e - lineSlot * (M - L);
    if (lineSlot < lineCount) {{
      scratch[lineSlot * LINE_STRIDE + t] = {zero};
    }}
  }}
"#
        )
    } else {
        String::new()
    };
    let scaled_first = complex_scale_expr(
        key.precision,
        &format!(
            "({})",
            complex_add_expr(key.precision, "firsts[lineSlot]", "scratch[index]")
        ),
        scale_ref,
    );
    let scaled_result = complex_scale_expr(
        key.precision,
        "scratch[lineSlot * LINE_STRIDE + q]",
        scale_ref,
    );

    // Each invocation holds its outputs while others still read the
    // convolution, then writes them back in natural order.
    let mut collect = String::new();
    let mut scatter = String::new();
    for slot in 0..(lines * l).div_ceil(workgroup_size) {
        let scaled_convolution = complex_scale_expr(
            key.precision,
            &format!("scratch[lineSlot{slot} * LINE_STRIDE + t{slot}]"),
            inverse_m_ref,
        );
        let scaled_wrap = complex_scale_expr(
            key.precision,
            &format!("scratch[lineSlot{slot} * LINE_STRIDE + wrap]"),
            inverse_m_ref,
        );
        let wrap_add = complex_add_expr(key.precision, "convolution", &scaled_wrap);
        let first_add = complex_add_expr(
            key.precision,
            &format!("firsts[lineSlot{slot}]"),
            "convolution",
        );
        collect.push_str(&format!(
            r#"  let e{slot}: u32 = lid.x + {slot}u * WORKGROUP_SIZE;
  let lineSlot{slot}: u32 = e{slot} / L;
  let t{slot}: u32 = e{slot} - lineSlot{slot} * L;
  var value{slot}: vec2<f32> = {zero};
  if (lineSlot{slot} < lineCount) {{
    var convolution: vec2<f32> = {scaled_convolution};
    let wrap: u32 = t{slot} + L;
    if (wrap < M) {{
      convolution = {wrap_add};
    }}
    value{slot} = {first_add};
  }}
"#
        ));
        scatter.push_str(&format!(
            "  if (lineSlot{slot} < lineCount) {{\n    scratch[lineSlot{slot} * LINE_STRIDE + perm[t{slot}] - 1u] = value{slot};\n  }}\n"
        ));
    }

    specialize_rader_wgsl(
        format!(
            r#"struct Params {{
  lines: u32,
  lineOffset: u32,
  pad0: u32,
  pad1: u32,
}};

@group(0) @binding(0) var<storage, read> input: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec2<f32>>;
// perm[t] is the natural index of convolution output t; perm[L + q] is the
// convolution input slot of x[q + 1].
@group(0) @binding(2) var<storage, read> perm: array<u32>;
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
  return {twiddle_inverse_value};
}}

const N: u32 = {n}u;
const L: u32 = {l}u;
const M: u32 = {m}u;
const STRIDE: u32 = {stride}u;
const WORKGROUP_SIZE: u32 = {workgroup_size}u;
const LINES: u32 = {lines}u;
const LINE_STRIDE: u32 = {line_stride}u;
const INVERSE_M: {scalar} = {inverse_m};
const SCALE: {scalar} = {scale};

var<workgroup> scratch: array<vec2<f32>, {scratch_len}>;
var<workgroup> firsts: array<vec2<f32>, {lines}>;

{line_base_fn}

@compute @workgroup_size({workgroup_size}, 1, 1)
fn main({entry_params}) {{
  {flat_workgroup_index}
  let groupLine: u32 = wgFlat * LINES;
  if (groupLine >= params.lines) {{
    return;
  }}
  let lineCount: u32 = min(LINES, params.lines - groupLine);
  let lineStart: u32 = params.lineOffset + groupLine;

  if (lid.x < lineCount) {{
    firsts[lid.x] = input[line_base(lineStart + lid.x)];
  }}
  for (var e: u32 = lid.x; e < LINES * L; e = e + WORKGROUP_SIZE) {{
    {split}
    if (lineSlot < lineCount) {{
      scratch[lineSlot * LINE_STRIDE + perm[L + q]] = input[line_base(lineStart + lineSlot) + (q + 1u) * STRIDE];
    }}
  }}
{zero_fill}  workgroupBarrier();

{forward_stages}
  // Bin 0 of each line's forward FFT is the sum of its x[1..N).
  for (var e: u32 = lid.x; e < LINES * M; e = e + WORKGROUP_SIZE) {{
    let lineSlot: u32 = e / M;
    let t: u32 = e - lineSlot * M;
    if (lineSlot < lineCount) {{
      let index: u32 = lineSlot * LINE_STRIDE + t;
      if (t == 0u) {{
        output[line_base(lineStart + lineSlot)] = {scaled_first};
      }}
      scratch[index] = c_mul(scratch[index], bfft[t]);
    }}
  }}
  workgroupBarrier();

{inverse_stages}
{collect}  workgroupBarrier();
{scatter}  workgroupBarrier();
  for (var e: u32 = lid.x; e < LINES * L; e = e + WORKGROUP_SIZE) {{
    {split}
    if (lineSlot < lineCount) {{
      output[line_base(lineStart + lineSlot) + (q + 1u) * STRIDE] = {scaled_result};
    }}
  }}
}}
"#,
            stride = key.stride_complex,
            scratch_len = line_stride * lines,
            entry_params = crate::runtime::dispatch::WGSL_FLAT_ENTRY_PARAMS,
            flat_workgroup_index = crate::runtime::dispatch::WGSL_FLAT_WORKGROUP_INDEX,
            scalar = staged_scalar_type(key.precision),
        ),
        key.precision,
    )
}

pub(crate) fn generate_rader_wgsl_for_key(key: &RaderStageKey) -> String {
    match key.kind {
        RaderKernelKind::Sum => generate_rader_sum_wgsl(key),
        RaderKernelKind::Pack => generate_rader_pack_wgsl(key),
        RaderKernelKind::Mul => generate_rader_mul_wgsl(key),
        RaderKernelKind::WriteY0 => generate_rader_write_y0_wgsl(key),
        RaderKernelKind::Post => generate_rader_post_wgsl(key),
    }
}

fn generate_rader_sum_wgsl(key: &RaderStageKey) -> String {
    let line_base_fn = wgsl_line_base_fn(key.rank, key.axis, &key.dims);
    let zero = complex_zero(key.precision);
    let acc_add = complex_add_expr(key.precision, "acc", "input[base + i * STRIDE]");
    let reduction_add =
        complex_add_expr(key.precision, "scratch[lid.x]", "scratch[lid.x + stride]");
    specialize_rader_wgsl(
        format!(
            r#"struct Params {{
  lines: u32,
  lineOffset: u32,
  pad0: u32,
  pad1: u32,
}};

@group(0) @binding(0) var<storage, read> input: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> sumAll: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> x0: array<vec2<f32>>;
@group(0) @binding(3) var<uniform> params: Params;

const N: u32 = {n}u;
const STRIDE: u32 = {stride}u;
const WORKGROUP_SIZE: u32 = {workgroup_size}u;

{line_base_fn}

var<workgroup> scratch: array<vec2<f32>, {workgroup_size}>;

@compute @workgroup_size({workgroup_size}, 1, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {{
  let lineLocal: u32 = (wid.z * nwg.y + wid.y) * nwg.x + wid.x;
  if (lineLocal >= params.lines) {{
    return;
  }}

  let base: u32 = line_base(params.lineOffset + lineLocal);
  var acc: vec2<f32> = {zero};
  var i: u32 = lid.x;
  loop {{
    if (i >= N) {{
      break;
    }}
    acc = {acc_add};
    i = i + WORKGROUP_SIZE;
  }}
  scratch[lid.x] = acc;
  workgroupBarrier();

  var stride: u32 = WORKGROUP_SIZE / 2u;
  loop {{
    if (stride == 0u) {{
      break;
    }}
    if (lid.x < stride) {{
      scratch[lid.x] = {reduction_add};
    }}
    workgroupBarrier();
    stride = stride / 2u;
  }}

  if (lid.x == 0u) {{
    sumAll[lineLocal] = scratch[0];
    x0[lineLocal] = input[base];
  }}
}}
"#,
            n = key.axis_length,
            stride = key.stride_complex,
            workgroup_size = key.workgroup_size,
            line_base_fn = line_base_fn,
        ),
        key.precision,
    )
}

fn generate_rader_pack_wgsl(key: &RaderStageKey) -> String {
    let line_base_fn = wgsl_line_base_fn(key.rank, key.axis, &key.dims);
    let zero = complex_zero(key.precision);
    specialize_rader_wgsl(
        format!(
            r#"struct Params {{
  lines: u32,
  lineOffset: u32,
  pad0: u32,
  pad1: u32,
}};

@group(0) @binding(0) var<storage, read> input: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> work: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> perm: array<u32>;
@group(0) @binding(3) var<uniform> params: Params;

const N: u32 = {n}u;
const L: u32 = {l}u;
const M: u32 = {m}u;
const STRIDE: u32 = {stride}u;

{line_base_fn}

@compute @workgroup_size({workgroup_size}, 1, 1)
fn main({entry_params}) {{
  {flat_index}

  let lineLocal: u32 = i / M;
  let t: u32 = i - lineLocal * M;
  let dst: u32 = lineLocal * M + t;
  if (t >= L) {{
    work[dst] = {zero};
    return;
  }}

  let base: u32 = line_base(params.lineOffset + lineLocal);
  let sourceIndex: u32 = perm[(L - 1u) - t];
  work[dst] = input[base + sourceIndex * STRIDE];
}}
"#,
            n = key.axis_length,
            l = key.axis_length - 1,
            m = key.convolution_length,
            stride = key.stride_complex,
            workgroup_size = key.workgroup_size,
            entry_params = crate::runtime::dispatch::WGSL_FLAT_ENTRY_PARAMS,
            flat_index = crate::runtime::dispatch::wgsl_flat_index_stmts(
                "i",
                "params.lines * M",
                key.workgroup_size,
            ),
            line_base_fn = line_base_fn,
        ),
        key.precision,
    )
}

fn generate_rader_mul_wgsl(key: &RaderStageKey) -> String {
    specialize_rader_wgsl(
        format!(
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

fn c_mul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {{
  return vec2<f32>(
    a.x * b.x - a.y * b.y,
    a.x * b.y + a.y * b.x
  );
}}

@compute @workgroup_size({workgroup_size}, 1, 1)
fn main({entry_params}) {{
  {flat_index}
  let t: u32 = i - (i / M) * M;
  work[i] = c_mul(work[i], bfft[t]);
}}
"#,
            m = key.convolution_length,
            workgroup_size = key.workgroup_size,
            entry_params = crate::runtime::dispatch::WGSL_FLAT_ENTRY_PARAMS,
            flat_index = crate::runtime::dispatch::wgsl_flat_index_stmts(
                "i",
                "params.total",
                key.workgroup_size
            ),
        ),
        key.precision,
    )
}

fn generate_rader_write_y0_wgsl(key: &RaderStageKey) -> String {
    let line_base_fn = wgsl_line_base_fn(key.rank, key.axis, &key.dims);
    let scale = format_staged_scalar(key.precision, key.scale_factor());
    let scale_ref = if key.precision == AxisPrecision::Df64 {
        scale.as_str()
    } else {
        "SCALE"
    };
    let scaled_sum = complex_scale_expr(key.precision, "sumAll[lineLocal]", scale_ref);
    specialize_rader_wgsl(
        format!(
            r#"struct Params {{
  lines: u32,
  lineOffset: u32,
  pad0: u32,
  pad1: u32,
}};

@group(0) @binding(0) var<storage, read> sumAll: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec2<f32>>;
@group(0) @binding(2) var<uniform> params: Params;

const SCALE: {scalar} = {scale};

{line_base_fn}

@compute @workgroup_size({workgroup_size}, 1, 1)
fn main({entry_params}) {{
  {flat_index}

  let base: u32 = line_base(params.lineOffset + lineLocal);
  output[base] = {scaled_sum};
}}
"#,
            scale = scale,
            line_base_fn = line_base_fn,
            workgroup_size = key.workgroup_size,
            entry_params = crate::runtime::dispatch::WGSL_FLAT_ENTRY_PARAMS,
            flat_index = crate::runtime::dispatch::wgsl_flat_index_stmts(
                "lineLocal",
                "params.lines",
                key.workgroup_size,
            ),
            scalar = staged_scalar_type(key.precision),
        ),
        key.precision,
    )
}

fn generate_rader_post_wgsl(key: &RaderStageKey) -> String {
    let line_base_fn = wgsl_line_base_fn(key.rank, key.axis, &key.dims);
    let scale = format_staged_scalar(key.precision, key.scale_factor());
    let wrap_add = complex_add_expr(key.precision, "value", "conv[baseWork + wrap]");
    let x0_add = complex_add_expr(key.precision, "x0[lineLocal]", "value");
    let scaled_value = match key.precision {
        AxisPrecision::Df64 => complex_scale_expr(key.precision, &x0_add, &scale),
        _ => format!("({x0_add}) * vec2<f32>(SCALE, SCALE)"),
    };
    specialize_rader_wgsl(
        format!(
            r#"struct Params {{
  lines: u32,
  lineOffset: u32,
  pad0: u32,
  pad1: u32,
}};

@group(0) @binding(0) var<storage, read> conv: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read> x0: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> perm: array<u32>;
@group(0) @binding(3) var<storage, read_write> output: array<vec2<f32>>;
@group(0) @binding(4) var<uniform> params: Params;

const L: u32 = {l}u;
const M: u32 = {m}u;
const STRIDE: u32 = {stride}u;
const SCALE: {scalar} = {scale};

{line_base_fn}

@compute @workgroup_size({workgroup_size}, 1, 1)
fn main({entry_params}) {{
  {flat_index}

  let lineLocal: u32 = i / L;
  let t: u32 = i - lineLocal * L;
  let baseWork: u32 = lineLocal * M;
  var value: vec2<f32> = conv[baseWork + t];
  let wrap: u32 = t + L;
  if (wrap < M) {{
    value = {wrap_add};
  }}

  let baseOutput: u32 = line_base(params.lineOffset + lineLocal);
  let outputIndex: u32 = perm[t];
  output[baseOutput + outputIndex * STRIDE] =
    {scaled_value};
}}
"#,
            l = key.axis_length - 1,
            m = key.convolution_length,
            stride = key.stride_complex,
            scale = scale,
            workgroup_size = key.workgroup_size,
            entry_params = crate::runtime::dispatch::WGSL_FLAT_ENTRY_PARAMS,
            flat_index = crate::runtime::dispatch::wgsl_flat_index_stmts(
                "i",
                "params.lines * L",
                key.workgroup_size,
            ),
            line_base_fn = line_base_fn,
            scalar = staged_scalar_type(key.precision),
        ),
        key.precision,
    )
}

fn format_staged_scalar(precision: AxisPrecision, value: f64) -> String {
    match precision {
        AxisPrecision::F32 => format_wgsl_f32(value as f32),
        AxisPrecision::F64 => precision.format_wgsl_scalar(value),
        AxisPrecision::Df64 => format_df64(value),
    }
}

fn format_df64(value: f64) -> String {
    let value = DoubleFloat::from_f64(value);
    format!(
        "Df64({}, {})",
        format_wgsl_f32_roundtrip(value.hi),
        format_wgsl_f32_roundtrip(value.lo)
    )
}

fn staged_scalar_type(precision: AxisPrecision) -> &'static str {
    match precision {
        AxisPrecision::Df64 => "Df64",
        _ => precision.wgsl_scalar_type(),
    }
}

fn complex_zero(precision: AxisPrecision) -> &'static str {
    match precision {
        AxisPrecision::Df64 => "vec4<f32>(0.0, 0.0, 0.0, 0.0)",
        _ => "vec2<f32>(0.0, 0.0)",
    }
}

fn complex_add_expr(precision: AxisPrecision, a: &str, b: &str) -> String {
    match precision {
        AxisPrecision::Df64 => format!("df64_complex_add({a}, {b})"),
        _ => format!("{a} + {b}"),
    }
}

fn complex_scale_expr(precision: AxisPrecision, value: &str, scale: &str) -> String {
    match precision {
        AxisPrecision::Df64 => format!("df64_complex_scale({value}, {scale})"),
        _ => format!("{value} * vec2<f32>({scale}, {scale})"),
    }
}

fn specialize_rader_wgsl(source: String, precision: AxisPrecision) -> String {
    if precision != AxisPrecision::Df64 {
        return precision.specialize_wgsl(source);
    }
    let source = source
        .replace(F32_COMPLEX_HELPERS, DF64_COMPLEX_HELPERS)
        .replace(F32_COMPLEX_MUL_HELPER, DF64_COMPLEX_MUL_HELPER)
        .replace("vec2<f32>", "vec4<f32>");
    format!("{}\n{source}", crate::kernels::DF64_WGSL)
}

const F32_COMPLEX_HELPERS: &str = r#"fn c_add(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
  return a + b;
}

fn c_sub(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
  return a - b;
}

fn c_mul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
  return vec2<f32>(
    a.x * b.x - a.y * b.y,
    a.x * b.y + a.y * b.x
  );
}"#;

const DF64_COMPLEX_HELPERS: &str = r#"fn c_add(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
  return df64_complex_add(a, b);
}

fn c_sub(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
  return df64_complex_sub(a, b);
}

fn c_mul(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
  return df64_complex_mul(a, b);
}"#;

const F32_COMPLEX_MUL_HELPER: &str = r#"fn c_mul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
  return vec2<f32>(
    a.x * b.x - a.y * b.y,
    a.x * b.y + a.y * b.x
  );
}"#;

const DF64_COMPLEX_MUL_HELPER: &str = r#"fn c_mul(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
  return df64_complex_mul(a, b);
}"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fused_rader_gate_accounts_for_scratch_and_tiny_line_floor() {
        assert!(fused_rader_supported_by_limits(
            2999,
            6000,
            AxisPrecision::F32,
            DEFAULT_FUSED_WORKGROUP_SIZE,
            DEFAULT_FUSED_MIN_CONVOLUTION_LENGTH,
            48 * 1024,
            256,
            256
        ));
        assert!(!fused_rader_supported_by_limits(
            2999,
            6000,
            AxisPrecision::F32,
            DEFAULT_FUSED_WORKGROUP_SIZE,
            DEFAULT_FUSED_MIN_CONVOLUTION_LENGTH,
            16 * 1024,
            256,
            256
        ));
        assert!(fused_rader_supported_by_limits(
            1009,
            2016,
            AxisPrecision::F32,
            DEFAULT_FUSED_WORKGROUP_SIZE,
            DEFAULT_FUSED_MIN_CONVOLUTION_LENGTH,
            16 * 1024,
            256,
            256
        ));
        assert!(!fused_rader_supported_by_limits(
            17,
            32,
            AxisPrecision::F32,
            DEFAULT_FUSED_WORKGROUP_SIZE,
            DEFAULT_FUSED_MIN_CONVOLUTION_LENGTH,
            48 * 1024,
            256,
            256
        ));
        assert!(!fused_rader_supported_by_limits(
            2999,
            6000,
            AxisPrecision::F32,
            DEFAULT_FUSED_WORKGROUP_SIZE,
            DEFAULT_FUSED_MIN_CONVOLUTION_LENGTH,
            47_999,
            256,
            256
        ));
        assert!(!fused_rader_supported_by_limits(
            2999,
            6000,
            AxisPrecision::F32,
            DEFAULT_FUSED_WORKGROUP_SIZE,
            DEFAULT_FUSED_MIN_CONVOLUTION_LENGTH,
            48 * 1024,
            255,
            256
        ));
        assert!(fused_rader_supported_by_limits(
            1500,
            3071,
            AxisPrecision::F64,
            DEFAULT_FUSED_WORKGROUP_SIZE,
            DEFAULT_FUSED_MIN_CONVOLUTION_LENGTH,
            48 * 1024,
            256,
            256
        ));
        assert!(!fused_rader_supported_by_limits(
            1500,
            3073,
            AxisPrecision::F64,
            DEFAULT_FUSED_WORKGROUP_SIZE,
            DEFAULT_FUSED_MIN_CONVOLUTION_LENGTH,
            48 * 1024,
            256,
            256
        ));
        assert!(fused_rader_supported_by_limits(
            1500,
            3071,
            AxisPrecision::Df64,
            DEFAULT_FUSED_WORKGROUP_SIZE,
            DEFAULT_FUSED_MIN_CONVOLUTION_LENGTH,
            48 * 1024,
            256,
            256
        ));
        assert!(!fused_rader_supported_by_limits(
            1500,
            3073,
            AxisPrecision::Df64,
            DEFAULT_FUSED_WORKGROUP_SIZE,
            DEFAULT_FUSED_MIN_CONVOLUTION_LENGTH,
            48 * 1024,
            256,
            256
        ));
        assert!(fused_rader_supported_by_limits(
            17,
            32,
            AxisPrecision::F32,
            64,
            32,
            48 * 1024,
            64,
            64
        ));
    }

    #[test]
    fn fused_rader_wgsl_inlines_both_ffts_and_preserves_permutation_contract() {
        let factors = crate::runtime::factor_supported_length(200).unwrap();
        let key = FusedPrimeStageKey::new(
            FusedPrimeKind::Rader,
            2,
            1,
            &[3, 101],
            101,
            3,
            200,
            &factors,
            FftDirection::Forward,
            FUSED_WORKGROUP_SIZE,
            false,
            1.0,
            AxisPrecision::F32,
        );
        let wgsl = generate_fused_rader_wgsl_for_key(&key);
        assert!(wgsl.contains("var<workgroup> scratch: array<vec2<f32>, 200>"));
        // Lines load and store in natural order; the permutation happens in
        // workgroup memory, and X[0] comes from the forward FFT's bin 0.
        assert!(wgsl.contains("let x0: vec2<f32> = input[base];"));
        assert!(wgsl.contains("scratch[perm[L + q]] = input[base + (q + 1u) * STRIDE];"));
        assert!(wgsl.contains("output[base] = (x0 + scratch[0]) * vec2<f32>(SCALE, SCALE);"));
        assert!(wgsl.contains("scratch[perm[lid.x + 0u * WORKGROUP_SIZE] - 1u] = value0;"));
        assert!(wgsl.contains("output[base + (q + 1u) * STRIDE]"));
        assert!(!wgsl.contains("reduction"));
        crate::runtime::assert_workgroup_var_written_before_read(&wgsl, "scratch");
        // The convolution padding must be zeroed in-kernel: pipelines skip
        // wgpu's workgroup zero fill.
        assert!(wgsl.contains("if (t >= L && t < M) {"));
        assert!(wgsl.contains("scratch[t] = vec2<f32>(0.0, 0.0);"));
        assert!(wgsl.contains("let wrap: u32 = t + L"));
        assert!(wgsl.contains("fn twiddle_forward"));
        assert!(wgsl.contains("fn twiddle_inverse"));
        assert!(wgsl.contains("const INVERSE_M: f32"));
        assert!(wgsl.contains("let wgFlat: u32"));
        assert!(!wgsl.contains("sin("));
        assert!(!wgsl.contains("cos("));
    }

    #[test]
    fn fused_rader_f32_scale_uses_roundtrip_literal_and_staged_post_keeps_parentheses() {
        let factors = crate::runtime::factor_supported_length(6000).unwrap();
        let key = FusedPrimeStageKey::new(
            FusedPrimeKind::Rader,
            1,
            0,
            &[2999],
            2999,
            1,
            6000,
            &factors,
            FftDirection::Inverse,
            FUSED_WORKGROUP_SIZE,
            true,
            1.0 / 2999.0,
            AxisPrecision::F32,
        );
        let wgsl = generate_fused_rader_wgsl_for_key(&key);
        let scale = format_wgsl_f32_roundtrip(1.0f32 / 2999.0);
        assert!(wgsl.contains(&format!("const SCALE: f32 = {scale};")));
        assert_ne!(scale, format_wgsl_f32(1.0f32 / 2999.0));

        let post_key = RaderStageKey::new(
            RaderKernelKind::Post,
            1,
            0,
            &[17],
            17,
            1,
            32,
            WORKGROUP_SIZE,
            true,
            1.0 / 17.0,
            AxisPrecision::F32,
        );
        let post = generate_rader_post_wgsl(&post_key);
        assert!(post.contains(
            "output[baseOutput + outputIndex * STRIDE] =\n    (x0[lineLocal] + value) * vec2<f32>(SCALE, SCALE);"
        ));
        assert!(!post.contains("x0[lineLocal] + value * vec2<f32>"));
    }

    #[test]
    fn fused_rader_wgsl_specializes_storage_and_scalars_to_f64() {
        let factors = crate::runtime::factor_supported_length(200).unwrap();
        let key = FusedPrimeStageKey::new(
            FusedPrimeKind::Rader,
            1,
            0,
            &[101],
            101,
            1,
            200,
            &factors,
            FftDirection::Inverse,
            FUSED_WORKGROUP_SIZE,
            true,
            1.0 / 101.0,
            AxisPrecision::F64,
        );
        let wgsl = generate_fused_rader_wgsl_for_key(&key);
        assert!(wgsl.contains("array<vec2<f64>, 200>"));
        assert!(wgsl.contains("let x0: vec2<f64> = input[base];"));
        assert!(wgsl.contains("const INVERSE_M: f64 = 0.005lf;"));
        assert!(!wgsl.contains("vec2<f32>"));
    }

    #[test]
    fn df64_rader_generators_use_split_complex_ops_for_fused_and_staged_paths() {
        let factors = crate::runtime::factor_supported_length(200).unwrap();
        let fused_key = FusedPrimeStageKey::new(
            FusedPrimeKind::Rader,
            1,
            0,
            &[101],
            101,
            1,
            200,
            &factors,
            FftDirection::Inverse,
            FUSED_WORKGROUP_SIZE,
            true,
            1.0 / 101.0,
            AxisPrecision::Df64,
        );
        let fused = generate_fused_rader_wgsl_for_key(&fused_key);
        assert!(fused.starts_with(crate::kernels::DF64_WGSL));
        assert!(fused.contains("array<vec4<f32>, 200>"));
        assert!(fused.contains("const INVERSE_M: Df64 = Df64("));
        assert!(fused.contains("return vec4<f32>(value.x, value.y, -value.z, -value.w)"));
        assert!(fused.contains("df64_complex_add(x0, scratch[0])"));
        assert!(fused.contains("df64_complex_scale(scratch[q], Df64("));
        assert!(!fused.contains("lineSum = lineSum + value"));
        assert!(!fused.contains("vec2<f64>"));

        for kind in [
            RaderKernelKind::Sum,
            RaderKernelKind::Pack,
            RaderKernelKind::Mul,
            RaderKernelKind::WriteY0,
            RaderKernelKind::Post,
        ] {
            let key = RaderStageKey::new(
                kind,
                1,
                0,
                &[17],
                17,
                1,
                32,
                WORKGROUP_SIZE,
                true,
                1.0 / 17.0,
                AxisPrecision::Df64,
            );
            let wgsl = generate_rader_wgsl_for_key(&key);
            assert!(wgsl.starts_with(crate::kernels::DF64_WGSL));
            assert!(wgsl.contains("array<vec4<f32>>"));
            assert!(!wgsl.contains("vec2<f64>"));
            assert!(!wgsl.contains("sin("));
            assert!(!wgsl.contains("cos("));
            if matches!(kind, RaderKernelKind::Sum) {
                crate::runtime::assert_workgroup_var_written_before_read(&wgsl, "scratch");
            }
        }
    }

    #[test]
    fn medium_prime_rader_schedules_need_enough_butterflies() {
        // 4240 = 80 * 53, 1380 = 60 * 23, 5002 = 82 * 61 = 2 * 41 * 61.
        assert_eq!(medium_prime_rader_schedule(4241), Some(vec![53, 16, 5]));
        assert_eq!(medium_prime_rader_schedule(1381), Some(vec![23, 10, 6]));
        assert_eq!(medium_prime_rader_schedule(5003), Some(vec![61, 41, 2]));
        assert_eq!(medium_prime_rader_schedule(613), Some(vec![17, 6, 6]));
        // Too few butterflies (946 = 22 * 43, 282 = 6 * 47), N - 1 smooth
        // (1008), or a prime factor above 61 (7726 = 2 * 3863).
        for n in [947, 283, 1009, 7727] {
            assert_eq!(medium_prime_rader_schedule(n), None, "N={n}");
        }
    }

    #[test]
    fn medium_prime_rader_convolutions_need_storage_and_f32() {
        let limits = |storage: u32| wgpu::Limits {
            max_compute_workgroup_storage_size: storage,
            max_compute_invocations_per_workgroup: 256,
            max_compute_workgroup_size_x: 256,
            ..wgpu::Limits::default()
        };
        let convolution = |n, precision, storage| {
            rader_axis_convolution(n, precision, 256, 0, &limits(storage)).unwrap()
        };
        assert_eq!(
            convolution(4241, AxisPrecision::F32, 49_152),
            (4240, Some(vec![53, 16, 5]))
        );
        assert_eq!(
            convolution(4241, AxisPrecision::F32, 16_384),
            (rader_convolution_length(4241).unwrap(), None)
        );
        assert_eq!(
            convolution(1381, AxisPrecision::F64, 49_152),
            (rader_convolution_length(1381).unwrap(), None)
        );
        assert_eq!(
            convolution(1381, AxisPrecision::F32, 16_384),
            (1380, Some(vec![23, 10, 6]))
        );
    }

    #[test]
    fn rader_permutation_covers_nonzero_prime_indices() {
        let mut values = rader_permutation(17).unwrap();
        values.sort_unstable();
        assert_eq!(values, (1..17).collect::<Vec<_>>());
    }

    #[test]
    fn rader_bfft_is_f64_generated_then_rounded_once() {
        let n = 17;
        let m = 32;
        let perm = rader_permutation(n).unwrap();
        for direction in [FftDirection::Forward, FftDirection::Inverse] {
            let actual = rader_bfft(n, m, direction, &perm).unwrap();
            let sign = if direction == FftDirection::Forward {
                -1.0
            } else {
                1.0
            };
            let mut kernel = vec![Complex64::default(); m];
            for (k, &index) in perm.iter().enumerate() {
                let angle = sign * std::f64::consts::TAU * index as f64 / n as f64;
                let (sin, cos) = angle.sin_cos();
                kernel[k] = Complex64::new(cos, sin);
            }
            let expected = fft_f64(&kernel, FftDirection::Forward);

            for (actual, expected) in actual.iter().zip(expected) {
                assert_eq!(actual.re.to_bits(), (expected.re as f32).to_bits());
                assert_eq!(actual.im.to_bits(), (expected.im as f32).to_bits());
            }
        }
    }

    #[test]
    fn rader_pipeline_key_rejects_non_rader_shader_stage() {
        assert_eq!(
            rader_stage_key(&ComputePipelineCacheKey::direct_dft_c2c_f32()),
            Err(FftError::LargeGraphStageUnsupported {
                stage: "rader-pipeline-key",
                reason: "Rader pipeline key has a non-Rader shader stage",
            })
        );
    }

    #[test]
    fn generated_kernels_contain_nd_constants() {
        let key = RaderStageKey::new(
            RaderKernelKind::Pack,
            2,
            1,
            &[4, 17],
            17,
            4,
            35,
            WORKGROUP_SIZE,
            false,
            1.0,
            AxisPrecision::F32,
        );
        let wgsl = generate_rader_wgsl_for_key(&key);
        assert!(wgsl.contains("const N: u32 = 17u;"));
        assert!(wgsl.contains("const L: u32 = 16u;"));
        assert!(wgsl.contains("const M: u32 = 35u;"));
        assert!(wgsl.contains("const STRIDE: u32 = 4u;"));
        assert!(wgsl.contains("let lines_per_batch: u32 = 4u;"));
    }
}
