use bytemuck::{Pod, Zeroable};

use crate::config::{FftDirection, Normalization};
use crate::error::{FftError, Result};
use crate::math::{reference_c2c_nd, to_interleaved_f32, Complex32};
use crate::runtime::axis_plan::{
    AxisLayout, AxisPlan, AxisPlanConfig, AxisPrecision, AxisStageKind,
};
use crate::runtime::axis_policy::{
    is_prime, mod_pow, next_power_of_two_at_least, next_smooth_at_least, primitive_root_prime,
};
use crate::runtime::buffer_view::BufferView;
use crate::runtime::dispatch::{max_workgroups_per_dimension, split_workgroups};
use crate::runtime::large_graph::{ElementFormat, HelperBufferRange};
use crate::runtime::nd_wgsl::{
    format_wgsl_f32, lines_per_batch, product, stride_for_axis, wgsl_line_base_fn,
};
use crate::runtime::pipeline_cache::{
    with_device_pipeline_cache, ComputePipelineCacheKey, PipelineLayoutCacheKey, RaderKernelKind,
    RaderStageKey, ShaderCacheKey,
};
use crate::runtime::window_scheduler::WindowScheduler;

const WORKGROUP_SIZE: u32 = 64;
const COMPLEX_F32_BYTES: u64 = 8;

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
}

pub(crate) struct RaderAxis {
    n: usize,
    m: usize,
    lines: u32,
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
    lines_params_buffer: wgpu::Buffer,
    total_params_buffer: wgpu::Buffer,
    perm_buffer: wgpu::Buffer,
    bfft_buffer: wgpu::Buffer,
    sum_buffer: wgpu::Buffer,
    x0_buffer: wgpu::Buffer,
    work_buffer: wgpu::Buffer,
    fft_buffer: wgpu::Buffer,
    work_fft_forward: AxisPlan,
    work_fft_inverse: AxisPlan,
}

impl RaderAxisConfig {
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
        let n = self.shape[self.axis];
        if !is_prime(n) {
            return Err(FftError::UnsupportedLength { len: n });
        }
        total_complex(&self.shape, self.batch)?;
        Ok(())
    }

    fn scale(&self) -> Result<f32> {
        let total = product(&self.shape) as f32;
        let scale = match (self.direction, self.normalization) {
            (_, Normalization::None) => 1.0,
            (FftDirection::Forward, Normalization::Forward) => 1.0 / total,
            (FftDirection::Inverse, Normalization::Inverse) => 1.0 / total,
            (_, Normalization::Orthogonal) => 1.0 / total.sqrt(),
            _ => 1.0,
        };
        Ok(scale)
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
        let min_conv = 2 * l - 1;
        let mut m = next_smooth_at_least(min_conv);
        if crate::runtime::factor_supported_length(m).is_err() {
            m = next_power_of_two_at_least(min_conv);
        }
        crate::runtime::factor_supported_length(m)?;

        let lines = checked_mul(config.batch, lines_per_batch(&config.shape, config.axis))?;
        let work_complex = checked_mul(lines, m)?;
        if work_complex > u32::MAX as usize {
            return Err(FftError::LengthTooLarge { len: work_complex });
        }
        let total_work_u32 = work_complex as u32;
        let lines_u32 = lines as u32;
        let stride_complex = stride_for_axis(&config.shape, config.axis);
        let scale = config.scale()?;
        let apply_scale = (scale - 1.0).abs() > f32::EPSILON;
        let perm = rader_permutation(n)?;
        let bfft = rader_bfft(n, m, config.direction, &perm)?;

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

        let perm_buffer = storage_buffer(
            device,
            "wgpu_fft.rader.perm",
            (perm.len() * std::mem::size_of::<u32>()) as u64,
            wgpu::BufferUsages::COPY_DST,
        )?;
        queue.write_buffer(&perm_buffer, 0, bytemuck::cast_slice(&perm));

        let bfft_values = to_interleaved_f32(&bfft);
        let bfft_buffer = storage_buffer(
            device,
            "wgpu_fft.rader.bfft",
            m as u64 * COMPLEX_F32_BYTES,
            wgpu::BufferUsages::COPY_DST,
        )?;
        queue.write_buffer(&bfft_buffer, 0, bytemuck::cast_slice(&bfft_values));

        let line_bytes = lines as u64 * COMPLEX_F32_BYTES;
        let work_bytes = work_complex as u64 * COMPLEX_F32_BYTES;
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

        let work_fft_forward = AxisPlan::new(
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
                precision: AxisPrecision::F32,
            },
        )?;
        let work_fft_inverse = AxisPlan::new(
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
                precision: AxisPrecision::F32,
            },
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
        );

        let sum_bind_group_layout =
            cached_layout(device, PipelineLayoutCacheKey::RaderSumInterleavedF32);
        let pack_bind_group_layout =
            cached_layout(device, PipelineLayoutCacheKey::RaderPackInterleavedF32);
        let mul_bind_group_layout =
            cached_layout(device, PipelineLayoutCacheKey::RaderMulInterleavedF32);
        let write_y0_bind_group_layout =
            cached_layout(device, PipelineLayoutCacheKey::RaderWriteY0InterleavedF32);
        let post_bind_group_layout =
            cached_layout(device, PipelineLayoutCacheKey::RaderPostInterleavedF32);

        let sum_pipeline = cached_pipeline(device, &keys.sum)?;
        let pack_pipeline = cached_pipeline(device, &keys.pack)?;
        let mul_pipeline = cached_pipeline(device, &keys.mul)?;
        let write_y0_pipeline = cached_pipeline(device, &keys.write_y0)?;
        let post_pipeline = cached_pipeline(device, &keys.post)?;

        Ok(Self {
            n,
            m,
            lines: lines_u32,
            workgroups_sum: lines_u32,
            workgroups_lines: lines_u32.div_ceil(WORKGROUP_SIZE),
            workgroups_work: total_work_u32.div_ceil(WORKGROUP_SIZE),
            workgroups_tail: (lines_u32 * l as u32).div_ceil(WORKGROUP_SIZE),
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
            lines_params_buffer,
            total_params_buffer,
            perm_buffer,
            bfft_buffer,
            sum_buffer,
            x0_buffer,
            work_buffer,
            fft_buffer,
            work_fft_forward,
            work_fft_inverse,
        })
    }

    pub(crate) fn workspace_size_bytes(&self) -> u64 {
        0
    }

    pub(crate) fn graph_forward_fft_stage_kinds(&self) -> Vec<AxisStageKind> {
        self.work_fft_forward.graph_stage_kinds()
    }

    pub(crate) fn graph_forward_fft_workspace_bytes(&self) -> u64 {
        self.work_fft_forward.workspace_size_bytes()
    }

    pub(crate) fn graph_inverse_fft_stage_kinds(&self) -> Vec<AxisStageKind> {
        self.work_fft_inverse.graph_stage_kinds()
    }

    pub(crate) fn graph_inverse_fft_workspace_bytes(&self) -> u64 {
        self.work_fft_inverse.workspace_size_bytes()
    }

    pub(crate) fn graph_helper_buffers(&self) -> [HelperBufferRange; 6] {
        [
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
                format: ElementFormat::ComplexF32,
            },
            HelperBufferRange {
                label: "rader-sum-helper",
                index: 2,
                size_bytes: self.sum_buffer.size(),
                format: ElementFormat::ComplexF32,
            },
            HelperBufferRange {
                label: "rader-x0-helper",
                index: 3,
                size_bytes: self.x0_buffer.size(),
                format: ElementFormat::ComplexF32,
            },
            HelperBufferRange {
                label: "rader-work-helper",
                index: 4,
                size_bytes: self.work_buffer.size(),
                format: ElementFormat::ComplexF32,
            },
            HelperBufferRange {
                label: "rader-fft-helper",
                index: 5,
                size_bytes: self.fft_buffer.size(),
                format: ElementFormat::ComplexF32,
            },
        ]
    }

    pub(crate) fn execute_views(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        input: BufferView<'_>,
        output: BufferView<'_>,
    ) -> Result<()> {
        debug_assert!(self.n > 1);
        debug_assert!(self.m >= 2 * (self.n - 1) - 1);
        debug_assert!(self.lines > 0);

        self.dispatch_sum(device, encoder, input.clone())?;
        self.dispatch_pack(device, encoder, input)?;
        self.work_fft_forward
            .execute(device, encoder, &self.work_buffer, &self.fft_buffer)?;
        self.dispatch_mul(device, encoder)?;
        self.work_fft_inverse
            .execute(device, encoder, &self.fft_buffer, &self.work_buffer)?;
        self.dispatch_write_y0(device, encoder, output.clone())?;
        self.dispatch_post(device, encoder, output)?;
        Ok(())
    }

    fn dispatch_sum(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        input: BufferView<'_>,
    ) -> Result<()> {
        let scheduler = WindowScheduler::for_device(device);
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("wgpu_fft.rader.sum.bind_group"),
            layout: &self.sum_bind_group_layout,
            entries: &[
                bind_view_entry(&scheduler, 0, input)?,
                bind_storage_entry(&scheduler, 1, &self.sum_buffer, ElementFormat::ComplexF32)?,
                bind_storage_entry(&scheduler, 2, &self.x0_buffer, ElementFormat::ComplexF32)?,
                bind_uniform_entry(3, &self.lines_params_buffer),
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("wgpu_fft.rader.sum.pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.sum_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let (x, y, z) =
            split_workgroups(self.workgroups_sum, max_workgroups_per_dimension(device))?;
        pass.dispatch_workgroups(x, y, z);
        Ok(())
    }

    fn dispatch_pack(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        input: BufferView<'_>,
    ) -> Result<()> {
        let scheduler = WindowScheduler::for_device(device);
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("wgpu_fft.rader.pack.bind_group"),
            layout: &self.pack_bind_group_layout,
            entries: &[
                bind_view_entry(&scheduler, 0, input)?,
                bind_storage_entry(&scheduler, 1, &self.work_buffer, ElementFormat::ComplexF32)?,
                bind_storage_entry(&scheduler, 2, &self.perm_buffer, ElementFormat::U32)?,
                bind_uniform_entry(3, &self.lines_params_buffer),
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("wgpu_fft.rader.pack.pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pack_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let (x, y, z) =
            split_workgroups(self.workgroups_work, max_workgroups_per_dimension(device))?;
        pass.dispatch_workgroups(x, y, z);
        Ok(())
    }

    fn dispatch_mul(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
    ) -> Result<()> {
        let scheduler = WindowScheduler::for_device(device);
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("wgpu_fft.rader.mul.bind_group"),
            layout: &self.mul_bind_group_layout,
            entries: &[
                bind_storage_entry(&scheduler, 0, &self.fft_buffer, ElementFormat::ComplexF32)?,
                bind_storage_entry(&scheduler, 1, &self.bfft_buffer, ElementFormat::ComplexF32)?,
                bind_uniform_entry(2, &self.total_params_buffer),
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("wgpu_fft.rader.mul.pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.mul_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let (x, y, z) =
            split_workgroups(self.workgroups_work, max_workgroups_per_dimension(device))?;
        pass.dispatch_workgroups(x, y, z);
        Ok(())
    }

    fn dispatch_write_y0(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        output: BufferView<'_>,
    ) -> Result<()> {
        let scheduler = WindowScheduler::for_device(device);
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("wgpu_fft.rader.write_y0.bind_group"),
            layout: &self.write_y0_bind_group_layout,
            entries: &[
                bind_storage_entry(&scheduler, 0, &self.sum_buffer, ElementFormat::ComplexF32)?,
                bind_view_entry(&scheduler, 1, output)?,
                bind_uniform_entry(2, &self.lines_params_buffer),
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("wgpu_fft.rader.write_y0.pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.write_y0_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let (x, y, z) =
            split_workgroups(self.workgroups_lines, max_workgroups_per_dimension(device))?;
        pass.dispatch_workgroups(x, y, z);
        Ok(())
    }

    fn dispatch_post(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        output: BufferView<'_>,
    ) -> Result<()> {
        let scheduler = WindowScheduler::for_device(device);
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("wgpu_fft.rader.post.bind_group"),
            layout: &self.post_bind_group_layout,
            entries: &[
                bind_storage_entry(&scheduler, 0, &self.work_buffer, ElementFormat::ComplexF32)?,
                bind_storage_entry(&scheduler, 1, &self.x0_buffer, ElementFormat::ComplexF32)?,
                bind_storage_entry(&scheduler, 2, &self.perm_buffer, ElementFormat::U32)?,
                bind_view_entry(&scheduler, 3, output)?,
                bind_uniform_entry(4, &self.lines_params_buffer),
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("wgpu_fft.rader.post.pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.post_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let (x, y, z) =
            split_workgroups(self.workgroups_tail, max_workgroups_per_dimension(device))?;
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
        scale: f32,
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
                WORKGROUP_SIZE,
                apply_scale,
                scale,
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
    let sign = match direction {
        FftDirection::Forward => 1.0,
        FftDirection::Inverse => -1.0,
    };
    let mut values = vec![Complex32::default(); m];
    for (k, &index) in perm.iter().enumerate() {
        let angle = sign * (-std::f32::consts::TAU * index as f32 / n as f32);
        let (sin, cos) = angle.sin_cos();
        values[k] = Complex32::new(cos, sin);
    }

    let config = crate::config::FftConfig::new(m).with_normalization(Normalization::None);
    reference_c2c_nd(&values, &config)
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
) -> Result<wgpu::BindGroupEntry<'a>> {
    Ok(wgpu::BindGroupEntry {
        binding,
        resource: scheduler.storage_binding_resource(&view, ElementFormat::ComplexF32)?,
    })
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
  var acc: vec2<f32> = vec2<f32>(0.0, 0.0);
  var i: u32 = lid.x;
  loop {{
    if (i >= N) {{
      break;
    }}
    acc = acc + input[base + i * STRIDE];
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
      scratch[lid.x] = scratch[lid.x] + scratch[lid.x + stride];
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
    )
}

fn generate_rader_pack_wgsl(key: &RaderStageKey) -> String {
    let line_base_fn = wgsl_line_base_fn(key.rank, key.axis, &key.dims);
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
    work[dst] = vec2<f32>(0.0, 0.0);
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
    )
}

fn generate_rader_mul_wgsl(key: &RaderStageKey) -> String {
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
    )
}

fn generate_rader_write_y0_wgsl(key: &RaderStageKey) -> String {
    let line_base_fn = wgsl_line_base_fn(key.rank, key.axis, &key.dims);
    let scale = format_wgsl_f32(key.scale_factor());
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

const SCALE: f32 = {scale};

{line_base_fn}

@compute @workgroup_size({workgroup_size}, 1, 1)
fn main({entry_params}) {{
  {flat_index}

  let base: u32 = line_base(params.lineOffset + lineLocal);
  output[base] = sumAll[lineLocal] * vec2<f32>(SCALE, SCALE);
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
    )
}

fn generate_rader_post_wgsl(key: &RaderStageKey) -> String {
    let line_base_fn = wgsl_line_base_fn(key.rank, key.axis, &key.dims);
    let scale = format_wgsl_f32(key.scale_factor());
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
const SCALE: f32 = {scale};

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
    value = value + conv[baseWork + wrap];
  }}

  let baseOutput: u32 = line_base(params.lineOffset + lineLocal);
  let outputIndex: u32 = perm[t];
  output[baseOutput + outputIndex * STRIDE] =
    (x0[lineLocal] + value) * vec2<f32>(SCALE, SCALE);
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
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rader_permutation_covers_nonzero_prime_indices() {
        let mut values = rader_permutation(17).unwrap();
        values.sort_unstable();
        assert_eq!(values, (1..17).collect::<Vec<_>>());
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
        );
        let wgsl = generate_rader_wgsl_for_key(&key);
        assert!(wgsl.contains("const N: u32 = 17u;"));
        assert!(wgsl.contains("const L: u32 = 16u;"));
        assert!(wgsl.contains("const M: u32 = 35u;"));
        assert!(wgsl.contains("const STRIDE: u32 = 4u;"));
        assert!(wgsl.contains("let lines_per_batch: u32 = 4u;"));
    }
}
