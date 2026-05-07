use bytemuck::{Pod, Zeroable};

use crate::config::{FftDirection, Normalization};
use crate::error::{FftError, Result};
use crate::math::{reference_c2c_nd_f64, to_interleaved_f32, Complex32, Complex64};
use crate::runtime::axis_plan::{
    AxisLayout, AxisPlan, AxisPlanConfig, AxisPrecision, AxisStageKind, AxisTwiddleLutPool,
};
use crate::runtime::axis_policy::next_smooth_at_least;
use crate::runtime::buffer_view::BufferView;
use crate::runtime::dispatch::{max_workgroups_per_dimension, split_workgroups};
use crate::runtime::large_graph::{ElementFormat, HelperBufferRange};
use crate::runtime::nd_wgsl::{
    format_wgsl_f32, lines_per_batch, product, stride_for_axis, wgsl_line_base_fn,
};
use crate::runtime::window_scheduler::WindowScheduler;

const WORKGROUP_SIZE: u32 = 64;
const COMPLEX_F32_BYTES: u64 = 8;

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
}

pub(crate) struct BluesteinAxis {
    n: usize,
    m: usize,
    lines: u32,
    workgroups_work: u32,
    workgroups_output: u32,
    pack_pipeline: wgpu::ComputePipeline,
    pack_bind_group_layout: wgpu::BindGroupLayout,
    mul_pipeline: wgpu::ComputePipeline,
    mul_bind_group_layout: wgpu::BindGroupLayout,
    post_pipeline: wgpu::ComputePipeline,
    post_bind_group_layout: wgpu::BindGroupLayout,
    lines_params_buffer: wgpu::Buffer,
    total_params_buffer: wgpu::Buffer,
    chirp_buffer: wgpu::Buffer,
    bfft_buffer: wgpu::Buffer,
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

impl BluesteinAxis {
    pub(crate) fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        config: BluesteinAxisConfig,
    ) -> Result<Self> {
        config.validate()?;

        let n = config.shape[config.axis];
        let m = next_smooth_at_least(2 * n - 1);
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

        let chirp = bluestein_chirp(n, config.direction);
        let bfft = bluestein_bfft(n, m, config.direction)?;

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

        let chirp_values = to_interleaved_f32(&chirp);
        let chirp_buffer = storage_buffer(
            device,
            "wgpu_fft.bluestein.chirp",
            n as u64 * COMPLEX_F32_BYTES,
            wgpu::BufferUsages::COPY_DST,
        )?;
        queue.write_buffer(&chirp_buffer, 0, bytemuck::cast_slice(&chirp_values));

        let bfft_values = to_interleaved_f32(&bfft);
        let bfft_buffer = storage_buffer(
            device,
            "wgpu_fft.bluestein.bfft",
            m as u64 * COMPLEX_F32_BYTES,
            wgpu::BufferUsages::COPY_DST,
        )?;
        queue.write_buffer(&bfft_buffer, 0, bytemuck::cast_slice(&bfft_values));

        let work_bytes = work_complex as u64 * COMPLEX_F32_BYTES;
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
                precision: AxisPrecision::F32,
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
                precision: AxisPrecision::F32,
            },
            &mut twiddle_lut_pool,
        )?;

        let pack_bind_group_layout = bind_group_layout(
            device,
            "wgpu_fft.bluestein.pack.bind_group_layout",
            &[
                storage_entry(0, true),
                storage_entry(1, false),
                storage_entry(2, true),
                uniform_entry(3),
            ],
        );
        let pack_pipeline = compute_pipeline(
            device,
            "wgpu_fft.bluestein.pack.pipeline",
            &pack_bind_group_layout,
            &generate_bluestein_pack_wgsl(
                config.shape.len(),
                config.axis,
                &config.shape,
                n,
                m,
                stride_complex,
            ),
        );

        let mul_bind_group_layout = bind_group_layout(
            device,
            "wgpu_fft.bluestein.mul.bind_group_layout",
            &[
                storage_entry(0, false),
                storage_entry(1, true),
                uniform_entry(2),
            ],
        );
        let mul_pipeline = compute_pipeline(
            device,
            "wgpu_fft.bluestein.mul.pipeline",
            &mul_bind_group_layout,
            &generate_bluestein_mul_wgsl(m),
        );

        let post_bind_group_layout = bind_group_layout(
            device,
            "wgpu_fft.bluestein.post.bind_group_layout",
            &[
                storage_entry(0, true),
                storage_entry(1, true),
                storage_entry(2, false),
                uniform_entry(3),
            ],
        );
        let post_pipeline = compute_pipeline(
            device,
            "wgpu_fft.bluestein.post.pipeline",
            &post_bind_group_layout,
            &generate_bluestein_post_wgsl(
                config.shape.len(),
                config.axis,
                &config.shape,
                n,
                m,
                stride_complex,
                scale,
            ),
        );

        Ok(Self {
            n,
            m,
            lines: lines_u32,
            workgroups_work: total_work_u32.div_ceil(WORKGROUP_SIZE),
            workgroups_output: (lines_u32 * n as u32).div_ceil(WORKGROUP_SIZE),
            pack_pipeline,
            pack_bind_group_layout,
            mul_pipeline,
            mul_bind_group_layout,
            post_pipeline,
            post_bind_group_layout,
            lines_params_buffer,
            total_params_buffer,
            chirp_buffer,
            bfft_buffer,
            work_buffer,
            fft_buffer,
            work_fft_forward,
            work_fft_inverse,
        })
    }

    pub(crate) fn workspace_size_bytes(&self) -> u64 {
        0
    }

    pub(crate) fn twiddle_lut_storage_bytes(&self) -> u64 {
        let bytes = self.work_fft_forward.twiddle_lut_storage_bytes();
        debug_assert_eq!(bytes, self.work_fft_inverse.twiddle_lut_storage_bytes());
        bytes
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

    pub(crate) fn graph_helper_buffers(&self) -> [HelperBufferRange; 4] {
        [
            HelperBufferRange {
                label: "bluestein-chirp-helper",
                index: 0,
                size_bytes: self.chirp_buffer.size(),
                format: ElementFormat::ComplexF32,
            },
            HelperBufferRange {
                label: "bluestein-bfft-helper",
                index: 1,
                size_bytes: self.bfft_buffer.size(),
                format: ElementFormat::ComplexF32,
            },
            HelperBufferRange {
                label: "bluestein-work-helper",
                index: 2,
                size_bytes: self.work_buffer.size(),
                format: ElementFormat::ComplexF32,
            },
            HelperBufferRange {
                label: "bluestein-fft-helper",
                index: 3,
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
        debug_assert!(self.m >= 2 * self.n - 1);
        debug_assert!(self.lines > 0);

        self.dispatch_pack(device, encoder, input)?;
        self.work_fft_forward
            .execute(device, encoder, &self.work_buffer, &self.fft_buffer)?;
        self.dispatch_mul(device, encoder)?;
        self.work_fft_inverse
            .execute(device, encoder, &self.fft_buffer, &self.work_buffer)?;
        self.dispatch_post(device, encoder, output)?;
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
            label: Some("wgpu_fft.bluestein.pack.bind_group"),
            layout: &self.pack_bind_group_layout,
            entries: &[
                bind_view_entry(&scheduler, 0, input)?,
                bind_storage_entry(&scheduler, 1, &self.work_buffer, ElementFormat::ComplexF32)?,
                bind_storage_entry(&scheduler, 2, &self.chirp_buffer, ElementFormat::ComplexF32)?,
                bind_uniform_entry(3, &self.lines_params_buffer),
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("wgpu_fft.bluestein.pack.pass"),
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
            label: Some("wgpu_fft.bluestein.mul.bind_group"),
            layout: &self.mul_bind_group_layout,
            entries: &[
                bind_storage_entry(&scheduler, 0, &self.fft_buffer, ElementFormat::ComplexF32)?,
                bind_storage_entry(&scheduler, 1, &self.bfft_buffer, ElementFormat::ComplexF32)?,
                bind_uniform_entry(2, &self.total_params_buffer),
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("wgpu_fft.bluestein.mul.pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.mul_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let (x, y, z) =
            split_workgroups(self.workgroups_work, max_workgroups_per_dimension(device))?;
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
            label: Some("wgpu_fft.bluestein.post.bind_group"),
            layout: &self.post_bind_group_layout,
            entries: &[
                bind_storage_entry(&scheduler, 0, &self.work_buffer, ElementFormat::ComplexF32)?,
                bind_storage_entry(&scheduler, 1, &self.chirp_buffer, ElementFormat::ComplexF32)?,
                bind_view_entry(&scheduler, 2, output)?,
                bind_uniform_entry(3, &self.lines_params_buffer),
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("wgpu_fft.bluestein.post.pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.post_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let (x, y, z) =
            split_workgroups(self.workgroups_output, max_workgroups_per_dimension(device))?;
        pass.dispatch_workgroups(x, y, z);
        Ok(())
    }
}

pub(crate) fn bluestein_chirp(n: usize, direction: FftDirection) -> Vec<Complex32> {
    let sign = transform_sign(direction);
    (0..n)
        .map(|i| round_complex64(bluestein_phase(n, i, sign)))
        .collect()
}

pub(crate) fn bluestein_bfft(
    n: usize,
    m: usize,
    direction: FftDirection,
) -> Result<Vec<Complex32>> {
    let sign = transform_sign(direction);
    let mut values = vec![Complex64::default(); m];
    values[0] = Complex64::new(1.0, 0.0);
    for i in 1..n {
        let value = bluestein_phase(n, i, -sign);
        values[i] = value;
        values[m - i] = value;
    }

    let config = crate::config::FftConfig::new(m).with_normalization(Normalization::None);
    Ok(reference_c2c_nd_f64(&values, &config)?
        .into_iter()
        .map(round_complex64)
        .collect())
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

fn bind_group_layout(
    device: &wgpu::Device,
    label: &'static str,
    entries: &[wgpu::BindGroupLayoutEntry],
) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries,
    })
}

fn compute_pipeline(
    device: &wgpu::Device,
    label: &'static str,
    bind_group_layout: &wgpu::BindGroupLayout,
    wgsl: &str,
) -> wgpu::ComputePipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(wgsl.into()),
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &[Some(bind_group_layout)],
        immediate_size: 0,
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    })
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
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

fn complex_mul_wgsl() -> &'static str {
    r#"fn c_mul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
  return vec2<f32>(
    a.x * b.x - a.y * b.y,
    a.x * b.y + a.y * b.x
  );
}"#
}

fn generate_bluestein_pack_wgsl(
    rank: usize,
    axis: usize,
    shape: &[usize],
    n: usize,
    m: usize,
    stride_complex: usize,
) -> String {
    let line_base_fn = wgsl_line_base_fn(rank, axis, shape);
    let complex_mul = complex_mul_wgsl();
    format!(
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
        n = n,
        m = m,
        stride = stride_complex,
        complex_mul = complex_mul,
        line_base_fn = line_base_fn,
        workgroup_size = WORKGROUP_SIZE,
        entry_params = crate::runtime::dispatch::WGSL_FLAT_ENTRY_PARAMS,
        flat_index = crate::runtime::dispatch::wgsl_flat_index_stmts(
            "i",
            "params.lines * M",
            WORKGROUP_SIZE,
        ),
    )
}

fn generate_bluestein_mul_wgsl(m: usize) -> String {
    let complex_mul = complex_mul_wgsl();
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

{complex_mul}

@compute @workgroup_size({workgroup_size}, 1, 1)
fn main({entry_params}) {{
  {flat_index}
  let t: u32 = i - (i / M) * M;
  work[i] = c_mul(work[i], bfft[t]);
}}
"#,
        m = m,
        complex_mul = complex_mul,
        workgroup_size = WORKGROUP_SIZE,
        entry_params = crate::runtime::dispatch::WGSL_FLAT_ENTRY_PARAMS,
        flat_index =
            crate::runtime::dispatch::wgsl_flat_index_stmts("i", "params.total", WORKGROUP_SIZE),
    )
}

fn generate_bluestein_post_wgsl(
    rank: usize,
    axis: usize,
    shape: &[usize],
    n: usize,
    m: usize,
    stride_complex: usize,
    scale: f32,
) -> String {
    let line_base_fn = wgsl_line_base_fn(rank, axis, shape);
    let complex_mul = complex_mul_wgsl();
    let scale = format_wgsl_f32(scale);
    format!(
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
const SCALE: f32 = {scale};

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
        n = n,
        m = m,
        stride = stride_complex,
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
    )
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let wgsl = generate_bluestein_pack_wgsl(2, 1, &[4, 14], 14, 27, 4);
        assert!(wgsl.contains("const N: u32 = 14u;"));
        assert!(wgsl.contains("const M: u32 = 27u;"));
        assert!(wgsl.contains("const STRIDE: u32 = 4u;"));
        assert!(wgsl.contains("let lines_per_batch: u32 = 4u;"));
    }
}
