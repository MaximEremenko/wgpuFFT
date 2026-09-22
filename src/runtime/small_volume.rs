//! Small C2C volumes transformed whole in one workgroup: when every axis of
//! a volume is transformed and the volume fits workgroup memory, one kernel
//! per FFT replaces one kernel per axis, so a small 2D or 3D transform pays
//! for one dispatch instead of two or three.
//!
//! A workgroup loads its volume into workgroup memory, transforms each axis
//! in place over all of the axis's lines, and stores the volume. Axes whose
//! length factors into radices up to 13 run the fused smooth stages; odd
//! lengths up to `FftTuning::direct_max_prime` run a direct DFT over
//! symmetric pairs, as the direct prime kernel does. Twiddles come from one
//! table of `L = lcm(dims)` points: axis `a` reads `W_L^(i L / n_a)`.

use bytemuck::{Pod, Zeroable};

use crate::config::{FftConfig, FftDirection, FftPrecision};
use crate::error::Result;
use crate::runtime::axis_plan::{
    complex_wgsl, fused_smooth_factors, generate_fused_smooth_butterfly_math_wgsl, rebind_in_place,
    scaled_complex_expr, AxisPrecision,
};
use crate::runtime::buffer_view::BufferView;
use crate::runtime::dispatch::{max_workgroups_per_dimension, split_workgroups};
use crate::runtime::pipeline_cache::{
    with_device_pipeline_cache, ComputePipelineCacheKey, LazyPipeline, SmallVolumeAxis,
    SmallVolumeKey,
};
use crate::runtime::recorder::CommandRecorder;
use crate::runtime::twiddle::create_twiddle_lut_buffer_for_len_with_precision;
use crate::runtime::window_scheduler::WindowScheduler;
use crate::tuning::FftLargeRoute;

/// Largest volume one small-volume workgroup transforms:
/// 64x64 ran as fast in one workgroup as in one kernel per axis, and
/// smaller volumes up to 64% faster (8x8x8).
const MAX_SMALL_VOLUME_ELEMENTS: usize = 4096;
/// Most workgroup-memory reads the direct axes may take, the volume times
/// the sum of their lengths: one workgroup reads every point of a line per
/// output, so 31x31 ran 24% faster in one workgroup and 41x41 14% slower.
const MAX_DIRECT_READS: usize = 1 << 16;
/// Longest twiddle table, `lcm` of the axis lengths.
const MAX_TWIDDLE_LENGTH: usize = 1 << 16;

/// The axis plans' parameters: the kernel transforms `total / ELEMENTS`
/// volumes from volume `line_offset`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct SmallVolumeParams {
    total: u32,
    base_index: u32,
    line_offset: u32,
    element_base: u32,
}

/// One kernel that transforms every axis of each volume of a small C2C
/// transform, one workgroup per volume.
pub(crate) struct SmallVolumePlan {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    in_place: LazyPipeline,
    params_buffer: wgpu::Buffer,
    twiddle_buffer: wgpu::Buffer,
    volumes: u32,
}

impl SmallVolumePlan {
    /// The plan of `config` when its whole volume fits one workgroup.
    pub(crate) fn try_new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        config: &FftConfig,
    ) -> Result<Option<Self>> {
        let Some(key) = small_volume_key(config, &device.limits())? else {
            return Ok(None);
        };
        let volumes = u32::try_from(config.batch()).unwrap_or(u32::MAX);
        let elements = key.dims.iter().product::<usize>() as u32;
        let pipeline_key = ComputePipelineCacheKey::small_volume(key.clone());
        let bind_group_layout = with_device_pipeline_cache(device, |cache| {
            cache.get_bind_group_layout(device, pipeline_key.layout)
        });
        let pipeline = with_device_pipeline_cache(device, |cache| {
            cache.get_compute_pipeline(
                device,
                &pipeline_key,
                "wgpu_fft.small_volume.pipeline",
                "wgpu_fft.small_volume.shader",
                || generate_small_volume_wgsl_for_key(&key),
            )
        });
        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("wgpu_fft.small_volume.params"),
            size: std::mem::size_of::<SmallVolumeParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(
            &params_buffer,
            0,
            bytemuck::bytes_of(&SmallVolumeParams {
                total: volumes.saturating_mul(elements),
                base_index: 0,
                line_offset: 0,
                element_base: 0,
            }),
        );
        let twiddle_buffer = create_twiddle_lut_buffer_for_len_with_precision(
            device,
            queue,
            "wgpu_fft.small_volume.twiddle_lut",
            key.twiddle_length,
            FftPrecision::F32,
        )?;
        Ok(Some(Self {
            pipeline,
            bind_group_layout,
            in_place: LazyPipeline::new(
                "wgpu_fft.small_volume.in_place",
                ComputePipelineCacheKey::small_volume(key.with_in_place()),
            ),
            params_buffer,
            twiddle_buffer,
            volumes,
        }))
    }

    /// Transforms `buffer` in place.
    pub(crate) fn execute_in_place(
        &self,
        device: &wgpu::Device,
        encoder: &mut CommandRecorder<'_>,
        buffer: BufferView<'_>,
    ) -> Result<()> {
        let (pipeline, layout) = self.in_place.get(device);
        let scheduler = WindowScheduler::for_device(device);
        let element_format = AxisPrecision::F32.element_format();
        let buffer_resource = scheduler.storage_binding_resource(&buffer, element_format)?;
        let twiddle_view = BufferView::whole(&self.twiddle_buffer);
        let twiddle_resource = scheduler.storage_binding_resource(&twiddle_view, element_format)?;
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("wgpu_fft.small_volume.in_place.bind_group"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffer_resource,
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: twiddle_resource,
                },
            ],
        });
        let pass = encoder.pass();
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let (x, y, z) = split_workgroups(self.volumes, max_workgroups_per_dimension(device))?;
        pass.dispatch_workgroups(x, y, z);
        Ok(())
    }

    pub(crate) fn twiddle_lut_storage_bytes(&self) -> u64 {
        self.twiddle_buffer.size()
    }

    pub(crate) fn workspace_size_bytes(&self) -> u64 {
        0
    }

    pub(crate) fn execute_views(
        &self,
        device: &wgpu::Device,
        encoder: &mut CommandRecorder<'_>,
        input: BufferView<'_>,
        output: BufferView<'_>,
    ) -> Result<()> {
        let scheduler = WindowScheduler::for_device(device);
        let element_format = AxisPrecision::F32.element_format();
        let input_resource = scheduler.storage_binding_resource(&input, element_format)?;
        let output_resource = scheduler.storage_binding_resource(&output, element_format)?;
        let twiddle_view = BufferView::whole(&self.twiddle_buffer);
        let twiddle_resource = scheduler.storage_binding_resource(&twiddle_view, element_format)?;
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("wgpu_fft.small_volume.bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_resource,
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output_resource,
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: twiddle_resource,
                },
            ],
        });
        let pass = encoder.pass();
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let (x, y, z) = split_workgroups(self.volumes, max_workgroups_per_dimension(device))?;
        pass.dispatch_workgroups(x, y, z);
        Ok(())
    }
}

/// The small-volume kernel of `config`, when it is an `f32` transform of
/// every axis of a volume that fits workgroup memory and
/// [`MAX_SMALL_VOLUME_ELEMENTS`], every axis's length factors into radices up
/// to 13 or is odd and at most `FftTuning::direct_max_prime` (within
/// [`MAX_DIRECT_READS`]), `FftTuning::fuse_small_volumes` allows it, and no
/// tuning forces other routes.
pub(crate) fn small_volume_key(
    config: &FftConfig,
    limits: &wgpu::Limits,
) -> Result<Option<SmallVolumeKey>> {
    let shape = config.shape();
    let tuning = config.tuning();
    let mut axes = config.axes().to_vec();
    axes.sort_unstable();
    if config.precision() != FftPrecision::F32
        || shape.len() < 2
        || axes != (0..shape.len()).collect::<Vec<_>>()
        || !tuning.force_rader_axes().is_empty()
        || !tuning.force_bluestein_axes().is_empty()
        || tuning.large_route() != FftLargeRoute::Auto
        || !tuning.fuse_small_volumes()
    {
        return Ok(None);
    }
    let elements = shape.iter().product::<usize>();
    if elements > MAX_SMALL_VOLUME_ELEMENTS {
        return Ok(None);
    }
    let Some(kinds) = volume_axes(shape, tuning.direct_max_prime()) else {
        return Ok(None);
    };
    let direct_lengths = shape
        .iter()
        .zip(&kinds)
        .filter(|(_, kind)| **kind == SmallVolumeAxis::Direct)
        .map(|(&n, _)| n)
        .sum::<usize>();
    if elements.saturating_mul(direct_lengths) > MAX_DIRECT_READS {
        return Ok(None);
    }
    let twiddle_length = shape.iter().fold(1usize, |lcm, &n| lcm / gcd(lcm, n) * n);
    if twiddle_length > MAX_TWIDDLE_LENGTH {
        return Ok(None);
    }
    let scale = config.scale()?;
    let key = SmallVolumeKey::new(
        shape,
        kinds,
        twiddle_length,
        config.direction(),
        tuning.fused_workgroup_size(),
        scale != 1.0,
        f64::from(scale),
    );
    Ok(key.supported_by_device_limits(limits).then_some(key))
}

/// How each axis of a small volume runs: stages of radices up to 16 (one
/// butterfly for lengths up to 16), or a direct DFT for odd lengths up to
/// `direct_max`; `None` when an axis has neither.
fn volume_axes(dims: &[usize], direct_max: usize) -> Option<Vec<SmallVolumeAxis>> {
    dims.iter()
        .map(|&n| {
            if let Ok(factors) = crate::runtime::factor_supported_length(n) {
                Some(if n <= 16 && n != 14 {
                    SmallVolumeAxis::Stages(vec![n])
                } else {
                    SmallVolumeAxis::Stages(fused_smooth_factors(n, &factors))
                })
            } else if n % 2 == 1 && n <= direct_max {
                Some(SmallVolumeAxis::Direct)
            } else {
                None
            }
        })
        .collect()
}

/// The leading axes of a multi-axis plan that run as one small-volume stage
/// over slabs of the volume, and its key: axes `0..k` for the largest
/// `k >= 2` whose slab fits one workgroup, when the plan transforms them
/// first. The later axes follow in place, so a 32x32x32 transform takes two
/// kernels instead of three. `apply_scale` scales the stage's output when
/// it covers every axis of the plan.
pub(crate) fn leading_slab(
    shape: &[usize],
    axes: &[usize],
    direction: FftDirection,
    workgroup_size: u32,
    apply_scale: bool,
    scale_factor: f64,
    limits: &wgpu::Limits,
) -> Option<(usize, SmallVolumeKey)> {
    let mut best = None;
    let mut elements = 1usize;
    for (k, &axis) in axes.iter().enumerate() {
        if axis != k {
            break;
        }
        elements = elements.checked_mul(shape[axis])?;
        if elements > MAX_SMALL_VOLUME_ELEMENTS {
            break;
        }
        let count = k + 1;
        if count < 2 {
            continue;
        }
        let dims = &shape[..count];
        let Some(kinds) = volume_axes(dims, 0) else {
            break;
        };
        let twiddle_length = dims.iter().fold(1usize, |lcm, &n| lcm / gcd(lcm, n) * n);
        if twiddle_length > MAX_TWIDDLE_LENGTH {
            break;
        }
        let covers_all = count == axes.len();
        let key = SmallVolumeKey::new(
            dims,
            kinds,
            twiddle_length,
            direction,
            workgroup_size,
            apply_scale && covers_all,
            if covers_all { scale_factor } else { 1.0 },
        );
        if !key.supported_by_device_limits(limits) {
            break;
        }
        best = Some((count, key));
    }
    best
}

fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

/// WGSL of a small-volume kernel.
pub(crate) fn generate_small_volume_wgsl_for_key(key: &SmallVolumeKey) -> String {
    let scale_factor = key.apply_scale.then_some(key.scale_factor());
    let dims = &key.dims;
    let elements = dims.iter().product::<usize>();
    let workgroup_size = key.workgroup_size as usize;
    let precision = AxisPrecision::F32;
    let mut functions = String::new();
    let mut tables = String::new();
    let mut table_loads = String::new();
    let mut body = String::new();
    let mut stride = 1usize;
    for (axis, (&n, kind)) in dims.iter().zip(&key.axes).enumerate() {
        debug_assert!(key.twiddle_length.is_multiple_of(n));
        let step = key.twiddle_length / n;
        let lines = elements / n;
        match kind {
            SmallVolumeAxis::Stages(radices) => {
                debug_assert_eq!(radices.iter().product::<usize>(), n);
                let value = match key.direction {
                    FftDirection::Forward => "value",
                    FftDirection::Inverse => "vec2<f32>(value.x, -value.y)",
                };
                functions.push_str(&format!(
                    "fn twiddle{axis}(index: u32) -> vec2<f32> {{\n  let value: vec2<f32> = axisTwiddles[index * {step}u];\n  return {value};\n}}\n\n"
                ));
                body.push_str(&stages_wgsl(
                    axis,
                    n,
                    stride,
                    lines,
                    radices,
                    key.direction,
                    workgroup_size,
                    precision,
                ));
            }
            SmallVolumeAxis::Direct => {
                debug_assert_eq!(n % 2, 1);
                tables.push_str(&format!(
                    "// (cos, sin) of 2 pi m / {n}.\nvar<workgroup> trig{axis}: array<vec2<f32>, {n}>;\n"
                ));
                table_loads.push_str(&format!(
                    "  for (var i: u32 = lid.x; i < {n}u; i = i + WORKGROUP_SIZE) {{\n    // The table holds W^i = (cos, -sin).\n    let root: vec2<f32> = axisTwiddles[i * {step}u];\n    trig{axis}[i] = vec2<f32>(root.x, -root.y);\n  }}\n"
                ));
                body.push_str(&direct_wgsl(
                    axis,
                    n,
                    stride,
                    lines,
                    key.direction,
                    workgroup_size,
                ));
            }
        }
        stride *= n;
    }
    // All loads issue before any store, so their latencies overlap.
    let mut loads = String::new();
    let mut stores = String::new();
    let rounds = elements.div_ceil(workgroup_size);
    for round in 0..rounds {
        let index = format!("(lid.x + {}u)", round * workgroup_size);
        let guard = if (round + 1) * workgroup_size > elements {
            format!("{index} < ELEMENTS")
        } else {
            String::from("true")
        };
        loads.push_str(&format!(
            "  var v{round}: vec2<f32> = vec2<f32>(0.0, 0.0);
  if ({guard}) {{
    v{round} = input[volumeBase + {index}];
  }}
"
        ));
        stores.push_str(&format!(
            "  if ({guard}) {{
    vol[{index}] = v{round};
  }}
"
        ));
    }
    let stored = scaled_complex_expr("vol[i]", scale_factor, precision);

    let source = format!(
        r#"struct Params {{
  total: u32,
  baseIndex: u32,
  lineOffset: u32,
  elementBase: u32,
}};

@group(0) @binding(0) var<storage, read> input: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec2<f32>>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read> axisTwiddles: array<vec2<f32>>;

{complex}

{functions}const ELEMENTS: u32 = {elements}u;
const WORKGROUP_SIZE: u32 = {workgroup_size}u;

var<workgroup> vol: array<vec2<f32>, {elements}>;
{tables}
@compute @workgroup_size({workgroup_size}, 1, 1)
fn main({entry_params}) {{
  {flat_workgroup_index}
  if (wgFlat >= params.total / ELEMENTS) {{
    return;
  }}
  let volumeBase: u32 = (params.lineOffset + wgFlat) * ELEMENTS - params.elementBase;
{loads}{stores}{table_loads}  workgroupBarrier();
{body}  for (var i: u32 = lid.x; i < ELEMENTS; i = i + WORKGROUP_SIZE) {{
    output[volumeBase + i] = {stored};
  }}
}}
"#,
        complex = complex_wgsl(),
        entry_params = crate::runtime::dispatch::WGSL_FLAT_ENTRY_PARAMS,
        flat_workgroup_index = crate::runtime::dispatch::WGSL_FLAT_WORKGROUP_INDEX,
    );
    if key.in_place {
        rebind_in_place(&source, "input", "output")
    } else {
        source
    }
}

/// Statements binding the line and unit (or task) of work item `w_{slot}` on
/// an axis of `units` items per line, and the line's first element: along
/// the line on the contiguous axis, across neighbouring lines on the others,
/// so neighbouring invocations touch neighbouring elements.
fn item_split(slot: usize, stride: usize, n: usize, lines: usize, units: usize) -> String {
    let (line, unit) = if stride == 1 {
        (
            format!("w_{slot} / {units}u"),
            format!("w_{slot} - line_{slot} * {units}u"),
        )
    } else {
        (
            format!("w_{slot} % {lines}u"),
            format!("w_{slot} / {lines}u"),
        )
    };
    format!(
        "    let line_{slot}: u32 = {line};\n    let unit_{slot}: u32 = {unit};\n    let lineStart_{slot}: u32 = (line_{slot} % {stride}u) + (line_{slot} / {stride}u) * {span}u;\n",
        span = stride * n,
    )
}

#[allow(clippy::too_many_arguments)]
fn stages_wgsl(
    axis: usize,
    n: usize,
    stride: usize,
    lines: usize,
    radices: &[usize],
    direction: FftDirection,
    workgroup_size: usize,
    precision: AxisPrecision,
) -> String {
    let mut out = String::new();
    let mut ns = 1usize;
    for &radix in radices {
        ns *= radix;
        let ns_div_r = ns / radix;
        let n_div_r = n / radix;
        let n_div_ns = n / ns;
        let units = n / radix;
        let items = lines * units;
        let slots = items.div_ceil(workgroup_size);
        let mut computes = String::new();
        let mut writes = String::new();
        for slot in 0..slots {
            let mut outputs = String::new();
            let mut slot_writes = String::new();
            for k in 0..radix {
                outputs.push_str(&format!(
                    "    var stageOut_{slot}_{k}: vec2<f32> = vec2<f32>(0.0, 0.0);\n"
                ));
                slot_writes.push_str(&format!(
                    "      vol[lineStart_{slot} + (block_{slot} * {ns}u + {k}u * {ns_div_r}u + j_{slot}) * {stride}u] = stageOut_{slot}_{k};\n"
                ));
            }
            let butterfly = generate_fused_smooth_butterfly_math_wgsl(
                radix,
                ns_div_r,
                n_div_r,
                n_div_ns,
                direction,
                slot,
                &|position| format!("vol[lineStart_{slot} + ({position}) * {stride}u]"),
                &format!("twiddle{axis}"),
                precision,
            );
            computes.push_str(&format!(
                "    let w_{slot}: u32 = lid.x + {offset}u;\n{split}    let block_{slot}: u32 = unit_{slot} / {ns_div_r}u;\n    let j_{slot}: u32 = unit_{slot} - block_{slot} * {ns_div_r}u;\n{outputs}    if (w_{slot} < {items}u) {{\n      let base_{slot}: u32 = unit_{slot};\n{butterfly}    }}\n",
                offset = slot * workgroup_size,
                split = item_split(slot, stride, n, lines, units),
            ));
            writes.push_str(&format!(
                "    if (w_{slot} < {items}u) {{\n{slot_writes}    }}\n"
            ));
        }
        out.push_str(&format!(
            "  {{ // axis {axis}: radix-{radix} butterflies\n{computes}    workgroupBarrier();\n{writes}    workgroupBarrier();\n  }}\n"
        ));
    }
    out
}

fn direct_wgsl(
    axis: usize,
    n: usize,
    stride: usize,
    lines: usize,
    direction: FftDirection,
    workgroup_size: usize,
) -> String {
    let half = (n - 1) / 2;
    let tasks = half + 1;
    let items = lines * tasks;
    let slots = items.div_ceil(workgroup_size);
    // `rotation` is the sine term's contribution to `X[k]`: `-i S` forward,
    // `+i S` inverse; `X[N - k]` takes the opposite.
    let rotation = match direction {
        FftDirection::Forward => "vec2<f32>(sine.y, -sine.x)",
        FftDirection::Inverse => "vec2<f32>(-sine.y, sine.x)",
    };
    let mut computes = String::new();
    let mut writes = String::new();
    for slot in 0..slots {
        computes.push_str(&format!(
            r#"    let w_{slot}: u32 = lid.x + {offset}u;
{split}    var low_{slot}: vec2<f32> = vec2<f32>(0.0, 0.0);
    var high_{slot}: vec2<f32> = vec2<f32>(0.0, 0.0);
    if (w_{slot} < {items}u) {{
      let first: vec2<f32> = vol[lineStart_{slot}];
      var cosine: vec2<f32> = vec2<f32>(0.0, 0.0);
      var sine: vec2<f32> = vec2<f32>(0.0, 0.0);
      var index: u32 = 0u;
      for (var j: u32 = 1u; j <= {half}u; j = j + 1u) {{
        let low: vec2<f32> = vol[lineStart_{slot} + j * {stride}u];
        let high: vec2<f32> = vol[lineStart_{slot} + ({n}u - j) * {stride}u];
        index = index + unit_{slot};
        if (index >= {n}u) {{
          index = index - {n}u;
        }}
        let angle: vec2<f32> = trig{axis}[index];
        cosine = cosine + (low + high) * angle.x;
        sine = sine + (low - high) * angle.y;
      }}
      let even: vec2<f32> = first + cosine;
      let rotation: vec2<f32> = {rotation};
      low_{slot} = even + rotation;
      high_{slot} = even - rotation;
    }}
"#,
            offset = slot * workgroup_size,
            split = item_split(slot, stride, n, lines, tasks),
        ));
        writes.push_str(&format!(
            r#"    if (w_{slot} < {items}u) {{
      vol[lineStart_{slot} + unit_{slot} * {stride}u] = low_{slot};
      if (unit_{slot} > 0u) {{
        vol[lineStart_{slot} + ({n}u - unit_{slot}) * {stride}u] = high_{slot};
      }}
    }}
"#
        ));
    }
    format!(
        "  {{ // axis {axis}: direct DFT\n{computes}    workgroupBarrier();\n{writes}    workgroupBarrier();\n  }}\n"
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tuning::FftTuning;

    fn limits(storage: u32) -> wgpu::Limits {
        wgpu::Limits {
            max_compute_workgroup_storage_size: storage,
            max_compute_invocations_per_workgroup: 256,
            max_compute_workgroup_size_x: 256,
            ..wgpu::Limits::default()
        }
    }

    fn key(config: &FftConfig, storage: u32) -> Option<SmallVolumeKey> {
        small_volume_key(config, &limits(storage)).unwrap()
    }

    #[test]
    fn small_volumes_need_every_axis_to_fit_one_workgroup() {
        let volume = key(&FftConfig::new_nd([64, 64]), 32_768).expect("64x64 fits 32 KiB");
        assert_eq!(
            volume.axes,
            [
                SmallVolumeAxis::Stages(vec![8, 8]),
                SmallVolumeAxis::Stages(vec![8, 8])
            ]
        );
        assert_eq!(volume.twiddle_length, 64);
        assert!(key(&FftConfig::new_nd([64, 64]), 16_384).is_none());
        assert!(key(&FftConfig::new_nd([64, 128]), 65_536).is_none());
        assert!(key(&FftConfig::new_nd([64, 64]).with_axes([1]), 32_768).is_none());
        assert!(key(&FftConfig::new(64), 32_768).is_none());
        assert!(key(
            &FftConfig::new_nd([64, 64])
                .with_tuning(FftTuning::default().with_fuse_small_volumes(false)),
            32_768
        )
        .is_none());

        // Odd lengths without radices up to 13 run direct DFTs within the
        // read budget: 31x31 takes 961 * 62 reads, 41x41 1681 * 82.
        let primes = key(&FftConfig::new_nd([31, 31]), 16_384).expect("31x31 fits");
        assert_eq!(
            primes.axes,
            [SmallVolumeAxis::Direct, SmallVolumeAxis::Direct]
        );
        assert!(key(&FftConfig::new_nd([41, 41]), 49_152).is_none());
        assert!(key(
            &FftConfig::new_nd([31, 31]).with_tuning(FftTuning::default().with_direct_max_prime(0)),
            16_384
        )
        .is_none());
        // Even lengths with a prime factor above 13 have no small-volume form.
        assert!(key(&FftConfig::new_nd([34, 8]), 49_152).is_none());
        let mixed = key(&FftConfig::new_nd([51, 8, 3]), 49_152).expect("51x8x3 fits");
        assert_eq!(
            mixed.axes,
            [
                SmallVolumeAxis::Direct,
                SmallVolumeAxis::Stages(vec![8]),
                SmallVolumeAxis::Stages(vec![3])
            ]
        );
        assert_eq!(mixed.twiddle_length, 408);
    }

    #[test]
    fn small_volume_kernels_write_workgroup_memory_before_reading_it() {
        for config in [
            FftConfig::new_nd([64, 64]),
            FftConfig::inverse_nd([16, 16, 16]),
            FftConfig::new_nd([51, 8, 3]),
        ] {
            let key = key(&config, 49_152).expect("fits");
            let wgsl = generate_small_volume_wgsl_for_key(&key);
            crate::runtime::assert_workgroup_var_written_before_read(&wgsl, "vol");
            for (axis, kind) in key.axes.iter().enumerate() {
                if *kind == SmallVolumeAxis::Direct {
                    crate::runtime::assert_workgroup_var_written_before_read(
                        &wgsl,
                        &format!("trig{axis}"),
                    );
                }
            }
            assert!(!wgsl.contains("sin(") && !wgsl.contains("cos("));
        }
    }
}
