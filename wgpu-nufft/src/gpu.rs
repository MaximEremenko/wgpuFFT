use std::num::NonZeroU64;

use wgpu::util::DeviceExt;
use wgpu_fft::{BufferView, FftConfig, FftDirection, FftPlan, FftPrecision, Normalization};

use crate::config::{ModeOrder, NufftConfig, NufftSign};
use crate::direct::mode_for_storage_index;
use crate::error::{NufftError, Result};
use crate::gpu_dispatch::split_workgroups;
#[cfg(feature = "gpu-profiling")]
use crate::gpu_profile::{GpuProfileQueryWriter, NufftGpuProfileLayout};
use crate::kernel::EsKernel;

const WORKGROUP_SIZE: u32 = 64;
const VECTOR_TILE: usize = 4;
const COMPLEX_F32_BYTES: u64 = 8;
const F32_BYTES: u64 = 4;

pub(crate) fn max_supported_workgroup_size(limits: &wgpu::Limits) -> u32 {
    limits
        .max_compute_invocations_per_workgroup
        .min(limits.max_compute_workgroup_size_x)
}

pub(crate) struct Type2GpuPlan {
    fft: FftPlan,
    amplitudes: wgpu::Buffer,
    fine_input: wgpu::Buffer,
    fine_output: wgpu::Buffer,
    predeconvolution_pipeline: wgpu::ComputePipeline,
    predeconvolution_layout: wgpu::BindGroupLayout,
    interpolation_pipeline: wgpu::ComputePipeline,
    interpolation_layout: wgpu::BindGroupLayout,
    predeconvolution_dispatch: (u32, u32, u32),
    max_workgroups_per_dimension: u32,
    batch_capacity: usize,
    mode_count: usize,
    max_storage_binding_bytes: u64,
}

impl Type2GpuPlan {
    pub(crate) fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        config: &NufftConfig,
        kernel: EsKernel,
        fine_length: usize,
        kernel_fourier_coefficients: &[f64],
    ) -> Result<Self> {
        if config.dimensions() != 1 {
            return Err(NufftError::GpuDimensionsUnsupported {
                kind: "type-2",
                actual: config.dimensions(),
                supported: 1,
            });
        }
        // Interpolation positions are ultimately addressed by f32 WGSL lanes.
        // Above 2^24, adjacent grid cells cannot be represented distinctly.
        let maximum_signed_length = (1usize << f32::MANTISSA_DIGITS)
            .saturating_sub(kernel.width())
            .min(i32::MAX as usize - kernel.width());
        if fine_length > maximum_signed_length {
            return Err(NufftError::GpuFineGridIndexUnsupported {
                length: fine_length,
                maximum: maximum_signed_length,
            });
        }
        let limits = device.limits();
        let maximum_workgroup_size = max_supported_workgroup_size(&limits);
        if maximum_workgroup_size < WORKGROUP_SIZE {
            return Err(NufftError::GpuWorkgroupSizeUnsupported {
                requested: WORKGROUP_SIZE,
                maximum: maximum_workgroup_size,
            });
        }

        let batch_capacity = config.batch();
        let mode_count = config.mode_count()?;
        let fine_element_count = checked_product(
            "type-2 batched fine-grid element count",
            fine_length,
            batch_capacity,
        )?;
        let fine_element_count_u32 =
            u32::try_from(fine_element_count).map_err(|_| NufftError::LengthOverflow {
                context: "type-2 batched fine-grid shader index space",
            })?;
        let fine_bytes = Self::complex_buffer_size_bytes("type-2 fine grid", fine_element_count)?;
        let amplitude_bytes =
            checked_buffer_size("type-2 deconvolution amplitudes", mode_count, F32_BYTES)?;
        let max_storage_binding_bytes = limits.max_storage_buffer_binding_size;
        validate_binding_limit("type-2 fine grid", fine_bytes, max_storage_binding_bytes)?;
        validate_binding_limit(
            "type-2 deconvolution amplitudes",
            amplitude_bytes,
            max_storage_binding_bytes,
        )?;
        if fine_bytes > limits.max_buffer_size {
            return Err(NufftError::GpuBufferBindingTooLarge {
                buffer: "type-2 fine grid",
                required_bytes: fine_bytes,
                limit_bytes: limits.max_buffer_size,
            });
        }

        let amplitudes = mode_amplitudes(config, kernel_fourier_coefficients)?;
        let amplitude_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("wgpu_nufft.type2.deconvolution_amplitudes"),
            contents: bytemuck::cast_slice(&amplitudes),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let fine_usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST;
        let fine_input = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("wgpu_nufft.type2.fine_input"),
            size: fine_bytes,
            usage: fine_usage,
            mapped_at_creation: false,
        });
        let fine_output = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("wgpu_nufft.type2.fine_output"),
            size: fine_bytes,
            usage: fine_usage,
            mapped_at_creation: false,
        });

        let fft_direction = match config.sign() {
            NufftSign::Positive => FftDirection::Inverse,
            NufftSign::Negative => FftDirection::Forward,
        };
        let fft_config = FftConfig::new(fine_length)
            .with_direction(fft_direction)
            .with_normalization(Normalization::None)
            .with_batch(batch_capacity)
            .with_precision(FftPrecision::F32);
        let fft = FftPlan::c2c(device, queue, fft_config).map_err(|source| {
            NufftError::FftShapeUnsupported {
                stage: "type-2 oversampled-grid C2C plan",
                source,
            }
        })?;

        let predeconvolution_pipeline = create_compute_pipeline(
            device,
            "wgpu_nufft.type2.predeconvolution",
            &generate_predeconvolution_wgsl(config, fine_length),
        );
        let predeconvolution_layout = predeconvolution_pipeline.get_bind_group_layout(0);
        let interpolation_pipeline = create_compute_pipeline(
            device,
            "wgpu_nufft.type2.interpolation",
            &generate_interpolation_wgsl(kernel, fine_length),
        );
        let interpolation_layout = interpolation_pipeline.get_bind_group_layout(0);

        let workgroups = fine_element_count_u32.div_ceil(WORKGROUP_SIZE);
        let max_workgroups_per_dimension = limits.max_compute_workgroups_per_dimension;
        let predeconvolution_dispatch = split_workgroups(workgroups, max_workgroups_per_dimension)?;

        Ok(Self {
            fft,
            amplitudes: amplitude_buffer,
            fine_input,
            fine_output,
            predeconvolution_pipeline,
            predeconvolution_layout,
            interpolation_pipeline,
            interpolation_layout,
            predeconvolution_dispatch,
            max_workgroups_per_dimension,
            batch_capacity,
            mode_count,
            max_storage_binding_bytes,
        })
    }

    pub(crate) fn point_buffer_size_bytes(point_count: usize) -> Result<u64> {
        checked_buffer_size("type-2 point buffer", point_count, F32_BYTES)
    }

    pub(crate) fn complex_buffer_size_bytes(
        buffer: &'static str,
        element_count: usize,
    ) -> Result<u64> {
        checked_buffer_size(buffer, element_count, COMPLEX_F32_BYTES)
    }

    #[allow(dead_code)]
    pub(crate) fn encode(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        point_count: usize,
        points: &wgpu::Buffer,
        coefficients: &wgpu::Buffer,
        output: &wgpu::Buffer,
    ) -> Result<()> {
        self.encode_batch(
            device,
            encoder,
            self.batch_capacity,
            point_count,
            points,
            coefficients,
            output,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_batch(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        active_batch: usize,
        point_count: usize,
        points: &wgpu::Buffer,
        coefficients: &wgpu::Buffer,
        output: &wgpu::Buffer,
    ) -> Result<()> {
        self.encode_impl(
            device,
            encoder,
            active_batch,
            point_count,
            points,
            coefficients,
            output,
            #[cfg(feature = "gpu-profiling")]
            GpuProfileQueryWriter::disabled(),
        )
    }

    #[cfg(feature = "gpu-profiling")]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_profiled(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        point_count: usize,
        points: &wgpu::Buffer,
        coefficients: &wgpu::Buffer,
        output: &wgpu::Buffer,
        query_set: &wgpu::QuerySet,
        first_query: u32,
    ) -> Result<NufftGpuProfileLayout> {
        if point_count == 0 {
            return Err(NufftError::GpuExecutionUnavailable {
                kind: "type-2 stage profiling",
                reason: "at least one point is required",
            });
        }
        let layout =
            NufftGpuProfileLayout::type2(first_query).map_err(|_| NufftError::LengthOverflow {
                context: "type-2 stage-profile query range",
            })?;
        self.encode_impl(
            device,
            encoder,
            self.batch_capacity,
            point_count,
            points,
            coefficients,
            output,
            GpuProfileQueryWriter::enabled(query_set, &layout),
        )?;
        Ok(layout)
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_impl(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        active_batch: usize,
        point_count: usize,
        points: &wgpu::Buffer,
        coefficients: &wgpu::Buffer,
        output: &wgpu::Buffer,
        #[cfg(feature = "gpu-profiling")] profile: GpuProfileQueryWriter<'_>,
    ) -> Result<()> {
        if active_batch == 0 || active_batch > self.batch_capacity {
            return Err(NufftError::InvalidBatch {
                actual: active_batch,
                maximum: self.batch_capacity,
            });
        }
        if point_count == 0 {
            return Ok(());
        }
        u32::try_from(point_count).map_err(|_| NufftError::LengthOverflow {
            context: "type-2 GPU point count",
        })?;
        let point_bytes = Self::point_buffer_size_bytes(point_count)?;
        let coefficient_elements = checked_product(
            "type-2 batched Fourier coefficient count",
            active_batch,
            self.mode_count,
        )?;
        let output_elements = checked_product(
            "type-2 batched output element count",
            active_batch,
            point_count,
        )?;
        let coefficient_bytes = Self::complex_buffer_size_bytes(
            "type-2 Fourier coefficient buffer",
            coefficient_elements,
        )?;
        let output_bytes =
            Self::complex_buffer_size_bytes("type-2 output buffer", output_elements)?;
        validate_external_storage_buffer(
            "type-2 point",
            points,
            point_bytes,
            self.max_storage_binding_bytes,
        )?;
        validate_external_storage_buffer(
            "type-2 Fourier coefficient",
            coefficients,
            coefficient_bytes,
            self.max_storage_binding_bytes,
        )?;
        validate_external_storage_buffer(
            "type-2 output",
            output,
            output_bytes,
            self.max_storage_binding_bytes,
        )?;
        if points == output {
            return Err(NufftError::GpuBufferAliasUnsupported {
                first: "type-2 point",
                second: "type-2 output",
            });
        }
        let vector_tiles = active_batch.div_ceil(VECTOR_TILE);
        let interpolation_elements = checked_product(
            "type-2 batched interpolation work item count",
            point_count,
            vector_tiles,
        )?;
        let interpolation_elements_u32 =
            u32::try_from(interpolation_elements).map_err(|_| NufftError::LengthOverflow {
                context: "type-2 batched interpolation shader index space",
            })?;
        let interpolation_dispatch = split_workgroups(
            interpolation_elements_u32.div_ceil(WORKGROUP_SIZE),
            self.max_workgroups_per_dimension,
        )?;

        let predeconvolution_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("wgpu_nufft.type2.predeconvolution.bind_group"),
            layout: &self.predeconvolution_layout,
            entries: &[
                binding_entry(0, coefficients, coefficient_bytes),
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.amplitudes.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.fine_input.as_entire_binding(),
                },
            ],
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("wgpu_nufft.type2.predeconvolution.pass"),
                timestamp_writes: {
                    #[cfg(feature = "gpu-profiling")]
                    {
                        profile.timestamp_writes(Some(0), Some(1))
                    }
                    #[cfg(not(feature = "gpu-profiling"))]
                    {
                        None
                    }
                },
            });
            pass.set_pipeline(&self.predeconvolution_pipeline);
            pass.set_bind_group(0, &predeconvolution_bind_group, &[]);
            pass.dispatch_workgroups(
                self.predeconvolution_dispatch.0,
                self.predeconvolution_dispatch.1,
                self.predeconvolution_dispatch.2,
            );
        }

        self.fft
            .execute_views(
                device,
                encoder,
                BufferView::whole(&self.fine_input),
                BufferView::whole(&self.fine_output),
            )
            .map_err(|source| NufftError::FftExecutionFailed {
                stage: "type-2 oversampled-grid C2C transform",
                source,
            })?;

        let interpolation_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("wgpu_nufft.type2.interpolation.bind_group"),
            layout: &self.interpolation_layout,
            entries: &[
                binding_entry(0, points, point_bytes),
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.fine_output.as_entire_binding(),
                },
                binding_entry(2, output, output_bytes),
            ],
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("wgpu_nufft.type2.interpolation.pass"),
                timestamp_writes: {
                    #[cfg(feature = "gpu-profiling")]
                    {
                        profile.timestamp_writes(Some(2), Some(3))
                    }
                    #[cfg(not(feature = "gpu-profiling"))]
                    {
                        None
                    }
                },
            });
            pass.set_pipeline(&self.interpolation_pipeline);
            pass.set_bind_group(0, &interpolation_bind_group, &[]);
            pass.dispatch_workgroups(
                interpolation_dispatch.0,
                interpolation_dispatch.1,
                interpolation_dispatch.2,
            );
        }
        Ok(())
    }

    #[cfg(feature = "gpu-profiling")]
    pub(crate) fn fft_diagnostics(&self) -> wgpu_fft::FftDiagnostics {
        self.fft.diagnostics()
    }
}

fn mode_amplitudes(config: &NufftConfig, coefficients: &[f64]) -> Result<Vec<f32>> {
    let mode_count = config.n_modes()[0];
    let mut amplitudes = Vec::new();
    amplitudes
        .try_reserve_exact(mode_count)
        .map_err(|_| NufftError::HostAllocationFailed {
            buffer: "type-2 deconvolution amplitudes",
            elements: mode_count,
        })?;
    for index in 0..mode_count {
        let mode = mode_for_storage_index(config.mode_order(), index, mode_count);
        let coefficient = coefficients[mode.unsigned_abs() as usize];
        let amplitude = (1.0 / coefficient) as f32;
        if !coefficient.is_finite() || coefficient == 0.0 || !amplitude.is_finite() {
            return Err(NufftError::InvalidKernelFourierCoefficient {
                mode,
                value: coefficient,
            });
        }
        amplitudes.push(amplitude);
    }
    Ok(amplitudes)
}

fn create_compute_pipeline(
    device: &wgpu::Device,
    label: &str,
    source: &str,
) -> wgpu::ComputePipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: None,
        module: &shader,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    })
}

fn generate_predeconvolution_wgsl(config: &NufftConfig, fine_length: usize) -> String {
    let batch_capacity = config.batch();
    let total_fine_count = fine_length * batch_capacity;
    let mode_count = config.n_modes()[0];
    let half = mode_count / 2;
    let nonnegative_count = mode_count.div_ceil(2);
    let centered = matches!(config.mode_order(), ModeOrder::Centered);
    format!(
        r#"const WORKGROUP_SIZE: u32 = {WORKGROUP_SIZE}u;
const MODE_COUNT: u32 = {mode_count}u;
const FINE_LENGTH: u32 = {fine_length}u;
const TOTAL_FINE_COUNT: u32 = {total_fine_count}u;
const HALF_MODE_COUNT: u32 = {half}u;
const NONNEGATIVE_COUNT: u32 = {nonnegative_count}u;
const CENTERED_ORDER: bool = {centered};

@group(0) @binding(0) var<storage, read> coefficients: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read> amplitudes: array<f32>;
@group(0) @binding(2) var<storage, read_write> fine_grid: array<vec2<f32>>;

@compute @workgroup_size({WORKGROUP_SIZE})
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {{
    let wg_flat = (wid.z * nwg.y + wid.y) * nwg.x + wid.x;
    if (wg_flat > (TOTAL_FINE_COUNT - 1u) / WORKGROUP_SIZE) {{ return; }}
    let index = wg_flat * WORKGROUP_SIZE + lid.x;
    if (index >= TOTAL_FINE_COUNT) {{ return; }}

    let batch_index = index / FINE_LENGTH;
    let fine_index = index - batch_index * FINE_LENGTH;
    let active_batch = arrayLength(&coefficients) / MODE_COUNT;
    if (batch_index >= active_batch) {{
        fine_grid[index] = vec2<f32>(0.0, 0.0);
        return;
    }}

    var source_index = 0u;
    var is_mapped = false;
    if (fine_index < NONNEGATIVE_COUNT) {{
        source_index = select(fine_index, HALF_MODE_COUNT + fine_index, CENTERED_ORDER);
        is_mapped = true;
    }} else if (fine_index >= FINE_LENGTH - HALF_MODE_COUNT) {{
        let magnitude = FINE_LENGTH - fine_index;
        source_index = select(MODE_COUNT - magnitude, HALF_MODE_COUNT - magnitude, CENTERED_ORDER);
        is_mapped = true;
    }}
    if (is_mapped) {{
        let coefficient_index = batch_index * MODE_COUNT + source_index;
        fine_grid[index] = coefficients[coefficient_index] * amplitudes[source_index];
    }} else {{
        fine_grid[index] = vec2<f32>(0.0, 0.0);
    }}
}}
"#,
    )
}

fn generate_interpolation_wgsl(kernel: EsKernel, fine_length: usize) -> String {
    let width = kernel.width();
    let beta = format_wgsl_f32(kernel.beta() as f32);
    let fine_length_f32 = format_wgsl_f32(fine_length as f32);
    let half_width = format_wgsl_f32(kernel.half_width() as f32);
    let position_scale = fine_length as f64 / std::f64::consts::TAU;
    let position_scale_hi = position_scale as f32;
    let position_scale_lo = (position_scale - f64::from(position_scale_hi)) as f32;
    let position_scale_hi = format_wgsl_f32(position_scale_hi);
    let position_scale_lo = format_wgsl_f32(position_scale_lo);
    let grid_origin = format_wgsl_f32((fine_length / 2) as f32);
    let entry = format!(
        r#"const WORKGROUP_SIZE: u32 = {WORKGROUP_SIZE}u;
const FINE_LENGTH: u32 = {fine_length}u;
const FINE_LENGTH_I32: i32 = {fine_length}i;
const FINE_LENGTH_F32: f32 = {fine_length_f32};
const VECTOR_TILE: u32 = {VECTOR_TILE}u;
const WIDTH: u32 = {width}u;
const WIDTH_F32: f32 = {width}.0;
const HALF_WIDTH: f32 = {half_width};
const BETA: f32 = {beta};
const POSITION_SCALE_HI: f32 = {position_scale_hi};
const POSITION_SCALE_LO: f32 = {position_scale_lo};
const GRID_ORIGIN: f32 = {grid_origin};

@group(0) @binding(0) var<storage, read> points: array<f32>;
@group(0) @binding(1) var<storage, read> fine_grid: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> output_values: array<vec2<f32>>;

fn es_weight(distance: f32) -> f32 {{
    let scaled = 2.0 * abs(distance) / WIDTH_F32;
    let squared = scaled * scaled;
    if (squared >= 1.0) {{ return 0.0; }}
    return exp(BETA * (sqrt(max(0.0, 1.0 - squared)) - 1.0));
}}

fn wrap_index(index: i32) -> u32 {{
    var wrapped = index;
    if (wrapped < 0) {{ wrapped = wrapped + FINE_LENGTH_I32; }}
    if (wrapped >= FINE_LENGTH_I32) {{ wrapped = wrapped - FINE_LENGTH_I32; }}
    return u32(wrapped);
}}

fn ceil_df64_to_i32(value: Df64) -> i32 {{
    let base = floor(value.hi);
    let remainder = df64_sub(value, Df64(base, 0.0));
    let has_positive_remainder = remainder.hi > 0.0 ||
        (remainder.hi == 0.0 && remainder.lo > 0.0);
    return i32(base) + select(0, 1, has_positive_remainder);
}}

fn position_is_negative(value: Df64) -> bool {{
    return value.hi < 0.0 || (value.hi == 0.0 && value.lo < 0.0);
}}

fn position_at_least_grid(value: Df64) -> bool {{
    return value.hi > FINE_LENGTH_F32 ||
        (value.hi == FINE_LENGTH_F32 && value.lo >= 0.0);
}}

fn fold_position(point: f32) -> Df64 {{
    let scaled = df64_mul(
        Df64(point, 0.0),
        Df64(POSITION_SCALE_HI, POSITION_SCALE_LO),
    );
    var position = df64_add(scaled, Df64(GRID_ORIGIN, 0.0));
    if (position_is_negative(position)) {{
        position = df64_add(position, Df64(FINE_LENGTH_F32, 0.0));
    }}
    if (position_is_negative(position)) {{
        position = df64_add(position, Df64(FINE_LENGTH_F32, 0.0));
    }}
    if (position_at_least_grid(position)) {{
        position = df64_sub(position, Df64(FINE_LENGTH_F32, 0.0));
    }}
    if (position_at_least_grid(position)) {{
        position = df64_sub(position, Df64(FINE_LENGTH_F32, 0.0));
    }}
    return position;
}}

@compute @workgroup_size({WORKGROUP_SIZE})
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {{
    let point_count = arrayLength(&points);
    let active_batch = arrayLength(&output_values) / point_count;
    let vector_tile_count = (active_batch + VECTOR_TILE - 1u) / VECTOR_TILE;
    let total = point_count * vector_tile_count;
    let wg_flat = (wid.z * nwg.y + wid.y) * nwg.x + wid.x;
    if (wg_flat > (total - 1u) / WORKGROUP_SIZE) {{ return; }}
    let work_index = wg_flat * WORKGROUP_SIZE + lid.x;
    if (work_index >= total) {{ return; }}

    let vector_tile_index = work_index / point_count;
    let point_index = work_index - vector_tile_index * point_count;
    let batch_start = vector_tile_index * VECTOR_TILE;

    let position = fold_position(points[point_index]);
    let shifted = df64_sub(position, Df64(HALF_WIDTH, 0.0));
    let start = ceil_df64_to_i32(shifted);
    var fine_indices: array<u32, WIDTH>;
    var weights: array<f32, WIDTH>;
    for (var offset = 0u; offset < WIDTH; offset = offset + 1u) {{
        let unwrapped = start + i32(offset);
        let distance = df64_sub(Df64(f32(unwrapped), 0.0), position);
        fine_indices[offset] = wrap_index(unwrapped);
        weights[offset] = es_weight(distance.hi + distance.lo);
    }}
    var sums: array<vec2<f32>, VECTOR_TILE>;
    for (var vector = 0u; vector < VECTOR_TILE; vector = vector + 1u) {{
        sums[vector] = vec2<f32>(0.0, 0.0);
    }}
    for (var offset = 0u; offset < WIDTH; offset = offset + 1u) {{
        for (var vector = 0u; vector < VECTOR_TILE; vector = vector + 1u) {{
            let batch_index = batch_start + vector;
            if (batch_index < active_batch) {{
                let fine_index = batch_index * FINE_LENGTH + fine_indices[offset];
                sums[vector] = sums[vector] + fine_grid[fine_index] * weights[offset];
            }}
        }}
    }}
    for (var vector = 0u; vector < VECTOR_TILE; vector = vector + 1u) {{
        let batch_index = batch_start + vector;
        if (batch_index < active_batch) {{
            output_values[batch_index * point_count + point_index] = sums[vector];
        }}
    }}
}}
"#,
    );
    format!("{}\n{entry}", wgpu_fft::kernels::DF64_WGSL)
}

fn format_wgsl_f32(value: f32) -> String {
    debug_assert!(value.is_finite());
    let mut formatted = value.to_string();
    if !formatted.contains('.') && !formatted.contains('e') && !formatted.contains('E') {
        formatted.push_str(".0");
    }
    formatted
}

fn checked_buffer_size(
    context: &'static str,
    elements: usize,
    bytes_per_element: u64,
) -> Result<u64> {
    u64::try_from(elements)
        .ok()
        .and_then(|count| count.checked_mul(bytes_per_element))
        .ok_or(NufftError::LengthOverflow { context })
}

fn checked_product(context: &'static str, left: usize, right: usize) -> Result<usize> {
    left.checked_mul(right)
        .ok_or(NufftError::LengthOverflow { context })
}

fn validate_binding_limit(buffer: &'static str, bytes: u64, limit: u64) -> Result<()> {
    if bytes > limit {
        Err(NufftError::GpuBufferBindingTooLarge {
            buffer,
            required_bytes: bytes,
            limit_bytes: limit,
        })
    } else {
        Ok(())
    }
}

fn validate_external_storage_buffer(
    label: &'static str,
    buffer: &wgpu::Buffer,
    required_bytes: u64,
    max_storage_binding_bytes: u64,
) -> Result<()> {
    if !buffer.usage().contains(wgpu::BufferUsages::STORAGE) {
        return Err(NufftError::GpuBufferMissingUsage {
            buffer: label,
            required_usage: "STORAGE",
        });
    }
    if buffer.size() < required_bytes {
        return Err(NufftError::GpuBufferTooSmall {
            buffer: label,
            required_bytes,
            actual_bytes: buffer.size(),
        });
    }
    validate_binding_limit(label, required_bytes, max_storage_binding_bytes)
}

fn binding_entry<'a>(
    binding: u32,
    buffer: &'a wgpu::Buffer,
    size: u64,
) -> wgpu::BindGroupEntry<'a> {
    wgpu::BindGroupEntry {
        binding,
        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer,
            offset: 0,
            size: NonZeroU64::new(size),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn predeconvolution_shader_maps_every_mode_order_and_zeros_holes() {
        for (modes, order) in [
            (1, ModeOrder::Centered),
            (4, ModeOrder::Centered),
            (5, ModeOrder::Centered),
            (4, ModeOrder::Fft),
            (5, ModeOrder::Fft),
        ] {
            let config = NufftConfig::new([modes], 1.0e-6).with_mode_order(order);
            let source = generate_predeconvolution_wgsl(&config, 16);
            assert!(source.contains("fine_grid[index] = vec2<f32>(0.0, 0.0);"));
            assert!(source.contains(&format!("const MODE_COUNT: u32 = {modes}u;")));
            assert!(source.contains(&format!(
                "const CENTERED_ORDER: bool = {};",
                matches!(order, ModeOrder::Centered)
            )));
        }
    }

    #[test]
    fn interpolation_shader_preserves_low_word_when_selecting_support_start() {
        let kernel = EsKernel::for_tolerance(1.0e-6, 2.0).unwrap();
        let source = generate_interpolation_wgsl(kernel, 65_536);
        assert!(source.contains("fn ceil_df64_to_i32(value: Df64) -> i32"));
        assert!(source.contains("let remainder = df64_sub(value, Df64(base, 0.0));"));
        assert!(source.contains("let start = ceil_df64_to_i32(shifted);"));
        assert!(!source.contains("ceil(shifted.hi + shifted.lo)"));
    }

    #[test]
    fn batched_shaders_use_transform_major_storage_and_vector_tiles() {
        let config = NufftConfig::new([8], 1.0e-6).with_batch(5);
        let predeconvolution = generate_predeconvolution_wgsl(&config, 16);
        assert!(predeconvolution.contains("const TOTAL_FINE_COUNT: u32 = 80u;"));
        assert!(predeconvolution.contains("let batch_index = index / FINE_LENGTH;"));
        assert!(predeconvolution
            .contains("let coefficient_index = batch_index * MODE_COUNT + source_index;"));
        assert!(predeconvolution.contains("if (batch_index >= active_batch)"));

        let kernel = EsKernel::for_tolerance(1.0e-6, 2.0).unwrap();
        let interpolation = generate_interpolation_wgsl(kernel, 16);
        assert!(interpolation.contains("const VECTOR_TILE: u32 = 4u;"));
        assert!(interpolation.contains("let batch_start = vector_tile_index * VECTOR_TILE;"));
        assert!(interpolation
            .contains("let fine_index = batch_index * FINE_LENGTH + fine_indices[offset];"));
        assert!(interpolation
            .contains("output_values[batch_index * point_count + point_index] = sums[vector];"));
    }

    #[test]
    fn workgroup_limit_accounts_for_the_x_dimension() {
        let limits = wgpu::Limits {
            max_compute_invocations_per_workgroup: 128,
            max_compute_workgroup_size_x: 32,
            ..wgpu::Limits::default()
        };
        assert_eq!(max_supported_workgroup_size(&limits), 32);

        let limits = wgpu::Limits {
            max_compute_invocations_per_workgroup: 16,
            max_compute_workgroup_size_x: 256,
            ..wgpu::Limits::default()
        };
        assert_eq!(max_supported_workgroup_size(&limits), 16);
    }
}
