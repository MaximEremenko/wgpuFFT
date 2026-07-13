use std::num::NonZeroU64;

use wgpu::util::DeviceExt;
use wgpu_fft::{BufferView, FftConfig, FftDirection, FftPlan, FftPrecision, Normalization};

use crate::config::{NufftConfig, NufftSign};
use crate::direct::mode_for_storage_index;
use crate::error::{NufftError, Result};
use crate::gpu::max_supported_workgroup_size;
use crate::kernel::EsKernel;

const WORKGROUP_SIZE: u32 = 64;
const COMPLEX_F32_BYTES: u64 = 8;
const F32_BYTES: u64 = 4;
const U32_BYTES: u64 = 4;

/// Device-specific resources for deterministic, atomics-free complex spreading.
///
/// Integer atomics are used only to construct bins. Each bin is then sorted by
/// original point index, and one invocation owns each fine-grid output cell.
pub(crate) struct Type1GpuPlan {
    fft: FftPlan,
    amplitudes: wgpu::Buffer,
    fine_input: wgpu::Buffer,
    fine_output: wgpu::Buffer,
    count_pipeline: wgpu::ComputePipeline,
    count_layout: wgpu::BindGroupLayout,
    prefix_pipeline: wgpu::ComputePipeline,
    prefix_layout: wgpu::BindGroupLayout,
    scatter_pipeline: wgpu::ComputePipeline,
    scatter_layout: wgpu::BindGroupLayout,
    sort_pipeline: wgpu::ComputePipeline,
    sort_layout: wgpu::BindGroupLayout,
    gather_pipeline: wgpu::ComputePipeline,
    gather_layout: wgpu::BindGroupLayout,
    deconvolution_pipeline: wgpu::ComputePipeline,
    deconvolution_layout: wgpu::BindGroupLayout,
    sort_dispatch: (u32, u32, u32),
    gather_dispatch: (u32, u32, u32),
    deconvolution_dispatch: (u32, u32, u32),
    max_workgroups_per_dimension: u32,
    mode_count: usize,
    fine_length: usize,
    max_storage_binding_bytes: u64,
    max_buffer_bytes: u64,
}

impl Type1GpuPlan {
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
                kind: "type-1",
                actual: config.dimensions(),
                supported: 1,
            });
        }
        // Both support indices and bin keys are represented by exact f32
        // integers in WGSL. Reserve the support halo beyond the last bin.
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

        let mode_count = config.mode_count()?;
        let fine_bytes = Self::complex_buffer_size_bytes("type-1 fine grid", fine_length)?;
        let amplitude_bytes =
            checked_buffer_size("type-1 deconvolution amplitudes", mode_count, F32_BYTES)?;
        let count_bytes = checked_buffer_size("type-1 bin counts", fine_length, U32_BYTES)?;
        let offset_count = fine_length
            .checked_add(1)
            .ok_or(NufftError::LengthOverflow {
                context: "type-1 bin offset count",
            })?;
        let offset_bytes = checked_buffer_size("type-1 bin offsets", offset_count, U32_BYTES)?;
        for (label, bytes) in [
            ("type-1 fine grid", fine_bytes),
            ("type-1 deconvolution amplitudes", amplitude_bytes),
            ("type-1 bin counts", count_bytes),
            ("type-1 bin offsets", offset_bytes),
        ] {
            validate_binding_limit(label, bytes, limits.max_storage_buffer_binding_size)?;
            validate_buffer_limit(label, bytes, limits.max_buffer_size)?;
        }

        let amplitudes = mode_amplitudes(config, kernel_fourier_coefficients)?;
        let amplitudes = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("wgpu_nufft.type1.deconvolution_amplitudes"),
            contents: bytemuck::cast_slice(&amplitudes),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let fine_usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST;
        let fine_input = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("wgpu_nufft.type1.fine_input"),
            size: fine_bytes,
            usage: fine_usage,
            mapped_at_creation: false,
        });
        let fine_output = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("wgpu_nufft.type1.fine_output"),
            size: fine_bytes,
            usage: fine_usage,
            mapped_at_creation: false,
        });

        let fft_direction = match config.sign() {
            NufftSign::Positive => FftDirection::Inverse,
            NufftSign::Negative => FftDirection::Forward,
        };
        let fft = FftPlan::c2c(
            device,
            queue,
            FftConfig::new(fine_length)
                .with_direction(fft_direction)
                .with_normalization(Normalization::None)
                .with_precision(FftPrecision::F32),
        )
        .map_err(|source| NufftError::FftShapeUnsupported {
            stage: "type-1 oversampled-grid C2C plan",
            source,
        })?;

        let count_pipeline = create_compute_pipeline(
            device,
            "wgpu_nufft.type1.bin_count",
            &generate_count_wgsl(fine_length),
        );
        let count_layout = count_pipeline.get_bind_group_layout(0);
        let prefix_pipeline = create_compute_pipeline(
            device,
            "wgpu_nufft.type1.bin_prefix",
            &generate_prefix_wgsl(),
        );
        let prefix_layout = prefix_pipeline.get_bind_group_layout(0);
        let scatter_pipeline = create_compute_pipeline(
            device,
            "wgpu_nufft.type1.bin_scatter",
            &generate_scatter_wgsl(fine_length),
        );
        let scatter_layout = scatter_pipeline.get_bind_group_layout(0);
        let sort_pipeline =
            create_compute_pipeline(device, "wgpu_nufft.type1.bin_sort", &generate_sort_wgsl());
        let sort_layout = sort_pipeline.get_bind_group_layout(0);
        let gather_pipeline = create_compute_pipeline(
            device,
            "wgpu_nufft.type1.spread_gather",
            &generate_gather_wgsl(kernel, fine_length),
        );
        let gather_layout = gather_pipeline.get_bind_group_layout(0);
        let deconvolution_pipeline = create_compute_pipeline(
            device,
            "wgpu_nufft.type1.deconvolution",
            &generate_deconvolution_wgsl(config, fine_length),
        );
        let deconvolution_layout = deconvolution_pipeline.get_bind_group_layout(0);

        let max_workgroups_per_dimension = limits.max_compute_workgroups_per_dimension;
        let sort_dispatch = dispatch_for_elements(fine_length, max_workgroups_per_dimension)?;
        let gather_dispatch = dispatch_for_elements(fine_length, max_workgroups_per_dimension)?;
        let deconvolution_dispatch =
            dispatch_for_elements(mode_count, max_workgroups_per_dimension)?;

        Ok(Self {
            fft,
            amplitudes,
            fine_input,
            fine_output,
            count_pipeline,
            count_layout,
            prefix_pipeline,
            prefix_layout,
            scatter_pipeline,
            scatter_layout,
            sort_pipeline,
            sort_layout,
            gather_pipeline,
            gather_layout,
            deconvolution_pipeline,
            deconvolution_layout,
            sort_dispatch,
            gather_dispatch,
            deconvolution_dispatch,
            max_workgroups_per_dimension,
            mode_count,
            fine_length,
            max_storage_binding_bytes: limits.max_storage_buffer_binding_size,
            max_buffer_bytes: limits.max_buffer_size,
        })
    }

    pub(crate) fn point_buffer_size_bytes(point_count: usize) -> Result<u64> {
        checked_buffer_size("type-1 point buffer", point_count, F32_BYTES)
    }

    pub(crate) fn strength_buffer_size_bytes(point_count: usize) -> Result<u64> {
        Self::complex_buffer_size_bytes("type-1 strength buffer", point_count)
    }

    pub(crate) fn complex_buffer_size_bytes(
        buffer: &'static str,
        element_count: usize,
    ) -> Result<u64> {
        checked_buffer_size(buffer, element_count, COMPLEX_F32_BYTES)
    }

    pub(crate) fn encode(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        point_count: usize,
        points: &wgpu::Buffer,
        strengths: &wgpu::Buffer,
        output: &wgpu::Buffer,
    ) -> Result<()> {
        let output_bytes =
            Self::complex_buffer_size_bytes("type-1 Fourier output buffer", self.mode_count)?;
        validate_external_storage_buffer(
            "type-1 Fourier output",
            output,
            output_bytes,
            self.max_storage_binding_bytes,
        )?;

        let output_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("wgpu_nufft.type1.deconvolution.bind_group"),
            layout: &self.deconvolution_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.fine_output.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.amplitudes.as_entire_binding(),
                },
                binding_entry(2, output, output_bytes),
            ],
        });

        if point_count == 0 {
            // Type-1 with no sources has a nonempty, all-zero mode output. Do
            // not leave stale plan scratch or return early as type-2 may.
            encoder.clear_buffer(&self.fine_input, 0, None);
            self.encode_fft(device, encoder)?;
            self.encode_deconvolution(encoder, &output_bind_group);
            return Ok(());
        }

        let point_count_u32 =
            u32::try_from(point_count).map_err(|_| NufftError::LengthOverflow {
                context: "type-1 GPU point count",
            })?;
        let point_bytes = Self::point_buffer_size_bytes(point_count)?;
        let strength_bytes = Self::strength_buffer_size_bytes(point_count)?;
        validate_external_storage_buffer(
            "type-1 point",
            points,
            point_bytes,
            self.max_storage_binding_bytes,
        )?;
        validate_external_storage_buffer(
            "type-1 strength",
            strengths,
            strength_bytes,
            self.max_storage_binding_bytes,
        )?;

        let count_bytes = checked_buffer_size("type-1 bin counts", self.fine_length, U32_BYTES)?;
        let offset_count = self
            .fine_length
            .checked_add(1)
            .ok_or(NufftError::LengthOverflow {
                context: "type-1 bin offset count",
            })?;
        let offset_bytes = checked_buffer_size("type-1 bin offsets", offset_count, U32_BYTES)?;
        let index_bytes =
            checked_buffer_size("type-1 sorted point indices", point_count, U32_BYTES)?;
        for (label, bytes) in [
            ("type-1 bin counts", count_bytes),
            ("type-1 bin cursors", count_bytes),
            ("type-1 bin offsets", offset_bytes),
            ("type-1 sorted point indices", index_bytes),
        ] {
            validate_binding_limit(label, bytes, self.max_storage_binding_bytes)?;
            validate_buffer_limit(label, bytes, self.max_buffer_bytes)?;
        }
        let point_dispatch = split_workgroups(
            point_count_u32.div_ceil(WORKGROUP_SIZE),
            self.max_workgroups_per_dimension,
        )?;

        let atomic_usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let bin_counts = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("wgpu_nufft.type1.bin_counts"),
            size: count_bytes,
            usage: atomic_usage,
            mapped_at_creation: false,
        });
        let bin_cursors = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("wgpu_nufft.type1.bin_cursors"),
            size: count_bytes,
            usage: atomic_usage,
            mapped_at_creation: false,
        });
        let bin_offsets = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("wgpu_nufft.type1.bin_offsets"),
            size: offset_bytes,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let sorted_indices = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("wgpu_nufft.type1.sorted_point_indices"),
            size: index_bytes,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let count_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("wgpu_nufft.type1.bin_count.bind_group"),
            layout: &self.count_layout,
            entries: &[
                binding_entry(0, points, point_bytes),
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: bin_counts.as_entire_binding(),
                },
            ],
        });
        let prefix_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("wgpu_nufft.type1.bin_prefix.bind_group"),
            layout: &self.prefix_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: bin_counts.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: bin_offsets.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: bin_cursors.as_entire_binding(),
                },
            ],
        });
        let scatter_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("wgpu_nufft.type1.bin_scatter.bind_group"),
            layout: &self.scatter_layout,
            entries: &[
                binding_entry(0, points, point_bytes),
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: bin_offsets.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: bin_cursors.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: sorted_indices.as_entire_binding(),
                },
            ],
        });
        let sort_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("wgpu_nufft.type1.bin_sort.bind_group"),
            layout: &self.sort_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: bin_offsets.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: sorted_indices.as_entire_binding(),
                },
            ],
        });
        let gather_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("wgpu_nufft.type1.spread_gather.bind_group"),
            layout: &self.gather_layout,
            entries: &[
                binding_entry(0, points, point_bytes),
                binding_entry(1, strengths, strength_bytes),
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: bin_offsets.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: sorted_indices.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.fine_input.as_entire_binding(),
                },
            ],
        });

        encoder.clear_buffer(&bin_counts, 0, None);
        encode_pass(
            encoder,
            "wgpu_nufft.type1.bin_count.pass",
            &self.count_pipeline,
            &count_bind_group,
            point_dispatch,
        );
        encode_pass(
            encoder,
            "wgpu_nufft.type1.bin_prefix.pass",
            &self.prefix_pipeline,
            &prefix_bind_group,
            (1, 1, 1),
        );
        encode_pass(
            encoder,
            "wgpu_nufft.type1.bin_scatter.pass",
            &self.scatter_pipeline,
            &scatter_bind_group,
            point_dispatch,
        );
        encode_pass(
            encoder,
            "wgpu_nufft.type1.bin_sort.pass",
            &self.sort_pipeline,
            &sort_bind_group,
            self.sort_dispatch,
        );
        encode_pass(
            encoder,
            "wgpu_nufft.type1.spread_gather.pass",
            &self.gather_pipeline,
            &gather_bind_group,
            self.gather_dispatch,
        );
        self.encode_fft(device, encoder)?;
        self.encode_deconvolution(encoder, &output_bind_group);
        Ok(())
    }

    fn encode_fft(&self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder) -> Result<()> {
        self.fft
            .execute_views(
                device,
                encoder,
                BufferView::whole(&self.fine_input),
                BufferView::whole(&self.fine_output),
            )
            .map_err(|source| NufftError::FftExecutionFailed {
                stage: "type-1 oversampled-grid C2C transform",
                source,
            })
    }

    fn encode_deconvolution(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        bind_group: &wgpu::BindGroup,
    ) {
        encode_pass(
            encoder,
            "wgpu_nufft.type1.deconvolution.pass",
            &self.deconvolution_pipeline,
            bind_group,
            self.deconvolution_dispatch,
        );
    }
}

fn mode_amplitudes(config: &NufftConfig, coefficients: &[f64]) -> Result<Vec<f32>> {
    let mode_count = config.n_modes()[0];
    let mut amplitudes = Vec::new();
    amplitudes
        .try_reserve_exact(mode_count)
        .map_err(|_| NufftError::HostAllocationFailed {
            buffer: "type-1 deconvolution amplitudes",
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

fn encode_pass(
    encoder: &mut wgpu::CommandEncoder,
    label: &str,
    pipeline: &wgpu::ComputePipeline,
    bind_group: &wgpu::BindGroup,
    dispatch: (u32, u32, u32),
) {
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some(label),
        timestamp_writes: None,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, bind_group, &[]);
    pass.dispatch_workgroups(dispatch.0, dispatch.1, dispatch.2);
}

fn generate_position_wgsl(fine_length: usize) -> String {
    let fine_length_f32 = format_wgsl_f32(fine_length as f32);
    let position_scale = fine_length as f64 / std::f64::consts::TAU;
    let position_scale_hi = position_scale as f32;
    let position_scale_lo = (position_scale - f64::from(position_scale_hi)) as f32;
    format!(
        r#"const FINE_LENGTH: u32 = {fine_length}u;
const FINE_LENGTH_I32: i32 = {fine_length}i;
const FINE_LENGTH_F32: f32 = {fine_length_f32};
const POSITION_SCALE_HI: f32 = {position_scale_hi};
const POSITION_SCALE_LO: f32 = {position_scale_lo};
const GRID_ORIGIN: f32 = {grid_origin};

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

fn floor_df64_to_i32(value: Df64) -> i32 {{
    let base = floor(value.hi);
    let remainder = df64_sub(value, Df64(base, 0.0));
    let has_negative_remainder = remainder.hi < 0.0 ||
        (remainder.hi == 0.0 && remainder.lo < 0.0);
    return i32(base) - select(0, 1, has_negative_remainder);
}}
"#,
        position_scale_hi = format_wgsl_f32(position_scale_hi),
        position_scale_lo = format_wgsl_f32(position_scale_lo),
        grid_origin = format_wgsl_f32((fine_length / 2) as f32),
    )
}

fn generate_count_wgsl(fine_length: usize) -> String {
    let position = generate_position_wgsl(fine_length);
    let entry = format!(
        r#"const WORKGROUP_SIZE: u32 = {WORKGROUP_SIZE}u;

@group(0) @binding(0) var<storage, read> points: array<f32>;
@group(0) @binding(1) var<storage, read_write> bin_counts: array<atomic<u32>>;

@compute @workgroup_size({WORKGROUP_SIZE})
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {{
    let total = arrayLength(&points);
    let wg_flat = (wid.z * nwg.y + wid.y) * nwg.x + wid.x;
    if (wg_flat > (total - 1u) / WORKGROUP_SIZE) {{ return; }}
    let point_index = wg_flat * WORKGROUP_SIZE + lid.x;
    if (point_index >= total) {{ return; }}

    let position = fold_position(points[point_index]);
    let bin = u32(floor_df64_to_i32(position));
    atomicAdd(&bin_counts[bin], 1u);
}}
"#,
    );
    format!("{}\n{position}\n{entry}", wgpu_fft::kernels::DF64_WGSL)
}

fn generate_prefix_wgsl() -> String {
    r#"@group(0) @binding(0) var<storage, read_write> bin_counts: array<atomic<u32>>;
@group(0) @binding(1) var<storage, read_write> bin_offsets: array<u32>;
@group(0) @binding(2) var<storage, read_write> bin_cursors: array<atomic<u32>>;

@compute @workgroup_size(1)
fn main() {
    let bin_count = arrayLength(&bin_counts);
    var running = 0u;
    for (var bin = 0u; bin < bin_count; bin = bin + 1u) {
        bin_offsets[bin] = running;
        running = running + atomicLoad(&bin_counts[bin]);
        atomicStore(&bin_cursors[bin], 0u);
    }
    bin_offsets[bin_count] = running;
}
"#
    .to_owned()
}

fn generate_scatter_wgsl(fine_length: usize) -> String {
    let position = generate_position_wgsl(fine_length);
    let entry = format!(
        r#"const WORKGROUP_SIZE: u32 = {WORKGROUP_SIZE}u;

@group(0) @binding(0) var<storage, read> points: array<f32>;
@group(0) @binding(1) var<storage, read> bin_offsets: array<u32>;
@group(0) @binding(2) var<storage, read_write> bin_cursors: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> sorted_indices: array<u32>;

@compute @workgroup_size({WORKGROUP_SIZE})
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {{
    let total = arrayLength(&points);
    let wg_flat = (wid.z * nwg.y + wid.y) * nwg.x + wid.x;
    if (wg_flat > (total - 1u) / WORKGROUP_SIZE) {{ return; }}
    let point_index = wg_flat * WORKGROUP_SIZE + lid.x;
    if (point_index >= total) {{ return; }}

    let position = fold_position(points[point_index]);
    let bin = u32(floor_df64_to_i32(position));
    let slot = bin_offsets[bin] + atomicAdd(&bin_cursors[bin], 1u);
    sorted_indices[slot] = point_index;
}}
"#,
    );
    format!("{}\n{position}\n{entry}", wgpu_fft::kernels::DF64_WGSL)
}

fn generate_sort_wgsl() -> String {
    format!(
        r#"const WORKGROUP_SIZE: u32 = {WORKGROUP_SIZE}u;

@group(0) @binding(0) var<storage, read> bin_offsets: array<u32>;
@group(0) @binding(1) var<storage, read_write> sorted_indices: array<u32>;

fn swap_indices(left: u32, right: u32) {{
    let temporary = sorted_indices[left];
    sorted_indices[left] = sorted_indices[right];
    sorted_indices[right] = temporary;
}}

fn sift_down(start: u32, count: u32, initial_root: u32) {{
    var root = initial_root;
    loop {{
        let child = root * 2u + 1u;
        if (child >= count) {{ break; }}
        var greatest = root;
        if (sorted_indices[start + greatest] < sorted_indices[start + child]) {{
            greatest = child;
        }}
        if (child + 1u < count &&
            sorted_indices[start + greatest] < sorted_indices[start + child + 1u]) {{
            greatest = child + 1u;
        }}
        if (greatest == root) {{ break; }}
        swap_indices(start + root, start + greatest);
        root = greatest;
    }}
}}

fn heap_sort(start: u32, end: u32) {{
    let count = end - start;
    if (count < 2u) {{ return; }}
    var root = count / 2u;
    loop {{
        if (root == 0u) {{ break; }}
        root = root - 1u;
        sift_down(start, count, root);
    }}
    var remaining = count;
    loop {{
        if (remaining <= 1u) {{ break; }}
        remaining = remaining - 1u;
        swap_indices(start, start + remaining);
        sift_down(start, remaining, 0u);
    }}
}}

@compute @workgroup_size({WORKGROUP_SIZE})
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {{
    let total_bins = arrayLength(&bin_offsets) - 1u;
    let wg_flat = (wid.z * nwg.y + wid.y) * nwg.x + wid.x;
    if (wg_flat > (total_bins - 1u) / WORKGROUP_SIZE) {{ return; }}
    let bin = wg_flat * WORKGROUP_SIZE + lid.x;
    if (bin >= total_bins) {{ return; }}
    heap_sort(bin_offsets[bin], bin_offsets[bin + 1u]);
}}
"#,
    )
}

fn generate_gather_wgsl(kernel: EsKernel, fine_length: usize) -> String {
    let position = generate_position_wgsl(fine_length);
    let width = kernel.width();
    let bin_radius = width.div_ceil(2);
    let entry = format!(
        r#"const WORKGROUP_SIZE: u32 = {WORKGROUP_SIZE}u;
const WIDTH: u32 = {width}u;
const WIDTH_I32: i32 = {width}i;
const WIDTH_F32: f32 = {width}.0;
const HALF_WIDTH: f32 = {half_width};
const BETA: f32 = {beta};
const BIN_RADIUS: i32 = {bin_radius}i;

@group(0) @binding(0) var<storage, read> points: array<f32>;
@group(0) @binding(1) var<storage, read> strengths: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> bin_offsets: array<u32>;
@group(0) @binding(3) var<storage, read> sorted_indices: array<u32>;
@group(0) @binding(4) var<storage, read_write> fine_grid: array<vec2<f32>>;

fn ceil_df64_to_i32(value: Df64) -> i32 {{
    let base = floor(value.hi);
    let remainder = df64_sub(value, Df64(base, 0.0));
    let has_positive_remainder = remainder.hi > 0.0 ||
        (remainder.hi == 0.0 && remainder.lo > 0.0);
    return i32(base) + select(0, 1, has_positive_remainder);
}}

fn wrap_bin(index: i32) -> u32 {{
    var wrapped = index;
    if (wrapped < 0) {{ wrapped = wrapped + FINE_LENGTH_I32; }}
    if (wrapped >= FINE_LENGTH_I32) {{ wrapped = wrapped - FINE_LENGTH_I32; }}
    return u32(wrapped);
}}

fn es_weight(distance: f32) -> f32 {{
    let scaled = 2.0 * abs(distance) / WIDTH_F32;
    let squared = scaled * scaled;
    if (squared >= 1.0) {{ return 0.0; }}
    return exp(BETA * (sqrt(max(0.0, 1.0 - squared)) - 1.0));
}}

@compute @workgroup_size({WORKGROUP_SIZE})
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {{
    let wg_flat = (wid.z * nwg.y + wid.y) * nwg.x + wid.x;
    if (wg_flat > (FINE_LENGTH - 1u) / WORKGROUP_SIZE) {{ return; }}
    let cell = wg_flat * WORKGROUP_SIZE + lid.x;
    if (cell >= FINE_LENGTH) {{ return; }}

    var sum = vec2<f32>(0.0, 0.0);
    for (var bin_offset = -BIN_RADIUS; bin_offset < BIN_RADIUS;
         bin_offset = bin_offset + 1) {{
        let bin = wrap_bin(i32(cell) + bin_offset);
        let begin = bin_offsets[bin];
        let end = bin_offsets[bin + 1u];
        for (var slot = begin; slot < end; slot = slot + 1u) {{
            let point_index = sorted_indices[slot];
            let point_position = fold_position(points[point_index]);
            let shifted = df64_sub(point_position, Df64(HALF_WIDTH, 0.0));
            let start = ceil_df64_to_i32(shifted);
            var unwrapped = i32(cell);
            if (unwrapped < start) {{ unwrapped = unwrapped + FINE_LENGTH_I32; }}
            if (unwrapped >= start + WIDTH_I32) {{
                unwrapped = unwrapped - FINE_LENGTH_I32;
            }}
            if (unwrapped >= start && unwrapped < start + WIDTH_I32) {{
                let distance = df64_sub(Df64(f32(unwrapped), 0.0), point_position);
                let weight = es_weight(distance.hi + distance.lo);
                sum = sum + strengths[point_index] * weight;
            }}
        }}
    }}
    fine_grid[cell] = sum;
}}
"#,
        half_width = format_wgsl_f32(kernel.half_width() as f32),
        beta = format_wgsl_f32(kernel.beta() as f32),
    );
    format!("{}\n{position}\n{entry}", wgpu_fft::kernels::DF64_WGSL)
}

fn generate_deconvolution_wgsl(config: &NufftConfig, fine_length: usize) -> String {
    let mode_count = config.n_modes()[0];
    let half = mode_count / 2;
    let nonnegative_count = mode_count.div_ceil(2);
    let centered = matches!(config.mode_order(), crate::config::ModeOrder::Centered);
    format!(
        r#"const WORKGROUP_SIZE: u32 = {WORKGROUP_SIZE}u;
const MODE_COUNT: u32 = {mode_count}u;
const FINE_LENGTH: u32 = {fine_length}u;
const HALF_MODE_COUNT: u32 = {half}u;
const NONNEGATIVE_COUNT: u32 = {nonnegative_count}u;
const CENTERED_ORDER: bool = {centered};

@group(0) @binding(0) var<storage, read> fine_grid: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read> amplitudes: array<f32>;
@group(0) @binding(2) var<storage, read_write> output_values: array<vec2<f32>>;

@compute @workgroup_size({WORKGROUP_SIZE})
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {{
    let wg_flat = (wid.z * nwg.y + wid.y) * nwg.x + wid.x;
    if (wg_flat > (MODE_COUNT - 1u) / WORKGROUP_SIZE) {{ return; }}
    let output_index = wg_flat * WORKGROUP_SIZE + lid.x;
    if (output_index >= MODE_COUNT) {{ return; }}

    var fine_index = 0u;
    if (CENTERED_ORDER) {{
        if (output_index < HALF_MODE_COUNT) {{
            fine_index = FINE_LENGTH - (HALF_MODE_COUNT - output_index);
        }} else {{
            fine_index = output_index - HALF_MODE_COUNT;
        }}
    }} else if (output_index < NONNEGATIVE_COUNT) {{
        fine_index = output_index;
    }} else {{
        fine_index = FINE_LENGTH - (MODE_COUNT - output_index);
    }}
    output_values[output_index] = fine_grid[fine_index] * amplitudes[output_index];
}}
"#,
    )
}

fn format_wgsl_f32(value: f32) -> String {
    debug_assert!(value.is_finite());
    let mut formatted = value.to_string();
    if !formatted.contains('.') && !formatted.contains('e') && !formatted.contains('E') {
        formatted.push_str(".0");
    }
    formatted
}

fn dispatch_for_elements(
    element_count: usize,
    max_workgroups_per_dimension: u32,
) -> Result<(u32, u32, u32)> {
    let elements = u32::try_from(element_count).map_err(|_| NufftError::LengthOverflow {
        context: "type-1 GPU dispatch element count",
    })?;
    split_workgroups(
        elements.div_ceil(WORKGROUP_SIZE),
        max_workgroups_per_dimension,
    )
}

fn split_workgroups(workgroups: u32, maximum: u32) -> Result<(u32, u32, u32)> {
    if maximum == 0 {
        return Err(NufftError::GpuDispatchUnsupported {
            workgroups,
            max_per_dimension: maximum,
        });
    }
    if workgroups == 0 {
        return Ok((0, 1, 1));
    }
    let total = u64::from(workgroups);
    let maximum_u64 = u64::from(maximum);
    let z = total.div_ceil(maximum_u64 * maximum_u64);
    if z > maximum_u64 {
        return Err(NufftError::GpuDispatchUnsupported {
            workgroups,
            max_per_dimension: maximum,
        });
    }
    let per_slice = total.div_ceil(z);
    let y = per_slice.div_ceil(maximum_u64);
    let x = per_slice.div_ceil(y);
    let covered = x.saturating_mul(y).saturating_mul(z);
    if x <= maximum_u64 && y <= maximum_u64 && covered <= u64::from(u32::MAX) + 1 {
        Ok((x as u32, y as u32, z as u32))
    } else if maximum >= 2048 {
        // Exactly covers the full u32 index space without overflowing the
        // shader's flattened workgroup arithmetic.
        Ok((2048, 2048, 1024))
    } else {
        Err(NufftError::GpuDispatchUnsupported {
            workgroups,
            max_per_dimension: maximum,
        })
    }
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

fn validate_buffer_limit(buffer: &'static str, bytes: u64, limit: u64) -> Result<()> {
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

fn binding_entry(binding: u32, buffer: &wgpu::Buffer, size: u64) -> wgpu::BindGroupEntry<'_> {
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
    use crate::config::ModeOrder;

    #[test]
    fn position_bin_keeps_a_negative_df64_low_word() {
        let source = generate_count_wgsl(65_536);
        assert!(source.contains("fn floor_df64_to_i32(value: Df64) -> i32"));
        assert!(source.contains("has_negative_remainder"));
        assert!(!source.contains("u32(floor(position.hi))"));
    }

    #[test]
    fn gather_is_one_writer_and_uses_exact_interpolation_transpose() {
        let kernel = EsKernel::for_tolerance(1.0e-6, 2.0).unwrap();
        let source = generate_gather_wgsl(kernel, 64);
        assert!(source.contains("const BIN_RADIUS: i32 = 4i;"));
        assert!(source.contains("let start = ceil_df64_to_i32(shifted);"));
        assert!(source.contains("unwrapped >= start && unwrapped < start + WIDTH_I32"));
        assert!(source.contains("fine_grid[cell] = sum;"));
        assert!(!source.contains("fine_grid[cell] = fine_grid[cell] + sum"));
    }

    #[test]
    fn each_bin_is_sorted_by_original_point_index() {
        let source = generate_sort_wgsl();
        assert!(source.contains("fn heap_sort(start: u32, end: u32)"));
        assert!(source.contains("sorted_indices[start + greatest] <"));
        assert!(source.contains("heap_sort(bin_offsets[bin], bin_offsets[bin + 1u]);"));
    }

    #[test]
    fn deconvolution_maps_both_mode_orders_and_odd_lengths() {
        for order in [ModeOrder::Centered, ModeOrder::Fft] {
            let config = NufftConfig::new([17], 1.0e-6).with_mode_order(order);
            let source = generate_deconvolution_wgsl(&config, 36);
            assert!(source.contains("const MODE_COUNT: u32 = 17u;"));
            assert!(source.contains("const HALF_MODE_COUNT: u32 = 8u;"));
            assert!(source.contains("const NONNEGATIVE_COUNT: u32 = 9u;"));
            assert!(source.contains("fine_grid[fine_index] * amplitudes[output_index]"));
        }
    }

    #[test]
    fn dispatch_split_covers_exactly_the_requested_workgroups() {
        let maximum = 65_535;
        for count in [1, maximum, maximum + 1, 4_294_967_295] {
            let (x, y, z) = split_workgroups(count, maximum).unwrap();
            assert!(x <= maximum && y <= maximum && z <= maximum);
            let covered = u64::from(x) * u64::from(y) * u64::from(z);
            assert!(covered >= u64::from(count));
            assert!(covered <= u64::from(u32::MAX) + 1);
        }
        assert!(split_workgroups(1, 0).is_err());
    }

    #[test]
    fn type1_buffer_sizes_are_exact_prefix_sizes() {
        assert_eq!(Type1GpuPlan::point_buffer_size_bytes(7).unwrap(), 28);
        assert_eq!(Type1GpuPlan::strength_buffer_size_bytes(7).unwrap(), 56);
        assert_eq!(
            Type1GpuPlan::complex_buffer_size_bytes("test", 11).unwrap(),
            88
        );
    }
}
