use std::num::NonZeroU64;

use wgpu::util::DeviceExt;
use wgpu_fft::{
    math::DoubleFloat, BufferView, FftConfig, FftDirection, FftPlan, FftPrecision, Normalization,
};

use crate::config::{NufftConfig, NufftSign};
use crate::direct::mode_for_storage_index;
use crate::error::{NufftError, Result};
use crate::gpu::max_supported_workgroup_size;
use crate::gpu_dispatch::split_workgroups;
#[cfg(feature = "gpu-profiling")]
use crate::gpu_profile::{GpuProfileQueryWriter, NufftGpuProfileLayout};
use crate::gpu_scan::GpuExclusiveScanU32;
use crate::gpu_type1_scratch::Type1ScratchBuffers;
use crate::kernel::EsKernel;

const DIMENSIONS: usize = 3;
const WORKGROUP_SIZE: u32 = 64;
const VECTOR_BLOCK_SIZE: usize = 4;
const TILED_GATHER_TILE_SHAPE: [usize; DIMENSIONS] = [8, 8, 4];
const TILED_GATHER_WORKGROUP_SIZE: u32 = 256;
const TILED_GATHER_SCAN_ELEMENTS: usize = 4096;
const TILED_GATHER_SCAN_BANK_WIDTH: usize = 32;
const TILED_GATHER_PADDED_SCAN_ELEMENTS: usize =
    TILED_GATHER_SCAN_ELEMENTS + TILED_GATHER_SCAN_ELEMENTS / TILED_GATHER_SCAN_BANK_WIDTH;
const TILED_GATHER_STORAGE_BUDGET_BYTES: usize = 47 * 1024;
const TILED_GATHER_CACHE_CAPACITY: usize = 300;
const COMPLEX_F32_BYTES: u64 = 8;
const COMPLEX_F64_BYTES: u64 = 16;
const POINT_F32_BYTES: u64 = DIMENSIONS as u64 * 4;
const POINT_F64_BYTES: u64 = DIMENSIONS as u64 * 8;
const F32_BYTES: u64 = 4;
const F64_BYTES: u64 = 8;
const U32_BYTES: u64 = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Type1Gather3d {
    #[cfg_attr(not(feature = "type1-3d-tile-prototype"), allow(dead_code))]
    Global,
    Tiled8x8x4,
}

/// Device-specific resources for deterministic, atomics-free 3D spreading.
///
/// Points are point-major `x, y, z` triples. Integer atomics construct one bin
/// per fine-grid cell, indices within each flattened axis-0-fast bin are sorted
/// into original point order, and one invocation owns each fine-grid output.
pub(crate) struct Type1GpuPlan3d {
    fft: FftPlan,
    amplitudes: wgpu::Buffer,
    fine_input: wgpu::Buffer,
    fine_output: wgpu::Buffer,
    scratch: Type1ScratchBuffers,
    count_pipeline: wgpu::ComputePipeline,
    count_layout: wgpu::BindGroupLayout,
    prefix_scan: GpuExclusiveScanU32,
    terminal_pipeline: wgpu::ComputePipeline,
    terminal_layout: wgpu::BindGroupLayout,
    scatter_pipeline: wgpu::ComputePipeline,
    scatter_layout: wgpu::BindGroupLayout,
    sort_pipeline: wgpu::ComputePipeline,
    sort_layout: wgpu::BindGroupLayout,
    gather_pipeline: wgpu::ComputePipeline,
    gather_layout: wgpu::BindGroupLayout,
    deconvolution_pipeline: wgpu::ComputePipeline,
    deconvolution_layout: wgpu::BindGroupLayout,
    sort_dispatch: (u32, u32, u32),
    gather_workgroups_per_vector_block: u32,
    max_workgroups_per_dimension: u32,
    mode_count: usize,
    fine_count: usize,
    batch_capacity: usize,
    precision: FftPrecision,
    max_storage_binding_bytes: u64,
    max_buffer_bytes: u64,
}

impl Type1GpuPlan3d {
    pub(crate) fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        config: &NufftConfig,
        kernel: EsKernel,
        fine_shape: [usize; DIMENSIONS],
        kernel_fourier_coefficients: [&[f64]; DIMENSIONS],
        gather: Type1Gather3d,
    ) -> Result<Self> {
        if config.dimensions() != DIMENSIONS {
            return Err(NufftError::GpuDimensionsUnsupported {
                kind: "type-1",
                actual: config.dimensions(),
                supported: DIMENSIONS,
            });
        }
        let precision = config.precision();
        let exact_integer_limit = match precision {
            FftPrecision::F32 | FftPrecision::Df64 => 1usize << f32::MANTISSA_DIGITS,
            FftPrecision::F64 => {
                usize::try_from(1u64 << f64::MANTISSA_DIGITS).unwrap_or(usize::MAX)
            }
        };
        let maximum_signed_length = exact_integer_limit
            .saturating_sub(kernel.width())
            .min(i32::MAX as usize - kernel.width());
        for &length in &fine_shape {
            if !length.is_multiple_of(2) {
                return Err(NufftError::FineGridMustBeEven { length });
            }
            if length > maximum_signed_length {
                return Err(NufftError::GpuFineGridIndexUnsupported {
                    length,
                    maximum: maximum_signed_length,
                });
            }
            u32::try_from(length).map_err(|_| NufftError::LengthOverflow {
                context: "type-1 3D fine-grid axis length",
            })?;
        }

        let limits = device.limits();
        let maximum_workgroup_size = max_supported_workgroup_size(&limits);
        if maximum_workgroup_size < WORKGROUP_SIZE {
            return Err(NufftError::GpuWorkgroupSizeUnsupported {
                requested: WORKGROUP_SIZE,
                maximum: maximum_workgroup_size,
            });
        }

        let fft_direction = match config.sign() {
            NufftSign::Positive => FftDirection::Inverse,
            NufftSign::Negative => FftDirection::Forward,
        };
        let fft_config = FftConfig::new_nd(fine_shape)
            .with_direction(fft_direction)
            .with_normalization(Normalization::None)
            .with_precision(precision);
        fft_config
            .validate()
            .map_err(|source| NufftError::FftShapeUnsupported {
                stage: "type-1 3D oversampled-grid C2C plan",
                source,
            })?;
        let fine_count =
            fft_config
                .total_complex_len()
                .map_err(|source| NufftError::FftShapeUnsupported {
                    stage: "type-1 3D oversampled-grid C2C plan",
                    source,
                })?;
        u32::try_from(fine_count).map_err(|_| NufftError::LengthOverflow {
            context: "type-1 3D fine-grid shader index space",
        })?;
        let batch_capacity = config.batch();
        let fft_config = fft_config.with_batch(batch_capacity);
        let mode_count = config.mode_count()?;
        u32::try_from(mode_count).map_err(|_| NufftError::LengthOverflow {
            context: "type-1 3D mode shader index space",
        })?;

        let total_fine_count =
            fine_count
                .checked_mul(batch_capacity)
                .ok_or(NufftError::LengthOverflow {
                    context: "batched type-1 3D fine-grid element count",
                })?;
        let fine_bytes = Self::complex_buffer_size_bytes_for_precision(
            "type-1 3D fine grid",
            total_fine_count,
            precision,
        )?;
        let amplitude_bytes = checked_buffer_size(
            "type-1 3D deconvolution amplitudes",
            mode_count,
            scalar_size_bytes(precision),
        )?;
        let count_bytes = checked_buffer_size("type-1 3D bin counts", fine_count, U32_BYTES)?;
        let offset_count = fine_count
            .checked_add(1)
            .ok_or(NufftError::LengthOverflow {
                context: "type-1 3D bin offset count",
            })?;
        u32::try_from(offset_count).map_err(|_| NufftError::LengthOverflow {
            context: "type-1 3D bin-offset shader index space",
        })?;
        let offset_bytes = checked_buffer_size("type-1 3D bin offsets", offset_count, U32_BYTES)?;
        for (label, bytes) in [
            ("type-1 3D fine grid", fine_bytes),
            ("type-1 3D deconvolution amplitudes", amplitude_bytes),
            ("type-1 3D bin counts", count_bytes),
            ("type-1 3D bin offsets", offset_bytes),
        ] {
            validate_binding_limit(label, bytes, limits.max_storage_buffer_binding_size)?;
            validate_buffer_limit(label, bytes, limits.max_buffer_size)?;
        }

        let amplitudes = mode_amplitude_bytes(config, kernel_fourier_coefficients, precision)?;
        let amplitudes = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("wgpu_nufft.type1_3d.deconvolution_amplitudes"),
            contents: &amplitudes,
            usage: wgpu::BufferUsages::STORAGE,
        });
        let fine_usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST;
        let fine_input = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("wgpu_nufft.type1_3d.fine_input"),
            size: fine_bytes,
            usage: fine_usage,
            mapped_at_creation: false,
        });
        let fine_output = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("wgpu_nufft.type1_3d.fine_output"),
            size: fine_bytes,
            usage: fine_usage,
            mapped_at_creation: false,
        });
        let scratch =
            Type1ScratchBuffers::new(device, "wgpu_nufft.type1_3d", count_bytes, offset_bytes);
        let fft = FftPlan::c2c(device, queue, fft_config).map_err(|source| {
            NufftError::FftShapeUnsupported {
                stage: "type-1 3D oversampled-grid C2C plan",
                source,
            }
        })?;

        let count_pipeline = create_compute_pipeline(
            device,
            "wgpu_nufft.type1_3d.bin_count",
            &generate_count_wgsl_for_precision(fine_shape, precision),
        );
        let count_layout = count_pipeline.get_bind_group_layout(0);
        let prefix_scan = GpuExclusiveScanU32::new(device, fine_count)?;
        let terminal_pipeline = create_compute_pipeline(
            device,
            "wgpu_nufft.type1_3d.bin_terminal",
            &generate_terminal_wgsl(),
        );
        let terminal_layout = terminal_pipeline.get_bind_group_layout(0);
        let scatter_pipeline = create_compute_pipeline(
            device,
            "wgpu_nufft.type1_3d.bin_scatter",
            &generate_scatter_wgsl_for_precision(fine_shape, precision),
        );
        let scatter_layout = scatter_pipeline.get_bind_group_layout(0);
        let sort_pipeline = create_compute_pipeline(
            device,
            "wgpu_nufft.type1_3d.bin_sort",
            &generate_sort_wgsl(),
        );
        let sort_layout = sort_pipeline.get_bind_group_layout(0);
        let max_workgroups_per_dimension = limits.max_compute_workgroups_per_dimension;
        let (gather_label, gather_source, gather_workgroups_per_vector_block) = match precision {
            FftPrecision::F64 => (
                "wgpu_nufft.type1_3d.spread_gather_f64",
                generate_gather_wgsl_f64(kernel, fine_shape),
                workgroups_for_elements(fine_count)?,
            ),
            FftPrecision::Df64 => (
                "wgpu_nufft.type1_3d.spread_gather_df64",
                generate_gather_wgsl_df64(kernel, fine_shape),
                workgroups_for_elements(fine_count)?,
            ),
            FftPrecision::F32 => match gather {
                Type1Gather3d::Global => (
                    "wgpu_nufft.type1_3d.spread_gather",
                    generate_gather_wgsl(kernel, fine_shape),
                    workgroups_for_elements(fine_count)?,
                ),
                Type1Gather3d::Tiled8x8x4 => {
                    let configuration =
                        tiled_gather_halo_shape(kernel.width()).and_then(|halo_shape| {
                            let cache_capacity = tiled_gather_cache_capacity(kernel.width())?;
                            let storage_bytes =
                                tiled_gather_storage_bytes(kernel.width(), cache_capacity)?;
                            Ok((halo_shape, cache_capacity, storage_bytes))
                        });
                    match configuration {
                        Ok((halo_shape, cache_capacity, storage_bytes))
                            if maximum_workgroup_size >= TILED_GATHER_WORKGROUP_SIZE
                                && fine_shape
                                    .iter()
                                    .zip(halo_shape)
                                    .all(|(&length, halo)| length >= halo)
                                && storage_bytes <= limits.max_compute_workgroup_storage_size =>
                        {
                            (
                                "wgpu_nufft.type1_3d.spread_gather_tiled8x8x4",
                                generate_tiled_gather_wgsl(kernel, fine_shape, cache_capacity)?,
                                tiled_gather_workgroups(fine_shape)?,
                            )
                        }
                        _ => (
                            "wgpu_nufft.type1_3d.spread_gather",
                            generate_gather_wgsl(kernel, fine_shape),
                            workgroups_for_elements(fine_count)?,
                        ),
                    }
                }
            },
        };
        let gather_pipeline = create_compute_pipeline(device, gather_label, &gather_source);
        let gather_layout = gather_pipeline.get_bind_group_layout(0);
        let deconvolution_pipeline = create_compute_pipeline(
            device,
            "wgpu_nufft.type1_3d.deconvolution",
            &generate_deconvolution_wgsl_for_precision(config, fine_shape, precision),
        );
        let deconvolution_layout = deconvolution_pipeline.get_bind_group_layout(0);

        let sort_dispatch = dispatch_for_elements(fine_count, max_workgroups_per_dimension)?;
        Ok(Self {
            fft,
            amplitudes,
            fine_input,
            fine_output,
            scratch,
            count_pipeline,
            count_layout,
            prefix_scan,
            terminal_pipeline,
            terminal_layout,
            scatter_pipeline,
            scatter_layout,
            sort_pipeline,
            sort_layout,
            gather_pipeline,
            gather_layout,
            deconvolution_pipeline,
            deconvolution_layout,
            sort_dispatch,
            gather_workgroups_per_vector_block,
            max_workgroups_per_dimension,
            mode_count,
            fine_count,
            batch_capacity,
            precision,
            max_storage_binding_bytes: limits.max_storage_buffer_binding_size,
            max_buffer_bytes: limits.max_buffer_size,
        })
    }

    pub(crate) fn point_buffer_size_bytes_for_precision(
        point_count: usize,
        precision: FftPrecision,
    ) -> Result<u64> {
        let bytes = match precision {
            FftPrecision::F32 => POINT_F32_BYTES,
            FftPrecision::F64 | FftPrecision::Df64 => POINT_F64_BYTES,
        };
        checked_buffer_size("type-1 3D point buffer", point_count, bytes)
    }

    pub(crate) fn strength_buffer_size_bytes_for_precision(
        point_count: usize,
        precision: FftPrecision,
    ) -> Result<u64> {
        Self::complex_buffer_size_bytes_for_precision(
            "type-1 3D strength buffer",
            point_count,
            precision,
        )
    }

    pub(crate) fn complex_buffer_size_bytes_for_precision(
        buffer: &'static str,
        element_count: usize,
        precision: FftPrecision,
    ) -> Result<u64> {
        checked_buffer_size(buffer, element_count, complex_size_bytes(precision))
    }

    /// Records only the deterministic point spreading stages into the
    /// oversampled fine grid. The grid remains owned by this plan and is
    /// overwritten on every call, including the zero-point case.
    #[allow(dead_code)]
    pub(crate) fn encode_spread(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        point_count: usize,
        points: &wgpu::Buffer,
        strengths: &wgpu::Buffer,
    ) -> Result<()> {
        self.encode_spread_batch(device, encoder, 1, point_count, points, strengths)
    }

    pub(crate) fn encode_spread_batch(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        active_batch: usize,
        point_count: usize,
        points: &wgpu::Buffer,
        strengths: &wgpu::Buffer,
    ) -> Result<()> {
        self.validate_active_batch(active_batch)?;
        self.encode_spread_impl(
            device,
            encoder,
            active_batch,
            point_count,
            points,
            strengths,
            #[cfg(feature = "gpu-profiling")]
            GpuProfileQueryWriter::disabled(),
        )
    }

    /// Returns the plan-owned fine grid populated by [`Self::encode_spread`].
    pub(crate) fn fine_grid_buffer(&self) -> &wgpu::Buffer {
        &self.fine_input
    }

    pub(crate) const fn fine_grid_element_count(&self) -> usize {
        self.fine_count
    }

    #[allow(dead_code)]
    pub(crate) fn encode(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        point_count: usize,
        points: &wgpu::Buffer,
        strengths: &wgpu::Buffer,
        output: &wgpu::Buffer,
    ) -> Result<()> {
        self.encode_batch(
            device,
            encoder,
            self.batch_capacity,
            point_count,
            points,
            strengths,
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
        strengths: &wgpu::Buffer,
        output: &wgpu::Buffer,
    ) -> Result<()> {
        self.validate_active_batch(active_batch)?;
        self.encode_impl(
            device,
            encoder,
            active_batch,
            point_count,
            points,
            strengths,
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
        strengths: &wgpu::Buffer,
        output: &wgpu::Buffer,
        query_set: &wgpu::QuerySet,
        first_query: u32,
    ) -> Result<NufftGpuProfileLayout> {
        if point_count == 0 {
            return Err(NufftError::GpuExecutionUnavailable {
                kind: "type-1 3D stage profiling",
                reason: "at least one point is required",
            });
        }
        let layout =
            NufftGpuProfileLayout::type1(first_query).map_err(|_| NufftError::LengthOverflow {
                context: "type-1 3D stage-profile query range",
            })?;
        self.encode_impl(
            device,
            encoder,
            self.batch_capacity,
            point_count,
            points,
            strengths,
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
        strengths: &wgpu::Buffer,
        output: &wgpu::Buffer,
        #[cfg(feature = "gpu-profiling")] profile: GpuProfileQueryWriter<'_>,
    ) -> Result<()> {
        let output_elements =
            self.mode_count
                .checked_mul(active_batch)
                .ok_or(NufftError::LengthOverflow {
                    context: "batched type-1 3D Fourier output element count",
                })?;
        let output_bytes = Self::complex_buffer_size_bytes_for_precision(
            "type-1 3D Fourier output buffer",
            output_elements,
            self.precision,
        )?;
        validate_external_storage_buffer(
            "type-1 3D Fourier output",
            output,
            output_bytes,
            self.max_storage_binding_bytes,
        )?;

        let output_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("wgpu_nufft.type1_3d.deconvolution.bind_group"),
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

        #[cfg(not(feature = "gpu-profiling"))]
        self.encode_spread_batch(
            device,
            encoder,
            active_batch,
            point_count,
            points,
            strengths,
        )?;
        #[cfg(feature = "gpu-profiling")]
        match profile {
            GpuProfileQueryWriter::Disabled => {
                self.encode_spread_batch(
                    device,
                    encoder,
                    active_batch,
                    point_count,
                    points,
                    strengths,
                )?;
            }
            GpuProfileQueryWriter::Enabled { .. } => {
                self.encode_spread_impl(
                    device,
                    encoder,
                    active_batch,
                    point_count,
                    points,
                    strengths,
                    profile,
                )?;
            }
        }
        self.encode_fft(device, encoder)?;
        let deconvolution_dispatch =
            dispatch_for_elements(output_elements, self.max_workgroups_per_dimension)?;
        self.encode_deconvolution(
            encoder,
            &output_bind_group,
            deconvolution_dispatch,
            #[cfg(feature = "gpu-profiling")]
            profile,
        );
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_spread_impl(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        active_batch: usize,
        point_count: usize,
        points: &wgpu::Buffer,
        strengths: &wgpu::Buffer,
        #[cfg(feature = "gpu-profiling")] profile: GpuProfileQueryWriter<'_>,
    ) -> Result<()> {
        if point_count == 0 {
            encoder.clear_buffer(self.fine_grid_buffer(), 0, None);
            return Ok(());
        }

        let point_count_u32 =
            u32::try_from(point_count).map_err(|_| NufftError::LengthOverflow {
                context: "type-1 3D GPU point count",
            })?;
        validate_point_coordinate_shader_index(point_count)?;
        let point_bytes = Self::point_buffer_size_bytes_for_precision(point_count, self.precision)?;
        let strength_elements =
            point_count
                .checked_mul(active_batch)
                .ok_or(NufftError::LengthOverflow {
                    context: "batched type-1 3D strength element count",
                })?;
        let strength_bytes =
            Self::strength_buffer_size_bytes_for_precision(strength_elements, self.precision)?;
        validate_external_storage_buffer(
            "type-1 3D point",
            points,
            point_bytes,
            self.max_storage_binding_bytes,
        )?;
        validate_external_storage_buffer(
            "type-1 3D strength",
            strengths,
            strength_bytes,
            self.max_storage_binding_bytes,
        )?;

        let fine_grid_element_count = self.fine_grid_element_count();
        let count_bytes =
            checked_buffer_size("type-1 3D bin counts", fine_grid_element_count, U32_BYTES)?;
        let offset_count =
            self.fine_grid_element_count()
                .checked_add(1)
                .ok_or(NufftError::LengthOverflow {
                    context: "type-1 3D bin offset count",
                })?;
        let offset_bytes = checked_buffer_size("type-1 3D bin offsets", offset_count, U32_BYTES)?;
        let index_bytes =
            checked_buffer_size("type-1 3D sorted point indices", point_count, U32_BYTES)?;
        for (label, bytes) in [
            ("type-1 3D bin counts", count_bytes),
            ("type-1 3D bin cursors", count_bytes),
            ("type-1 3D bin offsets", offset_bytes),
            ("type-1 3D sorted point indices", index_bytes),
        ] {
            validate_binding_limit(label, bytes, self.max_storage_binding_bytes)?;
            validate_buffer_limit(label, bytes, self.max_buffer_bytes)?;
        }
        let point_dispatch = split_workgroups(
            point_count_u32.div_ceil(WORKGROUP_SIZE),
            self.max_workgroups_per_dimension,
        )?;

        let bin_counts = &self.scratch.bin_counts;
        let bin_cursors = &self.scratch.bin_cursors;
        let bin_offsets = &self.scratch.bin_offsets;
        let sorted_indices = self.scratch.sorted_indices(index_bytes);

        let count_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("wgpu_nufft.type1_3d.bin_count.bind_group"),
            layout: &self.count_layout,
            entries: &[
                binding_entry(0, points, point_bytes),
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: bin_counts.as_entire_binding(),
                },
            ],
        });
        let terminal_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("wgpu_nufft.type1_3d.bin_terminal.bind_group"),
            layout: &self.terminal_layout,
            entries: &[
                binding_entry(0, bin_counts, count_bytes),
                binding_entry(1, bin_offsets, offset_bytes),
            ],
        });
        let scatter_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("wgpu_nufft.type1_3d.bin_scatter.bind_group"),
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
                binding_entry(3, &sorted_indices, index_bytes),
            ],
        });
        let sort_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("wgpu_nufft.type1_3d.bin_sort.bind_group"),
            layout: &self.sort_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: bin_offsets.as_entire_binding(),
                },
                binding_entry(1, &sorted_indices, index_bytes),
            ],
        });
        let active_fine_elements = fine_grid_element_count.checked_mul(active_batch).ok_or(
            NufftError::LengthOverflow {
                context: "active batched type-1 3D fine-grid element count",
            },
        )?;
        let active_fine_bytes = Self::complex_buffer_size_bytes_for_precision(
            "active batched type-1 3D fine grid",
            active_fine_elements,
            self.precision,
        )?;
        let gather_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("wgpu_nufft.type1_3d.spread_gather.bind_group"),
            layout: &self.gather_layout,
            entries: &[
                binding_entry(0, points, point_bytes),
                binding_entry(1, strengths, strength_bytes),
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: bin_offsets.as_entire_binding(),
                },
                binding_entry(3, &sorted_indices, index_bytes),
                binding_entry(4, &self.fine_input, active_fine_bytes),
            ],
        });
        let vector_blocks = active_batch.div_ceil(VECTOR_BLOCK_SIZE);
        let gather_workgroups = usize::try_from(self.gather_workgroups_per_vector_block)
            .ok()
            .and_then(|count| count.checked_mul(vector_blocks))
            .and_then(|count| u32::try_from(count).ok())
            .ok_or(NufftError::LengthOverflow {
                context: "batched type-1 3D gather workgroup count",
            })?;
        let gather_dispatch =
            split_workgroups(gather_workgroups, self.max_workgroups_per_dimension)?;

        #[cfg(feature = "gpu-profiling")]
        profile.encode_type1_start_marker(encoder);
        encoder.clear_buffer(bin_counts, 0, None);
        encoder.clear_buffer(bin_cursors, 0, None);
        encode_pass(
            encoder,
            "wgpu_nufft.type1_3d.bin_count.pass",
            &self.count_pipeline,
            &count_bind_group,
            point_dispatch,
            #[cfg(feature = "gpu-profiling")]
            profile.timestamp_writes(None, Some(1)),
        );
        self.prefix_scan
            .encode(device, encoder, bin_counts, bin_offsets)?;
        encode_pass(
            encoder,
            "wgpu_nufft.type1_3d.bin_terminal.pass",
            &self.terminal_pipeline,
            &terminal_bind_group,
            (1, 1, 1),
            #[cfg(feature = "gpu-profiling")]
            profile.timestamp_writes(None, Some(2)),
        );
        encode_pass(
            encoder,
            "wgpu_nufft.type1_3d.bin_scatter.pass",
            &self.scatter_pipeline,
            &scatter_bind_group,
            point_dispatch,
            #[cfg(feature = "gpu-profiling")]
            profile.timestamp_writes(None, Some(3)),
        );
        encode_pass(
            encoder,
            "wgpu_nufft.type1_3d.bin_sort.pass",
            &self.sort_pipeline,
            &sort_bind_group,
            self.sort_dispatch,
            #[cfg(feature = "gpu-profiling")]
            profile.timestamp_writes(None, Some(4)),
        );
        encode_pass(
            encoder,
            "wgpu_nufft.type1_3d.spread_gather.pass",
            &self.gather_pipeline,
            &gather_bind_group,
            gather_dispatch,
            #[cfg(feature = "gpu-profiling")]
            profile.timestamp_writes(None, Some(5)),
        );
        if active_batch < self.batch_capacity {
            encoder.clear_buffer(
                &self.fine_input,
                active_fine_bytes,
                Some(self.fine_input.size() - active_fine_bytes),
            );
        }
        Ok(())
    }

    fn encode_fft(&self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder) -> Result<()> {
        self.fft
            .execute_views(
                device,
                encoder,
                BufferView::whole(self.fine_grid_buffer()),
                BufferView::whole(&self.fine_output),
            )
            .map_err(|source| NufftError::FftExecutionFailed {
                stage: "type-1 3D oversampled-grid C2C transform",
                source,
            })
    }

    fn encode_deconvolution(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        bind_group: &wgpu::BindGroup,
        dispatch: (u32, u32, u32),
        #[cfg(feature = "gpu-profiling")] profile: GpuProfileQueryWriter<'_>,
    ) {
        encode_pass(
            encoder,
            "wgpu_nufft.type1_3d.deconvolution.pass",
            &self.deconvolution_pipeline,
            bind_group,
            dispatch,
            #[cfg(feature = "gpu-profiling")]
            profile.timestamp_writes(Some(6), Some(7)),
        );
    }

    #[cfg(feature = "gpu-profiling")]
    pub(crate) fn fft_diagnostics(&self) -> wgpu_fft::FftDiagnostics {
        self.fft.diagnostics()
    }

    fn validate_active_batch(&self, active_batch: usize) -> Result<()> {
        if active_batch == 0 || active_batch > self.batch_capacity {
            Err(NufftError::InvalidBatch {
                actual: active_batch,
                maximum: self.batch_capacity,
            })
        } else {
            Ok(())
        }
    }
}

fn mode_amplitudes(config: &NufftConfig, coefficients: [&[f64]; DIMENSIONS]) -> Result<Vec<f32>> {
    let mode_shape = config.n_modes();
    debug_assert_eq!(mode_shape.len(), DIMENSIONS);
    for axis in 0..DIMENSIONS {
        let required = mode_shape[axis] / 2 + 1;
        if coefficients[axis].len() < required {
            return Err(NufftError::InputLength {
                input: coefficient_axis_label(axis),
                expected: required,
                actual: coefficients[axis].len(),
            });
        }
    }
    let mode_count = config.mode_count()?;
    let mut amplitudes = Vec::new();
    amplitudes
        .try_reserve_exact(mode_count)
        .map_err(|_| NufftError::HostAllocationFailed {
            buffer: "type-1 3D deconvolution amplitudes",
            elements: mode_count,
        })?;
    for linear in 0..mode_count {
        let storage_0 = linear % mode_shape[0];
        let remaining = linear / mode_shape[0];
        let storage_1 = remaining % mode_shape[1];
        let storage_2 = remaining / mode_shape[1];
        let modes = [
            mode_for_storage_index(config.mode_order(), storage_0, mode_shape[0]),
            mode_for_storage_index(config.mode_order(), storage_1, mode_shape[1]),
            mode_for_storage_index(config.mode_order(), storage_2, mode_shape[2]),
        ];
        let values = [
            coefficients[0][modes[0].unsigned_abs() as usize],
            coefficients[1][modes[1].unsigned_abs() as usize],
            coefficients[2][modes[2].unsigned_abs() as usize],
        ];
        for (&mode, &coefficient) in modes.iter().zip(&values) {
            if !coefficient.is_finite() || coefficient == 0.0 {
                return Err(NufftError::InvalidKernelFourierCoefficient {
                    mode,
                    value: coefficient,
                });
            }
        }
        let denominator = values[0] * values[1] * values[2];
        let amplitude = (1.0 / denominator) as f32;
        if !denominator.is_finite() || denominator == 0.0 || !amplitude.is_finite() {
            return Err(NufftError::InvalidKernelFourierCoefficient {
                mode: modes[0],
                value: denominator,
            });
        }
        amplitudes.push(amplitude);
    }
    Ok(amplitudes)
}

fn mode_amplitudes_f64(
    config: &NufftConfig,
    coefficients: [&[f64]; DIMENSIONS],
) -> Result<Vec<f64>> {
    let mode_shape = config.n_modes();
    for axis in 0..DIMENSIONS {
        let required = mode_shape[axis] / 2 + 1;
        if coefficients[axis].len() < required {
            return Err(NufftError::InputLength {
                input: coefficient_axis_label(axis),
                expected: required,
                actual: coefficients[axis].len(),
            });
        }
    }
    let mode_count = config.mode_count()?;
    let mut amplitudes = Vec::new();
    amplitudes
        .try_reserve_exact(mode_count)
        .map_err(|_| NufftError::HostAllocationFailed {
            buffer: "type-1 3D deconvolution amplitudes",
            elements: mode_count,
        })?;
    for linear in 0..mode_count {
        let storage_0 = linear % mode_shape[0];
        let remaining = linear / mode_shape[0];
        let storage_1 = remaining % mode_shape[1];
        let storage_2 = remaining / mode_shape[1];
        let modes = [
            mode_for_storage_index(config.mode_order(), storage_0, mode_shape[0]),
            mode_for_storage_index(config.mode_order(), storage_1, mode_shape[1]),
            mode_for_storage_index(config.mode_order(), storage_2, mode_shape[2]),
        ];
        let values = [
            coefficients[0][modes[0].unsigned_abs() as usize],
            coefficients[1][modes[1].unsigned_abs() as usize],
            coefficients[2][modes[2].unsigned_abs() as usize],
        ];
        let denominator = values[0] * values[1] * values[2];
        let amplitude = 1.0 / denominator;
        if values
            .iter()
            .any(|value| !value.is_finite() || *value == 0.0)
            || !denominator.is_finite()
            || denominator == 0.0
            || !amplitude.is_finite()
        {
            return Err(NufftError::InvalidKernelFourierCoefficient {
                mode: modes[0],
                value: denominator,
            });
        }
        amplitudes.push(amplitude);
    }
    Ok(amplitudes)
}

fn mode_amplitude_bytes(
    config: &NufftConfig,
    coefficients: [&[f64]; DIMENSIONS],
    precision: FftPrecision,
) -> Result<Vec<u8>> {
    match precision {
        FftPrecision::F64 => {
            Ok(bytemuck::cast_slice(&mode_amplitudes_f64(config, coefficients)?).to_vec())
        }
        FftPrecision::Df64 => {
            let values = mode_amplitudes_f64(config, coefficients)?
                .into_iter()
                .map(DoubleFloat::from_f64)
                .collect::<Vec<_>>();
            Ok(bytemuck::cast_slice(&values).to_vec())
        }
        FftPrecision::F32 => {
            Ok(bytemuck::cast_slice(&mode_amplitudes(config, coefficients)?).to_vec())
        }
    }
}

const fn scalar_size_bytes(precision: FftPrecision) -> u64 {
    match precision {
        FftPrecision::F32 => F32_BYTES,
        FftPrecision::F64 | FftPrecision::Df64 => F64_BYTES,
    }
}

const fn complex_size_bytes(precision: FftPrecision) -> u64 {
    match precision {
        FftPrecision::F32 => COMPLEX_F32_BYTES,
        FftPrecision::F64 | FftPrecision::Df64 => COMPLEX_F64_BYTES,
    }
}

fn coefficient_axis_label(axis: usize) -> &'static str {
    match axis {
        0 => "axis-0 centered kernel Fourier coefficients",
        1 => "axis-1 centered kernel Fourier coefficients",
        _ => "axis-2 centered kernel Fourier coefficients",
    }
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
    #[cfg(feature = "gpu-profiling")] timestamp_writes: Option<
        wgpu::ComputePassTimestampWrites<'_>,
    >,
) {
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some(label),
        timestamp_writes: {
            #[cfg(feature = "gpu-profiling")]
            {
                timestamp_writes
            }
            #[cfg(not(feature = "gpu-profiling"))]
            {
                None
            }
        },
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, bind_group, &[]);
    pass.dispatch_workgroups(dispatch.0, dispatch.1, dispatch.2);
}

fn generate_position_wgsl(fine_shape: [usize; DIMENSIONS]) -> String {
    let scales = [
        fine_shape[0] as f64 / std::f64::consts::TAU,
        fine_shape[1] as f64 / std::f64::consts::TAU,
        fine_shape[2] as f64 / std::f64::consts::TAU,
    ];
    let scale_hi = [scales[0] as f32, scales[1] as f32, scales[2] as f32];
    let scale_lo = [
        (scales[0] - f64::from(scale_hi[0])) as f32,
        (scales[1] - f64::from(scale_hi[1])) as f32,
        (scales[2] - f64::from(scale_hi[2])) as f32,
    ];
    format!(
        r#"const FINE_0: u32 = {fine_0}u;
const FINE_1: u32 = {fine_1}u;
const FINE_2: u32 = {fine_2}u;
const FINE_01: u32 = {fine_01}u;
const FINE_0_I32: i32 = {fine_0}i;
const FINE_1_I32: i32 = {fine_1}i;
const FINE_2_I32: i32 = {fine_2}i;
const FINE_0_F32: f32 = {fine_0_f32};
const FINE_1_F32: f32 = {fine_1_f32};
const FINE_2_F32: f32 = {fine_2_f32};
const FINE_COUNT: u32 = {fine_count}u;
const POSITION_SCALE_0_HI: f32 = {scale_0_hi};
const POSITION_SCALE_0_LO: f32 = {scale_0_lo};
const POSITION_SCALE_1_HI: f32 = {scale_1_hi};
const POSITION_SCALE_1_LO: f32 = {scale_1_lo};
const POSITION_SCALE_2_HI: f32 = {scale_2_hi};
const POSITION_SCALE_2_LO: f32 = {scale_2_lo};
const GRID_ORIGIN_0: f32 = {origin_0};
const GRID_ORIGIN_1: f32 = {origin_1};
const GRID_ORIGIN_2: f32 = {origin_2};

fn position_is_negative(value: Df64) -> bool {{
    return value.hi < 0.0 || (value.hi == 0.0 && value.lo < 0.0);
}}

fn position_at_least_grid(value: Df64, fine_length: f32) -> bool {{
    return value.hi > fine_length ||
        (value.hi == fine_length && value.lo >= 0.0);
}}

fn fold_position(
    point: f32,
    scale_hi: f32,
    scale_lo: f32,
    grid_origin: f32,
    fine_length: f32,
) -> Df64 {{
    let scaled = df64_mul(Df64(point, 0.0), Df64(scale_hi, scale_lo));
    var position = df64_add(scaled, Df64(grid_origin, 0.0));
    if (position_is_negative(position)) {{
        position = df64_add(position, Df64(fine_length, 0.0));
    }}
    if (position_is_negative(position)) {{
        position = df64_add(position, Df64(fine_length, 0.0));
    }}
    if (position_at_least_grid(position, fine_length)) {{
        position = df64_sub(position, Df64(fine_length, 0.0));
    }}
    if (position_at_least_grid(position, fine_length)) {{
        position = df64_sub(position, Df64(fine_length, 0.0));
    }}
    return position;
}}

fn fold_position_0(point: f32) -> Df64 {{
    return fold_position(point, POSITION_SCALE_0_HI, POSITION_SCALE_0_LO,
        GRID_ORIGIN_0, FINE_0_F32);
}}

fn fold_position_1(point: f32) -> Df64 {{
    return fold_position(point, POSITION_SCALE_1_HI, POSITION_SCALE_1_LO,
        GRID_ORIGIN_1, FINE_1_F32);
}}

fn fold_position_2(point: f32) -> Df64 {{
    return fold_position(point, POSITION_SCALE_2_HI, POSITION_SCALE_2_LO,
        GRID_ORIGIN_2, FINE_2_F32);
}}

fn floor_df64_to_i32(value: Df64) -> i32 {{
    let base = floor(value.hi);
    let remainder = df64_sub(value, Df64(base, 0.0));
    let has_negative_remainder = remainder.hi < 0.0 ||
        (remainder.hi == 0.0 && remainder.lo < 0.0);
    return i32(base) - select(0, 1, has_negative_remainder);
}}
"#,
        fine_0 = fine_shape[0],
        fine_1 = fine_shape[1],
        fine_2 = fine_shape[2],
        fine_01 = fine_shape[0] * fine_shape[1],
        fine_0_f32 = format_wgsl_f32(fine_shape[0] as f32),
        fine_1_f32 = format_wgsl_f32(fine_shape[1] as f32),
        fine_2_f32 = format_wgsl_f32(fine_shape[2] as f32),
        fine_count = fine_shape[0] * fine_shape[1] * fine_shape[2],
        scale_0_hi = format_wgsl_f32(scale_hi[0]),
        scale_0_lo = format_wgsl_f32(scale_lo[0]),
        scale_1_hi = format_wgsl_f32(scale_hi[1]),
        scale_1_lo = format_wgsl_f32(scale_lo[1]),
        scale_2_hi = format_wgsl_f32(scale_hi[2]),
        scale_2_lo = format_wgsl_f32(scale_lo[2]),
        origin_0 = format_wgsl_f32((fine_shape[0] / 2) as f32),
        origin_1 = format_wgsl_f32((fine_shape[1] / 2) as f32),
        origin_2 = format_wgsl_f32((fine_shape[2] / 2) as f32),
    )
}

fn generate_position_wgsl_df64(fine_shape: [usize; DIMENSIONS]) -> String {
    generate_position_wgsl(fine_shape)
        .replace("point: f32,", "point: Df64,")
        .replace("Df64(point, 0.0)", "point")
        .replace(
            "fn fold_position_0(point: f32) -> Df64",
            "fn fold_position_0(point: Df64) -> Df64",
        )
        .replace(
            "fn fold_position_1(point: f32) -> Df64",
            "fn fold_position_1(point: Df64) -> Df64",
        )
        .replace(
            "fn fold_position_2(point: f32) -> Df64",
            "fn fold_position_2(point: Df64) -> Df64",
        )
}

fn generate_position_wgsl_f64(fine_shape: [usize; DIMENSIONS]) -> String {
    let scales = [
        fine_shape[0] as f64 / std::f64::consts::TAU,
        fine_shape[1] as f64 / std::f64::consts::TAU,
        fine_shape[2] as f64 / std::f64::consts::TAU,
    ];
    format!(
        r#"const FINE_0: u32 = {fine_0}u;
const FINE_1: u32 = {fine_1}u;
const FINE_2: u32 = {fine_2}u;
const FINE_01: u32 = {fine_01}u;
const FINE_0_I32: i32 = {fine_0}i;
const FINE_1_I32: i32 = {fine_1}i;
const FINE_2_I32: i32 = {fine_2}i;
const FINE_0_F64: f64 = {fine_0_f64};
const FINE_1_F64: f64 = {fine_1_f64};
const FINE_2_F64: f64 = {fine_2_f64};
const FINE_COUNT: u32 = {fine_count}u;
const POSITION_SCALE_0: f64 = {scale_0};
const POSITION_SCALE_1: f64 = {scale_1};
const POSITION_SCALE_2: f64 = {scale_2};
const GRID_ORIGIN_0: f64 = {origin_0};
const GRID_ORIGIN_1: f64 = {origin_1};
const GRID_ORIGIN_2: f64 = {origin_2};

fn fold_position(point: f64, scale: f64, origin: f64, fine_length: f64) -> f64 {{
    var position = point * scale + origin;
    if (position < 0.0lf) {{ position = position + fine_length; }}
    if (position < 0.0lf) {{ position = position + fine_length; }}
    if (position >= fine_length) {{ position = position - fine_length; }}
    if (position >= fine_length) {{ position = position - fine_length; }}
    return position;
}}

fn fold_position_0(point: f64) -> f64 {{
    return fold_position(point, POSITION_SCALE_0, GRID_ORIGIN_0, FINE_0_F64);
}}
fn fold_position_1(point: f64) -> f64 {{
    return fold_position(point, POSITION_SCALE_1, GRID_ORIGIN_1, FINE_1_F64);
}}
fn fold_position_2(point: f64) -> f64 {{
    return fold_position(point, POSITION_SCALE_2, GRID_ORIGIN_2, FINE_2_F64);
}}
"#,
        fine_0 = fine_shape[0],
        fine_1 = fine_shape[1],
        fine_2 = fine_shape[2],
        fine_01 = fine_shape[0] * fine_shape[1],
        fine_0_f64 = format_wgsl_f64(fine_shape[0] as f64),
        fine_1_f64 = format_wgsl_f64(fine_shape[1] as f64),
        fine_2_f64 = format_wgsl_f64(fine_shape[2] as f64),
        fine_count = fine_shape[0] * fine_shape[1] * fine_shape[2],
        scale_0 = format_wgsl_f64(scales[0]),
        scale_1 = format_wgsl_f64(scales[1]),
        scale_2 = format_wgsl_f64(scales[2]),
        origin_0 = format_wgsl_f64((fine_shape[0] / 2) as f64),
        origin_1 = format_wgsl_f64((fine_shape[1] / 2) as f64),
        origin_2 = format_wgsl_f64((fine_shape[2] / 2) as f64),
    )
}

fn generate_count_wgsl_for_precision(
    fine_shape: [usize; DIMENSIONS],
    precision: FftPrecision,
) -> String {
    match precision {
        FftPrecision::F64 => generate_count_wgsl_f64(fine_shape),
        FftPrecision::Df64 => generate_count_wgsl_df64(fine_shape),
        FftPrecision::F32 => generate_count_wgsl(fine_shape),
    }
}

fn generate_count_wgsl_df64(fine_shape: [usize; DIMENSIONS]) -> String {
    let position = generate_position_wgsl_df64(fine_shape);
    let entry = format!(
        r#"const WORKGROUP_SIZE: u32 = {WORKGROUP_SIZE}u;
@group(0) @binding(0) var<storage, read> points: array<Df64>;
@group(0) @binding(1) var<storage, read_write> bin_counts: array<atomic<u32>>;

@compute @workgroup_size({WORKGROUP_SIZE})
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {{
    let total = arrayLength(&points) / 3u;
    let wg_flat = (wid.z * nwg.y + wid.y) * nwg.x + wid.x;
    if (wg_flat > (total - 1u) / WORKGROUP_SIZE) {{ return; }}
    let point_index = wg_flat * WORKGROUP_SIZE + lid.x;
    if (point_index >= total) {{ return; }}
    let point_base = point_index * 3u;
    let bin_0 = u32(floor_df64_to_i32(fold_position_0(points[point_base])));
    let bin_1 = u32(floor_df64_to_i32(fold_position_1(points[point_base + 1u])));
    let bin_2 = u32(floor_df64_to_i32(fold_position_2(points[point_base + 2u])));
    atomicAdd(&bin_counts[bin_0 + FINE_0 * (bin_1 + FINE_1 * bin_2)], 1u);
}}
"#,
    );
    format!("{}\n{position}\n{entry}", wgpu_fft::kernels::DF64_WGSL)
}

fn generate_count_wgsl_f64(fine_shape: [usize; DIMENSIONS]) -> String {
    let position = generate_position_wgsl_f64(fine_shape);
    let entry = format!(
        r#"const WORKGROUP_SIZE: u32 = {WORKGROUP_SIZE}u;
@group(0) @binding(0) var<storage, read> points: array<f64>;
@group(0) @binding(1) var<storage, read_write> bin_counts: array<atomic<u32>>;

@compute @workgroup_size({WORKGROUP_SIZE})
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {{
    let total = arrayLength(&points) / 3u;
    let wg_flat = (wid.z * nwg.y + wid.y) * nwg.x + wid.x;
    if (wg_flat > (total - 1u) / WORKGROUP_SIZE) {{ return; }}
    let point_index = wg_flat * WORKGROUP_SIZE + lid.x;
    if (point_index >= total) {{ return; }}
    let point_base = point_index * 3u;
    let bin_0 = u32(floor(fold_position_0(points[point_base])));
    let bin_1 = u32(floor(fold_position_1(points[point_base + 1u])));
    let bin_2 = u32(floor(fold_position_2(points[point_base + 2u])));
    atomicAdd(&bin_counts[bin_0 + FINE_0 * (bin_1 + FINE_1 * bin_2)], 1u);
}}
"#,
    );
    format!("{position}\n{entry}")
}

fn generate_count_wgsl(fine_shape: [usize; DIMENSIONS]) -> String {
    let position = generate_position_wgsl(fine_shape);
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
    let total = arrayLength(&points) / 3u;
    let wg_flat = (wid.z * nwg.y + wid.y) * nwg.x + wid.x;
    if (wg_flat > (total - 1u) / WORKGROUP_SIZE) {{ return; }}
    let point_index = wg_flat * WORKGROUP_SIZE + lid.x;
    if (point_index >= total) {{ return; }}

    let point_base = point_index * 3u;
    let position_0 = fold_position_0(points[point_base]);
    let position_1 = fold_position_1(points[point_base + 1u]);
    let position_2 = fold_position_2(points[point_base + 2u]);
    let bin_0 = u32(floor_df64_to_i32(position_0));
    let bin_1 = u32(floor_df64_to_i32(position_1));
    let bin_2 = u32(floor_df64_to_i32(position_2));
    let bin = bin_0 + FINE_0 * (bin_1 + FINE_1 * bin_2);
    atomicAdd(&bin_counts[bin], 1u);
}}
"#,
    );
    format!("{}\n{position}\n{entry}", wgpu_fft::kernels::DF64_WGSL)
}

fn generate_terminal_wgsl() -> String {
    r#"@group(0) @binding(0) var<storage, read> bin_counts: array<u32>;
@group(0) @binding(1) var<storage, read_write> bin_offsets: array<u32>;

@compute @workgroup_size(1)
fn main() {
    let bin_count = arrayLength(&bin_counts);
    bin_offsets[bin_count] =
        bin_offsets[bin_count - 1u] + bin_counts[bin_count - 1u];
}
"#
    .to_owned()
}

fn generate_scatter_wgsl(fine_shape: [usize; DIMENSIONS]) -> String {
    let position = generate_position_wgsl(fine_shape);
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
    let total = arrayLength(&points) / 3u;
    let wg_flat = (wid.z * nwg.y + wid.y) * nwg.x + wid.x;
    if (wg_flat > (total - 1u) / WORKGROUP_SIZE) {{ return; }}
    let point_index = wg_flat * WORKGROUP_SIZE + lid.x;
    if (point_index >= total) {{ return; }}

    let point_base = point_index * 3u;
    let position_0 = fold_position_0(points[point_base]);
    let position_1 = fold_position_1(points[point_base + 1u]);
    let position_2 = fold_position_2(points[point_base + 2u]);
    let bin_0 = u32(floor_df64_to_i32(position_0));
    let bin_1 = u32(floor_df64_to_i32(position_1));
    let bin_2 = u32(floor_df64_to_i32(position_2));
    let bin = bin_0 + FINE_0 * (bin_1 + FINE_1 * bin_2);
    let slot = bin_offsets[bin] + atomicAdd(&bin_cursors[bin], 1u);
    sorted_indices[slot] = point_index;
}}
"#,
    );
    format!("{}\n{position}\n{entry}", wgpu_fft::kernels::DF64_WGSL)
}

fn generate_scatter_wgsl_for_precision(
    fine_shape: [usize; DIMENSIONS],
    precision: FftPrecision,
) -> String {
    match precision {
        FftPrecision::F64 => generate_scatter_wgsl_f64(fine_shape),
        FftPrecision::Df64 => generate_scatter_wgsl_df64(fine_shape),
        FftPrecision::F32 => generate_scatter_wgsl(fine_shape),
    }
}

fn generate_scatter_wgsl_df64(fine_shape: [usize; DIMENSIONS]) -> String {
    let position = generate_position_wgsl_df64(fine_shape);
    let entry = format!(
        r#"const WORKGROUP_SIZE: u32 = {WORKGROUP_SIZE}u;
@group(0) @binding(0) var<storage, read> points: array<Df64>;
@group(0) @binding(1) var<storage, read> bin_offsets: array<u32>;
@group(0) @binding(2) var<storage, read_write> bin_cursors: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> sorted_indices: array<u32>;

@compute @workgroup_size({WORKGROUP_SIZE})
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {{
    let total = arrayLength(&points) / 3u;
    let wg_flat = (wid.z * nwg.y + wid.y) * nwg.x + wid.x;
    if (wg_flat > (total - 1u) / WORKGROUP_SIZE) {{ return; }}
    let point_index = wg_flat * WORKGROUP_SIZE + lid.x;
    if (point_index >= total) {{ return; }}
    let point_base = point_index * 3u;
    let bin_0 = u32(floor_df64_to_i32(fold_position_0(points[point_base])));
    let bin_1 = u32(floor_df64_to_i32(fold_position_1(points[point_base + 1u])));
    let bin_2 = u32(floor_df64_to_i32(fold_position_2(points[point_base + 2u])));
    let bin = bin_0 + FINE_0 * (bin_1 + FINE_1 * bin_2);
    let slot = bin_offsets[bin] + atomicAdd(&bin_cursors[bin], 1u);
    sorted_indices[slot] = point_index;
}}
"#,
    );
    format!("{}\n{position}\n{entry}", wgpu_fft::kernels::DF64_WGSL)
}

fn generate_scatter_wgsl_f64(fine_shape: [usize; DIMENSIONS]) -> String {
    let position = generate_position_wgsl_f64(fine_shape);
    let entry = format!(
        r#"const WORKGROUP_SIZE: u32 = {WORKGROUP_SIZE}u;
@group(0) @binding(0) var<storage, read> points: array<f64>;
@group(0) @binding(1) var<storage, read> bin_offsets: array<u32>;
@group(0) @binding(2) var<storage, read_write> bin_cursors: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> sorted_indices: array<u32>;

@compute @workgroup_size({WORKGROUP_SIZE})
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {{
    let total = arrayLength(&points) / 3u;
    let wg_flat = (wid.z * nwg.y + wid.y) * nwg.x + wid.x;
    if (wg_flat > (total - 1u) / WORKGROUP_SIZE) {{ return; }}
    let point_index = wg_flat * WORKGROUP_SIZE + lid.x;
    if (point_index >= total) {{ return; }}
    let point_base = point_index * 3u;
    let bin_0 = u32(floor(fold_position_0(points[point_base])));
    let bin_1 = u32(floor(fold_position_1(points[point_base + 1u])));
    let bin_2 = u32(floor(fold_position_2(points[point_base + 2u])));
    let bin = bin_0 + FINE_0 * (bin_1 + FINE_1 * bin_2);
    let slot = bin_offsets[bin] + atomicAdd(&bin_cursors[bin], 1u);
    sorted_indices[slot] = point_index;
}}
"#,
    );
    format!("{position}\n{entry}")
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

fn generate_gather_wgsl(kernel: EsKernel, fine_shape: [usize; DIMENSIONS]) -> String {
    let position = generate_position_wgsl(fine_shape);
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
const VECTOR_BLOCK_SIZE: u32 = {VECTOR_BLOCK_SIZE}u;

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

fn wrap_bin(index: i32, fine_length: i32) -> u32 {{
    var wrapped = index;
    if (wrapped < 0) {{ wrapped = wrapped + fine_length; }}
    if (wrapped >= fine_length) {{ wrapped = wrapped - fine_length; }}
    return u32(wrapped);
}}

fn es_weight(distance: f32) -> f32 {{
    let scaled = 2.0 * abs(distance) / WIDTH_F32;
    let squared = scaled * scaled;
    if (squared >= 1.0) {{ return 0.0; }}
    return exp(BETA * (sqrt(max(0.0, 1.0 - squared)) - 1.0));
}}

fn support_cell(wrapped_cell: u32, start: i32, fine_length: i32) -> i32 {{
    var unwrapped = i32(wrapped_cell);
    if (unwrapped < start) {{ unwrapped = unwrapped + fine_length; }}
    if (unwrapped >= start + WIDTH_I32) {{
        unwrapped = unwrapped - fine_length;
    }}
    return unwrapped;
}}

@compute @workgroup_size({WORKGROUP_SIZE})
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {{
    let wg_flat = (wid.z * nwg.y + wid.y) * nwg.x + wid.x;
    let total_vectors = arrayLength(&fine_grid) / FINE_COUNT;
    let vector_blocks = (total_vectors + VECTOR_BLOCK_SIZE - 1u) / VECTOR_BLOCK_SIZE;
    let total_work = FINE_COUNT * vector_blocks;
    if (wg_flat > (total_work - 1u) / WORKGROUP_SIZE) {{ return; }}
    let work_index = wg_flat * WORKGROUP_SIZE + lid.x;
    if (work_index >= total_work) {{ return; }}
    let vector_block = work_index / FINE_COUNT;
    let cell = work_index - vector_block * FINE_COUNT;
    let first_vector = vector_block * VECTOR_BLOCK_SIZE;

    let cell_0 = cell % FINE_0;
    let cell_1 = (cell / FINE_0) % FINE_1;
    let cell_2 = cell / FINE_01;
    var sums: array<vec2<f32>, {VECTOR_BLOCK_SIZE}>;
    for (var vector_local = 0u; vector_local < VECTOR_BLOCK_SIZE;
         vector_local = vector_local + 1u) {{
        sums[vector_local] = vec2<f32>(0.0, 0.0);
    }}
    for (var bin_offset_2 = -BIN_RADIUS; bin_offset_2 < BIN_RADIUS;
         bin_offset_2 = bin_offset_2 + 1) {{
        let bin_2 = wrap_bin(i32(cell_2) + bin_offset_2, FINE_2_I32);
        for (var bin_offset_1 = -BIN_RADIUS; bin_offset_1 < BIN_RADIUS;
             bin_offset_1 = bin_offset_1 + 1) {{
            let bin_1 = wrap_bin(i32(cell_1) + bin_offset_1, FINE_1_I32);
            for (var bin_offset_0 = -BIN_RADIUS; bin_offset_0 < BIN_RADIUS;
                 bin_offset_0 = bin_offset_0 + 1) {{
                let bin_0 = wrap_bin(i32(cell_0) + bin_offset_0, FINE_0_I32);
                let bin = bin_0 + FINE_0 * (bin_1 + FINE_1 * bin_2);
                let begin = bin_offsets[bin];
                let end = bin_offsets[bin + 1u];
                for (var slot = begin; slot < end; slot = slot + 1u) {{
                    let point_index = sorted_indices[slot];
                    let point_base = point_index * 3u;
                    let position_0 = fold_position_0(points[point_base]);
                    let position_1 = fold_position_1(points[point_base + 1u]);
                    let position_2 = fold_position_2(points[point_base + 2u]);
                    let start_0 = ceil_df64_to_i32(
                        df64_sub(position_0, Df64(HALF_WIDTH, 0.0)));
                    let start_1 = ceil_df64_to_i32(
                        df64_sub(position_1, Df64(HALF_WIDTH, 0.0)));
                    let start_2 = ceil_df64_to_i32(
                        df64_sub(position_2, Df64(HALF_WIDTH, 0.0)));
                    let unwrapped_0 = support_cell(cell_0, start_0, FINE_0_I32);
                    let unwrapped_1 = support_cell(cell_1, start_1, FINE_1_I32);
                    let unwrapped_2 = support_cell(cell_2, start_2, FINE_2_I32);
                    if (unwrapped_0 >= start_0 && unwrapped_0 < start_0 + WIDTH_I32 &&
                        unwrapped_1 >= start_1 && unwrapped_1 < start_1 + WIDTH_I32 &&
                        unwrapped_2 >= start_2 && unwrapped_2 < start_2 + WIDTH_I32) {{
                        let distance_0 = df64_sub(Df64(f32(unwrapped_0), 0.0), position_0);
                        let distance_1 = df64_sub(Df64(f32(unwrapped_1), 0.0), position_1);
                        let distance_2 = df64_sub(Df64(f32(unwrapped_2), 0.0), position_2);
                        let weight_0 = es_weight(distance_0.hi + distance_0.lo);
                        let weight_1 = es_weight(distance_1.hi + distance_1.lo);
                        let weight_2 = es_weight(distance_2.hi + distance_2.lo);
                        let weight = (weight_0 * weight_1) * weight_2;
                        for (var vector_local = 0u; vector_local < VECTOR_BLOCK_SIZE;
                             vector_local = vector_local + 1u) {{
                            let vector_index = first_vector + vector_local;
                            if (vector_index < total_vectors) {{
                                let strength_index = vector_index *
                                    (arrayLength(&points) / 3u) + point_index;
                                sums[vector_local] = sums[vector_local] +
                                    strengths[strength_index] * weight;
                            }}
                        }}
                    }}
                }}
            }}
        }}
    }}
    for (var vector_local = 0u; vector_local < VECTOR_BLOCK_SIZE;
         vector_local = vector_local + 1u) {{
        let vector_index = first_vector + vector_local;
        if (vector_index < total_vectors) {{
            fine_grid[vector_index * FINE_COUNT + cell] = sums[vector_local];
        }}
    }}
}}
"#,
        half_width = format_wgsl_f32(kernel.half_width() as f32),
        beta = format_wgsl_f32(kernel.beta() as f32),
    );
    format!("{}\n{position}\n{entry}", wgpu_fft::kernels::DF64_WGSL)
}

fn generate_gather_wgsl_df64(kernel: EsKernel, fine_shape: [usize; DIMENSIONS]) -> String {
    let position = generate_position_wgsl_df64(fine_shape);
    let horner = generate_horner_wgsl_df64(kernel);
    let width = kernel.width();
    let bin_radius = width.div_ceil(2);
    let entry = format!(
        r#"const WORKGROUP_SIZE: u32 = {WORKGROUP_SIZE}u;
const WIDTH: u32 = {width}u;
const WIDTH_I32: i32 = {width}i;
const BIN_RADIUS: i32 = {bin_radius}i;
const VECTOR_BLOCK_SIZE: u32 = {VECTOR_BLOCK_SIZE}u;

@group(0) @binding(0) var<storage, read> points: array<Df64>;
@group(0) @binding(1) var<storage, read> strengths: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> bin_offsets: array<u32>;
@group(0) @binding(3) var<storage, read> sorted_indices: array<u32>;
@group(0) @binding(4) var<storage, read_write> fine_grid: array<vec4<f32>>;

fn ceil_df64_to_i32(value: Df64) -> i32 {{
    let base = floor(value.hi);
    let remainder = df64_sub(value, Df64(base, 0.0));
    let has_positive_remainder = remainder.hi > 0.0 ||
        (remainder.hi == 0.0 && remainder.lo > 0.0);
    return i32(base) + select(0, 1, has_positive_remainder);
}}

fn wrap_bin(index: i32, fine_length: i32) -> u32 {{
    var wrapped = index;
    if (wrapped < 0) {{ wrapped = wrapped + fine_length; }}
    if (wrapped >= fine_length) {{ wrapped = wrapped - fine_length; }}
    return u32(wrapped);
}}

fn support_cell(wrapped_cell: u32, start: i32, fine_length: i32) -> i32 {{
    var unwrapped = i32(wrapped_cell);
    if (unwrapped < start) {{ unwrapped = unwrapped + fine_length; }}
    if (unwrapped >= start + WIDTH_I32) {{ unwrapped = unwrapped - fine_length; }}
    return unwrapped;
}}

{horner}

@compute @workgroup_size({WORKGROUP_SIZE})
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {{
    let wg_flat = (wid.z * nwg.y + wid.y) * nwg.x + wid.x;
    let total_vectors = arrayLength(&fine_grid) / FINE_COUNT;
    let vector_blocks = (total_vectors + VECTOR_BLOCK_SIZE - 1u) / VECTOR_BLOCK_SIZE;
    let total_work = FINE_COUNT * vector_blocks;
    if (wg_flat > (total_work - 1u) / WORKGROUP_SIZE) {{ return; }}
    let work_index = wg_flat * WORKGROUP_SIZE + lid.x;
    if (work_index >= total_work) {{ return; }}
    let vector_block = work_index / FINE_COUNT;
    let cell = work_index - vector_block * FINE_COUNT;
    let first_vector = vector_block * VECTOR_BLOCK_SIZE;
    let cell_0 = cell % FINE_0;
    let cell_1 = (cell / FINE_0) % FINE_1;
    let cell_2 = cell / FINE_01;
    var sums: array<vec4<f32>, {VECTOR_BLOCK_SIZE}>;
    for (var vector_local = 0u; vector_local < VECTOR_BLOCK_SIZE;
         vector_local = vector_local + 1u) {{
        sums[vector_local] = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }}
    for (var bin_offset_2 = -BIN_RADIUS; bin_offset_2 < BIN_RADIUS;
         bin_offset_2 = bin_offset_2 + 1) {{
        let bin_2 = wrap_bin(i32(cell_2) + bin_offset_2, FINE_2_I32);
        for (var bin_offset_1 = -BIN_RADIUS; bin_offset_1 < BIN_RADIUS;
             bin_offset_1 = bin_offset_1 + 1) {{
            let bin_1 = wrap_bin(i32(cell_1) + bin_offset_1, FINE_1_I32);
            for (var bin_offset_0 = -BIN_RADIUS; bin_offset_0 < BIN_RADIUS;
                 bin_offset_0 = bin_offset_0 + 1) {{
                let bin_0 = wrap_bin(i32(cell_0) + bin_offset_0, FINE_0_I32);
                let bin = bin_0 + FINE_0 * (bin_1 + FINE_1 * bin_2);
                for (var slot = bin_offsets[bin]; slot < bin_offsets[bin + 1u];
                     slot = slot + 1u) {{
                    let point_index = sorted_indices[slot];
                    let point_base = point_index * 3u;
                    let position_0 = fold_position_0(points[point_base]);
                    let position_1 = fold_position_1(points[point_base + 1u]);
                    let position_2 = fold_position_2(points[point_base + 2u]);
                    let start_0 = ceil_df64_to_i32(df64_sub(position_0, DF64_HALF_WIDTH));
                    let start_1 = ceil_df64_to_i32(df64_sub(position_1, DF64_HALF_WIDTH));
                    let start_2 = ceil_df64_to_i32(df64_sub(position_2, DF64_HALF_WIDTH));
                    let unwrapped_0 = support_cell(cell_0, start_0, FINE_0_I32);
                    let unwrapped_1 = support_cell(cell_1, start_1, FINE_1_I32);
                    let unwrapped_2 = support_cell(cell_2, start_2, FINE_2_I32);
                    if (unwrapped_0 >= start_0 && unwrapped_0 < start_0 + WIDTH_I32 &&
                        unwrapped_1 >= start_1 && unwrapped_1 < start_1 + WIDTH_I32 &&
                        unwrapped_2 >= start_2 && unwrapped_2 < start_2 + WIDTH_I32) {{
                        let distance_0 = df64_sub(Df64(f32(unwrapped_0), 0.0), position_0);
                        let distance_1 = df64_sub(Df64(f32(unwrapped_1), 0.0), position_1);
                        let distance_2 = df64_sub(Df64(f32(unwrapped_2), 0.0), position_2);
                        let weight = df64_mul(
                            df64_mul(es_weight(distance_0), es_weight(distance_1)),
                            es_weight(distance_2),
                        );
                        for (var vector_local = 0u; vector_local < VECTOR_BLOCK_SIZE;
                             vector_local = vector_local + 1u) {{
                            let vector_index = first_vector + vector_local;
                            if (vector_index < total_vectors) {{
                                let strength_index = vector_index *
                                    (arrayLength(&points) / 3u) + point_index;
                                sums[vector_local] = df64_complex_add(
                                    sums[vector_local],
                                    df64_complex_scale(strengths[strength_index], weight),
                                );
                            }}
                        }}
                    }}
                }}
            }}
        }}
    }}
    for (var vector_local = 0u; vector_local < VECTOR_BLOCK_SIZE;
         vector_local = vector_local + 1u) {{
        let vector_index = first_vector + vector_local;
        if (vector_index < total_vectors) {{
            fine_grid[vector_index * FINE_COUNT + cell] = sums[vector_local];
        }}
    }}
}}
"#,
    );
    format!("{}\n{position}\n{entry}", wgpu_fft::kernels::DF64_WGSL)
}

fn generate_gather_wgsl_f64(kernel: EsKernel, fine_shape: [usize; DIMENSIONS]) -> String {
    let position = generate_position_wgsl_f64(fine_shape);
    let horner = generate_horner_wgsl_f64(kernel);
    let width = kernel.width();
    let bin_radius = width.div_ceil(2);
    let entry = format!(
        r#"const WORKGROUP_SIZE: u32 = {WORKGROUP_SIZE}u;
const WIDTH: u32 = {width}u;
const WIDTH_I32: i32 = {width}i;
const HALF_WIDTH: f64 = {half_width};
const BIN_RADIUS: i32 = {bin_radius}i;
const VECTOR_BLOCK_SIZE: u32 = {VECTOR_BLOCK_SIZE}u;

@group(0) @binding(0) var<storage, read> points: array<f64>;
@group(0) @binding(1) var<storage, read> strengths: array<vec2<f64>>;
@group(0) @binding(2) var<storage, read> bin_offsets: array<u32>;
@group(0) @binding(3) var<storage, read> sorted_indices: array<u32>;
@group(0) @binding(4) var<storage, read_write> fine_grid: array<vec2<f64>>;

fn wrap_bin(index: i32, fine_length: i32) -> u32 {{
    var wrapped = index;
    if (wrapped < 0) {{ wrapped = wrapped + fine_length; }}
    if (wrapped >= fine_length) {{ wrapped = wrapped - fine_length; }}
    return u32(wrapped);
}}

fn support_cell(wrapped_cell: u32, start: i32, fine_length: i32) -> i32 {{
    var unwrapped = i32(wrapped_cell);
    if (unwrapped < start) {{ unwrapped = unwrapped + fine_length; }}
    if (unwrapped >= start + WIDTH_I32) {{ unwrapped = unwrapped - fine_length; }}
    return unwrapped;
}}

{horner}

@compute @workgroup_size({WORKGROUP_SIZE})
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {{
    let wg_flat = (wid.z * nwg.y + wid.y) * nwg.x + wid.x;
    let total_vectors = arrayLength(&fine_grid) / FINE_COUNT;
    let vector_blocks = (total_vectors + VECTOR_BLOCK_SIZE - 1u) / VECTOR_BLOCK_SIZE;
    let total_work = FINE_COUNT * vector_blocks;
    if (wg_flat > (total_work - 1u) / WORKGROUP_SIZE) {{ return; }}
    let work_index = wg_flat * WORKGROUP_SIZE + lid.x;
    if (work_index >= total_work) {{ return; }}
    let vector_block = work_index / FINE_COUNT;
    let cell = work_index - vector_block * FINE_COUNT;
    let first_vector = vector_block * VECTOR_BLOCK_SIZE;
    let cell_0 = cell % FINE_0;
    let cell_1 = (cell / FINE_0) % FINE_1;
    let cell_2 = cell / FINE_01;
    var sums: array<vec2<f64>, {VECTOR_BLOCK_SIZE}>;
    for (var vector_local = 0u; vector_local < VECTOR_BLOCK_SIZE;
         vector_local = vector_local + 1u) {{
        sums[vector_local] = vec2<f64>(0.0lf, 0.0lf);
    }}
    for (var bin_offset_2 = -BIN_RADIUS; bin_offset_2 < BIN_RADIUS;
         bin_offset_2 = bin_offset_2 + 1) {{
        let bin_2 = wrap_bin(i32(cell_2) + bin_offset_2, FINE_2_I32);
        for (var bin_offset_1 = -BIN_RADIUS; bin_offset_1 < BIN_RADIUS;
             bin_offset_1 = bin_offset_1 + 1) {{
            let bin_1 = wrap_bin(i32(cell_1) + bin_offset_1, FINE_1_I32);
            for (var bin_offset_0 = -BIN_RADIUS; bin_offset_0 < BIN_RADIUS;
                 bin_offset_0 = bin_offset_0 + 1) {{
                let bin_0 = wrap_bin(i32(cell_0) + bin_offset_0, FINE_0_I32);
                let bin = bin_0 + FINE_0 * (bin_1 + FINE_1 * bin_2);
                for (var slot = bin_offsets[bin]; slot < bin_offsets[bin + 1u];
                     slot = slot + 1u) {{
                    let point_index = sorted_indices[slot];
                    let point_base = point_index * 3u;
                    let position_0 = fold_position_0(points[point_base]);
                    let position_1 = fold_position_1(points[point_base + 1u]);
                    let position_2 = fold_position_2(points[point_base + 2u]);
                    let start_0 = i32(ceil(position_0 - HALF_WIDTH));
                    let start_1 = i32(ceil(position_1 - HALF_WIDTH));
                    let start_2 = i32(ceil(position_2 - HALF_WIDTH));
                    let unwrapped_0 = support_cell(cell_0, start_0, FINE_0_I32);
                    let unwrapped_1 = support_cell(cell_1, start_1, FINE_1_I32);
                    let unwrapped_2 = support_cell(cell_2, start_2, FINE_2_I32);
                    if (unwrapped_0 >= start_0 && unwrapped_0 < start_0 + WIDTH_I32 &&
                        unwrapped_1 >= start_1 && unwrapped_1 < start_1 + WIDTH_I32 &&
                        unwrapped_2 >= start_2 && unwrapped_2 < start_2 + WIDTH_I32) {{
                        let weight_0 = es_weight(f64(unwrapped_0) - position_0);
                        let weight_1 = es_weight(f64(unwrapped_1) - position_1);
                        let weight_2 = es_weight(f64(unwrapped_2) - position_2);
                        let weight = (weight_0 * weight_1) * weight_2;
                        for (var vector_local = 0u; vector_local < VECTOR_BLOCK_SIZE;
                             vector_local = vector_local + 1u) {{
                            let vector_index = first_vector + vector_local;
                            if (vector_index < total_vectors) {{
                                let strength_index = vector_index *
                                    (arrayLength(&points) / 3u) + point_index;
                                sums[vector_local] = sums[vector_local] +
                                    strengths[strength_index] * weight;
                            }}
                        }}
                    }}
                }}
            }}
        }}
    }}
    for (var vector_local = 0u; vector_local < VECTOR_BLOCK_SIZE;
         vector_local = vector_local + 1u) {{
        let vector_index = first_vector + vector_local;
        if (vector_index < total_vectors) {{
            fine_grid[vector_index * FINE_COUNT + cell] = sums[vector_local];
        }}
    }}
}}
"#,
        half_width = format_wgsl_f64(kernel.half_width()),
    );
    format!("{position}\n{entry}")
}

fn generate_tiled_gather_wgsl(
    kernel: EsKernel,
    fine_shape: [usize; DIMENSIONS],
    cache_capacity: usize,
) -> Result<String> {
    let position = generate_position_wgsl(fine_shape);
    let width = kernel.width();
    let bin_radius = width.div_ceil(2);
    let halo_shape = tiled_gather_halo_shape(width)?;
    let halo_bin_count = halo_shape
        .iter()
        .try_fold(1usize, |count, &length| count.checked_mul(length))
        .ok_or(NufftError::LengthOverflow {
            context: "type-1 3D tiled gather halo-bin count",
        })?;
    let cache_weight_count =
        cache_capacity
            .checked_mul(width)
            .ok_or(NufftError::LengthOverflow {
                context: "type-1 3D tiled gather cached-weight count",
            })?;
    let cache_load_rounds = cache_capacity.div_ceil(TILED_GATHER_WORKGROUP_SIZE as usize);
    let (tile_counts, tile_count) = tiled_gather_tile_counts(fine_shape)?;
    debug_assert!(halo_bin_count < TILED_GATHER_SCAN_ELEMENTS);
    debug_assert!(cache_capacity > 0);
    let entry = format!(
        r#"const WORKGROUP_SIZE: u32 = {workgroup_size}u;
const TILE_0: u32 = {tile_0}u;
const TILE_1: u32 = {tile_1}u;
const TILE_2: u32 = {tile_2}u;
const TILE_COUNT_0: u32 = {tile_count_0}u;
const TILE_COUNT_1: u32 = {tile_count_1}u;
const TILE_COUNT: u32 = {tile_count}u;
const WIDTH: u32 = {width}u;
const WIDTH_I32: i32 = {width}i;
const WIDTH_F32: f32 = {width}.0;
const HALF_WIDTH: f32 = {half_width};
const BETA: f32 = {beta};
const BIN_RADIUS: i32 = {bin_radius}i;
const HALO_0: u32 = {halo_0}u;
const HALO_1: u32 = {halo_1}u;
const HALO_2: u32 = {halo_2}u;
const HALO_BIN_COUNT: u32 = {halo_bin_count}u;
const SCAN_ELEMENTS: u32 = {scan_elements}u;
const SCAN_ITEMS_PER_INVOCATION: u32 = {scan_items_per_invocation}u;
const CACHE_CAPACITY: u32 = {cache_capacity}u;
const CACHE_LOAD_ROUNDS: u32 = {cache_load_rounds}u;
const VECTOR_BLOCK_SIZE: u32 = {VECTOR_BLOCK_SIZE}u;

@group(0) @binding(0) var<storage, read> points: array<f32>;
@group(0) @binding(1) var<storage, read> strengths: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> bin_offsets: array<u32>;
@group(0) @binding(3) var<storage, read> sorted_indices: array<u32>;
@group(0) @binding(4) var<storage, read_write> fine_grid: array<vec2<f32>>;

var<workgroup> halo_prefix: array<u32, {padded_scan_elements}>;
var<workgroup> cached_starts_0: array<i32, {cache_capacity}>;
var<workgroup> cached_starts_1: array<i32, {cache_capacity}>;
var<workgroup> cached_starts_2: array<i32, {cache_capacity}>;
var<workgroup> cached_point_indices: array<u32, {cache_capacity}>;
var<workgroup> cached_weights_0: array<f32, {cache_weight_count}>;
var<workgroup> cached_weights_1: array<f32, {cache_weight_count}>;
var<workgroup> cached_weights_2: array<f32, {cache_weight_count}>;

fn physical_index(logical: u32) -> u32 {{
    return logical + (logical >> 5u);
}}

fn ceil_df64_to_i32(value: Df64) -> i32 {{
    let base = floor(value.hi);
    let remainder = df64_sub(value, Df64(base, 0.0));
    let has_positive_remainder = remainder.hi > 0.0 ||
        (remainder.hi == 0.0 && remainder.lo > 0.0);
    return i32(base) + select(0, 1, has_positive_remainder);
}}

fn wrap_bin(index: i32, fine_length: i32) -> u32 {{
    var wrapped = index;
    if (wrapped < 0) {{ wrapped = wrapped + fine_length; }}
    if (wrapped >= fine_length) {{ wrapped = wrapped - fine_length; }}
    return u32(wrapped);
}}

fn halo_global_bin(halo_bin: u32, tile_origin: vec3<u32>) -> u32 {{
    let halo_0 = halo_bin % HALO_0;
    let remaining = halo_bin / HALO_0;
    let halo_1 = remaining % HALO_1;
    let halo_2 = remaining / HALO_1;
    let bin_0 = wrap_bin(i32(tile_origin.x) + i32(halo_0) - BIN_RADIUS, FINE_0_I32);
    let bin_1 = wrap_bin(i32(tile_origin.y) + i32(halo_1) - BIN_RADIUS, FINE_1_I32);
    let bin_2 = wrap_bin(i32(tile_origin.z) + i32(halo_2) - BIN_RADIUS, FINE_2_I32);
    return bin_0 + FINE_0 * (bin_1 + FINE_1 * bin_2);
}}

fn halo_bin_for_record(record: u32) -> u32 {{
    var low = 0u;
    var high = HALO_BIN_COUNT;
    loop {{
        if (low >= high) {{ break; }}
        let middle = (low + high) / 2u;
        let middle_end = halo_prefix[physical_index(middle + 1u)];
        if (middle_end <= record) {{
            low = middle + 1u;
        }} else {{
            high = middle;
        }}
    }}
    return low;
}}

fn es_weight(distance: f32) -> f32 {{
    let scaled = 2.0 * abs(distance) / WIDTH_F32;
    let squared = scaled * scaled;
    if (squared >= 1.0) {{ return 0.0; }}
    return exp(BETA * (sqrt(max(0.0, 1.0 - squared)) - 1.0));
}}

fn support_cell(wrapped_cell: u32, start: i32, fine_length: i32) -> i32 {{
    var unwrapped = i32(wrapped_cell);
    if (unwrapped < start) {{ unwrapped = unwrapped + fine_length; }}
    if (unwrapped >= start + WIDTH_I32) {{
        unwrapped = unwrapped - fine_length;
    }}
    return unwrapped;
}}

@compute @workgroup_size({workgroup_size})
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {{
    let wg_flat = (wid.z * nwg.y + wid.y) * nwg.x + wid.x;
    let total_vectors = arrayLength(&fine_grid) / FINE_COUNT;
    let vector_blocks = (total_vectors + VECTOR_BLOCK_SIZE - 1u) / VECTOR_BLOCK_SIZE;
    if (wg_flat >= TILE_COUNT * vector_blocks) {{ return; }}

    let vector_block = wg_flat / TILE_COUNT;
    let first_vector = vector_block * VECTOR_BLOCK_SIZE;
    let tile_flat = wg_flat - vector_block * TILE_COUNT;
    let tile_0 = tile_flat % TILE_COUNT_0;
    let remaining_tile = tile_flat / TILE_COUNT_0;
    let tile_1 = remaining_tile % TILE_COUNT_1;
    let tile_2 = remaining_tile / TILE_COUNT_1;
    let tile_origin = vec3<u32>(tile_0 * TILE_0, tile_1 * TILE_1, tile_2 * TILE_2);

    let local_0 = lid.x % TILE_0;
    let remaining_local = lid.x / TILE_0;
    let local_1 = remaining_local % TILE_1;
    let local_2 = remaining_local / TILE_1;
    let local_cell = vec3<u32>(local_0, local_1, local_2);
    let cell = tile_origin + local_cell;
    let cell_in_bounds = cell.x < FINE_0 && cell.y < FINE_1 && cell.z < FINE_2;

    for (var item = 0u; item < SCAN_ITEMS_PER_INVOCATION; item = item + 1u) {{
        let logical = item * WORKGROUP_SIZE + lid.x;
        var count = 0u;
        if (logical < HALO_BIN_COUNT) {{
            let global_bin = halo_global_bin(logical, tile_origin);
            count = bin_offsets[global_bin + 1u] - bin_offsets[global_bin];
        }}
        halo_prefix[physical_index(logical)] = count;
    }}
    workgroupBarrier();

    var stride = 1u;
    loop {{
        let node_count = SCAN_ELEMENTS / (2u * stride);
        for (var node = lid.x; node < node_count; node = node + WORKGROUP_SIZE) {{
            let left = stride * (2u * node + 1u) - 1u;
            let right = stride * (2u * node + 2u) - 1u;
            halo_prefix[physical_index(right)] =
                halo_prefix[physical_index(right)] + halo_prefix[physical_index(left)];
        }}
        workgroupBarrier();
        if (stride == SCAN_ELEMENTS / 2u) {{ break; }}
        stride = stride * 2u;
    }}

    if (lid.x == 0u) {{
        halo_prefix[physical_index(SCAN_ELEMENTS - 1u)] = 0u;
    }}
    workgroupBarrier();

    stride = SCAN_ELEMENTS / 2u;
    loop {{
        let node_count = SCAN_ELEMENTS / (2u * stride);
        for (var node = lid.x; node < node_count; node = node + WORKGROUP_SIZE) {{
            let left = stride * (2u * node + 1u) - 1u;
            let right = stride * (2u * node + 2u) - 1u;
            let temporary = halo_prefix[physical_index(left)];
            halo_prefix[physical_index(left)] = halo_prefix[physical_index(right)];
            halo_prefix[physical_index(right)] =
                halo_prefix[physical_index(right)] + temporary;
        }}
        workgroupBarrier();
        if (stride == 1u) {{ break; }}
        stride = stride / 2u;
    }}

    let halo_total = workgroupUniformLoad(
        &halo_prefix[physical_index(HALO_BIN_COUNT)],
    );
    var sums: array<vec2<f32>, {VECTOR_BLOCK_SIZE}>;
    for (var vector_local = 0u; vector_local < VECTOR_BLOCK_SIZE;
         vector_local = vector_local + 1u) {{
        sums[vector_local] = vec2<f32>(0.0, 0.0);
    }}
    var batch_start = 0u;
    loop {{
        if (batch_start >= halo_total) {{ break; }}
        let batch_length = min(CACHE_CAPACITY, halo_total - batch_start);
        let batch_end = batch_start + batch_length;

        for (var load_round = 0u; load_round < CACHE_LOAD_ROUNDS;
             load_round = load_round + 1u) {{
            let cache_slot = load_round * WORKGROUP_SIZE + lid.x;
            if (cache_slot < batch_length) {{
                let record = batch_start + cache_slot;
                let halo_bin = halo_bin_for_record(record);
                let global_bin = halo_global_bin(halo_bin, tile_origin);
                let in_bin = record - halo_prefix[physical_index(halo_bin)];
                let point_slot = bin_offsets[global_bin] + in_bin;
                let point_index = sorted_indices[point_slot];
                let point_base = point_index * 3u;
                let position_0 = fold_position_0(points[point_base]);
                let position_1 = fold_position_1(points[point_base + 1u]);
                let position_2 = fold_position_2(points[point_base + 2u]);
                let start_0 = ceil_df64_to_i32(
                    df64_sub(position_0, Df64(HALF_WIDTH, 0.0)));
                let start_1 = ceil_df64_to_i32(
                    df64_sub(position_1, Df64(HALF_WIDTH, 0.0)));
                let start_2 = ceil_df64_to_i32(
                    df64_sub(position_2, Df64(HALF_WIDTH, 0.0)));
                cached_starts_0[cache_slot] = start_0;
                cached_starts_1[cache_slot] = start_1;
                cached_starts_2[cache_slot] = start_2;
                cached_point_indices[cache_slot] = point_index;
                for (var support = 0u; support < WIDTH; support = support + 1u) {{
                    let support_0 = start_0 + i32(support);
                    let support_1 = start_1 + i32(support);
                    let support_2 = start_2 + i32(support);
                    let distance_0 = df64_sub(Df64(f32(support_0), 0.0), position_0);
                    let distance_1 = df64_sub(Df64(f32(support_1), 0.0), position_1);
                    let distance_2 = df64_sub(Df64(f32(support_2), 0.0), position_2);
                    let weight_slot = cache_slot * WIDTH + support;
                    cached_weights_0[weight_slot] =
                        es_weight(distance_0.hi + distance_0.lo);
                    cached_weights_1[weight_slot] =
                        es_weight(distance_1.hi + distance_1.lo);
                    cached_weights_2[weight_slot] =
                        es_weight(distance_2.hi + distance_2.lo);
                }}
            }}
        }}
        workgroupBarrier();

        if (cell_in_bounds) {{
            for (var bin_offset_2 = -BIN_RADIUS; bin_offset_2 < BIN_RADIUS;
                 bin_offset_2 = bin_offset_2 + 1) {{
                let halo_2 = u32(i32(local_cell.z) + bin_offset_2 + BIN_RADIUS);
                for (var bin_offset_1 = -BIN_RADIUS; bin_offset_1 < BIN_RADIUS;
                     bin_offset_1 = bin_offset_1 + 1) {{
                    let halo_1 = u32(i32(local_cell.y) + bin_offset_1 + BIN_RADIUS);
                    for (var bin_offset_0 = -BIN_RADIUS; bin_offset_0 < BIN_RADIUS;
                         bin_offset_0 = bin_offset_0 + 1) {{
                        let halo_0 = u32(i32(local_cell.x) + bin_offset_0 + BIN_RADIUS);
                        let halo_bin = halo_0 + HALO_0 * (halo_1 + HALO_1 * halo_2);
                        let begin = max(
                            halo_prefix[physical_index(halo_bin)],
                            batch_start,
                        );
                        let end = min(
                            halo_prefix[physical_index(halo_bin + 1u)],
                            batch_end,
                        );
                        for (var record = begin; record < end; record = record + 1u) {{
                            let cache_slot = record - batch_start;
                            let start_0 = cached_starts_0[cache_slot];
                            let start_1 = cached_starts_1[cache_slot];
                            let start_2 = cached_starts_2[cache_slot];
                            let unwrapped_0 = support_cell(cell.x, start_0, FINE_0_I32);
                            let unwrapped_1 = support_cell(cell.y, start_1, FINE_1_I32);
                            let unwrapped_2 = support_cell(cell.z, start_2, FINE_2_I32);
                            if (unwrapped_0 >= start_0 && unwrapped_0 < start_0 + WIDTH_I32 &&
                                unwrapped_1 >= start_1 && unwrapped_1 < start_1 + WIDTH_I32 &&
                                unwrapped_2 >= start_2 && unwrapped_2 < start_2 + WIDTH_I32) {{
                                let support_0 = u32(unwrapped_0 - start_0);
                                let support_1 = u32(unwrapped_1 - start_1);
                                let support_2 = u32(unwrapped_2 - start_2);
                                let weight_0 = cached_weights_0[cache_slot * WIDTH + support_0];
                                let weight_1 = cached_weights_1[cache_slot * WIDTH + support_1];
                                let weight_2 = cached_weights_2[cache_slot * WIDTH + support_2];
                                let weight = (weight_0 * weight_1) * weight_2;
                                let point_index = cached_point_indices[cache_slot];
                                for (var vector_local = 0u;
                                     vector_local < VECTOR_BLOCK_SIZE;
                                     vector_local = vector_local + 1u) {{
                                    let vector_index = first_vector + vector_local;
                                    if (vector_index < total_vectors) {{
                                        let strength_index = vector_index *
                                            (arrayLength(&points) / 3u) + point_index;
                                        sums[vector_local] = sums[vector_local] +
                                            strengths[strength_index] * weight;
                                    }}
                                }}
                            }}
                        }}
                    }}
                }}
            }}
        }}
        workgroupBarrier();
        batch_start = batch_end;
    }}

    if (cell_in_bounds) {{
        let cell_index = cell.x + FINE_0 * (cell.y + FINE_1 * cell.z);
        for (var vector_local = 0u; vector_local < VECTOR_BLOCK_SIZE;
             vector_local = vector_local + 1u) {{
            let vector_index = first_vector + vector_local;
            if (vector_index < total_vectors) {{
                fine_grid[vector_index * FINE_COUNT + cell_index] = sums[vector_local];
            }}
        }}
    }}
}}
"#,
        workgroup_size = TILED_GATHER_WORKGROUP_SIZE,
        tile_0 = TILED_GATHER_TILE_SHAPE[0],
        tile_1 = TILED_GATHER_TILE_SHAPE[1],
        tile_2 = TILED_GATHER_TILE_SHAPE[2],
        tile_count_0 = tile_counts[0],
        tile_count_1 = tile_counts[1],
        tile_count = tile_count,
        width = width,
        half_width = format_wgsl_f32(kernel.half_width() as f32),
        beta = format_wgsl_f32(kernel.beta() as f32),
        bin_radius = bin_radius,
        halo_0 = halo_shape[0],
        halo_1 = halo_shape[1],
        halo_2 = halo_shape[2],
        halo_bin_count = halo_bin_count,
        scan_elements = TILED_GATHER_SCAN_ELEMENTS,
        scan_items_per_invocation =
            TILED_GATHER_SCAN_ELEMENTS / TILED_GATHER_WORKGROUP_SIZE as usize,
        cache_capacity = cache_capacity,
        cache_load_rounds = cache_load_rounds,
        padded_scan_elements = TILED_GATHER_PADDED_SCAN_ELEMENTS,
        cache_weight_count = cache_weight_count,
    );
    Ok(format!(
        "{}\n{position}\n{entry}",
        wgpu_fft::kernels::DF64_WGSL
    ))
}

fn generate_deconvolution_wgsl(config: &NufftConfig, fine_shape: [usize; DIMENSIONS]) -> String {
    let mode_shape = config.n_modes();
    debug_assert_eq!(mode_shape.len(), DIMENSIONS);
    let mode_count = mode_shape[0] * mode_shape[1] * mode_shape[2];
    let centered = matches!(config.mode_order(), crate::config::ModeOrder::Centered);
    format!(
        r#"const WORKGROUP_SIZE: u32 = {WORKGROUP_SIZE}u;
const MODE_0: u32 = {mode_0}u;
const MODE_1: u32 = {mode_1}u;
const MODE_2: u32 = {mode_2}u;
const MODE_01: u32 = {mode_01}u;
const MODE_COUNT: u32 = {mode_count}u;
const FINE_0: u32 = {fine_0}u;
const FINE_1: u32 = {fine_1}u;
const FINE_2: u32 = {fine_2}u;
const FINE_COUNT: u32 = {fine_count}u;
const CENTERED_ORDER: bool = {centered};

@group(0) @binding(0) var<storage, read> fine_grid: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read> amplitudes: array<f32>;
@group(0) @binding(2) var<storage, read_write> output_values: array<vec2<f32>>;

fn fine_axis_index(storage_index: u32, mode_length: u32, fine_length: u32) -> u32 {{
    let half_mode_count = mode_length / 2u;
    let nonnegative_count = (mode_length + 1u) / 2u;
    if (CENTERED_ORDER) {{
        if (storage_index < half_mode_count) {{
            return fine_length - (half_mode_count - storage_index);
        }}
        return storage_index - half_mode_count;
    }}
    if (storage_index < nonnegative_count) {{
        return storage_index;
    }}
    return fine_length - (mode_length - storage_index);
}}

@compute @workgroup_size({WORKGROUP_SIZE})
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {{
    let wg_flat = (wid.z * nwg.y + wid.y) * nwg.x + wid.x;
    let total = arrayLength(&output_values);
    if (wg_flat > (total - 1u) / WORKGROUP_SIZE) {{ return; }}
    let linear_index = wg_flat * WORKGROUP_SIZE + lid.x;
    if (linear_index >= total) {{ return; }}
    let vector_index = linear_index / MODE_COUNT;
    let output_index = linear_index - vector_index * MODE_COUNT;
    let storage_0 = output_index % MODE_0;
    let storage_1 = (output_index / MODE_0) % MODE_1;
    let storage_2 = output_index / MODE_01;
    let fine_0 = fine_axis_index(storage_0, MODE_0, FINE_0);
    let fine_1 = fine_axis_index(storage_1, MODE_1, FINE_1);
    let fine_2 = fine_axis_index(storage_2, MODE_2, FINE_2);
    let fine_index = fine_0 + FINE_0 * (fine_1 + FINE_1 * fine_2);
    output_values[linear_index] = fine_grid[vector_index * FINE_COUNT + fine_index] *
        amplitudes[output_index];
}}
"#,
        mode_0 = mode_shape[0],
        mode_1 = mode_shape[1],
        mode_2 = mode_shape[2],
        mode_01 = mode_shape[0] * mode_shape[1],
        fine_0 = fine_shape[0],
        fine_1 = fine_shape[1],
        fine_2 = fine_shape[2],
        fine_count = fine_shape[0] * fine_shape[1] * fine_shape[2],
    )
}

fn generate_deconvolution_wgsl_for_precision(
    config: &NufftConfig,
    fine_shape: [usize; DIMENSIONS],
    precision: FftPrecision,
) -> String {
    match precision {
        FftPrecision::F64 => generate_deconvolution_wgsl(config, fine_shape)
            .replace("vec2<f32>", "vec2<f64>")
            .replace("array<f32>", "array<f64>"),
        FftPrecision::Df64 => generate_deconvolution_wgsl_df64(config, fine_shape),
        FftPrecision::F32 => generate_deconvolution_wgsl(config, fine_shape),
    }
}

fn generate_deconvolution_wgsl_df64(
    config: &NufftConfig,
    fine_shape: [usize; DIMENSIONS],
) -> String {
    let mode_shape = config.n_modes();
    let mode_count = mode_shape[0] * mode_shape[1] * mode_shape[2];
    let centered = matches!(config.mode_order(), crate::config::ModeOrder::Centered);
    let entry = format!(
        r#"const WORKGROUP_SIZE: u32 = {WORKGROUP_SIZE}u;
const MODE_0: u32 = {mode_0}u;
const MODE_1: u32 = {mode_1}u;
const MODE_2: u32 = {mode_2}u;
const MODE_01: u32 = {mode_01}u;
const MODE_COUNT: u32 = {mode_count}u;
const FINE_0: u32 = {fine_0}u;
const FINE_1: u32 = {fine_1}u;
const FINE_2: u32 = {fine_2}u;
const FINE_COUNT: u32 = {fine_count}u;
const CENTERED_ORDER: bool = {centered};

@group(0) @binding(0) var<storage, read> fine_grid: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> amplitudes: array<Df64>;
@group(0) @binding(2) var<storage, read_write> output_values: array<vec4<f32>>;

fn fine_axis_index(storage_index: u32, mode_length: u32, fine_length: u32) -> u32 {{
    let half_mode_count = mode_length / 2u;
    let nonnegative_count = (mode_length + 1u) / 2u;
    if (CENTERED_ORDER) {{
        if (storage_index < half_mode_count) {{
            return fine_length - (half_mode_count - storage_index);
        }}
        return storage_index - half_mode_count;
    }}
    if (storage_index < nonnegative_count) {{ return storage_index; }}
    return fine_length - (mode_length - storage_index);
}}

@compute @workgroup_size({WORKGROUP_SIZE})
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {{
    let wg_flat = (wid.z * nwg.y + wid.y) * nwg.x + wid.x;
    let total = arrayLength(&output_values);
    if (wg_flat > (total - 1u) / WORKGROUP_SIZE) {{ return; }}
    let linear_index = wg_flat * WORKGROUP_SIZE + lid.x;
    if (linear_index >= total) {{ return; }}
    let vector_index = linear_index / MODE_COUNT;
    let output_index = linear_index - vector_index * MODE_COUNT;
    let storage_0 = output_index % MODE_0;
    let storage_1 = (output_index / MODE_0) % MODE_1;
    let storage_2 = output_index / MODE_01;
    let fine_0 = fine_axis_index(storage_0, MODE_0, FINE_0);
    let fine_1 = fine_axis_index(storage_1, MODE_1, FINE_1);
    let fine_2 = fine_axis_index(storage_2, MODE_2, FINE_2);
    let fine_index = fine_0 + FINE_0 * (fine_1 + FINE_1 * fine_2);
    output_values[linear_index] = df64_complex_scale(
        fine_grid[vector_index * FINE_COUNT + fine_index],
        amplitudes[output_index],
    );
}}
"#,
        mode_0 = mode_shape[0],
        mode_1 = mode_shape[1],
        mode_2 = mode_shape[2],
        mode_01 = mode_shape[0] * mode_shape[1],
        fine_0 = fine_shape[0],
        fine_1 = fine_shape[1],
        fine_2 = fine_shape[2],
        fine_count = fine_shape[0] * fine_shape[1] * fine_shape[2],
    );
    format!("{}\n{entry}", wgpu_fft::kernels::DF64_WGSL)
}

fn generate_horner_wgsl_f64(kernel: EsKernel) -> String {
    let table = kernel.horner_table();
    let coefficients = table
        .coefficients()
        .iter()
        .map(|&value| format_wgsl_f64(value))
        .collect::<Vec<_>>()
        .join(", ");
    let coefficient_total = table.coefficients().len();
    let coefficient_count = table.coefficient_count();
    format!(
        r#"const HORNER_COEFFICIENT_COUNT: u32 = {coefficient_count}u;
const HORNER_COEFFICIENTS: array<f64, {coefficient_total}> =
    array<f64, {coefficient_total}>({coefficients});

fn es_weight(distance: f64) -> f64 {{
    if (abs(distance) >= HALF_WIDTH) {{ return 0.0lf; }}
    let panel = u32(clamp(i32(ceil(distance + HALF_WIDTH)) - 1, 0, WIDTH_I32 - 1));
    let local = 2.0lf * (distance - f64(panel)) + f64(WIDTH_I32 - 1);
    var value = 0.0lf;
    for (var coefficient = 0u; coefficient < HORNER_COEFFICIENT_COUNT;
         coefficient = coefficient + 1u) {{
        value = value * local + HORNER_COEFFICIENTS[coefficient * WIDTH + panel];
    }}
    return value;
}}"#,
    )
}

fn generate_horner_wgsl_df64(kernel: EsKernel) -> String {
    let table = kernel.horner_table();
    let coefficients = table
        .coefficients()
        .iter()
        .map(|&value| {
            let value = DoubleFloat::from_f64(value);
            format!(
                "Df64({}, {})",
                format_wgsl_f32(value.hi),
                format_wgsl_f32(value.lo)
            )
        })
        .collect::<Vec<_>>()
        .join(", ");
    let coefficient_total = table.coefficients().len();
    let coefficient_count = table.coefficient_count();
    let half_width = DoubleFloat::from_f64(kernel.half_width());
    format!(
        r#"const HORNER_COEFFICIENT_COUNT: u32 = {coefficient_count}u;
const HORNER_COEFFICIENTS: array<Df64, {coefficient_total}> =
    array<Df64, {coefficient_total}>({coefficients});
const DF64_HALF_WIDTH: Df64 = Df64({half_width_hi}, {half_width_lo});

fn df64_abs(value: Df64) -> Df64 {{
    if (value.hi < 0.0 || (value.hi == 0.0 && value.lo < 0.0)) {{
        return df64_neg(value);
    }}
    return value;
}}

fn df64_at_least(left: Df64, right: Df64) -> bool {{
    return left.hi > right.hi || (left.hi == right.hi && left.lo >= right.lo);
}}

fn es_weight(distance: Df64) -> Df64 {{
    if (df64_at_least(df64_abs(distance), DF64_HALF_WIDTH)) {{
        return Df64(0.0, 0.0);
    }}
    let shifted = df64_add(distance, DF64_HALF_WIDTH);
    let panel_i32 = clamp(ceil_df64_to_i32(shifted) - 1, 0, WIDTH_I32 - 1);
    let panel = u32(panel_i32);
    let centered = df64_sub(distance, Df64(f32(panel_i32), 0.0));
    let local = df64_add(
        df64_mul(Df64(2.0, 0.0), centered),
        Df64(f32(WIDTH_I32 - 1), 0.0),
    );
    var value = Df64(0.0, 0.0);
    for (var coefficient = 0u; coefficient < HORNER_COEFFICIENT_COUNT;
         coefficient = coefficient + 1u) {{
        value = df64_add(
            df64_mul(value, local),
            HORNER_COEFFICIENTS[coefficient * WIDTH + panel],
        );
    }}
    return value;
}}"#,
        half_width_hi = format_wgsl_f32(half_width.hi),
        half_width_lo = format_wgsl_f32(half_width.lo),
    )
}

fn tiled_gather_halo_shape(width: usize) -> Result<[usize; DIMENSIONS]> {
    let diameter = width
        .div_ceil(2)
        .checked_mul(2)
        .and_then(|value| value.checked_sub(1))
        .ok_or(NufftError::LengthOverflow {
            context: "type-1 3D tiled gather halo diameter",
        })?;
    let mut halo_shape = [0usize; DIMENSIONS];
    for axis in 0..DIMENSIONS {
        halo_shape[axis] = TILED_GATHER_TILE_SHAPE[axis].checked_add(diameter).ok_or(
            NufftError::LengthOverflow {
                context: "type-1 3D tiled gather halo shape",
            },
        )?;
    }
    let halo_entries = halo_shape
        .iter()
        .try_fold(1usize, |count, &length| count.checked_mul(length))
        .and_then(|bins| bins.checked_add(1))
        .ok_or(NufftError::LengthOverflow {
            context: "type-1 3D tiled gather halo prefix",
        })?;
    if halo_entries > TILED_GATHER_SCAN_ELEMENTS {
        return Err(NufftError::LengthOverflow {
            context: "type-1 3D tiled gather scan capacity",
        });
    }
    Ok(halo_shape)
}

fn tiled_gather_cache_capacity(width: usize) -> Result<usize> {
    let storage_bytes = tiled_gather_storage_bytes(width, TILED_GATHER_CACHE_CAPACITY)? as usize;
    if storage_bytes > TILED_GATHER_STORAGE_BUDGET_BYTES {
        return Err(NufftError::LengthOverflow {
            context: "type-1 3D tiled gather storage budget",
        });
    }
    Ok(TILED_GATHER_CACHE_CAPACITY)
}

fn tiled_gather_storage_bytes(width: usize, cache_capacity: usize) -> Result<u32> {
    let prefix_bytes = TILED_GATHER_PADDED_SCAN_ELEMENTS
        .checked_mul(size_of::<u32>())
        .ok_or(NufftError::LengthOverflow {
            context: "type-1 3D tiled gather prefix storage bytes",
        })?;
    // Separate scalar start arrays avoid the 16-byte array stride of vec3<i32>.
    let per_point_bytes = DIMENSIONS
        .checked_mul(size_of::<i32>())
        .and_then(|starts| starts.checked_add(size_of::<u32>()))
        .and_then(|base| {
            width
                .checked_mul(DIMENSIONS)
                .and_then(|weights| weights.checked_mul(size_of::<f32>()))
                .and_then(|weights| base.checked_add(weights))
        })
        .ok_or(NufftError::LengthOverflow {
            context: "type-1 3D tiled gather per-point storage bytes",
        })?;
    let bytes = cache_capacity
        .checked_mul(per_point_bytes)
        .and_then(|cache| cache.checked_add(prefix_bytes))
        .ok_or(NufftError::LengthOverflow {
            context: "type-1 3D tiled gather workgroup storage bytes",
        })?;
    u32::try_from(bytes).map_err(|_| NufftError::LengthOverflow {
        context: "type-1 3D tiled gather workgroup storage size",
    })
}

fn tiled_gather_tile_counts(fine_shape: [usize; DIMENSIONS]) -> Result<([u32; DIMENSIONS], u32)> {
    let mut tile_counts = [0usize; DIMENSIONS];
    for axis in 0..DIMENSIONS {
        tile_counts[axis] = fine_shape[axis].div_ceil(TILED_GATHER_TILE_SHAPE[axis]);
    }
    let tile_count = tile_counts
        .iter()
        .try_fold(1usize, |count, &length| count.checked_mul(length))
        .ok_or(NufftError::LengthOverflow {
            context: "type-1 3D tiled gather tile count",
        })?;
    let tile_counts = [
        u32::try_from(tile_counts[0]).map_err(|_| NufftError::LengthOverflow {
            context: "type-1 3D tiled gather axis-0 tile count",
        })?,
        u32::try_from(tile_counts[1]).map_err(|_| NufftError::LengthOverflow {
            context: "type-1 3D tiled gather axis-1 tile count",
        })?,
        u32::try_from(tile_counts[2]).map_err(|_| NufftError::LengthOverflow {
            context: "type-1 3D tiled gather axis-2 tile count",
        })?,
    ];
    let tile_count = u32::try_from(tile_count).map_err(|_| NufftError::LengthOverflow {
        context: "type-1 3D tiled gather shader tile count",
    })?;
    Ok((tile_counts, tile_count))
}

#[cfg(test)]
fn tiled_gather_dispatch(
    fine_shape: [usize; DIMENSIONS],
    max_workgroups_per_dimension: u32,
) -> Result<(u32, u32, u32)> {
    let (_, tile_count) = tiled_gather_tile_counts(fine_shape)?;
    split_workgroups(tile_count, max_workgroups_per_dimension)
}

fn tiled_gather_workgroups(fine_shape: [usize; DIMENSIONS]) -> Result<u32> {
    let (_, tile_count) = tiled_gather_tile_counts(fine_shape)?;
    Ok(tile_count)
}

fn workgroups_for_elements(element_count: usize) -> Result<u32> {
    let elements = u32::try_from(element_count).map_err(|_| NufftError::LengthOverflow {
        context: "type-1 3D GPU dispatch element count",
    })?;
    Ok(elements.div_ceil(WORKGROUP_SIZE))
}

fn format_wgsl_f32(value: f32) -> String {
    debug_assert!(value.is_finite());
    let mut formatted = value.to_string();
    if !formatted.contains('.') && !formatted.contains('e') && !formatted.contains('E') {
        formatted.push_str(".0");
    }
    formatted
}

fn format_wgsl_f64(value: f64) -> String {
    debug_assert!(value.is_finite());
    let mut formatted = value.to_string();
    if !formatted.contains('.') && !formatted.contains('e') && !formatted.contains('E') {
        formatted.push_str(".0");
    }
    formatted.push_str("lf");
    formatted
}

fn dispatch_for_elements(
    elements: usize,
    max_workgroups_per_dimension: u32,
) -> Result<(u32, u32, u32)> {
    let elements = u32::try_from(elements).map_err(|_| NufftError::LengthOverflow {
        context: "type-1 3D GPU dispatch element count",
    })?;
    split_workgroups(
        elements.div_ceil(WORKGROUP_SIZE),
        max_workgroups_per_dimension,
    )
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

fn validate_point_coordinate_shader_index(point_count: usize) -> Result<()> {
    point_count
        .checked_mul(DIMENSIONS)
        .and_then(|count| u32::try_from(count).ok())
        .ok_or(NufftError::LengthOverflow {
            context: "type-1 3D point-coordinate shader index space",
        })?;
    Ok(())
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
        Err(NufftError::GpuBufferTooSmall {
            buffer,
            required_bytes: bytes,
            actual_bytes: limit,
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
    fn count_uses_xyz_points_and_axis_zero_fast_bins() {
        let source = generate_count_wgsl([64, 96, 128]);
        assert!(source.contains("let total = arrayLength(&points) / 3u;"));
        assert!(source.contains("let point_base = point_index * 3u;"));
        assert!(source.contains("points[point_base + 2u]"));
        assert!(source.contains("let bin = bin_0 + FINE_0 * (bin_1 + FINE_1 * bin_2);"));
    }

    #[test]
    fn position_folding_preserves_low_words_on_all_axes() {
        let source = generate_position_wgsl([65_536, 98_304, 131_072]);
        for axis in 0..DIMENSIONS {
            assert!(source.contains(&format!("const POSITION_SCALE_{axis}_LO: f32 =")));
            assert!(source.contains(&format!("POSITION_SCALE_{axis}_LO,")));
        }
        assert!(source.contains("fn fold_position_2(point: f32) -> Df64"));
    }

    #[test]
    fn gather_is_one_writer_with_xyz_tensor_weights() {
        let kernel = EsKernel::for_tolerance(1.0e-6, 2.0).unwrap();
        let source = generate_gather_wgsl(kernel, [64, 96, 128]);
        assert!(source.contains("for (var bin_offset_2 = -BIN_RADIUS"));
        assert!(source.contains("for (var bin_offset_1 = -BIN_RADIUS"));
        assert!(source.contains("for (var bin_offset_0 = -BIN_RADIUS"));
        assert!(source.contains("let cell_2 = cell / FINE_01;"));
        assert!(source.contains("let weight = (weight_0 * weight_1) * weight_2;"));
        assert!(
            source.contains("fine_grid[vector_index * FINE_COUNT + cell] = sums[vector_local];")
        );
        assert!(!source.contains("fine_grid[cell] = fine_grid[cell] + sum"));
    }

    #[test]
    fn native_f64_shaders_use_horner_without_transcendentals() {
        let kernel = EsKernel::for_tolerance(1.0e-6, 2.0).unwrap();
        let config = NufftConfig::new([17, 19, 21], 1.0e-6).with_precision(FftPrecision::F64);
        let source = [
            generate_count_wgsl_for_precision([36, 40, 44], FftPrecision::F64),
            generate_scatter_wgsl_for_precision([36, 40, 44], FftPrecision::F64),
            generate_gather_wgsl_f64(kernel, [36, 40, 44]),
            generate_deconvolution_wgsl_for_precision(&config, [36, 40, 44], FftPrecision::F64),
        ]
        .join("\n");
        assert!(source.contains("array<vec2<f64>>"));
        assert!(source.contains("HORNER_COEFFICIENTS"));
        for forbidden in ["exp(", "log(", "pow(", "sin(", "cos("] {
            assert!(!source.contains(forbidden), "found {forbidden} in f64 WGSL");
        }
    }

    #[test]
    fn tiled_gather_width_seven_uses_the_bounded_storage_budget() {
        assert_eq!(tiled_gather_halo_shape(7).unwrap(), [15, 15, 11]);
        assert_eq!(15 * 15 * 11, 2_475);
        assert_eq!(TILED_GATHER_SCAN_ELEMENTS, 4_096);
        assert_eq!(TILED_GATHER_PADDED_SCAN_ELEMENTS, 4_224);
        let capacity = tiled_gather_cache_capacity(7).unwrap();
        assert_eq!(capacity, 300);
        let storage_bytes = tiled_gather_storage_bytes(7, capacity).unwrap();
        assert_eq!(storage_bytes, 46_896);
        assert!(storage_bytes > 16 * 1024);
        assert!(storage_bytes <= 48 * 1024);
        assert!(storage_bytes <= TILED_GATHER_STORAGE_BUDGET_BYTES as u32);
        assert!(tiled_gather_cache_capacity(8).is_err());
    }

    #[test]
    fn tiled_gather_batches_xyz_weights_behind_uniform_barriers() {
        let kernel = EsKernel::for_tolerance(1.0e-6, 2.0).unwrap();
        let source = generate_tiled_gather_wgsl(kernel, [64, 96, 128], 300).unwrap();
        assert!(source.contains("@compute @workgroup_size(256)"));
        assert!(source.contains("const HALO_0: u32 = 15u;"));
        assert!(source.contains("const HALO_1: u32 = 15u;"));
        assert!(source.contains("const HALO_2: u32 = 11u;"));
        assert!(source.contains("const HALO_BIN_COUNT: u32 = 2475u;"));
        assert!(source.contains("var<workgroup> halo_prefix: array<u32, 4224>;"));
        assert!(source.contains("var<workgroup> cached_starts_0: array<i32, 300>;"));
        assert!(source.contains("var<workgroup> cached_starts_2: array<i32, 300>;"));
        assert!(source.contains("var<workgroup> cached_weights_2: array<f32, 2100>;"));
        assert!(source.contains("let halo_total = workgroupUniformLoad("));
        assert!(
            source.contains("let batch_length = min(CACHE_CAPACITY, halo_total - batch_start);")
        );
        assert!(source.contains("for (var bin_offset_2 = -BIN_RADIUS"));
        assert!(source.contains("for (var bin_offset_1 = -BIN_RADIUS"));
        assert!(source.contains("for (var bin_offset_0 = -BIN_RADIUS"));
        assert!(source.contains("let weight = (weight_0 * weight_1) * weight_2;"));
        assert!(source.contains("var<workgroup> cached_point_indices: array<u32, 300>;"));
        assert!(source
            .contains("fine_grid[vector_index * FINE_COUNT + cell_index] = sums[vector_local];"));
        assert!(!source.contains("atomicAdd"));
        assert!(source.matches("workgroupBarrier();").count() >= 6);
    }

    #[test]
    fn tiled_gather_dispatches_one_workgroup_per_eight_by_eight_by_four_tile() {
        assert_eq!(
            tiled_gather_tile_counts([512, 512, 512]).unwrap(),
            ([64, 64, 128], 524_288)
        );
        let dispatch = tiled_gather_dispatch([512, 512, 512], 65_535).unwrap();
        assert_ne!(dispatch, (524_288, 1, 1));
        assert!(dispatch.0 <= 65_535 && dispatch.1 <= 65_535 && dispatch.2 <= 65_535);
        assert!(u64::from(dispatch.0) * u64::from(dispatch.1) * u64::from(dispatch.2) >= 524_288);
        assert_eq!(
            tiled_gather_tile_counts([30, 36, 22]).unwrap(),
            ([4, 5, 6], 120)
        );
    }

    #[test]
    fn scatter_is_determinized_by_per_bin_sort() {
        let scatter = generate_scatter_wgsl([64, 96, 128]);
        assert!(scatter.contains("atomicAdd(&bin_cursors[bin], 1u)"));
        let sort = generate_sort_wgsl();
        assert!(sort.contains("heap_sort(bin_offsets[bin], bin_offsets[bin + 1u]);"));
        assert!(generate_terminal_wgsl()
            .contains("bin_offsets[bin_count - 1u] + bin_counts[bin_count - 1u]"));
    }

    #[test]
    fn deconvolution_maps_three_axes_for_both_mode_orders() {
        for order in [ModeOrder::Centered, ModeOrder::Fft] {
            let config = NufftConfig::new([17, 19, 21], 1.0e-6).with_mode_order(order);
            let source = generate_deconvolution_wgsl(&config, [36, 40, 44]);
            assert!(source.contains("let storage_1 = (output_index / MODE_0) % MODE_1;"));
            assert!(source.contains("let storage_2 = output_index / MODE_01;"));
            assert!(
                source.contains("let fine_index = fine_0 + FINE_0 * (fine_1 + FINE_1 * fine_2);")
            );
            assert!(source.contains(&format!(
                "const CENTERED_ORDER: bool = {};",
                matches!(order, ModeOrder::Centered)
            )));
        }
    }

    #[test]
    fn amplitudes_are_axis_zero_fast_tensor_products() {
        let config = NufftConfig::new([2, 2, 2], 1.0e-6).with_mode_order(ModeOrder::Fft);
        let coefficients_0 = [2.0, 3.0];
        let coefficients_1 = [5.0, 7.0];
        let coefficients_2 = [11.0, 13.0];
        let values =
            mode_amplitudes(&config, [&coefficients_0, &coefficients_1, &coefficients_2]).unwrap();
        let expected = [
            1.0 / (2.0 * 5.0 * 11.0),
            1.0 / (3.0 * 5.0 * 11.0),
            1.0 / (2.0 * 7.0 * 11.0),
            1.0 / (3.0 * 7.0 * 11.0),
            1.0 / (2.0 * 5.0 * 13.0),
            1.0 / (3.0 * 5.0 * 13.0),
            1.0 / (2.0 * 7.0 * 13.0),
            1.0 / (3.0 * 7.0 * 13.0),
        ];
        for (actual, expected) in values.into_iter().zip(expected) {
            assert_eq!(actual, expected as f32);
        }
    }

    #[test]
    fn buffer_and_shader_index_arithmetic_is_checked() {
        assert_eq!(
            Type1GpuPlan3d::point_buffer_size_bytes_for_precision(7, FftPrecision::F32).unwrap(),
            84
        );
        assert_eq!(
            Type1GpuPlan3d::strength_buffer_size_bytes_for_precision(7, FftPrecision::F32).unwrap(),
            56
        );
        assert!(validate_point_coordinate_shader_index(u32::MAX as usize / 3).is_ok());
        assert!(matches!(
            validate_point_coordinate_shader_index(u32::MAX as usize / 3 + 1),
            Err(NufftError::LengthOverflow {
                context: "type-1 3D point-coordinate shader index space"
            })
        ));
    }
}
