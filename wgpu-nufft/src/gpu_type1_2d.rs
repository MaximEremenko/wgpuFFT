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

const DIMENSIONS: usize = 2;
const WORKGROUP_SIZE: u32 = 64;
const VECTOR_BLOCK_SIZE: usize = 4;
const TILED_GATHER_TILE_SIDE: usize = 16;
const TILED_GATHER_WORKGROUP_SIZE: u32 = 256;
const TILED_GATHER_SCAN_ELEMENTS: usize = 1024;
const TILED_GATHER_SCAN_BANK_WIDTH: usize = 32;
const TILED_GATHER_PADDED_SCAN_ELEMENTS: usize =
    TILED_GATHER_SCAN_ELEMENTS + TILED_GATHER_SCAN_ELEMENTS / TILED_GATHER_SCAN_BANK_WIDTH;
const TILED_GATHER_STORAGE_BUDGET_BYTES: usize = 15 * 1024;
const COMPLEX_F32_BYTES: u64 = 8;
const COMPLEX_F64_BYTES: u64 = 16;
const POINT_F32_BYTES: u64 = DIMENSIONS as u64 * 4;
const POINT_F64_BYTES: u64 = DIMENSIONS as u64 * 8;
const F32_BYTES: u64 = 4;
const F64_BYTES: u64 = 8;
const U32_BYTES: u64 = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Type1Gather2d {
    #[cfg_attr(not(feature = "type1-2d-tile-prototype"), allow(dead_code))]
    Global,
    Tiled16,
}

/// Device-specific resources for deterministic, atomics-free 2D spreading.
///
/// Points are point-major `x, y` pairs. Integer atomics construct one bin per
/// fine-grid cell, the point indices in each flattened `x + F0*y` bin are
/// sorted into their original input order, and one invocation owns each
/// fine-grid output cell.
pub(crate) struct Type1GpuPlan2d {
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

impl Type1GpuPlan2d {
    pub(crate) fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        config: &NufftConfig,
        kernel: EsKernel,
        fine_shape: [usize; DIMENSIONS],
        kernel_fourier_coefficients: [&[f64]; DIMENSIONS],
        gather: Type1Gather2d,
    ) -> Result<Self> {
        if config.dimensions() != DIMENSIONS {
            return Err(NufftError::GpuDimensionsUnsupported {
                kind: "type-1",
                actual: config.dimensions(),
                supported: DIMENSIONS,
            });
        }
        // Each support coordinate must remain an exact integer in the selected
        // position-arithmetic precision. Reserve the support halo beyond the
        // final cell independently on both axes.
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
                stage: "type-1 2D oversampled-grid C2C plan",
                source,
            })?;
        let fine_count =
            fft_config
                .total_complex_len()
                .map_err(|source| NufftError::FftShapeUnsupported {
                    stage: "type-1 2D oversampled-grid C2C plan",
                    source,
                })?;
        let batch_capacity = config.batch();
        let fft_config = fft_config.with_batch(batch_capacity);
        let total_fine_count =
            fine_count
                .checked_mul(batch_capacity)
                .ok_or(NufftError::LengthOverflow {
                    context: "batched type-1 2D fine-grid element count",
                })?;
        let mode_count = config.mode_count()?;
        let fine_bytes = Self::complex_buffer_size_bytes_for_precision(
            "type-1 2D fine grid",
            total_fine_count,
            precision,
        )?;
        let amplitude_bytes = checked_buffer_size(
            "type-1 2D deconvolution amplitudes",
            mode_count,
            scalar_size_bytes(precision),
        )?;
        let count_bytes = checked_buffer_size("type-1 2D bin counts", fine_count, U32_BYTES)?;
        let offset_count = fine_count
            .checked_add(1)
            .ok_or(NufftError::LengthOverflow {
                context: "type-1 2D bin offset count",
            })?;
        let offset_bytes = checked_buffer_size("type-1 2D bin offsets", offset_count, U32_BYTES)?;
        for (label, bytes) in [
            ("type-1 2D fine grid", fine_bytes),
            ("type-1 2D deconvolution amplitudes", amplitude_bytes),
            ("type-1 2D bin counts", count_bytes),
            ("type-1 2D bin offsets", offset_bytes),
        ] {
            validate_binding_limit(label, bytes, limits.max_storage_buffer_binding_size)?;
            validate_buffer_limit(label, bytes, limits.max_buffer_size)?;
        }

        let amplitudes = mode_amplitude_bytes(config, kernel_fourier_coefficients, precision)?;
        let amplitudes = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("wgpu_nufft.type1_2d.deconvolution_amplitudes"),
            contents: &amplitudes,
            usage: wgpu::BufferUsages::STORAGE,
        });
        let fine_usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST;
        let fine_input = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("wgpu_nufft.type1_2d.fine_input"),
            size: fine_bytes,
            usage: fine_usage,
            mapped_at_creation: false,
        });
        let fine_output = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("wgpu_nufft.type1_2d.fine_output"),
            size: fine_bytes,
            usage: fine_usage,
            mapped_at_creation: false,
        });
        let scratch =
            Type1ScratchBuffers::new(device, "wgpu_nufft.type1_2d", count_bytes, offset_bytes);
        let fft = FftPlan::c2c(device, queue, fft_config).map_err(|source| {
            NufftError::FftShapeUnsupported {
                stage: "type-1 2D oversampled-grid C2C plan",
                source,
            }
        })?;

        let count_pipeline = create_compute_pipeline(
            device,
            "wgpu_nufft.type1_2d.bin_count",
            &generate_count_wgsl_for_precision(fine_shape, precision),
        );
        let count_layout = count_pipeline.get_bind_group_layout(0);
        let prefix_scan = GpuExclusiveScanU32::new(device, fine_count)?;
        let terminal_pipeline = create_compute_pipeline(
            device,
            "wgpu_nufft.type1_2d.bin_terminal",
            &generate_terminal_wgsl(),
        );
        let terminal_layout = terminal_pipeline.get_bind_group_layout(0);
        let scatter_pipeline = create_compute_pipeline(
            device,
            "wgpu_nufft.type1_2d.bin_scatter",
            &generate_scatter_wgsl_for_precision(fine_shape, precision),
        );
        let scatter_layout = scatter_pipeline.get_bind_group_layout(0);
        let sort_pipeline = create_compute_pipeline(
            device,
            "wgpu_nufft.type1_2d.bin_sort",
            &generate_sort_wgsl(),
        );
        let sort_layout = sort_pipeline.get_bind_group_layout(0);
        let max_workgroups_per_dimension = limits.max_compute_workgroups_per_dimension;
        let (gather_label, gather_source, gather_workgroups_per_vector_block) = match precision {
            FftPrecision::F64 => (
                "wgpu_nufft.type1_2d.spread_gather_f64",
                generate_gather_wgsl_f64(kernel, fine_shape),
                workgroups_for_elements(fine_count)?,
            ),
            FftPrecision::Df64 => (
                "wgpu_nufft.type1_2d.spread_gather_df64",
                generate_gather_wgsl_df64(kernel, fine_shape),
                workgroups_for_elements(fine_count)?,
            ),
            FftPrecision::F32 => match gather {
                Type1Gather2d::Global => (
                    "wgpu_nufft.type1_2d.spread_gather",
                    generate_gather_wgsl(kernel, fine_shape),
                    workgroups_for_elements(fine_count)?,
                ),
                Type1Gather2d::Tiled16 => {
                    let halo_side = tiled_gather_halo_side(kernel.width())?;
                    if fine_shape.iter().any(|&length| length < halo_side)
                        || maximum_workgroup_size < TILED_GATHER_WORKGROUP_SIZE
                    {
                        (
                            "wgpu_nufft.type1_2d.spread_gather",
                            generate_gather_wgsl(kernel, fine_shape),
                            workgroups_for_elements(fine_count)?,
                        )
                    } else {
                        let cache_capacity = tiled_gather_cache_capacity(kernel.width())?;
                        let storage_bytes =
                            tiled_gather_storage_bytes(kernel.width(), cache_capacity)?;
                        if storage_bytes > limits.max_compute_workgroup_storage_size {
                            (
                                "wgpu_nufft.type1_2d.spread_gather",
                                generate_gather_wgsl(kernel, fine_shape),
                                workgroups_for_elements(fine_count)?,
                            )
                        } else {
                            (
                                "wgpu_nufft.type1_2d.spread_gather_tiled16",
                                generate_tiled_gather_wgsl(kernel, fine_shape, cache_capacity)?,
                                tiled_gather_workgroups(fine_shape)?,
                            )
                        }
                    }
                }
            },
        };
        let gather_pipeline = create_compute_pipeline(device, gather_label, &gather_source);
        let gather_layout = gather_pipeline.get_bind_group_layout(0);
        let deconvolution_pipeline = create_compute_pipeline(
            device,
            "wgpu_nufft.type1_2d.deconvolution",
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
        checked_buffer_size("type-1 2D point buffer", point_count, bytes)
    }

    pub(crate) fn strength_buffer_size_bytes_for_precision(
        point_count: usize,
        precision: FftPrecision,
    ) -> Result<u64> {
        Self::complex_buffer_size_bytes_for_precision(
            "type-1 2D strength buffer",
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
                kind: "type-1 2D stage profiling",
                reason: "at least one point is required",
            });
        }
        let layout =
            NufftGpuProfileLayout::type1(first_query).map_err(|_| NufftError::LengthOverflow {
                context: "type-1 2D stage-profile query range",
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
                    context: "batched type-1 2D Fourier output element count",
                })?;
        let output_bytes = Self::complex_buffer_size_bytes_for_precision(
            "type-1 2D Fourier output buffer",
            output_elements,
            self.precision,
        )?;
        validate_external_storage_buffer(
            "type-1 2D Fourier output",
            output,
            output_bytes,
            self.max_storage_binding_bytes,
        )?;

        let output_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("wgpu_nufft.type1_2d.deconvolution.bind_group"),
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
                context: "type-1 2D GPU point count",
            })?;
        validate_point_coordinate_shader_index(point_count)?;
        let point_bytes = Self::point_buffer_size_bytes_for_precision(point_count, self.precision)?;
        let strength_elements =
            point_count
                .checked_mul(active_batch)
                .ok_or(NufftError::LengthOverflow {
                    context: "batched type-1 2D strength element count",
                })?;
        let strength_bytes =
            Self::strength_buffer_size_bytes_for_precision(strength_elements, self.precision)?;
        validate_external_storage_buffer(
            "type-1 2D point",
            points,
            point_bytes,
            self.max_storage_binding_bytes,
        )?;
        validate_external_storage_buffer(
            "type-1 2D strength",
            strengths,
            strength_bytes,
            self.max_storage_binding_bytes,
        )?;

        let fine_grid_element_count = self.fine_grid_element_count();
        let count_bytes =
            checked_buffer_size("type-1 2D bin counts", fine_grid_element_count, U32_BYTES)?;
        let offset_count =
            self.fine_grid_element_count()
                .checked_add(1)
                .ok_or(NufftError::LengthOverflow {
                    context: "type-1 2D bin offset count",
                })?;
        let offset_bytes = checked_buffer_size("type-1 2D bin offsets", offset_count, U32_BYTES)?;
        let index_bytes =
            checked_buffer_size("type-1 2D sorted point indices", point_count, U32_BYTES)?;
        for (label, bytes) in [
            ("type-1 2D bin counts", count_bytes),
            ("type-1 2D bin cursors", count_bytes),
            ("type-1 2D bin offsets", offset_bytes),
            ("type-1 2D sorted point indices", index_bytes),
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
            label: Some("wgpu_nufft.type1_2d.bin_count.bind_group"),
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
            label: Some("wgpu_nufft.type1_2d.bin_terminal.bind_group"),
            layout: &self.terminal_layout,
            entries: &[
                binding_entry(0, bin_counts, count_bytes),
                binding_entry(1, bin_offsets, offset_bytes),
            ],
        });
        let scatter_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("wgpu_nufft.type1_2d.bin_scatter.bind_group"),
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
            label: Some("wgpu_nufft.type1_2d.bin_sort.bind_group"),
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
                context: "active batched type-1 2D fine-grid element count",
            },
        )?;
        let active_fine_bytes = Self::complex_buffer_size_bytes_for_precision(
            "active batched type-1 2D fine grid",
            active_fine_elements,
            self.precision,
        )?;
        let gather_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("wgpu_nufft.type1_2d.spread_gather.bind_group"),
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
                context: "batched type-1 2D gather workgroup count",
            })?;
        let gather_dispatch =
            split_workgroups(gather_workgroups, self.max_workgroups_per_dimension)?;

        #[cfg(feature = "gpu-profiling")]
        profile.encode_type1_start_marker(encoder);
        encoder.clear_buffer(bin_counts, 0, None);
        encoder.clear_buffer(bin_cursors, 0, None);
        encode_pass(
            encoder,
            "wgpu_nufft.type1_2d.bin_count.pass",
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
            "wgpu_nufft.type1_2d.bin_terminal.pass",
            &self.terminal_pipeline,
            &terminal_bind_group,
            (1, 1, 1),
            #[cfg(feature = "gpu-profiling")]
            profile.timestamp_writes(None, Some(2)),
        );
        encode_pass(
            encoder,
            "wgpu_nufft.type1_2d.bin_scatter.pass",
            &self.scatter_pipeline,
            &scatter_bind_group,
            point_dispatch,
            #[cfg(feature = "gpu-profiling")]
            profile.timestamp_writes(None, Some(3)),
        );
        encode_pass(
            encoder,
            "wgpu_nufft.type1_2d.bin_sort.pass",
            &self.sort_pipeline,
            &sort_bind_group,
            self.sort_dispatch,
            #[cfg(feature = "gpu-profiling")]
            profile.timestamp_writes(None, Some(4)),
        );
        encode_pass(
            encoder,
            "wgpu_nufft.type1_2d.spread_gather.pass",
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
                stage: "type-1 2D oversampled-grid C2C transform",
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
            "wgpu_nufft.type1_2d.deconvolution.pass",
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
    let mode_count = config.mode_count()?;
    let mut amplitudes = Vec::new();
    amplitudes
        .try_reserve_exact(mode_count)
        .map_err(|_| NufftError::HostAllocationFailed {
            buffer: "type-1 2D deconvolution amplitudes",
            elements: mode_count,
        })?;
    for linear in 0..mode_count {
        let storage_0 = linear % mode_shape[0];
        let storage_1 = linear / mode_shape[0];
        let mode_0 = mode_for_storage_index(config.mode_order(), storage_0, mode_shape[0]);
        let mode_1 = mode_for_storage_index(config.mode_order(), storage_1, mode_shape[1]);
        let coefficient_0 = coefficients[0][mode_0.unsigned_abs() as usize];
        let coefficient_1 = coefficients[1][mode_1.unsigned_abs() as usize];
        for (mode, coefficient) in [(mode_0, coefficient_0), (mode_1, coefficient_1)] {
            if !coefficient.is_finite() || coefficient == 0.0 {
                return Err(NufftError::InvalidKernelFourierCoefficient {
                    mode,
                    value: coefficient,
                });
            }
        }
        let denominator = coefficient_0 * coefficient_1;
        let amplitude = (1.0 / denominator) as f32;
        if !denominator.is_finite() || denominator == 0.0 || !amplitude.is_finite() {
            return Err(NufftError::InvalidKernelFourierCoefficient {
                mode: mode_0,
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
    let mode_count = config.mode_count()?;
    let mut amplitudes = Vec::new();
    amplitudes
        .try_reserve_exact(mode_count)
        .map_err(|_| NufftError::HostAllocationFailed {
            buffer: "type-1 2D deconvolution amplitudes",
            elements: mode_count,
        })?;
    for linear in 0..mode_count {
        let storage_0 = linear % mode_shape[0];
        let storage_1 = linear / mode_shape[0];
        let mode_0 = mode_for_storage_index(config.mode_order(), storage_0, mode_shape[0]);
        let mode_1 = mode_for_storage_index(config.mode_order(), storage_1, mode_shape[1]);
        let coefficient_0 = coefficients[0][mode_0.unsigned_abs() as usize];
        let coefficient_1 = coefficients[1][mode_1.unsigned_abs() as usize];
        let denominator = coefficient_0 * coefficient_1;
        let amplitude = 1.0 / denominator;
        if !coefficient_0.is_finite()
            || coefficient_0 == 0.0
            || !coefficient_1.is_finite()
            || coefficient_1 == 0.0
            || !denominator.is_finite()
            || denominator == 0.0
            || !amplitude.is_finite()
        {
            return Err(NufftError::InvalidKernelFourierCoefficient {
                mode: mode_0,
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
    let scale_0 = fine_shape[0] as f64 / std::f64::consts::TAU;
    let scale_1 = fine_shape[1] as f64 / std::f64::consts::TAU;
    let scale_0_hi = scale_0 as f32;
    let scale_1_hi = scale_1 as f32;
    let scale_0_lo = (scale_0 - f64::from(scale_0_hi)) as f32;
    let scale_1_lo = (scale_1 - f64::from(scale_1_hi)) as f32;
    format!(
        r#"const FINE_0: u32 = {fine_0}u;
const FINE_1: u32 = {fine_1}u;
const FINE_0_I32: i32 = {fine_0}i;
const FINE_1_I32: i32 = {fine_1}i;
const FINE_0_F32: f32 = {fine_0_f32};
const FINE_1_F32: f32 = {fine_1_f32};
const FINE_COUNT: u32 = {fine_count}u;
const POSITION_SCALE_0_HI: f32 = {scale_0_hi};
const POSITION_SCALE_0_LO: f32 = {scale_0_lo};
const POSITION_SCALE_1_HI: f32 = {scale_1_hi};
const POSITION_SCALE_1_LO: f32 = {scale_1_lo};
const GRID_ORIGIN_0: f32 = {origin_0};
const GRID_ORIGIN_1: f32 = {origin_1};

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
    return fold_position(
        point,
        POSITION_SCALE_0_HI,
        POSITION_SCALE_0_LO,
        GRID_ORIGIN_0,
        FINE_0_F32,
    );
}}

fn fold_position_1(point: f32) -> Df64 {{
    return fold_position(
        point,
        POSITION_SCALE_1_HI,
        POSITION_SCALE_1_LO,
        GRID_ORIGIN_1,
        FINE_1_F32,
    );
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
        fine_0_f32 = format_wgsl_f32(fine_shape[0] as f32),
        fine_1_f32 = format_wgsl_f32(fine_shape[1] as f32),
        fine_count = fine_shape[0] * fine_shape[1],
        scale_0_hi = format_wgsl_f32(scale_0_hi),
        scale_0_lo = format_wgsl_f32(scale_0_lo),
        scale_1_hi = format_wgsl_f32(scale_1_hi),
        scale_1_lo = format_wgsl_f32(scale_1_lo),
        origin_0 = format_wgsl_f32((fine_shape[0] / 2) as f32),
        origin_1 = format_wgsl_f32((fine_shape[1] / 2) as f32),
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
}

fn generate_position_wgsl_f64(fine_shape: [usize; DIMENSIONS]) -> String {
    let scale_0 = fine_shape[0] as f64 / std::f64::consts::TAU;
    let scale_1 = fine_shape[1] as f64 / std::f64::consts::TAU;
    format!(
        r#"const FINE_0: u32 = {fine_0}u;
const FINE_1: u32 = {fine_1}u;
const FINE_0_I32: i32 = {fine_0}i;
const FINE_1_I32: i32 = {fine_1}i;
const FINE_0_F64: f64 = {fine_0_f64};
const FINE_1_F64: f64 = {fine_1_f64};
const FINE_COUNT: u32 = {fine_count}u;
const POSITION_SCALE_0: f64 = {scale_0};
const POSITION_SCALE_1: f64 = {scale_1};
const GRID_ORIGIN_0: f64 = {origin_0};
const GRID_ORIGIN_1: f64 = {origin_1};

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
"#,
        fine_0 = fine_shape[0],
        fine_1 = fine_shape[1],
        fine_0_f64 = format_wgsl_f64(fine_shape[0] as f64),
        fine_1_f64 = format_wgsl_f64(fine_shape[1] as f64),
        fine_count = fine_shape[0] * fine_shape[1],
        scale_0 = format_wgsl_f64(scale_0),
        scale_1 = format_wgsl_f64(scale_1),
        origin_0 = format_wgsl_f64((fine_shape[0] / 2) as f64),
        origin_1 = format_wgsl_f64((fine_shape[1] / 2) as f64),
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
    let total = arrayLength(&points) / 2u;
    let wg_flat = (wid.z * nwg.y + wid.y) * nwg.x + wid.x;
    if (wg_flat > (total - 1u) / WORKGROUP_SIZE) {{ return; }}
    let point_index = wg_flat * WORKGROUP_SIZE + lid.x;
    if (point_index >= total) {{ return; }}

    let point_base = point_index * 2u;
    let position_0 = fold_position_0(points[point_base]);
    let position_1 = fold_position_1(points[point_base + 1u]);
    let bin_0 = u32(floor_df64_to_i32(position_0));
    let bin_1 = u32(floor_df64_to_i32(position_1));
    atomicAdd(&bin_counts[bin_0 + FINE_0 * bin_1], 1u);
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
    let total = arrayLength(&points) / 2u;
    let wg_flat = (wid.z * nwg.y + wid.y) * nwg.x + wid.x;
    if (wg_flat > (total - 1u) / WORKGROUP_SIZE) {{ return; }}
    let point_index = wg_flat * WORKGROUP_SIZE + lid.x;
    if (point_index >= total) {{ return; }}
    let point_base = point_index * 2u;
    let bin_0 = u32(floor(fold_position_0(points[point_base])));
    let bin_1 = u32(floor(fold_position_1(points[point_base + 1u])));
    atomicAdd(&bin_counts[bin_0 + FINE_0 * bin_1], 1u);
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
    let total = arrayLength(&points) / 2u;
    let wg_flat = (wid.z * nwg.y + wid.y) * nwg.x + wid.x;
    if (wg_flat > (total - 1u) / WORKGROUP_SIZE) {{ return; }}
    let point_index = wg_flat * WORKGROUP_SIZE + lid.x;
    if (point_index >= total) {{ return; }}

    let point_base = point_index * 2u;
    let position_0 = fold_position_0(points[point_base]);
    let position_1 = fold_position_1(points[point_base + 1u]);
    let bin_0 = u32(floor_df64_to_i32(position_0));
    let bin_1 = u32(floor_df64_to_i32(position_1));
    let bin = bin_0 + FINE_0 * bin_1;
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
    let total = arrayLength(&points) / 2u;
    let wg_flat = (wid.z * nwg.y + wid.y) * nwg.x + wid.x;
    if (wg_flat > (total - 1u) / WORKGROUP_SIZE) {{ return; }}
    let point_index = wg_flat * WORKGROUP_SIZE + lid.x;
    if (point_index >= total) {{ return; }}

    let point_base = point_index * 2u;
    let position_0 = fold_position_0(points[point_base]);
    let position_1 = fold_position_1(points[point_base + 1u]);
    let bin_0 = u32(floor_df64_to_i32(position_0));
    let bin_1 = u32(floor_df64_to_i32(position_1));
    let bin = bin_0 + FINE_0 * bin_1;
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
    let total = arrayLength(&points) / 2u;
    let wg_flat = (wid.z * nwg.y + wid.y) * nwg.x + wid.x;
    if (wg_flat > (total - 1u) / WORKGROUP_SIZE) {{ return; }}
    let point_index = wg_flat * WORKGROUP_SIZE + lid.x;
    if (point_index >= total) {{ return; }}

    let point_base = point_index * 2u;
    let position_0 = fold_position_0(points[point_base]);
    let position_1 = fold_position_1(points[point_base + 1u]);
    let bin_0 = u32(floor_df64_to_i32(position_0));
    let bin_1 = u32(floor_df64_to_i32(position_1));
    let bin = bin_0 + FINE_0 * bin_1;
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
    let total = arrayLength(&points) / 2u;
    let wg_flat = (wid.z * nwg.y + wid.y) * nwg.x + wid.x;
    if (wg_flat > (total - 1u) / WORKGROUP_SIZE) {{ return; }}
    let point_index = wg_flat * WORKGROUP_SIZE + lid.x;
    if (point_index >= total) {{ return; }}
    let point_base = point_index * 2u;
    let bin_0 = u32(floor(fold_position_0(points[point_base])));
    let bin_1 = u32(floor(fold_position_1(points[point_base + 1u])));
    let bin = bin_0 + FINE_0 * bin_1;
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

fn support_cell(
    wrapped_cell: u32,
    start: i32,
    fine_length: i32,
) -> i32 {{
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
    let cell_1 = cell / FINE_0;
    var sums: array<vec2<f32>, {VECTOR_BLOCK_SIZE}>;
    for (var vector_local = 0u; vector_local < VECTOR_BLOCK_SIZE;
         vector_local = vector_local + 1u) {{
        sums[vector_local] = vec2<f32>(0.0, 0.0);
    }}
    for (var bin_offset_1 = -BIN_RADIUS; bin_offset_1 < BIN_RADIUS;
         bin_offset_1 = bin_offset_1 + 1) {{
        let bin_1 = wrap_bin(i32(cell_1) + bin_offset_1, FINE_1_I32);
        for (var bin_offset_0 = -BIN_RADIUS; bin_offset_0 < BIN_RADIUS;
             bin_offset_0 = bin_offset_0 + 1) {{
            let bin_0 = wrap_bin(i32(cell_0) + bin_offset_0, FINE_0_I32);
            let bin = bin_0 + FINE_0 * bin_1;
            let begin = bin_offsets[bin];
            let end = bin_offsets[bin + 1u];
            for (var slot = begin; slot < end; slot = slot + 1u) {{
                let point_index = sorted_indices[slot];
                let point_base = point_index * 2u;
                let position_0 = fold_position_0(points[point_base]);
                let position_1 = fold_position_1(points[point_base + 1u]);
                let start_0 = ceil_df64_to_i32(
                    df64_sub(position_0, Df64(HALF_WIDTH, 0.0)),
                );
                let start_1 = ceil_df64_to_i32(
                    df64_sub(position_1, Df64(HALF_WIDTH, 0.0)),
                );
                let unwrapped_0 = support_cell(cell_0, start_0, FINE_0_I32);
                let unwrapped_1 = support_cell(cell_1, start_1, FINE_1_I32);
                if (unwrapped_0 >= start_0 && unwrapped_0 < start_0 + WIDTH_I32 &&
                    unwrapped_1 >= start_1 && unwrapped_1 < start_1 + WIDTH_I32) {{
                    let distance_0 = df64_sub(
                        Df64(f32(unwrapped_0), 0.0),
                        position_0,
                    );
                    let distance_1 = df64_sub(
                        Df64(f32(unwrapped_1), 0.0),
                        position_1,
                    );
                    let weight = es_weight(distance_0.hi + distance_0.lo) *
                        es_weight(distance_1.hi + distance_1.lo);
                    for (var vector_local = 0u; vector_local < VECTOR_BLOCK_SIZE;
                         vector_local = vector_local + 1u) {{
                        let vector_index = first_vector + vector_local;
                        if (vector_index < total_vectors) {{
                            let strength_index = vector_index *
                                (arrayLength(&points) / 2u) + point_index;
                            sums[vector_local] = sums[vector_local] +
                                strengths[strength_index] * weight;
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
    let cell_1 = cell / FINE_0;
    var sums: array<vec4<f32>, {VECTOR_BLOCK_SIZE}>;
    for (var vector_local = 0u; vector_local < VECTOR_BLOCK_SIZE;
         vector_local = vector_local + 1u) {{
        sums[vector_local] = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }}
    for (var bin_offset_1 = -BIN_RADIUS; bin_offset_1 < BIN_RADIUS;
         bin_offset_1 = bin_offset_1 + 1) {{
        let bin_1 = wrap_bin(i32(cell_1) + bin_offset_1, FINE_1_I32);
        for (var bin_offset_0 = -BIN_RADIUS; bin_offset_0 < BIN_RADIUS;
             bin_offset_0 = bin_offset_0 + 1) {{
            let bin_0 = wrap_bin(i32(cell_0) + bin_offset_0, FINE_0_I32);
            let bin = bin_0 + FINE_0 * bin_1;
            for (var slot = bin_offsets[bin]; slot < bin_offsets[bin + 1u];
                 slot = slot + 1u) {{
                let point_index = sorted_indices[slot];
                let point_base = point_index * 2u;
                let position_0 = fold_position_0(points[point_base]);
                let position_1 = fold_position_1(points[point_base + 1u]);
                let start_0 = ceil_df64_to_i32(df64_sub(position_0, DF64_HALF_WIDTH));
                let start_1 = ceil_df64_to_i32(df64_sub(position_1, DF64_HALF_WIDTH));
                let unwrapped_0 = support_cell(cell_0, start_0, FINE_0_I32);
                let unwrapped_1 = support_cell(cell_1, start_1, FINE_1_I32);
                if (unwrapped_0 >= start_0 && unwrapped_0 < start_0 + WIDTH_I32 &&
                    unwrapped_1 >= start_1 && unwrapped_1 < start_1 + WIDTH_I32) {{
                    let distance_0 = df64_sub(Df64(f32(unwrapped_0), 0.0), position_0);
                    let distance_1 = df64_sub(Df64(f32(unwrapped_1), 0.0), position_1);
                    let weight = df64_mul(es_weight(distance_0), es_weight(distance_1));
                    for (var vector_local = 0u; vector_local < VECTOR_BLOCK_SIZE;
                         vector_local = vector_local + 1u) {{
                        let vector_index = first_vector + vector_local;
                        if (vector_index < total_vectors) {{
                            let strength_index = vector_index *
                                (arrayLength(&points) / 2u) + point_index;
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
    let cell_1 = cell / FINE_0;
    var sums: array<vec2<f64>, {VECTOR_BLOCK_SIZE}>;
    for (var vector_local = 0u; vector_local < VECTOR_BLOCK_SIZE;
         vector_local = vector_local + 1u) {{
        sums[vector_local] = vec2<f64>(0.0lf, 0.0lf);
    }}
    for (var bin_offset_1 = -BIN_RADIUS; bin_offset_1 < BIN_RADIUS;
         bin_offset_1 = bin_offset_1 + 1) {{
        let bin_1 = wrap_bin(i32(cell_1) + bin_offset_1, FINE_1_I32);
        for (var bin_offset_0 = -BIN_RADIUS; bin_offset_0 < BIN_RADIUS;
             bin_offset_0 = bin_offset_0 + 1) {{
            let bin_0 = wrap_bin(i32(cell_0) + bin_offset_0, FINE_0_I32);
            let bin = bin_0 + FINE_0 * bin_1;
            for (var slot = bin_offsets[bin]; slot < bin_offsets[bin + 1u];
                 slot = slot + 1u) {{
                let point_index = sorted_indices[slot];
                let point_base = point_index * 2u;
                let position_0 = fold_position_0(points[point_base]);
                let position_1 = fold_position_1(points[point_base + 1u]);
                let start_0 = i32(ceil(position_0 - HALF_WIDTH));
                let start_1 = i32(ceil(position_1 - HALF_WIDTH));
                let unwrapped_0 = support_cell(cell_0, start_0, FINE_0_I32);
                let unwrapped_1 = support_cell(cell_1, start_1, FINE_1_I32);
                if (unwrapped_0 >= start_0 && unwrapped_0 < start_0 + WIDTH_I32 &&
                    unwrapped_1 >= start_1 && unwrapped_1 < start_1 + WIDTH_I32) {{
                    let weight = es_weight(f64(unwrapped_0) - position_0) *
                        es_weight(f64(unwrapped_1) - position_1);
                    for (var vector_local = 0u; vector_local < VECTOR_BLOCK_SIZE;
                         vector_local = vector_local + 1u) {{
                        let vector_index = first_vector + vector_local;
                        if (vector_index < total_vectors) {{
                            let strength_index = vector_index *
                                (arrayLength(&points) / 2u) + point_index;
                            sums[vector_local] = sums[vector_local] +
                                strengths[strength_index] * weight;
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
    let halo_side = tiled_gather_halo_side(width)?;
    let halo_bin_count = halo_side
        .checked_mul(halo_side)
        .ok_or(NufftError::LengthOverflow {
            context: "type-1 2D tiled gather halo-bin count",
        })?;
    let cache_weight_count =
        cache_capacity
            .checked_mul(width)
            .ok_or(NufftError::LengthOverflow {
                context: "type-1 2D tiled gather cached-weight count",
            })?;
    let cache_load_rounds = cache_capacity.div_ceil(TILED_GATHER_WORKGROUP_SIZE as usize);
    let (tile_count_0, _tile_count_1, tile_count) = tiled_gather_tile_counts(fine_shape)?;
    debug_assert!(halo_bin_count < TILED_GATHER_SCAN_ELEMENTS);
    debug_assert!(cache_capacity > 0);
    let entry = format!(
        r#"const WORKGROUP_SIZE: u32 = {workgroup_size}u;
const TILE_SIDE: u32 = {tile_side}u;
const TILE_COUNT_0: u32 = {tile_count_0}u;
const TILE_COUNT: u32 = {tile_count}u;
const WIDTH: u32 = {width}u;
const WIDTH_I32: i32 = {width}i;
const WIDTH_F32: f32 = {width}.0;
const HALF_WIDTH: f32 = {half_width};
const BETA: f32 = {beta};
const BIN_RADIUS: i32 = {bin_radius}i;
const HALO_SIDE: u32 = {halo_side}u;
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
var<workgroup> cached_starts: array<vec2<i32>, {cache_capacity}>;
var<workgroup> cached_point_indices: array<u32, {cache_capacity}>;
var<workgroup> cached_weights_0: array<f32, {cache_weight_count}>;
var<workgroup> cached_weights_1: array<f32, {cache_weight_count}>;

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

fn halo_global_bin(halo_bin: u32, tile_origin: vec2<u32>) -> u32 {{
    let halo_0 = halo_bin % HALO_SIDE;
    let halo_1 = halo_bin / HALO_SIDE;
    let bin_0 = wrap_bin(i32(tile_origin.x + halo_0) - BIN_RADIUS, FINE_0_I32);
    let bin_1 = wrap_bin(i32(tile_origin.y + halo_1) - BIN_RADIUS, FINE_1_I32);
    return bin_0 + FINE_0 * bin_1;
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

fn support_cell(
    wrapped_cell: u32,
    start: i32,
    fine_length: i32,
) -> i32 {{
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
    let tile = vec2<u32>(tile_flat % TILE_COUNT_0, tile_flat / TILE_COUNT_0);
    let tile_origin = tile * TILE_SIDE;
    let local_cell = vec2<u32>(lid.x % TILE_SIDE, lid.x / TILE_SIDE);
    let cell = tile_origin + local_cell;
    let cell_in_bounds = cell.x < FINE_0 && cell.y < FINE_1;

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
                let point_base = point_index * 2u;
                let position_0 = fold_position_0(points[point_base]);
                let position_1 = fold_position_1(points[point_base + 1u]);
                let start_0 = ceil_df64_to_i32(
                    df64_sub(position_0, Df64(HALF_WIDTH, 0.0)),
                );
                let start_1 = ceil_df64_to_i32(
                    df64_sub(position_1, Df64(HALF_WIDTH, 0.0)),
                );
                cached_starts[cache_slot] = vec2<i32>(start_0, start_1);
                cached_point_indices[cache_slot] = point_index;
                for (var support = 0u; support < WIDTH; support = support + 1u) {{
                    let support_0 = start_0 + i32(support);
                    let support_1 = start_1 + i32(support);
                    let distance_0 = df64_sub(
                        Df64(f32(support_0), 0.0),
                        position_0,
                    );
                    let distance_1 = df64_sub(
                        Df64(f32(support_1), 0.0),
                        position_1,
                    );
                    let weight_slot = cache_slot * WIDTH + support;
                    cached_weights_0[weight_slot] =
                        es_weight(distance_0.hi + distance_0.lo);
                    cached_weights_1[weight_slot] =
                        es_weight(distance_1.hi + distance_1.lo);
                }}
            }}
        }}
        workgroupBarrier();

        if (cell_in_bounds) {{
            for (var bin_offset_1 = -BIN_RADIUS; bin_offset_1 < BIN_RADIUS;
                 bin_offset_1 = bin_offset_1 + 1) {{
                let halo_1 = u32(i32(local_cell.y) + bin_offset_1 + BIN_RADIUS);
                for (var bin_offset_0 = -BIN_RADIUS; bin_offset_0 < BIN_RADIUS;
                     bin_offset_0 = bin_offset_0 + 1) {{
                    let halo_0 = u32(i32(local_cell.x) + bin_offset_0 + BIN_RADIUS);
                    let halo_bin = halo_0 + HALO_SIDE * halo_1;
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
                        let start = cached_starts[cache_slot];
                        let unwrapped_0 = support_cell(cell.x, start.x, FINE_0_I32);
                        let unwrapped_1 = support_cell(cell.y, start.y, FINE_1_I32);
                        if (unwrapped_0 >= start.x && unwrapped_0 < start.x + WIDTH_I32 &&
                            unwrapped_1 >= start.y && unwrapped_1 < start.y + WIDTH_I32) {{
                            let support_0 = u32(unwrapped_0 - start.x);
                            let support_1 = u32(unwrapped_1 - start.y);
                            let weight = cached_weights_0[cache_slot * WIDTH + support_0] *
                                cached_weights_1[cache_slot * WIDTH + support_1];
                            let point_index = cached_point_indices[cache_slot];
                            for (var vector_local = 0u; vector_local < VECTOR_BLOCK_SIZE;
                                 vector_local = vector_local + 1u) {{
                                let vector_index = first_vector + vector_local;
                                if (vector_index < total_vectors) {{
                                    let strength_index = vector_index *
                                        (arrayLength(&points) / 2u) + point_index;
                                    sums[vector_local] = sums[vector_local] +
                                        strengths[strength_index] * weight;
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
        let cell_index = cell.x + FINE_0 * cell.y;
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
        tile_side = TILED_GATHER_TILE_SIDE,
        tile_count_0 = tile_count_0,
        tile_count = tile_count,
        width = width,
        half_width = format_wgsl_f32(kernel.half_width() as f32),
        beta = format_wgsl_f32(kernel.beta() as f32),
        bin_radius = bin_radius,
        halo_side = halo_side,
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
    let mode_count = mode_shape[0] * mode_shape[1];
    let centered = matches!(config.mode_order(), crate::config::ModeOrder::Centered);
    format!(
        r#"const WORKGROUP_SIZE: u32 = {WORKGROUP_SIZE}u;
const MODE_0: u32 = {mode_0}u;
const MODE_1: u32 = {mode_1}u;
const MODE_COUNT: u32 = {mode_count}u;
const FINE_0: u32 = {fine_0}u;
const FINE_1: u32 = {fine_1}u;
const FINE_COUNT: u32 = {fine_count}u;
const CENTERED_ORDER: bool = {centered};

@group(0) @binding(0) var<storage, read> fine_grid: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read> amplitudes: array<f32>;
@group(0) @binding(2) var<storage, read_write> output_values: array<vec2<f32>>;

fn fine_axis_index(
    storage_index: u32,
    mode_length: u32,
    fine_length: u32,
) -> u32 {{
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
    let storage_1 = output_index / MODE_0;
    let fine_0 = fine_axis_index(storage_0, MODE_0, FINE_0);
    let fine_1 = fine_axis_index(storage_1, MODE_1, FINE_1);
    let fine_index = fine_0 + FINE_0 * fine_1;
    output_values[linear_index] = fine_grid[vector_index * FINE_COUNT + fine_index] *
        amplitudes[output_index];
}}
"#,
        mode_0 = mode_shape[0],
        mode_1 = mode_shape[1],
        fine_0 = fine_shape[0],
        fine_1 = fine_shape[1],
        fine_count = fine_shape[0] * fine_shape[1],
    )
}

fn generate_deconvolution_wgsl_for_precision(
    config: &NufftConfig,
    fine_shape: [usize; DIMENSIONS],
    precision: FftPrecision,
) -> String {
    match precision {
        FftPrecision::F64 => generate_deconvolution_wgsl_f64(config, fine_shape),
        FftPrecision::Df64 => generate_deconvolution_wgsl_df64(config, fine_shape),
        FftPrecision::F32 => generate_deconvolution_wgsl(config, fine_shape),
    }
}

fn generate_deconvolution_wgsl_f64(
    config: &NufftConfig,
    fine_shape: [usize; DIMENSIONS],
) -> String {
    let mode_shape = config.n_modes();
    let mode_count = mode_shape[0] * mode_shape[1];
    let centered = matches!(config.mode_order(), crate::config::ModeOrder::Centered);
    format!(
        r#"const WORKGROUP_SIZE: u32 = {WORKGROUP_SIZE}u;
const MODE_0: u32 = {mode_0}u;
const MODE_1: u32 = {mode_1}u;
const MODE_COUNT: u32 = {mode_count}u;
const FINE_0: u32 = {fine_0}u;
const FINE_1: u32 = {fine_1}u;
const FINE_COUNT: u32 = {fine_count}u;
const CENTERED_ORDER: bool = {centered};

@group(0) @binding(0) var<storage, read> fine_grid: array<vec2<f64>>;
@group(0) @binding(1) var<storage, read> amplitudes: array<f64>;
@group(0) @binding(2) var<storage, read_write> output_values: array<vec2<f64>>;

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
    let storage_1 = output_index / MODE_0;
    let fine_0 = fine_axis_index(storage_0, MODE_0, FINE_0);
    let fine_1 = fine_axis_index(storage_1, MODE_1, FINE_1);
    let fine_index = fine_0 + FINE_0 * fine_1;
    output_values[linear_index] = fine_grid[vector_index * FINE_COUNT + fine_index] *
        amplitudes[output_index];
}}
"#,
        mode_0 = mode_shape[0],
        mode_1 = mode_shape[1],
        fine_0 = fine_shape[0],
        fine_1 = fine_shape[1],
        fine_count = fine_shape[0] * fine_shape[1],
    )
}

fn generate_deconvolution_wgsl_df64(
    config: &NufftConfig,
    fine_shape: [usize; DIMENSIONS],
) -> String {
    let mode_shape = config.n_modes();
    let mode_count = mode_shape[0] * mode_shape[1];
    let centered = matches!(config.mode_order(), crate::config::ModeOrder::Centered);
    let entry = format!(
        r#"const WORKGROUP_SIZE: u32 = {WORKGROUP_SIZE}u;
const MODE_0: u32 = {mode_0}u;
const MODE_1: u32 = {mode_1}u;
const MODE_COUNT: u32 = {mode_count}u;
const FINE_0: u32 = {fine_0}u;
const FINE_1: u32 = {fine_1}u;
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
    let storage_1 = output_index / MODE_0;
    let fine_0 = fine_axis_index(storage_0, MODE_0, FINE_0);
    let fine_1 = fine_axis_index(storage_1, MODE_1, FINE_1);
    let fine_index = fine_0 + FINE_0 * fine_1;
    output_values[linear_index] = df64_complex_scale(
        fine_grid[vector_index * FINE_COUNT + fine_index],
        amplitudes[output_index],
    );
}}
"#,
        mode_0 = mode_shape[0],
        mode_1 = mode_shape[1],
        fine_0 = fine_shape[0],
        fine_1 = fine_shape[1],
        fine_count = fine_shape[0] * fine_shape[1],
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

fn tiled_gather_halo_side(width: usize) -> Result<usize> {
    let radius = width.div_ceil(2);
    let halo_side = TILED_GATHER_TILE_SIDE
        .checked_add(radius.checked_mul(2).ok_or(NufftError::LengthOverflow {
            context: "type-1 2D tiled gather halo radius",
        })?)
        .and_then(|value| value.checked_sub(1))
        .ok_or(NufftError::LengthOverflow {
            context: "type-1 2D tiled gather halo side",
        })?;
    let halo_entries = halo_side
        .checked_mul(halo_side)
        .and_then(|bins| bins.checked_add(1))
        .ok_or(NufftError::LengthOverflow {
            context: "type-1 2D tiled gather halo prefix",
        })?;
    if halo_entries > TILED_GATHER_SCAN_ELEMENTS {
        return Err(NufftError::LengthOverflow {
            context: "type-1 2D tiled gather scan capacity",
        });
    }
    Ok(halo_side)
}

fn tiled_gather_cache_capacity(width: usize) -> Result<usize> {
    let prefix_bytes = TILED_GATHER_PADDED_SCAN_ELEMENTS
        .checked_mul(size_of::<u32>())
        .ok_or(NufftError::LengthOverflow {
            context: "type-1 2D tiled gather prefix storage",
        })?;
    let weight_bytes = width
        .checked_mul(2)
        .and_then(|weights| weights.checked_mul(size_of::<f32>()))
        .ok_or(NufftError::LengthOverflow {
            context: "type-1 2D tiled gather per-point weight storage",
        })?;
    let per_point_bytes = size_of::<[i32; 2]>()
        .checked_add(size_of::<u32>())
        .and_then(|bytes| bytes.checked_add(weight_bytes))
        .ok_or(NufftError::LengthOverflow {
            context: "type-1 2D tiled gather per-point storage",
        })?;
    let available = TILED_GATHER_STORAGE_BUDGET_BYTES
        .checked_sub(prefix_bytes)
        .ok_or(NufftError::LengthOverflow {
            context: "type-1 2D tiled gather storage budget",
        })?;
    let capacity = available / per_point_bytes;
    if capacity == 0 {
        return Err(NufftError::LengthOverflow {
            context: "type-1 2D tiled gather cache capacity",
        });
    }
    Ok(capacity)
}

fn tiled_gather_storage_bytes(width: usize, cache_capacity: usize) -> Result<u32> {
    let prefix_bytes = TILED_GATHER_PADDED_SCAN_ELEMENTS
        .checked_mul(size_of::<u32>())
        .ok_or(NufftError::LengthOverflow {
            context: "type-1 2D tiled gather prefix storage bytes",
        })?;
    let per_point_bytes = width
        .checked_mul(2)
        .and_then(|weights| weights.checked_mul(size_of::<f32>()))
        .and_then(|weights| weights.checked_add(size_of::<[i32; 2]>()))
        .and_then(|bytes| bytes.checked_add(size_of::<u32>()))
        .ok_or(NufftError::LengthOverflow {
            context: "type-1 2D tiled gather point-cache storage bytes",
        })?;
    let bytes = cache_capacity
        .checked_mul(per_point_bytes)
        .and_then(|cache| cache.checked_add(prefix_bytes))
        .ok_or(NufftError::LengthOverflow {
            context: "type-1 2D tiled gather workgroup storage bytes",
        })?;
    u32::try_from(bytes).map_err(|_| NufftError::LengthOverflow {
        context: "type-1 2D tiled gather workgroup storage size",
    })
}

fn tiled_gather_tile_counts(fine_shape: [usize; DIMENSIONS]) -> Result<(u32, u32, u32)> {
    let tiles_0 = fine_shape[0].div_ceil(TILED_GATHER_TILE_SIDE);
    let tiles_1 = fine_shape[1].div_ceil(TILED_GATHER_TILE_SIDE);
    let tile_count = tiles_0
        .checked_mul(tiles_1)
        .ok_or(NufftError::LengthOverflow {
            context: "type-1 2D tiled gather tile count",
        })?;
    let tiles_0 = u32::try_from(tiles_0).map_err(|_| NufftError::LengthOverflow {
        context: "type-1 2D tiled gather axis-0 tile count",
    })?;
    let tiles_1 = u32::try_from(tiles_1).map_err(|_| NufftError::LengthOverflow {
        context: "type-1 2D tiled gather axis-1 tile count",
    })?;
    let tile_count = u32::try_from(tile_count).map_err(|_| NufftError::LengthOverflow {
        context: "type-1 2D tiled gather shader tile count",
    })?;
    Ok((tiles_0, tiles_1, tile_count))
}

#[cfg(test)]
fn tiled_gather_dispatch(
    fine_shape: [usize; DIMENSIONS],
    max_workgroups_per_dimension: u32,
) -> Result<(u32, u32, u32)> {
    let (_, _, tile_count) = tiled_gather_tile_counts(fine_shape)?;
    split_workgroups(tile_count, max_workgroups_per_dimension)
}

fn tiled_gather_workgroups(fine_shape: [usize; DIMENSIONS]) -> Result<u32> {
    let (_, _, tile_count) = tiled_gather_tile_counts(fine_shape)?;
    Ok(tile_count)
}

fn workgroups_for_elements(element_count: usize) -> Result<u32> {
    let elements = u32::try_from(element_count).map_err(|_| NufftError::LengthOverflow {
        context: "type-1 2D GPU dispatch element count",
    })?;
    Ok(elements.div_ceil(WORKGROUP_SIZE))
}

fn dispatch_for_elements(
    element_count: usize,
    max_workgroups_per_dimension: u32,
) -> Result<(u32, u32, u32)> {
    let elements = u32::try_from(element_count).map_err(|_| NufftError::LengthOverflow {
        context: "type-1 2D GPU dispatch element count",
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
    let coordinate_count =
        point_count
            .checked_mul(DIMENSIONS)
            .ok_or(NufftError::LengthOverflow {
                context: "type-1 2D point-coordinate count",
            })?;
    u32::try_from(coordinate_count)
        .map(|_| ())
        .map_err(|_| NufftError::LengthOverflow {
            context: "type-1 2D point-coordinate shader index space",
        })
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
    fn count_uses_point_major_coordinates_and_axis_zero_fast_bins() {
        let source = generate_count_wgsl([64, 96]);
        assert!(source.contains("let total = arrayLength(&points) / 2u;"));
        assert!(source.contains("let point_base = point_index * 2u;"));
        assert!(source.contains("let bin = bin_0 + FINE_0 * bin_1;"));
        assert!(source.contains("const FINE_COUNT: u32 = 6144u;"));
    }

    #[test]
    fn position_folding_preserves_df64_low_words_on_both_axes() {
        let source = generate_position_wgsl([65_536, 98_304]);
        assert!(source.contains("const POSITION_SCALE_0_LO: f32"));
        assert!(source.contains("const POSITION_SCALE_1_LO: f32"));
        assert!(source.contains("fn floor_df64_to_i32(value: Df64) -> i32"));
        assert!(source.contains("has_negative_remainder"));
    }

    #[test]
    fn gather_is_one_writer_with_tensor_product_weights() {
        let kernel = EsKernel::for_tolerance(1.0e-6, 2.0).unwrap();
        let source = generate_gather_wgsl(kernel, [64, 96]);
        assert!(source.contains("const BIN_RADIUS: i32 = 4i;"));
        assert!(source.contains("for (var bin_offset_1 = -BIN_RADIUS"));
        assert!(source.contains("for (var bin_offset_0 = -BIN_RADIUS"));
        assert!(source.contains("es_weight(distance_0.hi + distance_0.lo) *"));
        assert!(source.contains("es_weight(distance_1.hi + distance_1.lo)"));
        assert!(
            source.contains("fine_grid[vector_index * FINE_COUNT + cell] = sums[vector_local];")
        );
        assert!(!source.contains("fine_grid[cell] = fine_grid[cell] + sum"));
    }

    #[test]
    fn native_f64_shaders_use_horner_without_transcendentals() {
        let kernel = EsKernel::for_tolerance(1.0e-6, 2.0).unwrap();
        let config = NufftConfig::new([17, 19], 1.0e-6).with_precision(FftPrecision::F64);
        let source = [
            generate_count_wgsl_for_precision([36, 40], FftPrecision::F64),
            generate_scatter_wgsl_for_precision([36, 40], FftPrecision::F64),
            generate_gather_wgsl_f64(kernel, [36, 40]),
            generate_deconvolution_wgsl_for_precision(&config, [36, 40], FftPrecision::F64),
        ]
        .join("\n");
        assert!(source.contains("array<vec2<f64>>"));
        assert!(source.contains("HORNER_COEFFICIENTS"));
        for forbidden in ["exp(", "log(", "pow(", "sin(", "cos("] {
            assert!(!source.contains(forbidden), "found {forbidden} in f64 WGSL");
        }
    }

    #[test]
    fn tiled_gather_width_seven_storage_stays_below_fifteen_kibibytes() {
        let capacity = tiled_gather_cache_capacity(7).unwrap();
        assert_eq!(capacity, 163);
        assert_eq!(TILED_GATHER_PADDED_SCAN_ELEMENTS, 1_056);
        assert_eq!(tiled_gather_storage_bytes(7, capacity).unwrap(), 15_308);
        assert!(
            tiled_gather_storage_bytes(7, capacity).unwrap()
                <= TILED_GATHER_STORAGE_BUDGET_BYTES as u32
        );
    }

    #[test]
    fn tiled_gather_batches_hoisted_weights_behind_uniform_barriers() {
        let kernel = EsKernel::for_tolerance(1.0e-6, 2.0).unwrap();
        let capacity = tiled_gather_cache_capacity(kernel.width()).unwrap();
        let source = generate_tiled_gather_wgsl(kernel, [64, 96], capacity).unwrap();
        assert!(source.contains("@compute @workgroup_size(256)"));
        assert!(source.contains("const HALO_SIDE: u32 = 23u;"));
        assert!(source.contains("const HALO_BIN_COUNT: u32 = 529u;"));
        assert!(source.contains("var<workgroup> halo_prefix: array<u32, 1056>;"));
        assert!(source.contains("var<workgroup> cached_starts: array<vec2<i32>, 163>;"));
        assert!(source.contains("var<workgroup> cached_point_indices: array<u32, 163>;"));
        assert!(source.contains("var<workgroup> cached_weights_0: array<f32, 1141>;"));
        assert!(source.contains("let halo_total = workgroupUniformLoad("));
        assert!(source.contains("if (batch_start >= halo_total) { break; }"));
        assert!(
            source.contains("let batch_length = min(CACHE_CAPACITY, halo_total - batch_start);")
        );
        assert!(source.contains("cached_weights_0[weight_slot] ="));
        assert!(source.contains("cached_weights_1[weight_slot] ="));
        assert!(source.contains("for (var bin_offset_1 = -BIN_RADIUS"));
        assert!(source.contains("for (var bin_offset_0 = -BIN_RADIUS"));
        assert!(source
            .contains("fine_grid[vector_index * FINE_COUNT + cell_index] = sums[vector_local];"));
        assert!(!source.contains("atomicAdd"));
    }

    #[test]
    fn tiled_gather_dispatches_one_workgroup_per_sixteen_by_sixteen_tile() {
        assert_eq!(tiled_gather_halo_side(7).unwrap(), 23);
        assert_eq!(
            tiled_gather_tile_counts([2_048, 2_048]).unwrap(),
            (128, 128, 16_384)
        );
        assert_eq!(
            tiled_gather_dispatch([2_048, 2_048], 65_535).unwrap(),
            (16_384, 1, 1)
        );
        assert_eq!(tiled_gather_tile_counts([30, 36]).unwrap(), (2, 3, 6));
    }

    #[test]
    fn each_flattened_bin_is_sorted_by_original_point_index() {
        let source = generate_sort_wgsl();
        assert!(source.contains("fn heap_sort(start: u32, end: u32)"));
        assert!(source.contains("sorted_indices[start + greatest] <"));
        assert!(source.contains("heap_sort(bin_offsets[bin], bin_offsets[bin + 1u]);"));
    }

    #[test]
    fn terminal_offset_extends_the_total_bin_scan() {
        let source = generate_terminal_wgsl();
        assert!(source.contains("bin_offsets[bin_count - 1u] + bin_counts[bin_count - 1u]"));
        assert!(!source.contains("for ("));
        assert!(!source.contains("atomic"));
    }

    #[test]
    fn deconvolution_maps_axes_independently_in_both_mode_orders() {
        for order in [ModeOrder::Centered, ModeOrder::Fft] {
            let config = NufftConfig::new([17, 19], 1.0e-6).with_mode_order(order);
            let source = generate_deconvolution_wgsl(&config, [36, 40]);
            assert!(source.contains("const MODE_0: u32 = 17u;"));
            assert!(source.contains("const MODE_1: u32 = 19u;"));
            assert!(source.contains("let storage_0 = output_index % MODE_0;"));
            assert!(source.contains("let storage_1 = output_index / MODE_0;"));
            assert!(source.contains("let fine_index = fine_0 + FINE_0 * fine_1;"));
        }
    }

    #[test]
    fn amplitudes_are_tensor_products_in_axis_zero_fast_order() {
        let config = NufftConfig::new([3, 5], 1.0e-6);
        let coefficients_0 = [2.0, 4.0];
        let coefficients_1 = [5.0, 10.0, 20.0];
        let values = mode_amplitudes(&config, [&coefficients_0, &coefficients_1]).unwrap();
        assert_eq!(values.len(), 15);
        // Centered storage index (0,0) is mode (-1,-2).
        assert_eq!(values[0].to_bits(), (1.0f32 / 80.0).to_bits());
        // Centered storage index (1,2) is mode (0,0), linear index 1+3*2.
        assert_eq!(values[7].to_bits(), (1.0f32 / 10.0).to_bits());
    }

    #[test]
    fn buffer_sizes_are_exact_point_major_prefixes() {
        assert_eq!(
            Type1GpuPlan2d::point_buffer_size_bytes_for_precision(7, FftPrecision::F32).unwrap(),
            56
        );
        assert_eq!(
            Type1GpuPlan2d::strength_buffer_size_bytes_for_precision(7, FftPrecision::F32).unwrap(),
            56
        );
        assert_eq!(
            Type1GpuPlan2d::complex_buffer_size_bytes_for_precision("test", 11, FftPrecision::F32,)
                .unwrap(),
            88
        );
    }

    #[test]
    fn point_coordinate_shader_index_checks_the_aos_scalar_count() {
        let maximum_points = u32::MAX as usize / DIMENSIONS;
        assert_eq!(
            validate_point_coordinate_shader_index(maximum_points),
            Ok(())
        );
        assert_eq!(
            validate_point_coordinate_shader_index(maximum_points + 1),
            Err(NufftError::LengthOverflow {
                context: "type-1 2D point-coordinate shader index space"
            })
        );
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn gpu_pipeline_matches_the_direct_oracle_when_requested() {
        if std::env::var_os("WGPU_FFT_RUN_GPU_TESTS").is_none() {
            return;
        }
        pollster::block_on(async {
            let Some(context) = wgpu_fft::device::request_default_device().await else {
                panic!("GPU tests were requested but no suitable adapter was found");
            };
            let config = NufftConfig::new([8, 10], 1.0e-6);
            let kernel = EsKernel::for_tolerance(config.eps(), config.sigma()).unwrap();
            let fine_shape = [
                crate::kernel::select_fine_grid_size(8, config.sigma(), kernel.width()).unwrap(),
                crate::kernel::select_fine_grid_size(10, config.sigma(), kernel.width()).unwrap(),
            ];
            let coefficients_0 = kernel.centered_fourier_coefficients(fine_shape[0]).unwrap();
            let coefficients_1 = kernel.centered_fourier_coefficients(fine_shape[1]).unwrap();
            let scope = context
                .device
                .push_error_scope(wgpu::ErrorFilter::Validation);
            let plan = Type1GpuPlan2d::new(
                &context.device,
                &context.queue,
                &config,
                kernel,
                fine_shape,
                [&coefficients_0, &coefficients_1],
                Type1Gather2d::Global,
            )
            .unwrap();

            let points = [-2.9f32, 2.8, -0.25, 0.75, 0.0, 0.0, 1.2, -1.7, 1.2, -1.7];
            let strengths = [0.75f32, -0.5, -0.25, 0.125, 1.0, 0.25, -0.6, 0.4, 0.2, -0.8];
            let points_buffer =
                context
                    .device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("wgpu_nufft.type1_2d.test.points"),
                        contents: bytemuck::cast_slice(&points),
                        usage: wgpu::BufferUsages::STORAGE,
                    });
            let strengths_buffer =
                context
                    .device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("wgpu_nufft.type1_2d.test.strengths"),
                        contents: bytemuck::cast_slice(&strengths),
                        usage: wgpu::BufferUsages::STORAGE,
                    });
            let output_bytes = config.mode_count().unwrap() as u64 * COMPLEX_F32_BYTES;
            let output = context.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("wgpu_nufft.type1_2d.test.output"),
                size: output_bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let readback = context.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("wgpu_nufft.type1_2d.test.readback"),
                size: output_bytes,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            let mut encoder =
                context
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("wgpu_nufft.type1_2d.test.encoder"),
                    });
            plan.encode(
                &context.device,
                &mut encoder,
                points.len() / DIMENSIONS,
                &points_buffer,
                &strengths_buffer,
                &output,
            )
            .unwrap();
            encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, output_bytes);
            context.queue.submit([encoder.finish()]);
            let slice = readback.slice(..);
            let (sender, receiver) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |result| {
                sender.send(result).unwrap();
            });
            context
                .device
                .poll(wgpu::PollType::wait_indefinitely())
                .unwrap();
            receiver.recv().unwrap().unwrap();
            let mapped = slice.get_mapped_range();
            let actual = bytemuck::cast_slice::<u8, f32>(&mapped);
            let coordinates = points
                .iter()
                .map(|&value| f64::from(value))
                .collect::<Vec<_>>();
            let strengths_f64 = strengths
                .chunks_exact(2)
                .map(|value| crate::Complex64::new(f64::from(value[0]), f64::from(value[1])))
                .collect::<Vec<_>>();
            let reference =
                crate::direct::reference_type1_f64(&config, &coordinates, &strengths_f64).unwrap();
            let (error_squared, norm_squared) = actual.chunks_exact(2).zip(&reference).fold(
                (0.0, 0.0),
                |(error, norm), (value, reference)| {
                    let delta_re = f64::from(value[0]) - reference.re;
                    let delta_im = f64::from(value[1]) - reference.im;
                    (
                        error + delta_re * delta_re + delta_im * delta_im,
                        norm + reference.re * reference.re + reference.im * reference.im,
                    )
                },
            );
            let relative_l2 = (error_squared / norm_squared.max(f64::MIN_POSITIVE)).sqrt();
            assert!(
                relative_l2 <= 8.0 * config.eps(),
                "2D type-1 relative L2 {relative_l2} exceeds {}",
                8.0 * config.eps()
            );
            drop(mapped);
            readback.unmap();
            if let Some(error) = scope.pop().await {
                panic!("2D type-1 pipeline validation failed: {error}");
            }
            #[cfg(windows)]
            std::mem::forget((plan, context));
            #[cfg(not(windows))]
            drop((plan, context));
        });
    }
}
