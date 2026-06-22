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

const WORKGROUP_SIZE: u32 = 64;
const VECTOR_BLOCK_SIZE: usize = 4;
const COMPLEX_F32_BYTES: u64 = 8;
const COMPLEX_F64_BYTES: u64 = 16;
const F32_BYTES: u64 = 4;
const F64_BYTES: u64 = 8;
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
    max_workgroups_per_dimension: u32,
    mode_count: usize,
    fine_length: usize,
    batch_capacity: usize,
    precision: FftPrecision,
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
        // Support indices and bin keys must remain exact in the selected
        // position-arithmetic precision. Reserve the support halo beyond the
        // last bin and retain the f32 bound for the portable paths.
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
        let batch_capacity = config.batch();
        let fine_element_count =
            fine_length
                .checked_mul(batch_capacity)
                .ok_or(NufftError::LengthOverflow {
                    context: "batched type-1 fine-grid element count",
                })?;
        let fine_bytes = Self::complex_buffer_size_bytes_for_precision(
            "type-1 fine grid",
            fine_element_count,
            precision,
        )?;
        let amplitude_bytes = checked_buffer_size(
            "type-1 deconvolution amplitudes",
            mode_count,
            scalar_size_bytes(precision),
        )?;
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

        let amplitudes = mode_amplitude_bytes(config, kernel_fourier_coefficients, precision)?;
        let amplitudes = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("wgpu_nufft.type1.deconvolution_amplitudes"),
            contents: &amplitudes,
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
        let scratch =
            Type1ScratchBuffers::new(device, "wgpu_nufft.type1", count_bytes, offset_bytes);

        let fft_direction = match config.sign() {
            NufftSign::Positive => FftDirection::Inverse,
            NufftSign::Negative => FftDirection::Forward,
        };
        let fft = FftPlan::c2c(
            device,
            queue,
            FftConfig::new(fine_length)
                .with_batch(batch_capacity)
                .with_direction(fft_direction)
                .with_normalization(Normalization::None)
                .with_precision(precision),
        )
        .map_err(|source| NufftError::FftShapeUnsupported {
            stage: "type-1 oversampled-grid C2C plan",
            source,
        })?;

        let count_pipeline = create_compute_pipeline(
            device,
            "wgpu_nufft.type1.bin_count",
            &generate_count_wgsl_for_precision(fine_length, precision),
        );
        let count_layout = count_pipeline.get_bind_group_layout(0);
        let prefix_scan = GpuExclusiveScanU32::new(device, fine_length)?;
        let terminal_pipeline = create_compute_pipeline(
            device,
            "wgpu_nufft.type1.bin_terminal",
            &generate_terminal_wgsl(),
        );
        let terminal_layout = terminal_pipeline.get_bind_group_layout(0);
        let scatter_pipeline = create_compute_pipeline(
            device,
            "wgpu_nufft.type1.bin_scatter",
            &generate_scatter_wgsl_for_precision(fine_length, precision),
        );
        let scatter_layout = scatter_pipeline.get_bind_group_layout(0);
        let sort_pipeline =
            create_compute_pipeline(device, "wgpu_nufft.type1.bin_sort", &generate_sort_wgsl());
        let sort_layout = sort_pipeline.get_bind_group_layout(0);
        let gather_pipeline = create_compute_pipeline(
            device,
            "wgpu_nufft.type1.spread_gather",
            &generate_gather_wgsl_for_precision(kernel, fine_length, precision),
        );
        let gather_layout = gather_pipeline.get_bind_group_layout(0);
        let deconvolution_pipeline = create_compute_pipeline(
            device,
            "wgpu_nufft.type1.deconvolution",
            &generate_deconvolution_wgsl_for_precision(config, fine_length, precision),
        );
        let deconvolution_layout = deconvolution_pipeline.get_bind_group_layout(0);

        let max_workgroups_per_dimension = limits.max_compute_workgroups_per_dimension;
        let sort_dispatch = dispatch_for_elements(fine_length, max_workgroups_per_dimension)?;

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
            max_workgroups_per_dimension,
            mode_count,
            fine_length,
            batch_capacity,
            precision,
            max_storage_binding_bytes: limits.max_storage_buffer_binding_size,
            max_buffer_bytes: limits.max_buffer_size,
        })
    }

    pub(crate) fn point_buffer_size_bytes(point_count: usize) -> Result<u64> {
        checked_buffer_size("type-1 point buffer", point_count, F32_BYTES)
    }

    pub(crate) fn point_buffer_size_bytes_for_precision(
        point_count: usize,
        precision: FftPrecision,
    ) -> Result<u64> {
        checked_buffer_size(
            "type-1 point buffer",
            point_count,
            scalar_size_bytes(precision),
        )
    }

    pub(crate) fn strength_buffer_size_bytes(point_count: usize) -> Result<u64> {
        Self::complex_buffer_size_bytes("type-1 strength buffer", point_count)
    }

    pub(crate) fn strength_buffer_size_bytes_for_precision(
        point_count: usize,
        precision: FftPrecision,
    ) -> Result<u64> {
        Self::complex_buffer_size_bytes_for_precision(
            "type-1 strength buffer",
            point_count,
            precision,
        )
    }

    pub(crate) fn complex_buffer_size_bytes(
        buffer: &'static str,
        element_count: usize,
    ) -> Result<u64> {
        checked_buffer_size(buffer, element_count, COMPLEX_F32_BYTES)
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
        self.fine_length
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
                kind: "type-1 stage profiling",
                reason: "at least one point is required",
            });
        }
        let layout =
            NufftGpuProfileLayout::type1(first_query).map_err(|_| NufftError::LengthOverflow {
                context: "type-1 stage-profile query range",
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
                    context: "batched type-1 Fourier output element count",
                })?;
        let output_bytes = Self::complex_buffer_size_bytes_for_precision(
            "type-1 Fourier output buffer",
            output_elements,
            self.precision,
        )?;
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
            // Type-1 with no sources has a nonempty, all-zero mode output. Do
            // not leave stale plan scratch or return early as type-2 may.
            encoder.clear_buffer(self.fine_grid_buffer(), 0, None);
            return Ok(());
        }

        let point_count_u32 =
            u32::try_from(point_count).map_err(|_| NufftError::LengthOverflow {
                context: "type-1 GPU point count",
            })?;
        let point_bytes = Self::point_buffer_size_bytes_for_precision(point_count, self.precision)?;
        let strength_elements =
            point_count
                .checked_mul(active_batch)
                .ok_or(NufftError::LengthOverflow {
                    context: "batched type-1 strength element count",
                })?;
        let strength_bytes =
            Self::strength_buffer_size_bytes_for_precision(strength_elements, self.precision)?;
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

        let fine_grid_element_count = self.fine_grid_element_count();
        let count_bytes =
            checked_buffer_size("type-1 bin counts", fine_grid_element_count, U32_BYTES)?;
        let offset_count =
            self.fine_grid_element_count()
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

        let bin_counts = &self.scratch.bin_counts;
        let bin_cursors = &self.scratch.bin_cursors;
        let bin_offsets = &self.scratch.bin_offsets;
        let sorted_indices = self.scratch.sorted_indices(index_bytes);

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
        let terminal_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("wgpu_nufft.type1.bin_terminal.bind_group"),
            layout: &self.terminal_layout,
            entries: &[
                binding_entry(0, bin_counts, count_bytes),
                binding_entry(1, bin_offsets, offset_bytes),
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
                binding_entry(3, &sorted_indices, index_bytes),
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
                binding_entry(1, &sorted_indices, index_bytes),
            ],
        });
        let active_fine_elements = fine_grid_element_count.checked_mul(active_batch).ok_or(
            NufftError::LengthOverflow {
                context: "active batched type-1 fine-grid element count",
            },
        )?;
        let active_fine_bytes = Self::complex_buffer_size_bytes_for_precision(
            "active batched type-1 fine grid",
            active_fine_elements,
            self.precision,
        )?;
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
                binding_entry(3, &sorted_indices, index_bytes),
                binding_entry(4, &self.fine_input, active_fine_bytes),
            ],
        });
        let vector_blocks = active_batch.div_ceil(VECTOR_BLOCK_SIZE);
        let gather_elements = fine_grid_element_count.checked_mul(vector_blocks).ok_or(
            NufftError::LengthOverflow {
                context: "batched type-1 gather element count",
            },
        )?;
        let gather_dispatch =
            dispatch_for_elements(gather_elements, self.max_workgroups_per_dimension)?;

        #[cfg(feature = "gpu-profiling")]
        profile.encode_type1_start_marker(encoder);
        encoder.clear_buffer(bin_counts, 0, None);
        encoder.clear_buffer(bin_cursors, 0, None);
        encode_pass(
            encoder,
            "wgpu_nufft.type1.bin_count.pass",
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
            "wgpu_nufft.type1.bin_terminal.pass",
            &self.terminal_pipeline,
            &terminal_bind_group,
            (1, 1, 1),
            #[cfg(feature = "gpu-profiling")]
            profile.timestamp_writes(None, Some(2)),
        );
        encode_pass(
            encoder,
            "wgpu_nufft.type1.bin_scatter.pass",
            &self.scatter_pipeline,
            &scatter_bind_group,
            point_dispatch,
            #[cfg(feature = "gpu-profiling")]
            profile.timestamp_writes(None, Some(3)),
        );
        encode_pass(
            encoder,
            "wgpu_nufft.type1.bin_sort.pass",
            &self.sort_pipeline,
            &sort_bind_group,
            self.sort_dispatch,
            #[cfg(feature = "gpu-profiling")]
            profile.timestamp_writes(None, Some(4)),
        );
        encode_pass(
            encoder,
            "wgpu_nufft.type1.spread_gather.pass",
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
                stage: "type-1 oversampled-grid C2C transform",
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
            "wgpu_nufft.type1.deconvolution.pass",
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

fn mode_amplitudes_f64(config: &NufftConfig, coefficients: &[f64]) -> Result<Vec<f64>> {
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
        let amplitude = 1.0 / coefficient;
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

fn mode_amplitude_bytes(
    config: &NufftConfig,
    coefficients: &[f64],
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

fn generate_position_wgsl_df64(fine_length: usize) -> String {
    generate_position_wgsl(fine_length)
        .replace(
            "fn fold_position(point: f32) -> Df64",
            "fn fold_position(point: Df64) -> Df64",
        )
        .replace("Df64(point, 0.0),", "point,")
}

fn generate_position_wgsl_f64(fine_length: usize) -> String {
    let position_scale = fine_length as f64 / std::f64::consts::TAU;
    format!(
        r#"const FINE_LENGTH: u32 = {fine_length}u;
const FINE_LENGTH_I32: i32 = {fine_length}i;
const FINE_LENGTH_F64: f64 = {fine_length_f64};
const POSITION_SCALE: f64 = {position_scale};
const GRID_ORIGIN: f64 = {grid_origin};

fn fold_position(point: f64) -> f64 {{
    var position = point * POSITION_SCALE + GRID_ORIGIN;
    if (position < 0.0lf) {{ position = position + FINE_LENGTH_F64; }}
    if (position < 0.0lf) {{ position = position + FINE_LENGTH_F64; }}
    if (position >= FINE_LENGTH_F64) {{ position = position - FINE_LENGTH_F64; }}
    if (position >= FINE_LENGTH_F64) {{ position = position - FINE_LENGTH_F64; }}
    return position;
}}
"#,
        fine_length_f64 = format_wgsl_f64(fine_length as f64),
        position_scale = format_wgsl_f64(position_scale),
        grid_origin = format_wgsl_f64((fine_length / 2) as f64),
    )
}

fn generate_count_wgsl_for_precision(fine_length: usize, precision: FftPrecision) -> String {
    match precision {
        FftPrecision::F64 => generate_count_wgsl_f64(fine_length),
        FftPrecision::Df64 => generate_count_wgsl_df64(fine_length),
        FftPrecision::F32 => generate_count_wgsl(fine_length),
    }
}

fn generate_count_wgsl_df64(fine_length: usize) -> String {
    let position = generate_position_wgsl_df64(fine_length);
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

fn generate_count_wgsl_f64(fine_length: usize) -> String {
    let position = generate_position_wgsl_f64(fine_length);
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
    let total = arrayLength(&points);
    let wg_flat = (wid.z * nwg.y + wid.y) * nwg.x + wid.x;
    if (wg_flat > (total - 1u) / WORKGROUP_SIZE) {{ return; }}
    let point_index = wg_flat * WORKGROUP_SIZE + lid.x;
    if (point_index >= total) {{ return; }}

    let position = fold_position(points[point_index]);
    atomicAdd(&bin_counts[u32(floor(position))], 1u);
}}
"#,
    );
    format!("{position}\n{entry}")
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

fn generate_scatter_wgsl_for_precision(fine_length: usize, precision: FftPrecision) -> String {
    match precision {
        FftPrecision::F64 => generate_scatter_wgsl_f64(fine_length),
        FftPrecision::Df64 => generate_scatter_wgsl_df64(fine_length),
        FftPrecision::F32 => generate_scatter_wgsl(fine_length),
    }
}

fn generate_scatter_wgsl_df64(fine_length: usize) -> String {
    let position = generate_position_wgsl_df64(fine_length);
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

fn generate_scatter_wgsl_f64(fine_length: usize) -> String {
    let position = generate_position_wgsl_f64(fine_length);
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
    let total = arrayLength(&points);
    let wg_flat = (wid.z * nwg.y + wid.y) * nwg.x + wid.x;
    if (wg_flat > (total - 1u) / WORKGROUP_SIZE) {{ return; }}
    let point_index = wg_flat * WORKGROUP_SIZE + lid.x;
    if (point_index >= total) {{ return; }}

    let bin = u32(floor(fold_position(points[point_index])));
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
    let total_vectors = arrayLength(&fine_grid) / FINE_LENGTH;
    let vector_blocks = (total_vectors + VECTOR_BLOCK_SIZE - 1u) / VECTOR_BLOCK_SIZE;
    let total_work = FINE_LENGTH * vector_blocks;
    if (wg_flat > (total_work - 1u) / WORKGROUP_SIZE) {{ return; }}
    let work_index = wg_flat * WORKGROUP_SIZE + lid.x;
    if (work_index >= total_work) {{ return; }}
    let vector_block = work_index / FINE_LENGTH;
    let cell = work_index - vector_block * FINE_LENGTH;
    let first_vector = vector_block * VECTOR_BLOCK_SIZE;

    var sums: array<vec2<f32>, {VECTOR_BLOCK_SIZE}>;
    for (var vector_local = 0u; vector_local < VECTOR_BLOCK_SIZE;
         vector_local = vector_local + 1u) {{
        sums[vector_local] = vec2<f32>(0.0, 0.0);
    }}
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
                for (var vector_local = 0u; vector_local < VECTOR_BLOCK_SIZE;
                     vector_local = vector_local + 1u) {{
                    let vector_index = first_vector + vector_local;
                    if (vector_index < total_vectors) {{
                        let strength_index = vector_index * arrayLength(&points) + point_index;
                        sums[vector_local] = sums[vector_local] +
                            strengths[strength_index] * weight;
                    }}
                }}
            }}
        }}
    }}
    for (var vector_local = 0u; vector_local < VECTOR_BLOCK_SIZE;
         vector_local = vector_local + 1u) {{
        let vector_index = first_vector + vector_local;
        if (vector_index < total_vectors) {{
            fine_grid[vector_index * FINE_LENGTH + cell] = sums[vector_local];
        }}
    }}
}}
"#,
        half_width = format_wgsl_f32(kernel.half_width() as f32),
        beta = format_wgsl_f32(kernel.beta() as f32),
    );
    format!("{}\n{position}\n{entry}", wgpu_fft::kernels::DF64_WGSL)
}

fn generate_gather_wgsl_for_precision(
    kernel: EsKernel,
    fine_length: usize,
    precision: FftPrecision,
) -> String {
    match precision {
        FftPrecision::F64 => generate_gather_wgsl_f64(kernel, fine_length),
        FftPrecision::Df64 => generate_gather_wgsl_df64(kernel, fine_length),
        FftPrecision::F32 => generate_gather_wgsl(kernel, fine_length),
    }
}

fn generate_gather_wgsl_df64(kernel: EsKernel, fine_length: usize) -> String {
    let position = generate_position_wgsl_df64(fine_length);
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

fn wrap_bin(index: i32) -> u32 {{
    var wrapped = index;
    if (wrapped < 0) {{ wrapped = wrapped + FINE_LENGTH_I32; }}
    if (wrapped >= FINE_LENGTH_I32) {{ wrapped = wrapped - FINE_LENGTH_I32; }}
    return u32(wrapped);
}}

{horner}

@compute @workgroup_size({WORKGROUP_SIZE})
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {{
    let wg_flat = (wid.z * nwg.y + wid.y) * nwg.x + wid.x;
    let total_vectors = arrayLength(&fine_grid) / FINE_LENGTH;
    let vector_blocks = (total_vectors + VECTOR_BLOCK_SIZE - 1u) / VECTOR_BLOCK_SIZE;
    let total_work = FINE_LENGTH * vector_blocks;
    if (wg_flat > (total_work - 1u) / WORKGROUP_SIZE) {{ return; }}
    let work_index = wg_flat * WORKGROUP_SIZE + lid.x;
    if (work_index >= total_work) {{ return; }}
    let vector_block = work_index / FINE_LENGTH;
    let cell = work_index - vector_block * FINE_LENGTH;
    let first_vector = vector_block * VECTOR_BLOCK_SIZE;

    var sums: array<vec4<f32>, {VECTOR_BLOCK_SIZE}>;
    for (var vector_local = 0u; vector_local < VECTOR_BLOCK_SIZE;
         vector_local = vector_local + 1u) {{
        sums[vector_local] = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }}
    for (var bin_offset = -BIN_RADIUS; bin_offset < BIN_RADIUS;
         bin_offset = bin_offset + 1) {{
        let bin = wrap_bin(i32(cell) + bin_offset);
        let begin = bin_offsets[bin];
        let end = bin_offsets[bin + 1u];
        for (var slot = begin; slot < end; slot = slot + 1u) {{
            let point_index = sorted_indices[slot];
            let point_position = fold_position(points[point_index]);
            let shifted = df64_sub(point_position, DF64_HALF_WIDTH);
            let start = ceil_df64_to_i32(shifted);
            var unwrapped = i32(cell);
            if (unwrapped < start) {{ unwrapped = unwrapped + FINE_LENGTH_I32; }}
            if (unwrapped >= start + WIDTH_I32) {{
                unwrapped = unwrapped - FINE_LENGTH_I32;
            }}
            if (unwrapped >= start && unwrapped < start + WIDTH_I32) {{
                let distance = df64_sub(Df64(f32(unwrapped), 0.0), point_position);
                let weight = es_weight(distance);
                for (var vector_local = 0u; vector_local < VECTOR_BLOCK_SIZE;
                     vector_local = vector_local + 1u) {{
                    let vector_index = first_vector + vector_local;
                    if (vector_index < total_vectors) {{
                        let strength_index = vector_index * arrayLength(&points) + point_index;
                        sums[vector_local] = df64_complex_add(
                            sums[vector_local],
                            df64_complex_scale(strengths[strength_index], weight),
                        );
                    }}
                }}
            }}
        }}
    }}
    for (var vector_local = 0u; vector_local < VECTOR_BLOCK_SIZE;
         vector_local = vector_local + 1u) {{
        let vector_index = first_vector + vector_local;
        if (vector_index < total_vectors) {{
            fine_grid[vector_index * FINE_LENGTH + cell] = sums[vector_local];
        }}
    }}
}}
"#,
    );
    format!("{}\n{position}\n{entry}", wgpu_fft::kernels::DF64_WGSL)
}

fn generate_gather_wgsl_f64(kernel: EsKernel, fine_length: usize) -> String {
    let position = generate_position_wgsl_f64(fine_length);
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

fn wrap_bin(index: i32) -> u32 {{
    var wrapped = index;
    if (wrapped < 0) {{ wrapped = wrapped + FINE_LENGTH_I32; }}
    if (wrapped >= FINE_LENGTH_I32) {{ wrapped = wrapped - FINE_LENGTH_I32; }}
    return u32(wrapped);
}}

{horner}

@compute @workgroup_size({WORKGROUP_SIZE})
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {{
    let wg_flat = (wid.z * nwg.y + wid.y) * nwg.x + wid.x;
    let total_vectors = arrayLength(&fine_grid) / FINE_LENGTH;
    let vector_blocks = (total_vectors + VECTOR_BLOCK_SIZE - 1u) / VECTOR_BLOCK_SIZE;
    let total_work = FINE_LENGTH * vector_blocks;
    if (wg_flat > (total_work - 1u) / WORKGROUP_SIZE) {{ return; }}
    let work_index = wg_flat * WORKGROUP_SIZE + lid.x;
    if (work_index >= total_work) {{ return; }}
    let vector_block = work_index / FINE_LENGTH;
    let cell = work_index - vector_block * FINE_LENGTH;
    let first_vector = vector_block * VECTOR_BLOCK_SIZE;

    var sums: array<vec2<f64>, {VECTOR_BLOCK_SIZE}>;
    for (var vector_local = 0u; vector_local < VECTOR_BLOCK_SIZE;
         vector_local = vector_local + 1u) {{
        sums[vector_local] = vec2<f64>(0.0lf, 0.0lf);
    }}
    for (var bin_offset = -BIN_RADIUS; bin_offset < BIN_RADIUS;
         bin_offset = bin_offset + 1) {{
        let bin = wrap_bin(i32(cell) + bin_offset);
        let begin = bin_offsets[bin];
        let end = bin_offsets[bin + 1u];
        for (var slot = begin; slot < end; slot = slot + 1u) {{
            let point_index = sorted_indices[slot];
            let point_position = fold_position(points[point_index]);
            let start = i32(ceil(point_position - HALF_WIDTH));
            var unwrapped = i32(cell);
            if (unwrapped < start) {{ unwrapped = unwrapped + FINE_LENGTH_I32; }}
            if (unwrapped >= start + WIDTH_I32) {{
                unwrapped = unwrapped - FINE_LENGTH_I32;
            }}
            if (unwrapped >= start && unwrapped < start + WIDTH_I32) {{
                let weight = es_weight(f64(unwrapped) - point_position);
                for (var vector_local = 0u; vector_local < VECTOR_BLOCK_SIZE;
                     vector_local = vector_local + 1u) {{
                    let vector_index = first_vector + vector_local;
                    if (vector_index < total_vectors) {{
                        let strength_index = vector_index * arrayLength(&points) + point_index;
                        sums[vector_local] = sums[vector_local] +
                            strengths[strength_index] * weight;
                    }}
                }}
            }}
        }}
    }}
    for (var vector_local = 0u; vector_local < VECTOR_BLOCK_SIZE;
         vector_local = vector_local + 1u) {{
        let vector_index = first_vector + vector_local;
        if (vector_index < total_vectors) {{
            fine_grid[vector_index * FINE_LENGTH + cell] = sums[vector_local];
        }}
    }}
}}
"#,
        half_width = format_wgsl_f64(kernel.half_width()),
    );
    format!("{position}\n{entry}")
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
    let total = arrayLength(&output_values);
    if (wg_flat > (total - 1u) / WORKGROUP_SIZE) {{ return; }}
    let linear_index = wg_flat * WORKGROUP_SIZE + lid.x;
    if (linear_index >= total) {{ return; }}
    let vector_index = linear_index / MODE_COUNT;
    let output_index = linear_index - vector_index * MODE_COUNT;

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
    output_values[linear_index] = fine_grid[vector_index * FINE_LENGTH + fine_index] *
        amplitudes[output_index];
}}
"#,
    )
}

fn generate_deconvolution_wgsl_for_precision(
    config: &NufftConfig,
    fine_length: usize,
    precision: FftPrecision,
) -> String {
    match precision {
        FftPrecision::F64 => generate_deconvolution_wgsl_f64(config, fine_length),
        FftPrecision::Df64 => generate_deconvolution_wgsl_df64(config, fine_length),
        FftPrecision::F32 => generate_deconvolution_wgsl(config, fine_length),
    }
}

fn generate_deconvolution_wgsl_df64(config: &NufftConfig, fine_length: usize) -> String {
    let mode_count = config.n_modes()[0];
    let half = mode_count / 2;
    let nonnegative_count = mode_count.div_ceil(2);
    let centered = matches!(config.mode_order(), crate::config::ModeOrder::Centered);
    let entry = format!(
        r#"const WORKGROUP_SIZE: u32 = {WORKGROUP_SIZE}u;
const MODE_COUNT: u32 = {mode_count}u;
const FINE_LENGTH: u32 = {fine_length}u;
const HALF_MODE_COUNT: u32 = {half}u;
const NONNEGATIVE_COUNT: u32 = {nonnegative_count}u;
const CENTERED_ORDER: bool = {centered};

@group(0) @binding(0) var<storage, read> fine_grid: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> amplitudes: array<Df64>;
@group(0) @binding(2) var<storage, read_write> output_values: array<vec4<f32>>;

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
    output_values[linear_index] = df64_complex_scale(
        fine_grid[vector_index * FINE_LENGTH + fine_index],
        amplitudes[output_index],
    );
}}
"#,
    );
    format!("{}\n{entry}", wgpu_fft::kernels::DF64_WGSL)
}

fn generate_deconvolution_wgsl_f64(config: &NufftConfig, fine_length: usize) -> String {
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

@group(0) @binding(0) var<storage, read> fine_grid: array<vec2<f64>>;
@group(0) @binding(1) var<storage, read> amplitudes: array<f64>;
@group(0) @binding(2) var<storage, read_write> output_values: array<vec2<f64>>;

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
    output_values[linear_index] = fine_grid[vector_index * FINE_LENGTH + fine_index] *
        amplitudes[output_index];
}}
"#,
    )
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
        assert!(
            source.contains("fine_grid[vector_index * FINE_LENGTH + cell] = sums[vector_local];")
        );
        assert!(source
            .contains("let strength_index = vector_index * arrayLength(&points) + point_index;"));
        assert!(!source.contains("fine_grid[cell] = fine_grid[cell] + sum"));
    }

    #[test]
    fn native_f64_shaders_use_horner_without_transcendentals() {
        let kernel = EsKernel::for_tolerance(1.0e-6, 2.0).unwrap();
        let config = NufftConfig::new([17], 1.0e-6).with_precision(FftPrecision::F64);
        let source = [
            generate_count_wgsl_for_precision(64, FftPrecision::F64),
            generate_scatter_wgsl_for_precision(64, FftPrecision::F64),
            generate_gather_wgsl_for_precision(kernel, 64, FftPrecision::F64),
            generate_deconvolution_wgsl_for_precision(&config, 64, FftPrecision::F64),
        ]
        .join("\n");
        assert!(source.contains("array<vec2<f64>>"));
        assert!(source.contains("HORNER_COEFFICIENTS"));
        for forbidden in ["exp(", "log(", "pow(", "sin(", "cos("] {
            assert!(!source.contains(forbidden), "found {forbidden} in f64 WGSL");
        }
    }

    #[test]
    fn each_bin_is_sorted_by_original_point_index() {
        let source = generate_sort_wgsl();
        assert!(source.contains("fn heap_sort(start: u32, end: u32)"));
        assert!(source.contains("sorted_indices[start + greatest] <"));
        assert!(source.contains("heap_sort(bin_offsets[bin], bin_offsets[bin + 1u]);"));
    }

    #[test]
    fn terminal_offset_extends_the_exclusive_scan() {
        let source = generate_terminal_wgsl();
        assert!(source.contains("bin_offsets[bin_count - 1u] + bin_counts[bin_count - 1u]"));
        assert!(!source.contains("for ("));
        assert!(!source.contains("atomic"));
    }

    #[test]
    fn deconvolution_maps_both_mode_orders_and_odd_lengths() {
        for order in [ModeOrder::Centered, ModeOrder::Fft] {
            let config = NufftConfig::new([17], 1.0e-6).with_mode_order(order);
            let source = generate_deconvolution_wgsl(&config, 36);
            assert!(source.contains("const MODE_COUNT: u32 = 17u;"));
            assert!(source.contains("const HALF_MODE_COUNT: u32 = 8u;"));
            assert!(source.contains("const NONNEGATIVE_COUNT: u32 = 9u;"));
            assert!(source.contains("fine_grid[vector_index * FINE_LENGTH + fine_index] *"));
        }
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
