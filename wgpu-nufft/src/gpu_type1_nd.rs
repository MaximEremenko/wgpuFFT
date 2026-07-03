//! Rank-generic (runtime-dimensional) type-1 GPU execution.
//!
//! Adoption phase 1, second half: the 1D type-1 pipeline generalized to any
//! rank. The stage list is identical to `gpu_type1.rs` — bin count (u32
//! atomics) -> exclusive scan -> terminal -> scatter (u32 atomics) -> per-bin
//! heap sort -> deterministic per-cell gather (one writer per fine cell) ->
//! oversampled C2C FFT -> deconvolution/truncation. Bins are fine-grid cells
//! flattened axis-0-fast; the scan, sort, and scratch machinery are already
//! rank-free and are reused as-is. Phase 1 supports `FftPrecision::F32` only.

use wgpu::util::DeviceExt;
use wgpu_fft::{BufferView, FftConfig, FftDirection, FftPlan, FftPrecision, Normalization};

use crate::config::{ModeOrder, NufftConfig, NufftSign};
use crate::error::{NufftError, Result};
use crate::gpu::max_supported_workgroup_size;
use crate::gpu_dispatch::split_workgroups;
use crate::gpu_nd::{
    binding_entry, checked_buffer_size, checked_product, create_compute_pipeline, format_wgsl_f32,
    mode_amplitudes_nd, validate_binding_limit, validate_external_storage_buffer,
};
#[cfg(feature = "gpu-profiling")]
use crate::gpu_profile::NufftGpuProfileLayout;
use crate::gpu_scan::GpuExclusiveScanU32;
use crate::gpu_type1_scratch::Type1ScratchBuffers;
use crate::kernel::EsKernel;

use std::fmt::Write as _;

const WORKGROUP_SIZE: u32 = 64;
const VECTOR_BLOCK_SIZE: usize = 4;
const COMPLEX_F32_BYTES: u64 = 8;
const F32_BYTES: u64 = 4;
const U32_BYTES: u64 = 4;

pub(crate) struct Type1GpuPlanNd {
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
    fine_product: usize,
    dimensions: usize,
    batch_capacity: usize,
    max_storage_binding_bytes: u64,
}

impl Type1GpuPlanNd {
    pub(crate) fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        config: &NufftConfig,
        kernel: EsKernel,
        fine_shape: &[usize],
        kernel_fourier_coefficients: &[Vec<f64>],
    ) -> Result<Self> {
        let dimensions = config.dimensions();
        debug_assert_eq!(fine_shape.len(), dimensions);
        debug_assert_eq!(kernel_fourier_coefficients.len(), dimensions);
        if config.precision() != FftPrecision::F32 {
            return Err(NufftError::PrecisionUnsupported {
                requested: config.precision(),
                stage: "rank-generic type-1 GPU plan",
                reason: "dimensions above three currently support FftPrecision::F32 only",
            });
        }
        // Support indices and bin keys must remain exact in f32 position
        // arithmetic; reserve the support halo beyond the last bin.
        let maximum_signed_length = (1usize << f32::MANTISSA_DIGITS)
            .saturating_sub(kernel.width())
            .min(i32::MAX as usize - kernel.width());
        for (&length, &modes) in fine_shape.iter().zip(config.n_modes()) {
            if length % 2 != 0 {
                return Err(NufftError::FineGridMustBeEven { length });
            }
            debug_assert!(length >= modes);
            if length > maximum_signed_length {
                return Err(NufftError::GpuFineGridIndexUnsupported {
                    length,
                    maximum: maximum_signed_length,
                });
            }
            u32::try_from(length).map_err(|_| NufftError::LengthOverflow {
                context: "rank-generic type-1 fine-grid axis length",
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

        let batch_capacity = config.batch();
        let mode_count = config.mode_count()?;
        let fine_product = fine_shape.iter().try_fold(1usize, |product, &length| {
            product
                .checked_mul(length)
                .ok_or(NufftError::LengthOverflow {
                    context: "rank-generic type-1 fine-grid element count",
                })
        })?;
        let fine_product_u32 =
            u32::try_from(fine_product).map_err(|_| NufftError::LengthOverflow {
                context: "rank-generic type-1 bin index space",
            })?;
        let fine_element_count = checked_product(
            "rank-generic type-1 batched fine-grid element count",
            fine_product,
            batch_capacity,
        )?;
        u32::try_from(fine_element_count).map_err(|_| NufftError::LengthOverflow {
            context: "rank-generic type-1 batched fine-grid shader index space",
        })?;
        let fine_bytes = checked_buffer_size(
            "rank-generic type-1 fine grid",
            fine_element_count,
            COMPLEX_F32_BYTES,
        )?;
        let count_bytes =
            checked_buffer_size("rank-generic type-1 bin counts", fine_product, U32_BYTES)?;
        let offset_bytes = checked_buffer_size(
            "rank-generic type-1 bin offsets",
            fine_product + 1,
            U32_BYTES,
        )?;
        let amplitude_bytes = checked_buffer_size(
            "rank-generic type-1 deconvolution amplitudes",
            mode_count,
            F32_BYTES,
        )?;
        let max_storage_binding_bytes = limits.max_storage_buffer_binding_size;
        for (label, bytes) in [
            ("rank-generic type-1 fine grid", fine_bytes),
            ("rank-generic type-1 bin counts", count_bytes),
            ("rank-generic type-1 bin offsets", offset_bytes),
            (
                "rank-generic type-1 deconvolution amplitudes",
                amplitude_bytes,
            ),
        ] {
            validate_binding_limit(label, bytes, max_storage_binding_bytes)?;
        }
        if fine_bytes > limits.max_buffer_size {
            return Err(NufftError::GpuBufferBindingTooLarge {
                buffer: "rank-generic type-1 fine grid",
                required_bytes: fine_bytes,
                limit_bytes: limits.max_buffer_size,
            });
        }

        let amplitudes = mode_amplitudes_nd(config, kernel_fourier_coefficients)?;
        let amplitude_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("wgpu_nufft.type1_nd.deconvolution_amplitudes"),
            contents: bytemuck::cast_slice(&amplitudes),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let fine_usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST;
        let fine_input = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("wgpu_nufft.type1_nd.fine_input"),
            size: fine_bytes,
            usage: fine_usage,
            mapped_at_creation: false,
        });
        let fine_output = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("wgpu_nufft.type1_nd.fine_output"),
            size: fine_bytes,
            usage: fine_usage,
            mapped_at_creation: false,
        });
        let scratch =
            Type1ScratchBuffers::new(device, "wgpu_nufft.type1_nd", count_bytes, offset_bytes);
        let prefix_scan = GpuExclusiveScanU32::new(device, fine_product)?;

        let fft_direction = match config.sign() {
            NufftSign::Positive => FftDirection::Inverse,
            NufftSign::Negative => FftDirection::Forward,
        };
        let fft_config = FftConfig::new_nd(fine_shape.to_vec())
            .with_direction(fft_direction)
            .with_normalization(Normalization::None)
            .with_batch(batch_capacity)
            .with_precision(FftPrecision::F32);
        let fft = FftPlan::c2c(device, queue, fft_config).map_err(|source| {
            NufftError::FftShapeUnsupported {
                stage: "rank-generic type-1 oversampled-grid C2C plan",
                source,
            }
        })?;

        let count_pipeline = create_compute_pipeline(
            device,
            "wgpu_nufft.type1_nd.bin_count",
            &generate_count_wgsl(fine_shape),
        );
        let count_layout = count_pipeline.get_bind_group_layout(0);
        let terminal_pipeline = create_compute_pipeline(
            device,
            "wgpu_nufft.type1_nd.bin_terminal",
            &generate_terminal_wgsl(),
        );
        let terminal_layout = terminal_pipeline.get_bind_group_layout(0);
        let scatter_pipeline = create_compute_pipeline(
            device,
            "wgpu_nufft.type1_nd.bin_scatter",
            &generate_scatter_wgsl(fine_shape),
        );
        let scatter_layout = scatter_pipeline.get_bind_group_layout(0);
        let sort_pipeline = create_compute_pipeline(
            device,
            "wgpu_nufft.type1_nd.bin_sort",
            &generate_sort_wgsl(),
        );
        let sort_layout = sort_pipeline.get_bind_group_layout(0);
        let gather_pipeline = create_compute_pipeline(
            device,
            "wgpu_nufft.type1_nd.spread_gather",
            &generate_gather_wgsl(kernel, fine_shape),
        );
        let gather_layout = gather_pipeline.get_bind_group_layout(0);
        let deconvolution_pipeline = create_compute_pipeline(
            device,
            "wgpu_nufft.type1_nd.deconvolution",
            &generate_deconvolution_wgsl(config, fine_shape),
        );
        let deconvolution_layout = deconvolution_pipeline.get_bind_group_layout(0);

        let max_workgroups_per_dimension = limits.max_compute_workgroups_per_dimension;
        let sort_dispatch = split_workgroups(
            fine_product_u32.div_ceil(WORKGROUP_SIZE),
            max_workgroups_per_dimension,
        )?;

        Ok(Self {
            fft,
            amplitudes: amplitude_buffer,
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
            fine_product,
            dimensions,
            batch_capacity,
            max_storage_binding_bytes,
        })
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
        if active_batch == 0 || active_batch > self.batch_capacity {
            return Err(NufftError::InvalidBatch {
                actual: active_batch,
                maximum: self.batch_capacity,
            });
        }
        u32::try_from(point_count).map_err(|_| NufftError::LengthOverflow {
            context: "rank-generic type-1 GPU point count",
        })?;
        let output_elements = checked_product(
            "rank-generic type-1 batched output element count",
            active_batch,
            self.mode_count,
        )?;
        let output_bytes = checked_buffer_size(
            "rank-generic type-1 output buffer",
            output_elements,
            COMPLEX_F32_BYTES,
        )?;
        validate_external_storage_buffer(
            "rank-generic type-1 output",
            output,
            output_bytes,
            self.max_storage_binding_bytes,
        )?;

        if point_count == 0 {
            // Type-1 with no sources has a nonempty, all-zero mode output:
            // clear the fine grid, then still run FFT + deconvolution so the
            // external output is fully overwritten.
            encoder.clear_buffer(&self.fine_input, 0, None);
        } else {
            self.encode_spread(
                device,
                encoder,
                active_batch,
                point_count,
                points,
                strengths,
            )?;
        }

        self.fft
            .execute_views(
                device,
                encoder,
                BufferView::whole(&self.fine_input),
                BufferView::whole(&self.fine_output),
            )
            .map_err(|source| NufftError::FftExecutionFailed {
                stage: "rank-generic type-1 oversampled-grid C2C transform",
                source,
            })?;

        let deconvolution_elements_u32 =
            u32::try_from(output_elements).map_err(|_| NufftError::LengthOverflow {
                context: "rank-generic type-1 deconvolution shader index space",
            })?;
        let deconvolution_dispatch = split_workgroups(
            deconvolution_elements_u32.div_ceil(WORKGROUP_SIZE),
            self.max_workgroups_per_dimension,
        )?;
        let deconvolution_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("wgpu_nufft.type1_nd.deconvolution.bind_group"),
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
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("wgpu_nufft.type1_nd.deconvolution.pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.deconvolution_pipeline);
            pass.set_bind_group(0, &deconvolution_bind_group, &[]);
            pass.dispatch_workgroups(
                deconvolution_dispatch.0,
                deconvolution_dispatch.1,
                deconvolution_dispatch.2,
            );
        }
        Ok(())
    }

    #[cfg(feature = "gpu-profiling")]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_profiled(
        &self,
        _device: &wgpu::Device,
        _encoder: &mut wgpu::CommandEncoder,
        _point_count: usize,
        _points: &wgpu::Buffer,
        _strengths: &wgpu::Buffer,
        _output: &wgpu::Buffer,
        _query_set: &wgpu::QuerySet,
        _first_query: u32,
    ) -> Result<NufftGpuProfileLayout> {
        Err(NufftError::GpuExecutionUnavailable {
            kind: "rank-generic type-1 stage profiling",
            reason: "stage profiling is not yet instrumented above three dimensions",
        })
    }

    #[cfg(feature = "gpu-profiling")]
    pub(crate) fn fft_diagnostics(&self) -> wgpu_fft::FftDiagnostics {
        self.fft.diagnostics()
    }

    fn encode_spread(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        active_batch: usize,
        point_count: usize,
        points: &wgpu::Buffer,
        strengths: &wgpu::Buffer,
    ) -> Result<()> {
        let coordinate_count = checked_product(
            "rank-generic type-1 coordinate scalar count",
            point_count,
            self.dimensions,
        )?;
        let point_bytes = checked_buffer_size(
            "rank-generic type-1 point buffer",
            coordinate_count,
            F32_BYTES,
        )?;
        let strength_elements = checked_product(
            "rank-generic type-1 batched strength count",
            active_batch,
            point_count,
        )?;
        let strength_bytes = checked_buffer_size(
            "rank-generic type-1 strength buffer",
            strength_elements,
            COMPLEX_F32_BYTES,
        )?;
        validate_external_storage_buffer(
            "rank-generic type-1 point",
            points,
            point_bytes,
            self.max_storage_binding_bytes,
        )?;
        validate_external_storage_buffer(
            "rank-generic type-1 strength",
            strengths,
            strength_bytes,
            self.max_storage_binding_bytes,
        )?;
        let index_bytes =
            checked_buffer_size("rank-generic type-1 sorted indices", point_count, U32_BYTES)?;
        validate_binding_limit(
            "rank-generic type-1 sorted indices",
            index_bytes,
            self.max_storage_binding_bytes,
        )?;
        let sorted_indices = self.scratch.sorted_indices(index_bytes);

        let point_count_u32 = point_count as u32;
        let point_dispatch = split_workgroups(
            point_count_u32.div_ceil(WORKGROUP_SIZE),
            self.max_workgroups_per_dimension,
        )?;

        encoder.clear_buffer(&self.scratch.bin_counts, 0, None);
        encoder.clear_buffer(&self.scratch.bin_cursors, 0, None);

        let count_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("wgpu_nufft.type1_nd.bin_count.bind_group"),
            layout: &self.count_layout,
            entries: &[
                binding_entry(0, points, point_bytes),
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.scratch.bin_counts.as_entire_binding(),
                },
            ],
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("wgpu_nufft.type1_nd.bin_count.pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.count_pipeline);
            pass.set_bind_group(0, &count_bind_group, &[]);
            pass.dispatch_workgroups(point_dispatch.0, point_dispatch.1, point_dispatch.2);
        }

        self.prefix_scan.encode(
            device,
            encoder,
            &self.scratch.bin_counts,
            &self.scratch.bin_offsets,
        )?;

        let terminal_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("wgpu_nufft.type1_nd.bin_terminal.bind_group"),
            layout: &self.terminal_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.scratch.bin_counts.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.scratch.bin_offsets.as_entire_binding(),
                },
            ],
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("wgpu_nufft.type1_nd.bin_terminal.pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.terminal_pipeline);
            pass.set_bind_group(0, &terminal_bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }

        let scatter_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("wgpu_nufft.type1_nd.bin_scatter.bind_group"),
            layout: &self.scatter_layout,
            entries: &[
                binding_entry(0, points, point_bytes),
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.scratch.bin_offsets.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.scratch.bin_cursors.as_entire_binding(),
                },
                binding_entry(3, &sorted_indices, index_bytes),
            ],
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("wgpu_nufft.type1_nd.bin_scatter.pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.scatter_pipeline);
            pass.set_bind_group(0, &scatter_bind_group, &[]);
            pass.dispatch_workgroups(point_dispatch.0, point_dispatch.1, point_dispatch.2);
        }

        let sort_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("wgpu_nufft.type1_nd.bin_sort.bind_group"),
            layout: &self.sort_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.scratch.bin_offsets.as_entire_binding(),
                },
                binding_entry(1, &sorted_indices, index_bytes),
            ],
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("wgpu_nufft.type1_nd.bin_sort.pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.sort_pipeline);
            pass.set_bind_group(0, &sort_bind_group, &[]);
            pass.dispatch_workgroups(
                self.sort_dispatch.0,
                self.sort_dispatch.1,
                self.sort_dispatch.2,
            );
        }

        let vector_blocks = self.batch_capacity.div_ceil(VECTOR_BLOCK_SIZE);
        let gather_elements = checked_product(
            "rank-generic type-1 gather work item count",
            self.fine_product,
            vector_blocks,
        )?;
        let gather_elements_u32 =
            u32::try_from(gather_elements).map_err(|_| NufftError::LengthOverflow {
                context: "rank-generic type-1 gather shader index space",
            })?;
        let gather_dispatch = split_workgroups(
            gather_elements_u32.div_ceil(WORKGROUP_SIZE),
            self.max_workgroups_per_dimension,
        )?;
        let gather_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("wgpu_nufft.type1_nd.spread_gather.bind_group"),
            layout: &self.gather_layout,
            entries: &[
                binding_entry(0, points, point_bytes),
                binding_entry(1, strengths, strength_bytes),
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.scratch.bin_offsets.as_entire_binding(),
                },
                binding_entry(3, &sorted_indices, index_bytes),
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.fine_input.as_entire_binding(),
                },
            ],
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("wgpu_nufft.type1_nd.spread_gather.pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.gather_pipeline);
            pass.set_bind_group(0, &gather_bind_group, &[]);
            pass.dispatch_workgroups(gather_dispatch.0, gather_dispatch.1, gather_dispatch.2);
        }

        // The gather derives active_batch from the bound strength size; the
        // fine-grid tail beyond active_batch may hold stale writes, so clear it
        // before the batched FFT.
        if active_batch < self.batch_capacity {
            let active_fine_bytes = checked_buffer_size(
                "rank-generic type-1 active fine bytes",
                checked_product(
                    "rank-generic type-1 active fine elements",
                    self.fine_product,
                    active_batch,
                )?,
                COMPLEX_F32_BYTES,
            )?;
            encoder.clear_buffer(&self.fine_input, active_fine_bytes, None);
        }
        Ok(())
    }
}

fn per_axis_position_constants(fine_shape: &[usize]) -> String {
    let mut source = String::new();
    for (axis, &length) in fine_shape.iter().enumerate() {
        let position_scale = length as f64 / std::f64::consts::TAU;
        let position_scale_hi = position_scale as f32;
        let position_scale_lo = (position_scale - f64::from(position_scale_hi)) as f32;
        let _ = writeln!(source, "const FINE_LENGTH{axis}: u32 = {length}u;");
        let _ = writeln!(source, "const FINE_LENGTH_I32_{axis}: i32 = {length}i;");
        let _ = writeln!(
            source,
            "const FINE_LENGTH_F32_{axis}: f32 = {};",
            format_wgsl_f32(length as f32)
        );
        let _ = writeln!(
            source,
            "const POSITION_SCALE_HI{axis}: f32 = {};",
            format_wgsl_f32(position_scale_hi)
        );
        let _ = writeln!(
            source,
            "const POSITION_SCALE_LO{axis}: f32 = {};",
            format_wgsl_f32(position_scale_lo)
        );
        let _ = writeln!(
            source,
            "const GRID_ORIGIN{axis}: f32 = {};",
            format_wgsl_f32((length / 2) as f32)
        );
    }
    source
}

fn shared_fold_functions() -> &'static str {
    "fn position_is_negative(value: Df64) -> bool {\n\
     \x20   return value.hi < 0.0 || (value.hi == 0.0 && value.lo < 0.0);\n}\n\n\
     fn position_at_least(value: Df64, limit: f32) -> bool {\n\
     \x20   return value.hi > limit || (value.hi == limit && value.lo >= 0.0);\n}\n\n\
     fn fold_position(point: f32, scale: vec2<f32>, origin: f32, limit: f32) -> Df64 {\n\
     \x20   let scaled = df64_mul(Df64(point, 0.0), Df64(scale.x, scale.y));\n\
     \x20   var position = df64_add(scaled, Df64(origin, 0.0));\n\
     \x20   if (position_is_negative(position)) { position = df64_add(position, Df64(limit, 0.0)); }\n\
     \x20   if (position_is_negative(position)) { position = df64_add(position, Df64(limit, 0.0)); }\n\
     \x20   if (position_at_least(position, limit)) { position = df64_sub(position, Df64(limit, 0.0)); }\n\
     \x20   if (position_at_least(position, limit)) { position = df64_sub(position, Df64(limit, 0.0)); }\n\
     \x20   return position;\n}\n\n\
     fn floor_df64_to_i32(value: Df64) -> i32 {\n\
     \x20   let base = floor(value.hi);\n\
     \x20   let remainder = df64_sub(value, Df64(base, 0.0));\n\
     \x20   let has_negative_remainder = remainder.hi < 0.0 ||\n\
     \x20       (remainder.hi == 0.0 && remainder.lo < 0.0);\n\
     \x20   return i32(base) - select(0, 1, has_negative_remainder);\n}\n\n\
     fn ceil_df64_to_i32(value: Df64) -> i32 {\n\
     \x20   let base = floor(value.hi);\n\
     \x20   let remainder = df64_sub(value, Df64(base, 0.0));\n\
     \x20   let has_positive_remainder = remainder.hi > 0.0 ||\n\
     \x20       (remainder.hi == 0.0 && remainder.lo > 0.0);\n\
     \x20   return i32(base) + select(0, 1, has_positive_remainder);\n}\n\n"
}

/// Emits per-axis statements computing `bin{axis}` (the folded fine cell) for
/// the point whose first coordinate lives at `points[base]`.
fn per_axis_bin_statements(dimensions: usize, indent: &str) -> String {
    let mut source = String::new();
    for axis in 0..dimensions {
        let _ = writeln!(
            source,
            "{indent}let position{axis} = fold_position(points[base + {axis}u], \
             vec2<f32>(POSITION_SCALE_HI{axis}, POSITION_SCALE_LO{axis}), GRID_ORIGIN{axis}, FINE_LENGTH_F32_{axis});\n\
             {indent}let bin{axis} = u32(floor_df64_to_i32(position{axis}));"
        );
    }
    // axis-0-fast flat bin index
    let mut fold = format!("bin{}", dimensions - 1);
    for axis in (0..dimensions.saturating_sub(1)).rev() {
        fold = format!("bin{axis} + FINE_LENGTH{axis} * ({fold})");
    }
    let _ = writeln!(source, "{indent}let bin = {fold};");
    source
}

fn generate_count_wgsl(fine_shape: &[usize]) -> String {
    let dimensions = fine_shape.len();
    let mut source = String::new();
    let _ = writeln!(source, "const WORKGROUP_SIZE: u32 = {WORKGROUP_SIZE}u;");
    let _ = writeln!(source, "const DIMS: u32 = {dimensions}u;");
    source.push_str(&per_axis_position_constants(fine_shape));
    source.push_str(
        "\n@group(0) @binding(0) var<storage, read> points: array<f32>;\n\
         @group(0) @binding(1) var<storage, read_write> bin_counts: array<atomic<u32>>;\n\n",
    );
    source.push_str(shared_fold_functions());
    source.push_str(
        "@compute @workgroup_size(WORKGROUP_SIZE)\n\
         fn main(\n\
         \x20   @builtin(local_invocation_id) lid: vec3<u32>,\n\
         \x20   @builtin(workgroup_id) wid: vec3<u32>,\n\
         \x20   @builtin(num_workgroups) nwg: vec3<u32>,\n\
         ) {\n\
         \x20   let total = arrayLength(&points) / DIMS;\n\
         \x20   let wg_flat = (wid.z * nwg.y + wid.y) * nwg.x + wid.x;\n\
         \x20   if (wg_flat > (total - 1u) / WORKGROUP_SIZE) { return; }\n\
         \x20   let point_index = wg_flat * WORKGROUP_SIZE + lid.x;\n\
         \x20   if (point_index >= total) { return; }\n\
         \x20   let base = point_index * DIMS;\n",
    );
    source.push_str(&per_axis_bin_statements(dimensions, "    "));
    source.push_str("    atomicAdd(&bin_counts[bin], 1u);\n}\n");
    format!("{}\n{source}", wgpu_fft::kernels::DF64_WGSL)
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

fn generate_scatter_wgsl(fine_shape: &[usize]) -> String {
    let dimensions = fine_shape.len();
    let mut source = String::new();
    let _ = writeln!(source, "const WORKGROUP_SIZE: u32 = {WORKGROUP_SIZE}u;");
    let _ = writeln!(source, "const DIMS: u32 = {dimensions}u;");
    source.push_str(&per_axis_position_constants(fine_shape));
    source.push_str(
        "\n@group(0) @binding(0) var<storage, read> points: array<f32>;\n\
         @group(0) @binding(1) var<storage, read> bin_offsets: array<u32>;\n\
         @group(0) @binding(2) var<storage, read_write> bin_cursors: array<atomic<u32>>;\n\
         @group(0) @binding(3) var<storage, read_write> sorted_indices: array<u32>;\n\n",
    );
    source.push_str(shared_fold_functions());
    source.push_str(
        "@compute @workgroup_size(WORKGROUP_SIZE)\n\
         fn main(\n\
         \x20   @builtin(local_invocation_id) lid: vec3<u32>,\n\
         \x20   @builtin(workgroup_id) wid: vec3<u32>,\n\
         \x20   @builtin(num_workgroups) nwg: vec3<u32>,\n\
         ) {\n\
         \x20   let total = arrayLength(&points) / DIMS;\n\
         \x20   let wg_flat = (wid.z * nwg.y + wid.y) * nwg.x + wid.x;\n\
         \x20   if (wg_flat > (total - 1u) / WORKGROUP_SIZE) { return; }\n\
         \x20   let point_index = wg_flat * WORKGROUP_SIZE + lid.x;\n\
         \x20   if (point_index >= total) { return; }\n\
         \x20   let base = point_index * DIMS;\n",
    );
    source.push_str(&per_axis_bin_statements(dimensions, "    "));
    source.push_str(
        "    let slot = bin_offsets[bin] + atomicAdd(&bin_cursors[bin], 1u);\n\
         \x20   sorted_indices[slot] = point_index;\n}\n",
    );
    format!("{}\n{source}", wgpu_fft::kernels::DF64_WGSL)
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

fn generate_gather_wgsl(kernel: EsKernel, fine_shape: &[usize]) -> String {
    let dimensions = fine_shape.len();
    let width = kernel.width();
    let bin_radius = width.div_ceil(2);
    let fine_product: usize = fine_shape.iter().product();

    let mut source = String::new();
    let _ = writeln!(source, "const WORKGROUP_SIZE: u32 = {WORKGROUP_SIZE}u;");
    let _ = writeln!(source, "const DIMS: u32 = {dimensions}u;");
    let _ = writeln!(source, "const WIDTH: u32 = {width}u;");
    let _ = writeln!(source, "const WIDTH_I32: i32 = {width}i;");
    let _ = writeln!(source, "const WIDTH_F32: f32 = {width}.0;");
    let _ = writeln!(
        source,
        "const HALF_WIDTH: f32 = {};",
        format_wgsl_f32(kernel.half_width() as f32)
    );
    let _ = writeln!(
        source,
        "const BETA: f32 = {};",
        format_wgsl_f32(kernel.beta() as f32)
    );
    let _ = writeln!(source, "const BIN_RADIUS: i32 = {bin_radius}i;");
    let _ = writeln!(
        source,
        "const VECTOR_BLOCK_SIZE: u32 = {VECTOR_BLOCK_SIZE}u;"
    );
    let _ = writeln!(source, "const FINE_PRODUCT: u32 = {fine_product}u;");
    source.push_str(&per_axis_position_constants(fine_shape));
    source.push_str(
        "\n@group(0) @binding(0) var<storage, read> points: array<f32>;\n\
         @group(0) @binding(1) var<storage, read> strengths: array<vec2<f32>>;\n\
         @group(0) @binding(2) var<storage, read> bin_offsets: array<u32>;\n\
         @group(0) @binding(3) var<storage, read> sorted_indices: array<u32>;\n\
         @group(0) @binding(4) var<storage, read_write> fine_grid: array<vec2<f32>>;\n\n",
    );
    source.push_str(shared_fold_functions());
    source.push_str(
        "fn es_weight(distance: f32) -> f32 {\n\
         \x20   let scaled = 2.0 * abs(distance) / WIDTH_F32;\n\
         \x20   let squared = scaled * scaled;\n\
         \x20   if (squared >= 1.0) { return 0.0; }\n\
         \x20   return exp(BETA * (sqrt(max(0.0, 1.0 - squared)) - 1.0));\n}\n\n\
         fn wrap_bin(index: i32, limit: i32) -> u32 {\n\
         \x20   var wrapped = index;\n\
         \x20   if (wrapped < 0) { wrapped = wrapped + limit; }\n\
         \x20   if (wrapped >= limit) { wrapped = wrapped - limit; }\n\
         \x20   return u32(wrapped);\n}\n\n",
    );
    source.push_str(
        "@compute @workgroup_size(WORKGROUP_SIZE)\n\
         fn main(\n\
         \x20   @builtin(local_invocation_id) lid: vec3<u32>,\n\
         \x20   @builtin(workgroup_id) wid: vec3<u32>,\n\
         \x20   @builtin(num_workgroups) nwg: vec3<u32>,\n\
         ) {\n\
         \x20   let wg_flat = (wid.z * nwg.y + wid.y) * nwg.x + wid.x;\n\
         \x20   let total_vectors = arrayLength(&fine_grid) / FINE_PRODUCT;\n\
         \x20   let point_count = arrayLength(&points) / DIMS;\n\
         \x20   let vector_blocks = (total_vectors + VECTOR_BLOCK_SIZE - 1u) / VECTOR_BLOCK_SIZE;\n\
         \x20   let total_work = FINE_PRODUCT * vector_blocks;\n\
         \x20   if (wg_flat > (total_work - 1u) / WORKGROUP_SIZE) { return; }\n\
         \x20   let work_index = wg_flat * WORKGROUP_SIZE + lid.x;\n\
         \x20   if (work_index >= total_work) { return; }\n\
         \x20   let vector_block = work_index / FINE_PRODUCT;\n\
         \x20   let cell = work_index - vector_block * FINE_PRODUCT;\n\
         \x20   let first_vector = vector_block * VECTOR_BLOCK_SIZE;\n\n\
         \x20   var remainder = cell;\n",
    );
    for axis in 0..dimensions {
        let _ = writeln!(
            source,
            "    let cell{axis} = remainder % FINE_LENGTH{axis}; remainder = remainder / FINE_LENGTH{axis};"
        );
    }
    source.push_str(
        "\n    var sums: array<vec2<f32>, VECTOR_BLOCK_SIZE>;\n\
         \x20   for (var vector_local = 0u; vector_local < VECTOR_BLOCK_SIZE;\n\
         \x20        vector_local = vector_local + 1u) {\n\
         \x20       sums[vector_local] = vec2<f32>(0.0, 0.0);\n\
         \x20   }\n",
    );
    // nested loops over the per-axis bin neighborhood, highest axis outermost
    for axis in (0..dimensions).rev() {
        let indent = "    ".repeat(dimensions - axis);
        let _ = writeln!(
            source,
            "{indent}for (var b{axis} = -BIN_RADIUS; b{axis} < BIN_RADIUS; b{axis} = b{axis} + 1) {{"
        );
        let inner = "    ".repeat(dimensions - axis + 1);
        let _ = writeln!(
            source,
            "{inner}let wrapped_bin{axis} = wrap_bin(i32(cell{axis}) + b{axis}, FINE_LENGTH_I32_{axis});"
        );
        if axis == dimensions - 1 {
            let _ = writeln!(source, "{inner}let flat{axis} = wrapped_bin{axis};");
        } else {
            let upper = axis + 1;
            let _ = writeln!(
                source,
                "{inner}let flat{axis} = wrapped_bin{axis} + FINE_LENGTH{axis} * flat{upper};"
            );
        }
    }
    let deepest = "    ".repeat(dimensions + 1);
    let _ = writeln!(
        source,
        "{deepest}let begin = bin_offsets[flat0];\n\
         {deepest}let end = bin_offsets[flat0 + 1u];\n\
         {deepest}for (var slot = begin; slot < end; slot = slot + 1u) {{\n\
         {deepest}    let point_index = sorted_indices[slot];\n\
         {deepest}    let point_base = point_index * DIMS;\n\
         {deepest}    var weight: f32 = 1.0;\n\
         {deepest}    var inside: bool = true;"
    );
    for axis in 0..dimensions {
        let _ = writeln!(
            source,
            "{deepest}    let point_position{axis} = fold_position(points[point_base + {axis}u], \
             vec2<f32>(POSITION_SCALE_HI{axis}, POSITION_SCALE_LO{axis}), GRID_ORIGIN{axis}, FINE_LENGTH_F32_{axis});\n\
             {deepest}    let start{axis} = ceil_df64_to_i32(df64_sub(point_position{axis}, Df64(HALF_WIDTH, 0.0)));\n\
             {deepest}    var unwrapped{axis} = i32(cell{axis});\n\
             {deepest}    if (unwrapped{axis} < start{axis}) {{ unwrapped{axis} = unwrapped{axis} + FINE_LENGTH_I32_{axis}; }}\n\
             {deepest}    if (unwrapped{axis} >= start{axis} + WIDTH_I32) {{ unwrapped{axis} = unwrapped{axis} - FINE_LENGTH_I32_{axis}; }}\n\
             {deepest}    if (unwrapped{axis} >= start{axis} && unwrapped{axis} < start{axis} + WIDTH_I32) {{\n\
             {deepest}        let distance{axis} = df64_sub(Df64(f32(unwrapped{axis}), 0.0), point_position{axis});\n\
             {deepest}        weight = weight * es_weight(distance{axis}.hi + distance{axis}.lo);\n\
             {deepest}    }} else {{\n\
             {deepest}        inside = false;\n\
             {deepest}    }}"
        );
    }
    let _ = writeln!(
        source,
        "{deepest}    if (inside) {{\n\
         {deepest}        for (var vector_local = 0u; vector_local < VECTOR_BLOCK_SIZE;\n\
         {deepest}             vector_local = vector_local + 1u) {{\n\
         {deepest}            let vector_index = first_vector + vector_local;\n\
         {deepest}            if (vector_index < total_vectors) {{\n\
         {deepest}                let strength_index = vector_index * point_count + point_index;\n\
         {deepest}                sums[vector_local] = sums[vector_local] +\n\
         {deepest}                    strengths[strength_index] * weight;\n\
         {deepest}            }}\n\
         {deepest}        }}\n\
         {deepest}    }}\n\
         {deepest}}}"
    );
    for axis in 0..dimensions {
        let indent = "    ".repeat(dimensions - axis);
        let _ = writeln!(source, "{indent}}}");
    }
    source.push_str(
        "\n    for (var vector_local = 0u; vector_local < VECTOR_BLOCK_SIZE;\n\
         \x20        vector_local = vector_local + 1u) {\n\
         \x20       let vector_index = first_vector + vector_local;\n\
         \x20       if (vector_index < total_vectors) {\n\
         \x20           fine_grid[vector_index * FINE_PRODUCT + cell] = sums[vector_local];\n\
         \x20       }\n\
         \x20   }\n}\n",
    );
    format!("{}\n{source}", wgpu_fft::kernels::DF64_WGSL)
}

fn generate_deconvolution_wgsl(config: &NufftConfig, fine_shape: &[usize]) -> String {
    let dimensions = config.dimensions();
    let n_modes = config.n_modes();
    let mode_count: usize = n_modes.iter().product();
    let fine_product: usize = fine_shape.iter().product();
    let centered = matches!(config.mode_order(), ModeOrder::Centered);

    let mut source = String::new();
    let _ = writeln!(source, "const WORKGROUP_SIZE: u32 = {WORKGROUP_SIZE}u;");
    let _ = writeln!(source, "const MODE_COUNT: u32 = {mode_count}u;");
    let _ = writeln!(source, "const FINE_PRODUCT: u32 = {fine_product}u;");
    let _ = writeln!(source, "const CENTERED_ORDER: bool = {centered};");
    for axis in 0..dimensions {
        let modes = n_modes[axis];
        let _ = writeln!(source, "const FINE{axis}: u32 = {}u;", fine_shape[axis]);
        let _ = writeln!(source, "const MODE{axis}: u32 = {modes}u;");
        let _ = writeln!(source, "const HALF{axis}: u32 = {}u;", modes / 2);
        let _ = writeln!(source, "const NONNEG{axis}: u32 = {}u;", modes.div_ceil(2));
    }
    source.push_str(
        "\n@group(0) @binding(0) var<storage, read> fine_grid: array<vec2<f32>>;\n\
         @group(0) @binding(1) var<storage, read> amplitudes: array<f32>;\n\
         @group(0) @binding(2) var<storage, read_write> output_values: array<vec2<f32>>;\n\n",
    );
    source.push_str(
        "fn fine_index_for_axis(output_index: u32, mode_count: u32, fine_length: u32, half: u32, nonnegative: u32) -> u32 {\n\
         \x20   if (CENTERED_ORDER) {\n\
         \x20       if (output_index < half) {\n\
         \x20           return fine_length - (half - output_index);\n\
         \x20       }\n\
         \x20       return output_index - half;\n\
         \x20   }\n\
         \x20   if (output_index < nonnegative) {\n\
         \x20       return output_index;\n\
         \x20   }\n\
         \x20   return fine_length - (mode_count - output_index);\n}\n\n",
    );
    source.push_str(
        "@compute @workgroup_size(WORKGROUP_SIZE)\n\
         fn main(\n\
         \x20   @builtin(local_invocation_id) lid: vec3<u32>,\n\
         \x20   @builtin(workgroup_id) wid: vec3<u32>,\n\
         \x20   @builtin(num_workgroups) nwg: vec3<u32>,\n\
         ) {\n\
         \x20   let wg_flat = (wid.z * nwg.y + wid.y) * nwg.x + wid.x;\n\
         \x20   let total = arrayLength(&output_values);\n\
         \x20   if (wg_flat > (total - 1u) / WORKGROUP_SIZE) { return; }\n\
         \x20   let linear_index = wg_flat * WORKGROUP_SIZE + lid.x;\n\
         \x20   if (linear_index >= total) { return; }\n\
         \x20   let vector_index = linear_index / MODE_COUNT;\n\
         \x20   let output_index = linear_index - vector_index * MODE_COUNT;\n\n\
         \x20   var remainder = output_index;\n",
    );
    for axis in 0..dimensions {
        let _ = writeln!(
            source,
            "    let mode_index{axis} = remainder % MODE{axis}; remainder = remainder / MODE{axis};\n    \
             let fine_index{axis} = fine_index_for_axis(mode_index{axis}, MODE{axis}, FINE{axis}, HALF{axis}, NONNEG{axis});"
        );
    }
    let mut fold = format!("fine_index{}", dimensions - 1);
    for axis in (0..dimensions.saturating_sub(1)).rev() {
        fold = format!("fine_index{axis} + FINE{axis} * ({fold})");
    }
    let _ = writeln!(source, "    let fine_flat = {fold};");
    source.push_str(
        "    output_values[linear_index] = fine_grid[vector_index * FINE_PRODUCT + fine_flat] *\n\
         \x20       amplitudes[output_index];\n}\n",
    );
    source
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nd_count_and_scatter_shaders_fold_per_axis_bins() {
        let source = generate_count_wgsl(&[16, 12, 10, 10]);
        assert!(source.contains("atomicAdd(&bin_counts[bin], 1u);"));
        assert!(source.contains("let bin3 = u32(floor_df64_to_i32(position3));"));
        assert!(source.contains("let bin = bin0 + FINE_LENGTH0 * (bin1 + FINE_LENGTH1 * (bin2 + FINE_LENGTH2 * (bin3)));"));
        let scatter = generate_scatter_wgsl(&[16, 12, 10, 10]);
        assert!(scatter.contains("atomicAdd(&bin_cursors[bin], 1u)"));
        assert!(scatter.contains("sorted_indices[slot] = point_index;"));
    }

    #[test]
    fn nd_gather_shader_uses_neighbor_bins_and_single_writer_cells() {
        let kernel = EsKernel::for_tolerance(1.0e-3, 2.0).unwrap();
        let source = generate_gather_wgsl(kernel, &[16, 12, 10, 10]);
        assert!(source.contains("for (var b3 = -BIN_RADIUS; b3 < BIN_RADIUS; b3 = b3 + 1) {"));
        assert!(source.contains("let flat0 = wrapped_bin0 + FINE_LENGTH0 * flat1;"));
        assert!(source.contains("let begin = bin_offsets[flat0];"));
        assert!(
            source.contains("fine_grid[vector_index * FINE_PRODUCT + cell] = sums[vector_local];")
        );
        assert!(!source.contains("atomicAdd"));
    }

    #[test]
    fn nd_deconvolution_shader_maps_modes_per_axis() {
        let config = NufftConfig::new([5, 4, 3, 3], 1.0e-3);
        let source = generate_deconvolution_wgsl(&config, &[12, 10, 8, 8]);
        assert!(source.contains("fn fine_index_for_axis("));
        assert!(source.contains("let mode_index3 = remainder % MODE3;"));
        assert!(source.contains(
            "let fine_flat = fine_index0 + FINE0 * (fine_index1 + FINE1 * (fine_index2 + FINE2 * (fine_index3)));"
        ));
    }
}
