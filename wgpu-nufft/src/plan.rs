use crate::config::NufftConfig;
use crate::direct::{reference_type1_f64, reference_type2_f64};
use crate::error::{NufftError, Result};
use crate::gpu::Type2GpuPlan;
use crate::gpu_2d::Type2GpuPlan2d;
use crate::gpu_3d::Type2GpuPlan3d;
#[cfg(feature = "gpu-profiling")]
use crate::gpu_profile::NufftGpuProfileLayout;
use crate::gpu_type1::Type1GpuPlan;
use crate::gpu_type1_2d::{Type1Gather2d, Type1GpuPlan2d};
use crate::gpu_type1_3d::{Type1Gather3d, Type1GpuPlan3d};
use crate::kernel::{select_fine_grid_size, EsKernel};
use crate::Complex64;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum NufftKind {
    Type1,
    Type2,
}

enum Type1GpuExecution {
    OneD(Type1GpuPlan),
    TwoD(Type1GpuPlan2d),
    ThreeD(Type1GpuPlan3d),
}

impl Type1GpuExecution {
    fn encode(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        point_count: usize,
        points: &wgpu::Buffer,
        strengths: &wgpu::Buffer,
        output: &wgpu::Buffer,
    ) -> Result<()> {
        match self {
            Self::OneD(plan) => {
                plan.encode(device, encoder, point_count, points, strengths, output)
            }
            Self::TwoD(plan) => {
                plan.encode(device, encoder, point_count, points, strengths, output)
            }
            Self::ThreeD(plan) => {
                plan.encode(device, encoder, point_count, points, strengths, output)
            }
        }
    }

    #[cfg(feature = "gpu-profiling")]
    #[allow(clippy::too_many_arguments)]
    fn encode_profiled(
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
        match self {
            Self::OneD(plan) => plan.encode_profiled(
                device,
                encoder,
                point_count,
                points,
                strengths,
                output,
                query_set,
                first_query,
            ),
            Self::TwoD(plan) => plan.encode_profiled(
                device,
                encoder,
                point_count,
                points,
                strengths,
                output,
                query_set,
                first_query,
            ),
            Self::ThreeD(plan) => plan.encode_profiled(
                device,
                encoder,
                point_count,
                points,
                strengths,
                output,
                query_set,
                first_query,
            ),
        }
    }

    #[cfg(feature = "gpu-profiling")]
    fn fft_diagnostics(&self) -> wgpu_fft::FftDiagnostics {
        match self {
            Self::OneD(plan) => plan.fft_diagnostics(),
            Self::TwoD(plan) => plan.fft_diagnostics(),
            Self::ThreeD(plan) => plan.fft_diagnostics(),
        }
    }
}

enum Type2GpuExecution {
    OneD(Type2GpuPlan),
    TwoD(Type2GpuPlan2d),
    ThreeD(Type2GpuPlan3d),
}

impl Type2GpuExecution {
    fn encode(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        point_count: usize,
        points: &wgpu::Buffer,
        coefficients: &wgpu::Buffer,
        output: &wgpu::Buffer,
    ) -> Result<()> {
        match self {
            Self::OneD(plan) => {
                plan.encode(device, encoder, point_count, points, coefficients, output)
            }
            Self::TwoD(plan) => {
                plan.encode(device, encoder, point_count, points, coefficients, output)
            }
            Self::ThreeD(plan) => {
                plan.encode(device, encoder, point_count, points, coefficients, output)
            }
        }
    }

    #[cfg(feature = "gpu-profiling")]
    #[allow(clippy::too_many_arguments)]
    fn encode_profiled(
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
        match self {
            Self::OneD(plan) => plan.encode_profiled(
                device,
                encoder,
                point_count,
                points,
                coefficients,
                output,
                query_set,
                first_query,
            ),
            Self::TwoD(plan) => plan.encode_profiled(
                device,
                encoder,
                point_count,
                points,
                coefficients,
                output,
                query_set,
                first_query,
            ),
            Self::ThreeD(plan) => plan.encode_profiled(
                device,
                encoder,
                point_count,
                points,
                coefficients,
                output,
                query_set,
                first_query,
            ),
        }
    }

    #[cfg(feature = "gpu-profiling")]
    fn fft_diagnostics(&self) -> wgpu_fft::FftDiagnostics {
        match self {
            Self::OneD(plan) => plan.fft_diagnostics(),
            Self::TwoD(plan) => plan.fft_diagnostics(),
            Self::ThreeD(plan) => plan.fft_diagnostics(),
        }
    }
}

/// Reusable NUFFT planning metadata.
///
/// Every plan supports the direct CPU fallback. GPU-prepared plans add private
/// pipeline and `wgpu-fft` state while preserving the same public metadata.
pub struct NufftPlan {
    kind: NufftKind,
    config: NufftConfig,
    kernel: EsKernel,
    fine_grid_shape: Vec<usize>,
    centered_kernel_fourier_coefficients: Vec<Vec<f64>>,
    gpu_type1: Option<Box<Type1GpuExecution>>,
    gpu_type2: Option<Box<Type2GpuExecution>>,
}

impl std::fmt::Debug for NufftPlan {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NufftPlan")
            .field("kind", &self.kind)
            .field("config", &self.config)
            .field("kernel", &self.kernel)
            .field("fine_grid_shape", &self.fine_grid_shape)
            .field(
                "centered_kernel_fourier_coefficients",
                &self.centered_kernel_fourier_coefficients,
            )
            .field("gpu_type1_ready", &self.gpu_type1.is_some())
            .field("gpu_type2_ready", &self.gpu_type2.is_some())
            .finish()
    }
}

impl NufftPlan {
    pub fn type1(config: NufftConfig) -> Result<Self> {
        Self::new(NufftKind::Type1, config)
    }

    pub fn type2(config: NufftConfig) -> Result<Self> {
        Self::new(NufftKind::Type2, config)
    }

    /// Builds a reusable one-, two-, or three-dimensional type-1 GPU plan.
    ///
    /// Execution is entirely GPU-resident and records commands into a caller
    /// supplied encoder. The plan is device-specific and may be reused with
    /// different caller-owned point, strength, and output buffers. Executions
    /// using the same plan must retain queue order because the plan reuses its
    /// fine-grid scratch buffers.
    pub fn type1_gpu(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        config: NufftConfig,
    ) -> Result<Self> {
        Self::type1_gpu_with_gathers(
            device,
            queue,
            config,
            Type1Gather2d::Tiled16,
            Type1Gather3d::Tiled8x8x4,
        )
    }

    /// Benchmark-only constructor retaining the original one-cell-per-lane 2D
    /// gather while the tiled prototype is evaluated.
    #[doc(hidden)]
    #[cfg(feature = "type1-2d-tile-prototype")]
    pub fn type1_gpu_with_global_2d_gather_for_testing(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        config: NufftConfig,
    ) -> Result<Self> {
        if config.dimensions() != 2 {
            return Err(NufftError::GpuDimensionsUnsupported {
                kind: "type-1 global 2D gather prototype control",
                actual: config.dimensions(),
                supported: 2,
            });
        }
        Self::type1_gpu_with_gathers(
            device,
            queue,
            config,
            Type1Gather2d::Global,
            Type1Gather3d::Global,
        )
    }

    /// Benchmark-only constructor selecting the 16x16 shared-memory 2D gather
    /// prototype. This surface is intentionally absent from default builds.
    #[doc(hidden)]
    #[cfg(feature = "type1-2d-tile-prototype")]
    pub fn type1_gpu_with_tiled_2d_gather_for_testing(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        config: NufftConfig,
    ) -> Result<Self> {
        if config.dimensions() != 2 {
            return Err(NufftError::GpuDimensionsUnsupported {
                kind: "type-1 tiled 2D gather prototype",
                actual: config.dimensions(),
                supported: 2,
            });
        }
        Self::type1_gpu_with_gathers(
            device,
            queue,
            config,
            Type1Gather2d::Tiled16,
            Type1Gather3d::Global,
        )
    }

    /// Benchmark-only constructor retaining the one-cell-per-lane 3D gather
    /// as a deterministic comparison path for the tiled default.
    #[doc(hidden)]
    #[cfg(feature = "type1-3d-tile-prototype")]
    pub fn type1_gpu_with_global_3d_gather_for_testing(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        config: NufftConfig,
    ) -> Result<Self> {
        if config.dimensions() != 3 {
            return Err(NufftError::GpuDimensionsUnsupported {
                kind: "type-1 global 3D gather prototype control",
                actual: config.dimensions(),
                supported: 3,
            });
        }
        Self::type1_gpu_with_gathers(
            device,
            queue,
            config,
            Type1Gather2d::Tiled16,
            Type1Gather3d::Global,
        )
    }

    /// Benchmark-only constructor explicitly requesting the `8x8x4`
    /// shared-memory 3D gather used by the public default. Devices without the
    /// required limits use the global fallback.
    #[doc(hidden)]
    #[cfg(feature = "type1-3d-tile-prototype")]
    pub fn type1_gpu_with_tiled_3d_gather_for_testing(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        config: NufftConfig,
    ) -> Result<Self> {
        if config.dimensions() != 3 {
            return Err(NufftError::GpuDimensionsUnsupported {
                kind: "type-1 tiled 3D gather prototype",
                actual: config.dimensions(),
                supported: 3,
            });
        }
        Self::type1_gpu_with_gathers(
            device,
            queue,
            config,
            Type1Gather2d::Tiled16,
            Type1Gather3d::Tiled8x8x4,
        )
    }

    fn type1_gpu_with_gathers(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        config: NufftConfig,
        gather_2d: Type1Gather2d,
        gather_3d: Type1Gather3d,
    ) -> Result<Self> {
        if config.dimensions() > 3 {
            return Err(NufftError::GpuDimensionsUnsupported {
                kind: "type-1",
                actual: config.dimensions(),
                supported: 3,
            });
        }
        let mut plan = Self::new(NufftKind::Type1, config)?;
        let gpu = match plan.config.dimensions() {
            1 => Type1GpuExecution::OneD(Type1GpuPlan::new(
                device,
                queue,
                &plan.config,
                plan.kernel,
                plan.fine_grid_shape[0],
                &plan.centered_kernel_fourier_coefficients[0],
            )?),
            2 => Type1GpuExecution::TwoD(Type1GpuPlan2d::new(
                device,
                queue,
                &plan.config,
                plan.kernel,
                [plan.fine_grid_shape[0], plan.fine_grid_shape[1]],
                [
                    plan.centered_kernel_fourier_coefficients[0].as_slice(),
                    plan.centered_kernel_fourier_coefficients[1].as_slice(),
                ],
                gather_2d,
            )?),
            3 => Type1GpuExecution::ThreeD(Type1GpuPlan3d::new(
                device,
                queue,
                &plan.config,
                plan.kernel,
                [
                    plan.fine_grid_shape[0],
                    plan.fine_grid_shape[1],
                    plan.fine_grid_shape[2],
                ],
                [
                    plan.centered_kernel_fourier_coefficients[0].as_slice(),
                    plan.centered_kernel_fourier_coefficients[1].as_slice(),
                    plan.centered_kernel_fourier_coefficients[2].as_slice(),
                ],
                gather_3d,
            )?),
            _ => unreachable!("validated NUFFT GPU plans have at least one dimension"),
        };
        plan.gpu_type1 = Some(Box::new(gpu));
        Ok(plan)
    }

    /// Builds a reusable one-, two-, or three-dimensional type-2 GPU plan.
    ///
    /// Execution is entirely GPU-resident and records commands into a caller
    /// supplied encoder. The plan is device-specific and may be reused with
    /// different caller-owned point, coefficient, and output buffers.
    /// Executions using the same plan must retain queue order because the plan
    /// reuses its fine-grid scratch buffers.
    pub fn type2_gpu(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        config: NufftConfig,
    ) -> Result<Self> {
        if config.dimensions() > 3 {
            return Err(NufftError::GpuDimensionsUnsupported {
                kind: "type-2",
                actual: config.dimensions(),
                supported: 3,
            });
        }
        let mut plan = Self::new(NufftKind::Type2, config)?;
        let gpu = match plan.config.dimensions() {
            1 => Type2GpuExecution::OneD(Type2GpuPlan::new(
                device,
                queue,
                &plan.config,
                plan.kernel,
                plan.fine_grid_shape[0],
                &plan.centered_kernel_fourier_coefficients[0],
            )?),
            2 => Type2GpuExecution::TwoD(Type2GpuPlan2d::new(
                device,
                queue,
                &plan.config,
                plan.kernel,
                &plan.fine_grid_shape,
                &plan.centered_kernel_fourier_coefficients,
            )?),
            3 => Type2GpuExecution::ThreeD(Type2GpuPlan3d::new(
                device,
                queue,
                &plan.config,
                plan.kernel,
                &plan.fine_grid_shape,
                &plan.centered_kernel_fourier_coefficients,
            )?),
            _ => unreachable!("validated NUFFT GPU plans have at least one dimension"),
        };
        plan.gpu_type2 = Some(Box::new(gpu));
        Ok(plan)
    }

    fn new(kind: NufftKind, config: NufftConfig) -> Result<Self> {
        config.validate()?;
        let kernel = EsKernel::for_tolerance(config.eps(), config.sigma())?;
        let fine_grid_shape = config
            .n_modes()
            .iter()
            .map(|&length| select_fine_grid_size(length, config.sigma(), kernel.width()))
            .collect::<Result<Vec<_>>>()?;
        wgpu_fft::FftConfig::new_nd(fine_grid_shape.clone())
            .validate()
            .map_err(|source| NufftError::FftShapeUnsupported {
                stage: "oversampled NUFFT fine-grid shape",
                source,
            })?;
        let centered_kernel_fourier_coefficients = fine_grid_shape
            .iter()
            .map(|&length| kernel.centered_fourier_coefficients(length))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            kind,
            config,
            kernel,
            fine_grid_shape,
            centered_kernel_fourier_coefficients,
            gpu_type1: None,
            gpu_type2: None,
        })
    }

    pub fn kind(&self) -> NufftKind {
        self.kind
    }

    pub fn config(&self) -> &NufftConfig {
        &self.config
    }

    pub fn kernel(&self) -> EsKernel {
        self.kernel
    }

    pub fn fine_grid_shape(&self) -> &[usize] {
        &self.fine_grid_shape
    }

    /// Returns the raw centered-grid half-spectrum for one axis. Values include
    /// the `(-1)^k` grid-origin phase.
    pub fn centered_kernel_fourier_coefficients(&self, axis: usize) -> Option<&[f64]> {
        self.centered_kernel_fourier_coefficients
            .get(axis)
            .map(Vec::as_slice)
    }

    /// Whether this plan owns the GPU resources needed for its transform kind.
    pub fn is_gpu_ready(&self) -> bool {
        match self.kind {
            NufftKind::Type1 => self.gpu_type1.is_some(),
            NufftKind::Type2 => self.gpu_type2.is_some(),
        }
    }

    /// Required bytes for `point_count` scalar `f32` coordinates in the
    /// legacy one-dimensional layout.
    pub fn required_type1_point_buffer_size_bytes(point_count: usize) -> Result<u64> {
        Type1GpuPlan::point_buffer_size_bytes(point_count)
    }

    /// Required bytes for this plan's point-major `f32` coordinate buffer.
    ///
    /// A point occupies one scalar per configured dimension, so a 3D plan
    /// expects `[x0, y0, z0, x1, y1, z1, ...]`.
    pub fn required_point_buffer_size_bytes(&self, point_count: usize) -> Result<u64> {
        let coordinate_count = point_count.checked_mul(self.config.dimensions()).ok_or(
            NufftError::LengthOverflow {
                context: "GPU point-coordinate count",
            },
        )?;
        u64::try_from(coordinate_count)
            .ok()
            .and_then(|count| count.checked_mul(4))
            .ok_or(NufftError::LengthOverflow {
                context: "GPU point-coordinate buffer size",
            })
    }

    /// Required bytes for `point_count` interleaved-complex `f32` strengths.
    pub fn required_type1_strength_buffer_size_bytes(point_count: usize) -> Result<u64> {
        Type1GpuPlan::strength_buffer_size_bytes(point_count)
    }

    /// Required bytes for this plan's interleaved-complex `f32` type-1 output.
    pub fn required_type1_output_buffer_size_bytes(&self) -> Result<u64> {
        Type1GpuPlan::complex_buffer_size_bytes(
            "type-1 Fourier mode output buffer",
            self.config.mode_count()?,
        )
    }

    /// Records a one-, two-, or three-dimensional type-1 NUFFT into `encoder` without
    /// submitting or reading data back to the host.
    ///
    /// `points` stores point-major `f32` coordinates (one scalar per configured
    /// dimension). `strengths` and
    /// `output` store interleaved complex values as `(re, im)` `f32` pairs. All
    /// buffers used by a nonempty transform require `STORAGE` usage and must
    /// belong to the plan's device. Coordinates must be finite and lie in
    /// the documented `[-3*pi, 3*pi]` interval. For zero points, the
    /// point and strength buffers are ignored while every output mode is
    /// overwritten with zero.
    pub fn encode_type1_gpu(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        point_count: usize,
        points: &wgpu::Buffer,
        strengths: &wgpu::Buffer,
        output: &wgpu::Buffer,
    ) -> Result<()> {
        if self.kind != NufftKind::Type1 {
            return Err(NufftError::GpuExecutionUnavailable {
                kind: "type-1",
                reason: "the plan was built for type-2 execution",
            });
        }
        let gpu = self
            .gpu_type1
            .as_deref()
            .ok_or(NufftError::GpuExecutionUnavailable {
                kind: "type-1",
                reason: "construct the plan with NufftPlan::type1_gpu",
            })?;
        gpu.encode(device, encoder, point_count, points, strengths, output)
    }

    /// Records a type-1 execution with GPU timestamp queries around its
    /// logical stages.
    ///
    /// This diagnostic entry point is available only with the
    /// `gpu-profiling` crate feature. The device must have been requested with
    /// [`wgpu::Features::TIMESTAMP_QUERY`], and `query_set` must be a timestamp
    /// query set with at least [`Self::gpu_profile_query_count`] entries
    /// available starting at `first_query`. No query resolve or readback is
    /// recorded by this method.
    #[cfg(feature = "gpu-profiling")]
    pub fn encode_type1_gpu_profiled(
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
        self.validate_gpu_profiling(device, "type-1")?;
        if self.kind != NufftKind::Type1 {
            return Err(NufftError::GpuExecutionUnavailable {
                kind: "type-1",
                reason: "the plan was built for type-2 execution",
            });
        }
        let gpu = self
            .gpu_type1
            .as_deref()
            .ok_or(NufftError::GpuExecutionUnavailable {
                kind: "type-1",
                reason: "construct the plan with NufftPlan::type1_gpu",
            })?;
        gpu.encode_profiled(
            device,
            encoder,
            point_count,
            points,
            strengths,
            output,
            query_set,
            first_query,
        )
    }

    /// Required bytes for `point_count` scalar `f32` coordinates in the
    /// legacy one-dimensional layout.
    pub fn required_type2_point_buffer_size_bytes(point_count: usize) -> Result<u64> {
        Type2GpuPlan::point_buffer_size_bytes(point_count)
    }

    /// Required bytes for this plan's interleaved-complex `f32` coefficients.
    pub fn required_type2_coefficient_buffer_size_bytes(&self) -> Result<u64> {
        Type2GpuPlan::complex_buffer_size_bytes(
            "type-2 Fourier coefficient buffer",
            self.config.mode_count()?,
        )
    }

    /// Required bytes for `point_count` interleaved-complex `f32` outputs.
    pub fn required_type2_output_buffer_size_bytes(point_count: usize) -> Result<u64> {
        Type2GpuPlan::complex_buffer_size_bytes("type-2 output buffer", point_count)
    }

    /// Records a one-, two-, or three-dimensional type-2 NUFFT into `encoder` without
    /// submitting or reading data back to the host.
    ///
    /// `points` stores point-major `f32` coordinates (one scalar per configured
    /// dimension). `coefficients`
    /// and `output` store interleaved complex values as `(re, im)` `f32`
    /// pairs. All three buffers require `STORAGE` usage and must belong to the
    /// same device used to construct the plan. Coordinates must be finite and
    /// lie in the documented `[-3*pi, 3*pi]` interval.
    pub fn encode_type2_gpu(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        point_count: usize,
        points: &wgpu::Buffer,
        coefficients: &wgpu::Buffer,
        output: &wgpu::Buffer,
    ) -> Result<()> {
        if self.kind != NufftKind::Type2 {
            return Err(NufftError::GpuExecutionUnavailable {
                kind: "type-2",
                reason: "the plan was built for type-1 execution",
            });
        }
        let gpu = self
            .gpu_type2
            .as_deref()
            .ok_or(NufftError::GpuExecutionUnavailable {
                kind: "type-2",
                reason: "construct the plan with NufftPlan::type2_gpu",
            })?;
        gpu.encode(device, encoder, point_count, points, coefficients, output)
    }

    /// Records a type-2 execution with GPU timestamp queries around its
    /// logical stages. See [`Self::encode_type1_gpu_profiled`] for query-set
    /// requirements.
    #[cfg(feature = "gpu-profiling")]
    pub fn encode_type2_gpu_profiled(
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
        self.validate_gpu_profiling(device, "type-2")?;
        if self.kind != NufftKind::Type2 {
            return Err(NufftError::GpuExecutionUnavailable {
                kind: "type-2",
                reason: "the plan was built for type-1 execution",
            });
        }
        let gpu = self
            .gpu_type2
            .as_deref()
            .ok_or(NufftError::GpuExecutionUnavailable {
                kind: "type-2",
                reason: "construct the plan with NufftPlan::type2_gpu",
            })?;
        gpu.encode_profiled(
            device,
            encoder,
            point_count,
            points,
            coefficients,
            output,
            query_set,
            first_query,
        )
    }

    /// Number of timestamp-query slots used by one profiled execution.
    #[cfg(feature = "gpu-profiling")]
    pub fn gpu_profile_query_count(&self) -> u32 {
        match self.kind {
            NufftKind::Type1 => NufftGpuProfileLayout::type1(0)
                .expect("the fixed type-1 profile layout must be valid")
                .query_count(),
            NufftKind::Type2 => NufftGpuProfileLayout::type2(0)
                .expect("the fixed type-2 profile layout must be valid")
                .query_count(),
        }
    }

    /// Diagnostics for the wgpu-fft plan embedded in a GPU-ready NUFFT plan.
    #[cfg(feature = "gpu-profiling")]
    pub fn gpu_fft_diagnostics(&self) -> Option<wgpu_fft::FftDiagnostics> {
        match self.kind {
            NufftKind::Type1 => self
                .gpu_type1
                .as_deref()
                .map(Type1GpuExecution::fft_diagnostics),
            NufftKind::Type2 => self
                .gpu_type2
                .as_deref()
                .map(Type2GpuExecution::fft_diagnostics),
        }
    }

    #[cfg(feature = "gpu-profiling")]
    fn validate_gpu_profiling(&self, device: &wgpu::Device, kind: &'static str) -> Result<()> {
        if device.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
            Ok(())
        } else {
            Err(NufftError::GpuExecutionUnavailable {
                kind,
                reason: "request the device with wgpu::Features::TIMESTAMP_QUERY",
            })
        }
    }

    /// Executes the exact direct CPU fallback for this plan's transform kind.
    pub fn execute_direct_f64(
        &self,
        coordinates: &[f64],
        input: &[Complex64],
    ) -> Result<Vec<Complex64>> {
        match self.kind {
            NufftKind::Type1 => reference_type1_f64(&self.config, coordinates, input),
            NufftKind::Type2 => reference_type2_f64(&self.config, coordinates, input),
        }
    }
}
