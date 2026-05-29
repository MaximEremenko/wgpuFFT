use crate::config::NufftConfig;
use crate::direct::{reference_type1_f64, reference_type2_f64};
use crate::error::{NufftError, Result};
use crate::gpu::Type2GpuPlan;
use crate::gpu_type1::Type1GpuPlan;
use crate::kernel::{select_fine_grid_size, EsKernel};
use crate::Complex64;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum NufftKind {
    Type1,
    Type2,
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
    gpu_type1: Option<Box<Type1GpuPlan>>,
    gpu_type2: Option<Box<Type2GpuPlan>>,
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

    /// Builds a reusable 1D type-1 GPU plan.
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
        if config.dimensions() != 1 {
            return Err(NufftError::GpuDimensionsUnsupported {
                kind: "type-1",
                actual: config.dimensions(),
                supported: 1,
            });
        }
        let mut plan = Self::new(NufftKind::Type1, config)?;
        let gpu = Type1GpuPlan::new(
            device,
            queue,
            &plan.config,
            plan.kernel,
            plan.fine_grid_shape[0],
            &plan.centered_kernel_fourier_coefficients[0],
        )?;
        plan.gpu_type1 = Some(Box::new(gpu));
        Ok(plan)
    }

    /// Builds a reusable 1D type-2 GPU plan.
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
        if config.dimensions() != 1 {
            return Err(NufftError::GpuDimensionsUnsupported {
                kind: "type-2",
                actual: config.dimensions(),
                supported: 1,
            });
        }
        let mut plan = Self::new(NufftKind::Type2, config)?;
        let gpu = Type2GpuPlan::new(
            device,
            queue,
            &plan.config,
            plan.kernel,
            plan.fine_grid_shape[0],
            &plan.centered_kernel_fourier_coefficients[0],
        )?;
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

    /// Required bytes for `point_count` scalar `f32` type-1 coordinates.
    pub fn required_type1_point_buffer_size_bytes(point_count: usize) -> Result<u64> {
        Type1GpuPlan::point_buffer_size_bytes(point_count)
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

    /// Records a 1D type-1 NUFFT into `encoder` without submitting or reading
    /// data back to the host.
    ///
    /// `points` stores `point_count` scalar `f32` coordinates. `strengths` and
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

    /// Required bytes for `point_count` scalar `f32` coordinates.
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

    /// Records a 1D type-2 NUFFT into `encoder` without submitting or reading
    /// data back to the host.
    ///
    /// `points` stores `point_count` scalar `f32` coordinates. `coefficients`
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
