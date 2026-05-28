use crate::config::NufftConfig;
use crate::direct::{reference_type1_f64, reference_type2_f64};
use crate::error::{NufftError, Result};
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
/// Phase A executes the direct CPU fallback. Later GPU phases add private
/// pipeline and `wgpu-fft` plan state while preserving this public metadata.
#[derive(Debug)]
pub struct NufftPlan {
    kind: NufftKind,
    config: NufftConfig,
    kernel: EsKernel,
    fine_grid_shape: Vec<usize>,
    centered_kernel_fourier_coefficients: Vec<Vec<f64>>,
}

impl NufftPlan {
    pub fn type1(config: NufftConfig) -> Result<Self> {
        Self::new(NufftKind::Type1, config)
    }

    pub fn type2(config: NufftConfig) -> Result<Self> {
        Self::new(NufftKind::Type2, config)
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
