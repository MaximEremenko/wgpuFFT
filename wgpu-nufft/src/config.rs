use crate::error::{NufftError, Result};
use crate::kernel::EsKernel;

pub const DEFAULT_EPS: f64 = 1.0e-6;
pub const DEFAULT_SIGMA: f64 = 2.0;

/// Sign in the NUFFT exponential.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum NufftSign {
    #[default]
    Positive,
    Negative,
}

impl NufftSign {
    /// FINUFFT treats every nonnegative `isign` as positive and every negative
    /// value as negative.
    pub const fn from_isign(isign: i32) -> Self {
        if isign >= 0 {
            Self::Positive
        } else {
            Self::Negative
        }
    }

    pub const fn isign(self) -> i32 {
        match self {
            Self::Positive => 1,
            Self::Negative => -1,
        }
    }

    pub(crate) const fn multiplier(self) -> f64 {
        self.isign() as f64
    }
}

/// Storage order for uniform Fourier modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ModeOrder {
    /// Increasing centered indices: `-N/2, ..., N/2-1` for even `N`.
    #[default]
    Centered,
    /// FFT order: nonnegative modes first, followed by negative modes.
    Fft,
}

impl ModeOrder {
    pub const fn modeord(self) -> i32 {
        match self {
            Self::Centered => 0,
            Self::Fft => 1,
        }
    }
}

/// Mathematical configuration shared by type-1 and type-2 plans.
#[derive(Debug, Clone, PartialEq)]
pub struct NufftConfig {
    n_modes: Vec<usize>,
    eps: f64,
    sign: NufftSign,
    mode_order: ModeOrder,
    sigma: f64,
}

impl NufftConfig {
    /// Creates a configuration. Dimensionality is derived from `n_modes`.
    pub fn new(n_modes: impl Into<Vec<usize>>, eps: f64) -> Self {
        Self {
            n_modes: n_modes.into(),
            eps,
            sign: NufftSign::Positive,
            mode_order: ModeOrder::Centered,
            sigma: DEFAULT_SIGMA,
        }
    }

    pub fn with_isign(mut self, isign: i32) -> Self {
        self.sign = NufftSign::from_isign(isign);
        self
    }

    pub fn with_sign(mut self, sign: NufftSign) -> Self {
        self.sign = sign;
        self
    }

    pub fn with_mode_order(mut self, mode_order: ModeOrder) -> Self {
        self.mode_order = mode_order;
        self
    }

    /// Typed builder alias named after FINUFFT's `modeord` option.
    pub fn with_modeord(self, mode_order: ModeOrder) -> Self {
        self.with_mode_order(mode_order)
    }

    pub fn with_sigma(mut self, sigma: f64) -> Self {
        self.sigma = sigma;
        self
    }

    pub fn dimensions(&self) -> usize {
        self.n_modes.len()
    }

    pub fn n_modes(&self) -> &[usize] {
        &self.n_modes
    }

    pub fn eps(&self) -> f64 {
        self.eps
    }

    pub fn sign(&self) -> NufftSign {
        self.sign
    }

    pub fn isign(&self) -> i32 {
        self.sign.isign()
    }

    pub fn mode_order(&self) -> ModeOrder {
        self.mode_order
    }

    pub fn modeord(&self) -> i32 {
        self.mode_order.modeord()
    }

    pub fn sigma(&self) -> f64 {
        self.sigma
    }

    pub fn mode_count(&self) -> Result<usize> {
        self.n_modes.iter().try_fold(1usize, |count, &length| {
            count.checked_mul(length).ok_or(NufftError::LengthOverflow {
                context: "total Fourier mode count",
            })
        })
    }

    pub fn validate(&self) -> Result<()> {
        let dimensions = self.dimensions();
        if !(1..=3).contains(&dimensions) {
            return Err(NufftError::InvalidDimensions {
                actual: dimensions,
                minimum: 1,
                maximum: 3,
            });
        }
        for (axis, &length) in self.n_modes.iter().enumerate() {
            if length == 0 {
                return Err(NufftError::ZeroMode { axis });
            }
        }
        self.mode_count()?;
        // A singleton NUFFT axis is mathematically meaningful, but wgpu-fft
        // rejects selecting a length-one FFT axis in a nontrivial transform.
        // Use only nontrivial axes here to reuse its checked u32 index-space
        // validation without narrowing the NUFFT shape contract.
        let mut fft_shape = self
            .n_modes
            .iter()
            .copied()
            .filter(|&length| length > 1)
            .collect::<Vec<_>>();
        if fft_shape.is_empty() {
            fft_shape.push(1);
        }
        wgpu_fft::FftConfig::new_nd(fft_shape)
            .validate()
            .map_err(|source| NufftError::FftShapeUnsupported {
                stage: "requested NUFFT mode shape",
                source,
            })?;
        EsKernel::for_tolerance(self.eps, self.sigma)?;
        Ok(())
    }
}

impl Default for NufftConfig {
    fn default() -> Self {
        Self::new([1], DEFAULT_EPS)
    }
}
