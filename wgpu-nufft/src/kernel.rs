use crate::error::{NufftError, Result};
use std::f64::consts::PI;

pub const MIN_ES_KERNEL_WIDTH: usize = 2;
pub const MAX_ES_KERNEL_WIDTH: usize = 16;

/// Fixed-width exponential-of-semicircle spreading kernel.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EsKernel {
    width: usize,
    beta: f64,
    sigma: f64,
}

impl EsKernel {
    /// Selects legacy FINUFFT/cuFINUFFT ES parameters for a requested tolerance.
    pub fn for_tolerance(eps: f64, sigma: f64) -> Result<Self> {
        validate_tolerance(eps)?;
        validate_sigma(sigma)?;

        let width_float = if sigma == 2.0 {
            (10.0 / eps).log10().ceil()
        } else {
            (1.0 / eps).ln() / (PI * (1.0 - 1.0 / sigma).sqrt())
        };
        let width = (width_float.ceil() as usize).max(MIN_ES_KERNEL_WIDTH);
        if width > MAX_ES_KERNEL_WIDTH {
            return Err(NufftError::KernelWidthUnsupported {
                eps,
                width,
                maximum: MAX_ES_KERNEL_WIDTH,
            });
        }

        let beta = if sigma == 2.0 {
            let beta_over_width = match width {
                2 => 2.20,
                3 => 2.26,
                4 => 2.38,
                _ => 2.30,
            };
            beta_over_width * width as f64
        } else {
            0.97 * PI * width as f64 * (1.0 - 1.0 / (2.0 * sigma))
        };

        Ok(Self { width, beta, sigma })
    }

    pub fn width(self) -> usize {
        self.width
    }

    pub fn beta(self) -> f64 {
        self.beta
    }

    pub fn sigma(self) -> f64 {
        self.sigma
    }

    pub fn half_width(self) -> f64 {
        self.width as f64 * 0.5
    }

    /// Evaluates the CUDA-compatible normalized ES kernel in grid-point units.
    /// The compact-support endpoint is defined as zero.
    pub fn evaluate(self, distance: f64) -> f64 {
        let absolute = distance.abs();
        if !absolute.is_finite() || absolute >= self.half_width() {
            return 0.0;
        }
        let scaled = 2.0 * absolute / self.width as f64;
        let radicand = 1.0 - scaled * scaled;
        (self.beta * (radicand.sqrt() - 1.0)).exp()
    }

    /// Computes FINUFFT's centered-grid half-spectrum `phi_hat[k]`, including
    /// the `(-1)^k` phase associated with a grid whose coordinate zero is `-pi`.
    pub fn centered_fourier_coefficients(self, fine_grid_length: usize) -> Result<Vec<f64>> {
        if !fine_grid_length.is_multiple_of(2) {
            return Err(NufftError::FineGridMustBeEven {
                length: fine_grid_length,
            });
        }
        if fine_grid_length == 0 {
            return Err(NufftError::ZeroMode { axis: 0 });
        }

        let q = (2.0 + 3.0 * self.half_width()).floor() as usize;
        let (nodes, weights) = gauss_legendre(2 * q);
        let mut quadrature_nodes = Vec::with_capacity(q);
        let mut weighted_kernel = Vec::with_capacity(q);
        for index in 0..q {
            let node = self.half_width() * nodes[index];
            quadrature_nodes.push(node);
            weighted_kernel.push(self.half_width() * weights[index] * self.evaluate(node));
        }

        let coefficient_count = fine_grid_length / 2 + 1;
        let mut coefficients = Vec::new();
        coefficients
            .try_reserve_exact(coefficient_count)
            .map_err(|_| NufftError::HostAllocationFailed {
                buffer: "centered ES kernel Fourier coefficients",
                elements: coefficient_count,
            })?;
        for mode in 0..=fine_grid_length / 2 {
            let frequency = 2.0 * PI * mode as f64 / fine_grid_length as f64;
            let mut value = quadrature_nodes
                .iter()
                .zip(&weighted_kernel)
                .map(|(&node, &weight)| 2.0 * weight * (frequency * node).cos())
                .sum::<f64>();
            if !mode.is_multiple_of(2) {
                value = -value;
            }
            coefficients.push(value);
        }
        Ok(coefficients)
    }
}

/// Chooses the smallest even length at least `max(ceil(sigma*N), 2*w)` that
/// `wgpu-fft` reports as directly factorable.
pub fn select_fine_grid_size(n_modes: usize, sigma: f64, kernel_width: usize) -> Result<usize> {
    if n_modes == 0 {
        return Err(NufftError::ZeroMode { axis: 0 });
    }
    if !(MIN_ES_KERNEL_WIDTH..=MAX_ES_KERNEL_WIDTH).contains(&kernel_width) {
        return Err(NufftError::InvalidKernelWidth {
            width: kernel_width,
            minimum: MIN_ES_KERNEL_WIDTH,
            maximum: MAX_ES_KERNEL_WIDTH,
        });
    }
    validate_sigma(sigma)?;

    let scaled_modes = if sigma == 2.0 {
        n_modes.checked_mul(2).ok_or(NufftError::LengthOverflow {
            context: "upsampled fine-grid length",
        })?
    } else {
        let scaled = sigma * n_modes as f64;
        if !scaled.is_finite() || scaled > usize::MAX as f64 {
            return Err(NufftError::LengthOverflow {
                context: "upsampled fine-grid length",
            });
        }
        scaled.ceil() as usize
    };
    let support_floor = kernel_width
        .checked_mul(2)
        .ok_or(NufftError::LengthOverflow {
            context: "kernel-support fine-grid floor",
        })?;
    let minimum = scaled_modes.max(support_floor);
    let mut candidate = if minimum.is_multiple_of(2) {
        minimum
    } else {
        minimum.checked_add(1).ok_or(NufftError::LengthOverflow {
            context: "even fine-grid length",
        })?
    };

    loop {
        if wgpu_fft::runtime::factor_supported_length(candidate).is_ok() {
            return Ok(candidate);
        }
        candidate = candidate.checked_add(2).ok_or(NufftError::LengthOverflow {
            context: "next supported fine-grid length",
        })?;
    }
}

fn validate_tolerance(eps: f64) -> Result<()> {
    if !eps.is_finite() {
        return Err(NufftError::InvalidTolerance {
            eps,
            reason: "tolerance must be finite",
        });
    }
    if eps <= 0.0 || eps >= 1.0 {
        return Err(NufftError::InvalidTolerance {
            eps,
            reason: "tolerance must lie strictly between zero and one",
        });
    }
    Ok(())
}

fn validate_sigma(sigma: f64) -> Result<()> {
    if !sigma.is_finite() {
        return Err(NufftError::InvalidSigma {
            sigma,
            reason: "upsampling factor must be finite",
        });
    }
    if sigma <= 1.0 {
        return Err(NufftError::InvalidSigma {
            sigma,
            reason: "upsampling factor must be greater than one",
        });
    }
    Ok(())
}

fn gauss_legendre(order: usize) -> (Vec<f64>, Vec<f64>) {
    let mut nodes = vec![0.0; order];
    let mut weights = vec![0.0; order];
    let pairs = order.div_ceil(2);
    for index in 0..pairs {
        let mut root = (PI * (index as f64 + 0.75) / (order as f64 + 0.5)).cos();
        loop {
            let (polynomial, previous) = legendre_pair(order, root);
            let derivative = order as f64 * (root * polynomial - previous) / (root * root - 1.0);
            let next = root - polynomial / derivative;
            if (next - root).abs() <= 4.0 * f64::EPSILON {
                root = next;
                break;
            }
            root = next;
        }
        let (polynomial, previous) = legendre_pair(order, root);
        let derivative = order as f64 * (root * polynomial - previous) / (root * root - 1.0);
        let weight = 2.0 / ((1.0 - root * root) * derivative * derivative);
        nodes[index] = -root;
        nodes[order - 1 - index] = root;
        weights[index] = weight;
        weights[order - 1 - index] = weight;
    }
    (nodes, weights)
}

fn legendre_pair(order: usize, x: f64) -> (f64, f64) {
    let mut previous = 1.0;
    if order == 0 {
        return (previous, 0.0);
    }
    let mut current = x;
    for degree in 2..=order {
        let next = ((2 * degree - 1) as f64 * x * current - (degree - 1) as f64 * previous)
            / degree as f64;
        previous = current;
        current = next;
    }
    (current, previous)
}
