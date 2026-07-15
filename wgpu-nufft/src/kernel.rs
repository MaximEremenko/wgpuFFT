use crate::error::{NufftError, Result};
use std::f64::consts::PI;

pub const MIN_ES_KERNEL_WIDTH: usize = 2;
pub const MAX_ES_KERNEL_WIDTH: usize = 16;
const MAX_HORNER_COEFFICIENTS: usize = 19;

/// Fixed-width exponential-of-semicircle spreading kernel.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EsKernel {
    width: usize,
    beta: f64,
    sigma: f64,
    eps: f64,
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

        Ok(Self {
            width,
            beta,
            sigma,
            eps,
        })
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

    pub fn eps(self) -> f64 {
        self.eps
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

    /// Fits FINUFFT's one-unit piecewise panels in host `f64` for a
    /// transcendental-free GPU Horner evaluator.
    pub(crate) fn horner_table(self) -> EsHornerTable {
        let fitted_count = (self.width + 3).min(MAX_HORNER_COEFFICIENTS);
        let mut coefficients = vec![0.0; self.width * fitted_count];
        for panel in 0..self.width {
            // FINUFFT fits the normalized kernel on a local Chebyshev
            // coordinate t in [-1, 1]. In grid units the corresponding
            // distance is (t + 2*j + 1 - width) / 2.
            let shift = (2 * panel + 1) as f64 - self.width as f64;
            let fitted = polynomial_fit_chebyshev(
                |local| self.evaluate((local + shift) * 0.5),
                fitted_count,
            );
            for (coefficient, value) in fitted.into_iter().enumerate() {
                coefficients[coefficient * self.width + panel] = value;
            }
        }
        // FINUFFT drops common leading rows which are insignificant at the
        // requested tolerance, while retaining enough terms for narrow and
        // high-order panels to remain well-conditioned.
        let minimum_count = 4.max(self.width.saturating_sub(4));
        let threshold = 0.05 * self.eps;
        let mut leading_rows = 0;
        while fitted_count - leading_rows > minimum_count
            && coefficients[leading_rows * self.width..(leading_rows + 1) * self.width]
                .iter()
                .all(|value| value.abs() < threshold)
        {
            leading_rows += 1;
        }
        if leading_rows != 0 {
            coefficients.drain(..leading_rows * self.width);
        }
        EsHornerTable {
            width: self.width,
            coefficient_count: fitted_count - leading_rows,
            coefficients,
        }
    }

    /// Computes FINUFFT's centered-grid half-spectrum `phi_hat[k]`, including
    /// the `(-1)^k` phase associated with a grid whose coordinate zero is `-pi`.
    pub fn centered_fourier_coefficients(self, fine_grid_length: usize) -> Result<Vec<f64>> {
        self.centered_fourier_coefficients_using(fine_grid_length, |distance| {
            self.evaluate(distance)
        })
    }

    /// Computes deconvolution coefficients from the same polynomial used by
    /// native-f64 and double-float GPU kernel evaluation.
    pub(crate) fn centered_fourier_coefficients_horner(
        self,
        fine_grid_length: usize,
        table: &EsHornerTable,
    ) -> Result<Vec<f64>> {
        debug_assert_eq!(table.width(), self.width);
        self.centered_fourier_coefficients_using(fine_grid_length, |distance| {
            table.evaluate(distance)
        })
    }

    fn centered_fourier_coefficients_using(
        self,
        fine_grid_length: usize,
        evaluate: impl Fn(f64) -> f64,
    ) -> Result<Vec<f64>> {
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
            weighted_kernel.push(self.half_width() * weights[index] * evaluate(node));
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

/// Host-generated, coefficient-major piecewise-Horner ES table.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct EsHornerTable {
    width: usize,
    coefficient_count: usize,
    coefficients: Vec<f64>,
}

impl EsHornerTable {
    pub(crate) fn width(&self) -> usize {
        self.width
    }

    pub(crate) fn coefficient_count(&self) -> usize {
        self.coefficient_count
    }

    pub(crate) fn coefficients(&self) -> &[f64] {
        &self.coefficients
    }

    pub(crate) fn evaluate(&self, distance: f64) -> f64 {
        let half_width = self.width as f64 * 0.5;
        if !distance.is_finite() || distance.abs() >= half_width {
            return 0.0;
        }
        // FINUFFT's panels use (lower, upper], hence an exact internal panel
        // boundary belongs to the panel on its left.
        let panel = ((distance + half_width).ceil() as isize - 1).clamp(0, self.width as isize - 1)
            as usize;
        let local = 2.0 * (distance - panel as f64) + (self.width - 1) as f64;
        let mut value = 0.0;
        for coefficient in 0..self.coefficient_count {
            value = value * local + self.coefficients[coefficient * self.width + panel];
        }
        value
    }
}

/// FINUFFT's type-1-Chebyshev interpolation followed by a Newton-to-monomial
/// conversion. Returned coefficients are in descending Horner order.
fn polynomial_fit_chebyshev(function: impl Fn(f64) -> f64, coefficient_count: usize) -> Vec<f64> {
    debug_assert!(coefficient_count > 0);
    let count = coefficient_count;
    let nodes = (0..count)
        .map(|index| (((2 * index + 1) as f64 * PI) / (2 * count) as f64).cos())
        .collect::<Vec<_>>();
    let mut divided = nodes.iter().map(|&node| function(node)).collect::<Vec<_>>();
    for order in 1..count {
        for index in (order..count).rev() {
            divided[index] =
                (divided[index] - divided[index - 1]) / (nodes[index] - nodes[index - order]);
        }
    }

    // Convert Newton form to monomials in ascending order.
    let mut coefficients = vec![0.0; count];
    let mut basis = vec![1.0];
    coefficients[0] = divided[0];
    for order in 1..count {
        let node = nodes[order - 1];
        let mut next_basis = vec![0.0; basis.len() + 1];
        for (power, &value) in basis.iter().enumerate() {
            next_basis[power] -= node * value;
            next_basis[power + 1] += value;
        }
        basis = next_basis;
        for (power, &value) in basis.iter().enumerate() {
            coefficients[power] += divided[order] * value;
        }
    }
    coefficients.reverse();
    coefficients
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_horner_tables_match_the_es_definition_across_supported_widths() {
        for eps in [1.0e-1, 1.0e-2, 1.0e-4, 1.0e-6, 1.0e-9, 1.0e-12] {
            let kernel = EsKernel::for_tolerance(eps, 2.0).unwrap();
            let table = kernel.horner_table();
            assert_eq!(table.width(), kernel.width());
            assert_eq!(
                table.coefficients().len(),
                table.width() * table.coefficient_count()
            );
            assert!(table.coefficient_count() <= (kernel.width() + 3).min(19));
            assert!(table.coefficient_count() >= 4.max(kernel.width().saturating_sub(4)));
            let half_width = kernel.half_width();
            let mut maximum_absolute_error = 0.0_f64;
            for sample in 0..=20_000 {
                let distance = -half_width + 2.0 * half_width * sample as f64 / 20_000.0;
                maximum_absolute_error = maximum_absolute_error
                    .max((table.evaluate(distance) - kernel.evaluate(distance)).abs());
            }
            eprintln!(
                "HORNER_TABLE eps={eps:.1e} width={} coefficients={} max_abs={maximum_absolute_error:.9e}",
                kernel.width(),
                table.coefficient_count()
            );
            assert!(
                maximum_absolute_error <= eps * 0.1,
                "eps={eps} width={} Horner absolute error {maximum_absolute_error}",
                kernel.width()
            );
        }
    }

    #[test]
    fn host_horner_endpoint_and_nonfinite_contract_matches_exact_kernel() {
        let kernel = EsKernel::for_tolerance(1.0e-12, 2.0).unwrap();
        let table = kernel.horner_table();
        for distance in [
            -kernel.half_width(),
            kernel.half_width(),
            f64::NEG_INFINITY,
            f64::INFINITY,
            f64::NAN,
        ] {
            assert_eq!(table.evaluate(distance), 0.0);
        }
    }

    #[test]
    fn horner_fourier_coefficients_integrate_the_polynomial_not_exact_exp() {
        let kernel = EsKernel::for_tolerance(1.0e-12, 2.0).unwrap();
        let table = kernel.horner_table();
        let actual = kernel
            .centered_fourier_coefficients_horner(128, &table)
            .unwrap();

        // Independent dense midpoint integration of the fitted polynomial.
        let samples = 200_000usize;
        let dx = kernel.width() as f64 / samples as f64;
        for mode in [0usize, 1, 7, 31, 64] {
            let frequency = 2.0 * PI * mode as f64 / 128.0;
            let mut integral = 0.0;
            for sample in 0..samples {
                let x = -kernel.half_width() + (sample as f64 + 0.5) * dx;
                integral += table.evaluate(x) * (frequency * x).cos();
            }
            let mut expected = integral * dx;
            if !mode.is_multiple_of(2) {
                expected = -expected;
            }
            assert!(
                (actual[mode] - expected).abs() <= 2.0e-12,
                "mode={mode} quadrature={} midpoint={expected}",
                actual[mode]
            );
        }
    }
}
