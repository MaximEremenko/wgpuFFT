use crate::config::{ModeOrder, NufftConfig};
use crate::error::{NufftError, Result};
use crate::Complex64;
use std::f64::consts::PI;

const MAX_COORDINATE_MAGNITUDE: f64 = 3.0 * PI;

/// Direct `O(M*N)` type-1 NDFT in the crate's mode and sign conventions.
pub fn reference_type1_f64(
    config: &NufftConfig,
    coordinates: &[f64],
    strengths: &[Complex64],
) -> Result<Vec<Complex64>> {
    config.validate()?;
    let point_count = validate_coordinates(config.dimensions(), coordinates)?;
    let strength_count =
        point_count
            .checked_mul(config.batch())
            .ok_or(NufftError::LengthOverflow {
                context: "batched type-1 strength count",
            })?;
    if strengths.len() != strength_count {
        return Err(NufftError::InputLength {
            input: "type-1 strengths",
            expected: strength_count,
            actual: strengths.len(),
        });
    }
    validate_complex("type-1 strengths", strengths)?;

    let mode_count = config.mode_count()?;
    let output_count =
        mode_count
            .checked_mul(config.batch())
            .ok_or(NufftError::LengthOverflow {
                context: "batched type-1 direct output count",
            })?;
    let mut output = zeroed_complex("type-1 direct output", output_count)?;
    for transform in 0..config.batch() {
        let strength_base = transform * point_count;
        let output_base = transform * mode_count;
        for linear_mode in 0..mode_count {
            let modes = modes_from_linear(linear_mode, config.n_modes(), config.mode_order());
            let mut sum = Complex64::default();
            for point in 0..point_count {
                let phase = point_phase(
                    point,
                    config.dimensions(),
                    coordinates,
                    &modes[..config.dimensions()],
                );
                add_rotated(
                    &mut sum,
                    strengths[strength_base + point],
                    config.sign().multiplier() * phase,
                );
            }
            output[output_base + linear_mode] = sum;
        }
    }
    Ok(output)
}

/// Direct `O(M*N)` type-2 NDFT in the crate's mode and sign conventions.
pub fn reference_type2_f64(
    config: &NufftConfig,
    coordinates: &[f64],
    coefficients: &[Complex64],
) -> Result<Vec<Complex64>> {
    config.validate()?;
    let point_count = validate_coordinates(config.dimensions(), coordinates)?;
    let mode_count = config.mode_count()?;
    let coefficient_count =
        mode_count
            .checked_mul(config.batch())
            .ok_or(NufftError::LengthOverflow {
                context: "batched type-2 Fourier coefficient count",
            })?;
    if coefficients.len() != coefficient_count {
        return Err(NufftError::InputLength {
            input: "type-2 Fourier coefficients",
            expected: coefficient_count,
            actual: coefficients.len(),
        });
    }
    validate_complex("type-2 Fourier coefficients", coefficients)?;

    let output_count =
        point_count
            .checked_mul(config.batch())
            .ok_or(NufftError::LengthOverflow {
                context: "batched type-2 direct output count",
            })?;
    let mut output = zeroed_complex("type-2 direct output", output_count)?;
    for transform in 0..config.batch() {
        let coefficient_base = transform * mode_count;
        let output_base = transform * point_count;
        for point in 0..point_count {
            let mut sum = Complex64::default();
            for linear_mode in 0..mode_count {
                let modes = modes_from_linear(linear_mode, config.n_modes(), config.mode_order());
                let phase = point_phase(
                    point,
                    config.dimensions(),
                    coordinates,
                    &modes[..config.dimensions()],
                );
                add_rotated(
                    &mut sum,
                    coefficients[coefficient_base + linear_mode],
                    config.sign().multiplier() * phase,
                );
            }
            output[output_base + point] = sum;
        }
    }
    Ok(output)
}

pub(crate) fn mode_for_storage_index(order: ModeOrder, index: usize, length: usize) -> i64 {
    match order {
        ModeOrder::Centered => index as i64 - (length / 2) as i64,
        ModeOrder::Fft => {
            if index < length.div_ceil(2) {
                index as i64
            } else {
                index as i64 - length as i64
            }
        }
    }
}

fn validate_coordinates(dimensions: usize, coordinates: &[f64]) -> Result<usize> {
    if !coordinates.len().is_multiple_of(dimensions) {
        return Err(NufftError::CoordinateLength {
            actual: coordinates.len(),
            dimensions,
        });
    }
    for (index, &coordinate) in coordinates.iter().enumerate() {
        let point = index / dimensions;
        let axis = index % dimensions;
        if !coordinate.is_finite() {
            return Err(NufftError::NonFiniteCoordinate {
                point,
                axis,
                value: coordinate,
            });
        }
        if coordinate.abs() > MAX_COORDINATE_MAGNITUDE {
            return Err(NufftError::PointOutOfRange {
                point,
                axis,
                value: coordinate,
                maximum_magnitude: MAX_COORDINATE_MAGNITUDE,
            });
        }
    }
    Ok(coordinates.len() / dimensions)
}

fn validate_complex(input: &'static str, values: &[Complex64]) -> Result<()> {
    for (index, value) in values.iter().enumerate() {
        if !value.re.is_finite() || !value.im.is_finite() {
            return Err(NufftError::NonFiniteComplex { input, index });
        }
    }
    Ok(())
}

fn modes_from_linear(mut linear: usize, n_modes: &[usize], order: ModeOrder) -> [i64; 3] {
    let mut modes = [0; 3];
    for (axis, &length) in n_modes.iter().enumerate() {
        let index = linear % length;
        linear /= length;
        modes[axis] = mode_for_storage_index(order, index, length);
    }
    modes
}

fn zeroed_complex(buffer: &'static str, elements: usize) -> Result<Vec<Complex64>> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(elements)
        .map_err(|_| NufftError::HostAllocationFailed { buffer, elements })?;
    values.resize(elements, Complex64::default());
    Ok(values)
}

fn point_phase(point: usize, dimensions: usize, coordinates: &[f64], modes: &[i64]) -> f64 {
    let base = point * dimensions;
    modes
        .iter()
        .enumerate()
        .map(|(axis, &mode)| mode as f64 * coordinates[base + axis])
        .sum()
}

fn add_rotated(sum: &mut Complex64, value: Complex64, angle: f64) {
    let (sin, cos) = angle.sin_cos();
    sum.re += value.re * cos - value.im * sin;
    sum.im += value.re * sin + value.im * cos;
}

#[cfg(test)]
mod tests {
    use super::zeroed_complex;
    use crate::NufftError;

    #[test]
    fn direct_output_allocation_failure_is_structured() {
        assert!(matches!(
            zeroed_complex("test direct output", usize::MAX),
            Err(NufftError::HostAllocationFailed {
                buffer: "test direct output",
                elements: usize::MAX,
            })
        ));
    }
}
