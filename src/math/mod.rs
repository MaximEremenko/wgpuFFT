use crate::config::{FftConfig, FftDirection, Normalization};
use crate::error::Result;
use bytemuck::{Pod, Zeroable};

mod double_float;

pub use double_float::{
    quick_two_sum_f32, split_f32, two_prod_f32, two_sum_f32, ComplexDoubleFloat, DoubleFloat,
};

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Pod, Zeroable)]
pub struct Complex32 {
    pub re: f32,
    pub im: f32,
}

impl Complex32 {
    pub const fn new(re: f32, im: f32) -> Self {
        Self { re, im }
    }

    pub fn abs_diff(self, other: Self) -> f32 {
        (self.re - other.re).abs().max((self.im - other.im).abs())
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Pod, Zeroable)]
pub struct Complex64 {
    pub re: f64,
    pub im: f64,
}

impl Complex64 {
    pub const fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    pub fn abs_diff(self, other: Self) -> f64 {
        (self.re - other.re).abs().max((self.im - other.im).abs())
    }
}

pub fn from_interleaved_f32(values: &[f32]) -> Vec<Complex32> {
    values
        .chunks_exact(2)
        .map(|pair| Complex32::new(pair[0], pair[1]))
        .collect()
}

pub fn to_interleaved_f32(values: &[Complex32]) -> Vec<f32> {
    values
        .iter()
        .flat_map(|value| [value.re, value.im])
        .collect()
}

pub fn from_interleaved_f64(values: &[f64]) -> Vec<Complex64> {
    values
        .chunks_exact(2)
        .map(|pair| Complex64::new(pair[0], pair[1]))
        .collect()
}

pub fn to_interleaved_f64(values: &[Complex64]) -> Vec<f64> {
    values
        .iter()
        .flat_map(|value| [value.re, value.im])
        .collect()
}

pub fn reference_dft(input: &[Complex32], config: FftConfig) -> Result<Vec<Complex32>> {
    reference_c2c_nd(input, &config)
}

pub fn reference_r2c_packed_interleaved(real: &[f32], config: &FftConfig) -> Result<Vec<f32>> {
    validate_real_reference_config(config, FftDirection::Forward)?;
    let total_real = config.total_complex_len()?;
    assert_eq!(
        real.len(),
        total_real,
        "reference real input length must match FftConfig::total_complex_len"
    );

    let complex = real
        .iter()
        .map(|&value| Complex32::new(value, 0.0))
        .collect::<Vec<_>>();
    let full = reference_c2c_nd(&complex, config)?;
    let packed_shape = packed_shape_for(config.shape());
    let packed_total = checked_product(&packed_shape) * config.batch();
    let packed_strides = strides_for_shape(&packed_shape);
    let full_strides = strides_for_shape(config.shape());
    let logical_full = config.logical_complex_len()?;
    let logical_packed = checked_product(&packed_shape);
    let mut output = Vec::with_capacity(packed_total * 2);

    for batch in 0..config.batch() {
        for packed_index in 0..logical_packed {
            let coords = coords_from_linear(packed_index, &packed_shape, &packed_strides);
            let full_index = linear_from_coords(&coords, &full_strides);
            let value = full[batch * logical_full + full_index];
            output.extend([value.re, value.im]);
        }
    }

    Ok(output)
}

pub fn reference_c2r_from_packed_interleaved(
    packed: &[f32],
    config: &FftConfig,
) -> Result<Vec<f32>> {
    validate_real_reference_config(config, FftDirection::Inverse)?;
    let packed_shape = packed_shape_for(config.shape());
    let logical_packed = checked_product(&packed_shape);
    let packed_total = logical_packed * config.batch();
    assert_eq!(
        packed.len(),
        packed_total * 2,
        "reference packed input length must match packed complex length"
    );

    let packed_complex = from_interleaved_f32(packed);
    let mut full = vec![Complex32::default(); config.total_complex_len()?];
    let packed_strides = strides_for_shape(&packed_shape);
    let full_strides = strides_for_shape(config.shape());
    let logical_full = config.logical_complex_len()?;

    for batch in 0..config.batch() {
        for full_index in 0..logical_full {
            let coords = coords_from_linear(full_index, config.shape(), &full_strides);
            let packed_coords = packed_coords_for_full_spectrum(&coords, config.shape());
            let packed_index = linear_from_coords(&packed_coords, &packed_strides);
            let mut value = packed_complex[batch * logical_packed + packed_index];
            if coords[0] >= packed_shape[0] {
                value.im = -value.im;
            }
            if is_self_conjugate_bin(&coords, config.shape()) {
                value.im = 0.0;
            }
            full[batch * logical_full + full_index] = value;
        }
    }

    let inverse = reference_c2c_nd(&full, config)?;
    Ok(inverse.into_iter().map(|value| value.re).collect())
}

pub fn reference_c2c_nd(input: &[Complex32], config: &FftConfig) -> Result<Vec<Complex32>> {
    config.validate()?;
    let total_complex = config.total_complex_len()?;
    assert_eq!(
        input.len(),
        total_complex,
        "reference input length must match FftConfig::total_complex_len"
    );

    let sign = match config.direction() {
        FftDirection::Forward => -1.0,
        FftDirection::Inverse => 1.0,
    };
    let tau = std::f32::consts::TAU;
    let shape = config.shape();
    let strides = strides_for_shape(shape);
    let logical_complex_len = config.logical_complex_len()?;

    let mut values = input.to_vec();
    for (axis_index, &axis) in config.axes().iter().enumerate() {
        let axis_len = shape[axis];
        let axis_len_f32 = axis_len as f32;
        let stride = strides[axis];
        let lines_per_batch = logical_complex_len / axis_len;
        let scale = if axis_index + 1 == config.axes().len() {
            config.scale()?
        } else {
            1.0
        };
        let mut output = vec![Complex32::default(); total_complex];

        for batch in 0..config.batch() {
            let batch_base = batch * logical_complex_len;
            for line in 0..lines_per_batch {
                let base = batch_base + line_base_for_axis(line, axis, shape, &strides);
                for k in 0..axis_len {
                    let mut sum = Complex32::default();
                    for n in 0..axis_len {
                        let value = values[base + n * stride];
                        let angle = sign * tau * (k as f32) * (n as f32) / axis_len_f32;
                        let (sin, cos) = angle.sin_cos();
                        sum.re += value.re * cos - value.im * sin;
                        sum.im += value.re * sin + value.im * cos;
                    }
                    output[base + k * stride] = Complex32::new(sum.re * scale, sum.im * scale);
                }
            }
        }

        values = output;
    }

    Ok(values)
}

/// `f64` reference transform: each line of each axis goes through
/// [`fft_f64`], which agrees with the plain DFT to a few units in the last
/// place while taking `O(n log n)` for smooth lengths.
pub fn reference_c2c_nd_f64(input: &[Complex64], config: &FftConfig) -> Result<Vec<Complex64>> {
    config.validate()?;
    let total_complex = config.total_complex_len()?;
    assert_eq!(
        input.len(),
        total_complex,
        "reference input length must match FftConfig::total_complex_len"
    );

    let shape = config.shape();
    let strides = strides_for_shape(shape);
    let logical_complex_len = config.logical_complex_len()?;
    let normalization_len = logical_complex_len as f64;
    let final_scale = match (config.direction(), config.normalization()) {
        (_, Normalization::None) => 1.0,
        (FftDirection::Forward, Normalization::Forward) => 1.0 / normalization_len,
        (FftDirection::Inverse, Normalization::Inverse) => 1.0 / normalization_len,
        (_, Normalization::Orthogonal) => 1.0 / normalization_len.sqrt(),
        _ => 1.0,
    };

    let mut values = input.to_vec();
    for (axis_index, &axis) in config.axes().iter().enumerate() {
        let axis_len = shape[axis];
        let stride = strides[axis];
        let lines_per_batch = logical_complex_len / axis_len;
        let scale = if axis_index + 1 == config.axes().len() {
            final_scale
        } else {
            1.0
        };
        let mut output = vec![Complex64::default(); total_complex];
        let mut line_values = vec![Complex64::default(); axis_len];

        for batch in 0..config.batch() {
            let batch_base = batch * logical_complex_len;
            for line in 0..lines_per_batch {
                let base = batch_base + line_base_for_axis(line, axis, shape, &strides);
                for (n, value) in line_values.iter_mut().enumerate() {
                    *value = values[base + n * stride];
                }
                let spectrum = fft_f64(&line_values, config.direction());
                for (k, value) in spectrum.into_iter().enumerate() {
                    output[base + k * stride] = Complex64::new(value.re * scale, value.im * scale);
                }
            }
        }

        values = output;
    }

    Ok(values)
}

/// Unnormalized DFT of `input` in `f64`, of any length: a mixed-radix
/// Cooley-Tukey FFT that splits off the smallest prime factor at each level
/// and transforms prime lengths directly, or through Bluestein's convolution
/// above [`MAX_DIRECT_PRIME_F64`], so it takes `O(n log n)` operations. It
/// builds the filter spectra of Rader and Bluestein axes, whose lengths reach
/// tens of thousands of points.
pub(crate) fn fft_f64(input: &[Complex64], direction: FftDirection) -> Vec<Complex64> {
    let n = input.len();
    if n <= 1 {
        return input.to_vec();
    }
    let sign = match direction {
        FftDirection::Forward => -1.0,
        FftDirection::Inverse => 1.0,
    };
    let roots = (0..n)
        .map(|k| {
            let (sin, cos) = (sign * std::f64::consts::TAU * k as f64 / n as f64).sin_cos();
            Complex64::new(cos, sin)
        })
        .collect::<Vec<_>>();
    let mut output = vec![Complex64::default(); n];
    fft_f64_recursive(input, 1, &mut output, &roots, 1, direction);
    output
}

/// Longest prime [`fft_f64`] transforms as a plain DFT.
const MAX_DIRECT_PRIME_F64: usize = 64;

/// Transforms the `output.len()` points `input[0], input[stride], ...` into
/// `output`, where `roots[k * root_stride]` is the `k`-th root of unity of
/// that length.
fn fft_f64_recursive(
    input: &[Complex64],
    stride: usize,
    output: &mut [Complex64],
    roots: &[Complex64],
    root_stride: usize,
    direction: FftDirection,
) {
    let mul = |a: Complex64, b: Complex64| {
        Complex64::new(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re)
    };
    let n = output.len();
    let p = (2..n)
        .take_while(|factor| factor * factor <= n)
        .find(|factor| n.is_multiple_of(*factor))
        .unwrap_or(n);
    if p == n && n > MAX_DIRECT_PRIME_F64 {
        let values = (0..n).map(|j| input[j * stride]).collect::<Vec<_>>();
        output.copy_from_slice(&bluestein_f64(&values, direction));
        return;
    }
    if p == n {
        for (k, value) in output.iter_mut().enumerate() {
            let mut sum = Complex64::default();
            for j in 0..n {
                let term = mul(input[j * stride], roots[(j * k % n) * root_stride]);
                sum.re += term.re;
                sum.im += term.im;
            }
            *value = sum;
        }
        return;
    }
    // Sub-transform r holds the points p * j + r; output point k + m * s
    // sums W_n^(r k) W_p^(r s) times point k of sub-transform r.
    let m = n / p;
    for r in 0..p {
        fft_f64_recursive(
            &input[r * stride..],
            stride * p,
            &mut output[r * m..(r + 1) * m],
            roots,
            root_stride * p,
            direction,
        );
    }
    let mut column = vec![Complex64::default(); p];
    for k in 0..m {
        for (r, value) in column.iter_mut().enumerate() {
            *value = mul(output[r * m + k], roots[r * k * root_stride]);
        }
        for s in 0..p {
            let mut sum = Complex64::default();
            for (r, value) in column.iter().enumerate() {
                let term = mul(*value, roots[(r * s % p) * m * root_stride]);
                sum.re += term.re;
                sum.im += term.im;
            }
            output[k + m * s] = sum;
        }
    }
}

/// DFT of `values` through Bluestein's convolution over a power-of-two
/// length: with the chirp `c_k = W^(k^2 / 2)`, output `k` is `c_k` times the
/// convolution of `x_j c_j` with the conjugate chirp.
fn bluestein_f64(values: &[Complex64], direction: FftDirection) -> Vec<Complex64> {
    let mul = |a: Complex64, b: Complex64| {
        Complex64::new(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re)
    };
    let n = values.len();
    let sign = match direction {
        FftDirection::Forward => -1.0,
        FftDirection::Inverse => 1.0,
    };
    let chirp = (0..n)
        .map(|k| {
            let square = (k as u128 * k as u128 % (2 * n as u128)) as f64;
            let (sin, cos) = (sign * std::f64::consts::PI * square / n as f64).sin_cos();
            Complex64::new(cos, sin)
        })
        .collect::<Vec<_>>();
    let m = (2 * n - 1).next_power_of_two();
    let mut a = vec![Complex64::default(); m];
    let mut b = vec![Complex64::default(); m];
    for k in 0..n {
        a[k] = mul(values[k], chirp[k]);
        let conjugate = Complex64::new(chirp[k].re, -chirp[k].im);
        b[k] = conjugate;
        if k > 0 {
            b[m - k] = conjugate;
        }
    }
    let product = fft_f64(&a, FftDirection::Forward)
        .into_iter()
        .zip(fft_f64(&b, FftDirection::Forward))
        .map(|(a, b)| mul(a, b))
        .collect::<Vec<_>>();
    let convolution = fft_f64(&product, FftDirection::Inverse);
    (0..n)
        .map(|k| {
            let value = mul(convolution[k], chirp[k]);
            Complex64::new(value.re / m as f64, value.im / m as f64)
        })
        .collect()
}

fn validate_real_reference_config(config: &FftConfig, direction: FftDirection) -> Result<()> {
    config.validate()?;
    if config.direction() != direction {
        return Err(crate::error::FftError::InvalidRealTransformDirection {
            transform: if direction == FftDirection::Forward {
                "r2c"
            } else {
                "c2r"
            },
            expected: match direction {
                FftDirection::Forward => "forward",
                FftDirection::Inverse => "inverse",
            },
            actual: match config.direction() {
                FftDirection::Forward => "forward",
                FftDirection::Inverse => "inverse",
            },
        });
    }
    let expected_axes = (0..config.shape().len()).collect::<Vec<_>>();
    if config.axes() != expected_axes.as_slice() {
        return Err(crate::error::FftError::UnsupportedRealAxes {
            expected: expected_axes,
            actual: config.axes().to_vec(),
        });
    }
    Ok(())
}

fn strides_for_shape(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; shape.len()];
    for index in 1..shape.len() {
        strides[index] = strides[index - 1] * shape[index - 1];
    }
    strides
}

fn packed_shape_for(shape: &[usize]) -> Vec<usize> {
    let mut packed = shape.to_vec();
    packed[0] = shape[0] / 2 + 1;
    packed
}

fn checked_product(values: &[usize]) -> usize {
    values.iter().product()
}

fn coords_from_linear(mut index: usize, shape: &[usize], strides: &[usize]) -> Vec<usize> {
    let mut coords = vec![0usize; shape.len()];
    for dim in (0..shape.len()).rev() {
        coords[dim] = index / strides[dim];
        index %= strides[dim];
    }
    coords
}

fn linear_from_coords(coords: &[usize], strides: &[usize]) -> usize {
    coords
        .iter()
        .zip(strides)
        .map(|(coord, stride)| coord * stride)
        .sum()
}

fn packed_coords_for_full_spectrum(coords: &[usize], shape: &[usize]) -> Vec<usize> {
    let packed_nx = shape[0] / 2 + 1;
    let mut packed = Vec::with_capacity(coords.len());
    let x = coords[0];
    packed.push(if x >= packed_nx { shape[0] - x } else { x });
    for dim in 1..shape.len() {
        if x >= packed_nx && coords[dim] != 0 {
            packed.push(shape[dim] - coords[dim]);
        } else {
            packed.push(coords[dim]);
        }
    }
    packed
}

fn is_self_conjugate_bin(coords: &[usize], shape: &[usize]) -> bool {
    let x = coords[0];
    let even_x = shape[0].is_multiple_of(2);
    if !(x == 0 || (even_x && x == shape[0] / 2)) {
        return false;
    }
    for dim in 1..shape.len() {
        if shape[dim].is_multiple_of(2) {
            if coords[dim] != 0 && coords[dim] != shape[dim] / 2 {
                return false;
            }
        } else if coords[dim] != 0 {
            return false;
        }
    }
    true
}

fn line_base_for_axis(mut line: usize, axis: usize, shape: &[usize], strides: &[usize]) -> usize {
    let mut base = 0usize;
    for dim_index in 0..shape.len() {
        if dim_index == axis {
            continue;
        }
        let coord = line % shape[dim_index];
        line /= shape[dim_index];
        base += coord * strides[dim_index];
    }
    base
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Normalization;

    fn assert_close(actual: &[Complex32], expected: &[Complex32]) {
        assert_eq!(actual.len(), expected.len());
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                actual.abs_diff(*expected) < 1.0e-5,
                "index {index}: actual={actual:?}, expected={expected:?}"
            );
        }
    }

    fn assert_f32_close(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() < 1.0e-5,
                "index {index}: actual={actual}, expected={expected}"
            );
        }
    }

    fn assert_close_f64(actual: &[Complex64], expected: &[Complex64]) {
        assert_eq!(actual.len(), expected.len());
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                actual.abs_diff(*expected) < 1.0e-12,
                "index {index}: actual={actual:?}, expected={expected:?}"
            );
        }
    }

    #[test]
    fn round_trips_interleaved_data() {
        let values = [1.0, 2.0, 3.0, 4.0];
        let complex = from_interleaved_f32(&values);
        assert_eq!(
            complex,
            vec![Complex32::new(1.0, 2.0), Complex32::new(3.0, 4.0)]
        );
        assert_eq!(to_interleaved_f32(&complex), values);
    }

    #[test]
    fn round_trips_interleaved_f64_data() {
        let values = [1.0, 2.0, 3.0, 4.0];
        let complex = from_interleaved_f64(&values);
        assert_eq!(
            complex,
            vec![Complex64::new(1.0, 2.0), Complex64::new(3.0, 4.0)]
        );
        assert_eq!(to_interleaved_f64(&complex), values);
    }

    #[test]
    fn computes_known_length_four_forward_dft() {
        let input = [
            Complex32::new(1.0, 0.0),
            Complex32::new(2.0, 0.0),
            Complex32::new(3.0, 0.0),
            Complex32::new(4.0, 0.0),
        ];
        let config = FftConfig::new(4).with_normalization(Normalization::None);
        let output = reference_dft(&input, config).unwrap();
        let expected = [
            Complex32::new(10.0, 0.0),
            Complex32::new(-2.0, 2.0),
            Complex32::new(-2.0, 0.0),
            Complex32::new(-2.0, -2.0),
        ];
        assert_close(&output, &expected);
    }

    #[test]
    fn f64_reference_computes_known_length_four_forward_dft() {
        let input = [
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(3.0, 0.0),
            Complex64::new(4.0, 0.0),
        ];
        let config = FftConfig::new(4).with_normalization(Normalization::None);
        let output = reference_c2c_nd_f64(&input, &config).unwrap();
        let expected = [
            Complex64::new(10.0, 0.0),
            Complex64::new(-2.0, 2.0),
            Complex64::new(-2.0, 0.0),
            Complex64::new(-2.0, -2.0),
        ];
        assert_close_f64(&output, &expected);
    }

    #[test]
    fn f64_fft_matches_a_plain_dft() {
        for n in [
            1, 2, 3, 4, 5, 12, 16, 17, 30, 49, 97, 128, 210, 282, 1060, 1523, 3946,
        ] {
            let input = (0..n)
                .map(|i| Complex64::new((i as f64 * 0.37).sin(), (i as f64 * 0.11).cos()))
                .collect::<Vec<_>>();
            for direction in [FftDirection::Forward, FftDirection::Inverse] {
                let sign = match direction {
                    FftDirection::Forward => -1.0,
                    FftDirection::Inverse => 1.0,
                };
                let expected = (0..n)
                    .map(|k| {
                        input
                            .iter()
                            .enumerate()
                            .fold(Complex64::default(), |sum, (j, value)| {
                                let angle =
                                    sign * std::f64::consts::TAU * ((j * k) % n) as f64 / n as f64;
                                let (sin, cos) = angle.sin_cos();
                                Complex64::new(
                                    sum.re + value.re * cos - value.im * sin,
                                    sum.im + value.re * sin + value.im * cos,
                                )
                            })
                    })
                    .collect::<Vec<_>>();
                let actual = fft_f64(&input, direction);
                let scale = expected
                    .iter()
                    .map(|value| value.re.abs().max(value.im.abs()))
                    .fold(1.0, f64::max);
                for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
                    assert!(
                        actual.abs_diff(*expected) < 1.0e-12 * scale,
                        "n={n} {direction:?} index {index}: {actual:?} vs {expected:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn f64_reference_computes_batched_odd_axis_subset() {
        let input = (1..=12)
            .map(|value| Complex64::new(value as f64, 0.0))
            .collect::<Vec<_>>();
        let config = FftConfig::new_nd([2, 3])
            .with_axes([1])
            .with_batch(2)
            .with_normalization(Normalization::None);
        let output = reference_c2c_nd_f64(&input, &config).unwrap();
        let root = 3.0f64.sqrt() * 0.5;
        let mut expected = vec![Complex64::default(); 12];
        for batch in 0..2 {
            let base = batch * 6;
            for x in 0..2 {
                let a = input[base + x].re;
                let b = input[base + 2 + x].re;
                let c = input[base + 4 + x].re;
                expected[base + x] = Complex64::new(a + b + c, 0.0);
                expected[base + 2 + x] = Complex64::new(a - 0.5 * (b + c), root * (c - b));
                expected[base + 4 + x] = Complex64::new(a - 0.5 * (b + c), root * (b - c));
            }
        }
        assert_close_f64(&output, &expected);
    }

    #[test]
    fn computes_known_r2c_packed_even_length() {
        let input = [1.0, 2.0, 3.0, 4.0];
        let config = FftConfig::new(4).with_normalization(Normalization::None);
        let packed = reference_r2c_packed_interleaved(&input, &config).unwrap();
        assert_f32_close(&packed, &[10.0, 0.0, -2.0, 2.0, -2.0, 0.0]);
    }

    #[test]
    fn r2c_c2r_round_trip_odd_length() {
        let input = [0.25, -1.0, 2.0, 0.5, -0.75];
        let forward = FftConfig::new(5).with_normalization(Normalization::None);
        let packed = reference_r2c_packed_interleaved(&input, &forward).unwrap();
        assert_eq!(packed.len(), 3 * 2);
        let recovered =
            reference_c2r_from_packed_interleaved(&packed, &FftConfig::inverse(5)).unwrap();
        assert_f32_close(&recovered, &input);
    }

    #[test]
    fn r2c_c2r_round_trip_nd_packs_axis_zero() {
        let input = [0.25, -1.0, 2.0, 0.5, -0.75, 1.25];
        let forward = FftConfig::new_nd([3, 2]).with_normalization(Normalization::None);
        let packed = reference_r2c_packed_interleaved(&input, &forward).unwrap();
        assert_eq!(packed.len(), 2 * 2 * 2);
        let recovered =
            reference_c2r_from_packed_interleaved(&packed, &FftConfig::inverse_nd([3, 2])).unwrap();
        assert_f32_close(&recovered, &input);
    }

    #[test]
    fn inverse_default_normalization_recovers_input() {
        let input = [
            Complex32::new(1.0, 0.25),
            Complex32::new(2.0, -0.5),
            Complex32::new(-3.0, 1.0),
            Complex32::new(4.0, 2.0),
        ];
        let spectrum = reference_dft(
            &input,
            FftConfig::new(4).with_normalization(Normalization::None),
        )
        .unwrap();
        let recovered = reference_dft(&spectrum, FftConfig::inverse(4)).unwrap();
        assert_close(&recovered, &input);
    }

    #[test]
    fn computes_axis_subset_for_nd_shape() {
        let input = [
            Complex32::new(1.0, 0.0),
            Complex32::new(2.0, 0.0),
            Complex32::new(3.0, 0.0),
            Complex32::new(4.0, 0.0),
            Complex32::new(5.0, 0.0),
            Complex32::new(6.0, 0.0),
        ];
        let config = FftConfig::new_nd([2, 3])
            .with_axes([0])
            .with_normalization(Normalization::None);
        let output = reference_c2c_nd(&input, &config).unwrap();
        let expected = [
            Complex32::new(3.0, 0.0),
            Complex32::new(-1.0, 0.0),
            Complex32::new(7.0, 0.0),
            Complex32::new(-1.0, 0.0),
            Complex32::new(11.0, 0.0),
            Complex32::new(-1.0, 0.0),
        ];
        assert_close(&output, &expected);
    }

    #[test]
    fn inverse_nd_default_normalization_recovers_input() {
        let input = [
            Complex32::new(1.0, 0.25),
            Complex32::new(2.0, -0.5),
            Complex32::new(-3.0, 1.0),
            Complex32::new(4.0, 2.0),
            Complex32::new(0.75, -1.5),
            Complex32::new(-2.25, 0.5),
        ];
        let forward = FftConfig::new_nd([2, 3]).with_normalization(Normalization::None);
        let spectrum = reference_c2c_nd(&input, &forward).unwrap();
        let recovered = reference_c2c_nd(&spectrum, &FftConfig::inverse_nd([2, 3])).unwrap();
        assert_close(&recovered, &input);
    }

    #[test]
    fn f64_reference_round_trips_batched_nd_data() {
        let input = (0..12)
            .map(|index| {
                let x = index as f64 + 1.0;
                Complex64::new(x * 0.25 - 1.0, x * -0.125 + 0.5)
            })
            .collect::<Vec<_>>();
        let forward = FftConfig::new_nd([2, 3])
            .with_batch(2)
            .with_normalization(Normalization::None);
        let spectrum = reference_c2c_nd_f64(&input, &forward).unwrap();
        let recovered =
            reference_c2c_nd_f64(&spectrum, &FftConfig::inverse_nd([2, 3]).with_batch(2)).unwrap();
        assert_close_f64(&recovered, &input);
    }
}
