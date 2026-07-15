use std::f64::consts::{PI, TAU};

use wgpu_fft::math::reference_c2c_nd_f64;
use wgpu_fft::{FftConfig, FftDirection, Normalization};
use wgpu_nufft::{
    reference_type1_f64, reference_type2_f64, select_fine_grid_size, Complex64, EsKernel,
    ModeOrder, NufftConfig, NufftError, NufftKind, NufftPlan, NufftSign,
};

fn close(actual: Complex64, expected: Complex64, tolerance: f64) {
    let error = ((actual.re - expected.re).powi(2) + (actual.im - expected.im).powi(2)).sqrt();
    assert!(
        error <= tolerance,
        "actual={actual:?}, expected={expected:?}, error={error}, tolerance={tolerance}"
    );
}

fn mode_for_index(order: ModeOrder, index: usize, length: usize) -> isize {
    match order {
        ModeOrder::Centered => index as isize - (length / 2) as isize,
        ModeOrder::Fft => {
            if index < length.div_ceil(2) {
                index as isize
            } else {
                index as isize - length as isize
            }
        }
    }
}

fn grid_index(mode: isize, length: usize) -> usize {
    mode.rem_euclid(length as isize) as usize
}

fn fft_for_sign(values: &[Complex64], sign: NufftSign) -> Vec<Complex64> {
    fft_nd_for_sign(values, &[values.len()], sign)
}

fn fft_nd_for_sign(values: &[Complex64], shape: &[usize], sign: NufftSign) -> Vec<Complex64> {
    let direction = match sign {
        NufftSign::Positive => FftDirection::Inverse,
        NufftSign::Negative => FftDirection::Forward,
    };
    reference_c2c_nd_f64(
        values,
        &FftConfig::new_nd(shape.to_vec())
            .with_direction(direction)
            .with_normalization(Normalization::None),
    )
    .unwrap()
}

fn fold_to_grid(coordinate: f64, length: usize) -> f64 {
    (coordinate / TAU + 0.5).rem_euclid(1.0) * length as f64
}

fn approximate_type1_1d(
    plan: &NufftPlan,
    coordinates: &[f64],
    strengths: &[Complex64],
) -> Vec<Complex64> {
    let fine_length = plan.fine_grid_shape()[0];
    let kernel = plan.kernel();
    let mut grid = vec![Complex64::default(); fine_length];
    for (&coordinate, &strength) in coordinates.iter().zip(strengths) {
        let position = fold_to_grid(coordinate, fine_length);
        let start = (position - kernel.half_width()).ceil() as isize;
        for offset in 0..kernel.width() {
            let unwrapped = start + offset as isize;
            let weight = kernel.evaluate(unwrapped as f64 - position);
            let index = unwrapped.rem_euclid(fine_length as isize) as usize;
            grid[index].re += strength.re * weight;
            grid[index].im += strength.im * weight;
        }
    }
    let transformed = fft_for_sign(&grid, plan.config().sign());
    let coefficients = plan.centered_kernel_fourier_coefficients(0).unwrap();
    (0..plan.config().n_modes()[0])
        .map(|index| {
            let mode = mode_for_index(
                plan.config().mode_order(),
                index,
                plan.config().n_modes()[0],
            );
            let divisor = coefficients[mode.unsigned_abs()];
            let value = transformed[grid_index(mode, fine_length)];
            Complex64::new(value.re / divisor, value.im / divisor)
        })
        .collect()
}

fn approximate_type2_1d(
    plan: &NufftPlan,
    coordinates: &[f64],
    input: &[Complex64],
) -> Vec<Complex64> {
    let fine_length = plan.fine_grid_shape()[0];
    let kernel = plan.kernel();
    let coefficients = plan.centered_kernel_fourier_coefficients(0).unwrap();
    let mut grid = vec![Complex64::default(); fine_length];
    for (index, &value) in input.iter().enumerate() {
        let mode = mode_for_index(
            plan.config().mode_order(),
            index,
            plan.config().n_modes()[0],
        );
        let divisor = coefficients[mode.unsigned_abs()];
        grid[grid_index(mode, fine_length)] =
            Complex64::new(value.re / divisor, value.im / divisor);
    }
    let transformed = fft_for_sign(&grid, plan.config().sign());
    coordinates
        .iter()
        .map(|&coordinate| {
            let position = fold_to_grid(coordinate, fine_length);
            let start = (position - kernel.half_width()).ceil() as isize;
            let mut sum = Complex64::default();
            for offset in 0..kernel.width() {
                let unwrapped = start + offset as isize;
                let weight = kernel.evaluate(unwrapped as f64 - position);
                let value = transformed[unwrapped.rem_euclid(fine_length as isize) as usize];
                sum.re += value.re * weight;
                sum.im += value.im * weight;
            }
            sum
        })
        .collect()
}

fn approximate_type1_2d(
    plan: &NufftPlan,
    coordinates: &[f64],
    strengths: &[Complex64],
) -> Vec<Complex64> {
    let fine = plan.fine_grid_shape();
    let modes = plan.config().n_modes();
    let kernel = plan.kernel();
    let mut grid = vec![Complex64::default(); fine[0] * fine[1]];
    for (point, &strength) in coordinates.chunks_exact(2).zip(strengths) {
        let position = [
            fold_to_grid(point[0], fine[0]),
            fold_to_grid(point[1], fine[1]),
        ];
        let start = [
            (position[0] - kernel.half_width()).ceil() as isize,
            (position[1] - kernel.half_width()).ceil() as isize,
        ];
        for offset1 in 0..kernel.width() {
            let unwrapped1 = start[1] + offset1 as isize;
            let weight1 = kernel.evaluate(unwrapped1 as f64 - position[1]);
            let grid1 = unwrapped1.rem_euclid(fine[1] as isize) as usize;
            for offset0 in 0..kernel.width() {
                let unwrapped0 = start[0] + offset0 as isize;
                let weight0 = kernel.evaluate(unwrapped0 as f64 - position[0]);
                let grid0 = unwrapped0.rem_euclid(fine[0] as isize) as usize;
                let value = &mut grid[grid0 + fine[0] * grid1];
                let weight = weight0 * weight1;
                value.re += strength.re * weight;
                value.im += strength.im * weight;
            }
        }
    }
    let transformed = fft_nd_for_sign(&grid, fine, plan.config().sign());
    let coefficient0 = plan.centered_kernel_fourier_coefficients(0).unwrap();
    let coefficient1 = plan.centered_kernel_fourier_coefficients(1).unwrap();
    let mut output = vec![Complex64::default(); modes[0] * modes[1]];
    for index1 in 0..modes[1] {
        let mode1 = mode_for_index(plan.config().mode_order(), index1, modes[1]);
        for index0 in 0..modes[0] {
            let mode0 = mode_for_index(plan.config().mode_order(), index0, modes[0]);
            let divisor = coefficient0[mode0.unsigned_abs()] * coefficient1[mode1.unsigned_abs()];
            let value =
                transformed[grid_index(mode0, fine[0]) + fine[0] * grid_index(mode1, fine[1])];
            output[index0 + modes[0] * index1] =
                Complex64::new(value.re / divisor, value.im / divisor);
        }
    }
    output
}

fn approximate_type2_2d(
    plan: &NufftPlan,
    coordinates: &[f64],
    input: &[Complex64],
) -> Vec<Complex64> {
    let fine = plan.fine_grid_shape();
    let modes = plan.config().n_modes();
    let kernel = plan.kernel();
    let coefficient0 = plan.centered_kernel_fourier_coefficients(0).unwrap();
    let coefficient1 = plan.centered_kernel_fourier_coefficients(1).unwrap();
    let mut grid = vec![Complex64::default(); fine[0] * fine[1]];
    for index1 in 0..modes[1] {
        let mode1 = mode_for_index(plan.config().mode_order(), index1, modes[1]);
        for index0 in 0..modes[0] {
            let mode0 = mode_for_index(plan.config().mode_order(), index0, modes[0]);
            let divisor = coefficient0[mode0.unsigned_abs()] * coefficient1[mode1.unsigned_abs()];
            let value = input[index0 + modes[0] * index1];
            grid[grid_index(mode0, fine[0]) + fine[0] * grid_index(mode1, fine[1])] =
                Complex64::new(value.re / divisor, value.im / divisor);
        }
    }
    let transformed = fft_nd_for_sign(&grid, fine, plan.config().sign());
    coordinates
        .chunks_exact(2)
        .map(|point| {
            let position = [
                fold_to_grid(point[0], fine[0]),
                fold_to_grid(point[1], fine[1]),
            ];
            let start = [
                (position[0] - kernel.half_width()).ceil() as isize,
                (position[1] - kernel.half_width()).ceil() as isize,
            ];
            let mut sum = Complex64::default();
            for offset1 in 0..kernel.width() {
                let unwrapped1 = start[1] + offset1 as isize;
                let weight1 = kernel.evaluate(unwrapped1 as f64 - position[1]);
                let grid1 = unwrapped1.rem_euclid(fine[1] as isize) as usize;
                for offset0 in 0..kernel.width() {
                    let unwrapped0 = start[0] + offset0 as isize;
                    let weight0 = kernel.evaluate(unwrapped0 as f64 - position[0]);
                    let grid0 = unwrapped0.rem_euclid(fine[0] as isize) as usize;
                    let value = transformed[grid0 + fine[0] * grid1];
                    let weight = weight0 * weight1;
                    sum.re += value.re * weight;
                    sum.im += value.im * weight;
                }
            }
            sum
        })
        .collect()
}

fn approximate_type2_3d(
    plan: &NufftPlan,
    coordinates: &[f64],
    input: &[Complex64],
) -> Vec<Complex64> {
    let fine = plan.fine_grid_shape();
    let modes = plan.config().n_modes();
    let kernel = plan.kernel();
    let coefficient0 = plan.centered_kernel_fourier_coefficients(0).unwrap();
    let coefficient1 = plan.centered_kernel_fourier_coefficients(1).unwrap();
    let coefficient2 = plan.centered_kernel_fourier_coefficients(2).unwrap();
    let mut grid = vec![Complex64::default(); fine[0] * fine[1] * fine[2]];
    for index2 in 0..modes[2] {
        let mode2 = mode_for_index(plan.config().mode_order(), index2, modes[2]);
        for index1 in 0..modes[1] {
            let mode1 = mode_for_index(plan.config().mode_order(), index1, modes[1]);
            for index0 in 0..modes[0] {
                let mode0 = mode_for_index(plan.config().mode_order(), index0, modes[0]);
                let divisor = coefficient0[mode0.unsigned_abs()]
                    * coefficient1[mode1.unsigned_abs()]
                    * coefficient2[mode2.unsigned_abs()];
                let value = input[index0 + modes[0] * (index1 + modes[1] * index2)];
                let fine_index = grid_index(mode0, fine[0])
                    + fine[0] * (grid_index(mode1, fine[1]) + fine[1] * grid_index(mode2, fine[2]));
                grid[fine_index] = Complex64::new(value.re / divisor, value.im / divisor);
            }
        }
    }
    let transformed = fft_nd_for_sign(&grid, fine, plan.config().sign());
    coordinates
        .chunks_exact(3)
        .map(|point| {
            let position = [
                fold_to_grid(point[0], fine[0]),
                fold_to_grid(point[1], fine[1]),
                fold_to_grid(point[2], fine[2]),
            ];
            let start = [
                (position[0] - kernel.half_width()).ceil() as isize,
                (position[1] - kernel.half_width()).ceil() as isize,
                (position[2] - kernel.half_width()).ceil() as isize,
            ];
            let mut sum = Complex64::default();
            for offset2 in 0..kernel.width() {
                let unwrapped2 = start[2] + offset2 as isize;
                let weight2 = kernel.evaluate(unwrapped2 as f64 - position[2]);
                let grid2 = unwrapped2.rem_euclid(fine[2] as isize) as usize;
                for offset1 in 0..kernel.width() {
                    let unwrapped1 = start[1] + offset1 as isize;
                    let weight1 = kernel.evaluate(unwrapped1 as f64 - position[1]);
                    let grid1 = unwrapped1.rem_euclid(fine[1] as isize) as usize;
                    for offset0 in 0..kernel.width() {
                        let unwrapped0 = start[0] + offset0 as isize;
                        let weight0 = kernel.evaluate(unwrapped0 as f64 - position[0]);
                        let grid0 = unwrapped0.rem_euclid(fine[0] as isize) as usize;
                        let value = transformed[grid0 + fine[0] * (grid1 + fine[1] * grid2)];
                        let weight = weight0 * weight1 * weight2;
                        sum.re += value.re * weight;
                        sum.im += value.im * weight;
                    }
                }
            }
            sum
        })
        .collect()
}

fn relative_l2(actual: &[Complex64], reference: &[Complex64]) -> f64 {
    let error = actual
        .iter()
        .zip(reference)
        .map(|(actual, reference)| {
            (actual.re - reference.re).powi(2) + (actual.im - reference.im).powi(2)
        })
        .sum::<f64>()
        .sqrt();
    let norm = reference
        .iter()
        .map(|value| value.re * value.re + value.im * value.im)
        .sum::<f64>()
        .sqrt();
    if norm == 0.0 {
        error
    } else {
        error / norm
    }
}

#[test]
fn direct_type1_matches_analytic_zero_point_and_duplicate_sum() {
    let config = NufftConfig::new([5], 1.0e-6);
    let strengths = [Complex64::new(1.5, -0.25), Complex64::new(-0.5, 2.0)];
    let output = reference_type1_f64(&config, &[0.0, 0.0], &strengths).unwrap();
    let expected = Complex64::new(1.0, 1.75);
    assert_eq!(output.len(), 5);
    for value in output {
        close(value, expected, 1.0e-14);
    }
}

#[test]
fn direct_type2_respects_sign_and_centered_mode_indices() {
    let coordinate = 0.37;
    let mut coefficients = vec![Complex64::default(); 5];
    coefficients[4] = Complex64::new(1.0, 0.0); // centered mode k=+2
    for sign in [NufftSign::Positive, NufftSign::Negative] {
        let config = NufftConfig::new([5], 1.0e-6).with_sign(sign);
        let output = reference_type2_f64(&config, &[coordinate], &coefficients).unwrap();
        let angle = sign.isign() as f64 * 2.0 * coordinate;
        let (sin, cos) = angle.sin_cos();
        close(output[0], Complex64::new(cos, sin), 1.0e-14);
    }
}

#[test]
fn centered_and_fft_mode_orders_are_exact_permutations() {
    let points = [-PI, -0.7, 0.0, 1.1, PI];
    let centered = NufftConfig::new([6], 1.0e-6);
    let fft = centered.clone().with_mode_order(ModeOrder::Fft);
    let centered_coefficients = (0..6)
        .map(|index| Complex64::new(index as f64 + 0.25, index as f64 * -0.2))
        .collect::<Vec<_>>();
    let fft_coefficients = [
        centered_coefficients[3],
        centered_coefficients[4],
        centered_coefficients[5],
        centered_coefficients[0],
        centered_coefficients[1],
        centered_coefficients[2],
    ];
    let a = reference_type2_f64(&centered, &points, &centered_coefficients).unwrap();
    let b = reference_type2_f64(&fft, &points, &fft_coefficients).unwrap();
    for (actual, expected) in a.into_iter().zip(b) {
        close(actual, expected, 1.0e-13);
    }
}

#[test]
fn type1_output_order_and_periodic_boundary_are_finufft_compatible() {
    let centered = NufftConfig::new([6], 1.0e-6);
    let fft = centered.clone().with_mode_order(ModeOrder::Fft);
    let strengths = [Complex64::new(0.75, -0.5), Complex64::new(0.75, -0.5)];
    let centered_output = reference_type1_f64(&centered, &[-PI, PI], &strengths).unwrap();
    let fft_output = reference_type1_f64(&fft, &[-PI, PI], &strengths).unwrap();
    let expected_fft = [
        centered_output[3],
        centered_output[4],
        centered_output[5],
        centered_output[0],
        centered_output[1],
        centered_output[2],
    ];
    for (actual, expected) in fft_output.into_iter().zip(expected_fft) {
        close(actual, expected, 1.0e-13);
    }

    let negative_pi =
        reference_type2_f64(&centered, &[-PI], &[Complex64::new(1.0, 0.25); 6]).unwrap();
    let positive_pi =
        reference_type2_f64(&centered, &[PI], &[Complex64::new(1.0, 0.25); 6]).unwrap();
    close(negative_pi[0], positive_pi[0], 1.0e-13);
}

#[test]
fn direct_nd_uses_dimension_zero_as_the_fast_mode_axis() {
    let config = NufftConfig::new([2, 3], 1.0e-6);
    let point = [0.3, -0.4];
    let output = reference_type1_f64(&config, &point, &[Complex64::new(1.0, 0.0)]).unwrap();
    let modes = [(-1, -1), (0, -1), (-1, 0), (0, 0), (-1, 1), (0, 1)];
    for (value, (kx, ky)) in output.into_iter().zip(modes) {
        let angle = kx as f64 * point[0] + ky as f64 * point[1];
        let (sin, cos) = angle.sin_cos();
        close(value, Complex64::new(cos, sin), 1.0e-14);
    }
}

#[test]
fn direct_type2_supports_three_dimensions_with_axis_zero_fastest() {
    let config = NufftConfig::new([2, 3, 3], 1.0e-6);
    let point = [0.2, -0.3, 0.4];
    let mut coefficients = vec![Complex64::default(); 18];
    coefficients[5] = Complex64::new(0.75, -0.5); // mode (0, +1, -1)
    let output = reference_type2_f64(&config, &point, &coefficients).unwrap();
    let angle = -0.3_f64 - 0.4;
    let (sin, cos) = angle.sin_cos();
    close(
        output[0],
        Complex64::new(0.75 * cos + 0.5 * sin, 0.75 * sin - 0.5 * cos),
        1.0e-14,
    );
}

#[test]
fn direct_type1_supports_three_dimensions_with_axis_zero_fastest() {
    let config = NufftConfig::new([2, 3, 2], 1.0e-6);
    let point = [0.2, -0.3, 0.4];
    let strength = Complex64::new(0.75, -0.5);
    let output = reference_type1_f64(&config, &point, &[strength]).unwrap();
    assert_eq!(output.len(), 12);
    for index2 in 0..2 {
        let mode2 = index2 as i64 - 1;
        for index1 in 0..3 {
            let mode1 = index1 as i64 - 1;
            for index0 in 0..2 {
                let mode0 = index0 as i64 - 1;
                let linear = index0 + 2 * (index1 + 3 * index2);
                let phase =
                    mode0 as f64 * point[0] + mode1 as f64 * point[1] + mode2 as f64 * point[2];
                close(output[linear], rotate_complex(strength, phase), 1.0e-14);
            }
        }
    }
}

fn rotate_complex(value: Complex64, phase: f64) -> Complex64 {
    let (sin, cos) = phase.sin_cos();
    Complex64::new(
        value.re * cos - value.im * sin,
        value.re * sin + value.im * cos,
    )
}

#[test]
fn singleton_axes_are_valid_nufft_dimensions() {
    for n_modes in [[1, 4, 1], [4, 1, 1], [1, 1, 4]] {
        let config = NufftConfig::new(n_modes, 1.0e-6);
        config.validate().unwrap();
        assert_eq!(config.mode_count().unwrap(), 4);
        assert_eq!(NufftPlan::type1(config.clone()).unwrap().config(), &config);

        let point = [0.27, -0.43, 0.61];
        let output = reference_type1_f64(&config, &point, &[Complex64::new(1.0, 0.0)]).unwrap();
        assert_eq!(output.len(), 4);
        let active_axis = n_modes.iter().position(|&length| length > 1).unwrap();
        for (index, value) in output.into_iter().enumerate() {
            let mode = index as isize - 2;
            let angle = mode as f64 * point[active_axis];
            let (sin, cos) = angle.sin_cos();
            close(value, Complex64::new(cos, sin), 1.0e-14);
        }
    }
}

#[test]
fn es_parameter_table_matches_fixed_sigma_finufft_values() {
    let expected = [
        (1.0e-1, 2, 4.40),
        (1.0e-2, 3, 6.78),
        (1.0e-3, 4, 9.52),
        (1.0e-4, 5, 11.50),
        (1.0e-5, 6, 13.80),
        (1.0e-6, 7, 16.10),
    ];
    for (eps, width, beta) in expected {
        let kernel = EsKernel::for_tolerance(eps, 2.0).unwrap();
        assert_eq!(kernel.width(), width);
        assert!((kernel.beta() - beta).abs() < 1.0e-13);
        assert_eq!(kernel.evaluate(0.0), 1.0);
        assert_eq!(kernel.evaluate(kernel.half_width()), 0.0);
        assert_eq!(kernel.evaluate(-kernel.half_width()), 0.0);
        assert!((kernel.evaluate(0.31) - kernel.evaluate(-0.31)).abs() < 1.0e-15);
    }
}

#[test]
fn fine_grid_is_even_supported_and_covers_kernel_support() {
    let kernel = EsKernel::for_tolerance(1.0e-6, 2.0).unwrap();
    assert_eq!(select_fine_grid_size(17, 2.0, kernel.width()).unwrap(), 36);
    assert_eq!(select_fine_grid_size(1, 2.0, kernel.width()).unwrap(), 14);
    for modes in 1..128 {
        let length = select_fine_grid_size(modes, 2.0, kernel.width()).unwrap();
        assert!(length.is_multiple_of(2));
        assert!(length >= 2 * modes);
        assert!(length >= 2 * kernel.width());
        assert!(wgpu_fft::runtime::factor_supported_length(length).is_ok());
        let minimum = (2 * modes).max(2 * kernel.width());
        let first_even = if minimum.is_multiple_of(2) {
            minimum
        } else {
            minimum + 1
        };
        for smaller in (first_even..length).step_by(2) {
            assert!(
                wgpu_fft::runtime::factor_supported_length(smaller).is_err(),
                "selected {length} for {modes} modes even though {smaller} is supported"
            );
        }
    }
}

#[test]
fn custom_sigma_uses_finufft_general_width_beta_and_grid_formulas() {
    let sigma = 1.25;
    let eps = 1.0e-6;
    let kernel = EsKernel::for_tolerance(eps, sigma).unwrap();
    let expected_width =
        ((1.0_f64 / eps).ln() / (PI * (1.0_f64 - 1.0_f64 / sigma).sqrt())).ceil() as usize;
    let expected_beta = 0.97 * PI * expected_width as f64 * (1.0 - 1.0 / (2.0 * sigma));
    assert_eq!(kernel.width(), expected_width);
    assert!((kernel.beta() - expected_beta).abs() < 1.0e-13);
    assert_eq!(
        select_fine_grid_size(17, sigma, kernel.width()).unwrap(),
        22
    );
}

#[test]
fn plan_is_reusable_and_exposes_type_specific_cpu_fallback() {
    let config = NufftConfig::new([8], 1.0e-5)
        .with_isign(-9)
        .with_modeord(ModeOrder::Fft);
    let type1 = NufftPlan::type1(config.clone()).unwrap();
    let type2 = NufftPlan::type2(config).unwrap();
    assert_eq!(type1.kind(), NufftKind::Type1);
    assert_eq!(type2.kind(), NufftKind::Type2);
    assert_eq!(type1.config().isign(), -1);
    assert_eq!(type1.config().modeord(), 1);
    assert_eq!(type1.fine_grid_shape(), type2.fine_grid_shape());
    assert_eq!(
        type1.centered_kernel_fourier_coefficients(0).unwrap(),
        type2.centered_kernel_fourier_coefficients(0).unwrap()
    );

    let points = [-0.4, 0.7];
    let strengths = [Complex64::new(1.0, 0.5), Complex64::new(-0.25, 0.75)];
    let first = type1.execute_direct_f64(&points, &strengths).unwrap();
    let second = type1.execute_direct_f64(&points, &strengths).unwrap();
    assert_eq!(first, second);
    assert!(!type1.is_gpu_ready());
    assert!(!type2.is_gpu_ready());
    assert_eq!(
        NufftPlan::required_type1_point_buffer_size_bytes(7).unwrap(),
        28
    );
    assert_eq!(type1.required_point_buffer_size_bytes(7).unwrap(), 28);
    assert_eq!(
        NufftPlan::required_type1_strength_buffer_size_bytes(7).unwrap(),
        56
    );
    assert_eq!(type1.required_type1_output_buffer_size_bytes().unwrap(), 64);
    assert_eq!(
        NufftPlan::required_type2_point_buffer_size_bytes(7).unwrap(),
        28
    );
    assert_eq!(type2.required_point_buffer_size_bytes(7).unwrap(), 28);
    assert_eq!(
        type2
            .required_type2_coefficient_buffer_size_bytes()
            .unwrap(),
        64
    );
    assert_eq!(
        NufftPlan::required_type2_output_buffer_size_bytes(7).unwrap(),
        56
    );
}

#[test]
fn plan_aware_gpu_point_buffer_size_is_dimension_aware() {
    let plan = NufftPlan::type2(NufftConfig::new([8, 12], 1.0e-6)).unwrap();
    assert_eq!(plan.required_point_buffer_size_bytes(7).unwrap(), 56);
    assert_eq!(
        plan.required_point_buffer_size_bytes(usize::MAX),
        Err(NufftError::LengthOverflow {
            context: "GPU point-coordinate count"
        })
    );
}

#[test]
fn validation_errors_are_structured() {
    assert!(matches!(
        NufftConfig::new(Vec::<usize>::new(), 1.0e-6).validate(),
        Err(NufftError::InvalidDimensions { actual: 0, .. })
    ));
    assert!(matches!(
        NufftConfig::new([4, 0], 1.0e-6).validate(),
        Err(NufftError::ZeroMode { axis: 1 })
    ));
    assert!(matches!(
        NufftConfig::new([4], 0.0).validate(),
        Err(NufftError::InvalidTolerance { .. })
    ));
    assert!(matches!(
        NufftConfig::new([4], 1.0e-6).with_sigma(1.0).validate(),
        Err(NufftError::InvalidSigma { .. })
    ));
    assert!(matches!(
        NufftConfig::new([4], 1.0e-16).validate(),
        Err(NufftError::KernelWidthUnsupported { .. })
    ));
    assert!(matches!(
        select_fine_grid_size(4, 2.0, 0),
        Err(NufftError::InvalidKernelWidth { .. })
    ));

    let config = NufftConfig::new([4, 4], 1.0e-6);
    assert!(matches!(
        reference_type1_f64(&config, &[0.0], &[]),
        Err(NufftError::CoordinateLength { .. })
    ));
    assert!(matches!(
        reference_type2_f64(
            &NufftConfig::new([4], 1.0e-6),
            &[4.0 * PI],
            &[Complex64::default(); 4]
        ),
        Err(NufftError::PointOutOfRange { .. })
    ));

    let oversized_fine_grid = NufftConfig::new([32_768, 32_768], 1.0e-6);
    assert!(oversized_fine_grid.validate().is_ok());
    assert!(matches!(
        NufftPlan::type1(oversized_fine_grid),
        Err(NufftError::FftShapeUnsupported {
            stage: "oversampled NUFFT fine-grid shape",
            ..
        })
    ));
}

#[test]
fn es_host_path_meets_finufft_relative_l2_acceptance_against_direct_ndft() {
    let coordinates = [-2.91, -1.37, -0.51, -0.03, 0.42, 1.19, 2.74];
    let strengths = coordinates
        .iter()
        .enumerate()
        .map(|(index, &x)| {
            Complex64::new(
                (0.7_f64 * x).cos() + index as f64 * 0.03,
                (1.1_f64 * x).sin() - index as f64 * 0.02,
            )
        })
        .collect::<Vec<_>>();
    let coefficients = (0..16)
        .map(|index| {
            let x = index as f64 + 0.5;
            Complex64::new((0.31 * x).cos(), (0.47 * x).sin() * 0.5)
        })
        .collect::<Vec<_>>();

    for eps in [1.0e-2, 1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6] {
        for sign in [NufftSign::Positive, NufftSign::Negative] {
            let config = NufftConfig::new([16], eps).with_sign(sign);

            let type1 = NufftPlan::type1(config.clone()).unwrap();
            let actual1 = approximate_type1_1d(&type1, &coordinates, &strengths);
            let reference1 = reference_type1_f64(&config, &coordinates, &strengths).unwrap();
            let error1 = relative_l2(&actual1, &reference1);
            assert!(
                error1 <= 4.0 * eps,
                "type1 eps={eps} sign={sign:?}: relative l2={error1}"
            );

            let type2 = NufftPlan::type2(config.clone()).unwrap();
            let actual2 = approximate_type2_1d(&type2, &coordinates, &coefficients);
            let reference2 = reference_type2_f64(&config, &coordinates, &coefficients).unwrap();
            let error2 = relative_l2(&actual2, &reference2);
            assert!(
                error2 <= 4.0 * eps,
                "type2 eps={eps} sign={sign:?}: relative l2={error2}"
            );
        }
    }
}

#[test]
fn tensor_product_es_host_path_meets_finufft_acceptance_in_two_dimensions() {
    let coordinates = [
        -PI, PI, -2.71, -1.93, -0.37, 0.11, 0.0, 0.0, 0.0, 0.0, 0.43, -0.82, 2.63, 2.91,
    ];
    let strengths = coordinates
        .chunks_exact(2)
        .enumerate()
        .map(|(index, point)| {
            Complex64::new(
                (0.31 * point[0] - 0.17 * point[1]).cos() + index as f64 * 0.02,
                (0.23 * point[0] + 0.41 * point[1]).sin() - index as f64 * 0.03,
            )
        })
        .collect::<Vec<_>>();
    let coefficients = (0..12 * 20)
        .map(|index| {
            let x = index as f64 + 0.5;
            Complex64::new((0.071 * x).cos(), 0.5 * (0.047 * x).sin())
        })
        .collect::<Vec<_>>();

    for eps in [1.0e-2, 1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6] {
        for sign in [NufftSign::Positive, NufftSign::Negative] {
            let config = NufftConfig::new([12, 20], eps).with_sign(sign);

            let type1 = NufftPlan::type1(config.clone()).unwrap();
            let actual1 = approximate_type1_2d(&type1, &coordinates, &strengths);
            let reference1 = reference_type1_f64(&config, &coordinates, &strengths).unwrap();
            let error1 = relative_l2(&actual1, &reference1);
            assert!(
                error1 <= 8.0 * eps,
                "2D type1 eps={eps} sign={sign:?}: relative l2={error1}"
            );

            let type2 = NufftPlan::type2(config.clone()).unwrap();
            let actual2 = approximate_type2_2d(&type2, &coordinates, &coefficients);
            let reference2 = reference_type2_f64(&config, &coordinates, &coefficients).unwrap();
            let error2 = relative_l2(&actual2, &reference2);
            assert!(
                error2 <= 8.0 * eps,
                "2D type2 eps={eps} sign={sign:?}: relative l2={error2}"
            );
        }
    }
}

#[test]
fn tensor_product_es_host_type2_meets_finufft_acceptance_in_three_dimensions() {
    let coordinates = [
        -PI, PI, 0.0, -2.71, -1.93, 2.37, -0.37, 0.11, -0.29, 0.0, 0.0, 0.0, 0.43, -0.82, 1.17,
        2.63, 2.91, -2.45,
    ];
    let mode_shape = [6, 8, 10];
    let coefficients = (0..mode_shape.iter().product())
        .map(|index| {
            let x = index as f64 + 0.5;
            Complex64::new((0.071 * x).cos(), 0.5 * (0.047 * x).sin())
        })
        .collect::<Vec<_>>();

    for eps in [1.0e-2, 1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6] {
        for sign in [NufftSign::Positive, NufftSign::Negative] {
            let config = NufftConfig::new(mode_shape, eps).with_sign(sign);
            let type2 = NufftPlan::type2(config.clone()).unwrap();
            assert_eq!(type2.fine_grid_shape().len(), 3);
            let actual = approximate_type2_3d(&type2, &coordinates, &coefficients);
            let reference = reference_type2_f64(&config, &coordinates, &coefficients).unwrap();
            let error = relative_l2(&actual, &reference);
            assert!(
                error <= 32.0 * eps,
                "3D type2 eps={eps} sign={sign:?}: relative l2={error}"
            );
        }
    }
}

#[test]
fn many_vector_direct_references_are_transform_major_and_match_single_loops() {
    let points = [0.13, -0.27, 0.71, 0.43];
    let batched = NufftConfig::new([3, 2], 1.0e-6)
        .with_sign(NufftSign::Negative)
        .with_mode_order(ModeOrder::Fft)
        .with_batch(3);
    let single = batched.clone().with_batch(1);

    let strengths = (0..6)
        .map(|index| {
            let x = index as f64 + 0.25;
            Complex64::new((0.31 * x).sin(), (0.17 * x).cos())
        })
        .collect::<Vec<_>>();
    let type1 = reference_type1_f64(&batched, &points, &strengths).unwrap();
    let expected_type1 = strengths
        .chunks_exact(2)
        .flat_map(|vector| reference_type1_f64(&single, &points, vector).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(type1, expected_type1);
    assert_eq!(
        reference_type1_f64(&batched, &points, &strengths[..5]),
        Err(NufftError::InputLength {
            input: "type-1 strengths",
            expected: 6,
            actual: 5,
        })
    );

    let coefficients = (0..18)
        .map(|index| {
            let x = index as f64 + 0.5;
            Complex64::new((0.11 * x).cos(), (0.23 * x).sin())
        })
        .collect::<Vec<_>>();
    let type2 = reference_type2_f64(&batched, &points, &coefficients).unwrap();
    let expected_type2 = coefficients
        .chunks_exact(6)
        .flat_map(|vector| reference_type2_f64(&single, &points, vector).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(type2, expected_type2);
    assert_eq!(
        reference_type2_f64(&batched, &points, &coefficients[..17]),
        Err(NufftError::InputLength {
            input: "type-2 Fourier coefficients",
            expected: 18,
            actual: 17,
        })
    );
}

#[test]
fn batch_aware_buffer_sizes_preserve_legacy_single_vector_helpers() {
    let config = NufftConfig::new([3, 2], 1.0e-6).with_batch(3);
    let type1 = NufftPlan::type1(config.clone()).unwrap();
    let type2 = NufftPlan::type2(config).unwrap();

    assert_eq!(
        NufftPlan::required_type1_strength_buffer_size_bytes(2).unwrap(),
        16
    );
    assert_eq!(
        type1
            .required_type1_strength_buffer_size_bytes_for_batch(2, 2)
            .unwrap(),
        32
    );
    assert_eq!(
        type1.required_type1_output_buffer_size_bytes().unwrap(),
        144
    );
    assert_eq!(
        type1
            .required_type1_output_buffer_size_bytes_for_batch(2)
            .unwrap(),
        96
    );
    assert_eq!(
        type2
            .required_type2_coefficient_buffer_size_bytes()
            .unwrap(),
        144
    );
    assert_eq!(
        type2
            .required_type2_coefficient_buffer_size_bytes_for_batch(2)
            .unwrap(),
        96
    );
    assert_eq!(
        NufftPlan::required_type2_output_buffer_size_bytes(2).unwrap(),
        16
    );
    assert_eq!(
        type2
            .required_type2_output_buffer_size_bytes_for_batch(2, 3)
            .unwrap(),
        48
    );

    for active_batch in [0, 4] {
        assert_eq!(
            type1.required_type1_output_buffer_size_bytes_for_batch(active_batch),
            Err(NufftError::InvalidBatch {
                actual: active_batch,
                maximum: 3,
            })
        );
    }
}
