use crate::config::{NufftSign, DEFAULT_EPS, DEFAULT_SIGMA};
use crate::error::{NufftError, Result};
use crate::gpu_type3::GpuType3Plan;
use crate::kernel::EsKernel;
use crate::Complex64;
use std::f64::consts::PI;
use std::fmt;

const CENTER_SNAP_FRACTION: f64 = 0.1;
const MAX_GRID_POINTS_U64: u64 = 1_000_000_000_000;

/// A closed coordinate interval supplied when constructing a reusable type-3 plan.
///
/// CPU and reference executions validate
/// every coordinate against them. A GPU-resident API cannot inspect arbitrary device
/// buffers while encoding, so it treats the caller-provided bounds as a contract.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NufftInterval {
    lower: f64,
    upper: f64,
}

impl NufftInterval {
    pub const fn new(lower: f64, upper: f64) -> Self {
        Self { lower, upper }
    }

    pub const fn lower(self) -> f64 {
        self.lower
    }

    pub const fn upper(self) -> f64 {
        self.upper
    }

    pub fn contains(self, value: f64) -> bool {
        value >= self.lower && value <= self.upper
    }

    fn validate(self, set: &'static str, axis: usize) -> Result<()> {
        if !self.lower.is_finite() || !self.upper.is_finite() {
            return Err(NufftError::InvalidType3Interval {
                set,
                axis,
                lower: self.lower,
                upper: self.upper,
                reason: "interval endpoints must be finite",
            });
        }
        if self.lower > self.upper {
            return Err(NufftError::InvalidType3Interval {
                set,
                axis,
                lower: self.lower,
                upper: self.upper,
                reason: "lower endpoint exceeds upper endpoint",
            });
        }
        Ok(())
    }

    fn half_width_and_center(self, set: &'static str, axis: usize) -> Result<(f64, f64)> {
        self.validate(set, axis)?;
        let mut half_width = (self.upper - self.lower) * 0.5;
        let mut center = (self.upper + self.lower) * 0.5;
        if !half_width.is_finite() || !center.is_finite() {
            return Err(NufftError::InvalidType3Interval {
                set,
                axis,
                lower: self.lower,
                upper: self.upper,
                reason: "interval width or center overflows f64",
            });
        }

        // This is the center-snapping rule. The comparison is deliberately
        // strict: exactly ten percent does not snap the center to zero.
        if center.abs() < CENTER_SNAP_FRACTION * half_width {
            half_width += center.abs();
            center = 0.0;
        }
        Ok((half_width, center))
    }
}

/// Mathematical and bounds configuration for a reusable type-3 NUFFT.
#[derive(Debug, Clone, PartialEq)]
pub struct NufftType3Config {
    source_bounds: Vec<NufftInterval>,
    target_bounds: Vec<NufftInterval>,
    eps: f64,
    sign: NufftSign,
    sigma: f64,
}

impl NufftType3Config {
    pub fn new(
        source_bounds: impl Into<Vec<NufftInterval>>,
        target_bounds: impl Into<Vec<NufftInterval>>,
        eps: f64,
    ) -> Self {
        Self {
            source_bounds: source_bounds.into(),
            target_bounds: target_bounds.into(),
            eps,
            sign: NufftSign::Positive,
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

    pub fn with_sigma(mut self, sigma: f64) -> Self {
        self.sigma = sigma;
        self
    }

    pub fn dimensions(&self) -> usize {
        self.source_bounds.len()
    }

    pub fn source_bounds(&self) -> &[NufftInterval] {
        &self.source_bounds
    }

    pub fn target_bounds(&self) -> &[NufftInterval] {
        &self.target_bounds
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

    pub fn sigma(&self) -> f64 {
        self.sigma
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
        if self.target_bounds.len() != dimensions {
            return Err(NufftError::InvalidDimensions {
                actual: self.target_bounds.len(),
                minimum: dimensions,
                maximum: dimensions,
            });
        }
        for (axis, &interval) in self.source_bounds.iter().enumerate() {
            interval.validate("source", axis)?;
        }
        for (axis, &interval) in self.target_bounds.iter().enumerate() {
            interval.validate("target", axis)?;
        }
        EsKernel::for_tolerance(self.eps, self.sigma)?;
        Ok(())
    }
}

impl Default for NufftType3Config {
    fn default() -> Self {
        Self::new(
            [NufftInterval::new(0.0, 0.0)],
            [NufftInterval::new(0.0, 0.0)],
            DEFAULT_EPS,
        )
    }
}

/// Rescaling metadata for one type-3 axis.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NufftType3AxisMetadata {
    source_half_width: f64,
    source_center: f64,
    target_half_width: f64,
    target_center: f64,
    outer_grid_length: usize,
    grid_spacing: f64,
    source_scale: f64,
}

impl NufftType3AxisMetadata {
    pub fn source_half_width(self) -> f64 {
        self.source_half_width
    }

    pub fn source_center(self) -> f64 {
        self.source_center
    }

    pub fn target_half_width(self) -> f64 {
        self.target_half_width
    }

    pub fn target_center(self) -> f64 {
        self.target_center
    }

    pub fn outer_grid_length(self) -> usize {
        self.outer_grid_length
    }

    /// Outer-grid spacing `h = 2*pi/nf`.
    pub fn grid_spacing(self) -> f64 {
        self.grid_spacing
    }

    /// Source rescaling factor `gamma`, where `x' = (x-C)/gamma`.
    pub fn source_scale(self) -> f64 {
        self.source_scale
    }

    pub fn rescale_source(self, coordinate: f64) -> f64 {
        (coordinate - self.source_center) / self.source_scale
    }

    pub fn rescale_target(self, frequency: f64) -> f64 {
        self.grid_spacing * self.source_scale * (frequency - self.target_center)
    }
}

/// Host-side plan metadata for type-3 rescaling and composition.
pub struct NufftType3Plan {
    config: NufftType3Config,
    kernel: EsKernel,
    axes: Vec<NufftType3AxisMetadata>,
    outer_grid_count: usize,
    gpu: Option<Box<GpuType3Plan>>,
}

impl fmt::Debug for NufftType3Plan {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("NufftType3Plan")
            .field("config", &self.config)
            .field("kernel", &self.kernel)
            .field("axes", &self.axes)
            .field("outer_grid_count", &self.outer_grid_count)
            .field("gpu_ready", &self.gpu.is_some())
            .finish()
    }
}

impl NufftType3Plan {
    pub fn new(config: NufftType3Config) -> Result<Self> {
        config.validate()?;
        let kernel = EsKernel::for_tolerance(config.eps, config.sigma)?;
        let maximum = maximum_type3_grid_points();
        let mut axes = Vec::new();
        axes.try_reserve_exact(config.dimensions()).map_err(|_| {
            NufftError::HostAllocationFailed {
                buffer: "type-3 axis metadata",
                elements: config.dimensions(),
            }
        })?;

        let mut outer_grid_count = 1usize;
        for axis in 0..config.dimensions() {
            let (source_half_width, source_center) =
                config.source_bounds[axis].half_width_and_center("source", axis)?;
            let (target_half_width, target_center) =
                config.target_bounds[axis].half_width_and_center("target", axis)?;
            let metadata = select_axis_metadata(
                axis,
                (source_half_width, source_center),
                (target_half_width, target_center),
                config.sigma,
                kernel.width(),
                maximum,
            )?;
            let requested = (outer_grid_count as u128) * (metadata.outer_grid_length as u128);
            if requested > maximum as u128 {
                return Err(NufftError::Type3GridVolumeUnsupported { requested, maximum });
            }
            outer_grid_count *= metadata.outer_grid_length;
            axes.push(metadata);
        }

        Ok(Self {
            config,
            kernel,
            axes,
            outer_grid_count,
            gpu: None,
        })
    }

    /// Builds reusable GPU resources for a portable `f32` type-3 NUFFT.
    ///
    /// Source and target coordinates remain caller-owned and GPU-resident.
    /// Their values must stay within the conservative intervals supplied in
    /// `config`; command encoding cannot inspect device-buffer contents to
    /// enforce that contract without a separate validation pass.
    pub fn new_gpu(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        config: NufftType3Config,
    ) -> Result<Self> {
        let mut plan = Self::new(config)?;
        let gpu = GpuType3Plan::new(device, queue, &plan)?;
        debug_assert_eq!(gpu.dimensions(), plan.config.dimensions());
        plan.gpu = Some(Box::new(gpu));
        Ok(plan)
    }

    pub fn config(&self) -> &NufftType3Config {
        &self.config
    }

    pub fn kernel(&self) -> EsKernel {
        self.kernel
    }

    pub fn axes(&self) -> &[NufftType3AxisMetadata] {
        &self.axes
    }

    pub fn outer_grid_shape(&self) -> Vec<usize> {
        self.axes
            .iter()
            .map(|axis| axis.outer_grid_length)
            .collect()
    }

    pub fn outer_grid_count(&self) -> usize {
        self.outer_grid_count
    }

    pub fn is_gpu_ready(&self) -> bool {
        self.gpu.is_some()
    }

    pub fn required_source_point_buffer_size_bytes(&self, source_count: usize) -> Result<u64> {
        coordinate_buffer_size_bytes(
            "type-3 source point buffer",
            source_count,
            self.config.dimensions(),
        )
    }

    pub fn required_target_point_buffer_size_bytes(&self, target_count: usize) -> Result<u64> {
        coordinate_buffer_size_bytes(
            "type-3 target point buffer",
            target_count,
            self.config.dimensions(),
        )
    }

    pub fn required_strength_buffer_size_bytes(&self, source_count: usize) -> Result<u64> {
        complex_buffer_size_bytes("type-3 source strength buffer", source_count)
    }

    pub fn required_output_buffer_size_bytes(&self, target_count: usize) -> Result<u64> {
        complex_buffer_size_bytes("type-3 output buffer", target_count)
    }

    /// Records a 1D, 2D, or 3D type-3 NUFFT into `encoder` without submission
    /// or host readback.
    ///
    /// Coordinates are point-major `f32`; strengths and output are complex
    /// `f32` pairs. Every buffer must include `STORAGE` usage. Ordered
    /// executions may reuse this plan; its grow-only scratch is overwritten on
    /// each call.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_gpu(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        source_count: usize,
        source_points: &wgpu::Buffer,
        strengths: &wgpu::Buffer,
        target_count: usize,
        target_points: &wgpu::Buffer,
        output: &wgpu::Buffer,
    ) -> Result<()> {
        let gpu = self
            .gpu
            .as_ref()
            .ok_or(NufftError::GpuExecutionUnavailable {
                kind: "type-3",
                reason: "plan was created without GPU resources",
            })?;
        gpu.encode(
            device,
            encoder,
            source_count,
            source_points,
            strengths,
            target_count,
            target_points,
            output,
        )
    }

    /// Executes the direct `O(M*K)` `f64` type-3 NDFT for reference or tiny
    /// problems using this plan's exact sign and bounds contract.
    pub fn execute_cpu(
        &self,
        source_coordinates: &[f64],
        target_frequencies: &[f64],
        strengths: &[Complex64],
    ) -> Result<Vec<Complex64>> {
        reference_type3_f64(
            &self.config,
            source_coordinates,
            target_frequencies,
            strengths,
        )
    }

    /// Sum of the pre-phase angles `D*x` across axes, without the `isign`.
    pub fn source_prephase_argument(&self, source: &[f64]) -> Result<f64> {
        self.validate_single_point("source", source)?;
        Ok(self
            .axes
            .iter()
            .zip(source)
            .map(|(axis, &x)| axis.target_center * x)
            .sum())
    }

    /// Sum of the post-phase angles `C*(s-D)` across axes, without `isign`.
    pub fn target_postphase_argument(&self, target: &[f64]) -> Result<f64> {
        self.validate_single_point("target", target)?;
        Ok(self
            .axes
            .iter()
            .zip(target)
            .map(|(axis, &s)| axis.source_center * (s - axis.target_center))
            .sum())
    }

    /// Separable continuous ES-kernel transform at a rescaled target point.
    pub fn target_kernel_fourier_transform(&self, target: &[f64]) -> Result<f64> {
        self.validate_single_point("target", target)?;
        self.axes
            .iter()
            .zip(target)
            .try_fold(1.0, |product, (axis, &s)| {
                Ok(product * es_kernel_fourier_transform(self.kernel, axis.rescale_target(s))?)
            })
    }

    fn validate_single_point(&self, set: &'static str, point: &[f64]) -> Result<()> {
        if point.len() != self.config.dimensions() {
            return Err(NufftError::Type3CoordinateLength {
                set,
                actual: point.len(),
                dimensions: self.config.dimensions(),
            });
        }
        let bounds = if set == "source" {
            &self.config.source_bounds
        } else {
            &self.config.target_bounds
        };
        validate_type3_coordinates(set, self.config.dimensions(), point, bounds).map(|_| ())
    }
}

/// Direct `O(M*K)` type-3 NDFT:
/// `f_k = sum_j c_j exp(isign * i * s_k dot x_j)`.
pub fn reference_type3_f64(
    config: &NufftType3Config,
    source_coordinates: &[f64],
    target_frequencies: &[f64],
    strengths: &[Complex64],
) -> Result<Vec<Complex64>> {
    config.validate()?;
    let dimensions = config.dimensions();
    let source_count = validate_type3_coordinates(
        "source",
        dimensions,
        source_coordinates,
        &config.source_bounds,
    )?;
    let target_count = validate_type3_coordinates(
        "target",
        dimensions,
        target_frequencies,
        &config.target_bounds,
    )?;
    if strengths.len() != source_count {
        return Err(NufftError::InputLength {
            input: "type-3 source strengths",
            expected: source_count,
            actual: strengths.len(),
        });
    }
    validate_complex("type-3 source strengths", strengths)?;

    let mut output = Vec::new();
    output
        .try_reserve_exact(target_count)
        .map_err(|_| NufftError::HostAllocationFailed {
            buffer: "type-3 direct output",
            elements: target_count,
        })?;
    for target in target_frequencies.chunks_exact(dimensions) {
        let mut sum = Complex64::default();
        for (source, &strength) in source_coordinates.chunks_exact(dimensions).zip(strengths) {
            let phase = source.iter().zip(target).map(|(&x, &s)| x * s).sum::<f64>();
            add_rotated(&mut sum, strength, config.sign.multiplier() * phase);
        }
        output.push(sum);
    }
    Ok(output)
}

/// Continuous, unshifted Fourier transform of the one-dimensional ES kernel.
///
/// The integral is evaluated on the positive half-support using the same
/// Gauss-Legendre construction as the arbitrary-frequency type-3 kernel
/// correction, then reflected through the kernel's even symmetry.
pub fn es_kernel_fourier_transform(kernel: EsKernel, frequency: f64) -> Result<f64> {
    if !frequency.is_finite() {
        return Err(NufftError::InvalidType3KernelFrequency { frequency });
    }
    let half_width = kernel.half_width();
    if !(frequency * half_width).is_finite() {
        return Err(NufftError::InvalidType3KernelFrequency { frequency });
    }
    let positive_nodes = kernel.width() + 2;
    let (nodes, weights) = gauss_legendre(positive_nodes * 2);
    Ok(nodes[positive_nodes..]
        .iter()
        .zip(&weights[positive_nodes..])
        .map(|(&node, &weight)| {
            let distance = half_width * node;
            2.0 * half_width * weight * kernel.evaluate(distance) * (frequency * distance).cos()
        })
        .sum())
}

fn select_axis_metadata(
    axis: usize,
    source: (f64, f64),
    target: (f64, f64),
    sigma: f64,
    kernel_width: usize,
    maximum: usize,
) -> Result<NufftType3AxisMetadata> {
    let (source_half_width, source_center) = source;
    let (target_half_width, target_center) = target;
    let mut x = source_half_width;
    let mut s = target_half_width;
    if x == 0.0 {
        if s == 0.0 {
            x = 1.0;
            s = 1.0;
        } else {
            x = 1.0 / s;
        }
    } else {
        s = s.max(1.0 / x);
    }

    let requested = 2.0 * sigma * s * x / PI + (kernel_width + 1) as f64;
    if !requested.is_finite() || requested > maximum as f64 {
        return Err(NufftError::Type3GridSizeUnsupported {
            axis,
            requested,
            maximum,
            reason: "source-target space-bandwidth product is too large",
        });
    }
    let minimum = (requested as usize).max(kernel_width * 2);
    let outer_grid_length = next_even_235(minimum, axis, maximum)?;
    let grid_spacing = 2.0 * PI / outer_grid_length as f64;
    let source_scale = outer_grid_length as f64 / (2.0 * sigma * s);
    if !source_scale.is_finite() || source_scale <= 0.0 {
        return Err(NufftError::Type3RescalingUnsupported {
            axis,
            quantity: "source rescaling factor gamma",
            value: source_scale,
            reason: "the planned interval widths produce a non-finite or non-positive scale",
        });
    }
    Ok(NufftType3AxisMetadata {
        source_half_width,
        source_center,
        target_half_width,
        target_center,
        outer_grid_length,
        grid_spacing,
        source_scale,
    })
}

fn next_even_235(minimum: usize, axis: usize, maximum: usize) -> Result<usize> {
    let minimum = minimum.max(2);
    let mut best = None;
    let mut power_of_two = 2usize;
    while power_of_two <= maximum {
        let mut times_three = power_of_two;
        while times_three <= maximum {
            let mut candidate = times_three;
            while candidate <= maximum {
                if candidate >= minimum {
                    best = Some(best.map_or(candidate, |old: usize| old.min(candidate)));
                    break;
                }
                let Some(next) = candidate.checked_mul(5) else {
                    break;
                };
                candidate = next;
            }
            let Some(next) = times_three.checked_mul(3) else {
                break;
            };
            times_three = next;
        }
        let Some(next) = power_of_two.checked_mul(2) else {
            break;
        };
        power_of_two = next;
    }
    best.ok_or(NufftError::Type3GridSizeUnsupported {
        axis,
        requested: minimum as f64,
        maximum,
        reason: "no even 2,3,5-smooth grid length fits the limit",
    })
}

fn maximum_type3_grid_points() -> usize {
    usize::try_from(MAX_GRID_POINTS_U64).unwrap_or(usize::MAX)
}

fn validate_type3_coordinates(
    set: &'static str,
    dimensions: usize,
    coordinates: &[f64],
    bounds: &[NufftInterval],
) -> Result<usize> {
    if !coordinates.len().is_multiple_of(dimensions) {
        return Err(NufftError::Type3CoordinateLength {
            set,
            actual: coordinates.len(),
            dimensions,
        });
    }
    for (index, &coordinate) in coordinates.iter().enumerate() {
        let point = index / dimensions;
        let axis = index % dimensions;
        if !coordinate.is_finite() {
            return Err(NufftError::Type3NonFiniteCoordinate {
                set,
                point,
                axis,
                value: coordinate,
            });
        }
        let interval = bounds[axis];
        if !interval.contains(coordinate) {
            return Err(NufftError::Type3CoordinateOutOfBounds {
                set,
                point,
                axis,
                value: coordinate,
                lower: interval.lower,
                upper: interval.upper,
            });
        }
    }
    Ok(coordinates.len() / dimensions)
}

fn coordinate_buffer_size_bytes(
    context: &'static str,
    point_count: usize,
    dimensions: usize,
) -> Result<u64> {
    let scalar_count = point_count
        .checked_mul(dimensions)
        .ok_or(NufftError::LengthOverflow { context })?;
    checked_type3_buffer_size(context, scalar_count, 4)
}

fn complex_buffer_size_bytes(context: &'static str, count: usize) -> Result<u64> {
    checked_type3_buffer_size(context, count, 8)
}

fn checked_type3_buffer_size(
    context: &'static str,
    count: usize,
    bytes_per_element: u64,
) -> Result<u64> {
    u64::try_from(count)
        .ok()
        .and_then(|count| count.checked_mul(bytes_per_element))
        .ok_or(NufftError::LengthOverflow { context })
}

fn validate_complex(input: &'static str, values: &[Complex64]) -> Result<()> {
    for (index, value) in values.iter().enumerate() {
        if !value.re.is_finite() || !value.im.is_finite() {
            return Err(NufftError::NonFiniteComplex { input, index });
        }
    }
    Ok(())
}

fn add_rotated(sum: &mut Complex64, value: Complex64, angle: f64) {
    let (sin, cos) = angle.sin_cos();
    sum.re += value.re * cos - value.im * sin;
    sum.im += value.re * sin + value.im * cos;
}

fn gauss_legendre(order: usize) -> (Vec<f64>, Vec<f64>) {
    let mut nodes = vec![0.0; order];
    let mut weights = vec![0.0; order];
    for index in 0..order.div_ceil(2) {
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
    fn direct_type3_oracle_matches_one_source_analytic_cases_in_all_dimensions() {
        for dimensions in 1..=3 {
            let source = [0.37, -0.21, 0.49][..dimensions].to_vec();
            let target = [-1.3, 0.72, 1.11][..dimensions].to_vec();
            let source_bounds = source
                .iter()
                .map(|&value| NufftInterval::new(value - 0.5, value + 0.5))
                .collect::<Vec<_>>();
            let target_bounds = target
                .iter()
                .map(|&value| NufftInterval::new(value - 0.5, value + 0.5))
                .collect::<Vec<_>>();
            let strength = Complex64::new(0.7, -0.4);
            for isign in [-1, 1] {
                let config =
                    NufftType3Config::new(source_bounds.clone(), target_bounds.clone(), 1.0e-6)
                        .with_isign(isign);
                let actual = reference_type3_f64(&config, &source, &target, &[strength]).unwrap();
                let angle = isign as f64
                    * source
                        .iter()
                        .zip(&target)
                        .map(|(&x, &s)| x * s)
                        .sum::<f64>();
                let expected = rotate(strength, angle);
                assert_complex_close(actual[0], expected, 8.0 * f64::EPSILON);
            }
        }
    }

    #[test]
    fn direct_type3_oracle_obeys_the_opposite_sign_adjoint_identity() {
        for dimensions in 1..=3 {
            let source_coordinates = [
                -0.71, 0.23, 0.41, //
                0.13, -0.57, 0.82, //
                0.64, 0.38, -0.29,
            ]
            .chunks_exact(3)
            .flat_map(|point| point[..dimensions].iter().copied())
            .collect::<Vec<_>>();
            let target_frequencies = [
                -1.31, 0.47, 0.93, //
                0.22, -0.76, 1.17, //
                1.09, 0.31, -0.58, //
                -0.43, 0.88, 0.14,
            ]
            .chunks_exact(3)
            .flat_map(|point| point[..dimensions].iter().copied())
            .collect::<Vec<_>>();
            let source_bounds = (0..dimensions)
                .map(|_| NufftInterval::new(-1.0, 1.0))
                .collect::<Vec<_>>();
            let target_bounds = (0..dimensions)
                .map(|_| NufftInterval::new(-1.5, 1.5))
                .collect::<Vec<_>>();
            let v = [
                Complex64::new(0.7, -0.2),
                Complex64::new(-0.4, 0.8),
                Complex64::new(0.3, 0.5),
            ];
            let u = [
                Complex64::new(-0.6, 0.1),
                Complex64::new(0.2, -0.9),
                Complex64::new(0.4, 0.7),
                Complex64::new(-0.3, -0.5),
            ];
            for sign in [NufftSign::Positive, NufftSign::Negative] {
                let forward =
                    NufftType3Config::new(source_bounds.clone(), target_bounds.clone(), 1.0e-12)
                        .with_sign(sign);
                let adjoint =
                    NufftType3Config::new(target_bounds.clone(), source_bounds.clone(), 1.0e-12)
                        .with_sign(match sign {
                            NufftSign::Positive => NufftSign::Negative,
                            NufftSign::Negative => NufftSign::Positive,
                        });
                let av =
                    reference_type3_f64(&forward, &source_coordinates, &target_frequencies, &v)
                        .unwrap();
                let astar_u =
                    reference_type3_f64(&adjoint, &target_frequencies, &source_coordinates, &u)
                        .unwrap();
                let lhs = hermitian_dot(&u, &av);
                let rhs = hermitian_dot(&astar_u, &v);
                let residual = ((lhs.re - rhs.re).hypot(lhs.im - rhs.im))
                    / lhs.re.hypot(lhs.im).max(rhs.re.hypot(rhs.im)).max(1.0);
                assert!(
                    residual <= 2.0e-15,
                    "dimensions={dimensions}, sign={sign:?}, residual={residual}"
                );
            }
        }
    }

    #[test]
    fn interval_center_snap_uses_strict_ten_percent_threshold() {
        let below = NufftInterval::new(-9.001, 10.999)
            .half_width_and_center("source", 0)
            .unwrap();
        assert_eq!(below.1, 0.0);
        assert!((below.0 - 10.999).abs() < 2.0e-14);

        let threshold = NufftInterval::new(-9.0, 11.0)
            .half_width_and_center("source", 0)
            .unwrap();
        assert_eq!(threshold, (10.0, 1.0));
    }

    #[test]
    fn degenerate_bounds_follow_the_space_bandwidth_clamp() {
        let both_zero = NufftType3Plan::new(NufftType3Config::new(
            [NufftInterval::new(2.0, 2.0)],
            [NufftInterval::new(-3.0, -3.0)],
            1.0e-6,
        ))
        .unwrap();
        let axis = both_zero.axes()[0];
        assert_eq!(axis.source_half_width(), 0.0);
        assert_eq!(axis.target_half_width(), 0.0);
        assert_eq!(axis.source_center(), 2.0);
        assert_eq!(axis.target_center(), -3.0);
        assert!(axis.outer_grid_length().is_multiple_of(2));
        assert_235_smooth(axis.outer_grid_length());
        assert!((axis.source_scale() - axis.outer_grid_length() as f64 / 4.0).abs() < 1.0e-14);

        let target_only = NufftType3Plan::new(NufftType3Config::new(
            [NufftInterval::new(4.0, 4.0)],
            [NufftInterval::new(-5.0, 5.0)],
            1.0e-6,
        ))
        .unwrap();
        let axis = target_only.axes()[0];
        assert!((axis.source_scale() - axis.outer_grid_length() as f64 / 20.0).abs() < 1.0e-14);
    }

    #[test]
    fn rescaling_phase_identity_holds_per_axis() {
        let plan = NufftType3Plan::new(NufftType3Config::new(
            [NufftInterval::new(1.0, 5.0), NufftInterval::new(-4.0, 1.0)],
            [NufftInterval::new(-3.0, 7.0), NufftInterval::new(2.0, 8.0)],
            1.0e-6,
        ))
        .unwrap();
        let source = [3.7, -1.25];
        let target = [5.2, 6.1];
        let direct = source.iter().zip(target).map(|(&x, s)| x * s).sum::<f64>();
        let split = plan.source_prephase_argument(&source).unwrap()
            + plan.target_postphase_argument(&target).unwrap()
            + plan
                .axes()
                .iter()
                .zip(source)
                .zip(target)
                .map(|((axis, x), s)| {
                    axis.rescale_source(x) * axis.rescale_target(s) / axis.grid_spacing()
                })
                .sum::<f64>();
        assert!((direct - split).abs() <= 8.0 * f64::EPSILON * direct.abs().max(1.0));
    }

    #[test]
    fn nonfinite_bounds_and_absurd_grids_are_structured_errors() {
        let nonfinite = NufftType3Config::new(
            [NufftInterval::new(f64::NEG_INFINITY, 1.0)],
            [NufftInterval::new(-1.0, 1.0)],
            1.0e-6,
        );
        assert!(matches!(
            NufftType3Plan::new(nonfinite),
            Err(NufftError::InvalidType3Interval { .. })
        ));

        let huge = NufftType3Config::new(
            [NufftInterval::new(0.0, 1.0e308)],
            [NufftInterval::new(0.0, 1.0e308)],
            1.0e-6,
        );
        assert!(matches!(
            NufftType3Plan::new(huge),
            Err(NufftError::Type3GridSizeUnsupported { .. })
        ));

        let huge_volume = NufftType3Config::new(
            [
                NufftInterval::new(-1.0e4, 1.0e4),
                NufftInterval::new(-1.0e4, 1.0e4),
                NufftInterval::new(-1.0e4, 1.0e4),
            ],
            [
                NufftInterval::new(-1.0e4, 1.0e4),
                NufftInterval::new(-1.0e4, 1.0e4),
                NufftInterval::new(-1.0e4, 1.0e4),
            ],
            1.0e-6,
        );
        assert!(matches!(
            NufftType3Plan::new(huge_volume),
            Err(NufftError::Type3GridVolumeUnsupported { .. })
        ));

        let infinite_scale = NufftType3Config::new(
            [NufftInterval::new(0.0, 1.7e308)],
            [NufftInterval::new(0.0, 0.0)],
            1.0e-6,
        );
        assert!(matches!(
            NufftType3Plan::new(infinite_scale),
            Err(NufftError::Type3RescalingUnsupported {
                quantity: "source rescaling factor gamma",
                ..
            })
        ));
    }

    #[test]
    fn direct_oracle_rejects_coordinates_outside_planned_bounds() {
        let config = NufftType3Config::new(
            [NufftInterval::new(-1.0, 1.0)],
            [NufftInterval::new(-2.0, 2.0)],
            1.0e-6,
        );
        assert!(matches!(
            reference_type3_f64(&config, &[1.000_000_1], &[0.0], &[Complex64::new(1.0, 0.0)]),
            Err(NufftError::Type3CoordinateOutOfBounds { set: "source", .. })
        ));
    }

    #[test]
    fn arbitrary_frequency_kernel_transform_matches_dense_quadrature() {
        let kernel = EsKernel::for_tolerance(1.0e-6, 2.0).unwrap();
        for frequency in [0.0, 0.23, 1.7, -2.4] {
            let actual = es_kernel_fourier_transform(kernel, frequency).unwrap();
            let expected = simpson_kernel_transform(kernel, frequency, 200_000);
            let relative = (actual - expected).abs() / expected.abs().max(1.0);
            assert!(
                relative < 3.0e-8,
                "frequency={frequency}, actual={actual}, expected={expected}, relative={relative}"
            );
        }
    }

    #[test]
    fn arbitrary_frequency_kernel_transform_rejects_unrepresentable_phase() {
        let kernel = EsKernel::for_tolerance(1.0e-6, 2.0).unwrap();
        assert!(matches!(
            es_kernel_fourier_transform(kernel, f64::MAX),
            Err(NufftError::InvalidType3KernelFrequency {
                frequency: f64::MAX
            })
        ));
    }

    fn rotate(value: Complex64, angle: f64) -> Complex64 {
        let (sin, cos) = angle.sin_cos();
        Complex64::new(
            value.re * cos - value.im * sin,
            value.re * sin + value.im * cos,
        )
    }

    fn hermitian_dot(left: &[Complex64], right: &[Complex64]) -> Complex64 {
        left.iter()
            .zip(right)
            .fold(Complex64::default(), |mut sum, (left, right)| {
                sum.re += left.re * right.re + left.im * right.im;
                sum.im += left.re * right.im - left.im * right.re;
                sum
            })
    }

    fn assert_complex_close(actual: Complex64, expected: Complex64, tolerance: f64) {
        assert!((actual.re - expected.re).abs() <= tolerance);
        assert!((actual.im - expected.im).abs() <= tolerance);
    }

    fn assert_235_smooth(mut length: usize) {
        for radix in [2, 3, 5] {
            while length.is_multiple_of(radix) {
                length /= radix;
            }
        }
        assert_eq!(length, 1);
    }

    fn simpson_kernel_transform(kernel: EsKernel, frequency: f64, intervals: usize) -> f64 {
        assert!(intervals.is_multiple_of(2));
        let half_width = kernel.half_width();
        let step = half_width / intervals as f64;
        let mut sum = kernel.evaluate(0.0);
        for index in 1..intervals {
            let x = index as f64 * step;
            let value = kernel.evaluate(x) * (frequency * x).cos();
            sum += if index.is_multiple_of(2) {
                2.0 * value
            } else {
                4.0 * value
            };
        }
        2.0 * step * sum / 3.0
    }
}
