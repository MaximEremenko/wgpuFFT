use crate::error::{FftError, Result};
use crate::tuning::FftTuning;

/// Scalar precision used by an FFT plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum FftPrecision {
    /// Native 32-bit floating-point storage and arithmetic.
    #[default]
    F32,
    /// Native 64-bit floating-point storage and arithmetic.
    F64,
    /// Portable double-float arithmetic stored as unevaluated `f32` hi/lo pairs.
    ///
    /// This provides roughly 44-48 effective mantissa bits without requiring a
    /// device feature, while retaining the exponent range of `f32`.
    Df64,
}

impl FftPrecision {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F64 => "f64",
            Self::Df64 => "df64",
        }
    }

    pub const fn scalar_size_bytes(self) -> u64 {
        match self {
            Self::F32 => 4,
            Self::F64 | Self::Df64 => 8,
        }
    }

    pub const fn complex_size_bytes(self) -> u64 {
        self.scalar_size_bytes() * 2
    }
}

/// Direction of a complex-to-complex transform.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum FftDirection {
    #[default]
    Forward,
    Inverse,
}

impl FftDirection {
    /// The other direction.
    pub(crate) fn opposite(self) -> Self {
        match self {
            Self::Forward => Self::Inverse,
            Self::Inverse => Self::Forward,
        }
    }
}

/// Scaling policy applied by both CPU reference helpers and GPU execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Normalization {
    /// Do not scale either transform direction.
    None,
    /// Scale forward transforms by `1 / product(shape)`.
    Forward,
    /// Scale inverse transforms by `1 / product(shape)`.
    #[default]
    Inverse,
    /// Scale both directions by `1 / sqrt(product(shape))`.
    Orthogonal,
}

/// Configuration for transforms over interleaved complex buffers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FftConfig {
    shape: Vec<usize>,
    axes: Vec<usize>,
    batch: usize,
    direction: FftDirection,
    normalization: Normalization,
    precision: FftPrecision,
    tuning: FftTuning,
}

impl FftConfig {
    pub fn new(len: usize) -> Self {
        Self::new_nd([len])
    }

    pub fn new_nd(shape: impl Into<Vec<usize>>) -> Self {
        let shape = shape.into();
        let axes = (0..shape.len()).collect();
        Self {
            shape,
            axes,
            batch: 1,
            direction: FftDirection::Forward,
            normalization: Normalization::Inverse,
            precision: FftPrecision::F32,
            tuning: FftTuning::default(),
        }
    }

    pub fn inverse(len: usize) -> Self {
        Self::new(len).with_direction(FftDirection::Inverse)
    }

    pub fn inverse_nd(shape: impl Into<Vec<usize>>) -> Self {
        Self::new_nd(shape).with_direction(FftDirection::Inverse)
    }

    pub fn with_direction(mut self, direction: FftDirection) -> Self {
        self.direction = direction;
        self
    }

    pub fn with_normalization(mut self, normalization: Normalization) -> Self {
        self.normalization = normalization;
        self
    }

    pub fn with_axes(mut self, axes: impl Into<Vec<usize>>) -> Self {
        self.axes = axes.into();
        self
    }

    pub fn with_batch(mut self, batch: usize) -> Self {
        self.batch = batch;
        self
    }

    pub fn with_precision(mut self, precision: FftPrecision) -> Self {
        self.precision = precision;
        self
    }

    pub fn with_tuning(mut self, tuning: FftTuning) -> Self {
        self.tuning = tuning;
        self
    }

    pub fn len(&self) -> usize {
        self.shape.first().copied().unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool {
        self.shape.is_empty() || self.shape.contains(&0)
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn axes(&self) -> &[usize] {
        &self.axes
    }

    pub fn batch(&self) -> usize {
        self.batch
    }

    pub fn direction(&self) -> FftDirection {
        self.direction
    }

    pub fn normalization(&self) -> Normalization {
        self.normalization
    }

    pub fn precision(&self) -> FftPrecision {
        self.precision
    }

    pub fn tuning(&self) -> &FftTuning {
        &self.tuning
    }

    pub fn validate(&self) -> Result<()> {
        if self.shape.is_empty() || self.shape.contains(&0) {
            return Err(FftError::ZeroLength);
        }

        if self.batch == 0 {
            return Err(FftError::ZeroBatch);
        }

        if self.axes.is_empty() {
            return Err(FftError::EmptyAxes);
        }

        let rank = self.shape.len();
        let mut seen_axes = vec![false; rank];
        for &axis in &self.axes {
            if axis >= rank {
                return Err(FftError::InvalidAxis { axis, rank });
            }
            if seen_axes[axis] {
                return Err(FftError::DuplicateAxis { axis });
            }
            seen_axes[axis] = true;

            let axis_len = self.shape[axis];
            if axis_len == 1 && self.total_complex_len()? != 1 {
                return Err(FftError::UnsupportedLength { len: axis_len });
            }
        }

        self.total_complex_len()?;
        self.tuning.validate_for_config(self)?;
        Ok(())
    }

    pub fn len_u32(&self) -> Result<u32> {
        self.validate()?;
        Ok(self.len() as u32)
    }

    pub fn total_complex_len(&self) -> Result<usize> {
        checked_product(&self.shape)?
            .checked_mul(self.batch)
            .ok_or(FftError::LengthTooLarge { len: usize::MAX })
            .and_then(|total| {
                if total > u32::MAX as usize {
                    Err(FftError::LengthTooLarge { len: total })
                } else {
                    Ok(total)
                }
            })
    }

    pub fn total_complex_len_u32(&self) -> Result<u32> {
        Ok(self.total_complex_len()? as u32)
    }

    pub fn logical_complex_len(&self) -> Result<usize> {
        checked_product(&self.shape)
    }

    /// Number of interleaved scalar values in the logical complex buffer.
    pub fn required_scalar_len(&self) -> Result<usize> {
        self.validate()?;
        self.total_complex_len()?
            .checked_mul(2)
            .ok_or(FftError::LengthTooLarge { len: usize::MAX })
    }

    /// Legacy name for [`Self::required_scalar_len`].
    ///
    /// The return value is a scalar count for all precisions; use
    /// [`Self::required_buffer_size_bytes`] when allocating storage.
    pub fn required_f32_len(&self) -> Result<usize> {
        self.required_scalar_len()
    }

    pub fn required_buffer_size_bytes(&self) -> Result<u64> {
        self.validate()?;
        (self.total_complex_len()? as u64)
            .checked_mul(self.precision.complex_size_bytes())
            .ok_or(FftError::LengthTooLarge { len: usize::MAX })
    }

    pub fn scale(&self) -> Result<f32> {
        self.validate()?;
        let len = self.logical_complex_len()? as f32;
        let scale = match (self.direction, self.normalization) {
            (_, Normalization::None) => 1.0,
            (FftDirection::Forward, Normalization::Forward) => 1.0 / len,
            (FftDirection::Inverse, Normalization::Inverse) => 1.0 / len,
            (_, Normalization::Orthogonal) => 1.0 / len.sqrt(),
            _ => 1.0,
        };
        Ok(scale)
    }

    pub fn scale_f64(&self) -> Result<f64> {
        self.validate()?;
        let len = self.logical_complex_len()? as f64;
        let scale = match (self.direction, self.normalization) {
            (_, Normalization::None) => 1.0,
            (FftDirection::Forward, Normalization::Forward) => 1.0 / len,
            (FftDirection::Inverse, Normalization::Inverse) => 1.0 / len,
            (_, Normalization::Orthogonal) => 1.0 / len.sqrt(),
            _ => 1.0,
        };
        Ok(scale)
    }
}

fn checked_product(values: &[usize]) -> Result<usize> {
    let mut product = 1usize;
    for &value in values {
        product = product
            .checked_mul(value)
            .ok_or(FftError::LengthTooLarge { len: usize::MAX })?;
    }
    Ok(product)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tuning::{FftLargeRoute, FftTuning, FftTuningErrorKind};

    #[test]
    fn rejects_zero_length() {
        assert_eq!(FftConfig::new(0).validate(), Err(FftError::ZeroLength));
        assert_eq!(
            FftConfig::new_nd([2, 0]).validate(),
            Err(FftError::ZeroLength)
        );
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn rejects_total_complex_count_above_u32_index_space() {
        assert_eq!(
            FftConfig::new_nd([65_536, 65_536]).validate(),
            Err(FftError::LengthTooLarge { len: 4_294_967_296 })
        );
    }

    #[test]
    fn validates_nd_axes_and_batch() {
        assert_eq!(FftConfig::new_nd([2, 3]).shape(), &[2, 3]);
        assert_eq!(FftConfig::new_nd([2, 3]).axes(), &[0, 1]);
        assert_eq!(FftConfig::new_nd([2, 3]).with_axes([1]).axes(), &[1]);
        assert_eq!(FftConfig::new_nd([2, 3]).with_batch(4).batch(), 4);
        assert_eq!(
            FftConfig::new_nd([2, 3]).with_batch(0).validate(),
            Err(FftError::ZeroBatch)
        );
        assert_eq!(
            FftConfig::new_nd([2, 3])
                .with_axes(Vec::<usize>::new())
                .validate(),
            Err(FftError::EmptyAxes)
        );
        assert_eq!(
            FftConfig::new_nd([2, 3]).with_axes([2]).validate(),
            Err(FftError::InvalidAxis { axis: 2, rank: 2 })
        );
        assert_eq!(
            FftConfig::new_nd([2, 3]).with_axes([0, 0]).validate(),
            Err(FftError::DuplicateAxis { axis: 0 })
        );
        assert!(FftConfig::new(17).validate().is_ok());
        assert!(FftConfig::new_nd([8, 29, 34]).validate().is_ok());
    }

    #[test]
    fn computes_required_buffer_size() {
        let config = FftConfig::new(8);
        assert_eq!(config.required_f32_len().unwrap(), 16);
        assert_eq!(config.required_buffer_size_bytes().unwrap(), 64);

        let nd = FftConfig::new_nd([2, 3]).with_batch(4);
        assert_eq!(nd.total_complex_len().unwrap(), 24);
        assert_eq!(nd.required_f32_len().unwrap(), 48);
        assert_eq!(nd.required_buffer_size_bytes().unwrap(), 192);

        let f64 = nd.with_precision(FftPrecision::F64);
        assert_eq!(f64.required_scalar_len().unwrap(), 48);
        assert_eq!(f64.required_f32_len().unwrap(), 48);
        assert_eq!(f64.required_buffer_size_bytes().unwrap(), 384);

        let df64 = f64.with_precision(FftPrecision::Df64);
        assert_eq!(df64.required_scalar_len().unwrap(), 48);
        assert_eq!(df64.required_buffer_size_bytes().unwrap(), 384);
    }

    #[test]
    fn required_buffer_size_preserves_config_validation_errors() {
        assert_eq!(
            FftConfig::new(0).required_buffer_size_bytes(),
            Err(FftError::ZeroLength)
        );
        assert_eq!(
            FftConfig::new(8)
                .with_batch(0)
                .with_precision(FftPrecision::F64)
                .required_buffer_size_bytes(),
            Err(FftError::ZeroBatch)
        );
        assert_eq!(
            FftConfig::new_nd([4, 4])
                .with_axes(Vec::<usize>::new())
                .required_buffer_size_bytes(),
            Err(FftError::EmptyAxes)
        );
    }

    #[test]
    fn precision_defaults_to_f32_and_has_stable_element_sizes() {
        assert_eq!(FftConfig::new(8).precision(), FftPrecision::F32);
        assert_eq!(FftPrecision::F32.scalar_size_bytes(), 4);
        assert_eq!(FftPrecision::F32.complex_size_bytes(), 8);
        assert_eq!(FftPrecision::F64.scalar_size_bytes(), 8);
        assert_eq!(FftPrecision::F64.complex_size_bytes(), 16);
        assert_eq!(FftPrecision::Df64.scalar_size_bytes(), 8);
        assert_eq!(FftPrecision::Df64.complex_size_bytes(), 16);
        assert_eq!(FftPrecision::Df64.as_str(), "df64");
        assert_eq!(
            FftConfig::new(8)
                .with_precision(FftPrecision::F64)
                .precision(),
            FftPrecision::F64
        );
    }

    #[test]
    fn applies_normalization_by_direction() {
        assert_eq!(
            FftConfig::new(4)
                .with_normalization(Normalization::Forward)
                .scale()
                .unwrap(),
            0.25
        );
        assert_eq!(
            FftConfig::inverse(4)
                .with_normalization(Normalization::Forward)
                .scale()
                .unwrap(),
            1.0
        );
        assert_eq!(FftConfig::inverse(4).scale().unwrap(), 0.25);
        assert_eq!(
            FftConfig::new(4)
                .with_normalization(Normalization::Orthogonal)
                .scale()
                .unwrap(),
            0.5
        );
        assert_eq!(FftConfig::inverse_nd([2, 3]).scale().unwrap(), 1.0 / 6.0);
        assert_eq!(
            FftConfig::inverse_nd([2, 3]).scale_f64().unwrap(),
            1.0 / 6.0
        );
    }

    #[test]
    fn tuning_defaults_are_part_of_config_identity() {
        let implicit = FftConfig::new_nd([8, 17]);
        let explicit = implicit.clone().with_tuning(FftTuning::default());
        assert_eq!(implicit, explicit);
        assert_eq!(implicit.tuning(), &FftTuning::default());
        assert_eq!(implicit.validate(), Ok(()));
    }

    #[test]
    fn validates_scalar_tuning_controls() {
        for (tuning, field) in [
            (
                FftTuning::default().with_workgroup_size(0),
                "workgroup_size",
            ),
            (
                FftTuning::default().with_fused_workgroup_size(96),
                "fused_workgroup_size",
            ),
            (
                FftTuning::default().with_rader_max_prime(1),
                "rader_max_prime",
            ),
            (
                FftTuning::default().with_large_chunk_max_batches(0),
                "large_chunk_max_batches",
            ),
            (FftTuning::default().with_grouped_batch(0), "grouped_batch"),
            (
                FftTuning::default().with_segmented_burst_depth(4),
                "segmented_burst_depth",
            ),
            (
                FftTuning::default().with_max_storage_buffer_binding_size(0),
                "max_storage_buffer_binding_size",
            ),
            (
                FftTuning::default().with_max_buffer_size(0),
                "max_buffer_size",
            ),
            (
                FftTuning::default()
                    .with_swap_to_2_stage_4_step(4096)
                    .with_swap_to_3_stage_4_step(1024),
                "swap_to_2_stage_4_step",
            ),
        ] {
            assert!(matches!(
                FftConfig::new(8).with_tuning(tuning).validate(),
                Err(FftError::InvalidTuning {
                    kind: FftTuningErrorKind::InvalidValue
                        | FftTuningErrorKind::ConflictingValues,
                    field: actual,
                    ..
                }) if actual == field
            ));
        }
    }

    #[test]
    fn validates_forced_axis_tuning() {
        let invalid_cases = [
            (
                FftConfig::new_nd([8, 17])
                    .with_tuning(FftTuning::default().with_force_rader_axes([1, 1])),
                FftTuningErrorKind::DuplicateAxis,
                "force_rader_axes",
            ),
            (
                FftConfig::new_nd([8, 17]).with_tuning(
                    FftTuning::default()
                        .with_force_rader_axes([1])
                        .with_force_bluestein_axes([1]),
                ),
                FftTuningErrorKind::ConflictingAxes,
                "force_rader_axes/force_bluestein_axes",
            ),
            (
                FftConfig::new_nd([8, 17])
                    .with_tuning(FftTuning::default().with_force_bluestein_axes([2])),
                FftTuningErrorKind::AxisOutOfRange,
                "force_bluestein_axes",
            ),
            (
                FftConfig::new_nd([8, 17])
                    .with_axes([0])
                    .with_tuning(FftTuning::default().with_force_rader_axes([1])),
                FftTuningErrorKind::AxisNotSelected,
                "force_rader_axes",
            ),
            (
                FftConfig::new_nd([8, 17])
                    .with_tuning(FftTuning::default().with_force_rader_axes([0])),
                FftTuningErrorKind::AxisAlgorithmIncompatible,
                "force_rader_axes",
            ),
            (
                FftConfig::new(1).with_tuning(FftTuning::default().with_force_bluestein_axes([0])),
                FftTuningErrorKind::AxisAlgorithmIncompatible,
                "force_bluestein_axes",
            ),
        ];
        for (config, kind, field) in invalid_cases {
            assert!(matches!(
                config.validate(),
                Err(FftError::InvalidTuning {
                    kind: actual_kind,
                    field: actual_field,
                    ..
                }) if actual_kind == kind && actual_field == field
            ));
        }

        let valid = FftConfig::new_nd([8, 17]).with_tuning(
            FftTuning::default()
                .with_rader_max_prime(13)
                .with_force_bluestein_axes([0])
                .with_force_rader_axes([1])
                .with_large_route(FftLargeRoute::ForceChunk),
        );
        assert_eq!(valid.validate(), Ok(()));
    }
}
