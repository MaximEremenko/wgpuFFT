use crate::error::{FftError, Result};

/// Scalar precision used by an FFT plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum FftPrecision {
    /// Native 32-bit floating-point storage and arithmetic.
    #[default]
    F32,
    /// Native 64-bit floating-point storage and arithmetic.
    F64,
}

impl FftPrecision {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F64 => "f64",
        }
    }

    pub const fn scalar_size_bytes(self) -> u64 {
        match self {
            Self::F32 => 4,
            Self::F64 => 8,
        }
    }

    pub const fn complex_size_bytes(self) -> u64 {
        self.scalar_size_bytes() * 2
    }
}

/// Direction of a complex-to-complex transform.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FftDirection {
    Forward,
    Inverse,
}

impl Default for FftDirection {
    fn default() -> Self {
        Self::Forward
    }
}

/// Scaling policy applied by both CPU reference helpers and GPU execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Normalization {
    /// Do not scale either transform direction.
    None,
    /// Scale forward transforms by `1 / product(shape)`.
    Forward,
    /// Scale inverse transforms by `1 / product(shape)`.
    Inverse,
    /// Scale both directions by `1 / sqrt(product(shape))`.
    Orthogonal,
}

impl Default for Normalization {
    fn default() -> Self {
        Self::Inverse
    }
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

    pub fn validate(&self) -> Result<()> {
        if self.shape.is_empty() || self.shape.iter().any(|&len| len == 0) {
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
    /// The return value is a scalar count for both precisions; use
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
}
