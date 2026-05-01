use crate::error::{FftError, Result};

pub(crate) mod axis_plan;
pub mod axis_policy;
pub(crate) mod bluestein_axis;
pub mod buffer_view;
pub mod c2c;
pub(crate) mod large_bridge;
pub(crate) mod large_chunk;
pub(crate) mod large_graph;
pub mod large_policy;
pub mod logical_io;
pub(crate) mod nd_wgsl;
pub mod pipeline_cache;
pub(crate) mod rader_axis;
pub mod real;
pub(crate) mod smooth_decompose;
pub(crate) mod stage_executor;
pub(crate) mod window_scheduler;

pub const SUPPORTED_RADICES: &[usize] = &[2, 3, 4, 5, 7, 8, 11, 13];

const FACTORIZATION_ORDER: &[usize] = &[13, 11, 8, 7, 5, 4, 3, 2];

pub fn factor_supported_length(len: usize) -> Result<Vec<usize>> {
    if len == 0 {
        return Err(FftError::ZeroLength);
    }

    let mut remaining = len;
    let mut factors = Vec::new();

    while remaining > 1 {
        let Some(&radix) = FACTORIZATION_ORDER
            .iter()
            .find(|&&candidate| remaining % candidate == 0)
        else {
            return Err(FftError::UnsupportedLength { len });
        };

        factors.push(radix);
        remaining /= radix;
    }

    Ok(factors)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn product(values: &[usize]) -> usize {
        values.iter().product()
    }

    #[test]
    fn factors_supported_acceptance_lengths() {
        for len in [2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 15, 16, 21] {
            let factors = factor_supported_length(len).unwrap();
            assert_eq!(product(&factors), len);
            assert!(factors.iter().all(|f| SUPPORTED_RADICES.contains(f)));
        }
    }

    #[test]
    fn rejects_unsupported_prime_for_this_milestone() {
        assert_eq!(
            factor_supported_length(17),
            Err(FftError::UnsupportedLength { len: 17 })
        );
    }
}
