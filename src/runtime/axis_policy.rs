use crate::error::{FftError, Result};
use crate::runtime::SUPPORTED_RADICES;

pub const DEFAULT_RADER_MAX_PRIME: usize = 4096;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AxisKind {
    Mixed,
    Rader,
    Bluestein,
}

impl AxisKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Mixed => "mixed",
            Self::Rader => "rader",
            Self::Bluestein => "bluestein",
        }
    }
}

pub fn resolve_axis_kinds_for_shape(shape: &[usize]) -> Result<Vec<AxisKind>> {
    resolve_axis_kinds_for_axes(shape, &(0..shape.len()).collect::<Vec<_>>())
}

pub(crate) fn resolve_axis_kinds_for_axes(
    shape: &[usize],
    axes: &[usize],
) -> Result<Vec<AxisKind>> {
    if shape.is_empty() || shape.iter().any(|&len| len == 0) {
        return Err(FftError::ZeroLength);
    }

    let rank = shape.len();
    let mut kinds = Vec::with_capacity(axes.len());
    for &axis in axes {
        if axis >= rank {
            return Err(FftError::InvalidAxis { axis, rank });
        }
        kinds.push(axis_kind_for_len(shape[axis]));
    }
    Ok(kinds)
}

pub(crate) fn axis_kind_for_len(len: usize) -> AxisKind {
    if crate::runtime::factor_supported_length(len).is_ok() {
        AxisKind::Mixed
    } else if is_prime(len) && len <= DEFAULT_RADER_MAX_PRIME {
        AxisKind::Rader
    } else {
        AxisKind::Bluestein
    }
}

pub(crate) fn is_prime(n: usize) -> bool {
    if n < 2 {
        return false;
    }
    if n % 2 == 0 {
        return n == 2;
    }

    let mut d = 3usize;
    while d <= n / d {
        if n % d == 0 {
            return false;
        }
        d += 2;
    }
    true
}

pub(crate) fn mod_pow(mut base: usize, mut exp: usize, modulus: usize) -> usize {
    debug_assert!(modulus > 0);
    let mut result = 1usize;
    base %= modulus;

    while exp > 0 {
        if exp & 1 == 1 {
            result = result * base % modulus;
        }
        base = base * base % modulus;
        exp >>= 1;
    }

    result
}

pub(crate) fn prime_factors(mut n: usize) -> Vec<usize> {
    let mut factors = Vec::new();
    let mut d = 2usize;
    while d <= n / d {
        if n % d == 0 {
            factors.push(d);
            while n % d == 0 {
                n /= d;
            }
        }
        d += if d == 2 { 1 } else { 2 };
    }
    if n > 1 {
        factors.push(n);
    }
    factors
}

pub(crate) fn primitive_root_prime(prime: usize) -> Option<usize> {
    if !is_prime(prime) {
        return None;
    }
    if prime == 2 {
        return Some(1);
    }

    let phi = prime - 1;
    let factors = prime_factors(phi);
    'candidate: for root in 2..prime {
        for &factor in &factors {
            if mod_pow(root, phi / factor, prime) == 1 {
                continue 'candidate;
            }
        }
        return Some(root);
    }
    None
}

pub(crate) fn next_smooth_at_least(min: usize) -> usize {
    let mut candidate = min.max(1);
    while !is_smooth_supported(candidate) {
        candidate += 1;
    }
    candidate
}

pub(crate) fn next_power_of_two_at_least(min: usize) -> usize {
    min.max(1).next_power_of_two()
}

fn is_smooth_supported(mut n: usize) -> bool {
    if n == 0 {
        return false;
    }
    for &radix in SUPPORTED_RADICES {
        while n % radix == 0 {
            n /= radix;
        }
    }
    n == 1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifies_smooth_prime_and_composite_lengths() {
        assert_eq!(axis_kind_for_len(8), AxisKind::Mixed);
        assert_eq!(axis_kind_for_len(21), AxisKind::Mixed);
        assert_eq!(axis_kind_for_len(17), AxisKind::Rader);
        assert_eq!(axis_kind_for_len(29), AxisKind::Rader);
        assert_eq!(axis_kind_for_len(34), AxisKind::Bluestein);
        assert_eq!(axis_kind_for_len(4099), AxisKind::Bluestein);
    }

    #[test]
    fn resolves_axis_kinds_for_shape_in_axis_order() {
        assert_eq!(
            resolve_axis_kinds_for_shape(&[8, 29, 34]).unwrap(),
            [AxisKind::Mixed, AxisKind::Rader, AxisKind::Bluestein]
        );
        assert_eq!(
            resolve_axis_kinds_for_axes(&[8, 29, 34], &[2, 0]).unwrap(),
            [AxisKind::Bluestein, AxisKind::Mixed]
        );
    }

    #[test]
    fn rejects_invalid_shapes_and_axes() {
        assert_eq!(resolve_axis_kinds_for_shape(&[]), Err(FftError::ZeroLength));
        assert_eq!(
            resolve_axis_kinds_for_shape(&[8, 0]),
            Err(FftError::ZeroLength)
        );
        assert_eq!(
            resolve_axis_kinds_for_axes(&[8], &[1]),
            Err(FftError::InvalidAxis { axis: 1, rank: 1 })
        );
    }

    #[test]
    fn primality_matches_policy_needs() {
        assert!(!is_prime(0));
        assert!(!is_prime(1));
        assert!(is_prime(2));
        assert!(is_prime(29));
        assert!(!is_prime(34));
    }

    #[test]
    fn primitive_root_generates_prime_field() {
        let root = primitive_root_prime(17).unwrap();
        let mut values = (0..16)
            .map(|exp| mod_pow(root, exp, 17))
            .collect::<Vec<_>>();
        values.sort_unstable();
        assert_eq!(values, (1..17).collect::<Vec<_>>());
    }

    #[test]
    fn smooth_length_helpers_match_supported_radices() {
        assert_eq!(next_smooth_at_least(31), 32);
        assert_eq!(next_smooth_at_least(34), 35);
        assert_eq!(next_power_of_two_at_least(33), 64);
    }
}
