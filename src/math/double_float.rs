use bytemuck::{Pod, Zeroable};

/// A portable double-float value represented as an unevaluated `hi + lo` sum.
///
/// This retains roughly 44-48 significant bits, but its exponent range remains
/// that of `f32`. It intentionally mirrors the pure-`f32` WGSL implementation
/// in `kernels/df64.wgsl`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Pod, Zeroable)]
pub struct DoubleFloat {
    pub hi: f32,
    pub lo: f32,
}

impl DoubleFloat {
    pub const fn new(hi: f32, lo: f32) -> Self {
        Self { hi, lo }
    }

    pub fn from_f64(value: f64) -> Self {
        let hi = value as f32;
        if !hi.is_finite() {
            return Self { hi, lo: 0.0 };
        }
        let lo = (value - f64::from(hi)) as f32;
        Self { hi, lo }
    }

    pub fn to_f64(self) -> f64 {
        f64::from(self.hi) + f64::from(self.lo)
    }

    pub fn add_df(self, rhs: Self) -> Self {
        let (s1, mut s2) = two_sum_f32(self.hi, rhs.hi);
        let (t1, t2) = two_sum_f32(self.lo, rhs.lo);
        s2 += t1;
        let (s1, mut s2) = quick_two_sum_f32(s1, s2);
        s2 += t2;
        let (hi, lo) = quick_two_sum_f32(s1, s2);
        Self { hi, lo }
    }

    pub fn sub_df(self, rhs: Self) -> Self {
        self.add_df(Self::new(-rhs.hi, -rhs.lo))
    }

    pub fn mul_df(self, rhs: Self) -> Self {
        let (p1, mut p2) = two_prod_f32(self.hi, rhs.hi);
        p2 += self.hi * rhs.lo;
        p2 += self.lo * rhs.hi;
        p2 += self.lo * rhs.lo;
        let (hi, lo) = quick_two_sum_f32(p1, p2);
        Self { hi, lo }
    }
}

/// Complex double-float in GPU storage order: real hi/lo, then imaginary hi/lo.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Pod, Zeroable)]
pub struct ComplexDoubleFloat {
    pub re_hi: f32,
    pub re_lo: f32,
    pub im_hi: f32,
    pub im_lo: f32,
}

impl ComplexDoubleFloat {
    pub const fn new(re: DoubleFloat, im: DoubleFloat) -> Self {
        Self {
            re_hi: re.hi,
            re_lo: re.lo,
            im_hi: im.hi,
            im_lo: im.lo,
        }
    }

    pub fn from_f64(re: f64, im: f64) -> Self {
        Self::new(DoubleFloat::from_f64(re), DoubleFloat::from_f64(im))
    }

    pub const fn re(self) -> DoubleFloat {
        DoubleFloat::new(self.re_hi, self.re_lo)
    }

    pub const fn im(self) -> DoubleFloat {
        DoubleFloat::new(self.im_hi, self.im_lo)
    }

    pub fn add_df(self, rhs: Self) -> Self {
        Self::new(self.re().add_df(rhs.re()), self.im().add_df(rhs.im()))
    }

    pub fn sub_df(self, rhs: Self) -> Self {
        Self::new(self.re().sub_df(rhs.re()), self.im().sub_df(rhs.im()))
    }

    pub fn mul_df(self, rhs: Self) -> Self {
        let re = self
            .re()
            .mul_df(rhs.re())
            .sub_df(self.im().mul_df(rhs.im()));
        let im = self
            .re()
            .mul_df(rhs.im())
            .add_df(self.im().mul_df(rhs.re()));
        Self::new(re, im)
    }
}

/// Error-free sum of two `f32` values, assuming individual operations round to
/// IEEE-754 `f32` as Rust requires.
pub fn two_sum_f32(a: f32, b: f32) -> (f32, f32) {
    let sum = a + b;
    let b_virtual = sum - a;
    let a_virtual = sum - b_virtual;
    let b_roundoff = b - b_virtual;
    let a_roundoff = a - a_virtual;
    (sum, a_roundoff + b_roundoff)
}

/// Error-free sum when `a` is at least as large in magnitude as `b`.
pub fn quick_two_sum_f32(a: f32, b: f32) -> (f32, f32) {
    let sum = a + b;
    (sum, b - (sum - a))
}

/// Split-based Dekker product. This deliberately does not use `mul_add`: the
/// corresponding WGSL must work on backends where contraction cannot be
/// disabled.
pub fn two_prod_f32(a: f32, b: f32) -> (f32, f32) {
    let product = a * b;
    let (a_hi, a_lo) = split_f32(a);
    let (b_hi, b_lo) = split_f32(b);
    let error = (((a_hi * b_hi - product) + a_hi * b_lo) + a_lo * b_hi) + a_lo * b_lo;
    (product, error)
}

/// Exact mantissa-bit form of Dekker's binary32 split.
///
/// The high word retains 12 significant bits and the low word is the exact
/// residual. Clearing fraction bits avoids the splitter-multiply overflow at
/// finite values adjacent to `f32::MAX`. Subnormal preservation on the GPU
/// remains backend-dependent because WebGPU backends may flush subnormals.
pub fn split_f32(value: f32) -> (f32, f32) {
    let hi = f32::from_bits(value.to_bits() & 0xffff_f000);
    (hi, value - hi)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_sum_recovers_exact_f32_sum() {
        let cases = [
            (1.0f32, f32::from_bits(0x3380_0000)),
            (1.0e10, -3.0),
            (1.000_000_1, -1.0),
            (-17.25, f32::from_bits(0x3580_0000)),
        ];
        for (a, b) in cases {
            let (hi, lo) = two_sum_f32(a, b);
            assert_eq!(f64::from(hi) + f64::from(lo), f64::from(a) + f64::from(b));
        }
    }

    #[test]
    fn split_two_prod_recovers_exact_f32_product() {
        let cases = [
            (1.000_000_1f32, 0.999_999_9f32),
            (12_345.125, -0.031_257_63),
            (-17.25, 3.000_000_2),
            (f32::from_bits(0x3f00_0001), f32::from_bits(0x3f7f_fffd)),
        ];
        for (a, b) in cases {
            let (hi, lo) = two_prod_f32(a, b);
            assert_eq!(f64::from(hi) + f64::from(lo), f64::from(a) * f64::from(b));
            assert_ne!(lo, 0.0, "case must exercise the product error word");
        }
    }

    #[test]
    fn mantissa_bit_split_is_exact_at_both_finite_extremes() {
        for value in [f32::MAX, -f32::MAX] {
            let (hi, lo) = split_f32(value);
            assert!(hi.is_finite() && lo.is_finite());
            assert_eq!(f64::from(hi) + f64::from(lo), f64::from(value));
            assert_eq!(hi.to_bits() & 0x0000_0fff, 0);
        }
    }

    #[test]
    fn two_prod_handles_max_times_min_normal_without_splitter_overflow() {
        let (hi, lo) = two_prod_f32(f32::MAX, f32::MIN_POSITIVE);
        assert!(hi.is_finite() && lo.is_finite());
        assert_eq!(
            f64::from(hi) + f64::from(lo),
            f64::from(f32::MAX) * f64::from(f32::MIN_POSITIVE)
        );
    }

    #[test]
    fn double_float_ops_improve_on_single_precision() {
        let a = DoubleFloat::from_f64(1.0 + 2.0f64.powi(-40));
        let b = DoubleFloat::from_f64(-1.0 + 2.0f64.powi(-41));
        let sum = a.add_df(b);
        let expected_sum = 3.0 * 2.0f64.powi(-41);
        assert!((sum.to_f64() - expected_sum).abs() <= 2.0f64.powi(-64));
        assert_eq!(a.hi + b.hi, 0.0, "the f32-only low-order result is lost");

        let x = DoubleFloat::from_f64(1.0 + 2.0f64.powi(-30));
        let y = DoubleFloat::from_f64(1.0 - 2.0f64.powi(-31));
        let product = x.mul_df(y).to_f64();
        let expected_product = x.to_f64() * y.to_f64();
        assert!((product - expected_product).abs() <= 2.0f64.powi(-65));
    }

    #[test]
    fn conversion_does_not_manufacture_nan_residuals_for_nonfinite_high_words() {
        for value in [f64::INFINITY, f64::NEG_INFINITY, f64::MAX, -f64::MAX] {
            let split = DoubleFloat::from_f64(value);
            assert!(split.hi.is_infinite());
            assert_eq!(split.lo, 0.0);
        }
        let split = DoubleFloat::from_f64(f64::NAN);
        assert!(split.hi.is_nan());
        assert_eq!(split.lo, 0.0);
    }

    #[test]
    fn complex_double_float_uses_vec4_storage_order_and_arithmetic() {
        assert_eq!(std::mem::size_of::<ComplexDoubleFloat>(), 16);
        let a = ComplexDoubleFloat::from_f64(1.25 + 2.0f64.powi(-38), -0.75);
        let b = ComplexDoubleFloat::from_f64(-0.5, 0.125 - 2.0f64.powi(-39));
        let product = a.mul_df(b);
        let expected_re = a.re().to_f64() * b.re().to_f64() - a.im().to_f64() * b.im().to_f64();
        let expected_im = a.re().to_f64() * b.im().to_f64() + a.im().to_f64() * b.re().to_f64();
        assert!((product.re().to_f64() - expected_re).abs() < 1.0e-13);
        assert!((product.im().to_f64() - expected_im).abs() < 1.0e-13);
    }
}
