// Portable double-float arithmetic. A Df64 is the unevaluated sum hi + lo;
// complex values use vec4<f32>(re_hi, re_lo, im_hi, im_lo).
//
// WGSL permits reassociation and contraction and has no `precise` qualifier.
// Named functions, local variables, and a plain bitcast round trip were all
// optimized through on tested release backends. Every elementary operation is
// therefore consumed by the integer sign/magnitude reconstruction below. The
// exact-word GPU canaries make this empirical source barrier part of the
// backend support contract.

struct Df64 {
    hi: f32,
    lo: f32,
}

fn df64_round_finite(value: f32) -> f32 {
    let bits = bitcast<u32>(value);
    let magnitude = bitcast<f32>(bits & 0x7fffffffu);
    if ((bits & 0x80000000u) != 0u) {
        return -magnitude;
    }
    return magnitude;
}

fn df64_add_rounded(a: f32, b: f32) -> f32 {
    var result = a + b;
    return df64_round_finite(result);
}

fn df64_sub_rounded(a: f32, b: f32) -> f32 {
    var result = a - b;
    return df64_round_finite(result);
}

fn df64_mul_rounded(a: f32, b: f32) -> f32 {
    var result = a * b;
    return df64_round_finite(result);
}

fn df64_two_sum(a: f32, b: f32) -> Df64 {
    let sum = df64_add_rounded(a, b);
    let b_virtual = df64_sub_rounded(sum, a);
    let a_virtual = df64_sub_rounded(sum, b_virtual);
    let b_roundoff = df64_sub_rounded(b, b_virtual);
    let a_roundoff = df64_sub_rounded(a, a_virtual);
    return Df64(sum, df64_add_rounded(a_roundoff, b_roundoff));
}

fn df64_quick_two_sum(a: f32, b: f32) -> Df64 {
    let sum = df64_add_rounded(a, b);
    let recovered_b = df64_sub_rounded(sum, a);
    return Df64(sum, df64_sub_rounded(b, recovered_b));
}

fn df64_split_raw(a: f32) -> Df64 {
    let split = df64_mul_rounded(4097.0, a);
    let hi = df64_sub_rounded(split, df64_sub_rounded(split, a));
    return Df64(hi, df64_sub_rounded(a, hi));
}

fn df64_split(a: f32) -> Df64 {
    // Exact power-of-two scaling prevents 4097*a from overflowing for large
    // finite values and gives tiny normal values enough headroom to split.
    // WebGPU may flush subnormals, so preservation below the normal range is
    // intentionally not part of the portable contract.
    let upper: f32 = 1.2676506002282294e30;
    let lower: f32 = 7.888609052210118e-31;
    let scale_up: f32 = 268435456.0;
    let scale_down: f32 = 3.725290298461914e-9;
    let magnitude = abs(a);
    if (magnitude > upper) {
        let split = df64_split_raw(df64_mul_rounded(a, scale_down));
        return Df64(
            df64_mul_rounded(split.hi, scale_up),
            df64_mul_rounded(split.lo, scale_up),
        );
    }
    if (magnitude != 0.0 && magnitude < lower) {
        let split = df64_split_raw(df64_mul_rounded(a, scale_up));
        return Df64(
            df64_mul_rounded(split.hi, scale_down),
            df64_mul_rounded(split.lo, scale_down),
        );
    }
    return df64_split_raw(a);
}

fn df64_two_prod(a: f32, b: f32) -> Df64 {
    let product = df64_mul_rounded(a, b);

    // Dekker splitting for binary32: 2^12 + 1. Do not replace this with an
    // fma-dependent product-error formula.
    let a_parts = df64_split(a);
    let b_parts = df64_split(b);

    let hi_product = df64_mul_rounded(a_parts.hi, b_parts.hi);
    let hi_error = df64_sub_rounded(hi_product, product);
    let cross_1 = df64_mul_rounded(a_parts.hi, b_parts.lo);
    let cross_2 = df64_mul_rounded(a_parts.lo, b_parts.hi);
    let lo_product = df64_mul_rounded(a_parts.lo, b_parts.lo);
    let error_1 = df64_add_rounded(hi_error, cross_1);
    let error_2 = df64_add_rounded(error_1, cross_2);
    let error = df64_add_rounded(error_2, lo_product);
    return Df64(product, error);
}

fn df64_neg(a: Df64) -> Df64 {
    return Df64(-a.hi, -a.lo);
}

fn df64_add(a: Df64, b: Df64) -> Df64 {
    let hi_sum = df64_two_sum(a.hi, b.hi);
    let lo_sum = df64_two_sum(a.lo, b.lo);
    let combined_lo = df64_add_rounded(hi_sum.lo, lo_sum.hi);
    let normalized = df64_quick_two_sum(hi_sum.hi, combined_lo);
    let remaining_lo = df64_add_rounded(normalized.lo, lo_sum.lo);
    return df64_quick_two_sum(normalized.hi, remaining_lo);
}

fn df64_sub(a: Df64, b: Df64) -> Df64 {
    return df64_add(a, df64_neg(b));
}

fn df64_mul(a: Df64, b: Df64) -> Df64 {
    let hi_product = df64_two_prod(a.hi, b.hi);
    let cross_1 = df64_mul_rounded(a.hi, b.lo);
    let cross_2 = df64_mul_rounded(a.lo, b.hi);
    let lo_product = df64_mul_rounded(a.lo, b.lo);
    let error_1 = df64_add_rounded(hi_product.lo, cross_1);
    let error_2 = df64_add_rounded(error_1, cross_2);
    let error = df64_add_rounded(error_2, lo_product);
    return df64_quick_two_sum(hi_product.hi, error);
}

fn df64_complex_real(a: vec4<f32>) -> Df64 {
    return Df64(a.x, a.y);
}

fn df64_complex_imag(a: vec4<f32>) -> Df64 {
    return Df64(a.z, a.w);
}

fn df64_complex_pack(re: Df64, im: Df64) -> vec4<f32> {
    return vec4<f32>(re.hi, re.lo, im.hi, im.lo);
}

fn df64_complex_add(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
    return df64_complex_pack(
        df64_add(df64_complex_real(a), df64_complex_real(b)),
        df64_add(df64_complex_imag(a), df64_complex_imag(b)),
    );
}

fn df64_complex_sub(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
    return df64_complex_pack(
        df64_sub(df64_complex_real(a), df64_complex_real(b)),
        df64_sub(df64_complex_imag(a), df64_complex_imag(b)),
    );
}

fn df64_complex_mul(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
    let ar = df64_complex_real(a);
    let ai = df64_complex_imag(a);
    let br = df64_complex_real(b);
    let bi = df64_complex_imag(b);
    let real = df64_sub(df64_mul(ar, br), df64_mul(ai, bi));
    let imag = df64_add(df64_mul(ar, bi), df64_mul(ai, br));
    return df64_complex_pack(real, imag);
}
