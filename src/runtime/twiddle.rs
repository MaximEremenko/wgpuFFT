use crate::config::FftPrecision;
use crate::error::{FftError, Result};
use crate::math::{Complex32, Complex64};

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct TwoLevelTwiddleLutF32 {
    pub(crate) coarse: Vec<Complex32>,
    pub(crate) fine: Vec<Complex32>,
    pub(crate) shift: u32,
    pub(crate) mask: u32,
}

/// Builds the canonical forward roots `exp(-2*pi*i*k/len)` using f64
/// trigonometry and rounds each component exactly once when storing it as f32.
pub(crate) fn twiddle_lut_f32(len: usize) -> Vec<Complex32> {
    assert!(len > 0, "a twiddle table requires a non-zero length");
    (0..len)
        .map(|index| canonical_twiddle_f32(len, index))
        .collect()
}

/// Builds the canonical forward roots `exp(-2*pi*i*k/len)` directly in f64.
pub(crate) fn twiddle_lut_f64(len: usize) -> Vec<Complex64> {
    assert!(len > 0, "a twiddle table requires a non-zero length");
    (0..len)
        .map(|index| canonical_twiddle_f64(len, index))
        .collect()
}

/// Builds two compact tables such that, for every `k < len`,
///
/// `W_len^k = coarse[k >> split_bits] * fine[k & fine_mask]`.
///
/// The fine-table length is a power of two near `sqrt(len)`, so the shader can
/// split an index with one shift and one mask. Both tables are generated with
/// f64 trigonometry before being rounded to f32.
pub(crate) fn two_level_twiddle_lut_f32(len: usize) -> TwoLevelTwiddleLutF32 {
    assert!(len > 0, "a twiddle table requires a non-zero length");
    assert!(
        len <= u32::MAX as usize,
        "two-level WGSL twiddle indices must fit in u32"
    );

    let index_bits = usize::BITS - (len - 1).leading_zeros();
    let shift = index_bits.div_ceil(2);
    let fine_len = 1usize << shift;
    let coarse_len = len.div_ceil(fine_len);
    let fine_mask = fine_len - 1;

    let coarse = (0..coarse_len)
        .map(|index| canonical_twiddle_f32(len, index * fine_len))
        .collect();
    let fine = (0..fine_len)
        .map(|index| canonical_twiddle_f32(len, index))
        .collect();

    TwoLevelTwiddleLutF32 {
        coarse,
        fine,
        shift,
        mask: fine_mask as u32,
    }
}

pub(crate) fn create_twiddle_lut_buffer(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &'static str,
    values: &[Complex32],
) -> Result<wgpu::Buffer> {
    let requested_bytes = validate_twiddle_lut_len(device, label, values.len())?;

    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: requested_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&buffer, 0, bytemuck::cast_slice(values));
    Ok(buffer)
}

pub(crate) fn create_twiddle_lut_buffer_for_len_with_precision(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &'static str,
    len: usize,
    precision: FftPrecision,
) -> Result<wgpu::Buffer> {
    validate_twiddle_lut_len_with_precision(device, label, len, precision)?;
    if precision == FftPrecision::F64 {
        let values = twiddle_lut_f64(len);
        return create_twiddle_lut_buffer_f64(device, queue, label, &values);
    }
    let values = twiddle_lut_f32(len);
    create_twiddle_lut_buffer(device, queue, label, &values)
}

fn create_twiddle_lut_buffer_f64(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &'static str,
    values: &[Complex64],
) -> Result<wgpu::Buffer> {
    let requested_bytes =
        validate_twiddle_lut_len_with_precision(device, label, values.len(), FftPrecision::F64)?;
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: requested_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&buffer, 0, bytemuck::cast_slice(values));
    Ok(buffer)
}

fn validate_twiddle_lut_len(device: &wgpu::Device, label: &'static str, len: usize) -> Result<u64> {
    validate_twiddle_lut_len_with_precision(device, label, len, FftPrecision::F32)
}

fn validate_twiddle_lut_len_with_precision(
    device: &wgpu::Device,
    label: &'static str,
    len: usize,
    precision: FftPrecision,
) -> Result<u64> {
    if len == 0 {
        return Err(FftError::ZeroLength);
    }
    let requested_bytes = (len as u64)
        .checked_mul(precision.complex_size_bytes())
        .ok_or(FftError::LengthTooLarge { len })?;
    let limits = device.limits();
    if requested_bytes > limits.max_buffer_size {
        return Err(FftError::HelperBufferTooLarge {
            helper_buffer: label,
            requested_bytes,
            max_buffer_size: limits.max_buffer_size,
        });
    }
    if requested_bytes > limits.max_storage_buffer_binding_size {
        return Err(FftError::WindowScheduleUnsupported {
            reason: "twiddle LUT exceeds max storage buffer binding size",
            requested_bytes,
            max_bind_bytes: limits.max_storage_buffer_binding_size,
        });
    }
    Ok(requested_bytes)
}

/// Returns the exact N-entry LUT index for the Stockham factor
/// `exp(-2*pi*i*r*q/ns)`, where `ns` is a divisor of `len`.
#[cfg(test)]
pub(crate) fn stockham_twiddle_index(len: usize, ns: usize, r: usize, q: usize) -> usize {
    assert!(len > 0);
    assert!(ns > 0 && len % ns == 0);
    assert!(r < ns);

    let phase = ((r as u128 * q as u128) % ns as u128) as usize;
    phase * (len / ns)
}

fn canonical_twiddle_f32(len: usize, index: usize) -> Complex32 {
    debug_assert!(len > 0);
    debug_assert!(index < len);
    let angle = -std::f64::consts::TAU * index as f64 / len as f64;
    let (sin, cos) = angle.sin_cos();
    Complex32::new(cos as f32, sin as f32)
}

fn canonical_twiddle_f64(len: usize, index: usize) -> Complex64 {
    debug_assert!(len > 0);
    debug_assert!(index < len);
    let angle = -std::f64::consts::TAU * index as f64 / len as f64;
    let (sin, cos) = angle.sin_cos();
    Complex64::new(cos, sin)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn c_mul(a: Complex32, b: Complex32) -> Complex32 {
        Complex32::new(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re)
    }

    #[test]
    fn full_lut_is_computed_with_f64_and_rounded_once() {
        let len = 4096;
        let lut = twiddle_lut_f32(len);
        for index in [0, 1, 17, 1023, 2048, 4095] {
            let angle = -std::f64::consts::TAU * index as f64 / len as f64;
            let (sin, cos) = angle.sin_cos();
            assert_eq!(lut[index], Complex32::new(cos as f32, sin as f32));
        }
    }

    #[test]
    fn f64_lut_preserves_host_sin_cos_bits() {
        let len = 4096;
        let lut = twiddle_lut_f64(len);
        for index in [0, 1, 17, 1023, 2048, 4095] {
            let angle = -std::f64::consts::TAU * index as f64 / len as f64;
            let (sin, cos) = angle.sin_cos();
            assert_eq!(lut[index].re.to_bits(), cos.to_bits());
            assert_eq!(lut[index].im.to_bits(), sin.to_bits());
        }
    }

    #[test]
    fn stockham_indices_are_exact_for_every_supported_stage() {
        for len in [24, 143, 315, 1001, 2187, 3000, 4096] {
            let factors = crate::runtime::factor_supported_length(len).unwrap();
            let mut ns = 1usize;
            for radix in factors {
                ns *= radix;
                for r in 0..ns {
                    for q in 0..radix {
                        let index = stockham_twiddle_index(len, ns, r, q);
                        assert!(index < len);
                        assert_eq!(index / (len / ns), (r * q) % ns);
                    }
                }
            }
        }
    }

    #[test]
    fn two_level_tables_reconstruct_full_lut_with_one_complex_multiply() {
        for len in [1, 2, 3, 17, 1001, 3000, 4096, 1_000_003] {
            let two_level = two_level_twiddle_lut_f32(len);
            let fine_len = (two_level.mask + 1) as usize;
            assert!(fine_len.is_power_of_two());
            assert_eq!(fine_len, 1usize << two_level.shift);
            assert_eq!(two_level.coarse.len(), len.div_ceil(fine_len));
            assert!(two_level.coarse.len() + two_level.fine.len() <= 2 * fine_len);

            let probes = [
                0,
                len / 7,
                len / 3,
                len / 2,
                (fine_len - 1).min(len - 1),
                fine_len.min(len - 1),
                (fine_len + 1).min(len - 1),
                (two_level.coarse.len() - 1) * fine_len,
                len - 1,
            ];
            for index in probes {
                let reconstructed = c_mul(
                    two_level.coarse[index >> two_level.shift],
                    two_level.fine[index & two_level.mask as usize],
                );
                let expected = canonical_twiddle_f32(len, index);
                assert!((reconstructed.re - expected.re).abs() <= 2.5e-7);
                assert!((reconstructed.im - expected.im).abs() <= 2.5e-7);
            }
        }

        for len in 1..=256 {
            let two_level = two_level_twiddle_lut_f32(len);
            for index in 0..len {
                let reconstructed = c_mul(
                    two_level.coarse[index >> two_level.shift],
                    two_level.fine[index & two_level.mask as usize],
                );
                let expected = canonical_twiddle_f32(len, index);
                assert!((reconstructed.re - expected.re).abs() <= 2.5e-7);
                assert!((reconstructed.im - expected.im).abs() <= 2.5e-7);
            }
        }
    }
}
