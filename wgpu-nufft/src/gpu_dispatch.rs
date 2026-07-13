use crate::error::{NufftError, Result};

/// Splits a flat workgroup count into a u32-safe 3D dispatch grid.
///
/// This is deliberately local rather than reaching through wgpu-fft's
/// crate-private runtime module or promoting raw WGSL string fragments into
/// its public API. NUFFT needs its own structured error, and barrier-using
/// kernels require a workgroup-uniform padding guard instead of the FFT
/// helper's lane-level element guard. Reconsider promotion when another
/// non-FFT consumer needs a stable typed dispatch abstraction.
pub(crate) fn split_workgroups(workgroups: u32, max_per_dimension: u32) -> Result<(u32, u32, u32)> {
    if max_per_dimension == 0 {
        return Err(NufftError::GpuDispatchUnsupported {
            workgroups,
            max_per_dimension,
        });
    }
    if workgroups == 0 {
        return Ok((0, 1, 1));
    }

    let total = u64::from(workgroups);
    let maximum = u64::from(max_per_dimension);
    let per_slice_limit = maximum * maximum;
    let z = total.div_ceil(per_slice_limit);
    if z > maximum {
        return Err(NufftError::GpuDispatchUnsupported {
            workgroups,
            max_per_dimension,
        });
    }

    let workgroups_per_slice = total.div_ceil(z);
    let y = workgroups_per_slice.div_ceil(maximum);
    let x = workgroups_per_slice.div_ceil(y);
    let covered = x
        .checked_mul(y)
        .and_then(|xy| xy.checked_mul(z))
        .unwrap_or(u64::MAX);
    if x <= maximum && y <= maximum && covered <= u64::from(u32::MAX) + 1 {
        return Ok((x as u32, y as u32, z as u32));
    }

    // Exactly covers the full u32 index space while keeping flattened WGSL
    // workgroup arithmetic in range.
    if max_per_dimension >= 2048 {
        return Ok((2048, 2048, 1024));
    }

    Err(NufftError::GpuDispatchUnsupported {
        workgroups,
        max_per_dimension,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_grid_covers_scan_and_full_u32_boundaries() {
        for workgroups in [0, 1, 65_535, 65_536, 65_537, 1 << 19, u32::MAX] {
            let (x, y, z) = split_workgroups(workgroups, 65_535).unwrap();
            assert!(x <= 65_535 && y <= 65_535 && z <= 65_535);
            let covered = u64::from(x) * u64::from(y) * u64::from(z);
            assert!(covered >= u64::from(workgroups));
            assert!(covered <= u64::from(u32::MAX) + 1);
        }
        assert_eq!(
            split_workgroups(u32::MAX, 65_535).unwrap(),
            (2048, 2048, 1024)
        );
    }

    #[test]
    fn split_grid_rejects_zero_or_insufficient_dimension_limits() {
        assert!(matches!(
            split_workgroups(1, 0),
            Err(NufftError::GpuDispatchUnsupported { .. })
        ));
        assert!(matches!(
            split_workgroups(2, 1),
            Err(NufftError::GpuDispatchUnsupported { .. })
        ));
    }

    #[test]
    fn future_three_dimensional_scan_uses_split_dispatch() {
        let bins = 1u32 << 27;
        let block_count = bins.div_ceil(2048);
        assert_eq!(block_count, 65_536);
        let grid = split_workgroups(block_count, 65_535).unwrap();
        assert_ne!(grid, (block_count, 1, 1));
        assert!(grid.0 <= 65_535 && grid.1 <= 65_535 && grid.2 <= 65_535);
        assert!(
            u64::from(grid.0) * u64::from(grid.1) * u64::from(grid.2) >= u64::from(block_count)
        );
    }
}
