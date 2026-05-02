use crate::error::{FftError, Result};

/// Splits a flat 1D workgroup count into a 3D dispatch grid where every
/// dimension respects the device `max_compute_workgroups_per_dimension`
/// limit.
///
/// Kernels linearize the grid back to a flat workgroup index with
/// `(wid.z * nwg.y + wid.y) * nwg.x + wid.x`, so the grid may cover more
/// workgroups than requested; shader-side bound checks skip the padding.
pub(crate) fn split_workgroups(count: u32, max_per_dimension: u32) -> Result<(u32, u32, u32)> {
    if max_per_dimension == 0 {
        return Err(FftError::DispatchWorkgroupsUnsupported {
            workgroups: count,
            max_per_dimension,
        });
    }
    if count == 0 {
        return Ok((0, 1, 1));
    }
    let count = u64::from(count);
    let max = u64::from(max_per_dimension);
    let max_per_slice = max * max;
    let z = count.div_ceil(max_per_slice);
    if z > max {
        return Err(FftError::DispatchWorkgroupsUnsupported {
            workgroups: count as u32,
            max_per_dimension,
        });
    }

    // Balance each used slice so counts just above a dimension boundary do
    // not dispatch almost twice the requested workgroups as padding.
    let workgroups_per_slice = count.div_ceil(z);
    let y = workgroups_per_slice.div_ceil(max);
    let x = workgroups_per_slice.div_ceil(y);
    let covered = x
        .checked_mul(y)
        .and_then(|xy| xy.checked_mul(z))
        .unwrap_or(u64::MAX);
    let u32_index_space = u64::from(u32::MAX) + 1;
    if x <= max && y <= max && covered <= u32_index_space {
        return Ok((x as u32, y as u32, z as u32));
    }

    // A full u32 index space has 2^32 entries and factors into a balanced
    // grid whose largest dimension is 2^11. This recovers safe near-u32::MAX
    // counts when the first balanced candidate has slightly too much padding.
    if max >= 2048 {
        return Ok((2048, 2048, 1024));
    }

    Err(FftError::DispatchWorkgroupsUnsupported {
        workgroups: count as u32,
        max_per_dimension,
    })
}

/// Reads the active per-dimension compute dispatch limit for a device.
pub(crate) fn max_workgroups_per_dimension(device: &wgpu::Device) -> u32 {
    device.limits().max_compute_workgroups_per_dimension
}

/// Standard WGSL compute-entry builtin parameters for kernels that linearize
/// the 3D dispatch grid produced by [`split_workgroups`].
pub(crate) const WGSL_FLAT_ENTRY_PARAMS: &str = "@builtin(local_invocation_id) lid: vec3<u32>, \
     @builtin(workgroup_id) wid: vec3<u32>, \
     @builtin(num_workgroups) nwg: vec3<u32>";

/// WGSL statement computing the flat workgroup index for the 3D grid
/// produced by [`split_workgroups`].
pub(crate) const WGSL_FLAT_WORKGROUP_INDEX: &str =
    "let wgFlat: u32 = (wid.z * nwg.y + wid.y) * nwg.x + wid.x;";

/// WGSL statements deriving a flat element index `{index_name}` from the 3D
/// dispatch grid produced by [`split_workgroups`].
///
/// The guards skip grid-padding workgroups and inactive lanes before either
/// index multiply/add can wrap u32. `total_expr` must be a u32 expression.
pub(crate) fn wgsl_flat_index_stmts(
    index_name: &str,
    total_expr: &str,
    workgroup_size: u32,
) -> String {
    format!(
        "let flatTotal: u32 = ({total_expr});\n  \
         if (flatTotal == 0u) {{\n    return;\n  }}\n  \
         {WGSL_FLAT_WORKGROUP_INDEX}\n  \
         let wgLast: u32 = (flatTotal - 1u) / {workgroup_size}u;\n  \
         if (wgFlat > wgLast) {{\n    return;\n  }}\n  \
         let wgBase: u32 = wgFlat * {workgroup_size}u;\n  \
         if (lid.x >= flatTotal - wgBase) {{\n    return;\n  }}\n  \
         let {index_name}: u32 = wgBase + lid.x;"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_count_dispatches_nothing() {
        assert_eq!(split_workgroups(0, 65535), Ok((0, 1, 1)));
    }

    #[test]
    fn small_counts_stay_one_dimensional() {
        assert_eq!(split_workgroups(1, 65535), Ok((1, 1, 1)));
        assert_eq!(split_workgroups(65535, 65535), Ok((65535, 1, 1)));
    }

    #[test]
    fn counts_above_the_limit_fold_into_y() {
        assert_eq!(split_workgroups(65536, 65535), Ok((32768, 2, 1)));
        assert_eq!(split_workgroups(65537, 65535), Ok((32769, 2, 1)));
        let (x, y, z) = split_workgroups(u32::MAX / 64, 65535).unwrap();
        assert!(y <= 65535 && z <= 65535);
        assert!(u64::from(x) * u64::from(y) * u64::from(z) >= u64::from(u32::MAX / 64));
        assert!(u64::from(x) * u64::from(y) * u64::from(z) <= u64::from(u32::MAX) + 1);
    }

    #[test]
    fn grid_never_exceeds_the_per_dimension_limit() {
        for count in [1u32, 63, 64, 65535, 65536, 1 << 22, u32::MAX / 64 + 1] {
            for max in [1u32, 2, 7, 256, 65535] {
                match split_workgroups(count, max) {
                    Ok((x, y, z)) => {
                        assert!(x <= max && y <= max && z <= max, "count {count} max {max}");
                        assert!(
                            u64::from(x) * u64::from(y) * u64::from(z) >= u64::from(count),
                            "count {count} max {max} not covered"
                        );
                    }
                    Err(FftError::DispatchWorkgroupsUnsupported { .. }) => {
                        assert!(
                            u64::from(max).pow(3) < u64::from(count),
                            "count {count} max {max} should have fit"
                        );
                    }
                    Err(other) => panic!("unexpected error: {other:?}"),
                }
            }
        }
    }

    #[test]
    fn impossible_grids_return_structured_errors() {
        assert_eq!(
            split_workgroups(2, 1),
            Err(FftError::DispatchWorkgroupsUnsupported {
                workgroups: 2,
                max_per_dimension: 1,
            })
        );
    }

    #[test]
    fn zero_dimension_limit_returns_a_structured_error() {
        assert_eq!(
            split_workgroups(1, 0),
            Err(FftError::DispatchWorkgroupsUnsupported {
                workgroups: 1,
                max_per_dimension: 0,
            })
        );
    }

    #[test]
    fn full_u32_index_space_uses_an_exact_safe_grid() {
        assert_eq!(split_workgroups(u32::MAX, 65535), Ok((2048, 2048, 1024)));
    }

    #[test]
    fn flat_index_guard_checks_the_last_group_before_adding_the_lane() {
        let wgsl = wgsl_flat_index_stmts("i", "params.total", 3);
        assert!(wgsl.contains("let wgLast: u32 = (flatTotal - 1u) / 3u;"));
        assert!(wgsl.contains("if (lid.x >= flatTotal - wgBase)"));
        assert!(wgsl.contains("let i: u32 = wgBase + lid.x;"));
    }
}
