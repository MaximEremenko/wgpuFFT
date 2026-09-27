use crate::error::{FftError, Result};

pub(crate) mod axis_plan;
pub mod axis_policy;
pub(crate) mod bluestein_axis;
pub mod buffer_view;
pub mod c2c;
pub(crate) mod direct_prime;
pub(crate) mod dispatch;
pub(crate) mod four_step;
pub(crate) mod large_bridge;
pub(crate) mod large_chunk;
pub(crate) mod large_graph;
pub mod large_policy;
pub mod logical_io;
pub(crate) mod nd_wgsl;
pub mod pipeline_cache;
pub(crate) mod rader_axis;
pub mod real;
pub(crate) mod recorder;
pub(crate) mod register_fft;
pub(crate) mod segmented_volume;
pub(crate) mod small_volume;
pub(crate) mod smooth_decompose;
pub(crate) mod stage_executor;
pub(crate) mod twiddle;
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
            .find(|&&candidate| remaining.is_multiple_of(candidate))
        else {
            return Err(FftError::UnsupportedLength { len });
        };

        factors.push(radix);
        remaining /= radix;
    }

    Ok(factors)
}

/// Asserts that the first access to workgroup variable `name` inside the
/// entry point is a store.
///
/// Pipelines skip wgpu's workgroup zero fill, so a generated kernel must
/// never read workgroup memory that it has not written.
#[cfg(test)]
pub(crate) fn assert_workgroup_var_written_before_read(wgsl: &str, name: &str) {
    let entry = &wgsl[wgsl.find("fn main(").expect("kernel has an entry point")..];
    let is_ident = |c: Option<char>| c.is_some_and(|c| c.is_alphanumeric() || c == '_');
    let mut search = 0;
    let end = loop {
        let index = search
            + entry[search..]
                .find(name)
                .unwrap_or_else(|| panic!("workgroup variable `{name}` is never accessed"));
        search = index + name.len();
        if !is_ident(entry[..index].chars().next_back())
            && !is_ident(entry[search..].chars().next())
        {
            break search;
        }
    };
    let mut rest = &entry[end..];
    if rest.starts_with('[') {
        let mut depth = 0;
        let close = rest
            .char_indices()
            .find_map(|(i, c)| {
                match c {
                    '[' => depth += 1,
                    ']' => {
                        depth -= 1;
                        if depth == 0 {
                            return Some(i);
                        }
                    }
                    _ => {}
                }
                None
            })
            .expect("index has a closing bracket");
        rest = &rest[close + 1..];
    }
    let rest = rest.trim_start();
    assert!(
        rest.starts_with('=') && !rest.starts_with("=="),
        "first access to workgroup variable `{name}` must be a store"
    );
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

    #[test]
    fn workgroup_store_check_accepts_nested_indices_and_scalars() {
        let wgsl = "fn helper() { let v = scratch[0]; }\n\
                    fn main() { scratchpad = 1; scratch[perm[t]] = v; x0 = scratch[1]; }";
        assert_workgroup_var_written_before_read(wgsl, "scratch");
        assert_workgroup_var_written_before_read("fn main() { x0 = input[0]; }", "x0");
    }

    #[test]
    #[should_panic(expected = "must be a store")]
    fn workgroup_store_check_rejects_read_first() {
        assert_workgroup_var_written_before_read(
            "fn main() { let v = scratch[0]; scratch[1] = v; }",
            "scratch",
        );
    }
}
