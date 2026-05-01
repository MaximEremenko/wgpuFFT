pub(crate) fn wgsl_line_base_fn(rank: usize, axis: usize, dims: &[usize]) -> String {
    assert!(rank >= 1, "rank must be at least one");
    assert_eq!(rank, dims.len(), "dims length must match rank");
    assert!(axis < rank, "axis must be in range");

    let n_total = product(dims);
    let lines_per_batch = lines_per_batch(dims, axis);
    let strides = strides_for_shape(dims);

    let mut decode = String::new();
    let mut rem_name = String::from("rem");
    let mut rem_init = true;
    for dim_index in 0..rank {
        if dim_index == axis {
            continue;
        }

        if rem_init {
            decode.push_str("  var rem: u32 = line - b * lines_per_batch;\n");
            rem_init = false;
        }

        let coord_name = format!("c{dim_index}");
        let next_rem = format!("{rem_name}_{dim_index}");
        decode.push_str(&format!(
            "  let {coord_name}: u32 = {rem_name} % {dim}u;\n",
            dim = dims[dim_index]
        ));
        decode.push_str(&format!(
            "  base = base + {coord_name} * {stride}u;\n",
            stride = strides[dim_index]
        ));
        decode.push_str(&format!(
            "  var {next_rem}: u32 = {rem_name} / {dim}u;\n",
            dim = dims[dim_index]
        ));
        rem_name = next_rem;
    }

    if decode.is_empty() {
        decode.push_str("  // axis-only line (rank=1): no non-axis coordinates\n");
    }

    format!(
        r#"fn line_base(line: u32) -> u32 {{
  let lines_per_batch: u32 = {lines_per_batch}u;
  let b: u32 = line / lines_per_batch;
  var base: u32 = b * {n_total}u;
{decode}  return base;
}}"#,
        lines_per_batch = lines_per_batch,
        n_total = n_total,
        decode = decode,
    )
}

pub(crate) fn stride_for_axis(dims: &[usize], axis: usize) -> usize {
    dims.iter().take(axis).product()
}

pub(crate) fn lines_per_batch(dims: &[usize], axis: usize) -> usize {
    product(dims) / dims[axis]
}

pub(crate) fn product(values: &[usize]) -> usize {
    values.iter().product()
}

pub(crate) fn strides_for_shape(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; shape.len()];
    for index in 1..shape.len() {
        strides[index] = strides[index - 1] * shape[index - 1];
    }
    strides
}

pub(crate) fn format_wgsl_f32(value: f32) -> String {
    assert!(value.is_finite(), "WGSL f32 constants must be finite");

    let mut formatted = format!("{value:.9}");
    while formatted.contains('.') && formatted.ends_with('0') {
        formatted.pop();
    }
    if formatted.ends_with('.') {
        formatted.push('0');
    }
    if !formatted.contains('.') {
        formatted.push_str(".0");
    }
    formatted
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn line_base_for_nd_axis_contains_expected_constants() {
        let wgsl = wgsl_line_base_fn(3, 1, &[4, 17, 2]);
        assert!(wgsl.contains("let lines_per_batch: u32 = 8u;"));
        assert!(wgsl.contains("var base: u32 = b * 136u;"));
        assert!(wgsl.contains("base = base + c0 * 1u;"));
        assert!(wgsl.contains("base = base + c2 * 68u;"));
    }

    #[test]
    fn shape_helpers_match_interleaved_complex_layout() {
        assert_eq!(stride_for_axis(&[4, 17, 2], 0), 1);
        assert_eq!(stride_for_axis(&[4, 17, 2], 1), 4);
        assert_eq!(stride_for_axis(&[4, 17, 2], 2), 68);
        assert_eq!(lines_per_batch(&[4, 17, 2], 1), 8);
        assert_eq!(strides_for_shape(&[4, 17, 2]), [1, 4, 68]);
    }
}
