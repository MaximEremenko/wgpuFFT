struct Params {
    len: u32,
    inverse: u32,
    scale: vec2<f32>,
};

@group(0) @binding(0)
var<storage, read> input: array<vec4<f32>>;

@group(0) @binding(1)
var<storage, read_write> output: array<vec4<f32>>;

@group(0) @binding(2)
var<uniform> params: Params;

@group(0) @binding(3)
var<storage, read> twiddle_lut: array<vec4<f32>>;

@compute @workgroup_size(64)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let wgFlat = (wid.z * nwg.y + wid.y) * nwg.x + wid.x;
    if (wgFlat > params.len / 64u) {
        return;
    }
    let k = wgFlat * 64u + lid.x;
    if (k >= params.len) {
        return;
    }

    var sum = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    var twiddle_index = 0u;

    for (var n = 0u; n < params.len; n = n + 1u) {
        let forward_twiddle = twiddle_lut[twiddle_index];
        let twiddle = select(
            forward_twiddle,
            vec4<f32>(
                forward_twiddle.x,
                forward_twiddle.y,
                -forward_twiddle.z,
                -forward_twiddle.w,
            ),
            params.inverse != 0u,
        );
        sum = df64_complex_add(sum, df64_complex_mul(input[n], twiddle));
        if (twiddle_index >= params.len - k) {
            twiddle_index = twiddle_index - (params.len - k);
        } else {
            twiddle_index = twiddle_index + k;
        }
    }

    let scale = Df64(params.scale.x, params.scale.y);
    output[k] = df64_complex_scale(sum, scale);
}
