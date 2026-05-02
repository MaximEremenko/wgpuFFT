const TAU: f32 = 6.28318530717958647692;

struct Params {
    len: u32,
    inverse: u32,
    scale: f32,
    _pad: u32,
};

@group(0) @binding(0)
var<storage, read> input: array<vec2<f32>>;

@group(0) @binding(1)
var<storage, read_write> output: array<vec2<f32>>;

@group(0) @binding(2)
var<uniform> params: Params;

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

    let sign = select(-1.0, 1.0, params.inverse != 0u);
    var sum = vec2<f32>(0.0, 0.0);

    for (var n = 0u; n < params.len; n = n + 1u) {
        let angle = sign * TAU * f32(k) * f32(n) / f32(params.len);
        let twiddle = vec2<f32>(cos(angle), sin(angle));
        let x = input[n];
        sum = sum + vec2<f32>(
            x.x * twiddle.x - x.y * twiddle.y,
            x.x * twiddle.y + x.y * twiddle.x,
        );
    }

    output[k] = sum * params.scale;
}
