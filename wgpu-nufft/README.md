# wgpu-nufft

`wgpu-nufft` is a portable nonuniform FFT library built only on `wgpu-fft`'s
public API. Planning is separate from execution and plans are reusable.

## Current scope

The current implementation provides:

- FINUFFT-compatible type-1 and type-2 definitions in one to three dimensions;
- direct `f64` NDFT execution as the correctness oracle and tiny-problem fallback;
- `sigma = 2` default exponential-of-semicircle kernel planning;
- host-`f64` centered-grid kernel Fourier coefficients;
- even fine-grid selection through `wgpu-fft`'s public supported-length
  factorization;
- reusable 1D type-2 `f32` GPU plans with caller-owned point, Fourier
  coefficient, and output buffers; and
- reusable 1D type-1 `f32` GPU plans with caller-owned point, strength, and
  Fourier-mode output buffers.

The GPU type-2 route pre-deconvolves and zero-pads on the GPU, executes a public
`wgpu-fft` C2C plan on the oversampled grid, and gathers the ES interpolation
kernel at each nonuniform point. The caller records execution into its own
command encoder and can consume the caller-owned output buffer without a
readback. Plan-owned fine-grid buffers are reused between ordered executions.
Points are scalar `f32` values, while coefficients and outputs are interleaved
complex `(re, im)` `f32` pairs; all require `STORAGE` buffer usage.
The Vulkan accuracy matrix enforces relative L2 error at most `4*eps` over the
complete random-plus-adversarial vector and `8*eps` for each isolated boundary,
cluster, duplicate, and seeded-random subset. The latter remains stricter than
FINUFFT's pinned float regression factors while preserving its ES parameters.

The GPU type-1 route bins point indices with portable `u32` atomics, restores a
stable input order within each bin, and assigns each fine-grid cell to exactly
one gathering invocation. It therefore needs no unavailable WGSL `f32`
atomics, and repeated executions accumulate strengths in deterministic point
order. The gathered grid passes through a public `wgpu-fft` C2C plan before
deconvolution and truncation into the caller-owned mode buffer. Both GPU routes
reuse plan-owned fine-grid scratch and must execute in queue order on a given
plan.

Deferred work includes 2D/3D GPU execution, type 3, batching, NUFFT `f64`/df64
GPU arithmetic, parallel bin scans and sorting, and WASM packaging.

## Mathematical conventions

For integer mode vectors `k`, nonuniform points `x_j`, and complex strengths
`c_j`, type 1 computes

```text
f_k = sum_j c_j exp(isign * i * dot(k, x_j))
```

and type 2 computes

```text
c_j = sum_k f_k exp(isign * i * dot(k, x_j)).
```

There is no normalization. Nonnegative `isign` selects the positive sign;
negative values select the negative sign. Centered mode order is the default,
and dimension zero is stored fastest. Coordinates follow FINUFFT's documented
type-1/type-2 input contract `|x| <= 3*pi` and are periodic modulo `2*pi`.
With the conventional Hermitian inner product, type-2 with sign `s` is adjoint
to type-1 with sign `-s`.
