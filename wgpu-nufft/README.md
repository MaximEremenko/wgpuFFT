# wgpu-nufft

`wgpu-nufft` is a portable nonuniform FFT library built only on `wgpu-fft`'s
public API. Planning is separate from execution and plans are reusable.

## Current scope

Phase A provides CPU-only foundations:

- Type-1 and type-2 definitions in one to three dimensions;
- direct `f64` NDFT execution as the correctness oracle and tiny-problem fallback;
- `sigma = 2` default exponential-of-semicircle kernel planning;
- host-`f64` centered-grid kernel Fourier coefficients; and
- even fine-grid selection through `wgpu-fft`'s public supported-length
  factorization.

There is no GPU NUFFT execution yet. The next slices add 1D type-2 gathering,
then deterministic no-atomic type-1 spreading. The planned spreading baseline
uses exclusive output ownership rather than unavailable WGSL `f32` atomics.

Deferred work includes 2D/3D GPU execution, type 3, batching, NUFFT `f64`/df64
GPU arithmetic, and WASM packaging.

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
and dimension zero is stored fastest. Coordinates follow the documented
type-1/type-2 input contract `|x| <= 3*pi` and are periodic modulo `2*pi`.
