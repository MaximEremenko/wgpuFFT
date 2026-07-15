# wgpu-nufft

`wgpu-nufft` is a portable nonuniform FFT library built only on `wgpu-fft`'s
public API. Planning is separate from execution and plans are reusable.

## Current scope

The current implementation provides:

- FINUFFT-compatible type-1, type-2, and type-3 definitions in one to three
  dimensions;
- direct `f64` NDFT execution for every transform kind as the correctness oracle
  and tiny-problem fallback;
- `sigma = 2` default exponential-of-semicircle kernel planning;
- host-`f64` centered-grid kernel Fourier coefficients;
- even fine-grid selection through `wgpu-fft`'s public supported-length
  factorization;
- reusable 1D, 2D, and 3D type-2 `f32` GPU plans with caller-owned point,
  Fourier-coefficient, and output buffers;
- reusable 1D, 2D, and 3D type-1 `f32` GPU plans with caller-owned point,
  strength, and Fourier-mode output buffers; and
- reusable 1D, 2D, and 3D type-3 `f32` GPU plans with caller-owned source
  points, strengths, target frequencies, and output buffers.

The GPU type-2 route pre-deconvolves and zero-pads on the GPU, executes a public
`wgpu-fft` C2C plan on the oversampled grid, and gathers the ES interpolation
kernel at each nonuniform point. The caller records execution into its own
command encoder and can consume the caller-owned output buffer without a
readback. Plan-owned fine-grid buffers are reused between ordered executions.
Points are point-major `f32` values (scalar in 1D and, for example,
`[x0, y0, z0, x1, y1, z1, ...]` in 3D), while coefficients and outputs are
interleaved complex `(re, im)` `f32` pairs; all require `STORAGE` buffer usage.
Call `NufftPlan::required_point_buffer_size_bytes` when sizing a plan's
coordinate buffer; the older transform-specific static helpers retain their 1D
contract.

The 1D Vulkan accuracy matrix enforces relative L2 error at most `4*eps` over
the complete random-plus-adversarial vector and `8*eps` for each isolated
boundary, cluster, duplicate, and seeded-random subset. The 2D and 3D matrices
cover the same tolerance range, signs, mode orders, and adversarial point
classes at FINUFFT CUDA's pinned `20*eps` float acceptance ceiling, plus
determinism, non-square/non-cubic grids, scratch reuse, and opposite-sign
adjoint checks.

The GPU type-1 route bins point indices with portable `u32` atomics, scans the
flattened 1D, 2D, or 3D bin counts hierarchically, restores a stable input order
within each bin, and assigns each fine-grid cell to exactly one gathering
invocation. It therefore needs no unavailable WGSL `f32` atomics, and repeated
executions accumulate strengths in deterministic point order. In 2D and 3D,
the ES spreading/interpolation weights and deconvolution amplitudes are tensor
products, with dimension zero stored fastest. Shared-memory tiles accelerate
the default multidimensional spreading path while retaining a global fallback
for device limits that cannot support the tile. The gathered grid passes
through a public `wgpu-fft` ND C2C plan before deconvolution and truncation into
the caller-owned mode buffer. The type-1/type-2 routes reuse plan-owned fine-grid
scratch. Type 1 also reuses fixed bin-count, cursor, and offset buffers and grows
its sorted-point-index buffer only when a larger point set requires it.
Executions on a given plan must remain in queue order.

The GPU type-3 route follows FINUFFT's rescaling composition: it rescales and
pre-phases nonuniform sources, spreads them to an outer uniform grid, evaluates
that grid through an inner type-2 plan, then applies the continuous ES-kernel
correction and post-phase at rescaled nonuniform target frequencies. A reusable
GPU plan cannot inspect arbitrary device buffers during construction, so
`NufftType3Config` requires conservative source and target intervals. Every
coordinate supplied to later GPU executions must remain finite and inside those
planned intervals. The direct CPU route validates this contract explicitly;
GPU execution treats it as a caller guarantee. Absurd space-bandwidth products
and rescalings that cannot fit the GPU index, buffer, or portable double-float
phase range return structured planning errors. Use the dimension-aware
`NufftType3Plan::required_*_buffer_size_bytes` methods to size each caller-owned
source, target, strength, and output buffer. Type-3 plans likewise reuse
grow-only internal scratch and require ordered execution on a given plan.

GPU NUFFT execution is currently `f32` only and processes one transform per
execution; NUFFT batching and `f64`/df64 arithmetic remain deferred. WASM
packaging is also deferred. The underlying `wgpu-fft` crate already exposes the
needed native-`f64` and portable-df64 C2C plumbing, but those precision paths
have not yet been threaded through the NUFFT kernels and buffers.

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

For nonuniform source points `x_j` and nonuniform target frequencies `s_k`,
type 3 computes

```text
f_k = sum_j c_j exp(isign * i * dot(s_k, x_j)).
```

There is no normalization. Nonnegative `isign` selects the positive sign;
negative values select the negative sign. Centered mode order is the default,
and dimension zero is stored fastest. Type 3 has no integer-mode ordering.
Type-1/type-2 coordinates follow FINUFFT's documented `|x| <= 3*pi` contract
and are periodic modulo `2*pi`; type-3 source and target domains are instead
the explicit intervals supplied at planning. With the conventional Hermitian
inner product, type-2 with sign `s` is adjoint to type-1 with sign `-s`; a
type-3 plan's adjoint swaps source and target sets and reverses the sign.
