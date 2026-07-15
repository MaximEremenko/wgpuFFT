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
- reusable 1D, 2D, and 3D type-2 `f32`, native-`f64`, and portable
  double-float GPU plans with caller-owned point, Fourier-coefficient, and
  output buffers, including FINUFFT-style many-vector batching;
- reusable 1D, 2D, and 3D type-1 plans in the same three precisions with
  caller-owned point, strength, and Fourier-mode output buffers, including
  many-vector batching;
- reusable 1D, 2D, and 3D type-3 plans in the same three precisions with
  caller-owned source points, strengths, target frequencies, and output
  buffers, also with many-vector batching.

The GPU type-2 route pre-deconvolves and zero-pads on the GPU, executes a public
`wgpu-fft` C2C plan on the oversampled grid, and gathers the ES interpolation
kernel at each nonuniform point. The caller records execution into its own
command encoder and can consume the caller-owned output buffer without a
readback. Plan-owned fine-grid buffers are reused between ordered executions.
Points are point-major values in the configured precision (scalar in 1D and, for example,
`[x0, y0, z0, x1, y1, z1, ...]` in 3D), while coefficients and outputs are
interleaved complex `(re, im)` pairs; all require `STORAGE` buffer usage.
Call `NufftPlan::required_point_buffer_size_bytes` when sizing a plan's
coordinate buffer; the older transform-specific static helpers retain their 1D
contract.

`NufftConfig::with_batch(ntr)` and `NufftType3Config::with_batch(ntr)` follow
FINUFFT's `ntransf` convention: every transform shares one point set, while
complex inputs and outputs are transform-major (`[transform][point or mode]`).
Type-1 binning, scan, and stable sorting run once per execution, and the spread
and interpolation kernels reuse each computed support weight across blocks of
up to four vectors. The oversampled-grid C2C plan uses `wgpu-fft`'s native batch
dimension. The ordinary encode methods execute the configured count; the
explicit `*_batch` methods may execute `1..=ntr` vectors on the same plan for
grow/shrink workflows. The embedded FFT remains capacity-sized in that case,
so a smaller active count is a reuse feature rather than a promise of
proportionally lower FFT work.

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

## Precision

`NufftConfig::with_precision` and `NufftType3Config::with_precision` select the
coordinate, complex-value, kernel, and fine-grid FFT precision together. `F32`
is the default and preserves the original shader path. `F64` uses scalar `f64`
coordinates and `vec2<f64>` complex values and requires a device created with
`wgpu::Features::SHADER_F64`; this is currently a Vulkan capability in wgpu.
Plans return a structured `PrecisionUnsupported` error before shader creation
when that feature is absent. `Df64` represents every scalar as an unevaluated
`hi + lo` pair of `f32` words and every complex value as
`(re_hi, re_lo, im_hi, im_lo)`. It requires no optional device feature and is
the portable high-precision route for DX12, Metal, and browser/WebGPU devices
that do not expose native shader `f64`.

The ES kernel cannot call `exp` in native-f64 SPIR-V, and df64 deliberately
contains no shader transcendental operations. Both high-precision routes fit
FINUFFT-style piecewise-Horner tables on the host in `f64` and use the same
polynomial for interpolation/spreading and Fourier deconvolution. Native-f64
tables store `f64` coefficients; df64 tables split each coefficient into its
high and low `f32` words. Type-3 phase factors use a transcendental-free,
range-reduced polynomial sine/cosine. Native f64 conservatively rejects planned
phase magnitudes above `1e6`; portable df64 uses the tighter, accuracy-driven
bound of `1024` because its two-f32 range reduction loses absolute precision as
the unreduced phase grows.

Df64 retains roughly 44-48 significant bits, but its exponent range remains
that of `f32` (approximately `1e-38` through `1e38`). Its split-based Dekker
products avoid relying on fused multiply-add contraction; exact-word arithmetic
canaries in `wgpu-fft` define the backend support contract. Vulkan and DX12 are
tested on the RTX 5090. Metal remains the riskiest untested backend because its
shader compiler enables fast-math transformations by default. Plan-aware
byte-size helpers account for 8-byte coordinates and 16-byte complex values in
both high-precision formats; legacy static helpers retain their documented
one-dimensional f32 ABI.

Both the crate and the `wgpu-web` JavaScript surface run on browser WebGPU.
Chrome/Tint testing at exact WebGPU defaults covers batched type 1/2/3 in one
through three dimensions for F32 and Df64. The browser compiler must first pass
all 96 exact df64 arithmetic words; otherwise only Df64 is disabled. Native F64
remains structurally unsupported in browsers because `SHADER_F64` is absent.
The browser wrapper keeps plans and caller buffers GPU-resident and exposes
explicit upload, awaited execution, and download operations.

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
