# RTX 5090 portable-df64 NUFFT - 2026-07-15

This archive closes the portable double-float NUFFT precision slice for types
1, 2, and 3 in one through three dimensions. It compares df64 with native f64
in the same Vulkan session on one GPU. No VkFFT, FINUFFT, or cuFINUFFT run was
performed for this slice.

## Environment and method

- GPU: NVIDIA GeForce RTX 5090 (`10de:2b85`)
- Vulkan validation/benchmark driver: NVIDIA 610.47
- DX12 portability validation driver: 32.0.16.1047
- df64 device features: none; the DX12 test explicitly requested a featureless
  device
- tolerance for performance cases: `eps=1e-6`, `sigma=2`
- two plan-recreated runs, two measured samples per run, one untimed warmup
- one command buffer and submit per sample
- wall clock from immediately before `queue.submit` through `device.poll(Wait)`
- plan construction, command encoding, uploads, and readback excluded
- caller-owned input/output buffers remain GPU-resident
- stderr is across the two plan-recreated run means

```powershell
cargo bench -p wgpu-nufft --bench nufft_precision_bench -- `
  --precision 'f64,df64' --case '1d,2d,3d,type3' --kind both `
  --adapter 'RTX 5090' --runs 2 --samples 2
```

The first df64 plan creation paid substantial shader-compilation cost (up to
12.8 seconds), but plan creation is outside the declared execution timing and
subsequent plan creation was much lower. The high stderr on the 1D/2D df64
type-2 rows reflects a slow first measured run; those ratios should be read as
directional rather than as finely separated throughput claims.

## Implementation and correctness

Df64 stores a scalar as `(hi, lo)` and a complex value as
`(re_hi, re_lo, im_hi, im_lo)`, using only f32 WGSL. Coordinates occupy 8 bytes
and complex values 16 bytes, the same storage footprint as native f64. Host-f64
Horner coefficients, amplitudes, and type-3 factors are split into high and low
f32 words. The shaders contain no f64 types and no `exp`, `log`, `pow`, `sin`,
or `cos` calls.

The Vulkan release matrix covered types 1/2/3, dimensions 1/2/3, both signs,
batch 2, centered and FFT mode orders, direct `Complex64` oracles, adjoints, and
type-1/type-2 capacity-to-smaller-active-batch scratch reuse. At `eps=1e-8`, the worst
relative L2 errors were:

| Transform | 1D | 2D | 3D |
|---|---:|---:|---:|
| Type 1 | 5.24547e-9 | 4.72471e-9 | 4.00126e-9 |
| Type 2 | 2.73936e-9 | 8.30987e-9 | 6.01560e-9 |
| Type 3 | 6.31694e-9 | 1.19349e-8 | 3.67061e-8 |

The tighter 1D `eps=1e-11` matrix measured:

| Transform | Relative L2 error |
|---|---:|
| Type 1 | 5.39388e-12 |
| Type 2 | 2.38099e-12 |
| Type 3 | 1.32685e-11 |

Type-1/type-2 adjoint residuals were at most `1.77107e-14`; type-3 residuals
were approximation-limited and at most `1.27521e-8` in the `eps=1e-8` matrix.
The phase-reduction review found that the native-f64 `1e6` phase bound was not
safe for df64: exact emulation reached about `2.9e-9` error there. Df64 now has
a structured `|phase| <= 1024` gate; uniform and adversarial quadrant-boundary
canaries measured a worst absolute sine/cosine error of `3.64975e-12`.

The featureless DX12 run exercised representative batched 1D type-1, type-2,
and type-3 plans and reproduced the Vulkan oracle errors (`2.73936e-9`,
`5.24547e-9`, and `6.31694e-9`). During development, optimized DX12 exposed a
compiler-sensitive NaN in the type-3 quadrature reduction. Keeping the compact
quadrature loop while using the mathematically bounded direct quadrant reducer
for its `|angle| < 8*pi` domain fixed it; the general source/post phase path
retains full range reduction. A focused GPU stage canary guards this case.

## Performance

| Case | Kind | native f64 ms | df64 ms | df64 / f64 |
|---|---|---:|---:|---:|
| 1D, N=M=262144 | Type 1 | 0.790950 +/- 0.005850 | 1.214875 +/- 0.047075 | 1.5360x |
| 1D, N=M=262144 | Type 2 | 0.335725 +/- 0.016325 | 0.800550 +/- 0.197000 | 2.3845x |
| 2D, 512x512, M=N | Type 1 | 10.921750 +/- 0.094150 | 12.428000 +/- 0.269050 | 1.1379x |
| 2D, 512x512, M=N | Type 2 | 0.498650 +/- 0.002950 | 0.978875 +/- 0.136975 | 1.9631x |
| 3D, 64x64x64, M=N | Type 1 | 161.971925 +/- 1.233525 | 184.467875 +/- 0.337925 | 1.1389x |
| 3D, 64x64x64, M=N | Type 2 | 1.463950 +/- 0.005750 | 5.223200 +/- 0.173800 | 3.5679x |
| Type 3, M=K=65536 | Type 3 | 16.397550 +/- 0.003050 | 28.367275 +/- 0.042025 | 1.7300x |

Df64 does not beat native f64 on this RTX 5090. Its value is availability on
backends without `SHADER_F64`, not peak speed. The near-1.14x multidimensional
type-1 ratios are not representative arithmetic ratios: both high-precision
paths use the same deterministic global-gather fallback because their doubled
kernel-weight storage does not fit the established f32 shared-memory tiles.
Type-2 exposes more of the software dd arithmetic cost, reaching 3.57x in 3D.

## Portability limits and validation

Df64 retains roughly 44-48 significant bits but has the f32 exponent range,
approximately `1e-38` through `1e38`. Values beyond that range overflow or
underflow even though significand precision is higher. Split-based Dekker
products avoid an FMA dependency, and the exact-word wgpu-fft canary passed all
96 expected words on Vulkan and DX12. Metal remains the riskiest untested
backend because its compiler enables fast-math transformations by default.

- `cargo fmt --check`
- `cargo test --workspace`: 299 wgpu-fft and 90 wgpu-nufft unit tests, plus 19
  CPU-foundation tests and all integration targets green
- scoped strict clippy for the wgpu-nufft library/tests and precision benchmark
- exhaustive release df64 NUFFT matrix on RTX 5090 Vulkan, NVIDIA 610.47
- representative featureless release df64 NUFFT matrix on RTX 5090 DX12,
  driver 32.0.16.1047
- independent adversarial review; the phase-bound defect above was fixed and no
  confirmed defect remained

With batching, native f64, and portable df64 complete, goal 4's planned NUFFT
wrap-up surface is closed. Native f64 remains preferred when available; df64 is
the portable high-precision fallback.
