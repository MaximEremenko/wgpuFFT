# RTX 5090 native-f64 NUFFT - 2026-07-14

This archive closes the native-f64 NUFFT precision slice for types 1, 2, and 3
in one through three dimensions. It is a same-session f32/f64 comparison on one
GPU, not a comparison with FINUFFT or cuFINUFFT.

## Environment and method

- GPU: NVIDIA GeForce RTX 5090 (`10de:2b85`), Vulkan, NVIDIA driver 610.47
- native-f64 device feature: `wgpu::Features::SHADER_F64`, explicitly enabled
- tolerance for the performance cases: `eps=1e-6`, `sigma=2`
- two plan-recreated runs, two measured samples per run, one untimed warmup
- one command buffer and submit per sample
- wall clock from immediately before `queue.submit` through `device.poll(Wait)`
- plan construction, command encoding, uploads, and readback excluded
- caller-owned input/output buffers remain GPU-resident
- stderr is across the two plan-recreated run means

```powershell
cargo bench -p wgpu-nufft --bench nufft_precision_bench -- `
  --precision 'f32,f64' --case '1d,2d,3d,type3' --kind both `
  --adapter 'RTX 5090' --runs 2 --samples 2
```

## Correctness

The release Vulkan precision suite instantiated every native-f64 shader on the
RTX 5090 and covered types 1/2/3, dimensions 1/2/3, both signs, batch 2,
boundary/duplicate/random points, direct `Complex64` oracles, per-vector
adjoints, and a feature-disabled structured-error device.

At `eps=1e-8`, the worst relative L2 error was `3.67061e-8` (3D type 3). A
tighter 1D submatrix at `eps=1e-12` measured:

| Transform | Positive sign | Negative sign |
|---|---:|---:|
| Type 1 | 5.66182e-13 | 5.66111e-13 |
| Type 2 | 3.73625e-13 | 4.27787e-13 |
| Type 3 | 7.98501e-13 | 9.52887e-13 |

The worst type-1/type-2 adjoint residual was `2.36358e-15`. Type-3 adjoint
residuals remain tolerance-limited because the NUFFT approximation, not native
floating-point roundoff, dominates them.

High-precision ES weights use host-fitted FINUFFT-style piecewise-Horner
polynomials. The same polynomial is integrated for deconvolution coefficients.
Native-f64 shaders contain no `exp`, `log`, `pow`, `sin`, or `cos`; type-3 uses
an arithmetic polynomial sin/cos with a structured `|phase| <= 1e6` plan gate.

## Performance

| Case | Kind | f32 ms | native f64 ms | f64 / f32 |
|---|---|---:|---:|---:|
| 1D, N=M=262144 | Type 1 | 0.321175 +/- 0.001575 | 0.963625 +/- 0.112575 | 3.0003x |
| 1D, N=M=262144 | Type 2 | 0.152025 +/- 0.004675 | 0.353525 +/- 0.002725 | 2.3254x |
| 2D, 512x512, M=N | Type 1 | 0.661125 +/- 0.080925 | 11.284975 +/- 0.027275 | 17.0694x |
| 2D, 512x512, M=N | Type 2 | 0.237075 +/- 0.002225 | 0.548325 +/- 0.023025 | 2.3129x |
| 3D, 64x64x64, M=N | Type 1 | 6.753050 +/- 0.336200 | 159.756000 +/- 0.158450 | 23.6569x |
| 3D, 64x64x64, M=N | Type 2 | 1.007000 +/- 0.043550 | 1.408750 +/- 0.049000 | 1.3990x |
| Type 3, M=K=65536 | Type 3 | 13.185675 +/- 0.342975 | 16.446850 +/- 0.082100 | 1.2473x |

The large multidimensional type-1 ratios have a specific implementation cause:
the established f32 path uses shared-memory bin tiles, while native f64 uses the
deterministic global gather because doubled cached weights exceed the portable
tile storage budget. This is an explicit safe fallback, not evidence that f64
arithmetic alone costs 17-24x. Type-2 and type-3 give the more representative
same-route precision ratios on this consumer-rate FP64 GPU.

## Validation

- `cargo fmt --check`
- `cargo test --workspace`: 299 wgpu-fft unit tests, 83 wgpu-nufft unit tests,
  19 CPU-foundation tests, and all integration targets green
- all wgpu-nufft release Vulkan GPU targets green on the adapter above
- scoped strict clippy for wgpu-nufft library/tests and the new benchmark
- adversarial review found no confirmed functional defects

Portable df64 remains explicitly structured-unsupported in this commit and is
the next precision slice.
