# Many-vector NUFFT batching on RTX 5090 - 2026-07-14

This archive closes the `f32` many-vector batching slice. A plan now has a
FINUFFT-style `ntransf` capacity, all transforms share one point set, and
complex values use transform-major layout. Types 1, 2, and 3 support batching
in one, two, and three dimensions. The timed matrix covers types 1 and 2;
type 3 is covered by oracle, loop-of-single, adjoint, and scratch-reuse tests.

The main implementation result is amortization, not a universal kernel-speed
win. Native batching reduces type-1 total time versus recording independent
single-vector executions by 2.76x at `ntr=4` and 3.71x at `ntr=16` in 1D, and
by 1.70x and 2.22x in 3D. Type-2 amortization is smaller and mixed because it
has no bin/scan/sort phase to share. The short Windows measurements also have
large outliers in several 2D and CUDA rows; those ratios are retained and
flagged rather than filtered.

## Implementation boundary

- `NufftConfig::with_batch(ntr)` and `NufftType3Config::with_batch(ntr)` set a
  reusable plan capacity. Ordinary encode calls execute the full configured
  count; explicit batch encode calls may execute `1..=ntr` active vectors.
- Points remain shared and point-major. Strengths, modes, and outputs are
  `[transform][point or mode]` interleaved complex `f32`, matching FINUFFT's
  `ntransf` convention.
- Type 1 runs count, hierarchical scan, stable scatter, and bin sorting once.
  Spread kernels evaluate each support weight once for blocks of four vectors.
  Type 2 likewise reuses interpolation support weights in four-vector blocks.
- The oversampled-grid FFT is a batched public `wgpu-fft` C2C plan. An active
  count below capacity still executes the capacity-sized FFT after clearing
  inactive grids, so active shrink is a correctness/reuse feature rather than
  a proportional-work guarantee.
- Type 3 composes the same capacity-batched spread, FFT, and interpolation
  paths. It was deliberately omitted from this short performance matrix.

## Method

The three cases all use `N=M=262144`: `262144`, `512x512`, and `64x64x64`.
Precision is `f32`/`complex64`, `eps=1e-6`, `sigma=2`, `isign=+1`, and centered
mode order. Counts are `ntr={1,4,16}`. Each result is the mean of two
independently recreated plans with two timed submissions per plan and one
excluded warmup. `+/-` is standard error across the two run means.

For wgpu, native batching is one batch plan and one encode. The loop control
records `ntr` sequential encodes through one reusable batch-1 plan. Each method
uses one command buffer and one submit; the wall clock starts immediately
before `queue.submit` and ends after `device.poll(Wait)`. Plan construction,
encoding, uploads, and readback are excluded.

cuFINUFFT uses native `Plan(n_trans=ntr)` with one shared `setpts` and one
execute. `gpu_maxbatchsize=min(ntr,8)` is explicit because the installed source
path's advertised zero/automatic branch is not reachable. At `ntr=1`, the
legacy harness path is pinned to one type-2 execution. CUDA spans are bounded
by explicit stream synchronizations and exclude transfers. The honest pairing
is:

- wgpu type 1 submit-to-wait versus cuFINUFFT `setpts+execute`;
- wgpu type 2 submit-to-wait versus cuFINUFFT execute-only.

## Native batching versus loop-of-single

All values are batch-total milliseconds.

| Shape | Kind | ntr | Native batch ms | Loop ms | Loop/native |
|---|---|---:|---:|---:|---:|
| 262144 | type 1 | 1 | 0.295650 +/- 0.000900 | 0.296900 +/- 0.003850 | 1.00x |
| 262144 | type 1 | 4 | 0.415550 +/- 0.008650 | 1.147875 +/- 0.004125 | 2.76x |
| 262144 | type 1 | 16 | 1.206650 +/- 0.063350 | 4.474600 +/- 0.024350 | 3.71x |
| 262144 | type 2 | 1 | 0.131975 +/- 0.001875 | 0.130575 +/- 0.000025 | 0.99x |
| 262144 | type 2 | 4 | 0.285900 +/- 0.000150 | 0.419400 +/- 0.002100 | 1.47x |
| 262144 | type 2 | 16 | 1.205500 +/- 0.182150 | 1.666300 +/- 0.048250 | 1.38x |
| 512x512 | type 1 | 1 | 0.883400 +/- 0.134550 | 0.975075 +/- 0.209625 | 1.10x |
| 512x512 | type 1 | 4 | 2.880275 +/- 0.889675 | 3.302850 +/- 0.478950 | 1.15x |
| 512x512 | type 1 | 16 | 8.127650 +/- 4.173400 | 10.503450 +/- 2.011750 | 1.29x |
| 512x512 | type 2 | 1 | 0.595825 +/- 0.304375 | 0.402600 +/- 0.051550 | 0.68x |
| 512x512 | type 2 | 4 | 4.398675 +/- 0.840025 | 1.495350 +/- 0.040750 | 0.34x |
| 512x512 | type 2 | 16 | 3.109825 +/- 0.031775 | 3.290550 +/- 0.289500 | 1.06x |
| 64x64x64 | type 1 | 1 | 8.162625 +/- 0.640025 | 8.566475 +/- 0.193175 | 1.05x |
| 64x64x64 | type 1 | 4 | 15.530350 +/- 3.314300 | 26.362575 +/- 0.337075 | 1.70x |
| 64x64x64 | type 1 | 16 | 46.974625 +/- 0.155925 | 104.258200 +/- 0.173850 | 2.22x |
| 64x64x64 | type 2 | 1 | 1.220025 +/- 0.072075 | 1.158025 +/- 0.000725 | 0.95x |
| 64x64x64 | type 2 | 4 | 3.920950 +/- 0.018350 | 3.572100 +/- 0.015850 | 0.91x |
| 64x64x64 | type 2 | 16 | 15.201675 +/- 0.003025 | 14.325050 +/- 0.031200 | 0.94x |

The 2D rows are too variable to support small-factor claims. In particular,
the non-monotonic type-2 `ntr=4`/`16` totals are a measurement warning, not an
optimization conclusion. The 1D and 3D type-1 curves are the clear result:
point preprocessing and support evaluation are genuinely amortized.

## Same-silicon cuFINUFFT context

The cuFINUFFT column is the paired span defined above. `wgpu/cu` above one
means CUDA was faster. A dagger marks a row where either implementation's
standard error exceeded roughly 40% of its mean, so its ratio is descriptive
only.

| Shape | Kind | ntr | wgpu batch ms | Paired cuFINUFFT ms | wgpu/cu |
|---|---|---:|---:|---:|---:|
| 262144 | type 1 | 1 | 0.295650 | 0.284050 +/- 0.036200 | 1.04x |
| 262144 | type 1 | 4 | 0.415550 | 0.362450 +/- 0.002850 | 1.15x |
| 262144 | type 1 | 16 | 1.206650 | 9.018950 +/- 0.508050 | 0.13x |
| 262144 | type 2 | 1 | 0.131975 | 0.089450 +/- 0.004650 | 1.48x |
| 262144 | type 2 | 4 | 0.285900 | 0.201875 +/- 0.035225 | 1.42x |
| 262144 | type 2 | 16 | 1.205500 | 0.875100 +/- 0.474250 | 1.38x dagger |
| 512x512 | type 1 | 1 | 0.883400 | 0.283875 +/- 0.000125 | 3.11x |
| 512x512 | type 1 | 4 | 2.880275 | 0.482250 +/- 0.014800 | 5.97x |
| 512x512 | type 1 | 16 | 8.127650 | 9.105450 +/- 7.898150 | 0.89x dagger |
| 512x512 | type 2 | 1 | 0.595825 | 0.098900 +/- 0.009350 | 6.03x dagger |
| 512x512 | type 2 | 4 | 4.398675 | 0.242825 +/- 0.010675 | 18.12x |
| 512x512 | type 2 | 16 | 3.109825 | 0.815625 +/- 0.007575 | 3.81x |
| 64x64x64 | type 1 | 1 | 8.162625 | 0.694325 +/- 0.017725 | 11.76x |
| 64x64x64 | type 1 | 4 | 15.530350 | 3.380425 +/- 0.039725 | 4.59x |
| 64x64x64 | type 1 | 16 | 46.974625 | 15.384175 +/- 6.620825 | 3.05x dagger |
| 64x64x64 | type 2 | 1 | 1.220025 | 0.640625 +/- 0.409775 | 1.90x dagger |
| 64x64x64 | type 2 | 4 | 3.920950 | 1.562875 +/- 0.075725 | 2.51x |
| 64x64x64 | type 2 | 16 | 15.201675 | 3.277925 +/- 0.004975 | 4.64x |

The CUDA `ntr=16` type-1 samples had large first-run stalls: 2D run means were
17.003600 and 1.207300 ms, and 3D run means were 22.005000 and 8.763350 ms.
The prescribed unfiltered means are shown. The 1D `ntr=16` type-1 result was
also anomalously slow despite low reported between-run standard error. No
headline portability claim is based on those rows. On stable rows, wgpu stays
close in 1D but cuFINUFFT remains materially faster in multidimensional cases;
batching completes the functional surface and does not erase the already
documented multidimensional CUDA gap.

## Correctness and regression evidence

The RTX 5090 Vulkan batch suite covered both signs, types 1/2 in 1D/2D/3D,
type 3 in 1D/2D/3D, transform-major loop-of-single comparison, per-vector
adjoints, and grow/shrink scratch reuse with different point counts. A batch
of five explicitly crosses the four-vector shader block and exercises its
partial tail. Batched versus looped outputs were bit-identical.

| Check | Worst observed result |
|---|---:|
| type-1/type-2 relative L2 versus direct f64 oracle | 2.274e-5 |
| type-3 relative L2 versus direct f64 oracle | 7.032e-6 |
| type-1/type-2 per-vector adjoint residual | 5.113e-8 |
| type-3 per-vector adjoint residual | 4.397e-7 |
| capacity-4 to active-2 changed-count reuse | bit-identical to fresh plans |

All cuFINUFFT native-batch correctness smokes for 1D/2D/3D and `ntr=4/16`
were below their `2e-5` or `3e-5` direct-oracle thresholds.

## Environment and provenance

- GPU: NVIDIA GeForce RTX 5090, PCI `0000:01:00.0`; NVIDIA driver 610.47.
- wgpu backend: native Windows Vulkan.
- CUDA: driver API 13.3 (`13030`), CuPy-linked runtime 12.9 (`12090`), toolkit
  path CUDA 12.8.
- Python 3.12.3, NumPy 2.4.0, CuPy 14.1.1, cuFINUFFT 2.5.1.
- Loaded cuFINUFFT DLL SHA-256:
  `DEA7281755EEB8A0E090822F14E2A3E09742F5CAEFCAF4C4FBFB0E0E717FDD53`.
- Measured base commit: `684d83f31298b65ff143f4f482afb86960fd4a89`.
- Rust batch harness SHA-256:
  `28183630049CEFFE06600679DF292460A0C08ADCCD65DE3A1BC4E5621F442C45`.
- CUDA harness SHA-256:
  `9F5FBA0516BE02FB90D782D6A85A884A3A16E1A448B90E9C1C294BFD1E3DCE9C`.

Commands:

```powershell
cargo bench -p wgpu-nufft --bench nufft_batch_bench -- --all --ntr 1 --ntr 4 --ntr 16 --runs 2 --samples 2 --adapter "RTX 5090"
python wgpu-nufft/benches/cufinufft_gpu_bench.py <dimension-case> --ntrans <1|4|16> --runs 2 --samples 2 --warmups 1
```

The reusable harnesses are
[`nufft_batch_bench.rs`](../../benches/nufft_batch_bench.rs) and
[`cufinufft_gpu_bench.py`](../../benches/cufinufft_gpu_bench.py).
