# 2D wgpu-nufft versus cuFINUFFT on RTX 5090 - 2026-07-13

The first 2D GPU implementation is correct and fully GPU-resident, but CUDA's
native implementation is substantially faster in this initial comparison. On
the same RTX 5090, cuFINUFFT leads the honestly paired spans by 6.91x and
17.75x for type 1 at 512 squared and 1024 squared, and by 2.86x and 3.65x for
type 2.

## Comparison boundary

Both implementations used identical logical `f32` points and `complex64`
values on the same GPU. Upload, download, output readback, input generation,
and plan construction are outside every headline timing. The mode and fine
grids use dimension zero as the fastest logical axis.

The fair span pairing follows the established 1D comparison:

- wgpu-nufft type 1 rebuilds its bins on every execution, so its
  submit-to-wait span is paired with cuFINUFFT `setpts+execute`;
- wgpu-nufft type 2 reads caller points directly and has no cached point
  preprocessing, so its submit-to-wait span is paired with cuFINUFFT
  execute-only after `setpts`.

All cuFINUFFT spans, including plan creation, `setpts`, execute, and combined,
are reported below. The wgpu timer starts immediately before `queue.submit`
and stops after an exact submission wait. The CUDA timer is host wall clock
around the named call or calls with the nonblocking CUDA stream synchronized
before and after each sample.

## Method

- Cases: square mode grids 512 by 512 and 1024 by 1024, with `M=N0*N1`
  nonuniform points.
- Precision and tolerance: `complex64`, `eps=1e-6`, `upsampfac=2.0`.
- Semantics: `isign=+1`, centered mode ordering, no normalization.
- Inputs: matching field-specific `u32` LCG streams, seed `0x4E554646`.
  Rust stores AoS `[x0,y0,...]`; CUDA receives equivalent separate device x/y
  arrays. The cuFINUFFT wrapper uses C mode shape `(N1,N0)` and reversed
  `setpts(x1,x0)`, which its own Python-to-column-major bridge restores to the
  same logical N0/x0-fast order.
- Statistics: three independently recreated plans, one excluded warmup per
  plan, ten samples per plan; `+/-` is standard error across the three run
  means. Type 1 records one transform per sample. Type 2 records/calls 32 and
  divides the synchronized span by 32.
- The wgpu and CUDA matrices ran back-to-back while the adapter was otherwise
  idle. Their commands took 8.8 and 45.2 seconds including startup and smoke
  checks, about 54 seconds total.

## Results

Times are milliseconds per transform. The bold cuFINUFFT value is the span
paired with the wgpu submit-to-wait measurement.

| Shape | Kind | wgpu submit-to-wait | cu plan | cu setpts | cu execute | cu setpts+execute | Same-silicon result |
|---:|---|---:|---:|---:|---:|---:|---|
| 512 x 512 | type 1 | 2.123800 +/- 0.021173 | 64.989133 +/- 0.615515 | 0.206200 +/- 0.006808 | 0.124493 +/- 0.004378 | **0.307247 +/- 0.010870** | cuFINUFFT 6.912x faster |
| 512 x 512 | type 2 | 0.182220 +/- 0.000190 | 65.440867 +/- 0.334280 | 0.069016 +/- 0.006047 | **0.063690 +/- 0.002337** | 0.151692 +/- 0.016153 | cuFINUFFT 2.861x faster |
| 1024 x 1024 | type 1 | 7.635960 +/- 0.042388 | 71.018533 +/- 0.926386 | 0.184410 +/- 0.002990 | 0.270850 +/- 0.008986 | **0.430173 +/- 0.004063** | cuFINUFFT 17.751x faster |
| 1024 x 1024 | type 2 | 0.646670 +/- 0.000293 | 70.326367 +/- 0.383903 | 0.066585 +/- 0.001382 | **0.176948 +/- 0.001308** | 0.259474 +/- 0.005103 | cuFINUFFT 3.655x faster |

Paired throughput is:

| Shape | Kind | wgpu Mpoints/s | cuFINUFFT Mpoints/s |
|---:|---|---:|---:|
| 512 x 512 | type 1 | 123.432 | 853.204 |
| 512 x 512 | type 2 | 1438.609 | 4115.950 |
| 1024 x 1024 | type 1 | 137.321 | 2437.566 |
| 1024 x 1024 | type 2 | 1621.502 | 5925.916 |

## Correctness and routing

The Vulkan suite covers `eps=1e-2..1e-6`, both signs, both mode orders, and
random, clustered, periodic-boundary, and duplicate point sets for both
transform types. At `eps=1e-6`, the worst measured relative L2 errors were
`4.324481e-6` for type 2 and `1.023760e-6` for type 1. Opposite-sign adjoint
residuals were `4.87e-9..1.91e-8`. Duplicate type-2 outputs and repeated type-1
executions were bit-identical. A non-square 256 by 1024 test measured
`6.748517e-5` for type 2 and `9.889337e-6` for type 1 at `eps=1e-5`.

The CUDA harness uses a deliberately non-square 8 by 6 smoke with distinct x
and y values, plus a zero-point analytic check. Its direct relative L2 errors
were `8.064752e-7` for type 1 and `1.010378e-6` for type 2; the zero-point errors
were `8.236331e-7` and `2.502721e-6`, all below the `2e-5` threshold. This smoke
also guards the wrapper's non-obvious Python/CUDA axis reversal.

The complete release workspace GPU run passed on NVIDIA GeForce RTX 5090,
Vulkan, driver 610.47. It included all existing wgpu-fft suites, both NUFFT
suites, the scan and profiling canaries, and the new 2D matrix.

## Interpretation

The type-1 result reflects the deliberate portable algorithm. WGSL has no
`f32` atomic add, so one invocation owns each fine-grid cell and gathers from
the surrounding flattened bins. For the width-7 kernel in 2D, that means an
8-by-8 bin neighborhood and tensor-product kernel evaluation. cuFINUFFT uses
CUDA-native spreading machinery. wgpu time grows 3.60x from 512 squared to
1024 squared, close to the 4x data growth, while the paired CUDA span grows
only 1.40x; the larger ratio is therefore a real crossover, not a noisy single
sample. A useful optimization follow-up is a shared-memory bin subproblem or
bin-group remapping, measured with 2D stage timestamps.

Type 2 is much closer, but CUDA still leads by 2.9-3.7x. The current aggregate
measurement cannot split that difference between the 2D fine-grid FFT and the
49-tap tensor interpolation. Per-stage 2D profiling is deferred rather than
guessing at the dominant component.

Deferred scope remains 3D GPU execution, type 3, batching, NUFFT f64/df64,
and WASM. No FINUFFT CPU, VkFFT, or unrelated sweep was rerun.

## Artifacts

- `wgpu-nufft-2d-results.txt`: adapter proof, limits, every raw wgpu sample,
  plan/encoding times, and aggregates.
- `cufinufft-2d-results.txt`: environment, correctness smoke, every CUDA
  plan/setpts/execute/combined sample, and aggregates.
- `run.toml`: commands, versions, hashes, source identity, and pairing rules.
- `../../benches/nufft_2d_bench.rs` and
  `../../benches/cufinufft_gpu_bench.py`: reusable benchmark harnesses.
