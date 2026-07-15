# 3D and type-3 wgpu-nufft versus cuFINUFFT on RTX 5090 - 2026-07-14

This archive closes the first functional 3D and type-3 NUFFT surface with a
same-silicon comparison against cuFINUFFT. For regular 3D type 1 and type 2,
CUDA's paired spans are 3.85-4.77x faster. For type 3, wgpu-nufft's complete
submit-to-wait span is 2.54x, 20.37x, and 27.21x faster than cuFINUFFT's
`setpts+execute` span in 1D, 2D, and 3D respectively.

The type-3 result is not evidence that wgpu's FFT core is faster. cuFINUFFT
spends 25.7-74.9 ms in `setpts`, which dominates its paired result, while its
execute-only spans are 0.18-0.54 ms. The measured advantage therefore belongs
to this GPU-resident execution/API boundary for changing points, not to an
isolated FFT comparison. A workload that reuses cuFINUFFT's preprocessed points
would compare against execute-only and has a different result.

## Comparison boundary

Both implementations ran in the same session on one NVIDIA GeForce RTX 5090
with matched `f32`/`complex64` inputs already resident on the GPU. Input
generation, upload, output readback, plan creation, and Rust command encoding
are outside the paired timings. Outputs were preallocated.

- wgpu-nufft measures host wall clock from immediately before `queue.submit`
  through the exact `device.poll(Wait)` for that submission.
- cuFINUFFT measures host wall clock around the named API span with its
  nonblocking CUDA stream explicitly synchronized before and after every
  sample.
- Type 1 is paired with cuFINUFFT `setpts+execute`, because wgpu-nufft repeats
  point binning and spreading on each execution.
- Type 2 is paired with cuFINUFFT execute-only, following an excluded `setpts`,
  because the established wgpu type-2 path reads caller points directly.
- Type 3 is paired with cuFINUFFT `setpts+execute`; the wgpu composition repeats
  source preprocessing/spreading and target rescaling/interpolation in its
  submitted work.

Every row uses two independently recreated plans, one excluded warmup per
plan, five samples per plan, and one transform per timed sample. Reported
uncertainty is standard error across the two recreated-plan means. CUDA
`setpts`, execute, and combined are independently sampled, so their displayed
means are not expected to add exactly.

Common parameters were `eps=1e-6`, `sigma=2.0`, positive sign, seed
`0x4E554646`, and matched field-specific 24-bit `u32` LCG streams. Regular 3D
uses centered mode order and axis zero as the fastest logical axis. Type 3 has
no mode-order setting; each dimension uses `M=K=65,536`, source interval center
`pi/4` and halfwidth `pi`, and target interval center `4` and halfwidth `16`.
Pinned `f32` endpoints give both libraries the same effective bounds.

## Regular 3D results

All times are milliseconds per transform. Bold cuFINUFFT values are the spans
paired with the wgpu submit-to-wait measurement.

| Shape and M | Kind | wgpu submit-to-wait | cu setpts | cu execute | cu setpts+execute | Same-silicon result |
|---:|---|---:|---:|---:|---:|---|
| 64^3 | type 1 | 4.893430 +/- 0.036830 | 0.202950 +/- 0.004810 | 0.572840 +/- 0.009400 | **1.025770 +/- 0.218110** | cuFINUFFT 4.77x faster |
| 64^3 | type 2 | 1.058960 +/- 0.024420 | 0.078500 +/- 0.010380 | **0.248390 +/- 0.021770** | 0.288770 +/- 0.002490 | cuFINUFFT 4.26x faster |
| 128^3 | type 1 | 37.843860 +/- 0.049820 | 0.575730 +/- 0.346670 | 9.142800 +/- 4.785900 | **9.661620 +/- 5.133920** | cuFINUFFT nominally 3.92x faster; noisy |
| 128^3 | type 2 | 7.500650 +/- 0.038430 | 0.146880 +/- 0.001680 | **1.949350 +/- 0.080190** | 2.023930 +/- 0.013010 | cuFINUFFT 3.85x faster |

The 128^3 CUDA type-1 row contains major cold outliers. Its second recreated
plan averaged 4.5277 ms for the combined span, while the prescribed mean of
both plans is 9.661620 ms with 5.133920 ms standard error. The table retains
the predeclared mean-of-run-means statistic; the 3.92x ratio should not be read
as a precise estimate.

The retained 8x8x4 workgroup-tiled type-1 spreader cleared its bounded
prototype gate before this matrix: its gather stage was 7.45x faster than the
global path at 64^3 and 6.98x faster at 128^3. Those are internal-path ratios,
not comparisons with CUDA. The remaining 3D gap is recorded without further
speculative optimization, as required by this functional-surface slice.

## Type-3 results

The paired cuFINUFFT span is combined `setpts+execute`. Component means are
shown to make the source of the result explicit.

| Dimensions | wgpu submit-to-wait | cu setpts | cu execute | cu setpts+execute | Paired result |
|---:|---:|---:|---:|---:|---|
| 1 | 10.240430 +/- 0.033290 | 25.668210 | 0.176490 | 25.963770 +/- 0.184930 | wgpu-nufft 2.54x faster |
| 2 | 2.459750 +/- 0.038930 | 49.720340 | 0.223790 | 50.114760 +/- 0.099680 | wgpu-nufft 20.37x faster |
| 3 | 2.762990 +/- 0.561670 | 74.935990 | 0.542270 | 75.182290 +/- 0.217170 | wgpu-nufft 27.21x faster |

cuFINUFFT `setpts` accounts for approximately 99% of each combined type-3
span and grows with dimension. Its execute-only kernel remains much shorter
than wgpu-nufft's end-to-end submission. The sober conclusion is that
wgpu-nufft exposes a favorable GPU-resident, execute-with-current-points API
for this workload; it does not establish general type-3 or FFT superiority.
The 3D wgpu result is also visibly noisy across only two recreated plans.

## Correctness and routing

The CUDA harness ran direct-NDFT smoke tests before timing:

| Transform | Positive-sign relative L2 | Negative-sign relative L2 |
|---|---:|---:|
| regular 3D type 1 | 1.0796e-6 | not sampled |
| regular 3D type 2 | 1.1839e-6 | not sampled |
| type 3, 1D | 3.63e-6 | 8.16e-6 |
| type 3, 2D | 7.68e-6 | 8.89e-6 |
| type 3, 3D | 1.215e-5 | 1.083e-5 |

The regular 3D smoke threshold was `3e-5`; the type-3 threshold was `1e-4`.
The wgpu Phase C oracle, opposite-sign adjoint, adversarial-point, and
back-to-back scratch-reuse suites passed on the same RTX 5090 Vulkan adapter.
The default-feature 2D and 3D release regressions also passed immediately
before close-out on Vulkan driver 610.47.

Phase A additionally executed one gated 256^3 type-2 correctness case. Its
512^3 fine grid occupied 1 GiB per complex buffer, selected wgpu-fft's normal
mixed-radix ND route on this adapter's approximately 2 GiB storage-binding
limit, and matched the sampled oracle at relative L2 `1.1234621e-6`. This was a
real large-volume consumer, but it did not require four-step routing on the
RTX 5090's unusually high binding limit.

## Environment and provenance

- GPU: NVIDIA GeForce RTX 5090, Vulkan and CUDA on NVIDIA driver 610.47.
- Rust stack: rustc 1.95.0, wgpu 29.0.3.
- CUDA toolkit: nvcc 12.8.93; CuPy-linked CUDA runtime 12.9; driver API 13.3.
- Python stack: Python 3.12.3, cuFINUFFT 2.5.1, CuPy 14.1.1, NumPy 2.4.0.
- Measured code commit: `45a1d12ea4ef825b34feeacfa41bc6e95ce14738`.
- Rust harness SHA-256:
  `9F0B934179213BF696F7F500A0D982BD3C2ACC5F0CBE01184366995007F60E32`.
- Python harness SHA-256:
  `233A437239849262E36F5D46C90C8F392C42DD546DD984A1BF76ED55D9CFE8C2`.
- Loaded cuFINUFFT DLL SHA-256:
  `DEA7281755EEB8A0E090822F14E2A3E09742F5CAEFCAF4C4FBFB0E0E717FDD53`.
- The local FINUFFT checkout was `d919b19bae518b858a2504c01867333488d40935` and
  was not used for CUDA timing. Type-3 mathematical semantics and rescaling
  were separately pinned and reviewed against FINUFFT reference commit
  `704cbfee`.

## Artifacts

The raw Rust and CUDA result logs are added alongside this document. They
contain adapter evidence, every sample, recreated-plan means, correctness
smokes, and the independently measured CUDA spans. `run.toml` records the
commands and machine-readable comparison contract. No FINUFFT-CPU or VkFFT
benchmark was rerun for this slice.
