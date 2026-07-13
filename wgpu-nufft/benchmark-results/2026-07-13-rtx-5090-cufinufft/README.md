# wgpu-nufft versus cuFINUFFT on RTX 5090 - 2026-07-13

This is the first same-silicon comparison between portable wgpu-nufft and
CUDA-native cuFINUFFT. At 262,144 points, wgpu-nufft is nominally 1.19x faster
for type 1 and 1.09x faster for type 2. At 1,048,576 points, cuFINUFFT is
nominally 1.37x faster for type 1 and 2.27x faster for type 2. The 262k type-2
and 1M type-1 ratios are noisy. The measured means show a crossover rather
than one constant factor: portable wgpu is competitive at 262k, CUDA has a
clear 1M type-2 advantage, and the 1M type-1 mean also favors CUDA but is not
statistically decisive here.

## Comparison boundary

Both implementations ran on the same NVIDIA GeForce RTX 5090 with identical
GPU-resident `f32`/`complex64` inputs. H2D, D2H, output readback, input
generation, and plan creation are outside every paired timing.

The fair paired span differs by transform type because the current APIs do
different amounts of point preprocessing inside an execution:

- wgpu-nufft type 1 repeats count, hierarchical scan, deterministic scatter,
  per-bin point-index sort, output-driven gather, FFT, and deconvolution on
  every execution. Its submit-to-wait result is therefore paired with
  cuFINUFFT `setpts+execute`, which includes point preprocessing/sorting plus
  spread, FFT, and deconvolution. Comparing it with cuFINUFFT execute-only
  would omit work that wgpu-nufft repeats.
- wgpu-nufft type 2 reads the caller's points directly during interpolation;
  it has no separate cached point preprocessing. Its submit-to-wait result is
  paired with cuFINUFFT execute-only. cuFINUFFT setpts and combined timings are
  still reported for changing-point workloads.

The wgpu timer starts immediately before `queue.submit` and ends after
`device.poll(Wait)` for that submission; command encoding is excluded and
reported separately in the raw log. The CUDA timer is host wall clock around
the requested cuFINUFFT call or calls, with the plan's nonblocking CUDA stream
explicitly synchronized before and after each sample. This includes Python and
native-call overhead inside the CUDA span. Type 1 uses one transform per timed
sample. Type 2 enqueues 32 transforms per sample and divides the synchronized
wall time by 32 in both harnesses.

## Method

- Cases: `N=M` in `{262144, 1048576}`, one dimension.
- Precision and tolerance: `complex64`, `eps=1e-6`, `upsampfac=2.0`.
- Semantics: `isign=+1`, centered ordering (`modeord=0`).
- Inputs: the exact Phase D field-specific `u32` LCG streams, seed
  `0x4E554646`, converted to `f32` identically to the Rust harness.
- Statistics: three independently recreated plans, one excluded warmup per
  plan, ten samples per plan; `+/-` is standard error across the three run
  means. All 30 raw samples are archived.
- Outputs were preallocated. No H2D or D2H transfer was issued inside a timed
  region.
- The cuFINUFFT command, including its correctness smoke test, took 82.2
  seconds. The final wgpu control took 6.4 seconds including Cargo startup, for
  about 88.6 seconds total—well below five minutes.

The installed cuFINUFFT 2.5.1 wheel was used. It ran successfully on Blackwell
compute capability 12.0, so the source-build fallback was not needed. On this
Windows installation, the harness retains CUDA and `cufinufft.libs` DLL search
handles and imports CuPy before cuFINUFFT; without that loader setup, importing
the wheel directly could not resolve its CUDA dependencies.

## Results

All times are milliseconds per transform. The bold cuFINUFFT column is the
span paired with the wgpu submit-to-wait measurement.

| N=M | Kind | wgpu submit-to-wait | cu plan creation | cu setpts | cu execute | cu setpts+execute | Fair comparison |
|---:|---|---:|---:|---:|---:|---:|---|
| 262,144 | type 1 | **0.329147 +/- 0.002084** | 118.612267 +/- 2.576640 | 0.360200 +/- 0.051900 | 0.147440 +/- 0.012700 | **0.392077 +/- 0.024473** | wgpu nominally 1.19x faster |
| 262,144 | type 2 | **0.104107 +/- 0.000702** | 173.897633 +/- 24.732560 | 0.099542 +/- 0.001334 | **0.113652 +/- 0.025912** | 0.302922 +/- 0.026318 | nominal wgpu 1.09x; inconclusive within cu noise |
| 1,048,576 | type 1 | **5.167313 +/- 0.418336** | 185.058000 +/- 2.494874 | 2.465640 +/- 0.031564 | 0.228607 +/- 0.018466 | **3.763740 +/- 0.919458** | cuFINUFFT nominally 1.37x faster |
| 1,048,576 | type 2 | **0.273338 +/- 0.001359** | 183.858533 +/- 3.117519 | 1.602796 +/- 0.138361 | **0.120167 +/- 0.008126** | 2.340909 +/- 0.485889 | cuFINUFFT 2.27x faster |

The corresponding paired throughputs are:

| N=M | Kind | wgpu Mpoints/s | cuFINUFFT Mpoints/s |
|---:|---|---:|---:|
| 262,144 | type 1 | 796.435 | 668.604 |
| 262,144 | type 2 | 2518.018 | 2306.548 |
| 1,048,576 | type 1 | 202.925 | 278.599 |
| 1,048,576 | type 2 | 3836.189 | 8725.991 |

## Interpretation and session drift

The final, fully archived wgpu control was run in the same session after the
CUDA measurements and only after `nvidia-smi` showed the device idle. Relative
to the prior hierarchical-scan archive, its four rows were 3.77%, 2.10%, 5.30%,
and 12.16% slower, respectively. Those archived values were not used in the
headline ratios. A preliminary wgpu run taken while a stale CUDA process was
still consuming 100% of the GPU was rejected before comparison and its log was
not archived.

The raw samples expose meaningful Windows scheduling jitter:

- cuFINUFFT 262k type-2 execute contains one 0.874522 ms per-transform sample;
  its three run means are 0.086538, 0.088961, and 0.165458 ms. The nominal 1.09x
  wgpu lead is therefore not statistically persuasive.
- cuFINUFFT 1M type-1 combined contains one 31.0332 ms sample; its run means are
  2.829900, 5.602580, and 2.858740 ms. Retaining the prescribed mean makes the
  CUDA result slower, so the direction of its large-case lead is not created by
  removing the outlier, but 1.37x should not be read as a precise ratio.
- wgpu 1M type-1 run means are 4.331200, 5.611850, and 5.558890 ms. Its own
  variability reinforces the same qualification.

Scaling by four in point and mode count increases paired type-1 time by 15.70x
for wgpu and 9.60x for cuFINUFFT. Within cuFINUFFT, type-1 setpts grows 6.85x
and execute grows 1.55x. Paired type-2 grows 2.63x for wgpu and 1.06x for
cuFINUFFT. These measurements establish worse large-case scaling in the current
wgpu route, but they do not identify a stage.

The installed cuFINUFFT was left on automatic method selection. Source-level
option and heuristic review indicates that this case resolves to its
shared-memory type-1 spreading method, whereas wgpu-nufft uses a deterministic,
atomic-free binned gather. That algorithmic difference is a plausible source
of the type-1 crossover, but no per-stage timestamp or elimination profile was
captured here. Attribution between binning/sort, gather/spread, and the
fine-grid FFT remains an explicit follow-up; the type-2 gap likewise cannot yet
be divided between FFT and interpolation.

## Correctness and provenance

Before timing, the CUDA harness compared both transform types at `N=32, M=41`
against direct f64 NDFTs and zero-point analytic results. The point set included
boundaries and duplicates. Relative L2 errors were:

| Check | Relative L2 error |
|---|---:|
| direct type 1 | 1.055241e-6 |
| direct type 2 | 1.480638e-6 |
| analytic type 1 | 8.629368e-7 |
| analytic type 2 | 8.226420e-7 |

All are below the harness threshold of `2e-5`. The unchanged wgpu-nufft route's
larger oracle and adjoint matrix is recorded in the
[Phase D](../2026-07-13-rtx-5090-1d/README.md) and
[hierarchical-scan](../2026-07-13-rtx-5090-1d-scan/README.md) archives.

Environment and measured artifacts:

- GPU: NVIDIA GeForce RTX 5090, compute capability 12.0, PCI
  `0000:01:00.0`; NVIDIA driver 610.47.
- CUDA driver API 13.3 (`13030`); CuPy-linked runtime 12.9 (`12090`);
  installed nvcc toolkit 12.8.93.
- cuFINUFFT 2.5.1, CuPy 14.1.1, NumPy 2.4.0, Python 3.12.3.
- Loaded cuFINUFFT DLL SHA-256:
  `DEA7281755EEB8A0E090822F14E2A3E09742F5CAEFCAF4C4FBFB0E0E717FDD53`.
- cuFINUFFT harness SHA-256:
  `094555A1B4AD1640FBAA47367F0AB55FA40B7129314E78C71B4E3FF60371B0CC`.
- Measured wgpuFFT base commit:
  `a0b6e778ddbd3b6f8ee390c70ed637a56e29b58a`.
- The local FINUFFT checkout was not used. Its observed identity was
  `d919b19bae518b858a2504c01867333488d40935`
  (`v2.5.0-132-gd919b19b`), while the installed wheel was the measured artifact.

## Archive files

- `cufinufft-gpu-results.txt` contains the environment banner, correctness
  smoke, all plan/setpts/execute/combined samples, run means, and aggregates.
- `wgpu-nufft-gpu-results.txt` contains adapter/backend proof and all wgpu raw
  samples and aggregates from the final control.
- `run.toml` records the commands, limits, package versions, hashes, pairing,
  and source identities.
- `../../benches/cufinufft_gpu_bench.py` is the reusable CUDA benchmark and
  correctness harness.
