# wgpu-nufft 1D Phase D results - 2026-07-13

This archive closes the first 1D NUFFT slice with short type-1 and type-2
measurements on the RTX 5090 and contextual FINUFFT CPU measurements on the
same machine.

## Comparison boundary

These numbers are deliberately **not** presented as an apples-to-apples CPU/GPU
speedup claim. The implementations run on different devices and make different
point-preprocessing choices:

- wgpu-nufft's headline is GPU-resident wall time from `queue.submit` through a
  wait for that submission. Plan creation, command encoding, host-to-device
  upload, readback, and caller-side synchronization before submit are excluded.
- FINUFFT's headline is its reusable single-precision plan's `execute` call.
  Plan creation and `setpts` are excluded and reported separately. FINUFFT can
  therefore reuse point sorting, while the current wgpu-nufft type-1 route
  rebuilds bins and their stable order on every execution.
- Type 1 uses one transform per GPU submission because each encode allocates
  about 7 MiB or 28 MiB of transient bin/sort scratch for these cases. Type 2
  has no such per-encode buffer allocation, so 32 transforms are recorded in
  each timed submission and the elapsed time is divided per transform. This
  amortizes Windows queue-wait jitter that overwhelmed sub-millisecond type-2
  work in the initial one-transform measurements.

The ratios below are useful implementation context on this one workstation,
not general hardware claims.

## Method

- wgpu-nufft implementation base: `77ef024` (`Execute 1D type-1 NUFFTs on the GPU`)
- GPU: NVIDIA GeForce RTX 5090 (`10de:2b85`, PCI `0000:01:00.0`), NVIDIA 610.47
- GPU backend: native Windows Vulkan through wgpu 29.0.3
- CPU: Intel Core i9-13900K, 24 physical / 32 logical cores
- FINUFFT: installed official Python wheel 2.4.1, complex64, bundled FFTW and
  OpenMP runtime, explicit `nthreads=32`
- transform parameters: 1D, `N=M`, f32/complex64, `eps=1e-6`, `sigma=2.0`,
  positive sign, centered/CMCL mode order, ES width 7 and beta 16.1
- cases: `N=M=262144` and `N=M=1048576`; fine grids are `2*N`
- input: identical deterministic field-specific u32 LCG streams in both
  harnesses, seed `0x4E554646`, points uniform in `[-pi, pi)`
- sampling: three plan recreations, one excluded warmup per plan, ten timed
  samples per plan; `+/-` is standard error across the three run means
- GPU type-1 transient scratch per encode: `12*nf + 4*(M+1)` bytes, 7 MiB
  plus 4 bytes and 28 MiB plus 4 bytes for the two cases
- the two archived commands took 20.352 s and 2.399 s respectively, 22.751 s
  total, well below the five-minute limit

FINUFFT warns that 32 OpenMP threads exceed the processor's 24 physical cores.
The archive retains its explicit all-logical-processor setting and records the
warning; no cross-thread-count performance claim is made.

## Results

All times are milliseconds per transform.

| N=M | Kind | wgpu-nufft submit-wait | GPU encode | GPU Mpoints/s | FINUFFT execute | FINUFFT setpts | FINUFFT Mpoints/s | Context on this host |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 262,144 | type 1 | **81.443230 +/- 0.796074** | 0.252780 | 3.219 | **5.246173 +/- 0.195396** | 2.659433 | 49.969 | FINUFFT execute 15.52x faster |
| 262,144 | type 2 | **0.144004 +/- 0.002654** | 0.068693 | 1820.394 | **3.233340 +/- 0.076052** | 0.588800 | 81.075 | resident GPU 22.45x faster |
| 1,048,576 | type 1 | **326.045310 +/- 0.767455** | 2.404173 | 3.216 | **12.648807 +/- 0.346898** | 15.530667 | 82.899 | FINUFFT execute 25.78x faster |
| 1,048,576 | type 2 | **0.383752 +/- 0.015428** | 0.073386 | 2732.434 | **9.678457 +/- 0.290714** | 2.233467 | 108.341 | resident GPU 25.22x faster |

The secondary point-preparation context leads to the same qualitative result.
For type 1, FINUFFT `setpts+execute` is 7.905607 ms and 28.179473 ms, while
GPU `encode+submit-wait` is 81.696010 ms and 328.449483 ms. For type 2 those
pairs are 3.822140 vs 0.212697 ms and 11.911923 vs 0.457138 ms. These sums
still exclude transfers and are not end-to-end application timings.

## Interpretation

Type 2 is already a strong GPU-resident baseline: deconvolution, the public
wgpu-fft plan, and interpolation remain below 0.4 ms at one million points once
queue-wait overhead is amortized.

Type 1 identifies the next optimization unambiguously. With uniformly random
points and a fine grid twice as large as the point set, most bins contain zero
or one point, so per-bin heapsort is not the dominant work here. The current
single-invocation exclusive prefix scan walks every fine-grid bin and type-1
time grows almost exactly fourfold when the fine grid grows fourfold
(81.44 -> 326.05 ms). A hierarchical parallel scan, followed by parallel bin
sorting for clustered inputs, is the required performance follow-up. The
current route remains the deterministic portable correctness baseline.

GPU plan creation is intentionally outside the execution headline. Raw plan
times include first-use shader/pipeline compilation and then cache-warm
recreations, which is why their spread is much larger than execution stderr.
No synthetic bandwidth number is reported; NUFFT traffic depends on the
spreading/interpolation strategy, so Mpoints/s is the honest common throughput
metric.

## Correctness and validation

- The Phase C `gpu_nufft` suite validates both signs and mode orders over
  `eps=1e-2..1e-6`, random, clustered, boundary, duplicate, odd-mode, zero-point,
  repeatability, and structured-error cases against the direct f64 NDFT oracle.
- Its worst type-1 relative L2 error at `eps=1e-6` is about `1.145e-6`; the
  worst type-1/type-2 adjoint residual is about `1.581e-8`.
- The performance harness intentionally performs no readback; this keeps its
  timing scope representative of GPU-resident chaining and relies on the
  separate correctness suite.
- `cargo fmt --all -- --check`, locked all-target workspace compilation, and
  strict no-dependency Clippy for `wgpu-nufft` all passed.
- Locked workspace tests passed, including 299 `wgpu-fft` unit tests and 24
  `wgpu-nufft` unit/CPU-reference tests.
- The release `gpu_nufft` matrix passed on the NVIDIA GeForce RTX 5090 through
  Vulkan with driver 610.47. The run reproduced the error and adjoint bounds
  above.
- Adversarial review corrected the CLI's stale sample-count help, the exact
  scratch-byte formula, and an unsupported thread-sensitivity provenance claim;
  no other confirmed defect remained.

## Archive files

- `wgpu-nufft-gpu-results.txt`: compact adapter, method, raw run means, and
  final GPU result lines
- `finufft-cpu-results.txt`: FINUFFT metadata, raw per-call samples, run means,
  and result lines
- `run.toml`: exact commands, versions, source identities, and binary hashes

The measured GPU harness hash and final archived hash differ only because the
post-run review corrected its `--help` text from five samples to the actual
default of ten. `run.toml` records both hashes; no timing-path code changed.

The local FINUFFT source checkout was clean at `d919b19b`
(`v2.5.0-132-gd919b19b`) but had no built library and was not used for these
numbers. The tested artifact is explicitly the installed FINUFFT 2.4.1 wheel;
the archive does not attribute its timings to the checkout.
