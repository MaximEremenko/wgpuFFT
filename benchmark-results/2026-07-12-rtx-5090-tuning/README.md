# RTX 5090 public-tuning results - 2026-07-12

This archive records the public `FftTuning` slice and a short workgroup-size
sweep. The API/default implementation is commit `286f96a`; documentation,
GPU coverage, and the benchmark harness are completed by the following Phase B
commit.

## Method

- GPU: NVIDIA GeForce RTX 5090 (`10de:2b85`), NVIDIA driver 610.47
- backend: native Windows Vulkan; wgpu `29.0.3`
- case: f32 C2C N=4096, batch=32768, 1 GiB logical buffer
- timing: FFT+iFFT pairs, normalization disabled, one encoder and submit per
  run, wall clock through device wait, two plan-recreated runs
- iteration rule: `min(200, floor(3 * 4096 MiB / bufferSize))`, giving 12
  pairs per run
- `--workgroup-size N` sets both staged and fused workgroup controls; this case
  uses the fused power-of-two route, so the fused control is the active one
- no VkFFT command was run

```powershell
$env:WGPU_BACKEND='vulkan'
cargo bench --bench fft_bench -- shape 4096 --batch 32768 `
  --adapter "RTX 5090" --runs 2 --iter-cap 200
cargo bench --bench fft_bench -- shape 4096 --batch 32768 `
  --adapter "RTX 5090" --runs 2 --iter-cap 200 --workgroup-size 32
# Repeated with --workgroup-size 64, 128, and 256.
```

All five points completed in seconds, well below the five-minute limit.

## Measured defaults

`FftTuning::default()` preserves the pre-tuning planner choices.

| Control | Default | Role |
|---|---:|---|
| staged workgroup | 64 | Stockham and linear helper kernels |
| fused workgroup | 256 | fused power-of-two, smooth, and prime kernels |
| Rader maximum | 4096 | largest non-smooth prime automatically routed to Rader |
| forced Rader / Bluestein axes | empty | automatic per-axis policy |
| large route | Auto | choose normal, chunk, four-step, or segmented execution |
| large-chunk batch cap | none | device/shape-derived chunk size |
| grouped batch | none | device/shape-derived four-step window grouping |
| two-/three-window swap thresholds | 0 / 0 | disabled |
| segmented burst depth | 2 | measured two-pair A/B staging ring |
| storage-binding / buffer caps | none | use real device limits |
| fused-prime convolution floor | 128 | keep tiny prime convolutions on staged kernels |

Transpose thresholds are not public because the Rust normal route has no
coalescing transpose. FFT-convolution tuning is not public because there is no
public convolution API. The single-stage smooth fusion choice remains internal
because it does not remove a global pass.

## Workgroup sweep

| Tuning | Raw pair ms | Average +/- stderr | Relative to default | Modeled GiB/s |
|---|---|---:|---:|---:|
| default (staged 64, fused 256) | 2.867392, 2.849617 | **2.858504 +/- 0.008888 ms** | 1.000x | 1399.333 |
| 32 / 32 | 8.124358, 8.181533 | 8.152946 +/- 0.028587 ms | 2.852x | 490.620 |
| 64 / 64 | 5.009575, 4.957800 | 4.983688 +/- 0.025888 ms | 1.743x | 802.619 |
| 128 / 128 | 3.801517, 3.771117 | 3.786317 +/- 0.015200 ms | 1.325x | 1056.436 |
| 256 / 256 | 2.859908, 2.870308 | **2.865108 +/- 0.005200 ms** | 1.002x | 1396.108 |

The archived prior-session default was 2.886904 +/- 0.053929 ms. The new
default is 0.98% faster, within normal session drift, and explicit 256 is only
0.23% slower than the same-session default. The sweep therefore demonstrates
that the public knob changes generated pipelines while preserving the measured
256-lane default for this fused case; it is not a recommendation to change the
default globally.

## Correctness and validation

- `cargo fmt --check`, `cargo check --all-targets`, and all 299 unit tests pass.
- Every release GPU integration target passes on the RTX 5090 Vulkan adapter,
  including the 2 GiB four-step cases, dispatch splitting, f32/f64/df64
  accuracy, real, segmented, fused-prime, and fused-power-of-two coverage.
- `gpu_tuning` verifies implicit/default identity; forced Bluestein and forced
  Rader correctness; 32/64/128/256 mixed/fused equivalence and distinct cache
  keys; public four-step/segmented caps; burst depths 1 through 3; and typed
  invalid-value, device-limit, and infeasible-route errors.
- Default f32 accuracy remains approximately `9.69e-8` to `2.15e-7` RMS on the
  established accuracy matrix.

