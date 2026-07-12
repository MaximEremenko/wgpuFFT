# RTX 5090 native-f64 results - 2026-07-12

This archive records native-f64 C2C correctness and short performance results
for the three precision slices: public precision/device plumbing, normal
mixed-radix kernels, and Rader/Bluestein plus mixed-algorithm ND execution.

## Method

- GPU: NVIDIA GeForce RTX 5090 (`10de:2b85`), PCI `0000:01:00.0`, NVIDIA
  driver 610.47, Vulkan, `SHADER_F64` enabled
- device workgroup storage: 48 KiB
- wgpu-fft starting point for Phase C: `1240ca4`
- accuracy oracle: host `Complex64`; RMS relative tolerance `1e-13`
- wgpu-fft timing: FFT+iFFT pairs, normalization disabled, one encoder and one
  submit, wall clock from submit through device wait, two plan-recreated runs
- iteration rule: `min(200, floor(3 * 4096 MiB / precision-sized bufferSize))`
- `--precision both` runs f32 then f64 in the same process on the same adapter
  and device; the fixed order is reported rather than hidden
- wgpu-fft is out of place; VkFFT sample 1 is in place
- wgpu-fft `+/-` is standard error across two runs; VkFFT's field named
  `std_error` is its population standard deviation across three runs

```powershell
$env:WGPU_FFT_RUN_GPU_TESTS='1'; $env:WGPU_BACKEND='vulkan'
cargo test --test gpu_f64 --release -- --nocapture

cargo bench --bench fft_bench -- shape 4096 --batch 16384 `
  --precision both --adapter "RTX 5090" --runs 2 --iter-cap 200
cargo bench --bench fft_bench -- shape 2048 --batch 32768 `
  --precision both --adapter "RTX 5090" --runs 2 --iter-cap 200

# VkFFT v1.3.4, Vulkan backend 0, Release; run exactly once
.\VkFFT_TestSuite.exe -d 0 -o vkfft_sample1_rtx5090.txt -vkfft 1
```

## Correctness

All native-f64 cases are below the requested `1e-13` RMS-relative threshold.

| Case | Selected path | RMS relative error |
|---|---|---:|
| N=2048 | fused pow2 | 1.5222e-15 |
| N=2048, forced 16 KiB | multipass pow2 | 1.5437e-15 |
| N=4096 inverse | multipass pow2 | 2.3378e-15 |
| N=3000 | fused smooth | 1.9707e-15 |
| Rader N=509, forward/inverse | fused prime | 1.1705e-15 / 1.2094e-15 |
| Rader N=1601 | native-f64 storage fallback | 1.9439e-15 |
| Bluestein N=515, forward/inverse | fused prime | 1.6079e-15 / 1.6482e-15 |
| Bluestein N=1544 | native-f64 storage fallback | 2.8762e-15 |
| ND `[2,509]`, batch 2, forward/inverse | mixed + Rader | 1.2494e-15 / 1.3029e-15 |
| ND `[2,515]`, batch 2, forward/inverse | mixed + Bluestein | 1.6595e-15 / 1.7981e-15 |

Forced-16-KiB fused-versus-multipass RMS difference is `5.2458e-16` for
Rader N=509 and exactly zero for Bluestein N=515. The no-`SHADER_F64`, real,
large-chunk, four-step, segmented-volume, helper-footprint, and storage-limit
boundaries all return structured `PrecisionUnsupported` errors.

`cargo test` passes 253 unit tests and every integration target. All eight
existing f32 RTX 5090 Vulkan release suites also pass; established f32 accuracy
remains approximately `9.69e-8` to `2.15e-7` RMS-relative.

## Performance

### Same-session f32 controls

| N | Batch | f32 path / passes | f32 pair time | f64 path / passes | f64 pair time | f64 / f32 | f32 / f64 modeled GiB/s |
|---:|---:|---|---:|---|---:|---:|---:|
| 2048 | 32768 | fused / 1 | 1.722038 +/- 0.014779 ms | fused / 1 | **9.326329 +/- 0.002837 ms** | 5.4159x | 1161.4 / 428.9 |
| 4096 | 16384 | fused / 1 | 1.635231 +/- 0.002506 ms | Stockham / 4 | **21.950896 +/- 0.680513 ms** | 13.4237x | 1223.1 / 728.9 |

At N=2048 both precisions fuse, yet f64 takes 5.42x rather than the 2x
bandwidth-only expectation and reaches only 37% of the modeled f32 bandwidth.
This is ALU-bound behavior from the RTX 5090's consumer-rate FP64 units, not a
LUT or dispatch defect. At N=4096, 64 KiB of f64 scratch cannot fit the 48 KiB
workgroup-storage limit, so f64 additionally expands from one fused pass to four
global Stockham passes. The 8x minimum byte/pass disadvantage becomes 13.42x
after the same FP64 arithmetic pressure.

### VkFFT sample 1

| N | Batch | VkFFT f64 pair time | wgpu-fft f64 pair time | wgpu-fft / VkFFT |
|---:|---:|---:|---:|---:|
| 2048 | 32768 | 7.543 +/- 0.211 ms | **9.326329 +/- 0.002837 ms** | 1.2364x |
| 4096 | 16384 | 8.678 +/- 0.540 ms | **21.950896 +/- 0.680513 ms** | 2.5295x |

The fitting N=2048 native-f64 fused kernel is within 23.6% of VkFFT. N=4096 is
2.53x slower because wgpu-fft deliberately obeys the 48 KiB shared-memory gate
and uses four global passes. This is the expected architectural boundary for
this first native-f64 implementation.

VkFFT sample 1 completed all 23 systems in 48.0 seconds with aggregate score
65,824. Its printed double-precision `benchmark` score normalizes buffer bytes
by `sizeof(float)/sizeof(double)`, so raw pair time is used for the comparison
instead of comparing that field to wgpu-fft's full-byte score. Sample 1001 was
not run.

## Archive integrity

- `vkfft-sample-1.txt`: 3,289 bytes, SHA-256
  `A9152C7BC3A9182B32D003517AA290C5426F1B415B31D130FE26F046F7331BCD`
- VkFFT executable: SHA-256
  `B47C142653090C6B357ED357A259972A14C5CD6C458FEBFCCD225A484E43287B`
- VkFFT `v1.3.4`: `066a17c17068c0f11c9298d848c2976c71fad1c1`
- glslang `12.3.1`: `4f3ae4b03dc3556f96f55467e139f852831199d0`

Exact machine/build metadata and source hashes are in `run.toml`; compact raw
wgpu-fft result lines are in `wgpu-fft-bench-summary.txt`.
