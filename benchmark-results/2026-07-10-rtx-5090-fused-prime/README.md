# RTX 5090 fused-prime results - 2026-07-10

Short validation of the one-workgroup Rader and Bluestein pipelines. VkFFT was
not rerun; all reference values below come from the archived
[RTX 5090 baseline](../2026-07-10-rtx-5090/README.md) and its saved
`vkfft-sample-1000.txt` rows.

## Method

- GPU: NVIDIA GeForce RTX 5090 (`10de:2b85`), Vulkan, driver 610.47
- starting point: `f3584d1` (`Generate FFT twiddles from host f64 tables`)
- timing: FFT+iFFT pairs recorded in one encoder, one submit, wall-clock
  submit-through-wait, two plan-recreated runs, iteration cap 200
- accuracy oracle: host `Complex64` transform
- fused diagnostics: one kernel pass, zero plan workspace, immutable
  permutation/chirp, BFFT, and child-twiddle LUT helpers only
- fusion floor: convolution lengths below 128 stay on the staged path because a
  256-lane workgroup would be mostly idle
- exact storage gate: `8*M + 8` bytes for Rader and `8*M` for Bluestein

```powershell
$env:WGPU_FFT_RUN_GPU_TESTS='1'; $env:WGPU_BACKEND='vulkan'
cargo test --test gpu_fused_prime --release -- --nocapture
cargo test --test gpu_accuracy --release -- --nocapture

cargo bench --bench fft_bench -- shape 2999 --batch 32768 --adapter "RTX 5090" --runs 2 --iter-cap 200
cargo bench --bench fft_bench -- shape 1009 --batch 131072 --adapter "RTX 5090" --runs 2 --iter-cap 200
cargo bench --bench fft_bench -- shape 4096 --batch 32768 --adapter "RTX 5090" --runs 2 --iter-cap 200
cargo bench --bench fft_bench -- shape 3000 --batch 32768 --adapter "RTX 5090" --runs 2 --iter-cap 200
```

## Accuracy

| Route/case | Forward max relative | Forward RMS relative | Inverse max relative | Inverse RMS relative |
|---|---:|---:|---:|---:|
| Rader N=101, batch 3 | 4.1800e-8 | 1.1281e-7 | 5.2578e-8 | 1.1628e-7 |
| Rader N=1009, batch 2 | 7.9341e-8 | 1.7543e-7 | 7.1972e-8 | 1.7889e-7 |
| Rader N=2999 | 8.0435e-8 | 1.9740e-7 | 7.9446e-8 | 1.9265e-7 |
| Bluestein N=221, batch 2 | 1.2448e-7 | 1.8081e-7 | 1.1208e-7 | 1.7874e-7 |
| Bluestein N=2026 | 7.7314e-8 | 2.0911e-7 | 7.0776e-8 | 2.0749e-7 |

The fused N=2999 result retains the prior staged route's approximately `2e-7`
RMS accuracy. A round-trip f32 literal is required for the internal `1/M`
normalization: formatting `1/6000` to nine decimal places creates a measurable
`~2e-6` scale bias, which the fused shaders now avoid.

The forced-16-KiB device test executes and compares both algorithms: Rader
N=2999 falls back to 17 kernels and Bluestein N=2026 to 11 kernels. Batched ND
and mixed-axis `AxisSequence` cases also match the f64 oracle. N=3256 remains
the unchanged 15-pass Bluestein fallback at 48 KiB.

## Performance

| N | Batch | Route/passes | Previous wgpu-fft | Fused-prime result | Change | Archived VkFFT | Current / VkFFT |
|---:|---:|---|---:|---:|---:|---:|---:|
| 2999 | 32768 | Rader, 7 -> 1 | 22.535766 +/- 0.015872 ms | **9.456181 +/- 0.011094 ms** | **2.383x faster** | 2.513 +/- 0.039 ms | 3.763x time |
| 1009 | 131072 | Rader, 7 -> 1 | not archived | **8.126392 +/- 0.050942 ms** | n/a | 3.304 +/- 0.032 ms | 2.460x time |
| 4096 | 32768 | fused pow2, 1 | 2.878008 +/- 0.031908 ms | 2.930096 +/- 0.054438 ms | +1.810% | 2.859 +/- 0.034 ms | 1.025x time |
| 3000 | 32768 | fused smooth, 1 | 2.099362 +/- 0.019244 ms | 2.157612 +/- 0.036731 ms | +2.775% | 2.083 +/- 0.019 ms | 1.036x time |

N=2999 eliminates the six intermediate global passes and the `lines*M` work
buffers, but does not reach the aspirational 1.5x VkFFT target. Each outer
Rader transform still performs a forward and inverse M=6000 convolution FFT;
an FFT+iFFT benchmark pair therefore executes four M=6000 FFTs, while the
48-KiB scratch allocation also limits resident workgroups. The remaining gap
is compute and occupancy, not diagnostic global-memory passes.

The N=3000 control crossed the 2% warning threshold. A second independent
two-run capture was 2.197262 +/- 0.030025 ms (+4.663% versus the archived
value). The standalone fused-smooth execution path and generated stage logic
are unchanged by this slice, so this is recorded as observed run-to-run/device
drift rather than attributed to the prime kernel; it is not hidden from the
comparison.
