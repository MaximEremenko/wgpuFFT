# RTX 5090 fused power-of-two results - 2026-07-10

Short validation run for the single-workgroup power-of-two C2C kernel. VkFFT
was not rerun; reference values come from the archived
[RTX 5090 baseline](../2026-07-10-rtx-5090/README.md).

## Method

- GPU: NVIDIA GeForce RTX 5090 (`10de:2b85`), driver 610.47, Vulkan
- `wgpu-fft`: fused-kernel change based on `e82ecf5`
- FFT+iFFT pairs, one command encoder and submit per run
- two plan-recreated runs, VkFFT traffic-budget iteration count, cap 200
- out-of-place buffers, normalization disabled for timed pairs

```powershell
cargo bench --bench fft_bench -- shape 4096 --batch 32768 --adapter "RTX 5090" --runs 2 --iter-cap 200
cargo bench --bench fft_bench -- shape 2048 --batch 32768 --adapter "RTX 5090" --runs 2 --iter-cap 200
cargo bench --bench fft_bench -- shape 1024 --batch 65536 --adapter "RTX 5090" --runs 2 --iter-cap 200
cargo bench --bench fft_bench -- shape 3256 --batch 32768 --adapter "RTX 5090" --runs 2 --iter-cap 200
```

## Results

| N | Batch | Logical buffer | Before `wgpu-fft` | Fused result | Diagnostic passes | Archived VkFFT | Comparison |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4096 | 32768 | 1024 MiB | 12.0085 +/- 0.0827 ms | **2.886904 +/- 0.053929 ms** | 4 -> 1 | 2.858 ms | 4.160x faster than before; 1.01% behind VkFFT |
| 2048 | 32768 | 512 MiB | not measured | **1.437954 +/- 0.000246 ms** | 1 | score 366286 at 1 GiB | score 364606.892; 0.46% below VkFFT |
| 1024 | 65536 | 512 MiB | not measured | **1.440115 +/- 0.016198 ms** | 1 | score 364320 at 1 GiB | score 364059.920; 0.07% below VkFFT |
| 3256 | 32768 | 814 MiB | not measured | 63.993220 +/- 0.022113 ms | 15 | 2.209 ms | Bluestein fallback control; route unchanged |

The N=4096 fused plan reports zero external workspace and estimated one-pass
axis traffic of 1385.6 GiB/s. The archived VkFFT run reported 1399.4 GiB/s.
The non-power-of-two control remains on Bluestein; no prior `wgpu-fft` timing
was archived for that exact case, so it is a routing control rather than a
before/after claim.
