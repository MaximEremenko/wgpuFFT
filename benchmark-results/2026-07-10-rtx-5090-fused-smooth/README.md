# RTX 5090 fused smooth-radix results - 2026-07-10

Short validation run for the single-workgroup smooth-radix C2C kernel. VkFFT
was not rerun; reference values come from the archived
[RTX 5090 sample-1000 baseline](../2026-07-10-rtx-5090/README.md) and its
preserved `vkfft-sample-1000.txt` output.

## Method

- GPU: NVIDIA GeForce RTX 5090 (`10de:2b85`), driver 610.47, Vulkan
- `wgpu-fft`: fused-smooth change based on `422e553`
- FFT+iFFT pairs recorded into one command encoder, one submit per run
- two plan-recreated runs, 3 x 4096 MiB traffic-budget iterations, cap 200
- out-of-place `wgpu-fft` buffers; archived VkFFT reference is in place
- normalization disabled for timed pairs

```powershell
cargo bench --bench fft_bench -- shape 3000 --batch 32768 --adapter "RTX 5090" --runs 2 --iter-cap 200
cargo bench --bench fft_bench -- shape 2187 --batch 32768 --adapter "RTX 5090" --runs 2 --iter-cap 200
cargo bench --bench fft_bench -- shape 2999 --batch 32768 --adapter "RTX 5090" --runs 2 --iter-cap 200
cargo bench --bench fft_bench -- shape 4096 --batch 32768 --adapter "RTX 5090" --runs 2 --iter-cap 200
```

## Results

| N | Batch | Route | Diagnostic passes, before -> now | Previous `wgpu-fft` | Fused-smooth result | Score | Archived VkFFT | Current / VkFFT |
|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 3000 | 32768 | mixed radix | 5 -> 1 | not archived | **2.615934 +/- 0.000322 ms** | 293585 | 2.083 +/- 0.019 ms, score 368621 | 1.256x time |
| 2187 | 32768 | mixed radix | 7 -> 1 | not archived | **1.769289 +/- 0.020180 ms** | 316439 | 1.549 +/- 0.004 ms, score 361384 | 1.142x time |
| 2999 | 32768 | Rader | 15 -> 7 | not archived | **22.535766 +/- 0.015872 ms** | 34068 | 2.513 +/- 0.039 ms, score 305473 | 8.968x time |
| 4096 | 32768 | fused pow2 | 1 -> 1 | 2.886904 +/- 0.053929 ms | **2.866408 +/- 0.025275 ms** | 365815 | 2.859 +/- 0.034 ms, score 366724 | 1.003x time |

The old pass counts for the newly measured cases are the prior graph topology,
not archived timings. Direct smooth transforms now perform one global load and
one global store: N=3000 reaches 79.6% of the archived VkFFT score and N=2187
reaches 87.6%. The remaining direct-smooth gap is kernel arithmetic and
workgroup efficiency rather than extra global Stockham passes.

N=2999 demonstrates the expected partial benefit for a Rader transform whose
smooth convolution child fits the 48 KiB storage gate. Its child FFTs fuse, but
the full Rader route still has seven diagnostic passes and remains about 9x
slower than VkFFT. N=3256 was intentionally not rerun: its convolution length
does not fit the gate, so this slice is not expected to materially change the
archived 63.99 ms control.

The unchanged power-of-two route is 0.71% faster than the previous short run
and 0.26% behind the archived VkFFT time, within run-to-run spread. Single-radix
lengths such as N=3, 5, 7, 11, and 13 remain on the existing one-pass Stockham
kernel because fusing cannot remove a pass and would over-dispatch 256 lanes per
short line.
