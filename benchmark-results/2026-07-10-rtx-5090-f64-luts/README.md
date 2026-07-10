# RTX 5090 host-f64 LUT results - 2026-07-10

Accuracy and short performance validation for host-generated f64 twiddle and
chirp tables rounded once to f32. VkFFT was not rerun; reference values come
from the archived [RTX 5090 baseline](../2026-07-10-rtx-5090/README.md).

## Method

- GPU: NVIDIA GeForce RTX 5090 (`10de:2b85`), driver 610.47, Vulkan
- starting point: `bd4af30` (`Fuse smooth-radix FFTs in one workgroup`)
- accuracy oracle: host `Complex64` separable DFT using the same input signal;
  the final oracle evaluates integer-reduced `(k*n) mod N` phases per term
- the baseline capture used an f64 root recurrence whose roughly 1e-12 phase
  drift is below the displayed GPU-error scale and does not change the rounded
  improvement factors
- max relative error: maximum complex absolute error divided by peak reference
  magnitude
- RMS relative error: square root of total error energy divided by total
  reference energy
- timing: FFT+iFFT pairs in one encoder and one submit, two plan-recreated
  runs, 3 x 4096 MiB traffic budget, iteration cap 200
- `wgpu-fft` `+/-` values are standard error across two runs; archived VkFFT
  `+/-` values are its reported population spread and are not the same statistic
- normalization disabled for accuracy and timed pairs
- `wgpu-fft` is out of place; the archived VkFFT reference is in place

```powershell
$env:WGPU_FFT_RUN_GPU_TESTS='1'; $env:WGPU_BACKEND='vulkan'
cargo test --test gpu_accuracy --release -- --nocapture

cargo bench --bench fft_bench -- shape 4096 --batch 32768 --adapter "RTX 5090" --runs 2 --iter-cap 200
cargo bench --bench fft_bench -- shape 3000 --batch 32768 --adapter "RTX 5090" --runs 2 --iter-cap 200
cargo bench --bench fft_bench -- shape 3256 --batch 32768 --adapter "RTX 5090" --runs 2 --iter-cap 200
```

## Accuracy

| Case | Max relative, before | Max relative, host-f64 LUT | Improvement | RMS relative, before | RMS relative, host-f64 LUT | Improvement |
|---|---:|---:|---:|---:|---:|---:|
| pow2 N=4096 | 1.322131678e-6 | **8.057843178e-8** | 16.41x | 1.343593662e-6 | **1.431400697e-7** | 9.39x |
| smooth N=3000 | 1.345612569e-6 | **5.365806660e-8** | 25.08x | 1.106654894e-6 | **1.267636983e-7** | 8.73x |
| Rader N=2999 | 1.322314778e-4 | **9.262729851e-8** | 1427.56x | 3.233542386e-4 | **1.957470729e-7** | 1651.90x |
| Bluestein N=3256 | 5.693303546e-4 | **6.234272195e-8** | 9132.27x | 8.602099702e-4 | **2.075463069e-7** | 4144.67x |
| batched ND 12x25, batch 2 | 1.637886119e-6 | **5.719177226e-8** | 28.64x | 1.269473199e-6 | **9.686874498e-8** | 13.11x |

Precision-sensitive inverse checks also pass: N=3000 max/RMS relative error is
`6.067223400e-8 / 1.274728861e-7`, Rader N=2999 is
`5.729889182e-8 / 1.938985334e-7`, and Bluestein N=3256 is
`9.096890614e-8 / 2.149297768e-7`.

The largest gains are in Rader and Bluestein because their previously f32
generated convolution kernels compounded phase error. The large-index
Bluestein chirp now reduces `i^2 mod 2N` in integer arithmetic before evaluating
the phase in f64.

## Performance

| N | Batch | Route | Previous `wgpu-fft` | Host-f64 LUT result | Change | Archived VkFFT | Current / VkFFT |
|---:|---:|---|---:|---:|---:|---:|---:|
| 4096 | 32768 | fused pow2 | 2.866408 +/- 0.025275 ms | **2.878008 +/- 0.031908 ms** | +0.405% | 2.859 +/- 0.034 ms | 1.0067x time |
| 3000 | 32768 | fused smooth | 2.615934 +/- 0.000322 ms | **2.099362 +/- 0.019244 ms** | -19.747% | 2.083 +/- 0.019 ms | 1.0079x time |
| 3256 | 32768 | Bluestein control | 63.993220 +/- 0.022113 ms | **64.374847 +/- 0.068480 ms** | +0.596% | 2.209 +/- 0.005 ms | 29.142x time |

No measured case regressed by 2%. The direct smooth kernel became faster than
the previous implementation because the exact LUT conversion also changed its
arbitrary-radix work mapping: one invocation now loads each butterfly input and
external twiddle once, then emits all radix outputs using f64-generated f32
radix constants. Multi-pass Stockham uses the same mapping, eliminating the
old chained twiddle powers while keeping the Bluestein control below the 2%
regression threshold.

Axis plans use one canonical N-entry read-only storage LUT per distinct axis
length. The large smooth twiddle-transpose path uses coarse and fine tables near
the square root of N, reconstructed with one complex multiply, so it does not
require an 8N-byte binding. Diagnostic stage summaries and pass counts are
unchanged; resident LUT allocation is additionally reported as
`helper:twiddle-luts-total` in diagnostic buffer requirements.
