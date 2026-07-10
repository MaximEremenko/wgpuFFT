# RTX 5090 VkFFT reference results — 2026-07-10

These are the native Vulkan reference results used to baseline `wgpu-fft`.
The raw output files are preserved unmodified:

- [VkFFT sample 0](vkfft-sample-0.txt): 25 reported 1 GiB systems,
  aggregate score `243422`.
- [VkFFT sample 1000](vkfft-sample-1000.txt): every length from 2 through
  4096 (4,095 reported systems), aggregate score `305226`.

## Environment

- GPU: NVIDIA GeForce RTX 5090 (`10de:2b85`)
- NVIDIA driver: 610.47
- Vulkan device API: 1.4.341
- VkFFT commit: `066a17c17068c0f11c9298d848c2976c71fad1c1`
- VkFFT backend: `0` (Vulkan)
- Build: Release, MSVC/Ninja
- Vulkan SDK: 1.4.350.0
- `wgpu-fft` benchmark commit: `03c2b246d4c6c859bc0d026364f2082be6ae6638`

Commands:

```powershell
VkFFT_TestSuite.exe -d 0 -o vkfft_sample0_rtx5090.txt -vkfft 0
VkFFT_TestSuite.exe -d 0 -o vkfft_sample1000_rtx5090.txt -vkfft 1000
```

Sample 0 completed in 47.7 seconds. Sample 1000 completed in 7,674.1
seconds (2:07:54.1).

## Integrity

```text
F45CB8A7A5B156E50ED639504FD5442E02EC6B073E1254E31B0E2093F3F13361  vkfft-sample-0.txt
4470673258BC8B637129FDE41F4AAB2FDCAD846AC3C4CA854E155E9FF67B0530  vkfft-sample-1000.txt
```

VkFFT labels its three-run population standard deviation as `std_error`.
The `wgpu-fft` harness reports both true standard error and the compatible
population-spread value. VkFFT runs in place with inverse normalization off;
the current `wgpu-fft` C2C executor is out of place and matches the disabled
normalization for timing parity.

## Sample 1000 summary

- Aggregate score: `305226`
- Median per-size score: `348385`
- Geometric-mean score: `280174.7`
- Best normalized result: N=3256, score `377351`, 2.209 ms for 814 MiB
- Worst normalized result: N=3284, score `61520`, 13.665 ms
- N=4096: score `366724`, 2.859 ms, 1398.9 GiB/s

N=268 is a noisy outlier at `6.591 ± 6.222 ms`; use the raw file rather than
treating that row as a stable performance floor.

## Initial comparison

For N=4096, batch=32768, and a 1 GiB logical buffer:

| Implementation | FFT+iFFT pair | Score | Axis passes/uploads |
|---|---:|---:|---:|
| VkFFT sample 0 | 2.858 ms | 366855 | 1 inferred upload |
| `wgpu-fft` | 12.0085 ± 0.0827 ms | 87319 | 4 diagnostic passes |

`wgpu-fft` is 4.202× slower in raw pair time. Its four-pass estimated axis
traffic is 1332.4 GiB/s versus VkFFT's reported 1399.4 GiB/s, so the dominant
gap is the four unfused global-memory Stockham passes. The first optimization
target is a narrowly gated, one-workgroup fused N=4096 kernel with the current
four-pass implementation retained as the device-limit fallback.
