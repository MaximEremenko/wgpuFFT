# RTX 5090 portable-df64 results - 2026-07-12

This archive records the three portable double-float slices: exact arithmetic
canaries, normal C2C execution, and prime/strided execution plus the short
same-session native-f64 comparison. VkFFT was not rerun; the reference values
below come from `../2026-07-12-rtx-5090-f64/`.

## Method

- GPU: NVIDIA GeForce RTX 5090 (`10de:2b85`), NVIDIA driver 610.47
- primary backend: Vulkan; portability check: DX12 driver `32.0.16.1047`
- wgpu `29.0.3`; device workgroup storage: 48 KiB
- df64 storage: `vec4<f32>` (`re_hi, re_lo, im_hi, im_lo`), 16 bytes/complex
- accuracy oracle: host `Complex64`; requested RMS-relative limit `1e-11`
- timing: FFT+iFFT pairs, normalization disabled, one encoder/submit per run,
  wall clock through device wait, two plan-recreated runs
- iteration rule: `min(200, floor(3 * 4096 MiB / bufferSize))`; both cases use
  a 1 GiB logical buffer and therefore record 12 pairs per run
- `--precision df64-f64` executes df64 then native f64 in one process on the
  same adapter/device with identical 16-byte topology and identical values
- wgpu-fft is out of place; archived VkFFT sample 1 is in place

```powershell
$env:WGPU_BACKEND='vulkan'
cargo bench --bench fft_bench -- shape 2048 --batch 32768 `
  --precision df64-f64 --adapter "RTX 5090" --runs 2 --iter-cap 200
cargo bench --bench fft_bench -- shape 4096 --batch 16384 `
  --precision df64-f64 --adapter "RTX 5090" --runs 2 --iter-cap 200
```

The two requested cases plus one warm N=2048 repeat completed well under five
minutes in total. No VkFFT command was executed.

## Correctness and routing

All measured df64 cases are below `1e-11` RMS-relative error. The worst result
is `3.7900e-14`, about seven decimal orders better than the established f32
errors near `1e-7`, while remaining above native-f64's few-`e-15` errors.

| Case | Selected path | RMS relative error |
|---|---|---:|
| N=2048 | fused pow2 | 5.0527e-15 |
| N=2048, forced 16 KiB | multipass pow2 | 6.5025e-15 |
| N=4096 inverse | multipass pow2 | 2.3799e-14 |
| N=3000 | fused smooth | 3.7900e-14 |
| Rader N=17 forward | staged prime | 1.0216e-15 |
| Rader N=67 inverse | fused prime | 4.7971e-15 |
| Bluestein N=34 inverse | staged prime | 6.2686e-15 |
| Bluestein N=68 forward | fused prime | 6.9398e-15 |
| ND `[3,17]`, batch 2, inverse | mixed + Rader AxisSequence | 2.6402e-15 |
| Strided ND `[8,15]`, batch 2, forward | pack + normal + unpack | 3.1525e-15 |
| Strided ND `[9,10]`, batch 2, inverse | pack + normal + unpack | 3.2399e-15 |

The exact arithmetic canary records 96 f32 words and passes on both Vulkan and
DX12 without requesting `SHADER_F64`. It covers TwoSum, QuickTwoSum,
mantissa-split Dekker `two_prod`, dd/complex operations, finite exponent-edge
products, and multiplication by a compile-time split `1/34` constant. The
DX12 end-to-end pass additionally executes fused-pow2 N=2, staged Rader N=17,
and a batched strided N=2 case; RMS errors range from `4.75e-16` to `1.02e-15`.
Metal remains untested and is the riskiest backend because WGSL has no
`precise`/no-contraction control and Metal commonly enables fast math.

Real transforms and all large/four-step/segmented df64 routes remain
structured `PrecisionUnsupported`; they do not silently fall back to f32.

## Performance

The warm N=2048 repeat is used in the table. It agreed with the initial ratio
(`3.012x` versus `3.083x`) while being collected after the N=4096 workload.

| N | Batch | Path / passes | df64 raw pair ms | df64 avg +/- stderr | same-session f64 avg +/- stderr | df64 / f64 | archived VkFFT f64 |
|---:|---:|---|---|---:|---:|---:|---:|
| 2048 | 32768 | fused / 1 | 32.4230, 16.1860 | **24.3045 +/- 8.1185 ms** | 8.0683 +/- 0.0652 ms | **3.012x** | 7.543 +/- 0.211 ms |
| 4096 | 16384 | Stockham / 4 | 59.2671, 42.2484 | **50.7577 +/- 8.5093 ms** | 19.0146 +/- 0.0640 ms | **2.669x** | 8.678 +/- 0.540 ms |

Modeled traffic bandwidth from the honest diagnostic pass counts is 164.6
GiB/s for df64 versus 495.8 GiB/s for f64 at N=2048, and 315.2 versus 841.5
GiB/s at N=4096. The same-session native-f64 controls are about 13.4% faster
than the older archived f64 run, which is why the verdict uses same-session
ratios rather than the archived controls.

Df64 shows a repeatable cold first-run penalty. Comparing only the second raw
runs reduces the ratios to about 1.99x (N=2048) and 2.21x (N=4096), but does not
change the conclusion: on this RTX 5090, portable df64 does **not** beat native
f64. The extra f32 instruction stream and source-level rounding barriers cost
more than the consumer FP64-rate disadvantage. The official headline remains
the complete two-run average: df64 is 3.01x slower at the fitting fused size
and 2.67x slower when both precisions use four global Stockham passes.

## Validation summary

- `cargo fmt --check` and `cargo test`: 272 unit tests pass
- Vulkan: all established GPU suites plus `gpu_df64` and the 96-word canary
  pass; the focused f32 fused-prime suite was rerun after restoring its original
  round-trip scale literal path
- DX12: `gpu_df64` (including staged Rader and strided execution) and the exact
  canary pass on the RTX 5090
- f32 regression accuracy remains approximately `9.69e-8` to `2.15e-7` RMS
