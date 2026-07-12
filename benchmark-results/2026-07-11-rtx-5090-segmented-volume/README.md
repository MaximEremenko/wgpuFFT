# RTX 5090 segmented full-volume results - 2026-07-11

This archive starts from `e7b5659` and records the two implementation phases
for GPU-resident C2C volumes split across multiple plan-owned buffers. VkFFT was
not rerun; the relevant comparison is the same wgpu-fft shape with and without
internal sharding in one process.

## Phase A: segmented arena and mixed-radix execution

Phase A adds a typed segmented-volume schedule for rank >= 2 mixed-radix C2C
plans. The logical volume is copied into an exact-cap arena of STORAGE buffers,
each no larger than the active `maxBufferSize`. Axis 0 uses staged row bursts;
non-front axes use slab gather, transpose, row FFT, transpose-back, and scatter;
normalization is applied once in segment-local windows; and the final arena is
copied to the contiguous caller endpoint. Every binding and copy range is
validated through `WindowScheduler` and `StageExecutor`.

### Method

The RTX 5090 Vulkan adapter reported a 2,147,483,644-byte storage-binding limit
and a 1 TiB `maxBufferSize`, so a real above-buffer-limit allocation is neither
possible nor useful on this 32 GiB card. Validation instead used the existing
test policy override while preserving the real device limits for caller
endpoints.

The paired benchmark used `[4096, 8000]`, batch 1, exactly 262,144,000 bytes
(250 MiB). Both variants used a 16 MiB binding cap. The unsharded control used
a 1 GiB buffer cap and selected four-step; the sharded variant used a 64 MiB
buffer cap and allocated four arena segments. Forward and inverse plans were
recreated for every run, variant order alternated, and each sample recorded one
FFT+iFFT pair in one encoder and one submission. Timing is wall clock from
submit through device wait; command encoding is outside the timed interval.

```powershell
$env:WGPU_BACKEND='vulkan'
cargo bench --bench fft_bench -- shape 4096x8000 `
  --plan-max-bind-bytes 16777216 `
  --compare-max-buffer-bytes 1073741824,67108864 `
  --runs 2 --iter-cap 1 --adapter "RTX 5090" --wait-timeout-secs 240
```

### Correctness and routing

Forced-limit equivalence covers rank-2 `[100,128]`, rank-3 `[16,25,32]`, and
batched rank-3 `[8,10,16]` with batch 10, in both forward/no-normalization and
inverse/default-normalization directions. Every 102,400-byte case uses exact
arena segments `[32768,32768,32768,4096]` and matches both the normal GPU route
and the `Complex64` oracle. A cross-normalized limit case verifies that a
requested binding cap larger than the forced buffer cap still selects three
segments. Offset and caller-segmented endpoint diagnostics are explicitly
blocked. Rader and Bluestein axes remain structured-unsupported in Phase A.

The hardware-sized proof uses two impulses in `[4096,8000]`, forces four arena
segments `[67108864,67108864,67108864,60817408]`, and checks every `k0` on
output lines `k1 = 0, 1, 7999` against the analytic f64 DFT within `1e-3`
absolute complex error. Adapter evidence was NVIDIA GeForce RTX 5090, NVIDIA
driver 610.47, Vulkan backend.

All 234 unit tests and the `gpu_c2c`, `gpu_real`, `gpu_dispatch_split`,
`gpu_fused_pow2`, `gpu_accuracy`, `gpu_fused_prime`, `gpu_four_step`, and
`gpu_segmented_volume` release Vulkan suites passed. The established accuracy
suite remained between approximately `9.7e-8` and `2.15e-7` RMS-relative.

The normal public constructor cannot select this route on this adapter because
the u32 complex-element cap is far below its reported 1 TiB `maxBufferSize`.
Today the executable path is therefore reached when an internal/test policy cap
is below the real device limit. On devices where the logical volume exceeds the
real device buffer limit, caller-segmented endpoints are still required and are
deferred below.

### Performance

| Variant | Execution | Arena | Raw pair samples | Pair time | Traffic-equivalent passes / FFT | Effective GiB/s |
|---|---|---:|---:|---:|---:|---:|
| Unsharded | `out-of-core-four-step` | none | 220.7870, 217.9290 ms | **219.358 +/- 1.429 ms** | 12 | 53.423 |
| Sharded | `segmented-full-volume` | 4 x <= 64 MiB | 400.7408, 407.0685 ms | **403.90465 +/- 3.16385 ms** | 14 | 33.849 |

The measured sharded/unsharded ratio is **1.841303x**, or 84.130% overhead for
this initial depth-1 schedule. The exact segmented traffic model counts upload
and download once, axis-0 gather/FFT/scatter, and each non-front
gather/transpose/FFT/transpose/scatter stage. It does not multiply logical
traffic by the number of arena segments. `[4096,8000]` deliberately stresses
the per-row slab copy path; Phase B uses a lower-command-count cubic control
when comparing burst depths.

### Deferred after Phase A

- Burst staging-buffer depth 1-3, measured default selection, and structured
  allocation failure reporting are Phase B.
- Prime/Rader/Bluestein axes inside segmented volumes remain unsupported.
- Caller-segmented or strided endpoints combined with the internal arena, and
  caller-provided workspace reuse, remain deferred.
- No host or disk staging is used.
