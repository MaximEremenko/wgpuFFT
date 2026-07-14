# wgpu-nufft type-1 scratch reuse on RTX 5090 - 2026-07-13

The suspected per-execution fresh-scratch lifecycle cost was real for the 1D
million-point case. Reusing plan-owned binning scratch reduced the ordinary
type-1 submit-to-wait time from **4.280 ms to 0.806 ms (5.31x)**. The matched
cuFINUFFT `setpts+execute` span was 0.939 ms, so wgpu-nufft is 1.16x faster in
this same session. Type 2, which does not use the type-1 scratch, stayed within
1.1% at 1M.

The fix did **not** explain the 2D performance gap. Type 1 at 1024x1024 improved
only 1.05x and remains 15.88x slower than cuFINUFFT. This makes 2D per-stage
timestamp attribution the next required measurement; no spreading rewrite is
justified from the allocation result alone.

## Diagnosis confirmation

Before the fix, every type-1 encode created fresh bin-count, cursor, offset, and
sorted-point-index buffers. wgpu's initialization tracker may insert first-use
work for fresh storage buffers at submission, outside the user compute-pass
timestamps. The causal A/B changed those buffer lifetimes and bound the cached
sorted storage to the active `M` range; the existing per-execution count/cursor
clears remained unchanged:

| 1D N=M=1,048,576 type 1 | Before | After | Change |
|---|---:|---:|---:|
| Ordinary submit-to-wait | 4.280190 +/- 0.024651 ms | 0.806217 +/- 0.009243 ms | **5.309x faster** |
| Timestamped GPU envelope | 0.677083 ms | 0.690502 ms | 1.98% higher |
| Profiled submit-to-wait | 4.704900 ms | 0.798127 ms | 5.895x faster |
| Outside timestamp envelope | 4.027817 ms (85.6%) | 0.107624 ms (13.5%) | **37.43x smaller** |

The timestamped shader work remained approximately flat while the previously
unattributed span collapsed. This confirms fresh allocation, first-use
initialization/residency, and retirement as the causal lifecycle rather than
scan, gather-spread, bind-group encoding, or the fine-grid FFT. This A/B does
not isolate those driver/runtime subcomponents from one another.

The plan now owns fixed-size count, cursor, and offset buffers and a mutex-
protected sorted-index buffer that grows when `M` exceeds its capacity. Old
`wgpu::Buffer` handles remain valid when a later encode grows the cache. Counts
and cursors are cleared explicitly with `CommandEncoder::clear_buffer` on every
execution; offsets and sorted indices are fully overwritten by their producers.
The allocation device is the plan's construction device, so an invalid
encode-time device cannot contaminate the cache.

## Method and comparison boundary

- GPU: NVIDIA GeForce RTX 5090, native Vulkan, NVIDIA driver 610.47.
- Precision and semantics: one- and two-dimensional `f32`, `eps=1e-6`,
  `sigma=2.0`, positive sign, centered mode order.
- Cases: 1D `N=M` in `{262144, 1048576}` and 2D square shapes in
  `{512x512, 1024x1024}` with `M` equal to the total mode count.
- Three independently recreated plans, one excluded warmup per plan, and ten
  samples per plan. Type 1 records one transform per sample. Type 2 records 32
  transforms in one encoder and divides by 32.
- The ordinary wgpu headline starts immediately before `queue.submit` and ends
  after waiting for that exact submission. Plan construction, command encoding,
  uploads, and readback are excluded. The excluded warmup grows the reusable
  cache, matching steady-state reusable-plan execution.
- All CUDA arrays remain GPU-resident. For type 1, wgpu performs binning on
  every execution, so its submit-to-wait span is paired with cuFINUFFT
  `setpts+execute`. For type 2, wgpu's submit-to-wait span is paired with
  cuFINUFFT `execute` after `setpts`. Plan creation and transfers are excluded
  on both sides.
- The before and after wgpu controls and both CUDA dimensions were run in the
  same session. Only these rows feed the ratios below. No FINUFFT CPU, VkFFT, or
  archived timing was substituted.

## Before and after

All entries are milliseconds per transform. Speedup is `before / after`; values
below one indicate a slowdown.

| Shape / N=M | Kind | Before | After | Speedup | Change |
|---|---|---:|---:|---:|---:|
| 1D 262,144 | type 1 | 0.345080 +/- 0.018340 | 0.356560 +/- 0.014947 | 0.968x | +3.33% |
| 1D 262,144 | type 2 control | 0.103426 +/- 0.000674 | 0.106591 +/- 0.000719 | 0.970x | +3.06% |
| 1D 1,048,576 | type 1 | 4.280190 +/- 0.024651 | **0.806217 +/- 0.009243** | **5.309x** | **-81.16%** |
| 1D 1,048,576 | type 2 control | 0.252452 +/- 0.000673 | 0.255140 +/- 0.000813 | 0.989x | +1.06% |
| 2D 512x512 | type 1 | 2.269213 +/- 0.009888 | 2.326083 +/- 0.010425 | 0.976x | +2.51% |
| 2D 512x512 | type 2 control | 0.182303 +/- 0.000451 | 0.183138 +/- 0.000508 | 0.995x | +0.46% |
| 2D 1024x1024 | type 1 | 8.051293 +/- 0.242137 | 7.643110 +/- 0.010662 | 1.053x | -5.07% |
| 2D 1024x1024 | type 2 control | 0.645132 +/- 0.000401 | 0.649564 +/- 0.000397 | 0.993x | +0.69% |

At 262k the before/after variation is comparable across type 1 and the untouched
type-2 control, so it is treated as session noise. At 1M, the 81.2% type-1 drop
with a 1.1% type-2 change is the material result. The 2D 1024x1024 improvement
is only 5.1%, and 512x512 moved in the opposite direction.

## Same-session cuFINUFFT comparison

`wgpu / CUDA` above one means CUDA is faster. Type-1 CUDA is the combined
`setpts+execute` span; type-2 CUDA is `execute` only.

| Shape / N=M | Kind | wgpu-nufft | cuFINUFFT paired span | wgpu / CUDA | Verdict |
|---|---|---:|---:|---:|---|
| 1D 262,144 | type 1 | 0.356560 | 0.337973 | 1.055x | near parity; CUDA 5.5% faster |
| 1D 262,144 | type 2 | 0.106591 | 0.054031 | 1.973x | CUDA faster |
| 1D 1,048,576 | type 1 | **0.806217** | 0.939173 | **0.858x** | **wgpu 1.165x faster** |
| 1D 1,048,576 | type 2 | 0.255140 | 0.089983 | 2.835x | CUDA faster |
| 2D 512x512 | type 1 | 2.326083 | 0.271340 | 8.573x | CUDA faster |
| 2D 512x512 | type 2 | 0.183138 | 0.059951 | 3.055x | CUDA faster |
| 2D 1024x1024 | type 1 | 7.643110 | 0.481210 | 15.883x | CUDA faster |
| 2D 1024x1024 | type 2 | 0.649564 | 0.179344 | 3.622x | CUDA faster |

The allocation fix reverses the 1D million-point type-1 verdict: the portable
Vulkan implementation now wins the matched span in this session. It does not
materially change 2D. Relative to this session's pre-fix wgpu rows and the same
CUDA run, the 1024x1024 type-1 ratio shrinks only from 16.73x to 15.88x; the
512x512 ratio grows from 8.36x to 8.57x.

## Memory and first-use tradeoff

The cache deliberately trades retained plan memory for stable repeated
execution. Approximate retained type-1 binning-scratch capacity is 7 MiB and
28 MiB for the two 1D cases, and 13 MiB and 52 MiB for the two 2D cases. This
does not include the plan's fine-grid FFT or scan-hierarchy storage. Capacity is
released when the plan is dropped and never shrinks while the plan is alive.
The benchmark's excluded warmup pays initial allocation and initialization, so
these numbers describe steady-state plan reuse, not first execution or later
capacity growth.

## Correctness and cache boundaries

The full 1D and 2D oracle and adjoint matrices passed unchanged on RTX 5090
Vulkan. New regressions reuse one plan for a 257-point execution followed by a
different seven-point execution. The first execution crosses a workgroup and
grow-cache boundary and deliberately poisons reusable storage; the second
uses unrelated boundary/duplicate points and smaller amplitudes. Both outputs
match the `Complex64` direct oracle. The 2D large and small relative-L2 errors
were `8.485105397e-7` and `7.972229570e-7`, respectively.

The adversarial review found one cache-lifetime defect before landing: growth
originally used the encode-time device, so a failed wrong-device encode could
persist an unusable allocation. Growth now always uses the plan's stored device.
No other confirmed boundary or stale-data defects remained.

## Validation

- `cargo fmt --all -- --check`
- `cargo test --locked --workspace`
- `cargo test --locked -p wgpu-nufft --all-features`
- Strict clippy for `wgpu-nufft` with only the ten pre-existing profiling API
  `too_many_arguments` findings allowed; all other warnings are denied.
- RTX 5090 Vulkan release: complete `gpu_nufft` and `gpu_nufft_2d` suites,
  including the grow-then-shrink regressions.

Workspace-wide strict clippy remains blocked by pre-existing warnings in the
untouched `wgpu-fft` crate (80 findings under Rust 1.95). This slice introduces
no new clippy finding.

## Archive contents

- `wgpu-nufft-before-1d-results.txt` and
  `wgpu-nufft-after-1d-results.txt`: ordinary 1D controls.
- `wgpu-nufft-before-1d-profile-results.txt` and
  `wgpu-nufft-after-1d-profile-results.txt`: causal timestamp A/B.
- `wgpu-nufft-before-2d-results.txt` and
  `wgpu-nufft-after-2d-results.txt`: ordinary 2D controls.
- `cufinufft-1d-results.txt` and `cufinufft-2d-results.txt`: same-session CUDA
  spans, environment, smoke tests, and raw samples.
- `run.toml`: exact methodology, commands, versions, commits, and SHA-256
  provenance.
