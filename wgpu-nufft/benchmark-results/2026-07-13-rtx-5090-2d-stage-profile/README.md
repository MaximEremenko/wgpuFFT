# wgpu-nufft 2D per-stage attribution on RTX 5090 - 2026-07-13

The remaining 2D gaps are not caused by allocation or the embedded FFT. At
`1024x1024`, type-1 gather-spread consumes **7.598 ms, 95.9%** of the timestamped
pipeline. Type-2 interpolation consumes **0.460 ms, 72.6%**; its complete
fine-grid ND FFT is only 0.141 ms. In both directions, work outside the
timestamp envelope is about 2% of the profiled host span.

This establishes two separate follow-ups. A bounded shared-memory tiled-gather
prototype is worthwhile for type 1, where one stage is overwhelmingly dominant.
It is not expected to reach CUDA parity by itself: even a 4x gather-stage win
projects to a 2.390 ms profiled host span, still 4.97x cuFINUFFT's paired
0.481 ms span. Type 2 needs interpolation-kernel work, not a return to the
rejected long-line FFT trial.

## Method

- GPU: NVIDIA GeForce RTX 5090, native Vulkan, NVIDIA driver 610.47.
- Case: 2D `f32`, mode shape `1024x1024`, `M=1,048,576`, fine-grid shape
  `2048x2048`, `eps=1e-6`, `sigma=2.0`, positive sign, centered mode order,
  axis zero fastest, point-major `[x,y]` coordinates.
- Inputs use the established field-specific u32 LCG streams and the same X/Y
  seeds as the ordinary 2D and cuFINUFFT harnesses.
- Three independently recreated plans, one excluded warmup per plan, and ten
  samples per plan. Type 1 records one transform per sample; type 2 records 32
  transforms in one encoder and divides every interval by 32.
- The adapter and requested device were explicitly gated on
  `wgpu::Features::TIMESTAMP_QUERY`; the Vulkan timestamp period was 1 ns.
- Adjacent `ComputePassTimestampWrites` intervals tile the pipeline envelope.
  Type 1 covers clear/count, hierarchical scan plus terminal offset, scatter,
  per-bin sort, gather-spread, the complete embedded wgpu-fft `execute_views`,
  and deconvolution. Type 2 covers predeconvolution/zero-fill, the complete ND
  FFT, and interpolation.
- These are command-envelope intervals, not isolated shader ALU timings.
  Clear/count includes two `clear_buffer` commands. FFT intervals include every
  child FFT command and adjacent inter-pass gaps. Component sums equal the
  first-to-last timestamp envelope algebraically.
- Query resolve/copy/map, command encoding, plan construction, uploads, and
  output readback are outside every decoded interval. A separate diagnostic
  host span times `queue.submit` plus the exact submission's `device.poll`; it
  includes profiling overhead and is not substituted for the ordinary control.
- The feature-off 2D control was rerun immediately before the profiler. No
  VkFFT or FINUFFT CPU run was performed. The cuFINUFFT values below are the
  fresh same-slice GPU-resident rows archived with the scratch-reuse comparison.

## Per-stage results

All values are milliseconds per transform. Percentages are shares of the
timestamped pipeline envelope.

| Kind | Clear/count | Scan/terminal | Scatter | Sort | Gather/spread | Fine-grid ND FFT | Pre/deconvolve | Interpolate | GPU envelope | Profiled host total | Outside envelope |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| type 1 | 0.038270 (0.48%) | 0.057822 (0.73%) | 0.037888 (0.48%) | 0.035840 (0.45%) | **7.597909 (95.92%)** | 0.143331 (1.81%) | 0.010439 (0.13%) | - | 7.921500 | 8.087997 | 0.166497 (2.06% host) |
| type 2 | - | - | - | - | - | 0.140710 (22.18%) | 0.033422 (5.27%) | **0.460395 (72.56%)** | 0.634528 | 0.649357 | 0.014828 (2.28% host) |

The ordinary feature-off controls were:

| Kind | Ordinary submit-to-wait | Profiled host total | Profiling uplift |
|---|---:|---:|---:|
| type 1 | 7.803810 +/- 0.134258 | 8.087997 +/- 0.157200 | 3.64% |
| type 2 | 0.644089 +/- 0.000477 | 0.649357 +/- 0.000422 | 0.82% |

The type-1 timestamp run has two visible performance bands, both entirely in
gather-spread; retaining all prescribed run means is therefore important. The
other stages remain stable, and the conclusion is insensitive to choosing the
mean or the minimum: gather-spread is still more than 95% of the measured GPU
pipeline.

## Same-silicon context after scratch reuse

The CUDA values are not newly rerun here; they are the same-session rows in
`../2026-07-13-rtx-5090-scratch-reuse/`. Pairing remains honest: type 1 compares
wgpu submit-to-wait with cuFINUFFT `setpts+execute`, while type 2 compares with
cuFINUFFT `execute` after `setpts`.

| Kind | Current wgpu control | cuFINUFFT paired span | wgpu / CUDA | Dominant remaining stage |
|---|---:|---:|---:|---|
| type 1 | 7.803810 | 0.481210 | 16.22x | gather-spread |
| type 2 | 0.644089 | 0.179344 | 3.59x | interpolation |

The scratch archive's prescribed wgpu means give 15.88x and 3.62x; the small
difference is session drift, not an optimization. Stage attribution now rules
out both the embedded ND FFT and unattributed host work as explanations for the
type-1 gap. It also reverses the working assumption for type 2: in 2D,
interpolation—not the fine-grid FFT—is the main cost.

## Profiling correctness canary

The opt-in GPU regression now covers both dimensions. The new 2D cases use a
non-square `32x48` mode shape and `M=1021`, then encode ordinary and profiled
type-1/type-2 executions into the same command buffer. Output words are compared
as raw `u32` values and are bit-identical. Every stage delta is positive and in
the expected seven-stage/type-1 or three-stage/type-2 order.

The 2D cases begin at nonzero query indices 3 and 5, so resolution and decoding
also prove that layouts are relative to their requested query range rather than
accidentally hard-coded to zero. The raw log identifies the RTX 5090 Vulkan
adapter and driver 610.47.

## Decision record: shared-memory type-1 gather

**Verdict: worth one timeboxed portable prototype, but not an open-ended CUDA-
parity effort.** Gather-spread is sufficiently dominant that a successful
prototype would materially improve end-to-end type 1. The measured non-gather
floor also shows that this design alone is unlikely to bring wgpu within 2x of
cuFINUFFT.

### Proposed kernel

1. Dispatch one 256-invocation workgroup per `16x16` fine-grid core tile using
   the existing 3D split-dispatch linearization. One lane owns one core cell;
   inactive tail lanes remain in uniform control flow and never return before a
   workgroup barrier.
2. For the current width-7 ES kernel, `R=ceil(width/2)=4` and the gather visits
   an `8x8` bin neighborhood. A workgroup stages the tile's periodic
   `(16 + 2R - 1)^2 = 23x23 = 529` one-cell bins. Fine-grid dimensions smaller
   than 23 use the existing global gather to avoid duplicated wrapped halo bins.
3. Lanes cooperatively load halo-bin counts and perform a Blelloch exclusive
   scan in a 1024-entry `var<workgroup>` u32 array. If the halo contains at most
   416 points, lanes load each point's folded two-word X/Y position and complex
   strength once into shared arrays. Each lane accumulates its owned cell through
   a 256-entry shared complex tile, then is the single writer that copies that
   cell to global memory. Declared shared storage is 4,096 bytes of prefix data,
   6,656 bytes of positions, 3,328 bytes of strengths, and 2,048 bytes of cell
   accumulators: 16,128 bytes, 256 bytes below the portable 16 KiB limit. Kernel
   parameters and loop scalars remain uniform/private; reduce the point cap if an
   implementation needs any additional workgroup variables. Every Blelloch
   barrier is unconditional and outside lane/index guards; tail lanes cannot
   return early.
4. Every owner lane traverses its original Y-major/X-major bin sequence and the
   existing ascending point-index order within each bin, reading the shared
   cache and executing the unchanged support test and tensor ES weights. It
   writes its cell once. There is no overlapping output ownership and therefore
   no float atomic.
5. If the 529-bin halo exceeds 416 points, the whole workgroup uniformly takes
   the current global-gather path for that tile. This preserves arbitrary
   clustered-point correctness without a divergent-barrier or unbounded-memory
   case. Uniform benchmark density is only `M/(F0*F1)=0.25`, so a tile expects
   about 132 halo points; 416 is roughly 3.15x headroom. This deliberately leaves
   a performance threshold for dense clusters, but the overflow route is exactly
   the current baseline rather than a new pathological algorithm.

WGSL validation cannot be expected to prove that `halo_total`, read from
workgroup memory, is uniform. All lanes must therefore execute fixed cooperative
cache-load slots (guarding stores only), reach an unconditional barrier outside
the capacity guard, and only then branch to cached or global gather; neither
branch may contain another workgroup barrier.

Because the entire halo is resident before accumulation, the owner can retain
the current per-cell addition order. A future implementation must require a
bitwise old-vs-tiled canary for uniform, boundary, duplicate, clustered,
overflow-fallback, and tail-tile cases. If caching folded positions changes bits
on any backend, cache raw coordinates instead and recompute folding per owner.

### Expected win and stop gates

The current shader examines approximately `256 * 64 * 0.25 = 4096` candidate
point records per tile. Shared staging loads about `529 * 0.25 = 132` unique
records. That is roughly 31x fewer global point/index fetches; accounting for
the current support-filtered strength loads gives about 24x fewer strength
fetches. ES `sqrt`/`exp` work and support tests remain, so the credible estimate
is only **2-4x on the gather stage**, not 24-31x.

| Gather-stage speedup | Projected timestamped pipeline | Projected profiled host span | Host-span speedup | Remaining ratio to 0.481210 ms CUDA |
|---:|---:|---:|---:|---:|
| 2x | 4.122545 ms | 4.289042 ms | 1.89x | 8.91x |
| 4x | 2.223068 ms | 2.389565 ms | 3.38x | 4.97x |

Reaching a 2x CUDA gap (`<=0.962420 ms`) would require an 11.9x gather-stage
speedup even if only the timestamped GPU floor is counted, or 16.1x when the
profiled host/outside floor is retained. That is well beyond the conservative
projection.

Prototype gates are therefore explicit: continue only if the first `16x16`
implementation improves gather-spread by at least 1.5x and ordinary end-to-end
type 1 by at least 1.25x, keeps oracle/adjoint and bitwise canaries green, keeps
overflow-tile performance no worse than the current global-gather baseline, and
regresses 1D/type-2 controls by no more than 2%.
Stop after that one geometry if it misses the gates; do not tune tiles in this
slice. Type-2 interpolation remains a separate future kernel-math/data-reuse
investigation.

## Validation

- `cargo fmt --all -- --check`
- Workspace-wide tests and all-feature wgpu-nufft tests.
- Strict wgpu-nufft clippy with all warnings denied except the pre-existing
  profiling API `too_many_arguments` signatures.
- RTX 5090 Vulkan release `gpu_nufft` and `gpu_nufft_2d`: the complete oracle,
  adversarial-point, grow/shrink scratch, and adjoint matrices remain green.
- RTX 5090 Vulkan release `gpu_stage_profile`: ordinary/profiled bit equality,
  positive intervals, nonzero query offsets, and explicit adapter evidence.

Workspace-wide strict clippy remains blocked by pre-existing warnings in the
untouched `wgpu-fft` crate under Rust 1.95; this change introduces no new lint.

## Archive contents

- `wgpu-nufft-2d-stage-profile-results.txt`: adapter, FFT route diagnostics,
  raw timestamp samples, stage means, and profiled host spans.
- `wgpu-nufft-2d-control-results.txt`: same-session feature-off controls.
- `gpu-stage-profile-results.txt`: bit-identical 1D/2D profiling canary with
  Vulkan adapter evidence.
- `run.toml`: exact method, commands, versions, and SHA-256 provenance.
