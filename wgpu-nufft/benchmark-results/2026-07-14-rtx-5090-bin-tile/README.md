# 2D NUFFT shared-memory bin tiles on RTX 5090 - 2026-07-14

The bounded `16x16` prototype is **accepted**. Against the retained global
gather in the same session, the timestamped type-1 gather-spread stage improved
by **7.73x at 512 squared** and **7.20x at 1024 squared**. A complete low-noise
ordinary repeat improved submit-to-wait type 1 by **4.87x** and **6.65x**. Both
results clear the decision record's 1.5x stage and 1.25x end-to-end gates by a
wide margin, so tiled spreading is now the public 2D type-1 default.

The same-silicon CUDA gap shrank from the prior 6.91x/17.75x to **1.46x/2.78x**
at 512 squared/1024 squared. CUDA parity was not the gate and has not been
claimed.

## Prototype design

One 256-invocation workgroup owns a `16x16` fine-grid tile. Each lane owns one
cell, accumulates in registers, and is the only writer for that cell. For the
width-7 ES kernel used at `eps=1e-6`, the workgroup builds a `23x23` periodic
bin halo and scans its 529 bin counts with a 1024-element bank-padded Blelloch
scan. The scan has uniform barriers and maps the existing bin offsets and
sorted point indices into one logical record stream for the tile.

Shared storage holds 154 records at a time: support starts, strengths, and the
seven X and seven Y weights for each point. If a halo contains more records,
all lanes execute another fixed-capacity batch; there is no fixed-occupancy
overflow path. Width 7 uses 15,312 bytes, below both the implementation's
15 KiB budget and WebGPU's 16 KiB portable workgroup-storage floor. The public
plan falls back to the original global gather when a fine-grid side is smaller
than the halo, 256 invocations are unavailable, or the shared-storage limit is
too small.

The old shader already formed a mathematically separable product of X and Y
weights. The actual optimization is that each point's `2w` weight evaluations
are hoisted into the shared cache and reused by the tile, instead of being
recomputed for every candidate point/cell pair. Binning, hierarchical scan,
deterministic per-bin sorting, FFT, and deconvolution are unchanged.

Batch traversal preserves the original halo-bin and sorted-point order. The
new path is therefore bit-identical to the global path on the tested Vulkan
backend; it does not merely pass an ulp tolerance.

## Method

- GPU: NVIDIA GeForce RTX 5090, Vulkan, NVIDIA driver 610.47.
- Data: `f32`/`complex64`, `eps=1e-6`, `sigma=2`, positive sign, centered
  mode order, `M=N0*N1`, and deterministic LCG seed `0x4E554646`.
- Fine grids: `1024x1024` for the 512-square case and `2048x2048` for the
  1024-square case.
- Statistics: three independently recreated plans, one excluded warmup per
  plan, ten retained samples per plan. `+/-` is standard error across the
  three run means; no samples were replaced by minima.
- Stage timings use `TIMESTAMP_QUERY` at adjacent compute-pass boundaries.
  The envelope and stages are GPU command intervals, not headline wall time.
- Ordinary wgpu timings start immediately before `queue.submit` and end after
  an exact-submission wait. Plan construction, encoding, uploads, readback,
  and output allocation are excluded.
- Type 1 records one transform per sample. Type 2 records 32 and divides the
  synchronized span by 32.

The deterministic uniform benchmark has a mean 23-by-23 halo population of
132.25 records. A host reconstruction gave approximately p99=160 and max=188,
so the formal timed workload includes tiles that cross the 154-record cache
capacity and execute a second batch. The dedicated dense test puts 513 points
in one bin and forces four batches.

## Stage result

Times are milliseconds per transform. Ratios are global divided by tiled.

| Shape | Interval | Global | Tiled16 | Speedup |
|---:|---|---:|---:|---:|
| 512 x 512 | gather-spread | 1.822942 +/- 0.018304 | 0.235783 +/- 0.006399 | **7.731x** |
| 512 x 512 | full GPU envelope | 1.941121 +/- 0.017946 | 0.353663 +/- 0.006490 | **5.489x** |
| 1024 x 1024 | gather-spread | 7.488068 +/- 0.290556 | 1.039770 +/- 0.187904 | **7.202x** |
| 1024 x 1024 | full GPU envelope | 7.808642 +/- 0.288807 | 1.422091 +/- 0.160425 | **5.491x** |

The complete type-1 timestamp means show that only gather changed materially:

| Stage | 512 global | 512 tiled | 1024 global | 1024 tiled |
|---|---:|---:|---:|---:|
| bin clear + count | 0.014260 | 0.014173 | 0.045773 | 0.114734 |
| scan + terminal | 0.028706 | 0.027648 | 0.050688 | 0.047924 |
| scatter | 0.013449 | 0.013449 | 0.037888 | 0.037888 |
| sort | 0.013278 | 0.013312 | 0.035840 | 0.035840 |
| gather-spread | 1.822942 | 0.235783 | 7.488068 | 1.039770 |
| fine-grid FFT | 0.043851 | 0.044655 | 0.139927 | 0.135601 |
| deconvolution | 0.004636 | 0.004643 | 0.010458 | 0.010334 |

The 1024 tiled count mean contains scheduling outliers in one timestamp run;
its shader and dispatch are identical between routes. They do not affect the
gather-stage verdict.

## Ordinary end-to-end result and controls

The prescribed first A/B pair is the formal stop-gate comparison. It contained
visible scheduling bands, so a second complete matrix was recorded. Every raw
sample from both matrices is archived.

| Matrix | Shape | Kind | Global | Tiled16 | Change |
|---|---:|---|---:|---:|---:|
| formal first | 512 x 512 | type 1 | 2.761983 +/- 0.677218 | 0.421290 +/- 0.009748 | **6.556x faster** |
| formal first | 1024 x 1024 | type 1 | 8.257687 +/- 0.699486 | 2.211197 +/- 0.105707 | **3.734x faster** |
| formal first | 512 x 512 | type 2 control | 0.200468 +/- 0.009963 | 0.193904 +/- 0.009336 | 3.274% faster |
| formal first | 1024 x 1024 | type 2 control | 0.749873 +/- 0.005885 | 0.746833 +/- 0.011039 | 0.405% faster |
| complete repeat | 512 x 512 | type 1 | 2.061307 +/- 0.015672 | 0.423360 +/- 0.009084 | **4.869x faster** |
| complete repeat | 1024 x 1024 | type 1 | 8.516020 +/- 0.470584 | 1.279647 +/- 0.004024 | **6.655x faster** |
| complete repeat | 512 x 512 | type 2 control | 0.204406 +/- 0.023204 | 0.209606 +/- 0.022425 | 2.544% slower |
| complete repeat | 1024 x 1024 | type 2 control | 0.748710 +/- 0.013717 | 0.747121 +/- 0.003938 | 0.212% faster |

The formal type-2 controls clear the 2% gate. The repeat's 512-square point
estimate is 2.54% slower, but its standard errors overlap broadly and neither
the type-2 source nor pipeline changed. This is reported as noise, not hidden
or normalized. The 1D implementation and routing are also untouched, and its
GPU oracle/adjoint suite remained green; no cross-session performance ratio is
inferred for it.

## Stop-gate verdict

| Gate | Required | Measured | Verdict |
|---|---:|---:|---|
| 512 gather stage | at least 1.5x | 7.731x | pass |
| 1024 gather stage | at least 1.5x | 7.202x | pass |
| 512 ordinary type 1 | at least 1.25x | 6.556x formal; 4.869x repeat | pass |
| 1024 ordinary type 1 | at least 1.25x | 3.734x formal; 6.655x repeat | pass |
| oracle/adjoint | unchanged | full matrix green | pass |
| global/tiled output | bit-identical | all dedicated cases exact | pass |
| overflow batching | no worse than global | timed max halo 188 > 154 and 7.2-7.7x gather win; dense 513 exact | pass |
| type-2 control | no more than 2% regression | formal: -3.274% / -0.405% | pass |

The runtime prototype is retained. No second tile geometry was explored.

## Correctness

The public tiled route passed the existing type-1/type-2 oracle matrix across
`eps=1e-2..1e-6`, random, clustered, boundary, and duplicate points, both
signs and both mode orders. Dedicated tiled/global canaries additionally cover
seeded 257-point input, a 513-point same-bin multi-batch input, and non-square
`256x1024` execution.

- Every dedicated global/tiled `f32` word matched bit-for-bit.
- Worst dedicated tiled type-1 relative L2 at `eps=1e-6`: `1.192100e-6`.
- Dedicated tiled adjoint residual range: `3.77e-9..7.64e-9`.
- Non-square type-1 relative L2 at `eps=1e-5`: `9.889337e-6`.
- Back-to-back 513-point then 7-point execution on one plan remained exact
  versus global; oracle errors were `1.069804e-6` and `1.128199e-6`.

The release test log identifies `NVIDIA GeForce RTX 5090`, Vulkan, driver
610.47. The timestamp canary also verified positive intervals and exact
ordinary/profiled output equality.

## Same-session cuFINUFFT comparison

The CUDA rerun used cuFINUFFT 2.5.1, CuPy 14.1.1, CUDA 12.8 runtime, and the
same RTX 5090. Inputs and outputs stayed resident on the GPU. The honest span
pairing remains:

- type 1: wgpu submit-to-wait versus cuFINUFFT `setpts+execute`;
- type 2: wgpu submit-to-wait versus cuFINUFFT execute-only after `setpts`.

The cleaner complete-repeat wgpu matrix is paired below. CUDA plan and the
unpaired spans remain in the raw log.

| Shape | Kind | wgpu | cu setpts | cu execute | cu paired | wgpu/CUDA |
|---:|---|---:|---:|---:|---:|---:|
| 512 x 512 | type 1 | 0.423360 | 0.197327 | 0.140270 | 0.289140 | **1.464x** |
| 512 x 512 | type 2 | 0.209606 | 0.072196 | 0.065670 | 0.065670 | **3.192x** |
| 1024 x 1024 | type 1 | 1.279647 | 0.204153 | 0.264000 | 0.459983 | **2.782x** |
| 1024 x 1024 | type 2 | 0.747121 | 0.064230 | 0.178244 | 0.178244 | **4.192x** |

The bounded portable spreading optimization is therefore worthwhile. It
collapses most of the type-1 gap without float atomics or CUDA-specific
machinery. The remaining 1024-square type-1 gap is 2.78x, while type 2 remains
an independent interpolation/ND-FFT problem and was not changed here.

## Type-2 and dimensional applicability

Applying the same tiling to type 2 is not trivial. Its current interpolation
kernel is already point-driven and computes the X/Y weights once per output
point. A shared tile would first require binning or sorting points and then
restoring caller output order, so no speculative second runtime commit was
made.

The 1D type-1 gather already has one output cell per invocation and far less
tensor-product reuse, so this exact 2D tile is not directly useful there. A 3D
version can reuse the bin/scan/batch idea, but the halo volume, workgroup shape,
and shared-weight footprint grow sharply; it needs its own bounded geometry
decision rather than mechanically extending `16x16`.

No VkFFT or FINUFFT-CPU run was performed.

## Validation

- `cargo fmt --all -- --check`
- `cargo test --locked --workspace`
- `cargo test --locked -p wgpu-nufft --all-features`
- `cargo clippy --locked -p wgpu-nufft --all-targets --no-deps -- -D warnings`
- RTX 5090 Vulkan release `gpu_nufft`, `gpu_nufft_2d`, and
  `gpu_stage_profile`; the final feature-on `gpu_nufft_2d` rerun passed after
  the review fixes.

All-feature strict clippy still exposes the pre-existing profiling API
`too_many_arguments` signatures; workspace-wide strict clippy also exposes
pre-existing Rust 1.95 warnings in untouched `wgpu-fft`. The scoped default
command above is clean, and this slice adds no unwaived warning.

Implementation commit: `70659a5` (`Tile 2D NUFFT spreading in shared memory`).

## Archive contents

- `stage-{512,1024}-{global,tiled}.txt`: adapter evidence, every timestamp
  sample, stage means, and host spans.
- `ordinary-first-{global,tiled}.txt`: formal stop-gate A/B matrix.
- `ordinary-repeat-{global,tiled}.txt`: complete low-noise repeat used in the
  CUDA table.
- `cufinufft-2d.txt`: environment, correctness smoke, and all CUDA spans.
- `gpu-nufft-2d.txt`: dedicated tiled correctness matrix and adapter evidence.
- `gpu-stage-profile.txt`: bit-identical profiling canary and raw intervals.
- `run.toml`: exact commands, method, limits, and SHA-256 provenance.
