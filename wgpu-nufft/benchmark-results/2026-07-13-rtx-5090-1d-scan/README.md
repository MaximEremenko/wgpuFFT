# wgpu-nufft hierarchical scan results - 2026-07-13

This follow-up replaces the serial type-1 bin-count prefix with a reusable
hierarchical GPU exclusive scan. At one million points, resident type-1
execution fell from 326.045 ms to 4.907 ms while preserving every output bit
in a direct old-path/new-path comparison.

## Comparison boundary

The measurement method is unchanged from the [Phase D archive](../2026-07-13-rtx-5090-1d/README.md):

- GPU time starts immediately before `queue.submit` and ends after waiting for
  that submission. Plan creation, command encoding, upload, readback, and
  caller-side synchronization before submit are excluded.
- Type 1 records one transform per submission. Type 2 records 32 transforms in
  one submission and reports the elapsed time per transform.
- Each result is three plan recreations, one excluded warmup per plan, and ten
  samples per plan. `+/-` is standard error across the three run means.
- FINUFFT values are archived CPU execute-only context from the prior Phase D
  session. FINUFFT was not rerun, and the ratios are not GPU-versus-CPU
  hardware claims.

## Implementation

Each 256-invocation workgroup performs a bank-padded Blelloch scan over 2,048
`u32` values in 8.25 KiB of workgroup memory. Block sums are scanned
recursively and reverse-order fixup passes add the block offsets. The type-1
route then writes the final `N+1` terminal offset in one invocation before its
unchanged deterministic scatter, per-bin point-index sort, gather, FFT, and
deconvolution stages.

The dispatch splitter remains private to `wgpu-nufft`. A trimmed local helper
was the deliberate choice: NUFFT needs its own structured error, and the scan's
padding guard must be workgroup-uniform because its shader contains barriers.
Publishing wgpu-fft's raw WGSL prologue would expose the wrong abstraction.
The private helper supports 3D dispatch grids, including the 65,536 scan blocks
required by a future `2^27`-bin case.

The scan adds 2,052 bytes and 8,196 bytes of persistent hierarchy storage to
the two measured plans. Per-execution type-1 transient scratch remains
`12*nf + 4*(M+1)` bytes: 7 MiB plus 4 bytes and 28 MiB plus 4 bytes.

## Results

All times are milliseconds per transform. `Change` compares with the archived
serial-scan GPU result. Type-2 drift is reported directly and is not used to
normalize the type-1 speedup.

| N=M | Kind | Serial-scan GPU | Hierarchical-scan GPU | Change | New Mpoints/s | Archived FINUFFT CPU execute | Same-host context |
|---:|---|---:|---:|---:|---:|---:|---|
| 262,144 | type 1 | 81.443230 +/- 0.796074 | **0.317203 +/- 0.007708** | **256.75x faster** | 826.423 | 5.246173 +/- 0.195396 | resident GPU 16.54x faster |
| 262,144 | type 2 control | 0.144004 +/- 0.002654 | **0.101967 +/- 0.000712** | 29.19% lower | 2570.874 | 3.233340 +/- 0.076052 | resident GPU 31.71x faster |
| 1,048,576 | type 1 | 326.045310 +/- 0.767455 | **4.907223 +/- 0.083107** | **66.44x faster** | 213.680 | 12.648807 +/- 0.346898 | resident GPU 2.58x faster |
| 1,048,576 | type 2 control | 0.383752 +/- 0.015428 | **0.243694 +/- 0.000605** | 36.50% lower | 4302.845 | 9.678457 +/- 0.290714 | resident GPU 39.72x faster |

The type-2 control moved downward rather than remaining within the prior
session's noise. This is recorded as cross-session drift and does not weaken
the no-regression conclusion; it is not applied as a correction to type 1.

Encoding time also fell from 0.252780 to 0.117840 ms at 262k and from 2.404173
to 0.668490 ms at 1M. Plan creation rose from 271.889 to 289.945 ms and from
311.878 to 337.611 ms respectively because the scan pipelines and persistent
hierarchy are now created with the reusable plan. Both are outside the
submit-to-wait headline.

## Correctness and portability

- Exact GPU scan tests matched a wrapping-`u32` CPU reference for lengths 0,
  1, 255, 256, 257, 65,535, `2^21`, `2^21+13`, block boundaries, all-zero,
  large-count, and randomized cases.
- A `4,194,317`-element case forced three scan levels and two reverse fixups.
- The exact scan suite passed on the RTX 5090 through both Vulkan and DX12.
- A temporary retained-serial probe ran both complete type-1 paths in one
  ordered command buffer for 3,072 fine bins and 8,193 mixed random,
  clustered, duplicate, and boundary points. All 12,288 output bytes were
  identical. The serial shader and probe were then deleted.
- The release `gpu_nufft` Vulkan oracle/adjoint matrix passed unchanged on the
  RTX 5090, NVIDIA 610.47. The worst type-1 relative L2 error at `eps=1e-6`
  remained `1.145e-6`; the worst adjoint residual remained `1.581e-8`.

## Environment and archive files

- GPU: NVIDIA GeForce RTX 5090 (`10de:2b85`, PCI `0000:01:00.0`)
- backend: native Windows Vulkan, NVIDIA 610.47, wgpu 29.0.3
- Rust: 1.95.0; Cargo: 1.95.0
- scan primitive commit: `2129837` (`Add hierarchical GPU prefix scan`)
- measured type-1 source SHA-256: `315B6D5AA1EEBE4426DE48D2DB8E4F652AECC93447C742A14CECD22EB090B722`
- unchanged benchmark harness SHA-256: `F6470BC60FFC9F29EBDC968D354ED6317082873875F3A34C67B32B84470DB50F`
- measured benchmark executable SHA-256: `6E1754418D614FA0CDDCD3F172CDD3267744ECF4233A677A323088A286CBA2D4`
- benchmark command wall time: about 7.7 seconds including a 1.73-second
  incremental release build, well below the five-minute cap

`wgpu-nufft-gpu-results.txt` records the run means and final result values.
`run.toml` records the exact command, method, limits, and source identities.
