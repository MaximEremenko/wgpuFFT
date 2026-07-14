# RTX 5090 long single-line FFT trial

Date: 2026-07-13 (America/New_York)

This is the Stage-2 timebox following the per-stage attribution in
`../2026-07-13-rtx-5090-stage-profile/`. It records a correct but rejected
optimization experiment; no candidate runtime code was retained.

## Candidate

The 2,097,152-point f32 C2C transform was split as `2048 x 1024`:

1. 2,048 fused 1,024-point FFTs over strided lines;
2. one host-f64-LUT twiddle transpose;
3. 1,024 fused 2,048-point FFTs over strided lines.

This reduced the diagnostic traffic-equivalent kernel count from seven
Stockham passes to three passes and retained one full-size workspace. A
temporary opt-in GPU equivalence test passed on the RTX 5090 for forward,
inverse, and normalized roundtrip execution against the seven-pass route.

## Method

- Adapter: NVIDIA GeForce RTX 5090, Vulkan, driver 610.47.
- wgpu timestamp period: 1 ns.
- Precision: f32; NUFFT `eps=1e-6`, `sigma=2`, positive sign, centered modes.
- Case: `N=M=1,048,576`, whose fine grid is 2,097,152 complex values.
- The existing timestamp profiler was run immediately before and after removing
  the candidate in the same machine session.
- One recreated plan, one excluded warmup, three measured samples; type 1 used
  one transform per sample and type 2 used 32.
- The acceptance gate was at least 1.5x on the measured fine-grid FFT stage.

Command:

```powershell
$env:WGPU_FFT_RUN_GPU_TESTS='1'
$env:WGPU_BACKEND='vulkan'
cargo bench --locked -p wgpu-nufft --features gpu-profiling --bench nufft_stage_profile -- --adapter "RTX 5090" --runs 1 --samples 3
```

## Results

All times are timestamped GPU milliseconds per transform. Speedup is baseline
divided by candidate.

| NUFFT direction | Stage | Seven-pass baseline | Three-pass candidate | Speedup |
|---|---|---:|---:|---:|
| Type 1 | Fine-grid FFT | 0.146763 | 0.135595 | **1.082x** |
| Type 1 | Full GPU envelope | 0.650859 | 0.634443 | 1.026x |
| Type 2 | Fine-grid FFT | 0.142410 | 0.134125 | **1.062x** |
| Type 2 | Full GPU envelope | 0.237141 | 0.222753 | 1.065x |

The 262,144-point control did not select the candidate. Its type-1/type-2 FFT
stage means were 0.062155/0.063050 ms in the candidate build and
0.060853/0.060973 ms after removal, consistent with noise and warm-state drift.

## Verdict

The candidate missed the required stage speedup by a wide margin and was
removed. Both fused child transforms access long strided lines, so the result is
consistent with non-coalesced global loads and stores consuming most of the
theoretical `7/3` traffic win. The child passes were not timestamped separately,
so this is a design-backed inference rather than a per-pass attribution.

A future long-line FFT should begin with a coalescing transpose and use a fused
child kernel capable of a transposed final store, or otherwise adopt a tiled
four/six-step layout. That is a distinct kernel project, not a small reuse of
the existing fused-axis machinery, so it is deferred rather than improvised in
this timebox.

No cuFINUFFT matrix was rerun: the prescribed rerun was conditional on a fix
passing the 1.5x gate, and no code change qualified to land.
