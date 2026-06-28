# ND NUFFT prototype (stage 1: numpy reference)

Standalone research prototype for rank-generic N-dimensional NUFFT with
per-dimension "translation-rank placement" — NOT part of the wgpu-fft /
wgpu-nufft crates. Design and measured foundations: the vault note
`03 Projects/wgpuFFT/ND NUFFT Math Design.md`.

Each dimension independently chooses how to pay its translation rank r(eps):

| placement | mechanism | fine grid | per-dim cost |
|---|---|---|---|
| `SPREAD_S2` | ES-kernel spreading, sigma=2 | 2N | w(eps) grid touches |
| `SPREAD_S125` | ES-kernel spreading, sigma=1.25 | 1.25N | wider w, 1.6x less memory |
| `RT` | sigma=1 Chebyshev-interpolated perturbation factor (Ruiz-Antolin/Townsend-style) | N (none) | K(eps) scaled FFT passes |

RT multiplicity is multiplicative across RT dims (K per dim), so the intended
use is RT on 1-2 memory-critical dims and spreading elsewhere.

## Files

- `nd_nufft.py` — the reference implementation (types 1 and 2, any d, both
  isigns, f64). Kernel parameters follow the standard ES rules exactly; RT factors
  are built by barycentric Chebyshev interpolation in the sub-cell offset with
  adaptive K selection.
- `validate.py` — accuracy matrix vs a direct NDFT oracle + adjoint identity,
  d = 1..5, all placements and mixes, both isigns, clustered points.

## Stage-1 verdict (2026-07-17)

`python validate.py` — 22/22 PASS. Relative l2 errors track eps for every
placement mix (e.g. 4D s1.25^3+RT at eps=1e-6: t1 4.3e-6, t2 4.2e-6); adjoint
residuals are at machine precision (<= 7e-12), confirming the type-1/type-2
composition is exactly consistent. This empirically answers the design note's
open question 10.1: spread and RT placements compose without error inflation.

## Stage-2 verdict (2026-07-17)

`python cost.py` — analytic update counters match instrumented execution
exactly on every placement mix. Two corrections fed back into the design note:
(1) RT multiplicity K multiplies spreading work (per-point updates =
prod(w_spread) * prod(K_rt)), so per-point work is placement-invariant — the
"rank budget" is paid identically everywhere; placement chooses only where
memory and FFT passes go. (2) Interpolation-based K is ~10 at eps=1e-6 vs the
SVD floor of 7 (stage-3 optimization: SVD-derived factors).
Placement policy (`choose_placements`): sigma=2 for d<=3, sigma=1.25 for d>=4,
RT only under hard memory walls — it is exactly what fits 64^4 into a 256 MiB
browser budget (250 MiB); >=96^4 in browsers needs streaming/segmented
execution (wgpu-fft's large-route machinery, if adopted).

## Stage-3 milestone 1 (2026-07-17): rank-generic GPU spreading works

`gpu_experiment/` — standalone scratch crate (workspace-detached), generates the
S1 gather-form spreading kernel in WGSL for ANY dimension (d appears only as
unrolled generated code): one workgroup per output tile, one register
accumulator per owned cell, shared-memory point batches, per-dim ES weights
with periodic wrap, atomic-free. Host-side binning duplicates points into
touching tiles (scratch shortcut; production reads neighbor bins instead —
duplication 21x (d=3) to 200x (small d=4 grids)).

Vulkan results (`cargo run --release`, then `python check.py`):

| case | grid | median | throughput |
|---|---|---|---|
| d=3 256^3 s1.25, M=1e6, w=10 | 0.2 GiB | 8.99 ms | 1.11e11 upd/s |
| d=4 48^4 s1.25, M=5e5, w=10 | 0.1 GiB | 68.6 ms | 7.29e10 upd/s |
| d=4 64^4 s1.25, M=5e5, w=10 | 0.3 GiB | 73.2 ms | 6.83e10 upd/s |
| d=5 24^5 s1.25 eps=1e-3, M=2e5, w=5 | 0.2 GiB | 35.0 ms | 1.79e10 upd/s |

Correctness (d=4 exported case): rel-l2 = 1.11e-6 vs f64 numpy reference —
pure f32 accumulation error. Throughput at d=4 matches the class of the
hand-tuned production 2D bin-tile kernel (~6.7e10 upd/s), i.e. rank-generic
codegen costs essentially nothing. No GPU NUFFT above d=3 exists elsewhere.

## Next steps

3b. End-to-end d=4 type-2/type-1 on GPU (spread + FFT + deconvolve) vs the
    numpy reference; production-style neighbor-bin reads instead of host
    duplication.
4.  Adoption decision for wgpu-nufft.
