# RTX 5090 four-step large-route results - 2026-07-11

Phase A validation of the GPU-resident, out-of-binding-window C2C executor.
No VkFFT run was made; this slice is a memory-management route, not a VkFFT
comparison.

## Phase A: rank-2 mixed-radix

- GPU: NVIDIA GeForce RTX 5090 (`10de:2b85`), Vulkan, driver 610.47
- starting point: `f62f6b1` (`Fuse fitting prime FFT pipelines in one workgroup`)
- adapter limits: `maxStorageBufferBindingSize=2,147,483,644`,
  `maxBufferSize=1,099,511,627,776`, storage-offset alignment 32 bytes
- route: one full-volume plan scratch, bind-sized axis windows, compact stripe
  transpose staging, and a final chunked scale pass when normalization requires it
- data remains GPU-resident; there is no host or disk staging
- mixed-axis policy uses one staging/upload lane (burst depth 1), matching the
  effective JS default; the oversized volume is covered by two sequential
  windows per axis, while grouped and burst-window tuning remain deferred

```powershell
$env:WGPU_FFT_RUN_GPU_TESTS='1'
$env:WGPU_BACKEND='vulkan'
cargo test --test gpu_four_step --release -- --nocapture

cargo bench --bench fft_bench -- shape 4096x65536 --batch 1 --adapter "RTX 5090" --runs 2 --iter-cap 1 --wait-timeout-secs 240
```

### Correctness and routing

The forced-limit equivalence test uses shape `[15, 14]`, batch 3, and a
256-byte binding cap. Its 120- and 112-byte lines deliberately produce
unaligned window starts, exercising the GPU-copy fallback rather than only
aligned direct bindings. Forward/no-normalization and inverse/default-scale
outputs match both the normal route and the host `Complex64` oracle. Aligned
offset and genuinely segmented input/output views pass as well.

The real adapter-sized case is `[4096, 65536]`, batch 1: exactly 2 GiB, four
bytes larger than the adapter's storage-binding limit. It selected
`large-out-of-core` / `out-of-core-four-step`. Two complex impulses were
transformed and every `k0` value on output lines `k1 = 0, 1, 65535` matched the
analytic f64 DFT within `5e-4` absolute complex error.

Diagnostics report the full transpose scratch, axis/stripe staging buffers,
and every child Stockham workspace. A full volume above `maxBufferSize`
returns the exact segmented-full-volume unsupported diagnostic. Phase A
explicitly defers strided logical I/O and caller-workspace reuse; endpoint
buffers require input `COPY_SRC` and output `COPY_SRC | COPY_DST`, while
`STORAGE` enables direct window bindings.

### Performance

| Shape | Batch | Buffer | Runs / pairs per run | Pair time | Graph stages / FFT | Traffic-equivalent passes / FFT | Effective GiB/s | Four-logical-stage GiB/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 4096 x 65536 | 1 | 2 GiB | 2 / 1 | **369.658 +/- 4.432 ms** | 9 | 13 | 281.341 | 86.566 |

The exact diagnostic graph contains one fused N=4096 axis pass, six Stockham
passes for N=65536, and two logical stripe transposes: nine graph stages. Each
stripe transpose performs a full-volume gather, transpose kernel, and scatter,
so the harness weights it as three traffic-equivalent passes. The resulting
physical model is `4 * 13 * bufferBytes = 104 GiB` per timed FFT+iFFT pair, or
281.341 GiB/s. The requested four-step logical-stage view treats each axis FFT
as one stage plus the two transposes: `4 * 4 * bufferBytes / pairSeconds =
86.566 GiB/s`. Both are reported so the abstract model is not mistaken for the
actual child-kernel and stripe-staging traffic.

The two raw pair samples were 374.0896 ms and 365.2265 ms. Combined forward
and inverse diagnostics reported 25,770,655,744 bytes of plan-owned helper
allocations; this is an allocation inventory, not simultaneously transferred
traffic or caller workspace.
