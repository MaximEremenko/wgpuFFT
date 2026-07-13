# wgpu-nufft 1D per-stage attribution on RTX 5090 - 2026-07-13

GPU timestamps reject one of the two starting hypotheses and narrow the other.
For type 2, the fine-grid FFT is the largest measured stage at both sizes and
accounts for 58.3% of the measured 262k-to-1M GPU-time increase. However, the
specific "deeper FFT" explanation was wrong: both fine grids use seven global
Stockham passes. For type 1, spreading is not the source of the 1M
superlinearity. The measured shader envelope is only 0.673 ms; 3.586 ms of the
4.259 ms profiled submit-to-wait span lies outside the timestamped passes.

## Method

- GPU: NVIDIA GeForce RTX 5090, native Vulkan, NVIDIA driver 610.47.
- Cases: one-dimensional `f32`, `N=M` in `{262144, 1048576}`, `eps=1e-6`,
  `sigma=2.0`, positive sign, centered mode order.
- Inputs and statistics are identical to the established 1D harness: three
  independently recreated plans, one excluded warmup per plan, ten samples per
  plan, and standard error across the three run means. Type 1 records one
  transform per sample; type 2 records 32 transforms in one encoder and divides
  by 32.
- The adapter and device were explicitly gated on
  `wgpu::Features::TIMESTAMP_QUERY`. The Vulkan timestamp period was 1 ns.
- Each recreated plan ran and discarded one complete profiled warmup, including
  query writes, resolve, copy, map, and decode.
- GPU intervals use `ComputePassTimestampWrites`. Type-1 boundaries cover
  clear/count, scan plus terminal offset, scatter, per-bin sort, gather-spread,
  the embedded wgpu-fft plan, and deconvolution. Type-2 boundaries cover
  predeconvolution, the embedded FFT, and interpolation. The adjacent intervals
  tile the reported GPU envelope exactly.
- Query resolve, query copy, map registration, command encoding, plan creation,
  uploads, and output readback are outside all GPU timestamp intervals. The
  diagnostic host span separately times `queue.submit` plus the exact
  submission's `device.poll`; it therefore includes query resolve/copy overhead
  and is not substituted for the ordinary control headline.
- The ordinary, feature-off control was rerun in the same session. Profiling is
  compiled out entirely unless the `gpu-profiling` feature is enabled.

The profiled and ordinary encoders were also compared on the GPU: type-1 and
type-2 outputs were bit-identical, every interval was monotonic and positive,
and the adapter log identifies Vulkan on the RTX 5090.

## Per-stage results

All entries are milliseconds per transform. Percentages are fractions of the
timestamped GPU envelope. `Host total` is the profiling-contaminated
submit-plus-poll span. `Outside envelope` is `Host total - GPU envelope`; it
contains driver/queue work and GPU commands outside the pass timestamps and is
not a named shader stage.

| N=M | Kind | Bin clear/count | Scan/terminal | Scatter | Sort | Gather/spread | Fine-grid FFT | Pre/deconvolve | Interpolate | GPU envelope | Host total | Outside envelope | Dominant measured work |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 262,144 | type 1 | 0.01298 (5.5%) | 0.02546 (10.7%) | 0.01331 (5.6%) | 0.00922 (3.9%) | **0.11081 (46.6%)** | 0.06113 (25.7%) | 0.00467 (2.0%) | - | 0.23757 | 0.35259 | 0.11502 (32.6% host) | gather/spread |
| 262,144 | type 2 | - | - | - | - | - | **0.06281 (70.3%)** | 0.00692 (7.7%) | 0.01961 (22.0%) | 0.08934 | 0.10729 | 0.01795 (16.7% host) | fine-grid FFT |
| 1,048,576 | type 1 | 0.02820 (4.2%) | 0.03588 (5.3%) | 0.03726 (5.5%) | 0.02154 (3.2%) | **0.39168 (58.2%)** | 0.14764 (21.9%) | 0.01041 (1.5%) | - | 0.67262 | 4.25909 | **3.58647 (84.2% host)** | outside timestamped passes |
| 1,048,576 | type 2 | - | - | - | - | - | **0.14937 (62.8%)** | 0.01920 (8.1%) | 0.06930 (29.1%) | 0.23788 | 0.25834 | 0.02046 (7.9% host) | fine-grid FFT |

The same-session feature-off controls were:

| N=M | Kind | Ordinary submit-to-wait | Minimum raw sample | Profiled host total |
|---:|---|---:|---:|---:|
| 262,144 | type 1 | 0.326683 +/- 0.011264 | 0.292400 | 0.352590 +/- 0.017503 |
| 262,144 | type 2 | 0.104545 +/- 0.001741 | 0.098225 | 0.107294 +/- 0.000205 |
| 1,048,576 | type 1 | 4.996113 +/- 0.562714 | 4.056000 | 4.259087 +/- 0.005362 |
| 1,048,576 | type 2 | 0.255607 +/- 0.002254 | 0.246919 | 0.258335 +/- 0.000508 |

The ordinary 1M type-1 run means were 4.324, 4.551, and 6.114 ms. The prescribed
mean retains that noisy third run; its 4.056 ms minimum agrees with the profiled
host span's 4.031 ms minimum. No archived cuFINUFFT number was rerun.

## Hypothesis verdicts

### Type 2: fine-grid FFT

The fine-grid FFT is the largest 1M type-2 stage (62.8%) and grows from 0.06281
to 0.14937 ms. Its 0.08656 ms increase is 58.3% of the total measured GPU
increase; interpolation contributes another 33.5%. This confirms that the FFT
is the first Stage-2 target.

The proposed reason was nevertheless incorrect. The 524,288-point route factors
as six radix-8 stages plus radix 2, while the 2,097,152-point route uses seven
radix-8 stages: both execute seven Stockham passes. The measured difference is
fourfold working-set/traffic scaling through the same pass count, not an extra
level. A line-internal two-step decomposition through fused sub-FFTs remains the
planned experiment because it can reduce global passes, but it must reproduce at
least a 1.5x FFT-stage improvement to land.

### Type 1: spreading side

The gather-spread shader grows 3.54x for four times as many points and adds only
0.281 ms. The full timestamped envelope adds 0.435 ms. In contrast, the profiled
host span adds 3.906 ms, of which 3.472 ms (88.9%) is outside the timestamped
passes. Hypothesis (b) is rejected: neither bin occupancy nor gather scaling
explains the observed superlinearity.

The strongest code-level suspect is the per-execution creation of the type-1 bin
counts, cursors, offsets, sorted-index buffers, bind groups, and their implicit
initialization/residency work. That is an inference, not a measured attribution:
GPU pass timestamps cannot see commands inserted before the first marker. In
accordance with the timebox, Stage 2 will not tune bin size or rewrite spreading
without evidence that those shaders dominate.

## Validation

- `cargo fmt --all -- --check`
- `cargo test --locked --workspace` (299 wgpu-fft unit tests plus all wgpu-nufft
  unit/integration/doc tests)
- `cargo test --locked -p wgpu-nufft --features gpu-profiling`
- RTX 5090 Vulkan release: `gpu_nufft` and `gpu_stage_profile`
- Both feature-off and feature-enabled all-target builds are warning-free.

## Archive files

- `wgpu-nufft-stage-profile-results.txt`: adapter/backend evidence, embedded FFT
  diagnostics, all raw stage/host samples, run means, and aggregates.
- `wgpu-nufft-control-results.txt`: same-session ordinary submit-to-wait control.
- `run.toml`: environment, commands, methodology, and artifact hashes.

