# Changelog

## 0.1.0 (unreleased)

First tagged release.

### Added

- Out-of-place C2C transforms in `f32`, native `f64` (Vulkan `SHADER_F64`), and
  portable double-float `df64`, over 1D/ND shapes, axis subsets, and batches.
  Routes cover mixed radix (2, 3, 4, 5, 7, 8, 11, 13), Rader, Bluestein, and
  fused single-workgroup kernels.
- R2C/C2R `f32` transforms with the WebGPU-FFT packed-spectrum convention.
- Large routes beyond one storage binding: batch chunking, smooth and axis
  decomposition, GPU-resident four-step, and segmented full-volume execution.
- Offset, segmented, and strided logical I/O views, caller-owned workspaces,
  structured diagnostics and errors, validated per-plan tuning, and
  pipeline-cache snapshots (versioned JSON with the `serde` feature).
- WebAssembly support on the browser WebGPU backend.
- `CpuFftPlan`, a native CPU backend behind the default `cpu` feature, with the
  same configuration and buffer layouts, built on `rustfft` and `realfft`.
- `FftRecorder`, which records several executions into one shared compute pass
  (`FftPlan::record`, `record_views`, `record_logical_views`).
- `WGPU_ADAPTER_NAME=<text>` selects the adapter of
  `device::request_default_device` (used by the GPU tests and examples) by
  name, for machines with several GPUs.
- `examples/quickstart.rs`, and CI for formatting, clippy, tests on Linux,
  Windows, and macOS, the wasm32 build, docs, and the 1.92 MSRV.

### Changed

- Requires `wgpu` 30.
- `FftTuning::rader_max_prime` defaults to 8192 instead of 4096. Rader axes
  whose own convolution would be longer than Bluestein's still run
  Bluestein's register kernel, so above 4096 only primes with a cyclic
  convolution in workgroup memory change kernels.

### Performance

- A plan execution records all its kernels in one compute pass instead of one
  pass per kernel.
- Fused kernels transform several lines per workgroup, and strided axes load
  and store element-major so neighbouring invocations touch neighbouring lines.
- Prime `f32` axes up to 127 (`FftTuning::direct_max_prime`) run one direct
  DFT kernel per axis instead of Rader's convolution: a 17x17 FFT+iFFT pair
  takes 8.5 µs instead of 49 µs, and N=17 batched over 1 GiB 2.8 ms instead of
  36 ms.
- Bluestein axes whose convolution does not fit workgroup memory, and Rader
  primes in the same position, run as one register-resident kernel for
  convolutions up to 16384 points (`f32`): over 1 GiB, N=3256 takes 7.4 ms
  instead of 79.5 ms and N=4093 6.5 ms instead of 84 ms;
  6841x6841 takes 7.5 ms instead of 30 ms. Lines of 16384
  points keep 32 values per invocation in 512 invocations.
- Rader's convolution is cyclic over `N - 1` points when that length is
  smooth, halving its FFTs: over 1 GiB, N=1009 takes 3.8 ms instead of 6.4 ms
  and N=2003 6.4 ms instead of 11.8 ms.
- The fused Rader kernel loads and stores lines coalesced and applies the
  permutation in workgroup memory: over 1 GiB, N=2003 takes 3.7 ms instead of
  6.4 ms. On strided axes too long for several lines per workgroup, it loads
  and stores two or four lines together and convolves them in turn:
  1229x1229 takes 131 µs instead of 150 µs.
- Strided power-of-two `f32` axes interleave up to eight lines per workgroup
  in registers when workgroup memory holds fewer, so their loads coalesce:
  2048x2048 takes 114 µs instead of 179 µs and 4096x4096 0.75 ms instead of
  1.07 ms.
- When every stage of an axis plan is a fused kernel, the stages after the
  first run in place on the output, so such plans need no workspace and
  cache-resident volumes stay in cache: 256x256x128 takes 348 µs instead of
  430 µs.
- Power-of-two `f32` lines of 16 to 512 points (64 when contiguous), and
  contiguous ones from 1024 points, run in register-resident kernels, which
  load straight into registers and pass through workgroup memory once per
  radix-16 stage: 128x128x64 takes 32 µs instead of 58 µs and
  1024x1024 29 µs instead of 39 µs. A non-default
  `FftTuning::with_fused_workgroup_size` keeps the workgroup-memory kernels.
- Multi-line fused kernels issue all their global loads before storing any,
  and pad each line in workgroup memory by one element when neighbouring
  invocations touch neighbouring lines, avoiding bank conflicts.
- Fused smooth kernels, and the convolutions of fused Rader and Bluestein
  kernels, run in few balanced stages of radices up to 16 (composite ones
  as in-register butterflies), with odd radices computed from symmetric
  pairs instead of plain DFTs and bank-conflict padding where the first
  stage needs it; strided ones take a power-of-two line count so each row
  fills whole 32-byte sectors, and multi-line ones read their first stage
  straight from global memory: 1920x1080 takes 56 µs instead of 92 µs,
  1280x720 29 µs instead of 43 µs, and 811x811 52 µs instead of 59 µs.
- Direct prime kernels spread the lines of small transforms over the GPU,
  and on long primes with enough lines produce up to four output pairs per
  invocation: 97x97 takes 13 µs instead of 19 µs and 97x97x97 83 µs instead
  of 98 µs.
- Bluestein convolutions of at most 4096 points run in registers when not
  much longer than the smooth workgroup-memory length (up to 1.85 times up
  to 2048 points, 1.45 times at 4096), and so do Rader axes whose
  convolution would be linear: 179x179x179 takes 0.60 ms instead of 0.79 ms
  and 2039x2039 0.34 ms instead of 0.45 ms. On strided axes they interleave
  up to eight lines per workgroup when each line keeps a whole exchange
  buffer: 947x947 takes 66 µs instead of 82 µs.
- Fused smooth and register kernels read about `2 sqrt(R)` stage twiddles
  per radix-R butterfly instead of `R - 1`, forming the rest as products:
  1523x1523 takes 187 µs instead of 220 µs and 2048x1024 47 µs instead of
  51 µs.
- Primes whose `N - 1` has prime factors from 17 to 61 besides radices up
  to 13 convolve cyclically over `N - 1` points in the fused Rader kernel
  (`f32`), each such factor as one straight-line radix-p stage, when a line
  holds at least 36 of its butterflies and workgroup memory holds the line:
  4241x4241 takes 2.3 ms instead of 5.2 ms, 1381x1381 130 µs
  instead of 187 µs, and N=6121 batched 512 times 0.10 ms instead of
  0.41 ms.
- Plan creation builds the filter spectra of Rader and Bluestein axes with
  an `f64` FFT instead of a plain DFT: an N=6841 plan takes 1.3 s instead
  of 4.6 s, including adapter start-up.
- An `f32` C2C transform of every axis of a small volume (up to 4096 points
  that fit workgroup memory) runs as one kernel per FFT, one workgroup per
  volume, instead of one kernel per axis (`FftTuning::fuse_small_volumes`):
  8x8x8 takes 3.6 µs instead of 10.2 µs, 31x31 6.2 µs instead of 8.5 µs,
  and 16x16x16 batched 256 times 15 µs instead of 32 µs.
- Axes too long for workgroup memory still run fused: contiguous
  power-of-two `f32` axes up to 16384 keep the line in registers in one
  kernel, and other long axes run as two fused passes (`N = N1 * N2`),
  instead of one Stockham pass per radix;
  `FftTuning::with_fuse_long_axes(false)` restores the Stockham stages.

### Fixed

- DX12 plan creation with the FXC compiler no longer takes minutes: kernels
  skip wgpu's redundant workgroup zero fill (a fused N=1024 plan went from 92 s
  to 0.54 s).
- DX12 with the DXC compiler: shader labels no longer contain `:`, which made
  DXC fail to compile many kernels.
- Debug builds no longer panic on Rader axes whose convolution is cyclic
  (`N - 1` smooth, such as N=101).
