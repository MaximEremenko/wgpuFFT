# wgpuFFT

Rust `wgpu` FFT library. The package name is `wgpu-fft`; the library target is
imported as `wgpu_fft`. This is the Rust counterpart of the JavaScript
[WebGPU-FFT](https://github.com/MaximEremenko/WebGPU-FFT) project. The
[wgpuNUFFT](https://github.com/MaximEremenko/wgpuNUFFT) project builds
nonuniform transforms on this crate and pins it as a Git submodule.

## Installation

This crate is intentionally kept GitHub-only and is not published on
crates.io. Depend on a tagged release, together with the matching `wgpu`
major version, because the public API takes `wgpu` types such as
`&wgpu::Device` and `&wgpu::Buffer`:

```toml
[dependencies]
wgpu = "30"
wgpu-fft = { git = "https://github.com/MaximEremenko/wgpuFFT", tag = "v0.1.0" }
```

The default `cpu` feature adds the host-memory [CPU backend](#cpu-backend).
The optional `serde` feature adds schema-versioned JSON persistence for
pipeline-cache snapshots. The minimum supported Rust version is 1.92.

## Quick start

[`examples/quickstart.rs`](examples/quickstart.rs) runs a forward FFT on the
default GPU and falls back to the CPU backend when no adapter is available:

```bash
cargo run --example quickstart
```

A plan records its transform into your own command encoder, reading and
writing caller-owned buffers of `plan.required_buffer_size_bytes()` bytes (input
`STORAGE | COPY_DST`, output `STORAGE | COPY_SRC` to read results back):

```rust
let gpu = wgpu_fft::device::request_default_device().await.expect("GPU adapter");
let plan = FftPlan::c2c(&gpu.device, &gpu.queue, FftConfig::new(1024))?;
let mut encoder = gpu.device.create_command_encoder(&Default::default());
plan.execute_checked(&gpu.device, &mut encoder, &input, &output)?;
gpu.queue.submit([encoder.finish()]);
```

`execute_checked` returns a structured error for undersized buffers or missing
buffer usages; `execute` panics in those cases.

Each `execute*` call opens and closes its own compute pass, which costs a few
microseconds of GPU time. To run several transforms back to back, record them
through one `FftRecorder`, which keeps a single pass open across executions:

```rust
let mut recorder = FftRecorder::new(&mut encoder);
forward.record(&gpu.device, &mut recorder, &a, &b)?;
inverse.record(&gpu.device, &mut recorder, &b, &a)?;
drop(recorder); // ends the pass; the encoder records other commands again
```

On an RTX 5090 this halves the time of a small forward and inverse pair (a
64x64 pair takes 8 µs instead of 17 µs).

This crate exposes a native Rust API for out-of-place complex-to-complex `f32`,
native `f64`, and portable double-float (`df64`) transforms, plus
real/packed-complex `f32` transforms, over 1D/ND shapes and batches. F32 and
native-f64 C2C buffers are interleaved complex scalars:

```text
[re0, im0, re1, im1, ...]
```

Df64 uses four `f32` words per complex element:

```text
[re_hi0, re_lo0, im_hi0, im_lo0, ...]
```

R2C/C2R use the [WebGPU-FFT](https://github.com/MaximEremenko/WebGPU-FFT)
packed-spectrum convention. For a logical real
shape `[N0, ...]`, the packed complex shape is `[floor(N0 / 2) + 1, ...]`, also
stored as interleaved complex values.

Power-of-two axes and multi-stage smooth axes use a single-workgroup fused
kernel when the complete line fits device workgroup storage (8 bytes per `f32`
complex element or 16 bytes per native-`f64`/`df64` complex element) and 256
invocations are supported by default. The fused workgroup size is tunable per
plan. This covers mixed-radix
lengths with radices `2, 3, 4, 5, 7, 8, 11, 13`. Longer contiguous
power-of-two `f32` lines keep their elements in registers within one kernel,
other long axes run as two fused passes (`N = N1 * N2`), and the remaining
lines use generated Stockham stages. Prime `f32` axes up to 127 run a direct
DFT kernel that pairs `X[k]` with `X[p - k]`. Other prime axes route through
Rader, unsupported composite axes route through Bluestein convolution over a
smooth internal length, and mixed-algorithm ND plans execute typed axis-sequence
stage graphs. When a convolution does not fit workgroup memory, Bluestein runs
its forward and inverse FFTs over a power-of-two length of up to 16384 points
in registers within one kernel (`f32`), and Rader primes whose convolution does
not fit use that kernel too. Rader convolves cyclically over `N - 1` points
when that length is smooth, and over a zero-padded smooth length otherwise. The direct DFT compute kernel remains as a length-one fallback.

Public `FftLogicalView` and `BufferView` execution APIs normalize whole-buffer,
offset, segmented, strided, and segmented+strided logical input/output views for
C2C, R2C, and C2R. Compatibility `FftIoView`/`BufferLayout` calls route through
the same logical I/O path. The runtime uses a central `WindowScheduler` for
storage bindings and copy windows, and stage graphs expose route kernels,
helper windows, pack/unpack staging, large chunking, smooth decomposition,
axis decomposition, and Rader/Bluestein bridge execution.

Structured diagnostics are part of the public surface. `FftPlan::diagnostics()`
summarizes the selected route graph, `diagnostics_for_device(...)` adds active
device-limit blockers, `diagnostics_for_limits(...)` preflights an explicit
`FftDeviceLimits` set, `diagnostics_for_views(...)`/`diagnostics_for_io_views(...)`
cover compatibility `BufferView` and `FftIoView` callers,
`diagnostics_for_views_with_workspace(...)`,
`diagnostics_for_io_views_with_workspace(...)`, and
`diagnostics_for_logical_views_with_workspace(...)` cover caller-owned
workspace, and `diagnostics_for_logical_views(...)` or
`diagnostics_for_logical_views_with_limits(...)` add endpoint layout, usage,
alignment, staging, and copy-window blockers.
Successful graph diagnostics include route-owned helper-buffer requirements
derived from helper-window stages.
Diagnostic plan constructors return `FftPlanCreationError` when planning fails,
and checked execution APIs return `FftExecutionError`; both preserve the
original `FftError` plus route, stage, layout, helper-buffer, and device-limit
diagnostics.

A thread-local per-device internal cache reuses bind group layouts, pipeline
layouts, shader modules, and compute pipelines for generated fused
power-of-two, fused smooth-radix, and Stockham kernels, Rader/Bluestein helpers,
real helpers, C2C/real layout helpers, smooth/strided helpers, and direct DFT
pipelines. Typed in-memory cache snapshots can be exported and imported through
`export_pipeline_cache_snapshot` and `import_pipeline_cache_snapshot`;
entries that exceed the target device's fused-kernel compute limits or require
an unavailable shader feature are skipped.

## CPU backend

The default `cpu` feature adds `CpuFftPlan`, which runs the same transforms on
host memory for machines without a usable GPU adapter. It takes the same
`FftConfig` and uses the same layouts as the GPU plans: interleaved complex
values, `[re_hi, re_lo, im_hi, im_lo]` for `df64`, and the packed real
spectrum. It is built on `rustfft` and `realfft` (AVX, SSE, or NEON where the
CPU supports them) and splits large transforms across CPU threads. C2C plans
support every precision, with `df64` computed in native `f64`; real transforms
support `f32` and `f64`.

```rust
use wgpu_fft::{CpuFftPlan, FftConfig};

let plan = CpuFftPlan::c2c(FftConfig::new(1024))?;
let input = vec![0.0f32; plan.required_input_len()];
let mut output = vec![0.0f32; plan.required_output_len()];
plan.execute(&input, &mut output)?;
```

`F64` plans execute with `execute_f64` on `f64` slices. Build with
`default-features = false` to leave the CPU backend out of GPU-only builds.

## Precision

- `FftPrecision::F32` uses native `f32` storage and arithmetic. It is the
  default and is supported by every backend.
- `FftPrecision::F64` uses native `f64` storage and arithmetic. The current
  implementation targets Vulkan devices exposing `wgpu::Features::SHADER_F64`;
  plan creation returns structured `PrecisionUnsupported` diagnostics when the
  feature is unavailable.
- `FftPrecision::Df64` represents each scalar as an unevaluated `hi + lo` pair
  of `f32` words and uses pure-f32 WGSL. It needs no optional device features
  and provides roughly 44-48 effective mantissa bits. Its exponent range is
  still the `f32` range (approximately `1e-38` through `1e38`), and preservation
  of subnormal low words is backend-dependent. Normal C2C mixed-radix, Rader,
  Bluestein, batched/ND, and strided routes support df64; real transforms and
  large execution routes remain structured-unsupported. Vulkan and DX12 exact
  arithmetic canaries are tested on the RTX 5090. Metal's fast-math compiler
  makes it the riskiest backend and it remains untested.

## Tuning

Attach validated per-plan controls with
`FftConfig::with_tuning(FftTuning::new()...)`. Defaults preserve the untuned
planner. Invalid values, incompatible forced algorithms, device-limit
violations, and infeasible forced routes return structured
`FftError::InvalidTuning`; `FftPlan::diagnostics()` reports both requested and
effective tuning. Limit overrides can only lower the adapter's real limits.

| Control | Default | Effect |
|---|---:|---|
| `workgroup_size` | `64` | Staged FFT and linear helper kernels. |
| `fused_workgroup_size` | `256` | Fused power-of-two, smooth, and prime kernels, subject to device invocation and storage limits. |
| `rader_max_prime` | `4096` | Largest non-smooth prime selected for Rader automatically. |
| `direct_max_prime` | `127` | Largest prime axis transformed by a direct DFT kernel instead of Rader (`f32`); `0` keeps Rader for every prime. |
| `force_rader_axes` | `[]` | Physical selected axes that must use Rader; infeasible requests fail instead of changing algorithm. |
| `force_bluestein_axes` | `[]` | Physical selected axes that must use Bluestein. |
| `large_route` | `Auto` | C2C `Auto`, `ForceChunk`, `ForceFourStep`, or `ForceSegmented`; forced real-transform routes are currently unsupported. |
| `large_chunk_max_batches` | `None` | Optional cap on batches per large-chunk staging/execution chunk. |
| `grouped_batch` | `None` | Optional preferred multiple for sequential four-step line windows. |
| `swap_to_2_stage_4_step` | `0` | Axis-length threshold that divides four-step binding capacity into two smaller sequential windows; `0` disables it. |
| `swap_to_3_stage_4_step` | `0` | Axis-length threshold that divides four-step binding capacity into three smaller sequential windows; `0` disables it. |
| `segmented_burst_depth` | `2` | Segmented full-volume A/B staging-ring depth (`1..=3`). |
| `max_storage_buffer_binding_size` | `None` | Optional planning cap, clamped to the device and effective buffer-size limit. |
| `max_buffer_size` | `None` | Optional planning cap, clamped to the device limit. |
| `fused_min_convolution_length` | `128` | Minimum Rader/Bluestein convolution length eligible for fused-prime execution. |
| `fuse_long_axes` | `true` | Runs an axis too long for workgroup memory as one register-resident fused kernel (contiguous power-of-two `f32` axes up to 16384 on 1024-invocation devices) or as two fused passes (`N = N1 * N2`); `false` keeps one Stockham pass per radix. |

The four-step swap thresholds change sequential window sizing; they do not
create concurrent window rings or add FFT stages. There is no public
normal-route coalescing-transpose threshold because that transpose route does
not exist in the Rust implementation. The direct-DFT 64-lane workgroup and
four-step 16x16 transpose tile are internal and are not changed by
`workgroup_size`. FFT convolution is used internally by Rader/Bluestein but is
not exposed as a public `fftconv`/`conv2d` API. Single-stage smooth-axis fusion
also remains an internal route choice because it would not remove a global
pass.

## Current Scope

- C2C `f32` over 1D/ND shapes on native `wgpu` backends.
- Native C2C `f64` over normal 1D/ND mixed-radix, Rader, Bluestein, and
  mixed-algorithm axis-sequence routes. Select it with
  `FftConfig::with_precision(FftPrecision::F64)`. It requires a Vulkan adapter
  and device exposing `wgpu::Features::SHADER_F64`; unsupported devices return
  structured `FftError::PrecisionUnsupported` errors.
- Portable C2C `df64` over normal 1D/ND mixed-radix, Rader, Bluestein, batched,
  and strided routes. It uses only core `f32` WGSL and no optional features.
- R2C/C2R `f32` over full-shape axes only.
- C2C axis subsets through `FftConfig::with_axes(...)`.
- Batch count through `FftConfig::with_batch(...)`.
- Validated per-plan performance tuning through `FftConfig::with_tuning(...)`;
  default tuning preserves the measured planner choices.
- Out-of-place execution only.
- Caller-owned `wgpu::Buffer` input and output.
- `FftPlan::r2c(...)`, `FftPlan::c2r(...)`, `create_r2c_plan(...)`, and
  `create_c2r_plan(...)`.
- `FftPlan::required_input_buffer_size_bytes()`,
  `required_output_buffer_size_bytes()`, `required_buffer_size_bytes()`, and
  `packed_shape()` for real-plan sizing.
- Plan-owned temporary buffers for multi-stage, mixed-algorithm, real, and large
  GPU routes.
- Optional caller-owned C2C workspace buffer for routes that report nonzero
  `workspace_size_bytes()`.
- Public `BufferView` for whole buffers, single slices, and segmented logical
  byte ranges over one or more `wgpu::Buffer`s.
- Public `FftLogicalView`/`FftLogicalLayout` for transform-generic contiguous,
  offset, segmented, strided, and segmented+strided logical I/O.
- Segmented input buffers need `COPY_SRC`; segmented output buffers need
  `COPY_DST`; direct storage windows need `STORAGE`.
- Four-step C2C endpoints currently require input `COPY_SRC` and output
  `COPY_SRC | COPY_DST`; `STORAGE` enables direct bind windows but is optional
  because unaligned and segmented windows use GPU-copy staging.
- Segmented full-volume C2C endpoints accept zero-offset views made exclusively
  from distinct whole physical buffers. Inputs require `COPY_SRC`; outputs
  require `COPY_DST`. Partial, aliased, offset, or strided endpoints remain
  structured errors.
- Public `FftIoView`/`BufferLayout` compatibility views for strided logical I/O.
- Fallible `FftPlan::*_with_diagnostics` constructors, `execute_checked`,
  `execute_views`, `execute_io_views`, `execute_logical_views`,
  workspace-aware variants for all three view layers, and
  `*_with_diagnostics` APIs validate endpoint size, layout, usage, alignment,
  workspace, helper buffers, stage graph windows, and device limits.
- Async `FftPlan::c2c_checked` and `c2c_checked_with_diagnostics` constructors
  additionally capture validation, internal, and out-of-memory errors raised
  during C2C plan construction.
- Large-route policy classifies normal, large-chunk, and large-out-of-core
  plans, with execution metadata for normal, batch chunk, smooth 1D
  decomposition, axis decomposition, Rader/Bluestein bridge routes, and
  GPU-resident rank>=2 four-step C2C routes. C2C and real transforms can execute
  batch `LargeChunk` when each chunk fits active binding limits. C2C and real
  routes can also execute binding-safe large decomposition through staged C2C
  child graphs when the required full-temp/helper buffers fit active limits.
- In this API, out-of-core means outside one storage-binding window: data stays
  GPU-resident. Rank>=2 C2C volumes with at least two selected axes can execute
  when one batch exceeds `maxStorageBufferBindingSize` but the full dataset and
  route-owned helpers fit `maxBufferSize`. Every selected smooth-axis line must
  fit the active binding cap; Rader and Bluestein axes can instead reuse their
  normal child plan or a bounded prime bridge. Automatically selected oversized
  Rader lines may use Bluestein; explicitly forced Rader returns
  `InvalidTuning` instead of changing algorithms. Rank 2 uses stripe
  transposes; higher ranks move each non-front axis through a tiled
  prefix-by-axis block permutation and restore canonical layout afterward. No
  host or disk staging is required for this route.
- Rank>=2 smooth C2C volumes with at least two selected axes and above the
  active policy `maxBufferSize` can use a plan-owned segmented GPU arena when
  every front row and non-front slab line fits a binding-safe staging window.
  Axis rows and slabs are staged through a configurable one-to-three-slot ring
  of A/B window pairs, with one segmented normalization pass and no host or
  disk staging. The measured burst depth defaults to 2.
  On hardware where the logical volume itself exceeds the real device
  `maxBufferSize`, callers provide the volume as distinct whole physical
  buffers through `BufferView`. Prime axes inside a segmented volume,
  partial/strided caller views, and caller-workspace reuse remain
  structured-unsupported.
- Large-chunk, GPU-resident four-step, and segmented full-volume execution are
  currently `f32` routes. Native-`f64` or `df64` plans that require one of
  those routes, and all real extended-precision transforms, return structured
  `PrecisionUnsupported` diagnostics rather than silently changing precision.
- Internal shader/module/pipeline cache keyed by generated fused power-of-two,
  fused smooth-radix, and Stockham stages, Rader helper, real helper, C2C
  strided/smooth helper, and direct DFT pipeline parameters.
- Typed in-memory pipeline cache snapshots expose WGSL shader code and stable
  pipeline key strings. The optional `serde` feature adds schema-versioned JSON
  persistence with source/key integrity validation.
- Route policy executes mixed-radix, Rader, Bluestein, and mixed algorithm
  sequences.
- R2C requires forward direction; C2R requires inverse direction. Real
  transforms currently use the full-shape axis set under the packed axis-0
  convention. Unsupported real axis subsets return structured diagnostics.
- In-place execution, `f16`, DCT/DST, public convolution, and nonuniform
  transforms remain out of scope. Nonuniform FFTs live in
  [wgpuNUFFT](https://github.com/MaximEremenko/wgpuNUFFT).
- The native test/example device helper requests the selected adapter's active
  limits so planner diagnostics and huge-route scheduling see the real storage
  binding and buffer-size limits exposed by that adapter.

## Backends and platforms

`request_default_device()` asks wgpu for a high-performance adapter from
Vulkan, Metal, or DX12 (the browser WebGPU backend on wasm). When several
backends expose the same GPU, Vulkan is listed first.

On DX12, wgpu compiles shaders with DXC when it is available: statically linked
through wgpu's `static-dxc` feature (which needs MSVC 14.41, Visual Studio 2022
17.11, or newer), or as `dxcompiler.dll` 1.8.2502 or newer next to the
executable or on `PATH`. Otherwise it falls back to the legacy FXC compiler.
Every route works with both, but FXC is much slower to create plans: on an RTX
5090 the GPU test suite takes about 260 s with FXC and 40 s with DXC. `df64`
plans and the register-resident kernels of 8192- and 16384-point `f32` axes
are the slowest; an N=8192 plan takes 2.6 s with FXC and 0.1 s on Vulkan.
Ship DXC with DX12 applications, or prefer Vulkan.

Tested on an NVIDIA RTX 5090 under Windows 11 with Vulkan, DX12 (FXC and DXC),
and Chrome 153's WebGPU. Metal, AMD, Intel, and Linux have not been tested.

## Commands

```bash
cargo fmt --check
cargo test
WGPU_BACKEND=vulkan WGPU_FFT_RUN_GPU_TESTS=1 cargo test --test gpu_c2c -- --nocapture
WGPU_BACKEND=vulkan WGPU_FFT_RUN_GPU_TESTS=1 cargo test --test gpu_real -- --nocapture
WGPU_BACKEND=vulkan WGPU_FFT_RUN_GPU_TESTS=1 cargo test --test gpu_f64 --release -- --nocapture
WGPU_BACKEND=vulkan WGPU_FFT_RUN_GPU_TESTS=1 cargo test --test gpu_df64 --release -- --nocapture
```

The environment-variable prefixes above are bash syntax. On Windows PowerShell
set the variables first, for example:

```powershell
$env:WGPU_BACKEND = 'dx12'; $env:WGPU_FFT_RUN_GPU_TESTS = '1'
cargo test --test gpu_df64_canary --release -- --nocapture
```

Browser (Wasm) tests run in Chrome's WebGPU implementation through
`wasm-bindgen-test-runner` and a ChromeDriver matching the installed Chrome
build (set `CHROMEDRIVER` or put `chromedriver.exe` on `PATH`):

```bat
web\run_browser_tests.cmd
```

The runner covers the smoke, correctness-matrix, and large-route tests; see
[web/README.md](web/README.md) for prerequisites. The installed
`wasm-bindgen-cli` version must exactly match the `wasm-bindgen` version in
`Cargo.lock` (currently 0.2.129).

The GPU integration tests are opt-in and skip unless `WGPU_FFT_RUN_GPU_TESTS=1`
is set. The native test helper excludes the GL backend by default because EGL
can crash on WSL/Linux systems where `/dev/dri` nodes exist but are not readable
by the current user. Set `WGPU_BACKEND=gl` explicitly only when GL/EGL access is
known to work; use `WGPU_BACKEND=vulkan` to force Vulkan.

The portable-df64 arithmetic groundwork has a separate exact-word GPU canary.
Run `cargo test --test gpu_df64_canary --release -- --nocapture` once with
`WGPU_BACKEND=vulkan` and once with `WGPU_BACKEND=dx12` (and
`WGPU_FFT_RUN_GPU_TESTS=1` in both cases). Optimization is the hazard: WGSL
provides no no-contract/`precise` qualifier, so these release-backend canaries
are part of the arithmetic support contract. Metal's fast-math compilation
makes it the riskiest backend for the error-free transforms; it is currently
untested. Double-float also retains the `f32` exponent range, and preservation
of subnormals is backend-dependent.

## License

Licensed under the Apache License, Version 2.0 ([LICENSE](LICENSE) or
<http://www.apache.org/licenses/LICENSE-2.0>).
