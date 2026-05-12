# wgpuFFT

Rust `wgpu` FFT library.

This crate exposes a native Rust API for out-of-place complex-to-complex and
real/packed-complex `f32` transforms over 1D/ND shapes and batches. C2C buffers
are interleaved complex:

```text
[re0, im0, re1, im1, ...]
```

R2C/C2R use the WebGPU-FFT packed-spectrum convention. For a logical real
shape `[N0, ...]`, the packed complex shape is `[floor(N0 / 2) + 1, ...]`, also
stored as interleaved complex values.

Power-of-two axes and multi-stage smooth axes use a single-workgroup fused
kernel when the complete line fits device workgroup storage (8 bytes per
complex element) and 256 invocations are supported. This covers mixed-radix
lengths with radices `2, 3, 4, 5, 7, 8, 11, 13`; single-stage smooth axes and
larger lines use generated Stockham stages. Other prime axes route through
Rader, unsupported composite axes route through Bluestein convolution over a
smooth internal length, and mixed-algorithm ND plans execute typed axis-sequence
stage graphs. The direct DFT compute kernel remains as a length-one fallback.

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
entries that exceed the target device's fused-kernel compute limits are skipped.

## Current Scope

- C2C over 1D/ND shapes.
- R2C/C2R `f32` over full-shape axes only.
- C2C axis subsets through `FftConfig::with_axes(...)`.
- Batch count through `FftConfig::with_batch(...)`.
- `f32` only.
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
- Segmented full-volume C2C endpoints currently require one zero-offset
  contiguous input buffer with `COPY_SRC` and output buffer with `COPY_DST`.
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
  route-owned helpers fit `maxBufferSize`. Mixed-radix axes use bind-sized FFT
  windows; Rader and Bluestein axes reuse their normal child plan or a bounded
  prime bridge, with oversized Rader lines deliberately falling back to
  Bluestein. Rank 2 uses stripe transposes; higher ranks move each non-front
  axis through a tiled prefix-by-axis block permutation and restore canonical
  layout afterward. No host or disk staging is required for this route.
- Rank>=2 smooth C2C volumes above the active policy `maxBufferSize` can use a
  plan-owned segmented GPU arena. Axis rows and non-front slabs are staged
  through a measured two-slot ring of binding-safe A/B window pairs, with one
  segmented normalization pass and no host or disk staging. The burst depth is
  currently an internal policy choice; a public tuning override is deferred.
  On hardware where the logical volume itself exceeds the real device
  `maxBufferSize`, caller-segmented endpoints are still required and remain
  deferred; the implemented route is directly executable when an
  internal/tuning cap is below the real endpoint limit. Prime axes inside a
  segmented volume, segmented/strided caller views, and caller-workspace reuse
  remain structured-unsupported.
- Internal shader/module/pipeline cache keyed by generated fused power-of-two,
  fused smooth-radix, and Stockham stages, Rader helper, real helper, C2C
  strided/smooth helper, and direct DFT pipeline parameters.
- Typed in-memory pipeline cache snapshots expose WGSL shader code and stable
  pipeline key strings.
- Route policy executes mixed-radix, Rader, Bluestein, and mixed algorithm
  sequences.
- R2C requires forward direction; C2R requires inverse direction. Real
  transforms currently use the full-shape axis set under the packed axis-0
  convention. Unsupported real axis subsets return structured diagnostics.
- In-place execution, f16/f64, DCT/DST, public convolution, NUFFT, and WASM
  wrapper work are out of scope.
- No serde/JSON cache snapshot persistence yet.
- Native `wgpu` first; WASM is planned later. `wgpuFFT` stays `wgpu`-only; use
  `WGPU_BACKEND=vulkan` for Vulkan/native validation where available.
- The native test/example device helper requests the selected adapter's active
  limits so planner diagnostics and huge-route scheduling see the real storage
  binding and buffer-size limits exposed by that adapter.

## Commands

```bash
cargo fmt --check
cargo test
WGPU_BACKEND=vulkan WGPU_FFT_RUN_GPU_TESTS=1 cargo test --test gpu_c2c -- --nocapture
WGPU_BACKEND=vulkan WGPU_FFT_RUN_GPU_TESTS=1 cargo test --test gpu_real -- --nocapture
```

The GPU integration tests are opt-in and skip unless `WGPU_FFT_RUN_GPU_TESTS=1`
is set. The native test helper excludes the GL backend by default because EGL
can crash on WSL/Linux systems where `/dev/dri` nodes exist but are not readable
by the current user. Set `WGPU_BACKEND=gl` explicitly only when GL/EGL access is
known to work; use `WGPU_BACKEND=vulkan` to force Vulkan.
