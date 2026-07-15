# Chrome WebGPU / WASM results — 2026-07-15

This archive records browser execution of `wgpu-fft` and, in later sections,
`wgpu-nufft`. Chrome compiles the WGSL through its WebGPU implementation and
Tint; the tests do not exercise naga's native shader-compilation path.

## Environment

- GPU: NVIDIA GeForce RTX 5090
- NVIDIA driver: 610.47
- OS: Microsoft Windows 10.0.26200.8737
- Chrome: 150.0.7871.116
- ChromeDriver: 150.0.7871.124, branch-heads/7871 revision 3359
- Rust: rustc 1.95.0 (59807616e, 2026-04-14)
- `wasm-bindgen`: 0.2.120
- `wgpu`: 29.0.3
- Phase A base commit: `d1b589928b467336f331f991916e0d562de6ffe9`

The checked-in `webdriver.json` requested Chrome with
`--enable-unsafe-webgpu`, `--disable-gpu-sandbox`, `--no-first-run`,
`--no-default-browser-check`, `--disable-background-networking`,
`--disable-component-update`, `--disable-breakpad`, and
`--disable-crash-reporter`. The runner was headless Chrome through
`wasm-bindgen-test-runner` and ChromeDriver.

## Phase A — build and smoke

The full workspace compiled for `wasm32-unknown-unknown`. An adapter-max browser
device executed a four-element C2C transform and matched the baked-in expected
values. Native workspace tests and the complete RTX 5090 Vulkan release GPU
matrix also remained green.

## Phase B — browser-default correctness

### Method

The correctness matrix deliberately requested a separate featureless device
with `required_limits: wgpu::Limits::default()`. The device reported exactly:

| Limit | Value |
|---|---:|
| `maxStorageBufferBindingSize` | 134,217,728 B (128 MiB) |
| `maxBufferSize` | 268,435,456 B (256 MiB) |
| `maxComputeWorkgroupStorageSize` | 16,384 B (16 KiB) |
| `maxComputeInvocationsPerWorkgroup` | 256 |

Small transforms were read back in full and compared with the Rust CPU
references. Large routes used a unit impulse and sampled outputs, including
samples immediately before and after physical segment boundaries. A forward,
unnormalized transform of that impulse is exactly `(1, 0)` at every output.

### Results

| Suite | Browser-default evidence | Result |
|---|---|---:|
| Smooth C2C | N=330, forward and inverse, mixed-radix route | PASS |
| Rader C2C | N=101, forward and inverse | PASS |
| Bluestein C2C | N=85, forward and inverse | PASS |
| Fused boundary | N=2048: one fused stage, zero workspace; relative L2 `1.342e-7` vs f64 oracle | PASS |
| Fused fallback | N=4096: Stockham multipass, nonzero workspace; relative L2 `1.365e-7` vs f64 oracle | PASS |
| Real transforms | R2C and C2R, N=34 Bluestein route | PASS |
| Native f64 gate | `PrecisionUnsupported { reason: "device-missing-shader-f64" }` | PASS |
| Df64 Tint canary | 4 adversarial cases, all 96 u32 words bit-exact | PASS |
| Natural four-step | shape `[4096, 4116]`, 134,873,088 B > 128 MiB binding | PASS |
| Natural segmented volume | shape `[4096, 8232]`, 269,746,176 B > 256 MiB buffer | PASS |

Browser-reported test times were 2.53 s for the core matrix, 0.82 s for the two
large-route cases, and 0.13 s for the smoke test. These are test-suite timings,
not benchmark results.

The segmented case used three whole physical buffers for each endpoint and
sampled across both the 128 MiB and 256 MiB logical boundaries. Enabling this
real browser use case required a bounded endpoint relaxation: segmented
full-volume execution now accepts zero-offset views composed exclusively of
distinct whole physical buffers. Partial or aliased buffers, nonzero logical
offsets, and strided logical I/O remain structured errors.

### Df64 verdict

Tint preserved every error-free-transform invariant exactly on this Chrome
build. Browser Df64 therefore remains enabled. This result is compiler- and
browser-version-specific, which is why the public canary API is retained for
the website wrapper instead of treating this one pass as a permanent compiler
guarantee. Browser F64 remains unavailable and fails through the existing
structured precision error.

### Native parity

After the browser changes, `cargo test --workspace` passed 300 `wgpu-fft` unit
tests, 90 `wgpu-nufft` unit tests, and 19 CPU-foundation integration tests. The
complete release GPU workspace command passed in 321.8 s on the RTX 5090 using
Vulkan and driver 610.47; adapter lines were emitted by the FFT and NUFFT
suites. The public df64 canary also remained bit-exact on native Vulkan and
DX12.

## Phase C — persistence and JavaScript surface

### Method

The `wgpu-web` wrapper keeps plans and caller-owned buffers GPU-resident and
exposes initialization, plan creation, upload, execution, download, and cache
persistence to JavaScript. Initialization runs the complete 96-word df64
canary; a failure disables browser df64 without disabling f32. Native f64 is
forwarded to the core planner and retains its structured browser-unsupported
error.

The optional `wgpu-fft/serde` feature serializes typed shader and pipeline keys
as schema-versioned JSON. Import regenerates every WGSL source from its typed
key and rejects altered source, forged stable-key projections, missing or
duplicate entries, and layout or entry-point mismatches. This is a validated
source/pipeline prewarm cache, not a browser driver-binary cache.

The browser demo stored the snapshot in `localStorage`, read it back from a
fresh same-origin document, imported it into a fresh WebGPU context, recreated
the plan, and executed an impulse transform.
The comparison used one f32 forward C2C case, N=4096 and batch=1024. Plans,
allocation, upload, and download were excluded. Each timed iteration included
command encoding, one submit, and queue completion. Both implementations ran
in Chrome 150/Tint with adapter-max limits, five warmups, and three alternating
20-iteration blocks. The JavaScript reference was the clean `WebGPU-FFT`
revision `fa45c93f524a69a96c9f55acfad865226bfccd29`.

### Results

| Check | Result |
|---|---:|
| Snapshot JSON size | 34,002 B |
| Cold plan creation | 56.10 ms |
| Snapshot import | 2.90 ms |
| Restored plan creation | 0.50 ms |
| Restored impulse output | bit-exact |
| wgpu-fft Rust/Wasm | 3.2367 ± 0.0319 ms |
| WebGPU-FFT JavaScript | 3.3183 ± 0.0859 ms |
| Rust/Wasm ÷ JavaScript | 0.975x |

An immediate repeat reversed the small lead: Rust/Wasm measured 3.3433 ms and
JavaScript 3.2350 ms, a 1.033x ratio. The defensible conclusion is parity
within roughly 3% session noise, not a directional host-language win. Both
recorded impulse outputs were exact. The active device exposed 2,147,483,648 B
`maxBufferSize`, 2,147,483,644 B `maxStorageBufferBindingSize`, and 32,768 B of
workgroup storage; the Rust plan reported the mixed-radix route.

Chrome's JavaScript adapter info reported vendor `NVIDIA` and architecture
`blackwell`, but the wgpu browser adapter exposed empty name and zero numeric
vendor/device IDs. The two requests used the same high-performance preference
and produced identical active limits, but browser privacy prevents an exact
device-ID equality proof; the machine-level RTX 5090 evidence remains the
native adapter log and `nvidia-smi`. The archived run used a fresh build in
headed Chrome and records Wasm SHA-256
`e8019094c6edd2ef2026d8ee62ed5a5665ce4ba5e31aa4ae0a9c74e05b8521fb`.

The full machine-readable record is [phase-c-result.json](phase-c-result.json).

## Phase D — in-browser GPU NUFFT

### Method

The browser requested a featureless device at exact WebGPU default limits and
ran batched NUFFT types 1, 2, and 3 in one through three dimensions through
Chrome/Tint. Mode shapes were `[17]`, `[8,12]`, and `[4,6,8]`; every case used
two transform-major vectors and deterministic boundary and duplicate points;
the type-3 fixtures also included a distinct near-clustered point.
Type-1/type-2 signs were `+,-,+` by dimension and type-3 used `-,+,-`, so both
exponential signs were exercised.

F32 used `eps=1e-5`; Df64 used `eps=1e-8`. The oracle was a direct f64 NDFT
built from the values actually represented by the uploaded F32 or Df64 words.
The Rust `wasm-bindgen-test` matrix and the public `wgpu-web` JavaScript surface
ran independently. The Rust matrix passed in 73.60 s. The archive-grade public
surface run rebuilt Wasm, launched headed Chrome directly, and completed in
85.1 s including build and browser startup. No native or CUDA reference was run.

The active limits were exactly:

| Limit | Value |
|---|---:|
| `maxStorageBufferBindingSize` | 134,217,728 B (128 MiB) |
| `maxBufferSize` | 268,435,456 B (256 MiB) |
| `maxComputeWorkgroupStorageSize` | 16,384 B (16 KiB) |
| `maxComputeInvocationsPerWorkgroup` | 256 |

### Correctness

Values below are relative L2 errors from the public JavaScript surface. The
independent Rust browser matrix produced the same rounded values.

| Precision | Type 1: 1D / 2D / 3D | Type 2: 1D / 2D / 3D | Type 3: 1D / 2D / 3D |
|---|---|---|---|
| F32 | `4.965e-6` / `1.453e-5` / `1.365e-5` | `2.818e-6` / `3.101e-5` / `3.472e-5` | `9.297e-6` / `1.577e-5` / `2.473e-5` |
| Df64 | `4.204e-9` / `1.691e-8` / `6.834e-9` | `1.750e-9` / `3.985e-8` / `1.309e-8` | `1.779e-8` / `2.126e-8` / `4.089e-8` |

Tint preserved all four adversarial df64 cases, exactly 96 of 96 storage words.
The df64 type-3 source-prephase bound of 1024 executed with relative L2
`3.498e-8`; a bound of 1025 returned the structured `source pre-phase`
accuracy-range error. A native-F64 NUFFT plan returned the expected structured
missing-`SHADER_F64` precision error.

### Verdict and scope

The complete `wgpu-nufft` transform surface now runs inside Chrome using the
browser's WebGPU implementation: types 1/2/3, 1D-3D, batched F32 and portable
Df64, with GPU-resident plans and caller buffers. This is the first in-browser
GPU NUFFT implementation known to this project. The result is specific to
Chrome 150.0.7871.116 and its Tint/backend compiler, which is why df64 remains
guarded by the runtime exact-word canary rather than a permanent browser claim.

The persisted pipeline snapshot covers the embedded `wgpu-fft` fine-grid
pipelines, not NUFFT-specific spread/interpolation shaders. Browser packaging
polish for npm/crates.io and IndexedDB storage beyond the localStorage demo are
optional follow-ups, not transform-surface gaps. The final Wasm SHA-256 was
`1acc9b983a886e20f7fee2a47819d0dd4153eff9f1bc2062501d6d2ce8ab3269`.
The machine-readable record is [phase-d-result.json](phase-d-result.json).
