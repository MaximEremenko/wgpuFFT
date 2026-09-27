//! Portable double-float backend invariant checks.
//!
//! WGSL permits floating-point reassociation and contraction. Those
//! optimizations can invalidate the error-free transforms used by the portable
//! double-float kernels even when ordinary FFT round trips still look
//! plausible. Call [`validate_df64_invariants`] once for each browser/compiler
//! combination before exposing [`crate::FftPrecision::Df64`] as a
//! high-precision option.

use std::fmt;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Waker};

use wgpu::util::DeviceExt;

use crate::math::{
    quick_two_sum_f32, split_f32, two_prod_f32, two_sum_f32, ComplexDoubleFloat, DoubleFloat,
};

const WORDS_PER_CASE: usize = 24;

/// Number of adversarial input cases exercised by the df64 backend canary.
pub const DF64_CANARY_CASE_COUNT: usize = 4;

/// Number of output words compared exactly by the df64 backend canary.
pub const DF64_CANARY_WORD_COUNT: usize = DF64_CANARY_CASE_COUNT * WORDS_PER_CASE;

/// Successful result of [`validate_df64_invariants`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Df64CanaryReport {
    /// Number of independent adversarial cases that ran.
    pub cases: usize,
    /// Number of `f32` bit patterns that matched the host implementation.
    pub exact_words: usize,
}

/// Failure reported by [`validate_df64_invariants`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Df64CanaryError {
    /// A core buffer allocation failed while preparing the canary.
    ResourceFailure { kind: &'static str, message: String },
    /// The backend rejected or could not compile the featureless f32 pipeline.
    PipelineFailure { kind: &'static str, message: String },
    /// A binding, command, or submission failed.
    ExecutionFailure { kind: &'static str, message: String },
    /// Native device polling failed before the readback callback ran.
    DevicePoll { message: String },
    /// The GPU readback buffer could not be mapped.
    MapFailed,
    /// The mapping callback disappeared without producing a result.
    MapCallbackDropped,
    /// The backend returned an unexpected number of result words.
    UnexpectedOutputLength { expected: usize, actual: usize },
    /// One error-free-transform result differed at the bit level.
    InvariantMismatch {
        case: usize,
        word: usize,
        expected_bits: u32,
        actual_bits: u32,
    },
}

impl fmt::Display for Df64CanaryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ResourceFailure { kind, message } => {
                write!(f, "df64 canary resource setup failed ({kind}): {message}")
            }
            Self::PipelineFailure { kind, message } => {
                write!(f, "df64 canary pipeline creation failed ({kind}): {message}")
            }
            Self::ExecutionFailure { kind, message } => {
                write!(f, "df64 canary execution failed ({kind}): {message}")
            }
            Self::DevicePoll { message } => {
                write!(f, "df64 canary device polling failed: {message}")
            }
            Self::MapFailed => f.write_str("df64 canary readback mapping failed"),
            Self::MapCallbackDropped => {
                f.write_str("df64 canary readback callback was dropped")
            }
            Self::UnexpectedOutputLength { expected, actual } => write!(
                f,
                "df64 canary returned {actual} words instead of {expected}"
            ),
            Self::InvariantMismatch {
                case,
                word,
                expected_bits,
                actual_bits,
            } => write!(
                f,
                "df64 backend invariant failed at case {case}, word {word}: actual=0x{actual_bits:08x}, expected=0x{expected_bits:08x}; WGSL contraction or reassociation may have broken an error-free transform"
            ),
        }
    }
}

impl std::error::Error for Df64CanaryError {}

/// Runs exact adversarial checks for the pure-f32 df64 arithmetic library.
///
/// This function requests no features and only creates core WebGPU resources.
/// The supplied `device` and `queue` must belong to one another. All 96 output
/// words must match the host implementation exactly; callers should disable
/// df64 when this function returns an error.
pub async fn validate_df64_invariants(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> Result<Df64CanaryReport, Df64CanaryError> {
    let cases = canary_inputs();
    let expected = cases
        .iter()
        .flat_map(|case| expected_words(*case))
        .collect::<Vec<_>>();
    debug_assert_eq!(expected.len(), DF64_CANARY_WORD_COUNT);

    let resource_scopes = CanaryErrorScopes::push(device);
    let input_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("wgpu_fft.df64_canary.input"),
        contents: bytemuck::cast_slice(&cases),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let output_bytes = (expected.len() * std::mem::size_of::<u32>()) as u64;
    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.df64_canary.output"),
        size: output_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu_fft.df64_canary.readback"),
        size: output_bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    if let Some((kind, message)) = resource_scopes.pop_error().await {
        return Err(Df64CanaryError::ResourceFailure { kind, message });
    }

    let pipeline_scopes = CanaryErrorScopes::push(device);
    let shader_source = format!("{}\n{}", crate::kernels::DF64_WGSL, CANARY_ENTRY_WGSL);
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("wgpu_fft.df64_canary.shader"),
        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("wgpu_fft.df64_canary.pipeline"),
        layout: None,
        module: &shader,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });
    if let Some((kind, message)) = pipeline_scopes.pop_error().await {
        return Err(Df64CanaryError::PipelineFailure { kind, message });
    }

    let execution_scopes = CanaryErrorScopes::push(device);
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("wgpu_fft.df64_canary.bind_group"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output_buffer.as_entire_binding(),
            },
        ],
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("wgpu_fft.df64_canary.encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("wgpu_fft.df64_canary.pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(DF64_CANARY_CASE_COUNT as u32, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&output_buffer, 0, &readback, 0, output_bytes);
    queue.submit(Some(encoder.finish()));
    if let Some((kind, message)) = execution_scopes.pop_error().await {
        return Err(Df64CanaryError::ExecutionFailure { kind, message });
    }

    let slice = readback.slice(..);
    let state = Arc::new(Mutex::new(MapState::default()));
    let callback = MapCallback::new(Arc::clone(&state));
    slice.map_async(wgpu::MapMode::Read, move |result| callback.complete(result));

    #[cfg(not(target_arch = "wasm32"))]
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(|error| Df64CanaryError::DevicePoll {
            message: error.to_string(),
        })?;

    MapFuture { state }
        .await
        .ok_or(Df64CanaryError::MapCallbackDropped)?
        .map_err(|_| Df64CanaryError::MapFailed)?;

    let mapped = slice
        .get_mapped_range()
        .map_err(|_| Df64CanaryError::MapFailed)?;
    let actual = bytemuck::cast_slice::<u8, u32>(&mapped).to_vec();
    drop(mapped);
    readback.unmap();

    if actual.len() != expected.len() {
        return Err(Df64CanaryError::UnexpectedOutputLength {
            expected: expected.len(),
            actual: actual.len(),
        });
    }
    for (index, (&actual_bits, &expected_bits)) in actual.iter().zip(&expected).enumerate() {
        if actual_bits != expected_bits {
            return Err(Df64CanaryError::InvariantMismatch {
                case: index / WORDS_PER_CASE,
                word: index % WORDS_PER_CASE,
                expected_bits,
                actual_bits,
            });
        }
    }

    Ok(Df64CanaryReport {
        cases: DF64_CANARY_CASE_COUNT,
        exact_words: DF64_CANARY_WORD_COUNT,
    })
}

struct CanaryErrorScopes {
    out_of_memory: wgpu::ErrorScopeGuard,
    internal: wgpu::ErrorScopeGuard,
    validation: wgpu::ErrorScopeGuard,
}

impl CanaryErrorScopes {
    fn push(device: &wgpu::Device) -> Self {
        let out_of_memory = device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let internal = device.push_error_scope(wgpu::ErrorFilter::Internal);
        let validation = device.push_error_scope(wgpu::ErrorFilter::Validation);
        Self {
            out_of_memory,
            internal,
            validation,
        }
    }

    async fn pop_error(self) -> Option<(&'static str, String)> {
        // Error scopes are a stack. Pop every scope in strict reverse order so
        // no compiler, allocation, or validation failure escapes uncaptured.
        let validation_pop = self.validation.pop();
        let internal_pop = self.internal.pop();
        let out_of_memory_pop = self.out_of_memory.pop();
        let validation_error = validation_pop.await;
        let internal_error = internal_pop.await;
        let out_of_memory_error = out_of_memory_pop.await;
        if let Some(error) = validation_error {
            Some(("validation", error.to_string()))
        } else if let Some(error) = internal_error {
            Some(("internal", error.to_string()))
        } else {
            out_of_memory_error.map(|error| ("out-of-memory", error.to_string()))
        }
    }
}

#[derive(Default)]
struct MapState {
    result: Option<Result<(), wgpu::BufferAsyncError>>,
    callback_dropped: bool,
    waker: Option<Waker>,
}

struct MapCallback {
    state: Arc<Mutex<MapState>>,
    completed: bool,
}

impl MapCallback {
    fn new(state: Arc<Mutex<MapState>>) -> Self {
        Self {
            state,
            completed: false,
        }
    }

    fn complete(mut self, result: Result<(), wgpu::BufferAsyncError>) {
        self.completed = true;
        let waker = {
            let mut state = self
                .state
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            state.result = Some(result);
            state.waker.take()
        };
        if let Some(waker) = waker {
            waker.wake();
        }
    }
}

impl Drop for MapCallback {
    fn drop(&mut self) {
        if self.completed {
            return;
        }
        let waker = {
            let mut state = self
                .state
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            state.callback_dropped = true;
            state.waker.take()
        };
        if let Some(waker) = waker {
            waker.wake();
        }
    }
}

struct MapFuture {
    state: Arc<Mutex<MapState>>,
}

impl Future for MapFuture {
    type Output = Option<Result<(), wgpu::BufferAsyncError>>;

    fn poll(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(result) = state.result.take() {
            Poll::Ready(Some(result))
        } else if state.callback_dropped {
            Poll::Ready(None)
        } else {
            state.waker = Some(context.waker().clone());
            Poll::Pending
        }
    }
}

fn canary_inputs() -> [[[f32; 4]; 2]; DF64_CANARY_CASE_COUNT] {
    [
        [
            [16_777_216.0, 1.0, 16_777_216.0, 4097.0],
            [-16_777_216.0, 1.0, 1.0, 4097.0],
        ],
        [
            [1.0, f32::from_bits(0x3380_0000), 1.000_000_1, 1.000_000_1],
            [-1.0, f32::from_bits(0x3300_0000), -1.0, 0.999_999_9],
        ],
        [
            [1.0e36, 1.0e28, 1.0e20, f32::from_bits(0x7b40_97ce)],
            [1.0e-30, 1.0e-37, 1.0, f32::from_bits(0x0da2_4260)],
        ],
        [
            [-12_345.125, 0.000_122_070_31, -33_554_432.0, -17.25],
            [0.031_257_63, -1.0e-10, 3.0, 3.000_000_2],
        ],
    ]
}

fn expected_words(input: [[f32; 4]; 2]) -> [u32; WORDS_PER_CASE] {
    let a = input[0];
    let b = input[1];
    let (sum_hi, sum_lo) = two_sum_f32(a[0], b[0]);
    let (quick_hi, quick_lo) = quick_two_sum_f32(a[2], b[2]);
    let (split_hi, split_lo) = split_f32(a[3]);
    let (prod_hi, prod_lo) = two_prod_f32(a[3], b[3]);
    let a_dd = DoubleFloat::new(a[0], a[1]);
    let b_dd = DoubleFloat::new(b[0], b[1]);
    let add = a_dd.add_df(b_dd);
    let mul = a_dd.mul_df(b_dd);
    let a_complex = ComplexDoubleFloat::new(a_dd, DoubleFloat::new(a[2], a[3]));
    let b_complex = ComplexDoubleFloat::new(b_dd, DoubleFloat::new(b[2], b[3]));
    let complex_add = a_complex.add_df(b_complex);
    let complex_mul = a_complex.mul_df(b_complex);
    let (edge_prod_hi, edge_prod_lo) = two_prod_f32(f32::MAX, f32::MIN_POSITIVE);
    let compile_time_scale = DoubleFloat::from_f64(1.0 / 34.0);
    let compile_time_product =
        DoubleFloat::new(1.0, f32::from_bits(0x3080_0000)).mul_df(compile_time_scale);
    [
        sum_hi.to_bits(),
        sum_lo.to_bits(),
        quick_hi.to_bits(),
        quick_lo.to_bits(),
        split_hi.to_bits(),
        split_lo.to_bits(),
        prod_hi.to_bits(),
        prod_lo.to_bits(),
        add.hi.to_bits(),
        add.lo.to_bits(),
        mul.hi.to_bits(),
        mul.lo.to_bits(),
        complex_add.re_hi.to_bits(),
        complex_add.re_lo.to_bits(),
        complex_add.im_hi.to_bits(),
        complex_add.im_lo.to_bits(),
        complex_mul.re_hi.to_bits(),
        complex_mul.re_lo.to_bits(),
        complex_mul.im_hi.to_bits(),
        complex_mul.im_lo.to_bits(),
        edge_prod_hi.to_bits(),
        edge_prod_lo.to_bits(),
        compile_time_product.hi.to_bits(),
        compile_time_product.lo.to_bits(),
    ]
}

const CANARY_ENTRY_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> inputs: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> outputs: array<u32>;

fn store_df64(base: u32, value: Df64) {
    outputs[base] = bitcast<u32>(value.hi);
    outputs[base + 1u] = bitcast<u32>(value.lo);
}

fn store_complex(base: u32, value: vec4<f32>) {
    outputs[base] = bitcast<u32>(value.x);
    outputs[base + 1u] = bitcast<u32>(value.y);
    outputs[base + 2u] = bitcast<u32>(value.z);
    outputs[base + 3u] = bitcast<u32>(value.w);
}

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let case_index = gid.x;
    let a = inputs[case_index * 2u];
    let b = inputs[case_index * 2u + 1u];
    let base = case_index * 24u;
    store_df64(base, df64_two_sum(a.x, b.x));
    store_df64(base + 2u, df64_quick_two_sum(a.z, b.z));
    store_df64(base + 4u, df64_split(a.w));
    store_df64(base + 6u, df64_two_prod(a.w, b.w));
    let a_dd = Df64(a.x, a.y);
    let b_dd = Df64(b.x, b.y);
    store_df64(base + 8u, df64_add(a_dd, b_dd));
    store_df64(base + 10u, df64_mul(a_dd, b_dd));
    store_complex(base + 12u, df64_complex_add(a, b));
    store_complex(base + 16u, df64_complex_mul(a, b));
    store_df64(base + 20u, df64_two_prod(
        bitcast<f32>(0x7f7fffffu),
        bitcast<f32>(0x00800000u),
    ));
    store_df64(base + 22u, df64_mul(
        Df64(1.0, bitcast<f32>(0x30800000u)),
        Df64(bitcast<f32>(0x3cf0f0f1u), bitcast<f32>(0xaef0f0f1u)),
    ));
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canary_vectors_remain_adversarial() {
        let expected = canary_inputs()
            .iter()
            .flat_map(|case| expected_words(*case))
            .collect::<Vec<_>>();
        assert_eq!(expected.len(), DF64_CANARY_WORD_COUNT);
        let first = &expected[..WORDS_PER_CASE];
        assert_eq!(first[2], 16_777_216.0f32.to_bits());
        assert_eq!(first[3], 1.0f32.to_bits());
        assert_eq!(first[4], 4096.0f32.to_bits());
        assert_eq!(first[5], 1.0f32.to_bits());
        assert_eq!(first[6], 16_785_408.0f32.to_bits());
        assert_eq!(first[7], 1.0f32.to_bits());
        assert_eq!(first[8], 2.0f32.to_bits());
        assert_eq!(first[9], 0.0f32.to_bits());
        assert!(
            expected
                .chunks_exact(WORDS_PER_CASE)
                .all(|words| words[7] != 0.0f32.to_bits()),
            "each two_prod canary must require a nonzero error word"
        );
        let edge = two_prod_f32(f32::MAX, f32::MIN_POSITIVE);
        let compile_time_scale = DoubleFloat::from_f64(1.0 / 34.0);
        assert_ne!(compile_time_scale.lo, 0.0);
        let compile_time_product =
            DoubleFloat::new(1.0, f32::from_bits(0x3080_0000)).mul_df(compile_time_scale);
        for words in expected.chunks_exact(WORDS_PER_CASE) {
            assert_eq!(words[20], edge.0.to_bits());
            assert_eq!(words[21], edge.1.to_bits());
            assert!(f32::from_bits(words[20]).is_finite());
            assert!(f32::from_bits(words[21]).is_finite());
            assert_eq!(words[22], compile_time_product.hi.to_bits());
            assert_eq!(words[23], compile_time_product.lo.to_bits());
        }
    }
}
