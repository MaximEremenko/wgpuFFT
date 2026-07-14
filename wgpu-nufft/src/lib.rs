//! Portable nonuniform FFT plans built on `wgpu-fft`.
//!
//! The crate provides reusable host-side type-1 and type-2 planning metadata,
//! FINUFFT-compatible direct `f64` reference transforms, and
//! exponential-of-semicircle kernel tables. Reusable 1D and 2D `f32` type-1
//! and type-2 plans encode spreading or interpolation around a fine-grid C2C
//! transform entirely on the GPU using caller-owned input and output buffers.

#![forbid(unsafe_code)]

mod config;
mod direct;
mod error;
mod gpu;
mod gpu_2d;
mod gpu_dispatch;
#[cfg(feature = "gpu-profiling")]
mod gpu_profile;
mod gpu_scan;
mod gpu_type1;
mod gpu_type1_2d;
mod kernel;
mod plan;

pub use config::{ModeOrder, NufftConfig, NufftSign, DEFAULT_EPS, DEFAULT_SIGMA};
pub use direct::{reference_type1_f64, reference_type2_f64};
pub use error::{NufftError, Result};
#[cfg(feature = "gpu-profiling")]
pub use gpu_profile::{NufftGpuProfileLayout, NufftGpuStage, NufftGpuStageQuery};
pub use kernel::{select_fine_grid_size, EsKernel, MAX_ES_KERNEL_WIDTH, MIN_ES_KERNEL_WIDTH};
pub use plan::{NufftKind, NufftPlan};
pub use wgpu_fft::math::Complex64;
