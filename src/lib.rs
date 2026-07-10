//! Rust `wgpu` FFT library.
//!
//! The crate supports out-of-place C2C, R2C, and C2R `f32` transforms over
//! 1D/ND shapes and batches. Public `FftLogicalView` and `BufferView` inputs
//! normalize contiguous, offset, segmented, strided, and segmented+strided
//! logical I/O before dispatch. Existing `FftIoView` compatibility views route
//! through the same logical I/O path.
//!
//! Power-of-two and multi-stage smooth axes execute in one workgroup when the
//! complete line fits device workgroup storage; single-stage smooth axes and
//! larger lines use mixed-radix Stockham stages. Other prime axes use Rader,
//! unsupported composite axes use Bluestein over a smooth convolution length,
//! and mixed-algorithm ND plans compose typed stage graphs. Large routes
//! include batch chunking, smooth and axis
//! decomposition, and Rader/Bluestein bridge execution when active `wgpu`
//! binding limits allow them.
//!
//! Use diagnostic constructors such as `FftPlan::c2c_with_diagnostics(...)`
//! when plan creation itself may fail, and use `FftPlan::diagnostics()`,
//! `diagnostics_for_device(...)`,
//! `diagnostics_for_limits(...)`, `diagnostics_for_views(...)`,
//! `diagnostics_for_io_views(...)`, `diagnostics_for_views_with_workspace(...)`,
//! `diagnostics_for_io_views_with_workspace(...)`,
//! `diagnostics_for_logical_views_with_workspace(...)`,
//! `diagnostics_for_logical_views(...)`, and
//! `diagnostics_for_logical_views_with_limits(...)` to inspect route, stage,
//! helper-buffer, workspace, logical-layout, and device-limit requirements.
//! Successful graph diagnostics include route-owned helper-buffer requirements
//! derived from helper-window stages.
//! Checked execution APIs, including the concise whole-buffer
//! `execute_checked(...)` and `execute_checked_with_workspace(...)` aliases,
//! return `FftExecutionError`, preserving the original `FftError` plus
//! structured diagnostics for validation or binding-safety failures. Pipeline
//! cache snapshots can be exported and imported as typed in-memory Rust values,
//! and long-running shape sweeps can explicitly clear thread-local per-device
//! cache entries.

pub mod config;
pub mod device;
pub mod diagnostics;
pub mod error;
pub mod kernels;
pub mod math;
pub mod plan;
pub mod runtime;

pub use config::{FftConfig, FftDirection, Normalization};
pub use diagnostics::{
    FftBlocker, FftBlockerKind, FftBufferRequirement, FftDeviceLimits, FftDiagnostics,
    FftRouteSummary, FftStageSummary,
};
pub use error::{FftError, FftExecutionError, FftPlanCreationError, Result};
pub use plan::{
    create_c2r_plan, create_c2r_plan_with_diagnostics, create_plan, create_plan_with_diagnostics,
    create_r2c_plan, create_r2c_plan_with_diagnostics, FftPlan, FftTransformKind,
};
pub use runtime::axis_policy::{AxisKind, DEFAULT_RADER_MAX_PRIME};
pub use runtime::buffer_view::{BufferLayout, BufferRange, BufferSegment, BufferView, FftIoView};
pub use runtime::c2c::C2cRoute;
pub use runtime::large_policy::{
    LargeExecutionKind, LargeFactorSplit, LargePolicyLimits, LargeRouteDiagnostics, LargeRouteMode,
    LargeRoutingPolicy, OutOfCoreAxisWindowPolicy, OutOfCoreAxisWindowPolicyInput, OutOfCorePlan,
    OutOfCorePlanInput, OutOfCoreWindow,
};
pub use runtime::logical_io::{FftEndpointFormat, FftLogicalLayout, FftLogicalView};
pub use runtime::pipeline_cache::{
    clear_thread_local_pipeline_cache, export_pipeline_cache_snapshot,
    import_pipeline_cache_snapshot, PipelineCacheSnapshot, PIPELINE_CACHE_SNAPSHOT_SCHEMA,
    PIPELINE_CACHE_SNAPSHOT_VERSION,
};
