use std::fmt;

use crate::diagnostics::{
    large_route_blocker, FftBlocker, FftBlockerKind, FftDiagnostics, FftRouteSummary,
};

pub type Result<T> = std::result::Result<T, FftError>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FftError {
    ZeroLength,
    ZeroBatch,
    LengthTooLarge {
        len: usize,
    },
    DispatchWorkgroupsUnsupported {
        workgroups: u32,
        max_per_dimension: u32,
    },
    UnsupportedLength {
        len: usize,
    },
    EmptyAxes,
    InvalidAxis {
        axis: usize,
        rank: usize,
    },
    DuplicateAxis {
        axis: usize,
    },
    UnsupportedAxisKind {
        axis: usize,
        len: usize,
        kind: &'static str,
    },
    InvalidRealTransformDirection {
        transform: &'static str,
        expected: &'static str,
        actual: &'static str,
    },
    UnsupportedRealAxes {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
    RealWorkspaceUnsupported {
        transform: &'static str,
    },
    StridedLayoutUnsupported {
        transform: &'static str,
    },
    StridedSegmentedUnsupported,
    BufferLayoutZeroStride,
    BufferLayoutBatchStrideTooSmall {
        required: u64,
        actual: u64,
    },
    BufferLayoutOutOfBounds {
        required_bytes: u64,
        actual_bytes: u64,
    },
    BufferLayoutTooLarge {
        value: u64,
        limit: u64,
    },
    WorkspaceTooSmall {
        required: u64,
        actual: u64,
    },
    SegmentedWorkspaceUnsupported,
    SegmentedBufferViewUnsupported {
        usage: &'static str,
    },
    BufferViewTooSmall {
        required: u64,
        actual: u64,
    },
    BufferViewEmptySegments,
    BufferSegmentZeroSize {
        index: usize,
    },
    BufferSegmentOutOfBounds {
        index: usize,
        offset: u64,
        size: u64,
        buffer_size: u64,
    },
    BufferViewOutOfBounds {
        offset: u64,
        size: u64,
        buffer_size: u64,
    },
    BufferViewWindowOutOfRange {
        offset: u64,
        size: u64,
        length: u64,
    },
    BufferViewCopyUnaligned {
        offset: u64,
        size: u64,
        alignment: u64,
    },
    BufferViewMissingUsage {
        usage: &'static str,
    },
    BufferViewOffsetUnaligned {
        offset: u64,
        alignment: u64,
    },
    LargeChunkUnsupported {
        reason: &'static str,
        bytes_per_batch: u64,
        max_bind_bytes: u64,
    },
    LargeRouteWorkspaceUnsupported {
        route_mode: &'static str,
    },
    LargeRouteLayoutUnsupported {
        route_mode: &'static str,
        layout: &'static str,
    },
    LargeRouteUnsupported {
        route_mode: &'static str,
        reason_codes: Vec<&'static str>,
    },
    LargeGraphStageUnsupported {
        stage: &'static str,
        reason: &'static str,
    },
    WindowScheduleUnsupported {
        reason: &'static str,
        requested_bytes: u64,
        max_bind_bytes: u64,
    },
    HelperBufferTooLarge {
        helper_buffer: &'static str,
        requested_bytes: u64,
        max_buffer_size: u64,
    },
    LargeBridgeUnsupported {
        route: &'static str,
        reason: &'static str,
    },
    OutOfCoreExecutionUnsupported {
        reason: &'static str,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FftExecutionError {
    error: FftError,
    diagnostics: FftDiagnostics,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FftPlanCreationError {
    error: FftError,
    diagnostics: FftDiagnostics,
}

impl FftExecutionError {
    pub fn new(error: FftError, diagnostics: FftDiagnostics) -> Self {
        Self { error, diagnostics }
    }

    pub fn error(&self) -> &FftError {
        &self.error
    }

    pub fn diagnostics(&self) -> &FftDiagnostics {
        &self.diagnostics
    }

    pub fn into_parts(self) -> (FftError, FftDiagnostics) {
        (self.error, self.diagnostics)
    }
}

impl FftPlanCreationError {
    pub fn new(error: FftError, diagnostics: FftDiagnostics) -> Self {
        Self { error, diagnostics }
    }

    pub fn from_error(error: FftError, transform: &'static str) -> Self {
        let diagnostics = error.diagnostics_for_transform(transform);
        Self { error, diagnostics }
    }

    pub fn error(&self) -> &FftError {
        &self.error
    }

    pub fn diagnostics(&self) -> &FftDiagnostics {
        &self.diagnostics
    }

    pub fn into_parts(self) -> (FftError, FftDiagnostics) {
        (self.error, self.diagnostics)
    }
}

impl fmt::Display for FftError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroLength => write!(f, "FFT length must be greater than zero"),
            Self::ZeroBatch => write!(f, "FFT batch count must be greater than zero"),
            Self::LengthTooLarge { len } => {
                write!(
                    f,
                    "FFT length {len} exceeds the current u32 GPU dispatch limit"
                )
            }
            Self::DispatchWorkgroupsUnsupported {
                workgroups,
                max_per_dimension,
            } => {
                write!(
                    f,
                    "dispatch of {workgroups} workgroups cannot fit a safely linearizable 3D grid with \
                     max_compute_workgroups_per_dimension {max_per_dimension}"
                )
            }
            Self::UnsupportedLength { len } => write!(
                f,
                "FFT length {len} is not supported by the current mixed-radix milestone"
            ),
            Self::EmptyAxes => write!(f, "FFT axes must not be empty"),
            Self::InvalidAxis { axis, rank } => write!(
                f,
                "FFT axis {axis} is out of range for a shape with rank {rank}"
            ),
            Self::DuplicateAxis { axis } => {
                write!(f, "FFT axis {axis} appears more than once")
            }
            Self::UnsupportedAxisKind { axis, len, kind } => write!(
                f,
                "FFT axis {axis} with length {len} selected {kind}, but that axis algorithm is not implemented yet"
            ),
            Self::InvalidRealTransformDirection {
                transform,
                expected,
                actual,
            } => write!(
                f,
                "{transform} requires {expected} direction, got {actual} direction"
            ),
            Self::UnsupportedRealAxes { expected, actual } => write!(
                f,
                "real FFT plans currently require all axes in order: expected {expected:?}, got {actual:?}"
            ),
            Self::RealWorkspaceUnsupported { transform } => write!(
                f,
                "{transform} caller-owned workspace is not supported yet; omit the workspace argument"
            ),
            Self::StridedLayoutUnsupported { transform } => write!(
                f,
                "{transform} strided buffer layouts are not supported yet"
            ),
            Self::StridedSegmentedUnsupported => write!(
                f,
                "non-contiguous strided FFT layouts are not supported for segmented buffer views"
            ),
            Self::BufferLayoutZeroStride => {
                write!(f, "FFT buffer layout element stride must be greater than zero")
            }
            Self::BufferLayoutBatchStrideTooSmall { required, actual } => write!(
                f,
                "FFT buffer layout batch stride is too small: required at least {required} complex elements, got {actual}"
            ),
            Self::BufferLayoutOutOfBounds {
                required_bytes,
                actual_bytes,
            } => write!(
                f,
                "FFT buffer layout is out of bounds: required {required_bytes} bytes, got {actual_bytes} bytes"
            ),
            Self::BufferLayoutTooLarge { value, limit } => write!(
                f,
                "FFT buffer layout value {value} exceeds supported limit {limit}"
            ),
            Self::WorkspaceTooSmall { required, actual } => write!(
                f,
                "FFT workspace buffer is too small: required {required} bytes, got {actual} bytes"
            ),
            Self::SegmentedWorkspaceUnsupported => write!(
                f,
                "segmented FFT workspace views are not supported yet; use a single-segment workspace or plan-owned workspace"
            ),
            Self::SegmentedBufferViewUnsupported { usage } => {
                write!(f, "segmented FFT buffer views are not supported for {usage}")
            }
            Self::BufferViewTooSmall { required, actual } => write!(
                f,
                "FFT buffer view is too small: required {required} bytes, got {actual} bytes"
            ),
            Self::BufferViewEmptySegments => {
                write!(f, "FFT buffer view must contain at least one segment")
            }
            Self::BufferSegmentZeroSize { index } => {
                write!(f, "FFT buffer view segment {index} must have non-zero size")
            }
            Self::BufferSegmentOutOfBounds {
                index,
                offset,
                size,
                buffer_size,
            } => write!(
                f,
                "FFT buffer view segment {index} is out of bounds: offset {offset}, size {size}, buffer size {buffer_size}"
            ),
            Self::BufferViewOutOfBounds {
                offset,
                size,
                buffer_size,
            } => write!(
                f,
                "FFT buffer view is out of bounds: offset {offset}, size {size}, buffer size {buffer_size}"
            ),
            Self::BufferViewWindowOutOfRange {
                offset,
                size,
                length,
            } => write!(
                f,
                "FFT buffer view window is out of range: offset {offset}, size {size}, logical length {length}"
            ),
            Self::BufferViewCopyUnaligned {
                offset,
                size,
                alignment,
            } => write!(
                f,
                "FFT buffer view copy range is not {alignment}-byte aligned: offset {offset}, size {size}"
            ),
            Self::BufferViewMissingUsage { usage } => {
                write!(f, "FFT buffer view segment is missing required {usage} usage")
            }
            Self::BufferViewOffsetUnaligned { offset, alignment } => write!(
                f,
                "FFT buffer view offset {offset} is not aligned to storage-buffer alignment {alignment}"
            ),
            Self::LargeChunkUnsupported {
                reason,
                bytes_per_batch,
                max_bind_bytes,
            } => write!(
                f,
                "large-route execution is unsupported: {reason} (bytes per batch {bytes_per_batch}, max binding bytes {max_bind_bytes})"
            ),
            Self::LargeRouteWorkspaceUnsupported { route_mode } => write!(
                f,
                "caller-owned workspace is not supported for {route_mode} FFT execution"
            ),
            Self::LargeRouteLayoutUnsupported { route_mode, layout } => write!(
                f,
                "{layout} layout is not supported for {route_mode} FFT execution"
            ),
            Self::LargeRouteUnsupported {
                route_mode,
                reason_codes,
            } => write!(
                f,
                "FFT plan requires unsupported large-buffer route {route_mode} (reasons: {})",
                reason_codes.join(",")
            ),
            Self::LargeGraphStageUnsupported { stage, reason } => write!(
                f,
                "large FFT graph stage {stage} is unsupported: {reason}"
            ),
            Self::WindowScheduleUnsupported {
                reason,
                requested_bytes,
                max_bind_bytes,
            } => {
                let limit_label = if reason.contains("max buffer size") {
                    "max buffer size"
                } else {
                    "max binding bytes"
                };
                write!(
                    f,
                    "large FFT window schedule is unsupported: {reason} (requested {requested_bytes} bytes, {limit_label} {max_bind_bytes})"
                )
            }
            Self::HelperBufferTooLarge {
                helper_buffer,
                requested_bytes,
                max_buffer_size,
            } => write!(
                f,
                "FFT helper buffer {helper_buffer} is too large: requested {requested_bytes} bytes, max buffer size {max_buffer_size}"
            ),
            Self::LargeBridgeUnsupported { route, reason } => {
                write!(f, "large FFT {route} bridge is unsupported: {reason}")
            }
            Self::OutOfCoreExecutionUnsupported { reason } => {
                write!(f, "out-of-core FFT execution is unsupported: {reason}")
            }
        }
    }
}

impl fmt::Display for FftExecutionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.error)
    }
}

impl fmt::Display for FftPlanCreationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.error)
    }
}

impl FftError {
    pub fn diagnostics(&self) -> FftDiagnostics {
        if let Self::LargeRouteUnsupported {
            route_mode,
            reason_codes,
        } = self
        {
            let mode = large_route_mode_for_str(route_mode);
            let blocker = large_route_blocker(mode, reason_codes);
            return FftDiagnostics::new(FftRouteSummary::from_large_route_error(
                mode,
                reason_codes,
            ))
            .with_blocker(blocker);
        }

        if let Self::LargeBridgeUnsupported { route, reason } = self {
            return large_bridge_diagnostics(route, reason);
        }

        if let Self::OutOfCoreExecutionUnsupported { reason } = self {
            return out_of_core_diagnostics(reason);
        }

        let blocker = match self {
            Self::ZeroLength => FftBlocker::new(FftBlockerKind::Validation, self.to_string())
                .with_route("plan-config")
                .with_stage("config-shape")
                .with_layout("length")
                .with_required_bytes(1)
                .with_actual_bytes(0),
            Self::ZeroBatch => FftBlocker::new(FftBlockerKind::Validation, self.to_string())
                .with_route("plan-config")
                .with_stage("config-batch")
                .with_layout("batch")
                .with_required_bytes(1)
                .with_actual_bytes(0),
            Self::LengthTooLarge { len } => {
                FftBlocker::new(FftBlockerKind::DeviceLimit, self.to_string())
                    .with_route("plan-config")
                    .with_stage("dispatch-dimensions")
                    .with_layout("workgroup grid")
                    .with_required_bytes(*len as u64)
                    .with_limit_bytes(u32::MAX as u64)
            }
            Self::DispatchWorkgroupsUnsupported { .. } => {
                FftBlocker::new(FftBlockerKind::DeviceLimit, self.to_string())
                    .with_route("dispatch-split")
                    .with_stage("dispatch-dimensions")
                    .with_layout("workgroup grid")
            }
            Self::EmptyAxes => FftBlocker::new(FftBlockerKind::Validation, self.to_string())
                .with_route("axis-policy")
                .with_stage("axis-policy")
                .with_layout("axes")
                .with_required_bytes(1)
                .with_actual_bytes(0),
            Self::InvalidAxis { axis, rank } => {
                FftBlocker::new(FftBlockerKind::Validation, self.to_string())
                    .with_route("axis-policy")
                    .with_stage("axis-policy")
                    .with_layout("axes")
                    .with_required_bytes(*rank as u64)
                    .with_actual_bytes(*axis as u64)
            }
            Self::DuplicateAxis { axis } => {
                FftBlocker::new(FftBlockerKind::Validation, self.to_string())
                    .with_route("axis-policy")
                    .with_stage(format!("axis-{axis}"))
                    .with_layout("axes")
            }
            Self::WorkspaceTooSmall { required, actual } => {
                FftBlocker::new(FftBlockerKind::Workspace, self.to_string())
                    .with_stage("workspace")
                    .with_helper_buffer("workspace")
                    .with_required_bytes(*required)
                    .with_actual_bytes(*actual)
            }
            Self::SegmentedWorkspaceUnsupported => {
                FftBlocker::new(FftBlockerKind::Workspace, self.to_string())
                    .with_stage("workspace")
                    .with_helper_buffer("workspace")
                    .with_layout("segmented")
            }
            Self::RealWorkspaceUnsupported { transform } => {
                FftBlocker::new(FftBlockerKind::Workspace, self.to_string())
                    .with_route(*transform)
                    .with_stage("workspace")
                    .with_helper_buffer("workspace")
            }
            Self::StridedLayoutUnsupported { transform } => {
                FftBlocker::new(FftBlockerKind::Layout, self.to_string())
                    .with_layout("strided")
                    .with_route(*transform)
                    .with_stage("logical-io-stage")
            }
            Self::StridedSegmentedUnsupported => {
                FftBlocker::new(FftBlockerKind::Layout, self.to_string())
                    .with_route("logical-io")
                    .with_stage("logical-io-stage")
                    .with_layout("segmented+strided")
            }
            Self::SegmentedBufferViewUnsupported { usage } => {
                FftBlocker::new(FftBlockerKind::Layout, self.to_string())
                    .with_route("logical-io")
                    .with_layout("segmented")
                    .with_stage(*usage)
            }
            Self::BufferLayoutZeroStride => {
                FftBlocker::new(FftBlockerKind::Layout, self.to_string())
                    .with_route("logical-io")
                    .with_stage("logical-layout")
                    .with_layout("element-stride")
                    .with_required_bytes(1)
                    .with_actual_bytes(0)
            }
            Self::BufferLayoutBatchStrideTooSmall { required, actual } => {
                FftBlocker::new(FftBlockerKind::Layout, self.to_string())
                    .with_route("logical-io")
                    .with_stage("logical-layout")
                    .with_layout("batch-stride")
                    .with_required_bytes(*required)
                    .with_actual_bytes(*actual)
            }
            Self::BufferLayoutOutOfBounds {
                required_bytes,
                actual_bytes,
            } => FftBlocker::new(FftBlockerKind::Layout, self.to_string())
                .with_route("logical-io")
                .with_stage("logical-view")
                .with_layout("logical layout span")
                .with_required_bytes(*required_bytes)
                .with_actual_bytes(*actual_bytes),
            Self::BufferLayoutTooLarge { value, limit } => {
                FftBlocker::new(FftBlockerKind::DeviceLimit, self.to_string())
                    .with_route("logical-io")
                    .with_stage("logical-layout")
                    .with_layout("logical layout span")
                    .with_required_bytes(*value)
                    .with_limit_bytes(*limit)
            }
            Self::BufferViewTooSmall { required, actual } => {
                FftBlocker::new(FftBlockerKind::Validation, self.to_string())
                    .with_route("logical-io")
                    .with_stage("buffer-view")
                    .with_layout("logical view")
                    .with_required_bytes(*required)
                    .with_actual_bytes(*actual)
            }
            Self::BufferViewEmptySegments => {
                FftBlocker::new(FftBlockerKind::Layout, self.to_string())
                    .with_route("logical-io")
                    .with_stage("buffer-view-segments")
                    .with_layout("segmented")
                    .with_required_bytes(1)
                    .with_actual_bytes(0)
            }
            Self::BufferSegmentZeroSize { index } => {
                FftBlocker::new(FftBlockerKind::Layout, self.to_string())
                    .with_route("logical-io")
                    .with_stage(format!("buffer-segment-{index}"))
                    .with_layout("segmented")
                    .with_required_bytes(1)
                    .with_actual_bytes(0)
            }
            Self::BufferSegmentOutOfBounds {
                index,
                offset,
                size,
                buffer_size,
            } => FftBlocker::new(FftBlockerKind::Layout, self.to_string())
                .with_route("logical-io")
                .with_stage(format!("buffer-segment-{index}"))
                .with_layout("physical segment")
                .with_required_bytes(offset.saturating_add(*size))
                .with_actual_bytes(*buffer_size),
            Self::BufferViewOutOfBounds {
                offset,
                size,
                buffer_size,
            } => FftBlocker::new(FftBlockerKind::Layout, self.to_string())
                .with_route("logical-io")
                .with_stage("buffer-view")
                .with_layout("physical span")
                .with_required_bytes(offset.saturating_add(*size))
                .with_actual_bytes(*buffer_size),
            Self::BufferViewWindowOutOfRange {
                offset,
                size,
                length,
            } => FftBlocker::new(FftBlockerKind::Layout, self.to_string())
                .with_route("logical-io")
                .with_stage("buffer-view-window")
                .with_layout("logical window")
                .with_required_bytes(offset.saturating_add(*size))
                .with_actual_bytes(*length),
            Self::BufferViewMissingUsage { usage } => buffer_usage_blocker(usage, self.to_string()),
            Self::BufferViewCopyUnaligned {
                offset,
                size,
                alignment,
            } => FftBlocker::new(FftBlockerKind::Alignment, self.to_string())
                .with_route("logical-io")
                .with_required_bytes(*alignment)
                .with_actual_bytes(offset.saturating_add(*size))
                .with_layout("copy range")
                .with_stage("copy-window"),
            Self::BufferViewOffsetUnaligned { offset, alignment } => {
                FftBlocker::new(FftBlockerKind::Alignment, self.to_string())
                    .with_route("logical-io")
                    .with_actual_bytes(*offset)
                    .with_required_bytes(*alignment)
                    .with_layout("storage binding")
                    .with_stage("storage-window")
            }
            Self::LargeChunkUnsupported {
                reason,
                bytes_per_batch,
                max_bind_bytes,
            } => large_chunk_blocker(reason, *bytes_per_batch, *max_bind_bytes, self.to_string()),
            Self::LargeRouteWorkspaceUnsupported { route_mode } => {
                FftBlocker::new(FftBlockerKind::Workspace, self.to_string())
                    .with_route(*route_mode)
                    .with_stage("workspace")
                    .with_helper_buffer("workspace")
            }
            Self::LargeRouteLayoutUnsupported { route_mode, layout } => {
                FftBlocker::new(FftBlockerKind::Layout, self.to_string())
                    .with_route(*route_mode)
                    .with_stage("large-route-layout")
                    .with_layout(*layout)
            }
            Self::LargeRouteUnsupported { .. } => unreachable!("handled above"),
            Self::LargeGraphStageUnsupported { stage, reason } => {
                large_graph_stage_blocker(stage, reason)
            }
            Self::WindowScheduleUnsupported {
                reason,
                requested_bytes,
                max_bind_bytes,
            } => {
                let mut blocker = FftBlocker::new(FftBlockerKind::DeviceLimit, *reason)
                    .with_stage(window_schedule_stage_for_reason(reason))
                    .with_required_bytes(*requested_bytes)
                    .with_limit_bytes(*max_bind_bytes);
                if let Some(layout) = window_schedule_layout_for_reason(reason) {
                    blocker = blocker.with_layout(layout);
                }
                blocker
            }
            Self::HelperBufferTooLarge {
                helper_buffer,
                requested_bytes,
                max_buffer_size,
            } => {
                let mut blocker = FftBlocker::new(FftBlockerKind::HelperBuffer, self.to_string())
                    .with_helper_buffer(*helper_buffer)
                    .with_stage(helper_stage_for_label(helper_buffer))
                    .with_required_bytes(*requested_bytes)
                    .with_limit_bytes(*max_buffer_size);
                if let Some(route) = helper_route_for_label(helper_buffer) {
                    blocker = blocker.with_route(route);
                }
                blocker
            }
            Self::LargeBridgeUnsupported { .. } => unreachable!("handled above"),
            Self::OutOfCoreExecutionUnsupported { .. } => unreachable!("handled above"),
            Self::UnsupportedAxisKind { axis, len, kind } => {
                FftBlocker::new(FftBlockerKind::Route, self.to_string())
                    .with_route(*kind)
                    .with_stage(format!("axis-{axis}"))
                    .with_layout("axis-kind")
                    .with_required_bytes(*len as u64)
            }
            Self::UnsupportedRealAxes { expected, actual } => {
                FftBlocker::new(FftBlockerKind::Unsupported, self.to_string())
                    .with_route("real")
                    .with_stage("real-axis-policy")
                    .with_layout("real-axes")
                    .with_required_bytes(expected.len() as u64)
                    .with_actual_bytes(actual.len() as u64)
            }
            Self::InvalidRealTransformDirection { transform, .. } => {
                FftBlocker::new(FftBlockerKind::Unsupported, self.to_string())
                    .with_route(*transform)
                    .with_stage("direction")
                    .with_layout("transform-direction")
            }
            Self::UnsupportedLength { len } => {
                FftBlocker::new(FftBlockerKind::Unsupported, self.to_string())
                    .with_route("mixed-radix")
                    .with_stage("mixed-radix-factorization")
                    .with_layout("length")
                    .with_required_bytes(*len as u64)
            }
        };
        let route = route_summary_for_blocker(&blocker);
        FftDiagnostics::new(route).with_blocker(blocker)
    }

    pub fn diagnostics_for_transform(&self, transform: &'static str) -> FftDiagnostics {
        self.diagnostics().with_transform(transform)
    }
}

fn large_route_mode_for_str(route_mode: &str) -> crate::runtime::large_policy::LargeRouteMode {
    match route_mode {
        "normal" => crate::runtime::large_policy::LargeRouteMode::Normal,
        "large-chunk" => crate::runtime::large_policy::LargeRouteMode::LargeChunk,
        _ => crate::runtime::large_policy::LargeRouteMode::LargeOutOfCore,
    }
}

fn buffer_usage_blocker(usage: &'static str, reason: String) -> FftBlocker {
    let mut blocker = FftBlocker::new(FftBlockerKind::BufferUsage, reason);
    match usage {
        "STORAGE" => {
            blocker = blocker
                .with_stage("storage-window")
                .with_layout("storage binding");
        }
        "COPY_SRC" => {
            blocker = blocker.with_stage("copy-window").with_layout("copy source");
        }
        "COPY_DST" => {
            blocker = blocker
                .with_stage("copy-window")
                .with_layout("copy destination");
        }
        _ => {
            blocker = blocker.with_stage(usage);
        }
    }
    blocker
}

fn large_chunk_blocker(
    reason: &'static str,
    bytes_per_batch: u64,
    max_bind_bytes: u64,
    display_reason: String,
) -> FftBlocker {
    let mut blocker = FftBlocker::new(large_chunk_blocker_kind(reason), display_reason)
        .with_route("large-chunk")
        .with_stage(large_chunk_stage_for_reason(reason))
        .with_required_bytes(bytes_per_batch)
        .with_limit_bytes(max_bind_bytes);
    if let Some(layout) = large_chunk_layout_for_reason(reason) {
        blocker = blocker.with_layout(layout);
    }
    if let Some(helper) = large_chunk_helper_for_reason(reason) {
        blocker = blocker.with_helper_buffer(helper);
    }
    blocker
}

fn large_chunk_blocker_kind(reason: &str) -> FftBlockerKind {
    if reason.contains("must be greater than zero") {
        FftBlockerKind::Validation
    } else if reason.contains("copy-aligned") || reason.contains("alignment") {
        FftBlockerKind::Alignment
    } else if reason.contains("workspace") || reason.contains("staging") {
        FftBlockerKind::HelperBuffer
    } else {
        FftBlockerKind::DeviceLimit
    }
}

fn large_chunk_stage_for_reason(reason: &str) -> &'static str {
    if reason.contains("child plan") || reason.contains("child route") {
        "large-chunk-child-route"
    } else if reason.contains("range") || reason.contains("overflowed") {
        "large-chunk-range"
    } else if reason.contains("decomposition")
        || reason.contains("axis")
        || reason.contains("smooth")
    {
        "decomposition-stage-graph"
    } else if reason.contains("staging") {
        "large-chunk-staging"
    } else if reason.contains("batch") {
        "large-batch-window"
    } else {
        "large-route-selection"
    }
}

fn large_chunk_layout_for_reason(reason: &str) -> Option<&'static str> {
    if reason.contains("storage binding") || reason.contains("binding limit") {
        Some("storage binding")
    } else if reason.contains("max buffer") {
        Some("buffer allocation")
    } else if reason.contains("range") || reason.contains("overflowed") {
        Some("chunk range")
    } else if reason.contains("copy-aligned") {
        Some("copy range")
    } else if reason.contains("workspace") {
        Some("workspace")
    } else if reason.contains("staging") {
        Some("staging buffer")
    } else {
        None
    }
}

fn large_chunk_helper_for_reason(reason: &str) -> Option<&'static str> {
    if reason.contains("workspace") {
        Some("large-chunk-workspace")
    } else if reason.contains("staging") {
        Some("large-chunk-staging")
    } else {
        None
    }
}

fn out_of_core_diagnostics(reason: &'static str) -> FftDiagnostics {
    let blocker = FftBlocker::new(FftBlockerKind::Route, reason)
        .with_route("large-out-of-core")
        .with_stage("out-of-core-execution")
        .with_layout("segmented full-volume GPU execution")
        .with_helper_buffer("segmented-full-volume");
    FftDiagnostics::new(FftRouteSummary {
        transform: "unknown",
        route: "large-out-of-core".to_owned(),
        large_route_mode: Some("large-out-of-core".to_owned()),
        execution_kind: Some("out-of-core-unsupported".to_owned()),
        reason_codes: vec!["out-of-core-execution-unsupported".to_owned()],
        attempted_routes: vec!["out-of-core-four-step".to_owned()],
    })
    .with_blocker(blocker)
}

fn window_schedule_stage_for_reason(reason: &str) -> &'static str {
    if reason.contains("real helper") {
        "real-helper-window"
    } else if reason.contains("out-of-core") {
        "out-of-core-staging-window"
    } else if reason.contains("decomposition")
        || reason.contains("smooth")
        || reason.contains("axis-line")
    {
        "decomposition-stage-graph"
    } else if reason.contains("copy range") {
        "copy-window"
    } else if reason.contains("large graph") || reason.contains("stage logical range") {
        "stage-graph"
    } else if reason.contains("storage binding")
        || reason.contains("storage-buffer binding")
        || reason.contains("binding limit")
        || reason.contains("binding limits")
    {
        "storage-window"
    } else if reason.contains("device limit") {
        "device-limits"
    } else {
        "window-schedule"
    }
}

fn window_schedule_layout_for_reason(reason: &str) -> Option<&'static str> {
    if reason.contains("logical range") {
        Some("logical range")
    } else if reason.contains("storage binding")
        || reason.contains("storage-buffer binding")
        || reason.contains("binding limit")
        || reason.contains("binding limits")
    {
        Some("storage binding")
    } else if reason.contains("copy range") {
        Some("copy range")
    } else if reason.contains("staging") {
        Some("staging window")
    } else if reason.contains("device limit") {
        Some("device limits")
    } else {
        None
    }
}

fn large_graph_stage_blocker(stage: &'static str, reason: &'static str) -> FftBlocker {
    let mut blocker = FftBlocker::new(FftBlockerKind::Unsupported, reason).with_stage(stage);
    if stage.contains("c2c-smooth") || reason.contains("smooth") || reason.contains("axis-line") {
        blocker = blocker.with_route("smooth-decomposition");
    } else if stage.contains("rader") || reason.contains("Rader") {
        blocker = blocker.with_route("rader");
    } else if stage.contains("real") {
        blocker = blocker.with_route("real");
    } else if stage.contains("logical") || reason.contains("segmented+strided") {
        blocker = blocker.with_route("logical-io");
    } else if stage.contains("large-axis-sequence") || reason.contains("AxisSequence") {
        blocker = blocker.with_route("axis-sequence");
    } else if stage.contains("axis-plan") || reason.contains("AxisPlan") {
        blocker = blocker.with_route("mixed-radix");
    }
    if reason.contains("logical range") {
        blocker = blocker
            .with_layout("logical range")
            .with_required_bytes(1)
            .with_actual_bytes(0);
    } else if stage.contains("real-strided") {
        blocker = blocker.with_layout("strided logical io");
    } else if stage.contains("shader-key") || reason.contains("shader key") {
        blocker = blocker.with_layout("stage graph");
    } else if reason.contains("segmented+strided") {
        blocker = blocker.with_layout("segmented+strided");
    } else if stage.contains("axis-plan")
        || stage.contains("buffer-flow")
        || reason.contains("buffer flow")
        || reason.contains("temp storage")
    {
        blocker = blocker.with_layout("stage graph");
        if reason.contains("temp storage") {
            let helper = if stage.contains("large-axis-sequence") {
                "large-axis-sequence-workspace"
            } else if stage.contains("axis-sequence") {
                "axis-sequence-workspace"
            } else if stage.contains("smooth-decomposition") {
                "smooth-decomposition-workspace"
            } else {
                "axis-plan-temp"
            };
            blocker = blocker.with_helper_buffer(helper);
        }
    } else if reason.contains("work item") {
        blocker = blocker
            .with_layout("stage graph")
            .with_required_bytes(1)
            .with_actual_bytes(0);
    } else if reason.contains("label") {
        blocker = blocker.with_layout("stage graph");
    }
    blocker
}

fn large_bridge_diagnostics(route: &str, reason: &str) -> FftDiagnostics {
    let route_label = large_bridge_route_label(route);
    let blocker = FftBlocker::new(large_bridge_blocker_kind(reason), reason)
        .with_route(route_label)
        .with_stage(large_bridge_stage_for_reason(reason));
    let blocker = if let Some(helper) = large_bridge_helper_for_reason(reason) {
        blocker.with_helper_buffer(helper)
    } else {
        blocker
    };

    FftDiagnostics::new(FftRouteSummary {
        transform: "c2c",
        route: route_label.to_owned(),
        large_route_mode: Some("large-chunk".to_owned()),
        execution_kind: Some(route_label.to_owned()),
        reason_codes: vec![large_bridge_reason_code(reason).to_owned()],
        attempted_routes: large_bridge_attempted_routes(route_label),
    })
    .with_blocker(blocker)
}

fn large_bridge_route_label(route: &str) -> &'static str {
    match route {
        "rader" | "rader-bridge" => "rader-bridge",
        "bluestein" | "bluestein-bridge" => "bluestein-bridge",
        _ => "large-bridge",
    }
}

fn large_bridge_blocker_kind(reason: &str) -> FftBlockerKind {
    match large_bridge_reason_code(reason) {
        "convolution-dispatch-u32-limit"
        | "convolution-window-unschedulable"
        | "convolution-child-route-unschedulable" => FftBlockerKind::DeviceLimit,
        "helper-buffer-exceeds-max-buffer" | "per-line-staging-exceeds-max-buffer" => {
            FftBlockerKind::HelperBuffer
        }
        _ => FftBlockerKind::Route,
    }
}

fn large_bridge_stage_for_reason(reason: &str) -> &'static str {
    if reason.contains("pipeline key") {
        return "large-bridge-pipeline-key";
    }
    match large_bridge_reason_code(reason) {
        "multi-axis-unsupported" => "large-bridge-axis-selection",
        "rader-prime-axis-required" | "bluestein-axis-too-short" => "large-bridge-axis-policy",
        "convolution-dispatch-u32-limit" => "large-bridge-convolution-dispatch",
        "convolution-window-unschedulable" => "large-bridge-convolution-window",
        "convolution-child-route-unschedulable" => "large-bridge-convolution-child-route",
        "helper-buffer-exceeds-max-buffer" | "per-line-staging-exceeds-max-buffer" => {
            "large-bridge-helper-windows"
        }
        _ => "large-bridge-route-selection",
    }
}

fn large_bridge_helper_for_reason(reason: &str) -> Option<&'static str> {
    match large_bridge_reason_code(reason) {
        "helper-buffer-exceeds-max-buffer" => Some("large-bridge-helper"),
        "per-line-staging-exceeds-max-buffer" => Some("large-bridge-staging"),
        "convolution-window-unschedulable" | "convolution-child-route-unschedulable" => {
            Some("large-bridge-convolution")
        }
        _ => None,
    }
}

fn large_bridge_reason_code(reason: &str) -> &'static str {
    if reason.contains("non-bridge shader stage") {
        return "bridge-pipeline-key-invalid";
    }
    if reason.contains("pipeline key") {
        return "bridge-pipeline-key-missing";
    }
    match reason {
        "large Rader/Bluestein bridge V1 supports one transformed axis" => "multi-axis-unsupported",
        "Rader bridge requires a prime axis length" => "rader-prime-axis-required",
        "Bluestein bridge requires an axis length above one" => "bluestein-axis-too-short",
        "large bridge convolution dispatch exceeds the current u32 limit" => {
            "convolution-dispatch-u32-limit"
        }
        "large bridge per-line staging exceeds the GPU buffer size limit" => {
            "per-line-staging-exceeds-max-buffer"
        }
        "large bridge helper buffer exceeds the GPU buffer size limit" => {
            "helper-buffer-exceeds-max-buffer"
        }
        "large bridge cannot schedule a legal convolution line window" => {
            "convolution-window-unschedulable"
        }
        "large bridge convolution cannot be routed under the current limits" => {
            "convolution-child-route-unschedulable"
        }
        _ => "large-bridge-unsupported",
    }
}

fn large_bridge_attempted_routes(route_label: &'static str) -> Vec<String> {
    [
        "large-chunk",
        route_label,
        "large-bridge-helper-windows",
        "large-bridge-stage-graph",
    ]
    .into_iter()
    .map(str::to_owned)
    .collect()
}

fn route_summary_for_blocker(blocker: &FftBlocker) -> FftRouteSummary {
    let route = blocker.route.as_deref().unwrap_or("unknown");
    FftRouteSummary::new(transform_for_route(route), route)
}

fn transform_for_route(route: &str) -> &'static str {
    match route {
        "r2c" => "r2c",
        "c2r" => "c2r",
        "real" => "real",
        "logical-io" => "unknown",
        "direct-dft"
        | "mixed-radix"
        | "rader"
        | "bluestein"
        | "axis-sequence"
        | "large-chunk"
        | "rader-bridge"
        | "bluestein-bridge"
        | "smooth-decomposition" => "c2c",
        _ => "unknown",
    }
}

fn helper_route_for_label(label: &str) -> Option<&'static str> {
    if label.contains("large_chunk") {
        Some("large-chunk")
    } else if label.contains("bridge.rader") {
        Some("rader-bridge")
    } else if label.contains("bridge.bluestein") {
        Some("bluestein-bridge")
    } else if label.contains("rader") {
        Some("rader")
    } else if label.contains("bluestein") {
        Some("bluestein")
    } else if label.contains("axis_sequence") {
        Some("axis-sequence")
    } else if label.contains("axis_plan") {
        Some("mixed-radix")
    } else if label.contains("c2c.decompose") || label.contains("c2c.smooth") {
        Some("smooth-decomposition")
    } else if label.contains("r2c") {
        Some("r2c")
    } else if label.contains("c2r") {
        Some("c2r")
    } else if label.contains("real.logical") {
        Some("logical-io")
    } else {
        None
    }
}

fn helper_stage_for_label(label: &str) -> &'static str {
    if label.contains("large_chunk") {
        "large-batch-window"
    } else if label.contains("bridge") {
        "large-bridge-helper-windows"
    } else if label.contains("axis_sequence") {
        "axis-sequence-workspace"
    } else if label.contains("axis_plan") {
        "mixed-radix-workspace"
    } else if label.contains("decompose") || label.contains("smooth") {
        "decomposition-stage-graph"
    } else if label.contains("logical") || label.contains("strided") {
        "logical-io-stage"
    } else {
        "helper-buffer"
    }
}

impl std::error::Error for FftError {}

impl std::error::Error for FftExecutionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.error)
    }
}

impl std::error::Error for FftPlanCreationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diagnostics::FftBlockerKind;

    #[test]
    fn diagnostics_describe_plan_shape_and_axis_blockers() {
        let zero_len = FftError::ZeroLength.diagnostics();
        let blocker = &zero_len.blockers()[0];
        assert_eq!(zero_len.route().route, "plan-config");
        assert_eq!(blocker.kind, FftBlockerKind::Validation);
        assert_eq!(blocker.route.as_deref(), Some("plan-config"));
        assert_eq!(blocker.stage.as_deref(), Some("config-shape"));
        assert_eq!(blocker.layout.as_deref(), Some("length"));
        assert_eq!(blocker.required_bytes, Some(1));
        assert_eq!(blocker.actual_bytes, Some(0));

        let zero_batch = FftError::ZeroBatch.diagnostics();
        let blocker = &zero_batch.blockers()[0];
        assert_eq!(zero_batch.route().route, "plan-config");
        assert_eq!(blocker.kind, FftBlockerKind::Validation);
        assert_eq!(blocker.stage.as_deref(), Some("config-batch"));
        assert_eq!(blocker.layout.as_deref(), Some("batch"));
        assert_eq!(blocker.required_bytes, Some(1));
        assert_eq!(blocker.actual_bytes, Some(0));

        let too_large = FftError::LengthTooLarge {
            len: (u32::MAX as usize) + 1,
        }
        .diagnostics();
        let blocker = &too_large.blockers()[0];
        assert_eq!(too_large.route().route, "plan-config");
        assert_eq!(blocker.kind, FftBlockerKind::DeviceLimit);
        assert_eq!(blocker.stage.as_deref(), Some("dispatch-dimensions"));
        assert_eq!(blocker.layout.as_deref(), Some("workgroup grid"));
        assert_eq!(blocker.required_bytes, Some((u32::MAX as u64) + 1));
        assert_eq!(blocker.limit_bytes, Some(u32::MAX as u64));

        let empty_axes = FftError::EmptyAxes.diagnostics();
        let blocker = &empty_axes.blockers()[0];
        assert_eq!(empty_axes.route().route, "axis-policy");
        assert_eq!(blocker.kind, FftBlockerKind::Validation);
        assert_eq!(blocker.stage.as_deref(), Some("axis-policy"));
        assert_eq!(blocker.layout.as_deref(), Some("axes"));
        assert_eq!(blocker.required_bytes, Some(1));
        assert_eq!(blocker.actual_bytes, Some(0));

        let invalid_axis = FftError::InvalidAxis { axis: 3, rank: 2 }.diagnostics();
        let blocker = &invalid_axis.blockers()[0];
        assert_eq!(invalid_axis.route().route, "axis-policy");
        assert_eq!(blocker.kind, FftBlockerKind::Validation);
        assert_eq!(blocker.stage.as_deref(), Some("axis-policy"));
        assert_eq!(blocker.layout.as_deref(), Some("axes"));
        assert_eq!(blocker.required_bytes, Some(2));
        assert_eq!(blocker.actual_bytes, Some(3));

        let duplicate_axis = FftError::DuplicateAxis { axis: 1 }.diagnostics();
        let blocker = &duplicate_axis.blockers()[0];
        assert_eq!(duplicate_axis.route().route, "axis-policy");
        assert_eq!(blocker.kind, FftBlockerKind::Validation);
        assert_eq!(blocker.stage.as_deref(), Some("axis-1"));
        assert_eq!(blocker.layout.as_deref(), Some("axes"));

        let unsupported_len = FftError::UnsupportedLength { len: 17 }.diagnostics();
        let blocker = &unsupported_len.blockers()[0];
        assert_eq!(unsupported_len.route().transform, "c2c");
        assert_eq!(unsupported_len.route().route, "mixed-radix");
        assert_eq!(blocker.kind, FftBlockerKind::Unsupported);
        assert_eq!(blocker.route.as_deref(), Some("mixed-radix"));
        assert_eq!(blocker.stage.as_deref(), Some("mixed-radix-factorization"));
        assert_eq!(blocker.layout.as_deref(), Some("length"));
        assert_eq!(blocker.required_bytes, Some(17));
    }

    #[test]
    fn diagnostics_describe_layout_blockers() {
        let diagnostics = FftError::StridedLayoutUnsupported { transform: "r2c" }.diagnostics();
        let blocker = &diagnostics.blockers()[0];
        assert_eq!(diagnostics.route().transform, "r2c");
        assert_eq!(diagnostics.route().route, "r2c");
        assert_eq!(blocker.kind, FftBlockerKind::Layout);
        assert_eq!(blocker.route.as_deref(), Some("r2c"));
        assert_eq!(blocker.stage.as_deref(), Some("logical-io-stage"));
        assert_eq!(blocker.layout.as_deref(), Some("strided"));

        let segmented_strided = FftError::StridedSegmentedUnsupported.diagnostics();
        let blocker = &segmented_strided.blockers()[0];
        assert_eq!(segmented_strided.route().route, "logical-io");
        assert_eq!(blocker.kind, FftBlockerKind::Layout);
        assert_eq!(blocker.route.as_deref(), Some("logical-io"));
        assert_eq!(blocker.stage.as_deref(), Some("logical-io-stage"));
        assert_eq!(blocker.layout.as_deref(), Some("segmented+strided"));

        let segmented = FftError::SegmentedBufferViewUnsupported {
            usage: "large graph storage window",
        }
        .diagnostics();
        let blocker = &segmented.blockers()[0];
        assert_eq!(segmented.route().route, "logical-io");
        assert_eq!(blocker.kind, FftBlockerKind::Layout);
        assert_eq!(blocker.route.as_deref(), Some("logical-io"));
        assert_eq!(blocker.layout.as_deref(), Some("segmented"));
        assert_eq!(blocker.stage.as_deref(), Some("large graph storage window"));
    }

    #[test]
    fn diagnostics_describe_logical_layout_stride_blockers() {
        let zero_stride = FftError::BufferLayoutZeroStride.diagnostics();
        let blocker = &zero_stride.blockers()[0];
        assert_eq!(zero_stride.route().route, "logical-io");
        assert_eq!(blocker.kind, FftBlockerKind::Layout);
        assert_eq!(blocker.route.as_deref(), Some("logical-io"));
        assert_eq!(blocker.stage.as_deref(), Some("logical-layout"));
        assert_eq!(blocker.layout.as_deref(), Some("element-stride"));
        assert_eq!(blocker.required_bytes, Some(1));
        assert_eq!(blocker.actual_bytes, Some(0));

        let batch_stride = FftError::BufferLayoutBatchStrideTooSmall {
            required: 16,
            actual: 8,
        }
        .diagnostics();
        let blocker = &batch_stride.blockers()[0];
        assert_eq!(batch_stride.route().route, "logical-io");
        assert_eq!(blocker.kind, FftBlockerKind::Layout);
        assert_eq!(blocker.route.as_deref(), Some("logical-io"));
        assert_eq!(blocker.stage.as_deref(), Some("logical-layout"));
        assert_eq!(blocker.layout.as_deref(), Some("batch-stride"));
        assert_eq!(blocker.required_bytes, Some(16));
        assert_eq!(blocker.actual_bytes, Some(8));

        let out_of_bounds = FftError::BufferLayoutOutOfBounds {
            required_bytes: 128,
            actual_bytes: 64,
        }
        .diagnostics();
        let blocker = &out_of_bounds.blockers()[0];
        assert_eq!(out_of_bounds.route().route, "logical-io");
        assert_eq!(blocker.kind, FftBlockerKind::Layout);
        assert_eq!(blocker.route.as_deref(), Some("logical-io"));
        assert_eq!(blocker.stage.as_deref(), Some("logical-view"));
        assert_eq!(blocker.layout.as_deref(), Some("logical layout span"));
        assert_eq!(blocker.required_bytes, Some(128));
        assert_eq!(blocker.actual_bytes, Some(64));

        let too_large = FftError::BufferLayoutTooLarge {
            value: 1024,
            limit: 512,
        }
        .diagnostics();
        let blocker = &too_large.blockers()[0];
        assert_eq!(too_large.route().route, "logical-io");
        assert_eq!(blocker.kind, FftBlockerKind::DeviceLimit);
        assert_eq!(blocker.route.as_deref(), Some("logical-io"));
        assert_eq!(blocker.stage.as_deref(), Some("logical-layout"));
        assert_eq!(blocker.layout.as_deref(), Some("logical layout span"));
        assert_eq!(blocker.required_bytes, Some(1024));
        assert_eq!(blocker.limit_bytes, Some(512));
    }

    #[test]
    fn diagnostics_describe_buffer_view_range_blockers() {
        let too_small = FftError::BufferViewTooSmall {
            required: 128,
            actual: 64,
        }
        .diagnostics();
        let blocker = &too_small.blockers()[0];
        assert_eq!(too_small.route().route, "logical-io");
        assert_eq!(blocker.kind, FftBlockerKind::Validation);
        assert_eq!(blocker.route.as_deref(), Some("logical-io"));
        assert_eq!(blocker.stage.as_deref(), Some("buffer-view"));
        assert_eq!(blocker.layout.as_deref(), Some("logical view"));
        assert_eq!(blocker.required_bytes, Some(128));
        assert_eq!(blocker.actual_bytes, Some(64));

        let empty = FftError::BufferViewEmptySegments.diagnostics();
        let blocker = &empty.blockers()[0];
        assert_eq!(empty.route().route, "logical-io");
        assert_eq!(blocker.kind, FftBlockerKind::Layout);
        assert_eq!(blocker.route.as_deref(), Some("logical-io"));
        assert_eq!(blocker.stage.as_deref(), Some("buffer-view-segments"));
        assert_eq!(blocker.layout.as_deref(), Some("segmented"));
        assert_eq!(blocker.required_bytes, Some(1));
        assert_eq!(blocker.actual_bytes, Some(0));

        let zero = FftError::BufferSegmentZeroSize { index: 3 }.diagnostics();
        let blocker = &zero.blockers()[0];
        assert_eq!(zero.route().route, "logical-io");
        assert_eq!(blocker.route.as_deref(), Some("logical-io"));
        assert_eq!(blocker.stage.as_deref(), Some("buffer-segment-3"));
        assert_eq!(blocker.layout.as_deref(), Some("segmented"));

        let segment = FftError::BufferSegmentOutOfBounds {
            index: 2,
            offset: 16,
            size: 96,
            buffer_size: 64,
        }
        .diagnostics();
        let blocker = &segment.blockers()[0];
        assert_eq!(segment.route().route, "logical-io");
        assert_eq!(blocker.kind, FftBlockerKind::Layout);
        assert_eq!(blocker.route.as_deref(), Some("logical-io"));
        assert_eq!(blocker.stage.as_deref(), Some("buffer-segment-2"));
        assert_eq!(blocker.layout.as_deref(), Some("physical segment"));
        assert_eq!(blocker.required_bytes, Some(112));
        assert_eq!(blocker.actual_bytes, Some(64));

        let view = FftError::BufferViewOutOfBounds {
            offset: 8,
            size: 128,
            buffer_size: 96,
        }
        .diagnostics();
        let blocker = &view.blockers()[0];
        assert_eq!(view.route().route, "logical-io");
        assert_eq!(blocker.route.as_deref(), Some("logical-io"));
        assert_eq!(blocker.stage.as_deref(), Some("buffer-view"));
        assert_eq!(blocker.layout.as_deref(), Some("physical span"));
        assert_eq!(blocker.required_bytes, Some(136));
        assert_eq!(blocker.actual_bytes, Some(96));

        let window = FftError::BufferViewWindowOutOfRange {
            offset: 32,
            size: 80,
            length: 96,
        }
        .diagnostics();
        let blocker = &window.blockers()[0];
        assert_eq!(window.route().route, "logical-io");
        assert_eq!(blocker.route.as_deref(), Some("logical-io"));
        assert_eq!(blocker.stage.as_deref(), Some("buffer-view-window"));
        assert_eq!(blocker.layout.as_deref(), Some("logical window"));
        assert_eq!(blocker.required_bytes, Some(112));
        assert_eq!(blocker.actual_bytes, Some(96));
    }

    #[test]
    fn diagnostics_describe_real_route_validation_blockers() {
        let direction = FftError::InvalidRealTransformDirection {
            transform: "c2r",
            expected: "inverse",
            actual: "forward",
        }
        .diagnostics();
        let blocker = &direction.blockers()[0];
        assert_eq!(direction.route().transform, "c2r");
        assert_eq!(direction.route().route, "c2r");
        assert_eq!(blocker.kind, FftBlockerKind::Unsupported);
        assert_eq!(blocker.route.as_deref(), Some("c2r"));
        assert_eq!(blocker.stage.as_deref(), Some("direction"));
        assert_eq!(blocker.layout.as_deref(), Some("transform-direction"));

        let workspace = FftError::RealWorkspaceUnsupported { transform: "r2c" }.diagnostics();
        let blocker = &workspace.blockers()[0];
        assert_eq!(workspace.route().transform, "r2c");
        assert_eq!(workspace.route().route, "r2c");
        assert_eq!(blocker.kind, FftBlockerKind::Workspace);
        assert_eq!(blocker.route.as_deref(), Some("r2c"));
        assert_eq!(blocker.stage.as_deref(), Some("workspace"));

        let axes = FftError::UnsupportedRealAxes {
            expected: vec![0, 1],
            actual: vec![1],
        }
        .diagnostics();
        let blocker = &axes.blockers()[0];
        assert_eq!(axes.route().transform, "real");
        assert_eq!(axes.route().route, "real");
        assert_eq!(blocker.kind, FftBlockerKind::Unsupported);
        assert_eq!(blocker.route.as_deref(), Some("real"));
        assert_eq!(blocker.stage.as_deref(), Some("real-axis-policy"));
        assert_eq!(blocker.layout.as_deref(), Some("real-axes"));
        assert_eq!(blocker.required_bytes, Some(2));
        assert_eq!(blocker.actual_bytes, Some(1));
    }

    #[test]
    fn diagnostics_describe_route_policy_blocker_context() {
        let axis = FftError::UnsupportedAxisKind {
            axis: 2,
            len: 97,
            kind: "rader",
        }
        .diagnostics();
        let blocker = &axis.blockers()[0];
        assert_eq!(axis.route().transform, "c2c");
        assert_eq!(axis.route().route, "rader");
        assert_eq!(blocker.kind, FftBlockerKind::Route);
        assert_eq!(blocker.route.as_deref(), Some("rader"));
        assert_eq!(blocker.stage.as_deref(), Some("axis-2"));
        assert_eq!(blocker.layout.as_deref(), Some("axis-kind"));
        assert_eq!(blocker.required_bytes, Some(97));

        let large_layout = FftError::LargeRouteLayoutUnsupported {
            route_mode: "large-chunk",
            layout: "segmented+strided",
        }
        .diagnostics();
        let blocker = &large_layout.blockers()[0];
        assert_eq!(large_layout.route().route, "large-chunk");
        assert_eq!(blocker.kind, FftBlockerKind::Layout);
        assert_eq!(blocker.route.as_deref(), Some("large-chunk"));
        assert_eq!(blocker.stage.as_deref(), Some("large-route-layout"));
        assert_eq!(blocker.layout.as_deref(), Some("segmented+strided"));
    }

    #[test]
    fn diagnostics_describe_missing_usage_window_context() {
        for (usage, stage, layout) in [
            ("STORAGE", "storage-window", "storage binding"),
            ("COPY_SRC", "copy-window", "copy source"),
            ("COPY_DST", "copy-window", "copy destination"),
        ] {
            let diagnostics = FftError::BufferViewMissingUsage { usage }.diagnostics();
            let blocker = &diagnostics.blockers()[0];
            assert_eq!(blocker.kind, FftBlockerKind::BufferUsage);
            assert_eq!(blocker.stage.as_deref(), Some(stage));
            assert_eq!(blocker.layout.as_deref(), Some(layout));
            assert!(blocker.reason.contains(usage));
        }
    }

    #[test]
    fn diagnostics_describe_workspace_blocker_context() {
        let too_small = FftError::WorkspaceTooSmall {
            required: 256,
            actual: 128,
        }
        .diagnostics();
        let blocker = &too_small.blockers()[0];
        assert_eq!(blocker.kind, FftBlockerKind::Workspace);
        assert_eq!(blocker.stage.as_deref(), Some("workspace"));
        assert_eq!(blocker.helper_buffer.as_deref(), Some("workspace"));
        assert_eq!(blocker.required_bytes, Some(256));
        assert_eq!(blocker.actual_bytes, Some(128));

        let segmented = FftError::SegmentedWorkspaceUnsupported.diagnostics();
        let blocker = &segmented.blockers()[0];
        assert_eq!(blocker.kind, FftBlockerKind::Workspace);
        assert_eq!(blocker.stage.as_deref(), Some("workspace"));
        assert_eq!(blocker.helper_buffer.as_deref(), Some("workspace"));
        assert_eq!(blocker.layout.as_deref(), Some("segmented"));

        let real = FftError::RealWorkspaceUnsupported { transform: "c2r" }.diagnostics();
        let blocker = &real.blockers()[0];
        assert_eq!(blocker.kind, FftBlockerKind::Workspace);
        assert_eq!(blocker.route.as_deref(), Some("c2r"));
        assert_eq!(blocker.stage.as_deref(), Some("workspace"));
        assert_eq!(blocker.helper_buffer.as_deref(), Some("workspace"));

        let large = FftError::LargeRouteWorkspaceUnsupported {
            route_mode: "large-chunk",
        }
        .diagnostics();
        let blocker = &large.blockers()[0];
        assert_eq!(blocker.kind, FftBlockerKind::Workspace);
        assert_eq!(blocker.route.as_deref(), Some("large-chunk"));
        assert_eq!(blocker.stage.as_deref(), Some("workspace"));
        assert_eq!(blocker.helper_buffer.as_deref(), Some("workspace"));
    }

    #[test]
    fn diagnostics_describe_device_limit_bytes() {
        let diagnostics = FftError::WindowScheduleUnsupported {
            reason: "storage binding window exceeds device limits",
            requested_bytes: 512,
            max_bind_bytes: 256,
        }
        .diagnostics();
        let blocker = &diagnostics.blockers()[0];
        assert_eq!(blocker.kind, FftBlockerKind::DeviceLimit);
        assert_eq!(blocker.stage.as_deref(), Some("storage-window"));
        assert_eq!(blocker.layout.as_deref(), Some("storage binding"));
        assert_eq!(blocker.required_bytes, Some(512));
        assert_eq!(blocker.limit_bytes, Some(256));

        let copy = FftError::WindowScheduleUnsupported {
            reason: "copy range exceeds max buffer size",
            requested_bytes: 2048,
            max_bind_bytes: 1024,
        }
        .diagnostics();
        let blocker = &copy.blockers()[0];
        assert_eq!(blocker.kind, FftBlockerKind::DeviceLimit);
        assert_eq!(blocker.stage.as_deref(), Some("copy-window"));
        assert_eq!(blocker.layout.as_deref(), Some("copy range"));
        assert_eq!(blocker.required_bytes, Some(2048));
        assert_eq!(blocker.limit_bytes, Some(1024));

        let decomposition = FftError::WindowScheduleUnsupported {
            reason: "C2C decomposition axis-line copy block does not fit the storage binding limit",
            requested_bytes: 4096,
            max_bind_bytes: 2048,
        }
        .diagnostics();
        let blocker = &decomposition.blockers()[0];
        assert_eq!(blocker.stage.as_deref(), Some("decomposition-stage-graph"));
        assert_eq!(blocker.layout.as_deref(), Some("storage binding"));
    }

    #[test]
    fn diagnostics_describe_large_chunk_route_blockers() {
        let binding = FftError::LargeChunkUnsupported {
            reason: "large-chunk bytes per batch exceed storage binding limit",
            bytes_per_batch: 4096,
            max_bind_bytes: 2048,
        }
        .diagnostics();
        let blocker = &binding.blockers()[0];
        assert_eq!(binding.route().transform, "c2c");
        assert_eq!(binding.route().route, "large-chunk");
        assert_eq!(blocker.kind, FftBlockerKind::DeviceLimit);
        assert_eq!(blocker.route.as_deref(), Some("large-chunk"));
        assert_eq!(blocker.stage.as_deref(), Some("large-batch-window"));
        assert_eq!(blocker.layout.as_deref(), Some("storage binding"));
        assert_eq!(blocker.required_bytes, Some(4096));
        assert_eq!(blocker.limit_bytes, Some(2048));

        let allocation = FftError::LargeChunkUnsupported {
            reason: "large-chunk bytes per batch exceed max buffer size",
            bytes_per_batch: 8192,
            max_bind_bytes: 4096,
        }
        .diagnostics();
        let blocker = &allocation.blockers()[0];
        assert_eq!(allocation.route().route, "large-chunk");
        assert_eq!(blocker.kind, FftBlockerKind::DeviceLimit);
        assert_eq!(blocker.stage.as_deref(), Some("large-batch-window"));
        assert_eq!(blocker.layout.as_deref(), Some("buffer allocation"));

        let workspace = FftError::LargeChunkUnsupported {
            reason: "smooth decomposition workspace exceeds max buffer size",
            bytes_per_batch: 16_384,
            max_bind_bytes: 8_192,
        }
        .diagnostics();
        let blocker = &workspace.blockers()[0];
        assert_eq!(workspace.route().route, "large-chunk");
        assert_eq!(blocker.kind, FftBlockerKind::HelperBuffer);
        assert_eq!(blocker.stage.as_deref(), Some("decomposition-stage-graph"));
        assert_eq!(blocker.layout.as_deref(), Some("buffer allocation"));
        assert_eq!(
            blocker.helper_buffer.as_deref(),
            Some("large-chunk-workspace")
        );

        let alignment = FftError::LargeChunkUnsupported {
            reason: "large-chunk bytes per batch must be non-zero and copy-aligned",
            bytes_per_batch: 10,
            max_bind_bytes: 256,
        }
        .diagnostics();
        let blocker = &alignment.blockers()[0];
        assert_eq!(alignment.route().route, "large-chunk");
        assert_eq!(blocker.kind, FftBlockerKind::Alignment);
        assert_eq!(blocker.stage.as_deref(), Some("large-batch-window"));
        assert_eq!(blocker.layout.as_deref(), Some("copy range"));

        let range = FftError::LargeChunkUnsupported {
            reason: "real chunk range offset overflowed u64",
            bytes_per_batch: 4096,
            max_bind_bytes: u64::MAX,
        }
        .diagnostics();
        let blocker = &range.blockers()[0];
        assert_eq!(range.route().route, "large-chunk");
        assert_eq!(blocker.kind, FftBlockerKind::DeviceLimit);
        assert_eq!(blocker.stage.as_deref(), Some("large-chunk-range"));
        assert_eq!(blocker.layout.as_deref(), Some("chunk range"));
    }

    #[test]
    fn diagnostics_describe_large_graph_stage_blocker_context() {
        let label = FftError::LargeGraphStageUnsupported {
            stage: "unknown",
            reason: "stage label must not be empty",
        }
        .diagnostics();
        let blocker = &label.blockers()[0];
        assert_eq!(blocker.kind, FftBlockerKind::Unsupported);
        assert_eq!(blocker.stage.as_deref(), Some("unknown"));
        assert_eq!(blocker.layout.as_deref(), Some("stage graph"));

        let work = FftError::LargeGraphStageUnsupported {
            stage: "stockham-stage",
            reason: "stage work item count must be non-zero",
        }
        .diagnostics();
        let blocker = &work.blockers()[0];
        assert_eq!(blocker.layout.as_deref(), Some("stage graph"));
        assert_eq!(blocker.required_bytes, Some(1));
        assert_eq!(blocker.actual_bytes, Some(0));

        let range = FftError::LargeGraphStageUnsupported {
            stage: "copy-window",
            reason: "stage logical range must not be empty",
        }
        .diagnostics();
        let blocker = &range.blockers()[0];
        assert_eq!(blocker.layout.as_deref(), Some("logical range"));
        assert_eq!(blocker.required_bytes, Some(1));
        assert_eq!(blocker.actual_bytes, Some(0));

        let smooth = FftError::LargeGraphStageUnsupported {
            stage: "c2c-smooth-axis-line-copy",
            reason: "axis-line copy called with non-axis kernel",
        }
        .diagnostics();
        let blocker = &smooth.blockers()[0];
        assert_eq!(smooth.route().transform, "c2c");
        assert_eq!(smooth.route().route, "smooth-decomposition");
        assert_eq!(blocker.route.as_deref(), Some("smooth-decomposition"));
        assert_eq!(blocker.stage.as_deref(), Some("c2c-smooth-axis-line-copy"));

        let rader = FftError::LargeGraphStageUnsupported {
            stage: "rader-pipeline-key",
            reason: "Rader pipeline key has a non-Rader shader stage",
        }
        .diagnostics();
        let blocker = &rader.blockers()[0];
        assert_eq!(rader.route().transform, "c2c");
        assert_eq!(rader.route().route, "rader");
        assert_eq!(blocker.route.as_deref(), Some("rader"));
        assert_eq!(blocker.stage.as_deref(), Some("rader-pipeline-key"));

        let logical = FftError::LargeGraphStageUnsupported {
            stage: "c2c-logical-output-stage",
            reason: "segmented+strided output requires a physical staging buffer",
        }
        .diagnostics();
        let blocker = &logical.blockers()[0];
        assert_eq!(logical.route().route, "logical-io");
        assert_eq!(blocker.route.as_deref(), Some("logical-io"));
        assert_eq!(blocker.stage.as_deref(), Some("c2c-logical-output-stage"));
        assert_eq!(blocker.layout.as_deref(), Some("segmented+strided"));

        let axis_plan = FftError::LargeGraphStageUnsupported {
            stage: "axis-plan-workspace",
            reason: "multi-stage AxisPlan requires temp storage",
        }
        .diagnostics();
        let blocker = &axis_plan.blockers()[0];
        assert_eq!(axis_plan.route().transform, "c2c");
        assert_eq!(axis_plan.route().route, "mixed-radix");
        assert_eq!(blocker.route.as_deref(), Some("mixed-radix"));
        assert_eq!(blocker.stage.as_deref(), Some("axis-plan-workspace"));
        assert_eq!(blocker.layout.as_deref(), Some("stage graph"));
        assert_eq!(blocker.helper_buffer.as_deref(), Some("axis-plan-temp"));

        let normal_sequence = FftError::LargeGraphStageUnsupported {
            stage: "axis-sequence-workspace",
            reason: "multi-step AxisSequencePlan requires temp storage",
        }
        .diagnostics();
        let blocker = &normal_sequence.blockers()[0];
        assert_eq!(normal_sequence.route().route, "axis-sequence");
        assert_eq!(blocker.route.as_deref(), Some("axis-sequence"));
        assert_eq!(blocker.layout.as_deref(), Some("stage graph"));
        assert_eq!(
            blocker.helper_buffer.as_deref(),
            Some("axis-sequence-workspace")
        );

        let large_sequence = FftError::LargeGraphStageUnsupported {
            stage: "large-axis-sequence-workspace",
            reason: "multi-step large AxisSequence requires temp storage",
        }
        .diagnostics();
        let blocker = &large_sequence.blockers()[0];
        assert_eq!(large_sequence.route().route, "axis-sequence");
        assert_eq!(blocker.route.as_deref(), Some("axis-sequence"));
        assert_eq!(blocker.layout.as_deref(), Some("stage graph"));
        assert_eq!(
            blocker.helper_buffer.as_deref(),
            Some("large-axis-sequence-workspace")
        );

        let smooth = FftError::LargeGraphStageUnsupported {
            stage: "smooth-decomposition-workspace",
            reason: "multi-step smooth decomposition requires temp storage",
        }
        .diagnostics();
        let blocker = &smooth.blockers()[0];
        assert_eq!(smooth.route().route, "smooth-decomposition");
        assert_eq!(blocker.route.as_deref(), Some("smooth-decomposition"));
        assert_eq!(blocker.layout.as_deref(), Some("stage graph"));
        assert_eq!(
            blocker.helper_buffer.as_deref(),
            Some("smooth-decomposition-workspace")
        );

        let real_strided = FftError::LargeGraphStageUnsupported {
            stage: "real-strided-kernel-kind",
            reason: "real strided copy kernel kind does not match endpoint format",
        }
        .diagnostics();
        let blocker = &real_strided.blockers()[0];
        assert_eq!(real_strided.route().route, "real");
        assert_eq!(blocker.route.as_deref(), Some("real"));
        assert_eq!(blocker.stage.as_deref(), Some("real-strided-kernel-kind"));
        assert_eq!(blocker.layout.as_deref(), Some("strided logical io"));

        let real_shader = FftError::LargeGraphStageUnsupported {
            stage: "real-shader-key",
            reason: "real pack/unpack shader key requires a non-empty shape",
        }
        .diagnostics();
        let blocker = &real_shader.blockers()[0];
        assert_eq!(real_shader.route().route, "real");
        assert_eq!(blocker.route.as_deref(), Some("real"));
        assert_eq!(blocker.stage.as_deref(), Some("real-shader-key"));
        assert_eq!(blocker.layout.as_deref(), Some("stage graph"));
    }

    #[test]
    fn diagnostics_describe_large_route_attempts_from_plan_failures() {
        let diagnostics = FftError::LargeRouteUnsupported {
            route_mode: "large-out-of-core",
            reason_codes: vec![
                "requires-large-bindings",
                "oversized-line-bindings",
                "axis-line-unsupported",
                "out-of-core-ineligible",
                "bytes-per-batch-exceeds-bind",
            ],
        }
        .diagnostics();

        assert_eq!(diagnostics.route().route, "large-out-of-core");
        assert_eq!(
            diagnostics.route().large_route_mode.as_deref(),
            Some("large-out-of-core")
        );
        assert_eq!(
            diagnostics.route().execution_kind.as_deref(),
            Some("out-of-core-unsupported")
        );
        assert!(diagnostics
            .route()
            .reason_codes
            .contains(&"axis-line-unsupported".to_owned()));
        for attempted in [
            "direct",
            "dispatch-split",
            "batch-chunk",
            "line-slice-or-two-step",
            "out-of-core-four-step",
        ] {
            assert!(diagnostics
                .route()
                .attempted_routes
                .contains(&attempted.to_owned()));
        }
        let blocker = &diagnostics.blockers()[0];
        assert_eq!(blocker.kind, FftBlockerKind::Route);
        assert_eq!(blocker.route.as_deref(), Some("large-out-of-core"));
        assert_eq!(blocker.stage.as_deref(), Some("large-route-selection"));
    }

    #[test]
    fn diagnostics_describe_raw_out_of_core_execution_blocker() {
        let diagnostics = FftError::OutOfCoreExecutionUnsupported {
            reason: "segmented full-volume GPU execution is not implemented",
        }
        .diagnostics();

        assert_eq!(diagnostics.route().route, "large-out-of-core");
        assert_eq!(
            diagnostics.route().large_route_mode.as_deref(),
            Some("large-out-of-core")
        );
        assert_eq!(
            diagnostics.route().execution_kind.as_deref(),
            Some("out-of-core-unsupported")
        );
        assert!(diagnostics
            .route()
            .attempted_routes
            .contains(&"out-of-core-four-step".to_owned()));
        let blocker = &diagnostics.blockers()[0];
        assert_eq!(blocker.kind, FftBlockerKind::Route);
        assert_eq!(blocker.route.as_deref(), Some("large-out-of-core"));
        assert_eq!(blocker.stage.as_deref(), Some("out-of-core-execution"));
        assert_eq!(
            blocker.layout.as_deref(),
            Some("segmented full-volume GPU execution")
        );
        assert_eq!(
            blocker.helper_buffer.as_deref(),
            Some("segmented-full-volume")
        );
    }

    #[test]
    fn diagnostics_describe_large_bridge_route_blockers() {
        let diagnostics = FftError::LargeBridgeUnsupported {
            route: "rader",
            reason: "large Rader/Bluestein bridge V1 supports one transformed axis",
        }
        .diagnostics();
        let blocker = &diagnostics.blockers()[0];
        assert_eq!(diagnostics.route().transform, "c2c");
        assert_eq!(diagnostics.route().route, "rader-bridge");
        assert_eq!(
            diagnostics.route().large_route_mode.as_deref(),
            Some("large-chunk")
        );
        assert_eq!(
            diagnostics.route().execution_kind.as_deref(),
            Some("rader-bridge")
        );
        assert_eq!(
            diagnostics.route().reason_codes,
            vec!["multi-axis-unsupported".to_owned()]
        );
        assert!(diagnostics
            .route()
            .attempted_routes
            .contains(&"large-bridge-stage-graph".to_owned()));
        assert_eq!(blocker.kind, FftBlockerKind::Route);
        assert_eq!(blocker.route.as_deref(), Some("rader-bridge"));
        assert_eq!(
            blocker.stage.as_deref(),
            Some("large-bridge-axis-selection")
        );
        assert_eq!(blocker.helper_buffer, None);
    }

    #[test]
    fn diagnostics_describe_large_bridge_limit_blockers() {
        let diagnostics = FftError::LargeBridgeUnsupported {
            route: "bluestein",
            reason: "large bridge convolution cannot be routed under the current limits",
        }
        .diagnostics();
        let blocker = &diagnostics.blockers()[0];
        assert_eq!(diagnostics.route().route, "bluestein-bridge");
        assert_eq!(
            diagnostics.route().reason_codes,
            vec!["convolution-child-route-unschedulable".to_owned()]
        );
        assert_eq!(blocker.kind, FftBlockerKind::DeviceLimit);
        assert_eq!(blocker.route.as_deref(), Some("bluestein-bridge"));
        assert_eq!(
            blocker.stage.as_deref(),
            Some("large-bridge-convolution-child-route")
        );
        assert_eq!(
            blocker.helper_buffer.as_deref(),
            Some("large-bridge-convolution")
        );
    }

    #[test]
    fn diagnostics_describe_large_bridge_missing_pipeline_key() {
        let diagnostics = FftError::LargeBridgeUnsupported {
            route: "rader",
            reason: "Rader bridge pack pipeline key is missing",
        }
        .diagnostics();
        let blocker = &diagnostics.blockers()[0];
        assert_eq!(diagnostics.route().route, "rader-bridge");
        assert_eq!(
            diagnostics.route().reason_codes,
            vec!["bridge-pipeline-key-missing".to_owned()]
        );
        assert_eq!(blocker.kind, FftBlockerKind::Route);
        assert_eq!(blocker.route.as_deref(), Some("rader-bridge"));
        assert_eq!(blocker.stage.as_deref(), Some("large-bridge-pipeline-key"));
    }

    #[test]
    fn diagnostics_describe_large_bridge_invalid_pipeline_key() {
        let diagnostics = FftError::LargeBridgeUnsupported {
            route: "bluestein",
            reason: "large bridge pipeline key has a non-bridge shader stage",
        }
        .diagnostics();
        let blocker = &diagnostics.blockers()[0];
        assert_eq!(diagnostics.route().route, "bluestein-bridge");
        assert_eq!(
            diagnostics.route().reason_codes,
            vec!["bridge-pipeline-key-invalid".to_owned()]
        );
        assert_eq!(blocker.kind, FftBlockerKind::Route);
        assert_eq!(blocker.route.as_deref(), Some("bluestein-bridge"));
        assert_eq!(blocker.stage.as_deref(), Some("large-bridge-pipeline-key"));
    }

    #[test]
    fn diagnostics_describe_helper_buffer_size_blockers() {
        let diagnostics = FftError::HelperBufferTooLarge {
            helper_buffer: "wgpu_fft.rader.work",
            requested_bytes: 2048,
            max_buffer_size: 1024,
        }
        .diagnostics();
        let blocker = &diagnostics.blockers()[0];
        assert_eq!(diagnostics.route().transform, "c2c");
        assert_eq!(diagnostics.route().route, "rader");
        assert_eq!(blocker.kind, FftBlockerKind::HelperBuffer);
        assert_eq!(blocker.route.as_deref(), Some("rader"));
        assert_eq!(blocker.stage.as_deref(), Some("helper-buffer"));
        assert_eq!(
            blocker.helper_buffer.as_deref(),
            Some("wgpu_fft.rader.work")
        );
        assert_eq!(blocker.required_bytes, Some(2048));
        assert_eq!(blocker.limit_bytes, Some(1024));
    }

    #[test]
    fn diagnostics_infer_helper_buffer_route_and_stage_from_labels() {
        for (label, route, stage) in [
            (
                "wgpu_fft.c2c.large_chunk.input_stage",
                "large-chunk",
                "large-batch-window",
            ),
            (
                "wgpu_fft.c2c.bridge.bluestein.work",
                "bluestein-bridge",
                "large-bridge-helper-windows",
            ),
            (
                "wgpu_fft.axis_sequence.temp",
                "axis-sequence",
                "axis-sequence-workspace",
            ),
            (
                "wgpu_fft.axis_plan.temp",
                "mixed-radix",
                "mixed-radix-workspace",
            ),
            (
                "wgpu_fft.real.logical.strided_input_stage",
                "logical-io",
                "logical-io-stage",
            ),
        ] {
            let diagnostics = FftError::HelperBufferTooLarge {
                helper_buffer: label,
                requested_bytes: 2048,
                max_buffer_size: 1024,
            }
            .diagnostics();
            let blocker = &diagnostics.blockers()[0];
            assert_eq!(blocker.route.as_deref(), Some(route));
            assert_eq!(blocker.stage.as_deref(), Some(stage));
            assert_eq!(blocker.helper_buffer.as_deref(), Some(label));
        }
    }

    #[test]
    fn execution_error_preserves_error_and_structured_diagnostics() {
        let error = FftError::WorkspaceTooSmall {
            required: 128,
            actual: 64,
        };
        let diagnostics = error.diagnostics();
        let execution_error = FftExecutionError::new(error.clone(), diagnostics.clone());

        assert_eq!(execution_error.error(), &error);
        assert_eq!(execution_error.diagnostics(), &diagnostics);
        assert_eq!(execution_error.to_string(), error.to_string());

        let (actual_error, actual_diagnostics) = execution_error.into_parts();
        assert_eq!(actual_error, error);
        assert_eq!(actual_diagnostics, diagnostics);
    }

    #[test]
    fn plan_creation_error_preserves_error_and_transform_diagnostics() {
        let error = FftError::LargeRouteUnsupported {
            route_mode: "large-out-of-core",
            reason_codes: vec!["forced-route-out-of-core"],
        };
        let creation_error = FftPlanCreationError::from_error(error.clone(), "c2c");

        assert_eq!(creation_error.error(), &error);
        assert_eq!(creation_error.diagnostics().route().transform, "c2c");
        assert_eq!(
            creation_error
                .diagnostics()
                .route()
                .large_route_mode
                .as_deref(),
            Some("large-out-of-core")
        );
        assert_eq!(creation_error.to_string(), error.to_string());

        let (actual_error, actual_diagnostics) = creation_error.into_parts();
        assert_eq!(actual_error, error);
        assert_eq!(actual_diagnostics.route().transform, "c2c");
    }
}
