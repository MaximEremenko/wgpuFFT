use crate::runtime::large_graph::{
    ElementFormat, LargeExecutionGraph, LargeStage, LargeStageKind, LogicalBufferId,
};
use crate::runtime::large_policy::{LargeExecutionKind, LargeRouteMode, LargeRoutingPolicy};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FftDiagnostics {
    route: FftRouteSummary,
    stages: Vec<FftStageSummary>,
    blockers: Vec<FftBlocker>,
    device_limits: Option<FftDeviceLimits>,
    buffer_requirements: Vec<FftBufferRequirement>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FftRouteSummary {
    pub transform: &'static str,
    pub route: String,
    pub large_route_mode: Option<String>,
    pub execution_kind: Option<String>,
    pub reason_codes: Vec<String>,
    pub attempted_routes: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FftStageSummary {
    pub label: String,
    pub kind: String,
    pub route: String,
    pub required_bytes: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FftBlocker {
    pub kind: FftBlockerKind,
    pub route: Option<String>,
    pub stage: Option<String>,
    pub layout: Option<String>,
    pub helper_buffer: Option<String>,
    pub reason: String,
    pub required_bytes: Option<u64>,
    pub actual_bytes: Option<u64>,
    pub limit_bytes: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FftBlockerKind {
    Validation,
    Route,
    Layout,
    HelperBuffer,
    DeviceLimit,
    BufferUsage,
    Alignment,
    Workspace,
    Unsupported,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FftDeviceLimits {
    pub max_storage_buffer_binding_size: u64,
    pub max_buffer_size: u64,
    pub min_storage_buffer_offset_alignment: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FftBufferRequirement {
    pub role: String,
    pub required_bytes: u64,
    pub format: String,
}

impl FftDiagnostics {
    pub fn new(route: FftRouteSummary) -> Self {
        Self {
            route,
            stages: Vec::new(),
            blockers: Vec::new(),
            device_limits: None,
            buffer_requirements: Vec::new(),
        }
    }

    pub fn route(&self) -> &FftRouteSummary {
        &self.route
    }

    pub fn stages(&self) -> &[FftStageSummary] {
        &self.stages
    }

    pub fn blockers(&self) -> &[FftBlocker] {
        &self.blockers
    }

    pub fn device_limits(&self) -> Option<FftDeviceLimits> {
        self.device_limits
    }

    pub fn buffer_requirements(&self) -> &[FftBufferRequirement] {
        &self.buffer_requirements
    }

    pub(crate) fn with_stage(mut self, stage: FftStageSummary) -> Self {
        self.stages.push(stage);
        self
    }

    pub(crate) fn with_blocker(mut self, blocker: FftBlocker) -> Self {
        self.blockers.push(blocker);
        self
    }

    pub(crate) fn with_device_limits(mut self, limits: FftDeviceLimits) -> Self {
        self.device_limits = Some(limits);
        self
    }

    pub(crate) fn with_buffer_requirement(mut self, requirement: FftBufferRequirement) -> Self {
        self.buffer_requirements.push(requirement);
        self
    }

    pub(crate) fn with_transform(mut self, transform: &'static str) -> Self {
        self.route.transform = transform;
        self
    }
}

impl FftRouteSummary {
    pub fn new(transform: &'static str, route: impl Into<String>) -> Self {
        Self {
            transform,
            route: route.into(),
            large_route_mode: None,
            execution_kind: None,
            reason_codes: Vec::new(),
            attempted_routes: Vec::new(),
        }
    }

    pub(crate) fn from_large_policy(
        transform: &'static str,
        route: impl Into<String>,
        policy: &LargeRoutingPolicy,
    ) -> Self {
        Self {
            transform,
            route: route.into(),
            large_route_mode: Some(policy.route_mode().as_str().to_owned()),
            execution_kind: Some(policy.execution_kind().as_str().to_owned()),
            reason_codes: policy
                .reason_codes()
                .iter()
                .map(|code| (*code).to_owned())
                .collect(),
            attempted_routes: policy
                .attempted_routes()
                .iter()
                .map(|route| (*route).to_owned())
                .collect(),
        }
    }

    pub(crate) fn from_large_route_error(
        mode: LargeRouteMode,
        reason_codes: &[&'static str],
    ) -> Self {
        Self {
            transform: "unknown",
            route: mode.as_str().to_owned(),
            large_route_mode: Some(mode.as_str().to_owned()),
            execution_kind: Some(
                match mode {
                    LargeRouteMode::Normal => LargeExecutionKind::Normal,
                    LargeRouteMode::LargeChunk => LargeExecutionKind::BatchChunk,
                    LargeRouteMode::LargeOutOfCore => LargeExecutionKind::OutOfCoreUnsupported,
                }
                .as_str()
                .to_owned(),
            ),
            reason_codes: reason_codes.iter().map(|code| (*code).to_owned()).collect(),
            attempted_routes: attempted_routes_for_large_route_error(reason_codes),
        }
    }
}

fn attempted_routes_for_large_route_error(reason_codes: &[&'static str]) -> Vec<String> {
    let mut routes = Vec::new();
    push_unique_string(&mut routes, "direct");
    if has_reason(
        reason_codes,
        &[
            "requires-large-bindings",
            "forced-route-chunk",
            "bytes-per-batch-exceeds-bind",
        ],
    ) {
        push_unique_string(&mut routes, "dispatch-split");
        push_unique_string(&mut routes, "batch-chunk");
    }
    if has_reason(
        reason_codes,
        &["oversized-line-bindings", "axis-line-unsupported"],
    ) {
        push_unique_string(&mut routes, "line-slice-or-two-step");
    }
    if has_reason(
        reason_codes,
        &[
            "out-of-core-eligible",
            "out-of-core-ineligible",
            "bytes-per-batch-exceeds-bind",
            "forced-route-out-of-core",
            "strided-prefers-out-of-core",
        ],
    ) {
        push_unique_string(&mut routes, "out-of-core-four-step");
    }
    if reason_codes.contains(&"forced-route-out-of-core") {
        push_unique_string(&mut routes, "forced-out-of-core");
    }
    routes
}

fn has_reason(reason_codes: &[&'static str], needles: &[&'static str]) -> bool {
    needles
        .iter()
        .any(|needle| reason_codes.iter().any(|code| code == needle))
}

fn push_unique_string(values: &mut Vec<String>, value: &'static str) {
    if !values.iter().any(|existing| existing == value) {
        values.push(value.to_owned());
    }
}

impl FftStageSummary {
    pub(crate) fn new(
        label: impl Into<String>,
        kind: impl Into<String>,
        route: impl Into<String>,
        required_bytes: Option<u64>,
    ) -> Self {
        Self {
            label: label.into(),
            kind: kind.into(),
            route: route.into(),
            required_bytes,
        }
    }
}

impl FftBlocker {
    pub fn new(kind: FftBlockerKind, reason: impl Into<String>) -> Self {
        Self {
            kind,
            route: None,
            stage: None,
            layout: None,
            helper_buffer: None,
            reason: reason.into(),
            required_bytes: None,
            actual_bytes: None,
            limit_bytes: None,
        }
    }

    pub(crate) fn with_route(mut self, route: impl Into<String>) -> Self {
        self.route = Some(route.into());
        self
    }

    pub(crate) fn with_stage(mut self, stage: impl Into<String>) -> Self {
        self.stage = Some(stage.into());
        self
    }

    pub(crate) fn with_layout(mut self, layout: impl Into<String>) -> Self {
        self.layout = Some(layout.into());
        self
    }

    pub(crate) fn with_helper_buffer(mut self, helper_buffer: impl Into<String>) -> Self {
        self.helper_buffer = Some(helper_buffer.into());
        self
    }

    pub(crate) fn with_required_bytes(mut self, required_bytes: u64) -> Self {
        self.required_bytes = Some(required_bytes);
        self
    }

    pub(crate) fn with_actual_bytes(mut self, actual_bytes: u64) -> Self {
        self.actual_bytes = Some(actual_bytes);
        self
    }

    pub(crate) fn with_limit_bytes(mut self, limit_bytes: u64) -> Self {
        self.limit_bytes = Some(limit_bytes);
        self
    }
}

impl FftDeviceLimits {
    pub(crate) fn from_policy(policy: &LargeRoutingPolicy) -> Self {
        Self {
            max_storage_buffer_binding_size: policy.max_bind_bytes,
            max_buffer_size: policy.max_buffer_size,
            min_storage_buffer_offset_alignment: 0,
        }
    }

    pub(crate) fn from_device(device: &wgpu::Device) -> Self {
        let limits = device.limits();
        Self {
            max_storage_buffer_binding_size: limits.max_storage_buffer_binding_size,
            max_buffer_size: limits.max_buffer_size,
            min_storage_buffer_offset_alignment: u64::from(
                limits.min_storage_buffer_offset_alignment,
            ),
        }
    }
}

impl FftBufferRequirement {
    pub(crate) fn new(
        role: impl Into<String>,
        required_bytes: u64,
        format: impl Into<String>,
    ) -> Self {
        Self {
            role: role.into(),
            required_bytes,
            format: format.into(),
        }
    }
}

pub(crate) fn stage_summaries_for_route(
    route: impl Into<String>,
    execution_kind: LargeExecutionKind,
    required_bytes: u64,
) -> Vec<FftStageSummary> {
    let route = route.into();
    let mut stages = vec![FftStageSummary::new(
        "logical-io-normalize",
        "logical-io",
        route.clone(),
        Some(required_bytes),
    )];
    match execution_kind {
        LargeExecutionKind::Normal => {
            stages.push(FftStageSummary::new(
                "normal-route",
                "fft-route",
                route,
                Some(required_bytes),
            ));
        }
        LargeExecutionKind::BatchChunk => {
            stages.push(FftStageSummary::new(
                "large-batch-window",
                "window-schedule",
                route.clone(),
                Some(required_bytes),
            ));
            stages.push(FftStageSummary::new(
                "large-batch-stage-graph",
                "stage-graph",
                route,
                Some(required_bytes),
            ));
        }
        LargeExecutionKind::Smooth1dDecomposition | LargeExecutionKind::AxisDecomposition => {
            stages.push(FftStageSummary::new(
                "decomposition-window-schedule",
                "window-schedule",
                route.clone(),
                Some(required_bytes),
            ));
            stages.push(FftStageSummary::new(
                "decomposition-stage-graph",
                "stage-graph",
                route,
                Some(required_bytes),
            ));
        }
        LargeExecutionKind::RaderBridge | LargeExecutionKind::BluesteinBridge => {
            stages.push(FftStageSummary::new(
                "large-bridge-helper-windows",
                "helper-buffer-window",
                route.clone(),
                Some(required_bytes),
            ));
            stages.push(FftStageSummary::new(
                "large-bridge-stage-graph",
                "stage-graph",
                route,
                Some(required_bytes),
            ));
        }
        LargeExecutionKind::OutOfCoreFourStep => {
            stages.push(FftStageSummary::new(
                "four-step-window-schedule",
                "window-schedule",
                route.clone(),
                Some(required_bytes),
            ));
            stages.push(FftStageSummary::new(
                "four-step-stage-graph",
                "stage-graph",
                route,
                Some(required_bytes),
            ));
        }
        LargeExecutionKind::SegmentedFullVolume => {
            stages.push(FftStageSummary::new(
                "segmented-volume-window-schedule",
                "window-schedule",
                route.clone(),
                Some(required_bytes),
            ));
            stages.push(FftStageSummary::new(
                "segmented-volume-stage-graph",
                "stage-graph",
                route,
                Some(required_bytes),
            ));
        }
        LargeExecutionKind::OutOfCoreUnsupported => {
            stages.push(FftStageSummary::new(
                "out-of-core-required",
                "unsupported",
                route,
                Some(required_bytes),
            ));
        }
    }
    stages
}

pub(crate) fn stage_summaries_from_graph(
    route: impl Into<String>,
    graph: &LargeExecutionGraph,
) -> Vec<FftStageSummary> {
    let route = route.into();
    graph
        .stages()
        .iter()
        .map(|stage| {
            let label = stage.label();
            FftStageSummary::new(
                label,
                stage_kind_label(stage.kind()),
                stage_route_for_label(label, &route),
                stage_required_bytes(stage),
            )
        })
        .collect()
}

pub(crate) fn helper_buffer_requirements_from_graph(
    graph: &LargeExecutionGraph,
) -> Vec<FftBufferRequirement> {
    let mut requirements: Vec<(LogicalBufferId, FftBufferRequirement)> = Vec::new();
    for stage in graph.stages() {
        let (label, range) = match stage {
            LargeStage::HelperWindow { label, range }
            | LargeStage::WindowedHelper { label, range } => (label, range),
            _ => continue,
        };
        let role = format!("helper:{label}");
        let format = element_format_label(range.format);
        if let Some(existing) = requirements.iter_mut().find(|(buffer, requirement)| {
            *buffer == range.buffer && requirement.role == role && requirement.format == format
        }) {
            existing.1.required_bytes = existing.1.required_bytes.max(range.size_bytes);
        } else {
            requirements.push((
                range.buffer,
                FftBufferRequirement::new(role, range.size_bytes, format),
            ));
        }
    }
    requirements
        .into_iter()
        .map(|(_, requirement)| requirement)
        .collect()
}

pub(crate) fn stage_route_for_label(label: &str, default_route: &str) -> String {
    let route = if label.starts_with("rader-bridge-") {
        "rader-bridge"
    } else if label.starts_with("bluestein-bridge-") {
        "bluestein-bridge"
    } else if label.starts_with("rader-") {
        "rader"
    } else if label.starts_with("bluestein-") {
        "bluestein"
    } else if label.starts_with("mixed-radix-")
        || label.starts_with("fused-pow2-")
        || label.starts_with("fused-smooth-")
    {
        "mixed-radix"
    } else if label.starts_with("direct-dft") {
        "direct-dft"
    } else if label.starts_with("large-chunk-") {
        "large-chunk"
    } else if label.starts_with("four-step-") || label.starts_with("segmented-volume-") {
        "large-out-of-core"
    } else if label.starts_with("large-axis-sequence-") || label.starts_with("axis-sequence-") {
        "axis-sequence"
    } else if label.starts_with("mixed-axis-") || label.starts_with("smooth-axis-") {
        "smooth-decomposition"
    } else if label.starts_with("r2c-") {
        "r2c"
    } else if label.starts_with("c2r-") {
        "c2r"
    } else {
        default_route
    };
    route.to_owned()
}

fn element_format_label(format: ElementFormat) -> &'static str {
    match format {
        ElementFormat::ComplexF32 => "complex-f32",
        ElementFormat::RealF32 => "real-f32",
        ElementFormat::PackedComplexF32 => "packed-complex-f32",
        ElementFormat::U32 => "u32",
    }
}

fn stage_kind_label(kind: LargeStageKind) -> &'static str {
    match kind {
        LargeStageKind::Copy => "copy",
        LargeStageKind::GatherScatter => "gather-scatter",
        LargeStageKind::HelperWindow => "helper-buffer-window",
        LargeStageKind::WindowedHelper => "windowed-helper-buffer",
        LargeStageKind::Kernel => "kernel",
        LargeStageKind::WindowedKernel => "windowed-kernel",
        LargeStageKind::TwiddleTranspose => "twiddle-transpose",
        LargeStageKind::StripeTranspose => "stripe-transpose",
        LargeStageKind::Permutation => "permutation",
        LargeStageKind::Scale => "scale",
        LargeStageKind::HostWindow => "host-window",
    }
}

fn stage_required_bytes(stage: &LargeStage) -> Option<u64> {
    Some(match stage {
        LargeStage::Copy { src, dst, .. } | LargeStage::GatherScatter { src, dst, .. } => {
            src.size_bytes.max(dst.size_bytes)
        }
        LargeStage::HelperWindow { range, .. }
        | LargeStage::WindowedHelper { range, .. }
        | LargeStage::Scale { range, .. } => range.size_bytes,
        LargeStage::Kernel {
            input,
            output,
            work_items,
            ..
        }
        | LargeStage::WindowedKernel {
            input,
            output,
            work_items,
            ..
        }
        | LargeStage::TwiddleTranspose {
            input,
            output,
            work_items,
            ..
        }
        | LargeStage::StripeTranspose {
            input,
            output,
            work_items,
            ..
        }
        | LargeStage::Permutation {
            input,
            output,
            work_items,
            ..
        } => input.size_bytes.max(output.size_bytes).max(*work_items),
        LargeStage::HostWindow { range, .. } => range.size_bytes,
    })
}

pub(crate) fn large_route_blocker(
    mode: LargeRouteMode,
    reason_codes: &[&'static str],
) -> FftBlocker {
    FftBlocker::new(
        FftBlockerKind::Route,
        format!(
            "route {} was selected but could not execute ({})",
            mode.as_str(),
            reason_codes.join(",")
        ),
    )
    .with_route(mode.as_str())
    .with_stage("large-route-selection")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::large_graph::{
        ElementFormat, LargeExecutionGraph, LargeStage, LogicalBufferId, LogicalRange,
        StageRequirements,
    };

    fn range() -> LogicalRange {
        LogicalRange::new(LogicalBufferId::Temp(0), 0, 128, ElementFormat::ComplexF32).unwrap()
    }

    fn req() -> StageRequirements {
        StageRequirements::new(64, 128, 1, 4, 128).unwrap()
    }

    #[test]
    fn stage_summaries_label_helper_windows_and_infer_embedded_routes() {
        let mut graph = LargeExecutionGraph::new("helpers");
        graph
            .push_stage(
                LargeStage::HelperWindow {
                    label: "rader-work-helper",
                    range: range(),
                },
                req(),
            )
            .unwrap();
        graph
            .push_stage(
                LargeStage::HelperWindow {
                    label: "mixed-radix-workspace",
                    range: range(),
                },
                req(),
            )
            .unwrap();
        graph
            .push_stage(
                LargeStage::Kernel {
                    label: "bluestein-bridge-pack",
                    input: range(),
                    output: range(),
                    work_items: 16,
                },
                req(),
            )
            .unwrap();
        graph
            .push_stage(
                LargeStage::Kernel {
                    label: "r2c-pack",
                    input: range(),
                    output: range(),
                    work_items: 16,
                },
                req(),
            )
            .unwrap();
        graph
            .push_stage(
                LargeStage::HostWindow {
                    label: "logical-output",
                    range: range(),
                },
                req(),
            )
            .unwrap();

        let summaries = stage_summaries_from_graph("axis-sequence", &graph);
        assert_eq!(summaries.len(), 5);
        assert_eq!(summaries[0].label, "rader-work-helper");
        assert_eq!(summaries[0].kind, "helper-buffer-window");
        assert_eq!(summaries[0].route, "rader");
        assert_eq!(summaries[0].required_bytes, Some(128));
        assert_eq!(summaries[1].route, "mixed-radix");
        assert_eq!(summaries[2].route, "bluestein-bridge");
        assert_eq!(summaries[3].route, "r2c");
        assert_eq!(summaries[4].route, "axis-sequence");

        let requirements = helper_buffer_requirements_from_graph(&graph);
        assert!(requirements.iter().any(|requirement| {
            requirement.role == "helper:rader-work-helper"
                && requirement.required_bytes == 128
                && requirement.format == "complex-f32"
        }));
        assert!(requirements.iter().any(|requirement| {
            requirement.role == "helper:mixed-radix-workspace"
                && requirement.required_bytes == 128
                && requirement.format == "complex-f32"
        }));
    }

    #[test]
    fn helper_requirements_keep_distinct_buffers_with_generic_labels() {
        let mut graph = LargeExecutionGraph::new("generic-axis-workspaces");
        for (index, size) in [(4u32, 128u64), (5, 64)] {
            graph
                .push_stage(
                    LargeStage::WindowedHelper {
                        label: "four-step-axis-child-workspace",
                        range: LogicalRange::new(
                            LogicalBufferId::Stage(index),
                            0,
                            size,
                            ElementFormat::ComplexF32,
                        )
                        .unwrap(),
                    },
                    StageRequirements::new(64, 128, 1, 4, size).unwrap(),
                )
                .unwrap();
        }

        let requirements = helper_buffer_requirements_from_graph(&graph);
        let generic = requirements
            .iter()
            .filter(|requirement| requirement.role == "helper:four-step-axis-child-workspace")
            .collect::<Vec<_>>();
        assert_eq!(generic.len(), 2);
        assert_eq!(
            generic
                .iter()
                .map(|requirement| requirement.required_bytes)
                .sum::<u64>(),
            192
        );
    }

    #[test]
    fn generic_fused_stages_are_attributed_to_mixed_radix() {
        assert_eq!(
            stage_route_for_label("fused-pow2-workgroup-stage", "r2c"),
            "mixed-radix"
        );
        assert_eq!(
            stage_route_for_label("fused-smooth-workgroup-stage", "r2c"),
            "mixed-radix"
        );
    }

    #[test]
    fn segmented_volume_stages_are_attributed_to_large_out_of_core() {
        assert_eq!(
            stage_route_for_label("segmented-volume-axis-slab-gather", "axis-sequence"),
            "large-out-of-core"
        );
    }

    #[test]
    fn permutation_summary_preserves_logical_volume() {
        let mut graph = LargeExecutionGraph::new("permutation");
        let input =
            LogicalRange::new(LogicalBufferId::Input, 0, 1024, ElementFormat::ComplexF32).unwrap();
        let output =
            LogicalRange::new(LogicalBufferId::Output, 0, 1024, ElementFormat::ComplexF32).unwrap();
        graph
            .push_stage(
                LargeStage::Permutation {
                    label: "four-step-permute-axis-to-front",
                    input,
                    output,
                    work_items: 128,
                },
                StageRequirements::new(256, 1024, 256, 4, 256).unwrap(),
            )
            .unwrap();

        let summaries = stage_summaries_from_graph("axis-sequence", &graph);
        assert_eq!(summaries.len(), 1);
        assert_eq!(summaries[0].kind, "permutation");
        assert_eq!(summaries[0].route, "large-out-of-core");
        assert_eq!(summaries[0].required_bytes, Some(1024));
    }
}
