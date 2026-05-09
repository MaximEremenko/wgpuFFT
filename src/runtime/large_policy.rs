use crate::error::{FftError, Result};
use crate::runtime::axis_policy::AxisKind;

const COMPLEX_F32_BYTES: u64 = 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LargeRouteMode {
    Normal,
    LargeChunk,
    LargeOutOfCore,
}

impl LargeRouteMode {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Normal => "normal",
            Self::LargeChunk => "large-chunk",
            Self::LargeOutOfCore => "large-out-of-core",
        }
    }

    const fn priority(self) -> u8 {
        match self {
            Self::Normal => 0,
            Self::LargeChunk => 1,
            Self::LargeOutOfCore => 2,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LargeRoutePreference {
    Auto,
    Chunk,
    OutOfCore,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LargeExecutionKind {
    Normal,
    BatchChunk,
    Smooth1dDecomposition,
    AxisDecomposition,
    RaderBridge,
    BluesteinBridge,
    OutOfCoreFourStep,
    OutOfCoreUnsupported,
}

impl LargeExecutionKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Normal => "normal",
            Self::BatchChunk => "batch-chunk",
            Self::Smooth1dDecomposition => "smooth-1d-decomposition",
            Self::AxisDecomposition => "axis-decomposition",
            Self::RaderBridge => "rader-bridge",
            Self::BluesteinBridge => "bluestein-bridge",
            Self::OutOfCoreFourStep => "out-of-core-four-step",
            Self::OutOfCoreUnsupported => "out-of-core-unsupported",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LargePolicyLimits {
    pub max_storage_buffer_binding_size: u64,
    pub max_buffer_size: u64,
}

impl From<&wgpu::Limits> for LargePolicyLimits {
    fn from(limits: &wgpu::Limits) -> Self {
        Self {
            max_storage_buffer_binding_size: limits.max_storage_buffer_binding_size,
            max_buffer_size: limits.max_buffer_size,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LargeRoutingPolicy {
    pub max_bind_bytes: u64,
    pub max_buffer_size: u64,
    pub needs_large_mode: bool,
    pub oversized_line_mode: bool,
    pub axis_supported: Option<Vec<bool>>,
    pub out_of_core_eligible: bool,
    pub requires_out_of_core: bool,
    pub prefers_out_of_core: bool,
    pub use_out_of_core: bool,
    pub route_mode: LargeRouteMode,
    pub execution_kind: LargeExecutionKind,
    pub reason_codes: Vec<&'static str>,
    pub attempted_routes: Vec<&'static str>,
    pub selected_axis: Option<usize>,
    pub factor_splits: Vec<LargeFactorSplit>,
    pub staging_bytes: Vec<u64>,
    pub unsupported_reason: Option<&'static str>,
}

impl LargeRoutingPolicy {
    pub fn route_mode(&self) -> LargeRouteMode {
        self.route_mode
    }

    pub fn execution_kind(&self) -> LargeExecutionKind {
        self.execution_kind
    }

    pub(crate) fn with_execution_kind(mut self, execution_kind: LargeExecutionKind) -> Self {
        self.execution_kind = execution_kind;
        push_unique(&mut self.reason_codes, execution_kind.as_str());
        self
    }

    pub(crate) fn with_route_mode(mut self, route_mode: LargeRouteMode) -> Self {
        self.route_mode = route_mode;
        self
    }

    pub(crate) fn with_diagnostics(
        mut self,
        selected_axis: Option<usize>,
        factor_splits: Vec<LargeFactorSplit>,
        staging_bytes: Vec<u64>,
        unsupported_reason: Option<&'static str>,
    ) -> Self {
        self.selected_axis = selected_axis;
        self.factor_splits = factor_splits;
        self.staging_bytes = staging_bytes;
        self.unsupported_reason = unsupported_reason;
        self
    }

    pub fn reason_codes(&self) -> &[&'static str] {
        &self.reason_codes
    }

    pub fn attempted_routes(&self) -> &[&'static str] {
        &self.attempted_routes
    }

    pub fn diagnostics(&self) -> LargeRouteDiagnostics {
        LargeRouteDiagnostics {
            route_mode: self.route_mode,
            execution_kind: self.execution_kind,
            selected_axis: self.selected_axis,
            factor_splits: self.factor_splits.clone(),
            staging_bytes: self.staging_bytes.clone(),
            unsupported_reason: self.unsupported_reason,
            reason_codes: self.reason_codes.clone(),
            attempted_routes: self.attempted_routes.clone(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct LargeRoutingPolicyInput<'a> {
    pub limits: LargePolicyLimits,
    pub required_binding_bytes: &'a [u64],
    pub line_bytes: &'a [u64],
    pub axis_kinds: Option<&'a [AxisKind]>,
    pub axis_lengths: Option<&'a [usize]>,
    pub allow_non_mixed_bounded_slicing: bool,
    pub allow_out_of_core: bool,
    pub disable_out_of_core: bool,
    pub rank: usize,
    pub min_out_of_core_rank: usize,
    pub bytes_per_batch: Option<u64>,
    pub has_strided_io: bool,
    pub prefer_out_of_core_for_strided: bool,
    pub requested_large_route: LargeRoutePreference,
}

impl<'a> LargeRoutingPolicyInput<'a> {
    pub fn new(limits: LargePolicyLimits, required_binding_bytes: &'a [u64]) -> Self {
        Self {
            limits,
            required_binding_bytes,
            line_bytes: &[],
            axis_kinds: None,
            axis_lengths: None,
            allow_non_mixed_bounded_slicing: false,
            allow_out_of_core: false,
            disable_out_of_core: false,
            rank: 1,
            min_out_of_core_rank: 2,
            bytes_per_batch: None,
            has_strided_io: false,
            prefer_out_of_core_for_strided: false,
            requested_large_route: LargeRoutePreference::Auto,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LargeRouteMetadata {
    pub route_mode: LargeRouteMode,
    pub execution_kind: LargeExecutionKind,
    pub reason_codes: Vec<&'static str>,
    pub attempted_routes: Vec<&'static str>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LargeFactorSplit {
    pub axis: Option<usize>,
    pub len: u64,
    pub factors: Vec<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LargeRouteDiagnostics {
    pub route_mode: LargeRouteMode,
    pub execution_kind: LargeExecutionKind,
    pub selected_axis: Option<usize>,
    pub factor_splits: Vec<LargeFactorSplit>,
    pub staging_bytes: Vec<u64>,
    pub unsupported_reason: Option<&'static str>,
    pub reason_codes: Vec<&'static str>,
    pub attempted_routes: Vec<&'static str>,
}

impl From<&LargeRoutingPolicy> for LargeRouteMetadata {
    fn from(policy: &LargeRoutingPolicy) -> Self {
        Self {
            route_mode: policy.route_mode,
            execution_kind: policy.execution_kind,
            reason_codes: policy.reason_codes.clone(),
            attempted_routes: policy.attempted_routes.clone(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct OutOfCoreAxisWindowPolicyInput {
    pub axis_len: usize,
    pub line_bytes: u64,
    pub lines_total: usize,
    pub max_bind_bytes: u64,
    pub axis_kind: AxisKind,
    pub storage_align: u64,
    pub swap_to_2_stage_4_step: usize,
    pub swap_to_3_stage_4_step: usize,
    pub grouped_batch: Option<usize>,
    pub out_of_core_burst_windows: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OutOfCoreAxisWindowPolicy {
    pub axis_kind: AxisKind,
    pub axis_len: usize,
    pub line_bytes: u64,
    pub lines_total: usize,
    pub max_lines_by_bind: usize,
    pub grouped_batch: Option<usize>,
    pub num_axis_uploads: usize,
    pub lines_per_chunk: usize,
    pub aligned_line_step: usize,
    pub burst_windows: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct OutOfCorePlanInput {
    pub axis_window: OutOfCoreAxisWindowPolicyInput,
    pub max_buffer_size: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OutOfCoreWindow {
    pub start_line: usize,
    pub line_count: usize,
    pub byte_offset: u64,
    pub byte_size: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OutOfCorePlan {
    pub axis_policy: OutOfCoreAxisWindowPolicy,
    pub upload_windows: Vec<OutOfCoreWindow>,
    pub download_windows: Vec<OutOfCoreWindow>,
    pub staging_buffer_bytes: u64,
    pub max_buffer_size: u64,
}

pub fn resolve_large_routing_policy(
    input: LargeRoutingPolicyInput<'_>,
) -> Result<LargeRoutingPolicy> {
    let max_bind_bytes = input.limits.max_storage_buffer_binding_size;
    let max_buffer_size = input.limits.max_buffer_size;
    let mut reason_codes = Vec::new();
    let mut attempted_routes = vec!["direct"];
    let needs_large_mode = input
        .required_binding_bytes
        .iter()
        .any(|&bytes| bytes > max_bind_bytes);
    let oversized_line_mode = input.line_bytes.iter().any(|&bytes| bytes > max_bind_bytes);

    if needs_large_mode {
        push_unique(&mut reason_codes, "requires-large-bindings");
        push_unique(&mut attempted_routes, "dispatch-split");
        push_unique(&mut attempted_routes, "batch-chunk");
    } else {
        push_unique(&mut reason_codes, "within-bindings");
    }
    if oversized_line_mode {
        push_unique(&mut reason_codes, "oversized-line-bindings");
        push_unique(&mut attempted_routes, "line-slice-or-two-step");
    }

    let axis_supported = evaluate_axis_support(
        input.axis_kinds,
        input.axis_lengths,
        max_bind_bytes,
        max_buffer_size,
        input.allow_non_mixed_bounded_slicing,
    );
    if axis_supported
        .as_ref()
        .is_some_and(|supported| supported.iter().any(|value| !*value))
    {
        push_unique(&mut reason_codes, "axis-line-unsupported");
    }

    let out_of_core_eligible = input.allow_out_of_core
        && !input.disable_out_of_core
        && input.rank >= input.min_out_of_core_rank
        && axis_supported
            .as_ref()
            .is_none_or(|supported| supported.iter().all(|value| *value));
    if out_of_core_eligible {
        push_unique(&mut reason_codes, "out-of-core-eligible");
    } else {
        push_unique(&mut reason_codes, "out-of-core-ineligible");
    }

    let requires_out_of_core = input.allow_out_of_core
        && needs_large_mode
        && input
            .bytes_per_batch
            .is_some_and(|bytes| bytes > max_bind_bytes);
    let prefers_out_of_core = input.allow_out_of_core
        && needs_large_mode
        && input.prefer_out_of_core_for_strided
        && input.has_strided_io
        && out_of_core_eligible;
    if input.allow_out_of_core && needs_large_mode {
        push_unique(&mut attempted_routes, "out-of-core-four-step");
    }
    if requires_out_of_core {
        push_unique(&mut reason_codes, "bytes-per-batch-exceeds-bind");
    }
    if prefers_out_of_core {
        push_unique(&mut reason_codes, "strided-prefers-out-of-core");
    }

    match input.requested_large_route {
        LargeRoutePreference::Auto => {}
        LargeRoutePreference::Chunk if needs_large_mode => {
            push_unique(&mut reason_codes, "forced-route-chunk");
        }
        LargeRoutePreference::OutOfCore if needs_large_mode => {
            push_unique(&mut reason_codes, "forced-route-out-of-core");
            push_unique(&mut attempted_routes, "forced-out-of-core");
        }
        _ => {}
    }

    if needs_large_mode
        && input.requested_large_route == LargeRoutePreference::Chunk
        && requires_out_of_core
    {
        return Err(FftError::LargeRouteUnsupported {
            route_mode: LargeRouteMode::LargeChunk.as_str(),
            reason_codes,
        });
    }
    if needs_large_mode
        && input.requested_large_route == LargeRoutePreference::OutOfCore
        && (!input.allow_out_of_core || !out_of_core_eligible)
    {
        return Err(FftError::LargeRouteUnsupported {
            route_mode: LargeRouteMode::LargeOutOfCore.as_str(),
            reason_codes,
        });
    }
    if requires_out_of_core && !out_of_core_eligible {
        return Err(FftError::LargeRouteUnsupported {
            route_mode: LargeRouteMode::LargeOutOfCore.as_str(),
            reason_codes,
        });
    }

    let use_out_of_core = (needs_large_mode
        && input.requested_large_route == LargeRoutePreference::OutOfCore)
        || requires_out_of_core
        || (input.requested_large_route != LargeRoutePreference::Chunk && prefers_out_of_core);
    let route_mode = if needs_large_mode {
        if use_out_of_core {
            LargeRouteMode::LargeOutOfCore
        } else {
            LargeRouteMode::LargeChunk
        }
    } else {
        LargeRouteMode::Normal
    };
    if route_mode == LargeRouteMode::LargeOutOfCore {
        push_unique(&mut attempted_routes, "out-of-core-four-step");
    }
    push_unique(&mut reason_codes, route_mode.as_str());

    let execution_kind = match route_mode {
        LargeRouteMode::Normal => LargeExecutionKind::Normal,
        LargeRouteMode::LargeChunk => LargeExecutionKind::BatchChunk,
        LargeRouteMode::LargeOutOfCore => LargeExecutionKind::OutOfCoreUnsupported,
    };

    Ok(LargeRoutingPolicy {
        max_bind_bytes,
        max_buffer_size,
        needs_large_mode,
        oversized_line_mode,
        axis_supported,
        out_of_core_eligible,
        requires_out_of_core,
        prefers_out_of_core,
        use_out_of_core,
        route_mode,
        execution_kind,
        reason_codes,
        attempted_routes,
        selected_axis: None,
        factor_splits: Vec::new(),
        staging_bytes: Vec::new(),
        unsupported_reason: None,
    })
}

pub fn merge_large_route_metadata(entries: &[LargeRouteMetadata]) -> LargeRouteMetadata {
    let mut route_mode = LargeRouteMode::Normal;
    let mut reason_codes = Vec::new();
    let mut attempted_routes = Vec::new();

    for entry in entries {
        if entry.route_mode.priority() > route_mode.priority() {
            route_mode = entry.route_mode;
        }
        for &code in &entry.reason_codes {
            push_unique(&mut reason_codes, code);
        }
        for &route in &entry.attempted_routes {
            push_unique(&mut attempted_routes, route);
        }
    }

    if route_mode == LargeRouteMode::LargeOutOfCore {
        push_unique(&mut attempted_routes, "out-of-core-four-step");
    }
    push_unique(&mut reason_codes, route_mode.as_str());

    LargeRouteMetadata {
        route_mode,
        execution_kind: match route_mode {
            LargeRouteMode::Normal => LargeExecutionKind::Normal,
            LargeRouteMode::LargeChunk => LargeExecutionKind::BatchChunk,
            LargeRouteMode::LargeOutOfCore => LargeExecutionKind::OutOfCoreUnsupported,
        },
        reason_codes,
        attempted_routes,
    }
}

pub fn resolve_out_of_core_axis_window_policy(
    input: OutOfCoreAxisWindowPolicyInput,
) -> Result<OutOfCoreAxisWindowPolicy> {
    if input.axis_len == 0 || input.line_bytes == 0 || input.lines_total == 0 {
        return Err(FftError::ZeroLength);
    }

    let max_lines_by_bind = if input.line_bytes <= input.max_bind_bytes {
        (input.max_bind_bytes / input.line_bytes).max(1) as usize
    } else {
        1
    };

    let mut num_axis_uploads = if input.swap_to_3_stage_4_step > 0
        && input.axis_len >= input.swap_to_3_stage_4_step
    {
        3
    } else if input.swap_to_2_stage_4_step > 0 && input.axis_len >= input.swap_to_2_stage_4_step {
        2
    } else if input.axis_kind != AxisKind::Mixed
        && input.axis_len >= 4096
        && max_lines_by_bind >= 16
    {
        3
    } else if input.axis_kind != AxisKind::Mixed && input.axis_len >= 1024 && max_lines_by_bind >= 8
    {
        2
    } else {
        1
    };
    num_axis_uploads = num_axis_uploads.clamp(1, max_lines_by_bind.min(3));

    let mut lines_per_chunk = (max_lines_by_bind / num_axis_uploads).max(1);
    if let Some(grouped_batch) = input.grouped_batch {
        if grouped_batch == 0 {
            return Err(FftError::ZeroBatch);
        }
        if lines_per_chunk > 1 {
            if lines_per_chunk >= grouped_batch {
                lines_per_chunk = (lines_per_chunk / grouped_batch).max(1) * grouped_batch;
            } else {
                lines_per_chunk = 1;
            }
        }
    }

    let storage_align = input.storage_align.max(1);
    let aligned_line_step = (storage_align / gcd(storage_align, input.line_bytes)).max(1) as usize;
    if aligned_line_step > 1 && lines_per_chunk >= aligned_line_step {
        lines_per_chunk = (lines_per_chunk / aligned_line_step).max(1) * aligned_line_step;
    }

    lines_per_chunk = lines_per_chunk.clamp(1, input.lines_total);

    Ok(OutOfCoreAxisWindowPolicy {
        axis_kind: input.axis_kind,
        axis_len: input.axis_len,
        line_bytes: input.line_bytes,
        lines_total: input.lines_total,
        max_lines_by_bind,
        grouped_batch: input.grouped_batch,
        num_axis_uploads,
        lines_per_chunk,
        aligned_line_step,
        burst_windows: input.out_of_core_burst_windows.max(1),
    })
}

pub fn plan_out_of_core_windows(input: OutOfCorePlanInput) -> Result<OutOfCorePlan> {
    let axis_policy = resolve_out_of_core_axis_window_policy(input.axis_window)?;
    let window_bytes = checked_mul_u64(axis_policy.lines_per_chunk as u64, axis_policy.line_bytes)?;
    let staging_buffer_bytes = checked_mul_u64(window_bytes, axis_policy.num_axis_uploads as u64)?;
    if staging_buffer_bytes > input.max_buffer_size {
        return Err(FftError::LargeChunkUnsupported {
            reason: "out-of-core staging windows exceed the GPU buffer size limit",
            bytes_per_batch: staging_buffer_bytes,
            max_bind_bytes: axis_policy.max_lines_by_bind as u64 * axis_policy.line_bytes,
        });
    }

    let mut upload_windows = Vec::new();
    let mut start_line = 0usize;
    while start_line < axis_policy.lines_total {
        let line_count = axis_policy
            .lines_per_chunk
            .min(axis_policy.lines_total - start_line);
        let byte_offset = checked_mul_u64(start_line as u64, axis_policy.line_bytes)?;
        let byte_size = checked_mul_u64(line_count as u64, axis_policy.line_bytes)?;
        if byte_offset % 4 != 0 || byte_size % 4 != 0 {
            return Err(FftError::BufferViewCopyUnaligned {
                offset: byte_offset,
                size: byte_size,
                alignment: 4,
            });
        }
        upload_windows.push(OutOfCoreWindow {
            start_line,
            line_count,
            byte_offset,
            byte_size,
        });
        start_line += line_count;
    }

    Ok(OutOfCorePlan {
        axis_policy,
        download_windows: upload_windows.clone(),
        upload_windows,
        staging_buffer_bytes,
        max_buffer_size: input.max_buffer_size,
    })
}

pub(crate) fn line_bytes_for_axis_len(axis_len: usize) -> Result<u64> {
    (axis_len as u64)
        .checked_mul(COMPLEX_F32_BYTES)
        .ok_or(FftError::LengthTooLarge { len: usize::MAX })
}

fn evaluate_axis_support(
    axis_kinds: Option<&[AxisKind]>,
    axis_lengths: Option<&[usize]>,
    max_bind_bytes: u64,
    max_buffer_size: u64,
    allow_non_mixed_bounded_slicing: bool,
) -> Option<Vec<bool>> {
    let (Some(axis_kinds), Some(axis_lengths)) = (axis_kinds, axis_lengths) else {
        return None;
    };
    if axis_kinds.len() != axis_lengths.len() {
        return None;
    }

    Some(
        axis_kinds
            .iter()
            .zip(axis_lengths)
            .map(|(&kind, &len)| match kind {
                AxisKind::Mixed => {
                    can_axis_len_fit_or_two_step(len, max_bind_bytes, max_buffer_size)
                }
                AxisKind::Rader | AxisKind::Bluestein => {
                    let line_bytes = line_bytes_for_axis_len(len).unwrap_or(u64::MAX);
                    if allow_non_mixed_bounded_slicing {
                        line_bytes <= max_buffer_size
                    } else {
                        line_bytes <= max_bind_bytes
                    }
                }
            })
            .collect(),
    )
}

fn can_axis_len_fit_or_two_step(
    axis_len: usize,
    max_bind_bytes: u64,
    max_buffer_size: u64,
) -> bool {
    let Ok(line_bytes) = line_bytes_for_axis_len(axis_len) else {
        return false;
    };
    if line_bytes <= max_bind_bytes {
        return true;
    }
    if line_bytes > max_buffer_size {
        return false;
    }

    let max_axis_elems = (max_bind_bytes / COMPLEX_F32_BYTES) as usize;
    if max_axis_elems < 2 {
        return false;
    }

    let root = (axis_len as f64).sqrt() as usize;
    for d in 2..=root {
        if axis_len % d != 0 {
            continue;
        }
        let q = axis_len / d;
        if d <= max_axis_elems
            && q <= max_axis_elems
            && crate::runtime::factor_supported_length(d).is_ok()
            && crate::runtime::factor_supported_length(q).is_ok()
        {
            return true;
        }
    }
    false
}

fn push_unique(list: &mut Vec<&'static str>, value: &'static str) {
    if !list.contains(&value) {
        list.push(value);
    }
}

fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    a.max(1)
}

fn checked_mul_u64(a: u64, b: u64) -> Result<u64> {
    a.checked_mul(b)
        .ok_or(FftError::LengthTooLarge { len: usize::MAX })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn limits(max_bind: u64) -> LargePolicyLimits {
        LargePolicyLimits {
            max_storage_buffer_binding_size: max_bind,
            max_buffer_size: 1 << 30,
        }
    }

    #[test]
    fn normal_route_when_all_required_bindings_fit() {
        let p = resolve_large_routing_policy(LargeRoutingPolicyInput {
            limits: limits(256),
            required_binding_bytes: &[64, 128, 256],
            line_bytes: &[64],
            ..LargeRoutingPolicyInput::new(limits(256), &[])
        })
        .unwrap();

        assert!(!p.needs_large_mode);
        assert!(!p.use_out_of_core);
        assert_eq!(p.route_mode, LargeRouteMode::Normal);
        assert_eq!(p.attempted_routes, ["direct"]);
        assert!(p.reason_codes.contains(&"within-bindings"));
        assert!(p.reason_codes.contains(&"normal"));
    }

    #[test]
    fn large_chunk_route_when_large_mode_is_needed() {
        let p = resolve_large_routing_policy(LargeRoutingPolicyInput {
            limits: limits(256),
            required_binding_bytes: &[1024],
            line_bytes: &[64],
            bytes_per_batch: Some(128),
            ..LargeRoutingPolicyInput::new(limits(256), &[])
        })
        .unwrap();

        assert!(p.needs_large_mode);
        assert!(!p.use_out_of_core);
        assert_eq!(p.route_mode, LargeRouteMode::LargeChunk);
        assert!(p.attempted_routes.contains(&"dispatch-split"));
        assert!(p.attempted_routes.contains(&"batch-chunk"));
        assert!(p.reason_codes.contains(&"requires-large-bindings"));
        assert!(p.reason_codes.contains(&"large-chunk"));
    }

    #[test]
    fn route_diagnostics_expose_attempts_and_optional_details() {
        let p = resolve_large_routing_policy(LargeRoutingPolicyInput {
            limits: limits(256),
            required_binding_bytes: &[1024],
            line_bytes: &[512],
            bytes_per_batch: Some(128),
            ..LargeRoutingPolicyInput::new(limits(256), &[])
        })
        .unwrap()
        .with_diagnostics(
            Some(0),
            vec![LargeFactorSplit {
                axis: Some(0),
                len: 64,
                factors: vec![8, 8],
            }],
            vec![256],
            Some("test diagnostic"),
        );

        let diagnostics = p.diagnostics();
        assert_eq!(diagnostics.selected_axis, Some(0));
        assert_eq!(diagnostics.factor_splits[0].factors, [8, 8]);
        assert_eq!(diagnostics.staging_bytes, [256]);
        assert_eq!(diagnostics.unsupported_reason, Some("test diagnostic"));
        assert!(diagnostics.attempted_routes.contains(&"batch-chunk"));
    }

    #[test]
    fn large_out_of_core_route_when_one_batch_exceeds_binding_limit() {
        let p = resolve_large_routing_policy(LargeRoutingPolicyInput {
            limits: limits(256),
            required_binding_bytes: &[4096],
            line_bytes: &[512, 32],
            axis_kinds: Some(&[AxisKind::Mixed, AxisKind::Mixed]),
            axis_lengths: Some(&[64, 4]),
            allow_out_of_core: true,
            rank: 2,
            bytes_per_batch: Some(1024),
            ..LargeRoutingPolicyInput::new(limits(256), &[])
        })
        .unwrap();

        assert!(p.needs_large_mode);
        assert!(p.requires_out_of_core);
        assert!(p.out_of_core_eligible);
        assert!(p.use_out_of_core);
        assert_eq!(p.route_mode, LargeRouteMode::LargeOutOfCore);
        assert!(p.attempted_routes.contains(&"out-of-core-four-step"));
        assert!(p.reason_codes.contains(&"bytes-per-batch-exceeds-bind"));
        assert!(p.reason_codes.contains(&"out-of-core-eligible"));
    }

    #[test]
    fn strided_preference_can_select_out_of_core() {
        let p = resolve_large_routing_policy(LargeRoutingPolicyInput {
            limits: limits(256),
            required_binding_bytes: &[1024],
            line_bytes: &[128, 64],
            axis_kinds: Some(&[AxisKind::Mixed, AxisKind::Mixed]),
            axis_lengths: Some(&[16, 8]),
            allow_out_of_core: true,
            rank: 2,
            bytes_per_batch: Some(128),
            has_strided_io: true,
            prefer_out_of_core_for_strided: true,
            ..LargeRoutingPolicyInput::new(limits(256), &[])
        })
        .unwrap();

        assert!(p.prefers_out_of_core);
        assert!(p.use_out_of_core);
        assert_eq!(p.route_mode, LargeRouteMode::LargeOutOfCore);
        assert!(p.reason_codes.contains(&"strided-prefers-out-of-core"));
    }

    #[test]
    fn rejects_required_out_of_core_when_axis_strategy_is_unsupported() {
        let err = resolve_large_routing_policy(LargeRoutingPolicyInput {
            limits: limits(64),
            required_binding_bytes: &[4096],
            line_bytes: &[136, 32],
            axis_kinds: Some(&[AxisKind::Mixed, AxisKind::Mixed]),
            axis_lengths: Some(&[17, 4]),
            allow_out_of_core: true,
            rank: 2,
            bytes_per_batch: Some(1024),
            ..LargeRoutingPolicyInput::new(limits(64), &[])
        })
        .unwrap_err();

        assert!(matches!(
            err,
            FftError::LargeRouteUnsupported {
                route_mode: "large-out-of-core",
                ..
            }
        ));
    }

    #[test]
    fn forced_chunk_rejects_when_out_of_core_is_required() {
        let err = resolve_large_routing_policy(LargeRoutingPolicyInput {
            limits: limits(64),
            required_binding_bytes: &[4096],
            line_bytes: &[32, 32],
            axis_kinds: Some(&[AxisKind::Mixed, AxisKind::Mixed]),
            axis_lengths: Some(&[4, 4]),
            allow_out_of_core: true,
            rank: 2,
            bytes_per_batch: Some(1024),
            requested_large_route: LargeRoutePreference::Chunk,
            ..LargeRoutingPolicyInput::new(limits(64), &[])
        })
        .unwrap_err();

        assert!(matches!(
            err,
            FftError::LargeRouteUnsupported {
                route_mode: "large-chunk",
                ..
            }
        ));
    }

    #[test]
    fn out_of_core_axis_window_policy_applies_staging_grouping_and_alignment() {
        let p = resolve_out_of_core_axis_window_policy(OutOfCoreAxisWindowPolicyInput {
            axis_len: 4096,
            line_bytes: 264,
            lines_total: 4096,
            max_bind_bytes: 65536,
            axis_kind: AxisKind::Bluestein,
            storage_align: 256,
            swap_to_2_stage_4_step: 1024,
            swap_to_3_stage_4_step: 4096,
            grouped_batch: Some(8),
            out_of_core_burst_windows: 3,
        })
        .unwrap();

        assert_eq!(p.num_axis_uploads, 3);
        assert_eq!(p.grouped_batch, Some(8));
        assert_eq!(p.burst_windows, 3);
        assert!(p.lines_per_chunk >= 1);
        assert!(p.lines_per_chunk <= p.max_lines_by_bind);
        assert_eq!((p.lines_per_chunk as u64 * p.line_bytes) % 256, 0);
    }

    #[test]
    fn out_of_core_plan_emits_upload_and_download_windows() {
        let plan = plan_out_of_core_windows(OutOfCorePlanInput {
            axis_window: OutOfCoreAxisWindowPolicyInput {
                axis_len: 1024,
                line_bytes: 128,
                lines_total: 10,
                max_bind_bytes: 512,
                axis_kind: AxisKind::Mixed,
                storage_align: 256,
                swap_to_2_stage_4_step: 0,
                swap_to_3_stage_4_step: 0,
                grouped_batch: Some(2),
                out_of_core_burst_windows: 2,
            },
            max_buffer_size: 2048,
        })
        .unwrap();

        assert_eq!(plan.axis_policy.lines_per_chunk, 4);
        assert_eq!(plan.staging_buffer_bytes, 512);
        assert_eq!(plan.upload_windows.len(), 3);
        assert_eq!(plan.upload_windows, plan.download_windows);
        assert_eq!(
            plan.upload_windows[2],
            OutOfCoreWindow {
                start_line: 8,
                line_count: 2,
                byte_offset: 1024,
                byte_size: 256,
            }
        );
    }

    #[test]
    fn out_of_core_plan_rejects_unaligned_windows_and_oversized_staging() {
        assert_eq!(
            plan_out_of_core_windows(OutOfCorePlanInput {
                axis_window: OutOfCoreAxisWindowPolicyInput {
                    axis_len: 16,
                    line_bytes: 6,
                    lines_total: 1,
                    max_bind_bytes: 64,
                    axis_kind: AxisKind::Mixed,
                    storage_align: 1,
                    swap_to_2_stage_4_step: 0,
                    swap_to_3_stage_4_step: 0,
                    grouped_batch: None,
                    out_of_core_burst_windows: 1,
                },
                max_buffer_size: 64,
            })
            .unwrap_err(),
            FftError::BufferViewCopyUnaligned {
                offset: 0,
                size: 6,
                alignment: 4,
            }
        );

        assert!(matches!(
            plan_out_of_core_windows(OutOfCorePlanInput {
                axis_window: OutOfCoreAxisWindowPolicyInput {
                    axis_len: 1024,
                    line_bytes: 128,
                    lines_total: 8,
                    max_bind_bytes: 1024,
                    axis_kind: AxisKind::Mixed,
                    storage_align: 1,
                    swap_to_2_stage_4_step: 1024,
                    swap_to_3_stage_4_step: 0,
                    grouped_batch: None,
                    out_of_core_burst_windows: 1,
                },
                max_buffer_size: 256,
            })
            .unwrap_err(),
            FftError::LargeChunkUnsupported { .. }
        ));
    }

    #[test]
    fn merge_metadata_promotes_out_of_core_and_merges_lists() {
        let merged = merge_large_route_metadata(&[
            LargeRouteMetadata {
                route_mode: LargeRouteMode::LargeChunk,
                execution_kind: LargeExecutionKind::BatchChunk,
                reason_codes: vec!["requires-large-bindings", "large-chunk"],
                attempted_routes: vec!["direct", "dispatch-split", "batch-chunk"],
            },
            LargeRouteMetadata {
                route_mode: LargeRouteMode::LargeOutOfCore,
                execution_kind: LargeExecutionKind::OutOfCoreUnsupported,
                reason_codes: vec!["bytes-per-batch-exceeds-bind", "out-of-core-eligible"],
                attempted_routes: vec!["out-of-core-four-step"],
            },
        ]);

        assert_eq!(merged.route_mode, LargeRouteMode::LargeOutOfCore);
        assert!(merged.reason_codes.contains(&"requires-large-bindings"));
        assert!(merged
            .reason_codes
            .contains(&"bytes-per-batch-exceeds-bind"));
        assert!(merged.reason_codes.contains(&"large-out-of-core"));
        assert!(merged.attempted_routes.contains(&"batch-chunk"));
        assert!(merged.attempted_routes.contains(&"out-of-core-four-step"));
    }
}
