use crate::config::FftConfig;
use crate::error::{FftError, Result};

/// Requested large-dataset routing behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
pub enum FftLargeRoute {
    #[default]
    Auto,
    ForceChunk,
    ForceFourStep,
    ForceSegmented,
}

impl FftLargeRoute {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::ForceChunk => "force-chunk",
            Self::ForceFourStep => "force-four-step",
            Self::ForceSegmented => "force-segmented",
        }
    }
}

/// Default of [`FftTuning::fused_workgroup_size`].
pub(crate) const DEFAULT_FUSED_WORKGROUP_SIZE: u32 = 256;

/// Advanced FFT planning and kernel-selection controls.
///
/// Defaults reproduce the library's untuned behavior. Fields are private so
/// future controls can be added without exposing struct-literal construction.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct FftTuning {
    workgroup_size: u32,
    fused_workgroup_size: u32,
    rader_max_prime: usize,
    direct_max_prime: usize,
    force_rader_axes: Vec<usize>,
    force_bluestein_axes: Vec<usize>,
    large_route: FftLargeRoute,
    large_chunk_max_batches: Option<usize>,
    grouped_batch: Option<usize>,
    swap_to_2_stage_4_step: usize,
    swap_to_3_stage_4_step: usize,
    segmented_burst_depth: usize,
    max_storage_buffer_binding_size: Option<u64>,
    max_buffer_size: Option<u64>,
    fused_min_convolution_length: usize,
    fuse_long_axes: bool,
}

impl Default for FftTuning {
    fn default() -> Self {
        Self {
            workgroup_size: 64,
            fused_workgroup_size: DEFAULT_FUSED_WORKGROUP_SIZE,
            rader_max_prime: 4096,
            direct_max_prime: 127,
            force_rader_axes: Vec::new(),
            force_bluestein_axes: Vec::new(),
            large_route: FftLargeRoute::Auto,
            large_chunk_max_batches: None,
            grouped_batch: None,
            swap_to_2_stage_4_step: 0,
            swap_to_3_stage_4_step: 0,
            segmented_burst_depth: 2,
            max_storage_buffer_binding_size: None,
            max_buffer_size: None,
            fused_min_convolution_length: 128,
            fuse_long_axes: true,
        }
    }
}

impl FftTuning {
    pub fn new() -> Self {
        Self::default()
    }

    pub const fn workgroup_size(&self) -> u32 {
        self.workgroup_size
    }

    /// Workgroup size of the fused kernels that transform their lines in
    /// workgroup memory. At its default of 256, `f32` power-of-two lines may
    /// run in register-resident kernels instead, which size their own
    /// workgroups; any other value keeps lines that workgroup memory holds in
    /// the workgroup-memory kernels.
    pub const fn fused_workgroup_size(&self) -> u32 {
        self.fused_workgroup_size
    }

    pub const fn rader_max_prime(&self) -> usize {
        self.rader_max_prime
    }

    /// Largest prime axis transformed by a direct DFT kernel instead of
    /// Rader's convolution (`f32` only); `0` keeps Rader for every prime.
    pub const fn direct_max_prime(&self) -> usize {
        self.direct_max_prime
    }

    pub fn force_rader_axes(&self) -> &[usize] {
        &self.force_rader_axes
    }

    pub fn force_bluestein_axes(&self) -> &[usize] {
        &self.force_bluestein_axes
    }

    pub const fn large_route(&self) -> FftLargeRoute {
        self.large_route
    }

    pub const fn large_chunk_max_batches(&self) -> Option<usize> {
        self.large_chunk_max_batches
    }

    pub const fn grouped_batch(&self) -> Option<usize> {
        self.grouped_batch
    }

    pub const fn swap_to_2_stage_4_step(&self) -> usize {
        self.swap_to_2_stage_4_step
    }

    pub const fn swap_to_3_stage_4_step(&self) -> usize {
        self.swap_to_3_stage_4_step
    }

    pub const fn segmented_burst_depth(&self) -> usize {
        self.segmented_burst_depth
    }

    pub const fn max_storage_buffer_binding_size(&self) -> Option<u64> {
        self.max_storage_buffer_binding_size
    }

    pub const fn max_buffer_size(&self) -> Option<u64> {
        self.max_buffer_size
    }

    pub const fn fused_min_convolution_length(&self) -> usize {
        self.fused_min_convolution_length
    }

    /// Whether axes too long for workgroup memory still run fused: as one
    /// register-resident kernel where the device supports it (contiguous
    /// power-of-two `f32` axes), else as two fused passes (`N = N1 * N2`),
    /// instead of one multi-pass Stockham stage per radix.
    pub const fn fuse_long_axes(&self) -> bool {
        self.fuse_long_axes
    }

    pub fn with_workgroup_size(mut self, value: u32) -> Self {
        self.workgroup_size = value;
        self
    }

    /// See [`Self::fused_workgroup_size`].
    pub fn with_fused_workgroup_size(mut self, value: u32) -> Self {
        self.fused_workgroup_size = value;
        self
    }

    pub fn with_rader_max_prime(mut self, value: usize) -> Self {
        self.rader_max_prime = value;
        self
    }

    /// See [`Self::direct_max_prime`].
    pub fn with_direct_max_prime(mut self, value: usize) -> Self {
        self.direct_max_prime = value;
        self
    }

    pub fn with_force_rader_axes(mut self, axes: impl Into<Vec<usize>>) -> Self {
        self.force_rader_axes = axes.into();
        self
    }

    pub fn with_force_bluestein_axes(mut self, axes: impl Into<Vec<usize>>) -> Self {
        self.force_bluestein_axes = axes.into();
        self
    }

    pub fn with_large_route(mut self, value: FftLargeRoute) -> Self {
        self.large_route = value;
        self
    }

    pub fn with_large_chunk_max_batches(mut self, value: impl Into<Option<usize>>) -> Self {
        self.large_chunk_max_batches = value.into();
        self
    }

    pub fn with_grouped_batch(mut self, value: impl Into<Option<usize>>) -> Self {
        self.grouped_batch = value.into();
        self
    }

    pub fn with_swap_to_2_stage_4_step(mut self, value: usize) -> Self {
        self.swap_to_2_stage_4_step = value;
        self
    }

    pub fn with_swap_to_3_stage_4_step(mut self, value: usize) -> Self {
        self.swap_to_3_stage_4_step = value;
        self
    }

    pub fn with_segmented_burst_depth(mut self, value: usize) -> Self {
        self.segmented_burst_depth = value;
        self
    }

    pub fn with_max_storage_buffer_binding_size(mut self, value: impl Into<Option<u64>>) -> Self {
        self.max_storage_buffer_binding_size = value.into();
        self
    }

    pub fn with_max_buffer_size(mut self, value: impl Into<Option<u64>>) -> Self {
        self.max_buffer_size = value.into();
        self
    }

    pub fn with_fused_min_convolution_length(mut self, value: usize) -> Self {
        self.fused_min_convolution_length = value;
        self
    }

    /// See [`Self::fuse_long_axes`]. Disabling it keeps multi-pass Stockham
    /// stages for long axes.
    pub fn with_fuse_long_axes(mut self, value: bool) -> Self {
        self.fuse_long_axes = value;
        self
    }

    /// Preserves scalar/kernel thresholds for an internal child plan while
    /// removing physical-axis and top-level route requests that must not be
    /// re-applied recursively.
    #[allow(dead_code)]
    pub(crate) fn for_internal_child(&self) -> Self {
        let mut child = self.clone();
        child.force_rader_axes.clear();
        child.force_bluestein_axes.clear();
        child.large_route = FftLargeRoute::Auto;
        child
    }

    pub(crate) fn validate_for_config(&self, config: &FftConfig) -> Result<()> {
        validate_workgroup_size("workgroup_size", self.workgroup_size)?;
        validate_workgroup_size("fused_workgroup_size", self.fused_workgroup_size)?;
        if self.rader_max_prime < 2 {
            return Err(invalid_tuning(
                FftTuningErrorKind::InvalidValue,
                "rader_max_prime",
                self.rader_max_prime,
                "must be at least 2",
            ));
        }
        validate_optional_positive("large_chunk_max_batches", self.large_chunk_max_batches)?;
        validate_optional_positive("grouped_batch", self.grouped_batch)?;
        if !(1..=3).contains(&self.segmented_burst_depth) {
            return Err(invalid_tuning(
                FftTuningErrorKind::InvalidValue,
                "segmented_burst_depth",
                self.segmented_burst_depth,
                "must be in the range 1..=3",
            ));
        }
        validate_optional_u64_positive(
            "max_storage_buffer_binding_size",
            self.max_storage_buffer_binding_size,
        )?;
        validate_optional_u64_positive("max_buffer_size", self.max_buffer_size)?;
        if self.swap_to_2_stage_4_step > 0
            && self.swap_to_3_stage_4_step > 0
            && self.swap_to_2_stage_4_step > self.swap_to_3_stage_4_step
        {
            return Err(invalid_tuning(
                FftTuningErrorKind::ConflictingValues,
                "swap_to_2_stage_4_step",
                self.swap_to_2_stage_4_step,
                "must not exceed swap_to_3_stage_4_step when both thresholds are active",
            ));
        }

        for &axis in &self.force_rader_axes {
            if self.force_bluestein_axes.contains(&axis) {
                return Err(invalid_tuning(
                    FftTuningErrorKind::ConflictingAxes,
                    "force_rader_axes/force_bluestein_axes",
                    axis,
                    "the same axis cannot be forced to both Rader and Bluestein",
                ));
            }
        }
        validate_forced_axes("force_rader_axes", &self.force_rader_axes, config, true)?;
        validate_forced_axes(
            "force_bluestein_axes",
            &self.force_bluestein_axes,
            config,
            false,
        )?;
        Ok(())
    }

    pub(crate) fn validate_for_device(&self, limits: &wgpu::Limits) -> Result<()> {
        let invocation_limit = limits.max_compute_invocations_per_workgroup;
        let x_limit = limits.max_compute_workgroup_size_x;
        validate_device_workgroup_size(
            "workgroup_size",
            self.workgroup_size,
            invocation_limit,
            x_limit,
        )?;
        validate_device_workgroup_size(
            "fused_workgroup_size",
            self.fused_workgroup_size,
            invocation_limit,
            x_limit,
        )
    }
}

/// Classification for a structured tuning validation error.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum FftTuningErrorKind {
    InvalidValue,
    ConflictingValues,
    DuplicateAxis,
    ConflictingAxes,
    AxisOutOfRange,
    AxisNotSelected,
    AxisAlgorithmIncompatible,
    DeviceLimit,
    RouteInfeasible,
    UnsupportedForTransform,
}

/// Requested and effective tuning attached to plan diagnostics.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct FftTuningSummary {
    requested: FftTuning,
    effective: FftTuning,
}

impl Default for FftTuningSummary {
    fn default() -> Self {
        let tuning = FftTuning::default();
        Self {
            requested: tuning.clone(),
            effective: tuning,
        }
    }
}

impl FftTuningSummary {
    pub fn new(requested: FftTuning, effective: FftTuning) -> Self {
        Self {
            requested,
            effective,
        }
    }

    pub fn requested(&self) -> &FftTuning {
        &self.requested
    }

    pub fn effective(&self) -> &FftTuning {
        &self.effective
    }
}

fn validate_workgroup_size(field: &'static str, value: u32) -> Result<()> {
    if value == 0 || !value.is_power_of_two() {
        return Err(invalid_tuning(
            FftTuningErrorKind::InvalidValue,
            field,
            value,
            "must be a non-zero power of two",
        ));
    }
    Ok(())
}

fn validate_optional_positive(field: &'static str, value: Option<usize>) -> Result<()> {
    if value == Some(0) {
        return Err(invalid_tuning(
            FftTuningErrorKind::InvalidValue,
            field,
            0,
            "must be greater than zero when provided",
        ));
    }
    Ok(())
}

fn validate_optional_u64_positive(field: &'static str, value: Option<u64>) -> Result<()> {
    if value == Some(0) {
        return Err(invalid_tuning(
            FftTuningErrorKind::InvalidValue,
            field,
            0,
            "must be greater than zero when provided",
        ));
    }
    Ok(())
}

fn validate_forced_axes(
    field: &'static str,
    axes: &[usize],
    config: &FftConfig,
    force_rader: bool,
) -> Result<()> {
    for (index, &axis) in axes.iter().enumerate() {
        if axes[..index].contains(&axis) {
            return Err(invalid_tuning(
                FftTuningErrorKind::DuplicateAxis,
                field,
                axis,
                "axis appears more than once",
            ));
        }
        if axis >= config.shape().len() {
            return Err(invalid_tuning(
                FftTuningErrorKind::AxisOutOfRange,
                field,
                axis,
                "axis is outside the configured shape rank",
            ));
        }
        if !config.axes().contains(&axis) {
            return Err(invalid_tuning(
                FftTuningErrorKind::AxisNotSelected,
                field,
                axis,
                "axis is not present in FftConfig::axes",
            ));
        }
        let len = config.shape()[axis];
        if force_rader {
            if len < 3 || !is_prime(len) {
                return Err(invalid_tuning(
                    FftTuningErrorKind::AxisAlgorithmIncompatible,
                    field,
                    axis,
                    "forced Rader axes must have prime length at least 3",
                ));
            }
        } else if len < 2 {
            return Err(invalid_tuning(
                FftTuningErrorKind::AxisAlgorithmIncompatible,
                field,
                axis,
                "forced Bluestein axes must have length at least 2",
            ));
        }
    }
    Ok(())
}

fn validate_device_workgroup_size(
    field: &'static str,
    value: u32,
    invocation_limit: u32,
    x_limit: u32,
) -> Result<()> {
    if value > invocation_limit || value > x_limit {
        return Err(invalid_tuning(
            FftTuningErrorKind::DeviceLimit,
            field,
            value,
            "exceeds max_compute_invocations_per_workgroup or max_compute_workgroup_size_x",
        ));
    }
    Ok(())
}

fn invalid_tuning(
    kind: FftTuningErrorKind,
    field: &'static str,
    value: impl ToString,
    reason: &'static str,
) -> FftError {
    FftError::InvalidTuning {
        kind,
        field,
        value: value.to_string(),
        reason,
    }
}

fn is_prime(value: usize) -> bool {
    if value < 2 {
        return false;
    }
    if value % 2 == 0 {
        return value == 2;
    }
    let mut divisor = 3usize;
    while divisor <= value / divisor {
        if value % divisor == 0 {
            return false;
        }
        divisor += 2;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_and_builders_are_publicly_observable() {
        let defaults = FftTuning::default();
        assert_eq!(defaults.workgroup_size(), 64);
        assert_eq!(defaults.fused_workgroup_size(), 256);
        assert_eq!(defaults.rader_max_prime(), 4096);
        assert_eq!(defaults.direct_max_prime(), 127);
        assert_eq!(defaults.large_route(), FftLargeRoute::Auto);
        assert_eq!(defaults.segmented_burst_depth(), 2);
        assert_eq!(defaults.fused_min_convolution_length(), 128);
        assert!(defaults.fuse_long_axes());

        let tuned = FftTuning::new()
            .with_workgroup_size(128)
            .with_fused_workgroup_size(64)
            .with_rader_max_prime(257)
            .with_direct_max_prime(31)
            .with_force_rader_axes([1])
            .with_force_bluestein_axes([0])
            .with_large_route(FftLargeRoute::ForceFourStep)
            .with_large_chunk_max_batches(Some(3))
            .with_grouped_batch(Some(4))
            .with_swap_to_2_stage_4_step(1024)
            .with_swap_to_3_stage_4_step(4096)
            .with_segmented_burst_depth(3)
            .with_max_storage_buffer_binding_size(Some(1 << 20))
            .with_max_buffer_size(Some(1 << 24))
            .with_fused_min_convolution_length(64)
            .with_fuse_long_axes(false);
        assert_eq!(tuned.workgroup_size(), 128);
        assert_eq!(tuned.direct_max_prime(), 31);
        assert!(!tuned.fuse_long_axes());
        assert_eq!(tuned.force_rader_axes(), &[1]);
        assert_eq!(tuned.force_bluestein_axes(), &[0]);
        assert_eq!(tuned.large_chunk_max_batches(), Some(3));
        assert_eq!(tuned.grouped_batch(), Some(4));
        assert_eq!(tuned.max_buffer_size(), Some(1 << 24));
    }

    #[test]
    fn summary_exposes_full_requested_and_effective_values() {
        let requested = FftTuning::default().with_grouped_batch(Some(8));
        let effective = requested.clone().with_grouped_batch(Some(4));
        let summary = FftTuningSummary::new(requested.clone(), effective.clone());
        assert_eq!(summary.requested(), &requested);
        assert_eq!(summary.effective(), &effective);
    }

    #[test]
    fn internal_child_clears_axis_and_route_forcing_only() {
        let parent = FftTuning::default()
            .with_workgroup_size(128)
            .with_force_rader_axes([1])
            .with_force_bluestein_axes([2])
            .with_large_route(FftLargeRoute::ForceSegmented)
            .with_grouped_batch(Some(4))
            .with_fuse_long_axes(false);
        let child = parent.for_internal_child();
        assert!(child.force_rader_axes().is_empty());
        assert!(child.force_bluestein_axes().is_empty());
        assert_eq!(child.large_route(), FftLargeRoute::Auto);
        assert_eq!(child.workgroup_size(), 128);
        assert_eq!(child.grouped_batch(), Some(4));
        assert!(!child.fuse_long_axes());
    }

    #[test]
    fn device_validation_checks_both_workgroup_controls() {
        let mut limits = wgpu::Limits::default();
        limits.max_compute_invocations_per_workgroup = 128;
        limits.max_compute_workgroup_size_x = 128;
        assert!(matches!(
            FftTuning::default().validate_for_device(&limits),
            Err(FftError::InvalidTuning {
                kind: FftTuningErrorKind::DeviceLimit,
                field: "fused_workgroup_size",
                ..
            })
        ));
        assert_eq!(
            FftTuning::default()
                .with_workgroup_size(128)
                .with_fused_workgroup_size(128)
                .validate_for_device(&limits),
            Ok(())
        );
        assert!(matches!(
            FftTuning::default()
                .with_workgroup_size(256)
                .with_fused_workgroup_size(64)
                .validate_for_device(&limits),
            Err(FftError::InvalidTuning {
                kind: FftTuningErrorKind::DeviceLimit,
                field: "workgroup_size",
                ..
            })
        ));
    }
}
