use crate::config::FftConfig;
use crate::error::{FftError, Result};
use crate::runtime::axis_policy::{resolve_axis_kinds_for_config, AxisKind};
use crate::runtime::factor_supported_length;
use crate::runtime::large_policy::{LargeExecutionKind, LargeFactorSplit, LargePolicyLimits};

const COMPLEX_F32_BYTES: u64 = 8;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SmoothDecompositionPlan {
    shape: Vec<usize>,
    steps: Vec<SmoothDecompositionStep>,
    required_buffer_size_bytes: u64,
    temp_buffer_size_bytes: u64,
    execution_kind: LargeExecutionKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SmoothDecompositionStep {
    MixedAxis(MixedAxisStep),
    SmoothAxis(SmoothAxisStep),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MixedAxisStep {
    axis: usize,
    len: u64,
    stride: u64,
    line_count: u64,
    line_bytes: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SmoothAxisStep {
    axis: usize,
    len: u64,
    stride: u64,
    line_count: u64,
    inner: u64,
    outer: u64,
    chunk_inner: u64,
    chunk_outer: u64,
    phase1_chunk_bytes: u64,
    phase2_chunk_bytes: u64,
    recursive_depth: u8,
}

impl SmoothDecompositionPlan {
    pub(crate) fn new(config: &FftConfig, limits: LargePolicyLimits) -> Result<Self> {
        validate_config(config, limits)?;
        let required_buffer_size_bytes = config.required_buffer_size_bytes()?;
        if required_buffer_size_bytes <= limits.max_storage_buffer_binding_size {
            return Err(FftError::LargeChunkUnsupported {
                reason:
                    "smooth decomposition is only used when a transform exceeds the binding limit",
                bytes_per_batch: required_buffer_size_bytes,
                max_bind_bytes: limits.max_storage_buffer_binding_size,
            });
        }

        let axis_kinds = resolve_axis_kinds_for_config(config)?;
        let mut steps = Vec::with_capacity(config.axes().len());
        let total_logical = config.logical_complex_len()? as u64;

        for (&axis, &kind) in config.axes().iter().zip(&axis_kinds) {
            let axis_len = config.shape()[axis] as u64;
            let line_bytes = checked_axis_bytes(axis_len, limits.max_storage_buffer_binding_size)?;
            if kind != AxisKind::Mixed {
                return Err(FftError::LargeChunkUnsupported {
                    reason: "large decomposition V1 only supports mixed-radix transformed axes",
                    bytes_per_batch: line_bytes,
                    max_bind_bytes: limits.max_storage_buffer_binding_size,
                });
            }

            if line_bytes > limits.max_storage_buffer_binding_size {
                steps.push(SmoothDecompositionStep::SmoothAxis(SmoothAxisStep::new(
                    config, axis, limits,
                )?));
            } else {
                steps.push(SmoothDecompositionStep::MixedAxis(MixedAxisStep::new(
                    config,
                    axis,
                    total_logical,
                    line_bytes,
                )?));
            }
        }

        if steps
            .iter()
            .all(|step| matches!(step, SmoothDecompositionStep::MixedAxis(_)))
            && required_buffer_size_bytes <= limits.max_storage_buffer_binding_size
        {
            return Err(FftError::LargeChunkUnsupported {
                reason: "axis decomposition is only used for large logical buffers",
                bytes_per_batch: required_buffer_size_bytes,
                max_bind_bytes: limits.max_storage_buffer_binding_size,
            });
        }

        let temp_buffer_size_bytes = if steps.len() > 1 {
            if required_buffer_size_bytes > limits.max_buffer_size {
                return Err(FftError::LargeChunkUnsupported {
                    reason: "multi-axis large decomposition V1 requires one full GPU temp buffer",
                    bytes_per_batch: required_buffer_size_bytes,
                    max_bind_bytes: limits.max_storage_buffer_binding_size,
                });
            }
            required_buffer_size_bytes
        } else {
            0
        };

        let execution_kind = if matches!(
            steps.as_slice(),
            [SmoothDecompositionStep::SmoothAxis(step)] if step.axis == 0 && config.shape().len() == 1
        ) {
            LargeExecutionKind::Smooth1dDecomposition
        } else {
            LargeExecutionKind::AxisDecomposition
        };

        Ok(Self {
            shape: config.shape().to_vec(),
            steps,
            required_buffer_size_bytes,
            temp_buffer_size_bytes,
            execution_kind,
        })
    }

    pub(crate) fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub(crate) fn steps(&self) -> &[SmoothDecompositionStep] {
        &self.steps
    }

    pub(crate) fn required_buffer_size_bytes(&self) -> u64 {
        self.required_buffer_size_bytes
    }

    pub(crate) fn temp_buffer_size_bytes(&self) -> u64 {
        self.temp_buffer_size_bytes
    }

    pub(crate) fn execution_kind(&self) -> LargeExecutionKind {
        self.execution_kind
    }

    pub(crate) fn selected_axis(&self) -> Option<usize> {
        self.steps
            .iter()
            .find_map(|step| match step {
                SmoothDecompositionStep::SmoothAxis(step) => Some(step.axis()),
                SmoothDecompositionStep::MixedAxis(_) => None,
            })
            .or_else(|| {
                self.steps.first().map(|step| match step {
                    SmoothDecompositionStep::SmoothAxis(step) => step.axis(),
                    SmoothDecompositionStep::MixedAxis(step) => step.axis(),
                })
            })
    }

    pub(crate) fn factor_splits(&self) -> Vec<LargeFactorSplit> {
        self.steps
            .iter()
            .map(|step| match step {
                SmoothDecompositionStep::SmoothAxis(step) => LargeFactorSplit {
                    axis: Some(step.axis()),
                    len: step.len(),
                    factors: vec![step.inner(), step.outer()],
                },
                SmoothDecompositionStep::MixedAxis(step) => LargeFactorSplit {
                    axis: Some(step.axis()),
                    len: step.len(),
                    factors: factor_supported_length(step.len() as usize)
                        .unwrap_or_default()
                        .into_iter()
                        .map(|factor| factor as u64)
                        .collect(),
                },
            })
            .collect()
    }

    pub(crate) fn staging_bytes(&self) -> Vec<u64> {
        let mut bytes = Vec::new();
        if self.temp_buffer_size_bytes > 0 {
            bytes.push(self.temp_buffer_size_bytes);
        }
        for step in &self.steps {
            match step {
                SmoothDecompositionStep::MixedAxis(step) => {
                    bytes.push(step.line_bytes());
                }
                SmoothDecompositionStep::SmoothAxis(step) => {
                    bytes.push(step.phase1_chunk_bytes());
                    bytes.push(step.phase2_chunk_bytes());
                }
            }
        }
        bytes.sort_unstable();
        bytes.dedup();
        bytes
    }
}

impl MixedAxisStep {
    fn new(config: &FftConfig, axis: usize, total_logical: u64, line_bytes: u64) -> Result<Self> {
        Ok(Self {
            axis,
            len: config.shape()[axis] as u64,
            stride: axis_stride(config.shape(), axis)?,
            line_count: line_count(total_logical, config.shape()[axis] as u64, config.batch())?,
            line_bytes,
        })
    }

    pub(crate) fn axis(self) -> usize {
        self.axis
    }

    pub(crate) fn len(self) -> u64 {
        self.len
    }

    pub(crate) fn stride(self) -> u64 {
        self.stride
    }

    pub(crate) fn line_count(self) -> u64 {
        self.line_count
    }

    pub(crate) fn line_bytes(self) -> u64 {
        self.line_bytes
    }
}

impl SmoothAxisStep {
    fn new(config: &FftConfig, axis: usize, limits: LargePolicyLimits) -> Result<Self> {
        let len = config.shape()[axis] as u64;
        let full_bytes = checked_axis_bytes(len, limits.max_storage_buffer_binding_size)?;
        let recursive_depth = recursive_factor_depth(len, limits.max_storage_buffer_binding_size)?;
        let (inner, outer, chunk_inner, chunk_outer, recursive_depth) =
            if let Some((inner, outer, chunk_inner, chunk_outer)) =
                choose_factor_split(len, limits.max_storage_buffer_binding_size)?
            {
                (inner, outer, chunk_inner, chunk_outer, 1)
            } else if recursive_depth > 1 {
                let Some((inner, outer, chunk_inner, chunk_outer)) =
                    choose_factor_split(len, limits.max_buffer_size)?
                else {
                    return Err(FftError::LargeChunkUnsupported {
                    reason:
                        "recursive smooth decomposition could not find a buffer-safe factor split",
                    bytes_per_batch: full_bytes,
                    max_bind_bytes: limits.max_storage_buffer_binding_size,
                });
                };
                (inner, outer, chunk_inner, chunk_outer, recursive_depth)
            } else {
                return Err(FftError::LargeChunkUnsupported {
                    reason: "smooth decomposition could not find a binding-safe factor split",
                    bytes_per_batch: full_bytes,
                    max_bind_bytes: limits.max_storage_buffer_binding_size,
                });
            };

        let phase1_chunk_bytes = checked_chunk_bytes(chunk_inner, outer, limits.max_buffer_size)?;
        let phase2_chunk_bytes = checked_chunk_bytes(chunk_outer, inner, limits.max_buffer_size)?;
        Ok(Self {
            axis,
            len,
            stride: axis_stride(config.shape(), axis)?,
            line_count: line_count(config.logical_complex_len()? as u64, len, config.batch())?,
            inner,
            outer,
            chunk_inner,
            chunk_outer,
            phase1_chunk_bytes,
            phase2_chunk_bytes,
            recursive_depth,
        })
    }

    pub(crate) fn axis(self) -> usize {
        self.axis
    }

    pub(crate) fn len(self) -> u64 {
        self.len
    }

    pub(crate) fn stride(self) -> u64 {
        self.stride
    }

    pub(crate) fn line_count(self) -> u64 {
        self.line_count
    }

    pub(crate) fn inner(self) -> u64 {
        self.inner
    }

    pub(crate) fn outer(self) -> u64 {
        self.outer
    }

    pub(crate) fn chunk_inner(self) -> u64 {
        self.chunk_inner
    }

    pub(crate) fn chunk_outer(self) -> u64 {
        self.chunk_outer
    }

    pub(crate) fn phase1_chunk_bytes(self) -> u64 {
        self.phase1_chunk_bytes
    }

    pub(crate) fn phase2_chunk_bytes(self) -> u64 {
        self.phase2_chunk_bytes
    }
}

fn validate_config(config: &FftConfig, limits: LargePolicyLimits) -> Result<()> {
    config.validate()?;
    for &axis in config.axes() {
        factor_supported_length(config.shape()[axis]).map_err(|_| {
            FftError::LargeChunkUnsupported {
                reason: "smooth decomposition V1 only supports smooth mixed-radix lengths",
                bytes_per_batch: config.required_buffer_size_bytes().unwrap_or(u64::MAX),
                max_bind_bytes: limits.max_storage_buffer_binding_size,
            }
        })?;
    }
    Ok(())
}

fn choose_factor_split(len: u64, max_bind_bytes: u64) -> Result<Option<(u64, u64, u64, u64)>> {
    let mut divisors = proper_smooth_divisors(len)?;
    divisors.sort_by_key(|&inner| {
        let outer = len / inner;
        inner.abs_diff(outer)
    });

    for inner in divisors {
        let outer = len / inner;
        let Some(chunk_inner) = largest_divisor_fitting(inner, outer, max_bind_bytes) else {
            continue;
        };
        let Some(chunk_outer) = largest_divisor_fitting(outer, inner, max_bind_bytes) else {
            continue;
        };
        return Ok(Some((inner, outer, chunk_inner, chunk_outer)));
    }
    Ok(None)
}

fn proper_smooth_divisors(len: u64) -> Result<Vec<u64>> {
    let mut divisors = Vec::new();
    for divisor in 2..len {
        if len % divisor != 0 {
            continue;
        }
        let other = len / divisor;
        if is_supported_smooth(divisor)? && is_supported_smooth(other)? {
            divisors.push(divisor);
        }
    }
    Ok(divisors)
}

fn is_supported_smooth(value: u64) -> Result<bool> {
    if value > usize::MAX as u64 {
        return Err(FftError::LengthTooLarge { len: usize::MAX });
    }
    Ok(factor_supported_length(value as usize).is_ok())
}

fn largest_divisor_fitting(
    axis_chunkable: u64,
    other_axis: u64,
    max_bind_bytes: u64,
) -> Option<u64> {
    (1..=axis_chunkable).rev().find(|candidate| {
        axis_chunkable % candidate == 0
            && checked_chunk_bytes(*candidate, other_axis, max_bind_bytes).is_ok()
    })
}

fn recursive_factor_depth(len: u64, max_bind_bytes: u64) -> Result<u8> {
    let max_elems = max_bind_bytes / COMPLEX_F32_BYTES;
    if len <= max_elems {
        return Ok(0);
    }
    let mut best = u8::MAX;
    for divisor in proper_smooth_divisors(len)? {
        let other = len / divisor;
        let left = recursive_factor_depth(divisor, max_bind_bytes)?;
        let right = recursive_factor_depth(other, max_bind_bytes)?;
        let depth = left.max(right).saturating_add(1);
        best = best.min(depth);
    }
    if best == u8::MAX {
        Ok(0)
    } else {
        Ok(best)
    }
}

fn checked_axis_bytes(axis_len: u64, max_bind_bytes: u64) -> Result<u64> {
    axis_len
        .checked_mul(COMPLEX_F32_BYTES)
        .ok_or(FftError::LargeChunkUnsupported {
            reason: "smooth decomposition axis byte size overflowed u64",
            bytes_per_batch: axis_len,
            max_bind_bytes,
        })
}

fn checked_chunk_bytes(a: u64, b: u64, max_bind_bytes: u64) -> Result<u64> {
    let bytes = a
        .checked_mul(b)
        .and_then(|value| value.checked_mul(COMPLEX_F32_BYTES))
        .ok_or(FftError::LargeChunkUnsupported {
            reason: "smooth decomposition chunk size overflowed u64",
            bytes_per_batch: a,
            max_bind_bytes,
        })?;
    if bytes == 0 || bytes % 4 != 0 || bytes > max_bind_bytes {
        return Err(FftError::LargeChunkUnsupported {
            reason: "smooth decomposition chunk does not fit the storage-buffer binding limit",
            bytes_per_batch: bytes,
            max_bind_bytes,
        });
    }
    Ok(bytes)
}

fn axis_stride(shape: &[usize], axis: usize) -> Result<u64> {
    let mut stride = 1u64;
    for &dim in shape.iter().take(axis) {
        stride = stride
            .checked_mul(dim as u64)
            .ok_or(FftError::LengthTooLarge { len: usize::MAX })?;
    }
    Ok(stride)
}

fn line_count(total_logical: u64, axis_len: u64, batch: usize) -> Result<u64> {
    if axis_len == 0 {
        return Err(FftError::ZeroLength);
    }
    total_logical
        .checked_div(axis_len)
        .and_then(|lines| lines.checked_mul(batch as u64))
        .ok_or(FftError::LengthTooLarge { len: usize::MAX })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Normalization;

    fn limits(max_bind: u64, max_buffer: u64) -> LargePolicyLimits {
        LargePolicyLimits {
            max_storage_buffer_binding_size: max_bind,
            max_buffer_size: max_buffer,
        }
    }

    #[test]
    fn plans_binding_safe_smooth_factor_split() {
        let config = FftConfig::new(64).with_normalization(Normalization::None);
        let plan = SmoothDecompositionPlan::new(&config, limits(256, 512)).unwrap();
        assert_eq!(
            plan.execution_kind(),
            LargeExecutionKind::Smooth1dDecomposition
        );
        assert_eq!(plan.required_buffer_size_bytes(), 64 * 8);
        assert_eq!(plan.temp_buffer_size_bytes(), 0);
        let [SmoothDecompositionStep::SmoothAxis(axis)] = plan.steps() else {
            panic!("expected one smooth axis step");
        };
        assert_eq!(axis.len(), 64);
        assert_eq!(axis.inner() * axis.outer(), 64);
        assert_eq!(axis.chunk_inner(), 4);
        assert_eq!(axis.chunk_outer(), 4);
        assert_eq!(axis.phase1_chunk_bytes(), 256);
        assert_eq!(axis.phase2_chunk_bytes(), 256);
    }

    #[test]
    fn plans_nd_axis_decomposition_for_one_oversized_axis() {
        let config = FftConfig::new_nd([64, 2]).with_normalization(Normalization::None);
        let plan = SmoothDecompositionPlan::new(&config, limits(256, 1024)).unwrap();
        assert_eq!(plan.execution_kind(), LargeExecutionKind::AxisDecomposition);
        assert_eq!(plan.steps().len(), 2);
        assert_eq!(plan.temp_buffer_size_bytes(), 1024);
        assert!(matches!(
            plan.steps()[0],
            SmoothDecompositionStep::SmoothAxis(step) if step.axis() == 0 && step.line_count() == 2
        ));
        assert!(matches!(
            plan.steps()[1],
            SmoothDecompositionStep::MixedAxis(step) if step.axis() == 1 && step.line_count() == 64
        ));
    }

    #[test]
    fn plans_axis_decomposition_when_only_full_buffer_is_large() {
        let config = FftConfig::new_nd([16, 16]).with_normalization(Normalization::None);
        let plan = SmoothDecompositionPlan::new(&config, limits(256, 2048)).unwrap();
        assert_eq!(plan.execution_kind(), LargeExecutionKind::AxisDecomposition);
        assert!(plan
            .steps()
            .iter()
            .all(|step| matches!(step, SmoothDecompositionStep::MixedAxis(_))));
    }

    #[test]
    fn plans_recursive_smooth_factor_split_when_one_two_step_split_cannot_bind() {
        let config = FftConfig::new(4096).with_normalization(Normalization::None);
        let plan = SmoothDecompositionPlan::new(&config, limits(256, 4096 * 8)).unwrap();
        let [SmoothDecompositionStep::SmoothAxis(axis)] = plan.steps() else {
            panic!("expected one recursive smooth axis");
        };
        assert!(axis.recursive_depth > 1);
        assert_eq!(axis.inner() * axis.outer(), 4096);
        assert_eq!(axis.phase1_chunk_bytes(), 4096 * 8);
        assert_eq!(axis.phase2_chunk_bytes(), 4096 * 8);
        assert_eq!(plan.factor_splits()[0].factors.len(), 2);
    }

    #[test]
    fn rejects_prime_axes() {
        assert_eq!(
            SmoothDecompositionPlan::new(&FftConfig::new(17), limits(64, 1024)).unwrap_err(),
            FftError::LargeChunkUnsupported {
                reason: "smooth decomposition V1 only supports smooth mixed-radix lengths",
                bytes_per_batch: 17 * 8,
                max_bind_bytes: 64,
            }
        );
    }

    #[test]
    fn plans_multiple_oversized_axes_when_full_temp_fits() {
        let config = FftConfig::new_nd([64, 64]).with_normalization(Normalization::None);
        let plan = SmoothDecompositionPlan::new(&config, limits(256, 64 * 64 * 8)).unwrap();
        assert_eq!(plan.execution_kind(), LargeExecutionKind::AxisDecomposition);
        assert_eq!(plan.steps().len(), 2);
        assert_eq!(plan.temp_buffer_size_bytes(), 64 * 64 * 8);
        assert!(matches!(
            plan.steps()[0],
            SmoothDecompositionStep::SmoothAxis(step) if step.axis() == 0 && step.line_count() == 64
        ));
        assert!(matches!(
            plan.steps()[1],
            SmoothDecompositionStep::SmoothAxis(step) if step.axis() == 1 && step.line_count() == 64
        ));
        assert_eq!(plan.factor_splits().len(), 2);
    }

    #[test]
    fn plans_batched_oversized_axis_when_each_transform_can_be_staged() {
        let config = FftConfig::new(64)
            .with_batch(2)
            .with_normalization(Normalization::None);
        let plan = SmoothDecompositionPlan::new(&config, limits(256, 64 * 2 * 8)).unwrap();
        assert_eq!(
            plan.execution_kind(),
            LargeExecutionKind::Smooth1dDecomposition
        );
        assert_eq!(plan.temp_buffer_size_bytes(), 0);
        let [SmoothDecompositionStep::SmoothAxis(axis)] = plan.steps() else {
            panic!("expected one smooth batched axis");
        };
        assert_eq!(axis.axis(), 0);
        assert_eq!(axis.line_count(), 2);
        assert_eq!(axis.phase1_chunk_bytes(), 256);
        assert_eq!(axis.phase2_chunk_bytes(), 256);
    }

    #[test]
    fn rejects_full_temp_that_exceeds_buffer_limit() {
        assert_eq!(
            SmoothDecompositionPlan::new(&FftConfig::new_nd([64, 64]), limits(256, 4096))
                .unwrap_err(),
            FftError::LargeChunkUnsupported {
                reason: "multi-axis large decomposition V1 requires one full GPU temp buffer",
                bytes_per_batch: 64 * 64 * 8,
                max_bind_bytes: 256,
            }
        );
    }

    #[test]
    fn rejects_multi_axis_temp_that_exceeds_buffer_limit() {
        assert_eq!(
            SmoothDecompositionPlan::new(&FftConfig::new_nd([64, 2]), limits(256, 512))
                .unwrap_err(),
            FftError::LargeChunkUnsupported {
                reason: "multi-axis large decomposition V1 requires one full GPU temp buffer",
                bytes_per_batch: 64 * 2 * 8,
                max_bind_bytes: 256,
            }
        );
    }
}
