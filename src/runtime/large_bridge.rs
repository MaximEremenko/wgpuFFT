use crate::config::{FftConfig, FftDirection, Normalization};
use crate::error::{FftError, Result};
use crate::runtime::axis_policy::{is_prime, next_power_of_two_at_least, next_smooth_at_least};
use crate::runtime::factor_supported_length;
use crate::runtime::large_policy::{LargeFactorSplit, LargePolicyLimits};
use crate::runtime::nd_wgsl::lines_per_batch;
use crate::runtime::smooth_decompose::SmoothDecompositionPlan;

const COMPLEX_F32_BYTES: u64 = 8;
const U32_DISPATCH_LIMIT: u64 = u32::MAX as u64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LargeBridgeRoute {
    Rader,
    Bluestein,
}

impl LargeBridgeRoute {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Rader => "rader",
            Self::Bluestein => "bluestein",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LargeBridgeChildRoute {
    Normal,
    SmoothDecomposition,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LargeBridgeBufferKind {
    Permutation,
    Chirp,
    Bfft,
    Sum,
    X0,
    Work,
    Fft,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LargeBridgeStagingKind {
    AxisLine,
    ConvolutionLine,
}

impl LargeBridgeBufferKind {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Permutation => "permutation",
            Self::Chirp => "chirp",
            Self::Bfft => "bfft",
            Self::Sum => "sum",
            Self::X0 => "x0",
            Self::Work => "work",
            Self::Fft => "fft",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct LargeBridgeBuffer {
    pub(crate) kind: LargeBridgeBufferKind,
    pub(crate) bytes: u64,
    pub(crate) bind_full: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct LargeBridgePlan {
    route: LargeBridgeRoute,
    axis: usize,
    axis_len: u64,
    line_count: u64,
    convolution_len: u64,
    convolution_factors: Vec<u64>,
    child_route: LargeBridgeChildRoute,
    chunk_line_count: u64,
    helper_buffers: Vec<LargeBridgeBuffer>,
    staging_bytes: Vec<u64>,
}

impl LargeBridgePlan {
    pub(crate) fn route(&self) -> LargeBridgeRoute {
        self.route
    }

    pub(crate) fn axis(&self) -> usize {
        self.axis
    }

    pub(crate) fn axis_len(&self) -> u64 {
        self.axis_len
    }

    pub(crate) fn line_count(&self) -> u64 {
        self.line_count
    }

    pub(crate) fn convolution_len(&self) -> u64 {
        self.convolution_len
    }

    #[cfg(test)]
    pub(crate) fn child_route(&self) -> LargeBridgeChildRoute {
        self.child_route
    }

    #[cfg(test)]
    pub(crate) fn chunk_line_count(&self) -> u64 {
        self.chunk_line_count
    }

    #[cfg(test)]
    pub(crate) fn helper_buffers(&self) -> &[LargeBridgeBuffer] {
        &self.helper_buffers
    }

    pub(crate) fn staging_bytes(&self) -> &[u64] {
        &self.staging_bytes
    }

    pub(crate) fn factor_splits(&self) -> Vec<LargeFactorSplit> {
        vec![
            LargeFactorSplit {
                axis: Some(self.axis),
                len: self.axis_len,
                factors: vec![self.convolution_len],
            },
            LargeFactorSplit {
                axis: None,
                len: self.convolution_len,
                factors: self.convolution_factors.clone(),
            },
        ]
    }
}

pub(crate) fn plan_large_bridge(
    config: &FftConfig,
    route: LargeBridgeRoute,
    limits: LargePolicyLimits,
) -> Result<LargeBridgePlan> {
    config.validate()?;
    if config.axes().len() != 1 {
        return Err(FftError::LargeBridgeUnsupported {
            route: route.as_str(),
            reason: "large Rader/Bluestein bridge V1 supports one transformed axis",
        });
    }

    let axis = config.axes()[0];
    let axis_len = *config.shape().get(axis).ok_or(FftError::InvalidAxis {
        axis,
        rank: config.shape().len(),
    })?;
    validate_route_axis(route, axis_len)?;

    let line_count = checked_mul_u64(
        config.batch() as u64,
        lines_per_batch(config.shape(), axis) as u64,
    )?;
    if line_count == 0 {
        return Err(FftError::ZeroBatch);
    }

    let convolution_len = convolution_len_for(route, axis_len)?;
    let convolution_factors = factor_supported_length(usize_from_u64(convolution_len)?)?
        .into_iter()
        .map(|value| value as u64)
        .collect::<Vec<_>>();
    let axis_line_bytes = checked_mul_u64(axis_len as u64, COMPLEX_F32_BYTES)?;
    let convolution_line_bytes = checked_mul_u64(convolution_len, COMPLEX_F32_BYTES)?;
    if checked_mul_u64(line_count, convolution_len)? > U32_DISPATCH_LIMIT {
        return Err(FftError::LargeBridgeUnsupported {
            route: route.as_str(),
            reason: "large bridge convolution dispatch exceeds the current u32 limit",
        });
    }
    for (kind, bytes) in [
        (LargeBridgeStagingKind::AxisLine, axis_line_bytes),
        (
            LargeBridgeStagingKind::ConvolutionLine,
            convolution_line_bytes,
        ),
    ] {
        if bytes > limits.max_buffer_size {
            return Err(FftError::HelperBufferTooLarge {
                helper_buffer: bridge_staging_label(route, kind),
                requested_bytes: bytes,
                max_buffer_size: limits.max_buffer_size,
            });
        }
    }

    let mut helper_buffers = helper_buffers_for(route, axis_len as u64, convolution_len)?;
    validate_helper_buffers(route, &helper_buffers, limits)?;

    let chunk_line_count = choose_chunk_line_count(convolution_line_bytes, limits).ok_or(
        FftError::LargeBridgeUnsupported {
            route: route.as_str(),
            reason: "large bridge cannot schedule a legal convolution line window",
        },
    )?;
    let child_route = choose_child_route(route, convolution_len, convolution_line_bytes, limits)?;

    helper_buffers.sort_by_key(|buffer| (buffer.bytes, buffer.kind.as_str()));
    let mut staging_bytes = vec![axis_line_bytes, convolution_line_bytes];
    staging_bytes.push(checked_mul_u64(chunk_line_count, convolution_line_bytes)?);
    staging_bytes.extend(helper_buffers.iter().map(|buffer| buffer.bytes));
    staging_bytes.sort_unstable();
    staging_bytes.dedup();

    Ok(LargeBridgePlan {
        route,
        axis,
        axis_len: axis_len as u64,
        line_count,
        convolution_len,
        convolution_factors,
        child_route,
        chunk_line_count,
        helper_buffers,
        staging_bytes,
    })
}

fn validate_route_axis(route: LargeBridgeRoute, axis_len: usize) -> Result<()> {
    match route {
        LargeBridgeRoute::Rader if !is_prime(axis_len) => Err(FftError::LargeBridgeUnsupported {
            route: route.as_str(),
            reason: "Rader bridge requires a prime axis length",
        }),
        LargeBridgeRoute::Bluestein if axis_len < 2 => Err(FftError::LargeBridgeUnsupported {
            route: route.as_str(),
            reason: "Bluestein bridge requires an axis length above one",
        }),
        _ => Ok(()),
    }
}

fn convolution_len_for(route: LargeBridgeRoute, axis_len: usize) -> Result<u64> {
    let min_conv = match route {
        LargeBridgeRoute::Rader => {
            let l = axis_len.checked_sub(1).ok_or(FftError::ZeroLength)?;
            l.checked_mul(2)
                .and_then(|value| value.checked_sub(1))
                .ok_or(FftError::LengthTooLarge { len: usize::MAX })?
        }
        LargeBridgeRoute::Bluestein => axis_len
            .checked_mul(2)
            .and_then(|value| value.checked_sub(1))
            .ok_or(FftError::LengthTooLarge { len: usize::MAX })?,
    };
    let mut convolution_len = next_smooth_at_least(min_conv);
    if route == LargeBridgeRoute::Rader && factor_supported_length(convolution_len).is_err() {
        convolution_len = next_power_of_two_at_least(min_conv);
    }
    factor_supported_length(convolution_len)?;
    Ok(convolution_len as u64)
}

fn helper_buffers_for(
    route: LargeBridgeRoute,
    axis_len: u64,
    convolution_len: u64,
) -> Result<Vec<LargeBridgeBuffer>> {
    let axis_line_bytes = checked_mul_u64(axis_len, COMPLEX_F32_BYTES)?;
    let convolution_line_bytes = checked_mul_u64(convolution_len, COMPLEX_F32_BYTES)?;
    let line_scalar_bytes = COMPLEX_F32_BYTES;
    let work_bytes = convolution_line_bytes;

    let mut buffers = match route {
        LargeBridgeRoute::Rader => vec![
            LargeBridgeBuffer {
                kind: LargeBridgeBufferKind::Permutation,
                bytes: checked_mul_u64(axis_len.saturating_sub(1), 4)?,
                bind_full: true,
            },
            LargeBridgeBuffer {
                kind: LargeBridgeBufferKind::Bfft,
                bytes: convolution_line_bytes,
                bind_full: true,
            },
            LargeBridgeBuffer {
                kind: LargeBridgeBufferKind::Sum,
                bytes: line_scalar_bytes,
                bind_full: false,
            },
            LargeBridgeBuffer {
                kind: LargeBridgeBufferKind::X0,
                bytes: line_scalar_bytes,
                bind_full: false,
            },
        ],
        LargeBridgeRoute::Bluestein => vec![
            LargeBridgeBuffer {
                kind: LargeBridgeBufferKind::Chirp,
                bytes: axis_line_bytes,
                bind_full: true,
            },
            LargeBridgeBuffer {
                kind: LargeBridgeBufferKind::Bfft,
                bytes: convolution_line_bytes,
                bind_full: true,
            },
        ],
    };
    buffers.push(LargeBridgeBuffer {
        kind: LargeBridgeBufferKind::Work,
        bytes: work_bytes,
        bind_full: false,
    });
    buffers.push(LargeBridgeBuffer {
        kind: LargeBridgeBufferKind::Fft,
        bytes: work_bytes,
        bind_full: false,
    });
    Ok(buffers)
}

fn validate_helper_buffers(
    route: LargeBridgeRoute,
    buffers: &[LargeBridgeBuffer],
    limits: LargePolicyLimits,
) -> Result<()> {
    for buffer in buffers {
        if buffer.bytes > limits.max_buffer_size {
            return Err(FftError::HelperBufferTooLarge {
                helper_buffer: bridge_helper_label(route, buffer.kind),
                requested_bytes: buffer.bytes,
                max_buffer_size: limits.max_buffer_size,
            });
        }
    }
    Ok(())
}

fn choose_chunk_line_count(convolution_line_bytes: u64, limits: LargePolicyLimits) -> Option<u64> {
    if convolution_line_bytes == 0 {
        return None;
    }
    let by_bind = limits.max_storage_buffer_binding_size / convolution_line_bytes;
    let by_buffer = limits.max_buffer_size / convolution_line_bytes;
    let line_count = by_bind.min(by_buffer);
    if line_count > 0 {
        Some(line_count)
    } else if convolution_line_bytes <= limits.max_buffer_size {
        Some(1)
    } else {
        None
    }
}

fn choose_child_route(
    route: LargeBridgeRoute,
    convolution_len: u64,
    convolution_line_bytes: u64,
    limits: LargePolicyLimits,
) -> Result<LargeBridgeChildRoute> {
    if convolution_line_bytes <= limits.max_storage_buffer_binding_size {
        return Ok(LargeBridgeChildRoute::Normal);
    }

    let config = FftConfig::new(usize_from_u64(convolution_len)?)
        .with_direction(FftDirection::Forward)
        .with_normalization(Normalization::None);
    if SmoothDecompositionPlan::new(&config, limits).is_ok() {
        return Ok(LargeBridgeChildRoute::SmoothDecomposition);
    }

    Err(FftError::LargeBridgeUnsupported {
        route: route.as_str(),
        reason: "large bridge convolution cannot be routed under the current limits",
    })
}

fn bridge_staging_label(route: LargeBridgeRoute, kind: LargeBridgeStagingKind) -> &'static str {
    match (route, kind) {
        (LargeBridgeRoute::Rader, LargeBridgeStagingKind::AxisLine) => {
            "wgpu_fft.c2c.bridge.rader.axis_line_stage"
        }
        (LargeBridgeRoute::Rader, LargeBridgeStagingKind::ConvolutionLine) => {
            "wgpu_fft.c2c.bridge.rader.convolution_line_stage"
        }
        (LargeBridgeRoute::Bluestein, LargeBridgeStagingKind::AxisLine) => {
            "wgpu_fft.c2c.bridge.bluestein.axis_line_stage"
        }
        (LargeBridgeRoute::Bluestein, LargeBridgeStagingKind::ConvolutionLine) => {
            "wgpu_fft.c2c.bridge.bluestein.convolution_line_stage"
        }
    }
}

fn bridge_helper_label(route: LargeBridgeRoute, kind: LargeBridgeBufferKind) -> &'static str {
    match (route, kind) {
        (LargeBridgeRoute::Rader, LargeBridgeBufferKind::Permutation) => {
            "wgpu_fft.c2c.bridge.rader.permutation"
        }
        (LargeBridgeRoute::Rader, LargeBridgeBufferKind::Chirp) => {
            "wgpu_fft.c2c.bridge.rader.chirp"
        }
        (LargeBridgeRoute::Rader, LargeBridgeBufferKind::Bfft) => "wgpu_fft.c2c.bridge.rader.bfft",
        (LargeBridgeRoute::Rader, LargeBridgeBufferKind::Sum) => "wgpu_fft.c2c.bridge.rader.sum",
        (LargeBridgeRoute::Rader, LargeBridgeBufferKind::X0) => "wgpu_fft.c2c.bridge.rader.x0",
        (LargeBridgeRoute::Rader, LargeBridgeBufferKind::Work) => "wgpu_fft.c2c.bridge.rader.work",
        (LargeBridgeRoute::Rader, LargeBridgeBufferKind::Fft) => "wgpu_fft.c2c.bridge.rader.fft",
        (LargeBridgeRoute::Bluestein, LargeBridgeBufferKind::Permutation) => {
            "wgpu_fft.c2c.bridge.bluestein.permutation"
        }
        (LargeBridgeRoute::Bluestein, LargeBridgeBufferKind::Chirp) => {
            "wgpu_fft.c2c.bridge.bluestein.chirp"
        }
        (LargeBridgeRoute::Bluestein, LargeBridgeBufferKind::Bfft) => {
            "wgpu_fft.c2c.bridge.bluestein.bfft"
        }
        (LargeBridgeRoute::Bluestein, LargeBridgeBufferKind::Sum) => {
            "wgpu_fft.c2c.bridge.bluestein.sum"
        }
        (LargeBridgeRoute::Bluestein, LargeBridgeBufferKind::X0) => {
            "wgpu_fft.c2c.bridge.bluestein.x0"
        }
        (LargeBridgeRoute::Bluestein, LargeBridgeBufferKind::Work) => {
            "wgpu_fft.c2c.bridge.bluestein.work"
        }
        (LargeBridgeRoute::Bluestein, LargeBridgeBufferKind::Fft) => {
            "wgpu_fft.c2c.bridge.bluestein.fft"
        }
    }
}

fn checked_mul_u64(a: u64, b: u64) -> Result<u64> {
    a.checked_mul(b)
        .ok_or(FftError::LengthTooLarge { len: usize::MAX })
}

fn usize_from_u64(value: u64) -> Result<usize> {
    usize::try_from(value).map_err(|_| FftError::LengthTooLarge { len: usize::MAX })
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
    fn plans_rader_bridge_with_chunked_convolution() {
        let config = FftConfig::new(17)
            .with_batch(8)
            .with_normalization(Normalization::None);
        let plan =
            plan_large_bridge(&config, LargeBridgeRoute::Rader, limits(512, 1 << 20)).unwrap();

        assert_eq!(plan.route(), LargeBridgeRoute::Rader);
        assert_eq!(plan.axis(), 0);
        assert_eq!(plan.convolution_len(), 32);
        assert_eq!(plan.child_route(), LargeBridgeChildRoute::Normal);
        assert!(plan.chunk_line_count() >= 2);
        assert!(plan
            .helper_buffers()
            .iter()
            .any(|buffer| buffer.kind == LargeBridgeBufferKind::Permutation));
        assert!(plan.helper_buffers().iter().any(|buffer| {
            buffer.kind == LargeBridgeBufferKind::Work
                && buffer.bytes == plan.convolution_len() * COMPLEX_F32_BYTES
        }));
        assert_eq!(plan.factor_splits()[1].factors, [8, 4]);
    }

    #[test]
    fn plans_bluestein_bridge_for_safe_single_axis_case() {
        let config = FftConfig::new(34).with_normalization(Normalization::None);
        let plan =
            plan_large_bridge(&config, LargeBridgeRoute::Bluestein, limits(1024, 1 << 20)).unwrap();

        assert_eq!(plan.route(), LargeBridgeRoute::Bluestein);
        assert_eq!(plan.convolution_len(), 70);
        assert_eq!(plan.child_route(), LargeBridgeChildRoute::Normal);
        assert!(plan.staging_bytes().contains(&(70 * COMPLEX_F32_BYTES)));
    }

    #[test]
    fn rejects_multiple_transformed_axes() {
        let err = plan_large_bridge(
            &FftConfig::new_nd([17, 4]).with_normalization(Normalization::None),
            LargeBridgeRoute::Rader,
            limits(512, 1 << 20),
        )
        .unwrap_err();

        assert_eq!(
            err,
            FftError::LargeBridgeUnsupported {
                route: "rader",
                reason: "large Rader/Bluestein bridge V1 supports one transformed axis",
            }
        );
    }

    #[test]
    fn plans_windowed_helpers_when_full_helper_binding_is_oversized() {
        let plan = plan_large_bridge(
            &FftConfig::new(4099).with_normalization(Normalization::None),
            LargeBridgeRoute::Bluestein,
            limits(1024, 1 << 30),
        )
        .unwrap();

        assert_eq!(plan.route(), LargeBridgeRoute::Bluestein);
        assert!(plan
            .helper_buffers()
            .iter()
            .any(|buffer| buffer.bytes > 1024 && buffer.bind_full));
        assert_eq!(plan.chunk_line_count(), 1);
        assert_eq!(
            plan.child_route(),
            LargeBridgeChildRoute::SmoothDecomposition
        );
    }

    #[test]
    fn plans_batched_bridge_with_per_line_helpers() {
        let plan = plan_large_bridge(
            &FftConfig::new(17)
                .with_batch(8)
                .with_normalization(Normalization::None),
            LargeBridgeRoute::Rader,
            limits(512, 1024),
        )
        .unwrap();

        assert_eq!(plan.line_count(), 8);
        assert!(plan.helper_buffers().iter().any(|buffer| {
            buffer.kind == LargeBridgeBufferKind::Work && buffer.bytes == 32 * COMPLEX_F32_BYTES
        }));
        assert!(!plan.staging_bytes().contains(&(8 * 32 * COMPLEX_F32_BYTES)));
    }

    #[test]
    fn rejects_per_line_staging_above_max_buffer() {
        let err = plan_large_bridge(
            &FftConfig::new(257).with_normalization(Normalization::None),
            LargeBridgeRoute::Rader,
            limits(512, 512),
        )
        .unwrap_err();

        assert_eq!(
            err,
            FftError::HelperBufferTooLarge {
                helper_buffer: "wgpu_fft.c2c.bridge.rader.axis_line_stage",
                requested_bytes: 2056,
                max_buffer_size: 512,
            }
        );
    }
}
