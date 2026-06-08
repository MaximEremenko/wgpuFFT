use std::fmt;

pub type Result<T> = std::result::Result<T, NufftError>;

/// Structured validation and CPU-reference errors for NUFFT planning.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub enum NufftError {
    InvalidDimensions {
        actual: usize,
        minimum: usize,
        maximum: usize,
    },
    ZeroMode {
        axis: usize,
    },
    InvalidTolerance {
        eps: f64,
        reason: &'static str,
    },
    InvalidSigma {
        sigma: f64,
        reason: &'static str,
    },
    KernelWidthUnsupported {
        eps: f64,
        width: usize,
        maximum: usize,
    },
    InvalidKernelWidth {
        width: usize,
        minimum: usize,
        maximum: usize,
    },
    LengthOverflow {
        context: &'static str,
    },
    CoordinateLength {
        actual: usize,
        dimensions: usize,
    },
    NonFiniteCoordinate {
        point: usize,
        axis: usize,
        value: f64,
    },
    PointOutOfRange {
        point: usize,
        axis: usize,
        value: f64,
        maximum_magnitude: f64,
    },
    InputLength {
        input: &'static str,
        expected: usize,
        actual: usize,
    },
    NonFiniteComplex {
        input: &'static str,
        index: usize,
    },
    FineGridMustBeEven {
        length: usize,
    },
    FftShapeUnsupported {
        stage: &'static str,
        source: wgpu_fft::FftError,
    },
    HostAllocationFailed {
        buffer: &'static str,
        elements: usize,
    },
    GpuDimensionsUnsupported {
        kind: &'static str,
        actual: usize,
        supported: usize,
    },
    GpuExecutionUnavailable {
        kind: &'static str,
        reason: &'static str,
    },
    GpuBufferTooSmall {
        buffer: &'static str,
        required_bytes: u64,
        actual_bytes: u64,
    },
    GpuBufferMissingUsage {
        buffer: &'static str,
        required_usage: &'static str,
    },
    GpuBufferBindingTooLarge {
        buffer: &'static str,
        required_bytes: u64,
        limit_bytes: u64,
    },
    GpuBufferAliasUnsupported {
        first: &'static str,
        second: &'static str,
    },
    GpuDispatchUnsupported {
        workgroups: u32,
        max_per_dimension: u32,
    },
    GpuWorkgroupSizeUnsupported {
        requested: u32,
        maximum: u32,
    },
    GpuWorkgroupStorageUnsupported {
        requested_bytes: u32,
        maximum_bytes: u32,
    },
    GpuFineGridIndexUnsupported {
        length: usize,
        maximum: usize,
    },
    FftExecutionFailed {
        stage: &'static str,
        source: wgpu_fft::FftError,
    },
    InvalidKernelFourierCoefficient {
        mode: i64,
        value: f64,
    },
}

impl fmt::Display for NufftError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidDimensions {
                actual,
                minimum,
                maximum,
            } => write!(
                f,
                "invalid NUFFT dimension count {actual}; expected {minimum}..={maximum}"
            ),
            Self::ZeroMode { axis } => {
                write!(f, "NUFFT mode count on axis {axis} must be nonzero")
            }
            Self::InvalidTolerance { eps, reason } => {
                write!(f, "invalid NUFFT tolerance {eps}: {reason}")
            }
            Self::InvalidSigma { sigma, reason } => {
                write!(f, "invalid NUFFT upsampling factor {sigma}: {reason}")
            }
            Self::KernelWidthUnsupported {
                eps,
                width,
                maximum,
            } => write!(
                f,
                "NUFFT tolerance {eps} requires ES kernel width {width}, exceeding maximum {maximum}"
            ),
            Self::InvalidKernelWidth {
                width,
                minimum,
                maximum,
            } => write!(
                f,
                "invalid ES kernel width {width}; expected {minimum}..={maximum}"
            ),
            Self::LengthOverflow { context } => {
                write!(f, "NUFFT length arithmetic overflow while computing {context}")
            }
            Self::CoordinateLength {
                actual,
                dimensions,
            } => write!(
                f,
                "point-coordinate length {actual} is not divisible by dimension count {dimensions}"
            ),
            Self::NonFiniteCoordinate {
                point,
                axis,
                value,
            } => write!(
                f,
                "point {point} coordinate on axis {axis} is not finite: {value}"
            ),
            Self::PointOutOfRange {
                point,
                axis,
                value,
                maximum_magnitude,
            } => write!(
                f,
                "point {point} coordinate on axis {axis} is {value}; type-1/type-2 inputs require |x| <= {maximum_magnitude}"
            ),
            Self::InputLength {
                input,
                expected,
                actual,
            } => write!(
                f,
                "{input} length is {actual}; expected {expected} complex values"
            ),
            Self::NonFiniteComplex { input, index } => {
                write!(f, "{input} complex value at index {index} is not finite")
            }
            Self::FineGridMustBeEven { length } => {
                write!(f, "fine-grid length {length} must be even")
            }
            Self::FftShapeUnsupported { stage, source } => {
                write!(f, "{stage} is unsupported by wgpu-fft: {source}")
            }
            Self::HostAllocationFailed { buffer, elements } => write!(
                f,
                "could not allocate {elements} host elements for {buffer}"
            ),
            Self::GpuDimensionsUnsupported {
                kind,
                actual,
                supported,
            } => write!(
                f,
                "GPU {kind} execution supports up to {supported} dimensions, but the plan has {actual}"
            ),
            Self::GpuExecutionUnavailable { kind, reason } => {
                write!(f, "GPU {kind} execution is unavailable: {reason}")
            }
            Self::GpuBufferTooSmall {
                buffer,
                required_bytes,
                actual_bytes,
            } => write!(
                f,
                "GPU {buffer} buffer has {actual_bytes} bytes; at least {required_bytes} bytes are required"
            ),
            Self::GpuBufferMissingUsage {
                buffer,
                required_usage,
            } => write!(
                f,
                "GPU {buffer} buffer is missing required {required_usage} usage"
            ),
            Self::GpuBufferBindingTooLarge {
                buffer,
                required_bytes,
                limit_bytes,
            } => write!(
                f,
                "GPU {buffer} binding requires {required_bytes} bytes, exceeding the device limit of {limit_bytes} bytes"
            ),
            Self::GpuBufferAliasUnsupported { first, second } => write!(
                f,
                "GPU {first} and {second} buffers may not alias in the same compute pass"
            ),
            Self::GpuDispatchUnsupported {
                workgroups,
                max_per_dimension,
            } => write!(
                f,
                "GPU NUFFT dispatch needs {workgroups} workgroups, which cannot fit a 3D grid with per-dimension limit {max_per_dimension}"
            ),
            Self::GpuWorkgroupSizeUnsupported { requested, maximum } => write!(
                f,
                "GPU NUFFT workgroup size {requested} exceeds the device limit {maximum}"
            ),
            Self::GpuWorkgroupStorageUnsupported {
                requested_bytes,
                maximum_bytes,
            } => write!(
                f,
                "GPU NUFFT workgroup storage requires {requested_bytes} bytes, exceeding the device limit of {maximum_bytes} bytes"
            ),
            Self::GpuFineGridIndexUnsupported { length, maximum } => write!(
                f,
                "GPU NUFFT fine-grid length {length} exceeds the exact f32 shader-index limit {maximum}"
            ),
            Self::FftExecutionFailed { stage, source } => {
                write!(f, "wgpu-fft failed while encoding {stage}: {source}")
            }
            Self::InvalidKernelFourierCoefficient { mode, value } => write!(
                f,
                "ES kernel Fourier coefficient for mode {mode} cannot be inverted as f32: {value}"
            ),
        }
    }
}

impl std::error::Error for NufftError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::FftShapeUnsupported { source, .. } | Self::FftExecutionFailed { source, .. } => {
                Some(source)
            }
            _ => None,
        }
    }
}
