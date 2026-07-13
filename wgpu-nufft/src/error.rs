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
                "point {point} coordinate on axis {axis} is {value}; FINUFFT type-1/type-2 inputs require |x| <= {maximum_magnitude}"
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
        }
    }
}

impl std::error::Error for NufftError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::FftShapeUnsupported { source, .. } => Some(source),
            _ => None,
        }
    }
}
