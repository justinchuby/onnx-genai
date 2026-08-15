use onnx_runtime_ir::DataType;

use crate::SessionError;

/// Result type for sequence value and byte-helper operations.
pub type SequenceResult<T> = std::result::Result<T, SequenceError>;

/// A typed failure from an ONNX sequence operation.
#[derive(Debug, thiserror::Error)]
pub enum SequenceError {
    #[error(
        "SequenceConstruct requires at least one tensor; use SequenceValue::empty(dtype) for an empty sequence"
    )]
    EmptyConstruct,

    #[error(
        "{op} element{index_suffix} dtype {actual:?} does not match expected {expected:?}; ONNX sequences are homogeneous. To fix: Cast the tensor to {expected:?}",
        index_suffix = index.map(|i| format!(" {i}")).unwrap_or_default()
    )]
    DtypeMismatch {
        op: &'static str,
        index: Option<usize>,
        expected: DataType,
        actual: DataType,
    },

    #[error(
        "{op} index {index} is out of bounds for a sequence of length {len} (valid range {range})",
        range = if *insertion {
            format!("[{}, {}]", -(*len as i128), *len)
        } else if *len == 0 {
            "empty (no valid indices)".to_string()
        } else {
            format!("[{}, {}]", -(*len as i128), *len as i128 - 1)
        }
    )]
    IndexOutOfBounds {
        op: &'static str,
        index: i64,
        len: usize,
        insertion: bool,
    },

    #[error("SequenceErase cannot erase from an empty sequence")]
    EmptyErase,

    #[error("{op} cannot represent sequence length {len} as an ONNX index")]
    LengthOverflow { op: &'static str, len: usize },

    #[error(
        "{op} axis {axis} is invalid for rank {rank}{new_axis_suffix}",
        new_axis_suffix = if *new_axis { " with new_axis=1" } else { "" }
    )]
    InvalidAxis {
        op: &'static str,
        axis: i64,
        rank: usize,
        new_axis: bool,
    },

    #[error("{op} has invalid split specification: {reason}")]
    InvalidSplit { op: &'static str, reason: String },

    #[error(
        "{op} element {index} has shape {actual:?}, incompatible with {expected:?}: {requirement}"
    )]
    ShapeMismatch {
        op: &'static str,
        index: usize,
        expected: Vec<usize>,
        actual: Vec<usize>,
        requirement: &'static str,
    },

    #[error("{op} does not support byte operations for sub-byte dtype {dtype:?}")]
    UnsupportedDtype { op: &'static str, dtype: DataType },

    #[error("{op} requires host-accessible sequence tensors, but element {index} is on {device}")]
    NonHostTensor {
        op: &'static str,
        index: usize,
        device: String,
    },

    #[error(
        "SequenceTensor cannot borrow contiguous bytes for shape {shape:?} on {device}: {reason}; use contiguous_bytes() to materialize the view"
    )]
    ByteBorrowUnavailable {
        shape: Vec<usize>,
        device: String,
        reason: &'static str,
    },

    #[error(
        "{op} received {actual} bytes for shape {shape:?} dtype {dtype:?}, expected {expected}"
    )]
    ByteLengthMismatch {
        op: &'static str,
        dtype: DataType,
        shape: Vec<usize>,
        expected: usize,
        actual: usize,
    },

    #[error("{op} shape/offset overflow while computing {context} for shape {shape:?}")]
    ShapeOverflow {
        op: &'static str,
        context: &'static str,
        shape: Vec<usize>,
    },

    #[error("{op} cannot allocate {bytes} bytes for {context}")]
    Allocation {
        op: &'static str,
        context: &'static str,
        bytes: usize,
    },

    #[error("{op} could not create a tensor: {source}")]
    TensorCreation {
        op: &'static str,
        #[source]
        source: SessionError,
    },
}

impl SequenceError {
    pub(crate) fn op(&self) -> &'static str {
        match self {
            Self::EmptyConstruct => "SequenceConstruct",
            Self::DtypeMismatch { op, .. }
            | Self::IndexOutOfBounds { op, .. }
            | Self::LengthOverflow { op, .. }
            | Self::InvalidAxis { op, .. }
            | Self::InvalidSplit { op, .. }
            | Self::ShapeMismatch { op, .. }
            | Self::UnsupportedDtype { op, .. }
            | Self::NonHostTensor { op, .. }
            | Self::ByteLengthMismatch { op, .. }
            | Self::ShapeOverflow { op, .. }
            | Self::Allocation { op, .. }
            | Self::TensorCreation { op, .. } => op,
            Self::EmptyErase => "SequenceErase",
            Self::ByteBorrowUnavailable { .. } => "SequenceTensor",
        }
    }
}

impl From<SequenceError> for SessionError {
    fn from(error: SequenceError) -> Self {
        if let SequenceError::ShapeOverflow { context, shape, .. } = error {
            return SessionError::ShapeOverflow {
                value: context.to_string(),
                dims: shape,
            };
        }
        let op = error.op().to_string();
        SessionError::SequenceOp {
            op,
            reason: error.to_string(),
        }
    }
}
