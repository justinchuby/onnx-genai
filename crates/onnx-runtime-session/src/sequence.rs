//! ONNX sequence values and byte-oriented Split/Concat helpers.
//!
//! Sequence elements reuse the session crate's [`Tensor`] representation. A
//! [`SeqTensor`] is an immutable `Arc<Tensor>` handle, so constructing, inserting,
//! erasing, and indexing a sequence only clone `Arc` handles; tensor storage is
//! never deep-copied by those operations.

use std::ops::Deref;
use std::sync::Arc;

use onnx_runtime_ir::DataType;

use crate::{SessionError, Tensor};

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

/// An immutable tensor handle used as one sequence element.
///
/// Cloning this type only bumps the `Arc` count. The contained [`Tensor`] and
/// its device allocation are shared without copying.
#[derive(Clone, Debug)]
pub struct SeqTensor {
    tensor: Arc<Tensor>,
}

impl SeqTensor {
    /// Wrap an existing session tensor in an immutable shared handle.
    pub fn new(tensor: Tensor) -> Self {
        Self {
            tensor: Arc::new(tensor),
        }
    }

    /// Build a host tensor from raw element bytes and share it as a sequence item.
    pub fn from_raw(dtype: DataType, shape: Vec<usize>, bytes: &[u8]) -> SequenceResult<Self> {
        validate_tensor_bytes("SequenceTensor", bytes, dtype, &shape)?;
        Tensor::from_raw(dtype, shape, bytes)
            .map(Self::new)
            .map_err(|source| SequenceError::TensorCreation {
                op: "SequenceTensor",
                source,
            })
    }

    /// The shared session tensor. Use this with [`Arc::ptr_eq`] to verify sharing.
    pub fn shared_tensor(&self) -> &Arc<Tensor> {
        &self.tensor
    }

    /// Base address of the shared tensor allocation.
    pub fn as_ptr(&self) -> *const std::ffi::c_void {
        self.tensor.device_ptr()
    }
}

impl Deref for SeqTensor {
    type Target = Tensor;

    fn deref(&self) -> &Self::Target {
        &self.tensor
    }
}

/// An ordered homogeneous list of immutable, shared tensors.
#[derive(Clone, Debug)]
pub struct SequenceValue {
    pub(crate) elem_dtype: DataType,
    pub(crate) items: Vec<SeqTensor>,
}

impl SequenceValue {
    /// Construct an empty sequence with its declared tensor element dtype.
    pub fn empty(elem_dtype: DataType) -> Self {
        Self {
            elem_dtype,
            items: Vec::new(),
        }
    }

    /// Construct a sequence without copying any element tensor storage.
    pub fn construct(items: Vec<SeqTensor>) -> SequenceResult<Self> {
        let elem_dtype = items
            .first()
            .map(|tensor| tensor.dtype)
            .ok_or(SequenceError::EmptyConstruct)?;
        for (index, tensor) in items.iter().enumerate() {
            if tensor.dtype != elem_dtype {
                return Err(SequenceError::DtypeMismatch {
                    op: "SequenceConstruct",
                    index: Some(index),
                    expected: elem_dtype,
                    actual: tensor.dtype,
                });
            }
        }
        Ok(Self { elem_dtype, items })
    }

    /// Return a new sequence with `value` inserted at `at`.
    ///
    /// `None` appends. Negative positions count from the end, and `len` is also
    /// accepted as an explicit append position.
    pub fn insert(&self, value: SeqTensor, at: Option<i64>) -> SequenceResult<Self> {
        if value.dtype != self.elem_dtype {
            return Err(SequenceError::DtypeMismatch {
                op: "SequenceInsert",
                index: None,
                expected: self.elem_dtype,
                actual: value.dtype,
            });
        }
        let index = match at {
            None => self.items.len(),
            Some(index) => resolve_index("SequenceInsert", index, self.items.len(), true)?,
        };
        let capacity = self
            .items
            .len()
            .checked_add(1)
            .ok_or(SequenceError::LengthOverflow {
                op: "SequenceInsert",
                len: self.items.len(),
            })?;
        let mut items = Vec::new();
        items
            .try_reserve_exact(capacity)
            .map_err(|_| SequenceError::Allocation {
                op: "SequenceInsert",
                context: "sequence handles",
                bytes: capacity.saturating_mul(std::mem::size_of::<SeqTensor>()),
            })?;
        items.extend_from_slice(&self.items[..index]);
        items.push(value);
        items.extend_from_slice(&self.items[index..]);
        Ok(Self {
            elem_dtype: self.elem_dtype,
            items,
        })
    }

    /// Return a new sequence with the selected element erased.
    ///
    /// `None` erases the last element. Negative indices count from the end.
    pub fn erase(&self, at: Option<i64>) -> SequenceResult<Self> {
        if self.items.is_empty() {
            return Err(SequenceError::EmptyErase);
        }
        let index = match at {
            None => self.items.len() - 1,
            Some(index) => resolve_index("SequenceErase", index, self.items.len(), false)?,
        };
        let capacity = self.items.len() - 1;
        let mut items = Vec::new();
        items
            .try_reserve_exact(capacity)
            .map_err(|_| SequenceError::Allocation {
                op: "SequenceErase",
                context: "sequence handles",
                bytes: capacity.saturating_mul(std::mem::size_of::<SeqTensor>()),
            })?;
        items.extend_from_slice(&self.items[..index]);
        items.extend_from_slice(&self.items[index + 1..]);
        Ok(Self {
            elem_dtype: self.elem_dtype,
            items,
        })
    }

    /// Return the selected shared tensor handle. Negative indices count from the end.
    pub fn at(&self, index: i64) -> SequenceResult<SeqTensor> {
        let index = resolve_index("SequenceAt", index, self.items.len(), false)?;
        Ok(self.items[index].clone())
    }

    /// Number of elements, matching ONNX `SequenceLength`.
    pub fn length(&self) -> usize {
        self.items.len()
    }

    pub(crate) fn len(&self) -> usize {
        self.length()
    }

    /// Declared homogeneous element dtype.
    pub fn elem_dtype(&self) -> DataType {
        self.elem_dtype
    }

    /// Ordered shared tensor handles.
    pub fn elements(&self) -> &[SeqTensor] {
        &self.items
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.items.is_empty()
    }
}

/// ONNX `SplitToSequence` split-input interpretation.
#[derive(Clone, Copy, Debug)]
pub enum SplitSpec<'a> {
    /// No split input: emit one slice per index along the selected axis.
    Each,
    /// Scalar split input: repeatedly take chunks of this size.
    Chunk(i64),
    /// Rank-1 split input: explicit extents that must sum to the axis extent.
    Sizes(&'a [i64]),
}

/// Split raw contiguous tensor bytes into a sequence of session tensors.
///
/// `keepdims` only affects [`SplitSpec::Each`], as required by ONNX; an explicit
/// split input always retains the split axis.
pub fn split(
    data: &[u8],
    dtype: DataType,
    shape: &[usize],
    axis: i64,
    split: SplitSpec<'_>,
    keepdims: bool,
) -> SequenceResult<SequenceValue> {
    const OP: &str = "SplitToSequence";
    let rank = shape.len();
    if rank == 0 {
        return Err(SequenceError::InvalidAxis {
            op: OP,
            axis,
            rank,
            new_axis: false,
        });
    }
    let axis = normalize_axis(OP, axis, rank, false)?;
    validate_tensor_bytes(OP, data, dtype, shape)?;
    let axis_dim = shape[axis];

    let (sizes, squeeze) = match split {
        SplitSpec::Each => {
            let mut sizes = Vec::new();
            sizes
                .try_reserve_exact(axis_dim)
                .map_err(|_| SequenceError::Allocation {
                    op: OP,
                    context: "split sizes",
                    bytes: axis_dim.saturating_mul(std::mem::size_of::<usize>()),
                })?;
            sizes.resize(axis_dim, 1);
            (sizes, !keepdims)
        }
        SplitSpec::Chunk(chunk) => {
            if chunk <= 0 {
                return Err(SequenceError::InvalidSplit {
                    op: OP,
                    reason: format!("scalar chunk size {chunk} must be positive"),
                });
            }
            let chunk = usize::try_from(chunk).map_err(|_| SequenceError::InvalidSplit {
                op: OP,
                reason: format!("scalar chunk size {chunk} cannot be represented"),
            })?;
            let count = axis_dim
                .checked_add(chunk - 1)
                .ok_or_else(|| overflow(OP, "split chunk count", shape))?
                / chunk;
            let mut sizes = Vec::new();
            sizes
                .try_reserve_exact(count)
                .map_err(|_| SequenceError::Allocation {
                    op: OP,
                    context: "split sizes",
                    bytes: count.saturating_mul(std::mem::size_of::<usize>()),
                })?;
            let mut remaining = axis_dim;
            while remaining != 0 {
                let size = remaining.min(chunk);
                sizes.push(size);
                remaining -= size;
            }
            (sizes, false)
        }
        SplitSpec::Sizes(values) => {
            let mut sizes = Vec::new();
            sizes
                .try_reserve_exact(values.len())
                .map_err(|_| SequenceError::Allocation {
                    op: OP,
                    context: "split sizes",
                    bytes: values.len().saturating_mul(std::mem::size_of::<usize>()),
                })?;
            let mut sum = 0usize;
            for &value in values {
                let value = usize::try_from(value).map_err(|_| SequenceError::InvalidSplit {
                    op: OP,
                    reason: format!("size {value} must be non-negative"),
                })?;
                sum = sum
                    .checked_add(value)
                    .ok_or_else(|| overflow(OP, "split size sum", shape))?;
                sizes.push(value);
            }
            if sum != axis_dim {
                return Err(SequenceError::InvalidSplit {
                    op: OP,
                    reason: format!("sizes sum to {sum}, but axis {axis} has extent {axis_dim}"),
                });
            }
            (sizes, false)
        }
    };

    let parts = split_axis(data, shape, axis, &sizes, dtype.byte_size())?;
    let mut items = Vec::new();
    items
        .try_reserve_exact(parts.len())
        .map_err(|_| SequenceError::Allocation {
            op: OP,
            context: "sequence handles",
            bytes: parts.len().saturating_mul(std::mem::size_of::<SeqTensor>()),
        })?;
    for (mut part_shape, bytes) in parts {
        if squeeze {
            part_shape.remove(axis);
        }
        let tensor = Tensor::from_raw(dtype, part_shape, &bytes)
            .map_err(|source| SequenceError::TensorCreation { op: OP, source })?;
        items.push(SeqTensor::new(tensor));
    }
    Ok(SequenceValue {
        elem_dtype: dtype,
        items,
    })
}

/// Concatenate a sequence along an existing axis or stack it on a new axis.
pub fn concat(sequence: &SequenceValue, axis: i64, new_axis: bool) -> SequenceResult<SeqTensor> {
    const OP: &str = "ConcatFromSequence";
    let first = sequence.items.first().ok_or(SequenceError::InvalidSplit {
        op: OP,
        reason: "cannot concatenate an empty sequence".to_string(),
    })?;
    let dtype = sequence.elem_dtype;
    let esize = dtype.byte_size();
    if esize == 0 {
        return Err(SequenceError::UnsupportedDtype { op: OP, dtype });
    }
    let rank = first.shape.len();
    let axis = normalize_axis(OP, axis, rank + usize::from(new_axis), new_axis)?;

    let mut shapes = Vec::new();
    let mut elements = Vec::new();
    shapes
        .try_reserve_exact(sequence.items.len())
        .map_err(|_| SequenceError::Allocation {
            op: OP,
            context: "element shapes",
            bytes: sequence
                .items
                .len()
                .saturating_mul(std::mem::size_of::<Vec<usize>>()),
        })?;
    elements
        .try_reserve_exact(sequence.items.len())
        .map_err(|_| SequenceError::Allocation {
            op: OP,
            context: "element byte slices",
            bytes: sequence
                .items
                .len()
                .saturating_mul(std::mem::size_of::<&[u8]>()),
        })?;
    for (index, item) in sequence.items.iter().enumerate() {
        if !item.device().is_host_accessible() {
            return Err(SequenceError::NonHostTensor {
                op: OP,
                index,
                device: format!("{:?}", item.device()),
            });
        }
        validate_tensor_bytes(OP, item.as_bytes(), item.dtype, &item.shape)?;
        shapes.push(clone_shape(OP, &item.shape)?);
        elements.push(item.as_bytes());
    }

    let (shape, bytes) = if new_axis {
        for (index, shape) in shapes.iter().enumerate().skip(1) {
            if shape != &shapes[0] {
                return Err(SequenceError::ShapeMismatch {
                    op: OP,
                    index,
                    expected: clone_shape(OP, &shapes[0])?,
                    actual: clone_shape(OP, shape)?,
                    requirement: "new_axis=1 requires identical shapes",
                });
            }
        }
        stack_new_axis(&elements, &shapes[0], axis, esize)?
    } else {
        for (index, shape) in shapes.iter().enumerate().skip(1) {
            let mismatch = shape.len() != rank
                || shape.iter().enumerate().any(|(dimension, &extent)| {
                    dimension != axis && extent != shapes[0][dimension]
                });
            if mismatch {
                return Err(SequenceError::ShapeMismatch {
                    op: OP,
                    index,
                    expected: clone_shape(OP, &shapes[0])?,
                    actual: clone_shape(OP, shape)?,
                    requirement: "all dimensions except the concat axis must match",
                });
            }
        }
        concat_axis(&elements, &shapes, axis, esize)?
    };
    let tensor = Tensor::from_raw(dtype, shape, &bytes)
        .map_err(|source| SequenceError::TensorCreation { op: OP, source })?;
    Ok(SeqTensor::new(tensor))
}

fn resolve_index(
    op: &'static str,
    index: i64,
    len: usize,
    insertion: bool,
) -> SequenceResult<usize> {
    let length = i64::try_from(len).map_err(|_| SequenceError::LengthOverflow { op, len })?;
    let resolved = if index < 0 {
        length.checked_add(index)
    } else {
        Some(index)
    };
    let valid = resolved.is_some_and(|value| {
        value >= 0
            && if insertion {
                value <= length
            } else {
                value < length
            }
    });
    if !valid {
        return Err(SequenceError::IndexOutOfBounds {
            op,
            index,
            len,
            insertion,
        });
    }
    usize::try_from(resolved.unwrap_or_default())
        .map_err(|_| SequenceError::LengthOverflow { op, len })
}

fn normalize_axis(
    op: &'static str,
    axis: i64,
    rank: usize,
    new_axis: bool,
) -> SequenceResult<usize> {
    let rank_i64 =
        i64::try_from(rank).map_err(|_| SequenceError::LengthOverflow { op, len: rank })?;
    let normalized = if axis < 0 {
        rank_i64.checked_add(axis)
    } else {
        Some(axis)
    };
    match normalized {
        Some(axis) if axis >= 0 && axis < rank_i64 => Ok(axis as usize),
        _ => Err(SequenceError::InvalidAxis {
            op,
            axis,
            rank: rank - usize::from(new_axis),
            new_axis,
        }),
    }
}

fn overflow(op: &'static str, context: &'static str, shape: &[usize]) -> SequenceError {
    SequenceError::ShapeOverflow {
        op,
        context,
        shape: shape.to_vec(),
    }
}

/// Multiply dimensions while still detecting overflow hidden by a zero extent.
fn checked_product(
    op: &'static str,
    context: &'static str,
    shape: &[usize],
) -> SequenceResult<usize> {
    let mut product = 1usize;
    let mut has_zero = false;
    for &dimension in shape {
        if dimension == 0 {
            has_zero = true;
        } else {
            product = product
                .checked_mul(dimension)
                .ok_or_else(|| overflow(op, context, shape))?;
        }
    }
    Ok(if has_zero { 0 } else { product })
}

fn checked_mul(
    op: &'static str,
    context: &'static str,
    lhs: usize,
    rhs: usize,
    shape: &[usize],
) -> SequenceResult<usize> {
    lhs.checked_mul(rhs)
        .ok_or_else(|| overflow(op, context, shape))
}

fn checked_add(
    op: &'static str,
    context: &'static str,
    lhs: usize,
    rhs: usize,
    shape: &[usize],
) -> SequenceResult<usize> {
    lhs.checked_add(rhs)
        .ok_or_else(|| overflow(op, context, shape))
}

fn addressable(
    op: &'static str,
    context: &'static str,
    bytes: usize,
    shape: &[usize],
) -> SequenceResult<usize> {
    if bytes > isize::MAX as usize {
        return Err(overflow(op, context, shape));
    }
    Ok(bytes)
}

fn zeroed_bytes(
    op: &'static str,
    context: &'static str,
    bytes: usize,
    shape: &[usize],
) -> SequenceResult<Vec<u8>> {
    addressable(op, context, bytes, shape)?;
    let mut output = Vec::new();
    output
        .try_reserve_exact(bytes)
        .map_err(|_| SequenceError::Allocation { op, context, bytes })?;
    output.resize(bytes, 0);
    Ok(output)
}

fn clone_shape(op: &'static str, shape: &[usize]) -> SequenceResult<Vec<usize>> {
    let bytes = shape
        .len()
        .checked_mul(std::mem::size_of::<usize>())
        .ok_or_else(|| overflow(op, "shape allocation", shape))?;
    let mut cloned = Vec::new();
    cloned
        .try_reserve_exact(shape.len())
        .map_err(|_| SequenceError::Allocation {
            op,
            context: "shape",
            bytes,
        })?;
    cloned.extend_from_slice(shape);
    Ok(cloned)
}

fn validate_tensor_bytes(
    op: &'static str,
    data: &[u8],
    dtype: DataType,
    shape: &[usize],
) -> SequenceResult<()> {
    if dtype.byte_size() == 0 {
        return Err(SequenceError::UnsupportedDtype { op, dtype });
    }
    let numel = checked_product(op, "tensor element count", shape)?;
    let expected = dtype
        .checked_storage_bytes(numel)
        .ok_or_else(|| overflow(op, "tensor byte count", shape))?;
    addressable(op, "tensor byte count", expected, shape)?;
    if data.len() != expected {
        return Err(SequenceError::ByteLengthMismatch {
            op,
            dtype,
            shape: clone_shape(op, shape)?,
            expected,
            actual: data.len(),
        });
    }
    Ok(())
}

/// Split already-validated contiguous bytes along one normalized axis.
pub(crate) fn split_axis(
    data: &[u8],
    shape: &[usize],
    axis: usize,
    sizes: &[usize],
    esize: usize,
) -> SequenceResult<Vec<(Vec<usize>, Vec<u8>)>> {
    const OP: &str = "SplitToSequence";
    if axis >= shape.len() {
        return Err(SequenceError::InvalidAxis {
            op: OP,
            axis: i64::try_from(axis).unwrap_or(i64::MAX),
            rank: shape.len(),
            new_axis: false,
        });
    }
    if esize == 0 {
        return Err(SequenceError::InvalidSplit {
            op: OP,
            reason: "element byte size must be positive".to_string(),
        });
    }
    let axis_dim = shape[axis];
    let mut size_sum = 0usize;
    for &size in sizes {
        size_sum = checked_add(OP, "split size sum", size_sum, size, shape)?;
    }
    if size_sum != axis_dim {
        return Err(SequenceError::InvalidSplit {
            op: OP,
            reason: format!("sizes sum to {size_sum}, but axis {axis} has extent {axis_dim}"),
        });
    }

    let outer = checked_product(OP, "split outer element count", &shape[..axis])?;
    let inner_elements = checked_product(OP, "split inner element count", &shape[axis + 1..])?;
    let inner = checked_mul(OP, "split inner byte count", inner_elements, esize, shape)?;
    let input_rows = checked_mul(OP, "split input row count", outer, axis_dim, shape)?;
    let input_bytes = checked_mul(OP, "split input byte count", input_rows, inner, shape)?;
    addressable(OP, "split input byte count", input_bytes, shape)?;
    if data.len() != input_bytes {
        return Err(SequenceError::ByteLengthMismatch {
            op: OP,
            dtype: DataType::Uint8,
            shape: clone_shape(OP, shape)?,
            expected: input_bytes,
            actual: data.len(),
        });
    }

    let mut output = Vec::new();
    output
        .try_reserve_exact(sizes.len())
        .map_err(|_| SequenceError::Allocation {
            op: OP,
            context: "split outputs",
            bytes: sizes
                .len()
                .saturating_mul(std::mem::size_of::<(Vec<usize>, Vec<u8>)>()),
        })?;
    let mut start = 0usize;
    for &size in sizes {
        let output_rows = checked_mul(OP, "split output row count", outer, size, shape)?;
        let output_bytes = checked_mul(OP, "split output byte count", output_rows, inner, shape)?;
        let mut bytes = zeroed_bytes(OP, "split output", output_bytes, shape)?;
        if inner != 0 && size != 0 {
            let copy_bytes = checked_mul(OP, "split copy width", size, inner, shape)?;
            for outer_index in 0..outer {
                let source_row = checked_add(
                    OP,
                    "split source row",
                    checked_mul(
                        OP,
                        "split source outer offset",
                        outer_index,
                        axis_dim,
                        shape,
                    )?,
                    start,
                    shape,
                )?;
                let source_offset =
                    checked_mul(OP, "split source byte offset", source_row, inner, shape)?;
                let source_end = checked_add(
                    OP,
                    "split source byte range",
                    source_offset,
                    copy_bytes,
                    shape,
                )?;
                let destination_offset = checked_mul(
                    OP,
                    "split destination byte offset",
                    outer_index,
                    copy_bytes,
                    shape,
                )?;
                let destination_end = checked_add(
                    OP,
                    "split destination byte range",
                    destination_offset,
                    copy_bytes,
                    shape,
                )?;
                bytes[destination_offset..destination_end]
                    .copy_from_slice(&data[source_offset..source_end]);
            }
        }
        let mut output_shape = clone_shape(OP, shape)?;
        output_shape[axis] = size;
        output.push((output_shape, bytes));
        start = checked_add(OP, "split axis cursor", start, size, shape)?;
    }
    Ok(output)
}

/// Concatenate already-validated contiguous element bytes along an existing axis.
pub(crate) fn concat_axis(
    elements: &[&[u8]],
    shapes: &[Vec<usize>],
    axis: usize,
    esize: usize,
) -> SequenceResult<(Vec<usize>, Vec<u8>)> {
    const OP: &str = "ConcatFromSequence";
    let base = shapes.first().ok_or(SequenceError::InvalidSplit {
        op: OP,
        reason: "cannot concatenate an empty sequence".to_string(),
    })?;
    if elements.len() != shapes.len() || axis >= base.len() || esize == 0 {
        return Err(SequenceError::InvalidSplit {
            op: OP,
            reason: "invalid element/shape count, axis, or element byte size".to_string(),
        });
    }
    let outer = checked_product(OP, "concat outer element count", &base[..axis])?;
    let inner_elements = checked_product(OP, "concat inner element count", &base[axis + 1..])?;
    let inner = checked_mul(OP, "concat inner byte count", inner_elements, esize, base)?;
    let mut total_axis = 0usize;
    for shape in shapes {
        total_axis = checked_add(OP, "concat axis extent", total_axis, shape[axis], base)?;
    }
    let output_rows = checked_mul(OP, "concat output row count", outer, total_axis, base)?;
    let output_bytes = checked_mul(OP, "concat output byte count", output_rows, inner, base)?;
    let mut bytes = zeroed_bytes(OP, "concat output", output_bytes, base)?;

    for (index, (element, shape)) in elements.iter().zip(shapes).enumerate() {
        let rows = checked_mul(OP, "concat source row count", outer, shape[axis], shape)?;
        let expected = checked_mul(OP, "concat source byte count", rows, inner, shape)?;
        addressable(OP, "concat source byte count", expected, shape)?;
        if element.len() != expected {
            return Err(SequenceError::ByteLengthMismatch {
                op: OP,
                dtype: DataType::Uint8,
                shape: clone_shape(OP, shape)?,
                expected,
                actual: element.len(),
            });
        }
        if shape.len() != base.len() {
            return Err(SequenceError::ShapeMismatch {
                op: OP,
                index,
                expected: clone_shape(OP, base)?,
                actual: clone_shape(OP, shape)?,
                requirement: "ranks must match",
            });
        }
    }

    if inner != 0 && total_axis != 0 {
        for outer_index in 0..outer {
            let mut axis_cursor = 0usize;
            for (element, shape) in elements.iter().zip(shapes) {
                let size = shape[axis];
                let copy_bytes = checked_mul(OP, "concat copy width", size, inner, base)?;
                let source_offset = checked_mul(
                    OP,
                    "concat source byte offset",
                    outer_index,
                    copy_bytes,
                    base,
                )?;
                let source_end = checked_add(
                    OP,
                    "concat source byte range",
                    source_offset,
                    copy_bytes,
                    base,
                )?;
                let destination_row = checked_add(
                    OP,
                    "concat destination row",
                    checked_mul(
                        OP,
                        "concat destination outer offset",
                        outer_index,
                        total_axis,
                        base,
                    )?,
                    axis_cursor,
                    base,
                )?;
                let destination_offset = checked_mul(
                    OP,
                    "concat destination byte offset",
                    destination_row,
                    inner,
                    base,
                )?;
                let destination_end = checked_add(
                    OP,
                    "concat destination byte range",
                    destination_offset,
                    copy_bytes,
                    base,
                )?;
                bytes[destination_offset..destination_end]
                    .copy_from_slice(&element[source_offset..source_end]);
                axis_cursor = checked_add(OP, "concat axis cursor", axis_cursor, size, base)?;
            }
        }
    }
    let mut output_shape = clone_shape(OP, base)?;
    output_shape[axis] = total_axis;
    Ok((output_shape, bytes))
}

/// Stack already-validated contiguous element bytes along a new axis.
pub(crate) fn stack_new_axis(
    elements: &[&[u8]],
    elem_shape: &[usize],
    axis: usize,
    esize: usize,
) -> SequenceResult<(Vec<usize>, Vec<u8>)> {
    const OP: &str = "ConcatFromSequence";
    if axis > elem_shape.len() || esize == 0 {
        return Err(SequenceError::InvalidSplit {
            op: OP,
            reason: "invalid new axis or element byte size".to_string(),
        });
    }
    let outer = checked_product(OP, "stack outer element count", &elem_shape[..axis])?;
    let inner_elements = checked_product(OP, "stack inner element count", &elem_shape[axis..])?;
    let inner = checked_mul(
        OP,
        "stack inner byte count",
        inner_elements,
        esize,
        elem_shape,
    )?;
    let source_bytes = checked_mul(OP, "stack source byte count", outer, inner, elem_shape)?;
    addressable(OP, "stack source byte count", source_bytes, elem_shape)?;
    for element in elements {
        if element.len() != source_bytes {
            return Err(SequenceError::ByteLengthMismatch {
                op: OP,
                dtype: DataType::Uint8,
                shape: clone_shape(OP, elem_shape)?,
                expected: source_bytes,
                actual: element.len(),
            });
        }
    }
    let output_rows = checked_mul(
        OP,
        "stacked tensor output row count",
        elements.len(),
        outer,
        elem_shape,
    )?;
    let output_bytes = checked_mul(
        OP,
        "stack output byte count",
        output_rows,
        inner,
        elem_shape,
    )?;
    let mut bytes = zeroed_bytes(OP, "stack output", output_bytes, elem_shape)?;
    if inner != 0 {
        for (element_index, element) in elements.iter().enumerate() {
            for outer_index in 0..outer {
                let source_offset = checked_mul(
                    OP,
                    "stack source byte offset",
                    outer_index,
                    inner,
                    elem_shape,
                )?;
                let source_end = checked_add(
                    OP,
                    "stack source byte range",
                    source_offset,
                    inner,
                    elem_shape,
                )?;
                let destination_row = checked_add(
                    OP,
                    "stack destination row",
                    checked_mul(
                        OP,
                        "stack destination outer offset",
                        outer_index,
                        elements.len(),
                        elem_shape,
                    )?,
                    element_index,
                    elem_shape,
                )?;
                let destination_offset = checked_mul(
                    OP,
                    "stack destination byte offset",
                    destination_row,
                    inner,
                    elem_shape,
                )?;
                let destination_end = checked_add(
                    OP,
                    "stack destination byte range",
                    destination_offset,
                    inner,
                    elem_shape,
                )?;
                bytes[destination_offset..destination_end]
                    .copy_from_slice(&element[source_offset..source_end]);
            }
        }
    }
    let shape_capacity = elem_shape
        .len()
        .checked_add(1)
        .ok_or_else(|| overflow(OP, "stack output rank", elem_shape))?;
    let mut output_shape = Vec::new();
    output_shape
        .try_reserve_exact(shape_capacity)
        .map_err(|_| SequenceError::Allocation {
            op: OP,
            context: "stack output shape",
            bytes: shape_capacity.saturating_mul(std::mem::size_of::<usize>()),
        })?;
    output_shape.extend_from_slice(&elem_shape[..axis]);
    output_shape.push(elements.len());
    output_shape.extend_from_slice(&elem_shape[axis..]);
    Ok((output_shape, bytes))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn elem(dtype: DataType, shape: &[usize], bytes: &[u8]) -> SeqTensor {
        SeqTensor::from_raw(dtype, shape.to_vec(), bytes).expect("valid test tensor")
    }

    #[test]
    fn value_ops_share_tensor_arcs_without_copying() {
        let original = elem(DataType::Uint8, &[1], &[7]);
        let sequence = SequenceValue::construct(vec![original.clone()]).expect("construct");
        assert!(Arc::ptr_eq(
            original.shared_tensor(),
            sequence.elements()[0].shared_tensor()
        ));

        let inserted = elem(DataType::Uint8, &[1], &[9]);
        let sequence = sequence.insert(inserted.clone(), Some(-1)).expect("insert");
        assert!(Arc::ptr_eq(
            inserted.shared_tensor(),
            sequence.at(0).expect("at").shared_tensor()
        ));
        assert!(Arc::ptr_eq(
            original.shared_tensor(),
            sequence.at(1).expect("at").shared_tensor()
        ));

        let erased = sequence.erase(Some(0)).expect("erase");
        assert!(Arc::ptr_eq(
            original.shared_tensor(),
            erased.at(-1).expect("negative at").shared_tensor()
        ));
    }

    #[test]
    fn empty_construct_insert_erase_at_and_length() {
        let empty = SequenceValue::empty(DataType::Uint8);
        assert_eq!(empty.length(), 0);
        let one = empty
            .insert(elem(DataType::Uint8, &[1], &[1]), None)
            .unwrap();
        let two = one
            .insert(elem(DataType::Uint8, &[1], &[2]), Some(-1))
            .unwrap();
        let three = two
            .insert(elem(DataType::Uint8, &[1], &[3]), Some(2))
            .unwrap();
        assert_eq!(three.length(), 3);
        assert_eq!(three.at(0).unwrap().as_bytes(), &[2]);
        assert_eq!(three.at(-1).unwrap().as_bytes(), &[3]);
        let erased = three.erase(Some(-2)).unwrap();
        assert_eq!(erased.length(), 2);
        assert_eq!(erased.at(0).unwrap().as_bytes(), &[2]);
        assert_eq!(erased.at(1).unwrap().as_bytes(), &[3]);
    }

    #[test]
    fn homogeneity_violation_is_typed_error() {
        let error = SequenceValue::construct(vec![
            elem(DataType::Uint8, &[1], &[1]),
            elem(DataType::Int64, &[1], &1i64.to_le_bytes()),
        ])
        .unwrap_err();
        assert!(matches!(
            error,
            SequenceError::DtypeMismatch {
                op: "SequenceConstruct",
                index: Some(1),
                ..
            }
        ));
    }

    #[test]
    fn split_concat_roundtrip_existing_axes_and_keepdims() {
        let data: Vec<u8> = (0..12).collect();
        for (shape, axis, keepdims) in [
            (vec![3, 4], 0, true),
            (vec![3, 4], 1, true),
            (vec![3, 4], 0, false),
        ] {
            let sequence = split(
                &data,
                DataType::Uint8,
                &shape,
                axis,
                SplitSpec::Each,
                keepdims,
            )
            .unwrap();
            let concat_axis = if keepdims { axis } else { 0 };
            let rebuilt = concat(&sequence, concat_axis, !keepdims).unwrap();
            assert_eq!(rebuilt.shape, shape);
            assert_eq!(rebuilt.as_bytes(), data);
        }
    }

    #[test]
    fn split_concat_roundtrip_explicit_sizes() {
        let data: Vec<u8> = (0..12).collect();
        let sequence = split(
            &data,
            DataType::Uint8,
            &[3, 4],
            1,
            SplitSpec::Sizes(&[1, 3]),
            false,
        )
        .unwrap();
        let rebuilt = concat(&sequence, 1, false).unwrap();
        assert_eq!(rebuilt.shape, vec![3, 4]);
        assert_eq!(rebuilt.as_bytes(), data);
    }

    #[test]
    fn stack_new_axis_variants() {
        let a = elem(DataType::Uint8, &[2], &[1, 2]);
        let b = elem(DataType::Uint8, &[2], &[3, 4]);
        let sequence = SequenceValue::construct(vec![a, b]).unwrap();
        let front = concat(&sequence, 0, true).unwrap();
        assert_eq!(front.shape, vec![2, 2]);
        assert_eq!(front.as_bytes(), &[1, 2, 3, 4]);
        let back = concat(&sequence, 1, true).unwrap();
        assert_eq!(back.shape, vec![2, 2]);
        assert_eq!(back.as_bytes(), &[1, 3, 2, 4]);
    }

    #[test]
    fn byte_count_above_isize_max_is_rejected() {
        let error = split_axis(&[], &[isize::MAX as usize + 1, 1], 1, &[1], 1).unwrap_err();
        assert!(matches!(error, SequenceError::ShapeOverflow { .. }));
    }

    #[test]
    fn sequence_values_are_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<SeqTensor>();
        assert_send_sync::<SequenceValue>();
    }
}
