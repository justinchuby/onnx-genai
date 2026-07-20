//! ONNX sequence values and zero-copy split views.
//!
//! Sequence elements reuse the session crate's [`Tensor`] representation. A
//! [`SeqTensor`] is an immutable view over an `Arc`-owned device allocation, so
//! constructing, inserting, erasing, indexing, and splitting a sequence only
//! clone handles and metadata. Tensor storage is never deep-copied by those
//! operations.

use std::sync::Arc;

use onnx_runtime_ir::{DataType, TensorLayout, compute_contiguous_strides};

use crate::tensor::{SharedTensorBuffer, host_bytes};
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

/// An immutable tensor view used as one sequence element.
///
/// Cloning this type only bumps the backing allocation's `Arc` count. Shape,
/// strides, and byte offset are metadata, so a split slice can share the source
/// allocation even when it is not contiguous.
#[derive(Clone, Debug)]
pub struct SeqTensor {
    storage: Arc<SharedTensorBuffer>,
    pub dtype: DataType,
    pub shape: Vec<usize>,
    pub layout: TensorLayout,
    byte_offset: usize,
}

impl SeqTensor {
    /// Wrap an existing session tensor in an immutable shared handle.
    pub fn new(tensor: Tensor) -> Self {
        let (storage, dtype, shape, layout) = tensor.into_shared_parts();
        Self {
            storage,
            dtype,
            shape,
            layout,
            byte_offset: 0,
        }
    }

    /// Build a host tensor from raw element bytes and share it as a sequence item.
    pub fn from_raw(dtype: DataType, shape: Vec<usize>, bytes: &[u8]) -> SequenceResult<Self> {
        validate_tensor_bytes("SequenceTensor", bytes, dtype, &shape)?;
        let mut storage = SharedTensorBuffer::allocate_cpu(bytes.len()).map_err(|source| {
            SequenceError::TensorCreation {
                op: "SequenceTensor",
                source,
            }
        })?;
        let allocator = Arc::clone(storage.allocator());
        allocator
            .copy_from_host(
                bytes,
                Arc::get_mut(&mut storage)
                    .expect("fresh sequence storage is uniquely owned")
                    .buffer_mut(),
            )
            .map_err(|source| SequenceError::TensorCreation {
                op: "SequenceTensor",
                source: source.into(),
            })?;
        Ok(Self {
            storage,
            dtype,
            shape,
            layout: TensorLayout::contiguous(),
            byte_offset: 0,
        })
    }

    pub(crate) fn from_shared(
        storage: Arc<SharedTensorBuffer>,
        dtype: DataType,
        shape: Vec<usize>,
        layout: TensorLayout,
        byte_offset: usize,
    ) -> SequenceResult<Self> {
        let strides = layout.resolved_strides(&shape);
        validate_view_bounds(
            "SequenceTensor",
            &shape,
            &strides,
            byte_offset,
            dtype,
            storage.buffer().len(),
        )?;
        Ok(Self {
            storage,
            dtype,
            shape,
            layout,
            byte_offset,
        })
    }

    /// Whether two handles share the same underlying device allocation.
    pub fn shares_storage_with(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.storage, &other.storage)
    }

    /// Number of live handles to the shared allocation.
    pub fn storage_strong_count(&self) -> usize {
        Arc::strong_count(&self.storage)
    }

    pub(crate) fn storage(&self) -> &Arc<SharedTensorBuffer> {
        &self.storage
    }

    /// Base address of the shared tensor allocation.
    pub fn as_ptr(&self) -> *const std::ffi::c_void {
        self.storage.buffer().as_ptr()
    }

    /// Byte offset of this view's logical origin from [`Self::as_ptr`].
    pub fn byte_offset(&self) -> usize {
        self.byte_offset
    }

    pub fn device(&self) -> onnx_runtime_ir::DeviceId {
        self.storage.buffer().device()
    }

    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    pub(crate) fn root_len(&self) -> usize {
        self.storage.buffer().len()
    }

    pub(crate) fn contiguous_bytes(&self) -> SequenceResult<Vec<u8>> {
        const OP: &str = "SequenceTensor";
        let esize = self.dtype.byte_size();
        if esize == 0 {
            return Err(SequenceError::UnsupportedDtype {
                op: OP,
                dtype: self.dtype,
            });
        }
        let mut copied;
        let root = if self.device().is_host_accessible() {
            host_bytes(self.storage.buffer())
        } else {
            copied = zeroed_bytes(OP, "device tensor download", self.root_len(), &self.shape)?;
            self.storage
                .allocator()
                .copy_to_host(self.storage.buffer(), &mut copied)
                .map_err(|source| SequenceError::TensorCreation {
                    op: OP,
                    source: source.into(),
                })?;
            &copied
        };
        let strides = self.layout.resolved_strides(&self.shape);
        if onnx_runtime_ir::is_contiguous(&self.shape, &strides) {
            let bytes = self
                .dtype
                .checked_storage_bytes(self.numel())
                .ok_or_else(|| overflow(OP, "tensor byte count", &self.shape))?;
            let end = checked_add(
                OP,
                "tensor byte range",
                self.byte_offset,
                bytes,
                &self.shape,
            )?;
            return Ok(root[self.byte_offset..end].to_vec());
        }
        gather_strided(
            root,
            &self.shape,
            &strides,
            self.byte_offset,
            self.dtype,
            esize,
        )
    }

    /// Borrow bytes directly when this is a contiguous host view.
    pub fn as_bytes(&self) -> &[u8] {
        assert!(
            self.device().is_host_accessible() && self.layout.is_contiguous(&self.shape),
            "SeqTensor::as_bytes requires a contiguous host tensor"
        );
        let bytes = self.dtype.storage_bytes(self.numel());
        &host_bytes(self.storage.buffer())[self.byte_offset..self.byte_offset + bytes]
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

    /// Declared homogeneous element dtype.
    pub fn elem_dtype(&self) -> DataType {
        self.elem_dtype
    }

    /// Ordered shared tensor handles.
    pub fn elements(&self) -> &[SeqTensor] {
        &self.items
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
    let input = SeqTensor::from_raw(dtype, shape.to_vec(), data)?;
    split_tensor(&input, axis, split, keepdims)
}

/// Split a shared tensor into metadata-only views over the same allocation.
///
/// No tensor bytes are copied. Every returned element owns an `Arc` clone of
/// `input`'s storage and records its own shape, strides, and byte offset.
pub fn split_tensor(
    input: &SeqTensor,
    axis: i64,
    split: SplitSpec<'_>,
    keepdims: bool,
) -> SequenceResult<SequenceValue> {
    const OP: &str = "SplitToSequence";
    let rank = input.shape.len();
    if rank == 0 {
        return Err(SequenceError::InvalidAxis {
            op: OP,
            axis,
            rank,
            new_axis: false,
        });
    }
    let axis = normalize_axis(OP, axis, rank, false)?;
    let axis_dim = input.shape[axis];
    let (sizes, squeeze) = split_sizes(OP, axis, axis_dim, split, keepdims, &input.shape)?;
    let input_strides = input.layout.resolved_strides(&input.shape);
    let esize = input.dtype.byte_size();
    if esize == 0 {
        return Err(SequenceError::UnsupportedDtype {
            op: OP,
            dtype: input.dtype,
        });
    }

    let mut items = Vec::new();
    items
        .try_reserve_exact(sizes.len())
        .map_err(|_| SequenceError::Allocation {
            op: OP,
            context: "sequence handles",
            bytes: sizes.len().saturating_mul(std::mem::size_of::<SeqTensor>()),
        })?;
    let mut start = 0usize;
    for size in sizes {
        let delta_elements = (start as i128)
            .checked_mul(input_strides[axis] as i128)
            .ok_or_else(|| overflow(OP, "split view element offset", &input.shape))?;
        let delta_bytes = delta_elements
            .checked_mul(esize as i128)
            .ok_or_else(|| overflow(OP, "split view byte offset", &input.shape))?;
        let byte_offset = (input.byte_offset as i128)
            .checked_add(delta_bytes)
            .and_then(|offset| usize::try_from(offset).ok())
            .ok_or_else(|| overflow(OP, "split view byte offset", &input.shape))?;

        let mut shape = clone_shape(OP, &input.shape)?;
        shape[axis] = size;
        let mut strides = input_strides.clone();
        if squeeze {
            shape.remove(axis);
            strides.remove(axis);
        }
        items.push(SeqTensor::from_shared(
            Arc::clone(input.storage()),
            input.dtype,
            shape,
            TensorLayout::strided(strides),
            byte_offset,
        )?);
        start = checked_add(OP, "split axis cursor", start, size, &input.shape)?;
    }
    Ok(SequenceValue {
        elem_dtype: input.dtype,
        items,
    })
}

fn split_sizes(
    op: &'static str,
    axis: usize,
    axis_dim: usize,
    split: SplitSpec<'_>,
    keepdims: bool,
    shape: &[usize],
) -> SequenceResult<(Vec<usize>, bool)> {
    match split {
        SplitSpec::Each => {
            let mut sizes = Vec::new();
            sizes
                .try_reserve_exact(axis_dim)
                .map_err(|_| SequenceError::Allocation {
                    op,
                    context: "split sizes",
                    bytes: axis_dim.saturating_mul(std::mem::size_of::<usize>()),
                })?;
            sizes.resize(axis_dim, 1);
            Ok((sizes, !keepdims))
        }
        SplitSpec::Chunk(chunk) => {
            if chunk <= 0 {
                return Err(SequenceError::InvalidSplit {
                    op,
                    reason: format!("scalar chunk size {chunk} must be positive"),
                });
            }
            let chunk = usize::try_from(chunk).map_err(|_| SequenceError::InvalidSplit {
                op,
                reason: format!("scalar chunk size {chunk} cannot be represented"),
            })?;
            let count = axis_dim
                .checked_add(chunk - 1)
                .ok_or_else(|| overflow(op, "split chunk count", shape))?
                / chunk;
            let mut sizes = Vec::new();
            sizes
                .try_reserve_exact(count)
                .map_err(|_| SequenceError::Allocation {
                    op,
                    context: "split sizes",
                    bytes: count.saturating_mul(std::mem::size_of::<usize>()),
                })?;
            let mut remaining = axis_dim;
            while remaining != 0 {
                let size = remaining.min(chunk);
                sizes.push(size);
                remaining -= size;
            }
            Ok((sizes, false))
        }
        SplitSpec::Sizes(values) => {
            let mut sizes = Vec::new();
            sizes
                .try_reserve_exact(values.len())
                .map_err(|_| SequenceError::Allocation {
                    op,
                    context: "split sizes",
                    bytes: values.len().saturating_mul(std::mem::size_of::<usize>()),
                })?;
            let mut sum = 0usize;
            for &value in values {
                let value = usize::try_from(value).map_err(|_| SequenceError::InvalidSplit {
                    op,
                    reason: format!("size {value} must be non-negative"),
                })?;
                sum = sum
                    .checked_add(value)
                    .ok_or_else(|| overflow(op, "split size sum", shape))?;
                sizes.push(value);
            }
            if sum != axis_dim {
                return Err(SequenceError::InvalidSplit {
                    op,
                    reason: format!("sizes sum to {sum}, but axis {axis} has extent {axis_dim}"),
                });
            }
            Ok((sizes, false))
        }
    }
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
    let output_rank = rank
        .checked_add(usize::from(new_axis))
        .ok_or_else(|| overflow(OP, "concat output rank", &first.shape))?;
    let axis = normalize_axis(OP, axis, output_rank, new_axis)?;

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
    for item in &sequence.items {
        let bytes = item.contiguous_bytes()?;
        validate_tensor_bytes(OP, &bytes, item.dtype, &item.shape)?;
        shapes.push(clone_shape(OP, &item.shape)?);
        elements.push(bytes);
    }
    let element_refs = elements.iter().map(Vec::as_slice).collect::<Vec<_>>();

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
        stack_new_axis(&element_refs, &shapes[0], axis, esize)?
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
        concat_axis(&element_refs, &shapes, axis, esize)?
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

fn validate_view_bounds(
    op: &'static str,
    shape: &[usize],
    strides: &[i64],
    byte_offset: usize,
    dtype: DataType,
    root_len: usize,
) -> SequenceResult<()> {
    if shape.len() != strides.len() {
        return Err(SequenceError::InvalidSplit {
            op,
            reason: format!(
                "view rank mismatch: shape has {} dims but strides has {}",
                shape.len(),
                strides.len()
            ),
        });
    }
    let esize = dtype.byte_size();
    if esize == 0 {
        return Err(SequenceError::UnsupportedDtype { op, dtype });
    }
    if shape.contains(&0) {
        return Ok(());
    }
    let mut min_element = 0i128;
    let mut max_element = 0i128;
    for (&dim, &stride) in shape.iter().zip(strides) {
        let span = (dim.saturating_sub(1) as i128)
            .checked_mul(stride as i128)
            .ok_or_else(|| overflow(op, "view stride span", shape))?;
        if span < 0 {
            min_element = min_element
                .checked_add(span)
                .ok_or_else(|| overflow(op, "view minimum offset", shape))?;
        } else {
            max_element = max_element
                .checked_add(span)
                .ok_or_else(|| overflow(op, "view maximum offset", shape))?;
        }
    }
    let origin = byte_offset as i128;
    let min_byte = origin
        .checked_add(
            min_element
                .checked_mul(esize as i128)
                .ok_or_else(|| overflow(op, "view minimum byte offset", shape))?,
        )
        .ok_or_else(|| overflow(op, "view minimum byte offset", shape))?;
    let end_byte = origin
        .checked_add(
            max_element
                .checked_mul(esize as i128)
                .ok_or_else(|| overflow(op, "view maximum byte offset", shape))?,
        )
        .and_then(|offset| offset.checked_add(esize as i128))
        .ok_or_else(|| overflow(op, "view byte range", shape))?;
    if min_byte < 0 || end_byte > root_len as i128 {
        return Err(SequenceError::InvalidSplit {
            op,
            reason: format!(
                "view byte range [{min_byte}, {end_byte}) exceeds backing allocation of {root_len} bytes"
            ),
        });
    }
    Ok(())
}

fn gather_strided(
    root: &[u8],
    shape: &[usize],
    strides: &[i64],
    byte_offset: usize,
    dtype: DataType,
    esize: usize,
) -> SequenceResult<Vec<u8>> {
    const OP: &str = "SequenceTensor";
    validate_view_bounds(OP, shape, strides, byte_offset, dtype, root.len())?;
    let numel = checked_product(OP, "view element count", shape)?;
    let bytes = checked_mul(OP, "view byte count", numel, esize, shape)?;
    let mut output = zeroed_bytes(OP, "strided tensor materialization", bytes, shape)?;
    if numel == 0 {
        return Ok(output);
    }
    let logical_strides = compute_contiguous_strides(shape);
    for linear in 0..numel {
        let mut remainder = linear;
        let mut source_element = 0i128;
        for dimension in 0..shape.len() {
            let coordinate = if shape[dimension] == 0 {
                0
            } else {
                remainder / logical_strides[dimension] as usize
            };
            if shape[dimension] != 0 {
                remainder %= logical_strides[dimension] as usize;
            }
            source_element += coordinate as i128 * strides[dimension] as i128;
        }
        let source = (byte_offset as i128 + source_element * esize as i128) as usize;
        output[linear * esize..(linear + 1) * esize].copy_from_slice(&root[source..source + esize]);
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
        assert!(original.shares_storage_with(&sequence.elements()[0]));

        let inserted = elem(DataType::Uint8, &[1], &[9]);
        let sequence = sequence.insert(inserted.clone(), Some(-1)).expect("insert");
        assert!(inserted.shares_storage_with(&sequence.at(0).expect("at")));
        assert!(original.shares_storage_with(&sequence.at(1).expect("at")));

        let erased = sequence.erase(Some(0)).expect("erase");
        assert!(original.shares_storage_with(&erased.at(-1).expect("negative at")));
    }

    #[test]
    fn moving_tensor_into_sequence_preserves_allocation_pointer() {
        let tensor = Tensor::from_raw(DataType::Uint8, vec![2], &[4, 5]).unwrap();
        let pointer = tensor.device_ptr();
        let element = SeqTensor::new(tensor);
        assert_eq!(element.as_ptr(), pointer);
        let sequence = SequenceValue::construct(vec![element.clone()]).unwrap();
        assert_eq!(sequence.at(0).unwrap().as_ptr(), pointer);
        assert_eq!(element.storage_strong_count(), 2);
    }

    #[test]
    fn split_produces_shared_strided_views_without_copying() {
        let input = elem(DataType::Uint8, &[2, 3], &[0, 1, 2, 3, 4, 5]);
        let sequence = split_tensor(&input, 1, SplitSpec::Sizes(&[1, 2]), true).expect("split");
        assert_eq!(sequence.length(), 2);
        assert!(input.shares_storage_with(&sequence.elements()[0]));
        assert!(input.shares_storage_with(&sequence.elements()[1]));
        assert_eq!(sequence.elements()[0].byte_offset(), 0);
        assert_eq!(sequence.elements()[1].byte_offset(), 1);
        assert_eq!(
            sequence.elements()[0].contiguous_bytes().unwrap(),
            vec![0, 3]
        );
        assert_eq!(
            sequence.elements()[1].contiguous_bytes().unwrap(),
            vec![1, 2, 4, 5]
        );
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
    fn empty_sequence_edges_are_clean_errors() {
        let empty = SequenceValue::empty(DataType::Uint8);
        assert!(matches!(empty.erase(None), Err(SequenceError::EmptyErase)));
        assert!(matches!(
            empty.at(0),
            Err(SequenceError::IndexOutOfBounds { len: 0, .. })
        ));
        assert!(matches!(
            concat(&empty, 0, false),
            Err(SequenceError::InvalidSplit { .. })
        ));
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
            let input = elem(DataType::Uint8, &shape, &data);
            let sequence = split_tensor(&input, axis, SplitSpec::Each, keepdims).unwrap();
            let concat_axis = if keepdims { axis } else { 0 };
            let rebuilt = concat(&sequence, concat_axis, !keepdims).unwrap();
            assert_eq!(rebuilt.shape, shape);
            assert_eq!(rebuilt.as_bytes(), data);
        }
    }

    #[test]
    fn split_concat_roundtrip_explicit_sizes() {
        let data: Vec<u8> = (0..12).collect();
        let input = elem(DataType::Uint8, &[3, 4], &data);
        let sequence = split_tensor(&input, 1, SplitSpec::Sizes(&[1, 3]), false).unwrap();
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
        let error = validate_tensor_bytes(
            "SplitToSequence",
            &[],
            DataType::Uint8,
            &[isize::MAX as usize + 1, 1],
        )
        .unwrap_err();
        assert!(matches!(error, SequenceError::ShapeOverflow { .. }));
    }

    #[test]
    fn sequence_values_are_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<SeqTensor>();
        assert_send_sync::<SequenceValue>();
    }
}
