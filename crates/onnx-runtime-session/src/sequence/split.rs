use std::sync::Arc;

use onnx_runtime_ir::{DataType, TensorLayout};

use super::{
    SeqTensor, SequenceError, SequenceResult, SequenceValue, checked_add, clone_shape,
    normalize_axis, overflow,
};

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
