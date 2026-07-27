use onnx_runtime_ir::DataType;

use crate::Tensor;

use super::{
    SeqTensor, SequenceError, SequenceResult, SequenceValue, addressable, checked_add, checked_mul,
    checked_product, clone_shape, normalize_axis, overflow, validate_view_bounds, zeroed_bytes,
};

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct ConcatCopyStats {
    pub destination_writes: usize,
    pub source_materializations: usize,
}

/// Fully validated geometry for `ConcatFromSequence`.
pub(crate) struct ConcatPlan {
    pub dtype: DataType,
    pub shape: Vec<usize>,
    pub bytes: usize,
    axis: usize,
    new_axis: bool,
    outer: usize,
    inner: usize,
    total_axis: usize,
}

impl ConcatPlan {
    pub(crate) fn new(sequence: &SequenceValue, axis: i64, new_axis: bool) -> SequenceResult<Self> {
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

        for (index, item) in sequence.items.iter().enumerate() {
            if item.dtype != dtype {
                return Err(SequenceError::DtypeMismatch {
                    op: OP,
                    index: Some(index),
                    expected: dtype,
                    actual: item.dtype,
                });
            }
            let mismatch = if new_axis {
                item.shape != first.shape
            } else {
                item.shape.len() != rank
                    || item.shape.iter().enumerate().any(|(dimension, &extent)| {
                        dimension != axis && extent != first.shape[dimension]
                    })
            };
            if mismatch {
                return Err(SequenceError::ShapeMismatch {
                    op: OP,
                    index,
                    expected: clone_shape(OP, &first.shape)?,
                    actual: clone_shape(OP, &item.shape)?,
                    requirement: if new_axis {
                        "new_axis=1 requires identical shapes"
                    } else {
                        "all dimensions except the concat axis must match"
                    },
                });
            }
            validate_view_bounds(
                OP,
                &item.shape,
                &item.layout.resolved_strides(&item.shape),
                item.byte_offset,
                item.dtype,
                item.root_len(),
            )?;
        }

        if new_axis {
            let outer = checked_product(OP, "stack outer element count", &first.shape[..axis])?;
            let inner_elements =
                checked_product(OP, "stack inner element count", &first.shape[axis..])?;
            let inner = checked_mul(
                OP,
                "stack inner byte count",
                inner_elements,
                esize,
                &first.shape,
            )?;
            let mut shape = Vec::new();
            shape
                .try_reserve_exact(output_rank)
                .map_err(|_| SequenceError::Allocation {
                    op: OP,
                    context: "stack output shape",
                    bytes: output_rank.saturating_mul(std::mem::size_of::<usize>()),
                })?;
            shape.extend_from_slice(&first.shape[..axis]);
            shape.push(sequence.items.len());
            shape.extend_from_slice(&first.shape[axis..]);
            let bytes = checked_mul(
                OP,
                "stack output byte count",
                checked_mul(
                    OP,
                    "stack output row count",
                    outer,
                    sequence.items.len(),
                    &shape,
                )?,
                inner,
                &shape,
            )?;
            Ok(Self {
                dtype,
                shape,
                bytes,
                axis,
                new_axis,
                outer,
                inner,
                total_axis: sequence.items.len(),
            })
        } else {
            let outer = checked_product(OP, "concat outer element count", &first.shape[..axis])?;
            let inner_elements =
                checked_product(OP, "concat inner element count", &first.shape[axis + 1..])?;
            let inner = checked_mul(
                OP,
                "concat inner byte count",
                inner_elements,
                esize,
                &first.shape,
            )?;
            let mut total_axis = 0usize;
            for item in &sequence.items {
                total_axis = checked_add(
                    OP,
                    "concat axis extent",
                    total_axis,
                    item.shape[axis],
                    &first.shape,
                )?;
            }
            let mut shape = clone_shape(OP, &first.shape)?;
            shape[axis] = total_axis;
            let bytes = checked_mul(
                OP,
                "concat output byte count",
                checked_mul(OP, "concat output row count", outer, total_axis, &shape)?,
                inner,
                &shape,
            )?;
            Ok(Self {
                dtype,
                shape,
                bytes,
                axis,
                new_axis,
                outer,
                inner,
                total_axis,
            })
        }
    }

    pub(crate) fn write<F>(
        &self,
        sequence: &SequenceValue,
        mut write: F,
    ) -> crate::Result<ConcatCopyStats>
    where
        F: FnMut(usize, &[u8]) -> crate::Result<()>,
    {
        const OP: &str = "ConcatFromSequence";
        let max_root = sequence
            .items
            .iter()
            .filter(|item| !item.device().is_host_accessible())
            .map(SeqTensor::root_len)
            .max()
            .unwrap_or(0);
        let mut scratch = if max_root == 0 {
            Vec::new()
        } else {
            zeroed_bytes(OP, "device source materialization", max_root, &self.shape)?
        };
        let mut stats = ConcatCopyStats::default();
        if self.new_axis {
            for outer_index in 0..self.outer {
                for (item_index, item) in sequence.items.iter().enumerate() {
                    let source_offset = checked_mul(
                        OP,
                        "stack source offset",
                        outer_index,
                        self.inner,
                        &item.shape,
                    )?;
                    let destination_row = checked_add(
                        OP,
                        "stack destination row",
                        checked_mul(
                            OP,
                            "stack destination outer offset",
                            outer_index,
                            self.total_axis,
                            &self.shape,
                        )?,
                        item_index,
                        &self.shape,
                    )?;
                    let destination_offset = checked_mul(
                        OP,
                        "stack destination byte offset",
                        destination_row,
                        self.inner,
                        &self.shape,
                    )?;
                    item.write_contiguous_range(
                        source_offset,
                        self.inner,
                        destination_offset,
                        &mut scratch,
                        &mut write,
                        &mut stats,
                    )?;
                }
            }
        } else {
            for outer_index in 0..self.outer {
                let mut axis_cursor = 0usize;
                for item in &sequence.items {
                    let copy_bytes = checked_mul(
                        OP,
                        "concat copy width",
                        item.shape[self.axis],
                        self.inner,
                        &item.shape,
                    )?;
                    let source_offset = checked_mul(
                        OP,
                        "concat source byte offset",
                        outer_index,
                        copy_bytes,
                        &item.shape,
                    )?;
                    let destination_row = checked_add(
                        OP,
                        "concat destination row",
                        checked_mul(
                            OP,
                            "concat destination outer offset",
                            outer_index,
                            self.total_axis,
                            &self.shape,
                        )?,
                        axis_cursor,
                        &self.shape,
                    )?;
                    let destination_offset = checked_mul(
                        OP,
                        "concat destination byte offset",
                        destination_row,
                        self.inner,
                        &self.shape,
                    )?;
                    item.write_contiguous_range(
                        source_offset,
                        copy_bytes,
                        destination_offset,
                        &mut scratch,
                        &mut write,
                        &mut stats,
                    )?;
                    axis_cursor = checked_add(
                        OP,
                        "concat axis cursor",
                        axis_cursor,
                        item.shape[self.axis],
                        &self.shape,
                    )?;
                }
            }
        }
        Ok(stats)
    }
}

/// Concatenate a sequence along an existing axis or stack it on a new axis.
pub fn concat(sequence: &SequenceValue, axis: i64, new_axis: bool) -> SequenceResult<SeqTensor> {
    const OP: &str = "ConcatFromSequence";
    let plan = ConcatPlan::new(sequence, axis, new_axis)?;
    let mut tensor = Tensor::allocate_cpu(plan.dtype, plan.shape.clone())
        .map_err(|source| SequenceError::TensorCreation { op: OP, source })?;
    plan.write(sequence, |offset, bytes| {
        tensor.copy_from_host_at(offset, bytes)
    })
    .map_err(|source| SequenceError::TensorCreation { op: OP, source })?;
    Ok(SeqTensor::new(tensor))
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
