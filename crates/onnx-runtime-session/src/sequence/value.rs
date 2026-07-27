use onnx_runtime_ir::DataType;

use super::{SeqTensor, SequenceError, SequenceResult};

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
