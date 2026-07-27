//! Constant tensor storage, weight references, and ONNX type descriptors.

use std::path::PathBuf;

use crate::dtype::DataType;
use crate::shape::Shape;

/// A concrete constant tensor held inline (e.g. an attribute value or a small
/// initializer). Element bytes are stored little-endian and densely packed.
///
/// Large model weights are referenced lazily via [`WeightRef`] instead.
#[derive(Clone, Debug, PartialEq)]
pub struct TensorData {
    pub name: Option<String>,
    pub dtype: DataType,
    /// Static dimensions (constants always have a fully known shape).
    pub dims: Vec<usize>,
    /// Raw little-endian element bytes. Sub-byte values are densely packed
    /// (two 4-bit or four 2-bit elements per byte); for [`DataType::String`]
    /// this is empty and `strings` is used instead.
    pub data: Vec<u8>,
    /// String payloads for [`DataType::String`] tensors.
    pub strings: Vec<String>,
}

impl TensorData {
    /// A numeric tensor from raw little-endian bytes.
    pub fn from_raw(dtype: DataType, dims: Vec<usize>, data: Vec<u8>) -> Self {
        Self {
            name: None,
            dtype,
            dims,
            data,
            strings: Vec::new(),
        }
    }

    /// Number of elements (product of dims; `1` for a scalar).
    pub fn numel(&self) -> usize {
        self.checked_numel().expect("tensor element count overflow")
    }

    /// Number of elements, or `None` when the dimensions overflow `usize`.
    pub fn checked_numel(&self) -> Option<usize> {
        checked_numel(&self.dims)
    }

    /// Expected byte length for `numel` elements of `dtype`, accounting for
    /// sub-byte packing.
    pub fn expected_bytes(&self) -> usize {
        self.checked_expected_bytes()
            .expect("tensor byte count overflow")
    }

    /// Expected byte length, or `None` when the element or byte count overflows.
    pub fn checked_expected_bytes(&self) -> Option<usize> {
        checked_expected_bytes(self.dtype, &self.dims)
    }
}

/// Number of elements in `dims`, or `None` when their product overflows.
pub fn checked_numel(dims: &[usize]) -> Option<usize> {
    dims.iter()
        .try_fold(1usize, |product, &dimension| product.checked_mul(dimension))
}

/// Dense storage size for `dtype` and `dims`, or `None` on geometry overflow.
pub fn checked_expected_bytes(dtype: DataType, dims: &[usize]) -> Option<usize> {
    let element_count = checked_numel(dims)?;
    if dtype == DataType::Undefined {
        return None;
    }
    if dtype.is_sub_byte() {
        let elements_per_byte = 8 / dtype.bit_size();
        return (element_count / elements_per_byte).checked_add(usize::from(
            !element_count.is_multiple_of(elements_per_byte),
        ));
    }
    element_count.checked_mul(dtype.byte_size())
}

/// A sparse constant tensor in COO form.
#[derive(Clone, Debug, PartialEq)]
pub struct SparseTensorData {
    /// Non-zero values.
    pub values: TensorData,
    /// Indices of the non-zero values (int64), shape `[nnz, rank]` or `[nnz]`.
    pub indices: TensorData,
    /// Dense shape.
    pub dims: Vec<usize>,
}

/// An ONNX `TypeProto`: the type of a value, which may be a tensor or a
/// container of tensors (see `docs/ORT2.md` §3.2).
#[derive(Clone, Debug, PartialEq)]
pub enum TypeProto {
    Tensor {
        dtype: DataType,
        shape: Shape,
    },
    Sequence(Box<TypeProto>),
    Optional(Box<TypeProto>),
    Map {
        key: DataType,
        value: Box<TypeProto>,
    },
    SparseTensor {
        dtype: DataType,
        shape: Shape,
    },
}

/// A reference to initializer (weight) data.
///
/// Small weights may be inlined; large weights are memory-mapped from an
/// external file at load time (see `docs/ORT2.md` §12). The IR only stores the
/// *reference*; the loader/`onnx-runtime-memory` crate performs the mmap.
#[derive(Clone, Debug, PartialEq)]
pub enum WeightRef {
    /// Weight bytes stored inline in the model.
    Inline(TensorData),
    /// Weight bytes located in an external file at `[offset, offset+length)`.
    External {
        path: PathBuf,
        offset: usize,
        length: usize,
        dtype: DataType,
        dims: Vec<usize>,
    },
}

impl WeightRef {
    /// The element type of the referenced weight.
    pub fn dtype(&self) -> DataType {
        match self {
            WeightRef::Inline(t) => t.dtype,
            WeightRef::External { dtype, .. } => *dtype,
        }
    }

    /// The static dimensions of the referenced weight.
    pub fn dims(&self) -> &[usize] {
        match self {
            WeightRef::Inline(t) => &t.dims,
            WeightRef::External { dims, .. } => dims,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tensor_numel_and_bytes() {
        let t = TensorData::from_raw(DataType::Float32, vec![2, 3], vec![0u8; 24]);
        assert_eq!(t.numel(), 6);
        assert_eq!(t.expected_bytes(), 24);
    }

    #[test]
    fn sub_byte_expected_bytes() {
        let t = TensorData::from_raw(DataType::Int4, vec![3], vec![0u8; 2]);
        assert_eq!(t.numel(), 3);
        assert_eq!(t.expected_bytes(), 2); // 3 packed nibbles -> 2 bytes
    }

    #[test]
    fn checked_geometry_rejects_overflow() {
        let tensor = TensorData::from_raw(DataType::Float32, vec![usize::MAX, 2], Vec::new());
        assert_eq!(tensor.checked_numel(), None);
        assert_eq!(tensor.checked_expected_bytes(), None);

        let byte_overflow =
            TensorData::from_raw(DataType::Float64, vec![usize::MAX / 4], Vec::new());
        assert_eq!(byte_overflow.checked_numel(), Some(usize::MAX / 4));
        assert_eq!(byte_overflow.checked_expected_bytes(), None);
    }

    #[test]
    fn weight_ref_accessors() {
        let w = WeightRef::External {
            path: PathBuf::from("weights.bin"),
            offset: 128,
            length: 4096,
            dtype: DataType::Float16,
            dims: vec![64, 32],
        };
        assert_eq!(w.dtype(), DataType::Float16);
        assert_eq!(w.dims(), &[64, 32]);
    }
}
