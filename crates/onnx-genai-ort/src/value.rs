//! ORT Values (tensors).

use crate::Result;

/// Tensor data types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataType {
    Float32,
    Float16,
    BFloat16,
    Float8E4M3,
    Float8E5M2,
    Int8,
    Int16,
    Int32,
    Int64,
    Uint8,
    Uint16,
    Uint32,
    Uint64,
    Bool,
}

impl DataType {
    /// Size in bytes of one element.
    pub fn size_of(&self) -> usize {
        match self {
            DataType::Float32 | DataType::Int32 | DataType::Uint32 => 4,
            DataType::Float16 | DataType::BFloat16 | DataType::Int16 | DataType::Uint16 => 2,
            DataType::Float8E4M3 | DataType::Float8E5M2 | DataType::Int8 | DataType::Uint8 | DataType::Bool => 1,
            DataType::Int64 | DataType::Uint64 => 8,
        }
    }
}

/// An ORT tensor value.
pub struct Value {
    // ptr: *mut ort_sys::OrtValue,
    shape: Vec<i64>,
    dtype: DataType,
}

impl Value {
    /// Create a tensor value with given shape and type.
    /// Memory is allocated on the device specified by MemoryInfo (via allocator).
    pub fn empty(shape: &[i64], dtype: DataType) -> Result<Self> {
        // TODO: Call OrtCreateTensorAsOrtValue via C API
        Ok(Self {
            shape: shape.to_vec(),
            dtype,
        })
    }

    /// Create a tensor from a slice (CPU, zero-copy if possible).
    pub fn from_slice_f32(data: &[f32], shape: &[i64]) -> Result<Self> {
        let expected_len: i64 = shape.iter().product();
        if data.len() != expected_len as usize {
            return Err(crate::OrtError::InvalidArgument(format!(
                "Data length {} doesn't match shape {:?} (expected {})",
                data.len(), shape, expected_len
            )));
        }
        // TODO: Call OrtCreateTensorWithDataAsOrtValue
        Ok(Self {
            shape: shape.to_vec(),
            dtype: DataType::Float32,
        })
    }

    /// Create a tensor from i64 data (for input_ids, attention_mask).
    pub fn from_slice_i64(data: &[i64], shape: &[i64]) -> Result<Self> {
        let expected_len: i64 = shape.iter().product();
        if data.len() != expected_len as usize {
            return Err(crate::OrtError::InvalidArgument(format!(
                "Data length {} doesn't match shape {:?} (expected {})",
                data.len(), shape, expected_len
            )));
        }
        Ok(Self {
            shape: shape.to_vec(),
            dtype: DataType::Int64,
        })
    }

    /// Get tensor shape.
    pub fn shape(&self) -> &[i64] {
        &self.shape
    }

    /// Get tensor data type.
    pub fn dtype(&self) -> DataType {
        self.dtype
    }

    /// Total number of elements.
    pub fn numel(&self) -> usize {
        self.shape.iter().product::<i64>() as usize
    }
}

impl Drop for Value {
    fn drop(&mut self) {
        // TODO: Call OrtReleaseValue
    }
}
