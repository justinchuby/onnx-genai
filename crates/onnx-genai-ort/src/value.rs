//! ORT Values (tensors).

use std::ptr::NonNull;
use std::sync::Arc;

use crate::{MemoryInfo, OrtError, Result};

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
            DataType::Float8E4M3
            | DataType::Float8E5M2
            | DataType::Int8
            | DataType::Uint8
            | DataType::Bool => 1,
            DataType::Int64 | DataType::Uint64 => 8,
        }
    }

    pub(crate) fn to_onnx(self) -> onnx_genai_ort_sys::ONNXTensorElementDataType {
        match self {
            DataType::Float32 => onnx_genai_ort_sys::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            DataType::Float16 => onnx_genai_ort_sys::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16,
            DataType::BFloat16 => onnx_genai_ort_sys::ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16,
            DataType::Float8E4M3 => onnx_genai_ort_sys::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN,
            DataType::Float8E5M2 => onnx_genai_ort_sys::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2,
            DataType::Int8 => onnx_genai_ort_sys::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8,
            DataType::Int16 => onnx_genai_ort_sys::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16,
            DataType::Int32 => onnx_genai_ort_sys::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,
            DataType::Int64 => onnx_genai_ort_sys::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
            DataType::Uint8 => onnx_genai_ort_sys::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8,
            DataType::Uint16 => onnx_genai_ort_sys::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16,
            DataType::Uint32 => onnx_genai_ort_sys::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32,
            DataType::Uint64 => onnx_genai_ort_sys::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64,
            DataType::Bool => onnx_genai_ort_sys::ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL,
        }
    }

    pub(crate) fn from_onnx(dtype: onnx_genai_ort_sys::ONNXTensorElementDataType) -> Result<Self> {
        match dtype {
            onnx_genai_ort_sys::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT => Ok(DataType::Float32),
            onnx_genai_ort_sys::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 => Ok(DataType::Float16),
            onnx_genai_ort_sys::ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16 => Ok(DataType::BFloat16),
            onnx_genai_ort_sys::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN => {
                Ok(DataType::Float8E4M3)
            }
            onnx_genai_ort_sys::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2 => {
                Ok(DataType::Float8E5M2)
            }
            onnx_genai_ort_sys::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8 => Ok(DataType::Int8),
            onnx_genai_ort_sys::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16 => Ok(DataType::Int16),
            onnx_genai_ort_sys::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 => Ok(DataType::Int32),
            onnx_genai_ort_sys::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 => Ok(DataType::Int64),
            onnx_genai_ort_sys::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8 => Ok(DataType::Uint8),
            onnx_genai_ort_sys::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16 => Ok(DataType::Uint16),
            onnx_genai_ort_sys::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32 => Ok(DataType::Uint32),
            onnx_genai_ort_sys::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64 => Ok(DataType::Uint64),
            onnx_genai_ort_sys::ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL => Ok(DataType::Bool),
            other => Err(OrtError::InvalidArgument(format!(
                "unsupported ONNX tensor element data type: {other}"
            ))),
        }
    }
}

enum TensorBacking {
    F32(Vec<f32>),
    F16(Vec<u16>),
    I64(Vec<i64>),
    Alias(Arc<Value>),
    None,
}

/// An ORT tensor value.
pub struct Value {
    ptr: NonNull<onnx_genai_ort_sys::OrtValue>,
    shape: Vec<i64>,
    dtype: DataType,
    backing: TensorBacking,
}

impl Value {
    /// Create a tensor value with given shape and type.
    ///
    /// Memory is allocated with the default CPU allocator. Use
    /// [`Value::empty_in`] to allocate on a specific (device) allocator.
    pub fn empty(shape: &[i64], dtype: DataType) -> Result<Self> {
        Self::empty_in(shape, dtype, &crate::Allocator::default_cpu()?)
    }

    /// Create an uninitialized tensor value on the memory owned by `allocator`.
    ///
    /// When `allocator` is a device allocator (e.g. CUDA or the WebGPU EP's
    /// `WebGPU_Buffer` allocator from [`crate::Allocator::for_session_device`]),
    /// the tensor is device-resident: binding it as both a `past_key_values.*`
    /// input and `present.*` output keeps the KV cache on-device across decode
    /// steps and eliminates the per-step host<->device copies that the default
    /// CPU allocator would incur under an accelerator EP. The contents are
    /// uninitialized; callers must ensure unwritten regions are masked out.
    pub fn empty_in(shape: &[i64], dtype: DataType, allocator: &crate::Allocator) -> Result<Self> {
        validate_shape(shape, None)?;
        let mut ptr = std::ptr::null_mut();
        let api = crate::error::api()?;
        let create = api
            .CreateTensorAsOrtValue
            .ok_or(OrtError::ApiUnavailable("CreateTensorAsOrtValue"))?;
        // SAFETY: `shape` points to `shape.len()` i64 dimensions, `ptr` is a
        // valid out-parameter, and `allocator` remains valid for the call.
        crate::error::check_status(unsafe {
            create(
                allocator.as_ptr(),
                shape.as_ptr(),
                shape.len(),
                dtype.to_onnx(),
                &mut ptr,
            )
        })?;
        Ok(Self {
            ptr: NonNull::new(ptr).ok_or(OrtError::NullPointer)?,
            shape: shape.to_vec(),
            dtype,
            backing: TensorBacking::None,
        })
    }

    /// Create a tensor from a slice (CPU, zero-copy if possible).
    pub fn from_slice_f32(data: &[f32], shape: &[i64]) -> Result<Self> {
        Self::from_vec_f32(data.to_vec(), shape)
    }

    /// Create a CPU Float16 tensor from IEEE-754 half-precision bit patterns.
    pub fn from_slice_f16_bits(data: &[u16], shape: &[i64]) -> Result<Self> {
        Self::from_vec_f16_bits(data.to_vec(), shape)
    }

    /// Create a CPU BFloat16 tensor from bfloat16 bit patterns.
    pub fn from_slice_bf16_bits(data: &[u16], shape: &[i64]) -> Result<Self> {
        Self::from_vec_bf16_bits(data.to_vec(), shape)
    }

    /// Create a CPU float tensor of `dtype` from f32 host data.
    ///
    /// Float32 binds directly; Float16 narrows each element via the IEEE-754
    /// single -> half conversion. Used to feed f32 host buffers (materialized KV,
    /// projected-state activations) into graphs whose float inputs are fp16,
    /// keeping the engine-facing data path f32 regardless of the graph dtype.
    pub fn from_f32_slice_as(data: &[f32], shape: &[i64], dtype: DataType) -> Result<Self> {
        match dtype {
            DataType::Float32 => Self::from_slice_f32(data, shape),
            DataType::Float16 => {
                let bits: Vec<u16> = data
                    .iter()
                    .map(|&x| half::f16::from_f32(x).to_bits())
                    .collect();
                Self::from_vec_f16_bits(bits, shape)
            }
            DataType::BFloat16 => {
                let bits: Vec<u16> = data
                    .iter()
                    .map(|&x| half::bf16::from_f32(x).to_bits())
                    .collect();
                Self::from_vec_bf16_bits(bits, shape)
            }
            other => Err(OrtError::InvalidArgument(format!(
                "cannot build a {other:?} tensor from f32 data"
            ))),
        }
    }

    /// Create a tensor from i64 data (for input_ids, attention_mask).
    pub fn from_slice_i64(data: &[i64], shape: &[i64]) -> Result<Self> {
        Self::from_vec_i64(data.to_vec(), shape)
    }

    /// Create a CPU tensor from owned f32 data.
    pub fn from_vec_f32(mut data: Vec<f32>, shape: &[i64]) -> Result<Self> {
        validate_shape(shape, Some(data.len()))?;
        let ptr = create_tensor_with_data(
            data.as_mut_ptr().cast(),
            data.len() * std::mem::size_of::<f32>(),
            shape,
            DataType::Float32,
        )?;
        Ok(Self {
            ptr,
            shape: shape.to_vec(),
            dtype: DataType::Float32,
            backing: TensorBacking::F32(data),
        })
    }

    /// Create a CPU Float16 tensor from owned IEEE-754 half-precision bit patterns.
    pub fn from_vec_f16_bits(mut data: Vec<u16>, shape: &[i64]) -> Result<Self> {
        validate_shape(shape, Some(data.len()))?;
        let ptr = create_tensor_with_data(
            data.as_mut_ptr().cast(),
            data.len() * std::mem::size_of::<u16>(),
            shape,
            DataType::Float16,
        )?;
        Ok(Self {
            ptr,
            shape: shape.to_vec(),
            dtype: DataType::Float16,
            backing: TensorBacking::F16(data),
        })
    }

    /// Create a CPU BFloat16 tensor from owned bfloat16 bit patterns.
    pub fn from_vec_bf16_bits(mut data: Vec<u16>, shape: &[i64]) -> Result<Self> {
        validate_shape(shape, Some(data.len()))?;
        let ptr = create_tensor_with_data(
            data.as_mut_ptr().cast(),
            data.len() * std::mem::size_of::<u16>(),
            shape,
            DataType::BFloat16,
        )?;
        Ok(Self {
            ptr,
            shape: shape.to_vec(),
            dtype: DataType::BFloat16,
            backing: TensorBacking::F16(data),
        })
    }

    /// Create a CPU tensor from owned i64 data.
    pub fn from_vec_i64(mut data: Vec<i64>, shape: &[i64]) -> Result<Self> {
        validate_shape(shape, Some(data.len()))?;
        let ptr = create_tensor_with_data(
            data.as_mut_ptr().cast(),
            data.len() * std::mem::size_of::<i64>(),
            shape,
            DataType::Int64,
        )?;
        Ok(Self {
            ptr,
            shape: shape.to_vec(),
            dtype: DataType::Int64,
            backing: TensorBacking::I64(data),
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

    /// Copy tensor data out as f32 values.
    pub fn to_vec_f32(&self) -> Result<Vec<f32>> {
        if self.dtype != DataType::Float32 {
            return Err(OrtError::InvalidArgument(format!(
                "requested f32 data from {:?} tensor",
                self.dtype
            )));
        }
        tensor_data_to_vec(self.ptr.as_ptr(), self.numel())
    }

    /// Copy tensor data out as f32 values, widening Float16 losslessly.
    ///
    /// Float32 tensors are copied directly; Float16 tensors are upcast via the
    /// IEEE-754 half → single conversion. Used by decode/logits/hidden-state
    /// readers that must consume fp16 GroupQueryAttention (GQA) outputs on the
    /// host without a separate device conversion pass.
    pub fn to_vec_f32_lossy(&self) -> Result<Vec<f32>> {
        match self.dtype {
            DataType::Float32 => self.to_vec_f32(),
            DataType::Float16 => {
                let numel = self.numel();
                let data = tensor_data_ptr(self.ptr.as_ptr())?;
                // SAFETY: an fp16 tensor holds `numel` contiguous u16 elements at
                // `data`, valid until this Value is released; we only read here.
                let bits = unsafe { std::slice::from_raw_parts(data.cast::<u16>(), numel) };
                // Reinterpret the raw bits as f16 and widen with half's vectorized
                // slice conversion (hardware F16C when available), which is far
                // faster than a per-element `from_bits().to_f32()` scalar loop on
                // the hot logits path (~152K elements per decode step).
                let halves: &[half::f16] = half::slice::HalfBitsSliceExt::reinterpret_cast(bits);
                Ok(half::slice::HalfFloatSliceExt::to_f32_vec(halves))
            }
            DataType::BFloat16 => Ok(self
                .to_vec_bf16_bits()?
                .into_iter()
                .map(|bits| half::bf16::from_bits(bits).to_f32())
                .collect()),
            other => Err(OrtError::InvalidArgument(format!(
                "cannot widen {other:?} tensor to f32"
            ))),
        }
    }

    /// Copy Float16 tensor data out as IEEE-754 half-precision bit patterns.
    pub fn to_vec_f16_bits(&self) -> Result<Vec<u16>> {
        if self.dtype != DataType::Float16 {
            return Err(OrtError::InvalidArgument(format!(
                "requested Float16 data from {:?} tensor",
                self.dtype
            )));
        }
        tensor_data_to_vec(self.ptr.as_ptr(), self.numel())
    }

    /// Copy BFloat16 tensor data out as bfloat16 bit patterns.
    pub fn to_vec_bf16_bits(&self) -> Result<Vec<u16>> {
        if self.dtype != DataType::BFloat16 {
            return Err(OrtError::InvalidArgument(format!(
                "requested BFloat16 data from {:?} tensor",
                self.dtype
            )));
        }
        tensor_data_to_vec(self.ptr.as_ptr(), self.numel())
    }

    /// Copy tensor data out as i64 values.
    pub fn to_vec_i64(&self) -> Result<Vec<i64>> {
        if self.dtype != DataType::Int64 {
            return Err(OrtError::InvalidArgument(format!(
                "requested i64 data from {:?} tensor",
                self.dtype
            )));
        }
        tensor_data_to_vec(self.ptr.as_ptr(), self.numel())
    }

    pub(crate) fn as_ptr(&self) -> *const onnx_genai_ort_sys::OrtValue {
        self.ptr.as_ptr()
    }

    pub(crate) fn raw_ptr_addr(&self) -> usize {
        self.ptr.as_ptr() as usize
    }

    /// Return the address of the tensor data buffer. Intended for tests and
    /// decode-session diagnostics that need to verify buffer reuse.
    pub fn data_ptr_addr(&self) -> Result<usize> {
        Ok(tensor_data_ptr(self.ptr.as_ptr())? as usize)
    }

    /// Overwrite the leading `data.len()` Int64 elements of this tensor in
    /// place, leaving the tensor's OrtValue (and its buffer address) unchanged.
    ///
    /// This is the update primitive for the static-shape captured decode loop:
    /// the persistent `input_ids` / `position_ids` / `attention_mask` buffers
    /// keep the fixed device/host addresses that a captured CUDA graph replays
    /// against, while their contents change every token. `data.len()` may be
    /// smaller than the tensor to update only a prefix (e.g. the valid region
    /// of a max-length attention mask).
    pub fn write_i64_prefix(&self, data: &[i64]) -> Result<()> {
        if self.dtype != DataType::Int64 {
            return Err(OrtError::InvalidArgument(format!(
                "write_i64_prefix requires an Int64 tensor, got {:?}",
                self.dtype
            )));
        }
        if data.len() > self.numel() {
            return Err(OrtError::InvalidArgument(format!(
                "write_i64_prefix length {} exceeds tensor capacity {}",
                data.len(),
                self.numel()
            )));
        }
        let dst = tensor_data_ptr(self.ptr.as_ptr())?.cast::<i64>();
        // SAFETY: `dst` points to at least `numel()` contiguous i64 elements
        // owned by this tensor; we write only the first `data.len()` of them.
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), dst, data.len()) };
        Ok(())
    }

    /// Set `count` consecutive `Int64` elements starting at `start` to `value`,
    /// in place, without allocating a temporary buffer.
    ///
    /// Companion to [`write_i64_prefix`](Self::write_i64_prefix) for the
    /// captured-decode attention mask: the mask's valid region grows by one
    /// element per token, so each step fills only the newly-valid tail
    /// (typically a single element) instead of rewriting the whole prefix —
    /// keeping the captured-decode step O(1) rather than O(context).
    pub fn fill_i64_range(&self, start: usize, count: usize, value: i64) -> Result<()> {
        if self.dtype != DataType::Int64 {
            return Err(OrtError::InvalidArgument(format!(
                "fill_i64_range requires an Int64 tensor, got {:?}",
                self.dtype
            )));
        }
        let end = start.checked_add(count).ok_or_else(|| {
            OrtError::InvalidArgument("fill_i64_range range overflows usize".into())
        })?;
        if end > self.numel() {
            return Err(OrtError::InvalidArgument(format!(
                "fill_i64_range end {} exceeds tensor capacity {}",
                end,
                self.numel()
            )));
        }
        if count == 0 {
            return Ok(());
        }
        let base = tensor_data_ptr(self.ptr.as_ptr())?.cast::<i64>();
        // SAFETY: `[start, start+count)` lies within the `numel()` contiguous
        // i64 elements owned by this tensor (checked above), so each written
        // element is in bounds.
        unsafe {
            let dst = base.add(start);
            for offset in 0..count {
                dst.add(offset).write(value);
            }
        }
        Ok(())
    }

    /// Deep-copy this tensor into a fresh host-owned [`Value`] with its own
    /// buffer. Used to snapshot a persistent captured-decode output buffer so
    /// the caller can consume it while the original is reused on the next step.
    pub fn clone_owned(&self) -> Result<Value> {
        match self.dtype {
            DataType::Float32 => Value::from_vec_f32(self.to_vec_f32()?, &self.shape),
            DataType::Float16 => Value::from_vec_f16_bits(self.to_vec_f16_bits()?, &self.shape),
            DataType::BFloat16 => Value::from_vec_bf16_bits(self.to_vec_bf16_bits()?, &self.shape),
            DataType::Int64 => Value::from_vec_i64(self.to_vec_i64()?, &self.shape),
            other => Err(OrtError::InvalidArgument(format!(
                "cannot clone tensor with dtype {other:?}"
            ))),
        }
    }

    /// Zero one row of a rank-3 row-major tensor shaped `[B, N, D]`.
    pub(crate) fn zero_rank3_row(&mut self, row: usize) -> Result<()> {
        if self.shape.len() != 3 {
            return Err(OrtError::InvalidArgument(format!(
                "zero_rank3_row requires rank-3 tensor, got {:?}",
                self.shape
            )));
        }
        let batch = self.shape[0] as usize;
        if row >= batch {
            return Err(OrtError::InvalidArgument(format!(
                "row {row} out of range for batch {batch}"
            )));
        }
        let row_len = (self.shape[1] as usize)
            .checked_mul(self.shape[2] as usize)
            .ok_or_else(|| {
                OrtError::InvalidArgument(format!("tensor shape too large: {:?}", self.shape))
            })?;
        let start = row
            .checked_mul(row_len)
            .ok_or_else(|| OrtError::InvalidArgument("row offset overflow".into()))?;
        match &mut self.backing {
            TensorBacking::F32(data) => data[start..start + row_len].fill(0.0),
            TensorBacking::F16(data) => data[start..start + row_len].fill(0),
            TensorBacking::None => match self.dtype {
                DataType::Float32 => {
                    let ptr = tensor_data_ptr(self.ptr.as_ptr())?.cast::<f32>();
                    // SAFETY: `start..start + row_len` lies within this tensor's
                    // row-major allocation, and ORT returned a mutable data pointer.
                    unsafe { std::slice::from_raw_parts_mut(ptr.add(start), row_len) }.fill(0.0);
                }
                DataType::Float16 | DataType::BFloat16 => {
                    let ptr = tensor_data_ptr(self.ptr.as_ptr())?.cast::<u16>();
                    // SAFETY: same bounds/invariants as the Float32 branch.
                    unsafe { std::slice::from_raw_parts_mut(ptr.add(start), row_len) }.fill(0);
                }
                dtype => {
                    return Err(OrtError::InvalidArgument(format!(
                        "cannot zero static-cache row for dtype {dtype:?}"
                    )));
                }
            },
            TensorBacking::I64(_) | TensorBacking::Alias(_) => {
                return Err(OrtError::InvalidArgument(
                    "cannot zero row for non-owned or non-KV tensor".into(),
                ));
            }
        }
        Ok(())
    }

    /// Repack selected rows to the prefix of a rank-3 row-major tensor.
    pub(crate) fn pack_rank3_rows_to_prefix(&mut self, sources: &[usize]) -> Result<()> {
        if self.shape.len() != 3 {
            return Err(OrtError::InvalidArgument(format!(
                "pack_rank3_rows_to_prefix requires rank-3 tensor, got {:?}",
                self.shape
            )));
        }
        let batch = self.shape[0] as usize;
        if sources.iter().any(|&row| row >= batch) {
            return Err(OrtError::InvalidArgument(format!(
                "row pack sources {sources:?} out of range for batch {batch}"
            )));
        }
        let row_len = (self.shape[1] as usize)
            .checked_mul(self.shape[2] as usize)
            .ok_or_else(|| {
                OrtError::InvalidArgument(format!("tensor shape too large: {:?}", self.shape))
            })?;
        match &mut self.backing {
            TensorBacking::F32(data) => {
                let mut prefix = Vec::with_capacity(sources.len() * row_len);
                for &src in sources {
                    let start = src * row_len;
                    prefix.extend_from_slice(&data[start..start + row_len]);
                }
                data[..prefix.len()].copy_from_slice(&prefix);
            }
            TensorBacking::F16(data) => {
                let mut prefix = Vec::with_capacity(sources.len() * row_len);
                for &src in sources {
                    let start = src * row_len;
                    prefix.extend_from_slice(&data[start..start + row_len]);
                }
                data[..prefix.len()].copy_from_slice(&prefix);
            }
            TensorBacking::None => match self.dtype {
                DataType::Float32 => {
                    let ptr = tensor_data_ptr(self.ptr.as_ptr())?.cast::<f32>();
                    let mut prefix = Vec::with_capacity(sources.len() * row_len);
                    for &src in sources {
                        // SAFETY: `src` was range-checked above.
                        let row =
                            unsafe { std::slice::from_raw_parts(ptr.add(src * row_len), row_len) };
                        prefix.extend_from_slice(row);
                    }
                    // SAFETY: the prefix length is at most the tensor allocation.
                    unsafe {
                        std::slice::from_raw_parts_mut(ptr, prefix.len()).copy_from_slice(&prefix);
                    }
                }
                DataType::Float16 | DataType::BFloat16 => {
                    let ptr = tensor_data_ptr(self.ptr.as_ptr())?.cast::<u16>();
                    let mut prefix = Vec::with_capacity(sources.len() * row_len);
                    for &src in sources {
                        // SAFETY: `src` was range-checked above.
                        let row =
                            unsafe { std::slice::from_raw_parts(ptr.add(src * row_len), row_len) };
                        prefix.extend_from_slice(row);
                    }
                    // SAFETY: the prefix length is at most the tensor allocation.
                    unsafe {
                        std::slice::from_raw_parts_mut(ptr, prefix.len()).copy_from_slice(&prefix);
                    }
                }
                dtype => {
                    return Err(OrtError::InvalidArgument(format!(
                        "cannot pack static-cache rows for dtype {dtype:?}"
                    )));
                }
            },
            TensorBacking::I64(_) | TensorBacking::Alias(_) => {
                return Err(OrtError::InvalidArgument(
                    "cannot pack rows for non-owned or non-KV tensor".into(),
                ));
            }
        }
        Ok(())
    }

    /// Create a no-copy CPU tensor alias over the prefix of an existing tensor.
    ///
    /// The returned OrtValue has its own shape but points at the same underlying
    /// tensor data as `owner`. `owner` is kept alive by the alias backing.
    pub fn alias_with_shape(owner: Arc<Value>, shape: &[i64]) -> Result<Self> {
        validate_shape(shape, None)?;
        let alias_numel = shape.iter().try_fold(1usize, |acc, &dim| {
            acc.checked_mul(dim as usize).ok_or_else(|| {
                OrtError::InvalidArgument(format!("tensor shape too large: {shape:?}"))
            })
        })?;
        if alias_numel > owner.numel() {
            return Err(OrtError::InvalidArgument(format!(
                "alias shape {:?} has {} elements, larger than owner shape {:?} with {} elements",
                shape,
                alias_numel,
                owner.shape(),
                owner.numel()
            )));
        }
        let data = tensor_data_ptr(owner.ptr.as_ptr())?;
        let ptr = create_tensor_with_data(
            data,
            alias_numel * owner.dtype.size_of(),
            shape,
            owner.dtype,
        )?;
        Ok(Self {
            ptr,
            shape: shape.to_vec(),
            dtype: owner.dtype,
            backing: TensorBacking::Alias(owner),
        })
    }

    pub(crate) unsafe fn from_raw(ptr: *mut onnx_genai_ort_sys::OrtValue) -> Result<Self> {
        let ptr = NonNull::new(ptr).ok_or(OrtError::NullPointer)?;
        let (shape, dtype) = tensor_shape_and_type(ptr.as_ptr())?;
        Ok(Self {
            ptr,
            shape,
            dtype,
            backing: TensorBacking::None,
        })
    }
}

impl Drop for Value {
    fn drop(&mut self) {
        let _keep_data_alive = match &self.backing {
            TensorBacking::F32(data) => data.len(),
            TensorBacking::F16(data) => data.len(),
            TensorBacking::I64(data) => data.len(),
            TensorBacking::Alias(owner) => owner.numel(),
            TensorBacking::None => 0,
        };
        if let Ok(api) = crate::error::api()
            && let Some(release) = api.ReleaseValue
        {
            // SAFETY: `ptr` is owned by this wrapper and released exactly once here.
            unsafe { release(self.ptr.as_ptr()) };
        }
    }
}

fn validate_shape(shape: &[i64], actual_len: Option<usize>) -> Result<()> {
    let mut expected_len = 1usize;
    for &dim in shape {
        if dim < 0 {
            return Err(OrtError::InvalidArgument(format!(
                "tensor shape contains negative dimension: {shape:?}"
            )));
        }
        expected_len = expected_len.checked_mul(dim as usize).ok_or_else(|| {
            OrtError::InvalidArgument(format!("tensor shape too large: {shape:?}"))
        })?;
    }
    if let Some(actual_len) = actual_len
        && actual_len != expected_len
    {
        return Err(OrtError::InvalidArgument(format!(
            "data length {actual_len} doesn't match shape {shape:?} (expected {expected_len})"
        )));
    }
    Ok(())
}

fn create_tensor_with_data(
    data: *mut std::ffi::c_void,
    bytes: usize,
    shape: &[i64],
    dtype: DataType,
) -> Result<NonNull<onnx_genai_ort_sys::OrtValue>> {
    let memory_info = MemoryInfo::cpu()?;
    let mut ptr = std::ptr::null_mut();
    let api = crate::error::api()?;
    let create = api
        .CreateTensorWithDataAsOrtValue
        .ok_or(OrtError::ApiUnavailable("CreateTensorWithDataAsOrtValue"))?;
    // SAFETY: `data` points to an owned Vec buffer held by Value::backing for
    // at least the lifetime of the OrtValue. `shape` is valid for the call.
    crate::error::check_status(unsafe {
        create(
            memory_info.as_ptr(),
            data,
            bytes,
            shape.as_ptr(),
            shape.len(),
            dtype.to_onnx(),
            &mut ptr,
        )
    })?;
    NonNull::new(ptr).ok_or(OrtError::NullPointer)
}

fn tensor_shape_and_type(
    value: *const onnx_genai_ort_sys::OrtValue,
) -> Result<(Vec<i64>, DataType)> {
    let api = crate::error::api()?;
    let get_info = api
        .GetTensorTypeAndShape
        .ok_or(OrtError::ApiUnavailable("GetTensorTypeAndShape"))?;
    let get_type = api
        .GetTensorElementType
        .ok_or(OrtError::ApiUnavailable("GetTensorElementType"))?;
    let get_dim_count = api
        .GetDimensionsCount
        .ok_or(OrtError::ApiUnavailable("GetDimensionsCount"))?;
    let get_dims = api
        .GetDimensions
        .ok_or(OrtError::ApiUnavailable("GetDimensions"))?;
    let release = api
        .ReleaseTensorTypeAndShapeInfo
        .ok_or(OrtError::ApiUnavailable("ReleaseTensorTypeAndShapeInfo"))?;

    let mut info = std::ptr::null_mut();
    // SAFETY: `value` is a valid ORT tensor value owned elsewhere; `info` is an
    // out-parameter released before returning.
    crate::error::check_status(unsafe { get_info(value, &mut info) })?;
    if info.is_null() {
        return Err(OrtError::NullPointer);
    }

    let result = (|| {
        let mut dtype = onnx_genai_ort_sys::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
        // SAFETY: `info` is a valid tensor type info pointer.
        crate::error::check_status(unsafe { get_type(info, &mut dtype) })?;
        let dtype = DataType::from_onnx(dtype)?;

        let mut dim_count = 0usize;
        // SAFETY: `info` is valid and `dim_count` is an out-parameter.
        crate::error::check_status(unsafe { get_dim_count(info, &mut dim_count) })?;
        let mut shape = vec![0i64; dim_count];
        // SAFETY: `shape` has `dim_count` slots for ORT to fill.
        crate::error::check_status(unsafe { get_dims(info, shape.as_mut_ptr(), dim_count) })?;
        Ok((shape, dtype))
    })();

    // SAFETY: `info` was allocated by ORT for this call and is released once.
    unsafe { release(info) };
    result
}

fn tensor_data_to_vec<T: Copy>(
    value: *mut onnx_genai_ort_sys::OrtValue,
    len: usize,
) -> Result<Vec<T>> {
    let api = crate::error::api()?;
    let get_data = api
        .GetTensorMutableData
        .ok_or(OrtError::ApiUnavailable("GetTensorMutableData"))?;
    let mut data = std::ptr::null_mut();
    // SAFETY: `value` is a valid tensor OrtValue; ORT returns a pointer valid
    // until the value is released. We immediately copy `len` elements out.
    crate::error::check_status(unsafe { get_data(value, &mut data) })?;
    if data.is_null() {
        return Err(OrtError::NullPointer);
    }

    // SAFETY: caller ensures `T` matches the tensor dtype and `len` is numel.
    let slice = unsafe { std::slice::from_raw_parts(data.cast::<T>(), len) };
    Ok(slice.to_vec())
}

fn tensor_data_ptr(value: *mut onnx_genai_ort_sys::OrtValue) -> Result<*mut std::ffi::c_void> {
    let api = crate::error::api()?;
    let get_data = api
        .GetTensorMutableData
        .ok_or(OrtError::ApiUnavailable("GetTensorMutableData"))?;
    let mut data = std::ptr::null_mut();
    // SAFETY: `value` is a valid tensor OrtValue; ORT returns a pointer valid
    // until the value is released. The caller keeps the owner alive.
    crate::error::check_status(unsafe { get_data(value, &mut data) })?;
    if data.is_null() {
        return Err(OrtError::NullPointer);
    }
    Ok(data)
}
