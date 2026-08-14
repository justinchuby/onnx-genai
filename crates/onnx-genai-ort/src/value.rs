//! ORT Values (tensors).

use std::borrow::Cow;
use std::ptr::NonNull;
use std::sync::Arc;

use crate::{MemoryInfo, OrtError, Result};

/// Alignment that satisfies every [`DataType`] element type.
///
/// The widest element this crate supports is 8 bytes (`Int64`/`Uint64`), so a
/// buffer aligned to this is aligned for every tensor dtype.
const MAX_ELEMENT_ALIGN: usize = 8;

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

    /// Alignment one element must be stored at.
    ///
    /// Every dtype here is a scalar whose alignment equals its size, but this is
    /// stated separately from [`size_of`](Self::size_of) because it answers a
    /// different question: `size_of` sizes an allocation, while this is the
    /// precondition `slice::from_raw_parts` and `ptr::write` impose on the
    /// pointer ORT hands back for the tensor's data.
    pub fn align_of(&self) -> usize {
        self.size_of()
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
    /// Raw little-endian element bytes for a tensor of arbitrary dtype (used by
    /// the backend-neutral component-session seam, which carries host tensors as
    /// opaque bytes).
    Bytes(ElementBytes),
    Alias(Arc<Value>),
    /// Memory this `Value` does not own.
    ///
    /// Used when a caller hands ORT a buffer it allocated itself — typically
    /// device memory from an external memory manager. The `Value` borrows it,
    /// so keeping it alive is the caller's obligation, enforced at the unsafe
    /// constructor rather than here.
    ///
    /// `host_accessible` records whether the memory info named a host device.
    /// Every accessor here reaches the bytes through `GetTensorMutableData`,
    /// which returns whatever address the tensor holds; dereferencing a device
    /// address on the CPU is a wild access, not an error, so the accessors
    /// consult this rather than trying and faulting.
    External {
        host_accessible: bool,
    },
    None,
}

/// Owned tensor bytes whose data pointer is aligned for **every** [`DataType`].
///
/// `Vec<u8>` only guarantees `align_of::<u8>() == 1`, and an *empty* `Vec<u8>`
/// is not even a real allocation: its pointer is the dangling address `0x1`.
/// `CreateTensorWithDataAsOrtValue` stores whatever pointer it is given and
/// `GetTensorMutableData` hands that same pointer straight back, so a
/// `Vec<u8>`-backed tensor read as `Float16`/`Float32`/`Int64` would build a
/// slice from a pointer that is not aligned for its element type — undefined
/// behaviour, which Rust's debug UB checks turn into a non-unwinding abort.
///
/// The caller's allocation is kept whenever it already satisfies
/// [`MAX_ELEMENT_ALIGN`] (which every non-empty `malloc`/`HeapAlloc` block does
/// in practice), so the common path stays a move with no extra copy. Otherwise
/// the bytes are re-homed into a `u64`-backed allocation, which the allocator
/// must return 8-byte aligned. Either way the pointer handed to ORT is aligned
/// for any dtype the tensor can carry.
enum ElementBytes {
    /// The caller's `Vec<u8>`, kept because it is already suitably aligned.
    Borrowed(Vec<u8>),
    /// Re-homed bytes: `words` holds `len` meaningful bytes plus tail padding.
    Realigned { words: Vec<u64>, len: usize },
}

impl ElementBytes {
    fn new(data: Vec<u8>) -> Self {
        if !data.is_empty() && (data.as_ptr() as usize).is_multiple_of(MAX_ELEMENT_ALIGN) {
            return Self::Borrowed(data);
        }
        let len = data.len();
        // `max(1)` keeps this a real allocation even for an empty tensor, so the
        // pointer ORT receives is a genuine aligned address rather than `Vec`'s
        // dangling `align_of::<u64>()` sentinel.
        let mut words = vec![0u64; len.div_ceil(MAX_ELEMENT_ALIGN).max(1)];
        // SAFETY: `words` owns at least `len` bytes (rounded up to whole u64s)
        // and does not overlap `data`; both are read/written as bytes, which has
        // alignment 1, so no alignment precondition applies to either side.
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), words.as_mut_ptr().cast::<u8>(), len);
        }
        Self::Realigned { words, len }
    }

    fn len(&self) -> usize {
        match self {
            Self::Borrowed(data) => data.len(),
            Self::Realigned { len, .. } => *len,
        }
    }

    /// Pointer to the first byte, aligned to [`MAX_ELEMENT_ALIGN`].
    fn as_mut_ptr(&mut self) -> *mut u8 {
        let ptr = match self {
            Self::Borrowed(data) => data.as_mut_ptr(),
            Self::Realigned { words, .. } => words.as_mut_ptr().cast::<u8>(),
        };
        debug_assert!(
            (ptr as usize).is_multiple_of(MAX_ELEMENT_ALIGN),
            "ElementBytes must hand ORT a pointer aligned for every element type"
        );
        ptr
    }
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

    /// Wrap memory this `Value` does not own, wherever `memory_info` says it
    /// lives.
    ///
    /// This is how an external memory manager hands ONNX Runtime a buffer it
    /// allocated itself — device memory in particular. Every other constructor
    /// here owns a `Vec` and reports host memory; this one owns nothing and
    /// takes the location as a parameter, because ORT uses it to decide whether
    /// the pointer is a host address or a device one.
    ///
    /// # Safety
    ///
    /// * `data` must be valid for `bytes` and remain so for the entire lifetime
    ///   of the returned `Value` **and of anything ORT derives from it**. ORT
    ///   does not copy, so a buffer freed early becomes a use-after-free inside
    ///   the runtime rather than a Rust error.
    /// * `memory_info` must describe where `data` actually lives. Saying host
    ///   for a device pointer makes ORT dereference a device address on the CPU.
    /// * `bytes` must cover `shape` at `dtype`, which is checked, but the check
    ///   cannot see whether the allocation behind `data` is really that large.
    pub unsafe fn from_external_memory(
        data: *mut std::ffi::c_void,
        bytes: usize,
        shape: &[i64],
        dtype: DataType,
        memory_info: &MemoryInfo,
    ) -> Result<Self> {
        if data.is_null() {
            return Err(OrtError::InvalidArgument(
                "cannot wrap a null pointer as an external tensor; allocate the buffer first, or \
                 use a Value constructor that owns its data"
                    .to_owned(),
            ));
        }
        // Every reader reaches this buffer as `*mut dtype`, and building a slice
        // or writing through a pointer that is not aligned for its element type
        // is undefined behaviour. This is the one constructor whose buffer we do
        // not allocate, so it is the only place the invariant can be violated —
        // report it here, where the caller can still fix the allocation.
        //
        // A zero-element tensor is exempt, and deliberately so: no reader ever
        // dereferences its data pointer, so there is nothing to misalign. The
        // exemption is also load-bearing, because the natural way to produce an
        // empty buffer is an empty `Vec<u8>`, whose dangling pointer is `0x1` —
        // non-null, and never aligned for a multi-byte dtype.
        if bytes > 0 && !(data as usize).is_multiple_of(dtype.align_of()) {
            return Err(OrtError::InvalidArgument(format!(
                "external buffer at {data:p} is not aligned to {} bytes as {dtype:?} elements \
                 require; allocate it with at least element alignment",
                dtype.align_of()
            )));
        }
        validate_shape(shape, None)?;
        let elements = shape
            .iter()
            .try_fold(1usize, |acc, &d| acc.checked_mul(d as usize))
            .ok_or_else(|| {
                OrtError::InvalidArgument(format!("tensor shape too large: {shape:?}"))
            })?;
        let needed = elements.checked_mul(dtype.size_of()).ok_or_else(|| {
            OrtError::InvalidArgument(format!("tensor shape too large: {shape:?}"))
        })?;
        if bytes < needed {
            return Err(OrtError::InvalidArgument(format!(
                "external buffer of {bytes} bytes is too small for shape {shape:?} at {dtype:?}, \
                 which needs {needed}; ORT would read past the end of the allocation"
            )));
        }
        let host_accessible = memory_info.device_name == "Cpu";
        let ptr = create_tensor_with_data_in(data, bytes, shape, dtype, memory_info)?;
        Ok(Self {
            ptr,
            shape: shape.to_vec(),
            dtype,
            backing: TensorBacking::External { host_accessible },
        })
    }

    /// Whether this value's bytes can be read or written through a host
    /// pointer.
    ///
    /// Only external tensors can answer `false`: every other constructor here
    /// owns a host `Vec`.
    pub fn is_host_accessible(&self) -> bool {
        !matches!(
            self.backing,
            TensorBacking::External {
                host_accessible: false
            }
        )
    }

    /// Reject host access to a tensor whose bytes are not on the host.
    ///
    /// The accessors reach the data through `GetTensorMutableData`, which hands
    /// back the tensor's own address with no indication of where it lives. On a
    /// device tensor that address is not dereferenceable from the CPU, so the
    /// alternative to this check is a wild read or store rather than a fault.
    fn ensure_host_accessible(&self, operation: &str) -> Result<()> {
        if self.is_host_accessible() && self.is_host_resident()? {
            return Ok(());
        }
        Err(OrtError::InvalidArgument(format!(
            "{operation} needs to reach this tensor's bytes through a host pointer, but it \
             lives on a device; copy it to the host first, or use the device-side helpers"
        )))
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

    /// Create a CPU tensor from raw little-endian element bytes of any dtype.
    ///
    /// This is the construction primitive for the backend-neutral
    /// component-session seam, which carries host tensors as opaque bytes so any
    /// dtype round-trips without a per-dtype host representation. `shape` must be
    /// fully static and `data.len()` must equal `numel * dtype.size_of()`.
    pub fn from_raw_bytes(data: Vec<u8>, shape: &[i64], dtype: DataType) -> Result<Self> {
        validate_shape(shape, None)?;
        let numel = shape.iter().fold(1usize, |acc, &dim| acc * dim as usize);
        let expected = numel * dtype.size_of();
        if data.len() != expected {
            return Err(OrtError::InvalidArgument(format!(
                "raw tensor byte length {} does not match a {:?} tensor of shape {:?} \
                 (expected {} bytes)",
                data.len(),
                dtype,
                shape,
                expected
            )));
        }
        // A `Vec<u8>` is only 1-byte aligned (and an empty one is the dangling
        // address `0x1`), but ORT hands this exact pointer back to every reader,
        // which casts it to the element type. Re-home it when necessary so the
        // tensor's data pointer is aligned for `dtype`.
        let mut data = ElementBytes::new(data);
        let ptr = create_tensor_with_data(data.as_mut_ptr().cast(), data.len(), shape, dtype)?;
        Ok(Self {
            ptr,
            shape: shape.to_vec(),
            dtype,
            backing: TensorBacking::Bytes(data),
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

    /// Copy the tensor's raw little-endian element bytes out of a CPU tensor.
    ///
    /// Counterpart to [`Value::from_raw_bytes`] used by the backend-neutral
    /// component-session seam. The tensor must be host-resident (the pipeline
    /// component path runs on CPU); the returned buffer is `numel *
    /// dtype.size_of()` bytes in row-major order.
    /// Whether this tensor's data lives in host memory the CPU may dereference.
    ///
    /// `GetTensorMutableData` hands back whatever address the tensor's allocator
    /// produced, which for a device-allocated value is a device pointer. Reading
    /// it as host memory is undefined behavior, so anything that dereferences the
    /// data pointer must check this first.
    pub fn is_host_resident(&self) -> Result<bool> {
        let api = crate::error::api()?;
        let get_memory_info = api
            .GetTensorMemoryInfo
            .ok_or(OrtError::ApiUnavailable("GetTensorMemoryInfo"))?;
        let get_device_type = api
            .MemoryInfoGetDeviceType
            .ok_or(OrtError::ApiUnavailable("MemoryInfoGetDeviceType"))?;
        let get_name = api
            .MemoryInfoGetName
            .ok_or(OrtError::ApiUnavailable("MemoryInfoGetName"))?;
        let mut memory_info = std::ptr::null();
        // SAFETY: `self.ptr` is a valid tensor OrtValue. ORT owns the returned
        // OrtMemoryInfo for the lifetime of the value, so it must not be freed.
        crate::error::check_status(unsafe {
            get_memory_info(self.ptr.as_ptr(), &mut memory_info)
        })?;
        if memory_info.is_null() {
            return Err(OrtError::NullPointer);
        }
        let mut device_type = onnx_genai_ort_sys::OrtMemoryInfoDeviceType_CPU;
        // SAFETY: `memory_info` is the non-null table ORT just returned, and
        // `device_type` is a valid out-parameter for the duration of the call.
        unsafe { get_device_type(memory_info, &mut device_type) };
        if device_type != onnx_genai_ort_sys::OrtMemoryInfoDeviceType_CPU {
            return Ok(false);
        }
        let mut name = std::ptr::null();
        // SAFETY: `memory_info` is valid and `name` is a live out-parameter.
        crate::error::check_status(unsafe { get_name(memory_info, &mut name) })?;
        if name.is_null() {
            return Err(OrtError::NullPointer);
        }
        // `CreateMemoryInfo`, used by the built-in CUDA allocator, does not
        // encode an OrtDevice type and ORT reports CPU here despite returning
        // a device pointer. The allocator name remains authoritative.
        let name = unsafe { std::ffi::CStr::from_ptr(name) }.to_string_lossy();
        Ok(!matches!(name.as_ref(), "Cuda" | "DML" | "WebGPU_Buffer"))
    }

    /// Return the allocator device ID recorded on this tensor.
    pub fn device_id(&self) -> Result<i32> {
        let api = crate::error::api()?;
        let get_memory_info = api
            .GetTensorMemoryInfo
            .ok_or(OrtError::ApiUnavailable("GetTensorMemoryInfo"))?;
        let get_device_id = api
            .MemoryInfoGetId
            .ok_or(OrtError::ApiUnavailable("MemoryInfoGetId"))?;
        let mut memory_info = std::ptr::null();
        // SAFETY: `self.ptr` is a valid tensor OrtValue. ORT owns the returned
        // OrtMemoryInfo for the lifetime of the value.
        crate::error::check_status(unsafe {
            get_memory_info(self.ptr.as_ptr(), &mut memory_info)
        })?;
        if memory_info.is_null() {
            return Err(OrtError::NullPointer);
        }
        let mut device_id = 0;
        // SAFETY: `memory_info` is valid and `device_id` is a live out-parameter.
        crate::error::check_status(unsafe { get_device_id(memory_info, &mut device_id) })?;
        Ok(device_id)
    }

    /// Borrow the tensor's raw little-endian element bytes.
    ///
    /// The borrowing counterpart to [`to_raw_bytes`](Self::to_raw_bytes), for
    /// readers that only scan the bytes (hashing, comparison) and would
    /// otherwise pay a full copy of a multi-megabyte tensor to do it.
    ///
    /// Errors for a device-resident tensor rather than handing back a slice over
    /// a device pointer.
    pub fn as_raw_bytes(&self) -> Result<&[u8]> {
        if !self.is_host_resident()? {
            return Err(OrtError::InvalidArgument(
                "cannot borrow bytes of a device-resident tensor; copy it to host first"
                    .to_string(),
            ));
        }
        let bytes = self.numel() * self.dtype.size_of();
        // A zero-element tensor borrows nothing, so it never needs a data
        // pointer — ORT may return null or a dangling sentinel for one.
        if bytes == 0 {
            return Ok(&[]);
        }
        let ptr = tensor_data_ptr(self.ptr.as_ptr())?;
        // SAFETY: `ptr` points to at least `bytes` contiguous bytes of this
        // tensor's row-major allocation, checked host-resident above and kept
        // alive by `self`, which also bounds the returned slice's lifetime.
        // `u8` has alignment 1, so any non-null `ptr` satisfies the alignment
        // precondition.
        Ok(unsafe { std::slice::from_raw_parts(ptr.cast::<u8>(), bytes) })
    }

    pub fn to_raw_bytes(&self) -> Result<Vec<u8>> {
        self.ensure_host_accessible("to_raw_bytes")?;
        let bytes = self.numel() * self.dtype.size_of();
        if bytes == 0 {
            return Ok(Vec::new());
        }
        let ptr = tensor_data_ptr(self.ptr.as_ptr())?;
        // SAFETY: `ptr` points to at least `bytes` contiguous bytes of this
        // host-resident tensor's row-major allocation, kept alive by `self`;
        // `u8` has alignment 1, so any non-null `ptr` is suitably aligned.
        let slice = unsafe { std::slice::from_raw_parts(ptr.cast::<u8>(), bytes) };
        Ok(slice.to_vec())
    }

    /// Copy tensor data out as f32 values.
    pub fn to_vec_f32(&self) -> Result<Vec<f32>> {
        self.ensure_host_accessible("to_vec_f32")?;
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
        self.ensure_host_accessible("to_vec_f32_lossy")?;
        match self.dtype {
            DataType::Float32 => self.to_vec_f32(),
            DataType::Float16 => {
                let numel = self.numel();
                if numel == 0 {
                    return Ok(Vec::new());
                }
                let data = tensor_data_ptr(self.ptr.as_ptr())?;
                // SAFETY: an fp16 tensor holds `numel` contiguous u16 elements at
                // `data`, valid until this Value is released; we only read here,
                // and `tensor_elements` upholds the alignment precondition.
                let bits = unsafe { tensor_elements::<u16>(data.cast::<u16>(), numel) };
                // Reinterpret the raw bits as f16 and widen with half's vectorized
                // slice conversion (hardware F16C when available), which is far
                // faster than a per-element `from_bits().to_f32()` scalar loop on
                // the hot logits path (~152K elements per decode step).
                let halves: &[half::f16] =
                    half::slice::HalfBitsSliceExt::reinterpret_cast(&bits[..]);
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

    /// Argmax over the final `vocab`-sized row of a `[.., vocab]` logits tensor.
    ///
    /// Reads the tensor in place and returns the index of the maximum element
    /// of its last row without allocating a host `Vec`. Semantics match the
    /// engine's greedy sampler exactly: NaNs are ignored, ties resolve to the
    /// lowest index, and an empty/all-NaN row selects index 0. Float16/BFloat16
    /// logits are widened with half's vectorized (hardware F16C) slice
    /// conversion before the scan, matching `to_vec_f32_lossy`.
    ///
    /// This is the host reduction behind the greedy decode fast path: instead
    /// of copying the whole ~150K-entry vocabulary out of the persistent logits
    /// buffer every token (and re-scanning it), the caller reads only the four
    /// bytes of the selected token id. The tensor must be host-readable
    /// (CPU-allocated), like every logits buffer the decode sessions bind as a
    /// CPU output.
    pub fn argmax_last_row(&self) -> Result<u32> {
        let vocab = self
            .shape
            .last()
            .copied()
            .filter(|dim| *dim > 0)
            .ok_or_else(|| {
                OrtError::InvalidArgument(format!(
                    "argmax_last_row requires a positive trailing dim, got shape {:?}",
                    self.shape
                ))
            })? as usize;
        let numel = self.numel();
        let offset = numel.checked_sub(vocab).ok_or_else(|| {
            OrtError::InvalidArgument(format!(
                "argmax_last_row row size {vocab} exceeds tensor length {numel}"
            ))
        })?;
        let data = tensor_data_ptr(self.ptr.as_ptr())?;
        // SAFETY: the tensor owns `numel` contiguous elements of `dtype` at
        // `data`, valid until the value is released; we only read the final row
        // `[offset, offset + vocab)`, which is in bounds by construction. The
        // row base is stepped in bytes so the arithmetic stays valid even when
        // `data` is not element-aligned, and `tensor_elements` then upholds the
        // alignment precondition itself.
        let row_base = unsafe { data.cast::<u8>().add(offset * self.dtype.size_of()) };
        let index = match self.dtype {
            DataType::Float32 => {
                let row = unsafe { tensor_elements::<f32>(row_base.cast::<f32>(), vocab) };
                argmax_row_f32(&row)
            }
            DataType::Float16 => {
                let bits = unsafe { tensor_elements::<u16>(row_base.cast::<u16>(), vocab) };
                argmax_f16_bits(&bits)
            }
            DataType::BFloat16 => {
                let bits = unsafe { tensor_elements::<u16>(row_base.cast::<u16>(), vocab) };
                argmax_bf16_bits(&bits)
            }
            other => {
                return Err(OrtError::InvalidArgument(format!(
                    "argmax_last_row does not support {other:?} logits"
                )));
            }
        };
        Ok(index as u32)
    }

    /// Copy Float16 tensor data out as IEEE-754 half-precision bit patterns.
    pub fn to_vec_f16_bits(&self) -> Result<Vec<u16>> {
        self.ensure_host_accessible("to_vec_f16_bits")?;
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
        self.ensure_host_accessible("to_vec_bf16_bits")?;
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
        self.ensure_host_accessible("to_vec_i64")?;
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

    /// Copy a host tensor into this existing host tensor without changing its
    /// OrtValue or buffer address. Stable workflow-island bindings use this to
    /// refresh request/loop inputs while preserving CUDA Graph replay addresses.
    pub fn copy_from_host(&self, source: &Value) -> Result<()> {
        self.ensure_host_accessible("copy_from_host destination")?;
        source.ensure_host_accessible("copy_from_host source")?;
        if self.dtype != source.dtype || self.shape != source.shape {
            return Err(OrtError::InvalidArgument(format!(
                "copy_from_host requires identical tensors, destination {:?} {:?}, source {:?} {:?}",
                self.dtype, self.shape, source.dtype, source.shape
            )));
        }
        let bytes = self
            .numel()
            .checked_mul(self.dtype.size_of())
            .ok_or_else(|| {
                OrtError::InvalidArgument("copy_from_host tensor byte size overflows".into())
            })?;
        let destination = tensor_data_ptr(self.ptr.as_ptr())?;
        let source = tensor_data_ptr(source.ptr.as_ptr())?;
        // SAFETY: shape/dtype equality guarantees both tensor buffers contain
        // at least `bytes` bytes and do not overlap (distinct OrtValues).
        unsafe { std::ptr::copy_nonoverlapping(source, destination, bytes) };
        Ok(())
    }

    /// Copy an identically typed/shaped tensor between host and CUDA memory.
    ///
    /// CPU-to-CPU copies use ordinary host memory. Any copy involving device
    /// memory uses the selected CUDA device explicitly so callers can refresh
    /// stable graph-capture buffers without replacing their addresses.
    pub fn copy_from_cuda(&self, source: &Value, device_id: i32) -> Result<()> {
        if self.shape != source.shape || self.dtype != source.dtype {
            return Err(OrtError::InvalidArgument(format!(
                "CUDA tensor copy requires identical tensors, destination {:?} {:?}, source {:?} {:?}",
                self.dtype, self.shape, source.dtype, source.shape
            )));
        }

        let destination_host = self.is_host_resident()?;
        let source_host = source.is_host_resident()?;
        if destination_host && source_host {
            return self.copy_from_host(source);
        }
        #[cfg(feature = "cuda")]
        {
            let bytes = self
                .numel()
                .checked_mul(self.dtype.size_of())
                .ok_or_else(|| {
                    OrtError::InvalidArgument("CUDA tensor copy byte size overflows".into())
                })?;
            let destination = self.data_ptr_addr()?;
            let source_ptr = source.data_ptr_addr()?;
            let _guard = crate::cuda_rt::DeviceGuard::set(device_id)?;
            match (destination_host, source_host) {
                (false, true) => {
                    crate::cuda_rt::memcpy_host_to_device(destination, source.as_raw_bytes()?)?
                }
                (true, false) => {
                    // SAFETY: destination is a host tensor with exactly `bytes`
                    // writable bytes and remains alive for the copy.
                    let destination_bytes =
                        unsafe { std::slice::from_raw_parts_mut(destination as *mut u8, bytes) };
                    crate::cuda_rt::memcpy_device_to_host(destination_bytes, source_ptr)?
                }
                (false, false) => {
                    crate::cuda_rt::memcpy_device_to_device(destination, source_ptr, bytes)?
                }
                (true, true) => unreachable!(),
            }
            return Ok(());
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = device_id;
            Err(OrtError::InvalidArgument(
                "tensor copy involves CUDA memory but this build has no CUDA support".into(),
            ))
        }
    }

    /// Enqueue a same-device CUDA copy without synchronizing the device.
    pub fn copy_from_cuda_async(&self, source: &Value, device_id: i32) -> Result<()> {
        if self.shape != source.shape || self.dtype != source.dtype {
            return Err(OrtError::InvalidArgument(format!(
                "asynchronous CUDA tensor copy requires identical tensors, destination {:?} {:?}, source {:?} {:?}",
                self.dtype, self.shape, source.dtype, source.shape
            )));
        }
        if self.is_host_resident()? || source.is_host_resident()? {
            return Err(OrtError::InvalidArgument(
                "asynchronous CUDA tensor copy requires device-resident tensors".into(),
            ));
        }
        #[cfg(feature = "cuda")]
        {
            let bytes = self
                .numel()
                .checked_mul(self.dtype.size_of())
                .ok_or_else(|| {
                    OrtError::InvalidArgument("CUDA tensor copy byte size overflows".into())
                })?;
            let _guard = crate::cuda_rt::DeviceGuard::set(device_id)?;
            return crate::cuda_rt::memcpy_device_to_device_async(
                self.data_ptr_addr()?,
                source.data_ptr_addr()?,
                bytes,
            );
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = device_id;
            Err(OrtError::InvalidArgument(
                "asynchronous CUDA copy requires the cuda feature".into(),
            ))
        }
    }

    /// Copy this tensor into a newly allocated CPU tensor.
    pub fn to_host_from_cuda(&self, device_id: i32) -> Result<Self> {
        let host = Self::empty(&self.shape, self.dtype)?;
        host.copy_from_cuda(self, device_id)?;
        Ok(host)
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
        self.ensure_host_accessible("write_i64_prefix")?;
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
        if data.is_empty() {
            return Ok(());
        }
        let dst = tensor_elements_mut_ptr::<i64>(self.ptr.as_ptr(), self.dtype)?;
        // SAFETY: `dst` points to at least `numel()` contiguous i64 elements
        // owned by this tensor and is element-aligned (checked above); we write
        // only the first `data.len()` of them.
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
        self.ensure_host_accessible("fill_i64_range")?;
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
        let base = tensor_elements_mut_ptr::<i64>(self.ptr.as_ptr(), self.dtype)?;
        // SAFETY: `[start, start+count)` lies within the `numel()` contiguous
        // i64 elements owned by this tensor (checked above) and `base` is
        // element-aligned, so each written element is in bounds and aligned.
        unsafe {
            let dst = base.add(start);
            for offset in 0..count {
                dst.add(offset).write(value);
            }
        }
        Ok(())
    }

    /// Overwrite the leading `data.len()` `Int64` elements of a **CUDA
    /// device-resident** tensor in place via a host->device copy, leaving the
    /// tensor's OrtValue (and its device buffer address) unchanged.
    ///
    /// Device counterpart of [`write_i64_prefix`](Self::write_i64_prefix): the
    /// captured decode loop keeps `input_ids` / `position_ids` device-resident so
    /// the captured CUDA graph reads them in place on every replay (no per-step
    /// clear + re-bind of the IoBinding set — see the note on
    /// [`DecodeSession::step_captured`](crate::decode) and issue
    /// microsoft/onnxruntime#29782). `device_id` pins the copy to the tensor's
    /// CUDA device.
    #[cfg(feature = "cuda")]
    pub fn write_i64_prefix_device(&self, data: &[i64], device_id: i32) -> Result<()> {
        if self.dtype != DataType::Int64 {
            return Err(OrtError::InvalidArgument(format!(
                "write_i64_prefix_device requires an Int64 tensor, got {:?}",
                self.dtype
            )));
        }
        if data.len() > self.numel() {
            return Err(OrtError::InvalidArgument(format!(
                "write_i64_prefix_device length {} exceeds tensor capacity {}",
                data.len(),
                self.numel()
            )));
        }
        if data.is_empty() {
            return Ok(());
        }
        let dst = self.data_ptr_addr()?;
        // SAFETY: `data` is a valid `[i64]`; reinterpreting it as bytes for the
        // duration of this call yields exactly `size_of_val(data)` initialized
        // bytes with no aliasing concerns (read-only view).
        let src = unsafe {
            std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), std::mem::size_of_val(data))
        };
        let _guard = crate::cuda_rt::DeviceGuard::set(device_id)?;
        crate::cuda_rt::memcpy_host_to_device(dst, src)
    }

    /// Set `count` consecutive `Int64` elements starting at `start` of a **CUDA
    /// device-resident** tensor to `value`, in place, via a host->device copy.
    ///
    /// Device counterpart of [`fill_i64_range`](Self::fill_i64_range) for the
    /// captured-decode attention mask (see
    /// [`write_i64_prefix_device`](Self::write_i64_prefix_device)). Stages the
    /// `count` values in a host buffer and copies them to the tensor's device
    /// memory at the element offset `start`.
    #[cfg(feature = "cuda")]
    pub fn fill_i64_range_device(
        &self,
        start: usize,
        count: usize,
        value: i64,
        device_id: i32,
    ) -> Result<()> {
        if self.dtype != DataType::Int64 {
            return Err(OrtError::InvalidArgument(format!(
                "fill_i64_range_device requires an Int64 tensor, got {:?}",
                self.dtype
            )));
        }
        let end = start.checked_add(count).ok_or_else(|| {
            OrtError::InvalidArgument("fill_i64_range_device range overflows usize".into())
        })?;
        if end > self.numel() {
            return Err(OrtError::InvalidArgument(format!(
                "fill_i64_range_device end {} exceeds tensor capacity {}",
                end,
                self.numel()
            )));
        }
        if count == 0 {
            return Ok(());
        }
        let dst = self
            .data_ptr_addr()?
            .checked_add(start * std::mem::size_of::<i64>())
            .ok_or_else(|| {
                OrtError::InvalidArgument("fill_i64_range_device offset overflows usize".into())
            })?;
        let host = vec![value; count];
        // SAFETY: `host` is a valid `[i64; count]`; the byte view is read-only
        // and lives for the duration of the copy.
        let src = unsafe {
            std::slice::from_raw_parts(host.as_ptr().cast::<u8>(), std::mem::size_of_val(&host[..]))
        };
        let _guard = crate::cuda_rt::DeviceGuard::set(device_id)?;
        crate::cuda_rt::memcpy_host_to_device(dst, src)
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
            // General, dtype-agnostic deep copy: round-trip the raw little-endian
            // element bytes for every remaining POD dtype (Bool, Int32, Int8,
            // Uint8/16/32/64, Int16, Float8*) rather than rejecting them per
            // dtype. `as_raw_bytes` errors precisely on a device-resident tensor
            // instead of reading a device pointer as host memory, so this never
            // silently corrupts a stray device value.
            other => Value::from_raw_bytes(self.as_raw_bytes()?.to_vec(), &self.shape, other),
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
        if row_len == 0 {
            return Ok(());
        }
        match &mut self.backing {
            TensorBacking::F32(data) => data[start..start + row_len].fill(0.0),
            TensorBacking::F16(data) => data[start..start + row_len].fill(0),
            TensorBacking::None => match self.dtype {
                DataType::Float32 => {
                    let ptr = tensor_elements_mut_ptr::<f32>(self.ptr.as_ptr(), self.dtype)?;
                    // SAFETY: `start..start + row_len` lies within this tensor's
                    // row-major allocation, ORT returned a mutable data pointer,
                    // and the pointer is element-aligned (checked above).
                    unsafe { std::slice::from_raw_parts_mut(ptr.add(start), row_len) }.fill(0.0);
                }
                DataType::Float16 | DataType::BFloat16 => {
                    let ptr = tensor_elements_mut_ptr::<u16>(self.ptr.as_ptr(), self.dtype)?;
                    // SAFETY: same bounds/invariants as the Float32 branch.
                    unsafe { std::slice::from_raw_parts_mut(ptr.add(start), row_len) }.fill(0);
                }
                dtype => {
                    return Err(OrtError::InvalidArgument(format!(
                        "cannot zero static-cache row for dtype {dtype:?}"
                    )));
                }
            },
            TensorBacking::I64(_) | TensorBacking::Bytes(_) | TensorBacking::Alias(_) => {
                return Err(OrtError::InvalidArgument(
                    "cannot zero row for non-owned or non-KV tensor".into(),
                ));
            }
            TensorBacking::External { .. } => {
                // The buffer may live on a device, and this path writes through
                // a host pointer. Refusing is the only safe answer: the caller
                // owns the memory and knows how to clear it where it lives.
                return Err(OrtError::InvalidArgument(
                    "cannot zero a row of an externally owned tensor from the host; the buffer \
                     may be device memory. Clear it through whatever allocated it, or pass an \
                     owned tensor instead"
                        .into(),
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
        if row_len == 0 || sources.is_empty() {
            return Ok(());
        }
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
                    let ptr = tensor_elements_mut_ptr::<f32>(self.ptr.as_ptr(), self.dtype)?;
                    let mut prefix = Vec::with_capacity(sources.len() * row_len);
                    for &src in sources {
                        // SAFETY: `src` was range-checked above and `ptr` is
                        // element-aligned (checked by `tensor_elements_mut_ptr`).
                        let row =
                            unsafe { std::slice::from_raw_parts(ptr.add(src * row_len), row_len) };
                        prefix.extend_from_slice(row);
                    }
                    // SAFETY: the prefix length is at most the tensor allocation,
                    // and `ptr` is element-aligned.
                    unsafe {
                        std::slice::from_raw_parts_mut(ptr, prefix.len()).copy_from_slice(&prefix);
                    }
                }
                DataType::Float16 | DataType::BFloat16 => {
                    let ptr = tensor_elements_mut_ptr::<u16>(self.ptr.as_ptr(), self.dtype)?;
                    let mut prefix = Vec::with_capacity(sources.len() * row_len);
                    for &src in sources {
                        // SAFETY: same bounds/alignment invariants as the
                        // Float32 branch.
                        let row =
                            unsafe { std::slice::from_raw_parts(ptr.add(src * row_len), row_len) };
                        prefix.extend_from_slice(row);
                    }
                    // SAFETY: the prefix length is at most the tensor allocation,
                    // and `ptr` is element-aligned.
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
            TensorBacking::I64(_) | TensorBacking::Bytes(_) | TensorBacking::Alias(_) => {
                return Err(OrtError::InvalidArgument(
                    "cannot pack rows for non-owned or non-KV tensor".into(),
                ));
            }
            TensorBacking::External { .. } => {
                // Same reasoning as zeroing: this repacks through a host
                // pointer, and an external buffer may be device memory.
                return Err(OrtError::InvalidArgument(
                    "cannot repack rows of an externally owned tensor from the host; the buffer \
                     may be device memory. Repack it through whatever allocated it, or pass an \
                     owned tensor instead"
                        .into(),
                ));
            }
        }
        Ok(())
    }

    /// Create a no-copy tensor alias over the prefix of an existing tensor.
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
        let memory_info = tensor_memory_info(owner.ptr.as_ptr())?;
        let ptr = create_tensor_with_data_at(
            memory_info,
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

    /// Convert an owned tensor into a no-copy alias with the requested shape.
    #[allow(clippy::arc_with_non_send_sync)]
    pub fn into_alias_with_shape(owner: Value, shape: &[i64]) -> Result<Self> {
        Self::alias_with_shape(Arc::new(owner), shape)
    }

    /// Create a no-copy tensor view while retaining a shared allocation owner.
    pub fn alias_from_shared_owner(owner: Arc<Value>, shape: &[i64]) -> Result<Self> {
        Self::alias_with_shape(owner, shape)
    }

    /// If this value is a no-copy alias over a shared owner, produce another
    /// alias over the same owner (O(1), no byte copy). Returns `None` for
    /// owned-backing tensors, which must be deep-copied to be shared.
    ///
    /// This lets read-only, per-step-invariant inputs (e.g. an encoder-decoder's
    /// static cross-attention KV) be re-bound every decode step without
    /// reallocating or memcpy-ing the underlying buffer.
    pub fn try_alias_clone(&self) -> Option<Result<Value>> {
        match &self.backing {
            TensorBacking::Alias(owner) => {
                Some(Value::alias_with_shape(Arc::clone(owner), &self.shape))
            }
            _ => None,
        }
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
            TensorBacking::Bytes(data) => data.len(),
            TensorBacking::Alias(owner) => owner.numel(),
            // Borrowed memory: nothing here keeps it alive, by construction.
            TensorBacking::External { .. } => 0,
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
    create_tensor_with_data_in(data, bytes, shape, dtype, &memory_info)
}

/// Wrap `data` as a tensor living where `memory_info` says it does.
///
/// Split from [`create_tensor_with_data`] so a caller can hand ORT memory it
/// allocated itself. The memory info is what tells ORT whether the pointer is
/// host or device; getting it wrong makes ORT read device memory as host
/// addresses, so it is a parameter rather than an assumption.
fn create_tensor_with_data_in(
    data: *mut std::ffi::c_void,
    bytes: usize,
    shape: &[i64],
    dtype: DataType,
    memory_info: &MemoryInfo,
) -> Result<NonNull<onnx_genai_ort_sys::OrtValue>> {
    create_tensor_with_data_at(memory_info.as_ptr(), data, bytes, shape, dtype)
}

fn create_tensor_with_data_at(
    memory_info: *const onnx_genai_ort_sys::OrtMemoryInfo,
    data: *mut std::ffi::c_void,
    bytes: usize,
    shape: &[i64],
    dtype: DataType,
) -> Result<NonNull<onnx_genai_ort_sys::OrtValue>> {
    let mut ptr = std::ptr::null_mut();
    let api = crate::error::api()?;
    let create = api
        .CreateTensorWithDataAsOrtValue
        .ok_or(OrtError::ApiUnavailable("CreateTensorWithDataAsOrtValue"))?;
    // SAFETY: `data` is valid for `bytes` for at least the lifetime of the
    // OrtValue -- owned by `Value::backing` for the owning constructors, and by
    // the caller's contract for the external one. `shape` is valid for the call.
    crate::error::check_status(unsafe {
        create(
            memory_info,
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

fn tensor_memory_info(
    value: *const onnx_genai_ort_sys::OrtValue,
) -> Result<*const onnx_genai_ort_sys::OrtMemoryInfo> {
    let api = crate::error::api()?;
    let get_memory_info = api
        .GetTensorMemoryInfo
        .ok_or(OrtError::ApiUnavailable("GetTensorMemoryInfo"))?;
    let mut memory_info = std::ptr::null();
    // SAFETY: `value` is a valid tensor OrtValue and ORT owns the returned
    // memory-info object for the tensor's lifetime.
    crate::error::check_status(unsafe { get_memory_info(value, &mut memory_info) })?;
    if memory_info.is_null() {
        return Err(OrtError::NullPointer);
    }
    Ok(memory_info)
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
    // A zero-element tensor has nothing to copy, and ORT is entitled to hand
    // back any pointer for it (including a caller's dangling `Vec` sentinel).
    // Answer without asking for the data pointer at all.
    if len == 0 {
        return Ok(Vec::new());
    }
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

    // SAFETY: caller ensures `T` matches the tensor dtype and `len` is numel, so
    // `data` covers `len` contiguous `T`s that stay valid until the value is
    // released. `tensor_elements` handles the alignment precondition itself.
    Ok(unsafe { tensor_elements::<T>(data.cast::<T>(), len) }.into_owned())
}

/// Read-only view of `len` elements of `T` at a raw ORT tensor data pointer.
///
/// `slice::from_raw_parts` requires the pointer to be aligned for `T` **even
/// when `len` is 0**, and ORT hands back whatever pointer the tensor holds — it
/// does not re-align a buffer supplied through `CreateTensorWithDataAsOrtValue`.
/// This borrows in place on the fast path (every tensor this crate allocates,
/// and every ORT-allocated output, is element-aligned) and falls back to an
/// owned byte-wise copy when it is not, so a misaligned buffer is read
/// correctly instead of being turned into an unaligned slice.
///
/// # Safety
///
/// `ptr` must point to `len` contiguous, initialized `T`s that outlive the
/// returned borrow, and `T` must be a plain-old-data type for which every bit
/// pattern of `size_of::<T>()` bytes is a valid value.
unsafe fn tensor_elements<'a, T: Copy>(ptr: *const T, len: usize) -> Cow<'a, [T]> {
    if len == 0 {
        return Cow::Borrowed(&[]);
    }
    if ptr.is_aligned() {
        // SAFETY: aligned, non-null, and valid for `len` elements per the
        // function contract.
        return Cow::Borrowed(unsafe { std::slice::from_raw_parts(ptr, len) });
    }
    let mut out = Vec::<T>::with_capacity(len);
    // SAFETY: both sides are copied as bytes (alignment 1), the source is valid
    // for `len * size_of::<T>()` bytes per the contract, and `out` has exactly
    // that much freshly reserved capacity in a distinct allocation. `T` is POD,
    // so the copied bytes initialize `len` valid values.
    unsafe {
        std::ptr::copy_nonoverlapping(
            ptr.cast::<u8>(),
            out.as_mut_ptr().cast::<u8>(),
            std::mem::size_of::<T>() * len,
        );
        out.set_len(len);
    }
    Cow::Owned(out)
}

/// Pointer to this tensor's elements as `*mut T`, for **in-place** writes.
///
/// The mutating helpers cannot route around a misaligned buffer the way
/// [`tensor_elements`] does — writing through a staging copy would not update
/// the tensor — so misalignment is reported instead of being ignored. Our own
/// constructors guarantee alignment, so this can only fire for a buffer handed
/// in through [`Value::from_external_memory`].
fn tensor_elements_mut_ptr<T>(
    value: *mut onnx_genai_ort_sys::OrtValue,
    dtype: DataType,
) -> Result<*mut T> {
    let ptr = tensor_data_ptr(value)?.cast::<T>();
    if !ptr.is_aligned() {
        return Err(OrtError::InvalidArgument(format!(
            "tensor data pointer {ptr:p} is not aligned to {} bytes as {dtype:?} elements \
             require, so it cannot be written in place",
            std::mem::align_of::<T>()
        )));
    }
    Ok(ptr)
}

/// Index of the maximum value in `row`, ignoring NaNs.
///
/// Matches the engine greedy sampler exactly: a vectorizable horizontal max
/// followed by a first-match search (so ties resolve to the lowest index), and
/// an empty/all-NaN row yields 0. Both passes are branch-free per element, so
/// the compiler autovectorizes them over the ~150K-entry vocabulary.
fn argmax_row_f32(row: &[f32]) -> usize {
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if max == f32::NEG_INFINITY {
        return row.iter().position(|value| !value.is_nan()).unwrap_or(0);
    }
    row.iter().position(|&value| value == max).unwrap_or(0)
}

/// Number of half-precision elements widened per chunk in [`argmax_half_bits`].
///
/// Sized so the scratch buffer (`CHUNK * 4` bytes = 16 KiB) stays on the stack
/// and comfortably within L1, while remaining large enough that half's
/// F16C-accelerated slice conversion and the two f32 reductions per chunk still
/// autovectorize with negligible per-chunk overhead.
const ARGMAX_WIDEN_CHUNK: usize = 4096;

/// Argmax over half-precision bits (f16 or bf16), ignoring NaNs, matching
/// [`argmax_row_f32`] exactly (max wins, lowest index on ties, index 0 for
/// empty/all-NaN/all-`-inf` input).
///
/// Widening to f32 first is worth ~2x over a scalar branch-on-NaN bit-keying
/// loop (94us vs 200us over a 151,936-entry vocabulary on this box): both the
/// F16C widen and the f32 `max` fold autovectorize, whereas a loop-carried
/// max/index update does not. Rather than widen the whole row into one heap
/// `Vec<f32>` (a ~600 KiB allocation for a large vocab), this streams the row
/// through a fixed 16 KiB stack buffer `ARGMAX_WIDEN_CHUNK` elements at a time.
/// Each chunk still runs two vectorized passes (a `max` fold, then a `position`
/// scan only when that chunk beats the running best), so the SIMD win is kept
/// with zero heap allocation. Strict `>` comparison across chunks preserves the
/// lowest-index-wins tie-break.
fn argmax_half_bits<H>(halves: &[H]) -> usize
where
    [H]: half::slice::HalfFloatSliceExt,
{
    use half::slice::HalfFloatSliceExt;
    let mut scratch = [0f32; ARGMAX_WIDEN_CHUNK];
    let mut best_value = f32::NEG_INFINITY;
    let mut best_index = 0usize;
    let mut found = false;
    for (chunk_index, chunk) in halves.chunks(ARGMAX_WIDEN_CHUNK).enumerate() {
        let widened = &mut scratch[..chunk.len()];
        chunk.convert_to_f32_slice(widened);
        let chunk_max = widened.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        if let Some(offset) = widened.iter().position(|&value| value == chunk_max)
            && (!found || chunk_max > best_value)
        {
            best_value = chunk_max;
            best_index = chunk_index * ARGMAX_WIDEN_CHUNK + offset;
            found = true;
        }
    }
    if found { best_index } else { 0 }
}

/// Argmax over raw binary16 bits, ignoring NaNs, matching [`argmax_row_f32`].
fn argmax_f16_bits(bits: &[u16]) -> usize {
    let halves: &[half::f16] = half::slice::HalfBitsSliceExt::reinterpret_cast(bits);
    argmax_half_bits(halves)
}

/// Argmax over raw bfloat16 bits, ignoring NaNs, matching [`argmax_row_f32`].
fn argmax_bf16_bits(bits: &[u16]) -> usize {
    let halves: &[half::bf16] = half::slice::HalfBitsSliceExt::reinterpret_cast(bits);
    argmax_half_bits(halves)
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

#[cfg(test)]
mod host_residency_tests {
    use super::*;

    #[test]
    fn a_host_tensor_reports_host_residency_and_lends_its_bytes() {
        let value = Value::from_slice_f32(&[1.0, 2.0], &[2]).expect("build a host tensor");
        assert!(value.is_host_resident().expect("memory info is available"));
        assert_eq!(value.device_id().expect("device ID is available"), 0);
        assert_eq!(
            value.as_raw_bytes().expect("host bytes are borrowable"),
            &[1.0f32, 2.0]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<u8>>()[..],
            "the borrowed bytes must be the tensor's little-endian elements"
        );
    }

    #[test]
    fn borrowed_bytes_match_the_copying_accessor() {
        let value = Value::from_slice_i64(&[7, -3, 0], &[3]).expect("build a host tensor");
        assert_eq!(
            value.as_raw_bytes().expect("host bytes"),
            &value.to_raw_bytes().expect("copied bytes")[..]
        );
    }
}

#[cfg(test)]
mod clone_owned_tests {
    use super::*;

    /// `clone_owned` must deep-copy EVERY POD dtype, not just the typed fast
    /// paths (f32/f16/bf16/i64). Before generalization it returned
    /// `InvalidArgument("cannot clone tensor with dtype ...")` for Bool, Int32,
    /// Uint8, etc., which is exactly the blocker gemma-3n's Bool audio mask hit.
    /// Dtype + shape + raw bytes must round-trip identically, and the clone must
    /// own an independent buffer.
    fn assert_clone_owned_round_trips(bytes: Vec<u8>, shape: &[i64], dtype: DataType) {
        let original = Value::from_raw_bytes(bytes.clone(), shape, dtype)
            .unwrap_or_else(|e| panic!("build {dtype:?} tensor: {e}"));
        let cloned = original
            .clone_owned()
            .unwrap_or_else(|e| panic!("clone_owned {dtype:?}: {e}"));

        assert_eq!(cloned.dtype(), dtype, "{dtype:?}: dtype must round-trip");
        assert_eq!(cloned.shape(), shape, "{dtype:?}: shape must round-trip");
        assert_eq!(
            cloned.to_raw_bytes().expect("cloned bytes"),
            bytes,
            "{dtype:?}: raw bytes must round-trip identically"
        );
        // The clone owns a distinct buffer, not an alias over the original's.
        assert_ne!(
            original.as_ptr() as usize,
            cloned.as_ptr() as usize,
            "{dtype:?}: clone must be an independent OrtValue"
        );
    }

    #[test]
    fn clone_owned_round_trips_bool() {
        // 1 byte per Bool element; ORT stores false/true as 0/1.
        assert_clone_owned_round_trips(vec![1, 0, 1, 1], &[4], DataType::Bool);
    }

    #[test]
    fn clone_owned_round_trips_int32() {
        let bytes: Vec<u8> = [1i32, -2, 3, -4]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        assert_clone_owned_round_trips(bytes, &[2, 2], DataType::Int32);
    }

    #[test]
    fn clone_owned_round_trips_uint8() {
        assert_clone_owned_round_trips(vec![0, 255, 7, 128], &[2, 2], DataType::Uint8);
    }

    #[test]
    fn clone_owned_round_trips_empty_bool_tensor() {
        // An empty tensor is a legitimate zero-element window and must clone.
        assert_clone_owned_round_trips(Vec::new(), &[0], DataType::Bool);
    }

    #[test]
    fn clone_owned_round_trips_multidim_bool() {
        // 2x3 Bool mask, exercising a multi-dimensional shape.
        assert_clone_owned_round_trips(vec![1, 0, 1, 0, 1, 1], &[2, 3], DataType::Bool);
    }

    #[test]
    fn clone_owned_still_round_trips_the_typed_fast_path() {
        // The generalization must not regress the existing typed dtypes.
        let original = Value::from_slice_i64(&[7, -3, 0, 42], &[4]).expect("i64 tensor");
        let cloned = original.clone_owned().expect("clone i64");
        assert_eq!(cloned.dtype(), DataType::Int64);
        assert_eq!(cloned.to_vec_i64().expect("i64 out"), vec![7, -3, 0, 42]);
    }
}

#[cfg(test)]
mod argmax_tests {
    use super::{argmax_bf16_bits, argmax_f16_bits, argmax_row_f32};

    /// Reference argmax mirroring the engine greedy sampler: max ignoring NaN,
    /// lowest index on ties, index 0 for empty/all-NaN input.
    fn reference(values: &[f32]) -> usize {
        let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        if max == f32::NEG_INFINITY {
            return values.iter().position(|value| !value.is_nan()).unwrap_or(0);
        }
        values.iter().position(|&v| v == max).unwrap_or(0)
    }

    #[test]
    fn matches_reference_on_various_rows() {
        let rows: &[&[f32]] = &[
            &[],
            &[1.0, 3.0, 3.0],
            &[3.0, 1.0, 3.0],
            &[-1.0, -2.0, -0.5],
            &[f32::NEG_INFINITY, f32::NEG_INFINITY],
            &[f32::NAN, f32::NEG_INFINITY],
            &[0.0, f32::NAN, 2.0, f32::NAN, 2.0],
            &[f32::NAN, f32::NAN],
            &[f32::INFINITY, 5.0, f32::INFINITY],
            &[-0.0, 0.0],
        ];
        for row in rows {
            assert_eq!(argmax_row_f32(row), reference(row), "mismatch for {row:?}");
        }
    }

    #[test]
    fn ties_pick_lowest_index() {
        assert_eq!(argmax_row_f32(&[2.0, 2.0, 2.0]), 0);
        assert_eq!(argmax_row_f32(&[1.0, 2.0, 2.0]), 1);
    }

    #[test]
    fn ignores_nan_and_handles_all_nan() {
        assert_eq!(argmax_row_f32(&[f32::NAN, 1.0, f32::NAN]), 1);
        assert_eq!(argmax_row_f32(&[f32::NAN, f32::NAN]), 0);
        assert_eq!(argmax_row_f32(&[f32::NAN, f32::NEG_INFINITY]), 1);
        assert_eq!(argmax_row_f32(&[f32::NEG_INFINITY; 3]), 0);
        for values in [
            vec![f32::NAN, f32::NEG_INFINITY],
            vec![f32::NEG_INFINITY; 3],
        ] {
            let f16 = values
                .iter()
                .map(|&value| half::f16::from_f32(value).to_bits())
                .collect::<Vec<_>>();
            let bf16 = values
                .iter()
                .map(|&value| half::bf16::from_f32(value).to_bits())
                .collect::<Vec<_>>();
            assert_eq!(argmax_f16_bits(&f16), reference(&values));
            assert_eq!(argmax_bf16_bits(&bf16), reference(&values));
        }
    }

    /// Exercise the chunked, no-alloc half-precision argmax across chunk
    /// boundaries (row length > `ARGMAX_WIDEN_CHUNK`) with the winner in a
    /// later chunk, cross-chunk ties, NaNs, and the all-NaN fallback.
    #[test]
    fn half_argmax_matches_reference_across_chunks() {
        let len = super::ARGMAX_WIDEN_CHUNK * 2 + 123;
        let cases: &[(usize, f32)] = &[
            (0, 3.0),
            (super::ARGMAX_WIDEN_CHUNK - 1, 3.0),
            (super::ARGMAX_WIDEN_CHUNK, 3.0),
            (super::ARGMAX_WIDEN_CHUNK + 7, 3.0),
            (len - 1, 3.0),
        ];
        for &(peak, value) in cases {
            let mut f32_row = vec![1.0f32; len];
            f32_row[peak] = value;
            // A cross-chunk tie a few chunks later must NOT displace the
            // lowest-index winner.
            if peak + super::ARGMAX_WIDEN_CHUNK < len {
                f32_row[peak + super::ARGMAX_WIDEN_CHUNK] = value;
            }
            let f16_bits: Vec<u16> = f32_row
                .iter()
                .map(|&v| half::f16::from_f32(v).to_bits())
                .collect();
            let bf16_bits: Vec<u16> = f32_row
                .iter()
                .map(|&v| half::bf16::from_f32(v).to_bits())
                .collect();
            assert_eq!(
                argmax_f16_bits(&f16_bits),
                reference(&f32_row),
                "f16 peak {peak}"
            );
            assert_eq!(
                argmax_bf16_bits(&bf16_bits),
                reference(&f32_row),
                "bf16 peak {peak}"
            );
        }

        // NaNs are ignored; a single finite value in a later chunk wins.
        let mut nan_row = vec![f32::NAN; len];
        nan_row[super::ARGMAX_WIDEN_CHUNK + 5] = 1.0;
        let nan_bits: Vec<u16> = nan_row
            .iter()
            .map(|&v| half::f16::from_f32(v).to_bits())
            .collect();
        assert_eq!(argmax_f16_bits(&nan_bits), super::ARGMAX_WIDEN_CHUNK + 5);

        // All-NaN falls back to index 0.
        let all_nan: Vec<u16> = vec![half::f16::from_f32(f32::NAN).to_bits(); len];
        assert_eq!(argmax_f16_bits(&all_nan), 0);
    }

    // A cheap xorshift so the parity fuzz has no external dependency.
    fn next_rand(state: &mut u64) -> u16 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        (*state >> 16) as u16
    }

    #[test]
    fn f16_bits_argmax_matches_widened_reference_exhaustively() {
        // For every representable half value at a fixed row position, the f16
        // reducer must select exactly what widening to f32 and scanning would.
        let others: [u16; 5] = [
            0x0000, // +0
            0x3C00, // +1
            0xBC00, // -1
            0x7BFF, // max finite
            0xFBFF, // min finite
        ];
        for raw in 0u16..=u16::MAX {
            for &other in &others {
                let bits = [other, raw, other];
                let widened: Vec<f32> = bits
                    .iter()
                    .map(|&b| half::f16::from_bits(b).to_f32())
                    .collect();
                assert_eq!(
                    argmax_f16_bits(&bits),
                    reference(&widened),
                    "f16 mismatch raw={raw:#06x} other={other:#06x}",
                );
            }
        }
    }

    #[test]
    fn f16_bits_argmax_matches_reference_on_random_rows() {
        let mut state = 0x9E3779B97F4A7C15u64;
        for _ in 0..2000 {
            let len = 1 + (next_rand(&mut state) % 64) as usize;
            let bits: Vec<u16> = (0..len).map(|_| next_rand(&mut state)).collect();
            let widened: Vec<f32> = bits
                .iter()
                .map(|&b| half::f16::from_bits(b).to_f32())
                .collect();
            assert_eq!(
                argmax_f16_bits(&bits),
                reference(&widened),
                "f16 random mismatch bits={bits:#06x?}",
            );
        }
    }

    #[test]
    fn bf16_bits_argmax_matches_reference_on_random_rows() {
        let mut state = 0xD1B54A32D192ED03u64;
        for _ in 0..4000 {
            let len = 1 + (next_rand(&mut state) % 64) as usize;
            let bits: Vec<u16> = (0..len).map(|_| next_rand(&mut state)).collect();
            let widened: Vec<f32> = bits
                .iter()
                .map(|&b| half::bf16::from_bits(b).to_f32())
                .collect();
            assert_eq!(
                argmax_bf16_bits(&bits),
                reference(&widened),
                "bf16 random mismatch bits={bits:#06x?}",
            );
        }
    }

    #[test]
    fn f16_bits_argmax_handles_signed_zero_and_all_nan() {
        // -0.0 then +0.0 => both equal max, lowest index wins.
        assert_eq!(argmax_f16_bits(&[0x8000, 0x0000]), 0);
        // A NaN (0x7E00) beside a finite value must be skipped.
        assert_eq!(argmax_f16_bits(&[0x7E00, 0x3C00]), 1);
        // All-NaN => index 0.
        assert_eq!(argmax_f16_bits(&[0x7E00, 0xFE00]), 0);
    }
}

/// Hardware validation of the device-resident captured-decode input helpers
/// (`write_i64_prefix_device` / `fill_i64_range_device`) added for the
/// IoBinding + CUDA-graph fix (issue microsoft/onnxruntime#29782). These are the
/// primitives that update `input_ids` / `position_ids` / `attention_mask` in
/// place on the device each token so the captured graph observes them on replay
/// without a per-step clear + re-bind of the whole IoBinding set.
///
/// Requires a CUDA GPU and a CUDA-enabled ONNX Runtime, so the tests are
/// `#[ignore]`d by default. Run from the repo root (PowerShell):
///
/// ```text
/// $ort = "ort-gpu\onnxruntime-win-x64-gpu_cuda12-1.28.0"
/// $env:ORT_ROOT = (Resolve-Path $ort)
/// $env:PATH = "$((Resolve-Path "$ort\lib"));$env:PATH"
/// cargo test -p onnx-genai-ort --features cuda --lib -- --ignored --nocapture cuda_device_write
/// ```
#[cfg(all(test, feature = "cuda"))]
mod cuda_device_write_tests {
    use super::{DataType, Value};
    use crate::{Allocator, Environment, Session, SessionOptions, ep_selection};
    use std::path::Path;

    const TINY_LLM: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/tiny-llm-sharedbuffer/model.onnx"
    );

    /// Build a CUDA session and return its device KV allocator together with the
    /// EP-bound device id and the owning [`Environment`], or `None` when CUDA is
    /// unavailable so the caller can skip gracefully.
    ///
    /// The `Environment` is returned (not dropped here) on purpose: ORT requires
    /// the `OrtEnv` to outlive every `OrtSession`. Releasing the session after
    /// its environment is a use-after-free that crashes with STATUS_ACCESS_VIOLATION
    /// at `ReleaseSession`. Keeping the env in the returned tuple lets the caller
    /// release everything in the correct order (value -> allocator -> session ->
    /// environment).
    fn cuda_device_allocator() -> Option<(Environment, Session, Allocator, i32)> {
        let env = Environment::new("cuda-device-write-test").expect("env");
        let options =
            SessionOptions::with_execution_provider(ep_selection("cuda")).with_intra_op_threads(1);
        let session = match Session::new(&env, Path::new(TINY_LLM), options) {
            Ok(session) => session,
            Err(error) => {
                eprintln!("cuda session build failed: {error}");
                return None;
            }
        };
        let Some(device_id) = session.cuda_device_id() else {
            eprintln!("cuda_device_id() is None (CUDA EP not attached?)");
            return None;
        };
        match session.device_kv_allocator() {
            Ok(Some(allocator)) => Some((env, session, allocator, device_id)),
            Ok(None) => {
                eprintln!(
                    "device_kv_allocator() returned None (CUDA runtime/cuDNN not on the search path?)"
                );
                None
            }
            Err(error) => {
                eprintln!("device_kv_allocator() failed: {error}");
                None
            }
        }
    }

    /// Read back the full `count`-element Int64 device tensor into a host `Vec`.
    fn read_device_i64(value: &Value, count: usize) -> Vec<i64> {
        let source = value.data_ptr_addr().expect("device pointer");
        let mut bytes = vec![0u8; count * std::mem::size_of::<i64>()];
        crate::cuda_rt::memcpy_device_to_host(&mut bytes, source).expect("device-to-host copy");
        bytes
            .chunks_exact(std::mem::size_of::<i64>())
            .map(|chunk| i64::from_ne_bytes(chunk.try_into().expect("i64 chunk")))
            .collect()
    }

    /// Release the CUDA resources in ORT's required ownership order: the device
    /// `Value` (frees memory through the allocator), then the `Allocator`, then
    /// the `Session`, and finally the `Environment`. ORT mandates that the
    /// `OrtEnv` outlive every `OrtSession`; releasing them out of order (e.g.
    /// dropping the environment first) is a use-after-free that crashes with
    /// STATUS_ACCESS_VIOLATION. Dropping in this order releases cleanly.
    fn release_cuda_resources_in_order(
        tensor: Value,
        allocator: Allocator,
        session: Session,
        env: Environment,
    ) {
        drop(tensor);
        drop(allocator);
        drop(session);
        drop(env);
    }

    #[test]
    #[ignore = "requires a CUDA GPU + CUDA-enabled ONNX Runtime"]
    fn cuda_device_write_prefix_round_trips_through_device_memory() {
        let Some((env, session, allocator, device_id)) = cuda_device_allocator() else {
            eprintln!("skipping: no CUDA device allocator available");
            return;
        };

        // A prefix write must land exactly, leaving the untouched tail alone.
        let capacity = 6;
        let tensor = Value::empty_in(&[1, capacity as i64], DataType::Int64, &allocator)
            .expect("device tensor");
        tensor
            .fill_i64_range_device(0, capacity, -1, device_id)
            .expect("initialize tail");
        let prefix = [7_i64, 11, 13, 17];
        tensor
            .write_i64_prefix_device(&prefix, device_id)
            .expect("prefix write");

        let read_back = read_device_i64(&tensor, capacity);
        assert_eq!(&read_back[..prefix.len()], &prefix);
        assert_eq!(&read_back[prefix.len()..], &[-1_i64, -1]);

        release_cuda_resources_in_order(tensor, allocator, session, env);
    }

    #[test]
    #[ignore = "requires a CUDA GPU + CUDA-enabled ONNX Runtime"]
    fn cuda_alias_preserves_device_residency_without_copying() {
        let Some((env, session, allocator, device_id)) = cuda_device_allocator() else {
            return;
        };
        let owner = Value::empty_in(&[4], DataType::Int64, &allocator).expect("device allocation");
        assert!(!owner.is_host_resident().expect("device residency"));
        let host = Value::from_slice_i64(&[9, 8, 7, 6], &[4]).expect("host tensor");
        let error = owner
            .copy_from_host(&host)
            .expect_err("CUDA allocation must not be host-accessible");
        assert!(
            error.to_string().contains("lives on a device"),
            "unexpected error: {error}"
        );
        let bytes = [1_i64, 2, 3, 4]
            .into_iter()
            .flat_map(i64::to_ne_bytes)
            .collect::<Vec<_>>();
        crate::cuda_rt::memcpy_host_to_device(owner.data_ptr_addr().unwrap(), &bytes)
            .expect("host-to-device copy");
        let owner_ptr = owner.data_ptr_addr().unwrap();
        let alias = Value::into_alias_with_shape(owner, &[2]).expect("device alias");
        let alias_clone = alias
            .try_alias_clone()
            .expect("alias backing")
            .expect("alias clone");
        assert_eq!(alias.device_id().expect("alias device"), device_id);
        assert_eq!(alias.data_ptr_addr().unwrap(), owner_ptr);
        assert_eq!(alias_clone.data_ptr_addr().unwrap(), owner_ptr);
        assert_eq!(read_device_i64(&alias, 2), vec![1, 2]);
        drop(alias_clone);
        drop(alias);
        drop(allocator);
        drop(session);
        drop(env);
    }

    #[test]
    #[ignore = "requires a CUDA GPU + CUDA-enabled ONNX Runtime"]
    fn cuda_device_fill_range_round_trips_through_device_memory() {
        let Some((env, session, allocator, device_id)) = cuda_device_allocator() else {
            eprintln!("skipping: no CUDA device allocator available");
            return;
        };

        // Zero the whole mask, then mark a valid region — the captured attention
        // mask update pattern (leading ones, zeroed tail).
        let capacity = 8;
        let mask = Value::empty_in(&[1, capacity as i64], DataType::Int64, &allocator)
            .expect("device mask");
        mask.fill_i64_range_device(0, capacity, 0, device_id)
            .expect("zero mask");
        mask.fill_i64_range_device(0, 5, 1, device_id)
            .expect("mark valid");

        let read_back = read_device_i64(&mask, capacity);
        assert_eq!(read_back, vec![1, 1, 1, 1, 1, 0, 0, 0]);

        release_cuda_resources_in_order(mask, allocator, session, env);
    }
}

#[cfg(test)]
mod external_memory_tests {
    use super::*;

    /// ORT must read the caller's buffer, not a copy of it.
    ///
    /// Constructing successfully proves nothing: a constructor that quietly
    /// copied would pass any test that only checks the value comes back. This
    /// mutates the source buffer *after* the tensor exists and requires the
    /// change to be visible through ORT, which is only true if nothing copied.
    #[test]
    fn ort_reads_through_to_the_callers_buffer_rather_than_a_copy() {
        let mut buffer = vec![1.0f32, 2.0, 3.0, 4.0];
        let info = MemoryInfo::cpu().expect("cpu memory info");
        let value = unsafe {
            Value::from_external_memory(
                buffer.as_mut_ptr().cast(),
                std::mem::size_of_val(&buffer[..]),
                &[2, 2],
                DataType::Float32,
                &info,
            )
        }
        .expect("wrapping a host buffer");

        buffer[0] = 99.0;
        let seen = value.to_vec_f32().expect("reading back the tensor");
        assert_eq!(
            seen[0], 99.0,
            "the tensor did not observe a write to the caller's buffer, so it copied"
        );
    }

    /// A buffer too small for the shape is refused before ORT can read past it.
    #[test]
    fn a_buffer_too_small_for_its_shape_is_refused() {
        let mut buffer = vec![0.0f32; 3];
        let info = MemoryInfo::cpu().expect("cpu memory info");
        let error = unsafe {
            Value::from_external_memory(
                buffer.as_mut_ptr().cast(),
                std::mem::size_of_val(&buffer[..]),
                &[2, 2],
                DataType::Float32,
                &info,
            )
        }
        .map(|_| ())
        .expect_err("3 floats cannot hold a 2x2 tensor");
        let message = error.to_string();
        assert!(
            message.contains("too small") && message.contains("16"),
            "the error should name the size actually needed: {message}"
        );
    }

    /// A null pointer is refused rather than handed to ORT.
    #[test]
    fn a_null_pointer_is_refused() {
        let info = MemoryInfo::cpu().expect("cpu memory info");
        let error = unsafe {
            Value::from_external_memory(std::ptr::null_mut(), 16, &[2, 2], DataType::Float32, &info)
        }
        .map(|_| ())
        .expect_err("null is not a buffer");
        assert!(error.to_string().contains("null pointer"), "{error}");
    }

    /// Host-side mutation helpers refuse externally owned tensors.
    ///
    /// They write through a host pointer, and an external buffer may be device
    /// memory — where that write would be a wild store rather than an error.
    #[test]
    fn host_side_mutation_refuses_an_externally_owned_tensor() {
        let mut buffer = vec![0.0f32; 8];
        let info = MemoryInfo::cpu().expect("cpu memory info");
        let mut value = unsafe {
            Value::from_external_memory(
                buffer.as_mut_ptr().cast(),
                std::mem::size_of_val(&buffer[..]),
                &[2, 2, 2],
                DataType::Float32,
                &info,
            )
        }
        .expect("wrapping a host buffer");

        let error = value
            .zero_rank3_row(0)
            .expect_err("zeroing an external tensor from the host must be refused");
        assert!(
            error.to_string().contains("externally owned"),
            "the refusal should say why: {error}"
        );
    }

    /// A larger buffer than the shape needs is allowed.
    ///
    /// Callers sub-allocate from a pool, so an exact fit is the exception.
    #[test]
    fn a_buffer_larger_than_the_shape_is_accepted() {
        let mut buffer = vec![7.0f32; 64];
        let info = MemoryInfo::cpu().expect("cpu memory info");
        let value = unsafe {
            Value::from_external_memory(
                buffer.as_mut_ptr().cast(),
                std::mem::size_of_val(&buffer[..]),
                &[2, 2],
                DataType::Float32,
                &info,
            )
        }
        .expect("a pool sub-allocation is normally larger than the tensor");
        assert_eq!(value.to_vec_f32().expect("read back")[0], 7.0);
    }
    /// A tensor whose memory was declared to live on a device must refuse every
    /// host accessor.
    ///
    /// These all reach the bytes through `GetTensorMutableData`, which returns
    /// the tensor's own address with no indication of where it lives. Without
    /// this check the failure is a wild read or store, not an error -- and it
    /// is reachable from entirely safe code once the unsafe constructor has
    /// been called correctly.
    #[test]
    fn host_accessors_refuse_a_tensor_declared_to_live_on_a_device() {
        let Ok(device_info) = MemoryInfo::dml(0) else {
            return; // no device memory info available here
        };
        // Host memory, deliberately mislabelled as device memory: this test is
        // about the bookkeeping, and a real device pointer is not needed to
        // prove the accessors consult it.
        let mut backing = vec![0i64; 4];
        let value = unsafe {
            Value::from_external_memory(
                backing.as_mut_ptr().cast(),
                std::mem::size_of_val(backing.as_slice()),
                &[4],
                DataType::Int64,
                &device_info,
            )
        }
        .expect("wrapping external device memory");

        assert!(!value.is_host_accessible());
        for (name, result) in [
            ("to_raw_bytes", value.to_raw_bytes().map(|_| ())),
            ("to_vec_i64", value.to_vec_i64().map(|_| ())),
            ("write_i64_prefix", value.write_i64_prefix(&[1, 2])),
            ("fill_i64_range", value.fill_i64_range(0, 2, 7)),
        ] {
            let error = result.expect_err("host access to device memory must be refused");
            assert!(
                error.to_string().contains(name),
                "the error must name the operation that was refused, got: {error}"
            );
        }
    }

    /// The same accessors must keep working when the external memory really is
    /// on the host, or the check above would just be breaking the feature.
    #[test]
    fn host_accessors_still_work_for_external_host_memory() {
        let info = MemoryInfo::cpu().expect("cpu memory info");
        let mut backing = vec![5i64, 6, 7, 8];
        let value = unsafe {
            Value::from_external_memory(
                backing.as_mut_ptr().cast(),
                std::mem::size_of_val(backing.as_slice()),
                &[4],
                DataType::Int64,
                &info,
            )
        }
        .expect("wrapping external host memory");

        assert!(value.is_host_accessible());
        assert_eq!(value.to_vec_i64().expect("read back"), vec![5, 6, 7, 8]);
    }
}

/// Regression coverage for the unaligned / zero-length tensor data pointer that
/// aborted `CLI ORT` on `main`.
///
/// `Value::from_raw_bytes` handed ORT the pointer of a `Vec<u8>`, which is only
/// 1-byte aligned and, when empty, is `Vec`'s dangling sentinel `0x1` rather
/// than an allocation. ORT stores that pointer verbatim and returns it from
/// `GetTensorMutableData`, so reading the tensor back as its element type built
/// a slice from a pointer not aligned for `T`. `slice::from_raw_parts` requires
/// alignment **even at length 0**, so the debug UB check aborted the process
/// (SIGABRT on Linux, `STATUS_STACK_BUFFER_OVERRUN` on Windows).
///
/// These are deterministic on every platform: the dangling-`Vec` sentinel, the
/// `from_raw_parts` precondition, and the synthetic misaligned pointers below
/// do not depend on the host allocator, the OS, or test ordering.
#[cfg(test)]
mod element_alignment_tests {
    use super::*;

    /// Every dtype, with the accessor a caller would actually reach for.
    fn read_back_empty(dtype: DataType) -> usize {
        let value = Value::from_raw_bytes(Vec::new(), &[0, 8], dtype)
            .unwrap_or_else(|e| panic!("build an empty {dtype:?} tensor: {e}"));
        assert_eq!(value.numel(), 0, "{dtype:?}");
        match dtype {
            DataType::Float32 => value.to_vec_f32().expect("f32 read").len(),
            DataType::Float16 => value.to_vec_f16_bits().expect("f16 read").len(),
            DataType::BFloat16 => value.to_vec_bf16_bits().expect("bf16 read").len(),
            DataType::Int64 => value.to_vec_i64().expect("i64 read").len(),
            _ => value.to_raw_bytes().expect("raw read").len(),
        }
    }

    /// The exact abort: an empty tensor of a multi-byte dtype, read back as its
    /// element type. Every one of these panicked the process before the fix; the
    /// 1-byte dtypes are included so the case that *did* work stays working.
    #[test]
    fn an_empty_tensor_reads_back_empty_for_every_dtype() {
        for dtype in [
            DataType::Float32,
            DataType::Float16,
            DataType::BFloat16,
            DataType::Int64,
            DataType::Int32,
            DataType::Uint32,
            DataType::Int16,
            DataType::Uint16,
            DataType::Int8,
            DataType::Uint8,
            DataType::Uint64,
            DataType::Bool,
            DataType::Float8E4M3,
            DataType::Float8E5M2,
        ] {
            assert_eq!(read_back_empty(dtype), 0, "{dtype:?} must read back empty");
        }
    }

    /// The invariant itself, asserted on the address ORT hands back rather than
    /// on a symptom: a `Vec<u8>`-backed tensor must still expose an
    /// element-aligned data pointer, empty or not.
    #[test]
    fn a_raw_bytes_tensor_exposes_an_element_aligned_data_pointer() {
        for dtype in [
            DataType::Float32,
            DataType::Float16,
            DataType::BFloat16,
            DataType::Int64,
            DataType::Int32,
            DataType::Bool,
        ] {
            for rows in [0i64, 1, 3] {
                let bytes = vec![0u8; (rows as usize) * 8 * dtype.size_of()];
                let value = Value::from_raw_bytes(bytes, &[rows, 8], dtype)
                    .unwrap_or_else(|e| panic!("build {dtype:?} [{rows}, 8]: {e}"));
                let address = value.data_ptr_addr().expect("data pointer");
                assert!(
                    address.is_multiple_of(dtype.align_of()),
                    "{dtype:?} [{rows}, 8]: ORT returned {address:#x}, which is not aligned to \
                     {} bytes",
                    dtype.align_of()
                );
                assert!(
                    address > 0xffff,
                    "{dtype:?} [{rows}, 8]: {address:#x} is a dangling sentinel, not an allocation"
                );
            }
        }
    }

    /// Non-empty bytes must survive the aligned backing unchanged — the fix must
    /// not silently truncate, pad, or reorder the tail that does not fill a
    /// whole 8-byte word.
    #[test]
    fn raw_bytes_round_trip_through_the_aligned_backing() {
        // 5 x u16 = 10 bytes: not a multiple of the 8-byte backing word, so the
        // partial tail word is exercised.
        let bits: Vec<u16> = vec![0x3C00, 0x0001, 0xFFFF, 0x8000, 0x1234];
        let bytes: Vec<u8> = bits.iter().flat_map(|v| v.to_le_bytes()).collect();
        let value = Value::from_raw_bytes(bytes.clone(), &[5], DataType::Float16)
            .expect("build a 5-element fp16 tensor");
        assert_eq!(value.to_raw_bytes().expect("raw bytes"), bytes);
        assert_eq!(value.to_vec_f16_bits().expect("f16 bits"), bits);
    }

    /// In-place writers must still work on a `Vec<u8>`-backed Int64 tensor,
    /// whose backing had alignment 1 before the fix — `copy_nonoverlapping` and
    /// `ptr::write` impose the same alignment precondition as `from_raw_parts`.
    #[test]
    fn in_place_i64_writes_work_on_a_raw_bytes_backed_tensor() {
        let value = Value::from_raw_bytes(vec![0u8; 4 * 8], &[4], DataType::Int64)
            .expect("build an i64 tensor from raw bytes");
        value.write_i64_prefix(&[11, 22]).expect("prefix write");
        value.fill_i64_range(2, 2, -7).expect("range fill");
        assert_eq!(
            value.to_vec_i64().expect("read back"),
            vec![11, 22, -7, -7],
            "in-place writes through a raw-bytes backing must land"
        );
    }

    /// A deliberately misaligned pointer, built by offsetting an 8-aligned
    /// allocation by one byte so the misalignment is guaranteed on every
    /// platform rather than left to the allocator.
    #[test]
    fn tensor_elements_copies_rather_than_borrowing_a_misaligned_pointer() {
        let mut storage = [0u64; 4];
        let base = storage.as_mut_ptr().cast::<u8>();
        assert!(
            (base as usize).is_multiple_of(8),
            "u64 storage is 8-aligned"
        );
        // SAFETY: `base + 1 .. base + 13` is inside the 32-byte allocation.
        let misaligned = unsafe { base.add(1) };
        let expected: [u32; 3] = [0xDEAD_BEEF, 0x0000_0001, 0xFFFF_FFFF];
        // SAFETY: writing 12 bytes at offset 1 of a 32-byte allocation, as bytes
        // (alignment 1), from a distinct source.
        unsafe {
            std::ptr::copy_nonoverlapping(
                expected.as_ptr().cast::<u8>(),
                misaligned,
                std::mem::size_of_val(&expected),
            );
        }

        // SAFETY: 3 initialized `u32`s worth of bytes live at `misaligned`, and
        // `storage` outlives the returned view.
        let view = unsafe { tensor_elements::<u32>(misaligned.cast::<u32>(), 3) };
        assert!(
            matches!(view, Cow::Owned(_)),
            "a misaligned pointer must be copied, never borrowed as a slice"
        );
        assert_eq!(
            &view[..],
            &expected[..],
            "the copy must preserve the values"
        );
    }

    /// The fast path must stay a borrow: a copy on every aligned read would be a
    /// silent per-token regression on the logits path.
    #[test]
    fn tensor_elements_borrows_an_aligned_pointer_in_place() {
        let storage = [1u32, 2, 3, 4];
        // SAFETY: `storage` is a live, aligned `[u32; 4]` that outlives the view.
        let view = unsafe { tensor_elements::<u32>(storage.as_ptr(), 4) };
        assert!(
            matches!(view, Cow::Borrowed(_)),
            "an aligned pointer must be borrowed in place, not copied"
        );
        assert_eq!(&view[..], &storage[..]);
    }

    /// Length 0 must never dereference or even align-check the pointer: this is
    /// the precondition the abort actually tripped.
    #[test]
    fn tensor_elements_yields_an_empty_slice_for_a_misaligned_zero_length_read() {
        // `0x1` is exactly what `Vec::<u8>::new().as_mut_ptr()` produces and what
        // ORT handed back in the failing job.
        let dangling = std::ptr::without_provenance::<u16>(1);
        // SAFETY: `len` is 0, so no element is read and the pointer is never
        // dereferenced.
        let view = unsafe { tensor_elements::<u16>(dangling, 0) };
        assert!(view.is_empty());
        assert!(matches!(view, Cow::Borrowed(_)));
    }

    /// An external buffer is the only one this crate does not allocate, so it is
    /// the only remaining way to violate the invariant. It must be refused at
    /// the boundary, loudly, instead of becoming UB on the first read.
    #[test]
    fn an_unaligned_external_buffer_is_refused_with_a_named_alignment() {
        let mut storage = [0u64; 4];
        let base = storage.as_mut_ptr().cast::<u8>();
        // SAFETY: one byte into a 32-byte allocation.
        let misaligned = unsafe { base.add(1) };
        let info = MemoryInfo::cpu().expect("cpu memory info");
        let error = unsafe {
            Value::from_external_memory(misaligned.cast(), 16, &[2, 2], DataType::Float32, &info)
        }
        .map(|_| ())
        .expect_err("a 1-byte-offset buffer is not aligned for f32");
        let message = error.to_string();
        assert!(
            message.contains("not aligned") && message.contains('4'),
            "the error must name the alignment f32 requires: {message}"
        );
    }

    /// A zero-element external buffer is accepted whatever its pointer, because
    /// no reader ever dereferences it — and the obvious way to spell "empty" in
    /// Rust, `Vec::<u8>::new()`, yields the misaligned dangling `0x1`.
    ///
    /// This is not hypothetical: `governed_allocator_session`'s real-session
    /// test feeds exactly this when a model input has zero elements.
    #[test]
    fn a_zero_length_external_buffer_is_accepted_despite_a_dangling_pointer() {
        let mut empty: Vec<u8> = Vec::new();
        assert_eq!(empty.as_mut_ptr() as usize, 1, "Vec's dangling sentinel");
        let info = MemoryInfo::cpu().expect("cpu memory info");
        let value = unsafe {
            Value::from_external_memory(
                empty.as_mut_ptr().cast(),
                0,
                &[0, 8],
                DataType::Float32,
                &info,
            )
        }
        .expect("a zero-element tensor has nothing to misalign");
        assert_eq!(value.numel(), 0);
        assert!(value.to_vec_f32().expect("read back").is_empty());
    }

    /// The rejection must be about alignment, not about external buffers: a
    /// correctly aligned one still works.
    #[test]
    fn an_aligned_external_buffer_is_still_accepted() {
        let mut storage = [0u64; 4];
        let info = MemoryInfo::cpu().expect("cpu memory info");
        let value = unsafe {
            Value::from_external_memory(
                storage.as_mut_ptr().cast(),
                std::mem::size_of_val(&storage),
                &[2, 2],
                DataType::Float32,
                &info,
            )
        }
        .expect("an 8-aligned buffer is aligned for f32");
        assert_eq!(value.to_vec_f32().expect("read back"), vec![0.0; 4]);
    }

    /// `clone_owned` deep-copies through the same accessors, so an empty tensor
    /// of a multi-byte dtype used to abort there too.
    #[test]
    fn clone_owned_round_trips_an_empty_float16_tensor() {
        let value =
            Value::from_raw_bytes(Vec::new(), &[0, 8], DataType::Float16).expect("empty fp16");
        let cloned = value.clone_owned().expect("clone an empty fp16 tensor");
        assert_eq!(cloned.dtype(), DataType::Float16);
        assert_eq!(cloned.shape(), &[0, 8]);
        assert!(cloned.to_vec_f16_bits().expect("read back").is_empty());
    }

    /// `to_vec_f32_lossy` widens fp16 through its own `from_raw_parts` site, so
    /// it needs the same zero-length rule as `tensor_data_to_vec`.
    #[test]
    fn lossy_f32_widening_handles_an_empty_half_tensor() {
        for dtype in [DataType::Float16, DataType::BFloat16, DataType::Float32] {
            let value = Value::from_raw_bytes(Vec::new(), &[0, 8], dtype)
                .unwrap_or_else(|e| panic!("empty {dtype:?}: {e}"));
            assert!(
                value.to_vec_f32_lossy().expect("widen").is_empty(),
                "{dtype:?} must widen to an empty vec"
            );
        }
    }

    /// The borrowing byte accessor must answer an empty tensor without asking
    /// ORT for a data pointer at all.
    #[test]
    fn byte_accessors_answer_an_empty_tensor_without_a_data_pointer() {
        let value =
            Value::from_raw_bytes(Vec::new(), &[0, 8], DataType::Float16).expect("empty fp16");
        assert!(value.as_raw_bytes().expect("borrowed bytes").is_empty());
        assert!(value.to_raw_bytes().expect("copied bytes").is_empty());
    }
}
