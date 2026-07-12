//! ORT Allocator and MemoryInfo.

use std::ffi::CString;
use std::ptr::NonNull;

use crate::{OrtError, Result};

/// Memory type (CPU or device).
#[derive(Debug, Clone, Copy)]
pub enum MemoryType {
    /// CPU memory accessible by the device.
    CpuInput,
    /// CPU memory for outputs.
    CpuOutput,
    /// Default device memory (GPU HBM, NPU memory, etc.).
    Default,
}

/// Allocator type.
#[derive(Debug, Clone, Copy)]
pub enum AllocatorType {
    /// Device allocator (e.g., CUDA allocator).
    Device,
    /// Arena allocator (pooled).
    Arena,
}

/// Describes where memory lives.
pub struct MemoryInfo {
    ptr: NonNull<onnx_genai_ort_sys::OrtMemoryInfo>,
    pub device_name: String,
    pub device_id: i32,
    pub memory_type: MemoryType,
}

impl MemoryInfo {
    /// Create CPU memory info.
    pub fn cpu() -> Result<Self> {
        let mut ptr = std::ptr::null_mut();
        let api = crate::error::api()?;
        let create = api
            .CreateCpuMemoryInfo
            .ok_or(OrtError::ApiUnavailable("CreateCpuMemoryInfo"))?;
        // SAFETY: `ptr` is a valid out-parameter. ORT initializes it on success;
        // this wrapper owns it and releases it in Drop.
        crate::error::check_status(unsafe {
            create(
                onnx_genai_ort_sys::OrtArenaAllocator,
                onnx_genai_ort_sys::OrtMemTypeDefault,
                &mut ptr,
            )
        })?;
        Ok(Self {
            ptr: NonNull::new(ptr).ok_or(OrtError::NullPointer)?,
            device_name: "Cpu".to_string(),
            device_id: 0,
            memory_type: MemoryType::Default,
        })
    }

    /// Create CUDA device memory info.
    pub fn cuda(device_id: i32) -> Result<Self> {
        Self::new_device("Cuda", device_id, MemoryType::Default)
    }

    /// Create DirectML device memory info.
    pub fn dml(device_id: i32) -> Result<Self> {
        Self::new_device("DML", device_id, MemoryType::Default)
    }

    fn new_device(device_name: &str, device_id: i32, memory_type: MemoryType) -> Result<Self> {
        let name = CString::new(device_name)
            .map_err(|_| OrtError::InvalidArgument("memory device name contains NUL".into()))?;
        let mut ptr = std::ptr::null_mut();
        let api = crate::error::api()?;
        let create = api
            .CreateMemoryInfo
            .ok_or(OrtError::ApiUnavailable("CreateMemoryInfo"))?;
        let mem_type = match memory_type {
            MemoryType::CpuInput => onnx_genai_ort_sys::OrtMemTypeCPUInput,
            MemoryType::CpuOutput => onnx_genai_ort_sys::OrtMemTypeCPUOutput,
            MemoryType::Default => onnx_genai_ort_sys::OrtMemTypeDefault,
        };
        // SAFETY: `name` is NUL-terminated and lives for the call; `ptr` is a
        // valid out-parameter. The returned pointer is owned by this wrapper.
        crate::error::check_status(unsafe {
            create(
                name.as_ptr(),
                onnx_genai_ort_sys::OrtDeviceAllocator,
                device_id,
                mem_type,
                &mut ptr,
            )
        })?;
        Ok(Self {
            ptr: NonNull::new(ptr).ok_or(OrtError::NullPointer)?,
            device_name: device_name.to_string(),
            device_id,
            memory_type,
        })
    }

    pub(crate) fn as_ptr(&self) -> *const onnx_genai_ort_sys::OrtMemoryInfo {
        self.ptr.as_ptr()
    }
}

impl Drop for MemoryInfo {
    fn drop(&mut self) {
        if let Ok(api) = crate::error::api()
            && let Some(release) = api.ReleaseMemoryInfo
        {
            // SAFETY: `ptr` is owned by this wrapper and released exactly once here.
            unsafe { release(self.ptr.as_ptr()) };
        }
    }
}

/// ORT Allocator.
pub struct Allocator {
    ptr: *mut onnx_genai_ort_sys::OrtAllocator,
    pub memory_info: MemoryInfo,
    owned: bool,
}

impl Allocator {
    /// Get the default CPU allocator.
    pub fn default_cpu() -> Result<Self> {
        let mut ptr = std::ptr::null_mut();
        let api = crate::error::api()?;
        let get = api
            .GetAllocatorWithDefaultOptions
            .ok_or(OrtError::ApiUnavailable("GetAllocatorWithDefaultOptions"))?;
        // SAFETY: `ptr` is a valid out-parameter. ORT returns a process-owned
        // default allocator that must not be released by this wrapper.
        crate::error::check_status(unsafe { get(&mut ptr) })?;
        Ok(Self {
            ptr,
            memory_info: MemoryInfo::cpu()?,
            owned: false,
        })
    }

    pub(crate) fn as_ptr(&self) -> *mut onnx_genai_ort_sys::OrtAllocator {
        self.ptr
    }
}

impl Drop for Allocator {
    fn drop(&mut self) {
        if self.owned
            && !self.ptr.is_null()
            && let Ok(api) = crate::error::api()
            && let Some(release) = api.ReleaseAllocator
        {
            // SAFETY: Owned allocators come from ORT CreateAllocator and are
            // released exactly once here. The default allocator is never owned.
            unsafe { release(self.ptr) };
        }
    }
}
