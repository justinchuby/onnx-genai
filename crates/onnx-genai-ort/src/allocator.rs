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
    #[cfg(feature = "cuda")]
    pub fn cuda(device_id: i32) -> Result<Self> {
        Self::new_device("Cuda", device_id, MemoryType::Default)
    }

    /// Create DirectML device memory info.
    pub fn dml(device_id: i32) -> Result<Self> {
        Self::new_device("DML", device_id, MemoryType::Default)
    }

    /// Create WebGPU device memory info matching the ORT WebGPU EP allocator.
    ///
    /// The WebGPU EP registers its `GpuBufferAllocator` with an `OrtMemoryInfo`
    /// named `WebGPU_Buffer` on a `GPU`/`DEFAULT`/`VendorIds::NONE` device (see
    /// ORT `core/providers/webgpu/allocator.{h,cc}`). The legacy
    /// `CreateCpuMemoryInfo`/`CreateMemoryInfo` C API only recognizes a fixed set
    /// of device names and rejects `WebGPU_Buffer` ("Specified device is not
    /// supported. Try CreateMemoryInfo_V2."), so we use `CreateMemoryInfo_V2`
    /// with an explicit device type to construct a matching handle.
    pub fn webgpu() -> Result<Self> {
        const WEBGPU_BUFFER: &str = "WebGPU_Buffer";
        let name = CString::new(WEBGPU_BUFFER)
            .map_err(|_| OrtError::InvalidArgument("memory device name contains NUL".into()))?;
        let mut ptr = std::ptr::null_mut();
        let api = crate::error::api()?;
        let create = api
            .CreateMemoryInfo_V2
            .ok_or(OrtError::ApiUnavailable("CreateMemoryInfo_V2"))?;
        // SAFETY: `name` is NUL-terminated and lives for the call; `ptr` is a
        // valid out-parameter. Parameters mirror the WebGPU EP's own
        // `GpuBufferAllocator` OrtMemoryInfo so `CreateAllocator` can match it:
        // device type GPU, vendor id NONE (0), device id 0, default memory type,
        // default alignment (0), device allocator.
        crate::error::check_status(unsafe {
            create(
                name.as_ptr(),
                onnx_genai_ort_sys::OrtMemoryInfoDeviceType_GPU,
                0,
                0,
                onnx_genai_ort_sys::OrtDeviceMemoryType_DEFAULT,
                0,
                onnx_genai_ort_sys::OrtDeviceAllocator,
                &mut ptr,
            )
        })?;
        Ok(Self {
            ptr: NonNull::new(ptr).ok_or(OrtError::NullPointer)?,
            device_name: WEBGPU_BUFFER.to_string(),
            device_id: 0,
            memory_type: MemoryType::Default,
        })
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

    /// Create an allocator for `session` that allocates on the device described
    /// by `memory_info` (e.g. the WebGPU EP's `WebGPU_Buffer` device).
    ///
    /// This wraps the session's internal EP allocator, so tensors created with
    /// it are device-resident. `CreateAllocator` fails if the session has no
    /// allocator matching `memory_info` (for example when the requested EP was
    /// not actually attached and the session fell back to CPU); callers should
    /// treat that error as "device allocation unavailable" and fall back to the
    /// CPU allocator. The returned allocator becomes invalid when the session is
    /// dropped, so it must not outlive the session it was created from.
    pub(crate) fn for_session_device(
        session: *const onnx_genai_ort_sys::OrtSession,
        memory_info: MemoryInfo,
    ) -> Result<Self> {
        let mut ptr = std::ptr::null_mut();
        let api = crate::error::api()?;
        let create = api
            .CreateAllocator
            .ok_or(OrtError::ApiUnavailable("CreateAllocator"))?;
        // SAFETY: `session` is a valid ORT session pointer and `memory_info` is a
        // valid handle; `ptr` is an out-parameter. On success ORT transfers an
        // owned allocator that this wrapper releases in Drop.
        crate::error::check_status(unsafe { create(session, memory_info.as_ptr(), &mut ptr) })?;
        Ok(Self {
            ptr,
            memory_info,
            owned: true,
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
