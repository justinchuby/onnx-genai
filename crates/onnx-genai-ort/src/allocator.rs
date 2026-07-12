//! ORT Allocator and MemoryInfo.

use crate::Result;

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
    // ptr: *mut ort_sys::OrtMemoryInfo,
    pub device_name: String,
    pub device_id: i32,
    pub memory_type: MemoryType,
}

impl MemoryInfo {
    /// Create CPU memory info.
    pub fn cpu() -> Result<Self> {
        Ok(Self {
            device_name: "Cpu".to_string(),
            device_id: 0,
            memory_type: MemoryType::Default,
        })
    }

    /// Create CUDA device memory info.
    pub fn cuda(device_id: i32) -> Result<Self> {
        Ok(Self {
            device_name: "Cuda".to_string(),
            device_id,
            memory_type: MemoryType::Default,
        })
    }

    /// Create DirectML device memory info.
    pub fn dml(device_id: i32) -> Result<Self> {
        Ok(Self {
            device_name: "DML".to_string(),
            device_id,
            memory_type: MemoryType::Default,
        })
    }
}

impl Drop for MemoryInfo {
    fn drop(&mut self) {
        // TODO: Call OrtReleaseMemoryInfo
    }
}

/// ORT Allocator.
pub struct Allocator {
    // ptr: *mut ort_sys::OrtAllocator,
    pub memory_info: MemoryInfo,
}

impl Allocator {
    /// Get the default CPU allocator.
    pub fn default_cpu() -> Result<Self> {
        Ok(Self {
            memory_info: MemoryInfo::cpu()?,
        })
    }
}
