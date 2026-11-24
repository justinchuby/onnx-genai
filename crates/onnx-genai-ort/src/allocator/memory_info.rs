//! Inert memory descriptions and explicitly resolved ORT handles.
use std::{cell::OnceCell, ffi::CString, ptr::NonNull, rc::Rc};

use super::{AllocatorType, MemoryType};
use crate::{OrtError, Result};

#[derive(Clone, Debug)]
pub(crate) enum MemoryDescriptor {
    Cpu {
        allocator: AllocatorType,
    },
    Device {
        name: String,
        id: i32,
        memory_type: MemoryType,
    },
    WebGpu,
}

impl MemoryDescriptor {
    pub(crate) fn device_name(&self) -> &str {
        match self {
            Self::Cpu { .. } => "Cpu",
            Self::Device { name, .. } => name,
            Self::WebGpu => "WebGPU_Buffer",
        }
    }
    pub(crate) fn device_id(&self) -> i32 {
        match self {
            Self::Device { id, .. } => *id,
            _ => 0,
        }
    }
    pub(crate) fn memory_type(&self) -> MemoryType {
        match self {
            Self::Device { memory_type, .. } => *memory_type,
            _ => MemoryType::Default,
        }
    }
    fn is_arena(&self) -> bool {
        matches!(
            self,
            Self::Cpu {
                allocator: AllocatorType::Arena
            }
        )
    }

    fn create_handle(&self) -> Result<NonNull<onnx_genai_ort_sys::OrtMemoryInfo>> {
        let api = crate::error::api()?;
        let mut ptr = std::ptr::null_mut();
        match self {
            Self::Cpu { allocator } => {
                let create = api
                    .CreateCpuMemoryInfo
                    .ok_or(OrtError::ApiUnavailable("CreateCpuMemoryInfo"))?;
                let kind = match allocator {
                    AllocatorType::Arena => onnx_genai_ort_sys::OrtArenaAllocator,
                    AllocatorType::Device => onnx_genai_ort_sys::OrtDeviceAllocator,
                };
                // SAFETY: ptr is a live out-parameter. Successful creation
                // transfers one handle to ResolvedMemoryInfo's shared owner.
                crate::error::check_status(unsafe {
                    create(kind, onnx_genai_ort_sys::OrtMemTypeDefault, &mut ptr)
                })?;
            }
            Self::Device {
                name,
                id,
                memory_type,
            } => {
                let name = CString::new(name.as_str()).map_err(|_| {
                    OrtError::InvalidArgument(
                        "memory device name contains NUL; supply a valid provider memory name"
                            .into(),
                    )
                })?;
                let create = api
                    .CreateMemoryInfo
                    .ok_or(OrtError::ApiUnavailable("CreateMemoryInfo"))?;
                let memory_type = match memory_type {
                    MemoryType::CpuInput => onnx_genai_ort_sys::OrtMemTypeCPUInput,
                    MemoryType::CpuOutput => onnx_genai_ort_sys::OrtMemTypeCPUOutput,
                    MemoryType::Default => onnx_genai_ort_sys::OrtMemTypeDefault,
                };
                // SAFETY: name lives through the call and ptr is a valid
                // out-parameter; all other arguments are explicit descriptor data.
                crate::error::check_status(unsafe {
                    create(
                        name.as_ptr(),
                        onnx_genai_ort_sys::OrtDeviceAllocator,
                        *id,
                        memory_type,
                        &mut ptr,
                    )
                })?;
            }
            Self::WebGpu => {
                let create = api
                    .CreateMemoryInfo_V2
                    .ok_or(OrtError::ApiUnavailable("CreateMemoryInfo_V2"))?;
                // This describes ORT's WebGPU allocator contract: GPU, vendor
                // NONE, device 0, DEFAULT memory, default alignment, Device kind.
                // SAFETY: the static C string and ptr out-parameter are valid.
                crate::error::check_status(unsafe {
                    create(
                        c"WebGPU_Buffer".as_ptr(),
                        onnx_genai_ort_sys::OrtMemoryInfoDeviceType_GPU,
                        0,
                        0,
                        onnx_genai_ort_sys::OrtDeviceMemoryType_DEFAULT,
                        0,
                        onnx_genai_ort_sys::OrtDeviceAllocator,
                        &mut ptr,
                    )
                })?;
            }
        }
        NonNull::new(ptr).ok_or(OrtError::NullPointer)
    }
}

#[derive(Debug)]
struct ResolvedHandle {
    ptr: NonNull<onnx_genai_ort_sys::OrtMemoryInfo>,
    descriptor: Rc<MemoryDescriptor>,
}

impl Drop for ResolvedHandle {
    fn drop(&mut self) {
        // Only a successful ORT call can construct this owner. Descriptions
        // that were never resolved never reach this destructor or load ORT.
        if let Ok(api) = crate::error::api()
            && let Some(release) = api.ReleaseMemoryInfo
        {
            // SAFETY: this is the last shared owner of exactly one ORT handle.
            unsafe { release(self.ptr.as_ptr()) };
        }
    }
}

/// A ready handle for an ORT adapter or callback. It owns its lifetime rather
/// than borrowing a descriptor that a caller could drop after registration.
#[derive(Clone, Debug)]
pub(crate) struct ResolvedMemoryInfo(Rc<ResolvedHandle>);
impl ResolvedMemoryInfo {
    pub(crate) fn as_ptr(&self) -> *const onnx_genai_ort_sys::OrtMemoryInfo {
        self.0.ptr.as_ptr()
    }
    pub(crate) fn is_arena(&self) -> bool {
        self.0.descriptor.is_arena()
    }
}

#[derive(Debug)]
struct MemoryInfoState {
    descriptor: Rc<MemoryDescriptor>,
    resolved: OnceCell<ResolvedMemoryInfo>,
}

/// Immutable memory placement. Creating, reading, cloning and dropping an
/// unresolved description does not initialize or load ONNX Runtime.
#[derive(Clone, Debug)]
pub struct MemoryInfo {
    state: Rc<MemoryInfoState>,
}
impl MemoryInfo {
    pub(crate) fn new(descriptor: MemoryDescriptor) -> Self {
        Self {
            state: Rc::new(MemoryInfoState {
                descriptor: Rc::new(descriptor),
                resolved: OnceCell::new(),
            }),
        }
    }
    pub(crate) fn descriptor(&self) -> MemoryDescriptor {
        (*self.state.descriptor).clone()
    }
    /// Describe CPU memory managed by an arena allocator.
    pub fn cpu() -> Result<Self> {
        Ok(Self::new(MemoryDescriptor::Cpu {
            allocator: AllocatorType::Arena,
        }))
    }
    /// Describe a device-kind CPU allocator suitable for ORT registration.
    pub fn cpu_device() -> Result<Self> {
        Ok(Self::new(MemoryDescriptor::Cpu {
            allocator: AllocatorType::Device,
        }))
    }
    /// Describe CUDA device memory without opening a reference runtime.
    #[cfg(feature = "cuda")]
    pub fn cuda(device_id: i32) -> Result<Self> {
        Self::new_device("Cuda", device_id, MemoryType::Default)
    }
    /// Describe DirectML device memory.
    pub fn dml(device_id: i32) -> Result<Self> {
        Self::new_device("DML", device_id, MemoryType::Default)
    }
    /// Describe the ORT WebGPU provider's GPU memory contract.
    pub fn webgpu() -> Result<Self> {
        Ok(Self::new(MemoryDescriptor::WebGpu))
    }
    fn new_device(name: &str, id: i32, memory_type: MemoryType) -> Result<Self> {
        if name.as_bytes().contains(&0) {
            return Err(OrtError::InvalidArgument(
                "memory device name contains NUL; supply a valid provider memory name".into(),
            ));
        }
        Ok(Self::new(MemoryDescriptor::Device {
            name: name.to_owned(),
            id,
            memory_type,
        }))
    }
    /// The declared provider memory name, without opening a runtime.
    pub fn device_name(&self) -> &str {
        self.state.descriptor.device_name()
    }
    /// The declared device index.
    pub fn device_id(&self) -> i32 {
        self.state.descriptor.device_id()
    }
    /// The declared CPU input/output or default memory classification.
    pub fn memory_type(&self) -> MemoryType {
        self.state.descriptor.memory_type()
    }
    pub(crate) fn is_host_accessible(&self) -> bool {
        matches!(*self.state.descriptor, MemoryDescriptor::Cpu { .. })
    }

    /// Explicit reference boundary. Cache only successful resolution; callback
    /// owners receive the ready type, so their Info callback cannot fail or load.
    pub(crate) fn resolve(&self) -> Result<ResolvedMemoryInfo> {
        if let Some(handle) = self.state.resolved.get() {
            return Ok(handle.clone());
        }
        let ptr = self.state.descriptor.create_handle().map_err(|source| OrtError::Context {
            operation: format!("resolve ORT memory info for {} device {} ({:?}); ensure the selected reference runtime supports this memory contract", self.device_name(), self.device_id(), self.memory_type()),
            source: Box::new(source),
        })?;
        let handle = ResolvedMemoryInfo(Rc::new(ResolvedHandle {
            ptr,
            descriptor: Rc::clone(&self.state.descriptor),
        }));
        self.state.resolved.set(handle.clone()).map_err(|_| OrtError::InvalidArgument("ORT memory-info resolution was re-entered; resolve the descriptor before exposing it to callbacks".into()))?;
        Ok(handle)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn descriptions_work_without_a_reference_runtime() {
        const CHILD: &str = "NXRT_TEST_MEMORY_DESCRIPTION_CHILD";
        if std::env::var_os(CHILD).is_none() {
            let scratch = tempfile::tempdir().unwrap();
            let missing = scratch.path().join("missing-ort-reference-runtime");
            assert!(!missing.exists());
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "allocator::memory_info::tests::descriptions_work_without_a_reference_runtime",
                    "--nocapture",
                ])
                .env(CHILD, "1")
                .env("ONNX_GENAI_ORT_LIB", &missing)
                .output()
                .unwrap();
            assert!(
                String::from_utf8_lossy(&output.stdout).contains("running 1 test"),
                "child filter must select this test: {}{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
            assert!(
                output.status.success(),
                "{}{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
            return;
        }
        let cpu = MemoryInfo::cpu().unwrap();
        let device = MemoryInfo::cpu_device().unwrap();
        let dml = MemoryInfo::dml(3).unwrap();
        let webgpu = MemoryInfo::webgpu().unwrap();
        assert_eq!(cpu.device_name(), "Cpu");
        assert_eq!(cpu.device_id(), 0);
        assert!(cpu.is_host_accessible());
        assert!(cpu.state.descriptor.is_arena());
        assert!(!device.state.descriptor.is_arena());
        assert_eq!(dml.device_name(), "DML");
        assert_eq!(dml.device_id(), 3);
        assert!(!dml.is_host_accessible());
        #[cfg(feature = "cuda")]
        {
            let cuda = MemoryInfo::cuda(2).unwrap();
            assert_eq!(cuda.device_name(), "Cuda");
            assert_eq!(cuda.device_id(), 2);
            assert!(!cuda.is_host_accessible());
            assert!(cuda.state.resolved.get().is_none());
        }
        assert_eq!(webgpu.device_name(), "WebGPU_Buffer");
        assert!(!webgpu.is_host_accessible());
        for description in [&cpu, &device, &dml, &webgpu] {
            assert_eq!(description.memory_type(), MemoryType::Default);
            let copy = description.clone();
            assert!(Rc::ptr_eq(&description.state, &copy.state));
            assert!(copy.state.resolved.get().is_none());
            drop(copy);
            assert!(description.state.resolved.get().is_none());
        }
        // Only the explicit adapter boundary needs the unavailable library.
        let error = cpu.resolve().unwrap_err();
        assert!(std::error::Error::source(&error).is_some());
        assert!(
            matches!(&error, OrtError::Context { source, .. } if matches!(source.as_ref(), OrtError::RuntimeLibrary(_)))
        );
        let message = error.to_string();
        assert!(message.contains("Cpu device 0"), "{message}");
        let selected = std::env::var("ONNX_GENAI_ORT_LIB").unwrap();
        assert!(message.contains(&selected), "{message}");
        assert!(cpu.state.resolved.get().is_none());
        assert_eq!(cpu.device_name(), "Cpu");
        assert_eq!(cpu.clone().device_id(), 0);
    }

    #[test]
    fn resolved_handles_are_shared_and_outlive_the_description() {
        for (description, arena) in [
            (MemoryInfo::cpu().unwrap(), true),
            (MemoryInfo::cpu_device().unwrap(), false),
        ] {
            let ready = description.resolve().unwrap();
            let copy = description.clone().resolve().unwrap();
            assert_eq!(ready.as_ptr(), copy.as_ptr());
            assert_eq!(ready.is_arena(), arena);
            let api = crate::error::api().unwrap();
            let mut kind = onnx_genai_ort_sys::OrtInvalidAllocator;
            // SAFETY: ready is a resolved owner and kind is a live out-parameter.
            crate::error::check_status(unsafe {
                api.MemoryInfoGetType.unwrap()(ready.as_ptr(), &mut kind)
            })
            .unwrap();
            assert_eq!(kind == onnx_genai_ort_sys::OrtArenaAllocator, arena);
            drop(description);
            drop(copy);
            // The remaining owner still keeps the real ORT handle valid.
            let mut id = -1;
            // SAFETY: ready still owns the handle and id is a live out-parameter.
            crate::error::check_status(unsafe {
                api.MemoryInfoGetId.unwrap()(ready.as_ptr(), &mut id)
            })
            .unwrap();
            assert_eq!(id, 0);
        }
    }
}
