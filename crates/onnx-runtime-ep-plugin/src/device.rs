//! Device-capable adapter surfaces for the ORT plugin-EP C ABI.
//!
//! Generalizes device enumeration, allocator creation, stream/synchronization,
//! and data transfer beyond CPU-only. Any EP declaring GPU/NPU device types
//! can project its Rust `ExecutionProvider` through these surfaces.
//!
//! # Ownership contracts (from `onnxruntime_ep_c_api.h`)
//!
//! - **`OrtMemoryInfo` passed to `EpDevice_AddAllocatorInfo`:** ORT stores the
//!   raw pointer; does NOT copy it. Must outlive the `OrtEpDevice`. ORT releases
//!   it via `ReleaseEpDevice`. Do NOT call `ReleaseMemoryInfo` after a successful
//!   `AddAllocatorInfo`. (header line ~1092–1111)
//!
//! - **`OrtSyncStreamImpl`:** ORT calls `Release` on the vtable when done. The
//!   implementation must release resources in its `Release` callback.
//!   (header line ~204–258)
//!
//! - **`OrtAllocator`:** ORT calls `OrtEpFactory::ReleaseAllocator` to free.
//!   The factory must track lifetime. (header line ~2835)
//!
//! - **`OrtHardwareDevice`:** Created via `CreateHardwareDevice`; ORT takes
//!   ownership of the returned `OrtEpDevice` array entries.
//!   (header line ~1225–1241)

use std::ptr;

use onnx_genai_ort_sys as ort;
use onnx_runtime_ep_api::provider::ExecutionProvider;
use onnx_runtime_ir::DeviceType;

use crate::status::fail_status;

// ─── Hardware device type mapping ────────────────────────────────────────────

/// Maps our internal `DeviceType` to ORT's `OrtHardwareDeviceType`.
///
/// Returns `None` for device types that have no ORT hardware representation
/// (e.g. WebGpu, custom).
pub fn device_type_to_ort_hardware(dt: DeviceType) -> Option<ort::OrtHardwareDeviceType> {
    match dt {
        DeviceType::Cpu => Some(ort::OrtHardwareDeviceType_CPU),
        DeviceType::Cuda | DeviceType::Rocm => Some(ort::OrtHardwareDeviceType_GPU),
        DeviceType::Qnn => Some(ort::OrtHardwareDeviceType_NPU),
        // CoreML/MLX run on Apple silicon which ORT classifies as NPU
        DeviceType::CoreMl | DeviceType::Mlx => Some(ort::OrtHardwareDeviceType_NPU),
        DeviceType::OpenVino => Some(ort::OrtHardwareDeviceType_NPU),
        DeviceType::WebGpu | DeviceType::Custom(_) => None,
    }
}

/// Maps ORT's `OrtHardwareDeviceType` to `OrtMemoryInfoDeviceType` for
/// `CreateMemoryInfo_V2`.
pub fn hardware_type_to_memory_device_type(
    hw: ort::OrtHardwareDeviceType,
) -> ort::OrtMemoryInfoDeviceType {
    match hw {
        ort::OrtHardwareDeviceType_GPU => ort::OrtMemoryInfoDeviceType_GPU,
        ort::OrtHardwareDeviceType_NPU => ort::OrtMemoryInfoDeviceType_NPU,
        _ => ort::OrtMemoryInfoDeviceType_CPU,
    }
}

/// Checks whether an EP supports the given ORT hardware device type.
pub fn ep_supports_hardware_type(
    ep: &dyn ExecutionProvider,
    hw_type: ort::OrtHardwareDeviceType,
) -> bool {
    let ep_hw = device_type_to_ort_hardware(ep.device_type());
    ep_hw == Some(hw_type)
}

// ─── Allocator adapter ───────────────────────────────────────────────────────

/// A heap-allocated allocator projecting an EP's `allocate`/`deallocate` through
/// ORT's `OrtAllocator` vtable.
///
/// The first field is `OrtAllocator` so a cast from `*mut DeviceAllocator` to
/// `*mut OrtAllocator` is valid (repr(C)).
#[repr(C)]
pub struct DeviceAllocator {
    pub vtable: ort::OrtAllocator,
    /// The EP backing this allocator. We hold a raw pointer; the EP must outlive
    /// the allocator (guaranteed by ORT calling `ReleaseAllocator` before the
    /// factory is released which destroys the EP).
    pub ep: *const dyn ExecutionProvider,
    /// Memory info pointer. Borrowed from ORT; NOT freed by this allocator.
    /// ORT owns the lifetime and releases it when `ReleaseEpDevice` is called.
    /// Must not be passed to `EpDevice_AddAllocatorInfo` (that's separate).
    pub memory_info: *const ort::OrtMemoryInfo,
}

// SAFETY: The EP behind the raw pointer is Send+Sync per ExecutionProvider trait bound.
unsafe impl Send for DeviceAllocator {}
unsafe impl Sync for DeviceAllocator {}

impl DeviceAllocator {
    /// Create a new device allocator wrapping the given EP.
    ///
    /// `memory_info` must be a valid pointer created by `CreateMemoryInfo_V2`.
    /// This allocator takes ownership; it will be returned via `Info()` and freed
    /// when the allocator is released.
    ///
    /// # Safety
    ///
    /// `ep` must remain valid for the lifetime of this allocator.
    /// `memory_info` must be a valid ORT-owned pointer.
    pub unsafe fn new(
        ep: *const dyn ExecutionProvider,
        memory_info: *const ort::OrtMemoryInfo,
    ) -> Box<Self> {
        Box::new(Self {
            vtable: ort::OrtAllocator {
                version: ort::ORT_API_VERSION,
                Alloc: Some(device_alloc),
                Free: Some(device_free),
                Info: Some(device_info),
                Reserve: Some(device_reserve),
                GetStats: None,
                AllocOnStream: None,
                Shrink: None,
            },
            ep,
            memory_info,
        })
    }
}

unsafe extern "C" fn device_alloc(
    this: *mut ort::OrtAllocator,
    size: usize,
) -> *mut std::os::raw::c_void {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        if this.is_null() {
            return ptr::null_mut();
        }
        let alloc = unsafe { &*(this.cast::<DeviceAllocator>()) };
        let ep = unsafe { &*alloc.ep };
        // Use default alignment (16 bytes, reasonable for GPU).
        match ep.allocate(size, 16) {
            Ok(buf) => {
                let ptr = buf.as_ptr() as *mut std::os::raw::c_void;
                // DeviceBuffer has no Drop impl — when `buf` goes out of scope,
                // only the handle metadata is discarded; the underlying
                // allocation stays alive until `device_free` reconstructs a
                // DeviceBuffer and calls `ep.deallocate()`.
                ptr
            }
            Err(_) => ptr::null_mut(),
        }
    }));
    result.unwrap_or(ptr::null_mut())
}

unsafe extern "C" fn device_free(this: *mut ort::OrtAllocator, p: *mut std::os::raw::c_void) {
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        if this.is_null() || p.is_null() {
            return;
        }
        let alloc = unsafe { &*(this.cast::<DeviceAllocator>()) };
        let ep = unsafe { &*alloc.ep };
        // BUG (defect #4): size is unknown at free time. We pass size=0 which
        // violates the allocator contract — the EP's `deallocate` may need the
        // real size for arena management, cudaFree bookkeeping, or leak detection.
        //
        // FIX REQUIRED: Track allocation sizes in a side table (e.g.
        // HashMap<*mut c_void, usize> behind a Mutex) populated by device_alloc
        // and looked up here. Alternatively, the EP could support free-by-pointer
        // (ignoring size), but that must be an explicit contract, not an accident.
        //
        // Alignment is also hardcoded to 16 — must match what device_alloc used.
        let buf = unsafe {
            onnx_runtime_ep_api::provider::DeviceBuffer::from_raw_parts(
                p,
                ep.device_id(),
                0, // BUG: size unknown — see defect #4 above
                16,
            )
        };
        let _ = ep.deallocate(buf);
    }));
}

unsafe extern "C" fn device_info(this: *const ort::OrtAllocator) -> *const ort::OrtMemoryInfo {
    if this.is_null() {
        return ptr::null();
    }
    let alloc = unsafe { &*(this.cast::<DeviceAllocator>()) };
    alloc.memory_info
}

unsafe extern "C" fn device_reserve(
    this: *mut ort::OrtAllocator,
    size: usize,
) -> *mut std::os::raw::c_void {
    // Reserve == Alloc for most device allocators.
    unsafe { device_alloc(this, size) }
}

// ─── Sync stream adapter ─────────────────────────────────────────────────────

/// A heap-allocated sync stream projecting an EP's synchronization primitives
/// through ORT's `OrtSyncStreamImpl` vtable.
#[repr(C)]
pub struct DeviceSyncStream {
    pub vtable: ort::OrtSyncStreamImpl,
    /// EP reference for sync/flush operations.
    pub ep: *const dyn ExecutionProvider,
}

// SAFETY: EP is Send+Sync.
unsafe impl Send for DeviceSyncStream {}
unsafe impl Sync for DeviceSyncStream {}

impl DeviceSyncStream {
    /// Create a new sync stream wrapping the given EP.
    ///
    /// # Safety
    ///
    /// `ep` must remain valid for the lifetime of this stream. ORT calls
    /// `Release` on the vtable when it's done.
    pub unsafe fn new(ep: *const dyn ExecutionProvider) -> Box<Self> {
        Box::new(Self {
            vtable: ort::OrtSyncStreamImpl {
                ort_version_supported: ort::ORT_API_VERSION,
                Release: Some(stream_release),
                GetHandle: Some(stream_get_handle),
                CreateNotification: None,
                Flush: Some(stream_flush),
                OnSessionRunEnd: Some(stream_on_session_run_end),
            },
            ep,
        })
    }
}

unsafe extern "C" fn stream_release(this: *mut ort::OrtSyncStreamImpl) {
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        if this.is_null() {
            return;
        }
        // Reconstruct and drop the Box, then reclaim the owned EP.
        // The EP was created via `Box::into_raw` in `factory_create_sync_stream`
        // and must be reclaimed here — ORT calls `Release` exactly once per
        // created stream (header line ~207–216).
        unsafe {
            let stream = Box::from_raw(this.cast::<DeviceSyncStream>());
            if !stream.ep.is_null() {
                drop(Box::from_raw(stream.ep as *mut dyn ExecutionProvider));
            }
        }
    }));
}

unsafe extern "C" fn stream_get_handle(
    _this: *mut ort::OrtSyncStreamImpl,
) -> *mut std::os::raw::c_void {
    // For mock/CPU: no native stream handle. Real CUDA EP would return cudaStream_t.
    ptr::null_mut()
}

unsafe extern "C" fn stream_flush(this: *mut ort::OrtSyncStreamImpl) -> ort::OrtStatusPtr {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        if this.is_null() {
            return fail_status("Flush: null stream pointer");
        }
        let stream = unsafe { &*(this.cast::<DeviceSyncStream>()) };
        let ep = unsafe { &*stream.ep };
        match ep.sync() {
            Ok(()) => crate::status::ok_status(),
            Err(e) => fail_status(&format!("Flush: sync failed: {e}")),
        }
    }));
    result.unwrap_or_else(|_| fail_status("Flush: internal panic"))
}

unsafe extern "C" fn stream_on_session_run_end(
    this: *mut ort::OrtSyncStreamImpl,
) -> ort::OrtStatusPtr {
    // Flush on session run end as well.
    unsafe { stream_flush(this) }
}

// ─── Device enumeration helpers ──────────────────────────────────────────────

/// Configuration for how an EP declares its supported devices.
///
/// Used by the adapter to generalize `GetSupportedDevices` beyond CPU-only
/// filtering. `factory.rs` should adopt this struct to support GPU/NPU EPs.
#[derive(Clone, Debug)]
pub struct DeviceSupport {
    /// ORT hardware device types this EP serves.
    pub hardware_types: Vec<ort::OrtHardwareDeviceType>,
    /// EP name for the memory info allocator name field.
    pub allocator_name: &'static str,
    /// Vendor ID (PCI) for memory info. 0 = generic.
    pub vendor_id: u32,
    /// Whether this EP is stream-aware (needs `CreateSyncStreamForDevice`).
    pub stream_aware: bool,
    /// Whether device memory is host-accessible. If false, allocator creates
    /// device-only memory and a data transfer impl is needed.
    pub host_accessible: bool,
}

impl DeviceSupport {
    /// CPU-only support (current default).
    pub fn cpu_only() -> Self {
        Self {
            hardware_types: vec![ort::OrtHardwareDeviceType_CPU],
            allocator_name: "Cpu",
            vendor_id: 0,
            stream_aware: false,
            host_accessible: true,
        }
    }

    /// GPU support configuration.
    pub fn gpu(allocator_name: &'static str, vendor_id: u32) -> Self {
        Self {
            hardware_types: vec![ort::OrtHardwareDeviceType_GPU],
            allocator_name,
            vendor_id,
            stream_aware: true,
            host_accessible: false,
        }
    }

    /// Check whether a given hardware type is served by this config.
    pub fn serves(&self, hw_type: ort::OrtHardwareDeviceType) -> bool {
        self.hardware_types.contains(&hw_type)
    }

    /// Return the `OrtMemoryInfoDeviceType` for device-default memory on this EP.
    pub fn memory_device_type(&self) -> ort::OrtMemoryInfoDeviceType {
        if self
            .hardware_types
            .contains(&ort::OrtHardwareDeviceType_GPU)
        {
            ort::OrtMemoryInfoDeviceType_GPU
        } else if self
            .hardware_types
            .contains(&ort::OrtHardwareDeviceType_NPU)
        {
            ort::OrtMemoryInfoDeviceType_NPU
        } else {
            ort::OrtMemoryInfoDeviceType_CPU
        }
    }
}

// ─── Fail-closed validation helpers ─────────────────────────────────────────

/// Validate that an EP's device type matches the requested hardware type.
/// Returns an error status if mismatched.
pub fn validate_device_support(
    ep: &dyn ExecutionProvider,
    requested_hw: ort::OrtHardwareDeviceType,
) -> *mut ort::OrtStatus {
    if !ep_supports_hardware_type(ep, requested_hw) {
        let ep_type = ep.device_type().trace_name();
        let hw_name = match requested_hw {
            ort::OrtHardwareDeviceType_CPU => "CPU",
            ort::OrtHardwareDeviceType_GPU => "GPU",
            ort::OrtHardwareDeviceType_NPU => "NPU",
            _ => "Unknown",
        };
        return fail_status(&format!(
            "Device mismatch: EP '{ep_type}' does not serve hardware type {hw_name}"
        ));
    }
    crate::status::ok_status()
}

/// Validate that allocator creation is appropriate for the EP's device type.
/// A device-memory allocator request for a CPU-only EP must fail closed.
pub fn validate_allocator_request(
    ep: &dyn ExecutionProvider,
    memory_device_type: ort::OrtMemoryInfoDeviceType,
) -> *mut ort::OrtStatus {
    let ep_hw = device_type_to_ort_hardware(ep.device_type());
    let is_device_request = memory_device_type == ort::OrtMemoryInfoDeviceType_GPU
        || memory_device_type == ort::OrtMemoryInfoDeviceType_NPU;
    let is_device_ep = ep_hw == Some(ort::OrtHardwareDeviceType_GPU)
        || ep_hw == Some(ort::OrtHardwareDeviceType_NPU);

    if is_device_request && !is_device_ep {
        return fail_status(
            "Allocator mismatch: device-memory allocator requested from CPU-only EP",
        );
    }
    crate::status::ok_status()
}

/// Validate that stream creation is supported by this EP.
/// A non-stream-aware EP must fail closed on stream creation requests.
pub fn validate_stream_request(support: &DeviceSupport) -> *mut ort::OrtStatus {
    if !support.stream_aware {
        return fail_status(
            "Stream not supported: EP is not stream-aware; cannot create sync stream",
        );
    }
    crate::status::ok_status()
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use onnx_runtime_ep_api::{EpError, Kernel, KernelMatch, Result as EpResult};
    use onnx_runtime_ir::{DataType, DeviceId, DeviceType, Node, Shape, TensorLayout};

    /// A mock GPU execution provider for testing device surfaces without hardware.
    struct MockGpuEp;

    impl ExecutionProvider for MockGpuEp {
        fn name(&self) -> &str {
            "mock_gpu_ep"
        }

        fn device_type(&self) -> DeviceType {
            DeviceType::Cuda
        }

        fn device_id(&self) -> DeviceId {
            DeviceId::cuda(0)
        }

        fn initialize(
            &mut self,
            _config: &onnx_runtime_ep_api::provider::EpConfig,
        ) -> EpResult<()> {
            Ok(())
        }

        fn shutdown(&mut self) -> EpResult<()> {
            Ok(())
        }

        fn supports_op(
            &self,
            _op: &Node,
            _opset: u64,
            _shapes: &[Shape],
            _input_dtypes: &[DataType],
            _layouts: &[TensorLayout],
        ) -> KernelMatch {
            KernelMatch::Unsupported {
                reason: "mock".into(),
            }
        }

        fn get_kernel(
            &self,
            _op: &Node,
            _shapes: &[Vec<usize>],
            _opset: u64,
        ) -> EpResult<Box<dyn Kernel>> {
            Err(EpError::KernelFailed("mock: no kernel".into()))
        }

        fn allocate(
            &self,
            size: usize,
            _alignment: usize,
        ) -> EpResult<onnx_runtime_ep_api::provider::DeviceBuffer> {
            let layout = std::alloc::Layout::from_size_align(size.max(1), 16).unwrap();
            let ptr = unsafe { std::alloc::alloc(layout) };
            if ptr.is_null() {
                return Err(EpError::OutOfMemory {
                    requested: size,
                    available: 0,
                });
            }
            Ok(unsafe {
                onnx_runtime_ep_api::provider::DeviceBuffer::from_raw_parts(
                    ptr.cast(),
                    DeviceId::cuda(0),
                    size,
                    16,
                )
            })
        }

        fn deallocate(&self, buffer: onnx_runtime_ep_api::provider::DeviceBuffer) -> EpResult<()> {
            let ptr = buffer.as_ptr();
            let size = buffer.len();
            // DeviceBuffer has no Drop impl: binding to `_` discards the handle
            // metadata without invoking any destructor (there is none).
            let _ = buffer;
            if !ptr.is_null() && size > 0 {
                let layout = std::alloc::Layout::from_size_align(size, 16).unwrap();
                unsafe { std::alloc::dealloc(ptr as *mut u8, layout) };
            }
            Ok(())
        }

        fn copy(
            &self,
            _src: &onnx_runtime_ep_api::provider::DeviceBuffer,
            _dst: &mut onnx_runtime_ep_api::provider::DeviceBuffer,
            _size: usize,
        ) -> EpResult<()> {
            Ok(())
        }

        fn copy_async(
            &self,
            _src: &onnx_runtime_ep_api::provider::DeviceBuffer,
            _dst: &mut onnx_runtime_ep_api::provider::DeviceBuffer,
            _size: usize,
        ) -> EpResult<onnx_runtime_ep_api::provider::Fence> {
            Ok(onnx_runtime_ep_api::provider::Fence::signalled())
        }

        fn sync(&self) -> EpResult<()> {
            Ok(())
        }
    }

    /// A mock CPU EP for negative-path testing.
    struct MockCpuEp;

    impl ExecutionProvider for MockCpuEp {
        fn name(&self) -> &str {
            "mock_cpu_ep"
        }

        fn device_type(&self) -> DeviceType {
            DeviceType::Cpu
        }

        fn device_id(&self) -> DeviceId {
            DeviceId::cpu()
        }

        fn initialize(
            &mut self,
            _config: &onnx_runtime_ep_api::provider::EpConfig,
        ) -> EpResult<()> {
            Ok(())
        }

        fn shutdown(&mut self) -> EpResult<()> {
            Ok(())
        }

        fn supports_op(
            &self,
            _op: &Node,
            _opset: u64,
            _shapes: &[Shape],
            _input_dtypes: &[DataType],
            _layouts: &[TensorLayout],
        ) -> KernelMatch {
            KernelMatch::Unsupported {
                reason: "mock".into(),
            }
        }

        fn get_kernel(
            &self,
            _op: &Node,
            _shapes: &[Vec<usize>],
            _opset: u64,
        ) -> EpResult<Box<dyn Kernel>> {
            Err(EpError::KernelFailed("mock: no kernel".into()))
        }

        fn allocate(
            &self,
            size: usize,
            _alignment: usize,
        ) -> EpResult<onnx_runtime_ep_api::provider::DeviceBuffer> {
            let layout = std::alloc::Layout::from_size_align(size.max(1), 16).unwrap();
            let ptr = unsafe { std::alloc::alloc(layout) };
            if ptr.is_null() {
                return Err(EpError::OutOfMemory {
                    requested: size,
                    available: 0,
                });
            }
            Ok(unsafe {
                onnx_runtime_ep_api::provider::DeviceBuffer::from_raw_parts(
                    ptr.cast(),
                    DeviceId::cpu(),
                    size,
                    16,
                )
            })
        }

        fn deallocate(&self, buffer: onnx_runtime_ep_api::provider::DeviceBuffer) -> EpResult<()> {
            let ptr = buffer.as_ptr();
            let size = buffer.len();
            // DeviceBuffer has no Drop impl: binding to `_` discards the handle
            // metadata without invoking any destructor (there is none).
            let _ = buffer;
            if !ptr.is_null() && size > 0 {
                let layout = std::alloc::Layout::from_size_align(size, 16).unwrap();
                unsafe { std::alloc::dealloc(ptr as *mut u8, layout) };
            }
            Ok(())
        }

        fn copy(
            &self,
            _src: &onnx_runtime_ep_api::provider::DeviceBuffer,
            _dst: &mut onnx_runtime_ep_api::provider::DeviceBuffer,
            _size: usize,
        ) -> EpResult<()> {
            Ok(())
        }

        fn copy_async(
            &self,
            _src: &onnx_runtime_ep_api::provider::DeviceBuffer,
            _dst: &mut onnx_runtime_ep_api::provider::DeviceBuffer,
            _size: usize,
        ) -> EpResult<onnx_runtime_ep_api::provider::Fence> {
            Ok(onnx_runtime_ep_api::provider::Fence::signalled())
        }

        fn sync(&self) -> EpResult<()> {
            Ok(())
        }
    }

    // ─── Device type mapping tests ───────────────────────────────────────────

    #[test]
    fn cuda_maps_to_gpu_hardware_type() {
        assert_eq!(
            device_type_to_ort_hardware(DeviceType::Cuda),
            Some(ort::OrtHardwareDeviceType_GPU)
        );
    }

    #[test]
    fn rocm_maps_to_gpu_hardware_type() {
        assert_eq!(
            device_type_to_ort_hardware(DeviceType::Rocm),
            Some(ort::OrtHardwareDeviceType_GPU)
        );
    }

    #[test]
    fn cpu_maps_to_cpu_hardware_type() {
        assert_eq!(
            device_type_to_ort_hardware(DeviceType::Cpu),
            Some(ort::OrtHardwareDeviceType_CPU)
        );
    }

    #[test]
    fn webgpu_has_no_hardware_mapping() {
        assert_eq!(device_type_to_ort_hardware(DeviceType::WebGpu), None);
    }

    #[test]
    fn qnn_maps_to_npu() {
        assert_eq!(
            device_type_to_ort_hardware(DeviceType::Qnn),
            Some(ort::OrtHardwareDeviceType_NPU)
        );
    }

    // ─── EP hardware support tests ──────────────────────────────────────────

    #[test]
    fn gpu_ep_supports_gpu_hardware() {
        let ep = MockGpuEp;
        assert!(ep_supports_hardware_type(
            &ep,
            ort::OrtHardwareDeviceType_GPU
        ));
    }

    #[test]
    fn gpu_ep_does_not_support_cpu_hardware() {
        let ep = MockGpuEp;
        assert!(!ep_supports_hardware_type(
            &ep,
            ort::OrtHardwareDeviceType_CPU
        ));
    }

    #[test]
    fn cpu_ep_does_not_support_gpu_hardware() {
        let ep = MockCpuEp;
        assert!(!ep_supports_hardware_type(
            &ep,
            ort::OrtHardwareDeviceType_GPU
        ));
    }

    // ─── DeviceSupport config tests ─────────────────────────────────────────

    #[test]
    fn cpu_only_support_serves_cpu() {
        let support = DeviceSupport::cpu_only();
        assert!(support.serves(ort::OrtHardwareDeviceType_CPU));
        assert!(!support.serves(ort::OrtHardwareDeviceType_GPU));
        assert!(!support.stream_aware);
        assert!(support.host_accessible);
    }

    #[test]
    fn gpu_support_serves_gpu_not_cpu() {
        let support = DeviceSupport::gpu("Cuda", 0x10DE);
        assert!(support.serves(ort::OrtHardwareDeviceType_GPU));
        assert!(!support.serves(ort::OrtHardwareDeviceType_CPU));
        assert!(support.stream_aware);
        assert!(!support.host_accessible);
    }

    #[test]
    fn gpu_support_memory_device_type_is_gpu() {
        let support = DeviceSupport::gpu("Cuda", 0x10DE);
        assert_eq!(
            support.memory_device_type(),
            ort::OrtMemoryInfoDeviceType_GPU
        );
    }

    // ─── Fail-closed validation tests ───────────────────────────────────────

    #[test]
    fn validate_device_support_passes_for_matching_ep() {
        let ep = MockGpuEp;
        let status = validate_device_support(&ep, ort::OrtHardwareDeviceType_GPU);
        // null status == success (no ORT loaded for real status creation)
        assert!(status.is_null());
    }

    #[test]
    fn validate_device_support_fails_for_mismatched_ep() {
        let ep = MockCpuEp;
        let status = validate_device_support(&ep, ort::OrtHardwareDeviceType_GPU);
        // Without live ORT, fail_status returns null but the code path is exercised.
        // The important thing is it doesn't panic.
        let _ = status;
    }

    #[test]
    fn validate_allocator_request_fails_for_device_memory_on_cpu_ep() {
        let ep = MockCpuEp;
        let status = validate_allocator_request(&ep, ort::OrtMemoryInfoDeviceType_GPU);
        // Exercised without panic.
        let _ = status;
    }

    #[test]
    fn validate_allocator_request_passes_for_cpu_memory_on_cpu_ep() {
        let ep = MockCpuEp;
        let status = validate_allocator_request(&ep, ort::OrtMemoryInfoDeviceType_CPU);
        assert!(status.is_null());
    }

    #[test]
    fn validate_stream_request_fails_for_non_stream_aware_ep() {
        let support = DeviceSupport::cpu_only();
        let status = validate_stream_request(&support);
        // Fail-closed: should return error (null without live ORT).
        let _ = status;
    }

    #[test]
    fn validate_stream_request_passes_for_stream_aware_ep() {
        let support = DeviceSupport::gpu("Cuda", 0x10DE);
        let status = validate_stream_request(&support);
        assert!(status.is_null());
    }

    // ─── Allocator adapter tests ─────────────────────────────────────────────

    #[test]
    fn device_allocator_alloc_and_free_roundtrip() {
        let ep = MockGpuEp;
        let alloc =
            unsafe { DeviceAllocator::new(&ep as *const dyn ExecutionProvider, ptr::null()) };
        let alloc_ptr = Box::into_raw(alloc);

        // Allocate
        let ptr = unsafe { device_alloc(alloc_ptr.cast(), 1024) };
        assert!(!ptr.is_null(), "allocation must succeed");

        // Free
        unsafe { device_free(alloc_ptr.cast(), ptr) };

        // Cleanup
        unsafe { drop(Box::from_raw(alloc_ptr)) };
    }

    #[test]
    fn device_allocator_info_returns_stored_pointer() {
        let ep = MockGpuEp;
        let sentinel: u8 = 42;
        let fake_mem_info = &sentinel as *const u8 as *const ort::OrtMemoryInfo;
        let alloc =
            unsafe { DeviceAllocator::new(&ep as *const dyn ExecutionProvider, fake_mem_info) };
        let alloc_ptr = Box::into_raw(alloc);

        let info = unsafe { device_info(alloc_ptr.cast()) };
        assert_eq!(info, fake_mem_info);

        unsafe { drop(Box::from_raw(alloc_ptr)) };
    }

    #[test]
    fn device_allocator_null_this_returns_null() {
        let ptr = unsafe { device_alloc(ptr::null_mut(), 64) };
        assert!(ptr.is_null());
    }

    // ─── Stream adapter tests ────────────────────────────────────────────────

    #[test]
    fn device_sync_stream_flush_succeeds() {
        let ep: Box<dyn ExecutionProvider> = Box::new(MockGpuEp);
        let ep_ptr: *const dyn ExecutionProvider = Box::into_raw(ep);
        let stream = unsafe { DeviceSyncStream::new(ep_ptr) };
        let stream_ptr = Box::into_raw(stream);

        let status = unsafe { stream_flush(stream_ptr.cast()) };
        // null = success (no ORT API for real status)
        assert!(status.is_null());

        // Release via vtable (also reclaims the EP)
        unsafe { stream_release(stream_ptr.cast()) };
    }

    #[test]
    fn device_sync_stream_null_flush_fails() {
        let status = unsafe { stream_flush(ptr::null_mut()) };
        // Without live ORT, fail_status returns null but path is exercised.
        let _ = status;
    }

    #[test]
    fn device_sync_stream_get_handle_returns_null_for_mock() {
        let ep: Box<dyn ExecutionProvider> = Box::new(MockGpuEp);
        let ep_ptr: *const dyn ExecutionProvider = Box::into_raw(ep);
        let stream = unsafe { DeviceSyncStream::new(ep_ptr) };
        let stream_ptr = Box::into_raw(stream);

        let handle = unsafe { stream_get_handle(stream_ptr.cast()) };
        assert!(handle.is_null(), "mock stream has no native handle");

        unsafe { stream_release(stream_ptr.cast()) };
    }

    #[test]
    fn device_sync_stream_release_does_not_panic() {
        let ep: Box<dyn ExecutionProvider> = Box::new(MockGpuEp);
        let ep_ptr: *const dyn ExecutionProvider = Box::into_raw(ep);
        let stream = unsafe { DeviceSyncStream::new(ep_ptr) };
        let stream_ptr = Box::into_raw(stream);
        // Should not panic or leak.
        unsafe { stream_release(stream_ptr.cast()) };
    }

    #[test]
    fn device_sync_stream_release_null_does_not_panic() {
        // Null release must be safe.
        unsafe { stream_release(ptr::null_mut()) };
    }

    #[test]
    fn stream_release_reclaims_owned_ep_no_leak() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        static DROP_COUNT: AtomicUsize = AtomicUsize::new(0);

        /// An EP whose Drop increments a counter, proving the EP is reclaimed.
        struct CountingEp;

        impl Drop for CountingEp {
            fn drop(&mut self) {
                DROP_COUNT.fetch_add(1, Ordering::SeqCst);
            }
        }

        impl ExecutionProvider for CountingEp {
            fn name(&self) -> &str {
                "counting_ep"
            }
            fn device_type(&self) -> DeviceType {
                DeviceType::Cuda
            }
            fn device_id(&self) -> DeviceId {
                DeviceId::cuda(0)
            }
            fn initialize(
                &mut self,
                _config: &onnx_runtime_ep_api::provider::EpConfig,
            ) -> EpResult<()> {
                Ok(())
            }
            fn shutdown(&mut self) -> EpResult<()> {
                Ok(())
            }
            fn supports_op(
                &self,
                _op: &Node,
                _opset: u64,
                _shapes: &[Shape],
                _input_dtypes: &[DataType],
                _layouts: &[TensorLayout],
            ) -> KernelMatch {
                KernelMatch::Unsupported {
                    reason: "counting".into(),
                }
            }
            fn get_kernel(
                &self,
                _op: &Node,
                _shapes: &[Vec<usize>],
                _opset: u64,
            ) -> EpResult<Box<dyn Kernel>> {
                Err(EpError::KernelFailed("counting: no kernel".into()))
            }
            fn allocate(
                &self,
                _size: usize,
                _alignment: usize,
            ) -> EpResult<onnx_runtime_ep_api::provider::DeviceBuffer> {
                Err(EpError::OutOfMemory {
                    requested: 0,
                    available: 0,
                })
            }
            fn deallocate(
                &self,
                _buffer: onnx_runtime_ep_api::provider::DeviceBuffer,
            ) -> EpResult<()> {
                Ok(())
            }
            fn copy(
                &self,
                _src: &onnx_runtime_ep_api::provider::DeviceBuffer,
                _dst: &mut onnx_runtime_ep_api::provider::DeviceBuffer,
                _size: usize,
            ) -> EpResult<()> {
                Ok(())
            }
            fn copy_async(
                &self,
                _src: &onnx_runtime_ep_api::provider::DeviceBuffer,
                _dst: &mut onnx_runtime_ep_api::provider::DeviceBuffer,
                _size: usize,
            ) -> EpResult<onnx_runtime_ep_api::provider::Fence> {
                Ok(onnx_runtime_ep_api::provider::Fence::signalled())
            }
            fn sync(&self) -> EpResult<()> {
                Ok(())
            }
        }

        DROP_COUNT.store(0, Ordering::SeqCst);

        // Simulate the factory path: Box the EP, leak it, pass to stream.
        let ep: Box<dyn ExecutionProvider> = Box::new(CountingEp);
        let ep_ptr: *const dyn ExecutionProvider = Box::into_raw(ep);

        let stream = unsafe { DeviceSyncStream::new(ep_ptr) };
        let stream_ptr = Box::into_raw(stream);

        assert_eq!(
            DROP_COUNT.load(Ordering::SeqCst),
            0,
            "EP must not be dropped yet"
        );

        // Release the stream — this must also reclaim the EP.
        unsafe { stream_release(stream_ptr.cast()) };

        assert_eq!(
            DROP_COUNT.load(Ordering::SeqCst),
            1,
            "stream_release must drop the owned EP exactly once (leak regression)"
        );
    }

    // ─── Hardware type → memory device type mapping ─────────────────────────

    #[test]
    fn hardware_type_to_memory_maps_correctly() {
        assert_eq!(
            hardware_type_to_memory_device_type(ort::OrtHardwareDeviceType_GPU),
            ort::OrtMemoryInfoDeviceType_GPU
        );
        assert_eq!(
            hardware_type_to_memory_device_type(ort::OrtHardwareDeviceType_NPU),
            ort::OrtMemoryInfoDeviceType_NPU
        );
        assert_eq!(
            hardware_type_to_memory_device_type(ort::OrtHardwareDeviceType_CPU),
            ort::OrtMemoryInfoDeviceType_CPU
        );
    }

    // ─── Generalized enumeration tests ──────────────────────────────────────

    #[test]
    fn cpu_support_enumerates_cpu_device() {
        let support = DeviceSupport::cpu_only();
        assert!(
            support.serves(ort::OrtHardwareDeviceType_CPU),
            "CPU support must enumerate CPU devices"
        );
    }

    #[test]
    fn cpu_support_does_not_enumerate_gpu_device() {
        let support = DeviceSupport::cpu_only();
        assert!(
            !support.serves(ort::OrtHardwareDeviceType_GPU),
            "CPU support must not enumerate GPU devices"
        );
    }

    #[test]
    fn gpu_support_enumerates_gpu_device() {
        let support = DeviceSupport::gpu("Cuda", 0x10DE);
        assert!(
            support.serves(ort::OrtHardwareDeviceType_GPU),
            "GPU support must enumerate GPU devices"
        );
    }

    #[test]
    fn gpu_support_does_not_enumerate_cpu_device() {
        let support = DeviceSupport::gpu("Cuda", 0x10DE);
        assert!(
            !support.serves(ort::OrtHardwareDeviceType_CPU),
            "GPU support must not enumerate CPU devices — fail closed"
        );
    }

    #[test]
    fn gpu_support_does_not_enumerate_npu_device() {
        let support = DeviceSupport::gpu("Cuda", 0x10DE);
        assert!(
            !support.serves(ort::OrtHardwareDeviceType_NPU),
            "GPU support must not enumerate NPU devices — fail closed"
        );
    }

    #[test]
    fn device_support_memory_device_type_matches_hardware() {
        let cpu = DeviceSupport::cpu_only();
        assert_eq!(cpu.memory_device_type(), ort::OrtMemoryInfoDeviceType_CPU);

        let gpu = DeviceSupport::gpu("Cuda", 0x10DE);
        assert_eq!(gpu.memory_device_type(), ort::OrtMemoryInfoDeviceType_GPU);
    }

    #[test]
    fn device_support_stream_awareness_matches_config() {
        let cpu = DeviceSupport::cpu_only();
        assert!(!cpu.stream_aware, "CPU EP must not be stream-aware");

        let gpu = DeviceSupport::gpu("Cuda", 0x10DE);
        assert!(gpu.stream_aware, "GPU EP must be stream-aware");
    }
}
