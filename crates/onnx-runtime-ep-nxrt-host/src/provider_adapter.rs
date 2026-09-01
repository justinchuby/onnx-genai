//! Adapter that wraps a loaded nxrt plugin EP as a `dyn ExecutionProvider`.
//!
//! The adapter holds:
//! - An `Arc`-clone of the plugin's `Library` (structural lifetime safety).
//! - A factory index to create EP instances via the vtable.
//! - The EP vtable pointer (owned, released on Drop).
//!
//! Panics inside the plugin boundary are caught with `std::panic::catch_unwind`
//! and surfaced as errors. Borrowed pointers from the ABI (e.g. tensor dims,
//! op type strings) are copied into owned Rust data before the callback returns.

use std::ffi::CStr;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::Arc;

use libloading::Library;
use onnx_runtime_ep_api::{
    DeviceBuffer, EpConfig, EpError, ExecutionProvider, Fence, Kernel, KernelMatch, Result,
};
use onnx_runtime_ep_nxrt_abi::NxrtEpVtable;
use onnx_runtime_ir::{DataType, DeviceId, DeviceType, Node, Shape, TensorLayout};

use crate::error::NxrtHostError;
use crate::loader::NxrtPlugin;

/// A loaded nxrt plugin exposed as a native `ExecutionProvider`.
///
/// # Lifetime invariant
///
/// `_library` (an `Arc<Library>`) ensures the library remains loaded while
/// `ep_vtable` (which points into the library's address space) is live.
pub struct NxrtExecutionProvider {
    /// Shared library handle — prevents unload while this EP exists.
    _library: Arc<Library>,
    /// EP vtable owned by this struct. Released via `release` on Drop.
    ep_vtable: *mut NxrtEpVtable,
    /// Cached EP name (copied from the borrowed vtable pointer).
    name: String,
    /// The plugin descriptor, kept alive for the factory set.
    _plugin: NxrtPlugin,
    /// Whether `initialize` has been called.
    initialized: bool,
}

// Safety: The nxrt ABI contract requires EP vtables be thread-safe.
unsafe impl Send for NxrtExecutionProvider {}
unsafe impl Sync for NxrtExecutionProvider {}

impl NxrtExecutionProvider {
    /// Create a new nxrt EP from a loaded plugin using factory at `factory_index`.
    ///
    /// Calls the factory's `create_ep` vtable entry for device ordinal 0.
    pub fn new(
        plugin: &NxrtPlugin,
        factory_index: usize,
    ) -> std::result::Result<Self, NxrtHostError> {
        Self::with_device(plugin, factory_index, 0)
    }

    /// Create a new nxrt EP from a loaded plugin for a specific device ordinal.
    pub fn with_device(
        plugin: &NxrtPlugin,
        factory_index: usize,
        device_ordinal: u32,
    ) -> std::result::Result<Self, NxrtHostError> {
        let factory_ptr =
            plugin
                .factory(factory_index)
                .ok_or_else(|| NxrtHostError::FactoryFailed {
                    path: plugin.path.clone().to_path_buf(),
                    status: format!(
                        "factory index {factory_index} out of range (plugin has {} factories)",
                        plugin.num_factories()
                    ),
                })?;

        // Validate struct_size covers the create_ep field before dereferencing.
        let factory_struct_size = unsafe { (*factory_ptr).struct_size } as usize;
        let create_ep_end =
            std::mem::offset_of!(onnx_runtime_ep_nxrt_abi::NxrtEpFactoryVtable, create_ep)
                + std::mem::size_of::<
                    unsafe extern "C" fn(
                        *mut std::ffi::c_void,
                        u32,
                        *mut *mut onnx_runtime_ep_nxrt_abi::NxrtEpVtable,
                    )
                        -> onnx_runtime_ep_nxrt_abi::NxrtStatus,
                >();
        if factory_struct_size < create_ep_end {
            return Err(NxrtHostError::FactoryFailed {
                path: plugin.path.clone().to_path_buf(),
                status: format!(
                    "factory struct_size ({factory_struct_size}) too small to contain create_ep \
                     (need at least {create_ep_end}). Plugin may be an older ABI version."
                ),
            });
        }

        let mut ep_ptr: *mut NxrtEpVtable = std::ptr::null_mut();
        let status =
            unsafe { ((*factory_ptr).create_ep)((*factory_ptr).ctx, device_ordinal, &mut ep_ptr) };

        if !status.is_ok() {
            let msg = status.message_str().unwrap_or("(no message)").to_owned();
            return Err(NxrtHostError::FactoryFailed {
                path: plugin.path.clone().to_path_buf(),
                status: msg,
            });
        }

        if ep_ptr.is_null() {
            return Err(NxrtHostError::FactoryFailed {
                path: plugin.path.clone().to_path_buf(),
                status: "create_ep returned Ok but EP pointer is null".into(),
            });
        }

        // Validate EP struct_size covers minimum required fields.
        let ep_struct_size = unsafe { (*ep_ptr).struct_size } as usize;
        let ep_min_size = std::mem::size_of::<NxrtEpVtable>();
        if ep_struct_size < ep_min_size {
            // Release the EP since we can't safely use it.
            let _ = catch_unwind(AssertUnwindSafe(|| {
                unsafe { ((*ep_ptr).release)((*ep_ptr).ctx) };
            }));
            return Err(NxrtHostError::FactoryFailed {
                path: plugin.path.clone().to_path_buf(),
                status: format!(
                    "EP struct_size ({ep_struct_size}) smaller than expected ({ep_min_size}). \
                     Plugin may be an older ABI version."
                ),
            });
        }

        // Copy the EP name from the borrowed pointer into owned data.
        let name = unsafe {
            let ep = &*ep_ptr;
            if ep.name.is_null() {
                plugin.name().to_owned()
            } else {
                CStr::from_ptr(ep.name as *const std::os::raw::c_char)
                    .to_string_lossy()
                    .into_owned()
            }
        };

        Ok(Self {
            _library: Arc::clone(&plugin.library),
            ep_vtable: ep_ptr,
            name,
            _plugin: plugin.clone(),
            initialized: false,
        })
    }
}

impl Drop for NxrtExecutionProvider {
    fn drop(&mut self) {
        if !self.ep_vtable.is_null() {
            let vtable = self.ep_vtable;
            let _ = catch_unwind(AssertUnwindSafe(|| {
                // Validate struct_size covers release+ctx before calling.
                // If the vtable is too small, we deliberately leak rather than
                // jump through a bogus pointer — UB/arbitrary code execution is
                // far worse than a resource leak.
                let release_end = std::mem::offset_of!(NxrtEpVtable, release)
                    + std::mem::size_of::<unsafe extern "C" fn(*mut std::ffi::c_void)>();
                let ctx_end = std::mem::offset_of!(NxrtEpVtable, ctx)
                    + std::mem::size_of::<*mut std::ffi::c_void>();
                let min_size = release_end.max(ctx_end);
                let struct_size = unsafe { (*vtable).struct_size } as usize;
                if struct_size < min_size {
                    // Leak deliberately: jumping through a bogus pointer is
                    // arbitrary code execution; leaking is merely wasteful.
                    eprintln!(
                        "WARNING: EP vtable struct_size ({struct_size}) too small to contain \
                         release/ctx ({min_size}), skipping release (leaking)"
                    );
                    return;
                }
                // SAFETY: We own the EP vtable per the ABI contract.
                unsafe { ((*vtable).release)((*vtable).ctx) };
            }));
            self.ep_vtable = std::ptr::null_mut();
        }
    }
}

impl ExecutionProvider for NxrtExecutionProvider {
    fn name(&self) -> &str {
        &self.name
    }

    fn device_type(&self) -> DeviceType {
        if self.ep_vtable.is_null() {
            return DeviceType::Custom(0);
        }
        let dt = unsafe { (*self.ep_vtable).device_type };
        match dt {
            0 => DeviceType::Cpu,
            1 => DeviceType::Cuda,
            2 => DeviceType::Rocm,
            3 => DeviceType::CoreMl,
            4 => DeviceType::Mlx,
            5 => DeviceType::WebGpu,
            6 => DeviceType::Qnn,
            7 => DeviceType::OpenVino,
            other => DeviceType::Custom(other & 0xFFF),
        }
    }

    fn device_id(&self) -> DeviceId {
        if self.ep_vtable.is_null() {
            return DeviceId::new(DeviceType::Custom(0), 0);
        }
        let dt = self.device_type();
        let id = unsafe { (*self.ep_vtable).device_id };
        DeviceId::new(dt, id)
    }

    fn initialize(&mut self, _config: &EpConfig) -> Result<()> {
        self.initialized = true;
        Ok(())
    }

    fn shutdown(&mut self) -> Result<()> {
        self.initialized = false;
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
        // Conservative: decline until capability query is fully wired.
        KernelMatch::Unsupported {
            reason: "nxrt capability query not yet wired through vtable".into(),
        }
    }

    fn get_kernel(
        &self,
        _op: &Node,
        _shapes: &[Vec<usize>],
        _opset: u64,
    ) -> Result<Box<dyn Kernel>> {
        Err(EpError::KernelFailed(
            "nxrt kernel dispatch not yet wired through vtable".into(),
        ))
    }

    fn allocate(&self, _size: usize, _alignment: usize) -> Result<DeviceBuffer> {
        Err(EpError::KernelFailed(
            "nxrt allocate not yet wired through vtable".into(),
        ))
    }

    fn deallocate(&self, _buffer: DeviceBuffer) -> Result<()> {
        Err(EpError::KernelFailed(
            "nxrt deallocate not yet wired through vtable".into(),
        ))
    }

    fn copy(&self, _src: &DeviceBuffer, _dst: &mut DeviceBuffer, _size: usize) -> Result<()> {
        Err(EpError::KernelFailed(
            "nxrt copy not yet wired through vtable".into(),
        ))
    }

    fn copy_async(
        &self,
        _src: &DeviceBuffer,
        _dst: &mut DeviceBuffer,
        _size: usize,
    ) -> Result<Fence> {
        Err(EpError::KernelFailed(
            "nxrt copy_async not yet wired through vtable".into(),
        ))
    }

    fn sync(&self) -> Result<()> {
        Ok(())
    }
}
