//! Adapter that wraps a loaded nxrt plugin EP as a `dyn ExecutionProvider`.
//!
//! The adapter holds:
//! - An `Arc`-clone of the plugin's `Library` (structural lifetime safety).
//! - The opaque EP handle obtained from the plugin's factory.
//! - Cached metadata (name, device count).
//!
//! Panics inside the plugin boundary are caught with `std::panic::catch_unwind`
//! and surfaced as `Err(EpError::EpPanicked)`.

use std::ffi::CString;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;

use libloading::Library;
use onnx_runtime_ep_api::{
    DeviceBuffer, EpConfig, EpError, ExecutionProvider, Fence, Kernel, KernelMatch, Result,
};
use onnx_runtime_ir::{DataType, DeviceId, DeviceType, Node, Shape, TensorLayout};

use crate::abi_contract::NxrtEpHandle;
use crate::error::NxrtHostError;
use crate::loader::{NxrtPlugin, create_ep_instance, destroy_ep_instance};

/// A loaded nxrt plugin exposed as a native `ExecutionProvider`.
///
/// # Lifetime invariant
///
/// `_library` (an `Arc<Library>`) is listed first and thus dropped last,
/// ensuring the library remains loaded while `handle` (which points into the
/// library's address space) is live. The `Arc` is shared with the originating
/// [`NxrtPlugin`], so the library is not unloaded until both the plugin
/// descriptor and all EP instances are dropped.
pub struct NxrtExecutionProvider {
    /// Shared library handle — prevents unload while this EP exists.
    _library: Arc<Library>,
    /// Opaque EP handle from the plugin's factory.
    handle: *mut NxrtEpHandle,
    /// Cached EP name for the trait method.
    name: String,
    /// The plugin descriptor, kept alive for destroy.
    plugin: NxrtPlugin,
    /// Whether `initialize` has been called.
    initialized: bool,
}

// Safety: The nxrt ABI contract requires that EP handles be thread-safe (the
// host may call from any thread). The library backing is reference-counted.
unsafe impl Send for NxrtExecutionProvider {}
unsafe impl Sync for NxrtExecutionProvider {}

impl NxrtExecutionProvider {
    /// Create a new nxrt EP from a loaded plugin.
    ///
    /// Calls the plugin's factory with the given JSON config. Validates that
    /// the plugin advertises at least one device.
    pub fn new(plugin: &NxrtPlugin, config_json: &str) -> std::result::Result<Self, NxrtHostError> {
        let config_cstr = CString::new(config_json).map_err(|_| NxrtHostError::FactoryFailed {
            path: plugin.path.clone(),
            status: "config_json contains interior NUL byte".into(),
        })?;

        let handle = create_ep_instance(plugin, &config_cstr)?;

        Ok(Self {
            _library: Arc::clone(&plugin.library),
            handle,
            name: plugin.name.clone(),
            plugin: plugin.clone(),
            initialized: false,
        })
    }
}

impl Drop for NxrtExecutionProvider {
    fn drop(&mut self) {
        // Best-effort destroy; catch panics at the C boundary.
        let handle = self.handle;
        let plugin = &self.plugin;
        let _ = catch_unwind(AssertUnwindSafe(|| {
            destroy_ep_instance(plugin, handle);
        }));
        self.handle = std::ptr::null_mut();
    }
}

impl ExecutionProvider for NxrtExecutionProvider {
    fn name(&self) -> &str {
        &self.name
    }

    fn device_type(&self) -> DeviceType {
        // nxrt plugins are accelerator plugins; report Custom until the ABI
        // carries a device-type enum.
        DeviceType::Custom(0)
    }

    fn device_id(&self) -> DeviceId {
        DeviceId::new(DeviceType::Custom(0), 0)
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
        // Until the ABI exposes a capability-query vtable, decline everything.
        // The session layer will then fall through to the CPU EP. This is
        // conservative (fail closed): we never claim support we cannot fulfil.
        KernelMatch::Unsupported {
            reason: "nxrt capability query not yet wired through ABI".into(),
        }
    }

    fn get_kernel(&self, _op: &Node, _shapes: &[Vec<usize>], _opset: u64) -> Result<Box<dyn Kernel>> {
        Err(EpError::KernelFailed(
            "nxrt kernel dispatch not yet wired through ABI".into(),
        ))
    }

    fn allocate(&self, _size: usize, _alignment: usize) -> Result<DeviceBuffer> {
        Err(EpError::KernelFailed(
            "nxrt allocate not yet wired through ABI".into(),
        ))
    }

    fn deallocate(&self, _buffer: DeviceBuffer) -> Result<()> {
        Err(EpError::KernelFailed(
            "nxrt deallocate not yet wired through ABI".into(),
        ))
    }

    fn copy(&self, _src: &DeviceBuffer, _dst: &mut DeviceBuffer, _size: usize) -> Result<()> {
        Err(EpError::KernelFailed(
            "nxrt copy not yet wired through ABI".into(),
        ))
    }

    fn copy_async(
        &self,
        _src: &DeviceBuffer,
        _dst: &mut DeviceBuffer,
        _size: usize,
    ) -> Result<Fence> {
        Err(EpError::KernelFailed(
            "nxrt copy_async not yet wired through ABI".into(),
        ))
    }

    fn sync(&self) -> Result<()> {
        Ok(())
    }
}
