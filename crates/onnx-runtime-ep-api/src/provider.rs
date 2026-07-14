//! The [`ExecutionProvider`] trait and its supporting types (§4.1).

use onnx_runtime_ir::{DeviceId, DeviceType, Graph, Node, NodeId, Shape, TensorLayout};

use crate::error::Result;
use crate::kernel::{Kernel, KernelMatch};

/// Index of an EP within an [`crate::registry::EpRegistry`].
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct EpId(pub u32);

/// Opaque, namespaced configuration passed to [`ExecutionProvider::initialize`].
#[derive(Clone, Debug, Default)]
pub struct EpConfig {
    /// Namespaced key/value options (e.g. `"cuda.arena_extend_strategy"`).
    pub options: std::collections::HashMap<String, String>,
}

/// A handle to a device allocation. Concrete EPs define the backing pointer.
#[derive(Debug)]
pub struct DeviceBuffer {
    pub device: DeviceId,
    pub size: usize,
    /// Raw device pointer (host pointer for CPU). Interpreted by the owning EP.
    pub ptr: *mut std::ffi::c_void,
}

// SAFETY (Phase 1 placeholder): device buffers are owned and synchronized by
// their EP; the concrete EP crates uphold Send/Sync. Documented as a downstream
// review item.
unsafe impl Send for DeviceBuffer {}
unsafe impl Sync for DeviceBuffer {}

/// A synchronization fence returned by async operations.
#[derive(Debug, Default)]
pub struct Fence {
    pub id: u64,
}

/// Marker for an EP exported as an ORT-compatible C ABI plugin (Phase 2).
#[derive(Debug, Default)]
pub struct OrtPluginExport {
    pub register_symbol: String,
}

/// An EP-specific optimization pass.
///
/// Placeholder trait: the full pass pipeline lives in `onnx-runtime-optimizer`
/// (Phase 2). Defined here so [`ExecutionProvider::custom_passes`] can name it
/// without a Phase 2 crate dependency.
pub trait OptimizerPass: Send + Sync {
    fn name(&self) -> &str;
}

/// The core EP interface. Every backend crate implements this (§4.1).
pub trait ExecutionProvider: Send + Sync {
    /// EP identifier (snake_case, e.g. `"cpu_ep"`, `"cuda_ep"`).
    fn name(&self) -> &str;

    fn device_type(&self) -> DeviceType;
    fn device_id(&self) -> DeviceId;

    /// Initialize device resources / load libraries.
    fn initialize(&mut self, config: &EpConfig) -> Result<()>;
    /// Release device resources.
    fn shutdown(&mut self) -> Result<()>;

    /// Whether this EP can run `op` with the given input shapes and layouts,
    /// and at what cost.
    fn supports_op(&self, op: &Node, shapes: &[Shape], layouts: &[TensorLayout]) -> KernelMatch;

    /// Get or create a kernel for `op` specialized to concrete `shapes`.
    fn get_kernel(&self, op: &Node, shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>>;

    /// Allocate device memory.
    fn allocate(&self, size: usize, alignment: usize) -> Result<DeviceBuffer>;
    /// Free device memory.
    fn deallocate(&self, buffer: DeviceBuffer) -> Result<()>;

    /// Synchronous copy (host↔device or device↔device).
    fn copy(&self, src: &DeviceBuffer, dst: &mut DeviceBuffer, size: usize) -> Result<()>;
    /// Asynchronous copy; returns a [`Fence`] to await.
    fn copy_async(&self, src: &DeviceBuffer, dst: &mut DeviceBuffer, size: usize) -> Result<Fence>;

    /// Block until all pending work on this EP completes.
    fn sync(&self) -> Result<()>;

    /// Export this EP as an ORT C ABI plugin, if supported (Phase 2).
    fn as_ort_plugin(&self) -> Option<OrtPluginExport> {
        None
    }

    /// EP-specific optimization passes, run after the generic optimizer.
    fn custom_passes(&self) -> Vec<Box<dyn OptimizerPass>> {
        Vec::new()
    }

    /// Nodes this EP claims unconditionally (bypassing cost-model placement).
    fn claim_nodes(&self, graph: &Graph) -> Vec<NodeId> {
        let _ = graph;
        Vec::new()
    }
}
