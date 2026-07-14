//! The [`ExecutionProvider`] trait and its supporting types (§4.1).

use std::ffi::c_void;
use std::ptr::NonNull;

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

/// An owning handle to a single device allocation.
///
/// # Ownership & lifetime
///
/// A `DeviceBuffer` is the **sole owner** of the allocation it names. It is
/// produced only by [`ExecutionProvider::allocate`] and released only by
/// [`ExecutionProvider::deallocate`], which consumes it *by value*. The owning
/// EP is both allocator and deallocator: the buffer records the [`DeviceId`]
/// (hence which EP instance) that may free it, so a buffer must never be handed
/// to a different EP. Ownership is unique — no two `DeviceBuffer`s ever alias
/// the same allocation.
///
/// # No `Drop`
///
/// `DeviceBuffer` deliberately does **not** implement [`Drop`]. Freeing device
/// memory generally needs the EP's context/stream (a CUDA context, an MLX
/// queue, an allocator arena) that this bare handle does not carry, so a silent
/// drop could not free correctly. Consequences:
/// * Dropping a `DeviceBuffer` without passing it to `deallocate` **leaks** the
///   allocation. It can never *double-free*, which is the memory-safety
///   property we prioritize (plan §4.4).
/// * The session layer owns the discipline of pairing every `allocate` with
///   exactly one `deallocate`. Higher layers may wrap this handle in an
///   RAII/`Arc` type that calls back into the EP; that policy lives above the
///   EP contract, not here.
///
/// # Access
///
/// The base address is reachable only through [`DeviceBuffer::as_ptr`]
/// (shared) and [`DeviceBuffer::as_mut_ptr`] (unique). Obtaining a pointer is
/// safe; *dereferencing* it is `unsafe` and valid only on host-accessible
/// devices ([`DeviceType::is_host_accessible`]) within the owning EP's context.
///
/// # Thread-safety
///
/// See the `Send`/`Sync` impls below for the exact invariant.
#[derive(Debug)]
pub struct DeviceBuffer {
    device: DeviceId,
    size: usize,
    align: usize,
    /// Non-null base address of the allocation. For CPU and MLX unified memory
    /// this is a dereferenceable host pointer; for CUDA/ROCm it is an opaque
    /// device address only meaningful inside the owning EP's context.
    ptr: NonNull<c_void>,
}

impl DeviceBuffer {
    /// Wrap a raw device allocation in an owning handle.
    ///
    /// # Safety
    ///
    /// The caller (the owning EP) must guarantee all of:
    /// * `ptr` is non-null and points to the start of an allocation of at least
    ///   `size` bytes on `device`, aligned to at least `align` bytes.
    /// * The allocation was produced by `device`'s EP and will be freed exactly
    ///   once, only by returning this handle to that EP's `deallocate` (or via
    ///   an equivalent raw free of the pointer obtained from
    ///   [`DeviceBuffer::into_raw`]).
    /// * No other live `DeviceBuffer` aliases the same allocation.
    ///
    /// `align` must be a power of two (checked in debug builds).
    pub unsafe fn from_raw_parts(
        ptr: *mut c_void,
        device: DeviceId,
        size: usize,
        align: usize,
    ) -> Self {
        debug_assert!(align.is_power_of_two(), "alignment must be a power of two");
        Self {
            device,
            size,
            align,
            ptr: NonNull::new(ptr).expect("DeviceBuffer::from_raw_parts: null pointer"),
        }
    }

    /// The device this allocation lives on (and whose EP must free it).
    pub fn device(&self) -> DeviceId {
        self.device
    }

    /// Allocation size in bytes.
    pub fn len(&self) -> usize {
        self.size
    }

    /// Whether the allocation is zero-length.
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    /// Alignment (bytes) the base pointer was allocated to.
    pub fn alignment(&self) -> usize {
        self.align
    }

    /// Shared base pointer. Safe to obtain; dereferencing is `unsafe` and only
    /// sound on host-accessible devices within the owning EP's context.
    pub fn as_ptr(&self) -> *const c_void {
        self.ptr.as_ptr()
    }

    /// Unique mutable base pointer. Requires `&mut self` so the borrow checker
    /// forbids two writers sharing one buffer — this is what makes the `Sync`
    /// impl sound (a shared `&DeviceBuffer` can never hand out a writable
    /// pointer through safe code).
    pub fn as_mut_ptr(&mut self) -> *mut c_void {
        self.ptr.as_ptr()
    }

    /// Consume the handle, returning the raw pointer *without* freeing it. The
    /// caller assumes the single-free obligation from [`DeviceBuffer::from_raw_parts`].
    pub fn into_raw(self) -> *mut c_void {
        self.ptr.as_ptr()
    }
}

// SAFETY: `DeviceBuffer` is an owning *handle* — it stores only a base address
// plus metadata and exposes no safe way to read or write the pointed-to memory
// (all access goes through `as_ptr`/`as_mut_ptr`, which are safe to *call* but
// `unsafe` to *use*). Moving the handle to another thread transfers ownership of
// the address; this is sound for every allocator we target — host `malloc`,
// CUDA device pointers, and MLX unified memory are all address-portable and not
// thread-affine at the pointer level. Any data race on the *contents* is
// prevented one layer up by `&`/`&mut` aliasing on `TensorView`/`TensorMut` and
// by the scheduler, not by this type. If a future EP wires a genuinely
// thread-affine allocator, it must wrap the handle in a non-`Send` owner rather
// than weaken this invariant (plan §4.4 flags this for a dedicated review when
// ep-cpu lands real memory).
unsafe impl Send for DeviceBuffer {}
// SAFETY: `&DeviceBuffer` grants no interior mutability — it can only produce a
// `*const` via `as_ptr` (a plain address copy) and read `Copy` metadata, so
// concurrent shared reads of the handle are race-free. Writing requires
// `as_mut_ptr`, which needs `&mut self`; obtaining a writable pointer therefore
// cannot happen through a shared reference in safe code. As with `Send`,
// mutating the underlying memory is gated behind `unsafe` pointer use whose
// synchronization is the caller's responsibility.
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

#[cfg(test)]
mod tests {
    use super::*;

    fn _assert_send_sync<T: Send + Sync>() {}

    /// Leak a boxed byte slice as a stand-in host allocation.
    fn host_alloc(size: usize, align: usize) -> DeviceBuffer {
        let boxed = vec![0u8; size].into_boxed_slice();
        let ptr = Box::into_raw(boxed) as *mut c_void;
        // SAFETY: `ptr` is a valid, unique, non-null allocation of `size` bytes
        // on the host, aligned to the allocator's guarantee (>= 1); we treat the
        // CPU EP as its owner and free it exactly once in `host_free`.
        unsafe { DeviceBuffer::from_raw_parts(ptr, DeviceId::cpu(), size, align) }
    }

    fn host_free(buf: DeviceBuffer) {
        let size = buf.len();
        let ptr = buf.into_raw() as *mut u8;
        // SAFETY: reconstruct the exact `Box<[u8]>` leaked in `host_alloc` so it
        // is freed once. `into_raw` consumed the handle, so no alias remains.
        unsafe {
            drop(Box::from_raw(std::ptr::slice_from_raw_parts_mut(ptr, size)));
        }
    }

    #[test]
    fn device_buffer_is_send_sync() {
        _assert_send_sync::<DeviceBuffer>();
    }

    #[test]
    fn buffer_metadata_and_single_free() {
        let mut buf = host_alloc(128, 64);
        assert_eq!(buf.len(), 128);
        assert!(!buf.is_empty());
        assert_eq!(buf.alignment(), 64);
        assert_eq!(buf.device(), DeviceId::cpu());
        assert!(!buf.as_ptr().is_null());
        assert!(!buf.as_mut_ptr().is_null());
        // Single free path — a double free here would trip ASan/Miri.
        host_free(buf);
    }

    #[test]
    fn buffer_moves_across_thread() {
        let buf = host_alloc(64, 16);
        let base = buf.as_ptr() as usize;
        let handle = std::thread::spawn(move || {
            assert_eq!(buf.len(), 64);
            assert_eq!(buf.as_ptr() as usize, base);
            buf // hand ownership back so the main thread frees it once
        });
        let buf = handle.join().unwrap();
        host_free(buf);
    }
}
