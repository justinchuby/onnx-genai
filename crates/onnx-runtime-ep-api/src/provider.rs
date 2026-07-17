//! The [`ExecutionProvider`] trait and its supporting types (§4.1).

use std::ffi::c_void;
use std::ptr::NonNull;

use onnx_runtime_ir::{DeviceId, DeviceType, Graph, Node, NodeId, Shape, TensorLayout};

use crate::epcontext::EpContext;
use crate::error::{EpError, Result};
use crate::kernel::{Kernel, KernelMatch};
use crate::weight::ExecutionProviderCapabilities;

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
    /// Whether this handle *owns* the pointed-to allocation.
    ///
    /// [`BufferOwner::Owned`] (the default for [`DeviceBuffer::from_raw_parts`])
    /// is the original contract: the owning EP must free it exactly once in
    /// `deallocate`. [`BufferOwner::Borrowed`] (from
    /// [`DeviceBuffer::from_borrowed_parts`]) aliases memory owned by *someone
    /// else* (e.g. an mmap'd weight file) — `deallocate` must **not** free it
    /// and it must never be written through.
    owner: BufferOwner,
}

/// Whether a [`DeviceBuffer`] owns the allocation it names, or merely borrows
/// (aliases) memory owned elsewhere.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BufferOwner {
    /// This handle is the sole owner; the owning EP frees it in `deallocate`.
    Owned,
    /// This handle aliases foreign memory (e.g. an mmap). `deallocate` must be
    /// a no-op free; the real owner must outlive the buffer and every use of it.
    Borrowed,
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
            owner: BufferOwner::Owned,
        }
    }

    /// Wrap **foreign, borrowed** memory in a non-owning `DeviceBuffer`.
    ///
    /// Unlike [`DeviceBuffer::from_raw_parts`], the returned handle does **not**
    /// own the allocation: it aliases memory owned by someone else (for example
    /// a `memmap2::Mmap` over an on-disk weight file). This lets an EP reference
    /// initializer bytes zero-copy instead of allocating + copying them into
    /// fresh RAM.
    ///
    /// [`is_borrowed`](DeviceBuffer::is_borrowed) returns `true`, and the owning
    /// EP's `deallocate` must treat it as a **no-op free** (the guard checks
    /// `is_borrowed()`). [`into_raw`](DeviceBuffer::into_raw) still yields the
    /// raw pointer, but the caller must **not** free it.
    ///
    /// # Safety
    ///
    /// The caller must guarantee all of:
    /// * `ptr` is non-null and points to the start of a readable region of at
    ///   least `size` bytes on `device`, aligned to at least `align` bytes.
    /// * The memory is owned by another object (e.g. an mmap) that **outlives
    ///   this buffer and every use of it** (read via `as_ptr`). Nothing else may
    ///   free or unmap it while this handle or any alias derived from it lives.
    /// * The buffer is treated as **read-only**: it is never written through
    ///   (`as_mut_ptr` must not be used to mutate borrowed memory) and is never
    ///   passed to an EP's `deallocate` expecting a free — `deallocate` skips
    ///   the free for borrowed buffers.
    ///
    /// `align` must be a power of two (checked in debug builds).
    pub unsafe fn from_borrowed_parts(
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
            ptr: NonNull::new(ptr).expect("DeviceBuffer::from_borrowed_parts: null pointer"),
            owner: BufferOwner::Borrowed,
        }
    }

    /// Whether this handle merely *borrows* (aliases) foreign memory rather than
    /// owning it. A borrowed buffer must never be freed by `deallocate`.
    pub fn is_borrowed(&self) -> bool {
        matches!(self.owner, BufferOwner::Borrowed)
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

    /// Consume the handle, returning the raw pointer *without* freeing it. For
    /// an owned buffer the caller assumes the single-free obligation from
    /// [`DeviceBuffer::from_raw_parts`]. For a **borrowed** buffer (see
    /// [`DeviceBuffer::from_borrowed_parts`]) the pointer must **not** be freed;
    /// check [`is_borrowed`](DeviceBuffer::is_borrowed) first if the caller
    /// intends to free.
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

/// The core EP interface. Every backend crate implements this (§4.1).
pub trait ExecutionProvider: Send + Sync {
    /// EP identifier (snake_case, e.g. `"cpu_ep"`, `"cuda_ep"`).
    fn name(&self) -> &str;

    fn device_type(&self) -> DeviceType;
    fn device_id(&self) -> DeviceId;

    /// Optional executor-to-EP capabilities. Stock EPs advertise none and
    /// continue receiving resident [`crate::TensorView`] inputs.
    fn capabilities(&self) -> ExecutionProviderCapabilities {
        ExecutionProviderCapabilities::stock()
    }

    /// Initialize device resources / load libraries.
    fn initialize(&mut self, config: &EpConfig) -> Result<()>;
    /// Release device resources.
    fn shutdown(&mut self) -> Result<()>;

    /// Whether this EP can run `op` at the model's effective `opset` with the
    /// given input shapes and layouts, and at what cost.
    ///
    /// Every [`KernelMatch::Unsupported`] result must carry an actionable reason:
    /// state what the EP accepts and, where possible, how to fix the model or
    /// registration rather than returning a bare decline.
    fn supports_op(
        &self,
        op: &Node,
        opset: u64,
        shapes: &[Shape],
        layouts: &[TensorLayout],
    ) -> KernelMatch;

    /// Get or create a kernel for `op` specialized to concrete `shapes`.
    ///
    /// `opset` is the effective operator-set version for `op`'s domain in the
    /// owning graph. EPs use it to select opset-specialized kernels (e.g. the
    /// opset-13 per-axis vs. the legacy opset-<13 2D-coercion `Softmax`).
    fn get_kernel(&self, op: &Node, shapes: &[Vec<usize>], opset: u64) -> Result<Box<dyn Kernel>>;

    /// Allocate device memory.
    fn allocate(&self, size: usize, alignment: usize) -> Result<DeviceBuffer>;
    /// Free device memory.
    fn deallocate(&self, buffer: DeviceBuffer) -> Result<()>;

    /// Synchronous copy (host↔device or device↔device).
    fn copy(&self, src: &DeviceBuffer, dst: &mut DeviceBuffer, size: usize) -> Result<()>;
    /// Asynchronous copy; returns a [`Fence`] to await.
    fn copy_async(&self, src: &DeviceBuffer, dst: &mut DeviceBuffer, size: usize) -> Result<Fence>;

    /// Synchronously upload host bytes into a buffer owned by this EP.
    fn copy_from_host(&self, src: &[u8], dst: &mut DeviceBuffer) -> Result<()> {
        if !dst.device().is_host_accessible() {
            return Err(EpError::KernelFailed(format!(
                "{}: host upload is not implemented for device {:?}",
                self.name(),
                dst.device()
            )));
        }
        if src.len() > dst.len() {
            return Err(EpError::KernelFailed(format!(
                "{}: host upload of {} bytes exceeds destination {} bytes",
                self.name(),
                src.len(),
                dst.len()
            )));
        }
        if src.is_empty() {
            return Ok(());
        }
        // SAFETY: host accessibility is checked above, `dst` is uniquely
        // borrowed, and its allocation is at least `src.len()` bytes.
        unsafe {
            std::ptr::copy_nonoverlapping(src.as_ptr(), dst.as_mut_ptr().cast(), src.len());
        }
        Ok(())
    }

    /// Synchronously upload host bytes into a byte range of a buffer owned by
    /// this EP.
    fn copy_from_host_at(
        &self,
        src: &[u8],
        dst: &mut DeviceBuffer,
        byte_offset: usize,
    ) -> Result<()> {
        let end = byte_offset.checked_add(src.len()).ok_or_else(|| {
            EpError::KernelFailed(format!("{}: host upload range overflows", self.name()))
        })?;
        if end > dst.len() {
            return Err(EpError::KernelFailed(format!(
                "{}: host upload range {byte_offset}..{end} exceeds destination {} bytes",
                self.name(),
                dst.len()
            )));
        }
        if src.is_empty() {
            return Ok(());
        }
        if !dst.device().is_host_accessible() {
            return Err(EpError::KernelFailed(format!(
                "{}: ranged host upload is not implemented for device {:?}",
                self.name(),
                dst.device()
            )));
        }
        // SAFETY: host accessibility and bounds are checked above, and `dst` is
        // uniquely borrowed for the duration of the copy.
        unsafe {
            std::ptr::copy_nonoverlapping(
                src.as_ptr(),
                dst.as_mut_ptr().cast::<u8>().add(byte_offset),
                src.len(),
            );
        }
        Ok(())
    }

    /// Synchronously download a buffer owned by this EP into host bytes.
    fn copy_to_host(&self, src: &DeviceBuffer, dst: &mut [u8]) -> Result<()> {
        if !src.device().is_host_accessible() {
            return Err(EpError::KernelFailed(format!(
                "{}: host download is not implemented for device {:?}",
                self.name(),
                src.device()
            )));
        }
        if dst.len() > src.len() {
            return Err(EpError::KernelFailed(format!(
                "{}: host download of {} bytes exceeds source {} bytes",
                self.name(),
                dst.len(),
                src.len()
            )));
        }
        if dst.is_empty() {
            return Ok(());
        }
        // SAFETY: host accessibility is checked above, `dst` is uniquely
        // borrowed, and `src` contains at least `dst.len()` readable bytes.
        unsafe {
            std::ptr::copy_nonoverlapping(src.as_ptr().cast(), dst.as_mut_ptr(), dst.len());
        }
        Ok(())
    }

    /// Block until all pending work on this EP completes.
    fn sync(&self) -> Result<()>;

    /// Export this EP as an ORT C ABI plugin, if supported (Phase 2).
    fn as_ort_plugin(&self) -> Option<OrtPluginExport> {
        None
    }

    /// EP-specific optimization passes, run after the generic optimizer.
    fn custom_passes(&self) -> Vec<Box<dyn onnx_runtime_optimizer::OptimizationPass>> {
        Vec::new()
    }

    /// Nodes this EP claims unconditionally (bypassing cost-model placement).
    fn claim_nodes(&self, graph: &Graph) -> Vec<NodeId> {
        let _ = graph;
        Vec::new()
    }

    /// The `EPContext` node `source` key(s) this EP accepts for compiled-context
    /// dispatch (`docs/ORT2.md` §55.6). The keys come from the EP's own
    /// config/data — **never** hardcoded in loader/session dispatch. An empty
    /// list (the default) means the EP does not participate in `EPContext`
    /// (e.g. the pure-Rust CPU EP has no compile step).
    fn context_source_keys(&self) -> Vec<String> {
        Vec::new()
    }

    /// Produce the runtime [`EpContext`] for this EP's freshly compiled subgraph
    /// (the §55.4 dump path calls this). Default: unsupported — an EP with no
    /// compile step returns [`EpError::UnsupportedContext`].
    fn save_context(&self) -> Result<EpContext> {
        Err(EpError::UnsupportedContext {
            ep: self.name().to_string(),
        })
    }

    /// Restore this EP from a runtime [`EpContext`], skipping convert+compile
    /// (the §55.3 load path calls this). Default: unsupported — an EP that does
    /// not consume `EPContext` returns [`EpError::UnsupportedContext`].
    fn load_context(&self, ctx: &EpContext) -> Result<()> {
        let _ = ctx;
        Err(EpError::UnsupportedContext {
            ep: self.name().to_string(),
        })
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

    #[test]
    fn owned_buffer_is_not_borrowed() {
        let buf = host_alloc(32, 16);
        assert!(
            !buf.is_borrowed(),
            "from_raw_parts must produce an owned buffer"
        );
        host_free(buf);
    }

    /// A borrowed buffer aliases memory owned by someone else (here a `Vec`):
    /// it reports `is_borrowed()`, exposes the aliased pointer, and consuming it
    /// via `into_raw` must NOT free the backing — the `Vec` stays valid.
    #[test]
    fn borrowed_buffer_aliases_without_owning() {
        let mut backing = vec![7u8; 64];
        let ptr = backing.as_mut_ptr() as *mut c_void;
        // SAFETY: `ptr`/`len` name `backing`'s live allocation (aligned to 1);
        // `backing` outlives the buffer and every use below, and we never write
        // through the borrowed handle.
        let buf = unsafe { DeviceBuffer::from_borrowed_parts(ptr, DeviceId::cpu(), 64, 1) };
        assert!(buf.is_borrowed());
        assert_eq!(buf.len(), 64);
        assert_eq!(buf.as_ptr(), ptr as *const c_void);
        // Consume without freeing: `into_raw` must never free a borrowed buffer.
        let raw = buf.into_raw();
        assert_eq!(raw, ptr);
        // `backing` is still fully valid — a free would be a use-after-free here.
        assert!(backing.iter().all(|&b| b == 7));
        backing[0] = 9;
        assert_eq!(backing[0], 9);
    }
}
