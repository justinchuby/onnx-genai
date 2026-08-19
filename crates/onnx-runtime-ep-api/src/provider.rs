//! The [`ExecutionProvider`] trait and its supporting types (§4.1).

use std::ffi::c_void;
use std::ptr::NonNull;

use onnx_runtime_ir::{
    DataType, DeviceId, DeviceType, Graph, GraphView, Node, NodeId, NodeIndex, Shape, TensorLayout,
};
use onnx_runtime_memory_governor::{ManagedAllocation, MemoryLease, MemoryRole, OwningAllocation};

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
/// # Two owning representations
///
/// * **Raw owning** ([`DeviceBuffer::from_raw_parts`]) — an address plus size
///   and alignment. The EP is trusted to pair it with exactly one free. This is
///   what the CPU EP and adapter/plugin paths use.
/// * **Bound owning** ([`DeviceBuffer::from_owning_allocation`]) — the exact
///   [`OwningAllocation`] that minted the address, carrying its binding identity
///   and allocation generation. Final release goes back through that owner, so
///   a stale handle over a reused address cannot free anything, and a release
///   can never be attributed to the wrong mechanism.
///
/// A bound buffer is still non-`Clone` and still has no `Drop`: the generation
/// is what makes the release safe, and the owner's own `Drop` quarantines
/// rather than frees.
///
/// # No `Drop`
///
/// `DeviceBuffer` deliberately does **not** implement [`Drop`]. Freeing device
/// memory generally needs the EP's context/stream (a CUDA context, an MLX
/// queue, an allocator arena) that this bare handle does not carry, so a silent
/// drop could not free correctly. Consequences:
/// * Dropping a `DeviceBuffer` without passing it to `deallocate` **leaks** the
///   allocation (a bound buffer instead quarantines it, so the bytes stay
///   accounted for). It can never *double-free*, which is the memory-safety
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
    /// `deallocate`. [`BufferOwner::Bound`] carries the generation-checked
    /// owner instead of trusting the raw triple. Borrowed handles alias memory
    /// owned by *someone else*. Read-only aliases come from
    /// [`DeviceBuffer::from_borrowed_parts`]; exclusive writable aliases come
    /// from [`DeviceBuffer::from_borrowed_mut_parts`]. `deallocate` must **not**
    /// free either borrowed kind.
    owner: BufferOwner,
}

/// Whether a [`DeviceBuffer`] owns the allocation it names, or merely borrows
/// (aliases) memory owned elsewhere.
#[derive(Debug)]
enum BufferOwner {
    /// This handle is the sole owner; the owning EP frees it in `deallocate`.
    Owned,
    /// This handle is the sole owner *and* carries the binding-issued owner
    /// that minted the address. Release consumes that owner, so it is validated
    /// against the binding identity and the allocation generation.
    Bound(Box<OwningAllocation>),
    /// Binding-issued ownership whose authority/process charges remain pinned
    /// until the Phase-4 structured release outcome settles.
    Managed(Box<ManagedAllocation>),
    /// This handle aliases foreign memory (e.g. an mmap). `deallocate` must be
    /// a no-op free; the real owner must outlive the buffer and every use of it.
    Borrowed,
    /// This handle has temporary exclusive write access to an allocation owned
    /// elsewhere. Deallocation remains a no-op.
    BorrowedMut,
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

    /// Wrap foreign memory in a non-owning, exclusively writable buffer handle.
    ///
    /// This is intended for persistent external output bindings: the real owner
    /// retains the allocation while an executor temporarily writes through this
    /// alias.
    ///
    /// # Safety
    ///
    /// The caller must guarantee all of:
    /// * `ptr` names a non-null writable allocation of at least `size` bytes on
    ///   `device`, aligned to at least `align` bytes.
    /// * The real owner outlives this handle and every operation using it.
    /// * No other writer accesses the allocation while this handle is live.
    /// * This handle is never used to free the allocation; `deallocate` treats
    ///   it as borrowed.
    pub unsafe fn from_borrowed_mut_parts(
        ptr: *mut c_void,
        device: DeviceId,
        size: usize,
        align: usize,
    ) -> Option<Self> {
        debug_assert!(align.is_power_of_two(), "alignment must be a power of two");
        Some(Self {
            device,
            size,
            align,
            ptr: NonNull::new(ptr)?,
            owner: BufferOwner::BorrowedMut,
        })
    }

    /// Whether this handle merely *borrows* (aliases) foreign memory rather than
    /// owning it. A borrowed buffer must never be freed by `deallocate`.
    pub fn is_borrowed(&self) -> bool {
        matches!(self.owner, BufferOwner::Borrowed | BufferOwner::BorrowedMut)
    }

    /// Wrap a **binding-issued owning allocation** in a `DeviceBuffer`.
    ///
    /// The buffer's address, size, and alignment are taken from `owner`, so a
    /// bound buffer can never describe a different region than the owner it
    /// carries. Final release consumes that owner
    /// ([`into_bound_owner`](Self::into_bound_owner)), which matches the binding
    /// identity and the allocation generation before anything is freed: a stale
    /// handle over a reused device address is refused instead of freeing a live
    /// allocation.
    ///
    /// This is safe to call — the safety obligations were discharged when the
    /// binding issued (or adopted) the allocation.
    pub fn from_owning_allocation(owner: OwningAllocation, device: DeviceId) -> Self {
        let ptr = owner.as_ptr();
        let size = owner.len();
        let align = owner.alignment().max(1);
        Self {
            device,
            size,
            align,
            ptr: NonNull::new(ptr.as_ptr().cast::<c_void>())
                .expect("an owning allocation holds a non-null address"),
            owner: BufferOwner::Bound(Box::new(owner)),
        }
    }

    /// Wrap a process-manager transaction result in a device buffer.
    ///
    /// The manager settlement token stays inseparable from physical ownership;
    /// consuming release must use [`into_bound_ownership`](Self::into_bound_ownership).
    pub fn from_managed_allocation(owner: ManagedAllocation, device: DeviceId) -> Self {
        let ptr = owner.as_ptr();
        let size = owner.len();
        let align = owner.alignment().max(1);
        Self {
            device,
            size,
            align,
            ptr: NonNull::new(ptr.as_ptr().cast::<c_void>())
                .expect("a managed allocation holds a non-null address"),
            owner: BufferOwner::Managed(Box::new(owner)),
        }
    }

    /// Whether this handle carries a binding-issued owner whose release is
    /// generation-validated.
    pub fn is_bound(&self) -> bool {
        matches!(self.owner, BufferOwner::Bound(_) | BufferOwner::Managed(_))
    }

    /// Borrow the binding-issued owner, for a bound capability call (commit,
    /// decommit, mapped-byte queries) that must be validated against the
    /// allocation generation.
    ///
    /// Returns `None` for raw-owning and borrowed buffers, which is how a
    /// generation-checked path fails closed on foreign memory.
    pub fn bound_owner(&self) -> Option<&OwningAllocation> {
        match &self.owner {
            BufferOwner::Bound(owner) => Some(owner),
            BufferOwner::Managed(owner) => Some(owner.owner_ref()),
            _ => None,
        }
    }

    /// Borrow process-manager ownership when this buffer carries it.
    pub fn managed_owner(&self) -> Option<&ManagedAllocation> {
        match &self.owner {
            BufferOwner::Managed(owner) => Some(owner),
            _ => None,
        }
    }

    /// Consume the handle and recover its complete binding-issued ownership.
    ///
    /// This is the only way a bound buffer's ownership leaves the handle, and it
    /// is what a provider's `deallocate` calls before preparing or deferring the
    /// physical release. `Err` hands the buffer back untouched when it is not
    /// bound, so a caller that requires generation-checked release can refuse
    /// foreign memory without losing it.
    pub fn into_bound_owner(self) -> std::result::Result<BoundBufferOwnership, Self> {
        match self.owner {
            BufferOwner::Bound(owner) => Ok(BoundBufferOwnership::Binding(*owner)),
            BufferOwner::Managed(owner) => Ok(BoundBufferOwnership::Managed(*owner)),
            owner => Err(Self { owner, ..self }),
        }
    }

    /// Consume either binding-issued owning representation without discarding a
    /// process-manager settlement token.
    pub fn into_bound_ownership(self) -> std::result::Result<BoundBufferOwnership, Self> {
        self.into_bound_owner()
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
    ///
    /// # Panics
    ///
    /// For a **bound** buffer ([`DeviceBuffer::from_owning_allocation`]) this
    /// panics rather than silently downgrading generation-checked ownership to
    /// a bare address. Handing out the raw pointer alone would let a caller free
    /// it without matching the binding identity or the allocation generation —
    /// exactly the stale-pointer free the binding exists to prevent — while the
    /// owner it left behind would quarantine the same bytes. Use
    /// [`into_raw_with_owner`](Self::into_raw_with_owner) when the raw address
    /// *and* the owner are both wanted, or
    /// [`into_bound_owner`](Self::into_bound_owner) to take ownership back.
    pub fn into_raw(self) -> *mut c_void {
        assert!(
            !self.is_bound(),
            "DeviceBuffer::into_raw: this buffer carries a binding-issued owning allocation, so \
             returning the raw pointer alone would bypass binding-identity and allocation-\
             generation validation on release. Use into_raw_with_owner or into_bound_owner."
        );
        self.ptr.as_ptr()
    }

    /// Consume the handle, returning the raw pointer **together with** the
    /// binding-issued owner when there is one.
    ///
    /// This is the explicit escape hatch [`into_raw`](Self::into_raw) refuses to
    /// be: the caller receives the address and the generation-checked ownership
    /// in the same step, so the release obligation travels with the pointer
    /// instead of being dropped on the floor. `None` means the buffer was raw
    /// owning or borrowed and the historical raw contract applies.
    pub fn into_raw_with_owner(self) -> (*mut c_void, Option<BoundBufferOwnership>) {
        let ptr = self.ptr.as_ptr();
        match self.owner {
            BufferOwner::Bound(owner) => (ptr, Some(BoundBufferOwnership::Binding(*owner))),
            BufferOwner::Managed(owner) => (ptr, Some(BoundBufferOwnership::Managed(*owner))),
            _ => (ptr, None),
        }
    }

    /// Consume the buffer into its raw address and complete binding ownership.
    pub fn into_raw_with_bound_ownership(self) -> (*mut c_void, Option<BoundBufferOwnership>) {
        let ptr = self.ptr.as_ptr();
        match self.owner {
            BufferOwner::Bound(owner) => (ptr, Some(BoundBufferOwnership::Binding(*owner))),
            BufferOwner::Managed(owner) => (ptr, Some(BoundBufferOwnership::Managed(*owner))),
            _ => (ptr, None),
        }
    }
}

/// Complete binding-issued ownership carried by a [`DeviceBuffer`].
#[derive(Debug)]
pub enum BoundBufferOwnership {
    Binding(OwningAllocation),
    Managed(ManagedAllocation),
}

impl BoundBufferOwnership {
    pub fn owner(&self) -> &OwningAllocation {
        match self {
            Self::Binding(owner) => owner,
            Self::Managed(owner) => owner.owner_ref(),
        }
    }
}

/// Executor workspace allocation with any compatibility lease it still needs.
///
/// Manager-aware providers retain charges inside the buffer's managed owner and
/// leave `lease` empty. The default adapter preserves the older
/// reserve-then-allocate contract for providers not yet migrated.
#[derive(Debug)]
pub struct WorkspaceAllocation {
    buffer: DeviceBuffer,
    lease: Option<MemoryLease>,
}

impl WorkspaceAllocation {
    pub fn new(buffer: DeviceBuffer, lease: Option<MemoryLease>) -> Self {
        Self { buffer, lease }
    }

    pub fn buffer(&self) -> &DeviceBuffer {
        &self.buffer
    }

    pub fn buffer_mut(&mut self) -> &mut DeviceBuffer {
        &mut self.buffer
    }

    pub fn into_parts(self) -> (DeviceBuffer, Option<MemoryLease>) {
        (self.buffer, self.lease)
    }
}

impl std::ops::Deref for WorkspaceAllocation {
    type Target = DeviceBuffer;

    fn deref(&self) -> &Self::Target {
        &self.buffer
    }
}

impl std::ops::DerefMut for WorkspaceAllocation {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.buffer
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
///
/// The `id` is an opaque, EP-private handle to a completion event recorded on a
/// transfer stream by [`ExecutionProvider::copy_async`]. Await it by passing the
/// fence back to [`ExecutionProvider::wait_fence`], which makes the EP's compute
/// stream wait on the recorded event so a later kernel never reads bytes the
/// asynchronous copy is still transferring.
///
/// The id `0` is reserved for an **already-signalled** fence: a fully
/// synchronous copy (e.g. the CPU EP, or a zero-byte transfer) needs no wait, so
/// [`Fence::default`] / [`Fence::signalled`] returns id `0` and
/// [`ExecutionProvider::wait_fence`] treats it as a no-op.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Fence {
    pub id: u64,
}

impl Fence {
    /// A fence that is already complete; awaiting it is a no-op.
    pub fn signalled() -> Self {
        Self { id: 0 }
    }

    /// Wrap an EP-private completion-event handle.
    pub fn new(id: u64) -> Self {
        Self { id }
    }

    /// Whether this fence is already complete (needs no wait).
    pub fn is_signalled(&self) -> bool {
        self.id == 0
    }
}

/// Resolved-shape facts needed by an EP's structural capture-region policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CaptureRegionShapeStatus {
    /// Every present node input has a concrete shape before capture.
    pub inputs_resolved: bool,
    /// Every node output has a concrete shape before capture.
    pub outputs_resolved: bool,
}

/// Structural reason an EP excludes a node from a device-graph capture region.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StructuralCaptureDecline {
    /// Host-driven control-flow or sequence semantics.
    HostControlFlowOrSequence,
    /// A data-dependent output shape was unresolved before capture.
    UnresolvedOutputShape,
    /// A data-dependent input shape was unresolved before capture.
    UnresolvedInputShape,
}

impl StructuralCaptureDecline {
    /// Stable diagnostic text matching the executor's original capture audit.
    pub const fn reason(self) -> &'static str {
        match self {
            Self::HostControlFlowOrSequence => {
                "control-flow and sequence nodes are not device-graph capturable"
            }
            Self::UnresolvedOutputShape => {
                "data-dependent output shape was unresolved before capture"
            }
            Self::UnresolvedInputShape => {
                "data-dependent input shape was unresolved before capture"
            }
        }
    }
}

/// Uploads host bytes into a raw device address for a device EP.
///
/// This is the narrow capability the plugin's fused-subgraph executor needs to
/// stage a host-resident boundary input into device memory when ORT runs an
/// interspersed CPU→device partition and never inserts the host→device copy
/// itself (issue #982). It is deliberately smaller than the full
/// [`ExecutionProvider`] surface — a device address and a length — so it can be
/// captured once at compile time and stored on the executor without holding an
/// EP reference (which would change EP teardown semantics).
///
/// Implementations must perform a **synchronous** upload: on return the bytes
/// are resident at `dst`, so the caller may launch a kernel that reads them.
pub trait HostToDeviceCopier: Send + Sync {
    /// Copy `src` host bytes into device destination `dst`.
    ///
    /// # Safety
    ///
    /// `dst` must point to a live device allocation, on this copier's device,
    /// of at least `src.len()` bytes.
    unsafe fn copy_host_to_device(&self, src: &[u8], dst: *mut c_void) -> Result<()>;
}

/// The core EP interface. Every backend crate implements this (§4.1).
pub trait ExecutionProvider: Send + Sync {
    /// EP identifier (snake_case, e.g. `"cpu_ep"`, `"cuda_ep"`).
    fn name(&self) -> &str;

    fn device_type(&self) -> DeviceType;
    fn device_id(&self) -> DeviceId;

    /// PCI vendor id of this EP's device memory (0 = generic/host). Used by the
    /// plugin executor to reconstruct the device `OrtMemoryInfo` ORT registered
    /// the device allocator against, as a fallback for staging host-resident
    /// boundary inputs when no device-resident `OrtValue` is otherwise visible
    /// (issue #982). Host EPs keep the default.
    fn memory_vendor_id(&self) -> u32 {
        0
    }

    /// A synchronous host→device uploader, or `None` for host EPs.
    ///
    /// Device EPs return a small [`HostToDeviceCopier`] the plugin's fused
    /// executor captures at compile time and uses to stage host-resident
    /// boundary inputs into device scratch before launching a device kernel
    /// (issue #982). Returning `None` (the default) opts an EP out of staging
    /// entirely: its inputs are used verbatim, exactly as before.
    fn host_to_device_copier(&self) -> Option<std::sync::Arc<dyn HostToDeviceCopier>> {
        None
    }

    /// Optional executor-to-EP capabilities. Stock EPs advertise none and
    /// continue receiving resident [`crate::TensorView`] inputs.
    fn capabilities(&self) -> ExecutionProviderCapabilities {
        ExecutionProviderCapabilities::stock()
    }

    /// Page a lazy weight into device memory for live dispatch (WEIGHT_OFFLOAD
    /// Phase 3b). Returns a [`crate::PagedWeight`] whose device pointer the
    /// executor substitutes into the weight's input view; the binding must be
    /// held for the kernel's lifetime so the residency is not reclaimed early.
    ///
    /// `key` is a stable per-weight identity (the executor passes the
    /// initializer's value id) an EP may use to cache/evict residency across
    /// decode steps. The default returns `None`: stock EPs never receive lazy
    /// handles and the executor falls back to the host-materialization route.
    fn page_lazy_weight(
        &self,
        key: u64,
        weight: &crate::LazyWeight,
        source: &dyn crate::MmapRegionSource,
    ) -> Result<Option<crate::PagedWeight>> {
        let _ = (key, weight, source);
        Ok(None)
    }

    /// Best-effort lookahead page-in for a lazy weight the executor knows will be
    /// needed by a later node. Returns `true` only when a transfer was actually
    /// enqueued, so callers can distinguish a real prefetch from a no-op or
    /// eviction-neutrality guard decline. The default is a no-op so providers
    /// that do not own a residency cache do not need to participate.
    fn prefetch_lazy_weight(
        &self,
        key: u64,
        weight: &crate::LazyWeight,
        source: &dyn crate::MmapRegionSource,
    ) -> Result<bool> {
        let _ = (key, weight, source);
        Ok(false)
    }

    /// Initialize device resources / load libraries.
    fn initialize(&mut self, config: &EpConfig) -> Result<()>;
    /// Release device resources.
    fn shutdown(&mut self) -> Result<()>;

    /// Whether this EP can run `op` at the model's effective `opset` with the
    /// given input shapes, dtypes, and layouts, and at what cost.
    ///
    /// Every [`KernelMatch::Unsupported`] result must carry an actionable reason:
    /// state what the EP accepts and, where possible, how to fix the model or
    /// registration rather than returning a bare decline.
    fn supports_op(
        &self,
        op: &Node,
        opset: u64,
        shapes: &[Shape],
        input_dtypes: &[DataType],
        layouts: &[TensorLayout],
    ) -> KernelMatch;

    /// Query one node through an immutable structural graph lens.
    ///
    /// This compatibility adapter allocates metadata arrays before calling
    /// [`Self::supports_op`]. EPs can override it with native indexed metadata
    /// traversal to make capability discovery allocation-free.
    fn supports_node(&self, view: &GraphView<'_>, node: NodeIndex, opset: u64) -> KernelMatch {
        let inputs = view.node_inputs(node);
        let shapes = inputs
            .iter()
            .map(|input| {
                input
                    .map(|value| view.value(value).shape.clone())
                    .unwrap_or_default()
            })
            .collect::<Vec<_>>();
        let input_dtypes = inputs
            .iter()
            .map(|input| {
                input
                    .map(|value| view.value(value).dtype)
                    .unwrap_or(DataType::Undefined)
            })
            .collect::<Vec<_>>();
        let layouts = inputs
            .iter()
            .map(|input| {
                input
                    .map(|value| view.value(value).layout.clone())
                    .unwrap_or_else(TensorLayout::contiguous)
            })
            .collect::<Vec<_>>();
        self.supports_op(view.node(node), opset, &shapes, &input_dtypes, &layouts)
    }

    /// Get or create a kernel for `op` specialized to concrete `shapes`.
    ///
    /// `opset` is the effective operator-set version for `op`'s domain in the
    /// owning graph. EPs use it to select opset-specialized kernels (e.g. the
    /// opset-13 per-axis vs. the legacy opset-<13 2D-coercion `Softmax`).
    fn get_kernel(&self, op: &Node, shapes: &[Vec<usize>], opset: u64) -> Result<Box<dyn Kernel>>;

    /// Apply EP-owned structural policy to one prospective capture-region node.
    ///
    /// The executor supplies only graph structure and resolved-shape presence.
    /// Kernel warmth and the selected compiled kernel's capture support remain
    /// executor-owned mechanism and are checked only after this hook admits the
    /// node. Implementations must decline when either shape-status field is
    /// false; admitting an unresolved shape violates the executor contract. The
    /// default preserves the original predicate precedence exactly.
    fn plan_capture_region(
        &self,
        node: &Node,
        shape_status: CaptureRegionShapeStatus,
    ) -> Option<StructuralCaptureDecline> {
        if is_control_flow_or_sequence(node) {
            return Some(StructuralCaptureDecline::HostControlFlowOrSequence);
        }
        if !shape_status.outputs_resolved {
            return Some(StructuralCaptureDecline::UnresolvedOutputShape);
        }
        if !shape_status.inputs_resolved {
            return Some(StructuralCaptureDecline::UnresolvedInputShape);
        }
        None
    }

    /// Allocate device memory.
    fn allocate(&self, size: usize, alignment: usize) -> Result<DeviceBuffer>;

    fn allocate_with_mapped_growth(
        &self,
        size: usize,
        alignment: usize,
        grant: onnx_runtime_memory_governor::MappedGrowthGrant,
    ) -> Result<DeviceBuffer> {
        let newly_mapped_bytes = self.mapped_bytes_for_allocation(size, alignment)?;
        let allocation = self.allocate(size, alignment)?;
        if let Err(error) = grant.commit_bytes(newly_mapped_bytes) {
            let _ = self.deallocate(allocation);
            return Err(EpError::Memory(error));
        }
        Ok(allocation)
    }

    /// Allocate executor workspace as one reserve/allocate/commit transaction.
    ///
    /// Providers with a process manager override this so the charge travels with
    /// physical ownership through deferred release. The default is the existing
    /// compatibility sequence for synchronous providers.
    fn allocate_workspace(
        &self,
        size: usize,
        alignment: usize,
        role: MemoryRole,
    ) -> Result<WorkspaceAllocation> {
        let target_mapped = self.mapped_bytes_for_allocation(size, alignment)?;
        let mut grant = self.prepare_mapped_growth(target_mapped, role)?;
        let lease = match self.reserve_workspace(size as u64, role) {
            Ok(lease) => lease,
            Err(error) => {
                drop(grant);
                return Err(error);
            }
        };
        let buffer = match grant.take() {
            Some(grant) => self.allocate_with_mapped_growth(size, alignment, grant)?,
            None => self.allocate(size, alignment)?,
        };
        Ok(WorkspaceAllocation::new(buffer, lease))
    }

    /// Replace executor workspace without granting the replacement until the
    /// old allocation's provider-specific release boundary is satisfied.
    ///
    /// Synchronous providers use this default. Deferred providers override it
    /// to await a structured release outcome with a finite deadline.
    fn replace_workspace(
        &self,
        old: Option<WorkspaceAllocation>,
        size: usize,
        alignment: usize,
        role: MemoryRole,
    ) -> Result<WorkspaceAllocation> {
        if let Some(old) = old {
            self.deallocate_workspace(old)?;
        }
        self.allocate_workspace(size, alignment, role)
    }

    /// Allocate device address space while committing only selected byte ranges.
    ///
    /// Providers whose allocator cannot reserve without committing should use
    /// the default, preserving eager allocation. CUDA VMM overrides this so
    /// shape-stable buffers such as KV can keep one virtual address while
    /// mapping physical granules only where the live sequence reaches.
    fn allocate_committed(
        &self,
        size: usize,
        alignment: usize,
        committed_ranges: &[std::ops::Range<usize>],
    ) -> Result<DeviceBuffer> {
        let _ = committed_ranges;
        self.allocate(size, alignment)
    }

    /// Ensure a byte range in an existing allocation is backed by physical
    /// memory. Eager providers committed everything at allocation time, so their
    /// default is a no-op.
    fn commit_allocation_range(
        &self,
        buffer: &DeviceBuffer,
        offset: usize,
        bytes: usize,
    ) -> Result<()> {
        let _ = (buffer, offset, bytes);
        Ok(())
    }

    /// Commit all listed ranges as one allocator transaction.
    fn commit_allocation_ranges(&self, ranges: &[(&DeviceBuffer, usize, usize)]) -> Result<()> {
        for &(buffer, offset, bytes) in ranges {
            self.commit_allocation_range(buffer, offset, bytes)?;
        }
        Ok(())
    }

    fn commit_allocation_ranges_with_mapped_growth(
        &self,
        ranges: &[(&DeviceBuffer, usize, usize)],
        grant: &mut onnx_runtime_memory_governor::MappedGrowthGrant,
    ) -> Result<u64> {
        let _ = grant;
        self.commit_allocation_ranges(ranges)?;
        self.mapped_bytes_for_allocation_ranges(ranges)
    }

    fn mapped_bytes_for_allocation_ranges(
        &self,
        ranges: &[(&DeviceBuffer, usize, usize)],
    ) -> Result<u64> {
        Ok(ranges.iter().fold(0_u64, |total, (_, _, bytes)| {
            total.saturating_add(*bytes as u64)
        }))
    }

    /// Release physical backing from a byte range in an existing allocation
    /// while preserving its virtual address. Lazy providers use this for
    /// transactional growth rollback. Eager providers return an actionable
    /// unsupported error: unlike commit, decommit has no eager equivalent.
    /// Returns the bytes actually unmapped after shared references are applied.
    fn decommit_allocation_range(
        &self,
        buffer: &DeviceBuffer,
        offset: usize,
        bytes: usize,
    ) -> Result<u64> {
        let _ = (buffer, offset, bytes);
        Err(EpError::KernelFailed(format!(
            "{}: partial decommit requires a VirtualBacking capability",
            self.name()
        )))
    }

    /// Physical bytes currently claimed by `buffer`. Eager providers return
    /// `buffer.len()`; lazy providers may report the committed subset.
    fn allocation_committed_bytes(&self, buffer: &DeviceBuffer) -> usize {
        buffer.len()
    }

    /// Free device memory.
    fn deallocate(&self, buffer: DeviceBuffer) -> Result<()>;

    /// Whether `deallocate` proves physical release before returning `Ok`.
    ///
    /// Providers that only enqueue release return `false`. A compatibility
    /// workspace lease then fails closed instead of refunding bytes whose
    /// terminal outcome is not observable through this legacy API.
    fn deallocation_settles_before_return(&self) -> bool {
        true
    }

    /// Release an executor workspace and then its compatibility lease.
    ///
    /// Manager-aware buffers carry no outer lease: their settlement token keeps
    /// the charge until the provider's actual structured release outcome.
    fn deallocate_workspace(&self, workspace: WorkspaceAllocation) -> Result<()> {
        let (buffer, lease) = workspace.into_parts();
        match self.deallocate(buffer) {
            Ok(()) if self.deallocation_settles_before_return() => {
                drop(lease);
                Ok(())
            }
            Ok(()) => {
                if let Some(lease) = lease {
                    Box::leak(Box::new(lease));
                }
                Ok(())
            }
            Err(error) => {
                if let Some(lease) = lease {
                    Box::leak(Box::new(lease));
                }
                Err(error)
            }
        }
    }

    /// Free device memory and report mapped-zone bytes actually unmapped.
    ///
    /// The report is based on global mapping references, not which allocation
    /// originally caused the mapping.
    fn deallocate_with_unmapped(&self, buffer: DeviceBuffer) -> Result<u64> {
        self.deallocate(buffer)?;
        Ok(0)
    }

    /// Synchronous copy (host↔device or device↔device).
    fn copy(&self, src: &DeviceBuffer, dst: &mut DeviceBuffer, size: usize) -> Result<()>;
    /// Asynchronous copy; returns a [`Fence`] to await.
    ///
    /// The copy is enqueued on a dedicated transfer stream (not the compute
    /// stream) so it can overlap compute already queued on the compute stream —
    /// this is the mechanism half of Phase-4 compute/transfer overlap for weight
    /// paging. The returned [`Fence`] names a completion event on that transfer
    /// stream; the caller must order any consumer of `dst` after the transfer by
    /// passing the fence to [`ExecutionProvider::wait_fence`] before launching a
    /// kernel that reads `dst`. A synchronous EP may perform the copy inline and
    /// return an already-signalled [`Fence::signalled`].
    fn copy_async(&self, src: &DeviceBuffer, dst: &mut DeviceBuffer, size: usize) -> Result<Fence>;

    /// Order this EP's compute stream after the transfer named by `fence`.
    ///
    /// Makes the compute stream wait on the fence's completion event (a
    /// stream-ordered, non-host-blocking cross-stream wait) so a subsequently
    /// launched kernel observes the fully-transferred bytes produced by the
    /// matching [`ExecutionProvider::copy_async`]. Awaiting an already-signalled
    /// fence ([`Fence::is_signalled`]) is a no-op. The default implementation is
    /// a no-op, correct for synchronous EPs whose `copy_async` already completed
    /// the transfer before returning.
    fn wait_fence(&self, _fence: &Fence) -> Result<()> {
        Ok(())
    }

    /// Record a completion event for all compute enqueued on this EP's compute
    /// stream so far, returning a [`Fence`] that later transfers can wait on.
    ///
    /// This is the write-after-read (WAR) half of double-buffered prefetch: once
    /// a kernel that *reads* a staging buffer has been launched on the compute
    /// stream, record a fence over it and pass that fence to
    /// [`ExecutionProvider::copy_wait_fence`] before enqueueing the async copy
    /// that *overwrites* the same buffer, so the transfer stream never clobbers
    /// bytes a still-running consumer is reading. The default implementation
    /// returns an already-signalled [`Fence::signalled`] — correct for
    /// synchronous EPs whose compute completes inline, making the paired
    /// [`ExecutionProvider::copy_wait_fence`] a no-op.
    fn record_compute_fence(&self) -> Result<Fence> {
        Ok(Fence::signalled())
    }

    /// Order this EP's transfer stream after the compute named by `fence`.
    ///
    /// Makes the transfer (copy) stream wait on the fence's completion event (a
    /// stream-ordered, non-host-blocking cross-stream wait) so an async copy
    /// enqueued afterwards does not overwrite a buffer while the prior consumer
    /// recorded by [`ExecutionProvider::record_compute_fence`] is still reading
    /// it (WAR hazard on double-buffer reuse). Awaiting an already-signalled
    /// fence ([`Fence::is_signalled`]) is a no-op, as is the default
    /// implementation — correct for synchronous EPs.
    fn copy_wait_fence(&self, _fence: &Fence) -> Result<()> {
        Ok(())
    }

    /// Whether this EP can select the first maximum f32 element on-device and
    /// return the token id together with its capture-error status.
    fn device_argmax_supported(&self) -> bool {
        false
    }

    /// Launch an allocation-free device argmax over `batch` sequences of
    /// `elements` contiguous `dtype` values (Float32 or Float16) each, laid out
    /// as a `[batch, elements]` row-major block. `result` receives, per sequence
    /// `s`, two native-endian u32 values at word offset `2*s`: the token id, then
    /// the latching device capture-error bitmask. At `batch == 1` this is the
    /// previous single-sequence contract byte-for-byte.
    fn device_argmax(
        &self,
        _logits: &DeviceBuffer,
        _elements: usize,
        _batch: usize,
        _dtype: DataType,
        _result: &mut DeviceBuffer,
    ) -> Result<()> {
        Err(EpError::KernelFailed(format!(
            "{}: device argmax is not supported",
            self.name()
        )))
    }

    /// Fold the just-selected greedy token (from a prior [`device_argmax`],
    /// `result[0]`) into the persistent decode bindings device-to-device, for
    /// the native CUDA device-token-loop: write the token as an `i64` into
    /// `input_ids`, write `next_position` into `position_ids`, set the mask `1`
    /// at `next_position` (guarded by `mask_len`), append the token to
    /// `scratch[step]`, and OR the shared capture-error word (`result[1]`) into
    /// `scratch[capacity]`. No host sync — the caller drains `scratch` once per
    /// chain. EPs without device kernels reject the request.
    ///
    /// [`device_argmax`]: ExecutionProvider::device_argmax
    #[allow(clippy::too_many_arguments)]
    fn device_token_writer(
        &self,
        _result: &DeviceBuffer,
        _input_ids: &DeviceBuffer,
        _position_ids: &DeviceBuffer,
        _attention_mask: &DeviceBuffer,
        _scratch: &DeviceBuffer,
        _capacity: usize,
        _next_position: i64,
        _mask_len: usize,
        _write_position: bool,
        _step: u32,
    ) -> Result<()> {
        Err(EpError::KernelFailed(format!(
            "{}: device token writer is not supported",
            self.name()
        )))
    }

    /// Begin recording the supplied, already-compiled kernel sequence into a
    /// device graph. EPs without graph support reject the request.
    fn begin_device_graph_capture(&self, _kernels: &[&dyn Kernel]) -> Result<()> {
        Err(EpError::KernelFailed(format!(
            "{}: device graph capture is not supported",
            self.name()
        )))
    }

    /// End device-graph capture and install the resulting executable.
    fn end_device_graph_capture(&self) -> Result<()> {
        Err(EpError::KernelFailed(format!(
            "{}: device graph capture is not supported",
            self.name()
        )))
    }

    /// Abort an in-progress device-graph capture, returning the stream and
    /// lifecycle to a clean idle state so a subsequent [`reset_device_graph`]
    /// succeeds. Called on the error path of segmented capture when a node
    /// fails mid-record: the capture must always be ended before reset, so the
    /// stream is not left wedged in capture mode. EPs without device graphs have
    /// nothing to abort.
    ///
    /// [`reset_device_graph`]: ExecutionProvider::reset_device_graph
    fn abort_device_graph_capture(&self) -> Result<()> {
        Ok(())
    }

    /// Replay the installed device graph.
    ///
    /// When the EP holds multiple captured **segments** (segmented capture), this
    /// replays every installed segment in capture order. For the single-graph
    /// fast path (one whole-subgraph capture) that is exactly the one graph.
    fn replay_device_graph(&self) -> Result<()> {
        Err(EpError::KernelFailed(format!(
            "{}: device graph replay is not supported",
            self.name()
        )))
    }

    /// Replay one captured **segment** by its zero-based capture-order index.
    ///
    /// Segmented capture claims a whole subgraph even when only parts are
    /// device-graph capturable: the executor captures each maximal capturable
    /// run as its own segment and, at replay time, launches the segment graphs
    /// in order while running the non-capturable seam nodes eagerly in between.
    /// EPs without segmented graph support reject the request.
    fn replay_device_graph_segment(&self, _index: usize) -> Result<()> {
        Err(EpError::KernelFailed(format!(
            "{}: segmented device graph replay is not supported",
            self.name()
        )))
    }

    /// Destroy any installed device graph before its referenced buffers move or
    /// are released.
    fn reset_device_graph(&self) -> Result<bool> {
        Ok(false)
    }

    /// Read (without clearing) any latching device-side capture-safety error a
    /// captured kernel recorded during graph replay, as a raw violation bitmask
    /// (zero when none). EPs without device graphs report no error.
    ///
    /// The decode loop calls this at the per-step logits device→host sync so an
    /// out-of-range bounds violation becomes a hard error before the produced
    /// token is consumed, without adding a separate synchronization.
    fn check_device_capture_error(&self) -> Result<u32> {
        Ok(0)
    }

    /// Explicit device allocation/free counters, when the EP exposes them.
    fn device_allocation_counts(&self) -> Option<(u64, u64)> {
        None
    }

    /// Reserve governed bytes for executor-owned kernel workspace.
    ///
    /// Providers whose allocator already charges committed bytes may return
    /// `None`; providers backed by an eager allocator retain the returned lease
    /// alongside the allocation. The default preserves compatibility for
    /// providers without a device-memory governor.
    fn reserve_workspace(
        &self,
        _bytes: u64,
        _role: onnx_runtime_memory_governor::MemoryRole,
    ) -> Result<Option<onnx_runtime_memory_governor::MemoryLease>> {
        Ok(None)
    }

    fn prepare_mapped_growth(
        &self,
        bytes: u64,
        role: onnx_runtime_memory_governor::MemoryRole,
    ) -> Result<Option<onnx_runtime_memory_governor::MappedGrowthGrant>> {
        // `role` describes content/lifetime. Providers whose allocator
        // suballocates shared granules must canonicalize it to the arena's
        // physical mapped-attribution zone.
        let _ = (bytes, role);
        Ok(None)
    }

    fn mapped_bytes_for_allocation(&self, bytes: usize, alignment: usize) -> Result<u64> {
        let _ = alignment;
        Ok(bytes as u64)
    }

    fn release_mapped_growth(&self, bytes: u64, role: onnx_runtime_memory_governor::MemoryRole) {
        // This must use the same canonical physical zone as
        // `prepare_mapped_growth`; allocation lifetime is not map ownership.
        let _ = (bytes, role);
    }

    /// Place any long-lived device memory this provider holds under `governor`.
    ///
    /// Some providers keep a standing pool for as long as a model is loaded --
    /// the CUDA weight-residency cache is one. A pool that picks its own size is
    /// a second claim on memory the governor is already dividing up, and neither
    /// side can see the other: grant the KV pool most of a card, let a residency
    /// cache default to some fraction of it, and both are individually satisfied
    /// while the device is oversubscribed.
    ///
    /// This is the seam that ends that. It is on the provider contract rather
    /// than on one backend because it is not a CUDA question: any provider with
    /// a standing pool has it, and a third-party provider should be able to join
    /// the same accounting rather than run a ledger of its own.
    ///
    /// Returns the bytes now governed. The default is zero -- most providers
    /// hold no standing pool, and saying so is not a failure.
    ///
    /// # Errors
    ///
    /// If the tier cannot afford what the provider already holds. That is worth
    /// failing on: it says the model does not fit *before* the pool is used,
    /// rather than at an allocation somewhere unrelated later.
    ///
    /// Whether the memory this provider hands out commits physically as it is
    /// used rather than when it is requested.
    ///
    /// A forwarder, not a fact of its own: a provider should preserve the
    /// selected allocator's explicit [`DeviceAllocator::commits_on_demand`]
    /// signal. That signal requires both lazy physical mapping and governor
    /// charging; optional `VirtualBacking` capability presence alone is not
    /// enough. It is repeated here only because a caller holding a session
    /// reaches the allocator through the provider.
    ///
    /// `false` is the safe default -- a consumer that believes `true` will
    /// under-reserve.
    ///
    /// [`DeviceAllocator::commits_on_demand`]: onnx_runtime_memory_governor::DeviceAllocator::commits_on_demand
    fn commits_on_demand(&self) -> bool {
        false
    }

    /// Resize a provider-owned weight-residency budget before it joins a
    /// governor, returning the budget that will be adopted.
    ///
    /// `--vram-limit` is resolved after the model and backend are known, but a
    /// CUDA EP is constructed before the engine can size native KV. This hook
    /// lets load-time admission subtract the non-weight device claims first,
    /// preventing #712's "weights took the whole limit, KV failed later" path.
    fn set_weight_residency_budget(&self, _budget_bytes: u64) -> Result<Option<u64>> {
        Ok(None)
    }

    fn adopt_memory_governor(
        &self,
        _governor: &dyn onnx_runtime_memory_governor::MemoryGovernor,
        _tier: onnx_runtime_memory_governor::Tier,
        _holder: onnx_runtime_memory_governor::HolderId,
    ) -> Result<u64> {
        Ok(0)
    }

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
    /// dispatch (`docs/architecture/ORT2.md` §55.6). The keys come from the EP's own
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

fn is_control_flow_or_sequence(node: &Node) -> bool {
    if !(node.domain.is_empty() || node.domain == "ai.onnx") {
        return false;
    }
    matches!(
        node.op_type.as_str(),
        "If" | "Loop"
            | "Scan"
            | "SequenceEmpty"
            | "SequenceConstruct"
            | "SequenceInsert"
            | "SequenceErase"
            | "SequenceAt"
            | "SequenceLength"
            | "SplitToSequence"
            | "ConcatFromSequence"
    )
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

    /// A host-backed binding, so the bound-ownership contract can be exercised
    /// without a device.
    fn host_binding() -> onnx_runtime_memory_governor::MemoryBinding {
        use onnx_runtime_memory_governor::{BindingRegistry, DeviceKey, HostAllocator};
        use std::sync::Arc;

        #[derive(Debug)]
        struct Pin;

        let registry = BindingRegistry::new().expect("registry");
        let context = registry
            .register_provider_context(DeviceKey::HOST, Arc::new(Pin))
            .expect("provider context");
        let authority = registry
            .register_authority(DeviceKey::HOST, Arc::new(Pin))
            .expect("authority");
        let mechanism = registry
            .register_allocator(context, authority, Arc::new(HostAllocator))
            .expect("allocator");
        registry.select(mechanism).expect("selection");
        registry.bind(DeviceKey::HOST).expect("binding")
    }

    #[test]
    fn a_bound_buffer_carries_the_owner_that_minted_it() {
        let binding = host_binding();
        let owner = binding.allocate_owning(256, 64).expect("owning allocation");
        let identity = owner.identity();
        let address = owner.as_ptr().as_ptr() as usize;
        let buffer = DeviceBuffer::from_owning_allocation(owner, DeviceId::cpu());
        assert!(buffer.is_bound());
        assert!(!buffer.is_borrowed(), "a bound buffer owns its allocation");
        assert_eq!(buffer.len(), 256);
        assert_eq!(buffer.alignment(), 64);
        assert_eq!(buffer.as_ptr() as usize, address);
        assert_eq!(
            buffer.bound_owner().expect("bound owner").identity(),
            identity,
            "the buffer never describes a different allocation than its owner"
        );
        // The only way ownership leaves the handle is the consuming extractor,
        // and what comes back is the same generation-checked owner.
        let BoundBufferOwnership::Binding(recovered) = buffer.into_bound_owner().expect("bound")
        else {
            panic!("plain binding owner changed representation");
        };
        assert_eq!(recovered.identity(), identity);
        let outcome = recovered.release_now().expect("release");
        assert!(outcome.is_complete());
    }

    #[test]
    fn a_managed_buffer_keeps_charge_attached_to_bound_ownership() {
        use onnx_runtime_memory_governor::{
            AllocationPublication, AllocationRequest, DeviceKey, HostAllocator, LeaseLedger,
            LedgerGovernor, MemoryGovernor, MemoryRole, ProcessMemoryManager, Tier,
        };
        use std::sync::Arc;

        #[derive(Debug)]
        struct Pin;

        let manager = ProcessMemoryManager::new().unwrap();
        let context = manager
            .register_provider_context(DeviceKey::HOST, "host context", Arc::new(Pin))
            .unwrap();
        let governor = Arc::new(LedgerGovernor::new(LeaseLedger::new_for_device(
            DeviceKey::HOST,
            0,
            1024,
            0,
        )));
        let authority = manager
            .register_authority(
                DeviceKey::HOST,
                "host authority",
                Arc::new(Pin),
                governor.clone() as Arc<dyn MemoryGovernor + Send + Sync>,
            )
            .unwrap();
        let holder = manager
            .register_holder(&authority, "workspace", None)
            .unwrap();
        let mechanism = manager
            .register_allocator(
                &context,
                &authority,
                "host allocator",
                Arc::new(HostAllocator),
            )
            .unwrap();
        let owner = manager
            .bind_registered(&mechanism)
            .unwrap()
            .allocate(
                AllocationRequest::managed(
                    128,
                    16,
                    Tier::Host,
                    MemoryRole::Workspace { step_scoped: false },
                    holder,
                    128,
                ),
                AllocationPublication::exclusive(128, 128, 128),
            )
            .unwrap();
        assert_eq!(governor.used(Tier::Host), 128);
        let buffer = DeviceBuffer::from_managed_allocation(owner, DeviceId::cpu());
        assert!(buffer.is_bound());
        assert!(buffer.managed_owner().is_some());
        let BoundBufferOwnership::Managed(owner) =
            buffer.into_bound_owner().expect("managed ownership")
        else {
            panic!("manager ownership changed representation");
        };
        owner.release_now().unwrap();
        assert_eq!(governor.used(Tier::Host), 0);
    }

    #[test]
    fn a_raw_or_borrowed_buffer_has_no_bound_owner() {
        let raw = host_alloc(64, 16);
        assert!(!raw.is_bound());
        assert!(raw.bound_owner().is_none());
        // Failing to extract hands the buffer back untouched rather than losing
        // it, which is what lets a generation-checked path fail closed.
        let raw = raw.into_bound_owner().expect_err("not bound");
        assert_eq!(raw.len(), 64);
        host_free(raw);
    }

    #[test]
    #[should_panic(expected = "would bypass binding-identity")]
    fn into_raw_refuses_to_strip_bound_ownership() {
        let binding = host_binding();
        let owner = binding.allocate_owning(128, 16).expect("owning allocation");
        let buffer = DeviceBuffer::from_owning_allocation(owner, DeviceId::cpu());
        // Handing out the address alone would let a caller free it without
        // matching the binding identity or the allocation generation.
        let _ = buffer.into_raw();
    }

    #[test]
    fn into_raw_with_owner_is_the_explicit_escape_hatch() {
        let binding = host_binding();
        let owner = binding.allocate_owning(128, 16).expect("owning allocation");
        let expected = owner.as_ptr().as_ptr() as usize;
        let buffer = DeviceBuffer::from_owning_allocation(owner, DeviceId::cpu());
        let (ptr, owner) = buffer.into_raw_with_owner();
        assert_eq!(ptr as usize, expected);
        let BoundBufferOwnership::Binding(owner) =
            owner.expect("the release obligation travels with the pointer")
        else {
            panic!("plain binding owner changed representation");
        };
        assert!(owner.release_now().expect("release").is_complete());

        // A raw-owning buffer keeps the historical contract: no owner, and the
        // caller still owes the free.
        let raw = host_alloc(32, 8);
        let (ptr, owner) = raw.into_raw_with_owner();
        assert!(owner.is_none());
        // SAFETY: reconstruct the exact `Box<[u8]>` leaked in `host_alloc`.
        unsafe {
            drop(Box::from_raw(std::ptr::slice_from_raw_parts_mut(
                ptr as *mut u8,
                32,
            )));
        }
    }

    #[test]
    fn dropping_a_bound_buffer_quarantines_instead_of_freeing() {
        let binding = host_binding();
        let owner = binding.allocate_owning(64, 16).expect("owning allocation");
        let buffer = DeviceBuffer::from_owning_allocation(owner, DeviceId::cpu());
        drop(buffer);
        let quarantined = binding.quarantined().expect("quarantine list");
        assert_eq!(
            quarantined.len(),
            1,
            "a dropped bound buffer stays accounted for instead of being freed"
        );
        assert_eq!(quarantined[0].retained_bytes, 64);
    }

    #[test]
    fn borrowed_mut_buffer_writes_without_owning() {
        let mut backing = vec![0u8; 8];
        let ptr = backing.as_mut_ptr() as *mut c_void;
        // SAFETY: `backing` exclusively owns this writable region and outlives
        // the temporary alias.
        let mut buffer =
            unsafe { DeviceBuffer::from_borrowed_mut_parts(ptr, DeviceId::cpu(), 8, 1) }
                .expect("non-null backing");
        assert!(buffer.is_borrowed());
        // SAFETY: the alias has exclusive access to all eight backing bytes.
        unsafe {
            std::ptr::copy_nonoverlapping([1u8, 2, 3].as_ptr(), buffer.as_mut_ptr().cast(), 3);
        }
        assert_eq!(buffer.into_raw(), ptr);
        assert_eq!(&backing[..3], &[1, 2, 3]);
        assert!(
            unsafe {
                DeviceBuffer::from_borrowed_mut_parts(std::ptr::null_mut(), DeviceId::cpu(), 0, 1)
            }
            .is_none()
        );
    }
}
