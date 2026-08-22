//! A memory allocator ONNX Runtime calls into, backed by our own governor.
//!
//! ## Why this exists
//!
//! The goal is one authority that owns device and host memory, with every
//! component leasing from it. That works for memory *we* hand ONNX Runtime —
//! KV tensors, inputs, outputs — because we allocate those. It does not work
//! for the memory ORT allocates for itself: its arena, its activations, its
//! session-initialisation scratch. Those are invisible to a governor that only
//! sees what it granted, so a budget derived from "device capacity minus what
//! we leased" is wrong by however much ORT took behind it.
//!
//! `OrtAllocator` is the seam ORT provides for exactly this. It is a plain
//! vtable — `Alloc`, `Free`, `Info` — that a session can be told to use instead
//! of its built-in allocator. Implementing it here puts ORT's own allocations
//! on the same ledger as everything else.
//!
//! ## Why it matters beyond accounting
//!
//! It makes the two backends symmetric. The native runtime already routes
//! allocation through `ExecutionProvider::allocate`, which a caller can
//! implement. Until now ORT had no equivalent, so "the same memory manager
//! governs both backends" was true only of the native one. With this, both
//! backends allocate through the same contract, and a third party can supply an
//! allocator to either.
//!
//! ## What it deliberately does not do
//!
//! It does not decide policy. On refusal it returns null, which is ORT's
//! documented signal for allocation failure, and lets ORT surface the error.
//! Silently falling back to the system allocator would put memory outside the
//! budget while reporting that the budget held — the failure this whole
//! contract exists to prevent.

use std::ffi::c_void;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use onnx_runtime_memory_governor::{
    DeviceAllocator, HolderId, HostAllocator, MemoryGovernor, MemoryLease, MemoryRole, Tier,
};

use crate::allocator::MemoryInfo;

/// Alignment ORT's own CPU allocator guarantees.
///
/// ORT does not tell us the alignment it needs, and its internal allocators
/// align to at least this, so kernels are entitled to assume it. Under-aligning
/// would fault only on the vector paths that require it, which is the worst
/// possible way to find out.
const ALLOCATION_ALIGNMENT: usize = 64;

/// Bytes reserved before every block to hold its [`MemoryLease`].
///
/// ORT's `Free` hands back only a pointer, so the size has to be recovered
/// somehow. The first version kept a `HashMap<address, (layout, lease)>`, which
/// cost a lock and a hash on **both** paths. Measured against the alternatives,
/// that side table was the dominant cost of governing an allocation — not the
/// lease, and not `malloc`.
///
/// Putting the lease *in* the block removes the table without giving up what
/// the table was holding. `Drop` still returns the bytes exactly once (G2),
/// because the lease is still a lease; it just lives in the memory it governs.
///
/// The header is a full [`ALLOCATION_ALIGNMENT`] so the pointer handed to ORT
/// keeps the alignment its kernels are entitled to assume.
const HEADER_BYTES: usize = ALLOCATION_ALIGNMENT;

// The header only works if a lease fits in it. Checked here rather than
// discovered as a heap overwrite.
const _: () = assert!(
    std::mem::size_of::<MemoryLease>() <= HEADER_BYTES,
    "a MemoryLease no longer fits in the block header"
);
const _: () = assert!(
    std::mem::align_of::<MemoryLease>() <= ALLOCATION_ALIGNMENT,
    "a MemoryLease needs more alignment than a block base provides"
);

/// Which [`MemoryRole`] to charge, split the way ONNX Runtime already splits
/// its own calls.
///
/// ORT calls `Reserve` for allocations made while **building** a session and
/// `Alloc` during `Run`, documented as being there precisely so a custom
/// allocator can tell them apart. That is a free signal for the distinction
/// eviction ordering depends on: session-init memory is weights and plan state,
/// which are immutable and re-readable from disk and therefore the cheapest
/// thing to give up; `Run` memory is activations, which are not.
///
/// Charging both to one role throws that away and makes every byte look equally
/// expensive to evict.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AllocationRoles {
    /// Charged for allocations ORT makes through `Reserve`, while building a
    /// session.
    pub initialization: MemoryRole,
    /// Charged for allocations ORT makes through `Alloc`, during `Run`.
    pub run: MemoryRole,
}

impl AllocationRoles {
    /// The split ORT's own `Alloc`/`Reserve` distinction implies.
    pub const fn split() -> Self {
        Self {
            initialization: MemoryRole::Weights,
            run: MemoryRole::Activation,
        }
    }

    /// Charge everything to one role.
    ///
    /// For a caller who knows the split does not apply — an allocator serving
    /// only one kind of memory — or who wants the pre-`Reserve` behaviour.
    pub const fn uniform(role: MemoryRole) -> Self {
        Self {
            initialization: role,
            run: role,
        }
    }
}

impl Default for AllocationRoles {
    fn default() -> Self {
        Self::split()
    }
}

struct GovernedAllocatorState {
    governor: Arc<dyn MemoryGovernor + Send + Sync>,
    /// Where the bytes come from.
    ///
    /// Swappable, so the same governance works over the system allocator, a
    /// device arena, or something a caller brought — and so the allocator a
    /// caller supplies serves the native backend too, rather than being an
    /// ORT-shaped thing they have to write twice.
    memory: Arc<dyn DeviceAllocator>,
    tier: Tier,
    roles: AllocationRoles,
    holder: HolderId,
    /// Observability, not accounting: the governor's books are authoritative.
    /// Relaxed atomics, because the alternative is the side table this design
    /// exists to remove.
    live_bytes: AtomicU64,
    live_count: AtomicUsize,
    /// Every allocation ever served, never decremented.
    ///
    /// `live_count` cannot answer "did this run allocate through us": ORT frees
    /// most of what it takes before a run returns, so a test sampling it
    /// afterwards reads zero whether or not a single byte went through here.
    total_count: AtomicU64,
    /// Bytes ever served, and the high-water mark of `live_bytes`.
    ///
    /// Both are reported through `GetStats`, which is how ORT-side tooling sees
    /// governed numbers without knowing this crate exists.
    total_bytes: AtomicU64,
    peak_bytes: AtomicU64,
    /// Allocations served through `Reserve` rather than `Alloc`.
    reserve_count: AtomicU64,
}

/// An `OrtAllocator` whose allocations are leased from a memory governor.
///
/// The struct starts with the C vtable so a pointer to it is a valid
/// `*mut OrtAllocator`, which is what ORT's registration API takes.
#[repr(C)]
pub struct GovernedAllocator {
    /// Must be first: ORT treats `&self` as an `OrtAllocator`.
    base: onnx_genai_ort_sys::OrtAllocator,
    memory_info: MemoryInfo,
    state: Arc<GovernedAllocatorState>,
}

impl GovernedAllocator {
    /// Build an allocator that leases from `governor` before handing ORT memory.
    ///
    /// `tier` must match where `memory_info` says the memory lives; charging
    /// host allocations to a device budget is the mis-accounting this is meant
    /// to remove, not introduce.
    /// Build a governed allocator over **host** memory.
    ///
    /// This implementation allocates with Rust's global allocator, so it can
    /// only back host memory. Accepting a CUDA `MemoryInfo` here would hand ORT
    /// a host pointer labelled as device memory — a wild access from a kernel,
    /// not an error. Device backing needs an allocator that actually owns
    /// device memory; that is a separate implementation of the same contract,
    /// not a configuration of this one.
    ///
    /// Returns an error rather than silently correcting the arguments, because
    /// a caller who passed `Tier::Device` believes their device budget is being
    /// charged.
    /// A governed allocator over device memory, given the allocator that owns
    /// it.
    ///
    /// # Why this is separate from [`new`]
    ///
    /// [`new`] backs itself with [`HostAllocator`] and then refuses any tier
    /// but `Host`, which is right: letting it claim `Device` would hand ONNX
    /// Runtime host pointers labelled as device memory, and ORT decides from
    /// the memory info whether a pointer may be read on the host. The failure
    /// would be silent and would look like a wrong answer rather than a bad
    /// pointer.
    ///
    /// Taking `memory` up front removes that possibility: the tier and the
    /// memory agree by construction rather than by a later call that might not
    /// happen.
    ///
    /// # What it unblocks
    ///
    /// This is how the ONNX Runtime path reaches an allocator that commits
    /// physically on demand and charges those commits. `GovernedAllocator`
    /// forwards [`DeviceAllocator::commits_on_demand`], so a session that
    /// registers one of these answers the same accounting question the native
    /// path does -- which is what lets a consumer size a KV cache without
    /// knowing which backend it got.
    ///
    /// [`new`]: Self::new
    /// [`DeviceAllocator::commits_on_demand`]: onnx_runtime_memory_governor::DeviceAllocator::commits_on_demand
    pub fn on_device(
        memory_info: MemoryInfo,
        memory: Arc<dyn DeviceAllocator>,
        governor: Arc<dyn MemoryGovernor + Send + Sync>,
        roles: AllocationRoles,
        holder: HolderId,
    ) -> crate::error::Result<Box<Self>> {
        let tier = memory.device().tier;
        if tier != Tier::Device {
            return Err(crate::error::OrtError::InvalidArgument(format!(
                "GovernedAllocator::on_device is for device memory, but the supplied allocator \
                 serves {tier:?}; use GovernedAllocator::new for host memory"
            )));
        }
        if memory_info.device_name == "Cpu" {
            return Err(crate::error::OrtError::InvalidArgument(String::from(
                "this allocator owns device memory but its memory info names 'Cpu'; ONNX \
                 Runtime would let the host dereference pointers that are not host-readable. \
                 Build it with MemoryInfo::cuda(device_id)",
            )));
        }
        let mut allocator = Self::new(
            MemoryInfo::cpu_device()?,
            Arc::clone(&governor),
            Tier::Host,
            roles,
            holder,
        )?;
        // Replace what `new` assumed: host memory info and a host allocator,
        // neither of which is right here. Both are swapped together so no
        // state exists where they disagree.
        allocator.memory_info = memory_info;
        allocator.state = Arc::new(GovernedAllocatorState {
            governor,
            memory,
            tier,
            roles,
            holder,
            live_bytes: AtomicU64::new(0),
            live_count: AtomicUsize::new(0),
            total_count: AtomicU64::new(0),
            total_bytes: AtomicU64::new(0),
            peak_bytes: AtomicU64::new(0),
            reserve_count: AtomicU64::new(0),
        });
        Ok(allocator)
    }

    pub fn new(
        memory_info: MemoryInfo,
        governor: Arc<dyn MemoryGovernor + Send + Sync>,
        tier: Tier,
        roles: AllocationRoles,
        holder: HolderId,
    ) -> crate::error::Result<Box<Self>> {
        if tier != Tier::Host {
            return Err(crate::error::OrtError::InvalidArgument(format!(
                "GovernedAllocator allocates host memory, so it cannot be charged to {tier:?}; \
                 pass Tier::Host, or use an allocator implementation that owns memory on that tier"
            )));
        }
        if memory_info.device_name != "Cpu" {
            return Err(crate::error::OrtError::InvalidArgument(format!(
                "GovernedAllocator allocates host memory, but its memory info names device \
                 '{}'; ONNX Runtime would treat the host pointers it returns as memory on \
                 that device. Pass MemoryInfo::cpu_device(), or use an allocator \
                 implementation that owns memory on '{}'",
                memory_info.device_name, memory_info.device_name
            )));
        }
        let mut allocator = Box::new(Self {
            base: onnx_genai_ort_sys::OrtAllocator {
                version: onnx_genai_ort_sys::ORT_API_VERSION,
                Alloc: Some(governed_alloc),
                Free: Some(governed_free),
                Info: Some(governed_info),
                Reserve: Some(governed_reserve),
                GetStats: Some(governed_get_stats),
                // Only meaningful for a stream-aware device
                // allocator. This one owns host memory, where there
                // is no stream to allocate on.
                AllocOnStream: None,
                Shrink: Some(governed_shrink),
            },
            memory_info,
            state: Arc::new(GovernedAllocatorState {
                governor,
                memory: Arc::new(HostAllocator),
                tier,
                roles,
                holder,
                live_bytes: AtomicU64::new(0),
                live_count: AtomicUsize::new(0),
                total_count: AtomicU64::new(0),
                total_bytes: AtomicU64::new(0),
                peak_bytes: AtomicU64::new(0),
                reserve_count: AtomicU64::new(0),
            }),
        });
        // The vtable is only reachable through this pointer, so it must not move
        // afterwards; returning a Box makes that the caller's problem to keep.
        let _ = &mut *allocator;
        Ok(allocator)
    }

    /// Take bytes from `memory` instead of the system allocator.
    ///
    /// This is the seam a caller replaces. Governance is unchanged — the same
    /// leases, roles and counters — only the source of the bytes moves.
    ///
    /// The allocator's [`DeviceKey`](onnx_runtime_memory_governor::DeviceKey)
    /// must agree with the tier this allocator was built for. ONNX Runtime
    /// decides from the memory info whether a pointer is a host address, so an
    /// allocator returning device memory under host memory info turns a kernel
    /// read into a wild access rather than an error.
    ///
    /// The same `Arc` can back the native execution provider, which is the
    /// point: a caller writes one allocator, not one per backend.
    pub fn with_memory(
        mut self: Box<Self>,
        memory: Arc<dyn DeviceAllocator>,
    ) -> crate::error::Result<Box<Self>> {
        let tier = self.state.tier;
        if memory.device().tier != tier {
            return Err(crate::error::OrtError::InvalidArgument(format!(
                "this allocator is charged to {tier:?} but the supplied memory serves {:?}; \
                 ONNX Runtime decides from the memory info whether a pointer may be read on \
                 the host, so the two must agree",
                memory.device().tier
            )));
        }
        let state = Arc::get_mut(&mut self.state).ok_or_else(|| {
            crate::error::OrtError::InvalidArgument(
                "the memory source cannot be replaced once the allocator is shared; set it \
                 before registering"
                    .into(),
            )
        })?;
        state.memory = memory;
        Ok(self)
    }
    /// The pointer to hand ORT's registration API.
    /// Whether the memory behind this allocator commits physically as it is
    /// used.
    ///
    /// Forwards to whatever `with_memory` installed. This is what lets the
    /// ONNX Runtime path answer the same question the native path does: the
    /// `OrtAllocator` seam is a wrapper, and the property belongs to the
    /// allocator underneath it rather than to the backend on top.
    pub fn commits_on_demand(&self) -> bool {
        self.state.memory.commits_on_demand()
    }
    pub fn as_ort_allocator(&mut self) -> *mut onnx_genai_ort_sys::OrtAllocator {
        std::ptr::from_mut(&mut self.base)
    }

    /// Bytes currently held by live allocations, including their headers.
    pub fn live_bytes(&self) -> u64 {
        self.state.live_bytes.load(Ordering::Relaxed)
    }

    /// Number of live allocations.
    pub fn live_count(&self) -> usize {
        self.state.live_count.load(Ordering::Relaxed)
    }

    /// Every allocation this allocator has ever served.
    ///
    /// Monotonic, so it can answer whether work flowed through here even after
    /// everything it allocated has been freed again.
    pub fn total_count(&self) -> u64 {
        self.state.total_count.load(Ordering::Relaxed)
    }

    /// Allocations ORT made through Reserve rather than Alloc.
    pub fn reserve_count(&self) -> u64 {
        self.state.reserve_count.load(Ordering::Relaxed)
    }
}

/// Recover the Rust allocator from the vtable pointer ORT passes back.
///
/// # Safety
///
/// `this` must be a pointer previously produced by
/// [`GovernedAllocator::as_ort_allocator`].
unsafe fn allocator_from_base<'a>(
    this: *const onnx_genai_ort_sys::OrtAllocator,
) -> &'a GovernedAllocator {
    // SAFETY: `base` is the first field of a `#[repr(C)]` struct, so the two
    // addresses coincide.
    unsafe { &*this.cast::<GovernedAllocator>() }
}

/// ORT's `Alloc`: memory taken during `Run`.
unsafe extern "C" fn governed_alloc(
    this: *mut onnx_genai_ort_sys::OrtAllocator,
    size: usize,
) -> *mut c_void {
    // SAFETY: ORT only calls this with a pointer we registered.
    let allocator = unsafe { allocator_from_base(this) };
    unsafe { governed_alloc_as(allocator, size, allocator.state.roles.run) }
}

/// ORT's `Reserve`: memory taken while **building** a session.
///
/// ORT documents this as existing so a custom allocator can separate session
/// initialization from `Run`. Taking it up is what lets weights and activations
/// be charged to different roles, which is what eviction ordering needs — see
/// [`AllocationRoles`].
unsafe extern "C" fn governed_reserve(
    this: *mut onnx_genai_ort_sys::OrtAllocator,
    size: usize,
) -> *mut c_void {
    // SAFETY: ORT only calls this with a pointer we registered.
    let allocator = unsafe { allocator_from_base(this) };
    allocator
        .state
        .reserve_count
        .fetch_add(1, Ordering::Relaxed);
    unsafe { governed_alloc_as(allocator, size, allocator.state.roles.initialization) }
}

/// # Safety
///
/// `allocator` must be a live `GovernedAllocator`.
unsafe fn governed_alloc_as(
    allocator: &GovernedAllocator,
    size: usize,
    role: MemoryRole,
) -> *mut c_void {
    let state = &allocator.state;

    if size == 0 {
        // ORT documents nullptr for a zero-size request; allocating a
        // zero-length layout would also be undefined.
        return std::ptr::null_mut();
    }
    let Some(total) = size.checked_add(HEADER_BYTES) else {
        return std::ptr::null_mut();
    };

    // Lease first. A refusal must not allocate, or the budget is decorative.
    let Ok(lease) = state
        .governor
        .reserve(state.tier, total as u64, role, state.holder)
    else {
        return std::ptr::null_mut();
    };

    // SAFETY: `layout` has a non-zero size and a valid power-of-two alignment.
    let Ok(base) = state.memory.allocate(total, ALLOCATION_ALIGNMENT) else {
        // Dropping the lease returns the bytes; failing to would leak budget on
        // every allocation failure.
        drop(lease);
        return std::ptr::null_mut();
    };
    let base = base.as_ptr();

    // SAFETY: the header is the first `HEADER_BYTES` of an allocation we just
    // made, it is large enough and aligned enough for a `MemoryLease` (both
    // asserted at compile time), and nothing else can observe it yet.
    unsafe { base.cast::<MemoryLease>().write(lease) };
    let live = state.live_bytes.fetch_add(total as u64, Ordering::Relaxed) + total as u64;
    state.live_count.fetch_add(1, Ordering::Relaxed);
    state.total_count.fetch_add(1, Ordering::Relaxed);
    state.total_bytes.fetch_add(total as u64, Ordering::Relaxed);
    state.peak_bytes.fetch_max(live, Ordering::Relaxed);
    // SAFETY: `HEADER_BYTES` is within the allocation, so the block pointer is
    // in bounds; it stays `ALLOCATION_ALIGNMENT`-aligned because the header is
    // exactly one alignment unit.
    unsafe { base.add(HEADER_BYTES) }.cast::<c_void>()
}

unsafe extern "C" fn governed_free(this: *mut onnx_genai_ort_sys::OrtAllocator, p: *mut c_void) {
    if p.is_null() {
        return;
    }
    // SAFETY: ORT only calls this with a pointer we registered.
    let allocator = unsafe { allocator_from_base(this) };
    let state = &allocator.state;

    // SAFETY: `p` came from `governed_alloc`, so the header sits immediately
    // before it and holds the lease. Reading it out moves ownership here, so
    // the lease is dropped exactly once (G2) even though no table tracked it.
    //
    // Unlike the side-table version there is no way to detect a foreign
    // pointer: the table could refuse an address it never handed out, a header
    // cannot. Freeing a pointer this allocator did not return was already
    // undefined, and ORT only frees what it was given.
    let base = unsafe { p.cast::<u8>().sub(HEADER_BYTES) };
    let lease = unsafe { base.cast::<MemoryLease>().read() };
    let total = lease.bytes() as usize;
    let Some(base) = std::ptr::NonNull::new(base) else {
        return;
    };
    // SAFETY: the pointer, size and alignment are the triple
    // `governed_alloc_as` obtained from this same allocator. The lease was
    // moved out above, so nothing reads the header after this.
    unsafe { state.memory.deallocate(base, total, ALLOCATION_ALIGNMENT) };
    state.live_bytes.fetch_sub(total as u64, Ordering::Relaxed);
    state.live_count.fetch_sub(1, Ordering::Relaxed);
    // `lease` drops here, returning the bytes to the governor.
}

/// ORT's `GetStats`: report the governed numbers through ORT's own interface.
///
/// The keys are the ones ORT documents for this slot, so tooling that already
/// reads allocator statistics sees governed memory without knowing this crate
/// exists. That is the whole point of implementing it rather than leaving the
/// slot null: our accounting stops being visible only to us.
///
/// `Limit` is what the governor will still grant on this tier plus what we
/// already hold — the ceiling as this allocator experiences it, not the
/// machine's. `NumArenaExtensions` and `NumArenaShrinkages` are omitted rather
/// than reported as zero: this allocator has no arena, and a zero would read as
/// "an arena that never extended".
unsafe extern "C" fn governed_get_stats(
    this: *const onnx_genai_ort_sys::OrtAllocator,
    out: *mut *mut onnx_genai_ort_sys::OrtKeyValuePairs,
) -> onnx_genai_ort_sys::OrtStatusPtr {
    // SAFETY: ORT only calls this with a pointer we registered.
    let allocator = unsafe { allocator_from_base(this) };
    let state = &allocator.state;

    let Ok(api) = crate::error::api() else {
        return std::ptr::null_mut();
    };
    let (Some(create), Some(add)) = (api.CreateKeyValuePairs, api.AddKeyValuePair) else {
        return std::ptr::null_mut();
    };

    let mut pairs = std::ptr::null_mut();
    // SAFETY: `pairs` is a valid out-parameter; ORT allocates and the caller
    // releases with ReleaseKeyValuePairs, as this slot's contract says.
    unsafe { create(&mut pairs) };
    if pairs.is_null() {
        return std::ptr::null_mut();
    }

    let in_use = state.live_bytes.load(Ordering::Relaxed);
    let entries = [
        ("Limit", state.governor.available(state.tier) + in_use),
        ("InUse", in_use),
        ("TotalAllocated", state.total_bytes.load(Ordering::Relaxed)),
        ("MaxInUse", state.peak_bytes.load(Ordering::Relaxed)),
        ("NumAllocs", state.total_count.load(Ordering::Relaxed)),
        ("NumReserves", state.reserve_count.load(Ordering::Relaxed)),
    ];
    for (key, value) in entries {
        let (Ok(key), Ok(value)) = (
            std::ffi::CString::new(key),
            std::ffi::CString::new(value.to_string()),
        ) else {
            continue;
        };
        // SAFETY: both strings are NUL-terminated and live across the call;
        // ORT copies them internally.
        unsafe { add(pairs, key.as_ptr(), value.as_ptr()) };
    }

    // SAFETY: `out` is ORT's out-parameter for this call.
    unsafe { *out = pairs };
    std::ptr::null_mut()
}

/// ORT's `Shrink`: release memory held but not in use.
///
/// This allocator pools nothing — every `Free` returns the block to the system
/// and its bytes to the governor — so there is nothing held to release, and
/// ORT's own documentation says this is a no-op for non-arena allocators.
///
/// Implemented rather than left null anyway, because null and "nothing to give"
/// are different answers: a caller that walks allocators looking for one that
/// participates should see that this one does, and get an honest zero. When a
/// device-backed allocator lands it *will* have an arena, and this is the hook
/// its pressure response belongs in.
unsafe extern "C" fn governed_shrink(
    _this: *mut onnx_genai_ort_sys::OrtAllocator,
) -> onnx_genai_ort_sys::OrtStatusPtr {
    std::ptr::null_mut()
}
unsafe extern "C" fn governed_info(
    this: *const onnx_genai_ort_sys::OrtAllocator,
) -> *const onnx_genai_ort_sys::OrtMemoryInfo {
    // SAFETY: ORT only calls this with a pointer we registered.
    let allocator = unsafe { allocator_from_base(this) };
    allocator.memory_info.as_ptr()
}

/// A [`GovernedAllocator`] installed on an ONNX Runtime environment.
///
/// Registration is environment-wide and keyed by memory info, and sessions opt
/// in with the `session.use_env_allocators` config entry.
///
/// # Why this leaks by default
///
/// Unregistering removes the environment's *registration*. It does not reclaim
/// the allocator from sessions that already took it: ONNX Runtime copies
/// environment allocators into session state and wraps the raw pointer with a
/// no-op deleter, so a session created with `use_env_allocators` keeps calling
/// `Alloc`/`Free` on this pointer for its whole life, and so does anything
/// still holding memory that allocator handed out.
///
/// There is no ORT API that reports when the last of those is gone. Freeing the
/// allocator on drop would therefore be a use-after-free whose symptom appears
/// somewhere unrelated. So [`Drop`] unregisters and then **deliberately leaks**
/// the allocator: one bounded leak per registration, which is a fixed cost
/// rather than a growing one. [`RegisteredAllocator::release`] is the escape
/// hatch for callers who can prove no session is left.
///
/// The guard borrows the [`Environment`](crate::env::Environment) so it cannot
/// outlive it — unregistering through a freed `OrtEnv*` would otherwise be
/// reachable from safe code.
pub struct RegisteredAllocator<'env> {
    environment: &'env crate::env::Environment,
    allocator: std::mem::ManuallyDrop<Box<GovernedAllocator>>,
    unregistered: bool,
}

impl RegisteredAllocator<'_> {
    /// Bytes currently held by allocations ORT made through this allocator.
    pub fn live_bytes(&self) -> u64 {
        self.allocator.live_bytes()
    }

    /// Number of live allocations ORT holds through this allocator.
    pub fn live_count(&self) -> usize {
        self.allocator.live_count()
    }

    /// Every allocation served since registration, never decremented.
    pub fn total_count(&self) -> u64 {
        self.allocator.total_count()
    }

    /// Allocations served through ORT's Reserve rather than Alloc.
    pub fn reserve_count(&self) -> u64 {
        self.allocator.reserve_count()
    }

    /// Remove this allocator from the environment, reporting failure.
    ///
    /// The allocator itself is still leaked; see the type documentation. Use
    /// this over [`Drop`] when a failed unregistration is worth knowing about.
    pub fn unregister(mut self) -> crate::error::Result<()> {
        self.unregister_once()
    }

    /// Unregister **and free** the allocator.
    ///
    /// # Safety
    ///
    /// Every session created while this allocator was registered must already
    /// be dropped, and no memory it returned may still be live. ONNX Runtime
    /// copies environment allocators into session state behind a no-op deleter
    /// and offers no way to observe that the last user is gone, so this cannot
    /// be checked here.
    ///
    /// On failure to unregister, the allocator is left leaked rather than
    /// freed: an allocator ORT may still reach is worth more than the bytes.
    pub unsafe fn release(mut self) -> crate::error::Result<()> {
        self.unregister_once()?;
        // SAFETY: taken exactly once — `self` is consumed and its `Drop` is
        // skipped by the `forget` below.
        let allocator = unsafe { std::mem::ManuallyDrop::take(&mut self.allocator) };
        drop(allocator);
        std::mem::forget(self);
        Ok(())
    }

    fn unregister_once(&mut self) -> crate::error::Result<()> {
        if self.unregistered {
            return Ok(());
        }
        let api = crate::error::api()?;
        let unregister = api
            .UnregisterAllocator
            .ok_or(crate::error::OrtError::ApiUnavailable(
                "UnregisterAllocator",
            ))?;
        // SAFETY: this pair was registered by `register_governed_allocator`;
        // `environment` is borrowed so the handle is live. The flag is set only
        // after success, so a failed attempt is retried by `Drop` rather than
        // being recorded as done.
        crate::error::check_status(unsafe {
            unregister(
                self.environment.as_ptr().cast_mut(),
                self.allocator.memory_info.as_ptr(),
            )
        })?;
        self.unregistered = true;
        Ok(())
    }
}

impl std::fmt::Debug for RegisteredAllocator<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RegisteredAllocator")
            .field("live_bytes", &self.live_bytes())
            .field("live_count", &self.live_count())
            .finish()
    }
}

impl Drop for RegisteredAllocator<'_> {
    fn drop(&mut self) {
        // Best effort: `Drop` has nowhere to report to.
        let _ = self.unregister_once();
        // `self.allocator` is `ManuallyDrop`, so it is leaked here on purpose.
        // See the type documentation for why freeing it would be unsound.
    }
}

/// Install `allocator` on `environment` so ORT allocates through the governor.
///
/// Sessions must additionally be created with the `session.use_env_allocators`
/// config entry, or they keep their own built-in allocator and nothing is
/// governed — which is why [`crate::session::SessionOptions`] exposes it rather
/// than this function setting it invisibly.
pub fn register_governed_allocator(
    environment: &crate::env::Environment,
    mut allocator: Box<GovernedAllocator>,
) -> crate::error::Result<RegisteredAllocator<'_>> {
    let api = crate::error::api()?;
    let register = api
        .RegisterAllocator
        .ok_or(crate::error::OrtError::ApiUnavailable("RegisterAllocator"))?;
    if allocator.memory_info.is_arena()? {
        // ORT's own message for this names an enum the caller never wrote down.
        return Err(crate::error::OrtError::InvalidArgument(
            "a governed allocator's memory info describes an arena allocator, but \
             ONNX Runtime reserves the arena kind for its internal arenas; build \
             the memory info with MemoryInfo::cpu_device() rather than \
             MemoryInfo::cpu(), even if the allocator pools internally"
                .into(),
        ));
    }
    // SAFETY: the allocator pointer stays valid because the guard owns the Box
    // and never frees it unless the caller opts in through `release`.
    crate::error::check_status(unsafe {
        register(
            environment.as_ptr().cast_mut(),
            allocator.as_ort_allocator(),
        )
    })?;
    Ok(RegisteredAllocator {
        environment,
        allocator: std::mem::ManuallyDrop::new(allocator),
        unregistered: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use onnx_runtime_memory_governor::{LeaseLedger, LedgerGovernor};
    use std::sync::Mutex;

    /// A device allocator must not be described to ONNX Runtime as host memory,
    /// or the reverse.
    ///
    /// ORT decides from the memory info whether a pointer may be dereferenced
    /// on the host. Getting this wrong does not produce a bad-pointer error --
    /// it produces wrong numbers, or a read of device memory from the host that
    /// happens to succeed on a unified-memory machine and not elsewhere.
    ///
    /// `on_device` takes the allocator up front precisely so the tier and the
    /// memory cannot disagree; these pin the two ways a caller could still try.
    #[test]
    fn a_device_allocator_and_its_memory_info_must_agree() {
        #[derive(Debug)]
        struct FakeDevice(bool);

        // SAFETY: never allocates, so the non-overlap and validity guarantees
        // hold vacuously.
        impl DeviceAllocator for FakeDevice {
            fn allocate(
                &self,
                _bytes: usize,
                _align: usize,
            ) -> std::result::Result<std::ptr::NonNull<u8>, onnx_runtime_memory_governor::MemoryError>
            {
                Err(onnx_runtime_memory_governor::MemoryError::InvalidRequest {
                    tier: "device",
                    requested: 0,
                    reason: "test double",
                })
            }

            unsafe fn deallocate(&self, _ptr: std::ptr::NonNull<u8>, _bytes: usize, _align: usize) {
            }

            fn device(&self) -> onnx_runtime_memory_governor::DeviceKey {
                onnx_runtime_memory_governor::DeviceKey::device(0)
            }

            fn commits_on_demand(&self) -> bool {
                self.0
            }
        }

        let governor: Arc<dyn MemoryGovernor + Send + Sync> =
            Arc::new(LedgerGovernor::new(LeaseLedger::new(1 << 20, 1 << 20, 0)));

        // Device memory described as Cpu: refused.
        let Ok(cpu_info) = MemoryInfo::cpu_device() else {
            eprintln!("SKIPPED: ONNX Runtime unavailable; this test did NOT run");
            return;
        };
        let error = match GovernedAllocator::on_device(
            cpu_info,
            Arc::new(FakeDevice(false)),
            Arc::clone(&governor),
            AllocationRoles::default(),
            HolderId::new(90),
        ) {
            Ok(_) => panic!("device memory must not be described as Cpu"),
            Err(error) => error,
        };
        assert!(
            format!("{error}").contains("MemoryInfo::cuda"),
            "the error should name the constructor that fixes it: {error}"
        );

        let Ok(device_info) = MemoryInfo::dml(0) else {
            return;
        };
        let device_allocator = GovernedAllocator::on_device(
            device_info,
            Arc::new(FakeDevice(true)),
            Arc::clone(&governor),
            AllocationRoles::default(),
            HolderId::new(92),
        )
        .expect("matching CUDA memory info");
        assert!(
            device_allocator.commits_on_demand(),
            "the ORT wrapper must preserve the allocator's explicit accounting signal"
        );

        // Host memory offered to the device constructor: refused too, and by
        // the allocator's own tier rather than by the memory info.
        let Ok(cpu_info) = MemoryInfo::cpu_device() else {
            return;
        };
        let error = match GovernedAllocator::on_device(
            cpu_info,
            Arc::new(HostAllocator),
            governor,
            AllocationRoles::default(),
            HolderId::new(91),
        ) {
            Ok(_) => panic!("host memory does not belong in the device constructor"),
            Err(error) => error,
        };
        assert!(
            format!("{error}").contains("GovernedAllocator::new"),
            "the error should point at the host constructor: {error}"
        );
    }

    fn allocator(budget: u64) -> (Box<GovernedAllocator>, LedgerGovernor) {
        allocator_with(
            budget,
            MemoryInfo::cpu_device().expect("cpu device memory info"),
        )
    }

    fn allocator_with(
        budget: u64,
        memory_info: MemoryInfo,
    ) -> (Box<GovernedAllocator>, LedgerGovernor) {
        let governor = LedgerGovernor::new(LeaseLedger::new(0, budget, 0));
        let allocator = GovernedAllocator::new(
            memory_info,
            Arc::new(governor.clone()),
            Tier::Host,
            AllocationRoles::split(),
            HolderId::new(9),
        );
        (allocator.expect("host allocator"), governor)
    }

    /// Every byte ORT takes is charged before it is handed over — including the
    /// header, which is memory this allocator really takes from the OS.
    ///
    /// Charging only the requested size would understate the budget by 64 bytes
    /// per allocation, which on a graph making thousands of them per step is a
    /// budget that quietly does not hold.
    #[test]
    fn an_allocation_is_charged_before_the_memory_is_returned() {
        let (mut alloc, governor) = allocator(4096);
        let ptr = unsafe { governed_alloc(alloc.as_ort_allocator(), 1024) };
        assert!(!ptr.is_null(), "a request within budget must succeed");
        let charged = 1024 + HEADER_BYTES as u64;
        assert_eq!(
            governor.available(Tier::Host),
            4096 - charged,
            "the header is real memory and must be charged too"
        );
        assert_eq!(alloc.live_bytes(), charged);

        unsafe { governed_free(alloc.as_ort_allocator(), ptr) };
        assert_eq!(
            governor.available(Tier::Host),
            4096,
            "freeing did not return the charge"
        );
        assert_eq!(alloc.live_count(), 0);
    }

    /// The block ORT is given is aligned and writable for its full requested
    /// size — the header must not eat into it.
    #[test]
    fn the_block_is_aligned_and_writable_past_the_header() {
        let (mut alloc, _) = allocator(1 << 20);
        for size in [1usize, 63, 64, 1000, 4096] {
            let ptr = unsafe { governed_alloc(alloc.as_ort_allocator(), size) };
            assert!(!ptr.is_null());
            assert_eq!(
                ptr as usize % ALLOCATION_ALIGNMENT,
                0,
                "a {size}-byte request produced a misaligned block"
            );
            // SAFETY: the block is ours for `size` bytes.
            unsafe { std::ptr::write_bytes(ptr.cast::<u8>(), 0xA5, size) };
            unsafe { governed_free(alloc.as_ort_allocator(), ptr) };
        }
    }

    /// A refused lease must return null and allocate nothing.
    ///
    /// Falling back to the system allocator would put memory outside the budget
    /// while every counter reported the budget held.
    #[test]
    fn an_over_budget_request_returns_null_rather_than_allocating_anyway() {
        let (mut alloc, governor) = allocator(1024);
        let ptr = unsafe { governed_alloc(alloc.as_ort_allocator(), 4096) };
        assert!(ptr.is_null(), "an over-budget request must fail");
        assert_eq!(
            governor.available(Tier::Host),
            1024,
            "a refused request consumed budget"
        );
        assert_eq!(alloc.live_bytes(), 0);
    }

    /// The budget is enforced across many allocations, not just one.
    #[test]
    fn allocations_are_refused_once_the_budget_is_exhausted() {
        let block = 1024usize;
        let charged = block + HEADER_BYTES;
        let budget = (charged * 4) as u64;
        let (mut alloc, governor) = allocator(budget);
        let mut pointers = Vec::new();
        for _ in 0..4 {
            let ptr = unsafe { governed_alloc(alloc.as_ort_allocator(), block) };
            assert!(!ptr.is_null());
            pointers.push(ptr);
        }
        assert_eq!(governor.available(Tier::Host), 0);

        let refused = unsafe { governed_alloc(alloc.as_ort_allocator(), 1) };
        assert!(refused.is_null(), "a full budget must refuse even one byte");

        for ptr in pointers {
            unsafe { governed_free(alloc.as_ort_allocator(), ptr) };
        }
        assert_eq!(governor.available(Tier::Host), budget);
    }

    /// Memory handed out is writable and distinct.
    ///
    /// The vtable could be wired to something that reports success without
    /// producing usable memory, which would show up far away as corruption.
    #[test]
    fn allocated_blocks_are_writable_and_do_not_overlap() {
        let (mut alloc, _governor) = allocator(1 << 20);
        let a = unsafe { governed_alloc(alloc.as_ort_allocator(), 256) };
        let b = unsafe { governed_alloc(alloc.as_ort_allocator(), 256) };
        assert!(!a.is_null() && !b.is_null());
        assert_ne!(a, b);

        // SAFETY: both blocks are 256 bytes of freshly allocated memory.
        unsafe {
            std::ptr::write_bytes(a.cast::<u8>(), 0xAA, 256);
            std::ptr::write_bytes(b.cast::<u8>(), 0xBB, 256);
            assert_eq!(*a.cast::<u8>(), 0xAA, "writing block a did not stick");
            assert_eq!(*b.cast::<u8>(), 0xBB, "block b was overwritten by a");
        }
        unsafe {
            governed_free(alloc.as_ort_allocator(), a);
            governed_free(alloc.as_ort_allocator(), b);
        }
    }

    /// Blocks are aligned to what ORT's kernels are entitled to assume.
    #[test]
    fn blocks_are_aligned_for_vectorised_kernels() {
        let (mut alloc, _governor) = allocator(1 << 20);
        for size in [1usize, 7, 64, 1000] {
            let ptr = unsafe { governed_alloc(alloc.as_ort_allocator(), size) };
            assert!(!ptr.is_null());
            assert_eq!(
                ptr as usize % ALLOCATION_ALIGNMENT,
                0,
                "a {size} byte block was under-aligned"
            );
            unsafe { governed_free(alloc.as_ort_allocator(), ptr) };
        }
    }

    /// A zero-size request is null, matching ORT's documented contract.
    #[test]
    fn a_zero_size_request_is_null_and_costs_nothing() {
        let (mut alloc, governor) = allocator(4096);
        let ptr = unsafe { governed_alloc(alloc.as_ort_allocator(), 0) };
        assert!(ptr.is_null());
        assert_eq!(governor.available(Tier::Host), 4096);
    }

    /// Freeing null is a no-op, as the C contract requires.
    #[test]
    fn freeing_null_is_a_no_op() {
        let (mut alloc, _governor) = allocator(4096);
        unsafe { governed_free(alloc.as_ort_allocator(), std::ptr::null_mut()) };
    }

    /// The `Info` callback reports the memory info the allocator was built with.
    ///
    /// ORT uses this to decide where a tensor lives, so a wrong answer would
    /// have it treat host memory as device memory.
    #[test]
    fn info_reports_the_allocators_own_memory_info() {
        let (mut alloc, _governor) = allocator(4096);
        let info = unsafe { governed_info(alloc.as_ort_allocator()) };
        assert_eq!(
            info,
            alloc.memory_info.as_ptr(),
            "Info returned a different OrtMemoryInfo than the allocator holds"
        );
    }
    /// Registration must actually reach ORT: a session created afterwards is
    /// expected to allocate through us, so a silently-dropped registration is
    /// the failure worth catching.
    ///
    /// The observable proof is ORT's own rule that unregistering a memory info
    /// with no allocator behind it is an error. The first unregister succeeding
    /// and the second failing is only possible if something was really there.
    #[test]
    fn registering_installs_the_allocator_and_unregistering_removes_it() {
        let env = match crate::env::Environment::new("governed-allocator-registration") {
            Ok(env) => env,
            Err(_) => return, // no ORT library available in this environment
        };
        let (alloc, governor) = allocator(1 << 20);
        let registered = register_governed_allocator(&env, alloc).expect("register");
        assert_eq!(registered.live_bytes(), 0, "nothing allocated yet");

        registered
            .unregister()
            .expect("unregistering a live registration must succeed");
        assert_eq!(
            governor.available(Tier::Host),
            1 << 20,
            "unregistering must return every byte"
        );

        let api = crate::error::api().expect("api");
        let info = MemoryInfo::cpu_device().expect("cpu device memory info");
        // SAFETY: a live environment and a live memory info handle.
        let status = unsafe {
            (api.UnregisterAllocator.expect("UnregisterAllocator"))(
                env.as_ptr().cast_mut(),
                info.as_ptr(),
            )
        };
        assert!(
            !status.is_null(),
            "unregistering twice must fail, otherwise the first call proves nothing"
        );
        // SAFETY: a non-null status is owned by the caller.
        unsafe { (api.ReleaseStatus.expect("ReleaseStatus"))(status) };
    }

    /// An arena-kind memory info is rejected up front with an actionable
    /// message, rather than by ORT naming an enum the caller never wrote.
    #[test]
    fn an_arena_memory_info_is_refused_with_a_message_that_says_what_to_do() {
        let env = match crate::env::Environment::new("governed-allocator-arena") {
            Ok(env) => env,
            Err(_) => return,
        };
        let (alloc, _) = allocator_with(4096, MemoryInfo::cpu().expect("cpu memory info"));
        let error = register_governed_allocator(&env, alloc).expect_err("arena must be refused");
        let message = error.to_string();
        assert!(
            message.contains("cpu_device"),
            "the error must name the constructor to use instead, got: {message}"
        );
    }
    /// The allocator uses Rust's global allocator, so it must refuse to be
    /// charged to a device tier. Accepting it would report a device budget
    /// consumed while every byte sat in host RAM.
    #[test]
    fn a_device_tier_is_refused_because_this_allocator_only_has_host_memory() {
        let result = GovernedAllocator::new(
            MemoryInfo::cpu_device().expect("cpu device memory info"),
            Arc::new(LedgerGovernor::new(LeaseLedger::new(0, 4096, 0))),
            Tier::Device,
            AllocationRoles::split(),
            HolderId::new(9),
        );
        let error = match result {
            Ok(_) => panic!("a device tier must be refused"),
            Err(error) => error,
        };
        assert!(
            error.to_string().contains("Tier::Host"),
            "the error must say which tier is valid, got: {error}"
        );
    }
    /// A request the system allocator cannot satisfy must give the budget back.
    ///
    /// The claim happens before the allocation, so a failure between them leaks
    /// budget on every occurrence unless the lease is dropped. Provoked with a
    /// size no allocator can serve but that the governor will happily grant.
    #[test]
    fn a_failed_system_allocation_returns_the_budget_it_had_claimed() {
        let (mut alloc, governor) = allocator(u64::MAX);
        let before = governor.available(Tier::Host);

        // Large enough that `alloc` must fail, small enough that the layout is
        // still valid and the governor's ceiling is not the thing refusing it.
        let ptr = unsafe { governed_alloc(alloc.as_ort_allocator(), isize::MAX as usize / 2) };
        assert!(ptr.is_null(), "the system allocator cannot serve this");
        assert_eq!(
            governor.available(Tier::Host),
            before,
            "a failed allocation kept the budget it had claimed"
        );
        assert_eq!(alloc.live_bytes(), 0);
        assert_eq!(alloc.live_count(), 0);
    }

    /// A block's bytes survive being handed out, i.e. writing the lease into the
    /// header does not overlap the block.
    #[test]
    fn the_header_does_not_overlap_the_block() {
        let (mut alloc, _) = allocator(1 << 20);
        let size = 256usize;
        let first = unsafe { governed_alloc(alloc.as_ort_allocator(), size) };
        let second = unsafe { governed_alloc(alloc.as_ort_allocator(), size) };
        unsafe {
            std::ptr::write_bytes(first.cast::<u8>(), 0x11, size);
            std::ptr::write_bytes(second.cast::<u8>(), 0x22, size);
            for offset in 0..size {
                assert_eq!(*first.cast::<u8>().add(offset), 0x11, "block one clobbered");
                assert_eq!(
                    *second.cast::<u8>().add(offset),
                    0x22,
                    "block two clobbered"
                );
            }
            // Freeing reads the lease back out of each header; if a block write
            // had reached its own header this would corrupt the ledger.
            governed_free(alloc.as_ort_allocator(), first);
            governed_free(alloc.as_ort_allocator(), second);
        }
        assert_eq!(alloc.live_count(), 0);
    }
    /// `Reserve` and `Alloc` must charge different roles, or the signal ORT
    /// hands us for free is thrown away.
    ///
    /// Roles decide eviction order — weights go before KV because they are
    /// immutable and re-readable — so charging session-init memory as
    /// activations makes the cheapest thing to evict look like the most
    /// expensive.
    #[test]
    fn reserve_charges_the_initialization_role_and_alloc_the_run_role() {
        let (_unused, governor) = allocator(1 << 20);
        let seen = Arc::new(Mutex::new(Vec::new()));
        let recording = RecordingGovernor {
            inner: governor,
            seen: Arc::clone(&seen),
        };
        let mut alloc = GovernedAllocator::new(
            MemoryInfo::cpu_device().expect("cpu device memory info"),
            Arc::new(recording),
            Tier::Host,
            AllocationRoles::split(),
            HolderId::new(9),
        )
        .expect("host allocator");

        let from_run = unsafe { governed_alloc(alloc.as_ort_allocator(), 128) };
        let from_init = unsafe { governed_reserve(alloc.as_ort_allocator(), 128) };

        let roles = seen.lock().expect("roles");
        assert_eq!(
            roles.as_slice(),
            &[MemoryRole::Activation, MemoryRole::Weights],
            "Alloc must charge the run role and Reserve the initialization role"
        );
        drop(roles);

        unsafe {
            governed_free(alloc.as_ort_allocator(), from_run);
            governed_free(alloc.as_ort_allocator(), from_init);
        }
    }

    /// `AllocationRoles::uniform` restores the single-role behaviour for a
    /// caller who knows the split does not apply.
    #[test]
    fn uniform_roles_charge_everything_the_same_way() {
        let (_, governor) = allocator(1 << 20);
        let seen = Arc::new(Mutex::new(Vec::new()));
        let mut alloc = GovernedAllocator::new(
            MemoryInfo::cpu_device().expect("cpu device memory info"),
            Arc::new(RecordingGovernor {
                inner: governor,
                seen: Arc::clone(&seen),
            }),
            Tier::Host,
            AllocationRoles::uniform(MemoryRole::KvCache),
            HolderId::new(9),
        )
        .expect("host allocator");

        let a = unsafe { governed_alloc(alloc.as_ort_allocator(), 64) };
        let b = unsafe { governed_reserve(alloc.as_ort_allocator(), 64) };
        assert_eq!(
            seen.lock().expect("roles").as_slice(),
            &[MemoryRole::KvCache, MemoryRole::KvCache]
        );
        unsafe {
            governed_free(alloc.as_ort_allocator(), a);
            governed_free(alloc.as_ort_allocator(), b);
        }
    }

    /// `GetStats` must report the governed numbers under ORT's documented keys,
    /// so tooling that already reads allocator statistics sees them.
    #[test]
    fn get_stats_reports_the_governed_numbers_under_orts_own_keys() {
        let (mut alloc, _) = allocator(1 << 20);
        let first = unsafe { governed_alloc(alloc.as_ort_allocator(), 1024) };
        let second = unsafe { governed_reserve(alloc.as_ort_allocator(), 512) };

        let stats = read_stats(&mut alloc);
        let charged_first = 1024 + HEADER_BYTES as u64;
        let charged_second = 512 + HEADER_BYTES as u64;

        assert_eq!(stats["InUse"], (charged_first + charged_second).to_string());
        assert_eq!(stats["NumAllocs"], "2");
        assert_eq!(stats["NumReserves"], "1", "one of the two came via Reserve");
        assert_eq!(
            stats["TotalAllocated"],
            (charged_first + charged_second).to_string()
        );

        unsafe { governed_free(alloc.as_ort_allocator(), second) };
        let after = read_stats(&mut alloc);
        assert_eq!(
            after["InUse"],
            charged_first.to_string(),
            "InUse must fall when memory is freed"
        );
        assert_eq!(
            after["MaxInUse"],
            (charged_first + charged_second).to_string(),
            "MaxInUse is a high-water mark and must not fall"
        );
        assert_eq!(after["NumAllocs"], "2", "NumAllocs is cumulative");

        unsafe { governed_free(alloc.as_ort_allocator(), first) };
    }

    /// A governor that records which role each reservation was charged to.
    #[derive(Debug)]
    struct RecordingGovernor {
        inner: LedgerGovernor,
        seen: Arc<Mutex<Vec<MemoryRole>>>,
    }

    impl onnx_runtime_memory_governor::MemoryGovernor for RecordingGovernor {
        fn authority_id(&self) -> onnx_runtime_memory_governor::MemoryAuthorityId {
            self.inner.authority_id()
        }

        fn reserve(
            &self,
            tier: Tier,
            bytes: u64,
            role: MemoryRole,
            holder: HolderId,
        ) -> Result<MemoryLease, onnx_runtime_memory_governor::MemoryError> {
            let lease = self.inner.reserve(tier, bytes, role, holder)?;
            self.seen.lock().expect("roles").push(role);
            Ok(lease)
        }

        fn available(&self, tier: Tier) -> u64 {
            self.inner.available(tier)
        }

        fn used(&self, tier: Tier) -> u64 {
            self.inner.used(tier)
        }
    }

    fn read_stats(allocator: &mut GovernedAllocator) -> std::collections::HashMap<String, String> {
        let api = crate::error::api().expect("api");
        let mut pairs = std::ptr::null_mut();
        // SAFETY: a live allocator and a valid out-parameter.
        let status = unsafe { governed_get_stats(allocator.as_ort_allocator(), &mut pairs) };
        assert!(status.is_null(), "GetStats must succeed");
        assert!(!pairs.is_null(), "GetStats must produce a key-value set");

        let mut keys: *const *const std::os::raw::c_char = std::ptr::null();
        let mut values: *const *const std::os::raw::c_char = std::ptr::null();
        let mut count = 0usize;
        // SAFETY: `pairs` was just produced by ORT.
        unsafe {
            (api.GetKeyValuePairs.expect("GetKeyValuePairs"))(
                pairs,
                &mut keys,
                &mut values,
                &mut count,
            )
        };
        let mut out = std::collections::HashMap::new();
        for index in 0..count {
            // SAFETY: ORT reports `count` valid NUL-terminated pairs.
            unsafe {
                let key = std::ffi::CStr::from_ptr(*keys.add(index));
                let value = std::ffi::CStr::from_ptr(*values.add(index));
                out.insert(
                    key.to_string_lossy().into_owned(),
                    value.to_string_lossy().into_owned(),
                );
            }
        }
        // SAFETY: released exactly once, as the slot's contract requires.
        unsafe { (api.ReleaseKeyValuePairs.expect("ReleaseKeyValuePairs"))(pairs) };
        out
    }
    /// A caller's own allocator really serves ORT's vtable.
    ///
    /// This is the substitutability claim, and constructing one proves nothing:
    /// an implementation that ignored `with_memory` and kept using the system
    /// allocator would pass any test that only checks the memory works. So the
    /// counting allocator asserts it was *called*, and that every block came
    /// back to it.
    #[test]
    fn a_caller_supplied_allocator_backs_ort_allocations() {
        let (_, governor) = allocator(1 << 20);
        let counters = Arc::new(CountingAllocator::default());
        let mut alloc = GovernedAllocator::new(
            MemoryInfo::cpu_device().expect("cpu device memory info"),
            Arc::new(governor),
            Tier::Host,
            AllocationRoles::split(),
            HolderId::new(9),
        )
        .expect("host allocator")
        .with_memory(Arc::clone(&counters) as Arc<dyn DeviceAllocator>)
        .expect("host memory for a host tier");

        let first = unsafe { governed_alloc(alloc.as_ort_allocator(), 256) };
        let second = unsafe { governed_reserve(alloc.as_ort_allocator(), 512) };
        assert_eq!(
            counters.allocations.load(Ordering::Relaxed),
            2,
            "ORT's allocations must come from the supplied allocator, not the system one"
        );

        // The memory has to actually work, not merely be counted.
        unsafe { std::ptr::write_bytes(first.cast::<u8>(), 0x5A, 256) };
        unsafe { std::ptr::write_bytes(second.cast::<u8>(), 0xA5, 512) };

        unsafe {
            governed_free(alloc.as_ort_allocator(), first);
            governed_free(alloc.as_ort_allocator(), second);
        }
        assert_eq!(
            counters.deallocations.load(Ordering::Relaxed),
            2,
            "every block must be returned to the allocator that produced it"
        );
        assert_eq!(
            counters.live_bytes.load(Ordering::Relaxed),
            0,
            "the supplied allocator must see its own bytes balance"
        );
    }

    /// An allocator serving a different tier is refused, rather than handing ORT
    /// device memory under host memory info.
    #[test]
    fn a_memory_source_from_the_wrong_tier_is_refused() {
        let (_, governor) = allocator(1 << 20);
        let error = GovernedAllocator::new(
            MemoryInfo::cpu_device().expect("cpu device memory info"),
            Arc::new(governor),
            Tier::Host,
            AllocationRoles::split(),
            HolderId::new(9),
        )
        .expect("host allocator")
        .with_memory(Arc::new(DeviceTierAllocator))
        .err()
        .expect("a device allocator cannot back a host-tier allocator");
        assert!(
            error.to_string().contains("Device"),
            "the error must name the mismatch, got: {error}"
        );
    }

    /// Counts what passes through it, so a test can tell "used" from "ignored".
    #[derive(Debug, Default)]
    struct CountingAllocator {
        inner: HostAllocator,
        allocations: AtomicU64,
        deallocations: AtomicU64,
        live_bytes: AtomicU64,
    }

    impl DeviceAllocator for CountingAllocator {
        fn allocate(
            &self,
            bytes: usize,
            align: usize,
        ) -> Result<std::ptr::NonNull<u8>, onnx_runtime_memory_governor::MemoryError> {
            let ptr = self.inner.allocate(bytes, align)?;
            self.allocations.fetch_add(1, Ordering::Relaxed);
            self.live_bytes.fetch_add(bytes as u64, Ordering::Relaxed);
            Ok(ptr)
        }

        unsafe fn deallocate(&self, ptr: std::ptr::NonNull<u8>, bytes: usize, align: usize) {
            // SAFETY: forwarded unchanged from this method's own contract.
            unsafe { self.inner.deallocate(ptr, bytes, align) };
            self.deallocations.fetch_add(1, Ordering::Relaxed);
            self.live_bytes.fetch_sub(bytes as u64, Ordering::Relaxed);
        }

        fn device(&self) -> onnx_runtime_memory_governor::DeviceKey {
            onnx_runtime_memory_governor::DeviceKey::HOST
        }
    }

    /// Claims to serve a device, so the tier check has something to refuse.
    #[derive(Debug)]
    struct DeviceTierAllocator;

    impl DeviceAllocator for DeviceTierAllocator {
        fn allocate(
            &self,
            _bytes: usize,
            _align: usize,
        ) -> Result<std::ptr::NonNull<u8>, onnx_runtime_memory_governor::MemoryError> {
            unreachable!("the tier check must refuse this allocator before it is used")
        }

        unsafe fn deallocate(&self, _ptr: std::ptr::NonNull<u8>, _bytes: usize, _align: usize) {
            unreachable!("the tier check must refuse this allocator before it is used")
        }

        fn device(&self) -> onnx_runtime_memory_governor::DeviceKey {
            onnx_runtime_memory_governor::DeviceKey::device(0)
        }
    }
}
