//! An out-of-tree-style nxmem memory plugin, used to exercise the ABI.
//!
//! This crate deliberately depends on **only** `onnx-runtime-memory-abi`. It
//! never sees a runtime trait, a governor type, or an internal enum, so it is
//! a faithful stand-in for a plugin shipped by a third party. It is built as a
//! `cdylib` and loaded at runtime with `dlopen`, which is what makes the ABI
//! test suite portable: it exercises a genuine dynamic-library boundary on
//! macOS, Linux, and Windows without needing a GPU.
//!
//! # The mechanisms it publishes
//!
//! | Name | Purpose |
//! |---|---|
//! | `eager` | required slots only, no optional capability — the minimal conforming plugin |
//! | `lazy` | virtual backing, shared mapping, deferred release, structured release |
//! | `short-struct` | publishes a deliberately undersized allocator vtable |
//! | `callback-probe` | calls the host's `request_reclaim` during `allocate` and fails when the host refuses |
//! | `legacy-1-0` | pins itself to the minor-0 prefix, so an older participant meets a newer host |
//! | `sticky` | never retires its queued releases, so unload stays refused |
//!
//! Publishing several *named* mechanisms from one module — rather than
//! switching behaviour on an environment variable or a global — keeps every
//! scenario independent and lets one test process cover them all.
//!
//! Backing storage is ordinary host memory from `std::alloc`, allocated **and
//! freed inside this module**. No allocator ownership crosses the boundary.

use std::alloc::{Layout, alloc, dealloc};
use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

use onnx_runtime_memory_abi::{
    NXMEM_ABI_VERSION_MAJOR, NXMEM_ABI_VERSION_MINOR, NXMEM_CAP_ALLOCATOR,
    NXMEM_CAP_DEFERRED_RELEASE, NXMEM_CAP_SHARED_MAPPING, NXMEM_CAP_STRUCTURED_RELEASE,
    NXMEM_CAP_VIRTUAL_BACKING, NxmemAllocRequest, NxmemAllocResult, NxmemAllocation,
    NxmemAllocatorFactoryVtable, NxmemAllocatorVtable, NxmemDeviceId, NxmemHostCallbacks,
    NxmemNegotiateRequest, NxmemNegotiateResponse, NxmemOpenRequest, NxmemRangeRequest,
    NxmemReclaimRequest, NxmemReleaseCompletion, NxmemReleaseOutcome, NxmemSharedMappingVtable,
    NxmemSharedPrefixCommitInfo, NxmemSharedPrefixCommitRequest, NxmemSharedPrefixHandle,
    NxmemStatus, NxmemStatusCode, NxmemUnloadReport, NxmemVersionRange, NxmemVirtualBackingVtable,
    catch_status_panic, catch_void_panic, check_identity, negotiate_as,
};

// ─── module-wide bookkeeping ────────────────────────────────────────────────

/// Live objects this module still owns, reported through the unload gate.
#[derive(Debug, Default)]
struct ModuleCounters {
    live_allocators: AtomicU64,
    live_allocations: AtomicU64,
    live_views: AtomicU64,
    live_capabilities: AtomicU64,
    queued_releases: AtomicU64,
}

fn counters() -> &'static ModuleCounters {
    static COUNTERS: OnceLock<ModuleCounters> = OnceLock::new();
    COUNTERS.get_or_init(ModuleCounters::default)
}

fn next_mechanism_id() -> u64 {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    NEXT.fetch_add(1, Ordering::AcqRel) + 1
}

fn next_ticket() -> u64 {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    NEXT.fetch_add(1, Ordering::AcqRel) + 1
}

fn next_prefix_handle() -> u64 {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    NEXT.fetch_add(1, Ordering::AcqRel) + 1
}

// ─── mechanism behaviour ────────────────────────────────────────────────────

/// Which behaviours a named mechanism exhibits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Behaviour {
    /// Capabilities the mechanism claims.
    capability_flags: u64,
    /// The minor level its vtables declare.
    abi_minor: u32,
    /// Publish an allocator vtable that is deliberately too short.
    short_allocator_struct: bool,
    /// Call the host's `request_reclaim` on every allocation, and fail the
    /// allocation when the host refuses.
    probe_host_callback: bool,
    /// Never retire a queued release, so unload stays refused.
    never_drain: bool,
}

impl Behaviour {
    const fn base() -> Self {
        Self {
            capability_flags: NXMEM_CAP_ALLOCATOR | NXMEM_CAP_STRUCTURED_RELEASE,
            abi_minor: 1,
            short_allocator_struct: false,
            probe_host_callback: false,
            never_drain: false,
        }
    }
}

/// One published mechanism.
struct Mechanism {
    name: &'static str,
    c_name: &'static std::ffi::CStr,
    behaviour: Behaviour,
}

const MECHANISMS: &[Mechanism] = &[
    Mechanism {
        name: "eager",
        c_name: c"eager",
        behaviour: Behaviour {
            // The minimal conforming mechanism: required slots only, no
            // optional capability at all.
            capability_flags: NXMEM_CAP_ALLOCATOR,
            ..Behaviour::base()
        },
    },
    Mechanism {
        name: "lazy",
        c_name: c"lazy",
        behaviour: Behaviour {
            capability_flags: NXMEM_CAP_ALLOCATOR
                | NXMEM_CAP_VIRTUAL_BACKING
                | NXMEM_CAP_SHARED_MAPPING
                | NXMEM_CAP_DEFERRED_RELEASE
                | NXMEM_CAP_STRUCTURED_RELEASE,
            ..Behaviour::base()
        },
    },
    Mechanism {
        name: "short-struct",
        c_name: c"short-struct",
        behaviour: Behaviour {
            short_allocator_struct: true,
            ..Behaviour::base()
        },
    },
    Mechanism {
        name: "callback-probe",
        c_name: c"callback-probe",
        behaviour: Behaviour {
            probe_host_callback: true,
            ..Behaviour::base()
        },
    },
    Mechanism {
        name: "legacy-1-0",
        c_name: c"legacy-1-0",
        behaviour: Behaviour {
            // A mechanism written against the minor-0 contract: it knows
            // nothing about structured release.
            capability_flags: NXMEM_CAP_ALLOCATOR | NXMEM_CAP_DEFERRED_RELEASE,
            abi_minor: 0,
            ..Behaviour::base()
        },
    },
    Mechanism {
        name: "sticky",
        c_name: c"sticky",
        behaviour: Behaviour {
            capability_flags: NXMEM_CAP_ALLOCATOR
                | NXMEM_CAP_DEFERRED_RELEASE
                | NXMEM_CAP_STRUCTURED_RELEASE,
            never_drain: true,
            ..Behaviour::base()
        },
    },
];

/// Mechanism names this plugin publishes, in factory order.
///
/// Exposed through the `rlib` half so the host's tests can name mechanisms
/// without duplicating the list.
pub const MECHANISM_NAMES: &[&str] = &[
    "eager",
    "lazy",
    "short-struct",
    "callback-probe",
    "legacy-1-0",
    "sticky",
];

// ─── allocator state ────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy)]
struct Block {
    address: usize,
    bytes: usize,
    align: usize,
    committed: usize,
}

#[derive(Debug, Clone, Copy)]
struct QueuedRelease {
    ticket: u64,
    allocation_id: u64,
    address: usize,
    bytes: usize,
    align: usize,
}

#[derive(Debug, Clone, Copy)]
struct Prefix {
    bytes: usize,
    address: usize,
    references: u64,
}

/// The plugin-side state of one opened mechanism.
///
/// Reference counted by `retain`/`release`. The state is destroyed only when
/// the count reaches zero **and** no queued release still names it, which is
/// what keeps a deferred free from touching a freed context.
struct AllocatorState {
    mechanism_id: u64,
    device: NxmemDeviceId,
    behaviour: Behaviour,
    refcount: AtomicU64,
    callbacks: *const NxmemHostCallbacks,
    inner: Mutex<AllocatorInner>,
    /// The vtables handed to the host. Boxed inside the state so their
    /// addresses stay stable, and freed only when the state is destroyed.
    allocator_vtable: Box<NxmemAllocatorVtable>,
    backing_vtable: Option<Box<NxmemVirtualBackingVtable>>,
    shared_vtable: Option<Box<NxmemSharedMappingVtable>>,
}

// SAFETY: every field is either immutable after construction or guarded by
// `inner`'s mutex / an atomic. `callbacks` points at host memory the host
// promises to keep alive past this allocator's final release and past every
// queued release that names it.
unsafe impl Send for AllocatorState {}
// SAFETY: as above.
unsafe impl Sync for AllocatorState {}

#[derive(Debug, Default)]
struct AllocatorInner {
    blocks: HashMap<u64, Block>,
    queue: Vec<QueuedRelease>,
    prefixes: HashMap<u64, Prefix>,
}

impl AllocatorState {
    fn check(&self, allocation: &NxmemAllocation) -> Result<(), NxmemStatus> {
        check_identity(
            self.mechanism_id,
            self.device,
            allocation.mechanism_id,
            allocation.device,
        )
    }

    /// Ask the host to make room.
    ///
    /// Called from inside `allocate` with **no plugin lock held**, because a
    /// host callback may re-enter this module.
    fn host_reclaim(&self, bytes: u64) -> Result<u64, NxmemStatus> {
        if self.callbacks.is_null() {
            return Ok(0);
        }
        // SAFETY: the host promises the table outlives this allocator.
        let callbacks = unsafe { &*self.callbacks };
        let Some(request_reclaim) = callbacks.request_reclaim else {
            return Ok(0);
        };
        let request = NxmemReclaimRequest::new(self.mechanism_id, self.device, bytes);
        let mut reclaimed = 0u64;
        // SAFETY: both pointers address valid locals that outlive the call.
        let status =
            unsafe { request_reclaim(callbacks.host_ctx, &raw const request, &raw mut reclaimed) };
        if status.is_ok() {
            Ok(reclaimed)
        } else {
            Err(status)
        }
    }

    /// Report one retired deferred release. Called with no plugin lock held.
    fn report_completion(&self, completion: &NxmemReleaseCompletion) -> Result<(), NxmemStatus> {
        if self.callbacks.is_null() {
            return Ok(());
        }
        // SAFETY: as above.
        let callbacks = unsafe { &*self.callbacks };
        let Some(release_completed) = callbacks.release_completed else {
            return Ok(());
        };
        // SAFETY: `completion` is a valid local that outlives the call.
        let status = unsafe { release_completed(callbacks.host_ctx, completion as *const _) };
        if status.is_ok() {
            Ok(())
        } else {
            Err(status)
        }
    }
}

fn layout_of(bytes: usize, align: usize) -> Result<Layout, NxmemStatus> {
    Layout::from_size_align(bytes.max(1), align.max(1)).map_err(|_| {
        NxmemStatus::with_message(
            NxmemStatusCode::InvalidArgument,
            "testplugin: the requested size and alignment are not a valid layout",
        )
    })
}

/// Free one block's backing storage.
///
/// The bytes were allocated by this module and are freed by this module, so no
/// allocator ownership crosses the boundary.
fn free_block(block: Block) {
    let Ok(layout) = Layout::from_size_align(block.bytes.max(1), block.align.max(1)) else {
        return;
    };
    if block.address == 0 {
        return;
    }
    // SAFETY: the address came from `alloc` with exactly this layout inside
    // this module, and every caller removes the record first so it is freed
    // exactly once.
    unsafe { dealloc(block.address as *mut u8, layout) };
}

/// # Safety
///
/// `ctx` must be a live `AllocatorState` pointer produced by `open_allocator`.
unsafe fn state<'a>(ctx: *mut c_void) -> Option<&'a AllocatorState> {
    if ctx.is_null() {
        return None;
    }
    // SAFETY: delegated to this function's contract.
    Some(unsafe { &*(ctx as *const AllocatorState) })
}

macro_rules! plugin_entry {
    ($ctx:expr) => {
        // SAFETY: the host only passes back a `ctx` this module published.
        match unsafe { state($ctx) } {
            Some(state) => state,
            None => {
                return NxmemStatus::with_message(
                    NxmemStatusCode::InvalidArgument,
                    "testplugin: null allocator context",
                );
            }
        }
    };
}

// ─── allocator vtable slots ─────────────────────────────────────────────────

unsafe extern "C" fn plugin_allocate(
    ctx: *mut c_void,
    request: *const NxmemAllocRequest,
    result_out: *mut NxmemAllocResult,
) -> NxmemStatus {
    // SAFETY: same contract; the eager path commits the whole allocation.
    unsafe { allocate_inner(ctx, request, result_out, true) }
}

/// The shared body of the eager and lazy allocation slots.
///
/// `commit_whole` is the only difference: the eager slot commits everything it
/// reserves, while `allocate_committed` commits exactly the ranges it was
/// handed — including none at all, which is the whole point of lazy backing.
///
/// # Safety
///
/// Same contract as the two slots that call it.
unsafe fn allocate_inner(
    ctx: *mut c_void,
    request: *const NxmemAllocRequest,
    result_out: *mut NxmemAllocResult,
    commit_whole: bool,
) -> NxmemStatus {
    catch_status_panic(|| {
        let state = plugin_entry!(ctx);
        if request.is_null() || result_out.is_null() {
            return NxmemStatus::with_message(
                NxmemStatusCode::InvalidArgument,
                "testplugin: allocate received a null pointer",
            );
        }
        // SAFETY: checked non-null; the host passes valid records.
        let request = unsafe { &*request };
        if let Err(status) = check_identity(
            state.mechanism_id,
            state.device,
            request.mechanism_id,
            request.device,
        ) {
            return status;
        }

        if state.behaviour.probe_host_callback {
            // A refusal is a normal outcome and is surfaced as a failed
            // allocation, never as an abort or a leak.
            if let Err(status) = state.host_reclaim(request.bytes) {
                return NxmemStatus::with_message(
                    NxmemStatusCode::CallbackFailed,
                    &format!(
                        "testplugin: the host refused to reclaim memory, so the allocation \
                         cannot proceed: {}",
                        status.describe()
                    ),
                );
            }
        }

        let bytes = request.bytes as usize;
        let align = request.align as usize;
        let layout = match layout_of(bytes, align) {
            Ok(layout) => layout,
            Err(status) => return status,
        };
        // SAFETY: `layout` has a non-zero size and a valid power-of-two align.
        let raw = unsafe { alloc(layout) };
        if raw.is_null() {
            return NxmemStatus::with_message(
                NxmemStatusCode::OutOfMemory,
                "testplugin: the backing host allocator refused the request",
            );
        }

        let committed = if commit_whole {
            bytes
        } else if request.committed_range_count == 0 || request.committed_ranges.is_null() {
            // A lazy reservation with no ranges commits nothing.
            0
        } else {
            // SAFETY: the host promises `committed_range_count` readable
            // ranges at `committed_ranges` for the duration of the call.
            let ranges = unsafe {
                core::slice::from_raw_parts(
                    request.committed_ranges,
                    request.committed_range_count as usize,
                )
            };
            ranges
                .iter()
                .map(|range| range.bytes as usize)
                .sum::<usize>()
        };

        let block = Block {
            address: raw as usize,
            bytes,
            align,
            committed: committed.min(bytes),
        };
        match state.inner.lock() {
            Ok(mut inner) => {
                inner.blocks.insert(request.allocation_id, block);
            }
            Err(_) => {
                // SAFETY: allocated just above with exactly this layout and
                // never handed out.
                unsafe { dealloc(raw, layout) };
                return NxmemStatus::with_message(
                    NxmemStatusCode::InternalError,
                    "testplugin: allocator state is poisoned",
                );
            }
        }
        counters().live_allocations.fetch_add(1, Ordering::AcqRel);

        let mut result = NxmemAllocResult::zeroed();
        result.ptr = raw;
        result.owned_bytes = block.committed as u64;
        result.mapped_bytes = block.committed as u64;
        // SAFETY: checked non-null above.
        unsafe { *result_out = result };
        NxmemStatus::ok()
    })
}

unsafe extern "C" fn plugin_deallocate(
    ctx: *mut c_void,
    allocation: *const NxmemAllocation,
    unmapped_out: *mut u64,
) -> NxmemStatus {
    catch_status_panic(|| {
        let state = plugin_entry!(ctx);
        if allocation.is_null() || unmapped_out.is_null() {
            return NxmemStatus::with_message(
                NxmemStatusCode::InvalidArgument,
                "testplugin: deallocate received a null pointer",
            );
        }
        // SAFETY: checked non-null.
        let allocation = unsafe { &*allocation };
        if let Err(status) = state.check(allocation) {
            return status;
        }
        let block = match state.inner.lock() {
            Ok(mut inner) => inner.blocks.remove(&allocation.allocation_id),
            Err(_) => {
                return NxmemStatus::with_message(
                    NxmemStatusCode::InternalError,
                    "testplugin: allocator state is poisoned",
                );
            }
        };
        let Some(block) = block else {
            return NxmemStatus::with_message(
                NxmemStatusCode::UnknownAllocation,
                "testplugin: that allocation id is not live on this mechanism",
            );
        };
        let unmapped = block.committed as u64;
        free_block(block);
        counters().live_allocations.fetch_sub(1, Ordering::AcqRel);
        // SAFETY: checked non-null above.
        unsafe { *unmapped_out = unmapped };
        NxmemStatus::ok()
    })
}

unsafe extern "C" fn plugin_release_allocation(
    ctx: *mut c_void,
    allocation: *const NxmemAllocation,
    outcome_out: *mut NxmemReleaseOutcome,
) -> NxmemStatus {
    catch_status_panic(|| {
        let state = plugin_entry!(ctx);
        if allocation.is_null() || outcome_out.is_null() {
            return NxmemStatus::with_message(
                NxmemStatusCode::InvalidArgument,
                "testplugin: release received a null pointer",
            );
        }
        // SAFETY: checked non-null.
        let allocation = unsafe { &*allocation };
        if let Err(status) = state.check(allocation) {
            return status;
        }
        let block = match state.inner.lock() {
            Ok(mut inner) => inner.blocks.remove(&allocation.allocation_id),
            Err(_) => {
                return NxmemStatus::with_message(
                    NxmemStatusCode::InternalError,
                    "testplugin: allocator state is poisoned",
                );
            }
        };
        let Some(block) = block else {
            // Nothing was mutated, so this is a `Failed` outcome, not a
            // quarantine. The distinction is load-bearing for host accounting.
            let outcome = NxmemReleaseOutcome::failed(
                allocation.bytes,
                NxmemStatus::with_message(
                    NxmemStatusCode::UnknownAllocation,
                    "testplugin: that allocation id is not live on this mechanism",
                ),
            );
            // SAFETY: checked non-null above.
            unsafe { *outcome_out = outcome };
            return NxmemStatus::ok();
        };

        let unmapped = block.committed as u64;
        let bytes = block.bytes as u64;
        free_block(block);
        counters().live_allocations.fetch_sub(1, Ordering::AcqRel);
        // SAFETY: checked non-null above.
        unsafe { *outcome_out = NxmemReleaseOutcome::complete(bytes, unmapped) };
        NxmemStatus::ok()
    })
}

unsafe extern "C" fn plugin_enqueue_release(
    ctx: *mut c_void,
    allocation: *const NxmemAllocation,
    ticket_out: *mut u64,
) -> NxmemStatus {
    catch_status_panic(|| {
        let state = plugin_entry!(ctx);
        if allocation.is_null() || ticket_out.is_null() {
            return NxmemStatus::with_message(
                NxmemStatusCode::InvalidArgument,
                "testplugin: enqueue_release received a null pointer",
            );
        }
        // SAFETY: checked non-null.
        let allocation = unsafe { &*allocation };
        if let Err(status) = state.check(allocation) {
            return status;
        }
        let ticket = next_ticket();
        match state.inner.lock() {
            Ok(mut inner) => {
                let Some(block) = inner.blocks.remove(&allocation.allocation_id) else {
                    return NxmemStatus::with_message(
                        NxmemStatusCode::UnknownAllocation,
                        "testplugin: that allocation id is not live on this mechanism",
                    );
                };
                inner.queue.push(QueuedRelease {
                    ticket,
                    allocation_id: allocation.allocation_id,
                    address: block.address,
                    bytes: block.bytes,
                    align: block.align,
                });
            }
            Err(_) => {
                return NxmemStatus::with_message(
                    NxmemStatusCode::InternalError,
                    "testplugin: allocator state is poisoned",
                );
            }
        }
        // A queued release still names this context, so it takes its own
        // reference. That is what keeps the mechanism — and, through the
        // host's `Arc<PluginModule>`, the module itself — pinned until the
        // free retires.
        state.refcount.fetch_add(1, Ordering::AcqRel);
        counters().queued_releases.fetch_add(1, Ordering::AcqRel);
        // SAFETY: checked non-null above.
        unsafe { *ticket_out = ticket };
        NxmemStatus::ok()
    })
}

unsafe extern "C" fn plugin_drain_releases(
    ctx: *mut c_void,
    max: u64,
    retired_out: *mut u64,
) -> NxmemStatus {
    catch_status_panic(|| {
        let state = plugin_entry!(ctx);
        if retired_out.is_null() {
            return NxmemStatus::with_message(
                NxmemStatusCode::InvalidArgument,
                "testplugin: drain_releases received a null pointer",
            );
        }
        if state.behaviour.never_drain {
            // SAFETY: checked non-null above.
            unsafe { *retired_out = 0 };
            return NxmemStatus::ok();
        }

        // Take the batch under the lock, then **drop the lock** before calling
        // back into the host: a host callback may re-enter this module.
        let batch: Vec<QueuedRelease> = match state.inner.lock() {
            Ok(mut inner) => {
                let take = (max as usize).min(inner.queue.len());
                inner.queue.drain(..take).collect()
            }
            Err(_) => {
                return NxmemStatus::with_message(
                    NxmemStatusCode::InternalError,
                    "testplugin: allocator state is poisoned",
                );
            }
        };

        let mut retired = 0u64;
        for queued in batch {
            free_block(Block {
                address: queued.address,
                bytes: queued.bytes,
                align: queued.align,
                committed: queued.bytes,
            });
            counters().live_allocations.fetch_sub(1, Ordering::AcqRel);
            counters().queued_releases.fetch_sub(1, Ordering::AcqRel);

            let completion = NxmemReleaseCompletion::new(
                queued.ticket,
                state.mechanism_id,
                queued.allocation_id,
                NxmemReleaseOutcome::complete(queued.bytes as u64, queued.bytes as u64),
            );
            let reported = state.report_completion(&completion);
            // Drop the reference the enqueue took whether or not the host
            // accepted the completion: the bytes are already gone, and holding
            // the reference would pin the module forever.
            release_state(state);
            retired += 1;
            if let Err(status) = reported {
                // SAFETY: checked non-null above. A partial drain still reports
                // how many entries retired, so the host's accounting stays
                // exact even on the failure path.
                unsafe { *retired_out = retired };
                return NxmemStatus::with_message(
                    NxmemStatusCode::CallbackFailed,
                    &format!(
                        "testplugin: the host rejected a release completion: {}",
                        status.describe()
                    ),
                );
            }
        }
        // SAFETY: checked non-null above.
        unsafe { *retired_out = retired };
        NxmemStatus::ok()
    })
}

unsafe extern "C" fn plugin_pending_release_count(
    ctx: *mut c_void,
    count_out: *mut u64,
) -> NxmemStatus {
    catch_status_panic(|| {
        let state = plugin_entry!(ctx);
        if count_out.is_null() {
            return NxmemStatus::with_message(
                NxmemStatusCode::InvalidArgument,
                "testplugin: pending_release_count received a null pointer",
            );
        }
        let count = match state.inner.lock() {
            Ok(inner) => inner.queue.len() as u64,
            Err(_) => {
                return NxmemStatus::with_message(
                    NxmemStatusCode::InternalError,
                    "testplugin: allocator state is poisoned",
                );
            }
        };
        // SAFETY: checked non-null above.
        unsafe { *count_out = count };
        NxmemStatus::ok()
    })
}

unsafe extern "C" fn plugin_retain(ctx: *mut c_void) {
    catch_void_panic(|| {
        // SAFETY: the host only passes back a `ctx` this module published.
        if let Some(state) = unsafe { state(ctx) } {
            state.refcount.fetch_add(1, Ordering::AcqRel);
        }
    });
}

unsafe extern "C" fn plugin_release(ctx: *mut c_void) {
    catch_void_panic(|| {
        // SAFETY: the host only passes back a `ctx` this module published.
        if let Some(state) = unsafe { state(ctx) } {
            release_state(state);
        }
    });
}

/// Drop one reference, destroying the state at zero.
///
/// A queued release holds its own reference, so the state cannot be destroyed
/// while a deferred free still names it. `retain`/`release` never call back
/// into the host, as the contract requires.
fn release_state(state: &AllocatorState) {
    let previous = state.refcount.fetch_sub(1, Ordering::AcqRel);
    if previous != 1 {
        return;
    }
    // Reclaim anything still outstanding rather than leaking it.
    if let Ok(mut inner) = state.inner.lock() {
        for (_, block) in inner.blocks.drain() {
            free_block(block);
            counters().live_allocations.fetch_sub(1, Ordering::AcqRel);
        }
        let queued: Vec<QueuedRelease> = inner.queue.drain(..).collect();
        for entry in queued {
            free_block(Block {
                address: entry.address,
                bytes: entry.bytes,
                align: entry.align,
                committed: entry.bytes,
            });
            counters().live_allocations.fetch_sub(1, Ordering::AcqRel);
            counters().queued_releases.fetch_sub(1, Ordering::AcqRel);
        }
        let prefixes: Vec<Prefix> = inner.prefixes.drain().map(|(_, prefix)| prefix).collect();
        for prefix in prefixes {
            counters()
                .live_capabilities
                .fetch_sub(prefix.references, Ordering::AcqRel);
            free_block(Block {
                address: prefix.address,
                bytes: prefix.bytes,
                align: PREFIX_ALIGN,
                committed: prefix.bytes,
            });
        }
    }
    counters().live_allocators.fetch_sub(1, Ordering::AcqRel);
    // SAFETY: the count reached zero, so this is the last reference. The
    // pointer came from `Box::into_raw` in `plugin_open_allocator` and is
    // reclaimed exactly once, here.
    drop(unsafe { Box::from_raw(state as *const AllocatorState as *mut AllocatorState) });
}

// ─── virtual-backing slots ──────────────────────────────────────────────────

unsafe extern "C" fn plugin_allocate_committed(
    ctx: *mut c_void,
    request: *const NxmemAllocRequest,
    result_out: *mut NxmemAllocResult,
) -> NxmemStatus {
    // SAFETY: same contract; the lazy path commits only what it was handed.
    unsafe { allocate_inner(ctx, request, result_out, false) }
}

/// Read and identity-check a range request.
///
/// # Safety
///
/// `request` must be null or point to a valid record for the call.
unsafe fn range_of<'a>(
    state: &AllocatorState,
    request: *const NxmemRangeRequest,
) -> Result<&'a NxmemRangeRequest, NxmemStatus> {
    if request.is_null() {
        return Err(NxmemStatus::with_message(
            NxmemStatusCode::InvalidArgument,
            "testplugin: range request is null",
        ));
    }
    // SAFETY: delegated to this function's contract.
    let request = unsafe { &*request };
    state.check(&request.allocation)?;
    Ok(request)
}

unsafe extern "C" fn plugin_commit_range(
    ctx: *mut c_void,
    request: *const NxmemRangeRequest,
) -> NxmemStatus {
    catch_status_panic(|| {
        let state = plugin_entry!(ctx);
        // SAFETY: the host passes a valid record or null.
        let request = match unsafe { range_of(state, request) } {
            Ok(request) => request,
            Err(status) => return status,
        };
        match state.inner.lock() {
            Ok(mut inner) => {
                let Some(block) = inner.blocks.get_mut(&request.allocation.allocation_id) else {
                    return NxmemStatus::with_message(
                        NxmemStatusCode::UnknownAllocation,
                        "testplugin: that allocation id is not live on this mechanism",
                    );
                };
                let end = (request.range.offset + request.range.bytes) as usize;
                if end > block.bytes {
                    return NxmemStatus::with_message(
                        NxmemStatusCode::InvalidArgument,
                        "testplugin: the range runs past the end of the allocation",
                    );
                }
                block.committed = block.committed.max(end);
                NxmemStatus::ok()
            }
            Err(_) => NxmemStatus::with_message(
                NxmemStatusCode::InternalError,
                "testplugin: allocator state is poisoned",
            ),
        }
    })
}

unsafe extern "C" fn plugin_decommit_range(
    ctx: *mut c_void,
    request: *const NxmemRangeRequest,
    unmapped_out: *mut u64,
) -> NxmemStatus {
    catch_status_panic(|| {
        let state = plugin_entry!(ctx);
        if unmapped_out.is_null() {
            return NxmemStatus::with_message(
                NxmemStatusCode::InvalidArgument,
                "testplugin: decommit_range received a null out-parameter",
            );
        }
        // SAFETY: the host passes a valid record or null.
        let request = match unsafe { range_of(state, request) } {
            Ok(request) => request,
            Err(status) => return status,
        };
        let unmapped = match state.inner.lock() {
            Ok(mut inner) => {
                let Some(block) = inner.blocks.get_mut(&request.allocation.allocation_id) else {
                    return NxmemStatus::with_message(
                        NxmemStatusCode::UnknownAllocation,
                        "testplugin: that allocation id is not live on this mechanism",
                    );
                };
                let start = request.range.offset as usize;
                let released = block
                    .committed
                    .saturating_sub(start)
                    .min(request.range.bytes as usize);
                block.committed -= released;
                released as u64
            }
            Err(_) => {
                return NxmemStatus::with_message(
                    NxmemStatusCode::InternalError,
                    "testplugin: allocator state is poisoned",
                );
            }
        };
        // SAFETY: checked non-null above.
        unsafe { *unmapped_out = unmapped };
        NxmemStatus::ok()
    })
}

/// The test plugin's physical granularity.
///
/// Present so the mapped-byte estimate is conservative in the same way a real
/// VMM's would be: only the mechanism knows its granularity.
const GRANULE: u64 = 4096;

/// Alignment used for shared prefix storage.
const PREFIX_ALIGN: usize = 64;

fn round_up_to_granule(bytes: u64) -> u64 {
    bytes.div_ceil(GRANULE) * GRANULE
}

unsafe extern "C" fn plugin_mapped_bytes_for_ranges(
    ctx: *mut c_void,
    requests: *const NxmemRangeRequest,
    count: u64,
    mapped_out: *mut u64,
) -> NxmemStatus {
    catch_status_panic(|| {
        let state = plugin_entry!(ctx);
        if mapped_out.is_null() || (requests.is_null() && count != 0) {
            return NxmemStatus::with_message(
                NxmemStatusCode::InvalidArgument,
                "testplugin: mapped_bytes_for_ranges received a null pointer",
            );
        }
        let mut total = 0u64;
        for index in 0..count {
            // SAFETY: the host promises `count` readable records at `requests`.
            let request = unsafe { &*requests.add(index as usize) };
            if let Err(status) = state.check(&request.allocation) {
                return status;
            }
            total = total.saturating_add(round_up_to_granule(request.range.bytes));
        }
        // SAFETY: checked non-null above.
        unsafe { *mapped_out = total };
        NxmemStatus::ok()
    })
}

unsafe extern "C" fn plugin_mapped_bytes_for_allocation(
    ctx: *mut c_void,
    request: *const NxmemAllocRequest,
    mapped_out: *mut u64,
) -> NxmemStatus {
    catch_status_panic(|| {
        let state = plugin_entry!(ctx);
        if request.is_null() || mapped_out.is_null() {
            return NxmemStatus::with_message(
                NxmemStatusCode::InvalidArgument,
                "testplugin: mapped_bytes_for_allocation received a null pointer",
            );
        }
        // SAFETY: checked non-null.
        let request = unsafe { &*request };
        if let Err(status) = check_identity(
            state.mechanism_id,
            state.device,
            request.mechanism_id,
            request.device,
        ) {
            return status;
        }
        // SAFETY: checked non-null above.
        unsafe { *mapped_out = round_up_to_granule(request.bytes) };
        NxmemStatus::ok()
    })
}

unsafe extern "C" fn plugin_committed_bytes(
    ctx: *mut c_void,
    allocation: *const NxmemAllocation,
    committed_out: *mut u64,
) -> NxmemStatus {
    catch_status_panic(|| {
        let state = plugin_entry!(ctx);
        if allocation.is_null() || committed_out.is_null() {
            return NxmemStatus::with_message(
                NxmemStatusCode::InvalidArgument,
                "testplugin: committed_bytes received a null pointer",
            );
        }
        // SAFETY: checked non-null.
        let allocation = unsafe { &*allocation };
        if let Err(status) = state.check(allocation) {
            return status;
        }
        let committed = match state.inner.lock() {
            Ok(inner) => inner
                .blocks
                .get(&allocation.allocation_id)
                .map(|block| block.committed as u64)
                .unwrap_or(0),
            Err(_) => 0,
        };
        // SAFETY: checked non-null above.
        unsafe { *committed_out = committed };
        NxmemStatus::ok()
    })
}

// ─── shared-mapping slots ───────────────────────────────────────────────────

unsafe extern "C" fn plugin_create_shared_prefix(
    ctx: *mut c_void,
    mechanism_id: u64,
    bytes: u64,
    handle_out: *mut NxmemSharedPrefixHandle,
) -> NxmemStatus {
    catch_status_panic(|| {
        let state = plugin_entry!(ctx);
        if handle_out.is_null() {
            return NxmemStatus::with_message(
                NxmemStatusCode::InvalidArgument,
                "testplugin: create_shared_prefix received a null out-parameter",
            );
        }
        if mechanism_id != state.mechanism_id {
            return NxmemStatus::with_message(
                NxmemStatusCode::WrongMechanism,
                "testplugin: a prefix may only be created on its own mechanism",
            );
        }
        let layout = match layout_of(bytes as usize, PREFIX_ALIGN) {
            Ok(layout) => layout,
            Err(status) => return status,
        };
        // SAFETY: `layout` has a non-zero size and a valid alignment.
        let raw = unsafe { alloc(layout) };
        if raw.is_null() {
            return NxmemStatus::with_message(
                NxmemStatusCode::OutOfMemory,
                "testplugin: the backing host allocator refused the prefix",
            );
        }
        let handle_id = next_prefix_handle();
        match state.inner.lock() {
            Ok(mut inner) => {
                inner.prefixes.insert(
                    handle_id,
                    Prefix {
                        bytes: bytes as usize,
                        address: raw as usize,
                        references: 1,
                    },
                );
            }
            Err(_) => {
                // SAFETY: allocated just above and never handed out.
                unsafe { dealloc(raw, layout) };
                return NxmemStatus::with_message(
                    NxmemStatusCode::InternalError,
                    "testplugin: allocator state is poisoned",
                );
            }
        }
        counters().live_capabilities.fetch_add(1, Ordering::AcqRel);

        let mut handle = NxmemSharedPrefixHandle::zeroed();
        handle.mechanism_id = state.mechanism_id;
        handle.handle = handle_id;
        handle.device = state.device;
        handle.device_ptr = raw as u64;
        handle.committed_physical_bytes = round_up_to_granule(bytes);
        handle.mapped_bytes = 0;
        handle.requested_bytes = bytes;
        // SAFETY: checked non-null above.
        unsafe { *handle_out = handle };
        NxmemStatus::ok()
    })
}

unsafe extern "C" fn plugin_retain_shared_prefix(
    ctx: *mut c_void,
    handle: *const NxmemSharedPrefixHandle,
) -> NxmemStatus {
    catch_status_panic(|| {
        let state = plugin_entry!(ctx);
        if handle.is_null() {
            return NxmemStatus::with_message(
                NxmemStatusCode::InvalidArgument,
                "testplugin: retain_shared_prefix received a null handle",
            );
        }
        // SAFETY: checked non-null.
        let handle = unsafe { &*handle };
        if handle.mechanism_id != state.mechanism_id {
            return NxmemStatus::with_message(
                NxmemStatusCode::WrongMechanism,
                "testplugin: that prefix belongs to another mechanism",
            );
        }
        match state.inner.lock() {
            Ok(mut inner) => match inner.prefixes.get_mut(&handle.handle) {
                Some(prefix) => {
                    prefix.references += 1;
                    counters().live_capabilities.fetch_add(1, Ordering::AcqRel);
                    NxmemStatus::ok()
                }
                None => NxmemStatus::with_message(
                    NxmemStatusCode::UnknownAllocation,
                    "testplugin: that prefix is not live on this mechanism",
                ),
            },
            Err(_) => NxmemStatus::with_message(
                NxmemStatusCode::InternalError,
                "testplugin: allocator state is poisoned",
            ),
        }
    })
}

unsafe extern "C" fn plugin_release_shared_prefix(
    ctx: *mut c_void,
    handle: *const NxmemSharedPrefixHandle,
) -> NxmemStatus {
    catch_status_panic(|| {
        let state = plugin_entry!(ctx);
        if handle.is_null() {
            return NxmemStatus::with_message(
                NxmemStatusCode::InvalidArgument,
                "testplugin: release_shared_prefix received a null handle",
            );
        }
        // SAFETY: checked non-null.
        let handle = unsafe { &*handle };
        if handle.mechanism_id != state.mechanism_id {
            return NxmemStatus::with_message(
                NxmemStatusCode::WrongMechanism,
                "testplugin: that prefix belongs to another mechanism",
            );
        }
        match state.inner.lock() {
            Ok(mut inner) => {
                let Some(prefix) = inner.prefixes.get_mut(&handle.handle) else {
                    return NxmemStatus::with_message(
                        NxmemStatusCode::UnknownAllocation,
                        "testplugin: that prefix is not live on this mechanism",
                    );
                };
                prefix.references -= 1;
                counters().live_capabilities.fetch_sub(1, Ordering::AcqRel);
                if prefix.references == 0 {
                    let prefix = inner.prefixes.remove(&handle.handle).expect("just seen");
                    free_block(Block {
                        address: prefix.address,
                        bytes: prefix.bytes,
                        align: PREFIX_ALIGN,
                        committed: prefix.bytes,
                    });
                }
                NxmemStatus::ok()
            }
            Err(_) => NxmemStatus::with_message(
                NxmemStatusCode::InternalError,
                "testplugin: allocator state is poisoned",
            ),
        }
    })
}

unsafe extern "C" fn plugin_incremental_owned_bytes(
    ctx: *mut c_void,
    handle: *const NxmemSharedPrefixHandle,
    bytes_out: *mut u64,
) -> NxmemStatus {
    catch_status_panic(|| {
        let state = plugin_entry!(ctx);
        if handle.is_null() || bytes_out.is_null() {
            return NxmemStatus::with_message(
                NxmemStatusCode::InvalidArgument,
                "testplugin: incremental_owned_bytes received a null pointer",
            );
        }
        // SAFETY: checked non-null.
        let handle = unsafe { &*handle };
        // A foreign prefix is refused rather than costed as free.
        if handle.mechanism_id != state.mechanism_id {
            return NxmemStatus::with_message(
                NxmemStatusCode::WrongMechanism,
                "testplugin: that prefix belongs to another mechanism, so its incremental cost \
                 cannot be reported as zero",
            );
        }
        if handle.device != state.device {
            return NxmemStatus::with_message(
                NxmemStatusCode::WrongDevice,
                "testplugin: that prefix lives on another device",
            );
        }
        let known = match state.inner.lock() {
            Ok(inner) => inner.prefixes.contains_key(&handle.handle),
            Err(_) => false,
        };
        if !known {
            return NxmemStatus::with_message(
                NxmemStatusCode::UnknownAllocation,
                "testplugin: that prefix is not live on this mechanism",
            );
        }
        // The physical bytes were charged once when the prefix was created, so
        // an additional mapping owns nothing new.
        // SAFETY: checked non-null above.
        unsafe { *bytes_out = 0 };
        NxmemStatus::ok()
    })
}

unsafe extern "C" fn plugin_commit_shared_prefix(
    ctx: *mut c_void,
    request: *const NxmemSharedPrefixCommitRequest,
    info_out: *mut NxmemSharedPrefixCommitInfo,
) -> NxmemStatus {
    catch_status_panic(|| {
        let state = plugin_entry!(ctx);
        if request.is_null() || info_out.is_null() {
            return NxmemStatus::with_message(
                NxmemStatusCode::InvalidArgument,
                "testplugin: commit_shared_prefix received a null pointer",
            );
        }
        // SAFETY: checked non-null.
        let request = unsafe { &*request };
        if let Err(status) = state.check(&request.allocation) {
            return status;
        }
        if request.prefix.mechanism_id != state.mechanism_id {
            return NxmemStatus::with_message(
                NxmemStatusCode::WrongMechanism,
                "testplugin: that prefix belongs to another mechanism",
            );
        }
        let prefix_bytes = match state.inner.lock() {
            Ok(mut inner) => {
                let Some(prefix) = inner.prefixes.get(&request.prefix.handle).copied() else {
                    return NxmemStatus::with_message(
                        NxmemStatusCode::UnknownAllocation,
                        "testplugin: that prefix is not live on this mechanism",
                    );
                };
                let end = request.byte_offset as usize + prefix.bytes;
                let Some(block) = inner.blocks.get_mut(&request.allocation.allocation_id) else {
                    return NxmemStatus::with_message(
                        NxmemStatusCode::UnknownAllocation,
                        "testplugin: that allocation id is not live on this mechanism",
                    );
                };
                if end > block.bytes {
                    return NxmemStatus::with_message(
                        NxmemStatusCode::InvalidArgument,
                        "testplugin: the prefix does not fit in the allocation at that offset",
                    );
                }
                block.committed = block.committed.max(end);
                prefix.bytes as u64
            }
            Err(_) => {
                return NxmemStatus::with_message(
                    NxmemStatusCode::InternalError,
                    "testplugin: allocator state is poisoned",
                );
            }
        };

        let mut info = NxmemSharedPrefixCommitInfo::zeroed();
        // The prefix was charged once at creation, so mapping it again owns
        // nothing new but does map bytes. Keeping those two axes distinct is
        // the whole point of this record.
        info.additional_owned_bytes = 0;
        info.newly_mapped_bytes = round_up_to_granule(prefix_bytes);
        info.granules = info.newly_mapped_bytes / GRANULE;
        // SAFETY: checked non-null above.
        unsafe { *info_out = info };
        NxmemStatus::ok()
    })
}

// ─── factory slots ──────────────────────────────────────────────────────────

unsafe extern "C" fn plugin_open_allocator(
    ctx: *mut c_void,
    request: *const NxmemOpenRequest,
    allocator_out: *mut *const NxmemAllocatorVtable,
) -> NxmemStatus {
    catch_status_panic(|| {
        if ctx.is_null() || request.is_null() || allocator_out.is_null() {
            return NxmemStatus::with_message(
                NxmemStatusCode::InvalidArgument,
                "testplugin: open_allocator received a null pointer",
            );
        }
        // SAFETY: `ctx` is the `&'static Mechanism` this module published in
        // its factory table; the host only echoes it back.
        let mechanism = unsafe { &*(ctx as *const Mechanism) };
        // SAFETY: checked non-null.
        let request = unsafe { &*request };

        let missing = request.required_capability_flags & !mechanism.behaviour.capability_flags;
        if missing != 0 {
            return NxmemStatus::with_message(
                NxmemStatusCode::UnsupportedCapability,
                &format!(
                    "testplugin: mechanism `{}` does not provide the required capabilities \
                     {missing:#x}",
                    mechanism.name
                ),
            );
        }

        let mechanism_id = next_mechanism_id();
        let behaviour = mechanism.behaviour;

        if behaviour.short_allocator_struct {
            // A plugin built against a mismatched header publishes a vtable
            // the host cannot read. It must not allocate state first: the host
            // has nothing well-formed enough to hand the state back through.
            // Returning a static, stateless vtable is the only shape of this
            // bug that does not also leak.
            // SAFETY: checked non-null above.
            unsafe { *allocator_out = short_allocator_vtable() };
            return NxmemStatus::ok();
        }
        // The vtable's level is the lesser of what the mechanism implements
        // and what the host negotiated: an older mechanism inside a newer
        // module keeps working, and a newer mechanism never offers a slot an
        // older host would not know how to call.
        let abi_minor = behaviour.abi_minor.min(request.abi_minor);

        let backing_vtable =
            (behaviour.capability_flags & NXMEM_CAP_VIRTUAL_BACKING != 0).then(|| {
                let mut vtable = NxmemVirtualBackingVtable::zeroed();
                vtable.abi_minor = abi_minor;
                vtable.mechanism_id = mechanism_id;
                vtable.allocate_committed = Some(plugin_allocate_committed);
                vtable.commit_range = Some(plugin_commit_range);
                vtable.decommit_range = Some(plugin_decommit_range);
                vtable.mapped_bytes_for_ranges = Some(plugin_mapped_bytes_for_ranges);
                vtable.mapped_bytes_for_allocation = Some(plugin_mapped_bytes_for_allocation);
                vtable.committed_bytes = Some(plugin_committed_bytes);
                Box::new(vtable)
            });
        let shared_vtable = (behaviour.capability_flags & NXMEM_CAP_SHARED_MAPPING != 0).then(|| {
            let mut vtable = NxmemSharedMappingVtable::zeroed();
            vtable.abi_minor = abi_minor;
            vtable.mechanism_id = mechanism_id;
            vtable.create_shared_prefix = Some(plugin_create_shared_prefix);
            vtable.retain_shared_prefix = Some(plugin_retain_shared_prefix);
            vtable.release_shared_prefix = Some(plugin_release_shared_prefix);
            vtable.incremental_owned_bytes = Some(plugin_incremental_owned_bytes);
            vtable.commit_shared_prefix = Some(plugin_commit_shared_prefix);
            Box::new(vtable)
        });

        let mut allocator_vtable = NxmemAllocatorVtable::zeroed();
        allocator_vtable.abi_minor = abi_minor;
        allocator_vtable.mechanism_id = mechanism_id;
        allocator_vtable.device = request.device;
        allocator_vtable.capability_flags = behaviour.capability_flags;
        allocator_vtable.name = mechanism.c_name.as_ptr().cast();
        allocator_vtable.allocate = Some(plugin_allocate);
        allocator_vtable.deallocate = Some(plugin_deallocate);
        allocator_vtable.retain = Some(plugin_retain);
        allocator_vtable.release = Some(plugin_release);
        if behaviour.capability_flags & NXMEM_CAP_DEFERRED_RELEASE != 0 {
            allocator_vtable.enqueue_release = Some(plugin_enqueue_release);
            allocator_vtable.drain_releases = Some(plugin_drain_releases);
            allocator_vtable.pending_release_count = Some(plugin_pending_release_count);
        }
        if abi_minor >= 1 && behaviour.capability_flags & NXMEM_CAP_STRUCTURED_RELEASE != 0 {
            allocator_vtable.release_allocation = Some(plugin_release_allocation);
        }
        let state = Box::new(AllocatorState {
            mechanism_id,
            device: request.device,
            behaviour,
            refcount: AtomicU64::new(1),
            callbacks: request.callbacks,
            inner: Mutex::new(AllocatorInner::default()),
            allocator_vtable: Box::new(allocator_vtable),
            backing_vtable,
            shared_vtable,
        });

        // Wire the vtables to the state now that it has a stable address.
        let state_ptr = Box::into_raw(state);
        // SAFETY: `state_ptr` came from `Box::into_raw` immediately above and
        // nothing else references it yet, so this is the only live borrow.
        let state = unsafe { &mut *state_ptr };
        let allocator_ctx = state_ptr as *mut c_void;
        state.allocator_vtable.ctx = allocator_ctx;
        if let Some(backing) = state.backing_vtable.as_mut() {
            backing.ctx = allocator_ctx;
        }
        if let Some(shared) = state.shared_vtable.as_mut() {
            shared.ctx = allocator_ctx;
        }
        state.allocator_vtable.virtual_backing = state
            .backing_vtable
            .as_ref()
            .map_or(core::ptr::null(), |backing| &raw const **backing);
        state.allocator_vtable.shared_mapping = state
            .shared_vtable
            .as_ref()
            .map_or(core::ptr::null(), |shared| &raw const **shared);

        counters().live_allocators.fetch_add(1, Ordering::AcqRel);
        // SAFETY: checked non-null above.
        unsafe { *allocator_out = &raw const *state.allocator_vtable };
        NxmemStatus::ok()
    })
}

/// A deliberately undersized allocator vtable.
///
/// `struct_size` claims fewer bytes than even the baseline prefix needs, so
/// the host must refuse it before reading a single function pointer. It is
/// stateless and `'static`, so refusing it leaks nothing.
struct ShortVtable(NxmemAllocatorVtable);

// SAFETY: immutable after construction and holding only null pointers.
unsafe impl Send for ShortVtable {}
// SAFETY: as above.
unsafe impl Sync for ShortVtable {}

fn short_allocator_vtable() -> *const NxmemAllocatorVtable {
    static VTABLE: OnceLock<ShortVtable> = OnceLock::new();
    &VTABLE
        .get_or_init(|| {
            let mut vtable = NxmemAllocatorVtable::zeroed();
            vtable.struct_size = (NxmemAllocatorVtable::MIN_STRUCT_SIZE_MINOR_0 as u32) / 2;
            ShortVtable(vtable)
        })
        .0
}

unsafe extern "C" fn plugin_factory_release(_ctx: *mut c_void) {
    // Factories are `&'static Mechanism`, so there is nothing to free. The
    // slot still exists because the contract requires an explicit release for
    // every ABI-owned object, and a real plugin may own state here.
    catch_void_panic(|| {
        FACTORY_RELEASES.fetch_add(1, Ordering::AcqRel);
    });
}

/// How many times the host has released a factory.
///
/// A statically linked `rlib` copy of this crate has its *own* copy of this
/// static, so a test cannot observe the loaded module's count by reading it
/// directly. [`NxmemTestpluginFactoryReleases`] exists to expose it from the
/// module the host actually loaded.
pub static FACTORY_RELEASES: AtomicU64 = AtomicU64::new(0);

/// The name of the test-only introspection symbol.
pub const SYMBOL_FACTORY_RELEASES: &[u8] = b"NxmemTestpluginFactoryReleases\0";

/// Test-only introspection: how many factories this module has had released.
///
/// **Not part of the nxmem ABI.** It exists so the ABI tests can assert on the
/// loaded module's own state rather than on a separate statically linked copy
/// of this crate.
#[unsafe(no_mangle)]
pub extern "C" fn NxmemTestpluginFactoryReleases() -> u64 {
    FACTORY_RELEASES.load(Ordering::Acquire)
}

/// Set once the module has published its factory vtables.
static FACTORIES_BUILT: AtomicBool = AtomicBool::new(false);

/// The published factory table.
///
/// A vtable holds raw pointers, so it is not automatically `Send`/`Sync`. The
/// table is built once and never mutated afterwards, and every pointer in it
/// addresses `'static` module data, so sharing it is sound.
struct FactoryTable(Vec<NxmemAllocatorFactoryVtable>);

// SAFETY: immutable after construction; every pointer targets `'static` data
// owned by this module.
unsafe impl Send for FactoryTable {}
// SAFETY: as above.
unsafe impl Sync for FactoryTable {}

fn factory_vtables() -> &'static [NxmemAllocatorFactoryVtable] {
    static VTABLES: OnceLock<FactoryTable> = OnceLock::new();
    &VTABLES
        .get_or_init(|| {
            FACTORIES_BUILT.store(true, Ordering::Release);
            FactoryTable(build_factory_table())
        })
        .0
}

fn build_factory_table() -> Vec<NxmemAllocatorFactoryVtable> {
    {
        MECHANISMS
            .iter()
            .map(|mechanism| {
                let mut vtable = NxmemAllocatorFactoryVtable::zeroed();
                vtable.abi_minor = mechanism.behaviour.abi_minor;
                vtable.name = mechanism.c_name.as_ptr().cast();
                // Host memory keeps the test suite portable: it exercises the
                // whole ABI on a machine with no accelerator.
                vtable.device = NxmemDeviceId::HOST;
                vtable.capability_flags = mechanism.behaviour.capability_flags;
                vtable.ctx = (mechanism as *const Mechanism).cast_mut().cast();
                vtable.open_allocator = Some(plugin_open_allocator);
                vtable.release = Some(plugin_factory_release);
                vtable
            })
            .collect()
    }
}

// ─── entry points ───────────────────────────────────────────────────────────

fn negotiate_impl(
    request: *const NxmemNegotiateRequest,
    response_out: *mut NxmemNegotiateResponse,
) -> NxmemStatus {
    // The module implements the current contract; each *mechanism* then
    // declares its own level, which is how one module can ship both a current
    // and a deliberately older mechanism.
    // SAFETY: `negotiate_as` null-checks and size-checks both pointers before
    // reading any field.
    unsafe {
        negotiate_as(
            request,
            response_out,
            NxmemVersionRange {
                major_min: NXMEM_ABI_VERSION_MAJOR,
                major_max: NXMEM_ABI_VERSION_MAJOR,
                minor_min: 0,
                minor_max: NXMEM_ABI_VERSION_MINOR,
            },
            NXMEM_CAP_ALLOCATOR
                | NXMEM_CAP_VIRTUAL_BACKING
                | NXMEM_CAP_SHARED_MAPPING
                | NXMEM_CAP_DEFERRED_RELEASE
                | NXMEM_CAP_STRUCTURED_RELEASE,
        )
    }
}

fn create_factories_impl(
    out_factories: *mut *const NxmemAllocatorFactoryVtable,
    max_factories: u64,
    out_count: *mut u64,
) -> NxmemStatus {
    if out_factories.is_null() || out_count.is_null() {
        return NxmemStatus::with_message(
            NxmemStatusCode::InvalidArgument,
            "testplugin: create_factories received a null pointer",
        );
    }
    let vtables = factory_vtables();
    let count = (vtables.len() as u64).min(max_factories);
    for (index, vtable) in vtables.iter().take(count as usize).enumerate() {
        // SAFETY: the host promises `max_factories` writable slots and `count`
        // never exceeds it.
        unsafe { *out_factories.add(index) = vtable as *const _ };
    }
    // SAFETY: checked non-null above.
    unsafe { *out_count = count };
    NxmemStatus::ok()
}

fn unload_readiness_impl(report_out: *mut NxmemUnloadReport) -> NxmemStatus {
    if report_out.is_null() {
        return NxmemStatus::with_message(
            NxmemStatusCode::InvalidArgument,
            "testplugin: unload readiness received a null pointer",
        );
    }
    let counters = counters();
    let mut report = NxmemUnloadReport::zeroed();
    report.live_allocators = counters.live_allocators.load(Ordering::Acquire);
    report.live_allocations = counters.live_allocations.load(Ordering::Acquire);
    report.live_views = counters.live_views.load(Ordering::Acquire);
    report.live_capabilities = counters.live_capabilities.load(Ordering::Acquire);
    report.queued_releases = counters.queued_releases.load(Ordering::Acquire);
    // SAFETY: checked non-null above.
    unsafe { *report_out = report };
    NxmemStatus::ok()
}

onnx_runtime_memory_abi::export_nxmem_plugin! {
    negotiate: negotiate_impl,
    factories: create_factories_impl,
    unload_readiness: unload_readiness_impl,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_published_names_match_the_mechanism_table() {
        let actual: Vec<&str> = MECHANISMS.iter().map(|mechanism| mechanism.name).collect();
        assert_eq!(actual, MECHANISM_NAMES);
    }

    #[test]
    fn every_mechanism_name_is_valid_c_and_utf8() {
        for mechanism in MECHANISMS {
            assert_eq!(
                mechanism.c_name.to_str().expect("valid UTF-8"),
                mechanism.name
            );
        }
    }

    #[test]
    fn every_mechanism_claims_the_allocator_capability() {
        for mechanism in MECHANISMS {
            assert_ne!(
                mechanism.behaviour.capability_flags & NXMEM_CAP_ALLOCATOR,
                0,
                "{} must be able to allocate",
                mechanism.name
            );
        }
    }

    #[test]
    fn the_legacy_mechanism_stays_on_the_baseline_prefix() {
        let legacy = MECHANISMS
            .iter()
            .find(|mechanism| mechanism.name == "legacy-1-0")
            .expect("the legacy mechanism exists");
        assert_eq!(legacy.behaviour.abi_minor, 0);
        assert_eq!(
            legacy.behaviour.capability_flags & NXMEM_CAP_STRUCTURED_RELEASE,
            0,
            "structured release does not exist at minor 0"
        );
    }

    #[test]
    fn granule_rounding_is_conservative() {
        assert_eq!(round_up_to_granule(0), 0);
        assert_eq!(round_up_to_granule(1), GRANULE);
        assert_eq!(round_up_to_granule(GRANULE), GRANULE);
        assert_eq!(round_up_to_granule(GRANULE + 1), GRANULE * 2);
    }

    #[test]
    fn the_factory_table_is_built_lazily_and_stays_stable() {
        let first = factory_vtables().as_ptr();
        let second = factory_vtables().as_ptr();
        assert_eq!(first, second, "the factory table must be stable");
        assert!(FACTORIES_BUILT.load(Ordering::Acquire));
        assert_eq!(factory_vtables().len(), MECHANISMS.len());
    }
}
