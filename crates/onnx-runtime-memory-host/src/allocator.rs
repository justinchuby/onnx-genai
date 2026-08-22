//! Wrapping an nxmem allocator vtable in the internal Rust interfaces.
//!
//! # Lock discipline
//!
//! This adapter keeps one small `Mutex` mapping a live pointer to the
//! `allocation_id` the plugin knows it by. **No ABI call is ever made while
//! that lock is held.** Every path takes the lock, mutates the map, drops the
//! guard, and only then enters the plugin. That mirrors the rule the rest of
//! the memory stack already obeys for trait objects — no trait-object call
//! under a governance lock — and matters more here, because a plugin may
//! block, call back into the host, or spawn threads.
//!
//! # Identity
//!
//! `allocation_id` comes from a monotonic counter and is **never**
//! pointer-derived, so a freed-and-reused address can never be mistaken for
//! the allocation that previously lived there. A pointer the host does not
//! recognise is refused outright rather than retried by address.

use std::collections::HashMap;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use onnx_runtime_memory_abi::{
    NXMEM_CAP_DEFERRED_RELEASE, NXMEM_CAP_SHARED_MAPPING, NXMEM_CAP_STRUCTURED_RELEASE,
    NXMEM_CAP_VIRTUAL_BACKING, NXMEM_RELEASE_COMPLETE, NXMEM_RELEASE_FAILED,
    NXMEM_RELEASE_QUARANTINED, NxmemAllocRequest, NxmemAllocResult, NxmemAllocation,
    NxmemAllocatorVtable, NxmemByteRange, NxmemHostCallbacks, NxmemOpenRequest, NxmemRangeRequest,
    NxmemReclaimRequest, NxmemReleaseCompletion, NxmemReleaseOutcome, NxmemSharedMappingVtable,
    NxmemSharedPrefixCommitInfo, NxmemSharedPrefixCommitRequest, NxmemSharedPrefixHandle,
    NxmemStatus, NxmemStatusCode, NxmemVirtualBackingVtable, catch_status_panic,
};
use onnx_runtime_memory_api::{
    AllocationCommitRange, AllocationReleaseOutcome, DeviceAllocator, DeviceKey, MemoryError,
    QuarantineReason, ReleaseAccounting, ResidualOwnership, SharedDevicePrefix, SharedMapping,
    SharedPrefixCommitInfo, Tier, VirtualBacking,
};

use crate::error::{PluginError, status_to_memory_error};
use crate::loader::{PluginFactory, PluginModule, device_id, device_key};

/// The host's reclaim hook, offered to a plugin under memory pressure.
///
/// # Threading and reentrancy
///
/// * Called from inside a plugin call the host is currently making, and from
///   plugin-owned worker threads.
/// * **Must not be invoked, and must not itself acquire, any governance lock.**
///   The host releases every accounting, charge, and registration lock before
///   entering a plugin, so a reclaim hook that re-acquires one is safe only if
///   it never blocks on a thread that is itself inside a plugin call.
/// * Returning an error is a normal outcome. The plugin must handle it and
///   must not abort.
pub trait HostReclaim: Send + Sync + std::fmt::Debug {
    /// Free up to `bytes` of host-cached memory on `device`.
    ///
    /// Returns the bytes actually reclaimed. Zero is a valid answer.
    fn request_reclaim(&self, device: DeviceKey, bytes: u64) -> Result<u64, String>;
}

/// One retired deferred release, as reported by the plugin.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RetiredRelease {
    /// The ticket the plugin issued when the release was queued.
    pub ticket: u64,
    /// The allocation the ticket refers to.
    pub allocation_id: u64,
    /// The allocation size the release was prepared for.
    pub allocation_bytes: u64,
    /// Bytes whose mapping reference transitioned to unmapped.
    pub unmapped_bytes: u64,
    /// Whether the release completed, quarantined, or failed.
    pub state: u32,
}

/// Host state reachable from a plugin callback.
///
/// Lives behind a stable heap address for the whole life of the allocator: the
/// plugin stores a raw pointer to it in its own state, and the contract
/// promises the table outlives the allocator's final `release` and every
/// queued release naming it.
#[derive(Debug)]
struct HostBridge {
    device: DeviceKey,
    reclaim: Option<Arc<dyn HostReclaim>>,
    reclaim_calls: AtomicU64,
    reclaim_failures: AtomicU64,
    retired: Mutex<Vec<RetiredRelease>>,
    module: Arc<PluginModule>,
    /// Deferred releases queued against **this allocator** that have not yet
    /// reported completion.
    ///
    /// The module-wide tally in [`PluginModule`] answers "may the module
    /// unload"; this one answers the narrower question "may this callback
    /// table be freed", which is what the plugin's stored
    /// `*const NxmemHostCallbacks` depends on. They are deliberately separate:
    /// an allocator can be dropped long before the module is, and it is the
    /// allocator's drop that would otherwise free the table.
    outstanding_releases: AtomicU64,
}

/// The callback table and the state behind it, as one heap-resident unit.
///
/// The plugin captures `&callbacks` at open and dereferences it later, from
/// its own threads. `callbacks.host_ctx` points at `bridge`, in the same
/// allocation. The ABI contract ([`onnx_runtime_memory_abi`], `NxmemHostCallbacks`)
/// says the host keeps this alive past the allocator's final `release` **and
/// past every queued release**, so its lifetime cannot simply be the
/// allocator's: a queued release outlives the allocator by construction.
///
/// Boxing the whole thing gives both members one stable address for the life
/// of the context, and lets [`AllocatorCore::drop`] decide its fate rather
/// than having the compiler free it unconditionally.
#[derive(Debug)]
struct HostCallbackContext {
    bridge: HostBridge,
    callbacks: NxmemHostCallbacks,
}

impl HostCallbackContext {
    /// Build a context whose callback table already points at its own bridge.
    fn new(minor: u32, bridge: HostBridge) -> Box<Self> {
        let mut context = Box::new(Self {
            bridge,
            callbacks: NxmemHostCallbacks {
                struct_size: core::mem::size_of::<NxmemHostCallbacks>() as u32,
                abi_minor: minor,
                // Patched below, once the box has its final address.
                host_ctx: core::ptr::null_mut(),
                request_reclaim: Some(host_request_reclaim),
                release_completed: Some(host_release_completed),
            },
        });
        // Boxing first and patching second is what makes the pointer stable:
        // moving the `Box` moves the pointer, never the allocation it names.
        context.callbacks.host_ctx = (&raw const context.bridge) as *mut core::ffi::c_void;
        context
    }
}

/// `request_reclaim` trampoline.
///
/// # Safety
///
/// `host_ctx` must be the pointer the host installed in the callback table and
/// `request` / `reclaimed_out` must be valid for the call.
unsafe extern "C" fn host_request_reclaim(
    host_ctx: *mut core::ffi::c_void,
    request: *const NxmemReclaimRequest,
    reclaimed_out: *mut u64,
) -> NxmemStatus {
    catch_status_panic(|| {
        if host_ctx.is_null() || request.is_null() || reclaimed_out.is_null() {
            return NxmemStatus::with_message(
                NxmemStatusCode::InvalidArgument,
                "nxmem: request_reclaim was called with a null pointer",
            );
        }
        // SAFETY: checked non-null; the host installed this pointer and keeps
        // the bridge alive past the allocator's final release.
        let bridge = unsafe { &*(host_ctx as *const HostBridge) };
        // SAFETY: checked non-null; the plugin passes a valid request.
        let request = unsafe { &*request };
        bridge.reclaim_calls.fetch_add(1, Ordering::AcqRel);

        let Some(reclaim) = bridge.reclaim.as_ref() else {
            bridge.reclaim_failures.fetch_add(1, Ordering::AcqRel);
            return NxmemStatus::with_message(
                NxmemStatusCode::UnsupportedCapability,
                "nxmem: this host offers no reclaim path",
            );
        };
        let Some(device) = device_key(request.device) else {
            bridge.reclaim_failures.fetch_add(1, Ordering::AcqRel);
            return NxmemStatus::with_message(
                NxmemStatusCode::InvalidArgument,
                "nxmem: the reclaim request named a tier this host does not know",
            );
        };
        if device != bridge.device {
            bridge.reclaim_failures.fetch_add(1, Ordering::AcqRel);
            return NxmemStatus::with_message(
                NxmemStatusCode::WrongDevice,
                "nxmem: the reclaim request named a device this allocator does not serve",
            );
        }

        match reclaim.request_reclaim(device, request.bytes) {
            Ok(reclaimed) => {
                // SAFETY: checked non-null above.
                unsafe { *reclaimed_out = reclaimed };
                NxmemStatus::ok()
            }
            Err(reason) => {
                bridge.reclaim_failures.fetch_add(1, Ordering::AcqRel);
                // SAFETY: checked non-null above. A failing callback must still
                // leave the out-parameter defined.
                unsafe { *reclaimed_out = 0 };
                NxmemStatus::with_message(NxmemStatusCode::CallbackFailed, &reason)
            }
        }
    })
}

/// `release_completed` trampoline.
///
/// # Safety
///
/// `host_ctx` must be the pointer the host installed and `completion` must be
/// valid for the call.
unsafe extern "C" fn host_release_completed(
    host_ctx: *mut core::ffi::c_void,
    completion: *const NxmemReleaseCompletion,
) -> NxmemStatus {
    catch_status_panic(|| {
        if host_ctx.is_null() || completion.is_null() {
            return NxmemStatus::with_message(
                NxmemStatusCode::InvalidArgument,
                "nxmem: release_completed was called with a null pointer",
            );
        }
        // SAFETY: checked non-null; the host keeps the bridge alive past every
        // queued release that can name it.
        let bridge = unsafe { &*(host_ctx as *const HostBridge) };
        // SAFETY: checked non-null.
        let completion = unsafe { &*completion };

        let record = RetiredRelease {
            ticket: completion.ticket,
            allocation_id: completion.allocation_id,
            allocation_bytes: completion.outcome.allocation_bytes,
            unmapped_bytes: completion.outcome.unmapped_bytes,
            state: completion.outcome.state,
        };
        match bridge.retired.lock() {
            Ok(mut retired) => retired.push(record),
            Err(_) => {
                return NxmemStatus::with_message(
                    NxmemStatusCode::InternalError,
                    "nxmem: the host completion log is poisoned",
                );
            }
        }
        bridge.module.release_retired();
        bridge.module.allocation_closed();
        // Last, because it is what permits the table this callback is running
        // against to be freed. Decrementing it earlier would open a window in
        // which `AllocatorCore::drop` could free the bridge under our feet.
        bridge.outstanding_releases.fetch_sub(1, Ordering::AcqRel);
        NxmemStatus::ok()
    })
}

/// A live allocation as the host knows it.
#[derive(Debug, Clone, Copy)]
struct LiveAllocation {
    allocation_id: u64,
    bytes: usize,
    align: usize,
}

/// How many times [`AllocatorCore::drop`] asks a plugin to drain before giving
/// up and leaking the callback table instead.
///
/// Draining is a courtesy, not an obligation: the plugin owns the queue and is
/// entitled to hold it. The bound exists so a plugin that reports progress it
/// is not making cannot hang a `drop`.
const MAX_DRAIN_PASSES_ON_DROP: usize = 16;

/// How many host callback tables this process has deliberately leaked.
static LEAKED_CALLBACK_TABLES: AtomicU64 = AtomicU64::new(0);

/// The shared core of a plugin-backed mechanism.
///
/// Every capability view holds an `Arc` of this, so the plugin allocator is
/// released only once the allocator **and** all of its capability views are
/// gone. No back-reference is stored anywhere, so this introduces no reference
/// cycle.
#[derive(Debug)]
pub struct AllocatorCore {
    vtable: NxmemAllocatorVtable,
    backing: Option<NxmemVirtualBackingVtable>,
    shared: Option<NxmemSharedMappingVtable>,
    device: DeviceKey,
    mechanism_id: u64,
    name: String,
    negotiated_minor: u32,
    capability_flags: u64,
    next_allocation_id: AtomicU64,
    live: Mutex<HashMap<usize, LiveAllocation>>,
    /// The callback table the plugin holds a raw pointer to.
    ///
    /// `Option` so [`Drop`] can *take* it: the table must survive not only
    /// this allocator's final `release` but every queued release that can
    /// still name it, and a queued release can outlive the allocator. Taking
    /// it lets `drop` decide between freeing and deliberately leaking, which a
    /// plain field cannot express.
    context: Option<Box<HostCallbackContext>>,
    module: Arc<PluginModule>,
}

// SAFETY: the vtables are plain-data copies of function pointers plus an
// opaque `ctx`. The nxmem contract requires every slot to be callable from any
// host thread and requires the plugin to synchronise its own state.
unsafe impl Send for AllocatorCore {}
// SAFETY: as above.
unsafe impl Sync for AllocatorCore {}

impl AllocatorCore {
    /// The mechanism instance id, used to reject cross-provider misuse.
    pub fn mechanism_id(&self) -> u64 {
        self.mechanism_id
    }

    /// The mechanism name published by the plugin.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The capabilities this mechanism actually provides.
    pub fn capability_flags(&self) -> u64 {
        self.capability_flags
    }

    /// The nxmem minor level this allocator's vtable was read at.
    pub fn abi_minor(&self) -> u32 {
        self.vtable.abi_minor.min(self.negotiated_minor)
    }

    /// The callback context, which exists for the whole life of the core.
    ///
    /// It is only ever `None` inside [`Drop::drop`], after the context has
    /// been taken to decide its fate.
    fn context(&self) -> &HostCallbackContext {
        self.context
            .as_ref()
            .expect("the callback context is taken only while the allocator core is being dropped")
    }

    fn bridge(&self) -> &HostBridge {
        &self.context().bridge
    }

    /// How many times the plugin has called back into the host for reclaim.
    pub fn reclaim_calls(&self) -> u64 {
        self.bridge().reclaim_calls.load(Ordering::Acquire)
    }

    /// How many of those reclaim calls the host refused.
    pub fn reclaim_failures(&self) -> u64 {
        self.bridge().reclaim_failures.load(Ordering::Acquire)
    }

    /// Deferred releases queued against this allocator that have not yet
    /// reported completion.
    ///
    /// While this is non-zero the plugin may still dereference the host
    /// callback table from one of its own threads, so the table must not be
    /// freed. It is exposed because "may this be torn down" is a question a
    /// caller is entitled to ask before dropping the allocator.
    pub fn outstanding_releases(&self) -> u64 {
        self.bridge().outstanding_releases.load(Ordering::Acquire)
    }

    /// The callback table handed to the plugin.
    ///
    /// Exposed so a caller can confirm the table the plugin holds is the one
    /// this host installed, and that its `abi_minor` matches negotiation.
    pub fn host_callbacks(&self) -> &NxmemHostCallbacks {
        &self.context().callbacks
    }

    /// Whether the plugin published a structured-release slot that survived
    /// prefix negotiation.
    ///
    /// This reports the *slot*, not the capability: a mechanism that declares
    /// minor 0 must read back with the slot nulled even if the bytes at that
    /// offset in the sender's memory were something else entirely.
    pub fn publishes_structured_release_slot(&self) -> bool {
        self.vtable.release_allocation.is_some()
    }

    /// How many host callback tables this process has deliberately leaked.
    ///
    /// One per allocator that was dropped while a queued release could still
    /// name its callback table. Process-wide, because the tables are freed —
    /// or not — independently of any one allocator.
    ///
    /// A non-zero value is not corruption; it is the cost of refusing to hand
    /// a plugin thread a dangling pointer. Treat it as a leak to be fixed by
    /// draining releases before dropping the allocator.
    pub fn leaked_callback_tables() -> u64 {
        LEAKED_CALLBACK_TABLES.load(Ordering::Acquire)
    }

    /// Deferred releases the plugin has reported as retired, in report order.
    pub fn retired_releases(&self) -> Vec<RetiredRelease> {
        self.bridge()
            .retired
            .lock()
            .map(|retired| retired.clone())
            .unwrap_or_default()
    }

    /// How many allocations the host still tracks for this mechanism.
    pub fn live_allocation_count(&self) -> usize {
        self.live.lock().map(|live| live.len()).unwrap_or(0)
    }

    /// Ask the plugin to retire everything it has queued, before teardown.
    ///
    /// Bounded rather than looping to completion: a mechanism is entitled to
    /// refuse to drain (the `sticky` mechanism in the test plugin does exactly
    /// that, and a real stream-ordered mechanism may simply not be ready), and
    /// spinning on it inside `drop` would hang the caller. When the plugin
    /// declines, the outstanding count stays non-zero and the caller's
    /// callback table is leaked rather than freed, which is the honest
    /// outcome.
    fn retire_queued_releases_on_drop(&self) {
        let Some(drain) = self.vtable.drain_releases else {
            return;
        };
        // Each pass must make progress or the loop stops. The bound is a
        // belt-and-braces guard against a plugin that reports a non-zero
        // retirement without ever emptying its queue.
        for _ in 0..MAX_DRAIN_PASSES_ON_DROP {
            if self.bridge().outstanding_releases.load(Ordering::Acquire) == 0 {
                return;
            }
            let mut retired = 0u64;
            // SAFETY: `ctx` came from this vtable and `retired` is a valid
            // local. No host lock is held, which is required because the
            // plugin calls back into the host from inside this call.
            let status = unsafe { drain(self.vtable.ctx, u64::MAX, &raw mut retired) };
            if !status.is_ok() || retired == 0 {
                return;
            }
        }
    }

    fn next_allocation_id(&self) -> u64 {
        // Monotonic and explicitly not pointer-derived, so a reused address
        // can never alias a previous allocation's identity.
        self.next_allocation_id.fetch_add(1, Ordering::AcqRel) + 1
    }

    fn allocation_record(&self, ptr: NonNull<u8>) -> Option<LiveAllocation> {
        self.live
            .lock()
            .ok()?
            .get(&(ptr.as_ptr() as usize))
            .copied()
    }

    /// Remove a live record, **releasing the lock before any ABI call**.
    fn take_allocation(&self, ptr: NonNull<u8>) -> Option<LiveAllocation> {
        let mut live = self.live.lock().ok()?;
        let record = live.remove(&(ptr.as_ptr() as usize));
        drop(live);
        record
    }

    fn insert_allocation(&self, ptr: NonNull<u8>, record: LiveAllocation) {
        if let Ok(mut live) = self.live.lock() {
            live.insert(ptr.as_ptr() as usize, record);
        }
    }

    fn abi_allocation(&self, ptr: NonNull<u8>, record: LiveAllocation) -> NxmemAllocation {
        NxmemAllocation::new(
            self.mechanism_id,
            record.allocation_id,
            device_id(self.device),
            ptr.as_ptr(),
            record.bytes as u64,
            record.align as u64,
        )
    }

    fn alloc_request(&self, allocation_id: u64, bytes: usize, align: usize) -> NxmemAllocRequest {
        NxmemAllocRequest::new(
            self.mechanism_id,
            allocation_id,
            device_id(self.device),
            bytes as u64,
            align as u64,
        )
    }

    fn tier(&self) -> Tier {
        self.device.tier
    }

    /// Common tail of every allocation path.
    fn finish_allocation(
        &self,
        request: &NxmemAllocRequest,
        result: NxmemAllocResult,
        bytes: usize,
        align: usize,
    ) -> Result<NonNull<u8>, MemoryError> {
        let ptr = NonNull::new(result.ptr).ok_or_else(|| MemoryError::AllocationFailed {
            tier: self.tier().name(),
            requested: bytes as u64,
            reason: String::from(
                "the memory plugin reported success but returned a null pointer; a successful \
                 allocation must return a live, unique address",
            ),
        })?;
        if !(ptr.as_ptr() as usize).is_multiple_of(align.max(1)) {
            // The plugin broke the alignment contract. Hand the bytes straight
            // back rather than letting a misaligned pointer escape.
            let record = LiveAllocation {
                allocation_id: request.allocation_id,
                bytes,
                align,
            };
            let allocation = self.abi_allocation(ptr, record);
            self.raw_release(&allocation);
            return Err(MemoryError::AllocationFailed {
                tier: self.tier().name(),
                requested: bytes as u64,
                reason: format!(
                    "the memory plugin returned an address that is not aligned to {align}; the \
                     bytes were handed straight back"
                ),
            });
        }
        self.insert_allocation(
            ptr,
            LiveAllocation {
                allocation_id: request.allocation_id,
                bytes,
                align,
            },
        );
        self.module.allocation_opened();
        Ok(ptr)
    }

    /// Terminal release of an allocation the host is no longer tracking.
    ///
    /// Used on the error paths above, where the record was never installed.
    fn raw_release(&self, allocation: &NxmemAllocation) {
        if let Some(release) = self.structured_release_slot() {
            let mut outcome = NxmemReleaseOutcome::zeroed();
            // SAFETY: `ctx` came from this vtable; `allocation` and `outcome`
            // are valid locals. No host lock is held.
            let _ = unsafe { release(self.vtable.ctx, allocation, &raw mut outcome) };
            return;
        }
        if let Some(deallocate) = self.vtable.deallocate {
            let mut unmapped = 0u64;
            // SAFETY: as above.
            let _ = unsafe { deallocate(self.vtable.ctx, allocation, &raw mut unmapped) };
        }
    }

    /// The structured-release slot, if it was both negotiated and provided.
    ///
    /// Negotiation fixes a ceiling; the vtable declares its own level. Both
    /// must permit the slot before it is called, which is what makes an older
    /// mechanism inside a newer module safe.
    fn structured_release_slot(
        &self,
    ) -> Option<
        unsafe extern "C" fn(
            *mut core::ffi::c_void,
            *const NxmemAllocation,
            *mut NxmemReleaseOutcome,
        ) -> NxmemStatus,
    > {
        if self.capability_flags & NXMEM_CAP_STRUCTURED_RELEASE == 0 {
            return None;
        }
        if self.negotiated_minor < 1 || self.vtable.abi_minor < 1 {
            return None;
        }
        self.vtable.release_allocation
    }

    /// The canonical release path. Never called with any lock held.
    fn release_allocation(
        &self,
        ptr: NonNull<u8>,
        bytes: usize,
        align: usize,
    ) -> AllocationReleaseOutcome {
        // Take the record and drop the guard before entering the plugin.
        let Some(record) = self.take_allocation(ptr) else {
            // An address the host does not recognise is refused outright. It
            // is never retried by pointer, because a freed-and-reused address
            // would then free somebody else's live allocation.
            return AllocationReleaseOutcome::failed(format!(
                "the memory plugin host does not recognise address {:p} as a live allocation of \
                 mechanism {}; it was already released, or it belongs to another mechanism",
                ptr.as_ptr(),
                self.mechanism_id
            ));
        };

        if record.bytes != bytes || record.align != align {
            // Put the record back: nothing was mutated, so the allocation is
            // exactly as live as the caller left it.
            self.insert_allocation(ptr, record);
            return AllocationReleaseOutcome::failed(format!(
                "release of address {:p} named {bytes} bytes at align {align} but the allocation \
                 is {} bytes at align {}; nothing was released",
                ptr.as_ptr(),
                record.bytes,
                record.align
            ));
        }

        let allocation = self.abi_allocation(ptr, record);

        if let Some(release) = self.structured_release_slot() {
            let mut outcome = NxmemReleaseOutcome::zeroed();
            // SAFETY: `ctx` came from this vtable; both pointers address valid
            // locals that outlive the call. No host lock is held.
            let status =
                unsafe { release(self.vtable.ctx, &raw const allocation, &raw mut outcome) };
            if !status.is_ok() {
                self.insert_allocation(ptr, record);
                return AllocationReleaseOutcome::failed(format!(
                    "the memory plugin refused release without mutating anything: {}",
                    status.describe()
                ));
            }
            return self.interpret_outcome(ptr, record, outcome);
        }

        let Some(deallocate) = self.vtable.deallocate else {
            self.insert_allocation(ptr, record);
            return AllocationReleaseOutcome::failed(String::from(
                "the memory plugin exposes no release slot at all; the allocation cannot be \
                 handed back",
            ));
        };
        let mut unmapped = 0u64;
        // SAFETY: as above.
        let status =
            unsafe { deallocate(self.vtable.ctx, &raw const allocation, &raw mut unmapped) };
        if !status.is_ok() {
            self.insert_allocation(ptr, record);
            return AllocationReleaseOutcome::failed(format!(
                "the memory plugin refused release without mutating anything: {}",
                status.describe()
            ));
        }
        self.module.allocation_closed();
        AllocationReleaseOutcome::complete(ReleaseAccounting {
            allocation_bytes: record.bytes as u64,
            unmapped_bytes: unmapped,
        })
    }

    fn interpret_outcome(
        &self,
        ptr: NonNull<u8>,
        record: LiveAllocation,
        outcome: NxmemReleaseOutcome,
    ) -> AllocationReleaseOutcome {
        let accounting = ReleaseAccounting {
            allocation_bytes: outcome.allocation_bytes.max(record.bytes as u64),
            unmapped_bytes: outcome.unmapped_bytes,
        };
        match outcome.state {
            NXMEM_RELEASE_COMPLETE => {
                self.module.allocation_closed();
                AllocationReleaseOutcome::complete(accounting)
            }
            NXMEM_RELEASE_QUARANTINED => {
                // The allocation is gone from the host's live map and the
                // plugin kept residual ownership. It must never be reused.
                self.module.allocation_closed();
                AllocationReleaseOutcome::quarantined(
                    accounting,
                    ResidualOwnership {
                        state: onnx_runtime_memory_api::AllocationReleaseState::PartiallyUnmapped,
                        reason: QuarantineReason::PartialRelease,
                        retained_bytes: outcome.residual_owned_bytes,
                        address: ptr.as_ptr() as usize,
                        align: record.align,
                    },
                )
            }
            NXMEM_RELEASE_FAILED => {
                // The plugin promises nothing was mutated, so the allocation is
                // as live as the caller left it. Restore the record.
                self.insert_allocation(ptr, record);
                AllocationReleaseOutcome::failed(format!(
                    "the memory plugin refused release without mutating anything: {}",
                    outcome.failure.describe()
                ))
            }
            _unknown => {
                // An unknown state is never guessed. Treat it as residual
                // ownership so nothing is reused or double-refunded.
                self.module.allocation_closed();
                AllocationReleaseOutcome::quarantined(
                    ReleaseAccounting {
                        allocation_bytes: record.bytes as u64,
                        unmapped_bytes: 0,
                    },
                    ResidualOwnership {
                        state: onnx_runtime_memory_api::AllocationReleaseState::PartiallyUnmapped,
                        reason: QuarantineReason::AllocatorRefused,
                        retained_bytes: record.bytes as u64,
                        address: ptr.as_ptr() as usize,
                        align: record.align,
                    },
                )
            }
        }
    }
}

impl Drop for AllocatorCore {
    fn drop(&mut self) {
        // Give the plugin the chance to retire what it has queued. Without
        // this, an ordinary drop leaves the module's queued tally non-zero
        // forever and the unload gate can never reopen — a deferral that
        // never resolves is indistinguishable from a leak.
        self.retire_queued_releases_on_drop();

        // Any allocation still in the map is a host-side leak, not something to
        // silently free: the plugin may already have been told about it. Report
        // through the plugin's terminal slot so the bytes are not stranded.
        let leaked: Vec<(usize, LiveAllocation)> = match self.live.lock() {
            Ok(mut live) => live.drain().collect(),
            Err(_) => Vec::new(),
        };
        for (address, record) in leaked {
            let Some(ptr) = NonNull::new(address as *mut u8) else {
                continue;
            };
            let allocation = self.abi_allocation(ptr, record);
            self.raw_release(&allocation);
            self.module.allocation_closed();
        }

        if let Some(release) = self.vtable.release {
            // SAFETY: `ctx` came from this vtable and `release` is called
            // exactly once, here. The callback table and bridge are still
            // alive: the context is taken below, after this call returns, so
            // the plugin may touch the table from inside `release` itself.
            unsafe { release(self.vtable.ctx) };
        }
        self.module.allocator_closed();

        // Now decide the callback table's fate. The synchronous `release`
        // above is *not* the last thing that can reach it: the ABI explicitly
        // permits a plugin to report a queued release from its own worker
        // thread, which is the whole point of deferred release. So the table
        // may only be freed once no queued release can still name it.
        let Some(context) = self.context.take() else {
            return;
        };
        if context.bridge.outstanding_releases.load(Ordering::Acquire) == 0 {
            drop(context);
            return;
        }
        // A release the plugin never retired still holds a raw pointer into
        // this table. Freeing it would be a use-after-free the moment the
        // plugin reports; leaking it is bounded by exactly the condition that
        // already keeps the module mapped forever, and a leak is the safe
        // direction — the same reasoning the plugin side applies to a block it
        // cannot account for.
        LEAKED_CALLBACK_TABLES.fetch_add(1, Ordering::AcqRel);
        core::mem::forget(context);
    }
}

/// A plugin-backed [`DeviceAllocator`].
///
/// Field order is load-bearing. The capability views are declared **before**
/// `core` so they drop first; each holds its own `Arc<AllocatorCore>`, so the
/// plugin's `release` runs only after every view is gone.
#[derive(Debug)]
pub struct PluginAllocator {
    backing: Option<PluginVirtualBacking>,
    shared: Option<PluginSharedMapping>,
    core: Arc<AllocatorCore>,
}

impl PluginAllocator {
    /// The shared core, for callers that need to inspect plugin-side state.
    pub fn core(&self) -> &Arc<AllocatorCore> {
        &self.core
    }

    /// Queue a stream-ordered release instead of freeing immediately.
    ///
    /// Returns the plugin's ticket. Until the matching completion arrives
    /// through `release_completed`, the module, factory, allocator, and
    /// callback table all stay pinned and unload stays refused.
    ///
    /// # Safety
    ///
    /// `ptr` must identify one live allocation from this mechanism with
    /// exactly this `bytes` and `align`, and must not be released twice.
    pub unsafe fn enqueue_release(
        &self,
        ptr: NonNull<u8>,
        bytes: usize,
        align: usize,
    ) -> Result<u64, MemoryError> {
        let core = &self.core;
        if core.capability_flags & NXMEM_CAP_DEFERRED_RELEASE == 0 {
            return Err(MemoryError::AllocationFailed {
                tier: core.tier().name(),
                requested: bytes as u64,
                reason: String::from(
                    "this mechanism does not advertise deferred release; use the canonical \
                     release path instead",
                ),
            });
        }
        let Some(enqueue) = core.vtable.enqueue_release else {
            return Err(MemoryError::AllocationFailed {
                tier: core.tier().name(),
                requested: bytes as u64,
                reason: String::from(
                    "this mechanism advertises deferred release but provides no enqueue slot",
                ),
            });
        };
        // Lock taken and released before the ABI call.
        let Some(record) = core.take_allocation(ptr) else {
            return Err(MemoryError::AllocationFailed {
                tier: core.tier().name(),
                requested: bytes as u64,
                reason: format!(
                    "address {:p} is not a live allocation of this mechanism",
                    ptr.as_ptr()
                ),
            });
        };
        if record.bytes != bytes || record.align != align {
            core.insert_allocation(ptr, record);
            return Err(MemoryError::AllocationFailed {
                tier: core.tier().name(),
                requested: bytes as u64,
                reason: format!(
                    "deferred release named {bytes} bytes at align {align} but the allocation is \
                     {} bytes at align {}",
                    record.bytes, record.align
                ),
            });
        }

        let allocation = core.abi_allocation(ptr, record);
        let mut ticket = 0u64;
        // Count the queued release *before* the call so a completion that
        // arrives synchronously from inside `enqueue_release` cannot decrement
        // a counter that was never incremented.
        core.module.release_queued();
        core.bridge()
            .outstanding_releases
            .fetch_add(1, Ordering::AcqRel);
        // SAFETY: `ctx` came from this vtable; both pointers address valid
        // locals. No host lock is held.
        let status = unsafe { enqueue(core.vtable.ctx, &raw const allocation, &raw mut ticket) };
        if !status.is_ok() {
            core.module.release_retired();
            core.bridge()
                .outstanding_releases
                .fetch_sub(1, Ordering::AcqRel);
            core.insert_allocation(ptr, record);
            return Err(status_to_memory_error(
                "enqueue_release",
                core.tier(),
                bytes as u64,
                &status,
            ));
        }
        Ok(ticket)
    }

    /// Retire up to `max` queued releases, driving the completion callbacks.
    pub fn drain_releases(&self, max: u64) -> Result<u64, MemoryError> {
        let core = &self.core;
        let Some(drain) = core.vtable.drain_releases else {
            return Ok(0);
        };
        let mut retired = 0u64;
        // SAFETY: `ctx` came from this vtable and `retired` is a valid local.
        // No host lock is held; the plugin will call back into the host from
        // inside this call, which is exactly why no lock may be held.
        let status = unsafe { drain(core.vtable.ctx, max, &raw mut retired) };
        if !status.is_ok() {
            return Err(status_to_memory_error(
                "drain_releases",
                core.tier(),
                0,
                &status,
            ));
        }
        Ok(retired)
    }

    /// How many releases the plugin still has queued.
    pub fn pending_release_count(&self) -> Result<u64, MemoryError> {
        let core = &self.core;
        let Some(pending) = core.vtable.pending_release_count else {
            return Ok(0);
        };
        let mut count = 0u64;
        // SAFETY: `ctx` came from this vtable and `count` is a valid local.
        let status = unsafe { pending(core.vtable.ctx, &raw mut count) };
        if !status.is_ok() {
            return Err(status_to_memory_error(
                "pending_release_count",
                core.tier(),
                0,
                &status,
            ));
        }
        Ok(count)
    }
}

impl DeviceAllocator for PluginAllocator {
    fn allocate(&self, bytes: usize, align: usize) -> Result<NonNull<u8>, MemoryError> {
        let core = &self.core;
        let Some(allocate) = core.vtable.allocate else {
            return Err(MemoryError::AllocationFailed {
                tier: core.tier().name(),
                requested: bytes as u64,
                reason: String::from("this mechanism provides no allocate slot"),
            });
        };
        let request = core.alloc_request(core.next_allocation_id(), bytes, align);
        let mut result = NxmemAllocResult::zeroed();
        // SAFETY: `ctx` came from this vtable; both pointers address valid
        // locals that outlive the call. No host lock is held, so the plugin may
        // call back into the host from inside this call.
        let status = unsafe { allocate(core.vtable.ctx, &raw const request, &raw mut result) };
        if !status.is_ok() {
            return Err(status_to_memory_error(
                "allocate",
                core.tier(),
                bytes as u64,
                &status,
            ));
        }
        core.finish_allocation(&request, result, bytes, align)
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, bytes: usize, align: usize) {
        // Route through the canonical structured path so there is exactly one
        // production release entry point to reason about and to test.
        let _ = self.core.release_allocation(ptr, bytes, align);
    }

    unsafe fn deallocate_with_unmapped(&self, ptr: NonNull<u8>, bytes: usize, align: usize) -> u64 {
        match self.core.release_allocation(ptr, bytes, align) {
            AllocationReleaseOutcome::Complete { accounting }
            | AllocationReleaseOutcome::Quarantined { accounting, .. } => accounting.unmapped_bytes,
            AllocationReleaseOutcome::Failed { .. } => 0,
        }
    }

    unsafe fn release(
        &self,
        ptr: NonNull<u8>,
        bytes: usize,
        align: usize,
    ) -> AllocationReleaseOutcome {
        self.core.release_allocation(ptr, bytes, align)
    }

    fn device(&self) -> DeviceKey {
        self.core.device
    }

    fn commits_on_demand(&self) -> bool {
        // Only a mechanism that both reserves lazily *and* reports its commits
        // to the host may claim this. Virtual backing alone is not enough.
        self.core.capability_flags & NXMEM_CAP_VIRTUAL_BACKING != 0 && self.backing.is_some()
    }

    fn as_virtual_backing(&self) -> Option<&dyn VirtualBacking> {
        self.backing
            .as_ref()
            .map(|backing| backing as &dyn VirtualBacking)
    }

    fn as_shared_mapping(&self) -> Option<&dyn SharedMapping> {
        self.shared
            .as_ref()
            .map(|shared| shared as &dyn SharedMapping)
    }
}

/// The lazy-backing capability of a plugin mechanism.
#[derive(Debug)]
pub struct PluginVirtualBacking {
    core: Arc<AllocatorCore>,
}

impl PluginVirtualBacking {
    fn vtable(&self) -> &NxmemVirtualBackingVtable {
        self.core
            .backing
            .as_ref()
            .expect("a virtual-backing view is only built when the vtable is present")
    }

    fn range_request(
        &self,
        ptr: NonNull<u8>,
        allocation_bytes: usize,
        align: usize,
        offset: usize,
        bytes: usize,
    ) -> Result<NxmemRangeRequest, MemoryError> {
        let record = self.core.allocation_record(ptr).ok_or_else(|| {
            MemoryError::AllocationFailed {
                tier: self.core.tier().name(),
                requested: bytes as u64,
                reason: format!(
                    "address {:p} is not a live allocation of this mechanism, so no range of it \
                     can be committed",
                    ptr.as_ptr()
                ),
            }
        })?;
        if record.bytes != allocation_bytes || record.align != align {
            return Err(MemoryError::AllocationFailed {
                tier: self.core.tier().name(),
                requested: bytes as u64,
                reason: format!(
                    "the range names a {allocation_bytes}-byte allocation at align {align} but \
                     that address is {} bytes at align {}",
                    record.bytes, record.align
                ),
            });
        }
        Ok(NxmemRangeRequest::new(
            self.core.abi_allocation(ptr, record),
            offset as u64,
            bytes as u64,
        ))
    }
}

impl VirtualBacking for PluginVirtualBacking {
    fn allocate_committed(
        &self,
        bytes: usize,
        align: usize,
        committed_ranges: &[std::ops::Range<usize>],
    ) -> Result<NonNull<u8>, MemoryError> {
        let core = &self.core;
        let Some(allocate) = self.vtable().allocate_committed else {
            return Err(MemoryError::AllocationFailed {
                tier: core.tier().name(),
                requested: bytes as u64,
                reason: String::from("this mechanism provides no allocate_committed slot"),
            });
        };
        let ranges: Vec<NxmemByteRange> = committed_ranges
            .iter()
            .map(|range| NxmemByteRange {
                offset: range.start as u64,
                bytes: range.len() as u64,
            })
            .collect();
        let mut request = core.alloc_request(core.next_allocation_id(), bytes, align);
        request.committed_ranges = ranges.as_ptr();
        request.committed_range_count = ranges.len() as u64;

        let mut result = NxmemAllocResult::zeroed();
        // SAFETY: `ctx` came from the backing vtable; `request` borrows `ranges`
        // which outlives the call, and `result` is a valid local. No host lock
        // is held.
        let status = unsafe { allocate(self.vtable().ctx, &raw const request, &raw mut result) };
        drop(ranges);
        if !status.is_ok() {
            return Err(status_to_memory_error(
                "allocate_committed",
                core.tier(),
                bytes as u64,
                &status,
            ));
        }
        core.finish_allocation(&request, result, bytes, align)
    }

    fn commit_allocation_range(
        &self,
        ptr: NonNull<u8>,
        allocation_bytes: usize,
        align: usize,
        offset: usize,
        bytes: usize,
    ) -> Result<(), MemoryError> {
        let Some(commit) = self.vtable().commit_range else {
            return Err(MemoryError::AllocationFailed {
                tier: self.core.tier().name(),
                requested: bytes as u64,
                reason: String::from("this mechanism provides no commit_range slot"),
            });
        };
        let request = self.range_request(ptr, allocation_bytes, align, offset, bytes)?;
        // SAFETY: `ctx` came from the backing vtable and `request` is a valid
        // local that outlives the call. No host lock is held.
        let status = unsafe { commit(self.vtable().ctx, &raw const request) };
        if !status.is_ok() {
            return Err(status_to_memory_error(
                "commit_range",
                self.core.tier(),
                bytes as u64,
                &status,
            ));
        }
        Ok(())
    }

    fn mapped_bytes_for_allocation_ranges(
        &self,
        ranges: &[AllocationCommitRange],
    ) -> Result<u64, MemoryError> {
        let Some(mapped) = self.vtable().mapped_bytes_for_ranges else {
            return Err(MemoryError::AllocationFailed {
                tier: self.core.tier().name(),
                requested: 0,
                reason: String::from("this mechanism provides no mapped_bytes_for_ranges slot"),
            });
        };
        let requests = ranges
            .iter()
            .map(|range| {
                self.range_request(
                    range.ptr,
                    range.allocation_bytes,
                    range.align,
                    range.offset,
                    range.bytes,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut out = 0u64;
        // SAFETY: `requests` outlives the call and its length is passed
        // alongside it; `out` is a valid local. No host lock is held.
        let status = unsafe {
            mapped(
                self.vtable().ctx,
                requests.as_ptr(),
                requests.len() as u64,
                &raw mut out,
            )
        };
        drop(requests);
        if !status.is_ok() {
            return Err(status_to_memory_error(
                "mapped_bytes_for_ranges",
                self.core.tier(),
                0,
                &status,
            ));
        }
        Ok(out)
    }

    fn mapped_bytes_for_allocation(&self, bytes: usize, align: usize) -> Result<u64, MemoryError> {
        let Some(mapped) = self.vtable().mapped_bytes_for_allocation else {
            return Err(MemoryError::AllocationFailed {
                tier: self.core.tier().name(),
                requested: bytes as u64,
                reason: String::from("this mechanism provides no mapped_bytes_for_allocation slot"),
            });
        };
        // A sizing query carries no allocation identity yet, so id zero is
        // used deliberately: it can never match a live allocation.
        let request = self.core.alloc_request(0, bytes, align);
        let mut out = 0u64;
        // SAFETY: both pointers address valid locals; no host lock is held.
        let status = unsafe { mapped(self.vtable().ctx, &raw const request, &raw mut out) };
        if !status.is_ok() {
            return Err(status_to_memory_error(
                "mapped_bytes_for_allocation",
                self.core.tier(),
                bytes as u64,
                &status,
            ));
        }
        Ok(out)
    }

    fn decommit_allocation_range(
        &self,
        ptr: NonNull<u8>,
        allocation_bytes: usize,
        align: usize,
        offset: usize,
        bytes: usize,
    ) -> Result<u64, MemoryError> {
        let Some(decommit) = self.vtable().decommit_range else {
            return Err(MemoryError::AllocationFailed {
                tier: self.core.tier().name(),
                requested: bytes as u64,
                reason: String::from("this mechanism provides no decommit_range slot"),
            });
        };
        let request = self.range_request(ptr, allocation_bytes, align, offset, bytes)?;
        let mut unmapped = 0u64;
        // SAFETY: both pointers address valid locals; no host lock is held.
        let status = unsafe { decommit(self.vtable().ctx, &raw const request, &raw mut unmapped) };
        if !status.is_ok() {
            return Err(status_to_memory_error(
                "decommit_range",
                self.core.tier(),
                bytes as u64,
                &status,
            ));
        }
        Ok(unmapped)
    }

    fn allocation_committed_bytes(
        &self,
        ptr: NonNull<u8>,
        allocation_bytes: usize,
        align: usize,
    ) -> usize {
        let Some(committed) = self.vtable().committed_bytes else {
            return 0;
        };
        let Some(record) = self.core.allocation_record(ptr) else {
            return 0;
        };
        if record.bytes != allocation_bytes || record.align != align {
            return 0;
        }
        let allocation = self.core.abi_allocation(ptr, record);
        let mut out = 0u64;
        // SAFETY: both pointers address valid locals; no host lock is held.
        let status = unsafe { committed(self.vtable().ctx, &raw const allocation, &raw mut out) };
        if !status.is_ok() {
            return 0;
        }
        out as usize
    }
}

/// The shared-physical-mapping capability of a plugin mechanism.
#[derive(Debug)]
pub struct PluginSharedMapping {
    core: Arc<AllocatorCore>,
}

impl PluginSharedMapping {
    fn vtable(&self) -> &NxmemSharedMappingVtable {
        self.core
            .shared
            .as_ref()
            .expect("a shared-mapping view is only built when the vtable is present")
    }

    /// Recover a prefix this capability actually created.
    ///
    /// A foreign, wrong-device, or wrong-mechanism prefix is refused here
    /// rather than reported as costing zero.
    fn own_prefix<'a>(
        &self,
        prefix: &'a dyn SharedDevicePrefix,
    ) -> Result<&'a PluginSharedPrefix, MemoryError> {
        let plugin_prefix = prefix
            .as_any()
            .downcast_ref::<PluginSharedPrefix>()
            .ok_or_else(|| MemoryError::AllocationFailed {
                tier: self.core.tier().name(),
                requested: 0,
                reason: String::from(
                    "the shared prefix was not created by a memory plugin, so this mechanism \
                     cannot map it",
                ),
            })?;
        if plugin_prefix.handle.mechanism_id != self.core.mechanism_id {
            return Err(MemoryError::AllocationFailed {
                tier: self.core.tier().name(),
                requested: 0,
                reason: format!(
                    "the shared prefix belongs to mechanism {} but this mechanism is {}; a \
                     cross-mechanism mapping is refused rather than costed as free",
                    plugin_prefix.handle.mechanism_id, self.core.mechanism_id
                ),
            });
        }
        if device_key(plugin_prefix.handle.device) != Some(self.core.device) {
            return Err(MemoryError::AllocationFailed {
                tier: self.core.tier().name(),
                requested: 0,
                reason: String::from(
                    "the shared prefix lives on a different device than this mechanism serves",
                ),
            });
        }
        Ok(plugin_prefix)
    }
}

impl SharedMapping for PluginSharedMapping {
    fn create_shared_prefix(
        &self,
        bytes: usize,
    ) -> Result<Box<dyn SharedDevicePrefix>, MemoryError> {
        let Some(create) = self.vtable().create_shared_prefix else {
            return Err(MemoryError::AllocationFailed {
                tier: self.core.tier().name(),
                requested: bytes as u64,
                reason: String::from("this mechanism provides no create_shared_prefix slot"),
            });
        };
        let mut handle = NxmemSharedPrefixHandle::zeroed();
        // SAFETY: `handle` is a valid writable local; no host lock is held.
        let status = unsafe {
            create(
                self.vtable().ctx,
                self.core.mechanism_id,
                bytes as u64,
                &raw mut handle,
            )
        };
        if !status.is_ok() {
            return Err(status_to_memory_error(
                "create_shared_prefix",
                self.core.tier(),
                bytes as u64,
                &status,
            ));
        }
        if handle.handle == 0 || handle.mechanism_id != self.core.mechanism_id {
            return Err(MemoryError::AllocationFailed {
                tier: self.core.tier().name(),
                requested: bytes as u64,
                reason: String::from(
                    "the memory plugin reported success but returned a prefix with no identity",
                ),
            });
        }
        Ok(Box::new(PluginSharedPrefix {
            handle,
            core: Arc::clone(&self.core),
        }))
    }

    fn incremental_owned_bytes_for_shared_prefix(
        &self,
        prefix: &dyn SharedDevicePrefix,
    ) -> Result<u64, MemoryError> {
        let prefix = self.own_prefix(prefix)?;
        let Some(incremental) = self.vtable().incremental_owned_bytes else {
            return Err(MemoryError::AllocationFailed {
                tier: self.core.tier().name(),
                requested: 0,
                reason: String::from("this mechanism provides no incremental_owned_bytes slot"),
            });
        };
        let mut out = 0u64;
        // SAFETY: both pointers address valid locals; no host lock is held.
        let status =
            unsafe { incremental(self.vtable().ctx, &raw const prefix.handle, &raw mut out) };
        if !status.is_ok() {
            return Err(status_to_memory_error(
                "incremental_owned_bytes",
                self.core.tier(),
                0,
                &status,
            ));
        }
        Ok(out)
    }

    fn commit_shared_prefix(
        &self,
        prefix: &dyn SharedDevicePrefix,
        ptr: NonNull<u8>,
        allocation_bytes: usize,
        byte_offset: usize,
    ) -> Result<SharedPrefixCommitInfo, MemoryError> {
        let prefix = self.own_prefix(prefix)?;
        let Some(commit) = self.vtable().commit_shared_prefix else {
            return Err(MemoryError::AllocationFailed {
                tier: self.core.tier().name(),
                requested: 0,
                reason: String::from("this mechanism provides no commit_shared_prefix slot"),
            });
        };
        let record =
            self.core
                .allocation_record(ptr)
                .ok_or_else(|| MemoryError::AllocationFailed {
                    tier: self.core.tier().name(),
                    requested: 0,
                    reason: format!(
                        "address {:p} is not a live allocation of this mechanism",
                        ptr.as_ptr()
                    ),
                })?;
        if record.bytes != allocation_bytes {
            return Err(MemoryError::AllocationFailed {
                tier: self.core.tier().name(),
                requested: 0,
                reason: format!(
                    "the mapping names a {allocation_bytes}-byte allocation but that address is \
                     {} bytes",
                    record.bytes
                ),
            });
        }

        let request = NxmemSharedPrefixCommitRequest::new(
            prefix.handle,
            self.core.abi_allocation(ptr, record),
            byte_offset as u64,
        );
        let mut info = NxmemSharedPrefixCommitInfo::zeroed();
        // SAFETY: both pointers address valid locals; no host lock is held.
        let status = unsafe { commit(self.vtable().ctx, &raw const request, &raw mut info) };
        if !status.is_ok() {
            return Err(status_to_memory_error(
                "commit_shared_prefix",
                self.core.tier(),
                0,
                &status,
            ));
        }
        Ok(SharedPrefixCommitInfo {
            additional_owned_bytes: info.additional_owned_bytes,
            newly_mapped_bytes: info.newly_mapped_bytes,
            granules: info.granules as usize,
        })
    }
}

/// A plugin-owned shared physical prefix.
///
/// Holds an `Arc<AllocatorCore>` so the mechanism cannot be released while a
/// prefix it owns is still alive.
#[derive(Debug)]
pub struct PluginSharedPrefix {
    handle: NxmemSharedPrefixHandle,
    core: Arc<AllocatorCore>,
}

impl PluginSharedPrefix {
    /// The plugin-side handle, for tests that need the raw identity.
    pub fn handle(&self) -> NxmemSharedPrefixHandle {
        self.handle
    }
}

impl SharedDevicePrefix for PluginSharedPrefix {
    fn device_ptr(&self) -> u64 {
        self.handle.device_ptr
    }

    fn committed_physical_bytes(&self) -> u64 {
        self.handle.committed_physical_bytes
    }

    fn mapped_bytes(&self) -> usize {
        self.handle.mapped_bytes as usize
    }

    fn requested_bytes(&self) -> usize {
        self.handle.requested_bytes as usize
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl Drop for PluginSharedPrefix {
    fn drop(&mut self) {
        let Some(shared) = self.core.shared.as_ref() else {
            return;
        };
        let Some(release) = shared.release_shared_prefix else {
            return;
        };
        // SAFETY: `ctx` came from this vtable; the handle is a valid local and
        // `release_shared_prefix` is called exactly once, here. No host lock is
        // held and this slot may not call back into the host.
        let _ = unsafe { release(shared.ctx, &raw const self.handle) };
    }
}

/// Give back an allocator the host has decided it cannot use.
///
/// `open_allocator` returning `Ok` means the plugin created state, so the host
/// owes it a `release` even when it then refuses the vtable. The refusal path
/// is the one place the host must reach into a struct it has already judged
/// untrustworthy, so it does the least it can: re-read the prefix, and call
/// `release` only if that succeeds and the slot is present. A plugin that
/// publishes a vtable too malformed to locate `release` cannot be given
/// anything back — such a plugin must not allocate state before returning.
///
/// The re-read deliberately goes through
/// [`NxmemAllocatorVtable::read_prefix_unvalidated`] rather than
/// `read_prefix`. Every memory-safety check still runs; what is skipped is
/// `validate_required`, which is the very check most refusals failed. Reading
/// through the validating entry point here would fail a second time, for the
/// same reason, and the plugin's state would be stranded — a null `allocate`
/// slot or a zero `mechanism_id` would leak rather than be handed back.
///
/// # Safety
///
/// `ptr` must be null or the pointer the plugin just returned.
unsafe fn abandon_allocator(ptr: *const NxmemAllocatorVtable, minor: u32) {
    // SAFETY: delegated to this function's contract; the unvalidated read
    // still performs every null, alignment, and size check before reading a
    // field, and copies rather than borrowing.
    let Ok(vtable) = (unsafe { NxmemAllocatorVtable::read_prefix_unvalidated(ptr, minor) }) else {
        return;
    };
    let Some(release) = vtable.release else {
        return;
    };
    // SAFETY: `ctx` came from this vtable and no host lock is held.
    unsafe { release(vtable.ctx) };
}

/// Holds the host's debt to the plugin until the allocator is accepted.
///
/// From the moment `open_allocator` returns `Ok`, the plugin has created state
/// and the host owes it exactly one `release`. Between that point and the
/// construction of the [`AllocatorCore`] the host applies a series of
/// independent checks, any of which may refuse the vtable — and every one of
/// those refusals still owes the release.
///
/// Discharging that debt with an explicit call on each early return is what
/// the previous shape did, and it is why five of the six refusal paths leaked:
/// each new check is a new opportunity to forget. A guard inverts the default,
/// so a path that forgets to think about the debt pays it anyway, and a
/// seventh check added later is covered before it is written.
struct AbandonOnDrop {
    vtable: *const NxmemAllocatorVtable,
    minor: u32,
    armed: bool,
}

impl AbandonOnDrop {
    /// Take on the debt for a vtable the plugin has just handed over.
    ///
    /// # Safety
    ///
    /// `vtable` must be null or the pointer the plugin returned from a
    /// successful `open_allocator`, and must stay valid until the guard is
    /// dropped or disarmed.
    unsafe fn new(vtable: *const NxmemAllocatorVtable, minor: u32) -> Self {
        Self {
            vtable,
            minor,
            armed: true,
        }
    }

    /// The host accepted the allocator: the debt now belongs to
    /// [`AllocatorCore::drop`], which makes the matching `release` call.
    fn accepted(mut self) {
        self.armed = false;
    }
}

impl Drop for AbandonOnDrop {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        // SAFETY: the pointer is the one the plugin returned, as promised by
        // `AbandonOnDrop::new`; `abandon_allocator` re-reads it defensively
        // and calls nothing but `release`. No host lock is held: this runs on
        // an early return from `open_allocator`, which takes none.
        unsafe { abandon_allocator(self.vtable, self.minor) };
    }
}

/// Open an allocator from a factory, wiring up the host callback table.
///
/// # The post-`Ok` obligation
///
/// Once the plugin returns `Ok` it has created state, and the host owes it a
/// `release` — even if the host then refuses the vtable on its own terms. The
/// `Ok` is what creates the debt; the host's later opinion of the vtable does
/// not cancel it. A refusal that just returns `Err` strands plugin state and,
/// because the module's live-allocator tally gates unload, permanently
/// disables the gate for the life of the process.
///
/// That obligation is discharged by [`AbandonOnDrop`] rather than by a call on
/// each rejection path, so a path added later is covered before it is written.
pub(crate) fn open_allocator(
    factory: &PluginFactory,
    required_capability_flags: u64,
    reclaim: Option<Arc<dyn HostReclaim>>,
) -> Result<PluginAllocator, PluginError> {
    let module = Arc::clone(factory.module());
    let minor = module.negotiated().minor;
    let factory_vtable = factory.vtable();
    let Some(open) = factory_vtable.open_allocator else {
        return Err(PluginError::Contract {
            path: module.path().display().to_string(),
            reason: format!(
                "factory `{}` provides no open_allocator slot",
                factory.name()
            ),
        });
    };

    // The bridge and the callback table must live at stable heap addresses
    // *before* the plugin sees them, and must outlive the allocator's final
    // release **and every queued release**. `HostCallbackContext` owns both in
    // one boxed unit whose teardown `AllocatorCore::drop` controls.
    let context = HostCallbackContext::new(
        minor,
        HostBridge {
            device: factory.device(),
            reclaim,
            reclaim_calls: AtomicU64::new(0),
            reclaim_failures: AtomicU64::new(0),
            retired: Mutex::new(Vec::new()),
            module: Arc::clone(&module),
            outstanding_releases: AtomicU64::new(0),
        },
    );

    let request = NxmemOpenRequest::new(
        minor,
        factory_vtable.device,
        required_capability_flags,
        &raw const context.callbacks,
    );
    let mut raw_allocator = core::ptr::null::<NxmemAllocatorVtable>();
    // SAFETY: `ctx` came from the factory vtable; `request` and
    // `raw_allocator` are valid locals that outlive the call. No host lock is
    // held.
    let status = unsafe {
        open(
            factory_vtable.ctx,
            &raw const request,
            &raw mut raw_allocator,
        )
    };
    if !status.is_ok() {
        return Err(PluginError::call("open_allocator", status));
    }

    // From here on the plugin has created state and the host owes it exactly
    // one `release`, whatever it decides about the vtable. The guard carries
    // that debt so no rejection path below has to remember it.
    // SAFETY: `raw_allocator` is whatever the plugin wrote in response to the
    // successful call above, which is precisely what the guard promises to
    // hand back.
    let abandon = unsafe { AbandonOnDrop::new(raw_allocator, minor) };

    // SAFETY: the plugin wrote this pointer in response to the call above.
    // `read_prefix` null-, alignment- and size-checks it before reading any
    // field, and copies rather than borrowing.
    let vtable = unsafe { NxmemAllocatorVtable::read_prefix(raw_allocator, minor) }
        .map_err(|status| PluginError::call("allocator vtable", status))?;
    let device = device_key(vtable.device).ok_or_else(|| PluginError::Contract {
        path: module.path().display().to_string(),
        reason: format!(
            "allocator `{}` declared tier code {} which this host does not know",
            factory.name(),
            vtable.device.tier
        ),
    })?;
    if device != factory.device() {
        return Err(PluginError::Contract {
            path: module.path().display().to_string(),
            reason: format!(
                "factory `{}` serves {:?} but opened an allocator for {device:?}; a mechanism \
                 must stay on one device",
                factory.name(),
                factory.device()
            ),
        });
    }
    if vtable.mechanism_id == 0 {
        return Err(PluginError::Contract {
            path: module.path().display().to_string(),
            reason: format!(
                "allocator `{}` published mechanism id zero, which can never be matched against \
                 an allocation",
                factory.name()
            ),
        });
    }
    let missing = required_capability_flags & !vtable.capability_flags;
    if missing != 0 {
        return Err(PluginError::Contract {
            path: module.path().display().to_string(),
            reason: format!(
                "allocator `{}` was opened requiring capabilities {missing:#x} that it does not \
                 provide",
                factory.name()
            ),
        });
    }

    // Capability vtables are read at the allocator's own level, and
    // `read_prefix` itself refuses a vtable that names a different mechanism.
    let backing = read_capability(
        vtable.virtual_backing,
        vtable.capability_flags & NXMEM_CAP_VIRTUAL_BACKING != 0,
        "virtual backing",
        &module,
        || {
            // SAFETY: the pointer came from the allocator vtable the plugin
            // just published; `read_prefix` null-, alignment- and size-checks
            // it before reading any field.
            unsafe {
                NxmemVirtualBackingVtable::read_prefix(
                    vtable.virtual_backing,
                    minor,
                    vtable.mechanism_id,
                )
            }
        },
    )?;
    let shared = read_capability(
        vtable.shared_mapping,
        vtable.capability_flags & NXMEM_CAP_SHARED_MAPPING != 0,
        "shared mapping",
        &module,
        || {
            // SAFETY: as above.
            unsafe {
                NxmemSharedMappingVtable::read_prefix(
                    vtable.shared_mapping,
                    minor,
                    vtable.mechanism_id,
                )
            }
        },
    )?;

    // SAFETY: the contract requires `name` to stay valid until the final
    // `release`, which has not been called.
    let name = unsafe { read_optional_c_string(vtable.name) }
        .unwrap_or_else(|| factory.name().to_string());

    // Every check has passed, so the host is taking the allocator. The debt
    // for the plugin's `release` transfers from the guard to `AllocatorCore`,
    // whose `Drop` makes the matching call.
    abandon.accepted();

    module.allocator_opened();
    let core = Arc::new(AllocatorCore {
        vtable,
        backing,
        shared,
        device,
        mechanism_id: vtable.mechanism_id,
        name,
        negotiated_minor: minor,
        capability_flags: vtable.capability_flags,
        next_allocation_id: AtomicU64::new(0),
        live: Mutex::new(HashMap::new()),
        context: Some(context),
        module,
    });

    let backing_view = core.backing.is_some().then(|| PluginVirtualBacking {
        core: Arc::clone(&core),
    });
    let shared_view = core.shared.is_some().then(|| PluginSharedMapping {
        core: Arc::clone(&core),
    });

    Ok(PluginAllocator {
        backing: backing_view,
        shared: shared_view,
        core,
    })
}

fn read_capability<T>(
    ptr: *const T,
    advertised: bool,
    label: &str,
    module: &Arc<PluginModule>,
    read: impl FnOnce() -> Result<T, NxmemStatus>,
) -> Result<Option<T>, PluginError> {
    if !advertised {
        // An unsupported capability is represented explicitly: the flag is
        // clear *and* the pointer is null. A non-null pointer behind a clear
        // flag is a contract violation, not a bonus feature, because the host
        // would have no negotiated basis for calling it.
        if !ptr.is_null() {
            return Err(PluginError::Contract {
                path: module.path().display().to_string(),
                reason: format!(
                    "the allocator published a {label} vtable without advertising the \
                     capability; an unsupported capability must be a null pointer and a clear flag"
                ),
            });
        }
        return Ok(None);
    }
    if ptr.is_null() {
        return Err(PluginError::Contract {
            path: module.path().display().to_string(),
            reason: format!("the allocator advertised {label} but published a null vtable for it"),
        });
    }
    let vtable = read().map_err(|status| PluginError::call("capability vtable", status))?;
    Ok(Some(vtable))
}

/// # Safety
///
/// `ptr` must be null or point to a NUL-terminated byte string valid for the
/// duration of the call.
unsafe fn read_optional_c_string(ptr: *const u8) -> Option<String> {
    if ptr.is_null() {
        return None;
    }
    // SAFETY: delegated to this function's contract.
    unsafe { core::ffi::CStr::from_ptr(ptr.cast()) }
        .to_str()
        .ok()
        .map(str::to_owned)
}
