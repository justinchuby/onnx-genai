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

use std::alloc::{Layout, alloc, dealloc};
use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::{Arc, Mutex};

use onnx_runtime_memory_governor::{HolderId, MemoryGovernor, MemoryRole, Tier};

use crate::allocator::MemoryInfo;

/// Alignment ORT's own CPU allocator guarantees.
///
/// ORT does not tell us the alignment it needs, and its internal allocators
/// align to at least this, so kernels are entitled to assume it. Under-aligning
/// would fault only on the vector paths that require it, which is the worst
/// possible way to find out.
const ALLOCATION_ALIGNMENT: usize = 64;

/// What one live allocation costs, so `Free` can return exactly that much.
struct LiveBlock {
    layout: Layout,
    /// Dropping this returns the bytes to the governor.
    _lease: onnx_runtime_memory_governor::MemoryLease,
}

struct GovernedAllocatorState {
    governor: Arc<dyn MemoryGovernor + Send + Sync>,
    tier: Tier,
    role: MemoryRole,
    holder: HolderId,
    /// ORT's `Free` hands back only a pointer, so the size has to be remembered.
    live: Mutex<HashMap<usize, LiveBlock>>,
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
    pub fn new(
        memory_info: MemoryInfo,
        governor: Arc<dyn MemoryGovernor + Send + Sync>,
        tier: Tier,
        role: MemoryRole,
        holder: HolderId,
    ) -> Box<Self> {
        let mut allocator = Box::new(Self {
            base: onnx_genai_ort_sys::OrtAllocator {
                version: onnx_genai_ort_sys::ORT_API_VERSION,
                Alloc: Some(governed_alloc),
                Free: Some(governed_free),
                Info: Some(governed_info),
                Reserve: Some(governed_alloc),
                GetStats: None,
                AllocOnStream: None,
                // Optional in the C contract, and only meaningful for an arena
                // that can hand capacity back. This allocator releases on Free,
                // so there is nothing to shrink.
                Shrink: None,
            },
            memory_info,
            state: Arc::new(GovernedAllocatorState {
                governor,
                tier,
                role,
                holder,
                live: Mutex::new(HashMap::new()),
            }),
        });
        // The vtable is only reachable through this pointer, so it must not move
        // afterwards; returning a Box makes that the caller's problem to keep.
        let _ = &mut *allocator;
        allocator
    }

    /// The pointer to hand ORT's registration API.
    pub fn as_ort_allocator(&mut self) -> *mut onnx_genai_ort_sys::OrtAllocator {
        std::ptr::from_mut(&mut self.base)
    }

    /// Bytes currently held by live allocations.
    pub fn live_bytes(&self) -> u64 {
        self.state
            .live
            .lock()
            .map(|live| live.values().map(|b| b.layout.size() as u64).sum())
            .unwrap_or(0)
    }

    /// Number of live allocations.
    pub fn live_count(&self) -> usize {
        self.state.live.lock().map(|live| live.len()).unwrap_or(0)
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

unsafe extern "C" fn governed_alloc(
    this: *mut onnx_genai_ort_sys::OrtAllocator,
    size: usize,
) -> *mut c_void {
    // SAFETY: ORT only calls this with a pointer we registered.
    let allocator = unsafe { allocator_from_base(this) };
    let state = &allocator.state;

    if size == 0 {
        // ORT documents nullptr for a zero-size request; allocating a
        // zero-length layout would also be undefined.
        return std::ptr::null_mut();
    }
    let Ok(layout) = Layout::from_size_align(size, ALLOCATION_ALIGNMENT) else {
        return std::ptr::null_mut();
    };

    // Lease first. A refusal must not allocate, or the budget is decorative.
    let Ok(lease) = state
        .governor
        .reserve(state.tier, size as u64, state.role, state.holder)
    else {
        return std::ptr::null_mut();
    };

    // SAFETY: `layout` has a non-zero size and a valid power-of-two alignment.
    let ptr = unsafe { alloc(layout) };
    if ptr.is_null() {
        // Dropping the lease here returns the bytes; failing to would leak
        // budget on every allocation failure.
        return std::ptr::null_mut();
    }

    match state.live.lock() {
        Ok(mut live) => {
            live.insert(
                ptr as usize,
                LiveBlock {
                    layout,
                    _lease: lease,
                },
            );
            ptr.cast::<c_void>()
        }
        Err(_) => {
            // The map is poisoned, so this block could never be freed through
            // `Free` and would leak both memory and budget. Give both back and
            // report failure instead.
            // SAFETY: `ptr` came from `alloc` with this exact `layout`.
            unsafe { dealloc(ptr, layout) };
            std::ptr::null_mut()
        }
    }
}

unsafe extern "C" fn governed_free(this: *mut onnx_genai_ort_sys::OrtAllocator, p: *mut c_void) {
    if p.is_null() {
        return;
    }
    // SAFETY: ORT only calls this with a pointer we registered.
    let allocator = unsafe { allocator_from_base(this) };

    let Ok(mut live) = allocator.state.live.lock() else {
        return;
    };
    let Some(block) = live.remove(&(p as usize)) else {
        // Freeing something we did not allocate would corrupt the heap, so it
        // is ignored rather than passed to `dealloc`. This cannot happen if ORT
        // honours the allocator contract.
        return;
    };
    drop(live);
    // SAFETY: the pointer and layout are the pair recorded by `governed_alloc`.
    unsafe { dealloc(p.cast::<u8>(), block.layout) };
    // `block` drops here, returning the lease.
}

unsafe extern "C" fn governed_info(
    this: *const onnx_genai_ort_sys::OrtAllocator,
) -> *const onnx_genai_ort_sys::OrtMemoryInfo {
    // SAFETY: ORT only calls this with a pointer we registered.
    let allocator = unsafe { allocator_from_base(this) };
    allocator.memory_info.as_ptr()
}

#[cfg(test)]
mod tests {
    use super::*;
    use onnx_runtime_memory_governor::{LeaseLedger, LedgerGovernor};

    const HOLDER: HolderId = HolderId::new(9);

    fn allocator(budget: u64) -> (Box<GovernedAllocator>, LedgerGovernor) {
        let governor = LedgerGovernor::new(LeaseLedger::new(0, budget, 0));
        let allocator = GovernedAllocator::new(
            MemoryInfo::cpu().expect("cpu memory info"),
            Arc::new(governor.clone()),
            Tier::Host,
            MemoryRole::Activation,
            HOLDER,
        );
        (allocator, governor)
    }

    /// Every byte ORT takes is leased before it is handed over.
    #[test]
    fn an_allocation_is_leased_before_the_memory_is_returned() {
        let (mut alloc, governor) = allocator(4096);
        let ptr = unsafe { governed_alloc(alloc.as_ort_allocator(), 1024) };
        assert!(!ptr.is_null(), "a request within budget must succeed");
        assert_eq!(governor.available(Tier::Host), 4096 - 1024);
        assert_eq!(alloc.live_bytes(), 1024);

        unsafe { governed_free(alloc.as_ort_allocator(), ptr) };
        assert_eq!(
            governor.available(Tier::Host),
            4096,
            "freeing did not return the lease"
        );
        assert_eq!(alloc.live_count(), 0);
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
        let (mut alloc, governor) = allocator(4096);
        let mut pointers = Vec::new();
        for _ in 0..4 {
            let ptr = unsafe { governed_alloc(alloc.as_ort_allocator(), 1024) };
            assert!(!ptr.is_null());
            pointers.push(ptr);
        }
        assert_eq!(governor.available(Tier::Host), 0);

        let refused = unsafe { governed_alloc(alloc.as_ort_allocator(), 1) };
        assert!(refused.is_null(), "a full budget must refuse even one byte");

        for ptr in pointers {
            unsafe { governed_free(alloc.as_ort_allocator(), ptr) };
        }
        assert_eq!(governor.available(Tier::Host), 4096);
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
}
