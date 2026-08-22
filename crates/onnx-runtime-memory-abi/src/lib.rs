//! # `onnx-runtime-memory-abi`
//!
//! The **nxmem** ABI: a versioned C interface that lets a dynamic library
//! supply a memory mechanism (ordinary allocator, optional lazy virtual
//! backing, optional shared physical mapping) to this runtime.
//!
//! This crate is deliberately dependency-free. A plugin author builds against
//! it, or against [`include/nxmem_memory_abi.h`][header] with no Rust at all.
//! The host adapter that wraps this ABI in the internal Rust traits lives in
//! `onnx-runtime-memory-host`; the internal traits themselves live in
//! `onnx-runtime-memory-api` and never cross this boundary.
//!
//! [header]: https://github.com/justinchuby/onnx-genai/blob/main/crates/onnx-runtime-memory-abi/include/nxmem_memory_abi.h
//!
//! # What may cross the boundary
//!
//! Only `#[repr(C)]` plain data, raw pointers, and `extern "C"` function
//! pointers. Specifically **not**: Rust trait objects, `Arc`, `Box`, `String`,
//! `Vec`, Rust enum layouts, Rust `Result`, or any object one module allocates
//! and the other frees. Enum-shaped values travel as raw `u32` wire codes with
//! checked accessors, because transmuting an unknown discriminant is undefined
//! behaviour. Panics are caught with `catch_unwind` before they can unwind
//! across the boundary; see [`catch_status_panic`] and [`catch_void_panic`].
//!
//! # Exported symbols
//!
//! A conforming plugin `cdylib` exports exactly three symbols:
//!
//! | Symbol | Constant | Purpose |
//! |---|---|---|
//! | `NxmemNegotiate` | [`NXMEM_SYMBOL_NEGOTIATE`] | version and capability handshake |
//! | `NxmemCreateAllocatorFactories` | [`NXMEM_SYMBOL_CREATE_ALLOCATOR_FACTORIES`] | enumerate mechanisms |
//! | `NxmemQueryUnloadReadiness` | [`NXMEM_SYMBOL_QUERY_UNLOAD_READINESS`] | report live objects so unload can be gated |
//!
//! All three are required. A library missing any of them is refused at load:
//! without the third the host cannot tell whether unloading would strand live
//! plugin state.
//!
//! # Lifetime and ownership rules
//!
//! | Object | Created by | Destroyed by | Rule |
//! |---|---|---|---|
//! | factory vtable | `NxmemCreateAllocatorFactories` | `factory.release(ctx)` | host calls it exactly once; releasing a factory must not invalidate allocators already opened from it |
//! | allocator vtable | `factory.open_allocator` | `allocator.release(ctx)` | reference counted via `retain`/`release`; the plugin destroys it only when the count reaches zero **and** no queued release still names it |
//! | virtual-backing vtable | exposed by an allocator | — | borrowed from the allocator; has no independent count and dies with it |
//! | shared-mapping vtable | exposed by an allocator | — | same as virtual backing |
//! | shared prefix | `create_shared_prefix` | `release_shared_prefix` | reference counted via `retain_shared_prefix`; physical bytes survive until the last reference *and* the last mapping retire |
//! | allocation | `allocate` / `allocate_committed` | `deallocate` / `release_allocation` / `enqueue_release` | released exactly once, by the mechanism that created it, keyed by `allocation_id` |
//! | host callback table | host | host | borrowed by the plugin for the whole allocator lifetime; the host keeps it alive past the final `release` and past every queued release |
//! | status / report / outcome | either | — | pure value types with inline storage; nothing to free |
//!
//! No object is ever freed by the module that did not allocate it, so the two
//! modules may be linked against different C runtimes.
//!
//! # Threading, reentrancy, and the no-blocking rule
//!
//! * Every slot may be called concurrently from multiple host threads unless
//!   its documentation says otherwise. A plugin must do its own
//!   synchronisation.
//! * **The host never holds a governance lock across a call into a plugin.**
//!   Allocation accounting, charge settlement, and registration locks are all
//!   released before the vtable is entered. This is the same rule the internal
//!   Rust interfaces already obey for trait objects, and it matters more here:
//!   a plugin may block, call back into the host, or spawn threads, and any of
//!   those under a governance lock would deadlock the process.
//! * **A plugin must not block while a host callback is executing.** Host
//!   callbacks are permitted to acquire host-side locks; a plugin that blocks
//!   waiting for another plugin thread from inside a callback can deadlock.
//! * Host callbacks ([`NxmemHostCallbacks`]) may be invoked reentrantly from
//!   inside a call the host is currently making, and from plugin-owned worker
//!   threads. They must not block indefinitely and must not re-enter the same
//!   allocator instance.
//! * A host callback returning a failure status is normal and expected. The
//!   plugin must handle it (typically by failing the operation with
//!   [`NxmemStatusCode::CallbackFailed`] or `OutOfMemory`) and must not abort,
//!   leak, or leave partial state.
//! * `retain`, `release`, and `release_shared_prefix` must not call back into
//!   the host at all: they run on teardown paths where the host may already be
//!   dismantling its own state.
//!
//! # Deferred release and unload gating
//!
//! `enqueue_release` hands an allocation to a plugin-owned queue instead of
//! freeing it immediately, which is what GPU stream ordering requires. Until
//! the corresponding completion is reported through
//! [`NxmemHostCallbacks::release_completed`]:
//!
//! * the host keeps the plugin module, the factory, the allocator, and the
//!   callback table pinned, and
//! * `NxmemQueryUnloadReadiness` must count the entry in `queued_releases`.
//!
//! The host refuses or defers unload while any of the counters in
//! [`NxmemUnloadReport`] is non-zero, and independently while it holds any
//! live handle of its own. Both gates exist so a bug on one side cannot
//! silently unmap code that is about to run.
//!
//! # Non-goals
//!
//! Internal policy, holder, victim-selection, and governor types are not
//! exposed here and never will be. Nothing that predates this contract is
//! promised ABI compatibility.

// `NxmemStatus` carries its message inline, in a fixed 256-byte buffer, so that
// no heap allocation is ever owned across the dynamic-library boundary. That
// makes every `Result<_, NxmemStatus>` "large" by clippy's measure. Boxing the
// error, which is what the lint asks for, would reintroduce exactly the
// cross-allocator ownership this ABI exists to forbid.
#![allow(clippy::result_large_err)]
#![deny(clippy::mem_forget)]

pub mod status;
pub mod types;
pub mod version;
pub mod vtable;

pub use status::{
    NXMEM_STATUS_MESSAGE_BUF, NXMEM_STATUS_MESSAGE_MAX, NxmemStatus, NxmemStatusCode,
    catch_status_panic, catch_void_panic,
};
pub use types::{
    NXMEM_RELEASE_COMPLETE, NXMEM_RELEASE_FAILED, NXMEM_RELEASE_QUARANTINED, NXMEM_TIER_DEVICE,
    NXMEM_TIER_DISK, NXMEM_TIER_HOST, NxmemAllocRequest, NxmemAllocResult, NxmemAllocation,
    NxmemByteRange, NxmemDeviceId, NxmemHostCallbacks, NxmemRangeRequest, NxmemReclaimRequest,
    NxmemReleaseCompletion, NxmemReleaseOutcome, NxmemSharedPrefixCommitInfo,
    NxmemSharedPrefixCommitRequest, NxmemSharedPrefixHandle, NxmemUnloadReport, check_identity,
};
pub use version::{
    NXMEM_ABI_VERSION_MAJOR, NXMEM_ABI_VERSION_MINOR, NXMEM_ABI_VERSION_MINOR_BASELINE,
    NXMEM_CAP_ALLOCATOR, NXMEM_CAP_DEFERRED_RELEASE, NXMEM_CAP_KNOWN_MASK,
    NXMEM_CAP_SHARED_MAPPING, NXMEM_CAP_STRUCTURED_RELEASE, NXMEM_CAP_VIRTUAL_BACKING,
    NegotiationRejection, NxmemNegotiateRequest, NxmemNegotiateResponse, NxmemVersionRange,
    capability_min_minor, describe_capabilities, negotiate, negotiate_as, validate_negotiation,
};
pub use vtable::{
    NxmemAllocatorFactoryVtable, NxmemAllocatorVtable, NxmemOpenRequest, NxmemSharedMappingVtable,
    NxmemVirtualBackingVtable,
};

/// Symbol name of the version handshake entry point.
pub const NXMEM_SYMBOL_NEGOTIATE: &[u8] = b"NxmemNegotiate";

/// Symbol name of the factory enumeration entry point.
pub const NXMEM_SYMBOL_CREATE_ALLOCATOR_FACTORIES: &[u8] = b"NxmemCreateAllocatorFactories";

/// Symbol name of the unload-readiness entry point.
pub const NXMEM_SYMBOL_QUERY_UNLOAD_READINESS: &[u8] = b"NxmemQueryUnloadReadiness";

/// Signature of `NxmemNegotiate`.
///
/// # Ownership
///
/// `request` is borrowed for the call. `response_out` is a host-owned buffer
/// the plugin fills. Neither pointer may be retained.
pub type NxmemNegotiateFn = unsafe extern "C" fn(
    request: *const NxmemNegotiateRequest,
    response_out: *mut NxmemNegotiateResponse,
) -> NxmemStatus;

/// Signature of `NxmemCreateAllocatorFactories`.
///
/// # Ownership
///
/// Each pointer written to `out_factories` is **owned by the host**, which
/// must call the factory's `release` slot exactly once. Writing fewer than
/// `max_factories` entries is normal; writing more is a contract violation and
/// the host clamps defensively.
pub type NxmemCreateAllocatorFactoriesFn = unsafe extern "C" fn(
    out_factories: *mut *const NxmemAllocatorFactoryVtable,
    max_factories: u64,
    out_count: *mut u64,
) -> NxmemStatus;

/// Signature of `NxmemQueryUnloadReadiness`.
///
/// # Contract
///
/// Reports how many objects the plugin still owns. The host refuses or defers
/// unload while [`NxmemUnloadReport::total`] is non-zero. A plugin must count
/// conservatively: reporting zero while anything is live invites the host to
/// unmap code that is still reachable.
pub type NxmemQueryUnloadReadinessFn =
    unsafe extern "C" fn(report_out: *mut NxmemUnloadReport) -> NxmemStatus;

/// Emit the three required `nxmem` entry points for a `cdylib`.
///
/// # Usage
///
/// ```rust,ignore
/// onnx_runtime_memory_abi::export_nxmem_plugin! {
///     negotiate: |request, response| unsafe {
///         onnx_runtime_memory_abi::negotiate(request, response)
///     },
///     factories: my_plugin::create_factories,
///     unload_readiness: my_plugin::unload_report,
/// }
/// ```
///
/// Every generated symbol is wrapped in `catch_unwind`, so a panic anywhere in
/// the plugin becomes [`NxmemStatusCode::InternalError`] instead of undefined
/// behaviour.
#[macro_export]
macro_rules! export_nxmem_plugin {
    (
        negotiate: $negotiate:expr,
        factories: $factories:expr,
        unload_readiness: $unload:expr $(,)?
    ) => {
        /// nxmem version handshake.
        ///
        /// # Safety
        ///
        /// Both pointers must be valid, aligned, and non-null.
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn NxmemNegotiate(
            request: *const $crate::NxmemNegotiateRequest,
            response_out: *mut $crate::NxmemNegotiateResponse,
        ) -> $crate::NxmemStatus {
            let call = $negotiate;
            $crate::catch_status_panic(|| call(request, response_out))
        }

        /// nxmem factory enumeration.
        ///
        /// # Safety
        ///
        /// `out_factories` must point to `max_factories` writable slots and
        /// `out_count` must be a valid, non-null pointer.
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn NxmemCreateAllocatorFactories(
            out_factories: *mut *const $crate::NxmemAllocatorFactoryVtable,
            max_factories: u64,
            out_count: *mut u64,
        ) -> $crate::NxmemStatus {
            let call = $factories;
            let status =
                $crate::catch_status_panic(|| call(out_factories, max_factories, out_count));
            if !status.is_ok() && !out_count.is_null() {
                // A failed enumeration must never leave a stale count behind.
                // SAFETY: `out_count` was checked non-null and the caller
                // guarantees it is writable.
                unsafe { *out_count = 0 };
            }
            status
        }

        /// nxmem unload-readiness report.
        ///
        /// # Safety
        ///
        /// `report_out` must be a valid, non-null, writable pointer.
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn NxmemQueryUnloadReadiness(
            report_out: *mut $crate::NxmemUnloadReport,
        ) -> $crate::NxmemStatus {
            let call = $unload;
            $crate::catch_status_panic(|| call(report_out))
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_three_required_symbol_names_are_distinct_and_non_empty() {
        let symbols = [
            NXMEM_SYMBOL_NEGOTIATE,
            NXMEM_SYMBOL_CREATE_ALLOCATOR_FACTORIES,
            NXMEM_SYMBOL_QUERY_UNLOAD_READINESS,
        ];
        for symbol in symbols {
            assert!(!symbol.is_empty());
            assert!(std::str::from_utf8(symbol).is_ok());
        }
        assert_ne!(symbols[0], symbols[1]);
        assert_ne!(symbols[1], symbols[2]);
        assert_ne!(symbols[0], symbols[2]);
    }
}
