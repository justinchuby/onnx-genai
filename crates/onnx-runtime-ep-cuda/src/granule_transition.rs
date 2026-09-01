//! #1810 Slice 4 — Content-preserving, stable-VA granule backing transition.
//!
//! Implements a production primitive for transitioning a contiguous range of
//! [`CudaReservation`] granules between [`PhysicalLocation::HostNuma`] and
//! [`PhysicalLocation::Device`] (and back) while:
//!
//! 1. **Preserving bytes exactly**: the old physical bytes are copied into the
//!    new backing before the stable VA is remapped.
//! 2. **Never changing the stable VA**: the pointer returned to any downstream
//!    caller remains valid throughout the transition; the VA is remapped
//!    at the identical offsets.
//! 3. **Enforcing a safe-point precondition** via the existing
//!    [`ResizeSafePoint`] / [`CudaRuntime::is_capturing`] /
//!    `CudaDeferredReleaseQueue::pending` machinery — no caller-owned implicit
//!    sync.
//! 4. **Draining compute readers** with [`CudaRuntime::drain_for_unmap`] before
//!    touching any physical mapping, proving no in-flight kernel can still be
//!    reading the old VA before the backing switches.
//! 5. **Transactional rollback** at every phase: any failure before "stable-VA
//!    map-new succeeds" restores the original state exactly (or reports Fatal).
//!
//! # Content preservation mechanism
//!
//! For each granule range to be transitioned:
//!
//! 1. Acquire new physical handles H_new[0..n] from `new_pool`.
//! 2. Map each H_new[i] at a **staging VMM window** address, set device access.
//! 3. Copy old stable-VA bytes into the staging VMM window via `dtod_async`.
//! 4. `drain_for_unmap` — syncs the copy AND proves all compute readers of the
//!    old stable VA have retired.
//! 5. Re-check safe point immediately before the stable-VA unmap.
//! 6. Per-granule atomic switch:
//!    (a) Unmap old H_old[i] from stable VA.
//!    (b) Map H_new[i] at stable VA (double-mapping: H_new is at BOTH staging and stable VA).
//!    (c) Set device access at stable VA.
//!    (d) Unmap H_new[i] from staging VA (double-mapping ends; content preserved in H_new).
//!    (e) Return H_old[i] to old_pool.
//!
//! The double-mapping (step b+d) is valid: CUDA VMM allows one handle to be
//! mapped at multiple virtual addresses simultaneously. The window between step
//! (a) (old unmapped) and step (b) (new mapped) is where the stable VA is
//! transiently unmapped — this is the atomic-ish switch. No compute kernel can
//! observe this window because `drain_for_unmap` already retired all readers.
//!
//! # Stream ordering guarantee
//!
//! `drain_for_unmap` calls `force_synchronize()` → `cuStreamSynchronize` on
//! the EP's single compute stream. Because the EP uses a **single, in-order
//! compute stream** for all kernel dispatch, synchronizing it proves:
//!   (a) The Phase 3 copy (stable VA → staging) has completed; staging has bytes.
//!   (b) Every kernel that could have been reading the old stable VA has retired.
//!
//! Any work submitted on a stream outside this process's CUDA context is by
//! definition unreachable (VMM mappings are per-context). Foreign concurrent
//! streams within the same context are impossible without access to the
//! `CudaRuntime`'s compute stream — and the safe-point contract (requiring
//! `routed_guards_active == 0`) prevents any dispatch from starting between the
//! drain and the unmap.
//!
//! **Remaining TOCTOU gap**: between the safe-point re-check (step 5) and the
//! actual `cuMemUnmap` (step 6a), a concurrent dispatch could theoretically
//! start IF the safe-point contract were violated. Callers that obey the contract
//! (via `VerifiedSafePoint`) have no such gap. See module-level doc for details.
//!
//! # Accounting equation
//!
//! ```text
//! BEFORE: old_pool owns n granules mapped at stable VA
//! DURING (step b): H_new[i] mapped at BOTH staging and stable VA.
//!   ONE physical allocation, two VAs — no double governor charge.
//! AFTER:  new_pool owns n granules mapped at stable VA
//!         old_pool returned n granules (−n×gran from old governor)
//! Net: old_pool −n×gran; new_pool +n×gran if cold-created (0 if pool hit)
//! ```
//!
//! # Lock order
//!
//! No new locks. All device-synchronizing driver calls are wrapped in
//! `synchronizing_section()`. No pool state lock is held across driver calls
//! or `drain_for_unmap`.

use std::sync::Arc;

use cudarc::driver::sys as cu;
use cudarc::driver::sys::CUdeviceptr;
use onnx_runtime_cuda_memory::capture_gate;
use onnx_runtime_cuda_memory::release::{DriverFault, MappedBlock};
use onnx_runtime_cuda_memory::virtual_memory::{
    CudaReservation, CudaVirtualBacking, PhysicalHandlePool, PhysicalLocation,
};
use onnx_runtime_ep_api::ResizeSafePoint;
use onnx_runtime_virtual_memory::VirtualBacking;

use crate::runtime::CudaRuntime;

// ---------------------------------------------------------------------------
// Public outcome types
// ---------------------------------------------------------------------------

/// Outcome of a [`transition_granule_range`] call.
#[derive(Debug)]
pub enum TransitionOutcome {
    /// Every granule transitioned; bytes bit-identical; stable VA unchanged.
    Committed {
        granules: usize,
        new_owned_bytes: u64,
        old_released_bytes: u64,
    },
    /// Refused because safe-point became invalid before the commit.
    /// Stable VA unchanged and all original mappings intact.
    Rejected { reason: &'static str },
    /// Failure occurred with **zero** side effects on the stable VA: the old
    /// mapping, its content, and its accounting are all exactly as they were
    /// before this call. This is the *only* condition under which
    /// `RolledBack` is returned — a partial commit or an ambiguous restore
    /// is always `Fatal`, never `RolledBack`.
    RolledBack { fault: DriverFault },
    /// A driver failure left this range in a state that cannot be reported as
    /// fully rolled back: some granules may have committed to the new
    /// backing, and/or a handle's mapping state could not be proven either
    /// mapped or unmapped.
    ///
    /// This is not "the whole range is unusable" — the committed prefix and
    /// the untouched suffix are reported exactly, so a caller can decide what
    /// remains safely usable. Only [`Fatal::poisoned_range`] identifies bytes
    /// that must never be read/written again.
    Fatal {
        /// The failure that interrupted the transition.
        transition_fault: DriverFault,
        /// The failure encountered while attempting to restore the old
        /// mapping, if a restore was attempted. `None` when no restore was
        /// attempted (e.g. the failure occurred with committed_count == 0
        /// but at a point where restore was not applicable — should not
        /// occur in practice, but kept `Option` for honesty).
        rollback_fault: Option<DriverFault>,
        /// Exact number of granules (from the start of the requested range)
        /// that successfully switched to the new backing before the failure.
        /// `reservation`'s block list already reflects this split: granules
        /// `[0, committed_count)` are on the new backing, the remainder is
        /// either still on the old backing (readable) or poisoned (see
        /// `poisoned_range`).
        committed_count: usize,
        /// Physical handles whose ownership/mapping state is no longer
        /// trusted (a mapping attempt AND its restore both failed, or a
        /// handle could not be safely returned). These are also recorded in
        /// `reservation`'s own `quarantined_blocks()` — this list is a copy
        /// for the caller's immediate inspection, not a second authority.
        quarantined: Vec<MappedBlock>,
        /// `Some((offset, len))` identifying a byte range within the request
        /// whose content is no longer proven readable (its handle's mapping
        /// state is ambiguous) — the caller must never route through this
        /// range again. `None` when every granule outside the committed
        /// prefix remains provably readable on its original backing.
        poisoned_range: Option<(usize, usize)>,
    },
}

impl TransitionOutcome {
    pub fn is_committed(&self) -> bool {
        matches!(self, Self::Committed { .. })
    }

    /// Whether the *entire requested range* remains exactly as it was before
    /// the call (no commit, no poison, no quarantine). `RolledBack` is the
    /// only outcome besides `Rejected`/`Committed` (0 granules) for which
    /// this is `true`.
    pub fn stable_va_intact(&self) -> bool {
        matches!(self, Self::RolledBack { .. } | Self::Rejected { .. })
            || matches!(self, Self::Committed { granules: 0, .. })
    }

    /// Whether any part of the requested range is no longer safely readable.
    pub fn has_poisoned_range(&self) -> bool {
        matches!(
            self,
            Self::Fatal {
                poisoned_range: Some(_),
                ..
            }
        )
    }
}

// ---------------------------------------------------------------------------
// Safe-point token
// ---------------------------------------------------------------------------

/// Proof that [`ResizeSafePoint::is_safe`] returned `true` at call time.
pub struct VerifiedSafePoint(#[allow(dead_code)] ResizeSafePoint);

/// Verify a safe point. Returns `Ok` when safe, or `Err(reason)` otherwise.
pub fn verify_safe_point(point: ResizeSafePoint) -> Result<VerifiedSafePoint, &'static str> {
    match point.blocking_reason() {
        Some(reason) => Err(reason),
        None => Ok(VerifiedSafePoint(point)),
    }
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

/// Transition granule range `[offset, offset + len)` to `new_location`.
///
/// See module-level documentation for the full contract.
#[allow(clippy::too_many_arguments)]
pub fn transition_granule_range(
    runtime: &CudaRuntime,
    reservation: &mut CudaReservation,
    backing: &CudaVirtualBacking,
    offset: usize,
    len: usize,
    new_location: PhysicalLocation,
    old_pool: &Arc<PhysicalHandlePool>,
    new_pool: &Arc<PhysicalHandlePool>,
    safe_point: &VerifiedSafePoint,
    recheck_safe_point: impl Fn() -> ResizeSafePoint,
) -> TransitionOutcome {
    transition_granule_range_inner(
        runtime,
        reservation,
        backing,
        offset,
        len,
        new_location,
        old_pool,
        new_pool,
        safe_point,
        recheck_safe_point,
        #[cfg(any(test, feature = "gpu-tests"))]
        None,
    )
}

/// Test-only entry point identical to [`transition_granule_range`] but with a
/// [`onnx_runtime_cuda_memory::release::DriverFaultPlan`] that can force any
/// individual Phase-8 `cuMemUnmap`/`cuMemMap`/`cuMemSetAccess` call (by 1-based
/// call count across the whole Phase 8 loop, per operation) to fail
/// deterministically, so `RolledBack`/`Fatal` classification, `committed_count`,
/// quarantine population, and leak-freedom can be proven for granule 0 and
/// later granules without relying on real, non-reproducible driver failures.
///
/// Not reachable from production: the parameter only exists under
/// `#[cfg(any(test, feature = "gpu-tests"))]`, mirroring the pattern already
/// used by [`onnx_runtime_cuda_memory::virtual_memory::CudaVirtualBacking::with_driver_faults`].
#[cfg(any(test, feature = "gpu-tests"))]
#[allow(clippy::too_many_arguments)]
pub fn transition_granule_range_with_phase8_faults(
    runtime: &CudaRuntime,
    reservation: &mut CudaReservation,
    backing: &CudaVirtualBacking,
    offset: usize,
    len: usize,
    new_location: PhysicalLocation,
    old_pool: &Arc<PhysicalHandlePool>,
    new_pool: &Arc<PhysicalHandlePool>,
    safe_point: &VerifiedSafePoint,
    recheck_safe_point: impl Fn() -> ResizeSafePoint,
    phase8_faults: Arc<onnx_runtime_cuda_memory::release::DriverFaultPlan>,
) -> TransitionOutcome {
    transition_granule_range_inner(
        runtime,
        reservation,
        backing,
        offset,
        len,
        new_location,
        old_pool,
        new_pool,
        safe_point,
        recheck_safe_point,
        Some(phase8_faults),
    )
}

#[allow(clippy::too_many_arguments)]
fn transition_granule_range_inner(
    runtime: &CudaRuntime,
    reservation: &mut CudaReservation,
    backing: &CudaVirtualBacking,
    offset: usize,
    len: usize,
    new_location: PhysicalLocation,
    old_pool: &Arc<PhysicalHandlePool>,
    new_pool: &Arc<PhysicalHandlePool>,
    _safe_point: &VerifiedSafePoint,
    recheck_safe_point: impl Fn() -> ResizeSafePoint,
    #[cfg(any(test, feature = "gpu-tests"))] phase8_faults: Option<
        Arc<onnx_runtime_cuda_memory::release::DriverFaultPlan>,
    >,
) -> TransitionOutcome {
    // ── Phase 0: argument validation ─────────────────────────────────────────
    if len == 0 {
        return TransitionOutcome::Committed {
            granules: 0,
            new_owned_bytes: 0,
            old_released_bytes: 0,
        };
    }
    if old_pool.location() == new_location {
        return TransitionOutcome::Committed {
            granules: 0,
            new_owned_bytes: 0,
            old_released_bytes: 0,
        };
    }

    let granularity = old_pool.granularity();
    if granularity == 0 || !offset.is_multiple_of(granularity) || !len.is_multiple_of(granularity) {
        return TransitionOutcome::RolledBack {
            fault: DriverFault::new(
                "transition_granule_range",
                format!("offset/len not aligned to granularity {granularity}"),
            ),
        };
    }
    let granule_count = len / granularity;

    // ── Phase 1: validate range fully committed ───────────────────────────────
    let old_blocks = backing.blocks_in_range_pub(reservation, offset, len);
    if old_blocks.len() != granule_count {
        return TransitionOutcome::RolledBack {
            fault: DriverFault::new(
                "transition_granule_range precondition",
                format!(
                    "expected {granule_count} committed blocks in [{offset}, {}), found {}",
                    offset + len,
                    old_blocks.len()
                ),
            ),
        };
    }

    // ── Phase 2: acquire new physical handles ─────────────────────────────────
    // Acquire all handles before touching the stable VA. A shortfall here has
    // zero impact on the reservation.
    let mut new_handles: Vec<(cu::CUmemGenericAllocationHandle, u64)> =
        Vec::with_capacity(granule_count);
    let mut acquire_fault: Option<DriverFault> = None;

    for _ in 0..granule_count {
        match new_pool.acquire_handle_raw() {
            Ok((h, c)) => new_handles.push((h, c)),
            Err(e) => {
                acquire_fault = Some(DriverFault::new(
                    "acquire_handle_raw (new-location handle)",
                    e.to_string(),
                ));
                break;
            }
        }
    }

    if let Some(fault) = acquire_fault {
        for (h, _) in new_handles {
            new_pool.return_handle_unmapped(h);
        }
        return TransitionOutcome::RolledBack { fault };
    }

    // ── Phase 3: reserve staging VMM window ───────────────────────────────────
    let staging_result = <CudaVirtualBacking as VirtualBacking>::reserve(backing, len);
    let staging_reservation = match staging_result {
        Ok(r) => r,
        Err(e) => {
            for (h, _) in new_handles {
                new_pool.return_handle_unmapped(h);
            }
            return TransitionOutcome::RolledBack {
                fault: DriverFault::new("cuMemAddressReserve (staging VMM)", e.to_string()),
            };
        }
    };
    let staging_base: CUdeviceptr = staging_reservation.base_ptr();
    let stable_base: CUdeviceptr = reservation.base_ptr();
    let device_ordinal = new_pool.device_ordinal_pub();

    // ── Phase 4: map new handles at staging VA + set device access ────────────
    let mut staging_mapped: usize = 0;
    let mut phase4_fault: Option<DriverFault> = None;
    {
        let _section = capture_gate::synchronizing_section();
        for (i, &(handle, _)) in new_handles.iter().enumerate() {
            let addr = staging_base + (i * granularity) as u64;
            // SAFETY: addr is inside the staging reservation (just reserved);
            // handle is a fresh, unmapped, owned handle from new_pool.
            if unsafe { cu::cuMemMap(addr, granularity, 0, handle, 0) }
                != cu::CUresult::CUDA_SUCCESS
            {
                phase4_fault = Some(DriverFault::new(
                    "cuMemMap (staging VMM, new handle)",
                    format!("granule {i}"),
                ));
                break;
            }
            staging_mapped += 1;

            let mut access: cu::CUmemAccessDesc = unsafe { std::mem::zeroed() };
            access.location.type_ = cu::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE;
            access.location.id = device_ordinal;
            access.flags = cu::CUmemAccess_flags::CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
            if unsafe { cu::cuMemSetAccess(addr, granularity, &access, 1) }
                != cu::CUresult::CUDA_SUCCESS
            {
                // Unmap what we just mapped so staging_mapped stays consistent.
                unsafe {
                    let _ = cu::cuMemUnmap(addr, granularity);
                }
                staging_mapped -= 1;
                phase4_fault = Some(DriverFault::new(
                    "cuMemSetAccess (staging VMM, new handle)",
                    format!("granule {i}"),
                ));
                break;
            }
        }
    }

    if let Some(fault) = phase4_fault {
        // Rollback: unmap staged handles, return all new handles to pool.
        {
            let _section = capture_gate::synchronizing_section();
            for i in 0..staging_mapped {
                unsafe {
                    let _ = cu::cuMemUnmap(staging_base + (i * granularity) as u64, granularity);
                }
            }
        }
        drop(staging_reservation);
        for (h, _) in new_handles {
            new_pool.return_handle_unmapped(h);
        }
        return TransitionOutcome::RolledBack { fault };
    }

    // ── Phase 5: copy old stable-VA bytes → staging VMM ──────────────────────
    // dtod_async issues on the compute stream. drain_for_unmap (Phase 6) will
    // sync both this copy AND any prior compute readers of the stable VA.
    //
    // SAFETY: stable_base + offset is fully mapped (Phase 1 validated);
    // staging_base is fully mapped (Phase 4); both are `len` bytes.
    if let Err(e) = unsafe { runtime.dtod_async(stable_base + offset as u64, staging_base, len) } {
        {
            let _section = capture_gate::synchronizing_section();
            for i in 0..granule_count {
                unsafe {
                    let _ = cu::cuMemUnmap(staging_base + (i * granularity) as u64, granularity);
                }
            }
        }
        drop(staging_reservation);
        for (h, _) in new_handles {
            new_pool.return_handle_unmapped(h);
        }
        return TransitionOutcome::RolledBack {
            fault: DriverFault::new("cuMemcpyDtoDAsync (stable VA → staging VMM)", e.to_string()),
        };
    }

    // ── Phase 6: drain for unmap ──────────────────────────────────────────────
    // Proves: (a) the copy landed in staging, (b) no compute reader of the old
    // stable VA is still in-flight. See module doc for multi-stream argument.
    if let Err(e) = runtime.drain_for_unmap() {
        {
            let _section = capture_gate::synchronizing_section();
            for i in 0..granule_count {
                unsafe {
                    let _ = cu::cuMemUnmap(staging_base + (i * granularity) as u64, granularity);
                }
            }
        }
        drop(staging_reservation);
        for (h, _) in new_handles {
            new_pool.return_handle_unmapped(h);
        }
        return TransitionOutcome::RolledBack {
            fault: DriverFault::new("drain_for_unmap (reader drain + copy sync)", e.to_string()),
        };
    }

    // ── Phase 7: re-check safe point immediately before stable-VA unmap ───────
    let recheck = recheck_safe_point();
    if let Some(reason) = recheck.blocking_reason() {
        {
            let _section = capture_gate::synchronizing_section();
            for i in 0..granule_count {
                unsafe {
                    let _ = cu::cuMemUnmap(staging_base + (i * granularity) as u64, granularity);
                }
            }
        }
        drop(staging_reservation);
        for (h, _) in new_handles {
            new_pool.return_handle_unmapped(h);
        }
        return TransitionOutcome::Rejected { reason };
    }

    // ── Phase 8: per-granule atomic switch ────────────────────────────────────
    // All driver calls are inside one synchronizing_section (device-syncing ops).
    //
    // Every failure branch below must resolve to exactly one of:
    //   - RolledBack: only when committed_count == 0 AND the old mapping/
    //     access for the interrupted granule is proven restored (or never
    //     touched). No handle is quarantined, no range is poisoned.
    //   - Fatal: whenever committed_count > 0 (a real prefix already
    //     switched, so the whole-range "old mapping intact" claim of
    //     RolledBack would be false) OR whenever this granule's own mapping
    //     state could not be proven either "new" or "restored old"
    //     (poisoned_range is set for exactly that granule, never the whole
    //     range, so untouched granules after it are correctly reported as
    //     still old-backed and readable).
    let mut committed_count = 0usize;
    let mut total_new_owned: u64 = 0;
    let mut total_old_released: u64 = 0;

    // Populated only on a Fatal exit from the loop below.
    struct FatalState {
        transition_fault: DriverFault,
        rollback_fault: Option<DriverFault>,
        quarantined: Vec<MappedBlock>,
        poisoned_range: Option<(usize, usize)>,
    }
    let mut fatal: Option<FatalState> = None;

    // Set device access at `addr`; returns whether it succeeded.
    let set_access = |addr: CUdeviceptr| -> bool {
        let mut access: cu::CUmemAccessDesc = unsafe { std::mem::zeroed() };
        access.location.type_ = cu::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE;
        access.location.id = device_ordinal;
        access.flags = cu::CUmemAccess_flags::CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        unsafe { cu::cuMemSetAccess(addr, granularity, &access, 1) == cu::CUresult::CUDA_SUCCESS }
    };

    // Deterministic fault injection (test/gpu-tests builds only): records one
    // call of `op` against `phase8_faults` and reports whether it must be
    // failed instead of issuing the real driver call. Every real Phase-8
    // driver call below is routed through the matching `*_checked` helper so
    // a scheduled fault always take effect regardless of which granule (0 or
    // later) or which call site it targets.
    #[cfg(any(test, feature = "gpu-tests"))]
    let should_inject = |op: onnx_runtime_cuda_memory::release::DriverOperation| -> bool {
        phase8_faults
            .as_ref()
            .is_some_and(|plan| plan.should_fail(op))
    };

    let unmap_checked = |addr: CUdeviceptr| -> bool {
        #[cfg(any(test, feature = "gpu-tests"))]
        if should_inject(onnx_runtime_cuda_memory::release::DriverOperation::Unmap) {
            return false;
        }
        unsafe { cu::cuMemUnmap(addr, granularity) == cu::CUresult::CUDA_SUCCESS }
    };
    let map_checked = |addr: CUdeviceptr, handle: cu::CUmemGenericAllocationHandle| -> bool {
        #[cfg(any(test, feature = "gpu-tests"))]
        if should_inject(onnx_runtime_cuda_memory::release::DriverOperation::Remap) {
            return false;
        }
        unsafe { cu::cuMemMap(addr, granularity, 0, handle, 0) == cu::CUresult::CUDA_SUCCESS }
    };
    let set_access_checked = |addr: CUdeviceptr| -> bool {
        #[cfg(any(test, feature = "gpu-tests"))]
        if should_inject(onnx_runtime_cuda_memory::release::DriverOperation::SetAccess) {
            return false;
        }
        set_access(addr)
    };

    {
        let _section = capture_gate::synchronizing_section();

        'granules: for (i, &(new_handle, charged)) in new_handles.iter().enumerate() {
            let old_block = old_blocks[i];
            let old_handle = old_block.handle;
            let stable_offset = old_block.offset;
            let stable_addr = stable_base + stable_offset as u64;
            let staging_addr = staging_base + (i * granularity) as u64;

            // Common cleanup for every abort path: unmap all remaining
            // staging mappings from `i` onward and return the still-staged
            // new handles that will not be used.
            let cleanup_remaining_staging_and_new = || {
                for j in i..granule_count {
                    unsafe {
                        let _ =
                            cu::cuMemUnmap(staging_base + (j * granularity) as u64, granularity);
                    }
                }
                for &(handle, _) in &new_handles[i..] {
                    new_pool.return_handle_unmapped(handle);
                }
            };

            // (a) Unmap old from stable VA.
            if !unmap_checked(stable_addr) {
                // `cuMemUnmap` failed *before* any mutation: the driver never
                // touched this mapping, so it is provably still old-backed
                // and readable, exactly as documented for `RolledBack`. This
                // fixes finding #3: this branch previously always reported
                // `Fatal`, unlike the equivalent zero-side-effect failures in
                // branches (b)/(c) below.
                cleanup_remaining_staging_and_new();
                let fault = DriverFault::new("cuMemUnmap (old, stable VA)", format!("granule {i}"));
                if committed_count == 0 {
                    drop(_section);
                    drop(staging_reservation);
                    return TransitionOutcome::RolledBack { fault };
                }
                // A real prefix already switched to the new backing, so the
                // whole-range "nothing changed" claim of RolledBack would be
                // false. Report Fatal with the exact committed prefix; the
                // interrupted granule and everything after it is untouched
                // and remains readable on the old backing (poisoned_range is
                // None — fixes finding #1/#2: no ambiguity, no empty
                // quarantine claim, and the caller can see exactly how much
                // committed).
                fatal = Some(FatalState {
                    transition_fault: fault,
                    rollback_fault: None,
                    quarantined: Vec::new(),
                    poisoned_range: None,
                });
                break 'granules;
            }

            // (b) Map new at stable VA (double-mapping: staging + stable).
            if !map_checked(stable_addr, new_handle) {
                // Stable VA is UNMAPPED. Try to restore the old mapping.
                let transition_fault =
                    DriverFault::new("cuMemMap (new, stable VA)", format!("granule {i}"));
                let restore_map_ok = map_checked(stable_addr, old_handle);
                let restore_ok = restore_map_ok && set_access_checked(stable_addr);

                cleanup_remaining_staging_and_new();

                if !restore_ok {
                    // Neither the new mapping nor the restore succeeded: the
                    // stable VA at this offset is either unmapped or mapped
                    // without correct access — not provably readable. This is
                    // the true handle/mapping ambiguity (finding #4). Fix:
                    // never silently drop `old_handle` here, and never route
                    // it back through `old_pool` (which would either put it
                    // back in the reuse pool for some unrelated future
                    // caller, or `cuMemRelease` it outright) — either would
                    // desync `reservation.quarantined` from the handle's
                    // real, driver-visible ownership. The reservation's own
                    // quarantine authority (not the pool, not a second
                    // authority) is the sole owner of an ambiguous handle
                    // from this point on.
                    //
                    // If the map itself succeeded but access failed, the
                    // mapping must first be torn down so `old_handle` is not
                    // simultaneously mapped at `stable_addr` AND quarantined.
                    if restore_map_ok {
                        let _ = unsafe { cu::cuMemUnmap(stable_addr, granularity) };
                    }
                    reservation.push_quarantined_block(old_block);
                    let poisoned = MappedBlock::new(stable_offset, granularity, old_handle);
                    fatal = Some(FatalState {
                        transition_fault,
                        rollback_fault: Some(DriverFault::new(
                            "cuMemMap (restore old, stable VA)",
                            format!("granule {i}: restore also failed"),
                        )),
                        quarantined: vec![poisoned],
                        poisoned_range: Some((stable_offset, granularity)),
                    });
                    break 'granules;
                }

                // Old mapping (+ access) fully restored: granule i is exactly
                // as it was before this call.
                if committed_count == 0 {
                    drop(_section);
                    drop(staging_reservation);
                    return TransitionOutcome::RolledBack {
                        fault: transition_fault,
                    };
                }
                fatal = Some(FatalState {
                    transition_fault,
                    rollback_fault: None,
                    quarantined: Vec::new(),
                    poisoned_range: None,
                });
                break 'granules;
            }

            // (c) Set device access at stable VA for the new handle.
            if !set_access_checked(stable_addr) {
                // New handle mapped at stable VA but access denied. Unmap
                // new, re-map old, restore old's access.
                let _ = unmap_checked(stable_addr);
                let transition_fault =
                    DriverFault::new("cuMemSetAccess (new, stable VA)", format!("granule {i}"));
                let restore_map_ok = map_checked(stable_addr, old_handle);
                let restore_ok = restore_map_ok && set_access_checked(stable_addr);

                cleanup_remaining_staging_and_new();

                if !restore_ok {
                    // Same handle-ambiguity fix as case (b): the mapping
                    // state of `old_handle` is no longer trusted, so it must
                    // go through the reservation's own quarantine authority
                    // exclusively — never also through `old_pool` (which
                    // would either recycle it to an unrelated caller or
                    // `cuMemRelease` it, desyncing pool/reservation state).
                    if restore_map_ok {
                        let _ = unmap_checked(stable_addr);
                    }
                    reservation.push_quarantined_block(old_block);
                    let poisoned = MappedBlock::new(stable_offset, granularity, old_handle);
                    fatal = Some(FatalState {
                        transition_fault,
                        rollback_fault: Some(DriverFault::new(
                            "cuMemMap/cuMemSetAccess (restore old after set-access fail)",
                            format!("granule {i}"),
                        )),
                        quarantined: vec![poisoned],
                        poisoned_range: Some((stable_offset, granularity)),
                    });
                    break 'granules;
                }

                if committed_count == 0 {
                    drop(_section);
                    drop(staging_reservation);
                    return TransitionOutcome::RolledBack {
                        fault: transition_fault,
                    };
                }
                fatal = Some(FatalState {
                    transition_fault,
                    rollback_fault: None,
                    quarantined: Vec::new(),
                    poisoned_range: None,
                });
                break 'granules;
            }

            // (d) Unmap new handle from staging VA (double-mapping ends).
            unsafe {
                let _ = cu::cuMemUnmap(staging_addr, granularity);
            }

            // (e) Return old handle to old_pool (it was unmapped in step a).
            let _ = old_pool.return_after_unmap_pub(old_handle);

            // Update reservation's block list.
            reservation.swap_block(
                old_block,
                MappedBlock::new(stable_offset, granularity, new_handle),
                Arc::clone(new_pool),
            );

            total_new_owned = total_new_owned.saturating_add(charged);
            total_old_released = total_old_released.saturating_add(granularity as u64);
            committed_count += 1;
        }
    }

    drop(staging_reservation);

    if let Some(state) = fatal {
        return TransitionOutcome::Fatal {
            transition_fault: state.transition_fault,
            rollback_fault: state.rollback_fault,
            committed_count,
            quarantined: state.quarantined,
            poisoned_range: state.poisoned_range,
        };
    }

    TransitionOutcome::Committed {
        granules: committed_count,
        new_owned_bytes: total_new_owned,
        old_released_bytes: total_old_released,
    }
}

// ---------------------------------------------------------------------------
// Timing-annotated variant (for benchmarks)
// ---------------------------------------------------------------------------

/// Per-phase timing breakdown from [`transition_granule_range_timed`].
#[derive(Clone, Debug, Default)]
pub struct TransitionTimings {
    pub drain_us: f64,
    pub total_us: f64,
}

/// Same as [`transition_granule_range`] but records total wall-clock timing.
#[allow(clippy::too_many_arguments)]
pub fn transition_granule_range_timed(
    runtime: &CudaRuntime,
    reservation: &mut CudaReservation,
    backing: &CudaVirtualBacking,
    offset: usize,
    len: usize,
    new_location: PhysicalLocation,
    old_pool: &Arc<PhysicalHandlePool>,
    new_pool: &Arc<PhysicalHandlePool>,
    safe_point: &VerifiedSafePoint,
    recheck_safe_point: impl Fn() -> ResizeSafePoint,
    timings: &mut TransitionTimings,
) -> TransitionOutcome {
    let t0 = std::time::Instant::now();
    let result = transition_granule_range(
        runtime,
        reservation,
        backing,
        offset,
        len,
        new_location,
        old_pool,
        new_pool,
        safe_point,
        recheck_safe_point,
    );
    timings.total_us = t0.elapsed().as_secs_f64() * 1e6;
    result
}
