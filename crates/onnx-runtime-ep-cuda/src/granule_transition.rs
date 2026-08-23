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
    /// Failure occurred before the stable-VA switch; rollback succeeded.
    /// Stable VA unchanged and all original mappings intact.
    RolledBack { fault: DriverFault },
    /// Failure AND rollback also failed. VA range is permanently unusable.
    Fatal {
        transition_fault: DriverFault,
        rollback_fault: DriverFault,
        quarantined: Vec<MappedBlock>,
    },
}

impl TransitionOutcome {
    pub fn is_committed(&self) -> bool {
        matches!(self, Self::Committed { .. })
    }
    pub fn stable_va_intact(&self) -> bool {
        !matches!(self, Self::Fatal { .. })
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
    _safe_point: &VerifiedSafePoint,
    recheck_safe_point: impl Fn() -> ResizeSafePoint,
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
    let mut committed_count = 0usize;
    let mut total_new_owned: u64 = 0;
    let mut total_old_released: u64 = 0;
    let mut fatal: Option<(DriverFault, DriverFault)> = None;

    {
        let _section = capture_gate::synchronizing_section();

        'granules: for (i, &(new_handle, charged)) in new_handles.iter().enumerate() {
            let old_block = old_blocks[i];
            let old_handle = old_block.handle;
            let stable_offset = old_block.offset;
            let stable_addr = stable_base + stable_offset as u64;
            let staging_addr = staging_base + (i * granularity) as u64;

            // (a) Unmap old from stable VA.
            if unsafe { cu::cuMemUnmap(stable_addr, granularity) } != cu::CUresult::CUDA_SUCCESS {
                // Old mapping intact. Unmap remaining staging handles.
                for j in i..granule_count {
                    unsafe {
                        let _ =
                            cu::cuMemUnmap(staging_base + (j * granularity) as u64, granularity);
                    }
                }
                // Return remaining new handles.
                for &(handle, _) in &new_handles[i..] {
                    new_pool.return_handle_unmapped(handle);
                }
                let fault = DriverFault::new("cuMemUnmap (old, stable VA)", format!("granule {i}"));
                if committed_count == 0 {
                    fatal = Some((fault.clone(), fault));
                } else {
                    // Some committed; some still on old backing. Fatal partial state.
                    fatal = Some((
                        fault.clone(),
                        DriverFault::new(
                            "partial-committed-rollback-impossible",
                            format!(
                                "{committed_count} granules already switched, remaining on old backing"
                            ),
                        ),
                    ));
                }
                break 'granules;
            }

            // (b) Map new at stable VA (double-mapping: staging + stable).
            if unsafe { cu::cuMemMap(stable_addr, granularity, 0, new_handle, 0) }
                != cu::CUresult::CUDA_SUCCESS
            {
                // Stable VA is UNMAPPED. Try to restore old mapping.
                let restore = unsafe { cu::cuMemMap(stable_addr, granularity, 0, old_handle, 0) };
                let transition_fault =
                    DriverFault::new("cuMemMap (new, stable VA)", format!("granule {i}"));
                if restore != cu::CUresult::CUDA_SUCCESS {
                    // Restore also failed. Fatal.
                    for j in i..granule_count {
                        unsafe {
                            let _ = cu::cuMemUnmap(
                                staging_base + (j * granularity) as u64,
                                granularity,
                            );
                        }
                    }
                    for &(handle, _) in &new_handles[i..] {
                        new_pool.return_handle_unmapped(handle);
                    }
                    fatal = Some((
                        transition_fault,
                        DriverFault::new(
                            "cuMemMap (restore old, stable VA)",
                            format!("granule {i}: restore also failed"),
                        ),
                    ));
                    break 'granules;
                }
                // Restored old handle at stable VA. Set access.
                let mut old_access: cu::CUmemAccessDesc = unsafe { std::mem::zeroed() };
                old_access.location.type_ = cu::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE;
                old_access.location.id = device_ordinal;
                old_access.flags = cu::CUmemAccess_flags::CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
                let _ = unsafe { cu::cuMemSetAccess(stable_addr, granularity, &old_access, 1) };
                // Unmap remaining staging, return new handles.
                for j in i..granule_count {
                    unsafe {
                        let _ =
                            cu::cuMemUnmap(staging_base + (j * granularity) as u64, granularity);
                    }
                }
                for &(handle, _) in &new_handles[i..] {
                    new_pool.return_handle_unmapped(handle);
                }
                if committed_count == 0 {
                    // Clean rollback; return RolledBack.
                    // Drop staging before returning.
                    drop(_section);
                    drop(staging_reservation);
                    return TransitionOutcome::RolledBack {
                        fault: transition_fault,
                    };
                }
                fatal = Some((
                    transition_fault,
                    DriverFault::new(
                        "partial-committed",
                        format!("{committed_count} committed before rollback at granule {i}"),
                    ),
                ));
                break 'granules;
            }

            // (c) Set device access at stable VA for the new handle.
            let mut access: cu::CUmemAccessDesc = unsafe { std::mem::zeroed() };
            access.location.type_ = cu::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE;
            access.location.id = device_ordinal;
            access.flags = cu::CUmemAccess_flags::CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
            if unsafe { cu::cuMemSetAccess(stable_addr, granularity, &access, 1) }
                != cu::CUresult::CUDA_SUCCESS
            {
                // New handle mapped at stable VA but no access. Unmap new, re-map old.
                let _ = unsafe { cu::cuMemUnmap(stable_addr, granularity) };
                let restore = unsafe { cu::cuMemMap(stable_addr, granularity, 0, old_handle, 0) };
                let transition_fault =
                    DriverFault::new("cuMemSetAccess (new, stable VA)", format!("granule {i}"));
                if restore != cu::CUresult::CUDA_SUCCESS {
                    for j in i..granule_count {
                        unsafe {
                            let _ = cu::cuMemUnmap(
                                staging_base + (j * granularity) as u64,
                                granularity,
                            );
                        }
                    }
                    for &(handle, _) in &new_handles[i..] {
                        new_pool.return_handle_unmapped(handle);
                    }
                    fatal = Some((
                        transition_fault,
                        DriverFault::new(
                            "cuMemMap (restore old after set-access fail)",
                            format!("granule {i}"),
                        ),
                    ));
                    break 'granules;
                }
                let mut old_access: cu::CUmemAccessDesc = unsafe { std::mem::zeroed() };
                old_access.location.type_ = cu::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE;
                old_access.location.id = device_ordinal;
                old_access.flags = cu::CUmemAccess_flags::CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
                let _ = unsafe { cu::cuMemSetAccess(stable_addr, granularity, &old_access, 1) };
                for j in i..granule_count {
                    unsafe {
                        let _ =
                            cu::cuMemUnmap(staging_base + (j * granularity) as u64, granularity);
                    }
                }
                for &(handle, _) in &new_handles[i..] {
                    new_pool.return_handle_unmapped(handle);
                }
                if committed_count == 0 {
                    drop(_section);
                    drop(staging_reservation);
                    return TransitionOutcome::RolledBack {
                        fault: transition_fault,
                    };
                }
                fatal = Some((
                    transition_fault,
                    DriverFault::new(
                        "partial-committed-after-access-fail",
                        format!("{committed_count} committed before rollback"),
                    ),
                ));
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
            );

            total_new_owned = total_new_owned.saturating_add(charged);
            total_old_released = total_old_released.saturating_add(granularity as u64);
            committed_count += 1;
        }
    }

    drop(staging_reservation);

    if let Some((tf, rf)) = fatal {
        return TransitionOutcome::Fatal {
            transition_fault: tf,
            rollback_fault: rf,
            quarantined: Vec::new(),
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
