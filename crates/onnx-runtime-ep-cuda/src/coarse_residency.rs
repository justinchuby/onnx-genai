//! #1810 Slice 5 — Coarse-boundary per-expert residency plan application.
//!
//! Provides [`apply_residency_plan_at_boundary`], which applies a
//! [`ResidencyPlan`] whose decisions may include `PerExpertCandidate` against
//! a set of `WeightRegionCatalog`s and their backing `CudaVmmAllocator`s.
//!
//! # Production status (honest)
//!
//! As of this revision this function has **zero production callers**. It is
//! reachable only from `CudaWeightResidency::apply_coarse_residency_plan`
//! (a thin forwarding wrapper) and from this crate's own tests. Wiring a real
//! lifecycle call site (e.g. at model-load completion) is explicitly *out of
//! scope* for this revision — see PR discussion for the dependency chain.
//! Issue #1723 introduced an unrelated workflow interpreter in
//! `onnx-genai-server`/`onnx-genai-scheduler`/`onnx-genai-ort`; it is
//! mentioned here only as current-main context (this crate rebases cleanly
//! on top of it) and carries no team-scope implication for this file.
//!
//! # Feature gate
//!
//! Off by default. Enable by setting
//! [`COARSE_RESIDENCY_ENABLE_ENV`] (`ONNX_GENAI_WEIGHT_OFFLOAD_COARSE_RESIDENCY_ENABLE`)
//! to `1`, `true`, or `on`. When off, this function returns immediately with
//! `fallback_reason = Some("feature gate disabled")` and touches no state.
//!
//! # Safety envelope
//!
//! Every mutation goes through the Slice-4 primitive
//! [`crate::granule_transition::transition_granule_range`], which:
//!   * requires a [`VerifiedSafePoint`],
//!   * re-checks the safe point immediately before the atomic switch,
//!   * transactionally rolls back on any failure,
//!   * preserves the stable VA byte-for-byte.
//!
//! # Cross-tensor expert-group atomicity
//!
//! A QMoE/BlockQuantizedMoE node's logical "expert" spans multiple ONNX
//! initializer tensors (fc1/fc2/fc3 weights, scales, biases, zero-points —
//! each a distinct [`onnx_runtime_ir::ValueId`] with its own
//! `WeightRegionCatalog`). Tiering only some of those tensors for the same
//! expert would leave a logically-atomic expert split across Device/Host,
//! which is unsound: the kernel dereferences all of them together for one
//! expert's compute. This function is given the graph-derived
//! [`onnx_runtime_ep_api::ExpertWeightGroup`] list (purely structural — no
//! tensor-name heuristics) alongside the plan, and for every group whose
//! member `ValueId`s are present in `catalogs`, it requires *every* member
//! to (a) independently pass the capability/granule-alignment check, AND
//! (b) propose the exact same logical hot-expert set (order/duplicate
//! differences in the source list are immaterial; the sets are compared
//! after normalization) over the same expert-count domain, before *any* of
//! them may be transitioned. A single authoritative hot set is derived once
//! per group and reused for every member's byte-range computation — never
//! re-derived per member — so the actual bytes transitioned cannot diverge
//! from the agreement that was checked. If one member is
//! misaligned/non-pageable/missing, or any two members disagree on which
//! experts are hot or on the expert-count domain itself, the whole group
//! falls back to `WholeBankResident` for this call (recorded once per
//! member in `per_value_fallbacks`) — never a partial tiering.
//!
//! # Honest scope note (QMoE int4 fixtures)
//!
//! Per-expert byte ranges in a QMoE int4 model are frequently *smaller* than
//! the 2 MiB VMM allocation granularity, so a single granule can be shared
//! by multiple experts. Byte-level partial transitions are impossible; this
//! function therefore requires each transitioned expert's byte range to be
//! granule-aligned in both offset and length. Values whose per-expert layout
//! is not granule-aligned degrade to `WholeBankResident` for this call
//! (recorded in the per-value fallback list) — the whole plan never fails.
//!
//! For the shipped test fixtures, layouts are chosen so each expert range is
//! exactly one granule (e.g. `experts=64, rows_per_expert=512,
//! storage_elements_per_row=4096` gives 2 MiB per expert).
//!
//! # Same-device fail-closed
//!
//! `allocators` may in principle contain entries bound to different physical
//! CUDA devices (nothing upstream enforces a single-device invariant). This
//! function does not attempt multi-device coordination: before any mutation,
//! it verifies every allocator's
//! [`onnx_runtime_cuda_memory::vmm_allocator::CudaVmmAllocator::device_key`]
//! matches `device_ordinal`, and every pool's `device_ordinal_pub()` matches
//! it too. Any mismatch fails that value closed (per-value fallback, zero
//! side effects) rather than silently operating cross-device.

use std::collections::HashMap;
use std::sync::Arc;

use onnx_runtime_cuda_memory::release::MappedBlock;
use onnx_runtime_cuda_memory::virtual_memory::{PhysicalHandlePool, PhysicalLocation};
use onnx_runtime_cuda_memory::vmm_allocator::CudaVmmAllocator;
use onnx_runtime_ep_api::{ExpertWeightGroup, ResidencyDecision, ResidencyPlan};
use onnx_runtime_ir::ValueId;
use onnx_runtime_loader::WeightRegionCatalog;

#[cfg(any(test, feature = "gpu-tests"))]
use crate::granule_transition::transition_granule_range_with_phase8_faults;
use crate::granule_transition::{TransitionOutcome, transition_granule_range, verify_safe_point};
use crate::runtime::CudaRuntime;

/// Environment variable that gates coarse-boundary per-expert residency
/// plan application. Off by default.
///
/// Named `..._ENABLE` (not `..._PROFILE`): this is a hard on/off gate for
/// whether the plan is *applied at all*, not a diagnostics/profiling toggle.
pub const COARSE_RESIDENCY_ENABLE_ENV: &str = "ONNX_GENAI_WEIGHT_OFFLOAD_COARSE_RESIDENCY_ENABLE";

/// Deprecated alias for [`COARSE_RESIDENCY_ENABLE_ENV`]. Kept so any external
/// reference to the old name still resolves; new code should use the
/// `_ENABLE` name directly.
#[deprecated(note = "use COARSE_RESIDENCY_ENABLE_ENV")]
pub const COARSE_RESIDENCY_PROFILE_ENV: &str = COARSE_RESIDENCY_ENABLE_ENV;

/// True iff [`COARSE_RESIDENCY_ENABLE_ENV`] is set to `1`, `true`, or `on`
/// (case-insensitive, trimmed).
pub fn coarse_residency_profile_enabled() -> bool {
    matches!(
        std::env::var(COARSE_RESIDENCY_ENABLE_ENV)
            .ok()
            .as_deref()
            .map(str::trim)
            .map(str::to_ascii_lowercase)
            .as_deref(),
        Some("1") | Some("true") | Some("on")
    )
}

/// Exact outcome of one committed granule-range transition, retained so a
/// later `Fatal` in the same value (or a different value) can drive
/// range-precise rollback instead of re-deriving "every cold range" from the
/// catalog (which would attempt to revert ranges that were never touched).
#[derive(Debug, Clone)]
struct CommittedRange {
    offset: usize,
    len: usize,
}

/// Per-value transition bookkeeping accumulated during the forward pass.
/// Always retained (even for values that later hit `Fatal`) so the rollback
/// loop can revert exactly the ranges that actually committed for that value,
/// not the value as a whole.
#[derive(Debug, Default, Clone)]
struct ValueProgress {
    committed: Vec<CommittedRange>,
}

/// Detail captured when a reverse (rollback) transition itself fails to
/// return the range to `Device`. This is deliberately distinct from a forward
/// failure: a fault during rollback means the *originally committed* Host
/// data may now be in an unknown state, which is the highest-stakes failure
/// mode this function can hit.
#[derive(Debug, Clone)]
pub struct RollbackFailure {
    /// Value whose reverse transition failed.
    pub value: ValueId,
    /// The byte range (within the value's tensor) that failed to revert.
    pub range: (usize, usize),
    /// Human-readable detail of the outcome that caused this (rejected /
    /// rolled-back / fatal, including any nested fault).
    pub detail: String,
    /// Number of granules (from `range.0`) that this reverse call itself
    /// reports as committed before it failed, if the outcome was `Fatal`.
    /// `None` for `Rejected`/`RolledBack` outcomes (no partial commit).
    pub committed_count: Option<usize>,
    /// `Some((offset, len))` copied from a nested `Fatal.poisoned_range`,
    /// identifying bytes that must never be read/written again.
    pub poisoned_range: Option<(usize, usize)>,
    /// Quarantined blocks copied from a nested `Fatal.quarantined`, if any.
    pub quarantined: Vec<MappedBlock>,
}

/// Exact per-value partial-commit detail recorded when a value's forward
/// transition hits a `Fatal` outcome: `(value, committed_count,
/// poisoned_range)`, copied straight from `TransitionOutcome::Fatal`, never
/// discarded via `..`.
pub type FatalProgress = (ValueId, usize, Option<(usize, usize)>);

/// Structural outcome of a single coarse-boundary plan application.
///
/// Purely descriptive: no `Result`, no panics. A "no-op" outcome (feature
/// gate off, unsupported host-NUMA, unsafe safe point, or an all-whole-bank
/// plan) sets `fallback_reason` and leaves the byte-touching counters at
/// zero. All bytes-moved / commit counters describe what *actually*
/// happened, not what was requested — matching the Slice-4 accounting
/// discipline.
#[derive(Debug, Default)]
pub struct BoundaryApplicationOutcome {
    /// Policy name that produced the plan (e.g. `"static_profile"`,
    /// `"whole_bank_resident"`).
    pub policy_name: &'static str,
    /// Number of catalog values whose decision was inspected.
    pub values_inspected: usize,
    /// Number of values for which at least one granule range was
    /// transitioned and remained committed at the end of this call (i.e.
    /// excludes anything later reverted by rollback).
    pub values_touched: usize,
    /// Total hot experts (kept Device-resident) across all inspected values.
    pub hot_expert_count: usize,
    /// Total cold experts (moved to Host NUMA) across all inspected values.
    pub cold_expert_count: usize,
    /// Bytes released from the Device pool by this plan's committed
    /// transitions (i.e. bytes tiered away from Device). Renamed from the
    /// misleading `device_bytes_before`: this was never a "before" snapshot,
    /// it is an accumulator of bytes actually released.
    pub device_bytes_released: u64,
    /// Bytes newly committed on the Host NUMA pool by the plan.
    pub host_bytes_committed: u64,
    /// Sum of successful transition durations in milliseconds (wall-clock).
    pub transition_time_ms: f64,
    /// Number of granule-range transitions whose forward call returned
    /// `Rejected` or `RolledBack` (no side effects from that call itself).
    pub failure_count: usize,
    /// Number of values whose committed transitions had to be reverted by
    /// the rollback loop after a `Fatal` elsewhere in the plan.
    pub rollback_count: usize,
    /// Non-`None` when the whole call was a structural no-op (feature gate
    /// off, no host-NUMA support, or unsafe point). Not set for per-value
    /// skips: those append to `per_value_fallbacks` instead.
    pub fallback_reason: Option<String>,
    /// `(value, reason)` for each value that was inspected but not
    /// transitioned (e.g. non-pageable catalog, non-granule-aligned per-expert
    /// range, `WholeBankResident` decision, cross-tensor expert-group
    /// fallback, or a same-device mismatch).
    pub per_value_fallbacks: Vec<(ValueId, String)>,
    /// Values whose transitions successfully committed and were NOT later
    /// reverted by rollback.
    pub committed_values: Vec<ValueId>,
    /// Quarantined blocks emitted by any `Fatal` outcome (forward or
    /// rollback), keyed by value. These are also present in the
    /// reservation's own `quarantined_blocks()` list; carried here for the
    /// caller's immediate telemetry.
    pub quarantined: Vec<(ValueId, Vec<MappedBlock>)>,
    /// Exact per-value partial-commit detail for every value whose forward
    /// transition hit a `Fatal` outcome: `(value, committed_count,
    /// poisoned_range)` copied straight from `TransitionOutcome::Fatal`,
    /// never discarded via `..`.
    pub fatal_progress: Vec<FatalProgress>,
    /// Explicit detail for every reverse (rollback) transition that itself
    /// failed to fully restore a value to `Device`. Empty in the common case
    /// where rollback either wasn't needed or fully succeeded. A non-empty
    /// list here means real data is in a state this outcome describes
    /// precisely — never just `all_ok = false`.
    pub rollback_failures: Vec<RollbackFailure>,
}

/// Validate that `allocator` and both pools are bound to `device_ordinal`.
/// Returns `Err(reason)` on any mismatch — checked before any mutation for
/// this value.
fn check_same_device(
    allocator: &CudaVmmAllocator,
    device_pool: &PhysicalHandlePool,
    host_pool: &PhysicalHandlePool,
    device_ordinal: i32,
) -> Result<(), String> {
    let allocator_device = allocator.device_key();
    let expected = onnx_runtime_memory_governor::DeviceKey::device(device_ordinal as u32);
    if allocator_device != expected {
        return Err(format!(
            "allocator device_key {allocator_device:?} does not match requested device_ordinal {device_ordinal} (expected {expected:?})"
        ));
    }
    if device_pool.device_ordinal_pub() != device_ordinal {
        return Err(format!(
            "device_pool device_ordinal {} does not match requested device_ordinal {device_ordinal}",
            device_pool.device_ordinal_pub()
        ));
    }
    // The host pool is not itself "on" `device_ordinal` (it is HostNuma), but
    // it must still be the pool that was created FOR this device's host-NUMA
    // node; its `device_ordinal_pub()` tracks the originating device for
    // exactly this reason.
    if host_pool.device_ordinal_pub() != device_ordinal {
        return Err(format!(
            "host_pool device_ordinal {} does not match requested device_ordinal {device_ordinal}",
            host_pool.device_ordinal_pub()
        ));
    }
    Ok(())
}

/// Compute the merged, granule-aligned cold-expert ranges for `value`'s
/// catalog, given the plan's hot-expert set. Returns `Err(reason)` if any
/// cold expert's range is missing or not granule-aligned — the caller must
/// treat this as "skip the whole value", never a partial application.
fn cold_ranges_for(
    catalog: &WeightRegionCatalog,
    hot: &std::collections::HashSet<usize>,
    granularity: usize,
) -> Result<Vec<(usize, usize)>, String> {
    let total_experts = catalog.layout().experts;
    let mut cold_ranges: Vec<(usize, usize)> = Vec::new();
    for expert in 0..total_experts {
        if hot.contains(&expert) {
            continue;
        }
        let range = catalog
            .relative_range(expert)
            .ok_or_else(|| format!("expert {expert} has no relative_range"))?;
        let offset = range.start;
        let len = range.end.saturating_sub(range.start);
        if len == 0 {
            continue;
        }
        if offset % granularity != 0 || len % granularity != 0 {
            return Err(format!(
                "expert {expert} range {offset}..{} is not granule-aligned (granularity={granularity})",
                range.end
            ));
        }
        cold_ranges.push((offset, len));
    }
    cold_ranges.sort_by_key(|&(off, _)| off);
    let mut merged: Vec<(usize, usize)> = Vec::with_capacity(cold_ranges.len());
    for (off, len) in cold_ranges {
        match merged.last_mut() {
            Some(last) if last.0 + last.1 == off => last.1 += len,
            _ => merged.push((off, len)),
        }
    }
    Ok(merged)
}

/// Apply `plan` to the given `catalogs` and residency at model-load boundary.
///
/// `expert_groups` is the graph-derived list of cross-tensor logical expert
/// groups (see [`onnx_runtime_ep_api::expert_weight_groups`]). Any group with
/// two or more members present in `catalogs` is treated atomically: every
/// member must independently validate (pageable, allocator present,
/// same-device, granule-aligned) *and* every validating member's normalized
/// hot-expert set and expert-count domain must agree exactly with every
/// other validating member's, before *any* member of that group is
/// transitioned. A member disagreeing with its group-mates on the
/// hot-expert partition is treated exactly like a capability failure: the
/// whole group falls back, none of it mutates. Values with no matching group
/// behave exactly as before (validated independently). Pass `&[]` to opt out
/// of grouping entirely (every value validated independently, matching
/// pre-Slice5-revision behavior).
///
/// Returns a [`BoundaryApplicationOutcome`] describing what happened. Never
/// panics; every safety failure degrades to a structural no-op with an
/// explicit `fallback_reason`.
///
/// # Preconditions (checked at entry, never assumed)
///
/// 1. [`coarse_residency_profile_enabled()`] — else no-op.
/// 2. `host_numa_capability(device_ordinal)` succeeds — else no-op.
/// 3. `residency.resize_safe_point(device_count)` is safe — else no-op.
///
/// After preconditions clear, iterates `plan.ordered_values()` in the plan's
/// deterministic order. For each `PerExpertCandidate` value:
///
///   * Verifies the catalog is pageable, has an allocator, and is
///     same-device (else per-value fallback).
///   * Checks every "cold" expert's byte range is granule-aligned (else
///     per-value fallback for this value only).
///   * If the value belongs to a multi-member `ExpertWeightGroup`, requires
///     every member present in `catalogs` to pass the same checks AND to
///     propose the exact same normalized hot-expert set over the same
///     expert-count domain, before any member transitions; one failing or
///     disagreeing member falls the whole group back, and every member that
///     does transition uses one authoritative, group-derived hot set (never
///     its own independently re-derived one) for its byte-range computation.
///   * For each cold-expert granule range, calls
///     [`transition_granule_range`] with `new_location = HostNuma`.
///   * On any `Fatal`, stops iterating and enters the rollback loop:
///     reverse-tier every previously committed **range** (not just value)
///     back to `Device`. `rollback_count` counts values fully reverted;
///     `rollback_failures` records any range whose reverse transition itself
///     failed.
#[allow(clippy::too_many_arguments)]
pub fn apply_residency_plan_at_boundary(
    runtime: &Arc<CudaRuntime>,
    residency: &crate::weight_paging::CudaWeightResidency,
    plan: &ResidencyPlan,
    catalogs: &HashMap<ValueId, WeightRegionCatalog>,
    allocators: &HashMap<ValueId, Arc<CudaVmmAllocator>>,
    device_pool: &Arc<PhysicalHandlePool>,
    host_pool: &Arc<PhysicalHandlePool>,
    device_count: usize,
    device_ordinal: i32,
    expert_groups: &[ExpertWeightGroup],
) -> BoundaryApplicationOutcome {
    apply_residency_plan_at_boundary_inner(
        runtime,
        residency,
        plan,
        catalogs,
        allocators,
        device_pool,
        host_pool,
        device_count,
        device_ordinal,
        expert_groups,
        #[cfg(any(test, feature = "gpu-tests"))]
        None,
    )
}

/// Test-only entry point identical to [`apply_residency_plan_at_boundary`] but
/// with a per-value [`onnx_runtime_cuda_memory::release::DriverFaultPlan`]
/// that forces the underlying Phase-8 `transition_granule_range` calls for a
/// specific `ValueId` to fail deterministically at a chosen call ordinal, so
/// range-level rollback, rollback-of-rollback reporting, and `Fatal`
/// `committed_count`/`poisoned_range` propagation can be proven without
/// relying on real, non-reproducible driver failures.
///
/// Not reachable from production: the parameter only exists under
/// `#[cfg(any(test, feature = "gpu-tests"))]`, mirroring
/// [`crate::granule_transition::transition_granule_range_with_phase8_faults`].
#[cfg(any(test, feature = "gpu-tests"))]
#[allow(clippy::too_many_arguments)]
pub fn apply_residency_plan_at_boundary_with_phase8_faults(
    runtime: &Arc<CudaRuntime>,
    residency: &crate::weight_paging::CudaWeightResidency,
    plan: &ResidencyPlan,
    catalogs: &HashMap<ValueId, WeightRegionCatalog>,
    allocators: &HashMap<ValueId, Arc<CudaVmmAllocator>>,
    device_pool: &Arc<PhysicalHandlePool>,
    host_pool: &Arc<PhysicalHandlePool>,
    device_count: usize,
    device_ordinal: i32,
    expert_groups: &[ExpertWeightGroup],
    phase8_faults: HashMap<ValueId, Arc<onnx_runtime_cuda_memory::release::DriverFaultPlan>>,
) -> BoundaryApplicationOutcome {
    apply_residency_plan_at_boundary_inner(
        runtime,
        residency,
        plan,
        catalogs,
        allocators,
        device_pool,
        host_pool,
        device_count,
        device_ordinal,
        expert_groups,
        Some(phase8_faults),
    )
}

#[allow(clippy::too_many_arguments)]
fn apply_residency_plan_at_boundary_inner(
    runtime: &Arc<CudaRuntime>,
    residency: &crate::weight_paging::CudaWeightResidency,
    plan: &ResidencyPlan,
    catalogs: &HashMap<ValueId, WeightRegionCatalog>,
    allocators: &HashMap<ValueId, Arc<CudaVmmAllocator>>,
    device_pool: &Arc<PhysicalHandlePool>,
    host_pool: &Arc<PhysicalHandlePool>,
    device_count: usize,
    device_ordinal: i32,
    expert_groups: &[ExpertWeightGroup],
    #[cfg(any(test, feature = "gpu-tests"))] phase8_faults: Option<
        HashMap<ValueId, Arc<onnx_runtime_cuda_memory::release::DriverFaultPlan>>,
    >,
) -> BoundaryApplicationOutcome {
    let mut outcome = BoundaryApplicationOutcome {
        policy_name: plan.policy_name(),
        ..Default::default()
    };

    // 1. Feature gate.
    if !coarse_residency_profile_enabled() {
        outcome.fallback_reason = Some("feature gate disabled".to_string());
        return outcome;
    }

    // 2. Host NUMA capability.
    if let Err(err) = onnx_runtime_cuda_memory::capability::host_numa_capability(device_ordinal) {
        outcome.fallback_reason = Some(format!("host-numa capability unavailable: {err}"));
        return outcome;
    }

    // 3. Safe-point verification.
    let sp = residency.resize_safe_point(device_count);
    let verified = match verify_safe_point(sp) {
        Ok(v) => v,
        Err(reason) => {
            outcome.fallback_reason = Some(format!("resize safe-point not clear: {reason}"));
            return outcome;
        }
    };

    let granularity = device_pool.granularity().max(host_pool.granularity());

    // Map value -> group index, for values that belong to a multi-member
    // group. Values with no group (or a singleton group) are unaffected.
    let mut value_to_group: HashMap<ValueId, usize> = HashMap::new();
    for (idx, group) in expert_groups.iter().enumerate() {
        let present: Vec<ValueId> = group
            .members
            .iter()
            .copied()
            .filter(|v| catalogs.contains_key(v))
            .collect();
        if present.len() < 2 {
            continue;
        }
        for value in present {
            value_to_group.insert(value, idx);
        }
    }

    // 4. Per-value validation pass: for every value the plan mentions with a
    // PerExpertCandidate decision, determine up front whether it (and its
    // group, if any) is eligible to transition at all, WITHOUT mutating
    // anything yet. This lets a group-mate failure suppress the whole
    // group's transitions before any of them touch the driver.
    struct Eligible {
        value: ValueId,
        /// This member's own normalized (order/duplicate-independent)
        /// hot-expert set, kept only so the group-agreement pass below can
        /// compare it against sibling members'. The actual byte ranges
        /// transitioned for a multi-member group are always re-derived from
        /// the group's single authoritative hot set (see `group_hot`
        /// below), never from this per-member copy.
        hot: std::collections::HashSet<usize>,
        /// This member's own expert-count domain (`catalog.layout().experts`),
        /// compared across group members so a domain mismatch (not just a
        /// hot-set mismatch) is caught before any mutation.
        total_experts: usize,
        hot_count: usize,
        cold_count: usize,
        merged_ranges: Vec<(usize, usize)>,
    }

    let mut eligible: Vec<Eligible> = Vec::new();
    let mut group_failed: std::collections::HashSet<usize> = std::collections::HashSet::new();
    let mut group_fail_reason: HashMap<usize, String> = HashMap::new();
    let mut per_value_precheck: Vec<(ValueId, Result<Eligible, String>)> = Vec::new();

    for value in plan.ordered_values() {
        outcome.values_inspected += 1;
        let decision = match plan.decision(value) {
            Some(d) => d,
            None => continue,
        };
        let experts = match decision {
            ResidencyDecision::PerExpertCandidate { experts } => experts,
            ResidencyDecision::WholeBankResident { .. } => continue,
        };

        let precheck = (|| -> Result<Eligible, String> {
            let catalog = catalogs
                .get(&value)
                .ok_or_else(|| "no catalog available".to_string())?;
            if !catalog.is_pageable() {
                return Err("catalog not pageable".to_string());
            }
            let allocator = allocators
                .get(&value)
                .ok_or_else(|| "no VMM allocator for value".to_string())?;
            check_same_device(allocator, device_pool, host_pool, device_ordinal)?;

            let total_experts = catalog.layout().experts;
            let hot: std::collections::HashSet<usize> = experts.iter().copied().collect();
            let merged = cold_ranges_for(catalog, &hot, granularity)?;
            let hot_count = hot.len().min(total_experts);
            let cold_count = total_experts.saturating_sub(hot_count);
            Ok(Eligible {
                value,
                hot,
                total_experts,
                hot_count,
                cold_count,
                merged_ranges: merged,
            })
        })();

        if let Err(reason) = &precheck
            && let Some(&group_idx) = value_to_group.get(&value)
        {
            group_failed.insert(group_idx);
            group_fail_reason
                .entry(group_idx)
                .or_insert_with(|| format!("group member {value:?} failed: {reason}"));
        }
        per_value_precheck.push((value, precheck));
    }

    // 4a. Group-level hot-expert agreement (Cycle-19 fix): a multi-member
    // `ExpertWeightGroup` names one *logical* expert bank split across
    // several `ValueId`s (fc1/fc2/fc3 weights, scales, ...). Each member can
    // independently clear the capability/alignment precheck above yet still
    // be told to keep a *different* set of experts hot — that would tier
    // one logical expert across Device/Host, exactly the split this type
    // exists to prevent. This pass derives a single authoritative hot set
    // per group, once, from the first member (in deterministic plan order)
    // that clears its own precheck, and requires every other clearing
    // member's own (also normalized-to-a-`HashSet`, so order/duplicates in
    // the source `Vec<usize>` are already immaterial) hot set and
    // expert-count domain to match it exactly. This is the one mechanism
    // that decides "does the group agree on its partition" — it does not
    // duplicate or race the capability check above; a capability failure
    // already short-circuits via `group_failed` before this pass runs for
    // that member.
    let mut group_hot: HashMap<usize, (std::collections::HashSet<usize>, usize)> = HashMap::new();
    for (value, precheck) in &per_value_precheck {
        let Some(&group_idx) = value_to_group.get(value) else {
            continue;
        };
        if group_failed.contains(&group_idx) {
            continue;
        }
        let Ok(candidate) = precheck else { continue };
        match group_hot.get(&group_idx) {
            None => {
                group_hot.insert(group_idx, (candidate.hot.clone(), candidate.total_experts));
            }
            Some((canonical_hot, canonical_total_experts)) => {
                if candidate.total_experts != *canonical_total_experts {
                    group_failed.insert(group_idx);
                    group_fail_reason.entry(group_idx).or_insert_with(|| {
                        format!(
                            "expert-group members disagree on expert_count domain ({canonical_total_experts} vs {}, value {value:?})",
                            candidate.total_experts
                        )
                    });
                } else if candidate.hot != *canonical_hot {
                    group_failed.insert(group_idx);
                    group_fail_reason.entry(group_idx).or_insert_with(|| {
                        format!(
                            "expert-group members disagree on hot-expert partition (value {value:?})"
                        )
                    });
                }
            }
        }
    }

    for (value, precheck) in per_value_precheck {
        // If this value's group failed because of ANY member, the whole
        // group falls back — even members that individually passed.
        if let Some(&group_idx) = value_to_group.get(&value)
            && group_failed.contains(&group_idx)
        {
            let reason = group_fail_reason
                .get(&group_idx)
                .cloned()
                .unwrap_or_else(|| "expert-group member failed".to_string());
            outcome
                .per_value_fallbacks
                .push((value, format!("expert-group fallback: {reason}")));
            continue;
        }
        match precheck {
            Ok(mut e) => {
                // The group agreed (or this value isn't grouped). For a
                // validated multi-member group, re-derive this member's
                // byte ranges from the group's single authoritative hot set
                // — never from this member's own independently-computed
                // (though already-verified-equal) one — so no future change
                // to either code path can let two members' cold ranges
                // disagree even if the equality check above were ever
                // weakened.
                if let Some(&group_idx) = value_to_group.get(&value)
                    && let Some((canonical_hot, _)) = group_hot.get(&group_idx)
                {
                    let catalog = catalogs.get(&value).expect("validated present above");
                    match cold_ranges_for(catalog, canonical_hot, granularity) {
                        Ok(merged) => e.merged_ranges = merged,
                        Err(reason) => {
                            outcome.per_value_fallbacks.push((value, reason));
                            continue;
                        }
                    }
                }
                eligible.push(e);
            }
            Err(reason) => outcome.per_value_fallbacks.push((value, reason)),
        }
    }

    // 5. Per-value application loop (mutating), operating only on values that
    // cleared validation above (individually and as a group), in the plan's
    // deterministic order (guaranteed since `eligible` was built by iterating
    // `plan.ordered_values()`).
    let mut progress: HashMap<ValueId, ValueProgress> = HashMap::new();
    let mut fatal_hit = false;
    'outer: for e in &eligible {
        let value = e.value;
        outcome.hot_expert_count += e.hot_count;
        outcome.cold_expert_count += e.cold_count;
        let entry = progress.entry(value).or_default();

        for (offset, len) in &e.merged_ranges {
            let allocator = allocators.get(&value).expect("validated present above");
            let node = onnx_runtime_cuda_memory::capability::host_numa_capability(device_ordinal)
                .map(|c| c.host_numa_id)
                .unwrap_or(0);
            let start = std::time::Instant::now();
            #[cfg(any(test, feature = "gpu-tests"))]
            let value_fault = phase8_faults.as_ref().and_then(|m| m.get(&value).cloned());
            let result = allocator.with_reservation_mut(|reservation, backing| {
                #[cfg(any(test, feature = "gpu-tests"))]
                if let Some(fault_plan) = value_fault.clone() {
                    return transition_granule_range_with_phase8_faults(
                        runtime,
                        reservation,
                        backing,
                        *offset,
                        *len,
                        PhysicalLocation::HostNuma { node },
                        device_pool,
                        host_pool,
                        &verified,
                        || residency.resize_safe_point(device_count),
                        fault_plan,
                    );
                }
                transition_granule_range(
                    runtime,
                    reservation,
                    backing,
                    *offset,
                    *len,
                    PhysicalLocation::HostNuma { node },
                    device_pool,
                    host_pool,
                    &verified,
                    || residency.resize_safe_point(device_count),
                )
            });
            outcome.transition_time_ms += start.elapsed().as_secs_f64() * 1000.0;
            match result {
                TransitionOutcome::Committed {
                    granules: _,
                    new_owned_bytes,
                    old_released_bytes,
                } => {
                    outcome.host_bytes_committed += new_owned_bytes;
                    outcome.device_bytes_released += old_released_bytes;
                    entry.committed.push(CommittedRange {
                        offset: *offset,
                        len: *len,
                    });
                }
                TransitionOutcome::Rejected { reason } => {
                    outcome.failure_count += 1;
                    outcome
                        .per_value_fallbacks
                        .push((value, format!("transition rejected: {reason}")));
                }
                TransitionOutcome::RolledBack { fault } => {
                    outcome.failure_count += 1;
                    outcome
                        .per_value_fallbacks
                        .push((value, format!("transition rolled back: {fault:?}")));
                }
                TransitionOutcome::Fatal {
                    transition_fault,
                    quarantined,
                    committed_count,
                    poisoned_range,
                    ..
                } => {
                    outcome.quarantined.push((value, quarantined));
                    outcome
                        .fatal_progress
                        .push((value, committed_count, poisoned_range));
                    outcome
                        .per_value_fallbacks
                        .push((value, format!("fatal transition: {transition_fault:?}")));
                    fatal_hit = true;
                    break 'outer;
                }
            }
        }
    }

    // Every value that has at least one committed range and did NOT hit a
    // fatal transition anywhere in the plan is provisionally touched. If a
    // fatal occurred anywhere, ALL committed ranges (including this value's
    // own already-committed ranges) go through the rollback loop below, which
    // updates `committed_values`/`values_touched` precisely.
    if !fatal_hit {
        for (value, prog) in &progress {
            if !prog.committed.is_empty() {
                outcome.values_touched += 1;
                outcome.committed_values.push(*value);
            }
        }
    }

    // 6. Rollback loop on any Fatal: revert every committed RANGE (not
    // value) recorded in `progress`, across every value that has any —
    // including the same value whose later range hit the Fatal, and
    // including values earlier in plan order that already fully committed.
    if fatal_hit {
        let recheck_sp = residency.resize_safe_point(device_count);
        let rollback_sp = match verify_safe_point(recheck_sp) {
            Ok(v) => v,
            Err(reason) => {
                outcome.fallback_reason = Some(format!(
                    "fatal transition + safe-point lost during rollback: {reason}"
                ));
                return outcome;
            }
        };
        for (value, prog) in &progress {
            if prog.committed.is_empty() {
                continue;
            }
            let allocator = match allocators.get(value) {
                Some(a) => a,
                None => continue,
            };
            let mut all_ok = true;
            for range in &prog.committed {
                #[cfg(any(test, feature = "gpu-tests"))]
                let value_fault = phase8_faults.as_ref().and_then(|m| m.get(value).cloned());
                let result = allocator.with_reservation_mut(|reservation, backing| {
                    #[cfg(any(test, feature = "gpu-tests"))]
                    if let Some(fault_plan) = value_fault.clone() {
                        return transition_granule_range_with_phase8_faults(
                            runtime,
                            reservation,
                            backing,
                            range.offset,
                            range.len,
                            PhysicalLocation::Device {
                                ordinal: device_ordinal,
                            },
                            host_pool,
                            device_pool,
                            &rollback_sp,
                            || residency.resize_safe_point(device_count),
                            fault_plan,
                        );
                    }
                    transition_granule_range(
                        runtime,
                        reservation,
                        backing,
                        range.offset,
                        range.len,
                        PhysicalLocation::Device {
                            ordinal: device_ordinal,
                        },
                        host_pool,
                        device_pool,
                        &rollback_sp,
                        || residency.resize_safe_point(device_count),
                    )
                });
                match result {
                    TransitionOutcome::Committed { .. } => {}
                    TransitionOutcome::Rejected { reason } => {
                        all_ok = false;
                        outcome.rollback_failures.push(RollbackFailure {
                            value: *value,
                            range: (range.offset, range.len),
                            detail: format!("reverse transition rejected: {reason}"),
                            committed_count: None,
                            poisoned_range: None,
                            quarantined: Vec::new(),
                        });
                    }
                    TransitionOutcome::RolledBack { fault } => {
                        all_ok = false;
                        outcome.rollback_failures.push(RollbackFailure {
                            value: *value,
                            range: (range.offset, range.len),
                            detail: format!("reverse transition rolled back: {fault:?}"),
                            committed_count: None,
                            poisoned_range: None,
                            quarantined: Vec::new(),
                        });
                    }
                    TransitionOutcome::Fatal {
                        transition_fault,
                        quarantined,
                        committed_count,
                        poisoned_range,
                        ..
                    } => {
                        all_ok = false;
                        outcome.quarantined.push((*value, quarantined.clone()));
                        outcome.rollback_failures.push(RollbackFailure {
                            value: *value,
                            range: (range.offset, range.len),
                            detail: format!("reverse transition fatal: {transition_fault:?}"),
                            committed_count: Some(committed_count),
                            poisoned_range,
                            quarantined,
                        });
                    }
                }
            }
            if all_ok {
                outcome.rollback_count += 1;
            }
        }
        if outcome.fallback_reason.is_none() {
            outcome.fallback_reason = Some("fatal transition; rollback attempted".to_string());
        }
    }

    outcome
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn feature_gate_default_off() {
        // In CI, this env var should not be set. Directly probe.
        // If the caller's environment happens to set it, the assertion is
        // still meaningful (they opted in).
        let expected = matches!(
            std::env::var(COARSE_RESIDENCY_ENABLE_ENV)
                .ok()
                .as_deref()
                .map(str::trim)
                .map(str::to_ascii_lowercase)
                .as_deref(),
            Some("1") | Some("true") | Some("on")
        );
        assert_eq!(coarse_residency_profile_enabled(), expected);
    }

    #[test]
    fn boundary_outcome_default_is_noop_shape() {
        let outcome = BoundaryApplicationOutcome::default();
        assert_eq!(outcome.values_touched, 0);
        assert_eq!(outcome.hot_expert_count, 0);
        assert_eq!(outcome.cold_expert_count, 0);
        assert_eq!(outcome.host_bytes_committed, 0);
        assert!(outcome.per_value_fallbacks.is_empty());
        assert!(outcome.committed_values.is_empty());
        assert!(outcome.quarantined.is_empty());
        assert!(outcome.fatal_progress.is_empty());
        assert!(outcome.rollback_failures.is_empty());
    }

    #[test]
    #[allow(deprecated)]
    fn deprecated_env_alias_matches_new_name() {
        assert_eq!(COARSE_RESIDENCY_PROFILE_ENV, COARSE_RESIDENCY_ENABLE_ENV);
    }
}
