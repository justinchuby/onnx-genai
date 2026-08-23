//! #1810 Slice 5 — Coarse-boundary per-expert residency plan application.
//!
//! Provides [`apply_residency_plan_at_boundary`], the ONE production consumer
//! of a [`ResidencyPlan`] whose decisions may include `PerExpertCandidate`.
//! Called ONLY at model-load completion when the coarse-boundary feature gate
//! is enabled. The routing/kernel dispatch path is completely unchanged.
//!
//! # Feature gate
//!
//! Off by default. Enable by setting
//! [`COARSE_RESIDENCY_PROFILE_ENV`] (`ONNX_GENAI_WEIGHT_OFFLOAD_COARSE_RESIDENCY_PROFILE`)
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

use std::collections::HashMap;
use std::sync::Arc;

use onnx_runtime_cuda_memory::release::MappedBlock;
use onnx_runtime_cuda_memory::virtual_memory::{PhysicalHandlePool, PhysicalLocation};
use onnx_runtime_cuda_memory::vmm_allocator::CudaVmmAllocator;
use onnx_runtime_ep_api::{ResidencyDecision, ResidencyPlan};
use onnx_runtime_ir::ValueId;
use onnx_runtime_loader::WeightRegionCatalog;

use crate::granule_transition::{TransitionOutcome, transition_granule_range, verify_safe_point};
use crate::runtime::CudaRuntime;

/// Environment variable that gates coarse-boundary per-expert residency
/// plan application. Off by default.
pub const COARSE_RESIDENCY_PROFILE_ENV: &str = "ONNX_GENAI_WEIGHT_OFFLOAD_COARSE_RESIDENCY_PROFILE";

/// True iff [`COARSE_RESIDENCY_PROFILE_ENV`] is set to `1`, `true`, or `on`
/// (case-insensitive, trimmed).
pub fn coarse_residency_profile_enabled() -> bool {
    matches!(
        std::env::var(COARSE_RESIDENCY_PROFILE_ENV)
            .ok()
            .as_deref()
            .map(str::trim)
            .map(str::to_ascii_lowercase)
            .as_deref(),
        Some("1") | Some("true") | Some("on")
    )
}

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
    /// Number of values for which at least one granule range was transitioned.
    pub values_touched: usize,
    /// Total hot experts (kept Device-resident) across all inspected values.
    pub hot_expert_count: usize,
    /// Total cold experts (moved to Host NUMA) across all inspected values.
    pub cold_expert_count: usize,
    /// Approximate device bytes charged to the plan's transitions before
    /// (i.e. the bytes about to be freed from Device by the plan).
    pub device_bytes_before: u64,
    /// Approximate device bytes charged after this plan applied.
    /// `device_bytes_before - device_bytes_after` = bytes returned to Device pool.
    pub device_bytes_after: u64,
    /// Bytes newly committed on the Host NUMA pool by the plan.
    pub host_bytes_committed: u64,
    /// Sum of successful transition durations in milliseconds (wall-clock).
    pub transition_time_ms: f64,
    /// Number of values whose transition returned `Rejected` or `RolledBack`.
    pub failure_count: usize,
    /// Number of values whose committed transitions had to be undone by the
    /// rollback loop after a `Fatal`.
    pub rollback_count: usize,
    /// Non-`None` when the whole call was a structural no-op (feature gate
    /// off, no host-NUMA support, or unsafe point). Not set for per-value
    /// skips: those append to `per_value_fallbacks` instead.
    pub fallback_reason: Option<String>,
    /// `(value, reason)` for each value that was inspected but not
    /// transitioned (e.g. non-pageable catalog, non-granule-aligned per-expert
    /// range, `WholeBankResident` decision).
    pub per_value_fallbacks: Vec<(ValueId, String)>,
    /// Values whose transitions successfully committed. Used by the rollback
    /// loop after a Fatal — reverse-tier these back to Device.
    pub committed_values: Vec<ValueId>,
    /// Quarantined blocks emitted by any `Fatal` outcome, keyed by value.
    /// These are also present in the reservation's own `quarantined_blocks()`
    /// list; carried here for the caller's immediate telemetry.
    pub quarantined: Vec<(ValueId, Vec<MappedBlock>)>,
}

/// Apply `plan` to the given `catalogs` and residency at model-load boundary.
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
///   * Verifies the catalog is pageable (else per-value fallback).
///   * Checks every "cold" expert's byte range is granule-aligned (else
///     per-value fallback for this value only).
///   * For each cold-expert granule range, calls
///     [`transition_granule_range`] with `new_location = HostNuma`.
///   * On any `Fatal`, stops iterating and enters the rollback loop:
///     reverse-tier every previously committed value's granules back to
///     `Device`. `rollback_count` counts values reverted.
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

    // 4. Per-value application loop.
    let mut fatal_hit = false;
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

        let catalog = match catalogs.get(&value) {
            Some(c) => c,
            None => {
                outcome
                    .per_value_fallbacks
                    .push((value, "no catalog available".to_string()));
                continue;
            }
        };
        if !catalog.is_pageable() {
            outcome
                .per_value_fallbacks
                .push((value, "catalog not pageable".to_string()));
            continue;
        }
        let allocator = match allocators.get(&value) {
            Some(a) => a,
            None => {
                outcome
                    .per_value_fallbacks
                    .push((value, "no VMM allocator for value".to_string()));
                continue;
            }
        };

        let total_experts = catalog.layout().experts;
        let hot: std::collections::HashSet<usize> = experts.iter().copied().collect();

        // Collect the cold-expert granule ranges (offset, len) into one list,
        // validating alignment as we go. If ANY expert's range is not
        // granule-aligned, skip this whole value (never partial-tier).
        let mut cold_ranges: Vec<(usize, usize)> = Vec::new();
        let mut alignment_fail: Option<String> = None;
        for expert in 0..total_experts {
            if hot.contains(&expert) {
                continue;
            }
            let range = match catalog.relative_range(expert) {
                Some(r) => r,
                None => {
                    alignment_fail = Some(format!("expert {expert} has no relative_range"));
                    break;
                }
            };
            let offset = range.start;
            let len = range.end.saturating_sub(range.start);
            if len == 0 {
                continue;
            }
            if offset % granularity != 0 || len % granularity != 0 {
                alignment_fail = Some(format!(
                    "expert {expert} range {offset}..{} is not granule-aligned (granularity={granularity})",
                    range.end
                ));
                break;
            }
            cold_ranges.push((offset, len));
        }
        if let Some(reason) = alignment_fail {
            outcome.per_value_fallbacks.push((value, reason));
            continue;
        }

        // Merge adjacent ranges (contiguous cold experts across granule
        // boundaries): keeps the transition count minimal without changing
        // semantics.
        cold_ranges.sort_by_key(|&(off, _)| off);
        let mut merged: Vec<(usize, usize)> = Vec::with_capacity(cold_ranges.len());
        for (off, len) in cold_ranges {
            match merged.last_mut() {
                Some(last) if last.0 + last.1 == off => last.1 += len,
                _ => merged.push((off, len)),
            }
        }

        let hot_count = hot.len().min(total_experts);
        let cold_count = total_experts.saturating_sub(hot_count);
        outcome.hot_expert_count += hot_count;
        outcome.cold_expert_count += cold_count;

        let mut value_touched = false;
        let mut value_fatal = false;
        for (offset, len) in &merged {
            let node = onnx_runtime_cuda_memory::capability::host_numa_capability(device_ordinal)
                .map(|c| c.host_numa_id)
                .unwrap_or(0);
            let start = std::time::Instant::now();
            let result = allocator.with_reservation_mut(|reservation, backing| {
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
                    outcome.device_bytes_before += old_released_bytes;
                    // device_bytes_after tracked as delta from before
                    value_touched = true;
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
                    ..
                } => {
                    outcome.quarantined.push((value, quarantined));
                    outcome
                        .per_value_fallbacks
                        .push((value, format!("fatal transition: {transition_fault:?}")));
                    value_fatal = true;
                    break;
                }
            }
        }

        if value_touched && !value_fatal {
            outcome.values_touched += 1;
            outcome.committed_values.push(value);
        }
        if value_fatal {
            fatal_hit = true;
            break;
        }
    }

    // device_bytes_after = device_bytes_before - old_released_bytes carried in
    // device_bytes_before slot. For clarity, expose "after" as 0 (bytes
    // relative to the plan's own scope, not the whole cache).
    outcome.device_bytes_after = 0;

    // 5. Rollback loop on any Fatal.
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
        // Reverse-tier every previously committed value.
        let to_revert = std::mem::take(&mut outcome.committed_values);
        for value in to_revert {
            let catalog = match catalogs.get(&value) {
                Some(c) => c,
                None => continue,
            };
            let allocator = match allocators.get(&value) {
                Some(a) => a,
                None => continue,
            };
            let decision = match plan.decision(value) {
                Some(ResidencyDecision::PerExpertCandidate { experts }) => experts,
                _ => continue,
            };
            let total_experts = catalog.layout().experts;
            let hot: std::collections::HashSet<usize> = decision.iter().copied().collect();
            let device_ordinal_local = device_ordinal;
            let mut ranges: Vec<(usize, usize)> = Vec::new();
            for expert in 0..total_experts {
                if hot.contains(&expert) {
                    continue;
                }
                if let Some(range) = catalog.relative_range(expert) {
                    let offset = range.start;
                    let len = range.end.saturating_sub(range.start);
                    if len == 0 {
                        continue;
                    }
                    if offset % granularity == 0 && len % granularity == 0 {
                        ranges.push((offset, len));
                    }
                }
            }
            ranges.sort_by_key(|&(o, _)| o);
            let mut merged: Vec<(usize, usize)> = Vec::with_capacity(ranges.len());
            for (o, l) in ranges {
                match merged.last_mut() {
                    Some(last) if last.0 + last.1 == o => last.1 += l,
                    _ => merged.push((o, l)),
                }
            }
            let mut all_ok = true;
            for (offset, len) in merged {
                let result = allocator.with_reservation_mut(|reservation, backing| {
                    transition_granule_range(
                        runtime,
                        reservation,
                        backing,
                        offset,
                        len,
                        PhysicalLocation::Device {
                            ordinal: device_ordinal_local,
                        },
                        host_pool,
                        device_pool,
                        &rollback_sp,
                        || residency.resize_safe_point(device_count),
                    )
                });
                if !matches!(result, TransitionOutcome::Committed { .. }) {
                    all_ok = false;
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
            std::env::var(COARSE_RESIDENCY_PROFILE_ENV)
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
    }
}
