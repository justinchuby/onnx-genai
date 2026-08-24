//! #1810 Slice 7B — boundary-time route-telemetry *consumer*.
//!
//! This is the smallest production caller that closes the loop opened by the
//! merged Slice-6/7A producer (`kernels::expert_route_telemetry`, PR #1922,
//! `e1ec495ee`) and the merged Slice-4/5 coarse-boundary residency lifecycle
//! (`granule_transition` + `coarse_residency`, PR #1854). The Slice-6 design
//! §8 names it exactly:
//!
//! > expose a boundary-time consumer that produces a per-expert desired-set,
//! > and feed that set to the **existing** Slice 4/5 coarse-boundary plan
//! > application (`coarse_residency.rs`) as its policy input — with **no** new
//! > allocator, **no** id→slot rewrite, and every mapping change still owned
//! > by PMM/VMM.
//!
//! [`consume_route_window_at_boundary`] is that consumer. Given one *already
//! completed* coarse-boundary telemetry window (a host [`TelemetrySnapshot`]
//! the caller obtained through the producer's existing stream-completion
//! authority — `QMoEKernel::route_telemetry_snapshot` / the `dtoh` that
//! self-synchronizes), it:
//!
//!   1. validates the window with the producer's own
//!      [`consume_and_validate`] (fail-closed on poison / overflow / stale
//!      epoch / foreign request / foreign device),
//!   2. turns the routed-expert union into a desired *hot set* and asks the
//!      existing, already-validated [`StaticProfileResidencyPolicy`] to shape
//!      a [`ResidencyPlan`] over the bank's catalogs (the "record → desired
//!      set" role the design calls `RouteObserverPolicy`; reused rather than
//!      duplicated so there is exactly one validated policy that emits
//!      `PerExpertCandidate`), and
//!   3. hands that plan to the existing
//!      [`CudaWeightResidency::apply_coarse_residency_plan`], which is the
//!      **sole** authority that maps/unmaps/accounts/quarantines/rolls back
//!      through PMM/VMM.
//!
//! # What this module never does (by construction)
//!
//! * It **allocates nothing**, opens no stream, owns no VA, and copies no
//!   device bytes. The snapshot is copied by the producer; the transition is
//!   executed by `coarse_residency`. This module is pure host glue between two
//!   existing authorities.
//! * It performs **no remap during capture/replay**: before consuming it
//!   re-reads the *existing* [`CudaWeightResidency::resize_safe_point`] and
//!   fails closed with [`RouteWindowConsumeOutcome::RejectedNotSafeBoundary`]
//!   if a graph is capturing/replaying, an admission is in flight, a deferred
//!   release has not settled, execution is multi-device, or a routed-residency
//!   guard is live. It is a **coarse-boundary** operation only — never a
//!   per-token remap.
//! * It adds **no new host sync** in steady state. The only synchronizing work
//!   is the producer's snapshot (already taken) and, when a plan is applied,
//!   the existing transition primitive's drain — both pre-existing authorities.
//! * It has **no silent fallback**: every path that does not tier returns a
//!   variant carrying the exact reason.
//!
//! # Default off / byte-identical
//!
//! Gated by the existing [`COARSE_RESIDENCY_ENABLE_ENV`]
//! (`coarse_residency_profile_enabled()`). When off — the shipped default —
//! this returns [`RouteWindowConsumeOutcome::Disabled`] before reading the
//! snapshot or touching any allocator, so ordinary inference (telemetry
//! disarmed *and* this gate off: two independent default-off switches) is
//! byte-identical.
//!
//! # Window ordering the caller owns
//!
//! This consumer covers snapshot → validate → plan → apply. Advancing the
//! window for the next accumulation interval stays with the producer's
//! guarded [`QMoEKernel::reset_route_telemetry_boundary`], which is itself
//! rejected during capture. The lawful boundary sequence is therefore:
//! `snapshot` (stream-completion authority) → `consume_route_window_at_boundary`
//! → `reset_route_telemetry_boundary` (before any new work), so a consumed
//! window is never re-consumed and the next window starts empty.
//!
//! # Production status (honest)
//!
//! Like `coarse_residency::apply_residency_plan_at_boundary` when it shipped
//! (Slice 5), this consumer has **no live decode-loop call site yet**: wiring
//! it into a running session's request boundary is the next slice. It ships
//! here as the production seam — reachable, default-off, and proven by the
//! GPU tests in `tests/route_residency_consume_gpu.rs` — so the telemetry the
//! producer already accumulates has a real, tested consumer to drive the
//! #1854 transition.
//!
//! [`TelemetrySnapshot`]: crate::kernels::expert_route_telemetry::TelemetrySnapshot
//! [`consume_and_validate`]: crate::kernels::expert_route_telemetry::consume_and_validate
//! [`StaticProfileResidencyPolicy`]: onnx_runtime_ep_api::StaticProfileResidencyPolicy
//! [`ResidencyPlan`]: onnx_runtime_ep_api::ResidencyPlan
//! [`CudaWeightResidency::apply_coarse_residency_plan`]: crate::weight_paging::CudaWeightResidency::apply_coarse_residency_plan
//! [`CudaWeightResidency::resize_safe_point`]: crate::weight_paging::CudaWeightResidency::resize_safe_point
//! [`QMoEKernel::route_telemetry_snapshot`]: crate::kernels::qmoe::QMoEKernel::route_telemetry_snapshot
//! [`QMoEKernel::reset_route_telemetry_boundary`]: crate::kernels::qmoe::QMoEKernel::reset_route_telemetry_boundary
//! [`COARSE_RESIDENCY_ENABLE_ENV`]: crate::coarse_residency::COARSE_RESIDENCY_ENABLE_ENV

use std::collections::HashMap;
use std::sync::Arc;

use onnx_runtime_cuda_memory::virtual_memory::PhysicalHandlePool;
use onnx_runtime_cuda_memory::vmm_allocator::CudaVmmAllocator;
use onnx_runtime_ep_api::{
    ExpertWeightGroup, LazyWeightBoundary, ResidencyPlan, StaticProfileResidencyPolicy,
    plan_residency,
};
use onnx_runtime_ir::ValueId;
use onnx_runtime_loader::WeightRegionCatalog;

use crate::coarse_residency::{BoundaryApplicationOutcome, coarse_residency_profile_enabled};
use crate::kernels::expert_route_telemetry::{
    RouteDecision, TelemetrySnapshot, consume_and_validate,
};
use crate::weight_paging::CudaWeightResidency;

/// Outcome of consuming one completed coarse-boundary route-telemetry window.
///
/// Exactly one variant is returned; each non-`Applied` variant carries the
/// precise reason nothing was tiered (design-discipline "carry the reason" —
/// there is no silent fallback).
#[derive(Debug)]
pub enum RouteWindowConsumeOutcome {
    /// The consumer is disabled ([`coarse_residency_profile_enabled`] is
    /// false, the shipped default). Pure no-op: the snapshot was not even
    /// read, no plan was built, and no allocator was touched — ordinary
    /// inference is byte-identical.
    Disabled,
    /// The current boundary is not safe to consume/apply at. Fail-closed:
    /// nothing tiered. `reason` is the exact
    /// [`onnx_runtime_ep_api::ResizeSafePoint::blocking_reason`] (capture/
    /// replay in flight, admission in flight, unsettled deferred release,
    /// multi-device, or a live routed-residency guard).
    RejectedNotSafeBoundary { reason: &'static str },
    /// The window was structurally valid to read but its decision was
    /// whole-bank — fail-closed on poison / overflow / stale epoch / foreign
    /// request / foreign device, or because it recorded no in-range experts.
    /// Nothing tiered; `reason` records why.
    WholeBank { reason: String },
    /// The window validated to a routed hot-set and the derived per-expert
    /// plan was applied through the existing #1854 lifecycle. `outcome`
    /// carries the PMM/VMM-authoritative result verbatim (values touched,
    /// bytes moved, rollbacks, quarantined blocks, per-value fallbacks, ...).
    Applied {
        /// The routed experts kept device-resident (ascending). The bank's
        /// other experts are the plan's cold set, tiered to host.
        routed_experts: Vec<usize>,
        /// The window's fixed epoch, echoed for the caller's boundary log.
        epoch: u32,
        /// The window's bounded in-range route count.
        count: u32,
        /// The exact boundary application outcome from `coarse_residency`.
        /// Boxed so this rich variant does not bloat the whole enum.
        outcome: Box<BoundaryApplicationOutcome>,
    },
}

/// Result of the gate/safe-point/validate/plan pre-pass shared by the
/// production consumer and its fault-injection test variant.
enum Prepared {
    /// A terminal outcome was reached before any plan could be applied.
    Early(RouteWindowConsumeOutcome),
    /// The window validated to a non-empty hot-set; the plan is ready to hand
    /// to a `coarse_residency` apply entry point.
    Ready {
        plan: ResidencyPlan,
        routed_experts: Vec<usize>,
        epoch: u32,
        count: u32,
    },
}

/// Gate, verify the safe boundary, validate the window, and (on a trustworthy
/// non-empty hot-set) shape the residency plan — without mutating anything.
#[allow(clippy::too_many_arguments)]
fn prepare_route_window(
    residency: &CudaWeightResidency,
    snapshot: &TelemetrySnapshot,
    expected_epoch: u32,
    expected_request: u32,
    expected_device: u32,
    bank_values: &[ValueId],
    boundary: LazyWeightBoundary,
    catalogs: &HashMap<ValueId, WeightRegionCatalog>,
    device_count: usize,
) -> Prepared {
    // 1. Default-off gate. Read before the snapshot so a disarmed/disabled
    //    build never inspects telemetry or touches an allocator.
    if !coarse_residency_profile_enabled() {
        return Prepared::Early(RouteWindowConsumeOutcome::Disabled);
    }

    // 2. Proven safe boundary. Reuse the existing residency authority rather
    //    than duplicating capture/admission/guard tracking. This is the "no
    //    remap during capture/replay" and "boundary-only, never per-token"
    //    gate; `apply_coarse_residency_plan` re-verifies it immediately before
    //    the atomic switch, but failing closed *here* keeps the honest reason.
    if let Some(reason) = residency.resize_safe_point(device_count).blocking_reason() {
        return Prepared::Early(RouteWindowConsumeOutcome::RejectedNotSafeBoundary { reason });
    }

    // 3. Validate the completed window with the producer's own consumer
    //    reference (fail-closed identity/epoch/poison/overflow contract).
    match consume_and_validate(
        &snapshot.header,
        &snapshot.bitmap,
        expected_epoch,
        expected_request,
        expected_device,
    ) {
        RouteDecision::WholeBank(reason) => {
            return Prepared::Early(RouteWindowConsumeOutcome::WholeBank { reason });
        }
        RouteDecision::HotSet(_) => {}
    }

    // 4. Decode the routed-expert union (the desired hot-set to keep resident).
    let routed_experts = snapshot.routed_experts();
    if routed_experts.is_empty() {
        // A window with a clean identity but an empty routed set would tier the
        // whole bank to host right before the model uses it. Fail closed.
        return Prepared::Early(RouteWindowConsumeOutcome::WholeBank {
            reason: "route window recorded no in-range experts; nothing to keep resident".into(),
        });
    }

    // 5. Shape the plan. Every present bank member is given the *same*
    //    authoritative hot-set, which is exactly what the #1854 expert-group
    //    agreement pass requires for an atomic cross-tensor transition. The
    //    reused `StaticProfileResidencyPolicy` + `plan_residency` validate each
    //    decision (nonpageable / out-of-range / duplicate → whole-bank).
    let profile: HashMap<ValueId, Vec<usize>> = bank_values
        .iter()
        .filter(|value| catalogs.contains_key(value))
        .map(|value| (*value, routed_experts.clone()))
        .collect();
    let policy = StaticProfileResidencyPolicy::new(profile);
    let candidates: Vec<(ValueId, LazyWeightBoundary, &WeightRegionCatalog)> = bank_values
        .iter()
        .filter_map(|value| {
            catalogs
                .get(value)
                .map(|catalog| (*value, boundary, catalog))
        })
        .collect();
    let plan = plan_residency(candidates, &policy, None);

    Prepared::Ready {
        plan,
        routed_experts,
        epoch: snapshot.epoch(),
        count: snapshot.count(),
    }
}

/// Consume one completed coarse-boundary route-telemetry window and, when
/// enabled and trustworthy, apply the resulting per-expert residency plan
/// through the existing #1854 coarse-residency lifecycle.
///
/// `snapshot` is a host copy of the window the producer already accumulated
/// and copied back at a stream-completion boundary (see
/// `QMoEKernel::route_telemetry_snapshot`). `bank_values` are the lazy-weight
/// [`ValueId`]s of the expert bank this window describes (the fc1/fc2/fc3/
/// scale tensors of one QMoE/BlockQuantizedMoE node); `expert_groups` ties
/// cross-tensor members into a logical expert so they transition atomically.
/// `expected_epoch`/`expected_request`/`expected_device` are the boundary
/// authority's identity for this window — a mismatch fails closed to
/// [`RouteWindowConsumeOutcome::WholeBank`].
///
/// See the module docs for the full invariant. This never allocates, never
/// remaps under capture, and never tiers without recording a reason.
#[allow(clippy::too_many_arguments)]
pub fn consume_route_window_at_boundary(
    residency: &CudaWeightResidency,
    snapshot: &TelemetrySnapshot,
    expected_epoch: u32,
    expected_request: u32,
    expected_device: u32,
    bank_values: &[ValueId],
    boundary: LazyWeightBoundary,
    catalogs: &HashMap<ValueId, WeightRegionCatalog>,
    allocators: &HashMap<ValueId, Arc<CudaVmmAllocator>>,
    device_pool: &Arc<PhysicalHandlePool>,
    host_pool: &Arc<PhysicalHandlePool>,
    device_count: usize,
    device_ordinal: i32,
    expert_groups: &[ExpertWeightGroup],
) -> RouteWindowConsumeOutcome {
    match prepare_route_window(
        residency,
        snapshot,
        expected_epoch,
        expected_request,
        expected_device,
        bank_values,
        boundary,
        catalogs,
        device_count,
    ) {
        Prepared::Early(outcome) => outcome,
        Prepared::Ready {
            plan,
            routed_experts,
            epoch,
            count,
        } => {
            let outcome = residency.apply_coarse_residency_plan(
                &plan,
                catalogs,
                allocators,
                device_pool,
                host_pool,
                device_count,
                device_ordinal,
                expert_groups,
            );
            RouteWindowConsumeOutcome::Applied {
                routed_experts,
                epoch,
                count,
                outcome: Box::new(outcome),
            }
        }
    }
}

/// Test-only entry point identical to [`consume_route_window_at_boundary`] but
/// routing the plan application through
/// [`crate::coarse_residency::apply_residency_plan_at_boundary_with_phase8_faults`],
/// so a deterministic driver fault can prove the consumer-driven transition
/// rolls back range-precisely and quarantines exactly like a real driver
/// failure would. Not reachable from production (the parameter only exists
/// under `#[cfg(any(test, feature = "gpu-tests"))]`).
#[cfg(any(test, feature = "gpu-tests"))]
#[allow(clippy::too_many_arguments)]
pub fn consume_route_window_at_boundary_with_phase8_faults(
    runtime: &Arc<crate::runtime::CudaRuntime>,
    residency: &CudaWeightResidency,
    snapshot: &TelemetrySnapshot,
    expected_epoch: u32,
    expected_request: u32,
    expected_device: u32,
    bank_values: &[ValueId],
    boundary: LazyWeightBoundary,
    catalogs: &HashMap<ValueId, WeightRegionCatalog>,
    allocators: &HashMap<ValueId, Arc<CudaVmmAllocator>>,
    device_pool: &Arc<PhysicalHandlePool>,
    host_pool: &Arc<PhysicalHandlePool>,
    device_count: usize,
    device_ordinal: i32,
    expert_groups: &[ExpertWeightGroup],
    phase8_faults: HashMap<ValueId, Arc<onnx_runtime_cuda_memory::release::DriverFaultPlan>>,
) -> RouteWindowConsumeOutcome {
    match prepare_route_window(
        residency,
        snapshot,
        expected_epoch,
        expected_request,
        expected_device,
        bank_values,
        boundary,
        catalogs,
        device_count,
    ) {
        Prepared::Early(outcome) => outcome,
        Prepared::Ready {
            plan,
            routed_experts,
            epoch,
            count,
        } => {
            let outcome =
                crate::coarse_residency::apply_residency_plan_at_boundary_with_phase8_faults(
                    runtime,
                    residency,
                    &plan,
                    catalogs,
                    allocators,
                    device_pool,
                    host_pool,
                    device_count,
                    device_ordinal,
                    expert_groups,
                    phase8_faults,
                );
            RouteWindowConsumeOutcome::Applied {
                routed_experts,
                epoch,
                count,
                outcome: Box::new(outcome),
            }
        }
    }
}
