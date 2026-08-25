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
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use onnx_runtime_cuda_memory::virtual_memory::PhysicalHandlePool;
use onnx_runtime_cuda_memory::vmm_allocator::CudaVmmAllocator;
use onnx_runtime_ep_api::{
    ExpertWeightGroup, LazyWeightBoundary, ResidencyPlan, Result, StaticProfileResidencyPolicy,
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

// ---------------------------------------------------------------------------
// Slice 7C — production boundary wiring.
//
// The consumer above is pure host glue but takes ~14 arguments and names CUDA
// types the EP-agnostic executor cannot. So the single production call site
// (`Executor::finish_device_validation` → `ExecutionProvider::
// consume_route_residency_at_boundary`) reaches it through the CUDA EP, which
// owns one optional `RouteResidencyBoundary` binding. The binding carries the
// producer window source plus every already-existing authority handle the
// consumer needs; the EP override drives snapshot → consume → reset exactly
// once per boundary and records the typed outcome here. Production installs no
// binding yet (honest "reachable seam" — matching how 7A/7B shipped), so the
// override is a lock + `None` check when the gate is on and a bare env read
// when it is off. The Slice-7C GPU tests install a binding and exercise the
// whole matrix through this same override.
// ---------------------------------------------------------------------------

/// The producer half of one route-telemetry window, abstracted so the boundary
/// caller can drive a real armed [`QMoEKernel`] in production and a controllable
/// double in tests without either side depending on the other. The two methods
/// are exactly the producer's existing stream-completion snapshot authority and
/// its guarded window advance — this trait adds *no* new mechanism, it only
/// names the ordered pair the boundary consumer must call.
///
/// [`QMoEKernel`]: crate::kernels::qmoe::QMoEKernel
pub trait RouteTelemetrySource: Send + Sync {
    /// Host copy of the current (already stream-completed) window, or `None`
    /// when telemetry is disarmed. Self-synchronizing; see
    /// `QMoEKernel::route_telemetry_snapshot`.
    fn route_telemetry_snapshot(&self) -> Result<Option<TelemetrySnapshot>>;

    /// Advance to the next accumulation window (epoch bump + re-zero). Rejected
    /// while the stream is capturing/replaying; see
    /// `QMoEKernel::reset_route_telemetry_boundary`.
    fn reset_route_telemetry_boundary(&self) -> Result<()>;
}

/// One expert bank's binding for the boundary consumer: the producer window
/// source plus the exact residency + catalog/allocator/pool/expert-group
/// arguments [`consume_route_window_at_boundary`] needs, and the boundary's own
/// identity (request/device) and monotonic expected epoch.
///
/// This is *pure binding* — it owns no new allocator and maps nothing; every
/// field is a handle to an existing authority (the residency is the EP's own
/// [`CudaWeightResidency`], the source is a live `QMoEKernel`). It is installed
/// on the CUDA EP via `CudaExecutionProvider::install_route_residency_boundary`.
/// Production wiring that constructs one from a live session's expert banks is a
/// later slice, so today it is populated only by the Slice-7C GPU tests.
pub struct RouteResidencyBoundary {
    source: Arc<dyn RouteTelemetrySource>,
    residency: Arc<CudaWeightResidency>,
    bank_values: Vec<ValueId>,
    boundary: LazyWeightBoundary,
    catalogs: HashMap<ValueId, WeightRegionCatalog>,
    allocators: HashMap<ValueId, Arc<CudaVmmAllocator>>,
    device_pool: Arc<PhysicalHandlePool>,
    host_pool: Arc<PhysicalHandlePool>,
    device_count: usize,
    device_ordinal: i32,
    expected_request: u32,
    expected_device: u32,
    /// The boundary epoch this consumer expects the just-completed window to
    /// carry. Starts at the armed epoch and advances in lockstep with the
    /// producer's window each time a window is consumed and reset, so a record
    /// that failed to advance (an older epoch) is caught as stale.
    expected_epoch: AtomicU32,
    expert_groups: Vec<ExpertWeightGroup>,
}

impl RouteResidencyBoundary {
    /// Bind one expert bank's producer source and residency authorities for the
    /// boundary consumer. `initial_epoch` is the epoch the first consumed window
    /// must carry (the producer arms at `1`).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        source: Arc<dyn RouteTelemetrySource>,
        residency: Arc<CudaWeightResidency>,
        bank_values: Vec<ValueId>,
        boundary: LazyWeightBoundary,
        catalogs: HashMap<ValueId, WeightRegionCatalog>,
        allocators: HashMap<ValueId, Arc<CudaVmmAllocator>>,
        device_pool: Arc<PhysicalHandlePool>,
        host_pool: Arc<PhysicalHandlePool>,
        device_count: usize,
        device_ordinal: i32,
        expected_request: u32,
        expected_device: u32,
        initial_epoch: u32,
        expert_groups: Vec<ExpertWeightGroup>,
    ) -> Self {
        Self {
            source,
            residency,
            bank_values,
            boundary,
            catalogs,
            allocators,
            device_pool,
            host_pool,
            device_count,
            device_ordinal,
            expected_request,
            expected_device,
            expected_epoch: AtomicU32::new(initial_epoch),
            expert_groups,
        }
    }

    fn expected_epoch(&self) -> u32 {
        self.expected_epoch.load(Ordering::Relaxed)
    }

    fn advance_epoch(&self) {
        self.expected_epoch.fetch_add(1, Ordering::Relaxed);
    }
}

/// Observability for the boundary consumer. Every boundary records its typed
/// reason/outcome here — there is no silent success and no silent whole-bank
/// (design-discipline "carry the reason"). Mirrors the crate's other EP-owned
/// metric surfaces; the CUDA EP exposes it through
/// `CudaExecutionProvider::route_residency_diagnostics`.
#[derive(Debug, Default)]
pub struct RouteResidencyDiagnostics {
    boundaries: AtomicU64,
    applied: AtomicU64,
    rejected: AtomicU64,
    whole_bank: AtomicU64,
    empty: AtomicU64,
    last_reason: Mutex<Option<String>>,
}

impl RouteResidencyDiagnostics {
    /// Total boundaries the consumer actually ran (gate on and a binding
    /// installed) — the reachability counter the wiring tests assert on.
    pub fn boundaries(&self) -> u64 {
        self.boundaries.load(Ordering::Relaxed)
    }

    /// Boundaries that applied a routed hot-set through the #1854 lifecycle.
    pub fn applied(&self) -> u64 {
        self.applied.load(Ordering::Relaxed)
    }

    /// Boundaries rejected before consume/reset because the point was unsafe.
    pub fn rejected(&self) -> u64 {
        self.rejected.load(Ordering::Relaxed)
    }

    /// Boundaries that fail-closed to whole-bank (poison/overflow/stale/foreign
    /// identity/empty routed set).
    pub fn whole_bank(&self) -> u64 {
        self.whole_bank.load(Ordering::Relaxed)
    }

    /// Boundaries where the source was disarmed (no window to consume).
    pub fn empty(&self) -> u64 {
        self.empty.load(Ordering::Relaxed)
    }

    /// The human reason of the most recent boundary (for diagnostics surfaces).
    pub fn last_reason(&self) -> Option<String> {
        self.last_reason.lock().unwrap().clone()
    }

    fn set_reason(&self, reason: String) {
        *self.last_reason.lock().unwrap() = Some(reason);
    }

    fn record_rejected(&self, reason: &str) {
        self.rejected.fetch_add(1, Ordering::Relaxed);
        self.set_reason(format!("rejected: {reason}"));
    }

    fn record_empty(&self, reason: &str) {
        self.empty.fetch_add(1, Ordering::Relaxed);
        self.set_reason(format!("empty: {reason}"));
    }

    fn record_outcome(&self, outcome: &RouteWindowConsumeOutcome) {
        match outcome {
            RouteWindowConsumeOutcome::Disabled => {
                self.set_reason("disabled".into());
            }
            RouteWindowConsumeOutcome::RejectedNotSafeBoundary { reason } => {
                self.rejected.fetch_add(1, Ordering::Relaxed);
                self.set_reason(format!("rejected: {reason}"));
            }
            RouteWindowConsumeOutcome::WholeBank { reason } => {
                self.whole_bank.fetch_add(1, Ordering::Relaxed);
                self.set_reason(format!("whole-bank: {reason}"));
            }
            RouteWindowConsumeOutcome::Applied {
                routed_experts,
                epoch,
                count,
                ..
            } => {
                self.applied.fetch_add(1, Ordering::Relaxed);
                self.set_reason(format!(
                    "applied hot-set of {} experts at epoch {epoch} (count {count})",
                    routed_experts.len()
                ));
            }
        }
    }
}

/// Whether a boundary outcome consumed the window (and therefore the producer
/// must advance to the next one). Disabled/Rejected never touched the window.
fn window_was_consumed(outcome: &RouteWindowConsumeOutcome) -> bool {
    matches!(
        outcome,
        RouteWindowConsumeOutcome::Applied { .. } | RouteWindowConsumeOutcome::WholeBank { .. }
    )
}

/// Drive one boundary for `binding`, recording the typed outcome in `diag`.
///
/// Ordering (the lawful boundary sequence from the module docs): fail-closed
/// safe-point pre-check → producer snapshot → the merged
/// [`consume_route_window_at_boundary`] → producer window advance — the reset
/// (and the expected-epoch advance that keeps stale detection honest) fires
/// **only** after a window was actually consumed, so an unsafe or disarmed
/// boundary neither snapshots nor resets. Reuses only the merged #1971 consumer
/// and #1854 lifecycle; adds no mapping, allocation, or host sync of its own.
///
/// The caller (the CUDA EP override) has already checked the default-off gate
/// and that a binding is installed, so reaching here means the consumer runs.
pub fn run_route_residency_boundary(
    binding: &RouteResidencyBoundary,
    diag: &RouteResidencyDiagnostics,
) -> Result<()> {
    diag.boundaries.fetch_add(1, Ordering::Relaxed);

    // Fail closed before the snapshot dtoh so an unsafe boundary (capture/
    // replay, admission in flight, unsettled deferred release, multi-device,
    // live routed guard) neither reads telemetry nor advances the window.
    if let Some(reason) = binding
        .residency
        .resize_safe_point(binding.device_count)
        .blocking_reason()
    {
        diag.record_rejected(reason);
        return Ok(());
    }

    let Some(snapshot) = binding.source.route_telemetry_snapshot()? else {
        diag.record_empty("route telemetry disarmed; no window to consume");
        return Ok(());
    };

    let outcome = consume_route_window_at_boundary(
        &binding.residency,
        &snapshot,
        binding.expected_epoch(),
        binding.expected_request,
        binding.expected_device,
        &binding.bank_values,
        binding.boundary,
        &binding.catalogs,
        &binding.allocators,
        &binding.device_pool,
        &binding.host_pool,
        binding.device_count,
        binding.device_ordinal,
        &binding.expert_groups,
    );

    if window_was_consumed(&outcome) {
        binding.source.reset_route_telemetry_boundary()?;
        binding.advance_epoch();
    }

    diag.record_outcome(&outcome);
    Ok(())
}

/// Test-only sibling of [`run_route_residency_boundary`] that routes the plan
/// application through the phase-8 driver-fault consumer, so a deterministic
/// unmap/map fault can prove the *caller-driven* transition rolls back
/// range-precisely and quarantines exactly like a real driver failure. Same
/// ordering and reset discipline as production; only the apply path differs.
#[cfg(any(test, feature = "gpu-tests"))]
pub fn run_route_residency_boundary_with_phase8_faults(
    runtime: &Arc<crate::runtime::CudaRuntime>,
    binding: &RouteResidencyBoundary,
    diag: &RouteResidencyDiagnostics,
    phase8_faults: HashMap<ValueId, Arc<onnx_runtime_cuda_memory::release::DriverFaultPlan>>,
) -> Result<()> {
    diag.boundaries.fetch_add(1, Ordering::Relaxed);

    if let Some(reason) = binding
        .residency
        .resize_safe_point(binding.device_count)
        .blocking_reason()
    {
        diag.record_rejected(reason);
        return Ok(());
    }

    let Some(snapshot) = binding.source.route_telemetry_snapshot()? else {
        diag.record_empty("route telemetry disarmed; no window to consume");
        return Ok(());
    };

    let outcome = consume_route_window_at_boundary_with_phase8_faults(
        runtime,
        &binding.residency,
        &snapshot,
        binding.expected_epoch(),
        binding.expected_request,
        binding.expected_device,
        &binding.bank_values,
        binding.boundary,
        &binding.catalogs,
        &binding.allocators,
        &binding.device_pool,
        &binding.host_pool,
        binding.device_count,
        binding.device_ordinal,
        &binding.expert_groups,
        phase8_faults,
    );

    if window_was_consumed(&outcome) {
        binding.source.reset_route_telemetry_boundary()?;
        binding.advance_epoch();
    }

    diag.record_outcome(&outcome);
    Ok(())
}

/// Compile-time proof that the production producer satisfies the boundary
/// source contract, so the GPU tests' controllable double stands in for a real
/// armed kernel without diverging from it.
#[allow(dead_code)]
fn _assert_qmoe_is_route_telemetry_source() {
    fn is_source<T: RouteTelemetrySource>() {}
    is_source::<crate::kernels::qmoe::QMoEKernel>();
}
