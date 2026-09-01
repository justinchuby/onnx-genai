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
    expert_weight_groups, plan_residency,
};
use onnx_runtime_ir::{Graph, NodeId, ValueId};
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
        snapshot.num_experts,
        usize::try_from(snapshot.routes_per_row)
            .expect("u32 routes-per-row telemetry contract fits usize"),
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
/// [`build_route_residency_boundary`] constructs one by property-based
/// discovery over a loaded graph's expert banks; the CUDA EP calls it through
/// `CudaExecutionProvider::try_install_route_residency_binding` after weights
/// are loaded and before decode capture.
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

    /// Number of bank values this binding covers (for install diagnostics).
    pub fn bank_value_count(&self) -> usize {
        self.bank_values.len()
    }

    fn advance_epoch(&self) {
        self.expected_epoch.fetch_add(1, Ordering::Relaxed);
    }
}

/// Why a production route-residency binding could not be constructed from a
/// loaded graph. Every variant is fail-closed: the EP installs *nothing* and
/// ordinary inference is untouched. There is no silent partial binding — the
/// reason is carried so diagnostics can surface exactly what was missing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RouteResidencyBindingReject {
    /// Property-based discovery found no routed expert group (no QMoE/
    /// BlockQuantizedMoE node with initializer-backed weight inputs). Dense-only
    /// or non-MoE graphs land here; there is nothing to tier.
    NoExpertGroups,
    /// More than one routed expert group was discovered. The single-binding
    /// install authority (one producer window source per EP/request/device)
    /// cannot yet cover multiple banks, so binding is refused rather than
    /// silently covering only one. Multi-bank binding is a later slice.
    MultipleBanksUnsupported { groups: usize },
    /// The discovered group's node has no armed route-telemetry producer source
    /// yet, but its boundary publishes one during resolved kernel compilation.
    /// Without a producer there is no window to consume, so no binding is
    /// installed until a later readiness epoch.
    NoTelemetrySource { node: NodeId },
    /// This boundary has no executor-scoped route-telemetry producer
    /// publication path. Waiting for another readiness epoch cannot change that,
    /// so the binding is terminally declined rather than left pending forever.
    TelemetryProducerUnsupported {
        node: NodeId,
        boundary: LazyWeightBoundary,
    },
    /// A group member weight has no region catalog — it was not classified/
    /// loaded, so the boundary consumer could not map its regions.
    MissingCatalog { value: ValueId },
    /// A group member weight has no backing VMM allocator — it was not paged/
    /// committed, so the boundary consumer had no allocator to tier against.
    MissingAllocator { value: ValueId },
    /// The discovered bank's members have no *per-bank dedicated* VMM
    /// reservation for the coarse route-residency plan to remap (issue #1810
    /// Slice 7E). The shipped [`CudaWeightResidency`](crate::weight_paging::CudaWeightResidency)
    /// packs every paged weight into one *shared* VMM reservation with per-key
    /// stable-VA slots (issue #716); the coarse plan addresses each bank at
    /// catalog-relative offsets, which only a per-bank reservation satisfies, so
    /// installing here would let the boundary consumer remap the wrong bytes.
    /// The seam fail-closes instead. Route producer sources (Slice-7E goal 2)
    /// and the real build-time install call (goal 3) are wired and fire; the
    /// per-bank-reservation bridge in the residency is the disclosed residual a
    /// later slice supplies, after which this same seam installs for real.
    NoPerBankReservation { value: ValueId },
}

impl RouteResidencyBindingReject {
    /// A stable human reason for diagnostics/decision surfaces.
    pub fn reason(&self) -> String {
        match self {
            RouteResidencyBindingReject::NoExpertGroups => {
                "no routed expert group discovered".to_string()
            }
            RouteResidencyBindingReject::MultipleBanksUnsupported { groups } => {
                format!("{groups} expert groups; single-binding authority covers one bank")
            }
            RouteResidencyBindingReject::NoTelemetrySource { node } => {
                format!("expert group node {node:?} has no armed telemetry source")
            }
            RouteResidencyBindingReject::TelemetryProducerUnsupported { node, boundary } => {
                format!(
                    "expert group node {node:?} at {boundary:?} has no supported executor-scoped \
                     route-telemetry producer"
                )
            }
            RouteResidencyBindingReject::MissingCatalog { value } => {
                format!("bank value {value:?} has no region catalog")
            }
            RouteResidencyBindingReject::MissingAllocator { value } => {
                format!("bank value {value:?} has no VMM allocator")
            }
            RouteResidencyBindingReject::NoPerBankReservation { value } => {
                format!(
                    "bank value {value:?} has no per-bank VMM reservation; shipped residency \
                     packs banks into one shared reservation (per-bank-reservation bridge is the \
                     Slice-7E residual)"
                )
            }
        }
    }
}

/// The outcome of a CUDA-EP attempt to install a production route-residency
/// binding. Only [`Installed`](Self::Installed) creates any boundary state; the
/// other three variants install nothing and add no overhead — the shipped
/// default-off path always lands on [`GateDisabled`](Self::GateDisabled).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RouteResidencyInstallOutcome {
    /// The default-off coarse-residency gate is disabled (the shipped default).
    /// Nothing is discovered, built, or installed.
    GateDisabled,
    /// The gate is on but this EP has no weight-offload/coarse-residency
    /// authority (no `CudaWeightResidency`), so there is nothing to tier
    /// against. Nothing is installed.
    OffloadDisabled,
    /// The gate is on and residency is present, but property-based binding
    /// fail-closed with a typed reason. Nothing is installed.
    Rejected(RouteResidencyBindingReject),
    /// A real binding was installed over `banks` bank values.
    Installed { banks: usize },
}

/// Property-based validation of a bindable expert bank against the artifacts an
/// install would supply, with no GPU handles required (so it is unit-testable).
///
/// Uses the typed [`expert_weight_groups`] discovery (via
/// [`LazyWeightBoundary::for_op`], never op/name allowlists) and fail-closes on
/// the exact reason nothing can be bound. `has_source`/`has_catalog`/
/// `has_allocator` are membership predicates over the maps the real builder
/// holds, so the pure classification here matches what
/// [`build_route_residency_boundary`] would construct.
pub(crate) fn validate_route_residency_binding(
    graph: &Graph,
    has_source: impl Fn(NodeId) -> bool,
    has_catalog: impl Fn(ValueId) -> bool,
    has_allocator: impl Fn(ValueId) -> bool,
) -> std::result::Result<ExpertWeightGroup, RouteResidencyBindingReject> {
    let mut groups = expert_weight_groups(graph);
    if groups.is_empty() {
        return Err(RouteResidencyBindingReject::NoExpertGroups);
    }
    if groups.len() > 1 {
        return Err(RouteResidencyBindingReject::MultipleBanksUnsupported {
            groups: groups.len(),
        });
    }
    let group = groups.pop().expect("exactly one group");
    if !has_source(group.node) {
        return Err(
            if group
                .boundary
                .route_telemetry_producer_may_appear_after_compilation()
            {
                RouteResidencyBindingReject::NoTelemetrySource { node: group.node }
            } else {
                RouteResidencyBindingReject::TelemetryProducerUnsupported {
                    node: group.node,
                    boundary: group.boundary,
                }
            },
        );
    }
    for member in &group.members {
        if !has_catalog(*member) {
            return Err(RouteResidencyBindingReject::MissingCatalog { value: *member });
        }
        if !has_allocator(*member) {
            return Err(RouteResidencyBindingReject::MissingAllocator { value: *member });
        }
    }
    Ok(group)
}

/// Construct a production [`RouteResidencyBoundary`] from a loaded graph's
/// expert banks by property-based discovery.
///
/// The bank identity, membership (fc1/fc2/fc3/scales/bias) and boundary kind
/// come entirely from [`expert_weight_groups`] — no model/layer/op-name
/// allowlist. The producer `source`, `catalogs`, `allocators`, and pools are
/// the EP's existing authorities, keyed by the discovered node/values; the
/// builder only *binds* them (it maps nothing and owns no new allocator). On
/// any missing artifact it fail-closes with a typed
/// [`RouteResidencyBindingReject`] so the caller installs nothing.
#[allow(clippy::too_many_arguments)]
pub fn build_route_residency_boundary(
    graph: &Graph,
    residency: Arc<CudaWeightResidency>,
    sources: &HashMap<NodeId, Arc<dyn RouteTelemetrySource>>,
    catalogs: HashMap<ValueId, WeightRegionCatalog>,
    allocators: HashMap<ValueId, Arc<CudaVmmAllocator>>,
    device_pool: Arc<PhysicalHandlePool>,
    host_pool: Arc<PhysicalHandlePool>,
    device_count: usize,
    device_ordinal: i32,
    expected_request: u32,
    expected_device: u32,
    initial_epoch: u32,
) -> std::result::Result<RouteResidencyBoundary, RouteResidencyBindingReject> {
    let group = validate_route_residency_binding(
        graph,
        |node| sources.contains_key(&node),
        |value| catalogs.contains_key(&value),
        |value| allocators.contains_key(&value),
    )?;
    let source = Arc::clone(&sources[&group.node]);
    let bank_values = group.members.clone();
    let boundary = group.boundary;
    Ok(RouteResidencyBoundary::new(
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
        initial_epoch,
        vec![group],
    ))
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
    installs: AtomicU64,
    declines: AtomicU64,
    last_install_reason: Mutex<Option<String>>,
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

    /// Bindings actually installed on the EP (gate on, a bindable bank found).
    /// The reachability counter the install-wiring tests assert on.
    pub fn installs(&self) -> u64 {
        self.installs.load(Ordering::Relaxed)
    }

    /// Install attempts that fail-closed to *no* binding (gate off, offload
    /// disabled, or a typed [`RouteResidencyBindingReject`]). Nothing is
    /// installed and no boundary work is created — there is no silent partial
    /// binding.
    pub fn declines(&self) -> u64 {
        self.declines.load(Ordering::Relaxed)
    }

    /// The human reason of the most recent install/decline (for diagnostics).
    pub fn last_install_reason(&self) -> Option<String> {
        self.last_install_reason.lock().unwrap().clone()
    }

    fn set_install_reason(&self, reason: String) {
        *self.last_install_reason.lock().unwrap() = Some(reason);
    }

    /// Record that a real binding was installed for `banks` bank values.
    pub(crate) fn record_install(&self, banks: usize) {
        self.installs.fetch_add(1, Ordering::Relaxed);
        self.set_install_reason(format!("installed binding over {banks} bank value(s)"));
    }

    /// Record that install fail-closed to no binding, carrying the reason.
    pub(crate) fn record_decline(&self, reason: &str) {
        self.declines.fetch_add(1, Ordering::Relaxed);
        self.set_install_reason(format!("declined: {reason}"));
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

#[cfg(test)]
mod binding_tests {
    //! CPU-only tests for the property-based binding *builder*'s discovery and
    //! typed fail-closed rejects. These need no GPU handles because
    //! [`validate_route_residency_binding`] classifies purely from the graph
    //! and membership predicates — the exact predicates
    //! [`build_route_residency_boundary`] evaluates against its real
    //! source/catalog/allocator maps. The successful *construction* (which does
    //! need a real residency/allocator) is proven by the GPU harness.
    use std::collections::HashSet;

    use onnx_runtime_ep_api::LazyWeightBoundary;
    use onnx_runtime_ir::{DataType, Graph, NodeId, TensorData, ValueId, WeightRef, static_shape};

    use super::{RouteResidencyBindingReject, validate_route_residency_binding};

    fn shape1(n: usize) -> onnx_runtime_ir::Shape {
        static_shape([n])
    }

    fn inline_initializer(graph: &mut Graph, name: &str) -> ValueId {
        let value = graph.create_named_value(name, DataType::Uint8, shape1(4));
        graph.set_initializer(
            value,
            WeightRef::Inline(TensorData::from_raw(DataType::Uint8, vec![4], vec![0u8; 4])),
        );
        value
    }

    /// A shape-faithful single-layer QMoE node: two graph values (hidden state,
    /// router probs) plus initializer-backed fc1/fc2/fc3 weights+scales+bias —
    /// exactly the input arity `expert_weight_groups` classifies.
    fn qmoe_node(graph: &mut Graph) -> (NodeId, Vec<ValueId>) {
        let input = graph.create_named_value("input", DataType::Float32, shape1(4));
        let router = graph.create_named_value("router_probs", DataType::Float32, shape1(4));
        let fc1_w = inline_initializer(graph, "fc1_experts_weights");
        let fc1_s = inline_initializer(graph, "fc1_scales");
        let fc1_b = inline_initializer(graph, "fc1_experts_bias");
        let fc2_w = inline_initializer(graph, "fc2_experts_weights");
        let fc2_s = inline_initializer(graph, "fc2_scales");
        let fc3_w = inline_initializer(graph, "fc3_experts_weights");
        let fc3_s = inline_initializer(graph, "fc3_scales");
        let output = graph.create_named_value("output", DataType::Float32, shape1(4));
        let mut node = onnx_runtime_ir::Node::new(
            NodeId(0),
            "QMoE",
            vec![
                Some(input),
                Some(router),
                Some(fc1_w),
                Some(fc1_s),
                Some(fc1_b),
                Some(fc2_w),
                Some(fc2_s),
                None,
                Some(fc3_w),
                Some(fc3_s),
            ],
            vec![output],
        );
        node.domain = "com.microsoft".to_string();
        let node_id = graph.insert_node(node);
        (
            node_id,
            vec![fc1_w, fc1_s, fc1_b, fc2_w, fc2_s, fc3_w, fc3_s],
        )
    }

    fn block_quantized_moe_node(graph: &mut Graph) -> (NodeId, ValueId) {
        let input = graph.create_named_value("input", DataType::Float32, shape1(4));
        let weight = inline_initializer(graph, "experts");
        let output = graph.create_named_value("output", DataType::Float32, shape1(4));
        let mut node = onnx_runtime_ir::Node::new(
            NodeId(0),
            "BlockQuantizedMoE",
            vec![Some(input), Some(weight)],
            vec![output],
        );
        node.domain = "pkg.nxrt".to_string();
        (graph.insert_node(node), weight)
    }

    fn always(_: NodeId) -> bool {
        true
    }
    fn always_v(_: ValueId) -> bool {
        true
    }

    #[test]
    fn binds_single_qmoe_bank_with_all_artifacts_present() {
        let mut graph = Graph::new();
        let (node, members) = qmoe_node(&mut graph);
        let group = validate_route_residency_binding(&graph, always, always_v, always_v)
            .expect("bindable bank");
        assert_eq!(group.node, node);
        assert_eq!(group.boundary, LazyWeightBoundary::QMoe);
        assert_eq!(group.members, members, "exact fc1/fc2/fc3 membership bound");
    }

    #[test]
    fn rejects_graph_with_no_expert_group() {
        // Dense-only graph: a MatMul is not a routed multi-tensor expert group.
        let mut graph = Graph::new();
        let w = inline_initializer(&mut graph, "dense_weight");
        let x = graph.create_named_value("x", DataType::Float32, shape1(4));
        let y = graph.create_named_value("y", DataType::Float32, shape1(4));
        graph.insert_node(onnx_runtime_ir::Node::new(
            NodeId(0),
            "MatMul",
            vec![Some(x), Some(w)],
            vec![y],
        ));
        assert_eq!(
            validate_route_residency_binding(&graph, always, always_v, always_v),
            Err(RouteResidencyBindingReject::NoExpertGroups)
        );
    }

    #[test]
    fn rejects_multiple_banks_as_single_binding_unsupported() {
        let mut graph = Graph::new();
        qmoe_node(&mut graph);
        qmoe_node(&mut graph);
        assert_eq!(
            validate_route_residency_binding(&graph, always, always_v, always_v),
            Err(RouteResidencyBindingReject::MultipleBanksUnsupported { groups: 2 })
        );
    }

    #[test]
    fn rejects_when_group_node_has_no_telemetry_source() {
        let mut graph = Graph::new();
        let (node, _) = qmoe_node(&mut graph);
        let err = validate_route_residency_binding(&graph, |_| false, always_v, always_v)
            .expect_err("no source");
        assert_eq!(err, RouteResidencyBindingReject::NoTelemetrySource { node });
    }

    #[test]
    fn missing_block_quantized_moe_producer_is_terminally_unsupported() {
        let mut graph = Graph::new();
        let (node, _) = block_quantized_moe_node(&mut graph);
        let err = validate_route_residency_binding(&graph, |_| false, always_v, always_v)
            .expect_err("BlockQuantizedMoE has no deferred producer");
        assert_eq!(
            err,
            RouteResidencyBindingReject::TelemetryProducerUnsupported {
                node,
                boundary: LazyWeightBoundary::BlockQuantizedMoe,
            }
        );
        let reason = err.reason();
        assert!(
            reason.contains("BlockQuantizedMoe") && reason.contains("no supported"),
            "terminal reason names the unsupported boundary capability: {reason}"
        );
    }

    #[test]
    fn rejects_when_a_bank_member_has_no_catalog() {
        let mut graph = Graph::new();
        let (_, members) = qmoe_node(&mut graph);
        // Every member classified except the first, which lacks a catalog.
        let with_catalog: HashSet<ValueId> = members[1..].iter().copied().collect();
        let err = validate_route_residency_binding(
            &graph,
            always,
            |v| with_catalog.contains(&v),
            always_v,
        )
        .expect_err("missing catalog");
        assert_eq!(
            err,
            RouteResidencyBindingReject::MissingCatalog { value: members[0] }
        );
    }

    #[test]
    fn rejects_when_a_bank_member_has_no_allocator() {
        let mut graph = Graph::new();
        let (_, members) = qmoe_node(&mut graph);
        let with_alloc: HashSet<ValueId> = members[1..].iter().copied().collect();
        let err =
            validate_route_residency_binding(&graph, always, always_v, |v| with_alloc.contains(&v))
                .expect_err("missing allocator");
        assert_eq!(
            err,
            RouteResidencyBindingReject::MissingAllocator { value: members[0] }
        );
    }

    #[test]
    fn reject_reasons_are_non_empty_and_carry_context() {
        assert!(
            !RouteResidencyBindingReject::NoExpertGroups
                .reason()
                .is_empty()
        );
        let r = RouteResidencyBindingReject::MultipleBanksUnsupported { groups: 3 }.reason();
        assert!(r.contains('3'), "reason carries the group count: {r}");
    }

    #[test]
    fn no_per_bank_reservation_reason_names_the_value_and_residual() {
        // The shipped enabled-path terminal decline: real discovery + retention
        // succeed, but the shared-reservation residency has no per-bank VMM
        // reservation for the coarse plan to address. The reason must name the
        // offending bank value and disclose the per-bank-reservation residual so
        // the honest typed outcome is self-describing in diagnostics.
        let value = ValueId(7);
        let r = RouteResidencyBindingReject::NoPerBankReservation { value }.reason();
        assert!(r.contains("7"), "reason names the bank value: {r}");
        assert!(
            r.contains("per-bank") && r.contains("reservation"),
            "reason discloses the per-bank-reservation residual: {r}"
        );
    }
}
