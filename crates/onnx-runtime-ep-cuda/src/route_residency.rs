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
//! # Production status
//!
//! The session executor invokes this consumer after its synchronized device
//! validation boundary. The CUDA provider discovers each expert bank from the
//! loaded graph, activates the exact shape-specific telemetry source on kernel
//! execution, and binds retained shared-arena VMM slots without introducing a
//! second allocator. The shipped gate remains default-off.
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
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use onnx_runtime_cuda_memory::virtual_memory::PhysicalHandlePool;
use onnx_runtime_cuda_memory::virtual_memory::PhysicalLocation;
use onnx_runtime_cuda_memory::vmm_allocator::CudaVmmAllocator;
use onnx_runtime_ep_api::{
    ExpertExecutionPhase, ExpertLayerResidencyMetrics, ExpertResidencyMetrics, ExpertWeightGroup,
    LazyWeightBoundary, ResidencyPlan, Result, RouteResidencyInstallState,
    StaticProfileResidencyPolicy, expert_weight_groups, plan_residency,
};
use onnx_runtime_ir::{Graph, NodeId, ValueId};
use onnx_runtime_loader::WeightRegionCatalog;

use crate::coarse_residency::{BoundaryApplicationOutcome, coarse_residency_profile_enabled};
use crate::granule_transition::{TransitionOutcome, transition_granule_range, verify_safe_point};
use crate::kernels::expert_route_telemetry::{
    RouteDecision, TelemetrySnapshot, consume_and_validate,
};
use crate::weight_paging::CudaWeightResidency;

pub const EXPERT_ACCOUNTING_ENABLE_ENV: &str = "ONNX_GENAI_FREETOKEN_EXPERT_ACCOUNTING";

pub fn expert_accounting_enabled() -> bool {
    matches!(
        std::env::var(EXPERT_ACCOUNTING_ENABLE_ENV)
            .ok()
            .as_deref()
            .map(str::trim)
            .map(str::to_ascii_lowercase)
            .as_deref(),
        Some("1") | Some("true") | Some("on")
    )
}

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

/// One expert-bank value's live allocation inside the production shared VMM
/// arena. Catalog ranges are value-relative; `base_offset` translates them to
/// the allocator reservation without creating a second allocator authority.
#[derive(Clone)]
pub struct RouteAllocationBinding {
    pub allocator: Arc<CudaVmmAllocator>,
    pub base_offset: usize,
    pub allocation_bytes: usize,
}

/// One concrete kernel source registered by the CUDA kernel factory.
#[derive(Clone)]
pub struct RegisteredRouteTelemetrySource {
    pub source: Arc<dyn RouteTelemetrySource>,
    pub phase: ExpertExecutionPhase,
    pub node_name: String,
    pub expected_epoch: Arc<AtomicU32>,
}

/// Provider-owned registry populated when concrete QMoE-family kernels are
/// compiled. A monotonically increasing generation lets the boundary owner
/// rebuild its bindings when a new shape variant replaces a source.
#[derive(Default)]
pub struct RouteTelemetryRegistry {
    sources: Mutex<HashMap<NodeId, RegisteredRouteTelemetrySource>>,
    generation: AtomicU64,
}

impl RouteTelemetryRegistry {
    pub fn activate(
        &self,
        node: NodeId,
        source: Arc<dyn RouteTelemetrySource>,
        phase: ExpertExecutionPhase,
        node_name: String,
        expected_epoch: Arc<AtomicU32>,
    ) {
        let mut sources = self.sources.lock().unwrap();
        let unchanged = sources.get(&node).is_some_and(|current| {
            Arc::ptr_eq(&current.source, &source)
                && current.phase == phase
                && current.node_name == node_name
        });
        if unchanged {
            return;
        }
        sources.insert(
            node,
            RegisteredRouteTelemetrySource {
                source,
                phase,
                node_name,
                expected_epoch,
            },
        );
        drop(sources);
        self.generation.fetch_add(1, Ordering::Release);
    }

    pub fn snapshot(&self) -> (u64, HashMap<NodeId, RegisteredRouteTelemetrySource>) {
        (
            self.generation.load(Ordering::Acquire),
            self.sources.lock().unwrap().clone(),
        )
    }

    pub fn clear(&self) {
        self.sources.lock().unwrap().clear();
        self.generation.fetch_add(1, Ordering::Release);
    }
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
    allocation_offsets: HashMap<ValueId, usize>,
    allocation_bytes: HashMap<ValueId, usize>,
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
    expected_epoch: Arc<AtomicU32>,
    expert_groups: Vec<ExpertWeightGroup>,
    node_id: NodeId,
    node_name: String,
    phase: ExpertExecutionPhase,
    tiers: Mutex<HashMap<(ValueId, usize), PhysicalLocation>>,
    tier_state_poisoned: AtomicBool,
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
        let node_id = expert_groups
            .first()
            .map(|group| group.node)
            .unwrap_or(NodeId(u32::MAX));
        let tiers = catalogs
            .iter()
            .flat_map(|(&value, catalog)| {
                (0..catalog.layout().experts).map(move |expert| {
                    (
                        (value, expert),
                        PhysicalLocation::Device {
                            ordinal: device_ordinal,
                        },
                    )
                })
            })
            .collect();
        Self {
            source,
            residency,
            bank_values,
            boundary,
            catalogs,
            allocators,
            allocation_offsets: HashMap::new(),
            allocation_bytes: HashMap::new(),
            device_pool,
            host_pool,
            device_count,
            device_ordinal,
            expected_request,
            expected_device,
            expected_epoch: Arc::new(AtomicU32::new(initial_epoch)),
            expert_groups,
            node_id,
            node_name: String::new(),
            phase: ExpertExecutionPhase::Decode,
            tiers: Mutex::new(tiers),
            tier_state_poisoned: AtomicBool::new(false),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_with_bindings(
        source: Arc<dyn RouteTelemetrySource>,
        residency: Arc<CudaWeightResidency>,
        bank_values: Vec<ValueId>,
        boundary: LazyWeightBoundary,
        catalogs: HashMap<ValueId, WeightRegionCatalog>,
        bindings: HashMap<ValueId, RouteAllocationBinding>,
        device_pool: Arc<PhysicalHandlePool>,
        host_pool: Arc<PhysicalHandlePool>,
        device_count: usize,
        device_ordinal: i32,
        expected_request: u32,
        expected_device: u32,
        initial_epoch: u32,
        expert_groups: Vec<ExpertWeightGroup>,
    ) -> Self {
        let node_id = expert_groups
            .first()
            .map(|group| group.node)
            .unwrap_or(NodeId(u32::MAX));
        let allocators = bindings
            .iter()
            .map(|(&value, binding)| (value, Arc::clone(&binding.allocator)))
            .collect();
        let allocation_offsets = bindings
            .iter()
            .map(|(&value, binding)| (value, binding.base_offset))
            .collect();
        let allocation_bytes = bindings
            .iter()
            .map(|(&value, binding)| (value, binding.allocation_bytes))
            .collect();
        let tiers = catalogs
            .iter()
            .flat_map(|(&value, catalog)| {
                (0..catalog.layout().experts).map(move |expert| {
                    (
                        (value, expert),
                        PhysicalLocation::Device {
                            ordinal: device_ordinal,
                        },
                    )
                })
            })
            .collect();
        Self {
            source,
            residency,
            bank_values,
            boundary,
            catalogs,
            allocators,
            allocation_offsets,
            allocation_bytes,
            device_pool,
            host_pool,
            device_count,
            device_ordinal,
            expected_request,
            expected_device,
            expected_epoch: Arc::new(AtomicU32::new(initial_epoch)),
            expert_groups,
            node_id,
            node_name: String::new(),
            phase: ExpertExecutionPhase::Decode,
            tiers: Mutex::new(tiers),
            tier_state_poisoned: AtomicBool::new(false),
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

    pub(crate) fn committed_bytes(&self) -> (u64, u64) {
        let tiers = self.tiers.lock().unwrap();
        let mut device = 0u64;
        let mut host = 0u64;
        for (&(value, expert), &location) in tiers.iter() {
            let Some(range) = self.catalogs[&value].relative_range(expert) else {
                continue;
            };
            let bytes = range.end.saturating_sub(range.start) as u64;
            match location {
                PhysicalLocation::Device { .. } => device = device.saturating_add(bytes),
                PhysicalLocation::HostNuma { .. } => host = host.saturating_add(bytes),
            }
        }
        (device, host)
    }

    pub(crate) fn tier_state_poisoned(&self) -> bool {
        self.tier_state_poisoned.load(Ordering::Acquire)
    }

    pub(crate) fn inherit_tier_state(&self, previous: &RouteResidencyBoundary) {
        if self.node_id != previous.node_id {
            return;
        }
        *self.tiers.lock().unwrap() = previous.tiers.lock().unwrap().clone();
        self.tier_state_poisoned.store(
            previous.tier_state_poisoned.load(Ordering::Acquire),
            Ordering::Release,
        );
    }

    pub(crate) fn node_id(&self) -> NodeId {
        self.node_id
    }
}

#[allow(clippy::too_many_arguments)]
pub fn build_route_residency_boundaries_with_bindings(
    graph: &Graph,
    residency: Arc<CudaWeightResidency>,
    sources: &HashMap<NodeId, RegisteredRouteTelemetrySource>,
    catalogs: &HashMap<ValueId, WeightRegionCatalog>,
    bindings: &HashMap<ValueId, RouteAllocationBinding>,
    device_pool: Arc<PhysicalHandlePool>,
    host_pool: Arc<PhysicalHandlePool>,
    device_count: usize,
    device_ordinal: i32,
    expected_request: u32,
    expected_device: u32,
    initial_epoch: u32,
) -> std::result::Result<Vec<RouteResidencyBoundary>, RouteResidencyBindingReject> {
    let groups = expert_weight_groups(graph);
    if groups.is_empty() {
        return Err(RouteResidencyBindingReject::NoExpertGroups);
    }
    let mut boundaries = Vec::with_capacity(groups.len());
    for group in groups {
        let registered = sources
            .get(&group.node)
            .ok_or(RouteResidencyBindingReject::NoTelemetrySource { node: group.node })?;
        let mut group_catalogs = HashMap::with_capacity(group.members.len());
        let mut group_bindings = HashMap::with_capacity(group.members.len());
        for &value in &group.members {
            let catalog = catalogs
                .get(&value)
                .cloned()
                .ok_or(RouteResidencyBindingReject::MissingCatalog { value })?;
            let binding = bindings
                .get(&value)
                .cloned()
                .ok_or(RouteResidencyBindingReject::MissingAllocator { value })?;
            group_catalogs.insert(value, catalog);
            group_bindings.insert(value, binding);
        }
        let mut boundary = RouteResidencyBoundary::new_with_bindings(
            Arc::clone(&registered.source),
            Arc::clone(&residency),
            group.members.clone(),
            group.boundary,
            group_catalogs,
            group_bindings,
            Arc::clone(&device_pool),
            Arc::clone(&host_pool),
            device_count,
            device_ordinal,
            expected_request,
            expected_device,
            initial_epoch,
            vec![group],
        );
        boundary.node_name = registered.node_name.clone();
        boundary.phase = registered.phase;
        boundary.expected_epoch = Arc::clone(&registered.expected_epoch);
        boundaries.push(boundary);
    }
    Ok(boundaries)
}

struct TrackedApplication {
    outcome: BoundaryApplicationOutcome,
    h2d_bytes: u64,
    page_ins: u64,
    completed: bool,
    ref_underflows: u64,
    byte_underflows: u64,
    unaccounted_bytes: u64,
}

#[derive(Clone, Copy)]
struct CompletedTransition {
    value: ValueId,
    expert: usize,
    offset: usize,
    len: usize,
    old: PhysicalLocation,
    new: PhysicalLocation,
}

fn apply_tracked_route_plan(
    binding: &RouteResidencyBoundary,
    routed_experts: &[usize],
) -> TrackedApplication {
    let before = onnx_runtime_cuda_memory::vmm_allocator::global_vmm_stats();
    let mut tracked = apply_tracked_route_plan_inner(binding, routed_experts);
    let after = onnx_runtime_cuda_memory::vmm_allocator::global_vmm_stats();
    tracked.ref_underflows = after.ref_underflows.saturating_sub(before.ref_underflows);
    tracked.byte_underflows = after.byte_underflows.saturating_sub(before.byte_underflows);
    tracked.unaccounted_bytes = after
        .unaccounted_committed_bytes
        .saturating_sub(before.unaccounted_committed_bytes);
    tracked
}

fn apply_tracked_route_plan_inner(
    binding: &RouteResidencyBoundary,
    routed_experts: &[usize],
) -> TrackedApplication {
    let mut outcome = BoundaryApplicationOutcome {
        policy_name: "route_window",
        values_inspected: binding.bank_values.len(),
        ..BoundaryApplicationOutcome::default()
    };
    if binding.tier_state_poisoned.load(Ordering::Acquire) {
        outcome.fallback_reason =
            Some("expert tier state is poisoned after an incomplete rollback".to_string());
        return TrackedApplication {
            outcome,
            h2d_bytes: 0,
            page_ins: 0,
            completed: false,
            ref_underflows: 0,
            byte_underflows: 0,
            unaccounted_bytes: 0,
        };
    }

    let verified =
        match verify_safe_point(binding.residency.resize_safe_point(binding.device_count)) {
            Ok(verified) => verified,
            Err(reason) => {
                outcome.fallback_reason = Some(format!("resize safe-point not clear: {reason}"));
                return TrackedApplication {
                    outcome,
                    h2d_bytes: 0,
                    page_ins: 0,
                    completed: false,
                    ref_underflows: 0,
                    byte_underflows: 0,
                    unaccounted_bytes: 0,
                };
            }
        };
    let granularity = binding
        .device_pool
        .granularity()
        .max(binding.host_pool.granularity());
    let routed = routed_experts
        .iter()
        .copied()
        .collect::<std::collections::HashSet<_>>();
    let host_location = binding.host_pool.location();
    let device_location = binding.device_pool.location();
    let tiers = binding.tiers.lock().unwrap();
    let mut planned = Vec::new();
    for &value in &binding.bank_values {
        let Some(catalog) = binding.catalogs.get(&value) else {
            outcome
                .per_value_fallbacks
                .push((value, "no catalog available".to_string()));
            continue;
        };
        if !catalog.is_pageable() {
            outcome
                .per_value_fallbacks
                .push((value, "catalog not pageable".to_string()));
            continue;
        }
        let Some(&allocation_base) = binding.allocation_offsets.get(&value) else {
            outcome
                .per_value_fallbacks
                .push((value, "no production allocation offset".to_string()));
            continue;
        };
        if allocation_base % granularity != 0 {
            outcome.per_value_fallbacks.push((
                value,
                format!(
                    "production allocation base {allocation_base} is not granule-aligned \
                     ({granularity})"
                ),
            ));
            continue;
        }
        let Some(&allocation_bytes) = binding.allocation_bytes.get(&value) else {
            outcome
                .per_value_fallbacks
                .push((value, "no production allocation size".to_string()));
            continue;
        };
        for expert in 0..catalog.layout().experts {
            let Some(range) = catalog.relative_range(expert) else {
                outcome
                    .per_value_fallbacks
                    .push((value, format!("expert {expert} has no relative range")));
                continue;
            };
            let len = range.end.saturating_sub(range.start);
            if len == 0 {
                continue;
            }
            if range.start % granularity != 0 || len % granularity != 0 {
                outcome.per_value_fallbacks.push((
                    value,
                    format!(
                        "expert {expert} range {}..{} is not granule-aligned ({granularity})",
                        range.start, range.end
                    ),
                ));
                continue;
            }
            if range.end > allocation_bytes {
                outcome.per_value_fallbacks.push((
                    value,
                    format!(
                        "expert {expert} range end {} exceeds allocation {allocation_bytes}",
                        range.end
                    ),
                ));
                continue;
            }
            let Some(offset) = allocation_base.checked_add(range.start) else {
                outcome
                    .per_value_fallbacks
                    .push((value, format!("expert {expert} allocation offset overflow")));
                continue;
            };
            let desired = if routed.contains(&expert) {
                device_location
            } else {
                host_location
            };
            let current = tiers
                .get(&(value, expert))
                .copied()
                .unwrap_or(device_location);
            if current != desired {
                planned.push(CompletedTransition {
                    value,
                    expert,
                    offset,
                    len,
                    old: current,
                    new: desired,
                });
            }
            if routed.contains(&expert) {
                outcome.hot_expert_count = outcome.hot_expert_count.saturating_add(1);
            } else {
                outcome.cold_expert_count = outcome.cold_expert_count.saturating_add(1);
            }
        }
    }
    drop(tiers);
    if !outcome.per_value_fallbacks.is_empty() {
        outcome.fallback_reason =
            Some("expert bank validation failed; no partial application attempted".to_string());
        return TrackedApplication {
            outcome,
            h2d_bytes: 0,
            page_ins: 0,
            completed: false,
            ref_underflows: 0,
            byte_underflows: 0,
            unaccounted_bytes: 0,
        };
    }

    let pools = |location: PhysicalLocation| {
        if matches!(location, PhysicalLocation::Device { .. }) {
            &binding.device_pool
        } else {
            &binding.host_pool
        }
    };
    let mut committed = Vec::new();
    for transition in &planned {
        let allocator = &binding.allocators[&transition.value];
        let result = allocator.with_reservation_mut(|reservation, backing| {
            transition_granule_range(
                binding.residency.runtime(),
                reservation,
                backing,
                transition.offset,
                transition.len,
                transition.new,
                pools(transition.old),
                pools(transition.new),
                &verified,
                || binding.residency.resize_safe_point(binding.device_count),
            )
        });
        match result {
            TransitionOutcome::Committed {
                granules,
                new_owned_bytes,
                old_released_bytes,
            } => {
                committed.push(*transition);
                if granules.saturating_mul(granularity) != transition.len
                    || new_owned_bytes > transition.len as u64
                    || !new_owned_bytes.is_multiple_of(granularity as u64)
                    || old_released_bytes != transition.len as u64
                {
                    outcome.failure_count = outcome.failure_count.saturating_add(1);
                    outcome.fallback_reason = Some(format!(
                        "transition physical-byte accounting mismatch: granules={granules}, \
                         new={new_owned_bytes}, released={old_released_bytes}, expected={}; \
                         newly owned bytes may be zero on a physical-pool hit",
                        transition.len,
                    ));
                    break;
                }
            }
            TransitionOutcome::Rejected { reason } => {
                outcome.failure_count = outcome.failure_count.saturating_add(1);
                outcome.fallback_reason = Some(format!("transition rejected: {reason}"));
                break;
            }
            TransitionOutcome::RolledBack { fault } => {
                outcome.failure_count = outcome.failure_count.saturating_add(1);
                outcome.fallback_reason = Some(format!("transition rolled back: {fault:?}"));
                break;
            }
            TransitionOutcome::Fatal {
                transition_fault,
                quarantined,
                committed_count,
                poisoned_range,
                ..
            } => {
                binding.tier_state_poisoned.store(true, Ordering::Release);
                outcome.failure_count = outcome.failure_count.saturating_add(1);
                outcome
                    .fatal_progress
                    .push((transition.value, committed_count, poisoned_range));
                outcome.quarantined.push((transition.value, quarantined));
                outcome.fallback_reason = Some(format!("fatal transition: {transition_fault:?}"));
                break;
            }
        }
    }
    if committed.len() != planned.len() {
        let mut rollback_failed = false;
        for transition in committed.iter().rev() {
            let allocator = &binding.allocators[&transition.value];
            let result = allocator.with_reservation_mut(|reservation, backing| {
                transition_granule_range(
                    binding.residency.runtime(),
                    reservation,
                    backing,
                    transition.offset,
                    transition.len,
                    transition.old,
                    pools(transition.new),
                    pools(transition.old),
                    &verified,
                    || binding.residency.resize_safe_point(binding.device_count),
                )
            });
            match result {
                TransitionOutcome::Committed {
                    granules,
                    new_owned_bytes,
                    old_released_bytes,
                } if granules.saturating_mul(granularity) == transition.len
                    && new_owned_bytes <= transition.len as u64
                    && new_owned_bytes.is_multiple_of(granularity as u64)
                    && old_released_bytes == transition.len as u64 =>
                {
                    outcome.rollback_count = outcome.rollback_count.saturating_add(1);
                }
                other => {
                    rollback_failed = true;
                    outcome
                        .rollback_failures
                        .push(crate::coarse_residency::RollbackFailure {
                            value: transition.value,
                            range: (transition.offset, transition.len),
                            detail: format!("reverse transition failed: {other:?}"),
                            committed_count: None,
                            poisoned_range: None,
                            quarantined: Vec::new(),
                        });
                }
            }
        }
        if rollback_failed {
            binding.tier_state_poisoned.store(true, Ordering::Release);
        }
        return TrackedApplication {
            outcome,
            h2d_bytes: 0,
            page_ins: 0,
            completed: false,
            ref_underflows: 0,
            byte_underflows: 0,
            unaccounted_bytes: 0,
        };
    }

    let mut h2d_bytes = 0u64;
    let mut page_ins = 0u64;
    let mut touched = std::collections::HashSet::new();
    let mut tiers = binding.tiers.lock().unwrap();
    for transition in committed {
        tiers.insert((transition.value, transition.expert), transition.new);
        touched.insert(transition.value);
        match (transition.old, transition.new) {
            (PhysicalLocation::HostNuma { .. }, PhysicalLocation::Device { .. }) => {
                h2d_bytes = h2d_bytes.saturating_add(transition.len as u64);
                page_ins = page_ins.saturating_add(1);
            }
            (PhysicalLocation::Device { .. }, PhysicalLocation::HostNuma { .. }) => {
                outcome.device_bytes_released = outcome
                    .device_bytes_released
                    .saturating_add(transition.len as u64);
                outcome.host_bytes_committed = outcome
                    .host_bytes_committed
                    .saturating_add(transition.len as u64);
            }
            _ => {}
        }
    }
    outcome.values_touched = touched.len();
    outcome.committed_values.extend(touched);
    TrackedApplication {
        outcome,
        h2d_bytes,
        page_ins,
        completed: true,
        ref_underflows: 0,
        byte_underflows: 0,
        unaccounted_bytes: 0,
    }
}

fn selected_service_bytes(
    binding: &RouteResidencyBoundary,
    routed_experts: &[usize],
) -> (u64, u64, u64) {
    let routed = routed_experts
        .iter()
        .copied()
        .collect::<std::collections::HashSet<_>>();
    let tiers = binding.tiers.lock().unwrap();
    let mut selected = 0u64;
    let mut hits = 0u64;
    let mut cpu_served = 0u64;
    for &value in &binding.bank_values {
        let Some(catalog) = binding.catalogs.get(&value) else {
            continue;
        };
        for &expert in &routed {
            let Some(range) = catalog.relative_range(expert) else {
                continue;
            };
            let bytes = range.end.saturating_sub(range.start) as u64;
            selected = selected.saturating_add(bytes);
            match tiers
                .get(&(value, expert))
                .copied()
                .unwrap_or(PhysicalLocation::Device {
                    ordinal: binding.device_ordinal,
                }) {
                PhysicalLocation::Device { .. } => hits = hits.saturating_add(bytes),
                PhysicalLocation::HostNuma { .. } => cpu_served = cpu_served.saturating_add(bytes),
            }
        }
    }
    (selected, hits, cpu_served)
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
    /// (its kernel never surfaced a window source to the EP). Without a producer
    /// there is no window to consume, so no binding is installed.
    NoTelemetrySource { node: NodeId },
    /// A group member weight has no region catalog — it was not classified/
    /// loaded, so the boundary consumer could not map its regions.
    MissingCatalog { value: ValueId },
    /// A group member weight has no backing VMM allocator — it was not paged/
    /// committed, so the boundary consumer had no allocator to tier against.
    MissingAllocator { value: ValueId },
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
            RouteResidencyBindingReject::MissingCatalog { value } => {
                format!("bank value {value:?} has no region catalog")
            }
            RouteResidencyBindingReject::MissingAllocator { value } => {
                format!("bank value {value:?} has no VMM allocator")
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
        return Err(RouteResidencyBindingReject::NoTelemetrySource { node: group.node });
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

#[allow(clippy::too_many_arguments)]
pub fn build_route_residency_boundary_with_bindings(
    graph: &Graph,
    residency: Arc<CudaWeightResidency>,
    sources: &HashMap<NodeId, Arc<dyn RouteTelemetrySource>>,
    catalogs: HashMap<ValueId, WeightRegionCatalog>,
    bindings: HashMap<ValueId, RouteAllocationBinding>,
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
        |value| bindings.contains_key(&value),
    )?;
    let source = Arc::clone(&sources[&group.node]);
    let bank_values = group.members.clone();
    let boundary = group.boundary;
    Ok(RouteResidencyBoundary::new_with_bindings(
        source,
        residency,
        bank_values,
        boundary,
        catalogs,
        bindings,
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
    install_state: Mutex<RouteResidencyInstallState>,
    successful_applications: AtomicU64,
    selected_bytes: AtomicU64,
    gpu_hit_bytes: AtomicU64,
    h2d_bytes: AtomicU64,
    cpu_served_bytes: AtomicU64,
    page_ins: AtomicU64,
    ref_underflows: AtomicU64,
    byte_underflows: AtomicU64,
    unaccounted_bytes: AtomicU64,
    layers: Mutex<HashMap<(NodeId, ExpertExecutionPhase), ExpertLayerResidencyMetrics>>,
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
        *self.install_state.lock().unwrap() = RouteResidencyInstallState::Installed { banks };
        self.set_install_reason(format!("installed binding over {banks} bank value(s)"));
    }

    /// Record that install fail-closed to no binding, carrying the reason.
    pub(crate) fn record_decline(&self, reason: &str) {
        self.declines.fetch_add(1, Ordering::Relaxed);
        *self.install_state.lock().unwrap() = if reason.contains("gate disabled") {
            RouteResidencyInstallState::GateDisabled
        } else if reason.contains("offload") {
            RouteResidencyInstallState::OffloadDisabled
        } else {
            RouteResidencyInstallState::Rejected(reason.to_string())
        };
        self.set_install_reason(format!("declined: {reason}"));
    }

    fn record_completed(
        &self,
        binding: &RouteResidencyBoundary,
        selected: u64,
        hits: u64,
        cpu_served: u64,
        h2d: u64,
        page_ins: u64,
    ) {
        self.selected_bytes.fetch_add(selected, Ordering::Relaxed);
        self.gpu_hit_bytes.fetch_add(hits, Ordering::Relaxed);
        self.cpu_served_bytes
            .fetch_add(cpu_served, Ordering::Relaxed);
        self.h2d_bytes.fetch_add(h2d, Ordering::Relaxed);
        self.page_ins.fetch_add(page_ins, Ordering::Relaxed);
        let reconciled = selected == hits.saturating_add(cpu_served) && cpu_served == h2d;
        if reconciled {
            self.successful_applications.fetch_add(1, Ordering::Relaxed);
        } else {
            self.unaccounted_bytes.fetch_add(
                selected
                    .abs_diff(hits.saturating_add(cpu_served))
                    .saturating_add(cpu_served.abs_diff(h2d)),
                Ordering::Relaxed,
            );
        }

        let mut layers = self.layers.lock().unwrap();
        let entry = layers
            .entry((binding.node_id, binding.phase))
            .or_insert_with(|| ExpertLayerResidencyMetrics {
                node_id: binding.node_id,
                node_name: binding.node_name.clone(),
                phase: binding.phase,
                selected_bytes: 0,
                gpu_hit_bytes: 0,
                h2d_bytes: 0,
                cpu_served_bytes: 0,
                page_ins: 0,
            });
        entry.selected_bytes = entry.selected_bytes.saturating_add(selected);
        entry.gpu_hit_bytes = entry.gpu_hit_bytes.saturating_add(hits);
        entry.h2d_bytes = entry.h2d_bytes.saturating_add(h2d);
        entry.cpu_served_bytes = entry.cpu_served_bytes.saturating_add(cpu_served);
        entry.page_ins = entry.page_ins.saturating_add(page_ins);
    }

    fn record_accounting_faults(&self, tracked: &TrackedApplication) {
        self.ref_underflows
            .fetch_add(tracked.ref_underflows, Ordering::Relaxed);
        self.byte_underflows
            .fetch_add(tracked.byte_underflows, Ordering::Relaxed);
        self.unaccounted_bytes
            .fetch_add(tracked.unaccounted_bytes, Ordering::Relaxed);
    }

    pub(crate) fn record_observation(
        &self,
        node_id: NodeId,
        node_name: &str,
        phase: ExpertExecutionPhase,
        selected: u64,
    ) {
        self.selected_bytes.fetch_add(selected, Ordering::Relaxed);
        self.gpu_hit_bytes.fetch_add(selected, Ordering::Relaxed);
        let mut layers = self.layers.lock().unwrap();
        let entry = layers
            .entry((node_id, phase))
            .or_insert_with(|| ExpertLayerResidencyMetrics {
                node_id,
                node_name: node_name.to_string(),
                phase,
                selected_bytes: 0,
                gpu_hit_bytes: 0,
                h2d_bytes: 0,
                cpu_served_bytes: 0,
                page_ins: 0,
            });
        entry.selected_bytes = entry.selected_bytes.saturating_add(selected);
        entry.gpu_hit_bytes = entry.gpu_hit_bytes.saturating_add(selected);
    }

    pub fn reset_measurement(&self) {
        self.boundaries.store(0, Ordering::Relaxed);
        self.applied.store(0, Ordering::Relaxed);
        self.rejected.store(0, Ordering::Relaxed);
        self.whole_bank.store(0, Ordering::Relaxed);
        self.empty.store(0, Ordering::Relaxed);
        self.successful_applications.store(0, Ordering::Relaxed);
        self.selected_bytes.store(0, Ordering::Relaxed);
        self.gpu_hit_bytes.store(0, Ordering::Relaxed);
        self.h2d_bytes.store(0, Ordering::Relaxed);
        self.cpu_served_bytes.store(0, Ordering::Relaxed);
        self.page_ins.store(0, Ordering::Relaxed);
        self.ref_underflows.store(0, Ordering::Relaxed);
        self.byte_underflows.store(0, Ordering::Relaxed);
        self.unaccounted_bytes.store(0, Ordering::Relaxed);
        self.layers.lock().unwrap().clear();
        *self.last_reason.lock().unwrap() = None;
    }

    pub fn snapshot(
        &self,
        device_committed_bytes: u64,
        host_committed_bytes: u64,
        oversubscribed_bytes: u64,
    ) -> ExpertResidencyMetrics {
        let mut layers = self
            .layers
            .lock()
            .unwrap()
            .values()
            .cloned()
            .collect::<Vec<_>>();
        layers.sort_by_key(|layer| {
            (
                layer.node_id.0,
                match layer.phase {
                    ExpertExecutionPhase::Prefill => 0u8,
                    ExpertExecutionPhase::Decode => 1u8,
                },
            )
        });
        ExpertResidencyMetrics {
            install_state: self.install_state.lock().unwrap().clone(),
            installs: self.installs.load(Ordering::Relaxed),
            boundaries: self.boundaries.load(Ordering::Relaxed),
            applied_boundaries: self.applied.load(Ordering::Relaxed),
            successful_applications: self.successful_applications.load(Ordering::Relaxed),
            rejected_boundaries: self.rejected.load(Ordering::Relaxed),
            selected_bytes: self.selected_bytes.load(Ordering::Relaxed),
            gpu_hit_bytes: self.gpu_hit_bytes.load(Ordering::Relaxed),
            h2d_bytes: self.h2d_bytes.load(Ordering::Relaxed),
            cpu_served_bytes: self.cpu_served_bytes.load(Ordering::Relaxed),
            page_ins: self.page_ins.load(Ordering::Relaxed),
            device_committed_bytes,
            host_committed_bytes,
            ref_underflows: self.ref_underflows.load(Ordering::Relaxed),
            byte_underflows: self.byte_underflows.load(Ordering::Relaxed),
            oversubscribed_bytes,
            unaccounted_bytes: self.unaccounted_bytes.load(Ordering::Relaxed),
            layers,
            last_reason: self.last_reason(),
        }
    }

    fn set_reason(&self, reason: String) {
        *self.last_reason.lock().unwrap() = Some(reason);
    }

    pub(crate) fn record_rejected(&self, reason: &str) {
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

    let outcome = match prepare_route_window(
        &binding.residency,
        &snapshot,
        binding.expected_epoch(),
        binding.expected_request,
        binding.expected_device,
        &binding.bank_values,
        binding.boundary,
        &binding.catalogs,
        binding.device_count,
    ) {
        Prepared::Early(outcome) => outcome,
        Prepared::Ready {
            plan,
            routed_experts,
            epoch,
            count,
        } => {
            let (selected, hits, cpu_served) = selected_service_bytes(binding, &routed_experts);
            if binding.allocation_offsets.is_empty() {
                let outcome = binding.residency.apply_coarse_residency_plan(
                    &plan,
                    &binding.catalogs,
                    &binding.allocators,
                    &binding.device_pool,
                    &binding.host_pool,
                    binding.device_count,
                    binding.device_ordinal,
                    &binding.expert_groups,
                );
                RouteWindowConsumeOutcome::Applied {
                    routed_experts,
                    epoch,
                    count,
                    outcome: Box::new(outcome),
                }
            } else {
                let tracked = apply_tracked_route_plan(binding, &routed_experts);
                diag.record_accounting_faults(&tracked);
                if !tracked.completed {
                    RouteWindowConsumeOutcome::WholeBank {
                        reason: tracked.outcome.fallback_reason.unwrap_or_else(|| {
                            "tracked route-residency application did not complete".to_string()
                        }),
                    }
                } else {
                    let outcome = tracked.outcome;
                    let h2d_bytes = tracked.h2d_bytes;
                    let page_ins = tracked.page_ins;
                    diag.record_completed(binding, selected, hits, cpu_served, h2d_bytes, page_ins);
                    RouteWindowConsumeOutcome::Applied {
                        routed_experts,
                        epoch,
                        count,
                        outcome: Box::new(outcome),
                    }
                }
            }
        }
    };

    if window_was_consumed(&outcome) {
        binding.source.reset_route_telemetry_boundary()?;
        binding.advance_epoch();
    }

    diag.record_outcome(&outcome);
    Ok(())
}

/// Move every bindable expert range to host physical backing at a verified
/// boundary, then open a fresh telemetry window. Used by deterministic
/// measurement setup after warm-up so the next real route must page selected
/// experts back to device before a later route can hit.
pub fn force_cold_route_residency_boundary(binding: &RouteResidencyBoundary) -> Result<()> {
    if binding.allocation_offsets.is_empty() {
        return Err(onnx_runtime_ep_api::EpError::KernelFailed(
            "force-cold requires production shared-arena allocation bindings".to_string(),
        ));
    }
    let tracked = apply_tracked_route_plan(binding, &[]);
    if !tracked.completed {
        return Err(onnx_runtime_ep_api::EpError::KernelFailed(format!(
            "force-cold expert residency failed closed: {:?}",
            tracked.outcome.fallback_reason
        )));
    }
    binding.source.reset_route_telemetry_boundary()?;
    binding.advance_epoch();
    Ok(())
}

/// Restore every production expert granule to device physical backing before
/// the shared arena is destroyed.
///
/// A mixed-location reservation cannot be handed to the arena's ordinary
/// device-pool disposal path: host-NUMA handles belong to a distinct physical
/// pool and governor tier. Teardown therefore restores the original all-device
/// topology while both pools and the CUDA context are still live.
pub fn restore_device_route_residency_boundary(binding: &RouteResidencyBoundary) -> Result<()> {
    if binding.allocation_offsets.is_empty() {
        return Ok(());
    }
    let experts = binding
        .catalogs
        .values()
        .map(|catalog| catalog.layout().experts)
        .max()
        .unwrap_or(0);
    let routed_experts = (0..experts).collect::<Vec<_>>();
    let tracked = apply_tracked_route_plan(binding, &routed_experts);
    if !tracked.completed {
        return Err(onnx_runtime_ep_api::EpError::KernelFailed(format!(
            "restore-device expert residency failed closed: {:?}",
            tracked.outcome.fallback_reason
        )));
    }
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
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU32, Ordering};

    use onnx_runtime_ep_api::{ExpertExecutionPhase, LazyWeightBoundary};
    use onnx_runtime_ir::{DataType, Graph, NodeId, TensorData, ValueId, WeightRef, static_shape};

    use super::{
        RouteResidencyBindingReject, RouteTelemetryRegistry, RouteTelemetrySource,
        validate_route_residency_binding,
    };

    struct EmptySource;

    impl RouteTelemetrySource for EmptySource {
        fn route_telemetry_snapshot(
            &self,
        ) -> onnx_runtime_ep_api::Result<
            Option<crate::kernels::expert_route_telemetry::TelemetrySnapshot>,
        > {
            Ok(None)
        }

        fn reset_route_telemetry_boundary(&self) -> onnx_runtime_ep_api::Result<()> {
            Ok(())
        }
    }

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
    fn registry_generation_changes_only_when_execution_activates_a_new_source() {
        let registry = RouteTelemetryRegistry::default();
        let first = Arc::new(EmptySource);
        let first_epoch = Arc::new(AtomicU32::new(3));
        registry.activate(
            NodeId(4),
            Arc::clone(&first) as Arc<dyn RouteTelemetrySource>,
            ExpertExecutionPhase::Prefill,
            "moe".to_string(),
            Arc::clone(&first_epoch),
        );
        let (generation, sources) = registry.snapshot();
        assert_eq!(generation, 1);
        assert_eq!(
            sources[&NodeId(4)].expected_epoch.load(Ordering::Acquire),
            3
        );

        registry.activate(
            NodeId(4),
            Arc::clone(&first) as Arc<dyn RouteTelemetrySource>,
            ExpertExecutionPhase::Prefill,
            "moe".to_string(),
            Arc::clone(&first_epoch),
        );
        assert_eq!(registry.snapshot().0, generation);

        let decode = Arc::new(EmptySource);
        let decode_epoch = Arc::new(AtomicU32::new(7));
        registry.activate(
            NodeId(4),
            decode as Arc<dyn RouteTelemetrySource>,
            ExpertExecutionPhase::Decode,
            "moe".to_string(),
            Arc::clone(&decode_epoch),
        );
        let (next_generation, sources) = registry.snapshot();
        assert_eq!(next_generation, generation + 1);
        assert_eq!(sources[&NodeId(4)].phase, ExpertExecutionPhase::Decode);
        assert_eq!(
            sources[&NodeId(4)].expected_epoch.load(Ordering::Acquire),
            7
        );
    }
}
