//! The [`ExecutionProvider`] trait and its supporting types (§4.1).

use std::any::Any;
use std::ffi::c_void;
use std::ptr::NonNull;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use onnx_runtime_ir::{
    DataType, DeviceId, DeviceType, Graph, GraphView, Node, NodeId, NodeIndex, Shape, TensorLayout,
};
use onnx_runtime_memory_governor::{
    AllocationIdentity, ManagedAllocation, MemoryLease, MemoryRole, OwningAllocation,
    ProviderContextIdentity,
};

use crate::epcontext::EpContext;
use crate::error::{EpError, Result};
use crate::kernel::{Kernel, KernelMatch};
use crate::weight::ExecutionProviderCapabilities;

fn allocate_non_reusable_identity(counter: &AtomicU64, exhausted: &'static str) -> u64 {
    counter
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |next| {
            next.checked_add(1)
        })
        .unwrap_or_else(|_| panic!("{exhausted}"))
}

/// Index of an EP within an [`crate::registry::EpRegistry`].
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct EpId(pub u32);

/// Process-unique identity of one executor instance sharing an execution
/// provider.
///
/// A session may own several executors (base decode, decode-inline, MTP verify)
/// over the same `Arc<dyn ExecutionProvider>`. Provider-owned artifacts whose
/// lifetime follows an executor use this identity instead of graph-local
/// [`NodeId`]s, which collide across sibling executors.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ExecutorInstanceId(u64);

impl ExecutorInstanceId {
    /// Reserved identity for direct provider tests and callers that do not own a
    /// session executor.
    pub const UNSCOPED: Self = Self(0);

    /// Allocate a process-unique executor identity under the session's
    /// lifecycle authority.
    pub fn fresh(authority: &ExecutorArtifactSessionAuthority) -> Self {
        let () = authority.private;
        static NEXT: AtomicU64 = AtomicU64::new(1);
        Self(allocate_non_reusable_identity(
            &NEXT,
            "executor instance id space exhausted; refusing to wrap and create an ABA collision",
        ))
    }

    /// Stable numeric representation for provider-owned maps and diagnostics.
    pub fn get(self) -> u64 {
        self.0
    }

    /// Reconstitute an identity stored in provider-owned atomic state.
    #[doc(hidden)]
    pub const fn from_raw(id: u64) -> Self {
        Self(id)
    }
}

/// Process-unique authority for one immutable execution-provider configuration.
///
/// Reconstructing an EP after changing configuration creates a new authority.
/// Executor artifact tokens from the previous provider generation therefore
/// cannot publish or finalize artifacts in the replacement.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ExecutorArtifactConfigAuthority(u64);

impl ExecutorArtifactConfigAuthority {
    /// Reserved authority for providers that own no executor-scoped artifacts.
    pub const UNSCOPED: Self = Self(0);

    /// Allocate a process-unique provider-configuration authority.
    pub fn fresh() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(1);
        Self(allocate_non_reusable_identity(
            &NEXT,
            "executor artifact configuration authority space exhausted; refusing to wrap and \
             create an ABA collision",
        ))
    }

    /// Stable numeric representation for diagnostics and provider validation.
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// Immutable route-residency input resolved before an executor compiles.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum ExecutorRouteResidencyConfig {
    /// Producer publication, telemetry, and request boundaries are forbidden.
    #[default]
    Disabled,
    /// Producer publication is permitted; finalization may still decline when
    /// the graph or provider artifacts cannot support route residency.
    Enabled,
}

/// Whether a kernel factory requires a session-issued executor scope.
///
/// Ordinary providers and kernels remain [`Unscoped`](Self::Unscoped). A
/// provider returns [`Required`](Self::Required) only for a factory that
/// publishes executor-generation-owned artifacts while compiling. Calling
/// [`ExecutionProvider::get_kernel`] for such a factory must fail clearly;
/// session compilation uses [`ExecutionProvider::get_kernel_for_executor`]
/// with the one capability issued for the executor generation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ExecutorKernelScope {
    #[default]
    Unscoped,
    Required,
}

/// Opaque capability held only by the crate that owns executor lifecycle
/// issuance.
///
/// Cargo features deliberately do not grant this authority: features are
/// additive across a dependency graph and therefore cannot identify the
/// caller that is allowed to issue a session capability.
///
/// ```compile_fail
/// # use onnx_runtime_ep_api::ExecutorArtifactSessionAuthority;
/// let _forged = ExecutorArtifactSessionAuthority { private: () };
/// ```
pub struct ExecutorArtifactSessionAuthority {
    private: (),
}

/// Provider-owned half of an executor artifact configuration.
///
/// A provider resolves only its own authority and immutable feature policy.
/// The session binds this template to the executor identity and a fresh
/// generation, so a provider cannot choose the owner accepted by the session.
///
/// Receiving a template and executor identity is insufficient to bind the
/// capability without the session's private issuer:
///
/// ```compile_fail
/// # use onnx_runtime_ep_api::{
/// #     ExecutorArtifactConfigTemplate, ExecutorInstanceId,
/// # };
/// fn forge(template: ExecutorArtifactConfigTemplate, executor: ExecutorInstanceId) {
///     let _forged = template.bind(executor);
/// }
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ExecutorArtifactConfigTemplate {
    authority: ExecutorArtifactConfigAuthority,
    device: DeviceId,
    route_residency: ExecutorRouteResidencyConfig,
}

impl ExecutorArtifactConfigTemplate {
    /// Construct a provider-resolved artifact configuration template.
    #[doc(hidden)]
    pub const fn resolved(
        authority: ExecutorArtifactConfigAuthority,
        device: DeviceId,
        route_residency: ExecutorRouteResidencyConfig,
    ) -> Self {
        Self {
            authority,
            device,
            route_residency,
        }
    }

    pub const fn device(self) -> DeviceId {
        self.device
    }

    pub const fn route_residency(self) -> ExecutorRouteResidencyConfig {
        self.route_residency
    }

    /// Bind this provider policy to one session-issued executor generation.
    #[must_use]
    pub fn bind(
        self,
        authority: &ExecutorArtifactSessionAuthority,
        executor: ExecutorInstanceId,
    ) -> ExecutorArtifactConfig {
        let () = authority.private;
        ExecutorArtifactConfig {
            authority: self.authority,
            executor,
            generation: ExecutorArtifactGeneration::fresh(),
            device: self.device,
            route_residency: self.route_residency,
        }
    }
}

/// Process-unique generation of one executor artifact configuration.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ExecutorArtifactGeneration(u64);

impl ExecutorArtifactGeneration {
    fn fresh() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(1);
        Self(allocate_non_reusable_identity(
            &NEXT,
            "executor artifact generation space exhausted; refusing to wrap and create an ABA \
             collision",
        ))
    }

    /// Stable numeric representation for diagnostics.
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// Typed immutable configuration for one executor/provider generation.
///
/// The same token is required at kernel-producer publication, artifact
/// finalization, and teardown. Environment variables may be inputs when the EP
/// is constructed, but are never consulted to mutate this token afterwards.
///
/// Receiving a bound configuration for diagnostics is insufficient to issue a
/// finalization proof without the session's private issuer:
///
/// ```compile_fail
/// # use onnx_runtime_ep_api::{
/// #     ExecutorArtifactConfig, ExecutorArtifactReadinessEpoch,
/// # };
/// fn forge(config: ExecutorArtifactConfig) {
///     let _proof = config.finalization_proof(ExecutorArtifactReadinessEpoch::INITIAL);
/// }
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ExecutorArtifactConfig {
    authority: ExecutorArtifactConfigAuthority,
    executor: ExecutorInstanceId,
    generation: ExecutorArtifactGeneration,
    device: DeviceId,
    route_residency: ExecutorRouteResidencyConfig,
}

impl ExecutorArtifactConfig {
    pub const fn authority(self) -> ExecutorArtifactConfigAuthority {
        self.authority
    }

    pub const fn executor(self) -> ExecutorInstanceId {
        self.executor
    }

    pub const fn generation(self) -> ExecutorArtifactGeneration {
        self.generation
    }

    pub const fn device(self) -> DeviceId {
        self.device
    }

    pub const fn route_residency(self) -> ExecutorRouteResidencyConfig {
        self.route_residency
    }

    /// Issue a scoped finalization proof for exactly this readiness epoch.
    #[must_use]
    pub fn finalization_proof(
        &self,
        authority: &ExecutorArtifactSessionAuthority,
        readiness: ExecutorArtifactReadinessEpoch,
    ) -> ExecutorArtifactFinalizationProof<'_> {
        let () = authority.private;
        match self.route_residency {
            ExecutorRouteResidencyConfig::Disabled => {
                ExecutorArtifactFinalizationProof::Disabled(ExecutorArtifactDisabledFinalization {
                    config: self,
                    readiness,
                })
            }
            ExecutorRouteResidencyConfig::Enabled => {
                ExecutorArtifactFinalizationProof::Enabled(ExecutorArtifactEnabledFinalization {
                    config: self,
                    readiness,
                })
            }
        }
    }
}

/// Monotonic executor-local epoch for concrete kernel/producer readiness.
///
/// The session advances this at the kernel-cache publication chokepoint whenever
/// any build, binding-preparation, or runtime-dispatch path creates a new
/// specialization. A provider that returns a pending proof is not called again
/// for the same epoch: another attempt requires a concrete compilation
/// transition.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ExecutorArtifactReadinessEpoch(u64);

impl ExecutorArtifactReadinessEpoch {
    pub const INITIAL: Self = Self(0);

    pub const fn new(epoch: u64) -> Self {
        Self(epoch)
    }

    pub const fn get(self) -> u64 {
        self.0
    }
}

/// Typed reason provider-artifact finalization cannot yet reach a terminal
/// outcome.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ExecutorArtifactPending {
    /// A graph node that requires an execution-time producer has not published
    /// it for this executor yet.
    ProducerUnavailable { node: NodeId },
    /// Provider-specific readiness which is not represented by a graph node.
    ProviderReadiness { reason: String },
}

impl ExecutorArtifactPending {
    pub fn reason(&self) -> String {
        match self {
            Self::ProducerUnavailable { node } => {
                format!("producer for graph node {node:?} is not registered")
            }
            Self::ProviderReadiness { reason } => reason.clone(),
        }
    }
}

/// Scoped proof offered to a provider for one exact executor generation and
/// readiness epoch.
///
/// The variants expose only legal constructors: a disabled generation can
/// complete only as disabled, while an enabled generation can require a
/// boundary, explicitly decline, or remain pending.
pub enum ExecutorArtifactFinalizationProof<'a> {
    Disabled(ExecutorArtifactDisabledFinalization<'a>),
    Enabled(ExecutorArtifactEnabledFinalization<'a>),
}

impl ExecutorArtifactFinalizationProof<'_> {
    pub fn config(&self) -> ExecutorArtifactConfig {
        match self {
            Self::Disabled(proof) => proof.config(),
            Self::Enabled(proof) => proof.config(),
        }
    }

    pub fn readiness(&self) -> ExecutorArtifactReadinessEpoch {
        match self {
            Self::Disabled(proof) => proof.readiness(),
            Self::Enabled(proof) => proof.readiness(),
        }
    }
}

/// Finalization authority for an immutable disabled generation.
///
/// A disabled proof deliberately has no `required`, `declined`, or `pending`
/// constructor:
///
/// ```compile_fail
/// # use onnx_runtime_ep_api::ExecutorArtifactDisabledFinalization;
/// fn forge(disabled: ExecutorArtifactDisabledFinalization<'_>) {
///     disabled.required();
/// }
/// ```
pub struct ExecutorArtifactDisabledFinalization<'a> {
    config: &'a ExecutorArtifactConfig,
    readiness: ExecutorArtifactReadinessEpoch,
}

impl ExecutorArtifactDisabledFinalization<'_> {
    pub fn config(&self) -> ExecutorArtifactConfig {
        *self.config
    }

    pub fn readiness(&self) -> ExecutorArtifactReadinessEpoch {
        self.readiness
    }

    #[must_use]
    pub fn complete(self) -> ExecutorArtifactFinalization {
        ExecutorArtifactFinalization::new(
            *self.config,
            self.readiness,
            ExecutorArtifactFinalizationOutcome::Complete {
                route_residency: ExecutorRouteResidency::Disabled,
            },
        )
    }
}

/// Finalization authority for an immutable enabled generation.
pub struct ExecutorArtifactEnabledFinalization<'a> {
    config: &'a ExecutorArtifactConfig,
    readiness: ExecutorArtifactReadinessEpoch,
}

impl ExecutorArtifactEnabledFinalization<'_> {
    pub fn config(&self) -> ExecutorArtifactConfig {
        *self.config
    }

    pub fn readiness(&self) -> ExecutorArtifactReadinessEpoch {
        self.readiness
    }

    #[must_use]
    pub fn required(self) -> ExecutorArtifactFinalization {
        ExecutorArtifactFinalization::new(
            *self.config,
            self.readiness,
            ExecutorArtifactFinalizationOutcome::Complete {
                route_residency: ExecutorRouteResidency::required_for(self.config.executor),
            },
        )
    }

    #[must_use]
    pub fn declined(self) -> ExecutorArtifactFinalization {
        ExecutorArtifactFinalization::new(
            *self.config,
            self.readiness,
            ExecutorArtifactFinalizationOutcome::Complete {
                route_residency: ExecutorRouteResidency::Declined,
            },
        )
    }

    #[must_use]
    pub fn pending(self, pending: ExecutorArtifactPending) -> ExecutorArtifactFinalization {
        ExecutorArtifactFinalization::new(
            *self.config,
            self.readiness,
            ExecutorArtifactFinalizationOutcome::Pending(pending),
        )
    }
}

/// Opaque provider response minted from a scoped finalization proof.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExecutorArtifactFinalization {
    config: ExecutorArtifactConfig,
    readiness: ExecutorArtifactReadinessEpoch,
    outcome: ExecutorArtifactFinalizationOutcome,
}

impl ExecutorArtifactFinalization {
    fn new(
        config: ExecutorArtifactConfig,
        readiness: ExecutorArtifactReadinessEpoch,
        outcome: ExecutorArtifactFinalizationOutcome,
    ) -> Self {
        Self {
            config,
            readiness,
            outcome,
        }
    }

    /// Verify that this response was minted for the exact capability and epoch
    /// currently held by the session.
    pub fn resolve(
        self,
        expected: ExecutorArtifactConfig,
        readiness: ExecutorArtifactReadinessEpoch,
    ) -> Result<ExecutorArtifactFinalizationOutcome> {
        if self.config != expected || self.readiness != readiness {
            return Err(EpError::KernelFailed(format!(
                "executor artifact finalization proof mismatch: returned authority {} device {:?} \
                 executor {} generation {} epoch {}, expected authority {} device {:?} executor {} \
                 generation {} epoch {}",
                self.config.authority().get(),
                self.config.device(),
                self.config.executor().get(),
                self.config.generation().get(),
                self.readiness.get(),
                expected.authority().get(),
                expected.device(),
                expected.executor().get(),
                expected.generation().get(),
                readiness.get(),
            )));
        }
        Ok(self.outcome)
    }
}

/// Session-validated result of executor artifact finalization.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ExecutorArtifactFinalizationOutcome {
    /// Every provider artifact required by this executor reached an honest
    /// terminal outcome.
    Complete {
        route_residency: ExecutorRouteResidency,
    },
    /// A readiness-dependent producer is not available yet. Nothing terminal
    /// was latched.
    Pending(ExecutorArtifactPending),
}

/// Resolved route-residency behavior for one executor.
///
/// This is produced only by artifact finalization, before request execution.
/// The hot path consumes this value directly, so the default-off and declined
/// paths cannot acquire provider locks, inspect producer registries, or perform
/// telemetry work.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ExecutorRouteResidency {
    /// Route residency is disabled by resolved provider configuration.
    #[default]
    Disabled,
    /// Route residency was enabled, but this executor structurally declined
    /// installation and therefore owns no request boundary.
    Declined,
    /// This executor owns a live request boundary that must run after its exact
    /// validation receipt is consumed.
    Required { owner: ExecutorInstanceId },
}

impl ExecutorRouteResidency {
    #[must_use]
    pub fn required_for(owner: ExecutorInstanceId) -> Self {
        Self::Required { owner }
    }

    #[must_use]
    pub fn owner(self) -> Option<ExecutorInstanceId> {
        match self {
            Self::Required { owner } => Some(owner),
            Self::Disabled | Self::Declined => None,
        }
    }

    #[must_use]
    pub fn is_required(self) -> bool {
        self.owner().is_some()
    }
}

/// Tie-break policy for [`ExecutionProvider::device_argmax`] when two or more
/// logits share the maximum value.
///
/// The default ([`ArgmaxTieBreak::LowestIndex`]) matches the canonical ONNX
/// `ArgMax` operator (`select_last_index=false`) and the host greedy references
/// `sample_greedy` / `argmax_logits_tensor` ("ties keep the lowest token id"),
/// which is the base-decode / ORT byte-identity contract.
/// [`ArgmaxTieBreak::HighestIndex`] instead keeps the highest token id on ties,
/// matching Rust's `Iterator::max_by` (returns the LAST maximal element) as used
/// by the engine/reference greedy `max_by` probes, and ONNX `ArgMax` with
/// `select_last_index=true`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum ArgmaxTieBreak {
    /// Ties resolve to the lowest token id (first maximal element).
    #[default]
    LowestIndex,
    /// Ties resolve to the highest token id (last maximal element).
    HighestIndex,
}

impl ArgmaxTieBreak {
    /// Whether ties select the LAST (highest-index) maximal element.
    #[must_use]
    pub fn select_last_index(self) -> bool {
        matches!(self, ArgmaxTieBreak::HighestIndex)
    }
}

/// Which captured device-graph slot an EP graph operation targets.
///
/// A decode EP historically owns exactly one captured graph (the `Primary`
/// slot): the shape-invariant M=1 decode step it replays every token. MTP
/// self-speculative decode adds a *second*, differently-shaped forward — the
/// fixed-width `M = k+1` verify step — that must be captured and replayed
/// independently of the M=1 step, because the two graphs bake different query
/// geometries and cannot share one slot without invalidating each other every
/// step (the empirically-measured `replays=0` MTP blocker; see
/// `gaff-mtp-graph-retain-capture-unsafe-blocker.md`). `Verify` names that
/// second slot so the executor can hold and replay both graphs by shape key on
/// the same EP/stream (one CUDA graph per shape, no per-step recapture).
///
/// EPs that support only a single captured graph (the historical behaviour)
/// accept `Primary` and reject `Verify`; the CUDA EP owns one
/// [`CudaGraphLifecycle`](../../onnx_runtime_ep_cuda) per slot.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum DeviceGraphSlot {
    /// The shape-invariant M=1 decode graph (the only slot before MTP).
    #[default]
    Primary,
    /// The fixed-width `M = k+1` speculative verify graph.
    Verify,
}

impl DeviceGraphSlot {
    /// Number of distinct captured-graph slots. Used to size per-slot host
    /// capture-state arrays on the executor so `Primary` (M=1 decode) and
    /// `Verify` (M=k+1) graphs can coexist without clobbering each other.
    pub const COUNT: usize = 2;

    /// Dense array index for this slot (`Primary` = 0, `Verify` = 1). `Primary`
    /// is index 0 so the historical single-slot code path — which only ever
    /// touches `Primary` — maps to slot 0 and stays byte-identical.
    #[inline]
    pub const fn index(self) -> usize {
        match self {
            DeviceGraphSlot::Primary => 0,
            DeviceGraphSlot::Verify => 1,
        }
    }
}

/// Immutable identity of one executor's device-graph namespace.
///
/// A provider may be shared by several sessions. The owner prevents one
/// executor's `Primary` or `Verify` graph from naming another executor's slot.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct DeviceGraphOwner(u64);

impl DeviceGraphOwner {
    /// Mint a process-unique owner identity. Identities are never reused.
    pub fn new() -> Self {
        static NEXT_OWNER: AtomicU64 = AtomicU64::new(1);
        let owner = NEXT_OWNER
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |next| {
                next.checked_add(1)
            })
            .unwrap_or_else(|_| {
                panic!(
                    "device validation owner identity space exhausted; refusing to wrap and \
                     create an ABA collision"
                )
            });
        Self(owner)
    }

    /// Stable process-local numeric identity.
    pub const fn get(self) -> u64 {
        self.0
    }
}

impl Default for DeviceGraphOwner {
    fn default() -> Self {
        Self::new()
    }
}

/// Immutable identity of one executor's deferred-validation namespace.
///
/// A provider may be shared by several sessions. Only the executor that opened
/// a validation generation, or an output binding carrying its exact token, may
/// consume that generation's result.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct DeviceValidationOwner(u64);

impl DeviceValidationOwner {
    /// Mint a process-unique owner identity. Identities are never reused.
    pub fn new() -> Self {
        static NEXT_OWNER: AtomicU64 = AtomicU64::new(1);
        let owner = NEXT_OWNER
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |next| {
                next.checked_add(1)
            })
            .unwrap_or_else(|_| {
                panic!(
                    "device validation owner identity space exhausted; refusing to wrap and \
                     create an ABA collision"
                )
            });
        Self(owner)
    }

    /// Stable process-local numeric identity.
    pub const fn get(self) -> u64 {
        self.0
    }
}

impl Default for DeviceValidationOwner {
    fn default() -> Self {
        Self::new()
    }
}

/// Setup-time proof that one deferred-validation owner is registered with an EP.
///
/// The provider-specific state is allocated only here. Submission paths borrow
/// this proof, so they neither look up an owner in a map nor clone a reference-
/// counted handle.
pub struct DeviceValidationRegistration {
    owner: DeviceValidationOwner,
    state: Box<dyn Any + Send + Sync>,
}

impl DeviceValidationRegistration {
    /// Construct a registration carrying provider-specific state.
    pub fn new<T>(owner: DeviceValidationOwner, state: T) -> Self
    where
        T: Any + Send + Sync,
    {
        Self {
            owner,
            state: Box::new(state),
        }
    }

    /// Registered owner identity.
    pub const fn owner(&self) -> DeviceValidationOwner {
        self.owner
    }

    /// Borrow provider-specific registration state.
    #[doc(hidden)]
    pub fn state<T: Any>(&self) -> Option<&T> {
        self.state.downcast_ref()
    }

    /// Mutably borrow provider-specific registration state during teardown.
    #[doc(hidden)]
    pub fn state_mut<T: Any>(&mut self) -> Option<&mut T> {
        self.state.downcast_mut()
    }
}

impl std::fmt::Debug for DeviceValidationRegistration {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("DeviceValidationRegistration")
            .field("owner", &self.owner)
            .finish_non_exhaustive()
    }
}

/// Exact identity of one submitted deferred-validation generation.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct DeviceValidationToken {
    owner: DeviceValidationOwner,
    generation: u64,
}

impl DeviceValidationToken {
    /// Construct a provider-issued validation token.
    pub const fn new(owner: DeviceValidationOwner, generation: u64) -> Self {
        Self { owner, generation }
    }

    pub const fn owner(self) -> DeviceValidationOwner {
        self.owner
    }

    pub const fn generation(self) -> u64 {
        self.generation
    }
}

/// Exact identity of one installed device-graph generation.
///
/// All replay, liveness, reset, and invalidation operations require this token.
/// Re-capture mints a new generation even for the same executor and slot.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct DeviceGraphToken {
    owner: DeviceGraphOwner,
    slot: DeviceGraphSlot,
    generation: u64,
}

impl DeviceGraphToken {
    /// Construct a provider-issued installation token.
    pub const fn new(owner: DeviceGraphOwner, slot: DeviceGraphSlot, generation: u64) -> Self {
        Self {
            owner,
            slot,
            generation,
        }
    }

    pub const fn owner(self) -> DeviceGraphOwner {
        self.owner
    }

    pub const fn slot(self) -> DeviceGraphSlot {
        self.slot
    }

    pub const fn generation(self) -> u64 {
        self.generation
    }
}

/// Opaque, namespaced configuration passed to [`ExecutionProvider::initialize`].
#[derive(Clone, Debug, Default)]
pub struct EpConfig {
    /// Namespaced key/value options (e.g. `"cuda.arena_extend_strategy"`).
    pub options: std::collections::HashMap<String, String>,
}

/// An owning handle to a single device allocation.
///
/// # Ownership & lifetime
///
/// A `DeviceBuffer` is the **sole owner** of the allocation it names. It is
/// produced only by [`ExecutionProvider::allocate`] and released only by
/// [`ExecutionProvider::deallocate`], which consumes it *by value*. The owning
/// EP is both allocator and deallocator: the buffer records the [`DeviceId`]
/// (hence which EP instance) that may free it, so a buffer must never be handed
/// to a different EP. Ownership is unique — no two `DeviceBuffer`s ever alias
/// the same allocation.
///
/// # Two owning representations
///
/// * **Raw owning** ([`DeviceBuffer::from_raw_parts`]) — an address plus size
///   and alignment. The EP is trusted to pair it with exactly one free. This is
///   what the CPU EP and adapter/plugin paths use.
/// * **Bound owning** ([`DeviceBuffer::from_owning_allocation`]) — the exact
///   [`OwningAllocation`] that minted the address, carrying its binding identity
///   and allocation generation. Final release goes back through that owner, so
///   a stale handle over a reused address cannot free anything, and a release
///   can never be attributed to the wrong mechanism.
///
/// A bound buffer is still non-`Clone` and still has no `Drop`: the generation
/// is what makes the release safe, and the owner's own `Drop` quarantines
/// rather than frees.
///
/// # No `Drop`
///
/// `DeviceBuffer` deliberately does **not** implement [`Drop`]. Freeing device
/// memory generally needs the EP's context/stream (a CUDA context, an MLX
/// queue, an allocator arena) that this bare handle does not carry, so a silent
/// drop could not free correctly. Consequences:
/// * Dropping a `DeviceBuffer` without passing it to `deallocate` **leaks** the
///   allocation (a bound buffer instead quarantines it, so the bytes stay
///   accounted for). It can never *double-free*, which is the memory-safety
///   property we prioritize (plan §4.4).
/// * The session layer owns the discipline of pairing every `allocate` with
///   exactly one `deallocate`. Higher layers may wrap this handle in an
///   RAII/`Arc` type that calls back into the EP; that policy lives above the
///   EP contract, not here.
///
/// # Access
///
/// The base address is reachable only through [`DeviceBuffer::as_ptr`]
/// (shared) and [`DeviceBuffer::as_mut_ptr`] (unique). Obtaining a pointer is
/// safe; *dereferencing* it is `unsafe` and valid only on host-accessible
/// devices ([`DeviceType::is_host_accessible`]) within the owning EP's context.
///
/// # Thread-safety
///
/// See the `Send`/`Sync` impls below for the exact invariant.
#[derive(Debug)]
pub struct DeviceBuffer {
    device: DeviceId,
    size: usize,
    align: usize,
    /// Non-null base address of the allocation. For CPU and MLX unified memory
    /// this is a dereferenceable host pointer; for CUDA/ROCm it is an opaque
    /// device address only meaningful inside the owning EP's context.
    ptr: NonNull<c_void>,
    /// Whether this handle *owns* the pointed-to allocation.
    ///
    /// [`BufferOwner::Owned`] (the default for [`DeviceBuffer::from_raw_parts`])
    /// is the original contract: the owning EP must free it exactly once in
    /// `deallocate`. [`BufferOwner::Bound`] carries the generation-checked
    /// owner instead of trusting the raw triple. Borrowed handles alias memory
    /// owned by *someone else*. Read-only aliases come from
    /// [`DeviceBuffer::from_borrowed_parts`]; exclusive writable aliases come
    /// from [`DeviceBuffer::from_borrowed_mut_parts`]. `deallocate` must **not**
    /// free either borrowed kind.
    owner: BufferOwner,
}

/// Whether a [`DeviceBuffer`] owns the allocation it names, or merely borrows
/// (aliases) memory owned elsewhere.
#[derive(Debug)]
enum BufferOwner {
    /// This handle is the sole owner; the owning EP frees it in `deallocate`.
    Owned,
    /// This handle is the sole owner *and* carries the binding-issued owner
    /// that minted the address. Release consumes that owner, so it is validated
    /// against the binding identity and the allocation generation.
    Bound(Box<OwningAllocation>),
    /// Binding-issued ownership whose authority/process charges remain pinned
    /// until the Phase-4 structured release outcome settles.
    Managed(Box<ManagedAllocation>),
    /// This handle aliases foreign memory (e.g. an mmap). `deallocate` must be
    /// a no-op free; the real owner must outlive the buffer and every use of it.
    Borrowed,
    /// This handle has temporary exclusive write access to an allocation owned
    /// elsewhere. Deallocation remains a no-op.
    BorrowedMut,
}

impl DeviceBuffer {
    /// Wrap a raw device allocation in an owning handle.
    ///
    /// # Safety
    ///
    /// The caller (the owning EP) must guarantee all of:
    /// * `ptr` is non-null and points to the start of an allocation of at least
    ///   `size` bytes on `device`, aligned to at least `align` bytes.
    /// * The allocation was produced by `device`'s EP and will be freed exactly
    ///   once, only by returning this handle to that EP's `deallocate` (or via
    ///   an equivalent raw free of the pointer obtained from
    ///   [`DeviceBuffer::into_raw`]).
    /// * No other live `DeviceBuffer` aliases the same allocation.
    ///
    /// `align` must be a power of two (checked in debug builds).
    pub unsafe fn from_raw_parts(
        ptr: *mut c_void,
        device: DeviceId,
        size: usize,
        align: usize,
    ) -> Self {
        debug_assert!(align.is_power_of_two(), "alignment must be a power of two");
        Self {
            device,
            size,
            align,
            ptr: NonNull::new(ptr).expect("DeviceBuffer::from_raw_parts: null pointer"),
            owner: BufferOwner::Owned,
        }
    }

    /// Wrap **foreign, borrowed** memory in a non-owning `DeviceBuffer`.
    ///
    /// Unlike [`DeviceBuffer::from_raw_parts`], the returned handle does **not**
    /// own the allocation: it aliases memory owned by someone else (for example
    /// a `memmap2::Mmap` over an on-disk weight file). This lets an EP reference
    /// initializer bytes zero-copy instead of allocating + copying them into
    /// fresh RAM.
    ///
    /// [`is_borrowed`](DeviceBuffer::is_borrowed) returns `true`, and the owning
    /// EP's `deallocate` must treat it as a **no-op free** (the guard checks
    /// `is_borrowed()`). [`into_raw`](DeviceBuffer::into_raw) still yields the
    /// raw pointer, but the caller must **not** free it.
    ///
    /// # Safety
    ///
    /// The caller must guarantee all of:
    /// * `ptr` is non-null and points to the start of a readable region of at
    ///   least `size` bytes on `device`, aligned to at least `align` bytes.
    /// * The memory is owned by another object (e.g. an mmap) that **outlives
    ///   this buffer and every use of it** (read via `as_ptr`). Nothing else may
    ///   free or unmap it while this handle or any alias derived from it lives.
    /// * The buffer is treated as **read-only**: it is never written through
    ///   (`as_mut_ptr` must not be used to mutate borrowed memory) and is never
    ///   passed to an EP's `deallocate` expecting a free — `deallocate` skips
    ///   the free for borrowed buffers.
    ///
    /// `align` must be a power of two (checked in debug builds).
    pub unsafe fn from_borrowed_parts(
        ptr: *mut c_void,
        device: DeviceId,
        size: usize,
        align: usize,
    ) -> Self {
        debug_assert!(align.is_power_of_two(), "alignment must be a power of two");
        Self {
            device,
            size,
            align,
            ptr: NonNull::new(ptr).expect("DeviceBuffer::from_borrowed_parts: null pointer"),
            owner: BufferOwner::Borrowed,
        }
    }

    /// Wrap foreign memory in a non-owning, exclusively writable buffer handle.
    ///
    /// This is intended for persistent external output bindings: the real owner
    /// retains the allocation while an executor temporarily writes through this
    /// alias.
    ///
    /// # Safety
    ///
    /// The caller must guarantee all of:
    /// * `ptr` names a non-null writable allocation of at least `size` bytes on
    ///   `device`, aligned to at least `align` bytes.
    /// * The real owner outlives this handle and every operation using it.
    /// * No other writer accesses the allocation while this handle is live.
    /// * This handle is never used to free the allocation; `deallocate` treats
    ///   it as borrowed.
    pub unsafe fn from_borrowed_mut_parts(
        ptr: *mut c_void,
        device: DeviceId,
        size: usize,
        align: usize,
    ) -> Option<Self> {
        debug_assert!(align.is_power_of_two(), "alignment must be a power of two");
        Some(Self {
            device,
            size,
            align,
            ptr: NonNull::new(ptr)?,
            owner: BufferOwner::BorrowedMut,
        })
    }

    /// Whether this handle merely *borrows* (aliases) foreign memory rather than
    /// owning it. A borrowed buffer must never be freed by `deallocate`.
    pub fn is_borrowed(&self) -> bool {
        matches!(self.owner, BufferOwner::Borrowed | BufferOwner::BorrowedMut)
    }

    /// Wrap a **binding-issued owning allocation** in a `DeviceBuffer`.
    ///
    /// The buffer's address, size, and alignment are taken from `owner`, so a
    /// bound buffer can never describe a different region than the owner it
    /// carries. Final release consumes that owner
    /// ([`into_bound_owner`](Self::into_bound_owner)), which matches the binding
    /// identity and the allocation generation before anything is freed: a stale
    /// handle over a reused device address is refused instead of freeing a live
    /// allocation.
    ///
    /// This is safe to call — the safety obligations were discharged when the
    /// binding issued (or adopted) the allocation.
    pub fn from_owning_allocation(owner: OwningAllocation, device: DeviceId) -> Self {
        let ptr = owner.as_ptr();
        let size = owner.len();
        let align = owner.alignment().max(1);
        Self {
            device,
            size,
            align,
            ptr: NonNull::new(ptr.as_ptr().cast::<c_void>())
                .expect("an owning allocation holds a non-null address"),
            owner: BufferOwner::Bound(Box::new(owner)),
        }
    }

    /// Wrap a process-manager transaction result in a device buffer.
    ///
    /// The manager settlement token stays inseparable from physical ownership;
    /// consuming release must use [`into_bound_ownership`](Self::into_bound_ownership).
    pub fn from_managed_allocation(owner: ManagedAllocation, device: DeviceId) -> Self {
        let ptr = owner.as_ptr();
        let size = owner.len();
        let align = owner.alignment().max(1);
        Self {
            device,
            size,
            align,
            ptr: NonNull::new(ptr.as_ptr().cast::<c_void>())
                .expect("a managed allocation holds a non-null address"),
            owner: BufferOwner::Managed(Box::new(owner)),
        }
    }

    /// Whether this handle carries a binding-issued owner whose release is
    /// generation-validated.
    pub fn is_bound(&self) -> bool {
        matches!(self.owner, BufferOwner::Bound(_) | BufferOwner::Managed(_))
    }

    /// Borrow the binding-issued owner, for a bound capability call (commit,
    /// decommit, mapped-byte queries) that must be validated against the
    /// allocation generation.
    ///
    /// Returns `None` for raw-owning and borrowed buffers, which is how a
    /// generation-checked path fails closed on foreign memory.
    pub fn bound_owner(&self) -> Option<&OwningAllocation> {
        match &self.owner {
            BufferOwner::Bound(owner) => Some(owner),
            BufferOwner::Managed(owner) => Some(owner.owner_ref()),
            _ => None,
        }
    }

    /// Borrow process-manager ownership when this buffer carries it.
    pub fn managed_owner(&self) -> Option<&ManagedAllocation> {
        match &self.owner {
            BufferOwner::Managed(owner) => Some(owner),
            _ => None,
        }
    }

    /// Allocation-specific release settlement when this buffer is manager-owned.
    pub fn managed_settlement_wait(
        &self,
    ) -> Option<onnx_runtime_memory_governor::AllocationSettlementWait> {
        self.managed_owner().map(ManagedAllocation::settlement_wait)
    }

    /// Consume the handle and recover its complete binding-issued ownership.
    ///
    /// This is the only way a bound buffer's ownership leaves the handle, and it
    /// is what a provider's `deallocate` calls before preparing or deferring the
    /// physical release. `Err` hands the buffer back untouched when it is not
    /// bound, so a caller that requires generation-checked release can refuse
    /// foreign memory without losing it.
    pub fn into_bound_owner(self) -> std::result::Result<BoundBufferOwnership, Self> {
        match self.owner {
            BufferOwner::Bound(owner) => Ok(BoundBufferOwnership::Binding(*owner)),
            BufferOwner::Managed(owner) => Ok(BoundBufferOwnership::Managed(*owner)),
            owner => Err(Self { owner, ..self }),
        }
    }

    /// Consume either binding-issued owning representation without discarding a
    /// process-manager settlement token.
    pub fn into_bound_ownership(self) -> std::result::Result<BoundBufferOwnership, Self> {
        self.into_bound_owner()
    }

    /// The device this allocation lives on (and whose EP must free it).
    pub fn device(&self) -> DeviceId {
        self.device
    }

    /// Allocation size in bytes.
    pub fn len(&self) -> usize {
        self.size
    }

    /// Whether the allocation is zero-length.
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    /// Alignment (bytes) the base pointer was allocated to.
    pub fn alignment(&self) -> usize {
        self.align
    }

    /// Shared base pointer. Safe to obtain; dereferencing is `unsafe` and only
    /// sound on host-accessible devices within the owning EP's context.
    pub fn as_ptr(&self) -> *const c_void {
        self.ptr.as_ptr()
    }

    /// Unique mutable base pointer. Requires `&mut self` so the borrow checker
    /// forbids two writers sharing one buffer — this is what makes the `Sync`
    /// impl sound (a shared `&DeviceBuffer` can never hand out a writable
    /// pointer through safe code).
    pub fn as_mut_ptr(&mut self) -> *mut c_void {
        self.ptr.as_ptr()
    }

    /// Consume the handle, returning the raw pointer *without* freeing it. For
    /// an owned buffer the caller assumes the single-free obligation from
    /// [`DeviceBuffer::from_raw_parts`]. For a **borrowed** buffer (see
    /// [`DeviceBuffer::from_borrowed_parts`]) the pointer must **not** be freed;
    /// check [`is_borrowed`](DeviceBuffer::is_borrowed) first if the caller
    /// intends to free.
    ///
    /// # Panics
    ///
    /// For a **bound** buffer ([`DeviceBuffer::from_owning_allocation`]) this
    /// panics rather than silently downgrading generation-checked ownership to
    /// a bare address. Handing out the raw pointer alone would let a caller free
    /// it without matching the binding identity or the allocation generation —
    /// exactly the stale-pointer free the binding exists to prevent — while the
    /// owner it left behind would quarantine the same bytes. Use
    /// [`into_raw_with_owner`](Self::into_raw_with_owner) when the raw address
    /// *and* the owner are both wanted, or
    /// [`into_bound_owner`](Self::into_bound_owner) to take ownership back.
    pub fn into_raw(self) -> *mut c_void {
        assert!(
            !self.is_bound(),
            "DeviceBuffer::into_raw: this buffer carries a binding-issued owning allocation, so \
             returning the raw pointer alone would bypass binding-identity and allocation-\
             generation validation on release. Use into_raw_with_owner or into_bound_owner."
        );
        self.ptr.as_ptr()
    }

    /// Consume the handle, returning the raw pointer **together with** the
    /// binding-issued owner when there is one.
    ///
    /// This is the explicit escape hatch [`into_raw`](Self::into_raw) refuses to
    /// be: the caller receives the address and the generation-checked ownership
    /// in the same step, so the release obligation travels with the pointer
    /// instead of being dropped on the floor. `None` means the buffer was raw
    /// owning or borrowed and the historical raw contract applies.
    pub fn into_raw_with_owner(self) -> (*mut c_void, Option<BoundBufferOwnership>) {
        let ptr = self.ptr.as_ptr();
        match self.owner {
            BufferOwner::Bound(owner) => (ptr, Some(BoundBufferOwnership::Binding(*owner))),
            BufferOwner::Managed(owner) => (ptr, Some(BoundBufferOwnership::Managed(*owner))),
            _ => (ptr, None),
        }
    }

    /// Consume the buffer into its raw address and complete binding ownership.
    pub fn into_raw_with_bound_ownership(self) -> (*mut c_void, Option<BoundBufferOwnership>) {
        let ptr = self.ptr.as_ptr();
        match self.owner {
            BufferOwner::Bound(owner) => (ptr, Some(BoundBufferOwnership::Binding(*owner))),
            BufferOwner::Managed(owner) => (ptr, Some(BoundBufferOwnership::Managed(*owner))),
            _ => (ptr, None),
        }
    }
}

/// Complete binding-issued ownership carried by a [`DeviceBuffer`].
#[derive(Debug)]
pub enum BoundBufferOwnership {
    Binding(OwningAllocation),
    Managed(ManagedAllocation),
}

impl BoundBufferOwnership {
    pub fn owner(&self) -> &OwningAllocation {
        match self {
            Self::Binding(owner) => owner,
            Self::Managed(owner) => owner.owner_ref(),
        }
    }
}

/// Executor workspace allocation with any compatibility lease it still needs.
///
/// Manager-aware providers retain charges inside the buffer's managed owner and
/// leave `lease` empty. The default adapter preserves the older
/// reserve-then-allocate contract for providers not yet migrated.
#[derive(Debug)]
pub struct WorkspaceAllocation {
    buffer: DeviceBuffer,
    lease: Option<MemoryLease>,
}

static QUARANTINED_WORKSPACE_LEASES: std::sync::OnceLock<std::sync::Mutex<Vec<MemoryLease>>> =
    std::sync::OnceLock::new();

fn quarantine_failed_workspace_lease(lease: MemoryLease) {
    eprintln!(
        "execution provider workspace deallocation failed before physical release was proven; \
         retaining its {} byte {:?} lease in compatibility quarantine",
        lease.bytes(),
        lease.tier()
    );
    QUARANTINED_WORKSPACE_LEASES
        .get_or_init(|| std::sync::Mutex::new(Vec::new()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .push(lease);
}

#[cfg(test)]
fn quarantined_workspace_lease_count() -> usize {
    QUARANTINED_WORKSPACE_LEASES
        .get_or_init(|| std::sync::Mutex::new(Vec::new()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .len()
}

impl WorkspaceAllocation {
    pub fn new(buffer: DeviceBuffer, lease: Option<MemoryLease>) -> Self {
        Self { buffer, lease }
    }

    pub fn buffer(&self) -> &DeviceBuffer {
        &self.buffer
    }

    pub fn buffer_mut(&mut self) -> &mut DeviceBuffer {
        &mut self.buffer
    }

    pub fn into_parts(self) -> (DeviceBuffer, Option<MemoryLease>) {
        (self.buffer, self.lease)
    }
}

impl std::ops::Deref for WorkspaceAllocation {
    type Target = DeviceBuffer;

    fn deref(&self) -> &Self::Target {
        &self.buffer
    }
}

impl std::ops::DerefMut for WorkspaceAllocation {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.buffer
    }
}

// SAFETY: `DeviceBuffer` is an owning *handle* — it stores only a base address
// plus metadata and exposes no safe way to read or write the pointed-to memory
// (all access goes through `as_ptr`/`as_mut_ptr`, which are safe to *call* but
// `unsafe` to *use*). Moving the handle to another thread transfers ownership of
// the address; this is sound for every allocator we target — host `malloc`,
// CUDA device pointers, and MLX unified memory are all address-portable and not
// thread-affine at the pointer level. Any data race on the *contents* is
// prevented one layer up by `&`/`&mut` aliasing on `TensorView`/`TensorMut` and
// by the scheduler, not by this type. If a future EP wires a genuinely
// thread-affine allocator, it must wrap the handle in a non-`Send` owner rather
// than weaken this invariant (plan §4.4 flags this for a dedicated review when
// ep-cpu lands real memory).
unsafe impl Send for DeviceBuffer {}
// SAFETY: `&DeviceBuffer` grants no interior mutability — it can only produce a
// `*const` via `as_ptr` (a plain address copy) and read `Copy` metadata, so
// concurrent shared reads of the handle are race-free. Writing requires
// `as_mut_ptr`, which needs `&mut self`; obtaining a writable pointer therefore
// cannot happen through a shared reference in safe code. As with `Send`,
// mutating the underlying memory is gated behind `unsafe` pointer use whose
// synchronization is the caller's responsibility.
unsafe impl Sync for DeviceBuffer {}

/// A synchronization fence returned by async operations.
///
/// The `id` is an opaque, EP-private handle to a completion event recorded on a
/// transfer stream by [`ExecutionProvider::copy_async`]. Await it by passing the
/// fence back to [`ExecutionProvider::wait_fence`], which makes the EP's compute
/// stream wait on the recorded event so a later kernel never reads bytes the
/// asynchronous copy is still transferring.
///
/// The id `0` is reserved for an **already-signalled** fence: a fully
/// synchronous copy (e.g. the CPU EP, or a zero-byte transfer) needs no wait, so
/// [`Fence::default`] / [`Fence::signalled`] returns id `0` and
/// [`ExecutionProvider::wait_fence`] treats it as a no-op.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Fence {
    pub id: u64,
}

impl Fence {
    /// A fence that is already complete; awaiting it is a no-op.
    pub fn signalled() -> Self {
        Self { id: 0 }
    }

    /// Wrap an EP-private completion-event handle.
    pub fn new(id: u64) -> Self {
        Self { id }
    }

    /// Whether this fence is already complete (needs no wait).
    pub fn is_signalled(&self) -> bool {
        self.id == 0
    }
}

/// Resolved-shape facts needed by an EP's structural capture-region policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CaptureRegionShapeStatus {
    /// Every present node input has a concrete shape before capture.
    pub inputs_resolved: bool,
    /// Every node output has a concrete shape before capture.
    pub outputs_resolved: bool,
}

/// Structural reason an EP excludes a node from a device-graph capture region.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StructuralCaptureDecline {
    /// Host-driven control-flow or sequence semantics.
    HostControlFlowOrSequence,
    /// A data-dependent output shape was unresolved before capture.
    UnresolvedOutputShape,
    /// A data-dependent input shape was unresolved before capture.
    UnresolvedInputShape,
}

impl StructuralCaptureDecline {
    /// Stable diagnostic text matching the executor's original capture audit.
    pub const fn reason(self) -> &'static str {
        match self {
            Self::HostControlFlowOrSequence => {
                "control-flow and sequence nodes are not device-graph capturable"
            }
            Self::UnresolvedOutputShape => {
                "data-dependent output shape was unresolved before capture"
            }
            Self::UnresolvedInputShape => {
                "data-dependent input shape was unresolved before capture"
            }
        }
    }
}

/// Uploads host bytes into a raw device address for a device EP.
///
/// This is the narrow capability the plugin's fused-subgraph executor needs to
/// stage a host-resident boundary input into device memory when ORT runs an
/// interspersed CPU→device partition and never inserts the host→device copy
/// itself (issue #982). It is deliberately smaller than the full
/// [`ExecutionProvider`] surface — a device address and a length — so it can be
/// captured once at compile time and stored on the executor without holding an
/// EP reference (which would change EP teardown semantics).
///
/// Implementations must perform a **synchronous** upload: on return the bytes
/// are resident at `dst`, so the caller may launch a kernel that reads them.
pub trait HostToDeviceCopier: Send + Sync {
    /// Copy `src` host bytes into device destination `dst`.
    ///
    /// # Safety
    ///
    /// `dst` must point to a live device allocation, on this copier's device,
    /// of at least `src.len()` bytes.
    unsafe fn copy_host_to_device(&self, src: &[u8], dst: *mut c_void) -> Result<()>;
}

/// Source-attributed device allocations made outside an EP's allocator seam.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RawDeviceAllocationSiteStats {
    pub file: &'static str,
    pub line: u32,
    pub requests: u64,
    pub requested_bytes: u64,
    pub driver_allocations: u64,
    pub driver_bytes: u64,
    pub pool_hits: u64,
    pub pool_hit_bytes: u64,
}

/// Immutable, provider-owned device allocation prepared from graph-constant
/// bytes before a kernel's first launch.
///
/// The allocation exposes only a read-only pointer plus the identities a
/// kernel needs to reject cross-provider/runtime substitution. It cannot be
/// converted back into a mutable [`DeviceBuffer`].
pub trait SealedDeviceAllocation: Send + Sync {
    fn ptr(&self) -> crate::DevicePtr;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn device(&self) -> DeviceId;
    fn provider_context(&self) -> ProviderContextIdentity;
    fn allocation_identity(&self) -> AllocationIdentity;
    fn runtime_identity(&self) -> usize;
}

/// The core EP interface. Every backend crate implements this (§4.1).
pub trait ExecutionProvider: Send + Sync {
    /// EP identifier (snake_case, e.g. `"cpu_ep"`, `"cuda_ep"`).
    fn name(&self) -> &str;

    fn device_type(&self) -> DeviceType;
    fn device_id(&self) -> DeviceId;

    /// PCI vendor id of this EP's device memory (0 = generic/host). Used by the
    /// plugin executor to reconstruct the device `OrtMemoryInfo` ORT registered
    /// the device allocator against, as a fallback for staging host-resident
    /// boundary inputs when no device-resident `OrtValue` is otherwise visible
    /// (issue #982). Host EPs keep the default.
    fn memory_vendor_id(&self) -> u32 {
        0
    }

    /// A synchronous host→device uploader, or `None` for host EPs.
    ///
    /// Device EPs return a small [`HostToDeviceCopier`] the plugin's fused
    /// executor captures at compile time and uses to stage host-resident
    /// boundary inputs into device scratch before launching a device kernel
    /// (issue #982). Returning `None` (the default) opts an EP out of staging
    /// entirely: its inputs are used verbatim, exactly as before.
    fn host_to_device_copier(&self) -> Option<std::sync::Arc<dyn HostToDeviceCopier>> {
        None
    }

    /// Optional executor-to-EP capabilities. Stock EPs advertise none and
    /// continue receiving resident [`crate::TensorView`] inputs.
    fn capabilities(&self) -> ExecutionProviderCapabilities {
        ExecutionProviderCapabilities::stock()
    }

    /// Identity of the concrete runtime/context used by kernels from this EP.
    ///
    /// Device providers that support sealed constants override this. `None`
    /// keeps stock providers out of the sealed-admission contract.
    fn runtime_identity(&self) -> Option<usize> {
        None
    }

    /// Identity of the provider memory context that owns sealed constants.
    fn provider_context_identity(&self) -> Option<ProviderContextIdentity> {
        None
    }

    /// Whether this provider replaces one graph-constant input with immutable
    /// provider-owned storage during kernel preparation.
    ///
    /// The session may omit the ordinary resident initializer buffer only when
    /// every consumer slot returns `true`; dispatch then requires the prepared
    /// kernel to supply an exact [`Kernel::constant_input_override`] before the
    /// input can reach execution. Stock providers retain the resident path.
    fn prepares_immutable_constant(&self, node: &Node, input_idx: usize) -> bool {
        let _ = (node, input_idx);
        false
    }

    /// Validate-before-upload sink used by kernels with immutable graph-weight
    /// contracts. The default fails closed; a provider must explicitly support
    /// generation-bound sealed allocations.
    fn upload_sealed_constant(
        &self,
        bytes: &[u8],
        alignment: usize,
    ) -> Result<Arc<dyn SealedDeviceAllocation>> {
        let _ = (bytes, alignment);
        Err(EpError::KernelFailed(format!(
            "{} does not support sealed constant admission",
            self.name()
        )))
    }

    /// Page a lazy weight into device memory for live dispatch (WEIGHT_OFFLOAD
    /// Phase 3b). Returns a [`crate::PagedWeight`] whose device pointer the
    /// executor substitutes into the weight's input view; the binding must be
    /// held for the kernel's lifetime so the residency is not reclaimed early.
    ///
    /// `key` is a stable per-weight identity (the executor passes the
    /// initializer's value id) an EP may use to cache/evict residency across
    /// decode steps. The default returns `None`: stock EPs never receive lazy
    /// handles and the executor falls back to the host-materialization route.
    fn page_lazy_weight(
        &self,
        key: u64,
        weight: &crate::LazyWeight,
        source: &dyn crate::MmapRegionSource,
    ) -> Result<Option<crate::PagedWeight>> {
        let _ = (key, weight, source);
        Ok(None)
    }

    /// Prove routed-bank residency for a QMoE-family dispatch and mint a
    /// guard the executor keeps alive for the kernel's lifetime, exactly like
    /// [`Self::page_lazy_weight`]'s `PagedWeight`.
    ///
    /// `requirement` names what the caller (today, always
    /// [`crate::RoutedResidencyRequirement::FusedRoutingUnknown`] — no QMoE or
    /// BlockQuantizedMoE kernel in this codebase surfaces routed expert ids to
    /// the host before or during dispatch) can prove before launch; `catalog`
    /// is the same per-boundary [`onnx_runtime_loader::WeightRegionCatalog`]
    /// `page_lazy_weight` callers already have from `expert_region_candidates`.
    /// The default returns `None`: stock EPs (and the CUDA EP when offload is
    /// disabled) never mint a guard and the executor does not gate resize on
    /// one, matching every other lazy-weight default in this trait.
    fn acquire_routed_residency(
        &self,
        key: u64,
        requirement: crate::RoutedResidencyRequirement,
        catalog: &onnx_runtime_loader::WeightRegionCatalog,
    ) -> Result<Option<Box<dyn crate::RoutedResidencyGuardHandle>>> {
        let _ = (key, requirement, catalog);
        Ok(None)
    }

    /// Best-effort lookahead page-in for a lazy weight the executor knows will be
    /// needed by a later node. Returns `true` only when a transfer was actually
    /// enqueued, so callers can distinguish a real prefetch from a no-op or
    /// eviction-neutrality guard decline. The default is a no-op so providers
    /// that do not own a residency cache do not need to participate.
    fn prefetch_lazy_weight(
        &self,
        key: u64,
        weight: &crate::LazyWeight,
        source: &dyn crate::MmapRegionSource,
    ) -> Result<bool> {
        let _ = (key, weight, source);
        Ok(false)
    }

    /// Initialize device resources / load libraries.
    fn initialize(&mut self, config: &EpConfig) -> Result<()>;
    /// Release device resources.
    fn shutdown(&mut self) -> Result<()>;

    /// Whether this EP can run `op` at the model's effective `opset` with the
    /// given input shapes, dtypes, and layouts, and at what cost.
    ///
    /// Every [`KernelMatch::Unsupported`] result must carry an actionable reason:
    /// state what the EP accepts and, where possible, how to fix the model or
    /// registration rather than returning a bare decline.
    fn supports_op(
        &self,
        op: &Node,
        opset: u64,
        shapes: &[Shape],
        input_dtypes: &[DataType],
        layouts: &[TensorLayout],
    ) -> KernelMatch;

    /// Query one node through an immutable structural graph lens.
    ///
    /// This compatibility adapter allocates metadata arrays before calling
    /// [`Self::supports_op`]. EPs can override it with native indexed metadata
    /// traversal to make capability discovery allocation-free.
    fn supports_node(&self, view: &GraphView<'_>, node: NodeIndex, opset: u64) -> KernelMatch {
        let inputs = view.node_inputs(node);
        let shapes = inputs
            .iter()
            .map(|input| {
                input
                    .map(|value| view.value(value).shape.clone())
                    .unwrap_or_default()
            })
            .collect::<Vec<_>>();
        let input_dtypes = inputs
            .iter()
            .map(|input| {
                input
                    .map(|value| view.value(value).dtype)
                    .unwrap_or(DataType::Undefined)
            })
            .collect::<Vec<_>>();
        let layouts = inputs
            .iter()
            .map(|input| {
                input
                    .map(|value| view.value(value).layout.clone())
                    .unwrap_or_else(TensorLayout::contiguous)
            })
            .collect::<Vec<_>>();
        self.supports_op(view.node(node), opset, &shapes, &input_dtypes, &layouts)
    }

    /// Get or create a kernel for `op` specialized to concrete `shapes`.
    ///
    /// `opset` is the effective operator-set version for `op`'s domain in the
    /// owning graph. EPs use it to select opset-specialized kernels (e.g. the
    /// opset-13 per-axis vs. the legacy opset-<13 2D-coercion `Softmax`).
    fn get_kernel(&self, op: &Node, shapes: &[Vec<usize>], opset: u64) -> Result<Box<dyn Kernel>>;

    /// Executor-scoped kernel creation under a resolved artifact configuration.
    ///
    /// The default preserves providers that own no executor-scoped artifacts.
    /// Providers whose factories publish producer handles override this method
    /// so compilation is attributed to the owning executor/provider generation
    /// rather than to a graph-local node id shared by sibling sessions. Such
    /// providers must reject foreign or mismatched configuration tokens before
    /// publishing producer state.
    fn get_kernel_for_executor(
        &self,
        config: ExecutorArtifactConfig,
        op: &Node,
        shapes: &[Vec<usize>],
        opset: u64,
    ) -> Result<Box<dyn Kernel>> {
        let _ = config;
        self.get_kernel(op, shapes, opset)
    }

    /// Classify whether `op` may be compiled without a session-issued executor
    /// generation.
    ///
    /// The default keeps generic providers and kernels unchanged. Providers
    /// that publish executor-owned artifacts must return
    /// [`ExecutorKernelScope::Required`] for exactly those factories.
    fn executor_kernel_scope(&self, _op: &Node) -> ExecutorKernelScope {
        ExecutorKernelScope::Unscoped
    }

    /// Apply EP-owned structural policy to one prospective capture-region node.
    ///
    /// The executor supplies only graph structure and resolved-shape presence.
    /// Kernel warmth and the selected compiled kernel's capture support remain
    /// executor-owned mechanism and are checked only after this hook admits the
    /// node. Implementations must decline when either shape-status field is
    /// false; admitting an unresolved shape violates the executor contract. The
    /// default preserves the original predicate precedence exactly.
    fn plan_capture_region(
        &self,
        node: &Node,
        shape_status: CaptureRegionShapeStatus,
    ) -> Option<StructuralCaptureDecline> {
        if is_control_flow_or_sequence(node) {
            return Some(StructuralCaptureDecline::HostControlFlowOrSequence);
        }
        if !shape_status.outputs_resolved {
            return Some(StructuralCaptureDecline::UnresolvedOutputShape);
        }
        if !shape_status.inputs_resolved {
            return Some(StructuralCaptureDecline::UnresolvedInputShape);
        }
        None
    }

    /// Allocate device memory.
    fn allocate(&self, size: usize, alignment: usize) -> Result<DeviceBuffer>;

    fn allocate_with_mapped_growth(
        &self,
        size: usize,
        alignment: usize,
        grant: onnx_runtime_memory_governor::MappedGrowthGrant,
    ) -> Result<DeviceBuffer> {
        let newly_mapped_bytes = self.mapped_bytes_for_allocation(size, alignment)?;
        let allocation = self.allocate(size, alignment)?;
        if let Err(error) = grant.commit_bytes(newly_mapped_bytes) {
            let _ = self.deallocate(allocation);
            return Err(EpError::Memory(error));
        }
        Ok(allocation)
    }

    /// Allocate executor workspace as one reserve/allocate/commit transaction.
    ///
    /// Providers with a process manager override this so the charge travels with
    /// physical ownership through deferred release. The default is the existing
    /// compatibility sequence for synchronous providers.
    fn allocate_workspace(
        &self,
        size: usize,
        alignment: usize,
        role: MemoryRole,
    ) -> Result<WorkspaceAllocation> {
        let target_mapped = self.mapped_bytes_for_allocation(size, alignment)?;
        let mut grant = self.prepare_mapped_growth(target_mapped, role)?;
        let lease = match self.reserve_workspace(size as u64, role) {
            Ok(lease) => lease,
            Err(error) => {
                drop(grant);
                return Err(error);
            }
        };
        let buffer = match grant.take() {
            Some(grant) => self.allocate_with_mapped_growth(size, alignment, grant)?,
            None => self.allocate(size, alignment)?,
        };
        Ok(WorkspaceAllocation::new(buffer, lease))
    }

    /// Replace executor workspace without granting the replacement until the
    /// old allocation's provider-specific release boundary is satisfied.
    ///
    /// Synchronous providers use this default. Deferred providers override it
    /// to await a structured release outcome with a finite deadline.
    fn replace_workspace(
        &self,
        old: Option<WorkspaceAllocation>,
        size: usize,
        alignment: usize,
        role: MemoryRole,
    ) -> Result<WorkspaceAllocation> {
        if let Some(old) = old {
            self.deallocate_workspace(old)?;
        }
        self.allocate_workspace(size, alignment, role)
    }

    /// Allocate device address space while committing only selected byte ranges.
    ///
    /// Providers whose allocator cannot reserve without committing should use
    /// the default, preserving eager allocation. CUDA VMM overrides this so
    /// shape-stable buffers such as KV can keep one virtual address while
    /// mapping physical granules only where the live sequence reaches.
    fn allocate_committed(
        &self,
        size: usize,
        alignment: usize,
        committed_ranges: &[std::ops::Range<usize>],
    ) -> Result<DeviceBuffer> {
        let _ = committed_ranges;
        self.allocate(size, alignment)
    }

    /// Ensure a byte range in an existing allocation is backed by physical
    /// memory. Eager providers committed everything at allocation time, so their
    /// default is a no-op.
    fn commit_allocation_range(
        &self,
        buffer: &DeviceBuffer,
        offset: usize,
        bytes: usize,
    ) -> Result<()> {
        let _ = (buffer, offset, bytes);
        Ok(())
    }

    /// Commit all listed ranges as one allocator transaction.
    fn commit_allocation_ranges(&self, ranges: &[(&DeviceBuffer, usize, usize)]) -> Result<()> {
        for &(buffer, offset, bytes) in ranges {
            self.commit_allocation_range(buffer, offset, bytes)?;
        }
        Ok(())
    }

    fn commit_allocation_ranges_with_mapped_growth(
        &self,
        ranges: &[(&DeviceBuffer, usize, usize)],
        grant: &mut onnx_runtime_memory_governor::MappedGrowthGrant,
    ) -> Result<u64> {
        let _ = grant;
        self.commit_allocation_ranges(ranges)?;
        self.mapped_bytes_for_allocation_ranges(ranges)
    }

    fn mapped_bytes_for_allocation_ranges(
        &self,
        ranges: &[(&DeviceBuffer, usize, usize)],
    ) -> Result<u64> {
        Ok(ranges.iter().fold(0_u64, |total, (_, _, bytes)| {
            total.saturating_add(*bytes as u64)
        }))
    }

    /// Release physical backing from a byte range in an existing allocation
    /// while preserving its virtual address. Lazy providers use this for
    /// transactional growth rollback. Eager providers return an actionable
    /// unsupported error: unlike commit, decommit has no eager equivalent.
    /// Returns the bytes actually unmapped after shared references are applied.
    fn decommit_allocation_range(
        &self,
        buffer: &DeviceBuffer,
        offset: usize,
        bytes: usize,
    ) -> Result<u64> {
        let _ = (buffer, offset, bytes);
        Err(EpError::KernelFailed(format!(
            "{}: partial decommit requires a VirtualBacking capability",
            self.name()
        )))
    }

    /// Physical bytes currently claimed by `buffer`. Eager providers return
    /// `buffer.len()`; lazy providers may report the committed subset.
    fn allocation_committed_bytes(&self, buffer: &DeviceBuffer) -> usize {
        buffer.len()
    }

    /// Free device memory.
    fn deallocate(&self, buffer: DeviceBuffer) -> Result<()>;

    /// Wait until releases previously accepted by [`Self::deallocate`] have
    /// reached a terminal state.
    ///
    /// Synchronous providers need no work here. Providers whose `deallocate`
    /// queues ownership behind device fences override this method and wait on
    /// their structured release queue. This is a lifecycle boundary for owners
    /// such as an ORT allocator; it is not a per-operation synchronization
    /// primitive.
    fn wait_for_deferred_releases(&self) -> Result<()> {
        Ok(())
    }

    /// Release an executor workspace and then its compatibility lease.
    ///
    /// The default is only for providers whose `deallocate` settles synchronously.
    /// An asynchronous provider must override this method and keep accounting
    /// attached to its own structured settlement.
    fn deallocate_workspace(&self, workspace: WorkspaceAllocation) -> Result<()> {
        let (buffer, lease) = workspace.into_parts();
        match self.deallocate(buffer) {
            Ok(()) => {
                drop(lease);
                Ok(())
            }
            Err(error) => {
                if let Some(lease) = lease {
                    quarantine_failed_workspace_lease(lease);
                }
                Err(error)
            }
        }
    }

    /// Free device memory and report mapped-zone bytes actually unmapped.
    ///
    /// The report is based on global mapping references, not which allocation
    /// originally caused the mapping.
    fn deallocate_with_unmapped(&self, buffer: DeviceBuffer) -> Result<u64> {
        self.deallocate(buffer)?;
        Ok(0)
    }

    /// Synchronous copy (host↔device or device↔device).
    fn copy(&self, src: &DeviceBuffer, dst: &mut DeviceBuffer, size: usize) -> Result<()>;
    /// Asynchronous copy; returns a [`Fence`] to await.
    ///
    /// The copy is enqueued on a dedicated transfer stream (not the compute
    /// stream) so it can overlap compute already queued on the compute stream —
    /// this is the mechanism half of Phase-4 compute/transfer overlap for weight
    /// paging. The returned [`Fence`] names a completion event on that transfer
    /// stream; the caller must order any consumer of `dst` after the transfer by
    /// passing the fence to [`ExecutionProvider::wait_fence`] before launching a
    /// kernel that reads `dst`. A synchronous EP may perform the copy inline and
    /// return an already-signalled [`Fence::signalled`].
    fn copy_async(&self, src: &DeviceBuffer, dst: &mut DeviceBuffer, size: usize) -> Result<Fence>;

    /// Order this EP's compute stream after the transfer named by `fence`.
    ///
    /// Makes the compute stream wait on the fence's completion event (a
    /// stream-ordered, non-host-blocking cross-stream wait) so a subsequently
    /// launched kernel observes the fully-transferred bytes produced by the
    /// matching [`ExecutionProvider::copy_async`]. Awaiting an already-signalled
    /// fence ([`Fence::is_signalled`]) is a no-op. The default implementation is
    /// a no-op, correct for synchronous EPs whose `copy_async` already completed
    /// the transfer before returning.
    fn wait_fence(&self, _fence: &Fence) -> Result<()> {
        Ok(())
    }

    /// Record a completion event for all compute enqueued on this EP's compute
    /// stream so far, returning a [`Fence`] that later transfers can wait on.
    ///
    /// This is the write-after-read (WAR) half of double-buffered prefetch: once
    /// a kernel that *reads* a staging buffer has been launched on the compute
    /// stream, record a fence over it and pass that fence to
    /// [`ExecutionProvider::copy_wait_fence`] before enqueueing the async copy
    /// that *overwrites* the same buffer, so the transfer stream never clobbers
    /// bytes a still-running consumer is reading. The default implementation
    /// returns an already-signalled [`Fence::signalled`] — correct for
    /// synchronous EPs whose compute completes inline, making the paired
    /// [`ExecutionProvider::copy_wait_fence`] a no-op.
    fn record_compute_fence(&self) -> Result<Fence> {
        Ok(Fence::signalled())
    }

    /// Order this EP's transfer stream after the compute named by `fence`.
    ///
    /// Makes the transfer (copy) stream wait on the fence's completion event (a
    /// stream-ordered, non-host-blocking cross-stream wait) so an async copy
    /// enqueued afterwards does not overwrite a buffer while the prior consumer
    /// recorded by [`ExecutionProvider::record_compute_fence`] is still reading
    /// it (WAR hazard on double-buffer reuse). Awaiting an already-signalled
    /// fence ([`Fence::is_signalled`]) is a no-op, as is the default
    /// implementation — correct for synchronous EPs.
    fn copy_wait_fence(&self, _fence: &Fence) -> Result<()> {
        Ok(())
    }

    /// Whether this EP can select the first maximum f32 element on-device and
    /// return the token id together with its capture-error status.
    fn device_argmax_supported(&self) -> bool {
        false
    }

    /// Launch an allocation-free device argmax over `batch` sequences of
    /// `elements` contiguous `dtype` values (Float32 or Float16) each, laid out
    /// as a `[batch, elements]` row-major block. `result` receives, per sequence
    /// `s`, two native-endian u32 values at word offset `2*s`: the token id, then
    /// the latching device capture-error bitmask. At `batch == 1` this is the
    /// previous single-sequence contract byte-for-byte.
    ///
    /// `tie_break` selects which token id wins when several logits share the
    /// maximum value; see [`ArgmaxTieBreak`]. [`ArgmaxTieBreak::LowestIndex`] is
    /// the base-decode / ORT byte-identity default.
    fn device_argmax(
        &self,
        _logits: &DeviceBuffer,
        _elements: usize,
        _batch: usize,
        _dtype: DataType,
        _result: &mut DeviceBuffer,
        _tie_break: ArgmaxTieBreak,
    ) -> Result<()> {
        Err(EpError::KernelFailed(format!(
            "{}: device argmax is not supported",
            self.name()
        )))
    }

    /// Fold the just-selected greedy token (from a prior [`device_argmax`],
    /// `result[0]`) into the persistent decode bindings device-to-device, for
    /// the native CUDA device-token-loop: write the token as an `i64` into
    /// `input_ids`, write `next_position` into `position_ids`, set the mask `1`
    /// at `next_position` (guarded by `mask_len`), append the token to
    /// `scratch[step]`, and OR the shared capture-error word (`result[1]`) into
    /// `scratch[capacity]`. No host sync — the caller drains `scratch` once per
    /// chain. EPs without device kernels reject the request.
    ///
    /// [`device_argmax`]: ExecutionProvider::device_argmax
    #[allow(clippy::too_many_arguments)]
    fn device_token_writer(
        &self,
        _result: &DeviceBuffer,
        _input_ids: &DeviceBuffer,
        _position_ids: &DeviceBuffer,
        _attention_mask: &DeviceBuffer,
        _scratch: &DeviceBuffer,
        _capacity: usize,
        _next_position: i64,
        _mask_len: usize,
        _write_position: bool,
        _step: u32,
    ) -> Result<()> {
        Err(EpError::KernelFailed(format!(
            "{}: device token writer is not supported",
            self.name()
        )))
    }

    /// Begin recording the supplied, already-compiled kernel sequence into a
    /// device graph. EPs without graph support reject the request.
    fn begin_device_graph_capture(&self, _kernels: &[&dyn Kernel]) -> Result<()> {
        Err(EpError::KernelFailed(format!(
            "{}: device graph capture is not supported",
            self.name()
        )))
    }

    /// End device-graph capture and install the resulting executable.
    fn end_device_graph_capture(&self) -> Result<()> {
        Err(EpError::KernelFailed(format!(
            "{}: device graph capture is not supported",
            self.name()
        )))
    }

    /// Abort an in-progress device-graph capture, returning the stream and
    /// lifecycle to a clean idle state so a subsequent [`reset_device_graph`]
    /// succeeds. Called on the error path of segmented capture when a node
    /// fails mid-record: the capture must always be ended before reset, so the
    /// stream is not left wedged in capture mode. EPs without device graphs have
    /// nothing to abort.
    ///
    /// [`reset_device_graph`]: ExecutionProvider::reset_device_graph
    fn abort_device_graph_capture(&self) -> Result<()> {
        Ok(())
    }

    /// Replay the installed device graph.
    ///
    /// When the EP holds multiple captured **segments** (segmented capture), this
    /// replays every installed segment in capture order. For the single-graph
    /// fast path (one whole-subgraph capture) that is exactly the one graph.
    fn replay_device_graph(&self) -> Result<()> {
        Err(EpError::KernelFailed(format!(
            "{}: device graph replay is not supported",
            self.name()
        )))
    }

    /// Replay one captured **segment** by its zero-based capture-order index.
    ///
    /// Segmented capture claims a whole subgraph even when only parts are
    /// device-graph capturable: the executor captures each maximal capturable
    /// run as its own segment and, at replay time, launches the segment graphs
    /// in order while running the non-capturable seam nodes eagerly in between.
    /// EPs without segmented graph support reject the request.
    fn replay_device_graph_segment(&self, _index: usize) -> Result<()> {
        Err(EpError::KernelFailed(format!(
            "{}: segmented device graph replay is not supported",
            self.name()
        )))
    }

    /// Destroy any installed device graph before its referenced buffers move or
    /// are released.
    fn reset_device_graph(&self) -> Result<bool> {
        Ok(false)
    }

    /// Slot-parameterized [`begin_device_graph_capture`]. The default routes the
    /// [`DeviceGraphSlot::Primary`] slot to the single-slot method and rejects
    /// any other slot, so EPs that own only one captured graph are unchanged.
    /// Multi-slot EPs (the CUDA EP) override this to record into the named slot.
    ///
    /// [`begin_device_graph_capture`]: ExecutionProvider::begin_device_graph_capture
    fn begin_device_graph_capture_in(
        &self,
        slot: DeviceGraphSlot,
        kernels: &[&dyn Kernel],
    ) -> Result<()> {
        match slot {
            DeviceGraphSlot::Primary => self.begin_device_graph_capture(kernels),
            other => Err(unsupported_graph_slot(self.name(), other)),
        }
    }

    /// Slot-parameterized [`end_device_graph_capture`].
    ///
    /// [`end_device_graph_capture`]: ExecutionProvider::end_device_graph_capture
    fn end_device_graph_capture_in(&self, slot: DeviceGraphSlot) -> Result<()> {
        match slot {
            DeviceGraphSlot::Primary => self.end_device_graph_capture(),
            other => Err(unsupported_graph_slot(self.name(), other)),
        }
    }

    /// Slot-parameterized [`abort_device_graph_capture`].
    ///
    /// [`abort_device_graph_capture`]: ExecutionProvider::abort_device_graph_capture
    fn abort_device_graph_capture_in(&self, slot: DeviceGraphSlot) -> Result<()> {
        match slot {
            DeviceGraphSlot::Primary => self.abort_device_graph_capture(),
            other => Err(unsupported_graph_slot(self.name(), other)),
        }
    }

    /// Slot-parameterized [`replay_device_graph`].
    ///
    /// [`replay_device_graph`]: ExecutionProvider::replay_device_graph
    fn replay_device_graph_in(&self, slot: DeviceGraphSlot) -> Result<()> {
        match slot {
            DeviceGraphSlot::Primary => self.replay_device_graph(),
            other => Err(unsupported_graph_slot(self.name(), other)),
        }
    }

    /// Slot-parameterized [`replay_device_graph_segment`].
    ///
    /// [`replay_device_graph_segment`]: ExecutionProvider::replay_device_graph_segment
    fn replay_device_graph_segment_in(&self, slot: DeviceGraphSlot, index: usize) -> Result<()> {
        match slot {
            DeviceGraphSlot::Primary => self.replay_device_graph_segment(index),
            other => Err(unsupported_graph_slot(self.name(), other)),
        }
    }

    /// Slot-parameterized [`reset_device_graph`]. A multi-slot EP resets only the
    /// named slot, leaving the other slot's installed graph intact.
    ///
    /// [`reset_device_graph`]: ExecutionProvider::reset_device_graph
    fn reset_device_graph_in(&self, slot: DeviceGraphSlot) -> Result<bool> {
        match slot {
            DeviceGraphSlot::Primary => self.reset_device_graph(),
            // No graph is ever installed in a non-Primary slot on a single-slot
            // EP, so there is nothing to reset (mirrors `reset_device_graph`'s
            // "no graph" return rather than erroring, so unconditional
            // per-slot reset sweeps are safe on every EP).
            DeviceGraphSlot::Verify => Ok(false),
        }
    }

    /// Whether the named slot currently holds a replayable installed graph
    /// executable.
    ///
    /// The executor uses this as a pre-replay liveness check: an installed graph
    /// can be reset out-of-band (e.g. a kernel-variant eviction retires kernels
    /// baked into a captured graph and resets its slot) while the executor's
    /// host-side capture signature/schedule stays live. Replaying an emptied slot
    /// would hard-error; querying this first lets the executor detect the
    /// desync and re-warm/re-capture gracefully instead.
    ///
    /// The default reports `true` ("assume present, replay as usual") so EPs that
    /// never lose an installed graph out-of-band keep their existing behavior;
    /// only EPs whose slots can be emptied out-of-band (the CUDA EP) override this
    /// with the real per-slot check.
    fn has_device_graph_in(&self, slot: DeviceGraphSlot) -> Result<bool> {
        let _ = slot;
        Ok(true)
    }

    /// Begin capture in an executor-owned namespace and return the exact
    /// installation token. `continuation` is supplied for later segments of the
    /// same capture and must identify the already-installed generation.
    fn begin_owned_device_graph_capture(
        &self,
        owner: DeviceGraphOwner,
        slot: DeviceGraphSlot,
        continuation: Option<DeviceGraphToken>,
        kernels: &[&dyn Kernel],
    ) -> Result<DeviceGraphToken> {
        self.begin_device_graph_capture_in(slot, kernels)?;
        Ok(continuation.unwrap_or_else(|| DeviceGraphToken::new(owner, slot, 1)))
    }

    /// End the active capture identified by `token`.
    fn end_owned_device_graph_capture(&self, token: DeviceGraphToken) -> Result<()> {
        self.end_device_graph_capture_in(token.slot())
    }

    /// Abort the active capture identified by `token`.
    fn abort_owned_device_graph_capture(&self, token: DeviceGraphToken) -> Result<()> {
        self.abort_device_graph_capture_in(token.slot())
    }

    /// Replay the exact installed graph generation identified by `token`.
    fn replay_owned_device_graph(&self, token: DeviceGraphToken) -> Result<()> {
        self.replay_device_graph_in(token.slot())
    }

    /// Replay one segment of the exact installed generation.
    fn replay_owned_device_graph_segment(
        &self,
        token: DeviceGraphToken,
        index: usize,
    ) -> Result<()> {
        self.replay_device_graph_segment_in(token.slot(), index)
    }

    /// Reset only the exact installed generation identified by `token`.
    fn reset_owned_device_graph(&self, token: DeviceGraphToken) -> Result<bool> {
        self.reset_device_graph_in(token.slot())
    }

    /// Retire empty graph-lifecycle slots for an executor owner at final drop.
    ///
    /// Ordinary reset deliberately retains the lifecycle so a repeated capture
    /// cannot reuse an earlier installation generation.
    fn retire_owned_device_graphs(&self, _owner: DeviceGraphOwner) -> Result<()> {
        Ok(())
    }

    /// Whether the exact installed generation identified by `token` is live.
    fn has_owned_device_graph(&self, token: DeviceGraphToken) -> Result<bool> {
        self.has_device_graph_in(token.slot())
    }

    /// Register one executor or persistent binding during setup.
    ///
    /// Owners are registered once at executor/binding setup. Providers with
    /// deferred validation may reject this call while a previous generation is
    /// still executing. The returned token is the submitting executor's exact
    /// authority for this generation.
    fn register_device_validation_owner(&self) -> Result<DeviceValidationRegistration> {
        let owner = DeviceValidationOwner::new();
        Ok(DeviceValidationRegistration::new(owner, ()))
    }

    /// Retire one executor/binding validation owner at teardown.
    fn unregister_device_validation_owner(
        &self,
        _registration: &mut DeviceValidationRegistration,
    ) -> Result<()> {
        Ok(())
    }

    /// Begin one top-level device-validation generation for `registration`.
    fn begin_device_validation(
        &self,
        registration: &DeviceValidationRegistration,
    ) -> Result<DeviceValidationToken> {
        Ok(DeviceValidationToken::new(registration.owner(), 0))
    }

    /// Add one pre-registered output binding as an exact recipient of the
    /// active submission. The returned token is sticky in that binding's
    /// owner-scoped slot until the binding participates in a later submission
    /// or is unregistered.
    fn add_device_validation_recipient(
        &self,
        submission: DeviceValidationToken,
        recipient: &DeviceValidationRegistration,
    ) -> Result<DeviceValidationToken> {
        Ok(DeviceValidationToken::new(
            recipient.owner(),
            submission.generation(),
        ))
    }

    /// Seal recipient attachment and make the submission consumable.
    fn activate_device_validation(&self, _submission: DeviceValidationToken) -> Result<()> {
        Ok(())
    }

    /// Recover an executor submission while its stack is unwinding.
    fn abort_device_validation_submission(
        &self,
        _submission: DeviceValidationToken,
    ) -> Result<u32> {
        Ok(0)
    }

    /// Whether top-level execution defers validation until a host-visible read.
    fn defers_device_validation(&self) -> bool {
        false
    }

    /// Consume the exact top-level device-validation generation after a host
    /// synchronization boundary. Implementations reject foreign and stale
    /// tokens; concurrent exact consumers converge on the same sticky result.
    fn consume_device_validation_error(
        &self,
        _registration: &DeviceValidationRegistration,
        _token: DeviceValidationToken,
    ) -> Result<u32> {
        Ok(0)
    }

    /// Consume this executor's completed route-telemetry window.
    ///
    /// The session invokes this only after synchronizing device work and
    /// consuming the exact owner-scoped deferred-validation receipt. The
    /// executor identity is mandatory: providers must not consult unscoped or
    /// process-global producer state to satisfy this boundary. Non-participating
    /// providers have no route lifecycle and keep the no-op default.
    fn consume_route_residency_at_boundary_for_executor(
        &self,
        _executor: ExecutorInstanceId,
    ) -> Result<()> {
        Ok(())
    }

    /// Resolve the provider-owned half of an immutable executor configuration.
    ///
    /// The session calls this exactly once before compiling any kernel for the
    /// executor, binds the returned template to its own executor identity and
    /// fresh generation, then carries that capability through publication,
    /// finalization, and teardown. Providers cannot choose the executor owner.
    /// Providers that own no executor-scoped artifacts keep the disabled
    /// default.
    ///
    /// Child integrations based on the pre-scoped API (including #2163) must
    /// refresh by removing direct template binding/finalization and letting the
    /// session build own issuance. Provider code should only consume the
    /// borrowed proof delivered to [`Self::finalize_executor_artifacts`].
    fn resolve_executor_artifact_config(&self) -> Result<ExecutorArtifactConfigTemplate> {
        Ok(ExecutorArtifactConfigTemplate::resolved(
            ExecutorArtifactConfigAuthority::UNSCOPED,
            self.device_id(),
            ExecutorRouteResidencyConfig::Disabled,
        ))
    }

    /// Authoritative transition for "all provider artifacts required by this
    /// executor's resolved compilation are finalized."
    ///
    /// Static build and every newly compiled symbolic/dynamic specialization
    /// invoke this same idempotent path after kernel factories have published
    /// their producer handles and before any execution, capture, or replay.
    /// `readiness` advances at every executor kernel-cache miss, including
    /// binding preparation and runtime dispatch; the executor never calls a
    /// provider twice for the same pending/failed epoch. Structural declines
    /// may latch as complete; readiness-dependent absence returns a pending
    /// proof without poisoning a later epoch. An `Err` is also fail-closed and
    /// may be retried only after a later compilation epoch. The default
    /// completes without side effects.
    fn finalize_executor_artifacts(
        &self,
        proof: ExecutorArtifactFinalizationProof<'_>,
        _graph: &Graph,
    ) -> Result<ExecutorArtifactFinalization> {
        Ok(match proof {
            ExecutorArtifactFinalizationProof::Disabled(disabled) => disabled.complete(),
            ExecutorArtifactFinalizationProof::Enabled(enabled) => enabled.declined(),
        })
    }

    /// Drain exactly the artifacts owned by `executor`.
    ///
    /// The default is a no-op. Participating providers must make this
    /// idempotent and must not clear producer/boundary state owned by sibling
    /// executors sharing the same provider.
    fn drain_executor_artifacts(&self, _config: ExecutorArtifactConfig) {}

    /// Explicit device allocation/free counters, when the EP exposes them.
    fn device_allocation_counts(&self) -> Option<(u64, u64)> {
        None
    }

    /// Opt-in attribution for allocations made outside [`Self::allocate`].
    fn raw_device_allocation_site_stats(&self) -> Vec<RawDeviceAllocationSiteStats> {
        Vec::new()
    }

    /// Reserve governed bytes for executor-owned kernel workspace.
    ///
    /// Providers whose allocator already charges committed bytes may return
    /// `None`; providers backed by an eager allocator retain the returned lease
    /// alongside the allocation. The default preserves compatibility for
    /// providers without a device-memory governor.
    fn reserve_workspace(
        &self,
        _bytes: u64,
        _role: onnx_runtime_memory_governor::MemoryRole,
    ) -> Result<Option<onnx_runtime_memory_governor::MemoryLease>> {
        Ok(None)
    }

    fn prepare_mapped_growth(
        &self,
        bytes: u64,
        role: onnx_runtime_memory_governor::MemoryRole,
    ) -> Result<Option<onnx_runtime_memory_governor::MappedGrowthGrant>> {
        // `role` describes content/lifetime. Providers whose allocator
        // suballocates shared granules must canonicalize it to the arena's
        // physical mapped-attribution zone.
        let _ = (bytes, role);
        Ok(None)
    }

    fn mapped_bytes_for_allocation(&self, bytes: usize, alignment: usize) -> Result<u64> {
        let _ = alignment;
        Ok(bytes as u64)
    }

    fn release_mapped_growth(&self, bytes: u64, role: onnx_runtime_memory_governor::MemoryRole) {
        // This must use the same canonical physical zone as
        // `prepare_mapped_growth`; allocation lifetime is not map ownership.
        let _ = (bytes, role);
    }

    /// Place any long-lived device memory this provider holds under `governor`.
    ///
    /// Some providers keep a standing pool for as long as a model is loaded --
    /// the CUDA weight-residency cache is one. A pool that picks its own size is
    /// a second claim on memory the governor is already dividing up, and neither
    /// side can see the other: grant the KV pool most of a card, let a residency
    /// cache default to some fraction of it, and both are individually satisfied
    /// while the device is oversubscribed.
    ///
    /// This is the seam that ends that. It is on the provider contract rather
    /// than on one backend because it is not a CUDA question: any provider with
    /// a standing pool has it, and a third-party provider should be able to join
    /// the same accounting rather than run a ledger of its own.
    ///
    /// Returns the bytes now governed. The default is zero -- most providers
    /// hold no standing pool, and saying so is not a failure.
    ///
    /// # Errors
    ///
    /// If the tier cannot afford what the provider already holds. That is worth
    /// failing on: it says the model does not fit *before* the pool is used,
    /// rather than at an allocation somewhere unrelated later.
    ///
    /// Whether the memory this provider hands out commits physically as it is
    /// used rather than when it is requested.
    ///
    /// A forwarder, not a fact of its own: a provider should preserve the
    /// selected allocator's explicit [`DeviceAllocator::commits_on_demand`]
    /// signal. That signal requires both lazy physical mapping and governor
    /// charging; optional `VirtualBacking` capability presence alone is not
    /// enough. It is repeated here only because a caller holding a session
    /// reaches the allocator through the provider.
    ///
    /// `false` is the safe default -- a consumer that believes `true` will
    /// under-reserve.
    ///
    /// [`DeviceAllocator::commits_on_demand`]: onnx_runtime_memory_governor::DeviceAllocator::commits_on_demand
    fn commits_on_demand(&self) -> bool {
        false
    }

    /// Resize a provider-owned weight-residency budget before it joins a
    /// governor, returning the budget that will be adopted.
    ///
    /// `--vram-limit` is resolved after the model and backend are known, but a
    /// CUDA EP is constructed before the engine can size native KV. This hook
    /// lets load-time admission subtract the non-weight device claims first,
    /// preventing #712's "weights took the whole limit, KV failed later" path.
    fn set_weight_residency_budget(&self, _budget_bytes: u64) -> Result<Option<u64>> {
        Ok(None)
    }

    fn adopt_memory_governor(
        &self,
        _governor: &dyn onnx_runtime_memory_governor::MemoryGovernor,
        _tier: onnx_runtime_memory_governor::Tier,
        _holder: onnx_runtime_memory_governor::HolderId,
    ) -> Result<u64> {
        Ok(0)
    }

    /// Synchronously upload host bytes into a buffer owned by this EP.
    fn copy_from_host(&self, src: &[u8], dst: &mut DeviceBuffer) -> Result<()> {
        if !dst.device().is_host_accessible() {
            return Err(EpError::KernelFailed(format!(
                "{}: host upload is not implemented for device {:?}",
                self.name(),
                dst.device()
            )));
        }
        if src.len() > dst.len() {
            return Err(EpError::KernelFailed(format!(
                "{}: host upload of {} bytes exceeds destination {} bytes",
                self.name(),
                src.len(),
                dst.len()
            )));
        }
        if src.is_empty() {
            return Ok(());
        }
        // SAFETY: host accessibility is checked above, `dst` is uniquely
        // borrowed, and its allocation is at least `src.len()` bytes.
        unsafe {
            std::ptr::copy_nonoverlapping(src.as_ptr(), dst.as_mut_ptr().cast(), src.len());
        }
        Ok(())
    }

    /// Synchronously upload host bytes into a byte range of a buffer owned by
    /// this EP.
    fn copy_from_host_at(
        &self,
        src: &[u8],
        dst: &mut DeviceBuffer,
        byte_offset: usize,
    ) -> Result<()> {
        let end = byte_offset.checked_add(src.len()).ok_or_else(|| {
            EpError::KernelFailed(format!("{}: host upload range overflows", self.name()))
        })?;
        if end > dst.len() {
            return Err(EpError::KernelFailed(format!(
                "{}: host upload range {byte_offset}..{end} exceeds destination {} bytes",
                self.name(),
                dst.len()
            )));
        }
        if src.is_empty() {
            return Ok(());
        }
        if !dst.device().is_host_accessible() {
            return Err(EpError::KernelFailed(format!(
                "{}: ranged host upload is not implemented for device {:?}",
                self.name(),
                dst.device()
            )));
        }
        // SAFETY: host accessibility and bounds are checked above, and `dst` is
        // uniquely borrowed for the duration of the copy.
        unsafe {
            std::ptr::copy_nonoverlapping(
                src.as_ptr(),
                dst.as_mut_ptr().cast::<u8>().add(byte_offset),
                src.len(),
            );
        }
        Ok(())
    }

    /// Synchronously download a buffer owned by this EP into host bytes.
    fn copy_to_host(&self, src: &DeviceBuffer, dst: &mut [u8]) -> Result<()> {
        if !src.device().is_host_accessible() {
            return Err(EpError::KernelFailed(format!(
                "{}: host download is not implemented for device {:?}",
                self.name(),
                src.device()
            )));
        }
        if dst.len() > src.len() {
            return Err(EpError::KernelFailed(format!(
                "{}: host download of {} bytes exceeds source {} bytes",
                self.name(),
                dst.len(),
                src.len()
            )));
        }
        if dst.is_empty() {
            return Ok(());
        }
        // SAFETY: host accessibility is checked above, `dst` is uniquely
        // borrowed, and `src` contains at least `dst.len()` readable bytes.
        unsafe {
            std::ptr::copy_nonoverlapping(src.as_ptr().cast(), dst.as_mut_ptr(), dst.len());
        }
        Ok(())
    }

    /// Block until all pending work on this EP completes.
    fn sync(&self) -> Result<()>;

    /// Copy `bytes` device→device, from `src[src_offset..]` into
    /// `dst[dst_offset..]`, both allocations owned by this EP.
    ///
    /// The default errors: only device EPs with a native device-to-device copy
    /// (CUDA) implement it. Used by the speculative recurrent-state snapshot to
    /// stage the destructive GDN/conv state into device scratch WITHOUT a PCIe
    /// round-trip through host memory. The copy is stream-ordered on the EP's
    /// compute stream, so it composes with the surrounding forward passes
    /// (snapshot before the verify overwrites the state; restore before the
    /// accepted-token re-advance) with no host synchronization.
    fn copy_device_to_device(
        &self,
        _src: &DeviceBuffer,
        _src_offset: usize,
        _dst: &mut DeviceBuffer,
        _dst_offset: usize,
        _bytes: usize,
    ) -> Result<()> {
        Err(EpError::KernelFailed(format!(
            "{}: device-to-device copy is not implemented",
            self.name()
        )))
    }

    /// EP-specific optimization passes, run after the generic optimizer.
    fn custom_passes(&self) -> Vec<Box<dyn onnx_runtime_optimizer::OptimizationPass>> {
        Vec::new()
    }

    /// Nodes this EP claims unconditionally (bypassing cost-model placement).
    fn claim_nodes(&self, graph: &Graph) -> Vec<NodeId> {
        let _ = graph;
        Vec::new()
    }

    /// The `EPContext` node `source` key(s) this EP accepts for compiled-context
    /// dispatch (`docs/architecture/ORT2.md` §55.6). The keys come from the EP's own
    /// config/data — **never** hardcoded in loader/session dispatch. An empty
    /// list (the default) means the EP does not participate in `EPContext`
    /// (e.g. the pure-Rust CPU EP has no compile step).
    fn context_source_keys(&self) -> Vec<String> {
        Vec::new()
    }

    /// Produce the runtime [`EpContext`] for this EP's freshly compiled subgraph
    /// (the §55.4 dump path calls this). Default: unsupported — an EP with no
    /// compile step returns [`EpError::UnsupportedContext`].
    fn save_context(&self) -> Result<EpContext> {
        Err(EpError::UnsupportedContext {
            ep: self.name().to_string(),
        })
    }

    /// Restore this EP from a runtime [`EpContext`], skipping convert+compile
    /// (the §55.3 load path calls this). Default: unsupported — an EP that does
    /// not consume `EPContext` returns [`EpError::UnsupportedContext`].
    fn load_context(&self, ctx: &EpContext) -> Result<()> {
        let _ = ctx;
        Err(EpError::UnsupportedContext {
            ep: self.name().to_string(),
        })
    }
}

/// Error for a graph-slot operation on an EP that does not own that slot.
fn unsupported_graph_slot(ep: &str, slot: DeviceGraphSlot) -> EpError {
    EpError::KernelFailed(format!(
        "{ep}: device graph slot {slot:?} is not supported (this EP owns only the Primary slot)"
    ))
}

fn is_control_flow_or_sequence(node: &Node) -> bool {
    if !(node.domain.is_empty() || node.domain == "ai.onnx") {
        return false;
    }
    matches!(
        node.op_type.as_str(),
        "If" | "Loop"
            | "Scan"
            | "SequenceEmpty"
            | "SequenceConstruct"
            | "SequenceInsert"
            | "SequenceErase"
            | "SequenceAt"
            | "SequenceLength"
            | "SplitToSequence"
            | "ConcatFromSequence"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    static TEST_SESSION_AUTHORITY: ExecutorArtifactSessionAuthority =
        ExecutorArtifactSessionAuthority { private: () };

    fn _assert_send_sync<T: Send + Sync>() {}

    /// Leak a boxed byte slice as a stand-in host allocation.
    fn host_alloc(size: usize, align: usize) -> DeviceBuffer {
        let boxed = vec![0u8; size].into_boxed_slice();
        let ptr = Box::into_raw(boxed) as *mut c_void;
        // SAFETY: `ptr` is a valid, unique, non-null allocation of `size` bytes
        // on the host, aligned to the allocator's guarantee (>= 1); we treat the
        // CPU EP as its owner and free it exactly once in `host_free`.
        unsafe { DeviceBuffer::from_raw_parts(ptr, DeviceId::cpu(), size, align) }
    }

    fn host_free(buf: DeviceBuffer) {
        let size = buf.len();
        let ptr = buf.into_raw() as *mut u8;
        // SAFETY: reconstruct the exact `Box<[u8]>` leaked in `host_alloc` so it
        // is freed once. `into_raw` consumed the handle, so no alias remains.
        unsafe {
            drop(Box::from_raw(std::ptr::slice_from_raw_parts_mut(ptr, size)));
        }
    }

    #[test]
    fn device_buffer_is_send_sync() {
        _assert_send_sync::<DeviceBuffer>();
    }

    #[test]
    fn artifact_template_binding_issues_unique_session_generations() {
        let template = ExecutorArtifactConfigTemplate::resolved(
            ExecutorArtifactConfigAuthority::fresh(),
            DeviceId::cpu(),
            ExecutorRouteResidencyConfig::Enabled,
        );
        let executor = ExecutorInstanceId::fresh(&TEST_SESSION_AUTHORITY);
        let first = template.bind(&TEST_SESSION_AUTHORITY, executor);
        let second = template.bind(&TEST_SESSION_AUTHORITY, executor);
        assert_eq!(first.executor(), executor);
        assert_eq!(first.device(), DeviceId::cpu());
        assert_ne!(first.generation(), second.generation());
    }

    #[test]
    fn identity_exhaustion_is_sticky_and_never_reuses_a_value() {
        let counter = AtomicU64::new(u64::MAX - 1);
        assert_eq!(
            allocate_non_reusable_identity(&counter, "exhausted"),
            u64::MAX - 1
        );
        assert_eq!(counter.load(Ordering::Relaxed), u64::MAX);

        for _ in 0..2 {
            let exhausted =
                std::panic::catch_unwind(|| allocate_non_reusable_identity(&counter, "exhausted"));
            assert!(exhausted.is_err());
            assert_eq!(
                counter.load(Ordering::Relaxed),
                u64::MAX,
                "exhaustion must not wrap or publish a reusable identity"
            );
        }
    }

    #[test]
    fn artifact_finalization_is_scoped_to_exact_capability_and_epoch() {
        let expected = ExecutorArtifactConfigTemplate::resolved(
            ExecutorArtifactConfigAuthority::fresh(),
            DeviceId::cpu(),
            ExecutorRouteResidencyConfig::Enabled,
        )
        .bind(
            &TEST_SESSION_AUTHORITY,
            ExecutorInstanceId::fresh(&TEST_SESSION_AUTHORITY),
        );
        let epoch = ExecutorArtifactReadinessEpoch::new(7);
        let ExecutorArtifactFinalizationProof::Enabled(enabled) =
            expected.finalization_proof(&TEST_SESSION_AUTHORITY, epoch)
        else {
            unreachable!("test configuration is enabled")
        };
        assert_eq!(
            enabled
                .required()
                .resolve(expected, epoch)
                .expect("exact proof resolves"),
            ExecutorArtifactFinalizationOutcome::Complete {
                route_residency: ExecutorRouteResidency::required_for(expected.executor()),
            }
        );

        let foreign = ExecutorArtifactConfigTemplate::resolved(
            ExecutorArtifactConfigAuthority::fresh(),
            DeviceId::cuda(1),
            ExecutorRouteResidencyConfig::Enabled,
        )
        .bind(
            &TEST_SESSION_AUTHORITY,
            ExecutorInstanceId::fresh(&TEST_SESSION_AUTHORITY),
        );
        let ExecutorArtifactFinalizationProof::Enabled(foreign_proof) =
            foreign.finalization_proof(&TEST_SESSION_AUTHORITY, epoch)
        else {
            unreachable!("foreign test configuration is enabled")
        };
        let error = foreign_proof
            .required()
            .resolve(expected, epoch)
            .expect_err("foreign provider/device/executor/generation must fail closed");
        assert!(error.to_string().contains("finalization proof mismatch"));

        let ExecutorArtifactFinalizationProof::Enabled(stale_epoch) = expected.finalization_proof(
            &TEST_SESSION_AUTHORITY,
            ExecutorArtifactReadinessEpoch::new(6),
        ) else {
            unreachable!("test configuration is enabled")
        };
        let error = stale_epoch
            .required()
            .resolve(expected, epoch)
            .expect_err("stale readiness epoch must fail closed");
        assert!(error.to_string().contains("epoch 6"));
        assert!(error.to_string().contains("epoch 7"));
    }

    #[test]
    fn buffer_metadata_and_single_free() {
        let mut buf = host_alloc(128, 64);
        assert_eq!(buf.len(), 128);
        assert!(!buf.is_empty());
        assert_eq!(buf.alignment(), 64);
        assert_eq!(buf.device(), DeviceId::cpu());
        assert!(!buf.as_ptr().is_null());
        assert!(!buf.as_mut_ptr().is_null());
        // Single free path — a double free here would trip ASan/Miri.
        host_free(buf);
    }

    #[test]
    fn buffer_moves_across_thread() {
        let buf = host_alloc(64, 16);
        let base = buf.as_ptr() as usize;
        let handle = std::thread::spawn(move || {
            assert_eq!(buf.len(), 64);
            assert_eq!(buf.as_ptr() as usize, base);
            buf // hand ownership back so the main thread frees it once
        });
        let buf = handle.join().unwrap();
        host_free(buf);
    }

    #[test]
    fn owned_buffer_is_not_borrowed() {
        let buf = host_alloc(32, 16);
        assert!(
            !buf.is_borrowed(),
            "from_raw_parts must produce an owned buffer"
        );
        host_free(buf);
    }

    /// A borrowed buffer aliases memory owned by someone else (here a `Vec`):
    /// it reports `is_borrowed()`, exposes the aliased pointer, and consuming it
    /// via `into_raw` must NOT free the backing — the `Vec` stays valid.
    #[test]
    fn borrowed_buffer_aliases_without_owning() {
        let mut backing = vec![7u8; 64];
        let ptr = backing.as_mut_ptr() as *mut c_void;
        // SAFETY: `ptr`/`len` name `backing`'s live allocation (aligned to 1);
        // `backing` outlives the buffer and every use below, and we never write
        // through the borrowed handle.
        let buf = unsafe { DeviceBuffer::from_borrowed_parts(ptr, DeviceId::cpu(), 64, 1) };
        assert!(buf.is_borrowed());
        assert_eq!(buf.len(), 64);
        assert_eq!(buf.as_ptr(), ptr as *const c_void);
        // Consume without freeing: `into_raw` must never free a borrowed buffer.
        let raw = buf.into_raw();
        assert_eq!(raw, ptr);
        // `backing` is still fully valid — a free would be a use-after-free here.
        assert!(backing.iter().all(|&b| b == 7));
        backing[0] = 9;
        assert_eq!(backing[0], 9);
    }

    /// A host-backed binding, so the bound-ownership contract can be exercised
    /// without a device.
    fn host_binding() -> onnx_runtime_memory_governor::MemoryBinding {
        use onnx_runtime_memory_governor::{BindingRegistry, DeviceKey, HostAllocator};
        use std::sync::Arc;

        #[derive(Debug)]
        struct Pin;

        let registry = BindingRegistry::new().expect("registry");
        let context = registry
            .register_provider_context(DeviceKey::HOST, Arc::new(Pin))
            .expect("provider context");
        let authority = registry
            .register_authority(DeviceKey::HOST, Arc::new(Pin))
            .expect("authority");
        let mechanism = registry
            .register_allocator(context, authority, Arc::new(HostAllocator))
            .expect("allocator");
        registry.select(mechanism).expect("selection");
        registry.bind(DeviceKey::HOST).expect("binding")
    }

    /// Give back the host bytes a test deliberately left quarantined.
    ///
    /// Quarantine is retention, not release: the runtime keeps the address and
    /// discharges it only at confirmed context termination, which makes no
    /// allocator call because the device state is gone by then. That is right
    /// for device memory and wrong for the host heap, where nothing else ever
    /// reclaims those bytes -- so a test that asserts quarantine happened is,
    /// under Miri's leak check, a test that leaks. Rather than exempt these
    /// tests from that check or weaken it globally, the test reclaims what it
    /// asked the runtime to retain.
    fn reclaim_quarantined(binding: &onnx_runtime_memory_governor::MemoryBinding) -> usize {
        use onnx_runtime_memory_governor::{DeviceAllocator, HostAllocator};

        let quarantined = binding.quarantined().expect("quarantine list");
        for record in &quarantined {
            let Some(ptr) = std::ptr::NonNull::new(record.address as *mut u8) else {
                continue;
            };
            // SAFETY: the record carries the exact address, size and alignment
            // `HostAllocator` handed out, the runtime has stopped tracking it as
            // live, and quarantined ownership is by construction not aliased by
            // any surviving handle.
            unsafe { HostAllocator.deallocate(ptr, record.bytes, record.align) };
        }
        quarantined.len()
    }

    #[test]
    fn a_bound_buffer_carries_the_owner_that_minted_it() {
        let binding = host_binding();
        let owner = binding.allocate_owning(256, 64).expect("owning allocation");
        let identity = owner.identity();
        let address = owner.as_ptr().as_ptr() as usize;
        let buffer = DeviceBuffer::from_owning_allocation(owner, DeviceId::cpu());
        assert!(buffer.is_bound());
        assert!(!buffer.is_borrowed(), "a bound buffer owns its allocation");
        assert_eq!(buffer.len(), 256);
        assert_eq!(buffer.alignment(), 64);
        assert_eq!(buffer.as_ptr() as usize, address);
        assert_eq!(
            buffer.bound_owner().expect("bound owner").identity(),
            identity,
            "the buffer never describes a different allocation than its owner"
        );
        // The only way ownership leaves the handle is the consuming extractor,
        // and what comes back is the same generation-checked owner.
        let BoundBufferOwnership::Binding(recovered) = buffer.into_bound_owner().expect("bound")
        else {
            panic!("plain binding owner changed representation");
        };
        assert_eq!(recovered.identity(), identity);
        let outcome = recovered.release_now().expect("release");
        assert!(outcome.is_complete());
    }

    #[test]
    fn a_managed_buffer_keeps_charge_attached_to_bound_ownership() {
        use onnx_runtime_memory_governor::{
            AllocationPublication, AllocationRequest, DeviceKey, HostAllocator, LeaseLedger,
            LedgerGovernor, MemoryGovernor, MemoryRole, ProcessMemoryManager, Tier,
        };
        use std::sync::Arc;

        #[derive(Debug)]
        struct Pin;

        let manager = ProcessMemoryManager::new().unwrap();
        let context = manager
            .register_provider_context(DeviceKey::HOST, "host context", Arc::new(Pin))
            .unwrap();
        let governor = Arc::new(LedgerGovernor::new(LeaseLedger::new_for_device(
            DeviceKey::HOST,
            0,
            1024,
            0,
        )));
        let authority = manager
            .register_authority(
                DeviceKey::HOST,
                "host authority",
                Arc::new(Pin),
                governor.clone() as Arc<dyn MemoryGovernor + Send + Sync>,
            )
            .unwrap();
        let holder = manager
            .register_holder(&authority, "workspace", None)
            .unwrap();
        let mechanism = manager
            .register_allocator(
                &context,
                &authority,
                "host allocator",
                Arc::new(HostAllocator),
            )
            .unwrap();
        let owner = manager
            .bind_registered(&mechanism)
            .unwrap()
            .allocate(
                AllocationRequest::managed(
                    128,
                    16,
                    Tier::Host,
                    MemoryRole::Workspace { step_scoped: false },
                    holder,
                    128,
                ),
                AllocationPublication::exclusive(128, 128, 128),
            )
            .unwrap();
        assert_eq!(governor.used(Tier::Host), 128);
        let buffer = DeviceBuffer::from_managed_allocation(owner, DeviceId::cpu());
        assert!(buffer.is_bound());
        assert!(buffer.managed_owner().is_some());
        let BoundBufferOwnership::Managed(owner) =
            buffer.into_bound_owner().expect("managed ownership")
        else {
            panic!("manager ownership changed representation");
        };
        owner.release_now().unwrap();
        assert_eq!(governor.used(Tier::Host), 0);
    }

    #[test]
    fn failed_workspace_deallocation_quarantine_keeps_compatibility_charge() {
        use onnx_runtime_memory_governor::{
            DeviceKey, HolderId, LeaseLedger, LedgerGovernor, MemoryGovernor, MemoryRole, Tier,
        };

        #[derive(Debug)]
        struct WorkspaceDeallocationEp {
            fail: bool,
        }

        impl ExecutionProvider for WorkspaceDeallocationEp {
            fn name(&self) -> &str {
                "workspace-deallocation-test"
            }

            fn device_type(&self) -> DeviceType {
                DeviceType::Cpu
            }

            fn device_id(&self) -> DeviceId {
                DeviceId::cpu()
            }

            fn initialize(&mut self, _config: &EpConfig) -> Result<()> {
                Ok(())
            }

            fn shutdown(&mut self) -> Result<()> {
                Ok(())
            }

            fn supports_op(
                &self,
                _op: &Node,
                _opset: u64,
                _shapes: &[Shape],
                _input_dtypes: &[DataType],
                _layouts: &[TensorLayout],
            ) -> KernelMatch {
                KernelMatch::unsupported("unused test provider")
            }

            fn get_kernel(
                &self,
                _op: &Node,
                _shapes: &[Vec<usize>],
                _opset: u64,
            ) -> Result<Box<dyn Kernel>> {
                Err(EpError::KernelFailed("unused test kernel".into()))
            }

            fn allocate(&self, _size: usize, _alignment: usize) -> Result<DeviceBuffer> {
                Err(EpError::KernelFailed("unused test allocation".into()))
            }

            fn deallocate(&self, _buffer: DeviceBuffer) -> Result<()> {
                if self.fail {
                    Err(EpError::KernelFailed(
                        "injected workspace deallocation failure".into(),
                    ))
                } else {
                    Ok(())
                }
            }

            fn copy(
                &self,
                _src: &DeviceBuffer,
                _dst: &mut DeviceBuffer,
                _size: usize,
            ) -> Result<()> {
                Err(EpError::KernelFailed("unused test copy".into()))
            }

            fn copy_async(
                &self,
                _src: &DeviceBuffer,
                _dst: &mut DeviceBuffer,
                _size: usize,
            ) -> Result<Fence> {
                Err(EpError::KernelFailed("unused test async copy".into()))
            }

            fn sync(&self) -> Result<()> {
                Ok(())
            }
        }

        fn borrowed_workspace(lease: MemoryLease, backing: &mut [u8]) -> WorkspaceAllocation {
            // SAFETY: `backing` outlives the synchronous deallocation call, and
            // the test EP never reads, writes, or frees the borrowed pointer.
            let buffer = unsafe {
                DeviceBuffer::from_borrowed_parts(
                    backing.as_mut_ptr().cast(),
                    DeviceId::cpu(),
                    backing.len(),
                    1,
                )
            };
            WorkspaceAllocation::new(buffer, Some(lease))
        }

        let failed_governor =
            LedgerGovernor::new(LeaseLedger::new_for_device(DeviceKey::HOST, 0, 1024, 0));
        let failed_lease = failed_governor
            .reserve(
                Tier::Host,
                64,
                MemoryRole::Workspace { step_scoped: true },
                HolderId::new(9),
            )
            .unwrap();
        let before = quarantined_workspace_lease_count();
        let mut failed_backing = vec![0_u8; 64];
        let error = WorkspaceDeallocationEp { fail: true }
            .deallocate_workspace(borrowed_workspace(failed_lease, &mut failed_backing))
            .unwrap_err();
        assert!(error.to_string().contains("injected"));
        assert_eq!(quarantined_workspace_lease_count(), before + 1);
        assert_eq!(
            failed_governor.used(Tier::Host),
            64,
            "failed deallocation must not advertise unsettled bytes as free"
        );

        let success_governor =
            LedgerGovernor::new(LeaseLedger::new_for_device(DeviceKey::HOST, 0, 1024, 0));
        let success_lease = success_governor
            .reserve(
                Tier::Host,
                64,
                MemoryRole::Workspace { step_scoped: true },
                HolderId::new(10),
            )
            .unwrap();
        let mut success_backing = vec![0_u8; 64];
        WorkspaceDeallocationEp { fail: false }
            .deallocate_workspace(borrowed_workspace(success_lease, &mut success_backing))
            .unwrap();
        assert_eq!(
            success_governor.used(Tier::Host),
            0,
            "successful synchronous deallocation must refund its outer lease"
        );
        assert_eq!(
            quarantined_workspace_lease_count(),
            before + 1,
            "success and failure paths must not be swapped"
        );
    }

    #[test]
    fn a_raw_or_borrowed_buffer_has_no_bound_owner() {
        let raw = host_alloc(64, 16);
        assert!(!raw.is_bound());
        assert!(raw.bound_owner().is_none());
        // Failing to extract hands the buffer back untouched rather than losing
        // it, which is what lets a generation-checked path fail closed.
        let raw = raw.into_bound_owner().expect_err("not bound");
        assert_eq!(raw.len(), 64);
        host_free(raw);
    }

    #[test]
    fn into_raw_refuses_to_strip_bound_ownership() {
        let binding = host_binding();
        let owner = binding.allocate_owning(128, 16).expect("owning allocation");
        let buffer = DeviceBuffer::from_owning_allocation(owner, DeviceId::cpu());
        // Handing out the address alone would let a caller free it without
        // matching the binding identity or the allocation generation.
        //
        // Caught rather than `#[should_panic]` so the test can still run after
        // the unwind: the buffer's `Drop` quarantines on the way out, and those
        // host bytes are the test's to give back.
        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = buffer.into_raw();
        }))
        .expect_err("into_raw must refuse a bound buffer");
        let message = panic
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| panic.downcast_ref::<&str>().copied())
            .expect("panic payload is a string");
        assert!(
            message.contains("would bypass binding-identity"),
            "unexpected panic message: {message}"
        );
        assert_eq!(
            reclaim_quarantined(&binding),
            1,
            "the refused buffer is retained, not silently freed"
        );
    }

    #[test]
    fn into_raw_with_owner_is_the_explicit_escape_hatch() {
        let binding = host_binding();
        let owner = binding.allocate_owning(128, 16).expect("owning allocation");
        let expected = owner.as_ptr().as_ptr() as usize;
        let buffer = DeviceBuffer::from_owning_allocation(owner, DeviceId::cpu());
        let (ptr, owner) = buffer.into_raw_with_owner();
        assert_eq!(ptr as usize, expected);
        let BoundBufferOwnership::Binding(owner) =
            owner.expect("the release obligation travels with the pointer")
        else {
            panic!("plain binding owner changed representation");
        };
        assert!(owner.release_now().expect("release").is_complete());

        // A raw-owning buffer keeps the historical contract: no owner, and the
        // caller still owes the free.
        let raw = host_alloc(32, 8);
        let (ptr, owner) = raw.into_raw_with_owner();
        assert!(owner.is_none());
        // SAFETY: reconstruct the exact `Box<[u8]>` leaked in `host_alloc`.
        unsafe {
            drop(Box::from_raw(std::ptr::slice_from_raw_parts_mut(
                ptr as *mut u8,
                32,
            )));
        }
    }

    #[test]
    fn dropping_a_bound_buffer_quarantines_instead_of_freeing() {
        let binding = host_binding();
        let owner = binding.allocate_owning(64, 16).expect("owning allocation");
        let buffer = DeviceBuffer::from_owning_allocation(owner, DeviceId::cpu());
        drop(buffer);
        let quarantined = binding.quarantined().expect("quarantine list");
        assert_eq!(
            quarantined.len(),
            1,
            "a dropped bound buffer stays accounted for instead of being freed"
        );
        assert_eq!(quarantined[0].retained_bytes, 64);
        assert_eq!(reclaim_quarantined(&binding), 1);
    }

    #[test]
    fn borrowed_mut_buffer_writes_without_owning() {
        let mut backing = vec![0u8; 8];
        let ptr = backing.as_mut_ptr() as *mut c_void;
        // SAFETY: `backing` exclusively owns this writable region and outlives
        // the temporary alias.
        let mut buffer =
            unsafe { DeviceBuffer::from_borrowed_mut_parts(ptr, DeviceId::cpu(), 8, 1) }
                .expect("non-null backing");
        assert!(buffer.is_borrowed());
        // SAFETY: the alias has exclusive access to all eight backing bytes.
        unsafe {
            std::ptr::copy_nonoverlapping([1u8, 2, 3].as_ptr(), buffer.as_mut_ptr().cast(), 3);
        }
        assert_eq!(buffer.into_raw(), ptr);
        assert_eq!(&backing[..3], &[1, 2, 3]);
        assert!(
            unsafe {
                DeviceBuffer::from_borrowed_mut_parts(std::ptr::null_mut(), DeviceId::cpu(), 0, 1)
            }
            .is_none()
        );
    }
}
