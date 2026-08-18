//! Registry-issued memory bindings and lifetime pins.
//!
//! This module is intentionally narrower than a process memory manager. It owns
//! registration identity, current-mechanism selection, binding/allocation
//! identity, owning allocation handles, and the `Arc`s required to keep one
//! mechanism usable. It does not own allocation policy, reservations, leases,
//! queue scheduling, or reclamation of quarantined ownership.
//!
//! Release ownership is split deliberately:
//!
//! * [`BoundAllocation`] is non-RAII Phase-3 metadata whose only release path is
//!   the explicit [`MemoryBinding::release`] migration adapter.
//! * [`OwningAllocation`] is the Phase-4 owner: not `Clone`, not `Copy`, one
//!   consuming release, and a `Drop` that quarantines rather than frees.
//! * [`crate::deferred::PreparedAllocationRelease`] is final ownership that has
//!   already been detached from the live record and may be queued.

use std::collections::HashMap;
use std::fmt::Debug;
use std::num::NonZeroU64;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

use crate::deferred::{
    AllocationReleaseOutcome, AllocationReleaseState, DeferredReleaseDisposition,
    DeferredReleaseQueue, PreparedAllocationRelease, PreparedReleasePins, QuarantineReason,
    QuarantinedAllocation,
};
use crate::{
    AllocationCommitRange, DeviceAllocator, DeviceKey, MemoryError, SharedDevicePrefix,
    SharedPrefixCommitInfo,
};

static NEXT_REGISTRY_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct OpaqueIdentity {
    registry: NonZeroU64,
    serial: NonZeroU64,
}

impl Debug for OpaqueIdentity {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_tuple("opaque")
            .field(&self.registry)
            .field(&self.serial)
            .finish()
    }
}

macro_rules! opaque_identity {
    ($name:ident) => {
        #[derive(Clone, Copy, PartialEq, Eq, Hash)]
        pub struct $name(OpaqueIdentity);

        impl Debug for $name {
            fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter
                    .debug_tuple(stringify!($name))
                    .field(&self.0)
                    .finish()
            }
        }
    };
}

opaque_identity!(AuthorityIdentity);
opaque_identity!(ProviderContextIdentity);
opaque_identity!(MechanismIdentity);
opaque_identity!(BindingId);

/// Opaque registry-issued generation of one [`MemoryBinding`].
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct BindingGeneration(NonZeroU64);

impl Debug for BindingGeneration {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_tuple("BindingGeneration")
            .field(&self.0)
            .finish()
    }
}

/// Opaque registry-issued generation of one allocation at one binding.
///
/// Generations are never derived from a pointer. Reusing the same virtual
/// address therefore cannot make metadata for an earlier allocation current.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct AllocationGeneration(NonZeroU64);

impl Debug for AllocationGeneration {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_tuple("AllocationGeneration")
            .field(&self.0)
            .finish()
    }
}

/// Complete identity of one manager-issued binding.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BindingIdentity {
    id: BindingId,
    generation: BindingGeneration,
    device: DeviceKey,
    mechanism: MechanismIdentity,
    provider_context: ProviderContextIdentity,
    authority: AuthorityIdentity,
}

impl BindingIdentity {
    pub const fn id(self) -> BindingId {
        self.id
    }

    pub const fn generation(self) -> BindingGeneration {
        self.generation
    }

    pub const fn device(self) -> DeviceKey {
        self.device
    }

    pub const fn mechanism(self) -> MechanismIdentity {
        self.mechanism
    }

    pub const fn provider_context(self) -> ProviderContextIdentity {
        self.provider_context
    }

    pub const fn authority(self) -> AuthorityIdentity {
        self.authority
    }
}

/// Identity of one allocation made through a binding.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct AllocationIdentity {
    binding: BindingIdentity,
    generation: AllocationGeneration,
}

impl AllocationIdentity {
    pub const fn binding(self) -> BindingIdentity {
        self.binding
    }

    pub const fn generation(self) -> AllocationGeneration {
        self.generation
    }
}

/// Resource retained by a registered provider context or authority.
///
/// The registry treats this value as an opaque lifetime pin. Concrete managers
/// may store CUDA contexts, provider libraries, accounting authorities, or a
/// composite resource owner here.
pub trait BindingResource: Send + Sync + Debug {}

impl<T> BindingResource for T where T: Send + Sync + Debug {}

/// Whether one registered allocator is self-contained or an explicitly trusted
/// composition of multiple inner mechanism interfaces.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MechanismCoherence {
    /// One allocator implementation supplies its own ordinary and optional
    /// capability interfaces.
    SelfContained,
    /// The registrar explicitly attested that a transparent/composite wrapper
    /// routes allocation, capabilities, and canonical release coherently.
    TrustedComposite,
}

/// Lifecycle of one registered mechanism.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MechanismLifecycle {
    /// New allocations and capability operations are accepted.
    Active,
    /// New work is rejected, but existing allocations may still be explicitly
    /// released through their pinned original mechanism.
    Retired,
    /// The device/context was lost. All operations, including explicit release,
    /// are rejected until external context/process termination is confirmed.
    DeviceLost,
    /// External context/process termination was observed. Metadata is terminal;
    /// no allocator callback is made while entering this state.
    Terminated,
}

/// Read-only lifecycle information. Taking a snapshot never invokes a provider.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MechanismSnapshot {
    pub identity: MechanismIdentity,
    pub device: DeviceKey,
    pub provider_context: ProviderContextIdentity,
    pub authority: AuthorityIdentity,
    pub coherence: MechanismCoherence,
    pub lifecycle: MechanismLifecycle,
    pub live_allocations: usize,
    pub active_operations: usize,
    /// Allocations whose live record was retired into a prepared release that
    /// has not settled yet. These are [`AllocationReleaseState::Queued`].
    pub queued_releases: usize,
    /// Allocations whose ownership was retained instead of released.
    pub quarantined_allocations: usize,
    /// Bytes still owned by quarantined ownership.
    pub quarantined_bytes: u64,
}

impl MechanismSnapshot {
    /// Whether any ownership is still outstanding in any non-terminal or
    /// retained state. Removal is unsafe while this is true.
    pub const fn retains_ownership(&self) -> bool {
        self.live_allocations != 0 || self.queued_releases != 0 || self.quarantined_allocations != 0
    }
}

/// Binding/registration failure before any caller-provided device action runs.
#[derive(Debug, thiserror::Error)]
pub enum BindingError {
    #[error("binding identity space is exhausted")]
    IdentityExhausted,
    #[error("the {kind} belongs to another binding registry")]
    ForeignRegistry { kind: &'static str },
    #[error("cannot register {subject} for {actual:?}; its registered device is {expected:?}")]
    DeviceMismatch {
        subject: &'static str,
        expected: DeviceKey,
        actual: DeviceKey,
    },
    #[error("provider context {0:?} is not registered")]
    UnregisteredProviderContext(ProviderContextIdentity),
    #[error("provider context {0:?} still has a registered mechanism")]
    ProviderContextInUse(ProviderContextIdentity),
    #[error("authority {0:?} is not registered")]
    UnregisteredAuthority(AuthorityIdentity),
    #[error("authority {0:?} still has a registered mechanism")]
    AuthorityInUse(AuthorityIdentity),
    #[error("mechanism {0:?} is not registered")]
    UnregisteredMechanism(MechanismIdentity),
    #[error("device {0:?} has no selected memory mechanism")]
    NoSelectedMechanism(DeviceKey),
    #[error("mechanism {mechanism:?} is {lifecycle:?}; {operation} is not permitted")]
    InactiveMechanism {
        mechanism: MechanismIdentity,
        lifecycle: MechanismLifecycle,
        operation: &'static str,
    },
    #[error("device {device:?} was lost: {reason}")]
    DeviceLost { device: DeviceKey, reason: Arc<str> },
    #[error("binding mismatch: expected {expected:?}, but metadata belongs to {actual:?}")]
    BindingMismatch {
        expected: BindingId,
        actual: BindingId,
    },
    #[error("allocation metadata {0:?} is stale or was already explicitly released")]
    StaleAllocation(AllocationIdentity),
    #[error(
        "allocation {identity:?} still has {views} outstanding view(s); physical release is not \
         permitted while a borrowed view or alias may still be used"
    )]
    OutstandingViews {
        identity: AllocationIdentity,
        views: usize,
    },
    #[error(
        "release of allocation {identity:?} left {retained_bytes} byte(s) in the {state} state: \
         {reason}"
    )]
    ReleaseQuarantined {
        identity: AllocationIdentity,
        state: AllocationReleaseState,
        reason: QuarantineReason,
        retained_bytes: u64,
    },
    #[error(
        "mechanism {mechanism:?} still owns {quarantined} quarantined allocation(s); removal \
         would lose ownership that was deliberately retained"
    )]
    QuarantinedOwnership {
        mechanism: MechanismIdentity,
        quarantined: usize,
    },
    #[error("view range {offset}..{end} exceeds allocation size {allocation_bytes}")]
    ViewOutOfBounds {
        offset: usize,
        end: usize,
        allocation_bytes: usize,
    },
    #[error("binding registry lock was poisoned while {operation}")]
    LockPoisoned { operation: &'static str },
    #[error(
        "provider context {context:?} still has {active_operations} active mechanism operation(s)"
    )]
    ContextNotQuiescent {
        context: ProviderContextIdentity,
        active_operations: usize,
    },
    #[error(transparent)]
    Memory(#[from] MemoryError),
}

#[derive(Debug)]
struct IdentitySource {
    registry: NonZeroU64,
    next: AtomicU64,
}

impl IdentitySource {
    fn new() -> Result<Self, BindingError> {
        let registry = next_nonzero(&NEXT_REGISTRY_ID)?;
        Ok(Self {
            registry,
            next: AtomicU64::new(1),
        })
    }

    fn opaque(&self) -> Result<OpaqueIdentity, BindingError> {
        Ok(OpaqueIdentity {
            registry: self.registry,
            serial: next_nonzero(&self.next)?,
        })
    }

    fn binding_generation(&self) -> Result<BindingGeneration, BindingError> {
        Ok(BindingGeneration(next_nonzero(&self.next)?))
    }

    fn allocation_generation(&self) -> Result<AllocationGeneration, BindingError> {
        Ok(AllocationGeneration(next_nonzero(&self.next)?))
    }
}

fn next_nonzero(counter: &AtomicU64) -> Result<NonZeroU64, BindingError> {
    let value = counter
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
            current.checked_add(1)
        })
        .map_err(|_| BindingError::IdentityExhausted)?;
    NonZeroU64::new(value).ok_or(BindingError::IdentityExhausted)
}

#[derive(Debug)]
struct ProviderContextEntry {
    identity: ProviderContextIdentity,
    device: DeviceKey,
    _resource: Arc<dyn BindingResource>,
}

#[derive(Debug)]
struct AuthorityEntry {
    identity: AuthorityIdentity,
    device: DeviceKey,
    _resource: Arc<dyn BindingResource>,
}

#[derive(Clone, Copy, Debug)]
struct AllocationRecord {
    identity: AllocationIdentity,
    ptr: usize,
    bytes: usize,
    align: usize,
}

#[derive(Debug)]
struct MechanismState {
    lifecycle: MechanismLifecycle,
    loss_reason: Option<Arc<str>>,
    allocations: HashMap<AllocationGeneration, AllocationRecord>,
    /// Prepared releases that have left the live map but have not settled.
    queued_releases: usize,
    /// Ownership retained instead of released, keyed by the generation it was
    /// prepared from so a settled request can never be recorded twice.
    quarantined: HashMap<AllocationGeneration, QuarantinedAllocation>,
}

/// One registered allocator together with the resources its destructor needs.
///
/// Declaration order here is load-bearing rather than incidental. Rust drops
/// struct fields in declaration order, so the allocator is destroyed first and
/// its `Drop` still observes live provider-context and authority resources. A
/// third-party allocator may release device state from `Drop`, and that work
/// needs its provider context (a CUDA context, a loaded provider library) alive.
/// The provider context is the deepest resource, so it is released last.
///
/// Keeping the three pins in one owner type means the ordering cannot be broken
/// by unrelated field edits to [`MechanismEntry`].
#[derive(Debug)]
struct MechanismResources {
    /// Destroyed first, while both pins below are still alive.
    allocator: Arc<dyn DeviceAllocator>,
    /// Outlives the allocator, so `Drop` can still settle accounting identity.
    authority: Arc<AuthorityEntry>,
    /// Outlives the allocator and the authority; released last.
    context: Arc<ProviderContextEntry>,
}

#[derive(Debug)]
pub(crate) struct MechanismEntry {
    identity: MechanismIdentity,
    device: DeviceKey,
    coherence: MechanismCoherence,
    state: Mutex<MechanismState>,
    active_operations: AtomicUsize,
    /// Declared last so the allocator and its pins are released only after the
    /// entry's own identity/lifecycle state is gone.
    resources: MechanismResources,
}

/// Whether a prepared release may still reach the allocator.
///
/// Read under the mechanism lock and acted on after that lock is dropped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ReleaseGate {
    /// `Active` or `Retired`: the pinned allocator may perform the release.
    Allowed,
    /// The device or context was lost. The allocator must never be called.
    DeviceLost,
    /// Termination was confirmed; the device state provably no longer exists.
    Terminated,
    /// The mechanism lock was poisoned; fail safe without calling anything.
    Poisoned,
}

impl MechanismEntry {
    fn allocator(&self) -> &dyn DeviceAllocator {
        self.resources.allocator.as_ref()
    }

    fn context_identity(&self) -> ProviderContextIdentity {
        self.resources.context.identity
    }

    fn authority_identity(&self) -> AuthorityIdentity {
        self.resources.authority.identity
    }

    fn lock_state(
        &self,
        operation: &'static str,
    ) -> Result<MutexGuard<'_, MechanismState>, BindingError> {
        self.state
            .lock()
            .map_err(|_| BindingError::LockPoisoned { operation })
    }

    fn inactive_error(&self, state: &MechanismState, operation: &'static str) -> BindingError {
        match state.lifecycle {
            MechanismLifecycle::DeviceLost => BindingError::DeviceLost {
                device: self.device,
                reason: state
                    .loss_reason
                    .clone()
                    .unwrap_or_else(|| Arc::from("provider did not supply a reason")),
            },
            lifecycle => BindingError::InactiveMechanism {
                mechanism: self.identity,
                lifecycle,
                operation,
            },
        }
    }

    fn begin_active(
        self: &Arc<Self>,
        operation: &'static str,
    ) -> Result<MechanismOperation, BindingError> {
        let state = self.lock_state(operation)?;
        if state.lifecycle != MechanismLifecycle::Active {
            return Err(self.inactive_error(&state, operation));
        }
        self.active_operations.fetch_add(1, Ordering::AcqRel);
        drop(state);
        Ok(MechanismOperation {
            mechanism: Arc::clone(self),
        })
    }

    /// Detach final ownership of `allocation` from the live record.
    ///
    /// The binding identity and the allocation generation are both matched, and
    /// the live record is removed **exactly once** under this mechanism's lock,
    /// so two racing final releases cannot both proceed and a stale handle over
    /// a reused virtual address cannot match. No allocator call is made here.
    fn begin_release(
        self: &Arc<Self>,
        expected_binding: BindingIdentity,
        allocation: &BoundAllocation,
    ) -> Result<MechanismOperation, BindingError> {
        let operation = "preparing explicit release";
        let mut state = self.lock_state(operation)?;
        match state.lifecycle {
            MechanismLifecycle::Active | MechanismLifecycle::Retired => {}
            _ => return Err(self.inactive_error(&state, operation)),
        }
        validate_binding_identity(expected_binding, allocation.identity.binding)?;
        let Some(record) = state.allocations.get(&allocation.identity.generation) else {
            return Err(BindingError::StaleAllocation(allocation.identity));
        };
        if !allocation.matches_record(record) {
            return Err(BindingError::StaleAllocation(allocation.identity));
        }
        state.allocations.remove(&allocation.identity.generation);
        state.queued_releases += 1;
        self.active_operations.fetch_add(1, Ordering::AcqRel);
        drop(state);
        Ok(MechanismOperation {
            mechanism: Arc::clone(self),
        })
    }

    /// Whether a prepared release may still call the allocator.
    pub(crate) fn release_gate(&self) -> ReleaseGate {
        let Ok(state) = self.state.lock() else {
            return ReleaseGate::Poisoned;
        };
        match state.lifecycle {
            MechanismLifecycle::Active | MechanismLifecycle::Retired => ReleaseGate::Allowed,
            MechanismLifecycle::DeviceLost => ReleaseGate::DeviceLost,
            MechanismLifecycle::Terminated => ReleaseGate::Terminated,
        }
    }

    /// Record that a queued release completed. Never calls the allocator.
    pub(crate) fn settle_release(&self, identity: AllocationIdentity) {
        let Ok(mut state) = self.state.lock() else {
            return;
        };
        state.queued_releases = state.queued_releases.saturating_sub(1);
        debug_assert!(
            !state.allocations.contains_key(&identity.generation),
            "a settled release must not leave a live record behind"
        );
    }

    /// Record retained ownership. Never calls the allocator and never waits.
    pub(crate) fn settle_quarantine(&self, record: QuarantinedAllocation) {
        let Ok(mut state) = self.state.lock() else {
            return;
        };
        state.queued_releases = state.queued_releases.saturating_sub(1);
        state.quarantined.insert(record.identity.generation, record);
    }

    pub(crate) fn allocator_arc(&self) -> Arc<dyn DeviceAllocator> {
        Arc::clone(&self.resources.allocator)
    }

    fn quarantined(&self) -> Result<Vec<QuarantinedAllocation>, BindingError> {
        let state = self.lock_state("listing quarantined ownership")?;
        Ok(state.quarantined.values().copied().collect())
    }

    fn record_allocation(&self, record: AllocationRecord) -> Result<(), BindingError> {
        let mut state = self.lock_state("recording allocation identity")?;
        state.allocations.insert(record.identity.generation, record);
        Ok(())
    }

    fn validate_allocation(
        &self,
        expected_binding: BindingIdentity,
        allocation: &BoundAllocation,
        operation: &'static str,
    ) -> Result<(), BindingError> {
        validate_binding_identity(expected_binding, allocation.identity.binding)?;
        let state = self.lock_state(operation)?;
        if state.lifecycle != MechanismLifecycle::Active {
            return Err(self.inactive_error(&state, operation));
        }
        let Some(record) = state.allocations.get(&allocation.identity.generation) else {
            return Err(BindingError::StaleAllocation(allocation.identity));
        };
        if !allocation.matches_record(record) {
            return Err(BindingError::StaleAllocation(allocation.identity));
        }
        Ok(())
    }

    fn validate_view(
        &self,
        expected_binding: BindingIdentity,
        view: &BoundMemoryView,
        operation: &'static str,
    ) -> Result<(), BindingError> {
        validate_binding_identity(expected_binding, view.identity.binding)?;
        let state = self.lock_state(operation)?;
        if state.lifecycle != MechanismLifecycle::Active {
            return Err(self.inactive_error(&state, operation));
        }
        let Some(record) = state.allocations.get(&view.identity.generation) else {
            return Err(BindingError::StaleAllocation(view.identity));
        };
        if record.identity != view.identity
            || record.ptr != view.allocation_ptr.as_ptr() as usize
            || record.bytes != view.allocation_bytes
            || record.align != view.align
        {
            return Err(BindingError::StaleAllocation(view.identity));
        }
        Ok(())
    }

    fn snapshot(&self) -> Result<MechanismSnapshot, BindingError> {
        let state = self.lock_state("taking a mechanism snapshot")?;
        Ok(MechanismSnapshot {
            identity: self.identity,
            device: self.device,
            provider_context: self.context_identity(),
            authority: self.authority_identity(),
            coherence: self.coherence,
            lifecycle: state.lifecycle,
            live_allocations: state.allocations.len(),
            active_operations: self.active_operations.load(Ordering::Acquire),
            queued_releases: state.queued_releases,
            quarantined_allocations: state.quarantined.len(),
            quarantined_bytes: state
                .quarantined
                .values()
                .map(|record| record.retained_bytes)
                .sum(),
        })
    }
}

#[derive(Debug)]
pub(crate) struct MechanismOperation {
    mechanism: Arc<MechanismEntry>,
}

impl Drop for MechanismOperation {
    fn drop(&mut self) {
        self.mechanism
            .active_operations
            .fetch_sub(1, Ordering::AcqRel);
    }
}

#[derive(Debug, Default)]
struct RegistryState {
    contexts: HashMap<ProviderContextIdentity, Arc<ProviderContextEntry>>,
    authorities: HashMap<AuthorityIdentity, Arc<AuthorityEntry>>,
    mechanisms: HashMap<MechanismIdentity, Arc<MechanismEntry>>,
    selected: HashMap<DeviceKey, MechanismIdentity>,
}

#[derive(Debug)]
struct RegistryInner {
    identities: Arc<IdentitySource>,
    state: Mutex<RegistryState>,
    #[cfg(test)]
    hooks: Mutex<Vec<RegistryHook>>,
}

/// What a test hook is keyed on.
#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum HookSubject {
    Mechanism(MechanismIdentity),
    Device(DeviceKey),
}

/// A point inside a registry transition a test may pause at.
///
/// Selection and lifecycle transitions each span the registry lock and a
/// mechanism lock, which may never be held together. Pausing between those
/// phases is what makes the resulting races reproducible by construction rather
/// than by timing.
#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum HookPhase {
    /// In `select`, after the candidate validated `Active` and before its
    /// selection is published.
    SelectAfterValidation,
    /// In `select`, after the selection is published and before the candidate is
    /// re-checked and possibly withdrawn.
    SelectAfterPublish,
    /// In `retire`, between the mechanism-lock lifecycle phase and the
    /// registry-lock selection phase.
    RetireBetweenPhases,
    /// In `invalidate_device`, between the mechanism-lock lifecycle phase and
    /// the registry-lock selection phase.
    InvalidateBetweenPhases,
}

#[cfg(test)]
#[derive(Clone, Debug)]
struct RegistryHook {
    subject: HookSubject,
    phase: HookPhase,
    entered: Arc<std::sync::Barrier>,
    resume: Arc<std::sync::Barrier>,
}

/// A narrow registry for provider/context pins and binding identity.
///
/// # Lock order
///
/// There are two lock classes: the registry lock protects registration and
/// current selection; each mechanism lock protects only lifecycle, allocation
/// identities, queued releases, and quarantined ownership. They are never held
/// together. Allocator/capability callbacks, deferred-queue callbacks, waits,
/// and device operations run with neither lock held.
#[derive(Clone, Debug)]
pub struct BindingRegistry {
    inner: Arc<RegistryInner>,
}

impl BindingRegistry {
    pub fn new() -> Result<Self, BindingError> {
        Ok(Self {
            inner: Arc::new(RegistryInner {
                identities: Arc::new(IdentitySource::new()?),
                state: Mutex::new(RegistryState::default()),
                #[cfg(test)]
                hooks: Mutex::new(Vec::new()),
            }),
        })
    }

    fn lock_state(
        &self,
        operation: &'static str,
    ) -> Result<MutexGuard<'_, RegistryState>, BindingError> {
        self.inner
            .state
            .lock()
            .map_err(|_| BindingError::LockPoisoned { operation })
    }

    pub fn register_provider_context(
        &self,
        device: DeviceKey,
        resource: Arc<dyn BindingResource>,
    ) -> Result<RegisteredProviderContext, BindingError> {
        let identity = ProviderContextIdentity(self.inner.identities.opaque()?);
        let entry = Arc::new(ProviderContextEntry {
            identity,
            device,
            _resource: resource,
        });
        self.lock_state("registering a provider context")?
            .contexts
            .insert(identity, entry);
        Ok(RegisteredProviderContext { identity, device })
    }

    pub fn register_authority(
        &self,
        device: DeviceKey,
        resource: Arc<dyn BindingResource>,
    ) -> Result<RegisteredAuthority, BindingError> {
        let identity = AuthorityIdentity(self.inner.identities.opaque()?);
        let entry = Arc::new(AuthorityEntry {
            identity,
            device,
            _resource: resource,
        });
        self.lock_state("registering an authority")?
            .authorities
            .insert(identity, entry);
        Ok(RegisteredAuthority { identity, device })
    }

    pub fn register_allocator(
        &self,
        context: RegisteredProviderContext,
        authority: RegisteredAuthority,
        allocator: Arc<dyn DeviceAllocator>,
    ) -> Result<RegisteredMechanism, BindingError> {
        self.register_mechanism(
            context,
            authority,
            allocator,
            MechanismCoherence::SelfContained,
        )
    }

    /// Register a transparent/composite wrapper as one trusted coherent bundle.
    ///
    /// # Safety
    ///
    /// The registrar must ensure ordinary allocation, optional capabilities, and
    /// canonical release all reach one coherent device mechanism, authority, and
    /// provider context. Rust cannot prove that a hostile split-inner wrapper
    /// satisfies this raw-pointer contract.
    pub unsafe fn register_trusted_composite(
        &self,
        context: RegisteredProviderContext,
        authority: RegisteredAuthority,
        allocator: Arc<dyn DeviceAllocator>,
    ) -> Result<RegisteredMechanism, BindingError> {
        self.register_mechanism(
            context,
            authority,
            allocator,
            MechanismCoherence::TrustedComposite,
        )
    }

    fn register_mechanism(
        &self,
        context: RegisteredProviderContext,
        authority: RegisteredAuthority,
        allocator: Arc<dyn DeviceAllocator>,
        coherence: MechanismCoherence,
    ) -> Result<RegisteredMechanism, BindingError> {
        self.ensure_local(context.identity.0, "provider context")?;
        self.ensure_local(authority.identity.0, "authority")?;
        let allocator_device = allocator.device();
        let (context_entry, authority_entry) = {
            let state = self.lock_state("looking up mechanism resources")?;
            let context_entry = state
                .contexts
                .get(&context.identity)
                .cloned()
                .ok_or(BindingError::UnregisteredProviderContext(context.identity))?;
            let authority_entry = state
                .authorities
                .get(&authority.identity)
                .cloned()
                .ok_or(BindingError::UnregisteredAuthority(authority.identity))?;
            (context_entry, authority_entry)
        };
        if context_entry.device != authority_entry.device {
            return Err(BindingError::DeviceMismatch {
                subject: "authority",
                expected: context_entry.device,
                actual: authority_entry.device,
            });
        }
        if context_entry.device != allocator_device {
            return Err(BindingError::DeviceMismatch {
                subject: "allocator",
                expected: context_entry.device,
                actual: allocator_device,
            });
        }

        let identity = MechanismIdentity(self.inner.identities.opaque()?);
        let mut state = self.lock_state("registering a mechanism")?;
        let entry = Arc::new(MechanismEntry {
            identity,
            device: context_entry.device,
            coherence,
            state: Mutex::new(MechanismState {
                lifecycle: MechanismLifecycle::Active,
                loss_reason: None,
                allocations: HashMap::new(),
                queued_releases: 0,
                quarantined: HashMap::new(),
            }),
            active_operations: AtomicUsize::new(0),
            resources: MechanismResources {
                allocator,
                authority: authority_entry,
                context: context_entry,
            },
        });
        state.mechanisms.insert(identity, entry);
        state.selected.entry(context.device).or_insert(identity);
        Ok(RegisteredMechanism {
            identity,
            device: context.device,
            coherence,
        })
    }

    /// Make `mechanism` the mechanism that later `bind(device)` calls use.
    ///
    /// A mechanism can be retired or lost between validation and publication, so
    /// the candidate is re-checked after it is published. When that re-check
    /// fails the selection is withdrawn, and the withdrawal never leaves a dead
    /// or unregistered mechanism selected: it will not overwrite a newer
    /// selection, restores the previous selection only while that is still
    /// registered and `Active`, and otherwise clears the selection so a later
    /// registration for the device can heal it.
    pub fn select(&self, mechanism: RegisteredMechanism) -> Result<(), BindingError> {
        self.ensure_local(mechanism.identity.0, "mechanism")?;
        let entry = {
            let state = self.lock_state("selecting a mechanism")?;
            state
                .mechanisms
                .get(&mechanism.identity)
                .cloned()
                .ok_or(BindingError::UnregisteredMechanism(mechanism.identity))?
        };
        let snapshot = entry.snapshot()?;
        if snapshot.lifecycle != MechanismLifecycle::Active {
            return Err(BindingError::InactiveMechanism {
                mechanism: mechanism.identity,
                lifecycle: snapshot.lifecycle,
                operation: "selecting a mechanism",
            });
        }
        #[cfg(test)]
        self.wait_at_hook(
            HookSubject::Mechanism(mechanism.identity),
            HookPhase::SelectAfterValidation,
        );
        let prior = self
            .lock_state("publishing mechanism selection")?
            .selected
            .insert(mechanism.device, mechanism.identity);
        #[cfg(test)]
        self.wait_at_hook(
            HookSubject::Mechanism(mechanism.identity),
            HookPhase::SelectAfterPublish,
        );
        let published = entry.snapshot()?;
        if published.lifecycle != MechanismLifecycle::Active {
            self.withdraw_failed_selection(mechanism.device, mechanism.identity, prior)?;
            return Err(BindingError::InactiveMechanism {
                mechanism: mechanism.identity,
                lifecycle: published.lifecycle,
                operation: "selecting a mechanism",
            });
        }
        Ok(())
    }

    /// Withdraw a published selection whose candidate turned out to be inactive.
    ///
    /// Three rules keep `selected` pointing only at a live registration:
    ///
    /// 1. the candidate is only replaced while it still owns `device`'s
    ///    selection, so a newer selection published by another thread is never
    ///    overwritten by a losing candidate;
    /// 2. `prior` is restored only while it is still registered *and* `Active`,
    ///    so a concurrently retired, lost, or removed prior is not resurrected;
    /// 3. otherwise the selection is cleared, because an absent selection is the
    ///    only state a later [`BindingRegistry::register_allocator`] can heal.
    ///
    /// The restored mechanism is re-checked after publication for the same
    /// reason the candidate was, and that retry carries no further fallback, so
    /// the loop runs at most twice.
    ///
    /// # Lock order
    ///
    /// The registry lock is released before every mechanism lifecycle snapshot,
    /// so the two lock classes are still never held together and no allocator or
    /// capability callback runs here.
    fn withdraw_failed_selection(
        &self,
        device: DeviceKey,
        candidate: MechanismIdentity,
        prior: Option<MechanismIdentity>,
    ) -> Result<(), BindingError> {
        const OPERATION: &str = "withdrawing inactive mechanism selection";
        let mut owner = candidate;
        let mut replacement = prior;
        loop {
            let restorable = match replacement {
                Some(identity) => {
                    let entry = {
                        let state = self.lock_state(OPERATION)?;
                        if state.selected.get(&device) != Some(&owner) {
                            return Ok(());
                        }
                        state.mechanisms.get(&identity).cloned()
                    };
                    match entry {
                        Some(entry)
                            if entry.snapshot()?.lifecycle == MechanismLifecycle::Active =>
                        {
                            Some(entry)
                        }
                        _ => None,
                    }
                }
                None => None,
            };

            let restored = {
                let mut state = self.lock_state(OPERATION)?;
                if state.selected.get(&device) != Some(&owner) {
                    return Ok(());
                }
                // Registration is re-confirmed under the lock that publishes it,
                // so a `remove` racing the lifecycle snapshot above cannot leave
                // an unregistered identity selected.
                match restorable {
                    Some(entry) if state.mechanisms.contains_key(&entry.identity) => {
                        state.selected.insert(device, entry.identity);
                        entry
                    }
                    _ => {
                        state.selected.remove(&device);
                        return Ok(());
                    }
                }
            };

            if restored.snapshot()?.lifecycle == MechanismLifecycle::Active {
                return Ok(());
            }
            owner = restored.identity;
            replacement = None;
        }
    }

    #[cfg(test)]
    fn install_hook(&self, hook: RegistryHook) {
        self.inner
            .hooks
            .lock()
            .expect("registry test hook lock")
            .push(hook);
    }

    #[cfg(test)]
    fn wait_at_hook(&self, subject: HookSubject, phase: HookPhase) {
        let hook = self
            .inner
            .hooks
            .lock()
            .expect("registry test hook lock")
            .iter()
            .find(|hook| hook.subject == subject && hook.phase == phase)
            .cloned();
        // The hook lock is released before waiting so a paused caller never
        // blocks the test thread that is about to release it.
        if let Some(hook) = hook {
            hook.entered.wait();
            hook.resume.wait();
        }
    }

    pub fn bind(&self, device: DeviceKey) -> Result<MemoryBinding, BindingError> {
        let entry = {
            let state = self.lock_state("looking up the selected mechanism")?;
            let identity = state
                .selected
                .get(&device)
                .copied()
                .ok_or(BindingError::NoSelectedMechanism(device))?;
            state
                .mechanisms
                .get(&identity)
                .cloned()
                .ok_or(BindingError::UnregisteredMechanism(identity))?
        };
        self.issue_binding(entry)
    }

    pub fn bind_registered(
        &self,
        mechanism: RegisteredMechanism,
    ) -> Result<MemoryBinding, BindingError> {
        self.ensure_local(mechanism.identity.0, "mechanism")?;
        let entry = self
            .lock_state("looking up a registered mechanism")?
            .mechanisms
            .get(&mechanism.identity)
            .cloned()
            .ok_or(BindingError::UnregisteredMechanism(mechanism.identity))?;
        self.issue_binding(entry)
    }

    fn issue_binding(&self, entry: Arc<MechanismEntry>) -> Result<MemoryBinding, BindingError> {
        let operation = entry.begin_active("issuing a binding")?;
        let identity = BindingIdentity {
            id: BindingId(self.inner.identities.opaque()?),
            generation: self.inner.identities.binding_generation()?,
            device: entry.device,
            mechanism: entry.identity,
            provider_context: entry.context_identity(),
            authority: entry.authority_identity(),
        };
        drop(operation);
        Ok(MemoryBinding {
            identity,
            identities: Arc::clone(&self.inner.identities),
            mechanism: entry,
        })
    }

    /// Stop issuing new work through `mechanism`.
    ///
    /// Existing allocations keep the original allocator/context/authority pinned
    /// and may still use [`MemoryBinding::release`] explicitly.
    ///
    /// The lifecycle is made terminal *before* the selection is dropped. The two
    /// lock classes cannot be held together, so the reverse order leaves a window
    /// in which a concurrent `select` that already validated this mechanism
    /// publishes it after the clear and still observes `Active` at its own
    /// re-check, wedging the device on a retired selection. Retiring first means
    /// any such `select` must fail its re-check and withdraw itself.
    pub fn retire(&self, mechanism: RegisteredMechanism) -> Result<(), BindingError> {
        self.ensure_local(mechanism.identity.0, "mechanism")?;
        let entry = {
            let state = self.lock_state("retiring a mechanism")?;
            state
                .mechanisms
                .get(&mechanism.identity)
                .cloned()
                .ok_or(BindingError::UnregisteredMechanism(mechanism.identity))?
        };
        {
            let mut mechanism_state = entry.lock_state("retiring a mechanism")?;
            if mechanism_state.lifecycle == MechanismLifecycle::Active {
                mechanism_state.lifecycle = MechanismLifecycle::Retired;
            }
        }
        #[cfg(test)]
        self.wait_at_hook(
            HookSubject::Mechanism(mechanism.identity),
            HookPhase::RetireBetweenPhases,
        );
        self.drop_selection_of(mechanism.device, mechanism.identity, "retiring a mechanism")
    }

    /// Drop `device`'s selection while it still names `mechanism`.
    ///
    /// Never touches a selection naming anything else, so a healthy selection
    /// published concurrently is left alone.
    fn drop_selection_of(
        &self,
        device: DeviceKey,
        mechanism: MechanismIdentity,
        operation: &'static str,
    ) -> Result<(), BindingError> {
        let mut state = self.lock_state(operation)?;
        if state.selected.get(&device) == Some(&mechanism) {
            state.selected.remove(&device);
        }
        Ok(())
    }

    /// Invalidate every mechanism and binding for `device`.
    ///
    /// This method changes identity/lifetime state only. It does not invoke a
    /// device callback, free physical memory, release a lease, or refund quota.
    ///
    /// Like [`BindingRegistry::retire`], every affected mechanism is made
    /// terminal before the selection is dropped, so a `select` racing device loss
    /// cannot leave a lost mechanism selected. Only a selection naming a
    /// mechanism this call actually invalidated is dropped, so a mechanism
    /// registered after this call returns is never deselected by it. A
    /// registration that lands while this call is in flight may still end up
    /// unselected, because the slot it tried to claim was held by an identity
    /// this call then dropped; that fails closed, and the next registration or
    /// explicit [`BindingRegistry::select`] restores a selection.
    pub fn invalidate_device(
        &self,
        device: DeviceKey,
        reason: impl Into<Arc<str>>,
    ) -> Result<(), BindingError> {
        let reason = reason.into();
        let entries = {
            let state = self.lock_state("invalidating a device")?;
            state
                .mechanisms
                .values()
                .filter(|entry| entry.device == device)
                .cloned()
                .collect::<Vec<_>>()
        };
        for entry in &entries {
            let mut state = entry.lock_state("invalidating a device binding")?;
            if state.lifecycle != MechanismLifecycle::Terminated {
                state.lifecycle = MechanismLifecycle::DeviceLost;
                state.loss_reason = Some(Arc::clone(&reason));
            }
        }
        #[cfg(test)]
        self.wait_at_hook(
            HookSubject::Device(device),
            HookPhase::InvalidateBetweenPhases,
        );
        let mut state = self.lock_state("invalidating a device")?;
        let invalidated = state
            .selected
            .get(&device)
            .is_some_and(|selected| entries.iter().any(|entry| entry.identity == *selected));
        if invalidated {
            state.selected.remove(&device);
        }
        Ok(())
    }

    /// Record externally observed provider-context/process termination.
    ///
    /// This is the device-loss teardown boundary. Allocation identities become
    /// terminal without calling the allocator. Accounting/delegated quota must be
    /// reconciled by the owning authority only after its own required process or
    /// context termination observation.
    pub fn confirm_context_terminated(
        &self,
        context: RegisteredProviderContext,
    ) -> Result<(), BindingError> {
        self.ensure_local(context.identity.0, "provider context")?;
        let entries = {
            let state = self.lock_state("looking up a terminated provider context")?;
            if !state.contexts.contains_key(&context.identity) {
                return Err(BindingError::UnregisteredProviderContext(context.identity));
            }
            state
                .mechanisms
                .values()
                .filter(|entry| entry.context_identity() == context.identity)
                .cloned()
                .collect::<Vec<_>>()
        };
        for entry in &entries {
            let state = entry.lock_state("checking provider context quiescence")?;
            if state.lifecycle != MechanismLifecycle::DeviceLost {
                return Err(entry.inactive_error(
                    &state,
                    "confirming termination before device-loss invalidation",
                ));
            }
            let active_operations = entry.active_operations.load(Ordering::Acquire);
            if active_operations != 0 {
                return Err(BindingError::ContextNotQuiescent {
                    context: context.identity,
                    active_operations,
                });
            }
        }
        for entry in entries {
            let mut state = entry.lock_state("confirming provider context termination")?;
            state.lifecycle = MechanismLifecycle::Terminated;
            state.allocations.clear();
            // Confirmed context/process termination is the one point where
            // quarantined device ownership provably no longer exists, so this is
            // where retained ownership is discharged. No allocator call is made.
            state.quarantined.clear();
        }
        Ok(())
    }

    /// Ownership this mechanism deliberately retained instead of releasing.
    ///
    /// Taking this list never invokes a provider and never calls an allocator.
    pub fn quarantined(
        &self,
        mechanism: RegisteredMechanism,
    ) -> Result<Vec<QuarantinedAllocation>, BindingError> {
        self.ensure_local(mechanism.identity.0, "mechanism")?;
        let entry = self
            .lock_state("looking up quarantined ownership")?
            .mechanisms
            .get(&mechanism.identity)
            .cloned()
            .ok_or(BindingError::UnregisteredMechanism(mechanism.identity))?;
        entry.quarantined()
    }

    /// Remove the registry's provider-context pin after all mechanism
    /// registrations using it have been removed. Existing binding handles keep
    /// their own pin until they retire.
    pub fn remove_provider_context(
        &self,
        context: RegisteredProviderContext,
    ) -> Result<(), BindingError> {
        self.ensure_local(context.identity.0, "provider context")?;
        let mut state = self.lock_state("removing a provider context")?;
        if !state.contexts.contains_key(&context.identity) {
            return Err(BindingError::UnregisteredProviderContext(context.identity));
        }
        if state
            .mechanisms
            .values()
            .any(|entry| entry.context_identity() == context.identity)
        {
            return Err(BindingError::ProviderContextInUse(context.identity));
        }
        state.contexts.remove(&context.identity);
        Ok(())
    }

    /// Remove the registry's authority pin after all mechanism registrations
    /// using it have been removed. This does not refund charges or delegated
    /// quota; the authority owner performs accounting reconciliation separately.
    pub fn remove_authority(&self, authority: RegisteredAuthority) -> Result<(), BindingError> {
        self.ensure_local(authority.identity.0, "authority")?;
        let mut state = self.lock_state("removing an authority")?;
        if !state.authorities.contains_key(&authority.identity) {
            return Err(BindingError::UnregisteredAuthority(authority.identity));
        }
        if state
            .mechanisms
            .values()
            .any(|entry| entry.authority_identity() == authority.identity)
        {
            return Err(BindingError::AuthorityInUse(authority.identity));
        }
        state.authorities.remove(&authority.identity);
        Ok(())
    }

    /// Remove a terminal/retired registration once no allocation metadata,
    /// queued release, quarantined ownership, or active callback remains.
    /// Existing binding/capability handles still pin the entry and resources,
    /// but remain inactive.
    ///
    /// Queued and quarantined ownership both block removal: a queued request
    /// still holds an active-operation pin, and quarantined ownership is
    /// reported separately so the caller learns *why* removal is unsafe rather
    /// than seeing a bare lifecycle complaint.
    pub fn remove(&self, mechanism: RegisteredMechanism) -> Result<(), BindingError> {
        self.ensure_local(mechanism.identity.0, "mechanism")?;
        let entry = {
            let state = self.lock_state("checking mechanism teardown")?;
            state
                .mechanisms
                .get(&mechanism.identity)
                .cloned()
                .ok_or(BindingError::UnregisteredMechanism(mechanism.identity))?
        };
        let snapshot = entry.snapshot()?;
        if snapshot.quarantined_allocations != 0 {
            return Err(BindingError::QuarantinedOwnership {
                mechanism: mechanism.identity,
                quarantined: snapshot.quarantined_allocations,
            });
        }
        if snapshot.lifecycle == MechanismLifecycle::Active
            || snapshot.live_allocations != 0
            || snapshot.queued_releases != 0
            || snapshot.active_operations != 0
        {
            return Err(BindingError::InactiveMechanism {
                mechanism: mechanism.identity,
                lifecycle: snapshot.lifecycle,
                operation: "removing a mechanism before it is quiescent",
            });
        }
        let mut state = self.lock_state("removing a mechanism")?;
        if state.selected.get(&mechanism.device) == Some(&mechanism.identity) {
            state.selected.remove(&mechanism.device);
        }
        state.mechanisms.remove(&mechanism.identity);
        Ok(())
    }

    pub fn snapshot(
        &self,
        mechanism: RegisteredMechanism,
    ) -> Result<MechanismSnapshot, BindingError> {
        self.ensure_local(mechanism.identity.0, "mechanism")?;
        let entry = self
            .lock_state("looking up a mechanism snapshot")?
            .mechanisms
            .get(&mechanism.identity)
            .cloned()
            .ok_or(BindingError::UnregisteredMechanism(mechanism.identity))?;
        entry.snapshot()
    }

    fn ensure_local(
        &self,
        identity: OpaqueIdentity,
        kind: &'static str,
    ) -> Result<(), BindingError> {
        if identity.registry != self.inner.identities.registry {
            return Err(BindingError::ForeignRegistry { kind });
        }
        Ok(())
    }
}

/// Registry handle for one provider context.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RegisteredProviderContext {
    identity: ProviderContextIdentity,
    device: DeviceKey,
}

impl RegisteredProviderContext {
    pub const fn identity(self) -> ProviderContextIdentity {
        self.identity
    }

    pub const fn device(self) -> DeviceKey {
        self.device
    }
}

/// Registry handle for one accounting authority.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RegisteredAuthority {
    identity: AuthorityIdentity,
    device: DeviceKey,
}

impl RegisteredAuthority {
    pub const fn identity(self) -> AuthorityIdentity {
        self.identity
    }

    pub const fn device(self) -> DeviceKey {
        self.device
    }
}

/// Registry handle for one allocator mechanism or trusted coherent bundle.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RegisteredMechanism {
    identity: MechanismIdentity,
    device: DeviceKey,
    coherence: MechanismCoherence,
}

impl RegisteredMechanism {
    pub const fn identity(self) -> MechanismIdentity {
        self.identity
    }

    pub const fn device(self) -> DeviceKey {
        self.device
    }

    pub const fn coherence(self) -> MechanismCoherence {
        self.coherence
    }
}

/// One binding to a registered device/mechanism/context/authority tuple.
///
/// Clones preserve the same binding identity and pin. A new registry lookup
/// receives a new binding id/generation even when it selects the same mechanism.
#[derive(Clone, Debug)]
pub struct MemoryBinding {
    identity: BindingIdentity,
    identities: Arc<IdentitySource>,
    mechanism: Arc<MechanismEntry>,
}

impl MemoryBinding {
    pub const fn identity(&self) -> BindingIdentity {
        self.identity
    }

    pub fn allocate(&self, bytes: usize, align: usize) -> Result<BoundAllocation, BindingError> {
        self.allocate_with(
            "allocating bound memory",
            |allocator| allocator.allocate(bytes, align),
            bytes,
            align,
        )
    }

    fn allocate_with(
        &self,
        operation: &'static str,
        allocate: impl FnOnce(&dyn DeviceAllocator) -> Result<NonNull<u8>, MemoryError>,
        bytes: usize,
        align: usize,
    ) -> Result<BoundAllocation, BindingError> {
        let active = self.mechanism.begin_active(operation)?;
        let ptr = allocate(self.mechanism.allocator())?;
        let generation = match self.identities.allocation_generation() {
            Ok(generation) => generation,
            Err(error) => {
                // SAFETY: this is the exact allocation returned immediately
                // above; identity issuance failed before it escaped.
                unsafe { self.mechanism.allocator().deallocate(ptr, bytes, align) };
                return Err(error);
            }
        };
        let identity = AllocationIdentity {
            binding: self.identity,
            generation,
        };
        let allocation = BoundAllocation {
            binding: self.clone(),
            identity,
            ptr,
            bytes,
            align,
        };
        if let Err(error) = self.mechanism.record_allocation(allocation.record()) {
            // SAFETY: identity recording failed before the allocation escaped.
            unsafe { self.mechanism.allocator().deallocate(ptr, bytes, align) };
            return Err(error);
        }
        drop(active);
        Ok(allocation)
    }

    /// Allocate and take **owning** responsibility for the result.
    ///
    /// The returned [`OwningAllocation`] has exactly one consuming release and a
    /// `Drop` that quarantines rather than frees, so a forgotten allocation is
    /// accounted for instead of silently double-freed or leaked without trace.
    pub fn allocate_owning(
        &self,
        bytes: usize,
        align: usize,
    ) -> Result<OwningAllocation, BindingError> {
        self.allocate(bytes, align).map(OwningAllocation::new)
    }

    /// Issue an allocation generation for memory this binding's **own**
    /// mechanism produced through a specialized entry point this crate cannot
    /// express, and take owning responsibility for it.
    ///
    /// This is the narrow adoption seam for a provider-specific allocation call
    /// (the CUDA VMM arena's mapped-capacity allocation is the motivating case:
    /// it needs a governor capacity token that is deliberately not part of the
    /// portable [`VirtualBacking`](crate::VirtualBacking) capability). Adoption
    /// registers the address under a fresh generation *before* it escapes, so
    /// every later view, commit, and release is generation-validated exactly
    /// like a binding-issued allocation. Nothing else about the lifecycle is
    /// relaxed.
    ///
    /// # Safety
    ///
    /// The caller must guarantee all of:
    ///
    /// * `ptr` is one live allocation of exactly `bytes` at `align` produced by
    ///   **this binding's selected mechanism** (the same coherent allocator that
    ///   would serve [`allocate`](Self::allocate)), not by another allocator or
    ///   another device.
    /// * The allocation is not already recorded by this or any other binding,
    ///   and no other owner exists for it. Adoption is the single point at which
    ///   ownership enters the binding, and the returned owner is its sole owner.
    /// * The allocation may be released by that mechanism's canonical
    ///   [`DeviceAllocator::release`] with exactly this `ptr`/`bytes`/`align`.
    pub unsafe fn adopt_allocation(
        &self,
        ptr: NonNull<u8>,
        bytes: usize,
        align: usize,
    ) -> Result<OwningAllocation, BindingError> {
        let active = self
            .mechanism
            .begin_active("adopting a device allocation")?;
        let generation = self.identities.allocation_generation()?;
        let identity = AllocationIdentity {
            binding: self.identity,
            generation,
        };
        let allocation = BoundAllocation {
            binding: self.clone(),
            identity,
            ptr,
            bytes,
            align,
        };
        self.mechanism.record_allocation(allocation.record())?;
        drop(active);
        Ok(OwningAllocation::new(allocation))
    }

    /// Detach final ownership of `allocation` without calling the allocator.
    ///
    /// This is the single preparation point for every release path. It matches
    /// the binding identity **and** the allocation generation and removes the
    /// live record exactly once under the per-mechanism lock, then returns an
    /// owned request that pins the allocator, authority, and provider context.
    ///
    /// # Lock order
    ///
    /// Only the mechanism lock is taken, and it is released before this method
    /// returns. Queue and allocator calls therefore always happen with no
    /// registry or mechanism lock held.
    pub fn prepare_release(
        &self,
        allocation: BoundAllocation,
    ) -> Result<PreparedAllocationRelease, ExplicitReleaseError> {
        let operation = match self.mechanism.begin_release(self.identity, &allocation) {
            Ok(operation) => operation,
            Err(error) => return Err(ExplicitReleaseError::unchanged(error, allocation)),
        };
        Ok(PreparedAllocationRelease::new(
            allocation.binding.clone(),
            allocation.identity,
            allocation.ptr,
            allocation.bytes,
            allocation.align,
            PreparedReleasePins {
                allocator: self.mechanism.allocator_arc(),
                authority: self.identity.authority,
                context: self.identity.provider_context,
                operation,
            },
        ))
    }

    /// Explicitly release through this binding's pinned original allocator.
    ///
    /// This is the documented **migration adapter** for Phase-3 non-RAII
    /// [`BoundAllocation`] metadata. It is routed through the same structured
    /// prepared-release path as owning handles, so the generation is validated
    /// and the live record is retired before the allocator is invoked. It
    /// completes synchronously and never enqueues.
    ///
    /// On pre-mutation failure (identity mismatch, stale generation, device
    /// loss) the allocation is returned inside [`ExplicitReleaseError`] exactly
    /// as before, so metadata and lifetime pins are not silently discarded.
    ///
    /// # Limitations
    ///
    /// The adapter cannot report the structured success outcome, because its
    /// signature returns `()`. Callers that need the release accounting or that
    /// need to defer release past a stream fence should migrate to
    /// [`OwningAllocation`] or [`MemoryBinding::prepare_release`].
    ///
    /// When the allocator quarantines residual ownership the adapter reports
    /// [`BindingError::ReleaseQuarantined`] and hands back the now-dead
    /// metadata together with the structured outcome
    /// ([`ExplicitReleaseError::outcome`]). That metadata can never be released
    /// again: its record was already retired, so every later operation on it
    /// fails with [`BindingError::StaleAllocation`].
    pub fn release(&self, allocation: BoundAllocation) -> Result<(), ExplicitReleaseError> {
        let prepared = self.prepare_release(allocation)?;
        let stale = self.stale_metadata(&prepared);
        match prepared.execute() {
            AllocationReleaseOutcome::Complete { .. } => Ok(()),
            outcome @ (AllocationReleaseOutcome::Quarantined { .. }
            | AllocationReleaseOutcome::Failed { .. }) => {
                let residual = outcome.residual();
                Err(ExplicitReleaseError::quarantined(
                    BindingError::ReleaseQuarantined {
                        identity: stale.identity,
                        state: outcome.state(),
                        reason: residual.map_or(QuarantineReason::AllocatorRefused, |residual| {
                            residual.reason
                        }),
                        retained_bytes: residual.map_or(0, |residual| residual.retained_bytes),
                    },
                    stale,
                    outcome,
                ))
            }
        }
    }

    /// Rebuild inert metadata for an allocation whose live record is already
    /// retired.
    ///
    /// Only the legacy adapter uses this, and only to keep its historical
    /// "the error hands the allocation back" shape. The rebuilt value can never
    /// be released again because its generation is no longer recorded.
    fn stale_metadata(&self, prepared: &PreparedAllocationRelease) -> BoundAllocation {
        BoundAllocation {
            binding: self.clone(),
            identity: prepared.identity(),
            ptr: prepared.as_ptr(),
            bytes: prepared.len(),
            align: prepared.alignment(),
        }
    }

    /// Ownership this binding's mechanism retained instead of releasing.
    pub fn quarantined(&self) -> Result<Vec<QuarantinedAllocation>, BindingError> {
        self.mechanism.quarantined()
    }

    /// Lifecycle and ownership counts for this binding's mechanism.
    pub fn mechanism_snapshot(&self) -> Result<MechanismSnapshot, BindingError> {
        self.mechanism.snapshot()
    }

    pub(crate) fn mechanism(&self) -> &Arc<MechanismEntry> {
        &self.mechanism
    }

    pub fn virtual_backing(&self) -> Result<Option<BoundVirtualBacking>, BindingError> {
        let operation = self
            .mechanism
            .begin_active("discovering virtual backing capability")?;
        let present = self.mechanism.allocator().as_virtual_backing().is_some();
        drop(operation);
        Ok(present.then(|| BoundVirtualBacking {
            binding: self.clone(),
        }))
    }

    pub fn shared_mapping(&self) -> Result<Option<BoundSharedMapping>, BindingError> {
        let operation = self
            .mechanism
            .begin_active("discovering shared mapping capability")?;
        let present = self.mechanism.allocator().as_shared_mapping().is_some();
        drop(operation);
        Ok(present.then(|| BoundSharedMapping {
            binding: self.clone(),
        }))
    }

    /// Validate a view, then invoke `operation` without a registry or mechanism
    /// lock held. This is the binding boundary for kernel/copy callbacks.
    pub fn with_view<R>(
        &self,
        view: &BoundMemoryView,
        operation: impl FnOnce(ValidatedMemoryView) -> R,
    ) -> Result<R, BindingError> {
        let active = self
            .mechanism
            .begin_active("validating a view for device use")?;
        self.mechanism
            .validate_view(self.identity, view, "validating a view for device use")?;
        let validated = ValidatedMemoryView {
            ptr: view.ptr,
            bytes: view.bytes,
        };
        let result = operation(validated);
        drop(active);
        Ok(result)
    }
}

/// Non-RAII allocation metadata issued by one [`MemoryBinding`].
///
/// Dropping this value does not free memory. Call [`MemoryBinding::release`]
/// explicitly through the same binding.
#[derive(Debug)]
pub struct BoundAllocation {
    binding: MemoryBinding,
    identity: AllocationIdentity,
    ptr: NonNull<u8>,
    bytes: usize,
    align: usize,
}

// SAFETY: this type is allocation metadata over a provider-defined device
// address. It exposes no safe dereference and pins a Send + Sync allocator and
// context. Moving or sharing the metadata does not access the pointed-to bytes.
unsafe impl Send for BoundAllocation {}
// SAFETY: shared access exposes only copied metadata and a non-dereferenced
// pointer; explicit release consumes the allocation.
unsafe impl Sync for BoundAllocation {}

impl BoundAllocation {
    pub const fn identity(&self) -> AllocationIdentity {
        self.identity
    }

    pub const fn binding(&self) -> &MemoryBinding {
        &self.binding
    }

    pub const fn as_ptr(&self) -> NonNull<u8> {
        self.ptr
    }

    pub const fn len(&self) -> usize {
        self.bytes
    }

    pub const fn is_empty(&self) -> bool {
        self.bytes == 0
    }

    pub const fn alignment(&self) -> usize {
        self.align
    }

    pub fn view(&self, offset: usize, bytes: usize) -> Result<BoundMemoryView, BindingError> {
        let end = offset
            .checked_add(bytes)
            .ok_or(BindingError::ViewOutOfBounds {
                offset,
                end: usize::MAX,
                allocation_bytes: self.bytes,
            })?;
        if end > self.bytes {
            return Err(BindingError::ViewOutOfBounds {
                offset,
                end,
                allocation_bytes: self.bytes,
            });
        }
        self.binding.mechanism.validate_allocation(
            self.binding.identity,
            self,
            "creating a bound view",
        )?;
        Ok(BoundMemoryView {
            binding: self.binding.clone(),
            identity: self.identity,
            allocation_ptr: self.ptr,
            ptr: NonNull::new(self.ptr.as_ptr().wrapping_add(offset))
                .expect("offset within a live allocation cannot produce null"),
            allocation_bytes: self.bytes,
            bytes,
            align: self.align,
        })
    }

    fn record(&self) -> AllocationRecord {
        AllocationRecord {
            identity: self.identity,
            ptr: self.ptr.as_ptr() as usize,
            bytes: self.bytes,
            align: self.align,
        }
    }

    fn matches_record(&self, record: &AllocationRecord) -> bool {
        record.identity == self.identity
            && record.ptr == self.ptr.as_ptr() as usize
            && record.bytes == self.bytes
            && record.align == self.align
    }
}

/// Cloneable metadata for a sub-range of one bound allocation.
#[derive(Clone, Debug)]
pub struct BoundMemoryView {
    binding: MemoryBinding,
    identity: AllocationIdentity,
    allocation_ptr: NonNull<u8>,
    ptr: NonNull<u8>,
    allocation_bytes: usize,
    bytes: usize,
    align: usize,
}

// SAFETY: like BoundAllocation, a view is inert metadata over an opaque address;
// validated access still requires the caller to uphold device synchronization.
unsafe impl Send for BoundMemoryView {}
// SAFETY: shared references cannot mutate memory through this type.
unsafe impl Sync for BoundMemoryView {}

impl BoundMemoryView {
    pub const fn binding(&self) -> &MemoryBinding {
        &self.binding
    }

    pub const fn allocation_identity(&self) -> AllocationIdentity {
        self.identity
    }

    pub const fn len(&self) -> usize {
        self.bytes
    }

    pub const fn is_empty(&self) -> bool {
        self.bytes == 0
    }
}

/// Pointer/extent exposed only after binding validation.
#[derive(Clone, Copy, Debug)]
pub struct ValidatedMemoryView {
    ptr: NonNull<u8>,
    bytes: usize,
}

// SAFETY: validation produces a copied opaque device address, not a Rust
// reference. Dereferencing remains unsafe and provider synchronization remains
// the caller's responsibility.
unsafe impl Send for ValidatedMemoryView {}
// SAFETY: the value has no safe memory access or interior mutation.
unsafe impl Sync for ValidatedMemoryView {}

impl ValidatedMemoryView {
    pub const fn as_ptr(self) -> NonNull<u8> {
        self.ptr
    }

    pub const fn len(self) -> usize {
        self.bytes
    }

    pub const fn is_empty(self) -> bool {
        self.bytes == 0
    }
}

/// Owning responsibility for exactly one [`BoundAllocation`].
///
/// This is the Phase-4 owner. It is deliberately **not** `Clone` and **not**
/// `Copy`: ownership of a physical allocation cannot be duplicated. Aliases are
/// expressed as [`OwnedView`]s, which can never release anything.
///
/// # Release paths
///
/// * [`release_now`](Self::release_now) — synchronous, no queue. This is the
///   CPU/eager path: one mechanism-lock preparation plus one allocator call.
/// * [`release_deferred`](Self::release_deferred) — hand final ownership to a
///   provider/context-owned [`DeferredReleaseQueue`], for GPU allocations whose
///   release must wait for a stream fence.
/// * [`prepare_release`](Self::prepare_release) — take the owned request and
///   route it manually.
///
/// All three consume the owner, so there is exactly one final release.
///
/// # Drop
///
/// Dropping an owner without releasing it **quarantines** the allocation: the
/// live record is retired under the mechanism lock and residual ownership is
/// recorded. `Drop` never calls the allocator, never enqueues, and never waits.
/// Freeing from `Drop` is what makes stale-pointer double frees possible, so
/// this type refuses to do it.
///
/// # Outstanding views block physical release
///
/// While any [`OwnedView`] (or clone of one) is alive, release is refused with
/// [`BindingError::OutstandingViews`] and the owner is handed back untouched.
#[derive(Debug)]
pub struct OwningAllocation {
    /// `None` only between a consuming method taking the allocation and the
    /// shell being dropped, which is what keeps `Drop` from quarantining an
    /// allocation that was already handed on.
    allocation: Option<BoundAllocation>,
    views: Arc<AtomicUsize>,
}

impl OwningAllocation {
    /// Take ownership of Phase-3 metadata.
    ///
    /// This is the migration entry point from [`BoundAllocation`] to owning
    /// semantics; the generation is still validated at release time.
    pub fn new(allocation: BoundAllocation) -> Self {
        Self {
            allocation: Some(allocation),
            views: Arc::new(AtomicUsize::new(0)),
        }
    }

    fn allocation(&self) -> &BoundAllocation {
        self.allocation
            .as_ref()
            .expect("an owning allocation holds its allocation until it is consumed")
    }

    pub fn identity(&self) -> AllocationIdentity {
        self.allocation().identity
    }

    pub fn binding(&self) -> &MemoryBinding {
        &self.allocation().binding
    }

    /// Borrow the allocation metadata for a bound capability call.
    ///
    /// [`BoundAllocation`] is not `Clone`, so a shared borrow can be handed to
    /// [`BoundVirtualBacking`] commit/decommit/query operations — every one of
    /// which re-validates the binding identity and the allocation generation —
    /// without giving up owning responsibility. There is no path from this
    /// borrow to a release: releasing still needs the owner by value.
    pub fn bound(&self) -> &BoundAllocation {
        self.allocation()
    }

    /// The owned address. Never dereferenced by this crate.
    pub fn as_ptr(&self) -> NonNull<u8> {
        self.allocation().ptr
    }

    pub fn len(&self) -> usize {
        self.allocation().bytes
    }

    pub fn is_empty(&self) -> bool {
        self.allocation().bytes == 0
    }

    pub fn alignment(&self) -> usize {
        self.allocation().align
    }

    /// Always [`AllocationReleaseState::Live`]; any other state means the owner
    /// was already consumed.
    pub const fn state(&self) -> AllocationReleaseState {
        AllocationReleaseState::Live
    }

    /// Borrow a sub-range. The returned view never releases anything and keeps
    /// this allocation from being physically released while it is alive.
    pub fn view(&self, offset: usize, bytes: usize) -> Result<OwnedView, BindingError> {
        let view = self.allocation().view(offset, bytes)?;
        self.views.fetch_add(1, Ordering::AcqRel);
        Ok(OwnedView {
            view,
            outstanding: Arc::clone(&self.views),
        })
    }

    /// How many borrowed views and aliases are still alive.
    pub fn outstanding_views(&self) -> usize {
        self.views.load(Ordering::Acquire)
    }

    /// Give up owning semantics and return to Phase-3 metadata.
    ///
    /// Documented migration adapter: the result must be released explicitly
    /// through [`MemoryBinding::release`], and dropping it releases nothing.
    pub fn into_bound(self) -> Result<BoundAllocation, OwningReleaseError> {
        self.take("disowning an allocation with outstanding views")
    }

    /// Detach final ownership without calling the allocator.
    pub fn prepare_release(self) -> Result<PreparedAllocationRelease, OwningReleaseError> {
        let views = Arc::clone(&self.views);
        let allocation = self.take("preparing release with outstanding views")?;
        let binding = allocation.binding.clone();
        binding.prepare_release(allocation).map_err(|error| {
            let (error, allocation) = error.into_parts();
            OwningReleaseError {
                error,
                allocation: Box::new(Self {
                    allocation: Some(allocation),
                    views,
                }),
            }
        })
    }

    /// Release immediately and synchronously through the pinned allocator.
    ///
    /// No queue is involved and no wait happens, so this is the low-overhead
    /// path for CPU/eager mechanisms.
    pub fn release_now(self) -> Result<AllocationReleaseOutcome, OwningReleaseError> {
        Ok(self.prepare_release()?.execute())
    }

    /// Hand final ownership to a provider/context-owned queue.
    ///
    /// The queue is called after every registry and mechanism lock is dropped.
    /// If the queue refuses, the exact prepared request is quarantined rather
    /// than freed or lost, and the rejection is reported.
    pub fn release_deferred(
        self,
        queue: &dyn DeferredReleaseQueue,
    ) -> Result<DeferredReleaseDisposition, OwningReleaseError> {
        let prepared = self.prepare_release()?;
        let identity = prepared.identity();
        match queue.enqueue(prepared) {
            Ok(()) => Ok(DeferredReleaseDisposition::Queued { identity }),
            Err(error) => {
                let rejection = error.rejection();
                Ok(DeferredReleaseDisposition::Quarantined {
                    identity,
                    rejection,
                    outcome: error.quarantine(),
                })
            }
        }
    }

    fn take(mut self, operation: &'static str) -> Result<BoundAllocation, OwningReleaseError> {
        let _ = operation;
        let views = Arc::clone(&self.views);
        let outstanding = views.load(Ordering::Acquire);
        let allocation = self
            .allocation
            .take()
            .expect("an owning allocation holds its allocation until it is consumed");
        if outstanding != 0 {
            return Err(OwningReleaseError {
                error: BindingError::OutstandingViews {
                    identity: allocation.identity,
                    views: outstanding,
                },
                allocation: Box::new(Self {
                    allocation: Some(allocation),
                    views,
                }),
            });
        }
        Ok(allocation)
    }
}

impl Drop for OwningAllocation {
    /// Quarantine an owner that was dropped without an explicit release.
    ///
    /// This never frees. It retires the live record under the mechanism lock and
    /// records residual ownership, so the bytes stay visible to accounting and
    /// block unsafe mechanism removal. When the record can no longer be
    /// prepared — device loss, termination, or an already stale generation — the
    /// metadata is simply dropped: in the device-loss case the live record is
    /// still recorded at the mechanism and is discharged by confirmed context
    /// termination.
    fn drop(&mut self) {
        let Some(allocation) = self.allocation.take() else {
            return;
        };
        let binding = allocation.binding.clone();
        if let Ok(prepared) = binding.prepare_release(allocation) {
            let _ = prepared.quarantine(QuarantineReason::OwnerDropped);
        }
    }
}

/// A borrowed, cloneable alias of one [`OwningAllocation`].
///
/// A view can never release anything: it has no release method and its `Drop`
/// only decrements the outstanding-view count. Clones are aliases, and every
/// alias independently keeps physical release blocked.
#[derive(Debug)]
pub struct OwnedView {
    view: BoundMemoryView,
    outstanding: Arc<AtomicUsize>,
}

impl Clone for OwnedView {
    fn clone(&self) -> Self {
        self.outstanding.fetch_add(1, Ordering::AcqRel);
        Self {
            view: self.view.clone(),
            outstanding: Arc::clone(&self.outstanding),
        }
    }
}

impl Drop for OwnedView {
    fn drop(&mut self) {
        self.outstanding.fetch_sub(1, Ordering::AcqRel);
    }
}

impl OwnedView {
    pub const fn view(&self) -> &BoundMemoryView {
        &self.view
    }

    pub const fn binding(&self) -> &MemoryBinding {
        self.view.binding()
    }

    pub const fn allocation_identity(&self) -> AllocationIdentity {
        self.view.allocation_identity()
    }

    pub const fn len(&self) -> usize {
        self.view.len()
    }

    pub const fn is_empty(&self) -> bool {
        self.view.is_empty()
    }
}

/// Owning-release failure that hands the exact owner back.
///
/// Every failure carried by this type is *pre-mutation*: nothing was released,
/// nothing was queued, and the returned [`OwningAllocation`] is still live and
/// still owns its allocation.
#[derive(Debug)]
pub struct OwningReleaseError {
    error: BindingError,
    /// Boxed so the common `Ok` path does not pay for an owner-sized `Err`.
    allocation: Box<OwningAllocation>,
}

impl OwningReleaseError {
    pub const fn error(&self) -> &BindingError {
        &self.error
    }

    pub const fn allocation(&self) -> &OwningAllocation {
        &self.allocation
    }

    /// Nothing was mutated, so the owner is still [`AllocationReleaseState::Live`].
    pub const fn state(&self) -> AllocationReleaseState {
        AllocationReleaseState::Live
    }

    pub fn into_parts(self) -> (BindingError, OwningAllocation) {
        (self.error, *self.allocation)
    }
}

impl std::fmt::Display for OwningReleaseError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.error, formatter)
    }
}

impl std::error::Error for OwningReleaseError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.error)
    }
}

/// Explicit release failure that preserves allocation metadata and pins.
///
/// Two dispositions share this type:
///
/// * **Unchanged** — the failure happened before any device mutation, and
///   [`into_parts`](Self::into_parts) returns the exact live allocation.
/// * **Quarantined** — the release was prepared and the allocator did not
///   complete it. [`outcome`](Self::outcome) carries the structured accounting
///   and residual facts, and the returned metadata is provably dead because its
///   generation record was already retired.
#[derive(Debug)]
pub struct ExplicitReleaseError {
    error: BindingError,
    allocation: Box<BoundAllocation>,
    /// Boxed so a successful release does not pay for an outcome-sized `Err`.
    outcome: Option<Box<AllocationReleaseOutcome>>,
}

impl ExplicitReleaseError {
    fn unchanged(error: BindingError, allocation: BoundAllocation) -> Self {
        Self {
            error,
            allocation: Box::new(allocation),
            outcome: None,
        }
    }

    fn quarantined(
        error: BindingError,
        allocation: BoundAllocation,
        outcome: AllocationReleaseOutcome,
    ) -> Self {
        Self {
            error,
            allocation: Box::new(allocation),
            outcome: Some(Box::new(outcome)),
        }
    }

    pub const fn error(&self) -> &BindingError {
        &self.error
    }

    /// The structured outcome, when the allocator was actually invoked.
    ///
    /// `None` means nothing was mutated and the allocation is still live.
    pub fn outcome(&self) -> Option<&AllocationReleaseOutcome> {
        self.outcome.as_deref()
    }

    /// Whether the allocation's ownership was retained after preparation.
    pub const fn is_quarantined(&self) -> bool {
        self.outcome.is_some()
    }

    /// The lifecycle state the allocation was left in.
    pub fn state(&self) -> AllocationReleaseState {
        self.outcome.as_deref().map_or(
            AllocationReleaseState::Live,
            AllocationReleaseOutcome::state,
        )
    }

    pub fn into_parts(self) -> (BindingError, BoundAllocation) {
        (self.error, *self.allocation)
    }
}

impl std::fmt::Display for ExplicitReleaseError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.error, formatter)
    }
}

impl std::error::Error for ExplicitReleaseError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.error)
    }
}

/// Virtual-backing access pinned to one binding.
#[derive(Clone, Debug)]
pub struct BoundVirtualBacking {
    binding: MemoryBinding,
}

impl BoundVirtualBacking {
    pub const fn binding_identity(&self) -> BindingIdentity {
        self.binding.identity
    }

    pub fn allocate_committed(
        &self,
        bytes: usize,
        align: usize,
        committed_ranges: &[std::ops::Range<usize>],
    ) -> Result<BoundAllocation, BindingError> {
        self.binding.allocate_with(
            "allocating through bound virtual backing",
            |allocator| {
                allocator
                    .as_virtual_backing()
                    .expect("capability presence is stable for a registered allocator")
                    .allocate_committed(bytes, align, committed_ranges)
            },
            bytes,
            align,
        )
    }

    pub fn commit_allocation_range(
        &self,
        allocation: &BoundAllocation,
        offset: usize,
        bytes: usize,
    ) -> Result<(), BindingError> {
        let active = self
            .binding
            .mechanism
            .begin_active("committing through bound virtual backing")?;
        self.binding.mechanism.validate_allocation(
            self.binding.identity,
            allocation,
            "validating allocation for virtual commit",
        )?;
        let capability = self
            .binding
            .mechanism
            .allocator()
            .as_virtual_backing()
            .expect("capability presence is stable for a registered allocator");
        capability.commit_allocation_range(
            allocation.ptr,
            allocation.bytes,
            allocation.align,
            offset,
            bytes,
        )?;
        drop(active);
        Ok(())
    }

    pub fn commit_allocation_ranges(
        &self,
        ranges: &[(&BoundAllocation, usize, usize)],
    ) -> Result<(), BindingError> {
        let active = self
            .binding
            .mechanism
            .begin_active("committing ranges through bound virtual backing")?;
        let mut raw = Vec::with_capacity(ranges.len());
        for &(allocation, offset, bytes) in ranges {
            self.binding.mechanism.validate_allocation(
                self.binding.identity,
                allocation,
                "validating allocation ranges for virtual commit",
            )?;
            raw.push(AllocationCommitRange {
                ptr: allocation.ptr,
                allocation_bytes: allocation.bytes,
                align: allocation.align,
                offset,
                bytes,
            });
        }
        self.binding
            .mechanism
            .allocator()
            .as_virtual_backing()
            .expect("capability presence is stable for a registered allocator")
            .commit_allocation_ranges(&raw)?;
        drop(active);
        Ok(())
    }

    pub fn mapped_bytes_for_allocation(
        &self,
        bytes: usize,
        align: usize,
    ) -> Result<u64, BindingError> {
        let active = self
            .binding
            .mechanism
            .begin_active("querying bound virtual backing")?;
        let mapped = self
            .binding
            .mechanism
            .allocator()
            .as_virtual_backing()
            .expect("capability presence is stable for a registered allocator")
            .mapped_bytes_for_allocation(bytes, align)?;
        drop(active);
        Ok(mapped)
    }

    pub fn decommit_allocation_range(
        &self,
        allocation: &BoundAllocation,
        offset: usize,
        bytes: usize,
    ) -> Result<u64, BindingError> {
        let active = self
            .binding
            .mechanism
            .begin_active("decommitting through bound virtual backing")?;
        self.binding.mechanism.validate_allocation(
            self.binding.identity,
            allocation,
            "validating allocation for virtual decommit",
        )?;
        let unmapped = self
            .binding
            .mechanism
            .allocator()
            .as_virtual_backing()
            .expect("capability presence is stable for a registered allocator")
            .decommit_allocation_range(
                allocation.ptr,
                allocation.bytes,
                allocation.align,
                offset,
                bytes,
            )?;
        drop(active);
        Ok(unmapped)
    }

    pub fn allocation_committed_bytes(
        &self,
        allocation: &BoundAllocation,
    ) -> Result<usize, BindingError> {
        let active = self
            .binding
            .mechanism
            .begin_active("querying bound allocation commitment")?;
        self.binding.mechanism.validate_allocation(
            self.binding.identity,
            allocation,
            "validating allocation commitment query",
        )?;
        let committed = self
            .binding
            .mechanism
            .allocator()
            .as_virtual_backing()
            .expect("capability presence is stable for a registered allocator")
            .allocation_committed_bytes(allocation.ptr, allocation.bytes, allocation.align);
        drop(active);
        Ok(committed)
    }
}

/// Shared-mapping access pinned to one binding.
#[derive(Clone, Debug)]
pub struct BoundSharedMapping {
    binding: MemoryBinding,
}

impl BoundSharedMapping {
    pub const fn binding_identity(&self) -> BindingIdentity {
        self.binding.identity
    }

    pub fn create_shared_prefix(&self, bytes: usize) -> Result<BoundSharedPrefix, BindingError> {
        let active = self
            .binding
            .mechanism
            .begin_active("creating a bound shared prefix")?;
        let prefix = self
            .binding
            .mechanism
            .allocator()
            .as_shared_mapping()
            .expect("capability presence is stable for a registered allocator")
            .create_shared_prefix(bytes)?;
        drop(active);
        Ok(BoundSharedPrefix {
            prefix,
            binding: self.binding.clone(),
        })
    }

    pub fn incremental_owned_bytes_for_shared_prefix(
        &self,
        prefix: &BoundSharedPrefix,
    ) -> Result<u64, BindingError> {
        let active = self
            .binding
            .mechanism
            .begin_active("querying a bound shared prefix")?;
        validate_binding_identity(self.binding.identity, prefix.binding.identity)?;
        let bytes = self
            .binding
            .mechanism
            .allocator()
            .as_shared_mapping()
            .expect("capability presence is stable for a registered allocator")
            .incremental_owned_bytes_for_shared_prefix(prefix.prefix.as_ref())?;
        drop(active);
        Ok(bytes)
    }

    pub fn commit_shared_prefix(
        &self,
        prefix: &BoundSharedPrefix,
        allocation: &BoundAllocation,
        byte_offset: usize,
    ) -> Result<SharedPrefixCommitInfo, BindingError> {
        let active = self
            .binding
            .mechanism
            .begin_active("committing a bound shared prefix")?;
        validate_binding_identity(self.binding.identity, prefix.binding.identity)?;
        self.binding.mechanism.validate_allocation(
            self.binding.identity,
            allocation,
            "validating allocation for shared prefix commit",
        )?;
        let info = self
            .binding
            .mechanism
            .allocator()
            .as_shared_mapping()
            .expect("capability presence is stable for a registered allocator")
            .commit_shared_prefix(
                prefix.prefix.as_ref(),
                allocation.ptr,
                allocation.bytes,
                byte_offset,
            )?;
        drop(active);
        Ok(info)
    }
}

/// Shared physical-prefix handle pinned to one binding.
///
/// Physical teardown remains driven by dropping this handle, not by
/// [`BindingRegistry`] lifecycle transitions. The prefix is dropped before its
/// binding pin so its provider context is still alive during teardown. Phase 4
/// owns stream ordering and partial-failure-safe physical release.
#[derive(Debug)]
pub struct BoundSharedPrefix {
    prefix: Box<dyn SharedDevicePrefix>,
    binding: MemoryBinding,
}

impl BoundSharedPrefix {
    pub const fn binding_identity(&self) -> BindingIdentity {
        self.binding.identity
    }

    pub fn device_ptr(&self) -> u64 {
        self.prefix.device_ptr()
    }

    pub fn committed_physical_bytes(&self) -> u64 {
        self.prefix.committed_physical_bytes()
    }

    pub fn mapped_bytes(&self) -> usize {
        self.prefix.mapped_bytes()
    }

    pub fn requested_bytes(&self) -> usize {
        self.prefix.requested_bytes()
    }
}

fn validate_binding_identity(
    expected: BindingIdentity,
    actual: BindingIdentity,
) -> Result<(), BindingError> {
    if expected != actual {
        return Err(BindingError::BindingMismatch {
            expected: expected.id,
            actual: actual.id,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Barrier};
    use std::thread;

    use crate::{BindingResource, HostAllocator};

    use super::*;

    /// A real [`BindingRegistry`] with one registered context/authority pair.
    ///
    /// Every selection test below drives the production registry; only the
    /// pause points are test-owned, so the code under test is unchanged.
    struct SelectionFixture {
        registry: BindingRegistry,
        context: RegisteredProviderContext,
        authority: RegisteredAuthority,
    }

    impl SelectionFixture {
        fn new() -> Self {
            let registry = BindingRegistry::new().expect("registry");
            let context = registry
                .register_provider_context(
                    DeviceKey::HOST,
                    Arc::new(()) as Arc<dyn BindingResource>,
                )
                .expect("context registration");
            let authority = registry
                .register_authority(DeviceKey::HOST, Arc::new(()) as Arc<dyn BindingResource>)
                .expect("authority registration");
            Self {
                registry,
                context,
                authority,
            }
        }

        /// Register one more mechanism. The first registration for a device also
        /// becomes its initial selection.
        fn mechanism(&self) -> RegisteredMechanism {
            self.registry
                .register_allocator(
                    self.context,
                    self.authority,
                    Arc::new(HostAllocator) as Arc<dyn DeviceAllocator>,
                )
                .expect("mechanism registration")
        }

        fn gate(&self, mechanism: RegisteredMechanism, phase: HookPhase) -> Gate {
            self.gate_subject(HookSubject::Mechanism(mechanism.identity), phase)
        }

        fn gate_device(&self, phase: HookPhase) -> Gate {
            self.gate_subject(HookSubject::Device(DeviceKey::HOST), phase)
        }

        fn gate_subject(&self, subject: HookSubject, phase: HookPhase) -> Gate {
            let entered = Arc::new(Barrier::new(2));
            let resume = Arc::new(Barrier::new(2));
            self.registry.install_hook(RegistryHook {
                subject,
                phase,
                entered: Arc::clone(&entered),
                resume: Arc::clone(&resume),
            });
            Gate { entered, resume }
        }

        fn select_on_thread(
            &self,
            mechanism: RegisteredMechanism,
        ) -> thread::JoinHandle<Result<(), BindingError>> {
            let registry = self.registry.clone();
            thread::spawn(move || registry.select(mechanism))
        }

        fn selected_mechanism(&self) -> Result<MechanismIdentity, BindingError> {
            self.registry
                .bind(DeviceKey::HOST)
                .map(|binding| binding.identity().mechanism())
        }

        fn assert_nothing_selected(&self) {
            let error = self
                .selected_mechanism()
                .expect_err("withdrawal must leave no selection to heal from");
            assert!(
                matches!(error, BindingError::NoSelectedMechanism(device) if device == DeviceKey::HOST),
                "selection was left pointing at a dead or unregistered mechanism: {error:?}"
            );
        }
    }

    /// One paused select, released only when the test says so.
    struct Gate {
        entered: Arc<Barrier>,
        resume: Arc<Barrier>,
    }

    impl Gate {
        fn wait_entered(&self) {
            self.entered.wait();
        }

        fn resume(&self) {
            self.resume.wait();
        }
    }

    fn assert_select_failed(
        result: Result<(), BindingError>,
        expected: RegisteredMechanism,
        expected_lifecycle: MechanismLifecycle,
    ) {
        let error = result.expect_err("select must fail once its candidate is inactive");
        assert!(
            matches!(
                error,
                BindingError::InactiveMechanism {
                    mechanism,
                    lifecycle,
                    operation: "selecting a mechanism",
                } if mechanism == expected.identity && lifecycle == expected_lifecycle
            ),
            "unexpected select error: {error:?}"
        );
    }

    /// Park a candidate after validation, retire it, then let it publish. The
    /// candidate is guaranteed to fail its post-publish re-check.
    fn publish_a_doomed_candidate(
        fixture: &SelectionFixture,
        candidate: RegisteredMechanism,
    ) -> (thread::JoinHandle<Result<(), BindingError>>, Gate) {
        let validated = fixture.gate(candidate, HookPhase::SelectAfterValidation);
        let published = fixture.gate(candidate, HookPhase::SelectAfterPublish);
        let selecting = fixture.select_on_thread(candidate);

        validated.wait_entered();
        // The candidate is not selected yet, so retiring it does not clear the
        // selection; it only makes the pending publication stale.
        fixture
            .registry
            .retire(candidate)
            .expect("retire candidate");
        validated.resume();

        // Returns with the candidate published and its prior recorded.
        published.wait_entered();
        (selecting, published)
    }

    #[test]
    fn failed_select_restores_the_prior_healthy_selection() {
        let fixture = SelectionFixture::new();
        let prior = fixture.mechanism();
        let candidate = fixture.mechanism();

        let (selecting, published) = publish_a_doomed_candidate(&fixture, candidate);
        published.resume();

        assert_select_failed(
            selecting.join().expect("select thread"),
            candidate,
            MechanismLifecycle::Retired,
        );
        assert_eq!(
            fixture
                .selected_mechanism()
                .expect("healthy prior restored"),
            prior.identity
        );
    }

    #[test]
    fn failed_select_clears_a_retired_prior_instead_of_restoring_it() {
        let fixture = SelectionFixture::new();
        let prior = fixture.mechanism();
        let candidate = fixture.mechanism();

        let (selecting, published) = publish_a_doomed_candidate(&fixture, candidate);
        // The prior is no longer selected, so retiring it cannot clear the
        // selection itself; only the withdrawal can notice it went inactive.
        fixture.registry.retire(prior).expect("retire prior");
        published.resume();

        assert_select_failed(
            selecting.join().expect("select thread"),
            candidate,
            MechanismLifecycle::Retired,
        );
        fixture.assert_nothing_selected();
    }

    #[test]
    fn failed_select_clears_a_removed_prior_instead_of_restoring_it() {
        let fixture = SelectionFixture::new();
        let prior = fixture.mechanism();
        let candidate = fixture.mechanism();

        let (selecting, published) = publish_a_doomed_candidate(&fixture, candidate);
        fixture.registry.retire(prior).expect("retire prior");
        fixture.registry.remove(prior).expect("remove prior");
        published.resume();

        assert_select_failed(
            selecting.join().expect("select thread"),
            candidate,
            MechanismLifecycle::Retired,
        );
        fixture.assert_nothing_selected();
    }

    #[test]
    fn failed_select_does_not_overwrite_a_newer_selection() {
        let fixture = SelectionFixture::new();
        let _prior = fixture.mechanism();
        let candidate = fixture.mechanism();
        let newer = fixture.mechanism();

        let (selecting, published) = publish_a_doomed_candidate(&fixture, candidate);
        // A healthy selection lands while the losing candidate is parked.
        fixture.registry.select(newer).expect("newer selection");
        published.resume();

        assert_select_failed(
            selecting.join().expect("select thread"),
            candidate,
            MechanismLifecycle::Retired,
        );
        assert_eq!(
            fixture
                .selected_mechanism()
                .expect("newer selection stands"),
            newer.identity
        );
    }

    #[test]
    fn two_failed_selects_never_leave_a_dead_mechanism_selected() {
        let fixture = SelectionFixture::new();
        let prior = fixture.mechanism();
        let first = fixture.mechanism();
        let second = fixture.mechanism();

        let first_published = fixture.gate(first, HookPhase::SelectAfterPublish);
        let second_validated = fixture.gate(second, HookPhase::SelectAfterValidation);
        let second_published = fixture.gate(second, HookPhase::SelectAfterPublish);

        // The first candidate publishes over the healthy prior and parks.
        let selecting_first = fixture.select_on_thread(first);
        first_published.wait_entered();

        // The second candidate validates while the first still owns selection,
        // so retiring it now does not clear the selection.
        let selecting_second = fixture.select_on_thread(second);
        second_validated.wait_entered();
        fixture.registry.retire(second).expect("retire second");
        second_validated.resume();

        // The second candidate is now selected and recorded the first as its
        // prior. Retiring the first makes that recorded prior dead.
        second_published.wait_entered();
        fixture.registry.retire(first).expect("retire first");

        // The first candidate withdraws while the second owns selection, so it
        // must leave the newer selection alone.
        first_published.resume();
        assert_select_failed(
            selecting_first.join().expect("first select thread"),
            first,
            MechanismLifecycle::Retired,
        );

        // The second candidate withdraws last and must not resurrect the first.
        second_published.resume();
        assert_select_failed(
            selecting_second.join().expect("second select thread"),
            second,
            MechanismLifecycle::Retired,
        );

        fixture.assert_nothing_selected();
        // The untouched healthy prior is still selectable.
        fixture.registry.select(prior).expect("reselect prior");
        assert_eq!(
            fixture.selected_mechanism().expect("prior reselected"),
            prior.identity
        );
    }

    #[test]
    fn a_later_registration_heals_a_cleared_selection() {
        let fixture = SelectionFixture::new();
        let prior = fixture.mechanism();
        let candidate = fixture.mechanism();

        let (selecting, published) = publish_a_doomed_candidate(&fixture, candidate);
        fixture.registry.retire(prior).expect("retire prior");
        fixture.registry.remove(prior).expect("remove prior");
        published.resume();

        assert_select_failed(
            selecting.join().expect("select thread"),
            candidate,
            MechanismLifecycle::Retired,
        );
        fixture.assert_nothing_selected();

        // A cleared selection is the only state a later registration can heal;
        // a stale identity left in the slot would make this registration a
        // no-op and wedge the device permanently.
        let healed = fixture.mechanism();
        assert_eq!(
            fixture
                .selected_mechanism()
                .expect("registration self-heal"),
            healed.identity
        );
    }

    #[test]
    fn retire_racing_a_select_cannot_leave_the_retired_mechanism_selected() {
        let fixture = SelectionFixture::new();
        let prior = fixture.mechanism();
        let candidate = fixture.mechanism();

        let validated = fixture.gate(candidate, HookPhase::SelectAfterValidation);
        let retiring_gate = fixture.gate(candidate, HookPhase::RetireBetweenPhases);

        // The candidate validates `Active` and parks before publishing.
        let selecting = fixture.select_on_thread(candidate);
        validated.wait_entered();

        // Retirement finishes its lifecycle phase and parks before its selection
        // phase. Clearing the selection first would leave exactly this window
        // open, because the candidate would still observe `Active` afterwards.
        let registry = fixture.registry.clone();
        let retiring = thread::spawn(move || registry.retire(candidate));
        retiring_gate.wait_entered();

        // The candidate publishes straight into that window and must still fail.
        validated.resume();
        assert_select_failed(
            selecting.join().expect("select thread"),
            candidate,
            MechanismLifecycle::Retired,
        );

        retiring_gate.resume();
        retiring
            .join()
            .expect("retire thread")
            .expect("retire must succeed");

        assert_eq!(
            fixture
                .selected_mechanism()
                .expect("healthy prior restored"),
            prior.identity
        );
    }

    #[test]
    fn device_loss_racing_a_select_cannot_leave_a_lost_mechanism_selected() {
        let fixture = SelectionFixture::new();
        let _prior = fixture.mechanism();
        let candidate = fixture.mechanism();

        let validated = fixture.gate(candidate, HookPhase::SelectAfterValidation);
        let losing = fixture.gate_device(HookPhase::InvalidateBetweenPhases);

        let selecting = fixture.select_on_thread(candidate);
        validated.wait_entered();

        // Device loss marks every mechanism lost and parks before dropping the
        // selection, mirroring the retirement race.
        let registry = fixture.registry.clone();
        let invalidating =
            thread::spawn(move || registry.invalidate_device(DeviceKey::HOST, "select race"));
        losing.wait_entered();

        validated.resume();
        assert_select_failed(
            selecting.join().expect("select thread"),
            candidate,
            MechanismLifecycle::DeviceLost,
        );

        losing.resume();
        invalidating
            .join()
            .expect("invalidate thread")
            .expect("invalidate must succeed");

        // Every mechanism on the device is lost, so there is no healthy prior to
        // fall back to and the selection must be empty rather than stale.
        fixture.assert_nothing_selected();
    }

    #[test]
    fn retirement_and_device_loss_drop_the_current_selection() {
        let fixture = SelectionFixture::new();
        let retired = fixture.mechanism();
        assert_eq!(
            fixture
                .selected_mechanism()
                .expect("first registration selects"),
            retired.identity
        );

        fixture.registry.retire(retired).expect("retire");
        fixture.assert_nothing_selected();

        let lost = fixture.mechanism();
        assert_eq!(
            fixture
                .selected_mechanism()
                .expect("registration self-heal"),
            lost.identity
        );

        fixture
            .registry
            .invalidate_device(DeviceKey::HOST, "device lost")
            .expect("invalidate");
        fixture.assert_nothing_selected();
    }
}
