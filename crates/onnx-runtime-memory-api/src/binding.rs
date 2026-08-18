//! Registry-issued memory bindings and lifetime pins.
//!
//! This module is intentionally narrower than a process memory manager. It owns
//! registration identity, current-mechanism selection, binding/allocation
//! identity, and the `Arc`s required to keep one mechanism usable. It does not
//! own allocation policy, reservations, leases, deferred release, or physical
//! release recovery.

use std::collections::HashMap;
use std::fmt::Debug;
use std::num::NonZeroU64;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

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
}

#[derive(Debug)]
struct MechanismEntry {
    identity: MechanismIdentity,
    device: DeviceKey,
    context: Arc<ProviderContextEntry>,
    authority: Arc<AuthorityEntry>,
    allocator: Arc<dyn DeviceAllocator>,
    coherence: MechanismCoherence,
    state: Mutex<MechanismState>,
    active_operations: AtomicUsize,
}

impl MechanismEntry {
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
        self.active_operations.fetch_add(1, Ordering::AcqRel);
        drop(state);
        Ok(MechanismOperation {
            mechanism: Arc::clone(self),
        })
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
            provider_context: self.context.identity,
            authority: self.authority.identity,
            coherence: self.coherence,
            lifecycle: state.lifecycle,
            live_allocations: state.allocations.len(),
            active_operations: self.active_operations.load(Ordering::Acquire),
        })
    }
}

#[derive(Debug)]
struct MechanismOperation {
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
    select_after_validation_hook: Mutex<Option<SelectAfterValidationHook>>,
}

#[cfg(test)]
#[derive(Clone, Debug)]
struct SelectAfterValidationHook {
    mechanism: MechanismIdentity,
    entered: Arc<std::sync::Barrier>,
    resume: Arc<std::sync::Barrier>,
}

/// A narrow registry for provider/context pins and binding identity.
///
/// # Lock order
///
/// There are two lock classes: the registry lock protects registration and
/// current selection; each mechanism lock protects only lifecycle and allocation
/// identities. They are never held together. Allocator/capability callbacks,
/// waits, and device operations run with neither lock held.
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
                select_after_validation_hook: Mutex::new(None),
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
            context: context_entry,
            authority: authority_entry,
            allocator,
            coherence,
            state: Mutex::new(MechanismState {
                lifecycle: MechanismLifecycle::Active,
                loss_reason: None,
                allocations: HashMap::new(),
            }),
            active_operations: AtomicUsize::new(0),
        });
        state.mechanisms.insert(identity, entry);
        state.selected.entry(context.device).or_insert(identity);
        Ok(RegisteredMechanism {
            identity,
            device: context.device,
            coherence,
        })
    }

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
        self.wait_after_select_validation(mechanism.identity);
        let prior = self
            .lock_state("publishing mechanism selection")?
            .selected
            .insert(mechanism.device, mechanism.identity);
        let published = entry.snapshot()?;
        if published.lifecycle != MechanismLifecycle::Active {
            let mut state = self.lock_state("withdrawing inactive mechanism selection")?;
            if state.selected.get(&mechanism.device) == Some(&mechanism.identity) {
                if let Some(prior) = prior {
                    state.selected.insert(mechanism.device, prior);
                } else {
                    state.selected.remove(&mechanism.device);
                }
            }
            return Err(BindingError::InactiveMechanism {
                mechanism: mechanism.identity,
                lifecycle: published.lifecycle,
                operation: "selecting a mechanism",
            });
        }
        Ok(())
    }

    #[cfg(test)]
    fn wait_after_select_validation(&self, mechanism: MechanismIdentity) {
        let hook = self
            .inner
            .select_after_validation_hook
            .lock()
            .expect("select test hook lock")
            .clone();
        if let Some(hook) = hook.filter(|hook| hook.mechanism == mechanism) {
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
            provider_context: entry.context.identity,
            authority: entry.authority.identity,
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
    pub fn retire(&self, mechanism: RegisteredMechanism) -> Result<(), BindingError> {
        self.ensure_local(mechanism.identity.0, "mechanism")?;
        let entry = {
            let mut state = self.lock_state("retiring a mechanism")?;
            if state.selected.get(&mechanism.device) == Some(&mechanism.identity) {
                state.selected.remove(&mechanism.device);
            }
            state
                .mechanisms
                .get(&mechanism.identity)
                .cloned()
                .ok_or(BindingError::UnregisteredMechanism(mechanism.identity))?
        };
        let mut mechanism_state = entry.lock_state("retiring a mechanism")?;
        if mechanism_state.lifecycle == MechanismLifecycle::Active {
            mechanism_state.lifecycle = MechanismLifecycle::Retired;
        }
        Ok(())
    }

    /// Invalidate every mechanism and binding for `device`.
    ///
    /// This method changes identity/lifetime state only. It does not invoke a
    /// device callback, free physical memory, release a lease, or refund quota.
    pub fn invalidate_device(
        &self,
        device: DeviceKey,
        reason: impl Into<Arc<str>>,
    ) -> Result<(), BindingError> {
        let reason = reason.into();
        let entries = {
            let mut state = self.lock_state("invalidating a device")?;
            state.selected.remove(&device);
            state
                .mechanisms
                .values()
                .filter(|entry| entry.device == device)
                .cloned()
                .collect::<Vec<_>>()
        };
        for entry in entries {
            let mut state = entry.lock_state("invalidating a device binding")?;
            if state.lifecycle != MechanismLifecycle::Terminated {
                state.lifecycle = MechanismLifecycle::DeviceLost;
                state.loss_reason = Some(Arc::clone(&reason));
            }
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
                .filter(|entry| entry.context.identity == context.identity)
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
        }
        Ok(())
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
            .any(|entry| entry.context.identity == context.identity)
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
            .any(|entry| entry.authority.identity == authority.identity)
        {
            return Err(BindingError::AuthorityInUse(authority.identity));
        }
        state.authorities.remove(&authority.identity);
        Ok(())
    }

    /// Remove a terminal/retired registration once no allocation metadata or
    /// active callback remains. Existing binding/capability handles still pin the
    /// entry and resources, but remain inactive.
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
        if snapshot.lifecycle == MechanismLifecycle::Active
            || snapshot.live_allocations != 0
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
        let ptr = allocate(self.mechanism.allocator.as_ref())?;
        let generation = match self.identities.allocation_generation() {
            Ok(generation) => generation,
            Err(error) => {
                // SAFETY: this is the exact allocation returned immediately
                // above; identity issuance failed before it escaped.
                unsafe { self.mechanism.allocator.deallocate(ptr, bytes, align) };
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
            unsafe { self.mechanism.allocator.deallocate(ptr, bytes, align) };
            return Err(error);
        }
        drop(active);
        Ok(allocation)
    }

    /// Explicitly release through this binding's pinned original allocator.
    ///
    /// This is not `Drop`-based ownership and does not enqueue or synchronize.
    /// On validation/device-loss failure the allocation is returned to the caller
    /// inside [`ExplicitReleaseError`] so its metadata and lifetime pins are not
    /// silently discarded.
    ///
    /// This Phase-3 path calls canonical [`DeviceAllocator::deallocate`]. It does
    /// not expose mapped-attribution refunds from
    /// [`DeviceAllocator::deallocate_with_unmapped`]; EP adoption must wait for
    /// Phase-4 accounting reconciliation.
    pub fn release(&self, allocation: BoundAllocation) -> Result<(), ExplicitReleaseError> {
        let operation = match self.mechanism.begin_release(self.identity, &allocation) {
            Ok(operation) => operation,
            Err(error) => {
                return Err(ExplicitReleaseError {
                    error,
                    allocation: Box::new(allocation),
                });
            }
        };
        let ptr = allocation.ptr;
        let bytes = allocation.bytes;
        let align = allocation.align;
        // SAFETY: begin_release matched and retired the exact live record before
        // this canonical whole-allocation release. No registry lock is held.
        unsafe { self.mechanism.allocator.deallocate(ptr, bytes, align) };
        drop(operation);
        Ok(())
    }

    pub fn virtual_backing(&self) -> Result<Option<BoundVirtualBacking>, BindingError> {
        let operation = self
            .mechanism
            .begin_active("discovering virtual backing capability")?;
        let present = self.mechanism.allocator.as_virtual_backing().is_some();
        drop(operation);
        Ok(present.then(|| BoundVirtualBacking {
            binding: self.clone(),
        }))
    }

    pub fn shared_mapping(&self) -> Result<Option<BoundSharedMapping>, BindingError> {
        let operation = self
            .mechanism
            .begin_active("discovering shared mapping capability")?;
        let present = self.mechanism.allocator.as_shared_mapping().is_some();
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

/// Explicit release failure that preserves allocation metadata and pins.
#[derive(Debug)]
pub struct ExplicitReleaseError {
    error: BindingError,
    allocation: Box<BoundAllocation>,
}

impl ExplicitReleaseError {
    pub const fn error(&self) -> &BindingError {
        &self.error
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
            .allocator
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
            .allocator
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
            .allocator
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
            .allocator
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
            .allocator
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
            .allocator
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
            .allocator
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
            .allocator
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

    #[test]
    fn failed_select_restores_the_prior_healthy_selection() {
        let registry = BindingRegistry::new().unwrap();
        let context = registry
            .register_provider_context(DeviceKey::HOST, Arc::new(()) as Arc<dyn BindingResource>)
            .unwrap();
        let authority = registry
            .register_authority(DeviceKey::HOST, Arc::new(()) as Arc<dyn BindingResource>)
            .unwrap();
        let prior = registry
            .register_allocator(
                context,
                authority,
                Arc::new(HostAllocator) as Arc<dyn DeviceAllocator>,
            )
            .unwrap();
        let candidate = registry
            .register_allocator(
                context,
                authority,
                Arc::new(HostAllocator) as Arc<dyn DeviceAllocator>,
            )
            .unwrap();
        let entered = Arc::new(Barrier::new(2));
        let resume = Arc::new(Barrier::new(2));
        *registry.inner.select_after_validation_hook.lock().unwrap() =
            Some(SelectAfterValidationHook {
                mechanism: candidate.identity,
                entered: Arc::clone(&entered),
                resume: Arc::clone(&resume),
            });

        let selecting_registry = registry.clone();
        let selecting = thread::spawn(move || selecting_registry.select(candidate));
        entered.wait();
        registry.retire(candidate).unwrap();
        resume.wait();

        let error = selecting.join().unwrap().unwrap_err();
        assert!(matches!(
            error,
            BindingError::InactiveMechanism {
                mechanism,
                lifecycle: MechanismLifecycle::Retired,
                operation: "selecting a mechanism",
            } if mechanism == candidate.identity
        ));
        let selected = registry.bind(DeviceKey::HOST).unwrap();
        assert_eq!(selected.identity().mechanism(), prior.identity);
    }
}
