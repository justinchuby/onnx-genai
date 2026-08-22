//! Process-scoped memory registration and allocation transaction coordination.
//!
//! [`ProcessMemoryManager`] is deliberately an orchestration layer. The
//! [`BindingRegistry`] remains the single registration, selection, binding, and
//! allocation-identity ledger. Governors remain the source of budget policy,
//! holders remain the only components allowed to choose what they reclaim, and
//! execution providers remain responsible for stream/copy/commit/decommit and
//! deferred-release ordering.
//!
//! # Lock order
//!
//! Registration is serialized by one manager registration gate and may then
//! enter the registry. The manager state lock is never held while entering the
//! registry. A finite process-limit transition takes the registration gate,
//! then the tier quota gate, allocation book, and allocation charge locks in
//! that order. Publication takes the quota gate before the allocation book;
//! settlement never retains a charge lock while entering either. No manager
//! lock is held while calling a governor, holder, allocator, capability, queue,
//! or fence. Pressure responders and allocation callbacks always run with every
//! manager and registry lock dropped. The registry's own rule remains stricter:
//! its registry and per-mechanism locks are never held together.

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, Weak};

use onnx_runtime_memory_api::{
    AllocationIdentity, AllocationReleaseOutcome, AllocationReleaseState, AuthorityIdentity,
    BindingError, BindingIdentity, BindingRegistry, BindingResource, BoundVirtualBacking,
    DeviceAllocator, DeviceKey, MechanismIdentity, MechanismSnapshot, MemoryBinding,
    OwningAllocation, PreparedAllocationRelease, ProviderContextIdentity, ProviderContextPin,
    ProviderContextPinError, ProviderContextPinSource, QuarantineReason, RegisteredAuthority,
    RegisteredMechanism, RegisteredProviderContext, ReleaseAccounting, ResidualOwnership,
};

use crate::{
    HolderId, MemoryError, MemoryGovernor, MemoryLease, MemoryRole, PressureResponder, Tier,
};

static NEXT_MANAGER_ID: AtomicU64 = AtomicU64::new(1);

/// Provider/context hook for process-wide device-loss publication.
pub trait DeviceLossListener: Send + Sync + fmt::Debug {
    fn mark_device_lost(&self, reason: &str);
}

/// Process-wide ceilings applied in addition to each authority's local budget.
///
/// The default is unbounded so adopting the manager does not silently change
/// existing admission. A finite process limit is a parent quota, not a second
/// physical-memory authority: every transaction still names exactly one local
/// authority.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProcessMemoryLimits {
    pub device_bytes: u64,
    pub host_bytes: u64,
    pub disk_bytes: u64,
}

impl ProcessMemoryLimits {
    pub const UNLIMITED: Self = Self {
        device_bytes: u64::MAX,
        host_bytes: u64::MAX,
        disk_bytes: u64::MAX,
    };

    const fn as_array(self) -> [u64; 3] {
        [self.device_bytes, self.host_bytes, self.disk_bytes]
    }
}

impl Default for ProcessMemoryLimits {
    fn default() -> Self {
        Self::UNLIMITED
    }
}

/// Manager-local canonical identity for one physical accounting authority.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ProcessAuthorityId {
    manager: u64,
    serial: u64,
}

/// Identity used to deduplicate shared physical ownership in snapshots.
///
/// It includes the canonical authority so aliases can never make one physical
/// allocation appear shared across independently grantable books.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SharedPhysicalIdentity {
    manager: u64,
    authority: ProcessAuthorityId,
    serial: u64,
}

/// How a transaction's authority charge is owned.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AllocationChargeMode {
    /// The manager obtains and retains a [`MemoryLease`] until physical release.
    Managed,
    /// The selected allocator/governor already owns the authority charge.
    ///
    /// This is the CUDA VMM mapped-growth path: the manager still coordinates
    /// process quota and observes the charge, but taking another lease would
    /// double-charge the same physical granules.
    AuthorityManaged,
    /// Compatibility allocation with no authority-visible charge.
    ///
    /// Snapshots report these bytes as unattributed rather than pretending the
    /// absence of accounting means zero residency.
    Compatibility,
}

#[derive(Debug)]
struct ProcessQuota {
    limits: [AtomicU64; 3],
    used: [AtomicU64; 3],
    gates: [Mutex<()>; 3],
}

impl ProcessQuota {
    fn new(limits: ProcessMemoryLimits) -> Arc<Self> {
        let limits = limits.as_array();
        Arc::new(Self {
            limits: std::array::from_fn(|index| AtomicU64::new(limits[index])),
            used: std::array::from_fn(|_| AtomicU64::new(0)),
            gates: std::array::from_fn(|_| Mutex::new(())),
        })
    }

    fn lock(&self, tier: Tier) -> MutexGuard<'_, ()> {
        self.gates[tier.index()]
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    fn limit(&self, tier: Tier) -> u64 {
        self.limits[tier.index()].load(Ordering::Acquire)
    }

    fn used(&self, tier: Tier) -> u64 {
        self.used[tier.index()].load(Ordering::Acquire)
    }

    fn available(&self, tier: Tier) -> u64 {
        self.limit(tier).saturating_sub(self.used(tier))
    }

    fn reserve(
        self: &Arc<Self>,
        tier: Tier,
        bytes: u64,
        role: MemoryRole,
    ) -> Result<ProcessQuotaLease, MemoryError> {
        let _gate = self.lock(tier);
        let index = tier.index();
        let used = self.used[index].load(Ordering::Acquire);
        let next = used.checked_add(bytes).ok_or(MemoryError::InvalidRequest {
            tier: tier.name(),
            requested: bytes,
            reason: "the process memory reservation overflows its byte counter",
        })?;
        let limit = self.limits[index].load(Ordering::Acquire);
        if next > limit {
            return Err(MemoryError::TierExhausted {
                tier: tier.name(),
                requested: bytes,
                used,
                limit,
                available: limit.saturating_sub(used),
                role,
            });
        }
        self.used[index].store(next, Ordering::Release);
        Ok(ProcessQuotaLease {
            quota: Arc::clone(self),
            tier,
            bytes,
        })
    }

    fn release(&self, tier: Tier, bytes: u64) {
        let _gate = self.lock(tier);
        let index = tier.index();
        let used = self.used[index].load(Ordering::Acquire);
        self.used[index].store(used.saturating_sub(bytes), Ordering::Release);
    }
}

#[derive(Debug)]
struct ProcessQuotaLease {
    quota: Arc<ProcessQuota>,
    tier: Tier,
    bytes: u64,
}

impl ProcessQuotaLease {
    fn shrink(&mut self, bytes: u64) -> u64 {
        let released = bytes.min(self.bytes);
        if released != 0 {
            self.quota.release(self.tier, released);
            self.bytes -= released;
        }
        released
    }
}

impl Drop for ProcessQuotaLease {
    fn drop(&mut self) {
        if self.bytes != 0 {
            self.quota.release(self.tier, self.bytes);
        }
    }
}

struct AuthorityRecord {
    id: ProcessAuthorityId,
    registered: RegisteredAuthority,
    label: Arc<str>,
    governor: Option<Arc<dyn MemoryGovernor + Send + Sync>>,
    memory_authority: Option<crate::MemoryAuthorityId>,
    process_delegations: Arc<Mutex<HashMap<Tier, ProcessQuotaLease>>>,
}

impl fmt::Debug for AuthorityRecord {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AuthorityRecord")
            .field("id", &self.id)
            .field("registered", &self.registered)
            .field("label", &self.label)
            .field("memory_authority", &self.memory_authority)
            .field(
                "process_delegations",
                &self
                    .process_delegations
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .len(),
            )
            .finish_non_exhaustive()
    }
}

struct AuthorityPin {
    label: Arc<str>,
    _resource: Arc<dyn BindingResource>,
    _governor: Option<Arc<dyn MemoryGovernor + Send + Sync>>,
}

impl fmt::Debug for AuthorityPin {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AuthorityPin")
            .field("label", &self.label)
            .field("governed", &self._governor.is_some())
            .finish_non_exhaustive()
    }
}

struct ContextRecord {
    registered: RegisteredProviderContext,
    label: Arc<str>,
    activity: Mutex<ContextActivity>,
    wake: Condvar,
}

impl fmt::Debug for ContextRecord {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let activity = self
            .activity
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        formatter
            .debug_struct("ContextRecord")
            .field("registered", &self.registered)
            .field("label", &self.label)
            .field("lifecycle", &activity.lifecycle)
            .field("active_transactions", &activity.active_transactions)
            .finish()
    }
}

#[derive(Debug)]
struct ContextActivity {
    lifecycle: ContextLifecycle,
    active_transactions: usize,
}

impl Default for ContextActivity {
    fn default() -> Self {
        Self {
            lifecycle: ContextLifecycle::Active,
            active_transactions: 0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ContextLifecycle {
    Active,
    Retiring,
    Lost,
    Terminated,
}

impl ContextRecord {
    fn begin_transaction(
        self: &Arc<Self>,
    ) -> Result<MemoryContextOperation, AllocationTransactionError> {
        let mut activity = self
            .activity
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if activity.lifecycle != ContextLifecycle::Active {
            return Err(AllocationTransactionError::TerminatedContext(
                self.registered.identity(),
            ));
        }
        activity.active_transactions = activity.active_transactions.checked_add(1).ok_or(
            AllocationTransactionError::InvalidPublication(
                "context transaction counter overflowed",
            ),
        )?;
        Ok(MemoryContextOperation {
            context: Arc::clone(self),
        })
    }

    fn ensure_active(&self) -> Result<(), AllocationTransactionError> {
        if self
            .activity
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .lifecycle
            != ContextLifecycle::Active
        {
            return Err(AllocationTransactionError::TerminatedContext(
                self.registered.identity(),
            ));
        }
        Ok(())
    }

    fn mark_retiring(&self) {
        let mut activity = self
            .activity
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if activity.lifecycle == ContextLifecycle::Active {
            activity.lifecycle = ContextLifecycle::Retiring;
        }
    }

    fn mark_lost(&self) {
        let mut activity = self
            .activity
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if activity.lifecycle != ContextLifecycle::Terminated {
            activity.lifecycle = ContextLifecycle::Lost;
        }
    }

    fn prepare_confirmation(&self) -> Result<bool, AllocationTransactionError> {
        let activity = self
            .activity
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        match activity.lifecycle {
            ContextLifecycle::Lost => Ok(true),
            ContextLifecycle::Terminated => Ok(false),
            lifecycle => Err(AllocationTransactionError::ContextNotLost {
                context: self.registered.identity(),
                lifecycle: match lifecycle {
                    ContextLifecycle::Active => "active",
                    ContextLifecycle::Retiring => "retiring",
                    ContextLifecycle::Lost => "lost",
                    ContextLifecycle::Terminated => "terminated",
                },
            }),
        }
    }

    fn mark_terminated(&self) {
        self.activity
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .lifecycle = ContextLifecycle::Terminated;
    }

    fn wait_quiescent(&self) {
        let mut activity = self
            .activity
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        while activity.active_transactions != 0 {
            activity = self
                .wake
                .wait(activity)
                .unwrap_or_else(|poisoned| poisoned.into_inner());
        }
    }
}

pub struct MemoryContextOperation {
    context: Arc<ContextRecord>,
}

impl fmt::Debug for MemoryContextOperation {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("MemoryContextOperation")
            .field("context", &self.context.registered.identity())
            .finish_non_exhaustive()
    }
}

impl Drop for MemoryContextOperation {
    fn drop(&mut self) {
        let mut activity = self
            .context
            .activity
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        activity.active_transactions = activity.active_transactions.saturating_sub(1);
        if activity.active_transactions == 0 {
            self.context.wake.notify_all();
        }
    }
}

#[derive(Debug)]
struct MechanismRecord {
    registered: RegisteredMechanism,
    context: Arc<ContextRecord>,
    authority: Arc<AuthorityRecord>,
    label: Arc<str>,
}

struct HolderRecord {
    id: HolderId,
    authority: ProcessAuthorityId,
    label: Arc<str>,
    responder: Option<Weak<dyn PressureResponder>>,
}

impl fmt::Debug for HolderRecord {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("HolderRecord")
            .field("id", &self.id)
            .field("authority", &self.authority)
            .field("label", &self.label)
            .field("has_responder", &self.responder.is_some())
            .finish()
    }
}

#[derive(Default)]
struct ManagerState {
    /// Transaction gates only. Registration membership remains solely in
    /// `BindingRegistry`.
    contexts: HashMap<ProviderContextIdentity, Arc<ContextRecord>>,
    governed_authorities: HashMap<crate::MemoryAuthorityId, Arc<AuthorityRecord>>,
    authorities: HashMap<ProcessAuthorityId, Arc<AuthorityRecord>>,
    authority_views: HashMap<AuthorityIdentity, Arc<AuthorityRecord>>,
    holders: HashMap<HolderId, Arc<HolderRecord>>,
    device_loss_listeners: HashMap<DeviceKey, Vec<Weak<dyn DeviceLossListener>>>,
    device_loss_generation: HashMap<DeviceKey, u64>,
    lost_devices: HashSet<DeviceKey>,
}

impl fmt::Debug for ManagerState {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ManagerState")
            .field("context_transaction_gates", &self.contexts.len())
            .field("authorities", &self.authorities.len())
            .field("authority_views", &self.authority_views.len())
            .field("holders", &self.holders.len())
            .field("device_loss_devices", &self.device_loss_listeners.len())
            .finish()
    }
}

struct ManagerInner {
    id: u64,
    registry: BindingRegistry,
    next_identity: AtomicU64,
    registration_gate: Mutex<()>,
    state: Mutex<ManagerState>,
    book: Arc<SettlementBook>,
}

impl fmt::Debug for ManagerInner {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ManagerInner")
            .field("id", &self.id)
            .finish_non_exhaustive()
    }
}

impl ManagerInner {
    fn lock_state(&self) -> MutexGuard<'_, ManagerState> {
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    fn next(&self) -> Result<u64, AllocationTransactionError> {
        self.next_identity
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
                value.checked_add(1)
            })
            .map_err(|_| AllocationTransactionError::IdentityExhausted)
    }

    fn authority(
        &self,
        id: ProcessAuthorityId,
    ) -> Result<Arc<AuthorityRecord>, AllocationTransactionError> {
        if id.manager != self.id {
            return Err(AllocationTransactionError::ForeignHandle("authority"));
        }
        self.lock_state()
            .authorities
            .get(&id)
            .cloned()
            .ok_or(AllocationTransactionError::Unregistered("authority"))
    }

    fn pressure(&self, authority: ProcessAuthorityId, tier: Tier, want: u64) -> u64 {
        let responders = {
            let state = self.lock_state();
            state
                .holders
                .values()
                .filter(|holder| holder.authority == authority)
                .filter_map(|holder| holder.responder.as_ref()?.upgrade())
                .collect::<Vec<_>>()
        };
        responders.into_iter().fold(0_u64, |released, responder| {
            released.saturating_add(responder.on_pressure(tier, want.saturating_sub(released)))
        })
    }
}

/// Process-level coordinator over one binding registry and multiple authorities.
#[derive(Clone, Debug)]
pub struct ProcessMemoryManager {
    inner: Arc<ManagerInner>,
}

impl ProcessMemoryManager {
    pub fn new() -> Result<Self, BindingError> {
        Self::with_limits(ProcessMemoryLimits::UNLIMITED)
    }

    pub fn with_limits(limits: ProcessMemoryLimits) -> Result<Self, BindingError> {
        let id = NEXT_MANAGER_ID
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
                value.checked_add(1)
            })
            .map_err(|_| BindingError::IdentityExhausted)?;
        let quota = ProcessQuota::new(limits);
        Ok(Self {
            inner: Arc::new(ManagerInner {
                id,
                registry: BindingRegistry::new()?,
                next_identity: AtomicU64::new(1),
                registration_gate: Mutex::new(()),
                state: Mutex::new(ManagerState::default()),
                book: Arc::new(SettlementBook::new(id, quota)),
            }),
        })
    }

    pub fn set_process_limit(&self, tier: Tier, bytes: u64) -> Result<(), MemoryError> {
        // Authority delegation and a finite-limit transition both create parent
        // coverage. Serialize them so the transition sees the canonical set of
        // authorities whose allocations are already covered by a delegation.
        let _registration = self
            .inner
            .registration_gate
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let authorities = {
            let state = self.inner.lock_state();
            state.authorities.values().cloned().collect::<Vec<_>>()
        };
        let delegated = authorities
            .into_iter()
            .filter_map(|authority| {
                authority
                    .process_delegations
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .contains_key(&tier)
                    .then_some(authority.id)
            })
            .collect::<HashSet<_>>();
        self.inner.book.set_process_limit(tier, bytes, &delegated)
    }

    pub fn process_limit(&self, tier: Tier) -> u64 {
        self.inner.book.quota.limit(tier)
    }

    pub fn process_used(&self, tier: Tier) -> u64 {
        self.inner.book.quota.used(tier)
    }

    /// Whether two handles refer to the exact same process manager instance.
    pub fn is_same_instance(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }

    pub fn downgrade(&self) -> WeakProcessMemoryManager {
        WeakProcessMemoryManager {
            inner: Arc::downgrade(&self.inner),
        }
    }

    pub fn register_provider_context(
        &self,
        device: DeviceKey,
        label: impl Into<Arc<str>>,
        resource: Arc<dyn BindingResource>,
    ) -> Result<RegisteredMemoryContext, BindingError> {
        let label = label.into();
        let _registration = self
            .inner
            .registration_gate
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let registered = self
            .inner
            .registry
            .register_provider_context(device, resource)?;
        let record = Arc::new(ContextRecord {
            registered,
            label,
            activity: Mutex::new(ContextActivity::default()),
            wake: Condvar::new(),
        });
        self.inner
            .lock_state()
            .contexts
            .insert(registered.identity(), Arc::clone(&record));
        Ok(RegisteredMemoryContext {
            manager: self.inner.id,
            record,
        })
    }

    /// Register or recover the canonical manager authority for `governor`.
    ///
    /// Re-registering the same [`crate::MemoryAuthorityId`] returns the existing
    /// authority identity. This is how multiple sessions and UMA/shared aliases
    /// reuse one grantable capacity rather than creating independent books.
    pub fn register_authority(
        &self,
        device: DeviceKey,
        label: impl Into<Arc<str>>,
        resource: Arc<dyn BindingResource>,
        governor: Arc<dyn MemoryGovernor + Send + Sync>,
    ) -> Result<RegisteredMemoryAuthority, AllocationTransactionError> {
        let label = label.into();
        // Third-party callbacks run before the non-reentrant registration gate.
        let memory_authority = governor.authority_id();
        let _registration = self
            .inner
            .registration_gate
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if memory_authority.device() != device {
            return Err(AllocationTransactionError::AuthorityDeviceMismatch {
                expected: device,
                actual: memory_authority.device(),
            });
        }
        let (canonical, existing_view) = {
            let state = self.inner.lock_state();
            let canonical = state.governed_authorities.get(&memory_authority).cloned();
            let existing_view = canonical.as_ref().and_then(|canonical| {
                state
                    .authority_views
                    .values()
                    .find(|record| {
                        record.id == canonical.id && record.registered.device() == device
                    })
                    .cloned()
            });
            (canonical, existing_view)
        };
        if let Some(record) = existing_view {
            return Ok(RegisteredMemoryAuthority {
                manager: self.inner.id,
                record,
            });
        }
        let pin = Arc::new(AuthorityPin {
            label: Arc::clone(&label),
            _resource: resource,
            _governor: Some(Arc::clone(&governor)),
        });
        let registered = self
            .inner
            .registry
            .register_authority(device, pin)
            .map_err(AllocationTransactionError::Binding)?;
        let id = match canonical.as_ref() {
            Some(canonical) => canonical.id,
            None => ProcessAuthorityId {
                manager: self.inner.id,
                serial: self.inner.next()?,
            },
        };
        let record = Arc::new(AuthorityRecord {
            id,
            registered,
            label,
            governor: Some(governor),
            memory_authority: Some(memory_authority),
            process_delegations: canonical.as_ref().map_or_else(
                || Arc::new(Mutex::new(HashMap::new())),
                |canonical| Arc::clone(&canonical.process_delegations),
            ),
        });
        let mut state = self.inner.lock_state();
        state
            .governed_authorities
            .insert(memory_authority, Arc::clone(&record));
        state.authorities.insert(id, Arc::clone(&record));
        state
            .authority_views
            .insert(registered.identity(), Arc::clone(&record));
        Ok(RegisteredMemoryAuthority {
            manager: self.inner.id,
            record,
        })
    }

    /// Register an allocator path whose bytes are not charged by a governor.
    pub fn register_compatibility_authority(
        &self,
        device: DeviceKey,
        label: impl Into<Arc<str>>,
        resource: Arc<dyn BindingResource>,
    ) -> Result<RegisteredMemoryAuthority, AllocationTransactionError> {
        let label = label.into();
        let _registration = self
            .inner
            .registration_gate
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let registered = self
            .inner
            .registry
            .register_authority(
                device,
                Arc::new(AuthorityPin {
                    label: Arc::clone(&label),
                    _resource: resource,
                    _governor: None,
                }),
            )
            .map_err(AllocationTransactionError::Binding)?;
        let id = ProcessAuthorityId {
            manager: self.inner.id,
            serial: self.inner.next()?,
        };
        let record = Arc::new(AuthorityRecord {
            id,
            registered,
            label,
            governor: None,
            memory_authority: None,
            process_delegations: Arc::new(Mutex::new(HashMap::new())),
        });
        let mut state = self.inner.lock_state();
        state.authorities.insert(id, Arc::clone(&record));
        state
            .authority_views
            .insert(registered.identity(), Arc::clone(&record));
        Ok(RegisteredMemoryAuthority {
            manager: self.inner.id,
            record,
        })
    }

    /// Register another device view of one canonical physical authority.
    ///
    /// UMA and shared-memory mechanisms can therefore use device-specific
    /// registry identities while all views retain one process authority and one
    /// governor. No independently grantable capacity is created.
    pub fn register_authority_alias(
        &self,
        canonical: &RegisteredMemoryAuthority,
        device: DeviceKey,
        label: impl Into<Arc<str>>,
        resource: Arc<dyn BindingResource>,
    ) -> Result<RegisteredMemoryAuthority, AllocationTransactionError> {
        self.ensure_manager(canonical.manager, "authority")?;
        let label = label.into();
        let _registration = self
            .inner
            .registration_gate
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let pin = Arc::new(AuthorityPin {
            label: Arc::clone(&label),
            _resource: resource,
            _governor: canonical.record.governor.clone(),
        });
        let registered = self
            .inner
            .registry
            .register_authority(device, pin)
            .map_err(AllocationTransactionError::Binding)?;
        let record = Arc::new(AuthorityRecord {
            id: canonical.record.id,
            registered,
            label,
            governor: canonical.record.governor.clone(),
            memory_authority: canonical.record.memory_authority,
            process_delegations: Arc::clone(&canonical.record.process_delegations),
        });
        self.inner
            .lock_state()
            .authority_views
            .insert(registered.identity(), Arc::clone(&record));
        Ok(RegisteredMemoryAuthority {
            manager: self.inner.id,
            record,
        })
    }

    pub fn register_holder(
        &self,
        authority: &RegisteredMemoryAuthority,
        label: impl Into<Arc<str>>,
        responder: Option<Arc<dyn PressureResponder>>,
    ) -> Result<RegisteredMemoryHolder, AllocationTransactionError> {
        self.ensure_manager(authority.manager, "authority")?;
        let raw = self.inner.next()?;
        let holder = HolderId::new(raw);
        let record = Arc::new(HolderRecord {
            id: holder,
            authority: authority.record.id,
            label: label.into(),
            responder: responder.as_ref().map(Arc::downgrade),
        });
        self.inner
            .lock_state()
            .holders
            .insert(holder, Arc::clone(&record));
        Ok(RegisteredMemoryHolder {
            manager: self.inner.id,
            record,
        })
    }

    /// Register a weak provider/context listener for device loss.
    pub fn register_device_loss_listener(
        &self,
        device: DeviceKey,
        listener: &Arc<dyn DeviceLossListener>,
    ) -> Result<u64, AllocationTransactionError> {
        let _registration = self
            .inner
            .registration_gate
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let (generation, lost) = {
            let mut state = self.inner.lock_state();
            state
                .device_loss_listeners
                .entry(device)
                .or_default()
                .push(Arc::downgrade(listener));
            (
                state
                    .device_loss_generation
                    .get(&device)
                    .copied()
                    .unwrap_or(0),
                state.lost_devices.contains(&device),
            )
        };
        drop(_registration);
        if lost {
            listener.mark_device_lost("device was already lost before listener registration");
        }
        Ok(generation)
    }

    /// Validate the generation captured before a provider registration began.
    pub fn finish_device_registration(
        &self,
        device: DeviceKey,
        generation: u64,
    ) -> Result<(), AllocationTransactionError> {
        let _registration = self
            .inner
            .registration_gate
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let state = self.inner.lock_state();
        let current = state
            .device_loss_generation
            .get(&device)
            .copied()
            .unwrap_or(0);
        if state.lost_devices.contains(&device) || current != generation {
            return Err(AllocationTransactionError::DeviceRegistrationLost {
                device,
                expected_generation: generation,
                actual_generation: current,
            });
        }
        Ok(())
    }

    /// Delegate a fixed slice of process quota to one local authority.
    ///
    /// Delegation is one-time per tier and consumes process capacity for the
    /// authority's lifetime. Transactions using that authority then set their
    /// per-allocation process reservation to zero: the local governor arbitrates
    /// within this already-exclusive slice, so no byte can be granted twice.
    pub fn delegate_authority_capacity(
        &self,
        authority: &RegisteredMemoryAuthority,
        tier: Tier,
        bytes: u64,
    ) -> Result<(), AllocationTransactionError> {
        self.ensure_manager(authority.manager, "authority")?;
        let governor = authority.record.governor.as_ref().ok_or(
            AllocationTransactionError::MissingGovernor(authority.record.id),
        )?;
        let local_capacity = governor
            .used(tier)
            .checked_add(governor.available(tier))
            .ok_or(MemoryError::InvalidRequest {
                tier: tier.name(),
                requested: bytes,
                reason: "the authority capacity overflows its byte counter",
            })?;
        if local_capacity > bytes {
            return Err(AllocationTransactionError::DelegationTooSmall {
                authority: authority.record.id,
                tier,
                delegated: bytes,
                local_capacity,
            });
        }
        let _registration = self
            .inner
            .registration_gate
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        {
            let delegations = authority
                .record
                .process_delegations
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if delegations.contains_key(&tier) {
                return Err(AllocationTransactionError::DelegationAlreadyExists {
                    authority: authority.record.id,
                    tier,
                });
            }
        }
        // Process quota is reserved after all validation and before publication.
        // No governor or holder callback runs under a manager state lock.
        let lease = self
            .inner
            .book
            .quota
            .reserve(tier, bytes, MemoryRole::Activation)?;
        authority
            .record
            .process_delegations
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .insert(tier, lease);
        Ok(())
    }

    pub fn register_allocator(
        &self,
        context: &RegisteredMemoryContext,
        authority: &RegisteredMemoryAuthority,
        label: impl Into<Arc<str>>,
        allocator: Arc<dyn DeviceAllocator>,
    ) -> Result<RegisteredMemoryMechanism, AllocationTransactionError> {
        self.ensure_manager(context.manager, "provider context")?;
        self.ensure_manager(authority.manager, "authority")?;
        let label = label.into();
        // Sample the third-party callback before entering the manager gate.
        let allocator_device = allocator.device();
        let _registration = self
            .inner
            .registration_gate
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        context.record.ensure_active()?;
        let registered = self
            .inner
            .registry
            .register_allocator_with_device(
                context.record.registered,
                authority.record.registered,
                allocator,
                allocator_device,
            )
            .map_err(AllocationTransactionError::Binding)?;
        let record = Arc::new(MechanismRecord {
            registered,
            context: Arc::clone(&context.record),
            authority: Arc::clone(&authority.record),
            label,
        });
        Ok(RegisteredMemoryMechanism {
            manager: self.inner.id,
            record,
        })
    }

    pub fn select(
        &self,
        mechanism: &RegisteredMemoryMechanism,
    ) -> Result<(), AllocationTransactionError> {
        self.ensure_manager(mechanism.manager, "mechanism")?;
        self.inner
            .registry
            .select(mechanism.record.registered)
            .map_err(AllocationTransactionError::Binding)
    }

    pub fn bind(
        &self,
        device: DeviceKey,
    ) -> Result<ScopedMemoryBinding, AllocationTransactionError> {
        let binding = self
            .inner
            .registry
            .bind(device)
            .map_err(AllocationTransactionError::Binding)?;
        let (authority, context) = {
            let state = self.inner.lock_state();
            (
                state
                    .authority_views
                    .get(&binding.identity().authority())
                    .cloned()
                    .ok_or(AllocationTransactionError::Unregistered("authority"))?,
                state
                    .contexts
                    .get(&binding.identity().provider_context())
                    .cloned()
                    .ok_or(AllocationTransactionError::Unregistered("provider context"))?,
            )
        };
        context.ensure_active()?;
        Ok(ScopedMemoryBinding {
            manager: Arc::clone(&self.inner),
            context,
            authority,
            binding,
        })
    }

    pub fn bind_registered(
        &self,
        mechanism: &RegisteredMemoryMechanism,
    ) -> Result<ScopedMemoryBinding, AllocationTransactionError> {
        self.ensure_manager(mechanism.manager, "mechanism")?;
        mechanism.record.context.ensure_active()?;
        let binding = self
            .inner
            .registry
            .bind_registered(mechanism.record.registered)
            .map_err(AllocationTransactionError::Binding)?;
        Ok(ScopedMemoryBinding {
            manager: Arc::clone(&self.inner),
            context: Arc::clone(&mechanism.record.context),
            authority: Arc::clone(&mechanism.record.authority),
            binding,
        })
    }

    pub fn retire(
        &self,
        mechanism: &RegisteredMemoryMechanism,
    ) -> Result<(), AllocationTransactionError> {
        self.ensure_manager(mechanism.manager, "mechanism")?;
        self.inner
            .registry
            .retire(mechanism.record.registered)
            .map_err(AllocationTransactionError::Binding)
    }

    pub fn remove_mechanism(
        &self,
        mechanism: &RegisteredMemoryMechanism,
    ) -> Result<(), AllocationTransactionError> {
        self.ensure_manager(mechanism.manager, "mechanism")?;
        let _registration = self
            .inner
            .registration_gate
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        self.inner
            .registry
            .remove(mechanism.record.registered)
            .map_err(AllocationTransactionError::Binding)?;
        Ok(())
    }

    pub fn unregister_holder(
        &self,
        holder: &RegisteredMemoryHolder,
    ) -> Result<(), AllocationTransactionError> {
        self.ensure_manager(holder.manager, "holder")?;
        if self
            .inner
            .book
            .snapshots()
            .iter()
            .any(|allocation| allocation.holder == holder.record.id)
        {
            return Err(AllocationTransactionError::HolderInUse(holder.record.id));
        }
        self.inner.lock_state().holders.remove(&holder.record.id);
        Ok(())
    }

    pub fn remove_provider_context(
        &self,
        context: &RegisteredMemoryContext,
    ) -> Result<(), AllocationTransactionError> {
        self.ensure_manager(context.manager, "provider context")?;
        let _registration = self
            .inner
            .registration_gate
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        context.record.mark_retiring();
        drop(_registration);
        context.record.wait_quiescent();
        let _registration = self
            .inner
            .registration_gate
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        self.inner
            .registry
            .remove_provider_context(context.record.registered)
            .map_err(AllocationTransactionError::Binding)?;
        self.inner
            .lock_state()
            .contexts
            .remove(&context.record.registered.identity());
        Ok(())
    }

    /// Stop accepting new provider-context operations without waiting.
    pub fn retire_context(
        &self,
        context: &RegisteredMemoryContext,
    ) -> Result<(), AllocationTransactionError> {
        self.ensure_manager(context.manager, "provider context")?;
        let _registration = self
            .inner
            .registration_gate
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        context.record.mark_retiring();
        Ok(())
    }

    pub fn remove_authority(
        &self,
        authority: &RegisteredMemoryAuthority,
    ) -> Result<(), AllocationTransactionError> {
        self.ensure_manager(authority.manager, "authority")?;
        let _registration = self
            .inner
            .registration_gate
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        self.inner
            .registry
            .remove_authority(authority.record.registered)
            .map_err(AllocationTransactionError::Binding)?;
        let removed = {
            let mut state = self.inner.lock_state();
            let removed = state
                .authority_views
                .remove(&authority.record.registered.identity());
            let remaining_view = state
                .authority_views
                .values()
                .find(|record| record.id == authority.record.id)
                .cloned();
            if let Some(remaining) = remaining_view {
                state
                    .authorities
                    .insert(authority.record.id, Arc::clone(&remaining));
                if let Some(memory_authority) = authority.record.memory_authority {
                    state
                        .governed_authorities
                        .insert(memory_authority, remaining);
                }
            } else {
                state.authorities.remove(&authority.record.id);
                if let Some(memory_authority) = authority.record.memory_authority {
                    state.governed_authorities.remove(&memory_authority);
                }
            }
            removed
        };
        drop(removed);
        Ok(())
    }

    /// Route device loss through the Phase-3 registry lifecycle.
    ///
    /// This changes no accounting. Charges remain live until a structured
    /// release outcome or confirmed context termination proves what disappeared.
    pub fn invalidate_device(
        &self,
        device: DeviceKey,
        reason: impl Into<Arc<str>>,
    ) -> Result<(), AllocationTransactionError> {
        let reason = reason.into();
        let registration = self
            .inner
            .registration_gate
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let (contexts, listeners) = {
            let mut state = self.inner.lock_state();
            let generation = state
                .device_loss_generation
                .get(&device)
                .copied()
                .unwrap_or(0)
                .checked_add(1)
                .ok_or(AllocationTransactionError::IdentityExhausted)?;
            state.device_loss_generation.insert(device, generation);
            state.lost_devices.insert(device);
            let contexts = state
                .contexts
                .values()
                .filter(|context| context.registered.device() == device)
                .cloned()
                .collect::<Vec<_>>();
            let listeners = state.device_loss_listeners.entry(device).or_default();
            listeners.retain(|listener| listener.strong_count() != 0);
            let listeners = listeners
                .iter()
                .filter_map(Weak::upgrade)
                .collect::<Vec<_>>();
            (contexts, listeners)
        };
        for context in contexts {
            context.mark_lost();
        }
        drop(registration);
        for listener in listeners {
            listener.mark_device_lost(&reason);
        }
        self.inner
            .registry
            .invalidate_device(device, reason)
            .map_err(AllocationTransactionError::Binding)
    }

    /// Confirm context termination and discharge charges only after the registry
    /// verifies every mechanism operation is quiescent.
    pub fn confirm_context_terminated(
        &self,
        context: &RegisteredMemoryContext,
    ) -> Result<(), AllocationTransactionError> {
        self.ensure_manager(context.manager, "provider context")?;
        if !context.record.prepare_confirmation()? {
            return Ok(());
        }
        // Waiting is on the context-local transaction counter only, with no
        // manager/registry/governor lock held.
        context.record.wait_quiescent();
        if let Err(error) = self
            .inner
            .registry
            .confirm_context_terminated(context.record.registered)
        {
            return Err(AllocationTransactionError::Binding(error));
        }
        context.record.mark_terminated();
        let charges = self
            .inner
            .book
            .charges_for_context(context.record.registered.identity());
        for charge in charges {
            self.inner.book.context_terminated(&charge);
        }
        Ok(())
    }

    pub fn mechanism_snapshot(
        &self,
        mechanism: &RegisteredMemoryMechanism,
    ) -> Result<MechanismSnapshot, AllocationTransactionError> {
        self.ensure_manager(mechanism.manager, "mechanism")?;
        self.inner
            .registry
            .snapshot(mechanism.record.registered)
            .map_err(AllocationTransactionError::Binding)
    }

    pub fn snapshot(&self) -> Result<ProcessMemorySnapshot, AllocationTransactionError> {
        let authorities = {
            let state = self.inner.lock_state();
            state.authorities.values().cloned().collect::<Vec<_>>()
        };
        let mechanism_snapshots = self
            .inner
            .registry
            .snapshots()
            .map_err(AllocationTransactionError::Binding)?;
        let authority_snapshots = authorities
            .iter()
            .map(|authority| {
                let (used, available, oversubscribed) = match authority.governor.as_ref() {
                    Some(governor) => (
                        std::array::from_fn(|index| governor.used(Tier::ALL[index])),
                        std::array::from_fn(|index| governor.available(Tier::ALL[index])),
                        std::array::from_fn(|index| {
                            governor.oversubscribed_bytes(Tier::ALL[index])
                        }),
                    ),
                    None => ([0; 3], [0; 3], [0; 3]),
                };
                AuthorityMemorySnapshot {
                    authority: authority.id,
                    label: Arc::clone(&authority.label),
                    memory_authority: authority.memory_authority,
                    device: authority.registered.device(),
                    governed: authority.governor.is_some(),
                    used,
                    available,
                    oversubscribed,
                }
            })
            .collect::<Vec<_>>();
        let allocations = self.inner.book.snapshots();
        let mut physical = HashMap::<SharedPhysicalIdentity, u64>::new();
        let mut attributed_allocation_charged_bytes = 0_u64;
        let mut process_reserved_bytes = 0_u64;
        let mut mapped_bytes = 0_u64;
        let mut unattributed_bytes = 0_u64;
        let mut unknown_physical_allocations = 0_usize;
        for allocation in &allocations {
            attributed_allocation_charged_bytes =
                attributed_allocation_charged_bytes.saturating_add(allocation.charged_bytes);
            process_reserved_bytes =
                process_reserved_bytes.saturating_add(allocation.process_reserved_bytes);
            mapped_bytes = mapped_bytes.saturating_add(allocation.mapped_bytes.unwrap_or(0));
            unattributed_bytes = unattributed_bytes.saturating_add(allocation.unattributed_bytes);
            match allocation.physical_bytes {
                Some(bytes) => {
                    physical
                        .entry(allocation.shared_physical)
                        .and_modify(|current| *current = (*current).max(bytes))
                        .or_insert(bytes);
                }
                None => unknown_physical_allocations += 1,
            }
        }
        Ok(ProcessMemorySnapshot {
            process_limits: ProcessMemoryLimits {
                device_bytes: self.process_limit(Tier::Device),
                host_bytes: self.process_limit(Tier::Host),
                disk_bytes: self.process_limit(Tier::Disk),
            },
            process_used: [
                self.process_used(Tier::Device),
                self.process_used(Tier::Host),
                self.process_used(Tier::Disk),
            ],
            authority_count: authorities.len(),
            charged_bytes: authority_snapshots.iter().fold(0_u64, |total, authority| {
                total.saturating_add(authority.used.iter().copied().sum::<u64>())
            }),
            authority_snapshots,
            mechanism_snapshots,
            allocations,
            attributed_allocation_charged_bytes,
            process_reserved_bytes,
            known_physical_bytes: physical.values().copied().sum(),
            mapped_bytes,
            unattributed_bytes,
            unknown_physical_allocations,
        })
    }

    fn ensure_manager(
        &self,
        manager: u64,
        kind: &'static str,
    ) -> Result<(), AllocationTransactionError> {
        if manager != self.inner.id {
            return Err(AllocationTransactionError::ForeignHandle(kind));
        }
        Ok(())
    }
}

/// Non-owning manager reference used by provider-context drain callbacks.
#[derive(Clone, Debug)]
pub struct WeakProcessMemoryManager {
    inner: Weak<ManagerInner>,
}

impl WeakProcessMemoryManager {
    pub fn upgrade(&self) -> Option<ProcessMemoryManager> {
        self.inner
            .upgrade()
            .map(|inner| ProcessMemoryManager { inner })
    }
}

#[derive(Clone, Debug)]
pub struct RegisteredMemoryContext {
    manager: u64,
    record: Arc<ContextRecord>,
}

impl RegisteredMemoryContext {
    pub fn identity(&self) -> ProviderContextIdentity {
        self.record.registered.identity()
    }

    pub fn label(&self) -> &str {
        &self.record.label
    }

    /// The device this context was registered for.
    pub fn device(&self) -> DeviceKey {
        self.record.registered.device()
    }

    /// A pin source that keeps this context from completing teardown.
    ///
    /// Handed to mechanisms that queue deferred releases — notably plugin
    /// allocators behind the nxmem ABI, which cannot depend on this crate.
    /// Each pin is one entry in the same transaction count
    /// [`ProcessMemoryManager::remove_provider_context`] waits on, so an
    /// outstanding release blocks teardown through the mechanism that already
    /// governs in-tree work rather than a parallel one.
    pub fn pin_source(&self) -> Arc<dyn ProviderContextPinSource> {
        Arc::new(ContextPinSource {
            record: Arc::clone(&self.record),
        })
    }
}

/// Hands out [`ContextPin`]s for one registered provider context.
#[derive(Debug)]
struct ContextPinSource {
    record: Arc<ContextRecord>,
}

impl ProviderContextPinSource for ContextPinSource {
    fn context(&self) -> ProviderContextIdentity {
        self.record.registered.identity()
    }

    fn pin(&self) -> Result<Box<dyn ProviderContextPin>, ProviderContextPinError> {
        match self.record.begin_transaction() {
            Ok(operation) => Ok(Box::new(ContextPin {
                context: self.record.registered.identity(),
                _operation: operation,
            })),
            // `begin_transaction` refuses for two reasons: the context is no
            // longer `Active`, or its counter would overflow. Both are
            // refusals to attach new work, which is what the caller needs to
            // know; neither may be reported as a successful unpinned queue.
            Err(AllocationTransactionError::TerminatedContext(identity)) => {
                Err(ProviderContextPinError::ContextUnavailable(identity))
            }
            Err(_) => Err(ProviderContextPinError::PinCountOverflow(
                self.record.registered.identity(),
            )),
        }
    }
}

/// One outstanding claim on a provider context.
#[derive(Debug)]
struct ContextPin {
    context: ProviderContextIdentity,
    /// Dropping this decrements the context's transaction count and wakes
    /// `wait_quiescent`. It is never read; the pin *is* the guard.
    _operation: MemoryContextOperation,
}

impl ProviderContextPin for ContextPin {
    fn context(&self) -> ProviderContextIdentity {
        self.context
    }
}

#[derive(Clone, Debug)]
pub struct RegisteredMemoryAuthority {
    manager: u64,
    record: Arc<AuthorityRecord>,
}

impl RegisteredMemoryAuthority {
    pub fn identity(&self) -> ProcessAuthorityId {
        self.record.id
    }

    pub fn binding_identity(&self) -> AuthorityIdentity {
        self.record.registered.identity()
    }

    pub fn memory_authority_id(&self) -> Option<crate::MemoryAuthorityId> {
        self.record.memory_authority
    }

    pub fn device(&self) -> DeviceKey {
        self.record.registered.device()
    }

    pub fn label(&self) -> &str {
        &self.record.label
    }

    pub fn has_process_delegation(&self, tier: Tier) -> bool {
        self.record
            .process_delegations
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .contains_key(&tier)
    }
}

#[derive(Clone, Debug)]
pub struct RegisteredMemoryMechanism {
    manager: u64,
    record: Arc<MechanismRecord>,
}

impl RegisteredMemoryMechanism {
    pub fn identity(&self) -> MechanismIdentity {
        self.record.registered.identity()
    }

    pub fn device(&self) -> DeviceKey {
        self.record.registered.device()
    }

    pub fn authority(&self) -> ProcessAuthorityId {
        self.record.authority.id
    }

    pub fn label(&self) -> &str {
        &self.record.label
    }
}

#[derive(Clone)]
pub struct RegisteredMemoryHolder {
    manager: u64,
    record: Arc<HolderRecord>,
}

impl fmt::Debug for RegisteredMemoryHolder {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RegisteredMemoryHolder")
            .field("id", &self.record.id)
            .field("authority", &self.record.authority)
            .field("label", &self.record.label)
            .finish()
    }
}

impl RegisteredMemoryHolder {
    pub fn id(&self) -> HolderId {
        self.record.id
    }

    pub fn authority(&self) -> ProcessAuthorityId {
        self.record.authority
    }

    pub fn label(&self) -> &str {
        &self.record.label
    }
}

/// Manager-issued binding pinned to one device/mechanism/context/authority.
#[derive(Clone, Debug)]
pub struct ScopedMemoryBinding {
    /// Dropped first so allocator teardown still observes every pin below.
    binding: MemoryBinding,
    context: Arc<ContextRecord>,
    authority: Arc<AuthorityRecord>,
    manager: Arc<ManagerInner>,
}

/// Cloneable provider-context transaction gate for non-allocation device work.
#[derive(Clone, Debug)]
pub struct MemoryContextScope {
    context: Arc<ContextRecord>,
}

impl MemoryContextScope {
    pub fn enter(&self) -> Result<MemoryContextOperation, AllocationTransactionError> {
        self.context.begin_transaction()
    }
}

impl ScopedMemoryBinding {
    pub fn identity(&self) -> BindingIdentity {
        self.binding.identity()
    }

    pub fn authority(&self) -> ProcessAuthorityId {
        self.authority.id
    }

    pub fn context_scope(&self) -> MemoryContextScope {
        MemoryContextScope {
            context: Arc::clone(&self.context),
        }
    }

    pub fn mechanism_snapshot(&self) -> Result<MechanismSnapshot, AllocationTransactionError> {
        self.binding
            .mechanism_snapshot()
            .map_err(AllocationTransactionError::Binding)
    }

    pub fn has_virtual_backing(&self) -> Result<bool, AllocationTransactionError> {
        self.binding
            .virtual_backing()
            .map(|capability| capability.is_some())
            .map_err(AllocationTransactionError::Binding)
    }

    pub fn has_shared_mapping(&self) -> Result<bool, AllocationTransactionError> {
        self.binding
            .shared_mapping()
            .map(|capability| capability.is_some())
            .map_err(AllocationTransactionError::Binding)
    }

    pub fn new_shared_physical_identity(
        &self,
    ) -> Result<SharedPhysicalIdentity, AllocationTransactionError> {
        Ok(SharedPhysicalIdentity {
            manager: self.manager.id,
            authority: self.authority.id,
            serial: self.manager.next()?,
        })
    }

    pub fn virtual_backing(
        &self,
    ) -> Result<Option<ScopedVirtualBacking>, AllocationTransactionError> {
        self.binding
            .virtual_backing()
            .map(|capability| {
                capability.map(|capability| ScopedVirtualBacking {
                    binding: self.identity(),
                    capability,
                })
            })
            .map_err(AllocationTransactionError::Binding)
    }

    pub fn allocate(
        &self,
        request: AllocationRequest,
        publication: AllocationPublication,
    ) -> Result<ManagedAllocation, AllocationTransactionError> {
        self.allocate_with(
            request,
            |context| context.allocate_owning(),
            |_| Ok(publication),
        )
    }

    /// Coordinate reserve → allocate → provider commit/map → publish.
    ///
    /// `allocate` and `commit` run with no manager, registry, or governor lock
    /// held. If either fails after allocation, rollback uses the Phase-4 owning
    /// release and reports quarantine rather than manufacturing success.
    pub fn allocate_with(
        &self,
        request: AllocationRequest,
        allocate: impl FnOnce(
            &ScopedAllocationContext<'_>,
        ) -> Result<OwningAllocation, AllocationStepError>,
        commit: impl FnOnce(&OwningAllocation) -> Result<AllocationPublication, AllocationStepError>,
    ) -> Result<ManagedAllocation, AllocationTransactionError> {
        let _context_transaction = self.context.begin_transaction()?;
        self.validate_request(&request)?;
        let (process_lease, authority_lease) = self.reserve(&request)?;
        let context = ScopedAllocationContext {
            binding: &self.binding,
            bytes: request.allocation_bytes,
            align: request.alignment,
        };
        let owner = match allocate(&context) {
            Ok(owner) => owner,
            Err(error) => {
                if error.retain_reservations {
                    if let Some(lease) = authority_lease {
                        std::mem::forget(lease);
                    }
                    if let Some(lease) = process_lease {
                        std::mem::forget(lease);
                    }
                    return Err(AllocationTransactionError::UnidentifiedOwnershipRetained {
                        stage: "allocate",
                        reason: error.reason,
                    });
                }
                return Err(AllocationTransactionError::Step {
                    stage: "allocate",
                    reason: error.reason,
                });
            }
        };
        if owner.identity().binding() != self.binding.identity() {
            return Err(self.rollback_after_failure(
                request,
                owner,
                process_lease,
                authority_lease,
                "allocate",
                Arc::from(
                    "the allocation callback returned ownership issued by another scoped binding",
                ),
            ));
        }
        if owner.len() != request.allocation_bytes || owner.alignment() != request.alignment {
            let reason = Arc::<str>::from(
                "the allocation callback returned an owner with a different size or alignment",
            );
            return Err(self.rollback_after_failure(
                request,
                owner,
                process_lease,
                authority_lease,
                "allocate",
                reason,
            ));
        }
        let publication = match commit(&owner) {
            Ok(publication) => publication,
            Err(error) => {
                return Err(self.rollback_after_failure(
                    request,
                    owner,
                    process_lease,
                    authority_lease,
                    "commit",
                    error.reason,
                ));
            }
        };
        if let Err(error) = self.validate_publication(&request, &publication) {
            return Err(self.rollback_after_failure(
                request,
                owner,
                process_lease,
                authority_lease,
                "publish",
                Arc::from(error.to_string()),
            ));
        }
        let mut request = request;
        let mut process_lease = process_lease;
        let mut authority_lease = authority_lease;
        if let Some(lease) = process_lease.as_mut() {
            lease.shrink(
                request
                    .process_reserve_bytes
                    .saturating_sub(publication.process_reserved_bytes),
            );
        }
        if let Some(lease) = authority_lease.as_mut() {
            lease.shrink(
                request
                    .authority_reserve_bytes
                    .saturating_sub(publication.charged_bytes),
            );
        }
        request.process_reserve_bytes = publication.process_reserved_bytes;
        request.authority_reserve_bytes = publication.charged_bytes;

        // Close the unlimited -> finite race between final validation and
        // publication. A transition either sees this allocation in the book or
        // completes first and makes the second validation enforce finite parent
        // coverage; it can never miss an unleased authority-managed charge.
        let quota_gate = self.manager.book.quota.lock(request.tier);
        if let Err(error) = self.validate_publication(&request, &publication) {
            drop(quota_gate);
            return Err(self.rollback_after_failure(
                request,
                owner,
                process_lease,
                authority_lease,
                "publish",
                Arc::from(error.to_string()),
            ));
        }
        let shared_physical = publication
            .shared_physical
            .unwrap_or(SharedPhysicalIdentity {
                manager: self.manager.id,
                authority: self.authority.id,
                serial: self.manager.next()?,
            });
        let charge = Arc::new(ChargeCell::new(ChargeState {
            identity: owner.identity(),
            authority: self.authority.id,
            authority_label: Arc::clone(&self.authority.label),
            memory_authority: self.authority.memory_authority,
            holder: request.holder.record.id,
            holder_label: Arc::clone(&request.holder.record.label),
            tier: request.tier,
            role: request.role,
            mode: request.charge_mode,
            state: ManagedAllocationState::Live,
            process_lease,
            authority_lease,
            charged_bytes: publication.charged_bytes,
            process_reserved_bytes: publication.process_reserved_bytes,
            physical_bytes: publication.physical_bytes,
            mapped_bytes: publication.mapped_bytes,
            unattributed_bytes: publication.unattributed_bytes,
            shared_physical,
        }));
        self.manager.book.publish(&charge);
        drop(quota_gate);
        let settlement = AllocationSettlementToken::new(Arc::clone(&self.manager.book), charge);
        Ok(ManagedAllocation {
            owner: Some(owner),
            settlement,
        })
    }

    fn validate_request(
        &self,
        request: &AllocationRequest,
    ) -> Result<(), AllocationTransactionError> {
        if request.holder.manager != self.manager.id {
            return Err(AllocationTransactionError::ForeignHandle("holder"));
        }
        if request.holder.record.authority != self.authority.id {
            return Err(AllocationTransactionError::HolderAuthorityMismatch {
                holder: request.holder.record.id,
                holder_authority: request.holder.record.authority,
                binding_authority: self.authority.id,
            });
        }
        if request.tier != self.identity().device().tier {
            return Err(AllocationTransactionError::TierMismatch {
                expected: self.identity().device().tier,
                actual: request.tier,
            });
        }
        if request.alignment == 0 || !request.alignment.is_power_of_two() {
            return Err(AllocationTransactionError::InvalidPublication(
                "allocation alignment must be a non-zero power of two",
            ));
        }
        let delegated_bytes = self
            .authority
            .process_delegations
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(&request.tier)
            .map(|lease| lease.bytes);
        if let (Some(delegated), Some(governor)) =
            (delegated_bytes, self.authority.governor.as_ref())
        {
            let used = governor.used(request.tier);
            let oversubscribed = governor.oversubscribed_bytes(request.tier);
            let local_extent = used.checked_add(governor.available(request.tier)).ok_or(
                MemoryError::InvalidRequest {
                    tier: request.tier.name(),
                    requested: request.authority_reserve_bytes,
                    reason: "the local authority extent overflows its byte counter",
                },
            )?;
            if oversubscribed != 0 || local_extent > delegated {
                return Err(AllocationTransactionError::StaleProcessDelegation {
                    authority: self.authority.id,
                    tier: request.tier,
                    delegated,
                    local_extent: local_extent.saturating_add(oversubscribed),
                });
            }
        }
        match request.charge_mode {
            AllocationChargeMode::Managed if self.authority.governor.is_none() => Err(
                AllocationTransactionError::MissingGovernor(self.authority.id),
            ),
            AllocationChargeMode::AuthorityManaged if self.authority.governor.is_none() => Err(
                AllocationTransactionError::MissingGovernor(self.authority.id),
            ),
            AllocationChargeMode::Managed | AllocationChargeMode::AuthorityManaged
                if request.authority_reserve_bytes != 0
                    && self.manager.book.quota.limit(request.tier) != u64::MAX
                    && delegated_bytes.is_none()
                    && request.process_reserve_bytes < request.authority_reserve_bytes =>
            {
                Err(AllocationTransactionError::InsufficientProcessCoverage {
                    authority: self.authority.id,
                    tier: request.tier,
                    required: request.authority_reserve_bytes,
                    reserved: request.process_reserve_bytes,
                })
            }
            AllocationChargeMode::Compatibility if request.authority_reserve_bytes != 0 => {
                Err(AllocationTransactionError::InvalidPublication(
                    "compatibility transactions cannot claim authority-charged bytes",
                ))
            }
            _ => Ok(()),
        }
    }

    fn validate_publication(
        &self,
        request: &AllocationRequest,
        publication: &AllocationPublication,
    ) -> Result<(), AllocationTransactionError> {
        if publication.charged_bytes > request.authority_reserve_bytes {
            return Err(AllocationTransactionError::InvalidPublication(
                "published authority charge exceeds the reserved maximum",
            ));
        }
        if publication.process_reserved_bytes > request.process_reserve_bytes {
            return Err(AllocationTransactionError::InvalidPublication(
                "published process charge exceeds the reserved maximum",
            ));
        }
        let delegated = self
            .authority
            .process_delegations
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .contains_key(&request.tier);
        if self.manager.book.quota.limit(request.tier) != u64::MAX
            && !delegated
            && publication.process_reserved_bytes < publication.charged_bytes
        {
            return Err(AllocationTransactionError::InsufficientProcessPublication {
                authority: self.authority.id,
                tier: request.tier,
                charged: publication.charged_bytes,
                process_reserved: publication.process_reserved_bytes,
            });
        }
        if request.charge_mode == AllocationChargeMode::Compatibility
            && publication.charged_bytes != 0
        {
            return Err(AllocationTransactionError::InvalidPublication(
                "compatibility transactions must publish zero authority charge",
            ));
        }
        if let Some(shared) = publication.shared_physical
            && (shared.manager != self.manager.id || shared.authority != self.authority.id)
        {
            return Err(AllocationTransactionError::InvalidPublication(
                "shared physical identity belongs to another manager or authority",
            ));
        }
        Ok(())
    }

    fn reserve(
        &self,
        request: &AllocationRequest,
    ) -> Result<(Option<ProcessQuotaLease>, Option<MemoryLease>), AllocationTransactionError> {
        let authority = self.manager.authority(self.authority.id)?;
        let mut pressured = false;
        loop {
            let authority_short = request.charge_mode == AllocationChargeMode::Managed
                && authority.governor.as_ref().is_some_and(|governor| {
                    governor.available(request.tier) < request.authority_reserve_bytes
                });
            let process_short =
                self.manager.book.quota.available(request.tier) < request.process_reserve_bytes;
            if !pressured && (authority_short || process_short) {
                self.manager.pressure(
                    authority.id,
                    request.tier,
                    request
                        .authority_reserve_bytes
                        .max(request.process_reserve_bytes),
                );
                pressured = true;
            }

            let process_lease = if request.process_reserve_bytes == 0 {
                None
            } else {
                match self.manager.book.quota.reserve(
                    request.tier,
                    request.process_reserve_bytes,
                    request.role,
                ) {
                    Ok(lease) => Some(lease),
                    Err(error) if !pressured && is_capacity_error(&error) => {
                        self.manager.pressure(
                            authority.id,
                            request.tier,
                            request.process_reserve_bytes,
                        );
                        pressured = true;
                        continue;
                    }
                    Err(error) => return Err(AllocationTransactionError::Memory(error)),
                }
            };

            let authority_lease = match request.charge_mode {
                AllocationChargeMode::Managed => {
                    let governor = authority
                        .governor
                        .as_ref()
                        .ok_or(AllocationTransactionError::MissingGovernor(authority.id))?;
                    match governor.reserve(
                        request.tier,
                        request.authority_reserve_bytes,
                        request.role,
                        request.holder.record.id,
                    ) {
                        Ok(lease) => Some(lease),
                        Err(error) if !pressured && is_capacity_error(&error) => {
                            drop(process_lease);
                            self.manager.pressure(
                                authority.id,
                                request.tier,
                                request.authority_reserve_bytes,
                            );
                            pressured = true;
                            continue;
                        }
                        Err(error) => {
                            drop(process_lease);
                            return Err(AllocationTransactionError::Memory(error));
                        }
                    }
                }
                AllocationChargeMode::AuthorityManaged | AllocationChargeMode::Compatibility => {
                    None
                }
            };
            return Ok((process_lease, authority_lease));
        }
    }

    fn rollback_after_failure(
        &self,
        request: AllocationRequest,
        owner: OwningAllocation,
        process_lease: Option<ProcessQuotaLease>,
        authority_lease: Option<MemoryLease>,
        stage: &'static str,
        reason: Arc<str>,
    ) -> AllocationTransactionError {
        let shared_physical = SharedPhysicalIdentity {
            manager: self.manager.id,
            authority: self.authority.id,
            serial: self.manager.next().unwrap_or(u64::MAX),
        };
        let charge = Arc::new(ChargeCell::new(ChargeState {
            identity: owner.identity(),
            authority: self.authority.id,
            authority_label: Arc::clone(&self.authority.label),
            memory_authority: self.authority.memory_authority,
            holder: request.holder.record.id,
            holder_label: Arc::clone(&request.holder.record.label),
            tier: request.tier,
            role: request.role,
            mode: request.charge_mode,
            state: ManagedAllocationState::Provisional,
            process_lease,
            authority_lease,
            charged_bytes: request.authority_reserve_bytes,
            process_reserved_bytes: request.process_reserve_bytes,
            // Virtual span length is not proof of committed physical residency.
            // The failure callback supplied no publication facts, so remain
            // unknown rather than equating address space with physical bytes.
            physical_bytes: None,
            mapped_bytes: None,
            unattributed_bytes: if request.charge_mode == AllocationChargeMode::Compatibility {
                owner.len() as u64
            } else {
                0
            },
            shared_physical,
        }));
        let managed = ManagedAllocation {
            owner: Some(owner),
            settlement: AllocationSettlementToken::new(Arc::clone(&self.manager.book), charge),
        };
        match managed.release_now() {
            Ok(outcome) if outcome.is_complete() => {
                AllocationTransactionError::Step { stage, reason }
            }
            Ok(outcome) => AllocationTransactionError::RollbackQuarantined {
                stage,
                reason,
                outcome,
            },
            Err(error) => AllocationTransactionError::RollbackPreparation {
                stage,
                reason,
                rollback: Arc::from(error.to_string()),
            },
        }
    }
}

fn is_capacity_error(error: &MemoryError) -> bool {
    matches!(
        error,
        MemoryError::TierExhausted { .. } | MemoryError::CapacityUnavailable { .. }
    )
}

/// Transaction request. Reservations are maxima; publication records exact axes.
#[derive(Clone, Debug)]
pub struct AllocationRequest {
    pub allocation_bytes: usize,
    pub alignment: usize,
    pub tier: Tier,
    pub role: MemoryRole,
    pub holder: RegisteredMemoryHolder,
    pub charge_mode: AllocationChargeMode,
    pub authority_reserve_bytes: u64,
    pub process_reserve_bytes: u64,
}

impl AllocationRequest {
    pub fn managed(
        allocation_bytes: usize,
        alignment: usize,
        tier: Tier,
        role: MemoryRole,
        holder: RegisteredMemoryHolder,
        reserve_bytes: u64,
    ) -> Self {
        Self {
            allocation_bytes,
            alignment,
            tier,
            role,
            holder,
            charge_mode: AllocationChargeMode::Managed,
            authority_reserve_bytes: reserve_bytes,
            process_reserve_bytes: reserve_bytes,
        }
    }

    pub fn authority_managed(
        allocation_bytes: usize,
        alignment: usize,
        tier: Tier,
        role: MemoryRole,
        holder: RegisteredMemoryHolder,
        reserve_bytes: u64,
    ) -> Self {
        Self {
            allocation_bytes,
            alignment,
            tier,
            role,
            holder,
            charge_mode: AllocationChargeMode::AuthorityManaged,
            authority_reserve_bytes: reserve_bytes,
            // The delegated authority owns the physical charge lifetime (for
            // example a VMM physical-handle pool may outlive one allocation).
            // A transaction-scoped process lease would refund too early when the
            // authority pools those bytes. Callers may opt into a process
            // reservation only when they can prove the lifetimes coincide.
            process_reserve_bytes: 0,
        }
    }

    pub fn compatibility(
        allocation_bytes: usize,
        alignment: usize,
        tier: Tier,
        role: MemoryRole,
        holder: RegisteredMemoryHolder,
    ) -> Self {
        Self {
            allocation_bytes,
            alignment,
            tier,
            role,
            holder,
            charge_mode: AllocationChargeMode::Compatibility,
            authority_reserve_bytes: 0,
            process_reserve_bytes: 0,
        }
    }

    pub fn with_process_reservation(mut self, bytes: u64) -> Self {
        self.process_reserve_bytes = bytes;
        self
    }
}

/// Exact facts published after provider commit/map succeeds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AllocationPublication {
    pub charged_bytes: u64,
    pub process_reserved_bytes: u64,
    pub physical_bytes: Option<u64>,
    pub mapped_bytes: Option<u64>,
    pub unattributed_bytes: u64,
    pub shared_physical: Option<SharedPhysicalIdentity>,
}

impl AllocationPublication {
    pub const fn exclusive(charged_bytes: u64, physical_bytes: u64, mapped_bytes: u64) -> Self {
        Self {
            charged_bytes,
            process_reserved_bytes: charged_bytes,
            physical_bytes: Some(physical_bytes),
            mapped_bytes: Some(mapped_bytes),
            unattributed_bytes: 0,
            shared_physical: None,
        }
    }

    pub const fn compatibility(physical_bytes: u64, mapped_bytes: u64) -> Self {
        Self {
            charged_bytes: 0,
            process_reserved_bytes: 0,
            physical_bytes: Some(physical_bytes),
            mapped_bytes: Some(mapped_bytes),
            unattributed_bytes: physical_bytes,
            shared_physical: None,
        }
    }

    pub const fn with_shared_physical(mut self, shared: SharedPhysicalIdentity) -> Self {
        self.shared_physical = Some(shared);
        self
    }
}

/// Error returned by an allocator/provider transaction callback.
#[derive(Clone, Debug)]
pub struct AllocationStepError {
    reason: Arc<str>,
    retain_reservations: bool,
}

impl AllocationStepError {
    pub fn new(reason: impl Into<Arc<str>>) -> Self {
        Self {
            reason: reason.into(),
            retain_reservations: false,
        }
    }

    /// Report physical ownership that could not receive an allocation identity.
    ///
    /// The manager deliberately leaks provisional reservations rather than
    /// manufacturing a refund for bytes the provider still owns.
    pub fn retained(reason: impl Into<Arc<str>>) -> Self {
        Self {
            reason: reason.into(),
            retain_reservations: true,
        }
    }
}

impl fmt::Display for AllocationStepError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.reason)
    }
}

impl std::error::Error for AllocationStepError {}

impl From<BindingError> for AllocationStepError {
    fn from(error: BindingError) -> Self {
        Self::new(error.to_string())
    }
}

impl From<MemoryError> for AllocationStepError {
    fn from(error: MemoryError) -> Self {
        Self::new(error.to_string())
    }
}

#[derive(Debug, thiserror::Error)]
pub enum AllocationTransactionError {
    #[error("process memory manager identity space is exhausted")]
    IdentityExhausted,
    #[error("the {0} belongs to another process memory manager")]
    ForeignHandle(&'static str),
    #[error("the {0} is not registered with this process memory manager")]
    Unregistered(&'static str),
    #[error("provider context {0:?} is terminal and accepts no new memory transaction")]
    TerminatedContext(ProviderContextIdentity),
    #[error("provider context {context:?} is {lifecycle}, not device-lost")]
    ContextNotLost {
        context: ProviderContextIdentity,
        lifecycle: &'static str,
    },
    #[error(
        "device {device:?} was lost while provider registration expected generation \
         {expected_generation} (current {actual_generation})"
    )]
    DeviceRegistrationLost {
        device: DeviceKey,
        expected_generation: u64,
        actual_generation: u64,
    },
    #[error("{0} still owns a manager-published allocation")]
    HolderInUse(HolderId),
    #[error("authority is registered for {actual:?}, not {expected:?}")]
    AuthorityDeviceMismatch {
        expected: DeviceKey,
        actual: DeviceKey,
    },
    #[error(
        "{holder} belongs to authority {holder_authority:?}, but the binding uses \
         {binding_authority:?}"
    )]
    HolderAuthorityMismatch {
        holder: HolderId,
        holder_authority: ProcessAuthorityId,
        binding_authority: ProcessAuthorityId,
    },
    #[error("binding serves {expected:?}, but the transaction requested {actual:?}")]
    TierMismatch { expected: Tier, actual: Tier },
    #[error("authority {0:?} has no governor; use compatibility accounting")]
    MissingGovernor(ProcessAuthorityId),
    #[error(
        "allocation for {authority:?} on {tier:?} reserves {reserved} process bytes for \
         {required} authority bytes without a fixed authority delegation"
    )]
    InsufficientProcessCoverage {
        authority: ProcessAuthorityId,
        tier: Tier,
        required: u64,
        reserved: u64,
    },
    #[error(
        "allocation for {authority:?} on {tier:?} publishes {charged} charged bytes but only \
         {process_reserved} process bytes without a fixed authority delegation"
    )]
    InsufficientProcessPublication {
        authority: ProcessAuthorityId,
        tier: Tier,
        charged: u64,
        process_reserved: u64,
    },
    #[error(
        "cannot delegate {delegated} bytes of {tier:?} process quota to {authority:?}: its local \
         capacity is {local_capacity} bytes"
    )]
    DelegationTooSmall {
        authority: ProcessAuthorityId,
        tier: Tier,
        delegated: u64,
        local_capacity: u64,
    },
    #[error("{authority:?} already has a process-quota delegation for {tier:?}")]
    DelegationAlreadyExists {
        authority: ProcessAuthorityId,
        tier: Tier,
    },
    #[error(
        "authority {authority:?} on {tier:?} has local extent {local_extent} bytes beyond its \
         {delegated}-byte process delegation; resize the delegation before granting more memory"
    )]
    StaleProcessDelegation {
        authority: ProcessAuthorityId,
        tier: Tier,
        delegated: u64,
        local_extent: u64,
    },
    #[error("invalid allocation publication: {0}")]
    InvalidPublication(&'static str),
    #[error("{stage} failed before publication: {reason}")]
    Step {
        stage: &'static str,
        reason: Arc<str>,
    },
    #[error(
        "{stage} failed before an allocation identity could be published ({reason}); provisional \
         reservations remain charged because the provider retained physical ownership"
    )]
    UnidentifiedOwnershipRetained {
        stage: &'static str,
        reason: Arc<str>,
    },
    #[error(
        "{stage} failed ({reason}); rollback retained ownership in state {}",
        outcome.state()
    )]
    RollbackQuarantined {
        stage: &'static str,
        reason: Arc<str>,
        outcome: AllocationReleaseOutcome,
    },
    #[error("{stage} failed ({reason}); rollback could not be prepared: {rollback}")]
    RollbackPreparation {
        stage: &'static str,
        reason: Arc<str>,
        rollback: Arc<str>,
    },
    #[error(transparent)]
    Binding(#[from] BindingError),
    #[error(transparent)]
    Memory(#[from] MemoryError),
}

/// Restricted allocation surface supplied only inside a manager transaction.
pub struct ScopedAllocationContext<'a> {
    binding: &'a MemoryBinding,
    bytes: usize,
    align: usize,
}

impl ScopedAllocationContext<'_> {
    pub fn allocate_owning(&self) -> Result<OwningAllocation, AllocationStepError> {
        self.binding
            .allocate_owning(self.bytes, self.align)
            .map_err(Into::into)
    }

    pub fn allocate_committed(
        &self,
        committed_ranges: &[std::ops::Range<usize>],
    ) -> Result<OwningAllocation, AllocationStepError> {
        let capability = self
            .binding
            .virtual_backing()?
            .ok_or_else(|| AllocationStepError::new("selected mechanism has no virtual backing"))?;
        capability
            .allocate_committed(self.bytes, self.align, committed_ranges)
            .map(OwningAllocation::new)
            .map_err(Into::into)
    }

    /// Adopt a provider-specific allocation from this binding's own mechanism.
    ///
    /// # Safety
    ///
    /// The same requirements as [`MemoryBinding::adopt_allocation`] apply.
    pub unsafe fn adopt_allocation(
        &self,
        ptr: NonNull<u8>,
    ) -> Result<OwningAllocation, AllocationStepError> {
        // SAFETY: delegated to this method's contract.
        unsafe {
            self.binding
                .adopt_allocation(ptr, self.bytes, self.align)
                .map_err(Into::into)
        }
    }
}

/// Non-allocating virtual-backing operations scoped to one binding.
#[derive(Clone, Debug)]
pub struct ScopedVirtualBacking {
    binding: BindingIdentity,
    capability: BoundVirtualBacking,
}

impl ScopedVirtualBacking {
    fn validate_owner(&self, owner: &OwningAllocation) -> Result<(), AllocationTransactionError> {
        if owner.identity().binding() != self.binding {
            return Err(AllocationTransactionError::InvalidPublication(
                "allocation belongs to another scoped binding",
            ));
        }
        Ok(())
    }

    pub fn commit_allocation_range(
        &self,
        owner: &OwningAllocation,
        offset: usize,
        bytes: usize,
    ) -> Result<(), AllocationTransactionError> {
        self.validate_owner(owner)?;
        self.capability
            .commit_allocation_range(owner.bound(), offset, bytes)
            .map_err(AllocationTransactionError::Binding)
    }

    pub fn commit_allocation_ranges(
        &self,
        ranges: &[(&OwningAllocation, usize, usize)],
    ) -> Result<(), AllocationTransactionError> {
        for (owner, _, _) in ranges {
            self.validate_owner(owner)?;
        }
        let raw = ranges
            .iter()
            .map(|(owner, offset, bytes)| (owner.bound(), *offset, *bytes))
            .collect::<Vec<_>>();
        self.capability
            .commit_allocation_ranges(&raw)
            .map_err(AllocationTransactionError::Binding)
    }

    pub fn mapped_bytes_for_allocation(
        &self,
        bytes: usize,
        align: usize,
    ) -> Result<u64, AllocationTransactionError> {
        self.capability
            .mapped_bytes_for_allocation(bytes, align)
            .map_err(AllocationTransactionError::Binding)
    }

    pub fn allocation_committed_bytes(
        &self,
        owner: &OwningAllocation,
    ) -> Result<usize, AllocationTransactionError> {
        self.validate_owner(owner)?;
        self.capability
            .allocation_committed_bytes(owner.bound())
            .map_err(AllocationTransactionError::Binding)
    }

    pub fn decommit_allocation_range(
        &self,
        owner: &OwningAllocation,
        offset: usize,
        bytes: usize,
    ) -> Result<u64, AllocationTransactionError> {
        self.validate_owner(owner)?;
        self.capability
            .decommit_allocation_range(owner.bound(), offset, bytes)
            .map_err(AllocationTransactionError::Binding)
    }
}

/// Allocation state reported by process snapshots.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ManagedAllocationState {
    Provisional,
    Live,
    Queued,
    Settling,
    Quarantined,
    DeviceLost,
    Released,
    ContextTerminated,
}

#[derive(Debug)]
struct ChargeState {
    identity: AllocationIdentity,
    authority: ProcessAuthorityId,
    authority_label: Arc<str>,
    memory_authority: Option<crate::MemoryAuthorityId>,
    holder: HolderId,
    holder_label: Arc<str>,
    tier: Tier,
    role: MemoryRole,
    mode: AllocationChargeMode,
    state: ManagedAllocationState,
    process_lease: Option<ProcessQuotaLease>,
    authority_lease: Option<MemoryLease>,
    charged_bytes: u64,
    process_reserved_bytes: u64,
    physical_bytes: Option<u64>,
    mapped_bytes: Option<u64>,
    unattributed_bytes: u64,
    shared_physical: SharedPhysicalIdentity,
}

#[derive(Debug)]
struct ChargeCell {
    state: Mutex<ChargeState>,
    wake: Condvar,
    settlement_tokens: AtomicUsize,
    terminal: AtomicBool,
}

impl ChargeCell {
    fn new(state: ChargeState) -> Self {
        Self {
            state: Mutex::new(state),
            wake: Condvar::new(),
            settlement_tokens: AtomicUsize::new(0),
            terminal: AtomicBool::new(false),
        }
    }

    fn lock(&self) -> MutexGuard<'_, ChargeState> {
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    fn mark_queued(&self) {
        let mut state = self.lock();
        if state.state == ManagedAllocationState::Live {
            state.state = ManagedAllocationState::Queued;
        }
    }

    fn mark_settlement_abandoned(&self) {
        let mut state = self.lock();
        if !matches!(
            state.state,
            ManagedAllocationState::Released
                | ManagedAllocationState::ContextTerminated
                | ManagedAllocationState::Quarantined
                | ManagedAllocationState::DeviceLost
        ) {
            state.state = ManagedAllocationState::Quarantined;
            self.wake.notify_all();
        }
    }

    fn settle(&self, outcome: &AllocationReleaseOutcome, terminal: ManagedAllocationState) -> bool {
        match outcome {
            AllocationReleaseOutcome::Complete { accounting } => {
                self.settle_retained(0, accounting.unmapped_bytes, terminal, true)
            }
            AllocationReleaseOutcome::Quarantined {
                accounting,
                residual,
            } => self.settle_retained(
                residual.retained_bytes,
                accounting.unmapped_bytes,
                terminal,
                false,
            ),
            AllocationReleaseOutcome::Failed { .. } => self.settle_retained(0, 0, terminal, false),
        }
    }

    fn settle_retained(
        &self,
        retained_bytes: u64,
        unmapped_bytes: u64,
        terminal: ManagedAllocationState,
        refund_charges: bool,
    ) -> bool {
        let (
            mut process_lease,
            mut authority_lease,
            process_refund,
            authority_refund,
            retained_process,
            retained_authority,
            prior_mapped,
        ) = {
            let mut state = self.lock();
            while state.state == ManagedAllocationState::Settling {
                state = self
                    .wake
                    .wait(state)
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
            }
            if matches!(
                state.state,
                ManagedAllocationState::Released | ManagedAllocationState::ContextTerminated
            ) {
                return false;
            }
            state.state = ManagedAllocationState::Settling;
            let retained_process = if refund_charges {
                state.process_reserved_bytes.min(retained_bytes)
            } else {
                state.process_reserved_bytes
            };
            let retained_authority = if refund_charges {
                state.charged_bytes.min(retained_bytes)
            } else {
                state.charged_bytes
            };
            (
                state.process_lease.take(),
                state.authority_lease.take(),
                state
                    .process_reserved_bytes
                    .saturating_sub(retained_process),
                state.charged_bytes.saturating_sub(retained_authority),
                retained_process,
                retained_authority,
                state.mapped_bytes,
            )
        };
        if let Some(lease) = process_lease.as_mut() {
            lease.shrink(process_refund);
        }
        if let Some(lease) = authority_lease.as_mut() {
            lease.shrink(authority_refund);
        }
        let mut state = self.lock();
        state.process_lease = (retained_process != 0).then_some(process_lease).flatten();
        state.authority_lease = (retained_authority != 0)
            .then_some(authority_lease)
            .flatten();
        state.process_reserved_bytes = retained_process;
        state.charged_bytes = retained_authority;
        if refund_charges || retained_bytes != 0 {
            state.physical_bytes = state.physical_bytes.map(|known| known.min(retained_bytes));
        }
        state.mapped_bytes = prior_mapped.map(|bytes| bytes.saturating_sub(unmapped_bytes));
        if refund_charges || retained_bytes != 0 {
            state.unattributed_bytes = state.unattributed_bytes.min(retained_bytes);
        }
        state.state = terminal;
        if matches!(
            terminal,
            ManagedAllocationState::Released | ManagedAllocationState::ContextTerminated
        ) {
            self.terminal.store(true, Ordering::Release);
        }
        let retained = retained_bytes != 0 || retained_process != 0 || retained_authority != 0;
        self.wake.notify_all();
        retained
    }

    fn snapshot(&self) -> ManagedAllocationSnapshot {
        let state = self.lock();
        ManagedAllocationSnapshot {
            identity: state.identity,
            binding: state.identity.binding(),
            authority: state.authority,
            authority_label: Arc::clone(&state.authority_label),
            memory_authority: state.memory_authority,
            holder: state.holder,
            holder_label: Arc::clone(&state.holder_label),
            tier: state.tier,
            role: state.role,
            charge_mode: state.mode,
            state: state.state,
            charged_bytes: state.charged_bytes,
            process_reserved_bytes: state.process_reserved_bytes,
            physical_bytes: state.physical_bytes,
            mapped_bytes: state.mapped_bytes,
            unattributed_bytes: state.unattributed_bytes,
            shared_physical: state.shared_physical,
        }
    }

    fn settlement_status(state: ManagedAllocationState) -> AllocationSettlementStatus {
        match state {
            ManagedAllocationState::Released | ManagedAllocationState::ContextTerminated => {
                AllocationSettlementStatus::Released
            }
            ManagedAllocationState::Quarantined | ManagedAllocationState::DeviceLost => {
                AllocationSettlementStatus::Retained(state)
            }
            ManagedAllocationState::Provisional
            | ManagedAllocationState::Live
            | ManagedAllocationState::Queued
            | ManagedAllocationState::Settling => AllocationSettlementStatus::Pending,
        }
    }

    fn wait_for_settlement(&self, timeout: std::time::Duration) -> AllocationSettlementStatus {
        let deadline = std::time::Instant::now() + timeout;
        let mut state = self.lock();
        loop {
            let status = Self::settlement_status(state.state);
            if status != AllocationSettlementStatus::Pending {
                return status;
            }
            let now = std::time::Instant::now();
            if now >= deadline {
                return AllocationSettlementStatus::Pending;
            }
            let remaining = deadline.saturating_duration_since(now);
            let (next, wait) = self
                .wake
                .wait_timeout(state, remaining)
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            state = next;
            if wait.timed_out() {
                return Self::settlement_status(state.state);
            }
        }
    }

    fn wait_until_not_settling(&self) {
        let mut state = self.lock();
        while state.state == ManagedAllocationState::Settling {
            state = self
                .wake
                .wait(state)
                .unwrap_or_else(|poisoned| poisoned.into_inner());
        }
    }
}

impl Drop for ChargeCell {
    fn drop(&mut self) {
        let state = self
            .state
            .get_mut()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if !matches!(
            state.state,
            ManagedAllocationState::Released | ManagedAllocationState::ContextTerminated
        ) {
            // Fail closed. If every manager/queue/token disappeared without a
            // terminal physical outcome, refunding would manufacture capacity
            // that may still be owned. Leaking the accounting token is noisy but
            // truthful and cannot permit reuse of retained bytes.
            if let Some(lease) = state.authority_lease.take() {
                std::mem::forget(lease);
            }
            if let Some(lease) = state.process_lease.take() {
                std::mem::forget(lease);
            }
        }
    }
}

#[derive(Default)]
struct AllocationBookState {
    live: HashMap<AllocationIdentity, Weak<ChargeCell>>,
    quarantined: HashMap<AllocationIdentity, Arc<ChargeCell>>,
}

struct SettlementBook {
    manager: u64,
    quota: Arc<ProcessQuota>,
    state: Mutex<AllocationBookState>,
}

impl fmt::Debug for SettlementBook {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SettlementBook")
            .field("manager", &self.manager)
            .finish_non_exhaustive()
    }
}

impl SettlementBook {
    fn new(manager: u64, quota: Arc<ProcessQuota>) -> Self {
        Self {
            manager,
            quota,
            state: Mutex::new(AllocationBookState::default()),
        }
    }

    fn lock(&self) -> MutexGuard<'_, AllocationBookState> {
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    fn set_process_limit(
        &self,
        tier: Tier,
        bytes: u64,
        delegated: &HashSet<ProcessAuthorityId>,
    ) -> Result<(), MemoryError> {
        loop {
            let quota_gate = self.quota.lock(tier);
            let mut book = self.lock();
            book.live.retain(|_, charge| charge.strong_count() != 0);
            let mut charges = book
                .live
                .values()
                .filter_map(Weak::upgrade)
                .chain(book.quarantined.values().cloned())
                .collect::<Vec<_>>();
            charges.sort_by_key(|charge| format!("{:?}", charge.snapshot().identity));
            charges.dedup_by_key(|charge| charge.snapshot().identity);
            let mut states = charges
                .iter()
                .map(|charge| charge.lock())
                .collect::<Vec<_>>();

            if let Some(settling) = states
                .iter()
                .position(|state| state.state == ManagedAllocationState::Settling)
            {
                drop(states);
                drop(book);
                drop(quota_gate);
                charges[settling].wait_until_not_settling();
                continue;
            }

            let mut additions = Vec::with_capacity(states.len());
            let mut additional = 0_u64;
            for state in &states {
                let covered_by_delegation = delegated.contains(&state.authority);
                let terminal = matches!(
                    state.state,
                    ManagedAllocationState::Released | ManagedAllocationState::ContextTerminated
                );
                let leased = state.process_lease.as_ref().map_or(0, |lease| lease.bytes);
                let uncovered = if state.tier == tier
                    && state.mode == AllocationChargeMode::AuthorityManaged
                    && !covered_by_delegation
                    && !terminal
                {
                    state.charged_bytes.saturating_sub(leased)
                } else {
                    0
                };
                additional =
                    additional
                        .checked_add(uncovered)
                        .ok_or(MemoryError::InvalidRequest {
                            tier: tier.name(),
                            requested: uncovered,
                            reason: "live authority-managed process coverage overflows its byte counter",
                        })?;
                let covered = leased
                    .checked_add(uncovered)
                    .ok_or(MemoryError::InvalidRequest {
                        tier: tier.name(),
                        requested: uncovered,
                        reason: "the allocation process lease overflows its byte counter",
                    })?;
                let reserved = state.process_reserved_bytes.checked_add(uncovered).ok_or(
                    MemoryError::InvalidRequest {
                        tier: tier.name(),
                        requested: uncovered,
                        reason: "the allocation process reservation overflows its byte counter",
                    },
                )?;
                additions.push((uncovered, covered, reserved));
            }
            let index = tier.index();
            let used = self.quota.used[index].load(Ordering::Acquire);
            let next = used
                .checked_add(additional)
                .ok_or(MemoryError::InvalidRequest {
                    tier: tier.name(),
                    requested: additional,
                    reason: "the process memory reservation overflows its byte counter",
                })?;
            if next > bytes {
                return Err(MemoryError::TierExhausted {
                    tier: tier.name(),
                    requested: 0,
                    used: next,
                    limit: bytes,
                    available: 0,
                    role: MemoryRole::Activation,
                });
            }

            for (state, (addition, covered, reserved)) in states.iter_mut().zip(additions) {
                if addition == 0 {
                    continue;
                }
                if let Some(lease) = state.process_lease.as_mut() {
                    lease.bytes = covered;
                } else {
                    state.process_lease = Some(ProcessQuotaLease {
                        quota: Arc::clone(&self.quota),
                        tier,
                        bytes: addition,
                    });
                }
                state.process_reserved_bytes = reserved;
            }
            self.quota.used[index].store(next, Ordering::Release);
            self.quota.limits[index].store(bytes, Ordering::Release);
            return Ok(());
        }
    }

    fn publish(&self, charge: &Arc<ChargeCell>) {
        self.lock()
            .live
            .insert(charge.snapshot().identity, Arc::downgrade(charge));
    }

    fn settle(&self, charge: &Arc<ChargeCell>, outcome: &AllocationReleaseOutcome) {
        let identity = charge.snapshot().identity;
        let terminal = match outcome.state() {
            AllocationReleaseState::Released => ManagedAllocationState::Released,
            AllocationReleaseState::DeviceLost => ManagedAllocationState::DeviceLost,
            _ => ManagedAllocationState::Quarantined,
        };
        let retained = charge.settle(outcome, terminal);
        let mut state = self.lock();
        state.live.remove(&identity);
        if retained {
            state.quarantined.insert(identity, Arc::clone(charge));
        } else {
            state.quarantined.remove(&identity);
        }
    }

    fn retain_unsettled(&self, charge: &Arc<ChargeCell>) {
        if charge.terminal.load(Ordering::Acquire) {
            return;
        }
        let identity = charge.snapshot().identity;
        let mut state = self.lock();
        // Re-check under the book lock. Context termination publishes the
        // terminal atomic before acquiring this lock to remove the entry, so
        // either this declines insertion or termination removes what was
        // inserted first. No book->charge lock inversion is introduced.
        if charge.terminal.load(Ordering::Acquire) {
            return;
        }
        state.live.remove(&identity);
        state.quarantined.insert(identity, Arc::clone(charge));
    }

    fn context_terminated(&self, charge: &Arc<ChargeCell>) {
        let identity = charge.snapshot().identity;
        charge.settle_retained(0, 0, ManagedAllocationState::ContextTerminated, true);
        let mut state = self.lock();
        state.live.remove(&identity);
        state.quarantined.remove(&identity);
    }

    fn charges_for_context(&self, context: ProviderContextIdentity) -> Vec<Arc<ChargeCell>> {
        let mut state = self.lock();
        state.live.retain(|_, charge| charge.strong_count() != 0);
        let mut charges = state
            .live
            .values()
            .filter_map(Weak::upgrade)
            .filter(|charge| charge.snapshot().binding.provider_context() == context)
            .collect::<Vec<_>>();
        charges.extend(
            state
                .quarantined
                .values()
                .filter(|charge| charge.snapshot().binding.provider_context() == context)
                .cloned(),
        );
        let mut seen = HashSet::new();
        charges.retain(|charge| seen.insert(charge.snapshot().identity));
        charges
    }

    fn snapshots(&self) -> Vec<ManagedAllocationSnapshot> {
        let mut state = self.lock();
        state.live.retain(|_, charge| charge.strong_count() != 0);
        let mut allocations = state
            .live
            .values()
            .filter_map(Weak::upgrade)
            .map(|charge| charge.snapshot())
            .collect::<Vec<_>>();
        allocations.extend(state.quarantined.values().map(|charge| charge.snapshot()));
        allocations.sort_by_key(|allocation| format!("{:?}", allocation.identity));
        allocations.dedup_by_key(|allocation| allocation.identity);
        allocations
    }
}

/// One-shot settlement handle retained through queued and quarantined release.
#[derive(Debug)]
pub struct AllocationSettlementToken {
    book: Arc<SettlementBook>,
    charge: Arc<ChargeCell>,
}

impl AllocationSettlementToken {
    fn new(book: Arc<SettlementBook>, charge: Arc<ChargeCell>) -> Self {
        charge.settlement_tokens.fetch_add(1, Ordering::AcqRel);
        Self { book, charge }
    }

    pub fn identity(&self) -> AllocationIdentity {
        self.charge.snapshot().identity
    }

    pub fn mark_queued(&self) {
        self.charge.mark_queued();
    }

    fn settle_verified(&self, outcome: &AllocationReleaseOutcome) {
        self.book.settle(&self.charge, outcome);
    }

    /// Settle from the exact outcome produced by this token's paired request.
    ///
    /// # Safety
    ///
    /// `outcome` must come from the [`PreparedAllocationRelease`] returned with
    /// this token by [`ManagedPreparedRelease::into_parts`]. A fabricated outcome
    /// can falsely refund physical ownership.
    #[doc(hidden)]
    pub unsafe fn settle(&self, outcome: &AllocationReleaseOutcome) {
        self.settle_verified(outcome);
    }
}

impl Clone for AllocationSettlementToken {
    fn clone(&self) -> Self {
        Self::new(Arc::clone(&self.book), Arc::clone(&self.charge))
    }
}

impl Drop for AllocationSettlementToken {
    fn drop(&mut self) {
        let previous = self.charge.settlement_tokens.fetch_sub(1, Ordering::AcqRel);
        debug_assert_ne!(previous, 0, "settlement token count underflow");
        if previous == 1 {
            let snapshot = self.charge.snapshot();
            if !matches!(
                snapshot.state,
                ManagedAllocationState::Released | ManagedAllocationState::ContextTerminated
            ) {
                self.charge.mark_settlement_abandoned();
                self.book.retain_unsettled(&self.charge);
            }
        }
    }
}

/// Owning allocation plus its exact-once accounting settlement.
#[derive(Debug)]
pub struct ManagedAllocation {
    owner: Option<OwningAllocation>,
    settlement: AllocationSettlementToken,
}

/// Result of waiting for one allocation's structured release settlement.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AllocationSettlementStatus {
    Pending,
    Released,
    Retained(ManagedAllocationState),
}

/// Cloneable, allocation-specific settlement waiter.
#[derive(Clone, Debug)]
pub struct AllocationSettlementWait {
    charge: Arc<ChargeCell>,
}

impl AllocationSettlementWait {
    pub fn identity(&self) -> AllocationIdentity {
        self.charge.snapshot().identity
    }

    pub fn status(&self) -> AllocationSettlementStatus {
        ChargeCell::settlement_status(self.charge.snapshot().state)
    }

    pub fn wait(&self, timeout: std::time::Duration) -> AllocationSettlementStatus {
        self.charge.wait_for_settlement(timeout)
    }
}

impl ManagedAllocation {
    fn owner(&self) -> &OwningAllocation {
        self.owner
            .as_ref()
            .expect("managed allocation owns its allocation until consumed")
    }

    pub fn identity(&self) -> AllocationIdentity {
        self.owner().identity()
    }

    pub fn as_ptr(&self) -> NonNull<u8> {
        self.owner().as_ptr()
    }

    pub fn len(&self) -> usize {
        self.owner().len()
    }

    pub fn is_empty(&self) -> bool {
        self.owner().is_empty()
    }

    pub fn alignment(&self) -> usize {
        self.owner().alignment()
    }

    pub fn owner_ref(&self) -> &OwningAllocation {
        self.owner()
    }

    /// Allocation-specific settlement observation for replacement admission.
    pub fn settlement_wait(&self) -> AllocationSettlementWait {
        AllocationSettlementWait {
            charge: Arc::clone(&self.settlement.charge),
        }
    }

    pub fn view(
        &self,
        offset: usize,
        bytes: usize,
    ) -> Result<onnx_runtime_memory_api::OwnedView, BindingError> {
        self.owner().view(offset, bytes)
    }

    pub fn prepare_release(mut self) -> Result<ManagedPreparedRelease, ManagedReleaseError> {
        let owner = self
            .owner
            .take()
            .expect("managed allocation owns its allocation until consumed");
        match owner.prepare_release() {
            Ok(request) => {
                self.settlement.mark_queued();
                Ok(ManagedPreparedRelease {
                    request: Some(request),
                    settlement: Some(self.settlement.clone()),
                })
            }
            Err(error) => {
                let (error, owner) = error.into_parts();
                self.owner = Some(owner);
                Err(ManagedReleaseError {
                    error,
                    allocation: Box::new(self),
                })
            }
        }
    }

    pub fn release_now(self) -> Result<AllocationReleaseOutcome, ManagedReleaseError> {
        let prepared = self.prepare_release()?;
        Ok(prepared.execute())
    }
}

impl Drop for ManagedAllocation {
    fn drop(&mut self) {
        let Some(owner) = self.owner.take() else {
            return;
        };
        let bytes = owner.len() as u64;
        let address = owner.as_ptr().as_ptr() as usize;
        let align = owner.alignment();
        match owner.prepare_release() {
            Ok(request) => {
                let outcome = request.quarantine(QuarantineReason::OwnerDropped);
                self.settlement.settle_verified(&outcome);
            }
            Err(error) => {
                let (binding_error, owner) = error.into_parts();
                let state = if matches!(binding_error, BindingError::DeviceLost { .. }) {
                    AllocationReleaseState::DeviceLost
                } else {
                    AllocationReleaseState::Quarantined
                };
                let outcome = AllocationReleaseOutcome::quarantined(
                    ReleaseAccounting::new(bytes, 0),
                    ResidualOwnership {
                        state,
                        reason: if state == AllocationReleaseState::DeviceLost {
                            QuarantineReason::DeviceLost
                        } else {
                            QuarantineReason::OwnerDropped
                        },
                        retained_bytes: bytes,
                        address,
                        align,
                    },
                );
                self.settlement.settle_verified(&outcome);
                drop(owner);
            }
        }
    }
}

/// Prepared Phase-4 release paired with the manager settlement token.
#[derive(Debug)]
pub struct ManagedPreparedRelease {
    request: Option<PreparedAllocationRelease>,
    settlement: Option<AllocationSettlementToken>,
}

impl ManagedPreparedRelease {
    pub fn identity(&self) -> AllocationIdentity {
        self.request
            .as_ref()
            .expect("prepared managed release owns its request")
            .identity()
    }

    pub fn execute(mut self) -> AllocationReleaseOutcome {
        let request = self
            .request
            .take()
            .expect("prepared managed release owns its request");
        let outcome = request.execute();
        if let Some(settlement) = self.settlement.take() {
            settlement.settle_verified(&outcome);
        }
        outcome
    }

    pub fn quarantine(mut self, reason: QuarantineReason) -> AllocationReleaseOutcome {
        let request = self
            .request
            .take()
            .expect("prepared managed release owns its request");
        let outcome = request.quarantine(reason);
        if let Some(settlement) = self.settlement.take() {
            settlement.settle_verified(&outcome);
        }
        outcome
    }

    /// Split this pair for a provider-owned deferred queue.
    ///
    /// # Safety
    ///
    /// The caller must keep the pair associated through every terminal path and
    /// settle only from the returned request's exact structured outcome.
    #[doc(hidden)]
    pub unsafe fn into_parts(mut self) -> (PreparedAllocationRelease, AllocationSettlementToken) {
        (
            self.request
                .take()
                .expect("prepared managed release owns its request"),
            self.settlement
                .take()
                .expect("prepared managed release owns its settlement token"),
        )
    }
}

impl Drop for ManagedPreparedRelease {
    fn drop(&mut self) {
        let (Some(request), Some(settlement)) = (self.request.take(), self.settlement.take())
        else {
            return;
        };
        let outcome = request.quarantine(QuarantineReason::AbandonedRequest);
        settlement.settle_verified(&outcome);
    }
}

#[derive(Debug)]
pub struct ManagedReleaseError {
    error: BindingError,
    allocation: Box<ManagedAllocation>,
}

impl ManagedReleaseError {
    pub fn error(&self) -> &BindingError {
        &self.error
    }

    pub fn allocation(&self) -> &ManagedAllocation {
        &self.allocation
    }

    pub fn into_parts(self) -> (BindingError, ManagedAllocation) {
        (self.error, *self.allocation)
    }
}

impl fmt::Display for ManagedReleaseError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.error, formatter)
    }
}

impl std::error::Error for ManagedReleaseError {}

/// Per-allocation observability. Accounting axes are intentionally independent.
#[derive(Clone, Debug)]
pub struct ManagedAllocationSnapshot {
    pub identity: AllocationIdentity,
    pub binding: BindingIdentity,
    pub authority: ProcessAuthorityId,
    pub authority_label: Arc<str>,
    pub memory_authority: Option<crate::MemoryAuthorityId>,
    pub holder: HolderId,
    pub holder_label: Arc<str>,
    pub tier: Tier,
    pub role: MemoryRole,
    pub charge_mode: AllocationChargeMode,
    pub state: ManagedAllocationState,
    pub charged_bytes: u64,
    pub process_reserved_bytes: u64,
    pub physical_bytes: Option<u64>,
    pub mapped_bytes: Option<u64>,
    pub unattributed_bytes: u64,
    pub shared_physical: SharedPhysicalIdentity,
}

/// Canonical authority accounting, independent of allocation residency.
#[derive(Clone, Debug)]
pub struct AuthorityMemorySnapshot {
    pub authority: ProcessAuthorityId,
    pub label: Arc<str>,
    pub memory_authority: Option<crate::MemoryAuthorityId>,
    pub device: DeviceKey,
    pub governed: bool,
    /// Device, host, disk charged bytes.
    pub used: [u64; 3],
    /// Device, host, disk locally grantable bytes.
    pub available: [u64; 3],
    /// Device, host, disk usage above the local ceiling.
    pub oversubscribed: [u64; 3],
}

/// Process snapshot. `charged`, `physical`, and `mapped` are separate axes.
#[derive(Clone, Debug)]
pub struct ProcessMemorySnapshot {
    pub process_limits: ProcessMemoryLimits,
    /// Device, host, disk process reservations.
    pub process_used: [u64; 3],
    pub authority_count: usize,
    pub authority_snapshots: Vec<AuthorityMemorySnapshot>,
    pub mechanism_snapshots: Vec<MechanismSnapshot>,
    pub allocations: Vec<ManagedAllocationSnapshot>,
    /// Sum of canonical authority usage; never inferred from residency.
    pub charged_bytes: u64,
    /// Charges attributable to currently manager-published allocations. This can
    /// be lower than `charged_bytes` when an authority owns retained pools.
    pub attributed_allocation_charged_bytes: u64,
    pub process_reserved_bytes: u64,
    pub known_physical_bytes: u64,
    pub mapped_bytes: u64,
    pub unattributed_bytes: u64,
    pub unknown_physical_allocations: usize,
}
