//! Coarse, synchronous Host workspace reuse over managed physical ownership.
//!
//! One checkout covers an execution's peak, not each kernel's temporary. Idle
//! slots retain their original charge. Growth replaces an idle slot before
//! admission; pressure releases only idle slots, outside every pool lock.
//! Device work with pending fences must keep its owner outside this Host pool.

use std::sync::{Arc, Mutex, Weak};

use crate::{
    AllocationPublication, AllocationRequest, AllocationTransactionError, ManagedAllocation,
    MemoryAuthorityId, PressureResponder, ProcessMemoryManager, RegisteredMemoryAuthority,
    RegisteredMemoryHolder, ScopedMemoryBinding,
};
use onnx_runtime_memory_api::{
    AllocationIdentity, AllocationReleaseOutcome, MemoryError, MemoryRole, Tier,
};

#[derive(Debug, thiserror::Error)]
pub enum WorkspacePoolError {
    #[error(
        "Host workspace rejects {bytes} bytes aligned to {alignment} for {role:?}; use nonzero addressable capacity, power-of-two alignment and a Workspace role"
    )]
    InvalidRequest {
        bytes: usize,
        alignment: usize,
        role: MemoryRole,
    },
    #[error(
        "Host workspace requires one Host domain, got authority {authority:?} ({authority_device:?}) and binding authority {binding_authority:?} ({binding_device:?}); bind both to the same registered Host authority"
    )]
    DomainMismatch {
        authority: crate::ProcessAuthorityId,
        binding_authority: crate::ProcessAuthorityId,
        authority_device: onnx_runtime_memory_api::DeviceKey,
        binding_device: onnx_runtime_memory_api::DeviceKey,
    },
    #[error("Host workspace transaction failed: {0}")]
    Transaction(#[from] AllocationTransactionError),
    #[error(
        "Host workspace release could not be prepared: {0}; the error retains its unchanged, funded allocation"
    )]
    ReleasePreparation(#[source] Box<crate::ManagedReleaseError>),
    #[error(
        "Host workspace {identity:?} release was incomplete ({outcome:?}); retained backing remains charged and cannot be reused"
    )]
    IncompleteRelease {
        identity: AllocationIdentity,
        outcome: Box<AllocationReleaseOutcome>,
    },
    #[error(
        "Host workspace return belongs to a different binding, holder or resource role; return it to its original pool"
    )]
    ForeignReturn { allocation: Box<ManagedAllocation> },
    #[error(
        "Host workspace still has {views} owned view(s); release those views before returning its slot"
    )]
    BorrowedReturn {
        views: usize,
        allocation: Box<ManagedAllocation>,
    },
}

#[derive(Debug)]
struct IdleSlot {
    owner: ManagedAllocation,
    role: MemoryRole,
}
#[derive(Debug, Default)]
struct IdleSlots {
    slots: Mutex<Vec<IdleSlot>>,
}
impl IdleSlots {
    fn release(&self, want: u64) -> Result<u64, WorkspacePoolError> {
        let mut released = 0u64;
        while released < want {
            let slot = {
                let mut slots = self
                    .slots
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                if slots.is_empty() {
                    None
                } else {
                    Some(slots.remove(0))
                }
            };
            let Some(slot) = slot else { break };
            released = released.saturating_add(self.release_slot(slot)?);
        }
        Ok(released)
    }
    fn release_slot(&self, slot: IdleSlot) -> Result<u64, WorkspacePoolError> {
        let identity = slot.owner.identity();
        let bytes = slot.owner.len() as u64;
        match slot.owner.release_now() {
            Ok(outcome) if outcome.is_complete() => Ok(bytes),
            Ok(outcome) => Err(WorkspacePoolError::IncompleteRelease {
                identity,
                outcome: Box::new(outcome),
            }),
            Err(error) => Err(WorkspacePoolError::ReleasePreparation(Box::new(error))),
        }
    }
}
#[derive(Debug)]
struct IdlePressure(Weak<IdleSlots>);
impl PressureResponder for IdlePressure {
    fn on_pressure(&self, tier: Tier, want: u64) -> Result<u64, MemoryError> {
        if tier != Tier::Host || want == 0 {
            return Ok(0);
        }
        self.0.upgrade().map_or(Ok(0), |idle| {
            idle.release(want)
                .map_err(|source| MemoryError::ReclaimFailed {
                    tier: tier.name(),
                    requested: want,
                    source: Box::new(source),
                })
        })
    }
}

#[derive(Debug)]
struct HostRegistration {
    manager: ProcessMemoryManager,
    context: crate::RegisteredMemoryContext,
    mechanism: crate::RegisteredMemoryMechanism,
}
impl Drop for HostRegistration {
    fn drop(&mut self) {
        if let Err(error) = self.manager.retire(&self.mechanism) {
            eprintln!(
                "Host workspace mechanism could not retire: {error}; finish outstanding managed operations before removal"
            );
            return;
        }
        if let Err(error) = self.manager.remove_mechanism(&self.mechanism) {
            eprintln!(
                "Host workspace registration remains pinned by outstanding ownership: {error}; finish managed releases before retiring its context"
            );
            return;
        }
        if let Err(error) = self.manager.remove_provider_context(&self.context) {
            eprintln!(
                "Host workspace context could not retire: {error}; finish outstanding context operations before removal"
            );
        }
    }
}

#[derive(Debug)]
struct PoolInner {
    manager: ProcessMemoryManager,
    binding: ScopedMemoryBinding,
    holder: RegisteredMemoryHolder,
    authority: RegisteredMemoryAuthority,
    idle: Arc<IdleSlots>,
    _pressure: Arc<IdlePressure>,
    // Last: the scoped binding must drop before unregistering private backing.
    _registration: Option<HostRegistration>,
}
impl Drop for PoolInner {
    fn drop(&mut self) {
        if let Err(error) = self.idle.release(u64::MAX) {
            eprintln!("Host workspace shutdown: {error}");
        }
        // An outstanding owner/quarantine keeps its holder published. Its
        // manager settlement remains authoritative after this pool is gone.
        let _ = self.manager.unregister_holder(&self.holder);
    }
}

/// Shared Host execution-slot backing. Clones share idle storage, not a budget.
///
/// The supplied binding must use an allocator whose completed physical release
/// does not silently park uncharged backing in another cache. Use HostAllocator
/// for the initial CPU integration. Ordinary tensor allocators are independent.
#[derive(Clone, Debug)]
pub struct HostWorkspacePool(Arc<PoolInner>);
impl HostWorkspacePool {
    pub fn new(
        manager: &ProcessMemoryManager,
        authority: &RegisteredMemoryAuthority,
        binding: ScopedMemoryBinding,
    ) -> Result<Self, WorkspacePoolError> {
        if authority.device().tier != Tier::Host
            || binding.identity().device().tier != Tier::Host
            || binding.authority() != authority.identity()
        {
            return Err(WorkspacePoolError::DomainMismatch {
                authority: authority.identity(),
                binding_authority: binding.authority(),
                authority_device: authority.device(),
                binding_device: binding.identity().device(),
            });
        }
        let idle = Arc::new(IdleSlots::default());
        let pressure = Arc::new(IdlePressure(Arc::downgrade(&idle)));
        let holder =
            manager.register_holder(authority, "execution workspace", Some(pressure.clone()))?;
        Ok(Self(Arc::new(PoolInner {
            manager: manager.clone(),
            binding,
            holder,
            authority: authority.clone(),
            idle,
            _pressure: pressure,
            _registration: None,
        })))
    }
    /// Bind an uncached Host backing allocator to the caller's actual books.
    /// Registration is cold and clones share this one resource domain.
    pub fn governed(
        manager: &ProcessMemoryManager,
        governor: Arc<dyn crate::MemoryGovernor + Send + Sync>,
    ) -> Result<Self, WorkspacePoolError> {
        let authority = manager.register_authority(
            onnx_runtime_memory_api::DeviceKey::HOST,
            "Host workspace authority",
            Arc::new(()),
            governor,
        )?;
        Self::with_host_backing(manager, &authority)
    }

    /// Explicitly ungoverned Host workspace. The manager reports physical bytes
    /// as unattributed rather than inventing a private unlimited budget.
    pub fn ungoverned() -> Result<Self, WorkspacePoolError> {
        let manager = ProcessMemoryManager::new().map_err(AllocationTransactionError::Binding)?;
        let authority = manager.register_compatibility_authority(
            onnx_runtime_memory_api::DeviceKey::HOST,
            "ungoverned Host workspace",
            Arc::new(()),
        )?;
        Self::with_host_backing(&manager, &authority)
    }
    fn with_host_backing(
        manager: &ProcessMemoryManager,
        authority: &RegisteredMemoryAuthority,
    ) -> Result<Self, WorkspacePoolError> {
        let context = manager
            .register_provider_context(
                onnx_runtime_memory_api::DeviceKey::HOST,
                "Host workspace",
                Arc::new(()),
            )
            .map_err(AllocationTransactionError::Binding)?;
        let mechanism = match manager.register_allocator(
            &context,
            authority,
            "uncached Host workspace backing",
            Arc::new(onnx_runtime_memory_api::HostAllocator),
        ) {
            Ok(mechanism) => mechanism,
            Err(error) => {
                let _ = manager.remove_provider_context(&context);
                return Err(error.into());
            }
        };
        let registration = HostRegistration {
            manager: manager.clone(),
            context,
            mechanism,
        };
        let binding = manager.bind_registered(&registration.mechanism)?;
        let mut pool = Self::new(manager, authority, binding)?;
        // No clone escapes construction; pressure holds only Weak<IdleSlots>.
        Arc::get_mut(&mut pool.0)
            .expect("new workspace domain is uniquely owned")
            ._registration = Some(registration);
        Ok(pool)
    }

    pub fn authority_id(&self) -> Option<MemoryAuthorityId> {
        self.0.authority.memory_authority_id()
    }

    /// Obtain one exclusively owned slot, reusing a sufficient idle allocation.
    /// No allocation/authority callback occurs on an admitted warm checkout.
    pub fn checkout(
        &self,
        mut bytes: usize,
        mut alignment: usize,
        role: MemoryRole,
    ) -> Result<ManagedAllocation, WorkspacePoolError> {
        if bytes == 0
            || bytes > isize::MAX as usize
            || alignment == 0
            || !alignment.is_power_of_two()
            || !matches!(role, MemoryRole::Workspace { .. })
            || std::alloc::Layout::from_size_align(bytes, alignment).is_err()
        {
            return Err(WorkspacePoolError::InvalidRequest {
                bytes,
                alignment,
                role,
            });
        }
        let _operation = self.0.binding.context_scope().enter()?;
        let replacement = {
            let mut slots = self
                .0
                .idle
                .slots
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if let Some(index) = slots
                .iter()
                .enumerate()
                .filter(|(_, slot)| {
                    slot.role == role
                        && slot.owner.len() >= bytes
                        && slot.owner.alignment() >= alignment
                })
                .min_by_key(|(_, slot)| slot.owner.len())
                .map(|(index, _)| index)
            {
                return Ok(slots.swap_remove(index).owner);
            }
            // Replace an existing lane instead of accumulating an idle slot
            // for every shape. Within one role, simultaneous checkouts add lanes.
            slots
                .iter()
                .enumerate()
                .filter(|(_, slot)| slot.role == role)
                .max_by_key(|(_, slot)| slot.owner.len())
                .map(|(index, _)| index)
                .map(|index| slots.swap_remove(index))
        };
        if let Some(old) = replacement {
            bytes = bytes.max(old.owner.len());
            alignment = alignment.max(old.owner.alignment());
            if std::alloc::Layout::from_size_align(bytes, alignment).is_err() {
                self.0
                    .idle
                    .slots
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .push(old);
                return Err(WorkspacePoolError::InvalidRequest {
                    bytes,
                    alignment,
                    role,
                });
            }
            self.0.idle.release_slot(old)?;
        }
        let bytes64 = bytes as u64;
        let delegated = self.0.authority.has_process_delegation(Tier::Host);
        let process_bytes = if delegated { 0 } else { bytes64 };
        let (request, mut publication) = if self.authority_id().is_some() {
            (
                AllocationRequest::managed(
                    bytes,
                    alignment,
                    Tier::Host,
                    role,
                    self.0.holder.clone(),
                    bytes64,
                )
                .with_process_reservation(process_bytes),
                AllocationPublication::exclusive(bytes64, bytes64, bytes64),
            )
        } else {
            (
                AllocationRequest::compatibility(
                    bytes,
                    alignment,
                    Tier::Host,
                    role,
                    self.0.holder.clone(),
                ),
                AllocationPublication::compatibility(bytes64, bytes64),
            )
        };
        if self.authority_id().is_some() {
            publication.process_reserved_bytes = process_bytes;
        }
        self.0
            .binding
            .allocate(request, publication)
            .map_err(Into::into)
    }

    /// Return a completed execution's owner. It remains allocated and funded.
    /// Callers must retain active/pending/pinned owners instead of returning them.
    pub fn recycle_completed(
        &self,
        allocation: ManagedAllocation,
    ) -> Result<(), WorkspacePoolError> {
        let (holder, role) = allocation.holder_and_role();
        if allocation.identity().binding() != self.0.binding.identity()
            || holder != self.0.holder.id()
            || !matches!(role, MemoryRole::Workspace { .. })
        {
            return Err(WorkspacePoolError::ForeignReturn {
                allocation: Box::new(allocation),
            });
        }
        let views = allocation.owner_ref().outstanding_views();
        if views != 0 {
            return Err(WorkspacePoolError::BorrowedReturn {
                views,
                allocation: Box::new(allocation),
            });
        }
        self.0
            .idle
            .slots
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(IdleSlot {
                owner: allocation,
                role,
            });
        Ok(())
    }
    /// Release idle backing and settle only verified physical release outcomes.
    pub fn trim_idle(&self, want: u64) -> Result<u64, WorkspacePoolError> {
        self.0.idle.release(want)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        HolderId, LeaseAccounting, LeaseLedger, LedgerGovernor, MemoryGovernor, MemoryLease,
        PressureRegistry,
    };
    use onnx_runtime_memory_api::{DeviceAllocator, DeviceKey, HostAllocator};
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[derive(Debug)]
    struct Books {
        inner: LedgerGovernor,
        claims: AtomicUsize,
        releases: AtomicUsize,
    }
    #[derive(Clone, Debug)]
    struct Counting(Arc<Books>);
    impl Counting {
        fn new(bytes: u64) -> Self {
            Self(Arc::new(Books {
                inner: LedgerGovernor::new(LeaseLedger::new_for_device(
                    DeviceKey::HOST,
                    0,
                    bytes,
                    0,
                )),
                claims: AtomicUsize::new(0),
                releases: AtomicUsize::new(0),
            }))
        }
        fn callbacks(&self) -> (usize, usize) {
            (
                self.0.claims.load(Ordering::Relaxed),
                self.0.releases.load(Ordering::Relaxed),
            )
        }
    }
    impl LeaseAccounting for Books {
        fn authority_id(&self) -> Option<MemoryAuthorityId> {
            Some(self.inner.authority_id(Tier::Host))
        }
        fn try_claim(&self, tier: Tier, bytes: u64, role: MemoryRole) -> Result<(), MemoryError> {
            self.claims.fetch_add(1, Ordering::Relaxed);
            self.inner.ledger().try_claim(tier, bytes, role)
        }
        fn release(&self, tier: Tier, bytes: u64) {
            self.releases.fetch_add(1, Ordering::Relaxed);
            self.inner.ledger().release(tier, bytes);
        }
    }
    impl MemoryGovernor for Counting {
        fn authority_id(&self, tier: Tier) -> MemoryAuthorityId {
            self.0.inner.authority_id(tier)
        }
        fn reserve(
            &self,
            tier: Tier,
            bytes: u64,
            role: MemoryRole,
            holder: HolderId,
        ) -> Result<MemoryLease, MemoryError> {
            self.0.try_claim(tier, bytes, role)?;
            Ok(MemoryLease::new(tier, bytes, role, holder, self.0.clone()))
        }
        fn pressure_registry(&self, tier: Tier) -> Option<PressureRegistry> {
            self.0.inner.pressure_registry(tier)
        }
        fn available(&self, tier: Tier) -> u64 {
            self.0.inner.available(tier)
        }
        fn used(&self, tier: Tier) -> u64 {
            self.0.inner.used(tier)
        }
    }
    fn fixture(
        bytes: u64,
    ) -> (
        Counting,
        ProcessMemoryManager,
        RegisteredMemoryAuthority,
        HostWorkspacePool,
    ) {
        let books = Counting::new(bytes);
        let manager = ProcessMemoryManager::new().unwrap();
        let context = manager
            .register_provider_context(DeviceKey::HOST, "workspace test", Arc::new(()))
            .unwrap();
        let authority = manager
            .register_authority(
                DeviceKey::HOST,
                "shared host",
                Arc::new(()),
                Arc::new(books.clone()),
            )
            .unwrap();
        let mechanism = manager
            .register_allocator(
                &context,
                &authority,
                "uncached host backing",
                Arc::new(HostAllocator),
            )
            .unwrap();
        let pool = HostWorkspacePool::new(
            &manager,
            &authority,
            manager.bind_registered(&mechanism).unwrap(),
        )
        .unwrap();
        (books, manager, authority, pool)
    }
    const ROLE: MemoryRole = MemoryRole::Workspace { step_scoped: true };

    #[test]
    fn warm_runs_reuse_peak_capacity_and_its_exact_original_grant() {
        let (books, manager, _, pool) = fixture(64);
        let owner = pool.checkout(16, 16, ROLE).unwrap();
        let first = owner.identity();
        assert_eq!(books.used(Tier::Host), 16);
        pool.recycle_completed(owner).unwrap();
        let small = pool.checkout(8, 8, ROLE).unwrap();
        assert_eq!(small.identity(), first);
        assert_eq!(small.len(), 16);
        pool.recycle_completed(small).unwrap();
        let before = books.callbacks();
        for _ in 0..4096 {
            let owner = pool.checkout(16, 16, ROLE).unwrap();
            assert_eq!(owner.identity(), first);
            pool.recycle_completed(owner).unwrap();
        }
        assert_eq!(
            books.callbacks(),
            before,
            "warm borrow/return must not call the issuer, including zero-byte accounting"
        );
        for (bytes, alignment) in [(usize::MAX, 16), (16, 1usize << (usize::BITS - 1))] {
            assert!(pool.checkout(bytes, alignment, ROLE).is_err());
        }
        assert_eq!(books.callbacks(), before);
        let original = pool.checkout(16, 16, ROLE).unwrap();
        assert_eq!(original.identity(), first);
        pool.recycle_completed(original).unwrap();
        // Replacing a lane releases its old 16 bytes before claiming 64.
        let grown = pool.checkout(64, 32, ROLE).unwrap();
        assert_ne!(grown.identity(), first);
        assert_eq!(books.used(Tier::Host), 64);
        assert_eq!(manager.process_used(Tier::Host), 64);
        assert_eq!(books.callbacks(), (before.0 + 1, before.1 + 1));
        let snapshot = manager.snapshot().unwrap();
        assert_eq!(snapshot.allocations.len(), 1);
        assert_eq!(snapshot.allocations[0].role, ROLE);
        pool.recycle_completed(grown).unwrap();
        assert_eq!(pool.trim_idle(1).unwrap(), 64);
        assert_eq!(books.used(Tier::Host), 0);
        assert_eq!(manager.process_used(Tier::Host), 0);
        assert!(manager.snapshot().unwrap().allocations.is_empty());
        let before = books.callbacks();
        for (bytes, alignment, role) in
            [(0, 16, ROLE), (16, 3, ROLE), (16, 16, MemoryRole::Weights)]
        {
            assert!(pool.checkout(bytes, alignment, role).is_err());
        }
        assert_eq!(books.callbacks(), before);
    }

    #[test]
    fn owned_host_registration_retires_after_idle_backing_and_its_binding() {
        let books = Counting::new(64);
        let manager = ProcessMemoryManager::new().unwrap();
        let pool = HostWorkspacePool::governed(&manager, Arc::new(books.clone())).unwrap();
        let mechanism = pool.0._registration.as_ref().unwrap().mechanism.clone();
        let owner = pool.checkout(64, 16, ROLE).unwrap();
        pool.recycle_completed(owner).unwrap();
        assert!(manager.mechanism_snapshot(&mechanism).is_ok());
        drop(pool);
        assert_eq!(books.used(Tier::Host), 0);
        assert!(manager.snapshot().unwrap().allocations.is_empty());
        assert!(
            manager.mechanism_snapshot(&mechanism).is_err(),
            "private backing mechanism must not accumulate after domains unload"
        );
    }

    #[test]
    fn alternating_capacity_alignment_and_owned_views_preserve_exclusive_warm_slots() {
        let (books, _, _, pool) = fixture(64);
        let initial = pool.checkout(64, 16, ROLE).unwrap();
        pool.recycle_completed(initial).unwrap();
        let aligned = pool.checkout(16, 128, ROLE).unwrap();
        assert_eq!(aligned.len(), 64);
        assert_eq!(aligned.alignment(), 128);
        let id = aligned.identity();
        let view = aligned.view(0, 16).unwrap();
        let error = pool.recycle_completed(aligned).unwrap_err();
        let WorkspacePoolError::BorrowedReturn { allocation, .. } = error else {
            panic!("owned views must keep the slot active")
        };
        assert!(pool.checkout(64, 16, ROLE).is_err());
        drop(view);
        pool.recycle_completed(*allocation).unwrap();
        let warm = books.callbacks();
        for _ in 0..32 {
            for (bytes, alignment) in [(64, 16), (16, 128)] {
                let owner = pool.checkout(bytes, alignment, ROLE).unwrap();
                assert_eq!(owner.identity(), id);
                pool.recycle_completed(owner).unwrap();
            }
        }
        assert_eq!(books.callbacks(), warm);
        drop(pool);
        assert_eq!(books.used(Tier::Host), 0);
    }

    #[test]
    fn concurrent_runs_own_distinct_slots_and_only_idle_storage_yields_to_kv() {
        let (books, _, _, pool) = fixture(128);
        for round in 0..2 {
            let entered = Arc::new(std::sync::Barrier::new(3));
            let leave = Arc::new(std::sync::Barrier::new(3));
            let before = books.callbacks();
            std::thread::scope(|scope| {
                let mut handles = vec![];
                for _ in 0..2 {
                    let (pool, entered, leave) = (pool.clone(), entered.clone(), leave.clone());
                    handles.push(scope.spawn(move || {
                        let owner = pool.checkout(64, 16, ROLE).unwrap();
                        let id = owner.identity();
                        entered.wait();
                        leave.wait();
                        pool.recycle_completed(owner).unwrap();
                        id
                    }));
                }
                entered.wait();
                assert_eq!(books.used(Tier::Host), 128);
                if round == 0 {
                    assert!(
                        books
                            .reserve(Tier::Host, 1, MemoryRole::KvCache, HolderId::new(99))
                            .is_err()
                    );
                } else {
                    assert_eq!(books.callbacks(), before);
                }
                leave.wait();
                let a = handles.remove(0).join().unwrap();
                let b = handles.remove(0).join().unwrap();
                assert_ne!(a, b);
            });
        }
        let live = pool.checkout(64, 16, ROLE).unwrap();
        let kv = books
            .reserve(Tier::Host, 64, MemoryRole::KvCache, HolderId::new(99))
            .unwrap();
        assert_eq!(books.used(Tier::Host), 128);
        assert_eq!(pool.trim_idle(u64::MAX).unwrap(), 0);
        assert_eq!(live.len(), 64);
        drop(kv);
        pool.recycle_completed(live).unwrap();
        drop(pool);
        assert_eq!(books.used(Tier::Host), 0);
    }

    #[test]
    fn delegation_and_foreign_returns_do_not_duplicate_or_lose_charges() {
        let (books, manager, authority, pool) = fixture(128);
        manager
            .delegate_authority_capacity(&authority, Tier::Host, 128)
            .unwrap();
        let owner = pool.checkout(64, 16, ROLE).unwrap();
        assert_eq!(books.used(Tier::Host), 64);
        assert_eq!(manager.process_used(Tier::Host), 128);
        let (_, _, _, foreign) = fixture(128);
        let error = foreign.recycle_completed(owner).unwrap_err();
        let WorkspacePoolError::ForeignReturn { allocation } = error else {
            panic!("return must preserve its original owner")
        };
        assert_eq!(books.used(Tier::Host), 64);
        pool.recycle_completed(*allocation).unwrap();
        assert_eq!(pool.trim_idle(u64::MAX).unwrap(), 64);
        assert_eq!(books.used(Tier::Host), 0);
        assert_eq!(
            manager.process_used(Tier::Host),
            128,
            "delegated process capacity belongs to the authority, not a workspace allocation"
        );
    }

    #[derive(Debug, Default)]
    struct RefuseRelease(Mutex<Option<(usize, usize, usize)>>);
    impl DeviceAllocator for RefuseRelease {
        fn allocate(
            &self,
            bytes: usize,
            alignment: usize,
        ) -> Result<std::ptr::NonNull<u8>, MemoryError> {
            let ptr = HostAllocator.allocate(bytes, alignment)?;
            *self.0.lock().unwrap() = Some((ptr.as_ptr() as usize, bytes, alignment));
            Ok(ptr)
        }
        unsafe fn deallocate(&self, ptr: std::ptr::NonNull<u8>, bytes: usize, alignment: usize) {
            unsafe { HostAllocator.deallocate(ptr, bytes, alignment) };
            self.0.lock().unwrap().take();
        }
        unsafe fn release(
            &self,
            _: std::ptr::NonNull<u8>,
            _: usize,
            _: usize,
        ) -> AllocationReleaseOutcome {
            AllocationReleaseOutcome::failed("injected backing release refusal")
        }
        fn device(&self) -> DeviceKey {
            DeviceKey::HOST
        }
    }
    #[test]
    fn incomplete_physical_release_never_refunds_or_reuses_the_backing() {
        let books = Counting::new(64);
        let manager = ProcessMemoryManager::new().unwrap();
        let context = manager
            .register_provider_context(DeviceKey::HOST, "failure test", Arc::new(()))
            .unwrap();
        let authority = manager
            .register_authority(
                DeviceKey::HOST,
                "shared books",
                Arc::new(()),
                Arc::new(books.clone()),
            )
            .unwrap();
        let allocator = Arc::new(RefuseRelease::default());
        let mechanism = manager
            .register_allocator(&context, &authority, "release refusal", allocator.clone())
            .unwrap();
        let pool = HostWorkspacePool::new(
            &manager,
            &authority,
            manager.bind_registered(&mechanism).unwrap(),
        )
        .unwrap();
        let owner = pool.checkout(64, 16, ROLE).unwrap();
        pool.recycle_completed(owner).unwrap();
        assert!(matches!(
            pool.trim_idle(64),
            Err(WorkspacePoolError::IncompleteRelease { .. })
        ));
        assert_eq!(books.used(Tier::Host), 64);
        assert_eq!(books.callbacks(), (1, 0));
        assert_eq!(manager.process_used(Tier::Host), 64);
        assert!(pool.checkout(64, 16, ROLE).is_err());
        assert_eq!(manager.snapshot().unwrap().allocations.len(), 1);
        // Test-only teardown of deliberately quarantined physical memory. No
        // production settlement is fabricated and no later operation uses it.
        if let Some((ptr, bytes, alignment)) = allocator.0.lock().unwrap().take() {
            unsafe {
                HostAllocator.deallocate(
                    std::ptr::NonNull::new(ptr as *mut u8).unwrap(),
                    bytes,
                    alignment,
                )
            };
        }
    }
}
