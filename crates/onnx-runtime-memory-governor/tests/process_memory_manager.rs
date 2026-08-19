use std::alloc::Layout;
use std::collections::HashMap;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Barrier, Mutex};
use std::thread;

use onnx_runtime_memory_governor::{
    AllocationChargeMode, AllocationPublication, AllocationReleaseOutcome, AllocationRequest,
    AllocationStepError, AllocationTransactionError, BindingRegistry, DeviceAllocator, DeviceKey,
    HolderId, HostAllocator, LeaseLedger, LedgerGovernor, MemoryError, MemoryGovernor, MemoryRole,
    ProcessMemoryLimits, ProcessMemoryManager, QuarantineReason, RegisteredMemoryAuthority,
    RegisteredMemoryContext, RegisteredMemoryHolder, RegisteredMemoryMechanism, ReleaseAccounting,
    ResidualOwnership, Tier,
};

#[derive(Debug, Default)]
struct Pin;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ReleaseBehavior {
    Complete,
    Quarantine,
}

#[derive(Debug)]
struct TestAllocator {
    device: DeviceKey,
    allocations: AtomicU64,
    releases: AtomicU64,
    fail_allocate: AtomicBool,
    release_behavior: Mutex<ReleaseBehavior>,
    live: Mutex<HashMap<usize, Layout>>,
}

impl TestAllocator {
    fn new(device: DeviceKey) -> Arc<Self> {
        Arc::new(Self {
            device,
            allocations: AtomicU64::new(0),
            releases: AtomicU64::new(0),
            fail_allocate: AtomicBool::new(false),
            release_behavior: Mutex::new(ReleaseBehavior::Complete),
            live: Mutex::new(HashMap::new()),
        })
    }

    fn quarantine(&self) {
        *self
            .release_behavior
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = ReleaseBehavior::Quarantine;
    }

    fn cleanup(&self) {
        let mut live = self
            .live
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        for (address, layout) in live.drain() {
            // SAFETY: every entry was inserted after a successful allocation and
            // is removed exactly once either here or by a complete release.
            unsafe { std::alloc::dealloc(address as *mut u8, layout) };
        }
    }
}

impl Drop for TestAllocator {
    fn drop(&mut self) {
        let live = self
            .live
            .get_mut()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        for (address, layout) in live.drain() {
            // SAFETY: same ownership argument as `cleanup`.
            unsafe { std::alloc::dealloc(address as *mut u8, layout) };
        }
    }
}

impl DeviceAllocator for TestAllocator {
    fn allocate(&self, bytes: usize, align: usize) -> Result<NonNull<u8>, MemoryError> {
        if self.fail_allocate.load(Ordering::Acquire) {
            return Err(MemoryError::AllocationFailed {
                tier: self.device.tier.name(),
                requested: bytes as u64,
                reason: "injected allocation failure".into(),
            });
        }
        let layout = Layout::from_size_align(bytes.max(1), align).map_err(|_| {
            MemoryError::InvalidRequest {
                tier: self.device.tier.name(),
                requested: bytes as u64,
                reason: "invalid test layout",
            }
        })?;
        // SAFETY: layout is non-zero and valid.
        let ptr = NonNull::new(unsafe { std::alloc::alloc(layout) }).ok_or_else(|| {
            MemoryError::AllocationFailed {
                tier: self.device.tier.name(),
                requested: bytes as u64,
                reason: "test allocator exhausted".into(),
            }
        })?;
        self.live
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .insert(ptr.as_ptr() as usize, layout);
        self.allocations.fetch_add(1, Ordering::Relaxed);
        Ok(ptr)
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, _bytes: usize, _align: usize) {
        let layout = self
            .live
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .remove(&(ptr.as_ptr() as usize))
            .expect("test allocation is released once");
        // SAFETY: the pointer/layout pair came from this allocator.
        unsafe { std::alloc::dealloc(ptr.as_ptr(), layout) };
        self.releases.fetch_add(1, Ordering::Relaxed);
    }

    unsafe fn release(
        &self,
        ptr: NonNull<u8>,
        bytes: usize,
        align: usize,
    ) -> AllocationReleaseOutcome {
        match *self
            .release_behavior
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
        {
            ReleaseBehavior::Complete => {
                // SAFETY: delegated to this method's contract.
                unsafe { self.deallocate(ptr, bytes, align) };
                AllocationReleaseOutcome::complete(ReleaseAccounting::eager(bytes as u64))
            }
            ReleaseBehavior::Quarantine => AllocationReleaseOutcome::quarantined(
                ReleaseAccounting::new(bytes as u64, 0),
                ResidualOwnership {
                    state: onnx_runtime_memory_governor::AllocationReleaseState::Quarantined,
                    reason: QuarantineReason::PartialRelease,
                    retained_bytes: bytes as u64,
                    address: ptr.as_ptr() as usize,
                    align,
                },
            ),
        }
    }

    fn device(&self) -> DeviceKey {
        self.device
    }
}

struct Fixture {
    manager: ProcessMemoryManager,
    context: RegisteredMemoryContext,
    authority: RegisteredMemoryAuthority,
    mechanism: RegisteredMemoryMechanism,
    holder: RegisteredMemoryHolder,
    governor: Arc<LedgerGovernor>,
    allocator: Arc<TestAllocator>,
}

impl Fixture {
    fn new(device: DeviceKey, authority_limit: u64, process_limits: ProcessMemoryLimits) -> Self {
        let manager = ProcessMemoryManager::with_limits(process_limits).expect("manager");
        let context = manager
            .register_provider_context(device, "test context", Arc::new(Pin))
            .expect("context");
        let governor = Arc::new(LedgerGovernor::new(LeaseLedger::new_for_device(
            device,
            authority_limit,
            authority_limit,
            authority_limit,
        )));
        let authority = manager
            .register_authority(
                device,
                "test authority",
                Arc::new(Pin),
                Arc::clone(&governor) as Arc<dyn MemoryGovernor + Send + Sync>,
            )
            .expect("authority");
        let allocator = TestAllocator::new(device);
        let mechanism = manager
            .register_allocator(
                &context,
                &authority,
                "test mechanism",
                Arc::clone(&allocator) as Arc<dyn DeviceAllocator>,
            )
            .expect("mechanism");
        manager.select(&mechanism).expect("select");
        let holder = manager
            .register_holder(&authority, "session", None)
            .expect("holder");
        Self {
            manager,
            context,
            authority,
            mechanism,
            holder,
            governor,
            allocator,
        }
    }

    fn allocate(&self, bytes: usize) -> onnx_runtime_memory_governor::ManagedAllocation {
        let binding = self
            .manager
            .bind_registered(&self.mechanism)
            .expect("binding");
        binding
            .allocate(
                AllocationRequest::managed(
                    bytes,
                    64,
                    self.mechanism.device().tier,
                    MemoryRole::Workspace { step_scoped: false },
                    self.holder.clone(),
                    bytes as u64,
                ),
                AllocationPublication::exclusive(bytes as u64, bytes as u64, bytes as u64),
            )
            .expect("allocation")
    }
}

fn unlimited() -> ProcessMemoryLimits {
    ProcessMemoryLimits::UNLIMITED
}

#[test]
fn multiple_sessions_share_a_canonical_authority_without_sharing_bindings() {
    let fixture = Fixture::new(DeviceKey::HOST, 4096, unlimited());
    let duplicate = fixture
        .manager
        .register_authority(
            DeviceKey::HOST,
            "same books",
            Arc::new(Pin),
            Arc::clone(&fixture.governor) as Arc<dyn MemoryGovernor + Send + Sync>,
        )
        .expect("canonical authority");
    assert_eq!(duplicate.identity(), fixture.authority.identity());
    assert_eq!(
        duplicate.binding_identity(),
        fixture.authority.binding_identity()
    );

    let second_holder = fixture
        .manager
        .register_holder(&duplicate, "second session", None)
        .expect("second holder");
    let first_binding = fixture
        .manager
        .bind_registered(&fixture.mechanism)
        .expect("first binding");
    let second_binding = fixture
        .manager
        .bind_registered(&fixture.mechanism)
        .expect("second binding");
    assert_ne!(first_binding.identity(), second_binding.identity());

    let first = first_binding
        .allocate(
            AllocationRequest::managed(
                256,
                64,
                Tier::Host,
                MemoryRole::Weights,
                fixture.holder.clone(),
                256,
            ),
            AllocationPublication::exclusive(256, 256, 256),
        )
        .unwrap();
    let second = second_binding
        .allocate(
            AllocationRequest::managed(
                512,
                64,
                Tier::Host,
                MemoryRole::KvCache,
                second_holder,
                512,
            ),
            AllocationPublication::exclusive(512, 512, 512),
        )
        .unwrap();
    assert_eq!(fixture.governor.used(Tier::Host), 768);
    first.release_now().unwrap();
    assert_eq!(fixture.governor.used(Tier::Host), 512);
    assert_eq!(second.len(), 512);
    second.release_now().unwrap();
    assert_eq!(fixture.governor.used(Tier::Host), 0);
}

#[test]
fn different_devices_and_authorities_are_not_conflated() {
    let manager = ProcessMemoryManager::new().unwrap();
    let mut fixtures = Vec::new();
    for index in 0..2 {
        let device = DeviceKey::device(index);
        let context = manager
            .register_provider_context(device, format!("cuda:{index}"), Arc::new(Pin))
            .unwrap();
        let governor = Arc::new(LedgerGovernor::new(LeaseLedger::new_for_device(
            device, 1024, 0, 0,
        )));
        let authority = manager
            .register_authority(
                device,
                format!("authority:{index}"),
                Arc::new(Pin),
                Arc::clone(&governor) as Arc<dyn MemoryGovernor + Send + Sync>,
            )
            .unwrap();
        let allocator = TestAllocator::new(device);
        let mechanism = manager
            .register_allocator(
                &context,
                &authority,
                format!("allocator:{index}"),
                Arc::clone(&allocator) as Arc<dyn DeviceAllocator>,
            )
            .unwrap();
        let holder = manager
            .register_holder(&authority, format!("session:{index}"), None)
            .unwrap();
        fixtures.push((governor, authority, mechanism, holder, allocator));
    }

    assert_ne!(fixtures[0].1.identity(), fixtures[1].1.identity());
    let mut allocations = Vec::new();
    for (governor, _, mechanism, holder, _) in &fixtures {
        let allocation = manager
            .bind_registered(mechanism)
            .unwrap()
            .allocate(
                AllocationRequest::managed(
                    128,
                    64,
                    Tier::Device,
                    MemoryRole::KvCache,
                    holder.clone(),
                    128,
                ),
                AllocationPublication::exclusive(128, 128, 128),
            )
            .unwrap();
        assert_eq!(governor.used(Tier::Device), 128);
        allocations.push(allocation);
    }
    allocations.remove(0).release_now().unwrap();
    assert_eq!(fixtures[0].0.used(Tier::Device), 0);
    assert_eq!(fixtures[1].0.used(Tier::Device), 128);
    allocations.remove(0).release_now().unwrap();
}

#[test]
fn uma_device_views_reuse_one_canonical_authority_and_physical_identity() {
    let manager = ProcessMemoryManager::new().unwrap();
    let governor = Arc::new(LedgerGovernor::new(LeaseLedger::new_for_device(
        DeviceKey::HOST,
        1024,
        1024,
        0,
    )));
    let host_authority = manager
        .register_authority(
            DeviceKey::HOST,
            "uma physical books",
            Arc::new(Pin),
            governor.clone() as Arc<dyn MemoryGovernor + Send + Sync>,
        )
        .unwrap();
    let device_authority = manager
        .register_authority_alias(
            &host_authority,
            DeviceKey::device(0),
            "uma device view",
            Arc::new(Pin),
        )
        .unwrap();
    assert_eq!(host_authority.identity(), device_authority.identity());
    assert_ne!(
        host_authority.binding_identity(),
        device_authority.binding_identity()
    );

    let host_context = manager
        .register_provider_context(DeviceKey::HOST, "host view", Arc::new(Pin))
        .unwrap();
    let device_context = manager
        .register_provider_context(DeviceKey::device(0), "device view", Arc::new(Pin))
        .unwrap();
    let host_mechanism = manager
        .register_allocator(
            &host_context,
            &host_authority,
            "host allocator",
            TestAllocator::new(DeviceKey::HOST) as Arc<dyn DeviceAllocator>,
        )
        .unwrap();
    let device_mechanism = manager
        .register_allocator(
            &device_context,
            &device_authority,
            "device allocator",
            TestAllocator::new(DeviceKey::device(0)) as Arc<dyn DeviceAllocator>,
        )
        .unwrap();
    let holder = manager
        .register_holder(&host_authority, "shared session", None)
        .unwrap();
    let host_binding = manager.bind_registered(&host_mechanism).unwrap();
    let device_binding = manager.bind_registered(&device_mechanism).unwrap();
    let physical = host_binding.new_shared_physical_identity().unwrap();
    let host = host_binding
        .allocate(
            AllocationRequest::managed(
                128,
                64,
                Tier::Host,
                MemoryRole::KvCache,
                holder.clone(),
                128,
            ),
            AllocationPublication::exclusive(128, 128, 128).with_shared_physical(physical),
        )
        .unwrap();
    let device = device_binding
        .allocate(
            AllocationRequest::authority_managed(
                128,
                64,
                Tier::Device,
                MemoryRole::KvCache,
                holder,
                0,
            ),
            AllocationPublication {
                charged_bytes: 0,
                process_reserved_bytes: 0,
                physical_bytes: Some(128),
                mapped_bytes: Some(128),
                unattributed_bytes: 0,
                shared_physical: Some(physical),
            },
        )
        .unwrap();
    let snapshot = manager.snapshot().unwrap();
    assert_eq!(snapshot.authority_count, 1);
    assert_eq!(snapshot.known_physical_bytes, 128);
    assert_eq!(snapshot.mapped_bytes, 256);
    host.release_now().unwrap();
    device.release_now().unwrap();
}

#[test]
fn canonical_authority_view_is_recreated_while_an_alias_remains() {
    let manager = ProcessMemoryManager::new().unwrap();
    let governor = Arc::new(LedgerGovernor::new(LeaseLedger::new_for_device(
        DeviceKey::HOST,
        1024,
        1024,
        0,
    )));
    let canonical = manager
        .register_authority(
            DeviceKey::HOST,
            "canonical",
            Arc::new(Pin),
            governor.clone() as Arc<dyn MemoryGovernor + Send + Sync>,
        )
        .unwrap();
    let alias = manager
        .register_authority_alias(&canonical, DeviceKey::device(0), "alias", Arc::new(Pin))
        .unwrap();
    let old_binding_identity = canonical.binding_identity();
    manager.remove_authority(&canonical).unwrap();
    let recreated = manager
        .register_authority(
            DeviceKey::HOST,
            "canonical recreated",
            Arc::new(Pin),
            governor as Arc<dyn MemoryGovernor + Send + Sync>,
        )
        .unwrap();
    assert_eq!(recreated.identity(), alias.identity());
    assert_eq!(recreated.device(), DeviceKey::HOST);
    assert_ne!(recreated.binding_identity(), old_binding_identity);
    assert_eq!(manager.snapshot().unwrap().authority_count, 1);
}

#[test]
fn mechanism_switch_keeps_old_allocation_on_its_original_release_path() {
    let fixture = Fixture::new(DeviceKey::HOST, 4096, unlimited());
    let old_binding = fixture.manager.bind_registered(&fixture.mechanism).unwrap();
    let old = old_binding
        .allocate(
            AllocationRequest::managed(
                128,
                64,
                Tier::Host,
                MemoryRole::Weights,
                fixture.holder.clone(),
                128,
            ),
            AllocationPublication::exclusive(128, 128, 128),
        )
        .unwrap();

    let replacement = TestAllocator::new(DeviceKey::HOST);
    let new_mechanism = fixture
        .manager
        .register_allocator(
            &fixture.context,
            &fixture.authority,
            "replacement",
            Arc::clone(&replacement) as Arc<dyn DeviceAllocator>,
        )
        .unwrap();
    fixture.manager.select(&new_mechanism).unwrap();
    let new = fixture
        .manager
        .bind(DeviceKey::HOST)
        .unwrap()
        .allocate(
            AllocationRequest::managed(
                256,
                64,
                Tier::Host,
                MemoryRole::Weights,
                fixture.holder.clone(),
                256,
            ),
            AllocationPublication::exclusive(256, 256, 256),
        )
        .unwrap();

    old.release_now().unwrap();
    assert_eq!(fixture.allocator.releases.load(Ordering::Relaxed), 1);
    assert_eq!(replacement.releases.load(Ordering::Relaxed), 0);
    new.release_now().unwrap();
    assert_eq!(replacement.releases.load(Ordering::Relaxed), 1);
}

#[test]
fn reserve_allocate_commit_and_publish_failures_roll_back_exactly() {
    let fixture = Fixture::new(
        DeviceKey::HOST,
        128,
        ProcessMemoryLimits {
            host_bytes: 128,
            ..unlimited()
        },
    );
    let binding = fixture.manager.bind_registered(&fixture.mechanism).unwrap();
    let request = || {
        AllocationRequest::managed(
            256,
            64,
            Tier::Host,
            MemoryRole::Weights,
            fixture.holder.clone(),
            256,
        )
    };
    let reserve_error = binding
        .allocate(request(), AllocationPublication::exclusive(256, 256, 256))
        .unwrap_err();
    assert!(matches!(
        reserve_error,
        AllocationTransactionError::Memory(MemoryError::TierExhausted { .. })
    ));
    assert_eq!(fixture.allocator.allocations.load(Ordering::Relaxed), 0);
    assert_eq!(fixture.governor.used(Tier::Host), 0);
    assert_eq!(fixture.manager.process_used(Tier::Host), 0);

    fixture.manager.set_process_limit(Tier::Host, 4096).unwrap();
    fixture.governor.ledger().set_limit(Tier::Host, 4096);
    fixture
        .allocator
        .fail_allocate
        .store(true, Ordering::Release);
    let allocation_error = binding
        .allocate(request(), AllocationPublication::exclusive(256, 256, 256))
        .unwrap_err();
    assert!(matches!(
        allocation_error,
        AllocationTransactionError::Step {
            stage: "allocate",
            ..
        }
    ));
    assert_eq!(fixture.governor.used(Tier::Host), 0);
    assert_eq!(fixture.manager.process_used(Tier::Host), 0);

    fixture
        .allocator
        .fail_allocate
        .store(false, Ordering::Release);
    let commit_error = binding
        .allocate_with(
            request(),
            |context| context.allocate_owning(),
            |_| Err(AllocationStepError::new("injected commit failure")),
        )
        .unwrap_err();
    assert!(matches!(
        commit_error,
        AllocationTransactionError::Step {
            stage: "commit",
            ..
        }
    ));
    assert_eq!(fixture.allocator.releases.load(Ordering::Relaxed), 1);
    assert_eq!(fixture.governor.used(Tier::Host), 0);
    assert_eq!(fixture.manager.process_used(Tier::Host), 0);

    let foreign = Fixture::new(DeviceKey::HOST, 4096, unlimited());
    let foreign_shared = foreign
        .manager
        .bind_registered(&foreign.mechanism)
        .unwrap()
        .new_shared_physical_identity()
        .unwrap();
    let publish_error = binding
        .allocate(
            request(),
            AllocationPublication::exclusive(256, 256, 256).with_shared_physical(foreign_shared),
        )
        .unwrap_err();
    assert!(matches!(
        publish_error,
        AllocationTransactionError::Step {
            stage: "publish",
            ..
        }
    ));
    assert_eq!(fixture.allocator.releases.load(Ordering::Relaxed), 2);
    assert_eq!(fixture.governor.used(Tier::Host), 0);
    assert_eq!(fixture.manager.process_used(Tier::Host), 0);
}

#[test]
fn transaction_rejects_owner_issued_by_another_binding() {
    let fixture = Fixture::new(DeviceKey::HOST, 4096, unlimited());
    let foreign_registry = BindingRegistry::new().unwrap();
    let foreign_context = foreign_registry
        .register_provider_context(DeviceKey::HOST, Arc::new(Pin))
        .unwrap();
    let foreign_authority = foreign_registry
        .register_authority(DeviceKey::HOST, Arc::new(Pin))
        .unwrap();
    let foreign_mechanism = foreign_registry
        .register_allocator(foreign_context, foreign_authority, Arc::new(HostAllocator))
        .unwrap();
    let foreign_binding = foreign_registry.bind_registered(foreign_mechanism).unwrap();
    let error = fixture
        .manager
        .bind_registered(&fixture.mechanism)
        .unwrap()
        .allocate_with(
            AllocationRequest::managed(
                128,
                64,
                Tier::Host,
                MemoryRole::Weights,
                fixture.holder.clone(),
                128,
            ),
            |_| foreign_binding.allocate_owning(128, 64).map_err(Into::into),
            |_| Ok(AllocationPublication::exclusive(128, 128, 128)),
        )
        .unwrap_err();
    assert!(matches!(
        error,
        AllocationTransactionError::Step {
            stage: "allocate",
            ..
        }
    ));
    assert_eq!(fixture.governor.used(Tier::Host), 0);
    assert_eq!(fixture.manager.process_used(Tier::Host), 0);
    assert_eq!(
        foreign_registry
            .snapshot(foreign_mechanism)
            .unwrap()
            .live_allocations,
        0
    );
}

#[test]
fn quarantined_rollback_keeps_charge_and_prevents_reuse() {
    let fixture = Fixture::new(DeviceKey::HOST, 4096, unlimited());
    fixture.allocator.quarantine();
    let binding = fixture.manager.bind_registered(&fixture.mechanism).unwrap();
    let error = binding
        .allocate_with(
            AllocationRequest::managed(
                256,
                64,
                Tier::Host,
                MemoryRole::Weights,
                fixture.holder.clone(),
                256,
            ),
            |context| context.allocate_owning(),
            |_| Err(AllocationStepError::new("publish refused")),
        )
        .unwrap_err();
    let retained_address = match &error {
        AllocationTransactionError::RollbackQuarantined { outcome, .. } => {
            outcome.residual().expect("retained ownership").address
        }
        other => panic!("expected quarantined rollback, got {other:?}"),
    };
    assert!(matches!(
        error,
        AllocationTransactionError::RollbackQuarantined { .. }
    ));
    assert_eq!(fixture.governor.used(Tier::Host), 256);
    assert_eq!(fixture.manager.process_used(Tier::Host), 256);
    assert_eq!(fixture.allocator.releases.load(Ordering::Relaxed), 0);
    let snapshot = fixture.manager.snapshot().unwrap();
    assert_eq!(snapshot.allocations.len(), 1);
    assert_eq!(
        snapshot.allocations[0].state,
        onnx_runtime_memory_governor::ManagedAllocationState::Quarantined
    );
    assert_eq!(snapshot.allocations[0].charged_bytes, 256);

    let second = binding
        .allocate(
            AllocationRequest::managed(
                256,
                64,
                Tier::Host,
                MemoryRole::Weights,
                fixture.holder.clone(),
                256,
            ),
            AllocationPublication::exclusive(256, 256, 256),
        )
        .unwrap();
    assert_ne!(
        second.as_ptr().as_ptr() as usize,
        retained_address,
        "the retained allocation is never returned to a free list"
    );
    drop(second);
    fixture.allocator.cleanup();
}

#[test]
fn quarantine_never_uses_allocation_length_to_refund_rounded_charge() {
    let fixture = Fixture::new(DeviceKey::HOST, 4096, unlimited());
    fixture.allocator.quarantine();
    let allocation = fixture
        .manager
        .bind_registered(&fixture.mechanism)
        .unwrap()
        .allocate(
            AllocationRequest::managed(
                1,
                1,
                Tier::Host,
                MemoryRole::Weights,
                fixture.holder.clone(),
                64,
            ),
            AllocationPublication::exclusive(64, 64, 1),
        )
        .unwrap();
    let outcome = allocation.release_now().unwrap();
    assert!(outcome.is_quarantined());
    let snapshot = fixture.manager.snapshot().unwrap();
    assert_eq!(snapshot.allocations[0].charged_bytes, 64);
    assert_eq!(snapshot.allocations[0].process_reserved_bytes, 64);
    assert_eq!(snapshot.allocations[0].physical_bytes, Some(1));
    assert_eq!(fixture.governor.used(Tier::Host), 64);
    assert_eq!(fixture.manager.process_used(Tier::Host), 64);
    fixture.allocator.cleanup();
}

#[test]
fn unidentified_retained_ownership_never_receives_a_false_refund() {
    let fixture = Fixture::new(
        DeviceKey::HOST,
        1024,
        ProcessMemoryLimits {
            host_bytes: 1024,
            ..unlimited()
        },
    );
    let error = fixture
        .manager
        .bind_registered(&fixture.mechanism)
        .unwrap()
        .allocate_with(
            AllocationRequest::managed(
                128,
                64,
                Tier::Host,
                MemoryRole::Weights,
                fixture.holder.clone(),
                128,
            ),
            |_| {
                Err(AllocationStepError::retained(
                    "provider allocated physical memory but identity publication failed",
                ))
            },
            |_| unreachable!("allocation never published"),
        )
        .unwrap_err();
    assert!(matches!(
        error,
        AllocationTransactionError::UnidentifiedOwnershipRetained { .. }
    ));
    assert_eq!(fixture.governor.used(Tier::Host), 128);
    assert_eq!(fixture.manager.process_used(Tier::Host), 128);
    assert_eq!(fixture.allocator.releases.load(Ordering::Relaxed), 0);
}

#[test]
fn shared_aliases_do_not_double_count_physical_or_charged_bytes() {
    let fixture = Fixture::new(DeviceKey::HOST, 4096, unlimited());
    let binding = fixture.manager.bind_registered(&fixture.mechanism).unwrap();
    let shared = binding.new_shared_physical_identity().unwrap();
    let canonical = binding
        .allocate(
            AllocationRequest::managed(
                256,
                64,
                Tier::Host,
                MemoryRole::KvCache,
                fixture.holder.clone(),
                256,
            ),
            AllocationPublication::exclusive(256, 256, 256).with_shared_physical(shared),
        )
        .unwrap();
    let alias = binding
        .allocate(
            AllocationRequest::authority_managed(
                256,
                64,
                Tier::Host,
                MemoryRole::KvCache,
                fixture.holder.clone(),
                0,
            ),
            AllocationPublication {
                charged_bytes: 0,
                process_reserved_bytes: 0,
                physical_bytes: Some(256),
                mapped_bytes: Some(256),
                unattributed_bytes: 0,
                shared_physical: Some(shared),
            },
        )
        .unwrap();
    let snapshot = fixture.manager.snapshot().unwrap();
    assert_eq!(snapshot.charged_bytes, 256);
    assert_eq!(snapshot.known_physical_bytes, 256);
    assert_eq!(snapshot.mapped_bytes, 512);
    assert_eq!(snapshot.unattributed_bytes, 0);
    canonical.release_now().unwrap();
    alias.release_now().unwrap();
}

#[test]
fn compatibility_bytes_are_labeled_unattributed() {
    let manager = ProcessMemoryManager::new().unwrap();
    let context = manager
        .register_provider_context(DeviceKey::HOST, "compat context", Arc::new(Pin))
        .unwrap();
    let authority = manager
        .register_compatibility_authority(DeviceKey::HOST, "compat", Arc::new(Pin))
        .unwrap();
    let holder = manager
        .register_holder(&authority, "compat caller", None)
        .unwrap();
    let allocator = TestAllocator::new(DeviceKey::HOST);
    let mechanism = manager
        .register_allocator(
            &context,
            &authority,
            "third party",
            Arc::clone(&allocator) as Arc<dyn DeviceAllocator>,
        )
        .unwrap();
    let allocation = manager
        .bind_registered(&mechanism)
        .unwrap()
        .allocate(
            AllocationRequest::compatibility(320, 64, Tier::Host, MemoryRole::Activation, holder),
            AllocationPublication::compatibility(320, 320),
        )
        .unwrap();
    let snapshot = manager.snapshot().unwrap();
    assert_eq!(snapshot.charged_bytes, 0);
    assert_eq!(snapshot.known_physical_bytes, 320);
    assert_eq!(snapshot.unattributed_bytes, 320);
    assert_eq!(
        snapshot.allocations[0].charge_mode,
        AllocationChargeMode::Compatibility
    );
    allocation.release_now().unwrap();
}

#[test]
fn concurrent_process_grants_never_oversubscribe() {
    let fixture = Fixture::new(
        DeviceKey::HOST,
        1 << 20,
        ProcessMemoryLimits {
            host_bytes: 1024,
            ..unlimited()
        },
    );
    let binding = fixture.manager.bind_registered(&fixture.mechanism).unwrap();
    let start = Arc::new(Barrier::new(17));
    let ready = Arc::new(Barrier::new(17));
    let release = Arc::new(Barrier::new(17));
    let successes = Arc::new(AtomicUsize::new(0));
    let threads = (0..16)
        .map(|_| {
            let binding = binding.clone();
            let holder = fixture.holder.clone();
            let start = Arc::clone(&start);
            let ready = Arc::clone(&ready);
            let release = Arc::clone(&release);
            let successes = Arc::clone(&successes);
            thread::spawn(move || {
                start.wait();
                let allocation = binding
                    .allocate(
                        AllocationRequest::managed(
                            128,
                            64,
                            Tier::Host,
                            MemoryRole::Workspace { step_scoped: true },
                            holder,
                            128,
                        ),
                        AllocationPublication::exclusive(128, 128, 128),
                    )
                    .ok();
                if allocation.is_some() {
                    successes.fetch_add(1, Ordering::Relaxed);
                }
                ready.wait();
                release.wait();
                if let Some(allocation) = allocation {
                    allocation.release_now().unwrap();
                }
            })
        })
        .collect::<Vec<_>>();
    start.wait();
    ready.wait();
    assert_eq!(successes.load(Ordering::Relaxed), 8);
    assert_eq!(fixture.manager.process_used(Tier::Host), 1024);
    release.wait();
    for thread in threads {
        thread.join().unwrap();
    }
    assert_eq!(fixture.manager.process_used(Tier::Host), 0);
}

#[test]
fn process_quota_checked_arithmetic_refuses_wraparound() {
    let manager = ProcessMemoryManager::with_limits(ProcessMemoryLimits {
        host_bytes: u64::MAX,
        ..unlimited()
    })
    .unwrap();
    let context = manager
        .register_provider_context(DeviceKey::HOST, "context", Arc::new(Pin))
        .unwrap();
    let authority = manager
        .register_compatibility_authority(DeviceKey::HOST, "compat", Arc::new(Pin))
        .unwrap();
    let holder = manager.register_holder(&authority, "holder", None).unwrap();
    let allocator = TestAllocator::new(DeviceKey::HOST);
    let mechanism = manager
        .register_allocator(
            &context,
            &authority,
            "allocator",
            allocator as Arc<dyn DeviceAllocator>,
        )
        .unwrap();
    let binding = manager.bind_registered(&mechanism).unwrap();
    let first = binding
        .allocate(
            AllocationRequest::compatibility(
                1,
                1,
                Tier::Host,
                MemoryRole::Activation,
                holder.clone(),
            )
            .with_process_reservation(u64::MAX),
            AllocationPublication {
                charged_bytes: 0,
                process_reserved_bytes: u64::MAX,
                physical_bytes: Some(1),
                mapped_bytes: Some(1),
                unattributed_bytes: 1,
                shared_physical: None,
            },
        )
        .unwrap();
    let error = binding
        .allocate(
            AllocationRequest::compatibility(1, 1, Tier::Host, MemoryRole::Activation, holder)
                .with_process_reservation(1),
            AllocationPublication {
                charged_bytes: 0,
                process_reserved_bytes: 1,
                physical_bytes: Some(1),
                mapped_bytes: Some(1),
                unattributed_bytes: 1,
                shared_physical: None,
            },
        )
        .unwrap_err();
    assert!(matches!(
        error,
        AllocationTransactionError::Memory(MemoryError::InvalidRequest { .. })
            | AllocationTransactionError::Memory(MemoryError::TierExhausted { .. })
    ));
    first.release_now().unwrap();
}

#[test]
fn authority_delegations_cannot_double_grant_process_capacity() {
    let manager = ProcessMemoryManager::with_limits(ProcessMemoryLimits {
        device_bytes: 1024,
        ..unlimited()
    })
    .unwrap();
    let mut authorities = Vec::new();
    for index in 0..2 {
        let device = DeviceKey::device(index);
        let context = manager
            .register_provider_context(device, format!("context:{index}"), Arc::new(Pin))
            .unwrap();
        let governor = Arc::new(LedgerGovernor::new(LeaseLedger::new_for_device(
            device, 600, 0, 0,
        )));
        let authority = manager
            .register_authority(
                device,
                format!("authority:{index}"),
                Arc::new(Pin),
                governor as Arc<dyn MemoryGovernor + Send + Sync>,
            )
            .unwrap();
        let allocator = TestAllocator::new(device);
        let mechanism = manager
            .register_allocator(
                &context,
                &authority,
                format!("allocator:{index}"),
                allocator as Arc<dyn DeviceAllocator>,
            )
            .unwrap();
        let holder = manager
            .register_holder(&authority, format!("holder:{index}"), None)
            .unwrap();
        authorities.push((authority, mechanism, holder));
    }

    manager
        .delegate_authority_capacity(&authorities[0].0, Tier::Device, 600)
        .unwrap();
    assert!(authorities[0].0.has_process_delegation(Tier::Device));
    let error = manager
        .delegate_authority_capacity(&authorities[1].0, Tier::Device, 600)
        .unwrap_err();
    assert!(matches!(
        error,
        AllocationTransactionError::Memory(MemoryError::TierExhausted { .. })
    ));
    assert_eq!(manager.process_used(Tier::Device), 600);
    assert!(manager.set_process_limit(Tier::Device, 599).is_err());
    assert_eq!(manager.process_limit(Tier::Device), 1024);

    let mut request = AllocationRequest::managed(
        128,
        64,
        Tier::Device,
        MemoryRole::KvCache,
        authorities[0].2.clone(),
        128,
    );
    request.process_reserve_bytes = 0;
    let allocation = manager
        .bind_registered(&authorities[0].1)
        .unwrap()
        .allocate(
            request,
            AllocationPublication {
                charged_bytes: 128,
                process_reserved_bytes: 0,
                physical_bytes: Some(128),
                mapped_bytes: Some(128),
                unattributed_bytes: 0,
                shared_physical: None,
            },
        )
        .unwrap();
    assert_eq!(manager.process_used(Tier::Device), 600);
    allocation.release_now().unwrap();
    assert_eq!(manager.process_used(Tier::Device), 600);
}

#[test]
fn authority_managed_charge_requires_parent_delegation_under_a_finite_limit() {
    let fixture = Fixture::new(
        DeviceKey::HOST,
        512,
        ProcessMemoryLimits {
            host_bytes: 512,
            ..unlimited()
        },
    );
    let binding = fixture.manager.bind_registered(&fixture.mechanism).unwrap();
    let request = || {
        AllocationRequest::authority_managed(
            128,
            64,
            Tier::Host,
            MemoryRole::Workspace { step_scoped: false },
            fixture.holder.clone(),
            128,
        )
    };
    let error = binding
        .allocate(
            request(),
            AllocationPublication {
                charged_bytes: 128,
                process_reserved_bytes: 0,
                physical_bytes: None,
                mapped_bytes: Some(128),
                unattributed_bytes: 0,
                shared_physical: None,
            },
        )
        .unwrap_err();
    assert!(matches!(
        error,
        AllocationTransactionError::InsufficientProcessCoverage { .. }
    ));
    let mut undercovered = request();
    undercovered.process_reserve_bytes = 1;
    let error = binding
        .allocate(
            undercovered,
            AllocationPublication {
                charged_bytes: 128,
                process_reserved_bytes: 1,
                physical_bytes: None,
                mapped_bytes: Some(128),
                unattributed_bytes: 0,
                shared_physical: None,
            },
        )
        .unwrap_err();
    assert!(matches!(
        error,
        AllocationTransactionError::InsufficientProcessCoverage {
            required: 128,
            reserved: 1,
            ..
        }
    ));
    fixture
        .manager
        .delegate_authority_capacity(&fixture.authority, Tier::Host, 512)
        .unwrap();
    fixture.governor.ledger().set_limit(Tier::Host, 600);
    let error = binding
        .allocate(
            request(),
            AllocationPublication {
                charged_bytes: 128,
                process_reserved_bytes: 0,
                physical_bytes: None,
                mapped_bytes: Some(128),
                unattributed_bytes: 0,
                shared_physical: None,
            },
        )
        .unwrap_err();
    assert!(matches!(
        error,
        AllocationTransactionError::StaleProcessDelegation { .. }
    ));
    fixture.governor.ledger().set_limit(Tier::Host, 512);
    let allocation = binding
        .allocate(
            request(),
            AllocationPublication {
                charged_bytes: 128,
                process_reserved_bytes: 0,
                physical_bytes: None,
                mapped_bytes: Some(128),
                unattributed_bytes: 0,
                shared_physical: None,
            },
        )
        .unwrap();
    assert_eq!(fixture.manager.process_used(Tier::Host), 512);
    allocation.release_now().unwrap();
    assert_eq!(fixture.manager.process_used(Tier::Host), 512);
}

#[derive(Debug)]
struct ReentrantReclaimer {
    manager: ProcessMemoryManager,
    allocation: Mutex<Option<onnx_runtime_memory_governor::ManagedAllocation>>,
    calls: AtomicUsize,
}

#[derive(Debug)]
struct LossListener {
    manager: ProcessMemoryManager,
    calls: AtomicUsize,
}

impl onnx_runtime_memory_governor::DeviceLossListener for LossListener {
    fn mark_device_lost(&self, _reason: &str) {
        let _ = self
            .manager
            .snapshot()
            .expect("device-loss reentrant snapshot");
        self.calls.fetch_add(1, Ordering::Relaxed);
    }
}

#[test]
fn device_loss_broadcasts_to_all_sibling_contexts_without_manager_locks() {
    let manager = ProcessMemoryManager::new().unwrap();
    let first = Arc::new(LossListener {
        manager: manager.clone(),
        calls: AtomicUsize::new(0),
    });
    let second = Arc::new(LossListener {
        manager: manager.clone(),
        calls: AtomicUsize::new(0),
    });
    let first_listener: Arc<dyn onnx_runtime_memory_governor::DeviceLossListener> = first.clone();
    let second_listener: Arc<dyn onnx_runtime_memory_governor::DeviceLossListener> = second.clone();
    let first_generation = manager
        .register_device_loss_listener(DeviceKey::device(0), &first_listener)
        .unwrap();
    manager
        .register_device_loss_listener(DeviceKey::device(0), &second_listener)
        .unwrap();
    manager
        .invalidate_device(DeviceKey::device(0), "shared device loss")
        .unwrap();
    assert_eq!(first.calls.load(Ordering::Relaxed), 1);
    assert_eq!(second.calls.load(Ordering::Relaxed), 1);
    assert!(matches!(
        manager.finish_device_registration(DeviceKey::device(0), first_generation),
        Err(AllocationTransactionError::DeviceRegistrationLost { .. })
    ));
    let late = Arc::new(LossListener {
        manager: manager.clone(),
        calls: AtomicUsize::new(0),
    });
    let late_listener: Arc<dyn onnx_runtime_memory_governor::DeviceLossListener> = late.clone();
    let late_generation = manager
        .register_device_loss_listener(DeviceKey::device(0), &late_listener)
        .unwrap();
    assert_eq!(late.calls.load(Ordering::Relaxed), 1);
    assert!(
        manager
            .finish_device_registration(DeviceKey::device(0), late_generation)
            .is_err()
    );
}

impl onnx_runtime_memory_governor::PressureResponder for ReentrantReclaimer {
    fn on_pressure(&self, _tier: Tier, _want: u64) -> u64 {
        self.calls.fetch_add(1, Ordering::Relaxed);
        let _ = self.manager.snapshot().expect("reentrant snapshot");
        self.allocation
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .take()
            .map_or(0, |allocation| {
                let bytes = allocation.len() as u64;
                allocation.release_now().expect("holder release");
                bytes
            })
    }
}

#[test]
fn pressure_callback_is_reentrant_and_runs_without_manager_locks() {
    let fixture = Fixture::new(
        DeviceKey::HOST,
        256,
        ProcessMemoryLimits {
            host_bytes: 256,
            ..unlimited()
        },
    );
    let reclaimer = Arc::new(ReentrantReclaimer {
        manager: fixture.manager.clone(),
        allocation: Mutex::new(None),
        calls: AtomicUsize::new(0),
    });
    let reclaiming_holder =
        fixture
            .manager
            .register_holder(
                &fixture.authority,
                "reclaiming holder",
                Some(Arc::clone(&reclaimer)
                    as Arc<dyn onnx_runtime_memory_governor::PressureResponder>),
            )
            .unwrap();
    let requester = fixture
        .manager
        .register_holder(&fixture.authority, "requester", None)
        .unwrap();
    let binding = fixture.manager.bind_registered(&fixture.mechanism).unwrap();
    let held = binding
        .allocate(
            AllocationRequest::managed(
                256,
                64,
                Tier::Host,
                MemoryRole::Weights,
                reclaiming_holder,
                256,
            ),
            AllocationPublication::exclusive(256, 256, 256),
        )
        .unwrap();
    *reclaimer
        .allocation
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(held);
    let replacement = binding
        .allocate(
            AllocationRequest::managed(256, 64, Tier::Host, MemoryRole::KvCache, requester, 256),
            AllocationPublication::exclusive(256, 256, 256),
        )
        .expect("pressure made room");
    assert_eq!(reclaimer.calls.load(Ordering::Relaxed), 1);
    replacement.release_now().unwrap();
}

#[test]
fn device_loss_retains_charge_until_context_termination() {
    let fixture = Fixture::new(DeviceKey::HOST, 4096, unlimited());
    let context_scope = fixture
        .manager
        .bind_registered(&fixture.mechanism)
        .unwrap()
        .context_scope();
    let allocation = fixture.allocate(512);
    fixture
        .manager
        .invalidate_device(DeviceKey::HOST, "injected device loss")
        .unwrap();
    assert!(matches!(
        context_scope.enter(),
        Err(AllocationTransactionError::TerminatedContext(_))
    ));
    drop(allocation);
    assert_eq!(fixture.governor.used(Tier::Host), 512);
    let lost = fixture.manager.snapshot().unwrap();
    assert_eq!(lost.allocations.len(), 1);
    assert_eq!(
        lost.allocations[0].state,
        onnx_runtime_memory_governor::ManagedAllocationState::DeviceLost
    );
    fixture
        .manager
        .confirm_context_terminated(&fixture.context)
        .unwrap();
    fixture
        .manager
        .confirm_context_terminated(&fixture.context)
        .expect("duplicate confirmation stays terminal and idempotent");
    assert_eq!(fixture.governor.used(Tier::Host), 0);
    assert!(fixture.manager.snapshot().unwrap().allocations.is_empty());
    fixture.allocator.cleanup();
}

#[test]
fn context_termination_waits_for_manager_publication_and_rejects_new_work() {
    let fixture = Fixture::new(DeviceKey::HOST, 4096, unlimited());
    let binding = fixture.manager.bind_registered(&fixture.mechanism).unwrap();
    let holder = fixture.holder.clone();
    let entered = Arc::new(Barrier::new(2));
    let resume = Arc::new(Barrier::new(2));
    let transaction = {
        let entered = Arc::clone(&entered);
        let resume = Arc::clone(&resume);
        thread::spawn(move || {
            binding.allocate_with(
                AllocationRequest::managed(
                    256,
                    64,
                    Tier::Host,
                    MemoryRole::Workspace { step_scoped: false },
                    holder,
                    256,
                ),
                |context| context.allocate_owning(),
                |_| {
                    entered.wait();
                    resume.wait();
                    Ok(AllocationPublication::exclusive(256, 256, 256))
                },
            )
        })
    };
    entered.wait();
    fixture
        .manager
        .invalidate_device(DeviceKey::HOST, "loss during publication")
        .unwrap();
    let (done_tx, done_rx) = std::sync::mpsc::channel();
    let manager = fixture.manager.clone();
    let context = fixture.context.clone();
    let confirmation = thread::spawn(move || {
        let result = manager.confirm_context_terminated(&context);
        done_tx.send(()).unwrap();
        result
    });
    assert!(
        done_rx
            .recv_timeout(std::time::Duration::from_millis(20))
            .is_err(),
        "termination must wait for manager publication"
    );
    resume.wait();
    let allocation = transaction.join().unwrap().unwrap();
    done_rx
        .recv_timeout(std::time::Duration::from_secs(1))
        .expect("termination completed after publication");
    confirmation.join().unwrap().unwrap();
    assert_eq!(fixture.governor.used(Tier::Host), 0);
    let error = fixture
        .manager
        .register_allocator(
            &fixture.context,
            &fixture.authority,
            "late mechanism",
            TestAllocator::new(DeviceKey::HOST) as Arc<dyn DeviceAllocator>,
        )
        .unwrap_err();
    assert!(matches!(
        error,
        AllocationTransactionError::TerminatedContext(_)
    ));
    drop(allocation);
    fixture.allocator.cleanup();
}

#[test]
fn manager_can_drop_before_a_live_allocation_settles() {
    let fixture = Fixture::new(DeviceKey::HOST, 4096, unlimited());
    let allocation = fixture.allocate(128);
    let governor = Arc::clone(&fixture.governor);
    let allocator = Arc::clone(&fixture.allocator);
    drop(fixture);
    assert_eq!(governor.used(Tier::Host), 128);
    allocation.release_now().unwrap();
    assert_eq!(governor.used(Tier::Host), 0);
    assert_eq!(allocator.releases.load(Ordering::Relaxed), 1);
}

#[test]
fn quiescent_registration_teardown_releases_registry_pins() {
    let fixture = Fixture::new(DeviceKey::HOST, 4096, unlimited());
    let allocation = fixture.allocate(64);
    allocation.release_now().unwrap();
    fixture.manager.retire(&fixture.mechanism).unwrap();
    fixture
        .manager
        .remove_mechanism(&fixture.mechanism)
        .unwrap();
    fixture.manager.unregister_holder(&fixture.holder).unwrap();
    fixture
        .manager
        .remove_provider_context(&fixture.context)
        .unwrap();
    fixture
        .manager
        .remove_authority(&fixture.authority)
        .unwrap();
    let snapshot = fixture.manager.snapshot().unwrap();
    assert_eq!(snapshot.authority_count, 0);
    assert!(snapshot.mechanism_snapshots.is_empty());
    assert!(snapshot.allocations.is_empty());
}

#[test]
fn manager_can_drop_with_a_prepared_release_still_live() {
    let fixture = Fixture::new(DeviceKey::HOST, 4096, unlimited());
    let prepared = fixture.allocate(192).prepare_release().unwrap();
    let governor = Arc::clone(&fixture.governor);
    let allocator = Arc::clone(&fixture.allocator);
    drop(fixture);
    assert_eq!(governor.used(Tier::Host), 192);
    assert!(prepared.execute().is_complete());
    assert_eq!(governor.used(Tier::Host), 0);
    assert_eq!(allocator.releases.load(Ordering::Relaxed), 1);
}

#[test]
fn abandoning_a_prepared_release_quarantines_without_refund() {
    let fixture = Fixture::new(DeviceKey::HOST, 4096, unlimited());
    let prepared = fixture.allocate(224).prepare_release().unwrap();
    assert_eq!(
        fixture.manager.snapshot().unwrap().allocations[0].state,
        onnx_runtime_memory_governor::ManagedAllocationState::Queued
    );
    drop(prepared);
    let snapshot = fixture.manager.snapshot().unwrap();
    assert_eq!(snapshot.allocations.len(), 1);
    assert_eq!(
        snapshot.allocations[0].state,
        onnx_runtime_memory_governor::ManagedAllocationState::Quarantined
    );
    assert_eq!(snapshot.allocations[0].charged_bytes, 224);
    assert_eq!(fixture.governor.used(Tier::Host), 224);
    assert_eq!(fixture.allocator.releases.load(Ordering::Relaxed), 0);
    fixture.allocator.cleanup();
}

#[test]
fn deterministic_switch_and_teardown_stress() {
    for round in 0..64 {
        let fixture = Fixture::new(DeviceKey::HOST, 8192, unlimited());
        let mut allocations = Vec::new();
        for index in 0..16 {
            let allocation = fixture.allocate(64 + (index % 4) * 64);
            if (round + index) % 3 == 0 {
                allocation.release_now().unwrap();
            } else {
                allocations.push(allocation);
            }
        }
        for allocation in allocations.into_iter().rev() {
            allocation.release_now().unwrap();
        }
        assert_eq!(fixture.governor.used(Tier::Host), 0);
        assert_eq!(fixture.manager.process_used(Tier::Host), 0);
        assert!(fixture.manager.snapshot().unwrap().allocations.is_empty());
    }
}

#[test]
fn erased_third_party_allocator_adapter_remains_supported() {
    let manager = ProcessMemoryManager::new().unwrap();
    let context = manager
        .register_provider_context(DeviceKey::HOST, "third-party context", Arc::new(Pin))
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
            "third-party authority",
            Arc::new(Pin),
            governor as Arc<dyn MemoryGovernor + Send + Sync>,
        )
        .unwrap();
    let holder = manager
        .register_holder(&authority, "third-party caller", None)
        .unwrap();
    let concrete = TestAllocator::new(DeviceKey::HOST);
    let erased: Arc<dyn DeviceAllocator> = concrete.clone();
    let mechanism = manager
        .register_allocator(&context, &authority, "erased allocator", erased)
        .unwrap();
    let allocation = manager
        .bind_registered(&mechanism)
        .unwrap()
        .allocate(
            AllocationRequest::managed(128, 64, Tier::Host, MemoryRole::Weights, holder, 128),
            AllocationPublication::exclusive(128, 128, 128),
        )
        .unwrap();
    allocation.release_now().unwrap();
    assert_eq!(concrete.allocations.load(Ordering::Relaxed), 1);
    assert_eq!(concrete.releases.load(Ordering::Relaxed), 1);
}

#[test]
fn holder_identity_cannot_cross_authority_boundaries() {
    let first = Fixture::new(DeviceKey::HOST, 1024, unlimited());
    let second_manager = ProcessMemoryManager::new().unwrap();
    let second_context = second_manager
        .register_provider_context(DeviceKey::HOST, "second", Arc::new(Pin))
        .unwrap();
    let second_governor = Arc::new(LedgerGovernor::new(LeaseLedger::new_for_device(
        DeviceKey::HOST,
        0,
        1024,
        0,
    )));
    let second_authority = second_manager
        .register_authority(
            DeviceKey::HOST,
            "second",
            Arc::new(Pin),
            second_governor as Arc<dyn MemoryGovernor + Send + Sync>,
        )
        .unwrap();
    let second_allocator = TestAllocator::new(DeviceKey::HOST);
    let second_mechanism = second_manager
        .register_allocator(
            &second_context,
            &second_authority,
            "second",
            second_allocator as Arc<dyn DeviceAllocator>,
        )
        .unwrap();
    let error = second_manager
        .bind_registered(&second_mechanism)
        .unwrap()
        .allocate(
            AllocationRequest::managed(64, 64, Tier::Host, MemoryRole::Weights, first.holder, 64),
            AllocationPublication::exclusive(64, 64, 64),
        )
        .unwrap_err();
    assert!(matches!(
        error,
        AllocationTransactionError::ForeignHandle("holder")
    ));
}

#[test]
fn holder_ids_are_manager_issued_not_caller_selected() {
    let fixture = Fixture::new(DeviceKey::HOST, 1024, unlimited());
    let second = fixture
        .manager
        .register_holder(&fixture.authority, "second", None)
        .unwrap();
    assert_ne!(fixture.holder.id(), second.id());
    assert_ne!(fixture.holder.id(), HolderId::new(0));
}
