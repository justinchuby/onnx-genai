use std::any::Any;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Barrier, Mutex, Weak};
use std::thread;

use onnx_runtime_memory_api::{
    AllocationCommitRange, BindingError, BindingRegistry, BindingResource, DeviceAllocator,
    DeviceKey, HostAllocator, MechanismCoherence, MechanismLifecycle, MemoryError,
    RegisteredMechanism, SharedDevicePrefix, SharedMapping, SharedPrefixCommitInfo, VirtualBacking,
};

#[derive(Debug)]
struct DropProbe(Arc<AtomicUsize>);

impl Drop for DropProbe {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

fn resource(probe: &Arc<DropProbe>) -> Arc<dyn BindingResource> {
    Arc::clone(probe) as Arc<dyn BindingResource>
}

#[derive(Debug)]
struct TestPrefix {
    bytes: usize,
}

impl SharedDevicePrefix for TestPrefix {
    fn device_ptr(&self) -> u64 {
        0x1000
    }

    fn committed_physical_bytes(&self) -> u64 {
        self.bytes as u64
    }

    fn mapped_bytes(&self) -> usize {
        self.bytes
    }

    fn requested_bytes(&self) -> usize {
        self.bytes
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[derive(Debug)]
struct ContextDropFlag {
    released: Arc<AtomicBool>,
}

impl Drop for ContextDropFlag {
    fn drop(&mut self) {
        self.released.store(true, Ordering::SeqCst);
    }
}

#[derive(Debug)]
struct ObservedPrefix {
    bytes: usize,
    context_released: Arc<AtomicBool>,
    dropped: Arc<AtomicBool>,
}

impl Drop for ObservedPrefix {
    fn drop(&mut self) {
        assert!(
            !self.context_released.load(Ordering::SeqCst),
            "shared-prefix teardown ran after its provider context pin was released"
        );
        self.dropped.store(true, Ordering::SeqCst);
    }
}

impl SharedDevicePrefix for ObservedPrefix {
    fn device_ptr(&self) -> u64 {
        0x2000
    }

    fn committed_physical_bytes(&self) -> u64 {
        self.bytes as u64
    }

    fn mapped_bytes(&self) -> usize {
        self.bytes
    }

    fn requested_bytes(&self) -> usize {
        self.bytes
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[derive(Debug)]
struct DropOrderMechanism {
    context_released: Arc<AtomicBool>,
    prefix_dropped: Arc<AtomicBool>,
}

impl DeviceAllocator for DropOrderMechanism {
    fn allocate(&self, bytes: usize, align: usize) -> Result<NonNull<u8>, MemoryError> {
        HostAllocator.allocate(bytes, align)
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, bytes: usize, align: usize) {
        // SAFETY: forwarded unchanged from this method's contract.
        unsafe { HostAllocator.deallocate(ptr, bytes, align) };
    }

    fn device(&self) -> DeviceKey {
        DeviceKey::HOST
    }

    fn as_shared_mapping(&self) -> Option<&dyn SharedMapping> {
        Some(self)
    }
}

impl SharedMapping for DropOrderMechanism {
    fn create_shared_prefix(
        &self,
        bytes: usize,
    ) -> Result<Box<dyn SharedDevicePrefix>, MemoryError> {
        Ok(Box::new(ObservedPrefix {
            bytes,
            context_released: Arc::clone(&self.context_released),
            dropped: Arc::clone(&self.prefix_dropped),
        }))
    }

    fn incremental_owned_bytes_for_shared_prefix(
        &self,
        prefix: &dyn SharedDevicePrefix,
    ) -> Result<u64, MemoryError> {
        Ok(prefix.committed_physical_bytes())
    }

    fn commit_shared_prefix(
        &self,
        _prefix: &dyn SharedDevicePrefix,
        _ptr: NonNull<u8>,
        _allocation_bytes: usize,
        _byte_offset: usize,
    ) -> Result<SharedPrefixCommitInfo, MemoryError> {
        Ok(SharedPrefixCommitInfo::default())
    }
}

/// A provider-context or authority resource that records when it is released.
#[derive(Debug)]
struct PinnedResource {
    label: &'static str,
    log: Arc<Mutex<Vec<String>>>,
}

impl Drop for PinnedResource {
    fn drop(&mut self) {
        self.log
            .lock()
            .expect("drop order log")
            .push(format!("{} released", self.label));
    }
}

/// A third-party allocator that releases device state from `Drop`.
///
/// Real provider allocators (CUDA pools, provider-library handles) need their
/// provider context alive to free anything, so this observer records whether the
/// registry's context and authority pins were still alive when it ran. The weak
/// references never keep the resources alive themselves.
#[derive(Debug)]
struct ContextDependentAllocator {
    log: Arc<Mutex<Vec<String>>>,
    context: Weak<PinnedResource>,
    authority: Weak<PinnedResource>,
}

impl Drop for ContextDependentAllocator {
    fn drop(&mut self) {
        let context_alive = self.context.upgrade().is_some();
        let authority_alive = self.authority.upgrade().is_some();
        self.log.lock().expect("drop order log").push(format!(
            "allocator released (context alive: {context_alive}, authority alive: {authority_alive})"
        ));
    }
}

impl DeviceAllocator for ContextDependentAllocator {
    fn allocate(&self, bytes: usize, align: usize) -> Result<NonNull<u8>, MemoryError> {
        HostAllocator.allocate(bytes, align)
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, bytes: usize, align: usize) {
        // SAFETY: forwarded unchanged from this method's contract.
        unsafe { HostAllocator.deallocate(ptr, bytes, align) };
    }

    fn device(&self) -> DeviceKey {
        DeviceKey::HOST
    }
}

#[derive(Debug)]
struct TestMechanism {
    device: DeviceKey,
    allocations: AtomicUsize,
    frees: AtomicUsize,
    virtual_operations: AtomicUsize,
    shared_operations: AtomicUsize,
}

impl TestMechanism {
    fn new(device: DeviceKey) -> Self {
        Self {
            device,
            allocations: AtomicUsize::new(0),
            frees: AtomicUsize::new(0),
            virtual_operations: AtomicUsize::new(0),
            shared_operations: AtomicUsize::new(0),
        }
    }
}

impl DeviceAllocator for TestMechanism {
    fn allocate(&self, bytes: usize, align: usize) -> Result<NonNull<u8>, MemoryError> {
        self.allocations.fetch_add(1, Ordering::SeqCst);
        HostAllocator.allocate(bytes, align)
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, bytes: usize, align: usize) {
        self.frees.fetch_add(1, Ordering::SeqCst);
        // SAFETY: forwarded unchanged from this method's contract.
        unsafe { HostAllocator.deallocate(ptr, bytes, align) };
    }

    fn device(&self) -> DeviceKey {
        self.device
    }

    fn as_virtual_backing(&self) -> Option<&dyn VirtualBacking> {
        Some(self)
    }

    fn as_shared_mapping(&self) -> Option<&dyn SharedMapping> {
        Some(self)
    }
}

impl VirtualBacking for TestMechanism {
    fn allocate_committed(
        &self,
        bytes: usize,
        align: usize,
        _committed_ranges: &[std::ops::Range<usize>],
    ) -> Result<NonNull<u8>, MemoryError> {
        self.virtual_operations.fetch_add(1, Ordering::SeqCst);
        self.allocate(bytes, align)
    }

    fn commit_allocation_range(
        &self,
        _ptr: NonNull<u8>,
        _allocation_bytes: usize,
        _align: usize,
        _offset: usize,
        _bytes: usize,
    ) -> Result<(), MemoryError> {
        self.virtual_operations.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }

    fn mapped_bytes_for_allocation_ranges(
        &self,
        ranges: &[AllocationCommitRange],
    ) -> Result<u64, MemoryError> {
        self.virtual_operations.fetch_add(1, Ordering::SeqCst);
        Ok(ranges.iter().map(|range| range.bytes as u64).sum())
    }

    fn mapped_bytes_for_allocation(&self, bytes: usize, _align: usize) -> Result<u64, MemoryError> {
        self.virtual_operations.fetch_add(1, Ordering::SeqCst);
        Ok(bytes as u64)
    }

    fn decommit_allocation_range(
        &self,
        _ptr: NonNull<u8>,
        _allocation_bytes: usize,
        _align: usize,
        _offset: usize,
        bytes: usize,
    ) -> Result<u64, MemoryError> {
        self.virtual_operations.fetch_add(1, Ordering::SeqCst);
        Ok(bytes as u64)
    }

    fn allocation_committed_bytes(
        &self,
        _ptr: NonNull<u8>,
        allocation_bytes: usize,
        _align: usize,
    ) -> usize {
        self.virtual_operations.fetch_add(1, Ordering::SeqCst);
        allocation_bytes
    }
}

impl SharedMapping for TestMechanism {
    fn create_shared_prefix(
        &self,
        bytes: usize,
    ) -> Result<Box<dyn SharedDevicePrefix>, MemoryError> {
        self.shared_operations.fetch_add(1, Ordering::SeqCst);
        Ok(Box::new(TestPrefix { bytes }))
    }

    fn incremental_owned_bytes_for_shared_prefix(
        &self,
        prefix: &dyn SharedDevicePrefix,
    ) -> Result<u64, MemoryError> {
        self.shared_operations.fetch_add(1, Ordering::SeqCst);
        Ok(prefix.committed_physical_bytes())
    }

    fn commit_shared_prefix(
        &self,
        _prefix: &dyn SharedDevicePrefix,
        _ptr: NonNull<u8>,
        _allocation_bytes: usize,
        _byte_offset: usize,
    ) -> Result<SharedPrefixCommitInfo, MemoryError> {
        self.shared_operations.fetch_add(1, Ordering::SeqCst);
        Ok(SharedPrefixCommitInfo {
            additional_owned_bytes: 0,
            newly_mapped_bytes: 64,
            granules: 1,
        })
    }
}

fn registered_test_mechanism(
    registry: &BindingRegistry,
    device: DeviceKey,
    allocator: Arc<TestMechanism>,
) -> (
    onnx_runtime_memory_api::RegisteredProviderContext,
    onnx_runtime_memory_api::RegisteredAuthority,
    RegisteredMechanism,
) {
    let context = registry
        .register_provider_context(device, Arc::new(()) as Arc<dyn BindingResource>)
        .expect("context registration");
    let authority = registry
        .register_authority(device, Arc::new(()) as Arc<dyn BindingResource>)
        .expect("authority registration");
    let mechanism = registry
        .register_allocator(context, authority, allocator as Arc<dyn DeviceAllocator>)
        .expect("mechanism registration");
    (context, authority, mechanism)
}

#[test]
fn binding_identity_names_every_registered_axis_and_opaque_generation() {
    let registry = BindingRegistry::new().unwrap();
    let allocator = Arc::new(TestMechanism::new(DeviceKey::HOST));
    let (context, authority, mechanism) =
        registered_test_mechanism(&registry, DeviceKey::HOST, allocator);

    let first = registry.bind(DeviceKey::HOST).unwrap();
    let second = registry.bind(DeviceKey::HOST).unwrap();
    assert_eq!(first.identity().device(), DeviceKey::HOST);
    assert_eq!(first.identity().mechanism(), mechanism.identity());
    assert_eq!(first.identity().provider_context(), context.identity());
    assert_eq!(first.identity().authority(), authority.identity());
    assert_ne!(first.identity().id(), second.identity().id());
    assert_ne!(
        first.identity().generation(),
        second.identity().generation()
    );
}

#[test]
fn context_survives_frontend_drop_until_bound_metadata_retires() {
    let context_drops = Arc::new(AtomicUsize::new(0));
    let authority_drops = Arc::new(AtomicUsize::new(0));
    let context_owner = Arc::new(DropProbe(Arc::clone(&context_drops)));
    let authority_owner = Arc::new(DropProbe(Arc::clone(&authority_drops)));
    let registry = BindingRegistry::new().unwrap();
    let context = registry
        .register_provider_context(DeviceKey::HOST, resource(&context_owner))
        .unwrap();
    let authority = registry
        .register_authority(DeviceKey::HOST, resource(&authority_owner))
        .unwrap();
    let allocator = Arc::new(TestMechanism::new(DeviceKey::HOST));
    registry
        .register_allocator(
            context,
            authority,
            allocator.clone() as Arc<dyn DeviceAllocator>,
        )
        .unwrap();
    let binding = registry.bind(DeviceKey::HOST).unwrap();
    let allocation = binding.allocate(64, 16).unwrap();

    drop(context_owner);
    drop(authority_owner);
    drop(registry);
    assert_eq!(context_drops.load(Ordering::SeqCst), 0);
    assert_eq!(authority_drops.load(Ordering::SeqCst), 0);

    binding.release(allocation).unwrap();
    assert_eq!(allocator.frees.load(Ordering::SeqCst), 1);
    drop(binding);
    assert_eq!(context_drops.load(Ordering::SeqCst), 1);
    assert_eq!(authority_drops.load(Ordering::SeqCst), 1);
}

#[test]
fn registry_owner_keeps_context_after_binding_retires_first() {
    let context_drops = Arc::new(AtomicUsize::new(0));
    let context_owner = Arc::new(DropProbe(Arc::clone(&context_drops)));
    let registry = BindingRegistry::new().unwrap();
    let context = registry
        .register_provider_context(DeviceKey::HOST, resource(&context_owner))
        .unwrap();
    let authority = registry
        .register_authority(DeviceKey::HOST, Arc::new(()) as Arc<dyn BindingResource>)
        .unwrap();
    let mechanism = registry
        .register_allocator(
            context,
            authority,
            Arc::new(TestMechanism::new(DeviceKey::HOST)) as Arc<dyn DeviceAllocator>,
        )
        .unwrap();
    let binding = registry.bind(DeviceKey::HOST).unwrap();
    let allocation = binding.allocate(64, 16).unwrap();
    binding.release(allocation).unwrap();
    drop(binding);
    drop(context_owner);
    assert_eq!(context_drops.load(Ordering::SeqCst), 0);

    registry.retire(mechanism).unwrap();
    registry.remove(mechanism).unwrap();
    assert_eq!(context_drops.load(Ordering::SeqCst), 0);
    registry.remove_provider_context(context).unwrap();
    registry.remove_authority(authority).unwrap();
    assert_eq!(context_drops.load(Ordering::SeqCst), 1);
    drop(registry);
    assert_eq!(context_drops.load(Ordering::SeqCst), 1);
}

#[test]
fn switch_pins_old_release_and_capability_while_new_binding_uses_new_mechanism() {
    let registry = BindingRegistry::new().unwrap();
    let context = registry
        .register_provider_context(DeviceKey::HOST, Arc::new(()) as Arc<dyn BindingResource>)
        .unwrap();
    let authority = registry
        .register_authority(DeviceKey::HOST, Arc::new(()) as Arc<dyn BindingResource>)
        .unwrap();
    let old_allocator = Arc::new(TestMechanism::new(DeviceKey::HOST));
    let new_allocator = Arc::new(TestMechanism::new(DeviceKey::HOST));
    let _old = registry
        .register_allocator(
            context,
            authority,
            old_allocator.clone() as Arc<dyn DeviceAllocator>,
        )
        .unwrap();
    let new = registry
        .register_allocator(
            context,
            authority,
            new_allocator.clone() as Arc<dyn DeviceAllocator>,
        )
        .unwrap();

    let old_binding = registry.bind(DeviceKey::HOST).unwrap();
    let old_capability = old_binding.virtual_backing().unwrap().unwrap();
    let old_allocation = old_binding.allocate(64, 16).unwrap();
    registry.select(new).unwrap();
    let new_binding = registry.bind(DeviceKey::HOST).unwrap();
    let new_allocation = new_binding.allocate(64, 16).unwrap();

    old_capability
        .commit_allocation_range(&old_allocation, 0, 16)
        .unwrap();
    assert_eq!(old_allocator.virtual_operations.load(Ordering::SeqCst), 1);
    assert_eq!(new_allocator.virtual_operations.load(Ordering::SeqCst), 0);
    old_binding.release(old_allocation).unwrap();
    new_binding.release(new_allocation).unwrap();
    assert_eq!(old_allocator.frees.load(Ordering::SeqCst), 1);
    assert_eq!(new_allocator.frees.load(Ordering::SeqCst), 1);
}

#[test]
fn wrong_device_authority_mechanism_and_binding_fail_before_operation() {
    let registry = BindingRegistry::new().unwrap();
    let host_context = registry
        .register_provider_context(DeviceKey::HOST, Arc::new(()) as Arc<dyn BindingResource>)
        .unwrap();
    let host_authority = registry
        .register_authority(DeviceKey::HOST, Arc::new(()) as Arc<dyn BindingResource>)
        .unwrap();
    let wrong_device = registry.register_allocator(
        host_context,
        host_authority,
        Arc::new(TestMechanism::new(DeviceKey::device(1))) as Arc<dyn DeviceAllocator>,
    );
    assert!(matches!(
        wrong_device,
        Err(BindingError::DeviceMismatch {
            subject: "allocator",
            ..
        })
    ));

    let first_allocator = Arc::new(TestMechanism::new(DeviceKey::HOST));
    let first_mechanism = registry
        .register_allocator(
            host_context,
            host_authority,
            first_allocator.clone() as Arc<dyn DeviceAllocator>,
        )
        .unwrap();
    let first = registry.bind_registered(first_mechanism).unwrap();
    let allocation = first.allocate(64, 16).unwrap();
    let view = allocation.view(0, 16).unwrap();

    let other_authority = registry
        .register_authority(DeviceKey::HOST, Arc::new(()) as Arc<dyn BindingResource>)
        .unwrap();
    let second_allocator = Arc::new(TestMechanism::new(DeviceKey::HOST));
    let second_mechanism = registry
        .register_allocator(
            host_context,
            other_authority,
            second_allocator.clone() as Arc<dyn DeviceAllocator>,
        )
        .unwrap();
    let second = registry.bind_registered(second_mechanism).unwrap();
    let second_capability = second.virtual_backing().unwrap().unwrap();
    let callbacks = AtomicUsize::new(0);

    assert!(matches!(
        second_capability.commit_allocation_range(&allocation, 0, 8),
        Err(BindingError::BindingMismatch { .. })
    ));
    assert!(matches!(
        second.with_view(&view, |_| callbacks.fetch_add(1, Ordering::SeqCst)),
        Err(BindingError::BindingMismatch { .. })
    ));
    assert_eq!(callbacks.load(Ordering::SeqCst), 0);
    assert_eq!(
        second_allocator.virtual_operations.load(Ordering::SeqCst),
        0
    );

    let release_error = second.release(allocation).unwrap_err();
    assert!(matches!(
        release_error.error(),
        BindingError::BindingMismatch { .. }
    ));
    let (_, allocation) = release_error.into_parts();
    first.release(allocation).unwrap();
    assert_eq!(first_allocator.frees.load(Ordering::SeqCst), 1);
}

#[derive(Debug)]
struct ReusingAllocator {
    device: DeviceKey,
    storage: Mutex<Box<[u8; 256]>>,
    live: AtomicBool,
    frees: AtomicUsize,
}

impl ReusingAllocator {
    fn new(device: DeviceKey) -> Self {
        Self {
            device,
            storage: Mutex::new(Box::new([0; 256])),
            live: AtomicBool::new(false),
            frees: AtomicUsize::new(0),
        }
    }
}

impl DeviceAllocator for ReusingAllocator {
    fn allocate(&self, bytes: usize, align: usize) -> Result<NonNull<u8>, MemoryError> {
        assert!(bytes <= 256);
        assert!(align <= std::mem::align_of::<usize>());
        assert!(
            !self.live.swap(true, Ordering::SeqCst),
            "test allocator permits only one live allocation"
        );
        let mut storage = self.storage.lock().unwrap();
        NonNull::new(storage.as_mut_ptr()).ok_or_else(|| MemoryError::AllocationFailed {
            tier: self.device.tier.name(),
            requested: bytes as u64,
            reason: "test storage unexpectedly had a null pointer".into(),
        })
    }

    unsafe fn deallocate(&self, _ptr: NonNull<u8>, _bytes: usize, _align: usize) {
        assert!(self.live.swap(false, Ordering::SeqCst));
        self.frees.fetch_add(1, Ordering::SeqCst);
    }

    fn device(&self) -> DeviceKey {
        self.device
    }
}

#[test]
fn reused_virtual_address_gets_new_generation_and_rejects_stale_view() {
    let registry = BindingRegistry::new().unwrap();
    let allocator = Arc::new(ReusingAllocator::new(DeviceKey::HOST));
    let context = registry
        .register_provider_context(DeviceKey::HOST, Arc::new(()) as Arc<dyn BindingResource>)
        .unwrap();
    let authority = registry
        .register_authority(DeviceKey::HOST, Arc::new(()) as Arc<dyn BindingResource>)
        .unwrap();
    registry
        .register_allocator(
            context,
            authority,
            allocator.clone() as Arc<dyn DeviceAllocator>,
        )
        .unwrap();
    let binding = registry.bind(DeviceKey::HOST).unwrap();
    let first = binding.allocate(64, std::mem::align_of::<usize>()).unwrap();
    let stale_view = first.view(0, 8).unwrap();
    let first_ptr = first.as_ptr();
    let first_identity = first.identity();
    binding.release(first).unwrap();

    let second = binding.allocate(64, std::mem::align_of::<usize>()).unwrap();
    assert_eq!(second.as_ptr(), first_ptr);
    assert_ne!(second.identity().generation(), first_identity.generation());
    let called = AtomicBool::new(false);
    assert!(matches!(
        binding.with_view(&stale_view, |_| called.store(true, Ordering::SeqCst)),
        Err(BindingError::StaleAllocation(identity)) if identity == first_identity
    ));
    assert!(!called.load(Ordering::SeqCst));
    binding.release(second).unwrap();
}

#[test]
fn trusted_composite_is_explicit_and_foreign_registration_is_rejected() {
    let registry = BindingRegistry::new().unwrap();
    let context = registry
        .register_provider_context(DeviceKey::HOST, Arc::new(()) as Arc<dyn BindingResource>)
        .unwrap();
    let authority = registry
        .register_authority(DeviceKey::HOST, Arc::new(()) as Arc<dyn BindingResource>)
        .unwrap();
    // SAFETY: TestMechanism implements all three interfaces on one object.
    let composite = unsafe {
        registry.register_trusted_composite(
            context,
            authority,
            Arc::new(TestMechanism::new(DeviceKey::HOST)) as Arc<dyn DeviceAllocator>,
        )
    }
    .unwrap();
    assert_eq!(
        registry.snapshot(composite).unwrap().coherence,
        MechanismCoherence::TrustedComposite
    );
    let binding = registry.bind_registered(composite).unwrap();
    assert!(binding.virtual_backing().unwrap().is_some());
    assert!(binding.shared_mapping().unwrap().is_some());

    let foreign_registry = BindingRegistry::new().unwrap();
    assert!(matches!(
        foreign_registry.select(composite),
        Err(BindingError::ForeignRegistry { kind: "mechanism" })
    ));
    assert!(matches!(
        foreign_registry.bind_registered(composite),
        Err(BindingError::ForeignRegistry { kind: "mechanism" })
    ));

    let foreign_context = foreign_registry
        .register_provider_context(DeviceKey::HOST, Arc::new(()) as Arc<dyn BindingResource>)
        .unwrap();
    assert!(matches!(
        registry.register_allocator(
            foreign_context,
            authority,
            Arc::new(TestMechanism::new(DeviceKey::HOST)) as Arc<dyn DeviceAllocator>,
        ),
        Err(BindingError::ForeignRegistry {
            kind: "provider context"
        })
    ));
}

#[test]
fn shared_prefix_and_allocation_must_have_the_same_binding() {
    let registry = BindingRegistry::new().unwrap();
    let allocator = Arc::new(TestMechanism::new(DeviceKey::HOST));
    registered_test_mechanism(&registry, DeviceKey::HOST, allocator.clone());
    let first = registry.bind(DeviceKey::HOST).unwrap();
    let second = registry.bind(DeviceKey::HOST).unwrap();
    let first_mapping = first.shared_mapping().unwrap().unwrap();
    let second_mapping = second.shared_mapping().unwrap().unwrap();
    let prefix = first_mapping.create_shared_prefix(64).unwrap();
    let allocation = first.allocate(64, 16).unwrap();
    let operations_before = allocator.shared_operations.load(Ordering::SeqCst);

    assert!(matches!(
        second_mapping.incremental_owned_bytes_for_shared_prefix(&prefix),
        Err(BindingError::BindingMismatch { .. })
    ));
    assert!(matches!(
        second_mapping.commit_shared_prefix(&prefix, &allocation, 0),
        Err(BindingError::BindingMismatch { .. })
    ));
    assert_eq!(
        allocator.shared_operations.load(Ordering::SeqCst),
        operations_before
    );
    first.release(allocation).unwrap();
}

#[test]
fn shared_prefix_drops_before_context_pin_after_terminal_registry_teardown() {
    let context_released = Arc::new(AtomicBool::new(false));
    let prefix_dropped = Arc::new(AtomicBool::new(false));
    let registry = BindingRegistry::new().unwrap();
    let context = registry
        .register_provider_context(
            DeviceKey::HOST,
            Arc::new(ContextDropFlag {
                released: Arc::clone(&context_released),
            }) as Arc<dyn BindingResource>,
        )
        .unwrap();
    let authority = registry
        .register_authority(DeviceKey::HOST, Arc::new(()) as Arc<dyn BindingResource>)
        .unwrap();
    let mechanism = registry
        .register_allocator(
            context,
            authority,
            Arc::new(DropOrderMechanism {
                context_released: Arc::clone(&context_released),
                prefix_dropped: Arc::clone(&prefix_dropped),
            }) as Arc<dyn DeviceAllocator>,
        )
        .unwrap();
    let binding = registry.bind(DeviceKey::HOST).unwrap();
    let mapping = binding.shared_mapping().unwrap().unwrap();
    let prefix = mapping.create_shared_prefix(64).unwrap();
    drop(mapping);
    drop(binding);

    registry
        .invalidate_device(DeviceKey::HOST, "drop-order regression")
        .unwrap();
    registry.confirm_context_terminated(context).unwrap();
    registry.remove(mechanism).unwrap();
    registry.remove_provider_context(context).unwrap();
    registry.remove_authority(authority).unwrap();
    drop(registry);
    assert!(!prefix_dropped.load(Ordering::SeqCst));
    assert!(!context_released.load(Ordering::SeqCst));

    drop(prefix);
    assert!(prefix_dropped.load(Ordering::SeqCst));
    assert!(context_released.load(Ordering::SeqCst));
}

#[test]
fn device_loss_invalidates_without_free_or_resource_refund() {
    let context_drops = Arc::new(AtomicUsize::new(0));
    let authority_drops = Arc::new(AtomicUsize::new(0));
    let context_owner = Arc::new(DropProbe(Arc::clone(&context_drops)));
    let authority_owner = Arc::new(DropProbe(Arc::clone(&authority_drops)));
    let registry = BindingRegistry::new().unwrap();
    let context = registry
        .register_provider_context(DeviceKey::HOST, resource(&context_owner))
        .unwrap();
    let authority = registry
        .register_authority(DeviceKey::HOST, resource(&authority_owner))
        .unwrap();
    let allocator = Arc::new(ReusingAllocator::new(DeviceKey::HOST));
    let mechanism = registry
        .register_allocator(
            context,
            authority,
            allocator.clone() as Arc<dyn DeviceAllocator>,
        )
        .unwrap();
    let binding = registry.bind(DeviceKey::HOST).unwrap();
    let allocation = binding.allocate(64, std::mem::align_of::<usize>()).unwrap();
    let view = allocation.view(0, 8).unwrap();

    registry
        .invalidate_device(DeviceKey::HOST, "simulated TDR")
        .unwrap();
    let called = AtomicBool::new(false);
    assert!(matches!(
        binding.with_view(&view, |_| called.store(true, Ordering::SeqCst)),
        Err(BindingError::DeviceLost { .. })
    ));
    assert!(!called.load(Ordering::SeqCst));
    let release_error = binding.release(allocation).unwrap_err();
    assert!(matches!(
        release_error.error(),
        BindingError::DeviceLost { .. }
    ));
    assert_eq!(allocator.frees.load(Ordering::SeqCst), 0);
    assert_eq!(context_drops.load(Ordering::SeqCst), 0);
    assert_eq!(authority_drops.load(Ordering::SeqCst), 0);

    let (_, allocation) = release_error.into_parts();
    registry.confirm_context_terminated(context).unwrap();
    assert_eq!(
        registry.snapshot(mechanism).unwrap().lifecycle,
        MechanismLifecycle::Terminated
    );
    registry.remove(mechanism).unwrap();
    registry.remove_provider_context(context).unwrap();
    registry.remove_authority(authority).unwrap();
    drop(registry);
    drop(context_owner);
    drop(authority_owner);
    assert_eq!(context_drops.load(Ordering::SeqCst), 0);
    assert_eq!(authority_drops.load(Ordering::SeqCst), 0);
    drop(view);
    drop(allocation);
    drop(binding);
    assert_eq!(context_drops.load(Ordering::SeqCst), 1);
    assert_eq!(authority_drops.load(Ordering::SeqCst), 1);
}

#[derive(Debug)]
struct BlockingAllocator {
    entered: Arc<Barrier>,
    resume: Arc<Barrier>,
}

impl DeviceAllocator for BlockingAllocator {
    fn allocate(&self, bytes: usize, align: usize) -> Result<NonNull<u8>, MemoryError> {
        self.entered.wait();
        self.resume.wait();
        HostAllocator.allocate(bytes, align)
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, bytes: usize, align: usize) {
        // SAFETY: forwarded unchanged from this method's contract.
        unsafe { HostAllocator.deallocate(ptr, bytes, align) };
    }

    fn device(&self) -> DeviceKey {
        DeviceKey::HOST
    }
}

#[test]
fn context_termination_requires_in_flight_allocator_callbacks_to_quiesce() {
    let registry = BindingRegistry::new().unwrap();
    let context = registry
        .register_provider_context(DeviceKey::HOST, Arc::new(()) as Arc<dyn BindingResource>)
        .unwrap();
    let authority = registry
        .register_authority(DeviceKey::HOST, Arc::new(()) as Arc<dyn BindingResource>)
        .unwrap();
    let entered = Arc::new(Barrier::new(2));
    let resume = Arc::new(Barrier::new(2));
    let mechanism = registry
        .register_allocator(
            context,
            authority,
            Arc::new(BlockingAllocator {
                entered: Arc::clone(&entered),
                resume: Arc::clone(&resume),
            }) as Arc<dyn DeviceAllocator>,
        )
        .unwrap();
    let binding = registry.bind(DeviceKey::HOST).unwrap();
    let worker_binding = binding.clone();
    let worker = thread::spawn(move || worker_binding.allocate(64, 16).unwrap());

    entered.wait();
    registry
        .invalidate_device(DeviceKey::HOST, "loss during allocation")
        .unwrap();
    assert!(matches!(
        registry.confirm_context_terminated(context),
        Err(BindingError::ContextNotQuiescent {
            active_operations: 1,
            ..
        })
    ));
    resume.wait();
    let allocation = worker.join().unwrap();
    // Confirmed termination discharges retained ownership without calling the
    // allocator, because on a real device the state it referred to is already
    // gone. On the host heap nothing else reclaims these bytes, so the test
    // gives back what it told the runtime to abandon rather than being exempted
    // from Miri's leak check.
    let abandoned = allocation.as_ptr();
    registry.confirm_context_terminated(context).unwrap();
    registry.remove(mechanism).unwrap();
    registry.remove_provider_context(context).unwrap();
    registry.remove_authority(authority).unwrap();
    drop(allocation);
    drop(binding);
    // SAFETY: the exact address, size and alignment `HostAllocator` handed out
    // above; the runtime has discharged its ownership and no handle survives.
    unsafe { HostAllocator.deallocate(abandoned, 64, 16) };
}

#[derive(Debug)]
struct ReentrantMechanism {
    hook: Mutex<Option<(BindingRegistry, RegisteredMechanism)>>,
    virtual_calls: AtomicUsize,
}

impl ReentrantMechanism {
    fn call_registry(&self) {
        let hook = self.hook.lock().unwrap().clone();
        if let Some((registry, mechanism)) = hook {
            registry.snapshot(mechanism).unwrap();
        }
    }
}

impl DeviceAllocator for ReentrantMechanism {
    fn allocate(&self, bytes: usize, align: usize) -> Result<NonNull<u8>, MemoryError> {
        self.call_registry();
        HostAllocator.allocate(bytes, align)
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, bytes: usize, align: usize) {
        self.call_registry();
        // SAFETY: forwarded unchanged from this method's contract.
        unsafe { HostAllocator.deallocate(ptr, bytes, align) };
    }

    fn device(&self) -> DeviceKey {
        DeviceKey::HOST
    }

    fn as_virtual_backing(&self) -> Option<&dyn VirtualBacking> {
        Some(self)
    }
}

impl VirtualBacking for ReentrantMechanism {
    fn allocate_committed(
        &self,
        bytes: usize,
        align: usize,
        _committed_ranges: &[std::ops::Range<usize>],
    ) -> Result<NonNull<u8>, MemoryError> {
        self.call_registry();
        HostAllocator.allocate(bytes, align)
    }

    fn commit_allocation_range(
        &self,
        _ptr: NonNull<u8>,
        _allocation_bytes: usize,
        _align: usize,
        _offset: usize,
        _bytes: usize,
    ) -> Result<(), MemoryError> {
        self.call_registry();
        self.virtual_calls.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }

    fn mapped_bytes_for_allocation_ranges(
        &self,
        ranges: &[AllocationCommitRange],
    ) -> Result<u64, MemoryError> {
        self.call_registry();
        Ok(ranges.iter().map(|range| range.bytes as u64).sum())
    }

    fn mapped_bytes_for_allocation(&self, bytes: usize, _align: usize) -> Result<u64, MemoryError> {
        self.call_registry();
        Ok(bytes as u64)
    }

    fn decommit_allocation_range(
        &self,
        _ptr: NonNull<u8>,
        _allocation_bytes: usize,
        _align: usize,
        _offset: usize,
        bytes: usize,
    ) -> Result<u64, MemoryError> {
        self.call_registry();
        Ok(bytes as u64)
    }

    fn allocation_committed_bytes(
        &self,
        _ptr: NonNull<u8>,
        allocation_bytes: usize,
        _align: usize,
    ) -> usize {
        self.call_registry();
        allocation_bytes
    }
}

#[test]
fn no_registry_or_mechanism_lock_is_held_across_allocator_or_capability_callbacks() {
    let registry = BindingRegistry::new().unwrap();
    let context = registry
        .register_provider_context(DeviceKey::HOST, Arc::new(()) as Arc<dyn BindingResource>)
        .unwrap();
    let authority = registry
        .register_authority(DeviceKey::HOST, Arc::new(()) as Arc<dyn BindingResource>)
        .unwrap();
    let allocator = Arc::new(ReentrantMechanism {
        hook: Mutex::new(None),
        virtual_calls: AtomicUsize::new(0),
    });
    let mechanism = registry
        .register_allocator(
            context,
            authority,
            allocator.clone() as Arc<dyn DeviceAllocator>,
        )
        .unwrap();
    assert!(matches!(
        registry.remove(mechanism),
        Err(BindingError::InactiveMechanism {
            lifecycle: MechanismLifecycle::Active,
            ..
        })
    ));
    *allocator.hook.lock().unwrap() = Some((registry.clone(), mechanism));

    let binding = registry.bind(DeviceKey::HOST).unwrap();
    let allocation = binding.allocate(64, 16).unwrap();
    binding
        .virtual_backing()
        .unwrap()
        .unwrap()
        .commit_allocation_range(&allocation, 0, 16)
        .unwrap();
    binding.release(allocation).unwrap();
    assert_eq!(allocator.virtual_calls.load(Ordering::SeqCst), 1);
    // The hook is what makes this allocator re-entrant, and it is also an
    // `Arc` cycle: the registry owns the mechanism, the mechanism owns this
    // allocator, and the hook owns the registry back. Leaving it set leaks the
    // whole graph, which Miri reports and which is a real trap for any provider
    // that keeps a registry handle inside its allocator for re-entrancy. Break
    // it once the callbacks under test have been observed.
    *allocator.hook.lock().unwrap() = None;
}

#[test]
fn concurrent_register_lookup_switch_invalidate_and_teardown_is_consistent() {
    const THREADS: usize = 6;
    const ITERATIONS: usize = 40;

    let registry = Arc::new(BindingRegistry::new().unwrap());
    let context = registry
        .register_provider_context(DeviceKey::HOST, Arc::new(()) as Arc<dyn BindingResource>)
        .unwrap();
    let authority = registry
        .register_authority(DeviceKey::HOST, Arc::new(()) as Arc<dyn BindingResource>)
        .unwrap();
    let barrier = Arc::new(Barrier::new(THREADS));
    let mut workers = Vec::new();

    for _worker in 0..(THREADS - 1) {
        let registry = Arc::clone(&registry);
        let barrier = Arc::clone(&barrier);
        workers.push(thread::spawn(move || {
            barrier.wait();
            for _ in 0..ITERATIONS {
                let allocator = Arc::new(TestMechanism::new(DeviceKey::HOST));
                let mechanism = registry
                    .register_allocator(context, authority, allocator as Arc<dyn DeviceAllocator>)
                    .unwrap();
                registry.select(mechanism).unwrap();
                let binding = registry.bind_registered(mechanism).unwrap();
                let allocation = binding.allocate(32, 8).unwrap();
                registry.snapshot(mechanism).unwrap();
                registry.retire(mechanism).unwrap();
                binding.release(allocation).unwrap();
                registry.remove(mechanism).unwrap();
            }
        }));
    }

    let registry_for_loss = Arc::clone(&registry);
    let barrier_for_loss = Arc::clone(&barrier);
    workers.push(thread::spawn(move || {
        barrier_for_loss.wait();
        for iteration in 0..ITERATIONS {
            let device = DeviceKey::device(10_000 + iteration as u32);
            let context = registry_for_loss
                .register_provider_context(device, Arc::new(()) as Arc<dyn BindingResource>)
                .unwrap();
            let authority = registry_for_loss
                .register_authority(device, Arc::new(()) as Arc<dyn BindingResource>)
                .unwrap();
            let mechanism = registry_for_loss
                .register_allocator(
                    context,
                    authority,
                    Arc::new(ReusingAllocator::new(device)) as Arc<dyn DeviceAllocator>,
                )
                .unwrap();
            let binding = registry_for_loss.bind(device).unwrap();
            let allocation = binding.allocate(16, 8).unwrap();
            registry_for_loss
                .invalidate_device(device, "stress loss")
                .unwrap();
            registry_for_loss
                .confirm_context_terminated(context)
                .unwrap();
            registry_for_loss.remove(mechanism).unwrap();
            drop(allocation);
            drop(binding);
        }
    }));

    for worker in workers {
        worker.join().unwrap();
    }
}

#[test]
fn erased_third_party_allocator_remains_supported() {
    let registry = BindingRegistry::new().unwrap();
    let context = registry
        .register_provider_context(DeviceKey::HOST, Arc::new(()) as Arc<dyn BindingResource>)
        .unwrap();
    let authority = registry
        .register_authority(DeviceKey::HOST, Arc::new(()) as Arc<dyn BindingResource>)
        .unwrap();
    let concrete = Arc::new(TestMechanism::new(DeviceKey::HOST));
    let erased: Arc<dyn DeviceAllocator> = concrete.clone();
    registry
        .register_allocator(context, authority, erased)
        .unwrap();
    let binding = registry.bind(DeviceKey::HOST).unwrap();
    let allocation = binding.allocate(64, 16).unwrap();
    assert!(binding.virtual_backing().unwrap().is_some());
    assert!(binding.shared_mapping().unwrap().is_some());
    binding.release(allocation).unwrap();
    assert_eq!(concrete.allocations.load(Ordering::SeqCst), 1);
    assert_eq!(concrete.frees.load(Ordering::SeqCst), 1);
}

#[test]
fn allocator_drop_runs_while_context_and_authority_pins_are_alive() {
    let log = Arc::new(Mutex::new(Vec::new()));
    let context_resource = Arc::new(PinnedResource {
        label: "context",
        log: Arc::clone(&log),
    });
    let authority_resource = Arc::new(PinnedResource {
        label: "authority",
        log: Arc::clone(&log),
    });
    let registry = BindingRegistry::new().unwrap();
    let context = registry
        .register_provider_context(
            DeviceKey::HOST,
            Arc::clone(&context_resource) as Arc<dyn BindingResource>,
        )
        .unwrap();
    let authority = registry
        .register_authority(
            DeviceKey::HOST,
            Arc::clone(&authority_resource) as Arc<dyn BindingResource>,
        )
        .unwrap();
    let mechanism = registry
        .register_allocator(
            context,
            authority,
            Arc::new(ContextDependentAllocator {
                log: Arc::clone(&log),
                context: Arc::downgrade(&context_resource),
                authority: Arc::downgrade(&authority_resource),
            }) as Arc<dyn DeviceAllocator>,
        )
        .unwrap();
    let binding = registry.bind(DeviceKey::HOST).unwrap();

    // Retire every registry-side and test-side owner, so the outstanding
    // binding becomes the only remaining pin on the allocator and its
    // resources and their drop order is decided entirely by the registry.
    registry.retire(mechanism).unwrap();
    registry.remove(mechanism).unwrap();
    registry.remove_provider_context(context).unwrap();
    registry.remove_authority(authority).unwrap();
    drop(registry);
    drop(context_resource);
    drop(authority_resource);
    assert!(
        log.lock().unwrap().is_empty(),
        "an outstanding binding must keep the allocator and both pins alive"
    );

    drop(binding);
    let events = log.lock().unwrap().clone();
    assert_eq!(
        events,
        vec![
            "allocator released (context alive: true, authority alive: true)".to_string(),
            "authority released".to_string(),
            "context released".to_string(),
        ],
        "the allocator destructor must run before its provider-context and authority pins"
    );
}
