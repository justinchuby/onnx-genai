//! Portable Phase-4 owning/deferred-release tests.
//!
//! Everything here runs on the host allocator, so the contract the CUDA EP will
//! consume is exercised without a device.

use std::collections::HashSet;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Barrier, Mutex, mpsc};
use std::thread;
use std::time::Duration;

use onnx_runtime_memory_api::{
    AllocationReleaseOutcome, AllocationReleaseState, BindingError, BindingRegistry,
    BindingResource, BoundAllocation, DeferredEnqueueError, DeferredEnqueueRejection,
    DeferredReleaseDisposition, DeferredReleaseQueue, DeviceAllocator, DeviceKey, HostAllocator,
    MechanismLifecycle, MemoryBinding, MemoryError, PreparedAllocationRelease, QuarantineReason,
    RegisteredMechanism, RegisteredProviderContext, ReleaseAccounting, ResidualOwnership,
};

/// How the recording allocator answers its next release.
#[derive(Clone, Copy, Debug)]
enum ReleaseMode {
    /// Free for real and report a complete outcome with no mapped refund.
    Complete,
    /// Free for real and report a complete outcome with a mapped refund.
    CompleteWithUnmapped(u64),
    /// Mutate part of the allocation and stop, retaining residual ownership.
    PartialQuarantine { unmapped: u64, retained: u64 },
    /// Refuse without mutating anything.
    FailUnchanged,
}

/// A host allocator that records every release and refuses to double free.
#[derive(Debug)]
struct RecordingAllocator {
    /// Addresses handed out and not yet released. Release asserts membership,
    /// so a double free is caught while legitimate address reuse after a real
    /// free is not mistaken for one.
    live: Mutex<HashSet<usize>>,
    release_calls: AtomicUsize,
    mode: Mutex<ReleaseMode>,
    /// Set after registration so a release callback can prove that neither the
    /// registry lock nor the mechanism lock is held while it runs.
    probe: Mutex<Option<(BindingRegistry, RegisteredMechanism)>>,
}

impl Default for RecordingAllocator {
    fn default() -> Self {
        Self {
            live: Mutex::new(HashSet::new()),
            release_calls: AtomicUsize::new(0),
            mode: Mutex::new(ReleaseMode::Complete),
            probe: Mutex::new(None),
        }
    }
}

impl RecordingAllocator {
    fn set_mode(&self, mode: ReleaseMode) {
        *self.mode.lock().expect("mode lock") = mode;
    }

    fn arm_probe(&self, registry: BindingRegistry, mechanism: RegisteredMechanism) {
        *self.probe.lock().expect("probe lock") = Some((registry, mechanism));
    }

    /// Break the `Arc` cycle the armed probe creates.
    ///
    /// The registry owns the mechanism, the mechanism owns this object, and the
    /// probe owns the registry back, so leaving it armed leaks the whole graph.
    /// That is a real trap for any provider holding a registry handle inside a
    /// callback, not just a test artifact, so it is broken explicitly rather
    /// than exempted from Miri's leak check.
    fn disarm_probe(&self) {
        *self.probe.lock().expect("probe lock") = None;
    }

    fn release_calls(&self) -> usize {
        self.release_calls.load(Ordering::SeqCst)
    }

    /// Addresses the allocator still owns, i.e. everything allocated and not
    /// released. Quarantined ownership deliberately stays here.
    fn live_addresses(&self) -> Vec<usize> {
        let mut live = self
            .live
            .lock()
            .expect("live set")
            .iter()
            .copied()
            .collect::<Vec<_>>();
        live.sort_unstable();
        live
    }

    fn record(&self, address: usize) {
        let mut live = self.live.lock().expect("live set");
        assert!(
            live.remove(&address),
            "address {address:#x} was released twice or was never allocated here"
        );
    }

    fn run_probe(&self) {
        let probe = self.probe.lock().expect("probe lock").clone();
        if let Some((registry, mechanism)) = probe {
            assert_locks_are_free(&registry, mechanism, "allocator release callback");
        }
    }
}

impl DeviceAllocator for RecordingAllocator {
    fn allocate(&self, bytes: usize, align: usize) -> Result<NonNull<u8>, MemoryError> {
        let ptr = HostAllocator.allocate(bytes, align)?;
        assert!(
            self.live
                .lock()
                .expect("live set")
                .insert(ptr.as_ptr() as usize),
            "the host allocator returned an address it had already handed out"
        );
        Ok(ptr)
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, bytes: usize, align: usize) {
        self.record(ptr.as_ptr() as usize);
        // SAFETY: forwarded unchanged from this method's contract.
        unsafe { HostAllocator.deallocate(ptr, bytes, align) };
    }

    unsafe fn release(
        &self,
        ptr: NonNull<u8>,
        bytes: usize,
        align: usize,
    ) -> AllocationReleaseOutcome {
        self.release_calls.fetch_add(1, Ordering::SeqCst);
        self.run_probe();
        let mode = *self.mode.lock().expect("mode lock");
        let address = ptr.as_ptr() as usize;
        match mode {
            ReleaseMode::Complete => {
                self.record(address);
                // SAFETY: the exact live allocation prepared for release.
                unsafe { HostAllocator.deallocate(ptr, bytes, align) };
                AllocationReleaseOutcome::complete(ReleaseAccounting::eager(bytes as u64))
            }
            ReleaseMode::CompleteWithUnmapped(unmapped) => {
                self.record(address);
                // SAFETY: the exact live allocation prepared for release.
                unsafe { HostAllocator.deallocate(ptr, bytes, align) };
                AllocationReleaseOutcome::complete(ReleaseAccounting::new(bytes as u64, unmapped))
            }
            ReleaseMode::PartialQuarantine { unmapped, retained } => {
                AllocationReleaseOutcome::quarantined(
                    ReleaseAccounting::new(bytes as u64, unmapped),
                    ResidualOwnership {
                        state: AllocationReleaseState::PartiallyUnmapped,
                        reason: QuarantineReason::PartialRelease,
                        retained_bytes: retained,
                        address,
                        align,
                    },
                )
            }
            ReleaseMode::FailUnchanged => {
                AllocationReleaseOutcome::failed("the driver refused; nothing was mutated")
            }
        }
    }

    fn device(&self) -> DeviceKey {
        DeviceKey::HOST
    }
}

/// Take both lock classes from another thread with a deadline.
///
/// If the caller were holding the registry lock or the mechanism lock while
/// invoking a callback, this would time out instead of hanging the suite.
fn assert_locks_are_free(registry: &BindingRegistry, mechanism: RegisteredMechanism, site: &str) {
    let registry = registry.clone();
    let (sender, receiver) = mpsc::channel();
    let worker = thread::spawn(move || {
        let observed = registry.snapshot(mechanism).is_ok();
        let _ = sender.send(observed);
    });
    let observed = receiver.recv_timeout(Duration::from_secs(10));
    assert_eq!(
        observed,
        Ok(true),
        "a registry or mechanism lock was still held across the {site}"
    );
    worker.join().expect("probe thread");
}

/// A bounded queue that keeps prepared requests until they are drained.
#[derive(Debug)]
struct BoundedQueue {
    capacity: usize,
    pending: Mutex<Vec<PreparedAllocationRelease>>,
    probe: Mutex<Option<(BindingRegistry, RegisteredMechanism)>>,
}

impl BoundedQueue {
    fn new(capacity: usize) -> Self {
        Self {
            capacity,
            pending: Mutex::new(Vec::new()),
            probe: Mutex::new(None),
        }
    }

    fn arm_probe(&self, registry: BindingRegistry, mechanism: RegisteredMechanism) {
        *self.probe.lock().expect("probe lock") = Some((registry, mechanism));
    }

    /// Break the `Arc` cycle the armed probe creates.
    ///
    /// The registry owns the mechanism, the mechanism owns this object, and the
    /// probe owns the registry back, so leaving it armed leaks the whole graph.
    /// That is a real trap for any provider holding a registry handle inside a
    /// callback, not just a test artifact, so it is broken explicitly rather
    /// than exempted from Miri's leak check.
    fn disarm_probe(&self) {
        *self.probe.lock().expect("probe lock") = None;
    }

    /// Execute every queued request with the queue lock released first.
    fn drain(&self) -> Vec<AllocationReleaseOutcome> {
        let taken = {
            let mut pending = self.pending.lock().expect("queue lock");
            std::mem::take(&mut *pending)
        };
        taken
            .into_iter()
            .map(PreparedAllocationRelease::execute)
            .collect()
    }
}

impl DeferredReleaseQueue for BoundedQueue {
    fn enqueue(&self, request: PreparedAllocationRelease) -> Result<(), DeferredEnqueueError> {
        let probe = self.probe.lock().expect("probe lock").clone();
        if let Some((registry, mechanism)) = probe {
            assert_locks_are_free(&registry, mechanism, "deferred queue enqueue callback");
        }
        let mut pending = self.pending.lock().expect("queue lock");
        if pending.len() >= self.capacity {
            return Err(DeferredEnqueueError::new(
                DeferredEnqueueRejection::Full,
                request,
            ));
        }
        pending.push(request);
        Ok(())
    }

    fn pending(&self) -> usize {
        self.pending.lock().expect("queue lock").len()
    }
}

/// A queue that always refuses and hands the exact request back.
#[derive(Debug, Default)]
struct RefusingQueue {
    rejection_count: AtomicUsize,
}

impl DeferredReleaseQueue for RefusingQueue {
    fn enqueue(&self, request: PreparedAllocationRelease) -> Result<(), DeferredEnqueueError> {
        self.rejection_count.fetch_add(1, Ordering::SeqCst);
        Err(DeferredEnqueueError::new(
            DeferredEnqueueRejection::Closed,
            request,
        ))
    }
}

#[derive(Debug)]
struct DropProbe(Arc<AtomicUsize>);

impl Drop for DropProbe {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

struct Fixture {
    registry: BindingRegistry,
    context: RegisteredProviderContext,
    mechanism: RegisteredMechanism,
    binding: MemoryBinding,
    allocator: Arc<RecordingAllocator>,
    context_drops: Arc<AtomicUsize>,
}

impl Fixture {
    fn new() -> Self {
        let registry = BindingRegistry::new().expect("registry");
        let context_drops = Arc::new(AtomicUsize::new(0));
        let context = registry
            .register_provider_context(
                DeviceKey::HOST,
                Arc::new(DropProbe(Arc::clone(&context_drops))) as Arc<dyn BindingResource>,
            )
            .expect("context");
        let authority = registry
            .register_authority(DeviceKey::HOST, Arc::new(()) as Arc<dyn BindingResource>)
            .expect("authority");
        let allocator = Arc::new(RecordingAllocator::default());
        let mechanism = registry
            .register_allocator(
                context,
                authority,
                Arc::clone(&allocator) as Arc<dyn DeviceAllocator>,
            )
            .expect("mechanism");
        let binding = registry.bind(DeviceKey::HOST).expect("binding");
        Self {
            registry,
            context,
            mechanism,
            binding,
            allocator,
            context_drops,
        }
    }

    fn snapshot(&self) -> onnx_runtime_memory_api::MechanismSnapshot {
        self.registry.snapshot(self.mechanism).expect("snapshot")
    }

    /// Give back the host bytes this fixture's runtime deliberately retained.
    ///
    /// Quarantine is retention, not release: the address stays owned and is
    /// discharged only at confirmed context termination, which makes no
    /// allocator call because on a real device that state is already gone. On
    /// the host heap nothing else reclaims those bytes, so a test that asserts
    /// quarantine happened is, under Miri's leak check, a test that leaks.
    /// Reclaiming here keeps the check on for everything else instead of
    /// exempting these tests from it.
    fn reclaim_retained(&self) -> usize {
        // A test that has already torn the mechanism down has no quarantine to
        // read; that is not a reclaim failure, so it reports zero rather than
        // panicking. Such a test must reclaim before teardown instead.
        let Ok(retained) = self.registry.quarantined(self.mechanism) else {
            return 0;
        };
        for record in &retained {
            let Some(ptr) = NonNull::new(record.address as *mut u8) else {
                continue;
            };
            // SAFETY: the exact address, size and alignment the recording
            // allocator handed out; the runtime has stopped tracking it as live
            // and quarantined ownership is by construction unaliased.
            unsafe { self.allocator.deallocate(ptr, record.bytes, record.align) };
        }
        retained.len()
    }

    fn assert_quiescent(&self) {
        let snapshot = self.snapshot();
        assert_eq!(snapshot.live_allocations, 0, "{snapshot:?}");
        assert_eq!(snapshot.queued_releases, 0, "{snapshot:?}");
        assert_eq!(snapshot.quarantined_allocations, 0, "{snapshot:?}");
        assert_eq!(snapshot.quarantined_bytes, 0, "{snapshot:?}");
        assert!(!snapshot.retains_ownership(), "{snapshot:?}");
    }
}

/// A provider whose specialized allocation call this crate cannot express still
/// gets a generation before the address escapes, and its release is validated
/// exactly like a binding-issued one.
#[test]
fn an_adopted_allocation_is_generation_checked_like_an_issued_one() {
    let fixture = Fixture::new();
    // Stands in for the CUDA VMM arena's mapped-capacity allocation: made
    // through the *same* mechanism, but by a call this crate has no vocabulary
    // for.
    let ptr = fixture
        .allocator
        .allocate(512, 64)
        .expect("specialized path");
    // SAFETY: one live 512-byte allocation at alignment 64 from this binding's
    // own selected mechanism, not recorded by any binding and owned by nobody
    // else.
    let owner = unsafe { fixture.binding.adopt_allocation(ptr, 512, 64) }.expect("adoption");
    assert_eq!(owner.as_ptr(), ptr);
    assert_eq!(owner.len(), 512);
    assert_eq!(
        owner.identity().binding(),
        fixture.binding.identity(),
        "an adopted allocation belongs to the binding that adopted it"
    );
    assert_eq!(fixture.snapshot().live_allocations, 1);

    // A view keeps it from being released, exactly like an issued allocation.
    let view = owner.view(0, 64).expect("view");
    let error = owner.prepare_release().expect_err("outstanding view");
    let (_, owner) = error.into_parts();
    drop(view);

    let outcome = owner.release_now().expect("release");
    assert!(outcome.is_complete(), "{outcome:?}");
    assert_eq!(
        outcome.accounting().expect("accounting").allocation_bytes,
        512
    );
    fixture.assert_quiescent();
}

/// Adoption is the only step that changes: a dropped adopted owner quarantines
/// rather than freeing, so a forgotten specialized allocation stays accounted.
#[test]
fn a_dropped_adopted_allocation_is_quarantined_not_freed() {
    let fixture = Fixture::new();
    let ptr = fixture
        .allocator
        .allocate(256, 32)
        .expect("specialized path");
    // SAFETY: as above — one live allocation from this binding's mechanism.
    let owner = unsafe { fixture.binding.adopt_allocation(ptr, 256, 32) }.expect("adoption");
    drop(owner);
    let quarantined = fixture.binding.quarantined().expect("quarantine");
    assert_eq!(quarantined.len(), 1);
    assert_eq!(quarantined[0].retained_bytes, 256);
    assert_eq!(quarantined[0].address, ptr.as_ptr() as usize);
    assert_eq!(
        fixture.allocator.release_calls(),
        0,
        "quarantine never calls the allocator"
    );
    assert_eq!(
        fixture.allocator.live_addresses(),
        vec![ptr.as_ptr() as usize],
        "the allocator still owns the retained bytes"
    );
    // Give the host allocation back so the test leaks nothing.
    // SAFETY: the quarantined allocation was never released, and nothing else
    // can reach it now.
    unsafe { HostAllocator.deallocate(ptr, 256, 32) };
}

#[test]
fn cpu_immediate_owning_release_completes_synchronously_without_a_queue() {
    let fixture = Fixture::new();
    let owning = fixture
        .binding
        .allocate_owning(4096, 64)
        .expect("owning allocation");
    assert_eq!(owning.state(), AllocationReleaseState::Live);
    assert_eq!(owning.len(), 4096);
    assert_eq!(fixture.snapshot().live_allocations, 1);

    let outcome = owning.release_now().expect("immediate release");

    assert_eq!(
        outcome,
        AllocationReleaseOutcome::complete(ReleaseAccounting::new(4096, 0)),
        "an eager mechanism reports zero unmapped bytes as a complete release"
    );
    assert_eq!(outcome.state(), AllocationReleaseState::Released);
    assert_eq!(fixture.allocator.release_calls(), 1);
    fixture.assert_quiescent();
}

#[test]
fn zero_byte_allocation_releases_completely_with_zero_bytes() {
    let fixture = Fixture::new();
    let owning = fixture.binding.allocate_owning(0, 64).expect("zero bytes");
    assert!(owning.is_empty());

    let outcome = owning.release_now().expect("zero-byte release");

    assert_eq!(
        outcome,
        AllocationReleaseOutcome::complete(ReleaseAccounting::new(0, 0)),
        "zero bytes is a valid complete result, not an error proxy"
    );
    assert!(outcome.is_complete());
    fixture.assert_quiescent();
}

#[test]
fn mapped_refunds_are_reported_and_zero_is_never_a_failure_proxy() {
    let fixture = Fixture::new();
    fixture
        .allocator
        .set_mode(ReleaseMode::CompleteWithUnmapped(2048));
    let owning = fixture.binding.allocate_owning(4096, 64).expect("owning");

    let outcome = owning.release_now().expect("release");

    assert_eq!(outcome.unmapped_bytes(), 2048);
    assert_eq!(
        outcome.accounting(),
        Some(ReleaseAccounting::new(4096, 2048))
    );
    assert!(outcome.is_complete());
    fixture.assert_quiescent();
}

#[test]
fn queued_release_defers_the_allocator_until_the_queue_executes_it() {
    let fixture = Fixture::new();
    let queue = BoundedQueue::new(4);
    let owning = fixture.binding.allocate_owning(8192, 256).expect("owning");
    let identity = owning.identity();

    let disposition = owning.release_deferred(&queue).expect("enqueue");

    assert_eq!(
        disposition,
        DeferredReleaseDisposition::Queued { identity },
        "the queue took final ownership"
    );
    assert_eq!(disposition.state(), AllocationReleaseState::Queued);
    assert_eq!(queue.pending(), 1);
    assert_eq!(
        fixture.allocator.release_calls(),
        0,
        "a queued release must not touch the allocator before it is executed"
    );

    let queued = fixture.snapshot();
    assert_eq!(
        queued.live_allocations, 0,
        "the live record is already gone"
    );
    assert_eq!(queued.queued_releases, 1);
    assert_eq!(
        queued.active_operations, 1,
        "a queued request pins its mechanism"
    );
    assert!(queued.retains_ownership());

    let outcomes = queue.drain();

    assert_eq!(
        outcomes,
        vec![AllocationReleaseOutcome::complete(ReleaseAccounting::new(
            8192, 0
        ))]
    );
    assert_eq!(fixture.allocator.release_calls(), 1);
    assert_eq!(queue.pending(), 0);
    fixture.assert_quiescent();
}

#[test]
fn enqueue_failure_quarantines_the_exact_request() {
    let fixture = Fixture::new();
    let queue = RefusingQueue::default();
    let owning = fixture.binding.allocate_owning(1024, 64).expect("owning");
    let identity = owning.identity();

    let disposition = owning.release_deferred(&queue).expect("prepared");

    let DeferredReleaseDisposition::Quarantined {
        identity: quarantined_identity,
        rejection,
        outcome,
    } = disposition
    else {
        panic!("a refusing queue must not report a queued disposition: {disposition:?}");
    };
    assert_eq!(quarantined_identity, identity);
    assert_eq!(rejection, DeferredEnqueueRejection::Closed);
    assert_eq!(
        outcome,
        AllocationReleaseOutcome::quarantined(
            ReleaseAccounting::new(1024, 0),
            ResidualOwnership {
                state: AllocationReleaseState::Quarantined,
                reason: QuarantineReason::EnqueueRejected(DeferredEnqueueRejection::Closed),
                retained_bytes: 1024,
                address: outcome.residual().expect("residual").address,
                align: 64,
            }
        )
    );
    assert_eq!(
        fixture.allocator.release_calls(),
        0,
        "a refused request must never be freed"
    );

    let snapshot = fixture.snapshot();
    assert_eq!(snapshot.quarantined_allocations, 1);
    assert_eq!(snapshot.quarantined_bytes, 1024);
    assert_eq!(snapshot.queued_releases, 0);

    let quarantined = fixture.binding.quarantined().expect("quarantine list");
    assert_eq!(quarantined.len(), 1);
    assert_eq!(quarantined[0].identity, identity);
    assert_eq!(quarantined[0].bytes, 1024);
    assert_eq!(
        quarantined[0].reason,
        QuarantineReason::EnqueueRejected(DeferredEnqueueRejection::Closed)
    );
    fixture.reclaim_retained();
}

#[test]
fn a_refused_request_can_be_recovered_and_executed_by_the_caller() {
    let fixture = Fixture::new();
    let queue = RefusingQueue::default();
    let owning = fixture.binding.allocate_owning(2048, 64).expect("owning");
    let identity = owning.identity();
    let prepared = owning.prepare_release().expect("prepared");
    assert_eq!(prepared.state(), AllocationReleaseState::Queued);
    assert_eq!(prepared.authority(), identity.binding().authority());
    assert_eq!(
        prepared.provider_context(),
        identity.binding().provider_context()
    );

    let error = queue.enqueue(prepared).expect_err("the queue refuses");
    assert_eq!(error.rejection(), DeferredEnqueueRejection::Closed);
    assert_eq!(error.request().identity(), identity);

    let recovered = error.into_request();
    assert_eq!(
        recovered.identity(),
        identity,
        "the exact request came back"
    );

    let outcome = recovered.execute();

    assert!(outcome.is_complete(), "{outcome:?}");
    assert_eq!(fixture.allocator.release_calls(), 1);
    fixture.assert_quiescent();
}

#[test]
fn a_dropped_enqueue_error_quarantines_rather_than_losing_the_request() {
    let fixture = Fixture::new();
    let queue = RefusingQueue::default();
    let owning = fixture.binding.allocate_owning(512, 64).expect("owning");
    let prepared = owning.prepare_release().expect("prepared");

    drop(queue.enqueue(prepared).expect_err("the queue refuses"));

    assert_eq!(fixture.allocator.release_calls(), 0);
    let snapshot = fixture.snapshot();
    assert_eq!(snapshot.quarantined_allocations, 1);
    assert_eq!(snapshot.queued_releases, 0);
    assert_eq!(
        fixture.binding.quarantined().expect("quarantine list")[0].reason,
        QuarantineReason::AbandonedRequest
    );
    fixture.reclaim_retained();
}

#[test]
fn an_abandoned_prepared_request_is_quarantined_not_freed() {
    let fixture = Fixture::new();
    let owning = fixture.binding.allocate_owning(4096, 64).expect("owning");
    let identity = owning.identity();

    drop(owning.prepare_release().expect("prepared"));

    assert_eq!(
        fixture.allocator.release_calls(),
        0,
        "Drop must never free from a prepared request"
    );
    let quarantined = fixture.binding.quarantined().expect("quarantine list");
    assert_eq!(quarantined.len(), 1);
    assert_eq!(quarantined[0].identity, identity);
    assert_eq!(quarantined[0].reason, QuarantineReason::AbandonedRequest);
    assert_eq!(quarantined[0].state, AllocationReleaseState::Quarantined);
    assert_eq!(quarantined[0].retained_bytes, 4096);
    assert_eq!(fixture.snapshot().queued_releases, 0);
    fixture.reclaim_retained();
}

#[test]
fn a_dropped_owning_allocation_is_quarantined_not_freed() {
    let fixture = Fixture::new();
    let identity = {
        let owning = fixture.binding.allocate_owning(256, 64).expect("owning");
        owning.identity()
    };

    assert_eq!(fixture.allocator.release_calls(), 0);
    let quarantined = fixture.binding.quarantined().expect("quarantine list");
    assert_eq!(quarantined.len(), 1);
    assert_eq!(quarantined[0].identity, identity);
    assert_eq!(quarantined[0].reason, QuarantineReason::OwnerDropped);
    assert_eq!(fixture.snapshot().live_allocations, 0);
    fixture.reclaim_retained();
}

#[test]
fn outstanding_and_cloned_views_block_physical_release() {
    let fixture = Fixture::new();
    let owning = fixture.binding.allocate_owning(4096, 64).expect("owning");
    let view = owning.view(0, 1024).expect("view");
    let alias = view.clone();
    assert_eq!(owning.outstanding_views(), 2);
    assert_eq!(alias.len(), 1024);
    assert_eq!(alias.allocation_identity(), owning.identity());

    let error = owning
        .release_now()
        .expect_err("outstanding views must block release");
    assert!(
        matches!(
            error.error(),
            BindingError::OutstandingViews { views: 2, .. }
        ),
        "{error:?}"
    );
    assert_eq!(error.state(), AllocationReleaseState::Live);
    let (_, owning) = error.into_parts();
    assert_eq!(
        fixture.allocator.release_calls(),
        0,
        "a blocked release must not reach the allocator"
    );
    assert_eq!(fixture.snapshot().live_allocations, 1);

    drop(alias);
    assert_eq!(owning.outstanding_views(), 1);
    let error = owning
        .release_now()
        .expect_err("one view still outstanding");
    let (_, owning) = error.into_parts();

    drop(view);
    assert_eq!(owning.outstanding_views(), 0);
    assert!(owning.release_now().expect("release").is_complete());
    fixture.assert_quiescent();
}

#[test]
fn a_view_that_outlives_its_owner_can_never_be_validated_or_release_anything() {
    let fixture = Fixture::new();
    let owning = fixture.binding.allocate_owning(4096, 64).expect("owning");
    let view = owning.view(64, 128).expect("view");
    drop(owning);

    let error = fixture
        .binding
        .with_view(view.view(), |_| ())
        .expect_err("a quarantined allocation has no live record");

    assert!(
        matches!(error, BindingError::StaleAllocation(_)),
        "{error:?}"
    );
    assert_eq!(fixture.allocator.release_calls(), 0);
    fixture.reclaim_retained();
}

#[test]
fn a_retired_record_cannot_be_released_twice() {
    let fixture = Fixture::new();
    fixture.allocator.set_mode(ReleaseMode::PartialQuarantine {
        unmapped: 2048,
        retained: 2048,
    });
    let allocation: BoundAllocation = fixture
        .binding
        .allocate_owning(4096, 64)
        .expect("owning")
        .into_bound()
        .expect("no outstanding views");
    let identity = allocation.identity();

    // The legacy adapter routes through the prepared path, so a partial
    // mutation comes back as an honest quarantine plus dead metadata.
    let error = fixture
        .binding
        .release(allocation)
        .expect_err("partial release is not a success");
    assert!(error.is_quarantined());
    assert_eq!(error.state(), AllocationReleaseState::PartiallyUnmapped);
    assert_eq!(
        error.outcome().and_then(AllocationReleaseOutcome::residual),
        Some(ResidualOwnership {
            state: AllocationReleaseState::PartiallyUnmapped,
            reason: QuarantineReason::PartialRelease,
            retained_bytes: 2048,
            address: error
                .outcome()
                .and_then(AllocationReleaseOutcome::residual)
                .expect("residual")
                .address,
            align: 64,
        })
    );
    assert_eq!(
        error
            .outcome()
            .and_then(AllocationReleaseOutcome::accounting),
        Some(ReleaseAccounting::new(4096, 2048)),
        "the bytes that really were unmapped are reported"
    );
    let (_, stale) = error.into_parts();
    assert_eq!(stale.identity(), identity);

    fixture.allocator.set_mode(ReleaseMode::Complete);
    let second = fixture
        .binding
        .release(stale)
        .expect_err("the record was already retired");
    assert!(
        matches!(second.error(), BindingError::StaleAllocation(_)),
        "{second:?}"
    );
    assert!(
        !second.is_quarantined(),
        "a stale rejection happens before any mutation"
    );
    assert_eq!(
        fixture.allocator.release_calls(),
        1,
        "the second attempt must never reach the allocator"
    );

    let snapshot = fixture.snapshot();
    assert_eq!(snapshot.quarantined_allocations, 1);
    assert_eq!(snapshot.quarantined_bytes, 2048);
    let removal = fixture
        .registry
        .remove(fixture.mechanism)
        .expect_err("quarantined ownership blocks removal");
    assert!(
        matches!(
            removal,
            BindingError::QuarantinedOwnership { quarantined: 1, .. }
        ),
        "{removal:?}"
    );
    fixture.reclaim_retained();
}

#[test]
fn an_allocator_failure_after_preparation_is_conservatively_quarantined() {
    let fixture = Fixture::new();
    fixture.allocator.set_mode(ReleaseMode::FailUnchanged);
    let owning = fixture.binding.allocate_owning(4096, 64).expect("owning");
    let identity = owning.identity();

    let outcome = owning.release_now().expect("prepared successfully");

    assert!(
        outcome.is_quarantined(),
        "the live record is gone, so live ownership cannot be restored: {outcome:?}"
    );
    let residual = outcome.residual().expect("residual");
    assert_eq!(residual.reason, QuarantineReason::AllocatorRefused);
    assert_eq!(residual.retained_bytes, 4096);
    assert_eq!(outcome.unmapped_bytes(), 0);
    assert_eq!(
        fixture.binding.quarantined().expect("quarantine list")[0].identity,
        identity
    );
    fixture.reclaim_retained();
}

#[test]
fn address_reuse_cannot_resurrect_a_released_generation() {
    let fixture = Fixture::new();
    let first = fixture.binding.allocate_owning(4096, 64).expect("first");
    let first_identity = first.identity();
    let stale_view = first.view(0, 64).expect("view");
    let first_address = first.as_ptr().as_ptr() as usize;
    drop(stale_view);
    assert!(first.release_now().expect("release").is_complete());

    let second = fixture.binding.allocate_owning(4096, 64).expect("second");
    let second_identity = second.identity();
    assert_ne!(
        first_identity.generation(),
        second_identity.generation(),
        "generations are never derived from an address"
    );
    if second.as_ptr().as_ptr() as usize == first_address {
        // The allocator handed back the same virtual address; the reused
        // address must still not make the earlier generation current.
        assert_ne!(first_identity, second_identity);
    }

    assert!(second.release_now().expect("release").is_complete());
    assert_eq!(fixture.allocator.release_calls(), 2);
    assert!(
        fixture.allocator.live_addresses().is_empty(),
        "every address was released exactly once"
    );
    fixture.assert_quiescent();
}

#[test]
fn device_loss_refuses_preparation_and_keeps_the_owner() {
    let fixture = Fixture::new();
    let owning = fixture.binding.allocate_owning(4096, 64).expect("owning");
    fixture
        .registry
        .invalidate_device(DeviceKey::HOST, "test-induced loss")
        .expect("invalidate");

    let error = owning
        .release_now()
        .expect_err("a lost device must not be released through");

    assert!(matches!(error.error(), BindingError::DeviceLost { .. }));
    assert_eq!(error.state(), AllocationReleaseState::Live);
    assert_eq!(
        fixture.allocator.release_calls(),
        0,
        "device loss must never call the allocator"
    );
    let (_, owning) = error.into_parts();
    assert_eq!(owning.len(), 4096, "the owner is handed back intact");
    // Device loss is terminal without an allocator call, so the record never
    // reaches the quarantine list the fixture can walk. The address is the
    // test's only remaining handle on these host bytes.
    let abandoned = owning.as_ptr();
    drop(owning);
    // SAFETY: the exact address, size and alignment the recording allocator
    // handed out; the runtime has given up ownership and no handle survives.
    unsafe { fixture.allocator.deallocate(abandoned, 4096, 64) };
}

#[test]
fn a_queued_request_becomes_device_lost_quarantine_and_keeps_its_pins() {
    let fixture = Fixture::new();
    let queue = BoundedQueue::new(4);
    let owning = fixture.binding.allocate_owning(4096, 64).expect("owning");
    let identity = owning.identity();
    assert!(
        owning
            .release_deferred(&queue)
            .expect("enqueue")
            .is_queued()
    );

    fixture
        .registry
        .invalidate_device(DeviceKey::HOST, "loss while queued")
        .expect("invalidate");
    assert_eq!(
        fixture.context_drops.load(Ordering::SeqCst),
        0,
        "a queued request keeps its provider-context pin alive"
    );

    let outcomes = queue.drain();

    assert_eq!(outcomes.len(), 1);
    let residual = outcomes[0].residual().expect("residual");
    assert_eq!(residual.state, AllocationReleaseState::DeviceLost);
    assert_eq!(residual.reason, QuarantineReason::DeviceLost);
    assert_eq!(residual.retained_bytes, 4096);
    assert_eq!(
        fixture.allocator.release_calls(),
        0,
        "device loss must never call the allocator, even from a queue"
    );
    let quarantined = fixture.binding.quarantined().expect("quarantine list");
    assert_eq!(quarantined.len(), 1);
    assert_eq!(quarantined[0].identity, identity);
    assert_eq!(quarantined[0].state, AllocationReleaseState::DeviceLost);
    fixture.reclaim_retained();
}

#[test]
fn queued_ownership_blocks_teardown_until_it_settles() {
    let fixture = Fixture::new();
    let queue = BoundedQueue::new(4);
    let owning = fixture.binding.allocate_owning(4096, 64).expect("owning");
    assert!(
        owning
            .release_deferred(&queue)
            .expect("enqueue")
            .is_queued()
    );

    fixture.registry.retire(fixture.mechanism).expect("retire");
    let blocked = fixture
        .registry
        .remove(fixture.mechanism)
        .expect_err("a queued release blocks removal");
    assert!(
        matches!(blocked, BindingError::InactiveMechanism { .. }),
        "{blocked:?}"
    );

    fixture
        .registry
        .invalidate_device(DeviceKey::HOST, "teardown")
        .expect("invalidate");
    let not_quiescent = fixture
        .registry
        .confirm_context_terminated(fixture.context)
        .expect_err("a queued release blocks context termination");
    assert!(
        matches!(not_quiescent, BindingError::ContextNotQuiescent { .. }),
        "{not_quiescent:?}"
    );
    assert_eq!(fixture.context_drops.load(Ordering::SeqCst), 0);

    let outcomes = queue.drain();
    assert!(outcomes[0].is_quarantined());
    assert_eq!(fixture.snapshot().quarantined_allocations, 1);
    // Reclaimed here rather than at the end: the confirmed termination below
    // discharges the quarantine record, after which the address is no longer
    // recoverable through the registry.
    assert_eq!(fixture.reclaim_retained(), 1);

    fixture
        .registry
        .confirm_context_terminated(fixture.context)
        .expect("termination discharges quarantined ownership");
    let terminated = fixture.snapshot();
    assert_eq!(terminated.lifecycle, MechanismLifecycle::Terminated);
    assert_eq!(terminated.quarantined_allocations, 0);
    assert_eq!(terminated.queued_releases, 0);

    fixture.registry.remove(fixture.mechanism).expect("remove");
    assert_eq!(
        fixture.allocator.release_calls(),
        0,
        "teardown never calls the allocator"
    );
    fixture.reclaim_retained();
}

#[test]
fn queue_and_allocator_callbacks_never_run_under_a_registry_or_mechanism_lock() {
    let fixture = Fixture::new();
    fixture
        .allocator
        .arm_probe(fixture.registry.clone(), fixture.mechanism);
    let queue = BoundedQueue::new(4);
    queue.arm_probe(fixture.registry.clone(), fixture.mechanism);

    let owning = fixture.binding.allocate_owning(4096, 64).expect("owning");
    assert!(
        owning
            .release_deferred(&queue)
            .expect("enqueue")
            .is_queued()
    );
    let outcomes = queue.drain();

    assert!(outcomes[0].is_complete(), "{outcomes:?}");
    assert_eq!(fixture.allocator.release_calls(), 1);
    fixture.assert_quiescent();
    fixture.allocator.disarm_probe();
    queue.disarm_probe();
}

#[test]
fn concurrent_final_releases_free_every_allocation_exactly_once() {
    let fixture = Fixture::new();
    const THREADS: usize = 8;
    const PER_THREAD: usize = 32;
    let barrier = Arc::new(Barrier::new(THREADS));

    thread::scope(|scope| {
        for _ in 0..THREADS {
            let binding = fixture.binding.clone();
            let barrier = Arc::clone(&barrier);
            scope.spawn(move || {
                let owned = (0..PER_THREAD)
                    .map(|index| {
                        binding
                            .allocate_owning(64 * (index + 1), 64)
                            .expect("owning")
                    })
                    .collect::<Vec<_>>();
                barrier.wait();
                for owning in owned {
                    let outcome = owning.release_now().expect("release");
                    assert!(outcome.is_complete(), "{outcome:?}");
                }
            });
        }
    });

    assert_eq!(fixture.allocator.release_calls(), THREADS * PER_THREAD);
    assert!(
        fixture.allocator.live_addresses().is_empty(),
        "every allocation was released exactly once"
    );
    fixture.assert_quiescent();
}

#[test]
fn deferred_stress_never_grows_the_queue_without_bound() {
    let fixture = Fixture::new();
    const CAPACITY: usize = 8;
    const ALLOCATIONS: usize = 256;
    let queue = BoundedQueue::new(CAPACITY);
    let mut completed = 0usize;

    for index in 0..ALLOCATIONS {
        if queue.pending() == CAPACITY {
            for outcome in queue.drain() {
                assert!(outcome.is_complete(), "{outcome:?}");
                completed += 1;
            }
        }
        let owning = fixture
            .binding
            .allocate_owning(64 * (index % 7 + 1), 64)
            .expect("owning");
        let disposition = owning.release_deferred(&queue).expect("enqueue");
        assert!(disposition.is_queued(), "{disposition:?}");
        assert!(
            queue.pending() <= CAPACITY,
            "the pending queue exceeded its bound: {}",
            queue.pending()
        );
        assert!(
            fixture.snapshot().queued_releases <= CAPACITY,
            "queued ownership exceeded the queue bound"
        );
    }

    for outcome in queue.drain() {
        assert!(outcome.is_complete(), "{outcome:?}");
        completed += 1;
    }

    assert_eq!(completed, ALLOCATIONS);
    assert_eq!(queue.pending(), 0);
    assert_eq!(fixture.allocator.release_calls(), ALLOCATIONS);
    assert!(
        fixture.allocator.live_addresses().is_empty(),
        "every allocation was released exactly once"
    );
    fixture.assert_quiescent();
}

#[test]
fn the_legacy_adapter_still_validates_generation_before_the_allocator_runs() {
    let fixture = Fixture::new();
    let other = fixture
        .registry
        .bind(DeviceKey::HOST)
        .expect("second binding");
    let allocation = fixture.binding.allocate(4096, 64).expect("allocation");

    let error = other
        .release(allocation)
        .expect_err("a different binding must not release this allocation");
    assert!(
        matches!(error.error(), BindingError::BindingMismatch { .. }),
        "{error:?}"
    );
    assert!(!error.is_quarantined());
    assert_eq!(fixture.allocator.release_calls(), 0);

    let (_, allocation) = error.into_parts();
    fixture.binding.release(allocation).expect("owner releases");
    assert_eq!(fixture.allocator.release_calls(), 1);
    fixture.assert_quiescent();
}
