//! Deferred release must keep the provider/context pinned, not just the module.
//!
//! The module half of that contract was already enforced: a queued release
//! holds an `Arc<PluginModule>`, so the plugin's code stays mapped. The
//! provider half is a different failure. A stream-ordered free retires against
//! whatever owns the device resources — a CUDA context, an arena — and if that
//! owner finished teardown first, the plugin unmaps handles teardown already
//! released. **Nothing on the Rust side reports this.** `release_completed`
//! touches only host state, which deliberately outlives the allocator, so the
//! host stays quiet while the driver is handed a double free.
//!
//! That silence is the reason these tests assert on teardown *blocking* rather
//! than on an error being returned. There is no error to observe.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use onnx_runtime_memory_abi::{NXMEM_CAP_ALLOCATOR, NXMEM_CAP_DEFERRED_RELEASE};
use onnx_runtime_memory_api::{DeviceAllocator, DeviceKey};
use onnx_runtime_memory_governor::{ProcessMemoryManager, RegisteredMemoryContext};
use onnx_runtime_memory_host::{MemoryPlugin, PluginAllocator};

#[path = "support/testplugin.rs"]
mod testplugin;

/// `BindingResource` is blanket-implemented for `Send + Sync + Debug`, so this
/// needs no impl of its own; it exists to give the context a distinct owner.
#[derive(Debug, Default)]
struct ContextResource;

fn load() -> MemoryPlugin {
    MemoryPlugin::load(testplugin::testplugin_path()).expect("the test plugin loads")
}

fn context(manager: &ProcessMemoryManager) -> RegisteredMemoryContext {
    manager
        .register_provider_context(DeviceKey::HOST, "plugin context", Arc::new(ContextResource))
        .expect("the context registers")
}

fn open_bound(
    plugin: &MemoryPlugin,
    mechanism: &str,
    context: &RegisteredMemoryContext,
) -> PluginAllocator {
    plugin
        .factory(mechanism)
        .expect("the mechanism is published")
        .open_with_provider_context(
            NXMEM_CAP_ALLOCATOR | NXMEM_CAP_DEFERRED_RELEASE,
            None,
            context.pin_source(),
        )
        .expect("the mechanism opens")
}

/// Teardown must wait for a queued release, and must resume once it retires.
///
/// Both halves matter and a test that checks only one is worse than no test.
/// Asserting only that teardown blocks passes for an implementation that never
/// unblocks — a deadlock reads as a successful pin. Asserting only that it
/// eventually returns passes for no pin at all.
#[test]
fn a_queued_release_blocks_provider_context_teardown_until_it_retires() {
    let manager = Arc::new(ProcessMemoryManager::new().expect("manager"));
    let registered = context(&manager);
    let plugin = load();
    let allocator = open_bound(&plugin, "lazy", &registered);

    let ptr = allocator.allocate(4096, 64).expect("allocation");
    // SAFETY: a live allocation from this allocator with matching parameters.
    let _ticket = unsafe { allocator.enqueue_release(ptr, 4096, 64) }.expect("queued release");

    let finished = Arc::new(AtomicBool::new(false));
    let teardown = {
        let manager = Arc::clone(&manager);
        let finished = Arc::clone(&finished);
        std::thread::spawn(move || {
            manager
                .remove_provider_context(&registered)
                .expect("teardown completes once the release retires");
            finished.store(true, Ordering::Release);
        })
    };

    // A negative result needs a window wide enough that "not yet" is a real
    // observation rather than a scheduling artifact.
    std::thread::sleep(Duration::from_millis(250));
    assert!(
        !finished.load(Ordering::Acquire),
        "provider-context teardown returned while a deferred release was still \
         outstanding; the plugin would retire that free against a dismantled context"
    );

    assert_eq!(
        allocator.drain_releases(64).expect("drain"),
        1,
        "the queued release retires"
    );

    teardown.join().expect("the teardown thread does not panic");
    assert!(
        finished.load(Ordering::Acquire),
        "teardown must resume once the release has retired; a pin that never \
         releases is a deadlock, not a guarantee"
    );
}

/// The control: without a queued release, the same teardown is not delayed.
///
/// This is what makes the test above evidence. `remove_provider_context` waits
/// for quiescence generally, so "teardown blocked" only implicates the release
/// if teardown is otherwise prompt on an identical allocator.
#[test]
fn provider_context_teardown_is_not_blocked_without_a_queued_release() {
    let manager = ProcessMemoryManager::new().expect("manager");
    let registered = context(&manager);
    let plugin = load();
    let allocator = open_bound(&plugin, "lazy", &registered);

    let ptr = allocator.allocate(4096, 64).expect("allocation");
    // SAFETY: a live allocation from this allocator with matching parameters.
    unsafe { allocator.deallocate(ptr, 4096, 64) };

    manager
        .remove_provider_context(&registered)
        .expect("teardown completes with nothing outstanding");
}

/// Once a context stops accepting work, a deferred release must be refused.
///
/// Queuing it unpinned would be the same defect with a narrower window: the
/// release would name a context already committed to teardown. The allocation
/// stays live so the caller can still free it through the canonical path,
/// which is the only remaining correct move.
#[test]
fn a_retiring_provider_context_refuses_further_deferred_releases() {
    let manager = ProcessMemoryManager::new().expect("manager");
    let registered = context(&manager);
    let plugin = load();
    let allocator = open_bound(&plugin, "lazy", &registered);

    let ptr = allocator.allocate(4096, 64).expect("allocation");
    manager
        .retire_context(&registered)
        .expect("the context stops accepting work");

    // SAFETY: a live allocation from this allocator with matching parameters.
    let refused = unsafe { allocator.enqueue_release(ptr, 4096, 64) };
    let error = refused.expect_err("a retiring context must refuse a deferred release");
    let message = error.to_string();
    assert!(
        message.contains("no longer accepting work"),
        "the refusal must name the cause; got: {message}"
    );

    assert_eq!(
        allocator.pending_release_count().expect("query"),
        0,
        "a refused release must not be queued"
    );
    // SAFETY: the refused enqueue left the allocation live and unqueued, with
    // its original bytes and align.
    unsafe { allocator.deallocate(ptr, 4096, 64) };
}

/// An unbound allocator keeps working exactly as before.
///
/// Standalone use — a plugin with no execution provider behind it — is the
/// case every existing ABI test exercises. It has no context to outlive, so
/// requiring one would be a regression rather than a tightening.
#[test]
fn an_unbound_allocator_still_queues_deferred_releases() {
    let plugin = load();
    let allocator = plugin
        .factory("lazy")
        .expect("the mechanism is published")
        .open(NXMEM_CAP_ALLOCATOR | NXMEM_CAP_DEFERRED_RELEASE, None)
        .expect("the mechanism opens");

    let ptr = allocator.allocate(4096, 64).expect("allocation");
    // SAFETY: a live allocation from this allocator with matching parameters.
    unsafe { allocator.enqueue_release(ptr, 4096, 64) }.expect("queued release");
    assert_eq!(allocator.drain_releases(64).expect("drain"), 1);
}
