//! The callback table's pinning, in a process of its own.
//!
//! Like the unload gate's terminal case, this uses the `sticky` mechanism,
//! which deliberately never retires a queued release. That permanently poisons
//! the module's process-wide counters, so any two such tests sharing a process
//! would read each other's residue and the exact assertions below would have to
//! be softened into inequalities — which is how a test stops testing anything.
//! Hence a binary of its own.

use onnx_runtime_memory_abi::{NXMEM_CAP_ALLOCATOR, NXMEM_CAP_DEFERRED_RELEASE};
use onnx_runtime_memory_api::DeviceAllocator;
use onnx_runtime_memory_host::{AllocatorCore as PluginAllocatorCore, MemoryPlugin};

#[path = "support/testplugin.rs"]
mod testplugin;

/// **A queued release keeps the host's callback table alive.**
///
/// The plugin holds a raw pointer to the callback table, captured at open, and
/// the contract explicitly permits it to report a completion from one of its
/// own threads — that is the entire reason deferred release exists. So the
/// table may only be freed once no queued release can still name it.
///
/// `sticky` never retires anything, so dropping its allocator is precisely the
/// case where freeing would be wrong. The host must leak the table instead,
/// and say so.
#[test]
fn dropping_an_allocator_with_a_queued_release_leaks_its_callback_table() {
    let plugin = MemoryPlugin::load(testplugin::testplugin_path()).expect("the test plugin loads");
    let allocator = plugin
        .factory("sticky")
        .expect("the mechanism is published")
        .open(NXMEM_CAP_ALLOCATOR | NXMEM_CAP_DEFERRED_RELEASE, None)
        .expect("the mechanism opens");

    let ptr = allocator.allocate(4096, 64).expect("allocation");
    // SAFETY: a live allocation from this allocator with matching parameters.
    let _ticket = unsafe { allocator.enqueue_release(ptr, 4096, 64) }.expect("queued release");
    assert_eq!(
        allocator.core().outstanding_releases(),
        1,
        "the queued release is what pins the table"
    );

    let before = PluginAllocatorCore::leaked_callback_tables();
    drop(allocator);
    assert_eq!(
        PluginAllocatorCore::leaked_callback_tables() - before,
        1,
        "a table a queued release can still name must be leaked, not freed"
    );
}
