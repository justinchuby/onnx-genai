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
use onnx_runtime_memory_host::{AllocatorCore as PluginAllocatorCore, MemoryPlugin, PluginModule};

#[path = "support/testplugin.rs"]
mod testplugin;
use testplugin::drain_calls;

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
    // Opening the allocator took two module references: one for the core and
    // one for the callback context's bridge. Dropping the allocator must
    // release the first and **retain** the second, because the retained one is
    // inside the very box the host has decided not to free.
    let pins_before = std::sync::Arc::strong_count(plugin.module());
    let drain_calls_before = drain_calls(&plugin);
    drop(allocator);
    assert_eq!(
        PluginAllocatorCore::leaked_callback_tables() - before,
        1,
        "a table a queued release can still name must be leaked, not freed"
    );
    // The drain loop must stop the moment a pass retires nothing. `sticky`
    // never retires, so the first pass already proves further passes are
    // pointless: a host that kept asking would cross the ABI boundary fifteen
    // more times to be told the same thing, while the caller waits inside
    // `drop`. Counted from inside the module, so this is what the host really
    // did rather than what it believes it did.
    assert_eq!(
        drain_calls(&plugin) - drain_calls_before,
        1,
        "a pass that retires nothing must end the loop, not restart it"
    );
    // This is the assertion that does not go through the leak counter at all.
    // The leaked box owns an `Arc<PluginModule>`; if the box were freed rather
    // than forgotten, that reference would be released here and the count
    // would fall by two instead of one. It is the box's *contents* surviving
    // that is under test, which is what a plugin thread dereferencing
    // `host_ctx` actually depends on.
    assert_eq!(
        std::sync::Arc::strong_count(plugin.module()),
        pins_before - 1,
        "only the core's module reference may be released; the leaked callback \
         table still owns its own"
    );

    // The module must also never reach `dlclose`: the leaked table's own
    // module reference is what guarantees that, whatever the platform would
    // otherwise do with a refcount-zero mapping.
    let unmapped_before = PluginModule::modules_unmapped();
    drop(plugin);
    assert_eq!(
        PluginModule::modules_unmapped(),
        unmapped_before,
        "a module a queued release can still re-enter must not be unmapped"
    );
}
