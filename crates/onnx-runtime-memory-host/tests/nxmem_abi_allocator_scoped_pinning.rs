//! The callback table's pinning is *per allocator*, in a process of its own.
//!
//! Like the other pinning binaries this uses `sticky`, which never retires a
//! queued release and therefore poisons the module's process-wide counters for
//! the rest of the process. Sharing a process with another such test would
//! make every assertion below read the other's residue.

use onnx_runtime_memory_abi::{NXMEM_CAP_ALLOCATOR, NXMEM_CAP_DEFERRED_RELEASE};
use onnx_runtime_memory_api::DeviceAllocator;
use onnx_runtime_memory_host::{AllocatorCore as PluginAllocatorCore, MemoryPlugin};

#[path = "support/testplugin.rs"]
mod testplugin;

/// **One allocator's queued release must not pin another allocator's table.**
///
/// Each allocator owns its own callback table, and the plugin stores a raw
/// pointer to *that* table in *that* allocator's state. So the question "may
/// this table be freed" is per-allocator: it is answered by the releases
/// queued against this allocator, not by everything the module has queued.
///
/// The module-wide tally answers a different question — "may the module
/// unload" — and the two genuinely diverge, because an allocator can be
/// dropped long before the module is. This test is what makes that divergence
/// real rather than asserted in a comment: two allocators are open on one
/// module, only one of them has a queued release, and the other is dropped.
/// A host that consulted the module-wide tally would leak a table that nothing
/// can ever name again.
#[test]
fn a_queued_release_on_one_allocator_does_not_pin_anothers_table() {
    let plugin = MemoryPlugin::load(testplugin::testplugin_path()).expect("the test plugin loads");

    // `sticky` refuses to retire, so its queued release stays outstanding for
    // the rest of the process and keeps the module-wide tally non-zero.
    let sticky = plugin
        .factory("sticky")
        .expect("the mechanism is published")
        .open(NXMEM_CAP_ALLOCATOR | NXMEM_CAP_DEFERRED_RELEASE, None)
        .expect("the mechanism opens");
    // `lazy` is an ordinary, well-behaved mechanism on the same module, and
    // nothing is ever queued against it.
    let lazy = plugin
        .factory("lazy")
        .expect("the mechanism is published")
        .open(NXMEM_CAP_ALLOCATOR | NXMEM_CAP_DEFERRED_RELEASE, None)
        .expect("the mechanism opens");

    let ptr = sticky.allocate(4096, 64).expect("allocation");
    // SAFETY: a live allocation from this allocator with matching parameters.
    let _ticket = unsafe { sticky.enqueue_release(ptr, 4096, 64) }.expect("queued release");

    assert_eq!(
        sticky.core().outstanding_releases(),
        1,
        "the queued release belongs to `sticky`"
    );
    assert_eq!(
        lazy.core().outstanding_releases(),
        0,
        "and to `sticky` only; nothing was ever queued against `lazy`"
    );
    assert_eq!(
        plugin.module().host_live_counts().queued_releases,
        1,
        "while the module-wide tally sees it, which is the whole divergence"
    );

    // Dropping `lazy` must free `lazy`'s table. Nothing the plugin holds can
    // name it: `sticky`'s queued release stores `sticky`'s callback pointer.
    let leaks_before = PluginAllocatorCore::leaked_callback_tables();
    drop(lazy);
    assert_eq!(
        PluginAllocatorCore::leaked_callback_tables(),
        leaks_before,
        "a release queued against a different allocator must not leak this \
         allocator's callback table"
    );

    // And the converse, so this cannot pass by never leaking at all: the
    // allocator that *does* own the queued release still leaks its table.
    drop(sticky);
    assert_eq!(
        PluginAllocatorCore::leaked_callback_tables() - leaks_before,
        1,
        "the allocator whose own release is outstanding must still leak its table"
    );
}
