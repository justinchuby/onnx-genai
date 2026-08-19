//! The unload gate's terminal case, in a process of its own.
//!
//! A plugin module's live-object counters are process-wide, because unloading
//! a module is a process-wide act. The `sticky` mechanism deliberately never
//! retires a queued release, so once this test runs the module can never
//! honestly report itself unloadable again. That is the behaviour under test —
//! and it is exactly why this case lives in its own integration-test binary
//! rather than sharing a process with the rest of the suite.

use onnx_runtime_memory_abi::{NXMEM_CAP_ALLOCATOR, NXMEM_CAP_DEFERRED_RELEASE};
use onnx_runtime_memory_api::DeviceAllocator;
use onnx_runtime_memory_host::MemoryPlugin;

#[path = "support/testplugin.rs"]
mod testplugin;

/// **Unload with live objects**, queued-free case.
///
/// The host must not unmap a module that still owes a free: the module's code
/// is what will run when the free finally retires.
#[test]
fn unload_is_refused_while_a_queued_release_has_not_retired() {
    let plugin = MemoryPlugin::load(testplugin::testplugin_path()).expect("the test plugin loads");
    let allocator = plugin
        .factory("sticky")
        .expect("the mechanism is published")
        .open(NXMEM_CAP_ALLOCATOR | NXMEM_CAP_DEFERRED_RELEASE, None)
        .expect("the mechanism opens");

    let ptr = allocator.allocate(4096, 64).expect("allocation");
    // SAFETY: live allocation with matching parameters.
    let _ticket = unsafe { allocator.enqueue_release(ptr, 4096, 64) }.expect("queued release");

    assert_eq!(
        allocator.drain_releases(64).expect("a drain is attempted"),
        0,
        "this mechanism deliberately retires nothing"
    );
    assert_eq!(
        allocator.pending_release_count().expect("query"),
        1,
        "the plugin still owes the free"
    );

    let rejection = plugin
        .try_unload()
        .expect_err("a queued release must block unload");
    assert_eq!(
        rejection.host.queued_releases, 1,
        "the host's own tally must show the queued release: {:?}",
        rejection.host
    );
    assert_eq!(
        rejection.report.queued_releases, 1,
        "and so must the plugin's: {:?}",
        rejection.report
    );
    let plugin = rejection
        .into_plugin()
        .expect("the refusal hands the plugin back so the caller can retire work");

    // Dropping the allocator lets the plugin reclaim its own storage, but the
    // host's queued tally is what keeps the gate shut, and it never clears.
    let leaks_before = MemoryPlugin::forced_module_leaks();
    drop(allocator);
    let rejection = plugin
        .try_unload()
        .expect_err("the host must not unmap a module that still owes a free");
    assert_eq!(
        rejection.report.queued_releases, 1,
        "the plugin is what still owes the free: {:?}",
        rejection.report
    );

    // The refusal hands the plugin back, and nothing here can retire the
    // release, so the only remaining exit is a drop — which must keep the
    // module mapped rather than unmap code the free will run.
    let plugin = rejection
        .into_plugin()
        .expect("the refusal returns the plugin");
    drop(plugin);
    assert_eq!(
        MemoryPlugin::forced_module_leaks() - leaks_before,
        1,
        "dropping a plugin that still owes a free must keep its module mapped"
    );
}
