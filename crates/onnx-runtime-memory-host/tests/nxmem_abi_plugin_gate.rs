//! The *plugin's* half of the unload gate, in a process of its own.
//!
//! The gate has two independent halves: what the host knows it is holding, and
//! what the plugin says it still owns. The host's half is easy to exercise
//! because the host counts it. The plugin's half is the one that matters most
//! and is the hardest to reach, because in every ordinary scenario the host is
//! still holding something too — so the host's check fires first and the
//! plugin's check is never the reason for the refusal.
//!
//! `self-retaining` closes that gap. It takes a reference to its own allocator
//! state at open and never drops it, modelling a plugin that keeps an
//! allocator alive for its own worker thread after the host has let go. Once
//! the host drops its handle the host's tallies are all zero, so the *only*
//! thing that can refuse the unload is the plugin's report.
//!
//! That state is never released, so the module's process-wide counters stay
//! poisoned for the rest of the process. Hence a binary of its own.

use onnx_runtime_memory_abi::NXMEM_CAP_ALLOCATOR;
use onnx_runtime_memory_host::{MemoryPlugin, PluginModule};

#[path = "support/testplugin.rs"]
mod testplugin;

/// **Unload is refused on the plugin's word alone.**
///
/// With the host's own tallies at zero, a host that trusts only itself would
/// happily `dlclose` a module that still owns an allocator — and the plugin's
/// worker thread would then run unmapped code.
#[test]
fn unload_is_refused_when_only_the_plugin_still_owns_something() {
    let plugin = MemoryPlugin::load(testplugin::testplugin_path()).expect("the test plugin loads");
    let allocator = plugin
        .factory("self-retaining")
        .expect("the mechanism is published")
        .open(NXMEM_CAP_ALLOCATOR, None)
        .expect("the mechanism opens");

    // Let go of everything the host tracks. The plugin's extra reference means
    // its state survives, and `release` therefore does not tear it down.
    drop(allocator);

    let rejection = plugin
        .try_unload()
        .expect_err("a plugin that still owns an allocator must block unload");

    // This is the whole point: the host has nothing left to complain about, so
    // if the refusal did not come from the plugin's report it did not happen.
    assert_eq!(
        rejection.host.allocators, 0,
        "the host must have let go of its allocator: {:?}",
        rejection.host
    );
    assert_eq!(
        rejection.host.allocations, 0,
        "and of its allocations: {:?}",
        rejection.host
    );
    assert_eq!(
        rejection.host.queued_releases, 0,
        "and of its queued releases: {:?}",
        rejection.host
    );
    assert_eq!(
        rejection.host.total(),
        0,
        "so the host's half of the gate is open: {:?}",
        rejection.host
    );

    assert_eq!(
        rejection.report.live_allocators, 1,
        "the plugin still owns the allocator, and that must be what refuses: {:?}",
        rejection.report
    );
    assert!(
        rejection.report.total() >= 1,
        "the plugin's report is the only thing keeping the module mapped: {:?}",
        rejection.report
    );

    // And the refusal is honest about being unrecoverable from here: the
    // plugin comes back, but nothing the host can do will retire the
    // plugin-side reference, so the only exit is a drop that keeps the module
    // mapped.
    let leaks_before = MemoryPlugin::forced_module_leaks();
    let unmapped_before = PluginModule::modules_unmapped();
    let plugin = rejection
        .into_plugin()
        .expect("the refusal hands the plugin back");
    drop(plugin);
    assert_eq!(
        MemoryPlugin::forced_module_leaks() - leaks_before,
        1,
        "a module the plugin still owns objects in must stay mapped"
    );
    // The counter above only records the *decision*. This one records what
    // actually happened to the mapping: `PluginModule::drop` runs if and only
    // if the last strong reference went, and dropping its `library` field is
    // the `dlclose`. Nothing outside this handle pinned the module — the host
    // let go of its allocator before `try_unload` — so if the drop above did
    // not deliberately retain a reference, the module really would unmap here
    // and the plugin's own worker would be left running unmapped code.
    assert_eq!(
        PluginModule::modules_unmapped(),
        unmapped_before,
        "the module must not reach dlclose while the plugin still owns an allocator"
    );
}
