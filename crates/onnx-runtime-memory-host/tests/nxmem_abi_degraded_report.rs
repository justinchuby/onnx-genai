//! The degraded path: a plugin that cannot say what it still owns.
//!
//! Both gates — [`MemoryPlugin::try_unload`] and `Drop` — ask the plugin for
//! an unload report before deciding. Everything else in this suite exercises
//! the case where that question is *answered*. This exercises the case where
//! it fails, which is the one where the decision is least obvious and most
//! consequential: "I do not know what I still own" must be treated exactly
//! like "I still own something", because unmapping on a failed query is a
//! guess about executable memory a live object may still enter.
//!
//! This needs a binary of its own. The refusal is a process-global toggle
//! inside the loaded module, and it is deliberately one-way — a module that
//! has been told to refuse cannot be trusted to stop, and it will be leaked
//! rather than unmapped, so nothing else may share the process with it.

use onnx_runtime_memory_host::{MemoryPlugin, PluginModule};

#[path = "support/testplugin.rs"]
mod testplugin;

/// Ask the loaded module to start failing its unload-readiness query.
fn make_the_plugin_refuse_to_report(plugin: &MemoryPlugin) {
    // SAFETY: `NxmemTestpluginRefuseUnloadReport` is an `extern "C" fn()`
    // exported by the test plugin, and the module stays mapped for the borrow.
    let symbol: libloading::Symbol<'_, unsafe extern "C" fn()> = unsafe {
        plugin
            .module()
            .library()
            .get(onnx_runtime_memory_testplugin::SYMBOL_REFUSE_UNLOAD_REPORT)
    }
    .expect("the test plugin exports its readiness-refusal switch");
    // SAFETY: as above.
    unsafe { symbol() };
}

/// **A plugin that cannot report readiness is never unmapped.**
///
/// Nothing is live here on either side: the host holds no allocators and the
/// module owns no objects. The *only* thing standing between this module and
/// `dlclose` is that it declined to answer. A host that treated a failed query
/// as "nothing is live" would unmap on exactly the evidence that says it must
/// not, and would do so silently — which is why this asserts both that the
/// unmap did not happen and that the refusal was recorded as a deliberate
/// leak rather than as an ordinary clean exit.
#[test]
fn a_plugin_that_cannot_report_readiness_is_neither_unloaded_nor_unmapped() {
    let plugin = MemoryPlugin::load(testplugin::testplugin_path()).expect("the test plugin loads");
    assert_eq!(
        plugin.module().host_live_counts().total(),
        0,
        "the host is holding nothing, so only the plugin's answer can matter"
    );
    plugin
        .module()
        .unload_report()
        .expect("the module answers before it is asked to stop");

    make_the_plugin_refuse_to_report(&plugin);
    assert!(
        plugin.module().unload_report().is_err(),
        "the switch must actually have taken effect, or this test proves nothing"
    );

    // `try_unload` has a channel to refuse through, and must use it.
    let rejection = plugin
        .try_unload()
        .expect_err("a plugin that cannot report readiness must not be unloaded");
    assert!(
        rejection
            .reason
            .contains("could not report unload readiness"),
        "the refusal must name the reason, got: {}",
        rejection.reason
    );
    let plugin = rejection
        .into_plugin()
        .expect("the caller gets the handle back; this refusal is not terminal");

    // `Drop` has no channel to refuse through, so it takes the only safe
    // option left and keeps the module mapped forever.
    let leaks_before = MemoryPlugin::forced_module_leaks();
    let unmapped_before = PluginModule::modules_unmapped();
    drop(plugin);
    assert_eq!(
        PluginModule::modules_unmapped(),
        unmapped_before,
        "a module that could not say what it still owns must not reach dlclose"
    );
    assert_eq!(
        MemoryPlugin::forced_module_leaks() - leaks_before,
        1,
        "and the leak must be recorded, so it is a reported cost rather than a silent one"
    );
}
