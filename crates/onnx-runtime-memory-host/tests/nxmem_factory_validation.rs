//! Factory enumeration transfers the whole response before validation starts.
//! A failure in the middle must therefore release the accepted prefix, the
//! failing entry, and every unvisited suffix entry while the module is loaded.

use onnx_runtime_memory_host::MemoryPlugin;

#[path = "support/testplugin.rs"]
mod testplugin;

#[test]
fn validation_failure_releases_every_transferred_factory_exactly_once() {
    let path = testplugin::testplugin_path();
    // Keep an independent loader reference so the module's own counters remain
    // readable after `MemoryPlugin::load` returns its validation error.
    let library = unsafe { libloading::Library::new(&path) }.expect("open test plugin");
    let set_fault: libloading::Symbol<'_, unsafe extern "C" fn(u64)> = unsafe {
        library.get(onnx_runtime_memory_testplugin::SYMBOL_FACTORY_VALIDATION_FAULT_SLOT)
    }
    .expect("factory fault-injection symbol");
    let releases: libloading::Symbol<'_, unsafe extern "C" fn() -> u64> =
        unsafe { library.get(onnx_runtime_memory_testplugin::SYMBOL_FACTORY_RELEASES) }
            .expect("factory release counter");

    let before = unsafe { releases() };
    unsafe { set_fault(3) };
    let error = MemoryPlugin::load(&path).expect_err("the third factory omits its open slot");
    unsafe { set_fault(0) };

    assert!(
        error.to_string().contains("factory vtable"),
        "the injected validation failure should be the reported cause: {error}"
    );
    assert_eq!(
        unsafe { releases() } - before,
        onnx_runtime_memory_testplugin::MECHANISM_NAMES.len() as u64,
        "every transferred factory, including the failing and unvisited entries, must be released"
    );
}
