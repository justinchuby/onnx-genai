//! Shared helper: locate (and build) the nxmem test plugin cdylib.
//!
//! Included with `#[path]` by each integration-test binary. Some binaries use
//! only `testplugin_path`, so the helpers are allowed to be unused.
#![allow(dead_code)]

use std::path::PathBuf;

/// Locate the test plugin cdylib, building it if it is not there yet.
///
/// Resolution order mirrors the existing nxrt ABI tests: an explicit override,
/// then `CARGO_TARGET_DIR`, then the workspace default layout. If the library
/// genuinely cannot be produced the helper **panics loudly** — it never lets a
/// test pass by quietly doing nothing.
pub fn testplugin_path() -> PathBuf {
    if let Ok(explicit) = std::env::var("NXMEM_TESTPLUGIN_PATH") {
        let path = PathBuf::from(explicit);
        assert!(
            path.exists(),
            "NXMEM_TESTPLUGIN_PATH names {path:?}, which does not exist"
        );
        return path;
    }

    static PATH: std::sync::OnceLock<PathBuf> = std::sync::OnceLock::new();
    PATH.get_or_init(build_testplugin).clone()
}

/// Build the cdylib and return where it landed.
///
/// The build is unconditional. Cargo builds only the `rlib` target of a
/// dev-dependency, so the `cdylib` on disk can easily be stale — and a test
/// suite silently exercising a stale artifact is worse than one that takes an
/// extra second. If the build cannot be done, this panics loudly rather than
/// letting anything pass by default.
pub fn build_testplugin() -> PathBuf {
    let profile = std::env::var("PROFILE").unwrap_or_else(|_| String::from("debug"));
    let libname = if cfg!(target_os = "linux") {
        "libonnx_runtime_memory_testplugin.so"
    } else if cfg!(target_os = "macos") {
        "libonnx_runtime_memory_testplugin.dylib"
    } else {
        "onnx_runtime_memory_testplugin.dll"
    };

    let mut command = std::process::Command::new(
        std::env::var("CARGO").unwrap_or_else(|_| String::from("cargo")),
    );
    command.args(["build", "-p", "onnx-runtime-memory-testplugin"]);
    if profile != "debug" {
        command.args(["--profile", &profile]);
    }
    let status = command
        .status()
        .expect("cargo must be runnable to build the nxmem test plugin");
    assert!(status.success(), "building the nxmem test plugin failed");

    let mut candidates = Vec::new();
    if let Ok(target_dir) = std::env::var("CARGO_TARGET_DIR") {
        candidates.push(PathBuf::from(target_dir).join(&profile).join(libname));
    }
    candidates.push(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(|crates| crates.parent())
            .expect("the crate lives two levels below the workspace root")
            .join("target")
            .join(&profile)
            .join(libname),
    );
    for candidate in &candidates {
        if candidate.exists() {
            return candidate.clone();
        }
    }
    panic!(
        "the nxmem test plugin cdylib is missing after a successful build; looked in \
         {candidates:?}. Set NXMEM_TESTPLUGIN_PATH when using a custom target directory."
    );
}

/// How many times the host has entered the plugin's `drain_releases`.
///
/// Read out of the **loaded module**, because how many times the host crosses
/// the ABI boundary is a property of the host's own loop that no host-side
/// counter could honestly report — the host would only be reading back its own
/// belief about what it did.
pub fn drain_calls(plugin: &onnx_runtime_memory_host::MemoryPlugin) -> u64 {
    // SAFETY: `NxmemTestpluginDrainCalls` is an `extern "C" fn() -> u64`
    // exported by the test plugin, and the module stays mapped for as long as
    // the borrow of `plugin` lasts.
    let symbol: libloading::Symbol<'_, unsafe extern "C" fn() -> u64> = unsafe {
        plugin
            .module()
            .library()
            .get(onnx_runtime_memory_testplugin::SYMBOL_DRAIN_CALLS)
    }
    .expect("the test plugin exports its drain-call counter");
    // SAFETY: as above.
    unsafe { symbol() }
}

/// How many terminal free calls the host has made into the plugin.
///
/// Read out of the **loaded module** for the same reason as [`drain_calls`]:
/// whether the host actually handed an allocation back is only visible from
/// the far side of the boundary.
pub fn terminal_releases(plugin: &onnx_runtime_memory_host::MemoryPlugin) -> u64 {
    // SAFETY: `NxmemTestpluginTerminalReleases` is an `extern "C" fn() -> u64`
    // exported by the test plugin, and the module stays mapped for as long as
    // the borrow of `plugin` lasts.
    let symbol: libloading::Symbol<'_, unsafe extern "C" fn() -> u64> = unsafe {
        plugin
            .module()
            .library()
            .get(onnx_runtime_memory_testplugin::SYMBOL_TERMINAL_RELEASES)
    }
    .expect("the test plugin exports its terminal-release counter");
    // SAFETY: as above.
    unsafe { symbol() }
}

/// How many of the host's terminal free calls went through the **minor-1**
/// structured slot rather than the baseline `deallocate` slot.
pub fn structured_releases(plugin: &onnx_runtime_memory_host::MemoryPlugin) -> u64 {
    // SAFETY: `NxmemTestpluginStructuredReleases` is an
    // `extern "C" fn() -> u64` exported by the test plugin, and the module
    // stays mapped for as long as the borrow of `plugin` lasts.
    let symbol: libloading::Symbol<'_, unsafe extern "C" fn() -> u64> = unsafe {
        plugin
            .module()
            .library()
            .get(onnx_runtime_memory_testplugin::SYMBOL_STRUCTURED_RELEASES)
    }
    .expect("the test plugin exports its structured-release counter");
    // SAFETY: as above.
    unsafe { symbol() }
}

/// Whether the allocator vtable the plugin most recently published carries a
/// populated `release_allocation`: `1` yes, `0` no, `u64::MAX` no live vtable.
///
/// The host cannot answer this for itself. It clamps the prefix it reads, so a
/// slot it declined to adopt looks exactly like a slot that was never offered,
/// and an assertion that the host stayed out of the slot proves nothing until
/// the slot is known to have been there.
pub fn published_structured_slot(plugin: &onnx_runtime_memory_host::MemoryPlugin) -> u64 {
    // SAFETY: `NxmemTestpluginPublishedStructuredSlot` is an
    // `extern "C" fn() -> u64` exported by the test plugin, and the module
    // stays mapped for as long as the borrow of `plugin` lasts.
    let symbol: libloading::Symbol<'_, unsafe extern "C" fn() -> u64> = unsafe {
        plugin
            .module()
            .library()
            .get(onnx_runtime_memory_testplugin::SYMBOL_PUBLISHED_STRUCTURED_SLOT)
    }
    .expect("the test plugin exports its published-slot accessor");
    // SAFETY: as above.
    unsafe { symbol() }
}

/// Whether the plugin currently holds a parked allocator state pointer.
pub fn parked_state_is_set(plugin: &onnx_runtime_memory_host::MemoryPlugin) -> u64 {
    // SAFETY: `NxmemTestpluginParkedStateIsSet` is an
    // `extern "C" fn() -> u64` exported by the test plugin, and the module
    // stays mapped for as long as the borrow of `plugin` lasts.
    let symbol: libloading::Symbol<'_, unsafe extern "C" fn() -> u64> = unsafe {
        plugin
            .module()
            .library()
            .get(onnx_runtime_memory_testplugin::SYMBOL_PARKED_STATE_IS_SET)
    }
    .expect("the test plugin exports its parked-state accessor");
    // SAFETY: as above.
    unsafe { symbol() }
}
