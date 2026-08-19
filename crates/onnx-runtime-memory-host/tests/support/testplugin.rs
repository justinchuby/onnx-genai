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
