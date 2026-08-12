#![allow(dead_code)]
//! Shared test helper: locate (or auto-build) the cpu-plugin cdylib.
//!
//! Resolution order:
//! 1. `NXRT_CPU_PLUGIN_PATH` env var (explicit override)
//! 2. `CARGO_TARGET_DIR` / profile / platform-libname
//! 3. workspace root / target / profile / platform-libname
//! 4. Auto-build via `cargo build -p onnx-runtime-ep-cpu-plugin`, then retry (3)
//!
//! The profile defaults to `"debug"`; set `PROFILE=release` to test release builds.
//! Platform-appropriate library names are used (`.so` / `.dylib` / `.dll`).

use std::path::PathBuf;

/// Platform-appropriate cdylib filename for `onnx-runtime-ep-cpu-plugin`.
fn cdylib_filename() -> &'static str {
    if cfg!(target_os = "linux") {
        "libonnx_runtime_ep_cpu_plugin.so"
    } else if cfg!(target_os = "macos") {
        "libonnx_runtime_ep_cpu_plugin.dylib"
    } else {
        "onnx_runtime_ep_cpu_plugin.dll"
    }
}

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

/// Locate the cpu-plugin cdylib, building it if absent.
///
/// Panics with an actionable message if the build itself fails.
pub fn find_cpu_plugin_cdylib() -> PathBuf {
    // 1. Explicit override
    if let Ok(p) = std::env::var("NXRT_CPU_PLUGIN_PATH") {
        let path = PathBuf::from(p);
        assert!(
            path.exists(),
            "NXRT_CPU_PLUGIN_PATH set to {path:?} but file does not exist"
        );
        return path;
    }

    let profile = std::env::var("PROFILE").unwrap_or_else(|_| "debug".to_string());
    let libname = cdylib_filename();

    // 2. CARGO_TARGET_DIR
    if let Ok(target_dir) = std::env::var("CARGO_TARGET_DIR") {
        let path = PathBuf::from(target_dir).join(&profile).join(libname);
        if path.exists() {
            return path;
        }
    }

    // 3. Workspace default
    let default_path = workspace_root().join("target").join(&profile).join(libname);
    if default_path.exists() {
        return default_path;
    }

    // 4. Auto-build
    eprintln!(
        "cpu-plugin cdylib not found — building it now \
         (cargo build -p onnx-runtime-ep-cpu-plugin)..."
    );
    let status = std::process::Command::new("cargo")
        .args(["build", "-p", "onnx-runtime-ep-cpu-plugin"])
        .status()
        .expect("failed to invoke cargo");
    assert!(
        status.success(),
        "cargo build -p onnx-runtime-ep-cpu-plugin FAILED. \
         Cannot run integration tests without the cdylib."
    );

    // Re-check after build
    let path = workspace_root().join("target").join(&profile).join(libname);
    assert!(
        path.exists(),
        "cpu-plugin cdylib still not found at {path:?} after build. \
         Set NXRT_CPU_PLUGIN_PATH if using a custom target dir."
    );
    path
}

/// Same as `find_cpu_plugin_cdylib` but returns `Option` for tests that skip
/// when the cdylib is absent (e.g. e2e tests that also need ORT).
pub fn find_cpu_plugin_cdylib_optional() -> Option<PathBuf> {
    // 1. Explicit override
    if let Ok(p) = std::env::var("NXRT_CPU_PLUGIN_PATH") {
        let path = PathBuf::from(p);
        if path.exists() {
            return Some(path);
        }
        return None;
    }

    let profile = std::env::var("PROFILE").unwrap_or_else(|_| "debug".to_string());
    let libname = cdylib_filename();

    // 2. CARGO_TARGET_DIR
    if let Ok(target_dir) = std::env::var("CARGO_TARGET_DIR") {
        let path = PathBuf::from(target_dir).join(&profile).join(libname);
        if path.exists() {
            return Some(path);
        }
    }

    // 3. Workspace default
    let path = workspace_root().join("target").join(&profile).join(libname);
    if path.exists() {
        return Some(path);
    }

    // 4. Auto-build
    eprintln!(
        "cpu-plugin cdylib not found — building it now \
         (cargo build -p onnx-runtime-ep-cpu-plugin)..."
    );
    let status = std::process::Command::new("cargo")
        .args(["build", "-p", "onnx-runtime-ep-cpu-plugin"])
        .status()
        .expect("failed to invoke cargo");
    if !status.success() {
        eprintln!("cargo build -p onnx-runtime-ep-cpu-plugin failed; skipping test");
        return None;
    }

    let path = workspace_root().join("target").join(&profile).join(libname);
    if path.exists() { Some(path) } else { None }
}
