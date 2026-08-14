#![allow(dead_code)]
//! Thin shim over [`onnx_runtime_ort_testkit::find_plugin_cdylib`].
//!
//! The resolution logic (env override → `cargo build -p …` → target dir scan,
//! memoised per package) lives in the testkit crate so every plugin's tests
//! share one implementation. This file only pins the package name and adapts
//! the two return shapes this crate's tests already use.

use std::path::PathBuf;

/// Cargo package that produces the cdylib these tests load.
const PACKAGE: &str = "onnx-runtime-ep-cpu-plugin";

/// Locate the cpu-plugin cdylib, building it if needed.
///
/// # Panics
///
/// Panics with an actionable message when the cdylib cannot be produced.
pub fn find_cpu_plugin_cdylib() -> PathBuf {
    onnx_runtime_ort_testkit::find_plugin_cdylib(PACKAGE).unwrap_or_else(|| {
        panic!(
            "{PACKAGE} cdylib could not be located or built. \
             Set NXRT_CPU_PLUGIN_PATH to point at a prebuilt library."
        )
    })
}

/// Same as [`find_cpu_plugin_cdylib`] but returns `None` for tests that skip
/// when the cdylib is absent (e.g. e2e tests that also need real ORT).
pub fn find_cpu_plugin_cdylib_optional() -> Option<PathBuf> {
    onnx_runtime_ort_testkit::find_plugin_cdylib(PACKAGE)
}
