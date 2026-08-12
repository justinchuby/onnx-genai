//! Shared ORT discovery logic for integration tests.
//!
//! Resolution order:
//! 1. `NXRT_ORT_LIB_DIR` env var (explicit override)
//! 2. `CARGO_TARGET_DIR` / debug/build / onnx-genai-ort-sys-*/out/ort-prebuilt/lib
//! 3. workspace root / target / debug/build / onnx-genai-ort-sys-*/out/ort-prebuilt/lib
//!
//! Included in integration tests via `#[path = "common/ort_discovery.rs"] mod ort_discovery;`.
//! **Keep a single copy here** — do not duplicate into individual test files.

use std::path::{Path, PathBuf};

pub fn ort_lib_name() -> &'static str {
    if cfg!(target_os = "windows") {
        "onnxruntime.dll"
    } else if cfg!(target_os = "macos") {
        "libonnxruntime.dylib"
    } else {
        "libonnxruntime.so"
    }
}

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

fn scan_build_dir(build_dir: &Path) -> Option<PathBuf> {
    let entries = std::fs::read_dir(build_dir).ok()?;
    for entry in entries.flatten() {
        let name = entry.file_name();
        if name.to_string_lossy().starts_with("onnx-genai-ort-sys-") {
            let lib_dir = entry.path().join("out/ort-prebuilt/lib");
            if lib_dir.join(ort_lib_name()).exists() {
                return Some(lib_dir);
            }
        }
    }
    None
}

pub fn find_ort_lib_dir() -> Option<PathBuf> {
    // 1. Explicit override
    if let Ok(dir) = std::env::var("NXRT_ORT_LIB_DIR") {
        let p = PathBuf::from(dir);
        if p.join(ort_lib_name()).exists() {
            return Some(p);
        }
    }

    // 2. CARGO_TARGET_DIR
    if let Ok(target_dir) = std::env::var("CARGO_TARGET_DIR") {
        let build_dir = Path::new(&target_dir).join("debug/build");
        if let Some(d) = scan_build_dir(&build_dir) {
            return d.into();
        }
    }

    // 3. Workspace default
    let build_dir = workspace_root().join("target/debug/build");
    scan_build_dir(&build_dir)
}
