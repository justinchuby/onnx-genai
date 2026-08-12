//! Shared, test-only helpers for locating a real upstream ONNX Runtime and the
//! plugin cdylibs built out of this workspace.
//!
//! # Why this crate exists
//!
//! Before this crate, `find_ort_lib_dir()` / `ort_lib_name()` were copy-pasted
//! byte-for-byte into `crates/onnx-runtime-ep-cpu-plugin/tests/common/ort_discovery.rs`
//! **and** `crates/onnx-runtime-ep-plugin/tests/common/ort_discovery.rs`, and
//! the cdylib-resolution logic was hard-coded to a single package name. Every
//! new plugin that wants real-ORT coverage would have added another copy.
//! This crate is the single source of truth; test files depend on it as a
//! `dev-dependency` instead of `#[path = ...] mod`-including a duplicate.
//!
//! The crate is `publish = false` and carries no runtime dependencies — it is
//! never linked into shipped artifacts.
//!
//! # Environment variables
//!
//! | Variable | Effect |
//! |---|---|
//! | `NXRT_ORT_LIB_DIR` | Explicit directory containing the ORT shared library. |
//! | `NXRT_REQUIRE_ORT_TESTS=1` | Turn "skip because ORT is missing" into a hard failure. Used in CI to prove the real-ORT tests actually ran. |
//! | `NXRT_<PLUGIN>_PLUGIN_PATH` | Explicit path to a plugin cdylib (see [`find_plugin_cdylib`]). |
//! | `PROFILE` | `debug` (default) or `release`; selects the cargo target subdirectory. |

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

/// Platform-appropriate filename of the upstream ONNX Runtime shared library.
pub fn ort_lib_name() -> &'static str {
    if cfg!(target_os = "windows") {
        "onnxruntime.dll"
    } else if cfg!(target_os = "macos") {
        "libonnxruntime.dylib"
    } else {
        "libonnxruntime.so"
    }
}

/// Root of the cargo workspace this crate lives in.
fn workspace_root() -> PathBuf {
    // <root>/crates/onnx-runtime-ort-testkit -> <root>
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("testkit manifest dir must be <workspace>/crates/<crate>")
        .to_path_buf()
}

/// Look for `onnx-genai-ort-sys-*/out/ort-prebuilt/lib` under a cargo `build` dir.
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

/// Locate the directory containing a real `libonnxruntime`.
///
/// Resolution order:
/// 1. `NXRT_ORT_LIB_DIR` (explicit override)
/// 2. `$CARGO_TARGET_DIR/debug/build/onnx-genai-ort-sys-*/out/ort-prebuilt/lib`
/// 3. `<workspace>/target/debug/build/onnx-genai-ort-sys-*/out/ort-prebuilt/lib`
pub fn find_ort_lib_dir() -> Option<PathBuf> {
    if let Ok(dir) = std::env::var("NXRT_ORT_LIB_DIR") {
        let p = PathBuf::from(dir);
        if p.join(ort_lib_name()).exists() {
            return Some(p);
        }
    }

    if let Ok(target_dir) = std::env::var("CARGO_TARGET_DIR") {
        let build_dir = Path::new(&target_dir).join("debug/build");
        if let Some(d) = scan_build_dir(&build_dir) {
            return Some(d);
        }
    }

    let build_dir = workspace_root().join("target/debug/build");
    scan_build_dir(&build_dir)
}

/// Full path to the ORT shared library, if one can be found.
pub fn find_ort_lib() -> Option<PathBuf> {
    find_ort_lib_dir().map(|d| d.join(ort_lib_name()))
}

/// `true` when the suite is required to actually exercise real ORT
/// (`NXRT_REQUIRE_ORT_TESTS=1`), so missing prerequisites must fail loudly
/// rather than silently skipping.
pub fn ort_tests_required() -> bool {
    std::env::var("NXRT_REQUIRE_ORT_TESTS").as_deref() == Ok("1")
}

/// Unwrap an optional prerequisite, honouring [`ort_tests_required`].
///
/// Returns `None` after printing a loud skip banner when the resource is
/// missing and skipping is permitted; panics when `NXRT_REQUIRE_ORT_TESTS=1`.
///
/// ```no_run
/// # use onnx_runtime_ort_testkit as testkit;
/// # fn test() {
/// let Some(dir) = testkit::require_or_skip(testkit::find_ort_lib_dir(), "real ORT not found")
/// else {
///     return;
/// };
/// # let _ = dir;
/// # }
/// ```
#[must_use]
pub fn require_or_skip<T>(resource: Option<T>, what: &str) -> Option<T> {
    match resource {
        Some(v) => Some(v),
        None => {
            assert!(
                !ort_tests_required(),
                "NXRT_REQUIRE_ORT_TESTS=1 but required resource unavailable — {what} cannot run"
            );
            eprintln!("\n*** SKIPPED: {what} ***\n");
            None
        }
    }
}

/// Platform-appropriate cdylib filename for a cargo package name.
///
/// `onnx-runtime-ep-cpu-plugin` → `libonnx_runtime_ep_cpu_plugin.so` on Linux.
pub fn cdylib_filename(package: &str) -> String {
    let stem = package.replace('-', "_");
    if cfg!(target_os = "linux") {
        format!("lib{stem}.so")
    } else if cfg!(target_os = "macos") {
        format!("lib{stem}.dylib")
    } else {
        format!("{stem}.dll")
    }
}

/// Environment-variable override name for a plugin package.
///
/// `onnx-runtime-ep-cpu-plugin` → `NXRT_CPU_PLUGIN_PATH`.
/// `onnx-runtime-ep-shared-mock-plugin` → `NXRT_SHARED_MOCK_PLUGIN_PATH`.
fn plugin_path_env_var(package: &str) -> String {
    let short = package
        .strip_prefix("onnx-runtime-ep-")
        .unwrap_or(package)
        .replace('-', "_")
        .to_uppercase();
    format!("NXRT_{short}_PATH")
}

fn cdylib_candidates(package: &str) -> Vec<PathBuf> {
    let profile = std::env::var("PROFILE").unwrap_or_else(|_| "debug".to_string());
    let libname = cdylib_filename(package);
    let mut out = Vec::new();
    if let Ok(target_dir) = std::env::var("CARGO_TARGET_DIR") {
        out.push(PathBuf::from(target_dir).join(&profile).join(&libname));
    }
    out.push(
        workspace_root()
            .join("target")
            .join(&profile)
            .join(&libname),
    );
    out
}

/// Locate a plugin cdylib built out of this workspace, rebuilding it first.
///
/// Resolution order:
/// 1. `NXRT_<PLUGIN>_PATH` (see [`plugin_path_env_var`]; explicit override,
///    never rebuilt)
/// 2. `cargo build -p <package>` unless `NXRT_SKIP_PLUGIN_REBUILD=1`
/// 3. `$CARGO_TARGET_DIR/<profile>/<libname>`
/// 4. `<workspace>/target/<profile>/<libname>`
///
/// # Why it always rebuilds
///
/// `cargo test -p <pkg> --test <name>` builds the *test* target and the lib
/// **rlib**, but does **not** refresh the `cdylib` artifact in `target/<profile>/`.
/// A test that merely checks "does the file exist" therefore happily loads a
/// cdylib built from older source and reports green — which is exactly how a
/// deliberately regressed executor still passed its own conformance suite
/// during development. Rebuilding unconditionally is cheap when nothing
/// changed and removes the stale-artifact failure mode entirely.
///
/// The result is memoised per package for the lifetime of the test binary, so
/// a suite with dozens of ORT tests pays for at most one `cargo build` probe.
///
/// Returns `None` when the cdylib cannot be produced. Callers that treat a
/// missing cdylib as fatal should wrap the result in [`require_or_skip`] or
/// `.expect(..)`.
pub fn find_plugin_cdylib(package: &str) -> Option<PathBuf> {
    static CACHE: OnceLock<Mutex<HashMap<String, Option<PathBuf>>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut guard = cache
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some(cached) = guard.get(package) {
        return cached.clone();
    }
    let resolved = resolve_plugin_cdylib(package);
    guard.insert(package.to_string(), resolved.clone());
    resolved
}

fn resolve_plugin_cdylib(package: &str) -> Option<PathBuf> {
    let env_var = plugin_path_env_var(package);
    if let Ok(p) = std::env::var(&env_var) {
        let path = PathBuf::from(p);
        if path.exists() {
            return Some(path);
        }
        eprintln!("{env_var} set to {path:?} but the file does not exist");
        return None;
    }

    if std::env::var("NXRT_SKIP_PLUGIN_REBUILD").as_deref() != Ok("1") {
        let status = std::process::Command::new(
            std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_string()),
        )
        .args(["build", "-p", package])
        .status();
        match status {
            Ok(s) if s.success() => {}
            // Fall through: a previously built artifact is better than nothing,
            // but only if one already exists.
            Ok(s) => eprintln!("cargo build -p {package} failed with {s}"),
            Err(e) => eprintln!("failed to invoke cargo to build {package}: {e}"),
        }
    }

    cdylib_candidates(package).into_iter().find(|p| p.exists())
}

/// Platform-correct, NUL-terminated encoding of a filesystem path for ORT APIs.
///
/// On Windows, ORT path-taking APIs (`CreateSession`,
/// `RegisterExecutionProviderLibrary`, …) expect `*const u16` (NUL-terminated
/// UTF-16, matching `wchar_t`). On Unix they expect `*const c_char`
/// (NUL-terminated UTF-8).
///
/// # Lifetime
///
/// The [`OrtPathBuf::as_ptr`] return borrows `self` — bind the `OrtPathBuf` to
/// a local variable that outlives every FFI call that uses the pointer.
pub struct OrtPathBuf {
    #[cfg(windows)]
    buf: Vec<u16>,
    #[cfg(not(windows))]
    buf: std::ffi::CString,
}

impl OrtPathBuf {
    /// Encode `path` into the platform-correct ORT representation.
    ///
    /// # Panics
    ///
    /// Panics if the path contains an interior NUL byte (which would be
    /// invalid for any OS path anyway), or — on Unix — is not valid UTF-8.
    pub fn new(path: impl AsRef<Path>) -> Self {
        let path = path.as_ref();
        #[cfg(windows)]
        {
            use std::os::windows::ffi::OsStrExt;
            let mut wide: Vec<u16> = path.as_os_str().encode_wide().collect();
            assert!(
                !wide.contains(&0),
                "ORT path contains interior NUL: {path:?}"
            );
            wide.push(0);
            Self { buf: wide }
        }
        #[cfg(not(windows))]
        {
            let s = path
                .to_str()
                .unwrap_or_else(|| panic!("ORT path is not valid UTF-8: {path:?}"));
            Self {
                buf: std::ffi::CString::new(s)
                    .unwrap_or_else(|_| panic!("ORT path contains interior NUL: {path:?}")),
            }
        }
    }

    /// Pointer suitable for passing to ORT `ORTCHAR_T*` parameters.
    #[cfg(windows)]
    pub fn as_ptr(&self) -> *const u16 {
        self.buf.as_ptr()
    }

    /// Pointer suitable for passing to ORT `ORTCHAR_T*` parameters.
    #[cfg(not(windows))]
    pub fn as_ptr(&self) -> *const std::os::raw::c_char {
        self.buf.as_ptr()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cdylib_filename_maps_package_to_platform_name() {
        let name = cdylib_filename("onnx-runtime-ep-cpu-plugin");
        if cfg!(target_os = "linux") {
            assert_eq!(name, "libonnx_runtime_ep_cpu_plugin.so");
        } else if cfg!(target_os = "macos") {
            assert_eq!(name, "libonnx_runtime_ep_cpu_plugin.dylib");
        } else {
            assert_eq!(name, "onnx_runtime_ep_cpu_plugin.dll");
        }
    }

    #[test]
    fn plugin_path_env_var_strips_the_common_prefix() {
        assert_eq!(
            plugin_path_env_var("onnx-runtime-ep-cpu-plugin"),
            "NXRT_CPU_PLUGIN_PATH"
        );
        assert_eq!(
            plugin_path_env_var("onnx-runtime-ep-shared-mock-plugin"),
            "NXRT_SHARED_MOCK_PLUGIN_PATH"
        );
    }

    #[test]
    fn workspace_root_contains_the_root_manifest() {
        assert!(
            workspace_root().join("Cargo.toml").exists(),
            "workspace_root() = {:?} has no Cargo.toml",
            workspace_root()
        );
    }

    #[test]
    fn ort_lib_name_is_platform_correct() {
        let name = ort_lib_name();
        assert!(name.contains("onnxruntime"), "unexpected name {name}");
    }
}
