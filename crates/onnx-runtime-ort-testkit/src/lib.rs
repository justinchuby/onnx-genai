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
//! | `NXRT_SKIP_PLUGIN_REBUILD=1` | Never shell out to `cargo build`; use whatever artifact already exists. |
//!
//! The cargo profile, target directory, and `--target` triple are **derived
//! from the running test binary** ([`build_layout`]), never guessed from
//! `PROFILE` (which cargo only sets for build scripts, so a `--release` test
//! run would have silently resolved `target/debug/...` and loaded a stale
//! cdylib).

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
/// 2. `<derived target dir>/<derived profile>/build/onnx-genai-ort-sys-*/out/ort-prebuilt/lib`
/// 3. `$CARGO_TARGET_DIR/{debug,release}/build/...`
/// 4. `<workspace>/target/{debug,release}/build/...`
///
/// Build-script output lives under `<target-dir>/<profile>/build/` even when
/// `--target` is used, so the triple is deliberately not part of the path.
pub fn find_ort_lib_dir() -> Option<PathBuf> {
    if let Ok(dir) = std::env::var("NXRT_ORT_LIB_DIR") {
        let p = PathBuf::from(dir);
        if p.join(ort_lib_name()).exists() {
            return Some(p);
        }
    }

    let mut roots: Vec<PathBuf> = Vec::new();
    let derived = build_layout();
    if let Some(layout) = &derived {
        roots.push(layout.target_dir.join(&layout.profile_dir_name));
    }
    let mut push_profiles = |root: PathBuf| {
        for profile in ["debug", "release"] {
            roots.push(root.join(profile));
        }
    };
    if let Ok(target_dir) = std::env::var("CARGO_TARGET_DIR") {
        push_profiles(PathBuf::from(target_dir));
    }
    push_profiles(workspace_root().join("target"));

    roots
        .into_iter()
        .find_map(|root| scan_build_dir(&root.join("build")))
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

/// Where the running test binary was built, derived from `current_exe()`.
///
/// Cargo lays test binaries out as
/// `<target-dir>/[<triple>/]<profile-dir>/deps/<name>-<hash>`, and puts a
/// package's `cdylib` next to `deps` in the same `<profile-dir>`. Deriving the
/// layout from the actual executable is the only way to be right for
/// `--release`, a custom `--profile`, `--target`, and `CARGO_TARGET_DIR` at
/// once. `PROFILE` is **not** usable here: cargo sets it for build scripts
/// only, so it is absent during `cargo test` and defaulting it to `debug`
/// makes a release run load a stale debug cdylib.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BuildLayout {
    /// Directory holding this profile's artifacts (the parent of `deps/`).
    pub profile_dir: PathBuf,
    /// Cargo target root (`CARGO_TARGET_DIR` or `<workspace>/target`).
    pub target_dir: PathBuf,
    /// Profile *directory* name (`debug`, `release`, or a custom profile's dir).
    pub profile_dir_name: String,
    /// `--target` triple, when the test binary was cross-compiled.
    pub target_triple: Option<String>,
}

impl BuildLayout {
    /// Cargo arguments that reproduce this layout in a nested build.
    fn cargo_args(&self) -> Vec<String> {
        let mut args = vec![
            "--target-dir".to_string(),
            self.target_dir.display().to_string(),
        ];
        // `debug` is the `dev` profile's directory name; every other directory
        // name equals its profile name.
        match self.profile_dir_name.as_str() {
            "debug" => {}
            other => {
                args.push("--profile".to_string());
                args.push(other.to_string());
            }
        }
        if let Some(triple) = &self.target_triple {
            args.push("--target".to_string());
            args.push(triple.clone());
        }
        args
    }
}

/// Architectures that can start a Rust target triple.
///
/// Matching the architecture — rather than merely counting `-` separators — is
/// what keeps `cargo llvm-cov`'s `target/llvm-cov-target/<profile>` layout from
/// being misread: `llvm-cov-target` has three dash-separated segments and would
/// otherwise be mistaken for a triple, making the nested build pass
/// `--target llvm-cov-target` and fail with *"could not find specification for
/// target"*.
const TRIPLE_ARCHES: &[&str] = &[
    "aarch64",
    "arm",
    "armebv7r",
    "armv5te",
    "armv7",
    "armv7a",
    "armv7r",
    "i586",
    "i686",
    "loongarch64",
    "m68k",
    "mips",
    "mips64",
    "mips64el",
    "mipsel",
    "nvptx64",
    "powerpc",
    "powerpc64",
    "powerpc64le",
    "riscv32i",
    "riscv32im",
    "riscv32imac",
    "riscv32imc",
    "riscv64gc",
    "riscv64imac",
    "s390x",
    "sparc64",
    "sparcv9",
    "thumbv6m",
    "thumbv7em",
    "thumbv7m",
    "thumbv7neon",
    "wasm32",
    "wasm64",
    "x86_64",
    "x86_64h",
];

/// A path component is a target triple if it is `arch-vendor-os[-env]` and
/// `arch` is a real Rust target architecture.
fn looks_like_triple(component: &str) -> bool {
    let mut parts = component.split('-');
    let Some(arch) = parts.next() else {
        return false;
    };
    TRIPLE_ARCHES.contains(&arch) && parts.count() >= 2
}

/// Cargo writes `CACHEDIR.TAG` at the root of a target directory, which is the
/// only structural (rather than name-based) way to tell a target root from an
/// intervening `--target <triple>` directory.
fn is_target_root(dir: &Path) -> bool {
    dir.join("CACHEDIR.TAG").is_file()
}

/// Derive [`BuildLayout`] from the running test binary.
///
/// Returns `None` when `current_exe()` is unavailable or does not have the
/// expected `.../<profile>/deps/<bin>` shape (e.g. a binary copied elsewhere).
pub fn build_layout() -> Option<BuildLayout> {
    let exe = std::env::current_exe().ok()?;
    let deps_dir = exe.parent()?;
    // Integration tests live in `deps/`; a plain `--bin` lives directly in the
    // profile dir. Accept both.
    let profile_dir = if deps_dir.file_name().map(|n| n == "deps").unwrap_or(false) {
        deps_dir.parent()?
    } else {
        deps_dir
    };
    let profile_dir_name = profile_dir.file_name()?.to_str()?.to_string();
    let parent = profile_dir.parent()?;
    let parent_name = parent.file_name().and_then(|n| n.to_str()).unwrap_or("");
    // `CACHEDIR.TAG` is authoritative; the name check is only the fallback for
    // a target dir that lacks one.
    let cross_compiled = if is_target_root(parent) {
        false
    } else if parent.parent().map(is_target_root).unwrap_or(false) {
        true
    } else {
        looks_like_triple(parent_name)
    };
    let (target_dir, target_triple) = if cross_compiled {
        (
            parent.parent()?.to_path_buf(),
            Some(parent_name.to_string()),
        )
    } else {
        (parent.to_path_buf(), None)
    };
    Some(BuildLayout {
        profile_dir: profile_dir.to_path_buf(),
        target_dir,
        profile_dir_name,
        target_triple,
    })
}

fn cdylib_candidates(package: &str) -> Vec<PathBuf> {
    let libname = cdylib_filename(package);
    let mut out = Vec::new();
    // The build the current test binary came from is always the best match.
    if let Some(layout) = build_layout() {
        out.push(layout.profile_dir.join(&libname));
    }
    // Fallbacks for an unusual layout: keep looking under both plausible
    // target roots, but only for the profile we actually derived (defaulting
    // to `debug` only when nothing could be derived at all).
    let profile = build_layout()
        .map(|l| l.profile_dir_name)
        .unwrap_or_else(|| "debug".to_string());
    if let Ok(target_dir) = std::env::var("CARGO_TARGET_DIR") {
        out.push(PathBuf::from(target_dir).join(&profile).join(&libname));
    }
    out.push(
        workspace_root()
            .join("target")
            .join(&profile)
            .join(&libname),
    );
    out.dedup();
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
        let mut cmd = std::process::Command::new(
            std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_string()),
        );
        cmd.args(["build", "-p", package]);
        // Build into the same target dir / profile / triple this test binary
        // came from, or the "rebuild" would refresh a cdylib we never load.
        if let Some(layout) = build_layout() {
            cmd.args(layout.cargo_args());
        }
        match cmd.status() {
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

    /// The derived layout must describe the build this very test binary came
    /// from — otherwise a `--release` run resolves a stale `debug` cdylib.
    #[test]
    fn build_layout_matches_the_running_test_binary() {
        let layout = build_layout().expect("build_layout must resolve for a cargo test binary");
        let exe = std::env::current_exe().expect("current_exe");
        assert!(
            exe.starts_with(&layout.profile_dir),
            "profile dir {:?} is not an ancestor of the test binary {:?}",
            layout.profile_dir,
            exe
        );
        assert!(
            exe.starts_with(&layout.target_dir),
            "target dir {:?} is not an ancestor of the test binary {:?}",
            layout.target_dir,
            exe
        );
        assert!(
            !layout.profile_dir_name.is_empty(),
            "profile dir name must not be empty"
        );
        // `cfg!(debug_assertions)` is the only profile fact a test can check
        // without trusting the same derivation it is validating.
        if cfg!(debug_assertions) {
            assert_ne!(
                layout.profile_dir_name, "release",
                "a debug-assertions build cannot have come from target/release"
            );
        }
    }

    /// The nested `cargo build` must target the same directory/profile/triple,
    /// or the "always rebuild" guarantee refreshes an artifact nobody loads.
    #[test]
    fn nested_cargo_args_reproduce_the_running_layout() {
        let layout = build_layout().expect("build_layout");
        let args = layout.cargo_args();
        let joined = args.join(" ");
        assert!(
            joined.contains("--target-dir"),
            "nested build must pin the target dir: {joined}"
        );
        assert!(
            args.contains(&layout.target_dir.display().to_string()),
            "nested build must use the derived target dir: {joined}"
        );
        match layout.profile_dir_name.as_str() {
            "debug" => assert!(
                !joined.contains("--profile"),
                "the dev profile is cargo's default and needs no flag: {joined}"
            ),
            other => assert!(
                args.windows(2)
                    .any(|w| w[0] == "--profile" && w[1] == other),
                "nested build must pass --profile {other}: {joined}"
            ),
        }
        match &layout.target_triple {
            Some(triple) => assert!(
                args.windows(2)
                    .any(|w| w[0] == "--target" && w[1] == *triple),
                "nested build must pass --target {triple}: {joined}"
            ),
            None => assert!(
                !joined.contains("--target "),
                "no --target may be passed for a host build: {joined}"
            ),
        }
    }

    #[test]
    fn triple_detection_accepts_triples_and_rejects_lookalikes() {
        assert!(looks_like_triple("x86_64-unknown-linux-gnu"));
        assert!(looks_like_triple("aarch64-apple-darwin"));
        assert!(looks_like_triple("x86_64-pc-windows-msvc"));
        assert!(looks_like_triple("wasm32-unknown-unknown"));
        assert!(!looks_like_triple("debug"));
        assert!(!looks_like_triple("release"));
        assert!(!looks_like_triple("bench-fast"));
        // `cargo llvm-cov` builds into `target/llvm-cov-target/<profile>`. That
        // name has a triple's shape but no architecture; reading it as one made
        // the nested build pass `--target llvm-cov-target`, which rustc rejects
        // with "could not find specification for target".
        assert!(!looks_like_triple("llvm-cov-target"));
        assert!(!looks_like_triple("some-other-dir"));
    }

    /// Whatever runner this suite is under — plain `cargo test`, `cargo
    /// llvm-cov`, or a cross build — the derived target dir must be a real
    /// cargo target root, not an intermediate directory.
    #[test]
    fn derived_target_dir_is_an_actual_cargo_target_root() {
        let layout = build_layout().expect("build_layout");
        assert!(
            is_target_root(&layout.target_dir),
            "{:?} has no CACHEDIR.TAG, so it is not a cargo target root",
            layout.target_dir
        );
        if let Some(triple) = &layout.target_triple {
            assert!(
                looks_like_triple(triple),
                "derived --target {triple} is not a target triple"
            );
        }
    }

    /// The first cdylib candidate must sit next to this test binary's `deps`
    /// directory — the artifact a nested build actually refreshes.
    #[test]
    fn cdylib_candidates_lead_with_the_running_profile_dir() {
        let layout = build_layout().expect("build_layout");
        let candidates = cdylib_candidates("onnx-runtime-ep-cpu-plugin");
        assert_eq!(
            candidates.first(),
            Some(
                &layout
                    .profile_dir
                    .join(cdylib_filename("onnx-runtime-ep-cpu-plugin"))
            ),
            "candidates: {candidates:?}"
        );
    }
}
