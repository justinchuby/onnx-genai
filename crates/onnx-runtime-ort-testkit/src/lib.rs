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
//! | `NXRT_ORT_LIB_DIR` | Explicit directory containing the ORT shared library. Also the way to resolve a refusal when several stale `onnx-genai-ort-sys-*` build dirs disagree about their ORT version. |
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

/// The ORT version this workspace pins, read from `ort-sys`' build script.
///
/// `ort-sys` declares `const ORT_VERSION: &str = "x.y.z"` and downloads exactly
/// that tarball, so the build script is the single source of truth. The testkit
/// carries no dependencies (it is `publish = false` and must never be linked
/// into a shipped artifact), so it reads the declaration rather than importing
/// it.
///
/// `None` means the pin could not be established — the source tree is not
/// present beside the test binary, which happens when a prebuilt binary is run
/// on a machine that no longer has the checkout. Callers must degrade
/// *visibly*, never silently.
fn pinned_ort_version() -> Option<String> {
    static PIN: OnceLock<Option<String>> = OnceLock::new();
    PIN.get_or_init(|| {
        let build_rs = workspace_root().join("crates/onnx-genai-ort/ort-sys/build.rs");
        let text = std::fs::read_to_string(build_rs).ok()?;
        parse_pinned_version(&text)
    })
    .clone()
}

/// Extract `const ORT_VERSION: &str = "..."` from `ort-sys`' build script.
///
/// Split out from the read so the parse can be tested against the real file
/// contents without a filesystem fixture.
fn parse_pinned_version(build_rs: &str) -> Option<String> {
    let line = build_rs
        .lines()
        .find(|line| line.trim_start().starts_with("const ORT_VERSION"))?;
    let value = line.split('"').nth(1)?;
    (!value.is_empty()).then(|| value.to_string())
}

/// Version recorded by the ORT release tarball itself, as extracted by
/// `ort-sys`' build script.
///
/// `<candidate>/out/ort-prebuilt/lib` is the library directory, so the marker
/// sits one level up. Absent or unreadable is reported as `None` rather than
/// guessed at: an unknown version cannot be shown to match anything.
fn candidate_ort_version(lib_dir: &Path) -> Option<String> {
    let marker = lib_dir.parent()?.join("VERSION_NUMBER");
    let text = std::fs::read_to_string(marker).ok()?;
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return None;
    }
    Some(trimmed.to_string())
}

/// Every `onnx-genai-ort-sys-*/out/ort-prebuilt/lib` under a cargo `build` dir
/// that actually holds a loadable library, in a stable order.
///
/// `read_dir` yields entries in filesystem order, which is neither sorted nor
/// stable across machines, so the names are sorted before use. Returning *all*
/// of them — rather than the first — is what lets the caller notice that a
/// stale build directory is sitting next to the current one.
fn scan_build_dir(build_dir: &Path) -> Vec<PathBuf> {
    let Ok(entries) = std::fs::read_dir(build_dir) else {
        return Vec::new();
    };
    let mut found: Vec<PathBuf> = entries
        .flatten()
        .filter(|entry| {
            entry
                .file_name()
                .to_string_lossy()
                .starts_with("onnx-genai-ort-sys-")
        })
        .map(|entry| entry.path().join("out/ort-prebuilt/lib"))
        .filter(|lib_dir| lib_dir.join(ort_lib_name()).exists())
        .collect();
    found.sort();
    found
}

/// Reduce the candidates found under one `build` dir to the single one that may
/// be used, or panic naming what was found.
///
/// A cargo `build` directory accumulates one `onnx-genai-ort-sys-<hash>` per
/// distinct feature/profile/dependency resolution, and **nothing ever removes
/// the old ones**. After an ORT version bump a developer's tree therefore holds
/// several, only one of which matches the pin — six stale `1.28.0` beside one
/// `1.29.0` is a real observed case. Taking the first `read_dir` entry picks
/// among them effectively at random.
///
/// The loud failure is `GetApi(ORT_API_VERSION=<n>) returned null`. The quiet
/// one is worse and is the reason this function exists: when the stale library
/// satisfies the requested API version — which it does whenever the pin moves
/// *backwards*, and can do across a bump — everything runs and the results are
/// attributed to the version the workspace *declares* rather than the one that
/// executed. Benchmarks are the obvious casualty, but any behavioural test
/// compared against upstream is equally affected.
///
/// So the pin decides. Candidates that are not the pinned version are not
/// eligible, however many or few there are, and a single stale extraction is
/// refused exactly like a mixed set. Only when the pin cannot be read at all
/// does this fall back to requiring the candidates to agree with each other.
fn single_candidate(build_dir: &Path) -> Option<PathBuf> {
    let candidates = scan_build_dir(build_dir);
    let versions: Vec<Option<String>> = candidates
        .iter()
        .map(|d| candidate_ort_version(d))
        .collect();
    let pin = pinned_ort_version();
    if pin.is_none() && !candidates.is_empty() {
        warn_pin_unreadable();
    }
    match resolve_candidates(&candidates, &versions, pin.as_deref()) {
        Resolved::Absent => None,
        Resolved::Chosen(dir) => Some(dir),
        Resolved::Ambiguous => panic!(
            "{}",
            ambiguity_message(build_dir, &candidates, &versions, pin.as_deref())
        ),
    }
}

/// Outcome of the selection rule.
/// Outcome of the selection rule.
///
/// `Absent` and `Ambiguous` are separate variants on purpose. A first draft
/// collapsed both into `None`, which turned every tree with no ORT at all into
/// a panic — absence is the ordinary case that must fall through to the next
/// root and then to [`require_or_skip`]. Two conditions sharing one value is
/// the same defect this whole change is about, one level down.
#[derive(Debug, PartialEq, Eq)]
enum Resolved {
    /// Nothing here; keep looking.
    Absent,
    /// Exactly one eligible ORT.
    Chosen(PathBuf),
    /// Nothing eligible, or no way to tell which is; refuse rather than pick.
    Ambiguous,
}

/// Pure core of [`single_candidate`].
///
/// Split out from the filesystem walk so the rule can be tested directly — a
/// `build` directory with a curated mix of ORT versions cannot be conjured on
/// demand inside a unit test.
fn resolve_candidates(
    candidates: &[PathBuf],
    versions: &[Option<String>],
    pin: Option<&str>,
) -> Resolved {
    if candidates.is_empty() {
        return Resolved::Absent;
    }

    if let Some(pin) = pin {
        // Several build hashes matching the pin is the normal case — different
        // profiles and feature sets each extract the same tarball — and they
        // are interchangeable, so the sorted-first one is a stable choice.
        let mut matching = candidates
            .iter()
            .zip(versions)
            .filter(|(_, version)| version.as_deref() == Some(pin))
            .map(|(dir, _)| dir.clone());
        return matching
            .next()
            .map_or(Resolved::Ambiguous, Resolved::Chosen);
    }

    // The pin is unknown, so staleness cannot be detected; fall back to the
    // weaker property that at least the candidates do not contradict each
    // other. `single_candidate` warns when it takes this path.
    if candidates.len() == 1 {
        return Resolved::Chosen(candidates[0].clone());
    }
    let first = &versions[0];
    if first.is_some() && versions.iter().all(|v| v == first) {
        Resolved::Chosen(candidates[0].clone())
    } else {
        Resolved::Ambiguous
    }
}

/// Warn, once per process, that staleness cannot be detected.
///
/// The pin is only unreadable when the source tree is gone from beside the test
/// binary, which is rare but real (a prebuilt binary shipped to another
/// machine). Degrading to the weaker "candidates must agree" rule is the right
/// behaviour, but doing it silently would leave a reader believing a check ran
/// that did not — the failure this whole change exists to prevent.
fn warn_pin_unreadable() {
    static WARNED: OnceLock<()> = OnceLock::new();
    WARNED.get_or_init(|| {
        eprintln!(
            "warning: could not read the pinned ORT version from \
             crates/onnx-genai-ort/ort-sys/build.rs; ORT discovery can no longer \
             detect a stale extraction, only a disagreement between several. \
             Set NXRT_ORT_LIB_DIR to be certain which library is loaded."
        );
    });
}

/// Diagnostic listing every candidate, the version it holds, and the pin.
///
/// Prints the whole set deliberately: the actionable facts are which
/// directories exist and which one the reader wants, and a message naming only
/// the conflict would send them back to the filesystem to find that out.
fn ambiguity_message(
    build_dir: &Path,
    candidates: &[PathBuf],
    versions: &[Option<String>],
    pin: Option<&str>,
) -> String {
    let mut msg = match pin {
        Some(pin) => format!(
            "no usable ONNX Runtime: none of the {} candidate directories under {} \
             holds the pinned version {}.\n",
            candidates.len(),
            build_dir.display(),
            pin
        ),
        None => format!(
            "ambiguous ONNX Runtime discovery: {} candidate directories under {} \
             hold different (or unidentifiable) ORT versions, and the pinned \
             version could not be read, so which one a test loads would depend \
             on filesystem ordering.\n",
            candidates.len(),
            build_dir.display()
        ),
    };
    for (dir, version) in candidates.iter().zip(versions) {
        msg.push_str(&format!(
            "  {} -> {}\n",
            version.as_deref().unwrap_or("<no VERSION_NUMBER>"),
            dir.display()
        ));
    }
    msg.push_str(
        "Rebuild so ort-sys extracts the pinned release, set NXRT_ORT_LIB_DIR to \
         the directory you mean, or `cargo clean` to drop the stale build \
         directories. Refusing to guess: a stale-but-loadable ORT runs to \
         completion and reports results under the wrong version.",
    );
    msg
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
///
/// # Panics
///
/// When a `build` directory holds several `onnx-genai-ort-sys-*` extractions
/// that disagree about their ORT version — see [`single_candidate`]. Set
/// `NXRT_ORT_LIB_DIR` to resolve it; the override is checked first and never
/// panics.
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
        .find_map(|root| single_candidate(&root.join("build")))
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
    find_plugin_cdylib_with_features(package, &[])
}

/// Same as [`find_plugin_cdylib`], but the rebuild enables `features`.
///
/// Without this, `cargo test -p <pkg> --features <f>` builds the *test* binary
/// with `<f>` and then this helper's `cargo build -p <pkg>` **overwrites** the
/// cdylib with a default-feature build. The suite then measures and asserts
/// against a library that has none of the code the feature selects — silently,
/// and in the direction that hides regressions. Callers pass the features they
/// were compiled with so the artifact under test matches the test binary.
pub fn find_plugin_cdylib_with_features(package: &str, features: &[&str]) -> Option<PathBuf> {
    static CACHE: OnceLock<Mutex<HashMap<String, Option<PathBuf>>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut guard = cache
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    // Keyed by package *and* features because they are different builds, even
    // though cargo writes both to the same `target/<profile>/<libname>`. One
    // process asking for two feature sets would therefore rebuild over itself;
    // no caller does (each test binary's feature set is a `cfg!` constant), and
    // this key at least keeps the two answers from being silently conflated.
    let key = format!("{package}\u{1}{}", features.join(","));
    if let Some(cached) = guard.get(&key) {
        return cached.clone();
    }
    let resolved = resolve_plugin_cdylib(package, features);
    guard.insert(key, resolved.clone());
    resolved
}

fn resolve_plugin_cdylib(package: &str, features: &[&str]) -> Option<PathBuf> {
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
        if !features.is_empty() {
            cmd.args(["--features", &features.join(",")]);
        }
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

    /// Build a fake cargo `build` directory holding one `onnx-genai-ort-sys-*`
    /// per `(hash, version)`, each with a library file so it counts as a real
    /// candidate. `None` writes no `VERSION_NUMBER` at all.
    fn fake_build_dir(root: &Path, candidates: &[(&str, Option<&str>)]) -> PathBuf {
        let build = root.join("build");
        for (hash, version) in candidates {
            let prebuilt = build
                .join(format!("onnx-genai-ort-sys-{hash}"))
                .join("out/ort-prebuilt");
            let lib = prebuilt.join("lib");
            std::fs::create_dir_all(&lib).expect("create fake candidate");
            std::fs::write(lib.join(ort_lib_name()), b"not a real library")
                .expect("write fake library");
            if let Some(version) = version {
                std::fs::write(prebuilt.join("VERSION_NUMBER"), version)
                    .expect("write VERSION_NUMBER");
            }
        }
        build
    }

    /// A scratch directory under the crate's own `target`, removed on drop.
    ///
    /// Deliberately not `/tmp`: the workspace target dir is already
    /// write-guaranteed for a test run, and a stray directory there is visible
    /// rather than hidden in a system-wide location.
    struct Scratch(PathBuf);

    impl Scratch {
        fn new(tag: &str) -> Self {
            let dir = workspace_root()
                .join("target/testkit-scratch")
                .join(format!("{tag}-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).expect("create scratch");
            Self(dir)
        }

        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// A version string no real ORT release will ever carry, so a test can
    /// say "not the pin" without hard-coding what the pin currently is.
    const NOT_THE_PIN: &str = "0.0.0-stale";

    fn pin_or_skip() -> String {
        pinned_ort_version().expect(
            "the workspace pin must be readable from the testkit; without it \
             every pin-based check below degrades silently",
        )
    }

    fn dirs(names: &[&str]) -> Vec<PathBuf> {
        names.iter().map(PathBuf::from).collect()
    }

    fn versions(values: &[Option<&str>]) -> Vec<Option<String>> {
        values
            .iter()
            .map(|v| v.map(std::string::ToString::to_string))
            .collect()
    }

    /// Anti-vacuity guard for every pin-based test in this module, and for the
    /// production rule itself. If the pin stops being readable the selection
    /// quietly weakens to "candidates must agree", which is exactly the
    /// property that failed to catch six stale extractions beside one current
    /// one — they agreed with each other.
    #[test]
    fn the_workspace_pin_is_readable_and_looks_like_a_version() {
        let pin = pin_or_skip();
        assert!(
            pin.split('.').count() >= 2 && pin.starts_with(char::is_numeric),
            "pin {pin:?} does not look like an ORT version"
        );
        assert_ne!(pin, NOT_THE_PIN);
    }

    #[test]
    fn the_pin_is_parsed_from_the_ort_sys_declaration() {
        assert_eq!(
            parse_pinned_version("const ORT_VERSION: &str = \"1.29.0\";").as_deref(),
            Some("1.29.0")
        );
        assert_eq!(
            parse_pinned_version("    const ORT_VERSION: &str = \"9.9.9\";\n").as_deref(),
            Some("9.9.9")
        );
        assert_eq!(
            parse_pinned_version("const OTHER: &str = \"1.29.0\";"),
            None
        );
        assert_eq!(
            parse_pinned_version("const ORT_VERSION: &str = \"\";"),
            None
        );
        assert_eq!(parse_pinned_version(""), None);
    }

    /// The observed case, and the reason this exists: after a version bump the
    /// tree held six 1.28.0 extractions beside one 1.29.0. The pinned one is
    /// eligible and the stale ones are not, so this *resolves* — it does not
    /// merely refuse.
    #[test]
    fn only_the_pinned_version_is_eligible() {
        let candidates = dirs(&["/b/a/lib", "/b/b/lib", "/b/c/lib", "/b/d/lib"]);
        let found = versions(&[
            Some(NOT_THE_PIN),
            Some(NOT_THE_PIN),
            Some("1.29.0"),
            Some(NOT_THE_PIN),
        ]);
        assert_eq!(
            resolve_candidates(&candidates, &found, Some("1.29.0")),
            Resolved::Chosen(PathBuf::from("/b/c/lib"))
        );
    }

    /// A single stale extraction is refused exactly like a mixed set. Accepting
    /// a lone candidate unchecked was a real hole in the first version of this
    /// change: it is reachable whenever the pin moves *backwards*, because the
    /// newer leftover library answers the older API version quite happily and
    /// nothing looks wrong.
    #[test]
    fn a_lone_stale_extraction_is_refused_not_accepted_for_being_alone() {
        let candidates = dirs(&["/b/only/lib"]);
        let found = versions(&[Some(NOT_THE_PIN)]);
        assert_eq!(
            resolve_candidates(&candidates, &found, Some("1.29.0")),
            Resolved::Ambiguous
        );
        assert_eq!(
            resolve_candidates(&candidates, &versions(&[Some("1.29.0")]), Some("1.29.0")),
            Resolved::Chosen(PathBuf::from("/b/only/lib")),
            "the same lone candidate at the pin must be accepted"
        );
    }

    /// Several build hashes are normal — profiles and feature sets each get
    /// their own — and extractions of the same release are interchangeable, so
    /// the choice among them only has to be stable.
    #[test]
    fn several_pinned_extractions_resolve_to_a_stable_one() {
        let candidates = dirs(&["/b/a/lib", "/b/b/lib", "/b/c/lib"]);
        let found = versions(&[Some("1.29.0"), Some("1.29.0"), Some("1.29.0")]);
        assert_eq!(
            resolve_candidates(&candidates, &found, Some("1.29.0")),
            Resolved::Chosen(PathBuf::from("/b/a/lib"))
        );
    }

    /// An unreadable marker is not the pin. Treating "unknown" as a match would
    /// reinstate the guess.
    #[test]
    fn a_candidate_with_no_marker_is_never_eligible() {
        let candidates = dirs(&["/b/a/lib"]);
        assert_eq!(
            resolve_candidates(&candidates, &versions(&[None]), Some("1.29.0")),
            Resolved::Ambiguous
        );
    }

    /// Absence must stay distinct from refusal. A first draft collapsed them
    /// and turned every tree with no ORT into a panic.
    #[test]
    fn no_candidates_is_absence_under_either_rule() {
        assert_eq!(
            resolve_candidates(&[], &[], Some("1.29.0")),
            Resolved::Absent
        );
        assert_eq!(resolve_candidates(&[], &[], None), Resolved::Absent);
    }

    /// Without a pin, staleness is undetectable and only self-contradiction
    /// remains. Weaker on purpose, and `single_candidate` says so on stderr.
    #[test]
    fn without_a_pin_the_rule_falls_back_to_agreement() {
        let two = dirs(&["/b/a/lib", "/b/b/lib"]);
        assert_eq!(
            resolve_candidates(&two, &versions(&[Some("1.28.0"), Some("1.29.0")]), None),
            Resolved::Ambiguous
        );
        assert_eq!(
            resolve_candidates(&two, &versions(&[Some("1.28.0"), Some("1.28.0")]), None),
            Resolved::Chosen(PathBuf::from("/b/a/lib")),
            "agreeing candidates are all this rule can accept"
        );
        assert_eq!(
            resolve_candidates(&two, &versions(&[Some("1.28.0"), None]), None),
            Resolved::Ambiguous,
            "unknown is not agreement"
        );
        assert_eq!(
            resolve_candidates(&dirs(&["/b/a/lib"]), &versions(&[None]), None),
            Resolved::Chosen(PathBuf::from("/b/a/lib")),
            "a lone candidate is all there is to choose; the warning covers it"
        );
    }

    /// Mutation control. The pre-fix rule was "first `read_dir` entry whose
    /// library exists". This shows it does not merely return *something* — on
    /// the observed layout it returns the **stale** directory, while the fix
    /// returns the pinned one. Revert the fix and these two agree.
    #[test]
    fn first_entry_wins_would_have_chosen_the_stale_directory() {
        let scratch = Scratch::new("mutation");
        let pin = pin_or_skip();
        let build = fake_build_dir(
            scratch.path(),
            &[("aaa", Some(NOT_THE_PIN)), ("bbb", Some(&pin))],
        );

        let candidates = scan_build_dir(&build);
        assert_eq!(candidates.len(), 2, "both candidates must be found");
        let first_entry_wins = candidates[0].clone();
        assert!(
            first_entry_wins.ends_with("onnx-genai-ort-sys-aaa/out/ort-prebuilt/lib"),
            "the pre-fix rule would have taken {first_entry_wins:?}"
        );

        let chosen = single_candidate(&build).expect("the pinned extraction is eligible");
        assert!(
            chosen.ends_with("onnx-genai-ort-sys-bbb/out/ort-prebuilt/lib"),
            "the fix must pick the pinned extraction, got {chosen:?}"
        );
        assert_ne!(
            chosen, first_entry_wins,
            "if these coincide the control proves nothing"
        );
    }

    /// End to end on a real directory tree, with the workspace's real pin.
    #[test]
    fn a_build_dir_holding_only_stale_extractions_is_refused() {
        let scratch = Scratch::new("stale");
        let build = fake_build_dir(
            scratch.path(),
            &[("aaa", Some(NOT_THE_PIN)), ("bbb", Some(NOT_THE_PIN))],
        );
        let err = std::panic::catch_unwind(|| single_candidate(&build))
            .expect_err("a tree with no pinned ORT must refuse");
        let msg = err
            .downcast_ref::<String>()
            .expect("panic payload is the diagnostic");
        assert!(msg.contains("no usable ONNX Runtime"), "{msg}");
        assert!(msg.contains(NOT_THE_PIN), "{msg}");
        assert!(msg.contains(&pin_or_skip()), "{msg}");
    }

    #[test]
    fn an_empty_build_dir_is_absence_not_refusal() {
        let scratch = Scratch::new("empty");
        let build = scratch.path().join("build");
        std::fs::create_dir_all(&build).expect("create empty build dir");
        assert!(single_candidate(&build).is_none());
        assert!(single_candidate(&scratch.path().join("nonexistent")).is_none());
    }

    /// The ordering `scan_build_dir` imposes is what makes any of this
    /// reproducible; `read_dir` alone is filesystem order.
    #[test]
    fn candidates_come_back_in_a_stable_order() {
        let scratch = Scratch::new("order");
        let pin = pin_or_skip();
        let build = fake_build_dir(
            scratch.path(),
            &[
                ("ccc", Some(&pin)),
                ("aaa", Some(&pin)),
                ("bbb", Some(&pin)),
            ],
        );
        let names: Vec<String> = scan_build_dir(&build)
            .iter()
            .map(|p| {
                p.parent()
                    .and_then(Path::parent)
                    .and_then(Path::parent)
                    .and_then(|d| d.file_name())
                    .expect("candidate has a build-hash ancestor")
                    .to_string_lossy()
                    .into_owned()
            })
            .collect();
        assert_eq!(
            names,
            vec![
                "onnx-genai-ort-sys-aaa",
                "onnx-genai-ort-sys-bbb",
                "onnx-genai-ort-sys-ccc"
            ],
            "scan_build_dir must sort its candidates"
        );
    }

    /// A directory holding no library is not a candidate, however plausibly it
    /// is named — an interrupted or failed extraction leaves exactly that.
    #[test]
    fn a_directory_without_a_library_is_not_a_candidate() {
        let scratch = Scratch::new("nolib");
        let build = scratch.path().join("build");
        std::fs::create_dir_all(build.join("onnx-genai-ort-sys-aaa/out/ort-prebuilt/lib"))
            .expect("create libraryless candidate");
        assert!(scan_build_dir(&build).is_empty());
    }

    /// Both message forms must carry every version found and a way out; the
    /// pinned form must also say what it was looking for, or the reader cannot
    /// tell which directory they need.
    #[test]
    fn the_refusal_names_every_version_and_the_way_out() {
        let candidates = dirs(&["/x/aaa/lib", "/x/bbb/lib"]);
        let found = versions(&[Some("1.28.0"), None]);

        let pinned = ambiguity_message(Path::new("/x"), &candidates, &found, Some("1.29.0"));
        assert!(pinned.contains("no usable ONNX Runtime"), "{pinned}");
        assert!(pinned.contains("1.29.0"), "{pinned}");
        assert!(pinned.contains("1.28.0"), "{pinned}");
        assert!(pinned.contains("<no VERSION_NUMBER>"), "{pinned}");
        assert!(pinned.contains("/x/aaa/lib"), "{pinned}");
        assert!(pinned.contains("/x/bbb/lib"), "{pinned}");
        assert!(pinned.contains("NXRT_ORT_LIB_DIR"), "{pinned}");

        let unpinned = ambiguity_message(Path::new("/x"), &candidates, &found, None);
        assert!(
            unpinned.contains("ambiguous ONNX Runtime discovery"),
            "{unpinned}"
        );
        assert!(unpinned.contains("1.28.0"), "{unpinned}");
        assert!(unpinned.contains("NXRT_ORT_LIB_DIR"), "{unpinned}");
    }

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
