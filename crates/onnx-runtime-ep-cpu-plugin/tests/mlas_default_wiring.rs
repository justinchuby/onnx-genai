//! MLAS is a default feature of the shipped cdylib, and it is *ours*.
//!
//! Two claims are load-bearing for this repository's CPU story and neither is
//! visible in a compiled artifact:
//!
//! 1. **On by default.** The library users install links MLAS unless they ask
//!    otherwise. A `default` list that quietly loses `mlas` costs an order of
//!    magnitude on quantized matmul and nothing fails — the pure-Rust build is
//!    numerically correct, just slow, so every other test stays green.
//! 2. **Internal, not delegated.** MLAS is linked into our plugin as a private
//!    backend of our own execution provider. It is not ORT's CPU EP, it does
//!    not share ORT's copy of MLAS, and it holds no state across the DSO
//!    boundary. That is only true while every MLAS symbol stays local to this
//!    cdylib.
//!
//! `plugin_ort_e2e.rs` proves the complementary runtime claim (no node is left
//! to `CPUExecutionProvider`, with `session.disable_cpu_ep_fallback=1`); this
//! file proves the build-graph and link-level claims that make it meaningful.

mod cdylib_resolve;

use std::path::{Path, PathBuf};

fn manifest(crate_dir: &str) -> String {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join(crate_dir)
        .join("Cargo.toml");
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

/// The value of a `key = [...]` array in a manifest, as raw text.
fn array_value(manifest: &str, key: &str) -> String {
    let start = manifest
        .find(&format!("\n{key} = ["))
        .unwrap_or_else(|| panic!("manifest declares `{key}`"));
    let open = manifest[start..].find('[').expect("array opens") + start;
    let close = manifest[open..].find(']').expect("array closes") + open;
    manifest[open + 1..close].to_string()
}

fn array_members(manifest: &str, key: &str) -> Vec<String> {
    array_value(manifest, key)
        .split(',')
        .map(|item| item.trim().trim_matches('"').to_string())
        .filter(|item| !item.is_empty())
        .collect()
}

/// The shipped cdylib links MLAS unless the builder opts out.
///
/// This is the whole point of the 2026-08-17 direction: our CPU EP owns its
/// nodes and calls MLAS *inside* that ownership, so the fast build has to be
/// the one you get by default. Losing `mlas` from this list is invisible at
/// runtime — correctness is unchanged, only speed — which is why it needs an
/// assertion rather than a comment.
#[test]
fn mlas_is_a_default_feature_of_the_shipped_plugin() {
    let defaults = array_members(&manifest("."), "default");
    assert!(
        defaults.iter().any(|f| f == "mlas"),
        "onnx-runtime-ep-cpu-plugin's default features are {defaults:?}; without `mlas` the \
         wheel ships a cdylib that is an order of magnitude slower on quantized matmul and \
         nothing else fails"
    );
}

/// So does the EP crate itself, for anyone embedding it directly.
///
/// The plugin is one consumer. `onnx-runtime-ep-cpu` is published, and a
/// dependent that writes `onnx-runtime-ep-cpu = "…"` must get the same
/// backend the wheel ships, or "the CPU EP" means two different things.
#[test]
fn the_cpu_ep_ships_mlas_in_its_own_default_features() {
    let defaults = array_members(&manifest("../onnx-runtime-ep-cpu"), "default");
    assert!(
        defaults.iter().any(|f| f == "mlas"),
        "onnx-runtime-ep-cpu's default features are {defaults:?}; a direct dependent would get \
         a different backend from the one the plugin ships"
    );
}

/// The opt-out actually opts out.
///
/// `--no-default-features` on this crate only removes MLAS because every edge
/// to `onnx-runtime-ep-cpu` pins `default-features = false`. Leave one edge
/// unpinned and Cargo's feature unification turns the EP's own `default` list
/// back on — including `mlas` — so the flag becomes a no-op that still
/// compiles, still passes, and ships an MLAS cdylib under a pure-Rust label.
/// Dev-dependencies count: they unify with normal dependencies under
/// `cargo test`, which is where the numeric opt-out tests run.
#[test]
fn every_edge_to_the_cpu_ep_pins_default_features_false() {
    let manifest = manifest(".");
    let unpinned: Vec<&str> = manifest
        .lines()
        .filter(|line| line.trim_start().starts_with("onnx-runtime-ep-cpu = "))
        .filter(|line| !line.contains("default-features = false"))
        .collect();
    assert!(
        unpinned.is_empty(),
        "these onnx-runtime-ep-cpu dependency edges do not pin `default-features = false`, so \
         `--no-default-features` on this crate would not remove MLAS: {unpinned:?}"
    );
}

/// One MLAS, linked once.
///
/// Two `mlas-sys` versions in the graph would mean two copies of the vendored
/// C++ in one cdylib: duplicate thread-pool configuration, duplicate one-time
/// CPU dispatch initialisation, and kernels chosen by whichever copy a given
/// call site was linked against. Cargo permits it whenever two semver-major
/// ranges coexist, and nothing else would report it.
#[test]
fn exactly_one_mlas_sys_in_the_dependency_graph() {
    let workspace_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("workspace root exists above this crate");
    let output =
        std::process::Command::new(std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_string()))
            .args(["metadata", "--format-version", "1", "--no-deps"])
            .current_dir(&workspace_root)
            .output();

    let output = match output {
        Ok(o) if o.status.success() => o,
        Ok(o) => {
            eprintln!(
                "⏭ exactly_one_mlas_sys_in_the_dependency_graph: skipped — cargo metadata \
                 failed: {}",
                String::from_utf8_lossy(&o.stderr)
            );
            return;
        }
        Err(e) => {
            eprintln!(
                "⏭ exactly_one_mlas_sys_in_the_dependency_graph: skipped — cargo not \
                 runnable: {e}"
            );
            return;
        }
    };

    let metadata: serde_json::Value =
        serde_json::from_slice(&output.stdout).expect("cargo metadata emits JSON");
    let copies: Vec<String> = metadata["packages"]
        .as_array()
        .expect("cargo metadata lists packages")
        .iter()
        .filter(|package| package["name"] == "mlas-sys")
        .map(|package| package["version"].as_str().unwrap_or("?").to_string())
        .collect();

    assert_eq!(
        copies.len(),
        1,
        "expected exactly one mlas-sys in the workspace, found {}: {copies:?}",
        copies.len()
    );
}

/// MLAS is linked into this cdylib and invisible from outside it.
///
/// The distinction this file exists for: our EP *uses* MLAS, it does not
/// *delegate* to ORT's. If any MLAS symbol were dynamically exported, the
/// loader could bind our calls to the copy inside `libonnxruntime`, or bind
/// ORT's calls to ours — one MLAS thread pool and one dispatch table shared
/// across two libraries that configure them differently. If any were
/// *undefined*, we would not have our own copy at all; we would be calling
/// whatever the process happened to provide, which is the delegation this
/// architecture rules out.
///
/// `l1_no_symbol_leakage` does not cover this: it exempts every `_Z`-prefixed
/// C++ symbol, which is exactly how MLAS is mangled.
#[test]
fn mlas_is_statically_private_to_this_cdylib() {
    if !cfg!(target_os = "linux") {
        eprintln!(
            "⏭ mlas_is_statically_private_to_this_cdylib: skipped (ELF-only check, this is {})",
            std::env::consts::OS
        );
        return;
    }
    let Some(path) = cdylib_resolve::find_cpu_plugin_cdylib_optional() else {
        eprintln!("⏭ mlas_is_statically_private_to_this_cdylib: skipped — cdylib not found");
        return;
    };

    let Some(all) = nm(&path, &["--format=posix"]) else {
        return;
    };
    let Some(dynamic) = nm(&path, &["--dynamic", "--format=posix"]) else {
        return;
    };

    let mlas_symbols = |table: &str| -> Vec<String> {
        table
            .lines()
            .filter(|line| line.to_ascii_lowercase().contains("mlas"))
            .map(|line| line.split_whitespace().next().unwrap_or(line).to_string())
            .collect()
    };

    let linked = mlas_symbols(&all);
    let exposed = mlas_symbols(&dynamic);

    if cfg!(feature = "mlas") {
        assert!(
            !linked.is_empty(),
            "this cdylib was built with the `mlas` feature but its symbol table contains no \
             MLAS symbols at {} — the vendored kernels are not actually linked in",
            path.display()
        );
    }
    assert!(
        exposed.is_empty(),
        "{} of this cdylib's {} MLAS symbols are in the *dynamic* table: {:?}. MLAS is an \
         internal backend of this EP; exporting or importing it lets the loader share one \
         copy's thread pool and dispatch state with ORT's own MLAS.",
        exposed.len(),
        linked.len(),
        &exposed[..exposed.len().min(10)]
    );

    eprintln!(
        "✓ mlas_is_statically_private_to_this_cdylib: {} MLAS symbols linked, 0 dynamic",
        linked.len()
    );
}

/// `nm` output, or `None` with a skip message when `nm` is unavailable.
fn nm(path: &Path, args: &[&str]) -> Option<String> {
    match std::process::Command::new("nm")
        .args(args)
        .arg(path)
        .output()
    {
        Ok(o) if o.status.success() => Some(String::from_utf8_lossy(&o.stdout).into_owned()),
        Ok(o) => {
            eprintln!(
                "⏭ mlas_is_statically_private_to_this_cdylib: skipped — nm {args:?} failed: {}",
                String::from_utf8_lossy(&o.stderr)
            );
            None
        }
        Err(e) => {
            eprintln!("⏭ mlas_is_statically_private_to_this_cdylib: skipped — nm not found: {e}");
            None
        }
    }
}
