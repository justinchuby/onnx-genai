//! MLAS is an *opt-in* backend of the shipped cdylib — not a default — and
//! when it is linked it is *ours*.
//!
//! Two claims are load-bearing for this repository's CPU story and neither is
//! visible in a compiled artifact:
//!
//! 1. **Off by default.** The library users install links *our native
//!    kernels*, not MLAS, unless the builder explicitly opts in with
//!    `--features mlas`. This is deliberate policy: a user who installs the
//!    wheel must get our kernels so the native gap stays visible and the
//!    absorption work in `docs/performance/ABSORBING_MLAS.md` keeps a forcing
//!    function. MLAS is a reference implementation and a graduation gate, not
//!    the shipped backend. A `default` list that silently *gained* `mlas`
//!    would ship the vendored C++ against that policy and nothing would fail —
//!    the MLAS build is numerically correct, just no longer ours — which is
//!    why it needs an assertion rather than a comment.
//! 2. **Internal, not delegated.** When MLAS *is* linked (under `--features
//!    mlas`) it is linked into our plugin as a private backend of our own
//!    execution provider. It is not ORT's CPU EP, it does not share ORT's copy
//!    of MLAS, and it holds no state across the DSO boundary. That is only true
//!    while every MLAS symbol stays local to this cdylib.
//!
//! `plugin_ort_e2e.rs` proves the complementary runtime claim (no node is left
//! to `CPUExecutionProvider`, with `session.disable_cpu_ep_fallback=1`); this
//! file proves the build-graph and link-level claims that make it meaningful.

mod cdylib_resolve;

use std::collections::{HashMap, HashSet};
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

/// The `default` feature list of a manifest, treating a *missing* `default`
/// key as an empty list.
///
/// Dropping `mlas` from the default is expressed by removing the `default`
/// array entirely (a cdylib with no `default` key has no default features),
/// which is exactly the policy this file pins: nothing is on unless a feature
/// names it. A missing key is therefore a pass, not a parse error.
fn default_features(manifest: &str) -> Vec<String> {
    if manifest.contains("\ndefault = [") {
        array_members(manifest, "default")
    } else {
        Vec::new()
    }
}

/// The shipped cdylib does *not* link MLAS unless the builder opts in.
///
/// This is the policy the 2026-08-17 correction restored: MLAS is an internal
/// reference backend and a graduation gate, not the thing we ship. The default
/// wheel links our native kernels, so a user who installs it runs our code and
/// every native gap stays load-bearing — the forcing function for absorption.
/// Re-gaining `mlas` in this list is invisible at runtime (an MLAS build is
/// numerically correct, just no longer ours), which is why it needs an
/// assertion rather than a comment. Feature unification is the likely way it
/// would drift back on in a merge; this test stops that silently.
#[test]
fn mlas_is_not_a_default_feature_of_the_shipped_plugin() {
    let defaults = default_features(&manifest("."));
    assert!(
        !defaults.iter().any(|f| f == "mlas"),
        "onnx-runtime-ep-cpu-plugin's default features are {defaults:?}; `mlas` must not be \
         among them — the shipped wheel is our native kernels, and MLAS is opt-in via \
         `--features mlas`. Shipping it by default bundles the vendored C++ against policy and \
         hides the native gap absorption exists to close"
    );
}

/// Neither does the EP crate itself, for anyone embedding it directly.
///
/// The plugin is one consumer. `onnx-runtime-ep-cpu` is published, and a
/// dependent that writes `onnx-runtime-ep-cpu = "…"` must get the same native
/// backend the wheel ships, or "the CPU EP" means two different things — and
/// the direct dependent would silently pull in the vendored MLAS C++ it never
/// asked for.
#[test]
fn the_cpu_ep_does_not_ship_mlas_in_its_own_default_features() {
    let defaults = default_features(&manifest("../onnx-runtime-ep-cpu"));
    assert!(
        !defaults.iter().any(|f| f == "mlas"),
        "onnx-runtime-ep-cpu's default features are {defaults:?}; `mlas` must not be among \
         them — a direct dependent would otherwise get a different (MLAS) backend from the \
         native one the plugin ships, and would link the vendored C++ unbidden"
    );
}

/// Every edge to the CPU EP pins `default-features = false`.
///
/// With MLAS off by default this is no longer what makes an opt-out work —
/// there is nothing to opt out of. Its purpose now is the reverse: it stops
/// feature unification from silently re-enabling MLAS. If one edge to
/// `onnx-runtime-ep-cpu` left the EP's defaults on and the EP crate ever
/// regained `mlas` in its `default` list (or gained it transitively), Cargo
/// would unify that back into this cdylib and the shipped wheel would link the
/// vendored C++ with no flag naming it. Pinning every edge keeps the feature
/// set of this crate explicit, so MLAS can only ever arrive here by an
/// on-purpose `--features mlas`. Dev-dependencies count: they unify with
/// normal dependencies under `cargo test`, which is where the numeric tests
/// run.
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
///
/// This is a duplicate-runtime-state check, not a defaults check: it is only
/// meaningful in the graph where `mlas-sys` is actually linked. With MLAS now
/// opt-in, that graph is the one resolved with `--features mlas` — the default
/// graph carries no `mlas-sys` node at all (which the two policy tests above
/// pin). So this resolves the plugin *with* `mlas` enabled and asserts that
/// even then there is exactly one copy.
#[test]
fn exactly_one_mlas_sys_in_the_dependency_graph() {
    let plugin_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let output =
        std::process::Command::new(std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_string()))
            .args([
                "metadata",
                "--format-version",
                "1",
                "--offline",
                "--features",
                "mlas",
            ])
            .current_dir(&plugin_dir)
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

    // Walk the *resolved* graph outward from this plugin rather than counting
    // workspace members. `--no-deps` would list only the 54 crates in this
    // repository, where `mlas-sys` is trivially unique no matter what the
    // resolver did -- the duplicate this test exists to catch arrives from
    // outside the workspace and would never appear there.
    let id_of = |name: &str| -> Option<String> {
        metadata["packages"]
            .as_array()?
            .iter()
            .find(|package| package["name"] == name)
            .and_then(|package| package["id"].as_str())
            .map(str::to_string)
    };
    let version_of: HashMap<String, String> = metadata["packages"]
        .as_array()
        .expect("cargo metadata lists packages")
        .iter()
        .filter_map(|package| {
            Some((
                package["id"].as_str()?.to_string(),
                format!(
                    "{}@{}",
                    package["name"].as_str()?,
                    package["version"].as_str()?
                ),
            ))
        })
        .collect();
    let edges: HashMap<String, Vec<String>> = metadata["resolve"]["nodes"]
        .as_array()
        .expect("a full `cargo metadata` carries a resolve graph")
        .iter()
        .filter_map(|node| {
            Some((
                node["id"].as_str()?.to_string(),
                node["dependencies"]
                    .as_array()?
                    .iter()
                    .filter_map(|dep| Some(dep.as_str()?.to_string()))
                    .collect(),
            ))
        })
        .collect();

    let root = id_of("onnx-runtime-ep-cpu-plugin").expect("this crate is in the metadata");
    let mut seen: HashSet<String> = HashSet::new();
    let mut queue = vec![root];
    let mut copies: Vec<String> = Vec::new();
    while let Some(id) = queue.pop() {
        if !seen.insert(id.clone()) {
            continue;
        }
        if version_of
            .get(&id)
            .is_some_and(|nv| nv.starts_with("mlas-sys@"))
        {
            copies.push(version_of[&id].clone());
        }
        queue.extend(edges.get(&id).into_iter().flatten().cloned());
    }

    // The walk itself has to be load-bearing: a typo in the root name or a
    // metadata format change would otherwise leave `copies` empty and pass.
    assert!(
        seen.len() > 1,
        "reachability walk found no dependencies of the plugin at all -- the graph was not          traversed, so the count below would be meaningless"
    );
    assert_eq!(
        copies.len(),
        1,
        "expected exactly one mlas-sys reachable from the shipped plugin, found {}: {copies:?}",
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
    if !cfg!(feature = "mlas") {
        eprintln!(
            "⏭ mlas_is_statically_private_to_this_cdylib: skipped — this test binary was built \
             without the `mlas` feature, so the cdylib it would probe links no MLAS at all. A \
             probe that reported '0 exported' on a binary that never linked MLAS would pass \
             vacuously; run `cargo test -p onnx-runtime-ep-cpu-plugin --features mlas` to \
             exercise it."
        );
        return;
    }
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

    assert!(
        !linked.is_empty(),
        "this cdylib was built with the `mlas` feature but its symbol table contains no \
         MLAS symbols at {} — the vendored kernels are not actually linked in",
        path.display()
    );
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
