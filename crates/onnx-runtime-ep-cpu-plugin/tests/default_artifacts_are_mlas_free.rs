//! The shipped artifacts contain no MLAS. Falsified, not asserted in prose.
//!
//! The repository's direction is that the production CPU EP is **native**. MLAS
//! is a research reference: something we measure our own kernels against and
//! absorb capability from (`docs/performance/CPU_MLAS_MIGRATION.md`), never
//! something a default build links, activates, or routes to. Rejecting MLAS by
//! default is *not* a re-opening of delegation to ORT's built-in
//! `CPUExecutionProvider` either — neither route is permitted, and
//! `plugin_ort_e2e.rs` proves the second with
//! `session.disable_cpu_ep_fallback=1`.
//!
//! That policy is invisible in a compiled artifact and silent when broken: a
//! cdylib that quietly gained MLAS is numerically correct, just differently
//! implemented, so every other test in this repository stays green. It is
//! therefore stated here as executable falsifiers at the three levels where it
//! can be broken:
//!
//! | level | what would break it | test |
//! |---|---|---|
//! | Cargo features | `mlas` reachable from the EP's or plugin's `default` | `no_default_feature_list_activates_mlas` |
//! | feature graph | *any* workspace crate defaulting `mlas` | `no_workspace_crate_makes_mlas_a_default` |
//! | resolved graph | `mlas-sys` compiled into a default build | `a_default_build_resolves_no_mlas_sys` |
//! | linked binary | MLAS object code in the shipped cdylib | `the_default_cdylib_contains_no_mlas_symbols` |
//! | package | the wheel asking cargo for `mlas` | `the_wheel_never_proactively_enables_mlas` |
//! | documentation | the manifest telling readers MLAS is a default | `the_workspace_manifest_does_not_document_mlas_as_a_default` |
//!
//! Each probe is checked to be load-bearing before its claim is asserted — a
//! zero count from a probe that read nothing is the failure mode these tests
//! exist to avoid, and is how the situation in #1091 arose (measuring a
//! configuration nobody shipped).

mod cdylib_resolve;

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

fn repo_file(relative: &str) -> String {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(relative);
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

/// No `default` feature list, transitively, turns MLAS on.
///
/// Three ways this could regress, all of them one word: `default` on the
/// plugin gaining `"mlas"`; `default` on the EP gaining it; or the `full`
/// umbrella — which `default` does contain — gaining it, which would activate
/// MLAS for every consumer without the word `default` appearing anywhere near
/// the change.
#[test]
fn no_default_feature_list_activates_mlas() {
    let plugin = repo_file("crates/onnx-runtime-ep-cpu-plugin/Cargo.toml");
    let ep = repo_file("crates/onnx-runtime-ep-cpu/Cargo.toml");

    // The plugin declares `mlas` as an opt-in and has no `default` at all;
    // either shape is fine, an MLAS-bearing one is not.
    assert!(
        plugin.contains("\nmlas = [\"onnx-runtime-ep-cpu/mlas\"]"),
        "the plugin must still be able to opt in to the MLAS reference for research builds"
    );
    if plugin.contains("\ndefault = [") {
        let defaults = array_members(&plugin, "default");
        assert!(
            !defaults.iter().any(|f| activates_mlas(f)),
            "the shipped plugin cdylib must not enable MLAS by default, found {defaults:?}"
        );
    }

    let ep_defaults = array_members(&ep, "default");
    assert!(
        !ep_defaults.is_empty(),
        "probe read no default features from the EP manifest, so the assertion below \
         would pass vacuously"
    );
    assert!(
        !ep_defaults.iter().any(|f| activates_mlas(f)),
        "onnx-runtime-ep-cpu must not enable MLAS by default, found {ep_defaults:?}"
    );

    let full = array_members(&ep, "full");
    assert!(
        !full.is_empty(),
        "probe read no members of the `full` umbrella, so the assertion below would \
         pass vacuously"
    );
    assert!(
        !full.iter().any(|f| activates_mlas(f)),
        "the `full` umbrella is in `default`, so it must not imply MLAS either, found {full:?}"
    );
}

/// *Every* workspace crate keeps MLAS out of its `default` closure — not just
/// the two this file used to name.
///
/// [`no_default_feature_list_activates_mlas`] hardcodes the EP and its plugin.
/// Seven crates in this workspace declare an `mlas` feature, and the five it
/// does not name (`onnx-genai-bench`, `-cli`, `-engine`, `-server`,
/// `onnx-runtime-session`) chain into the EP's. Any one of them could put
/// `"mlas"` in its own `default` and light up the vendored C++ for everything
/// downstream of it, with both existing manifest checks staying green because
/// neither reads that file.
///
/// So this enumerates `crates/*/Cargo.toml` instead of listing them, walks each
/// crate's `default` to a fixed point, and fails on the first one that can
/// reach MLAS. A crate added tomorrow is covered without anyone remembering to
/// come back here.
#[test]
fn no_workspace_crate_makes_mlas_a_default() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let mut declare_mlas = Vec::new();
    let mut have_defaults = Vec::new();

    let mut crates: Vec<PathBuf> = std::fs::read_dir(root.join("crates"))
        .expect("workspace has a crates/ directory")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.join("Cargo.toml").is_file())
        .collect();
    crates.sort();
    assert!(
        crates.len() > 10,
        "probe found only {} crates, so the sweep below would be near-vacuous",
        crates.len()
    );

    for dir in &crates {
        let name = dir.file_name().unwrap().to_string_lossy().to_string();
        let manifest = std::fs::read_to_string(dir.join("Cargo.toml")).expect("readable manifest");
        let features = feature_table(&manifest);

        if features.keys().any(|f| activates_mlas(f)) {
            declare_mlas.push(name.clone());
        }
        let Some(defaults) = features.get("default") else {
            continue;
        };
        if defaults.is_empty() {
            continue;
        }
        have_defaults.push(name.clone());

        // Walk `default` to a fixed point. A crate reaches MLAS if any name in
        // the closure mentions it, whether as its own `mlas`, a `dep:mlas-sys`,
        // or a `some-crate/mlas` passed through to a dependency.
        let mut seen: HashSet<String> = HashSet::new();
        let mut queue: Vec<String> = defaults.clone();
        while let Some(feature) = queue.pop() {
            assert!(
                !activates_mlas(&feature),
                "crate `{name}` reaches MLAS from its `default` feature via \
                 `{feature}`. The production CPU EP is native; MLAS is an \
                 explicit research reference, so nothing may be able to enable \
                 it without asking. Closure walked so far: {seen:?}"
            );
            if !seen.insert(feature.clone()) {
                continue;
            }
            if let Some(members) = features.get(&feature) {
                queue.extend(members.iter().cloned());
            }
        }
    }

    // Both lists are the probe's own vacuity check: an empty sweep, a rename of
    // the feature, or a `crates/` layout change would otherwise pass silently.
    assert!(
        declare_mlas.len() >= 5,
        "probe found only {declare_mlas:?} declaring an MLAS feature; the sweep \
         is reading less of the workspace than it used to and would pass vacuously"
    );
    assert!(
        have_defaults.len() >= 5,
        "probe walked defaults for only {have_defaults:?}; the feature-table \
         parser has stopped reading manifests it used to read"
    );
}

/// Every `name = [...]` under `[features]`, as a map.
///
/// Stops at the next `[section]` so a `[dependencies]` entry that happens to
/// look like a feature list cannot be mistaken for one.
fn feature_table(manifest: &str) -> HashMap<String, Vec<String>> {
    let mut table = HashMap::new();
    let Some(start) = manifest.find("\n[features]") else {
        return table;
    };
    let body = &manifest[start + "\n[features]".len()..];
    let end = body.find("\n[").unwrap_or(body.len());

    let mut rest = &body[..end];
    while let Some(open) = rest.find('[') {
        let name = rest[..open].trim_end().trim_end_matches('=').trim();
        let name = name.rsplit('\n').next().unwrap_or(name).trim();
        let Some(close) = rest[open..].find(']') else {
            break;
        };
        let members: Vec<String> = rest[open + 1..open + close]
            .split(',')
            // Drop trailing comments so `"a", # mlas is not here` cannot be
            // read as a member named after MLAS.
            .map(|item| {
                item.split('#')
                    .next()
                    .unwrap_or("")
                    .trim()
                    .trim_matches('"')
                    .to_string()
            })
            .filter(|item| !item.is_empty())
            .collect();
        if !name.is_empty() && !name.contains(' ') {
            table.insert(name.to_string(), members);
        }
        rest = &rest[open + close + 1..];
    }
    table
}

/// The workspace manifest's prose does not claim MLAS is a default.
///
/// This is a regression test for a real drift, not a style rule. Until this
/// change the workspace `Cargo.toml` told readers that "the CPU EP's default
/// features include vendored MLAS; opt out with `--no-default-features`" and
/// that crates "enable the CPU EP's default `mlas` feature". Both were false —
/// `mlas` has never been in any `default` list — and both survived because
/// every test in this file reads feature *tables* and none reads the comments
/// around them.
///
/// False prose here is expensive out of proportion to its size: it is the first
/// thing a contributor reads when deciding whether a build needs a C++
/// toolchain, and it is the justification someone would cite for "MLAS is
/// already on by default, so linking it here changes nothing" — which is how
/// #1091 happened.
///
/// Deliberately a deny-list of literal claims rather than a negation-aware
/// reading of English: the corrected prose necessarily also puts "default" and
/// "MLAS" in one sentence ("No member's default features compile the vendored
/// MLAS C++/asm"), so proximity alone cannot separate a claim from its denial.
/// The claims below are affirmative in any context.
#[test]
fn the_workspace_manifest_does_not_document_mlas_as_a_default() {
    let manifest = repo_file("Cargo.toml");

    // Comments only, whitespace-collapsed so a claim that wraps across two
    // `#` lines reads the same as one that does not — which is how the
    // "default\n# features include vendored MLAS" claim escaped notice.
    let prose: String = manifest
        .lines()
        .map(|l| l.trim())
        .filter(|l| l.starts_with('#'))
        .map(|l| l.trim_start_matches('#').trim())
        .collect::<Vec<_>>()
        .join(" ")
        .to_ascii_lowercase();
    let prose = prose.split_whitespace().collect::<Vec<_>>().join(" ");

    assert!(
        prose.contains("mlas"),
        "probe read no MLAS prose from the workspace manifest, so every claim \
         below would pass vacuously"
    );

    const FALSE_CLAIMS: &[&str] = &[
        "default `mlas` feature",
        "default mlas feature",
        "default features include vendored mlas",
        "default features include mlas",
        "mlas is enabled by default",
        "mlas is on by default",
        "enables mlas by default",
        "links mlas by default",
    ];
    for claim in FALSE_CLAIMS {
        assert!(
            !prose.contains(claim),
            "the workspace manifest claims \"{claim}\", which is false: `mlas` is \
             not reachable from any `default` list (see \
             `no_workspace_crate_makes_mlas_a_default`). Prose that says a \
             shipped build already links MLAS is how a change that actually \
             links it gets waved through."
        );
    }
}

/// Whether a feature-list entry turns the MLAS reference on.
///
/// Exact `== "mlas"` is too narrow: a `default` list reaches MLAS just as well
/// through `"onnx-runtime-ep-cpu/mlas"`, `"dep:mlas-sys"` or `"mlas-sys/..."`,
/// and none of those is the bare word. Nothing else in this workspace is named
/// after MLAS, so matching the substring costs no false positives and removes
/// the need to predict which spelling a regression will use.
fn activates_mlas(feature: &str) -> bool {
    feature.to_ascii_lowercase().contains("mlas")
}

/// A default build resolves no `mlas-sys` at all.
///
/// The feature check above reads manifests; this one reads what Cargo actually
/// decided. They can disagree: an unconditional (non-`optional`) dependency, a
/// `dep:` alias, or a third crate in the workspace enabling
/// `onnx-runtime-ep-cpu/mlas` would all leave every `default` list innocent and
/// still compile the vendored C++ into the build.
#[test]
fn a_default_build_resolves_no_mlas_sys() {
    let workspace_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("workspace root exists above this crate");
    let Some(host) = host_triple() else {
        return;
    };
    let output =
        std::process::Command::new(std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_string()))
            .args([
                "metadata",
                "--format-version",
                "1",
                "--offline",
                "--no-default-features",
                "--features",
                "",
                "--filter-platform",
                &host,
            ])
            .current_dir(&workspace_root)
            .output();

    let output = match output {
        Ok(o) if o.status.success() => o,
        Ok(o) => {
            let stderr = String::from_utf8_lossy(&o.stderr).into_owned();
            gate_probe_failed(&format!("cargo metadata failed: {stderr}"));
            return;
        }
        Err(e) => {
            gate_probe_failed(&format!("cargo is not runnable: {e}"));
            return;
        }
    };

    let metadata: serde_json::Value =
        serde_json::from_slice(&output.stdout).expect("cargo metadata emits JSON");

    let nodes = metadata["resolve"]["nodes"]
        .as_array()
        .expect("a full `cargo metadata` carries a resolve graph");

    // The EP's own resolved feature set is the direct statement: `mlas` absent
    // here means nothing in the workspace turned it on for a default build.
    let ep_features: Vec<String> = nodes
        .iter()
        .find(|node| {
            node["id"]
                .as_str()
                .is_some_and(|id| id.contains("onnx-runtime-ep-cpu#"))
        })
        .and_then(|node| node["features"].as_array())
        .map(|features| {
            features
                .iter()
                .filter_map(|f| Some(f.as_str()?.to_string()))
                .collect()
        })
        .expect("onnx-runtime-ep-cpu is in the resolve graph");
    assert!(
        !ep_features.is_empty(),
        "probe read no resolved features, so the assertion below would pass vacuously"
    );
    assert!(
        !ep_features.iter().any(|f| f == "mlas"),
        "a default workspace build resolved `mlas` on onnx-runtime-ep-cpu: {ep_features:?}"
    );

    // And no edge from the shipped plugin reaches mlas-sys.
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
    let edges: HashMap<String, Vec<String>> = nodes
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

    let root = metadata["packages"]
        .as_array()
        .and_then(|packages| {
            packages
                .iter()
                .find(|package| package["name"] == "onnx-runtime-ep-cpu-plugin")
        })
        .and_then(|package| package["id"].as_str())
        .expect("this crate is in the metadata")
        .to_string();

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

    // The walk has to be load-bearing: a renamed root or a metadata format
    // change would otherwise leave `copies` empty and pass for the wrong
    // reason.
    assert!(
        seen.len() > 1,
        "reachability walk found no dependencies of the plugin at all — the graph was \
         not traversed, so the count below would be meaningless"
    );
    assert!(
        copies.is_empty(),
        "a default build of the shipped plugin reaches mlas-sys: {copies:?}"
    );
}

/// The shipped cdylib carries no MLAS object code.
///
/// The end of the chain, and the only level a user can inspect. Everything
/// above is about intent; this is about the file. Note it must read the *whole*
/// symbol table, not just the dynamic one: MLAS is statically linked when it is
/// present at all, so a dynamic-table-only probe would report zero either way
/// and pass for a reason unrelated to its claim.
#[test]
fn the_default_cdylib_contains_no_mlas_symbols() {
    if cfg!(feature = "mlas") {
        eprintln!(
            "⏭ the_default_cdylib_contains_no_mlas_symbols: skipped — this is a \
             `--features mlas` research build, which is expected to contain MLAS"
        );
        return;
    }
    if !cfg!(target_os = "linux") {
        eprintln!(
            "⏭ the_default_cdylib_contains_no_mlas_symbols: skipped (ELF-only check, this is {})",
            std::env::consts::OS
        );
        return;
    }
    let Some(path) = cdylib_resolve::find_cpu_plugin_cdylib_optional() else {
        gate_probe_failed("the default cdylib was not found, so there was nothing to inspect");
        return;
    };

    let Some(all) = nm(&path, &["--format=posix"]) else {
        return;
    };
    let Some(dynamic) = nm(&path, &["--dynamic", "--format=posix"]) else {
        return;
    };

    // Load-bearing check on the probe itself: an unreadable or stripped binary
    // yields an empty table, from which "no MLAS symbols" follows trivially.
    let total = all.lines().filter(|line| !line.trim().is_empty()).count();
    assert!(
        total > 100,
        "nm read only {total} symbols from {} — the probe did not work, so a zero \
         MLAS count below would mean nothing",
        path.display()
    );

    let linked = mlas_symbols(&all);
    let exposed = mlas_symbols(&dynamic);

    assert!(
        linked.is_empty(),
        "{} of {total} symbols in the default cdylib at {} are MLAS: {:?}. The shipped \
         CPU EP is native; MLAS belongs only in an explicit `--features mlas` research \
         build.",
        linked.len(),
        path.display(),
        &linked[..linked.len().min(10)]
    );
    assert!(
        exposed.is_empty(),
        "the default cdylib exports or imports MLAS symbols: {:?}",
        &exposed[..exposed.len().min(10)]
    );

    eprintln!(
        "✓ the_default_cdylib_contains_no_mlas_symbols: 0 of {total} symbols are MLAS \
         in {}",
        path.display()
    );
}

/// The wheel never asks cargo for MLAS unless a human explicitly opts in.
///
/// `setup.py` used to enable MLAS on a per-target allow-list, which meant the
/// published artifact carried it on four targets without anything in the
/// release process saying so. The polarity is now opt-in: MLAS is only linked
/// when `NXRT_EP_CPU_RESEARCH_MLAS=1` is set, which release automation never
/// sets, and such a wheel is not publishable.
#[test]
fn the_wheel_never_proactively_enables_mlas() {
    let setup = repo_file("python/nxrt-ep-cpu/setup.py");

    assert!(
        setup.contains("MLAS_OPT_IN_ENV = \"NXRT_EP_CPU_RESEARCH_MLAS\""),
        "setup.py must gate MLAS behind an explicit research opt-in"
    );

    let body = setup
        .split("def _mlas_features(")
        .nth(1)
        .expect("setup.py defines _mlas_features")
        .split("\nCARGO_FEATURES")
        .next()
        .expect("_mlas_features is followed by CARGO_FEATURES");
    assert!(
        !body.is_empty(),
        "probe extracted an empty _mlas_features body, so the assertions below would \
         pass vacuously"
    );

    // The only path that returns the feature must be behind the opt-in, so the
    // opt-in check has to come before every `return [MLAS_FEATURE]`.
    let opt_in_at = body
        .find("os.environ.get(MLAS_OPT_IN_ENV) != \"1\"")
        .expect("_mlas_features must short-circuit on the opt-in being unset");
    let returns_feature: Vec<usize> = body
        .match_indices("return [MLAS_FEATURE]")
        .map(|(at, _)| at)
        .collect();
    assert_eq!(
        returns_feature.len(),
        1,
        "expected exactly one path that enables MLAS, found {}",
        returns_feature.len()
    );
    assert!(
        returns_feature[0] > opt_in_at,
        "setup.py can return the MLAS feature without passing the opt-in gate first"
    );

    // The old inverted escape hatch must be gone: an opt-*out* named variable
    // implies an on-by-default build, which is the policy this test enforces
    // against.
    assert!(
        !setup.contains("NXRT_EP_CPU_NO_MLAS"),
        "setup.py still carries the opt-out escape hatch from the MLAS-by-default \
         design; with MLAS off by default there is nothing to opt out of"
    );

    // Everything above constrains `_mlas_features`, which matters only while
    // `_mlas_features` is what the build reads. `CARGO_FEATURES = [MLAS_FEATURE]`
    // would ship an MLAS wheel past every assertion in this file — including
    // `_verify_features`, which derives what it expects *from* `CARGO_FEATURES`
    // and so agrees with whatever it is told.
    assert!(
        setup.contains("CARGO_FEATURES: list[str] = _mlas_features()"),
        "setup.py must take its cargo features from the gated `_mlas_features()`; \
         assigning CARGO_FEATURES directly bypasses the opt-in checked above"
    );
    let assignments = setup.match_indices("\nCARGO_FEATURES").count();
    assert_eq!(
        assignments, 1,
        "expected exactly one top-level CARGO_FEATURES binding, found {assignments}; \
         a later rebinding would silently win over the gated one"
    );
}

/// MLAS object-code symbols in an `nm --format=posix` table.
///
/// Naive `contains("mlas")` is wrong in both directions, and its false
/// positives are the interesting ones: this crate's *own* Rust items are
/// legitimately named after the route they call — `try_prefill_mlas_nt`,
/// `mlas_sqnbit_owns_fp32_compute`, `CpuBackend::Mlas` — and they appear in the
/// symbol table of a build that contains no MLAS whatsoever. A test that fails
/// on those is not measuring what it claims to.
///
/// So Rust-mangled symbols are excluded first (v0 `_R…`, legacy
/// `_ZN…17h<16 hex>E`), and only then is the remaining C/C++ namespace — which
/// is where the vendored kernels actually live, as `Mlas*` or `_Z…Mlas…` —
/// searched. `linking_the_reference_would_be_visible_to_this_probe` is the
/// positive control that keeps this from becoming a filter that matches
/// nothing.
fn mlas_symbols(table: &str) -> Vec<String> {
    table
        .lines()
        .filter_map(|line| line.split_whitespace().next())
        .filter(|name| !is_rust_mangled(name))
        .filter(|name| name.to_ascii_lowercase().contains("mlas"))
        .map(str::to_string)
        .collect()
}

/// Whether a symbol name came out of the Rust mangler, in either scheme.
fn is_rust_mangled(name: &str) -> bool {
    if name.starts_with("_R") {
        return true;
    }
    // Legacy scheme: the item path is suffixed `17h<16 hex digits>E`.
    let Some(hash_at) = name.rfind("17h") else {
        return false;
    };
    let tail = &name[hash_at + 3..];
    tail.len() == 17 && tail.ends_with('E') && tail[..16].bytes().all(|b| b.is_ascii_hexdigit())
}

/// The probe above is a filter, and a filter that matches nothing passes every
/// test that uses it. This is its positive control: the names this repository
/// is known to link when the reference *is* present must be classified as MLAS,
/// and the native build's own Rust items — the real false positives observed
/// here — must not.
#[test]
fn linking_the_reference_would_be_visible_to_this_probe() {
    // `nm --format=posix` emits `name type value size`, name first.
    let vendored = "\
MlasGemm T 3967e0 1f4\n\
MlasSgemmKernelZero T 396a10 88\n\
MlasComputeSoftmax T 396c20 210\n\
_ZN12MlasQNBitGemm7ComputeEv t 396e40 c4\n";
    assert_eq!(
        mlas_symbols(vendored).len(),
        4,
        "the probe must classify vendored MLAS object code as MLAS, or a zero \
         count from it means nothing"
    );

    // Observed verbatim in a native build of this cdylib.
    let ours = "\
_RNvMs5_NtNtCsjjCjGxbJBjY_19onnx_runtime_ep_cpu7kernels12matmul_nbitsNtB5_17MatMulNBitsKernel19try_prefill_mlas_nt t 1a2b3c 40\n\
_RNvMs5_NtNtCsjjCjGxbJBjY_19onnx_runtime_ep_cpu7kernels12matmul_nbitsNtB5_17MatMulNBitsKernel29mlas_sqnbit_owns_fp32_compute t 1a2b80 40\n\
_ZN19onnx_runtime_ep_cpu7backend10CpuBackend4Mlas17h0123456789abcdefE t 1a2bc0 20\n";
    assert_eq!(
        mlas_symbols(ours),
        Vec::<String>::new(),
        "this crate's own Rust items are named after the routes they call and \
         must not be mistaken for vendored MLAS object code"
    );
}

/// The triple this test process runs on.
///
/// `cargo metadata` without `--filter-platform` insists on resolving the
/// dependency graph of every target Cargo has ever heard of — Android, wasm —
/// which an `--offline` lane has no reason to have vendored. That made the gate
/// fail for a reason with nothing to do with MLAS. Filtering to the host keeps
/// the resolve honest (it is the platform whose artifact is being probed) and
/// resolvable from a cache that only ever built for it.
fn host_triple() -> Option<String> {
    let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".to_string());
    let output = match std::process::Command::new(rustc).arg("-vV").output() {
        Ok(o) if o.status.success() => o,
        Ok(o) => {
            return gate_probe_failed(&format!(
                "rustc -vV failed: {}",
                String::from_utf8_lossy(&o.stderr)
            ));
        }
        Err(e) => return gate_probe_failed(&format!("rustc is not runnable: {e}")),
    };
    let text = String::from_utf8_lossy(&output.stdout).into_owned();
    match text.lines().find_map(|line| line.strip_prefix("host: ")) {
        Some(triple) if !triple.trim().is_empty() => Some(triple.trim().to_string()),
        _ => gate_probe_failed("rustc -vV printed no host triple"),
    }
}

/// `nm` output, or `None` with a skip message when `nm` is unavailable.
fn nm(path: &Path, args: &[&str]) -> Option<String> {
    match std::process::Command::new("nm")
        .args(args)
        .arg(path)
        .output()
    {
        Ok(o) if o.status.success() => Some(String::from_utf8_lossy(&o.stdout).into_owned()),
        Ok(o) => gate_probe_failed(&format!(
            "nm {args:?} failed on {}: {}",
            path.display(),
            String::from_utf8_lossy(&o.stderr)
        )),
        Err(e) => gate_probe_failed(&format!("nm is not runnable: {e}")),
    }
}

/// Whether this configuration is one the merge gate is *required* to run in.
///
/// The gate is defined on the shipped configuration: a default-feature build on
/// Linux, which is where release artifacts are produced. Elsewhere — a
/// `--features mlas` research build, or a platform whose object format this
/// probe cannot read — skipping is honest.
fn gate_must_run() -> bool {
    !cfg!(feature = "mlas") && cfg!(target_os = "linux")
}

/// Report a probe that could not run.
///
/// A test that returns without asserting is a *pass*, so a probe that fails to
/// execute quietly converts the merge gate into no gate at all — the same
/// failure mode as measuring a configuration nobody ships. Where the gate is
/// required, an unusable probe is therefore a failure and not a skip.
fn gate_probe_failed(why: &str) -> Option<String> {
    // A lane that builds only a subset of the workspace caches only that
    // subset's packages, but the resolve probe reads every manifest in the
    // lockfile. That shows up here as an `--offline` download failure for some
    // crate with no connection to MLAS, which reads like a mystery unless the
    // remedy is named at the point of failure.
    let hint = if why.contains("--offline was specified") {
        "\nhint: this is a missing package in the local registry, not an MLAS \
         finding. A lane that builds part of the workspace must run \
         `cargo fetch --locked` before the probe reads the whole resolve."
    } else {
        ""
    };
    assert!(
        !gate_must_run(),
        "the MLAS-free gate could not run its probe on the configuration it \
         exists to check ({why}); passing here would prove nothing{hint}"
    );
    eprintln!("⏭ skipped — {why}");
    None
}
