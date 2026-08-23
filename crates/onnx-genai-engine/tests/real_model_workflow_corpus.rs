//! Real model packages, executed as workflow packages.
//!
//! `onnx-genai-metadata`'s unit tests pin the workflow/ABI round-trip against a
//! hand-written decoder. These run whatever real packages this machine has
//! through the engine's own loader — including the `genai_config.json` importer,
//! which converts a *foreign* producer's format into this project's one
//! representation. That is what covers the shapes a hand-written fixture never
//! has: shared-buffer decoders, static caches, hybrid recurrent state, MoE
//! graphs, and models with dozens of KV pairs.
//!
//! Every case skips with a reason when its package is absent, and
//! `corpus_inventory` prints exactly which artifacts were missing — a silent
//! skip would let this suite report success on a weightless machine.

use std::path::{Path, PathBuf};

use onnx_genai_engine::{Engine, EngineConfig};

/// Candidate real packages, labelled by the cache/ABI shape each one covers.
fn corpus() -> Vec<(&'static str, PathBuf)> {
    let foundry = home().join(".foundry/cache/models/Microsoft");
    let mut entries = vec![
        (
            "phi35-mini int4 (shared-buffer past/present)",
            foundry.join("Phi-3.5-mini-instruct-generic-cpu-2/v2"),
        ),
        (
            "qwen3-0.6b int4 (past/present)",
            foundry.join("qwen3-0.6b-generic-cpu-4/v4"),
        ),
        // A multi-component (vision+text+embedding) compatibility package with
        // no `inference_metadata.yaml`: not loadable as a bare decoder and not
        // resolvable as a workflow, so it is expected to skip with a reason.
        // Kept in the inventory so its absence from coverage is visible.
        (
            "qwen3.5-0.8b (multi-component compatibility package)",
            foundry.join("qwen3.5-0.8b-generic-cpu-2/v2"),
        ),
        (
            "qwen2.5-0.5b cuda (static cache)",
            foundry.join("qwen2.5-0.5b-instruct-cuda-gpu-4/v4-bs128"),
        ),
        (
            "qwen2.5-coder-7b cpu",
            foundry.join("qwen2.5-coder-7b-instruct-generic-cpu-4/v4"),
        ),
        (
            "phi4-mini cuda",
            foundry.join("Phi-4-mini-instruct-cuda-gpu-5/v5"),
        ),
        (
            "gpt-oss-20b cpu (MoE)",
            foundry.join("gpt-oss-20b-generic-cpu-1/v1"),
        ),
        (
            "qwen3.5-2b text",
            foundry.join("qwen3.5-2b-text-generic-cpu-1/v1"),
        ),
    ];
    if let Some(extra) = std::env::var_os("ONNX_GENAI_CANONICAL_CORPUS") {
        for dir in std::env::split_paths(&extra) {
            entries.push(("ONNX_GENAI_CANONICAL_CORPUS entry", dir));
        }
    }
    entries
}

fn home() -> PathBuf {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/root"))
}

/// Fail a corpus case that covered nothing while a corpus was present.
///
/// Two situations look identical from inside the loop and are not the same
/// thing:
///
/// * **Packages are installed and none was covered.** Every one of them failed
///   to load, or the case skipped them all. Returning `Ok` there reports success
///   without checking anything — the silent pass this suite exists to prevent.
/// * **This machine has no corpus.** A CI runner has no multi-gigabyte model
///   directories and never will. There is nothing to check and nothing wrong.
///
/// They are distinguished by asking whether any candidate directory exists,
/// which is a fact about the machine rather than an environment variable a run
/// can set to silence a real failure. The weightless case prints what it did
/// not cover, so a green run stays legible instead of merely quiet.
fn require_corpus_coverage(case: &str, covered: usize) {
    if covered > 0 {
        return;
    }
    let present = corpus()
        .into_iter()
        .filter(|(_, dir)| dir.is_dir())
        .map(|(label, _)| label)
        .collect::<Vec<_>>();
    assert!(
        present.is_empty(),
        "{case} covered none of the {} corpus package(s) installed on this machine ({present:?}). \
         Every one of them failed to load or was skipped, which is a failure rather than an \
         absence.",
        present.len()
    );
    eprintln!("CANONICAL_CORPUS {case}: this machine has no corpus package installed");
}

fn open(dir: &Path) -> Option<Engine> {
    if !dir.is_dir() {
        return None;
    }
    match Engine::from_dir(dir, EngineConfig::default()) {
        Ok(engine) => Some(engine),
        Err(error) => {
            eprintln!("skipping {}: {error:#}", dir.display());
            None
        }
    }
}

/// Report exactly which corpus artifacts this machine has.
///
/// Never fails: its job is to make coverage legible, so a run on a weightless
/// box is distinguishable from one that genuinely covered the corpus.
#[test]
fn corpus_inventory() {
    let (mut present, mut missing) = (Vec::new(), Vec::new());
    for (label, dir) in corpus() {
        if dir.is_dir() {
            present.push((label, dir));
        } else {
            missing.push((label, dir));
        }
    }
    eprintln!(
        "CANONICAL_CORPUS present={} missing={}",
        present.len(),
        missing.len()
    );
    for (label, dir) in &present {
        eprintln!("  present: {label} -> {}", dir.display());
    }
    for (label, dir) in &missing {
        eprintln!("  MISSING: {label} -> {}", dir.display());
    }
}

/// Every real package presents a workflow, deterministically.
///
/// Determinism matters because the `genai_config.json` importer *converts* a
/// foreign format: a conversion that varied run to run would mean the runtime
/// executed something different each load, which is exactly the drift a single
/// serialized representation exists to prevent.
#[test]
fn real_packages_present_a_workflow_deterministically() {
    let mut covered = 0usize;
    for (label, dir) in corpus() {
        let Some(engine) = open(&dir) else { continue };
        let first = engine
            .package_workflow_document()
            .unwrap_or_else(|error| panic!("{label}: no workflow: {error:#}"));
        for _ in 0..3 {
            assert_eq!(
                engine.package_workflow_document().expect("repeat"),
                first,
                "{label}: the workflow this package presents is not stable"
            );
        }
        assert!(
            engine.package_workflow().is_some(),
            "{label}: a loaded package must present a workflow"
        );
        covered += 1;
        eprintln!(
            "CORPUS_WORKFLOW {label} components={} graphs={} loop={}",
            engine.workflow_component_count(),
            engine.workflow_graph_component_count(),
            engine.workflow_declares_generation_loop()
        );
    }
    eprintln!("CORPUS_DETERMINISM covered={covered}");
    require_corpus_coverage(
        "real_packages_present_a_workflow_deterministically",
        covered,
    );
}

/// The decoder component's ports are exactly the resolved ABI's ports.
///
/// This is "no second writable answer" made checkable on real packages: the
/// ABI the optimized decode path binds is *derived from* the workflow, so the
/// two cannot disagree about the graph — there is nowhere else for either to
/// have come from.
#[test]
fn decoder_ports_mirror_the_resolved_abi() {
    let mut covered = 0usize;
    for (label, dir) in corpus() {
        let Some(engine) = open(&dir) else { continue };
        // Only a package whose whole graph is one decoder has a single ABI to
        // mirror; a composite one addresses its components explicitly.
        if engine.workflow_graph_component_count() != 1 {
            continue;
        }
        let workflow = engine.package_workflow().expect("loaded package").clone();
        let io = engine
            .metadata()
            .decoder_io()
            .unwrap_or_else(|| panic!("{label}: a single decoder must resolve an ABI"))
            .clone();
        let component = onnx_genai_metadata::sole_decoder_component(&workflow)
            .unwrap_or_else(|| panic!("{label}: no sole decoder component"));
        let decoder = &workflow.components[component];

        for port in io
            .token_input
            .iter()
            .chain(io.inputs_embeds_input.iter())
            .chain(io.attention_mask_input.iter())
            .chain(io.position_ids_input.iter())
            .chain(io.kv_inputs.iter().flatten())
        {
            assert!(
                decoder.ports.inputs.contains_key(port),
                "{label}: resolved ABI names input '{port}' the workflow does not declare"
            );
        }
        for port in io
            .logits_output
            .iter()
            .chain(io.hidden_output.iter())
            .chain(io.kv_outputs.iter().flatten())
        {
            assert!(
                decoder.ports.outputs.contains_key(port),
                "{label}: resolved ABI names output '{port}' the workflow does not declare"
            );
        }
        covered += 1;
    }
    eprintln!("CORPUS_PORT_MIRROR covered={covered}");
    require_corpus_coverage("decoder_ports_mirror_the_resolved_abi", covered);
}

/// No package declares the retired `model.io` block.
///
/// The point of the invariant, checked on whatever real packages exist rather
/// than only on fixtures: a package that still carried the retired shape would
/// be refused at load with a migration error, so finding one here would mean the
/// corpus is testing something that cannot run.
#[test]
fn no_real_package_declares_the_retired_block() {
    let mut covered = 0usize;
    for (label, dir) in corpus() {
        if !dir.is_dir() {
            continue;
        }
        if let Some(path) = onnx_genai_metadata::find_metadata_path(&dir) {
            let text = std::fs::read_to_string(&path).expect("metadata is readable");
            let document: serde_yaml::Value =
                serde_yaml::from_str(&text).expect("metadata is YAML");
            assert!(
                document
                    .get("model")
                    .and_then(|model| model.get("io"))
                    .is_none(),
                "{label}: package still declares the retired model.io block"
            );
            covered += 1;
        }
    }
    eprintln!("CORPUS_NO_RETIRED_BLOCK covered={covered}");
}

/// A composite workflow package presents the workflow it serialized.
#[test]
fn a_composite_workflow_package_presents_its_components() -> anyhow::Result<()> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/onnx_genai_workflows/gemma4_chained");
    let engine = Engine::from_dir(&root, EngineConfig::default())?;
    assert!(engine.workflow_graph_component_count() > 1);
    Ok(())
}
