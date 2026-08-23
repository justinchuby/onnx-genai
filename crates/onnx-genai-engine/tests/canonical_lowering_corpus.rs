//! The canonical lowering, exercised against the real model corpus.
//!
//! `onnx-genai-metadata`'s unit tests pin the translation against a hand-written
//! ABI. These run it over whatever real packages this machine has, through the
//! engine's own loader — which is the only thing that resolves a compatibility
//! (`genai_config.json`) package's ABI out of the ONNX graph. That is what
//! covers the shapes a hand-written fixture never has: shared-buffer decoders,
//! static caches, hybrid recurrent state, MoE graphs, and models with dozens of
//! KV pairs.
//!
//! Every case skips with a reason when its package is absent, and
//! `corpus_inventory` prints exactly which artifacts were missing — a silent
//! skip would let this suite report success on a weightless machine.

use std::path::{Path, PathBuf};

use onnx_genai_engine::{Engine, EngineConfig, WorkflowProvenance};

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

/// Fail a corpus case that covered nothing, unless the machine declares it has
/// no corpus.
///
/// A case that iterates an empty corpus and returns `Ok` reports success without
/// checking anything — exactly the silent pass this suite's doc claims not to
/// allow. A weightless machine must therefore opt out explicitly with
/// `ONNX_GENAI_ALLOW_EMPTY_CORPUS=1`, which turns a green run into a legible
/// "covered nothing" rather than an indistinguishable pass.
fn require_corpus_coverage(case: &str, covered: usize) {
    if covered > 0 {
        return;
    }
    assert!(
        std::env::var_os("ONNX_GENAI_ALLOW_EMPTY_CORPUS").is_some(),
        "{case} covered no real packages. Install at least one corpus package (see \
         corpus_inventory for the list this machine is missing), or set \
         ONNX_GENAI_ALLOW_EMPTY_CORPUS=1 to acknowledge a weightless run."
    );
    eprintln!("CANONICAL_CORPUS {case}: covered nothing (ONNX_GENAI_ALLOW_EMPTY_CORPUS set)");
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

/// Every real decoder package lowers, deterministically, and reports itself as
/// lowered rather than authored.
#[test]
fn real_packages_lower_deterministically() {
    let mut covered = 0usize;
    for (label, dir) in corpus() {
        let Some(engine) = open(&dir) else { continue };
        if engine.is_workflow() {
            assert_eq!(
                engine.workflow_provenance(),
                WorkflowProvenance::Authored,
                "{label}"
            );
            continue;
        }
        let first = engine
            .canonical_workflow_document()
            .unwrap_or_else(|error| panic!("{label}: lowering failed: {error:#}"));
        for _ in 0..3 {
            assert_eq!(
                engine.canonical_workflow_document().expect("repeat"),
                first,
                "{label}: lowering is not deterministic"
            );
        }
        engine
            .canonical_workflow()
            .unwrap_or_else(|error| panic!("{label}: lowered document did not parse: {error:#}"));
        assert_eq!(
            engine.workflow_provenance(),
            WorkflowProvenance::Lowered,
            "{label}: a lowerable decoder must report itself lowered"
        );
        covered += 1;
        eprintln!("CANONICAL_CORPUS_LOWERED {label}");
    }
    eprintln!("CANONICAL_CORPUS_DETERMINISM covered={covered}");
    require_corpus_coverage("real_packages_lower_deterministically", covered);
}

/// The lowered decoder component's ports are exactly the resolved ABI's ports.
///
/// This is "no second writable answer" made checkable on real packages: the
/// lowered form cannot say anything the ABI does not, so the two can never
/// disagree about the graph.
#[test]
fn lowered_ports_mirror_the_resolved_abi() {
    let mut covered = 0usize;
    for (label, dir) in corpus() {
        let Some(engine) = open(&dir) else { continue };
        if engine.is_workflow() {
            continue;
        }
        let Ok(workflow) = engine.canonical_workflow() else {
            continue;
        };
        let io = engine
            .metadata()
            .decoder_io()
            .unwrap_or_else(|| panic!("{label}: lowered without a resolved ABI"))
            .clone();
        let decoder = &workflow.components[onnx_genai_metadata::DECODER_COMPONENT];

        let mut expected_inputs = Vec::new();
        match io.sequence_source {
            Some(onnx_genai_metadata::SequenceInputKind::InputsEmbeds) => {
                expected_inputs.extend(io.inputs_embeds_input.clone())
            }
            _ => expected_inputs.extend(io.token_input.clone()),
        }
        expected_inputs.extend(io.attention_mask_input.clone());
        expected_inputs.extend(io.position_ids_input.clone());
        expected_inputs.extend(io.kv_inputs.clone().unwrap_or_default());
        expected_inputs.sort();
        let mut actual_inputs = decoder.ports.inputs.keys().cloned().collect::<Vec<_>>();
        actual_inputs.sort();
        assert_eq!(
            actual_inputs, expected_inputs,
            "{label}: lowered decoder inputs diverged from the resolved ABI"
        );

        let mut expected_outputs = Vec::new();
        expected_outputs.extend(io.logits_output.clone());
        expected_outputs.extend(io.hidden_output.clone());
        expected_outputs.extend(io.kv_outputs.clone().unwrap_or_default());
        expected_outputs.sort();
        let mut actual_outputs = decoder.ports.outputs.keys().cloned().collect::<Vec<_>>();
        actual_outputs.sort();
        assert_eq!(
            actual_outputs, expected_outputs,
            "{label}: lowered decoder outputs diverged from the resolved ABI"
        );

        assert_eq!(
            workflow.state.len(),
            io.kv_inputs.as_ref().map_or(0, Vec::len),
            "{label}: lowered state cells do not match the declared KV pairs"
        );
        covered += 1;
    }
    eprintln!("CANONICAL_CORPUS_PORT_MIRROR covered={covered}");
    require_corpus_coverage("lowered_ports_mirror_the_resolved_abi", covered);
}

/// Lowering never mutates the package's serialized metadata.
///
/// The parent's constraint: `model.io` stays the sole serialized answer, so
/// `validate_model_io_against_workflow` never sees a pair and no published
/// package needs re-authoring.
#[test]
fn real_packages_keep_model_io_as_the_sole_serialized_answer() {
    let mut covered = 0usize;
    for (label, dir) in corpus() {
        let Some(engine) = open(&dir) else { continue };
        if engine.is_workflow() {
            continue;
        }
        let _ = engine.canonical_workflow_document();
        assert!(
            engine.metadata().pipeline.is_none(),
            "{label}: lowering wrote a workflow back into the package metadata"
        );
        // The authored file is untouched and still passes validation as written.
        if let Some(path) = onnx_genai_metadata::find_metadata_path(&dir) {
            let reread = onnx_genai_metadata::load_metadata(&path)
                .unwrap_or_else(|error| panic!("{label}: metadata no longer loads: {error}"));
            assert!(
                reread.pipeline.is_none(),
                "{label}: the package on disk gained a pipeline section"
            );
            assert!(
                onnx_genai_metadata::validate_metadata(&reread).is_ok(),
                "{label}: the authored package stopped validating"
            );
        }
        covered += 1;
    }
    eprintln!("CANONICAL_CORPUS_UNMUTATED covered={covered}");
    require_corpus_coverage(
        "real_packages_keep_model_io_as_the_sole_serialized_answer",
        covered,
    );
}

/// An authored workflow package is never lowered beside its own workflow.
#[test]
fn an_authored_workflow_package_is_not_lowered() -> anyhow::Result<()> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/onnx_genai_workflows/gemma4_chained");
    let engine = Engine::from_dir(&root, EngineConfig::default())?;
    assert_eq!(engine.workflow_provenance(), WorkflowProvenance::Authored);
    let error = engine
        .canonical_workflow_document()
        .expect_err("an authored workflow must not be lowered beside itself");
    assert!(
        format!("{error:#}").contains("already canonical"),
        "{error:#}"
    );
    Ok(())
}
