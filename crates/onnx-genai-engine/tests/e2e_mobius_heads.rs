//! Pinned-Mobius-head real-model smoke harness.
//!
//! The declarative manifest lives at `tests/e2e/mobius_heads.json`. It records
//! the source Mobius PR head, the directory below `ONNX_GENAI_E2E_MODEL_DIR`,
//! and a deterministic greedy smoke assertion for every target.
//!
//! This is intentionally opt-in because the exported artifacts are not
//! versioned in this repository. Materialize each engine-ready model directory
//! as `$ONNX_GENAI_E2E_MODEL_DIR/<artifact_subdir>` with the manifest's
//! `required_files`; then run:
//!
//! ```bash
//! ONNX_GENAI_E2E_MODEL_DIR=/path/to/mobius-artifacts \
//! cargo test -p onnx-genai-engine --test e2e_mobius_heads -- --ignored --nocapture
//! ```
//!
//! No download is attempted. Missing roots, models, or required files are
//! reported and skipped so an absent artifact never fails CI.

use std::path::{Path, PathBuf};

use onnx_genai_engine::{Engine, EngineConfig, GeneratePrompt, GenerateRequest};
use serde::Deserialize;

const MANIFEST: &str = include_str!("../../../tests/e2e/mobius_heads.json");

#[derive(Debug, Deserialize)]
struct MobiusHeadManifest {
    models: Vec<MobiusHead>,
}

#[derive(Debug, Deserialize)]
struct MobiusHead {
    name: String,
    mobius_pr: u32,
    mobius_commit: String,
    artifact_subdir: PathBuf,
    required_files: Vec<PathBuf>,
    prompt: String,
    expected_text_contains: String,
    max_new_tokens: usize,
}

fn manifest() -> anyhow::Result<MobiusHeadManifest> {
    Ok(serde_json::from_str(MANIFEST)?)
}

fn artifact_root() -> Option<PathBuf> {
    std::env::var_os("ONNX_GENAI_E2E_MODEL_DIR").map(PathBuf::from)
}

fn model_is_present(root: &Path, model: &MobiusHead) -> bool {
    let model_dir = root.join(&model.artifact_subdir);
    if !model_dir.is_dir() {
        eprintln!(
            "skipping {} (Mobius PR #{} @ {}): artifact directory is absent: {}",
            model.name,
            model.mobius_pr,
            model.mobius_commit,
            model_dir.display()
        );
        return false;
    }

    let missing = model
        .required_files
        .iter()
        .map(|file| model_dir.join(file))
        .filter(|path| !path.is_file())
        .collect::<Vec<_>>();
    if !missing.is_empty() {
        eprintln!(
            "skipping {} (Mobius PR #{} @ {}): missing required artifact files: {}",
            model.name,
            model.mobius_pr,
            model.mobius_commit,
            missing
                .iter()
                .map(|path| path.display().to_string())
                .collect::<Vec<_>>()
                .join(", ")
        );
        return false;
    }

    true
}

#[test]
#[ignore = "requires pinned Mobius artifacts via ONNX_GENAI_E2E_MODEL_DIR"]
fn pinned_mobius_heads_generate_smoke_output() -> anyhow::Result<()> {
    let manifest = manifest()?;
    let Some(root) = artifact_root() else {
        eprintln!(
            "skipping pinned Mobius E2E: set ONNX_GENAI_E2E_MODEL_DIR to a directory containing \
             the manifest's artifact_subdir entries"
        );
        return Ok(());
    };
    if !root.is_dir() {
        eprintln!(
            "skipping pinned Mobius E2E: ONNX_GENAI_E2E_MODEL_DIR is absent: {}",
            root.display()
        );
        return Ok(());
    }

    for model in &manifest.models {
        if !model_is_present(&root, model) {
            continue;
        }

        let model_dir = root.join(&model.artifact_subdir);
        let mut engine = Engine::from_dir(&model_dir, EngineConfig::default())?;
        let mut request = GenerateRequest::new(GeneratePrompt::Text(model.prompt.clone()));
        request.options.max_new_tokens = model.max_new_tokens;
        request.options.temperature = 0.0;
        request.options.greedy = true;
        request.options.stop_on_eos = false;

        let result = engine.generate(request)?;
        assert!(
            !result.token_ids.is_empty(),
            "{} generated no tokens (Mobius PR #{} @ {})",
            model.name,
            model.mobius_pr,
            model.mobius_commit
        );
        assert!(
            result.text.contains(&model.expected_text_contains),
            "{} output did not contain {:?}: {:?}",
            model.name,
            model.expected_text_contains,
            result.text
        );
    }

    Ok(())
}
