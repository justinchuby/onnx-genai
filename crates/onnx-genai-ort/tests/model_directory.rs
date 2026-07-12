use std::path::{Path, PathBuf};

use onnx_genai_ort::{
    ModelDirectory, ProposalType, SpeculatorConfigSource, SpeculatorProposerKind,
    SpeculatorProposerStatus, detect_speculator,
};

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

#[test]
fn detects_huggingface_speculator_config() {
    let directory =
        ModelDirectory::load(fixture("speculator-eagle3")).expect("model directory resolves");
    let speculator = directory.speculator.expect("speculator is detected");

    assert_eq!(speculator.proposal_type, ProposalType::Eagle3);
    assert_eq!(speculator.num_speculative_tokens, 4);
    assert_eq!(
        speculator
            .verifier
            .as_ref()
            .and_then(|verifier| verifier.name_or_path.as_deref()),
        Some("Qwen/Qwen3-8B")
    );
    assert_eq!(
        speculator
            .verifier
            .as_ref()
            .map(|verifier| verifier.architectures.as_slice()),
        Some(["Qwen3ForCausalLM".to_string()].as_slice())
    );
    assert_eq!(speculator.source, SpeculatorConfigSource::HuggingFaceConfig);
    assert_eq!(
        speculator.proposer,
        SpeculatorProposerStatus::NotYetSupported(SpeculatorProposerKind::Eagle3)
    );
}

#[test]
fn unknown_proposal_type_is_preserved() {
    let speculator =
        detect_speculator(&fixture("speculator-unknown")).expect("speculator is detected");

    assert_eq!(
        speculator.proposal_type,
        ProposalType::Unknown("future-tree-v2".to_string())
    );
    assert_eq!(speculator.num_speculative_tokens, 7);
    assert_eq!(
        speculator.proposer,
        SpeculatorProposerStatus::Unknown("future-tree-v2".to_string())
    );
}

#[test]
fn native_inference_metadata_takes_precedence() {
    let speculator =
        detect_speculator(&fixture("speculator-native")).expect("speculator is detected");

    assert_eq!(speculator.proposal_type, ProposalType::Mtp);
    assert_eq!(speculator.num_speculative_tokens, 3);
    assert_eq!(speculator.source, SpeculatorConfigSource::InferenceMetadata);
}
