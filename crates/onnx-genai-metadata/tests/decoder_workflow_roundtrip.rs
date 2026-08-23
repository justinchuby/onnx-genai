//! Every converted package round-trips through the recognizer.
//!
//! `decoder_workflow` builds a workflow from a decoder ABI; `decoder_abi` reads
//! one back. The two are only trustworthy as a pair, and a mechanical
//! conversion of fourteen packages is only checkable if that pair is exact on
//! the packages themselves rather than on a hand-written example.
//!
//! What this pins is the property that made the conversion safe to do at all:
//! reading a converted package's workflow must yield the ports its graph
//! actually has. A regression here means the runtime silently binds a different
//! tensor than the package's author intended — the failure mode that a
//! by-eye review of fourteen YAML files would never catch.

use std::path::{Path, PathBuf};

use onnx_genai_metadata::{DecoderAbi, load_metadata, sole_decoder_component};

/// Every single-decoder fixture this repository maintains.
fn converted_packages() -> Vec<PathBuf> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    [
        "tests/fixtures/tiny-llm",
        "tests/fixtures/tiny-llm-sharedbuffer",
        "tests/fixtures/tiny-llm-scatter",
        "tests/fixtures/tiny-llm-explicit-io",
        "tests/fixtures/tiny-reasoning",
        "tests/fixtures/tiny-native-engine",
        "tests/fixtures/tiny-native-sub4-engine",
        "tests/fixtures/tiny-native-scalar-gqa",
        "tests/fixtures/tiny-mtp-full",
        "tests/fixtures/tiny-gemma4-assistant",
        "tests/fixtures/tiny-gemma4-assistant-mixed",
        "tests/fixtures/tiny-deepseek-v2-qmoe-attention",
        "tests/fixtures/tiny-glm52-qmoe-indexshare",
        "crates/onnx-genai-engine/tests/fixtures/model-package-cpu/cpu",
    ]
    .iter()
    .map(|relative| root.join(relative))
    .collect()
}

fn abi_of(package: &Path) -> DecoderAbi {
    let path = package.join("inference_metadata.yaml");
    let metadata = load_metadata(&path)
        .unwrap_or_else(|error| panic!("{}: metadata must load: {error}", package.display()));
    metadata
        .decoder_io()
        .unwrap_or_else(|| {
            panic!(
                "{}: a converted single-decoder package must resolve an ABI from its workflow",
                package.display()
            )
        })
        .clone()
}

/// Every converted package resolves its ABI from its workflow, and from nothing
/// else.
#[test]
fn every_converted_package_resolves_its_abi_from_its_workflow() {
    for package in converted_packages() {
        let path = package.join("inference_metadata.yaml");
        let metadata = load_metadata(&path)
            .unwrap_or_else(|error| panic!("{}: must load: {error}", package.display()));
        let workflow = &metadata
            .pipeline
            .as_ref()
            .unwrap_or_else(|| panic!("{}: must declare a workflow", package.display()))
            .workflow;
        assert!(
            sole_decoder_component(workflow).is_some(),
            "{}: must present exactly one decoder component",
            package.display()
        );
        let abi = abi_of(&package);
        assert!(
            abi.token_input.is_some() || abi.inputs_embeds_input.is_some(),
            "{}: must name the sequence that drives its loop",
            package.display()
        );
        assert!(
            abi.logits_output.is_some(),
            "{}: must name the output its token policy scores",
            package.display()
        );
    }
}

/// Rebuilding each package's workflow from its own resolved ABI reproduces that
/// ABI exactly.
///
/// This is the conversion's correctness property applied to the real packages:
/// `decoder_abi(decoder_workflow(abi)) == abi`. Because the workflow the
/// fixture ships was itself produced by `decoder_workflow`, a discrepancy here
/// means the committed package and the current builder disagree — i.e. the
/// fixtures have drifted from the code that generated them.
#[test]
fn rebuilding_each_package_reproduces_its_abi() {
    use onnx_genai_metadata::decoder_workflow::{DecoderFacts, decoder_workflow};

    for package in converted_packages() {
        let abi = abi_of(&package);
        // Carry the real port contracts across so the rebuild states the same
        // graph the fixture does; the ranks differ per cache discipline and a
        // default would make this test assert a shape no graph has.
        let path = package.join("inference_metadata.yaml");
        let metadata = load_metadata(&path).expect("loads");
        let workflow = &metadata.pipeline.as_ref().expect("workflow").workflow;
        let component = sole_decoder_component(workflow).expect("decoder");
        let declaration = &workflow.components[component];
        let port_contracts = declaration
            .ports
            .inputs
            .iter()
            .chain(declaration.ports.outputs.iter())
            .map(|(port, contract)| (port.clone(), contract.clone()))
            .collect();

        let rebuilt = decoder_workflow(
            &abi,
            "model.onnx",
            &DecoderFacts {
                max_sequence_length: None,
                eos_token_ids: Vec::new(),
                port_contracts,
            },
        )
        .unwrap_or_else(|error| panic!("{}: its own ABI must rebuild: {error}", package.display()));
        let read_back = onnx_genai_metadata::decoder_abi(
            &rebuilt,
            sole_decoder_component(&rebuilt).expect("rebuilt decoder"),
        )
        .expect("rebuilt workflow must be readable");

        assert_eq!(
            read_back,
            abi,
            "{}: rebuild changed the ABI",
            package.display()
        );
    }
}

/// A package with more than one graph is never "a single decoder", however
/// recognizable its decoder is.
///
/// This is the distinction that a real published package caught: a 187-component
/// any-to-any model has exactly one component carrying `token_ids`/`logits`
/// roles — its text head — so "has a recognizable decoder" was true of it. Had
/// that stood in for "is only a decoder", the loader would have handed a
/// 186-graph package to the fused single-graph executor, which cannot run the
/// other 185. The same mistake classifies every vision-language package.
#[test]
fn a_multi_component_package_is_not_a_single_decoder() {
    use onnx_genai_metadata::is_single_decoder_workflow;

    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    for (relative, expected_single) in [
        // One ONNX graph plus the runtime's token policy.
        ("tests/fixtures/tiny-llm", true),
        ("tests/fixtures/tiny-llm-scatter", true),
        // A vision encoder, a projector and a decoder. Its decoder is
        // recognizable; the package is not a decoder.
        ("tests/fixtures/onnx_genai_workflows/vlm", false),
        ("tests/fixtures/onnx_genai_workflows/gemma4_chained", false),
        ("tests/fixtures/onnx_genai_workflows/tts", false),
    ] {
        let path = root.join(relative).join("inference_metadata.yaml");
        let metadata =
            load_metadata(&path).unwrap_or_else(|error| panic!("{relative}: must load: {error}"));
        let workflow = &metadata
            .pipeline
            .as_ref()
            .unwrap_or_else(|| panic!("{relative}: must declare a workflow"))
            .workflow;
        assert_eq!(
            is_single_decoder_workflow(workflow),
            expected_single,
            "{relative}: misclassified; it declares {} components",
            workflow.components.len()
        );
    }
}
