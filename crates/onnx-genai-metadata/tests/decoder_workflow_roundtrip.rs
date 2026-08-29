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

use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
};

use onnx_genai_metadata::schema::{
    CompressedRecordFormat, CompressionRatio, StateGroupContract, StateGroupProperties, StateKind,
    WorkflowComponent, WorkflowSpec,
};
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
        "crates/onnx-genai-engine/tests/fixtures/tiny-deepseek-v4-csa",
        "crates/onnx-genai-engine/tests/fixtures/tiny-deepseek-v4-csa-schedule",
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

fn state_group_semantics(
    workflow: &WorkflowSpec,
    component: &str,
) -> BTreeMap<String, StateGroupContract> {
    workflow
        .serving
        .as_ref()
        .expect("decoder workflow must declare serving state")
        .state_service
        .groups
        .iter()
        .filter(|(_, group)| group.ports.contains_key(component))
        .map(|(name, group)| {
            let mut normalized = group.clone();
            for aliases in normalized.ports.values_mut() {
                *aliases = std::mem::take(aliases)
                    .into_values()
                    .map(|alias| (alias.input.clone(), alias))
                    .collect();
            }
            (name.clone(), normalized)
        })
        .collect()
}

fn assert_heterogeneous_compressed_fixture(
    package: &Path,
    declaration: &WorkflowComponent,
    groups: &BTreeMap<String, StateGroupContract>,
) {
    let fixture = package
        .file_name()
        .and_then(|name| name.to_str())
        .expect("fixture path has a UTF-8 name");
    if !matches!(
        fixture,
        "tiny-deepseek-v4-csa" | "tiny-deepseek-v4-csa-schedule"
    ) {
        return;
    }

    for (record, carry) in [
        ("compressed_records.0", "compressed_carries.0"),
        ("compressed_records.1", "compressed_carries.1"),
    ] {
        assert_eq!(
            groups.get(record).expect("fixture record group").layout,
            "batch_record_feature",
            "{}: {record} must retain its record layout",
            package.display()
        );
        assert_eq!(
            groups.get(carry).expect("fixture carry group").layout,
            "batch_carry_slot_stream_feature",
            "{}: {carry} must retain its carry layout",
            package.display()
        );
    }
    assert!(
        groups
            .values()
            .all(|group| { !group.reuse.prefix_reusable && !group.reuse.evictable_prefix }),
        "{}: compressed groups deliberately decline both reuse capabilities",
        package.display()
    );

    for (name, ratio, format, dtype) in [
        (
            "compressed_records.0",
            CompressionRatio::Ratio4,
            CompressedRecordFormat::Fp8E4m3Block64,
            "uint8",
        ),
        (
            "compressed_records.1",
            CompressionRatio::Ratio128,
            CompressedRecordFormat::F32,
            "float32",
        ),
    ] {
        let group = groups.get(name).expect("fixture record group");
        assert!(matches!(
            &group.properties,
            Some(StateGroupProperties::CompressedAttention {
                ratio: declared_ratio,
                record_format,
                ..
            }) if *declared_ratio == ratio && *record_format == format
        ));
        for alias in group
            .ports
            .get("decoder")
            .expect("compressed group binds decoder")
            .values()
        {
            assert_eq!(
                declaration
                    .ports
                    .inputs
                    .get(&alias.input)
                    .expect("compressed input contract")
                    .dtype,
                dtype,
                "{}: {name} input {} changed identity",
                package.display(),
                alias.input
            );
            assert_eq!(
                declaration
                    .ports
                    .outputs
                    .get(alias.output.as_ref().expect("read-write compressed output"))
                    .expect("compressed output contract")
                    .dtype,
                dtype,
                "{}: {name} output changed identity",
                package.display()
            );
        }
    }
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
        let expected_groups = state_group_semantics(workflow, component);
        let expected_compressed = expected_groups
            .iter()
            .filter(|(_, group)| group.kind == StateKind::CompressedAttention)
            .map(|(name, group)| (name.clone(), group.clone()))
            .collect();
        assert_heterogeneous_compressed_fixture(
            package.as_path(),
            declaration,
            &expected_compressed,
        );
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
                port_contracts,
            },
        )
        .unwrap_or_else(|error| panic!("{}: its own ABI must rebuild: {error}", package.display()));
        let rendered = serde_yaml::to_string(&rebuilt).unwrap_or_else(|error| {
            panic!("{}: rebuild must serialize: {error}", package.display())
        });
        let reemitted = serde_yaml::from_str(&rendered).unwrap_or_else(|error| {
            panic!(
                "{}: serialized rebuild must parse as a workflow: {error}",
                package.display()
            )
        });
        let mut rebuilt_metadata = onnx_genai_metadata::schema::InferenceMetadata::default();
        rebuilt_metadata.pipeline = Some(onnx_genai_metadata::schema::PipelineSpec {
            workflow: reemitted,
        });
        onnx_genai_metadata::validation::validate_metadata(&rebuilt_metadata).unwrap_or_else(
            |errors| {
                panic!(
                    "{}: rebuilt workflow must satisfy the production validator: {errors:#?}",
                    package.display()
                )
            },
        );
        let rebuilt = &rebuilt_metadata
            .pipeline
            .as_ref()
            .expect("pipeline")
            .workflow;
        let rebuilt_component = sole_decoder_component(rebuilt).expect("reemitted rebuilt decoder");
        let rebuilt_declaration = &rebuilt.components[rebuilt_component];
        assert_eq!(
            state_group_semantics(rebuilt, rebuilt_component),
            expected_groups,
            "{}: rebuild changed state-group semantics",
            package.display()
        );
        for group in &abi.state_groups {
            for port in &group.ports {
                if let Some(expected) = declaration.ports.inputs.get(&port.input) {
                    assert_eq!(
                        rebuilt_declaration.ports.inputs.get(&port.input),
                        Some(expected),
                        "{}: rebuild changed physical input contract for {}.{}",
                        package.display(),
                        group.name,
                        port.input
                    );
                }
                if let Some(expected) = declaration.ports.outputs.get(&port.output) {
                    assert_eq!(
                        rebuilt_declaration.ports.outputs.get(&port.output),
                        Some(expected),
                        "{}: rebuild changed physical output contract for {}.{}",
                        package.display(),
                        group.name,
                        port.output
                    );
                }
            }
        }
        let read_back = onnx_genai_metadata::decoder_abi(
            rebuilt,
            sole_decoder_component(rebuilt).expect("rebuilt decoder"),
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
