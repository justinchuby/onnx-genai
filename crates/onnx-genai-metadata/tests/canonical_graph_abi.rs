//! One serialized expression of a package's executable graph ABI.
//!
//! A package used to be able to say what its decode step looks like twice: once
//! as `model.io`, and once as the component ports, invoke bindings and state
//! groups the workflow engine actually executes. Two writable answers to one
//! question is a defect whatever their contents, because nothing forces them to
//! agree and the runtime that reads only one never learns that the other said
//! something else.
//!
//! The workflow is now the only place that answer is written. These tests pin
//! the consequence that matters: a bare single ONNX decoder and a composite
//! multi-graph package are not two representations with a shared subset — they
//! are the same representation, resolved by the same call, and a package that
//! tries to restate the answer beside it is refused.

use std::path::{Path, PathBuf};

use onnx_genai_metadata::{
    InferenceMetadata, KvOwnership, PortRole, SequenceInputKind, load_metadata, validate_metadata,
};

fn package(name: &str) -> InferenceMetadata {
    let path = fixture(name);
    let metadata = load_metadata(&path).expect("canonical package parses");
    validate_metadata(&metadata).expect("canonical package validates");
    metadata
}

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/onnx_genai_workflows")
        .join(name)
        .join("inference_metadata.yaml")
}

/// The bare case: one ONNX file, driven from the workflow like everything else.
///
/// This package has a single executable graph. It is the case an optimized
/// single-model path exists for, and historically the case that justified a
/// serialized `model.io`. It carries no `model` block at all, so if the ABI
/// resolves here it resolved from component ports, port roles and the state
/// group — the same facts the workflow engine executes.
#[test]
fn a_bare_single_onnx_decoder_declares_its_abi_only_in_the_workflow() {
    let metadata = package("decoder");
    assert!(
        metadata.model.is_none(),
        "the bare decoder package must not carry a model block"
    );

    let io = metadata
        .decoder_io()
        .expect("the decode ABI resolves from the workflow alone");
    assert!(
        !metadata.decoder_io_is_legacy(),
        "the ABI must be recognized, not read from a serialized block"
    );

    assert_eq!(io.token_input.as_deref(), Some("input_ids"));
    assert_eq!(io.attention_mask_input.as_deref(), Some("attention_mask"));
    assert_eq!(io.position_ids_input.as_deref(), Some("position_ids"));
    assert_eq!(io.logits_output.as_deref(), Some("logits"));
    assert_eq!(io.sequence_source, Some(SequenceInputKind::TokenIds));
    assert_eq!(io.kv_ownership, Some(KvOwnership::Owned));
    assert_eq!(
        io.kv_inputs.as_deref(),
        Some(["past_key_values.0.key".to_string()].as_slice())
    );
    assert_eq!(
        io.kv_outputs.as_deref(),
        Some(["present.0.key".to_string()].as_slice())
    );
}

/// The composite case: three ONNX files, resolved by the identical call.
///
/// Nothing in the resolution changes shape because the package grew a vision
/// encoder and an embedding graph. The decoder is still recognized structurally
/// — it consumes the autoregressive sequence and owns attention state — and the
/// only difference visible in the resolved ABI is the one the graph really has:
/// this decoder is fed embeddings rather than tokens.
#[test]
fn a_composite_package_resolves_its_decoder_abi_through_the_same_call() {
    let metadata = package("vlm");
    assert!(
        metadata.model.is_none(),
        "the composite package must not carry a model block"
    );

    let io = metadata
        .decoder_io()
        .expect("the composite decode ABI resolves from the workflow alone");
    assert!(!metadata.decoder_io_is_legacy());

    assert_eq!(io.inputs_embeds_input.as_deref(), Some("inputs_embeds"));
    assert_eq!(io.token_input, None);
    assert_eq!(io.sequence_source, Some(SequenceInputKind::InputsEmbeds));
    assert_eq!(io.attention_mask_input.as_deref(), Some("attention_mask"));
    assert_eq!(io.position_ids_input.as_deref(), Some("position_ids"));
    assert_eq!(io.logits_output.as_deref(), Some("logits"));
    assert_eq!(io.kv_ownership, Some(KvOwnership::Owned));
}

/// Split key/value buffers keep their halves in the order the producer declared.
///
/// A split cache exposes two shape-identical buffers per layer, and the map key
/// that names them is a producer label whose lexicographic order is not the
/// graph's. The declared role is what pairs them, so this asserts the resolved
/// order rather than trusting that `cache_0` happened to sort before `cache_1`.
#[test]
fn a_split_cache_resolves_keys_before_values_within_a_layer() {
    let metadata = package("vlm");
    let io = metadata.decoder_io().expect("composite ABI resolves");

    assert_eq!(
        io.kv_inputs.as_deref(),
        Some(
            [
                "past_key_values.0.key".to_string(),
                "past_key_values.0.value".to_string()
            ]
            .as_slice()
        )
    );
    assert_eq!(
        io.kv_outputs.as_deref(),
        Some(["present.0.key".to_string(), "present.0.value".to_string()].as_slice())
    );
}

/// Both packages answer the same questions, from the same fields, by the same call.
///
/// The point is not that the two ABIs are equal — they describe different graphs
/// and must differ. It is that every fact either one supplies is supplied by the
/// same mechanism, so there is no fact a bare package can express that a
/// composite one cannot, and no second serialized form that only one of them
/// uses.
#[test]
fn the_bare_and_composite_packages_share_one_representation() {
    let bare = package("decoder");
    let composite = package("vlm");

    for metadata in [&bare, &composite] {
        let io = metadata.decoder_io().expect("ABI resolves");
        assert!(!metadata.decoder_io_is_legacy());
        assert!(metadata.model.is_none());
        assert!(io.sequence_source.is_some(), "sequence source is declared");
        assert_eq!(io.kv_ownership, Some(KvOwnership::Owned));
        assert!(io.kv_inputs.is_some() && io.kv_outputs.is_some());
        assert_eq!(
            io.kv_inputs.as_ref().map(Vec::len),
            io.kv_outputs.as_ref().map(Vec::len),
            "every past buffer has exactly one present buffer"
        );
        assert_eq!(io.logits_output.as_deref(), Some("logits"));
    }

    // The graphs really are different, so a test that passed by resolving
    // nothing would be worthless.
    assert_ne!(
        bare.decoder_io().unwrap().sequence_source,
        composite.decoder_io().unwrap().sequence_source
    );
}

/// The recognizer identifies the decoder by structure, never by component name.
///
/// `decoder/` calls its graph `model` and `vlm/` calls its graph `decoder`. If
/// recognition consulted the name, exactly one of these would resolve.
#[test]
fn the_decoder_is_recognized_by_structure_not_by_component_name() {
    for (name, component) in [("decoder", "model"), ("vlm", "decoder")] {
        let metadata = package(name);
        let workflow = &metadata
            .pipeline
            .as_ref()
            .expect("workflow package")
            .workflow;
        assert_eq!(
            onnx_genai_metadata::sole_decoder_component(workflow),
            Some(component),
            "package '{name}' must recognize its decode graph"
        );
    }
}

/// A component's declared roles name ports the component actually has.
///
/// A role is only worth trusting if it points at a real port; a role naming a
/// port the component does not declare would resolve to nothing while looking
/// like an answer.
#[test]
fn declared_port_roles_name_declared_ports() {
    for name in ["decoder", "vlm"] {
        let metadata = package(name);
        let workflow = &metadata
            .pipeline
            .as_ref()
            .expect("workflow package")
            .workflow;
        for (component, declaration) in &workflow.components {
            let ports = &declaration.ports;
            for (port, role) in &ports.roles {
                let is_output = matches!(role, PortRole::Logits | PortRole::HiddenStates);
                let declared = if is_output {
                    ports.outputs.contains_key(port)
                } else {
                    ports.inputs.contains_key(port)
                };
                assert!(
                    declared,
                    "package '{name}' component '{component}' gives port '{port}' role \
                     {role:?} but does not declare that port"
                );
            }
        }
    }
}
