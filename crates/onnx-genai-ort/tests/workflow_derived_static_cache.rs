//! A workflow-only package drives the static-cache decoder.
//!
//! The scatter driver takes a [`ModelIoSpec`], and that type shares its name
//! with the serialized `model.io` block this branch retired. The two are not
//! the same thing: the parameter is the *resolved* decode ABI that
//! [`InferenceMetadata::decoder_io`] returns, and a workflow package
//! synthesizes it from the `state_service` group's scatter update and the
//! decoder component's declared port roles.
//!
//! The distinction is invisible in a type signature, so it is pinned here
//! against a real ONNX graph instead: the package beside the fixture declares
//! no `model:` block at all, and the driver still classifies the graph and
//! executes prefill and decode through it. Without this test the only evidence
//! that the canonical form reaches the runtime is a chain of call sites, which
//! is exactly the kind of claim that reads as unsatisfiable from the outside.

use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use onnx_genai_metadata::InferenceMetadata;
use onnx_genai_ort::{
    Environment, Session, SessionOptions, StaticCacheDecodeOptions, StaticCacheDecodeSession,
};

fn deterministic_session_options() -> SessionOptions {
    SessionOptions::default().with_intra_op_threads(1)
}

fn test_environment() -> &'static Environment {
    static ENVIRONMENT: OnceLock<Environment> = OnceLock::new();
    ENVIRONMENT.get_or_init(|| Environment::new("workflow-static-cache-test").expect("env"))
}

fn ort_test_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

fn tiny_scatter_llm() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/tiny-llm-scatter/model.onnx.textproto")
}

/// The package that ships beside the scatter graph, workflow and all.
fn workflow_package() -> InferenceMetadata {
    let document =
        include_str!("../../../tests/fixtures/tiny-llm-scatter-workflow/inference_metadata.yaml");
    let metadata: InferenceMetadata = serde_yaml::from_str(document).expect("package parses");
    onnx_genai_metadata::validate_metadata(&metadata).expect("package validates");
    metadata
}

#[test]
fn a_package_with_no_model_block_still_binds_the_scatter_abi() {
    let metadata = workflow_package();
    assert!(
        metadata.model.is_none(),
        "the fixture must prove the workflow alone is sufficient"
    );
    let io = metadata
        .decoder_io()
        .expect("workflow derives a decode ABI");
    let cache = io
        .static_cache
        .as_ref()
        .expect("the scatter update derives a static-cache ABI");

    // The control ports are rank-one integer vectors and so are mutually
    // shape-indistinguishable; they can only come from the declaration.
    assert_eq!(cache.write_indices_input, "write_indices");
    assert_eq!(cache.kv_sequence_length_input, "nonpad_kv_seqlen");
    assert_eq!(cache.key_cache_inputs, ["key_cache.0"]);
    assert_eq!(cache.value_cache_inputs, ["value_cache.0"]);
    assert_eq!(cache.key_cache_outputs, ["updated_key_cache.0"]);
    assert_eq!(cache.value_cache_outputs, ["updated_value_cache.0"]);
    assert!(
        !metadata.decoder_io_is_legacy(),
        "the ABI must come from the workflow, not from a legacy fallback"
    );
}

#[test]
fn the_scatter_driver_runs_from_the_workflow_derived_abi() {
    let _guard = ort_test_lock().lock().expect("ORT test lock");
    let session = Session::new(
        test_environment(),
        &tiny_scatter_llm(),
        deterministic_session_options(),
    )
    .expect("session");

    let metadata = workflow_package();
    let io = metadata
        .decoder_io()
        .expect("workflow derives a decode ABI");

    let signature = StaticCacheDecodeSession::detect(&session, Some(io))
        .expect("detect")
        .expect("a workflow-declared scatter ABI classifies the graph");
    assert_eq!(signature.layers, 1);
    assert_eq!(signature.max_len, 16);

    let mut decode = StaticCacheDecodeSession::new(
        &session,
        StaticCacheDecodeOptions { batch_size: 1 },
        Some(io),
    )
    .expect("static decode session");
    let prefill = decode.prefill(&[1, 5], &[0, 1]).expect("prefill");
    assert_eq!(prefill.shape(), &[1, 2, 32]);
    assert_eq!(decode.current_len(), 2);

    // One decode step must advance the cursor by exactly one row position,
    // which is what makes the next scatter land on the end of the valid prefix.
    decode.step(&[7], &[2]).expect("decode step");
    assert_eq!(decode.current_len(), 3);
}

/// A component that declares no port map still yields a scatter ABI.
///
/// The scatter ABI is a fact about the state service, not about a component's
/// port list: the control-port names come from the group's `write_indices_ports`
/// and `kv_length_ports`, and the per-layer buffers from the group's own port
/// aliases. A producer that treats the `.onnx` file as the authoritative port
/// list — and so declines to transcribe every input contract into YAML — is
/// therefore still able to publish a bindable static cache.
///
/// This is pinned because the opposite reading is easy to reach: absent and
/// empty port maps are indistinguishable after `#[serde(default)]`, so a
/// consumer that treats absence as a claim of non-existence would reject
/// exactly the producers that declined to duplicate the graph.
#[test]
fn declared_roles_alone_yield_the_scatter_abi() {
    // Remove only the component's port map, leaving the rest of the document
    // byte-identical, so the assertion isolates that one field.
    let mut document: serde_yaml::Value = serde_yaml::from_str(include_str!(
        "../../../tests/fixtures/tiny-llm-scatter-workflow/inference_metadata.yaml"
    ))
    .expect("fixture parses");
    let component = document
        .get_mut("pipeline")
        .and_then(|pipeline| pipeline.get_mut("workflow"))
        .and_then(|workflow| workflow.get_mut("components"))
        .and_then(|components| components.get_mut("model"))
        .and_then(serde_yaml::Value::as_mapping_mut)
        .expect("the fixture declares the decoder component");
    let ports = component
        .get_mut(serde_yaml::Value::from("ports"))
        .and_then(serde_yaml::Value::as_mapping_mut)
        .expect("the fixture declares ports");
    assert!(
        ports.remove(serde_yaml::Value::from("inputs")).is_some()
            && ports.remove(serde_yaml::Value::from("outputs")).is_some(),
        "the fixture must transcribe contracts for their removal to prove anything"
    );

    let metadata: InferenceMetadata =
        serde_yaml::from_value(document).expect("a portless component still parses");
    let io = metadata
        .decoder_io()
        .expect("the state service alone derives an ABI");
    let cache = io
        .static_cache
        .as_ref()
        .expect("the scatter ABI comes from the state group, not the port map");
    assert_eq!(cache.write_indices_input, "write_indices");
    assert_eq!(cache.kv_sequence_length_input, "nonpad_kv_seqlen");
    assert_eq!(cache.key_cache_inputs, ["key_cache.0"]);
    // The one-line role declaration is honored on its own: the token port is
    // resolved from `roles`, not guessed from the spelling "input_ids".
    assert_eq!(io.token_input.as_deref(), Some("input_ids"));
}

/// Transcribed port contracts do not substitute for a declared role.
///
/// A producer migrating to the workflow-only form may reasonably assume that
/// writing full `TensorContract`s for every port is the *more* complete
/// declaration, and that roles are the optional shorthand. It is the other way
/// around. Every field of the decode ABI is resolved by role, so contracts
/// without roles resolve nothing: `decoder_io()` returns `None` and the runtime
/// silently falls back to inferring ports from shapes — the behaviour the
/// canonical form exists to remove.
///
/// That failure used to be invisible: the document validated. It is now
/// rejected at validation, naming the component and the missing role, so the
/// mistake surfaces where it is cheap to fix instead of as wrong ports at
/// inference time.
#[test]
fn port_contracts_do_not_substitute_for_a_declared_role() {
    let mut document: serde_yaml::Value = serde_yaml::from_str(include_str!(
        "../../../tests/fixtures/tiny-llm-scatter-workflow/inference_metadata.yaml"
    ))
    .expect("the canonical package parses");
    let component = document
        .get_mut("pipeline")
        .and_then(|value| value.get_mut("workflow"))
        .and_then(|value| value.get_mut("components"))
        .and_then(|value| value.get_mut("model"))
        .and_then(serde_yaml::Value::as_mapping_mut)
        .expect("the canonical package declares the model component");
    let ports = component
        .get_mut(serde_yaml::Value::from("ports"))
        .and_then(serde_yaml::Value::as_mapping_mut)
        .expect("the fixture declares ports");
    // Drop only the roles, leaving every transcribed contract in place.
    assert!(
        ports.remove(serde_yaml::Value::from("roles")).is_some(),
        "the fixture must declare roles for their removal to prove anything"
    );

    let metadata: InferenceMetadata =
        serde_yaml::from_str(&serde_yaml::to_string(&document).expect("re-serializes"))
            .expect("the roleless document still parses");

    assert!(
        metadata.decoder_io().is_none(),
        "contracts alone resolve no ABI, which is exactly why this must not validate"
    );

    let errors = onnx_genai_metadata::validation::validate_metadata(&metadata)
        .expect_err("a sole decoder that declares no sequence role is rejected");
    assert!(
        errors
            .iter()
            .any(|error| error.contains("declares no token_ids or inputs_embeds role")),
        "the rejection must name the missing role, got: {errors:?}"
    );
}
