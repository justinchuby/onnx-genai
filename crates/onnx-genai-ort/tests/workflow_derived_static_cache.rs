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
