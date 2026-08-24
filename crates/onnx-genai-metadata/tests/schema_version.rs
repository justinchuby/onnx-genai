//! What a document's declared schema version buys, and what it has to cost.
//!
//! Every structure in this schema denies unknown fields, so absence — not
//! tolerance — is the compatibility mechanism: a package that uses nothing new
//! keeps loading on every runtime it loaded on before, and one that uses
//! something new needs a reader that knows it. Two things follow, and both are
//! tested here. A reader has to compare versions *before* it deserializes, or a
//! newer document is reported as a typo. And a document has to declare the
//! version whose fields it actually uses, or the comparison is made against a
//! number that is not true.

use onnx_genai_metadata::{
    INITIAL_SCHEMA_VERSION, SCHEMA_VERSION, SUPPORTED_SCHEMA_VERSION, parse_metadata,
    parse_metadata_json, validate_metadata, version,
};

/// The smallest document that says nothing new.
const PLAIN: &str = "model:\n  vocab_size: 32000\n";

/// The smallest document that uses a field 1.1 introduced.
const PADDED: &str = r#"
pipeline:
  workflow:
    manifest:
      capabilities: [workflow_ssa, typed_emit]
    inputs:
      pixel_values:
        contract:
          dtype: float32
          rank: 3
          shape: [batch, tiles, hidden]
          batch_layout: { kind: request_aligned, axis: 0 }
          padding: [{ dimension: tiles, valid_lengths: tile_lengths }]
        role: { kind: opaque }
        source: { kind: application, name: pixel_values }
      tile_lengths:
        contract: { dtype: int64, rank: 1, shape: [batch], batch_layout: { kind: shared } }
        role: { kind: opaque }
        source: { kind: application, name: tile_lengths }
    outputs:
      tokens:
        contract: { dtype: int64, rank: 2, shape: [batch, generated], batch_layout: { kind: request_aligned, axis: 0 } }
        role: tokens
        stage: pre_adapter
    components:
      vision:
        implementation: { kind: onnx, artifact: vision.onnx }
        ports:
          inputs:
            pixels:
              dtype: float32
              rank: 3
              shape: [batch, tiles, hidden]
              batch_layout: { kind: request_aligned, axis: 0 }
              padding: [{ dimension: tiles, valid_lengths: lengths }]
            lengths: { dtype: int64, rank: 1, shape: [batch], batch_layout: { kind: shared } }
          outputs:
            token: { dtype: int64, rank: 2, shape: [batch, generated], batch_layout: { kind: request_aligned, axis: 0 } }
    steps:
      - kind: invoke
        component: vision
        inputs: { pixels: pixel_values, lengths: tile_lengths }
        outputs: { token: raw }
      - kind: emit
        value: raw
        output: tokens
        mode: replace
"#;

fn with_version(document: &str, spelling: Option<&str>) -> String {
    match spelling {
        Some(spelling) => format!("schema_version: \"{spelling}\"\n{document}"),
        None => document.to_string(),
    }
}

#[test]
fn every_spelling_the_first_version_was_ever_written_in_still_loads() {
    // Nothing ever forced a canonical spelling, so four exist. A reader that
    // compared strings would refuse three correct packages.
    for spelling in [None, Some("v1"), Some("1"), Some("1.0"), Some("v1.0")] {
        let document = with_version(PLAIN, spelling);
        parse_metadata(&document, Some("yaml"))
            .unwrap_or_else(|error| panic!("{spelling:?} must still load: {error}"));
    }
}

#[test]
fn the_constant_an_emitter_stamps_is_the_canonical_spelling_of_that_version() {
    assert_eq!(SCHEMA_VERSION, "1.0");
    assert_eq!(
        version::normalize(Some(SCHEMA_VERSION)).expect("the emitted constant is a version"),
        INITIAL_SCHEMA_VERSION
    );
}

#[test]
fn a_newer_document_is_refused_by_version_rather_than_by_the_first_field_it_uses() {
    // This is the whole point of reading the version first. The document below
    // is well formed at 1.2 and merely unreadable here; without the gate the
    // reader would report `unknown field` and send someone hunting for a typo.
    let document =
        format!("schema_version: \"1.2\"\nfuture_section: {{ shape: circular }}\n{PLAIN}");
    let error = parse_metadata(&document, Some("yaml")).expect_err("1.2 is newer than this build");
    let error = error.to_string();
    assert!(
        error.contains("schema version 1.2") && error.contains("reads up to 1.1"),
        "{error}"
    );
    assert!(
        !error.contains("unknown field"),
        "a newer document is not a malformed one: {error}"
    );
}

#[test]
fn a_different_major_version_is_a_different_contract() {
    let error = parse_metadata(&with_version(PLAIN, Some("2.0")), Some("yaml"))
        .expect_err("2.0 is a different contract");
    assert!(error.to_string().contains("major version"), "{error}");
}

#[test]
fn a_version_no_one_can_compare_says_how_to_write_one() {
    let error = parse_metadata(&with_version(PLAIN, Some("latest")), Some("yaml"))
        .expect_err("'latest' is not a version");
    let error = error.to_string();
    assert!(error.contains("'<major>.<minor>'"), "{error}");
    assert!(error.contains("1.1"), "{error}");
}

#[test]
fn the_gate_is_on_the_path_a_file_takes() {
    // The gate is worth nothing if a loader can go around it, so this exercises
    // the real file entry point rather than the string one.
    let directory = tempfile::tempdir().expect("temporary directory");
    let path = directory.path().join("inference_metadata.yaml");
    std::fs::write(&path, with_version(PLAIN, Some("1.2"))).expect("write the document");
    let error =
        onnx_genai_metadata::load_metadata(&path).expect_err("1.2 is newer than this build");
    assert!(error.to_string().contains("reads up to 1.1"), "{error}");

    std::fs::write(&path, with_version(PLAIN, Some("v1"))).expect("write the document");
    onnx_genai_metadata::load_metadata(&path).expect("an old spelling still loads from a file");
}

#[test]
fn the_gate_is_on_the_path_a_document_built_in_memory_takes() {
    // A lowering that builds its own document is exactly as capable of stamping
    // a version it does not mean as a file on disk is.
    let mut document = serde_json::json!({ "model": { "vocab_size": 32000 } });
    document["schema_version"] = serde_json::json!("1.2");
    let error = parse_metadata_json(&document).expect_err("1.2 is newer than this build");
    assert!(error.to_string().contains("reads up to 1.1"), "{error}");

    document["schema_version"] = serde_json::json!(SCHEMA_VERSION);
    parse_metadata_json(&document).expect("the canonical base version parses");
}

#[test]
fn a_document_that_uses_a_batching_field_declares_the_version_that_introduced_it() {
    for spelling in [None, Some("v1"), Some("1.0")] {
        let document = with_version(PADDED, spelling);
        let metadata = parse_metadata(&document, Some("yaml")).expect("the document parses");
        let errors = validate_metadata(&metadata).expect_err("the declared version is not true");
        let reported = errors.join("\n");
        assert!(
            reported.contains("declares padding on"),
            "{spelling:?}: {reported}"
        );
        assert!(
            reported.contains("which schema version 1.1 introduced"),
            "{spelling:?}: {reported}"
        );
    }
}

#[test]
fn declaring_the_version_the_fields_belong_to_is_all_it_takes() {
    let metadata = parse_metadata(&with_version(PADDED, Some("1.1")), Some("yaml"))
        .expect("the document parses");
    validate_metadata(&metadata).expect("a truthfully versioned document is valid");
    assert_eq!(
        version::normalize(metadata.schema_version.as_deref()).expect("a version"),
        SUPPORTED_SCHEMA_VERSION
    );
}

#[test]
fn a_document_that_uses_nothing_new_is_free_to_say_nothing() {
    // The compatibility promise runs the other way too: absence must stay valid,
    // or every existing package would have to be re-emitted to keep loading.
    let metadata = parse_metadata(PLAIN, Some("yaml")).expect("the document parses");
    assert!(metadata.schema_version.is_none());
    validate_metadata(&metadata).expect("an unversioned plain document is valid");
}
