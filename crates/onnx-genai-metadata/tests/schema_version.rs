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
//!
//! One limit on the first sentence, tested in `encoder_batching.rs` rather than
//! here: it holds for fields that were *added*, not for a field that was
//! reshaped. While this schema is pre-release a field may change shape outright
//! — `token_packed` moved its ownership pair into a `levels` chain — and then a
//! document using nothing new still does not load, because the spelling it uses
//! is gone rather than merely older. That is a refusal naming the migration,
//! never a silent reinterpretation, and it is not something a version number
//! arbitrates.

use onnx_genai_metadata::{
    INITIAL_SCHEMA_VERSION, SCHEMA_VERSION, SUPPORTED_SCHEMA_VERSION, WorkflowOutputFamily,
    parse_metadata, parse_metadata_json, validate_metadata, version,
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
          shape: [batch, tiles, hidden]
          batch_layout: { kind: request_aligned, axis: 0 }
          padding: [{ dimension: tiles, valid_lengths: tile_lengths }]
        role: { kind: opaque }
        source: { kind: application, name: pixel_values }
      tile_lengths:
        contract: { dtype: int64, shape: [batch], batch_layout: { kind: request_aligned, axis: 0 } }
        role: { kind: opaque }
        source: { kind: application, name: tile_lengths }
    outputs:
      tokens:
        contract: { dtype: int64, shape: [batch, generated], batch_layout: { kind: request_aligned, axis: 0 } }
        role: tokens
        stage: pre_adapter
    components:
      vision:
        implementation: { kind: onnx, artifact: vision.onnx }
        ports:
          inputs:
            pixels:
              dtype: float32
              shape: [batch, tiles, hidden]
              batch_layout: { kind: request_aligned, axis: 0 }
              padding: [{ dimension: tiles, valid_lengths: lengths }]
            lengths: { dtype: int64, shape: [batch], batch_layout: { kind: request_aligned, axis: 0 } }
          outputs:
            token: { dtype: int64, shape: [batch, generated], batch_layout: { kind: request_aligned, axis: 0 } }
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

const OUTPUT_WITHOUT_FAMILY: &str = r#"
pipeline:
  workflow:
    manifest: { capabilities: [workflow_ssa, typed_emit] }
    outputs:
      answer:
        contract: { dtype: int64, shape: [sequence] }
        role: tensor
        stage: pre_adapter
    components: {}
    steps: []
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
fn the_constant_an_emitter_stamps_is_the_one_it_already_stamped() {
    // Changing this would rewrite the version string of every document a writer
    // touches, and with it the semantic identity, to say something no reader
    // distinguishes from what it said before.
    assert_eq!(SCHEMA_VERSION, "v1");
    assert_eq!(
        version::normalize(Some(SCHEMA_VERSION)).expect("the emitted constant is a version"),
        INITIAL_SCHEMA_VERSION
    );
    assert_eq!(INITIAL_SCHEMA_VERSION.to_string(), "v1.0");
    assert_eq!(SUPPORTED_SCHEMA_VERSION.to_string(), "v1.6");
}

#[test]
fn a_newer_document_is_refused_by_version_rather_than_by_the_first_field_it_uses() {
    // This is the whole point of reading the version first. The document below
    // is well formed at 1.7 and merely unreadable here; without the gate the
    // reader would report `unknown field` and send someone hunting for a typo.
    let document =
        format!("schema_version: \"1.7\"\nfuture_section: {{ shape: circular }}\n{PLAIN}");
    let error = parse_metadata(&document, Some("yaml")).expect_err("1.7 is newer than this build");
    let error = error.to_string();
    assert!(
        error.contains("schema version v1.7") && error.contains("reads up to v1.6"),
        "{error}"
    );
    assert!(
        !error.contains("unknown field"),
        "a newer document is not a malformed one: {error}"
    );
}

#[test]
fn v1_4_without_family_keeps_legacy_materialized_round_trip() {
    let document = with_version(OUTPUT_WITHOUT_FAMILY, Some("v1.4"));
    let metadata = parse_metadata(&document, Some("yaml")).expect("legacy output parses");
    let output = &metadata
        .pipeline
        .as_ref()
        .expect("pipeline")
        .workflow
        .outputs["answer"];
    assert_eq!(output.family, WorkflowOutputFamily::Materialized);
    assert!(!output.family_authored);

    let round_trip = serde_yaml::to_string(output).expect("legacy output serializes");
    assert!(
        !round_trip.contains("family:"),
        "serialization must not upgrade a v1.4 output by adding v1.5 semantics: {round_trip}"
    );
    let reparsed: onnx_genai_metadata::WorkflowOutput =
        serde_yaml::from_str(&round_trip).expect("legacy output round-trip parses");
    assert!(!reparsed.family_authored);
}

#[test]
fn v1_4_rejects_each_explicit_output_family_at_the_version_boundary() {
    for family in [
        "{ kind: materialized }",
        "{ kind: events }",
        "{ kind: revisions, version: \"1\" }",
    ] {
        let document = with_version(
            &OUTPUT_WITHOUT_FAMILY.replace(
                "stage: pre_adapter",
                &format!("family: {family}\n        stage: pre_adapter"),
            ),
            Some("v1.4"),
        );
        let error = parse_metadata(&document, Some("yaml"))
            .expect_err("v1.4 must not opt into output-family semantics");
        let reported = error.to_string();
        assert!(
            reported.contains("pipeline.workflow.outputs.answer.family")
                && reported.contains("authored schema version v1.4")
                && reported.contains("minimum schema version v1.5")
                && reported.contains("migrate/re-emit"),
            "{family}: {reported}"
        );
    }
}

#[test]
fn typed_deserialization_retains_authored_family_for_validation_and_admission() {
    let document = with_version(
        &OUTPUT_WITHOUT_FAMILY.replace(
            "stage: pre_adapter",
            "family: { kind: materialized }\n        stage: pre_adapter",
        ),
        Some("v1.4"),
    );
    let metadata: onnx_genai_metadata::InferenceMetadata =
        serde_yaml::from_str(&document).expect("typed deserializer retains the declaration");
    assert!(
        metadata
            .pipeline
            .as_ref()
            .expect("pipeline")
            .workflow
            .outputs["answer"]
            .family_authored
    );
    let errors = validate_metadata(&metadata).expect_err("typed validation must apply the gate");
    assert!(
        errors
            .join("\n")
            .contains("pipeline.workflow.outputs.answer.family"),
        "{errors:#?}"
    );
}

#[test]
fn typed_validation_gates_other_v1_5_output_protocol_fields() {
    for authored in [
        "stream: named\n        mode: append",
        "mode: retract",
        "mode: finalize",
    ] {
        let document = with_version(
            &OUTPUT_WITHOUT_FAMILY.replace(
                "steps: []",
                &format!(
                    "steps:\n      - kind: emit\n        value: produced\n        output: answer\n        {authored}"
                ),
            ),
            Some("v1.4"),
        );
        let metadata: onnx_genai_metadata::InferenceMetadata =
            serde_yaml::from_str(&document).expect("typed deserializer retains the emit");
        let errors =
            validate_metadata(&metadata).expect_err("typed validation must apply the v1.5 gate");
        assert!(
            errors
                .iter()
                .any(|error| error.contains("minimum schema version v1.5")),
            "{authored}: {errors:#?}"
        );
    }
}

#[test]
fn v1_5_requires_family_and_valid_families_round_trip() {
    let without_family = with_version(OUTPUT_WITHOUT_FAMILY, Some("v1.5"));
    let error = parse_metadata(&without_family, Some("yaml"))
        .expect_err("v1.5 output protocol must name an output family");
    assert!(
        error
            .to_string()
            .contains("pipeline.workflow.outputs.answer.family is required"),
        "{error}"
    );

    for family in [
        "{ kind: materialized }",
        "{ kind: events }",
        "{ kind: revisions, version: \"1\" }",
    ] {
        let document = with_version(
            &OUTPUT_WITHOUT_FAMILY.replace(
                "stage: pre_adapter",
                &format!("family: {family}\n        stage: pre_adapter"),
            ),
            Some("v1.5"),
        );
        let metadata = parse_metadata(&document, Some("yaml")).expect("v1.5 family parses");
        let output = &metadata
            .pipeline
            .as_ref()
            .expect("pipeline")
            .workflow
            .outputs["answer"];
        let serialized = serde_yaml::to_string(output).expect("v1.5 family serializes");
        assert!(serialized.contains("family:"), "{family}: {serialized}");
        let reparsed: onnx_genai_metadata::WorkflowOutput =
            serde_yaml::from_str(&serialized).expect("v1.5 family round-trips");
        assert!(reparsed.family_authored);
        assert_eq!(reparsed.family, output.family);
    }

    let unsupported_revision_version = with_version(
        &OUTPUT_WITHOUT_FAMILY.replace(
            "stage: pre_adapter",
            "family: { kind: revisions, version: \"2\" }\n        stage: pre_adapter",
        ),
        Some("v1.5"),
    );
    let metadata =
        parse_metadata(&unsupported_revision_version, Some("yaml")).expect("typed document parses");
    let errors = validate_metadata(&metadata).expect_err("unknown revision version is refused");
    assert!(
        errors
            .join("\n")
            .contains("outputs.answer.family.version is '2'"),
        "{errors:#?}"
    );
}

#[test]
fn v1_4_rejects_stream_and_typed_revision_only_emit_operations() {
    for authored in [
        "stream: named\n        mode: append",
        "mode: retract",
        "mode: finalize",
    ] {
        let document = with_version(
            &OUTPUT_WITHOUT_FAMILY.replace(
                "steps: []",
                &format!(
                    "steps:\n      - kind: emit\n        value: produced\n        output: answer\n        {authored}"
                ),
            ),
            Some("v1.4"),
        );
        let error = parse_metadata(&document, Some("yaml"))
            .expect_err("v1.5-only emit declarations must fail below the boundary");
        let reported = error.to_string();
        assert!(
            reported.contains("authored schema version v1.4")
                && reported.contains("minimum schema version v1.5")
                && reported.contains("pipeline.workflow.steps[0]"),
            "{authored}: {reported}"
        );
    }
}

#[test]
fn output_family_rejects_an_illegal_emit_at_its_authored_site() {
    let document = r#"
schema_version: "v1.5"
pipeline:
  workflow:
    manifest: { capabilities: [workflow_ssa, typed_emit] }
    outputs:
      answer:
        contract: { dtype: int64, shape: [sequence] }
        role: tensor
        family: { kind: materialized }
        stage: pre_adapter
    components: {}
    steps:
      - kind: emit
        value: produced
        output: answer
        mode: event
"#;
    let metadata = parse_metadata(document, Some("yaml")).expect("document parses");
    let errors = validate_metadata(&metadata).expect_err("event is not materialized");
    let reported = errors.join("\n");
    assert!(
        reported.contains("pipeline.workflow.steps[0] selects Event for output 'answer'"),
        "{reported}"
    );
}

#[test]
fn retired_streaming_emit_is_rejected_with_output_family_migration_guidance() {
    let document = r#"
pipeline:
  workflow:
    manifest: { capabilities: [workflow_ssa, typed_emit, streaming_emit] }
    components: {}
    steps: []
"#;
    let error = parse_metadata(document, Some("yaml"))
        .expect_err("retired streaming capability has no parallel authority");
    let reported = error.to_string();
    assert!(
        reported.contains("retired capability `streaming_emit`")
            && reported.contains("canonical `family`"),
        "{reported}"
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
    assert!(error.contains("'v<major>.<minor>'"), "{error}");
    assert!(error.contains("v1.6"), "{error}");

    // Three components is not this grammar either, however plausible it looks.
    assert!(parse_metadata(&with_version(PLAIN, Some("v1.2.3")), Some("yaml")).is_err());
}

#[test]
fn the_gate_is_on_the_path_a_file_takes() {
    // The gate is worth nothing if a loader can go around it, so this exercises
    // the real file entry point rather than the string one.
    let directory = tempfile::tempdir().expect("temporary directory");
    let path = directory.path().join("inference_metadata.yaml");
    std::fs::write(&path, with_version(PLAIN, Some("1.7"))).expect("write the document");
    let error =
        onnx_genai_metadata::load_metadata(&path).expect_err("1.7 is newer than this build");
    assert!(error.to_string().contains("reads up to v1.6"), "{error}");

    std::fs::write(&path, with_version(PLAIN, Some("v1"))).expect("write the document");
    onnx_genai_metadata::load_metadata(&path).expect("an old spelling still loads from a file");
}

#[test]
fn the_gate_is_on_the_path_a_document_built_in_memory_takes() {
    // A lowering that builds its own document is exactly as capable of stamping
    // a version it does not mean as a file on disk is.
    let mut document = serde_json::json!({ "model": { "vocab_size": 32000 } });
    document["schema_version"] = serde_json::json!("1.7");
    let error = parse_metadata_json(&document).expect_err("1.7 is newer than this build");
    assert!(error.to_string().contains("reads up to v1.6"), "{error}");

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
            reported.contains("which schema version v1.1 introduced"),
            "{spelling:?}: {reported}"
        );
    }
}

#[test]
fn declaring_the_version_the_fields_belong_to_is_all_it_takes() {
    // `v1.1` is the canonical spelling a new batching document emits, and `1.1`
    // is an accepted synonym on read for the same reason the four spellings of
    // the first version are.
    for spelling in ["v1.1", "1.1"] {
        let metadata = parse_metadata(&with_version(PADDED, Some(spelling)), Some("yaml"))
            .expect("the document parses");
        validate_metadata(&metadata)
            .unwrap_or_else(|errors| panic!("'{spelling}' is truthful: {errors:#?}"));
        assert_eq!(
            version::normalize(metadata.schema_version.as_deref()).expect("a version"),
            version::BATCHING_SCHEMA_VERSION
        );
    }
}

#[test]
fn a_document_that_uses_nothing_new_is_free_to_say_nothing() {
    // The compatibility promise runs the other way too: absence must stay valid,
    // or every existing package would have to be re-emitted to keep loading.
    let metadata = parse_metadata(PLAIN, Some("yaml")).expect("the document parses");
    assert!(metadata.schema_version.is_none());
    validate_metadata(&metadata).expect("an unversioned plain document is valid");
}

#[test]
fn tokenizer_special_tokens_require_the_version_that_introduced_them() {
    let document = r#"
schema_version: v1.1
package:
  tokenizer:
    special_tokens:
      eos_token_id: [2]
"#;
    let metadata = parse_metadata(document, Some("yaml")).expect("the document parses");
    let errors = validate_metadata(&metadata).expect_err("the declared version is not true");
    assert!(
        errors
            .iter()
            .any(|error| error.contains("schema version v1.2 introduced")),
        "{errors:?}"
    );

    let document = r#"
schema_version: v1.2
package:
  tokenizer:
    special_tokens:
      eos_token_id: [2]
"#;
    let metadata = parse_metadata(document, Some("yaml")).expect("the document parses");
    validate_metadata(&metadata).expect("v1.2 truthfully declares token facts");
}

#[test]
fn the_canonical_spelling_of_a_new_batching_document_carries_the_v() {
    // Two versions are in play and they are canonically spelled the same way.
    // `v1` is what a writer stamps on a document that uses nothing new; `v1.1`
    // is what a document using a batching field must say. Dropping the `v` from
    // the newer one would leave one schema with two house styles.
    assert_eq!(SCHEMA_VERSION, "v1");
    assert_eq!(version::BATCHING_SCHEMA_VERSION.to_string(), "v1.1");
    assert_eq!(SUPPORTED_SCHEMA_VERSION.to_string(), "v1.6");

    // And it is the spelling the document is *told* to write, not merely one the
    // reader tolerates.
    let errors = validate_metadata(&parse_metadata(PADDED, Some("yaml")).expect("parses"))
        .expect_err("an unversioned batching document is not truthful");
    let reported = errors.join("\n");
    assert!(
        reported.contains("declare schema_version 'v1.1'"),
        "{reported}"
    );
}

#[test]
fn a_synonym_is_read_but_is_not_the_canonical_spelling() {
    // `1.1` normalizes to the same version for the same reason the four
    // spellings of the first version do — a reader compares versions, not
    // strings — but a writer emitting a new batching document writes `v1.1`.
    assert_eq!(
        version::normalize(Some("1.1")).expect("a synonym is a version"),
        version::BATCHING_SCHEMA_VERSION
    );
    assert_ne!("1.1", version::BATCHING_SCHEMA_VERSION.to_string());
}

#[test]
fn reading_a_document_never_rewrites_the_version_it_declares() {
    // A version string is part of a package's semantic identity, so normalizing
    // for comparison must not become normalizing on disk. A reader that
    // helpfully rewrote `v1` to `1.0` would change what every existing package
    // hashes to in exchange for nothing a reader can observe — so the spelling
    // survives the load verbatim, and two spellings of one version stay two
    // distinct identities rather than being silently merged.
    let mut identities = std::collections::BTreeSet::new();
    for spelling in [None, Some("v1"), Some("1"), Some("1.0"), Some("v1.0")] {
        let document = with_version(PLAIN, spelling);
        let metadata = parse_metadata(&document, Some("yaml"))
            .unwrap_or_else(|error| panic!("{spelling:?} loads: {error}"));
        assert_eq!(metadata.schema_version.as_deref(), spelling, "{spelling:?}");
        identities.insert(
            onnx_genai_metadata::semantic_identity_of_str(&document)
                .unwrap_or_else(|error| panic!("{spelling:?} has an identity: {error}")),
        );
    }
    assert_eq!(
        identities.len(),
        5,
        "normalizing for comparison must not collapse the identities on disk"
    );
}
