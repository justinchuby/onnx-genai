use onnx_genai_metadata::{
    extensions::{
        ATEM_XML_V1, BUILTIN_EXTENSIONS, DFLASH_FLAT_BLOCK_V1, FallbackClass, SPECULATIVE_V1,
        SupportStatus, TAGGED_JSON_V1, TOKEN_CONTEXT_V1, extension_registry_markdown, find,
    },
    inference_metadata_schema_json, parse_metadata,
    version::{
        CANONICAL_SPECULATION_SCHEMA_VERSION, DFLASH_SCHEMA_VERSION, TOKEN_CONTEXT_SCHEMA_VERSION,
        TOOL_PROTOCOL_SCHEMA_VERSION,
    },
};

#[test]
fn committed_extension_registry_is_generated_from_the_machine_source() {
    assert_eq!(
        include_str!("../../../docs/genai/METADATA_EXTENSION_REGISTRY.md"),
        extension_registry_markdown(),
        "regenerate with `cargo run -p onnx-genai-metadata --bin gen_extension_registry`"
    );
}

#[test]
fn registry_lists_exact_current_optional_semantic_extensions() {
    for id in [
        TAGGED_JSON_V1,
        ATEM_XML_V1,
        TOKEN_CONTEXT_V1,
        SPECULATIVE_V1,
        DFLASH_FLAT_BLOCK_V1,
    ] {
        assert_eq!(
            find(id.identity, id.version).map(|descriptor| descriptor.id),
            Some(id),
            "{}@{} must have one registry row",
            id.identity,
            id.version
        );
    }
    assert_eq!(
        BUILTIN_EXTENSIONS
            .iter()
            .filter(|descriptor| descriptor.id == DFLASH_FLAT_BLOCK_V1)
            .count(),
        1,
        "an identity/version pair must be unambiguous"
    );
    assert!(
        BUILTIN_EXTENSIONS
            .iter()
            .all(|descriptor| descriptor.fallback == FallbackClass::SemanticRequired),
        "runtime optimizations are not package extension requirements"
    );
    assert!(BUILTIN_EXTENSIONS.iter().any(|descriptor| {
        descriptor.id.identity == "onnx-genai.dflash-flat-block"
            && descriptor.status == SupportStatus::KnownButUnavailable
    }));
    assert_eq!(
        find(TAGGED_JSON_V1.identity, TAGGED_JSON_V1.version)
            .expect("registered tool protocol")
            .schema_floor,
        TOOL_PROTOCOL_SCHEMA_VERSION
    );
    assert_eq!(
        find(TOKEN_CONTEXT_V1.identity, TOKEN_CONTEXT_V1.version)
            .expect("registered token context")
            .schema_floor,
        TOKEN_CONTEXT_SCHEMA_VERSION
    );
    assert_eq!(
        find(SPECULATIVE_V1.identity, SPECULATIVE_V1.version)
            .expect("registered speculation")
            .schema_floor,
        CANONICAL_SPECULATION_SCHEMA_VERSION
    );
    assert_eq!(
        find(DFLASH_FLAT_BLOCK_V1.identity, DFLASH_FLAT_BLOCK_V1.version)
            .expect("registered DFlash")
            .schema_floor,
        DFLASH_SCHEMA_VERSION
    );
}

#[test]
fn schema_no_longer_exposes_negotiable_core_capability_lists() {
    let schema = inference_metadata_schema_json().expect("schema serializes");
    assert!(
        !schema.contains("\"required_capabilities\""),
        "core conformance must be selected by schema_version, not a top-level list"
    );
    let manifest = &serde_json::to_string(
        &serde_json::from_str::<serde_json::Value>(&schema).expect("schema is JSON")["$defs"]
            ["WorkflowManifest"],
    )
    .expect("manifest serializes");
    assert!(
        !manifest.contains("\"capabilities\""),
        "workflow structure must not redundantly advertise core semantics"
    );
}

#[test]
fn retired_capability_lists_have_actionable_typed_extension_migration() {
    let error = parse_metadata(
        "schema_version: v1\nrequired_capabilities: [workflow_ssa]\n",
        None,
    )
    .expect_err("the core capability list is retired")
    .to_string();
    assert!(
        error.contains("required_capabilities")
            && error.contains("schema_version")
            && error.contains("typed identity/version"),
        "{error}"
    );

    let error = parse_metadata(
        "schema_version: v1\npipeline:\n  workflow:\n    manifest:\n      capabilities: [typed_emit]\n",
        None,
    )
    .expect_err("workflow core capability list is retired")
    .to_string();
    assert!(
        error.contains("manifest.capabilities") && error.contains("typed workflow"),
        "{error}"
    );
}
