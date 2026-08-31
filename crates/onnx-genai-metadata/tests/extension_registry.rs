use onnx_genai_metadata::{
    extensions::{
        ATEM_XML_V1, BUILTIN_EXTENSIONS, CORE_CONFORMANCE, DFLASH_FLAT_BLOCK_V1,
        ExtensionAdmissionError, ExtensionConsumerSupport, ExtensionSurface, FallbackClass,
        KV_CHECKPOINT_V1, SPECULATIVE_V1, SupportStatus, TAGGED_JSON_V1, TOKEN_CONTEXT_V1,
        admit_exact, extension_registry_markdown, find,
    },
    inference_metadata_schema_json, parse_metadata,
    version::{
        CANONICAL_SPECULATION_SCHEMA_VERSION, DFLASH_SCHEMA_VERSION, INITIAL_SCHEMA_VERSION,
        PUBLICATION_MODE_SCHEMA_VERSION, SUPPORTED_SCHEMA_VERSION, SchemaVersion,
        TOKEN_CONTEXT_SCHEMA_VERSION, TOOL_PROTOCOL_SCHEMA_VERSION,
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
    let unique = BUILTIN_EXTENSIONS
        .iter()
        .map(|descriptor| (descriptor.id.identity, descriptor.id.version))
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(
        unique.len(),
        BUILTIN_EXTENSIONS.len(),
        "every extension identity/version pair must have exactly one authoritative row"
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
fn core_conformance_covers_every_supported_schema_floor() {
    assert_eq!(
        INITIAL_SCHEMA_VERSION.major, SUPPORTED_SCHEMA_VERSION.major,
        "the current minor-version catalogue assumes one supported major"
    );
    let expected = (INITIAL_SCHEMA_VERSION.minor..=SUPPORTED_SCHEMA_VERSION.minor)
        .map(|minor| SchemaVersion::new(SUPPORTED_SCHEMA_VERSION.major, minor))
        .collect::<Vec<_>>();
    let catalogued = CORE_CONFORMANCE
        .iter()
        .map(|(version, obligations)| {
            assert!(
                !obligations.trim().is_empty(),
                "{version} must name its mandatory reader obligations"
            );
            *version
        })
        .collect::<Vec<_>>();
    assert_eq!(
        catalogued, expected,
        "every schema floor accepted by this reader must have exactly one ordered core-conformance row"
    );
}

#[test]
fn publication_mode_is_mandatory_core_conformance() {
    let obligation = CORE_CONFORMANCE
        .iter()
        .find_map(|(version, obligations)| {
            (*version == PUBLICATION_MODE_SCHEMA_VERSION).then_some(*obligations)
        })
        .expect("publication-mode floor has a core-conformance row");
    for required in [
        "`commit_only`",
        "`provisional_revisions`",
        "reconciles revisions transactionally",
        "`commit`",
        "`abort_to_baseline`",
    ] {
        assert!(
            obligation.contains(required),
            "publication-mode obligation must contain {required}: {obligation}"
        );
    }
    assert!(
        BUILTIN_EXTENSIONS
            .iter()
            .all(|descriptor| descriptor.declaration != "pipeline.workflow.publication_mode"),
        "publication_mode is core schema conformance, not an optional semantic extension"
    );
}

#[test]
fn exact_pair_admission_separates_registry_knowledge_from_consumer_support() {
    let admitted = admit_exact(
        ExtensionSurface::ToolProtocol,
        TAGGED_JSON_V1.identity,
        TAGGED_JSON_V1.version,
        "package.tool_protocol",
        ExtensionConsumerSupport::Supported {
            scope: "tagged-json v1 envelopes",
        },
        "select a supported tool protocol",
    )
    .expect("an exact implemented pair with a matching reader is admitted");
    assert_eq!(admitted.descriptor.id, TAGGED_JSON_V1);

    let error = admit_exact(
        ExtensionSurface::ToolProtocol,
        TAGGED_JSON_V1.identity,
        TAGGED_JSON_V1.version,
        "package.tool_protocol",
        ExtensionConsumerSupport::Unsupported {
            scope: "CPU profile",
            reason: "the requested backend/profile is outside this reader's support scope",
            guidance: "select the supported backend/profile",
        },
        "select a supported tool protocol",
    )
    .expect_err("registry-known must not imply runtime-supported");
    assert!(matches!(
        *error,
        ExtensionAdmissionError::ConsumerUnavailable { .. }
    ));
    let message = error.to_string();
    assert!(
        message.contains("tagged-json@v1")
            && message.contains("CPU profile")
            && message.contains("backend/profile"),
        "{message}"
    );
}

#[test]
fn checkpoint_exact_pairs_fail_closed_with_registry_guidance() {
    let support = ExtensionConsumerSupport::Unsupported {
        scope: "portable state checkpoint adapters on every backend/profile",
        reason: "no portable checkpoint adapter is installed",
        guidance: "omit checkpoint to keep state private",
    };
    let known = admit_exact(
        ExtensionSurface::StateCheckpoint,
        KV_CHECKPOINT_V1.identity,
        KV_CHECKPOINT_V1.version,
        "pipeline.workflow.serving.state_service.groups.decoder_cache.checkpoint",
        support,
        "use a registered implemented checkpoint pair or omit checkpoint",
    )
    .expect_err("known unavailable checkpoint must fail closed");
    assert!(matches!(
        *known,
        ExtensionAdmissionError::RegistryUnavailable { .. }
    ));

    let version = admit_exact(
        ExtensionSurface::StateCheckpoint,
        KV_CHECKPOINT_V1.identity,
        "2",
        "pipeline.workflow.serving.state_service.groups.decoder_cache.checkpoint",
        support,
        "use a registered implemented checkpoint pair or omit checkpoint",
    )
    .expect_err("unknown checkpoint version must fail closed");
    assert!(matches!(
        *version,
        ExtensionAdmissionError::UnknownVersion { .. }
    ));

    let identity = admit_exact(
        ExtensionSurface::StateCheckpoint,
        "onnx-genai.tensor-checkpoint",
        "1",
        "pipeline.workflow.serving.state_service.groups.decoder_cache.checkpoint",
        support,
        "use a registered implemented checkpoint pair or omit checkpoint",
    )
    .expect_err("invented checkpoint identity must fail closed");
    assert!(matches!(
        *identity,
        ExtensionAdmissionError::UnknownIdentity { .. }
    ));
    let message = identity.to_string();
    assert!(
        message.contains("onnx-genai.tensor-checkpoint@1")
            && message.contains("onnx-genai.kv-checkpoint@1")
            && message.contains("decoder_cache"),
        "{message}"
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
