#![cfg(feature = "native-backend")]

use onnx_genai_engine::{
    CompressedStateLoadRefusal, CompressedStatePathStats, NativeDecodeDevice,
    NativeDecodeMetadataRefusal, NativeDecodeSession, compressed_state_map_lookups,
};
use onnx_genai_metadata::{
    MetadataError, SUPPORTED_SCHEMA_VERSIONS, SchemaDocumentContext, SchemaFamily, SchemaVersion,
};

#[test]
fn external_callers_can_match_typed_compressed_state_refusal() {
    let error = anyhow::Error::new(CompressedStateLoadRefusal::UnsupportedDevice);
    assert!(matches!(
        error.downcast_ref::<CompressedStateLoadRefusal>(),
        Some(CompressedStateLoadRefusal::UnsupportedDevice)
    ));

    fn accepts_metadata_refusal(_: &NativeDecodeMetadataRefusal) {}
    let _ = accepts_metadata_refusal;
}

#[test]
fn external_callers_can_match_future_schema_without_string_parsing() {
    let error = onnx_genai_metadata::parse_metadata(
        "schema_version: \"v1.9\"\nfuture_section: {}\n",
        Some("yaml"),
    )
    .expect_err("future schema must be refused");
    let MetadataError::UnsupportedSchema(unsupported) = error else {
        panic!("future schema was not preserved as a public typed refusal");
    };
    assert_eq!(unsupported.family, SchemaFamily::InferenceMetadata);
    assert_eq!(unsupported.observed, SchemaVersion::new(1, 9));
    assert_eq!(unsupported.supported, SUPPORTED_SCHEMA_VERSIONS);
    assert_eq!(unsupported.document, SchemaDocumentContext::InMemory);
}

#[test]
fn public_surface_does_not_export_state_authority_internals() {
    let public_reexports = |source: &str| {
        let mut result = String::new();
        let mut in_reexport = false;
        for line in source.lines() {
            if line.trim_start().starts_with("pub use ") {
                in_reexport = true;
            }
            if in_reexport {
                result.push_str(line);
                result.push('\n');
                if line.contains(';') {
                    in_reexport = false;
                }
            }
        }
        result
    };
    let surface = format!(
        "{}{}",
        public_reexports(include_str!("../src/lib.rs")),
        public_reexports(include_str!("../src/native_decode/mod.rs"))
    );
    for private in [
        "RecordStateSpec",
        "CarryStateSpec",
        "CompressedStatePlan",
        "CompressedStateIndexes",
        "CompressedStateTransitionIndex",
        "device_scratch",
        "state_past_names",
    ] {
        assert!(
            !surface.contains(private),
            "private state authority leaked through a public re-export: {private}"
        );
    }
}

#[test]
fn absent_state_production_session_has_zero_lookup_through_teardown() -> anyhow::Result<()> {
    let model = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/tiny-llm/model.onnx.textproto");
    let before = compressed_state_map_lookups();
    {
        let mut session = NativeDecodeSession::load(model, NativeDecodeDevice::Cpu)?;
        session.decode(&[1, 2, 3], 0)?;
        for token in [4, 5, 6] {
            let past = session.current_len();
            session.decode(&[token], past)?;
        }
        session.rewind(4)?;
        session.reset()?;
        assert_eq!(
            session.compressed_state_path_stats(),
            CompressedStatePathStats::default()
        );
    }
    assert_eq!(
        compressed_state_map_lookups(),
        before,
        "default-off prefill/warm/rewind/reset/teardown must not probe a compressed-state map"
    );
    Ok(())
}
