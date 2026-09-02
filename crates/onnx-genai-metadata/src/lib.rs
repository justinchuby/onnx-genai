//! ONNX inference metadata parser and types.
//!
//! Implements the spec from <https://github.com/onnx/onnx/issues/8184>

/// The spelling of the base schema version that emitters stamp.
///
/// It stays `v1` because that is what every writer in this repository already
/// stamps and what 19 in-tree documents already say. Readers normalize the
/// spellings in the wild — absent, `v1`, `1`, `1.0` — onto
/// [`INITIAL_SCHEMA_VERSION`], so changing what is emitted would rewrite bytes
/// and semantic identities to say something no reader distinguishes.
///
/// A package that uses a field a later version introduced states that later
/// version instead, and must: see [`version`] and its typed feature-floor
/// constants, including [`COMPRESSED_STATE_SCHEMA_VERSION`].
pub const SCHEMA_VERSION: &str = "v1";

pub mod cache;
mod decoder_abi;
pub mod decoder_workflow;
pub mod extensions;
mod graph_cardinality;
pub mod identity;
mod lowering;
pub mod parser;
pub mod schema;
pub mod session_state;
mod state_plan;
pub mod tool_protocol;
pub mod validation;
pub mod version;

pub use cache::{CacheDependencies, cache_dependencies};
pub use decoder_abi::decoder_abi;
pub use graph_cardinality::{
    DecoderEvidence, GraphCardinality, WorkflowClassification, classify_workflow,
    is_single_decoder_workflow, sole_decoder_component,
};
pub use identity::{IDENTITY_SCHEME, semantic_identity, semantic_identity_of_str};
pub use lowering::{CompiledWorkflow, compile_workflow};
pub use parser::{
    MtpProposerSpec, SpeculatorConfigSource, SpeculatorDescriptor, SpeculatorProposerKind,
    SpeculatorProposerStatus, find_metadata_path, inspect_legacy_speculator_for_migration,
    load_metadata, load_metadata_from_dir, load_metadata_package, load_metadata_with_identity,
    load_pipeline_spec, parse_metadata, parse_metadata_json, resolve_package_artifact,
};
pub use schema::*;
pub use session_state::{
    SessionStateCarrier, SessionStateFacts, classify_session_state, session_group_aliases,
};
pub use state_plan::{
    ResolvedStateCell, ResolvedStatePlan, StateCarrySource, StateCarrySourceKind, StateFinalWriter,
    StateIdentity, StateLifecycle, StateReader, StateSemanticRole, StateServiceParticipation,
    StateSnapshotParticipation, StateSource, StateTransactionParticipation, StateUpdateRelation,
    StateWriter, resolve_state_plan, validate_state_plan,
};
pub use tool_protocol::{
    MAX_TOOL_CALL_ID_BYTES, MAX_TOOL_CALLS, MAX_TOOL_NAME_BYTES, MAX_TOOL_PAYLOAD_BYTES, ToolCall,
    ToolCallStream, ToolParseOutcome, ToolProtocol, ToolProtocolError,
    resolve as resolve_tool_protocol,
};
pub use validation::{
    PipelineValidationError, validate, validate_metadata, validate_pipeline_spec,
};
pub use version::{
    BATCHING_SCHEMA_VERSION, CANONICAL_SPECULATION_SCHEMA_VERSION, COMPRESSED_STATE_SCHEMA_VERSION,
    DFLASH_SCHEMA_VERSION, INITIAL_SCHEMA_VERSION, OUTPUT_PROTOCOL_SCHEMA_VERSION,
    PUBLICATION_MODE_SCHEMA_VERSION, SUPPORTED_SCHEMA_VERSION, SchemaVersion,
    TOKEN_AUTHORITY_SCHEMA_VERSION, TOOL_PROTOCOL_SCHEMA_VERSION,
};

/// Generates the inference-metadata JSON Schema with deterministic object-key ordering.
pub fn inference_metadata_schema_json() -> Result<String, serde_json::Error> {
    let schema = schemars::generate::SchemaSettings::draft2020_12()
        .into_generator()
        .into_root_schema_for::<InferenceMetadata>();
    let mut value = serde_json::to_value(schema)?;
    sort_json_object_keys(&mut value);
    serde_json::to_string_pretty(&value)
}

fn sort_json_object_keys(value: &mut serde_json::Value) {
    match value {
        serde_json::Value::Object(object) => {
            for value in object.values_mut() {
                sort_json_object_keys(value);
            }
            object.sort_keys();
        }
        serde_json::Value::Array(array) => {
            for value in array {
                sort_json_object_keys(value);
            }
        }
        _ => {}
    }
}

/// Error type for metadata operations.
#[derive(Debug, thiserror::Error)]
pub enum MetadataError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Parse error: {0}")]
    Parse(String),
}
