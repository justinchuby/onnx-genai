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
/// version instead, and must: see [`version`] and
/// [`BATCHING_SCHEMA_VERSION`] / [`TOKEN_AUTHORITY_SCHEMA_VERSION`].
pub const SCHEMA_VERSION: &str = "v1";

/// Built-in serialized capability identifiers.
///
/// A model package lists the identifiers it needs in
/// [`schema::InferenceMetadata::required_capabilities`]; a runtime that does not
/// advertise one of them fails the load through [`validation::validate`] with a
/// precise, actionable missing-capability error rather than guessing from the
/// model's identity. Workflow manifests use the same vocabulary.
///
/// The two serialized capability fields are intentionally open to
/// extension-defined identifiers. [`BUILTIN`] is the closed catalogue of
/// identifiers defined by this crate; it is the source used by the
/// documentation drift test.
pub mod capabilities {
    macro_rules! define_capabilities {
        ($( $(#[$meta:meta])* $name:ident = $value:literal; )+) => {
            $(
                $(#[$meta])*
                pub const $name: &str = $value;
            )+

            /// Every capability identifier built into this crate.
            pub const BUILTIN: &[&str] = &[$($name),+];
        };
    }

    define_capabilities! {
        /// Runtime-owned key/value state.
        KV_CACHE = "kv_cache";
        /// Grouped-query-attention execution.
        GROUPED_QUERY_ATTENTION = "grouped_query_attention";
        /// Multi-head-attention execution.
        MULTI_HEAD_ATTENTION = "multi_head_attention";
        /// Prefix-state reuse.
        PREFIX_CACHE = "prefix_cache";
        /// Legacy loop-control admission.
        CONTROL_FLOW_LOOP = "control_flow_loop";

        /// A typed image preprocessing transform program is required.
        IMAGE_PREPROCESSING_PROGRAM = "image_preprocessing_program";
        /// The program emits more than one packed image tensor output.
        PACKED_IMAGE_OUTPUTS = "packed_image_outputs";
        /// A declared multi-axis position-id program is required.
        POSITION_PROGRAM = "position_program";
        /// Multi-axis (rank greater than one) position coordinates are required.
        MULTI_AXIS_POSITIONS = "multi_axis_positions";
        /// Fixed-shape loop-carried recurrent state with replace semantics.
        LOOP_CARRIED_STATE = "loop_carried_state";
        /// A decoder consumes raw-token and routed-sequence inputs together.
        DUAL_SEQUENCE_INPUTS = "dual_sequence_inputs";

        /// Typed SSA workflow execution.
        WORKFLOW_SSA = "workflow_ssa";
        /// Explicit linear effect-token semantics.
        LINEAR_EFFECTS = "linear_effects";
        /// Runtime serving and state-service contracts.
        SERVING_SERVICE_CONTRACT = "serving_service_contract";
        /// Parameter-adapter application.
        PARAMETER_ADAPTERS = "parameter_adapters";
        /// Different adapter sets in one batch.
        HETEROGENEOUS_ADAPTER_BATCHING = "heterogeneous_adapter_batching";
        /// Runtime-leased state that outlives one invocation.
        SESSION_STATE_LEASE = "session_state_lease";
        /// State whose growth is bounded by metadata.
        BOUNDED_STATE_RECURRENCE = "bounded_state_recurrence";
        /// Droppable state that cannot affect semantic output.
        ADVISORY_STATE = "advisory_state";
        /// Runtime-visible adaptive speculative proposal sizing.
        ADAPTIVE_PROPOSAL_BUDGET = "adaptive_proposal_budget";
        /// Stateful grammar-guidance adapter execution.
        GRAMMAR_GUIDANCE_ADAPTER = "grammar_guidance_adapter";
        /// Stateful telemetry adapter execution.
        TELEMETRY_ADAPTER = "telemetry_adapter";
        /// Nested loop and branch execution.
        NESTED_CONTROL_FLOW = "nested_control_flow";
        /// Typed loop induction values.
        LOOP_INDUCTION_VALUES = "loop_induction_values";
        /// Typed workflow emission.
        TYPED_EMIT = "typed_emit";
        /// Ragged valid-prefix emission.
        EMIT_VALID_LENGTH = "emit_valid_length";
        /// Observable presence for an optional input.
        INPUT_PRESENCE = "input_presence";
        /// Planner-internal explicit transfer nodes.
        EXPLICIT_TRANSFER = "explicit_transfer";
        /// Graph-internal stateful token-context feature injection.
        TOKEN_CONTEXT = "token_context";
        /// Versioned workflow-native speculative proposal and verification.
        CANONICAL_SPECULATION = "canonical_speculation";
    }
}

pub mod cache;
mod decoder_abi;
pub mod decoder_workflow;
mod graph_cardinality;
pub mod identity;
mod lowering;
pub mod parser;
pub mod schema;
pub mod session_state;
mod state_plan;
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
    StateIdentity, StateLifecycle, StateReader, StateSnapshotParticipation, StateSource,
    StateTransactionParticipation, StateUpdateRelation, StateWriter, resolve_state_plan,
    validate_state_plan,
};
pub use validation::{
    CapabilityReport, PipelineValidationError, RuntimeCapabilities, derived_capabilities, validate,
    validate_metadata, validate_pipeline_spec, validate_structure_and_capabilities,
};
pub use version::{
    BATCHING_SCHEMA_VERSION, CANONICAL_SPECULATION_SCHEMA_VERSION, INITIAL_SCHEMA_VERSION,
    OUTPUT_PROTOCOL_SCHEMA_VERSION, SUPPORTED_SCHEMA_VERSION, SchemaVersion,
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
    #[error("Unsupported capabilities: {0:?}")]
    Unsupported(Vec<String>),
}
