//! ONNX inference metadata parser and types.
//!
//! Implements the spec from <https://github.com/onnx/onnx/issues/8184>

/// Current inference-metadata schema version. Emitters should stamp this into
/// [`schema::InferenceMetadata::schema_version`]; readers treat an absent value
/// as this version.
pub const SCHEMA_VERSION: &str = "v1";

/// Well-known capability identifiers for the typed multimodal contract.
///
/// A model package lists the identifiers it needs in
/// [`schema::InferenceMetadata::required_capabilities`]; a runtime that does not
/// advertise one of them fails the load through [`validation::validate`] with a
/// precise, actionable missing-capability error rather than guessing from the
/// model's identity. The strings are stable data, not runtime branches.
pub mod capabilities {
    /// A typed image preprocessing transform program is required.
    pub const IMAGE_PREPROCESSING_PROGRAM: &str = "image_preprocessing_program";
    /// The program emits more than one packed image tensor output.
    pub const PACKED_IMAGE_OUTPUTS: &str = "packed_image_outputs";
    /// A declared multi-axis position-id program is required.
    pub const POSITION_PROGRAM: &str = "position_program";
    /// Multi-axis (rank > 1) position coordinates are required.
    pub const MULTI_AXIS_POSITIONS: &str = "multi_axis_positions";
    /// Fixed-shape loop-carried recurrent state (replace semantics) is required.
    pub const LOOP_CARRIED_STATE: &str = "loop_carried_state";
    /// A decoder that consumes a raw token input and a routed sequence input
    /// simultaneously is required.
    pub const DUAL_SEQUENCE_INPUTS: &str = "dual_sequence_inputs";
}

pub mod component;
pub mod parser;
pub mod schema;
pub mod validation;

pub use component::{
    ComponentDataType, ComponentError, ComponentIo, ComponentSession, ComponentTensor,
};
pub use parser::{
    MtpProposerSpec, SharedKvProposerSpec, SpeculatorConfigSource, SpeculatorDescriptor,
    SpeculatorProposerKind, SpeculatorProposerStatus, detect_speculator, load_metadata,
    load_pipeline_spec, resolve_speculator_config,
};
pub use schema::*;
pub use validation::{
    PipelineValidationError, RuntimeCapabilities, validate, validate_pipeline_spec,
};

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
