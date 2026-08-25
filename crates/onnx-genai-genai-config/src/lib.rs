//! Compatibility layer that converts an onnxruntime-genai `genai_config.json`
//! into the native onnx-genai [`InferenceMetadata`] spec.
//!
//! onnx-genai's own `inference_metadata.yaml` remains the preferred, canonical
//! source of truth. This crate exists purely as an *auto-detection fallback*:
//! the many ORT-genai / Foundry Local models in the wild ship only a
//! `genai_config.json` and no `inference_metadata.yaml`, yet they carry the same
//! information the runtime needs.
//!
//! This converter performs a COMPLETE one-way conversion of the pieces of
//! `genai_config.json` that map cleanly onto the native spec:
//!
//! * the decoder graph I/O ports (`io` block), including `%d`-expanded KV cache
//!   input/output name lists,
//! * generation / search defaults (`generation`),
//! * special token ids (`tokens`),
//! * attention dimensions, max sequence length, vocab size, and the shared-KV
//!   buffer hint (`model.*` + `kv_cache.native_dtype`), and
//! * multi-model shapes — multimodal (embedding + vision/speech), encoder-decoder
//!   (ASR / whisper), and split decoder-pipelines — emitted as a `pipeline`.
//!
//! Shapes that the native spec cannot yet represent are ignored rather than
//! failing, so loading stays forward-compatible. See the `NOTE:` in
//! [`GenAiConfig::to_inference_metadata`] for the specific fields skipped.
//!
//! The KV native dtype (which lives in the ONNX graph, not in
//! `genai_config.json`) is passed in by the caller, so this crate only depends
//! on the metadata and preprocessing crates — never on `onnx-genai-ort`.

mod compatibility;
mod graph_io;
mod import;
mod json_builders;
mod loading;
mod wire_types;

pub(crate) use compatibility::incomplete;
pub use compatibility::{DerivedDecoderIo, DerivedStatePair};
pub(crate) use graph_io::*;
pub use graph_io::{GraphTensorInfo, ModelGraphInfo};
pub use import::{
    CONSUMED_KEYS, ImportOptions, ImportReport, KNOWN_DROPPED_KEYS, drop_reason, import,
    import_from_dir, import_from_path, unrepresentable_keys,
};
pub(crate) use json_builders::*;
pub use loading::*;
pub use wire_types::*;

/// Canonical file name onnxruntime-genai uses for its model config.
pub const GENAI_CONFIG_FILE: &str = "genai_config.json";

// Conventional default tensor names (mirrors onnxruntime-genai `Config::Defaults`).
const DEFAULT_INPUT_IDS: &str = "input_ids";
const DEFAULT_LOGITS: &str = "logits";
const DEFAULT_PAST_KEY: &str = "past_key_values.%d.key";
const DEFAULT_PAST_VALUE: &str = "past_key_values.%d.value";
const DEFAULT_PRESENT_KEY: &str = "present.%d.key";
const DEFAULT_PRESENT_VALUE: &str = "present.%d.value";
const DEFAULT_ENCODER_HIDDEN_STATES: &str = "encoder_hidden_states";

/// Errors produced while locating, reading, or parsing a `genai_config.json`.
#[derive(Debug, thiserror::Error)]
pub enum GenAiConfigError {
    /// The file could not be read.
    #[error("failed to read genai_config.json: {0}")]
    Io(#[from] std::io::Error),
    /// The file was not valid JSON or did not match the expected shape.
    #[error("failed to parse genai_config.json: {0}")]
    Parse(#[from] serde_json::Error),
    /// The legacy config carries facts the new metadata contract does not.
    ///
    /// Import is one-way and fail-closed: dropping a key silently would let a
    /// package claim semantics its metadata no longer states. Pass
    /// `--allow-lossy` (`ImportOptions::allow_lossy`) to accept the loss and
    /// receive the dropped keys in the import report.
    #[error(
        "genai_config.json carries facts the inference-metadata contract does not represent: \
         {keys}. Why: import is one-way and fail-closed, so a dropped key never silently \
         changes what a package means. How to fix: re-export the package with a native \
         inference_metadata.yaml that declares these facts, or re-run the import with \
         --allow-lossy to accept and record the loss"
    )]
    LossyImport {
        /// Dropped key paths, each with its reason when one is recorded.
        keys: String,
    },
    /// A compatibility package omitted semantics needed by the typed pipeline.
    #[error(
        "cannot synthesize compatibility pipeline metadata: missing required semantics: {missing}. \
         Why: compatibility loading is allowed only when genai_config.json, config.json, \
         processor config, and ONNX graph interfaces explicitly describe every pipeline, \
         preprocessing, position, KV-cache, and fixed-state behavior; the loader never guesses \
         from model.type or a model name. How to fix: regenerate the package with native \
         inference_metadata.json (preferred), or export a complete compatibility package that \
         declares the missing facts"
    )]
    IncompletePipeline {
        /// Missing or inconsistent semantic facts.
        missing: String,
    },
    /// The package describes a valid model family that the current
    /// inference-metadata contract cannot execute (e.g. an RNN-T transducer).
    /// Declined honestly rather than mis-bound as a supported shape.
    #[error(
        "unsupported pipeline family: {family}. {reason}. \
         Why: this family is structurally distinct from every executable shape \
         (single-decoder, multimodal, encoder-decoder, decoder-pipeline) and the \
         loader will not fabricate bindings it cannot honor. How to fix: add native \
         support for this family, or supply a native inference_metadata.json that \
         declares an executable pipeline"
    )]
    UnsupportedPipelineFamily {
        /// Human-readable family name (e.g. `"RNN-T transducer"`).
        family: String,
        /// What makes it unexecutable and what it would take to support it.
        reason: String,
    },
    /// A multimodal package declares an image-preprocessing program that the
    /// runtime cannot represent losslessly (e.g. Qwen-style `smart_resize`,
    /// which has no faithful stretch/crop/pad equivalent). The vision path is
    /// therefore unusable, but the same package can still drive a text-only
    /// decode pipeline (embedding + decoder). Callers that want text decode
    /// should fall back to the text-only synthesis; callers that require the
    /// image path must treat this as fatal. This is distinct from
    /// [`GenAiConfigError::IncompletePipeline`], which signals genuinely
    /// missing facts rather than a representable-but-unsupported transform.
    #[error(
        "image preprocessing is not representable by the runtime: {detail}. \
         Why: the declared vision preprocessing has no lossless runtime encoding, so \
         substituting an approximation would silently corrupt image inputs. How to fix: \
         run this package for text-only decode (the embedding+decoder path is unaffected), \
         or supply a native inference_metadata.json that declares a representable \
         preprocessing program for the image path"
    )]
    UnrepresentablePreprocessing {
        /// What preprocessing behavior could not be represented.
        detail: String,
    },
}

#[cfg(test)]
mod tests;
