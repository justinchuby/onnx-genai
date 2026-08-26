//! Typed structs for all inference metadata spec sections.

use std::collections::BTreeMap;

use schemars::JsonSchema;
use serde::{Deserialize, Deserializer, Serialize};

fn deserialize_non_empty_string<'de, D>(deserializer: D) -> Result<String, D::Error>
where
    D: Deserializer<'de>,
{
    let value = String::deserialize(deserializer)?;
    if value.is_empty() {
        return Err(serde::de::Error::custom("presence keys must not be empty"));
    }
    Ok(value)
}

mod decoder_abi;
mod generation;
mod hardware;
mod ir;
mod package;
mod pipeline;

pub use schema_vocabulary::{LENGTH_CONTENT_ROLES, PACK_OFFSETS_CONTENT, PACK_OWNER_CONTENT};

pub use decoder_abi::*;
pub use generation::*;
pub use hardware::*;
pub use ir::*;
pub use package::*;
pub use pipeline::*;

/// ONNX inference metadata consumed by runtimes and emitted by model builders.
///
/// Every top-level section is optional for incremental adoption. The v1 surface
/// is closed so removed scheduling and model-family fields fail fast.
#[derive(Debug, Clone, Default, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
#[schemars(
    deny_unknown_fields,
    title = "ONNX Inference Metadata",
    description = "Portable, runtime-agnostic inference metadata for ONNX generative models. The v1 top-level surface is closed; executable composite packages use pipeline.workflow.",
    extend("$id" = "https://github.com/onnx/onnx/issues/8184"),
    transform = schema_helpers::inference_metadata_constraints
)]
pub struct InferenceMetadata {
    /// Schema version of this inference-metadata document, e.g. `"v1.1"`.
    ///
    /// Canonically `"v<major>.<minor>"`, matching the `v1` that every writer in
    /// this repository already stamps. Absence means the initial `v1.0`
    /// contract, as do the spellings `"v1"`, `"1"`, `"1.0"` and `"v1.0"` that
    /// predate a canonical form; readers normalize before comparing, so all of
    /// them keep loading and `"1.1"` is an accepted synonym for `"v1.1"`. A
    /// document that uses nothing new keeps stamping what it already stamped —
    /// rewriting an existing spelling would change the package's semantic
    /// identity to say something no reader distinguishes. Bump the major only
    /// for breaking schema changes, and the minor whenever a document may carry
    /// a field an older reader would not know.
    ///
    /// An additive field keeps the same major version, but note what that does
    /// and does not buy: the v1 surface is closed — every structure here denies
    /// unknown fields — so a reader built before a field was added rejects a
    /// document that uses it rather than ignoring it. Compatibility with older
    /// readers therefore comes from a new field being *absent by default*, not
    /// from tolerance of unknown keys. A producer that emits one is declaring
    /// that the package needs a reader which understands it, and a package that
    /// does not emit it keeps loading everywhere it loaded before.
    ///
    /// That is a statement about *additive* fields, and it is not a promise that
    /// this schema never reshapes one it already had. It does, while it is
    /// pre-release: `token_packed` moved its `offsets` and `owner` into a
    /// `levels` chain in v1.1, and a document written against the flat spelling
    /// does not load at any version. A version says which fields a reader must
    /// understand; it is not a compatibility guarantee spanning a reshape, and
    /// nothing here silently reads an old spelling as a new one. See
    /// `reject_flat_token_packed`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub schema_version: Option<String>,

    /// Numeric model and control-token facts.
    ///
    /// Token strings and chat templates remain in tokenizer assets. Request
    /// EOS inputs may override these defaults, but must not repeat them as
    /// authored workflow literals.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tokens: Option<TokenFacts>,

    /// Capability identifiers that a runtime MUST support or refuse to load the model.
    #[serde(default)]
    #[schemars(
        extend("examples" = [["kv_cache", "grouped_query_attention"]]),
        inner(length(min = 1))
    )]
    pub required_capabilities: Vec<String>,

    /// Build-time model properties and runtime-configurable capabilities.
    #[serde(default)]
    pub model: Option<ModelCapabilities>,

    /// Model weight quantization intent, independent of the packed representation.
    #[serde(default)]
    pub quantization: Option<QuantizationIntent>,

    /// Declarative multi-model pipeline and its dataflow graph.
    #[serde(default)]
    pub pipeline: Option<PipelineSpec>,

    /// Runtime-managed LoRA adapters for bare or composite model packages.
    ///
    /// This is the migrated `InferenceMetadata.adapters` contract from native
    /// LoRA phases 1 and 2. Composite execution references workflow SSA inputs,
    /// but artifact identity, target resolution, and lifecycle remain package
    /// metadata rather than workflow control-flow nodes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub adapters: Option<AdapterServiceContract>,

    /// Minimum and beneficial hardware capabilities used for distribution matching.
    #[serde(default)]
    pub hardware_requirements: Option<HardwareRequirements>,

    /// Declared, architecture-neutral input preprocessing programs.
    ///
    /// Carries the typed multimodal preprocessing contracts: the still-image,
    /// video, and audio transform programs and their named tensor outputs.
    /// Every operation and output is generic, parameterized data — never a
    /// model family, vendor string, or baked-in shape. Absent means the model
    /// declares no native preprocessing program and a runtime must obtain it
    /// elsewhere or fail.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub preprocessing: Option<PreprocessingSpec>,

    /// Exact package facts needed to interpret request data correctly.
    ///
    /// Tokenizer artifacts, vocabulary size, special tokens, and the constraint
    /// dialects the package's parser accepts. Grammars and JSON Schemas
    /// themselves are request data, not package metadata.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub package: Option<PackageFacts>,

    /// Authoritative generation defaults and the structural override surface.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub generation: Option<GenerationContract>,

    /// Executable task profiles sharing this package's common facts.
    ///
    /// Every profile carries its own version and requirement class. A strict
    /// reader may skip an `ignorable` profile it does not understand; unknown
    /// core fields still fail.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub profiles: BTreeMap<String, TaskProfile>,

    /// Portable speculative-decoding compatibility facts.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speculative: Option<SpeculativeContract>,

    /// Decode-step ABI recognized from the canonical workflow, resolved once on
    /// first use.
    ///
    /// This is a cache of a pure function of `pipeline.workflow`, never part of
    /// the document. Keeping it derived rather than serialized is what makes
    /// the workflow the single authority: there is no second place a producer
    /// could write an answer that disagrees.
    #[serde(skip)]
    #[schemars(skip)]
    decoder_io: std::sync::OnceLock<Option<DecoderAbi>>,
}

impl InferenceMetadata {
    /// The resolved decode-step graph ABI for this package's decoder.
    ///
    /// Derived from `pipeline.workflow` and nowhere else. There is no second
    /// place a producer can state a different answer, because there is no
    /// second place a producer can state one at all.
    pub fn decoder_io(&self) -> Option<&DecoderAbi> {
        self.decoder_io
            .get_or_init(|| {
                let workflow = &self.pipeline.as_ref()?.workflow;
                let component = crate::graph_cardinality::sole_decoder_component(workflow)?;
                crate::decoder_abi::decoder_abi(workflow, component)
            })
            .as_ref()
    }

    /// Install a decode ABI a loader recovered from the graph itself.
    ///
    /// A package imported from the foreign `genai_config.json` format carries no
    /// workflow of ours, so its importer builds one and resolves the ABI from
    /// it. That result is derived exactly like a recognized one, so it belongs
    /// in the resolved slot.
    ///
    /// Replaces any previously resolved ABI, which is why it takes `&mut self`:
    /// a loader derives an ABI before the package is shared.
    pub fn set_derived_decoder_io(&mut self, io: DecoderAbi) {
        self.decoder_io = std::sync::OnceLock::from(Some(io));
    }
}

mod schema_vocabulary {
    use schemars::JsonSchema;

    macro_rules! extensible_string {
        (
            $(#[$meta:meta])*
            $name:ident,
            $transform:ident,
            $values:ident,
            [$($value:literal),+ $(,)?]
        ) => {
            $(#[$meta])*
            #[derive(JsonSchema)]
            #[schemars(with = "String", transform = super::schema_helpers::$transform)]
            pub(super) struct $name;

            pub(super) const $values: &[&str] = &[$($value),+];
        };
    }

    extensible_string!(
        /// Attention architecture vocabulary with an extension branch.
        AttentionType,
        attention_type,
        ATTENTION_TYPE,
        [
            "multi_head",
            "multi_head_attention",
            "grouped_query",
            "group_query_attention",
            "grouped_query_attention",
            "gqa",
            "multi_latent",
            "multi_latent_attention",
            "mla"
        ]
    );

    extensible_string!(
        /// Scalar dtype vocabulary with common ONNX and runtime aliases.
        DType,
        dtype,
        DTYPE,
        [
            "float32",
            "fp32",
            "float16",
            "fp16",
            "half",
            "bfloat16",
            "bf16",
            "float8_e4m3fn",
            "fp8_e4m3fn",
            "float8_e4m3",
            "fp8_e4m3",
            "float8_e5m2",
            "fp8_e5m2",
            "int8",
            "uint8",
            "int4",
            "uint4"
        ]
    );

    extensible_string!(
        /// Tensor-boundary dtype vocabulary, including non-numeric pipeline values.
        TensorDType,
        tensor_dtype,
        TENSOR_DTYPE,
        [
            "float32",
            "fp32",
            "float16",
            "fp16",
            "bfloat16",
            "bf16",
            "float8_e4m3fn",
            "float8_e5m2",
            "int64",
            "int32",
            "int8",
            "uint8",
            "bool",
            "string"
        ]
    );

    extensible_string!(
        /// Weight precision and quantization-recipe vocabulary.
        Precision,
        precision,
        PRECISION,
        [
            "float32",
            "fp32",
            "float16",
            "fp16",
            "bfloat16",
            "bf16",
            "float8_e4m3fn",
            "float8_e5m2",
            "int8",
            "int4",
            "int4_group128"
        ]
    );

    extensible_string!(
        /// Generic pixel transform-operation vocabulary.
        ///
        /// One vocabulary spans stills and clips. A still-image program never
        /// selects frames; a clip program samples and pads them and then does
        /// the same spatial work. `sample_frames` and `pad_frames` are the two
        /// operations a temporal axis adds, not a second program dialect.
        VisionTransformOp,
        vision_transform_op,
        VISION_TRANSFORM_OP,
        [
            "decode",
            "decode_rgb",
            "convert_rgb",
            "sample_frames",
            "resize",
            "rescale",
            "normalize",
            "tile",
            "flatten",
            "patchify",
            "pad",
            "pad_frames",
            "emit_original_size",
            "emit_transformed_size",
            "emit_validity_mask",
            "emit_patch_coordinates",
            "emit_grid_coordinates"
        ]
    );

    extensible_string!(
        /// Generic pixel-output content-role vocabulary.
        ///
        /// A program that packs the media of several requests into one batch
        /// emits the values that map packed entries back to request rows,
        /// alongside the pixels themselves. Two roles cover every level:
        /// `pack_offsets` and `pack_owner`. Which level a value serves is
        /// stated by the ownership chain that references it, not by its role,
        /// so a frames-to-clips pair and a clips-to-rows pair carry the same
        /// two roles and a runtime resolves both the same way whatever the
        /// medium calls its levels. `valid_lengths` says how much of a padded
        /// dimension is real, and is shared with the audio vocabulary rather
        /// than duplicated.
        ///
        /// They are content roles like any other: the runtime reads what a
        /// tensor means from here, never from the name a producer chose. What
        /// each value must structurally satisfy is stated by its tensor
        /// contract — `batch_layout` and `padding` — which is what the
        /// validator enforces.
        VisionOutputContent,
        vision_output_content,
        VISION_OUTPUT_CONTENT,
        [
            "pixels",
            "patch_coordinates",
            "grid_dimensions",
            "original_size",
            "transformed_size",
            "validity_mask",
            "valid_lengths",
            "pack_offsets",
            "pack_owner"
        ]
    );

    /// Content roles that state how much of a padded dimension is real.
    ///
    /// `valid_lengths` is the medium-independent spelling, added because a
    /// padded video, a padded image grid, and a padded text segment all need one
    /// and none of them is audio. The two audio names that predate it say the
    /// same thing about a particular unit and remain correct, so a program that
    /// already emits `frame_lengths` points its `padding` entry at that value
    /// rather than emitting a second copy under a new name.
    ///
    /// This list is *informative*: a `padding` entry names its length value by
    /// name, and nothing dispatches on which of these spellings the value
    /// carries. Deciding validity by role would refuse correct audio programs
    /// and would put two statements where the reference already makes one.
    /// `valid_frames` and `valid_samples` are deliberately absent — those are
    /// counts a graph consumes as ordinary inputs, not statements about padding.
    pub const LENGTH_CONTENT_ROLES: [&str; 3] =
        ["valid_lengths", "sample_lengths", "frame_lengths"];

    /// Content role of the exclusive prefix offsets of an ownership level.
    pub const PACK_OFFSETS_CONTENT: &str = "pack_offsets";

    /// Content role of the owner map of an ownership level.
    pub const PACK_OWNER_CONTENT: &str = "pack_owner";

    extensible_string!(
        /// Optional-thumbnail ordering vocabulary.
        ThumbnailOrder,
        thumbnail_order,
        THUMBNAIL_ORDER,
        ["none", "prepend", "append"]
    );

    extensible_string!(
        /// Generic audio transform-operation vocabulary.
        ///
        /// One vocabulary spans every declared audio program. A CTC acoustic
        /// model normalizes raw samples and never builds a spectrogram; a
        /// speech-to-text encoder pads to a fixed window and takes a log-mel.
        /// Both are the same kind of declaration, so both draw their operation
        /// names from here rather than from a per-family list.
        AudioTransformOp,
        audio_transform_op,
        AUDIO_TRANSFORM_OP,
        [
            "decode",
            "resample",
            "downmix",
            "rescale",
            "zero_mean_unit_variance",
            "normalize",
            "pad",
            "trim",
            "frame",
            "spectrogram",
            "log_mel",
            "log_mel_spectrogram",
            "emit_valid_frames",
            "emit_valid_samples",
            "emit_sample_lengths",
            "emit_validity_mask"
        ]
    );

    extensible_string!(
        /// Generic audio-output content-role vocabulary.
        ///
        /// Audio arrived at lengths first and named them for what they count:
        /// `sample_lengths` counts raw samples, `frame_lengths` counts frames
        /// after framing, and `valid_frames`/`valid_samples` are the same fact
        /// spelled as a count of the real entries. They stay, because a program
        /// that already emits one keeps loading and because the finer name is
        /// worth reading. `valid_lengths` joins them as the medium-independent
        /// spelling, and all five are one role to the validator: the value a
        /// `padding` entry names must carry one of them, whichever the producer
        /// chose. See [`LENGTH_CONTENT_ROLES`].
        AudioOutputContent,
        audio_output_content,
        AUDIO_OUTPUT_CONTENT,
        [
            "waveform",
            "features",
            "audio_features",
            "valid_frames",
            "valid_samples",
            "sample_lengths",
            "frame_lengths",
            "valid_lengths",
            "validity_mask",
            "pack_offsets",
            "pack_owner"
        ]
    );

    extensible_string!(
        /// Frame-synchronous sequence-decoding algorithm vocabulary.
        SequenceDecodingKind,
        sequence_decoding_kind,
        SEQUENCE_DECODING_KIND,
        ["ctc", "greedy_argmax"]
    );

    extensible_string!(
        /// Class-id -> string mapping source vocabulary.
        DecodingVocabularySource,
        decoding_vocabulary_source,
        DECODING_VOCABULARY_SOURCE,
        ["tokenizer", "inline"]
    );

    extensible_string!(
        /// Loop-carried state initialization vocabulary.
        StateInitKind,
        state_init_kind,
        STATE_INIT_KIND,
        ["zeros"]
    );

    extensible_string!(
        /// Loop-carried state update-semantics vocabulary.
        StateUpdateKind,
        state_update_kind,
        STATE_UPDATE_KIND,
        ["replace"]
    );

    extensible_string!(
        /// Sparse expert graph representation vocabulary.
        MoERepresentation,
        moe_representation,
        MOE_REPRESENTATION,
        ["dense_fallback", "moe", "qmoe"]
    );

    extensible_string!(
        /// Router score-operation vocabulary.
        MoERouterScoreFunction,
        moe_router_score_function,
        MOE_ROUTER_SCORE_FUNCTION,
        ["softmax", "sigmoid"]
    );

    extensible_string!(
        /// Router expert-selection vocabulary.
        MoERouterSelectionMethod,
        moe_router_selection_method,
        MOE_ROUTER_SELECTION_METHOD,
        ["top_k", "grouped_top_k", "sparse_mixer"]
    );

    extensible_string!(
        /// Group-scoring reduction vocabulary.
        MoEGroupScore,
        moe_group_score,
        MOE_GROUP_SCORE,
        ["maximum", "top_2_sum"]
    );
}

mod schema_helpers {
    use schemars::Schema;
    use serde_json::json;

    /// Publish the retired `model.io` block's rejection in the JSON schema.
    ///
    /// `ModelCapabilities` does not deny unknown fields, so a validator working
    /// from the schema alone would silently accept `model.io` and a producer
    /// would not learn it had published an unloadable package until a runtime
    /// refused it. The constraint therefore forbids the key outright — the same
    /// rule `parser::reject_retired_model_io` enforces, so the published schema
    /// and the loader cannot disagree about what is valid.
    ///
    /// An earlier revision forbade `model.io` only *beside* a `pipeline`, from
    /// when the block was still an import-only source. It is not a source at
    /// all now, so the narrower rule would accept exactly the documents the
    /// loader rejects.
    pub(super) fn inference_metadata_constraints(schema: &mut Schema) {
        schema
            .ensure_object()
            .entry("allOf")
            .or_insert_with(|| json!([]))
            .as_array_mut()
            .expect("allOf inserted as an array")
            .push(json!({
                "not": {
                    "required": ["model"],
                    "properties": {
                        "model": {
                            "required": ["io"]
                        }
                    }
                }
            }));
    }

    pub(super) fn loop_state_pair(schema: &mut Schema) {
        let required = schema
            .ensure_object()
            .entry("required")
            .or_insert_with(|| json!([]))
            .as_array_mut()
            .expect("required inserted as an array");
        for property in ["init", "update"] {
            if !required.iter().any(|name| name == property) {
                required.push(json!(property));
            }
        }
    }

    pub(super) fn proposal_type(schema: &mut Schema) {
        extensible_string_enum(
            schema,
            &[
                "eagle",
                "eagle3",
                "eagle-3",
                "peagle",
                "p-eagle",
                "mtp",
                "dflash",
                "d-flash",
                "shared_kv",
                "shared-kv",
            ],
        );
    }

    pub(super) fn attention_type(schema: &mut Schema) {
        extensible_string_enum(schema, super::schema_vocabulary::ATTENTION_TYPE);
    }

    pub(super) fn dtype(schema: &mut Schema) {
        extensible_string_enum(schema, super::schema_vocabulary::DTYPE);
    }

    pub(super) fn tensor_dtype(schema: &mut Schema) {
        extensible_string_enum(schema, super::schema_vocabulary::TENSOR_DTYPE);
    }

    pub(super) fn precision(schema: &mut Schema) {
        extensible_string_enum(schema, super::schema_vocabulary::PRECISION);
    }

    pub(super) fn vision_transform_op(schema: &mut Schema) {
        extensible_string_enum(schema, super::schema_vocabulary::VISION_TRANSFORM_OP);
    }

    pub(super) fn vision_output_content(schema: &mut Schema) {
        extensible_string_enum(schema, super::schema_vocabulary::VISION_OUTPUT_CONTENT);
    }

    pub(super) fn audio_transform_op(schema: &mut Schema) {
        extensible_string_enum(schema, super::schema_vocabulary::AUDIO_TRANSFORM_OP);
    }

    pub(super) fn audio_output_content(schema: &mut Schema) {
        extensible_string_enum(schema, super::schema_vocabulary::AUDIO_OUTPUT_CONTENT);
    }

    pub(super) fn thumbnail_order(schema: &mut Schema) {
        extensible_string_enum(schema, super::schema_vocabulary::THUMBNAIL_ORDER);
    }

    pub(super) fn sequence_decoding_kind(schema: &mut Schema) {
        extensible_string_enum(schema, super::schema_vocabulary::SEQUENCE_DECODING_KIND);
    }

    pub(super) fn decoding_vocabulary_source(schema: &mut Schema) {
        extensible_string_enum(schema, super::schema_vocabulary::DECODING_VOCABULARY_SOURCE);
    }

    pub(super) fn state_init_kind(schema: &mut Schema) {
        extensible_string_enum(schema, super::schema_vocabulary::STATE_INIT_KIND);
    }

    pub(super) fn state_update_kind(schema: &mut Schema) {
        extensible_string_enum(schema, super::schema_vocabulary::STATE_UPDATE_KIND);
    }

    pub(super) fn moe_representation(schema: &mut Schema) {
        extensible_string_enum(schema, super::schema_vocabulary::MOE_REPRESENTATION);
    }

    pub(super) fn moe_router_score_function(schema: &mut Schema) {
        extensible_string_enum(schema, super::schema_vocabulary::MOE_ROUTER_SCORE_FUNCTION);
    }

    pub(super) fn moe_router_selection_method(schema: &mut Schema) {
        extensible_string_enum(
            schema,
            super::schema_vocabulary::MOE_ROUTER_SELECTION_METHOD,
        );
    }

    pub(super) fn moe_group_score(schema: &mut Schema) {
        extensible_string_enum(schema, super::schema_vocabulary::MOE_GROUP_SCORE);
    }

    pub(super) fn moe_router(schema: &mut Schema) {
        schema.ensure_object().insert(
            "allOf".into(),
            json!([{
                "if": {
                    "properties": {
                        "selection_method": {"const": "grouped_top_k"}
                    },
                    "required": ["selection_method"]
                },
                "then": {
                    "required": ["group_count", "groups_per_token", "group_score"]
                }
            }]),
        );
    }

    fn extensible_string_enum(schema: &mut Schema, known_values: &[&str]) {
        let known_values = json!(known_values);
        let object = schema.ensure_object();
        object.insert("type".into(), json!("string"));
        object.insert(
            "oneOf".into(),
            json!([
                {
                    "title": "Known standard value",
                    "enum": known_values.clone()
                },
                {
                    "title": "Forward-compatible extension value",
                    "type": "string",
                    "not": {"enum": known_values}
                }
            ]),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Deserialize, Serialize)]
    struct OptionalModalityDocument {
        io: DecoderAbi,
    }

    #[test]
    fn optional_modality_schema_round_trips() {
        let old_yaml = r#"
io:
  sequence_source: token_ids
"#;
        let old: OptionalModalityDocument =
            serde_yaml::from_str(old_yaml).expect("old metadata deserializes");
        assert!(old.io.optional_inputs.is_empty());
        assert_eq!(
            serde_yaml::to_value(&old).expect("old metadata serializes"),
            serde_yaml::from_str::<serde_yaml::Value>(old_yaml).expect("old YAML parses")
        );

        let new_yaml = r#"
io:
  optional_inputs:
    audio_features:
      presence: audio
      absent:
        kind: zeros
        shape: [0, sequence_len]
"#;
        let new: OptionalModalityDocument =
            serde_yaml::from_str(new_yaml).expect("optional-modality metadata deserializes");
        let optional = new
            .io
            .optional_inputs
            .get("audio_features")
            .expect("optional input is preserved");
        assert_eq!(optional.presence, "audio");
        assert_eq!(optional.absent.kind, AbsentInputKind::Zeros);
        assert_eq!(
            optional.absent.shape,
            [
                TensorDimension::Fixed(0),
                TensorDimension::Symbol("sequence_len".into())
            ]
        );
        assert_eq!(
            serde_yaml::to_value(&new).expect("optional-modality metadata serializes"),
            serde_yaml::from_str::<serde_yaml::Value>(new_yaml).expect("new YAML parses")
        );
        assert_eq!(
            serde_yaml::to_value(AbsentInputKind::Zeros).expect("kind serializes"),
            serde_yaml::Value::String("zeros".into())
        );

        assert!(
            serde_yaml::from_str::<TensorDimension>("-1").is_err(),
            "negative fixed dimensions must be rejected"
        );
        assert!(
            serde_yaml::from_str::<OptionalInputSpec>(
                "presence: ''\nabsent:\n  kind: zeros\n  shape: [0]\n"
            )
            .is_err(),
            "empty presence keys must be rejected"
        );
    }

    #[test]
    fn attention_config_parses_sliding_window_and_sink_tokens() {
        let yaml = r#"
attention:
  type: grouped_query
  sliding_window: 4096
  sink_tokens: 4
max_sequence_length: 131072
"#;
        let model: ModelCapabilities = serde_yaml::from_str(yaml).expect("parses");
        let attention = model.attention.expect("attention section");
        assert_eq!(attention.sliding_window, Some(4096));
        assert_eq!(attention.sink_tokens, Some(4));
    }

    #[test]
    fn attention_config_defaults_sink_tokens_to_none() {
        let yaml = r#"
attention:
  type: grouped_query
  sliding_window: 4096
"#;
        let model: ModelCapabilities = serde_yaml::from_str(yaml).expect("parses");
        let attention = model.attention.expect("attention section");
        assert_eq!(attention.sliding_window, Some(4096));
        assert_eq!(attention.sink_tokens, None);
    }
}
