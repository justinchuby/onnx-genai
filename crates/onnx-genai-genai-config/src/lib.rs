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
//! on `serde`/`serde_json` and the metadata spec — never on `onnx-genai-ort`.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use onnx_genai_metadata::{InferenceMetadata, SCHEMA_VERSION};
use serde::Deserialize;
use serde_json::{Map, Value, json};

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
}

/// Forward-compatible view of an onnxruntime-genai `genai_config.json`.
///
/// Unknown fields are ignored so future ORT-genai additions do not break loading.
#[derive(Debug, Clone, Deserialize)]
pub struct GenAiConfig {
    /// The `model` section.
    pub model: GenAiModel,
    /// The `search` section (generation defaults, incl. share-buffer hint).
    #[serde(default)]
    pub search: GenAiSearch,
}

/// The `model` section of `genai_config.json`.
#[derive(Debug, Clone, Deserialize)]
pub struct GenAiModel {
    /// Architecture identifier (e.g. `"qwen2"`, `"whisper"`, `"decoder-pipeline"`).
    #[serde(rename = "type", default)]
    pub model_type: Option<String>,
    /// Maximum total context length in tokens.
    #[serde(default)]
    pub context_length: Option<usize>,
    /// Vocabulary size.
    #[serde(default)]
    pub vocab_size: Option<usize>,

    // Special / control token ids.
    #[serde(default)]
    pub pad_token_id: Option<i64>,
    #[serde(default)]
    pub bos_token_id: Option<i64>,
    #[serde(default)]
    pub eos_token_id: Option<EosTokenId>,
    #[serde(default)]
    pub sep_token_id: Option<i64>,
    #[serde(default)]
    pub decoder_start_token_id: Option<i64>,
    #[serde(default)]
    pub image_token_id: Option<i64>,
    #[serde(default)]
    pub video_token_id: Option<i64>,
    #[serde(default)]
    pub vision_start_token_id: Option<i64>,

    /// Decoder graph properties (required).
    pub decoder: GenAiDecoder,
    /// Optional encoder graph (encoder-decoder / ASR models).
    #[serde(default)]
    pub encoder: Option<GenAiEncoder>,
    /// Optional embedding graph (multimodal models).
    #[serde(default)]
    pub embedding: Option<GenAiEmbedding>,
    /// Optional vision graph (VLMs).
    #[serde(default)]
    pub vision: Option<GenAiVision>,
    /// Optional speech / audio-embedding graph.
    #[serde(default)]
    pub speech: Option<GenAiSpeech>,
}

/// `eos_token_id` accepts either a scalar or an array; both normalize to a list.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum EosTokenId {
    /// A single end-of-stream token id.
    Single(i64),
    /// Several end-of-stream token ids.
    Many(Vec<i64>),
}

impl EosTokenId {
    fn to_vec(&self) -> Vec<i64> {
        match self {
            EosTokenId::Single(v) => vec![*v],
            EosTokenId::Many(v) => v.clone(),
        }
    }
}

/// The `model.decoder` section of `genai_config.json`.
#[derive(Debug, Clone, Deserialize)]
pub struct GenAiDecoder {
    /// ONNX filename for the (unsplit) decoder graph.
    #[serde(default)]
    pub filename: Option<String>,
    /// Per-head hidden dimension.
    #[serde(default)]
    pub head_size: Option<usize>,
    /// Number of query/attention heads.
    #[serde(default)]
    pub num_attention_heads: Option<usize>,
    /// Number of key/value heads (< attention heads implies GQA).
    #[serde(default)]
    pub num_key_value_heads: Option<usize>,
    /// Number of decoder layers.
    #[serde(default)]
    pub num_hidden_layers: Option<usize>,
    /// Graph input port names.
    #[serde(default)]
    pub inputs: DecoderInputs,
    /// Graph output port names.
    #[serde(default)]
    pub outputs: DecoderOutputs,
    /// Split decoder-pipeline stages (`decoder-pipeline` models).
    #[serde(default)]
    pub pipeline: Vec<BTreeMap<String, PipelineStageModel>>,
}

/// Decoder graph input port names (values are graph tensor names).
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct DecoderInputs {
    pub input_ids: Option<String>,
    pub inputs_embeds: Option<String>,
    pub attention_mask: Option<String>,
    pub position_ids: Option<String>,
    pub past_key_names: Option<String>,
    pub past_value_names: Option<String>,
    /// Combined key/value KV input pattern (when key/value are one tensor).
    pub past_names: Option<String>,
    pub cross_past_key_names: Option<String>,
    pub cross_past_value_names: Option<String>,
    pub encoder_hidden_states: Option<String>,
}

/// Decoder graph output port names (values are graph tensor names).
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct DecoderOutputs {
    pub logits: Option<String>,
    pub present_key_names: Option<String>,
    pub present_value_names: Option<String>,
    /// Combined key/value KV output pattern.
    pub present_names: Option<String>,
    pub output_cross_qk_names: Option<String>,
}

/// The `model.encoder` section (encoder-decoder / ASR models).
#[derive(Debug, Clone, Deserialize)]
pub struct GenAiEncoder {
    #[serde(default)]
    pub filename: Option<String>,
    #[serde(default)]
    pub num_attention_heads: Option<usize>,
    #[serde(default)]
    pub num_hidden_layers: Option<usize>,
    #[serde(default)]
    pub inputs: EncoderInputs,
    #[serde(default)]
    pub outputs: EncoderOutputs,
}

/// Encoder graph input port names.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct EncoderInputs {
    pub input_ids: Option<String>,
    pub audio_features: Option<String>,
    pub attention_mask: Option<String>,
}

/// Encoder graph output port names.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct EncoderOutputs {
    pub encoder_hidden_states: Option<String>,
    pub cross_present_key_names: Option<String>,
    pub cross_present_value_names: Option<String>,
}

/// The `model.embedding` section (multimodal token embedder).
#[derive(Debug, Clone, Deserialize)]
pub struct GenAiEmbedding {
    #[serde(default)]
    pub filename: Option<String>,
    #[serde(default)]
    pub inputs: EmbeddingInputs,
    #[serde(default)]
    pub outputs: EmbeddingOutputs,
}

/// Embedding graph input port names.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct EmbeddingInputs {
    pub input_ids: Option<String>,
    pub image_features: Option<String>,
    pub audio_features: Option<String>,
}

/// Embedding graph output port names.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct EmbeddingOutputs {
    pub inputs_embeds: Option<String>,
}

/// The `model.vision` section (VLM image encoder).
#[derive(Debug, Clone, Deserialize)]
pub struct GenAiVision {
    #[serde(default)]
    pub filename: Option<String>,
    #[serde(default)]
    pub inputs: VisionInputs,
    #[serde(default)]
    pub outputs: VisionOutputs,
}

/// Vision graph input port names.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct VisionInputs {
    pub pixel_values: Option<String>,
    pub image_sizes: Option<String>,
    pub image_grid_thw: Option<String>,
}

/// Vision graph output port names.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct VisionOutputs {
    pub image_features: Option<String>,
}

/// The `model.speech` section (audio embedder).
#[derive(Debug, Clone, Deserialize)]
pub struct GenAiSpeech {
    #[serde(default)]
    pub filename: Option<String>,
    #[serde(default)]
    pub inputs: SpeechInputs,
    #[serde(default)]
    pub outputs: SpeechOutputs,
}

/// Speech graph input port names.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct SpeechInputs {
    pub audio_embeds: Option<String>,
    pub attention_mask: Option<String>,
}

/// Speech graph output port names.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct SpeechOutputs {
    pub audio_features: Option<String>,
}

/// One split model inside `decoder.pipeline[]`.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct PipelineStageModel {
    pub filename: Option<String>,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
}

/// The `search` section of `genai_config.json`.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct GenAiSearch {
    /// Whether the runtime may own a single shared, max-length KV buffer that is
    /// aliased `present.* -> past_key_values.*` across decode steps.
    #[serde(default)]
    pub past_present_share_buffer: bool,
    /// Maximum generated length declared by the model author.
    #[serde(default)]
    pub max_length: Option<usize>,
    #[serde(default)]
    pub do_sample: Option<bool>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_k: Option<usize>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub repetition_penalty: Option<f32>,
    #[serde(default)]
    pub num_beams: Option<usize>,
    #[serde(default)]
    pub num_return_sequences: Option<usize>,
    #[serde(default)]
    pub min_length: Option<usize>,
    #[serde(default)]
    pub length_penalty: Option<f32>,
    #[serde(default)]
    pub no_repeat_ngram_size: Option<usize>,
    #[serde(default)]
    pub diversity_penalty: Option<f32>,
    #[serde(default)]
    pub early_stopping: Option<bool>,
}

/// Coarse structural family a `genai_config.json` describes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ModelShape {
    /// A single, unsplit decoder graph.
    SingleDecoder,
    /// Embedding + vision/speech front-ends feeding a decoder (multimodal).
    Multimodal,
    /// Encoder + cross-attention decoder (ASR / whisper).
    EncoderDecoder,
    /// A single decoder split into an ordered set of sub-graphs.
    DecoderPipeline,
}

impl GenAiConfig {
    /// Whether the decoder uses grouped/multi-query attention (strictly fewer KV
    /// heads than attention heads).
    pub fn is_group_query_attention(&self) -> bool {
        matches!(
            (
                self.model.decoder.num_key_value_heads,
                self.model.decoder.num_attention_heads,
            ),
            (Some(kv), Some(attn)) if kv < attn
        )
    }

    /// Whether the decoder is served by the ONNX Runtime `GroupQueryAttention`
    /// op. The Microsoft ONNX exporter maps attention onto the GQA op whenever
    /// key/value heads are declared and do not exceed the query heads — this
    /// includes full multi-head attention (`kv == attn`), which is just GQA with
    /// group size 1. The GQA op supports `past_present_share_buffer` at any head
    /// ratio, so this (not the strict GQA-vs-MHA ratio) is the correct gate for
    /// the runtime-owned shared KV buffer path.
    pub fn uses_group_query_attention_op(&self) -> bool {
        matches!(
            (
                self.model.decoder.num_key_value_heads,
                self.model.decoder.num_attention_heads,
            ),
            (Some(kv), Some(attn)) if kv >= 1 && kv <= attn
        )
    }

    /// Maximum total sequence length usable to pre-size a shared KV buffer,
    /// preferring the explicit `context_length` then `search.max_length`.
    pub fn max_sequence_length(&self) -> Option<usize> {
        self.model.context_length.or(self.search.max_length)
    }

    /// Whether this model advertises the runtime-owned shared KV buffer path.
    pub fn shared_kv_buffer_supported(&self) -> bool {
        self.search.past_present_share_buffer
            && self.uses_group_query_attention_op()
            && self.max_sequence_length().is_some()
    }

    fn shape(&self) -> ModelShape {
        if self.model.encoder.is_some() {
            ModelShape::EncoderDecoder
        } else if self.model.embedding.is_some()
            || self.model.vision.is_some()
            || self.model.speech.is_some()
        {
            ModelShape::Multimodal
        } else if !self.model.decoder.pipeline.is_empty() {
            ModelShape::DecoderPipeline
        } else {
            ModelShape::SingleDecoder
        }
    }

    /// Convert into native [`InferenceMetadata`].
    ///
    /// `kv_native_dtype` is the KV cache scalar dtype read from the ONNX graph by
    /// the caller (e.g. `"float16"` / `"float32"`); it is not present in
    /// `genai_config.json`. The runtime-owned shared KV buffer path is enabled —
    /// by emitting `kv_cache.native_dtype` — only when the model declares
    /// `search.past_present_share_buffer`, uses the GQA op, has a known max
    /// sequence length, and a share-buffer-compatible KV dtype is provided.
    ///
    /// NOTE: shapes/tensors the native spec cannot yet represent are intentionally
    /// skipped (loading never fails on them): RNN-T joiner graphs, VAD, Conformer
    /// NeMo `cache_last_channel`/`cache_last_time` state, LSTM/RNN decoder states
    /// (`rnn_states`, `lstm_hidden_state`, `lstm_cell_state`), paged-attention
    /// `block_table`, beam `cache_indirection`, `output_cross_qk`, and the
    /// per-session `session_options`/`run_options`.
    pub fn to_inference_metadata(
        &self,
        kv_native_dtype: Option<&str>,
    ) -> Result<InferenceMetadata, GenAiConfigError> {
        let shape = self.shape();

        let mut model = Map::new();
        model.insert("attention".into(), self.attention_json());
        insert_usize(&mut model, "max_sequence_length", self.max_sequence_length());
        insert_usize(&mut model, "vocab_size", self.model.vocab_size);

        if shape == ModelShape::SingleDecoder {
            let io = self.decoder_io_json(false);
            if !io.is_empty() {
                model.insert("io".into(), Value::Object(io));
            }
        }

        let mut root = Map::new();
        root.insert("schema_version".into(), json!(SCHEMA_VERSION));
        root.insert("model".into(), Value::Object(model));

        match shape {
            ModelShape::SingleDecoder => {}
            ModelShape::EncoderDecoder => {
                root.insert("pipeline".into(), self.encoder_decoder_pipeline_json());
            }
            ModelShape::Multimodal => {
                root.insert("pipeline".into(), self.multimodal_pipeline_json());
            }
            ModelShape::DecoderPipeline => {
                root.insert("pipeline".into(), self.decoder_pipeline_json());
            }
        }

        if let Some(generation) = self.generation_json() {
            root.insert("generation".into(), generation);
        }
        if let Some(tokens) = self.tokens_json() {
            root.insert("tokens".into(), tokens);
        }

        if self.shared_kv_buffer_supported()
            && let Some(dtype) = kv_native_dtype
            && is_share_buffer_kv_dtype(dtype)
        {
            root.insert("kv_cache".into(), json!({ "native_dtype": dtype }));
        }

        Ok(serde_json::from_value(Value::Object(root))?)
    }

    fn attention_json(&self) -> Value {
        let mut attention = Map::new();
        attention.insert(
            "type".into(),
            json!(if self.uses_group_query_attention_op() {
                "group_query_attention"
            } else {
                "multi_head_attention"
            }),
        );
        insert_usize(
            &mut attention,
            "num_kv_heads",
            self.model.decoder.num_key_value_heads,
        );
        insert_usize(
            &mut attention,
            "num_attention_heads",
            self.model.decoder.num_attention_heads,
        );
        insert_usize(&mut attention, "head_dim", self.model.decoder.head_size);
        Value::Object(attention)
    }

    /// Build the decoder `io` block.
    ///
    /// KV `%d`-name patterns are expanded over `0..num_hidden_layers`. When key
    /// and value are separate patterns, the lists interleave `[key_i, value_i]`
    /// per layer; a combined `past_names`/`present_names` pattern yields one entry
    /// per layer. `kv_inputs` and `kv_outputs` are expanded with the same
    /// ordering so they pair positionally. Cross-attention KV (encoder-decoder)
    /// is expanded the same way into `cross_kv_inputs`/`cross_kv_outputs`.
    fn decoder_io_json(&self, include_cross: bool) -> Map<String, Value> {
        let dec = &self.model.decoder;
        let layers = dec.num_hidden_layers;
        let mut io = Map::new();

        // Token vs embeds input are mutually exclusive: an `inputs_embeds` port
        // means the graph is embeds-driven; otherwise it is token-driven.
        if let Some(embeds) = dec.inputs.inputs_embeds.as_deref() {
            io.insert("inputs_embeds_input".into(), json!(embeds));
        } else {
            io.insert(
                "token_input".into(),
                json!(dec.inputs.input_ids.as_deref().unwrap_or(DEFAULT_INPUT_IDS)),
            );
        }
        if let Some(mask) = dec.inputs.attention_mask.as_deref() {
            io.insert("attention_mask_input".into(), json!(mask));
        }
        if let Some(pos) = dec.inputs.position_ids.as_deref() {
            io.insert("position_ids_input".into(), json!(pos));
        }
        io.insert(
            "logits_output".into(),
            json!(dec.outputs.logits.as_deref().unwrap_or(DEFAULT_LOGITS)),
        );

        if let Some(kv_inputs) = expand_kv(
            dec.inputs.past_names.as_deref(),
            dec.inputs.past_key_names.as_deref(),
            dec.inputs.past_value_names.as_deref(),
            DEFAULT_PAST_KEY,
            DEFAULT_PAST_VALUE,
            layers,
        ) {
            io.insert("kv_inputs".into(), json!(kv_inputs));
        }
        if let Some(kv_outputs) = expand_kv(
            dec.outputs.present_names.as_deref(),
            dec.outputs.present_key_names.as_deref(),
            dec.outputs.present_value_names.as_deref(),
            DEFAULT_PRESENT_KEY,
            DEFAULT_PRESENT_VALUE,
            layers,
        ) {
            io.insert("kv_outputs".into(), json!(kv_outputs));
        }

        if include_cross {
            let cross_inputs = expand_cross_kv(
                dec.inputs.cross_past_key_names.as_deref(),
                dec.inputs.cross_past_value_names.as_deref(),
                layers,
            );
            let cross_outputs = self.model.encoder.as_ref().and_then(|enc| {
                expand_cross_kv(
                    enc.outputs.cross_present_key_names.as_deref(),
                    enc.outputs.cross_present_value_names.as_deref(),
                    layers,
                )
            });
            if cross_inputs.is_some() || cross_outputs.is_some() {
                io.insert(
                    "encoder_hidden_states_input".into(),
                    json!(
                        dec.inputs
                            .encoder_hidden_states
                            .as_deref()
                            .unwrap_or(DEFAULT_ENCODER_HIDDEN_STATES)
                    ),
                );
            }
            if let Some(cross_inputs) = cross_inputs {
                io.insert("cross_kv_inputs".into(), json!(cross_inputs));
            }
            if let Some(cross_outputs) = cross_outputs {
                io.insert("cross_kv_outputs".into(), json!(cross_outputs));
            }
        }

        io
    }

    fn multimodal_pipeline_json(&self) -> Value {
        let mut models = Map::new();
        let mut dataflow: Vec<Value> = Vec::new();
        let mut phases = Map::new();
        let mut prompt_encoder: Option<String> = None;

        if let Some(vision) = &self.model.vision {
            models.insert(
                "vision_encoder".into(),
                component_json(filename_or(&vision.filename, "vision.onnx"), "encoder", None),
            );
            phases.insert("vision_encoder".into(), run_on("prompt_only"));
            prompt_encoder.get_or_insert_with(|| "vision_encoder".into());
            if self.model.embedding.is_some() {
                let from = vision.outputs.image_features.as_deref().unwrap_or("image_features");
                let to = self
                    .model
                    .embedding
                    .as_ref()
                    .and_then(|e| e.inputs.image_features.as_deref())
                    .unwrap_or("image_features");
                dataflow.push(edge(&format!("vision_encoder.{from}"), &format!("embedding.{to}")));
            }
        }

        if let Some(speech) = &self.model.speech {
            models.insert(
                "audio_encoder".into(),
                component_json(filename_or(&speech.filename, "speech.onnx"), "encoder", None),
            );
            phases.insert("audio_encoder".into(), run_on("prompt_only"));
            prompt_encoder.get_or_insert_with(|| "audio_encoder".into());
            if self.model.embedding.is_some() {
                let from = speech.outputs.audio_features.as_deref().unwrap_or("audio_features");
                let to = self
                    .model
                    .embedding
                    .as_ref()
                    .and_then(|e| e.inputs.audio_features.as_deref())
                    .unwrap_or("audio_features");
                dataflow.push(edge(&format!("audio_encoder.{from}"), &format!("embedding.{to}")));
            }
        }

        if let Some(embedding) = &self.model.embedding {
            let mut io = Map::new();
            if let Some(input_ids) = embedding.inputs.input_ids.as_deref() {
                io.insert("token_input".into(), json!(input_ids));
            }
            let io = (!io.is_empty()).then_some(Value::Object(io));
            models.insert(
                "embedding".into(),
                component_json(filename_or(&embedding.filename, "embedding.onnx"), "embedding", io),
            );
            phases.insert("embedding".into(), run_on("every_step"));

            let from = embedding.outputs.inputs_embeds.as_deref().unwrap_or("inputs_embeds");
            let to = self
                .model
                .decoder
                .inputs
                .inputs_embeds
                .as_deref()
                .unwrap_or("inputs_embeds");
            dataflow.push(edge(&format!("embedding.{from}"), &format!("decoder.{to}")));
        }

        let decoder_io = self.decoder_io_json(false);
        let decoder_io = (!decoder_io.is_empty()).then_some(Value::Object(decoder_io));
        models.insert(
            "decoder".into(),
            component_json(filename_or(&self.model.decoder.filename, "decoder.onnx"), "decoder", decoder_io),
        );
        phases.insert("decoder".into(), run_on("every_step"));

        let strategy = composite_encode_decode(prompt_encoder.as_deref(), "decoder");

        let mut pipeline = Map::new();
        pipeline.insert("models".into(), Value::Object(models));
        pipeline.insert("dataflow".into(), Value::Array(dataflow));
        pipeline.insert("strategy".into(), strategy);
        pipeline.insert("phases".into(), Value::Object(phases));
        if let Some(image_token_id) = self.model.image_token_id {
            pipeline.insert(
                "vision".into(),
                json!({ "image_placeholder_token_id": image_token_id }),
            );
        }
        Value::Object(pipeline)
    }

    fn encoder_decoder_pipeline_json(&self) -> Value {
        let encoder = self.model.encoder.as_ref();
        let mut models = Map::new();
        models.insert(
            "encoder".into(),
            component_json(
                filename_or(&encoder.and_then(|e| e.filename.clone()), "encoder.onnx"),
                "encoder",
                None,
            ),
        );
        let decoder_io = self.decoder_io_json(true);
        let decoder_io = (!decoder_io.is_empty()).then_some(Value::Object(decoder_io));
        models.insert(
            "decoder".into(),
            component_json(filename_or(&self.model.decoder.filename, "decoder.onnx"), "decoder", decoder_io),
        );

        let enc_hidden = encoder
            .and_then(|e| e.outputs.encoder_hidden_states.as_deref())
            .unwrap_or(DEFAULT_ENCODER_HIDDEN_STATES);
        let dec_hidden = self
            .model
            .decoder
            .inputs
            .encoder_hidden_states
            .as_deref()
            .unwrap_or(DEFAULT_ENCODER_HIDDEN_STATES);
        let dataflow = vec![edge(
            &format!("encoder.{enc_hidden}"),
            &format!("decoder.{dec_hidden}"),
        )];

        let mut phases = Map::new();
        phases.insert("encoder".into(), run_on("prompt_only"));
        phases.insert("decoder".into(), run_on("every_step"));

        let strategy = composite_encode_decode(Some("encoder"), "decoder");

        let mut pipeline = Map::new();
        pipeline.insert("models".into(), Value::Object(models));
        pipeline.insert("dataflow".into(), Value::Array(dataflow));
        pipeline.insert("strategy".into(), strategy);
        pipeline.insert("phases".into(), Value::Object(phases));
        Value::Object(pipeline)
    }

    fn decoder_pipeline_json(&self) -> Value {
        // NOTE: the split decoder graphs are wired by raw graph tensor names,
        // which contain dots (e.g. `past_key_values.0.key`) and cannot be
        // expressed as `component.port` dataflow endpoints yet, so the dataflow
        // is left empty; only the component list and ordering are captured.
        let mut models = Map::new();
        let mut last_stage: Option<String> = None;
        for stage in &self.model.decoder.pipeline {
            for (name, spec) in stage {
                let role = pipeline_stage_role(name);
                models.insert(
                    name.clone(),
                    component_json(filename_or(&spec.filename, &format!("{name}.onnx")), role, None),
                );
                last_stage = Some(name.clone());
            }
        }

        let decoder = last_stage.unwrap_or_else(|| "decoder".into());
        let strategy = json!({ "kind": "autoregressive", "decoder": decoder });

        let mut pipeline = Map::new();
        pipeline.insert("models".into(), Value::Object(models));
        pipeline.insert("dataflow".into(), Value::Array(Vec::new()));
        pipeline.insert("strategy".into(), strategy);
        Value::Object(pipeline)
    }

    fn generation_json(&self) -> Option<Value> {
        let s = &self.search;
        let mut m = Map::new();
        insert_bool(&mut m, "do_sample", s.do_sample);
        insert_f32(&mut m, "temperature", s.temperature);
        insert_usize(&mut m, "top_k", s.top_k);
        insert_f32(&mut m, "top_p", s.top_p);
        insert_f32(&mut m, "repetition_penalty", s.repetition_penalty);
        insert_usize(&mut m, "num_beams", s.num_beams);
        insert_usize(&mut m, "num_return_sequences", s.num_return_sequences);
        insert_usize(&mut m, "min_length", s.min_length);
        insert_usize(&mut m, "max_length", s.max_length);
        insert_f32(&mut m, "length_penalty", s.length_penalty);
        insert_usize(&mut m, "no_repeat_ngram_size", s.no_repeat_ngram_size);
        insert_f32(&mut m, "diversity_penalty", s.diversity_penalty);
        insert_bool(&mut m, "early_stopping", s.early_stopping);
        (!m.is_empty()).then_some(Value::Object(m))
    }

    fn tokens_json(&self) -> Option<Value> {
        let model = &self.model;
        let mut m = Map::new();
        insert_i64(&mut m, "pad_token_id", model.pad_token_id);
        insert_i64(&mut m, "bos_token_id", model.bos_token_id);
        if let Some(eos) = &model.eos_token_id {
            m.insert("eos_token_id".into(), json!(eos.to_vec()));
        }
        insert_i64(&mut m, "sep_token_id", model.sep_token_id);
        insert_i64(&mut m, "decoder_start_token_id", model.decoder_start_token_id);
        insert_i64(&mut m, "image_token_id", model.image_token_id);
        insert_i64(&mut m, "video_token_id", model.video_token_id);
        insert_i64(&mut m, "vision_start_token_id", model.vision_start_token_id);
        (!m.is_empty()).then_some(Value::Object(m))
    }
}

/// Path to a `genai_config.json` inside `model_dir`, if one exists.
pub fn find_in_dir(model_dir: &Path) -> Option<PathBuf> {
    let path = model_dir.join(GENAI_CONFIG_FILE);
    path.is_file().then_some(path)
}

/// Load and parse a `genai_config.json` from an explicit path.
pub fn load(path: &Path) -> Result<GenAiConfig, GenAiConfigError> {
    let content = std::fs::read_to_string(path)?;
    Ok(serde_json::from_str(&content)?)
}

/// Best-effort compatibility metadata for a model directory.
///
/// Returns `Ok(None)` when the directory has no `genai_config.json`.
pub fn inference_metadata_from_dir(
    model_dir: &Path,
    kv_native_dtype: Option<&str>,
) -> Result<Option<InferenceMetadata>, GenAiConfigError> {
    let Some(path) = find_in_dir(model_dir) else {
        return Ok(None);
    };
    let config = load(&path)?;
    Ok(Some(config.to_inference_metadata(kv_native_dtype)?))
}

// ---- helpers -------------------------------------------------------------

fn expand_pattern(pattern: &str, layers: usize) -> Vec<String> {
    (0..layers)
        .map(|i| pattern.replace("%d", &i.to_string()))
        .collect()
}

/// Expand a self-attention KV name pattern for all layers.
///
/// A combined pattern yields one name per layer; separate key/value patterns
/// (falling back to the conventional defaults) interleave `[key_i, value_i]`.
fn expand_kv(
    combined: Option<&str>,
    key: Option<&str>,
    value: Option<&str>,
    default_key: &str,
    default_value: &str,
    layers: Option<usize>,
) -> Option<Vec<String>> {
    let layers = layers?;
    if layers == 0 {
        return None;
    }
    if let Some(combined) = combined {
        return Some(expand_pattern(combined, layers));
    }
    let key = key.unwrap_or(default_key);
    let value = value.unwrap_or(default_value);
    let mut out = Vec::with_capacity(layers * 2);
    for i in 0..layers {
        out.push(key.replace("%d", &i.to_string()));
        out.push(value.replace("%d", &i.to_string()));
    }
    Some(out)
}

/// Expand a cross-attention KV name pattern; requires both key and value to be
/// declared (no default injection). Interleaves `[key_i, value_i]` per layer.
fn expand_cross_kv(key: Option<&str>, value: Option<&str>, layers: Option<usize>) -> Option<Vec<String>> {
    let (key, value, layers) = (key?, value?, layers?);
    if layers == 0 {
        return None;
    }
    let mut out = Vec::with_capacity(layers * 2);
    for i in 0..layers {
        out.push(key.replace("%d", &i.to_string()));
        out.push(value.replace("%d", &i.to_string()));
    }
    Some(out)
}

fn component_json(filename: String, role: &str, io: Option<Value>) -> Value {
    let mut m = Map::new();
    m.insert("filename".into(), json!(filename));
    m.insert("type".into(), json!(role));
    if let Some(io) = io {
        m.insert("io".into(), io);
    }
    Value::Object(m)
}

fn edge(from: &str, to: &str) -> Value {
    json!({ "from": from, "to": to })
}

fn run_on(phase: &str) -> Value {
    json!({ "run_on": phase })
}

/// A `composite` strategy: an optional single-pass encode stage followed by an
/// autoregressive decode stage.
fn composite_encode_decode(prompt_component: Option<&str>, decoder: &str) -> Value {
    let mut stages: Vec<Value> = Vec::new();
    if let Some(component) = prompt_component {
        stages.push(json!({
            "name": "encode",
            "run_on": "prompt_only",
            "strategy": { "kind": "single_pass", "model": component },
        }));
    }
    stages.push(json!({
        "name": "decode",
        "run_on": "every_step",
        "strategy": { "kind": "autoregressive", "decoder": decoder },
    }));
    json!({ "kind": "composite", "stages": stages })
}

fn pipeline_stage_role(name: &str) -> &'static str {
    match name {
        "embeddings" | "embedding" => "embedding",
        "language_model_head" | "lm_head" => "lm_head",
        _ => "decoder",
    }
}

fn filename_or(filename: &Option<String>, fallback: &str) -> String {
    filename.clone().unwrap_or_else(|| fallback.to_string())
}

fn insert_usize(map: &mut Map<String, Value>, key: &str, value: Option<usize>) {
    if let Some(value) = value {
        map.insert(key.into(), json!(value));
    }
}

fn insert_i64(map: &mut Map<String, Value>, key: &str, value: Option<i64>) {
    if let Some(value) = value {
        map.insert(key.into(), json!(value));
    }
}

fn insert_f32(map: &mut Map<String, Value>, key: &str, value: Option<f32>) {
    if let Some(value) = value {
        map.insert(key.into(), json!(value));
    }
}

fn insert_bool(map: &mut Map<String, Value>, key: &str, value: Option<bool>) {
    if let Some(value) = value {
        map.insert(key.into(), json!(value));
    }
}

/// Whether a dtype string denotes a KV dtype the share-buffer GQA path supports
/// (16- or 32-bit floating point). Mirrors the engine's gate.
fn is_share_buffer_kv_dtype(dtype: &str) -> bool {
    matches!(
        dtype.to_ascii_lowercase().as_str(),
        "float16" | "fp16" | "half" | "bfloat16" | "bf16" | "float32" | "fp32" | "float"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn qwen_config() -> GenAiConfig {
        serde_json::from_str(
            r#"{
                "model": {
                    "type": "qwen2",
                    "context_length": 32768,
                    "decoder": {
                        "head_size": 64,
                        "hidden_size": 896,
                        "num_attention_heads": 14,
                        "num_hidden_layers": 24,
                        "num_key_value_heads": 2
                    }
                },
                "search": { "past_present_share_buffer": true, "max_length": 32768 }
            }"#,
        )
        .expect("valid genai_config")
    }

    #[test]
    fn detects_gqa_and_capacity() {
        let cfg = qwen_config();
        assert!(cfg.is_group_query_attention());
        assert_eq!(cfg.max_sequence_length(), Some(32768));
        assert!(cfg.shared_kv_buffer_supported());
    }

    #[test]
    fn converts_and_enables_share_buffer_with_fp16() {
        let cfg = qwen_config();
        let md = cfg.to_inference_metadata(Some("float16")).unwrap();
        assert_eq!(md.schema_version.as_deref(), Some("v1"));
        let attention = md
            .model
            .as_ref()
            .and_then(|m| m.attention.as_ref())
            .expect("attention");
        assert_eq!(attention.attention_type, "group_query_attention");
        assert_eq!(attention.num_kv_heads, Some(2));
        assert_eq!(attention.num_attention_heads, Some(14));
        assert_eq!(attention.head_dim, Some(64));
        assert_eq!(
            md.model.as_ref().and_then(|m| m.max_sequence_length),
            Some(32768)
        );
        assert_eq!(
            md.kv_cache
                .as_ref()
                .and_then(|kv| kv.native_dtype.as_deref()),
            Some("float16")
        );
    }

    #[test]
    fn converts_and_enables_share_buffer_with_bf16() {
        let cfg = qwen_config();
        let md = cfg.to_inference_metadata(Some("bfloat16")).unwrap();
        assert_eq!(
            md.kv_cache
                .as_ref()
                .and_then(|kv| kv.native_dtype.as_deref()),
            Some("bfloat16")
        );
    }

    #[test]
    fn omits_kv_cache_when_share_buffer_disabled() {
        let mut cfg = qwen_config();
        cfg.search.past_present_share_buffer = false;
        let md = cfg.to_inference_metadata(Some("float16")).unwrap();
        assert!(md.kv_cache.is_none());
    }

    #[test]
    fn omits_kv_cache_for_unsupported_dtype() {
        let cfg = qwen_config();
        let md = cfg.to_inference_metadata(Some("int8")).unwrap();
        assert!(md.kv_cache.is_none());
    }

    #[test]
    fn omits_kv_cache_when_dtype_unknown() {
        let cfg = qwen_config();
        let md = cfg.to_inference_metadata(None).unwrap();
        assert!(md.kv_cache.is_none());
        assert!(md.model.and_then(|m| m.attention).is_some());
    }

    #[test]
    fn full_mha_via_gqa_op_is_share_buffer_eligible() {
        let mut cfg = qwen_config();
        cfg.model.decoder.num_attention_heads = Some(14);
        cfg.model.decoder.num_key_value_heads = Some(14);
        let md = cfg.to_inference_metadata(Some("float16")).unwrap();
        assert!(!cfg.is_group_query_attention());
        assert!(cfg.uses_group_query_attention_op());
        assert!(cfg.shared_kv_buffer_supported());
        assert!(md.kv_cache.is_some());
        assert_eq!(
            md.model.and_then(|m| m.attention).map(|a| a.attention_type),
            Some("group_query_attention".to_string())
        );
    }

    #[test]
    fn model_without_kv_heads_is_multi_head_and_not_share_buffer() {
        let mut cfg = qwen_config();
        cfg.model.decoder.num_key_value_heads = None;
        let md = cfg.to_inference_metadata(Some("float16")).unwrap();
        assert!(!cfg.uses_group_query_attention_op());
        assert!(!cfg.shared_kv_buffer_supported());
        assert!(md.kv_cache.is_none());
        assert_eq!(
            md.model.and_then(|m| m.attention).map(|a| a.attention_type),
            Some("multi_head_attention".to_string())
        );
    }

    // ---- Complete-coverage conversion tests -----------------------------

    /// gpt2: combined `past_%d` / `present_%d` KV patterns, scalar token ids,
    /// no `search` block.
    fn gpt2_config() -> GenAiConfig {
        serde_json::from_str(
            r#"{
                "model": {
                    "type": "gpt2",
                    "pad_token_id": 98,
                    "bos_token_id": 98,
                    "eos_token_id": 98,
                    "vocab_size": 1000,
                    "context_length": 512,
                    "decoder": {
                        "num_key_value_heads": 4,
                        "head_size": 8,
                        "num_hidden_layers": 5,
                        "inputs": { "past_names": "past_%d" },
                        "outputs": { "present_names": "present_%d" }
                    }
                }
            }"#,
        )
        .expect("valid gpt2 genai_config")
    }

    #[test]
    fn gpt2_expands_combined_kv_and_tokens() {
        let md = gpt2_config().to_inference_metadata(None).unwrap();

        let io = md
            .model
            .as_ref()
            .and_then(|m| m.io.as_ref())
            .expect("decoder io");
        // Combined pattern -> one entry per layer, in order.
        assert_eq!(
            io.kv_inputs.as_deref(),
            Some(&["past_0", "past_1", "past_2", "past_3", "past_4"].map(String::from)[..])
        );
        assert_eq!(
            io.kv_outputs.as_deref(),
            Some(
                &["present_0", "present_1", "present_2", "present_3", "present_4"]
                    .map(String::from)[..]
            )
        );
        // No inputs_embeds -> token-driven with the conventional default name.
        assert_eq!(io.token_input.as_deref(), Some("input_ids"));
        assert_eq!(io.logits_output.as_deref(), Some("logits"));

        let tokens = md.tokens.as_ref().expect("tokens");
        assert_eq!(tokens.pad_token_id, Some(98));
        assert_eq!(tokens.bos_token_id, Some(98));
        assert_eq!(tokens.eos_token_id.as_deref(), Some(&[98i64][..]));

        // No `search` block -> no generation defaults.
        assert!(md.generation.is_none());
        assert_eq!(md.model.and_then(|m| m.vocab_size), Some(1000));
    }

    /// Loads the real onnxruntime-genai fixtures from disk and asserts every
    /// one converts without error. Gated on `ORT_GENAI_TEST_MODELS` pointing at
    /// `onnxruntime-genai/test/test_models` so it stays hermetic by default.
    #[test]
    fn real_fixtures_convert_without_error() {
        let Ok(root) = std::env::var("ORT_GENAI_TEST_MODELS") else {
            return;
        };
        let root = std::path::Path::new(&root);
        let fixtures = [
            "hf-internal-testing/tiny-random-gpt2-fp32",
            "audio-preprocessing",
            "vision-preprocessing",
            "qwen-vision-preprocessing",
            "pipeline-model",
        ];
        for fixture in fixtures {
            let dir = root.join(fixture);
            if !dir.join(GENAI_CONFIG_FILE).is_file() {
                continue;
            }
            let md = inference_metadata_from_dir(&dir, Some("float16"))
                .unwrap_or_else(|e| panic!("{fixture}: {e}"))
                .unwrap_or_else(|| panic!("{fixture}: no genai_config.json"));
            assert_eq!(md.schema_version.as_deref(), Some("v1"), "{fixture}");
        }
    }

    #[test]
    fn whisper_encoder_decoder_pipeline_with_cross_kv() {
        let cfg: GenAiConfig = serde_json::from_str(WHISPER_JSON).unwrap();
        let md = cfg.to_inference_metadata(None).unwrap();

        let pipeline = md.pipeline.as_ref().expect("asr pipeline");
        assert!(pipeline.models.contains_key("encoder"));
        assert!(pipeline.models.contains_key("decoder"));
        assert!(matches!(
            pipeline.strategy.kind,
            onnx_genai_metadata::PipelineStrategyKind::Composite
        ));
        // encoder -> decoder cross-attention hidden-states dataflow.
        assert_eq!(pipeline.dataflow.len(), 1);
        assert_eq!(pipeline.dataflow[0].from, "encoder.encoder_hidden_states");
        assert_eq!(pipeline.dataflow[0].to, "decoder.encoder_hidden_states");

        let io = pipeline.models["decoder"].io.as_ref().expect("decoder io");
        assert_eq!(io.token_input.as_deref(), Some("input_ids"));
        assert_eq!(
            io.kv_inputs.as_deref(),
            Some(&["past_key_self_0", "past_value_self_0"].map(String::from)[..])
        );
        assert_eq!(
            io.kv_outputs.as_deref(),
            Some(&["present_key_self_0", "present_value_self_0"].map(String::from)[..])
        );
        assert_eq!(
            io.cross_kv_inputs.as_deref(),
            Some(&["past_key_cross_0", "past_value_cross_0"].map(String::from)[..])
        );
        assert_eq!(
            io.cross_kv_outputs.as_deref(),
            Some(&["present_key_cross_0", "present_value_cross_0"].map(String::from)[..])
        );
        assert_eq!(io.encoder_hidden_states_input.as_deref(), Some("encoder_hidden_states"));

        // Generation defaults come from `search`.
        let generation = md.generation.as_ref().expect("generation");
        assert_eq!(generation.max_length, Some(448));
        assert_eq!(generation.do_sample, Some(false));
        assert_eq!(generation.num_beams, Some(1));
    }

    #[test]
    fn phi3v_multimodal_pipeline_with_image_token() {
        let cfg: GenAiConfig = serde_json::from_str(PHI3V_JSON).unwrap();
        let md = cfg.to_inference_metadata(None).unwrap();

        let pipeline = md.pipeline.as_ref().expect("multimodal pipeline");
        assert!(pipeline.models.contains_key("vision_encoder"));
        assert!(pipeline.models.contains_key("embedding"));
        assert!(pipeline.models.contains_key("decoder"));

        // vision -> embedding -> decoder dataflow.
        let edges: Vec<(&str, &str)> = pipeline
            .dataflow
            .iter()
            .map(|e| (e.from.as_str(), e.to.as_str()))
            .collect();
        assert!(edges.contains(&("vision_encoder.image_features", "embedding.image_features")));
        assert!(edges.contains(&("embedding.inputs_embeds", "decoder.inputs_embeds")));

        // Embeds-driven decoder io.
        let io = pipeline.models["decoder"].io.as_ref().expect("decoder io");
        assert_eq!(io.inputs_embeds_input.as_deref(), Some("inputs_embeds"));
        assert!(io.token_input.is_none());

        // phi3v declares no image_token_id, so no vision expansion contract.
        assert!(pipeline.vision.is_none());
    }

    #[test]
    fn qwen_vlm_image_token_id_is_propagated() {
        let cfg: GenAiConfig = serde_json::from_str(QWEN_VLM_JSON).unwrap();
        let md = cfg.to_inference_metadata(None).unwrap();
        let pipeline = md.pipeline.as_ref().expect("multimodal pipeline");
        assert_eq!(
            pipeline
                .vision
                .as_ref()
                .and_then(|v| v.image_placeholder_token_id),
            Some(151_655)
        );
        let tokens = md.tokens.as_ref().expect("tokens");
        assert_eq!(tokens.image_token_id, Some(151_655));
        assert_eq!(tokens.video_token_id, Some(151_656));
        assert_eq!(tokens.vision_start_token_id, Some(151_652));
        // eos as array normalizes to a vec.
        assert_eq!(tokens.eos_token_id.as_deref(), Some(&[151_645, 151_643][..]));
    }

    #[test]
    fn decoder_pipeline_emits_split_models() {
        let cfg: GenAiConfig = serde_json::from_str(DECODER_PIPELINE_JSON).unwrap();
        let md = cfg.to_inference_metadata(None).unwrap();
        let pipeline = md.pipeline.as_ref().expect("decoder pipeline");
        assert!(pipeline.models.contains_key("embeddings"));
        assert!(pipeline.models.contains_key("transformer"));
        assert!(pipeline.models.contains_key("language_model_head"));
        assert_eq!(pipeline.models["embeddings"].role, "embedding");
        assert_eq!(pipeline.models["language_model_head"].role, "lm_head");
        assert_eq!(pipeline.models["transformer"].role, "decoder");
    }

    const WHISPER_JSON: &str = r#"{
        "model": {
            "type": "whisper",
            "bos_token_id": 50257,
            "eos_token_id": 50257,
            "pad_token_id": 50257,
            "context_length": 448,
            "vocab_size": 51865,
            "decoder": {
                "filename": "dummy_decoder.onnx",
                "head_size": 64,
                "num_attention_heads": 6,
                "num_hidden_layers": 1,
                "num_key_value_heads": 6,
                "inputs": {
                    "input_ids": "input_ids",
                    "past_key_names": "past_key_self_%d",
                    "past_value_names": "past_value_self_%d",
                    "cross_past_key_names": "past_key_cross_%d",
                    "cross_past_value_names": "past_value_cross_%d"
                },
                "outputs": {
                    "logits": "logits",
                    "present_key_names": "present_key_self_%d",
                    "present_value_names": "present_value_self_%d",
                    "output_cross_qk_names": "output_cross_qk_%d"
                }
            },
            "encoder": {
                "filename": "dummy_encoder.onnx",
                "num_attention_heads": 6,
                "num_hidden_layers": 1,
                "inputs": { "audio_features": "audio_features" },
                "outputs": {
                    "encoder_hidden_states": "encoder_hidden_states",
                    "cross_present_key_names": "present_key_cross_%d",
                    "cross_present_value_names": "present_value_cross_%d"
                }
            }
        },
        "search": {
            "do_sample": false,
            "early_stopping": true,
            "length_penalty": 1.0,
            "max_length": 448,
            "min_length": 0,
            "num_beams": 1,
            "num_return_sequences": 1,
            "past_present_share_buffer": false,
            "repetition_penalty": 1.0,
            "temperature": 1.0,
            "top_k": 1,
            "top_p": 1.0
        }
    }"#;

    const PHI3V_JSON: &str = r#"{
        "model": {
            "type": "phi3v",
            "bos_token_id": 1,
            "eos_token_id": 32007,
            "pad_token_id": 32000,
            "context_length": 131072,
            "vocab_size": 32064,
            "decoder": {
                "filename": "dummy_text.onnx",
                "head_size": 96,
                "num_attention_heads": 32,
                "num_hidden_layers": 1,
                "num_key_value_heads": 32,
                "inputs": {
                    "inputs_embeds": "inputs_embeds",
                    "attention_mask": "attention_mask",
                    "past_key_names": "past_key_values.%d.key",
                    "past_value_names": "past_key_values.%d.value"
                },
                "outputs": {
                    "logits": "logits",
                    "present_key_names": "present.%d.key",
                    "present_value_names": "present.%d.value"
                }
            },
            "embedding": {
                "filename": "dummy_embedding.onnx",
                "inputs": { "input_ids": "input_ids", "image_features": "image_features" },
                "outputs": { "inputs_embeds": "inputs_embeds" }
            },
            "vision": {
                "filename": "dummy_vision.onnx",
                "inputs": { "pixel_values": "pixel_values", "image_sizes": "image_sizes" },
                "outputs": { "image_features": "image_features" }
            }
        },
        "search": { "past_present_share_buffer": true, "max_length": 131072 }
    }"#;

    const QWEN_VLM_JSON: &str = r#"{
        "model": {
            "type": "qwen2_5_vl",
            "bos_token_id": 151643,
            "eos_token_id": [151645, 151643],
            "pad_token_id": 151643,
            "image_token_id": 151655,
            "video_token_id": 151656,
            "vision_start_token_id": 151652,
            "context_length": 128000,
            "vocab_size": 152064,
            "decoder": {
                "filename": "dummy_text.onnx",
                "head_size": 128,
                "num_attention_heads": 28,
                "num_hidden_layers": 1,
                "num_key_value_heads": 4,
                "inputs": {
                    "inputs_embeds": "inputs_embeds",
                    "attention_mask": "attention_mask",
                    "position_ids": "position_ids",
                    "past_key_names": "past_key_values.%d.key",
                    "past_value_names": "past_key_values.%d.value"
                },
                "outputs": {
                    "logits": "logits",
                    "present_key_names": "present.%d.key",
                    "present_value_names": "present.%d.value"
                }
            },
            "embedding": {
                "filename": "dummy_embedding.onnx",
                "inputs": { "input_ids": "input_ids", "image_features": "image_features" },
                "outputs": { "inputs_embeds": "inputs_embeds" }
            },
            "vision": {
                "filename": "dummy_vision.onnx",
                "inputs": { "pixel_values": "pixel_values", "image_grid_thw": "image_grid_thw" },
                "outputs": { "image_features": "image_features" }
            }
        },
        "search": { "past_present_share_buffer": true, "max_length": 128000 }
    }"#;

    const DECODER_PIPELINE_JSON: &str = r#"{
        "model": {
            "type": "decoder-pipeline",
            "bos_token_id": 50256,
            "eos_token_id": 50256,
            "pad_token_id": 50256,
            "context_length": 2048,
            "vocab_size": 51200,
            "decoder": {
                "head_size": 80,
                "num_attention_heads": 32,
                "num_hidden_layers": 1,
                "num_key_value_heads": 32,
                "inputs": {
                    "input_ids": "input_ids",
                    "attention_mask": "attention_mask",
                    "past_key_names": "past_key_values.%d.key",
                    "past_value_names": "past_key_values.%d.value"
                },
                "outputs": {
                    "logits": "logits",
                    "present_key_names": "present.%d.key",
                    "present_value_names": "present.%d.value"
                },
                "pipeline": [
                    {
                        "embeddings": {
                            "filename": "embeds.onnx",
                            "inputs": ["input_ids"],
                            "outputs": ["/model/embed_tokens/Gather/output_0"]
                        },
                        "transformer": {
                            "filename": "transformer.onnx",
                            "inputs": ["/model/embed_tokens/Gather/output_0", "attention_mask", "past_key_values.0.key", "past_key_values.0.value"],
                            "outputs": ["hidden_states", "present.0.key", "present.0.value"]
                        },
                        "language_model_head": {
                            "filename": "lm_head.onnx",
                            "inputs": ["hidden_states"],
                            "outputs": ["logits"]
                        }
                    }
                ]
            }
        },
        "search": { "past_present_share_buffer": true, "max_length": 2048 }
    }"#;
}
