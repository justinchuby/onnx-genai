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

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use onnx_genai_metadata::{InferenceMetadata, SCHEMA_VERSION, capabilities};
use onnx_genai_preprocess::image::ImagePreprocessor;
use serde::Deserialize;
use serde_json::{Map, Value, json};

mod json_builders;
mod loading;
mod wire_types;

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
}

/// One graph tensor declaration supplied by the package loader.
///
/// This inventory comes from the ONNX graph interface itself, so compatibility
/// conversion can preserve actual sparse ports, ranks, and dtypes instead of
/// expanding architecture-sized guesses.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphTensorInfo {
    pub name: String,
    pub dtype: String,
    /// One entry per axis; `None` denotes a symbolic dimension.
    pub dimensions: Vec<Option<usize>>,
}

/// Explicit input/output inventory for one ONNX component.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ModelGraphInfo {
    pub inputs: Vec<GraphTensorInfo>,
    pub outputs: Vec<GraphTensorInfo>,
}

/// ONNX graph inventories required to synthesize a strict multimodal pipeline.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PipelineGraphInfo {
    pub vision: ModelGraphInfo,
    pub embedding: ModelGraphInfo,
    pub decoder: ModelGraphInfo,
}

/// ONNX graph inventories required to synthesize a strict encoder-decoder
/// (audio/text sequence-to-sequence) pipeline, e.g. Whisper.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct EncoderDecoderGraphInfo {
    pub encoder: ModelGraphInfo,
    pub decoder: ModelGraphInfo,
}

#[derive(Debug, Default, Deserialize)]
struct CompatibilityConfig {
    #[serde(default)]
    text_config: Option<CompatibilityTextConfig>,
}

#[derive(Debug, Default, Deserialize)]
struct CompatibilityTextConfig {
    #[serde(default)]
    rope_parameters: Option<CompatibilityRopeParameters>,
}

#[derive(Debug, Default, Deserialize)]
struct CompatibilityRopeParameters {
    #[serde(default)]
    mrope_section: Option<Vec<usize>>,
}

struct DecoderStateMetadata {
    kv_inputs: Vec<String>,
    kv_outputs: Vec<String>,
    state_pairs: Vec<Value>,
    kv_dtype: String,
}

/// Coarse structural family a `genai_config.json` describes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ModelShape {
    /// A single, unsplit decoder graph.
    SingleDecoder,
    /// Embedding + vision/speech front-ends feeding a decoder (multimodal).
    Multimodal,
    /// Encoder + cross-attention decoder (ASR / whisper).
    EncoderDecoder,
    /// RNN-T transducer: streaming encoder + LSTM prediction network + joint
    /// network (+ optional VAD / streaming caches). A DISTINCT pipeline family
    /// the current inference-metadata contract does not execute. Recognized so
    /// it is never silently mis-bound as an encoder-decoder model.
    Transducer,
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
        self.search.past_present_share_buffer == Some(true)
            && self.uses_group_query_attention_op()
            && self.max_sequence_length().is_some()
    }

    pub(crate) fn shape(&self) -> ModelShape {
        // Recognize the RNN-T transducer family BEFORE the encoder check: a
        // transducer also declares `model.encoder`, but its encoder is a
        // streaming Conformer with cache state (not a Whisper encoder emitting
        // cross-KV), its decoder is an LSTM prediction network (no attention KV),
        // and it carries a joint network. Classifying it as EncoderDecoder here
        // would silently fabricate Whisper-style cross-KV bindings that do not
        // exist. Detection is structural (joint network / LSTM decoder states),
        // never keyed on `model.type` or a model name.
        if self.is_transducer() {
            ModelShape::Transducer
        } else if self.model.encoder.is_some() {
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

    /// Whether this package describes an RNN-T transducer (e.g. Nemotron speech:
    /// Conformer encoder + LSTM prediction network + joint network + VAD).
    ///
    /// Detected purely from structure — never from `model.type` or a model name:
    /// a transducer declares a `model.joiner` joint network, and/or its decoder
    /// is an LSTM prediction network driven by `targets` + LSTM hidden/cell state
    /// with no attention KV. Either signal is sufficient. This is a DISTINCT
    /// pipeline family the current inference-metadata contract cannot execute;
    /// it must not be classified or bound as an encoder-decoder model.
    pub fn is_transducer(&self) -> bool {
        self.model.joiner.is_some() || self.decoder_is_lstm_prediction_network()
    }

    /// Whether the decoder graph is an LSTM prediction network (RNN-T) rather
    /// than an attention transformer decoder: it consumes LSTM hidden/cell state
    /// and exposes no self- or cross-attention KV ports.
    fn decoder_is_lstm_prediction_network(&self) -> bool {
        let inputs = &self.model.decoder.inputs;
        (inputs.lstm_hidden_state.is_some() || inputs.lstm_cell_state.is_some())
            && inputs.past_key_names.is_none()
            && inputs.past_value_names.is_none()
            && inputs.past_names.is_none()
            && inputs.cross_past_key_names.is_none()
            && inputs.cross_past_value_names.is_none()
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
    /// skipped (loading never fails on them): VAD, Conformer NeMo
    /// `cache_last_channel`/`cache_last_time` state, LSTM/RNN decoder states
    /// (`rnn_states`, `lstm_hidden_state`, `lstm_cell_state`), paged-attention
    /// `block_table`, beam `cache_indirection`, `output_cross_qk`, and the
    /// per-session `session_options`/`run_options`.
    ///
    /// EXCEPTION: an RNN-T transducer package (joint/joiner network and/or an
    /// LSTM prediction network) is NOT skipped-into an encoder-decoder spec — it
    /// is recognized by [`Self::is_transducer`] and declined with an explicit
    /// [`GenAiConfigError::UnsupportedPipelineFamily`], because silently emitting
    /// a Whisper-style cross-attention spec for it would fabricate bindings the
    /// graphs do not expose.
    pub fn to_inference_metadata(
        &self,
        kv_native_dtype: Option<&str>,
    ) -> Result<InferenceMetadata, GenAiConfigError> {
        self.build_inference_metadata(kv_native_dtype, None)
    }

    /// Like [`Self::to_inference_metadata`], but derives the single-decoder
    /// KV/state topology from the decoder's actual ONNX graph inventory instead
    /// of expanding KV name patterns over a uniform per-layer count.
    ///
    /// This is what lets hybrid SSM/attention decoders (sparse dense-KV layers
    /// plus fixed `conv_state`/`recurrent_state` recurrent layers) load: only the
    /// ports the graph truly exposes are declared. Uniform dense-KV decoders
    /// produce byte-identical metadata to the pattern-expanded path.
    pub fn to_inference_metadata_with_graph(
        &self,
        kv_native_dtype: Option<&str>,
        decoder_graph: &ModelGraphInfo,
    ) -> Result<InferenceMetadata, GenAiConfigError> {
        self.build_inference_metadata(kv_native_dtype, Some(decoder_graph))
    }

    fn build_inference_metadata(
        &self,
        kv_native_dtype: Option<&str>,
        decoder_graph: Option<&ModelGraphInfo>,
    ) -> Result<InferenceMetadata, GenAiConfigError> {
        let shape = self.shape();

        // Decline the transducer family up front: it has no executable
        // representation, so there is nothing honest to synthesize.
        if shape == ModelShape::Transducer {
            return Err(transducer_unsupported());
        }

        let mut model = Map::new();
        model.insert("attention".into(), self.attention_json());
        insert_usize(
            &mut model,
            "max_sequence_length",
            self.max_sequence_length(),
        );
        insert_usize(&mut model, "vocab_size", self.model.vocab_size);

        if shape == ModelShape::SingleDecoder {
            let io = self.decoder_io_json(false, decoder_graph);
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
            // Declined above; unreachable but kept explicit for exhaustiveness.
            ModelShape::Transducer => return Err(transducer_unsupported()),
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
        if self.uses_group_query_attention_op() {
            attention.insert(
                "key_sequence_lengths".into(),
                json!({ "scalar_broadcast": "unit_batch" }),
            );
        }
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
    fn decoder_io_json(
        &self,
        include_cross: bool,
        decoder_graph: Option<&ModelGraphInfo>,
    ) -> Map<String, Value> {
        let dec = &self.model.decoder;
        let layers = dec.num_hidden_layers;
        let mut io = Map::new();

        if let Some(token) = dec.inputs.input_ids.as_deref() {
            io.insert("token_input".into(), json!(token));
        }
        if let Some(embeds) = dec.inputs.inputs_embeds.as_deref() {
            io.insert("inputs_embeds_input".into(), json!(embeds));
        } else if dec.inputs.input_ids.is_none() {
            io.insert("token_input".into(), json!(DEFAULT_INPUT_IDS));
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

        // When the loader supplies the decoder's actual ONNX graph inventory,
        // derive the KV/state topology from it rather than blindly expanding the
        // `%d` KV name patterns over `0..num_hidden_layers`. Hybrid SSM/attention
        // models (e.g. qwen3.5: linear-attention layers expose
        // `conv_state`/`recurrent_state`, only the periodic full-attention layers
        // expose dense `key`/`value`) have a SPARSE dense-KV port set plus fixed
        // recurrent state ports. Trusting the uniform per-layer assumption there
        // declares ports the graph never exposes and aborts warmup. The
        // graph-derived path emits exactly the ports the graph presents: sparse
        // `kv_inputs`/`kv_outputs` for dense layers and `state_pairs` for the
        // recurrent state. A uniform dense-KV model yields an identical KV list
        // and no state pairs, so existing models are unaffected. Any structural
        // mismatch (missing name patterns, dtype disagreement) falls back to the
        // pattern expansion below so no currently loading model regresses.
        let graph_state =
            decoder_graph.and_then(|graph| self.graph_decoder_state(graph).ok().flatten());
        if let Some(state) = graph_state {
            io.insert("kv_inputs".into(), json!(state.kv_inputs));
            io.insert("kv_outputs".into(), json!(state.kv_outputs));
            if !state.state_pairs.is_empty() {
                io.insert("state_pairs".into(), Value::Array(state.state_pairs));
            }
        } else {
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
                component_json(
                    filename_or(&vision.filename, "vision.onnx"),
                    "encoder",
                    None,
                ),
            );
            phases.insert("vision_encoder".into(), run_on("prompt_only"));
            prompt_encoder.get_or_insert_with(|| "vision_encoder".into());
            if self.model.embedding.is_some() {
                let from = vision
                    .outputs
                    .image_features
                    .as_deref()
                    .unwrap_or("image_features");
                let to = self
                    .model
                    .embedding
                    .as_ref()
                    .and_then(|e| e.inputs.image_features.as_deref())
                    .unwrap_or("image_features");
                dataflow.push(edge(
                    &format!("vision_encoder.{from}"),
                    &format!("embedding.{to}"),
                ));
            }
        }

        if let Some(speech) = &self.model.speech {
            models.insert(
                "audio_encoder".into(),
                component_json(
                    filename_or(&speech.filename, "speech.onnx"),
                    "encoder",
                    None,
                ),
            );
            phases.insert("audio_encoder".into(), run_on("prompt_only"));
            prompt_encoder.get_or_insert_with(|| "audio_encoder".into());
            if self.model.embedding.is_some() {
                let from = speech
                    .outputs
                    .audio_features
                    .as_deref()
                    .unwrap_or("audio_features");
                let to = self
                    .model
                    .embedding
                    .as_ref()
                    .and_then(|e| e.inputs.audio_features.as_deref())
                    .unwrap_or("audio_features");
                dataflow.push(edge(
                    &format!("audio_encoder.{from}"),
                    &format!("embedding.{to}"),
                ));
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
                component_json(
                    filename_or(&embedding.filename, "embedding.onnx"),
                    "embedding",
                    io,
                ),
            );
            phases.insert("embedding".into(), run_on("every_step"));

            let from = embedding
                .outputs
                .inputs_embeds
                .as_deref()
                .unwrap_or("inputs_embeds");
            let to = self
                .model
                .decoder
                .inputs
                .inputs_embeds
                .as_deref()
                .unwrap_or("inputs_embeds");
            dataflow.push(edge(&format!("embedding.{from}"), &format!("decoder.{to}")));
        }

        let decoder_io = self.decoder_io_json(false, None);
        let decoder_io = (!decoder_io.is_empty()).then_some(Value::Object(decoder_io));
        models.insert(
            "decoder".into(),
            component_json(
                filename_or(&self.model.decoder.filename, "decoder.onnx"),
                "decoder",
                decoder_io,
            ),
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
        let decoder_io = self.decoder_io_json(true, None);
        let decoder_io = (!decoder_io.is_empty()).then_some(Value::Object(decoder_io));
        models.insert(
            "decoder".into(),
            component_json(
                filename_or(&self.model.decoder.filename, "decoder.onnx"),
                "decoder",
                decoder_io,
            ),
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
                    component_json(
                        filename_or(&spec.filename, &format!("{name}.onnx")),
                        role,
                        None,
                    ),
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
        insert_i64(
            &mut m,
            "decoder_start_token_id",
            model.decoder_start_token_id,
        );
        insert_i64(&mut m, "image_token_id", model.image_token_id);
        insert_i64(&mut m, "video_token_id", model.video_token_id);
        insert_i64(&mut m, "vision_start_token_id", model.vision_start_token_id);
        (!m.is_empty()).then_some(Value::Object(m))
    }
}

impl GenAiConfig {
    pub(crate) fn to_strict_pipeline_metadata(
        &self,
        model_dir: &Path,
        graphs: &PipelineGraphInfo,
    ) -> Result<InferenceMetadata, GenAiConfigError> {
        let vision = required_ref(self.model.vision.as_ref(), "model.vision")?;
        let embedding = required_ref(self.model.embedding.as_ref(), "model.embedding")?;
        let vision_filename = required_str(vision.filename.as_deref(), "model.vision.filename")?;
        let embedding_filename =
            required_str(embedding.filename.as_deref(), "model.embedding.filename")?;
        let decoder_filename = required_str(
            self.model.decoder.filename.as_deref(),
            "model.decoder.filename",
        )?;
        let processor_filename = required_str(
            vision.config_filename.as_deref(),
            "model.vision.config_filename",
        )?;

        let compatibility_config: CompatibilityConfig =
            load_auxiliary_json(&model_dir.join("config.json"), "config.json")?;
        let processor: ProcessorConfig = load_auxiliary_json(
            &model_dir.join(processor_filename),
            "processor config declared by model.vision.config_filename",
        )?;

        let vision_pixel = required_str(
            vision.inputs.pixel_values.as_deref(),
            "model.vision.inputs.pixel_values",
        )?;
        let vision_grid = required_str(
            vision.inputs.image_grid_thw.as_deref(),
            "model.vision.inputs.image_grid_thw",
        )?;
        let vision_features = required_str(
            vision.outputs.image_features.as_deref(),
            "model.vision.outputs.image_features",
        )?;
        let embedding_tokens = required_str(
            embedding.inputs.input_ids.as_deref(),
            "model.embedding.inputs.input_ids",
        )?;
        let embedding_image = required_str(
            embedding.inputs.image_features.as_deref(),
            "model.embedding.inputs.image_features",
        )?;
        let embedding_output = required_str(
            embedding.outputs.inputs_embeds.as_deref(),
            "model.embedding.outputs.inputs_embeds",
        )?;
        let decoder_embeds = required_str(
            self.model.decoder.inputs.inputs_embeds.as_deref(),
            "model.decoder.inputs.inputs_embeds",
        )?;
        let decoder_mask = required_str(
            self.model.decoder.inputs.attention_mask.as_deref(),
            "model.decoder.inputs.attention_mask",
        )?;
        let decoder_position = required_str(
            self.model.decoder.inputs.position_ids.as_deref(),
            "model.decoder.inputs.position_ids",
        )?;
        let decoder_logits = required_str(
            self.model.decoder.outputs.logits.as_deref(),
            "model.decoder.outputs.logits",
        )?;
        let image_token_id = required_copy(self.model.image_token_id, "model.image_token_id")?;
        let past_present_share_buffer = required_copy(
            self.search.past_present_share_buffer,
            "search.past_present_share_buffer",
        )?;
        required_positive(vision.spatial_merge_size, "model.vision.spatial_merge_size")?;
        required_positive(vision.patch_size, "model.vision.patch_size")?;

        let vision_pixel_info = require_graph_input(&graphs.vision, vision_pixel, "vision")?;
        let vision_grid_info = require_graph_input(&graphs.vision, vision_grid, "vision")?;
        let vision_features_info = require_graph_output(&graphs.vision, vision_features, "vision")?;
        require_graph_input(&graphs.embedding, embedding_tokens, "embedding")?;
        let embedding_image_info =
            require_graph_input(&graphs.embedding, embedding_image, "embedding")?;
        let embedding_output_info =
            require_graph_output(&graphs.embedding, embedding_output, "embedding")?;
        let decoder_embeds_info = require_graph_input(&graphs.decoder, decoder_embeds, "decoder")?;
        require_graph_input(&graphs.decoder, decoder_mask, "decoder")?;
        let position_info = require_graph_input(&graphs.decoder, decoder_position, "decoder")?;
        require_graph_output(&graphs.decoder, decoder_logits, "decoder")?;

        require_same_dtype(
            vision_features_info,
            embedding_image_info,
            "vision image-features dataflow",
        )?;
        require_same_dtype(
            embedding_output_info,
            decoder_embeds_info,
            "embedding-to-decoder dataflow",
        )?;

        let sections = compatibility_config
            .text_config
            .and_then(|text| text.rope_parameters)
            .and_then(|rope| rope.mrope_section);
        if position_info.dimensions.len() != 3 {
            return Err(incomplete(format!(
                "decoder position input rank 3 required by the declared image_grid_thw processor summary (got rank {})",
                position_info.dimensions.len()
            )));
        }
        if sections.is_none() {
            return Err(incomplete(
                "config.json text_config.rope_parameters.mrope_section for the multi-axis position input",
            ));
        }
        if let Some(sections) = &sections
            && sections.len() != position_info.dimensions.len()
        {
            return Err(incomplete(format!(
                "position section count ({}) does not match the ONNX position rank ({})",
                sections.len(),
                position_info.dimensions.len()
            )));
        }
        if sections
            .as_ref()
            .is_some_and(|sections| sections.contains(&0))
        {
            return Err(incomplete(
                "config.json text_config.rope_parameters.mrope_section entries must be greater than zero",
            ));
        }

        let DecoderStateMetadata {
            kv_inputs,
            kv_outputs,
            state_pairs,
            kv_dtype,
        } = self.strict_decoder_state(&graphs.decoder)?;
        let has_state_pairs = !state_pairs.is_empty();
        let preprocessing =
            processor_program_json(&processor, vision, vision_pixel_info, vision_grid_info)?;

        let mut decoder_io = Map::new();
        if let Some(token) = self.model.decoder.inputs.input_ids.as_deref() {
            require_graph_input(&graphs.decoder, token, "decoder")?;
            decoder_io.insert("token_input".into(), json!(token));
        }
        decoder_io.insert("inputs_embeds_input".into(), json!(decoder_embeds));
        decoder_io.insert("attention_mask_input".into(), json!(decoder_mask));
        decoder_io.insert("position_ids_input".into(), json!(decoder_position));
        decoder_io.insert("logits_output".into(), json!(decoder_logits));
        decoder_io.insert("kv_inputs".into(), json!(kv_inputs));
        decoder_io.insert("kv_outputs".into(), json!(kv_outputs));
        decoder_io.insert(
            "kv_update".into(),
            json!(if past_present_share_buffer {
                "shared_buffer"
            } else {
                "append"
            }),
        );
        if has_state_pairs {
            decoder_io.insert("state_pairs".into(), Value::Array(state_pairs));
        }

        let mut embedding_io = Map::new();
        embedding_io.insert("token_input".into(), json!(embedding_tokens));

        let mut models = Map::new();
        models.insert(
            "vision_encoder".into(),
            component_json(vision_filename.to_owned(), "vision_encoder", None),
        );
        models.insert(
            "embedding".into(),
            component_json(
                embedding_filename.to_owned(),
                "embedding",
                Some(Value::Object(embedding_io)),
            ),
        );
        models.insert(
            "decoder".into(),
            component_json(
                decoder_filename.to_owned(),
                "decoder",
                Some(Value::Object(decoder_io)),
            ),
        );

        let dataflow = vec![
            edge_with_dtype(
                &format!("vision_encoder.{vision_features}"),
                &format!("embedding.{embedding_image}"),
                &vision_features_info.dtype,
            ),
            edge_with_dtype(
                &format!("embedding.{embedding_output}"),
                &format!("decoder.{decoder_embeds}"),
                &embedding_output_info.dtype,
            ),
        ];
        let mut phases = Map::new();
        phases.insert("vision_encoder".into(), run_on("prompt_only"));
        phases.insert("embedding".into(), run_on("every_step"));
        phases.insert("decoder".into(), run_on("every_step"));

        let strategy = json!({
            "kind": "composite",
            "stages": [
                {
                    "name": "encode_vision",
                    "run_on": "prompt_only",
                    "strategy": { "kind": "single_pass", "model": "vision_encoder" }
                },
                {
                    "name": "embed_tokens",
                    "run_on": "every_step",
                    "strategy": { "kind": "single_pass", "model": "embedding" }
                },
                {
                    "name": "decode",
                    "run_on": "every_step",
                    "strategy": { "kind": "autoregressive", "decoder": "decoder" }
                }
            ]
        });

        let positions = json!({
            "input": decoder_position,
            "rank": position_info.dimensions.len(),
            "axes": ["temporal", "height", "width"],
            "sections": sections,
            "dtype": position_info.dtype,
            "continuation": "from_grid",
            "processor_summaries": [vision_grid]
        });

        let mut pipeline = Map::new();
        pipeline.insert("models".into(), Value::Object(models));
        pipeline.insert("dataflow".into(), Value::Array(dataflow));
        pipeline.insert("strategy".into(), strategy);
        pipeline.insert("phases".into(), Value::Object(phases));
        pipeline.insert(
            "vision".into(),
            json!({
                "image_placeholder_token_id": image_token_id,
                "image_token_id": image_token_id,
                "token_count_source": "from_grid",
                "placeholder_per_image": true
            }),
        );
        pipeline.insert("positions".into(), positions);

        let mut required_capabilities = vec![
            capabilities::IMAGE_PREPROCESSING_PROGRAM,
            capabilities::POSITION_PROGRAM,
        ];
        if preprocessing["image"]["outputs"]
            .as_array()
            .is_some_and(|outputs| outputs.len() > 1)
        {
            required_capabilities.push(capabilities::PACKED_IMAGE_OUTPUTS);
        }
        required_capabilities.push(capabilities::MULTI_AXIS_POSITIONS);
        if has_state_pairs {
            required_capabilities.push(capabilities::LOOP_CARRIED_STATE);
        }

        let mut model = Map::new();
        model.insert("attention".into(), self.attention_json());
        insert_usize(
            &mut model,
            "max_sequence_length",
            self.max_sequence_length(),
        );
        insert_usize(&mut model, "vocab_size", self.model.vocab_size);

        let mut root = Map::new();
        root.insert("schema_version".into(), json!(SCHEMA_VERSION));
        root.insert("required_capabilities".into(), json!(required_capabilities));
        root.insert("model".into(), Value::Object(model));
        root.insert("preprocessing".into(), preprocessing);
        root.insert("pipeline".into(), Value::Object(pipeline));
        if let Some(generation) = self.generation_json() {
            root.insert("generation".into(), generation);
        }
        if let Some(tokens) = self.tokens_json() {
            root.insert("tokens".into(), tokens);
        }
        if past_present_share_buffer && is_share_buffer_kv_dtype(&kv_dtype) {
            root.insert("kv_cache".into(), json!({ "native_dtype": kv_dtype }));
        }

        let metadata: InferenceMetadata = serde_json::from_value(Value::Object(root))?;
        let image_program = metadata
            .preprocessing
            .as_ref()
            .and_then(|preprocessing| preprocessing.image.as_ref())
            .ok_or_else(|| incomplete("synthesized typed image preprocessing program"))?;
        let pixel_shape = vision_pixel_info
            .dimensions
            .iter()
            .map(|dimension| match dimension {
                Some(dimension) => i64::try_from(*dimension).map_err(|_| {
                    incomplete(format!(
                        "vision pixel input '{}' dimension {dimension} fits in i64",
                        vision_pixel_info.name
                    ))
                }),
                None => Ok(-1),
            })
            .collect::<Result<Vec<_>, _>>()?;
        ImagePreprocessor::from_input_and_program(&pixel_shape, image_program).map_err(|error| {
            incomplete(format!(
                "synthesized image preprocessing program is not executable by ImagePreprocessor: {error}"
            ))
        })?;
        Ok(metadata)
    }

    /// Strict encoder-decoder pipeline synth (audio/text sequence-to-sequence).
    ///
    /// Recognized purely from the encoder-decoder SHAPE of `genai_config.json`
    /// (a declared `model.encoder` with cross-attention KV outputs feeding the
    /// decoder's cross-attention KV inputs), never from `model.type` or a model
    /// name, so any encoder-decoder family (Whisper audio, and other
    /// sequence-to-sequence encoders) synthesizes the same way. Every port,
    /// rank, and dtype fact is validated against the authoritative ONNX graphs.
    pub(crate) fn to_strict_encoder_decoder_pipeline_metadata(
        &self,
        graphs: &EncoderDecoderGraphInfo,
    ) -> Result<InferenceMetadata, GenAiConfigError> {
        let encoder = required_ref(self.model.encoder.as_ref(), "model.encoder")?;
        let decoder = &self.model.decoder;
        let encoder_filename = required_str(encoder.filename.as_deref(), "model.encoder.filename")?;
        let decoder_filename = required_str(decoder.filename.as_deref(), "model.decoder.filename")?;

        // Encoder prompt input, keyed off the declared input SHAPE, not a model
        // name: audio front-ends declare `audio_features`, text encoders declare
        // `input_ids`. Exactly one must be present.
        let (encoder_input_field, encoder_input) = match (
            encoder.inputs.audio_features.as_deref(),
            encoder.inputs.input_ids.as_deref(),
        ) {
            (Some(audio), None) => ("model.encoder.inputs.audio_features", audio),
            (None, Some(ids)) => ("model.encoder.inputs.input_ids", ids),
            (Some(_), Some(_)) => {
                return Err(incomplete(
                    "model.encoder declares both audio_features and input_ids; exactly one encoder prompt input is required",
                ));
            }
            (None, None) => {
                return Err(incomplete(
                    "model.encoder.inputs.audio_features or model.encoder.inputs.input_ids",
                ));
            }
        };
        let encoder_input = required_str(Some(encoder_input), encoder_input_field)?;
        require_graph_input(&graphs.encoder, encoder_input, "encoder")?;

        let encoder_hidden = required_str(
            encoder.outputs.encoder_hidden_states.as_deref(),
            "model.encoder.outputs.encoder_hidden_states",
        )?;
        require_graph_output(&graphs.encoder, encoder_hidden, "encoder")?;

        let token = required_str(
            decoder.inputs.input_ids.as_deref(),
            "model.decoder.inputs.input_ids",
        )?;
        require_graph_input(&graphs.decoder, token, "decoder")?;
        let logits = required_str(
            decoder.outputs.logits.as_deref(),
            "model.decoder.outputs.logits",
        )?;
        require_graph_output(&graphs.decoder, logits, "decoder")?;

        // Self-attention KV: the growing per-step cache. Matched by pattern
        // against the decoder graph so only the ports the graph truly exposes are
        // declared, paired positionally as `[key_i, value_i, ...]`.
        let (self_input_indices, self_kv_inputs, self_input_dtype) = strict_indexed_kv(
            &graphs.decoder.inputs,
            required_str(
                decoder.inputs.past_key_names.as_deref(),
                "model.decoder.inputs.past_key_names",
            )?,
            required_str(
                decoder.inputs.past_value_names.as_deref(),
                "model.decoder.inputs.past_value_names",
            )?,
            "decoder self-attention past key/value",
        )?;
        let (self_output_indices, self_kv_outputs, self_output_dtype) = strict_indexed_kv(
            &graphs.decoder.outputs,
            required_str(
                decoder.outputs.present_key_names.as_deref(),
                "model.decoder.outputs.present_key_names",
            )?,
            required_str(
                decoder.outputs.present_value_names.as_deref(),
                "model.decoder.outputs.present_value_names",
            )?,
            "decoder self-attention present key/value",
        )?;
        if self_input_indices != self_output_indices {
            return Err(incomplete(
                "decoder self-attention past/present KV do not have identical layer indices",
            ));
        }
        if self_input_dtype != self_output_dtype {
            return Err(incomplete(format!(
                "decoder self-attention past KV dtype {self_input_dtype} does not match present KV dtype {self_output_dtype}"
            )));
        }

        // Cross-attention KV static routing. The encoder computes the cross KV
        // ONCE from the audio/text prompt and emits `present_*_cross_%d`; those
        // feed the decoder's `past_*_cross_%d` inputs and never grow or update
        // across decode steps. This is why they are wired as pipeline dataflow
        // edges from the encoder to the decoder (a prompt-time prologue result),
        // distinct from the growing self-attention cache the decoder owns.
        let (cross_input_indices, cross_kv_inputs, cross_input_dtype) = strict_indexed_kv(
            &graphs.decoder.inputs,
            required_str(
                decoder.inputs.cross_past_key_names.as_deref(),
                "model.decoder.inputs.cross_past_key_names",
            )?,
            required_str(
                decoder.inputs.cross_past_value_names.as_deref(),
                "model.decoder.inputs.cross_past_value_names",
            )?,
            "decoder cross-attention past key/value",
        )?;
        let (cross_output_indices, cross_kv_outputs, cross_output_dtype) = strict_indexed_kv(
            &graphs.encoder.outputs,
            required_str(
                encoder.outputs.cross_present_key_names.as_deref(),
                "model.encoder.outputs.cross_present_key_names",
            )?,
            required_str(
                encoder.outputs.cross_present_value_names.as_deref(),
                "model.encoder.outputs.cross_present_value_names",
            )?,
            "encoder cross-attention present key/value",
        )?;
        if cross_input_indices != cross_output_indices {
            return Err(incomplete(
                "encoder-produced and decoder-consumed cross-attention KV do not have identical layer indices",
            ));
        }
        if cross_input_dtype != cross_output_dtype {
            return Err(incomplete(format!(
                "encoder cross-attention KV dtype {cross_output_dtype} does not match decoder cross-attention KV dtype {cross_input_dtype}"
            )));
        }
        // Cross-attention KV static routing is declared through the decoder's
        // paired `cross_kv_inputs` (the decoder's `past_*_cross_%d` ports) and
        // `cross_kv_outputs` (the encoder's `present_*_cross_%d` ports), matched
        // positionally per layer. The runtime binds these encoder-produced KV
        // tensors as stateful decoder inputs computed ONCE at prompt time, so
        // they are NOT wired as per-step dataflow edges (doing so would
        // double-bind the port). `dataflow` carries only genuine per-invocation
        // tensor edges, e.g. the encoder hidden-states edge below when present.
        let mut dataflow: Vec<Value> = Vec::new();

        let mut decoder_io = Map::new();
        decoder_io.insert("token_input".into(), json!(token));
        if let Some(mask) = decoder.inputs.attention_mask.as_deref() {
            require_graph_input(&graphs.decoder, mask, "decoder")?;
            decoder_io.insert("attention_mask_input".into(), json!(mask));
        }
        if let Some(position) = decoder.inputs.position_ids.as_deref() {
            require_graph_input(&graphs.decoder, position, "decoder")?;
            decoder_io.insert("position_ids_input".into(), json!(position));
        }
        // Some encoder-decoder decoders also consume the encoder hidden states
        // directly (computing cross KV internally); route it only when declared
        // AND actually present as a decoder graph input.
        if let Some(decoder_hidden) = decoder.inputs.encoder_hidden_states.as_deref()
            && require_graph_input(&graphs.decoder, decoder_hidden, "decoder").is_ok()
        {
            decoder_io.insert("encoder_hidden_states_input".into(), json!(decoder_hidden));
            dataflow.push(edge_with_dtype(
                &format!("encoder.{encoder_hidden}"),
                &format!("decoder.{decoder_hidden}"),
                &cross_output_dtype,
            ));
        }
        decoder_io.insert("logits_output".into(), json!(logits));
        decoder_io.insert("kv_inputs".into(), json!(self_kv_inputs));
        decoder_io.insert("kv_outputs".into(), json!(self_kv_outputs));
        decoder_io.insert("kv_update".into(), json!("append"));
        decoder_io.insert("cross_kv_inputs".into(), json!(cross_kv_inputs));
        decoder_io.insert("cross_kv_outputs".into(), json!(cross_kv_outputs));

        let mut encoder_io = Map::new();
        // Audio front-ends declare the mel `audio_features` input; text encoders
        // reuse the ordinary `token_input`. Keyed off the encoder-input SHAPE
        // resolved above, never the model name.
        let encoder_input_role = if encoder_input_field.ends_with("audio_features") {
            "audio_features_input"
        } else {
            "token_input"
        };
        encoder_io.insert(encoder_input_role.into(), json!(encoder_input));

        let mut models = Map::new();
        models.insert(
            "encoder".into(),
            component_json(
                encoder_filename.to_owned(),
                "encoder",
                Some(Value::Object(encoder_io)),
            ),
        );
        models.insert(
            "decoder".into(),
            component_json(
                decoder_filename.to_owned(),
                "decoder",
                Some(Value::Object(decoder_io)),
            ),
        );

        let mut phases = Map::new();
        phases.insert("encoder".into(), run_on("prompt_only"));
        phases.insert("decoder".into(), run_on("every_step"));

        let strategy = composite_encode_decode(Some("encoder"), "decoder");

        let mut pipeline = Map::new();
        pipeline.insert("models".into(), Value::Object(models));
        pipeline.insert("dataflow".into(), Value::Array(dataflow));
        pipeline.insert("strategy".into(), strategy);
        pipeline.insert("phases".into(), Value::Object(phases));

        let mut model = Map::new();
        model.insert("attention".into(), self.attention_json());
        insert_usize(
            &mut model,
            "max_sequence_length",
            self.max_sequence_length(),
        );
        insert_usize(&mut model, "vocab_size", self.model.vocab_size);

        let mut root = Map::new();
        root.insert("schema_version".into(), json!(SCHEMA_VERSION));
        root.insert("model".into(), Value::Object(model));
        root.insert("pipeline".into(), Value::Object(pipeline));
        if let Some(generation) = self.generation_json() {
            root.insert("generation".into(), generation);
        }
        if let Some(tokens) = self.tokens_json() {
            root.insert("tokens".into(), tokens);
        }

        Ok(serde_json::from_value(Value::Object(root))?)
    }

    fn strict_decoder_state(
        &self,
        graph: &ModelGraphInfo,
    ) -> Result<DecoderStateMetadata, GenAiConfigError> {
        let decoder = &self.model.decoder;
        let past_key = required_str(
            decoder.inputs.past_key_names.as_deref(),
            "model.decoder.inputs.past_key_names",
        )?;
        let past_value = required_str(
            decoder.inputs.past_value_names.as_deref(),
            "model.decoder.inputs.past_value_names",
        )?;
        let present_key = required_str(
            decoder.outputs.present_key_names.as_deref(),
            "model.decoder.outputs.present_key_names",
        )?;
        let present_value = required_str(
            decoder.outputs.present_value_names.as_deref(),
            "model.decoder.outputs.present_value_names",
        )?;

        let past_key_names = match_indexed_tensors(&graph.inputs, past_key)?;
        let past_value_names = match_indexed_tensors(&graph.inputs, past_value)?;
        let present_key_names = match_indexed_tensors(&graph.outputs, present_key)?;
        let present_value_names = match_indexed_tensors(&graph.outputs, present_value)?;
        let indices = exact_index_set(
            &[
                &past_key_names,
                &past_value_names,
                &present_key_names,
                &present_value_names,
            ],
            "actual sparse key/value graph ports",
        )?;
        if indices.is_empty() {
            return Err(incomplete(
                "at least one actual decoder key/value graph-port pair",
            ));
        }

        let mut kv_inputs = Vec::with_capacity(indices.len() * 2);
        let mut kv_outputs = Vec::with_capacity(indices.len() * 2);
        let mut kv_dtype = None;
        for index in indices {
            let past_key = past_key_names[&index];
            let past_value = past_value_names[&index];
            let present_key = present_key_names[&index];
            let present_value = present_value_names[&index];
            require_same_dtype(past_key, present_key, "key cache input/output")?;
            require_same_dtype(past_value, present_value, "value cache input/output")?;
            require_same_dtype(past_key, past_value, "key/value cache")?;
            if let Some(canonical_dtype) = kv_dtype.as_deref() {
                if canonical_dtype != past_key.dtype {
                    return Err(incomplete(format!(
                        "all decoder key/value cache tensors must use one dtype: canonical dtype is {canonical_dtype}, but '{}' is {}",
                        past_key.name, past_key.dtype
                    )));
                }
            } else {
                kv_dtype = Some(past_key.dtype.clone());
            }
            kv_inputs.extend([past_key.name.clone(), past_value.name.clone()]);
            kv_outputs.extend([present_key.name.clone(), present_value.name.clone()]);
        }

        let past_prefix = common_pattern_prefix(past_key, past_value)?;
        let present_prefix = common_pattern_prefix(present_key, present_value)?;
        let kv_input_names = kv_inputs
            .iter()
            .map(String::as_str)
            .collect::<BTreeSet<_>>();
        let kv_output_names = kv_outputs
            .iter()
            .map(String::as_str)
            .collect::<BTreeSet<_>>();
        let state_inputs = suffix_tensor_map(
            &graph.inputs,
            past_prefix,
            &kv_input_names,
            "fixed-state inputs",
        )?;
        let state_outputs = suffix_tensor_map(
            &graph.outputs,
            present_prefix,
            &kv_output_names,
            "fixed-state outputs",
        )?;
        if state_inputs.keys().collect::<Vec<_>>() != state_outputs.keys().collect::<Vec<_>>() {
            return Err(incomplete(format!(
                "fixed-state input/output suffixes do not pair exactly (inputs: {:?}, outputs: {:?})",
                state_inputs.keys().collect::<Vec<_>>(),
                state_outputs.keys().collect::<Vec<_>>()
            )));
        }
        let mut state_pairs = Vec::with_capacity(state_inputs.len());
        for (suffix, input) in state_inputs {
            let output = state_outputs[&suffix];
            require_same_dtype(input, output, "fixed-state input/output")?;
            if input.dimensions != output.dimensions {
                return Err(incomplete(format!(
                    "fixed-state pair '{}'/'{}' has different ONNX shapes",
                    input.name, output.name
                )));
            }
            state_pairs.push(json!({
                "input": input.name,
                "output": output.name,
                "init": "zeros",
                "update": "replace"
            }));
        }

        Ok(DecoderStateMetadata {
            kv_inputs,
            kv_outputs,
            state_pairs,
            kv_dtype: kv_dtype.expect("non-empty KV indices establish a dtype"),
        })
    }

    /// Best-effort graph-derived decoder KV/state topology for the single-decoder
    /// compatibility path.
    ///
    /// Returns `Ok(None)` when the config lacks the separate key/value name
    /// patterns [`strict_decoder_state`](Self::strict_decoder_state) needs, and
    /// swallows structural derivation errors into `Ok(None)` so callers can fall
    /// back to uniform `%d` pattern expansion without regressing any model that
    /// loads today. When it does return `Some`, the ports are exactly those the
    /// ONNX graph exposes (sparse dense KV plus fixed recurrent `state_pairs`).
    fn graph_decoder_state(
        &self,
        decoder: &ModelGraphInfo,
    ) -> Result<Option<DecoderStateMetadata>, GenAiConfigError> {
        let dec = &self.model.decoder;
        // Require the separate key/value input+output patterns; combined-name or
        // default-only configs stay on the pattern-expansion path.
        if dec.inputs.past_key_names.is_none()
            || dec.inputs.past_value_names.is_none()
            || dec.outputs.present_key_names.is_none()
            || dec.outputs.present_value_names.is_none()
        {
            return Ok(None);
        }
        Ok(self.strict_decoder_state(decoder).ok())
    }
}

// ---- helpers -------------------------------------------------------------

fn incomplete(missing: impl Into<String>) -> GenAiConfigError {
    GenAiConfigError::IncompletePipeline {
        missing: missing.into(),
    }
}

/// Honest decline for an RNN-T transducer package. The transducer family
/// (streaming Conformer encoder with cache state + LSTM prediction network +
/// joint network + optional VAD, driven by a blank-symbol greedy transducer
/// loop) has no representation in the current inference-metadata contract, so
/// the loader declines with a descriptive reason instead of fabricating a
/// Whisper-style cross-attention encoder-decoder spec that does not match the
/// graphs.
pub(crate) fn transducer_unsupported() -> GenAiConfigError {
    GenAiConfigError::UnsupportedPipelineFamily {
        family: "RNN-T transducer".into(),
        reason: "the package declares a joint (joiner) network and/or an LSTM prediction \
                 network (targets + lstm_hidden_state/lstm_cell_state, no attention KV), i.e. a \
                 Conformer-Transducer topology. Executing it needs a joint-network greedy \
                 transducer decode loop (blank_id / max_symbols_per_step), streaming encoder \
                 cache state (cache_last_channel/cache_last_time), and VAD segmentation — none of \
                 which the encoder-decoder cross-attention contract models"
            .into(),
    }
}

pub(crate) fn required_str<'a>(
    value: Option<&'a str>,
    field: &str,
) -> Result<&'a str, GenAiConfigError> {
    value
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| incomplete(field))
}

fn required_ref<'a, T>(value: Option<&'a T>, field: &str) -> Result<&'a T, GenAiConfigError> {
    value.ok_or_else(|| incomplete(field))
}

fn required_copy<T: Copy>(value: Option<T>, field: &str) -> Result<T, GenAiConfigError> {
    value.ok_or_else(|| incomplete(field))
}

fn required_positive(value: Option<usize>, field: &str) -> Result<usize, GenAiConfigError> {
    value
        .filter(|value| *value > 0)
        .ok_or_else(|| incomplete(format!("{field} must be greater than zero")))
}

fn load_auxiliary_json<T>(path: &Path, description: &str) -> Result<T, GenAiConfigError>
where
    T: serde::de::DeserializeOwned,
{
    let content = std::fs::read_to_string(path).map_err(|error| {
        incomplete(format!(
            "{description} at {} could not be read: {error}",
            path.display()
        ))
    })?;
    serde_json::from_str(&content).map_err(|error| {
        incomplete(format!(
            "{description} at {} is not valid for compatibility conversion: {error}",
            path.display()
        ))
    })
}

fn require_graph_input<'a>(
    graph: &'a ModelGraphInfo,
    name: &str,
    component: &str,
) -> Result<&'a GraphTensorInfo, GenAiConfigError> {
    graph
        .inputs
        .iter()
        .find(|tensor| tensor.name == name)
        .ok_or_else(|| incomplete(format!("{component} ONNX input '{name}'")))
}

fn require_graph_output<'a>(
    graph: &'a ModelGraphInfo,
    name: &str,
    component: &str,
) -> Result<&'a GraphTensorInfo, GenAiConfigError> {
    graph
        .outputs
        .iter()
        .find(|tensor| tensor.name == name)
        .ok_or_else(|| incomplete(format!("{component} ONNX output '{name}'")))
}

fn require_same_dtype(
    left: &GraphTensorInfo,
    right: &GraphTensorInfo,
    description: &str,
) -> Result<(), GenAiConfigError> {
    if left.dtype == right.dtype {
        Ok(())
    } else {
        Err(incomplete(format!(
            "{description} dtype agreement: '{}' is {}, but '{}' is {}",
            left.name, left.dtype, right.name, right.dtype
        )))
    }
}

/// Match a paired key/value `%d` name pattern against `tensors` and return the
/// ordered layer index set, the interleaved `[key_0, value_0, key_1, ...]`
/// names verified to exist in the graph, and the single shared cache dtype.
///
/// Unlike [`GenAiConfig::strict_decoder_state`], this does not require the key
/// and value patterns to share a common textual prefix (Whisper self/cross KV
/// use distinct `..._key_self_%d` / `..._value_self_%d` prefixes), and it does
/// not derive any fixed-state `state_pairs`, so encoder-decoder cross-attention
/// and cross-QK ports are never misread as recurrent state.
fn strict_indexed_kv(
    tensors: &[GraphTensorInfo],
    key_pattern: &str,
    value_pattern: &str,
    description: &str,
) -> Result<(Vec<usize>, Vec<String>, String), GenAiConfigError> {
    let keys = match_indexed_tensors(tensors, key_pattern)?;
    let values = match_indexed_tensors(tensors, value_pattern)?;
    let indices = exact_index_set(&[&keys, &values], description)?;
    if indices.is_empty() {
        return Err(incomplete(format!(
            "at least one {description} graph-port pair"
        )));
    }
    let mut names = Vec::with_capacity(indices.len() * 2);
    let mut dtype: Option<String> = None;
    for index in &indices {
        let key = keys[index];
        let value = values[index];
        require_same_dtype(key, value, description)?;
        match dtype.as_deref() {
            Some(canonical) if canonical != key.dtype => {
                return Err(incomplete(format!(
                    "all {description} tensors must use one dtype: canonical dtype is {canonical}, but '{}' is {}",
                    key.name, key.dtype
                )));
            }
            None => dtype = Some(key.dtype.clone()),
            _ => {}
        }
        names.push(key.name.clone());
        names.push(value.name.clone());
    }
    Ok((
        indices,
        names,
        dtype.expect("non-empty KV indices establish a dtype"),
    ))
}

fn split_indexed_pattern(pattern: &str) -> Result<(&str, &str), GenAiConfigError> {
    let Some((prefix, suffix)) = pattern.split_once("%d") else {
        return Err(incomplete(format!(
            "indexed tensor pattern '{pattern}' must contain %d"
        )));
    };
    if suffix.contains("%d") {
        return Err(incomplete(format!(
            "indexed tensor pattern '{pattern}' must contain exactly one %d"
        )));
    }
    Ok((prefix, suffix))
}

fn match_indexed_tensors<'a>(
    tensors: &'a [GraphTensorInfo],
    pattern: &str,
) -> Result<BTreeMap<usize, &'a GraphTensorInfo>, GenAiConfigError> {
    let (prefix, suffix) = split_indexed_pattern(pattern)?;
    let mut matched = BTreeMap::new();
    for tensor in tensors {
        let Some(index) = tensor
            .name
            .strip_prefix(prefix)
            .and_then(|name| name.strip_suffix(suffix))
        else {
            continue;
        };
        let index = index.parse::<usize>().map_err(|_| {
            incomplete(format!(
                "ONNX tensor '{}' matches pattern '{pattern}' but has a non-numeric index",
                tensor.name
            ))
        })?;
        if matched.insert(index, tensor).is_some() {
            return Err(incomplete(format!(
                "ONNX graph has duplicate tensors for pattern '{pattern}' index {index}"
            )));
        }
    }
    Ok(matched)
}

fn exact_index_set(
    maps: &[&BTreeMap<usize, &GraphTensorInfo>],
    description: &str,
) -> Result<Vec<usize>, GenAiConfigError> {
    let Some(first) = maps.first() else {
        return Ok(Vec::new());
    };
    let expected = first.keys().copied().collect::<Vec<_>>();
    if maps
        .iter()
        .skip(1)
        .all(|map| map.keys().copied().eq(expected.iter().copied()))
    {
        Ok(expected)
    } else {
        Err(incomplete(format!(
            "{description} do not have identical layer indices"
        )))
    }
}

fn common_pattern_prefix<'a>(first: &'a str, second: &'a str) -> Result<&'a str, GenAiConfigError> {
    let (first_prefix, _) = split_indexed_pattern(first)?;
    let (second_prefix, _) = split_indexed_pattern(second)?;
    if first_prefix == second_prefix {
        Ok(first_prefix)
    } else {
        Err(incomplete(format!(
            "key/value patterns '{first}' and '{second}' must share the same state prefix"
        )))
    }
}

fn suffix_tensor_map<'a>(
    tensors: &'a [GraphTensorInfo],
    prefix: &str,
    excluded: &BTreeSet<&str>,
    description: &str,
) -> Result<BTreeMap<String, &'a GraphTensorInfo>, GenAiConfigError> {
    let mut matched = BTreeMap::new();
    for tensor in tensors {
        if excluded.contains(tensor.name.as_str()) {
            continue;
        }
        let Some(suffix) = tensor.name.strip_prefix(prefix) else {
            continue;
        };
        if suffix.is_empty() {
            return Err(incomplete(format!(
                "{description} contains an empty suffix for '{}'",
                tensor.name
            )));
        }
        if matched.insert(suffix.to_owned(), tensor).is_some() {
            return Err(incomplete(format!(
                "{description} contains duplicate suffix '{suffix}'"
            )));
        }
    }
    Ok(matched)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn processor_config(smart_resize: Value, temporal_patch_size: Value) -> ProcessorConfig {
        serde_json::from_value(json!({
            "processor": {
                "transforms": [
                    { "operation": { "type": "DecodeImage" } },
                    {
                        "operation": {
                            "type": "Resize",
                            "attrs": {
                                "width": 32,
                                "height": 32,
                                "smart_resize": smart_resize
                            }
                        }
                    },
                    {
                        "operation": {
                            "type": "Rescale",
                            "attrs": { "rescale_factor": 0.00392156862745098_f64 }
                        }
                    },
                    {
                        "operation": {
                            "type": "Normalize",
                            "attrs": {
                                "mean": [0.5, 0.5, 0.5],
                                "std": [0.5, 0.5, 0.5]
                            }
                        }
                    },
                    {
                        "operation": {
                            "type": "PatchImage",
                            "attrs": {
                                "patch_size": 16,
                                "temporal_patch_size": temporal_patch_size,
                                "merge_size": 2
                            }
                        }
                    }
                ]
            }
        }))
        .expect("processor fixture")
    }

    fn processor_vision() -> GenAiVision {
        serde_json::from_value(json!({
            "patch_size": 16,
            "spatial_merge_size": 2,
            "inputs": {
                "pixel_values": "pixel_values",
                "image_grid_thw": "image_grid_thw"
            }
        }))
        .expect("vision fixture")
    }

    fn processor_tensor(name: &str, dtype: &str) -> GraphTensorInfo {
        GraphTensorInfo {
            name: name.to_owned(),
            dtype: dtype.to_owned(),
            dimensions: vec![None, None],
        }
    }

    #[test]
    fn processor_requires_numeric_smart_resize_flag() {
        let mut missing = processor_config(json!(0), json!(1));
        missing
            .processor
            .transforms
            .iter_mut()
            .find(|transform| transform.operation.operation_type == "Resize")
            .expect("resize transform")
            .operation
            .attrs
            .remove("smart_resize");
        let missing_error = processor_program_json(
            &missing,
            &processor_vision(),
            &processor_tensor("pixel_values", "float32"),
            &processor_tensor("image_grid_thw", "int64"),
        )
        .expect_err("missing smart_resize must fail")
        .to_string();
        assert!(missing_error.contains("smart_resize"));
        assert!(missing_error.contains("numeric flag 0 or 1"));

        for value in [Value::Null, json!("false"), json!(2)] {
            let error = processor_program_json(
                &processor_config(value, json!(1)),
                &processor_vision(),
                &processor_tensor("pixel_values", "float32"),
                &processor_tensor("image_grid_thw", "int64"),
            )
            .expect_err("invalid smart_resize must fail")
            .to_string();
            assert!(error.contains("smart_resize"));
            assert!(error.contains("numeric flag 0 or 1"));
        }
    }

    #[test]
    fn processor_rejects_unexecutable_smart_resize() {
        let error = processor_program_json(
            &processor_config(json!(1), json!(1)),
            &processor_vision(),
            &processor_tensor("pixel_values", "float32"),
            &processor_tensor("image_grid_thw", "int64"),
        )
        .expect_err("smart resize must fail until executable")
        .to_string();
        assert!(error.contains("smart_resize=false"));
        assert!(error.contains("stretch/crop/pad"));
    }

    #[test]
    fn processor_emits_executable_temporal_patch_size() {
        let program = processor_program_json(
            &processor_config(json!(0), json!(2)),
            &processor_vision(),
            &processor_tensor("pixel_values", "float32"),
            &processor_tensor("image_grid_thw", "int64"),
        )
        .expect("temporal patching is executable");
        let patchify = program["image"]["transforms"]
            .as_array()
            .expect("transforms")
            .iter()
            .find(|transform| transform["op"] == "patchify")
            .expect("patchify");
        assert_eq!(patchify["temporal_patch_size"], 2);
        assert_eq!(patchify["merge_size"], 2);
        assert_eq!(patchify["channel_order"], "channels_first");
    }

    fn hybrid_graph_tensor(name: &str, dtype: &str, dims: &[Option<usize>]) -> GraphTensorInfo {
        GraphTensorInfo {
            name: name.to_string(),
            dtype: dtype.to_string(),
            dimensions: dims.to_vec(),
        }
    }

    /// A hybrid SSM/attention decoder (qwen3.5-shaped): four layers where the
    /// odd layers are dense full-attention (`key`/`value`) and the even layers
    /// are linear-attention recurrent (`conv_state`/`recurrent_state`). The
    /// genai_config only carries the uniform `%d` KV pattern and a layer count,
    /// so deriving metadata from the graph is the only way to avoid declaring the
    /// six non-existent dense-KV ports for the recurrent layers.
    fn hybrid_config() -> GenAiConfig {
        serde_json::from_str(
            r#"{
                "model": {
                    "type": "qwen3_5_text",
                    "context_length": 4096,
                    "decoder": {
                        "head_size": 256,
                        "hidden_size": 2048,
                        "num_attention_heads": 8,
                        "num_hidden_layers": 4,
                        "num_key_value_heads": 2,
                        "inputs": {
                            "input_ids": "input_ids",
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
                    }
                },
                "search": { "past_present_share_buffer": true, "max_length": 4096 }
            }"#,
        )
        .expect("valid hybrid genai_config")
    }

    fn hybrid_decoder_graph() -> ModelGraphInfo {
        let sym = |_n: &str| None;
        let dense = [Some(1), Some(2), sym("seq"), Some(256)];
        let conv = [Some(1), Some(6144), Some(3)];
        let recur = [Some(1), Some(16), Some(128), Some(128)];
        let mut inputs = vec![
            hybrid_graph_tensor("input_ids", "int64", &[Some(1), sym("seq")]),
            hybrid_graph_tensor("attention_mask", "int64", &[Some(1), sym("seq")]),
            hybrid_graph_tensor("position_ids", "int64", &[Some(1), sym("seq")]),
        ];
        let mut outputs = vec![hybrid_graph_tensor(
            "logits",
            "float32",
            &[Some(1), sym("seq"), Some(248320)],
        )];
        for layer in 0..4 {
            if layer % 2 == 1 {
                inputs.push(hybrid_graph_tensor(
                    &format!("past_key_values.{layer}.key"),
                    "float32",
                    &dense,
                ));
                inputs.push(hybrid_graph_tensor(
                    &format!("past_key_values.{layer}.value"),
                    "float32",
                    &dense,
                ));
                outputs.push(hybrid_graph_tensor(
                    &format!("present.{layer}.key"),
                    "float32",
                    &dense,
                ));
                outputs.push(hybrid_graph_tensor(
                    &format!("present.{layer}.value"),
                    "float32",
                    &dense,
                ));
            } else {
                inputs.push(hybrid_graph_tensor(
                    &format!("past_key_values.{layer}.conv_state"),
                    "float32",
                    &conv,
                ));
                inputs.push(hybrid_graph_tensor(
                    &format!("past_key_values.{layer}.recurrent_state"),
                    "float32",
                    &recur,
                ));
                outputs.push(hybrid_graph_tensor(
                    &format!("present.{layer}.conv_state"),
                    "float32",
                    &conv,
                ));
                outputs.push(hybrid_graph_tensor(
                    &format!("present.{layer}.recurrent_state"),
                    "float32",
                    &recur,
                ));
            }
        }
        ModelGraphInfo { inputs, outputs }
    }

    #[test]
    fn hybrid_decoder_derives_sparse_kv_and_state_pairs() {
        let cfg = hybrid_config();
        let graph = hybrid_decoder_graph();
        let md = cfg
            .to_inference_metadata_with_graph(Some("float32"), &graph)
            .expect("hybrid metadata");
        let io = md
            .model
            .as_ref()
            .and_then(|m| m.io.as_ref())
            .expect("decoder io");

        // Only the two dense full-attention layers (1, 3) expose key/value; the
        // recurrent layers must NOT appear in the KV lists.
        assert_eq!(
            io.kv_inputs.as_deref(),
            Some(
                [
                    "past_key_values.1.key",
                    "past_key_values.1.value",
                    "past_key_values.3.key",
                    "past_key_values.3.value",
                ]
                .map(String::from)
                .as_slice()
            )
        );
        assert_eq!(
            io.kv_outputs.as_deref(),
            Some(
                [
                    "present.1.key",
                    "present.1.value",
                    "present.3.key",
                    "present.3.value",
                ]
                .map(String::from)
                .as_slice()
            )
        );

        // The four recurrent ports (conv_state + recurrent_state for layers 0, 2)
        // are declared as fixed loop-carried state pairs, replaced each step.
        let pairs = io.state_pairs.as_ref().expect("state pairs");
        let mut got: Vec<(String, String)> = pairs
            .iter()
            .map(|pair| (pair.input.clone(), pair.output.clone()))
            .collect();
        got.sort();
        let mut want = vec![
            (
                "past_key_values.0.conv_state".to_string(),
                "present.0.conv_state".to_string(),
            ),
            (
                "past_key_values.0.recurrent_state".to_string(),
                "present.0.recurrent_state".to_string(),
            ),
            (
                "past_key_values.2.conv_state".to_string(),
                "present.2.conv_state".to_string(),
            ),
            (
                "past_key_values.2.recurrent_state".to_string(),
                "present.2.recurrent_state".to_string(),
            ),
        ];
        want.sort();
        assert_eq!(got, want);
        for pair in pairs {
            assert_eq!(pair.init.as_deref(), Some("zeros"));
            assert_eq!(pair.update.as_deref(), Some("replace"));
        }
    }

    #[test]
    fn uniform_decoder_graph_matches_pattern_expansion() {
        // A dense-KV model must produce the SAME kv_inputs whether or not the
        // graph is supplied, and must never gain state pairs.
        let cfg = qwen_config();
        let without_graph = cfg.to_inference_metadata(Some("float16")).unwrap();
        let expected_kv = without_graph
            .model
            .as_ref()
            .and_then(|m| m.io.as_ref())
            .and_then(|io| io.kv_inputs.clone())
            .expect("pattern-expanded kv inputs");

        let mut inputs = Vec::new();
        let mut outputs = Vec::new();
        for layer in 0..24 {
            inputs.push(hybrid_graph_tensor(
                &format!("past_key_values.{layer}.key"),
                "float16",
                &[Some(1), Some(2), None, Some(64)],
            ));
            inputs.push(hybrid_graph_tensor(
                &format!("past_key_values.{layer}.value"),
                "float16",
                &[Some(1), Some(2), None, Some(64)],
            ));
            outputs.push(hybrid_graph_tensor(
                &format!("present.{layer}.key"),
                "float16",
                &[Some(1), Some(2), None, Some(64)],
            ));
            outputs.push(hybrid_graph_tensor(
                &format!("present.{layer}.value"),
                "float16",
                &[Some(1), Some(2), None, Some(64)],
            ));
        }
        let graph = ModelGraphInfo { inputs, outputs };
        let with_graph = cfg
            .to_inference_metadata_with_graph(Some("float16"), &graph)
            .unwrap();
        let io = with_graph
            .model
            .as_ref()
            .and_then(|m| m.io.as_ref())
            .expect("io");
        assert_eq!(io.kv_inputs.as_ref(), Some(&expected_kv));
        assert!(io.state_pairs.is_none());
    }

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
            attention
                .key_sequence_lengths
                .as_ref()
                .and_then(|spec| spec.scalar_broadcast),
            Some(onnx_genai_metadata::SequenceLengthScalarBroadcast::UnitBatch)
        );
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
        cfg.search.past_present_share_buffer = Some(false);
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
            md.model
                .and_then(|m| m.attention)
                .map(|a| (a.attention_type, a.key_sequence_lengths)),
            Some((
                "group_query_attention".to_string(),
                Some(onnx_genai_metadata::KeySequenceLengthsSpec {
                    scalar_broadcast: Some(
                        onnx_genai_metadata::SequenceLengthScalarBroadcast::UnitBatch
                    ),
                })
            ))
        );
    }

    #[test]
    fn non_gqa_op_omits_scalar_key_sequence_lengths_permission() {
        let mut cfg = qwen_config();
        cfg.model.decoder.num_key_value_heads = None;
        let md = cfg.to_inference_metadata(Some("float16")).unwrap();
        assert_eq!(
            md.model
                .and_then(|m| m.attention)
                .map(|a| (a.attention_type, a.key_sequence_lengths)),
            Some(("multi_head_attention".to_string(), None))
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
                &[
                    "present_0",
                    "present_1",
                    "present_2",
                    "present_3",
                    "present_4"
                ]
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
        assert_eq!(
            io.encoder_hidden_states_input.as_deref(),
            Some("encoder_hidden_states")
        );

        // Generation defaults come from `search`.
        let generation = md.generation.as_ref().expect("generation");
        assert_eq!(generation.max_length, Some(448));
        assert_eq!(generation.do_sample, Some(false));
        assert_eq!(generation.num_beams, Some(1));
    }

    #[test]
    fn whisper_strict_encoder_decoder_synth_routes_cross_kv() {
        // Strict, graph-verified encoder-decoder synth (the path the ORT compat
        // loader uses). Unlike the pattern-expanded `to_inference_metadata`, the
        // cross-attention KV is wired as explicit encoder->decoder dataflow edges
        // (static, computed once by the encoder), and the audio prompt input is
        // surfaced on the encoder component.
        let cfg: GenAiConfig = serde_json::from_str(WHISPER_JSON).unwrap();
        let graphs = EncoderDecoderGraphInfo {
            encoder: ModelGraphInfo {
                inputs: vec![hybrid_graph_tensor(
                    "audio_features",
                    "float32",
                    &[Some(1), Some(80), Some(3000)],
                )],
                outputs: vec![
                    hybrid_graph_tensor(
                        "encoder_hidden_states",
                        "float32",
                        &[Some(1), Some(1500), Some(384)],
                    ),
                    hybrid_graph_tensor(
                        "present_key_cross_0",
                        "float32",
                        &[Some(1), Some(6), Some(1500), Some(64)],
                    ),
                    hybrid_graph_tensor(
                        "present_value_cross_0",
                        "float32",
                        &[Some(1), Some(6), Some(1500), Some(64)],
                    ),
                ],
            },
            decoder: ModelGraphInfo {
                inputs: vec![
                    hybrid_graph_tensor("input_ids", "int64", &[Some(1), None]),
                    hybrid_graph_tensor(
                        "past_key_self_0",
                        "float32",
                        &[Some(1), Some(6), None, Some(64)],
                    ),
                    hybrid_graph_tensor(
                        "past_value_self_0",
                        "float32",
                        &[Some(1), Some(6), None, Some(64)],
                    ),
                    hybrid_graph_tensor(
                        "past_key_cross_0",
                        "float32",
                        &[Some(1), Some(6), Some(1500), Some(64)],
                    ),
                    hybrid_graph_tensor(
                        "past_value_cross_0",
                        "float32",
                        &[Some(1), Some(6), Some(1500), Some(64)],
                    ),
                ],
                outputs: vec![
                    hybrid_graph_tensor("logits", "float32", &[Some(1), None, Some(51865)]),
                    hybrid_graph_tensor(
                        "present_key_self_0",
                        "float32",
                        &[Some(1), Some(6), None, Some(64)],
                    ),
                    hybrid_graph_tensor(
                        "present_value_self_0",
                        "float32",
                        &[Some(1), Some(6), None, Some(64)],
                    ),
                ],
            },
        };

        let metadata = cfg
            .to_strict_encoder_decoder_pipeline_metadata(&graphs)
            .expect("strict encoder-decoder synth");
        let pipeline = metadata.pipeline.as_ref().expect("pipeline");
        onnx_genai_metadata::validate_pipeline_spec(pipeline).expect("valid pipeline spec");

        // Encoder + decoder components.
        assert_eq!(pipeline.models["encoder"].role, "encoder");
        assert_eq!(pipeline.models["decoder"].role, "decoder");

        // Audio prompt input surfaced on the encoder.
        let encoder_io = pipeline.models["encoder"].io.as_ref().expect("encoder io");
        assert_eq!(
            encoder_io.audio_features_input.as_deref(),
            Some("audio_features")
        );

        // Decoder self-KV grows; cross-KV is present as static routing.
        let decoder_io = pipeline.models["decoder"].io.as_ref().expect("decoder io");
        assert_eq!(decoder_io.logits_output.as_deref(), Some("logits"));
        assert_eq!(decoder_io.kv_update.as_deref(), Some("append"));
        assert_eq!(
            decoder_io.kv_inputs.as_deref(),
            Some(&["past_key_self_0", "past_value_self_0"].map(String::from)[..])
        );
        assert_eq!(
            decoder_io.kv_outputs.as_deref(),
            Some(&["present_key_self_0", "present_value_self_0"].map(String::from)[..])
        );
        assert_eq!(
            decoder_io.cross_kv_inputs.as_deref(),
            Some(&["past_key_cross_0", "past_value_cross_0"].map(String::from)[..])
        );
        assert_eq!(
            decoder_io.cross_kv_outputs.as_deref(),
            Some(&["present_key_cross_0", "present_value_cross_0"].map(String::from)[..])
        );

        // Cross-attention KV static routing is declared by the positional pairing
        // of the decoder's cross_kv_inputs (past_*_cross) with cross_kv_outputs
        // (the encoder-produced present_*_cross), computed ONCE by the encoder —
        // NOT recomputed each step and NOT a per-step dataflow edge. This decoder
        // has no encoder_hidden_states input, so no dataflow edge is synthesized.
        assert!(
            pipeline.dataflow.is_empty(),
            "cross-KV is stateful routing, not per-step edges: {:?}",
            pipeline.dataflow
        );

        assert!(matches!(
            pipeline.strategy.kind,
            onnx_genai_metadata::PipelineStrategyKind::Composite
        ));
    }

    // A faithful, trimmed synthetic derived from the real Microsoft
    // `nemotron_speech` genai_config.json (Conformer-Transducer / RNN-T):
    // a streaming Conformer encoder with cache state, an LSTM prediction
    // network (`targets` + `lstm_hidden_state`/`lstm_cell_state`, no attention
    // KV), a joint (joiner) network, and a Silero VAD. The multi-GB .onnx
    // weights are not needed — recognition is driven from the JSON alone.
    const NEMOTRON_TRANSDUCER_JSON: &str = r#"{
        "model": {
            "type": "nemotron_speech",
            "vocab_size": 13088,
            "subsampling_factor": 8,
            "blank_id": 13087,
            "max_symbols_per_step": 10,
            "encoder": {
                "filename": "encoder.onnx",
                "hidden_size": 1024,
                "num_hidden_layers": 24,
                "inputs": {
                    "audio_features": "audio_signal",
                    "cache_last_channel": "cache_last_channel",
                    "cache_last_time": "cache_last_time",
                    "cache_last_channel_len": "cache_last_channel_len",
                    "lang_id": "lang_id"
                },
                "outputs": {
                    "encoder_outputs": "outputs",
                    "output_lengths": "encoded_lengths",
                    "cache_last_channel_next": "cache_last_channel_next",
                    "cache_last_time_next": "cache_last_time_next",
                    "cache_last_channel_len_next": "cache_last_channel_len_next"
                }
            },
            "decoder": {
                "filename": "decoder.onnx",
                "hidden_size": 640,
                "num_hidden_layers": 2,
                "inputs": {
                    "targets": "targets",
                    "lstm_hidden_state": "h_in",
                    "lstm_cell_state": "c_in"
                },
                "outputs": {
                    "outputs": "decoder_output",
                    "lstm_hidden_state": "h_out",
                    "lstm_cell_state": "c_out"
                }
            },
            "joiner": {
                "filename": "joint.onnx",
                "inputs": {
                    "encoder_outputs": "encoder_output",
                    "decoder_outputs": "decoder_output"
                },
                "outputs": { "logits": "joint_output" }
            },
            "vad": {
                "filename": "silero_vad.onnx",
                "threshold": 0.3
            }
        }
    }"#;

    #[test]
    fn nemotron_transducer_is_not_encoder_decoder() {
        let cfg: GenAiConfig = serde_json::from_str(NEMOTRON_TRANSDUCER_JSON).unwrap();
        // Detected structurally as a transducer even though it declares
        // `model.encoder` (which alone would look like an encoder-decoder).
        assert!(cfg.is_transducer());
        assert_eq!(cfg.shape(), ModelShape::Transducer);
        assert_ne!(cfg.shape(), ModelShape::EncoderDecoder);
    }

    #[test]
    fn nemotron_transducer_declines_instead_of_fabricating_cross_kv() {
        let cfg: GenAiConfig = serde_json::from_str(NEMOTRON_TRANSDUCER_JSON).unwrap();
        // The non-strict synthesis path (the auto-detection fallback) must NOT
        // fabricate a Whisper-style encoder-decoder spec (with default
        // `input_ids`/`logits` ports and non-existent `past_key_values.*` /
        // `present.*` cross/self KV). It declines with the honest family error.
        let error = cfg
            .to_inference_metadata(None)
            .expect_err("transducer must not synthesize a pipeline");
        match error {
            GenAiConfigError::UnsupportedPipelineFamily { family, .. } => {
                assert_eq!(family, "RNN-T transducer");
            }
            other => panic!("expected UnsupportedPipelineFamily, got {other:?}"),
        }
    }

    #[test]
    fn nemotron_transducer_strict_from_dir_declines() {
        // The strict encoder-decoder loader entry point declines a transducer
        // directory explicitly rather than returning Ok(None) or fabricating.
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("target")
            .join("test-tmp")
            .join(format!(
                "nemotron_transducer_{}_{}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join(GENAI_CONFIG_FILE), NEMOTRON_TRANSDUCER_JSON).unwrap();
        let graphs = EncoderDecoderGraphInfo::default();
        let result = encoder_decoder_pipeline_inference_metadata_from_dir(&dir, &graphs);
        std::fs::remove_dir_all(&dir).ok();
        match result {
            Err(GenAiConfigError::UnsupportedPipelineFamily { family, .. }) => {
                assert_eq!(family, "RNN-T transducer");
            }
            other => panic!("expected UnsupportedPipelineFamily, got {other:?}"),
        }
    }

    #[test]
    fn transducer_detected_from_lstm_decoder_without_joiner() {
        // Even without a `joiner` section, an LSTM prediction network (targets +
        // LSTM hidden/cell state, no attention KV) is a transducer signal.
        let json = r#"{
            "model": {
                "type": "some_transducer",
                "encoder": {
                    "filename": "encoder.onnx",
                    "inputs": { "audio_features": "audio_signal" },
                    "outputs": { "encoder_outputs": "outputs" }
                },
                "decoder": {
                    "filename": "decoder.onnx",
                    "num_hidden_layers": 2,
                    "inputs": {
                        "targets": "targets",
                        "lstm_hidden_state": "h_in",
                        "lstm_cell_state": "c_in"
                    },
                    "outputs": { "outputs": "decoder_output" }
                }
            }
        }"#;
        let cfg: GenAiConfig = serde_json::from_str(json).unwrap();
        assert!(cfg.is_transducer());
        assert_eq!(cfg.shape(), ModelShape::Transducer);
    }

    #[test]
    fn whisper_still_classifies_as_encoder_decoder_not_transducer() {
        // No regression: a real cross-attention encoder-decoder (Whisper) is
        // still EncoderDecoder and is never mistaken for a transducer.
        let cfg: GenAiConfig = serde_json::from_str(WHISPER_JSON).unwrap();
        assert!(!cfg.is_transducer());
        assert_eq!(cfg.shape(), ModelShape::EncoderDecoder);
    }

    #[test]
    fn phi3v_and_decoder_pipeline_are_not_transducers() {
        // No regression for the other shapes.
        let vlm: GenAiConfig = serde_json::from_str(PHI3V_JSON).unwrap();
        assert!(!vlm.is_transducer());
        assert_eq!(vlm.shape(), ModelShape::Multimodal);
        let pipe: GenAiConfig = serde_json::from_str(DECODER_PIPELINE_JSON).unwrap();
        assert!(!pipe.is_transducer());
        assert_eq!(pipe.shape(), ModelShape::DecoderPipeline);
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
        assert_eq!(
            tokens.eos_token_id.as_deref(),
            Some(&[151_645, 151_643][..])
        );
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
