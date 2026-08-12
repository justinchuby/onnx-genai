use std::collections::BTreeSet;
use std::path::Path;

use onnx_genai_metadata::{InferenceMetadata, SCHEMA_VERSION, capabilities};
use onnx_genai_preprocess::image::ImagePreprocessor;
use serde::Deserialize;
use serde_json::{Map, Value, json};

use crate::*;

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

/// How strictly a fixed-state input/output port pair's ONNX shapes must agree
/// during decoder-state derivation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum StateShapeMatch {
    /// Every axis must match exactly, including symbolic vs concrete. Used by the
    /// config-driven `strict_decoder_state` path, whose contract is unchanged.
    Exact,
    /// Same rank, with symbolic (unknown) axes treated as wildcards: a concrete
    /// axis matches an unknown axis, only two differing concrete extents fail.
    /// Stock exports frequently leave a `present.*` recurrent-state shape fully
    /// symbolic (`[?,?,?]`) even though the paired `past_*` input carries the
    /// concrete running-state extent, so the config-free graph fallback must not
    /// reject that legitimate pairing.
    AllowSymbolic,
}

impl StateShapeMatch {
    fn shapes_pair(self, input: &[Option<usize>], output: &[Option<usize>]) -> bool {
        match self {
            StateShapeMatch::Exact => input == output,
            StateShapeMatch::AllowSymbolic => {
                input.len() == output.len()
                    && input.iter().zip(output).all(|(a, b)| match (a, b) {
                        (Some(a), Some(b)) => a == b,
                        _ => true,
                    })
            }
        }
    }
}

/// A single loop-carried recurrent state port pair (`conv_state`/
/// `recurrent_state`), threaded in→out and replaced wholesale each decode step
/// rather than appended along the sequence axis like growable KV.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DerivedStatePair {
    /// Graph input port carrying the state into the step.
    pub input: String,
    /// Graph output port carrying the updated state out of the step.
    pub output: String,
}

/// Decoder KV/state topology derived purely from an ONNX graph's port
/// inventory, without a `genai_config.json`. Returned by
/// [`GenAiConfig::derive_decoder_io_from_graph`]; consumed by the native
/// loader's auto-derive fallback when the sidecar declares no `io` block.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DerivedDecoderIo {
    /// Growable dense KV input ports (`past_key_values.%d.key`/`.value`), paired
    /// positionally with `kv_outputs`.
    pub kv_inputs: Vec<String>,
    /// Growable dense KV output ports (`present.%d.key`/`.value`).
    pub kv_outputs: Vec<String>,
    /// Fixed loop-carried recurrent state pairs (empty for pure-dense decoders).
    pub state_pairs: Vec<DerivedStatePair>,
    /// Canonical dtype spelling shared by every KV cache tensor.
    pub kv_dtype: String,
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
        let phases = models
            .keys()
            .map(|name| (name.clone(), run_on("every_step")))
            .collect();

        let mut pipeline = Map::new();
        pipeline.insert("models".into(), Value::Object(models));
        pipeline.insert("dataflow".into(), Value::Array(Vec::new()));
        pipeline.insert("strategy".into(), strategy);
        pipeline.insert("phases".into(), Value::Object(phases));
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
        // The split-VLM decoder is driven by `inputs_embeds` produced by the
        // embedding component (the vision encoder raises image features into the
        // same embedding stream); it declares no token-id input. Declare the
        // sequence source explicitly so decode resolves the embeds input rather
        // than defaulting to a (non-existent) token input. This mirrors the
        // text-only fallback (`to_strict_text_only_pipeline_metadata`).
        decoder_io.insert("sequence_source".into(), json!("inputs_embeds"));
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
                    "strategy": { "kind": "single_pass", "model": "vision_encoder" }
                },
                {
                    "name": "embed_tokens",
                    "strategy": { "kind": "single_pass", "model": "embedding" }
                },
                {
                    "name": "decode",
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
                "token_count_summary": vision_grid,
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

    /// Strict **text-only** decode-pipeline synthesis for a multimodal
    /// (embedding + decoder) compatibility package whose image path is unusable.
    ///
    /// A split VLM package pairs an `embedding` graph (token ids [+ optional
    /// image features] → `inputs_embeds`) with an `inputs_embeds`-driven
    /// `decoder`. When the package's declared image preprocessing is not
    /// representable by the runtime (see
    /// [`GenAiConfigError::UnrepresentablePreprocessing`]), the vision path
    /// cannot be honored — but text decode never touches vision. This synthesis
    /// produces the same embedding→decoder autoregressive pipeline with the
    /// vision component, image preprocessing, image dataflow, and grid-derived
    /// positions removed. It is driven purely by the package's declared modality
    /// shape (a split embedding+decoder package that can accept token ids
    /// without image features), never by a model name: any such package that is
    /// image-unusable synthesizes text decode the same way.
    ///
    /// Positions are declared with `linear_increment` continuation instead of
    /// the VLM `from_grid` program: for a pure-text sequence every multi-axis
    /// (mrope) coordinate stream advances identically with the sequence
    /// position, which `linear_increment` produces for any rank (`[t, t, …]`),
    /// so no processor grid summary is required. All decoder KV / loop-carried
    /// state facts are validated against the authoritative ONNX decoder graph
    /// exactly as the multimodal path does.
    pub(crate) fn to_strict_text_only_pipeline_metadata(
        &self,
        graphs: &PipelineGraphInfo,
    ) -> Result<InferenceMetadata, GenAiConfigError> {
        let embedding = required_ref(self.model.embedding.as_ref(), "model.embedding")?;
        let embedding_filename =
            required_str(embedding.filename.as_deref(), "model.embedding.filename")?;
        let decoder_filename = required_str(
            self.model.decoder.filename.as_deref(),
            "model.decoder.filename",
        )?;

        let embedding_tokens = required_str(
            embedding.inputs.input_ids.as_deref(),
            "model.embedding.inputs.input_ids",
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
        let past_present_share_buffer = required_copy(
            self.search.past_present_share_buffer,
            "search.past_present_share_buffer",
        )?;

        require_graph_input(&graphs.embedding, embedding_tokens, "embedding")?;
        let embedding_output_info =
            require_graph_output(&graphs.embedding, embedding_output, "embedding")?;
        let decoder_embeds_info = require_graph_input(&graphs.decoder, decoder_embeds, "decoder")?;
        require_graph_input(&graphs.decoder, decoder_mask, "decoder")?;
        let position_info = require_graph_input(&graphs.decoder, decoder_position, "decoder")?;
        require_graph_output(&graphs.decoder, decoder_logits, "decoder")?;

        require_same_dtype(
            embedding_output_info,
            decoder_embeds_info,
            "embedding-to-decoder dataflow",
        )?;

        let position_rank = position_info.dimensions.len();
        if position_rank == 0 {
            return Err(incomplete(format!(
                "decoder position input '{decoder_position}' declares a positive rank"
            )));
        }

        let DecoderStateMetadata {
            kv_inputs,
            kv_outputs,
            state_pairs,
            kv_dtype,
        } = self.strict_decoder_state(&graphs.decoder)?;
        let has_state_pairs = !state_pairs.is_empty();
        let multi_axis = position_rank > 1;

        let mut decoder_io = Map::new();
        // The decoder is driven by `inputs_embeds` produced by the embedding
        // component (text.onnx declares no token-id input); declare the sequence
        // source explicitly so decode resolves the embeds input rather than
        // defaulting to a (non-existent) token input.
        decoder_io.insert("sequence_source".into(), json!("inputs_embeds"));
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

        // Text decode drives the embedding with token ids only. A split VLM
        // embedding graph also declares an `image_features` input that the
        // vision path would supply; with no vision component it has no dataflow
        // edge, so it is declared as an optional input whose absent value is an
        // empty (zero image-token) tensor. The image-token axis collapses to 0
        // and any fixed feature width is preserved, so a pure-text prompt (no
        // image placeholder tokens) never gathers from it.
        let mut embedding_io = Map::new();
        embedding_io.insert("token_input".into(), json!(embedding_tokens));
        if let Some(image_features) = embedding.inputs.image_features.as_deref() {
            let image_info = require_graph_input(&graphs.embedding, image_features, "embedding")?;
            let absent_shape = image_info
                .dimensions
                .iter()
                .map(|dimension| json!(dimension.unwrap_or(0)))
                .collect::<Vec<_>>();
            let mut optional_inputs = Map::new();
            optional_inputs.insert(
                image_features.to_owned(),
                json!({
                    "presence": "image_features",
                    "absent": { "kind": "zeros", "shape": absent_shape }
                }),
            );
            embedding_io.insert("optional_inputs".into(), Value::Object(optional_inputs));
        }

        let mut models = Map::new();
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

        let dataflow = vec![edge_with_dtype(
            &format!("embedding.{embedding_output}"),
            &format!("decoder.{decoder_embeds}"),
            &embedding_output_info.dtype,
        )];
        let mut phases = Map::new();
        phases.insert("embedding".into(), run_on("every_step"));
        phases.insert("decoder".into(), run_on("every_step"));

        let strategy = json!({
            "kind": "composite",
            "stages": [
                {
                    "name": "embed_tokens",
                    "strategy": { "kind": "single_pass", "model": "embedding" }
                },
                {
                    "name": "decode",
                    "strategy": { "kind": "autoregressive", "decoder": "decoder" }
                }
            ]
        });

        // Pure-text positions: every coordinate stream advances with the
        // sequence position, which `linear_increment` yields for any rank. No
        // processor grid summary is read, and no section widths are needed
        // because there is no grid-derived coordinate program.
        let mut positions = Map::new();
        positions.insert("input".into(), json!(decoder_position));
        positions.insert("rank".into(), json!(position_rank));
        if multi_axis {
            positions.insert(
                "axes".into(),
                json!(
                    (0..position_rank)
                        .map(|axis| format!("axis_{axis}"))
                        .collect::<Vec<_>>()
                ),
            );
        }
        positions.insert("dtype".into(), json!(position_info.dtype));
        positions.insert("continuation".into(), json!("linear_increment"));

        let mut pipeline = Map::new();
        pipeline.insert("models".into(), Value::Object(models));
        pipeline.insert("dataflow".into(), Value::Array(dataflow));
        pipeline.insert("strategy".into(), strategy);
        pipeline.insert("phases".into(), Value::Object(phases));
        pipeline.insert("positions".into(), Value::Object(positions));

        let mut required_capabilities = vec![capabilities::POSITION_PROGRAM];
        if multi_axis {
            required_capabilities.push(capabilities::MULTI_AXIS_POSITIONS);
        }
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

        Ok(serde_json::from_value(Value::Object(root))?)
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
        let (encoder_input_field, encoder_input, encoder_input_role) = match (
            encoder.inputs.audio_features.as_deref(),
            encoder.inputs.input_ids.as_deref(),
        ) {
            (Some(audio), None) => (
                "model.encoder.inputs.audio_features",
                audio,
                "audio_features_input",
            ),
            (None, Some(ids)) => ("model.encoder.inputs.input_ids", ids, "token_input"),
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
        // The encoder prompt-input role (`audio_features_input` vs `token_input`)
        // is taken directly from WHICH explicit genai-config field the exporter
        // declared (`audio_features` vs `input_ids`), captured in the match
        // above — never re-derived by string-matching the port name.
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

        Self::decoder_state_from_patterns(
            graph,
            past_key,
            past_value,
            present_key,
            present_value,
            StateShapeMatch::Exact,
        )
    }

    /// Derive a decoder's KV/state topology from the ONNX graph interface using
    /// the four indexed `%d` key/value name patterns.
    ///
    /// This is the pure, config-independent core of
    /// [`strict_decoder_state`](Self::strict_decoder_state): it depends only on
    /// the graph inventory and the four pattern strings, so callers that lack a
    /// `genai_config.json` (e.g. stock native exports carrying only an
    /// `inference_metadata.yaml`) can reuse the exact same guarded derivation via
    /// [`derive_decoder_io_from_graph`](Self::derive_decoder_io_from_graph). The
    /// guard is inherited unchanged: only `suffix_tensor_map` (not
    /// `strict_indexed_kv`) finds the non-KV recurrent `state_pairs`, so
    /// cross-attention/Whisper KV is never misclassified as running state.
    fn decoder_state_from_patterns(
        graph: &ModelGraphInfo,
        past_key: &str,
        past_value: &str,
        present_key: &str,
        present_value: &str,
        state_shape_match: StateShapeMatch,
    ) -> Result<DecoderStateMetadata, GenAiConfigError> {
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
            if !state_shape_match.shapes_pair(&input.dimensions, &output.dimensions) {
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

    /// Derive a decoder's KV/state I/O topology directly from an ONNX graph's
    /// port inventory, using the conventional onnxruntime-genai key/value name
    /// patterns (`past_key_values.%d.key`/`.value` → `present.%d.key`/`.value`).
    ///
    /// This is the config-free entry point for stock native exports that ship an
    /// `inference_metadata.yaml` without an explicit `io` block (and no
    /// `genai_config.json`): the native loader calls it as an additive fallback
    /// so hybrid linear-attention decoders (which carry recurrent
    /// `conv_state`/`recurrent_state` ports the shape-inference path cannot
    /// classify) still load. It reuses the exact guarded derivation
    /// ([`decoder_state_from_patterns`](Self::decoder_state_from_patterns)), so
    /// dense cross-attention/Whisper KV is never misclassified as running state.
    ///
    /// Structural derivation failures are swallowed into `None` (mirroring
    /// [`graph_decoder_state`](Self::graph_decoder_state)) so a caller can fall
    /// back to its existing shape-inference path without regressing any model
    /// that loads today.
    pub fn derive_decoder_io_from_graph(graph: &ModelGraphInfo) -> Option<DerivedDecoderIo> {
        let metadata = Self::decoder_state_from_patterns(
            graph,
            "past_key_values.%d.key",
            "past_key_values.%d.value",
            "present.%d.key",
            "present.%d.value",
            StateShapeMatch::AllowSymbolic,
        )
        .ok()?;
        let state_pairs = metadata
            .state_pairs
            .iter()
            .map(|pair| {
                Some(DerivedStatePair {
                    input: pair.get("input")?.as_str()?.to_owned(),
                    output: pair.get("output")?.as_str()?.to_owned(),
                })
            })
            .collect::<Option<Vec<_>>>()?;
        Some(DerivedDecoderIo {
            kv_inputs: metadata.kv_inputs,
            kv_outputs: metadata.kv_outputs,
            state_pairs,
            kv_dtype: metadata.kv_dtype,
        })
    }

    /// Best-effort auto-derived [`ModelIoSpec`] for a stock decoder export whose
    /// sidecar declares no `io` block, built purely from an ONNX graph's port
    /// inventory.
    ///
    /// This is the single canonical glue shared by both auto-derive callers (the
    /// native decode driver's live-session `derive_fallback_io` and the engine
    /// loader's disk-graph `maybe_fill_hybrid_io_from_graph`): it reuses the
    /// guarded [`derive_decoder_io_from_graph`](Self::derive_decoder_io_from_graph)
    /// classifier, applies the recurrent-hybrid safety gate (non-empty
    /// `state_pairs`), binds the conventional non-KV ports by name-presence in the
    /// graph interface, and assembles the `ModelIoSpec`.
    ///
    /// Returns `None` (leaving the caller's `io = None` shape-inference path
    /// untouched) unless the derivation yields at least one recurrent state pair —
    /// the exact case the shape-inference path cannot resolve. Pure-dense decoders
    /// (no state pairs) always return `None`.
    pub fn derive_model_io_spec_from_graph(
        graph: &ModelGraphInfo,
    ) -> Option<onnx_genai_metadata::ModelIoSpec> {
        use onnx_genai_metadata::{LoopStatePair, ModelIoSpec};

        let derived = Self::derive_decoder_io_from_graph(graph)?;
        // Safety gate: derive only when the graph actually yielded KV ports.
        //
        // This used to require a non-empty `state_pairs`, i.e. the recurrent
        // hybrid case, on the reasoning that pure-dense decoders keep
        // `io = None` and use the shape-inference path. But the only caller
        // (`maybe_fill_hybrid_io_from_graph`) runs *after* a declared or
        // pattern-expanded `io` block has already been established, and returns
        // early when one exists. So reaching here with a dense graph means the
        // config produced no port contract at all, and returning `None` does not
        // preserve a working path — it leaves the model with no KV geometry and
        // fails the load with "per-layer KV page geometry is unknown".
        //
        // That is what blocks DeepSeek-V2 (MLA, #1012): its `genai_config.json`
        // declares a single `decoder.head_size: 128` and no `model.io`, while its
        // KV is asymmetric — key head_size 192 (qk_nope 128 + qk_rope 64), value
        // 128. A scalar cannot express that, but the graph shapes can, and
        // `kv_cache_bytes_for_tensors` already sums each tensor independently,
        // so asymmetry needs no new arithmetic once the specs exist.
        //
        // Gating on KV ports instead keeps the property that matters: we never
        // manufacture a geometry. A graph that yields no KV ports still returns
        // `None` and still fails loudly, rather than loading against a guessed
        // budget — the failure mode of #947.
        if derived.kv_inputs.is_empty() || derived.kv_outputs.is_empty() {
            return None;
        }
        let input_names: BTreeSet<&str> = graph.inputs.iter().map(|t| t.name.as_str()).collect();
        let output_names: BTreeSet<&str> = graph.outputs.iter().map(|t| t.name.as_str()).collect();
        let present_input = |name: &str| input_names.contains(name).then(|| name.to_owned());
        let present_output = |name: &str| output_names.contains(name).then(|| name.to_owned());
        let state_pairs = derived
            .state_pairs
            .into_iter()
            .map(|pair| LoopStatePair {
                input: pair.input,
                output: pair.output,
                init: Some("zeros".to_owned()),
                update: Some("replace".to_owned()),
            })
            .collect::<Vec<_>>();
        // A dense decoder has no recurrent state; say so with `None` rather than
        // an empty list, so downstream cannot read "declared, and empty" as
        // different from "not applicable".
        let state_pairs = (!state_pairs.is_empty()).then_some(state_pairs);
        Some(ModelIoSpec {
            sequence_source: None,
            kv_ownership: None,
            kv_layout: None,
            token_input: present_input("input_ids"),
            inputs_embeds_input: None,
            attention_mask_input: present_input("attention_mask"),
            position_ids_input: present_input("position_ids"),
            logits_output: present_output("logits"),
            hidden_output: None,
            kv_inputs: (!derived.kv_inputs.is_empty()).then_some(derived.kv_inputs),
            kv_outputs: (!derived.kv_outputs.is_empty()).then_some(derived.kv_outputs),
            encoder_hidden_states_input: None,
            audio_features_input: None,
            cross_kv_inputs: None,
            cross_kv_outputs: None,
            kv_update: None,
            state_pairs,
            optional_inputs: std::collections::BTreeMap::new(),
            static_cache: None,
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

pub(crate) fn incomplete(missing: impl Into<String>) -> GenAiConfigError {
    GenAiConfigError::IncompletePipeline {
        missing: missing.into(),
    }
}

/// Honest decline for a multimodal package whose declared image preprocessing
/// has no lossless runtime encoding (e.g. Qwen-style `smart_resize`). The image
/// path is refused rather than approximated, but the caller may fall back to
/// text-only decode via [`GenAiConfig::to_strict_text_only_pipeline_metadata`].
pub(crate) fn unrepresentable_preprocessing(detail: impl Into<String>) -> GenAiConfigError {
    GenAiConfigError::UnrepresentablePreprocessing {
        detail: detail.into(),
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
