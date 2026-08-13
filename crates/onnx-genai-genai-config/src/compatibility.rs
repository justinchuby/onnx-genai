use onnx_genai_metadata::{InferenceMetadata, SCHEMA_VERSION};
use serde_json::{Map, Value, json};
use std::collections::BTreeSet;

use crate::*;

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
    /// group size 1. This determines the semantic attention description only;
    /// it does not select a KV storage or execution path.
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
    /// `kv_native_dtype` is retained for API compatibility with callers that
    /// inspect decoder graph state. It does not select inference execution or
    /// emit KV storage metadata.
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
        _kv_native_dtype: Option<&str>,
        decoder_graph: Option<&ModelGraphInfo>,
    ) -> Result<InferenceMetadata, GenAiConfigError> {
        let shape = self.shape();

        if shape != ModelShape::SingleDecoder {
            return Err(incomplete(
                "composite packages require native inference_metadata with pipeline.workflow; \
                 legacy genai_config pipeline synthesis has been removed",
            ));
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
}

impl GenAiConfig {
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
        // Safety gate: only the recurrent-hybrid case the shape-inference path
        // cannot handle. Pure-dense decoders (no state pairs) keep `io = None`.
        if derived.state_pairs.is_empty() {
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
            state_pairs: Some(state_pairs),
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

pub(crate) fn required_str<'a>(
    value: Option<&'a str>,
    field: &str,
) -> Result<&'a str, GenAiConfigError> {
    value
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| incomplete(field))
}
