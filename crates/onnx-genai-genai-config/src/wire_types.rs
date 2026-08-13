use std::collections::BTreeMap;

use serde::Deserialize;

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
    /// Optional RNN-T joint (joiner) network fusing encoder + prediction-network
    /// outputs into per-step logits. Its presence marks a transducer topology,
    /// which is NOT an encoder-decoder (cross-attention) model.
    #[serde(default)]
    pub joiner: Option<GenAiJoiner>,
    /// Optional voice-activity-detection graph (e.g. Silero VAD) used by
    /// streaming transducer packages for segmentation.
    #[serde(default)]
    pub vad: Option<GenAiVad>,
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
    /// RNN-T prediction-network label input (previous non-blank token). Present
    /// instead of `input_ids` in transducer prediction networks.
    pub targets: Option<String>,
    /// RNN-T prediction-network LSTM hidden state input (`h_in`).
    pub lstm_hidden_state: Option<String>,
    /// RNN-T prediction-network LSTM cell state input (`c_in`).
    pub lstm_cell_state: Option<String>,
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

/// The `model.joiner` section (RNN-T joint network).
///
/// The joint network combines the encoder output and the prediction-network
/// (decoder) output into per-step logits over the vocabulary plus a blank
/// symbol. It has no analog in a cross-attention encoder-decoder model, so its
/// mere presence identifies a transducer package. Only the fields needed to
/// DETECT and describe the family are parsed; the joint is not yet executable.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct GenAiJoiner {
    #[serde(default)]
    pub filename: Option<String>,
    #[serde(default)]
    pub inputs: JoinerInputs,
    #[serde(default)]
    pub outputs: JoinerOutputs,
}

/// Joint-network graph input port names.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct JoinerInputs {
    pub encoder_outputs: Option<String>,
    pub decoder_outputs: Option<String>,
}

/// Joint-network graph output port names.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct JoinerOutputs {
    pub logits: Option<String>,
}

/// The `model.vad` section (voice-activity-detection front-end, e.g. Silero).
///
/// Only parsed so streaming transducer packages describe cleanly; VAD
/// segmentation is not part of the current inference-metadata contract.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct GenAiVad {
    #[serde(default)]
    pub filename: Option<String>,
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
    pub config_filename: Option<String>,
    #[serde(default)]
    pub spatial_merge_size: Option<usize>,
    #[serde(default)]
    pub patch_size: Option<usize>,
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
    pub past_present_share_buffer: Option<bool>,
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
