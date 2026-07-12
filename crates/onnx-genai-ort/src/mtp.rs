//! Low-level Multi-Token Prediction (MTP) head execution built on ORT.
//!
//! An MTP head is a small decoder module distinct from the target model. It
//! consumes a hidden state plus an embedded token and produces the next hidden
//! state (`mtp_hidden`).
//!
//! The MTP head graph exposes no `logits` output and no embedding/LM-head
//! weights. [`MtpDecodeSession`] owns only one head forward pass plus its KV
//! buffer state and rewind cursor. The engine owns embedding, LM-head
//! projection, token selection, and the multi-step proposal loop.

//! ## Head I/O (fixture `tests/fixtures/tiny-qwen35-mtp/`)
//!
//! Inputs:
//! - `inputs_embeds`         `f32 [B, T, H]`  — embedding of the previous token
//! - `hidden_states`         `f32 [B, T, H]`  — target/prev-step hidden state
//! - `attention_mask`        `i64 [B, P+T]`
//! - `position_ids`          `i64 [B, T]`
//! - `past_key_values.0.key` `f32 [B, KVH, P, D]`
//! - `past_key_values.0.value` `f32 [B, KVH, P, D]`
//!
//! Outputs:
//! - `mtp_hidden`            `f32 [B, T, H]`
//! - `present.0.key`         `f32 [B, KVH, P+T, D]`
//! - `present.0.value`       `f32 [B, KVH, P+T, D]`

#![allow(clippy::arc_with_non_send_sync)]
// ORT Values are session-owned handles. These Arcs provide shared ownership inside
// one decode session; they are not used to move Values across threads.

use std::collections::HashMap;
use std::sync::Arc;

use crate::{DataType, IoBinding, MemoryInfo, OrtError, Result, Session, TensorInfo, Value};

/// Introspected MTP-head graph signature.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MtpHeadSignature {
    /// Model hidden size `H` (last dim of `inputs_embeds` / `mtp_hidden`).
    pub hidden_size: usize,
    /// Number of key/value heads `KVH` in the head's own cache.
    pub kv_heads: usize,
    /// Per-head dimension `D` of the head's own cache.
    pub head_dim: usize,
    /// Number of KV layers in the head (single-layer for Qwen3.6 MTP).
    pub layers: usize,
    /// KV tensor element type.
    pub dtype: DataType,
}

/// Strategy for the MTP head's own key/value cache while chaining `k` drafts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MtpDraftKvMode {
    /// Grow the head's cache across draft steps by rebinding each `present.*`
    /// output as the next step's `past_key_values.*` input (zero-copy). Correct
    /// for full-size MTP heads whose attention mask supports `P > 0`.
    GrowCache,
    /// Run every draft step with an empty past and a single token, threading
    /// state solely through the `hidden_states` input. Required by heads whose
    /// tiny/degenerate mask subgraph only executes for `past_len = 0, seq = 1`
    /// (the `tiny-qwen35-mtp` fixture, Mobius `da92170`).
    HiddenThreaded,
}

/// Options for [`MtpDecodeSession`].
#[derive(Debug, Clone)]
pub struct MtpDecodeOptions {
    /// Draft-cache strategy. Defaults to [`MtpDraftKvMode::HiddenThreaded`],
    /// which is safe for both the tiny fixture and any head that only needs the
    /// hidden state threaded forward.
    pub kv_mode: MtpDraftKvMode,
    /// Batch size for the head forward. Speculation uses 1.
    pub batch_size: i64,
}

impl Default for MtpDecodeOptions {
    fn default() -> Self {
        Self {
            kv_mode: MtpDraftKvMode::HiddenThreaded,
            batch_size: 1,
        }
    }
}

#[derive(Debug, Clone)]
struct MtpKvPair {
    past: String,
    present: String,
    input: TensorInfo,
    seq_axis: usize,
}

/// Stateful runner for an MTP-head ONNX graph.
///
/// Holds the head's own single-layer KV state (when growing) and runs one
/// forward step at a time. It does not select tokens or drive a proposal loop.
pub struct MtpDecodeSession<'a> {
    session: &'a Session,
    binding: IoBinding,
    signature: MtpHeadSignature,
    mode: MtpDraftKvMode,
    batch_size: i64,
    kv_pairs: Vec<MtpKvPair>,
    current_kv: HashMap<String, Arc<Value>>,
    kv_len: usize,
    embeds_input: String,
    hidden_input: String,
    mask_input: Option<String>,
    position_input: Option<String>,
    hidden_output: String,
}

impl<'a> MtpDecodeSession<'a> {
    /// Detect an MTP-head signature from graph I/O, if present.
    pub fn detect(session: &Session) -> Result<Option<MtpHeadSignature>> {
        Ok(detect_mtp_head(session)?.map(|(signature, _, _)| signature))
    }

    /// Create an MTP decode session from a head graph.
    pub fn new(session: &'a Session, options: MtpDecodeOptions) -> Result<Self> {
        let (signature, kv_pairs, io) = detect_mtp_head(session)?.ok_or_else(|| {
            OrtError::InvalidArgument(
                "model is not an MTP head (needs inputs_embeds + hidden_states inputs and an \
                 mtp_hidden output)"
                    .into(),
            )
        })?;
        Ok(Self {
            session,
            binding: IoBinding::new(session)?,
            signature,
            mode: options.kv_mode,
            batch_size: options.batch_size,
            kv_pairs,
            current_kv: HashMap::new(),
            kv_len: 0,
            embeds_input: io.embeds_input,
            hidden_input: io.hidden_input,
            mask_input: io.mask_input,
            position_input: io.position_input,
            hidden_output: io.hidden_output,
        })
    }

    /// The introspected head signature.
    pub fn signature(&self) -> &MtpHeadSignature {
        &self.signature
    }

    /// Selected draft-cache strategy.
    pub fn mode(&self) -> MtpDraftKvMode {
        self.mode
    }

    /// Current head KV length (always 0 in [`MtpDraftKvMode::HiddenThreaded`]).
    pub fn past_len(&self) -> usize {
        self.kv_len
    }

    /// Drop head KV state and reset the cursor.
    pub fn reset(&mut self) {
        self.current_kv.clear();
        self.kv_len = 0;
    }

    /// Rewind the head cache to `target_len` (verify/reject support).
    ///
    /// Only meaningful for [`MtpDraftKvMode::GrowCache`]; in
    /// [`MtpDraftKvMode::HiddenThreaded`] the cache is empty so this only
    /// validates `target_len == 0`.
    pub fn rewind(&mut self, target_len: usize) -> Result<()> {
        if target_len > self.kv_len {
            return Err(OrtError::InvalidArgument(format!(
                "cannot rewind MTP cache from {} to larger length {}",
                self.kv_len, target_len
            )));
        }
        if self.mode == MtpDraftKvMode::HiddenThreaded {
            self.kv_len = target_len;
            return Ok(());
        }
        if target_len == 0 {
            self.current_kv.clear();
            self.kv_len = 0;
            return Ok(());
        }
        let mut rewound = HashMap::with_capacity(self.current_kv.len());
        for pair in &self.kv_pairs {
            let value = self.current_kv.get(&pair.past).ok_or_else(|| {
                OrtError::InvalidArgument(format!("missing MTP KV tensor '{}'", pair.past))
            })?;
            let mut shape = value.shape().to_vec();
            shape[pair.seq_axis] = i64::try_from(target_len)
                .map_err(|_| OrtError::InvalidArgument("rewind length exceeds i64".into()))?;
            let prefix_is_contiguous = shape.iter().take(pair.seq_axis).all(|&dim| dim == 1);
            if !prefix_is_contiguous {
                return Err(OrtError::InvalidArgument(
                    "MTP KV rewind requires a leading batch of 1".into(),
                ));
            }
            rewound.insert(
                pair.past.clone(),
                Arc::new(Value::alias_with_shape(Arc::clone(value), &shape)?),
            );
        }
        self.current_kv = rewound;
        self.kv_len = target_len;
        Ok(())
    }

    /// Run one MTP-head forward and return `mtp_hidden` (`seq_len * H` floats).
    ///
    /// `inputs_embeds` and `hidden_states` are row-major `[1, seq_len, H]`.
    /// `position_start` is the position id of the first token in the step.
    pub fn step(
        &mut self,
        inputs_embeds: &[f32],
        hidden_states: &[f32],
        position_start: i64,
    ) -> Result<Vec<f32>> {
        let hidden = self.signature.hidden_size;
        if inputs_embeds.is_empty() || !inputs_embeds.len().is_multiple_of(hidden) {
            return Err(OrtError::InvalidArgument(format!(
                "inputs_embeds length {} must be a non-zero multiple of hidden {hidden}",
                inputs_embeds.len()
            )));
        }
        if hidden_states.len() != inputs_embeds.len() {
            return Err(OrtError::InvalidArgument(
                "hidden_states length must match inputs_embeds length".into(),
            ));
        }
        let seq_len = inputs_embeds.len() / hidden;
        let seq_i64 = i64::try_from(seq_len)
            .map_err(|_| OrtError::InvalidArgument("seq_len exceeds i64".into()))?;

        let embeds =
            Value::from_slice_f32(inputs_embeds, &[self.batch_size, seq_i64, hidden as i64])?;
        let hidden_value =
            Value::from_slice_f32(hidden_states, &[self.batch_size, seq_i64, hidden as i64])?;

        let past = if self.mode == MtpDraftKvMode::GrowCache {
            self.kv_len
        } else {
            0
        };
        let total = past + seq_len;
        let mask: Option<Value> = if self.mask_input.is_some() {
            let data = vec![1i64; self.batch_size as usize * total];
            Some(Value::from_slice_i64(
                &data,
                &[self.batch_size, total as i64],
            )?)
        } else {
            None
        };
        let positions: Option<Value> = if self.position_input.is_some() {
            let mut data = Vec::with_capacity(self.batch_size as usize * seq_len);
            for _ in 0..self.batch_size {
                for offset in 0..seq_len as i64 {
                    data.push(position_start + offset);
                }
            }
            Some(Value::from_slice_i64(&data, &[self.batch_size, seq_i64])?)
        } else {
            None
        };

        self.binding.clear()?;
        self.binding.bind_input(&self.embeds_input, &embeds)?;
        self.binding.bind_input(&self.hidden_input, &hidden_value)?;
        if let (Some(name), Some(value)) = (self.mask_input.as_ref(), mask.as_ref()) {
            self.binding.bind_input(name, value)?;
        }
        if let (Some(name), Some(value)) = (self.position_input.as_ref(), positions.as_ref()) {
            self.binding.bind_input(name, value)?;
        }
        self.bind_kv_inputs()?;

        for output in self.session.output_names() {
            self.binding
                .bind_output_to_device(output, &MemoryInfo::cpu()?)?;
        }
        self.session.run_with_binding(&self.binding)?;

        let outputs = self.binding.output_values()?;
        let mut mtp_hidden = None;
        let present_to_past = self
            .kv_pairs
            .iter()
            .map(|pair| (pair.present.as_str(), pair.past.as_str()))
            .collect::<HashMap<_, _>>();
        let mut next_kv = HashMap::with_capacity(self.kv_pairs.len());
        for (name, value) in self.session.output_names().iter().zip(outputs) {
            if *name == self.hidden_output {
                mtp_hidden = Some(value.to_vec_f32()?);
            } else if let Some(past_name) = present_to_past.get(name.as_str()) {
                next_kv.insert((*past_name).to_string(), Arc::new(value));
            }
        }

        if self.mode == MtpDraftKvMode::GrowCache {
            self.current_kv = next_kv;
            self.kv_len = total;
        }
        mtp_hidden
            .ok_or_else(|| OrtError::InvalidArgument("MTP head did not produce mtp_hidden".into()))
    }

    fn bind_kv_inputs(&mut self) -> Result<()> {
        for pair in &self.kv_pairs {
            let value = if self.mode == MtpDraftKvMode::GrowCache
                && let Some(value) = self.current_kv.get(&pair.past)
            {
                Arc::clone(value)
            } else {
                Arc::new(empty_past_value(&pair.input)?)
            };
            self.binding.bind_input(&pair.past, &value)?;
        }
        Ok(())
    }
}

struct MtpIo {
    embeds_input: String,
    hidden_input: String,
    mask_input: Option<String>,
    position_input: Option<String>,
    hidden_output: String,
}

fn detect_mtp_head(session: &Session) -> Result<Option<(MtpHeadSignature, Vec<MtpKvPair>, MtpIo)>> {
    let embeds_input = session
        .inputs()
        .iter()
        .find(|input| matches_name(&input.name, "inputs_embeds"));
    let hidden_input = session
        .inputs()
        .iter()
        .find(|input| matches_name(&input.name, "hidden_states"));
    let hidden_output = session
        .outputs()
        .iter()
        .find(|output| matches_name(&output.name, "mtp_hidden"));
    let (Some(embeds_input), Some(hidden_input), Some(hidden_output)) =
        (embeds_input, hidden_input, hidden_output)
    else {
        return Ok(None);
    };

    let hidden_size = last_positive_dim(&embeds_input.shape).ok_or_else(|| {
        OrtError::InvalidArgument("inputs_embeds must have a static hidden dimension".into())
    })?;

    let kv_pairs = infer_mtp_kv_pairs(session)?;
    let (kv_heads, head_dim, dtype) = if let Some(pair) = kv_pairs.first() {
        let shape = &pair.input.shape;
        let kv_heads = usize::try_from(shape[1].max(1)).unwrap_or(1);
        let head_dim = usize::try_from(shape[shape.len() - 1].max(1)).unwrap_or(1);
        (kv_heads, head_dim, pair.input.dtype)
    } else {
        (0, 0, DataType::Float32)
    };

    let mask_input = session
        .inputs()
        .iter()
        .find(|input| matches_name(&input.name, "attention_mask"))
        .map(|input| input.name.clone());
    let position_input = session
        .inputs()
        .iter()
        .find(|input| matches_name(&input.name, "position_ids"))
        .map(|input| input.name.clone());

    let signature = MtpHeadSignature {
        hidden_size,
        kv_heads,
        head_dim,
        layers: kv_pairs
            .iter()
            .filter(|pair| {
                kv_suffix(&pair.present)
                    .map(|suffix| suffix.ends_with("key"))
                    .unwrap_or(false)
            })
            .count(),
        dtype,
    };
    let io = MtpIo {
        embeds_input: embeds_input.name.clone(),
        hidden_input: hidden_input.name.clone(),
        mask_input,
        position_input,
        hidden_output: hidden_output.name.clone(),
    };
    Ok(Some((signature, kv_pairs, io)))
}

fn infer_mtp_kv_pairs(session: &Session) -> Result<Vec<MtpKvPair>> {
    let mut pairs = Vec::new();
    for output in session.outputs() {
        if !is_present_output(&output.name) {
            continue;
        }
        let Some(suffix) = kv_suffix(&output.name) else {
            continue;
        };
        let Some(input) = session
            .inputs()
            .iter()
            .find(|input| kv_suffix(&input.name).as_deref() == Some(suffix.as_str()))
        else {
            continue;
        };
        if input.dtype != DataType::Float32 && input.dtype != DataType::Float16 {
            return Err(OrtError::InvalidArgument(format!(
                "MTP KV input '{}' must be Float32 or Float16, got {:?}",
                input.name, input.dtype
            )));
        }
        if input.shape.len() < 3 {
            return Err(OrtError::InvalidArgument(format!(
                "MTP KV input '{}' has unsupported shape {:?}",
                input.name, input.shape
            )));
        }
        let seq_axis = input.shape.len() - 2;
        pairs.push(MtpKvPair {
            past: input.name.clone(),
            present: output.name.clone(),
            input: input.clone(),
            seq_axis,
        });
    }
    Ok(pairs)
}

fn empty_past_value(info: &TensorInfo) -> Result<Value> {
    let seq_axis = info.shape.len() - 2;
    let mut shape = Vec::with_capacity(info.shape.len());
    for (axis, &dim) in info.shape.iter().enumerate() {
        let value = if axis == 0 {
            1
        } else if axis == seq_axis {
            0
        } else if dim > 0 {
            dim
        } else {
            return Err(OrtError::InvalidArgument(format!(
                "cannot infer static dimension {axis} for empty MTP KV input '{}'",
                info.name
            )));
        };
        shape.push(value);
    }
    Value::empty(&shape, info.dtype)
}

fn matches_name(name: &str, target: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    lower == target || lower.ends_with(&format!(".{target}"))
}

fn last_positive_dim(shape: &[i64]) -> Option<usize> {
    shape.last().and_then(|&dim| usize::try_from(dim).ok())
}

fn is_present_output(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    lower.contains("present") && (lower.contains("key") || lower.contains("value"))
}

fn kv_suffix(name: &str) -> Option<String> {
    let lower = name.to_ascii_lowercase();
    for prefix in [
        "past_key_values.",
        "present_key_values.",
        "past.",
        "present.",
    ] {
        if let Some(suffix) = lower.strip_prefix(prefix) {
            return Some(suffix.to_string());
        }
    }
    None
}
