//! Low-level EAGLE-3 draft-head execution built on ORT.
//!
//! The head consumes a target token embedding, concatenated low/mid/high target
//! hidden states, and the previously recycled draft hidden state. It produces
//! draft logits plus the hidden state to recycle into the next autoregressive
//! draft step.

#![allow(clippy::arc_with_non_send_sync)]

use std::collections::HashMap;
use std::sync::Arc;

use crate::{DataType, IoBinding, MemoryInfo, OrtError, Result, Session, TensorInfo, Value};

/// Introspected EAGLE-3 draft-head graph signature.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Eagle3HeadSignature {
    /// Token embedding and recycled-hidden width.
    pub hidden_size: usize,
    /// Width of the concatenated target hidden-state input.
    pub fused_hidden_size: usize,
    /// Number of logits emitted by the draft head.
    pub draft_vocab_size: usize,
    /// Number of key/value heads in the draft head's cache.
    pub kv_heads: usize,
    /// Per-head dimension in the draft head's cache.
    pub head_dim: usize,
    /// Number of key/value cache layers.
    pub layers: usize,
    /// Cache tensor element type.
    pub dtype: DataType,
}

/// Strategy for EAGLE-3's internal draft key/value cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Eagle3DraftKvMode {
    /// Grow the draft cache across autoregressive proposal steps.
    GrowCache,
    /// Use an empty cache for every step and thread only `next_hidden`.
    HiddenThreaded,
}

/// Options for [`Eagle3DecodeSession`].
#[derive(Debug, Clone)]
pub struct Eagle3DecodeOptions {
    /// Internal draft-cache strategy.
    pub kv_mode: Eagle3DraftKvMode,
    /// Batch size. The speculative proposer currently uses one sequence.
    pub batch_size: i64,
}

impl Default for Eagle3DecodeOptions {
    fn default() -> Self {
        Self {
            kv_mode: Eagle3DraftKvMode::HiddenThreaded,
            batch_size: 1,
        }
    }
}

/// Outputs from one EAGLE-3 draft-head forward.
#[derive(Debug, Clone, PartialEq)]
pub struct Eagle3StepOutput {
    /// Last-position draft logits.
    pub logits: Vec<f32>,
    /// Hidden state to recycle into the next draft step.
    pub hidden: Vec<f32>,
}

#[derive(Debug, Clone)]
struct Eagle3KvPair {
    past: String,
    present: String,
    input: TensorInfo,
    seq_axis: usize,
}

/// Stateful runner for one EAGLE-3 draft-head forward at a time.
pub struct Eagle3DecodeSession<'a> {
    session: &'a Session,
    binding: IoBinding,
    signature: Eagle3HeadSignature,
    mode: Eagle3DraftKvMode,
    batch_size: i64,
    kv_pairs: Vec<Eagle3KvPair>,
    current_kv: HashMap<String, Arc<Value>>,
    kv_len: usize,
    embeds_input: String,
    fused_hidden_input: String,
    recycled_hidden_input: String,
    mask_input: Option<String>,
    position_input: Option<String>,
    logits_output: String,
    hidden_output: String,
}

impl<'a> Eagle3DecodeSession<'a> {
    /// Detect an EAGLE-3 draft-head signature from graph I/O.
    pub fn detect(session: &Session) -> Result<Option<Eagle3HeadSignature>> {
        Ok(detect_eagle3_head(session)?.map(|(signature, _, _)| signature))
    }

    /// Create a stateful EAGLE-3 draft-head runner.
    pub fn new(session: &'a Session, options: Eagle3DecodeOptions) -> Result<Self> {
        let (signature, kv_pairs, io) = detect_eagle3_head(session)?.ok_or_else(|| {
            OrtError::InvalidArgument(
                "model is not an EAGLE-3 head (needs inputs_embeds, fused_hidden, \
                 recycled_hidden, draft_logits, and next_hidden)"
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
            fused_hidden_input: io.fused_hidden_input,
            recycled_hidden_input: io.recycled_hidden_input,
            mask_input: io.mask_input,
            position_input: io.position_input,
            logits_output: io.logits_output,
            hidden_output: io.hidden_output,
        })
    }

    /// The introspected head signature.
    pub fn signature(&self) -> &Eagle3HeadSignature {
        &self.signature
    }

    /// Selected draft-cache strategy.
    pub fn mode(&self) -> Eagle3DraftKvMode {
        self.mode
    }

    /// Current internal draft-cache length.
    pub fn past_len(&self) -> usize {
        self.kv_len
    }

    /// Drop all internal draft state.
    pub fn reset(&mut self) {
        self.current_kv.clear();
        self.kv_len = 0;
    }

    /// Rewind the internal draft cache to an accepted prefix.
    pub fn rewind(&mut self, target_len: usize) -> Result<()> {
        if target_len > self.kv_len {
            return Err(OrtError::InvalidArgument(format!(
                "cannot rewind EAGLE-3 cache from {} to larger length {}",
                self.kv_len, target_len
            )));
        }
        if self.mode == Eagle3DraftKvMode::HiddenThreaded {
            self.kv_len = target_len;
            return Ok(());
        }
        if target_len == 0 {
            self.reset();
            return Ok(());
        }
        let mut rewound = HashMap::with_capacity(self.current_kv.len());
        for pair in &self.kv_pairs {
            let value = self.current_kv.get(&pair.past).ok_or_else(|| {
                OrtError::InvalidArgument(format!("missing EAGLE-3 KV tensor '{}'", pair.past))
            })?;
            let mut shape = value.shape().to_vec();
            shape[pair.seq_axis] = i64::try_from(target_len)
                .map_err(|_| OrtError::InvalidArgument("rewind length exceeds i64".into()))?;
            let prefix_is_contiguous = shape.iter().take(pair.seq_axis).all(|&dim| dim == 1);
            if !prefix_is_contiguous {
                return Err(OrtError::InvalidArgument(
                    "EAGLE-3 KV rewind requires a leading batch of 1".into(),
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

    /// Run one draft-head forward.
    ///
    /// Inputs are row-major tensors with shapes `[1, T, H]`, `[1, T, F]`,
    /// and `[1, T, H]`, respectively.
    pub fn step(
        &mut self,
        inputs_embeds: &[f32],
        fused_hidden: &[f32],
        recycled_hidden: &[f32],
        position_start: i64,
    ) -> Result<Eagle3StepOutput> {
        let hidden = self.signature.hidden_size;
        if inputs_embeds.is_empty() || !inputs_embeds.len().is_multiple_of(hidden) {
            return Err(OrtError::InvalidArgument(format!(
                "inputs_embeds length {} must be a non-zero multiple of hidden {hidden}",
                inputs_embeds.len()
            )));
        }
        let seq_len = inputs_embeds.len() / hidden;
        if recycled_hidden.len() != inputs_embeds.len() {
            return Err(OrtError::InvalidArgument(
                "recycled_hidden length must match inputs_embeds length".into(),
            ));
        }
        let expected_fused = seq_len * self.signature.fused_hidden_size;
        if fused_hidden.len() != expected_fused {
            return Err(OrtError::InvalidArgument(format!(
                "fused_hidden length {} != sequence {seq_len} * fused hidden {}",
                fused_hidden.len(),
                self.signature.fused_hidden_size
            )));
        }

        let seq_i64 = i64::try_from(seq_len)
            .map_err(|_| OrtError::InvalidArgument("seq_len exceeds i64".into()))?;
        let embeds =
            Value::from_slice_f32(inputs_embeds, &[self.batch_size, seq_i64, hidden as i64])?;
        let fused = Value::from_slice_f32(
            fused_hidden,
            &[
                self.batch_size,
                seq_i64,
                self.signature.fused_hidden_size as i64,
            ],
        )?;
        let recycled =
            Value::from_slice_f32(recycled_hidden, &[self.batch_size, seq_i64, hidden as i64])?;

        let past = if self.mode == Eagle3DraftKvMode::GrowCache {
            self.kv_len
        } else {
            0
        };
        let total = past + seq_len;
        let mask = if self.mask_input.is_some() {
            let data = vec![1i64; self.batch_size as usize * total];
            Some(Value::from_slice_i64(
                &data,
                &[self.batch_size, total as i64],
            )?)
        } else {
            None
        };
        let positions = if self.position_input.is_some() {
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
        self.binding.bind_input(&self.fused_hidden_input, &fused)?;
        self.binding
            .bind_input(&self.recycled_hidden_input, &recycled)?;
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
        let present_to_past = self
            .kv_pairs
            .iter()
            .map(|pair| (pair.present.as_str(), pair.past.as_str()))
            .collect::<HashMap<_, _>>();
        let mut next_kv = HashMap::with_capacity(self.kv_pairs.len());
        let mut logits = None;
        let mut next_hidden = None;
        for (name, value) in self.session.output_names().iter().zip(outputs) {
            if *name == self.logits_output {
                logits = Some(last_row_f32(&value, self.signature.draft_vocab_size)?);
            } else if *name == self.hidden_output {
                next_hidden = Some(last_row_f32(&value, self.signature.hidden_size)?);
            } else if let Some(past_name) = present_to_past.get(name.as_str()) {
                next_kv.insert((*past_name).to_string(), Arc::new(value));
            }
        }
        if self.mode == Eagle3DraftKvMode::GrowCache {
            self.current_kv = next_kv;
            self.kv_len = total;
        }
        Ok(Eagle3StepOutput {
            logits: logits.ok_or_else(|| {
                OrtError::InvalidArgument("EAGLE-3 head did not produce draft_logits".into())
            })?,
            hidden: next_hidden.ok_or_else(|| {
                OrtError::InvalidArgument("EAGLE-3 head did not produce next_hidden".into())
            })?,
        })
    }

    fn bind_kv_inputs(&mut self) -> Result<()> {
        for pair in &self.kv_pairs {
            let value = if self.mode == Eagle3DraftKvMode::GrowCache
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

struct Eagle3Io {
    embeds_input: String,
    fused_hidden_input: String,
    recycled_hidden_input: String,
    mask_input: Option<String>,
    position_input: Option<String>,
    logits_output: String,
    hidden_output: String,
}

fn detect_eagle3_head(
    session: &Session,
) -> Result<Option<(Eagle3HeadSignature, Vec<Eagle3KvPair>, Eagle3Io)>> {
    let input = |target: &str| {
        session
            .inputs()
            .iter()
            .find(|input| matches_name(&input.name, target))
    };
    let output = |target: &str| {
        session
            .outputs()
            .iter()
            .find(|output| matches_name(&output.name, target))
    };
    let (Some(embeds), Some(fused), Some(recycled), Some(logits), Some(hidden)) = (
        input("inputs_embeds"),
        input("fused_hidden"),
        input("recycled_hidden"),
        output("draft_logits"),
        output("next_hidden"),
    ) else {
        return Ok(None);
    };
    for info in [embeds, fused, recycled, logits, hidden] {
        if info.dtype != DataType::Float32 {
            return Err(OrtError::InvalidArgument(format!(
                "EAGLE-3 tensor '{}' must be Float32, got {:?}",
                info.name, info.dtype
            )));
        }
    }
    let hidden_size = last_positive_dim(&embeds.shape).ok_or_else(|| {
        OrtError::InvalidArgument("inputs_embeds must have a static hidden dimension".into())
    })?;
    if last_positive_dim(&recycled.shape) != Some(hidden_size)
        || last_positive_dim(&hidden.shape) != Some(hidden_size)
    {
        return Err(OrtError::InvalidArgument(
            "EAGLE-3 inputs_embeds, recycled_hidden, and next_hidden widths must match".into(),
        ));
    }
    let fused_hidden_size = last_positive_dim(&fused.shape).ok_or_else(|| {
        OrtError::InvalidArgument("fused_hidden must have a static last dimension".into())
    })?;
    let draft_vocab_size = last_positive_dim(&logits.shape).ok_or_else(|| {
        OrtError::InvalidArgument("draft_logits must have a static vocabulary dimension".into())
    })?;
    let kv_pairs = infer_eagle3_kv_pairs(session)?;
    let (kv_heads, head_dim, dtype) = if let Some(pair) = kv_pairs.first() {
        let shape = &pair.input.shape;
        (
            usize::try_from(shape[1].max(1)).unwrap_or(1),
            usize::try_from(shape[shape.len() - 1].max(1)).unwrap_or(1),
            pair.input.dtype,
        )
    } else {
        (0, 0, DataType::Float32)
    };
    let signature = Eagle3HeadSignature {
        hidden_size,
        fused_hidden_size,
        draft_vocab_size,
        kv_heads,
        head_dim,
        layers: kv_pairs
            .iter()
            .filter(|pair| kv_suffix(&pair.present).is_some_and(|suffix| suffix.ends_with("key")))
            .count(),
        dtype,
    };
    let io = Eagle3Io {
        embeds_input: embeds.name.clone(),
        fused_hidden_input: fused.name.clone(),
        recycled_hidden_input: recycled.name.clone(),
        mask_input: input("attention_mask").map(|info| info.name.clone()),
        position_input: input("position_ids").map(|info| info.name.clone()),
        logits_output: logits.name.clone(),
        hidden_output: hidden.name.clone(),
    };
    Ok(Some((signature, kv_pairs, io)))
}

fn infer_eagle3_kv_pairs(session: &Session) -> Result<Vec<Eagle3KvPair>> {
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
                "EAGLE-3 KV input '{}' must be Float32 or Float16, got {:?}",
                input.name, input.dtype
            )));
        }
        if input.shape.len() < 3 {
            return Err(OrtError::InvalidArgument(format!(
                "EAGLE-3 KV input '{}' has unsupported shape {:?}",
                input.name, input.shape
            )));
        }
        pairs.push(Eagle3KvPair {
            past: input.name.clone(),
            present: output.name.clone(),
            seq_axis: input.shape.len() - 2,
            input: input.clone(),
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
                "cannot infer static dimension {axis} for empty EAGLE-3 KV input '{}'",
                info.name
            )));
        };
        shape.push(value);
    }
    Value::empty(&shape, info.dtype)
}

fn last_row_f32(value: &Value, width: usize) -> Result<Vec<f32>> {
    let data = value.to_vec_f32()?;
    if data.len() < width || !data.len().is_multiple_of(width) {
        return Err(OrtError::InvalidArgument(format!(
            "EAGLE-3 output length {} is not a positive multiple of width {width}",
            data.len()
        )));
    }
    Ok(data[data.len() - width..].to_vec())
}

fn matches_name(name: &str, target: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    lower == target || lower.ends_with(&format!(".{target}"))
}

fn last_positive_dim(shape: &[i64]) -> Option<usize> {
    shape
        .last()
        .filter(|&&dim| dim > 0)
        .and_then(|&dim| usize::try_from(dim).ok())
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
