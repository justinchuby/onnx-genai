//! Low-level shared-KV draft proposer execution built on ORT.
//!
//! The shared-KV proposer (originally introduced for Gemma4 `*-assistant` draft
//! models) is a small draft module distinct from both MTP and EAGLE-3. It owns
//! **no** key/value cache of its own; instead it reads slices of the *target*
//! model's paged KV cache through `shared_kv.*` inputs. It has its own internal
//! `lm_head`, so it emits full draft `logits` directly, plus a `projected_state`
//! that is threaded into the next step's `inputs_embeds`.
//!
//! ## Graph I/O (structural signature)
//!
//! Float tensors may be Float32 or Float16; the activation dtype is taken from
//! `inputs_embeds` and the shared-KV inputs must match it. Outputs are widened
//! to f32 on read, so the engine-facing API stays f32 regardless of graph dtype.
//!
//! Inputs:
//! - `inputs_embeds`                       `[B, q, 2*H]`
//!   concat of (previous `projected_state`, current `projected_state`).
//! - `position_ids`                        `i64 [B, q]`
//! - `attention_mask`                      `i64 [B, kv_len]`
//! - `shared_kv.<group>.key` / `.value`    `[B, kv_heads, kv_len, head_dim]`
//!   slices of the target model's KV buffer (one pair per attention type).
//!
//! Outputs:
//! - `logits`                              `[B, q, vocab]`
//! - `projected_state`                     `[B, q, H]`
//!
//! [`SharedKvProposerSession`] owns exactly one forward pass. It does not
//! select tokens, thread `projected_state`, or extract the target KV slices;
//! the engine owns that policy (mirroring the MTP/EAGLE-3 ownership split).

#![allow(clippy::arc_with_non_send_sync)]

use std::collections::BTreeMap;

use crate::{DataType, IoBinding, MemoryInfo, OrtError, Result, Session, Value};

/// A single shared-KV binding group discovered in an assistant graph.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SharedKvSpec {
    /// Group name, e.g. `sliding_attention` or `full_attention`.
    pub name: String,
    /// Assistant input name for this group's keys (`shared_kv.<name>.key`).
    pub key_input: String,
    /// Assistant input name for this group's values (`shared_kv.<name>.value`).
    pub value_input: String,
    /// Number of key/value heads in this slice (`shape[1]`).
    pub kv_heads: usize,
    /// Per-head dimension in this slice (`shape[3]`).
    pub head_dim: usize,
}

/// Introspected shared-KV proposer graph signature.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SharedKvProposerSignature {
    /// Target backbone hidden size `H` (last dim of `projected_state`).
    pub backbone_hidden_size: usize,
    /// Width of `inputs_embeds` (`2*H`, concat of prev/cur projected states).
    pub inputs_embeds_width: usize,
    /// Vocabulary size of the assistant's own `logits` output.
    pub vocab_size: usize,
    /// Shared-KV binding groups, ordered by input name.
    pub shared_kv: Vec<SharedKvSpec>,
    /// Element type of the assistant's float tensors.
    pub dtype: DataType,
}

/// Externally provided contents of one `shared_kv.<name>` binding.
///
/// The engine slices these from the target model's paged KV cache. `key` and
/// `value` are row-major `[kv_heads, kv_len, head_dim]` (batch 1).
#[derive(Debug, Clone, PartialEq)]
pub struct SharedKvInput {
    /// Group name matching a [`SharedKvSpec::name`].
    pub name: String,
    /// Row-major `[kv_heads, kv_len, head_dim]` key slice.
    pub key: Vec<f32>,
    /// Row-major `[kv_heads, kv_len, head_dim]` value slice.
    pub value: Vec<f32>,
    /// Number of key/value heads.
    pub kv_heads: usize,
    /// Number of KV positions in the slice.
    pub kv_len: usize,
    /// Per-head dimension.
    pub head_dim: usize,
}

/// Output of one assistant forward.
#[derive(Debug, Clone, PartialEq)]
pub struct SharedKvProposerStepOutput {
    /// Last-position draft logits (`vocab` floats).
    pub logits: Vec<f32>,
    /// Last-position projected state (`H` floats), threaded into the next step.
    pub projected_state: Vec<f32>,
}

struct SharedKvProposerIo {
    embeds_input: String,
    mask_input: Option<String>,
    position_input: Option<String>,
    logits_output: String,
    projected_state_output: String,
}

/// Stateful runner for one shared-KV proposer forward at a time.
pub struct SharedKvProposerSession<'a> {
    session: &'a Session,
    binding: IoBinding,
    signature: SharedKvProposerSignature,
    batch_size: i64,
    embeds_input: String,
    mask_input: Option<String>,
    position_input: Option<String>,
    logits_output: String,
    projected_state_output: String,
}

impl<'a> SharedKvProposerSession<'a> {
    /// Detect a shared-KV proposer signature from graph I/O, if present.
    pub fn detect(session: &Session) -> Result<Option<SharedKvProposerSignature>> {
        Ok(detect_shared_kv_proposer(session)?.map(|(signature, _)| signature))
    }

    /// Create a shared-KV proposer decode session from a proposer graph.
    pub fn new(session: &'a Session) -> Result<Self> {
        let (signature, io) = detect_shared_kv_proposer(session)?.ok_or_else(|| {
            OrtError::InvalidArgument(
                "model is not a shared-KV proposer (needs inputs_embeds + shared_kv.* inputs and \
                 logits + projected_state outputs, without mtp_hidden)"
                    .into(),
            )
        })?;
        Ok(Self {
            session,
            binding: IoBinding::new(session)?,
            signature,
            batch_size: 1,
            embeds_input: io.embeds_input,
            mask_input: io.mask_input,
            position_input: io.position_input,
            logits_output: io.logits_output,
            projected_state_output: io.projected_state_output,
        })
    }

    /// The introspected assistant signature.
    pub fn signature(&self) -> &SharedKvProposerSignature {
        &self.signature
    }

    /// Run one assistant forward.
    ///
    /// `inputs_embeds` is row-major `[1, q, 2*H]`. `shared_kv` must provide one
    /// entry per discovered [`SharedKvSpec`]. `position_start` is the
    /// position id of the first (and, for `q == 1`, only) token in the step.
    pub fn step(
        &mut self,
        inputs_embeds: &[f32],
        position_start: i64,
        shared_kv: &[SharedKvInput],
    ) -> Result<SharedKvProposerStepOutput> {
        let width = self.signature.inputs_embeds_width;
        if inputs_embeds.is_empty() || !inputs_embeds.len().is_multiple_of(width) {
            return Err(OrtError::InvalidArgument(format!(
                "inputs_embeds length {} must be a non-zero multiple of width {width}",
                inputs_embeds.len()
            )));
        }
        let seq_len = inputs_embeds.len() / width;
        let seq_i64 = i64::try_from(seq_len)
            .map_err(|_| OrtError::InvalidArgument("seq_len exceeds i64".into()))?;

        let embeds =
            Value::from_f32_slice_as(inputs_embeds, &[self.batch_size, seq_i64, width as i64], self.signature.dtype)?;

        // Build the shared-KV tensors, validating them against the graph specs.
        let mut kv_values = Vec::with_capacity(self.signature.shared_kv.len() * 2);
        let mut mask_len = 0usize;
        for spec in &self.signature.shared_kv {
            let input = shared_kv
                .iter()
                .find(|input| input.name == spec.name)
                .ok_or_else(|| {
                    OrtError::InvalidArgument(format!(
                        "missing shared_kv input for group '{}'",
                        spec.name
                    ))
                })?;
            if input.kv_heads != spec.kv_heads || input.head_dim != spec.head_dim {
                return Err(OrtError::InvalidArgument(format!(
                    "shared_kv group '{}' provided [{}, _, {}] but graph expects [{}, _, {}]",
                    spec.name, input.kv_heads, input.head_dim, spec.kv_heads, spec.head_dim
                )));
            }
            let expected = input.kv_heads * input.kv_len * input.head_dim;
            if input.key.len() != expected || input.value.len() != expected {
                return Err(OrtError::InvalidArgument(format!(
                    "shared_kv group '{}' key/value length must be kv_heads {} * kv_len {} * \
                     head_dim {} = {expected}",
                    spec.name, input.kv_heads, input.kv_len, input.head_dim
                )));
            }
            mask_len = mask_len.max(input.kv_len);
            let shape = [
                self.batch_size,
                input.kv_heads as i64,
                input.kv_len as i64,
                input.head_dim as i64,
            ];
            kv_values.push((
                spec.key_input.clone(),
                Value::from_f32_slice_as(&input.key, &shape, self.signature.dtype)?,
            ));
            kv_values.push((
                spec.value_input.clone(),
                Value::from_f32_slice_as(&input.value, &shape, self.signature.dtype)?,
            ));
        }
        if mask_len == 0 {
            mask_len = seq_len;
        }

        let mask: Option<Value> = if self.mask_input.is_some() {
            let data = vec![1i64; self.batch_size as usize * mask_len];
            Some(Value::from_slice_i64(
                &data,
                &[self.batch_size, mask_len as i64],
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
        if let (Some(name), Some(value)) = (self.mask_input.as_ref(), mask.as_ref()) {
            self.binding.bind_input(name, value)?;
        }
        if let (Some(name), Some(value)) = (self.position_input.as_ref(), positions.as_ref()) {
            self.binding.bind_input(name, value)?;
        }
        for (name, value) in &kv_values {
            self.binding.bind_input(name, value)?;
        }
        for output in self.session.output_names() {
            self.binding
                .bind_output_to_device(output, &MemoryInfo::cpu()?)?;
        }
        self.session.run_with_binding(&self.binding)?;

        let outputs = self.binding.output_values()?;
        let mut logits = None;
        let mut projected_state = None;
        for (name, value) in self.session.output_names().iter().zip(outputs) {
            if *name == self.logits_output {
                logits = Some(last_row_f32(&value, self.signature.vocab_size)?);
            } else if *name == self.projected_state_output {
                projected_state =
                    Some(last_row_f32(&value, self.signature.backbone_hidden_size)?);
            }
        }
        Ok(SharedKvProposerStepOutput {
            logits: logits.ok_or_else(|| {
                OrtError::InvalidArgument("shared-KV proposer did not produce logits".into())
            })?,
            projected_state: projected_state.ok_or_else(|| {
                OrtError::InvalidArgument(
                    "shared-KV proposer did not produce projected_state".into(),
                )
            })?,
        })
    }
}

fn detect_shared_kv_proposer(
    session: &Session,
) -> Result<Option<(SharedKvProposerSignature, SharedKvProposerIo)>> {
    let embeds_input = session
        .inputs()
        .iter()
        .find(|input| matches_name(&input.name, "inputs_embeds"));
    let logits_output = session
        .outputs()
        .iter()
        .find(|output| matches_name(&output.name, "logits"));
    let projected_output = session
        .outputs()
        .iter()
        .find(|output| matches_name(&output.name, "projected_state"));
    // MTP heads emit mtp_hidden; a shared-KV proposer must not, so the two graph
    // families stay unambiguous.
    let has_mtp_hidden = session
        .outputs()
        .iter()
        .any(|output| matches_name(&output.name, "mtp_hidden"));
    let (Some(embeds_input), Some(logits_output), Some(projected_output)) =
        (embeds_input, logits_output, projected_output)
    else {
        return Ok(None);
    };
    if has_mtp_hidden {
        return Ok(None);
    }

    // The proposer's float tensors may be either Float32 or Float16 (fp16 KV
    // exports, e.g. the real Gemma4 E2B assistant, feed the shared KV and
    // `inputs_embeds` as half precision). The activation dtype is taken from
    // `inputs_embeds`; shared-KV inputs must match it, while the float outputs
    // are read back through a lossless f16->f32 widening so logits/projected
    // state stay f32 on the engine-facing API regardless of the graph dtype.
    for info in [embeds_input, logits_output, projected_output] {
        if !is_supported_float(info.dtype) {
            return Err(OrtError::InvalidArgument(format!(
                "shared-KV proposer tensor '{}' must be Float32 or Float16, got {:?}",
                info.name, info.dtype
            )));
        }
    }
    let float_dtype = embeds_input.dtype;
    let shared_kv = shared_kv_specs(session, float_dtype)?;
    if shared_kv.is_empty() {
        return Ok(None);
    }

    let backbone_hidden_size = last_positive_dim(&projected_output.shape).ok_or_else(|| {
        OrtError::InvalidArgument("projected_state must have a static hidden dimension".into())
    })?;
    let inputs_embeds_width = last_positive_dim(&embeds_input.shape).ok_or_else(|| {
        OrtError::InvalidArgument("inputs_embeds must have a static last dimension".into())
    })?;
    if inputs_embeds_width != 2 * backbone_hidden_size {
        return Err(OrtError::InvalidArgument(format!(
            "inputs_embeds width {inputs_embeds_width} must be 2 * projected_state hidden \
             {backbone_hidden_size}"
        )));
    }
    let vocab_size = last_positive_dim(&logits_output.shape).ok_or_else(|| {
        OrtError::InvalidArgument("logits must have a static vocabulary dimension".into())
    })?;

    let signature = SharedKvProposerSignature {
        backbone_hidden_size,
        inputs_embeds_width,
        vocab_size,
        shared_kv,
        dtype: float_dtype,
    };
    let io = SharedKvProposerIo {
        embeds_input: embeds_input.name.clone(),
        mask_input: session
            .inputs()
            .iter()
            .find(|input| matches_name(&input.name, "attention_mask"))
            .map(|input| input.name.clone()),
        position_input: session
            .inputs()
            .iter()
            .find(|input| matches_name(&input.name, "position_ids"))
            .map(|input| input.name.clone()),
        logits_output: logits_output.name.clone(),
        projected_state_output: projected_output.name.clone(),
    };
    Ok(Some((signature, io)))
}

/// Discover `shared_kv.<name>.{key,value}` input pairs, grouped by `<name>`.
///
/// Every shared-KV input must be a rank-4 float tensor matching `float_dtype`
/// (the assistant's activation dtype, taken from `inputs_embeds`), so f16 and
/// f32 KV exports are both accepted as long as they are internally consistent.
fn shared_kv_specs(session: &Session, float_dtype: DataType) -> Result<Vec<SharedKvSpec>> {
    let mut keys: BTreeMap<String, &crate::TensorInfo> = BTreeMap::new();
    let mut values: BTreeMap<String, &crate::TensorInfo> = BTreeMap::new();
    for input in session.inputs() {
        let Some((group, kind)) = shared_kv_group(&input.name) else {
            continue;
        };
        if input.dtype != float_dtype {
            return Err(OrtError::InvalidArgument(format!(
                "shared_kv input '{}' must match the assistant activation dtype {:?}, got {:?}",
                input.name, float_dtype, input.dtype
            )));
        }
        if input.shape.len() != 4 {
            return Err(OrtError::InvalidArgument(format!(
                "shared_kv input '{}' must be rank 4 [B, kv_heads, kv_len, head_dim], got {:?}",
                input.name, input.shape
            )));
        }
        match kind {
            SharedKvKind::Key => {
                keys.insert(group, input);
            }
            SharedKvKind::Value => {
                values.insert(group, input);
            }
        }
    }

    let mut specs = Vec::new();
    for (group, key) in &keys {
        let Some(value) = values.get(group) else {
            return Err(OrtError::InvalidArgument(format!(
                "shared_kv group '{group}' has a key input without a matching value input"
            )));
        };
        let kv_heads = usize::try_from(key.shape[1].max(1)).unwrap_or(1);
        let head_dim = usize::try_from(key.shape[3].max(1)).unwrap_or(1);
        specs.push(SharedKvSpec {
            name: group.clone(),
            key_input: key.name.clone(),
            value_input: value.name.clone(),
            kv_heads,
            head_dim,
        });
    }
    for group in values.keys() {
        if !keys.contains_key(group) {
            return Err(OrtError::InvalidArgument(format!(
                "shared_kv group '{group}' has a value input without a matching key input"
            )));
        }
    }
    Ok(specs)
}

enum SharedKvKind {
    Key,
    Value,
}

/// Split `shared_kv.<group>.key`/`.value` into `(group, kind)`.
fn shared_kv_group(name: &str) -> Option<(String, SharedKvKind)> {
    let lower = name.to_ascii_lowercase();
    let rest = lower.strip_prefix("shared_kv.")?;
    if let Some(group) = rest.strip_suffix(".key") {
        Some((group.to_string(), SharedKvKind::Key))
    } else {
        rest.strip_suffix(".value")
            .map(|group| (group.to_string(), SharedKvKind::Value))
    }
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

fn last_row_f32(value: &Value, width: usize) -> Result<Vec<f32>> {
    let data = value.to_vec_f32_lossy()?;
    if width == 0 || data.len() < width || !data.len().is_multiple_of(width) {
        return Err(OrtError::InvalidArgument(format!(
            "shared-KV proposer output length {} is not a positive multiple of width {width}",
            data.len()
        )));
    }
    Ok(data[data.len() - width..].to_vec())
}

/// Whether `dtype` is a float type the shared-KV proposer can bind/read.
fn is_supported_float(dtype: DataType) -> bool {
    matches!(dtype, DataType::Float32 | DataType::Float16)
}
