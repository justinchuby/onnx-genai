//! Incremental decode helpers built on ORT IoBinding.

use std::collections::HashMap;
use std::sync::Arc;

use crate::{DataType, IoBinding, MemoryInfo, OrtError, Result, Session, TensorInfo, Value};

/// KV binding strategy selected for a decode session.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodeKvMode {
    /// ORT allocates `present.*` outputs; those OrtValues are rebound as next
    /// step's `past_key_values.*` inputs. No Rust-side KV copy is performed.
    ZeroCopyRebind,
    /// Caller/model-declared `past_present_share_buffer` mode. One max-length
    /// OrtValue per KV tensor is bound as both past input and present output.
    SharedBuffer,
}

/// Options for [`DecodeSession`].
#[derive(Debug, Clone)]
pub struct DecodeSessionOptions {
    /// Batch size for empty/shared KV buffers. Generation currently uses 1.
    pub batch_size: i64,
    /// Maximum logical context length. Required for shared-buffer mode.
    pub max_length: Option<usize>,
    /// Override ORT custom metadata detection of `past_present_share_buffer`.
    pub past_present_share_buffer: Option<bool>,
}

impl Default for DecodeSessionOptions {
    fn default() -> Self {
        Self {
            batch_size: 1,
            max_length: None,
            past_present_share_buffer: None,
        }
    }
}

#[derive(Debug, Clone)]
struct KvPair {
    past: String,
    present: String,
    input: TensorInfo,
    seq_axis: usize,
}

/// A stateful IoBinding decode runner that keeps KV OrtValues inside ORT.
pub struct DecodeSession<'a> {
    session: &'a Session,
    binding: IoBinding,
    kv_pairs: Vec<KvPair>,
    current_kv: HashMap<String, Arc<Value>>,
    current_len: usize,
    mode: DecodeKvMode,
}

impl<'a> DecodeSession<'a> {
    /// Create a decode session and infer KV input/output pairs from graph names.
    pub fn new(session: &'a Session, options: DecodeSessionOptions) -> Result<Self> {
        let kv_pairs = infer_kv_pairs(session)?;
        let share_buffer = options
            .past_present_share_buffer
            .unwrap_or(session.past_present_share_buffer_supported());
        let mode = if share_buffer {
            DecodeKvMode::SharedBuffer
        } else {
            DecodeKvMode::ZeroCopyRebind
        };
        let mut this = Self {
            session,
            binding: IoBinding::new(session)?,
            kv_pairs,
            current_kv: HashMap::new(),
            current_len: 0,
            mode,
        };
        if mode == DecodeKvMode::SharedBuffer {
            let max_length = options.max_length.ok_or_else(|| {
                OrtError::InvalidArgument(
                    "DecodeSession shared-buffer mode requires max_length".into(),
                )
            })?;
            this.allocate_shared_buffers(options.batch_size, max_length)?;
        }
        Ok(this)
    }

    /// The selected KV binding strategy.
    pub fn mode(&self) -> DecodeKvMode {
        self.mode
    }

    /// Current logical KV length in tokens.
    pub fn past_len(&self) -> usize {
        self.current_len
    }

    /// Run one incremental decode step and return the logits OrtValue.
    ///
    /// `attention_mask` is the full `[batch, past + new]` mask flattened row-major,
    /// while `position_ids` covers only `new_input_ids`.
    pub fn step(
        &mut self,
        new_input_ids: &[i64],
        attention_mask: &[i64],
        position_ids: &[i64],
    ) -> Result<Value> {
        if new_input_ids.is_empty() {
            return Err(OrtError::InvalidArgument(
                "decode step requires at least one input id".into(),
            ));
        }
        let seq_len = i64::try_from(new_input_ids.len())
            .map_err(|_| OrtError::InvalidArgument("input length exceeds i64".into()))?;
        let total_len = i64::try_from(attention_mask.len())
            .map_err(|_| OrtError::InvalidArgument("attention mask length exceeds i64".into()))?;
        if position_ids.len() != new_input_ids.len() {
            return Err(OrtError::InvalidArgument(
                "position_ids length must match input_ids length".into(),
            ));
        }

        let input_ids = Value::from_slice_i64(new_input_ids, &[1, seq_len])?;
        let attention_mask = Value::from_slice_i64(attention_mask, &[1, total_len])?;
        let position_ids = Value::from_slice_i64(position_ids, &[1, seq_len])?;

        self.binding.clear()?;
        self.bind_standard_inputs(&input_ids, &attention_mask, &position_ids)?;
        self.bind_kv_inputs()?;
        for output in self.session.output_names() {
            if self.mode == DecodeKvMode::SharedBuffer
                && let Some(pair) = self.kv_pairs.iter().find(|pair| pair.present == *output)
            {
                let value = self.current_kv.get(&pair.past).ok_or_else(|| {
                    OrtError::InvalidArgument(format!(
                        "missing shared KV buffer for '{}'",
                        pair.past
                    ))
                })?;
                self.binding.bind_output(output, value)?;
            } else {
                self.binding
                    .bind_output_to_device(output, &MemoryInfo::cpu()?)?;
            }
        }

        self.session.run_with_binding(&self.binding)?;
        let mut logits = None;
        let outputs = self.binding.output_values()?;
        self.rotate_outputs(outputs, &mut logits)?;
        self.current_len = self
            .current_len
            .checked_add(new_input_ids.len())
            .ok_or_else(|| OrtError::InvalidArgument("decode length overflow".into()))?;
        logits.ok_or_else(|| OrtError::InvalidArgument("model did not produce logits".into()))
    }

    /// Rewind to a smaller logical KV length.
    ///
    /// In zero-copy-rebind mode this rebinds a compact prefix tensor for each
    /// current present buffer. This is no-copy when the prefix is contiguous in
    /// memory; otherwise rewind performs a one-time compacting slice copy for
    /// correctness. In shared-buffer mode only the logical cursor changes; stale
    /// data beyond `target_len` remains in the buffers and must be masked out by
    /// subsequent attention masks/position ids.
    pub fn rewind(&mut self, target_len: usize) -> Result<()> {
        if target_len > self.current_len {
            return Err(OrtError::InvalidArgument(format!(
                "cannot rewind from {} to larger length {}",
                self.current_len, target_len
            )));
        }
        if target_len == self.current_len {
            return Ok(());
        }
        if target_len == 0 {
            if self.mode == DecodeKvMode::ZeroCopyRebind {
                self.current_kv.clear();
            }
            self.current_len = 0;
            return Ok(());
        }
        if self.mode == DecodeKvMode::ZeroCopyRebind {
            let mut rewound = HashMap::with_capacity(self.current_kv.len());
            for pair in &self.kv_pairs {
                let value = self.current_kv.get(&pair.past).ok_or_else(|| {
                    OrtError::InvalidArgument(format!("missing KV tensor '{}'", pair.past))
                })?;
                let mut shape = value.shape().to_vec();
                shape[pair.seq_axis] = i64::try_from(target_len).map_err(|_| {
                    OrtError::InvalidArgument("target rewind length exceeds i64".into())
                })?;
                rewound.insert(
                    pair.past.clone(),
                    Arc::new(prefix_value(value, &shape, pair.seq_axis)?),
                );
            }

            fn prefix_value(value: &Arc<Value>, shape: &[i64], seq_axis: usize) -> Result<Value> {
                let owner_shape = value.shape();
                let prefix_is_contiguous = owner_shape.iter().take(seq_axis).all(|&dim| dim == 1);
                if prefix_is_contiguous {
                    return Value::alias_with_shape(Arc::clone(value), shape);
                }

                match value.dtype() {
                    DataType::Float32 => {
                        let data = value.to_vec_f32()?;
                        let prefix = copy_prefix(&data, owner_shape, shape);
                        Value::from_vec_f32(prefix, shape)
                    }
                    DataType::Float16 => {
                        let data = value.to_vec_f16_bits()?;
                        let prefix = copy_prefix(&data, owner_shape, shape);
                        Value::from_vec_f16_bits(prefix, shape)
                    }
                    dtype => Err(OrtError::InvalidArgument(format!(
                        "cannot rewind KV tensor with dtype {dtype:?}"
                    ))),
                }
            }

            fn copy_prefix<T: Copy>(
                data: &[T],
                input_shape: &[i64],
                output_shape: &[i64],
            ) -> Vec<T> {
                let output_len = output_shape.iter().product::<i64>() as usize;
                let mut output = Vec::with_capacity(output_len);
                let input_strides = row_major_strides(input_shape);
                for mut linear in 0..output_len {
                    let mut input_offset = 0usize;
                    for (axis, &dim) in output_shape.iter().enumerate().rev() {
                        let index = linear % dim as usize;
                        linear /= dim as usize;
                        input_offset += index * input_strides[axis];
                    }
                    output.push(data[input_offset]);
                }
                output
            }

            fn row_major_strides(shape: &[i64]) -> Vec<usize> {
                let mut strides = vec![1; shape.len()];
                for axis in (0..shape.len().saturating_sub(1)).rev() {
                    strides[axis] = strides[axis + 1] * shape[axis + 1] as usize;
                }
                strides
            }
            self.current_kv = rewound;
        }
        self.current_len = target_len;
        Ok(())
    }

    /// Reset the decode cursor and drop zero-copy-rebind KV state.
    pub fn reset(&mut self) {
        if self.mode == DecodeKvMode::ZeroCopyRebind {
            self.current_kv.clear();
        }
        self.current_len = 0;
    }

    fn bind_standard_inputs(
        &mut self,
        input_ids: &Value,
        attention_mask: &Value,
        position_ids: &Value,
    ) -> Result<()> {
        for input in self.session.inputs() {
            let lower = input.name.to_ascii_lowercase();
            if lower == "input_ids" || lower.ends_with(".input_ids") {
                self.binding.bind_input(&input.name, input_ids)?;
            } else if lower == "attention_mask" || lower.ends_with(".attention_mask") {
                self.binding.bind_input(&input.name, attention_mask)?;
            } else if lower == "position_ids" || lower.ends_with(".position_ids") {
                self.binding.bind_input(&input.name, position_ids)?;
            }
        }
        Ok(())
    }

    fn bind_kv_inputs(&mut self) -> Result<()> {
        for pair in &self.kv_pairs {
            let value = if let Some(value) = self.current_kv.get(&pair.past) {
                Arc::clone(value)
            } else {
                Arc::new(empty_past_value(&pair.input)?)
            };
            self.binding.bind_input(&pair.past, &value)?;
        }
        Ok(())
    }

    fn rotate_outputs(&mut self, outputs: Vec<Value>, logits: &mut Option<Value>) -> Result<()> {
        if self.mode == DecodeKvMode::SharedBuffer {
            for (name, value) in self.session.output_names().iter().zip(outputs) {
                if is_logits_output(name) {
                    *logits = Some(value);
                    break;
                }
            }
            return Ok(());
        }

        let present_to_past = self
            .kv_pairs
            .iter()
            .map(|pair| (pair.present.as_str(), pair.past.as_str()))
            .collect::<HashMap<_, _>>();
        let mut next_kv = HashMap::with_capacity(self.kv_pairs.len());
        for (name, value) in self.session.output_names().iter().zip(outputs) {
            if let Some(past_name) = present_to_past.get(name.as_str()) {
                next_kv.insert((*past_name).to_string(), Arc::new(value));
            } else if is_logits_output(name) || logits.is_none() {
                *logits = Some(value);
            }
        }
        self.current_kv = next_kv;
        Ok(())
    }

    fn allocate_shared_buffers(&mut self, batch_size: i64, max_length: usize) -> Result<()> {
        let max_length = i64::try_from(max_length)
            .map_err(|_| OrtError::InvalidArgument("max_length exceeds i64".into()))?;
        for pair in &self.kv_pairs {
            let mut shape = pair.input.shape.clone();
            for (axis, dim) in shape.iter_mut().enumerate() {
                if axis == 0 {
                    *dim = batch_size;
                } else if axis == pair.seq_axis {
                    *dim = max_length;
                } else if *dim < 0 {
                    return Err(OrtError::InvalidArgument(format!(
                        "cannot infer shared-buffer static dimension {axis} for '{}'",
                        pair.past
                    )));
                }
            }
            self.current_kv.insert(
                pair.past.clone(),
                Arc::new(Value::empty(&shape, pair.input.dtype)?),
            );
        }
        Ok(())
    }
}

fn infer_kv_pairs(session: &Session) -> Result<Vec<KvPair>> {
    let input_names = session.input_names();
    let mut pairs = Vec::new();
    for output in session.outputs() {
        if !is_present_output(&output.name) {
            continue;
        }
        let Some(suffix) = kv_suffix(&output.name) else {
            continue;
        };
        let Some(past_name) = input_names
            .iter()
            .find(|input| kv_suffix(input).as_deref() == Some(suffix.as_str()))
        else {
            continue;
        };
        let input = session
            .inputs()
            .iter()
            .find(|input| input.name == *past_name)
            .expect("past name came from session inputs")
            .clone();
        if input.dtype != DataType::Float32 && input.dtype != DataType::Float16 {
            return Err(OrtError::InvalidArgument(format!(
                "KV input '{}' must be Float32 or Float16, got {:?}",
                input.name, input.dtype
            )));
        }
        if input.shape.len() < 3 {
            return Err(OrtError::InvalidArgument(format!(
                "KV input '{}' has unsupported shape {:?}",
                input.name, input.shape
            )));
        }
        let seq_axis = input.shape.len() - 2;
        pairs.push(KvPair {
            past: past_name.clone(),
            present: output.name.clone(),
            input,
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
                "cannot infer static dimension {axis} for empty KV input '{}'",
                info.name
            )));
        };
        shape.push(value);
    }
    Value::empty(&shape, info.dtype)
}

fn is_logits_output(name: &str) -> bool {
    name.to_ascii_lowercase().contains("logits")
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
