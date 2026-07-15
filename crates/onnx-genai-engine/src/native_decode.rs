//! Native nxrt adapter for the engine's existing decode loop.

use crate::config::{GenerateOptions, GenerateResult};
use crate::decode::DecodeBackend;
use crate::decode_loop::{DecodeLoopBackend, DecodeLoopState, run_decode_loop};
use crate::logits::{ProcessorChain, TokenId};
use anyhow::{Context, bail};
use onnx_genai_ort::Tokenizer;
use onnx_runtime_ir::{DataType, Dim};
use onnx_runtime_session::{DevicePreference, InferenceSession, Tensor};
use std::collections::HashMap;
use std::path::Path;

/// Device requested for a native decode session.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum NativeDecodeDevice {
    #[default]
    Cpu,
    Cuda {
        index: Option<u32>,
    },
}

/// Stateful decoder-with-past adapter over the pure-Rust native runtime.
pub struct NativeDecodeSession {
    session: InferenceSession,
    input_ids: String,
    attention_mask: String,
    position_ids: String,
    logits: String,
    kv_inputs: Vec<String>,
    present_to_past: HashMap<String, String>,
    past: HashMap<String, Tensor>,
    current_len: usize,
}

impl NativeDecodeSession {
    /// Load a decoder-with-past ONNX model on the requested native device.
    pub fn load(path: impl AsRef<Path>, device: NativeDecodeDevice) -> anyhow::Result<Self> {
        if matches!(device, NativeDecodeDevice::Cuda { .. }) {
            bail!(
                "native CUDA decode is not available yet: onnx-runtime-session currently \
                 accepts GPU preferences but its executor still instantiates only the CPU EP"
            );
        }
        let preference = match device {
            NativeDecodeDevice::Cpu => DevicePreference::Cpu,
            NativeDecodeDevice::Cuda { index } => DevicePreference::Gpu { index },
        };
        let session = InferenceSession::builder()
            .model(path)
            .device(preference)
            .build()
            .context("load native decoder model")?;
        Self::from_session(session)
    }

    /// Wrap an already-built native session, validating its decoder-with-past I/O.
    pub fn from_session(session: InferenceSession) -> anyhow::Result<Self> {
        let input_names = session
            .inputs()
            .iter()
            .map(|meta| meta.name.clone())
            .collect::<Vec<_>>();
        let output_names = session
            .outputs()
            .iter()
            .map(|meta| meta.name.clone())
            .collect::<Vec<_>>();

        let input_ids = find_name(&input_names, &["input_ids", "decoder_input_ids"])
            .context("native decoder is missing input_ids")?;
        let attention_mask = find_name(&input_names, &["attention_mask"])
            .context("native decoder is missing attention_mask")?;
        let position_ids = find_name(&input_names, &["position_ids"])
            .context("native decoder is missing position_ids")?;
        let logits = find_name(&output_names, &["logits"])
            .context("native decoder is missing logits output")?;
        let kv_inputs = input_names
            .iter()
            .filter(|name| is_past_name(name))
            .cloned()
            .collect::<Vec<_>>();
        let present_outputs = output_names
            .iter()
            .filter(|name| is_present_name(name))
            .cloned()
            .collect::<Vec<_>>();
        if kv_inputs.is_empty() || present_outputs.is_empty() {
            bail!(
                "native decode requires decoder-with-past I/O; past inputs: {:?}, present outputs: {:?}",
                kv_inputs,
                present_outputs
            );
        }

        let mut present_to_past = HashMap::new();
        for output in &present_outputs {
            let Some(input) = matching_past_name(output, &kv_inputs) else {
                bail!(
                    "native decoder present output '{output}' has no matching past input; inputs: {:?}",
                    kv_inputs
                );
            };
            present_to_past.insert(output.clone(), input);
        }
        if present_to_past.len() != kv_inputs.len() {
            bail!(
                "native decoder has incomplete past/present pairs; past inputs: {:?}, present outputs: {:?}",
                kv_inputs,
                present_outputs
            );
        }

        Ok(Self {
            session,
            input_ids,
            attention_mask,
            position_ids,
            logits,
            kv_inputs,
            present_to_past,
            past: HashMap::new(),
            current_len: 0,
        })
    }

    pub fn current_len(&self) -> usize {
        self.current_len
    }

    pub fn decode(
        &mut self,
        token_ids: &[TokenId],
        past_len: usize,
    ) -> anyhow::Result<Vec<Vec<f32>>> {
        <Self as DecodeBackend>::decode(self, token_ids, past_len)
    }

    /// Rewind by prefix-slicing every carried host KV tensor.
    pub fn rewind(&mut self, target_len: usize) -> anyhow::Result<()> {
        <Self as DecodeBackend>::rewind(self, target_len)
    }

    pub fn reset(&mut self) -> anyhow::Result<()> {
        <Self as DecodeBackend>::reset(self)
    }

    /// Generate through the engine's shared token loop, not a backend-local loop.
    pub fn generate(
        &mut self,
        prompt_tokens: &[TokenId],
        options: &GenerateOptions,
        chain: &ProcessorChain,
        tokenizer: &Tokenizer,
    ) -> anyhow::Result<GenerateResult> {
        if prompt_tokens.is_empty() {
            bail!("native generation requires at least one prompt token");
        }
        self.reset()?;
        let mut backend = NativeLoopAdapter {
            session: self,
            prompt_tokens: prompt_tokens.to_vec(),
            pending_tokens: prompt_tokens.to_vec(),
        };
        let mut state = DecodeLoopState::new(0, options.seed, options.top_logprobs);
        run_decode_loop(
            &mut backend,
            &mut state,
            options,
            chain,
            tokenizer,
            options.max_context,
            None,
        )
    }

    fn make_empty_past(&self, name: &str) -> anyhow::Result<Tensor> {
        let meta = self
            .session
            .inputs()
            .iter()
            .find(|meta| meta.name == name)
            .with_context(|| format!("missing native KV metadata for '{name}'"))?;
        if meta.shape.len() < 3 {
            bail!(
                "native KV input '{name}' has unsupported shape {:?}",
                meta.shape
            );
        }
        let seq_axis = meta.shape.len() - 2;
        let mut shape = Vec::with_capacity(meta.shape.len());
        for (axis, dim) in meta.shape.iter().copied().enumerate() {
            let value = if axis == 0 {
                1
            } else if axis == seq_axis {
                0
            } else if let Dim::Static(value) = dim {
                value
            } else {
                bail!(
                    "cannot infer native empty KV dimension {axis} for '{name}' shape {:?}",
                    meta.shape
                );
            };
            shape.push(value);
        }
        let bytes = meta
            .dtype
            .checked_storage_bytes(shape.iter().product())
            .with_context(|| format!("unsupported KV dtype {:?} for '{name}'", meta.dtype))?;
        Tensor::from_raw(meta.dtype, shape, &vec![0; bytes])
            .with_context(|| format!("create empty native KV tensor '{name}'"))
    }
}

impl DecodeBackend for NativeDecodeSession {
    fn current_len(&self) -> usize {
        self.current_len
    }

    fn decode(&mut self, token_ids: &[TokenId], past_len: usize) -> anyhow::Result<Vec<Vec<f32>>> {
        if token_ids.is_empty() {
            bail!("native decode requires at least one token");
        }
        if past_len != self.current_len {
            bail!(
                "native decode past length mismatch: caller supplied {past_len}, adapter holds {}",
                self.current_len
            );
        }
        let total_len = past_len
            .checked_add(token_ids.len())
            .context("native decode context length overflow")?;
        let ids = token_ids
            .iter()
            .map(|&id| i64::from(id))
            .collect::<Vec<_>>();
        let positions = (past_len..total_len)
            .map(|position| i64::try_from(position).context("position id exceeds i64 range"))
            .collect::<anyhow::Result<Vec<_>>>()?;
        let input_ids = Tensor::from_i64(&[1, token_ids.len()], &ids)?;
        let attention_mask = Tensor::from_i64(&[1, total_len], &vec![1; total_len])?;
        let position_ids = Tensor::from_i64(&[1, token_ids.len()], &positions)?;

        let mut owned = Vec::with_capacity(3 + self.kv_inputs.len());
        owned.push((self.input_ids.clone(), input_ids));
        owned.push((self.attention_mask.clone(), attention_mask));
        owned.push((self.position_ids.clone(), position_ids));
        for name in &self.kv_inputs {
            let tensor = match self.past.remove(name) {
                Some(tensor) => tensor,
                None => self.make_empty_past(name)?,
            };
            owned.push((name.clone(), tensor));
        }
        let bindings = owned
            .iter()
            .map(|(name, tensor)| (name.as_str(), tensor))
            .collect::<Vec<_>>();
        let outputs = match self.session.run(&bindings) {
            Ok(outputs) => outputs,
            Err(error) => {
                let diagnosis = diagnose_native_failure(&self.session, &error.to_string());
                bail!("native decoder forward pass failed{diagnosis}: {error}");
            }
        };
        let names = self
            .session
            .outputs()
            .iter()
            .map(|meta| meta.name.clone())
            .collect::<Vec<_>>();
        let mut named = names.into_iter().zip(outputs).collect::<HashMap<_, _>>();
        let logits = named
            .remove(&self.logits)
            .with_context(|| format!("native decoder omitted logits output '{}'", self.logits))?;
        let logits = extract_logits(&logits, token_ids.len())?;

        let mut next_past = HashMap::with_capacity(self.kv_inputs.len());
        for (present, past) in &self.present_to_past {
            let tensor = named
                .remove(present)
                .with_context(|| format!("native decoder omitted present output '{present}'"))?;
            let seq_axis =
                tensor.shape.len().checked_sub(2).with_context(|| {
                    format!("native present tensor '{present}' rank is below 2")
                })?;
            if tensor.shape[seq_axis] != total_len {
                bail!(
                    "native present tensor '{present}' sequence length {} does not match {total_len}",
                    tensor.shape[seq_axis]
                );
            }
            next_past.insert(past.clone(), tensor);
        }
        self.past = next_past;
        self.current_len = total_len;
        Ok(logits)
    }

    fn rewind(&mut self, target_len: usize) -> anyhow::Result<()> {
        if target_len > self.current_len {
            bail!(
                "cannot rewind native KV from {} forward to {target_len}",
                self.current_len
            );
        }
        if target_len == self.current_len {
            return Ok(());
        }
        if target_len == 0 {
            self.past.clear();
            self.current_len = 0;
            return Ok(());
        }
        for (name, tensor) in &mut self.past {
            let axis = tensor
                .shape
                .len()
                .checked_sub(2)
                .with_context(|| format!("native KV tensor '{name}' rank is below 2"))?;
            *tensor = prefix_slice(tensor, axis, target_len)
                .with_context(|| format!("rewind native KV tensor '{name}'"))?;
        }
        self.current_len = target_len;
        Ok(())
    }
}

struct NativeLoopAdapter<'a> {
    session: &'a mut NativeDecodeSession,
    prompt_tokens: Vec<TokenId>,
    pending_tokens: Vec<TokenId>,
}

impl DecodeLoopBackend for NativeLoopAdapter<'_> {
    fn context_len(&self) -> usize {
        self.session.current_len() + self.pending_tokens.len()
    }

    fn processor_prompt_tokens(&self) -> Vec<TokenId> {
        self.prompt_tokens.clone()
    }

    fn next_logits(&mut self) -> anyhow::Result<Vec<f32>> {
        let past_len = self.session.current_len();
        self.session
            .decode(&self.pending_tokens, past_len)?
            .pop()
            .context("native decoder produced no logits")
    }

    fn commit_token(&mut self, token_id: TokenId) -> anyhow::Result<()> {
        self.pending_tokens.clear();
        self.pending_tokens.push(token_id);
        Ok(())
    }
}

fn find_name(names: &[String], candidates: &[&str]) -> Option<String> {
    names.iter().find_map(|name| {
        let lower = name.to_ascii_lowercase();
        candidates
            .iter()
            .any(|candidate| lower == *candidate || lower.ends_with(&format!(".{candidate}")))
            .then(|| name.clone())
    })
}

fn is_past_name(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    lower.starts_with("past_key_values.") || lower.starts_with("past.")
}

fn is_present_name(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    lower.starts_with("present_key_values.") || lower.starts_with("present.")
}

fn matching_past_name(output: &str, inputs: &[String]) -> Option<String> {
    let lower = output.to_ascii_lowercase();
    let suffix = lower
        .strip_prefix("present_key_values.")
        .or_else(|| lower.strip_prefix("present."))?;
    inputs.iter().find_map(|input| {
        let input_lower = input.to_ascii_lowercase();
        (input_lower.strip_prefix("past_key_values.") == Some(suffix)
            || input_lower.strip_prefix("past.") == Some(suffix))
        .then(|| input.clone())
    })
}

fn extract_logits(tensor: &Tensor, sequence_len: usize) -> anyhow::Result<Vec<Vec<f32>>> {
    let vocab = match tensor.shape.as_slice() {
        [1, seq, vocab] if *seq == sequence_len => *vocab,
        [seq, vocab] if *seq == sequence_len => *vocab,
        shape => bail!(
            "native logits must have shape [1, sequence, vocab] or [sequence, vocab], got {shape:?}"
        ),
    };
    let values = tensor_to_f32(tensor)?;
    Ok(values.chunks_exact(vocab).map(<[f32]>::to_vec).collect())
}

fn tensor_to_f32(tensor: &Tensor) -> anyhow::Result<Vec<f32>> {
    match tensor.dtype {
        DataType::Float32 => Ok(tensor.to_vec_f32()),
        DataType::Float16 => Ok(tensor
            .as_bytes()
            .chunks_exact(2)
            .map(|bytes| f16_to_f32(u16::from_le_bytes([bytes[0], bytes[1]])))
            .collect()),
        DataType::BFloat16 => Ok(tensor
            .as_bytes()
            .chunks_exact(2)
            .map(|bytes| f32::from_bits(u32::from(u16::from_le_bytes([bytes[0], bytes[1]])) << 16))
            .collect()),
        dtype => bail!("native logits must be Float32, Float16, or BFloat16, got {dtype:?}"),
    }
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = (u32::from(bits & 0x8000)) << 16;
    let exponent = (bits >> 10) & 0x1f;
    let fraction = u32::from(bits & 0x03ff);
    let value = match exponent {
        0 if fraction == 0 => sign,
        0 => {
            let shift = fraction.leading_zeros() - 21;
            let normalized = (fraction << (shift + 1)) & 0x03ff;
            sign | ((127 - 15 - shift) << 23) | (normalized << 13)
        }
        0x1f => sign | 0x7f80_0000 | (fraction << 13),
        _ => sign | ((u32::from(exponent) + 112) << 23) | (fraction << 13),
    };
    f32::from_bits(value)
}

fn prefix_slice(tensor: &Tensor, axis: usize, len: usize) -> anyhow::Result<Tensor> {
    let axis_len = *tensor
        .shape
        .get(axis)
        .context("native KV slice axis out of bounds")?;
    if len > axis_len {
        bail!("native KV slice length {len} exceeds axis length {axis_len}");
    }

    let inner = tensor.shape[axis + 1..].iter().product::<usize>();
    let outer = tensor.shape[..axis].iter().product::<usize>();
    let elem_bytes = tensor
        .dtype
        .checked_storage_bytes(1)
        .context("native KV dtype has no fixed storage size")?;
    let source_stride = axis_len * inner * elem_bytes;
    let kept_stride = len * inner * elem_bytes;
    let source = tensor.as_bytes();
    let mut bytes = Vec::with_capacity(outer * kept_stride);
    for index in 0..outer {
        let start = index * source_stride;
        bytes.extend_from_slice(&source[start..start + kept_stride]);
    }
    let mut shape = tensor.shape.clone();
    shape[axis] = len;
    Tensor::from_raw(tensor.dtype, shape, &bytes).context("create sliced native KV tensor")
}

fn diagnose_native_failure(session: &InferenceSession, error: &str) -> String {
    if error.contains("f32 kernel input requires Float32, got Int64") {
        for (_, node) in session.graph().nodes.iter() {
            if node.op_type == "Gather"
                && let Some(data) = node.inputs.first().copied().flatten()
                && session.graph().value(data).dtype == DataType::Int64
            {
                return " (native CPU Gather lacks Int64 data support)".to_string();
            }
        }
    }
    String::new()
}

#[cfg(test)]
mod tests {
    use super::*;
    use onnx_runtime_ir::{Attribute, Graph, Node, NodeId, Shape};

    fn insert_op(
        graph: &mut Graph,
        op_type: &str,
        inputs: Vec<onnx_runtime_ir::ValueId>,
        output: onnx_runtime_ir::ValueId,
        attributes: &[(&str, Attribute)],
    ) {
        let mut node = Node::new(
            NodeId(0),
            op_type,
            inputs.into_iter().map(Some).collect(),
            vec![output],
        );
        for (name, value) in attributes {
            node.attributes.insert((*name).to_string(), value.clone());
        }
        graph.insert_node(node);
    }

    fn tiny_decoder() -> InferenceSession {
        let mut graph = Graph::new();
        graph.opset_imports.insert(String::new(), 11);
        let batch = graph.intern_symbol("batch");
        let sequence = graph.intern_symbol("sequence");
        let total = graph.intern_symbol("total");
        let past = graph.intern_symbol("past");
        let shape = |dims: &[Dim]| -> Shape { dims.to_vec() };

        let input_ids = graph.create_named_value(
            "input_ids",
            DataType::Int64,
            shape(&[batch.into(), sequence.into()]),
        );
        let attention_mask = graph.create_named_value(
            "attention_mask",
            DataType::Int64,
            shape(&[batch.into(), total.into()]),
        );
        let position_ids = graph.create_named_value(
            "position_ids",
            DataType::Int64,
            shape(&[batch.into(), sequence.into()]),
        );
        let past_key = graph.create_named_value(
            "past_key_values.0.key",
            DataType::Float32,
            shape(&[batch.into(), 1.into(), past.into(), 1.into()]),
        );
        let past_value = graph.create_named_value(
            "past_key_values.0.value",
            DataType::Float32,
            shape(&[batch.into(), 1.into(), past.into(), 1.into()]),
        );
        for input in [
            input_ids,
            attention_mask,
            position_ids,
            past_key,
            past_value,
        ] {
            graph.add_input(input);
        }

        let cast = graph.create_value(DataType::Float32, shape(&[batch.into(), sequence.into()]));
        insert_op(
            &mut graph,
            "Cast",
            vec![input_ids],
            cast,
            &[("to", Attribute::Int(1))],
        );
        let current_kv = graph.create_value(
            DataType::Float32,
            shape(&[batch.into(), 1.into(), sequence.into(), 1.into()]),
        );
        insert_op(
            &mut graph,
            "Unsqueeze",
            vec![cast],
            current_kv,
            &[("axes", Attribute::Ints(vec![1, 3]))],
        );

        let logits = graph.create_named_value(
            "logits",
            DataType::Float32,
            shape(&[batch.into(), sequence.into(), 1.into()]),
        );
        insert_op(
            &mut graph,
            "Unsqueeze",
            vec![cast],
            logits,
            &[("axes", Attribute::Ints(vec![2]))],
        );
        let present_key = graph.create_named_value(
            "present.0.key",
            DataType::Float32,
            shape(&[batch.into(), 1.into(), total.into(), 1.into()]),
        );
        insert_op(
            &mut graph,
            "Concat",
            vec![past_key, current_kv],
            present_key,
            &[("axis", Attribute::Int(2))],
        );
        let present_value = graph.create_named_value(
            "present.0.value",
            DataType::Float32,
            shape(&[batch.into(), 1.into(), total.into(), 1.into()]),
        );
        insert_op(
            &mut graph,
            "Concat",
            vec![past_value, current_kv],
            present_value,
            &[("axis", Attribute::Int(2))],
        );
        for output in [logits, present_key, present_value] {
            graph.add_output(output);
        }
        InferenceSession::from_graph(graph).expect("build tiny decoder")
    }

    #[test]
    fn native_decode_advances_kv_and_rewinds() {
        let mut session = NativeDecodeSession::from_session(tiny_decoder()).expect("load decoder");
        let logits = session.decode(&[1, 2, 3], 0).expect("prefill");
        assert_eq!(logits.len(), 3);
        assert_eq!(logits[0].len(), 1);
        assert_eq!(session.current_len(), 3);

        let logits = session.decode(&[4], 3).expect("decode");
        assert_eq!(logits.len(), 1);
        assert_eq!(logits[0].len(), 1);
        assert_eq!(session.current_len(), 4);

        session.rewind(2).expect("rewind");
        assert_eq!(session.current_len(), 2);
        session.decode(&[5], 2).expect("decode after rewind");
        assert_eq!(session.current_len(), 3);
    }
}
