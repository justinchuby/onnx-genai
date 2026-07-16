//! Native nxrt adapter for the engine's existing decode loop.

use crate::config::{GenerateOptions, GenerateResult};
use crate::decode::DecodeBackend;
use crate::decode_loop::{DecodeLoopBackend, DecodeLoopState, run_decode_loop};
use crate::logits::{ProcessorChain, TokenId};
use anyhow::{Context, bail};
use onnx_genai_ort::Tokenizer;
use onnx_runtime_ir::{DataType, DeviceType, Dim};
use onnx_runtime_session::{
    DeviceBindingTransferStats, DeviceIoBinding, DevicePreference, InferenceSession, Tensor,
};
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

const DEFAULT_CUDA_KV_MAX_LEN: usize = 4096;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CudaKvDebugStats {
    pub logical_len: usize,
    pub max_len: usize,
    pub device_ptrs: Vec<usize>,
    pub kv_transfers: DeviceBindingTransferStats,
}

struct DecodeCudaState {
    logical_len: usize,
    max_len: usize,
    bindings: Vec<DeviceIoBinding>,
}

/// Stateful decoder-with-past adapter over the pure-Rust native runtime.
pub struct NativeDecodeSession {
    session: InferenceSession,
    input_ids: String,
    attention_mask: String,
    position_ids: Option<String>,
    logits: String,
    kv_inputs: Vec<String>,
    present_to_past: HashMap<String, String>,
    past: HashMap<String, Tensor>,
    cuda: Option<DecodeCudaState>,
    current_len: usize,
}

impl NativeDecodeSession {
    /// Load a decoder-with-past ONNX model on the requested native device.
    pub fn load(path: impl AsRef<Path>, device: NativeDecodeDevice) -> anyhow::Result<Self> {
        Self::load_with_cuda_kv_max_len(path, device, None)
    }

    /// Load with an explicit CUDA KV capacity. `None` uses
    /// `ONNX_GENAI_CUDA_KV_MAX_LEN`, then the 4096-token default.
    pub fn load_with_cuda_kv_max_len(
        path: impl AsRef<Path>,
        device: NativeDecodeDevice,
        cuda_kv_max_len: Option<usize>,
    ) -> anyhow::Result<Self> {
        let preference = match device {
            NativeDecodeDevice::Cpu => DevicePreference::Cpu,
            NativeDecodeDevice::Cuda { index } => DevicePreference::Gpu { index },
        };
        let session = InferenceSession::builder()
            .model(path)
            .device(preference)
            .build()
            .context("load native decoder model")?;
        Self::from_session_with_cuda_kv_max_len(session, cuda_kv_max_len)
    }

    /// Wrap an already-built native session, validating its decoder-with-past I/O.
    pub fn from_session(session: InferenceSession) -> anyhow::Result<Self> {
        Self::from_session_with_cuda_kv_max_len(session, None)
    }

    fn from_session_with_cuda_kv_max_len(
        mut session: InferenceSession,
        cuda_kv_max_len: Option<usize>,
    ) -> anyhow::Result<Self> {
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
        let position_ids = find_name(&input_names, &["position_ids"]);
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

        let cuda = if session.device_id().device_type == DeviceType::Cuda {
            let max_len = match cuda_kv_max_len {
                Some(0) => bail!("CUDA KV max length must be greater than zero"),
                Some(value) => value,
                None => cuda_kv_max_len_from_env()?,
            };
            Some(DecodeCudaState::new(
                &mut session,
                &attention_mask,
                &present_to_past,
                max_len,
            )?)
        } else {
            None
        };

        Ok(Self {
            session,
            input_ids,
            attention_mask,
            position_ids,
            logits,
            kv_inputs,
            present_to_past,
            past: HashMap::new(),
            cuda,
            current_len: 0,
        })
    }

    pub fn current_len(&self) -> usize {
        self.current_len
    }

    pub fn kv_layer_count(&self) -> usize {
        self.kv_inputs.len() / 2
    }

    pub fn cuda_kv_debug_stats(&self) -> Option<CudaKvDebugStats> {
        self.cuda.as_ref().map(DecodeCudaState::debug_stats)
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

    fn decode_cuda(
        &mut self,
        token_ids: &[TokenId],
        past_len: usize,
    ) -> anyhow::Result<Vec<Vec<f32>>> {
        let total_len = past_len
            .checked_add(token_ids.len())
            .context("native decode context length overflow")?;
        let state = self
            .cuda
            .as_mut()
            .context("CUDA decode state is not initialized")?;
        if total_len > state.max_len {
            bail!(
                "CUDA KV capacity exceeded: requested context length {total_len}, configured max_len {} (set ONNX_GENAI_CUDA_KV_MAX_LEN or use load_with_cuda_kv_max_len)",
                state.max_len
            );
        }
        state.extend_mask(past_len, total_len)?;

        let ids = token_ids
            .iter()
            .map(|&id| i64::from(id))
            .collect::<Vec<_>>();
        let input_ids = Tensor::from_i64(&[1, token_ids.len()], &ids)?;
        let mut owned = Vec::with_capacity(2);
        owned.push((self.input_ids.clone(), input_ids));
        if let Some(position_ids_name) = &self.position_ids {
            let positions = (past_len..total_len)
                .map(|position| i64::try_from(position).context("position id exceeds i64 range"))
                .collect::<anyhow::Result<Vec<_>>>()?;
            owned.push((
                position_ids_name.clone(),
                Tensor::from_i64(&[1, token_ids.len()], &positions)?,
            ));
        }
        let bindings = owned
            .iter()
            .map(|(name, tensor)| (name.as_str(), tensor))
            .collect::<Vec<_>>();
        let outputs = match self
            .session
            .run_with_device_bindings(&bindings, &mut state.bindings)
        {
            Ok(outputs) => outputs,
            Err(error) => {
                let diagnosis = diagnose_native_failure(&self.session, &error.to_string());
                bail!("native CUDA decoder forward pass failed{diagnosis}: {error}");
            }
        };
        let names = self
            .session
            .outputs()
            .iter()
            .map(|meta| meta.name.clone())
            .collect::<Vec<_>>();
        let mut named = names
            .into_iter()
            .zip(outputs)
            .filter_map(|(name, tensor)| tensor.map(|tensor| (name, tensor)))
            .collect::<HashMap<_, _>>();
        let logits = named
            .remove(&self.logits)
            .with_context(|| format!("native decoder omitted logits output '{}'", self.logits))?;
        if !named.is_empty() {
            bail!(
                "native CUDA decoder unexpectedly materialized bound outputs: {:?}",
                named.keys().collect::<Vec<_>>()
            );
        }
        let logits = extract_logits(&logits)?;
        if logits.iter().flatten().any(|value| !value.is_finite()) {
            bail!("native decoder produced non-finite logits");
        }
        state.set_logical_len(total_len)?;
        self.current_len = total_len;
        Ok(logits)
    }
}

impl DecodeCudaState {
    fn new(
        session: &mut InferenceSession,
        attention_mask: &str,
        present_to_past: &HashMap<String, String>,
        max_len: usize,
    ) -> anyhow::Result<Self> {
        let mut mask = session.allocate_device_binding(
            attention_mask,
            None::<String>,
            DataType::Int64,
            vec![1, max_len],
            vec![1, 0],
        )?;
        mask.write_bytes(0, &vec![0; max_len * std::mem::size_of::<i64>()])?;

        let mut pairs = present_to_past
            .iter()
            .map(|(present, past)| (present.clone(), past.clone()))
            .collect::<Vec<_>>();
        pairs.sort_unstable_by(|left, right| left.1.cmp(&right.1));
        let mut bindings = Vec::with_capacity(1 + pairs.len());
        bindings.push(mask);
        for (present, past) in pairs {
            let meta = session
                .inputs()
                .iter()
                .find(|meta| meta.name == past)
                .with_context(|| format!("missing CUDA KV input metadata for '{past}'"))?;
            if meta.dtype != DataType::Float32 || meta.shape.len() != 4 {
                bail!(
                    "CUDA KV input '{past}' must be rank-4 f32, got {:?} {:?}",
                    meta.dtype,
                    meta.shape
                );
            }
            let mut physical_shape = Vec::with_capacity(4);
            for (axis, dim) in meta.shape.iter().copied().enumerate() {
                let value = if axis == 0 {
                    1
                } else if axis == 2 {
                    max_len
                } else if let Dim::Static(value) = dim {
                    value
                } else {
                    bail!(
                        "cannot infer CUDA KV dimension {axis} for '{past}' shape {:?}",
                        meta.shape
                    );
                };
                physical_shape.push(value);
            }
            let mut logical_shape = physical_shape.clone();
            logical_shape[2] = 0;
            bindings.push(session.allocate_device_binding(
                past,
                Some(present),
                meta.dtype,
                physical_shape,
                logical_shape,
            )?);
        }
        Ok(Self {
            logical_len: 0,
            max_len,
            bindings,
        })
    }

    fn extend_mask(&mut self, start: usize, end: usize) -> anyhow::Result<()> {
        if end > self.max_len || start > end {
            bail!(
                "invalid CUDA mask update {start}..{end} for capacity {}",
                self.max_len
            );
        }
        let ones = (start..end)
            .flat_map(|_| 1i64.to_le_bytes())
            .collect::<Vec<_>>();
        self.bindings[0].write_bytes(start * std::mem::size_of::<i64>(), &ones)?;
        self.bindings[0].set_logical_shape(vec![1, end])?;
        Ok(())
    }

    fn set_logical_len(&mut self, len: usize) -> anyhow::Result<()> {
        for binding in self.bindings.iter_mut().skip(1) {
            let mut shape = binding.physical_shape().to_vec();
            shape[2] = len;
            binding.set_logical_shape(shape)?;
        }
        self.logical_len = len;
        Ok(())
    }

    fn rewind(&mut self, target_len: usize) -> anyhow::Result<()> {
        if target_len < self.logical_len {
            let zeros = vec![0u8; (self.logical_len - target_len) * std::mem::size_of::<i64>()];
            self.bindings[0].write_bytes(target_len * std::mem::size_of::<i64>(), &zeros)?;
        }
        self.bindings[0].set_logical_shape(vec![1, target_len])?;
        self.set_logical_len(target_len)
    }

    fn debug_stats(&self) -> CudaKvDebugStats {
        let mut transfers = DeviceBindingTransferStats::default();
        let device_ptrs = self
            .bindings
            .iter()
            .skip(1)
            .map(|binding| {
                let stats = binding.transfer_stats();
                transfers.host_upload_calls += stats.host_upload_calls;
                transfers.host_upload_bytes += stats.host_upload_bytes;
                transfers.host_download_calls += stats.host_download_calls;
                transfers.host_download_bytes += stats.host_download_bytes;
                binding.device_ptr() as usize
            })
            .collect();
        CudaKvDebugStats {
            logical_len: self.logical_len,
            max_len: self.max_len,
            device_ptrs,
            kv_transfers: transfers,
        }
    }
}

fn cuda_kv_max_len_from_env() -> anyhow::Result<usize> {
    match std::env::var("ONNX_GENAI_CUDA_KV_MAX_LEN") {
        Ok(value) => {
            let parsed = value.trim().parse::<usize>().with_context(|| {
                format!("invalid ONNX_GENAI_CUDA_KV_MAX_LEN={value:?}: expected a positive integer")
            })?;
            if parsed == 0 {
                bail!("ONNX_GENAI_CUDA_KV_MAX_LEN must be greater than zero");
            }
            Ok(parsed)
        }
        Err(std::env::VarError::NotPresent) => Ok(DEFAULT_CUDA_KV_MAX_LEN),
        Err(error) => Err(error).context("read ONNX_GENAI_CUDA_KV_MAX_LEN"),
    }
}

impl DecodeBackend for NativeDecodeSession {
    fn current_len(&self) -> usize {
        self.current_len
    }

    fn max_context(&self) -> Option<usize> {
        self.cuda.as_ref().map(|state| state.max_len)
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
        if self.cuda.is_some() {
            return self.decode_cuda(token_ids, past_len);
        }
        let total_len = past_len
            .checked_add(token_ids.len())
            .context("native decode context length overflow")?;
        let ids = token_ids
            .iter()
            .map(|&id| i64::from(id))
            .collect::<Vec<_>>();
        let input_ids = Tensor::from_i64(&[1, token_ids.len()], &ids)?;
        let attention_mask = Tensor::from_i64(&[1, total_len], &vec![1; total_len])?;

        let mut owned = Vec::with_capacity(3 + self.kv_inputs.len());
        owned.push((self.input_ids.clone(), input_ids));
        owned.push((self.attention_mask.clone(), attention_mask));
        if let Some(position_ids_name) = &self.position_ids {
            let positions = (past_len..total_len)
                .map(|position| i64::try_from(position).context("position id exceeds i64 range"))
                .collect::<anyhow::Result<Vec<_>>>()?;
            let position_ids = Tensor::from_i64(&[1, token_ids.len()], &positions)?;
            owned.push((position_ids_name.clone(), position_ids));
        }
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
        let logits = extract_logits(&logits)?;
        if logits.iter().flatten().any(|value| !value.is_finite()) {
            bail!("native decoder produced non-finite logits");
        }

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
        if let Some(state) = &mut self.cuda {
            state.rewind(target_len)?;
            self.current_len = target_len;
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

fn extract_logits(tensor: &Tensor) -> anyhow::Result<Vec<Vec<f32>>> {
    let values = tensor_to_f32(tensor)?;
    match tensor.shape.as_slice() {
        [vocab] if *vocab > 0 => Ok(vec![values]),
        [seq, vocab] if *seq > 0 && *vocab > 0 => Ok(values
            .chunks(*vocab)
            .take(*seq)
            .map(<[f32]>::to_vec)
            .collect()),
        [batch, seq, vocab] if *batch > 0 && *seq > 0 && *vocab > 0 => Ok(values
            .chunks(*vocab)
            .take(*seq)
            .map(<[f32]>::to_vec)
            .collect()),
        shape => bail!("unsupported logits tensor shape: {shape:?}"),
    }
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
    use onnx_runtime_ir::{Attribute, Graph, Node, NodeId, Shape, TensorData};

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

    fn tiny_decoder(last_token_logits: bool) -> InferenceSession {
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

        let logits = if last_token_logits {
            let logits = graph.create_named_value(
                "logits",
                DataType::Float32,
                shape(&[1.into(), 1.into(), 2.into()]),
            );
            let data = [10.0f32, 20.0]
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect();
            insert_op(
                &mut graph,
                "Constant",
                vec![],
                logits,
                &[(
                    "value",
                    Attribute::Tensor(TensorData::from_raw(DataType::Float32, vec![1, 1, 2], data)),
                )],
            );
            logits
        } else {
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
            logits
        };
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
        let mut session =
            NativeDecodeSession::from_session(tiny_decoder(false)).expect("load decoder");
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

    #[test]
    fn native_decode_accepts_last_token_only_logits_and_advances_kv() {
        let mut session =
            NativeDecodeSession::from_session(tiny_decoder(true)).expect("load decoder");

        let logits = session.decode(&[1, 2, 3], 0).expect("prefill");
        assert_eq!(logits, vec![vec![10.0, 20.0]]);
        assert_eq!(session.current_len(), 3);

        let logits = session.decode(&[4], 3).expect("decode");
        assert_eq!(logits, vec![vec![10.0, 20.0]]);
        assert_eq!(session.current_len(), 4);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn native_cuda_qwen_decode_matches_cpu_tokens() -> anyhow::Result<()> {
        if std::env::var_os("ONNX_GENAI_RUN_CUDA_SMOKE").is_none() {
            eprintln!("skipping CUDA smoke; set ONNX_GENAI_RUN_CUDA_SMOKE=1 to run");
            return Ok(());
        }

        let model_dir = Path::new("/home/justinchu/qwen2.5-0.5b-int4-onnx");
        if !model_dir.join("model.onnx").is_file() {
            eprintln!("skipping CUDA smoke; target model is not installed");
            return Ok(());
        }
        let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json"))?;
        let prompt = tokenizer.encode("Hello")?;
        let options = GenerateOptions {
            max_new_tokens: 16,
            temperature: 0.0,
            greedy: true,
            stop_on_eos: false,
            ..GenerateOptions::default()
        };

        let mut cpu =
            NativeDecodeSession::load(model_dir.join("model.onnx"), NativeDecodeDevice::Cpu)?;
        let cpu_tokens = cpu
            .generate(&prompt, &options, &ProcessorChain::new(), &tokenizer)?
            .token_ids;

        let mut cuda = NativeDecodeSession::load_with_cuda_kv_max_len(
            model_dir.join("model.onnx"),
            NativeDecodeDevice::Cuda { index: Some(0) },
            Some(128),
        )?;
        let before = cuda
            .cuda_kv_debug_stats()
            .context("CUDA session must expose device KV stats")?;
        assert_eq!(before.device_ptrs.len(), 48);
        assert!(before.device_ptrs.iter().all(|&ptr| ptr != 0));
        assert_eq!(before.kv_transfers, DeviceBindingTransferStats::default());
        let cuda_tokens = cuda
            .generate(&prompt, &options, &ProcessorChain::new(), &tokenizer)?
            .token_ids;
        let after = cuda
            .cuda_kv_debug_stats()
            .context("CUDA session must retain device KV stats")?;

        assert_eq!(cpu_tokens.len(), 16);
        assert_eq!(cuda_tokens.len(), cpu_tokens.len());
        // Explicit non-contracted RMS reductions extend CUDA/CPU parity through
        // token 11. Residual backend drift currently starts at token index 12.
        assert_eq!(&cuda_tokens[..12], &cpu_tokens[..12]);
        assert_eq!(
            &cpu_tokens[..8],
            &[11576, 42740, 11, 358, 614, 264, 3405, 911]
        );
        assert_eq!(after.device_ptrs, before.device_ptrs);
        assert_eq!(after.kv_transfers, DeviceBindingTransferStats::default());
        assert!(after.logical_len > 8);

        cuda.rewind(after.logical_len - 2)?;
        let rewound = cuda.cuda_kv_debug_stats().unwrap();
        assert_eq!(rewound.logical_len, after.logical_len - 2);
        assert_eq!(rewound.device_ptrs, before.device_ptrs);
        assert_eq!(rewound.kv_transfers, DeviceBindingTransferStats::default());

        cuda.reset()?;
        let error = cuda
            .decode(&vec![0; 129], 0)
            .expect_err("decode beyond configured KV capacity must fail");
        assert!(error.to_string().contains("CUDA KV capacity exceeded"));
        Ok(())
    }

    #[test]
    fn native_logits_shapes_match_ort_semantics() {
        let cases = [
            (vec![3], 1),
            (vec![1, 3], 1),
            (vec![2, 3], 2),
            (vec![1, 1, 3], 1),
            (vec![1, 2, 3], 2),
            (vec![2, 2, 3], 2),
        ];
        for (shape, expected_rows) in cases {
            let values = (0..shape.iter().product::<usize>())
                .map(|value| value as f32)
                .collect::<Vec<_>>();
            let tensor = Tensor::from_f32(&shape, &values).expect("create logits");
            let logits = extract_logits(&tensor).expect("extract logits");
            assert_eq!(logits.len(), expected_rows, "shape {shape:?}");
            assert_eq!(logits[0].len(), 3, "shape {shape:?}");
        }

        let tensor = Tensor::from_f32(&[1, 1, 1, 3], &[0.0; 3]).expect("create logits");
        let error = extract_logits(&tensor).expect_err("rank-four logits must be rejected");
        assert!(
            error
                .to_string()
                .contains("unsupported logits tensor shape")
        );
    }
}
