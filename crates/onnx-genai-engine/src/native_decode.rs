//! Native nxrt adapter for the engine's existing decode loop.

use crate::config::{GenerateOptions, GenerateResult, GenerateTokenCallback};
use crate::decode::DecodeBackend;
use crate::decode_loop::{DecodeLoopBackend, DecodeLoopState, run_decode_loop};
use crate::logits::{ProcessorChain, TokenId};
use anyhow::{Context, bail};
use onnx_genai_ort::Tokenizer;
use onnx_runtime_ir::{DataType, DeviceType, Dim};
use onnx_runtime_session::{
    DeviceAllocationCounts, DeviceBindingTransferStats, DeviceGraphCaptureResult, DeviceIoBinding,
    DevicePreference, InferenceSession, Tensor,
};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

/// Device requested for a native decode session.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum NativeDecodeDevice {
    #[default]
    Cpu,
    Cuda {
        index: Option<u32>,
    },
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct NativeDecodeCudaOptions {
    pub kv_max_len: Option<usize>,
    pub graph_capture: Option<bool>,
}

const DEFAULT_CUDA_KV_MAX_LEN: usize = 4096;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CudaKvDebugStats {
    pub logical_len: usize,
    pub max_len: usize,
    pub device_ptrs: Vec<usize>,
    pub kv_transfers: DeviceBindingTransferStats,
    pub graph: CudaGraphDebugStats,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CudaGraphDebugStats {
    pub enabled: bool,
    pub captures: u64,
    pub replays: u64,
    pub fallbacks: u64,
    pub allocation_counts: DeviceAllocationCounts,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DecodeCudaGraphPhase {
    NeedsWarmup,
    Armed,
    Ready,
    Unsupported,
}

struct DecodeCudaState {
    logical_len: usize,
    max_len: usize,
    bindings: Vec<DeviceIoBinding>,
    base_binding_count: usize,
    kv_binding_range: std::ops::Range<usize>,
    input_ids_binding: usize,
    position_ids_binding: Option<usize>,
    logits_binding: usize,
    logits_shape: Vec<usize>,
    graph_enabled: bool,
    graph_phase: DecodeCudaGraphPhase,
    graph_captures: u64,
    graph_replays: u64,
    graph_fallbacks: u64,
    graph_fallback_reason: Option<String>,
}

struct DecodeCudaIo<'a> {
    input_ids: &'a str,
    attention_mask: &'a str,
    position_ids: Option<&'a str>,
    logits: &'a str,
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
        Self::load_with_cuda_options(path, device, NativeDecodeCudaOptions::default())
    }

    pub(crate) fn load_with_weight_offload_host_cache(
        path: impl AsRef<Path>,
        device: NativeDecodeDevice,
        host_cache: onnx_runtime_ep_cpu::WeightOffloadHostCache,
    ) -> anyhow::Result<Self> {
        let preference = match device {
            NativeDecodeDevice::Cpu => DevicePreference::Cpu,
            NativeDecodeDevice::Cuda { index } => DevicePreference::Gpu { index },
        };
        let mut builder = InferenceSession::builder().model(path).device(preference);
        if device == NativeDecodeDevice::Cpu {
            let ep =
                onnx_runtime_ep_cpu::CpuExecutionProvider::initialized_with_weight_offload_host_cache(
                    host_cache,
                )
                .context("initialize native CPU execution provider")?;
            builder = builder.execution_provider(Arc::new(ep));
        }
        let session = builder.build().context("load native decoder model")?;
        Self::from_session_with_cuda_kv_max_len(session, None)
    }

    /// Load with an explicit CUDA KV capacity. `None` uses
    /// `ONNX_GENAI_CUDA_KV_MAX_LEN`, then the 4096-token default.
    pub fn load_with_cuda_kv_max_len(
        path: impl AsRef<Path>,
        device: NativeDecodeDevice,
        cuda_kv_max_len: Option<usize>,
    ) -> anyhow::Result<Self> {
        Self::load_with_cuda_options(
            path,
            device,
            NativeDecodeCudaOptions {
                kv_max_len: cuda_kv_max_len,
                graph_capture: None,
            },
        )
    }

    /// Load with explicit native-CUDA decode options. Unspecified graph capture
    /// follows `ONNX_GENAI_CUDA_GRAPH` and remains disabled by default.
    pub fn load_with_cuda_options(
        path: impl AsRef<Path>,
        device: NativeDecodeDevice,
        options: NativeDecodeCudaOptions,
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
        Self::from_session_with_cuda_options(session, options)
    }

    /// Wrap an already-built native session, validating its decoder-with-past I/O.
    pub fn from_session(session: InferenceSession) -> anyhow::Result<Self> {
        Self::from_session_with_cuda_options(session, NativeDecodeCudaOptions::default())
    }

    fn from_session_with_cuda_kv_max_len(
        session: InferenceSession,
        cuda_kv_max_len: Option<usize>,
    ) -> anyhow::Result<Self> {
        Self::from_session_with_cuda_options(
            session,
            NativeDecodeCudaOptions {
                kv_max_len: cuda_kv_max_len,
                graph_capture: None,
            },
        )
    }

    fn from_session_with_cuda_options(
        mut session: InferenceSession,
        cuda_options: NativeDecodeCudaOptions,
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
            let max_len = match cuda_options.kv_max_len {
                Some(0) => bail!("CUDA KV max length must be greater than zero"),
                Some(value) => value,
                None => cuda_kv_max_len_from_env()?,
            };
            let graph_enabled = cuda_options
                .graph_capture
                .unwrap_or_else(|| onnx_genai_runtime_config::runtime_config().cuda_graph);
            Some(DecodeCudaState::new(
                &mut session,
                DecodeCudaIo {
                    input_ids: &input_ids,
                    attention_mask: &attention_mask,
                    position_ids: position_ids.as_deref(),
                    logits: &logits,
                },
                &present_to_past,
                max_len,
                graph_enabled,
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
        self.cuda
            .as_ref()
            .map(|state| state.debug_stats(&self.session))
    }

    pub fn cuda_graph_fallback_reason(&self) -> Option<&str> {
        self.cuda
            .as_ref()
            .and_then(|state| state.graph_fallback_reason.as_deref())
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
        self.generate_with_callback(prompt_tokens, options, chain, tokenizer, None)
    }

    /// Generate through the shared loop and optionally stream generated tokens.
    pub(crate) fn generate_with_callback(
        &mut self,
        prompt_tokens: &[TokenId],
        options: &GenerateOptions,
        chain: &ProcessorChain,
        tokenizer: &Tokenizer,
        callback: Option<&mut GenerateTokenCallback<'_>>,
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
            callback,
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

        if token_ids.len() == 1 {
            state.write_decode_inputs(token_ids[0], past_len)?;
            if let Err(error) = state.run_one_token(&mut self.session) {
                let diagnosis = diagnose_native_failure(&self.session, &error.to_string());
                bail!("native CUDA decoder forward pass failed{diagnosis}: {error}");
            }
            let logits = state.read_logits()?;
            // Detection-before-consumption: the logits read above is the single
            // per-step device→host sync. Piggyback on it to poll the shared
            // capture-error word (no extra synchronize). If a captured replay
            // violates a device-side bound, kernels latch the flag and avoid the
            // unsafe access, so fail hard before consuming the produced token.
            let capture_error = self.session.check_device_capture_error()?;
            if capture_error != 0 {
                let _ = state.invalidate_graph(&mut self.session);
                bail!(
                    "native CUDA decoder aborted: device capture validation violation (flags=0x{capture_error:x}) detected during captured graph replay; the produced token was rejected before consumption and the decode graph was invalidated"
                );
            }
            if logits.iter().flatten().any(|value| !value.is_finite()) {
                bail!("native decoder produced non-finite logits");
            }
            state.set_logical_len(total_len)?;
            self.current_len = total_len;
            return Ok(logits);
        }

        state.invalidate_graph(&mut self.session)?;
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
            .run_with_device_bindings(&bindings, &mut state.bindings[..state.base_binding_count])
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
        io: DecodeCudaIo<'_>,
        present_to_past: &HashMap<String, String>,
        max_len: usize,
        graph_enabled: bool,
    ) -> anyhow::Result<Self> {
        let mut mask = session.allocate_device_binding(
            io.attention_mask,
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
        let mut bindings = Vec::with_capacity(4 + pairs.len());
        bindings.push(mask);
        let kv_start = bindings.len();
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
        let kv_end = bindings.len();
        let base_binding_count = bindings.len();

        let input_ids_binding = bindings.len();
        bindings.push(session.allocate_device_binding(
            io.input_ids,
            None::<String>,
            DataType::Int64,
            vec![1, 1],
            vec![1, 1],
        )?);
        let position_ids_binding = if let Some(position_ids) = io.position_ids {
            let index = bindings.len();
            bindings.push(session.allocate_device_binding(
                position_ids,
                None::<String>,
                DataType::Int64,
                vec![1, 1],
                vec![1, 1],
            )?);
            Some(index)
        } else {
            None
        };

        let logits_meta = session
            .outputs()
            .iter()
            .find(|meta| meta.name == io.logits)
            .with_context(|| format!("missing CUDA logits output metadata for '{}'", io.logits))?;
        if logits_meta.dtype != DataType::Float32 || logits_meta.shape.is_empty() {
            bail!(
                "CUDA logits output '{}' must be non-scalar f32, got {:?} {:?}",
                io.logits,
                logits_meta.dtype,
                logits_meta.shape
            );
        }
        let logits_shape = logits_meta
            .shape
            .iter()
            .map(|dim| match dim {
                Dim::Static(value) => *value,
                Dim::Symbolic(_) => 1,
            })
            .collect::<Vec<_>>();
        let logits_binding = bindings.len();
        bindings.push(session.allocate_device_output_binding(
            io.logits,
            DataType::Float32,
            logits_shape.clone(),
            logits_shape.clone(),
        )?);

        Ok(Self {
            logical_len: 0,
            max_len,
            bindings,
            base_binding_count,
            kv_binding_range: kv_start..kv_end,
            input_ids_binding,
            position_ids_binding,
            logits_binding,
            logits_shape,
            graph_enabled,
            graph_phase: DecodeCudaGraphPhase::NeedsWarmup,
            graph_captures: 0,
            graph_replays: 0,
            graph_fallbacks: 0,
            graph_fallback_reason: None,
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
        for binding in &mut self.bindings[self.kv_binding_range.clone()] {
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

    fn write_decode_inputs(&mut self, token_id: TokenId, position: usize) -> anyhow::Result<()> {
        self.bindings[self.input_ids_binding].write_bytes(0, &i64::from(token_id).to_le_bytes())?;
        if let Some(index) = self.position_ids_binding {
            let position = i64::try_from(position).context("position id exceeds i64 range")?;
            self.bindings[index].write_bytes(0, &position.to_le_bytes())?;
        }
        Ok(())
    }

    fn run_one_token(&mut self, session: &mut InferenceSession) -> anyhow::Result<()> {
        if !self.graph_enabled {
            session.run_with_device_bindings(&[], &mut self.bindings)?;
            return Ok(());
        }

        match self.graph_phase {
            DecodeCudaGraphPhase::NeedsWarmup => {
                session.run_with_device_bindings(&[], &mut self.bindings)?;
                self.graph_phase = DecodeCudaGraphPhase::Armed;
            }
            DecodeCudaGraphPhase::Armed => {
                match session.try_capture_with_device_bindings(&[], &mut self.bindings)? {
                    DeviceGraphCaptureResult::Captured(outputs) => {
                        if outputs.iter().any(Option::is_some) {
                            bail!("captured CUDA decode unexpectedly materialized a host output");
                        }
                        self.graph_captures += 1;
                        self.graph_phase = DecodeCudaGraphPhase::Ready;
                    }
                    DeviceGraphCaptureResult::NotCapturable(reason) => {
                        self.graph_fallbacks += 1;
                        self.graph_phase = DecodeCudaGraphPhase::Unsupported;
                        self.graph_fallback_reason = Some(reason.clone());
                        tracing::warn!(
                            "native CUDA decode graph capture disabled for this generation: {reason}"
                        );
                        session.run_with_device_bindings(&[], &mut self.bindings)?;
                    }
                }
            }
            DecodeCudaGraphPhase::Ready => {
                session.replay_device_graph(&mut self.bindings)?;
                self.graph_replays += 1;
            }
            DecodeCudaGraphPhase::Unsupported => {
                session.run_with_device_bindings(&[], &mut self.bindings)?;
            }
        }
        Ok(())
    }

    fn read_logits(&mut self) -> anyhow::Result<Vec<Vec<f32>>> {
        let bytes = self.bindings[self.logits_binding].read_bytes()?;
        let logits = Tensor::from_raw(DataType::Float32, self.logits_shape.clone(), &bytes)?;
        extract_logits(&logits)
    }

    fn invalidate_graph(&mut self, session: &mut InferenceSession) -> anyhow::Result<()> {
        session.reset_device_graph()?;
        self.graph_phase = DecodeCudaGraphPhase::NeedsWarmup;
        Ok(())
    }

    fn debug_stats(&self, session: &InferenceSession) -> CudaKvDebugStats {
        let mut transfers = DeviceBindingTransferStats::default();
        let device_ptrs = self.bindings[self.kv_binding_range.clone()]
            .iter()
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
            graph: CudaGraphDebugStats {
                enabled: self.graph_enabled,
                captures: self.graph_captures,
                replays: self.graph_replays,
                fallbacks: self.graph_fallbacks,
                allocation_counts: session.device_allocation_counts().unwrap_or_default(),
            },
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
        // Single-token CPU decode: run the whole forward inside one decode-pool
        // installation so the ~121 per-op `MatMulNBits` projections execute
        // inline on resident workers instead of each re-installing the pool
        // (eliminating the per-op fork-join crossing). Prefill (M>1) and the CUDA
        // path (handled above) must keep using the global pool, so gate on M==1.
        let run_result = if token_ids.len() == 1 {
            onnx_runtime_ep_cpu::with_decode_pool_scope(|| self.session.run(&bindings))
        } else {
            self.session.run(&bindings)
        };
        let outputs = match run_result {
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
            state.invalidate_graph(&mut self.session)?;
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

impl Drop for NativeDecodeSession {
    fn drop(&mut self) {
        if let Some(state) = &mut self.cuda {
            let _ = state.invalidate_graph(&mut self.session);
        }
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

    #[cfg(feature = "cuda")]
    fn capture_safe_cuda_decoder(
        graph_capture: bool,
        max_len: usize,
    ) -> anyhow::Result<NativeDecodeSession> {
        use prost::Message;

        let mut graph = Graph::new();
        graph.opset_imports.insert(String::new(), 13);
        let total = graph.intern_symbol("total");
        let past = graph.intern_symbol("past");

        let input_ids =
            graph.create_named_value("input_ids", DataType::Int64, vec![1.into(), 1.into()]);
        let attention_mask = graph.create_named_value(
            "attention_mask",
            DataType::Int64,
            vec![1.into(), total.into()],
        );
        let position_ids =
            graph.create_named_value("position_ids", DataType::Int64, vec![1.into(), 1.into()]);
        let past_key = graph.create_named_value(
            "past_key_values.0.key",
            DataType::Float32,
            vec![1.into(), 1.into(), past.into(), 1.into()],
        );
        let past_value = graph.create_named_value(
            "past_key_values.0.value",
            DataType::Float32,
            vec![1.into(), 1.into(), past.into(), 1.into()],
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

        let logits =
            graph.create_named_value("logits", DataType::Float32, vec![1.into(), 1.into()]);
        insert_op(
            &mut graph,
            "Cast",
            vec![input_ids],
            logits,
            &[("to", Attribute::Int(DataType::Float32 as i64))],
        );
        let present_key = graph.create_named_value(
            "present.0.key",
            DataType::Float32,
            vec![1.into(), 1.into(), past.into(), 1.into()],
        );
        insert_op(
            &mut graph,
            "Cast",
            vec![past_key],
            present_key,
            &[("to", Attribute::Int(DataType::Float32 as i64))],
        );
        let present_value = graph.create_named_value(
            "present.0.value",
            DataType::Float32,
            vec![1.into(), 1.into(), past.into(), 1.into()],
        );
        insert_op(
            &mut graph,
            "Cast",
            vec![past_value],
            present_value,
            &[("to", Attribute::Int(DataType::Float32 as i64))],
        );
        for output in [logits, present_key, present_value] {
            graph.add_output(output);
        }

        let model = onnx_rs::Model::new(graph).to_proto()?.encode_to_vec();
        let session = InferenceSession::builder()
            .model_bytes(&model)
            .device(DevicePreference::Gpu { index: Some(0) })
            .build()
            .context("build capture-safe CUDA decoder")?;
        NativeDecodeSession::from_session_with_cuda_options(
            session,
            NativeDecodeCudaOptions {
                kv_max_len: Some(max_len),
                graph_capture: Some(graph_capture),
            },
        )
    }

    #[cfg(feature = "cuda")]
    fn binding_addresses(session: &NativeDecodeSession) -> Vec<usize> {
        session
            .cuda
            .as_ref()
            .expect("CUDA state")
            .bindings
            .iter()
            .map(|binding| binding.device_ptr() as usize)
            .collect()
    }

    #[cfg(feature = "cuda")]
    fn input_update_stats(session: &NativeDecodeSession) -> [DeviceBindingTransferStats; 3] {
        let state = session.cuda.as_ref().expect("CUDA state");
        [
            state.bindings[0].transfer_stats(),
            state.bindings[state.input_ids_binding].transfer_stats(),
            state.bindings[state.position_ids_binding.expect("position_ids binding")]
                .transfer_stats(),
        ]
    }

    #[cfg(feature = "cuda")]
    fn assert_single_value_uploads(
        before: [DeviceBindingTransferStats; 3],
        after: [DeviceBindingTransferStats; 3],
    ) {
        for (before, after) in before.into_iter().zip(after) {
            assert_eq!(after.host_upload_calls, before.host_upload_calls + 1);
            assert_eq!(
                after.host_upload_bytes,
                before.host_upload_bytes + std::mem::size_of::<i64>() as u64
            );
        }
    }

    #[cfg(feature = "cuda")]
    fn assert_decode_bindings(
        session: &mut NativeDecodeSession,
        addresses: &[usize],
        token: TokenId,
        position: usize,
        max_len: usize,
    ) -> anyhow::Result<()> {
        assert_eq!(binding_addresses(session), addresses);
        let state = session.cuda.as_mut().expect("CUDA state");

        let input = state.bindings[state.input_ids_binding].read_bytes()?;
        assert_eq!(
            i64::from_le_bytes(input.try_into().expect("one input id")),
            i64::from(token)
        );

        let position_bytes = state.bindings
            [state.position_ids_binding.expect("position_ids binding")]
        .read_bytes()?;
        assert_eq!(
            i64::from_le_bytes(position_bytes.try_into().expect("one position id")),
            position as i64
        );

        let mask = state.bindings[0]
            .read_bytes()?
            .chunks_exact(std::mem::size_of::<i64>())
            .map(|bytes| i64::from_le_bytes(bytes.try_into().unwrap()))
            .collect::<Vec<_>>();
        assert_eq!(mask.len(), max_len);
        assert!(mask[..=position].iter().all(|&value| value == 1));
        assert!(mask[position + 1..].iter().all(|&value| value == 0));
        assert_eq!(state.bindings[0].logical_shape(), &[1, position + 1]);
        for binding in &state.bindings[state.kv_binding_range.clone()] {
            assert_eq!(binding.logical_shape()[2], position + 1);
        }
        Ok(())
    }

    #[cfg(feature = "cuda")]
    fn run_capture_safe_decode(
        session: &mut NativeDecodeSession,
        tokens: &[TokenId],
        addresses: &[usize],
        max_len: usize,
    ) -> anyhow::Result<Vec<Vec<u32>>> {
        let mut logits = Vec::with_capacity(tokens.len());
        for (position, &token) in tokens.iter().enumerate() {
            let before = input_update_stats(session);
            let step = session.decode(&[token], position)?;
            let after = input_update_stats(session);
            assert_single_value_uploads(before, after);
            assert_decode_bindings(session, addresses, token, position, max_len)?;
            logits.push(
                step.into_iter()
                    .flatten()
                    .map(f32::to_bits)
                    .collect::<Vec<_>>(),
            );
        }
        Ok(logits)
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

    #[cfg(feature = "cuda")]
    #[test]
    fn native_cuda_capture_replay_is_bit_exact_and_refreshes_decode_inputs() -> anyhow::Result<()> {
        if std::env::var_os("ONNX_GENAI_RUN_CUDA_SMOKE").is_none() {
            eprintln!("skipping CUDA smoke; set ONNX_GENAI_RUN_CUDA_SMOKE=1 to run");
            return Ok(());
        }

        const MAX_LEN: usize = 16;
        const TOKENS: [TokenId; 10] = [3, 17, 5, 29, 11, 23, 7, 31, 13, 2];

        let mut eager = capture_safe_cuda_decoder(false, MAX_LEN)?;
        let eager_addresses = binding_addresses(&eager);
        let eager_first = run_capture_safe_decode(&mut eager, &TOKENS, &eager_addresses, MAX_LEN)?;
        let eager_stats = eager.cuda_kv_debug_stats().expect("CUDA stats");
        assert!(!eager_stats.graph.enabled);
        assert_eq!(eager_stats.graph.captures, 0);
        assert_eq!(eager_stats.graph.replays, 0);
        assert_eq!(eager_stats.graph.fallbacks, 0);
        assert!(eager.cuda_graph_fallback_reason().is_none());

        let mut captured = capture_safe_cuda_decoder(true, MAX_LEN)?;
        let captured_addresses = binding_addresses(&captured);
        let captured_first =
            run_capture_safe_decode(&mut captured, &TOKENS, &captured_addresses, MAX_LEN)?;
        let first_stats = captured.cuda_kv_debug_stats().expect("CUDA stats");
        assert!(first_stats.graph.enabled);
        assert_eq!(first_stats.graph.captures, 1);
        assert_eq!(first_stats.graph.replays, TOKENS.len() as u64 - 2);
        assert_eq!(first_stats.graph.fallbacks, 0);
        assert!(captured.cuda_graph_fallback_reason().is_none());
        assert_eq!(captured_first, eager_first);
        assert_eq!(
            captured_first,
            TOKENS
                .iter()
                .map(|&token| vec![(token as f32).to_bits()])
                .collect::<Vec<_>>()
        );
        assert_eq!(captured_addresses, binding_addresses(&captured));
        assert_eq!(
            first_stats.kv_transfers,
            DeviceBindingTransferStats::default()
        );

        eager.reset()?;
        captured.reset()?;
        let eager_second = run_capture_safe_decode(&mut eager, &TOKENS, &eager_addresses, MAX_LEN)?;
        let captured_second =
            run_capture_safe_decode(&mut captured, &TOKENS, &captured_addresses, MAX_LEN)?;
        let second_stats = captured.cuda_kv_debug_stats().expect("CUDA stats");
        assert_eq!(captured_second, eager_second);
        assert_eq!(captured_second, captured_first);
        assert_eq!(second_stats.graph.captures, 2);
        assert_eq!(second_stats.graph.replays, 2 * (TOKENS.len() as u64 - 2));
        assert_eq!(second_stats.graph.fallbacks, 0);
        assert_eq!(captured_addresses, binding_addresses(&captured));
        assert_eq!(
            second_stats.kv_transfers,
            DeviceBindingTransferStats::default()
        );

        eprintln!(
            "native CUDA capture-safe decode parity: captures={} replays={} fallbacks={} steps_per_generation={}",
            second_stats.graph.captures,
            second_stats.graph.replays,
            second_stats.graph.fallbacks,
            TOKENS.len()
        );
        Ok(())
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
        const HORIZON: usize = 64;
        let generate = |session: &mut NativeDecodeSession| -> anyhow::Result<(Vec<TokenId>, u128)> {
            let mut logits = session
                .decode(&prompt, 0)?
                .pop()
                .context("prefill must produce logits")?;
            let mut tokens = Vec::with_capacity(HORIZON);
            let mut decode_nanos = 0u128;
            for step in 0..HORIZON {
                let token = logits
                    .iter()
                    .enumerate()
                    .max_by(|(_, left), (_, right)| left.total_cmp(right))
                    .map(|(index, _)| index as TokenId)
                    .context("logits must not be empty")?;
                tokens.push(token);
                if step + 1 == HORIZON {
                    break;
                }
                let start = std::time::Instant::now();
                logits = session
                    .decode(&[token], session.current_len())?
                    .pop()
                    .context("decode must produce logits")?;
                decode_nanos += start.elapsed().as_nanos();
            }
            Ok((tokens, decode_nanos))
        };

        let mut cpu =
            NativeDecodeSession::load(model_dir.join("model.onnx"), NativeDecodeDevice::Cpu)?;
        let (cpu_tokens, _) = generate(&mut cpu)?;
        drop(cpu);

        let mut eager = NativeDecodeSession::load_with_cuda_options(
            model_dir.join("model.onnx"),
            NativeDecodeDevice::Cuda { index: Some(0) },
            NativeDecodeCudaOptions {
                kv_max_len: Some(128),
                graph_capture: Some(false),
            },
        )?;
        let eager_before = eager
            .cuda_kv_debug_stats()
            .context("CUDA session must expose device KV stats")?;
        let (eager_tokens, eager_nanos) = generate(&mut eager)?;
        let eager_after = eager.cuda_kv_debug_stats().unwrap();
        assert!(!eager_after.graph.enabled);
        assert_eq!(eager_after.graph.captures, 0);
        assert_eq!(eager_after.graph.replays, 0);
        drop(eager);

        let mut captured = NativeDecodeSession::load_with_cuda_options(
            model_dir.join("model.onnx"),
            NativeDecodeDevice::Cuda { index: Some(0) },
            NativeDecodeCudaOptions {
                kv_max_len: Some(128),
                graph_capture: Some(true),
            },
        )?;
        let captured_before = captured.cuda_kv_debug_stats().unwrap();
        let (captured_tokens, captured_nanos) = generate(&mut captured)?;
        let captured_after = captured.cuda_kv_debug_stats().unwrap();

        assert_eq!(cpu_tokens.len(), HORIZON);
        assert_eq!(eager_tokens, cpu_tokens);
        assert_eq!(captured_tokens, eager_tokens);
        assert_eq!(
            &cpu_tokens[..8],
            &[11576, 42740, 11, 358, 614, 264, 3405, 911]
        );
        assert_eq!(eager_before.device_ptrs, eager_after.device_ptrs);
        assert_eq!(captured_before.device_ptrs, captured_after.device_ptrs);
        assert_eq!(
            captured_after.kv_transfers,
            DeviceBindingTransferStats::default()
        );
        assert!(captured_after.graph.enabled);
        assert_eq!(captured_after.graph.captures, 0);
        assert_eq!(captured_after.graph.replays, 0);
        assert_eq!(captured_after.graph.fallbacks, 1);
        let fallback_reason = captured
            .cuda_graph_fallback_reason()
            .context("Qwen graph fallback must expose its diagnostic")?;
        assert!(fallback_reason.contains("GroupQueryAttention"));
        assert!(fallback_reason.contains("data-dependent output shape"));

        let eager_us = eager_nanos as f64 / (HORIZON - 1) as f64 / 1000.0;
        let captured_us = captured_nanos as f64 / (HORIZON - 1) as f64 / 1000.0;
        eprintln!(
            "native CUDA decode wall-time: eager={eager_us:.1} us/token, graph-flag={captured_us:.1} us/token, delta={:.1}%",
            (captured_us / eager_us - 1.0) * 100.0
        );

        captured.rewind(captured_after.logical_len - 2)?;
        let rewound = captured.cuda_kv_debug_stats().unwrap();
        assert_eq!(rewound.logical_len, captured_after.logical_len - 2);
        assert_eq!(rewound.device_ptrs, captured_before.device_ptrs);
        assert_eq!(rewound.kv_transfers, DeviceBindingTransferStats::default());

        captured.reset()?;
        let (second_tokens, _) = generate(&mut captured)?;
        assert_eq!(second_tokens, captured_tokens);

        captured.reset()?;
        let error = captured
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
