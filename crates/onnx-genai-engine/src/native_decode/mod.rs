//! Native nxrt adapter for the engine's existing decode loop.

use crate::config::{GenerateOptions, GenerateResult, GenerateTokenCallback};
use crate::decode::DecodeBackend;
use crate::decode_loop::{DecodeLoopBackend, DecodeLoopState, run_decode_loop};
use crate::logits::{ProcessorChain, TokenId};
use crate::sampling::sample_greedy;
use anyhow::{Context, bail};
use onnx_genai_metadata::{KvOwnership, ModelIoSpec, SequenceInputKind, SharedKvGroup};
use onnx_genai_ort::Tokenizer;
use onnx_genai_ort::decode_contract::{
    KvNamingConvention, has_past_prefix, has_present_prefix, matching_past_input,
};
use onnx_runtime_ir::{DataType, DeviceType, Dim, SymbolId};
use onnx_runtime_session::{
    CaptureDeclineReport, DecodePrecision, DeviceAllocationCounts, DeviceBindingTransferStats,
    DeviceGraphCaptureResult, DeviceIoBinding, DevicePreference, InferenceSession, Tensor,
};
use onnx_runtime_tracer::{Args, TraceContext, capture_rejected};
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::Arc;

mod backend;
mod cpu;
mod cuda;
mod io;
mod load;
mod paged_gqa;
mod proposer;
mod tensor;
#[cfg(test)]
mod tests;

use backend::*;
use cpu::DecodeCpuKvState;
use cpu::*;
use cuda::DecodeCudaState;
use cuda::*;
pub use cuda::{CudaGraphDebugStats, CudaKvDebugStats};
use io::*;
pub use paged_gqa::{
    GQA_PRESENT_ALLOCATIONS, PagedGqaConfig, flat_gqa_decode_step, gqa_present_allocations,
    paged_gqa_decode_step,
};
pub(crate) use proposer::NativeProposerSession;
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
use tensor::*;

/// Counter proving the incremental-prefill fast path fires.
pub static NATIVE_SESSION_INCREMENTAL_PREFILL_TEST_HITS: AtomicU64 = AtomicU64::new(0);

/// Device requested for a native decode session.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub enum NativeDecodeDevice {
    #[default]
    Cpu,
    Cuda {
        index: Option<u32>,
    },
    Plugin {
        library: std::path::PathBuf,
        registration_name: Option<String>,
        provider_name: String,
    },
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct NativeDecodeCudaOptions {
    pub kv_max_len: Option<usize>,
    pub graph_capture: Option<bool>,
}

/// Stateful decoder-with-past adapter over the pure-Rust native runtime.
pub struct NativeDecodeSession {
    session: InferenceSession,
    step_inputs: Vec<NativeStepInputBinding>,
    logits: String,
    hidden_output: Option<String>,
    kv_inputs: Vec<String>,
    present_to_past: HashMap<String, String>,
    past: HashMap<String, Tensor>,
    cuda: Option<DecodeCudaState>,
    cpu_kv: Option<DecodeCpuKvState>,
    trace: TraceContext,
    pub(crate) current_len: usize,
    last_hidden: Option<Vec<f32>>,
    uses_decode_pool: bool,
    has_plugin_fused: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum NativeStepInputSource {
    TokenIds,
    InputsEmbeds,
    AttentionMask,
    PositionIds,
    Routed,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct NativeStepInputBinding {
    name: String,
    source: NativeStepInputSource,
}

impl NativeDecodeSession {
    /// Load a decoder-with-past ONNX model on the requested native device.
    pub fn load(path: impl AsRef<Path>, device: NativeDecodeDevice) -> anyhow::Result<Self> {
        Self::load_with_cuda_options(path, device, NativeDecodeCudaOptions::default())
    }
}

impl NativeDecodeSession {
    pub fn current_len(&self) -> usize {
        self.current_len
    }

    pub fn kv_layer_count(&self) -> usize {
        self.kv_inputs.len() / 2
    }

    /// Last target hidden-state row produced by the most recent forward.
    pub(crate) fn last_hidden(&self) -> Option<&[f32]> {
        self.last_hidden.as_deref()
    }

    /// Materialize metadata-declared shared-KV references from the target's
    /// current host cache. Native CUDA keeps KV device-resident and is not yet
    /// exposed through this CPU tensor contract.
    pub(crate) fn shared_kv_inputs(
        &self,
        groups: &[SharedKvGroup],
    ) -> anyhow::Result<Vec<(String, Tensor)>> {
        if self.cuda.is_some() {
            bail!(
                "native shared-KV proposer execution currently requires a CPU target session; CUDA target KV references need device-binding alias support"
            );
        }
        let mut inputs = Vec::with_capacity(groups.len() * 2);
        for group in groups {
            let key_target = group.target_key_input.as_deref().with_context(|| {
                format!(
                    "shared_kv group '{}' is missing target_key_input; declare the exact target decoder KV input name",
                    group.name
                )
            })?;
            let value_target = group.target_value_input.as_deref().with_context(|| {
                format!(
                    "shared_kv group '{}' is missing target_value_input; declare the exact target decoder KV input name",
                    group.name
                )
            })?;
            let key_input = group.key_input.as_deref().with_context(|| {
                format!(
                    "shared_kv group '{}' is missing key_input; declare the exact proposer input name",
                    group.name
                )
            })?;
            let value_input = group.value_input.as_deref().with_context(|| {
                format!(
                    "shared_kv group '{}' is missing value_input; declare the exact proposer input name",
                    group.name
                )
            })?;
            let key = self.past.get(key_target).with_context(|| {
                format!(
                    "target shared-KV key '{}' for group '{}' is unavailable; run the target decoder before invoking the proposer and ensure io.kv_inputs names this cache",
                    key_target, group.name
                )
            })?;
            let value = self.past.get(value_target).with_context(|| {
                format!(
                    "target shared-KV value '{}' for group '{}' is unavailable; run the target decoder before invoking the proposer and ensure io.kv_inputs names this cache",
                    value_target, group.name
                )
            })?;
            inputs.push((key_input.to_owned(), key.clone()));
            inputs.push((value_input.to_owned(), value.clone()));
        }
        Ok(inputs)
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

    /// Structural reasons, if any, why CUDA graph capture was declined at
    /// binding time because an auxiliary graph output carries an unresolved
    /// symbolic dimension (not batch or query-seq) that cannot be collapsed to
    /// a fixed persistent device binding. Empty when every auxiliary output was
    /// statically bindable. Decode still runs eagerly when this is non-empty.
    pub fn cuda_auxiliary_bind_declines(&self) -> &[String] {
        self.cuda
            .as_ref()
            .map(|state| state.auxiliary_bind_declines.as_slice())
            .unwrap_or(&[])
    }

    /// Structured reasons from the most recent CUDA graph fallback.
    pub fn cuda_graph_fallback_report(&self) -> Option<&CaptureDeclineReport> {
        self.cuda
            .as_ref()
            .and_then(|state| state.graph_fallback_report.as_ref())
    }

    /// Attach the shared runtime trace context used for capture-fallback events
    /// and per-op executor spans (kernel-variant + capture-rejection reasons).
    pub fn set_trace_context(&mut self, trace: TraceContext) {
        self.session.set_trace_context(trace.clone());
        self.trace = trace;
    }

    /// Dormant option (c) bring-up control (WP4): arm the padded single M=maxK
    /// captured verify graph and retain the captured graph across `rewind`. No-op
    /// on non-CUDA sessions. Not wired into any live decode path yet; exercised
    /// only by the option-(c) rewind-correctness tests.
    #[cfg(test)]
    pub(crate) fn configure_padded_verify_capture(&mut self, max_query_rows: usize) {
        if let Some(state) = self.cuda.as_mut() {
            state.configure_padded_verify_capture(max_query_rows);
        }
    }

    /// Toggle the option (c) "rewind retains the captured graph" guard directly.
    /// Dormant: bring-up / correctness tests only.
    #[cfg(test)]
    pub(crate) fn set_retain_graph_on_rewind(&mut self, retain: bool) {
        if let Some(state) = self.cuda.as_mut() {
            state.set_retain_graph_on_rewind(retain);
        }
    }

    /// Fixed query-row capacity of the dormant padded verify capture, or `None`.
    #[cfg(test)]
    pub(crate) fn padded_query_capacity(&self) -> Option<usize> {
        self.cuda
            .as_ref()
            .and_then(DecodeCudaState::padded_query_capacity)
    }

    pub fn decode(
        &mut self,
        token_ids: &[TokenId],
        past_len: usize,
    ) -> anyhow::Result<Vec<Vec<f32>>> {
        <Self as DecodeBackend>::decode(self, token_ids, past_len)
    }

    /// Run one target step with arbitrary named tensors supplied by pipeline
    /// routing. Generated roles (token ids, attention mask, and position ids)
    /// come from `ModelIoSpec`; every other non-KV graph input is resolved by its
    /// exact graph port name from `step_inputs`.
    pub(crate) fn decode_with_step_inputs(
        &mut self,
        token_ids: &[TokenId],
        past_len: usize,
        step_inputs: &[(String, Tensor)],
    ) -> anyhow::Result<Vec<Vec<f32>>> {
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
            if !step_inputs.is_empty() {
                bail!(
                    "native CUDA target decode does not yet accept routed host step inputs; use the CPU native device until generic device bindings are implemented"
                );
            }
            return self.decode_cuda(token_ids, past_len);
        }
        if self.cpu_kv.is_some() {
            return match self.decode_cpu_inplace(token_ids, past_len, false, step_inputs)? {
                NativeCpuDecodeResult::Logits(logits) => Ok(logits),
                NativeCpuDecodeResult::Token(_) => unreachable!("logits decode requested"),
            };
        }
        match self.decode_cpu(token_ids, past_len, false, step_inputs)? {
            NativeCpuDecodeResult::Logits(logits) => Ok(logits),
            NativeCpuDecodeResult::Token(_) => unreachable!("logits decode requested"),
        }
    }

    /// Rewind by prefix-slicing every carried host KV tensor.
    pub fn rewind(&mut self, target_len: usize) -> anyhow::Result<()> {
        self.rewind_inner(target_len)
    }

    pub fn reset(&mut self) -> anyhow::Result<()> {
        self.rewind(0)
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

    /// Generate incrementally: reuse KV state up to `resume_from` and only
    /// prefill `prompt_tokens[resume_from..]`. The caller must ensure that
    /// `prompt_tokens[..resume_from]` matches the tokens already in the KV cache.
    ///
    /// If `resume_from > current_len`, behaves like full generation from 0.
    /// If `resume_from < current_len`, rewinds the KV cache to `resume_from`.
    pub(crate) fn generate_incremental_with_callback(
        &mut self,
        prompt_tokens: &[TokenId],
        resume_from: usize,
        options: &GenerateOptions,
        chain: &ProcessorChain,
        tokenizer: &Tokenizer,
        callback: Option<&mut GenerateTokenCallback<'_>>,
    ) -> anyhow::Result<GenerateResult> {
        if prompt_tokens.is_empty() {
            bail!("native generation requires at least one prompt token");
        }
        let resume_from = resume_from.min(prompt_tokens.len());
        if resume_from == 0 || resume_from > self.current_len {
            // Full reset path — no valid KV prefix to reuse.
            return self.generate_with_callback(prompt_tokens, options, chain, tokenizer, callback);
        }
        // Rewind KV to resume_from if we've advanced beyond it (e.g. diverged prefix).
        if self.current_len > resume_from {
            self.rewind(resume_from)?;
        }
        NATIVE_SESSION_INCREMENTAL_PREFILL_TEST_HITS.fetch_add(1, AtomicOrdering::Relaxed);

        let new_tokens = &prompt_tokens[resume_from..];
        if new_tokens.is_empty() {
            bail!(
                "incremental generation requires at least one new token beyond the cached prefix"
            );
        }
        let mut backend = NativeLoopAdapter {
            session: self,
            prompt_tokens: prompt_tokens.to_vec(),
            pending_tokens: new_tokens.to_vec(),
        };
        let mut state = DecodeLoopState::new(resume_from, options.seed, options.top_logprobs);
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
        make_empty_input_tensor(&self.session, name)
    }

    fn step_input_name(&self, source: NativeStepInputSource) -> Option<&str> {
        self.step_inputs
            .iter()
            .find(|binding| binding.source == source)
            .map(|binding| binding.name.as_str())
    }
}
