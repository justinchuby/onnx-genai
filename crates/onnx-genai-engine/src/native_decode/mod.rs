//! Native nxrt adapter for the engine's existing decode loop.

use crate::config::{GenerateOptions, GenerateResult, GenerateTokenCallback};
use crate::decode::DecodeBackend;
use crate::decode_loop::{DecodeLoopBackend, DecodeLoopState, run_decode_loop};
use crate::logits::{ProcessorChain, TokenId};
use crate::sampling::sample_greedy;
use anyhow::{Context, bail};
use onnx_genai_metadata::{
    KvOwnership, LoopStatePair, ModelIoSpec, SequenceInputKind, SharedKvGroup,
};
use onnx_genai_ort::Tokenizer;
use onnx_runtime_ir::{DataType, DeviceType, Dim, SymbolId};
use onnx_runtime_session::{
    CaptureDeclineReport, DecodePrecision, DeviceAllocationCounts, DeviceBindingTransferStats,
    DeviceGraphCaptureResult, DeviceIoBinding, DevicePreference, InferenceSession, Tensor,
};
use onnx_runtime_tracer::{Args, TraceContext, capture_rejected};
use std::collections::{BTreeMap, HashMap, HashSet};
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
/// Process-global count of decode steps that took the Inc3c captured
/// per-step-input branch (as opposed to the eager owned branch). This is a test
/// instrumentation seam only: it lets a parity test prove *non-tautologically*
/// that enabling the flag genuinely routes the `inputs_embeds`/routed decode
/// step through the captured graph path instead of silently declining to eager.
/// The captured and eager branches are distinct functions, so a non-zero delta
/// here after a run with the flag on — and a zero delta with it off — is direct
/// evidence the intended path executed.
pub static NATIVE_DECODER_CAPTURED_STEP_INPUT_DECODES: AtomicU64 = AtomicU64::new(0);
pub static NATIVE_SESSION_INCREMENTAL_PREFILL_TEST_HITS: AtomicU64 = AtomicU64::new(0);

/// Device requested for a native decode session.
///
/// Defined outside this module so ungated code can name it; re-exported here so
/// every existing `native_decode::NativeDecodeDevice` path still resolves.
pub use crate::native_decode_device::NativeDecodeDevice;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct NativeDecodeCudaOptions {
    pub kv_max_len: Option<usize>,
    pub metadata_max_len: Option<usize>,
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
    /// Coordinate rank of the `position_ids` input, derived from the graph's
    /// declared physical shape (`1` = conventional `[1, S]`; `N > 1` = multi-axis
    /// mrope `[N, 1, S]`). All position tensors this session builds honor it, so
    /// a rank-3 mrope decoder gets rank-3 coordinates while a rank-2 decoder is
    /// byte-identical to before.
    position_rank: usize,
    /// Inc-1b PR-2 decode-inline plan state. Graph-property gated: the sibling
    /// is built (and single-token decode steps routed to it) automatically iff
    /// the model's decode graph has an inlineable single-trip (extent==1)
    /// recurrent `Scan` that `inline_single_trip_scan_bodies` can lower. For a
    /// model with no such `Scan` this latches `Disabled` and every decode step
    /// uses today's Scan child-session executor unchanged (no sibling built).
    decode_inline: DecodeInlineState,
}

/// State of the Inc-1b PR-2 decode-specialized inlined-body plan for a session.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DecodeInlineState {
    /// Not yet probed (the sibling build is lazy, at first decode step).
    Untried,
    /// The model has no single-trip-eligible `Scan` (a dense / non-hybrid
    /// decoder): use the main (Scan child-session) executor for every step.
    Disabled,
    /// A decode-inline sibling executor is built; route single-token decode
    /// steps to it (extent≠1 steps still fall back to the main executor).
    Enabled,
}

/// Pure decision for whether a decode step routes to the decode-inline sibling
/// exec (Inc-1b PR-2/PR-3). Encapsulates guard #2: the decode-inline plan is a
/// single-iteration (scan-axis extent 1) specialization, so it is used only for
/// a single-token step and only when a sibling was actually built; every
/// multi-token (extent≠1) step falls back to the main Scan executor.
///
/// `has_eager_step_inputs` is the PR-3 scope-lock (Harry #588 rec #4): a decoder
/// with `inputs_embeds`/routed ports is never routed to the sibling regardless of
/// state/readiness/token-count, because those per-step ports are served by the
/// eager/captured step-input paths, not the token fast path the sibling
/// specializes. This keeps the author's stated scope locked by construction.
fn route_decode_inline_decision(
    state: DecodeInlineState,
    sibling_ready: bool,
    token_count: usize,
    has_eager_step_inputs: bool,
) -> bool {
    !has_eager_step_inputs
        && state == DecodeInlineState::Enabled
        && sibling_ready
        && token_count == 1
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

    /// Whether this session keeps its self-attention KV as plain host tensors
    /// that can be read out for paged present-KV mirroring and re-seeded from a
    /// materialized paged prefix (GAP-3 Inc-C).
    ///
    /// True only for the host-resident *growable* CPU path: the CUDA
    /// (device-resident) and in-place CPU-KV (`GroupQueryAttention` append)
    /// stores do not expose a host present tensor here, and f16 / non-rank-4
    /// caches would not round-trip losslessly through the f32 paged store — both
    /// are deferred to Inc-D. Every declared KV past input must be a rank-4 f32
    /// cache (`[1, num_kv_heads, seq, head_dim]`).
    pub(crate) fn supports_host_kv_mirror(&self) -> bool {
        if self.cuda.is_some() || self.cpu_kv.is_some() || self.kv_inputs.is_empty() {
            return false;
        }
        self.kv_inputs.iter().all(|name| {
            self.session
                .inputs()
                .iter()
                .find(|meta| &meta.name == name)
                .is_some_and(|meta| meta.dtype == DataType::Float32 && meta.shape.len() == 4)
        })
    }

    /// Whether this session keeps its self-attention KV **device-resident** in
    /// rank-4 CUDA bindings that can be read out for paged present-KV mirroring
    /// and re-seeded from a materialized paged prefix — `f32` (GAP-3 Inc-D) or
    /// `f16` (GAP-3 Inc-D.1).
    ///
    /// This is the device-resident counterpart of
    /// [`supports_host_kv_mirror`](Self::supports_host_kv_mirror): it lifts the
    /// Inc-C gate for a native CUDA decoder whose present-KV lives in device
    /// buffers, landing that KV in the *same* host f32 paged store via the same
    /// `extract_present_token` / `append_token_kv` geometry (a mechanical device
    /// read-out, [`DecodeCudaState::read_present_kv`]). f16 caches are widened to
    /// f32 with the same `half` convert ORT uses (bit-exact round-trip); `bf16`,
    /// non-rank-4, and in-place / CPU-resident caches stay gated to the non-paged
    /// fallback — no silent-wrong paged run.
    pub(crate) fn supports_device_kv_mirror(&self) -> bool {
        if self.kv_inputs.is_empty() {
            return false;
        }
        self.cuda
            .as_ref()
            .is_some_and(DecodeCudaState::kv_bindings_paged_rank4)
    }

    /// The most recent step's accumulated present KV for one self-attention past
    /// input, as a host f32 buffer plus the shape whose row-major strides address
    /// it, or `None` before any step ran / when the past input is unknown.
    ///
    /// Unifies the host-growable path (GAP-3 Inc-C — the growable host cache the
    /// CPU decode path leaves in `self.past`, returned with its compact
    /// `[1, num_kv_heads, total_len, head_dim]` shape) and the device-resident
    /// path (GAP-3 Inc-D — the capacity-padded CUDA binding, returned with its
    /// physical `[1, num_kv_heads, max_len, head_dim]` shape). In both cases the
    /// caller slices out the freshly-decoded tokens with the same
    /// `extract_present_token` geometry the ORT decoder uses, so all three paths
    /// mirror byte-identical pages.
    pub(crate) fn present_kv(
        &mut self,
        past_name: &str,
    ) -> anyhow::Result<Option<(Vec<f32>, Vec<usize>)>> {
        if let Some(cuda) = self.cuda.as_mut() {
            return cuda.read_present_kv(past_name);
        }
        Ok(self.host_present_kv(past_name))
    }

    /// The most recent step's accumulated present KV for one self-attention past
    /// input, as a host f32 buffer plus its `[1, num_kv_heads, total_len,
    /// head_dim]` shape, or `None` before any step ran. Reads the growable host
    /// cache the CPU decode path leaves in `self.past` keyed by the past-input
    /// name; the caller slices out the freshly-decoded tokens with the same
    /// `extract_present_token` geometry the ORT decoder uses.
    pub(crate) fn host_present_kv(&self, past_name: &str) -> Option<(Vec<f32>, Vec<usize>)> {
        self.past
            .get(past_name)
            .map(|tensor| (tensor.to_vec_f32(), tensor.shape.clone()))
    }

    /// Seed the growable host KV cache from a materialized paged prefix so a
    /// later request that shares a prompt prefix resumes without recomputing it.
    ///
    /// `entries` are `(past_input_name, row_major_f32, shape)` triples the caller
    /// built from the paged cache with the same `[1, num_kv_heads, seq, head_dim]`
    /// layout the ORT decoder injects (`kv_bridge::past_shape`), so native and
    /// ORT prefix reuse are byte-identical. Only valid on the host-growable path
    /// (`supports_host_kv_mirror`).
    pub(crate) fn seed_growable_kv(
        &mut self,
        entries: Vec<(String, Vec<f32>, Vec<usize>)>,
        current_len: usize,
    ) -> anyhow::Result<()> {
        if self.cuda.is_some() || self.cpu_kv.is_some() {
            bail!(
                "native paged prefix reuse requires the host-growable KV path; this session keeps \
                 KV device-resident or in-place (Inc-D)"
            );
        }
        for (name, data, shape) in entries {
            let tensor = Tensor::from_f32(&shape, &data)
                .with_context(|| format!("seed native paged prefix KV '{name}'"))?;
            self.past.insert(name, tensor);
        }
        self.current_len = current_len;
        Ok(())
    }

    /// Seed a materialized paged prefix into this session's KV state, dispatching
    /// to the host-growable path (GAP-3 Inc-C) or the device-resident CUDA path
    /// (GAP-3 Inc-D) by how the session keeps its KV. `entries` carry the same
    /// compact `[1, num_kv_heads, seq, head_dim]` layout for both paths, so
    /// native (host or device) and ORT prefix reuse stay byte-identical.
    pub(crate) fn seed_kv(
        &mut self,
        entries: Vec<(String, Vec<f32>, Vec<usize>)>,
        current_len: usize,
    ) -> anyhow::Result<()> {
        if self.cuda.is_some() {
            return self.seed_device_kv(entries, current_len);
        }
        self.seed_growable_kv(entries, current_len)
    }

    /// Device-resident counterpart of [`seed_growable_kv`](Self::seed_growable_kv):
    /// write the shared prefix into the CUDA KV bindings, advance the mask/KV
    /// logical length, and commit `current_len` so the next step appends after it
    /// (GAP-3 Inc-D). The `&mut self.session` / `&mut self.cuda` split borrow lets
    /// the device seed grow the KV bucket if the prefix exceeds the current
    /// capacity, exactly as a decode step would.
    fn seed_device_kv(
        &mut self,
        entries: Vec<(String, Vec<f32>, Vec<usize>)>,
        current_len: usize,
    ) -> anyhow::Result<()> {
        let cuda = self
            .cuda
            .as_mut()
            .context("device paged prefix reuse requires a CUDA decode session")?;
        cuda.seed_prefix(&mut self.session, &entries, current_len)?;
        self.current_len = current_len;
        Ok(())
    }

    /// Build the per-step `position_ids` tensor for the half-open sequence range
    /// `[past_len, total_len)`, honoring the decoder's declared coordinate rank.
    ///
    /// A rank-1 (conventional) decoder yields `[1, S]`; a rank-N mrope decoder
    /// yields `[N, 1, S]` with every coordinate axis advancing linearly with the
    /// sequence position (the pure-text `linear_increment` continuation). The
    /// flat values + shape come from the shared [`crate::decode::position_ids_from_starts`]
    /// helper, so native and ORT build byte-identical positions.
    fn build_step_positions(&self, past_len: usize, total_len: usize) -> anyhow::Result<Tensor> {
        let input_len = total_len
            .checked_sub(past_len)
            .context("position range end precedes its start")?;
        let absolute_start = i64::try_from(past_len).context("position id exceeds i64 range")?;
        let starts = vec![absolute_start; self.position_rank];
        let (data, shape) = crate::decode::position_ids_from_starts(&starts, input_len)?;
        let dims = shape.iter().map(|&dim| dim as usize).collect::<Vec<_>>();
        Ok(Tensor::from_i64(&dims, &data)?)
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

    /// Bytes of fixed-size recurrent state this decoder keeps, and the tier they
    /// live on.
    ///
    /// Zero for a decoder with no recurrent layers, which is most of them, and
    /// not a failure.
    ///
    /// One instance, not one per concurrent sequence. Native decode runs a
    /// single serialized session: other sequences retain tokens and are
    /// re-prefilled rather than each holding a live state tensor, so multiplying
    /// by the scheduler's batch size would reserve up to 32x memory that is
    /// never allocated and refuse models that fit.
    ///
    /// The tier comes from where the state actually is rather than from the
    /// caller. On the CPU backend it is built through `shared_cpu_ep()` and is
    /// host memory; a CUDA session's fixed-state bindings genuinely occupy the
    /// device. Charging the wrong one is the same class of mistake as the KV
    /// pool being charged to `Device` in the pipeline while the engine charged
    /// it to `Host`.
    pub fn recurrent_state_reservation(
        &self,
    ) -> anyhow::Result<(u64, onnx_runtime_memory_governor::Tier)> {
        let bytes = crate::native_decode::tensor::recurrent_state_bytes_per_sequence(
            &self.session,
            &self.present_to_past,
        )?;
        let tier = if self.session.device_id().is_host_accessible() {
            onnx_runtime_memory_governor::Tier::Host
        } else {
            onnx_runtime_memory_governor::Tier::Device
        };
        Ok((bytes, tier))
    }

    /// Place any long-lived device memory this session's provider holds under
    /// `governor`.
    ///
    /// The execution provider is built before the engine's governor exists, so
    /// a provider that keeps a standing pool -- the CUDA weight-residency cache
    /// is the one that does -- sizes it for itself. Until this is called that
    /// size is a second claim on memory the governor is already dividing up, and
    /// neither side can see the other.
    ///
    /// Returns the bytes now governed; zero means the provider holds no standing
    /// pool, which is the common case and not a failure.
    pub fn adopt_memory_governor(
        &self,
        governor: &dyn onnx_runtime_memory_governor::MemoryGovernor,
        tier: onnx_runtime_memory_governor::Tier,
        holder: onnx_runtime_memory_governor::HolderId,
    ) -> anyhow::Result<u64> {
        Ok(self.session.adopt_memory_governor(governor, tier, holder)?)
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
        self.maybe_enable_decode_inline(token_ids);
        if self.cuda.is_some() {
            return self.decode_cuda(token_ids, past_len, step_inputs);
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

    /// Inc-1b PR-2: lazily probe and, if the model is eligible, build the
    /// decode-specialized inlined-body sibling executor at the **first decode
    /// step** (not at load — only models that actually take the hybrid decode
    /// path pay the transform+compile, once). Graph-property gated: the sibling
    /// is built iff the decode graph has a single-trip-eligible recurrent
    /// `Scan`. A probe that finds no such `Scan` latches `Disabled`, so a dense
    /// decoder never builds a sibling, never retries, and stays byte-identical
    /// on the main path.
    ///
    /// A build failure is non-fatal: it latches `Disabled` and decode proceeds
    /// on the byte-identical main (Scan child-session) executor.
    fn maybe_enable_decode_inline(&mut self, token_ids: &[TokenId]) {
        if self.decode_inline != DecodeInlineState::Untried {
            return;
        }
        // Probe at the first genuine decode step (a single new token); a
        // multi-token prefill step is not a decode step and keeps the main plan.
        if token_ids.len() != 1 {
            return;
        }
        self.decode_inline = match self.session.enable_decode_inline() {
            Ok(true) => {
                tracing::debug!("Inc-1b: decode-inline plan enabled (single-trip Scan lowered)");
                DecodeInlineState::Enabled
            }
            Ok(false) => DecodeInlineState::Disabled,
            Err(error) => {
                tracing::warn!(
                    "Inc-1b: decode-inline plan build failed, staying on the Scan child-session \
                     path: {error}"
                );
                DecodeInlineState::Disabled
            }
        };
    }

    /// Whether this decode step should route to the decode-inline sibling exec
    /// (Inc-1b PR-2/PR-3). Guard #2 (runtime scan-axis extent==1 fallback): only a
    /// single-token step is single-trip; any multi-token step falls back to the
    /// main (Scan) executor so a wrongly-collapsed graph is never run.
    ///
    /// Scope-lock (Harry #588 PR-3 rec #4): never route a decoder that declares
    /// `inputs_embeds`/routed ports. Those ports are uploaded per step and are
    /// served by the eager/captured step-input paths, not the token fast path the
    /// decode-inline sibling specializes; excluding them here locks the author's
    /// stated scope by construction on every entry path (both `decode_cuda` and
    /// the `decode_cuda_greedy` fast path), independent of the caller's dispatch.
    fn route_decode_inline(&self, token_ids: &[TokenId]) -> bool {
        route_decode_inline_decision(
            self.decode_inline,
            self.session.decode_inline_ready(),
            token_ids.len(),
            self.has_eager_step_inputs(),
        )
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
