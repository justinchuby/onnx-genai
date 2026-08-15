//! Native nxrt adapter for the engine's existing decode loop.

use crate::config::{GenerateOptions, GenerateResult, GenerateTokenCallback};
use crate::decode::DecodeBackend;
use crate::decode_loop::{DecodeLoopBackend, DecodeLoopState, run_decode_loop};
use crate::logits::{ProcessorChain, TokenId};
use crate::sampling::sample_greedy;
use anyhow::{Context, bail};
use onnx_genai_metadata::{KvOwnership, ModelIoSpec, SequenceInputKind, SharedKvGroup};
use onnx_genai_ort::Tokenizer;
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
mod kv_commit;
mod load;
mod paged_gqa;
mod proposer;
mod tensor;
#[cfg(feature = "cuda")]
pub(crate) use tensor::recurrent_state_bytes_from_graph;
#[cfg(test)]
mod tests;

#[cfg(all(test, feature = "cuda"))]
mod leverb_phase0_probe;

use backend::*;
use cpu::DecodeCpuKvState;
use cpu::*;
use cuda::DecodeCudaState;
use cuda::*;
pub use cuda::{CudaGraphDebugStats, CudaKvDebugStats};

#[cfg(feature = "cuda")]
pub(crate) fn configured_cuda_kv_max_len() -> anyhow::Result<Option<usize>> {
    cuda::cuda_kv_max_len_from_env()
}
use io::*;
pub(crate) use load::NativeDecodeLoadOptions;
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
    pub weight_offload_enabled: Option<bool>,
    /// Whether live weight offload runs on the **stable virtual address** VMM
    /// paging path (issue #716). When `Some(true)`, every retained weight is
    /// served from a reserved-once device VA whose physical granules are
    /// mapped/unmapped underneath, so a captured CUDA graph that baked a weight
    /// pointer stays valid across evict→repage — which is what lets whole-step
    /// graph capture stay ON while offload is active. `Some(false)`/`None` means
    /// the pointer-unstable `alloc_raw`/`free_raw` path, so offload keeps forcing
    /// capture OFF as before.
    pub weight_offload_stable_va: Option<bool>,
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

/// Deep-copy of the native loop-carried tensors at a semantic prefix boundary.
///
/// Host-native decoding keeps dense KV and recurrent state in the same `past`
/// map, so the production snapshot stores the whole map to make restoring a
/// boundary actually executable. The recurrent entries are still identified
/// from the declared present→past pairs for sizing and gating.
pub(crate) struct NativePastSnapshot {
    past: HashMap<String, Tensor>,
    len: usize,
    bytes: u64,
}

impl NativePastSnapshot {
    pub(crate) fn bytes(&self) -> u64 {
        self.bytes
    }
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
    pub(crate) fn inference_session(&self) -> &InferenceSession {
        &self.session
    }

    pub fn current_len(&self) -> usize {
        self.current_len
    }

    /// Whether this session has a CUDA decode state (captured verify is only
    /// available — and only worth arming — on the CUDA execution provider).
    pub fn is_cuda(&self) -> bool {
        self.cuda.is_some()
    }

    /// Force the retained captured-verify graph to re-warm on its next call.
    /// No-op on non-CUDA sessions or before the first captured verify.
    pub fn reset_captured_verify(&mut self) {
        if let Some(state) = self.cuda.as_mut() {
            state.reset_verify_capture_phase();
        }
    }

    pub fn kv_layer_count(&self) -> usize {
        self.kv_inputs.len() / 2
    }

    /// Stage 2a (#750): run **one fused forward** over `token_ids.len()`
    /// independent single-token rows on the batch axis, with an **empty past**,
    /// returning one `[vocab]` logits row per batch row.
    ///
    /// This is the batch-N *input binding + fused forward* validated in isolation
    /// from the batched KV *layout* (stage 2b). Every row is a fresh length-1
    /// sequence at position 0: `input_ids [N,1]`, `attention_mask [N,1]`,
    /// `position_ids [N,1]` (or `[rank,N,1]` for multi-axis mrope) and empty
    /// `past_key_values.* [N, heads, 0, head_dim]`. Because the past is empty
    /// there is no KV row addressing to change, and the present-KV outputs are
    /// materialized and discarded. The rows are genuinely independent (no
    /// cross-row attention), so row `i` is byte-identical to a batch-1 forward of
    /// `token_ids[i]` — the guard `native_fused_batch_prefill_row_identical`
    /// asserts exactly this.
    ///
    /// The point being measured is **weight-streaming amortization**: the CUDA
    /// weight-offload residency is keyed purely by weight identity with no batch
    /// dimension, so this single forward pages each weight in at most once and
    /// emits `N` rows — `htod_bytes` is (near-)batch-invariant, so
    /// `htod_bytes / N` falls ~`1/N`.
    ///
    /// This is a stateless probe: it runs a plain eager `session.run` with owned
    /// inputs and does **not** touch the persistent decode state (`self.past`,
    /// `self.current_len`, the CUDA persistent bindings), so it never engages
    /// CUDA-graph capture (an eager forward is not captured) and leaves an
    /// in-progress decode untouched. Decoders with `inputs_embeds`/routed
    /// per-step ports are rejected — the probe only binds token ids.
    pub fn run_fused_batch_prefill(
        &mut self,
        token_ids: &[TokenId],
    ) -> anyhow::Result<Vec<Vec<f32>>> {
        self.run_fused_batch_forward(token_ids, 0)
    }

    /// Stage 2b (#750): the stage 2a fused forward generalized to a **non-empty
    /// batched KV past** of `past_len` positions per row.
    ///
    /// Every row is a single new token at position `past_len` attending over
    /// `past_len` zero-seeded past-KV positions: `input_ids [N,1]`,
    /// `attention_mask [N, past_len + 1]` (all valid), `position_ids [N,1] =
    /// past_len` (or `[rank,N,1]` for mrope) and `past_key_values.* [N, heads,
    /// past_len, head_dim]`. Unlike stage 2a's empty past, this drives the ONNX
    /// attention **batch coupling across QKV / mask / past-KV** at `N > 1` with
    /// real past content — the coupling `persistent_state_shapes` pins to batch 1
    /// in the persistent decode path — and commits `N × past_len` of KV so the
    /// weight-residency reclaim under the elastic budget (#866) is observable as
    /// `htod_bytes_per_token` rising with `N` and `past_len`.
    ///
    /// Still a **stateless** probe: it never touches `self.past`,
    /// `self.current_len`, or the CUDA persistent bindings, so `cuda.rs` KV
    /// governance is untouched and CUDA-graph capture is not engaged (`past_len =
    /// 0` is byte-identical to [`run_fused_batch_prefill`]). Row `i` is
    /// byte-identical to a batch-1 forward of `token_ids[i]` at the same
    /// `past_len` (the rows are independent; zero past is identical per row).
    pub fn run_fused_batch_forward(
        &mut self,
        token_ids: &[TokenId],
        past_len: usize,
    ) -> anyhow::Result<Vec<Vec<f32>>> {
        if token_ids.is_empty() {
            bail!("run_fused_batch_forward requires at least one token");
        }
        let batch = token_ids.len();
        if self.has_eager_step_inputs() {
            bail!(
                "run_fused_batch_forward supports token-id decoders only; this decoder declares inputs_embeds/routed per-step ports"
            );
        }
        let token_input = self
            .step_input_name(NativeStepInputSource::TokenIds)
            .context("native decoder has no token input binding")?
            .to_owned();
        let mask_input = self
            .step_input_name(NativeStepInputSource::AttentionMask)
            .map(str::to_owned);
        let position_input = self
            .step_input_name(NativeStepInputSource::PositionIds)
            .map(str::to_owned);

        let ids = token_ids
            .iter()
            .map(|&id| i64::from(id))
            .collect::<Vec<_>>();
        let mut owned: Vec<(String, Tensor)> = Vec::with_capacity(3 + self.kv_inputs.len());
        // input_ids: [N, 1] — one new token per row.
        owned.push((token_input, Tensor::from_i64(&[batch, 1], &ids)?));
        // attention_mask: [N, past_len + 1] — every past position and the new
        // token are valid (uniform length, no ragged admission).
        if let Some(mask) = mask_input {
            let mask_len = past_len + 1;
            owned.push((
                mask,
                Tensor::from_i64(&[batch, mask_len], &vec![1i64; batch * mask_len])?,
            ));
        }
        // position_ids: every row's new token is at position `past_len`. Rank 1
        // -> [N, 1]; a multi-axis mrope decoder (rank > 1) -> [rank, N, 1].
        if let Some(position) = position_input {
            let pos = i64::try_from(past_len).context("past_len exceeds i64 range")?;
            let tensor = if self.position_rank <= 1 {
                Tensor::from_i64(&[batch, 1], &vec![pos; batch])?
            } else {
                Tensor::from_i64(
                    &[self.position_rank, batch, 1],
                    &vec![pos; self.position_rank * batch],
                )?
            };
            owned.push((position, tensor));
        }
        // Length-`past_len` batched past for every KV / recurrent-state input,
        // batch axis = N.
        for name in &self.kv_inputs {
            let tensor = make_past_input_tensor_batched(&self.session, name, batch, past_len)?;
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
                bail!("native fused batch forward failed{diagnosis}: {error}");
            }
        };
        let logits = self
            .session
            .outputs()
            .iter()
            .zip(outputs)
            .find(|(meta, _)| meta.name == self.logits)
            .map(|(_, tensor)| tensor)
            .with_context(|| format!("native decoder omitted logits output '{}'", self.logits))?;
        let rows = extract_batch_row_logits(&logits, batch)?;
        if rows.iter().flatten().any(|value| !value.is_finite()) {
            bail!("native fused batch forward produced non-finite logits");
        }
        Ok(rows)
    }

    pub(crate) fn supports_past_snapshots(&self) -> bool {
        self.cuda.is_none() && self.cpu_kv.is_none() && self.has_recurrent_state()
    }

    fn recurrent_past_names(&self) -> HashSet<String> {
        let declared = self
            .present_to_past
            .values()
            .map(String::as_str)
            .collect::<HashSet<_>>();
        self.session
            .inputs()
            .iter()
            .filter(|meta| declared.contains(meta.name.as_str()))
            .filter(|meta| is_recurrent_state_shape(&meta.shape))
            .map(|meta| meta.name.clone())
            .collect()
    }

    pub(crate) fn has_recurrent_state(&self) -> bool {
        !self.recurrent_past_names().is_empty()
    }

    pub(crate) fn snapshot_past(&self) -> anyhow::Result<NativePastSnapshot> {
        if !self.supports_past_snapshots() {
            bail!("native past snapshots require host past tensors and recurrent state");
        }
        let recurrent = self.recurrent_past_names();
        if !recurrent.iter().all(|name| self.past.contains_key(name)) {
            bail!("cannot snapshot recurrent prefix before all recurrent tensors are materialized");
        }
        let mut bytes = 0u64;
        let mut past = HashMap::with_capacity(self.past.len());
        for (name, tensor) in &self.past {
            bytes =
                bytes.saturating_add(u64::try_from(tensor.as_bytes().len()).unwrap_or(u64::MAX));
            past.insert(
                name.clone(),
                tensor.try_clone().map_err(anyhow::Error::from)?,
            );
        }
        Ok(NativePastSnapshot {
            past,
            len: self.current_len,
            bytes,
        })
    }

    pub(crate) fn restore_past_snapshot(
        &mut self,
        snapshot: &NativePastSnapshot,
    ) -> anyhow::Result<()> {
        if !self.supports_past_snapshots() {
            bail!("native past snapshots require host past tensors and recurrent state");
        }
        let mut restored = HashMap::with_capacity(snapshot.past.len());
        for (name, tensor) in &snapshot.past {
            restored.insert(
                name.clone(),
                tensor.try_clone().map_err(anyhow::Error::from)?,
            );
        }
        self.past = restored;
        self.current_len = snapshot.len;
        self.last_hidden = None;
        Ok(())
    }

    pub(crate) fn prefill_prefix(&mut self, tokens: &[TokenId]) -> anyhow::Result<()> {
        if tokens.is_empty() {
            return Ok(());
        }
        let past_len = self.current_len;
        self.decode_argmax(tokens, past_len)?
            .context("native prefix prefill produced no argmax token")?;
        Ok(())
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
    ///
    /// Hybrid recurrent decoders (conv/recurrent `fixed_state` alongside
    /// attention KV) are excluded: prefix mirroring restores only attention KV,
    /// but their unmasked recurrent/conv state is not reconstructed from a reused
    /// prefix, so a mirrored continuation would run a fresh-zero recurrent state
    /// against a reused attention prefix and silently emit wrong logits (#695).
    /// Gating the mirror off forces a full recompute for these models — correct,
    /// if slower — until per-prefix recurrent-state restore lands.
    pub(crate) fn supports_host_kv_mirror(&self) -> bool {
        if self.cuda.is_some()
            || self.cpu_kv.is_some()
            || self.kv_inputs.is_empty()
            || self.has_recurrent_state()
        {
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
    ///
    /// Hybrid recurrent decoders (conv/recurrent `fixed_state` alongside
    /// attention KV) are excluded for the same reason as the host path
    /// ([`supports_host_kv_mirror`](Self::supports_host_kv_mirror)): device
    /// prefix reuse (`DecodeCudaState::seed_prefix`) restores only attention KV,
    /// while the recurrent/conv state is reconstructed only on a full
    /// `rewind(0)`, so a mirrored continuation runs a fresh-zero recurrent state
    /// against a reused attention prefix and silently emits wrong logits (#695).
    pub(crate) fn supports_device_kv_mirror(&self) -> bool {
        if self.kv_inputs.is_empty() || self.has_recurrent_state() {
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

    /// Named decode-level predicate that declined CUDA graph capture, either
    /// before the first attempt or during the runtime capture audit.
    pub fn cuda_graph_decline_reason(&self) -> Option<&str> {
        self.cuda
            .as_ref()
            .and_then(|state| state.graph_decline_reason.as_deref())
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
    /// How many recurrent-state instances physically exist at once.
    ///
    /// One: native decode runs a single serialized session, and other sequences
    /// retain tokens and are re-prefilled rather than each holding a live state
    /// tensor.
    ///
    /// Explicit rather than a bare `1` in the arithmetic, because the number
    /// that belongs here is *how many rows exist*, not *how many sequences the
    /// scheduler admits*. Reserving `max_batch_size` of them over-counts by up
    /// to 32x against memory that is never allocated, which is what the first
    /// version of this did. If native decode later keeps several rows live,
    /// this is the one place to change, and the caller's arithmetic stays right.
    pub fn concurrent_state_rows(&self) -> usize {
        1
    }

    pub fn recurrent_state_reservation(
        &self,
    ) -> anyhow::Result<(u64, onnx_runtime_memory_governor::Tier)> {
        let per_row = crate::native_decode::tensor::recurrent_state_bytes_per_sequence(
            &self.session,
            &self.present_to_past,
        )?;
        let bytes = per_row.saturating_mul(self.concurrent_state_rows() as u64);
        let tier = if self.session.device_id().is_host_accessible() {
            onnx_runtime_memory_governor::Tier::Host
        } else {
            onnx_runtime_memory_governor::Tier::Device
        };
        Ok((bytes, tier))
    }

    /// Whether this session's execution provider commits device memory as it
    /// is used rather than when it is requested.
    ///
    /// Decides whether a worst-case figure -- KV at the model's full context,
    /// for instance -- has to be *held* or merely *checked*.
    pub fn commits_on_demand(&self) -> bool {
        self.session.commits_on_demand()
    }

    /// Bytes of KV this session's past/present tensors hold at full context,
    /// with the tier they are actually allocated on.
    ///
    /// The native path's page table is bookkeeping only, so unlike the ONNX
    /// Runtime path there is no pool whose construction leases the KV. Without
    /// this the ledger's largest per-sequence cost is simply missing, and every
    /// consumer that reads a tier total -- admission, the profile breakdown, a
    /// third-party governor deciding whether to grant -- works from a number
    /// that is low by an unknown amount.
    pub fn kv_reservation(
        &self,
        max_context: usize,
    ) -> anyhow::Result<(u64, onnx_runtime_memory_governor::Tier)> {
        let bytes = crate::native_decode::tensor::kv_cache_bytes_per_sequence(
            &self.session,
            &self.present_to_past,
            max_context,
        )?;
        // Same reasoning as `recurrent_state_reservation`: where these live is a
        // fact about the running system, not about the holder. A CPU EP session
        // holds them in host memory.
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

    /// Resize a provider-owned weight-residency budget before governor adoption.
    ///
    /// The native CUDA loader initially knows only the explicit VRAM limit. Once
    /// the session has loaded, it can size KV/recurrent state and leave those
    /// bytes out of the weight budget instead of reproducing #712 by letting
    /// weights consume the whole ceiling.
    pub fn set_weight_residency_budget(&self, budget_bytes: u64) -> anyhow::Result<Option<u64>> {
        Ok(self.session.set_weight_residency_budget(budget_bytes)?)
    }

    /// Largest set of lazy weights one native executor node may need resident
    /// at the same time.
    pub fn max_lazy_weight_working_set_bytes(&self) -> u64 {
        self.session.max_lazy_weight_working_set_bytes()
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

    /// WP4 speculative driver switch: retain the captured M=width verify graph
    /// across the per-step `rewind` (option (c) contents-only mutation) instead
    /// of invalidating it (the eager default). Only the captured-verify
    /// speculative path (and correctness tests) flip this on; a no-op on non-CUDA
    /// sessions.
    pub fn set_retain_graph_on_rewind(&mut self, retain: bool) {
        if let Some(state) = self.cuda.as_mut() {
            state.retain_graph_on_rewind = retain;
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

    /// Batch-N greedy decode step (stage 2b-impl-4, #750). Steps the pinned
    /// `batch` sequences together — one token per sequence — and returns the
    /// `batch` selected token ids. Requires a CUDA decode session pinned at the
    /// same batch extent (via `ONNX_GENAI_NATIVE_DECODE_BATCH`) whose logits
    /// binding supports the device-argmax fast path; otherwise it errors. This is
    /// the reachable batch-N exercise seam (the single-sequence generation driver
    /// is unchanged and stays the batch-1 byte-identity reference); it is used by
    /// the batch measurement harness in `profile_native`.
    pub fn decode_greedy_batch(
        &mut self,
        token_ids: &[TokenId],
        past_len: usize,
    ) -> anyhow::Result<Vec<TokenId>> {
        if past_len != self.current_len {
            bail!(
                "native batch decode past length mismatch: caller supplied {past_len}, adapter holds {}",
                self.current_len
            );
        }
        if self.cuda.is_none() {
            bail!("native batch greedy decode requires a CUDA decode session");
        }
        self.decode_cuda_greedy_batch(token_ids, past_len)
    }

    /// Ragged batch-N greedy decode step (stage 3a, #750). Steps the pinned
    /// `batch` sequences together where row `r` sits at its own logical length
    /// `past_lens[r]` and advances only if `advances[r]`. Returns the `batch`
    /// selected token ids. This is the ragged generalization of
    /// [`Self::decode_greedy_batch`]: per-row attention-mask window, per-row
    /// `position_ids`, and a per-row logical length, so genuinely
    /// different-length requests can share one fused forward (the geometry a
    /// continuous batcher needs). Like the uniform entry point it requires a CUDA
    /// decode session pinned at the same batch extent whose logits binding
    /// supports the device-argmax fast path. The single-sequence generation
    /// driver is unchanged and stays the batch-1 byte-identity reference; this is
    /// exercised by `profile_native --ragged-solo-equivalence-prompts`.
    pub fn decode_greedy_batch_ragged(
        &mut self,
        token_ids: &[TokenId],
        past_lens: &[usize],
        advances: &[bool],
    ) -> anyhow::Result<Vec<TokenId>> {
        if self.cuda.is_none() {
            bail!("native ragged batch greedy decode requires a CUDA decode session");
        }
        self.decode_cuda_greedy_batch_ragged(token_ids, past_lens, advances)
    }

    /// The pinned persistent-decode batch extent (1 unless
    /// `ONNX_GENAI_NATIVE_DECODE_BATCH` requested batch-N and a CUDA session was
    /// built). Lets a harness confirm the session actually bound the batch grid.
    pub fn native_decode_batch(&self) -> usize {
        self.cuda.as_ref().map(DecodeCudaState::batch).unwrap_or(1)
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

    pub(crate) fn prepare_generation_workspace_for_query_rows(
        &mut self,
        prompt_tokens: &[TokenId],
        query_rows: usize,
    ) -> anyhow::Result<onnx_runtime_session::WorkspaceRequirement> {
        self.prepare_generation_workspace_inner(prompt_tokens, query_rows, true)
    }

    pub(crate) fn prepare_generation_workspace_preserving_state(
        &mut self,
        prompt_tokens: &[TokenId],
    ) -> anyhow::Result<onnx_runtime_session::WorkspaceRequirement> {
        self.prepare_generation_workspace_inner(prompt_tokens, prompt_tokens.len(), false)
    }

    pub(crate) fn prepare_generation_workspace_with_step_inputs(
        &mut self,
        tokens: &[TokenId],
        past_len: usize,
        step_inputs: &[(String, Tensor)],
    ) -> anyhow::Result<onnx_runtime_session::WorkspaceRequirement> {
        if tokens.is_empty() {
            bail!("native workspace preparation requires at least one input token");
        }
        if self.cuda.is_some() {
            self.prepare_cuda_prefill_workspace_with_step_inputs(tokens, past_len, step_inputs)
        } else {
            Ok(onnx_runtime_session::WorkspaceRequirement::NONE)
        }
    }

    fn prepare_generation_workspace_inner(
        &mut self,
        prompt_tokens: &[TokenId],
        query_rows: usize,
        reset: bool,
    ) -> anyhow::Result<onnx_runtime_session::WorkspaceRequirement> {
        if prompt_tokens.is_empty() {
            bail!("native workspace preparation requires at least one prompt token");
        }
        if reset {
            self.reset()?;
        }
        if self.cuda.is_some() {
            if query_rows <= prompt_tokens.len() {
                self.prepare_cuda_prefill_workspace(prompt_tokens)
            } else {
                let mut planning_tokens = Vec::with_capacity(query_rows);
                planning_tokens.extend_from_slice(prompt_tokens);
                planning_tokens.resize(query_rows, prompt_tokens[0]);
                self.prepare_cuda_prefill_workspace(&planning_tokens)
            }
        } else {
            Ok(onnx_runtime_session::WorkspaceRequirement::NONE)
        }
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
