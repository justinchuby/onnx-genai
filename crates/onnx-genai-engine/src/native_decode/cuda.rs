use super::*;

use super::kv_commit::{self, KvBindingGeometry, KvCommitLayout};

/// Resolve the physical KV-cache layout for the native CUDA binding layer.
///
/// The authoritative source is the model's [`ModelIoSpec::kv_layout`] descriptor
/// (a seq-major descriptor selects [`KvCommitLayout::SeqMajor`]). Absent metadata
/// means head-major, preserving the historical behavior exactly. The
/// `ONNX_GENAI_CUDA_KV_LAYOUT` environment variable (`seq_major` / `bsnh` vs
/// `head_major` / `bnsh`) overrides the descriptor for residency diagnostics; it
/// never changes which kernels run, only how the binding layer accounts for and
/// commits residency.
pub(crate) fn resolve_cuda_kv_layout(
    metadata: Option<&onnx_genai_metadata::KvCacheLayout>,
) -> KvCommitLayout {
    if let Ok(value) = std::env::var("ONNX_GENAI_CUDA_KV_LAYOUT") {
        match value.trim().to_ascii_lowercase().as_str() {
            "seq_major" | "seq-major" | "bsnh" => return KvCommitLayout::SeqMajor,
            "head_major" | "head-major" | "bnsh" => return KvCommitLayout::HeadMajor,
            _ => {}
        }
    }
    match metadata.and_then(|layout| layout.gqa_attribute_value()) {
        Some(1) => KvCommitLayout::SeqMajor,
        _ => KvCommitLayout::HeadMajor,
    }
}

/// Purely structural signals that gate whether whole-step CUDA graph capture is
/// *auto-attempted* on the native decode path. Never derived from a model or
/// architecture name (RULES.md §2/§2.1) — only from device placement and the
/// declared KV-ownership metadata. When these hold, per-step decode topology is
/// static and the KV cache is device-resident and owned, so a captured graph can
/// replay safely. The runtime decline machinery in `DecodeCudaState::new`
/// remains the final safety net: if a would-be capture still carries a dynamic
/// auxiliary seam it is transparently declined and decode continues eagerly.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct GraphCaptureStructuralSafety {
    /// Decode runs on a CUDA device (device-resident, replayable bindings).
    pub(crate) device_is_cuda: bool,
    /// KV cache is owned/device-resident (not a borrowed shared-KV proposer).
    pub(crate) kv_ownership: KvOwnership,
}

impl GraphCaptureStructuralSafety {
    /// True when structural conditions make whole-step capture safe to attempt.
    #[cfg(test)]
    pub(crate) fn is_capture_safe(self) -> bool {
        self.device_is_cuda && self.kv_ownership == KvOwnership::Owned
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct GraphCaptureDecision {
    enabled: bool,
    decline_reason: Option<String>,
}

impl GraphCaptureDecision {
    fn enabled() -> Self {
        Self {
            enabled: true,
            decline_reason: None,
        }
    }

    fn declined(predicate: &str, detail: impl std::fmt::Display) -> Self {
        Self {
            enabled: false,
            decline_reason: Some(format!(
                "predicate `{predicate}` declined capture: {detail}"
            )),
        }
    }

    pub(crate) fn is_enabled(&self) -> bool {
        self.enabled
    }

    #[cfg(test)]
    pub(crate) fn decline_reason(&self) -> Option<&str> {
        self.decline_reason.as_deref()
    }

    fn decline_if_enabled(&mut self, predicate: &str, detail: impl std::fmt::Display) {
        if self.enabled {
            *self = Self::declined(predicate, detail);
        }
    }

    fn into_decline_reason(self) -> Option<String> {
        self.decline_reason
    }
}

/// Resolve whether whole-step CUDA graph capture should be attempted for the
/// native decode path, honoring explicit overrides before the structural
/// auto-decision.
///
/// Precedence:
/// 0. Live weight offload (`ONNX_GENAI_WEIGHT_OFFLOAD=1`) forces capture OFF
///    **only when its paging path is not stable-VA**. The legacy pager's
///    alloc/copy/free ops return a different device pointer per page-in, and a
///    captured graph bakes pointers into its recorded nodes, so the two are
///    mutually exclusive on that path (the decision records the named predicate).
///    When offload runs on the stable virtual-address VMM path (issue #716),
///    every retained weight keeps a reserved-once device VA whose physical
///    granules are remapped underneath, which is capture-safe, so this no longer
///    forces capture OFF and the normal precedence below applies.
/// 1. Programmatic `NativeDecodeCudaOptions::graph_capture` (`Some`) wins next.
/// 2. An explicitly-set `ONNX_GENAI_CUDA_GRAPH` env var (`=0` forces OFF, `=1`
///    forces ON) is honored next.
/// 3. When neither is set, auto-decide from `structural` safety: attempt capture
///    only when the decode topology is structurally graph-safe.
pub(crate) fn resolve_graph_capture_decision(
    programmatic: Option<bool>,
    env_explicit: bool,
    env_value: bool,
    structural: GraphCaptureStructuralSafety,
    weight_offload_enabled: bool,
    weight_offload_stable_va: bool,
) -> GraphCaptureDecision {
    if weight_offload_enabled && !weight_offload_stable_va {
        return GraphCaptureDecision::declined(
            "weight_offload_enabled && !weight_offload_stable_va",
            "weight offload is using the pointer-unstable paging path",
        );
    }
    if let Some(explicit) = programmatic {
        return if explicit {
            GraphCaptureDecision::enabled()
        } else {
            GraphCaptureDecision::declined(
                "NativeDecodeCudaOptions::graph_capture",
                "the programmatic override is Some(false)",
            )
        };
    }
    if env_explicit {
        return if env_value {
            GraphCaptureDecision::enabled()
        } else {
            GraphCaptureDecision::declined(
                "ONNX_GENAI_CUDA_GRAPH",
                "the process-wide runtime configuration captured an explicit value of 0 on first use",
            )
        };
    }
    if !structural.device_is_cuda {
        return GraphCaptureDecision::declined(
            "GraphCaptureStructuralSafety::device_is_cuda",
            "the decode device is not CUDA",
        );
    }
    if structural.kv_ownership != KvOwnership::Owned {
        return GraphCaptureDecision::declined(
            "GraphCaptureStructuralSafety::kv_ownership",
            format_args!(
                "KV ownership is {:?}, but capture requires Owned",
                structural.kv_ownership
            ),
        );
    }
    GraphCaptureDecision::enabled()
}

#[cfg(test)]
pub(crate) fn resolve_graph_capture_enabled(
    programmatic: Option<bool>,
    env_explicit: bool,
    env_value: bool,
    structural: GraphCaptureStructuralSafety,
    weight_offload_enabled: bool,
    weight_offload_stable_va: bool,
) -> bool {
    resolve_graph_capture_decision(
        programmatic,
        env_explicit,
        env_value,
        structural,
        weight_offload_enabled,
        weight_offload_stable_va,
    )
    .is_enabled()
}

fn cuda_step_profile_enabled() -> bool {
    std::env::var_os("ONNX_GENAI_PROFILE_CUDA_DECODE_STEPS").is_some_and(|value| {
        let value = value.to_string_lossy();
        matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        )
    })
}

#[derive(Clone, Copy, Debug, Default)]
struct StepOffloadSnapshot {
    materialize_ns: u64,
    htod_ns: u64,
    admit_sync_ns: u64,
    page_ins: u64,
    staging_fill_bytes: u64,
    staging_fill_regions: u64,
    staging_fill_calls: u64,
    materialize_fallback_calls: u64,
    htod_bytes: u64,
    vram_alloc_ns: u64,
    vram_free_ns: u64,
}

impl StepOffloadSnapshot {
    fn read() -> Self {
        #[cfg(feature = "cuda")]
        {
            let stats = onnx_runtime_ep_cuda::global_offload_stats();
            Self {
                materialize_ns: stats.materialize_ns,
                htod_ns: stats.htod_ns,
                admit_sync_ns: stats.admit_sync_ns,
                page_ins: stats.page_ins,
                staging_fill_bytes: stats.staging_fill_bytes,
                staging_fill_regions: stats.staging_fill_regions,
                staging_fill_calls: stats.staging_fill_calls,
                materialize_fallback_calls: stats.materialize_fallback_calls,
                htod_bytes: stats.htod_bytes,
                vram_alloc_ns: stats.vram_alloc_ns,
                vram_free_ns: stats.vram_free_ns,
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            Self::default()
        }
    }

    fn delta(self, before: Self) -> Self {
        Self {
            materialize_ns: self.materialize_ns.saturating_sub(before.materialize_ns),
            htod_ns: self.htod_ns.saturating_sub(before.htod_ns),
            admit_sync_ns: self.admit_sync_ns.saturating_sub(before.admit_sync_ns),
            page_ins: self.page_ins.saturating_sub(before.page_ins),
            staging_fill_bytes: self
                .staging_fill_bytes
                .saturating_sub(before.staging_fill_bytes),
            staging_fill_regions: self
                .staging_fill_regions
                .saturating_sub(before.staging_fill_regions),
            staging_fill_calls: self
                .staging_fill_calls
                .saturating_sub(before.staging_fill_calls),
            materialize_fallback_calls: self
                .materialize_fallback_calls
                .saturating_sub(before.materialize_fallback_calls),
            htod_bytes: self.htod_bytes.saturating_sub(before.htod_bytes),
            vram_alloc_ns: self.vram_alloc_ns.saturating_sub(before.vram_alloc_ns),
            vram_free_ns: self.vram_free_ns.saturating_sub(before.vram_free_ns),
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct CudaStepWallBreakdown {
    run_ms: f64,
    logits_read_ms: f64,
    capture_check_ms: f64,
    finite_check_ms: f64,
}

struct CudaStepProfile {
    past_len: usize,
    total_len: usize,
    before: StepOffloadSnapshot,
    start: std::time::Instant,
}

impl CudaStepProfile {
    fn begin(past_len: usize, total_len: usize) -> Option<Self> {
        if !cuda_step_profile_enabled() {
            return None;
        }
        onnx_runtime_session::enable_exec_phase_profile_for_process();
        onnx_runtime_session::reset_exec_phase_profile();
        Some(Self {
            past_len,
            total_len,
            before: StepOffloadSnapshot::read(),
            start: std::time::Instant::now(),
        })
    }

    fn finish(self, path: &'static str, wall: CudaStepWallBreakdown) {
        let total_ms = self.start.elapsed().as_secs_f64() * 1_000.0;
        let delta = StepOffloadSnapshot::read().delta(self.before);
        let staging_ms = ns_to_ms(delta.materialize_ns);
        let h2d_ms = ns_to_ms(delta.htod_ns);
        let admit_sync_ms = ns_to_ms(delta.admit_sync_ns);
        let vram_alloc_ms = ns_to_ms(delta.vram_alloc_ns);
        let vram_free_ms = ns_to_ms(delta.vram_free_ns);
        let phase_stats = onnx_runtime_session::exec_phase_stats();
        let phase_ms = |phase: &str| -> f64 {
            phase_stats
                .iter()
                .find_map(|(name, total_ns, _)| (*name == phase).then_some(*total_ns))
                .map(|ns| ns as f64 / 1_000_000.0)
                .unwrap_or(0.0)
        };
        let kernel_host_ms = phase_ms("exec_kernel.compute");
        let build_inputs_ms = phase_ms("exec_kernel.build_inputs");
        let build_inputs_attributed_ms =
            staging_ms + h2d_ms + admit_sync_ms + vram_alloc_ms + vram_free_ms;
        let build_inputs_unattributed_ms = (build_inputs_ms - build_inputs_attributed_ms).max(0.0);
        let executor_other_ms = wall.run_ms - build_inputs_ms - kernel_host_ms;
        let run_unattributed_ms = build_inputs_unattributed_ms + executor_other_ms;
        let residual_ms = total_ms
            - staging_ms
            - h2d_ms
            - admit_sync_ms
            - vram_alloc_ms
            - vram_free_ms
            - kernel_host_ms
            - build_inputs_unattributed_ms
            - executor_other_ms
            - wall.logits_read_ms
            - wall.capture_check_ms
            - wall.finite_check_ms;
        static HEADER: std::sync::Once = std::sync::Once::new();
        HEADER.call_once(|| {
            eprintln!(
                "[onnx-genai-cuda-step] path,past_len,total_len,total_ms,staging_fill_ms,h2d_copy_ms,kernel_host_dispatch_ms,admit_sync_ms,vram_alloc_ms,vram_free_ms,build_inputs_unattributed_ms,executor_other_ms,run_unattributed_ms,logits_read_sync_ms,capture_check_ms,finite_check_ms,residual_ms,page_ins,staging_fill_bytes,staging_fill_regions,staging_fill_calls,materialize_fallback_calls,h2d_bytes"
            );
        });
        eprintln!(
            "[onnx-genai-cuda-step] {path},{},{},{total_ms:.3},{staging_ms:.3},{h2d_ms:.3},{kernel_host_ms:.3},{admit_sync_ms:.3},{vram_alloc_ms:.3},{vram_free_ms:.3},{build_inputs_unattributed_ms:.3},{executor_other_ms:.3},{run_unattributed_ms:.3},{:.3},{:.3},{:.3},{residual_ms:.3},{},{},{},{},{},{}",
            self.past_len,
            self.total_len,
            wall.logits_read_ms,
            wall.capture_check_ms,
            wall.finite_check_ms,
            delta.page_ins,
            delta.staging_fill_bytes,
            delta.staging_fill_regions,
            delta.staging_fill_calls,
            delta.materialize_fallback_calls,
            delta.htod_bytes
        );
    }
}

fn ns_to_ms(ns: u64) -> f64 {
    ns as f64 / 1_000_000.0
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CudaKvDebugStats {
    pub logical_len: usize,
    pub max_len: usize,
    pub kv_committed_len: usize,
    pub hard_max_len: usize,
    pub max_len_source: String,
    pub device_ptrs: Vec<usize>,
    pub kv_committed_bytes: usize,
    pub kv_physical_bytes_by_binding: Vec<usize>,
    pub kv_transfers: DeviceBindingTransferStats,
    pub kv_growth_events: u64,
    pub kv_growth_d2d_copy_bytes: u64,
    /// `true` when the KV bindings are stored seq-major (BSNH) and the binding
    /// layer accounts residency by the dense live-prefix rule; `false` for the
    /// default head-major (BNSH) flat-bucket accounting. Surfaced so the
    /// committed-bytes measurement can attribute a residency number to the
    /// layout that produced it.
    pub kv_layout_seq_major: bool,
    pub graph: CudaGraphDebugStats,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct CudaDeviceMemorySnapshot {
    pub(crate) free_bytes: usize,
    pub(crate) total_bytes: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct CudaKvCapacity {
    pub(crate) max_len: usize,
    pub(crate) source: String,
    pub(crate) metadata_max_len: Option<usize>,
    pub(crate) device_memory: Option<CudaDeviceMemorySnapshot>,
    pub(crate) bytes_per_token: usize,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct CudaGraphDebugStats {
    pub enabled: bool,
    pub captures: u64,
    pub replays: u64,
    pub fallbacks: u64,
    pub invalidations: u64,
    /// Number of KV-cache growth events that **kept** the captured graph instead
    /// of invalidating it, because the growth provably changed none of the
    /// captured graph's baked dependencies (seq-major fixed full-context stride:
    /// stable device pointers, unchanged physical shapes, capacity-independent
    /// addressing). Head-major and the legacy realloc path never keep — they are
    /// counted under `invalidations`. Surfaced so an operator can attribute why a
    /// graph survived a growth (#804-style named reporting).
    pub growth_keeps: u64,
    pub allocation_counts: DeviceAllocationCounts,
    /// Named decode-level predicate that declined capture, whether before the
    /// first attempt or during the runtime capture audit.
    pub decline_reason: Option<String>,
    /// The most recent KV-growth capture decision (kept vs invalidated) with the
    /// named predicate that produced it, so an operator can see *why* a graph was
    /// kept or invalidated across a growth — a silent "kept" is as dangerous as a
    /// silent capture decline.
    pub growth_decision: Option<String>,
    /// Structured reasons from the most recent capture fallback.
    pub fallback_report: Option<CaptureDeclineReport>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DecodeCudaGraphPhase {
    NeedsWarmup,
    Armed,
    Ready,
    Unsupported,
}

pub(crate) struct DecodeCudaState {
    logical_len: usize,
    /// Current physical KV bucket for the native CUDA decode session. The hard
    /// maximum lives in `capacity.max_len`; this bucket grows on demand via the
    /// shared `onnx_genai_kv::kv_capacity_bucket` policy. Growth is a capture
    /// boundary: reallocation invalidates the old graph, then the normal
    /// warm-up/capture/replay state machine captures the new bucket.
    pub(crate) max_len: usize,
    pub(crate) bindings: Vec<DeviceIoBinding>,
    pub(crate) base_binding_count: usize,
    pub(crate) kv_binding_range: std::ops::Range<usize>,
    /// Bindings for fixed-size recurrent/conv states (hybrid linear-attention
    /// `conv_state`/`recurrent_state`). Unlike growable KV — which is masked and
    /// tracked by `logical_len`, so stale slots are inert — these are wholesale
    /// rolling caches with no masking. They are zero-initialized once in `new()`;
    /// `rewind(0)` must re-zero them so a reused session starts every generation
    /// from the declared `init: zeros` state (see `rewind`). Empty for pure-KV
    /// decoders.
    pub(crate) fixed_state_binding_range: std::ops::Range<usize>,
    pub(crate) auxiliary_binding_range: std::ops::Range<usize>,
    pub(crate) input_ids_binding: usize,
    pub(crate) position_ids_binding: Option<usize>,
    /// Coordinate rank of the `position_ids` device binding (`1` = `[1, 1]`
    /// conventional; `N > 1` = `[N, 1, 1]` multi-axis mrope). The captured step
    /// writes `position_rank` copies of the current position, one per axis.
    position_rank: usize,
    /// Persistent per-step input bindings (embeds + routed ports) written each
    /// step then replayed by the captured decode graph (Inc3c). Empty unless the
    /// capture-step-inputs path is active.
    captured_step_inputs: Vec<CapturedStepInputBinding>,
    /// When `true`, single-token decode with `inputs_embeds`/routed ports writes
    /// those ports into their persistent bindings and drives the captured
    /// `run_one_token` state machine (Inc3c) instead of the eager owned path.
    /// Gated by `ONNX_GENAI_NATIVE_DECODER_CAPTURE_STEP_INPUTS` (default off) and
    /// only enabled when graph capture is structurally available.
    capture_step_inputs: bool,
    logits_binding: usize,
    logits_shape: Vec<usize>,
    logits_dtype: DataType,
    greedy_result: DeviceIoBinding,
    pub(crate) graph_enabled: bool,
    /// `true` when the attention-mask binding exposes its *logical* valid length
    /// (not the padded physical capacity) to at least one consumer that is not a
    /// capacity-aware kernel — e.g. GLM-5.2's `indexer` arithmetic branch, which
    /// combines a logical-width score with the mask and would break if the mask
    /// leaked at physical `max_len`. When set, single-token decode must expose the
    /// mask at the growing logical length rather than freezing it to `max_len`,
    /// which also forfeits CUDA-graph capture (mirroring the eager prefill path).
    mask_exposes_logical: bool,
    graph_phase: DecodeCudaGraphPhase,
    /// Inc-1b PR-3: capture phase of the **decode-inline sibling** graph, tracked
    /// separately from `graph_phase` because the sibling executor owns its own
    /// capture schedule. Only advanced on the routed single-token decode path
    /// (a model with an inlineable single-trip `Scan`); `NeedsWarmup` and
    /// dormant otherwise.
    /// The sibling shares the main executor's EP (one graph slot + one latch), so
    /// this and `graph_phase` are never both past `NeedsWarmup` in one generation.
    inline_graph_phase: DecodeCudaGraphPhase,
    graph_captures: u64,
    graph_replays: u64,
    graph_fallbacks: u64,
    graph_invalidations: u64,
    kv_growth_events: u64,
    kv_growth_d2d_copy_bytes: u64,
    /// Count of KV-growth events that kept the captured graph (seq-major
    /// fixed-stride commit-on-demand path). Mutually exclusive with the
    /// growth-attributable slice of `graph_invalidations`.
    graph_growth_keeps: u64,
    /// Highest committed KV sequence length, in tokens. For the seq-major
    /// fixed-stride path this is the commit high-water mark that advances on
    /// growth **without** changing the reported physical shape; for head-major
    /// and legacy realloc it tracks the reported physical capacity.
    kv_committed_len: usize,
    /// Named rationale for the most recent KV-growth capture decision.
    graph_growth_decision: Option<String>,
    capacity: CudaKvCapacity,
    /// The KV bindings reserve their full context address range while the CUDA
    /// VMM allocator maps only the token stripes reached so far.
    kv_commits_on_demand: bool,
    /// The physical KV-cache layout this session's bindings are stored in,
    /// resolved from [`ModelIoSpec::kv_layout`] (absent = head-major, exactly
    /// the historical behavior). Consumed by the residency accounting and the
    /// commit-geometry decision so the binding layer is no longer layout-blind:
    /// seq-major's live prefix is one dense contiguous run, so its committed
    /// windows follow `ceil(live_bytes / granule)` rather than a flat bucket.
    kv_layout: KvCommitLayout,
    pub(crate) graph_decline_reason: Option<String>,
    pub(crate) graph_fallback_reason: Option<String>,
    pub(crate) graph_fallback_report: Option<CaptureDeclineReport>,
    /// Structural reasons, recorded at binding time, why one or more auxiliary
    /// graph outputs could not be persistently bound (an unresolved symbolic
    /// dimension that is not batch or query-seq). Non-empty here means CUDA
    /// graph capture was declined up front and the eager device path is in
    /// force for this generation. Empty when every auxiliary output was
    /// statically bindable.
    pub(crate) auxiliary_bind_declines: Vec<String>,
    /// When `false` (today's default), `NativeDecodeSession::rewind` invalidates
    /// the captured decode graph before rolling the device KV back — correct for
    /// the eager M=K verify path (option (b)), which captures nothing.
    ///
    /// When `true`, rewind performs a *contents-only* mutation (zero the mask
    /// tail + truncate the KV logical length) and **retains** the captured graph.
    /// This is the option (c) invariant: a single fixed-topology M=maxK graph
    /// whose device-binding pointers stay invariant while only buffer contents /
    /// logical shapes change across steps — exactly the data-driven mutation the
    /// captured graph already tolerates on the M=1 replay path. Kept dormant
    /// (default `false`) until WP4 graduates verify to the captured path.
    pub(crate) retain_graph_on_rewind: bool,
    /// Dormant option (c) scaffolding: the fixed query-row capacity (M=maxK) a
    /// padded single-capture verify graph would be captured at. `None` today —
    /// the eager verify path (option (b)) captures nothing. Set only by the
    /// dormant `configure_padded_verify_capture` switch (not on the hot path).
    #[cfg(test)]
    pub(crate) padded_query_capacity: Option<usize>,
}

pub(crate) struct DecodeCudaIo<'a> {
    pub(crate) input_ids: &'a str,
    /// Present only when the decoder's sequence source is `inputs_embeds` (a
    /// fused VLM decoder) rather than raw token ids. Carries the embed port
    /// name plus its element dtype and hidden width so the sequence device
    /// binding is allocated as a float `[1, 1, hidden]` embedding instead of an
    /// `Int64 [1, 1]` token id. `None` keeps the token-id path byte-identical.
    pub(crate) inputs_embeds: Option<CudaEmbedsBinding<'a>>,
    pub(crate) attention_mask: &'a str,
    pub(crate) position_ids: Option<&'a str>,
    pub(crate) logits: &'a str,
    /// Routed non-KV step-input ports (Inc3c). Empty for a pure token-id or
    /// embeds-only decoder. Each becomes a persistent device binding so the
    /// Inc3c capture path can replay the step; the eager path (default) ignores
    /// these and re-binds owned inputs instead.
    pub(crate) routed: Vec<CudaRoutedBinding<'a>>,
}

/// Metadata for a CUDA `inputs_embeds` sequence input (Inc3a). The native CUDA
/// decoder receives one token's embedding per step as a routed host tensor and
/// binds it on-device, keeping the KV cache device-resident.
pub(crate) struct CudaEmbedsBinding<'a> {
    pub(crate) name: &'a str,
    pub(crate) dtype: DataType,
    pub(crate) hidden: usize,
}

/// Metadata for a persistent CUDA routed step-input binding (Inc3c capture). A
/// routed port (e.g. Gemma 3n's `per_layer_inputs`) that the decoder consumes
/// each step. The Inc3c capture path allocates a fixed `[1, 1, width]` device
/// binding (dynamic batch/sequence dims collapsed to 1) and writes the one-token
/// bytes into it each step — instead of re-binding a fresh owned input — so the
/// decode graph can be CUDA-graph captured and replayed.
pub(crate) struct CudaRoutedBinding<'a> {
    pub(crate) name: &'a str,
    pub(crate) dtype: DataType,
    /// Fixed device shape for a single decode step (dynamic dims collapsed to 1).
    pub(crate) shape: Vec<usize>,
}

/// A per-step input port with a persistent device binding written each step then
/// replayed by the captured decode graph (Inc3c): the `inputs_embeds` sequence
/// binding plus any routed ports. Position/token ids are generated (written via
/// [`DecodeCudaState::write_decode_inputs`]), not listed here.
struct CapturedStepInputBinding {
    name: String,
    binding_index: usize,
    byte_len: usize,
}

/// Whether the Inc3c captured per-step-input path is enabled. It is now
/// **default-on** (the CUDA-graph capture perf win for multi-component/routed
/// decoders): a capture-eligible `inputs_embeds`/routed decode step reuses the
/// persistent bindings + `run_one_token` graph instead of paying the eager
/// per-step kernel-launch cost. The env var
/// `ONNX_GENAI_NATIVE_DECODER_CAPTURE_STEP_INPUTS` is an **opt-out** escape
/// hatch: a falsy value (`0`/`false`/`no`/`off`) forces the eager owned path.
/// Any other value (including unset or truthy) keeps capture on. Structural
/// eligibility gates (`graph_enabled`, non-empty `captured_step_inputs`) still
/// decide whether an individual decoder actually captures, so an ineligible
/// decoder auto-declines to eager regardless of this flag.
fn capture_step_inputs_enabled() -> bool {
    capture_step_inputs_from_env_value(
        std::env::var("ONNX_GENAI_NATIVE_DECODER_CAPTURE_STEP_INPUTS")
            .ok()
            .as_deref(),
    )
}

/// Pure opt-out parse for `ONNX_GENAI_NATIVE_DECODER_CAPTURE_STEP_INPUTS`.
/// Capture is **default-on**: only an explicit falsy value
/// (`0`/`false`/`no`/`off`, case/whitespace-insensitive) opts out to eager;
/// unset (`None`) or any other value keeps capture enabled.
fn capture_step_inputs_from_env_value(value: Option<&str>) -> bool {
    match value {
        Some(value) => !matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "0" | "false" | "no" | "off"
        ),
        None => true,
    }
}

pub(crate) fn trace_capture_declines(trace: &TraceContext, report: &CaptureDeclineReport) {
    for decline in &report.entries {
        if let Some(node_id) = decline.node_id {
            capture_rejected(
                trace,
                node_id,
                decline.op_type.as_str(),
                decline.domain.as_str(),
                decline.reason.as_str(),
            );
        }
    }
}

impl NativeDecodeSession {
    pub(crate) fn prepare_cuda_prefill_workspace(
        &mut self,
        token_ids: &[TokenId],
    ) -> anyhow::Result<onnx_runtime_session::WorkspaceRequirement> {
        self.prepare_cuda_prefill_workspace_with_step_inputs(token_ids, 0, &[])
    }

    pub(crate) fn prepare_cuda_prefill_workspace_with_step_inputs(
        &mut self,
        token_ids: &[TokenId],
        past_len: usize,
        step_inputs: &[(String, Tensor)],
    ) -> anyhow::Result<onnx_runtime_session::WorkspaceRequirement> {
        if self.has_eager_step_inputs() {
            let workspace_nodes = self.session.workspace_node_locations();
            if workspace_nodes.is_empty() {
                return Ok(onnx_runtime_session::WorkspaceRequirement::NONE);
            }
        }
        let position_input = self
            .step_input_name(NativeStepInputSource::PositionIds)
            .map(str::to_owned);
        let input_len = token_ids.len();
        let total_len = past_len
            .checked_add(input_len)
            .context("native CUDA workspace preparation length overflow")?;
        let supplied = step_inputs
            .iter()
            .map(|(name, tensor)| (name.as_str(), tensor))
            .collect::<HashMap<_, _>>();
        let mut owned =
            if let Some(name) = self.step_input_name(NativeStepInputSource::InputsEmbeds) {
                if let Some(tensor) = supplied.get(name) {
                    vec![(name.to_owned(), (*tensor).try_clone()?)]
                } else {
                    let meta = self
                        .session
                        .inputs()
                        .iter()
                        .find(|meta| meta.name == name)
                        .with_context(|| {
                            format!("missing native CUDA inputs_embeds metadata for '{name}'")
                        })?;
                    let hidden = match meta.shape.last() {
                        Some(Dim::Static(hidden)) => *hidden,
                        _ => bail!(
                            "prepare-only QMoE workspace planning cannot conservatively bind \
                             inputs_embeds '{name}': expected a static hidden width, got {:?}",
                            meta.shape
                        ),
                    };
                    vec![(
                        name.to_owned(),
                        Tensor::zeros(meta.dtype, vec![1, input_len, hidden])?,
                    )]
                }
            } else {
                let token_input = self
                    .step_input_name(NativeStepInputSource::TokenIds)
                    .context("native CUDA decoder has no token or inputs_embeds input binding")?
                    .to_owned();
                let ids = token_ids
                    .iter()
                    .map(|&id| i64::from(id))
                    .collect::<Vec<_>>();
                vec![(token_input, Tensor::from_i64(&[1, input_len], &ids)?)]
            };
        for (name, tensor) in step_inputs {
            if !owned.iter().any(|(bound, _)| bound == name) {
                owned.push((name.clone(), tensor.try_clone()?));
            }
        }
        if let Some(position_input) = position_input {
            owned.push((
                position_input,
                self.build_step_positions(past_len, total_len)?,
            ));
        }
        let state = self
            .cuda
            .as_mut()
            .context("CUDA decode state is not initialized")?;
        state.ensure_capacity(&mut self.session, total_len)?;
        state.extend_mask(past_len, total_len, total_len)?;
        let inputs = owned
            .iter()
            .map(|(name, tensor)| (name.as_str(), tensor))
            .collect::<Vec<_>>();
        self.session
            .prepare_with_device_bindings(&inputs, &mut state.bindings[..state.base_binding_count])
            .context("prepare native CUDA prefill workspace")
    }

    /// Run one eager (uncaptured) `[1, K]` device forward pass and return host
    /// `[K, vocab]` logits.
    ///
    /// This is the shared body of `decode_cuda`'s multi-token branch and the
    /// `decode_cuda_eager` verify path: invalidate any captured graph, rebuild
    /// the host token/position input tensors, run against the device KV/mask
    /// bindings, collect and validate the logits output, and advance the KV
    /// logical length. `error_context` (`"decoder"` or `"verify"`) only selects
    /// the wording of the two diagnostic messages so the extraction stays
    /// byte-identical to the two original inlined bodies.
    ///
    /// The caller resolves the token/position input names and computes
    /// `total_len`, and is responsible for the preceding `extend_mask` call
    /// (whose exposed length differs between the two paths).
    #[allow(clippy::too_many_arguments)]
    fn run_cuda_eager_rows(
        &mut self,
        token_ids: &[TokenId],
        past_len: usize,
        total_len: usize,
        token_input: &str,
        position_input: Option<&str>,
        error_context: &str,
    ) -> anyhow::Result<Vec<Vec<f32>>> {
        let ids = token_ids
            .iter()
            .map(|&id| i64::from(id))
            .collect::<Vec<_>>();
        let input_ids = Tensor::from_i64(&[1, token_ids.len()], &ids)?;
        let mut owned = Vec::with_capacity(2);
        owned.push((token_input.to_owned(), input_ids));
        if let Some(position_ids_name) = position_input {
            owned.push((
                position_ids_name.to_owned(),
                self.build_step_positions(past_len, total_len)?,
            ));
        }
        self.run_cuda_eager_rows_owned(owned, total_len, error_context)
    }

    /// Shared eager (uncaptured) device forward body: bind the caller-provided
    /// non-KV inputs (`owned` — a token or `inputs_embeds` sequence tensor plus
    /// optional position ids) against the persistent device KV/mask bindings,
    /// run, collect host `[K, vocab]` logits, and advance the KV logical length.
    ///
    /// Both the token path (via [`Self::run_cuda_eager_rows`]) and the
    /// `inputs_embeds` path (Inc3a) share this body; only the construction of
    /// `owned` differs, so the KV device-residency guarantee is identical.
    fn run_cuda_eager_rows_owned(
        &mut self,
        owned: Vec<(String, Tensor)>,
        total_len: usize,
        error_context: &str,
    ) -> anyhow::Result<Vec<Vec<f32>>> {
        let state = self
            .cuda
            .as_mut()
            .context("CUDA decode state is not initialized")?;
        state.invalidate_graph(&mut self.session)?;
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
                return Err(anyhow::Error::new(error).context(format!(
                    "native CUDA {error_context} forward pass failed{diagnosis}"
                )));
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
                "native CUDA {error_context} unexpectedly materialized bound outputs: {:?}",
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

    pub(crate) fn decode_cuda(
        &mut self,
        token_ids: &[TokenId],
        past_len: usize,
        step_inputs: &[(String, Tensor)],
    ) -> anyhow::Result<Vec<Vec<f32>>> {
        let total_len = past_len
            .checked_add(token_ids.len())
            .context("native decode context length overflow")?;
        // Any declared `inputs_embeds` or generic `Routed` port forces the eager
        // device path: those ports arrive as per-step host tensors and are bound
        // as owned uploads, while the KV + attention-mask stay device-resident in
        // the persistent bindings (never round-tripping). The captured single-
        // token fast path below writes only the fixed token/mask/KV bindings, so
        // it cannot carry routed ports — a pure token-id decode (no routed ports)
        // keeps that byte-identical path. Inc3a introduced this for embeds; Inc3b
        // generalizes it to arbitrary routed ports via one shared owned build.
        if self.has_eager_step_inputs() {
            // Inc3c: for a single-token decode step, drive the per-step ports
            // through their persistent device bindings + the captured
            // `run_one_token` state machine (recovering the graph-capture fast
            // path lost to the eager owned uploads). This is default-on for
            // capture-eligible decoders (`state.capture_step_inputs`); multi-token
            // prefill, an ineligible decoder, or the `…CAPTURE_STEP_INPUTS=0`
            // opt-out use the eager owned path.
            let capture = token_ids.len() == 1
                && self
                    .cuda
                    .as_ref()
                    .is_some_and(|state| state.capture_step_inputs);
            if capture {
                return self.decode_cuda_captured_step_inputs(
                    token_ids,
                    past_len,
                    total_len,
                    step_inputs,
                );
            }
            return self.decode_cuda_eager_step_inputs(token_ids, past_len, total_len, step_inputs);
        }
        if let Some((name, _)) = step_inputs.first() {
            bail!(
                "native CUDA target decode received routed host step input '{name}' but the decoder declares no matching inputs_embeds/routed port; declare a pipeline dataflow edge to this exact decoder port"
            );
        }
        let token_input = self
            .step_input_name(NativeStepInputSource::TokenIds)
            .context("native CUDA decoder has no token input binding")?
            .to_owned();
        let position_input = self
            .step_input_name(NativeStepInputSource::PositionIds)
            .map(str::to_owned);
        // Inc-1b PR-2: decide decode-inline routing before borrowing `cuda`
        // mutably below (the decision reads `self.session`/`self.decode_inline`).
        let route_inline = self.route_decode_inline(token_ids);
        let state = self
            .cuda
            .as_mut()
            .context("CUDA decode state is not initialized")?;
        if total_len > state.capacity.max_len {
            bail!("{}", state.capacity_exceeded_error(total_len));
        }
        let grew = state.ensure_capacity(&mut self.session, total_len)?;
        // Single-token decode freezes the mask to the current physical bucket so
        // the step is CUDA-graph-capture eligible. This is not a fixed-capacity
        // constraint: when `ensure_capacity` grows the bucket, it invalidates the
        // old capture and the existing state machine re-captures against the new
        // buffers, matching ORT shared-buffer growth. Multi-token prefill keeps
        // the growing logical length (prefix-sensitive causal island). A mask
        // whose logical valid length feeds a non-capacity-aware consumer (see
        // `decode_mask_expose_len`) cannot be frozen and uses `total_len`.
        let mask_expose = if token_ids.len() == 1 {
            state.decode_mask_expose_len(total_len)
        } else {
            total_len
        };
        state.extend_mask(if grew { 0 } else { past_len }, total_len, mask_expose)?;

        if token_ids.len() == 1 {
            state.write_decode_inputs(token_ids[0], past_len)?;
            if route_inline {
                // Inc-1b PR-3: route this single-token decode step to the
                // decode-specialized inlined-body sibling exec and drive its
                // CUDA-graph capture state machine so the inlined body folds into
                // a replayed device graph (capture engages only because
                // `route_inline` is true — i.e. the model has an inlineable
                // single-trip recurrent `Scan` and a sibling was built). It binds the identical persistent device
                // KV/state bindings, so recurrent-state continuity across the
                // prefill→decode boundary is preserved (design §3). The main
                // exec's capture machine stays dormant on decode, so the shared
                // EP's single graph slot + capture-error latch are owned solely by
                // the sibling.
                let step_profile = CudaStepProfile::begin(past_len, total_len);
                let mut step_wall = CudaStepWallBreakdown::default();
                let run_start = std::time::Instant::now();
                if let Err(error) = state.run_one_token_inline(&mut self.session, &self.trace) {
                    let diagnosis = diagnose_native_failure(&self.session, &error.to_string());
                    return Err(error.context(format!(
                        "native CUDA decode-inline forward pass failed{diagnosis}"
                    )));
                }
                step_wall.run_ms = run_start.elapsed().as_secs_f64() * 1_000.0;
                let logits_start = std::time::Instant::now();
                let logits = state.read_logits()?;
                step_wall.logits_read_ms = logits_start.elapsed().as_secs_f64() * 1_000.0;
                // Detection-before-consumption (Harry #588 PR-3 rec #1): the
                // logits read above is the single per-step device→host sync.
                // Piggyback on it to poll the shared capture-error word — a
                // captured sibling replay that violated a device-side bound
                // latches the flag; fail hard before consuming the produced token.
                // The latch lives on the shared EP, so this reads the sibling's
                // replay result even though the poll is the main-exec-facing call.
                let capture_check_start = std::time::Instant::now();
                let capture_error = self.session.check_device_capture_error()?;
                step_wall.capture_check_ms = capture_check_start.elapsed().as_secs_f64() * 1_000.0;
                if capture_error != 0 {
                    let _ = state.invalidate_graph(&mut self.session);
                    bail!(
                        "native CUDA decode-inline aborted: device capture validation violation (flags=0x{capture_error:x}) detected during captured graph replay; the produced token was rejected before consumption and the decode-inline graph was invalidated"
                    );
                }
                let finite_check_start = std::time::Instant::now();
                if logits.iter().flatten().any(|value| !value.is_finite()) {
                    bail!("native decoder produced non-finite logits");
                }
                step_wall.finite_check_ms = finite_check_start.elapsed().as_secs_f64() * 1_000.0;
                if let Some(profile) = step_profile {
                    profile.finish("decode_inline", step_wall);
                }
                state.set_logical_len(total_len)?;
                self.current_len = total_len;
                return Ok(logits);
            }
            let step_profile = CudaStepProfile::begin(past_len, total_len);
            let mut step_wall = CudaStepWallBreakdown::default();
            let run_start = std::time::Instant::now();
            if let Err(error) = state.run_one_token(&mut self.session, &self.trace) {
                let diagnosis = diagnose_native_failure(&self.session, &error.to_string());
                return Err(error.context(format!(
                    "native CUDA decoder forward pass failed{diagnosis}"
                )));
            }
            step_wall.run_ms = run_start.elapsed().as_secs_f64() * 1_000.0;
            let logits_start = std::time::Instant::now();
            let logits = state.read_logits()?;
            step_wall.logits_read_ms = logits_start.elapsed().as_secs_f64() * 1_000.0;
            // Detection-before-consumption: the logits read above is the single
            // per-step device→host sync. Piggyback on it to poll the shared
            // capture-error word (no extra synchronize). If a captured replay
            // violates a device-side bound, kernels latch the flag and avoid the
            // unsafe access, so fail hard before consuming the produced token.
            let capture_check_start = std::time::Instant::now();
            let capture_error = self.session.check_device_capture_error()?;
            step_wall.capture_check_ms = capture_check_start.elapsed().as_secs_f64() * 1_000.0;
            if capture_error != 0 {
                let _ = state.invalidate_graph(&mut self.session);
                bail!(
                    "native CUDA decoder aborted: device capture validation violation (flags=0x{capture_error:x}) detected during captured graph replay; the produced token was rejected before consumption and the decode graph was invalidated"
                );
            }
            let finite_check_start = std::time::Instant::now();
            if logits.iter().flatten().any(|value| !value.is_finite()) {
                bail!("native decoder produced non-finite logits");
            }
            step_wall.finite_check_ms = finite_check_start.elapsed().as_secs_f64() * 1_000.0;
            if let Some(profile) = step_profile {
                profile.finish("decode", step_wall);
            }
            state.set_logical_len(total_len)?;
            self.current_len = total_len;
            return Ok(logits);
        }

        self.run_cuda_eager_rows(
            token_ids,
            past_len,
            total_len,
            &token_input,
            position_input.as_deref(),
            "decoder",
        )
    }

    /// True when the decoder declares any `inputs_embeds` or generic `Routed`
    /// step input — the ports that must be uploaded per step and therefore force
    /// the eager (uncaptured) CUDA device path in [`Self::decode_cuda`], and are
    /// excluded from the decode-inline fast path by `route_decode_inline`.
    pub(super) fn has_eager_step_inputs(&self) -> bool {
        self.step_input_name(NativeStepInputSource::InputsEmbeds)
            .is_some()
            || self
                .step_input_name(NativeStepInputSource::Routed)
                .is_some()
    }

    /// Inc3c captured single-token decode for decoders with `inputs_embeds`
    /// and/or routed ports. Instead of re-binding fresh owned inputs (the eager
    /// path), it writes the one-token embedding + routed bytes into their
    /// **persistent** device bindings — mirroring how the token path writes the
    /// token id via [`DecodeCudaState::write_decode_inputs`] — then drives the
    /// CUDA-graph warmup/capture/replay state machine (`run_one_token`). The mask
    /// is frozen to the physical bucket (capture-eligible) and the KV cache stays
    /// device-resident and is advanced inside the captured graph, exactly like
    /// the token-id captured path. This is the Inc3c perf lever: it recovers the
    /// graph-capture fast path the eager per-step uploads forfeited.
    fn decode_cuda_captured_step_inputs(
        &mut self,
        token_ids: &[TokenId],
        past_len: usize,
        total_len: usize,
        supplied: &[(String, Tensor)],
    ) -> anyhow::Result<Vec<Vec<f32>>> {
        debug_assert_eq!(
            token_ids.len(),
            1,
            "captured step-input path is single-token"
        );
        super::NATIVE_DECODER_CAPTURED_STEP_INPUT_DECODES
            .fetch_add(1, super::AtomicOrdering::Relaxed);
        let position = past_len;
        let state = self
            .cuda
            .as_mut()
            .context("CUDA decode state is not initialized")?;
        if total_len > state.capacity.max_len {
            bail!("{}", state.capacity_exceeded_error(total_len));
        }
        let grew = state.ensure_capacity(&mut self.session, total_len)?;
        // Freeze the mask to the physical bucket so the step stays capture
        // eligible (identical to the token-id captured path); bucket growth
        // re-captures against the new buffers.
        let mask_expose = state.decode_mask_expose_len(total_len);
        state.extend_mask(if grew { 0 } else { past_len }, total_len, mask_expose)?;
        state.write_captured_step_inputs(supplied, position)?;

        let step_profile = CudaStepProfile::begin(past_len, total_len);
        let mut step_wall = CudaStepWallBreakdown::default();
        let run_start = std::time::Instant::now();
        if let Err(error) = state.run_one_token(&mut self.session, &self.trace) {
            let diagnosis = diagnose_native_failure(&self.session, &error.to_string());
            return Err(error.context(format!(
                "native CUDA decoder forward pass failed{diagnosis}"
            )));
        }
        step_wall.run_ms = run_start.elapsed().as_secs_f64() * 1_000.0;
        let logits_start = std::time::Instant::now();
        let logits = state.read_logits()?;
        step_wall.logits_read_ms = logits_start.elapsed().as_secs_f64() * 1_000.0;
        // Detection-before-consumption: piggyback on the single logits sync to
        // poll the shared capture-error word (mirrors the token captured path).
        let capture_check_start = std::time::Instant::now();
        let capture_error = self.session.check_device_capture_error()?;
        step_wall.capture_check_ms = capture_check_start.elapsed().as_secs_f64() * 1_000.0;
        if capture_error != 0 {
            let _ = self
                .cuda
                .as_mut()
                .context("CUDA decode state is not initialized")?
                .invalidate_graph(&mut self.session);
            bail!(
                "native CUDA decoder aborted: device capture validation violation (flags=0x{capture_error:x}) detected during captured graph replay of a per-step-input decode; the produced token was rejected before consumption and the decode graph was invalidated"
            );
        }
        let finite_check_start = std::time::Instant::now();
        if logits.iter().flatten().any(|value| !value.is_finite()) {
            bail!("native decoder produced non-finite logits");
        }
        step_wall.finite_check_ms = finite_check_start.elapsed().as_secs_f64() * 1_000.0;
        if let Some(profile) = step_profile {
            profile.finish("captured_step_inputs", step_wall);
        }
        let state = self
            .cuda
            .as_mut()
            .context("CUDA decode state is not initialized")?;
        state.set_logical_len(total_len)?;
        self.current_len = total_len;
        Ok(logits)
    }

    /// Generic eager CUDA decode step for decoders with `inputs_embeds` and/or
    /// arbitrary `Routed` ports (Inc3a embeds, generalized in Inc3b). Builds the
    /// owned per-step upload set from **every** declared non-KV step input the
    /// same way the CPU path does (`prepare_cpu_step_inputs`) — generating token
    /// ids / position ids, pulling `inputs_embeds`/routed tensors from `supplied`
    /// by exact graph-port name — with the single CUDA-specific exclusion that
    /// `attention_mask` is a **persistent device binding** (filled by
    /// `extend_mask`), never an owned input. So only the small per-step tensors
    /// (embedding + any routed hidden/state, one token's worth) cross host→
    /// device; the attention mask and KV cache stay device-resident on the GPU.
    fn decode_cuda_eager_step_inputs(
        &mut self,
        token_ids: &[TokenId],
        past_len: usize,
        total_len: usize,
        supplied: &[(String, Tensor)],
    ) -> anyhow::Result<Vec<Vec<f32>>> {
        let owned =
            self.prepare_cuda_owned_step_inputs(token_ids, past_len, total_len, supplied)?;

        let state = self
            .cuda
            .as_mut()
            .context("CUDA decode state is not initialized")?;
        if total_len > state.capacity.max_len {
            bail!("{}", state.capacity_exceeded_error(total_len));
        }
        let grew = state.ensure_capacity(&mut self.session, total_len)?;
        // These uploads are fresh device inputs every step, so this path is not
        // CUDA-graph captured; expose the growing logical mask width (matching the
        // eager token forward) rather than freezing to physical capacity.
        state.extend_mask(if grew { 0 } else { past_len }, total_len, total_len)?;

        self.run_cuda_eager_rows_owned(owned, total_len, "decoder")
    }

    /// Build the owned per-step upload set for the eager CUDA path from every
    /// declared non-KV step input, mirroring the CPU `prepare_cpu_step_inputs`
    /// contract but skipping `attention_mask` (a persistent CUDA device binding).
    /// Errors on any routed port that was not supplied, or any supplied tensor
    /// that maps to no declared port — the same validation the CPU path applies.
    fn prepare_cuda_owned_step_inputs(
        &self,
        token_ids: &[TokenId],
        past_len: usize,
        total_len: usize,
        supplied: &[(String, Tensor)],
    ) -> anyhow::Result<Vec<(String, Tensor)>> {
        let mut supplied_map = HashMap::with_capacity(supplied.len());
        for (name, tensor) in supplied {
            if supplied_map.insert(name.as_str(), tensor).is_some() {
                bail!("native CUDA decode received duplicate routed step input '{name}'");
            }
        }

        let mut owned = Vec::with_capacity(self.step_inputs.len());
        for binding in &self.step_inputs {
            let tensor = match binding.source {
                // The attention mask is a persistent device binding on CUDA
                // (filled by `extend_mask`), never an owned per-step upload.
                NativeStepInputSource::AttentionMask => continue,
                NativeStepInputSource::TokenIds => {
                    let ids = token_ids.iter().map(|&id| i64::from(id)).collect::<Vec<_>>();
                    Tensor::from_i64(&[1, token_ids.len()], &ids)?
                }
                NativeStepInputSource::PositionIds => {
                    self.build_step_positions(past_len, total_len)?
                }
                NativeStepInputSource::InputsEmbeds => supplied_map
                    .remove(binding.name.as_str())
                    .cloned()
                    .with_context(|| {
                        format!(
                            "declared inputs_embeds input '{}' was not supplied to the native CUDA decode step; route the current embedding component output to this exact decoder port",
                            binding.name
                        )
                    })?,
                NativeStepInputSource::Routed => supplied_map
                    .remove(binding.name.as_str())
                    .cloned()
                    .with_context(|| {
                        format!(
                            "native CUDA decode graph input '{}' has no generated role and no routed step tensor; declare a pipeline dataflow edge to this exact decoder port",
                            binding.name
                        )
                    })?,
            };
            owned.push((binding.name.clone(), tensor));
        }

        if !supplied_map.is_empty() {
            let mut unknown = supplied_map.keys().copied().collect::<Vec<_>>();
            unknown.sort_unstable();
            bail!(
                "native CUDA decode received routed step inputs that are not declared graph ports: {unknown:?}"
            );
        }
        Ok(owned)
    }

    /// Speculative **verify** primitive (option (b): the safe eager M=K path).
    ///
    /// Runs the `draft` candidate tokens (K = `draft.len()`) through the target
    /// in a single eager forward and returns `[K, vocab]` host logits — one
    /// predicted-distribution row per draft position. This is the primitive
    /// WP2/WP3 build on: the driver compares each row's argmax against `draft`
    /// to find the accepted prefix (plus the free bonus token) and then rewinds
    /// the device KV to the committed length.
    ///
    /// It never enters the M=1 captured-graph greedy hot path — it always takes
    /// the eager multi-token forward (`decode_cuda_eager`) so the 762 tok/s plain
    /// path stays byte-identical. Greedy is the target regime, but returning raw
    /// logits also lets a driver fall back to host sampling for non-greedy
    /// requests. `past` must equal the committed length (`current_len`).
    pub fn decode_verify(
        &mut self,
        draft: &[TokenId],
        past: usize,
    ) -> anyhow::Result<Vec<Vec<f32>>> {
        if draft.is_empty() {
            bail!("native decode_verify requires at least one draft token");
        }
        if past != self.current_len {
            bail!(
                "native decode_verify past length mismatch: caller supplied {past}, adapter holds {}",
                self.current_len
            );
        }
        if self.cuda.is_some() {
            return self.decode_cuda_eager(draft, past);
        }
        // CPU sessions already run any M>1 forward eagerly through the shared
        // decode path, which returns the full [K, vocab] rows verify needs.
        <Self as DecodeBackend>::decode(self, draft, past)
    }

    /// Eager multi-token (M=K) CUDA forward used by the verify primitive.
    ///
    /// Self-contained on purpose: it mirrors `decode_cuda`'s eager branch but is
    /// a *separate* method so the M=1 captured-graph hot path in `decode_cuda`
    /// stays byte-identical and out of verify's blast radius. It invalidates any
    /// captured graph (option (b) captures nothing), rebuilds host `[1,K]`
    /// input/position tensors, runs against the device KV/mask bindings, and
    /// advances the KV logical length to `past_len + K`.
    ///
    /// The whole pass is wrapped in its own trace span so Deckard's per-op
    /// timings under it remain attributable to the verify forward.
    fn decode_cuda_eager(
        &mut self,
        token_ids: &[TokenId],
        past_len: usize,
    ) -> anyhow::Result<Vec<Vec<f32>>> {
        let _verify_span = self
            .trace
            .span("native_decode_verify", "spec")
            .with_args(Args::new().with("rows", token_ids.len() as u64));
        let total_len = past_len
            .checked_add(token_ids.len())
            .context("native decode context length overflow")?;
        let token_input = self
            .step_input_name(NativeStepInputSource::TokenIds)
            .context("native CUDA decoder has no token input binding")?
            .to_owned();
        let position_input = self
            .step_input_name(NativeStepInputSource::PositionIds)
            .map(str::to_owned);
        let state = self
            .cuda
            .as_mut()
            .context("CUDA decode state is not initialized")?;
        if total_len > state.capacity.max_len {
            bail!("{}", state.capacity_exceeded_error(total_len));
        }
        let grew = state.ensure_capacity(&mut self.session, total_len)?;
        state.extend_mask(if grew { 0 } else { past_len }, total_len, total_len)?;
        self.run_cuda_eager_rows(
            token_ids,
            past_len,
            total_len,
            &token_input,
            position_input.as_deref(),
            "verify",
        )
    }

    pub(crate) fn decode_cuda_greedy(
        &mut self,
        token_id: TokenId,
        past_len: usize,
    ) -> anyhow::Result<TokenId> {
        let route_inline = self.route_decode_inline(std::slice::from_ref(&token_id));
        let total_len = past_len
            .checked_add(1)
            .context("native decode context length overflow")?;
        let state = self
            .cuda
            .as_mut()
            .context("CUDA decode state is not initialized")?;
        if total_len > state.capacity.max_len {
            bail!("{}", state.capacity_exceeded_error(total_len));
        }
        let grew = state.ensure_capacity(&mut self.session, total_len)?;
        state.extend_mask(
            if grew { 0 } else { past_len },
            total_len,
            state.decode_mask_expose_len(total_len),
        )?;
        state.write_decode_inputs(token_id, past_len)?;
        if route_inline {
            // Inc-1b PR-3: run this single-token decode step through the
            // decode-specialized inlined-body sibling exec, driving its CUDA-graph
            // capture state machine so the inlined body folds into a replayed
            // device graph (flag-gated via `route_inline`). It binds the identical
            // persistent device KV/state bindings, so recurrent-state continuity
            // across the prefill→decode boundary is preserved (design §3). The
            // greedy token is read with the same device-argmax kernel the captured
            // path uses, so tie-breaking is byte-identical and the full logits
            // never round-trip to the host.
            if let Err(error) = state.run_one_token_inline(&mut self.session, &self.trace) {
                let diagnosis = diagnose_native_failure(&self.session, &error.to_string());
                return Err(error.context(format!(
                    "native CUDA decode-inline forward pass failed{diagnosis}"
                )));
            }
            // Detection-before-consumption (Harry #588 PR-3 rec #1): the greedy
            // device-argmax read already returns the shared capture-error word;
            // reject the token before consumption if a captured sibling replay
            // latched a device-side violation.
            let (token_id, capture_error) = state.read_greedy_result()?;
            if capture_error != 0 {
                let _ = state.invalidate_graph(&mut self.session);
                bail!(
                    "native CUDA decode-inline aborted: device capture validation violation (flags=0x{capture_error:x}) detected during captured graph replay; the produced token was rejected before consumption and the decode-inline graph was invalidated"
                );
            }
            state.set_logical_len(total_len)?;
            self.current_len = total_len;
            return Ok(token_id);
        }
        if let Err(error) = state.run_one_token(&mut self.session, &self.trace) {
            let diagnosis = diagnose_native_failure(&self.session, &error.to_string());
            return Err(error.context(format!(
                "native CUDA decoder forward pass failed{diagnosis}"
            )));
        }
        let (token_id, capture_error) = state.read_greedy_result()?;
        if capture_error != 0 {
            let _ = state.invalidate_graph(&mut self.session);
            bail!(
                "native CUDA decoder aborted: device capture validation violation (flags=0x{capture_error:x}) detected during captured graph replay; the produced token was rejected before consumption and the decode graph was invalidated"
            );
        }
        state.set_logical_len(total_len)?;
        self.current_len = total_len;
        Ok(token_id)
    }
}

impl DecodeCudaState {
    pub(crate) fn kv_bytes_per_token(
        session: &InferenceSession,
        present_to_past: &HashMap<String, String>,
        fixed_state_inputs: &HashSet<String>,
    ) -> anyhow::Result<usize> {
        let mut bytes = std::mem::size_of::<i64>();
        for past in present_to_past.values() {
            if fixed_state_inputs.contains(past) {
                continue;
            }
            let meta = session
                .inputs()
                .iter()
                .find(|meta| meta.name == *past)
                .with_context(|| format!("missing CUDA KV input metadata for '{past}'"))?;
            if !matches!(
                meta.dtype,
                DataType::Float32 | DataType::Float16 | DataType::BFloat16
            ) || meta.shape.len() != 4
            {
                bail!(
                    "CUDA KV input '{past}' must be rank-4 f32, f16 or bf16, got {:?} {:?}",
                    meta.dtype,
                    meta.shape
                );
            }
            let mut elements_per_token = 1usize;
            for (axis, dim) in meta.shape.iter().copied().enumerate() {
                let dim = if axis == 0 || axis == 2 {
                    1
                } else if let Dim::Static(value) = dim {
                    value
                } else {
                    bail!(
                        "cannot infer CUDA KV dimension {axis} for '{past}' shape {:?}",
                        meta.shape
                    );
                };
                elements_per_token = elements_per_token.checked_mul(dim).with_context(|| {
                    format!(
                        "CUDA KV bytes-per-token overflow for '{past}' shape {:?}",
                        meta.shape
                    )
                })?;
            }
            bytes = bytes
                .checked_add(
                    meta.dtype
                        .checked_storage_bytes(elements_per_token)
                        .with_context(|| {
                            format!(
                                "CUDA KV bytes-per-token overflow for '{past}' shape {:?}",
                                meta.shape
                            )
                        })?,
                )
                .with_context(|| {
                    format!(
                        "CUDA KV bytes-per-token overflow for '{past}' shape {:?}",
                        meta.shape
                    )
                })?;
        }
        Ok(bytes)
    }

    /// Build the padded physical and initial logical shapes for a persistent KV
    /// (`fixed == false`) or recurrent-state (`fixed == true`) binding.
    ///
    /// The KV shape is always **BNSH** `[1, kv_heads, max_len, head_dim]`,
    /// growing on axis 2, for *both* the head-major and seq-major layouts. This
    /// is deliberate and is the KV binding contract: the CUDA GQA node validates
    /// `past_key`/`past_value` as `[batch, kv_heads, seq, head_dim]` and reads
    /// `present_capacity` from axis 2 **regardless** of the `kv_layout`
    /// attribute (which only re-specializes the kernel's stride arithmetic). A
    /// seq-major binding therefore keeps this BNSH metadata shape; its BSNH
    /// physical byte layout — and the capacity-independent fixed per-token stride
    /// that lets it grow without moving data — is expressed by the growth/commit
    /// geometry (`kv_growth_byte_layout`, `apply_vmm_growth`,
    /// `build_grown_buffers`), not by permuting this shape. See
    /// `docs/MEMORY_ARCHITECTURE.md`, "KV layout and residency".
    pub(crate) fn persistent_state_shapes(
        name: &str,
        dtype: DataType,
        shape: &[Dim],
        max_len: usize,
        fixed: bool,
    ) -> anyhow::Result<(Vec<usize>, Vec<usize>)> {
        if !matches!(
            dtype,
            DataType::Float32 | DataType::Float16 | DataType::BFloat16
        ) {
            bail!(
                "CUDA decoder state input '{name}' must be f32, f16 or bf16, got {dtype:?} {shape:?}"
            );
        }
        if !fixed {
            if shape.len() != 4 {
                bail!(
                    "CUDA KV input '{name}' must be rank-4 f32, f16 or bf16, got {dtype:?} {shape:?}"
                );
            }
            let mut physical_shape = Vec::with_capacity(4);
            for (axis, dim) in shape.iter().copied().enumerate() {
                let value = if axis == 0 {
                    1
                } else if axis == 2 {
                    max_len
                } else if let Dim::Static(value) = dim {
                    value
                } else {
                    bail!("cannot infer CUDA KV dimension {axis} for '{name}' shape {shape:?}");
                };
                physical_shape.push(value);
            }
            let mut logical_shape = physical_shape.clone();
            logical_shape[2] = 0;
            return Ok((physical_shape, logical_shape));
        }

        let physical_shape = shape
            .iter()
            .copied()
            .enumerate()
            .map(|(axis, dim)| {
                if axis == 0 {
                    Ok(1)
                } else if let Dim::Static(value) = dim {
                    Ok(value)
                } else {
                    bail!(
                        "cannot allocate fixed CUDA decoder state '{name}': dimension {axis} in shape {shape:?} is symbolic; export a static recurrent-state geometry"
                    )
                }
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        checked_shape_bytes(&physical_shape, dtype).with_context(|| {
            format!(
                "fixed CUDA decoder state allocation size overflow for '{name}' shape {physical_shape:?}"
            )
        })?;
        Ok((physical_shape.clone(), physical_shape))
    }

    fn capacity_exceeded_error(&self, requested: usize) -> String {
        cuda_kv_capacity_exceeded_message(requested, &self.capacity)
    }

    fn growth_failed_error(
        &self,
        old_capacity: usize,
        new_capacity: usize,
        error: anyhow::Error,
        memory_at_failure: Result<CudaDeviceMemorySnapshot, String>,
    ) -> anyhow::Error {
        let memory = memory_at_failure
            .map(|memory| {
                format!(
                    "CUDA free={} bytes, total={} bytes",
                    memory.free_bytes, memory.total_bytes
                )
            })
            .unwrap_or_else(|err| format!("CUDA free-memory query failed: {err}"));
        let new_bytes = new_capacity.saturating_mul(self.capacity.bytes_per_token);
        let transient_bytes = old_capacity
            .saturating_add(new_capacity)
            .saturating_mul(self.capacity.bytes_per_token);
        anyhow::anyhow!(
            "CUDA KV capacity growth failed while growing from {old_capacity} to {new_capacity} tokens: {error}. \
             The attempted new KV allocation is approximately {new_bytes} bytes and the transient peak is approximately {transient_bytes} bytes because growth keeps the old bucket live until the new bucket and valid-prefix copy are complete. \
             {memory}; KV bytes/token: {}. The session state was left unchanged; reset or retry with a shorter prompt/max_new_tokens, set ONNX_GENAI_CUDA_KV_MAX_LEN/load_with_cuda_kv_max_len to an explicit smaller cap, or free VRAM used by other processes.",
            self.capacity.bytes_per_token
        )
    }

    /// True when this session uses the seq-major fixed full-context stride path:
    /// VMM commit-on-demand backing *and* a seq-major (BSNH) KV layout. On this
    /// path the KV bindings report a physical shape pinned at the hard-max
    /// context length while only the reached token stripes are committed, so
    /// bucket growth changes no captured-graph dependency and the decode graph is
    /// kept across growth. Head-major and the legacy realloc path return false.
    fn seq_major_fixed_stride(&self) -> bool {
        self.kv_commits_on_demand && self.kv_layout.is_seq_major()
    }

    /// A conservative signature of every binding's captured-graph dependencies
    /// that KV growth could plausibly disturb: the device pointer (baked into
    /// recorded graph nodes) and the reported physical shape (baked into kernel
    /// argument extents / strides). If any entry differs across a growth commit,
    /// the captured graph's assumptions changed and we must invalidate — this is
    /// the defense-in-depth check behind the "provably unchanged -> keep" rule.
    fn binding_growth_signature(&self) -> Vec<(usize, Vec<usize>)> {
        self.bindings
            .iter()
            .map(|binding| {
                (
                    binding.device_ptr() as usize,
                    binding.physical_shape().to_vec(),
                )
            })
            .collect()
    }

    /// Per-binding [`KvBindingGeometry`] for a rank-4 BNSH KV binding
    /// `[batch, kv_heads, capacity, head_dim]`, so the shared
    /// [`kv_commit`] geometry (not a duplicated inline formula) decides the
    /// committed byte ranges. Returns the geometry and the full-context
    /// `capacity` (grow-axis stride) the layout is committed against.
    fn kv_binding_geometry(
        &self,
        binding: &DeviceIoBinding,
    ) -> anyhow::Result<(KvBindingGeometry, usize)> {
        let shape = binding.physical_shape();
        if shape.len() != 4 {
            bail!(
                "VMM-backed CUDA KV binding '{}' must be rank 4, got {:?}",
                binding.input_name(),
                shape
            );
        }
        let elem_bytes = binding.dtype.checked_storage_bytes(1).with_context(|| {
            format!(
                "VMM-backed CUDA KV '{}' has unsized dtype {:?}",
                binding.input_name(),
                binding.dtype
            )
        })?;
        let geometry = KvBindingGeometry {
            kv_heads: shape[1],
            head_dim: shape[3],
            elem_bytes,
        };
        Ok((geometry, shape[2]))
    }

    /// KV-only commit requests for the seq-major fixed path: commit the dense
    /// live-prefix ranges [`kv_commit::live_prefix_ranges`] computes for
    /// [`KvCommitLayout::SeqMajor`] — a single contiguous run
    /// `0..(committed_len × kv_heads × head_dim × elem)` per KV binding, because
    /// seq-major bytes are token-contiguous. This is the *same* geometry unit
    /// the driver-level residency measurement (`vmm_kv_layout_residency_gpu`) and
    /// the `kv_commit.rs` unit tests exercise, so the live commit path and the
    /// measured floor cannot drift apart. The mask is fully committed at
    /// construction and never grows here, so — unlike [`Self::vmm_growth_requests`]
    /// — no mask range is appended. Re-committing the already-mapped prefix is
    /// idempotent; the governor only bills the newly mapped granules.
    fn seq_major_kv_commit_requests(
        &self,
        committed_len: usize,
    ) -> anyhow::Result<Vec<(usize, usize, usize)>> {
        let mut requested = Vec::<(usize, usize, usize)>::new();
        for index in self.kv_binding_range.clone() {
            let binding = &self.bindings[index];
            let (geometry, capacity) = self.kv_binding_geometry(binding)?;
            let ranges = kv_commit::live_prefix_ranges(
                KvCommitLayout::SeqMajor,
                geometry,
                capacity,
                committed_len,
            )
            .with_context(|| {
                format!(
                    "seq-major VMM-backed CUDA KV '{}' dense-prefix commit ranges overflow \
                     (capacity {capacity}, committed_len {committed_len})",
                    binding.input_name()
                )
            })?;
            for range in ranges {
                requested.push((index, range.start, range.end - range.start));
            }
        }
        Ok(requested)
    }

    fn seq_major_commit_mapped_bytes(&self, committed_len: usize) -> anyhow::Result<u64> {
        let requested = self.seq_major_kv_commit_requests(committed_len)?;
        let ranges = requested
            .iter()
            .map(|&(index, offset, bytes)| (&self.bindings[index], offset, bytes))
            .collect::<Vec<_>>();
        self.bindings[0]
            .mapped_bytes_for_binding_ranges(&ranges)
            .context("size seq-major native CUDA KV commit transaction")
    }

    fn commit_seq_major_kv(
        &mut self,
        committed_len: usize,
        grant: Option<&mut onnx_runtime_memory_governor::MappedGrowthGrant>,
    ) -> anyhow::Result<u64> {
        let requested = self.seq_major_kv_commit_requests(committed_len)?;
        let ranges = requested
            .iter()
            .map(|&(index, offset, bytes)| (&self.bindings[index], offset, bytes))
            .collect::<Vec<_>>();
        match grant {
            Some(grant) => {
                self.bindings[0].commit_binding_ranges_with_mapped_growth(&ranges, grant)
            }
            None => {
                self.bindings[0].commit_binding_ranges(&ranges)?;
                Ok(0)
            }
        }
        .context("commit seq-major native CUDA KV binding ranges atomically")
    }

    /// Zero the newly committed dense token stripe `[old_committed, new_committed)`
    /// on every KV binding. Not strictly required for correctness (the kernel
    /// reads only `[0, total_lengths)`), but it keeps the padding suffix zeroed
    /// exactly as the growing-bucket path does, guarding against any future
    /// over-read touching uninitialized physical pages.
    fn zero_seq_major_committed_tail(
        &self,
        old_committed: usize,
        new_committed: usize,
    ) -> anyhow::Result<()> {
        for index in self.kv_binding_range.clone() {
            let binding = &self.bindings[index];
            let mut shape = binding.physical_shape().to_vec();
            shape[2] = old_committed;
            let old_bytes = checked_shape_bytes(&shape, binding.dtype)
                .context("seq-major committed-tail old size overflow")?;
            shape[2] = new_committed;
            let new_bytes = checked_shape_bytes(&shape, binding.dtype)
                .context("seq-major committed-tail new size overflow")?;
            let ptr = binding.device_ptr() as usize;
            native_cuda_memset_zero(ptr + old_bytes, new_bytes - old_bytes)?;
        }
        Ok(())
    }

    /// Record and surface a KV-growth capture decision (kept vs invalidated) with
    /// the named rationale, so an operator can attribute why a captured graph
    /// survived or was discarded across a growth. A silent "kept" is as dangerous
    /// as a silent capture decline, so every keep is both counted and logged.
    fn record_growth_decision(&mut self, kept: bool, reason: &str) {
        if kept {
            self.graph_growth_keeps = self.graph_growth_keeps.saturating_add(1);
        }
        let verb = if kept { "kept" } else { "invalidated" };
        tracing::info!(
            kept,
            reason,
            growth_keeps = self.graph_growth_keeps,
            invalidations = self.graph_invalidations,
            "native CUDA decode graph {verb} across KV growth"
        );
        self.graph_growth_decision = Some(format!("{verb}: {reason}"));
    }

    /// Grow the seq-major fixed full-context stride KV cache by committing more
    /// token stripes on demand, keeping the captured decode graph alive.
    ///
    /// The reported physical shape stays pinned at the hard maximum and every
    /// device pointer stays fixed, so none of the captured graph's baked
    /// dependencies change; we verify that invariant (device pointers + physical
    /// shapes) before deciding to keep, and invalidate if anything moved. This is
    /// the "provably unchanged -> keep" branch: seq-major addressing at batch
    /// index 0 is capacity-independent (proven byte-identical with 0 bytes moved
    /// by #805), so the graph replays correctly into the newly committed pages.
    fn ensure_capacity_seq_major_fixed(
        &mut self,
        session: &mut InferenceSession,
        required: usize,
    ) -> anyhow::Result<bool> {
        if required > self.capacity.max_len {
            bail!("{}", self.capacity_exceeded_error(required));
        }
        if required <= self.kv_committed_len {
            return Ok(false);
        }
        let old_committed = self.kv_committed_len;
        let new_committed = onnx_genai_kv::kv_capacity_bucket(required, self.capacity.max_len);

        // Defense in depth: snapshot the dependency signature before mutating any
        // mapping so we can prove nothing the captured graph baked in moved.
        let before = self.binding_growth_signature();

        let mapped_growth_bytes = self.seq_major_commit_mapped_bytes(new_committed)?;
        let mut grant = session
            .prepare_mapped_growth(
                mapped_growth_bytes,
                onnx_runtime_memory_governor::MemoryRole::KvCache,
            )
            .context("prepare transactional seq-major native CUDA KV commit")?;
        tracing::info!(
            mapped_growth_bytes,
            grant_prepared = grant.is_some(),
            "prepared seq-major native CUDA KV commit transaction"
        );
        let actual_mapped_bytes = self
            .commit_seq_major_kv(new_committed, grant.as_mut())
            .map_err(|error| {
                let memory = cuda_device_memory_snapshot(session.device_id().index as i32)
                    .map_err(|error| error.to_string());
                self.growth_failed_error(old_committed, new_committed, error, memory)
            })?;
        if let Some(grant) = grant {
            grant
                .commit_bytes(actual_mapped_bytes)
                .context("commit seq-major native CUDA KV mapped-growth attribution")?;
        }
        native_cuda_device_barrier(session)?;
        self.zero_seq_major_committed_tail(old_committed, new_committed)?;

        // The reported physical shape and device pointers are unchanged by an
        // on-demand commit; verify that before we keep the graph. If anything did
        // move, default to invalidating — a wrong keep would corrupt output.
        let after = self.binding_growth_signature();
        if before == after {
            self.record_growth_decision(
                true,
                "seq-major fixed full-context stride: device pointers and physical shapes unchanged, \
                 batch-0 addressing capacity-independent, mask fully committed",
            );
        } else {
            self.invalidate_graph(session)?;
            self.record_growth_decision(
                false,
                "seq-major commit unexpectedly changed a binding device pointer or physical shape",
            );
        }

        self.kv_committed_len = new_committed;
        self.kv_growth_events += 1;
        // Seq-major growth moves no KV bytes — the valid prefix keeps its offsets.
        tracing::info!(
            old_committed_len = old_committed,
            new_committed_len = new_committed,
            hard_max_len = self.capacity.max_len,
            "committed seq-major native CUDA KV stripe on demand (fixed stride)"
        );
        Ok(true)
    }

    pub(super) fn ensure_capacity(
        &mut self,
        session: &mut InferenceSession,
        required: usize,
    ) -> anyhow::Result<bool> {
        if self.seq_major_fixed_stride() {
            return self.ensure_capacity_seq_major_fixed(session, required);
        }
        if self.kv_commits_on_demand {
            if required > self.capacity.max_len {
                bail!("{}", self.capacity_exceeded_error(required));
            }
            if required <= self.max_len {
                return Ok(false);
            }
            let old_capacity = self.max_len;
            let new_capacity = onnx_genai_kv::kv_capacity_bucket(required, self.capacity.max_len);
            let valid_len = self.logical_len;
            let mapped_growth_bytes = self.vmm_growth_mapped_bytes(new_capacity)?;
            let mut grant = session
                .prepare_mapped_growth(
                    mapped_growth_bytes,
                    onnx_runtime_memory_governor::MemoryRole::KvCache,
                )
                .context("prepare transactional native CUDA KV growth")?;
            tracing::info!(
                mapped_growth_bytes,
                grant_prepared = grant.is_some(),
                "prepared native CUDA KV mapped-growth transaction"
            );
            let actual_mapped_bytes = self
                .commit_vmm_growth(new_capacity, grant.as_mut())
                .map_err(|error| {
                    let memory = cuda_device_memory_snapshot(session.device_id().index as i32)
                        .map_err(|error| error.to_string());
                    self.growth_failed_error(old_capacity, new_capacity, error, memory)
                })?;
            if let Some(grant) = grant {
                grant
                    .commit_bytes(actual_mapped_bytes)
                    .context("commit native CUDA KV mapped-growth attribution")?;
            }
            native_cuda_device_barrier(session)?;
            self.apply_vmm_growth(new_capacity, valid_len)?;
            self.invalidate_graph(session)?;
            self.max_len = new_capacity;
            self.kv_growth_events += 1;
            // Seq-major grows in place on a fixed per-token stride, so the valid
            // prefix keeps its byte offsets and no KV data is copied; head-major
            // re-strides every head stripe, moving `valid_len × bytes_per_token`.
            let moved_bytes_per_token = if self.kv_layout.is_seq_major() {
                0
            } else {
                self.capacity.bytes_per_token as u64
            };
            self.kv_growth_d2d_copy_bytes = self
                .kv_growth_d2d_copy_bytes
                .saturating_add((valid_len as u64).saturating_mul(moved_bytes_per_token));
            tracing::info!(
                old_len = old_capacity,
                new_len = new_capacity,
                valid_len,
                hard_max_len = self.capacity.max_len,
                "grew VMM-backed native CUDA KV capacity bucket in place"
            );
            return Ok(true);
        }
        let mut backend = NativeCudaCapacityBackend {
            state: self,
            session,
        };
        match onnx_genai_kv::ensure_kv_capacity(&mut backend, required)? {
            onnx_genai_kv::KvCapacityGrowth::Unchanged => Ok(false),
            onnx_genai_kv::KvCapacityGrowth::Grew {
                old_capacity,
                new_capacity,
                valid_len,
            } => {
                backend.state.kv_growth_events += 1;
                backend.state.kv_growth_d2d_copy_bytes =
                    backend.state.kv_growth_d2d_copy_bytes.saturating_add(
                        (valid_len as u64)
                            .saturating_mul(backend.state.capacity.bytes_per_token as u64),
                    );
                tracing::info!(
                    old_len = old_capacity,
                    new_len = new_capacity,
                    valid_len,
                    hard_max_len = backend.state.capacity.max_len,
                    "grew native CUDA KV capacity bucket"
                );
                Ok(true)
            }
        }
    }

    /// Commit every byte the next VMM-backed bucket can touch before mutating
    /// any live KV.
    ///
    /// This keeps growth transactional with respect to model state: a tight
    /// card may refuse the new granule, but then all layer bindings still expose
    /// the old shapes and hold the old layout. Extra committed granules are
    /// only installed after their lease succeeds, so the governor remains the
    /// single source of memory truth.
    fn commit_vmm_growth(
        &mut self,
        new_capacity: usize,
        grant: Option<&mut onnx_runtime_memory_governor::MappedGrowthGrant>,
    ) -> anyhow::Result<u64> {
        let requested = self.vmm_growth_requests(new_capacity)?;
        let ranges = requested
            .iter()
            .map(|&(index, offset, bytes)| (&self.bindings[index], offset, bytes))
            .collect::<Vec<_>>();
        match grant {
            Some(grant) => {
                self.bindings[0].commit_binding_ranges_with_mapped_growth(&ranges, grant)
            }
            None => {
                self.bindings[0].commit_binding_ranges(&ranges)?;
                Ok(0)
            }
        }
        .context("commit native CUDA KV binding ranges atomically")
    }

    fn vmm_growth_mapped_bytes(&self, new_capacity: usize) -> anyhow::Result<u64> {
        let requested = self.vmm_growth_requests(new_capacity)?;
        let ranges = requested
            .iter()
            .map(|&(index, offset, bytes)| (&self.bindings[index], offset, bytes))
            .collect::<Vec<_>>();
        self.bindings[0]
            .mapped_bytes_for_binding_ranges(&ranges)
            .context("size native CUDA KV mapped-growth transaction")
    }

    /// Growth commit requests for the **head-major** (and non-fixed-stride)
    /// VMM path: a single flat range `0..(new_capacity × kv_heads × head_dim ×
    /// elem)` per KV binding, plus the mask island.
    ///
    /// This is deliberately left as the flat **bucket** range rather than routed
    /// through [`kv_commit::live_prefix_ranges`], and it is byte-identical to the
    /// dense head-major geometry here: head-major grows a *packed* bucket whose
    /// per-head stride is `new_capacity` (the bucket, not a fixed full context),
    /// so each head's stripe is contiguous with the next and the `kv_heads`
    /// live-prefix fragments `live_prefix_ranges` would return tile the same
    /// `[0, bucket_bytes)` run — they touch exactly the same physical granules.
    /// The `kv_heads×` scatter that separates head-major from seq-major only
    /// appears on a *fixed full-context* stride (where a head stripe is
    /// `max_len × head_dim × elem` apart, crossing a granule once
    /// `capacity × head_dim × elem ≥ granule`, i.e. ≈8192 tokens at head_dim
    /// 128/fp16, #776); the growing bucket keeps head-major dense, so its dense
    /// ranges equal its bucket ranges and it stays byte-identical (see
    /// `docs/MEMORY_ARCHITECTURE.md`, "KV layout and residency"). Seq-major
    /// instead reports a fixed full-context stride and commits its dense prefix
    /// through [`Self::seq_major_kv_commit_requests`].
    fn vmm_growth_requests(
        &self,
        new_capacity: usize,
    ) -> anyhow::Result<Vec<(usize, usize, usize)>> {
        let mut requested = Vec::<(usize, usize, usize)>::new();
        for index in self.kv_binding_range.clone() {
            let binding = &self.bindings[index];
            let mut new_shape = binding.physical_shape().to_vec();
            if new_shape.len() != 4 {
                bail!(
                    "VMM-backed CUDA KV binding '{}' must be rank 4, got {:?}",
                    binding.input_name(),
                    new_shape
                );
            }
            new_shape[2] = new_capacity;
            let bytes = checked_shape_bytes(&new_shape, binding.dtype).with_context(|| {
                format!(
                    "VMM-backed CUDA KV '{}' growth size overflows for shape {:?}",
                    binding.input_name(),
                    new_shape
                )
            })?;
            requested.push((index, 0, bytes));
        }
        let mask_bytes = new_capacity
            .checked_mul(std::mem::size_of::<i64>())
            .with_context(|| {
                format!("VMM-backed CUDA mask growth overflows for capacity {new_capacity}")
            })?;
        requested.push((0, 0, mask_bytes));
        Ok(requested)
    }

    /// Grow VMM-backed KV in place, keeping the reserved address range stable.
    ///
    /// The allocation is sized for full context at construction, but kernels see
    /// only the current bucket as the physical shape. On growth we commit the
    /// larger bucket, move each head's valid prefix from the old stride to the
    /// new stride, and zero the newly visible suffix. This prevents the failure
    /// where a "reserved" KV cache secretly commits the full context at load,
    /// while preserving the fixed-stride padded shape CUDA graph capture expects
    /// inside one bucket.
    fn apply_vmm_growth(&mut self, new_capacity: usize, valid_len: usize) -> anyhow::Result<()> {
        // All fallible commits have already succeeded, so the expected
        // tight-card failure path is transactional. A later CUDA copy/memset
        // failure can still leave mixed old/new strides; #699 tracks adding a
        // poison/reset path rather than hiding that rarer device failure here.
        let layout = self.kv_layout;
        for binding in &mut self.bindings[self.kv_binding_range.clone()] {
            let old_shape = binding.physical_shape().to_vec();
            if old_shape.len() != 4 {
                bail!(
                    "VMM-backed CUDA KV binding '{}' must be rank 4, got {:?}",
                    binding.input_name(),
                    old_shape
                );
            }
            let mut new_shape = old_shape.clone();
            new_shape[2] = new_capacity;
            let elem = binding.dtype.checked_storage_bytes(1).with_context(|| {
                format!(
                    "VMM-backed CUDA KV '{}' has unsized dtype {:?}",
                    binding.input_name(),
                    binding.dtype
                )
            })?;
            let ptr = binding.device_ptr() as usize;
            // Move/zero over the *physical* byte layout. Head-major re-strides
            // each head stripe on axis 2 (unchanged); seq-major grows on axis 1
            // with a capacity-independent per-token stride, so the in-place copy
            // is a no-op and only the newly grown tail is zeroed — no KV moves.
            let (old_bytes, grow_axis) = kv_growth_byte_layout(&old_shape, layout)?;
            let (new_bytes, _) = kv_growth_byte_layout(&new_shape, layout)?;
            copy_kv_prefix_device_to_device_in_place(
                ptr, &old_bytes, &new_bytes, grow_axis, valid_len, elem,
            )?;
            zero_kv_suffix_device(ptr, &new_bytes, grow_axis, valid_len, elem)?;
            let mut logical_shape = new_shape.clone();
            logical_shape[2] = valid_len;
            binding.set_physical_and_logical_shapes(new_shape, logical_shape)?;
        }
        self.grow_vmm_mask_in_place(new_capacity, valid_len)?;
        Ok(())
    }

    fn initial_vmm_kv_committed_range(
        physical_shape: &[usize],
        dtype: DataType,
    ) -> anyhow::Result<std::ops::Range<usize>> {
        let bytes = checked_shape_bytes(physical_shape, dtype)
            .context("initial VMM-backed CUDA KV bucket size overflow")?;
        Ok(0..bytes)
    }

    fn full_vmm_kv_allocation_bytes(
        physical_shape: &[usize],
        dtype: DataType,
        max_len: usize,
    ) -> anyhow::Result<usize> {
        let mut full_shape = physical_shape.to_vec();
        full_shape[2] = max_len;
        checked_shape_bytes(&full_shape, dtype)
            .context("full VMM-backed CUDA KV reservation size overflow")
    }

    fn grow_vmm_mask_in_place(
        &mut self,
        new_capacity: usize,
        valid_len: usize,
    ) -> anyhow::Result<()> {
        let bytes = new_capacity
            .checked_mul(std::mem::size_of::<i64>())
            .with_context(|| {
                format!("VMM-backed CUDA mask growth overflows for capacity {new_capacity}")
            })?;
        native_cuda_memset_zero(self.bindings[0].device_ptr() as usize, bytes)?;
        self.bindings[0]
            .set_physical_and_logical_shapes(vec![1, new_capacity], vec![1, valid_len])?;
        Ok(())
    }

    /// Collect the symbolic dimension ids that the native decoder structurally
    /// pins to `1` at decode time. Batch (axis 0 of every input) and query-seq
    /// (the remaining `input_ids` / `position_ids` axes, which are bound to a
    /// single token) are the only symbols that [`persistent_output_shape`] may
    /// safely collapse to `1`. `batch_only` restricts collection to axis 0 for
    /// inputs whose non-batch axes grow with the sequence (attention_mask and
    /// the past-KV tensors, whose total_seq axis is *not* a decode unit).
    pub(crate) fn collect_unit_symbols(
        shape: &[Dim],
        batch_only: bool,
        out: &mut HashSet<SymbolId>,
    ) {
        for (axis, dim) in shape.iter().enumerate() {
            if batch_only && axis != 0 {
                continue;
            }
            if let Dim::Symbolic(symbol) = dim {
                out.insert(*symbol);
            }
        }
    }

    /// First structurally-unresolved symbolic axis of an auxiliary output: a
    /// `Dim::Symbolic` that is *not* one of the decode-unit (batch / query-seq)
    /// symbols. Such a dimension is data-dependent (e.g. an accumulator indexed
    /// by total_seq / past+1), so collapsing it to `1` in a persistent device
    /// binding would under-allocate. Returns `(axis, symbol)` of the offender.
    pub(crate) fn unresolved_symbolic_axis(
        shape: &[Dim],
        unit_symbols: &HashSet<SymbolId>,
    ) -> Option<(usize, SymbolId)> {
        shape.iter().enumerate().find_map(|(axis, dim)| match dim {
            Dim::Symbolic(symbol) if !unit_symbols.contains(symbol) => Some((axis, *symbol)),
            _ => None,
        })
    }

    pub(crate) fn persistent_output_shape(
        name: &str,
        dtype: DataType,
        shape: &[Dim],
    ) -> anyhow::Result<Vec<usize>> {
        if matches!(dtype, DataType::Undefined | DataType::String) {
            bail!(
                "cannot bind auxiliary CUDA graph output '{name}' persistently: dtype {dtype:?} does not have fixed-size device tensor storage, but CUDA graph capture requires every declared graph output to use stable device storage; export this output as a numeric tensor or remove the unused graph output"
            );
        }
        let shape = shape
            .iter()
            .map(|dim| match dim {
                Dim::Static(value) => *value,
                Dim::Symbolic(_) => 1,
            })
            .collect::<Vec<_>>();
        let elements = shape.iter().try_fold(1usize, |product, &dim| {
            product.checked_mul(dim).with_context(|| {
                format!(
                    "cannot bind auxiliary CUDA graph output '{name}' persistently: shape {shape:?} overflows the device allocation size; export a bounded output shape or remove the unused graph output"
                )
            })
        })?;
        dtype.checked_storage_bytes(elements).with_context(|| {
            format!(
                "cannot bind auxiliary CUDA graph output '{name}' persistently: dtype {dtype:?} shape {shape:?} has no representable device allocation size; export a fixed-size numeric tensor or remove the unused graph output"
            )
        })?;
        Ok(shape)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        session: &mut InferenceSession,
        io: DecodeCudaIo<'_>,
        present_to_past: &HashMap<String, String>,
        fixed_state_inputs: &HashSet<String>,
        capacity: CudaKvCapacity,
        mut graph_capture: GraphCaptureDecision,
        position_rank: usize,
        kv_layout: KvCommitLayout,
    ) -> anyhow::Result<Self> {
        let kv_commits_on_demand = session.commits_on_demand();
        // Seq-major KV addressing is capacity-independent at batch index 0 (the
        // only index native decode uses): token `t` always lands at
        // `t * kv_heads * head_dim`, and the baked `cache_capacity` scalar only
        // scales the batch term (= 0 here). So a seq-major VMM binding can report
        // a *fixed* full-context physical stride while committing token stripes
        // on demand — growth then changes neither device pointer nor physical
        // shape nor addressing, and the captured decode graph survives it. Head-
        // major uses `cache_capacity` as the per-head stride, so its addressing
        // shifts on growth and it must keep the growing-bucket (re-capture) model.
        // The legacy realloc path (no VMM) also keeps the growing-bucket model.
        let seq_major_fixed = kv_commits_on_demand && kv_layout.is_seq_major();
        let initial_bucket_len = onnx_genai_kv::kv_capacity_bucket(0, capacity.max_len);
        // The capacity reported to the bindings (physical axis-2 / mask island).
        // Seq-major fixed stride pins this at the hard maximum from the start;
        // everything else starts at the initial bucket and grows it.
        let reported_len = if seq_major_fixed {
            capacity.max_len
        } else {
            initial_bucket_len
        };
        let max_len = reported_len;
        let mask_bytes = max_len
            .checked_mul(std::mem::size_of::<i64>())
            .context("initial CUDA mask size overflow")?;
        let full_mask_bytes = capacity
            .max_len
            .checked_mul(std::mem::size_of::<i64>())
            .context("full CUDA mask reservation size overflow")?;
        // Seq-major fixed stride commits the whole (tiny) mask at construction so
        // the mask island is shape-static at the hard max and never grows; every
        // other path commits only the initial mask bucket and grows it in place.
        let mask_committed_bytes = if seq_major_fixed {
            full_mask_bytes
        } else {
            mask_bytes
        };
        let mask = if kv_commits_on_demand {
            let committed = std::iter::once(0..mask_committed_bytes).collect::<Vec<_>>();
            session.allocate_device_binding_committed(
                io.attention_mask,
                None::<String>,
                DataType::Int64,
                vec![1, max_len],
                vec![1, max_len],
                full_mask_bytes,
                committed,
            )?
        } else {
            session.allocate_device_binding(
                io.attention_mask,
                None::<String>,
                DataType::Int64,
                vec![1, max_len],
                vec![1, max_len],
            )?
        };
        native_cuda_memset_zero(mask.device_ptr() as usize, mask_committed_bytes)?;

        let mut pairs = present_to_past
            .iter()
            .map(|(present, past)| (present.clone(), past.clone()))
            .collect::<Vec<_>>();
        pairs.sort_unstable_by(|left, right| left.1.cmp(&right.1));
        pairs.sort_by_key(|(_, past)| fixed_state_inputs.contains(past));
        let mut bindings = Vec::with_capacity(4 + pairs.len());
        bindings.push(mask);
        let kv_start = bindings.len();
        for (present, past) in pairs
            .iter()
            .filter(|(_, past)| !fixed_state_inputs.contains(past))
        {
            let meta = session
                .inputs()
                .iter()
                .find(|meta| meta.name == *past)
                .with_context(|| format!("missing CUDA KV input metadata for '{past}'"))?;
            let (physical_shape, logical_shape) =
                Self::persistent_state_shapes(past, meta.dtype, &meta.shape, max_len, false)?;
            let binding = if kv_commits_on_demand {
                let allocation_bytes = Self::full_vmm_kv_allocation_bytes(
                    &physical_shape,
                    meta.dtype,
                    capacity.max_len,
                )
                .with_context(|| {
                    format!("sizing full VMM-backed CUDA KV reservation for '{past}'")
                })?;
                let initial_range = if seq_major_fixed {
                    // Physical shape is pinned at the hard max, but only the
                    // initial dense token prefix is mapped; growth commits more
                    // stripes without ever changing the reported shape.
                    let mut initial_shape = physical_shape.clone();
                    initial_shape[2] = initial_bucket_len;
                    Self::initial_vmm_kv_committed_range(&initial_shape, meta.dtype).with_context(
                        || format!("sizing initial VMM-backed CUDA KV bucket for '{past}'"),
                    )?
                } else {
                    Self::initial_vmm_kv_committed_range(&physical_shape, meta.dtype).with_context(
                        || format!("sizing initial VMM-backed CUDA KV bucket for '{past}'"),
                    )?
                };
                session.allocate_device_binding_committed(
                    past.clone(),
                    Some(present.clone()),
                    meta.dtype,
                    physical_shape,
                    logical_shape,
                    allocation_bytes,
                    vec![initial_range],
                )?
            } else {
                session.allocate_device_binding(
                    past.clone(),
                    Some(present.clone()),
                    meta.dtype,
                    physical_shape,
                    logical_shape,
                )?
            };
            bindings.push(binding);
        }
        let kv_end = bindings.len();
        if kv_commits_on_demand {
            for binding in &mut bindings[kv_start..kv_end] {
                // Zero only the committed region. For seq-major fixed stride the
                // physical shape is the hard max but only `initial_bucket_len`
                // tokens are mapped, so zeroing the full shape would touch
                // uncommitted pages and fault.
                let mut committed_shape = binding.physical_shape().to_vec();
                if seq_major_fixed {
                    committed_shape[2] = initial_bucket_len;
                }
                let bytes =
                    checked_shape_bytes(&committed_shape, binding.dtype).with_context(|| {
                        format!(
                            "initial VMM-backed CUDA KV '{}' bucket size overflows for shape {:?}",
                            binding.input_name(),
                            committed_shape
                        )
                    })?;
                native_cuda_memset_zero(binding.device_ptr() as usize, bytes)?;
            }
        }
        for (present, past) in pairs
            .iter()
            .filter(|(_, past)| fixed_state_inputs.contains(past))
        {
            let meta = session
                .inputs()
                .iter()
                .find(|meta| meta.name == *past)
                .with_context(|| format!("missing fixed CUDA state input metadata for '{past}'"))?;
            let (physical_shape, logical_shape) =
                Self::persistent_state_shapes(past, meta.dtype, &meta.shape, max_len, true)?;
            let binding = session.allocate_device_binding(
                past.clone(),
                Some(present.clone()),
                meta.dtype,
                physical_shape.clone(),
                logical_shape,
            )?;
            native_cuda_memset_zero(
                binding.device_ptr() as usize,
                checked_shape_bytes(&physical_shape, meta.dtype).with_context(|| {
                    format!(
                        "fixed CUDA decoder state allocation size overflow for '{past}' shape {physical_shape:?}"
                    )
                })?,
            )?;
            bindings.push(binding);
        }
        let fixed_state_end = bindings.len();

        let logits_meta = session
            .outputs()
            .iter()
            .find(|meta| meta.name == io.logits)
            .with_context(|| format!("missing CUDA logits output metadata for '{}'", io.logits))?;
        if !matches!(
            logits_meta.dtype,
            DataType::Float32 | DataType::Float16 | DataType::BFloat16
        ) || logits_meta.shape.is_empty()
        {
            bail!(
                "CUDA logits output '{}' must be non-scalar f32, f16 or bf16, got {:?} {:?}",
                io.logits,
                logits_meta.dtype,
                logits_meta.shape
            );
        }
        let logits_dtype = logits_meta.dtype;
        let logits_shape =
            Self::persistent_output_shape(io.logits, logits_dtype, &logits_meta.shape)?;
        let logits_device_binding = session.allocate_device_output_binding(
            io.logits,
            logits_dtype,
            logits_shape.clone(),
            logits_shape.clone(),
        )?;

        let present_outputs = present_to_past.keys().cloned().collect::<HashSet<_>>();
        let auxiliary_meta = session
            .outputs()
            .iter()
            .filter(|meta| meta.name != io.logits && !present_outputs.contains(&meta.name))
            .cloned()
            .collect::<Vec<_>>();

        // Structural safe-collapse analysis for auxiliary outputs. The native
        // decoder pins batch and query-seq to `1` at decode, so a symbolic aux
        // dimension is only safe to collapse to `1` when it is one of those
        // structurally-unit axes. Gather every symbol the decoder binds to `1`:
        // input_ids / position_ids (bound to `[1, 1]`) on all axes, plus the
        // batch axis (axis 0) of attention_mask and each past-KV input. Any
        // other symbolic aux dim (e.g. one indexed by total_seq / past+1) is
        // data-dependent and must not be collapsed. See RULES.md §2 — this is a
        // purely structural signal, never a model-name gate.
        let mut unit_symbols: HashSet<SymbolId> = HashSet::new();
        let sequence_input_name = io
            .inputs_embeds
            .as_ref()
            .map(|embeds| embeds.name)
            .unwrap_or(io.input_ids);
        if let Some(meta) = session
            .inputs()
            .iter()
            .find(|meta| meta.name == sequence_input_name)
        {
            Self::collect_unit_symbols(&meta.shape, false, &mut unit_symbols);
        }
        if let Some(position_ids) = io.position_ids
            && let Some(meta) = session
                .inputs()
                .iter()
                .find(|meta| meta.name == position_ids)
        {
            Self::collect_unit_symbols(&meta.shape, false, &mut unit_symbols);
        }
        if let Some(meta) = session
            .inputs()
            .iter()
            .find(|meta| meta.name == io.attention_mask)
        {
            Self::collect_unit_symbols(&meta.shape, true, &mut unit_symbols);
        }
        for past in present_to_past.values() {
            if let Some(meta) = session.inputs().iter().find(|meta| &meta.name == past) {
                Self::collect_unit_symbols(&meta.shape, true, &mut unit_symbols);
            }
        }

        let auxiliary_start = bindings.len();
        let mut declined_auxiliary: Vec<String> = Vec::new();
        for meta in auxiliary_meta {
            if let Some((axis, symbol)) = Self::unresolved_symbolic_axis(&meta.shape, &unit_symbols)
            {
                // The output's extent on this axis is data-dependent and not
                // structurally identifiable as batch or query-seq. Collapsing
                // it to `1` (as a persistent device binding requires) would
                // under-allocate, so we deliberately do NOT bind it. The eager
                // executor JIT-sizes and materializes this output every step,
                // so decode still works; only CUDA graph capture is forfeited
                // (capture demands a stable device address for every output).
                let symbol_label = session
                    .graph()
                    .symbol_constraints
                    .get(&symbol)
                    .and_then(|constraints| constraints.name.clone())
                    .unwrap_or_else(|| format!("symbol#{}", symbol.0));
                declined_auxiliary.push(format!(
                    "'{}' (axis {axis} is symbolic dim '{symbol_label}', not structurally batch or query-seq)",
                    meta.name
                ));
                continue;
            }
            let shape = Self::persistent_output_shape(&meta.name, meta.dtype, &meta.shape)?;
            bindings.push(
                session
                    .allocate_device_output_binding(
                        &meta.name,
                        meta.dtype,
                        shape.clone(),
                        shape,
                    )
                    .with_context(|| {
                        format!(
                            "failed to allocate persistent CUDA device binding for auxiliary graph output '{}'; CUDA graph capture requires every declared output to keep a stable device address",
                            meta.name
                        )
                    })?,
            );
        }
        let auxiliary_end = bindings.len();
        let base_binding_count = bindings.len();

        // If any auxiliary output could not be persistently bound, CUDA graph
        // capture is impossible (an unbound output would materialize on the
        // host mid-capture). Decline capture up front, with a clear structural
        // reason, and fall back to the eager device path — which still decodes
        // correctly by dynamically allocating the unbindable output each step.
        if !declined_auxiliary.is_empty() {
            if graph_capture.is_enabled() {
                graph_capture.decline_if_enabled(
                    "auxiliary_outputs_have_fixed_persistent_shapes",
                    format_args!(
                        "auxiliary output(s) {} carry unresolved symbolic dimensions",
                        declined_auxiliary.join(", ")
                    ),
                );
            } else {
                tracing::debug!(
                    "native CUDA decode leaving auxiliary output(s) {} unbound (unresolved symbolic dimensions); eager path allocates them dynamically",
                    declined_auxiliary.join(", ")
                );
            }
        }

        let input_ids_binding = bindings.len();
        if let Some(embeds) = &io.inputs_embeds {
            // Fused VLM decoder (Inc3a): the sequence input is a float
            // `[1, 1, hidden]` embedding, not an `Int64 [1, 1]` token id. This
            // persistent binding keeps the sequence-binding index valid; the
            // eager inputs_embeds decode path binds the per-step embedding as an
            // owned input rather than through this slot, so it is never the
            // captured token-write target.
            bindings.push(session.allocate_device_binding(
                embeds.name,
                None::<String>,
                embeds.dtype,
                vec![1, 1, embeds.hidden],
                vec![1, 1, embeds.hidden],
            )?);
        } else {
            bindings.push(session.allocate_device_binding(
                io.input_ids,
                None::<String>,
                DataType::Int64,
                vec![1, 1],
                vec![1, 1],
            )?);
        }
        let position_ids_binding = if let Some(position_ids) = io.position_ids {
            let index = bindings.len();
            // A rank-N mrope decoder declares `position_ids [N, B, S]`; the single
            // decode step collapses batch/sequence to 1 → `[N, 1, 1]`. Rank 1 is
            // the conventional `[1, 1]`, byte-identical to before.
            let shape = if position_rank == 1 {
                vec![1, 1]
            } else {
                vec![position_rank, 1, 1]
            };
            bindings.push(session.allocate_device_binding(
                position_ids,
                None::<String>,
                DataType::Int64,
                shape.clone(),
                shape,
            )?);
            Some(index)
        } else {
            None
        };
        // Inc3c capture: record the per-step supplied ports that get a persistent
        // device binding — the `inputs_embeds` sequence binding (already allocated
        // above as `input_ids_binding`) plus each routed port (allocated here).
        // The Inc3c capture path writes these each step then replays the captured
        // graph; the default eager path leaves them untouched.
        let mut captured_step_inputs = Vec::with_capacity(io.routed.len() + 1);
        if let Some(embeds) = &io.inputs_embeds {
            captured_step_inputs.push(CapturedStepInputBinding {
                name: embeds.name.to_string(),
                binding_index: input_ids_binding,
                byte_len: embeds.dtype.storage_bytes(embeds.hidden),
            });
        }
        for routed in &io.routed {
            let index = bindings.len();
            let elems: usize = routed.shape.iter().product();
            bindings.push(session.allocate_device_binding(
                routed.name,
                None::<String>,
                routed.dtype,
                routed.shape.clone(),
                routed.shape.clone(),
            )?);
            captured_step_inputs.push(CapturedStepInputBinding {
                name: routed.name.to_string(),
                binding_index: index,
                byte_len: routed.dtype.storage_bytes(elems),
            });
        }
        let logits_binding = bindings.len();
        bindings.push(logits_device_binding);

        #[cfg(feature = "cuda")]
        let argmax_words = {
            let vocab = *logits_shape
                .last()
                .context("CUDA logits shape has no vocabulary dimension")?;
            2 + onnx_runtime_ep_cuda::device_argmax_scratch_words(vocab)
        };
        #[cfg(not(feature = "cuda"))]
        let argmax_words = 2;
        let greedy_result = session.allocate_device_output_binding(
            "__native_greedy_argmax",
            DataType::Uint32,
            vec![argmax_words],
            vec![2],
        )?;

        // A graph records launch geometry, so replay is unsafe when a persistent
        // binding exposes a growing logical prefix instead of fixed capacity.
        // Surfacing *which* bindings force the eager fallback is essential for
        // capture bring-up of new architectures, so the decision below retains
        // their names for the session warning and debug/profile statistics.
        let dynamic_logical: Vec<String> = bindings
            .iter()
            .filter(|binding| binding.has_dynamic_logical_input_shape())
            .map(|binding| {
                format!(
                    "{} (physical {:?} vs logical {:?})",
                    binding.input_name(),
                    binding.physical_shape(),
                    binding.logical_shape()
                )
            })
            .collect();
        if graph_capture.is_enabled() && !dynamic_logical.is_empty() {
            graph_capture.decline_if_enabled(
                "persistent_inputs_have_fixed_logical_shapes",
                format_args!(
                    "input binding(s) {} expose a growing logical prefix instead of fixed capacity",
                    dynamic_logical.join(", ")
                ),
            );
        }
        // The attention-mask binding (bindings[0]) is allocated with the
        // consumer-scoped capacity policy: it exposes its logical valid length
        // whenever any consumer is not a padded-capacity-safe kernel (Shape /
        // ReduceSum). Such a mask cannot be frozen to physical `max_len` during
        // single-token decode — doing so leaks the padded width into that
        // consumer (e.g. GLM-5.2's indexer `Add`, which broadcasts the mask
        // against a logical-width score). At construction the mask's logical and
        // physical shapes are still equal (`max_len`), so it is not yet caught by
        // the `has_dynamic_logical_input_shape` scan above; recognise it here from
        // the static policy so decode drives it at the growing logical length and,
        // like any growing logical input, forfeits CUDA-graph capture.
        let mask_exposes_logical = bindings
            .first()
            .is_some_and(DeviceIoBinding::exposes_logical_input_shape);
        if graph_capture.is_enabled() && mask_exposes_logical {
            graph_capture.decline_if_enabled(
                "attention_mask_consumers_are_capacity_aware",
                format_args!(
                    "attention-mask binding '{}' exposes its logical valid length to a non-capacity-aware consumer",
                    bindings[0].input_name()
                ),
            );
        }
        let graph_enabled = graph_capture.is_enabled();
        let graph_decline_reason = graph_capture.into_decline_reason();
        if let Some(reason) = &graph_decline_reason {
            tracing::warn!(
                "native CUDA decode graph capture disabled: {reason}; decode continues eagerly"
            );
        }

        // Inc3c: enable the captured per-step-input path when (a) graph capture
        // is structurally available (`graph_enabled`), (b) the decoder actually
        // has per-step supplied ports (embeds/routed), and (c) the opt-out env
        // `ONNX_GENAI_NATIVE_DECODER_CAPTURE_STEP_INPUTS` is not falsy. This is
        // now **default-on**: a capture-eligible multi-component decoder reuses
        // the `run_one_token` graph instead of the eager owned uploads. The
        // structural gates (a)/(b) still auto-decline ineligible decoders (a
        // growing-logical binding or a mask exposed to a non-capacity-aware
        // consumer clears `graph_enabled`), so default-on never captures a wrong
        // decoder; the eager owned path stays as the byte-identical fallback.
        let capture_step_inputs =
            graph_enabled && !captured_step_inputs.is_empty() && capture_step_inputs_enabled();

        // Pin the fixed-capacity KV sequence-axis symbols CONSTANT so CUDA-graph
        // capture ADMITS the GroupQueryAttention (and capacity-Attention) nodes
        // instead of vetoing each per-layer node as a growing-seq eager seam. The
        // runtime has just bound fixed-capacity, device-resident KV at physical
        // `[.., max_len, ..]` with the valid attended length read on-device, so
        // the attention kernels' launch grids are capacity-sized (constant within
        // a capture) and a captured replay is shape-static. Gated on
        // `graph_enabled`: a growing/paged KV decoder clears it and never pins,
        // preserving the classifier's growing-seq veto for those paths. KV growth
        // / capacity-bucket rebucket invalidates and re-captures the graph, so a
        // pinned symbol is never replayed against a stale grid.
        if graph_enabled {
            let pinned = session.pin_fixed_capacity_kv_capture_symbols();
            if pinned > 0 {
                tracing::debug!(
                    max_len,
                    pinned,
                    "native CUDA decode: pinned fixed-capacity KV seq symbol(s) for graph capture"
                );
            }
        }

        Ok(Self {
            logical_len: 0,
            max_len,
            bindings,
            base_binding_count,
            kv_binding_range: kv_start..kv_end,
            fixed_state_binding_range: kv_end..fixed_state_end,
            auxiliary_binding_range: auxiliary_start..auxiliary_end,
            input_ids_binding,
            position_ids_binding,
            position_rank,
            captured_step_inputs,
            capture_step_inputs,
            logits_binding,
            logits_shape,
            logits_dtype,
            greedy_result,
            graph_enabled,
            mask_exposes_logical,
            graph_phase: DecodeCudaGraphPhase::NeedsWarmup,
            inline_graph_phase: DecodeCudaGraphPhase::NeedsWarmup,
            graph_captures: 0,
            graph_replays: 0,
            graph_fallbacks: 0,
            graph_invalidations: 0,
            kv_growth_events: 0,
            kv_growth_d2d_copy_bytes: 0,
            graph_growth_keeps: 0,
            // Seq-major fixed stride starts with only the initial dense token
            // prefix mapped; every other path reports its physical capacity as
            // fully committed (head-major grows the bucket, legacy reallocates).
            kv_committed_len: if seq_major_fixed {
                initial_bucket_len
            } else {
                reported_len
            },
            graph_growth_decision: None,
            capacity,
            kv_commits_on_demand,
            kv_layout,
            graph_decline_reason,
            graph_fallback_reason: None,
            graph_fallback_report: None,
            auxiliary_bind_declines: declined_auxiliary,
            retain_graph_on_rewind: false,
            #[cfg(test)]
            padded_query_capacity: None,
        })
    }

    /// Write the valid `1`s for keys `[start, end)` and set the mask's exposed
    /// logical length. `expose_len` is the last-dim extent the graph's mask
    /// island (and hence the Attention kernel) sees: for a single-token decode
    /// step it is the current physical bucket (`max_len`), which freezes the
    /// island to a shape-static `[1,1,1,max_len]` additive bias (correct for a
    /// single query row — verified: the padding suffix maps to `-inf`, the valid
    /// prefix to `0`). Bucket growth invalidates and re-captures the graph at the
    /// new bucket, exactly like ORT shared-buffer KV, so capture eligibility does
    /// not require a session-wide fixed capacity. Multi-token prefill passes
    /// `end` (the growing valid length) because the causal island is
    /// prefix-sensitive for `q_seq > 1` and must see the exact logical length.
    /// The last-dim extent to expose for the attention mask on a single-token
    /// decode step. Frozen to the current physical bucket (`max_len`) so the step
    /// stays CUDA-graph-capture eligible — *unless* the mask binding exposes its
    /// logical valid length to a non-capacity-aware consumer
    /// (`mask_exposes_logical`), in which case the true valid length (`total_len`)
    /// must be used or the padded width leaks into that consumer's arithmetic
    /// (see [`Self::mask_exposes_logical`]).
    fn decode_mask_expose_len(&self, total_len: usize) -> usize {
        if self.mask_exposes_logical {
            total_len
        } else {
            self.max_len
        }
    }

    fn extend_mask(&mut self, start: usize, end: usize, expose_len: usize) -> anyhow::Result<()> {
        if end > self.max_len || start > end || expose_len > self.max_len || end > expose_len {
            bail!(
                "invalid CUDA mask update {start}..{end} (expose {expose_len}) for capacity {}",
                self.max_len
            );
        }
        let ones = (start..end)
            .flat_map(|_| 1i64.to_le_bytes())
            .collect::<Vec<_>>();
        self.bindings[0].write_bytes(start * std::mem::size_of::<i64>(), &ones)?;
        self.bindings[0].set_logical_shape(vec![1, expose_len])?;
        Ok(())
    }

    pub(crate) fn set_logical_len(&mut self, len: usize) -> anyhow::Result<()> {
        for binding in &mut self.bindings[self.kv_binding_range.clone()] {
            let mut shape = binding.physical_shape().to_vec();
            shape[2] = len;
            binding.set_logical_shape(shape)?;
        }
        self.logical_len = len;
        Ok(())
    }

    /// Whether every self-attention KV binding is a rank-4 CUDA cache
    /// (`[1, num_kv_heads, max_len, head_dim]`) in a dtype the paged store can
    /// round-trip losslessly — `f32` (GAP-3 Inc-D) or `f16` (GAP-3 Inc-D.1). The
    /// host paged store is `f32`; ORT already widens `f16` present-KV to `f32`
    /// when mirroring and narrows back on inject (`kv_bridge::mirror_present_kv_to_pages`
    /// / `load_materialized_past`), and the native read/seed use the same `half`
    /// convert ([`kv_dtype_to_f32`] / [`f32_slice_to_dtype_bytes`]), so the
    /// f16→f32→f16 round-trip is bit-exact and byte-comparable against ORT.
    ///
    /// `bf16`, non-rank-4, and in-place / CPU-resident caches stay gated to the
    /// non-paged fallback (Inc-D.2 and later) — no silent-wrong paged run.
    #[allow(dead_code)]
    pub(crate) fn kv_bindings_paged_rank4(&self) -> bool {
        let range = self.kv_binding_range.clone();
        if range.is_empty() {
            return false;
        }
        self.bindings[range].iter().all(|binding| {
            matches!(binding.dtype, DataType::Float32 | DataType::Float16)
                && binding.physical_shape().len() == 4
        })
    }

    /// Read the most recent step's accumulated present KV out of the device
    /// binding whose past-input port is `past_name`, as a host f32 buffer plus
    /// the shape whose row-major strides address it (GAP-3 Inc-D).
    ///
    /// The buffer returned is the FULL capacity-padded allocation
    /// (`[1, H, max_len, head_dim]`), and the shape returned is the matching
    /// **physical/capacity** shape — see [`device_present_kv_view`] for why the
    /// physical (not logical valid) shape is the one that correctly strides the
    /// padded buffer. The caller slices out the freshly-decoded tokens with the
    /// same `extract_present_token` geometry the host-growable and ORT paths use,
    /// so all three mirror byte-identical pages.
    ///
    /// Runs after the decode step's own device→host sync (the KV bindings hold
    /// the committed present KV once the step returns), so no extra
    /// synchronization is required here — this is a pure post-step reader,
    /// mirroring the per-step logits read in [`Self::read_logits`].
    ///
    /// The binding may be `f32` (Inc-D) or `f16` (Inc-D.1); the raw device bytes
    /// are widened to `f32` with the same `half` convert ORT uses
    /// ([`kv_dtype_to_f32`]) so the mirrored pages are byte-identical to ORT's.
    #[allow(dead_code)]
    pub(crate) fn read_present_kv(
        &mut self,
        past_name: &str,
    ) -> anyhow::Result<Option<(Vec<f32>, Vec<usize>)>> {
        let Some(index) = self
            .kv_binding_range
            .clone()
            .find(|&idx| self.bindings[idx].input_name() == past_name)
        else {
            return Ok(None);
        };
        // The device present-KV read-out below strides the padded buffer with
        // hard-coded head-major (BNSH) arithmetic: `capacity_head_stride =
        // physical_shape[2] * head_dim * elem` with a per-head compaction. Under
        // a seq-major (BSNH) physical buffer the same offsets index the wrong
        // bytes (heads are interleaved per token, not laid out as capacity
        // stripes), so this host mirror would silently return mis-indexed KV.
        // Only our own GQA kernel understands the seq-major byte geometry; a
        // host mirror / paged-prefix consumer of `present_*` is a fourth place
        // the layout lives that the seq-major byte geometry does not yet honor.
        // Refuse rather than mis-map (the runtime's hard "error, never
        // mis-index" gate for the converted path).
        if self.kv_layout.is_seq_major() {
            bail!(
                "device present-KV read-out for '{past_name}' is unavailable under the seq-major \
                 (BSNH) KV layout: the host mirror strides the padded buffer with head-major \
                 (BNSH) arithmetic and would mis-index the interleaved seq-major bytes. Seq-major \
                 is supported only on the pure decode path where our GQA kernel is the sole \
                 consumer of the device KV; host-mirror / paged-prefix reuse must run under \
                 head-major KV."
            );
        }
        let binding = &mut self.bindings[index];
        let dtype = binding.dtype;
        let mut physical_shape = binding.physical_shape().to_vec();
        let bytes = if self.kv_commits_on_demand {
            let seq_len = self.logical_len;
            let elem = dtype.checked_storage_bytes(1).with_context(|| {
                format!("device present KV '{past_name}' has unsized dtype {dtype:?}")
            })?;
            let heads = physical_shape[1];
            let head_dim = physical_shape[3];
            let capacity_head_stride = physical_shape[2] * head_dim * elem;
            let live_head_bytes = seq_len * head_dim * elem;
            let mut compact = Vec::with_capacity(heads * live_head_bytes);
            for head in 0..heads {
                compact.extend(
                    binding.read_bytes_range(head * capacity_head_stride, live_head_bytes)?,
                );
            }
            physical_shape[2] = seq_len;
            compact
        } else {
            binding
                .read_bytes()
                .with_context(|| format!("read device present KV for '{past_name}'"))?
        };
        let tensor = Tensor::from_raw(dtype, physical_shape.clone(), &bytes)
            .with_context(|| format!("interpret device present KV bytes for '{past_name}'"))?;
        let values = kv_dtype_to_f32(&tensor)
            .with_context(|| format!("widen device present KV for '{past_name}' to f32"))?;
        Ok(Some(device_present_kv_view(
            values,
            &physical_shape,
            binding.logical_shape(),
        )))
    }

    /// Seed a materialized paged prefix into the device KV bindings so a request
    /// that shares a prompt prefix resumes without recomputing it (GAP-3 Inc-D
    /// device counterpart of the host-growable [`NativeDecodeSession::seed_growable_kv`]).
    ///
    /// `entries` are `(past_input_name, row_major_f32, [1, H, seq_len, head_dim])`
    /// triples — the same compact prefix layout the host path seeds. Because the
    /// device buffer is capacity-strided (`max_len`) while the prefix is compact
    /// (`seq_len`), each head is written into its own capacity-offset slot rather
    /// than as one contiguous blob. The attention-mask prefix `[0, seq_len)` is
    /// marked attendable (the per-step decode only extends `[seq_len, total)`),
    /// and the KV logical length is advanced so the next step appends at
    /// `seq_len`.
    #[allow(dead_code)]
    pub(crate) fn seed_prefix(
        &mut self,
        session: &mut InferenceSession,
        entries: &[(String, Vec<f32>, Vec<usize>)],
        seq_len: usize,
    ) -> anyhow::Result<()> {
        if seq_len == 0 {
            bail!("device paged prefix reuse requires a non-empty prefix");
        }
        // The device seed below writes each head into its own capacity-offset
        // slot (`head * capacity_head_stride`), i.e. head-major (BNSH) byte
        // arithmetic. Under a seq-major (BSNH) physical buffer that offset
        // addresses the wrong bytes, so a shared-prefix seed would corrupt the
        // KV. Prefix sharing therefore does not "fall out" under seq-major with
        // the current device seed: refuse rather than mis-map (the runtime's
        // hard "error, never mis-index" gate for the converted path).
        if self.kv_layout.is_seq_major() {
            bail!(
                "device paged prefix reuse is unavailable under the seq-major (BSNH) KV layout: \
                 the device seed writes each head at a head-major (BNSH) capacity offset and would \
                 mis-index the interleaved seq-major bytes. Seq-major is supported only on the \
                 pure decode path where our GQA kernel is the sole consumer of the device KV; \
                 prefix reuse must run under head-major KV."
            );
        }
        self.ensure_capacity(session, seq_len)?;
        for (name, data, shape) in entries {
            let Some(index) = self
                .kv_binding_range
                .clone()
                .find(|&idx| self.bindings[idx].input_name() == name)
            else {
                bail!("device paged prefix names unknown KV past input '{name}'");
            };
            let binding = &mut self.bindings[index];
            let dtype = binding.dtype;
            let elem_size = dtype.checked_storage_bytes(1).with_context(|| {
                format!("device paged prefix KV '{name}' has unsized dtype {dtype:?}")
            })?;
            let physical = binding.physical_shape().to_vec();
            let heads = physical[1];
            let head_dim = physical[3];
            let max_len = physical[2];
            let expected = vec![1_usize, heads, seq_len, head_dim];
            if shape != &expected {
                bail!(
                    "device paged prefix KV '{name}' shape {shape:?} does not match the binding's \
                     [1, {heads}, {seq_len}, {head_dim}] capacity layout"
                );
            }
            if data.len() != heads * seq_len * head_dim {
                bail!(
                    "device paged prefix KV '{name}' has {} values, expected {}",
                    data.len(),
                    heads * seq_len * head_dim
                );
            }
            let compact_head_stride = seq_len * head_dim;
            let capacity_head_stride = max_len * head_dim;
            for head in 0..heads {
                let compact = &data[head * compact_head_stride..(head + 1) * compact_head_stride];
                // Narrow the f32 paged prefix back to the binding's dtype with the
                // same `half` convert ORT injects with, so f16 seed bytes are the
                // exact inverse of the f16→f32 read-out (bit-exact round-trip).
                let bytes = f32_slice_to_dtype_bytes(dtype, compact)?;
                binding.write_bytes(head * capacity_head_stride * elem_size, &bytes)?;
            }
        }
        self.set_logical_len(seq_len)?;
        let expose = self.decode_mask_expose_len(seq_len);
        self.extend_mask(0, seq_len, expose)?;
        Ok(())
    }

    pub(crate) fn rewind(&mut self, target_len: usize) -> anyhow::Result<()> {
        if target_len < self.logical_len {
            let zeros = vec![0u8; (self.logical_len - target_len) * std::mem::size_of::<i64>()];
            self.bindings[0].write_bytes(target_len * std::mem::size_of::<i64>(), &zeros)?;
        }
        self.bindings[0].set_logical_shape(vec![1, target_len])?;
        if target_len == 0 {
            // Fixed-size recurrent/conv states are unmasked rolling caches: a
            // reused session would otherwise inherit the previous generation's
            // terminal state, corrupting generation #2+. A full reset restores
            // the declared `init: zeros`. Speculative recurrent rewind to a
            // non-zero length is unsupported (mirrors the CPU path), so only the
            // reset boundary re-zeros here.
            for index in self.fixed_state_binding_range.clone() {
                let binding = &self.bindings[index];
                let bytes = checked_shape_bytes(binding.physical_shape(), binding.dtype)
                    .with_context(|| {
                        format!(
                            "fixed CUDA decoder state re-zero size overflow for binding {index} shape {:?}",
                            binding.physical_shape()
                        )
                    })?;
                native_cuda_memset_zero(binding.device_ptr() as usize, bytes)?;
            }
        }
        self.set_logical_len(target_len)
    }

    fn write_decode_inputs(&mut self, token_id: TokenId, position: usize) -> anyhow::Result<()> {
        self.bindings[self.input_ids_binding].write_bytes(0, &i64::from(token_id).to_le_bytes())?;
        self.write_position_binding(position)
    }

    /// Write the current position into the persistent `position_ids` device
    /// binding, replicated across every declared coordinate axis (`position_rank`
    /// copies). For a rank-1 decoder this is a single `i64` at offset 0, identical
    /// to before; a rank-N mrope decoder gets `[position; N]` — the one-token
    /// `linear_increment` coordinate for all axes.
    fn write_position_binding(&mut self, position: usize) -> anyhow::Result<()> {
        if let Some(index) = self.position_ids_binding {
            let position = i64::try_from(position).context("position id exceeds i64 range")?;
            let bytes = position.to_le_bytes();
            let axis_bytes = std::mem::size_of::<i64>();
            for axis in 0..self.position_rank {
                self.bindings[index].write_bytes(axis * axis_bytes, &bytes)?;
            }
        }
        Ok(())
    }

    /// Inc3c: write the one-token `inputs_embeds`/routed bytes into their
    /// persistent device bindings (and the generated position id), so the
    /// captured decode graph can be replayed. Each supplied tensor must map by
    /// exact port name to a declared persistent binding and match its byte
    /// capacity; any unmapped supplied tensor or unfilled port is an error — the
    /// same strict contract the eager owned build applies.
    fn write_captured_step_inputs(
        &mut self,
        supplied: &[(String, Tensor)],
        position: usize,
    ) -> anyhow::Result<()> {
        let mut supplied_map = HashMap::with_capacity(supplied.len());
        for (name, tensor) in supplied {
            if supplied_map.insert(name.as_str(), tensor).is_some() {
                bail!("native CUDA decode received duplicate routed step input '{name}'");
            }
        }
        for captured in &self.captured_step_inputs {
            let tensor = supplied_map.remove(captured.name.as_str()).with_context(|| {
                format!(
                    "declared per-step input '{}' was not supplied to the captured native CUDA decode step; route the producing component output to this exact decoder port",
                    captured.name
                )
            })?;
            let bytes = tensor.as_bytes();
            if bytes.len() != captured.byte_len {
                bail!(
                    "captured native CUDA decode step input '{}' has {} bytes but its persistent device binding holds {} — the per-step port geometry is not fixed to one token",
                    captured.name,
                    bytes.len(),
                    captured.byte_len
                );
            }
            self.bindings[captured.binding_index].write_bytes(0, bytes)?;
        }
        if !supplied_map.is_empty() {
            let mut unknown = supplied_map.keys().copied().collect::<Vec<_>>();
            unknown.sort_unstable();
            bail!(
                "captured native CUDA decode received routed step inputs that are not declared graph ports: {unknown:?}"
            );
        }
        self.write_position_binding(position)
    }

    fn run_one_token(
        &mut self,
        session: &mut InferenceSession,
        trace: &TraceContext,
    ) -> anyhow::Result<()> {
        debug_assert!(self.auxiliary_binding_range.end <= self.base_binding_count);
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
                        tracing::debug!(
                            capacity = self.max_len,
                            captures = self.graph_captures,
                            "native CUDA decode graph captured"
                        );
                    }
                    DeviceGraphCaptureResult::NotCapturable(report) => {
                        self.graph_fallbacks += 1;
                        self.graph_phase = DecodeCudaGraphPhase::Unsupported;
                        trace_capture_declines(trace, &report);
                        let reason = report.to_string();
                        self.graph_decline_reason = Some(reason.clone());
                        self.graph_fallback_reason = Some(reason.clone());
                        self.graph_fallback_report = Some(report);
                        tracing::warn!(
                            "native CUDA decode graph capture disabled for this generation: {reason}"
                        );
                        session.run_with_device_bindings(&[], &mut self.bindings)?;
                    }
                }
            }
            DecodeCudaGraphPhase::Ready => {
                let still_valid = session.replay_device_graph(&mut self.bindings)?;
                self.graph_replays += 1;
                if !still_valid {
                    // A control-flow branch flip (e.g. LongRoPE short↔long at the
                    // context threshold) changed a seeded output shape and retired
                    // the captured graph after producing this token eagerly.
                    // Re-warm and re-capture for the new branch.
                    self.graph_phase = DecodeCudaGraphPhase::NeedsWarmup;
                }
            }
            DecodeCudaGraphPhase::Unsupported => {
                session.run_with_device_bindings(&[], &mut self.bindings)?;
            }
        }
        Ok(())
    }

    /// Inc-1b PR-3: run one **routed** single-token decode step through the
    /// decode-inline sibling executor, driving the same warm-up → arm → capture →
    /// replay CUDA-graph state machine as [`Self::run_one_token`] but on the
    /// sibling entry points (`cohaagen-inc1b-pr3-scope.md`). The inlined body ops
    /// fold into the captured graph via the identical segmenter/warm-seeded
    /// machinery; body ops that sync/host-read (Transpose/Slice/Tile/ReduceSum)
    /// stay quarantined to eager seams by the sibling executor's own
    /// `capture_quarantine_ops`, exactly like the main path. Reached only when
    /// `route_inline` is true (a model with an inlineable single-trip recurrent
    /// `Scan` whose sibling was built), so capture engagement here is
    /// graph-property-gated; with no such `Scan` this is never called and the
    /// sibling is never even built.
    ///
    /// The main decode capture machine ([`Self::run_one_token`]) is NOT reached on
    /// routed steps, so the shared EP's single graph slot + capture-error latch
    /// are owned solely by the sibling — no double-capture, no cross-latch bleed.
    fn run_one_token_inline(
        &mut self,
        session: &mut InferenceSession,
        trace: &TraceContext,
    ) -> anyhow::Result<()> {
        debug_assert!(self.auxiliary_binding_range.end <= self.base_binding_count);
        if !self.graph_enabled {
            session.run_decode_inline_with_device_bindings(&[], &mut self.bindings)?;
            return Ok(());
        }

        match self.inline_graph_phase {
            DecodeCudaGraphPhase::NeedsWarmup => {
                session.run_decode_inline_with_device_bindings(&[], &mut self.bindings)?;
                self.inline_graph_phase = DecodeCudaGraphPhase::Armed;
            }
            DecodeCudaGraphPhase::Armed => {
                match session
                    .try_capture_decode_inline_with_device_bindings(&[], &mut self.bindings)?
                {
                    DeviceGraphCaptureResult::Captured(outputs) => {
                        if outputs.iter().any(Option::is_some) {
                            bail!(
                                "captured CUDA decode-inline step unexpectedly materialized a host output"
                            );
                        }
                        self.graph_captures += 1;
                        self.inline_graph_phase = DecodeCudaGraphPhase::Ready;
                        tracing::debug!(
                            capacity = self.max_len,
                            captures = self.graph_captures,
                            segments = session.decode_inline_captured_graph_segment_count(),
                            "native CUDA decode-inline graph captured"
                        );
                    }
                    DeviceGraphCaptureResult::NotCapturable(report) => {
                        self.graph_fallbacks += 1;
                        self.inline_graph_phase = DecodeCudaGraphPhase::Unsupported;
                        trace_capture_declines(trace, &report);
                        let reason = report.to_string();
                        self.graph_decline_reason = Some(reason.clone());
                        self.graph_fallback_reason = Some(reason.clone());
                        self.graph_fallback_report = Some(report);
                        tracing::warn!(
                            "native CUDA decode-inline graph capture disabled for this generation: {reason}"
                        );
                        session.run_decode_inline_with_device_bindings(&[], &mut self.bindings)?;
                    }
                }
            }
            DecodeCudaGraphPhase::Ready => {
                let still_valid = session.replay_decode_inline_device_graph(&mut self.bindings)?;
                self.graph_replays += 1;
                if !still_valid {
                    // A control-flow seam retired the sibling's captured graph
                    // after producing this token eagerly; re-warm and re-capture.
                    self.inline_graph_phase = DecodeCudaGraphPhase::NeedsWarmup;
                }
            }
            DecodeCudaGraphPhase::Unsupported => {
                session.run_decode_inline_with_device_bindings(&[], &mut self.bindings)?;
            }
        }
        Ok(())
    }

    fn read_logits(&mut self) -> anyhow::Result<Vec<Vec<f32>>> {
        let bytes = self.bindings[self.logits_binding].read_bytes()?;
        let logits = Tensor::from_raw(self.logits_dtype, self.logits_shape.clone(), &bytes)?;
        extract_logits(&logits)
    }

    pub(crate) fn greedy_fastpath_supported(&self) -> bool {
        self.bindings[self.logits_binding].device_argmax_supported()
    }

    fn read_greedy_result(&mut self) -> anyhow::Result<(TokenId, u32)> {
        let vocab = *self
            .logits_shape
            .last()
            .context("CUDA logits shape has no vocabulary dimension")?;
        self.bindings[self.logits_binding].device_argmax(vocab, &mut self.greedy_result)?;
        let mut bytes = [0_u8; 2 * std::mem::size_of::<u32>()];
        self.greedy_result.read_bytes_into(&mut bytes)?;
        Ok((
            u32::from_ne_bytes(
                bytes[..4]
                    .try_into()
                    .map_err(|_| anyhow::anyhow!("four token-id bytes"))?,
            ),
            u32::from_ne_bytes(
                bytes[4..]
                    .try_into()
                    .map_err(|_| anyhow::anyhow!("four capture-error bytes"))?,
            ),
        ))
    }

    pub(crate) fn invalidate_graph(
        &mut self,
        session: &mut InferenceSession,
    ) -> anyhow::Result<()> {
        session.reset_device_graph()?;
        // The sibling shares the main executor's EP, so the reset above already
        // cleared the shared EP graph + capture-error latch; additionally clear
        // the sibling executor's host-side capture schedule so a routed decode
        // step re-warms rather than replaying a graph the EP has dropped.
        session.reset_decode_inline_device_graph()?;
        self.graph_phase = DecodeCudaGraphPhase::NeedsWarmup;
        self.inline_graph_phase = DecodeCudaGraphPhase::NeedsWarmup;
        self.graph_invalidations += 1;
        Ok(())
    }

    /// Dormant option (c) switch (kept off until WP4). Arm the padded single
    /// M=maxK captured verify graph: fix the query-row capacity at `max_query_rows`
    /// and retain the captured graph across `rewind` (contents-only mutation)
    /// instead of invalidating it. Not reachable from the plain M=1 hot path nor
    /// the eager (option (b)) verify path; only a future WP4 driver flips it on.
    #[cfg(test)]
    pub(crate) fn configure_padded_verify_capture(&mut self, max_query_rows: usize) {
        self.padded_query_capacity = Some(max_query_rows);
        self.retain_graph_on_rewind = true;
    }

    /// Toggle whether `rewind` retains the captured decode graph (option (c),
    /// contents-only mutation) or invalidates it (option (b), the eager default).
    /// Dormant: only exercised by option-(c) correctness tests until WP4.
    #[cfg(test)]
    pub(crate) fn set_retain_graph_on_rewind(&mut self, retain: bool) {
        self.retain_graph_on_rewind = retain;
    }

    /// Fixed query-row capacity (M=maxK) of the dormant padded verify capture, or
    /// `None` while the eager (option (b)) verify path is in force.
    #[cfg(test)]
    pub(crate) fn padded_query_capacity(&self) -> Option<usize> {
        self.padded_query_capacity
    }

    pub(crate) fn debug_stats(&self, session: &InferenceSession) -> CudaKvDebugStats {
        let mut transfers = DeviceBindingTransferStats::default();
        let mut kv_committed_bytes = 0usize;
        let mut kv_physical_bytes_by_binding = Vec::new();
        let device_ptrs = self.bindings[self.kv_binding_range.clone()]
            .iter()
            .map(|binding| {
                let stats = binding.transfer_stats();
                transfers.host_upload_calls += stats.host_upload_calls;
                transfers.host_upload_bytes += stats.host_upload_bytes;
                transfers.host_download_calls += stats.host_download_calls;
                transfers.host_download_bytes += stats.host_download_bytes;
                kv_committed_bytes += binding.committed_bytes();
                kv_physical_bytes_by_binding.push(
                    checked_shape_bytes(binding.physical_shape(), binding.dtype).unwrap_or(0),
                );
                binding.device_ptr() as usize
            })
            .collect();
        CudaKvDebugStats {
            logical_len: self.logical_len,
            max_len: self.max_len,
            kv_committed_len: self.kv_committed_len,
            hard_max_len: self.capacity.max_len,
            max_len_source: self.capacity.source.clone(),
            device_ptrs,
            kv_committed_bytes,
            kv_physical_bytes_by_binding,
            kv_transfers: transfers,
            kv_growth_events: self.kv_growth_events,
            kv_growth_d2d_copy_bytes: self.kv_growth_d2d_copy_bytes,
            kv_layout_seq_major: self.kv_layout.is_seq_major(),
            graph: CudaGraphDebugStats {
                enabled: self.graph_enabled,
                captures: self.graph_captures,
                replays: self.graph_replays,
                fallbacks: self.graph_fallbacks,
                invalidations: self.graph_invalidations,
                growth_keeps: self.graph_growth_keeps,
                allocation_counts: session.device_allocation_counts().unwrap_or_default(),
                decline_reason: self.graph_decline_reason.clone(),
                growth_decision: self.graph_growth_decision.clone(),
                fallback_report: self.graph_fallback_report.clone(),
            },
        }
    }
}

struct NativeCudaGrownBuffers {
    replacements: Vec<(usize, DeviceIoBinding)>,
}

struct NativeCudaCapacityBackend<'a> {
    state: &'a mut DecodeCudaState,
    session: &'a mut InferenceSession,
}

impl onnx_genai_kv::KvCapacityGrowthBackend for NativeCudaCapacityBackend<'_> {
    type Error = anyhow::Error;
    type GrownBuffers = NativeCudaGrownBuffers;
    type GrownMask = DeviceIoBinding;

    fn current_capacity(&self) -> usize {
        self.state.max_len
    }

    fn hard_max_capacity(&self) -> usize {
        self.state.capacity.max_len
    }

    fn valid_len(&self) -> usize {
        self.state.logical_len
    }

    fn capacity_exceeded(&self, required: usize) -> Self::Error {
        anyhow::anyhow!(self.state.capacity_exceeded_error(required))
    }

    fn build_grown_buffers(
        &mut self,
        new_capacity: usize,
        valid_len: usize,
    ) -> anyhow::Result<Self::GrownBuffers> {
        let old_capacity = self.state.max_len;
        let layout = self.state.kv_layout;
        (|| {
            native_cuda_device_barrier(self.session)?;
            let mut replacements = Vec::with_capacity(self.state.kv_binding_range.len());
            for index in self.state.kv_binding_range.clone() {
                let old = &self.state.bindings[index];
                let mut physical_shape = old.physical_shape().to_vec();
                physical_shape[2] = new_capacity;
                let mut logical_shape = physical_shape.clone();
                logical_shape[2] = valid_len;
                let new_binding = self.session.allocate_device_binding(
                    old.input_name().to_owned(),
                    old.output_name().map(str::to_owned),
                    old.dtype,
                    physical_shape.clone(),
                    logical_shape,
                )?;
                let dst = new_binding.device_ptr() as usize;
                let src = old.device_ptr() as usize;
                let total_bytes =
                    checked_shape_bytes(&physical_shape, old.dtype).with_context(|| {
                        format!(
                            "CUDA KV grow allocation size overflow for '{}' shape {:?}",
                            old.input_name(),
                            physical_shape
                        )
                    })?;
                native_cuda_memset_zero(dst, total_bytes)?;
                // Copy the live prefix following the physical byte layout: one
                // contiguous run for seq-major (grow axis 1), one fragment per
                // head stripe for head-major (grow axis 2, unchanged).
                let elem = old.dtype.checked_storage_bytes(1).with_context(|| {
                    format!(
                        "CUDA KV grow copy element size overflow for '{}'",
                        old.input_name()
                    )
                })?;
                let (old_bytes, grow_axis) = kv_growth_byte_layout(old.physical_shape(), layout)?;
                let (new_bytes, _) = kv_growth_byte_layout(&physical_shape, layout)?;
                copy_kv_prefix_device_to_device(
                    dst, src, &old_bytes, &new_bytes, grow_axis, valid_len, elem,
                )?;
                replacements.push((index, new_binding));
            }
            Ok(NativeCudaGrownBuffers { replacements })
        })()
        .map_err(|error| {
            let memory = cuda_device_memory_snapshot(self.session.device_id().index as i32)
                .map_err(|error| error.to_string());
            self.state
                .growth_failed_error(old_capacity, new_capacity, error, memory)
        })
    }

    fn build_grown_mask(
        &mut self,
        new_capacity: usize,
        valid_len: usize,
    ) -> anyhow::Result<Option<Self::GrownMask>> {
        let old_capacity = self.state.max_len;
        (|| {
            let mut new_mask = self.session.allocate_device_binding(
                self.state.bindings[0].input_name().to_owned(),
                self.state.bindings[0].output_name().map(str::to_owned),
                DataType::Int64,
                vec![1, new_capacity],
                vec![1, valid_len],
            )?;
            native_cuda_memset_zero(
                new_mask.device_ptr() as usize,
                new_capacity * std::mem::size_of::<i64>(),
            )?;
            new_mask.set_logical_shape(vec![1, valid_len])?;
            Ok(Some(new_mask))
        })()
        .map_err(|error| {
            let memory = cuda_device_memory_snapshot(self.session.device_id().index as i32)
                .map_err(|error| error.to_string());
            self.state
                .growth_failed_error(old_capacity, new_capacity, error, memory)
        })
    }

    fn invalidate_capture(&mut self) -> anyhow::Result<()> {
        native_cuda_device_barrier(self.session)?;
        self.state.invalidate_graph(self.session)
    }

    fn commit_grown_capacity(
        &mut self,
        new_capacity: usize,
        grown_buffers: Self::GrownBuffers,
        grown_mask: Option<Self::GrownMask>,
    ) -> anyhow::Result<()> {
        if let Some(mask) = grown_mask {
            self.state.bindings[0] = mask;
        }
        for (index, binding) in grown_buffers.replacements {
            self.state.bindings[index] = binding;
        }
        self.state.max_len = new_capacity;
        Ok(())
    }
}

pub(crate) fn cuda_kv_max_len_from_env() -> anyhow::Result<Option<usize>> {
    match std::env::var("ONNX_GENAI_CUDA_KV_MAX_LEN") {
        Ok(value) => {
            let parsed = value.trim().parse::<usize>().with_context(|| {
                format!("invalid ONNX_GENAI_CUDA_KV_MAX_LEN={value:?}: expected a positive integer")
            })?;
            if parsed == 0 {
                bail!("ONNX_GENAI_CUDA_KV_MAX_LEN must be greater than zero");
            }
            Ok(Some(parsed))
        }
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(error) => Err(error).context("read ONNX_GENAI_CUDA_KV_MAX_LEN"),
    }
}

fn checked_shape_bytes(shape: &[usize], dtype: DataType) -> Option<usize> {
    let elements = shape
        .iter()
        .try_fold(1usize, |product, &dim| product.checked_mul(dim))?;
    dtype.checked_storage_bytes(elements)
}

#[cfg(feature = "cuda")]
fn native_cuda_device_barrier(session: &InferenceSession) -> anyhow::Result<()> {
    let _guard = onnx_genai_ort::cuda_rt::DeviceGuard::set(session.device_id().index as i32)?;
    onnx_genai_ort::cuda_rt::device_synchronize()?;
    Ok(())
}

#[cfg(not(feature = "cuda"))]
fn native_cuda_device_barrier(_session: &InferenceSession) -> anyhow::Result<()> {
    bail!("native CUDA KV growth requires the onnx-genai-engine `cuda` feature")
}

#[cfg(feature = "cuda")]
fn native_cuda_memset_zero(dst: usize, bytes: usize) -> anyhow::Result<()> {
    onnx_genai_ort::cuda_rt::memset_zero(dst, bytes)?;
    Ok(())
}

#[cfg(not(feature = "cuda"))]
fn native_cuda_memset_zero(_dst: usize, _bytes: usize) -> anyhow::Result<()> {
    bail!("native CUDA KV growth requires the onnx-genai-engine `cuda` feature")
}

/// Map a BNSH-declared KV binding shape `[batch, kv_heads, capacity, head_dim]`
/// to the physical byte-layout shape and grow-axis index the resolved
/// [`KvCommitLayout`] actually stores it in.
///
/// The binding *metadata* shape reported to the CUDA EP is always BNSH — the
/// GQA node validates `past_key`/`past_value` as `[batch, kv_heads, seq,
/// head_dim]` and reads `present_capacity` from axis 2 *regardless of the
/// `kv_layout` attribute* (only the kernel's stride arithmetic changes). So this
/// permutation never leaves the growth machinery; it drives the copy/zero
/// geometry over raw device bytes so that growth is faithful to how the bytes
/// are really laid out:
///
/// * **Head-major BNSH**: the byte layout equals the declared shape and the grow
///   axis is 2. Each head owns its own `capacity × head_dim` stripe, so growth
///   re-strides every head — byte-identical to the historical behavior.
/// * **Seq-major BSNH**: the bytes are `[batch, capacity, kv_heads, head_dim]`
///   and the grow axis is 1. A token's whole KV (all heads) is contiguous with a
///   fixed `kv_heads × head_dim` stride that is **independent of capacity**, so
///   the live prefix keeps its byte offsets across growth and no KV data moves.
///   This is the fixed-stride property #797 measured at the driver level, here
///   realized on the engine growth path.
pub(super) fn kv_growth_byte_layout(
    bnsh_shape: &[usize],
    layout: KvCommitLayout,
) -> anyhow::Result<(Vec<usize>, usize)> {
    if bnsh_shape.len() != 4 {
        bail!("CUDA KV binding must be rank-4 BNSH, got shape {bnsh_shape:?}");
    }
    Ok(match layout {
        KvCommitLayout::HeadMajor => (bnsh_shape.to_vec(), 2),
        KvCommitLayout::SeqMajor => {
            let (batch, kv_heads, capacity, head_dim) =
                (bnsh_shape[0], bnsh_shape[1], bnsh_shape[2], bnsh_shape[3]);
            (vec![batch, capacity, kv_heads, head_dim], 1)
        }
    })
}

#[cfg(feature = "cuda")]
fn copy_kv_prefix_device_to_device(
    dst: usize,
    src: usize,
    old_shape: &[usize],
    new_shape: &[usize],
    seq_axis: usize,
    valid_len: usize,
    elem_size: usize,
) -> anyhow::Result<()> {
    if valid_len == 0 {
        return Ok(());
    }
    if old_shape.len() != new_shape.len() || seq_axis >= old_shape.len() {
        bail!("invalid CUDA KV grow copy shapes {old_shape:?} -> {new_shape:?} on axis {seq_axis}");
    }
    let old_cap = old_shape[seq_axis];
    let new_cap = new_shape[seq_axis];
    if valid_len > old_cap || valid_len > new_cap {
        bail!(
            "invalid CUDA KV grow copy valid prefix {valid_len} for shapes {old_shape:?} -> {new_shape:?}"
        );
    }
    let blocks = old_shape[..seq_axis]
        .iter()
        .try_fold(1usize, |product, &dim| product.checked_mul(dim))
        .context("CUDA KV grow copy block count overflow")?;
    let inner = old_shape[seq_axis + 1..]
        .iter()
        .try_fold(1usize, |product, &dim| product.checked_mul(dim))
        .context("CUDA KV grow copy inner size overflow")?;
    let segment_bytes = valid_len
        .checked_mul(inner)
        .and_then(|elements| elements.checked_mul(elem_size))
        .context("CUDA KV grow copy segment size overflow")?;
    let old_stride = old_cap
        .checked_mul(inner)
        .and_then(|elements| elements.checked_mul(elem_size))
        .context("CUDA KV grow copy old stride overflow")?;
    let new_stride = new_cap
        .checked_mul(inner)
        .and_then(|elements| elements.checked_mul(elem_size))
        .context("CUDA KV grow copy new stride overflow")?;
    for block in 0..blocks {
        let src_offset = block
            .checked_mul(old_stride)
            .context("CUDA KV grow copy source offset overflow")?;
        let dst_offset = block
            .checked_mul(new_stride)
            .context("CUDA KV grow copy destination offset overflow")?;
        onnx_genai_ort::cuda_rt::memcpy_device_to_device(
            dst + dst_offset,
            src + src_offset,
            segment_bytes,
        )?;
    }
    Ok(())
}

#[cfg(feature = "cuda")]
fn copy_kv_prefix_device_to_device_in_place(
    ptr: usize,
    old_shape: &[usize],
    new_shape: &[usize],
    seq_axis: usize,
    valid_len: usize,
    elem_size: usize,
) -> anyhow::Result<()> {
    if valid_len == 0 {
        return Ok(());
    }
    if old_shape.len() != new_shape.len() || seq_axis >= old_shape.len() {
        bail!(
            "invalid CUDA KV in-place grow copy shapes {old_shape:?} -> {new_shape:?} on axis {seq_axis}"
        );
    }
    let old_cap = old_shape[seq_axis];
    let new_cap = new_shape[seq_axis];
    let blocks = old_shape[..seq_axis]
        .iter()
        .try_fold(1usize, |product, &dim| product.checked_mul(dim))
        .context("CUDA KV in-place grow copy block count overflow")?;
    let inner = old_shape[seq_axis + 1..]
        .iter()
        .try_fold(1usize, |product, &dim| product.checked_mul(dim))
        .context("CUDA KV in-place grow copy inner size overflow")?;
    let segment_bytes = valid_len
        .checked_mul(inner)
        .and_then(|elements| elements.checked_mul(elem_size))
        .context("CUDA KV in-place grow copy segment size overflow")?;
    let old_stride = old_cap
        .checked_mul(inner)
        .and_then(|elements| elements.checked_mul(elem_size))
        .context("CUDA KV in-place grow copy old stride overflow")?;
    let new_stride = new_cap
        .checked_mul(inner)
        .and_then(|elements| elements.checked_mul(elem_size))
        .context("CUDA KV in-place grow copy new stride overflow")?;
    for block in (0..blocks).rev() {
        let src_offset = block
            .checked_mul(old_stride)
            .context("CUDA KV in-place grow copy source offset overflow")?;
        let dst_offset = block
            .checked_mul(new_stride)
            .context("CUDA KV in-place grow copy destination offset overflow")?;
        match in_place_copy_route(src_offset, dst_offset, segment_bytes) {
            InPlaceCopyRoute::Noop => {}
            InPlaceCopyRoute::Scratch => {
                let src_start = ptr + src_offset;
                let mut scratch = vec![0u8; segment_bytes];
                onnx_genai_ort::cuda_rt::memcpy_device_to_host(&mut scratch, src_start)?;
                onnx_genai_ort::cuda_rt::memcpy_host_to_device(ptr + dst_offset, &scratch)?;
            }
            InPlaceCopyRoute::DeviceToDevice => {
                onnx_genai_ort::cuda_rt::memcpy_device_to_device(
                    ptr + dst_offset,
                    ptr + src_offset,
                    segment_bytes,
                )?;
            }
        }
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[cfg(feature = "cuda")]
pub(super) enum InPlaceCopyRoute {
    Noop,
    DeviceToDevice,
    Scratch,
}

#[cfg(feature = "cuda")]
pub(super) fn in_place_copy_route(
    src_offset: usize,
    dst_offset: usize,
    segment_bytes: usize,
) -> InPlaceCopyRoute {
    if src_offset == dst_offset || segment_bytes == 0 {
        return InPlaceCopyRoute::Noop;
    }
    let src_end = src_offset + segment_bytes;
    let dst_end = dst_offset + segment_bytes;
    if dst_offset < src_end && src_offset < dst_end {
        InPlaceCopyRoute::Scratch
    } else {
        InPlaceCopyRoute::DeviceToDevice
    }
}

#[cfg(not(feature = "cuda"))]
fn copy_kv_prefix_device_to_device_in_place(
    _ptr: usize,
    _old_shape: &[usize],
    _new_shape: &[usize],
    _seq_axis: usize,
    _valid_len: usize,
    _elem_size: usize,
) -> anyhow::Result<()> {
    bail!("native CUDA KV growth requires the onnx-genai-engine `cuda` feature")
}

#[cfg(feature = "cuda")]
fn zero_kv_suffix_device(
    ptr: usize,
    shape: &[usize],
    seq_axis: usize,
    valid_len: usize,
    elem_size: usize,
) -> anyhow::Result<()> {
    if shape.is_empty() || seq_axis >= shape.len() || valid_len >= shape[seq_axis] {
        return Ok(());
    }
    let cap = shape[seq_axis];
    let blocks = shape[..seq_axis]
        .iter()
        .try_fold(1usize, |product, &dim| product.checked_mul(dim))
        .context("CUDA KV suffix zero block count overflow")?;
    let inner = shape[seq_axis + 1..]
        .iter()
        .try_fold(1usize, |product, &dim| product.checked_mul(dim))
        .context("CUDA KV suffix zero inner size overflow")?;
    let stride = cap
        .checked_mul(inner)
        .and_then(|elements| elements.checked_mul(elem_size))
        .context("CUDA KV suffix zero stride overflow")?;
    let suffix_offset = valid_len
        .checked_mul(inner)
        .and_then(|elements| elements.checked_mul(elem_size))
        .context("CUDA KV suffix zero offset overflow")?;
    let suffix_bytes = (cap - valid_len)
        .checked_mul(inner)
        .and_then(|elements| elements.checked_mul(elem_size))
        .context("CUDA KV suffix zero byte count overflow")?;
    for block in 0..blocks {
        native_cuda_memset_zero(ptr + block * stride + suffix_offset, suffix_bytes)?;
    }
    Ok(())
}

#[cfg(not(feature = "cuda"))]
fn zero_kv_suffix_device(
    _ptr: usize,
    _shape: &[usize],
    _seq_axis: usize,
    _valid_len: usize,
    _elem_size: usize,
) -> anyhow::Result<()> {
    bail!("native CUDA KV growth requires the onnx-genai-engine `cuda` feature")
}

#[cfg(not(feature = "cuda"))]
fn copy_kv_prefix_device_to_device(
    _dst: usize,
    _src: usize,
    _old_shape: &[usize],
    _new_shape: &[usize],
    _seq_axis: usize,
    _valid_len: usize,
    _elem_size: usize,
) -> anyhow::Result<()> {
    bail!("native CUDA KV growth requires the onnx-genai-engine `cuda` feature")
}

pub(crate) fn resolve_cuda_kv_capacity(
    programmatic_max_len: Option<usize>,
    env_max_len: Option<usize>,
    metadata_max_len: Option<usize>,
    bytes_per_token: usize,
    device_memory: Option<CudaDeviceMemorySnapshot>,
) -> anyhow::Result<CudaKvCapacity> {
    if bytes_per_token == 0 {
        bail!("CUDA KV bytes per token must be greater than zero");
    }
    let (configured, source) = if let Some(max_len) = programmatic_max_len {
        if let Some(metadata_len) = metadata_max_len
            && metadata_len < max_len
        {
            (
                metadata_len,
                format!(
                    "programmatic load_with_cuda_kv_max_len clamped by model.max_sequence_length ({metadata_len})"
                ),
            )
        } else {
            (max_len, "programmatic load_with_cuda_kv_max_len".to_owned())
        }
    } else if let Some(max_len) = env_max_len {
        if let Some(metadata_len) = metadata_max_len
            && metadata_len < max_len
        {
            (
                metadata_len,
                format!(
                    "ONNX_GENAI_CUDA_KV_MAX_LEN clamped by model.max_sequence_length ({metadata_len})"
                ),
            )
        } else {
            (max_len, "ONNX_GENAI_CUDA_KV_MAX_LEN".to_owned())
        }
    } else if let Some(max_len) = metadata_max_len {
        (max_len, "model.max_sequence_length".to_owned())
    } else {
        (
            usize::MAX,
            "unbounded (model.max_sequence_length unavailable)".to_owned(),
        )
    };
    if configured == 0 {
        bail!(
            "derived CUDA KV max length is zero from {source}; free more device memory or set a smaller ONNX_GENAI_CUDA_KV_MAX_LEN/load_with_cuda_kv_max_len value"
        );
    }
    Ok(CudaKvCapacity {
        max_len: configured,
        source,
        metadata_max_len,
        device_memory,
        bytes_per_token,
    })
}

pub(crate) fn cuda_kv_capacity_exceeded_message(
    requested: usize,
    capacity: &CudaKvCapacity,
) -> String {
    let metadata = capacity
        .metadata_max_len
        .map(|value| value.to_string())
        .unwrap_or_else(|| "unavailable".to_owned());
    let memory = capacity
        .device_memory
        .map(|memory| {
            format!(
                "CUDA free={} bytes, total={} bytes",
                memory.free_bytes, memory.total_bytes
            )
        })
        .unwrap_or_else(|| "CUDA free-memory query unavailable".to_owned());
    format!(
        "CUDA KV capacity exceeded: requested context length {requested}, configured max_len {} (source: {}; model.max_sequence_length: {metadata}; {memory}; KV bytes/token: {}). Increase the limit with ONNX_GENAI_CUDA_KV_MAX_LEN or load_with_cuda_kv_max_len if the device has enough VRAM, or reduce prompt/max_new_tokens.",
        capacity.max_len, capacity.source, capacity.bytes_per_token
    )
}

#[cfg(feature = "cuda")]
pub(crate) fn cuda_device_memory_snapshot(
    device_id: i32,
) -> anyhow::Result<CudaDeviceMemorySnapshot> {
    let memory = onnx_genai_ort::cuda_rt::device_memory_info(device_id)
        .context("query CUDA free device memory for native KV capacity")?;
    Ok(CudaDeviceMemorySnapshot {
        free_bytes: memory.free_bytes,
        total_bytes: memory.total_bytes,
    })
}

#[cfg(not(feature = "cuda"))]
pub(crate) fn cuda_device_memory_snapshot(
    _device_id: i32,
) -> anyhow::Result<CudaDeviceMemorySnapshot> {
    bail!("CUDA free-memory query requires the onnx-genai-engine `cuda` feature")
}

/// Pair a capacity-padded device KV read-out with the shape whose row-major
/// strides address it. Returns the **physical/capacity** shape
/// (`[1, H, max_len, head_dim]`), NOT the logical valid shape
/// (`[1, H, valid_len, head_dim]`).
///
/// The device buffer is allocated at `max_len` in the sequence axis and never
/// re-packed, so the head-axis stride is `max_len * head_dim`. Feeding the
/// logical valid shape to [`extract_present_token`] would instead compute a head
/// stride of `valid_len * head_dim` and, whenever `max_len > valid_len` and
/// `H > 1`, silently read the wrong head rows (the sequence and head-dim strides
/// coincide at `H == 1`, so the bug is invisible for single-head caches). This is
/// the one device-specific geometry decision Inc-D adds on top of the Inc-C host
/// mirror; [`device_kv_view_uses_physical_stride`] pins it at `H == 2`.
fn device_present_kv_view(
    buffer: Vec<f32>,
    physical_shape: &[usize],
    _logical_shape: &[usize],
) -> (Vec<f32>, Vec<usize>) {
    (buffer, physical_shape.to_vec())
}

#[cfg(test)]
mod device_kv_view_tests {
    use super::device_present_kv_view;
    use crate::kv_bridge::extract_present_token;
    use onnx_genai_kv::{KvDType, PageTensorConfig};

    /// A capacity-padded KV buffer read from a device binding must be addressed by
    /// its **physical** shape, not its logical valid length. This reproduces the
    /// exact production read+extract composition at `H == 2, max_len=4, valid=2`
    /// (the geometry is invisible at the `H == 1` integration fixture) and proves
    /// the physical stride recovers the two valid heads while the logical stride —
    /// the Inc-D-specific "max_len wrinkle" bug — reads the padding instead.
    #[test]
    fn device_kv_view_uses_physical_stride() {
        let heads = 2usize;
        let max_len = 4usize;
        let valid = 2usize;
        let head_dim = 2usize;
        // Row-major [1, H, max_len, head_dim]; valid tokens carry distinctive
        // values, padded tail carries a sentinel that must never be read back.
        let mut buffer = vec![-999.0f32; heads * max_len * head_dim];
        for head in 0..heads {
            for seq in 0..valid {
                for dim in 0..head_dim {
                    let value = (head as f32 + 1.0) * 100.0 + seq as f32 * 10.0 + dim as f32;
                    buffer[head * max_len * head_dim + seq * head_dim + dim] = value;
                }
            }
        }
        let physical = vec![1usize, heads, max_len, head_dim];
        let logical = vec![1usize, heads, valid, head_dim];
        let config = PageTensorConfig {
            num_layers: 1,
            num_kv_heads: heads,
            head_dim,
            page_size: 1,
            dtype: KvDType::F32,
        };

        // Production view: physical shape.
        let (data, shape) = device_present_kv_view(buffer.clone(), &physical, &logical);
        assert_eq!(shape, physical, "device view must carry the physical shape");
        let shape_i64 = shape.iter().map(|&d| d as i64).collect::<Vec<_>>();
        let token0 = extract_present_token(&data, &shape_i64, config, 0).unwrap();
        let token1 = extract_present_token(&data, &shape_i64, config, 1).unwrap();
        assert_eq!(token0, vec![100.0, 101.0, 200.0, 201.0]);
        assert_eq!(token1, vec![110.0, 111.0, 210.0, 211.0]);

        // The logical valid shape (the max_len wrinkle bug) reads padding for the
        // second head — so a physical→logical mutation of the production view
        // diverges here, keeping the reader non-vacuous at H >= 2.
        let logical_i64 = logical.iter().map(|&d| d as i64).collect::<Vec<_>>();
        let wrong0 = extract_present_token(&buffer, &logical_i64, config, 0).unwrap();
        assert_ne!(
            wrong0, token0,
            "logical-stride read must diverge from the physical-stride read at H >= 2"
        );
    }
}

#[cfg(test)]
mod capture_step_inputs_gate_tests {
    use super::capture_step_inputs_from_env_value;

    /// Capture is default-on: an unset env keeps the graph-capture perf path.
    #[test]
    fn unset_env_defaults_to_capture_on() {
        assert!(capture_step_inputs_from_env_value(None));
    }

    /// The opt-out escape hatch: only explicit falsy values force eager.
    #[test]
    fn falsy_values_opt_out_to_eager() {
        for value in ["0", "false", "no", "off", " OFF ", "False", "No"] {
            assert!(
                !capture_step_inputs_from_env_value(Some(value)),
                "{value:?} must opt out to the eager owned path"
            );
        }
    }

    /// Truthy or unrecognized values keep capture on (default-on, opt-out only).
    #[test]
    fn truthy_and_unknown_values_keep_capture_on() {
        for value in ["1", "true", "yes", "on", "", "anything"] {
            assert!(
                capture_step_inputs_from_env_value(Some(value)),
                "{value:?} must keep capture on"
            );
        }
    }
}
