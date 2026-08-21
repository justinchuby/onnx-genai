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
    weight_offload_stable_va: Option<bool>,
) -> GraphCaptureDecision {
    // `weight_offload_stable_va` is three-state on purpose: `Some(true)` proved
    // stable (capture-safe), `Some(false)` proved unstable by the effective
    // offload policy, and `None` means no policy was supplied so the caller took
    // the conservative default. The first two are runtime facts; the third is a
    // harness/plumbing gap that must not masquerade as a runtime fact — print the
    // predicate inputs and their source so a declined capture is diagnosable.
    let stable_va_safe = weight_offload_stable_va == Some(true);
    if weight_offload_enabled && !stable_va_safe {
        let source = match weight_offload_stable_va {
            Some(false) => "effective offload policy: pointer-unstable paging path",
            None => "caller default, cuda_offload_policy not supplied (unwrap_or false)",
            Some(true) => unreachable!("stable_va_safe would be true"),
        };
        return GraphCaptureDecision::declined(
            "weight_offload_enabled && !weight_offload_stable_va",
            format_args!(
                "weight_offload_enabled={weight_offload_enabled} \
                 weight_offload_stable_va={stable_va_safe} (source: {source})"
            ),
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
    weight_offload_stable_va: Option<bool>,
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

/// Upper bound on the device-token-loop chain depth `k`, matching the scratch
/// token-log capacity. A small `k` keeps the host stop/EOS checks cheap (at most
/// `k - 1` speculatively-run replays are discarded when generation stops
/// mid-chain) while still amortizing the host token round-trip across the chain.
pub(crate) const DEVICE_TOKEN_LOOP_MAX_K: usize = 16;
/// Default chain depth when `ONNX_GENAI_DEVICE_TOKEN_LOOP` is set to an enabling
/// value without an explicit count (e.g. `1`/`true`/`on`).
const DEVICE_TOKEN_LOOP_DEFAULT_K: usize = 4;

/// Parse the requested device-token-loop chain depth from
/// `ONNX_GENAI_DEVICE_TOKEN_LOOP`. `0`/unset/`false`/`off` disable it; a bare
/// enabling value (`1`/`true`/`yes`/`on`) selects [`DEVICE_TOKEN_LOOP_DEFAULT_K`];
/// an explicit integer `>= 2` selects that depth, clamped to
/// [`DEVICE_TOKEN_LOOP_MAX_K`].
fn device_token_loop_k_from_env() -> usize {
    let Some(value) = std::env::var_os("ONNX_GENAI_DEVICE_TOKEN_LOOP") else {
        return 0;
    };
    let value = value.to_string_lossy();
    let value = value.trim().to_ascii_lowercase();
    match value.as_str() {
        "" | "0" | "false" | "no" | "off" => 0,
        "1" | "true" | "yes" | "on" => DEVICE_TOKEN_LOOP_DEFAULT_K,
        other => match other.parse::<usize>() {
            Ok(0) => 0,
            Ok(1) => DEVICE_TOKEN_LOOP_DEFAULT_K,
            Ok(k) => k.min(DEVICE_TOKEN_LOOP_MAX_K),
            Err(_) => 0,
        },
    }
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
    vram_free_sync_ns: u64,
}

impl StepOffloadSnapshot {
    fn read() -> Self {
        #[cfg(feature = "native-cuda")]
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
                vram_free_sync_ns: stats.vram_free_sync_ns,
            }
        }
        #[cfg(not(feature = "native-cuda"))]
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
            vram_free_sync_ns: self
                .vram_free_sync_ns
                .saturating_sub(before.vram_free_sync_ns),
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
        // Phase accounting only. This deliberately does not switch on the
        // activation-memory planner, which used to come along with it: the
        // planner re-plans every activation on every run, so it taxed the very
        // decode steps this profiler exists to time. A caller that wants its
        // stats asks for them with `enable_activation_memory_plan_for_process`
        // or `NXRT_ACTIVATION_MEMORY_PLAN=1`.
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
        let vram_free_sync_ms = ns_to_ms(delta.vram_free_sync_ns);
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
            staging_ms + h2d_ms + admit_sync_ms + vram_alloc_ms + vram_free_ms + vram_free_sync_ms;
        let build_inputs_unattributed_ms = (build_inputs_ms - build_inputs_attributed_ms).max(0.0);
        let executor_other_ms = wall.run_ms - build_inputs_ms - kernel_host_ms;
        let run_unattributed_ms = build_inputs_unattributed_ms + executor_other_ms;
        let residual_ms = total_ms
            - staging_ms
            - h2d_ms
            - admit_sync_ms
            - vram_alloc_ms
            - vram_free_ms
            - vram_free_sync_ms
            - kernel_host_ms
            - build_inputs_unattributed_ms
            - executor_other_ms
            - wall.logits_read_ms
            - wall.capture_check_ms
            - wall.finite_check_ms;
        static HEADER: std::sync::Once = std::sync::Once::new();
        HEADER.call_once(|| {
            eprintln!(
                "[onnx-genai-cuda-step] path,past_len,total_len,total_ms,staging_fill_ms,h2d_copy_ms,kernel_host_dispatch_ms,admit_sync_ms,vram_alloc_ms,vram_free_ms,vram_free_sync_ms,build_inputs_unattributed_ms,executor_other_ms,run_unattributed_ms,logits_read_sync_ms,capture_check_ms,finite_check_ms,residual_ms,page_ins,staging_fill_bytes,staging_fill_regions,staging_fill_calls,materialize_fallback_calls,h2d_bytes"
            );
        });
        eprintln!(
            "[onnx-genai-cuda-step] {path},{},{},{total_ms:.3},{staging_ms:.3},{h2d_ms:.3},{kernel_host_ms:.3},{admit_sync_ms:.3},{vram_alloc_ms:.3},{vram_free_ms:.3},{vram_free_sync_ms:.3},{build_inputs_unattributed_ms:.3},{executor_other_ms:.3},{run_unattributed_ms:.3},{:.3},{:.3},{:.3},{residual_ms:.3},{},{},{},{},{},{}",
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

/// Widen the query-seq axis of a token/position binding shape (the LAST axis:
/// `[1, 1]` → `[1, M]`, `[rank, 1, 1]` → `[rank, 1, M]`) to `M`, for the padded
/// fixed-M verify bindings. Returns `None` when the shape is empty or the axis
/// being widened is not currently `1` (so the caller declines verify capture
/// rather than mis-shaping a binding). No hardcoded dims — the width comes from
/// `num_speculative_tokens + 1`.
fn widen_query_last(shape: &[usize], m: usize) -> Option<Vec<usize>> {
    let last = shape.len().checked_sub(1)?;
    if shape[last] != 1 {
        return None;
    }
    let mut widened = shape.to_vec();
    widened[last] = m;
    Some(widened)
}

/// Widen the query-seq axis of a logits/aux output shape (the `len-2` axis,
/// since a trailing feature dim (vocab/hidden) is last: `[1, 1, vocab]` →
/// `[1, M, vocab]`) to `M`. Returns `None` when the shape has rank < 2 or the
/// query-seq axis is not currently `1` (decline verify capture rather than
/// mis-shape an output binding).
fn widen_query_seq(shape: &[usize], m: usize) -> Option<Vec<usize>> {
    let axis = shape.len().checked_sub(2)?;
    if shape[axis] != 1 {
        return None;
    }
    let mut widened = shape.to_vec();
    widened[axis] = m;
    Some(widened)
}

/// Result of one [`NativeDecodeSession::leverb_phase0_capture_attempt`] call —
/// a THROWAWAY Lever-B Phase-0 probe (leverb-phase0), not part of any shipping
/// contract.
#[cfg(all(test, feature = "native-cuda"))]
#[derive(Clone, Debug, Default)]
pub(crate) struct LeverBPhase0CaptureAttempt {
    /// Query rows in the attempted forward (`k_max` for M=K, `1` for M=1).
    pub rows: usize,
    /// Committed past length the forward attended.
    pub past_len: usize,
    /// Physical KV bucket (`max_len`) the capture was frozen to.
    pub bucket: usize,
    /// Whether `try_capture_with_device_bindings` instantiated a device graph.
    pub captured: bool,
    /// Installed captured segment count (`1` = whole-subgraph, `>=2` = seamed).
    pub segments: usize,
    /// Decline reason when not captured, or a note when captured-with-caveat.
    pub decline: Option<String>,
    /// Device (allocations, frees) observed across the capture run — Phase-0
    /// pass criterion (a) requires zero alloc/free inside the captured region.
    pub alloc_delta: Option<(u64, u64)>,
    /// Device (allocations, frees) observed across the Increment-0 pre-capture
    /// warm forward at the M=K shape (grows the capture-safe scratch arena so
    /// the captured region itself stays alloc-free). `None` for the raw Phase-0
    /// probe, which performs no warm pass.
    pub warm_alloc_delta: Option<(u64, u64)>,
    /// Per-replay wall (ns), GPU-inclusive (synchronized by a logits read).
    pub replay_walls_ns: Vec<u64>,
    /// Increment-0 only: a compact summary of the eager seam nodes that split a
    /// segmented M=K capture (`op_type[seam_reason] × count`), the root cause of
    /// a >1-segment capture that never reaches the whole-graph replay fast path.
    pub seam_summary: Option<String>,
    /// Captured-vs-eager token parity (only populated when the attempt is asked
    /// to `collect_parity` with real token inputs): the per-row greedy argmax of
    /// the EAGER pre-capture warm forward. Empty when parity was not collected.
    pub warm_argmax: Vec<i64>,
    /// Captured-vs-eager token parity: the per-row greedy argmax of the last
    /// CAPTURED-graph replay over the identical device bindings. Empty when
    /// parity was not collected. Compared against [`Self::warm_argmax`] this is
    /// the "captured M=K matches eager M=K" token-equality cell.
    pub replay_argmax: Vec<i64>,
    /// Whether the raw logits bytes of the eager warm forward and the captured
    /// replay were byte-for-byte identical (the strongest captured-vs-eager
    /// statement). `None` when parity was not collected or capture declined.
    pub logits_byte_identical: Option<bool>,
}

/// Per-row greedy argmax over a `[rows, vocab]` logits buffer of raw device
/// bytes. Supports the three logits dtypes the decoder emits (f32/f16/bf16). Ties
/// resolve to the lowest index (matching the decoder's argmax tie-break).
#[cfg(all(test, feature = "native-cuda"))]
fn logits_rows_argmax(bytes: &[u8], dtype: DataType, rows: usize, vocab: usize) -> Vec<i64> {
    let mut out = Vec::with_capacity(rows);
    let decode = |b: &[u8]| -> f32 {
        match dtype {
            DataType::Float16 => half::f16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32(),
            DataType::BFloat16 => {
                let bits = (u32::from(u16::from_le_bytes([b[0], b[1]]))) << 16;
                f32::from_bits(bits)
            }
            _ => f32::from_le_bytes([b[0], b[1], b[2], b[3]]),
        }
    };
    let width = match dtype {
        DataType::Float16 | DataType::BFloat16 => 2,
        _ => 4,
    };
    for row in 0..rows {
        let base = row * vocab * width;
        let mut best_idx: i64 = 0;
        let mut best_val = f32::NEG_INFINITY;
        for col in 0..vocab {
            let off = base + col * width;
            if off + width > bytes.len() {
                break;
            }
            let val = decode(&bytes[off..off + width]);
            if val > best_val {
                best_val = val;
                best_idx = col as i64;
            }
        }
        out.push(best_idx);
    }
    out
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
    /// Configured device-token-loop chain depth (`0` = disabled/not armed).
    pub device_token_loop_k: usize,
    /// Chained device-token-loop replays run this session (each advanced the
    /// device one token without a host argmax→next-token round-trip).
    pub device_token_loop_steps: u64,
    /// Verify-slot (fixed-M option-c MTP verify) CUDA-graph captures.
    pub verify_captures: u64,
    /// Verify-slot replays — the proof the M=K verify graph is reused across
    /// steps instead of recaptured every verify (the pre-#1650 replays=0 pin).
    pub verify_replays: u64,
    /// Verify-slot capture declines (forward ran eagerly).
    pub verify_fallbacks: u64,
    /// Verify-slot invalidations (a binding shape/pointer change retired the
    /// captured verify graph; it re-warms and re-captures).
    pub verify_invalidations: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DecodeCudaGraphPhase {
    NeedsWarmup,
    Armed,
    Ready,
    Unsupported,
}

/// Result of a host-logits ragged batch step (stage 3b, #750): one `[vocab]` f32
/// row per batch slot plus the device→host transfer cost of reading them, so a
/// caller can report the D2H price of the host-logits sampler seam instead of
/// hiding it.
pub struct RaggedLogitsStep {
    /// Per-row host logits (`logits[r]` is row `r`'s `[vocab]` distribution).
    pub logits: Vec<Vec<f32>>,
    /// Bytes transferred device→host for this step's logits read.
    pub d2h_bytes: usize,
    /// Wall time of the logits device→host copy.
    pub d2h_time: std::time::Duration,
}

pub(crate) struct DecodeCudaState {
    /// Number of sequences the persistent decode bindings are shaped for (KV /
    /// mask / input / position / logits batch axis). **Constructor-fixed to `1`
    /// today** (stage 2b-impl-1, #750): the value is threaded through the shape
    /// and IO-binding layer so the batch extent is no longer a hard-coded `1`
    /// literal at those sites, but every current entry point still constructs
    /// the state at batch 1, so decode remains byte-identical. The token-writer,
    /// KV growth/commit geometry, device argmax and batch-symbol capture pinning
    /// that a real batch-N step also needs are owned by later increments
    /// (2b-impl-2..4) and are *not* generalized here.
    batch: usize,
    logical_len: usize,
    /// Per-row logical KV length (stage 3a, #750): `row_lens[r]` is the number of
    /// tokens sequence `r` has committed to its KV slice. In a *uniform* batch
    /// every entry equals `logical_len`; a *ragged* batch lets rows sit at
    /// genuinely different lengths (some admitted/advanced more than others). The
    /// per-row length drives that row's attention-mask window (which keys it may
    /// attend) and its `position_ids` value; the model derives each row's
    /// `seqlens_k` from its own mask window and writes present KV at its own
    /// offset, so a shared physical KV buffer carries rows of different lengths.
    /// `logical_len` remains the shared *physical* extent (`max(row_lens)`) used
    /// for capacity/KV-shape bookkeeping. Length `batch`, reset to `0` on
    /// `rewind(0)`.
    row_lens: Vec<usize>,
    /// Per-row admission state (stage 3b, #750). `row_active[r]` is `true` while
    /// row `r` holds a live sequence and `false` once it has been retired by
    /// [`Self::deactivate_row`] and is available for backfill by
    /// [`Self::assign_row`]. Every row starts `true` so the uniform/ragged batch
    /// entry points (which never call assign/deactivate) behave exactly as before
    /// — the whole batch is active from construction. Only the continuous-batch
    /// seam toggles this to swap a finished row's occupant mid-flight while its
    /// peers keep their captured graph. This is host-side bookkeeping; it changes
    /// no device binding shape, so toggling it never invalidates the captured
    /// decode graph.
    row_active: Vec<bool>,
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
    pub(crate) capacity: CudaKvCapacity,
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
    /// **Dormant seam (default `false`).** When `true`, the captured M=1 decode
    /// graph would be retained across a speculative verify+commit cycle instead
    /// of being torn down twice per step (once by the eager M>1 verify forward,
    /// once by the commit rewind), the two invalidations that pin MTP at
    /// `replays=0` (empirically 30 verify + 21 rewind invalidations over 10
    /// verify steps).
    ///
    /// Kept OFF because enabling it is **not capture-safe** as-is: the eager M>1
    /// verify reserves a larger StepScoped `step_workspace` that is freed after
    /// the run, so a later M=1 replay reads the captured graph's now-stale
    /// workspace pointer and yields non-finite logits (GPU-verified; the finite
    /// guard catches it — no silent corruption). Retaining across the rewind
    /// alone, or across the verify alone, is safe but useless (the other site
    /// still tears the graph down, so `replays` stays 0); retaining across both —
    /// the only config that actually replays — is what corrupts. A real speedup
    /// requires the M=K verify itself captured into a fixed-shape replayable
    /// graph with a pinned workspace (option-c padded verify capture), since the
    /// eager verify otherwise pays the full per-op launch overhead that graphed
    /// greedy avoids. See the decision note for the GPU evidence and plan.
    pub(crate) retain_decode_graph_across_spec: bool,
    /// Dormant option (c) scaffolding: the fixed query-row capacity (M=maxK) a
    /// padded single-capture verify graph would be captured at. `None` today —
    /// the eager verify path (option (b)) captures nothing. Set only by the
    /// dormant `configure_padded_verify_capture` switch (not on the hot path).
    #[cfg(test)]
    pub(crate) padded_query_capacity: Option<usize>,
    /// Device-resident token-feedback loop (opt-in via `ONNX_GENAI_DEVICE_TOKEN_LOOP`).
    /// `0` disables it; `k > 0` chains `k` captured decode replays back-to-back
    /// with a device token-writer stitched between them, so the host leaves the
    /// per-step critical path and drains `k` selected token ids in one D2H read.
    /// Only armed when the topology is device-loopable (see `device_token_loop_ready`).
    device_token_loop_k: usize,
    /// `true` when the persistent-decode topology supports the device token loop:
    /// batch 1, graph capture engaged, the attention-mask frozen to physical
    /// `max_len` (not logical-exposed), an i64 `input_ids`/mask, and — when the
    /// model exposes a `position_ids` binding — a rank-1 i64 one. When `false`
    /// the loop stays off regardless of `device_token_loop_k`.
    device_token_loop_ready: bool,
    /// `true` when the model exposes a persistent `position_ids` binding that the
    /// device token-writer must advance each step; `false` when position is
    /// derived from the attention mask alone (no position binding — e.g. GLM-4),
    /// in which case the writer only folds the token id and mask bit.
    device_token_loop_write_position: bool,
    /// Scratch device binding for the token loop: `k` u32 token-log slots
    /// followed by one u32 capture-error accumulator word (index `k`). The
    /// device token-writer appends each step's selected token and ORs the shared
    /// capture-error word here; the host drains all `k + 1` words in one D2H per
    /// chain. `None` unless the loop is armed.
    device_token_loop_scratch: Option<DeviceIoBinding>,
    /// Count of device-token-loop chained replays actually run (each advances the
    /// device one token without a host round-trip). A diagnostics counter.
    device_token_loop_steps: u64,
    /// Option-c native MTP verify capture (WP4). `Some(M)` once the fixed
    /// verify width M = k+1 (num_speculative_tokens + 1) is configured; the
    /// padded verify bindings below are allocated at that width and the M=K
    /// verify forward captures into the main executor's Verify graph slot.
    /// `None` keeps the eager verify path (byte-identical legacy behavior).
    verify_width: Option<usize>,
    /// Capture phase of the fixed-M verify graph, driven independently of the
    /// M=1 `graph_phase`/`inline_graph_phase` because it lives in a *separate* EP
    /// graph slot (Verify) on the main executor. `NeedsWarmup` and dormant until
    /// `verify_width` is configured.
    verify_graph_phase: DecodeCudaGraphPhase,
    /// Persistent padded `[1, M]` token-id binding for the verify forward. Its
    /// device pointer is stable across steps so the captured Verify graph replays
    /// against a fixed address. `None` until verify capture is configured.
    verify_token_binding: Option<DeviceIoBinding>,
    /// Persistent padded `[1, M]` position-ids binding for the verify forward
    /// (present only when the decoder declares a position input). Stable pointer.
    verify_position_binding: Option<DeviceIoBinding>,
    /// Persistent padded `[1, M, vocab]` logits binding for the verify forward.
    /// The production logits binding is single-token `[1, 1, vocab]`, so an M=K
    /// forward's `[1, M, vocab]` logits would materialize to host (vetoing
    /// capture); a device-resident padded binding lands them on-device with a
    /// stable pointer. `None` until configured.
    verify_logits_binding: Option<DeviceIoBinding>,
    /// Persistent padded `[1, M, ...]` bindings for every auxiliary graph output
    /// (e.g. the MTP hidden seed `hidden_states.63`), in `auxiliary_binding_range`
    /// order. An M=K forward produces `[1, M, ...]` aux outputs that do not fit
    /// the `[1, 1, ...]` decode aux bindings; without a padded device binding
    /// they materialize to host and veto capture. The verify discards these
    /// (the proposer reads its hidden seed from the M=1 base decode), but they
    /// must stay device-resident for the graph to be capturable. Empty until
    /// configured or when the decoder declares no auxiliary outputs.
    verify_aux_bindings: Vec<DeviceIoBinding>,
    /// Diagnostics: captures / replays of the fixed-M verify graph (the Verify
    /// slot), tracked separately from the Primary M=1 counters so both slots'
    /// reuse can be proven independently.
    verify_graph_captures: u64,
    verify_graph_replays: u64,
    verify_graph_fallbacks: u64,
    verify_graph_invalidations: u64,
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

/// The narrowest query width worth padding a prefill forward up to.
///
/// Below this the forward is so cheap that the padded rows cost more than the
/// recompile they avoid.
const MIN_PREFILL_QUERY_WIDTH: usize = 16;

/// How many distinct query widths a prefill is allowed to use, from the chunk
/// width down.
///
/// The kernel cache keeps a bounded number of compiled variants per node and
/// evicts the rest, so a shape set larger than that bound thrashes instead of
/// caching. Raising the bound is not the answer — a retained variant owns device
/// scratch, and a bound wide enough for eight prefill widths pushed a
/// 5.5k-token prompt past the mapped-memory ceiling. So the ladder is sized to
/// fit the bound that exists: three prefill widths, leaving one slot for the
/// single-token decode shape.
///
/// Keeping it above one is what stops a short prompt from paying for a full
/// chunk of arithmetic.
const PREFILL_QUERY_WIDTH_STEPS: usize = 3;

/// The query width a multi-row forward should actually run at.
///
/// A forward is dispatched through a kernel cache keyed by its concrete input
/// shapes, so a query width nobody has run before recompiles every node in the
/// graph — for a 30B decoder, on the order of a thousand kernels. Prompt lengths
/// are effectively unique per request, so left alone every request pays that
/// bill, and the compiled variants pile up as device workspace nobody accounts
/// for (#1362).
///
/// Rounding the width up to a power of two collapses that unbounded set to a
/// handful of shapes shared by every request, at the cost of at most one
/// duplicated row of arithmetic for each real one. It is the same bargain the
/// KV cache already takes: GQA binds `past_key`/`past_value` at a fixed physical
/// capacity and carries the true length in `seqlens_k`, rather than resizing the
/// cache to the exact sequence and reshaping the graph every step.
///
/// `cap` is the declared prefill chunk width, which is the largest width this
/// model is willing to run at; a forward wider than that is left alone, since it
/// already amortizes its compile over enough work to not matter.
/// Whether query-axis prefill padding is enabled at all.
///
/// Padding trades duplicated arithmetic for a smaller set of compiled shapes,
/// and which side wins depends on the model and the request mix. The switch
/// exists so the trade can be measured against the same binary rather than
/// argued about.
fn prefill_query_padding_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        !matches!(
            std::env::var("ONNX_GENAI_PREFILL_QUERY_PADDING")
                .unwrap_or_default()
                .as_str(),
            "0" | "off" | "false"
        )
    })
}

fn prefill_query_width(rows: usize, cap: usize) -> usize {
    if rows <= 1 || cap == 0 || rows > cap {
        return rows;
    }
    // Even steps rather than powers of two: a power-of-two ladder rounds 311
    // rows up to 512, and the 65% of duplicated arithmetic that buys costs more
    // than the recompile it saves. Even steps up to the chunk width bound the
    // shape set just as well while capping the waste at one step.
    // Round the step up, not down, so the last step lands exactly on the chunk
    // width instead of just short of it and leaving a stray fourth width above.
    let step = cap
        .div_ceil(PREFILL_QUERY_WIDTH_STEPS)
        .max(MIN_PREFILL_QUERY_WIDTH);
    rows.div_ceil(step).saturating_mul(step).min(cap).max(rows)
}

/// Extend a `[1, rows, …]` per-token step tensor to `target_rows` rows by
/// repeating its last row.
///
/// The duplicated rows are never read: their outputs are dropped and the KV they
/// write sits past the sequence's logical length, where the next forward
/// overwrites it. Repeating a real row rather than zeroing keeps the padded
/// values in the same numeric range as the real ones, so the padding cannot be
/// what makes a forward overflow.
/// A multi-row forward widened to a padded query width.
struct PaddedPrefill {
    /// The prompt tokens followed by repeats of the last one.
    tokens: Vec<TokenId>,
    /// The supplied per-token ports, each widened to match.
    step_inputs: Vec<(String, Tensor)>,
}

fn pad_step_tensor(tensor: &Tensor, rows: usize, target_rows: usize) -> Option<Tensor> {
    if target_rows <= rows || rows == 0 {
        return None;
    }
    let [1, tensor_rows, rest @ ..] = tensor.shape.as_slice() else {
        return None;
    };
    if *tensor_rows != rows {
        return None;
    }
    let bytes = tensor.as_bytes();
    let row_bytes = bytes.len().checked_div(rows)?;
    if row_bytes == 0 || row_bytes * rows != bytes.len() {
        return None;
    }
    let mut padded = Vec::with_capacity(row_bytes * target_rows);
    padded.extend_from_slice(bytes);
    let last_row = &bytes[bytes.len() - row_bytes..];
    for _ in rows..target_rows {
        padded.extend_from_slice(last_row);
    }
    let mut shape = vec![1, target_rows];
    shape.extend_from_slice(rest);
    Tensor::from_raw(tensor.dtype, shape, &padded).ok()
}

impl NativeDecodeSession {
    /// The query width this multi-row forward should run at, and the padded
    /// token ids to run it with.
    ///
    /// Returns `None` when the forward must run at its exact width. Padding is
    /// refused for a decoder carrying recurrent or convolutional state: that
    /// state is not masked and not addressed by a logical length, so a
    /// duplicated row would advance it past the real sequence and there is
    /// nowhere to put the extra step back. Attention KV has no such problem —
    /// the padded rows land beyond the logical length, where the next forward
    /// overwrites them.
    fn padded_prefill_plan(&self, token_ids: &[TokenId]) -> Option<(Vec<TokenId>, usize)> {
        if !self.prefill_query_padding
            || !prefill_query_padding_enabled()
            || self.has_recurrent_state()
        {
            return None;
        }
        let cap = self.prefill_chunk_size.map(NonZeroUsize::get).unwrap_or(0);
        let width = prefill_query_width(token_ids.len(), cap);
        if width == token_ids.len() {
            return None;
        }
        let mut padded = Vec::with_capacity(width);
        padded.extend_from_slice(token_ids);
        let last = *token_ids.last()?;
        padded.resize(width, last);
        Some((padded, width))
    }

    /// The padded token ids and per-step tensors a multi-row forward should run
    /// with, or `None` to run at the exact width.
    ///
    /// Every supplied per-token tensor has to be paddable for the plan to hold:
    /// a port whose rows this code cannot recognize (anything not shaped
    /// `[1, rows, …]`) might not be per-token at all, and inventing rows for it
    /// would change what the forward computes.
    fn padded_prefill_inputs(
        &self,
        token_ids: &[TokenId],
        step_inputs: &[(String, Tensor)],
    ) -> Option<PaddedPrefill> {
        let (tokens, width) = self.padded_prefill_plan(token_ids)?;
        let mut padded_inputs = Vec::with_capacity(step_inputs.len());
        for (name, tensor) in step_inputs {
            let padded = pad_step_tensor(tensor, token_ids.len(), width)?;
            padded_inputs.push((name.clone(), padded));
        }
        Some(PaddedPrefill {
            tokens,
            step_inputs: padded_inputs,
        })
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
        // A generation runs *two* forward shapes against the same session with
        // distinct governed-workspace lifetimes: the multi-token prefill/verify
        // shape here, and the steady-state single-token decode step. The
        // prepare-only planner reserves one workspace slot per *lifetime class*
        // from the shapes it is handed, and default-domain `Attention` classifies
        // its composite workspace by route — multi-row query (`q_seq > 1`) is the
        // per-call StepScoped route, single-row query (`batch == 1, q_seq == 1`)
        // is the capture-eligible SessionPersistent route. Preparing only the
        // multi-row prefill shape therefore reserves the StepScoped slot and
        // leaves the SessionPersistent slot empty, so the first single-token
        // decode reaches the `Attention` node with an unprepared
        // SessionPersistent workspace and fails (#1179). GQA/MoE sidestep this by
        // charging SessionPersistent for both routes, but MHA does not.
        //
        // Reserve the decode route's SessionPersistent slot up front by driving
        // one additional single-query-row pass. Workspace planning now binds the
        // growing KV cache at its physical capacity (see
        // `prepare_external_bindings_mode`), so this single-token pass reserves a
        // SessionPersistent workspace sized for the full-capacity attended
        // length — a valid upper bound for every single-token decode step until
        // the KV cache rebuckets. `reserve_prepared_workspace` only ever grows a
        // slot, so a model that already reserved a larger SessionPersistent slot
        // for the prefill shape (GQA/MoE) sees this as a no-op; MHA needs the
        // top-up. The subsequent prefill forward re-establishes the KV/mask
        // device state before it executes, so the decode pass's residual
        // single-token mask is not observed.
        // Plan at the width the forward will actually run at: a padded prefill
        // reserves workspace for its padded query rows, not its real ones.
        let padded = self.padded_prefill_inputs(token_ids, step_inputs);
        let (token_ids, step_inputs) = match &padded {
            Some(padded) => (padded.tokens.as_slice(), padded.step_inputs.as_slice()),
            None => (token_ids, step_inputs),
        };
        let prefill = self.run_cuda_workspace_prepare_pass(token_ids, past_len, step_inputs)?;
        if token_ids.len() > 1 {
            let last_token = &token_ids[token_ids.len() - 1..];
            self.run_cuda_workspace_prepare_pass(last_token, 0, &[])
                .context("prepare native CUDA single-token decode workspace")?;
        }
        Ok(prefill)
    }

    /// Resolve concrete metadata for one forward shape (`token_ids` at
    /// `past_len`, plus any routed `step_inputs`) and reserve the governed
    /// kernel workspace slots for it without executing any graph node. Shared by
    /// the multi-token prefill/verify reservation and the single-token decode
    /// top-up so both bind symbols and size workspaces through the identical
    /// path.
    fn run_cuda_workspace_prepare_pass(
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
        // A multi-row forward is keyed in the kernel cache by its concrete input
        // shapes, so a query width nobody has run before recompiles every node in
        // the graph. Report the width and the compile count it cost, since that
        // is the only place the cost of a novel prefill shape is observable.
        let compiles_before = self.session.cache_stats().misses;
        // The token axis is dim 1 for both `[1, rows]` token ids and
        // `[1, rows, hidden]` embeddings; the last dim would report the hidden
        // size for the latter.
        let query_rows = owned
            .first()
            .and_then(|(_, tensor)| tensor.shape.get(1).copied())
            .unwrap_or(0);
        let state = self
            .cuda
            .as_mut()
            .context("CUDA decode state is not initialized")?;
        // An eager M>1 verify/prefill forward tears down the captured M=1 decode
        // graph here so the plain M=1 hot path re-warms cleanly — EXCEPT once
        // option-c verify capture is active, where each graph lives in its own
        // per-slot host capture state and must not be disturbed by the other.
        if state.verify_width.is_some() {
            // Option-c verify capture is active. With per-slot host capture state
            // the main executor holds the M=1 decode graph in its `Primary` slot
            // and the fixed-M verify graph in its `Verify` slot; `Primary` is the
            // current slot on this eager path (the base decode / prefill runs
            // here), so we must NOT invalidate — that would tear down the M=1
            // decode graph every verify step, which is exactly what pinned MTP at
            // replays=0 before option-c. The captured verify graph is owned by
            // `run_verify_captured` on the `Verify` slot and is untouched here.
            // This eager forward is either the one-time prefill or a tail verify
            // at width != M; a plain eager run leaves both installed graphs intact
            // so the next step still replays the matching one.
        } else if !state.retain_decode_graph_across_spec {
            state.invalidate_graph(&mut self.session)?;
        }
        // Auxiliary graph outputs (e.g. the MTP hidden-state seed
        // `hidden_states.63`) get a persistent device binding whose symbolic
        // query-seq axis is collapsed to `1` for the captured decode step
        // (`persistent_output_shape`). A multi-row eager forward (`m > 1`
        // prefill/verify) produces `[1, m, hidden]`, which cannot fit that
        // `[1, 1, hidden]` binding, so we EXCLUDE the auxiliary tail from the
        // device-binding slice here and let those outputs materialize to host
        // this step — mirroring how `logits` (bound past `base_binding_count`)
        // already materializes. The captured single-token path keeps the
        // persistent binding (it produces `m == 1`), preserving capture-safety.
        // For pure-attention models with no auxiliary outputs this range is
        // empty and the slice is byte-identical to `..base_binding_count`.
        let aux_start = state.auxiliary_binding_range.start;
        let aux_names = state.bindings[state.auxiliary_binding_range.clone()]
            .iter()
            .filter_map(|binding| binding.output_name().map(str::to_owned))
            .collect::<HashSet<_>>();
        let bindings = owned
            .iter()
            .map(|(name, tensor)| (name.as_str(), tensor))
            .collect::<Vec<_>>();
        let outputs = match self
            .session
            .run_with_device_bindings(&bindings, &mut state.bindings[..aux_start])
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
        // The declared hidden seed (if any) materialized to host because it is an
        // auxiliary output; record its last row so the MTP proposer can read it
        // via `last_hidden()`. Inert when no hidden output is declared.
        let hidden = self
            .hidden_output
            .as_ref()
            .and_then(|name| named.remove(name.as_str()));
        // Every remaining materialized output must be an auxiliary output we
        // deliberately excluded from the device slice; a non-auxiliary
        // materialization signals a real binding bug and must still fail.
        if let Some(unexpected) = named.keys().find(|name| !aux_names.contains(name.as_str())) {
            bail!(
                "native CUDA {error_context} unexpectedly materialized bound output '{unexpected}'"
            );
        }
        self.last_hidden = match (&self.hidden_output, hidden) {
            (Some(name), Some(tensor)) => Some(
                extract_last_row(&tensor)
                    .with_context(|| format!("read native decoder hidden output '{name}'"))?,
            ),
            (Some(name), None) => {
                bail!("native decoder omitted declared hidden output '{name}'")
            }
            (None, _) => None,
        };
        let logits = extract_logits(&logits)?;
        if logits.iter().flatten().any(|value| !value.is_finite()) {
            bail!("native decoder produced non-finite logits");
        }
        state.set_logical_len(total_len)?;
        self.current_len = total_len;
        let stats = self.session.cache_stats();
        tracing::debug!(
            query_rows,
            total_len,
            kernels_compiled = stats.misses - compiles_before,
            kernel_cache_entries = stats.entries,
            kernel_cache_evictions = stats.evictions,
            "native CUDA multi-row forward"
        );
        Ok(logits)
    }

    /// Option-c native MTP verify capture (WP4). Allocate the persistent padded
    /// `[1, M]` token / position, `[1, M, vocab]` logits and `[1, M, ...]`
    /// auxiliary device bindings the captured fixed-M verify forward needs, pin
    /// the main executor's StepScoped workspace at the M=K peak (capture-safe),
    /// and route the main executor's CUDA-graph slot to `Verify` so the captured
    /// verify graph never touches the M=1 decode graph (`Primary`, driven by the
    /// decode-inline sibling on a hybrid recurrent model).
    ///
    /// `M = width` is derived from the caller's `num_speculative_tokens + 1`; no
    /// hardcoded dims — every padded shape is widened from the live decode
    /// binding's own physical shape. A no-op (`verify_width` stays `None`, eager
    /// verify path in force) when: not a CUDA graph-enabled session, already
    /// configured, `width < 2`, the decoder needs eager per-step inputs
    /// (inputs_embeds / routed ports), the token input is not int64, a
    /// multi-axis (mrope) position binding is present, the model carries no
    /// recurrent state (so decode is NOT routed to the inline sibling and the
    /// main-exec Primary slot is still the decode graph — retargeting it would
    /// break greedy), or any logits/aux output does not collapse query-seq to a
    /// unit `len-2` axis.
    pub(crate) fn configure_verify_capture(&mut self, width: usize) -> anyhow::Result<()> {
        if width < 2 || self.has_eager_step_inputs() || !self.has_recurrent_state() {
            return Ok(());
        }
        // Capturing the M=K verify into the `Verify` graph slot is now sound even
        // when the interleaved M=1 base decode / commit re-advance run on the
        // SAME (main) executor. The executor holds per-slot host capture state
        // (device_graph_signature + schedule + warm-shape/quarantine, indexed by
        // graph_slot), so an M=1 decode captured into the `Primary` slot no
        // longer clobbers the `Verify` slot's signature between the verify's
        // capture and its replay. Both graphs coexist and each replays by shape
        // key. This lifts the earlier `enable_decode_inline()` gate: on models
        // whose recurrent Scan is not single-trip-inlineable (no decode-inline
        // sibling — e.g. this artifact) the M=1 decode simply runs on the main
        // executor's `Primary` slot while the verify runs on its `Verify` slot.
        // Gather the padded binding plan from the live decode bindings (immutable
        // borrow), then allocate (session borrow), then store (state borrow).
        struct VerifyBindingPlan {
            token_name: String,
            token_dtype: DataType,
            token_shape: Vec<usize>,
            position: Option<(String, DataType, Vec<usize>)>,
            logits_name: String,
            logits_dtype: DataType,
            logits_shape: Vec<usize>,
            aux: Vec<(String, DataType, Vec<usize>)>,
        }
        let plan = {
            let Some(state) = self.cuda.as_ref() else {
                return Ok(());
            };
            if !state.graph_enabled || state.verify_width.is_some() {
                return Ok(());
            }
            if state.position_ids_binding.is_some() && state.position_rank != 1 {
                return Ok(());
            }
            let token = &state.bindings[state.input_ids_binding];
            if token.dtype != DataType::Int64 {
                return Ok(());
            }
            let token_name = token.input_name().to_string();
            let token_dtype = token.dtype;
            let Some(token_shape) = widen_query_last(token.physical_shape(), width) else {
                return Ok(());
            };
            let position = match state.position_ids_binding {
                Some(idx) => {
                    let binding = &state.bindings[idx];
                    let Some(shape) = widen_query_last(binding.physical_shape(), width) else {
                        return Ok(());
                    };
                    Some((binding.input_name().to_string(), binding.dtype, shape))
                }
                None => None,
            };
            let logits = &state.bindings[state.logits_binding];
            let logits_name = logits
                .output_name()
                .context("verify capture: logits binding has no output name")?
                .to_string();
            let Some(logits_shape) = widen_query_seq(logits.physical_shape(), width) else {
                return Ok(());
            };
            let mut aux = Vec::new();
            for idx in state.auxiliary_binding_range.clone() {
                let binding = &state.bindings[idx];
                let name = binding
                    .output_name()
                    .context("verify capture: auxiliary binding has no output name")?
                    .to_string();
                let Some(shape) = widen_query_seq(binding.physical_shape(), width) else {
                    return Ok(());
                };
                aux.push((name, binding.dtype, shape));
            }
            VerifyBindingPlan {
                token_name,
                token_dtype,
                token_shape,
                position,
                logits_name,
                logits_dtype: state.logits_dtype,
                logits_shape,
                aux,
            }
        };

        let token_binding = self.session.allocate_device_binding(
            plan.token_name,
            None::<String>,
            plan.token_dtype,
            plan.token_shape.clone(),
            plan.token_shape,
        )?;
        let position_binding = match plan.position {
            Some((name, dtype, shape)) => Some(self.session.allocate_device_binding(
                name,
                None::<String>,
                dtype,
                shape.clone(),
                shape,
            )?),
            None => None,
        };
        let logits_binding = self.session.allocate_device_output_binding(
            plan.logits_name,
            plan.logits_dtype,
            plan.logits_shape.clone(),
            plan.logits_shape,
        )?;
        let mut aux_bindings = Vec::with_capacity(plan.aux.len());
        for (name, dtype, shape) in plan.aux {
            aux_bindings.push(self.session.allocate_device_output_binding(
                name,
                dtype,
                shape.clone(),
                shape,
            )?);
        }

        let state = self
            .cuda
            .as_mut()
            .context("CUDA decode state is not initialized")?;
        state.verify_token_binding = Some(token_binding);
        state.verify_position_binding = position_binding;
        state.verify_logits_binding = Some(logits_binding);
        state.verify_aux_bindings = aux_bindings;
        state.verify_width = Some(width);
        state.verify_graph_phase = DecodeCudaGraphPhase::NeedsWarmup;

        // Pin the main executor's StepScoped workspace at the M=K peak so the
        // captured verify graph replays against a stable scratch pointer even
        // though the interleaved M=1 decode step reserves a smaller scratch
        // (#1647). The graph slot is NOT set here: the main executor stays on
        // `Primary` for M=1 decode and `run_verify_captured` flips it to `Verify`
        // only around the verify forward, then back — safe because the executor's
        // per-slot host capture state keeps both graphs' signatures independent.
        self.session.set_main_exec_pin_step_workspace(true);
        Ok(())
    }

    /// True once [`Self::configure_verify_capture`] has installed the fixed-M
    /// verify capture (the padded bindings + Verify graph slot are live).
    #[allow(dead_code)] // diagnostics accessor; also exercised by verify_capture_helper_tests
    pub(crate) fn verify_capture_active(&self) -> bool {
        self.cuda
            .as_ref()
            .is_some_and(|state| state.verify_width.is_some())
    }

    /// The configured fixed verify width M (= k+1), or `None` when verify capture
    /// is not active.
    pub(crate) fn verify_capture_width(&self) -> Option<usize> {
        self.cuda.as_ref().and_then(|state| state.verify_width)
    }

    /// Run the fixed-M verify forward through the captured `Verify` graph slot
    /// (option-c): swap the persistent padded bindings into the device binding
    /// vector, drive the warm-up → arm(capture) → replay state machine on the
    /// main executor's Verify slot, read the `[1, M, vocab]` logits into `M`
    /// host rows, then swap the M=1 decode bindings back. Only called when
    /// `token_ids.len()` equals the configured `M` (the caller falls back to the
    /// eager path otherwise). Advances the KV logical length to `total_len`; the
    /// destructive recurrent/conv advance it performs is discarded by the
    /// driver's snapshot→restore→re-advance commit.
    fn run_verify_captured(
        &mut self,
        token_ids: &[TokenId],
        past_len: usize,
        total_len: usize,
    ) -> anyhow::Result<Vec<Vec<f32>>> {
        let m = token_ids.len();
        let _verify_span = self
            .trace
            .span("native_decode_verify_captured", "spec")
            .with_args(Args::new().with("rows", m as u64));
        // Write the padded token ids + positions into their persistent device
        // bindings (stable pointers, so the captured graph replays against them).
        {
            let state = self
                .cuda
                .as_mut()
                .context("CUDA decode state is not initialized")?;
            let mut token_bytes = Vec::with_capacity(m * 8);
            for &id in token_ids {
                token_bytes.extend_from_slice(&i64::from(id).to_ne_bytes());
            }
            state
                .verify_token_binding
                .as_mut()
                .context("verify capture: missing padded token binding")?
                .write_bytes(0, &token_bytes)?;
            if let Some(position) = state.verify_position_binding.as_mut() {
                let mut position_bytes = Vec::with_capacity(m * 8);
                for pos in past_len..total_len {
                    position_bytes.extend_from_slice(&(pos as i64).to_ne_bytes());
                }
                position.write_bytes(0, &position_bytes)?;
            }
        }

        // Swap the padded bindings into the persistent vector, retarget the main
        // executor to the Verify slot, run the verify graph phase (which reads
        // the logits while swapped in), then restore the Primary slot + swap the
        // M=1 decode bindings back — unconditionally, even on error. The slot
        // flip is a pure retarget (per-slot host capture state), so the M=1
        // decode graph in Primary is untouched and still replays next step.
        self.cuda
            .as_mut()
            .context("CUDA decode state is not initialized")?
            .swap_verify_bindings();
        self.session
            .set_main_exec_graph_slot(DeviceGraphSlot::Verify)?;
        let outcome = self.run_verify_graph_phase(m);
        self.session
            .set_main_exec_graph_slot(DeviceGraphSlot::Primary)?;
        self.cuda
            .as_mut()
            .context("CUDA decode state is not initialized")?
            .swap_verify_bindings();
        let logits = outcome?;

        let state = self
            .cuda
            .as_mut()
            .context("CUDA decode state is not initialized")?;
        state.set_logical_len(total_len)?;
        self.current_len = total_len;
        Ok(logits)
    }

    /// Drive the Verify-slot capture state machine for one fixed-M verify forward
    /// (assumes the padded bindings are already swapped into `state.bindings`),
    /// then read the `[1, M, vocab]` logits into `m` host rows. Mirrors
    /// [`DecodeCudaState::run_one_token`] but on the Verify slot with its own
    /// `verify_graph_phase` and counters.
    /// Decide the verify-graph phase after a captured verify graph is retired by
    /// a host-signature clobber. The first couple of clobbers re-warm (in case
    /// the churn was a one-off KV-bucket growth that self-stabilizes), but a
    /// persistent clobber — the single-executor two-slot contention this build
    /// cannot yet resolve — latches to `Unsupported` so the verify runs plain
    /// eager with no recapture overhead (never worse than the pre-option-c path).
    fn verify_phase_after_invalidation(invalidations: u64) -> DecodeCudaGraphPhase {
        const MAX_VERIFY_RECAPTURES: u64 = 2;
        if invalidations >= MAX_VERIFY_RECAPTURES {
            DecodeCudaGraphPhase::Unsupported
        } else {
            DecodeCudaGraphPhase::NeedsWarmup
        }
    }

    fn run_verify_graph_phase(&mut self, m: usize) -> anyhow::Result<Vec<Vec<f32>>> {
        let state = self
            .cuda
            .as_mut()
            .context("CUDA decode state is not initialized")?;
        if !state.graph_enabled {
            self.session
                .run_with_device_bindings(&[], &mut state.bindings[..])?;
        } else {
            match state.verify_graph_phase {
                DecodeCudaGraphPhase::NeedsWarmup => {
                    // Warm the M=K shape (reserves + — because the main exec is
                    // pinned — retains the StepScoped scratch) so the capture
                    // below performs no device alloc/free inside the region.
                    self.session
                        .run_with_device_bindings(&[], &mut state.bindings[..])?;
                    state.verify_graph_phase = DecodeCudaGraphPhase::Armed;
                }
                DecodeCudaGraphPhase::Armed => {
                    match self
                        .session
                        .try_capture_with_device_bindings(&[], &mut state.bindings[..])?
                    {
                        DeviceGraphCaptureResult::Captured(outputs) => {
                            if outputs.iter().any(Option::is_some) {
                                bail!(
                                    "captured CUDA verify forward unexpectedly materialized a host output"
                                );
                            }
                            state.verify_graph_captures += 1;
                            state.verify_graph_phase = DecodeCudaGraphPhase::Ready;
                            tracing::debug!(
                                rows = m,
                                captures = state.verify_graph_captures,
                                "native CUDA verify graph captured"
                            );
                        }
                        DeviceGraphCaptureResult::NotCapturable(report) => {
                            state.verify_graph_fallbacks += 1;
                            state.verify_graph_phase = DecodeCudaGraphPhase::Unsupported;
                            trace_capture_declines(&self.trace, &report);
                            let reason = report.to_string();
                            state.graph_decline_reason = Some(reason.clone());
                            tracing::warn!(
                                "native CUDA verify graph capture disabled for this generation: {reason}"
                            );
                            self.session
                                .run_with_device_bindings(&[], &mut state.bindings[..])?;
                        }
                    }
                }
                DecodeCudaGraphPhase::Ready => {
                    // The main executor now holds PER-SLOT host capture state, so
                    // the interleaved M=1 decode (Primary slot) no longer clobbers
                    // this M=K verify graph (Verify slot): each slot keeps its own
                    // signature/schedule and replays independently. A replay can
                    // still legitimately retire the graph — e.g. the pinned
                    // StepScoped workspace grew on the first verify and moved its
                    // scratch pointer, or a control-flow branch flip — so we keep
                    // the graceful path: count the invalidation, run eagerly to
                    // still produce this step's logits, re-warm once (self-
                    // stabilizing after the workspace settles at the M=K peak),
                    // and only latch to permanent-eager if churn persists.
                    match self.session.replay_device_graph(&mut state.bindings[..]) {
                        Ok(true) => {
                            state.verify_graph_replays += 1;
                        }
                        Ok(false) => {
                            state.verify_graph_replays += 1;
                            state.verify_graph_invalidations += 1;
                            state.verify_graph_phase =
                                Self::verify_phase_after_invalidation(state.verify_graph_invalidations);
                            self.session
                                .run_with_device_bindings(&[], &mut state.bindings[..])?;
                        }
                        Err(_) => {
                            state.verify_graph_invalidations += 1;
                            state.verify_graph_phase =
                                Self::verify_phase_after_invalidation(state.verify_graph_invalidations);
                            self.session
                                .run_with_device_bindings(&[], &mut state.bindings[..])?;
                        }
                    }
                }
                DecodeCudaGraphPhase::Unsupported => {
                    self.session
                        .run_with_device_bindings(&[], &mut state.bindings[..])?;
                }
            }
        }

        // Read the padded [1, M, vocab] logits (swapped into the logits slot).
        let vocab = *state
            .logits_shape
            .last()
            .context("CUDA logits shape has no vocabulary dimension")?;
        let bytes = state.bindings[state.logits_binding].read_bytes()?;
        let tensor = Tensor::from_raw(state.logits_dtype, vec![1, m, vocab], &bytes)?;
        let logits = extract_logits(&tensor)?;
        if logits.iter().flatten().any(|value| !value.is_finite()) {
            bail!("native CUDA verify forward produced non-finite logits");
        }
        Ok(logits)
    }

    /// THROWAWAY Lever-B Phase-0 capture-stability probe (leverb-phase0).
    ///
    /// NOT wired into any decode path — invoked only by the `#[ignore]`d
    /// `leverb_phase0_probe` test. It attempts to CUDA-graph capture a single
    /// fixed-shape `[1, m]` (padded M=K) forward against the existing persistent
    /// KV/mask/logits bindings and, if the capture instantiates, replays it
    /// `replays` times, timing each replay wall (the replay is synchronized by a
    /// D2H read of the logits binding, mirroring the real per-step logits sync).
    ///
    /// This measures Phase-0 pass criteria (a) "instantiates capture-safe" and
    /// (c) "per-verify replay wall" on the REAL batched GQA/GEMM (`MatMulNBits`)
    /// kernels — it does not hand-roll a toy graph. It deliberately does NOT
    /// commit KV or advance the logical length: the captured/replayed forward
    /// dirties device KV, and KV-commit correctness is explicitly out of Phase-0
    /// scope, so the caller MUST discard the session afterwards.
    #[cfg(all(test, feature = "native-cuda"))]
    pub(crate) fn leverb_phase0_capture_attempt(
        &mut self,
        m: usize,
        replays: usize,
    ) -> anyhow::Result<LeverBPhase0CaptureAttempt> {
        let past_len = self.current_len;
        let total_len = past_len
            .checked_add(m)
            .context("leverb phase0 probe length overflow")?;
        let token_input = self
            .step_input_name(NativeStepInputSource::TokenIds)
            .context("native CUDA decoder has no token input binding")?
            .to_owned();
        let position_input = self
            .step_input_name(NativeStepInputSource::PositionIds)
            .map(str::to_owned);

        // Build the fixed padded `[1, m]` token/position inputs. Token *values*
        // are irrelevant to Phase-0 (acceptance/KV correctness is out of scope);
        // a constant id keeps the shape fixed so a captured graph can replay
        // without re-instantiation.
        let ids = vec![1_i64; m];
        let input_ids = Tensor::from_i64(&[1, m], &ids)?;
        let mut owned: Vec<(String, Tensor)> = Vec::with_capacity(2);
        owned.push((token_input, input_ids));
        if let Some(position_ids_name) = position_input {
            owned.push((
                position_ids_name,
                self.build_step_positions(past_len, total_len)?,
            ));
        }

        let state = self
            .cuda
            .as_mut()
            .context("CUDA decode state is not initialized")?;
        if total_len > state.capacity.max_len {
            bail!("{}", state.capacity_exceeded_error(total_len));
        }
        // Keep the whole probe inside one physical bucket: growth is a capture
        // boundary and is exercised separately by the M=1 stability loop.
        let grew = state.ensure_capacity(&mut self.session, total_len)?;
        state.extend_mask(if grew { 0 } else { past_len }, total_len, total_len)?;
        // Start from a clean graph slot so the capture attempt is unambiguous.
        state.invalidate_graph(&mut self.session)?;

        let borrowed = owned
            .iter()
            .map(|(name, tensor)| (name.as_str(), tensor))
            .collect::<Vec<_>>();
        let base = state.base_binding_count;

        let alloc_before = self.session.device_allocation_counts();
        let capture = self
            .session
            .try_capture_with_device_bindings(&borrowed, &mut state.bindings[..base])?;
        let alloc_after = self.session.device_allocation_counts();
        let alloc_delta = match (alloc_before, alloc_after) {
            (Some(before), Some(after)) => Some((
                after.allocations.saturating_sub(before.allocations),
                after.frees.saturating_sub(before.frees),
            )),
            _ => None,
        };
        let segments = self.session.captured_graph_segment_count();

        let mut attempt = LeverBPhase0CaptureAttempt {
            rows: m,
            past_len,
            bucket: state.max_len,
            captured: false,
            segments,
            decline: None,
            alloc_delta,
            warm_alloc_delta: None,
            replay_walls_ns: Vec::new(),
            seam_summary: None,
            warm_argmax: Vec::new(),
            replay_argmax: Vec::new(),
            logits_byte_identical: None,
        };

        match capture {
            DeviceGraphCaptureResult::Captured(outputs) => {
                attempt.captured = true;
                if outputs.iter().any(Option::is_some) {
                    // A materialized (host-returned) output means the padded M=K
                    // logits do not fit the single-token logits binding; still a
                    // capture, but record it as a decline note for the findings.
                    attempt.decline = Some(
                        "captured, but the forward materialized a host output (padded logits do not fit the single-token logits binding)".to_string(),
                    );
                }
                for _ in 0..replays {
                    let start = std::time::Instant::now();
                    let still_valid = self
                        .session
                        .replay_device_graph(&mut state.bindings[..base])?;
                    // Force stream completion with a D2H read of the logits
                    // binding (the real hot path syncs on the per-step logits
                    // read); this makes the wall the GPU-inclusive replay time.
                    let _ = state.bindings[state.logits_binding].read_bytes()?;
                    attempt
                        .replay_walls_ns
                        .push(start.elapsed().as_nanos() as u64);
                    if !still_valid {
                        break;
                    }
                }
            }
            DeviceGraphCaptureResult::NotCapturable(report) => {
                attempt.decline = Some(report.to_string());
            }
        }
        // Leave the graph slot clean; state is intentionally left dirty.
        state.invalidate_graph(&mut self.session)?;
        Ok(attempt)
    }

    /// THROWAWAY Lever-B **Increment-0** capture attempt (leverb-increment0):
    /// the same real M=K forward as [`Self::leverb_phase0_capture_attempt`], but
    /// with the three capture-enablement fixes the Phase-0 (a)-FAIL identified,
    /// applied as a test-only overlay (NOT wired into the decode path):
    ///
    ///   1. **Persistent padded `[1, m, vocab]` logits binding.** The production
    ///      logits binding is single-token `[1, 1, vocab]`, so an M=K forward's
    ///      `[1, m, vocab]` logits do not fit and materialize to the host — which
    ///      makes capture reject ("every graph output must use a persistent
    ///      device binding"). A fresh padded output binding for the same logits
    ///      output name lands them in device memory instead.
    ///   2. **Alloc-free captured region via a pre-capture warm forward at the
    ///      M=K shape.** The capture-safe scratch arena is grown by a normal
    ///      `run_with_device_bindings` at `[1, m]` *before* `BeginCapture`, so the
    ///      captured region itself performs no device alloc/free (#854/#867).
    ///   3. **KV-symbol pin** — inherited: the constructor already pins the
    ///      fixed-capacity KV seq symbols for a graph-enabled session, so the
    ///      batched GQA attention nodes admit capture at any query width `m`.
    ///
    /// Like the Phase-0 probe this dirties device KV (KV-commit correctness is
    /// out of scope) and does NOT advance the logical length, so the caller MUST
    /// discard the session afterwards. It drives the real batched GQA/GEMM M=K
    /// kernels through the real persistent KV/mask bindings — it does not
    /// hand-roll a toy graph. The padded token/position/logits bindings are
    /// swapped into the persistent binding vector for the duration of the
    /// attempt and restored before returning.
    #[cfg(all(test, feature = "native-cuda"))]
    pub(crate) fn leverb_increment0_capture_attempt(
        &mut self,
        m: usize,
        replays: usize,
    ) -> anyhow::Result<LeverBPhase0CaptureAttempt> {
        self.leverb_increment0_capture_attempt_inner(m, replays, None, false)
    }

    /// Captured-vs-eager token-parity variant of the Increment-0 probe: writes
    /// the supplied `tokens` (cycled to fill the M rows) instead of a constant,
    /// records the EAGER warm-forward per-row argmax and the CAPTURED replay
    /// per-row argmax, and compares the raw logits bytes. This fills the
    /// "captured M=K == eager M=K, same Marlin config, no tiled oracle" cell.
    #[cfg(all(test, feature = "native-cuda"))]
    pub(crate) fn leverb_increment0_token_parity_attempt(
        &mut self,
        m: usize,
        tokens: &[i64],
    ) -> anyhow::Result<LeverBPhase0CaptureAttempt> {
        self.leverb_increment0_capture_attempt_inner(m, 1, Some(tokens), true)
    }

    #[cfg(all(test, feature = "native-cuda"))]
    fn leverb_increment0_capture_attempt_inner(
        &mut self,
        m: usize,
        replays: usize,
        tokens: Option<&[i64]>,
        collect_parity: bool,
    ) -> anyhow::Result<LeverBPhase0CaptureAttempt> {
        let past_len = self.current_len;
        let total_len = past_len
            .checked_add(m)
            .context("leverb increment0 probe length overflow")?;
        let token_input = self
            .step_input_name(NativeStepInputSource::TokenIds)
            .context("native CUDA decoder has no token input binding")?
            .to_owned();
        let position_input = self
            .step_input_name(NativeStepInputSource::PositionIds)
            .map(str::to_owned);

        // Resolve the immutable facts we need from state before allocating the
        // padded bindings (which borrows `self.session`); the two live on
        // disjoint fields so both borrows coexist.
        let state = self
            .cuda
            .as_mut()
            .context("CUDA decode state is not initialized")?;
        if total_len > state.capacity.max_len {
            bail!("{}", state.capacity_exceeded_error(total_len));
        }
        let grew = state.ensure_capacity(&mut self.session, total_len)?;
        state.extend_mask(if grew { 0 } else { past_len }, total_len, total_len)?;
        state.invalidate_graph(&mut self.session)?;

        let bucket = state.max_len;
        let vocab = *state
            .logits_shape
            .last()
            .context("CUDA logits shape has no vocabulary dimension")?;
        let logits_dtype = state.logits_dtype;
        let logits_index = state.logits_binding;
        let input_ids_index = state.input_ids_binding;
        let position_ids_index = state.position_ids_binding;
        let logits_name = state.bindings[logits_index]
            .output_name()
            .context("logits binding has no output name")?
            .to_owned();

        // ---- FIX 1: persistent padded [1, m, vocab] logits device binding ----
        let padded_logits = self.session.allocate_device_output_binding(
            logits_name,
            logits_dtype,
            vec![1, m, vocab],
            vec![1, m, vocab],
        )?;
        // ---- padded [1, m] token binding (device-resident, like run_one_token,
        // so the capture takes no owned host inputs) ----
        let mut padded_ids = self.session.allocate_device_binding(
            token_input,
            None::<String>,
            DataType::Int64,
            vec![1, m],
            vec![1, m],
        )?;
        let mut token_bytes = Vec::with_capacity(m * 8);
        for i in 0..m {
            // For a timing/segments attempt the token value is irrelevant to
            // capture/timing (KV correctness is out of scope); a constant keeps
            // the device buffer fixed for replay. For a parity attempt we cycle
            // real prompt tokens so the argmax is non-degenerate (avoids a
            // constant-token near-tie).
            let tok = match tokens {
                Some(t) if !t.is_empty() => t[i % t.len()],
                _ => 1_i64,
            };
            token_bytes.extend_from_slice(&tok.to_ne_bytes());
        }
        padded_ids.write_bytes(0, &token_bytes)?;

        let padded_positions = if let Some(position_input) = position_input.as_deref() {
            let mut binding = self.session.allocate_device_binding(
                position_input,
                None::<String>,
                DataType::Int64,
                vec![1, m],
                vec![1, m],
            )?;
            let mut pos_bytes = Vec::with_capacity(m * 8);
            for pos in past_len..total_len {
                pos_bytes.extend_from_slice(&(pos as i64).to_ne_bytes());
            }
            binding.write_bytes(0, &pos_bytes)?;
            Some(binding)
        } else {
            None
        };

        // Swap the padded bindings into the persistent vector. The originals are
        // parked and restored before returning.
        let orig_ids = std::mem::replace(&mut state.bindings[input_ids_index], padded_ids);
        let orig_logits = std::mem::replace(&mut state.bindings[logits_index], padded_logits);
        let orig_positions = position_ids_index
            .zip(padded_positions)
            .map(|(index, binding)| {
                (
                    index,
                    std::mem::replace(&mut state.bindings[index], binding),
                )
            });

        // ---- FIX 2: warm the capture-safe scratch arena at the M=K shape so the
        // captured region below performs no device alloc/free ----
        let warm_before = self.session.device_allocation_counts();
        let warm = self
            .session
            .run_with_device_bindings(&[], &mut state.bindings[..]);
        let warm_after = self.session.device_allocation_counts();
        if let Err(error) = warm {
            // Restore before surfacing the error.
            let restored_ids = std::mem::replace(&mut state.bindings[input_ids_index], orig_ids);
            let restored_logits = std::mem::replace(&mut state.bindings[logits_index], orig_logits);
            drop(restored_ids);
            drop(restored_logits);
            if let Some((index, binding)) = orig_positions {
                drop(std::mem::replace(&mut state.bindings[index], binding));
            }
            state.invalidate_graph(&mut self.session)?;
            return Err(error).context("leverb increment0 warm forward at M=K");
        }
        let warm_alloc_delta = match (warm_before, warm_after) {
            (Some(before), Some(after)) => Some((
                after.allocations.saturating_sub(before.allocations),
                after.frees.saturating_sub(before.frees),
            )),
            _ => None,
        };

        // Snapshot the EAGER warm-forward logits before capture overwrites the
        // padded binding on replay. This is the "eager M=K" side of the
        // captured-vs-eager token-parity cell.
        let warm_logits_bytes = if collect_parity {
            Some(state.bindings[logits_index].read_bytes()?)
        } else {
            None
        };

        // ---- Capture the real M=K forward against the full device binding set
        // (KV + mask + token + position + padded logits), no owned host inputs. ----
        let alloc_before = self.session.device_allocation_counts();
        let capture = self
            .session
            .try_capture_with_device_bindings(&[], &mut state.bindings[..]);
        let alloc_after = self.session.device_allocation_counts();
        let alloc_delta = match (alloc_before, alloc_after) {
            (Some(before), Some(after)) => Some((
                after.allocations.saturating_sub(before.allocations),
                after.frees.saturating_sub(before.frees),
            )),
            _ => None,
        };

        let segments = self.session.captured_graph_segment_count();
        // Summarize the eager seam nodes that split a segmented capture: the
        // root cause of a >1-segment M=K graph that never reaches the whole-
        // subgraph replay fast path. Fold to `op_type[seam] × count`.
        let seam_summary = {
            let mut counts: std::collections::BTreeMap<String, usize> =
                std::collections::BTreeMap::new();
            for decline in self.session.capture_segmentation() {
                let seam = decline
                    .seam_reason
                    .map(|reason| format!("{reason:?}"))
                    .unwrap_or_else(|| "graph".to_string());
                *counts
                    .entry(format!("{}[{}]", decline.op_type, seam))
                    .or_default() += 1;
            }
            if counts.is_empty() {
                None
            } else {
                Some(
                    counts
                        .into_iter()
                        .map(|(key, count)| format!("{key}×{count}"))
                        .collect::<Vec<_>>()
                        .join(", "),
                )
            }
        };

        let mut attempt = LeverBPhase0CaptureAttempt {
            rows: m,
            past_len,
            bucket,
            captured: false,
            segments,
            decline: None,
            alloc_delta,
            warm_alloc_delta,
            replay_walls_ns: Vec::new(),
            seam_summary,
            warm_argmax: Vec::new(),
            replay_argmax: Vec::new(),
            logits_byte_identical: None,
        };

        let mut last_replay_bytes: Option<Vec<u8>> = None;
        match capture {
            Ok(DeviceGraphCaptureResult::Captured(outputs)) => {
                attempt.captured = true;
                if outputs.iter().any(Option::is_some) {
                    attempt.decline = Some(
                        "captured, but still materialized a host output (padded logits binding did not absorb every output)".to_string(),
                    );
                }
                for _ in 0..replays {
                    let start = std::time::Instant::now();
                    let still_valid = self.session.replay_device_graph(&mut state.bindings[..])?;
                    // Sync on the padded logits D2H read (the real hot path syncs
                    // on the per-step logits read), making the wall GPU-inclusive.
                    let replay_bytes = state.bindings[logits_index].read_bytes()?;
                    if collect_parity {
                        last_replay_bytes = Some(replay_bytes);
                    }
                    attempt
                        .replay_walls_ns
                        .push(start.elapsed().as_nanos() as u64);
                    if !still_valid {
                        break;
                    }
                }
            }
            Ok(DeviceGraphCaptureResult::NotCapturable(report)) => {
                attempt.decline = Some(report.to_string());
            }
            Err(error) => {
                attempt.decline = Some(format!("capture attempt errored: {error}"));
            }
        }

        // Captured-vs-eager token parity: compare the eager warm-forward logits
        // against the captured replay logits over the identical device bindings.
        if let (Some(warm_bytes), Some(replay_bytes)) =
            (warm_logits_bytes.as_ref(), last_replay_bytes.as_ref())
        {
            attempt.warm_argmax = logits_rows_argmax(warm_bytes, logits_dtype, m, vocab);
            attempt.replay_argmax = logits_rows_argmax(replay_bytes, logits_dtype, m, vocab);
            attempt.logits_byte_identical = Some(warm_bytes == replay_bytes);
        }

        // Restore the persistent bindings and leave the graph slot clean; the KV
        // state is intentionally left dirty (caller discards the session).
        let restored_ids = std::mem::replace(&mut state.bindings[input_ids_index], orig_ids);
        let restored_logits = std::mem::replace(&mut state.bindings[logits_index], orig_logits);
        drop(restored_ids);
        drop(restored_logits);
        if let Some((index, binding)) = orig_positions {
            drop(std::mem::replace(&mut state.bindings[index], binding));
        }
        state.invalidate_graph(&mut self.session)?;
        Ok(attempt)
    }

    /// Run one forward over `token_ids`, padding the query axis to a shape the
    /// kernel cache is likely to hold already.
    ///
    /// The padded rows are pure waste arithmetically, but they are also
    /// harmless: the causal mask keeps every real row from reading a padded
    /// one, the padded rows' logits are dropped, and the KV they write lands
    /// past the sequence's logical length, so the rewind below both hides it
    /// from the next forward and lets that forward overwrite it.
    pub(crate) fn decode_cuda(
        &mut self,
        token_ids: &[TokenId],
        past_len: usize,
        step_inputs: &[(String, Tensor)],
    ) -> anyhow::Result<Vec<Vec<f32>>> {
        let rows = token_ids.len();
        let Some(padded) = self.padded_prefill_inputs(token_ids, step_inputs) else {
            return self.decode_cuda_exact(token_ids, past_len, step_inputs);
        };
        let padded_rows = padded.tokens.len();
        let mut logits = self.decode_cuda_exact(&padded.tokens, past_len, &padded.step_inputs)?;
        if logits.len() < padded_rows {
            // This decoder does not answer one logits row per query row, so
            // there is no way to tell a real row's logits from a padded row's.
            // Give up on padding for good and redo the forward as asked.
            self.prefill_query_padding = false;
            self.rewind_inner(past_len)
                .context("discard a padded native CUDA prefill this decoder cannot map back")?;
            return self.decode_cuda_exact(token_ids, past_len, step_inputs);
        }
        logits.truncate(rows);
        let real_len = past_len
            .checked_add(rows)
            .context("native decode context length overflow")?;
        self.rewind_inner(real_len)
            .context("roll a padded native CUDA prefill back to its real length")?;
        Ok(logits)
    }

    fn decode_cuda_exact(
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
            state.prepare_decode_workspace_after_capacity_growth(&mut self.session, grew)?;
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
                if let Some(hidden_output) = self.hidden_output.as_deref() {
                    self.last_hidden = Some(read_aux_hidden_last_row(state, hidden_output)?);
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
            if let Some(hidden_output) = self.hidden_output.as_deref() {
                self.last_hidden = Some(read_aux_hidden_last_row(state, hidden_output)?);
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
        state.prepare_decode_workspace_after_capacity_growth(&mut self.session, grew)?;

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

    /// [`Self::decode_cuda_captured_step_inputs`], but selecting the greedy token
    /// with the device argmax instead of returning host logits.
    ///
    /// The forward pass is the identical captured replay; only the epilogue
    /// differs, and that epilogue is where the cost was. Reading logits moves the
    /// whole vocabulary to the host every token — 404 KB for this repo's 202048
    /// -entry vocabulary — drains the stream to do it, and then walks all 202048
    /// values on the CPU for the finiteness check. Reading the greedy result
    /// moves eight bytes.
    ///
    /// Tie-breaking is `device_argmax`'s lowest-index rule, which is the same
    /// rule the host `sample_greedy` uses, so the token stream is unchanged.
    /// The device argmax also returns the shared capture-error word, so the
    /// detection-before-consumption check is preserved without a second sync
    /// (this is why there is no separate `check_device_capture_error` call).
    ///
    /// The dropped non-finite check is not a loss of safety here: a NaN logit
    /// cannot win an argmax whose comparison is a strict `>` seeded from
    /// negative infinity, and the capture-error word still catches a device-side
    /// violation. The host path keeps its check, since it is the path that hands
    /// raw logits to processors.
    pub(crate) fn decode_cuda_captured_step_inputs_greedy(
        &mut self,
        token_ids: &[TokenId],
        past_len: usize,
        total_len: usize,
        supplied: &[(String, Tensor)],
    ) -> anyhow::Result<TokenId> {
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
        let mask_expose = state.decode_mask_expose_len(total_len);
        state.extend_mask(if grew { 0 } else { past_len }, total_len, mask_expose)?;
        state.write_captured_step_inputs(supplied, position)?;
        state.prepare_decode_workspace_after_capacity_growth(&mut self.session, grew)?;

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
                "native CUDA decoder aborted: device capture validation violation (flags=0x{capture_error:x}) detected during captured graph replay of a per-step-input greedy decode; the produced token was rejected before consumption and the decode graph was invalidated"
            );
        }
        state.set_logical_len(total_len)?;
        self.current_len = total_len;
        Ok(token_id)
    }

    /// Whether a single-token step carrying routed per-step ports can take the
    /// captured device-argmax epilogue above.
    pub(crate) fn captured_step_input_greedy_supported(&self) -> bool {
        self.has_eager_step_inputs()
            && self
                .cuda
                .as_ref()
                .is_some_and(|state| state.capture_step_inputs && state.greedy_fastpath_supported())
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
        // Re-prepare the governed workspace when the KV bucket grew so the
        // persistent `::Attention` fp32 score scratch is sized for the new,
        // larger bucket (Bug 1). #1189 added this to the captured/greedy decode
        // paths but not to this routed-step eager path, which can rebucket just
        // the same; without it the first execute past a power-of-two boundary
        // needs exactly 2× the reserved workspace and trips the prepared-
        // workspace invariant. Cheap: only fires on the rare `grew` transition.
        state.prepare_decode_workspace_after_capacity_growth(&mut self.session, grew)?;

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
        // Re-prepare the governed workspace when the KV bucket grew so the
        // persistent `::Attention` fp32 score scratch is sized for the new,
        // larger bucket (Bug 1). #1189 added this guard to the captured/greedy
        // decode paths but not to this spec-decode verify path, which rebuckets
        // when `past + K` crosses a power-of-two boundary; without it the verify
        // forward trips the prepared-workspace invariant. Only fires on `grew`.
        state.prepare_decode_workspace_after_capacity_growth(&mut self.session, grew)?;
        // Option-c: when verify capture is configured and this verify hits the
        // fixed width M, run it through the captured Verify graph slot (replayable
        // across steps) instead of the eager per-op-launch path. A tail verify at
        // width != M falls through to the eager path, which leaves the captured
        // verify graph installed for the next full-width step.
        if self.verify_capture_width() == Some(token_ids.len()) {
            return self.run_verify_captured(token_ids, past_len, total_len);
        }
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
        state.prepare_decode_workspace_after_capacity_growth(&mut self.session, grew)?;
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

    /// The configured device-token-loop chain depth for this session (`0` when
    /// the loop is disabled or the topology is not device-loopable).
    pub(crate) fn device_token_loop_k(&self) -> usize {
        self.cuda
            .as_ref()
            .map_or(0, DecodeCudaState::device_token_loop_k)
    }

    /// Device-resident token-feedback loop (opt-in via
    /// `ONNX_GENAI_DEVICE_TOKEN_LOOP`): enqueue up to `k_request` captured decode
    /// replays back-to-back with a device token-writer stitched between them, so
    /// the host leaves the per-step critical path (no argmax read-back sync, no
    /// next-token H2D between replays), and drain the accumulated token ids in
    /// **one** D2H read.
    ///
    /// Byte-identical to the per-token [`Self::decode_cuda_greedy`] path: the
    /// device token-writer folds the *same* device-argmax token (`greedy_result[0]`)
    /// into the *same* persistent `input_ids`/`position_ids`/mask bindings the
    /// host writes would, so the greedy token sequence is unchanged. Capture-error
    /// latching is preserved (the per-step words are OR-ed on-device and rejected
    /// at drain).
    ///
    /// Any structural reason to decline — the loop is off, the graph is not yet
    /// in the replay-ready phase, an inline-routed step, or a capacity boundary —
    /// falls back to a single captured [`Self::decode_cuda_greedy`] step, which
    /// owns the warmup/capture/growth state machine. Returns the `1..=k` selected
    /// token ids in order.
    pub(crate) fn decode_cuda_greedy_loop(
        &mut self,
        seed_token: TokenId,
        past_len: usize,
        k_request: usize,
    ) -> anyhow::Result<Vec<TokenId>> {
        // Keep inline-routing state coherent with the per-token entry point,
        // then decide whether the device loop applies to this step.
        self.maybe_enable_decode_inline(std::slice::from_ref(&seed_token));
        let route_inline = self.route_decode_inline(std::slice::from_ref(&seed_token));
        let applies = {
            let state = self
                .cuda
                .as_ref()
                .context("CUDA decode state is not initialized")?;
            state.device_token_loop_ready
                && state.device_token_loop_k >= 2
                && k_request >= 2
                && state.graph_phase == DecodeCudaGraphPhase::Ready
        };
        if route_inline || !applies {
            return Ok(vec![self.decode_cuda_greedy(seed_token, past_len)?]);
        }

        // Clamp the chain depth to the configured cap and keep the whole chain
        // inside the current KV bucket so no reallocation (which retires the
        // captured graph) happens mid-chain.
        let (cap_k, max_len, hard_max_len) = {
            let state = self.cuda.as_ref().unwrap();
            (
                state.device_token_loop_k,
                state.max_len,
                state.hard_max_len(),
            )
        };
        let mut k = k_request.min(cap_k);
        k = k.min(max_len.saturating_sub(past_len));
        if k < 2 || past_len + k > hard_max_len {
            return Ok(vec![self.decode_cuda_greedy(seed_token, past_len)?]);
        }
        let total_len = past_len + k;

        // Ensure capacity for the full chain up-front; a growth here retires the
        // captured graph, so route this step through the single captured path.
        let grew = {
            let state = self.cuda.as_mut().unwrap();
            state.ensure_capacity(&mut self.session, total_len)?
        };
        if grew || self.cuda.as_ref().unwrap().graph_phase != DecodeCudaGraphPhase::Ready {
            return Ok(vec![self.decode_cuda_greedy(seed_token, past_len)?]);
        }

        // Prime the first step on the host (one async H2D, no sync) so the chain
        // is byte-identical to the per-token path regardless of prior device
        // state, and clear the capture-error accumulator.
        {
            let state = self.cuda.as_mut().unwrap();
            state.write_decode_inputs(seed_token, past_len)?;
            let expose = state.decode_mask_expose_len(past_len + 1);
            state.extend_mask(past_len, past_len + 1, expose)?;
            state.reset_device_token_loop_error()?;
        }

        // Enqueue k replays back-to-back with the device token-writer stitched
        // between them — no host sync inside the loop.
        let mut produced = 0_usize;
        for step in 0..k {
            let state = self.cuda.as_mut().unwrap();
            if let Err(error) = state.run_one_token(&mut self.session, &self.trace) {
                let diagnosis = diagnose_native_failure(&self.session, &error.to_string());
                return Err(error.context(format!(
                    "native CUDA device-token-loop forward pass failed{diagnosis}"
                )));
            }
            let still_ready = state.graph_phase == DecodeCudaGraphPhase::Ready;
            state.run_device_argmax()?;
            let next_position = i64::try_from(past_len + 1 + step)
                .context("device token loop position exceeds i64 range")?;
            state.device_token_writer_step(next_position, step as u32)?;
            state.device_token_loop_steps += 1;
            produced += 1;
            if !still_ready {
                // A control-flow branch flip retired the captured graph after
                // this step (its token was produced eagerly and is valid); stop
                // chaining and let the next call re-warm on the single-step path.
                break;
            }
        }

        // One D2H drain for the whole chain: the accumulated token ids plus the
        // OR-ed capture-error word (detection-before-consumption).
        let (tokens, capture_error) = {
            let state = self.cuda.as_mut().unwrap();
            state.drain_device_token_loop(produced)?
        };
        if capture_error != 0 {
            let state = self.cuda.as_mut().unwrap();
            let _ = state.invalidate_graph(&mut self.session);
            bail!(
                "native CUDA device-token-loop aborted: device capture validation violation (flags=0x{capture_error:x}) detected during a chained captured graph replay; the produced tokens were rejected before consumption and the decode graph was invalidated"
            );
        }

        let new_len = past_len + produced;
        {
            let state = self.cuda.as_mut().unwrap();
            // Undo the device token-writer's trailing mask bit one position past
            // the last consumed token so the persistent mask matches the
            // per-token path exactly (byte-identical, and no stale `1` leaks into
            // a later prefill after reset).
            state.clear_trailing_mask_bit(new_len)?;
            state.set_logical_len(new_len)?;
        }
        self.current_len = new_len;
        Ok(tokens)
    }

    /// sequences together, one token each, and return the `batch` selected token
    /// ids (row `i` = the greedy token of sequence `i`). Every sequence shares
    /// the uniform decode length `past_len → past_len + 1`, so a single mask
    /// window and one `run_one_token` forward drive all rows; the batched device
    /// argmax reads N rows back.
    ///
    /// This is the reachable batch-N entry point: `tokens.len()` must equal the
    /// pinned session batch. It is NOT wired into the single-sequence generation
    /// driver — presenting N real sequences with N independently seeded KV states
    /// is the stage 2c caller — so this is exercised directly by the batch
    /// measurement harness (`profile_native --native-decode-batch-sweep`) and by
    /// row-identity assertions there. The greedy device argmax already returns N
    /// rows (2b-impl-3), so no logits round-trip is needed.
    ///
    /// Uniform batches are the special case of the ragged path (stage 3a, #750):
    /// every row shares `past_len` and advances, so this forwards to
    /// [`Self::decode_cuda_greedy_batch_ragged`] with a uniform per-row length
    /// and an all-`true` advance mask. The delegation is byte-identical to the
    /// former single-shared-window implementation (identical mask windows,
    /// identical positions).
    pub(crate) fn decode_cuda_greedy_batch(
        &mut self,
        tokens: &[TokenId],
        past_len: usize,
    ) -> anyhow::Result<Vec<TokenId>> {
        let batch = self
            .cuda
            .as_ref()
            .context("CUDA decode state is not initialized")?
            .batch;
        let past_lens = vec![past_len; batch];
        let advances = vec![true; batch];
        self.decode_cuda_greedy_batch_ragged(tokens, &past_lens, &advances)
    }

    /// Ragged batch-N greedy decode step (stage 3a, #750): step `batch`
    /// sequences together, one token each, where row `r` sits at its own logical
    /// length `past_lens[r]` and advances only if `advances[r]`. Returns the
    /// `batch` selected token ids (row `i` = the greedy token of sequence `i`).
    ///
    /// Per-row geometry:
    /// - **Length.** Each row's attention-mask window is `past_lens[r] + 1` (its
    ///   own prefix plus the new token); the model reduces that to the row's
    ///   `seqlens_k = past_lens[r]` and writes present KV at that per-row offset.
    /// - **Position.** `position_ids[r] = past_lens[r]`, so each row's rotary
    ///   position and causal frame match what it would see run alone.
    /// - **Physical extent.** The shared KV logical length is advanced to
    ///   `max(past_lens) + 1`; shorter rows ignore the padded suffix via their
    ///   own (shorter) mask window.
    ///
    /// The mask width is frozen to the physical bucket, so only mask *values*
    /// vary per step and CUDA-graph capture survives per-row lengths. A held row
    /// (`advances[r] == false`) re-attends its own prefix and reprocesses at its
    /// current position; its present KV write lands at its unchanged offset
    /// (overwritten harmlessly next step) and its logical length does not grow —
    /// the mechanism a continuous batcher uses to stall a row while its peers
    /// advance.
    pub(crate) fn decode_cuda_greedy_batch_ragged(
        &mut self,
        tokens: &[TokenId],
        past_lens: &[usize],
        advances: &[bool],
    ) -> anyhow::Result<Vec<TokenId>> {
        let profile_past = past_lens.iter().copied().max().unwrap_or(0);
        let profile_total = profile_past.saturating_add(1);
        let step_profile = CudaStepProfile::begin(profile_past, profile_total);
        let mut step_wall = CudaStepWallBreakdown::default();

        let run_start = std::time::Instant::now();
        let valid_lens = self.run_ragged_forward(tokens, past_lens, advances)?;
        step_wall.run_ms = run_start.elapsed().as_secs_f64() * 1_000.0;

        let state = self
            .cuda
            .as_mut()
            .context("CUDA decode state is not initialized")?;
        let read_start = std::time::Instant::now();
        let rows = state.read_greedy_result_batch()?;
        step_wall.logits_read_ms = read_start.elapsed().as_secs_f64() * 1_000.0;
        for (sequence, (_, capture_error)) in rows.iter().enumerate() {
            if *capture_error != 0 {
                let _ = state.invalidate_graph(&mut self.session);
                bail!(
                    "native CUDA batch decoder aborted: device capture validation violation \
                     (flags=0x{capture_error:x}) on sequence {sequence} during captured graph \
                     replay; the produced tokens were rejected before consumption and the decode \
                     graph was invalidated"
                );
            }
        }
        self.commit_ragged_advance(&valid_lens, advances)?;
        if let Some(profile) = step_profile {
            profile.finish("decode_batch", step_wall);
        }
        Ok(rows.into_iter().map(|(token, _)| token).collect())
    }

    /// Host-logits ragged batch-N step (stage 3b, #750): identical geometry to
    /// [`Self::decode_cuda_greedy_batch_ragged`], but instead of the device-argmax
    /// fast path it reads the full `[batch, 1, vocab]` logits back to the host —
    /// one `[vocab]` row per batch slot — so a real host sampler (top-k/top-p,
    /// temperature, penalties) can drive selection. The device-argmax path is
    /// **not** removed: it stays the default for greedy decode because it never
    /// pays the D2H of the full logits. This is the seam the
    /// `ContinuousBatchManager` sampler consumes.
    ///
    /// The device capture-error latch is still checked *before* the host logits
    /// are consumed (detection-before-consumption): the cheap 8-byte-per-row
    /// argmax read-back doubles as the capture-validation read, and a latched
    /// violation invalidates the graph and rejects the step. The full-logits D2H
    /// cost is returned so the caller can report it honestly rather than bury it.
    pub(crate) fn decode_cuda_greedy_batch_ragged_logits(
        &mut self,
        tokens: &[TokenId],
        past_lens: &[usize],
        advances: &[bool],
    ) -> anyhow::Result<RaggedLogitsStep> {
        let valid_lens = self.run_ragged_forward(tokens, past_lens, advances)?;
        let state = self
            .cuda
            .as_mut()
            .context("CUDA decode state is not initialized")?;
        // Detection-before-consumption: read the shared capture-error latch (the
        // argmax read-back carries it) and reject before touching the logits.
        let guard = state.read_greedy_result_batch()?;
        for (sequence, (_, capture_error)) in guard.iter().enumerate() {
            if *capture_error != 0 {
                let _ = state.invalidate_graph(&mut self.session);
                bail!(
                    "native CUDA batch decoder aborted: device capture validation violation \
                     (flags=0x{capture_error:x}) on sequence {sequence} during captured graph \
                     replay; the produced logits were rejected before consumption and the decode \
                     graph was invalidated"
                );
            }
        }
        let (logits, d2h_bytes, d2h_time) = state.read_batch_row_logits()?;
        self.commit_ragged_advance(&valid_lens, advances)?;
        Ok(RaggedLogitsStep {
            logits,
            d2h_bytes,
            d2h_time,
        })
    }

    /// Shared ragged-step preamble: validate the per-row inputs, size the mask
    /// window and physical extent, ensure KV capacity, write the ragged mask and
    /// the per-row token/position inputs, and run the fused forward. Returns the
    /// per-row valid mask widths (`past_lens[r] + 1`) so the caller can advance
    /// the stepped rows after it has validated/consumed the forward's output.
    /// Does **not** read results or advance `row_lens` — those differ between the
    /// device-argmax and host-logits consumers.
    fn run_ragged_forward(
        &mut self,
        tokens: &[TokenId],
        past_lens: &[usize],
        advances: &[bool],
    ) -> anyhow::Result<Vec<usize>> {
        let state = self
            .cuda
            .as_mut()
            .context("CUDA decode state is not initialized")?;
        if tokens.len() != state.batch {
            bail!(
                "native CUDA batch greedy decode expects {} tokens (pinned batch), got {}",
                state.batch,
                tokens.len()
            );
        }
        if past_lens.len() != state.batch || advances.len() != state.batch {
            bail!(
                "native CUDA ragged batch greedy decode expects {} per-row lengths and advances, \
                 got {} / {}",
                state.batch,
                past_lens.len(),
                advances.len()
            );
        }
        if !state.greedy_fastpath_supported() {
            bail!(
                "native CUDA batch greedy decode requires the device-argmax fast path; this \
                 decoder's logits binding does not support it"
            );
        }
        // Per-row valid mask width (prefix + the new token) and the shared
        // physical extent this step must cover.
        let mut valid_lens = Vec::with_capacity(state.batch);
        for &past_len in past_lens {
            valid_lens.push(
                past_len
                    .checked_add(1)
                    .context("native decode context length overflow")?,
            );
        }
        let max_total = valid_lens
            .iter()
            .copied()
            .max()
            .context("native CUDA ragged batch requires at least one row")?;
        if max_total > state.capacity.max_len {
            bail!("{}", state.capacity_exceeded_error(max_total));
        }
        state.ensure_capacity(&mut self.session, max_total)?;
        state.extend_mask_ragged(&valid_lens, state.decode_mask_expose_len(max_total))?;
        state.write_decode_inputs_batch(tokens, past_lens)?;
        if let Err(error) = state.run_one_token(&mut self.session, &self.trace) {
            let diagnosis = diagnose_native_failure(&self.session, &error.to_string());
            return Err(error.context(format!(
                "native CUDA batch decoder forward pass failed{diagnosis}"
            )));
        }
        Ok(valid_lens)
    }

    /// Advance the stepped rows' logical lengths after a ragged forward and
    /// re-derive the shared physical extent. A held row (`advances[r] == false`)
    /// keeps its length.
    fn commit_ragged_advance(
        &mut self,
        valid_lens: &[usize],
        advances: &[bool],
    ) -> anyhow::Result<()> {
        let state = self
            .cuda
            .as_mut()
            .context("CUDA decode state is not initialized")?;
        for (row, &advance) in advances.iter().enumerate() {
            if advance {
                state.row_lens[row] = valid_lens[row];
            }
        }
        let new_max = state.row_lens.iter().copied().max().unwrap_or(0);
        state.set_logical_len(new_max)?;
        self.current_len = new_max;
        Ok(())
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
    /// The KV shape is always **BNSH** `[batch, kv_heads, max_len, head_dim]`,
    /// growing on axis 2, for *both* the head-major and seq-major layouts. The
    /// `batch` axis-0 extent is threaded from the caller (stage 2b-impl-1, #750)
    /// rather than hard-coded to `1`; every current caller passes `1`, so the
    /// emitted shape is unchanged. Threading `batch` must **not** disturb the
    /// axis order: the CUDA GQA node reads `present_capacity` from BNSH axis 2
    /// regardless of `kv_layout`, so axis 0 stays batch and axis 2 stays seq.
    /// This is deliberate and is the KV binding contract: the CUDA GQA node validates
    /// `past_key`/`past_value` as `[batch, kv_heads, seq, head_dim]` and reads
    /// `present_capacity` from axis 2 **regardless** of the `kv_layout`
    /// attribute (which only re-specializes the kernel's stride arithmetic). A
    /// seq-major binding therefore keeps this BNSH metadata shape; its BSNH
    /// physical byte layout — and the capacity-independent fixed per-token stride
    /// that lets it grow without moving data — is expressed by the growth/commit
    /// geometry (`kv_growth_byte_layout`, `apply_vmm_growth`,
    /// `build_grown_buffers`), not by permuting this shape. See
    /// `docs/memory/MEMORY_ARCHITECTURE.md`, "KV layout and residency".
    pub(crate) fn persistent_state_shapes(
        name: &str,
        dtype: DataType,
        shape: &[Dim],
        batch: usize,
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
                    batch
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
                    Ok(batch)
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

    /// Whether this decoder is on the VMM (commit-on-demand) KV path. Exposed so
    /// a regression test can assert it is *actually* exercising the VMM path
    /// rather than silently falling back to the eager path (which would make a
    /// "VMM reservation" assertion prove nothing).
    ///
    /// `cfg(test)` as well as the feature: its only caller is the assertion in
    /// `native_decode::tests`, so in the plain `lib` target it is genuinely
    /// dead and `-D dead-code` rejects it. It is `pub(crate)`, so no integration
    /// test outside this crate could reach it anyway.
    #[cfg(all(feature = "native-cuda", test))]
    pub(crate) fn kv_commits_on_demand(&self) -> bool {
        self.kv_commits_on_demand
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
    /// [`KvCommitLayout::SeqMajor`] — one contiguous run
    /// `0..(committed_len × kv_heads × head_dim × elem)` **per sequence** (batch
    /// axis 0), each `capacity × kv_heads × head_dim × elem` apart, because
    /// seq-major bytes are token-contiguous within a sequence and the per-
    /// sequence stride is the fixed full-context `capacity`. At `batch == 1`
    /// (the only value any current caller constructs, stage 2b-impl-2, #750) this
    /// is exactly one run and byte-identical to the previous single-sequence
    /// form; the batch axis is outermost, so committing `batch` sequences never
    /// relocates any sequence's bytes. This is the *same* geometry unit the
    /// driver-level residency measurement (`vmm_kv_layout_residency_gpu`) and the
    /// `kv_commit.rs` unit tests exercise, so the live commit path and the
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
                self.batch,
                capacity,
                committed_len,
            )
            .with_context(|| {
                format!(
                    "seq-major VMM-backed CUDA KV '{}' dense-prefix commit ranges overflow \
                     (capacity {capacity}, batch {}, committed_len {committed_len})",
                    binding.input_name(),
                    self.batch,
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
    /// for **every sequence** on every KV binding. Not strictly required for
    /// correctness (the kernel reads only `[0, total_lengths)`), but it keeps the
    /// padding suffix zeroed exactly as the growing-bucket path does, guarding
    /// against any future over-read touching uninitialized physical pages.
    ///
    /// Seq-major stores a sequence's tokens token-contiguously, so a sequence's
    /// committed tail is one contiguous run; the `batch` sequences are a fixed
    /// full-context `capacity × bytes_per_token` apart (batch axis outermost),
    /// so this zeroes one run per sequence with no relocation (stage 2b-impl-2,
    /// #750). At `batch == 1` this is a single memset byte-identical to the
    /// previous contiguous-tail form.
    fn zero_seq_major_committed_tail(
        &self,
        old_committed: usize,
        new_committed: usize,
    ) -> anyhow::Result<()> {
        if new_committed <= old_committed {
            return Ok(());
        }
        for index in self.kv_binding_range.clone() {
            let binding = &self.bindings[index];
            let (geometry, capacity) = self.kv_binding_geometry(binding)?;
            let bytes_per_token = geometry
                .bytes_per_token()
                .context("seq-major committed-tail bytes-per-token overflow")?;
            let per_sequence_stride = capacity
                .checked_mul(bytes_per_token)
                .context("seq-major committed-tail per-sequence stride overflow")?;
            let tail_offset = old_committed
                .checked_mul(bytes_per_token)
                .context("seq-major committed-tail offset overflow")?;
            let tail_bytes = (new_committed - old_committed)
                .checked_mul(bytes_per_token)
                .context("seq-major committed-tail byte count overflow")?;
            let ptr = binding.device_ptr() as usize;
            for sequence in 0..self.batch {
                let base = sequence
                    .checked_mul(per_sequence_stride)
                    .context("seq-major committed-tail sequence base overflow")?;
                native_cuda_memset_zero(ptr + base + tail_offset, tail_bytes)?;
            }
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
    /// `docs/memory/MEMORY_ARCHITECTURE.md`, "KV layout and residency"). Seq-major
    /// instead reports a fixed full-context stride and commits its dense prefix
    /// through [`Self::seq_major_kv_commit_requests`].
    ///
    /// **Batch generality (stage 2b-impl-2, #750).** The per-binding KV range is
    /// `checked_shape_bytes` of the physical shape whose axis 0 is `batch`, so
    /// the flat range already spans `batch × kv_heads × new_capacity × head_dim`
    /// bytes: a head-major packed bucket is dense from offset 0 across all `batch`
    /// sequences, so one flat `0..bucket_bytes` run is correct at any batch (this
    /// path only ever runs head-major — seq-major takes the fixed-stride commit).
    /// The mask island spans `batch` rows. At `batch == 1` this is byte-identical.
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
            .and_then(|row_bytes| row_bytes.checked_mul(self.batch))
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

    /// Resolve the `usize::MAX` "unbounded" capacity sentinel to a concrete
    /// upper bound for the **VMM** path's up-front virtual reservation.
    ///
    /// The non-VMM path grows KV buckets until the device OOMs, so it never
    /// needs a concrete `max_len` and `resolve_cuda_kv_capacity` deliberately
    /// leaves it at `usize::MAX` for a model with no `max_sequence_length`
    /// metadata (the sentinel from #367). The VMM path, in contrast, reserves
    /// the full context's *address range* up front, which structurally requires
    /// a real bound — `usize::MAX` cannot be reserved and today overflows the
    /// reservation arithmetic (issue #1266).
    ///
    /// The largest sequence length the device could *ever* hold is
    /// `device_free / bytes_per_token`, so reserve exactly that. On the VMM
    /// path the reservation is **virtual-only** — physical pages are committed
    /// on demand out of a fixed 64 GiB address arena — so an over-large but
    /// finite bound is nearly free, and a decode that outgrows it still fails
    /// at the same physical ceiling the sentinel implied. This resolves the
    /// bound *only here, only for the VMM reservation*; `capacity.max_len` and
    /// the non-VMM path that relies on the sentinel are left untouched.
    ///
    /// Requires a device free-memory reading. Without one the VMM path has no
    /// bound to reserve, so this errors with actionable guidance rather than
    /// overflowing or silently guessing a bound.
    fn vmm_unbounded_reservation_len(capacity: &CudaKvCapacity) -> anyhow::Result<usize> {
        // CEILING / KNOWN LIMITATION (tracked in issue #1288):
        // For a metadata-less model on the VMM path we reserve up-front virtual
        // address space for `free_bytes / bytes_per_token` tokens. Adding the
        // per-token mask (8 B/token on top of bytes_per_token) makes the total
        // carved span ~1.2x device_free. All EP device allocations — decoder
        // WEIGHTS, KV, and decode SCRATCH — carve from the SAME single VMM arena
        // of RESERVATION_BYTES = 64 << 30 (see onnx-runtime-ep-cuda provider
        // `memory()`), and the reservation is virtual-only (only committed
        // ranges claim physical granules). Measured: loading qwen05b-q4 with VMM
        // already occupies ~440 MiB across 517 spans in that arena before any KV
        // reservation. Consequence: on a GPU with ~53 GiB or more free VRAM this
        // carve can exceed the free virtual span and fail LOUDLY at construction.
        // That regime is untestable on the RTX 4060 (8 GB) this was developed on,
        // so we deliberately do NOT clamp here. A correct clamp would need (a) a
        // public free-VA accessor on the VMM allocator (none exists today; the
        // free `Spans` are not exposed and `committed_and_reserved()` returns
        // physical-committed + capacity, not free VA) and (b) a decode-scratch
        // margin policy, because scratch shares the same arena and an over-tight
        // clamp would fail later at scratch-carve time instead of loudly here.
        let device_memory = capacity.device_memory.as_ref().with_context(|| {
            format!(
                "cannot size the VMM-backed CUDA KV reservation for a model with no \
                 max_sequence_length metadata ({}): the device free-memory query is \
                 unavailable, so there is no bound to reserve up front. Set \
                 ONNX_GENAI_CUDA_KV_MAX_LEN or load_with_cuda_kv_max_len to a concrete \
                 length, or run without ONNX_GENAI_CUDA_VMM to use the grow-on-demand path.",
                capacity.source
            )
        })?;
        // `bytes_per_token` is validated > 0 in `resolve_cuda_kv_capacity`.
        let bound = device_memory.free_bytes / capacity.bytes_per_token.max(1);
        if bound == 0 {
            bail!(
                "device has too little free memory ({} bytes) to reserve a VMM-backed \
                 CUDA KV cache at {} bytes/token",
                device_memory.free_bytes,
                capacity.bytes_per_token
            );
        }
        Ok(bound)
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
        let old_shape = self.bindings[0].physical_shape().to_vec();
        if old_shape != [self.batch, self.max_len] {
            bail!(
                "VMM-backed CUDA mask binding must be [{}, {}] before growth, got {:?}",
                self.batch,
                self.max_len,
                old_shape
            );
        }
        let new_shape = vec![self.batch, new_capacity];
        let ptr = self.bindings[0].device_ptr() as usize;
        let elem = std::mem::size_of::<i64>();
        // The mask carries one row per sequence. Growing its capacity changes
        // every batch>0 row stride just like head-major KV, so preserve the
        // valid prefix before clearing only the newly exposed suffix. Clearing
        // the whole allocation here used to erase all prior `1` entries; GQA
        // then derived a zero past length at the first bucket boundary.
        copy_kv_prefix_device_to_device_in_place(ptr, &old_shape, &new_shape, 1, valid_len, elem)?;
        zero_kv_suffix_device(ptr, &new_shape, 1, valid_len, elem)?;
        self.bindings[0].set_physical_and_logical_shapes(new_shape, vec![self.batch, valid_len])?;
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
        batch: usize,
    ) -> anyhow::Result<Vec<usize>> {
        if matches!(dtype, DataType::Undefined | DataType::String) {
            bail!(
                "cannot bind auxiliary CUDA graph output '{name}' persistently: dtype {dtype:?} does not have fixed-size device tensor storage, but CUDA graph capture requires every declared graph output to use stable device storage; export this output as a numeric tensor or remove the unused graph output"
            );
        }
        // The batch axis (0) collapses to the threaded `batch` extent; every
        // other symbolic axis is a decode-unit (query-seq) that stays `1`. At
        // batch 1 (the only value any current caller passes) this is byte-
        // identical to the historical "collapse every symbolic dim to 1".
        let shape = shape
            .iter()
            .enumerate()
            .map(|(axis, dim)| match dim {
                Dim::Static(value) => *value,
                Dim::Symbolic(_) if axis == 0 => batch,
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
        requested_batch: Option<usize>,
    ) -> anyhow::Result<Self> {
        let kv_commits_on_demand = session.commits_on_demand();
        // Stage 2b-impl-4 (#750): the persistent decode bindings are shaped for
        // `batch` sequences and batch-N is now turned ON. The batch extent comes
        // from `requested_batch` (how `--max-batch` reaches here) or, failing
        // that, `ONNX_GENAI_NATIVE_DECODE_BATCH`, defaulting to `1`. It is pinned
        // for the whole session so CUDA-graph capture binds a stable batch shape
        // (capture requires stable shapes). At the default `batch == 1`
        // every emitted shape, IO binding, mask/token/position write, KV commit
        // geometry and device-argmax read-back is byte-identical to the previous
        // hard-coded `1`, so decode stays the #750 byte-identity reference.
        let batch = resolve_native_decode_batch(requested_batch)?;
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
        // The VMM path reserves the full context's virtual address range up
        // front, so it needs a concrete `max_len`. A metadata-less model leaves
        // `capacity.max_len` at the `usize::MAX` sentinel, which cannot be
        // reserved (issue #1266); resolve it — only for the VMM reservation —
        // to the largest token count the device could ever hold. On the non-VMM
        // path (or a model that declares its max length) this is just
        // `capacity.max_len`, so behaviour there is unchanged.
        let vmm_reserved_max_len = if kv_commits_on_demand && capacity.max_len == usize::MAX {
            Self::vmm_unbounded_reservation_len(&capacity)?
        } else {
            capacity.max_len
        };
        // The capacity reported to the bindings (physical axis-2 / mask island).
        // Seq-major fixed stride pins this at the hard maximum from the start;
        // everything else starts at the initial bucket and grows it.
        let reported_len = if seq_major_fixed {
            vmm_reserved_max_len
        } else {
            initial_bucket_len
        };
        let max_len = reported_len;
        // The mask binding is `[batch, len]`, so its committed / reserved byte
        // extents span `batch` rows (stage 2b-impl-2, #750). Every current
        // caller constructs at `batch == 1`, so the emitted byte counts are
        // byte-identical to the previous single-row `len × i64` computation.
        let mask_bytes = batch
            .checked_mul(max_len)
            .and_then(|elements| elements.checked_mul(std::mem::size_of::<i64>()))
            .context("initial CUDA mask size overflow")?;
        let mask = if kv_commits_on_demand {
            // The VMM committed-range binding reserves the full logical mask
            // island (`batch × vmm_reserved_max_len`) up front and commits only
            // the initial bucket. `vmm_reserved_max_len` is the sentinel
            // resolved to a concrete device-memory bound for a metadata-less
            // model (issue #1266), so this reservation multiply is always
            // bounded on the VMM path. The non-VMM binding below never reserves
            // the full extent, so it must not compute this value at all
            // (computing it unconditionally overflowed and aborted construction
            // for any metadata-less model even though the non-VMM path never
            // uses it).
            let full_mask_bytes = batch
                .checked_mul(vmm_reserved_max_len)
                .and_then(|elements| elements.checked_mul(std::mem::size_of::<i64>()))
                .context("full CUDA mask reservation size overflow")?;
            // Seq-major fixed stride commits the whole (tiny) mask at
            // construction so the mask island is shape-static at the hard max
            // and never grows; every other path commits only the initial mask
            // bucket and grows it in place.
            let mask_committed_bytes = if seq_major_fixed {
                full_mask_bytes
            } else {
                mask_bytes
            };
            let committed = std::iter::once(0..mask_committed_bytes).collect::<Vec<_>>();
            let binding = session.allocate_device_binding_committed(
                io.attention_mask,
                None::<String>,
                DataType::Int64,
                vec![batch, max_len],
                vec![batch, max_len],
                full_mask_bytes,
                committed,
            )?;
            native_cuda_memset_zero(binding.device_ptr() as usize, mask_committed_bytes)?;
            binding
        } else {
            // The non-VMM binding allocates exactly `[batch, max_len]` and never
            // reserves the full logical extent, so it never touches
            // `capacity.max_len` (which is `usize::MAX` for a metadata-less
            // model). Its committed region is the whole `mask_bytes` allocation.
            let binding = session.allocate_device_binding(
                io.attention_mask,
                None::<String>,
                DataType::Int64,
                vec![batch, max_len],
                vec![batch, max_len],
            )?;
            native_cuda_memset_zero(binding.device_ptr() as usize, mask_bytes)?;
            binding
        };

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
            let (physical_shape, logical_shape) = Self::persistent_state_shapes(
                past,
                meta.dtype,
                &meta.shape,
                batch,
                max_len,
                false,
            )?;
            let binding = if kv_commits_on_demand {
                let allocation_bytes = Self::full_vmm_kv_allocation_bytes(
                    &physical_shape,
                    meta.dtype,
                    vmm_reserved_max_len,
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
                Self::persistent_state_shapes(past, meta.dtype, &meta.shape, batch, max_len, true)?;
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
            Self::persistent_output_shape(io.logits, logits_dtype, &logits_meta.shape, batch)?;
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
            // Batch-symbol pinning (stage 2b-impl-4, #750): auxiliary outputs
            // whose axis 0 is the batch symbol are bound at the pinned `batch`
            // extent, exactly like the logits binding, so CUDA-graph capture
            // admits a batch-N grid for every declared output rather than sizing
            // one output to `1` and mis-shaping the replay. `persistent_output_shape`
            // maps a symbolic axis 0 → `batch` and every other symbolic (decode-
            // unit) axis → `1`; the `unresolved_symbolic_axis` gate above has
            // already declined any output with a non-unit symbolic axis. At
            // `batch == 1` this is byte-identical to the historical collapse.
            let shape = Self::persistent_output_shape(&meta.name, meta.dtype, &meta.shape, batch)?;
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
                vec![batch, 1, embeds.hidden],
                vec![batch, 1, embeds.hidden],
            )?);
        } else {
            bindings.push(session.allocate_device_binding(
                io.input_ids,
                None::<String>,
                DataType::Int64,
                vec![batch, 1],
                vec![batch, 1],
            )?);
        }
        let position_ids_binding = if let Some(position_ids) = io.position_ids {
            let index = bindings.len();
            // A rank-N mrope decoder declares `position_ids [N, B, S]`; the single
            // decode step collapses sequence to 1 → `[N, batch, 1]`. Rank 1 is
            // the conventional `[batch, 1]`. At `batch == 1` this is byte-
            // identical to the historical `[1, 1]` / `[N, 1, 1]`.
            let shape = if position_rank == 1 {
                vec![batch, 1]
            } else {
                vec![position_rank, batch, 1]
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

        #[cfg(feature = "native-cuda")]
        let argmax_words = {
            let vocab = *logits_shape
                .last()
                .context("CUDA logits shape has no vocabulary dimension")?;
            // The device-argmax result buffer holds a `2 × batch` header (a token
            // id + capture-error pair per sequence) plus per-sequence scratch
            // (stage 2b-impl-3, #750). At `batch == 1` this is byte-identical to
            // the previous `2 + scratch_words(vocab)` allocation.
            2 * batch + onnx_runtime_ep_cuda::device_argmax_scratch_words(vocab, batch)
        };
        #[cfg(not(feature = "native-cuda"))]
        let argmax_words = 2 * batch;
        let greedy_result = session.allocate_device_output_binding(
            "__native_greedy_argmax",
            DataType::Uint32,
            vec![argmax_words],
            vec![2 * batch],
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

        // Arm the device-resident token-feedback loop (opt-in) when the topology
        // is device-loopable: graph capture engaged, a single sequence, a rank-1
        // i64 `position_ids` binding, and an i64 attention mask frozen to
        // physical `max_len` (not logical-exposed). The scratch binding holds `k`
        // u32 token-log slots plus one u32 capture-error accumulator word; the
        // device token-writer folds each step's token straight into the
        // persistent `input_ids`/`position_ids`/mask bindings device-to-device.
        let device_token_loop_k_req = device_token_loop_k_from_env();
        let input_ids_is_i64 = bindings[input_ids_binding].dtype == DataType::Int64;
        let position_is_i64 =
            position_ids_binding.is_some_and(|index| bindings[index].dtype == DataType::Int64);
        let mask_is_i64 = bindings
            .first()
            .is_some_and(|binding| binding.dtype == DataType::Int64);
        // When the model has a persistent position_ids binding it must be a
        // rank-1 i64 binding the writer can advance; when there is none, position
        // is derived from the mask alone (still byte-identical) and the writer
        // only folds the token id and mask bit.
        let position_topology_ok = match position_ids_binding {
            Some(_) => position_rank == 1 && position_is_i64,
            None => true,
        };
        let device_token_loop_write_position = position_ids_binding.is_some();
        let device_token_loop_ready = device_token_loop_k_req > 0
            && graph_enabled
            && !mask_exposes_logical
            && batch == 1
            && input_ids_is_i64
            && mask_is_i64
            && position_topology_ok;
        let device_token_loop_k = if device_token_loop_ready {
            device_token_loop_k_req
        } else {
            0
        };
        let device_token_loop_scratch = if device_token_loop_ready {
            let words = device_token_loop_k + 1;
            Some(session.allocate_device_output_binding(
                "__native_device_token_loop",
                DataType::Uint32,
                vec![words],
                vec![words],
            )?)
        } else {
            None
        };
        if device_token_loop_k_req > 0 {
            tracing::info!(
                requested_k = device_token_loop_k_req,
                armed = device_token_loop_ready,
                effective_k = device_token_loop_k,
                write_position = device_token_loop_write_position,
                "native CUDA device token loop configuration"
            );
        }

        Ok(Self {
            batch,
            logical_len: 0,
            row_lens: vec![0; batch],
            row_active: vec![true; batch],
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
            // Dormant seam (default off): retaining the M=1 graph across a spec
            // verify+commit cycle is not capture-safe until the M=K verify is
            // itself captured with a pinned workspace. See the field docs and the
            // decision note for the GPU evidence.
            retain_decode_graph_across_spec: false,
            #[cfg(test)]
            padded_query_capacity: None,
            device_token_loop_k,
            device_token_loop_ready,
            device_token_loop_write_position,
            device_token_loop_scratch,
            device_token_loop_steps: 0,
            verify_width: None,
            verify_graph_phase: DecodeCudaGraphPhase::NeedsWarmup,
            verify_token_binding: None,
            verify_position_binding: None,
            verify_logits_binding: None,
            verify_aux_bindings: Vec::new(),
            verify_graph_captures: 0,
            verify_graph_replays: 0,
            verify_graph_fallbacks: 0,
            verify_graph_invalidations: 0,
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

    fn prepare_decode_workspace_after_capacity_growth(
        &mut self,
        session: &mut InferenceSession,
        grew: bool,
    ) -> anyhow::Result<()> {
        if grew {
            session
                .prepare_with_device_bindings(&[], &mut self.bindings)
                .context("prepare native CUDA decode workspace after KV capacity growth")?;
        }
        Ok(())
    }

    fn extend_mask(&mut self, start: usize, end: usize, expose_len: usize) -> anyhow::Result<()> {
        if end > self.max_len || start > end || expose_len > self.max_len || end > expose_len {
            bail!(
                "invalid CUDA mask update {start}..{end} (expose {expose_len}) for capacity {}",
                self.max_len
            );
        }
        // The mask binding is physically `[batch, max_len]`, so row `r`'s
        // `[start, end)` valid span sits `r × max_len` i64s from the base. Batch
        // decode is uniform-length (all sequences step together), so every row
        // gets the identical `1`s window (stage 2b-impl-4, #750). At `batch == 1`
        // this is a single write at `start × i64` — byte-identical to before.
        let ones = (start..end)
            .flat_map(|_| 1i64.to_le_bytes())
            .collect::<Vec<_>>();
        let row_stride = self.max_len * std::mem::size_of::<i64>();
        for sequence in 0..self.batch {
            let offset = sequence * row_stride + start * std::mem::size_of::<i64>();
            self.bindings[0].write_bytes(offset, &ones)?;
        }
        self.bindings[0].set_logical_shape(vec![self.batch, expose_len])?;
        Ok(())
    }

    /// Ragged per-row attention-mask update (stage 3a, #750). Unlike
    /// [`Self::extend_mask`], which writes one identical `1`s window to every row
    /// (uniform-length batch), this writes `valid_lens[r]` leading `1`s to row
    /// `r` and reflects that row's own length. The model reduces each row's mask
    /// to its own `seqlens_k` (= `valid_lens[r] - 1`) and writes present KV at
    /// that per-row offset, so a shared physical KV buffer carries rows of
    /// genuinely different lengths.
    ///
    /// The full `[0, valid_lens[r])` window is (re)written each step. Because
    /// per-row lengths are monotonic under decode (a row's window only grows, or
    /// stays put when the row is held), this leaves the persistent mask buffer in
    /// the same state an incremental write would, while remaining robust across a
    /// bucket reallocation (which discards prior contents). `expose_len` is
    /// frozen to the physical bucket (`max_len`) exactly as the uniform path
    /// does, so only mask *values* — not the mask *shape* — vary per step and
    /// CUDA-graph capture survives (the reduction sees a fixed-width island whose
    /// `1`-count, not its extent, encodes each row's length).
    ///
    /// At `batch == 1` with a single `valid_lens[0]` this is byte-identical to
    /// `extend_mask(0, valid_lens[0], expose_len)`.
    fn extend_mask_ragged(
        &mut self,
        valid_lens: &[usize],
        expose_len: usize,
    ) -> anyhow::Result<()> {
        if valid_lens.len() != self.batch {
            bail!(
                "ragged CUDA mask update expects {} per-row lengths, got {}",
                self.batch,
                valid_lens.len()
            );
        }
        if expose_len > self.max_len {
            bail!(
                "invalid ragged CUDA mask expose {expose_len} for capacity {}",
                self.max_len
            );
        }
        let word = std::mem::size_of::<i64>();
        let row_stride = self.max_len * word;
        for (sequence, &valid) in valid_lens.iter().enumerate() {
            if valid > expose_len {
                bail!(
                    "ragged CUDA mask row {sequence} valid length {valid} exceeds expose {expose_len}"
                );
            }
            let ones = (0..valid)
                .flat_map(|_| 1i64.to_le_bytes())
                .collect::<Vec<_>>();
            let offset = sequence * row_stride;
            self.bindings[0].write_bytes(offset, &ones)?;
        }
        self.bindings[0].set_logical_shape(vec![self.batch, expose_len])?;
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

    /// Zero the whole attention-mask row for sequence `r` (stage 3b, #750).
    ///
    /// The ragged mask writer ([`Self::extend_mask_ragged`]) only (re)writes each
    /// row's `[0, valid_lens[r])` leading `1`s, and the model derives that row's
    /// `seqlens_k` from the *count* of `1`s in its window. A row that is being
    /// recycled for a freshly-admitted sequence would therefore inherit its
    /// previous occupant's trailing `1`s beyond the new (shorter) window, which
    /// would inflate the new sequence's key count and let it attend stale KV.
    /// Wiping the row to all `0`s here means the next `extend_mask_ragged` leaves
    /// exactly the new sequence's window set — no leakage across the reuse
    /// boundary. This is a pure data write into the persistent mask binding (the
    /// same binding every step writes), so it changes no shape and does not
    /// disturb the captured graph or any peer row.
    fn zero_mask_row(&mut self, row: usize) -> anyhow::Result<()> {
        let word = std::mem::size_of::<i64>();
        let row_stride = self.max_len * word;
        let zeros = vec![0u8; row_stride];
        self.bindings[0].write_bytes(row * row_stride, &zeros)?;
        Ok(())
    }

    /// Retire row `r`: mark it inactive so its slot may be recycled (stage 3b,
    /// #750). Host-side only — no device binding shape changes — so a captured
    /// decode graph is untouched and the peer rows keep replaying. The row's KV
    /// and mask are left as-is; [`Self::assign_row`] resets them when the slot is
    /// reused, and until then a deactivated row is simply never stepped
    /// (`advances[r] == false`).
    pub(crate) fn deactivate_row(&mut self, row: usize) -> anyhow::Result<()> {
        if row >= self.batch {
            bail!(
                "native CUDA deactivate_row {row} out of range for pinned batch {}",
                self.batch
            );
        }
        self.row_active[row] = false;
        Ok(())
    }

    /// Admit a fresh sequence into row `r` (stage 3b, #750): reset that row's
    /// cursor to length 0, wipe its mask window so no stale key survives the
    /// reuse boundary, and mark it active. Peers are untouched — their
    /// `row_lens`, mask windows and KV are unchanged — and no binding is
    /// reshaped or reallocated, so the captured decode graph survives (the next
    /// fused step replays the same graph with the reset row simply sitting at
    /// length 0). The row's stale KV is progressively overwritten as the new
    /// sequence prefills (each step writes present KV at its own ascending
    /// offset before that offset is ever attended), and the zeroed mask means it
    /// only ever attends keys it has itself just written.
    pub(crate) fn assign_row(&mut self, row: usize) -> anyhow::Result<()> {
        if row >= self.batch {
            bail!(
                "native CUDA assign_row {row} out of range for pinned batch {}",
                self.batch
            );
        }
        self.zero_mask_row(row)?;
        self.row_lens[row] = 0;
        self.row_active[row] = true;
        Ok(())
    }

    /// Active logical rows in ascending order (stage 3b, #750).
    pub(crate) fn active_rows(&self) -> Vec<usize> {
        (0..self.batch).filter(|&r| self.row_active[r]).collect()
    }

    /// Current logical length of row `r`.
    pub(crate) fn row_len(&self, row: usize) -> anyhow::Result<usize> {
        self.row_lens
            .get(row)
            .copied()
            .with_context(|| format!("native CUDA row_len {row} out of range"))
    }

    /// Read the current `[batch, 1, vocab]` logits binding back to the host as
    /// one `[vocab]` f32 row per batch slot (stage 3b, #750). This is the
    /// host-logits seam the continuous-batch sampler consumes when a real
    /// (non-greedy) sampler is attached, in contrast to the device-argmax fast
    /// path which never round-trips the full logits. Returns the rows plus the
    /// number of bytes transferred device→host and the wall time of that copy so
    /// the caller can report the D2H cost honestly (`[B,1,vocab]` at vocab
    /// 151936 is ~608 KB per row per step in f32).
    fn read_batch_row_logits(
        &mut self,
    ) -> anyhow::Result<(Vec<Vec<f32>>, usize, std::time::Duration)> {
        let start = std::time::Instant::now();
        let bytes = self.bindings[self.logits_binding].read_bytes()?;
        let elapsed = start.elapsed();
        let transferred = bytes.len();
        let logits = Tensor::from_raw(self.logits_dtype, self.logits_shape.clone(), &bytes)?;
        let rows = extract_batch_row_logits(&logits, self.batch)?;
        Ok((rows, transferred, elapsed))
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
        self.row_lens.fill(seq_len);
        let expose = self.decode_mask_expose_len(seq_len);
        self.extend_mask(0, seq_len, expose)?;
        Ok(())
    }

    pub(crate) fn rewind(&mut self, target_len: usize) -> anyhow::Result<()> {
        if target_len < self.logical_len {
            // Clear the now-invalid mask tail on *every* row (the mask binding is
            // physically `[batch, max_len]`). A uniform batch shares one length so
            // one row would suffice, but a ragged batch (stage 3a, #750) can leave
            // per-row `1`s beyond `target_len`; zeroing every row's tail keeps a
            // reused batched session from inheriting a stale attend window.
            let zeros = vec![0u8; (self.logical_len - target_len) * std::mem::size_of::<i64>()];
            let row_stride = self.max_len * std::mem::size_of::<i64>();
            for sequence in 0..self.batch {
                let offset = sequence * row_stride + target_len * std::mem::size_of::<i64>();
                self.bindings[0].write_bytes(offset, &zeros)?;
            }
        }
        self.bindings[0].set_logical_shape(vec![self.batch, target_len])?;
        // Ragged per-row lengths collapse back to the uniform rewind target
        // (stage 3a, #750): after a rewind every row shares `target_len`.
        self.row_lens.fill(target_len);
        if target_len == 0 {
            // Fixed-size recurrent/conv states are unmasked rolling caches: a
            // reused session would otherwise inherit the previous generation's
            // terminal state, corrupting generation #2+. A full reset restores
            // the declared `init: zeros`. A speculative rewind to a *non-zero*
            // length cannot prefix-slice these destructive caches, so it is
            // handled out-of-band by snapshot + accepted-token re-advance
            // (`snapshot_fixed_states` / `restore_fixed_states`, driven by
            // `NativeDecodeSession::commit_recurrent_state_to_accepted`); only
            // the reset boundary re-zeros here.
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

    /// Copy every fixed-size recurrent/conv binding off the device into host
    /// buffers keyed by binding index, capturing the state as of the last
    /// committed token. Paired with [`Self::restore_fixed_states`] for
    /// speculative recurrent-state commit; the bindings are the existing
    /// `fixed_state_binding_range`, so no layer/dim geometry is hardcoded.
    pub(crate) fn snapshot_fixed_states(&mut self) -> anyhow::Result<Vec<(usize, Vec<u8>)>> {
        let mut snapshot = Vec::with_capacity(self.fixed_state_binding_range.len());
        for index in self.fixed_state_binding_range.clone() {
            let binding = &mut self.bindings[index];
            let bytes = checked_shape_bytes(binding.physical_shape(), binding.dtype)
                .with_context(|| {
                    format!(
                        "fixed CUDA decoder state snapshot size overflow for binding {index} shape {:?}",
                        binding.physical_shape()
                    )
                })?;
            let host = binding
                .read_bytes_range(0, bytes)
                .with_context(|| format!("read device recurrent/conv state for binding {index}"))?;
            snapshot.push((index, host));
        }
        Ok(snapshot)
    }

    /// Write a [`Self::snapshot_fixed_states`] capture back into the device
    /// recurrent/conv bindings. Restoring bytes leaves every binding's shape
    /// unchanged, so this never invalidates a captured decode graph.
    pub(crate) fn restore_fixed_states(
        &mut self,
        snapshot: &[(usize, Vec<u8>)],
    ) -> anyhow::Result<()> {
        for (index, host) in snapshot {
            let binding = self.bindings.get_mut(*index).with_context(|| {
                format!("recurrent snapshot names out-of-range fixed-state binding {index}")
            })?;
            let bytes = checked_shape_bytes(binding.physical_shape(), binding.dtype)
                .with_context(|| {
                    format!(
                        "fixed CUDA decoder state restore size overflow for binding {index} shape {:?}",
                        binding.physical_shape()
                    )
                })?;
            if host.len() != bytes {
                bail!(
                    "recurrent snapshot for binding {index} has {} bytes but the binding needs {bytes}",
                    host.len()
                );
            }
            binding.write_bytes(0, host).with_context(|| {
                format!("restore device recurrent/conv state for binding {index}")
            })?;
        }
        Ok(())
    }

    fn write_decode_inputs(&mut self, token_id: TokenId, position: usize) -> anyhow::Result<()> {
        // Single-token decode replicates the token/position across all `batch`
        // rows so the persistent batch grid stays coherent when a single-sequence
        // caller drives a batch-N binding (stage 2b-impl-4, #750). At `batch == 1`
        // this is a single i64 write at offset 0 — byte-identical to before.
        self.write_decode_inputs_batch(&vec![token_id; self.batch], &vec![position; self.batch])
    }

    /// Write N selected token ids into the N `input_ids` slots and N positions
    /// into the `position_ids` binding, one per sequence (stage 2b-impl-4, #750).
    /// The `input_ids` binding is physically `[batch, 1]`, so sequence `s`'s
    /// token is one i64 at offset `s × 8`; positions fan out per
    /// [`Self::write_position_binding_batch`]. `tokens.len()` and
    /// `positions.len()` must both equal the pinned `batch`.
    fn write_decode_inputs_batch(
        &mut self,
        tokens: &[TokenId],
        positions: &[usize],
    ) -> anyhow::Result<()> {
        if tokens.len() != self.batch || positions.len() != self.batch {
            bail!(
                "native CUDA batch decode expects {} tokens and {} positions, got {} / {}",
                self.batch,
                self.batch,
                tokens.len(),
                positions.len()
            );
        }
        let word = std::mem::size_of::<i64>();
        for (sequence, token) in tokens.iter().enumerate() {
            self.bindings[self.input_ids_binding]
                .write_bytes(sequence * word, &i64::from(*token).to_le_bytes())?;
        }
        self.write_position_binding_batch(positions)
    }

    /// Write the current position into the persistent `position_ids` device
    /// binding, replicated across every declared coordinate axis (`position_rank`
    /// copies). For a rank-1 decoder this is a single `i64` at offset 0, identical
    /// to before; a rank-N mrope decoder gets `[position; N]` — the one-token
    /// `linear_increment` coordinate for all axes.
    fn write_position_binding(&mut self, position: usize) -> anyhow::Result<()> {
        self.write_position_binding_batch(&vec![position; self.batch])
    }

    /// Write N per-sequence positions into the persistent `position_ids` binding
    /// (stage 2b-impl-4, #750). A rank-1 decoder binds `position_ids [batch, 1]`,
    /// so sequence `s`'s position is one i64 at offset `s × 8`. A rank-R mrope
    /// decoder binds `[R, batch, 1]`; element `(r, s)` sits at `(r × batch + s) × 8`
    /// and every coordinate axis gets the same per-sequence position (the
    /// one-token `linear_increment`). At `batch == 1, rank == 1` this is a single
    /// i64 at offset 0 — byte-identical to before.
    fn write_position_binding_batch(&mut self, positions: &[usize]) -> anyhow::Result<()> {
        if positions.len() != self.batch {
            bail!(
                "native CUDA batch decode expects {} positions, got {}",
                self.batch,
                positions.len()
            );
        }
        if let Some(index) = self.position_ids_binding {
            let axis_bytes = std::mem::size_of::<i64>();
            for axis in 0..self.position_rank {
                for (sequence, position) in positions.iter().enumerate() {
                    let position =
                        i64::try_from(*position).context("position id exceeds i64 range")?;
                    let offset = (axis * self.batch + sequence) * axis_bytes;
                    self.bindings[index].write_bytes(offset, &position.to_le_bytes())?;
                }
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

    /// The configured device-token-loop chain depth (`0` when disabled or the
    /// topology is not device-loopable).
    pub(crate) fn device_token_loop_k(&self) -> usize {
        self.device_token_loop_k
    }

    /// Launch the device argmax over the current logits into `greedy_result`
    /// **without** the host read-back, so a chained replay can proceed on-stream
    /// with no host sync (the device token-writer consumes `greedy_result[0]`).
    fn run_device_argmax(&mut self) -> anyhow::Result<()> {
        let vocab = *self
            .logits_shape
            .last()
            .context("CUDA logits shape has no vocabulary dimension")?;
        self.bindings[self.logits_binding].device_argmax(
            vocab,
            self.batch,
            &mut self.greedy_result,
        )?;
        Ok(())
    }

    /// Clear the device-token-loop capture-error accumulator (the u32 word at
    /// index `k` of the scratch binding) before enqueuing a fresh chain.
    fn reset_device_token_loop_error(&mut self) -> anyhow::Result<()> {
        let k = self.device_token_loop_k;
        let scratch = self
            .device_token_loop_scratch
            .as_mut()
            .context("device token loop scratch is not allocated")?;
        scratch.write_bytes(k * std::mem::size_of::<u32>(), &0u32.to_ne_bytes())?;
        Ok(())
    }

    /// Clear the single trailing attention-mask `1` the device token-writer set
    /// one position beyond the chain's last consumed token. The writer sets
    /// `mask[next_position]` on *every* step (so the following in-chain replay
    /// attends the new token); on the final step there is no following replay, so
    /// that bit is one past `current_len` and must be undone to leave the mask in
    /// exactly the state the per-token path leaves — otherwise, with the mask
    /// frozen to physical width (`mask_exposes_logical == false`), that stray `1`
    /// inflates the derived sequence length for a later prefill after `reset()`.
    /// No-op when the position is at/over physical capacity (the writer guarded
    /// it, so nothing was written).
    fn clear_trailing_mask_bit(&mut self, position: usize) -> anyhow::Result<()> {
        if position >= self.max_len {
            return Ok(());
        }
        let word = std::mem::size_of::<i64>();
        let row_stride = self.max_len * word;
        for sequence in 0..self.batch {
            let offset = sequence * row_stride + position * word;
            self.bindings[0].write_bytes(offset, &0i64.to_le_bytes())?;
        }
        Ok(())
    }

    /// Stitch the just-selected token (in `greedy_result`) into the persistent
    /// decode bindings for the next replay: token id → `input_ids`,
    /// `next_position` → `position_ids`, mask `1` at `next_position`, token →
    /// `scratch[step]`, capture-error OR-ed into `scratch[k]`. Device-to-device,
    /// no host sync.
    fn device_token_writer_step(&self, next_position: i64, step: u32) -> anyhow::Result<()> {
        let scratch = self
            .device_token_loop_scratch
            .as_ref()
            .context("device token loop scratch is not allocated")?;
        let position_binding = if self.device_token_loop_write_position {
            let position_index = self
                .position_ids_binding
                .context("device token loop requires a position_ids binding")?;
            Some(&self.bindings[position_index])
        } else {
            None
        };
        self.greedy_result.device_token_writer(
            &self.bindings[self.input_ids_binding],
            position_binding,
            &self.bindings[0],
            scratch,
            self.device_token_loop_k,
            next_position,
            self.max_len,
            step,
        )?;
        Ok(())
    }

    /// Drain the enqueued chain in one D2H read: the first `produced` token-log
    /// slots as token ids, plus the OR-ed capture-error accumulator word.
    fn drain_device_token_loop(&mut self, produced: usize) -> anyhow::Result<(Vec<TokenId>, u32)> {
        let k = self.device_token_loop_k;
        let scratch = self
            .device_token_loop_scratch
            .as_mut()
            .context("device token loop scratch is not allocated")?;
        let word = std::mem::size_of::<u32>();
        let mut bytes = vec![0_u8; (k + 1) * word];
        scratch.read_bytes_into(&mut bytes)?;
        let mut tokens = Vec::with_capacity(produced);
        for slot in 0..produced {
            let base = slot * word;
            tokens.push(u32::from_ne_bytes(
                bytes[base..base + word]
                    .try_into()
                    .map_err(|_| anyhow::anyhow!("four token-id bytes"))?,
            ));
        }
        let error = u32::from_ne_bytes(
            bytes[k * word..k * word + word]
                .try_into()
                .map_err(|_| anyhow::anyhow!("four capture-error bytes"))?,
        );
        Ok((tokens, error))
    }

    pub(crate) fn greedy_fastpath_supported(&self) -> bool {
        self.bindings[self.logits_binding].device_argmax_supported()
    }

    /// The pinned persistent-decode batch extent (stage 2b-impl-4, #750).
    pub(crate) fn batch(&self) -> usize {
        self.batch
    }

    /// The hard physical KV capacity (`max_len`) ceiling this decode state can
    /// grow to. Used by the continuous-batch manager to clamp per-request context
    /// limits to what the persistent bindings can actually hold.
    pub(crate) fn hard_max_len(&self) -> usize {
        self.capacity.max_len
    }

    /// Run the batched device argmax over the `[batch, 1, vocab]` logits binding
    /// and read back the `batch` selected token ids paired with the shared
    /// device capture-error word (stage 2b-impl-3, #750). Row `i` of the returned
    /// vector is the greedy token of sequence `i`; every row carries the same
    /// latched capture-error bitmask (the kernel copies the one shared word into
    /// each slot). At `batch == 1` the argmax launch geometry and the 8-byte
    /// read-back are byte-identical to the previous single-sequence path.
    fn read_greedy_result_batch(&mut self) -> anyhow::Result<Vec<(TokenId, u32)>> {
        let vocab = *self
            .logits_shape
            .last()
            .context("CUDA logits shape has no vocabulary dimension")?;
        self.bindings[self.logits_binding].device_argmax(
            vocab,
            self.batch,
            &mut self.greedy_result,
        )?;
        let word = std::mem::size_of::<u32>();
        let mut bytes = vec![0_u8; self.batch * 2 * word];
        self.greedy_result.read_bytes_into(&mut bytes)?;
        let mut rows = Vec::with_capacity(self.batch);
        for sequence in 0..self.batch {
            let base = sequence * 2 * word;
            let token = u32::from_ne_bytes(
                bytes[base..base + word]
                    .try_into()
                    .map_err(|_| anyhow::anyhow!("four token-id bytes"))?,
            );
            let capture_error = u32::from_ne_bytes(
                bytes[base + word..base + 2 * word]
                    .try_into()
                    .map_err(|_| anyhow::anyhow!("four capture-error bytes"))?,
            );
            rows.push((token, capture_error));
        }
        Ok(rows)
    }

    fn read_greedy_result(&mut self) -> anyhow::Result<(TokenId, u32)> {
        let rows = self.read_greedy_result_batch()?;
        rows.into_iter()
            .next()
            .context("device argmax returned no rows")
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

    /// Swap the persistent padded verify bindings (`[1, M]` token / position,
    /// `[1, M, vocab]` logits, `[1, M, ...]` aux) into the device binding vector
    /// in place of the M=1 decode bindings, or swap them back out. `std::mem::swap`
    /// is its own inverse, so calling this twice around a verify forward restores
    /// the decode bindings exactly. Inert (no fields set) until
    /// [`NativeDecodeSession::configure_verify_capture`] has run.
    fn swap_verify_bindings(&mut self) {
        let token_index = self.input_ids_binding;
        let position_index = self.position_ids_binding;
        let logits_index = self.logits_binding;
        let aux_start = self.auxiliary_binding_range.start;
        if let Some(token) = self.verify_token_binding.as_mut() {
            std::mem::swap(&mut self.bindings[token_index], token);
        }
        if let (Some(index), Some(position)) =
            (position_index, self.verify_position_binding.as_mut())
        {
            std::mem::swap(&mut self.bindings[index], position);
        }
        if let Some(logits) = self.verify_logits_binding.as_mut() {
            std::mem::swap(&mut self.bindings[logits_index], logits);
        }
        for (offset, aux) in self.verify_aux_bindings.iter_mut().enumerate() {
            std::mem::swap(&mut self.bindings[aux_start + offset], aux);
        }
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
                device_token_loop_k: self.device_token_loop_k,
                device_token_loop_steps: self.device_token_loop_steps,
                verify_captures: self.verify_graph_captures,
                verify_replays: self.verify_graph_replays,
                verify_fallbacks: self.verify_graph_fallbacks,
                verify_invalidations: self.verify_graph_invalidations,
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
                vec![self.state.batch, new_capacity],
                vec![self.state.batch, valid_len],
            )?;
            native_cuda_memset_zero(
                new_mask.device_ptr() as usize,
                self.state
                    .batch
                    .checked_mul(new_capacity)
                    .and_then(|elements| elements.checked_mul(std::mem::size_of::<i64>()))
                    .with_context(|| {
                        format!(
                            "legacy realloc CUDA mask growth overflows for capacity {new_capacity}"
                        )
                    })?,
            )?;
            new_mask.set_logical_shape(vec![self.state.batch, valid_len])?;
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

/// Read the last row of a declared hidden-state auxiliary output from its
/// persistent CUDA device binding (the captured single-token decode step leaves
/// this output device-resident). Used to seed the MTP proposer with the target's
/// hidden state after a captured decode step; the eager multi-row path reads the
/// same value from the host-materialized output instead. Locating the binding by
/// its declared output name keeps this free of hardcoded layer/dim assumptions.
fn read_aux_hidden_last_row(
    state: &mut DecodeCudaState,
    hidden_output: &str,
) -> anyhow::Result<Vec<f32>> {
    let index = state
        .auxiliary_binding_range
        .clone()
        .find(|&index| state.bindings[index].output_name() == Some(hidden_output))
        .with_context(|| {
            format!(
                "declared hidden output '{hidden_output}' has no persistent CUDA auxiliary binding; the MTP hidden seed must be a bound graph output"
            )
        })?;
    let binding = &mut state.bindings[index];
    let dtype = binding.dtype;
    let shape = binding.physical_shape().to_vec();
    let bytes = binding.read_bytes()?;
    let tensor = Tensor::from_raw(dtype, shape, &bytes)?;
    extract_last_row(&tensor)
        .with_context(|| format!("read native decoder hidden output '{hidden_output}'"))
}

#[cfg(feature = "native-cuda")]
fn native_cuda_device_barrier(session: &InferenceSession) -> anyhow::Result<()> {
    let _guard = onnx_genai_ort::cuda_rt::DeviceGuard::set(session.device_id().index as i32)?;
    onnx_genai_ort::cuda_rt::device_synchronize()?;
    Ok(())
}

#[cfg(not(feature = "native-cuda"))]
fn native_cuda_device_barrier(_session: &InferenceSession) -> anyhow::Result<()> {
    bail!("native CUDA KV growth requires the onnx-genai-engine `cuda` feature")
}

#[cfg(feature = "native-cuda")]
fn native_cuda_memset_zero(dst: usize, bytes: usize) -> anyhow::Result<()> {
    onnx_genai_ort::cuda_rt::memset_zero(dst, bytes)?;
    Ok(())
}

#[cfg(not(feature = "native-cuda"))]
fn native_cuda_memset_zero(_dst: usize, _bytes: usize) -> anyhow::Result<()> {
    bail!("native CUDA KV growth requires the onnx-genai-engine `cuda` feature")
}

/// Resolve the persistent decode batch extent (stage 2b-impl-4, #750).
///
/// An explicit `configured` value — from `NativeDecodeCudaOptions::decode_batch`,
/// which is how `--max-batch` reaches here — wins over
/// `ONNX_GENAI_NATIVE_DECODE_BATCH`. Neither set resolves to `1`, the
/// byte-identity default that keeps every current single-sequence entry point
/// unchanged. A value `> 1` turns batch-N on: the KV / mask / input / position /
/// logits bindings are shaped `[N, …]`, the token writer fans N selected tokens
/// into N slots, and the device argmax reads N rows. The value is pinned for the
/// whole session (capture requires a stable batch shape), so it is read exactly
/// once at construction. `0` is rejected — a zero-width decode has no meaning.
///
/// Both sources are validated the same way, so an explicit request cannot bypass
/// a check the environment variable is subject to.
fn resolve_native_decode_batch(configured: Option<usize>) -> anyhow::Result<usize> {
    if let Some(batch) = configured {
        if batch == 0 {
            bail!("native decode batch must be > 0, got 0");
        }
        return Ok(batch);
    }
    match std::env::var("ONNX_GENAI_NATIVE_DECODE_BATCH") {
        Ok(raw) if !raw.trim().is_empty() => {
            let batch: usize = raw.trim().parse().with_context(|| {
                format!("ONNX_GENAI_NATIVE_DECODE_BATCH must be a positive integer, got '{raw}'")
            })?;
            if batch == 0 {
                bail!("ONNX_GENAI_NATIVE_DECODE_BATCH must be > 0, got 0");
            }
            Ok(batch)
        }
        _ => Ok(1),
    }
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
///
/// **Batch-N refusal (stage 2b-impl-2, #750).** This helper feeds the two
/// *growing-bucket* copy paths ([`DecodeCudaState::apply_vmm_growth`] and the
/// legacy realloc `build_grown_buffers`), whose per-block stride *is* the
/// mutable bucket capacity. Head-major keeps the batch axis outermost, so each
/// `(batch, head)` block re-strides independently and batch-N growth is exact.
/// Seq-major is different: with `batch > 1` the per-sequence stride is the
/// growing bucket capacity, so growing the bucket relocates every sequence
/// `b > 0`'s KV bytes. That relocation would defeat the whole point of the
/// seq-major fixed-stride design (the "no KV moves / keep the captured graph"
/// contract) and the `moved_bytes_per_token == 0` growth accounting. Rather than
/// silently compute a wrong (capacity-dependent) stride, this refuses seq-major
/// growing-bucket growth at `batch > 1` with a named error. The relocation-free
/// batch-N seq-major path is the *fixed full-context stride* commit
/// ([`DecodeCudaState::seq_major_kv_commit_requests`]), which never calls this
/// helper. Turning batch-N on end-to-end is 2b-impl-4.
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
            if batch > 1 {
                bail!(
                    "native CUDA seq-major (BSNH) KV *growing-bucket* growth cannot be made \
                     batch-general for batch {batch} > 1 without relocating KV: a sequence's \
                     per-sequence stride is the mutable bucket capacity ({capacity}), so growing \
                     the bucket moves every sequence b>0's already-written KV bytes. Head-major \
                     (BNSH) does not have this problem because its batch axis is outermost and \
                     re-strides per (batch, head) block without moving batch 0. The relocation-free \
                     batch-N seq-major path is the fixed full-context stride commit-on-demand path, \
                     not this growing bucket. Refusing rather than computing a capacity-dependent \
                     stride (stage 2b-impl-2, #750; batch-N is turned on in 2b-impl-4)."
                );
            }
            (vec![batch, capacity, kv_heads, head_dim], 1)
        }
    })
}

#[cfg(feature = "native-cuda")]
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

#[cfg(feature = "native-cuda")]
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
#[cfg(feature = "native-cuda")]
pub(super) enum InPlaceCopyRoute {
    Noop,
    DeviceToDevice,
    Scratch,
}

#[cfg(feature = "native-cuda")]
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

#[cfg(not(feature = "native-cuda"))]
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

#[cfg(feature = "native-cuda")]
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

#[cfg(not(feature = "native-cuda"))]
fn zero_kv_suffix_device(
    _ptr: usize,
    _shape: &[usize],
    _seq_axis: usize,
    _valid_len: usize,
    _elem_size: usize,
) -> anyhow::Result<()> {
    bail!("native CUDA KV growth requires the onnx-genai-engine `cuda` feature")
}

#[cfg(not(feature = "native-cuda"))]
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

#[cfg(feature = "native-cuda")]
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

#[cfg(not(feature = "native-cuda"))]
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
mod decode_batch_resolution_tests {
    use super::resolve_native_decode_batch;

    /// The default is 1, which is the #750 byte-identity reference: every shape,
    /// binding and read-back at batch 1 matches the previous hard-coded value.
    #[test]
    fn nothing_requested_and_no_env_resolves_to_one() {
        // Guard against a stray environment on the test host: this assertion is
        // only meaningful when the variable is genuinely absent.
        if std::env::var("ONNX_GENAI_NATIVE_DECODE_BATCH").is_ok() {
            return;
        }
        assert_eq!(resolve_native_decode_batch(None).unwrap(), 1);
    }

    /// #1064: an explicit request must win over the environment, so `--max-batch`
    /// is authoritative rather than advisory. Before this, batch-N could only be
    /// turned on by an environment variable, and the server refused `--max-batch
    /// N` because the capability was read from a session nobody had asked to
    /// build in batch shape.
    #[test]
    fn an_explicit_request_is_honoured() {
        assert_eq!(resolve_native_decode_batch(Some(4)).unwrap(), 4);
        assert_eq!(resolve_native_decode_batch(Some(1)).unwrap(), 1);
    }

    /// A zero-width decode has no meaning, and both sources are validated the
    /// same way — an explicit request must not bypass a check the environment
    /// variable is subject to.
    #[test]
    fn zero_is_refused_from_either_source() {
        let error = resolve_native_decode_batch(Some(0)).expect_err("zero-width decode");
        assert!(error.to_string().contains("must be > 0"), "{error}");
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

#[cfg(test)]
mod prefill_query_width_tests {
    use super::{Tensor, pad_step_tensor, prefill_query_width};

    /// A single-row decode step keeps its exact shape: the steady-state decode
    /// kernel is the one shape that must never be perturbed.
    #[test]
    fn a_single_query_row_is_never_padded() {
        assert_eq!(prefill_query_width(1, 512), 1);
    }

    /// A model that declares no chunk width declares no working width either,
    /// so there is nothing to round up to.
    #[test]
    fn without_a_declared_chunk_width_nothing_is_padded() {
        for rows in [2, 37, 512, 4096] {
            assert_eq!(prefill_query_width(rows, 0), rows);
        }
    }

    /// Widths land on a three-step ladder up to the chunk width, leaving the
    /// kernel cache's four-variant per-node bound one slot for the single-token
    /// decode shape.
    #[test]
    fn widths_round_up_to_a_three_step_ladder() {
        let widths: std::collections::BTreeSet<usize> = (2..=512)
            .map(|rows| prefill_query_width(rows, 512))
            .collect();
        assert_eq!(widths.into_iter().collect::<Vec<_>>(), vec![171, 342, 512]);
    }

    /// Padding never costs more than one step of duplicated rows, and never
    /// runs narrower than the real work.
    #[test]
    fn padding_never_narrows_and_never_overshoots_a_step() {
        for rows in 2..=512 {
            let width = prefill_query_width(rows, 512);
            assert!(width >= rows, "{rows} rows must not run at {width}");
            assert!(
                width - rows < 171,
                "{rows} rows wasted {} rows",
                width - rows
            );
        }
    }

    /// A forward wider than the declared chunk already amortizes its compile
    /// over enough work, and has no ladder step above it to round to.
    #[test]
    fn a_forward_wider_than_the_chunk_runs_exactly() {
        assert_eq!(prefill_query_width(513, 512), 513);
    }

    /// A tiny chunk width still gets a floor, so a two-row forward is not
    /// rounded to a ladder finer than the work is worth.
    #[test]
    fn a_tiny_chunk_width_keeps_the_minimum_step() {
        assert_eq!(prefill_query_width(2, 32), 16);
        assert_eq!(prefill_query_width(17, 32), 32);
    }

    /// The ladder never spends more slots than the kernel cache's per-node
    /// bound leaves for prefill, whatever chunk width a model declares.
    #[test]
    fn the_ladder_never_outgrows_its_step_count() {
        for cap in [32usize, 128, 256, 512, 1024, 4096] {
            let widths: std::collections::BTreeSet<usize> = (2..=cap)
                .map(|rows| prefill_query_width(rows, cap))
                .collect();
            assert!(
                widths.len() <= super::PREFILL_QUERY_WIDTH_STEPS,
                "chunk width {cap} produced {widths:?}"
            );
            assert!(widths.contains(&cap), "chunk width {cap} must be a step");
        }
    }

    /// Padded rows repeat the last real row, so the extra rows carry values in
    /// the same range as the real ones.
    #[test]
    fn padding_a_step_tensor_repeats_its_last_row() {
        let tensor = Tensor::from_f32(&[1, 3, 2], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let padded = pad_step_tensor(&tensor, 3, 5).expect("a [1, rows, width] tensor pads");
        assert_eq!(padded.shape, vec![1, 5, 2]);
        assert_eq!(
            padded.to_vec_f32(),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 5.0, 6.0, 5.0, 6.0]
        );
    }

    /// A port whose rows this code cannot recognize is refused rather than
    /// guessed at: inventing rows for it would change what the forward computes.
    #[test]
    fn a_tensor_that_is_not_per_token_refuses_to_pad() {
        let wrong_rows = Tensor::from_i64(&[1, 4], &[1, 2, 3, 4]).unwrap();
        assert!(pad_step_tensor(&wrong_rows, 3, 5).is_none());
        let wrong_batch = Tensor::from_i64(&[2, 3], &[1, 2, 3, 4, 5, 6]).unwrap();
        assert!(pad_step_tensor(&wrong_batch, 3, 5).is_none());
        let rank_one = Tensor::from_i64(&[3], &[1, 2, 3]).unwrap();
        assert!(pad_step_tensor(&rank_one, 3, 5).is_none());
    }

    /// Asking for no extra rows is not padding, and reports so rather than
    /// handing back a needless copy.
    #[test]
    fn padding_to_the_same_width_is_declined() {
        let tensor = Tensor::from_i64(&[1, 3], &[1, 2, 3]).unwrap();
        assert!(pad_step_tensor(&tensor, 3, 3).is_none());
    }
}

#[cfg(test)]
mod verify_capture_helper_tests {
    use super::{DecodeCudaGraphPhase, NativeDecodeSession, widen_query_last, widen_query_seq};

    /// `widen_query_last` widens the trailing query-seq axis of a token/position
    /// binding shape to the fixed verify width M, and only when that axis is
    /// currently unit (so the padded verify binding is derived from geometry, not
    /// hardcoded dims).
    #[test]
    fn widen_query_last_replaces_unit_trailing_axis() {
        assert_eq!(widen_query_last(&[1, 1], 2), Some(vec![1, 2]));
        assert_eq!(widen_query_last(&[3, 1, 1], 4), Some(vec![3, 1, 4]));
        // Non-unit trailing axis (an already-widened / prefill shape) is declined.
        assert_eq!(widen_query_last(&[1, 2], 3), None);
        assert_eq!(widen_query_last(&[], 2), None);
    }

    /// `widen_query_seq` widens the `len-2` query-seq axis of a logits/aux output
    /// shape (`[1, 1, feat]` -> `[1, M, feat]`), leaving the trailing feature dim
    /// intact, and only when the query-seq axis is currently unit.
    #[test]
    fn widen_query_seq_replaces_unit_query_axis() {
        assert_eq!(widen_query_seq(&[1, 1, 248320], 2), Some(vec![1, 2, 248320]));
        assert_eq!(widen_query_seq(&[1, 1, 5120], 3), Some(vec![1, 3, 5120]));
        // Non-unit query axis or too-low rank is declined.
        assert_eq!(widen_query_seq(&[1, 2, 5120], 3), None);
        assert_eq!(widen_query_seq(&[5120], 2), None);
    }

    /// The verify graph re-warms on the first clobber (in case the churn was a
    /// one-off KV-bucket growth) but latches to eager (`Unsupported`) once the
    /// clobber persists, so a single-executor two-slot contention never becomes a
    /// per-step recapture cost.
    #[test]
    fn verify_phase_latches_to_eager_after_repeated_invalidation() {
        assert_eq!(
            NativeDecodeSession::verify_phase_after_invalidation(1),
            DecodeCudaGraphPhase::NeedsWarmup
        );
        assert_eq!(
            NativeDecodeSession::verify_phase_after_invalidation(2),
            DecodeCudaGraphPhase::Unsupported
        );
        assert_eq!(
            NativeDecodeSession::verify_phase_after_invalidation(9),
            DecodeCudaGraphPhase::Unsupported
        );
    }

    /// Greedy-inertness contract: a decoder that never configured verify capture
    /// (every non-MTP path, and MTP on a model without a decode-inline sibling)
    /// reports the verify capture inactive, so the padded-binding swap + Verify
    /// graph slot machinery stays dormant and the decode path is byte-identical.
    #[test]
    fn verify_capture_inactive_by_default() {
        fn assert_inactive(session: &NativeDecodeSession) {
            assert!(!session.verify_capture_active());
            assert_eq!(session.verify_capture_width(), None);
        }
        let _ = assert_inactive;
    }
}
