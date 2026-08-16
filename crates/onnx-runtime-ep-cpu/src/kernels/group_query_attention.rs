//! `com.microsoft::GroupQueryAttention` — optimized CPU GQA kernel.
//!
//! Implements unpacked Q/K/V and packed QKV inputs, BNSH KV caches, causal and
//! local-window masking, rotary embedding, and score softcap. Packed KV,
//! quantized caches, attention bias, smooth softmax/head sink, and QK capture
//! are rejected.
//!
//! ## Performance design (M=1 decode, long context)
//!
//! The decode hot path is a GEMV over the KV cache, executed per
//! `(batch, query_head, query_seq)` row.  Three targeted optimizations reduce
//! GQA latency at long context relative to the scalar reference:
//!
//! 1. **Attended-window scoring only**: scores are computed and stored only for
//!    the `[local_start, causal_limit]` range; unattended positions are never
//!    written to a full-length scratch buffer.
//! 2. **Shared decode SDPA core**: Q·K scoring, softcap, softmax, and P·V
//!    accumulation delegate to [`super::sdpa::sdpa_decode_row`], including its
//!    AVX2+FMA dot/AXPY implementation.
//!
//! ### Precision contract (RULES.md §4 / cross-EP parity)
//! Softmax uses the **exact** `(score - max) as f64).exp() as f32` path, unchanged
//! from the original.  The dot-product and AXPY SIMD paths may reorder f32
//! additions (parallel accumulator reduction).  Under the standard
//! floating-point model, a length-`n` dot product has forward error proportional
//! to `γ_n × Σ|a_i b_i|`, where `γ_n = n u / (1 - n u)` and the unit roundoff
//! for round-to-nearest f32 is `u = 0.5 × f32::EPSILON`.  This is a numerical
//! parity contract, not a universal greedy-token identity guarantee; model-level
//! greedy parity is established empirically by profiling.

use super::sdpa::{
    DecodePartial, SoftmaxExp, combine_decode_partials, sdpa_decode_group, sdpa_decode_partial,
    sdpa_decode_row,
};
use super::{check_arity, to_dense_i64};
use crate::dtype::{to_dense_f32_widen, widen_bf16_slice_into, write_dense_f32_narrow};
use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::Node;

// Below this many row × key × head-dimension elements, Rayon synchronization
// costs more than the attention work on the decode pool.
const MIN_PARALLEL_ATTENTION_WORK: usize = 160 * 1024;

// Flash-decoding (split-KV) engages only when a single head's attended KV window
// is at least this long: below it the per-head path already parallelizes well
// enough that the split's extra fork-join, per-call scratch allocation, and
// combine cost is not worth paying (measured: a slight regression at a ~1024
// window on a single memory-bandwidth-bound socket, a clear win from ~2048 up).
const SPLIT_MIN_KV: usize = 1536;

// Minimum KV rows per split chunk. Splitting finer than this lets fork-join
// overhead (~50µs per decode dispatch) dominate the per-chunk streaming work, so
// the split count is capped so every chunk still streams at least this many KV
// rows.
const SPLIT_MIN_CHUNK: usize = 512;

/// Whether flash-decoding KV splitting is enabled (default on). Set
/// `ONNX_GENAI_ATTENTION_SPLIT=0` (or `false`/`off`) to force the per-head decode
/// path — used to A/B the split against the baseline on the same binary.
fn attention_split_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| match std::env::var("ONNX_GENAI_ATTENTION_SPLIT") {
        Ok(value) => !matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "0" | "false" | "off"
        ),
        Err(_) => true,
    })
}

/// Count of decode attention forwards that engaged the flash-decoding split
/// path. Cheap observability for A/B runs; read via [`attention_split_count`].
static ATTENTION_SPLIT_COUNT: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// Number of decode forwards that took the flash-decoding split path so far.
pub fn attention_split_count() -> usize {
    ATTENTION_SPLIT_COUNT.load(std::sync::atomic::Ordering::Relaxed)
}

/// Test-only tri-state override of [`group_fusion_enabled`]: `-1` defers to the
/// environment, `0` forces off, `1` forces on.
static GROUP_FUSION_OVERRIDE: std::sync::atomic::AtomicI8 = std::sync::atomic::AtomicI8::new(-1);

/// Whether the M=1 KV-group-fused decode path is enabled. **Opt-in**: set
/// `ONNX_GENAI_GQA_GROUP_FUSED=1` (or `true`/`on`) to enable it.
///
/// The fused schedule is bit-identical and, where both gates open, is a
/// 1.25x – 2.76x win on the attention operator, 1.31x end to end at a fixed
/// decode width, and a 6% – 22% wall-clock win in a model-level A/B against a
/// real ORT CPU session (see [`group_fused_min_kv_bytes`] for the matrix).
///
/// It is nonetheless still off by default. [`group_fused_min_kv_bytes`] now
/// reads the host's last-level cache and can only *tighten* the calibrated
/// 8 MiB threshold, which closes the large-L3 hole the fixed constant had. What
/// remains unproven is the other direction: the sweep that forced the traffic
/// gate fully open could not separate the 1 – 4 MiB per-head range from a +-9%
/// contention noise floor, so the true crossover on an unfamiliar host is still
/// unknown, and no model-level measurement exists for the batch >= 2 regime the
/// pool gate also admits.
///
/// Flip the default once a wall-clock A/B on a quiet host resolves the
/// crossover to better than the noise floor at a reachable geometry.
fn group_fusion_enabled() -> bool {
    match GROUP_FUSION_OVERRIDE.load(std::sync::atomic::Ordering::Relaxed) {
        0 => false,
        1 => true,
        _ => group_fusion_env_default(),
    }
}

/// Environment default for [`group_fusion_enabled`], latched on first read.
fn group_fusion_env_default() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| match std::env::var("ONNX_GENAI_GQA_GROUP_FUSED") {
        Ok(value) => matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "on"
        ),
        Err(_) => false,
    })
}

/// Count of decode attention forwards that engaged the KV-group-fused path.
/// Cheap reachability evidence for A/B runs; read via [`group_fused_count`].
static GROUP_FUSED_CALLS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Number of decode forwards that took the KV-group-fused path so far.
pub fn group_fused_count() -> usize {
    GROUP_FUSED_CALLS.load(std::sync::atomic::Ordering::Relaxed)
}

/// Serialises tests that force [`group_fusion_enabled`] on, so the override
/// cannot leak into a test running concurrently in the same process.
#[cfg(test)]
static GROUP_FUSION_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Forces [`group_fusion_enabled`] on for the lifetime of the guard. Tests need
/// this because the path is opt-in and the environment default latches in a
/// `OnceLock`, so it cannot be toggled twice in one process.
#[cfg(test)]
struct GroupFusionOverride {
    _guard: std::sync::MutexGuard<'static, ()>,
}

#[cfg(test)]
impl GroupFusionOverride {
    fn forced_on() -> Self {
        let guard = GROUP_FUSION_TEST_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        GROUP_FUSION_OVERRIDE.store(1, std::sync::atomic::Ordering::Relaxed);
        // Pin the traffic threshold too: the bit-identity tests pick a geometry
        // that must reach the fused path, and the production threshold now
        // depends on the runner's cache size.
        GROUP_FUSED_MIN_KV_BYTES_PIN.store(8 << 20, std::sync::atomic::Ordering::Relaxed);
        Self { _guard: guard }
    }
}

#[cfg(test)]
impl Drop for GroupFusionOverride {
    fn drop(&mut self) {
        GROUP_FUSION_OVERRIDE.store(-1, std::sync::atomic::Ordering::Relaxed);
        GROUP_FUSED_MIN_KV_BYTES_PIN.store(0, std::sync::atomic::Ordering::Relaxed);
    }
}

/// Test-only pin for the per-head traffic threshold (`0` = unset). Without it
/// every gate assertion would depend on the runner's last-level cache, so a
/// CI host with a small L3 would silently take a different branch than the
/// developer machine the test was written on.
#[cfg(test)]
static GROUP_FUSED_MIN_KV_BYTES_PIN: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// Pins the per-head traffic threshold for the lifetime of the guard.
#[cfg(test)]
struct FusedTrafficThresholdPin {
    _guard: std::sync::MutexGuard<'static, ()>,
}

#[cfg(test)]
impl FusedTrafficThresholdPin {
    fn bytes_per_head(bytes: usize) -> Self {
        let guard = GROUP_FUSION_TEST_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        GROUP_FUSED_MIN_KV_BYTES_PIN.store(bytes, std::sync::atomic::Ordering::Relaxed);
        Self { _guard: guard }
    }
}

#[cfg(test)]
impl Drop for FusedTrafficThresholdPin {
    fn drop(&mut self) {
        GROUP_FUSED_MIN_KV_BYTES_PIN.store(0, std::sync::atomic::Ordering::Relaxed);
    }
}

/// Whether collapsing the decode schedule to one task per `(batch, kv_head)`
/// still keeps every decode worker busy.
///
/// KV-group fusion trades parallel tasks for KV-cache traffic: it turns
/// `batch * num_heads` independent attention rows into `batch * kv_num_heads`
/// group tasks. Paired A/B on an 8-worker decode pool shows the trade is only
/// worth taking while the pool stays saturated — starving it costs far more
/// than the traffic saving is worth:
///
/// | geometry (heads/kv_heads/head_size) | fused tasks | fused vs per-head |
/// |---|---|---|
/// | 14/2/64   | 2 | 0.32x – 0.39x (2.6x slower) |
/// | 28/4/128  | 4 | 0.59x – 0.75x (1.7x slower) |
/// | 32/8/128  | 8 | 0.89x – 2.05x (length dependent) |
///
/// Repeating the sweep on a 16-worker pool (the width this host actually
/// resolves for decode) reproduces the same shape: half-covered pools
/// (8 fused tasks against 16 workers) lose 0.64x – 0.87x on short caches, while
/// fully-covered pools (16 fused tasks) win 1.24x – 2.76x. Half coverage does
/// recover on very long caches, but the sign is not stable across pool widths —
/// it lost at every length measured on the 8-worker pool — so the gate stays at
/// full coverage: "the fused task count still covers the workers".
/// A single-worker scope (`worker_count <= 1`) has no parallelism to lose, so
/// the traffic saving is taken unconditionally there.
fn group_fusion_saturates_pool(fused_tasks: usize, worker_count: usize) -> bool {
    fused_tasks >= worker_count
}

/// Last-level cache assumed when the host's cache topology cannot be read.
///
/// Guessing *small* here is the safe direction: the topology term can only make
/// the gate stricter (see [`group_fused_min_kv_bytes`]), so an unreadable cache
/// falls back to exactly the calibrated constant and nothing changes.
const DEFAULT_LAST_LEVEL_CACHE_BYTES: usize = 0;

/// Per-head threshold measured directly on the calibration host, and the floor
/// the topology term may never go below.
///
/// The topology term can raise this but never lower it, so no host can end up
/// admitting a working set that the calibration host measured as a loss.
const GROUP_FUSED_CALIBRATED_MIN_KV_BYTES: usize = 8 << 20;

/// Bytes of the largest cache level shared by this thread's CPU, read once from
/// `/sys/devices/system/cpu/cpu0/cache/index*`.
///
/// std-only on purpose: `onnx-runtime-cpuinfo` is a cmake+bindgen FFI crate that
/// nothing in the EP depends on, and pulling it in for one integer would put a
/// native build step on the CPU EP's critical path.
fn last_level_cache_bytes() -> usize {
    static BYTES: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *BYTES.get_or_init(|| probe_last_level_cache_bytes().unwrap_or(DEFAULT_LAST_LEVEL_CACHE_BYTES))
}

/// Largest `size` among the cache levels sysfs reports for cpu0. Returns `None`
/// on any platform or container where the directory is absent or unparseable.
fn probe_last_level_cache_bytes() -> Option<usize> {
    let mut best = None;
    for entry in std::fs::read_dir("/sys/devices/system/cpu/cpu0/cache").ok()? {
        let path = entry.ok()?.path().join("size");
        let Ok(raw) = std::fs::read_to_string(&path) else {
            continue;
        };
        if let Some(bytes) = parse_cache_size(raw.trim()) {
            best = Some(best.map_or(bytes, |current: usize| current.max(bytes)));
        }
    }
    best
}

/// sysfs writes cache sizes as a decimal with a `K`/`M`/`G` suffix, e.g. `32768K`.
fn parse_cache_size(raw: &str) -> Option<usize> {
    let (digits, scale) = match raw.as_bytes().last()? {
        b'K' | b'k' => (&raw[..raw.len() - 1], 1 << 10),
        b'M' | b'm' => (&raw[..raw.len() - 1], 1 << 20),
        b'G' | b'g' => (&raw[..raw.len() - 1], 1 << 30),
        b'0'..=b'9' => (raw, 1),
        _ => return None,
    };
    digits.parse::<usize>().ok()?.checked_mul(scale)
}

/// Test/calibration override for [`group_fused_min_kv_bytes`], in bytes.
/// `ONNX_GENAI_GQA_GROUP_FUSED_MIN_KV_BYTES=0` forces the traffic gate open,
/// which is how the crossover below was measured on a new host.
fn group_fused_min_kv_bytes_override() -> Option<usize> {
    static OVERRIDE: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    *OVERRIDE.get_or_init(|| {
        std::env::var("ONNX_GENAI_GQA_GROUP_FUSED_MIN_KV_BYTES")
            .ok()?
            .trim()
            .parse::<usize>()
            .ok()
    })
}

/// Attended KV bytes per KV head (K and V, at the f32 width SDPA reads) below
/// which KV-group fusion is not worth taking.
///
/// ```text
/// max(GROUP_FUSED_CALIBRATED_MIN_KV_BYTES, last_level_cache_bytes / fused_tasks)
/// ```
///
/// Re-reading a KV head `group` times is only expensive once those re-reads
/// miss the last-level cache. While the *aggregate* concurrently attended KV
/// (`bytes_per_head * fused_tasks`) still fits LLC, the repeat traffic is served
/// at cache bandwidth, the fusion's saving is worth little, and its extra
/// score-buffer traffic makes it a net loss. That is the topology term.
///
/// It is a `max`, not a replacement, and that asymmetry is deliberate. The
/// 8 MiB figure is measured; the topology term is a *model*. Taking the larger
/// of the two means the topology term can only ever make the gate **stricter**
/// than the calibration, never looser, so no host can be talked into a working
/// set the calibration measured as a loss. Concretely:
///
/// - On the calibration host (32 MiB LLC shared by 16 CPUs, 8 fused tasks for a
///   batch-1 8-KV-head decode) the topology term is 4 MiB, the `max` keeps
///   8 MiB, and behaviour is unchanged.
/// - On a 256 MiB-LLC part the topology term is 32 MiB, so the gate stays shut
///   through the 8-16 MiB range where the KV still fits that cache. This is the
///   large-L3 failure mode the fixed constant could not express, and it is the
///   whole reason for the change.
///
/// *Kernel sweep, 8-worker pool, `head_size = 128`* (ratio = fused / per-head,
/// >1 is a fusion win) — the origin of the 8 MiB figure:
///
/// | attended KV per head | group 2 | group 3 | group 4 | group 6 | group 8 |
/// |---|---|---|---|---|---|
/// | 1 MiB (kv 1024)  | 1.05x | 0.94x | 1.12x | 0.83x | 0.81x |
/// | 2 MiB (kv 2048)  | 1.00x | 1.18x | 1.11x | 0.81x | 0.83x |
/// | 4 MiB (kv 4096)  | 0.96x | 1.18x | 0.89x | 0.83x | 0.82x |
/// | 8 MiB (kv 8192)  | 1.32x | 1.44x | 1.41x | 1.11x | 1.00x |
/// | 16 MiB (kv 16384)| 1.49x | 1.68x | 2.05x | 1.61x | 1.56x |
///
/// A model-level A/B against a real ORT CPU session (llama-3-8B head geometry,
/// batch 1, the fused path forced on vs off in the same binary, interleaved,
/// p50 of 3 trials x 8 runs) confirms the *reachable* range is a win and that
/// the unreachable range is measurement noise:
///
/// | attended KV/head | t=1 | t=4 | t=8 | t=16 | t=32 |
/// |---|---|---|---|---|---|
/// | 8 MiB  | 0.93 | 0.93 | 0.91 | 1.01 | 0.99 |
/// | 16 MiB | 0.94 | 0.84 | 0.85 | 1.02 | 0.99 |
/// | 32 MiB | 0.85 | 0.78 | 0.78 | 1.00 | 1.01 |
///
/// (fused/unfused wall clock, lower is better. t=16 and t=32 are *controls*:
/// `group_fusion_saturates_pool` closes the gate there for 8 fused tasks, so
/// those columns must read 1.00 and do, to within 2%.)
///
/// Deliberately **not** claimed: that the topology term identifies the true
/// crossover. A follow-up sweep with the traffic gate forced fully open put the
/// 4 MiB per-head point at 0.91-1.19 across thread counts, i.e. inside a
/// contention noise floor of +-9% measured from the no-op controls. 4 MiB is
/// what `llc / fused_tasks` alone would admit on this host, and the measurement
/// does not support admitting it - which is exactly why the `max` keeps 8 MiB.
///
/// `ONNX_GENAI_GQA_GROUP_FUSED_MIN_KV_BYTES` overrides the whole computation,
/// including the floor, for recalibration on an unfamiliar host.
fn group_fused_min_kv_bytes(fused_tasks: usize) -> usize {
    #[cfg(test)]
    {
        let pinned = GROUP_FUSED_MIN_KV_BYTES_PIN.load(std::sync::atomic::Ordering::Relaxed);
        if pinned != 0 {
            return pinned;
        }
    }
    if let Some(bytes) = group_fused_min_kv_bytes_override() {
        return bytes;
    }
    (last_level_cache_bytes() / fused_tasks.max(1)).max(GROUP_FUSED_CALIBRATED_MIN_KV_BYTES)
}

/// KV tokens a decode step actually streams per head.
///
/// `local_window_size` caps the attended window independently of how long the
/// cache is, so the traffic gate must see this rather than
/// `total_sequence_length`: a sliding-window model at long context reads a
/// short window out of a long cache and belongs on the per-head path.
fn fused_attended_window(total_sequence_length: usize, local_window_size: i64) -> usize {
    if local_window_size > 0 {
        total_sequence_length.min(local_window_size as usize)
    } else {
        total_sequence_length
    }
}

/// Whether the attended window is long enough for the fusion's traffic saving
/// to beat its overhead. See [`group_fused_min_kv_bytes`].
fn group_fusion_pays_for_traffic(
    window: usize,
    k_dim: usize,
    v_dim: usize,
    fused_tasks: usize,
) -> bool {
    window
        .saturating_mul(k_dim.saturating_add(v_dim))
        .saturating_mul(size_of::<f32>())
        >= group_fused_min_kv_bytes(fused_tasks)
}

/// Contiguous `[start, end)` bounds of chunk `chunk` when the KV window
/// `[lo, hi)` is split into `split_count` even contiguous pieces (the earlier
/// chunks absorb the remainder). A `split_count` larger than the window length
/// leaves the trailing chunks empty (`start == end`), which
/// [`sdpa_decode_partial`] and [`combine_decode_partials`] both handle.
fn split_chunk_bounds(lo: usize, hi: usize, split_count: usize, chunk: usize) -> (usize, usize) {
    let length = hi - lo;
    let base = length / split_count;
    let remainder = length % split_count;
    let start = lo + chunk * base + chunk.min(remainder);
    let end = start + base + usize::from(chunk < remainder);
    (start, end)
}

/// Raw, `Sync` view over the flash-decoding partial scratch. Each KV-chunk task
/// writes only its own `task_index` slot, so the shared `*mut` never aliases.
struct SplitPartialScratch {
    partials: *mut DecodePartial,
    outputs: *mut f64,
}

// SAFETY: the scheduler runs each task index exactly once, and every task writes
// only `partials[task_index]` and `outputs[task_index * v_head_size ..]`, so no
// two workers ever touch the same element.
unsafe impl Sync for SplitPartialScratch {}

pub struct GroupQueryAttentionKernel {
    num_heads: usize,
    kv_num_heads: usize,
    scale: Option<f32>,
    do_rotary: bool,
    rotary_interleaved: bool,
    local_window_size: i64,
    softcap: f32,
}

pub struct GroupQueryAttentionFactory;

impl KernelFactory for GroupQueryAttentionFactory {
    fn create(&self, node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let required_heads = |name: &str| -> Result<usize> {
            let value = node.attr(name).and_then(|a| a.as_int()).ok_or_else(|| {
                EpError::KernelFailed(format!(
                    "GroupQueryAttention: missing required `{name}` attribute"
                ))
            })?;
            usize::try_from(value)
                .ok()
                .filter(|&v| v > 0)
                .ok_or_else(|| {
                    EpError::KernelFailed(format!("GroupQueryAttention: `{name}` must be > 0"))
                })
        };
        let num_heads = required_heads("num_heads")?;
        let kv_num_heads = required_heads("kv_num_heads")?;
        if !num_heads.is_multiple_of(kv_num_heads) {
            return Err(EpError::KernelFailed(format!(
                "GroupQueryAttention: num_heads {num_heads} must be a multiple of kv_num_heads {kv_num_heads}"
            )));
        }

        for name in ["k_quant_type", "v_quant_type"] {
            if let Some(value) = node.attr(name)
                && value.as_str() != Some("NONE")
            {
                return Err(EpError::KernelFailed(format!(
                    "GroupQueryAttention: `{name}` other than NONE is not yet supported by the f32 CPU kernel"
                )));
            }
        }
        if node
            .attr("kv_cache_bit_width")
            .and_then(|a| a.as_int())
            .unwrap_or(0)
            != 0
        {
            return Err(EpError::KernelFailed(
                "GroupQueryAttention: quantized KV cache is not yet supported".into(),
            ));
        }
        if node.attr("qk_output").and_then(|a| a.as_int()).unwrap_or(0) != 0 {
            return Err(EpError::KernelFailed(
                "GroupQueryAttention: qk_output is not yet supported".into(),
            ));
        }
        if node
            .attr("smooth_softmax")
            .and_then(|a| a.as_int())
            .unwrap_or(0)
            != 0
        {
            return Err(EpError::KernelFailed(
                "GroupQueryAttention: smooth_softmax is not yet supported".into(),
            ));
        }

        let softcap = node
            .attr("softcap")
            .and_then(|a| a.as_float())
            .unwrap_or(0.0);
        if softcap < 0.0 {
            return Err(EpError::KernelFailed(
                "GroupQueryAttention: softcap must be non-negative".into(),
            ));
        }

        Ok(Box::new(GroupQueryAttentionKernel {
            num_heads,
            kv_num_heads,
            scale: node.attr("scale").and_then(|a| a.as_float()),
            do_rotary: node.attr("do_rotary").and_then(|a| a.as_int()).unwrap_or(0) != 0,
            rotary_interleaved: node
                .attr("rotary_interleaved")
                .and_then(|a| a.as_int())
                .unwrap_or(0)
                != 0,
            local_window_size: node
                .attr("local_window_size")
                .and_then(|a| a.as_int())
                .unwrap_or(-1),
            softcap,
        }))
    }
}

struct Bhsd {
    data: Vec<f32>,
    batch: usize,
    heads: usize,
    seq: usize,
    dim: usize,
}

impl Bhsd {
    fn from_bsh(view: &TensorView, heads: usize, name: &str) -> Result<Self> {
        if view.shape.len() != 3 {
            return Err(EpError::KernelFailed(format!(
                "GroupQueryAttention: unpacked {name} must be rank 3 [B,S,H*D], got {:?}",
                view.shape
            )));
        }
        let (batch, seq, hidden) = (view.shape[0], view.shape[1], view.shape[2]);
        if !hidden.is_multiple_of(heads) {
            return Err(EpError::KernelFailed(format!(
                "GroupQueryAttention: {name} hidden size {hidden} is not divisible by {heads} heads"
            )));
        }
        let dim = hidden / heads;
        let src = to_dense_f32_widen("GroupQueryAttention", view)?;
        let mut data = vec![0.0; src.len()];
        for b in 0..batch {
            for s in 0..seq {
                for h in 0..heads {
                    for d in 0..dim {
                        data[((b * heads + h) * seq + s) * dim + d] =
                            src[((b * seq + s) * heads + h) * dim + d];
                    }
                }
            }
        }
        Ok(Self {
            data,
            batch,
            heads,
            seq,
            dim,
        })
    }

    fn from_packed_qkv(
        view: &TensorView,
        num_heads: usize,
        kv_num_heads: usize,
    ) -> Result<(Self, Self, Self)> {
        if view.shape.len() != 3 {
            return Err(EpError::KernelFailed(format!(
                "GroupQueryAttention: packed query must be rank 3 [B,S,(N+2*Nk)*D], got {:?}",
                view.shape
            )));
        }
        let (batch, seq, hidden) = (view.shape[0], view.shape[1], view.shape[2]);
        let packed_heads = num_heads + 2 * kv_num_heads;
        if !hidden.is_multiple_of(packed_heads) {
            return Err(EpError::KernelFailed(format!(
                "GroupQueryAttention: packed QKV hidden size {hidden} is not divisible by num_heads + 2*kv_num_heads ({packed_heads})"
            )));
        }
        let dim = hidden / packed_heads;
        if dim == 0 {
            return Err(EpError::KernelFailed(
                "GroupQueryAttention: packed QKV head size must be positive".into(),
            ));
        }

        let src = to_dense_f32_widen("GroupQueryAttention", view)?;
        let q_hidden = num_heads * dim;
        let kv_hidden = kv_num_heads * dim;
        let mut q = vec![0.0; batch * num_heads * seq * dim];
        let mut k = vec![0.0; batch * kv_num_heads * seq * dim];
        let mut v = vec![0.0; k.len()];
        for b in 0..batch {
            for s in 0..seq {
                let src_base = (b * seq + s) * hidden;
                for h in 0..num_heads {
                    for d in 0..dim {
                        q[((b * num_heads + h) * seq + s) * dim + d] = src[src_base + h * dim + d];
                    }
                }
                for h in 0..kv_num_heads {
                    for d in 0..dim {
                        let dst = ((b * kv_num_heads + h) * seq + s) * dim + d;
                        k[dst] = src[src_base + q_hidden + h * dim + d];
                        v[dst] = src[src_base + q_hidden + kv_hidden + h * dim + d];
                    }
                }
            }
        }

        Ok((
            Self {
                data: q,
                batch,
                heads: num_heads,
                seq,
                dim,
            },
            Self {
                data: k,
                batch,
                heads: kv_num_heads,
                seq,
                dim,
            },
            Self {
                data: v,
                batch,
                heads: kv_num_heads,
                seq,
                dim,
            },
        ))
    }
}

/// Borrowed reference to a BNSH KV cache input that widens **incrementally**
/// into the caller's `present` buffer.
///
/// The decode hot path used to widen the entire growing past cache (`f16`→`f32`)
/// into an owned buffer and then copy it again into `present_k`/`present_v` every
/// token — an `O(sequence_length)` widen plus an `O(sequence_length)` copy per
/// step. Profiling attributed ~40% of GroupQueryAttention to that pair. Instead,
/// this keeps only the raw view (for the common contiguous `f16`/`f32` cache) and
/// widens each per-head run *directly into* the destination `present` slice via
/// [`widen_run`](PastCache::widen_run), eliminating the intermediate materialize
/// and the copy. Exotic layouts (strided, `bf16`, `f64`) fall back to a one-time
/// dense widen, so generality is preserved.
struct PastCache<'a> {
    src: PastSrc<'a>,
    seq: usize,
    dim: usize,
    batch: usize,
}

/// Backing storage strategy for a [`PastCache`] head-run widen.
enum PastSrc<'a> {
    /// Contiguous `f32` cache: the run is copied verbatim.
    F32(&'a [f32]),
    /// Contiguous `f16` cache (raw `u16` bits): the run is F16C/scalar widened.
    F16(&'a [u16]),
    /// Non-contiguous or non-`f16`/`f32` cache widened once up front.
    Dense(Vec<f32>),
}

impl<'a> PastCache<'a> {
    fn from_cache(view: &'a TensorView<'a>, heads: usize, name: &str) -> Result<Self> {
        if view.shape.len() != 4 || view.shape[1] != heads {
            return Err(EpError::KernelFailed(format!(
                "GroupQueryAttention: {name} must use BNSH layout [B,{heads},S,D], got {:?}",
                view.shape
            )));
        }
        view.validate()?;
        let len = view.numel();
        let src = if len == 0 {
            PastSrc::Dense(Vec::new())
        } else if view.dtype == onnx_runtime_ir::DataType::Float32 && view.is_contiguous() {
            // SAFETY: a validated contiguous Float32 view addresses exactly `len`
            // initialized f32 elements from `data_ptr`, kept alive for `'a`.
            PastSrc::F32(unsafe { std::slice::from_raw_parts(view.data_ptr::<f32>(), len) })
        } else if view.dtype == onnx_runtime_ir::DataType::Float16 && view.is_contiguous() {
            // SAFETY: a validated contiguous Float16 view addresses exactly `len`
            // 2-byte elements; `half::f16` is `repr(transparent)` over `u16`.
            PastSrc::F16(unsafe { std::slice::from_raw_parts(view.data_ptr::<u16>(), len) })
        } else {
            PastSrc::Dense(to_dense_f32_widen("GroupQueryAttention", view)?.into_owned())
        };
        Ok(Self {
            src,
            seq: view.shape[2],
            dim: view.shape[3],
            batch: view.shape[0],
        })
    }

    /// Widen the contiguous `[start, start + dst.len())` element run of this
    /// cache (row-major BNSH element offsets) into `dst`.
    #[inline]
    fn widen_run(&self, start: usize, dst: &mut [f32]) {
        let len = dst.len();
        match &self.src {
            PastSrc::F32(s) => dst.copy_from_slice(&s[start..start + len]),
            PastSrc::F16(s) => crate::dtype::widen_f16_slice_into(&s[start..start + len], dst),
            PastSrc::Dense(s) => dst.copy_from_slice(&s[start..start + len]),
        }
    }
}

fn scalar_i64(view: &TensorView, name: &str) -> Result<usize> {
    let values = to_dense_i64(view)?;
    if values.len() != 1 || values[0] < 0 {
        return Err(EpError::KernelFailed(format!(
            "GroupQueryAttention: {name} must be one non-negative int32 scalar"
        )));
    }
    Ok(values[0] as usize)
}

fn rotate(
    tensor: &mut Bhsd,
    cos: &[f32],
    sin: &[f32],
    cache_rows: usize,
    rotary_dim: usize,
    positions: &[usize],
    interleaved: bool,
) -> Result<()> {
    if rotary_dim == 0 || rotary_dim > tensor.dim || !rotary_dim.is_multiple_of(2) {
        return Err(EpError::KernelFailed(format!(
            "GroupQueryAttention: rotary dimension {rotary_dim} must be positive, even, and no larger than head_size {}",
            tensor.dim
        )));
    }
    let half = rotary_dim / 2;
    if cos.len() != cache_rows * half || sin.len() != cache_rows * half {
        return Err(EpError::KernelFailed(format!(
            "GroupQueryAttention: cos_cache/sin_cache must have shape [max_sequence_length,{half}]"
        )));
    }
    for b in 0..tensor.batch {
        for s in 0..tensor.seq {
            let pos = positions[b * tensor.seq + s];
            if pos >= cache_rows {
                return Err(EpError::KernelFailed(format!(
                    "GroupQueryAttention: rotary position {pos} exceeds cache rows {cache_rows}"
                )));
            }
            for h in 0..tensor.heads {
                for k in 0..half {
                    let (d0, d1) = if interleaved {
                        (2 * k, 2 * k + 1)
                    } else {
                        (k, k + half)
                    };
                    let i0 = ((b * tensor.heads + h) * tensor.seq + s) * tensor.dim + d0;
                    let i1 = ((b * tensor.heads + h) * tensor.seq + s) * tensor.dim + d1;
                    let (x0, x1) = (tensor.data[i0], tensor.data[i1]);
                    let (c, sn) = (cos[pos * half + k], sin[pos * half + k]);
                    tensor.data[i0] = c * x0 - sn * x1;
                    tensor.data[i1] = sn * x0 + c * x1;
                }
            }
        }
    }
    Ok(())
}

/// Widen only the first `rows` rows (`rows * half` contiguous elements) of a
/// rank-2 `[cache_rows, half]` rotary `cos`/`sin` cache into `f32`.
///
/// The rotary caches ship the model's *entire* position table (commonly
/// `max_position_embeddings` = tens of thousands of rows). Decode/prefill only
/// index positions up to the live context length, so widening the whole cache
/// (`f16`→`f32`) on every `GroupQueryAttention` call was an `O(cache_rows)`
/// per-token cost dwarfing the attention itself; this bounds it to the rows
/// actually addressed. Contiguous `f16`/`f32` caches take the fast path; exotic
/// layouts fall back to a full widen + truncate (correct, rarely hit).
fn widen_rotary_prefix(op: &str, view: &TensorView, rows: usize, half: usize) -> Result<Vec<f32>> {
    view.validate()?;
    let count = rows * half;
    if count == 0 {
        return Ok(Vec::new());
    }
    if view.dtype == onnx_runtime_ir::DataType::Float16 && view.is_contiguous() {
        // SAFETY: a validated contiguous Float16 view addresses `numel() >= count`
        // 2-byte elements; `half::f16` is `repr(transparent)` over `u16`.
        let src = unsafe { std::slice::from_raw_parts(view.data_ptr::<u16>(), count) };
        let mut dst = vec![0.0f32; count];
        crate::dtype::widen_f16_slice_into(src, &mut dst);
        return Ok(dst);
    }
    if view.dtype == onnx_runtime_ir::DataType::BFloat16 && view.is_contiguous() {
        // SAFETY: a validated contiguous BFloat16 view addresses `numel() >= count`
        // 2-byte elements; BF16 widening is the exact high-half f32 bit shift.
        let src = unsafe { std::slice::from_raw_parts(view.data_ptr::<u16>(), count) };
        let mut dst = vec![0.0f32; count];
        widen_bf16_slice_into(src, &mut dst);
        return Ok(dst);
    }
    if view.dtype == onnx_runtime_ir::DataType::Float32 && view.is_contiguous() {
        // SAFETY: a validated contiguous Float32 view addresses `numel() >= count`
        // initialized f32 elements.
        let src = unsafe { std::slice::from_raw_parts(view.data_ptr::<f32>(), count) };
        return Ok(src.to_vec());
    }
    let full = to_dense_f32_widen(op, view)?;
    Ok(full[..count.min(full.len())].to_vec())
}

fn write_decode_output(out: &mut TensorMut, data: &[f32]) -> Result<()> {
    if out.dtype != onnx_runtime_ir::DataType::Float32 || !out.is_contiguous() {
        return write_dense_f32_narrow("GroupQueryAttention", out, data);
    }
    out.validate()?;
    if out.numel() != data.len() {
        return Err(EpError::KernelFailed(format!(
            "GroupQueryAttention: output element count {} does not match produced {}",
            out.numel(),
            data.len()
        )));
    }
    if data.is_empty() {
        return Ok(());
    }
    // SAFETY: validation plus the contiguous Float32 layout prove the output
    // spans exactly data.len() writable f32 elements.
    let dst = unsafe { std::slice::from_raw_parts_mut(out.data_ptr_mut::<f32>(), data.len()) };
    dst.copy_from_slice(data);
    Ok(())
}

// ── temporary within-GQA phase profiling (gated by ONNX_GENAI_PROFILE_GQA) ────
#[cfg(feature = "gqa_phase_profile")]
mod phase_prof {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::Instant;

    pub static WIDEN_NS: AtomicU64 = AtomicU64::new(0);
    pub static PRESENT_NS: AtomicU64 = AtomicU64::new(0);
    pub static ATTN_NS: AtomicU64 = AtomicU64::new(0);
    pub static OUT_NS: AtomicU64 = AtomicU64::new(0);
    pub static TOTAL_NS: AtomicU64 = AtomicU64::new(0);
    pub static CALLS: AtomicU64 = AtomicU64::new(0);

    pub fn enabled() -> bool {
        static E: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *E.get_or_init(|| std::env::var("ONNX_GENAI_PROFILE_GQA").is_ok_and(|v| v == "1"))
    }

    pub struct Phase(Option<(Instant, &'static AtomicU64)>);
    impl Phase {
        pub fn start(acc: &'static AtomicU64) -> Self {
            if enabled() {
                Phase(Some((Instant::now(), acc)))
            } else {
                Phase(None)
            }
        }
    }
    impl Drop for Phase {
        fn drop(&mut self) {
            if let Some((t, acc)) = self.0 {
                acc.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
            }
        }
    }

    pub fn tick() {
        if !enabled() {
            return;
        }
        let calls = CALLS.fetch_add(1, Ordering::Relaxed) + 1;
        if calls.is_multiple_of(240) {
            let w = WIDEN_NS.load(Ordering::Relaxed) as f64 / 1e6;
            let p = PRESENT_NS.load(Ordering::Relaxed) as f64 / 1e6;
            let a = ATTN_NS.load(Ordering::Relaxed) as f64 / 1e6;
            let o = OUT_NS.load(Ordering::Relaxed) as f64 / 1e6;
            let total = TOTAL_NS.load(Ordering::Relaxed) as f64 / 1e6;
            let tot = w + p + a + o;
            let other = total - tot;
            eprintln!(
                "[gqa-phase] calls={calls} exec_total={total:.1}ms widen={w:.1}ms({wp:.1}%) present={p:.1}ms({pp:.1}%) attn={a:.1}ms({ap:.1}%) out={o:.1}ms({op:.1}%) other={other:.1}ms({ot:.1}%)",
                wp = 100.0 * w / total,
                pp = 100.0 * p / total,
                ap = 100.0 * a / total,
                op = 100.0 * o / total,
                ot = 100.0 * other / total,
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────

impl GroupQueryAttentionKernel {
    /// Decide whether the present KV outputs alias the past KV inputs onto one
    /// persistent, full-capacity buffer (a present==past device binding — the
    /// CPU analogue of the CUDA in-place KV cache). When they do, the append-only
    /// fast path can write the current step's K/V rows directly into the buffer
    /// instead of widening/copying the whole growing past and round-tripping the
    /// present output.
    ///
    /// The gate is purely structural so it can never fire for an ordinary run:
    /// the present output pointer must be byte-identical to the past input
    /// pointer, both caches must be contiguous f32 at the full physical capacity
    /// (`present_len` elements), and that capacity must already cover the new
    /// `total` (`present_sequence_length == cache.seq`, which the caller derives
    /// as `max(cache.seq, total)`). Key and value must be distinct buffers.
    fn detect_inplace_kv(
        &self,
        inputs: &[TensorView],
        outputs: &[TensorMut],
        present_sequence_length: usize,
        present_len: usize,
        past_key: Option<&PastCache>,
    ) -> bool {
        use onnx_runtime_ir::DataType::Float32;
        let Some(cache) = past_key else {
            return false;
        };
        // Capacity must already hold the new total; `present_sequence_length`
        // equals the physical cache extent exactly in that case.
        if outputs.len() < 3 || present_sequence_length != cache.seq {
            return false;
        }
        if inputs.len() < 5 {
            return false;
        }
        // Restrict to contiguous f32 caches at the exact physical capacity.
        for view in [&inputs[3], &inputs[4]] {
            if view.is_absent()
                || view.dtype != Float32
                || !view.is_contiguous()
                || view.numel() != present_len
            {
                return false;
            }
        }
        for out in [&outputs[1], &outputs[2]] {
            if out.dtype != Float32 || !out.is_contiguous() || out.numel() != present_len {
                return false;
            }
        }
        // Structural aliasing: each present output origin must be the exact same
        // address as its past input origin, computed identically to `data_ptr`.
        let in_ptr =
            |view: &TensorView| (view.data.0 as *const u8).wrapping_add(view.byte_offset) as usize;
        let out_ptr =
            |out: &TensorMut| (out.data.0 as *const u8).wrapping_add(out.byte_offset) as usize;
        let pk_in = in_ptr(&inputs[3]);
        let pv_in = in_ptr(&inputs[4]);
        let pk_out = out_ptr(&outputs[1]);
        let pv_out = out_ptr(&outputs[2]);
        pk_in != 0 && pv_in != 0 && pk_in == pk_out && pv_in == pv_out && pk_in != pv_in
    }
}

impl Kernel for GroupQueryAttentionKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        #[cfg(feature = "gqa_phase_profile")]
        let _total_phase = phase_prof::Phase::start(&phase_prof::TOTAL_NS);
        check_arity("GroupQueryAttention", inputs, outputs, 7, 14, 1)?;
        if outputs.len() > 3 {
            return Err(EpError::KernelFailed(
                "GroupQueryAttention: output_qk is not yet supported".into(),
            ));
        }
        let packed_qkv = inputs[1].is_absent() && inputs[2].is_absent();
        if inputs[1].is_absent() != inputs[2].is_absent() {
            return Err(EpError::KernelFailed(
                "GroupQueryAttention: key and value must both be present for unpacked Q/K/V or both absent for packed QKV".into(),
            ));
        }
        if inputs.get(10).is_some_and(|v| !v.is_absent()) {
            return Err(EpError::KernelFailed(
                "GroupQueryAttention: attention_bias is not yet supported".into(),
            ));
        }
        if inputs.get(11).is_some_and(|v| !v.is_absent()) {
            return Err(EpError::KernelFailed(
                "GroupQueryAttention: head_sink is not yet supported".into(),
            ));
        }
        if inputs.get(12).is_some_and(|v| !v.is_absent())
            || inputs.get(13).is_some_and(|v| !v.is_absent())
        {
            return Err(EpError::KernelFailed(
                "GroupQueryAttention: quantized-cache k_scale/v_scale inputs are not yet supported"
                    .into(),
            ));
        }
        if self.local_window_size == 0 || self.local_window_size < -1 {
            return Err(EpError::KernelFailed(
                "GroupQueryAttention: local_window_size must be -1 or a positive integer".into(),
            ));
        }

        let (mut q, mut k, v) = if packed_qkv {
            Bhsd::from_packed_qkv(&inputs[0], self.num_heads, self.kv_num_heads)?
        } else {
            (
                Bhsd::from_bsh(&inputs[0], self.num_heads, "query")?,
                Bhsd::from_bsh(&inputs[1], self.kv_num_heads, "key")?,
                Bhsd::from_bsh(&inputs[2], self.kv_num_heads, "value")?,
            )
        };
        if q.batch != k.batch
            || q.batch != v.batch
            || k.seq != v.seq
            || k.dim != q.dim
            || v.dim != q.dim
        {
            return Err(EpError::KernelFailed(
                "GroupQueryAttention: incompatible query/key/value batch, sequence, or head dimensions".into(),
            ));
        }

        let has_past_key = !inputs[3].is_absent();
        let has_past_value = !inputs[4].is_absent();
        if has_past_key != has_past_value {
            return Err(EpError::KernelFailed(
                "GroupQueryAttention: past_key and past_value must be provided together".into(),
            ));
        }
        #[cfg(feature = "gqa_phase_profile")]
        let _widen_phase = phase_prof::Phase::start(&phase_prof::WIDEN_NS);
        let past_key = has_past_key
            .then(|| PastCache::from_cache(&inputs[3], self.kv_num_heads, "past_key"))
            .transpose()?;
        let past_value = has_past_value
            .then(|| PastCache::from_cache(&inputs[4], self.kv_num_heads, "past_value"))
            .transpose()?;
        #[cfg(feature = "gqa_phase_profile")]
        drop(_widen_phase);
        if let (Some(pk), Some(pv)) = (&past_key, &past_value)
            && (pk.batch != q.batch
                || pv.batch != q.batch
                || pk.seq != pv.seq
                || pk.dim != q.dim
                || pv.dim != q.dim)
        {
            return Err(EpError::KernelFailed(
                "GroupQueryAttention: past_key/past_value dimensions are incompatible with Q/K/V"
                    .into(),
            ));
        }

        let seqlens = normalized_sequence_lengths(&inputs[5], q.batch)?;
        if seqlens.iter().any(|&x| x < 0) {
            return Err(EpError::KernelFailed(
                "GroupQueryAttention: seqlens_k must be non-negative int32 [batch_size]".into(),
            ));
        }

        let total_sequence_length = scalar_i64(&inputs[6], "total_sequence_length")?;
        let totals: Vec<usize> = seqlens.iter().map(|&x| x as usize + 1).collect();
        if totals.iter().copied().max().unwrap_or(0) != total_sequence_length {
            return Err(EpError::KernelFailed(format!(
                "GroupQueryAttention: total_sequence_length {total_sequence_length} must equal max(seqlens_k + 1)"
            )));
        }
        let mut past_lengths = Vec::with_capacity(q.batch);
        let mut query_starts = Vec::with_capacity(q.batch);
        for &total in &totals {
            let past = total.checked_sub(k.seq).ok_or_else(|| {
                EpError::KernelFailed(
                    "GroupQueryAttention: seqlens_k + 1 is shorter than current key sequence"
                        .into(),
                )
            })?;
            if past > past_key.as_ref().map_or(0, |cache| cache.seq) {
                return Err(EpError::KernelFailed(
                    "GroupQueryAttention: effective past length exceeds past cache extent".into(),
                ));
            }
            past_lengths.push(past);
            query_starts.push(total.checked_sub(q.seq).ok_or_else(|| {
                EpError::KernelFailed(
                    "GroupQueryAttention: total sequence is shorter than query sequence".into(),
                )
            })?);
        }

        if self.do_rotary {
            let cos_view = inputs.get(7).filter(|v| !v.is_absent()).ok_or_else(|| {
                EpError::KernelFailed("GroupQueryAttention: do_rotary=1 requires cos_cache".into())
            })?;
            let sin_view = inputs.get(8).filter(|v| !v.is_absent()).ok_or_else(|| {
                EpError::KernelFailed("GroupQueryAttention: do_rotary=1 requires sin_cache".into())
            })?;
            if cos_view.shape.len() != 2 || sin_view.shape != cos_view.shape {
                return Err(EpError::KernelFailed(
                    "GroupQueryAttention: cos_cache and sin_cache must have equal rank-2 shapes"
                        .into(),
                ));
            }
            let rotary_half = cos_view.shape[1];
            let rotary_dim = rotary_half.checked_mul(2).ok_or_else(|| {
                EpError::KernelFailed(
                    "GroupQueryAttention: rotary cache dimension is too large".into(),
                )
            })?;
            if rotary_half == 0 || rotary_dim > q.dim {
                return Err(EpError::KernelFailed(format!(
                    "GroupQueryAttention: rotary cache dimension {rotary_half} implies rotary dimension {rotary_dim}, which must be positive and no larger than head_size {}",
                    q.dim
                )));
            }
            let explicit_position_ids = inputs.get(9).filter(|v| !v.is_absent());
            let query_positions = if let Some(position_ids) = explicit_position_ids {
                let ids = to_dense_i64(position_ids)?;
                if position_ids.shape != [q.batch, q.seq] || ids.iter().any(|&x| x < 0) {
                    return Err(EpError::KernelFailed(
                        "GroupQueryAttention: position_ids must be non-negative int64 [batch_size, sequence_length]".into(),
                    ));
                }
                ids.into_iter().map(|x| x as usize).collect()
            } else {
                let mut ids = Vec::with_capacity(q.batch * q.seq);
                for &total in &totals {
                    let start = total.checked_sub(q.seq).ok_or_else(|| {
                        EpError::KernelFailed(
                            "GroupQueryAttention: total sequence is shorter than query sequence"
                                .into(),
                        )
                    })?;
                    ids.extend((0..q.seq).map(|s| start + s));
                }
                ids
            };
            let key_positions = if explicit_position_ids.is_some() && k.seq == q.seq {
                query_positions.clone()
            } else {
                let mut ids = Vec::with_capacity(k.batch * k.seq);
                for &total in &totals {
                    let start = total.checked_sub(k.seq).ok_or_else(|| {
                        EpError::KernelFailed(
                            "GroupQueryAttention: total sequence is shorter than key sequence"
                                .into(),
                        )
                    })?;
                    ids.extend((0..k.seq).map(|s| start + s));
                }
                ids
            };
            let cache_rows = cos_view.shape[0];
            // Only positions up to the live context length are indexed; widening
            // the whole (often 32k-row) cache every call was the dominant GQA
            // decode cost. Bound the widen to the addressed row prefix.
            let max_position = query_positions
                .iter()
                .chain(key_positions.iter())
                .copied()
                .max()
                .unwrap_or(0);
            if max_position >= cache_rows {
                return Err(EpError::KernelFailed(format!(
                    "GroupQueryAttention: rotary position {max_position} exceeds cache rows {cache_rows}"
                )));
            }
            let rows_needed = max_position + 1;
            let cos =
                widen_rotary_prefix("GroupQueryAttention", cos_view, rows_needed, rotary_half)?;
            let sin =
                widen_rotary_prefix("GroupQueryAttention", sin_view, rows_needed, rotary_half)?;
            rotate(
                &mut q,
                &cos,
                &sin,
                rows_needed,
                rotary_dim,
                &query_positions,
                self.rotary_interleaved,
            )?;
            rotate(
                &mut k,
                &cos,
                &sin,
                rows_needed,
                rotary_dim,
                &key_positions,
                self.rotary_interleaved,
            )?;
        }

        let cache_dim = q.dim;
        #[cfg(feature = "gqa_phase_profile")]
        let _present_phase = phase_prof::Phase::start(&phase_prof::PRESENT_NS);
        let present_sequence_length = past_key.as_ref().map_or(total_sequence_length, |cache| {
            cache.seq.max(total_sequence_length)
        });
        let present_len = q.batch * self.kv_num_heads * present_sequence_length * cache_dim;

        // ── In-place persistent KV fast path ────────────────────────────────
        // When the session has bound each `present` output onto its `past`
        // input (a present==past device binding — the CPU analogue of the CUDA
        // in-place KV cache), the past prefix is ALREADY resident in the output
        // buffer at full physical capacity. Detect that purely *structurally* —
        // the present output pointer aliases the past input pointer, both are
        // contiguous f32, and capacity already covers the new total — then
        // append only the current step's K/V rows in place and attend directly
        // over the buffer. This eliminates the O(capacity) past widen/copy and
        // the entire present round-trip the copy path pays every token. Any
        // non-aliased call (every ordinary model/run and test) is byte-identical
        // to before because it falls through to the copy path below.
        let in_place = self.detect_inplace_kv(
            inputs,
            outputs,
            present_sequence_length,
            present_len,
            past_key.as_ref(),
        );

        // Owned present storage backs only the copy path; the in-place path
        // writes straight into the aliased output buffer and never touches these.
        let owned_present_k: Vec<f32>;
        let owned_present_v: Vec<f32>;
        let (present_k, present_v): (&[f32], &[f32]) = if in_place {
            let pk_ptr = outputs[1].data_ptr_mut::<f32>();
            let pv_ptr = outputs[2].data_ptr_mut::<f32>();
            // Release the immutable past borrows before mutating the aliased
            // buffer: past_key/past_value view the SAME memory as pk_ptr/pv_ptr,
            // so no live `&[f32]` may coexist with the `&mut [f32]` below.
            drop(past_key);
            drop(past_value);
            // SAFETY: `detect_inplace_kv` proved outputs[1]/[2] are contiguous
            // f32 buffers of exactly `present_len` elements. They are the graph
            // outputs, exclusively owned by this kernel invocation. The past
            // prefix [0, past_len) per head is already resident from prior steps;
            // we write only the current [past_len, total) rows, all within each
            // head's [0, present_sequence_length) region, and never read
            // uninitialized capacity (attention is causal-bounded to `total`).
            let present_k = unsafe { std::slice::from_raw_parts_mut(pk_ptr, present_len) };
            let present_v = unsafe { std::slice::from_raw_parts_mut(pv_ptr, present_len) };
            for (b, &past_len) in past_lengths.iter().enumerate() {
                for h in 0..self.kv_num_heads {
                    let head = b * self.kv_num_heads + h;
                    let dst_base = head * present_sequence_length * cache_dim;
                    let cur = k.seq * cache_dim;
                    let dst_cur = dst_base + past_len * cache_dim;
                    let src_cur = head * k.seq * cache_dim;
                    present_k[dst_cur..dst_cur + cur]
                        .copy_from_slice(&k.data[src_cur..src_cur + cur]);
                    present_v[dst_cur..dst_cur + cur]
                        .copy_from_slice(&v.data[src_cur..src_cur + cur]);
                }
            }
            (&*present_k, &*present_v)
        } else {
            // A "tail" is any padding row beyond a batch's logical `total` that
            // is emitted into the present output but never attended; those rows
            // must be zero. In steady decode every batch's `total` exactly fills
            // `present_sequence_length`, so the per-(b,h) loop below overwrites
            // every element and pre-zeroing is pure waste — skip it in that case.
            let has_tail = totals.iter().any(|&t| t < present_sequence_length);
            let (mut present_k, mut present_v) = if has_tail {
                (vec![0.0f32; present_len], vec![0.0f32; present_len])
            } else {
                let mut present_k = Vec::<f32>::with_capacity(present_len);
                let mut present_v = Vec::<f32>::with_capacity(present_len);
                // SAFETY: `!has_tail` ⇒ every batch's `total == present_sequence_length`,
                // so for each `(b, h)` the loop below writes the past prefix
                // `[0, past_len)` and the current span `[past_len, total)` =
                // `[0, present_sequence_length)` rows, i.e. every element of both
                // buffers, before any read (attention and the output narrow both
                // run strictly after this loop). No uninitialized element is observed.
                unsafe {
                    present_k.set_len(present_len);
                    present_v.set_len(present_len);
                }
                (present_k, present_v)
            };
            for (b, &past_len) in past_lengths.iter().enumerate() {
                for h in 0..self.kv_num_heads {
                    let head = b * self.kv_num_heads + h;
                    let dst_base = head * present_sequence_length * cache_dim;
                    // `present_k`/`present_v` and the past caches are both
                    // BNSH-contiguous, so for a fixed (b, h) the `[s, d]` block is
                    // a single contiguous run in each: widen the whole past prefix
                    // directly into `present` (F16C for f16), skipping the separate
                    // owned widen + f32 copy the decode path used to pay every token.
                    if past_len > 0 {
                        let copy = past_len * cache_dim;
                        let pk = past_key.as_ref().unwrap();
                        let pv = past_value.as_ref().unwrap();
                        let src = head * pk.seq * cache_dim;
                        pk.widen_run(src, &mut present_k[dst_base..dst_base + copy]);
                        pv.widen_run(src, &mut present_v[dst_base..dst_base + copy]);
                    }
                    // Append the current token(s) directly after the past prefix;
                    // the current K/V blocks are contiguous in `[s, d]` as well.
                    let cur = k.seq * cache_dim;
                    let dst_cur = dst_base + past_len * cache_dim;
                    let src_cur = head * k.seq * cache_dim;
                    present_k[dst_cur..dst_cur + cur]
                        .copy_from_slice(&k.data[src_cur..src_cur + cur]);
                    present_v[dst_cur..dst_cur + cur]
                        .copy_from_slice(&v.data[src_cur..src_cur + cur]);
                }
            }
            owned_present_k = present_k;
            owned_present_v = present_v;
            (&owned_present_k[..], &owned_present_v[..])
        };

        let scale = self
            .scale
            .filter(|&scale| scale != 0.0)
            .unwrap_or_else(|| 1.0 / (cache_dim as f32).sqrt());
        #[cfg(feature = "gqa_phase_profile")]
        {
            drop(_present_phase);
        }
        #[cfg(feature = "gqa_phase_profile")]
        let _attn_phase = phase_prof::Phase::start(&phase_prof::ATTN_NS);
        let group = self.num_heads / self.kv_num_heads;
        let attention_rows = q.batch * q.seq * self.num_heads;
        let mut y_bhsd = vec![0.0; attention_rows * v.dim];
        let compute_row = |b: usize, qh: usize, qs: usize, output_row: &mut [f32]| {
            let kvh = qh / group;
            let causal_limit = query_starts[b] + qs;
            let local_start = if self.local_window_size > 0 {
                (causal_limit + 1).saturating_sub(self.local_window_size as usize)
            } else {
                0
            };
            let q_base = ((b * self.num_heads + qh) * q.seq + qs) * cache_dim;
            let q_row = &q.data[q_base..q_base + cache_dim];
            let kv_head_base = (b * self.kv_num_heads + kvh) * present_sequence_length;
            let k_head = &present_k
                [kv_head_base * cache_dim..(kv_head_base + present_sequence_length) * cache_dim];
            let v_head =
                &present_v[kv_head_base * v.dim..(kv_head_base + present_sequence_length) * v.dim];
            sdpa_decode_row(
                q_row,
                k_head,
                v_head,
                present_sequence_length,
                local_start,
                causal_limit + 1,
                scale,
                (self.softcap != 0.0).then_some(self.softcap),
                SoftmaxExp::F64Intermediate,
                output_row,
            );
        };
        let attention_work = attention_rows
            .saturating_mul(total_sequence_length)
            .saturating_mul(cache_dim);
        let worker_count = crate::kernels::matmul_nbits::active_decode_worker_count();
        // KV-group fusion (M=1 decode): every query head in a GQA group attends
        // the *same* KV head over the *same* window, so scoring them one row at
        // a time streams the KV cache `group` times per layer per token. At M=1
        // the window is a GEMV (~2 flops per loaded float), so that repeat
        // traffic — not the arithmetic — is the cost. `sdpa_decode_group` runs
        // the whole group in one pass over K and V, cutting KV bytes by `group`
        // while performing bit-identical arithmetic per head.
        //
        // The fusion costs parallelism: it collapses `batch * num_heads` tasks
        // into `batch * kv_num_heads`. Measured on an 8-worker decode pool,
        // starving the pool dominates the traffic saving by a wide margin — a
        // 2-KV-head geometry runs ~2.6x *slower* fused, a 4-KV-head one ~1.7x
        // slower. And while the attended KV still fits the last-level cache the
        // repeat reads are cheap, so the saving does not pay for the fused
        // path's score-buffer traffic. Both conditions must hold; see
        // `group_fusion_saturates_pool` and `group_fusion_pays_for_traffic`.
        //
        // Measured end to end on Qwen3-0.6B-int4 at ~11.8k context with
        // `ONNX_GENAI_CPU_DECODE_THREADS=8` (the only batch-1 configuration on a
        // 16-core host where both gates open): median per-pair ratio 1.31x over
        // 5 interleaved trials, token output identical in every run. Note this
        // does not beat the *default* 16-worker per-head config (129 ms/token) —
        // narrowing the pool to reach the gate costs more than the fusion
        // returns at batch 1. It is a win within a given decode width, not a new
        // best configuration, which is part of why the path is opt-in.
        let attended_window = fused_attended_window(total_sequence_length, self.local_window_size);
        let group_fused = q.seq == 1
            && group > 1
            && group_fusion_saturates_pool(q.batch * self.kv_num_heads, worker_count)
            && group_fusion_pays_for_traffic(
                attended_window,
                cache_dim,
                v.dim,
                q.batch * self.kv_num_heads,
            )
            && group_fusion_enabled();
        let compute_group = |b: usize, kvh: usize, out_block: &mut [f32], scores: &mut Vec<f32>| {
            let causal_limit = query_starts[b];
            let local_start = if self.local_window_size > 0 {
                (causal_limit + 1).saturating_sub(self.local_window_size as usize)
            } else {
                0
            };
            let q_base = (b * self.num_heads + kvh * group) * cache_dim;
            let q_group = &q.data[q_base..q_base + group * cache_dim];
            let kv_head_base = (b * self.kv_num_heads + kvh) * present_sequence_length;
            let k_head = &present_k
                [kv_head_base * cache_dim..(kv_head_base + present_sequence_length) * cache_dim];
            let v_head =
                &present_v[kv_head_base * v.dim..(kv_head_base + present_sequence_length) * v.dim];
            sdpa_decode_group(
                q_group,
                k_head,
                v_head,
                present_sequence_length,
                group,
                cache_dim,
                v.dim,
                local_start,
                causal_limit + 1,
                scale,
                (self.softcap != 0.0).then_some(self.softcap),
                SoftmaxExp::F64Intermediate,
                out_block,
                scores,
            );
        };
        // Flash-decoding (split-KV): when the attended window is long and the
        // decode pool has more workers than query heads, the per-head schedule
        // leaves cores idle (it parallelizes only over `attention_rows`). Split
        // each head's window into `split_count` contiguous KV chunks so those
        // idle cores stream the KV cache in parallel, then combine the per-chunk
        // softmax partials with the online-rescale reduction.
        let split_count = if attention_split_enabled()
            && attention_rows >= 1
            && worker_count > attention_rows
            && total_sequence_length >= SPLIT_MIN_KV
        {
            // One idle-core budget per head (floored): only split when there is
            // real headroom past the per-head schedule.
            let cores_per_head = worker_count / attention_rows;
            // Keep each chunk above the fork-join grain floor.
            let grain_limit = (total_sequence_length / SPLIT_MIN_CHUNK).max(1);
            cores_per_head.min(grain_limit)
        } else {
            1
        };
        if split_count > 1 {
            ATTENTION_SPLIT_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let v_head_size = v.dim;
            let num_tasks = attention_rows * split_count;
            let mut partials = vec![
                DecodePartial {
                    max: f64::NEG_INFINITY,
                    sum: 0.0,
                };
                num_tasks
            ];
            let mut partial_outputs = vec![0.0f64; num_tasks * v_head_size];
            let scratch = SplitPartialScratch {
                partials: partials.as_mut_ptr(),
                outputs: partial_outputs.as_mut_ptr(),
            };
            let scratch = &scratch;
            let softcap = (self.softcap != 0.0).then_some(self.softcap);
            crate::kernels::matmul_nbits::decode_parallel_index_tasks(num_tasks, |task_index| {
                let row_index = task_index / split_count;
                let chunk = task_index % split_count;
                let b = row_index / (self.num_heads * q.seq);
                let row_in_batch = row_index % (self.num_heads * q.seq);
                let qh = row_in_batch / q.seq;
                let qs = row_in_batch % q.seq;
                let kvh = qh / group;
                let causal_limit = query_starts[b] + qs;
                let local_start = if self.local_window_size > 0 {
                    (causal_limit + 1).saturating_sub(self.local_window_size as usize)
                } else {
                    0
                };
                let (chunk_lo, chunk_hi) =
                    split_chunk_bounds(local_start, causal_limit + 1, split_count, chunk);
                let q_base = ((b * self.num_heads + qh) * q.seq + qs) * cache_dim;
                let q_row = &q.data[q_base..q_base + cache_dim];
                let kv_head_base = (b * self.kv_num_heads + kvh) * present_sequence_length;
                let k_head = &present_k[kv_head_base * cache_dim
                    ..(kv_head_base + present_sequence_length) * cache_dim];
                let v_head = &present_v[kv_head_base * v_head_size
                    ..(kv_head_base + present_sequence_length) * v_head_size];
                // SAFETY: this task owns slot `task_index` exclusively (each
                // index runs once), so these writes never alias another task.
                let partial_output = unsafe {
                    std::slice::from_raw_parts_mut(
                        scratch.outputs.add(task_index * v_head_size),
                        v_head_size,
                    )
                };
                let partial = sdpa_decode_partial(
                    q_row,
                    k_head,
                    v_head,
                    present_sequence_length,
                    chunk_lo,
                    chunk_hi,
                    scale,
                    softcap,
                    partial_output,
                );
                unsafe {
                    *scratch.partials.add(task_index) = partial;
                }
            });
            for row_index in 0..attention_rows {
                let base = row_index * split_count;
                combine_decode_partials(
                    &partials[base..base + split_count],
                    &partial_outputs[base * v_head_size..(base + split_count) * v_head_size],
                    v_head_size,
                    &mut y_bhsd[row_index * v_head_size..(row_index + 1) * v_head_size],
                );
            }
        } else if group_fused {
            // One task per `(batch, kv_head)` group; each writes the `group`
            // contiguous output rows of that group (`q_seq == 1`, so a query
            // head's single output row sits at `(b * num_heads + qh) * v.dim`
            // and the group's rows are adjacent).
            let group_rows = q.batch * self.kv_num_heads;
            GROUP_FUSED_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if group_rows > 1 && attention_work >= MIN_PARALLEL_ATTENTION_WORK {
                // One `group * window` score buffer per *worker*, not per task:
                // the closure runs once per group, so allocating inside it
                // would put a multi-hundred-KiB malloc in the decode hot loop.
                thread_local! {
                    static GROUP_SCORES: std::cell::RefCell<Vec<f32>> =
                        const { std::cell::RefCell::new(Vec::new()) };
                }
                crate::kernels::matmul_nbits::decode_parallel_output_row_blocks(
                    &mut y_bhsd,
                    group * v.dim,
                    group_rows,
                    |row_index, out_block| {
                        let b = row_index / self.kv_num_heads;
                        let kvh = row_index % self.kv_num_heads;
                        GROUP_SCORES.with(|scores| {
                            compute_group(b, kvh, out_block, &mut scores.borrow_mut());
                        });
                    },
                );
            } else {
                let mut scores = Vec::new();
                for b in 0..q.batch {
                    for kvh in 0..self.kv_num_heads {
                        let base = (b * self.kv_num_heads + kvh) * group * v.dim;
                        compute_group(b, kvh, &mut y_bhsd[base..base + group * v.dim], &mut scores);
                    }
                }
            }
        } else if attention_rows > 1 && attention_work >= MIN_PARALLEL_ATTENTION_WORK {
            // Route through the active decode pool (the same resident workers the
            // MatMulNBits projections use). Under the persistent SPMD scope this
            // avoids falling to the global Rayon pool, which would contend with
            // the SPMD pool's pinned, spinning workers; under numa-split/flat
            // scopes it runs on their bounded pool exactly as before.
            crate::kernels::matmul_nbits::decode_parallel_output_row_blocks(
                &mut y_bhsd,
                v.dim,
                attention_rows,
                |row_index, output_row| {
                    let b = row_index / (self.num_heads * q.seq);
                    let row_in_batch = row_index % (self.num_heads * q.seq);
                    let qh = row_in_batch / q.seq;
                    let qs = row_in_batch % q.seq;
                    compute_row(b, qh, qs, output_row);
                },
            );
        } else {
            for b in 0..q.batch {
                for qh in 0..self.num_heads {
                    for qs in 0..q.seq {
                        let row_index = (b * self.num_heads + qh) * q.seq + qs;
                        compute_row(
                            b,
                            qh,
                            qs,
                            &mut y_bhsd[row_index * v.dim..(row_index + 1) * v.dim],
                        );
                    }
                }
            }
        }
        let mut output = vec![0.0; y_bhsd.len()];
        #[cfg(feature = "gqa_phase_profile")]
        {
            drop(_attn_phase);
        }
        #[cfg(feature = "gqa_phase_profile")]
        let _out_phase = phase_prof::Phase::start(&phase_prof::OUT_NS);
        for b in 0..q.batch {
            for s in 0..q.seq {
                for h in 0..self.num_heads {
                    for d in 0..v.dim {
                        output[((b * q.seq + s) * self.num_heads + h) * v.dim + d] =
                            y_bhsd[((b * self.num_heads + h) * q.seq + s) * v.dim + d];
                    }
                }
            }
        }
        let decode_fast_write = q.seq == 1 && k.seq == 1;
        if decode_fast_write {
            write_decode_output(&mut outputs[0], &output)?;
        } else {
            write_dense_f32_narrow("GroupQueryAttention", &mut outputs[0], &output)?;
        }
        // In the in-place fast path the present outputs ARE the past buffer and
        // were already updated in place above, so re-emitting them would be a
        // redundant self-copy — skip it. The copy path materializes them here.
        if !in_place {
            if outputs.len() >= 2 {
                if decode_fast_write {
                    write_decode_output(&mut outputs[1], present_k)?;
                } else {
                    write_dense_f32_narrow("GroupQueryAttention", &mut outputs[1], present_k)?;
                }
            }
            if outputs.len() >= 3 {
                if decode_fast_write {
                    write_decode_output(&mut outputs[2], present_v)?;
                } else {
                    write_dense_f32_narrow("GroupQueryAttention", &mut outputs[2], present_v)?;
                }
            }
        }
        #[cfg(feature = "gqa_phase_profile")]
        {
            drop(_out_phase);
            drop(_total_phase);
            phase_prof::tick();
        }
        Ok(())
    }

    fn supports_strided_input(&self, _input_idx: usize) -> bool {
        true
    }

    /// GQA FLOPs are `4 * head_size * batch * num_heads * seq_q * seq_k`
    /// (see [`crate::kernels::flops::gqa_flops`]). We deliberately return `None`:
    /// the dominant factor `seq_k` is the KV-cache occupancy, which the op
    /// receives as the runtime **value** inputs `seqlens_k` (input 5) and
    /// `total_sequence_length` (input 6), not as a static tensor shape. It
    /// therefore cannot be known at kernel-build time. Per issue #995 the honest
    /// representation of an unmeasurable quantity is `None`, never a fabricated
    /// number; the cost model computes the real figure via `gqa_flops` once the
    /// KV length is known at placement/runtime.
    fn estimated_flops(&self) -> Option<u64> {
        None
    }
}

/// Normalize the one exporter layout that is unambiguous: a rank-zero
/// `seqlens_k` represents the sole row of a unit batch.
fn normalized_sequence_lengths(view: &TensorView, batch: usize) -> Result<Vec<i64>> {
    if view.dtype != onnx_runtime_ir::DataType::Int32 {
        return Err(EpError::KernelFailed(format!(
            "GroupQueryAttention: seqlens_k must be int32, got {:?}",
            view.dtype
        )));
    }
    let scalar = view.shape.is_empty() && view.numel() == 1;
    if scalar {
        if batch == 1 {
            return to_dense_i64(view);
        }
        return Err(EpError::KernelFailed(format!(
            "GroupQueryAttention: scalar seqlens_k can only be promoted to [1] when batch_size is 1, got batch_size {batch}; provide int32 [batch_size] values for every row"
        )));
    }
    if view.shape != [batch] && view.shape != [batch, 1] {
        return Err(EpError::KernelFailed(format!(
            "GroupQueryAttention: seqlens_k must be int32 [batch_size], [batch_size, 1], or a scalar for batch_size 1; got shape {:?}",
            view.shape
        )));
    }
    to_dense_i64(view)
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::CpuExecutionProvider;
    use crate::kernels::testutil::Owned;
    use onnx_runtime_ep_api::{ExecutionProvider, TensorView};
    use onnx_runtime_ir::{Attribute, DataType, Graph, Node, NodeId, static_shape};
    use onnx_runtime_loader::Model;

    fn absent() -> TensorView<'static> {
        TensorView::absent(DataType::Float32)
    }

    fn kernel(attrs: &[(&str, Attribute)]) -> Box<dyn Kernel> {
        let mut graph = Graph::new();
        graph.opset_imports.insert("com.microsoft".into(), 1);
        let inputs = [
            ("query", DataType::Float32, vec![1, 1, 8]),
            ("key", DataType::Float32, vec![1, 1, 4]),
            ("value", DataType::Float32, vec![1, 1, 4]),
            ("past_key", DataType::Float32, vec![1, 2, 0, 2]),
            ("past_value", DataType::Float32, vec![1, 2, 0, 2]),
            ("seqlens_k", DataType::Int32, vec![1]),
            ("total_sequence_length", DataType::Int32, vec![]),
        ]
        .into_iter()
        .map(|(name, dtype, shape)| {
            let value = graph.create_named_value(name, dtype, static_shape(shape));
            graph.add_input(value);
            Some(value)
        })
        .collect();
        let output = graph.create_named_value("output", DataType::Float32, static_shape([1, 1, 8]));
        let mut node = Node::new(NodeId(0), "GroupQueryAttention", inputs, vec![output]);
        node.domain = "com.microsoft".into();
        for (name, value) in attrs {
            node.attributes.insert((*name).into(), value.clone());
        }
        let id = graph.insert_node(node);
        graph.add_output(output);
        let model = Model::new(&graph);
        CpuExecutionProvider::new()
            .get_kernel(model.graph.node(id), &[], 1)
            .unwrap()
    }

    fn gqa_kernel(extra: &[(&str, Attribute)]) -> Box<dyn Kernel> {
        gqa_kernel_with_heads(4, 2, extra)
    }

    fn gqa_kernel_with_heads(
        num_heads: i64,
        kv_num_heads: i64,
        extra: &[(&str, Attribute)],
    ) -> Box<dyn Kernel> {
        let mut attrs = vec![
            ("num_heads", Attribute::Int(num_heads)),
            ("kv_num_heads", Attribute::Int(kv_num_heads)),
        ];
        attrs.extend_from_slice(extra);
        kernel(&attrs)
    }

    fn reference(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        q_seq: usize,
        total: usize,
        past: usize,
    ) -> Vec<f32> {
        let (qh, kvh, d) = (4, 2, 2);
        let mut out = vec![0.0; q_seq * qh * d];
        for s in 0..q_seq {
            for h in 0..qh {
                let kh = h / (qh / kvh);
                let mut scores = vec![0.0; past + s + 1];
                for j in 0..scores.len() {
                    scores[j] = (0..d)
                        .map(|x| q[(s * qh + h) * d + x] * k[(kh * total + j) * d + x])
                        .sum::<f32>()
                        / (d as f32).sqrt();
                }
                let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let sum: f32 = scores
                    .iter_mut()
                    .map(|x| {
                        *x = ((*x - max) as f64).exp() as f32;
                        *x
                    })
                    .sum();
                for x in &mut scores {
                    *x /= sum;
                }
                for x in 0..d {
                    out[(s * qh + h) * d + x] = scores
                        .iter()
                        .enumerate()
                        .map(|(j, p)| p * v[(kh * total + j) * d + x])
                        .sum();
                }
            }
        }
        out
    }

    fn reference_with_geometry(
        query: &[f32],
        key: &[f32],
        value: &[f32],
        query_sequence_length: usize,
        total_sequence_length: usize,
        past_sequence_length: usize,
        query_head_count: usize,
        key_value_head_count: usize,
        head_width: usize,
    ) -> Vec<f32> {
        let mut output = vec![0.0; query_sequence_length * query_head_count * head_width];
        for sequence_index in 0..query_sequence_length {
            for query_head_index in 0..query_head_count {
                let key_value_head_index =
                    query_head_index / (query_head_count / key_value_head_count);
                let attended_key_count = past_sequence_length + sequence_index + 1;
                let mut scores = vec![0.0; attended_key_count];
                for (key_index, score) in scores.iter_mut().enumerate() {
                    let query_base =
                        (sequence_index * query_head_count + query_head_index) * head_width;
                    let key_base =
                        (key_value_head_index * total_sequence_length + key_index) * head_width;
                    *score = query[query_base..query_base + head_width]
                        .iter()
                        .zip(&key[key_base..key_base + head_width])
                        .map(|(query_element, key_element)| query_element * key_element)
                        .sum::<f32>()
                        / (head_width as f32).sqrt();
                }
                let maximum_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let probability_sum: f32 = scores
                    .iter_mut()
                    .map(|score| {
                        *score = ((*score - maximum_score) as f64).exp() as f32;
                        *score
                    })
                    .sum();
                for score in &mut scores {
                    *score /= probability_sum;
                }
                let output_base =
                    (sequence_index * query_head_count + query_head_index) * head_width;
                for dimension_index in 0..head_width {
                    output[output_base + dimension_index] = scores
                        .iter()
                        .enumerate()
                        .map(|(key_index, probability)| {
                            probability
                                * value[(key_value_head_index * total_sequence_length + key_index)
                                    * head_width
                                    + dimension_index]
                        })
                        .sum();
                }
            }
        }
        output
    }

    fn mixed_scale_value(index: usize, seed: u64) -> f32 {
        let mut state = (index as u64)
            .wrapping_add(seed)
            .wrapping_add(0x9e37_79b9_7f4a_7c15);
        state ^= state >> 30;
        state = state.wrapping_mul(0xbf58_476d_1ce4_e5b9);
        state ^= state >> 27;
        state = state.wrapping_mul(0x94d0_49bb_1331_11eb);
        state ^= state >> 31;
        let signed_unit = (((state >> 40) as u32) as f32 / ((1_u32 << 24) as f32)) * 2.0 - 1.0;
        let scale = [0.03125_f32, 0.125, 0.5, 2.0][((state >> 8) & 3) as usize];
        signed_unit * scale
    }

    fn close(got: &[f32], want: &[f32]) {
        assert_eq!(got.len(), want.len());
        for (i, (a, b)) in got.iter().zip(want).enumerate() {
            assert!((a - b).abs() < 1e-5, "{i}: {a} != {b}");
        }
    }

    fn reference_rope_bsh(
        input: &[f32],
        seq: usize,
        heads: usize,
        positions: &[usize],
        cos: &[f32],
        sin: &[f32],
    ) -> Vec<f32> {
        let mut output = input.to_vec();
        for s in 0..seq {
            for h in 0..heads {
                let base = (s * heads + h) * 2;
                let (x0, x1) = (input[base], input[base + 1]);
                output[base] = cos[positions[s]] * x0 - sin[positions[s]] * x1;
                output[base + 1] = sin[positions[s]] * x0 + cos[positions[s]] * x1;
            }
        }
        output
    }

    fn bsh_to_bnsh(input: &[f32], seq: usize, heads: usize) -> Vec<f32> {
        let mut output = vec![0.0; input.len()];
        for s in 0..seq {
            for h in 0..heads {
                output[(h * seq + s) * 2..(h * seq + s + 1) * 2]
                    .copy_from_slice(&input[(s * heads + h) * 2..(s * heads + h + 1) * 2]);
            }
        }
        output
    }

    fn reference_rope_bsh_geometry(
        input: &[f32],
        seq: usize,
        heads: usize,
        head_width: usize,
        rotary_dim: usize,
        positions: &[usize],
        cos: &[f32],
        sin: &[f32],
    ) -> Vec<f32> {
        let half = rotary_dim / 2;
        let mut output = input.to_vec();
        // Keep the shared sequence index for the input, output, and position arrays.
        #[allow(clippy::needless_range_loop)]
        for s in 0..seq {
            for h in 0..heads {
                let base = (s * heads + h) * head_width;
                for k in 0..half {
                    let x0 = input[base + k];
                    let x1 = input[base + half + k];
                    let cache = positions[s] * half + k;
                    output[base + k] = cos[cache] * x0 - sin[cache] * x1;
                    output[base + half + k] = sin[cache] * x0 + cos[cache] * x1;
                }
            }
        }
        output
    }

    fn bsh_to_bnsh_geometry(
        input: &[f32],
        seq: usize,
        heads: usize,
        head_width: usize,
    ) -> Vec<f32> {
        let mut output = vec![0.0; input.len()];
        for s in 0..seq {
            for h in 0..heads {
                let src = (s * heads + h) * head_width;
                let dst = (h * seq + s) * head_width;
                output[dst..dst + head_width].copy_from_slice(&input[src..src + head_width]);
            }
        }
        output
    }

    #[test]
    fn prefill_gqa_grouping_and_causal_match_reference() {
        let q = vec![
            1., 0., 1., 0., 0., 1., 0., 1., 0., 1., 0., 1., 1., 0., 1., 0., 1., 1., 1., 1., -1.,
            1., -1., 1.,
        ];
        let k_bsh = vec![1., 0., 0., 1., 0., 1., 1., 0., 1., 1., -1., 1.];
        let v_bsh = vec![1., 2., 10., 20., 3., 4., 30., 40., 5., 6., 50., 60.];
        let mut k_bnsh = vec![0.0; 12];
        let mut v_bnsh = vec![0.0; 12];
        for h in 0..2 {
            for s in 0..3 {
                for d in 0..2 {
                    k_bnsh[(h * 3 + s) * 2 + d] = k_bsh[(s * 2 + h) * 2 + d];
                    v_bnsh[(h * 3 + s) * 2 + d] = v_bsh[(s * 2 + h) * 2 + d];
                }
            }
        }
        let mut out = Owned::zeros_f32(&[1, 3, 8]);
        let mut pk = Owned::zeros_f32(&[1, 2, 3, 2]);
        let mut pv = Owned::zeros_f32(&[1, 2, 3, 2]);
        gqa_kernel(&[])
            .execute(
                &[
                    Owned::f32(&[1, 3, 8], &q).view(),
                    Owned::f32(&[1, 3, 4], &k_bsh).view(),
                    Owned::f32(&[1, 3, 4], &v_bsh).view(),
                    absent(),
                    absent(),
                    Owned::i32(&[1], &[2]).view(),
                    Owned::i32(&[], &[3]).view(),
                ],
                &mut [out.view_mut(), pk.view_mut(), pv.view_mut()],
            )
            .unwrap();
        close(&out.to_f32(), &reference(&q, &k_bnsh, &v_bnsh, 3, 3, 0));
        assert_eq!(pk.shape, vec![1, 2, 3, 2]);
        close(&pk.to_f32(), &k_bnsh);
        close(&pv.to_f32(), &v_bnsh);
    }

    #[test]
    fn unit_batch_scalar_seqlens_matches_canonical_vector() {
        let q = [1., 0., 1., 0., 0., 1., 0., 1.];
        let k = [1., 0., 0., 1.];
        let v = [1., 2., 10., 20.];
        let run = |seqlens_shape: &[usize]| {
            let mut out = Owned::zeros_f32(&[1, 1, 8]);
            let mut present_k = Owned::zeros_f32(&[1, 2, 1, 2]);
            let mut present_v = Owned::zeros_f32(&[1, 2, 1, 2]);
            gqa_kernel(&[])
                .execute(
                    &[
                        Owned::f32(&[1, 1, 8], &q).view(),
                        Owned::f32(&[1, 1, 4], &k).view(),
                        Owned::f32(&[1, 1, 4], &v).view(),
                        absent(),
                        absent(),
                        Owned::i32(seqlens_shape, &[0]).view(),
                        Owned::i32(&[], &[1]).view(),
                    ],
                    &mut [out.view_mut(), present_k.view_mut(), present_v.view_mut()],
                )
                .unwrap();
            (out.to_f32(), present_k.to_f32(), present_v.to_f32())
        };

        assert_eq!(run(&[]), run(&[1]));
    }

    #[test]
    fn scalar_seqlens_rejects_multi_row_batch_and_wrong_dtype() {
        let multi_row = Owned::i32(&[], &[0]);
        let error = normalized_sequence_lengths(&multi_row.view(), 2)
            .expect_err("a scalar cannot represent two batch rows");
        assert!(format!("{error}").contains("batch_size 2"));

        let wrong_dtype = Owned::i64(&[], &[0]);
        let error = normalized_sequence_lengths(&wrong_dtype.view(), 1)
            .expect_err("seqlens_k must be int32");
        assert!(format!("{error}").contains("must be int32"));
    }

    #[test]
    fn large_prefill_parallel_path_matches_reference() {
        let seq = 160;
        let q = (0..seq * 8)
            .map(|i| ((i % 17) as f32 - 8.0) / 8.0)
            .collect::<Vec<_>>();
        let k_bsh = (0..seq * 4)
            .map(|i| ((i % 13) as f32 - 6.0) / 7.0)
            .collect::<Vec<_>>();
        let v_bsh = (0..seq * 4)
            .map(|i| ((i % 19) as f32 - 9.0) / 9.0)
            .collect::<Vec<_>>();
        let k_bnsh = bsh_to_bnsh(&k_bsh, seq, 2);
        let v_bnsh = bsh_to_bnsh(&v_bsh, seq, 2);
        let mut out = Owned::zeros_f32(&[1, seq, 8]);

        gqa_kernel(&[])
            .execute(
                &[
                    Owned::f32(&[1, seq, 8], &q).view(),
                    Owned::f32(&[1, seq, 4], &k_bsh).view(),
                    Owned::f32(&[1, seq, 4], &v_bsh).view(),
                    absent(),
                    absent(),
                    Owned::i32(&[1], &[(seq - 1) as i32]).view(),
                    Owned::i32(&[], &[seq as i32]).view(),
                ],
                &mut [out.view_mut()],
            )
            .unwrap();

        close(&out.to_f32(), &reference(&q, &k_bnsh, &v_bnsh, seq, seq, 0));
    }

    #[test]
    fn packed_qkv_matches_unpacked_and_independent_reference() {
        let q = vec![
            1., 0., 1., 0., 0., 1., 0., 1., 0., 1., 0., 1., 1., 0., 1., 0.,
        ];
        let k_bsh = vec![1., 0., 0., 1., 0., 1., 1., 0.];
        let v_bsh = vec![1., 2., 10., 20., 3., 4., 30., 40.];
        let mut packed = Vec::with_capacity(q.len() + k_bsh.len() + v_bsh.len());
        for s in 0..2 {
            packed.extend_from_slice(&q[s * 8..(s + 1) * 8]);
            packed.extend_from_slice(&k_bsh[s * 4..(s + 1) * 4]);
            packed.extend_from_slice(&v_bsh[s * 4..(s + 1) * 4]);
        }
        let k_bnsh = bsh_to_bnsh(&k_bsh, 2, 2);
        let v_bnsh = bsh_to_bnsh(&v_bsh, 2, 2);
        let want = reference(&q, &k_bnsh, &v_bnsh, 2, 2, 0);

        let mut unpacked_out = Owned::zeros_f32(&[1, 2, 8]);
        let mut packed_out = Owned::zeros_f32(&[1, 2, 8]);
        let mut unpacked_k = Owned::zeros_f32(&[1, 2, 2, 2]);
        let mut unpacked_v = Owned::zeros_f32(&[1, 2, 2, 2]);
        let mut packed_k = Owned::zeros_f32(&[1, 2, 2, 2]);
        let mut packed_v = Owned::zeros_f32(&[1, 2, 2, 2]);
        gqa_kernel(&[])
            .execute(
                &[
                    Owned::f32(&[1, 2, 8], &q).view(),
                    Owned::f32(&[1, 2, 4], &k_bsh).view(),
                    Owned::f32(&[1, 2, 4], &v_bsh).view(),
                    absent(),
                    absent(),
                    Owned::i32(&[1], &[1]).view(),
                    Owned::i32(&[], &[2]).view(),
                ],
                &mut [
                    unpacked_out.view_mut(),
                    unpacked_k.view_mut(),
                    unpacked_v.view_mut(),
                ],
            )
            .unwrap();
        gqa_kernel(&[])
            .execute(
                &[
                    Owned::f32(&[1, 2, 16], &packed).view(),
                    absent(),
                    absent(),
                    absent(),
                    absent(),
                    Owned::i32(&[1], &[1]).view(),
                    Owned::i32(&[], &[2]).view(),
                ],
                &mut [
                    packed_out.view_mut(),
                    packed_k.view_mut(),
                    packed_v.view_mut(),
                ],
            )
            .unwrap();

        close(&unpacked_out.to_f32(), &want);
        close(&packed_out.to_f32(), &want);
        close(&packed_out.to_f32(), &unpacked_out.to_f32());
        close(&packed_k.to_f32(), &unpacked_k.to_f32());
        close(&packed_v.to_f32(), &unpacked_v.to_f32());
    }

    #[test]
    fn decode_appends_past_and_matches_reference() {
        let q = vec![1., 0., 1., 0., 0., 1., 0., 1.];
        let past_k = vec![1., 0., 0., 1., 10., 0., 0., 10.];
        let past_v = vec![1., 2., 3., 4., 10., 20., 30., 40.];
        let cur_k = vec![1., 1., 10., 10.];
        let cur_v = vec![5., 6., 50., 60.];
        let mut all_k = vec![0.0; 12];
        let mut all_v = vec![0.0; 12];
        for h in 0..2 {
            all_k[h * 6..h * 6 + 4].copy_from_slice(&past_k[h * 4..h * 4 + 4]);
            all_v[h * 6..h * 6 + 4].copy_from_slice(&past_v[h * 4..h * 4 + 4]);
            all_k[h * 6 + 4..h * 6 + 6].copy_from_slice(&cur_k[h * 2..h * 2 + 2]);
            all_v[h * 6 + 4..h * 6 + 6].copy_from_slice(&cur_v[h * 2..h * 2 + 2]);
        }
        let mut out = Owned::zeros_f32(&[1, 1, 8]);
        let mut pk = Owned::zeros_f32(&[1, 2, 3, 2]);
        let mut pv = Owned::zeros_f32(&[1, 2, 3, 2]);
        gqa_kernel(&[])
            .execute(
                &[
                    Owned::f32(&[1, 1, 8], &q).view(),
                    Owned::f32(&[1, 1, 4], &cur_k).view(),
                    Owned::f32(&[1, 1, 4], &cur_v).view(),
                    Owned::f32(&[1, 2, 2, 2], &past_k).view(),
                    Owned::f32(&[1, 2, 2, 2], &past_v).view(),
                    Owned::i32(&[1], &[2]).view(),
                    Owned::i32(&[], &[3]).view(),
                ],
                &mut [out.view_mut(), pk.view_mut(), pv.view_mut()],
            )
            .unwrap();
        close(&pk.to_f32(), &all_k);
        close(&pv.to_f32(), &all_v);
        close(&out.to_f32(), &reference(&q, &all_k, &all_v, 1, 3, 2));
    }

    fn run_nonstandard_head_width_decode(head_width: usize, rotary_dim: usize) {
        const QUERY_HEAD_COUNT: usize = 4;
        const KEY_VALUE_HEAD_COUNT: usize = 2;
        const PAST_SEQUENCE_LENGTH: usize = 2;
        const TOTAL_SEQUENCE_LENGTH: usize = PAST_SEQUENCE_LENGTH + 1;
        let half = rotary_dim / 2;

        let query: Vec<f32> = (0..QUERY_HEAD_COUNT * head_width)
            .map(|i| mixed_scale_value(i, 0x4811))
            .collect();
        let current_key: Vec<f32> = (0..KEY_VALUE_HEAD_COUNT * head_width)
            .map(|i| mixed_scale_value(i, 0x4822))
            .collect();
        let current_value: Vec<f32> = (0..KEY_VALUE_HEAD_COUNT * head_width)
            .map(|i| mixed_scale_value(i, 0x4833))
            .collect();
        let past_key: Vec<f32> = (0..KEY_VALUE_HEAD_COUNT * PAST_SEQUENCE_LENGTH * head_width)
            .map(|i| mixed_scale_value(i, 0x4844))
            .collect();
        let past_value: Vec<f32> = (0..KEY_VALUE_HEAD_COUNT * PAST_SEQUENCE_LENGTH * head_width)
            .map(|i| mixed_scale_value(i, 0x4855))
            .collect();
        let cos: Vec<f32> = (0..TOTAL_SEQUENCE_LENGTH * half)
            .map(|i| (i as f32 * 0.013).cos())
            .collect();
        let sin: Vec<f32> = (0..TOTAL_SEQUENCE_LENGTH * half)
            .map(|i| (i as f32 * 0.013).sin())
            .collect();

        let query_rotated = reference_rope_bsh_geometry(
            &query,
            1,
            QUERY_HEAD_COUNT,
            head_width,
            rotary_dim,
            &[PAST_SEQUENCE_LENGTH],
            &cos,
            &sin,
        );
        let current_key_rotated_bsh = reference_rope_bsh_geometry(
            &current_key,
            1,
            KEY_VALUE_HEAD_COUNT,
            head_width,
            rotary_dim,
            &[PAST_SEQUENCE_LENGTH],
            &cos,
            &sin,
        );
        let current_key_rotated = bsh_to_bnsh_geometry(
            &current_key_rotated_bsh,
            1,
            KEY_VALUE_HEAD_COUNT,
            head_width,
        );
        let current_value_bnsh =
            bsh_to_bnsh_geometry(&current_value, 1, KEY_VALUE_HEAD_COUNT, head_width);

        let mut expected_key = vec![0.0; KEY_VALUE_HEAD_COUNT * TOTAL_SEQUENCE_LENGTH * head_width];
        let mut expected_value = expected_key.clone();
        for h in 0..KEY_VALUE_HEAD_COUNT {
            let past_src = h * PAST_SEQUENCE_LENGTH * head_width;
            let present_dst = h * TOTAL_SEQUENCE_LENGTH * head_width;
            let past_len = PAST_SEQUENCE_LENGTH * head_width;
            expected_key[present_dst..present_dst + past_len]
                .copy_from_slice(&past_key[past_src..past_src + past_len]);
            expected_value[present_dst..present_dst + past_len]
                .copy_from_slice(&past_value[past_src..past_src + past_len]);
            expected_key[present_dst + past_len..present_dst + past_len + head_width]
                .copy_from_slice(&current_key_rotated[h * head_width..(h + 1) * head_width]);
            expected_value[present_dst + past_len..present_dst + past_len + head_width]
                .copy_from_slice(&current_value_bnsh[h * head_width..(h + 1) * head_width]);
        }
        let expected_output = reference_with_geometry(
            &query_rotated,
            &expected_key,
            &expected_value,
            1,
            TOTAL_SEQUENCE_LENGTH,
            PAST_SEQUENCE_LENGTH,
            QUERY_HEAD_COUNT,
            KEY_VALUE_HEAD_COUNT,
            head_width,
        );

        let mut output = Owned::zeros_f32(&[1, 1, QUERY_HEAD_COUNT * head_width]);
        let mut present_key =
            Owned::zeros_f32(&[1, KEY_VALUE_HEAD_COUNT, TOTAL_SEQUENCE_LENGTH, head_width]);
        let mut present_value =
            Owned::zeros_f32(&[1, KEY_VALUE_HEAD_COUNT, TOTAL_SEQUENCE_LENGTH, head_width]);
        gqa_kernel_with_heads(
            QUERY_HEAD_COUNT as i64,
            KEY_VALUE_HEAD_COUNT as i64,
            &[("do_rotary", Attribute::Int(1))],
        )
        .execute(
            &[
                Owned::f32(&[1, 1, QUERY_HEAD_COUNT * head_width], &query).view(),
                Owned::f32(&[1, 1, KEY_VALUE_HEAD_COUNT * head_width], &current_key).view(),
                Owned::f32(&[1, 1, KEY_VALUE_HEAD_COUNT * head_width], &current_value).view(),
                Owned::f32(
                    &[1, KEY_VALUE_HEAD_COUNT, PAST_SEQUENCE_LENGTH, head_width],
                    &past_key,
                )
                .view(),
                Owned::f32(
                    &[1, KEY_VALUE_HEAD_COUNT, PAST_SEQUENCE_LENGTH, head_width],
                    &past_value,
                )
                .view(),
                Owned::i32(&[1], &[PAST_SEQUENCE_LENGTH as i32]).view(),
                Owned::i32(&[], &[TOTAL_SEQUENCE_LENGTH as i32]).view(),
                Owned::f32(&[TOTAL_SEQUENCE_LENGTH, half], &cos).view(),
                Owned::f32(&[TOTAL_SEQUENCE_LENGTH, half], &sin).view(),
            ],
            &mut [
                output.view_mut(),
                present_key.view_mut(),
                present_value.view_mut(),
            ],
        )
        .unwrap();

        close(&present_key.to_f32(), &expected_key);
        close(&present_value.to_f32(), &expected_value);
        close(&output.to_f32(), &expected_output);
    }

    #[test]
    fn decode_head_dim_48_with_rotary_matches_reference_and_kv_cache() {
        run_nonstandard_head_width_decode(48, 48);
    }

    #[test]
    fn decode_head_dim_80_with_partial_rotary_matches_reference_and_kv_cache() {
        run_nonstandard_head_width_decode(80, 32);
    }

    fn run_r1_grouping_decode(head_width: usize, rotary_dimension: usize) {
        const QUERY_HEAD_COUNT: usize = 12;
        const KEY_VALUE_HEAD_COUNT: usize = 2;
        const INITIAL_PAST_LENGTH: usize = 2;
        const DECODE_STEPS: usize = 4;
        const MAX_SEQUENCE_LENGTH: usize = INITIAL_PAST_LENGTH + DECODE_STEPS;

        let rotary_half = rotary_dimension / 2;
        let cosine: Vec<f32> = (0..MAX_SEQUENCE_LENGTH * rotary_half)
            .map(|index| (index as f32 * 0.013).cos())
            .collect();
        let sine: Vec<f32> = (0..MAX_SEQUENCE_LENGTH * rotary_half)
            .map(|index| (index as f32 * 0.013).sin())
            .collect();
        let mut past_key: Vec<f32> = (0..KEY_VALUE_HEAD_COUNT * INITIAL_PAST_LENGTH * head_width)
            .map(|index| mixed_scale_value(index, 0x6411))
            .collect();
        let mut past_value: Vec<f32> = (0..past_key.len())
            .map(|index| mixed_scale_value(index, 0x6422))
            .collect();

        for step in 0..DECODE_STEPS {
            let past_length = INITIAL_PAST_LENGTH + step;
            let total_length = past_length + 1;
            let query: Vec<f32> = (0..QUERY_HEAD_COUNT * head_width)
                .map(|index| mixed_scale_value(index, 0x6433 + step as u64))
                .collect();
            let current_key: Vec<f32> = (0..KEY_VALUE_HEAD_COUNT * head_width)
                .map(|index| mixed_scale_value(index, 0x6444 + step as u64))
                .collect();
            let current_value: Vec<f32> = (0..KEY_VALUE_HEAD_COUNT * head_width)
                .map(|index| mixed_scale_value(index, 0x6455 + step as u64))
                .collect();
            let query_rotated = reference_rope_bsh_geometry(
                &query,
                1,
                QUERY_HEAD_COUNT,
                head_width,
                rotary_dimension,
                &[past_length],
                &cosine,
                &sine,
            );
            let current_key_rotated = bsh_to_bnsh_geometry(
                &reference_rope_bsh_geometry(
                    &current_key,
                    1,
                    KEY_VALUE_HEAD_COUNT,
                    head_width,
                    rotary_dimension,
                    &[past_length],
                    &cosine,
                    &sine,
                ),
                1,
                KEY_VALUE_HEAD_COUNT,
                head_width,
            );
            let current_value_bnsh =
                bsh_to_bnsh_geometry(&current_value, 1, KEY_VALUE_HEAD_COUNT, head_width);
            let mut expected_key = vec![0.0; KEY_VALUE_HEAD_COUNT * total_length * head_width];
            let mut expected_value = expected_key.clone();
            for head in 0..KEY_VALUE_HEAD_COUNT {
                let past_source = head * past_length * head_width;
                let present_destination = head * total_length * head_width;
                let past_elements = past_length * head_width;
                expected_key[present_destination..present_destination + past_elements]
                    .copy_from_slice(&past_key[past_source..past_source + past_elements]);
                expected_value[present_destination..present_destination + past_elements]
                    .copy_from_slice(&past_value[past_source..past_source + past_elements]);
                expected_key[present_destination + past_elements
                    ..present_destination + past_elements + head_width]
                    .copy_from_slice(
                        &current_key_rotated[head * head_width..(head + 1) * head_width],
                    );
                expected_value[present_destination + past_elements
                    ..present_destination + past_elements + head_width]
                    .copy_from_slice(
                        &current_value_bnsh[head * head_width..(head + 1) * head_width],
                    );
            }
            let expected_output = reference_with_geometry(
                &query_rotated,
                &expected_key,
                &expected_value,
                1,
                total_length,
                past_length,
                QUERY_HEAD_COUNT,
                KEY_VALUE_HEAD_COUNT,
                head_width,
            );

            let mut output = Owned::zeros_f32(&[1, 1, QUERY_HEAD_COUNT * head_width]);
            let mut present_key =
                Owned::zeros_f32(&[1, KEY_VALUE_HEAD_COUNT, total_length, head_width]);
            let mut present_value =
                Owned::zeros_f32(&[1, KEY_VALUE_HEAD_COUNT, total_length, head_width]);
            gqa_kernel_with_heads(
                QUERY_HEAD_COUNT as i64,
                KEY_VALUE_HEAD_COUNT as i64,
                &[("do_rotary", Attribute::Int(1))],
            )
            .execute(
                &[
                    Owned::f32(&[1, 1, QUERY_HEAD_COUNT * head_width], &query).view(),
                    Owned::f32(&[1, 1, KEY_VALUE_HEAD_COUNT * head_width], &current_key).view(),
                    Owned::f32(&[1, 1, KEY_VALUE_HEAD_COUNT * head_width], &current_value).view(),
                    Owned::f32(
                        &[1, KEY_VALUE_HEAD_COUNT, past_length, head_width],
                        &past_key,
                    )
                    .view(),
                    Owned::f32(
                        &[1, KEY_VALUE_HEAD_COUNT, past_length, head_width],
                        &past_value,
                    )
                    .view(),
                    Owned::i32(&[1], &[past_length as i32]).view(),
                    Owned::i32(&[], &[total_length as i32]).view(),
                    Owned::f32(&[MAX_SEQUENCE_LENGTH, rotary_half], &cosine).view(),
                    Owned::f32(&[MAX_SEQUENCE_LENGTH, rotary_half], &sine).view(),
                ],
                &mut [
                    output.view_mut(),
                    present_key.view_mut(),
                    present_value.view_mut(),
                ],
            )
            .unwrap();

            close(&present_key.to_f32(), &expected_key);
            close(&present_value.to_f32(), &expected_value);
            close(&output.to_f32(), &expected_output);
            past_key = expected_key;
            past_value = expected_value;
        }
    }

    #[test]
    fn decode_r1_grouping_head_dim_64_matches_reference_across_steps() {
        run_r1_grouping_decode(64, 64);
    }

    #[test]
    fn decode_r1_grouping_head_dim_128_matches_reference_across_steps() {
        run_r1_grouping_decode(128, 128);
    }

    #[test]
    fn decode_widens_f16_past_cache_before_materializing_present_cache() {
        let q = vec![1., 0., 1., 0., 0., 1., 0., 1.];
        let past_k = vec![1., 0., 0., 1., 10., 0., 0., 10.];
        let past_v = vec![1., 2., 3., 4., 10., 20., 30., 40.];
        let cur_k = vec![1., 1., 10., 10.];
        let cur_v = vec![5., 6., 50., 60.];
        let expected_k = vec![1., 0., 0., 1., 1., 1., 10., 0., 0., 10., 10., 10.];
        let expected_v = vec![1., 2., 3., 4., 5., 6., 10., 20., 30., 40., 50., 60.];
        let mut out = Owned::zeros_f32(&[1, 1, 8]);
        let mut pk = Owned::zeros_f32(&[1, 2, 3, 2]);
        let mut pv = Owned::zeros_f32(&[1, 2, 3, 2]);
        gqa_kernel(&[])
            .execute(
                &[
                    Owned::f32(&[1, 1, 8], &q).view(),
                    Owned::f32(&[1, 1, 4], &cur_k).view(),
                    Owned::f32(&[1, 1, 4], &cur_v).view(),
                    Owned::f16(&[1, 2, 2, 2], &past_k).view(),
                    Owned::f16(&[1, 2, 2, 2], &past_v).view(),
                    Owned::i32(&[1], &[2]).view(),
                    Owned::i32(&[], &[3]).view(),
                ],
                &mut [out.view_mut(), pk.view_mut(), pv.view_mut()],
            )
            .unwrap();
        close(&pk.to_f32(), &expected_k);
        close(&pv.to_f32(), &expected_v);
        close(
            &out.to_f32(),
            &reference(&q, &expected_k, &expected_v, 1, 3, 2),
        );
    }

    /// Materialize an f16 cache as the pre-`eedbf93` decode path did before
    /// copying its dense f32 result into `present`.
    fn old_full_widen_f16(bits: &[u16]) -> Vec<f32> {
        bits.iter()
            .map(|&bits| half::f16::from_bits(bits).to_f32())
            .collect()
    }

    fn assert_f32_bits_eq(actual: &[f32], expected: &[f32], label: &str) {
        assert_eq!(actual.len(), expected.len(), "{label} length");
        for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            assert_eq!(
                actual.to_bits(),
                expected.to_bits(),
                "{label}[{index}]: {actual:?} != {expected:?}"
            );
        }
    }

    fn rotary_tensor(batch: usize, heads: usize, seq: usize, dim: usize, seed: f32) -> Bhsd {
        Bhsd {
            data: (0..batch * heads * seq * dim)
                .map(|index| seed + index as f32 * 0.03125)
                .collect(),
            batch,
            heads,
            seq,
            dim,
        }
    }

    fn assert_bounded_rotary_matches_full_widen_bitwise(
        cos_view: TensorView,
        sin_view: TensorView,
        positions: &[usize],
        batch: usize,
        seq: usize,
        label: &str,
    ) {
        let half = cos_view.shape[1];
        let cache_rows = cos_view.shape[0];
        let rows_needed = positions.iter().copied().max().unwrap() + 1;
        let full_cos = to_dense_f32_widen("test", &cos_view).unwrap().into_owned();
        let full_sin = to_dense_f32_widen("test", &sin_view).unwrap().into_owned();
        let bounded_cos = widen_rotary_prefix("test", &cos_view, rows_needed, half).unwrap();
        let bounded_sin = widen_rotary_prefix("test", &sin_view, rows_needed, half).unwrap();

        for (tensor_label, heads, seed) in [("query", 2, 0.25), ("key", 1, -0.75)] {
            let mut full = rotary_tensor(batch, heads, seq, half * 2, seed);
            let mut bounded = rotary_tensor(batch, heads, seq, half * 2, seed);
            rotate(
                &mut full,
                &full_cos,
                &full_sin,
                cache_rows,
                half * 2,
                positions,
                false,
            )
            .unwrap();
            rotate(
                &mut bounded,
                &bounded_cos,
                &bounded_sin,
                rows_needed,
                half * 2,
                positions,
                false,
            )
            .unwrap();
            assert_f32_bits_eq(
                &bounded.data,
                &full.data,
                &format!("{label} {tensor_label} bounded rotary"),
            );
        }
    }

    #[test]
    fn rotary_bounded_widen_is_bit_identical_to_full_cache_for_f16_and_f32() {
        let cache_rows = 11;
        let half = 3;
        let positions = [5, 3, 1];
        let cos: Vec<f32> = (0..cache_rows * half)
            .map(|index| (index as f32 * 0.17).cos())
            .collect();
        let sin: Vec<f32> = (0..cache_rows * half)
            .map(|index| (index as f32 * 0.17).sin())
            .collect();
        let cos_f16 = Owned::f16(&[cache_rows, half], &cos);
        let sin_f16 = Owned::f16(&[cache_rows, half], &sin);
        assert_bounded_rotary_matches_full_widen_bitwise(
            cos_f16.view(),
            sin_f16.view(),
            &positions,
            1,
            3,
            "f16",
        );

        let cos_f32 = Owned::f32(&[cache_rows, half], &cos);
        let sin_f32 = Owned::f32(&[cache_rows, half], &sin);
        assert_bounded_rotary_matches_full_widen_bitwise(
            cos_f32.view(),
            sin_f32.view(),
            &positions,
            1,
            3,
            "f32",
        );
    }

    #[test]
    fn rotary_strided_cache_fallback_matches_contiguous_fast_path_bitwise() {
        let cache_rows = 8;
        let half = 3;
        let rows_needed = 6;
        let positions = [5, 3, 1];
        let cos: Vec<f32> = (0..cache_rows * half)
            .map(|index| (index as f32 * 0.23).cos())
            .collect();
        let sin: Vec<f32> = (0..cache_rows * half)
            .map(|index| (index as f32 * 0.23).sin())
            .collect();
        let mut cos_transposed = vec![0.0; cos.len()];
        let mut sin_transposed = vec![0.0; sin.len()];
        for row in 0..cache_rows {
            for column in 0..half {
                cos_transposed[column * cache_rows + row] = cos[row * half + column];
                sin_transposed[column * cache_rows + row] = sin[row * half + column];
            }
        }
        let cos_contiguous = Owned::f32(&[cache_rows, half], &cos);
        let sin_contiguous = Owned::f32(&[cache_rows, half], &sin);
        let cos_strided = Owned::f32(&[cache_rows, half], &cos_transposed)
            .with_view(&[cache_rows, half], &[1, cache_rows as i64]);
        let sin_strided = Owned::f32(&[cache_rows, half], &sin_transposed)
            .with_view(&[cache_rows, half], &[1, cache_rows as i64]);

        let fast_cos =
            widen_rotary_prefix("test", &cos_contiguous.view(), rows_needed, half).unwrap();
        let fast_sin =
            widen_rotary_prefix("test", &sin_contiguous.view(), rows_needed, half).unwrap();
        let fallback_cos =
            widen_rotary_prefix("test", &cos_strided.view(), rows_needed, half).unwrap();
        let fallback_sin =
            widen_rotary_prefix("test", &sin_strided.view(), rows_needed, half).unwrap();
        assert_f32_bits_eq(&fallback_cos, &fast_cos, "strided cos fallback");
        assert_f32_bits_eq(&fallback_sin, &fast_sin, "strided sin fallback");

        let mut fast = rotary_tensor(1, 2, 3, half * 2, 0.5);
        let mut fallback = rotary_tensor(1, 2, 3, half * 2, 0.5);
        rotate(
            &mut fast,
            &fast_cos,
            &fast_sin,
            rows_needed,
            half * 2,
            &positions,
            false,
        )
        .unwrap();
        rotate(
            &mut fallback,
            &fallback_cos,
            &fallback_sin,
            rows_needed,
            half * 2,
            &positions,
            false,
        )
        .unwrap();
        assert_f32_bits_eq(&fallback.data, &fast.data, "strided rotary fallback");
    }

    #[test]
    fn rotary_batch_descending_position_ids_match_full_cache_bitwise() {
        let cache_rows = 9;
        let half = 2;
        let positions = [5, 3, 1, 7, 0, 2];
        let cos: Vec<f32> = (0..cache_rows * half)
            .map(|index| (index as f32 * 0.31).cos())
            .collect();
        let sin: Vec<f32> = (0..cache_rows * half)
            .map(|index| (index as f32 * 0.31).sin())
            .collect();
        let cos_cache = Owned::f32(&[cache_rows, half], &cos);
        let sin_cache = Owned::f32(&[cache_rows, half], &sin);
        assert_bounded_rotary_matches_full_widen_bitwise(
            cos_cache.view(),
            sin_cache.view(),
            &positions,
            2,
            3,
            "batch descending position ids",
        );
    }

    /// Compares lazy per-head f16 widening with the old eager whole-cache
    /// widen, using the production kernel for attention on both sides.
    #[test]
    fn lazy_widen_bit_identical_to_full_widen_multistep() {
        const BATCH: usize = 2;
        const QUERY_HEAD_COUNT: usize = 4;
        const KEY_VALUE_HEAD_COUNT: usize = 2;
        // Deliberately not a vector width: this exercises scalar/vector tails.
        const HEAD_WIDTH: usize = 7;
        const STEPS: usize = 6;

        let kernel =
            gqa_kernel_with_heads(QUERY_HEAD_COUNT as i64, KEY_VALUE_HEAD_COUNT as i64, &[]);
        let mut past_key_bits = Vec::new();
        let mut past_value_bits = Vec::new();

        for step in 0..STEPS {
            let past_sequence_length = step;
            // Step zero has no tail and takes the uninitialized-present fast
            // path. Later steps make batch 1 one token shorter than batch 0,
            // forcing the zero-filled tail path while the shared cache grows.
            let past_lengths = [step as i32, step.saturating_sub(1) as i32];
            let present_sequence_length = step + 1;
            let query: Vec<f32> = (0..BATCH * QUERY_HEAD_COUNT * HEAD_WIDTH)
                .map(|index| mixed_scale_value(index + step * 97, 0x1111))
                .collect();
            let current_key: Vec<f32> = (0..BATCH * KEY_VALUE_HEAD_COUNT * HEAD_WIDTH)
                .map(|index| mixed_scale_value(index + step * 31, 0x2222))
                .collect();
            let current_value: Vec<f32> = (0..BATCH * KEY_VALUE_HEAD_COUNT * HEAD_WIDTH)
                .map(|index| mixed_scale_value(index + step * 57, 0x3333))
                .collect();

            let past_shape = [
                BATCH,
                KEY_VALUE_HEAD_COUNT,
                past_sequence_length,
                HEAD_WIDTH,
            ];
            let past_key = Owned::f16_bits(&past_shape, &past_key_bits);
            let past_value = Owned::f16_bits(&past_shape, &past_value_bits);
            let full_key = old_full_widen_f16(&past_key_bits);
            let full_value = old_full_widen_f16(&past_value_bits);
            let mut lazy_output = Owned::zeros_f32(&[BATCH, 1, QUERY_HEAD_COUNT * HEAD_WIDTH]);
            let mut lazy_present_key = Owned::zeros(
                DataType::Float16,
                &[
                    BATCH,
                    KEY_VALUE_HEAD_COUNT,
                    present_sequence_length,
                    HEAD_WIDTH,
                ],
            );
            let mut lazy_present_value = Owned::zeros(
                DataType::Float16,
                &[
                    BATCH,
                    KEY_VALUE_HEAD_COUNT,
                    present_sequence_length,
                    HEAD_WIDTH,
                ],
            );
            kernel
                .execute(
                    &[
                        Owned::f32(&[BATCH, 1, QUERY_HEAD_COUNT * HEAD_WIDTH], &query).view(),
                        Owned::f32(&[BATCH, 1, KEY_VALUE_HEAD_COUNT * HEAD_WIDTH], &current_key)
                            .view(),
                        Owned::f32(
                            &[BATCH, 1, KEY_VALUE_HEAD_COUNT * HEAD_WIDTH],
                            &current_value,
                        )
                        .view(),
                        past_key.view(),
                        past_value.view(),
                        Owned::i32(&[BATCH], &past_lengths).view(),
                        Owned::i32(&[], &[present_sequence_length as i32]).view(),
                    ],
                    &mut [
                        lazy_output.view_mut(),
                        lazy_present_key.view_mut(),
                        lazy_present_value.view_mut(),
                    ],
                )
                .unwrap();

            let mut full_output = Owned::zeros_f32(&[BATCH, 1, QUERY_HEAD_COUNT * HEAD_WIDTH]);
            let mut full_present_key = Owned::zeros(
                DataType::Float16,
                &[
                    BATCH,
                    KEY_VALUE_HEAD_COUNT,
                    present_sequence_length,
                    HEAD_WIDTH,
                ],
            );
            let mut full_present_value = Owned::zeros(
                DataType::Float16,
                &[
                    BATCH,
                    KEY_VALUE_HEAD_COUNT,
                    present_sequence_length,
                    HEAD_WIDTH,
                ],
            );
            // This is the old path: materialize every f16 cache element before
            // calling the same production attention implementation.
            kernel
                .execute(
                    &[
                        Owned::f32(&[BATCH, 1, QUERY_HEAD_COUNT * HEAD_WIDTH], &query).view(),
                        Owned::f32(&[BATCH, 1, KEY_VALUE_HEAD_COUNT * HEAD_WIDTH], &current_key)
                            .view(),
                        Owned::f32(
                            &[BATCH, 1, KEY_VALUE_HEAD_COUNT * HEAD_WIDTH],
                            &current_value,
                        )
                        .view(),
                        Owned::f32(&past_shape, &full_key).view(),
                        Owned::f32(&past_shape, &full_value).view(),
                        Owned::i32(&[BATCH], &past_lengths).view(),
                        Owned::i32(&[], &[present_sequence_length as i32]).view(),
                    ],
                    &mut [
                        full_output.view_mut(),
                        full_present_key.view_mut(),
                        full_present_value.view_mut(),
                    ],
                )
                .unwrap();

            assert_f32_bits_eq(
                &lazy_output.to_f32(),
                &full_output.to_f32(),
                "attention output",
            );
            assert_eq!(
                lazy_present_key.to_u16_bits(),
                full_present_key.to_u16_bits(),
                "present key"
            );
            assert_eq!(
                lazy_present_value.to_u16_bits(),
                full_present_value.to_u16_bits(),
                "present value"
            );

            // Carry the production f16 cache forward. Exact equality above makes
            // an omitted write in the !has_tail uninitialized-buffer path fail
            // before it can be hidden by a later f16 round-trip.
            past_key_bits = lazy_present_key.to_u16_bits();
            past_value_bits = lazy_present_value.to_u16_bits();
        }
    }

    /// Independently verifies the `!has_tail` (uninitialized `Vec::set_len`)
    /// present fast path with a NONZERO past — the case Chew flagged as
    /// unverified in `lazy_widen_bit_identical_to_full_widen_multistep` (whose
    /// only no-tail iteration is step zero, where the past is empty and both
    /// sides share the production present-construction).
    ///
    /// Every batch has `past_len(3) + current(1) == present_sequence_length(4)
    /// == total`, so `has_tail == false` and `present_k`/`present_v` are built
    /// via `Vec::with_capacity` + `set_len` with NO zero-fill, then the past
    /// prefix is materialized by `widen_run` into never-pre-initialized memory
    /// AND `past_len > 0`. `head_dim == 7` is not a multiple of 8, so the F16C
    /// widen tail path runs; `q_heads(4) > kv_heads(2)` exercises GQA group
    /// broadcast.
    ///
    /// The expected present is assembled BY HAND from the known past f16 bits +
    /// the current-step K/V in BNSH order — it does NOT route through the
    /// production present-construction path, so it cannot share an offset,
    /// skipped-row, or read-before-write bug with the fast path. A wrong
    /// destination offset, a missing row, or an uninitialized element in the
    /// `set_len` fast path makes the bit-exact assertion FAIL rather than being
    /// masked (both sides corrupting identically), which is exactly the gap the
    /// full-widen parity test could not close.
    #[test]
    fn no_tail_with_past_present_independently_bit_exact() {
        const BATCH: usize = 2;
        const QUERY_HEAD_COUNT: usize = 4; // GQA group broadcast: q_heads > kv_heads
        const KEY_VALUE_HEAD_COUNT: usize = 2; // multiple kv-heads
        const HEAD_WIDTH: usize = 7; // NOT a multiple of 8 -> F16C widen tail path
        const PAST_LEN: usize = 3; // nonzero past K/V
        const CURRENT_LEN: usize = 1; // decode step
        const PRESENT_LEN: usize = PAST_LEN + CURRENT_LEN; // == total for every batch

        let kernel =
            gqa_kernel_with_heads(QUERY_HEAD_COUNT as i64, KEY_VALUE_HEAD_COUNT as i64, &[]);

        // Past cache as raw f16 bit patterns (valid halves via from_f32).
        let past_key_bits: Vec<u16> = (0..BATCH * KEY_VALUE_HEAD_COUNT * PAST_LEN * HEAD_WIDTH)
            .map(|index| half::f16::from_f32(mixed_scale_value(index, 0xA1A1)).to_bits())
            .collect();
        let past_value_bits: Vec<u16> = (0..BATCH * KEY_VALUE_HEAD_COUNT * PAST_LEN * HEAD_WIDTH)
            .map(|index| half::f16::from_f32(mixed_scale_value(index, 0xB2B2)).to_bits())
            .collect();

        let query: Vec<f32> = (0..BATCH * QUERY_HEAD_COUNT * HEAD_WIDTH)
            .map(|index| mixed_scale_value(index, 0xC3C3))
            .collect();
        let current_key: Vec<f32> = (0..BATCH * KEY_VALUE_HEAD_COUNT * HEAD_WIDTH)
            .map(|index| mixed_scale_value(index, 0xD4D4))
            .collect();
        let current_value: Vec<f32> = (0..BATCH * KEY_VALUE_HEAD_COUNT * HEAD_WIDTH)
            .map(|index| mixed_scale_value(index, 0xE5E5))
            .collect();

        // seqlens_k = total - 1. With current seq == 1, total == past_len + 1,
        // so seqlens_k == past_len; total_sequence_length == max(seqlens_k) + 1
        // == PRESENT_LEN for every batch => has_tail == false.
        let seqlens_k = [PAST_LEN as i32; BATCH];
        let total_sequence_length = PRESENT_LEN as i32;

        let past_shape = [BATCH, KEY_VALUE_HEAD_COUNT, PAST_LEN, HEAD_WIDTH];
        let present_shape = [BATCH, KEY_VALUE_HEAD_COUNT, PRESENT_LEN, HEAD_WIDTH];

        let mut lazy_output =
            Owned::zeros_f32(&[BATCH, CURRENT_LEN, QUERY_HEAD_COUNT * HEAD_WIDTH]);
        let mut lazy_present_key = Owned::zeros(DataType::Float16, &present_shape);
        let mut lazy_present_value = Owned::zeros(DataType::Float16, &present_shape);

        kernel
            .execute(
                &[
                    Owned::f32(&[BATCH, CURRENT_LEN, QUERY_HEAD_COUNT * HEAD_WIDTH], &query).view(),
                    Owned::f32(
                        &[BATCH, CURRENT_LEN, KEY_VALUE_HEAD_COUNT * HEAD_WIDTH],
                        &current_key,
                    )
                    .view(),
                    Owned::f32(
                        &[BATCH, CURRENT_LEN, KEY_VALUE_HEAD_COUNT * HEAD_WIDTH],
                        &current_value,
                    )
                    .view(),
                    Owned::f16_bits(&past_shape, &past_key_bits).view(),
                    Owned::f16_bits(&past_shape, &past_value_bits).view(),
                    Owned::i32(&[BATCH], &seqlens_k).view(),
                    Owned::i32(&[], &[total_sequence_length]).view(),
                ],
                &mut [
                    lazy_output.view_mut(),
                    lazy_present_key.view_mut(),
                    lazy_present_value.view_mut(),
                ],
            )
            .unwrap();

        // ── Independently assemble the expected present in BNSH order ──
        // For each (batch, kv_head): rows [0, PAST_LEN) are the past f16 bits
        // verbatim (widen f16->f32 then narrow f32->f16 is lossless), and row
        // PAST_LEN is the current-step value narrowed to f16. This mirror is
        // written WITHOUT the production fast path, so it cannot share an
        // offset/skip/read-before-write bug with the code under test.
        let build_expected_present = |past_bits: &[u16], current: &[f32]| -> Vec<u16> {
            let mut expected = vec![0u16; BATCH * KEY_VALUE_HEAD_COUNT * PRESENT_LEN * HEAD_WIDTH];
            for batch_index in 0..BATCH {
                for kv_head_index in 0..KEY_VALUE_HEAD_COUNT {
                    let head = batch_index * KEY_VALUE_HEAD_COUNT + kv_head_index;
                    // Past prefix rows, copied bit-for-bit.
                    for sequence_index in 0..PAST_LEN {
                        for dimension_index in 0..HEAD_WIDTH {
                            let destination = (head * PRESENT_LEN + sequence_index) * HEAD_WIDTH
                                + dimension_index;
                            let source =
                                (head * PAST_LEN + sequence_index) * HEAD_WIDTH + dimension_index;
                            expected[destination] = past_bits[source];
                        }
                    }
                    // Current decode row, narrowed to f16.
                    for dimension_index in 0..HEAD_WIDTH {
                        let destination =
                            (head * PRESENT_LEN + PAST_LEN) * HEAD_WIDTH + dimension_index;
                        let source = head * HEAD_WIDTH + dimension_index;
                        expected[destination] = half::f16::from_f32(current[source]).to_bits();
                    }
                }
            }
            expected
        };
        let expected_present_key = build_expected_present(&past_key_bits, &current_key);
        let expected_present_value = build_expected_present(&past_value_bits, &current_value);

        // Bit-exact: guards the uninitialized `set_len` fast path against
        // read-before-write / wrong-offset bugs (Chew's reject on 8638ec6).
        assert_eq!(
            lazy_present_key.to_u16_bits(),
            expected_present_key,
            "no-tail present key must match the hand-assembled BNSH cache"
        );
        assert_eq!(
            lazy_present_value.to_u16_bits(),
            expected_present_value,
            "no-tail present value must match the hand-assembled BNSH cache"
        );

        // ── Attention output vs the old full-widen reference (kept per Chew) ──
        // Feeding the already-widened f16 past as dense f32 gives the pre-eedbf93
        // decode path; attention inputs are bit-identical, so the output must be
        // bit-identical too.
        let full_key = old_full_widen_f16(&past_key_bits);
        let full_value = old_full_widen_f16(&past_value_bits);
        let mut full_output =
            Owned::zeros_f32(&[BATCH, CURRENT_LEN, QUERY_HEAD_COUNT * HEAD_WIDTH]);
        let mut full_present_key = Owned::zeros(DataType::Float16, &present_shape);
        let mut full_present_value = Owned::zeros(DataType::Float16, &present_shape);
        kernel
            .execute(
                &[
                    Owned::f32(&[BATCH, CURRENT_LEN, QUERY_HEAD_COUNT * HEAD_WIDTH], &query).view(),
                    Owned::f32(
                        &[BATCH, CURRENT_LEN, KEY_VALUE_HEAD_COUNT * HEAD_WIDTH],
                        &current_key,
                    )
                    .view(),
                    Owned::f32(
                        &[BATCH, CURRENT_LEN, KEY_VALUE_HEAD_COUNT * HEAD_WIDTH],
                        &current_value,
                    )
                    .view(),
                    Owned::f32(&past_shape, &full_key).view(),
                    Owned::f32(&past_shape, &full_value).view(),
                    Owned::i32(&[BATCH], &seqlens_k).view(),
                    Owned::i32(&[], &[total_sequence_length]).view(),
                ],
                &mut [
                    full_output.view_mut(),
                    full_present_key.view_mut(),
                    full_present_value.view_mut(),
                ],
            )
            .unwrap();

        assert_f32_bits_eq(
            &lazy_output.to_f32(),
            &full_output.to_f32(),
            "no-tail attention output",
        );
    }

    #[test]
    fn decode_batch_ragged_past_lengths_materialize_independently() {
        let q = vec![
            1., 0., 1., 0., 0., 1., 0., 1., 1., 1., 1., -1., -1., 1., -1., -1.,
        ];
        let past_k = vec![
            1., 0., 91., 92., 93., 94., 0., 1., 95., 96., 97., 98., 2., 0., 3., 0., 4., 0., 5., 0.,
            6., 0., 7., 0.,
        ];
        let past_v = vec![
            1., 2., 71., 72., 73., 74., 3., 4., 75., 76., 77., 78., 10., 20., 30., 40., 50., 60.,
            70., 80., 90., 100., 110., 120.,
        ];
        let cur_k = vec![1., 1., 10., 10., 8., 0., 9., 0.];
        let cur_v = vec![5., 6., 50., 60., 130., 140., 150., 160.];
        let expected_k = vec![
            1., 0., 1., 1., 0., 0., 0., 0., 0., 1., 10., 10., 0., 0., 0., 0., 2., 0., 3., 0., 4.,
            0., 8., 0., 5., 0., 6., 0., 7., 0., 9., 0.,
        ];
        let expected_v = vec![
            1., 2., 5., 6., 0., 0., 0., 0., 3., 4., 50., 60., 0., 0., 0., 0., 10., 20., 30., 40.,
            50., 60., 130., 140., 70., 80., 90., 100., 110., 120., 150., 160.,
        ];
        let mut out = Owned::zeros_f32(&[2, 1, 8]);
        let mut pk = Owned::zeros_f32(&[2, 2, 4, 2]);
        let mut pv = Owned::zeros_f32(&[2, 2, 4, 2]);
        gqa_kernel(&[])
            .execute(
                &[
                    Owned::f32(&[2, 1, 8], &q).view(),
                    Owned::f32(&[2, 1, 4], &cur_k).view(),
                    Owned::f32(&[2, 1, 4], &cur_v).view(),
                    Owned::f32(&[2, 2, 3, 2], &past_k).view(),
                    Owned::f32(&[2, 2, 3, 2], &past_v).view(),
                    Owned::i32(&[2], &[1, 3]).view(),
                    Owned::i32(&[], &[4]).view(),
                ],
                &mut [out.view_mut(), pk.view_mut(), pv.view_mut()],
            )
            .unwrap();
        close(&pk.to_f32(), &expected_k);
        close(&pv.to_f32(), &expected_v);
        let mut want = reference(
            &q[..8],
            &[1., 0., 1., 1., 0., 1., 10., 10.],
            &[1., 2., 5., 6., 3., 4., 50., 60.],
            1,
            2,
            1,
        );
        want.extend(reference(
            &q[8..],
            &expected_k[16..],
            &expected_v[16..],
            1,
            4,
            3,
        ));
        close(&out.to_f32(), &want);
    }

    #[test]
    fn decode_preserves_fixed_cache_capacity_and_appends_at_logical_length() {
        let q = vec![1., 0., 1., 0., 0., 1., 0., 1.];
        let past_k = vec![
            1., 0., 0., 1., 91., 92., 93., 94., 95., 96., 10., 0., 0., 10., 81., 82., 83., 84.,
            85., 86.,
        ];
        let past_v = vec![
            1., 2., 3., 4., 71., 72., 73., 74., 75., 76., 10., 20., 30., 40., 61., 62., 63., 64.,
            65., 66.,
        ];
        let cur_k = vec![1., 1., 10., 10.];
        let cur_v = vec![5., 6., 50., 60.];
        let expected_k = vec![
            1., 0., 0., 1., 1., 1., 0., 0., 0., 0., 10., 0., 0., 10., 10., 10., 0., 0., 0., 0.,
        ];
        let expected_v = vec![
            1., 2., 3., 4., 5., 6., 0., 0., 0., 0., 10., 20., 30., 40., 50., 60., 0., 0., 0., 0.,
        ];
        let mut out = Owned::zeros_f32(&[1, 1, 8]);
        let mut pk = Owned::zeros_f32(&[1, 2, 5, 2]);
        let mut pv = Owned::zeros_f32(&[1, 2, 5, 2]);
        gqa_kernel(&[])
            .execute(
                &[
                    Owned::f32(&[1, 1, 8], &q).view(),
                    Owned::f32(&[1, 1, 4], &cur_k).view(),
                    Owned::f32(&[1, 1, 4], &cur_v).view(),
                    Owned::f32(&[1, 2, 5, 2], &past_k).view(),
                    Owned::f32(&[1, 2, 5, 2], &past_v).view(),
                    Owned::i32(&[1], &[2]).view(),
                    Owned::i32(&[], &[3]).view(),
                ],
                &mut [out.view_mut(), pk.view_mut(), pv.view_mut()],
            )
            .unwrap();
        assert_eq!(pk.shape, vec![1, 2, 5, 2]);
        assert_eq!(pv.shape, vec![1, 2, 5, 2]);
        close(&pk.to_f32(), &expected_k);
        close(&pv.to_f32(), &expected_v);
        close(
            &out.to_f32(),
            &reference(&q, &expected_k, &expected_v, 1, 5, 2),
        );
    }

    #[test]
    fn rotary_path_matches_rotated_reference() {
        let q = vec![1., 2., 3., 4., 5., 6., 7., 8.];
        let k = vec![1., 2., 3., 4.];
        let v = vec![1., 2., 3., 4.];
        let cos = vec![0.0];
        let sin = vec![1.0];
        let q_rot = vec![-2., 1., -4., 3., -6., 5., -8., 7.];
        let k_rot_bsh = vec![-2., 1., -4., 3.];
        let k_rot_bnsh = vec![-2., 1., -4., 3.];
        let mut out = Owned::zeros_f32(&[1, 1, 8]);
        gqa_kernel(&[("do_rotary", Attribute::Int(1))])
            .execute(
                &[
                    Owned::f32(&[1, 1, 8], &q).view(),
                    Owned::f32(&[1, 1, 4], &k).view(),
                    Owned::f32(&[1, 1, 4], &v).view(),
                    absent(),
                    absent(),
                    Owned::i32(&[1], &[0]).view(),
                    Owned::i32(&[], &[1]).view(),
                    Owned::f32(&[1, 1], &cos).view(),
                    Owned::f32(&[1, 1], &sin).view(),
                ],
                &mut [out.view_mut()],
            )
            .unwrap();
        let _ = k_rot_bsh;
        close(&out.to_f32(), &reference(&q_rot, &k_rot_bnsh, &v, 1, 1, 0));
    }

    #[test]
    fn rotary_explicit_position_ids_apply_to_query_and_key() {
        let q = vec![
            1., 2., 2., -1., -1., 3., 4., 2., 3., -2., 1., 4., -3., 1., 2., 5.,
        ];
        let k = vec![2., 1., -1., 3., 4., -2., 2., 5.];
        let v = vec![1., 2., 10., 20., 3., 4., 30., 40.];
        let angles = [0.0_f32, 0.2, 0.7, 1.1, 1.6];
        let cos: Vec<f32> = angles.iter().map(|angle| angle.cos()).collect();
        let sin: Vec<f32> = angles.iter().map(|angle| angle.sin()).collect();
        let positions = [2_usize, 4];
        let q_rot = reference_rope_bsh(&q, 2, 4, &positions, &cos, &sin);
        let k_rot_bsh = reference_rope_bsh(&k, 2, 2, &positions, &cos, &sin);
        let k_rot_bnsh = bsh_to_bnsh(&k_rot_bsh, 2, 2);
        let v_bnsh = bsh_to_bnsh(&v, 2, 2);
        let mut out = Owned::zeros_f32(&[1, 2, 8]);
        let mut present_k = Owned::zeros_f32(&[1, 2, 2, 2]);
        gqa_kernel(&[("do_rotary", Attribute::Int(1))])
            .execute(
                &[
                    Owned::f32(&[1, 2, 8], &q).view(),
                    Owned::f32(&[1, 2, 4], &k).view(),
                    Owned::f32(&[1, 2, 4], &v).view(),
                    absent(),
                    absent(),
                    Owned::i32(&[1], &[1]).view(),
                    Owned::i32(&[], &[2]).view(),
                    Owned::f32(&[5, 1], &cos).view(),
                    Owned::f32(&[5, 1], &sin).view(),
                    Owned::i64(&[1, 2], &[2, 4]).view(),
                ],
                &mut [out.view_mut(), present_k.view_mut()],
            )
            .unwrap();
        close(&present_k.to_f32(), &k_rot_bnsh);
        close(
            &out.to_f32(),
            &reference(&q_rot, &k_rot_bnsh, &v_bnsh, 2, 2, 0),
        );
    }

    #[test]
    fn widen_rotary_prefix_bounds_widen_to_row_prefix() {
        // A cache far larger than the addressed prefix: only the first `rows`
        // rows may be widened, and trailing rows (poisoned with NaN) must never
        // be touched. `half_dim = 3` is not a multiple of 8, exercising the
        // F16C scalar tail in the widen path.
        let half_dim = 3usize;
        let cache_rows = 40usize;
        let rows = 4usize;
        let mut data = vec![0.0f32; cache_rows * half_dim];
        for (i, slot) in data.iter_mut().enumerate() {
            *slot = (i as f32) * 0.25 - 3.0;
        }

        for slot in data.iter_mut().skip(rows * half_dim) {
            *slot = f32::NAN;
        }
        let cache = Owned::f16(&[cache_rows, half_dim], &data);
        let prefix = super::widen_rotary_prefix("test", &cache.view(), rows, half_dim).unwrap();
        assert_eq!(prefix.len(), rows * half_dim);
        for k in 0..rows * half_dim {
            let expected = half::f16::from_f32(data[k]).to_f32();
            assert_eq!(prefix[k], expected, "prefix element {k}");
        }
        assert!(
            prefix.iter().all(|v| v.is_finite()),
            "poisoned tail rows leaked into the widened prefix"
        );
    }

    #[test]
    fn widen_rotary_prefix_bounds_bf16_to_row_prefix() {
        let half_dim = 3usize;
        let cache_rows = 40usize;
        let rows = 4usize;
        let mut data = vec![0.0f32; cache_rows * half_dim];
        for (i, slot) in data.iter_mut().enumerate() {
            *slot = (i as f32) * 0.25 - 3.0;
        }
        for slot in data.iter_mut().skip(rows * half_dim) {
            *slot = f32::NAN;
        }
        let cache = Owned::bf16(&[cache_rows, half_dim], &data);
        let prefix = super::widen_rotary_prefix("test", &cache.view(), rows, half_dim).unwrap();
        assert_eq!(prefix.len(), rows * half_dim);
        for k in 0..rows * half_dim {
            let expected = half::bf16::from_f32(data[k]).to_f32();
            assert_eq!(prefix[k], expected, "prefix element {k}");
        }
        assert!(
            prefix.iter().all(|v| v.is_finite()),
            "poisoned tail rows leaked into the widened prefix"
        );
    }

    #[test]
    fn rotary_oversized_cache_only_reads_addressed_prefix() {
        // Identical setup to `rotary_explicit_position_ids_apply_to_query_and_key`
        // but with a 4096-row rotary cache whose rows past the max addressed
        // position (4) are NaN. The prefix-bounded widen must ignore them and
        // reproduce the exact-size cache result bit-for-bit (parity lock for the
        // widen-placement optimization).
        let q = vec![
            1., 2., 2., -1., -1., 3., 4., 2., 3., -2., 1., 4., -3., 1., 2., 5.,
        ];
        let k = vec![2., 1., -1., 3., 4., -2., 2., 5.];
        let v = vec![1., 2., 10., 20., 3., 4., 30., 40.];
        let angles = [0.0_f32, 0.2, 0.7, 1.1, 1.6];
        let cos: Vec<f32> = angles.iter().map(|angle| angle.cos()).collect();
        let sin: Vec<f32> = angles.iter().map(|angle| angle.sin()).collect();
        let positions = [2_usize, 4];
        let q_rot = reference_rope_bsh(&q, 2, 4, &positions, &cos, &sin);
        let k_rot_bsh = reference_rope_bsh(&k, 2, 2, &positions, &cos, &sin);
        let k_rot_bnsh = bsh_to_bnsh(&k_rot_bsh, 2, 2);
        let v_bnsh = bsh_to_bnsh(&v, 2, 2);
        let big_rows = 4096usize;
        let mut cos_big = vec![f32::NAN; big_rows];
        let mut sin_big = vec![f32::NAN; big_rows];
        cos_big[..cos.len()].copy_from_slice(&cos);
        sin_big[..sin.len()].copy_from_slice(&sin);
        let mut out = Owned::zeros_f32(&[1, 2, 8]);
        let mut present_k = Owned::zeros_f32(&[1, 2, 2, 2]);
        gqa_kernel(&[("do_rotary", Attribute::Int(1))])
            .execute(
                &[
                    Owned::f32(&[1, 2, 8], &q).view(),
                    Owned::f32(&[1, 2, 4], &k).view(),
                    Owned::f32(&[1, 2, 4], &v).view(),
                    absent(),
                    absent(),
                    Owned::i32(&[1], &[1]).view(),
                    Owned::i32(&[], &[2]).view(),
                    Owned::f32(&[big_rows, 1], &cos_big).view(),
                    Owned::f32(&[big_rows, 1], &sin_big).view(),
                    Owned::i64(&[1, 2], &[2, 4]).view(),
                ],
                &mut [out.view_mut(), present_k.view_mut()],
            )
            .unwrap();
        close(&present_k.to_f32(), &k_rot_bnsh);
        close(
            &out.to_f32(),
            &reference(&q_rot, &k_rot_bnsh, &v_bnsh, 2, 2, 0),
        );
    }

    #[test]
    fn local_window_masks_older_cache_tokens() {
        let q = [0.0; 8];
        let past_k = [0.0; 8];
        let past_v = [1., 1., 2., 2., 10., 10., 20., 20.];
        let cur_k = [0.0; 4];
        let cur_v = [9., 9., 90., 90.];
        let mut out = Owned::zeros_f32(&[1, 1, 8]);
        gqa_kernel(&[("local_window_size", Attribute::Int(1))])
            .execute(
                &[
                    Owned::f32(&[1, 1, 8], &q).view(),
                    Owned::f32(&[1, 1, 4], &cur_k).view(),
                    Owned::f32(&[1, 1, 4], &cur_v).view(),
                    Owned::f32(&[1, 2, 2, 2], &past_k).view(),
                    Owned::f32(&[1, 2, 2, 2], &past_v).view(),
                    Owned::i32(&[1], &[2]).view(),
                    Owned::i32(&[], &[3]).view(),
                ],
                &mut [out.view_mut()],
            )
            .unwrap();
        close(&out.to_f32(), &[9., 9., 9., 9., 90., 90., 90., 90.]);
    }

    #[test]
    fn softcap_matches_independent_score_transform() {
        let q = [
            2., 0., 2., 0., 2., 0., 2., 0., 2., 0., 2., 0., 2., 0., 2., 0.,
        ];
        let k = [1., 0., 1., 0., 4., 0., 4., 0.];
        let v = [1., 0., 10., 0., 3., 0., 30., 0.];
        let mut out = Owned::zeros_f32(&[1, 2, 8]);
        gqa_kernel(&[("softcap", Attribute::Float(1.5))])
            .execute(
                &[
                    Owned::f32(&[1, 2, 8], &q).view(),
                    Owned::f32(&[1, 2, 4], &k).view(),
                    Owned::f32(&[1, 2, 4], &v).view(),
                    absent(),
                    absent(),
                    Owned::i32(&[1], &[1]).view(),
                    Owned::i32(&[], &[2]).view(),
                ],
                &mut [out.view_mut()],
            )
            .unwrap();
        let s0 = 1.5 * ((2.0 / 2.0_f32.sqrt()) / 1.5_f32).tanh();
        let s1 = 1.5 * ((8.0 / 2.0_f32.sqrt()) / 1.5_f32).tanh();
        let p1 = (s1 - s0).exp() / (1.0 + (s1 - s0).exp());
        let expected_second = 1.0 * (1.0 - p1) + 3.0 * p1;
        let expected = [
            1.,
            0.,
            1.,
            0.,
            10.,
            0.,
            10.,
            0.,
            expected_second,
            0.,
            expected_second,
            0.,
            expected_second * 10.0,
            0.,
            expected_second * 10.0,
            0.,
        ];
        close(&out.to_f32(), &expected);
    }

    #[test]
    fn explicit_zero_scale_matches_default_scale() {
        let q = [
            1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0.,
        ];
        let k = [0., 0., 0., 0., 4., 0., 4., 0.];
        let v = [1., 0., 1., 0., 9., 0., 9., 0.];
        let run = |attrs: &[(&str, Attribute)]| {
            let mut out = Owned::zeros_f32(&[1, 2, 8]);
            gqa_kernel(attrs)
                .execute(
                    &[
                        Owned::f32(&[1, 2, 8], &q).view(),
                        Owned::f32(&[1, 2, 4], &k).view(),
                        Owned::f32(&[1, 2, 4], &v).view(),
                        absent(),
                        absent(),
                        Owned::i32(&[1], &[1]).view(),
                        Owned::i32(&[], &[2]).view(),
                    ],
                    &mut [out.view_mut()],
                )
                .unwrap();
            out.to_f32()
        };
        let default = run(&[]);
        let zero = run(&[("scale", Attribute::Float(0.0))]);
        close(&zero, &default);
        assert!(zero[8] > 8.0, "zero scale produced uniform attention");
    }

    // ── New tests covering the vectorized decode hot path ──────────────────

    /// Verifies realistic-width M=1 decode against a scalar full-attention
    /// implementation. The pseudo-random mixed-scale inputs are non-periodic
    /// over the fixture and produce cancellation in the 128-element dot products.
    ///
    /// The tolerance covers both dot-product reordering and hundreds of fused
    /// probability-weighted value accumulations. It validates the runtime
    /// dispatch path, including the scalar fallback on hosts without AVX2+FMA.
    #[test]
    fn gqa_decode_long_context_matches_reference() {
        const PAST_SEQUENCE_LENGTH: usize = 255;
        const TOTAL_SEQUENCE_LENGTH: usize = PAST_SEQUENCE_LENGTH + 1;
        const QUERY_HEAD_COUNT: usize = 4;
        const KEY_VALUE_HEAD_COUNT: usize = 2;
        const HEAD_WIDTH: usize = 128;

        let query: Vec<f32> = (0..QUERY_HEAD_COUNT * HEAD_WIDTH)
            .map(|index| mixed_scale_value(index, 0x1234))
            .collect();
        let current_key: Vec<f32> = (0..KEY_VALUE_HEAD_COUNT * HEAD_WIDTH)
            .map(|index| mixed_scale_value(index, 0x5678))
            .collect();
        let current_value: Vec<f32> = (0..KEY_VALUE_HEAD_COUNT * HEAD_WIDTH)
            .map(|index| mixed_scale_value(index, 0x9abc))
            .collect();
        let past_key: Vec<f32> = (0..KEY_VALUE_HEAD_COUNT * PAST_SEQUENCE_LENGTH * HEAD_WIDTH)
            .map(|index| mixed_scale_value(index, 0xdef0))
            .collect();
        let past_value: Vec<f32> = (0..KEY_VALUE_HEAD_COUNT * PAST_SEQUENCE_LENGTH * HEAD_WIDTH)
            .map(|index| mixed_scale_value(index, 0x2468))
            .collect();

        let mut full_key = vec![0.0f32; KEY_VALUE_HEAD_COUNT * TOTAL_SEQUENCE_LENGTH * HEAD_WIDTH];
        let mut full_value =
            vec![0.0f32; KEY_VALUE_HEAD_COUNT * TOTAL_SEQUENCE_LENGTH * HEAD_WIDTH];
        for head_index in 0..KEY_VALUE_HEAD_COUNT {
            let past_base = head_index * PAST_SEQUENCE_LENGTH * HEAD_WIDTH;
            let full_base = head_index * TOTAL_SEQUENCE_LENGTH * HEAD_WIDTH;
            full_key[full_base..full_base + PAST_SEQUENCE_LENGTH * HEAD_WIDTH].copy_from_slice(
                &past_key[past_base..past_base + PAST_SEQUENCE_LENGTH * HEAD_WIDTH],
            );
            full_value[full_base..full_base + PAST_SEQUENCE_LENGTH * HEAD_WIDTH].copy_from_slice(
                &past_value[past_base..past_base + PAST_SEQUENCE_LENGTH * HEAD_WIDTH],
            );
            for dimension_index in 0..HEAD_WIDTH {
                full_key[full_base + PAST_SEQUENCE_LENGTH * HEAD_WIDTH + dimension_index] =
                    current_key[head_index * HEAD_WIDTH + dimension_index];
                full_value[full_base + PAST_SEQUENCE_LENGTH * HEAD_WIDTH + dimension_index] =
                    current_value[head_index * HEAD_WIDTH + dimension_index];
            }
        }

        let expected = reference_with_geometry(
            &query,
            &full_key,
            &full_value,
            1,
            TOTAL_SEQUENCE_LENGTH,
            PAST_SEQUENCE_LENGTH,
            QUERY_HEAD_COUNT,
            KEY_VALUE_HEAD_COUNT,
            HEAD_WIDTH,
        );

        let mut output = Owned::zeros_f32(&[1, 1, QUERY_HEAD_COUNT * HEAD_WIDTH]);
        let mut present_key =
            Owned::zeros_f32(&[1, KEY_VALUE_HEAD_COUNT, TOTAL_SEQUENCE_LENGTH, HEAD_WIDTH]);
        let mut present_value =
            Owned::zeros_f32(&[1, KEY_VALUE_HEAD_COUNT, TOTAL_SEQUENCE_LENGTH, HEAD_WIDTH]);
        gqa_kernel(&[])
            .execute(
                &[
                    Owned::f32(&[1, 1, QUERY_HEAD_COUNT * HEAD_WIDTH], &query).view(),
                    Owned::f32(&[1, 1, KEY_VALUE_HEAD_COUNT * HEAD_WIDTH], &current_key).view(),
                    Owned::f32(&[1, 1, KEY_VALUE_HEAD_COUNT * HEAD_WIDTH], &current_value).view(),
                    Owned::f32(
                        &[1, KEY_VALUE_HEAD_COUNT, PAST_SEQUENCE_LENGTH, HEAD_WIDTH],
                        &past_key,
                    )
                    .view(),
                    Owned::f32(
                        &[1, KEY_VALUE_HEAD_COUNT, PAST_SEQUENCE_LENGTH, HEAD_WIDTH],
                        &past_value,
                    )
                    .view(),
                    Owned::i32(&[1], &[PAST_SEQUENCE_LENGTH as i32]).view(),
                    Owned::i32(&[], &[TOTAL_SEQUENCE_LENGTH as i32]).view(),
                ],
                &mut [
                    output.view_mut(),
                    present_key.view_mut(),
                    present_value.view_mut(),
                ],
            )
            .unwrap();

        for (index, (actual, expected)) in output.to_f32().iter().zip(&expected).enumerate() {
            let tolerance = 2.0e-5 + 2.0e-5 * expected.abs();
            assert!(
                (actual - expected).abs() <= tolerance,
                "attention output {index}: actual {actual}, expected {expected}, difference {}, tolerance {tolerance}",
                (actual - expected).abs()
            );
        }
    }

    /// The KV-group-fused decode path must be *bit-identical* to the
    /// per-query-head path and must actually be reached.
    ///
    /// The path is opt-in, so the test forces it on for its own duration. The
    /// A/B is then driven through the reachability gate: the same inputs run on
    /// a decode pool the fused schedule saturates (`batch * kv_num_heads >=
    /// workers`) and on one it does not. The first must take the fused branch —
    /// asserted against [`group_fused_count`], so the test cannot pass
    /// vacuously — and both must agree to the last bit.
    #[test]
    fn group_fused_decode_is_bit_identical_to_per_head_decode() {
        let _fusion = GroupFusionOverride::forced_on();
        // Long enough that `group_fusion_pays_for_traffic` opens the gate
        // (8 MiB of attended K+V per KV head at f32 width).
        const PAST_SEQUENCE_LENGTH: usize = 8_192;
        const TOTAL_SEQUENCE_LENGTH: usize = PAST_SEQUENCE_LENGTH + 1;
        const QUERY_HEAD_COUNT: usize = 4;
        const KEY_VALUE_HEAD_COUNT: usize = 2;
        const HEAD_WIDTH: usize = 128;

        let query: Vec<f32> = (0..QUERY_HEAD_COUNT * HEAD_WIDTH)
            .map(|index| mixed_scale_value(index, 0x7f01))
            .collect();
        let current_key: Vec<f32> = (0..KEY_VALUE_HEAD_COUNT * HEAD_WIDTH)
            .map(|index| mixed_scale_value(index, 0x7f02))
            .collect();
        let current_value: Vec<f32> = (0..KEY_VALUE_HEAD_COUNT * HEAD_WIDTH)
            .map(|index| mixed_scale_value(index, 0x7f03))
            .collect();
        let past_key: Vec<f32> = (0..KEY_VALUE_HEAD_COUNT * PAST_SEQUENCE_LENGTH * HEAD_WIDTH)
            .map(|index| mixed_scale_value(index, 0x7f04))
            .collect();
        let past_value: Vec<f32> = (0..KEY_VALUE_HEAD_COUNT * PAST_SEQUENCE_LENGTH * HEAD_WIDTH)
            .map(|index| mixed_scale_value(index, 0x7f05))
            .collect();

        // `local_window_size` and `softcap` are set so the fused path's masking
        // and softcap branches are covered too, not just the plain causal case.
        let run = |workers: usize| -> Vec<f32> {
            let mut output = Owned::zeros_f32(&[1, 1, QUERY_HEAD_COUNT * HEAD_WIDTH]);
            let mut present_key =
                Owned::zeros_f32(&[1, KEY_VALUE_HEAD_COUNT, TOTAL_SEQUENCE_LENGTH, HEAD_WIDTH]);
            let mut present_value =
                Owned::zeros_f32(&[1, KEY_VALUE_HEAD_COUNT, TOTAL_SEQUENCE_LENGTH, HEAD_WIDTH]);
            let kernel_attrs = [
                ("local_window_size", Attribute::Int(8_192)),
                ("softcap", Attribute::Float(30.0)),
            ];
            rayon::ThreadPoolBuilder::new()
                .num_threads(workers)
                .build()
                .expect("test decode pool")
                .install(|| {
                    let kernel = gqa_kernel_with_heads(
                        QUERY_HEAD_COUNT as i64,
                        KEY_VALUE_HEAD_COUNT as i64,
                        &kernel_attrs,
                    );
                    kernel
                        .execute(
                            &[
                                Owned::f32(&[1, 1, QUERY_HEAD_COUNT * HEAD_WIDTH], &query).view(),
                                Owned::f32(
                                    &[1, 1, KEY_VALUE_HEAD_COUNT * HEAD_WIDTH],
                                    &current_key,
                                )
                                .view(),
                                Owned::f32(
                                    &[1, 1, KEY_VALUE_HEAD_COUNT * HEAD_WIDTH],
                                    &current_value,
                                )
                                .view(),
                                Owned::f32(
                                    &[1, KEY_VALUE_HEAD_COUNT, PAST_SEQUENCE_LENGTH, HEAD_WIDTH],
                                    &past_key,
                                )
                                .view(),
                                Owned::f32(
                                    &[1, KEY_VALUE_HEAD_COUNT, PAST_SEQUENCE_LENGTH, HEAD_WIDTH],
                                    &past_value,
                                )
                                .view(),
                                Owned::i32(&[1], &[PAST_SEQUENCE_LENGTH as i32]).view(),
                                Owned::i32(&[], &[TOTAL_SEQUENCE_LENGTH as i32]).view(),
                            ],
                            &mut [
                                output.view_mut(),
                                present_key.view_mut(),
                                present_value.view_mut(),
                            ],
                        )
                        .unwrap();
                });
            output.to_f32()
        };

        // `KEY_VALUE_HEAD_COUNT` fused tasks against 2 workers: gate open.
        let before = group_fused_count();
        let fused = run(2);
        assert!(
            group_fused_count() > before,
            "fused decode path was never reached — the A/B would be vacuous"
        );
        // Same tasks against more workers than the fused schedule can fill:
        // gate closed, per-query-head schedule.
        let per_head = run(KEY_VALUE_HEAD_COUNT * 2);

        assert_eq!(
            fused.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            per_head.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            "KV-group fusion must not change a single bit of the decode output"
        );
    }

    /// The fused schedule is calibrated on one host's LLC and has no
    /// model-level evidence for the batch >= 2 regime its gate admits, so it
    /// must stay opt-in. Guards against someone flipping the default back.
    #[test]
    fn group_fusion_is_opt_in_by_default() {
        if std::env::var("ONNX_GENAI_GQA_GROUP_FUSED").is_ok() {
            // The caller opted in for this process; nothing to assert.
            return;
        }
        let _serialise = GROUP_FUSION_TEST_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        assert!(
            !group_fusion_enabled(),
            "KV-group-fused decode must stay opt-in (ONNX_GENAI_GQA_GROUP_FUSED=1)"
        );
    }

    /// The fusion trades parallel tasks for KV traffic, so it must only engage
    /// while the collapsed schedule still covers every decode worker.
    #[test]
    fn group_fusion_gate_requires_a_saturated_decode_pool() {
        // Qwen2.5-0.5B (2 KV heads) and Qwen2.5-7B (4) starve an 8-wide pool.
        assert!(!group_fusion_saturates_pool(2, 8));
        assert!(!group_fusion_saturates_pool(4, 8));
        // Llama-3.1-8B / Qwen3-0.6B (8 KV heads) fill it exactly.
        assert!(group_fusion_saturates_pool(8, 8));
        // Batching restores saturation for a narrow-KV model.
        assert!(group_fusion_saturates_pool(4 * 2, 8));
        // A single-worker scope has no parallelism to lose.
        assert!(group_fusion_saturates_pool(2, 1));
    }

    /// Below the last-level-cache crossover the repeat KV reads the fusion
    /// removes are served from cache, so the fusion must stay off. The
    /// threshold is `llc / fused_tasks`, so it is stated here relative to the
    /// host's own reported cache rather than as a hard-coded byte count.
    #[test]
    fn group_fusion_gate_requires_out_of_cache_kv() {
        const TASKS: usize = 8;
        let _unpinned = FusedTrafficThresholdPin::bytes_per_head(0);
        let per_head = group_fused_min_kv_bytes(TASKS);
        // head_size 128 streams 1 KiB of K+V per attended token.
        let crossover_tokens = per_head / 1024;
        assert!(!group_fusion_pays_for_traffic(
            crossover_tokens - 1,
            128,
            128,
            TASKS
        ));
        assert!(group_fusion_pays_for_traffic(
            crossover_tokens,
            128,
            128,
            TASKS
        ));
        // head_size 64 needs twice the window for the same bytes.
        assert!(!group_fusion_pays_for_traffic(
            crossover_tokens,
            64,
            64,
            TASKS
        ));
        assert!(group_fusion_pays_for_traffic(
            2 * crossover_tokens,
            64,
            64,
            TASKS
        ));
        // Asymmetric K/V head widths count both.
        assert!(group_fusion_pays_for_traffic(
            crossover_tokens,
            128,
            192,
            TASKS
        ));
        // Degenerate inputs must not overflow into a spurious open gate.
        assert!(!group_fusion_pays_for_traffic(0, 128, 128, TASKS));
        assert!(group_fusion_pays_for_traffic(usize::MAX, 128, 128, TASKS));
        assert!(group_fusion_pays_for_traffic(usize::MAX, 128, 128, 0));
    }

    /// The topology term may only ever *tighten* the calibrated threshold. This
    /// is the safety property the whole change rests on: an unreadable cache,
    /// an absurd cache, a zero task count - none of them may produce a gate
    /// looser than the 8 MiB that was actually measured.
    #[test]
    fn topology_term_can_only_tighten_the_calibrated_threshold() {
        let _unpinned = FusedTrafficThresholdPin::bytes_per_head(0);
        for tasks in [0usize, 1, 2, 3, 8, 16, 64, 1024, usize::MAX] {
            let threshold = group_fused_min_kv_bytes(tasks);
            assert!(
                threshold >= GROUP_FUSED_CALIBRATED_MIN_KV_BYTES,
                "tasks={tasks} produced {threshold}, looser than the calibration"
            );
        }
        // Monotone in the task count: more concurrent heads never demands a
        // *larger* per-head working set.
        let mut previous = usize::MAX;
        for tasks in [1usize, 2, 4, 8, 16, 32] {
            let threshold = group_fused_min_kv_bytes(tasks);
            assert!(threshold <= previous, "threshold grew at tasks={tasks}");
            previous = threshold;
        }
    }

    /// Falsifier for the losing side. On a large last-level cache the fixed
    /// 8 MiB constant opened the gate while the aggregate KV still fit cache -
    /// the regime the kernel sweep measures at 0.64x - 0.9x. The topology term
    /// exists to keep it shut there, so pin a large-cache threshold and assert
    /// that a working set inside that cache is declined.
    #[test]
    fn a_large_last_level_cache_keeps_the_gate_shut_inside_cache() {
        // 256 MiB LLC / 8 fused tasks = 32 MiB per head.
        let _pin = FusedTrafficThresholdPin::bytes_per_head((256 << 20) / 8);
        // 16 MiB per head (16384 tokens at head_size 128) still fits that
        // cache in aggregate, so the fusion must decline even though it clears
        // the calibrated 8 MiB.
        assert!(!group_fusion_pays_for_traffic(16_384, 128, 128, 8));
        // 32 MiB per head no longer fits, so it opens.
        assert!(group_fusion_pays_for_traffic(32_768, 128, 128, 8));
    }

    /// The threshold has to be reproducible from the host's reported topology,
    /// not from whatever a benchmark happened to leave in the environment.
    #[test]
    fn last_level_cache_probe_is_sane_and_parses_sysfs_units() {
        let _unpinned = FusedTrafficThresholdPin::bytes_per_head(0);
        assert_eq!(parse_cache_size("32768K"), Some(32 << 20));
        assert_eq!(parse_cache_size("1024K"), Some(1 << 20));
        assert_eq!(parse_cache_size("8M"), Some(8 << 20));
        assert_eq!(parse_cache_size("1G"), Some(1 << 30));
        assert_eq!(parse_cache_size("4096"), Some(4096));
        assert_eq!(parse_cache_size(""), None);
        assert_eq!(parse_cache_size("K"), None);
        assert_eq!(parse_cache_size("12X"), None);
        assert_eq!(parse_cache_size("99999999999999999999K"), None);
        // Whatever the host reports, it must be a plausible cache size: at
        // least an L1 and no larger than a terabyte.
        let llc = last_level_cache_bytes();
        assert!(
            (16 << 10..=1 << 40).contains(&llc),
            "implausible last-level cache {llc}"
        );
    }

    /// A sliding-window model streams its window, not its cache, so the traffic
    /// gate must be fed the attended window — otherwise a long cache opens the
    /// gate for a short window that is well below the crossover.
    #[test]
    fn fused_traffic_gate_sees_the_local_window_not_the_cache_length() {
        // No local window: the whole cache is attended.
        assert_eq!(fused_attended_window(8_192, 0), 8_192);
        assert_eq!(fused_attended_window(8_192, -1), 8_192);
        // A short sliding window over a long cache caps the attended tokens...
        assert_eq!(fused_attended_window(32_768, 4_096), 4_096);
        // ...and that is what closes the gate. Pin the threshold so the
        // assertion does not depend on the runner's last-level cache.
        let _pin = FusedTrafficThresholdPin::bytes_per_head(8 << 20);
        assert!(group_fusion_pays_for_traffic(
            fused_attended_window(32_768, 0),
            128,
            128,
            8
        ));
        assert!(!group_fusion_pays_for_traffic(
            fused_attended_window(32_768, 4_096),
            128,
            128,
            8
        ));
        // A window longer than the cache cannot attend past the cache.
        assert_eq!(fused_attended_window(512, 4_096), 512);
    }

    #[test]
    fn decode_bf16_kv_state_matches_widened_f32_reference() {
        let q = vec![1., 0., 1., 0., 0., 1., 0., 1.];
        let past_k = vec![1., 0., 0., 1., 10., 0., 0., 10.];
        let past_v = vec![1., 2., 3., 4., 10., 20., 30., 40.];
        let cur_k = vec![1., 1., 10., 10.];
        let cur_v = vec![5., 6., 50., 60.];
        let q = Owned::bf16(&[1, 1, 8], &q);
        let cur_k = Owned::bf16(&[1, 1, 4], &cur_k);
        let cur_v = Owned::bf16(&[1, 1, 4], &cur_v);
        let past_k = Owned::bf16(&[1, 2, 2, 2], &past_k);
        let past_v = Owned::bf16(&[1, 2, 2, 2], &past_v);
        let mut out = Owned::zeros(DataType::BFloat16, &[1, 1, 8]);
        let mut present_k = Owned::zeros(DataType::BFloat16, &[1, 2, 3, 2]);
        let mut present_v = Owned::zeros(DataType::BFloat16, &[1, 2, 3, 2]);
        gqa_kernel(&[])
            .execute(
                &[
                    q.view(),
                    cur_k.view(),
                    cur_v.view(),
                    past_k.view(),
                    past_v.view(),
                    Owned::i32(&[1], &[2]).view(),
                    Owned::i32(&[], &[3]).view(),
                ],
                &mut [out.view_mut(), present_k.view_mut(), present_v.view_mut()],
            )
            .unwrap();
        let expected_k = vec![1., 0., 0., 1., 1., 1., 10., 0., 0., 10., 10., 10.];
        let expected_v = vec![1., 2., 3., 4., 5., 6., 10., 20., 30., 40., 50., 60.];
        let expected = reference(&q.to_bf16_as_f32(), &expected_k, &expected_v, 1, 3, 2);
        let expected: Vec<_> = expected
            .into_iter()
            .map(half::bf16::from_f32)
            .map(half::bf16::to_f32)
            .collect();
        assert_eq!(out.to_bf16_as_f32(), expected);
        assert_eq!(
            present_k.to_u16_bits(),
            expected_k
                .into_iter()
                .map(half::bf16::from_f32)
                .map(half::bf16::to_bits)
                .collect::<Vec<_>>()
        );
        assert_eq!(
            present_v.to_u16_bits(),
            expected_v
                .into_iter()
                .map(half::bf16::from_f32)
                .map(half::bf16::to_bits)
                .collect::<Vec<_>>()
        );
    }

    // ── In-place persistent KV (present==past device binding) ────────────────
    //
    // These tests exercise the append-only fast path that fires when the
    // executor aliases each `present` output onto its `past` input at full
    // physical capacity (the CPU analogue of the CUDA in-place KV cache). They
    // construct that aliasing directly with raw pointers — exactly as the
    // executor does under `run_with_device_bindings` — and prove byte-for-byte
    // parity with the ordinary copy path, correct fallback when NOT aliased, and
    // correctness across the prefill→decode boundary.

    use onnx_runtime_ep_api::{DevicePtr, DevicePtrMut, TensorMut};
    use onnx_runtime_ir::compute_contiguous_strides;

    const IP_NUM_HEADS: usize = 4;
    const IP_KV_HEADS: usize = 2;
    const IP_DIM: usize = 2;

    fn raw_gqa_kernel(local_window_size: i64, do_rotary: bool) -> GroupQueryAttentionKernel {
        GroupQueryAttentionKernel {
            num_heads: IP_NUM_HEADS,
            kv_num_heads: IP_KV_HEADS,
            scale: None,
            do_rotary,
            rotary_interleaved: false,
            local_window_size,
            softcap: 0.0,
        }
    }

    /// Build a `[1, KV, capacity, DIM]` BNSH capacity buffer, filling the first
    /// `past` sequence rows of each head from `past_data` (laid out
    /// `[1, KV, past, DIM]`) and every capacity row beyond `past` with `tail`.
    fn build_capacity_buffer(
        capacity: usize,
        past: usize,
        past_data: &[f32],
        tail: f32,
    ) -> Vec<f32> {
        let mut buf = vec![tail; IP_KV_HEADS * capacity * IP_DIM];
        for h in 0..IP_KV_HEADS {
            for s in 0..past {
                for x in 0..IP_DIM {
                    buf[(h * capacity + s) * IP_DIM + x] = past_data[(h * past + s) * IP_DIM + x];
                }
            }
        }
        buf
    }

    /// Extract the valid `[0, total)` sequence prefix of every head from a
    /// `[1, KV, capacity, DIM]` buffer, yielding a dense `[1, KV, total, DIM]`.
    fn head_prefix(buf: &[f32], capacity: usize, total: usize) -> Vec<f32> {
        let mut out = Vec::with_capacity(IP_KV_HEADS * total * IP_DIM);
        for h in 0..IP_KV_HEADS {
            for s in 0..total {
                for x in 0..IP_DIM {
                    out.push(buf[(h * capacity + s) * IP_DIM + x]);
                }
            }
        }
        out
    }

    /// Run one step through the copy path: a distinct `[1, KV, past, DIM]` past
    /// input and freshly-allocated present outputs. Returns
    /// `(attention_output, present_key, present_value)` with the present caches
    /// densely shaped `[1, KV, total, DIM]`.
    #[allow(clippy::too_many_arguments)]
    fn run_copy_step(
        kernel: &dyn Kernel,
        past: usize,
        total: usize,
        q_seq: usize,
        query: &[f32],
        cur_k: &[f32],
        cur_v: &[f32],
        past_key: &[f32],
        past_value: &[f32],
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let mut out = Owned::zeros_f32(&[1, q_seq, IP_NUM_HEADS * IP_DIM]);
        let mut pk = Owned::zeros_f32(&[1, IP_KV_HEADS, total, IP_DIM]);
        let mut pv = Owned::zeros_f32(&[1, IP_KV_HEADS, total, IP_DIM]);
        let past_k_owned = Owned::f32(&[1, IP_KV_HEADS, past, IP_DIM], past_key);
        let past_v_owned = Owned::f32(&[1, IP_KV_HEADS, past, IP_DIM], past_value);
        kernel
            .execute(
                &[
                    Owned::f32(&[1, q_seq, IP_NUM_HEADS * IP_DIM], query).view(),
                    Owned::f32(&[1, q_seq, IP_KV_HEADS * IP_DIM], cur_k).view(),
                    Owned::f32(&[1, q_seq, IP_KV_HEADS * IP_DIM], cur_v).view(),
                    past_k_owned.view(),
                    past_v_owned.view(),
                    Owned::i32(&[1], &[(total - 1) as i32]).view(),
                    Owned::i32(&[], &[total as i32]).view(),
                ],
                &mut [out.view_mut(), pk.view_mut(), pv.view_mut()],
            )
            .unwrap();
        (out.to_f32(), pk.to_f32(), pv.to_f32())
    }

    /// Run one step through the in-place path: `kbuf`/`vbuf` are a
    /// `[1, KV, capacity, DIM]` buffer whose `[0, past)` rows already hold the
    /// past cache. The current step's K/V is appended in place and the buffers
    /// are mutated; returns the attention output.
    #[allow(clippy::too_many_arguments)]
    fn run_inplace_step(
        kernel: &dyn Kernel,
        capacity: usize,
        total: usize,
        q_seq: usize,
        query: &[f32],
        cur_k: &[f32],
        cur_v: &[f32],
        kbuf: &mut [f32],
        vbuf: &mut [f32],
    ) -> Vec<f32> {
        let kv_shape = [1usize, IP_KV_HEADS, capacity, IP_DIM];
        let kv_strides = compute_contiguous_strides(&kv_shape);
        let k_ptr = kbuf.as_mut_ptr();
        let v_ptr = vbuf.as_mut_ptr();
        let query_owned = Owned::f32(&[1, q_seq, IP_NUM_HEADS * IP_DIM], query);
        let ck = Owned::f32(&[1, q_seq, IP_KV_HEADS * IP_DIM], cur_k);
        let cv = Owned::f32(&[1, q_seq, IP_KV_HEADS * IP_DIM], cur_v);
        let seqlens = Owned::i32(&[1], &[(total - 1) as i32]);
        let tsl = Owned::i32(&[], &[total as i32]);
        let mut out = Owned::zeros_f32(&[1, q_seq, IP_NUM_HEADS * IP_DIM]);
        // Past inputs (const) and present outputs (mut) intentionally alias the
        // same capacity buffers — this is the structural signal the kernel gates
        // on. Raw pointers mirror the executor's device-binding wiring.
        let past_k_view = TensorView::new(
            DevicePtr(k_ptr as *const std::ffi::c_void),
            DataType::Float32,
            &kv_shape,
            &kv_strides,
            onnx_runtime_ir::DeviceId::cpu(),
        );
        let past_v_view = TensorView::new(
            DevicePtr(v_ptr as *const std::ffi::c_void),
            DataType::Float32,
            &kv_shape,
            &kv_strides,
            onnx_runtime_ir::DeviceId::cpu(),
        );
        let present_k = TensorMut::new(
            DevicePtrMut(k_ptr as *mut std::ffi::c_void),
            DataType::Float32,
            &kv_shape,
            &kv_strides,
            onnx_runtime_ir::DeviceId::cpu(),
        );
        let present_v = TensorMut::new(
            DevicePtrMut(v_ptr as *mut std::ffi::c_void),
            DataType::Float32,
            &kv_shape,
            &kv_strides,
            onnx_runtime_ir::DeviceId::cpu(),
        );
        kernel
            .execute(
                &[
                    query_owned.view(),
                    ck.view(),
                    cv.view(),
                    past_k_view,
                    past_v_view,
                    seqlens.view(),
                    tsl.view(),
                ],
                &mut [out.view_mut(), present_k, present_v],
            )
            .unwrap();
        out.to_f32()
    }

    #[test]
    fn inplace_decode_matches_copy_path_with_spare_capacity() {
        let kernel = gqa_kernel(&[]);
        let past = 2usize;
        let total = 3usize;
        let capacity = 6usize; // capacity strictly greater than total
        let query = vec![1., 0., 1., 0., 0., 1., 0., 1.];
        // Past cache laid out [1, KV, past, DIM].
        let past_k = vec![1., 0., 0., 1., 10., 0., 0., 10.];
        let past_v = vec![1., 2., 3., 4., 10., 20., 30., 40.];
        let cur_k = vec![1., 1., 10., 10.];
        let cur_v = vec![5., 6., 50., 60.];

        let (copy_out, copy_pk, copy_pv) = run_copy_step(
            kernel.as_ref(),
            past,
            total,
            1,
            &query,
            &cur_k,
            &cur_v,
            &past_k,
            &past_v,
        );

        // The tail sentinel proves the kernel never rewrites capacity beyond the
        // live length: rows [total, capacity) must survive untouched.
        const TAIL: f32 = -999.0;
        let mut kbuf = build_capacity_buffer(capacity, past, &past_k, TAIL);
        let mut vbuf = build_capacity_buffer(capacity, past, &past_v, TAIL);
        let inplace_out = run_inplace_step(
            kernel.as_ref(),
            capacity,
            total,
            1,
            &query,
            &cur_k,
            &cur_v,
            &mut kbuf,
            &mut vbuf,
        );

        assert_eq!(
            inplace_out, copy_out,
            "attention output must be byte-identical"
        );
        assert_eq!(
            head_prefix(&kbuf, capacity, total),
            copy_pk,
            "present_key prefix mismatch"
        );
        assert_eq!(
            head_prefix(&vbuf, capacity, total),
            copy_pv,
            "present_value prefix mismatch"
        );
        // Tail untouched: append-only, no capacity rewrite.
        for h in 0..IP_KV_HEADS {
            for s in total..capacity {
                for x in 0..IP_DIM {
                    assert_eq!(
                        kbuf[(h * capacity + s) * IP_DIM + x],
                        TAIL,
                        "key tail rewritten"
                    );
                    assert_eq!(
                        vbuf[(h * capacity + s) * IP_DIM + x],
                        TAIL,
                        "value tail rewritten"
                    );
                }
            }
        }
    }

    #[test]
    fn inplace_decode_matches_copy_path_at_exact_capacity() {
        // capacity == total ⇒ the in-place buffer layout is identical to the
        // copy path's present, so the whole cache compares byte-for-byte.
        let kernel = gqa_kernel(&[]);
        let past = 3usize;
        let total = 4usize;
        let capacity = 4usize;
        let query = vec![0.5, -0.5, 1., 0., 0.25, 1., -1., 0.75];
        let past_k = vec![1., 0., 0., 1., 2., -1., 10., 0., 0., 10., 5., 5.];
        let past_v = vec![1., 2., 3., 4., 5., 6., 10., 20., 30., 40., 50., 60.];
        let cur_k = vec![0.5, 0.5, 7., 7.];
        let cur_v = vec![7., 8., 70., 80.];

        let (copy_out, copy_pk, copy_pv) = run_copy_step(
            kernel.as_ref(),
            past,
            total,
            1,
            &query,
            &cur_k,
            &cur_v,
            &past_k,
            &past_v,
        );

        let mut kbuf = build_capacity_buffer(capacity, past, &past_k, 0.0);
        let mut vbuf = build_capacity_buffer(capacity, past, &past_v, 0.0);
        let inplace_out = run_inplace_step(
            kernel.as_ref(),
            capacity,
            total,
            1,
            &query,
            &cur_k,
            &cur_v,
            &mut kbuf,
            &mut vbuf,
        );

        assert_eq!(inplace_out, copy_out);
        assert_eq!(kbuf, copy_pk);
        assert_eq!(vbuf, copy_pv);
    }

    #[test]
    fn inplace_prefill_then_decode_boundary_matches_copy_path() {
        // Prefill P tokens into an empty capacity buffer, then decode one token,
        // driving BOTH the copy path and the in-place path with the same inputs
        // and asserting identical logits/cache at every step.
        let kernel = gqa_kernel(&[]);
        let capacity = 8usize;
        // ── Prefill: past=0, total=P, q_seq=P ──
        let prefill = 3usize;
        let query_p = vec![
            1., 0., 0., 1., 1., 1., 0., 0., // s0
            0., 1., 1., 0., 0., 1., 1., 1., // s1
            1., 1., 0., 0., 1., 0., 0., 1., // s2
        ];
        let cur_k_p = vec![1., 0., 0., 1., 0., 1., 1., 0., 1., 1., 0., 1.];
        let cur_v_p = vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.];
        let empty: Vec<f32> = Vec::new();
        let (copy_out_p, copy_pk_p, copy_pv_p) = run_copy_step(
            kernel.as_ref(),
            0,
            prefill,
            prefill,
            &query_p,
            &cur_k_p,
            &cur_v_p,
            &empty,
            &empty,
        );
        let mut kbuf = vec![0.0f32; IP_KV_HEADS * capacity * IP_DIM];
        let mut vbuf = vec![0.0f32; IP_KV_HEADS * capacity * IP_DIM];
        let inplace_out_p = run_inplace_step(
            kernel.as_ref(),
            capacity,
            prefill,
            prefill,
            &query_p,
            &cur_k_p,
            &cur_v_p,
            &mut kbuf,
            &mut vbuf,
        );
        assert_eq!(inplace_out_p, copy_out_p, "prefill logits mismatch");
        assert_eq!(
            head_prefix(&kbuf, capacity, prefill),
            copy_pk_p,
            "prefill key mismatch"
        );
        assert_eq!(
            head_prefix(&vbuf, capacity, prefill),
            copy_pv_p,
            "prefill value mismatch"
        );

        // ── Decode: past=P, total=P+1, q_seq=1 — reads the prefilled rows ──
        let total = prefill + 1;
        let query_d = vec![0.3, 0.7, 1., -1., 0., 1., 0.5, 0.5];
        let cur_k_d = vec![0.9, 0.1, 2., 2.];
        let cur_v_d = vec![13., 14., 15., 16.];
        // The copy path's past is the prefill present cache [1, KV, P, DIM].
        let (copy_out_d, copy_pk_d, copy_pv_d) = run_copy_step(
            kernel.as_ref(),
            prefill,
            total,
            1,
            &query_d,
            &cur_k_d,
            &cur_v_d,
            &copy_pk_p,
            &copy_pv_p,
        );
        let inplace_out_d = run_inplace_step(
            kernel.as_ref(),
            capacity,
            total,
            1,
            &query_d,
            &cur_k_d,
            &cur_v_d,
            &mut kbuf,
            &mut vbuf,
        );
        assert_eq!(inplace_out_d, copy_out_d, "decode logits mismatch");
        assert_eq!(
            head_prefix(&kbuf, capacity, total),
            copy_pk_d,
            "decode key mismatch"
        );
        assert_eq!(
            head_prefix(&vbuf, capacity, total),
            copy_pv_d,
            "decode value mismatch"
        );
    }

    #[test]
    fn inplace_matches_copy_path_with_rotary_and_local_window() {
        // The append fast path must be invariant to rotary embedding and local
        // (sliding-window) masking, which alter K storage and attention bounds.
        let past = 4usize;
        let total = 5usize;
        let capacity = 7usize;
        let extra = vec![
            ("num_heads", Attribute::Int(IP_NUM_HEADS as i64)),
            ("kv_num_heads", Attribute::Int(IP_KV_HEADS as i64)),
            ("local_window_size", Attribute::Int(3)),
        ];
        let kernel = kernel(&extra);
        let query: Vec<f32> = (0..IP_NUM_HEADS * IP_DIM)
            .map(|i| mixed_scale_value(i, 11))
            .collect();
        let past_k: Vec<f32> = (0..IP_KV_HEADS * past * IP_DIM)
            .map(|i| mixed_scale_value(i, 22))
            .collect();
        let past_v: Vec<f32> = (0..IP_KV_HEADS * past * IP_DIM)
            .map(|i| mixed_scale_value(i, 33))
            .collect();
        let cur_k: Vec<f32> = (0..IP_KV_HEADS * IP_DIM)
            .map(|i| mixed_scale_value(i, 44))
            .collect();
        let cur_v: Vec<f32> = (0..IP_KV_HEADS * IP_DIM)
            .map(|i| mixed_scale_value(i, 55))
            .collect();

        let (copy_out, copy_pk, copy_pv) = run_copy_step(
            kernel.as_ref(),
            past,
            total,
            1,
            &query,
            &cur_k,
            &cur_v,
            &past_k,
            &past_v,
        );
        let mut kbuf = build_capacity_buffer(capacity, past, &past_k, 42.0);
        let mut vbuf = build_capacity_buffer(capacity, past, &past_v, 42.0);
        let inplace_out = run_inplace_step(
            kernel.as_ref(),
            capacity,
            total,
            1,
            &query,
            &cur_k,
            &cur_v,
            &mut kbuf,
            &mut vbuf,
        );

        assert_eq!(inplace_out, copy_out);
        assert_eq!(head_prefix(&kbuf, capacity, total), copy_pk);
        assert_eq!(head_prefix(&vbuf, capacity, total), copy_pv);
    }

    #[test]
    fn detect_inplace_kv_gate_true_only_on_structural_aliasing() {
        let kernel = raw_gqa_kernel(0, false);
        let capacity = 5usize;
        let total = 3usize;
        let present_seq = capacity; // == cache.seq for the aliased buffer
        let present_len = IP_KV_HEADS * present_seq * IP_DIM;
        let kv_shape = [1usize, IP_KV_HEADS, capacity, IP_DIM];
        let kv_strides = compute_contiguous_strides(&kv_shape);
        let mut kbuf = vec![0.0f32; present_len];
        let mut vbuf = vec![0.0f32; present_len];
        let k_ptr = kbuf.as_mut_ptr();
        let v_ptr = vbuf.as_mut_ptr();
        let cpu = onnx_runtime_ir::DeviceId::cpu();
        let make_view = |ptr: *const f32| {
            TensorView::new(
                DevicePtr(ptr as *const std::ffi::c_void),
                DataType::Float32,
                &kv_shape,
                &kv_strides,
                cpu,
            )
        };
        let make_mut = |ptr: *mut f32| {
            TensorMut::new(
                DevicePtrMut(ptr as *mut std::ffi::c_void),
                DataType::Float32,
                &kv_shape,
                &kv_strides,
                cpu,
            )
        };
        let past_view_gate = make_view(k_ptr);
        let past_key = PastCache::from_cache(&past_view_gate, IP_KV_HEADS, "past_key").unwrap();

        // Aliased (present==past) ⇒ fast path fires.
        let out0 = Owned::zeros_f32(&[1, 1, IP_NUM_HEADS * IP_DIM]);
        let inputs_aliased = [
            out0.view(),
            out0.view(),
            out0.view(),
            make_view(k_ptr),
            make_view(v_ptr),
            out0.view(),
            out0.view(),
        ];
        let outputs_aliased = [make_mut(k_ptr.cast()), make_mut(k_ptr), make_mut(v_ptr)];
        assert!(
            kernel.detect_inplace_kv(
                &inputs_aliased,
                &outputs_aliased,
                present_seq,
                present_len,
                Some(&past_key)
            ),
            "structural present==past aliasing must be detected"
        );

        // Distinct present buffers ⇒ copy path.
        let mut other_k = vec![0.0f32; present_len];
        let mut other_v = vec![0.0f32; present_len];
        let outputs_distinct = [
            make_mut(k_ptr.cast()),
            make_mut(other_k.as_mut_ptr()),
            make_mut(other_v.as_mut_ptr()),
        ];
        assert!(
            !kernel.detect_inplace_kv(
                &inputs_aliased,
                &outputs_distinct,
                present_seq,
                present_len,
                Some(&past_key)
            ),
            "non-aliased present must fall back to the copy path"
        );

        // Capacity does not cover total (present_sequence_length != cache.seq).
        assert!(
            !kernel.detect_inplace_kv(
                &inputs_aliased,
                &outputs_aliased,
                total,
                present_len,
                Some(&past_key)
            ),
            "capacity-limited case must fall back"
        );

        // No past cache ⇒ copy path.
        assert!(
            !kernel.detect_inplace_kv(
                &inputs_aliased,
                &outputs_aliased,
                present_seq,
                present_len,
                None
            ),
            "absent past must fall back"
        );
    }

    #[test]
    fn detect_inplace_kv_gate_rejects_f16_cache() {
        let kernel = raw_gqa_kernel(0, false);
        let capacity = 4usize;
        let present_seq = capacity;
        let present_len = IP_KV_HEADS * present_seq * IP_DIM;
        let kv_shape = [1usize, IP_KV_HEADS, capacity, IP_DIM];
        let kv_strides = compute_contiguous_strides(&kv_shape);
        let cpu = onnx_runtime_ir::DeviceId::cpu();
        // f16 aliased buffer: the append path only supports contiguous f32, so
        // the gate must reject it and preserve the widen/copy path.
        let mut kbuf = Owned::f16(&kv_shape, &vec![0.0; present_len]);
        let mut vbuf = Owned::f16(&kv_shape, &vec![0.0; present_len]);
        let k_ptr = kbuf.bytes.as_mut_ptr() as *mut std::ffi::c_void;
        let v_ptr = vbuf.bytes.as_mut_ptr() as *mut std::ffi::c_void;
        let past_view = TensorView::new(
            DevicePtr(k_ptr),
            DataType::Float16,
            &kv_shape,
            &kv_strides,
            cpu,
        );
        let past_key = PastCache::from_cache(&past_view, IP_KV_HEADS, "past_key").unwrap();
        let out0 = Owned::zeros_f32(&[1, 1, IP_NUM_HEADS * IP_DIM]);
        let inputs = [
            out0.view(),
            out0.view(),
            out0.view(),
            TensorView::new(
                DevicePtr(k_ptr),
                DataType::Float16,
                &kv_shape,
                &kv_strides,
                cpu,
            ),
            TensorView::new(
                DevicePtr(v_ptr),
                DataType::Float16,
                &kv_shape,
                &kv_strides,
                cpu,
            ),
            out0.view(),
            out0.view(),
        ];
        let outputs = [
            TensorMut::new(
                DevicePtrMut(out0.bytes.as_ptr() as *mut std::ffi::c_void),
                DataType::Float32,
                &kv_shape,
                &kv_strides,
                cpu,
            ),
            TensorMut::new(
                DevicePtrMut(k_ptr),
                DataType::Float16,
                &kv_shape,
                &kv_strides,
                cpu,
            ),
            TensorMut::new(
                DevicePtrMut(v_ptr),
                DataType::Float16,
                &kv_shape,
                &kv_strides,
                cpu,
            ),
        ];
        assert!(
            !kernel.detect_inplace_kv(&inputs, &outputs, present_seq, present_len, Some(&past_key)),
            "f16 caches must not take the f32-only in-place path"
        );
    }
}
