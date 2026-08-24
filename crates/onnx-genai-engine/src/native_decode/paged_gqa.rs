//! Runtime-managed (paged) GroupQueryAttention decode — **Tier B**.
//!
//! The CPU `GroupQueryAttention` kernel's copy path allocates a fresh, fully
//! populated `present_k`/`present_v` (`[B, N_kv, total, D]`) every decode step,
//! copies the whole past prefix into it, and round-trips it back out as graph
//! outputs — O(S) traffic per step that compounds to O(S²) over a generation
//! (see `docs/memory/GQA_KV_MATERIALIZATION_DESIGN.md` §3). Tier A removed the
//! redundant per-element copies but kept the fresh-present allocation and the
//! output round-trip, which are "part of the SSA contract". Tier B eliminates
//! them by making the runtime own a **persistent, append-only KV buffer** the
//! attention reads directly.
//!
//! This module implements that append-only buffer on top of the existing paged
//! KV abstraction ([`onnx_genai_kv::PagedKvCache`]) rather than duplicating it:
//!
//! * **Append is O(1).** Each decode step writes only the current token's `D`
//!   f32 per KV head into the persistent pages
//!   ([`PagedKvCache::append_token_kv`]). A new page is allocated only when the
//!   previous one fills (once per `page_size` tokens), so the steady decode
//!   step performs **zero** fresh present allocations — proven by
//!   [`GQA_PRESENT_ALLOCATIONS`].
//! * **Attention reads pages in place.** For each query row the SDPA core reads
//!   each attended token's K/V row straight from its owning page
//!   ([`PagedKvCache::head_token_row`]) via
//!   [`sdpa_decode_row_accessor`], with **no** per-step concat into a fresh
//!   present buffer and **no** output copy. Because that accessor runs the exact
//!   same `dot`/`scale`/`softcap`/f64-softmax/`axpy` sequence as the contiguous
//!   [`onnx_runtime_ep_cpu::kernels::sdpa::sdpa_decode_row`], the output is
//!   **bit-for-bit identical** to the Tier A fresh-present path for identical
//!   inputs.
//!
//! The primitive here is layer-local and drives one attention layer's paged
//! cache. Wiring it into the live engine session (binding the decoder's
//! `present.*`/`past.*` KV ports onto paged storage so the whole model decodes
//! through it, replacing the kernel's present materialization) and the CUDA GQA
//! Tier B path are deferred — see the PR description.

use anyhow::{Context, Result};
use onnx_genai_kv::{KvKind, LayerKv, PagedKvCache, SequenceId};
use onnx_runtime_ep_cpu::kernels::sdpa::{SoftmaxExp, sdpa_decode_row, sdpa_decode_row_accessor};
use std::sync::atomic::{AtomicU64, Ordering};

/// Count of full `present` K/V buffer allocations performed by GQA decode.
///
/// The runtime-managed (paged) path never allocates a present buffer — it
/// appends into persistent pages and attends in place — so this counter stays
/// flat across steady decode steps. The flat (fresh-present) reference
/// [`flat_gqa_decode_step`] increments it once per step. A test asserts the
/// paged path leaves it unchanged while the flat path grows it, making the
/// "no per-step allocation" optimization falsifiable (mirrors the paged-cache
/// eviction-counter style added in #286).
pub static GQA_PRESENT_ALLOCATIONS: AtomicU64 = AtomicU64::new(0);

/// Read [`GQA_PRESENT_ALLOCATIONS`].
pub fn gqa_present_allocations() -> u64 {
    GQA_PRESENT_ALLOCATIONS.load(Ordering::Relaxed)
}

/// Fixed GQA geometry / policy for one attention layer.
#[derive(Clone, Copy, Debug)]
pub struct PagedGqaConfig {
    /// Number of query heads `N`.
    pub num_heads: usize,
    /// Number of KV heads `N_kv` (`N_kv <= N`; query head `qh` reads kv head
    /// `qh / (N / N_kv)`).
    pub num_kv_heads: usize,
    /// Per-head dimension `D` (K, V, and Q share it here).
    pub head_dim: usize,
    /// Score scale; `1/sqrt(D)` when the model does not override it.
    pub scale: f32,
    /// Optional `softcap * tanh(score / softcap)` logit clamp.
    pub softcap: Option<f32>,
    /// Sliding-window size (`> 0`), or `0` for full causal attention.
    pub local_window: usize,
}

impl PagedGqaConfig {
    fn group(&self) -> usize {
        self.num_heads / self.num_kv_heads
    }

    /// Causal / sliding-window read range `[lo, hi)` for a query at absolute
    /// position `query_pos`, matching the CPU GQA kernel exactly.
    fn window(&self, query_pos: usize) -> (usize, usize) {
        let hi = query_pos + 1;
        let lo = if self.local_window > 0 {
            hi.saturating_sub(self.local_window)
        } else {
            0
        };
        (lo, hi)
    }
}

/// Append one decode step's K/V for a single-layer paged cache and compute the
/// step's GQA attention output, reading K/V **directly from the pages**.
///
/// Layout (batch 1, BNSH):
/// * `q`  — `[num_heads,    q_seq, head_dim]`
/// * `k`, `v` — `[num_kv_heads, q_seq, head_dim]` (this step's new tokens)
/// * `out` — `[num_heads,    q_seq, head_dim]` (attention output, filled here)
///
/// `past_len` is the number of tokens already resident in `cache` for `seq`
/// before this step. The cache must be configured with exactly one layer whose
/// geometry matches `cfg` (`num_kv_heads`/`head_dim`) and F32 precision.
///
/// This performs **no** fresh present allocation and leaves
/// [`GQA_PRESENT_ALLOCATIONS`] untouched. Pages are allocated only when a page
/// fills, so steady decode (`q_seq == 1`) is allocation-free on the fast path.
#[allow(clippy::too_many_arguments)]
pub fn paged_gqa_decode_step(
    cache: &mut PagedKvCache,
    seq: SequenceId,
    cfg: &PagedGqaConfig,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    q_seq: usize,
    past_len: usize,
    out: &mut [f32],
) -> Result<()> {
    let d = cfg.head_dim;
    let nkv = cfg.num_kv_heads;
    debug_assert_eq!(q.len(), cfg.num_heads * q_seq * d);
    debug_assert_eq!(k.len(), nkv * q_seq * d);
    debug_assert_eq!(v.len(), nkv * q_seq * d);
    debug_assert_eq!(out.len(), cfg.num_heads * q_seq * d);

    // Append each new token's K/V into the persistent pages. `k`/`v` are BNSH
    // (`[head, token, dim]`), so a single token's KV heads are strided; gather
    // them into a small reused `[num_kv_heads, head_dim]` staging row — this is
    // O(D) per KV head, never the O(N_kv * total * D) present buffer.
    let mut key_row = vec![0.0f32; nkv * d];
    let mut value_row = vec![0.0f32; nkv * d];
    for t in 0..q_seq {
        for h in 0..nkv {
            let src = (h * q_seq + t) * d;
            key_row[h * d..h * d + d].copy_from_slice(&k[src..src + d]);
            value_row[h * d..h * d + d].copy_from_slice(&v[src..src + d]);
        }
        let appended = cache
            .append_token_kv(
                seq,
                &[LayerKv {
                    key: &key_row,
                    value: &value_row,
                }],
            )
            .context("appending token KV into paged cache")?;
        debug_assert_eq!(appended, past_len + t);
    }

    // Attend: each query row reads its KV window straight from the pages.
    let group = cfg.group();
    for qh in 0..cfg.num_heads {
        let kvh = qh / group;
        for qs in 0..q_seq {
            let query_pos = past_len + qs;
            let (lo, hi) = cfg.window(query_pos);
            let q_base = (qh * q_seq + qs) * d;
            let q_row = &q[q_base..q_base + d];
            let out_base = (qh * q_seq + qs) * d;
            let out_row = &mut out[out_base..out_base + d];
            sdpa_decode_row_accessor(
                q_row,
                |s| {
                    cache
                        .head_token_row(seq, 0, KvKind::Key, kvh, s)
                        .expect("paged key row in range")
                        .expect("paged key row is f32")
                },
                |s| {
                    cache
                        .head_token_row(seq, 0, KvKind::Value, kvh, s)
                        .expect("paged value row in range")
                        .expect("paged value row is f32")
                },
                lo,
                hi,
                cfg.scale,
                cfg.softcap,
                SoftmaxExp::F64Intermediate,
                // No head sink: this reference path models plain GQA, and
                // `PagedGqaConfig` carries no sink term to forward. Keeping it
                // `None` is what the call did before `head_sink` was threaded
                // through, so the paged and fresh-present oracles stay in
                // agreement.
                None,
                out_row,
            );
        }
    }
    Ok(())
}

/// Fresh-present (Tier A) reference GQA decode step, used as the parity oracle.
///
/// Allocates a full `present_k`/`present_v` for `[num_kv_heads, total, head_dim]`
/// (incrementing [`GQA_PRESENT_ALLOCATIONS`]), concatenates the past history and
/// the current step's K/V into it, then attends with the contiguous
/// [`sdpa_decode_row`]. `past_k`/`past_v` are the full history so far, laid out
/// `[num_kv_heads, past_len, head_dim]`. Returns the grown present buffers so a
/// caller can carry them as the next step's past.
#[allow(clippy::type_complexity, clippy::too_many_arguments)]
pub fn flat_gqa_decode_step(
    cfg: &PagedGqaConfig,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    past_k: &[f32],
    past_v: &[f32],
    q_seq: usize,
    past_len: usize,
    out: &mut [f32],
) -> (Vec<f32>, Vec<f32>) {
    let d = cfg.head_dim;
    let nkv = cfg.num_kv_heads;
    let total = past_len + q_seq;
    let present_len = nkv * total * d;

    // The defect Tier B removes: a fresh full present allocation every step.
    GQA_PRESENT_ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
    let mut present_k = vec![0.0f32; present_len];
    let mut present_v = vec![0.0f32; present_len];
    for h in 0..nkv {
        let dst_base = h * total * d;
        if past_len > 0 {
            let src = h * past_len * d;
            present_k[dst_base..dst_base + past_len * d]
                .copy_from_slice(&past_k[src..src + past_len * d]);
            present_v[dst_base..dst_base + past_len * d]
                .copy_from_slice(&past_v[src..src + past_len * d]);
        }
        let dst_cur = dst_base + past_len * d;
        let src_cur = h * q_seq * d;
        present_k[dst_cur..dst_cur + q_seq * d].copy_from_slice(&k[src_cur..src_cur + q_seq * d]);
        present_v[dst_cur..dst_cur + q_seq * d].copy_from_slice(&v[src_cur..src_cur + q_seq * d]);
    }

    let group = cfg.group();
    for qh in 0..cfg.num_heads {
        let kvh = qh / group;
        let head_base = kvh * total * d;
        let k_head = &present_k[head_base..head_base + total * d];
        let v_head = &present_v[head_base..head_base + total * d];
        for qs in 0..q_seq {
            let query_pos = past_len + qs;
            let (lo, hi) = cfg.window(query_pos);
            let q_base = (qh * q_seq + qs) * d;
            let q_row = &q[q_base..q_base + d];
            let out_base = (qh * q_seq + qs) * d;
            let out_row = &mut out[out_base..out_base + d];
            sdpa_decode_row(
                q_row,
                k_head,
                v_head,
                total,
                lo,
                hi,
                cfg.scale,
                cfg.softcap,
                SoftmaxExp::F64Intermediate,
                // No head sink: this reference path models plain GQA, and
                // `PagedGqaConfig` carries no sink term to forward.
                None,
                out_row,
            );
        }
    }
    (present_k, present_v)
}

#[cfg(test)]
mod tests {
    use super::*;
    use onnx_genai_kv::{KvDType, PageTensorConfig, PagedKvCache};

    /// Serializes tests in this module: they share the process-global
    /// [`GQA_PRESENT_ALLOCATIONS`] counter, so the no-alloc assertions would race
    /// against the parity tests' flat-path increments without this.
    static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn serial() -> std::sync::MutexGuard<'static, ()> {
        SERIAL.lock().unwrap_or_else(|poison| poison.into_inner())
    }

    /// Small deterministic PRNG so the parity tests exercise real, non-trivial
    /// values without pulling in an rng dependency.
    struct Lcg(u64);
    impl Lcg {
        fn next_f32(&mut self) -> f32 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
            // Map the top 24 bits into roughly [-1, 1).
            let bits = (self.0 >> 40) as u32;
            (bits as f32 / (1u32 << 23) as f32) - 1.0
        }
        fn fill(&mut self, len: usize) -> Vec<f32> {
            (0..len).map(|_| self.next_f32()).collect()
        }
    }

    fn make_cache(cfg: &PagedGqaConfig, page_size: usize, capacity_tokens: usize) -> PagedKvCache {
        // One page per `page_size` tokens, plus slack, so appends never fail.
        let pages = capacity_tokens / page_size + 4;
        PagedKvCache::new_with_tensor_config(
            PageTensorConfig {
                num_layers: 1,
                num_kv_heads: cfg.num_kv_heads,
                head_dim: cfg.head_dim,
                page_size,
                dtype: KvDType::F32,
            },
            pages,
        )
    }

    /// Drive `steps` decode steps through both the paged and the flat
    /// fresh-present paths and assert the attention output is bit-for-bit
    /// identical at every step. `prefill` tokens are consumed in the first step
    /// (`q_seq = prefill`), then one token per step.
    fn assert_parity(cfg: &PagedGqaConfig, page_size: usize, prefill: usize, steps: usize) {
        let mut rng = Lcg(0x9E3779B97F4A7C15 ^ (cfg.num_heads as u64) << 8);
        let d = cfg.head_dim;
        let nkv = cfg.num_kv_heads;

        let total_tokens = prefill + steps.saturating_sub(1);
        let mut cache = make_cache(cfg, page_size, total_tokens + prefill);
        let seq = cache.create_sequence();

        let mut past_k: Vec<f32> = Vec::new();
        let mut past_v: Vec<f32> = Vec::new();
        let mut past_len = 0usize;

        for step in 0..steps {
            let q_seq = if step == 0 { prefill } else { 1 };
            let q = rng.fill(cfg.num_heads * q_seq * d);
            let k = rng.fill(nkv * q_seq * d);
            let v = rng.fill(nkv * q_seq * d);

            let mut out_flat = vec![0.0f32; cfg.num_heads * q_seq * d];
            let (present_k, present_v) = flat_gqa_decode_step(
                cfg,
                &q,
                &k,
                &v,
                &past_k,
                &past_v,
                q_seq,
                past_len,
                &mut out_flat,
            );

            let mut out_paged = vec![0.0f32; cfg.num_heads * q_seq * d];
            paged_gqa_decode_step(
                &mut cache,
                seq,
                cfg,
                &q,
                &k,
                &v,
                q_seq,
                past_len,
                &mut out_paged,
            )
            .expect("paged decode step");

            for (i, (a, b)) in out_flat.iter().zip(&out_paged).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "step {step} elem {i}: flat {a} != paged {b} (bit mismatch)"
                );
            }

            past_k = present_k;
            past_v = present_v;
            past_len += q_seq;
        }
    }

    fn cfg(num_heads: usize, num_kv_heads: usize, head_dim: usize) -> PagedGqaConfig {
        PagedGqaConfig {
            num_heads,
            num_kv_heads,
            head_dim,
            scale: 1.0 / (head_dim as f32).sqrt(),
            softcap: None,
            local_window: 0,
        }
    }

    #[test]
    fn paged_matches_flat_mha_multistep() {
        let _serial = serial();
        // num_kv_heads == num_heads (plain MHA), several decode steps growing
        // the past, page boundaries crossed (page_size 4, ~12 tokens).
        assert_parity(&cfg(2, 2, 8), 4, 1, 12);
    }

    #[test]
    fn paged_matches_flat_group_broadcast() {
        let _serial = serial();
        // num_kv_heads < num_heads: query heads share kv heads. A wrong head map
        // would diverge immediately.
        assert_parity(&cfg(8, 2, 16), 4, 1, 10);
    }

    #[test]
    fn paged_matches_flat_mqa() {
        let _serial = serial();
        // Multi-query attention: a single kv head shared by all query heads.
        assert_parity(&cfg(4, 1, 8), 3, 1, 9);
    }

    #[test]
    fn paged_matches_flat_prefill_then_decode() {
        let _serial = serial();
        // First step is a multi-token prefill (q_seq = 5), then decode grows it
        // across page boundaries.
        assert_parity(&cfg(4, 2, 8), 4, 5, 8);
    }

    #[test]
    fn paged_matches_flat_page_boundary_exact() {
        let _serial = serial();
        // page_size 1 forces a fresh page every token — stresses the per-token
        // page mapping in `head_token_row`.
        assert_parity(&cfg(2, 2, 8), 1, 1, 7);
    }

    #[test]
    fn paged_matches_flat_with_softcap() {
        let _serial = serial();
        let mut c = cfg(4, 2, 8);
        c.softcap = Some(20.0);
        assert_parity(&c, 4, 1, 10);
    }

    #[test]
    fn paged_matches_flat_sliding_window() {
        let _serial = serial();
        // Local (sliding-window) attention: only the last `local_window` tokens
        // are attended. Parity must hold once the window starts sliding.
        let mut c = cfg(4, 2, 8);
        c.local_window = 3;
        assert_parity(&c, 4, 1, 12);
    }

    #[test]
    fn steady_decode_allocates_zero_present_buffers() {
        let _serial = serial();
        // The decisive Tier-B invariant: the paged decode step performs no fresh
        // present K/V allocation, while the flat path performs exactly one per
        // step. Asserted via the shared `GQA_PRESENT_ALLOCATIONS` counter.
        let c = cfg(4, 2, 8);
        let d = c.head_dim;
        let nkv = c.num_kv_heads;
        let steps = 16;
        let page_size = 4;

        let mut rng = Lcg(0xDEADBEEF);
        let mut cache = make_cache(&c, page_size, steps + 1);
        let seq = cache.create_sequence();

        let before = gqa_present_allocations();
        for step in 0..steps {
            let q = rng.fill(c.num_heads * d);
            let k = rng.fill(nkv * d);
            let v = rng.fill(nkv * d);
            let mut out = vec![0.0f32; c.num_heads * d];
            paged_gqa_decode_step(&mut cache, seq, &c, &q, &k, &v, 1, step, &mut out)
                .expect("paged decode step");
        }
        assert_eq!(
            gqa_present_allocations(),
            before,
            "paged decode must not allocate any present buffer across {steps} steps"
        );

        // The flat reference, by contrast, allocates exactly one present buffer
        // per step — proving the counter actually observes present allocations.
        let mut past_k: Vec<f32> = Vec::new();
        let mut past_v: Vec<f32> = Vec::new();
        let before_flat = gqa_present_allocations();
        for step in 0..steps {
            let q = rng.fill(c.num_heads * d);
            let k = rng.fill(nkv * d);
            let v = rng.fill(nkv * d);
            let mut out = vec![0.0f32; c.num_heads * d];
            let (pk, pv) =
                flat_gqa_decode_step(&c, &q, &k, &v, &past_k, &past_v, 1, step, &mut out);
            past_k = pk;
            past_v = pv;
        }
        assert_eq!(
            gqa_present_allocations() - before_flat,
            steps as u64,
            "flat reference must allocate one present buffer per step"
        );
    }
}
