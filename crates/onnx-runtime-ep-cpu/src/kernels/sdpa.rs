//! Shared **scaled-dot-product-attention (SDPA) core** — the one place the
//! attention math lives, so the many attention ops in this crate
//! (`com.microsoft::MultiHeadAttention`, `ai.onnx::Attention`,
//! `GroupQueryAttention`, `com.microsoft::FusedAttention`, …) stop
//! copy-pasting the `QKᵀ → scale → [softcap] → +bias → +mask → softmax → ·V`
//! sequence and instead adapt onto this primitive.
//!
//! ## What lives here vs. in the adapter
//!
//! This core is deliberately **pure f32 math over dense `BNSH` buffers**. It
//! knows nothing about tensor layouts, packed QKV, bias projection, or KV
//! caches — those are *adapter* responsibilities, because they differ per op
//! and are cheap reshapes/concats. The adapter's job is to normalize its
//! op-specific inputs into the [`SdpaTensors`] contract (query
//! `[B, Nq, Sq, Dh]`, key `[B, Nkv, Tk, Dh]`, value `[B, Nkv, Tk, Dv]`, all
//! contiguous f32), then call [`sdpa_f32`]. This keeps the numerics in exactly
//! one place while letting each op keep its own I/O quirks.
//!
//! The pluggable variation the core itself expresses:
//!
//! * **GQA / MQA head sharing** — `num_kv_heads ≤ num_heads`; query head `n`
//!   reads kv head `n / (num_heads / num_kv_heads)`. `num_kv_heads == num_heads`
//!   is plain MHA.
//! * **Differing V head size** — `v_head_size` (`Dv`) is independent of the
//!   Q/K `head_size` (`Dh`).
//! * **Scale placement** — [`ScaleMode::PostDot`] multiplies the raw dot by
//!   `scale` (ORT's MHA/fused path, folded into the GEMM `alpha`);
//!   [`ScaleMode::SplitSqrt`] pre-scales each operand by `√scale` (ORT's
//!   `ai.onnx::Attention` overflow-safe path).
//! * **Softcap** — optional `softcap · tanh(score / softcap)` logit clamp
//!   (`ai.onnx::Attention`), applied right after the scale as ORT does.
//! * **Additive attention bias** — a per-`(b, head, i, j)` float addend
//!   ([`AttnBias`]); [`BroadcastBias`] covers the `(B|1, N|1, S, T)` broadcast
//!   the contrib ops use.
//! * **Additive key mask** — a per-`(b, i, j)` float addend ([`KeyMask`]),
//!   covering key-padding masks; it is head-independent, matching ORT.
//! * **Causal masking with a past-KV offset** — key `j` is masked for query `i`
//!   when `j > past_seq + i`, using a caller-chosen fill (`f32::MIN` for MHA).
//! * **Optional QK score capture** — the logits or probabilities
//!   (`[B, Nq, Sq, Tk]`) at a caller-chosen pipeline stage ([`QkCaptureStage`])
//!   for ops that emit `qk_matmul_output`.
//!
//! ## Numerical contract (why this is a *drop-in* factoring)
//!
//! The per-`(b, head, i)` inner sequence is byte-for-byte the loop the
//! standalone MHA kernel used to run:
//!
//! ```text
//! score = dot(Q_i, K_j)                 # plain f32 fma-free accumulation
//! score = scale · score                 # PostDot   (or operands pre-scaled)
//! score = softcap·tanh(score/softcap)   # only when softcap set
//! score += attn_bias(b, n, i, j)        # 0.0 when absent (identity add)
//! score += key_mask(b, i, j)            # 0.0 when absent (identity add)
//! score  = causal_fill  if j > past+i   # override, matching ORT's merged mask
//! probs  = softmax(score)               # subtract row max, then normalize
//! out_i += probs_j · V_j                # plain f32 accumulation
//! ```
//!
//! The addends are applied in this exact order (never pre-summed) so that a
//! migrated op reproduces its reference goldens *bit-for-bit*, not merely
//! within tolerance. `f16`/`bf16` widen at the adapter boundary (Q/K/V are
//! already f32 here).
//!
//! ## Scalar reference vs. MLAS-GEMM fast path
//!
//! [`sdpa_f32_scalar`] is the byte-exact reference above: a scalar triple loop
//! whose numerics the parity goldens pin. It is retained unchanged as the
//! oracle the tolerance tests cross-check against.
//!
//! [`sdpa_f32`] is the adapter-facing entry point. When the crate is built
//! `--features mlas` and no [`QkCapture`] is requested, it runs a **fast path**
//! that (a) computes `QKᵀ` and `P·V` as real MLAS SGEMMs (batched over
//! `batch·head`, GQA/MQA kv heads gathered by group), (b) applies
//! `scale → softcap → bias → mask → causal` per **row** on plain slices (same
//! order as the scalar loop), and (c) rayon-parallelizes across the
//! `(batch, head)` tiles on the crate's shared pool (no oversubscription — MLAS
//! itself tiles onto that same pool). GEMM reorders float accumulation, so the
//! fast path is **not** bit-identical to the scalar loop; it is gated by
//! tolerance against both the scalar reference and live ORT 1.26 (which also
//! uses MLAS, so the fast path often matches ORT *more* closely than the scalar
//! path). Any shape the fast path cannot serve — or a [`QkCapture`] request, or
//! a non-`mlas` build — transparently falls back to [`sdpa_f32_scalar`], so the
//! output is always correct.

/// Query/key/value operands for one SDPA call, as dense contiguous f32 buffers
/// in `BNSH` (`[batch, heads, seq, dim]`) order.
///
/// * `q`  — `[batch, num_heads, q_seq, head_size]`
/// * `k`  — `[batch, num_kv_heads, kv_seq, head_size]`
/// * `v`  — `[batch, num_kv_heads, kv_seq, v_head_size]`
pub struct SdpaTensors<'a> {
    pub q: &'a [f32],
    pub k: &'a [f32],
    pub v: &'a [f32],
    pub batch: usize,
    /// Number of query heads (`Nq`).
    pub num_heads: usize,
    /// Number of key/value heads (`Nkv ≤ Nq`); `Nq` for plain MHA.
    pub num_kv_heads: usize,
    /// Query sequence length (`Sq`).
    pub q_seq: usize,
    /// Total key/value sequence length after any cache concat (`Tk`).
    pub kv_seq: usize,
    /// Q/K head dimension (`Dh`).
    pub head_size: usize,
    /// V head dimension (`Dv`); may differ from `head_size`.
    pub v_head_size: usize,
}

/// How the score `scale` is applied to the raw `Q·Kᵀ` dot product.
#[derive(Clone, Copy, Debug)]
pub enum ScaleMode {
    /// Multiply the completed dot product by `scale` (ORT folds this into the
    /// GEMM `alpha`; used by MHA and `FusedAttention`).
    PostDot(f32),
    /// Pre-scale each Q and K element by `√scale` before the dot, so extreme
    /// magnitudes can't overflow the accumulation (ORT's `ai.onnx::Attention`).
    SplitSqrt(f32),
}

/// Fixed SDPA parameters (everything that isn't the Q/K/V data or the
/// bias/mask hooks).
pub struct SdpaConfig {
    /// Score scaling strategy.
    pub scale: ScaleMode,
    /// Optional `softcap · tanh(score / softcap)` logit clamp; `None` disables.
    pub softcap: Option<f32>,
    /// Apply lower-triangular causal masking (with the `past_seq` offset).
    pub causal: bool,
    /// Length of any KV already in the cache, shifting the causal frontier:
    /// key `j` is visible to query `i` iff `j <= past_seq + i`.
    pub past_seq: usize,
    /// Additive fill written into causally-masked positions (`f32::MIN` in ORT).
    pub causal_fill: f32,
}

/// Per-`(batch, head, query, key)` additive attention bias.
///
/// Called once per score; return `0.0` to contribute nothing. Kept as a trait
/// (rather than an `Option<&[f32]>`) so ops with exotic bias broadcasts plug in
/// without the core knowing their layout.
///
/// The `Sync` bound lets the [`sdpa_f32`] fast path share a single `&dyn
/// AttnBias` across the rayon workers that own disjoint `(batch, head)` tiles;
/// every adapter hook here holds only shared `&[f32]`/scalars, so it is `Sync`.
pub trait AttnBias: Sync {
    fn at(&self, b: usize, head: usize, i: usize, j: usize) -> f32;
}

/// Per-`(batch, query, key)` additive key mask (head-independent, as in ORT's
/// key-padding masks). Return `0.0` to keep a key, a large negative fill to
/// mask it.
pub trait KeyMask: Sync {
    fn at(&self, b: usize, i: usize, j: usize) -> f32;
}

/// No-op attention bias (contributes `0.0` everywhere).
pub struct NoBias;
impl AttnBias for NoBias {
    #[inline]
    fn at(&self, _b: usize, _head: usize, _i: usize, _j: usize) -> f32 {
        0.0
    }
}

/// No-op key mask (keeps every key).
pub struct NoMask;
impl KeyMask for NoMask {
    #[inline]
    fn at(&self, _b: usize, _i: usize, _j: usize) -> f32 {
        0.0
    }
}

/// Additive attention bias with the contrib-op `(B|1, N|1, S, T)` broadcast:
/// leading batch and head dims may each be `1` (broadcast) or full.
pub struct BroadcastBias<'a> {
    data: &'a [f32],
    dims: [usize; 4],
}

impl<'a> BroadcastBias<'a> {
    /// `dims` is the bias tensor's `[B|1, N|1, S, T]` shape; `data` its
    /// row-major contents.
    pub fn new(data: &'a [f32], dims: [usize; 4]) -> Self {
        Self { data, dims }
    }
}

impl AttnBias for BroadcastBias<'_> {
    #[inline]
    fn at(&self, b: usize, head: usize, i: usize, j: usize) -> f32 {
        let b0 = if self.dims[0] == 1 { 0 } else { b };
        let n0 = if self.dims[1] == 1 { 0 } else { head };
        let off = (((b0 * self.dims[1] + n0) * self.dims[2] + i) * self.dims[3]) + j;
        self.data[off]
    }
}

/// Which point in the per-score pipeline a [`QkCapture`] records.
///
/// `ai.onnx::Attention`'s `qk_matmul_output_mode` selects one of these; MHA and
/// `FusedAttention` capture at [`PreSoftmax`](QkCaptureStage::PreSoftmax).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QkCaptureStage {
    /// Right after the score scale, before softcap (Attention mode `0`).
    PostScale,
    /// After softcap, before bias/mask (Attention mode `1`; identical to
    /// [`PostScale`](QkCaptureStage::PostScale) when softcap is disabled).
    PostSoftcap,
    /// After bias/mask/causal, before softmax (default; MHA/Fused
    /// `qk_matmul_output`, Attention mode `2`).
    PreSoftmax,
    /// After the softmax normalization — i.e. the probabilities (Attention
    /// mode `3`).
    PostSoftmax,
}

/// Optional QK score capture target for ops that emit `qk_matmul_output`.
///
/// Holds the logits (or, for [`QkCaptureStage::PostSoftmax`], the
/// probabilities) in `[batch, num_heads, q_seq, kv_seq]` order, recorded at the
/// pipeline point named by `stage`.
pub struct QkCapture<'a> {
    pub scores: &'a mut [f32],
    pub stage: QkCaptureStage,
}

/// Run scaled-dot-product attention over `t`, writing the context into `y`
/// (`[batch, num_heads, q_seq, v_head_size]`, `BNSH`).
///
/// This is the **adapter-facing entry point**. It dispatches to the
/// MLAS-GEMM + rayon fast path ([`sdpa_f32_fast`]) when the crate is built
/// `--features mlas`, no [`QkCapture`] is requested, and the shape is
/// non-empty; otherwise it runs the scalar reference ([`sdpa_f32_scalar`]).
/// Both honour the exact `scale → softcap → bias → mask → causal → softmax`
/// sequence documented at the module level; the fast path only reorders the two
/// matmul accumulations (via GEMM), so it agrees with the scalar path to tight
/// tolerance rather than bit-for-bit.
///
/// `bias` and `mask` are applied additively in that order (pass [`NoBias`] /
/// [`NoMask`] to skip). When `qk` is `Some`, the requested pipeline stage is
/// copied out; that path is always served by the scalar reference so the
/// captured logits stay bit-identical.
pub fn sdpa_f32(
    t: &SdpaTensors,
    cfg: &SdpaConfig,
    bias: &dyn AttnBias,
    mask: &dyn KeyMask,
    y: &mut [f32],
    qk: Option<QkCapture>,
) {
    #[cfg(feature = "mlas")]
    {
        // The fast path handles every masking/scale mode, but it does not emit
        // a QkCapture (that stays on the scalar reference so the captured
        // logits are bit-identical) and needs a non-empty problem.
        let non_empty = t.batch > 0
            && t.num_heads > 0
            && t.q_seq > 0
            && t.kv_seq > 0
            && t.head_size > 0
            && t.v_head_size > 0;
        if qk.is_none() && non_empty {
            sdpa_f32_fast(t, cfg, bias, mask, y);
            return;
        }
    }
    sdpa_f32_scalar(t, cfg, bias, mask, y, qk);
}

/// Byte-exact scalar SDPA reference — the oracle the parity goldens pin.
///
/// See the module docs for the exact numerical sequence; it is a bit-for-bit
/// factoring of the standalone MHA loop and is retained unchanged so the
/// tolerance tests (and the fast path) have a fixed reference to check against.
pub fn sdpa_f32_scalar(
    t: &SdpaTensors,
    cfg: &SdpaConfig,
    bias: &dyn AttnBias,
    mask: &dyn KeyMask,
    y: &mut [f32],
    mut qk: Option<QkCapture>,
) {
    let SdpaTensors {
        q,
        k,
        v,
        batch,
        num_heads,
        num_kv_heads,
        q_seq,
        kv_seq,
        head_size,
        v_head_size,
    } = *t;

    debug_assert_eq!(q.len(), batch * num_heads * q_seq * head_size);
    debug_assert_eq!(k.len(), batch * num_kv_heads * kv_seq * head_size);
    debug_assert_eq!(v.len(), batch * num_kv_heads * kv_seq * v_head_size);
    debug_assert_eq!(y.len(), batch * num_heads * q_seq * v_head_size);
    debug_assert!(num_kv_heads > 0 && num_heads.is_multiple_of(num_kv_heads));

    // Query heads per kv head (GQA/MQA sharing factor; 1 for plain MHA).
    let heads_per_kv = num_heads / num_kv_heads;

    // Score-scale placement.
    let (post_scale, operand_scale) = match cfg.scale {
        ScaleMode::PostDot(s) => (s, 1.0f32),
        ScaleMode::SplitSqrt(s) => (1.0f32, s.sqrt()),
    };

    let mut scores = vec![0.0f32; kv_seq];
    for b in 0..batch {
        for n in 0..num_heads {
            let kv_n = n / heads_per_kv;
            for i in 0..q_seq {
                let q_base = ((b * num_heads + n) * q_seq + i) * head_size;
                let cap_base = ((b * num_heads + n) * q_seq + i) * kv_seq;
                // scores[j] = scale·(Q·Kᵀ) [+softcap] + bias + mask [→ causal].
                for (j, sc) in scores.iter_mut().enumerate() {
                    let k_base = ((b * num_kv_heads + kv_n) * kv_seq + j) * head_size;
                    let mut acc = 0.0f32;
                    for p in 0..head_size {
                        acc += (q[q_base + p] * operand_scale) * (k[k_base + p] * operand_scale);
                    }
                    let mut s = acc * post_scale;
                    if let Some(cap) = qk.as_mut()
                        && cap.stage == QkCaptureStage::PostScale
                    {
                        cap.scores[cap_base + j] = s;
                    }
                    if let Some(softcap) = cfg.softcap {
                        s = softcap * (s / softcap).tanh();
                    }
                    if let Some(cap) = qk.as_mut()
                        && cap.stage == QkCaptureStage::PostSoftcap
                    {
                        cap.scores[cap_base + j] = s;
                    }
                    s += bias.at(b, n, i, j);
                    s += mask.at(b, i, j);
                    if cfg.causal && (j as i64) > cfg.past_seq as i64 + i as i64 {
                        s = cfg.causal_fill;
                    }
                    *sc = s;
                }

                if let Some(cap) = qk.as_mut()
                    && cap.stage == QkCaptureStage::PreSoftmax
                {
                    cap.scores[cap_base..cap_base + kv_seq].copy_from_slice(&scores);
                }

                // Numerically-stable softmax (subtract row max, matching ORT's
                // MlasComputeSoftmax and this crate's softmax kernel). A fully
                // masked row (every score `-inf`) yields a zero row rather than
                // NaN — matching ORT's guarded softmax. Fills that stay finite
                // (e.g. MHA's `f32::MIN`) never trigger this branch, so MHA's
                // numerics are unchanged.
                let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                if max == f32::NEG_INFINITY {
                    for sc in scores.iter_mut() {
                        *sc = 0.0;
                    }
                } else {
                    let mut sum = 0.0f32;
                    for sc in scores.iter_mut() {
                        let e = (*sc - max).exp();
                        *sc = e;
                        sum += e;
                    }
                    let inv = 1.0 / sum;
                    for sc in scores.iter_mut() {
                        *sc *= inv;
                    }
                }

                if let Some(cap) = qk.as_mut()
                    && cap.stage == QkCaptureStage::PostSoftmax
                {
                    cap.scores[cap_base..cap_base + kv_seq].copy_from_slice(&scores);
                }

                // context = probs · V.
                let y_base = ((b * num_heads + n) * q_seq + i) * v_head_size;
                for c in 0..v_head_size {
                    let mut acc = 0.0f32;
                    for (j, &p) in scores.iter().enumerate() {
                        let v_idx = ((b * num_kv_heads + kv_n) * kv_seq + j) * v_head_size + c;
                        acc += p * v[v_idx];
                    }
                    y[y_base + c] = acc;
                }
            }
        }
    }
}

/// MLAS-GEMM + rayon fast path behind [`sdpa_f32`].
///
/// Per `(batch, head)` tile it runs two SGEMMs — `logits = scale · Q·Kᵀ` and
/// `context = probs · V` — with the `softcap → bias → mask → causal → softmax`
/// epilogue applied per row on plain slices, in the exact order the scalar
/// reference uses. GQA/MQA share kv heads by group (`kv = head / (Nq/Nkv)`).
/// Tiles are fanned across the crate's shared rayon pool via
/// `par_chunks_mut`; MLAS tiles its own GEMM work onto that same pool, so there
/// is no oversubscription.
#[cfg(feature = "mlas")]
fn sdpa_f32_fast(
    t: &SdpaTensors,
    cfg: &SdpaConfig,
    bias: &dyn AttnBias,
    mask: &dyn KeyMask,
    y: &mut [f32],
) {
    use rayon::prelude::*;

    let SdpaTensors {
        q,
        k,
        v,
        batch,
        num_heads,
        num_kv_heads,
        q_seq,
        kv_seq,
        head_size,
        v_head_size,
    } = *t;

    debug_assert_eq!(q.len(), batch * num_heads * q_seq * head_size);
    debug_assert_eq!(k.len(), batch * num_kv_heads * kv_seq * head_size);
    debug_assert_eq!(v.len(), batch * num_kv_heads * kv_seq * v_head_size);
    debug_assert_eq!(y.len(), batch * num_heads * q_seq * v_head_size);
    debug_assert!(num_kv_heads > 0 && num_heads.is_multiple_of(num_kv_heads));

    let heads_per_kv = num_heads / num_kv_heads;

    // Both scale modes reduce to `alpha · (Q·K)` under a GEMM: `PostDot(s)`
    // multiplies the dot by `s`, and `SplitSqrt(s)` pre-scales each operand by
    // `√s` so the product carries `s`. Folding `s` into the GEMM `alpha` matches
    // ORT's own MLAS path (`alpha = scale`) and stays within tolerance of the
    // scalar loop's per-operand scaling.
    let alpha = match cfg.scale {
        ScaleMode::PostDot(s) => s,
        ScaleMode::SplitSqrt(s) => s,
    };

    // One tile per `(b, head)`, contiguous in `y` as `[b, head, q_seq, Dv]`.
    let tile_v = q_seq * v_head_size;
    y.par_chunks_mut(tile_v)
        .enumerate()
        .for_each(|(bh, y_tile)| {
            let b = bh / num_heads;
            let n = bh % num_heads;
            let kv_n = n / heads_per_kv;

            let q_off = ((b * num_heads + n) * q_seq) * head_size;
            let k_off = ((b * num_kv_heads + kv_n) * kv_seq) * head_size;
            let v_off = ((b * num_kv_heads + kv_n) * kv_seq) * v_head_size;
            let q_tile = &q[q_off..q_off + q_seq * head_size];
            let k_tile = &k[k_off..k_off + kv_seq * head_size];
            let v_tile = &v[v_off..v_off + kv_seq * v_head_size];

            // logits[q_seq, kv_seq] = alpha · Q · Kᵀ.
            let mut logits = vec![0.0f32; q_seq * kv_seq];
            mlas_sys::sgemm(
                false,
                true,
                q_seq,
                kv_seq,
                head_size,
                alpha,
                q_tile,
                head_size,
                k_tile,
                head_size,
                0.0,
                &mut logits,
                kv_seq,
            );

            // Per-row epilogue: softcap → bias → mask → causal → softmax, on
            // plain slices, in the scalar reference's exact add order.
            for i in 0..q_seq {
                let row = &mut logits[i * kv_seq..i * kv_seq + kv_seq];
                for (j, s) in row.iter_mut().enumerate() {
                    let mut val = *s;
                    if let Some(softcap) = cfg.softcap {
                        val = softcap * (val / softcap).tanh();
                    }
                    val += bias.at(b, n, i, j);
                    val += mask.at(b, i, j);
                    if cfg.causal && (j as i64) > cfg.past_seq as i64 + i as i64 {
                        val = cfg.causal_fill;
                    }
                    *s = val;
                }

                // Numerically-stable softmax with the fully-masked-row → zero
                // guard (matching the scalar reference and ORT).
                let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                if max == f32::NEG_INFINITY {
                    for s in row.iter_mut() {
                        *s = 0.0;
                    }
                } else {
                    let mut sum = 0.0f32;
                    for s in row.iter_mut() {
                        let e = (*s - max).exp();
                        *s = e;
                        sum += e;
                    }
                    let inv = 1.0 / sum;
                    for s in row.iter_mut() {
                        *s *= inv;
                    }
                }
            }

            // context[q_seq, Dv] = probs · V.
            mlas_sys::sgemm(
                false,
                false,
                q_seq,
                v_head_size,
                kv_seq,
                1.0,
                &logits,
                kv_seq,
                v_tile,
                v_head_size,
                0.0,
                y_tile,
                v_head_size,
            );
        });
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Straightforward f32 SDPA reference for cross-checking the core on small
    /// shapes (single head, no bias/mask, PostDot scale).
    fn reference(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        s: usize,
        dh: usize,
        dv: usize,
        scale: f32,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; s * dv];
        for i in 0..s {
            let mut scores = vec![0.0f32; s];
            for (j, sc) in scores.iter_mut().enumerate() {
                let mut acc = 0.0f32;
                for p in 0..dh {
                    acc += q[i * dh + p] * k[j * dh + p];
                }
                *sc = acc * scale;
            }
            let m = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let sum: f32 = scores.iter().map(|x| (x - m).exp()).sum();
            for c in 0..dv {
                let mut acc = 0.0f32;
                for (j, sc) in scores.iter().enumerate() {
                    acc += ((sc - m).exp() / sum) * v[j * dv + c];
                }
                out[i * dv + c] = acc;
            }
        }
        out
    }

    #[test]
    fn postdot_matches_reference() {
        let (s, dh, dv) = (3usize, 4usize, 2usize);
        let q: Vec<f32> = (0..s * dh).map(|x| (x as f32) * 0.1 - 0.5).collect();
        let k: Vec<f32> = (0..s * dh).map(|x| (x as f32) * 0.05).collect();
        let v: Vec<f32> = (0..s * dv).map(|x| (x as f32) * 0.2).collect();
        let scale = 1.0 / (dh as f32).sqrt();
        let t = SdpaTensors {
            q: &q,
            k: &k,
            v: &v,
            batch: 1,
            num_heads: 1,
            num_kv_heads: 1,
            q_seq: s,
            kv_seq: s,
            head_size: dh,
            v_head_size: dv,
        };
        let cfg = SdpaConfig {
            scale: ScaleMode::PostDot(scale),
            softcap: None,
            causal: false,
            past_seq: 0,
            causal_fill: f32::MIN,
        };
        let mut y = vec![0.0f32; s * dv];
        sdpa_f32_scalar(&t, &cfg, &NoBias, &NoMask, &mut y, None);
        let want = reference(&q, &k, &v, s, dh, dv, scale);
        for (a, b) in y.iter().zip(want.iter()) {
            assert!((a - b).abs() < 1e-6, "got {y:?} want {want:?}");
        }
    }

    #[test]
    fn causal_masks_future_keys() {
        // With causal masking and past_seq=0, query 0 must attend only key 0.
        let (s, dh, dv) = (2usize, 2usize, 2usize);
        let q = vec![1.0f32, 0.0, 0.0, 1.0];
        let k = vec![1.0f32, 0.0, 0.0, 1.0];
        let v = vec![10.0f32, 20.0, 30.0, 40.0];
        let t = SdpaTensors {
            q: &q,
            k: &k,
            v: &v,
            batch: 1,
            num_heads: 1,
            num_kv_heads: 1,
            q_seq: s,
            kv_seq: s,
            head_size: dh,
            v_head_size: dv,
        };
        let cfg = SdpaConfig {
            scale: ScaleMode::PostDot(1.0),
            softcap: None,
            causal: true,
            past_seq: 0,
            causal_fill: f32::MIN,
        };
        let mut y = vec![0.0f32; s * dv];
        sdpa_f32_scalar(&t, &cfg, &NoBias, &NoMask, &mut y, None);
        // Query 0 attends only key 0 → exactly V row 0.
        assert!((y[0] - 10.0).abs() < 1e-6 && (y[1] - 20.0).abs() < 1e-6);
    }

    #[test]
    fn gqa_head_sharing_reads_grouped_kv() {
        // 2 query heads, 1 kv head: both query heads must read the same kv head.
        let (s, dh, dv) = (1usize, 2usize, 2usize);
        let q = vec![1.0f32, 0.0, /*h1*/ 0.0, 1.0];
        let k = vec![1.0f32, 1.0]; // single kv head, single key
        let v = vec![5.0f32, 7.0];
        let t = SdpaTensors {
            q: &q,
            k: &k,
            v: &v,
            batch: 1,
            num_heads: 2,
            num_kv_heads: 1,
            q_seq: s,
            kv_seq: s,
            head_size: dh,
            v_head_size: dv,
        };
        let cfg = SdpaConfig {
            scale: ScaleMode::PostDot(1.0),
            softcap: None,
            causal: false,
            past_seq: 0,
            causal_fill: f32::MIN,
        };
        let mut y = vec![0.0f32; 2 * s * dv];
        sdpa_f32_scalar(&t, &cfg, &NoBias, &NoMask, &mut y, None);
        // Single key → softmax is 1.0 → both heads output V row 0.
        for h in 0..2 {
            assert!((y[h * dv] - 5.0).abs() < 1e-6 && (y[h * dv + 1] - 7.0).abs() < 1e-6);
        }
    }

    #[test]
    fn splitsqrt_scale_equivalent_to_postdot_for_moderate_values() {
        // √scale-on-operands and scale-on-dot agree closely for moderate mags.
        let (s, dh, dv) = (2usize, 3usize, 2usize);
        let q: Vec<f32> = (0..s * dh).map(|x| (x as f32) * 0.3).collect();
        let k: Vec<f32> = (0..s * dh).map(|x| (x as f32) * 0.2 - 0.1).collect();
        let v: Vec<f32> = (0..s * dv).map(|x| (x as f32) * 0.5).collect();
        let scale = 1.0 / (dh as f32).sqrt();
        let base = SdpaTensors {
            q: &q,
            k: &k,
            v: &v,
            batch: 1,
            num_heads: 1,
            num_kv_heads: 1,
            q_seq: s,
            kv_seq: s,
            head_size: dh,
            v_head_size: dv,
        };
        let mut y_post = vec![0.0f32; s * dv];
        sdpa_f32_scalar(
            &base,
            &SdpaConfig {
                scale: ScaleMode::PostDot(scale),
                softcap: None,
                causal: false,
                past_seq: 0,
                causal_fill: f32::MIN,
            },
            &NoBias,
            &NoMask,
            &mut y_post,
            None,
        );
        let mut y_split = vec![0.0f32; s * dv];
        sdpa_f32_scalar(
            &base,
            &SdpaConfig {
                scale: ScaleMode::SplitSqrt(scale),
                softcap: None,
                causal: false,
                past_seq: 0,
                causal_fill: f32::MIN,
            },
            &NoBias,
            &NoMask,
            &mut y_split,
            None,
        );
        for (a, b) in y_post.iter().zip(y_split.iter()) {
            assert!((a - b).abs() < 1e-5, "post {y_post:?} split {y_split:?}");
        }
    }

    /// Deterministic pseudo-random f32 fill in `[-1, 1)` for parity fixtures.
    #[cfg(feature = "mlas")]
    fn fill(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        (0..n)
            .map(|_| {
                s ^= s >> 30;
                s = s.wrapping_mul(0xBF58_476D_1CE4_E5B9);
                s ^= s >> 27;
                ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
            })
            .collect()
    }

    /// A dense additive key mask driven from a `[batch, q, kv]` buffer, used to
    /// exercise the fast path's per-row mask application.
    #[cfg(feature = "mlas")]
    struct DenseKeyMask<'a> {
        data: &'a [f32],
        q_seq: usize,
        kv_seq: usize,
    }
    #[cfg(feature = "mlas")]
    impl KeyMask for DenseKeyMask<'_> {
        fn at(&self, b: usize, i: usize, j: usize) -> f32 {
            self.data[(b * self.q_seq + i) * self.kv_seq + j]
        }
    }

    /// The MLAS-GEMM fast path must agree with the scalar reference to tight
    /// tolerance across the full mode matrix (GQA, scale placement, softcap,
    /// bias, mask, causal, decode `Sq=1` and prefill shapes). GEMM reorders the
    /// accumulation, so this is a tolerance — not byte — check.
    #[cfg(feature = "mlas")]
    #[test]
    fn fast_path_matches_scalar_reference() {
        struct Shape {
            name: &'static str,
            batch: usize,
            nq: usize,
            nkv: usize,
            sq: usize,
            tk: usize,
            dh: usize,
            dv: usize,
            causal: bool,
            past: usize,
            softcap: Option<f32>,
            split_sqrt: bool,
            with_bias: bool,
            with_mask: bool,
        }
        let shapes = [
            Shape {
                name: "mha-prefill",
                batch: 2,
                nq: 4,
                nkv: 4,
                sq: 7,
                tk: 7,
                dh: 8,
                dv: 8,
                causal: false,
                past: 0,
                softcap: None,
                split_sqrt: false,
                with_bias: false,
                with_mask: false,
            },
            Shape {
                name: "mha-causal",
                batch: 1,
                nq: 3,
                nkv: 3,
                sq: 6,
                tk: 6,
                dh: 5,
                dv: 5,
                causal: true,
                past: 0,
                softcap: None,
                split_sqrt: false,
                with_bias: false,
                with_mask: false,
            },
            Shape {
                name: "gqa",
                batch: 2,
                nq: 8,
                nkv: 2,
                sq: 5,
                tk: 5,
                dh: 4,
                dv: 4,
                causal: false,
                past: 0,
                softcap: None,
                split_sqrt: false,
                with_bias: false,
                with_mask: false,
            },
            Shape {
                name: "mqa-decode",
                batch: 2,
                nq: 6,
                nkv: 1,
                sq: 1,
                tk: 9,
                dh: 8,
                dv: 8,
                causal: false,
                past: 8,
                softcap: None,
                split_sqrt: false,
                with_bias: false,
                with_mask: false,
            },
            Shape {
                name: "cross-diff-dv",
                batch: 1,
                nq: 2,
                nkv: 2,
                sq: 4,
                tk: 6,
                dh: 5,
                dv: 3,
                causal: false,
                past: 0,
                softcap: None,
                split_sqrt: false,
                with_bias: true,
                with_mask: false,
            },
            Shape {
                name: "softcap",
                batch: 1,
                nq: 2,
                nkv: 2,
                sq: 5,
                tk: 5,
                dh: 6,
                dv: 6,
                causal: false,
                past: 0,
                softcap: Some(30.0),
                split_sqrt: false,
                with_bias: false,
                with_mask: false,
            },
            Shape {
                name: "split-sqrt-mask",
                batch: 2,
                nq: 3,
                nkv: 3,
                sq: 4,
                tk: 5,
                dh: 7,
                dv: 7,
                causal: false,
                past: 0,
                softcap: None,
                split_sqrt: true,
                with_bias: false,
                with_mask: true,
            },
            Shape {
                name: "causal-past-decode",
                batch: 1,
                nq: 4,
                nkv: 4,
                sq: 1,
                tk: 12,
                dh: 8,
                dv: 8,
                causal: true,
                past: 11,
                softcap: None,
                split_sqrt: false,
                with_bias: false,
                with_mask: false,
            },
        ];

        for sh in &shapes {
            let q = fill(sh.batch * sh.nq * sh.sq * sh.dh, 1 + sh.sq as u64);
            let k = fill(sh.batch * sh.nkv * sh.tk * sh.dh, 2 + sh.tk as u64);
            let v = fill(sh.batch * sh.nkv * sh.tk * sh.dv, 3 + sh.dv as u64);
            let scale = 1.0 / (sh.dh as f32).sqrt();
            let t = SdpaTensors {
                q: &q,
                k: &k,
                v: &v,
                batch: sh.batch,
                num_heads: sh.nq,
                num_kv_heads: sh.nkv,
                q_seq: sh.sq,
                kv_seq: sh.tk,
                head_size: sh.dh,
                v_head_size: sh.dv,
            };
            let cfg = SdpaConfig {
                scale: if sh.split_sqrt {
                    ScaleMode::SplitSqrt(scale)
                } else {
                    ScaleMode::PostDot(scale)
                },
                softcap: sh.softcap,
                causal: sh.causal,
                past_seq: sh.past,
                causal_fill: f32::MIN,
            };
            let bias_data = fill(sh.batch * sh.nq * sh.sq * sh.tk, 7);
            let mask_data: Vec<f32> = fill(sh.batch * sh.sq * sh.tk, 9)
                .into_iter()
                .map(|x| if x < -0.5 { -1.0e9 } else { 0.0 })
                .collect();
            let no_bias = NoBias;
            let bc_bias = BroadcastBias::new(&bias_data, [sh.batch, sh.nq, sh.sq, sh.tk]);
            let bias: &dyn AttnBias = if sh.with_bias { &bc_bias } else { &no_bias };
            let no_mask = NoMask;
            let dm = DenseKeyMask {
                data: &mask_data,
                q_seq: sh.sq,
                kv_seq: sh.tk,
            };
            let mask: &dyn KeyMask = if sh.with_mask { &dm } else { &no_mask };

            let out_len = sh.batch * sh.nq * sh.sq * sh.dv;
            let mut y_scalar = vec![0.0f32; out_len];
            sdpa_f32_scalar(&t, &cfg, bias, mask, &mut y_scalar, None);
            let mut y_fast = vec![0.0f32; out_len];
            sdpa_f32_fast(&t, &cfg, bias, mask, &mut y_fast);

            let mut max_abs = 0.0f32;
            let mut worst = 0.0f32;
            for (a, b) in y_fast.iter().zip(y_scalar.iter()) {
                let abs = (a - b).abs();
                max_abs = max_abs.max(abs);
                // Combined tolerance `atol + rtol·|ref|` (numpy allclose style),
                // so a near-zero reference doesn't inflate a pure relative ratio.
                worst = worst.max(abs - (1e-5 + 1e-4 * b.abs()));
            }
            // GEMM reassociation over these small K (≤8) reduces the f32 dot to
            // a few ULP; softmax + P·V keep it bounded. atol 1e-5 / rtol 1e-4
            // matches the crate's ORT-parity tolerances with margin.
            assert!(
                worst <= 0.0,
                "shape {}: fast vs scalar exceeds atol+rtol (max_abs={max_abs:e})",
                sh.name
            );
        }
    }

    /// Provisional fast-vs-scalar throughput probe (run with
    /// `cargo test -p onnx-runtime-ep-cpu --features mlas -- --ignored --nocapture
    /// sdpa_fast_provisional_bench`). Numbers are PROVISIONAL — the CI host is
    /// shared, so treat the printed speedups as indicative, not authoritative.
    #[cfg(feature = "mlas")]
    #[test]
    #[ignore = "provisional microbench; shared host — run manually with --nocapture"]
    fn sdpa_fast_provisional_bench() {
        use std::time::Instant;

        fn run(name: &str, batch: usize, nq: usize, nkv: usize, sq: usize, tk: usize, dh: usize) {
            let q = fill(batch * nq * sq * dh, 11);
            let k = fill(batch * nkv * tk * dh, 22);
            let v = fill(batch * nkv * tk * dh, 33);
            let t = SdpaTensors {
                q: &q,
                k: &k,
                v: &v,
                batch,
                num_heads: nq,
                num_kv_heads: nkv,
                q_seq: sq,
                kv_seq: tk,
                head_size: dh,
                v_head_size: dh,
            };
            let cfg = SdpaConfig {
                scale: ScaleMode::PostDot(1.0 / (dh as f32).sqrt()),
                softcap: None,
                causal: sq > 1,
                past_seq: tk - sq,
                causal_fill: f32::MIN,
            };
            let out_len = batch * nq * sq * dh;
            let mut y = vec![0.0f32; out_len];

            let iters = 20;
            // Warm up + time scalar.
            sdpa_f32_scalar(&t, &cfg, &NoBias, &NoMask, &mut y, None);
            let t0 = Instant::now();
            for _ in 0..iters {
                sdpa_f32_scalar(&t, &cfg, &NoBias, &NoMask, &mut y, None);
            }
            let scalar = t0.elapsed().as_secs_f64() / iters as f64;
            // Warm up + time fast.
            sdpa_f32_fast(&t, &cfg, &NoBias, &NoMask, &mut y);
            let t1 = Instant::now();
            for _ in 0..iters {
                sdpa_f32_fast(&t, &cfg, &NoBias, &NoMask, &mut y);
            }
            let fast = t1.elapsed().as_secs_f64() / iters as f64;
            println!(
                "[sdpa-bench PROVISIONAL] {name:>16}: scalar {:>9.3} ms  fast {:>9.3} ms  speedup {:>5.2}x",
                scalar * 1e3,
                fast * 1e3,
                scalar / fast
            );
        }

        println!("[sdpa-bench] PROVISIONAL numbers — shared host, treat as indicative only");
        run("prefill", 1, 32, 32, 512, 512, 128);
        run("decode", 1, 32, 32, 1, 513, 128);
        run("gqa-prefill", 1, 32, 8, 512, 512, 128);
    }
}
