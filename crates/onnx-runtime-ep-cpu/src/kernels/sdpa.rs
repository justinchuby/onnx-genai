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
//! * **Optional QK score capture** — the pre-softmax logits (`[B, Nq, Sq, Tk]`)
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
//! already f32 here); a future MLAS-GEMM / rayon-tiled fast path can replace the
//! `dot`/`·V` accumulations without touching the masking/softmax sequence.

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
pub trait AttnBias {
    fn at(&self, b: usize, head: usize, i: usize, j: usize) -> f32;
}

/// Per-`(batch, query, key)` additive key mask (head-independent, as in ORT's
/// key-padding masks). Return `0.0` to keep a key, a large negative fill to
/// mask it.
pub trait KeyMask {
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

/// Optional QK score capture target for ops that emit `qk_matmul_output`.
///
/// Holds the pre-softmax logits in `[batch, num_heads, q_seq, kv_seq]` order.
pub struct QkCapture<'a> {
    pub scores: &'a mut [f32],
}

/// Run scaled-dot-product attention over `t`, writing the context into `y`
/// (`[batch, num_heads, q_seq, v_head_size]`, `BNSH`).
///
/// `bias` and `mask` are applied additively in that order (pass [`NoBias`] /
/// [`NoMask`] to skip). When `qk` is `Some`, the pre-softmax logits are copied
/// out before the softmax. See the module docs for the exact numerical
/// sequence — it is a bit-for-bit factoring of the standalone MHA loop.
pub fn sdpa_f32(
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
                // scores[j] = scale·(Q·Kᵀ) [+softcap] + bias + mask [→ causal].
                for (j, sc) in scores.iter_mut().enumerate() {
                    let k_base = ((b * num_kv_heads + kv_n) * kv_seq + j) * head_size;
                    let mut acc = 0.0f32;
                    for p in 0..head_size {
                        acc += (q[q_base + p] * operand_scale) * (k[k_base + p] * operand_scale);
                    }
                    let mut s = acc * post_scale;
                    if let Some(softcap) = cfg.softcap {
                        s = softcap * (s / softcap).tanh();
                    }
                    s += bias.at(b, n, i, j);
                    s += mask.at(b, i, j);
                    if cfg.causal && (j as i64) > cfg.past_seq as i64 + i as i64 {
                        s = cfg.causal_fill;
                    }
                    *sc = s;
                }

                if let Some(cap) = qk.as_mut() {
                    let base = ((b * num_heads + n) * q_seq + i) * kv_seq;
                    cap.scores[base..base + kv_seq].copy_from_slice(&scores);
                }

                // Numerically-stable softmax (subtract row max, matching ORT's
                // MlasComputeSoftmax and this crate's softmax kernel).
                let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
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
        sdpa_f32(&t, &cfg, &NoBias, &NoMask, &mut y, None);
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
        sdpa_f32(&t, &cfg, &NoBias, &NoMask, &mut y, None);
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
        sdpa_f32(&t, &cfg, &NoBias, &NoMask, &mut y, None);
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
        sdpa_f32(
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
        sdpa_f32(
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
}
