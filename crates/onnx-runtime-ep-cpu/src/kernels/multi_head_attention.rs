//! `com.microsoft::MultiHeadAttention` (opset 1): scaled dot-product attention
//! taking **separate** query/key/value inputs (unlike the packed-QKV
//! `com.microsoft::Attention`). This is the reference f32 CPU kernel, written
//! for **numerical parity with onnxruntime 1.26.0's** unfused CPU MHA path
//! (`contrib_ops/cpu/bert/attention_cpu_base.h` +
//! `multihead_attention.cc`); correctness first, optimization later.
//!
//! This kernel is a **thin adapter over the shared [`super::sdpa`] core**: it
//! parses/validates the MHA inputs, normalizes the various Q/K/V layouts,
//! projects the optional bias, and concatenates any past-KV cache into dense
//! `BNSH` f32 buffers, then hands the `QKᵀ → scale → +bias → +mask → softmax →
//! ·V` math to [`super::sdpa::sdpa_f32`]. The core reproduces this kernel's
//! original loop bit-for-bit, so the ORT goldens remain the regression gate.
//!
//! ## Semantics (matching ORT's `ComputeAttentionProbs`)
//!
//! ```text
//! scores = scale · (Q · Kᵀ)              # scale defaults to 1/sqrt(qk_head_size),
//!                                        # applied to the raw dot product
//! scores = scores + attention_bias       # optional additive float bias
//! scores = scores + mask                 # key_padding_mask → mask_filter_value,
//!                                        # causal (unidirectional) → f32::MIN
//! probs  = softmax(scores, axis=-1)      # numerically stable (subtract row max)
//! out    = probs · V
//! ```
//!
//! `scale` multiplies the raw `Q·Kᵀ` (ORT folds it into the GEMM `alpha`), and
//! the mask is **additive**: padding-masked positions get `mask_filter_value`
//! (default `-10000`), causal-masked positions get `f32::MIN` — both matching
//! ORT exactly so a fully-padded row yields ORT's (near-uniform) distribution
//! rather than a special-cased zero row.
//!
//! ## Supported layouts (Whisper-tiny decoder self- & cross-attention)
//!
//! * `Q_K_V_BSNH`: query `(B, S, D)`, key `(B, L, D)`, value `(B, L, D_v)`
//!   (all rank 3). `D = num_heads·head_size`, `D_v = num_heads·v_head_size`;
//!   `v_head_size` may differ from `head_size`.
//! * `Q_K_V_BSNH_BNSH_BNSH` (cross-attention): query `(B, S, D)`, key/value
//!   already transposed to `(B, num_heads, L, H)` / `(B, num_heads, L, H_v)`.
//!   Per ORT, key/value bias is assumed zero in this layout (only the query
//!   bias slice is applied).
//! * Optional in-op KV cache: `past_key`/`past_value` `(B, num_heads, P, H)`
//!   concatenated in front along the sequence axis → `present_key`/
//!   `present_value` `(B, num_heads, P+L, H)`. Causal masking uses `S > 1` as
//!   ORT does, so an incremental decode step (`S = 1`) attends the whole cache.
//!
//! ## Optional inputs (ORT slot order)
//!
//! `query(0)`, `key(1)`, `value(2)`, `bias(3)`, `key_padding_mask(4)`,
//! `attention_bias(5)`, `past_key(6)`, `past_value(7)`. `key_padding_mask`
//! (int32/int64) supports the `(B, T)`, `(B)`/`(3B+2)` and `(B, S, T)` forms.
//!
//! ## Rejected (clean error, never a silent miscompute)
//!
//! Packed-QKV (rank-5 query) and packed-KV (rank-5 key), the
//! `DecoderMaskedMultiHeadAttention` extras (`past_sequence_length`,
//! `cache_indirection`), and non-f32 Q/K/V all error actionably.

use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{DataType, Node};

use super::check_arity;
use super::sdpa::{
    AttnBias, BroadcastBias, KeyMask, NoBias, QkCapture, ScaleMode, SdpaConfig, SdpaTensors,
    sdpa_f32,
};
use crate::dtype::{to_dense_f32_widen, write_dense_f32_narrow};

/// f32 `MultiHeadAttention` kernel carrying the resolved attributes.
pub struct MultiHeadAttentionKernel {
    num_heads: usize,
    /// Explicit score scale; `None` → default `1/sqrt(qk_head_size)`.
    scale: Option<f32>,
    /// Additive fill for padding-masked positions (ORT default `-10000`).
    mask_filter_value: f32,
    /// Apply a causal (lower-triangular) mask when the query length is `> 1`.
    unidirectional: bool,
}

/// Factory for [`MultiHeadAttentionKernel`], reading the contrib-op attributes.
pub struct MultiHeadAttentionFactory;

impl KernelFactory for MultiHeadAttentionFactory {
    fn create(&self, node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let num_heads = node
            .attr("num_heads")
            .and_then(|a| a.as_int())
            .ok_or_else(|| {
                EpError::KernelFailed(
                    "MultiHeadAttention: missing required `num_heads` attribute".into(),
                )
            })?;
        if num_heads <= 0 {
            return Err(EpError::KernelFailed(format!(
                "MultiHeadAttention: num_heads must be > 0, got {num_heads}"
            )));
        }
        let scale = node.attr("scale").and_then(|a| a.as_float());
        let mask_filter_value = node
            .attr("mask_filter_value")
            .and_then(|a| a.as_float())
            .unwrap_or(-10000.0);
        let unidirectional = node
            .attr("unidirectional")
            .and_then(|a| a.as_int())
            .unwrap_or(0)
            == 1;
        Ok(Box::new(MultiHeadAttentionKernel {
            num_heads: num_heads as usize,
            scale,
            mask_filter_value,
            unidirectional,
        }))
    }
}

/// `[batch, heads, seq, dim]` dense f32 buffer with its resolved dims.
struct Bnsh {
    data: Vec<f32>,
    batch: usize,
    heads: usize,
    seq: usize,
    dim: usize,
}

impl Bnsh {
    #[inline]
    fn at(&self, b: usize, h: usize, s: usize, d: usize) -> f32 {
        self.data[((b * self.heads + h) * self.seq + s) * self.dim + d]
    }
}

/// Materialize a Q/K/V input into a dense `[batch, heads, seq, dim]` f32 buffer.
///
/// * A rank-4 input `(batch, heads, seq, dim)` is read as-is (already
///   transposed, as in cross-attention / past-KV decoding).
/// * A rank-3 input `(batch, seq, heads·dim)` is reshaped to
///   `(batch, seq, heads, dim)` and transposed to `(batch, heads, seq, dim)`.
///
/// When `bias` is provided (only in the rank-3 BSNH path), it is a per-hidden
/// slice of length `heads·dim` added elementwise (broadcast over batch/seq),
/// mirroring ORT's `MaybeTransposeToBNSHAndAddBias`.
fn load_bnsh(
    view: &TensorView,
    num_heads: usize,
    bias: Option<&[f32]>,
    name: &str,
) -> Result<Bnsh> {
    match view.shape.len() {
        4 => {
            let (batch, heads, seq, dim) =
                (view.shape[0], view.shape[1], view.shape[2], view.shape[3]);
            if heads != num_heads {
                return Err(EpError::KernelFailed(format!(
                    "MultiHeadAttention: rank-4 {name} dim 1 ({heads}) must equal num_heads \
                     ({num_heads})"
                )));
            }
            let mut data = to_dense_f32_widen("MultiHeadAttention", view)?.into_owned();
            if let Some(bias) = bias {
                for b in 0..batch {
                    for h in 0..heads {
                        for s in 0..seq {
                            let base = ((b * heads + h) * seq + s) * dim;
                            for d in 0..dim {
                                data[base + d] += bias[h * dim + d];
                            }
                        }
                    }
                }
            }
            Ok(Bnsh {
                data,
                batch,
                heads,
                seq,
                dim,
            })
        }
        3 => {
            if num_heads == 0 {
                return Err(EpError::KernelFailed(
                    "MultiHeadAttention: num_heads must be > 0".into(),
                ));
            }
            let (batch, seq, hidden) = (view.shape[0], view.shape[1], view.shape[2]);
            if !hidden.is_multiple_of(num_heads) {
                return Err(EpError::KernelFailed(format!(
                    "MultiHeadAttention: rank-3 {name} hidden size {hidden} is not divisible by \
                     num_heads {num_heads}"
                )));
            }
            let dim = hidden / num_heads;
            let src = to_dense_f32_widen("MultiHeadAttention", view)?;
            let mut data = vec![0.0f32; batch * num_heads * seq * dim];
            for b in 0..batch {
                for s in 0..seq {
                    for h in 0..num_heads {
                        for d in 0..dim {
                            let src_i = ((b * seq + s) * num_heads + h) * dim + d;
                            let dst_i = ((b * num_heads + h) * seq + s) * dim + d;
                            let mut v = src[src_i];
                            if let Some(bias) = bias {
                                v += bias[h * dim + d];
                            }
                            data[dst_i] = v;
                        }
                    }
                }
            }
            Ok(Bnsh {
                data,
                batch,
                heads: num_heads,
                seq,
                dim,
            })
        }
        other => Err(EpError::KernelFailed(format!(
            "MultiHeadAttention: {name} must be rank 3 (B,S,hidden) or rank 4 (B,N,S,head_size); \
             packed layouts (rank {other}) are unsupported"
        ))),
    }
}

/// Concatenate an optional past cache `[B, N, P, dim]` in front of `cur`
/// `[B, N, L, dim]` along the sequence axis → `[B, N, P+L, dim]`.
fn concat_cache(past: Option<&Bnsh>, cur: &Bnsh, name: &str) -> Result<Bnsh> {
    let Some(past) = past else {
        return Ok(Bnsh {
            data: cur.data.clone(),
            batch: cur.batch,
            heads: cur.heads,
            seq: cur.seq,
            dim: cur.dim,
        });
    };
    if past.batch != cur.batch || past.heads != cur.heads || past.dim != cur.dim {
        return Err(EpError::KernelFailed(format!(
            "MultiHeadAttention: past_{name} dims (b={},h={},d={}) incompatible with current \
             (b={},h={},d={})",
            past.batch, past.heads, past.dim, cur.batch, cur.heads, cur.dim
        )));
    }
    let (batch, heads, dim) = (cur.batch, cur.heads, cur.dim);
    let total = past.seq + cur.seq;
    let mut data = vec![0.0f32; batch * heads * total * dim];
    for b in 0..batch {
        for h in 0..heads {
            for d in 0..dim {
                for j in 0..past.seq {
                    let dst = ((b * heads + h) * total + j) * dim + d;
                    data[dst] = past.at(b, h, j, d);
                }
                for j in 0..cur.seq {
                    let dst = ((b * heads + h) * total + past.seq + j) * dim + d;
                    data[dst] = cur.at(b, h, j, d);
                }
            }
        }
    }
    Ok(Bnsh {
        data,
        batch,
        heads,
        seq: total,
        dim,
    })
}

/// Resolved key-padding mask, matching ORT's `GetMaskType` + `PrepareMask`.
enum PadMask {
    None,
    /// `(batch, total_seq)` raw mask; `> 0` keeps, else `mask_filter_value`.
    Raw2d(Vec<i64>),
    /// `(batch)` right-padding key lengths; keys `j >= len[b]` are masked.
    KeyLen {
        lens: Vec<i64>,
    },
    /// `(3·batch + 2)` right+left padding: `len[b]` and `start[b]`.
    KeyLenStart {
        lens: Vec<i64>,
        starts: Vec<i64>,
    },
    /// `(batch, q_seq, total_seq)` per-position mask; `> 0` keeps.
    Mask3d(Vec<i64>),
}

impl PadMask {
    /// Resolve the `key_padding_mask` input, validating its shape against the
    /// derived `(batch, q_seq, total_seq)` exactly like ORT's `GetMaskType`.
    fn resolve(view: &TensorView, batch: usize, q_seq: usize, total_seq: usize) -> Result<PadMask> {
        let dims = view.shape;
        let raw = super::to_dense_i64(view)?;
        match *dims {
            [b] if b == batch => Ok(PadMask::KeyLen { lens: raw }),
            [b] if b == 3 * batch + 2 => {
                // Layout: [key_len(B), ... , start(B) ...]; ORT reads mask[b]
                // (right pad end) and mask[b + batch] (left pad start).
                let lens = raw[..batch].to_vec();
                let starts = raw[batch..2 * batch].to_vec();
                Ok(PadMask::KeyLenStart { lens, starts })
            }
            [b, t] if b == batch && t == total_seq => Ok(PadMask::Raw2d(raw)),
            [b, s, t] if b == batch && s == q_seq && t == total_seq => Ok(PadMask::Mask3d(raw)),
            _ => Err(EpError::KernelFailed(format!(
                "MultiHeadAttention: unsupported key_padding_mask shape {dims:?} for \
                 batch={batch}, q_seq={q_seq}, total_seq={total_seq} (expected (B,), (3B+2), \
                 (B, T) or (B, S, T))"
            ))),
        }
    }

    /// Additive mask bias for logical position `(b, i, j)` — `0.0` to keep,
    /// `filter` for a padding-masked key. Matches ORT's `PrepareMask` (before
    /// the causal override, which the caller applies).
    #[inline]
    fn bias(
        &self,
        b: usize,
        i: usize,
        j: usize,
        q_seq: usize,
        total_seq: usize,
        filter: f32,
    ) -> f32 {
        match self {
            PadMask::None => 0.0,
            PadMask::Raw2d(m) => keep_or_filter(m[b * total_seq + j] > 0, filter),
            PadMask::KeyLen { lens } => {
                let end = lens[b].clamp(0, total_seq as i64);
                keep_or_filter((j as i64) < end, filter)
            }
            PadMask::KeyLenStart { lens, starts } => {
                let end = lens[b].clamp(0, total_seq as i64);
                let start = starts[b].clamp(0, total_seq as i64);
                keep_or_filter((j as i64) < end && (j as i64) >= start, filter)
            }
            PadMask::Mask3d(m) => keep_or_filter(m[(b * q_seq + i) * total_seq + j] > 0, filter),
        }
    }
}

#[inline]
fn keep_or_filter(keep: bool, filter: f32) -> f32 {
    if keep { 0.0 } else { filter }
}

/// Adapts an MHA [`PadMask`] to the shared core's [`KeyMask`] hook by carrying
/// the derived `(q_seq, total_seq)` and the resolved `mask_filter_value`.
struct MhaKeyMask<'a> {
    mask: &'a PadMask,
    q_seq: usize,
    total_seq: usize,
    filter: f32,
}

impl KeyMask for MhaKeyMask<'_> {
    #[inline]
    fn at(&self, b: usize, i: usize, j: usize) -> f32 {
        self.mask
            .bias(b, i, j, self.q_seq, self.total_seq, self.filter)
    }
}

impl Kernel for MultiHeadAttentionKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        check_arity("MultiHeadAttention", inputs, outputs, 3, 10, 1)?;

        // DecoderMaskedMultiHeadAttention extras are out of scope.
        if inputs.len() > 8 && !inputs[8].is_absent() {
            return Err(EpError::KernelFailed(
                "MultiHeadAttention: the `past_sequence_length` input (DecoderMaskedMHA) is not \
                 supported"
                    .into(),
            ));
        }
        if inputs.len() > 9 && !inputs[9].is_absent() {
            return Err(EpError::KernelFailed(
                "MultiHeadAttention: the `cache_indirection` input (DecoderMaskedMHA) is not \
                 supported"
                    .into(),
            ));
        }

        let query = &inputs[0];
        let key = &inputs[1];
        let value = &inputs[2];
        if query.shape.len() != 3 {
            return Err(EpError::KernelFailed(format!(
                "MultiHeadAttention: query must be rank 3 (B, S, hidden); packed-QKV layouts \
                 (rank {}) are unsupported",
                query.shape.len()
            )));
        }
        if key.is_absent() || value.is_absent() {
            return Err(EpError::KernelFailed(
                "MultiHeadAttention: separate key and value inputs are required (packed QKV/KV is \
                 unsupported)"
                    .into(),
            ));
        }
        if key.shape.len() != value.shape.len() || !(key.shape.len() == 3 || key.shape.len() == 4) {
            return Err(EpError::KernelFailed(format!(
                "MultiHeadAttention: key and value must both be rank 3 (B, L, hidden) or rank 4 \
                 (B, N, L, head_size), got key rank {}, value rank {}",
                key.shape.len(),
                value.shape.len()
            )));
        }

        for (name, v) in [("query", query), ("key", key), ("value", value)] {
            if !matches!(
                v.dtype,
                DataType::Float32 | DataType::Float16 | DataType::BFloat16
            ) {
                return Err(EpError::KernelFailed(format!(
                    "MultiHeadAttention: {name} dtype {:?} not supported (f32 kernel)",
                    v.dtype
                )));
            }
        }

        let num_heads = self.num_heads;
        let batch = query.shape[0];
        let q_seq = query.shape[1];
        let q_hidden = query.shape[2];
        if !q_hidden.is_multiple_of(num_heads) {
            return Err(EpError::KernelFailed(format!(
                "MultiHeadAttention: query hidden {q_hidden} not divisible by num_heads {num_heads}"
            )));
        }
        let head_size = q_hidden / num_heads;
        let is_cross_bnsh = key.shape.len() == 4;

        // v_hidden_size (= num_heads · v_head_size) sizes output-0's hidden dim.
        let v_hidden = if is_cross_bnsh {
            num_heads * value.shape[3]
        } else {
            value.shape[2]
        };

        // Optional bias `(D + D + D_v)` split into per-projection slices. In the
        // rank-4 (BNSH) key/value layout, ORT assumes zero key/value bias, so
        // only the query slice is applied.
        let bias_vec = if inputs.len() > 3 && !inputs[3].is_absent() {
            let bias = to_dense_f32_widen("MultiHeadAttention", &inputs[3])?.into_owned();
            let expected = 2 * q_hidden + v_hidden;
            if bias.len() != expected {
                return Err(EpError::KernelFailed(format!(
                    "MultiHeadAttention: bias length {} must equal 2*hidden + v_hidden = {expected}",
                    bias.len()
                )));
            }
            Some(bias)
        } else {
            None
        };
        let q_bias = bias_vec.as_ref().map(|b| &b[0..q_hidden]);
        let k_bias = bias_vec.as_ref().map(|b| &b[q_hidden..2 * q_hidden]);
        let v_bias = bias_vec
            .as_ref()
            .map(|b| &b[2 * q_hidden..2 * q_hidden + v_hidden]);

        let q = load_bnsh(query, num_heads, q_bias, "query")?;
        let (k_cur, v_cur) = if is_cross_bnsh {
            (
                load_bnsh(key, num_heads, None, "key")?,
                load_bnsh(value, num_heads, None, "value")?,
            )
        } else {
            (
                load_bnsh(key, num_heads, k_bias, "key")?,
                load_bnsh(value, num_heads, v_bias, "value")?,
            )
        };

        // Optional in-op KV cache (inputs 6 and 7), rank-4 (B, N, P, H).
        let has_past_key = inputs.len() > 6 && !inputs[6].is_absent();
        let has_past_value = inputs.len() > 7 && !inputs[7].is_absent();
        if has_past_key != has_past_value {
            return Err(EpError::KernelFailed(
                "MultiHeadAttention: past_key and past_value must be provided together".into(),
            ));
        }
        let past_key = if has_past_key {
            Some(load_bnsh(&inputs[6], num_heads, None, "past_key")?)
        } else {
            None
        };
        let past_value = if has_past_value {
            Some(load_bnsh(&inputs[7], num_heads, None, "past_value")?)
        } else {
            None
        };
        let past_seq = past_key.as_ref().map(|p| p.seq).unwrap_or(0);

        let key = concat_cache(past_key.as_ref(), &k_cur, "key")?;
        let value = concat_cache(past_value.as_ref(), &v_cur, "value")?;

        let total_seq = key.seq;
        let v_head_size = value.dim;
        if key.dim != head_size {
            return Err(EpError::KernelFailed(format!(
                "MultiHeadAttention: key head_size {} != query head_size {head_size}",
                key.dim
            )));
        }
        if value.seq != total_seq {
            return Err(EpError::KernelFailed(format!(
                "MultiHeadAttention: key seq {total_seq} != value seq {}",
                value.seq
            )));
        }
        if key.batch != batch || value.batch != batch {
            return Err(EpError::KernelFailed(
                "MultiHeadAttention: query, key, value must share the batch dimension".into(),
            ));
        }
        debug_assert_eq!(v_hidden, num_heads * v_head_size);

        // Optional key_padding_mask (input 4).
        let pad_mask = if inputs.len() > 4 && !inputs[4].is_absent() {
            PadMask::resolve(&inputs[4], batch, q_seq, total_seq)?
        } else {
            PadMask::None
        };

        // Optional attention_bias (input 5): additive float `(B|1, N|1, S, T)`.
        let attn_bias = if inputs.len() > 5 && !inputs[5].is_absent() {
            let m = &inputs[5];
            if m.shape.len() != 4 {
                return Err(EpError::KernelFailed(format!(
                    "MultiHeadAttention: attention_bias must be rank 4 (B|1, N|1, S, T), got rank {}",
                    m.shape.len()
                )));
            }
            let (bd0, bd1, bd2, bd3) = (m.shape[0], m.shape[1], m.shape[2], m.shape[3]);
            if !(bd0 == batch || bd0 == 1)
                || !(bd1 == num_heads || bd1 == 1)
                || bd2 != q_seq
                || bd3 != total_seq
            {
                return Err(EpError::KernelFailed(format!(
                    "MultiHeadAttention: attention_bias shape {:?} incompatible with (B|1={batch}, \
                     N|1={num_heads}, S={q_seq}, T={total_seq})",
                    m.shape
                )));
            }
            Some((
                to_dense_f32_widen("MultiHeadAttention", m)?.into_owned(),
                [bd0, bd1, bd2, bd3],
            ))
        } else {
            None
        };

        let scale = self
            .scale
            .unwrap_or_else(|| 1.0 / (head_size as f32).sqrt());
        // Matches ORT: causal only when unidirectional AND the query spans more
        // than one token (an incremental decode step attends the whole cache).
        let causal = self.unidirectional && q_seq > 1;
        let filter = self.mask_filter_value;

        let mut y = vec![0.0f32; batch * num_heads * q_seq * v_head_size];
        let want_qk = outputs.len() >= 4;
        let mut qk_out = if want_qk {
            vec![0.0f32; batch * num_heads * q_seq * total_seq]
        } else {
            Vec::new()
        };

        // Delegate the QKᵀ → scale → +bias → +mask → softmax → ·V math to the
        // shared SDPA core. MHA is plain multi-head (num_kv_heads == num_heads),
        // PostDot scale, no softcap; the layout/cache/bias normalization above
        // has already reduced Q/K/V to dense BNSH f32 buffers.
        let tensors = SdpaTensors {
            q: &q.data,
            k: &key.data,
            v: &value.data,
            batch,
            num_heads,
            num_kv_heads: num_heads,
            q_seq,
            kv_seq: total_seq,
            head_size,
            v_head_size,
        };
        let cfg = SdpaConfig {
            scale: ScaleMode::PostDot(scale),
            softcap: None,
            causal,
            past_seq,
            causal_fill: f32::MIN,
        };
        let bias_hook: &dyn AttnBias = match &attn_bias {
            Some((data, dims)) => &BroadcastBias::new(data, *dims),
            None => &NoBias,
        };
        let mask_hook = MhaKeyMask {
            mask: &pad_mask,
            q_seq,
            total_seq,
            filter,
        };
        let qk_cap = want_qk.then_some(QkCapture {
            scores: &mut qk_out,
        });
        sdpa_f32(&tensors, &cfg, bias_hook, &mask_hook, &mut y, qk_cap);
        let mut y3 = vec![0.0f32; batch * q_seq * v_hidden];
        for b in 0..batch {
            for n in 0..num_heads {
                for s in 0..q_seq {
                    for c in 0..v_head_size {
                        let src = ((b * num_heads + n) * q_seq + s) * v_head_size + c;
                        let dst = (b * q_seq + s) * v_hidden + n * v_head_size + c;
                        y3[dst] = y[src];
                    }
                }
            }
        }
        write_dense_f32_narrow("MultiHeadAttention", &mut outputs[0], &y3)?;

        // present_key / present_value (outputs 1, 2), always 4D `(B, N, T, H)`.
        if outputs.len() >= 2 {
            write_dense_f32_narrow("MultiHeadAttention", &mut outputs[1], &key.data)?;
        }
        if outputs.len() >= 3 {
            write_dense_f32_narrow("MultiHeadAttention", &mut outputs[2], &value.data)?;
        }
        if want_qk {
            write_dense_f32_narrow("MultiHeadAttention", &mut outputs[3], &qk_out)?;
        }
        Ok(())
    }

    fn supports_strided_input(&self, _input_idx: usize) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::testutil::Owned;
    use onnx_runtime_ir::{Attribute, NodeId};

    fn kernel(attrs: &[(&str, Attribute)]) -> Result<Box<dyn Kernel>> {
        let mut node = Node::new(NodeId(0), "MultiHeadAttention", vec![], vec![]);
        node.domain = "com.microsoft".to_string();
        for (name, value) in attrs {
            node.attributes.insert((*name).to_string(), value.clone());
        }
        MultiHeadAttentionFactory.create(&node, &[])
    }

    #[test]
    fn factory_requires_positive_num_heads() {
        assert!(kernel(&[]).is_err(), "missing num_heads must be rejected");
        assert!(
            kernel(&[("num_heads", Attribute::Int(0))]).is_err(),
            "num_heads=0 must be rejected"
        );
        assert!(kernel(&[("num_heads", Attribute::Int(2))]).is_ok());
    }

    #[test]
    fn rejects_packed_qkv_rank5_query() {
        let k = kernel(&[("num_heads", Attribute::Int(2))]).unwrap();
        let q = Owned::f32(&[1, 2, 2, 3, 4], &[0.0; 48]);
        let key = Owned::f32(&[1, 2, 8], &[0.0; 16]);
        let val = Owned::f32(&[1, 2, 8], &[0.0; 16]);
        let mut out = Owned::f32(&[1, 2, 8], &[0.0; 16]);
        let err = k.execute(&[q.view(), key.view(), val.view()], &mut [out.view_mut()]);
        assert!(err.is_err(), "rank-5 packed query must be rejected");
    }

    #[test]
    fn rejects_mismatched_key_value_rank() {
        let k = kernel(&[("num_heads", Attribute::Int(2))]).unwrap();
        let q = Owned::f32(&[1, 2, 8], &[0.0; 16]);
        let key = Owned::f32(&[1, 2, 8], &[0.0; 16]); // rank 3
        let val = Owned::f32(&[1, 2, 2, 4], &[0.0; 16]); // rank 4
        let mut out = Owned::f32(&[1, 2, 8], &[0.0; 16]);
        let err = k.execute(&[q.view(), key.view(), val.view()], &mut [out.view_mut()]);
        assert!(err.is_err(), "mismatched key/value ranks must be rejected");
    }

    #[test]
    fn basic_self_attention_is_finite_and_shaped() {
        // Single head, S=2, H=2: verify against a hand-rolled SDPA reference.
        let k = kernel(&[("num_heads", Attribute::Int(1))]).unwrap();
        let q = vec![0.1f32, 0.2, 0.3, 0.4];
        let key = vec![0.5f32, 0.6, 0.7, 0.8];
        let v = vec![1.0f32, 2.0, 3.0, 4.0];
        let qt = Owned::f32(&[1, 2, 2], &q);
        let kt = Owned::f32(&[1, 2, 2], &key);
        let vt = Owned::f32(&[1, 2, 2], &v);
        let mut out = Owned::f32(&[1, 2, 2], &[0.0; 4]);
        k.execute(&[qt.view(), kt.view(), vt.view()], &mut [out.view_mut()])
            .unwrap();

        // Reference: scale = 1/sqrt(2); softmax(scale*Q·Kᵀ)·V, single head.
        let scale = 1.0f32 / 2.0f32.sqrt();
        let mut expected = vec![0.0f32; 4];
        for i in 0..2 {
            let mut scores = [0.0f32; 2];
            for (j, sc) in scores.iter_mut().enumerate() {
                *sc = scale * (q[i * 2] * key[j * 2] + q[i * 2 + 1] * key[j * 2 + 1]);
            }
            let m = scores[0].max(scores[1]);
            let e: Vec<f32> = scores.iter().map(|s| (s - m).exp()).collect();
            let sum: f32 = e.iter().sum();
            for c in 0..2 {
                expected[i * 2 + c] = (e[0] * v[c] + e[1] * v[2 + c]) / sum;
            }
        }
        let got = out.to_f32();
        for (a, b) in got.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-6, "got {got:?}, expected {expected:?}");
        }
    }
}
