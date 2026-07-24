//! `com.microsoft::Attention` (contrib, opset 1): the **packed-QKV** BERT/GPT
//! attention op. Unlike `com.microsoft::MultiHeadAttention` (separate,
//! already-projected Q/K/V), this op takes the raw hidden state plus a merged
//! Q/K/V projection weight and does the input projection itself, then runs the
//! same scaled-dot-product attention. Distinct from the standard
//! `ai.onnx::Attention` (see [`super::attention`]).
//!
//! This kernel is a **thin adapter over the shared [`super::sdpa`] core**: it
//! (a) projects `input @ weights + bias` and splits the result into contiguous
//! `BNSH` f32 Q/K/V, (b) concatenates any `past` KV cache, (c) resolves the
//! `mask_index` / `attention_bias`, (d) hands the `QKᵀ → scale → +bias → +mask
//! → [causal] → softmax → ·V` math to [`super::sdpa::sdpa_f32`], and (e) writes
//! the `present` KV cache out. The SDPA numerics are shared verbatim with MHA
//! and `ai.onnx::Attention`, so the ORT goldens remain the regression gate.
//!
//! ## Contract (matching onnxruntime 1.26.0's `contrib_ops/cpu/bert/attention.cc`)
//!
//! Inputs (ORT slot order):
//! * `input(0)` — `(batch, seq, input_hidden)`; the hidden state to project.
//! * `weights(1)` — merged Q/K/V weight `(input_hidden, q_hidden + k_hidden +
//!   v_hidden)`; `q_hidden == k_hidden`.
//! * `bias(2)` — projection bias `(q_hidden + k_hidden + v_hidden)`. Required by
//!   ORT's CPU kernel (it dereferences the shape unconditionally), so a clean
//!   error is raised when absent.
//! * `mask_index(3)` — optional int32/int64 mask; supports raw `(B, T)` /
//!   `(B, S, T)` (0 = masked), and index forms `(B)` (right-pad key length),
//!   `(2B)` (right-pad end + left-pad start) and `(3B+2)` (end positions in the
//!   leading `B`, matching ORT's fall-through).
//! * `past(4)` — optional KV cache `(2, batch, num_heads, past_seq, head_size)`;
//!   `past[0]` = key, `past[1]` = value. Requires `q_hidden == v_hidden` and is
//!   mutually exclusive with `attention_bias` (as in ORT).
//! * `attention_bias(5)` — optional additive float `(B|1, N|1, S, T)`.
//!
//! Attributes: `num_heads` (required), `scale` (default `1/sqrt(head_size)`,
//! applied to the raw `Q·Kᵀ`), `mask_filter_value` (default `-10000`),
//! `unidirectional` (causal when set AND `seq > 1`, matching ORT), and
//! `qkv_hidden_sizes` (`[q, k, v]`; defaults to `weights.dim1 / 3` each).
//!
//! Outputs: `output(0)` `(batch, seq, v_hidden)`, optional `present(1)`
//! `(2, batch, num_heads, total_seq, head_size)`.
//!
//! ## Rejected (clean error, never a silent miscompute)
//!
//! `do_rotary` (fuse RotaryEmbedding separately, as ORT instructs),
//! `past_present_share_buffer` / `past_sequence_length` (the
//! DecoderMaskedMultiHeadAttention buffer-sharing path), 4D Megatron masks
//! (ORT's CPU kernel is `ORT_NOT_IMPLEMENTED` for these), and non-float
//! input/weights/bias.

use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{DataType, Node};

use super::check_arity;
use super::sdpa::{
    AttnBias, BroadcastBias, KeyMask, NoBias, ScaleMode, SdpaConfig, SdpaTensors, sdpa_f32,
};
use crate::dtype::{to_dense_f32_widen, write_dense_f32_narrow};

/// f32 `com.microsoft::Attention` kernel carrying the resolved attributes.
pub struct MsftAttentionKernel {
    num_heads: usize,
    /// Explicit score scale; `None` → default `1/sqrt(head_size)`.
    scale: Option<f32>,
    /// Additive fill for padding-masked positions (ORT default `-10000`).
    mask_filter_value: f32,
    /// Apply a causal (lower-triangular) mask when the query length is `> 1`.
    unidirectional: bool,
    /// Optional `[q_hidden, k_hidden, v_hidden]` override.
    qkv_hidden_sizes: Option<[usize; 3]>,
}

/// Factory for [`MsftAttentionKernel`], reading the contrib-op attributes.
pub struct MsftAttentionFactory;

impl KernelFactory for MsftAttentionFactory {
    fn create(&self, node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let num_heads = node
            .attr("num_heads")
            .and_then(|a| a.as_int())
            .ok_or_else(|| {
                EpError::KernelFailed("Attention: missing required `num_heads` attribute".into())
            })?;
        if num_heads <= 0 {
            return Err(EpError::KernelFailed(format!(
                "Attention: num_heads must be > 0, got {num_heads}"
            )));
        }

        // Unsupported variants are rejected up front (typed error, not a panic).
        if node.attr("do_rotary").and_then(|a| a.as_int()).unwrap_or(0) == 1 {
            return Err(EpError::KernelFailed(
                "Attention: `do_rotary` is not supported (fuse MHA + RotaryEmbedding instead, as \
                 ORT's CPU kernel requires)"
                    .into(),
            ));
        }
        if node
            .attr("past_present_share_buffer")
            .and_then(|a| a.as_int())
            .unwrap_or(0)
            == 1
        {
            return Err(EpError::KernelFailed(
                "Attention: `past_present_share_buffer` (DecoderMaskedMHA buffer sharing) is not \
                 supported"
                    .into(),
            ));
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

        let qkv_hidden_sizes = match node.attr("qkv_hidden_sizes").and_then(|a| a.as_ints()) {
            Some(sizes) => {
                if sizes.len() != 3 {
                    return Err(EpError::KernelFailed(format!(
                        "Attention: qkv_hidden_sizes must have 3 elements, got {}",
                        sizes.len()
                    )));
                }
                let n = num_heads;
                for s in sizes {
                    if *s <= 0 || s % n != 0 {
                        return Err(EpError::KernelFailed(format!(
                            "Attention: qkv_hidden_sizes element {s} must be > 0 and divisible by \
                             num_heads {num_heads}"
                        )));
                    }
                }
                if sizes[0] != sizes[1] {
                    return Err(EpError::KernelFailed(
                        "Attention: qkv_hidden_sizes[0] (q_hidden) must equal qkv_hidden_sizes[1] \
                         (k_hidden)"
                            .into(),
                    ));
                }
                Some([sizes[0] as usize, sizes[1] as usize, sizes[2] as usize])
            }
            None => None,
        };

        Ok(Box::new(MsftAttentionKernel {
            num_heads: num_heads as usize,
            scale,
            mask_filter_value,
            unidirectional,
            qkv_hidden_sizes,
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

/// Concatenate an optional past cache `[B, N, P, dim]` in front of `cur`
/// `[B, N, L, dim]` along the sequence axis → `[B, N, P+L, dim]`.
fn concat_cache(past: Option<&Bnsh>, cur: &Bnsh) -> Bnsh {
    let Some(past) = past else {
        return Bnsh {
            data: cur.data.clone(),
            batch: cur.batch,
            heads: cur.heads,
            seq: cur.seq,
            dim: cur.dim,
        };
    };
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
    Bnsh {
        data,
        batch,
        heads,
        seq: total,
        dim,
    }
}

/// Resolved `mask_index`, matching ORT's `CheckMask` + `PrepareMask`.
enum PadMask {
    None,
    /// `(batch, total_seq)` raw mask; `> 0` keeps, else `mask_filter_value`.
    Raw2d(Vec<i64>),
    /// `(batch, q_seq, total_seq)` per-position mask; `> 0` keeps.
    Mask3d(Vec<i64>),
    /// Right-side padding key length per batch; keys `j >= len[b]` are masked
    /// (the `(B)` and `(3B+2)` forms — ORT only reads the leading `B` for the
    /// latter).
    KeyLen {
        lens: Vec<i64>,
    },
    /// `(2B)` right-pad end + left-pad start: keys `j < start` or `j >= end`
    /// are masked.
    KeyLenStart {
        lens: Vec<i64>,
        starts: Vec<i64>,
    },
}

impl PadMask {
    fn resolve(view: &TensorView, batch: usize, q_seq: usize, total_seq: usize) -> Result<PadMask> {
        let dims = view.shape;
        let raw = super::to_dense_i64(view)?;
        match *dims {
            [b] if b == batch => Ok(PadMask::KeyLen { lens: raw }),
            [b] if b == 2 * batch => {
                let lens = raw[..batch].to_vec();
                let starts = raw[batch..2 * batch].to_vec();
                Ok(PadMask::KeyLenStart { lens, starts })
            }
            // ORT's PrepareMask falls through the `(3B+2)` form to the plain
            // right-pad path, using only the leading `B` end positions.
            [b] if b == 3 * batch + 2 => Ok(PadMask::KeyLen {
                lens: raw[..batch].to_vec(),
            }),
            [b, t] if b == batch && t == total_seq => Ok(PadMask::Raw2d(raw)),
            // ORT treats a `(B,1)`/`(1,1)` 2D mask as a no-op ("dummy") mask.
            [b, 1] if b == batch || b == 1 => Ok(PadMask::None),
            [b, s, t] if b == batch && s == q_seq && t == total_seq => Ok(PadMask::Mask3d(raw)),
            _ => Err(EpError::KernelFailed(format!(
                "Attention: unsupported mask_index shape {dims:?} for batch={batch}, \
                 q_seq={q_seq}, total_seq={total_seq} (expected (B,), (2B), (3B+2), (B, T) or \
                 (B, S, T); 4D Megatron masks are not supported on CPU)"
            ))),
        }
    }

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
            PadMask::Mask3d(m) => keep_or_filter(m[(b * q_seq + i) * total_seq + j] > 0, filter),
            PadMask::KeyLen { lens } => {
                let end = lens[b].clamp(0, total_seq as i64);
                keep_or_filter((j as i64) < end, filter)
            }
            PadMask::KeyLenStart { lens, starts } => {
                let end = lens[b].clamp(0, total_seq as i64);
                let start = starts[b].clamp(0, total_seq as i64);
                keep_or_filter((j as i64) < end && (j as i64) >= start, filter)
            }
        }
    }
}

#[inline]
fn keep_or_filter(keep: bool, filter: f32) -> f32 {
    if keep { 0.0 } else { filter }
}

/// Adapts a [`PadMask`] to the shared core's [`KeyMask`] hook.
struct MsftAttnKeyMask<'a> {
    mask: &'a PadMask,
    q_seq: usize,
    total_seq: usize,
    filter: f32,
}

impl KeyMask for MsftAttnKeyMask<'_> {
    #[inline]
    fn at(&self, b: usize, i: usize, j: usize) -> f32 {
        self.mask
            .bias(b, i, j, self.q_seq, self.total_seq, self.filter)
    }
}

impl MsftAttentionKernel {
    /// Project `input @ weights + bias` and split into contiguous `BNSH` Q/K/V.
    ///
    /// `input` is `(batch, seq, input_hidden)`, `weights` is
    /// `(input_hidden, q_hidden + k_hidden + v_hidden)` row-major, `bias` (when
    /// present) is that same width. Column `c < q_hidden` feeds Q head
    /// `c / head_size`; `[q_hidden, 2*q_hidden)` feeds K; the remainder feeds V.
    #[allow(clippy::too_many_arguments)]
    fn project(
        &self,
        input: &[f32],
        weights: &[f32],
        bias: Option<&[f32]>,
        batch: usize,
        seq: usize,
        input_hidden: usize,
        q_hidden: usize,
        v_hidden: usize,
    ) -> (Bnsh, Bnsh, Bnsh) {
        let n = self.num_heads;
        let head_size = q_hidden / n;
        let v_head_size = v_hidden / n;
        let d_t = 2 * q_hidden + v_hidden;

        let mut q = vec![0.0f32; batch * n * seq * head_size];
        let mut k = vec![0.0f32; batch * n * seq * head_size];
        let mut v = vec![0.0f32; batch * n * seq * v_head_size];

        for b in 0..batch {
            for s in 0..seq {
                let in_base = (b * seq + s) * input_hidden;
                for c in 0..d_t {
                    let mut acc = bias.map_or(0.0, |bi| bi[c]);
                    for d in 0..input_hidden {
                        acc += input[in_base + d] * weights[d * d_t + c];
                    }
                    if c < q_hidden {
                        let h = c / head_size;
                        let hh = c % head_size;
                        q[((b * n + h) * seq + s) * head_size + hh] = acc;
                    } else if c < 2 * q_hidden {
                        let cc = c - q_hidden;
                        let h = cc / head_size;
                        let hh = cc % head_size;
                        k[((b * n + h) * seq + s) * head_size + hh] = acc;
                    } else {
                        let cc = c - 2 * q_hidden;
                        let h = cc / v_head_size;
                        let hh = cc % v_head_size;
                        v[((b * n + h) * seq + s) * v_head_size + hh] = acc;
                    }
                }
            }
        }

        (
            Bnsh {
                data: q,
                batch,
                heads: n,
                seq,
                dim: head_size,
            },
            Bnsh {
                data: k,
                batch,
                heads: n,
                seq,
                dim: head_size,
            },
            Bnsh {
                data: v,
                batch,
                heads: n,
                seq,
                dim: v_head_size,
            },
        )
    }
}

impl Kernel for MsftAttentionKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        check_arity("Attention", inputs, outputs, 2, 7, 1)?;

        // DecoderMaskedMHA buffer-sharing extra is out of scope.
        if inputs.len() > 6 && !inputs[6].is_absent() {
            return Err(EpError::KernelFailed(
                "Attention: the `past_sequence_length` input (past_present_share_buffer) is not \
                 supported"
                    .into(),
            ));
        }

        let input = &inputs[0];
        let weights = &inputs[1];
        if input.shape.len() != 3 {
            return Err(EpError::KernelFailed(format!(
                "Attention: input must be rank 3 (B, S, input_hidden), got rank {}",
                input.shape.len()
            )));
        }
        if weights.is_absent() || weights.shape.len() != 2 {
            return Err(EpError::KernelFailed(
                "Attention: weights must be a present rank-2 tensor (input_hidden, \
                 q_hidden + k_hidden + v_hidden)"
                    .into(),
            ));
        }
        for (name, v) in [("input", input), ("weights", weights)] {
            if !matches!(
                v.dtype,
                DataType::Float32 | DataType::Float16 | DataType::BFloat16
            ) {
                return Err(EpError::KernelFailed(format!(
                    "Attention: {name} dtype {:?} not supported (f32 kernel)",
                    v.dtype
                )));
            }
        }

        let num_heads = self.num_heads;
        let batch = input.shape[0];
        let seq = input.shape[1];
        let input_hidden = input.shape[2];
        if weights.shape[0] != input_hidden {
            return Err(EpError::KernelFailed(format!(
                "Attention: weights dim 0 ({}) must equal input hidden size {input_hidden}",
                weights.shape[0]
            )));
        }
        let d_t = weights.shape[1];

        // Resolve q/k/v hidden sizes (explicit override or the default split).
        let (q_hidden, v_hidden) = match self.qkv_hidden_sizes {
            Some([q, k, v]) => {
                if q + k + v != d_t {
                    return Err(EpError::KernelFailed(format!(
                        "Attention: qkv_hidden_sizes sum ({}) must equal weights dim 1 ({d_t})",
                        q + k + v
                    )));
                }
                (q, v)
            }
            None => {
                if !d_t.is_multiple_of(3) {
                    return Err(EpError::KernelFailed(format!(
                        "Attention: weights dim 1 ({d_t}) is not divisible by 3; supply \
                         qkv_hidden_sizes for asymmetric Q/K/V"
                    )));
                }
                (d_t / 3, d_t / 3)
            }
        };
        if !q_hidden.is_multiple_of(num_heads) || !v_hidden.is_multiple_of(num_heads) {
            return Err(EpError::KernelFailed(format!(
                "Attention: q_hidden {q_hidden} and v_hidden {v_hidden} must both be divisible by \
                 num_heads {num_heads}"
            )));
        }
        let head_size = q_hidden / num_heads;
        let v_head_size = v_hidden / num_heads;

        // bias (input 2) — required by ORT's CPU kernel.
        if inputs.len() < 3 || inputs[2].is_absent() {
            return Err(EpError::KernelFailed(
                "Attention: the projection `bias` input is required".into(),
            ));
        }
        let bias = to_dense_f32_widen("Attention", &inputs[2])?.into_owned();
        if bias.len() != d_t {
            return Err(EpError::KernelFailed(format!(
                "Attention: bias length {} must equal weights dim 1 ({d_t})",
                bias.len()
            )));
        }

        let input_data = to_dense_f32_widen("Attention", input)?;
        let weights_data = to_dense_f32_widen("Attention", weights)?;

        let (q, k_cur, v_cur) = self.project(
            &input_data,
            &weights_data,
            Some(&bias),
            batch,
            seq,
            input_hidden,
            q_hidden,
            v_hidden,
        );

        // Optional past KV cache (input 4): (2, B, N, P, H).
        let has_past = inputs.len() > 4 && !inputs[4].is_absent();
        let has_attn_bias = inputs.len() > 5 && !inputs[5].is_absent();
        if has_past && has_attn_bias {
            return Err(EpError::KernelFailed(
                "Attention: `past` and `attention_bias` cannot both be provided (matching ORT)"
                    .into(),
            ));
        }
        let (past_key, past_value) = if has_past {
            if q_hidden != v_hidden {
                return Err(EpError::KernelFailed(
                    "Attention: `past` state requires q_hidden == v_hidden".into(),
                ));
            }
            let pv = &inputs[4];
            if pv.shape.len() != 5 || pv.shape[0] != 2 {
                return Err(EpError::KernelFailed(format!(
                    "Attention: past must be rank 5 (2, B, N, P, H), got shape {:?}",
                    pv.shape
                )));
            }
            let (pb, pn, pp, ph) = (pv.shape[1], pv.shape[2], pv.shape[3], pv.shape[4]);
            if pb != batch || pn != num_heads || ph != head_size {
                return Err(EpError::KernelFailed(format!(
                    "Attention: past shape (2, {pb}, {pn}, {pp}, {ph}) incompatible with (2, \
                     B={batch}, N={num_heads}, P, H={head_size})"
                )));
            }
            let dense = to_dense_f32_widen("Attention", pv)?;
            let chunk = batch * num_heads * pp * head_size;
            let make = |off: usize| Bnsh {
                data: dense[off..off + chunk].to_vec(),
                batch,
                heads: num_heads,
                seq: pp,
                dim: head_size,
            };
            (Some(make(0)), Some(make(chunk)))
        } else {
            (None, None)
        };
        let past_seq = past_key.as_ref().map(|p| p.seq).unwrap_or(0);

        let key = concat_cache(past_key.as_ref(), &k_cur);
        let value = concat_cache(past_value.as_ref(), &v_cur);
        let total_seq = key.seq;

        // Optional mask_index (input 3).
        let pad_mask = if inputs.len() > 3 && !inputs[3].is_absent() {
            PadMask::resolve(&inputs[3], batch, seq, total_seq)?
        } else {
            PadMask::None
        };

        // Optional attention_bias (input 5): additive float (B|1, N|1, S, T).
        let attn_bias = if has_attn_bias {
            let m = &inputs[5];
            if m.shape.len() != 4 {
                return Err(EpError::KernelFailed(format!(
                    "Attention: attention_bias must be rank 4 (B|1, N|1, S, T), got rank {}",
                    m.shape.len()
                )));
            }
            let (bd0, bd1, bd2, bd3) = (m.shape[0], m.shape[1], m.shape[2], m.shape[3]);
            if !(bd0 == batch || bd0 == 1)
                || !(bd1 == num_heads || bd1 == 1)
                || bd2 != seq
                || bd3 != total_seq
            {
                return Err(EpError::KernelFailed(format!(
                    "Attention: attention_bias shape {:?} incompatible with (B|1={batch}, \
                     N|1={num_heads}, S={seq}, T={total_seq})",
                    m.shape
                )));
            }
            Some((
                to_dense_f32_widen("Attention", m)?.into_owned(),
                [bd0, bd1, bd2, bd3],
            ))
        } else {
            None
        };

        let scale = self
            .scale
            .unwrap_or_else(|| 1.0 / (head_size as f32).sqrt());
        // Matches ORT: causal only when unidirectional AND seq > 1.
        let causal = self.unidirectional && seq > 1;
        let filter = self.mask_filter_value;

        let mut y = vec![0.0f32; batch * num_heads * seq * v_head_size];
        let tensors = SdpaTensors {
            q: &q.data,
            k: &key.data,
            v: &value.data,
            batch,
            num_heads,
            num_kv_heads: num_heads,
            q_seq: seq,
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
        let mask_hook = MsftAttnKeyMask {
            mask: &pad_mask,
            q_seq: seq,
            total_seq,
            filter,
        };
        sdpa_f32(&tensors, &cfg, bias_hook, &mask_hook, &mut y, None);

        // context (B, N, S, v_head_size) → output (B, S, v_hidden).
        let mut y3 = vec![0.0f32; batch * seq * v_hidden];
        for b in 0..batch {
            for n in 0..num_heads {
                for s in 0..seq {
                    for c in 0..v_head_size {
                        let src = ((b * num_heads + n) * seq + s) * v_head_size + c;
                        let dst = (b * seq + s) * v_hidden + n * v_head_size + c;
                        y3[dst] = y[src];
                    }
                }
            }
        }
        write_dense_f32_narrow("Attention", &mut outputs[0], &y3)?;

        // present (output 1): (2, B, N, T, H) = concat(past, current) K then V.
        if outputs.len() >= 2 {
            let chunk = batch * num_heads * total_seq * head_size;
            let mut present = vec![0.0f32; 2 * chunk];
            present[..chunk].copy_from_slice(&key.data);
            present[chunk..].copy_from_slice(&value.data);
            write_dense_f32_narrow("Attention", &mut outputs[1], &present)?;
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
        let mut node = Node::new(NodeId(0), "Attention", vec![], vec![]);
        node.domain = "com.microsoft".to_string();
        for (name, value) in attrs {
            node.attributes.insert((*name).to_string(), value.clone());
        }
        MsftAttentionFactory.create(&node, &[])
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
    fn factory_rejects_do_rotary_and_share_buffer() {
        assert!(
            kernel(&[
                ("num_heads", Attribute::Int(2)),
                ("do_rotary", Attribute::Int(1)),
            ])
            .is_err(),
            "do_rotary=1 must be rejected"
        );
        assert!(
            kernel(&[
                ("num_heads", Attribute::Int(2)),
                ("past_present_share_buffer", Attribute::Int(1)),
            ])
            .is_err(),
            "past_present_share_buffer=1 must be rejected"
        );
    }

    #[test]
    fn rejects_missing_bias() {
        let k = kernel(&[("num_heads", Attribute::Int(1))]).unwrap();
        let input = Owned::f32(&[1, 2, 4], &[0.0; 8]);
        let weights = Owned::f32(&[4, 12], &[0.0; 48]);
        let mut out = Owned::f32(&[1, 2, 4], &[0.0; 8]);
        let err = k.execute(&[input.view(), weights.view()], &mut [out.view_mut()]);
        assert!(err.is_err(), "missing bias must be rejected");
    }

    #[test]
    fn basic_packed_qkv_matches_hand_rolled() {
        // 1 head, input_hidden=2, S=2, head_size=2, v_head_size=2.
        let k = kernel(&[("num_heads", Attribute::Int(1))]).unwrap();
        let input = vec![0.1f32, 0.2, 0.3, 0.4]; // (1,2,2)
        // weights (2, 6): columns [Q(2) | K(2) | V(2)].
        let weights: Vec<f32> = (0..12).map(|x| (x as f32) * 0.05 - 0.2).collect();
        let bias: Vec<f32> = (0..6).map(|x| (x as f32) * 0.01).collect();
        let it = Owned::f32(&[1, 2, 2], &input);
        let wt = Owned::f32(&[2, 6], &weights);
        let bt = Owned::f32(&[6], &bias);
        let mut out = Owned::f32(&[1, 2, 2], &[0.0; 4]);
        k.execute(&[it.view(), wt.view(), bt.view()], &mut [out.view_mut()])
            .unwrap();

        // Hand-rolled reference: project, then single-head SDPA.
        let s = 2usize;
        let (hidden, vh) = (2usize, 2usize);
        let proj = |tok: usize, col: usize| -> f32 {
            let mut acc = bias[col];
            for d in 0..2 {
                acc += input[tok * 2 + d] * weights[d * 6 + col];
            }
            acc
        };
        let mut q = vec![0.0f32; s * hidden];
        let mut kk = vec![0.0f32; s * hidden];
        let mut vv = vec![0.0f32; s * vh];
        for t in 0..s {
            for c in 0..hidden {
                q[t * hidden + c] = proj(t, c);
                kk[t * hidden + c] = proj(t, hidden + c);
            }
            for c in 0..vh {
                vv[t * vh + c] = proj(t, 2 * hidden + c);
            }
        }
        let scale = 1.0f32 / (hidden as f32).sqrt();
        let mut expected = vec![0.0f32; s * vh];
        for i in 0..s {
            let mut scores = [0.0f32; 2];
            for (j, sc) in scores.iter_mut().enumerate() {
                *sc = scale
                    * (q[i * hidden] * kk[j * hidden] + q[i * hidden + 1] * kk[j * hidden + 1]);
            }
            let m = scores[0].max(scores[1]);
            let e: Vec<f32> = scores.iter().map(|x| (x - m).exp()).collect();
            let sum: f32 = e.iter().sum();
            for c in 0..vh {
                expected[i * vh + c] = (e[0] * vv[c] + e[1] * vv[vh + c]) / sum;
            }
        }
        let got = out.to_f32();
        for (a, b) in got.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-6, "got {got:?}, expected {expected:?}");
        }
    }
}
