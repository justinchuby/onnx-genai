//! Standard `ai.onnx::Attention` (opset 23–26): scaled dot-product attention
//! (SDPA) with multi-head / grouped-query head sharing, an optional additive or
//! boolean attention mask, causal masking, and an in-op KV cache
//! (`past_key`/`past_value` → `present_key`/`present_value`).
//!
//! This is the *standard* ONNX operator, distinct from the private
//! `com.microsoft::FusedAttention` fusion node (see [`super::fused_attention`]),
//! which only reproduces the plain `MatMul → scale → [+mask] → Softmax →
//! MatMul` core the optimizer fuses. Standard `Attention` is a richer op: it
//! reshapes 3D `(batch, seq, hidden)` inputs into heads, supports GQA/MQA head
//! sharing, concatenates a past KV cache, offset-aware causal masking, softcap,
//! and emits up to four outputs (`Y`, `present_key`, `present_value`,
//! `qk_matmul_output`).
//!
//! ## Semantics (per the spec's applied pattern)
//!
//! ```text
//! scores = (Q·√scale) · (K·√scale)ᵀ      # √scale folded into each operand so
//!                                        # extreme magnitudes don't overflow;
//!                                        # scale defaults to 1/sqrt(head_size)
//! scores = softcap · tanh(scores/softcap)  # only when softcap != 0
//! scores = scores + attn_bias            # attn_mask (add/-inf) and causal mask
//! probs  = softmax(scores, axis=-1)      # numerically stable; fully-masked → 0
//! Y      = probs · V
//! ```
//!
//! ## Versioning (opset 23 vs 24–26)
//!
//! `Attention` was added at opset 23 and revised at opset 24 (no newer version
//! exists, so a single opset-24 kernel serves model opsets 24, 25 and 26). The
//! one semantic delta handled per registered `since_version`:
//!
//! * `nonpad_kv_seqlen` (7th input) — an external-cache per-batch valid-token
//!   count — is honored for v24+ and rejected for v23 (it did not exist there).
//!
//! `qk_matmul_output_mode` has the **same** meaning in both versions (the opset
//! 23 and 24 schema descriptions are identical): `0` = raw QK, `1` = after
//! softcap (before mask), `2` = after mask+softcap, `3` = after softmax.
//!
//! ## Supported vs. unimplemented
//!
//! * dtype: **f32 only** for v1 (matches the crate's other reference kernels;
//!   f16/bf16 is a follow-up — see the crate dtype-coverage effort). Non-f32
//!   Q/K/V error actionably.
//! * `qk_matmul_output_mode`: modes **0, 1, 2, 3** implemented per spec; any
//!   other value errors.

use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{DataType, Node};

use super::{check_arity, to_dense_f32, write_dense_f32};

/// f32 standard-`Attention` kernel carrying the resolved attributes.
pub struct AttentionKernel {
    /// Explicit score scale; `None` → default `1/sqrt(head_size)`.
    scale: Option<f32>,
    is_causal: bool,
    q_num_heads: Option<usize>,
    kv_num_heads: Option<usize>,
    qk_matmul_output_mode: i64,
    /// Softcap value; `0.0` disables it.
    softcap: f32,
    /// The registered opset version this kernel serves (23, or 24 for 24–26).
    /// Controls `nonpad_kv_seqlen` acceptance (opset 24+ only).
    since_version: u32,
}

/// Factory for [`AttentionKernel`], reading the standard-`Attention` attributes.
/// `since_version` selects the opset semantics (23 vs 24–26).
pub struct AttentionFactory {
    pub since_version: u32,
}

impl KernelFactory for AttentionFactory {
    fn create(&self, node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let scale = node.attr("scale").and_then(|a| a.as_float());
        let is_causal = node.attr("is_causal").and_then(|a| a.as_int()).unwrap_or(0) != 0;
        let q_num_heads = node
            .attr("q_num_heads")
            .and_then(|a| a.as_int())
            .map(|v| v as usize);
        let kv_num_heads = node
            .attr("kv_num_heads")
            .and_then(|a| a.as_int())
            .map(|v| v as usize);
        let qk_matmul_output_mode = node
            .attr("qk_matmul_output_mode")
            .and_then(|a| a.as_int())
            .unwrap_or(0);
        let softcap = node
            .attr("softcap")
            .and_then(|a| a.as_float())
            .unwrap_or(0.0);
        if !(0..=3).contains(&qk_matmul_output_mode) {
            return Err(EpError::KernelFailed(format!(
                "Attention: qk_matmul_output_mode {qk_matmul_output_mode} is not supported \
                 (only 0, 1, 2, 3 are implemented)"
            )));
        }
        Ok(Box::new(AttentionKernel {
            scale,
            is_causal,
            q_num_heads,
            kv_num_heads,
            qk_matmul_output_mode,
            softcap,
            since_version: self.since_version,
        }))
    }
}

/// `[batch, heads, seq, dim]` dense f32 buffer with its resolved dims.
struct Bhsd {
    data: Vec<f32>,
    batch: usize,
    heads: usize,
    seq: usize,
    dim: usize,
}

impl Bhsd {
    #[inline]
    fn at(&self, b: usize, h: usize, s: usize, d: usize) -> f32 {
        self.data[((b * self.heads + h) * self.seq + s) * self.dim + d]
    }
}

/// Materialize a Q/K/V input into a dense `[batch, heads, seq, dim]` f32 buffer.
///
/// A 4D input `(batch, heads, seq, dim)` is read as-is. A 3D input
/// `(batch, seq, heads·dim)` is reshaped to `(batch, seq, heads, dim)` and
/// transposed to `(batch, heads, seq, dim)`; `num_heads` (from the
/// `q_num_heads`/`kv_num_heads` attributes) is required and must divide the
/// hidden size.
fn to_bhsd(view: &TensorView, name: &str, num_heads: Option<usize>) -> Result<Bhsd> {
    let shape = view.shape;
    match shape.len() {
        4 => {
            let (batch, heads, seq, dim) = (shape[0], shape[1], shape[2], shape[3]);
            let data = to_dense_f32(view)?;
            Ok(Bhsd {
                data,
                batch,
                heads,
                seq,
                dim,
            })
        }
        3 => {
            let heads = num_heads.ok_or_else(|| {
                EpError::KernelFailed(format!(
                    "Attention: 3D {name} input requires the corresponding \
                     q_num_heads/kv_num_heads attribute"
                ))
            })?;
            if heads == 0 {
                return Err(EpError::KernelFailed(format!(
                    "Attention: {name} num_heads must be > 0"
                )));
            }
            let (batch, seq, hidden) = (shape[0], shape[1], shape[2]);
            if hidden % heads != 0 {
                return Err(EpError::KernelFailed(format!(
                    "Attention: 3D {name} hidden size {hidden} is not divisible by num_heads \
                     {heads}"
                )));
            }
            let dim = hidden / heads;
            // Source is contiguous over (batch, seq, heads, dim); transpose the
            // seq/heads axes to produce (batch, heads, seq, dim).
            let src = to_dense_f32(view)?;
            let mut data = vec![0.0f32; batch * heads * seq * dim];
            for b in 0..batch {
                for s in 0..seq {
                    for h in 0..heads {
                        for d in 0..dim {
                            let src_i = ((b * seq + s) * heads + h) * dim + d;
                            let dst_i = ((b * heads + h) * seq + s) * dim + d;
                            data[dst_i] = src[src_i];
                        }
                    }
                }
            }
            Ok(Bhsd {
                data,
                batch,
                heads,
                seq,
                dim,
            })
        }
        other => Err(EpError::KernelFailed(format!(
            "Attention: {name} must be rank 3 or 4, got rank {other}"
        ))),
    }
}

/// Concatenate an optional past cache `[batch, heads, past_seq, dim]` in front
/// of `cur` `[batch, heads, cur_seq, dim]` along the sequence axis, returning
/// the present cache `[batch, heads, past_seq+cur_seq, dim]`.
fn concat_cache(past: Option<&Bhsd>, cur: &Bhsd, name: &str) -> Result<Bhsd> {
    let Some(past) = past else {
        return Ok(Bhsd {
            data: cur.data.clone(),
            batch: cur.batch,
            heads: cur.heads,
            seq: cur.seq,
            dim: cur.dim,
        });
    };
    if past.batch != cur.batch || past.heads != cur.heads || past.dim != cur.dim {
        return Err(EpError::KernelFailed(format!(
            "Attention: past_{name} dims (b={},h={},d={}) incompatible with current \
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
    Ok(Bhsd {
        data,
        batch,
        heads,
        seq: total,
        dim,
    })
}

/// A resolved attention mask, materialized to a broadcastable
/// `[batch, q_heads, q_seq, total_seq]` bias generator.
enum Mask {
    None,
    /// Float mask (added to scores). Stored dense over its own shape with the
    /// leading dims to broadcast.
    Float {
        data: Vec<f32>,
        shape: Vec<usize>,
    },
    /// Boolean mask (`true` = keep). `false` positions contribute `-inf`.
    Bool {
        data: Vec<bool>,
        shape: Vec<usize>,
    },
}

impl Mask {
    /// The additive bias for logical index `(b, h, i, j)`; masked-out positions
    /// (bool `false`, or `j` past a short mask's last dim) yield `-inf`. A
    /// rank-0 (scalar) mask broadcasts to every score position.
    fn bias(&self, b: usize, h: usize, i: usize, j: usize, total_seq: usize) -> f32 {
        match self {
            Mask::None => 0.0,
            Mask::Float { data, shape } => Self::lookup_f32(data, shape, b, h, i, j, total_seq),
            Mask::Bool { data, shape } => {
                // A last dim shorter than total_seq is padded with -inf; a
                // rank-0 scalar mask has no last dim and applies everywhere.
                if !shape.is_empty() {
                    let last = shape[shape.len() - 1];
                    if j >= last && last < total_seq {
                        return f32::NEG_INFINITY;
                    }
                }
                if Self::lookup_bool(data, shape, b, h, i, j) {
                    0.0
                } else {
                    f32::NEG_INFINITY
                }
            }
        }
    }

    fn lookup_f32(
        data: &[f32],
        shape: &[usize],
        b: usize,
        h: usize,
        i: usize,
        j: usize,
        total_seq: usize,
    ) -> f32 {
        if !shape.is_empty() {
            let last = shape[shape.len() - 1];
            if j >= last && last < total_seq {
                return f32::NEG_INFINITY;
            }
        }
        data[Self::offset(shape, b, h, i, j)]
    }

    fn lookup_bool(data: &[bool], shape: &[usize], b: usize, h: usize, i: usize, j: usize) -> bool {
        data[Self::offset(shape, b, h, i, j)]
    }

    /// Row-major offset into a mask broadcastable to `[b, h, i, j]`. The mask
    /// may have rank 1..=4; missing leading dims broadcast, and any size-1 dim
    /// broadcasts.
    fn offset(shape: &[usize], b: usize, h: usize, i: usize, j: usize) -> usize {
        let full = [b, h, i, j];
        let rank = shape.len();
        let mut off = 0usize;
        for (k, &dim) in shape.iter().enumerate() {
            // Align the mask's trailing axes with [b, h, i, j].
            let logical = full[4 - rank + k];
            let idx = if dim == 1 { 0 } else { logical };
            off = off * dim + idx;
        }
        off
    }
}

impl Kernel for AttentionKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        check_arity("Attention", inputs, outputs, 3, 7, 1)?;

        let q_rank = inputs[0].shape.len();
        let q = to_bhsd(&inputs[0], "Q", self.q_num_heads)?;
        let k_cur = to_bhsd(&inputs[1], "K", self.kv_num_heads)?;
        let v_cur = to_bhsd(&inputs[2], "V", self.kv_num_heads)?;

        // Optional past KV cache (inputs 4 and 5). They must be used together.
        // Presence is decided by input-slot binding (a null "absent" view for an
        // omitted optional input), NOT by an empty shape — a genuinely present
        // rank-0 tensor also has an empty shape but must not be treated as
        // absent.
        let has_past_key = inputs.len() > 4 && !inputs[4].is_absent();
        let has_past_value = inputs.len() > 5 && !inputs[5].is_absent();
        if has_past_key != has_past_value {
            return Err(EpError::KernelFailed(
                "Attention: past_key and past_value must be provided together".into(),
            ));
        }
        let past_key = if has_past_key {
            Some(to_bhsd(&inputs[4], "past_key", self.kv_num_heads)?)
        } else {
            None
        };
        let past_value = if has_past_value {
            Some(to_bhsd(&inputs[5], "past_value", self.kv_num_heads)?)
        } else {
            None
        };
        let past_seq = past_key.as_ref().map(|p| p.seq).unwrap_or(0);

        // `nonpad_kv_seqlen` (7th input, opset 24+): per-batch count of valid
        // (non-padding) KV tokens, used when the KV cache lives outside the op.
        // It shifts the causal frontier by `nonpad_kv_seqlen[b] - q_seq` and is
        // mutually exclusive with an in-op past cache.
        let has_nonpad = inputs.len() > 6 && !inputs[6].is_absent();
        if has_nonpad && self.since_version < 24 {
            return Err(EpError::KernelFailed(
                "Attention: the optional `nonpad_kv_seqlen` input was added in opset 24 and is \
                 not valid for opset 23"
                    .into(),
            ));
        }
        if has_nonpad && (has_past_key || has_past_value) {
            return Err(EpError::KernelFailed(
                "Attention: `nonpad_kv_seqlen` must not be used together with past_key/past_value \
                 (external vs. in-op KV cache)"
                    .into(),
            ));
        }
        let nonpad_kv_seqlen: Option<Vec<i64>> = if has_nonpad {
            let seqlen = super::to_dense_i64(&inputs[6])?;
            if seqlen.len() != q.batch {
                return Err(EpError::KernelFailed(format!(
                    "Attention: nonpad_kv_seqlen length {} must equal batch_size {}",
                    seqlen.len(),
                    q.batch
                )));
            }
            Some(seqlen)
        } else {
            None
        };

        // present_key/value = concat(past, current) along the sequence axis.
        let key = concat_cache(past_key.as_ref(), &k_cur, "key")?;
        let value = concat_cache(past_value.as_ref(), &v_cur, "value")?;

        let batch = q.batch;
        let q_heads = q.heads;
        let q_seq = q.seq;
        let head_size = q.dim;
        let kv_heads = key.heads;
        let total_seq = key.seq;
        let v_head_size = value.dim;

        if key.dim != head_size {
            return Err(EpError::KernelFailed(format!(
                "Attention: Q head_size {head_size} != K head_size {}",
                key.dim
            )));
        }
        if value.seq != total_seq {
            return Err(EpError::KernelFailed(format!(
                "Attention: present_key seq {total_seq} != present_value seq {}",
                value.seq
            )));
        }
        if key.batch != batch || value.batch != batch {
            return Err(EpError::KernelFailed(
                "Attention: Q, K, V must share the batch dimension".into(),
            ));
        }
        if kv_heads == 0 || q_heads % kv_heads != 0 {
            return Err(EpError::KernelFailed(format!(
                "Attention: q_num_heads {q_heads} must be a positive multiple of kv_num_heads \
                 {kv_heads} (MHA/GQA/MQA)"
            )));
        }
        let group = q_heads / kv_heads;
        crate::trace::record_kernel_metrics(inputs, outputs, || {
            // Dominant SDPA work: QKᵀ and probability·V MACs (two FLOPs each),
            // plus the standalone softmax estimate over every score.
            let score_elements = (batch as u64)
                .saturating_mul(q_heads as u64)
                .saturating_mul(q_seq as u64)
                .saturating_mul(total_seq as u64);
            let qk_flops = score_elements
                .saturating_mul(head_size as u64)
                .saturating_mul(2);
            let pv_flops = score_elements
                .saturating_mul(v_head_size as u64)
                .saturating_mul(2);
            let softmax_flops = score_elements.saturating_mul(4).saturating_add(
                (batch as u64)
                    .saturating_mul(q_heads as u64)
                    .saturating_mul(q_seq as u64),
            );
            qk_flops
                .saturating_add(pv_flops)
                .saturating_add(softmax_flops)
        });

        let scale = self
            .scale
            .unwrap_or_else(|| 1.0 / (head_size as f32).sqrt());
        // Fold `sqrt(scale)` into each Q and K operand so the dot product is
        // `(Q·√scale)·(K·√scale)` rather than `scale·(Q·K)`. This matches the
        // spec's `Q*sqrt(scale)`, `K*sqrt(scale)` pattern and avoids overflowing
        // an intermediate `Q·Kᵀ` for extreme magnitudes.
        let sqrt_scale = scale.sqrt();

        // Resolve the attention mask (input 3), if present. Presence is decided
        // by input-slot binding, so a rank-0 (scalar) mask is honored rather
        // than mistaken for an omitted input.
        let mask = if inputs.len() > 3 && !inputs[3].is_absent() {
            let m = &inputs[3];
            match m.dtype {
                DataType::Bool => Mask::Bool {
                    data: super::to_dense_bytes(m)?.iter().map(|&b| b != 0).collect(),
                    shape: m.shape.to_vec(),
                },
                DataType::Float32 => Mask::Float {
                    data: to_dense_f32(m)?,
                    shape: m.shape.to_vec(),
                },
                other => {
                    return Err(EpError::KernelFailed(format!(
                        "Attention: attn_mask dtype {other:?} not supported (expected bool or f32)"
                    )));
                }
            }
        } else {
            Mask::None
        };

        let mut y = vec![0.0f32; batch * q_heads * q_seq * v_head_size];
        // qk_matmul_output buffer, produced only when a 4th output is present.
        let want_qk = outputs.len() >= 4;
        let mut qk_out = if want_qk {
            vec![0.0f32; batch * q_heads * q_seq * total_seq]
        } else {
            Vec::new()
        };

        // `qk_matmul_output_mode` has the same meaning across opsets 23–26 (the
        // schema descriptions are identical): 1 = after softcap (before mask),
        // 2 = after mask+softcap.
        let (softcap_mode, mask_mode) = (1, 2);

        let mut scores = vec![0.0f32; total_seq];
        for b in 0..batch {
            // Per-batch causal offset: query in-block index `i` attends key `j`
            // iff `j <= i + offset`. With an external cache the offset is
            // `nonpad_kv_seqlen[b] - q_seq`; with an in-op past cache it is
            // `past_seq`; otherwise 0. A negative offset fully masks leading
            // query rows (→ zero output rows).
            let offset: i64 = match &nonpad_kv_seqlen {
                Some(seqlen) => seqlen[b] - q_seq as i64,
                None => past_seq as i64,
            };
            // Per-batch padding frontier: with `nonpad_kv_seqlen`, keys at
            // `j >= nonpad_kv_seqlen[b]` are padding in the external KV cache
            // and must be masked to -inf REGARDLESS of causal mode. This
            // composes with (intersects) the causal frontier and `attn_mask`.
            let pad_limit: Option<i64> = nonpad_kv_seqlen.as_ref().map(|seqlen| seqlen[b]);
            for qh in 0..q_heads {
                let kvh = qh / group;
                for i in 0..q_seq {
                    // Stage 1: scaled Q·Kᵀ scores for this query row, with
                    // sqrt(scale) folded into each operand (overflow-safe).
                    for (j, sc) in scores.iter_mut().enumerate() {
                        let mut acc = 0.0f32;
                        for p in 0..head_size {
                            acc += (q.at(b, qh, i, p) * sqrt_scale)
                                * (key.at(b, kvh, j, p) * sqrt_scale);
                        }
                        *sc = acc;
                    }
                    // qk mode 0: raw (scaled) QK matmul output.
                    if want_qk && self.qk_matmul_output_mode == 0 {
                        let base = ((b * q_heads + qh) * q_seq + i) * total_seq;
                        qk_out[base..base + total_seq].copy_from_slice(&scores);
                    }

                    // Stage 2: softcap (before mask), applied when nonzero.
                    if self.softcap != 0.0 {
                        for sc in scores.iter_mut() {
                            *sc = self.softcap * (*sc / self.softcap).tanh();
                        }
                    }
                    // qk mode 1: after softcap, before mask.
                    if want_qk && self.qk_matmul_output_mode == softcap_mode {
                        let base = ((b * q_heads + qh) * q_seq + i) * total_seq;
                        qk_out[base..base + total_seq].copy_from_slice(&scores);
                    }

                    // Stage 3: attention mask + causal frontier (additive bias).
                    let causal_limit = i as i64 + offset;
                    for (j, sc) in scores.iter_mut().enumerate() {
                        // Padding mask: applies regardless of `is_causal`.
                        let is_pad = pad_limit.is_some_and(|limit| (j as i64) >= limit);
                        if is_pad {
                            *sc = f32::NEG_INFINITY;
                            continue;
                        }
                        if self.is_causal && (j as i64) > causal_limit {
                            *sc = f32::NEG_INFINITY;
                            continue;
                        }
                        let bias = mask.bias(b, qh, i, j, total_seq);
                        *sc += bias;
                    }
                    // qk mode 2: after mask+softcap, before softmax.
                    if want_qk && self.qk_matmul_output_mode == mask_mode {
                        let base = ((b * q_heads + qh) * q_seq + i) * total_seq;
                        qk_out[base..base + total_seq].copy_from_slice(&scores);
                    }

                    // Stage 4: numerically-stable softmax with a fully-masked
                    // row guard (all -inf → zero row, not NaN).
                    let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
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
                    // qk mode 3: post-softmax probabilities.
                    if want_qk && self.qk_matmul_output_mode == 3 {
                        let base = ((b * q_heads + qh) * q_seq + i) * total_seq;
                        qk_out[base..base + total_seq].copy_from_slice(&scores);
                    }

                    // Stage 5: Y = probs · V.
                    let y_base = ((b * q_heads + qh) * q_seq + i) * v_head_size;
                    for c in 0..v_head_size {
                        let mut acc = 0.0f32;
                        for (j, &p) in scores.iter().enumerate() {
                            acc += p * value.at(b, kvh, j, c);
                        }
                        y[y_base + c] = acc;
                    }
                }
            }
        }

        // Write Y, reshaping back to the rank of Q.
        if q_rank == 3 {
            // (batch, q_heads, q_seq, v_head_size) → (batch, q_seq, q_heads·v)
            let hidden = q_heads * v_head_size;
            let mut y3 = vec![0.0f32; batch * q_seq * hidden];
            for b in 0..batch {
                for h in 0..q_heads {
                    for s in 0..q_seq {
                        for c in 0..v_head_size {
                            let src = ((b * q_heads + h) * q_seq + s) * v_head_size + c;
                            let dst = (b * q_seq + s) * hidden + h * v_head_size + c;
                            y3[dst] = y[src];
                        }
                    }
                }
            }
            write_dense_f32(&mut outputs[0], &y3)?;
        } else {
            write_dense_f32(&mut outputs[0], &y)?;
        }

        // present_key / present_value (outputs 1 and 2), always 4D.
        if outputs.len() >= 2 {
            write_dense_f32(&mut outputs[1], &key.data)?;
        }
        if outputs.len() >= 3 {
            write_dense_f32(&mut outputs[2], &value.data)?;
        }
        if want_qk {
            write_dense_f32(&mut outputs[3], &qk_out)?;
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

    /// Naive reference SDPA oracle over dense `[batch, heads, seq, dim]` f32
    /// buffers, supporting GQA head sharing, an additive bias, and causal
    /// masking with a `past_seq` offset. Independent of the kernel's loops.
    #[allow(clippy::too_many_arguments)]
    fn reference(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        batch: usize,
        q_heads: usize,
        kv_heads: usize,
        q_seq: usize,
        total_seq: usize,
        head_size: usize,
        v_head_size: usize,
        scale: f32,
        is_causal: bool,
        causal_offset: i64,
        bias: impl Fn(usize, usize, usize, usize) -> f32,
    ) -> Vec<f32> {
        let group = q_heads / kv_heads;
        let mut out = vec![0.0f32; batch * q_heads * q_seq * v_head_size];
        for b in 0..batch {
            for qh in 0..q_heads {
                let kvh = qh / group;
                for i in 0..q_seq {
                    let mut scores = vec![0.0f32; total_seq];
                    for (j, sc) in scores.iter_mut().enumerate() {
                        let mut acc = 0.0f32;
                        for p in 0..head_size {
                            let qi = ((b * q_heads + qh) * q_seq + i) * head_size + p;
                            let kj = ((b * kv_heads + kvh) * total_seq + j) * head_size + p;
                            acc += q[qi] * k[kj];
                        }
                        let mut s = acc * scale + bias(b, qh, i, j);
                        if is_causal && (j as i64) > i as i64 + causal_offset {
                            s = f32::NEG_INFINITY;
                        }
                        *sc = s;
                    }
                    let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    if max == f32::NEG_INFINITY {
                        continue; // zero row
                    }
                    let mut sum = 0.0f32;
                    for sc in scores.iter_mut() {
                        *sc = (*sc - max).exp();
                        sum += *sc;
                    }
                    for sc in scores.iter_mut() {
                        *sc /= sum;
                    }
                    for c in 0..v_head_size {
                        let mut acc = 0.0f32;
                        for (j, &p) in scores.iter().enumerate() {
                            let vj = ((b * kv_heads + kvh) * total_seq + j) * v_head_size + c;
                            acc += p * v[vj];
                        }
                        out[((b * q_heads + qh) * q_seq + i) * v_head_size + c] = acc;
                    }
                }
            }
        }
        out
    }

    fn approx(a: &[f32], b: &[f32], atol: f32) {
        assert_eq!(
            a.len(),
            b.len(),
            "length mismatch: {} vs {}",
            a.len(),
            b.len()
        );
        for (i, (x, y)) in a.iter().zip(b).enumerate() {
            assert!(
                (x - y).abs() < atol,
                "element {i}: {x} vs {y}\n{a:?}\nvs\n{b:?}"
            );
        }
    }

    fn bsh_to_bhsd(src: &[f32], batch: usize, seq: usize, heads: usize, dim: usize) -> Vec<f32> {
        let mut dst = vec![0.0; src.len()];
        for b in 0..batch {
            for s in 0..seq {
                for h in 0..heads {
                    for d in 0..dim {
                        dst[((b * heads + h) * seq + s) * dim + d] =
                            src[((b * seq + s) * heads + h) * dim + d];
                    }
                }
            }
        }
        dst
    }

    fn bhsd_to_bsh(src: &[f32], batch: usize, heads: usize, seq: usize, dim: usize) -> Vec<f32> {
        let mut dst = vec![0.0; src.len()];
        for b in 0..batch {
            for h in 0..heads {
                for s in 0..seq {
                    for d in 0..dim {
                        dst[((b * seq + s) * heads + h) * dim + d] =
                            src[((b * heads + h) * seq + s) * dim + d];
                    }
                }
            }
        }
        dst
    }

    fn concat_bsh(
        first: &[f32],
        second: &[f32],
        batch: usize,
        first_seq: usize,
        second_seq: usize,
        hidden: usize,
    ) -> Vec<f32> {
        let total_seq = first_seq + second_seq;
        let mut dst = vec![0.0; batch * total_seq * hidden];
        for b in 0..batch {
            let first_src = b * first_seq * hidden;
            let second_src = b * second_seq * hidden;
            let dst_base = b * total_seq * hidden;
            dst[dst_base..dst_base + first_seq * hidden]
                .copy_from_slice(&first[first_src..first_src + first_seq * hidden]);
            dst[dst_base + first_seq * hidden..dst_base + total_seq * hidden]
                .copy_from_slice(&second[second_src..second_src + second_seq * hidden]);
        }
        dst
    }

    #[allow(clippy::too_many_arguments)]
    fn concat_bhsd(
        past: &[f32],
        current: &[f32],
        batch: usize,
        heads: usize,
        past_seq: usize,
        current_seq: usize,
        dim: usize,
    ) -> Vec<f32> {
        let total_seq = past_seq + current_seq;
        let mut dst = vec![0.0; batch * heads * total_seq * dim];
        for b in 0..batch {
            for h in 0..heads {
                for s in 0..past_seq {
                    let src = ((b * heads + h) * past_seq + s) * dim;
                    let out = ((b * heads + h) * total_seq + s) * dim;
                    dst[out..out + dim].copy_from_slice(&past[src..src + dim]);
                }
                for s in 0..current_seq {
                    let src = ((b * heads + h) * current_seq + s) * dim;
                    let out = ((b * heads + h) * total_seq + past_seq + s) * dim;
                    dst[out..out + dim].copy_from_slice(&current[src..src + dim]);
                }
            }
        }
        dst
    }

    fn conformance_values(len: usize, salt: usize) -> Vec<f32> {
        (0..len)
            .map(|i| (((i * 17 + salt * 29) % 97) as f32 - 48.0) / 128.0)
            .collect()
    }

    fn kernel(
        scale: Option<f32>,
        is_causal: bool,
        q_num_heads: Option<usize>,
        kv_num_heads: Option<usize>,
        qk_mode: i64,
        softcap: f32,
    ) -> AttentionKernel {
        kernel_v(
            24,
            scale,
            is_causal,
            q_num_heads,
            kv_num_heads,
            qk_mode,
            softcap,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn kernel_v(
        since_version: u32,
        scale: Option<f32>,
        is_causal: bool,
        q_num_heads: Option<usize>,
        kv_num_heads: Option<usize>,
        qk_mode: i64,
        softcap: f32,
    ) -> AttentionKernel {
        AttentionKernel {
            scale,
            is_causal,
            q_num_heads,
            kv_num_heads,
            qk_matmul_output_mode: qk_mode,
            softcap,
            since_version,
        }
    }

    /// An omitted optional input slot (null-backed placeholder view).
    fn absent() -> TensorView<'static> {
        TensorView::absent(DataType::Float32)
    }

    #[test]
    fn mha_4d_no_mask_matches_reference() {
        let (b, h, sq, sk, d, dv) = (2, 2, 3, 4, 5, 6);
        let q: Vec<f32> = (0..b * h * sq * d)
            .map(|i| (i as f32 * 0.1).sin())
            .collect();
        let k: Vec<f32> = (0..b * h * sk * d)
            .map(|i| (i as f32 * 0.07).cos())
            .collect();
        let v: Vec<f32> = (0..b * h * sk * dv)
            .map(|i| (i as f32 * 0.03) - 0.5)
            .collect();
        let scale = 0.3f32;

        let want = reference(
            &q,
            &k,
            &v,
            b,
            h,
            h,
            sq,
            sk,
            d,
            dv,
            scale,
            false,
            0,
            |_, _, _, _| 0.0,
        );

        let qv = Owned::f32(&[b, h, sq, d], &q);
        let kv = Owned::f32(&[b, h, sk, d], &k);
        let vv = Owned::f32(&[b, h, sk, dv], &v);
        let mut out = Owned::zeros_f32(&[b, h, sq, dv]);
        kernel(Some(scale), false, None, None, 0, 0.0)
            .execute(&[qv.view(), kv.view(), vv.view()], &mut [out.view_mut()])
            .unwrap();
        approx(&out.to_f32(), &want, 1e-5);
    }

    #[test]
    fn default_scale_is_inv_sqrt_head_size() {
        let (b, h, sq, sk, d, dv) = (1, 1, 2, 2, 4, 3);
        let q: Vec<f32> = (0..b * h * sq * d).map(|i| i as f32 * 0.2).collect();
        let k: Vec<f32> = (0..b * h * sk * d).map(|i| i as f32 * 0.1).collect();
        let v: Vec<f32> = (0..b * h * sk * dv).map(|i| i as f32 * 0.05).collect();
        let scale = 1.0 / (d as f32).sqrt();
        let want = reference(
            &q,
            &k,
            &v,
            b,
            h,
            h,
            sq,
            sk,
            d,
            dv,
            scale,
            false,
            0,
            |_, _, _, _| 0.0,
        );
        let mut out = Owned::zeros_f32(&[b, h, sq, dv]);
        kernel(None, false, None, None, 0, 0.0)
            .execute(
                &[
                    Owned::f32(&[b, h, sq, d], &q).view(),
                    Owned::f32(&[b, h, sk, d], &k).view(),
                    Owned::f32(&[b, h, sk, dv], &v).view(),
                ],
                &mut [out.view_mut()],
            )
            .unwrap();
        approx(&out.to_f32(), &want, 1e-6);
    }

    #[test]
    fn three_d_reshape_matches_four_d() {
        // 3D inputs with q_num_heads/kv_num_heads reshape to the same result as
        // the equivalent 4D layout.
        let (b, h, sq, sk, d, dv) = (2, 2, 2, 3, 2, 2);
        let scale = 0.5f32;
        // Build 3D buffers: Q (b, sq, h*d), K (b, sk, h*d), V (b, sk, h*dv).
        let q3: Vec<f32> = (0..b * sq * h * d)
            .map(|i| (i as f32 * 0.13).sin())
            .collect();
        let k3: Vec<f32> = (0..b * sk * h * d)
            .map(|i| (i as f32 * 0.09).cos())
            .collect();
        let v3: Vec<f32> = (0..b * sk * h * dv).map(|i| i as f32 * 0.02).collect();

        // 4D equivalents obtained by transposing (b, s, h, d) → (b, h, s, d).
        let to4d = |src: &[f32], s: usize, dd: usize| {
            let mut out = vec![0.0f32; b * h * s * dd];
            for bb in 0..b {
                for ss in 0..s {
                    for hh in 0..h {
                        for e in 0..dd {
                            let si = ((bb * s + ss) * h + hh) * dd + e;
                            let di = ((bb * h + hh) * s + ss) * dd + e;
                            out[di] = src[si];
                        }
                    }
                }
            }
            out
        };
        let q4 = to4d(&q3, sq, d);
        let k4 = to4d(&k3, sk, d);
        let v4 = to4d(&v3, sk, dv);
        let want4 = reference(
            &q4,
            &k4,
            &v4,
            b,
            h,
            h,
            sq,
            sk,
            d,
            dv,
            scale,
            false,
            0,
            |_, _, _, _| 0.0,
        );
        // Convert reference (b, h, sq, dv) back to 3D (b, sq, h*dv) for compare.
        let mut want3 = vec![0.0f32; b * sq * h * dv];
        for bb in 0..b {
            for hh in 0..h {
                for ss in 0..sq {
                    for c in 0..dv {
                        let si = ((bb * h + hh) * sq + ss) * dv + c;
                        let di = (bb * sq + ss) * (h * dv) + hh * dv + c;
                        want3[di] = want4[si];
                    }
                }
            }
        }

        let mut out = Owned::zeros_f32(&[b, sq, h * dv]);
        kernel(Some(scale), false, Some(h), Some(h), 0, 0.0)
            .execute(
                &[
                    Owned::f32(&[b, sq, h * d], &q3).view(),
                    Owned::f32(&[b, sk, h * d], &k3).view(),
                    Owned::f32(&[b, sk, h * dv], &v3).view(),
                ],
                &mut [out.view_mut()],
            )
            .unwrap();
        approx(&out.to_f32(), &want3, 1e-5);
    }

    #[test]
    fn mla_gqa_decode_with_asymmetric_head_dims_matches_hand_computed_result() {
        let (batch, q_heads, kv_heads, seq, past_seq) = (1, 4, 2, 1, 2);
        let (qk_head_dim, v_head_dim) = (2, 1);
        let total_seq = past_seq + seq;

        // Zero Q makes all three cache positions equiprobable. GQA maps query
        // heads 0/1 to KV head 0 and 2/3 to KV head 1.
        let q = [0.0; 8];
        let k = [10.0, 20.0, 30.0, 40.0];
        let v = [5.0, 8.0];
        let past_key = [1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0];
        let past_value = [1.0, 3.0, 2.0, 4.0];

        let mut y = Owned::zeros_f32(&[batch, seq, q_heads * v_head_dim]);
        let mut present_key = Owned::zeros_f32(&[batch, kv_heads, total_seq, qk_head_dim]);
        let mut present_value = Owned::zeros_f32(&[batch, kv_heads, total_seq, v_head_dim]);
        kernel(None, true, Some(q_heads), Some(kv_heads), 0, 0.0)
            .execute(
                &[
                    Owned::f32(&[batch, seq, q_heads * qk_head_dim], &q).view(),
                    Owned::f32(&[batch, seq, kv_heads * qk_head_dim], &k).view(),
                    Owned::f32(&[batch, seq, kv_heads * v_head_dim], &v).view(),
                    absent(),
                    Owned::f32(&[batch, kv_heads, past_seq, qk_head_dim], &past_key).view(),
                    Owned::f32(&[batch, kv_heads, past_seq, v_head_dim], &past_value).view(),
                ],
                &mut [
                    y.view_mut(),
                    present_key.view_mut(),
                    present_value.view_mut(),
                ],
            )
            .unwrap();

        assert_eq!(y.shape, [batch, seq, q_heads * v_head_dim]);
        assert_eq!(present_key.shape, [batch, kv_heads, total_seq, qk_head_dim]);
        assert_eq!(
            present_value.shape,
            [batch, kv_heads, total_seq, v_head_dim]
        );
        approx(&y.to_f32(), &[3.0, 3.0, 14.0 / 3.0, 14.0 / 3.0], 1e-6);
        assert_eq!(
            present_key.to_f32(),
            [
                1.0, 0.0, 0.0, 1.0, 10.0, 20.0, 2.0, 0.0, 0.0, 2.0, 30.0, 40.0
            ]
        );
        assert_eq!(present_value.to_f32(), [1.0, 3.0, 5.0, 2.0, 4.0, 8.0]);
    }

    fn asymmetric_3d_prefill_decode_case(kv_heads: usize) {
        let (batch, q_heads, prefill_seq, decode_seq) = (1usize, 4usize, 3usize, 1usize);
        let (qk_head_dim, v_head_dim) = (192usize, 128usize);
        assert_ne!(qk_head_dim, v_head_dim);

        let q_prefill =
            conformance_values(batch * prefill_seq * q_heads * qk_head_dim, 1 + kv_heads);
        let k_prefill =
            conformance_values(batch * prefill_seq * kv_heads * qk_head_dim, 3 + kv_heads);
        let v_prefill =
            conformance_values(batch * prefill_seq * kv_heads * v_head_dim, 5 + kv_heads);
        let q_prefill_bhsd = bsh_to_bhsd(&q_prefill, batch, prefill_seq, q_heads, qk_head_dim);
        let k_prefill_bhsd = bsh_to_bhsd(&k_prefill, batch, prefill_seq, kv_heads, qk_head_dim);
        let v_prefill_bhsd = bsh_to_bhsd(&v_prefill, batch, prefill_seq, kv_heads, v_head_dim);
        let scale = 1.0 / (qk_head_dim as f32).sqrt();
        let prefill_oracle_bhsd = reference(
            &q_prefill_bhsd,
            &k_prefill_bhsd,
            &v_prefill_bhsd,
            batch,
            q_heads,
            kv_heads,
            prefill_seq,
            prefill_seq,
            qk_head_dim,
            v_head_dim,
            scale,
            true,
            0,
            |_, _, _, _| 0.0,
        );
        let prefill_oracle = bhsd_to_bsh(
            &prefill_oracle_bhsd,
            batch,
            q_heads,
            prefill_seq,
            v_head_dim,
        );

        let mut prefill_y = Owned::zeros_f32(&[batch, prefill_seq, q_heads * v_head_dim]);
        let mut present_key = Owned::zeros_f32(&[batch, kv_heads, prefill_seq, qk_head_dim]);
        let mut present_value = Owned::zeros_f32(&[batch, kv_heads, prefill_seq, v_head_dim]);
        kernel(None, true, Some(q_heads), Some(kv_heads), 0, 0.0)
            .execute(
                &[
                    Owned::f32(&[batch, prefill_seq, q_heads * qk_head_dim], &q_prefill).view(),
                    Owned::f32(&[batch, prefill_seq, kv_heads * qk_head_dim], &k_prefill).view(),
                    Owned::f32(&[batch, prefill_seq, kv_heads * v_head_dim], &v_prefill).view(),
                ],
                &mut [
                    prefill_y.view_mut(),
                    present_key.view_mut(),
                    present_value.view_mut(),
                ],
            )
            .unwrap();

        assert_eq!(
            prefill_y.shape,
            [batch, prefill_seq, q_heads * v_head_dim],
            "3D Attention output hidden width must use V head width"
        );
        assert_eq!(
            present_key.shape,
            [batch, kv_heads, prefill_seq, qk_head_dim],
            "present_key must preserve Q/K head width"
        );
        assert_eq!(
            present_value.shape,
            [batch, kv_heads, prefill_seq, v_head_dim],
            "present_value must preserve V head width"
        );
        approx(&prefill_y.to_f32(), &prefill_oracle, 1e-5);
        assert_eq!(present_key.to_f32(), k_prefill_bhsd);
        assert_eq!(present_value.to_f32(), v_prefill_bhsd);

        let q_decode = conformance_values(batch * decode_seq * q_heads * qk_head_dim, 7 + kv_heads);
        let k_decode =
            conformance_values(batch * decode_seq * kv_heads * qk_head_dim, 9 + kv_heads);
        let v_decode =
            conformance_values(batch * decode_seq * kv_heads * v_head_dim, 11 + kv_heads);
        let q_decode_bhsd = bsh_to_bhsd(&q_decode, batch, decode_seq, q_heads, qk_head_dim);
        let k_decode_bhsd = bsh_to_bhsd(&k_decode, batch, decode_seq, kv_heads, qk_head_dim);
        let v_decode_bhsd = bsh_to_bhsd(&v_decode, batch, decode_seq, kv_heads, v_head_dim);
        let full_key = concat_bhsd(
            &k_prefill_bhsd,
            &k_decode_bhsd,
            batch,
            kv_heads,
            prefill_seq,
            decode_seq,
            qk_head_dim,
        );
        let full_value = concat_bhsd(
            &v_prefill_bhsd,
            &v_decode_bhsd,
            batch,
            kv_heads,
            prefill_seq,
            decode_seq,
            v_head_dim,
        );
        let total_seq = prefill_seq + decode_seq;
        let decode_oracle_bhsd = reference(
            &q_decode_bhsd,
            &full_key,
            &full_value,
            batch,
            q_heads,
            kv_heads,
            decode_seq,
            total_seq,
            qk_head_dim,
            v_head_dim,
            scale,
            true,
            prefill_seq as i64,
            |_, _, _, _| 0.0,
        );
        let decode_oracle =
            bhsd_to_bsh(&decode_oracle_bhsd, batch, q_heads, decode_seq, v_head_dim);

        let mut decode_y = Owned::zeros_f32(&[batch, decode_seq, q_heads * v_head_dim]);
        let mut decode_present_key = Owned::zeros_f32(&[batch, kv_heads, total_seq, qk_head_dim]);
        let mut decode_present_value = Owned::zeros_f32(&[batch, kv_heads, total_seq, v_head_dim]);
        kernel(None, true, Some(q_heads), Some(kv_heads), 0, 0.0)
            .execute(
                &[
                    Owned::f32(&[batch, decode_seq, q_heads * qk_head_dim], &q_decode).view(),
                    Owned::f32(&[batch, decode_seq, kv_heads * qk_head_dim], &k_decode).view(),
                    Owned::f32(&[batch, decode_seq, kv_heads * v_head_dim], &v_decode).view(),
                    absent(),
                    present_key.view(),
                    present_value.view(),
                ],
                &mut [
                    decode_y.view_mut(),
                    decode_present_key.view_mut(),
                    decode_present_value.view_mut(),
                ],
            )
            .unwrap();

        assert_eq!(decode_y.shape, [batch, decode_seq, q_heads * v_head_dim]);
        assert_eq!(
            decode_present_key.shape,
            [batch, kv_heads, total_seq, qk_head_dim]
        );
        assert_eq!(
            decode_present_value.shape,
            [batch, kv_heads, total_seq, v_head_dim]
        );
        approx(&decode_y.to_f32(), &decode_oracle, 1e-5);
        assert_eq!(decode_present_key.to_f32(), full_key);
        assert_eq!(decode_present_value.to_f32(), full_value);

        let full_q_bsh = concat_bsh(
            &q_prefill,
            &q_decode,
            batch,
            prefill_seq,
            decode_seq,
            q_heads * qk_head_dim,
        );
        let full_k_bsh = concat_bsh(
            &k_prefill,
            &k_decode,
            batch,
            prefill_seq,
            decode_seq,
            kv_heads * qk_head_dim,
        );
        let full_v_bsh = concat_bsh(
            &v_prefill,
            &v_decode,
            batch,
            prefill_seq,
            decode_seq,
            kv_heads * v_head_dim,
        );
        let mut full_y = Owned::zeros_f32(&[batch, total_seq, q_heads * v_head_dim]);
        let mut full_present_key = Owned::zeros_f32(&[batch, kv_heads, total_seq, qk_head_dim]);
        let mut full_present_value = Owned::zeros_f32(&[batch, kv_heads, total_seq, v_head_dim]);
        kernel(None, true, Some(q_heads), Some(kv_heads), 0, 0.0)
            .execute(
                &[
                    Owned::f32(&[batch, total_seq, q_heads * qk_head_dim], &full_q_bsh).view(),
                    Owned::f32(&[batch, total_seq, kv_heads * qk_head_dim], &full_k_bsh).view(),
                    Owned::f32(&[batch, total_seq, kv_heads * v_head_dim], &full_v_bsh).view(),
                ],
                &mut [
                    full_y.view_mut(),
                    full_present_key.view_mut(),
                    full_present_value.view_mut(),
                ],
            )
            .unwrap();

        let expected_full_y = concat_bsh(
            &prefill_oracle,
            &decode_oracle,
            batch,
            prefill_seq,
            decode_seq,
            q_heads * v_head_dim,
        );
        approx(&full_y.to_f32(), &expected_full_y, 1e-5);
        assert_eq!(full_present_key.to_f32(), decode_present_key.to_f32());
        assert_eq!(full_present_value.to_f32(), decode_present_value.to_f32());
    }

    #[test]
    fn asymmetric_3d_prefill_decode_gqa_matches_scalar_oracle() {
        asymmetric_3d_prefill_decode_case(2);
    }

    #[test]
    fn asymmetric_3d_prefill_decode_mqa_matches_scalar_oracle() {
        asymmetric_3d_prefill_decode_case(1);
    }

    #[test]
    fn gqa_head_sharing() {
        // q_heads=4, kv_heads=2 → group of 2 query heads share each KV head.
        let (b, qh, kvh, sq, sk, d, dv) = (1, 4, 2, 2, 3, 3, 2);
        let scale = 0.4f32;
        let q: Vec<f32> = (0..b * qh * sq * d)
            .map(|i| (i as f32 * 0.11).sin())
            .collect();
        let k: Vec<f32> = (0..b * kvh * sk * d)
            .map(|i| (i as f32 * 0.08).cos())
            .collect();
        let v: Vec<f32> = (0..b * kvh * sk * dv)
            .map(|i| i as f32 * 0.04 - 1.0)
            .collect();
        let want = reference(
            &q,
            &k,
            &v,
            b,
            qh,
            kvh,
            sq,
            sk,
            d,
            dv,
            scale,
            false,
            0,
            |_, _, _, _| 0.0,
        );
        let mut out = Owned::zeros_f32(&[b, qh, sq, dv]);
        kernel(Some(scale), false, None, None, 0, 0.0)
            .execute(
                &[
                    Owned::f32(&[b, qh, sq, d], &q).view(),
                    Owned::f32(&[b, kvh, sk, d], &k).view(),
                    Owned::f32(&[b, kvh, sk, dv], &v).view(),
                ],
                &mut [out.view_mut()],
            )
            .unwrap();
        approx(&out.to_f32(), &want, 1e-5);
    }

    #[test]
    fn causal_masking_blocks_future() {
        let (b, h, s, d, dv) = (1, 1, 4, 3, 2);
        let scale = 0.5f32;
        let q: Vec<f32> = (0..b * h * s * d)
            .map(|i| (i as f32 * 0.17).sin())
            .collect();
        let k: Vec<f32> = (0..b * h * s * d)
            .map(|i| (i as f32 * 0.05).cos())
            .collect();
        let v: Vec<f32> = (0..b * h * s * dv).map(|i| i as f32 * 0.1).collect();
        let want = reference(
            &q,
            &k,
            &v,
            b,
            h,
            h,
            s,
            s,
            d,
            dv,
            scale,
            true,
            0,
            |_, _, _, _| 0.0,
        );
        let mut out = Owned::zeros_f32(&[b, h, s, dv]);
        kernel(Some(scale), true, None, None, 0, 0.0)
            .execute(
                &[
                    Owned::f32(&[b, h, s, d], &q).view(),
                    Owned::f32(&[b, h, s, d], &k).view(),
                    Owned::f32(&[b, h, s, dv], &v).view(),
                ],
                &mut [out.view_mut()],
            )
            .unwrap();
        approx(&out.to_f32(), &want, 1e-5);

        // Independent structural check: with causal masking, query 0's output
        // must equal V[0] (it can attend only to key 0, softmax → 1.0).
        let got = out.to_f32();
        approx(&got[0..dv], &v[0..dv], 1e-5);
    }

    #[test]
    fn float_additive_mask() {
        let (b, h, sq, sk, d, dv) = (1, 1, 2, 3, 2, 2);
        let scale = 0.6f32;
        let q = [1.0f32, 2.0, -1.0, 0.5];
        let k = [1.0f32, 0.0, 0.0, 1.0, 1.0, 1.0];
        let v = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mask = [0.0f32, -1e4, 0.0, 0.0, 0.0, -1e4]; // (sq=2, sk=3)
        let want = reference(
            &q,
            &k,
            &v,
            b,
            h,
            h,
            sq,
            sk,
            d,
            dv,
            scale,
            false,
            0,
            |_, _, i, j| mask[i * sk + j],
        );
        let mut out = Owned::zeros_f32(&[b, h, sq, dv]);
        kernel(Some(scale), false, None, None, 0, 0.0)
            .execute(
                &[
                    Owned::f32(&[b, h, sq, d], &q).view(),
                    Owned::f32(&[b, h, sk, d], &k).view(),
                    Owned::f32(&[b, h, sk, dv], &v).view(),
                    Owned::f32(&[sq, sk], &mask).view(),
                ],
                &mut [out.view_mut()],
            )
            .unwrap();
        approx(&out.to_f32(), &want, 1e-4);
    }

    #[test]
    fn bool_mask_matches_neg_inf_bias() {
        let (b, h, sq, sk, d, dv) = (1, 1, 2, 3, 2, 2);
        let scale = 0.6f32;
        let q = [1.0f32, 2.0, -1.0, 0.5];
        let k = [1.0f32, 0.0, 0.0, 1.0, 1.0, 1.0];
        let v = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        // true = keep. Row 0 drops key 1; row 1 drops key 2.
        let keep = [true, false, true, true, true, false];
        let want = reference(
            &q,
            &k,
            &v,
            b,
            h,
            h,
            sq,
            sk,
            d,
            dv,
            scale,
            false,
            0,
            |_, _, i, j| {
                if keep[i * sk + j] {
                    0.0
                } else {
                    f32::NEG_INFINITY
                }
            },
        );
        let mut out = Owned::zeros_f32(&[b, h, sq, dv]);
        kernel(Some(scale), false, None, None, 0, 0.0)
            .execute(
                &[
                    Owned::f32(&[b, h, sq, d], &q).view(),
                    Owned::f32(&[b, h, sk, d], &k).view(),
                    Owned::f32(&[b, h, sk, dv], &v).view(),
                    Owned::bool_(&[sq, sk], &keep).view(),
                ],
                &mut [out.view_mut()],
            )
            .unwrap();
        approx(&out.to_f32(), &want, 1e-5);
    }

    #[test]
    fn fully_masked_row_is_zero_not_nan() {
        let (b, h, sq, sk, d, dv) = (1, 1, 2, 2, 2, 2);
        let q = [1.0f32, 2.0, 3.0, 4.0];
        let k = [1.0f32, 0.0, 0.0, 1.0];
        let v = [5.0f32, 6.0, 7.0, 8.0];
        // Row 0 fully masked (all false), row 1 all kept.
        let keep = [false, false, true, true];
        let mut out = Owned::zeros_f32(&[b, h, sq, dv]);
        kernel(Some(0.5), false, None, None, 0, 0.0)
            .execute(
                &[
                    Owned::f32(&[b, h, sq, d], &q).view(),
                    Owned::f32(&[b, h, sk, d], &k).view(),
                    Owned::f32(&[b, h, sk, dv], &v).view(),
                    Owned::bool_(&[sq, sk], &keep).view(),
                ],
                &mut [out.view_mut()],
            )
            .unwrap();
        let got = out.to_f32();
        assert!(got.iter().all(|x| x.is_finite()), "no NaN/inf: {got:?}");
        approx(&got[0..dv], &[0.0, 0.0], 1e-6);
    }

    #[test]
    fn kv_cache_concat_and_present_outputs() {
        // past_seq=2, current kv_seq=1 → total_seq=3. Verify Y uses the full
        // history and present_key/value = concat(past, current).
        let (b, h, sq, d, dv) = (1, 1, 1, 2, 2);
        let past_seq = 2usize;
        let kv_seq = 1usize;
        let total = past_seq + kv_seq;
        let scale = 0.5f32;

        let q = [0.5f32, -0.5];
        let past_k = [1.0f32, 0.0, 0.0, 1.0]; // (past_seq=2, d=2)
        let cur_k = [1.0f32, 1.0]; // (kv_seq=1, d=2)
        let past_v = [1.0f32, 2.0, 3.0, 4.0];
        let cur_v = [5.0f32, 6.0];

        let mut full_k = past_k.to_vec();
        full_k.extend_from_slice(&cur_k);
        let mut full_v = past_v.to_vec();
        full_v.extend_from_slice(&cur_v);

        let want = reference(
            &q,
            &full_k,
            &full_v,
            b,
            h,
            h,
            sq,
            total,
            d,
            dv,
            scale,
            false,
            past_seq as i64,
            |_, _, _, _| 0.0,
        );

        let mut y = Owned::zeros_f32(&[b, h, sq, dv]);
        let mut pk = Owned::zeros_f32(&[b, h, total, d]);
        let mut pv = Owned::zeros_f32(&[b, h, total, dv]);
        kernel(Some(scale), false, None, None, 0, 0.0)
            .execute(
                &[
                    Owned::f32(&[b, h, sq, d], &q).view(),
                    Owned::f32(&[b, h, kv_seq, d], &cur_k).view(),
                    Owned::f32(&[b, h, kv_seq, dv], &cur_v).view(),
                    absent(), // omitted attn_mask
                    Owned::f32(&[b, h, past_seq, d], &past_k).view(),
                    Owned::f32(&[b, h, past_seq, dv], &past_v).view(),
                ],
                &mut [y.view_mut(), pk.view_mut(), pv.view_mut()],
            )
            .unwrap();
        approx(&y.to_f32(), &want, 1e-5);
        approx(&pk.to_f32(), &full_k, 1e-6);
        approx(&pv.to_f32(), &full_v, 1e-6);
    }

    #[test]
    fn causal_kv_cache_offset() {
        // With a past cache, causal offset = past_seq: query 0 attends keys
        // [0, past_seq]. past_seq=2, kv_seq=2, q_seq=2, total=4.
        let (b, h, sq, d, dv) = (1, 1, 2, 2, 2);
        let past_seq = 2usize;
        let kv_seq = 2usize;
        let total = past_seq + kv_seq;
        let scale = 0.5f32;
        let q: Vec<f32> = (0..b * h * sq * d)
            .map(|i| (i as f32 * 0.3).sin())
            .collect();
        let cur_k: Vec<f32> = (0..kv_seq * d).map(|i| (i as f32 * 0.2).cos()).collect();
        let cur_v: Vec<f32> = (0..kv_seq * dv).map(|i| i as f32 * 0.5).collect();
        let past_k: Vec<f32> = (0..past_seq * d).map(|i| i as f32 * 0.1).collect();
        let past_v: Vec<f32> = (0..past_seq * dv).map(|i| i as f32 * 0.3).collect();
        let mut full_k = past_k.clone();
        full_k.extend_from_slice(&cur_k);
        let mut full_v = past_v.clone();
        full_v.extend_from_slice(&cur_v);
        let want = reference(
            &q,
            &full_k,
            &full_v,
            b,
            h,
            h,
            sq,
            total,
            d,
            dv,
            scale,
            true,
            past_seq as i64,
            |_, _, _, _| 0.0,
        );
        let mut y = Owned::zeros_f32(&[b, h, sq, dv]);
        kernel(Some(scale), true, None, None, 0, 0.0)
            .execute(
                &[
                    Owned::f32(&[b, h, sq, d], &q).view(),
                    Owned::f32(&[b, h, kv_seq, d], &cur_k).view(),
                    Owned::f32(&[b, h, kv_seq, dv], &cur_v).view(),
                    absent(),
                    Owned::f32(&[b, h, past_seq, d], &past_k).view(),
                    Owned::f32(&[b, h, past_seq, dv], &past_v).view(),
                ],
                &mut [y.view_mut()],
            )
            .unwrap();
        approx(&y.to_f32(), &want, 1e-5);
    }

    #[test]
    fn softcap_changes_output() {
        let (b, h, sq, sk, d, dv) = (1, 1, 2, 2, 2, 2);
        let scale = 1.0f32;
        let q = [3.0f32, 4.0, -2.0, 1.0];
        let k = [2.0f32, 1.0, -1.0, 3.0];
        let v = [1.0f32, 0.0, 0.0, 1.0];
        let softcap = 2.0f32;
        let want = reference(
            &q,
            &k,
            &v,
            b,
            h,
            h,
            sq,
            sk,
            d,
            dv,
            scale,
            false,
            0,
            |_, _, _, _| 0.0,
        )
        .into_iter()
        .collect::<Vec<_>>();
        // With softcap, scores are squashed → different distribution.
        let want_capped = {
            // Reference with softcap applied on the raw scaled scores.
            let mut out = vec![0.0f32; sq * dv];
            for i in 0..sq {
                let mut scores = [0.0f32; 2];
                for (j, sc) in scores.iter_mut().enumerate() {
                    let mut acc = 0.0f32;
                    for p in 0..d {
                        acc += q[i * d + p] * k[j * d + p];
                    }
                    let s = acc * scale;
                    *sc = softcap * (s / softcap).tanh();
                }
                let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0;
                for sc in scores.iter_mut() {
                    *sc = (*sc - max).exp();
                    sum += *sc;
                }
                for sc in scores.iter_mut() {
                    *sc /= sum;
                }
                for c in 0..dv {
                    let mut acc = 0.0f32;
                    for (j, &p) in scores.iter().enumerate() {
                        acc += p * v[j * dv + c];
                    }
                    out[i * dv + c] = acc;
                }
            }
            out
        };

        let mut out = Owned::zeros_f32(&[b, h, sq, dv]);
        kernel(Some(scale), false, None, None, 0, softcap)
            .execute(
                &[
                    Owned::f32(&[b, h, sq, d], &q).view(),
                    Owned::f32(&[b, h, sk, d], &k).view(),
                    Owned::f32(&[b, h, sk, dv], &v).view(),
                ],
                &mut [out.view_mut()],
            )
            .unwrap();
        approx(&out.to_f32(), &want_capped, 1e-5);
        // Sanity: softcap actually changed the result vs. no-softcap.
        assert!(
            out.to_f32()
                .iter()
                .zip(&want)
                .any(|(a, b)| (a - b).abs() > 1e-3),
            "softcap should change the output"
        );
    }

    #[test]
    fn qk_matmul_output_mode0_is_scaled_scores() {
        let (b, h, sq, sk, d, dv) = (1, 1, 2, 2, 2, 2);
        let scale = 0.5f32;
        let q = [1.0f32, 2.0, 3.0, 4.0];
        let k = [1.0f32, 1.0, 2.0, 0.0];
        let v = [1.0f32, 0.0, 0.0, 1.0];
        let mut expected = [0.0f32; 4];
        for i in 0..sq {
            for j in 0..sk {
                let mut acc = 0.0f32;
                for p in 0..d {
                    acc += q[i * d + p] * k[j * d + p];
                }
                expected[i * sk + j] = acc * scale;
            }
        }
        let mut y = Owned::zeros_f32(&[b, h, sq, dv]);
        let mut pk = Owned::zeros_f32(&[b, h, sk, d]);
        let mut pv = Owned::zeros_f32(&[b, h, sk, dv]);
        let mut qk = Owned::zeros_f32(&[b, h, sq, sk]);
        kernel(Some(scale), false, None, None, 0, 0.0)
            .execute(
                &[
                    Owned::f32(&[b, h, sq, d], &q).view(),
                    Owned::f32(&[b, h, sk, d], &k).view(),
                    Owned::f32(&[b, h, sk, dv], &v).view(),
                ],
                &mut [y.view_mut(), pk.view_mut(), pv.view_mut(), qk.view_mut()],
            )
            .unwrap();
        approx(&qk.to_f32(), &expected, 1e-6);
    }

    #[test]
    fn qk_matmul_output_mode3_is_softmax() {
        let (b, h, sq, sk, d, dv) = (1, 1, 2, 2, 2, 2);
        let scale = 0.5f32;
        let q = [1.0f32, 0.0, 0.0, 1.0];
        let k = [1.0f32, 3.0, 2.0, 4.0];
        let v = [1.0f32, 0.0, 0.0, 1.0];
        let mut y = Owned::zeros_f32(&[b, h, sq, dv]);
        let mut pk = Owned::zeros_f32(&[b, h, sk, d]);
        let mut pv = Owned::zeros_f32(&[b, h, sk, dv]);
        let mut qk = Owned::zeros_f32(&[b, h, sq, sk]);
        kernel(Some(scale), false, None, None, 3, 0.0)
            .execute(
                &[
                    Owned::f32(&[b, h, sq, d], &q).view(),
                    Owned::f32(&[b, h, sk, d], &k).view(),
                    Owned::f32(&[b, h, sk, dv], &v).view(),
                ],
                &mut [y.view_mut(), pk.view_mut(), pv.view_mut(), qk.view_mut()],
            )
            .unwrap();
        let probs = qk.to_f32();
        // Each softmax row sums to 1.
        assert!((probs[0] + probs[1] - 1.0).abs() < 1e-6);
        assert!((probs[2] + probs[3] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn large_logits_are_numerically_stable() {
        let (b, h, sq, sk, d, dv) = (1, 1, 2, 3, 2, 2);
        let scale = 1.0f32;
        // Large magnitudes that would overflow a naive exp without max-subtract.
        let q = [100.0f32, 100.0, -100.0, -100.0];
        let k = [100.0f32, 100.0, -100.0, -100.0, 50.0, 50.0];
        let v = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut out = Owned::zeros_f32(&[b, h, sq, dv]);
        kernel(Some(scale), false, None, None, 0, 0.0)
            .execute(
                &[
                    Owned::f32(&[b, h, sq, d], &q).view(),
                    Owned::f32(&[b, h, sk, d], &k).view(),
                    Owned::f32(&[b, h, sk, dv], &v).view(),
                ],
                &mut [out.view_mut()],
            )
            .unwrap();
        assert!(out.to_f32().iter().all(|x| x.is_finite()));
    }

    #[test]
    fn factory_rejects_unsupported_qk_mode() {
        use onnx_runtime_ir::{Attribute, NodeId};
        let mut node = Node::new(NodeId(0), "Attention", vec![], vec![]);
        node.attributes
            .insert("qk_matmul_output_mode".to_string(), Attribute::Int(5));
        let err = AttentionFactory { since_version: 24 }.create(&node, &[]);
        assert!(err.is_err(), "qk_matmul_output_mode=5 must be rejected");
    }

    #[test]
    fn factory_accepts_valid_attributes() {
        use onnx_runtime_ir::{Attribute, NodeId};
        let mut node = Node::new(NodeId(0), "Attention", vec![], vec![]);
        node.attributes
            .insert("is_causal".to_string(), Attribute::Int(1));
        node.attributes
            .insert("scale".to_string(), Attribute::Float(0.25));
        node.attributes
            .insert("qk_matmul_output_mode".to_string(), Attribute::Int(3));
        assert!(
            AttentionFactory { since_version: 24 }
                .create(&node, &[])
                .is_ok()
        );
    }

    #[test]
    fn nonpad_kv_seqlen_rejected_for_opset23() {
        // The 7th input was added in opset 24; the v23 kernel must reject it.
        let (b, h, s, d, dv) = (1, 1, 2, 2, 2);
        let q = vec![0.1f32; b * h * s * d];
        let k = vec![0.1f32; b * h * s * d];
        let v = vec![0.1f32; b * h * s * dv];
        let seqlen = [2i64];
        let mut out = Owned::zeros_f32(&[b, h, s, dv]);
        let err = kernel_v(23, Some(0.5), false, None, None, 0, 0.0).execute(
            &[
                Owned::f32(&[b, h, s, d], &q).view(),
                Owned::f32(&[b, h, s, d], &k).view(),
                Owned::f32(&[b, h, s, dv], &v).view(),
                absent(),
                absent(),
                absent(),
                Owned::i64(&[b], &seqlen).view(),
            ],
            &mut [out.view_mut()],
        );
        assert!(err.is_err(), "nonpad_kv_seqlen must error for opset 23");
    }

    #[test]
    fn nonpad_kv_seqlen_rejected_with_past_cache() {
        // nonpad_kv_seqlen (external cache) is mutually exclusive with an in-op
        // past_key/past_value cache.
        let (b, h, s, d, dv) = (1, 1, 2, 2, 2);
        let q = vec![0.1f32; b * h * s * d];
        let k = vec![0.1f32; b * h * s * d];
        let v = vec![0.1f32; b * h * s * dv];
        let past_k = vec![0.1f32; b * h * s * d];
        let past_v = vec![0.1f32; b * h * s * dv];
        let seqlen = [2i64];
        let mut out = Owned::zeros_f32(&[b, h, s, dv]);
        let err = kernel_v(24, Some(0.5), false, None, None, 0, 0.0).execute(
            &[
                Owned::f32(&[b, h, s, d], &q).view(),
                Owned::f32(&[b, h, s, d], &k).view(),
                Owned::f32(&[b, h, s, dv], &v).view(),
                absent(),
                Owned::f32(&[b, h, s, d], &past_k).view(),
                Owned::f32(&[b, h, s, dv], &past_v).view(),
                Owned::i64(&[b], &seqlen).view(),
            ],
            &mut [out.view_mut()],
        );
        assert!(
            err.is_err(),
            "nonpad_kv_seqlen with past_key/past_value must error"
        );
    }

    #[test]
    fn non_divisible_gqa_errors() {
        // q_heads=3, kv_heads=2 → 3 % 2 != 0.
        let (b, qh, kvh, s, d, dv) = (1, 3, 2, 2, 2, 2);
        let q = vec![0.1f32; b * qh * s * d];
        let k = vec![0.1f32; b * kvh * s * d];
        let v = vec![0.1f32; b * kvh * s * dv];
        let mut out = Owned::zeros_f32(&[b, qh, s, dv]);
        let err = kernel(Some(0.5), false, None, None, 0, 0.0).execute(
            &[
                Owned::f32(&[b, qh, s, d], &q).view(),
                Owned::f32(&[b, kvh, s, d], &k).view(),
                Owned::f32(&[b, kvh, s, dv], &v).view(),
            ],
            &mut [out.view_mut()],
        );
        assert!(err.is_err(), "non-divisible GQA must error");
    }

    // ---- Job A: rejection-fix regression tests ----

    #[test]
    fn sqrt_scale_avoids_overflow() {
        // Bug 1: Q=K=1e30, scale=1e-30. The naive `Q·Kᵀ` = 1e60 overflows f32
        // *before* scaling; folding sqrt(scale) into each operand keeps the
        // score finite (~1e30).
        let (b, h, sq, sk, d, dv) = (1, 1, 1, 1, 1, 1);
        let q = [1e30f32];
        let k = [1e30f32];
        let v = [7.0f32];
        let scale = 1e-30f32;
        let mut y = Owned::zeros_f32(&[b, h, sq, dv]);
        let mut pk = Owned::zeros_f32(&[b, h, sk, d]);
        let mut pv = Owned::zeros_f32(&[b, h, sk, dv]);
        let mut qk = Owned::zeros_f32(&[b, h, sq, sk]);
        kernel(Some(scale), false, None, None, 0, 0.0)
            .execute(
                &[
                    Owned::f32(&[b, h, sq, d], &q).view(),
                    Owned::f32(&[b, h, sk, d], &k).view(),
                    Owned::f32(&[b, h, sk, dv], &v).view(),
                ],
                &mut [y.view_mut(), pk.view_mut(), pv.view_mut(), qk.view_mut()],
            )
            .unwrap();
        // The naive unscaled product overflows; the folded score does not.
        assert!(!(1e30f32 * 1e30f32).is_finite());
        let score = qk.to_f32()[0];
        assert!(score.is_finite(), "score must be finite, got {score}");
        assert!((score - 1e30).abs() < 1e26, "score ~1e30, got {score}");
        approx(&y.to_f32(), &v, 1e-3);
    }

    #[test]
    fn scalar_bool_false_mask_zeros_all_rows() {
        // Bug 2: a rank-0 boolean `false` masks every score → all-zero output.
        let (b, h, sq, sk, d, dv) = (1, 1, 2, 2, 2, 2);
        let q = [1.0f32, 2.0, 3.0, 4.0];
        let k = [1.0f32, 0.0, 0.0, 1.0];
        let v = [5.0f32, 6.0, 7.0, 8.0];
        let mask = [false];
        let mut out = Owned::zeros_f32(&[b, h, sq, dv]);
        kernel(Some(0.5), false, None, None, 0, 0.0)
            .execute(
                &[
                    Owned::f32(&[b, h, sq, d], &q).view(),
                    Owned::f32(&[b, h, sk, d], &k).view(),
                    Owned::f32(&[b, h, sk, dv], &v).view(),
                    Owned::bool_(&[], &mask).view(),
                ],
                &mut [out.view_mut()],
            )
            .unwrap();
        let got = out.to_f32();
        assert!(
            got.iter().all(|x| *x == 0.0),
            "scalar false → zero: {got:?}"
        );
    }

    #[test]
    fn scalar_bool_true_mask_is_noop() {
        // Bug 2: a rank-0 boolean `true` keeps every score (equivalent to no
        // mask), and is *not* mistaken for an omitted input.
        let (b, h, sq, sk, d, dv) = (1, 1, 2, 3, 2, 2);
        let scale = 0.5f32;
        let q: Vec<f32> = (0..b * h * sq * d)
            .map(|i| (i as f32 * 0.2).sin())
            .collect();
        let k: Vec<f32> = (0..b * h * sk * d)
            .map(|i| (i as f32 * 0.1).cos())
            .collect();
        let v: Vec<f32> = (0..b * h * sk * dv).map(|i| i as f32 * 0.3).collect();
        let want = reference(
            &q,
            &k,
            &v,
            b,
            h,
            h,
            sq,
            sk,
            d,
            dv,
            scale,
            false,
            0,
            |_, _, _, _| 0.0,
        );
        let mask = [true];
        let mut out = Owned::zeros_f32(&[b, h, sq, dv]);
        kernel(Some(scale), false, None, None, 0, 0.0)
            .execute(
                &[
                    Owned::f32(&[b, h, sq, d], &q).view(),
                    Owned::f32(&[b, h, sk, d], &k).view(),
                    Owned::f32(&[b, h, sk, dv], &v).view(),
                    Owned::bool_(&[], &mask).view(),
                ],
                &mut [out.view_mut()],
            )
            .unwrap();
        approx(&out.to_f32(), &want, 1e-5);
    }

    #[test]
    fn scalar_float_mask_adds_bias() {
        // Bug 2: a rank-0 float mask is added to every score.
        let (b, h, sq, sk, d, dv) = (1, 1, 1, 2, 2, 2);
        let scale = 0.5f32;
        let q = [1.0f32, 2.0];
        let k = [1.0f32, 0.0, 0.0, 1.0];
        let v = [1.0f32, 0.0, 0.0, 1.0];
        let bias = [3.0f32];
        let mut y = Owned::zeros_f32(&[b, h, sq, dv]);
        let mut pk = Owned::zeros_f32(&[b, h, sk, d]);
        let mut pv = Owned::zeros_f32(&[b, h, sk, dv]);
        let mut qk = Owned::zeros_f32(&[b, h, sq, sk]);
        // qk mode 2 (v24) = scores after mask addition.
        kernel(Some(scale), false, None, None, 2, 0.0)
            .execute(
                &[
                    Owned::f32(&[b, h, sq, d], &q).view(),
                    Owned::f32(&[b, h, sk, d], &k).view(),
                    Owned::f32(&[b, h, sk, dv], &v).view(),
                    Owned::f32(&[], &bias).view(),
                ],
                &mut [y.view_mut(), pk.view_mut(), pv.view_mut(), qk.view_mut()],
            )
            .unwrap();
        let mut expected = [0.0f32; 2];
        for (j, e) in expected.iter_mut().enumerate() {
            let mut acc = 0.0f32;
            for p in 0..d {
                acc += q[p] * k[j * d + p];
            }
            *e = acc * scale + 3.0;
        }
        approx(&qk.to_f32(), &expected, 1e-4);
    }

    #[test]
    fn negative_softcap_is_applied() {
        // Bug 3: softcap applies when nonzero (not only when > 0). A negative
        // softcap yields the same odd-function transform as its positive twin.
        let (b, h, sq, sk, d, dv) = (1, 1, 2, 2, 2, 2);
        let scale = 1.0f32;
        let q = [3.0f32, 4.0, -2.0, 1.0];
        let k = [2.0f32, 1.0, -1.0, 3.0];
        let v = [1.0f32, 0.0, 0.0, 1.0];
        let softcap = -2.0f32;
        let want = {
            let mut out = vec![0.0f32; sq * dv];
            for i in 0..sq {
                let mut scores = [0.0f32; 2];
                for (j, sc) in scores.iter_mut().enumerate() {
                    let mut acc = 0.0f32;
                    for p in 0..d {
                        acc += q[i * d + p] * k[j * d + p];
                    }
                    let s = acc * scale;
                    *sc = softcap * (s / softcap).tanh();
                }
                let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0;
                for sc in scores.iter_mut() {
                    *sc = (*sc - max).exp();
                    sum += *sc;
                }
                for sc in scores.iter_mut() {
                    *sc /= sum;
                }
                for c in 0..dv {
                    let mut acc = 0.0f32;
                    for (j, &p) in scores.iter().enumerate() {
                        acc += p * v[j * dv + c];
                    }
                    out[i * dv + c] = acc;
                }
            }
            out
        };
        let want_nocap = reference(
            &q,
            &k,
            &v,
            b,
            h,
            h,
            sq,
            sk,
            d,
            dv,
            scale,
            false,
            0,
            |_, _, _, _| 0.0,
        );
        let mut out = Owned::zeros_f32(&[b, h, sq, dv]);
        kernel(Some(scale), false, None, None, 0, softcap)
            .execute(
                &[
                    Owned::f32(&[b, h, sq, d], &q).view(),
                    Owned::f32(&[b, h, sk, d], &k).view(),
                    Owned::f32(&[b, h, sk, dv], &v).view(),
                ],
                &mut [out.view_mut()],
            )
            .unwrap();
        approx(&out.to_f32(), &want, 1e-5);
        assert!(
            out.to_f32()
                .iter()
                .zip(&want_nocap)
                .any(|(a, b)| (a - b).abs() > 1e-3),
            "negative softcap must change the output"
        );
    }

    // ---- Job B: opset-24 semantic deltas ----

    #[test]
    fn qk_modes_are_version_independent() {
        // All four modes have the SAME meaning across opsets 23–26 (the schema
        // descriptions are identical): 0 = raw scaled QK, 1 = after softcap,
        // 2 = after mask+softcap, 3 = after softmax. There is no 1↔2 swap.
        let (b, h, sq, sk, d, dv) = (1, 1, 1, 2, 2, 2);
        let scale = 0.5f32;
        let softcap = 3.0f32;
        let q = [1.0f32, 2.0];
        let k = [1.0f32, 1.0, 2.0, 0.0];
        let v = [1.0f32, 0.0, 0.0, 1.0];
        let maskv = [0.5f32, -1.0];
        let scaled: Vec<f32> = (0..sk)
            .map(|j| {
                let mut acc = 0.0f32;
                for p in 0..d {
                    acc += q[p] * k[j * d + p];
                }
                acc * scale
            })
            .collect();
        let capped: Vec<f32> = scaled
            .iter()
            .map(|s| softcap * (s / softcap).tanh())
            .collect();
        let masked: Vec<f32> = capped.iter().zip(&maskv).map(|(s, m)| s + m).collect();
        let softmaxed = {
            let max = masked.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut e: Vec<f32> = masked.iter().map(|s| (s - max).exp()).collect();
            let sum: f32 = e.iter().sum();
            for x in e.iter_mut() {
                *x /= sum;
            }
            e
        };
        let run = |ver: u32, mode: i64| {
            let mut y = Owned::zeros_f32(&[b, h, sq, dv]);
            let mut pk = Owned::zeros_f32(&[b, h, sk, d]);
            let mut pv = Owned::zeros_f32(&[b, h, sk, dv]);
            let mut qk = Owned::zeros_f32(&[b, h, sq, sk]);
            kernel_v(ver, Some(scale), false, None, None, mode, softcap)
                .execute(
                    &[
                        Owned::f32(&[b, h, sq, d], &q).view(),
                        Owned::f32(&[b, h, sk, d], &k).view(),
                        Owned::f32(&[b, h, sk, dv], &v).view(),
                        Owned::f32(&[sq, sk], &maskv).view(),
                    ],
                    &mut [y.view_mut(), pk.view_mut(), pv.view_mut(), qk.view_mut()],
                )
                .unwrap();
            qk.to_f32()
        };
        approx(&run(24, 0), &scaled, 1e-5);
        approx(&run(23, 0), &scaled, 1e-5);
        approx(&run(24, 3), &softmaxed, 1e-5);
        approx(&run(23, 3), &softmaxed, 1e-5);
        // Modes are version-independent: 1 = after softcap, 2 = after mask.
        approx(&run(24, 1), &capped, 1e-5);
        approx(&run(24, 2), &masked, 1e-5);
        approx(&run(23, 1), &capped, 1e-5);
        approx(&run(23, 2), &masked, 1e-5);
    }

    #[test]
    fn nonpad_kv_seqlen_offset_positive() {
        // offset = nonpad - q_seq > 0 shifts the causal diagonal to the right.
        let (b, h, sq, sk, d, dv) = (1, 1, 2, 4, 2, 2);
        let scale = 0.5f32;
        let q: Vec<f32> = (0..b * h * sq * d)
            .map(|i| (i as f32 * 0.3).sin())
            .collect();
        let k: Vec<f32> = (0..b * h * sk * d)
            .map(|i| (i as f32 * 0.2).cos())
            .collect();
        let v: Vec<f32> = (0..b * h * sk * dv).map(|i| i as f32 * 0.1).collect();
        let nonpad = [4i64];
        let offset = nonpad[0] - sq as i64; // 2
        let want = reference(
            &q,
            &k,
            &v,
            b,
            h,
            h,
            sq,
            sk,
            d,
            dv,
            scale,
            true,
            offset,
            |_, _, _, _| 0.0,
        );
        let mut out = Owned::zeros_f32(&[b, h, sq, dv]);
        kernel_v(24, Some(scale), true, None, None, 0, 0.0)
            .execute(
                &[
                    Owned::f32(&[b, h, sq, d], &q).view(),
                    Owned::f32(&[b, h, sk, d], &k).view(),
                    Owned::f32(&[b, h, sk, dv], &v).view(),
                    absent(),
                    absent(),
                    absent(),
                    Owned::i64(&[b], &nonpad).view(),
                ],
                &mut [out.view_mut()],
            )
            .unwrap();
        approx(&out.to_f32(), &want, 1e-5);
    }

    #[test]
    fn nonpad_kv_seqlen_offset_zero_is_lower_triangular() {
        // offset = 0 recovers standard lower-triangular causal masking.
        let (b, h, sq, sk, d, dv) = (1, 1, 2, 2, 2, 2);
        let scale = 0.5f32;
        let q: Vec<f32> = (0..b * h * sq * d)
            .map(|i| (i as f32 * 0.3).sin())
            .collect();
        let k: Vec<f32> = (0..b * h * sk * d)
            .map(|i| (i as f32 * 0.2).cos())
            .collect();
        let v: Vec<f32> = (0..b * h * sk * dv).map(|i| i as f32 * 0.1).collect();
        let nonpad = [2i64];
        let want = reference(
            &q,
            &k,
            &v,
            b,
            h,
            h,
            sq,
            sk,
            d,
            dv,
            scale,
            true,
            0,
            |_, _, _, _| 0.0,
        );
        let mut out = Owned::zeros_f32(&[b, h, sq, dv]);
        kernel_v(24, Some(scale), true, None, None, 0, 0.0)
            .execute(
                &[
                    Owned::f32(&[b, h, sq, d], &q).view(),
                    Owned::f32(&[b, h, sk, d], &k).view(),
                    Owned::f32(&[b, h, sk, dv], &v).view(),
                    absent(),
                    absent(),
                    absent(),
                    Owned::i64(&[b], &nonpad).view(),
                ],
                &mut [out.view_mut()],
            )
            .unwrap();
        approx(&out.to_f32(), &want, 1e-5);
    }

    #[test]
    fn nonpad_kv_seqlen_offset_negative_zeros_leading_rows() {
        // offset < 0 fully masks leading query rows → zero output rows (no NaN).
        let (b, h, sq, sk, d, dv) = (1, 1, 2, 2, 2, 2);
        let scale = 0.5f32;
        let q: Vec<f32> = (0..b * h * sq * d)
            .map(|i| (i as f32 * 0.3 + 0.1).sin())
            .collect();
        let k: Vec<f32> = (0..b * h * sk * d)
            .map(|i| (i as f32 * 0.2).cos())
            .collect();
        let v: Vec<f32> = (0..b * h * sk * dv).map(|i| i as f32 * 0.1 + 0.2).collect();
        let nonpad = [1i64];
        let offset = nonpad[0] - sq as i64; // -1
        let want = reference(
            &q,
            &k,
            &v,
            b,
            h,
            h,
            sq,
            sk,
            d,
            dv,
            scale,
            true,
            offset,
            |_, _, _, _| 0.0,
        );
        let mut out = Owned::zeros_f32(&[b, h, sq, dv]);
        kernel_v(24, Some(scale), true, None, None, 0, 0.0)
            .execute(
                &[
                    Owned::f32(&[b, h, sq, d], &q).view(),
                    Owned::f32(&[b, h, sk, d], &k).view(),
                    Owned::f32(&[b, h, sk, dv], &v).view(),
                    absent(),
                    absent(),
                    absent(),
                    Owned::i64(&[b], &nonpad).view(),
                ],
                &mut [out.view_mut()],
            )
            .unwrap();
        let got = out.to_f32();
        assert!(got.iter().all(|x| x.is_finite()), "no NaN/inf: {got:?}");
        approx(&got[0..dv], &[0.0, 0.0], 1e-6);
        approx(&got, &want, 1e-5);
    }

    #[test]
    fn nonpad_kv_seqlen_masks_padding_for_noncausal() {
        // is_causal=false with nonpad < kv_seq: keys j >= nonpad are padding in
        // the external KV cache and must contribute nothing, EVEN though no
        // causal frontier is present. Compare against a reference that masks
        // those keys to -inf, and assert it differs from the no-nonpad result.
        let (b, h, sq, sk, d, dv) = (1, 1, 2, 4, 2, 2);
        let scale = 0.5f32;
        let q: Vec<f32> = (0..b * h * sq * d)
            .map(|i| (i as f32 * 0.3).sin())
            .collect();
        let k: Vec<f32> = (0..b * h * sk * d)
            .map(|i| (i as f32 * 0.2).cos())
            .collect();
        let v: Vec<f32> = (0..b * h * sk * dv).map(|i| i as f32 * 0.1).collect();
        let nonpad = [2i64];
        // Reference: non-causal, but padding keys (j >= nonpad) masked to -inf.
        let pad_bias = |_b: usize, _h: usize, _i: usize, j: usize| {
            if (j as i64) >= nonpad[0] {
                f32::NEG_INFINITY
            } else {
                0.0
            }
        };
        let want = reference(
            &q, &k, &v, b, h, h, sq, sk, d, dv, scale, false, 0, pad_bias,
        );
        // Sanity: the padding mask must actually change the result vs. no mask.
        let no_mask = reference(
            &q,
            &k,
            &v,
            b,
            h,
            h,
            sq,
            sk,
            d,
            dv,
            scale,
            false,
            0,
            |_, _, _, _| 0.0,
        );
        assert!(
            want.iter().zip(&no_mask).any(|(a, c)| (a - c).abs() > 1e-4),
            "padding mask must differ from the unmasked result"
        );
        let mut out = Owned::zeros_f32(&[b, h, sq, dv]);
        kernel_v(24, Some(scale), false, None, None, 0, 0.0)
            .execute(
                &[
                    Owned::f32(&[b, h, sq, d], &q).view(),
                    Owned::f32(&[b, h, sk, d], &k).view(),
                    Owned::f32(&[b, h, sk, dv], &v).view(),
                    absent(),
                    absent(),
                    absent(),
                    Owned::i64(&[b], &nonpad).view(),
                ],
                &mut [out.view_mut()],
            )
            .unwrap();
        let got = out.to_f32();
        assert!(got.iter().all(|x| x.is_finite()), "no NaN/inf: {got:?}");
        approx(&got, &want, 1e-5);
    }

    #[test]
    fn nonpad_kv_seqlen_causal_and_padding_compose() {
        // Causal + nonpad together: padding keys beyond nonpad are masked AND
        // the causal frontier holds. The kernel offset is nonpad - q_seq.
        let (b, h, sq, sk, d, dv) = (1, 1, 2, 4, 2, 2);
        let scale = 0.5f32;
        let q: Vec<f32> = (0..b * h * sq * d)
            .map(|i| (i as f32 * 0.3).sin())
            .collect();
        let k: Vec<f32> = (0..b * h * sk * d)
            .map(|i| (i as f32 * 0.2).cos())
            .collect();
        let v: Vec<f32> = (0..b * h * sk * dv).map(|i| i as f32 * 0.1).collect();
        let nonpad = [3i64];
        let offset = nonpad[0] - sq as i64; // 1
        // Reference: causal frontier (via offset) intersected with padding mask.
        let pad_bias = |_b: usize, _h: usize, _i: usize, j: usize| {
            if (j as i64) >= nonpad[0] {
                f32::NEG_INFINITY
            } else {
                0.0
            }
        };
        let want = reference(
            &q, &k, &v, b, h, h, sq, sk, d, dv, scale, true, offset, pad_bias,
        );
        let mut out = Owned::zeros_f32(&[b, h, sq, dv]);
        kernel_v(24, Some(scale), true, None, None, 0, 0.0)
            .execute(
                &[
                    Owned::f32(&[b, h, sq, d], &q).view(),
                    Owned::f32(&[b, h, sk, d], &k).view(),
                    Owned::f32(&[b, h, sk, dv], &v).view(),
                    absent(),
                    absent(),
                    absent(),
                    Owned::i64(&[b], &nonpad).view(),
                ],
                &mut [out.view_mut()],
            )
            .unwrap();
        let got = out.to_f32();
        assert!(got.iter().all(|x| x.is_finite()), "no NaN/inf: {got:?}");
        approx(&got, &want, 1e-5);
    }

    #[test]
    fn nonpad_kv_seqlen_qk_output_reflects_padding_mask() {
        // mode-2 (after-mask) qk output must be -inf at padding key positions,
        // and mode-3 (post-softmax) must be 0 there, for non-causal attention.
        let (b, h, sq, sk, d, dv) = (1, 1, 2, 3, 2, 2);
        let scale = 0.5f32;
        let q: Vec<f32> = (0..b * h * sq * d)
            .map(|i| (i as f32 * 0.3).sin())
            .collect();
        let k: Vec<f32> = (0..b * h * sk * d)
            .map(|i| (i as f32 * 0.2).cos())
            .collect();
        let v: Vec<f32> = (0..b * h * sk * dv).map(|i| i as f32 * 0.1).collect();
        let nonpad = [2i64];

        // mode 2: after-mask scores. Padding column j=2 must be -inf.
        let mut y = Owned::zeros_f32(&[b, h, sq, dv]);
        let mut pk = Owned::zeros_f32(&[b, h, sk, d]);
        let mut pv = Owned::zeros_f32(&[b, h, sk, dv]);
        let mut qk = Owned::zeros_f32(&[b, h, sq, sk]);
        kernel_v(24, Some(scale), false, None, None, 2, 0.0)
            .execute(
                &[
                    Owned::f32(&[b, h, sq, d], &q).view(),
                    Owned::f32(&[b, h, sk, d], &k).view(),
                    Owned::f32(&[b, h, sk, dv], &v).view(),
                    absent(),
                    absent(),
                    absent(),
                    Owned::i64(&[b], &nonpad).view(),
                ],
                &mut [y.view_mut(), pk.view_mut(), pv.view_mut(), qk.view_mut()],
            )
            .unwrap();
        let masked = qk.to_f32();
        for i in 0..sq {
            assert_eq!(
                masked[i * sk + 2],
                f32::NEG_INFINITY,
                "mode-2 padding column must be -inf"
            );
            assert!(masked[i * sk].is_finite() && masked[i * sk + 1].is_finite());
        }

        // mode 3: post-softmax probabilities. Padding column must be 0, and each
        // row's valid probabilities sum to 1.
        let mut y3 = Owned::zeros_f32(&[b, h, sq, dv]);
        let mut pk3 = Owned::zeros_f32(&[b, h, sk, d]);
        let mut pv3 = Owned::zeros_f32(&[b, h, sk, dv]);
        let mut qk3 = Owned::zeros_f32(&[b, h, sq, sk]);
        kernel_v(24, Some(scale), false, None, None, 3, 0.0)
            .execute(
                &[
                    Owned::f32(&[b, h, sq, d], &q).view(),
                    Owned::f32(&[b, h, sk, d], &k).view(),
                    Owned::f32(&[b, h, sk, dv], &v).view(),
                    absent(),
                    absent(),
                    absent(),
                    Owned::i64(&[b], &nonpad).view(),
                ],
                &mut [
                    y3.view_mut(),
                    pk3.view_mut(),
                    pv3.view_mut(),
                    qk3.view_mut(),
                ],
            )
            .unwrap();
        let probs = qk3.to_f32();
        for i in 0..sq {
            assert_eq!(probs[i * sk + 2], 0.0, "mode-3 padding prob must be 0");
            let s = probs[i * sk] + probs[i * sk + 1];
            assert!((s - 1.0).abs() < 1e-6, "valid probs must sum to 1: {s}");
        }
    }
}
