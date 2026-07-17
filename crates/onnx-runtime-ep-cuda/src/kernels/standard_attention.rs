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

use std::borrow::Cow;
use std::ffi::c_void;
use std::sync::Arc;

use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{DataType, Node};

use crate::runtime::{CudaRuntime, cuptr};

/// Return the claim-time dtype denial for data-bearing Attention inputs.
pub(crate) fn unsupported_reason(input_dtypes: &[DataType]) -> Option<Cow<'static, str>> {
    for &index in &[0, 1, 2, 4, 5] {
        let Some(&dtype) = input_dtypes.get(index) else {
            continue;
        };
        if dtype != DataType::Float32 {
            let dtype = match dtype {
                DataType::Float16 => "f16".into(),
                DataType::BFloat16 => "bf16".into(),
                other => format!("{other:?}"),
            };
            return Some(Cow::Owned(format!(
                "Attention: dtype {dtype} not supported on CUDA yet (f32 only; f16/bf16 follow-up)"
            )));
        }
    }
    None
}

/// f32 standard-`Attention` kernel carrying the resolved attributes.
pub struct StandardAttentionKernel {
    runtime: Arc<CudaRuntime>,
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
pub struct StandardAttentionFactory {
    pub runtime: Arc<CudaRuntime>,
    pub since_version: u32,
}

impl KernelFactory for StandardAttentionFactory {
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
        Ok(Box::new(StandardAttentionKernel {
            runtime: self.runtime.clone(),
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

fn check_arity(
    name: &str,
    inputs: &[TensorView],
    outputs: &[TensorMut],
    min: usize,
    max: usize,
    min_outputs: usize,
) -> Result<()> {
    if !(min..=max).contains(&inputs.len()) || outputs.len() < min_outputs {
        return Err(EpError::KernelFailed(format!(
            "{name}: expected {min}..={max} inputs and at least {min_outputs} outputs"
        )));
    }
    Ok(())
}

fn dense_bytes(runtime: &CudaRuntime, view: &TensorView) -> Result<Vec<u8>> {
    if !view.is_contiguous() {
        return Err(EpError::KernelFailed(
            "Attention: non-contiguous inputs are not supported".into(),
        ));
    }
    let mut bytes = vec![0u8; view.dtype.storage_bytes(view.numel())];
    unsafe {
        runtime.dtoh(&mut bytes, cuptr(view.data_ptr::<u8>() as *const c_void))?;
    }
    Ok(bytes)
}

fn dense_f32(runtime: &CudaRuntime, view: &TensorView) -> Result<Vec<f32>> {
    if view.dtype != DataType::Float32 {
        return Err(EpError::KernelFailed(format!(
            "Attention: expected f32 input, got {:?}",
            view.dtype
        )));
    }
    Ok(dense_bytes(runtime, view)?
        .chunks_exact(4)
        .map(|b| f32::from_ne_bytes(b.try_into().unwrap()))
        .collect())
}

fn dense_i64(runtime: &CudaRuntime, view: &TensorView) -> Result<Vec<i64>> {
    if view.dtype != DataType::Int64 {
        return Err(EpError::KernelFailed(
            "Attention: nonpad_kv_seqlen must be int64".into(),
        ));
    }
    Ok(dense_bytes(runtime, view)?
        .chunks_exact(8)
        .map(|b| i64::from_ne_bytes(b.try_into().unwrap()))
        .collect())
}

fn write_f32(runtime: &CudaRuntime, output: &mut TensorMut, values: &[f32]) -> Result<()> {
    if output.dtype != DataType::Float32
        || !output.is_contiguous()
        || output.numel() != values.len()
    {
        return Err(EpError::KernelFailed(
            "Attention: output must be contiguous f32 with the expected shape".into(),
        ));
    }
    let bytes = unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
    };
    unsafe {
        runtime.htod(bytes, cuptr(output.data_ptr_mut::<u8>() as *const c_void))?;
    }
    Ok(())
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
fn to_bhsd(
    runtime: &CudaRuntime,
    view: &TensorView,
    name: &str,
    num_heads: Option<usize>,
) -> Result<Bhsd> {
    let shape = view.shape;
    match shape.len() {
        4 => {
            let (batch, heads, seq, dim) = (shape[0], shape[1], shape[2], shape[3]);
            let data = dense_f32(runtime, view)?;
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
            let src = dense_f32(runtime, view)?;
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

impl Kernel for StandardAttentionKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        check_arity("Attention", inputs, outputs, 3, 7, 1)?;
        // Inputs may have been uploaded asynchronously on the EP stream.
        self.runtime.synchronize()?;

        let q_rank = inputs[0].shape.len();
        let q = to_bhsd(&self.runtime, &inputs[0], "Q", self.q_num_heads)?;
        let k_cur = to_bhsd(&self.runtime, &inputs[1], "K", self.kv_num_heads)?;
        let v_cur = to_bhsd(&self.runtime, &inputs[2], "V", self.kv_num_heads)?;

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
            Some(to_bhsd(
                &self.runtime,
                &inputs[4],
                "past_key",
                self.kv_num_heads,
            )?)
        } else {
            None
        };
        let past_value = if has_past_value {
            Some(to_bhsd(
                &self.runtime,
                &inputs[5],
                "past_value",
                self.kv_num_heads,
            )?)
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
            let seqlen = dense_i64(&self.runtime, &inputs[6])?;
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
                    data: dense_bytes(&self.runtime, m)?
                        .iter()
                        .map(|&b| b != 0)
                        .collect(),
                    shape: m.shape.to_vec(),
                },
                DataType::Float32 => Mask::Float {
                    data: dense_f32(&self.runtime, m)?,
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
            write_f32(&self.runtime, &mut outputs[0], &y3)?;
        } else {
            write_f32(&self.runtime, &mut outputs[0], &y)?;
        }

        // present_key / present_value (outputs 1 and 2), always 4D.
        if outputs.len() >= 2 {
            write_f32(&self.runtime, &mut outputs[1], &key.data)?;
        }
        if outputs.len() >= 3 {
            write_f32(&self.runtime, &mut outputs[2], &value.data)?;
        }
        if want_qk {
            write_f32(&self.runtime, &mut outputs[3], &qk_out)?;
        }
        self.runtime.synchronize()?;
        Ok(())
    }

    fn supports_strided_input(&self, _input_idx: usize) -> bool {
        false
    }
}
