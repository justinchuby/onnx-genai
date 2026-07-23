//! `com.microsoft::LinearAttention` — gated delta-rule linear attention
//! (Gated DeltaNet, as used by Qwen3-Next / Qwen3.5) with a per-head recurrent
//! state matrix.
//!
//! Faithful CPU port of ONNX Runtime's contrib kernel
//! (`contrib_ops/cpu/bert/linear_attention.cc`), verified to fp32 epsilon
//! against ORT 1.26 (see `kernels::qwen35_goldens`).
//!
//! ## Contract
//!
//! Inputs (packed, channels-last):
//!
//! * `query` — `[B, T, H_q · d_k]` (already L2-normalized and scaled by the
//!   producer for Qwen3.5).
//! * `key` — `[B, T, n_k · d_k]` (already L2-normalized).
//! * `value` — `[B, T, H_kv · d_v]`.
//! * `past_state` — optional `[B, H_kv, d_k, d_v]`; zeros when absent.
//! * `decay` — optional log-decay gate `g`, `[B, T, H_kv]` (per head) or
//!   `[B, T, H_kv · d_k]` (per key dim). The step decay is `exp(g)`.
//! * `beta` — optional gate, `[B, T, H_kv]` (per head) or `[B, T, 1]`.
//!
//! Outputs:
//!
//! * `output` — `[B, T, max(H_q, H_kv) · d_v]`.
//! * `present_state` — `[B, H_kv, d_k, d_v]`.
//!
//! `update_rule` selects which gates participate:
//!
//! | rule          | decay | delta (retrieval + beta) |
//! |---------------|:-----:|:------------------------:|
//! | `linear`      |   —   |            —             |
//! | `gated`       |   ✓   |            —             |
//! | `delta`       |   —   |            ✓             |
//! | `gated_delta` |   ✓   |            ✓             |
//!
//! Per timestep `t`, per kv-head `h` (state `S` is `[d_k, d_v]`, row-major
//! `S[i·d_v + j]`):
//!
//! ```text
//! decay:      S        *= exp(g_t)                       (gated / gated_delta)
//! retrieval:  r[j]      = Σ_i S[i, j] · k_t[i]           (delta / gated_delta)
//! delta:      d[j]      = beta_t · (v_t[j] − r[j])
//!             S[i, j]  += k_t[i] · d[j]
//! linear:     S[i, j]  += k_t[i] · v_t[j]                (linear / gated)
//! readout:    o_t[j]    = scale · Σ_i q_t[i] · S[i, j]   (uses the updated S)
//! ```
//!
//! `scale` defaults to `1 / sqrt(d_k)` when the attribute is `0`. GQA is
//! supported in both directions: when `H_q ≥ H_kv` several query heads share a
//! kv state (`heads_per_group = H_q / H_kv`); when `H_q < H_kv` (inverse GQA)
//! several kv states map to one query head. `n_k` (key head count) may be
//! smaller than `H_kv`, in which case several kv heads share a key head.
//!
//! Compute is in `f32` (ORT's CPU kernel is `float`-only); `f16`/`bf16` inputs
//! are widened and outputs narrowed so the kernel stays dtype-parameterized
//! (RULES.md §2) without changing the arithmetic.

use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::Node;

use super::check_arity;
use crate::dtype::{to_dense_f32_widen, write_dense_f32_narrow};

#[derive(Clone, Copy, PartialEq, Eq)]
enum UpdateRule {
    Linear,
    Gated,
    Delta,
    GatedDelta,
}

impl UpdateRule {
    fn needs_decay(self) -> bool {
        matches!(self, UpdateRule::Gated | UpdateRule::GatedDelta)
    }
    fn needs_delta(self) -> bool {
        matches!(self, UpdateRule::Delta | UpdateRule::GatedDelta)
    }
}

pub struct LinearAttentionKernel {
    q_num_heads: usize,
    kv_num_heads: usize,
    update_rule: UpdateRule,
    /// `None` => resolve to `1 / sqrt(d_k)` at execute time.
    scale: Option<f32>,
}

pub struct LinearAttentionFactory;

impl KernelFactory for LinearAttentionFactory {
    fn create(&self, node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let read_heads = |name: &str| -> Result<usize> {
            node.attr(name)
                .and_then(|a| a.as_int())
                .and_then(|v| usize::try_from(v).ok())
                .filter(|&v| v > 0)
                .ok_or_else(|| {
                    EpError::KernelFailed(format!(
                        "LinearAttention: `{name}` must be a positive integer"
                    ))
                })
        };
        let q_num_heads = read_heads("q_num_heads")?;
        let kv_num_heads = read_heads("kv_num_heads")?;

        let update_rule = match node.attr("update_rule").and_then(|a| a.as_str()) {
            Some("linear") => UpdateRule::Linear,
            Some("gated") => UpdateRule::Gated,
            Some("delta") => UpdateRule::Delta,
            None | Some("gated_delta") => UpdateRule::GatedDelta,
            Some(other) => {
                return Err(EpError::KernelFailed(format!(
                    "LinearAttention: update_rule must be one of linear, gated, delta, \
                     gated_delta; got {other:?}"
                )));
            }
        };

        // ORT resolves a `0` (or missing) scale attribute to `1 / sqrt(d_k)`.
        let scale = match node.attr("scale").and_then(|a| a.as_float()) {
            Some(s) if s != 0.0 => Some(s),
            _ => None,
        };

        Ok(Box::new(LinearAttentionKernel {
            q_num_heads,
            kv_num_heads,
            update_rule,
            scale,
        }))
    }
}

impl Kernel for LinearAttentionKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        // query, key, value required; past_state, decay, beta optional.
        check_arity("LinearAttention", inputs, outputs, 3, 6, 1)?;

        let q_shape = inputs[0].shape;
        let k_shape = inputs[1].shape;
        let v_shape = inputs[2].shape;
        if q_shape.len() != 3 || k_shape.len() != 3 || v_shape.len() != 3 {
            return Err(EpError::KernelFailed(format!(
                "LinearAttention: query/key/value must be rank 3 [B, T, H·D], got {q_shape:?}, \
                 {k_shape:?}, {v_shape:?}"
            )));
        }
        let (batch, seq, q_hidden) = (q_shape[0], q_shape[1], q_shape[2]);
        if k_shape[0] != batch || v_shape[0] != batch || k_shape[1] != seq || v_shape[1] != seq {
            return Err(EpError::KernelFailed(format!(
                "LinearAttention: query/key/value batch and sequence dims must agree; got \
                 {q_shape:?}, {k_shape:?}, {v_shape:?}"
            )));
        }

        let q_num_heads = self.q_num_heads;
        let kv_num_heads = self.kv_num_heads;
        if !q_hidden.is_multiple_of(q_num_heads) {
            return Err(EpError::KernelFailed(format!(
                "LinearAttention: query hidden {q_hidden} not divisible by q_num_heads {q_num_heads}"
            )));
        }
        let d_k = q_hidden / q_num_heads;
        if d_k == 0 || !k_shape[2].is_multiple_of(d_k) {
            return Err(EpError::KernelFailed(format!(
                "LinearAttention: key hidden {} not divisible by d_k {d_k}",
                k_shape[2]
            )));
        }
        let n_k_heads = k_shape[2] / d_k;
        if !v_shape[2].is_multiple_of(kv_num_heads) {
            return Err(EpError::KernelFailed(format!(
                "LinearAttention: value hidden {} not divisible by kv_num_heads {kv_num_heads}",
                v_shape[2]
            )));
        }
        let d_v = v_shape[2] / kv_num_heads;

        // Head mapping (mirrors ORT). Standard GQA: H_q >= H_kv. Inverse GQA:
        // H_q < H_kv (heads_per_group = 0 signals the inverse mapping).
        let heads_per_group = if q_num_heads >= kv_num_heads {
            if !q_num_heads.is_multiple_of(kv_num_heads) {
                return Err(EpError::KernelFailed(format!(
                    "LinearAttention: q_num_heads {q_num_heads} must be a multiple of kv_num_heads \
                     {kv_num_heads}"
                )));
            }
            q_num_heads / kv_num_heads
        } else {
            if !kv_num_heads.is_multiple_of(q_num_heads) {
                return Err(EpError::KernelFailed(format!(
                    "LinearAttention: kv_num_heads {kv_num_heads} must be a multiple of \
                     q_num_heads {q_num_heads} (inverse GQA)"
                )));
            }
            0
        };
        if !kv_num_heads.is_multiple_of(n_k_heads) {
            return Err(EpError::KernelFailed(format!(
                "LinearAttention: kv_num_heads {kv_num_heads} must be a multiple of n_k_heads \
                 {n_k_heads}"
            )));
        }
        let kv_per_k_head = kv_num_heads / n_k_heads;

        let scale = self.scale.unwrap_or_else(|| 1.0 / (d_k as f32).sqrt());

        let needs_decay = self.update_rule.needs_decay();
        let needs_delta = self.update_rule.needs_delta();

        // Resolve optional inputs. ONNX omits trailing optionals by shortening
        // the input list; the model always supplies all six.
        let past_state = inputs.get(3);
        let decay = inputs.get(4);
        let beta = inputs.get(5);

        if needs_decay && decay.is_none() {
            return Err(EpError::KernelFailed(
                "LinearAttention: decay input is required for update_rule=gated/gated_delta".into(),
            ));
        }
        if needs_delta && beta.is_none() {
            return Err(EpError::KernelFailed(
                "LinearAttention: beta input is required for update_rule=delta/gated_delta".into(),
            ));
        }

        // decay layout: per-head (last dim H_kv) or per-key-dim (H_kv · d_k).
        let (decay_data, decay_per_key_dim) = match decay {
            Some(view) if needs_decay => {
                let s = view.shape;
                if s.len() != 3 || s[0] != batch || s[1] != seq {
                    return Err(EpError::KernelFailed(format!(
                        "LinearAttention: decay must be [B={batch}, T={seq}, ...], got {s:?}"
                    )));
                }
                let per_key = if s[2] == kv_num_heads * d_k {
                    true
                } else if s[2] == kv_num_heads {
                    false
                } else {
                    return Err(EpError::KernelFailed(format!(
                        "LinearAttention: decay last dim must be H_kv={kv_num_heads} or \
                         H_kv·d_k={}, got {}",
                        kv_num_heads * d_k,
                        s[2]
                    )));
                };
                (Some(to_dense_f32_widen("LinearAttention", view)?), per_key)
            }
            _ => (None, false),
        };

        // beta layout: per-head (last dim H_kv) or shared (last dim 1).
        let (beta_data, beta_per_head) = match beta {
            Some(view) if needs_delta => {
                let s = view.shape;
                if s.len() != 3 || s[0] != batch || s[1] != seq {
                    return Err(EpError::KernelFailed(format!(
                        "LinearAttention: beta must be [B={batch}, T={seq}, ...], got {s:?}"
                    )));
                }
                let per_head = if s[2] == kv_num_heads {
                    true
                } else if s[2] == 1 {
                    false
                } else {
                    return Err(EpError::KernelFailed(format!(
                        "LinearAttention: beta last dim must be H_kv={kv_num_heads} or 1, got {}",
                        s[2]
                    )));
                };
                (Some(to_dense_f32_widen("LinearAttention", view)?), per_head)
            }
            _ => (None, false),
        };

        let q = to_dense_f32_widen("LinearAttention", &inputs[0])?;
        let k = to_dense_f32_widen("LinearAttention", &inputs[1])?;
        let v = to_dense_f32_widen("LinearAttention", &inputs[2])?;

        let state_per_head = d_k * d_v;
        let total_state = batch * kv_num_heads * state_per_head;
        let state_init = if let Some(view) = past_state {
            let s = view.shape;
            if s.len() != 4 || s[0] != batch || s[1] != kv_num_heads || s[2] != d_k || s[3] != d_v {
                return Err(EpError::KernelFailed(format!(
                    "LinearAttention: past_state must be [B={batch}, H_kv={kv_num_heads}, \
                     d_k={d_k}, d_v={d_v}], got {s:?}"
                )));
            }
            Some(to_dense_f32_widen("LinearAttention", view)?)
        } else {
            None
        };

        let output_hidden = q_num_heads.max(kv_num_heads) * d_v;
        let mut output = vec![0.0f32; batch * seq * output_hidden];
        let (primary_output, present_outputs) = outputs.split_at_mut(1);

        // The recurrent state is a large decode-hot tensor. When the present
        // output is native contiguous f32, use its executor-owned buffer as the
        // working state and copy the past state into it once. This avoids a
        // per-layer 1 MiB allocation plus a second full-state copy on Qwen3.5.
        let direct_state = present_outputs.first_mut().filter(|present| {
            present.dtype == onnx_runtime_ir::DataType::Float32
                && present.is_contiguous()
                && present.device.is_host_accessible()
        });
        let mut owned_state;
        let (state, state_is_direct): (&mut [f32], bool) = if let Some(present) = direct_state {
            let state = unsafe {
                std::slice::from_raw_parts_mut(present.data_ptr_mut::<f32>(), total_state)
            };
            if let Some(initial) = state_init.as_deref() {
                state.copy_from_slice(initial);
            } else {
                state.fill(0.0);
            }
            (state, true)
        } else {
            owned_state = state_init
                .map(|initial| initial.into_owned())
                .unwrap_or_else(|| vec![0.0f32; total_state]);
            (&mut owned_state, false)
        };

        let params = HeadParams {
            seq,
            d_k,
            d_v,
            q_num_heads,
            kv_num_heads,
            n_k_heads,
            heads_per_group,
            output_hidden,
            scale,
            needs_decay,
            decay_per_key_dim,
            needs_delta,
            beta_per_head,
        };

        for b in 0..batch {
            for h_kv in 0..kv_num_heads {
                let h_k = h_kv / kv_per_k_head;
                let s_off = (b * kv_num_heads + h_kv) * state_per_head;
                process_head(
                    &mut state[s_off..s_off + state_per_head],
                    &q,
                    &k,
                    &v,
                    decay_data.as_deref(),
                    beta_data.as_deref(),
                    &mut output,
                    b,
                    h_kv,
                    h_k,
                    &params,
                );
            }
        }

        write_dense_f32_narrow("LinearAttention", &mut primary_output[0], &output)?;
        if !state_is_direct && let Some(present) = present_outputs.first_mut() {
            write_dense_f32_narrow("LinearAttention", present, state)?;
        }
        Ok(())
    }
}

/// Immutable per-invocation dimensions/flags shared by every head task.
struct HeadParams {
    seq: usize,
    d_k: usize,
    d_v: usize,
    q_num_heads: usize,
    kv_num_heads: usize,
    n_k_heads: usize,
    heads_per_group: usize,
    output_hidden: usize,
    scale: f32,
    needs_decay: bool,
    decay_per_key_dim: bool,
    needs_delta: bool,
    beta_per_head: bool,
}

/// Process one `(batch, kv_head)` pair across all timesteps, updating `state`
/// in place and writing the corresponding query-head output rows. The step
/// order (decay → retrieval → delta/linear update → readout) matches ORT so the
/// arithmetic is bit-comparable.
#[allow(clippy::too_many_arguments)]
fn process_head(
    state: &mut [f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    decay: Option<&[f32]>,
    beta: Option<&[f32]>,
    output: &mut [f32],
    batch_idx: usize,
    h_kv: usize,
    h_k: usize,
    p: &HeadParams,
) {
    let d_k = p.d_k;
    let d_v = p.d_v;
    let mut retrieved = vec![0.0f32; d_v];

    for t in 0..p.seq {
        let row = batch_idx * p.seq + t;
        let kt = &k[row * (p.n_k_heads * d_k) + h_k * d_k..][..d_k];
        let vt = &v[row * (p.kv_num_heads * d_v) + h_kv * d_v..][..d_v];

        // ---- Step 1: decay S *= exp(g_t) ----
        if p.needs_decay {
            let decay = decay.expect("decay presence validated");
            if p.decay_per_key_dim {
                let gt = &decay[row * (p.kv_num_heads * d_k) + h_kv * d_k..][..d_k];
                for (i, &g) in gt.iter().enumerate() {
                    let exp_g = g.exp();
                    let s_row = &mut state[i * d_v..i * d_v + d_v];
                    for s in s_row.iter_mut() {
                        *s *= exp_g;
                    }
                }
            } else {
                let exp_g = decay[row * p.kv_num_heads + h_kv].exp();
                for s in state.iter_mut() {
                    *s *= exp_g;
                }
            }
        }

        if p.needs_delta {
            // ---- Step 2: retrieval r = Sᵀ k_t (over d_k) ----
            retrieved.fill(0.0);
            for (i, &ki) in kt.iter().enumerate() {
                let s_row = &state[i * d_v..(i + 1) * d_v];
                for (r, &s) in retrieved.iter_mut().zip(s_row) {
                    *r += s * ki;
                }
            }
            // ---- Step 3: delta update  S += k_t ⊗ (beta·(v_t − r)) ----
            let bt = if p.beta_per_head {
                beta.expect("beta presence validated")[row * p.kv_num_heads + h_kv]
            } else {
                beta.expect("beta presence validated")[row]
            };
            for (r, &vv) in retrieved.iter_mut().zip(vt.iter()) {
                *r = bt * (vv - *r);
            }
            for i in 0..d_k {
                let ki = kt[i];
                let s_row = &mut state[i * d_v..i * d_v + d_v];
                for (s, &d) in s_row.iter_mut().zip(retrieved.iter()) {
                    *s += ki * d;
                }
            }
        } else {
            // ---- linear / gated: S += k_t ⊗ v_t ----
            for i in 0..d_k {
                let ki = kt[i];
                let s_row = &mut state[i * d_v..i * d_v + d_v];
                for (s, &vv) in s_row.iter_mut().zip(vt.iter()) {
                    *s += ki * vv;
                }
            }
        }

        // ---- Step 4: readout o_t = scale · q_tᵀ S (updated S) ----
        if p.heads_per_group > 0 {
            for g in 0..p.heads_per_group {
                let h_q = h_kv * p.heads_per_group + g;
                readout(q, output, state, row, h_q, h_kv, p);
            }
        } else {
            // Inverse GQA: this kv head's query head, output slot is h_kv.
            let h_q = h_kv * p.q_num_heads / p.kv_num_heads;
            readout(q, output, state, row, h_q, h_kv, p);
        }
    }
}

/// One query-head readout `o[j] = scale · Σ_i q[i]·S[i, j]`. `out_head` is the
/// output head slot: the query head for standard GQA, the kv head for inverse.
#[inline]
fn readout(
    q: &[f32],
    output: &mut [f32],
    state: &[f32],
    row: usize,
    h_q: usize,
    h_kv: usize,
    p: &HeadParams,
) {
    let d_k = p.d_k;
    let d_v = p.d_v;
    let out_head = if p.heads_per_group > 0 { h_q } else { h_kv };
    let qt = &q[row * (p.q_num_heads * d_k) + h_q * d_k..][..d_k];
    let ot = &mut output[row * p.output_hidden + out_head * d_v..][..d_v];
    ot.fill(0.0);
    for (i, &qi) in qt.iter().enumerate() {
        let s_row = &state[i * d_v..(i + 1) * d_v];
        for (o, &s) in ot.iter_mut().zip(s_row) {
            *o += qi * s;
        }
    }
    for o in ot {
        *o *= p.scale;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::qwen35_goldens::la as g;
    use crate::kernels::testutil::Owned;
    use onnx_runtime_ir::{Attribute, Node, NodeId};

    fn bits(a: &[u32]) -> Vec<f32> {
        a.iter().map(|&b| f32::from_bits(b)).collect()
    }

    fn kernel(h: i64, scale: f32) -> Box<dyn Kernel> {
        let mut node = Node::new(NodeId(0), "LinearAttention", vec![], vec![]);
        node.domain = "com.microsoft".to_string();
        node.attributes
            .insert("q_num_heads".to_string(), Attribute::Int(h));
        node.attributes
            .insert("kv_num_heads".to_string(), Attribute::Int(h));
        node.attributes.insert(
            "update_rule".to_string(),
            Attribute::String(b"gated_delta".to_vec()),
        );
        node.attributes
            .insert("scale".to_string(), Attribute::Float(scale));
        LinearAttentionFactory.create(&node, &[]).unwrap()
    }

    fn assert_close(got: &[f32], want: &[f32], tag: &str) {
        assert_eq!(got.len(), want.len(), "{tag} length");
        for (i, (&a, &b)) in got.iter().zip(want).enumerate() {
            let diff = (a - b).abs();
            let rel = diff / b.abs().max(1e-6);
            assert!(
                diff <= 1e-4 || rel <= 1e-3,
                "{tag}[{i}]: got {a}, want {b} (abs {diff}, rel {rel})"
            );
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn run_case(
        dims: [usize; 5],
        scale: f32,
        q: &[u32],
        k: &[u32],
        v: &[u32],
        st: &[u32],
        gg: &[u32],
        be: &[u32],
    ) -> (Vec<f32>, Vec<f32>) {
        let [batch, h, dk, dv, s] = dims;
        let q = Owned::f32(&[batch, s, h * dk], &bits(q));
        let k = Owned::f32(&[batch, s, h * dk], &bits(k));
        let v = Owned::f32(&[batch, s, h * dv], &bits(v));
        let st = Owned::f32(&[batch, h, dk, dv], &bits(st));
        let gd = Owned::f32(&[batch, s, h], &bits(gg));
        let bd = Owned::f32(&[batch, s, h], &bits(be));
        let mut out = Owned::zeros_f32(&[batch, s, h * dv]);
        let mut present = Owned::zeros_f32(&[batch, h, dk, dv]);
        let ins = [
            q.view(),
            k.view(),
            v.view(),
            st.view(),
            gd.view(),
            bd.view(),
        ];
        let mut outs = [out.view_mut(), present.view_mut()];
        kernel(h as i64, scale).execute(&ins, &mut outs).unwrap();
        (out.to_f32(), present.to_f32())
    }

    #[test]
    fn ort_parity_prefill() {
        let (o, p) = run_case(
            g::LAA_DIMS,
            g::LAA_SCALE,
            &g::LAA_Q,
            &g::LAA_K,
            &g::LAA_V,
            &g::LAA_STATE,
            &g::LAA_G,
            &g::LAA_BETA,
        );
        assert_close(&o, &bits(&g::LAA_O), "LAA out");
        assert_close(&p, &bits(&g::LAA_PRESENT), "LAA present");
    }

    #[test]
    fn ort_parity_decode_with_state() {
        // S=1 decode carrying a non-zero incoming recurrent state.
        let (o, p) = run_case(
            g::LAB_DIMS,
            g::LAB_SCALE,
            &g::LAB_Q,
            &g::LAB_K,
            &g::LAB_V,
            &g::LAB_STATE,
            &g::LAB_G,
            &g::LAB_BETA,
        );
        assert_close(&o, &bits(&g::LAB_O), "LAB out");
        assert_close(&p, &bits(&g::LAB_PRESENT), "LAB present");
    }

    #[test]
    fn ort_parity_asymmetric_dk_dv_and_scale() {
        // Dk=2, Dv=5, scale=0.5, 3 heads, 2 timesteps.
        let (o, p) = run_case(
            g::LAC_DIMS,
            g::LAC_SCALE,
            &g::LAC_Q,
            &g::LAC_K,
            &g::LAC_V,
            &g::LAC_STATE,
            &g::LAC_G,
            &g::LAC_BETA,
        );
        assert_close(&o, &bits(&g::LAC_O), "LAC out");
        assert_close(&p, &bits(&g::LAC_PRESENT), "LAC present");
    }

    /// Independent hand-computed single-step, single-head gated-delta reference.
    #[test]
    fn hand_computed_single_step_recurrence() {
        // d_k = d_v = 2, one head, one timestep. state = [[1,0],[0,1]].
        let q = Owned::f32(&[1, 1, 2], &[1.0, 2.0]);
        let k = Owned::f32(&[1, 1, 2], &[0.5, -0.5]);
        let v = Owned::f32(&[1, 1, 2], &[3.0, 4.0]);
        let st = Owned::f32(&[1, 1, 2, 2], &[1.0, 0.0, 0.0, 1.0]);
        let g_log = Owned::f32(&[1, 1, 1], &[-0.5]); // decay = exp(-0.5)
        let beta = Owned::f32(&[1, 1, 1], &[0.25]);
        let mut out = Owned::zeros_f32(&[1, 1, 2]);
        let mut present = Owned::zeros_f32(&[1, 1, 2, 2]);
        let ins = [
            q.view(),
            k.view(),
            v.view(),
            st.view(),
            g_log.view(),
            beta.view(),
        ];
        let mut outs = [out.view_mut(), present.view_mut()];
        kernel(1, 1.0).execute(&ins, &mut outs).unwrap();

        // Reference computed by hand:
        let decay = (-0.5f32).exp();
        // S after decay: [[decay,0],[0,decay]]
        let mut s = [[decay, 0.0f32], [0.0, decay]];
        let kk = [0.5f32, -0.5];
        let vv = [3.0f32, 4.0];
        // retrieval r[j] = sum_i S[i][j]*k[i]
        let r = [
            s[0][0] * kk[0] + s[1][0] * kk[1],
            s[0][1] * kk[0] + s[1][1] * kk[1],
        ];
        let bt = 0.25f32;
        let d = [bt * (vv[0] - r[0]), bt * (vv[1] - r[1])];
        for i in 0..2 {
            for j in 0..2 {
                s[i][j] += kk[i] * d[j];
            }
        }
        let qq = [1.0f32, 2.0];
        let o_ref = [
            qq[0] * s[0][0] + qq[1] * s[1][0],
            qq[0] * s[0][1] + qq[1] * s[1][1],
        ];
        let present_ref = [s[0][0], s[0][1], s[1][0], s[1][1]];
        assert_close(&out.to_f32(), &o_ref, "hand out");
        assert_close(&present.to_f32(), &present_ref, "hand present");
    }

    #[test]
    fn scale_zero_resolves_to_inv_sqrt_dk() {
        // scale attribute 0.0 -> 1/sqrt(d_k). Verify by comparing to explicit.
        let mut node = Node::new(NodeId(0), "LinearAttention", vec![], vec![]);
        node.domain = "com.microsoft".to_string();
        node.attributes
            .insert("q_num_heads".to_string(), Attribute::Int(1));
        node.attributes
            .insert("kv_num_heads".to_string(), Attribute::Int(1));
        node.attributes.insert(
            "update_rule".to_string(),
            Attribute::String(b"gated_delta".to_vec()),
        );
        node.attributes
            .insert("scale".to_string(), Attribute::Float(0.0));
        let k0 = LinearAttentionFactory.create(&node, &[]).unwrap();

        let q = Owned::f32(&[1, 1, 4], &[1.0, 2.0, 3.0, 4.0]);
        let k = Owned::f32(&[1, 1, 4], &[0.1, 0.2, 0.3, 0.4]);
        let v = Owned::f32(&[1, 1, 2], &[1.0, -1.0]);
        let st = Owned::f32(&[1, 1, 4, 2], &[0.0; 8]);
        let gd = Owned::f32(&[1, 1, 1], &[-0.1]);
        let bd = Owned::f32(&[1, 1, 1], &[0.5]);
        let mut o0 = Owned::zeros_f32(&[1, 1, 2]);
        let mut p0 = Owned::zeros_f32(&[1, 1, 4, 2]);
        let ins = [
            q.view(),
            k.view(),
            v.view(),
            st.view(),
            gd.view(),
            bd.view(),
        ];
        let mut outs = [o0.view_mut(), p0.view_mut()];
        k0.execute(&ins, &mut outs).unwrap();

        // Same with explicit scale = 1/sqrt(4) = 0.5.
        let mut oe = Owned::zeros_f32(&[1, 1, 2]);
        let mut pe = Owned::zeros_f32(&[1, 1, 4, 2]);
        let ins2 = [
            q.view(),
            k.view(),
            v.view(),
            st.view(),
            gd.view(),
            bd.view(),
        ];
        let mut outs2 = [oe.view_mut(), pe.view_mut()];
        kernel(1, 0.5).execute(&ins2, &mut outs2).unwrap();
        assert_close(&o0.to_f32(), &oe.to_f32(), "scale0 vs explicit");
    }

    #[test]
    fn rejects_missing_beta_for_gated_delta() {
        let k = kernel(1, 1.0);
        let q = Owned::f32(&[1, 1, 2], &[1.0, 2.0]);
        let kk = Owned::f32(&[1, 1, 2], &[0.5, -0.5]);
        let v = Owned::f32(&[1, 1, 2], &[3.0, 4.0]);
        let st = Owned::f32(&[1, 1, 2, 2], &[1.0, 0.0, 0.0, 1.0]);
        let gd = Owned::f32(&[1, 1, 1], &[-0.5]);
        let mut out = Owned::zeros_f32(&[1, 1, 2]);
        let mut present = Owned::zeros_f32(&[1, 1, 2, 2]);
        // Only 5 inputs -> beta missing.
        let ins = [q.view(), kk.view(), v.view(), st.view(), gd.view()];
        let mut outs = [out.view_mut(), present.view_mut()];
        assert!(k.execute(&ins, &mut outs).is_err());
    }
}
