//! `com.microsoft::CausalConvWithState` — causal depthwise 1-D convolution with
//! a rolling state cache (Mamba / linear-attention "short conv").
//!
//! Faithful CPU port of ONNX Runtime's contrib kernel
//! (`contrib_ops/cpu/bert/causal_conv_with_state.cc`), verified to fp32 epsilon
//! against ORT 1.26 (see `kernels::qwen35_goldens`). The op keeps the trailing
//! `K-1` input frames from the previous step in `conv_state` so that decode
//! (`seq == 1`) stays causal across autoregressive steps.
//!
//! ## Contract (`ndim = 1`)
//!
//! * `x` — `[B, C, L]` activations (channels-first).
//! * `weight` — `[C, 1, K]` depthwise filter (one filter per channel,
//!   `group == C`).
//! * `bias` — optional `[C]`.
//! * `past_state` — optional `[B, C, K-1]`; treated as zeros when absent.
//!
//! Outputs:
//!
//! * `y` — `[B, C, L]`.
//! * `present_state` — `[B, C, K-1]`.
//!
//! Per `(b, c)` let `seq = concat(past_state[b, c, :], x[b, c, :])` (length
//! `(K-1) + L`). Then
//!
//! ```text
//! y[b, c, t] = activation( bias[c] + Σ_{j=0..K-1} weight[c, 0, j] · seq[t + j] )
//! present_state[b, c, :] = seq[-(K-1):]
//! ```
//!
//! `activation` is one of `none` (passthrough), `silu` or `swish` (both apply
//! `x·sigmoid(x)`; the op has no learnable Swish β, so they are identical). The
//! SiLU pass reuses the shared MLAS-backed [`silu_f32_slice`], matching ORT's
//! logistic numerics (RULES.md §4).
//!
//! Compute is in `f32` (ORT's CPU kernel is `float`-only); `f16`/`bf16`
//! activations are widened to `f32` and the results narrowed back so the kernel
//! stays dtype-parameterized (RULES.md §2) without changing the arithmetic.

use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::Node;

use super::check_arity;
use crate::dtype::{to_dense_f32_widen, write_dense_f32_narrow};
use crate::kernels::activations::silu_f32_slice;

/// Post-convolution activation. `Swish` with the op's implicit `β = 1` is
/// identical to `Silu`, so both map to the same SiLU pass.
#[derive(Clone, Copy, PartialEq, Eq)]
enum ConvActivation {
    None,
    Silu,
}

pub struct CausalConvWithStateKernel {
    activation: ConvActivation,
}

pub struct CausalConvWithStateFactory;

impl KernelFactory for CausalConvWithStateFactory {
    fn create(&self, node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let ndim = node.attr("ndim").and_then(|a| a.as_int()).unwrap_or(1);
        if ndim != 1 {
            return Err(EpError::KernelFailed(format!(
                "CausalConvWithState: only ndim=1 (channels-first [B, C, L]) is supported by the \
                 CPU kernel, got ndim={ndim}"
            )));
        }
        let activation = match node.attr("activation").and_then(|a| a.as_str()) {
            None | Some("none") => ConvActivation::None,
            Some("silu") | Some("swish") => ConvActivation::Silu,
            Some(other) => {
                return Err(EpError::KernelFailed(format!(
                    "CausalConvWithState: activation must be one of none, silu, swish; got {other:?}"
                )));
            }
        };
        Ok(Box::new(CausalConvWithStateKernel { activation }))
    }
}

impl Kernel for CausalConvWithStateKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        // x, weight required; bias and past_state optional.
        check_arity("CausalConvWithState", inputs, outputs, 2, 4, 1)?;

        let x_shape = inputs[0].shape;
        if x_shape.len() != 3 {
            return Err(EpError::KernelFailed(format!(
                "CausalConvWithState: input must be rank 3 [B, C, L] for ndim=1, got shape {x_shape:?}"
            )));
        }
        let (batch, channels, length) = (x_shape[0], x_shape[1], x_shape[2]);

        let w_shape = inputs[1].shape;
        if w_shape.len() != 3 || w_shape[0] != channels || w_shape[1] != 1 {
            return Err(EpError::KernelFailed(format!(
                "CausalConvWithState: weight must be depthwise [C, 1, K] with C={channels}, got \
                 shape {w_shape:?}"
            )));
        }
        let kernel_size = w_shape[2];
        if kernel_size == 0 {
            return Err(EpError::KernelFailed(
                "CausalConvWithState: kernel size K must be >= 1".into(),
            ));
        }
        let pad = kernel_size - 1;

        let has_bias = inputs.len() >= 3;
        let has_state = inputs.len() >= 4;

        let x = to_dense_f32_widen("CausalConvWithState", &inputs[0])?;
        let weight = to_dense_f32_widen("CausalConvWithState", &inputs[1])?;
        let bias = if has_bias {
            let b = to_dense_f32_widen("CausalConvWithState", &inputs[2])?;
            if b.len() != channels {
                return Err(EpError::KernelFailed(format!(
                    "CausalConvWithState: bias must be 1D with size C={channels}, got {} elements",
                    b.len()
                )));
            }
            Some(b)
        } else {
            None
        };
        let state = if has_state {
            let s_shape = inputs[3].shape;
            if s_shape.len() != 3
                || s_shape[0] != batch
                || s_shape[1] != channels
                || s_shape[2] != pad
            {
                return Err(EpError::KernelFailed(format!(
                    "CausalConvWithState: past_state must be [B={batch}, C={channels}, K-1={pad}], \
                     got shape {s_shape:?}"
                )));
            }
            Some(to_dense_f32_widen("CausalConvWithState", &inputs[3])?)
        } else {
            None
        };

        let (primary_output, present_outputs) = outputs.split_at_mut(1);
        let out_len = batch * channels * length;
        let direct_out = primary_output[0].dtype == onnx_runtime_ir::DataType::Float32
            && primary_output[0].is_contiguous()
            && primary_output[0].device.is_host_accessible();
        let mut owned_out;
        let out: &mut [f32] = if direct_out {
            // SAFETY: the executor owns an exclusive contiguous output buffer
            // whose validated shape contains exactly `out_len` f32 values.
            unsafe {
                std::slice::from_raw_parts_mut(primary_output[0].data_ptr_mut::<f32>(), out_len)
            }
        } else {
            owned_out = vec![0.0f32; out_len];
            &mut owned_out
        };

        let present_len = batch * channels * pad;
        let direct_present = present_outputs.first().is_some_and(|present| {
            present.dtype == onnx_runtime_ir::DataType::Float32
                && present.is_contiguous()
                && present.device.is_host_accessible()
        });
        let mut owned_present;
        let mut present = if let Some(output) = present_outputs.first_mut() {
            if direct_present {
                // SAFETY: same output-buffer contract as `out`, with
                // `present_len` validated by the shape checks above.
                Some(unsafe {
                    std::slice::from_raw_parts_mut(output.data_ptr_mut::<f32>(), present_len)
                })
            } else {
                owned_present = vec![0.0f32; present_len];
                Some(owned_present.as_mut_slice())
            }
        } else {
            None
        };

        // Depthwise: every (batch, channel) is fully independent. `window`
        // slides over `[state | x]`; the accumulation order matches ORT's inner
        // `for k in 0..K` loop so the pre-activation values are bit-comparable.
        for b in 0..batch {
            for c in 0..channels {
                let x_row = &x[(b * channels + c) * length..(b * channels + c) * length + length];
                let w_row = &weight[c * kernel_size..c * kernel_size + kernel_size];
                let bias_c = bias.as_ref().map_or(0.0, |bv| bv[c]);
                let state_row = state
                    .as_ref()
                    .map(|sv| &sv[(b * channels + c) * pad..(b * channels + c) * pad + pad]);

                let out_row =
                    &mut out[(b * channels + c) * length..(b * channels + c) * length + length];
                if length == 1 {
                    let mut acc = bias_c;
                    for (k, &w) in w_row[..pad].iter().enumerate() {
                        acc += w * state_row.map_or(0.0, |s| s[k]);
                    }
                    acc += w_row[pad] * x_row[0];
                    out_row[0] = acc;
                } else {
                    for (t, out_t) in out_row.iter_mut().enumerate() {
                        let mut acc = bias_c;
                        for (k, &w) in w_row.iter().enumerate() {
                            // Position `t + k` in the virtual `[state(pad) | x(L)]`
                            // sequence: the first `pad` slots come from state.
                            let pos = t + k;
                            let val = if pos < pad {
                                state_row.map_or(0.0, |s| s[pos])
                            } else {
                                x_row[pos - pad]
                            };
                            acc += w * val;
                        }
                        *out_t = acc;
                    }
                }

                // present_state = last `pad` frames of `[state | x]`.
                if let Some(present) = present.as_deref_mut() {
                    let present_row =
                        &mut present[(b * channels + c) * pad..(b * channels + c) * pad + pad];
                    if length == 1 && pad > 0 {
                        for (slot, source) in present_row[..pad - 1].iter_mut().zip(1..pad) {
                            *slot = state_row.map_or(0.0, |s| s[source]);
                        }
                        present_row[pad - 1] = x_row[0];
                    } else {
                        for (p, slot) in present_row.iter_mut().enumerate() {
                            // Global index into the virtual sequence of the p-th tail
                            // frame: (pad + L) - pad + p = L + p.
                            let pos = length + p;
                            *slot = if pos < pad {
                                state_row.map_or(0.0, |s| s[pos])
                            } else {
                                x_row[pos - pad]
                            };
                        }
                    }
                }
            }
        }

        if self.activation == ConvActivation::Silu {
            // SiLU over the dense output (MLAS logistic where available,
            // matching ORT's activation numerics).
            let mut activated = vec![0.0f32; out.len()];
            silu_f32_slice(out, &mut activated);
            out.copy_from_slice(&activated);
        }

        if !direct_out {
            write_dense_f32_narrow("CausalConvWithState", &mut primary_output[0], out)?;
        }
        if !direct_present
            && let (Some(output), Some(present)) = (present_outputs.first_mut(), present.as_deref())
        {
            write_dense_f32_narrow("CausalConvWithState", output, present)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::qwen35_goldens::conv as g;
    use crate::kernels::testutil::Owned;
    use onnx_runtime_ir::{Attribute, DataType, Node, NodeId};

    fn bits(a: &[u32]) -> Vec<f32> {
        a.iter().map(|&b| f32::from_bits(b)).collect()
    }

    fn kernel(silu: bool) -> Box<dyn Kernel> {
        let mut node = Node::new(NodeId(0), "CausalConvWithState", vec![], vec![]);
        node.domain = "com.microsoft".to_string();
        node.attributes
            .insert("ndim".to_string(), Attribute::Int(1));
        node.attributes.insert(
            "activation".to_string(),
            Attribute::String(if silu {
                b"silu".to_vec()
            } else {
                b"none".to_vec()
            }),
        );
        CausalConvWithStateFactory.create(&node, &[]).unwrap()
    }

    #[allow(clippy::too_many_arguments)]
    fn run_case(
        dims: [usize; 4],
        silu: bool,
        x: &[u32],
        w: &[u32],
        b: &[u32],
        st: &[u32],
    ) -> (Vec<f32>, Vec<f32>) {
        let [batch, c, k, s] = dims;
        let pad = k - 1;
        let x = Owned::f32(&[batch, c, s], &bits(x));
        let w = Owned::f32(&[c, 1, k], &bits(w));
        let bias = Owned::f32(&[c], &bits(b));
        let state = Owned::f32(&[batch, c, pad], &bits(st));
        let mut y = Owned::zeros_f32(&[batch, c, s]);
        let mut present = Owned::zeros_f32(&[batch, c, pad]);
        let ins = [x.view(), w.view(), bias.view(), state.view()];
        let mut outs = [y.view_mut(), present.view_mut()];
        kernel(silu).execute(&ins, &mut outs).unwrap();
        (y.to_f32(), present.to_f32())
    }

    fn assert_close(got: &[f32], want: &[f32], tag: &str) {
        assert_eq!(got.len(), want.len(), "{tag} length");
        for (i, (&a, &b)) in got.iter().zip(want).enumerate() {
            let diff = (a - b).abs();
            let rel = diff / b.abs().max(1e-6);
            assert!(
                diff <= 1e-4 || rel <= 1e-4,
                "{tag}[{i}]: got {a}, want {b} (abs {diff}, rel {rel})"
            );
        }
    }

    #[test]
    fn ort_parity_prefill_silu() {
        let (y, p) = run_case(
            g::CONVA_DIMS,
            g::CONVA_SILU,
            &g::CONVA_X,
            &g::CONVA_W,
            &g::CONVA_B,
            &g::CONVA_STATE,
        );
        assert_close(&y, &bits(&g::CONVA_Y), "CONVA y");
        assert_close(&p, &bits(&g::CONVA_PRESENT), "CONVA present");
    }

    #[test]
    fn ort_parity_decode_silu() {
        // S=1 decode: present_state mixes trailing past-state frames with x.
        let (y, p) = run_case(
            g::CONVB_DIMS,
            g::CONVB_SILU,
            &g::CONVB_X,
            &g::CONVB_W,
            &g::CONVB_B,
            &g::CONVB_STATE,
        );
        assert_close(&y, &bits(&g::CONVB_Y), "CONVB y");
        assert_close(&p, &bits(&g::CONVB_PRESENT), "CONVB present");
    }

    #[test]
    fn ort_parity_no_activation_k3() {
        let (y, p) = run_case(
            g::CONVC_DIMS,
            g::CONVC_SILU,
            &g::CONVC_X,
            &g::CONVC_W,
            &g::CONVC_B,
            &g::CONVC_STATE,
        );
        assert_close(&y, &bits(&g::CONVC_Y), "CONVC y");
        assert_close(&p, &bits(&g::CONVC_PRESENT), "CONVC present");
    }

    #[test]
    fn optional_bias_and_state_default_to_zero() {
        // Without bias/state: a single-channel K=2 conv over [0, x] = pure taps.
        let mut node = Node::new(NodeId(0), "CausalConvWithState", vec![], vec![]);
        node.domain = "com.microsoft".to_string();
        node.attributes
            .insert("ndim".to_string(), Attribute::Int(1));
        node.attributes.insert(
            "activation".to_string(),
            Attribute::String(b"none".to_vec()),
        );
        let kern = CausalConvWithStateFactory.create(&node, &[]).unwrap();
        // x = [c][t] = [[1, 2]], w = [w0, w1] = [3, 5]
        let x = Owned::f32(&[1, 1, 2], &[1.0, 2.0]);
        let w = Owned::f32(&[1, 1, 2], &[3.0, 5.0]);
        let mut y = Owned::zeros_f32(&[1, 1, 2]);
        let mut present = Owned::zeros_f32(&[1, 1, 1]);
        let ins = [x.view(), w.view()];
        let mut outs = [y.view_mut(), present.view_mut()];
        kern.execute(&ins, &mut outs).unwrap();
        // seq = [state(0), 1, 2]; y[0] = w0*0 + w1*1 = 5; y[1] = w0*1 + w1*2 = 13.
        assert_eq!(y.to_f32(), vec![5.0, 13.0]);
        // present = last 1 frame of [0, 1, 2] = 2.
        assert_eq!(present.to_f32(), vec![2.0]);
    }

    #[test]
    fn rejects_unknown_activation() {
        let mut node = Node::new(NodeId(0), "CausalConvWithState", vec![], vec![]);
        node.domain = "com.microsoft".to_string();
        node.attributes.insert(
            "activation".to_string(),
            Attribute::String(b"gelu".to_vec()),
        );
        assert!(CausalConvWithStateFactory.create(&node, &[]).is_err());
    }

    #[test]
    fn f16_activations_widen_and_narrow() {
        // Same math as CONVC but with f16 in/out; tolerance loosened for half.
        let [batch, c, k, s] = g::CONVC_DIMS;
        let pad = k - 1;
        let x = Owned::f16(&[batch, c, s], &bits(&g::CONVC_X));
        let w = Owned::f16(&[c, 1, k], &bits(&g::CONVC_W));
        let bias = Owned::f16(&[c], &bits(&g::CONVC_B));
        let state = Owned::f16(&[batch, c, pad], &bits(&g::CONVC_STATE));
        let mut y = Owned::zeros(DataType::Float16, &[batch, c, s]);
        let mut present = Owned::zeros(DataType::Float16, &[batch, c, pad]);
        let ins = [x.view(), w.view(), bias.view(), state.view()];
        let mut outs = [y.view_mut(), present.view_mut()];
        kernel(false).execute(&ins, &mut outs).unwrap();
        let got = y.to_f16_as_f32();
        let want = bits(&g::CONVC_Y);
        for (i, (&a, &b)) in got.iter().zip(&want).enumerate() {
            assert!((a - b).abs() <= 2e-2, "f16 y[{i}]: got {a}, want {b}");
        }
    }
}
