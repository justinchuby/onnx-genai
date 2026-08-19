//! `com.microsoft::FusedAttention`: the optimizer's fusion of the scaled
//! dot-product-attention (SDPA) core into a single node (`docs/architecture/ORT2.md` §18.2).
//!
//! The fused node computes
//!
//! ```text
//! out = Softmax( (Q · Kᵀ) * scale  [+ mask] , axis=-1 ) · V
//! ```
//!
//! reproducing exactly the `MatMul → (Mul|Div) → [Add] → Softmax → MatMul`
//! chain the [`AttentionFusion`](onnx_runtime_optimizer) pass matched and
//! replaced. Inputs are `[Q, K, V]` plus an OPTIONAL trailing `[mask]`;
//! attributes are:
//!
//! * `scale` (f32) — the concrete multiplier folded from the score scaling
//!   (`* c` for a `Mul`, `1/c` for a `Div`), applied to the raw `Q·Kᵀ` scores.
//! * `k_transposed` (int, 0/1) — whether the `K` input is **already**
//!   transposed into `[…, head_dim, seq_k]` (`1`, the score `MatMul` consumed it
//!   as-is, so the kernel does a plain `Q·K`) or is in `[…, seq_k, head_dim]`
//!   layout (`0`, the kernel transposes its last two dims internally to form
//!   `Kᵀ`). This lets the matcher optionally absorb a clean last-two-axis
//!   `Transpose` that produced `Kᵀ`.
//!
//! Every stage reuses a shared single-source-of-truth helper: the batched
//! [`matmul_dense`](super::matmul::matmul_dense) GEMM for both `Q·Kᵀ` and
//! `probs·V`, and
//! [`scale_mask_softmax_rows`](super::softmax::scale_mask_softmax_rows) for the
//! scale/mask/softmax stage — the same numerically-stable (max-subtract)
//! reduction the standalone `Softmax` kernel uses, with the additive mask
//! resolved through [`broadcast_effective_strides`](super::add::broadcast_effective_strides)
//! so the broadcast is folded into the parallel fan-out rather than walked
//! serially beforehand.

use onnx_runtime_ep_api::{
    DeviceId, EpError, Kernel, KernelFactory, Result, TensorMut, TensorView,
};
use onnx_runtime_ir::{DataType, Node, broadcast_shapes, compute_contiguous_strides};

use super::add::broadcast_effective_strides;
use super::check_arity;
use super::matmul::matmul_dense;
use super::softmax::{MaskPlan, scale_mask_softmax_rows};
use crate::dtype::{to_dense_f32_widen, write_dense_f32_narrow};
use crate::strided::numel;

/// f32 SDPA kernel carrying the folded `scale` and the `k_transposed` flag.
pub struct FusedAttentionKernel {
    scale: f32,
    /// Whether the `K` input is already `[…, head_dim, seq_k]` (score MatMul
    /// consumed it as-is → plain `Q·K`); otherwise the kernel transposes K's
    /// last two dims to form `Kᵀ`.
    k_transposed: bool,
}

/// Factory for [`FusedAttentionKernel`], reading the `scale` and
/// `k_transposed` attributes the optimizer synthesized.
pub struct FusedAttentionFactory;

impl KernelFactory for FusedAttentionFactory {
    fn create(&self, node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let scale = node
            .attr("scale")
            .and_then(|a| a.as_float())
            .ok_or_else(|| {
                EpError::KernelFailed("FusedAttention: missing f32 `scale` attribute".into())
            })?;
        let k_transposed = node
            .attr("k_transposed")
            .and_then(|a| a.as_int())
            .unwrap_or(0)
            != 0;
        Ok(Box::new(FusedAttentionKernel {
            scale,
            k_transposed,
        }))
    }
}

/// The `[…, m, n]` shape of `MatMul(a, b)` for rank-≥2 operands (attention
/// tensors are always at least `[batch, seq, dim]`), broadcasting the leading
/// batch dims and checking the contraction dim.
fn matmul_result_shape(a: &[usize], b: &[usize], stage: &str) -> Result<Vec<usize>> {
    if a.len() < 2 || b.len() < 2 {
        return Err(EpError::KernelFailed(format!(
            "FusedAttention: {stage} operands must be rank ≥ 2 (got {a:?}, {b:?})"
        )));
    }
    let (m, ka) = (a[a.len() - 2], a[a.len() - 1]);
    let (kb, n) = (b[b.len() - 2], b[b.len() - 1]);
    if ka != kb {
        return Err(EpError::KernelFailed(format!(
            "FusedAttention: {stage} contraction mismatch ({ka} vs {kb})"
        )));
    }
    let mut shape = broadcast_shapes(&a[..a.len() - 2], &b[..b.len() - 2]).map_err(EpError::Ir)?;
    shape.push(m);
    shape.push(n);
    Ok(shape)
}

impl Kernel for FusedAttentionKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        check_arity("FusedAttention", inputs, outputs, 3, 4, 1)?;
        let q = &inputs[0];
        let k = &inputs[1];
        let v = &inputs[2];
        let has_mask = inputs.len() == 4;

        if q.shape.len() < 2 || k.shape.len() < 2 || v.shape.len() < 2 {
            return Err(EpError::KernelFailed(
                "FusedAttention: Q, K, V must each be rank ≥ 2".into(),
            ));
        }

        // Stage 1: scores = Q · Kᵀ. When `k_transposed`, K is already Kᵀ and we
        // multiply as-is; otherwise expose Kᵀ as a strided view (swap the last
        // two axes) so the shared GEMM never copies to transpose.
        let (scores, scores_shape) = if self.k_transposed {
            let s = matmul_dense(q, k)?;
            let shape = matmul_result_shape(q.shape, k.shape, "Q·K")?;
            (s, shape)
        } else {
            let rank = k.shape.len();
            let mut kt_shape = k.shape.to_vec();
            kt_shape.swap(rank - 2, rank - 1);
            let mut kt_strides = k.strides.to_vec();
            kt_strides.swap(rank - 2, rank - 1);
            let kt = TensorView::new(k.data, k.dtype, &kt_shape, &kt_strides, k.device)
                .with_byte_offset(k.byte_offset);
            let s = matmul_dense(q, &kt)?;
            let shape = matmul_result_shape(q.shape, &kt_shape, "Q·Kᵀ")?;
            (s, shape)
        };

        // Stages 2+3 fused: scale, optional additive mask, and the last-axis
        // softmax in a single parallel pass over score rows.
        //
        // Previously these were three full passes over the score matrix — a
        // serial scalar `*= scale`, a serial element-at-a-time broadcast add
        // through a closure, and only then the (parallel) softmax. The score
        // matrix is `batch·heads·seq_q·seq_k` floats, by far the largest buffer
        // here (33 MiB for a 32-head 512-token prefill), so two serial passes
        // over it capped the whole kernel's thread scaling: the masked cells
        // could not go faster than their serial prologue no matter how many
        // workers the softmax had. Folding them into the softmax fan-out means
        // each row is scaled, masked and normalized while it is still hot in
        // cache, and every element-touching stage is parallel.
        let seq_k = *scores_shape.last().unwrap();
        let n = numel(&scores_shape);
        let outer = n.checked_div(seq_k).unwrap_or(0);
        // `scores` is dead after this point, so the whole pipeline runs in
        // place - allocating a second score-sized buffer only to copy into it
        // was pure overhead.
        let mut probs = scores;
        if has_mask {
            let mask = to_dense_f32_widen("FusedAttention", &inputs[3])?;
            let eff = broadcast_effective_strides(inputs[3].shape, &scores_shape)?;
            scale_mask_softmax_rows(
                &mut probs,
                outer,
                seq_k,
                self.scale,
                Some(MaskPlan {
                    values: &mask,
                    eff: &eff,
                    scores_shape: &scores_shape,
                }),
            );
        } else {
            scale_mask_softmax_rows(&mut probs, outer, seq_k, self.scale, None);
        }

        // Stage 4: out = probs · V. Wrap the dense `probs` buffer (contiguous
        // over `scores_shape`) as a view so the shared GEMM handles the batched
        // multiply and leading-dim broadcast against V.
        let probs_strides = compute_contiguous_strides(&scores_shape);
        let probs_view = TensorView::new(
            onnx_runtime_ep_api::DevicePtr(probs.as_ptr() as *const std::ffi::c_void),
            DataType::Float32,
            &scores_shape,
            &probs_strides,
            DeviceId::cpu(),
        );
        let out = matmul_dense(&probs_view, v)?;
        write_dense_f32_narrow("FusedAttention", &mut outputs[0], &out)
    }

    fn supports_strided_input(&self, _input_idx: usize) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::testutil::Owned;

    /// Reference SDPA on 2-D `[seq_q, d]` Q, `[seq_k, d]` K, `[seq_k, dv]` V:
    /// `Softmax((Q·Kᵀ)*scale [+ mask], axis=-1) · V`, computed with independent
    /// small-matrix loops so it cross-checks the kernel's fused single pass.
    #[allow(clippy::too_many_arguments)]
    fn reference(
        q: &[f32],
        sq: usize,
        d: usize,
        k: &[f32],
        sk: usize,
        v: &[f32],
        dv: usize,
        scale: f32,
        mask: Option<&[f32]>,
    ) -> Vec<f32> {
        // scores[i,j] = scale * sum_p Q[i,p]*K[j,p]  (+ mask[i,j])
        let mut scores = vec![0.0f32; sq * sk];
        for i in 0..sq {
            for j in 0..sk {
                let mut acc = 0.0f32;
                for p in 0..d {
                    acc += q[i * d + p] * k[j * d + p];
                }
                scores[i * sk + j] = acc * scale + mask.map_or(0.0, |m| m[i * sk + j]);
            }
        }
        // row softmax
        for i in 0..sq {
            let row = &mut scores[i * sk..i * sk + sk];
            let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for x in row.iter_mut() {
                *x = (*x - max).exp();
                sum += *x;
            }
            for x in row.iter_mut() {
                *x /= sum;
            }
        }
        // out[i,c] = sum_j probs[i,j]*V[j,c]
        let mut out = vec![0.0f32; sq * dv];
        for i in 0..sq {
            for c in 0..dv {
                let mut acc = 0.0f32;
                for j in 0..sk {
                    acc += scores[i * sk + j] * v[j * dv + c];
                }
                out[i * dv + c] = acc;
            }
        }
        out
    }

    fn approx(a: &[f32], b: &[f32], atol: f32) {
        assert_eq!(a.len(), b.len(), "length mismatch");
        for (i, (x, y)) in a.iter().zip(b).enumerate() {
            assert!(
                (x - y).abs() < atol,
                "element {i}: {x} vs {y} ({a:?} vs {b:?})"
            );
        }
    }

    #[test]
    fn sdpa_unmasked_pretransposed_k_matches_reference() {
        // sq=2, d=3, sk=2, dv=2. K given pre-transposed as [d, sk] (k_transposed=1).
        let q = [1.0f32, 0.0, -1.0, 0.5, 2.0, 1.0];
        let k_natural = [1.0f32, 2.0, 0.0, -1.0, 1.0, 3.0]; // [sk=2, d=3]
        let v = [1.0f32, 0.0, 0.0, 2.0]; // [sk=2, dv=2]
        let scale = 0.5f32;

        // Pre-transposed K^T is [d=3, sk=2].
        let mut kt = vec![0.0f32; 3 * 2];
        for j in 0..2 {
            for p in 0..3 {
                kt[p * 2 + j] = k_natural[j * 3 + p];
            }
        }
        let want = reference(&q, 2, 3, &k_natural, 2, &v, 2, scale, None);

        let qv = Owned::f32(&[2, 3], &q);
        let kv = Owned::f32(&[3, 2], &kt);
        let vv = Owned::f32(&[2, 2], &v);
        let mut out = Owned::zeros_f32(&[2, 2]);
        FusedAttentionKernel {
            scale,
            k_transposed: true,
        }
        .execute(&[qv.view(), kv.view(), vv.view()], &mut [out.view_mut()])
        .unwrap();
        approx(&out.to_f32(), &want, 1e-6);
    }

    #[test]
    fn sdpa_unmasked_internal_transpose_k_matches_reference() {
        // Same data, but K given in natural [sk, d] layout with k_transposed=0
        // so the kernel transposes it internally. Result must be identical.
        let q = [1.0f32, 0.0, -1.0, 0.5, 2.0, 1.0];
        let k_natural = [1.0f32, 2.0, 0.0, -1.0, 1.0, 3.0]; // [sk=2, d=3]
        let v = [1.0f32, 0.0, 0.0, 2.0];
        let scale = 0.5f32;
        let want = reference(&q, 2, 3, &k_natural, 2, &v, 2, scale, None);

        let qv = Owned::f32(&[2, 3], &q);
        let kv = Owned::f32(&[2, 3], &k_natural);
        let vv = Owned::f32(&[2, 2], &v);
        let mut out = Owned::zeros_f32(&[2, 2]);
        FusedAttentionKernel {
            scale,
            k_transposed: false,
        }
        .execute(&[qv.view(), kv.view(), vv.view()], &mut [out.view_mut()])
        .unwrap();
        approx(&out.to_f32(), &want, 1e-6);
    }

    #[test]
    fn sdpa_masked_matches_reference() {
        // sq=2, d=2, sk=3, dv=2, additive mask that heavily suppresses one key.
        let q = [1.0f32, 2.0, -1.0, 0.5];
        let k_natural = [1.0f32, 0.0, 0.0, 1.0, 1.0, 1.0]; // [sk=3, d=2]
        let v = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // [sk=3, dv=2]
        let mask = [0.0f32, -1e4, 0.0, 0.0, 0.0, -1e4]; // [sq=2, sk=3]
        let scale = 0.7f32;
        let want = reference(&q, 2, 2, &k_natural, 3, &v, 2, scale, Some(&mask));

        // K pre-transposed to [d=2, sk=3].
        let mut kt = vec![0.0f32; 2 * 3];
        for j in 0..3 {
            for p in 0..2 {
                kt[p * 3 + j] = k_natural[j * 2 + p];
            }
        }
        let qv = Owned::f32(&[2, 2], &q);
        let kv = Owned::f32(&[2, 3], &kt);
        let vv = Owned::f32(&[3, 2], &v);
        let mv = Owned::f32(&[2, 3], &mask);
        let mut out = Owned::zeros_f32(&[2, 2]);
        FusedAttentionKernel {
            scale,
            k_transposed: true,
        }
        .execute(
            &[qv.view(), kv.view(), vv.view(), mv.view()],
            &mut [out.view_mut()],
        )
        .unwrap();
        approx(&out.to_f32(), &want, 1e-6);
    }

    #[test]
    fn sdpa_batched_leading_dims() {
        // [batch=2, heads=1, seq, dim] shapes — the real BERT SDPA rank-4 case.
        // Two independent batches with different Q so outputs differ per batch.
        let sq = 2;
        let d = 2;
        let sk = 2;
        let dv = 2;
        let scale = 0.25f32;

        // batch 0 and batch 1 Q/K/V, laid out contiguously.
        let q = [
            1.0f32, 0.0, 0.0, 1.0, // batch 0
            2.0, 1.0, -1.0, 0.5, // batch 1
        ];
        let k_nat = [
            1.0f32, 1.0, 0.0, 2.0, // batch 0
            0.5, -1.0, 1.0, 1.0, // batch 1
        ];
        let v = [
            1.0f32, 0.0, 0.0, 1.0, // batch 0
            2.0, 3.0, 4.0, 5.0, // batch 1
        ];

        // K^T per batch: [batch,heads,d,sk].
        let mut kt = vec![0.0f32; q.len()];
        for b in 0..2 {
            for j in 0..sk {
                for p in 0..d {
                    kt[b * d * sk + p * sk + j] = k_nat[b * sk * d + j * d + p];
                }
            }
        }

        let qv = Owned::f32(&[2, 1, sq, d], &q);
        let kv = Owned::f32(&[2, 1, d, sk], &kt);
        let vv = Owned::f32(&[2, 1, sk, dv], &v);
        let mut out = Owned::zeros_f32(&[2, 1, sq, dv]);
        FusedAttentionKernel {
            scale,
            k_transposed: true,
        }
        .execute(&[qv.view(), kv.view(), vv.view()], &mut [out.view_mut()])
        .unwrap();
        let got = out.to_f32();

        // Reference per batch.
        for b in 0..2usize {
            let want = reference(
                &q[b * sq * d..(b + 1) * sq * d],
                sq,
                d,
                &k_nat[b * sk * d..(b + 1) * sk * d],
                sk,
                &v[b * sk * dv..(b + 1) * sk * dv],
                dv,
                scale,
                None,
            );
            approx(&got[b * sq * dv..(b + 1) * sq * dv], &want, 1e-6);
        }
    }

    /// Deterministic pseudorandom stream so a failure is reproducible.
    fn lcg(seed: u64, n: usize) -> Vec<f32> {
        let mut x = seed;
        (0..n)
            .map(|_| {
                x = x
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((x >> 33) as f32 / (1u64 << 31) as f32) - 1.0
            })
            .collect()
    }

    /// The scale/mask/softmax stage runs as one parallel fan-out, so every
    /// worker resolves its own mask offsets from a *global* row index rather
    /// than walking the mask serially from the start. This shape is large
    /// enough to be split across both rayon chunks and row tiles, and the mask
    /// differs per (batch, query) row by a wide margin, so any off-by-one in
    /// that arithmetic makes a row attend to the wrong keys and shows up here.
    #[test]
    fn sdpa_masked_broadcast_matches_reference_across_parallel_chunks() {
        // rows = b*h*sq = 512 and
        // rows*sk = 524288 (> MIN_PARALLEL_SOFTMAX_ELEMENTS), so the work is
        // split across workers. A 1024-key row is 4 KiB, so a `ROW_TILE_BYTES`
        // tile holds only 8 rows and every worker also loops over tiles —
        // including on a single-core host, where the whole 512 rows arrive as
        // one serial run of 64 tiles.
        let (b, h, sq, sk, d, dv) = (2usize, 4usize, 64usize, 1024usize, 4usize, 4usize);
        let scale = 0.125f32;
        let q = lcg(1, b * h * sq * d);
        let k_nat = lcg(2, b * h * sk * d);
        let v = lcg(3, b * h * sk * dv);

        // Kᵀ per (batch, head): [b, h, d, sk].
        let mut kt = vec![0.0f32; k_nat.len()];
        for bh in 0..b * h {
            for j in 0..sk {
                for p in 0..d {
                    kt[bh * d * sk + p * sk + j] = k_nat[bh * sk * d + j * d + p];
                }
            }
        }

        // Mask [b, 1, sq, sk] — broadcasts over heads, contiguous over keys.
        // A causal-style cutoff that shifts with both batch and query row.
        let mut mask = vec![0.0f32; b * sq * sk];
        for bi in 0..b {
            for i in 0..sq {
                let cutoff = (i + 1) * sk / sq - bi * 3;
                for j in 0..sk {
                    mask[(bi * sq + i) * sk + j] = if j >= cutoff { -1.0e4 } else { 0.0 };
                }
            }
        }

        let qv = Owned::f32(&[b, h, sq, d], &q);
        let kv = Owned::f32(&[b, h, d, sk], &kt);
        let vv = Owned::f32(&[b, h, sk, dv], &v);
        let mv = Owned::f32(&[b, 1, sq, sk], &mask);
        let mut out = Owned::zeros_f32(&[b, h, sq, dv]);
        FusedAttentionKernel {
            scale,
            k_transposed: true,
        }
        .execute(
            &[qv.view(), kv.view(), vv.view(), mv.view()],
            &mut [out.view_mut()],
        )
        .unwrap();
        let got = out.to_f32();

        for bi in 0..b {
            for hi in 0..h {
                let bh = bi * h + hi;
                let want = reference(
                    &q[bh * sq * d..(bh + 1) * sq * d],
                    sq,
                    d,
                    &k_nat[bh * sk * d..(bh + 1) * sk * d],
                    sk,
                    &v[bh * sk * dv..(bh + 1) * sk * dv],
                    dv,
                    scale,
                    Some(&mask[bi * sq * sk..(bi + 1) * sq * sk]),
                );
                approx(&got[bh * sq * dv..(bh + 1) * sq * dv], &want, 1e-5);
            }
        }
    }

    /// A mask of extent 1 on the key axis takes the `step == 0` branch, where
    /// one mask value is splatted across the whole score row. Same parallel
    /// shape, so it also covers the chunk/tile split.
    #[test]
    fn sdpa_key_broadcast_mask_matches_reference() {
        let (b, h, sq, sk, d, dv) = (2usize, 4usize, 16usize, 256usize, 4usize, 4usize);
        let scale = 0.5f32;
        let q = lcg(11, b * h * sq * d);
        let k_nat = lcg(12, b * h * sk * d);
        let v = lcg(13, b * h * sk * dv);
        let mut kt = vec![0.0f32; k_nat.len()];
        for bh in 0..b * h {
            for j in 0..sk {
                for p in 0..d {
                    kt[bh * d * sk + p * sk + j] = k_nat[bh * sk * d + j * d + p];
                }
            }
        }

        // Mask [sq, 1]: right-aligns to the query axis, extent 1 over keys.
        let per_row = lcg(14, sq);
        let mut expanded = vec![0.0f32; sq * sk];
        for i in 0..sq {
            for j in 0..sk {
                expanded[i * sk + j] = per_row[i];
            }
        }

        let qv = Owned::f32(&[b, h, sq, d], &q);
        let kv = Owned::f32(&[b, h, d, sk], &kt);
        let vv = Owned::f32(&[b, h, sk, dv], &v);
        let mv = Owned::f32(&[sq, 1], &per_row);
        let mut out = Owned::zeros_f32(&[b, h, sq, dv]);
        FusedAttentionKernel {
            scale,
            k_transposed: true,
        }
        .execute(
            &[qv.view(), kv.view(), vv.view(), mv.view()],
            &mut [out.view_mut()],
        )
        .unwrap();
        let got = out.to_f32();

        for bh in 0..b * h {
            let want = reference(
                &q[bh * sq * d..(bh + 1) * sq * d],
                sq,
                d,
                &k_nat[bh * sk * d..(bh + 1) * sk * d],
                sk,
                &v[bh * sk * dv..(bh + 1) * sk * dv],
                dv,
                scale,
                Some(&expanded),
            );
            approx(&got[bh * sq * dv..(bh + 1) * sq * dv], &want, 1e-5);
        }
    }

    #[test]
    fn sdpa_softmax_stage_matches_row_softmax() {
        // Cross-check that the fused softmax stage produces a proper row-softmax:
        // with an identity V and no mask, the output equals softmax(scores),
        // proving the extracted softmax helper is applied over the last
        // axis. scores = Q·Kᵀ with Q = identity, so scores == Kᵀ.
        let q = [1.0f32, 0.0, 0.0, 1.0]; // [2,2] identity
        let kt = [1.0f32, 3.0, 2.0, 4.0]; // K^T [2,2]
        let v = [1.0f32, 0.0, 0.0, 1.0]; // identity => out == softmax(scores)
        let mut out = Owned::zeros_f32(&[2, 2]);
        FusedAttentionKernel {
            scale: 1.0,
            k_transposed: true,
        }
        .execute(
            &[
                Owned::f32(&[2, 2], &q).view(),
                Owned::f32(&[2, 2], &kt).view(),
                Owned::f32(&[2, 2], &v).view(),
            ],
            &mut [out.view_mut()],
        )
        .unwrap();

        // Rows [1,3] and [2,4] both softmax to [0.11920, 0.88080].
        approx(
            &out.to_f32(),
            &[0.119_202_92, 0.880_797_1, 0.119_202_92, 0.880_797_1],
            1e-6,
        );
        // Each output row (a softmax distribution) sums to 1.
        let r = out.to_f32();
        assert!((r[0] + r[1] - 1.0).abs() < 1e-6);
        assert!((r[2] + r[3] - 1.0).abs() < 1e-6);
    }
    #[test]
    fn fused_attention_bf16_matches_widened_f32_reference() {
        let q = Owned::bf16(&[1, 2], &[1., -1.]);
        let k = Owned::bf16(&[2, 2], &[1., 0., 0., 1.]);
        let v = Owned::bf16(&[2, 2], &[2., -1., -2., 3.]);
        let mut out = Owned::zeros(DataType::BFloat16, &[1, 2]);
        FusedAttentionKernel {
            scale: 1.,
            k_transposed: false,
        }
        .execute(&[q.view(), k.view(), v.view()], &mut [out.view_mut()])
        .unwrap();
        let want = reference(
            &q.to_bf16_as_f32(),
            1,
            2,
            &k.to_bf16_as_f32(),
            2,
            &v.to_bf16_as_f32(),
            2,
            1.,
            None,
        );
        let want: Vec<_> = want
            .into_iter()
            .map(half::bf16::from_f32)
            .map(half::bf16::to_f32)
            .collect();
        assert_eq!(out.to_bf16_as_f32(), want);
    }
}
