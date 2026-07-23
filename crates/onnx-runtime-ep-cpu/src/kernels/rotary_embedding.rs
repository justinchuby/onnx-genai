//! Standard `ai.onnx::RotaryEmbedding` (opset 23): rotary position embedding
//! (RoPE) applied to query/key token embeddings.
//!
//! Faithful port of the ONNX reference
//! (`onnx/reference/ops/op_rotary_embedding.py`). The op rotates pairs of
//! channels in the last (head) dimension by a position-dependent angle supplied
//! precomputed as `cos_cache` / `sin_cache`:
//!
//! ```text
//! real = cos·x1 - sin·x2
//! imag = sin·x1 + cos·x2
//! ```
//!
//! where `(x1, x2)` are either the two halves of the rotary sub-vector
//! (`interleaved=0`, the GPT-NeoX / rotate-half convention) or adjacent
//! even/odd channels (`interleaved=1`, the GPT-J convention).
//!
//! ## Inputs / attributes (per the spec)
//!
//! * `X` — 4D `(batch, num_heads, seq, head_size)` or 3D
//!   `(batch, seq, hidden)`. For the 3D form `num_heads` (attribute) must be
//!   set and `hidden = num_heads·head_size`.
//! * `cos_cache`, `sin_cache` — when `position_ids` is provided: 2D
//!   `(max_pos+1, rotary_dim/2)`, gathered by position. When `position_ids` is
//!   absent: 3D `(batch, seq, rotary_dim/2)`, indexed directly.
//! * `position_ids` (optional) — 2D `(batch, seq)` integer indices.
//! * `interleaved` (default 0), `num_heads` (default 0), `rotary_embedding_dim`
//!   (default 0 → full rotation over `head_size`).
//!
//! The same `cos`/`sin` row applies to every head at a given `(batch, seq)`.
//! Channels at or beyond `rotary_embedding_dim` pass through unrotated.

use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::Node;

use super::{check_arity, to_dense_f32, to_dense_i64, write_dense_f32};

/// f32 RotaryEmbedding kernel carrying the resolved attributes.
pub struct RotaryEmbeddingKernel {
    interleaved: bool,
    num_heads: usize,
    rotary_embedding_dim: usize,
    /// `com.microsoft::RotaryEmbedding` orders inputs as
    /// `(X, position_ids, cos_cache, sin_cache)`; the standard `ai.onnx` op uses
    /// `(X, cos_cache, sin_cache, position_ids?)`. The rotation math is identical.
    contrib: bool,
}

/// Factory reading `interleaved` (0), `num_heads` (0), `rotary_embedding_dim` (0).
pub struct RotaryEmbeddingFactory;

impl KernelFactory for RotaryEmbeddingFactory {
    fn create(&self, node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        rotary_kernel_from_node(node, false)
    }
}

/// Factory for the `com.microsoft::RotaryEmbedding` contrib op, which orders its
/// inputs as `(X, position_ids, cos_cache, sin_cache)`.
pub struct RotaryEmbeddingContribFactory;

impl KernelFactory for RotaryEmbeddingContribFactory {
    fn create(&self, node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        rotary_kernel_from_node(node, true)
    }
}

fn rotary_kernel_from_node(node: &Node, contrib: bool) -> Result<Box<dyn Kernel>> {
    let interleaved = node
        .attr("interleaved")
        .and_then(|a| a.as_int())
        .unwrap_or(0)
        != 0;
    let num_heads = node
        .attr("num_heads")
        .and_then(|a| a.as_int())
        .unwrap_or(0)
        .max(0) as usize;
    let rotary_embedding_dim = node
        .attr("rotary_embedding_dim")
        .and_then(|a| a.as_int())
        .unwrap_or(0)
        .max(0) as usize;
    Ok(Box::new(RotaryEmbeddingKernel {
        interleaved,
        num_heads,
        rotary_embedding_dim,
        contrib,
    }))
}

impl Kernel for RotaryEmbeddingKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        check_arity("RotaryEmbedding", inputs, outputs, 3, 4, 1)?;
        // Input order differs between the standard and contrib ops.
        let (cos_i, sin_i, pos_i) = if self.contrib {
            // com.microsoft: (X, position_ids, cos_cache, sin_cache).
            if inputs.len() < 4 {
                return Err(EpError::KernelFailed(
                    "RotaryEmbedding (com.microsoft): expected 4 inputs \
                     (X, position_ids, cos_cache, sin_cache)"
                        .into(),
                ));
            }
            (2, 3, Some(1))
        } else {
            // ai.onnx: (X, cos_cache, sin_cache, position_ids?).
            (1, 2, if inputs.len() == 4 { Some(3) } else { None })
        };
        let x = to_dense_f32(&inputs[0])?;
        let cos_cache = to_dense_f32(&inputs[cos_i])?;
        let sin_cache = to_dense_f32(&inputs[sin_i])?;
        let position_ids = match pos_i {
            Some(i) => Some(to_dense_i64(&inputs[i])?),
            None => None,
        };

        let x_shape = inputs[0].shape;
        // Resolve batch/heads/seq/head_size in the canonical [B, S, H, D] view.
        let (batch, seq, heads, head_size, is_4d) = match x_shape.len() {
            4 => {
                // [batch, num_heads, seq, head_size]
                (x_shape[0], x_shape[2], x_shape[1], x_shape[3], true)
            }
            3 => {
                if self.num_heads == 0 {
                    return Err(EpError::KernelFailed(
                        "RotaryEmbedding: num_heads must be set for a 3D input".into(),
                    ));
                }
                let hidden = x_shape[2];
                if !hidden.is_multiple_of(self.num_heads) {
                    return Err(EpError::KernelFailed(format!(
                        "RotaryEmbedding: hidden {hidden} not divisible by num_heads {}",
                        self.num_heads
                    )));
                }
                (
                    x_shape[0],
                    x_shape[1],
                    self.num_heads,
                    hidden / self.num_heads,
                    false,
                )
            }
            r => {
                return Err(EpError::KernelFailed(format!(
                    "RotaryEmbedding: X must be rank 3 or 4, got rank {r}"
                )));
            }
        };

        let rotary_dim = if self.rotary_embedding_dim == 0 {
            head_size
        } else {
            self.rotary_embedding_dim
        };
        if rotary_dim > head_size || !rotary_dim.is_multiple_of(2) {
            return Err(EpError::KernelFailed(format!(
                "RotaryEmbedding: rotary_embedding_dim {rotary_dim} invalid for head_size {head_size}"
            )));
        }
        let half = rotary_dim / 2;

        // Zero-sized input: nothing to rotate. Emit an empty output rather than
        // underflowing on the `batch-1`/`seq-1` bounds computation below.
        if x.is_empty() {
            return write_dense_f32(&mut outputs[0], &[]);
        }

        let expected_cache_shape = if position_ids.is_some() {
            if inputs[cos_i].shape.len() != 2 {
                return Err(EpError::KernelFailed(format!(
                    "RotaryEmbedding: with position_ids, cos_cache/sin_cache must be rank 2 [max_position,{half}], got {:?}",
                    inputs[cos_i].shape
                )));
            }
            inputs[cos_i].shape[1] == half
        } else {
            inputs[cos_i].shape == [batch, seq, half]
        };
        if inputs[sin_i].shape != inputs[cos_i].shape || !expected_cache_shape {
            return Err(EpError::KernelFailed(format!(
                "RotaryEmbedding: cos_cache/sin_cache shapes must match the resolved rotary dimension {rotary_dim}; got cos={:?}, sin={:?}",
                inputs[cos_i].shape, inputs[sin_i].shape
            )));
        }

        // With `position_ids` present, validate its shape matches [batch, seq].
        if let Some(pos) = &position_ids {
            let pos_shape = inputs[pos_i.expect("pos index present")].shape;
            let expected = batch * seq;
            if pos.len() != expected {
                return Err(EpError::KernelFailed(format!(
                    "RotaryEmbedding: position_ids has {} elements, expected {expected} ([batch={batch}, seq={seq}]); shape {pos_shape:?}",
                    pos.len()
                )));
            }
        }

        // cos/sin lookup: with position_ids the caches are 2D [max_pos, half]
        // gathered by position; without, they are 3D [batch, seq, half]. Every
        // requested row is bounds-checked (a gathered position may exceed the
        // cache extent even when the final position does not).
        let cache_stride = half; // last-dim size of both cache layouts
        let cache_row = |b: usize, s: usize| -> Result<usize> {
            let row = if let Some(pos) = &position_ids {
                let p = pos[b * seq + s];
                if p < 0 {
                    return Err(EpError::KernelFailed(
                        "RotaryEmbedding: negative position id".into(),
                    ));
                }
                usize::try_from(p).map_err(|_| {
                    EpError::KernelFailed(
                        "RotaryEmbedding: position id exceeds supported range".into(),
                    )
                })?
            } else {
                b * seq + s
            };
            let offset = row.checked_mul(cache_stride).ok_or_else(|| {
                EpError::KernelFailed(format!(
                    "RotaryEmbedding: position {row} exceeds cos/sin cache extent"
                ))
            })?;
            let end = offset.checked_add(half).ok_or_else(|| {
                EpError::KernelFailed(format!(
                    "RotaryEmbedding: position {row} exceeds cos/sin cache extent"
                ))
            })?;
            if offset > cos_cache.len()
                || end > cos_cache.len()
                || offset > sin_cache.len()
                || end > sin_cache.len()
            {
                return Err(EpError::KernelFailed(format!(
                    "RotaryEmbedding: position {row} exceeds cos/sin cache extent (row width {half})"
                )));
            }
            Ok(offset)
        };

        // Flat index of element (b, h, s, d) in X's native layout.
        let idx = |b: usize, h: usize, s: usize, d: usize| -> usize {
            if is_4d {
                // [B, H, S, D]
                ((b * heads + h) * seq + s) * head_size + d
            } else {
                // [B, S, H*D]
                (b * seq + s) * (heads * head_size) + h * head_size + d
            }
        };

        let mut y = vec![0.0f32; x.len()];
        for b in 0..batch {
            for s in 0..seq {
                let crow = cache_row(b, s)?;
                for h in 0..heads {
                    // Rotary sub-vector.
                    for k in 0..half {
                        let cos = cos_cache[crow + k];
                        let sin = sin_cache[crow + k];
                        let (d1, d2) = if self.interleaved {
                            (2 * k, 2 * k + 1)
                        } else {
                            (k, k + half)
                        };
                        let x1 = x[idx(b, h, s, d1)];
                        let x2 = x[idx(b, h, s, d2)];
                        y[idx(b, h, s, d1)] = cos * x1 - sin * x2;
                        y[idx(b, h, s, d2)] = sin * x1 + cos * x2;
                    }
                    // Pass-through channels beyond the rotary sub-vector.
                    for d in rotary_dim..head_size {
                        y[idx(b, h, s, d)] = x[idx(b, h, s, d)];
                    }
                }
            }
        }

        write_dense_f32(&mut outputs[0], &y)
    }

    fn supports_strided_input(&self, _input_idx: usize) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::testutil::Owned;

    #[test]
    fn rope_half_rotation_hand_computed() {
        // 4D X [1,1,1,4]: head_size=4, half=2, full rotation, non-interleaved.
        // x = [1, 2, 3, 4]; x1=[1,2], x2=[3,4].
        // cos=[c0,c1], sin=[s0,s1] at position 0.
        let c0 = 0.5f32;
        let c1 = 0.8f32;
        let s0 = (1.0f32 - c0 * c0).sqrt();
        let s1 = (1.0f32 - c1 * c1).sqrt();
        let x = Owned::f32(&[1, 1, 1, 4], &[1., 2., 3., 4.]);
        // 3D caches [B,S,half] = [1,1,2] (no position_ids).
        let cos = Owned::f32(&[1, 1, 2], &[c0, c1]);
        let sin = Owned::f32(&[1, 1, 2], &[s0, s1]);
        let mut out = Owned::zeros_f32(&[1, 1, 1, 4]);
        RotaryEmbeddingKernel {
            interleaved: false,
            num_heads: 0,
            rotary_embedding_dim: 0,
            contrib: false,
        }
        .execute(&[x.view(), cos.view(), sin.view()], &mut [out.view_mut()])
        .unwrap();
        // real = cos*x1 - sin*x2; imag = sin*x1 + cos*x2 (concat: [real, imag]).
        let want = [
            c0 * 1.0 - s0 * 3.0,
            c1 * 2.0 - s1 * 4.0,
            s0 * 1.0 + c0 * 3.0,
            s1 * 2.0 + c1 * 4.0,
        ];
        for (g, w) in out.to_f32().iter().zip(&want) {
            assert!((g - w).abs() < 1e-6, "got {g}, want {w}");
        }
    }

    #[test]
    fn rope_interleaved_hand_computed() {
        // Same values, interleaved: x1=even=[1,3], x2=odd=[2,4].
        let c0 = 0.5f32;
        let c1 = 0.8f32;
        let s0 = (1.0f32 - c0 * c0).sqrt();
        let s1 = (1.0f32 - c1 * c1).sqrt();
        let x = Owned::f32(&[1, 1, 1, 4], &[1., 2., 3., 4.]);
        let cos = Owned::f32(&[1, 1, 2], &[c0, c1]);
        let sin = Owned::f32(&[1, 1, 2], &[s0, s1]);
        let mut out = Owned::zeros_f32(&[1, 1, 1, 4]);
        RotaryEmbeddingKernel {
            interleaved: true,
            num_heads: 0,
            rotary_embedding_dim: 0,
            contrib: false,
        }
        .execute(&[x.view(), cos.view(), sin.view()], &mut [out.view_mut()])
        .unwrap();
        // out[0]=real0, out[1]=imag0, out[2]=real1, out[3]=imag1.
        let want = [
            c0 * 1.0 - s0 * 2.0,
            s0 * 1.0 + c0 * 2.0,
            c1 * 3.0 - s1 * 4.0,
            s1 * 3.0 + c1 * 4.0,
        ];
        for (g, w) in out.to_f32().iter().zip(&want) {
            assert!((g - w).abs() < 1e-6, "got {g}, want {w}");
        }
    }

    #[test]
    fn rope_zero_angle_is_identity() {
        // cos=1, sin=0 → output equals input regardless of layout.
        let x = Owned::f32(&[1, 2, 1, 4], &[1., 2., 3., 4., 5., 6., 7., 8.]);
        let cos = Owned::f32(&[1, 1, 2], &[1., 1.]);
        let sin = Owned::f32(&[1, 1, 2], &[0., 0.]);
        let mut out = Owned::zeros_f32(&[1, 1, 2, 4]);
        RotaryEmbeddingKernel {
            interleaved: false,
            num_heads: 0,
            rotary_embedding_dim: 0,
            contrib: false,
        }
        .execute(&[x.view(), cos.view(), sin.view()], &mut [out.view_mut()])
        .unwrap();
        assert_eq!(out.to_f32(), vec![1., 2., 3., 4., 5., 6., 7., 8.]);
    }

    #[test]
    fn rope_3d_with_num_heads_and_position_ids() {
        // 3D X [1,2,4]: hidden=4, num_heads=2 → head_size=2, half=1.
        // position_ids [1,2] = [0, 1] gathering 2D caches [max_pos=2, half=1].
        let x = Owned::f32(&[1, 2, 4], &[1., 2., 3., 4., 5., 6., 7., 8.]);
        let cos = Owned::f32(&[2, 1], &[1.0, 0.0]); // pos0: cos=1; pos1: cos=0
        let sin = Owned::f32(&[2, 1], &[0.0, 1.0]); // pos0: sin=0; pos1: sin=1
        let pos = Owned::i64(&[1, 2], &[0, 1]);
        let mut out = Owned::zeros_f32(&[1, 2, 4]);
        RotaryEmbeddingKernel {
            interleaved: false,
            num_heads: 2,
            rotary_embedding_dim: 0,
            contrib: false,
        }
        .execute(
            &[x.view(), cos.view(), sin.view(), pos.view()],
            &mut [out.view_mut()],
        )
        .unwrap();
        // head_size=2, half=1, non-interleaved: x1=d0, x2=d1.
        // seq0 (pos0, cos=1,sin=0): identity → [1,2,3,4].
        // seq1 (pos1, cos=0,sin=1): real=-x2, imag=x1.
        //   head0: x=[5,6] → [-6, 5]; head1: x=[7,8] → [-8, 7].
        let want = [1., 2., 3., 4., -6., 5., -8., 7.];
        for (g, w) in out.to_f32().iter().zip(&want) {
            assert!((g - w).abs() < 1e-6, "got {g}, want {w}");
        }
    }

    #[test]
    fn rope_partial_rotary_dim_passes_through_tail() {
        // head_size=4, rotary_embedding_dim=2 → only first 2 channels rotate.
        let x = Owned::f32(&[1, 1, 1, 4], &[1., 2., 3., 4.]);
        let cos = Owned::f32(&[1, 1, 1], &[0.0]);
        let sin = Owned::f32(&[1, 1, 1], &[1.0]);
        let mut out = Owned::zeros_f32(&[1, 1, 1, 4]);
        RotaryEmbeddingKernel {
            interleaved: false,
            num_heads: 0,
            rotary_embedding_dim: 2,
            contrib: false,
        }
        .execute(&[x.view(), cos.view(), sin.view()], &mut [out.view_mut()])
        .unwrap();
        // half=1: x1=d0=1, x2=d1=2; cos=0,sin=1 → real=-2, imag=1. Tail [3,4] unchanged.
        assert_eq!(out.to_f32(), vec![-2., 1., 3., 4.]);
    }

    fn run_nonstandard_head_size(head_size: usize, rotary_dim: usize) {
        let half = rotary_dim / 2;
        let x: Vec<f32> = (0..2 * head_size)
            .map(|i| i as f32 * 0.03125 - 1.5)
            .collect();
        let cos: Vec<f32> = (0..2 * half).map(|i| (i as f32 * 0.017).cos()).collect();
        let sin: Vec<f32> = (0..2 * half).map(|i| (i as f32 * 0.017).sin()).collect();
        let mut expected = x.clone();
        for s in 0..2 {
            for k in 0..half {
                let base = s * head_size;
                let cache = s * half + k;
                let x0 = x[base + k];
                let x1 = x[base + half + k];
                expected[base + k] = cos[cache] * x0 - sin[cache] * x1;
                expected[base + half + k] = sin[cache] * x0 + cos[cache] * x1;
            }
        }

        let input = Owned::f32(&[1, 1, 2, head_size], &x);
        let cos_cache = Owned::f32(&[1, 2, half], &cos);
        let sin_cache = Owned::f32(&[1, 2, half], &sin);
        let mut output = Owned::zeros_f32(&[1, 1, 2, head_size]);
        RotaryEmbeddingKernel {
            interleaved: false,
            num_heads: 0,
            rotary_embedding_dim: rotary_dim,
            contrib: false,
        }
        .execute(
            &[input.view(), cos_cache.view(), sin_cache.view()],
            &mut [output.view_mut()],
        )
        .unwrap();

        for (index, (actual, expected)) in output.to_f32().iter().zip(&expected).enumerate() {
            assert!(
                (actual - expected).abs() < 1e-6,
                "{index}: {actual} != {expected}"
            );
        }
    }

    #[test]
    fn rope_head_dim_48_matches_reference() {
        run_nonstandard_head_size(48, 48);
    }

    #[test]
    fn rope_head_dim_80_partial_rotary_matches_reference() {
        run_nonstandard_head_size(80, 32);
    }

    #[test]
    fn rope_zero_sized_input_returns_empty() {
        // Empty batch: no rows to rotate. Must not panic on batch-1 underflow.
        let x = Owned::f32(&[0, 1, 1, 4], &[]);
        let cos = Owned::f32(&[1, 1, 2], &[1., 1.]);
        let sin = Owned::f32(&[1, 1, 2], &[0., 0.]);
        let mut out = Owned::zeros_f32(&[0, 1, 1, 4]);
        RotaryEmbeddingKernel {
            interleaved: false,
            num_heads: 0,
            rotary_embedding_dim: 0,
            contrib: false,
        }
        .execute(&[x.view(), cos.view(), sin.view()], &mut [out.view_mut()])
        .unwrap();
        assert!(out.to_f32().is_empty());
    }

    #[test]
    fn rope_out_of_range_position_errors() {
        // 3D X [1,2,4], num_heads=2, position_ids gather rows [0, 5] but the
        // 2D cache only has 2 rows → clean error (not a panic) on the second row.
        let x = Owned::f32(&[1, 2, 4], &[1., 2., 3., 4., 5., 6., 7., 8.]);
        let cos = Owned::f32(&[2, 1], &[1.0, 0.0]);
        let sin = Owned::f32(&[2, 1], &[0.0, 1.0]);
        let pos = Owned::i64(&[1, 2], &[0, 5]);
        let mut out = Owned::zeros_f32(&[1, 2, 4]);
        let err = RotaryEmbeddingKernel {
            interleaved: false,
            num_heads: 2,
            rotary_embedding_dim: 0,
            contrib: false,
        }
        .execute(
            &[x.view(), cos.view(), sin.view(), pos.view()],
            &mut [out.view_mut()],
        );
        assert!(err.is_err(), "out-of-range position must return an error");
    }

    #[test]
    fn rope_i64_max_position_errors_without_overflow() {
        // The checked cache-row arithmetic must reject this before it can wrap
        // and be used as an in-bounds cache offset.
        let x = Owned::f32(&[1, 1, 4], &[1., 2., 3., 4.]);
        let cos = Owned::f32(&[1, 2], &[1.0, 1.0]);
        let sin = Owned::f32(&[1, 2], &[0.0, 0.0]);
        let pos = Owned::i64(&[1, 1], &[i64::MAX]);
        let mut out = Owned::zeros_f32(&[1, 1, 4]);
        let err = RotaryEmbeddingKernel {
            interleaved: false,
            num_heads: 2,
            rotary_embedding_dim: 0,
            contrib: false,
        }
        .execute(
            &[x.view(), cos.view(), sin.view(), pos.view()],
            &mut [out.view_mut()],
        );
        assert!(err.is_err(), "i64::MAX position must return an error");
    }

    #[test]
    fn rope_negative_position_errors() {
        let x = Owned::f32(&[1, 1, 4], &[1., 2., 3., 4.]);
        let cos = Owned::f32(&[1, 2], &[1.0, 1.0]);
        let sin = Owned::f32(&[1, 2], &[0.0, 0.0]);
        let pos = Owned::i64(&[1, 1], &[-1]);
        let mut out = Owned::zeros_f32(&[1, 1, 4]);
        let err = RotaryEmbeddingKernel {
            interleaved: false,
            num_heads: 2,
            rotary_embedding_dim: 0,
            contrib: false,
        }
        .execute(
            &[x.view(), cos.view(), sin.view(), pos.view()],
            &mut [out.view_mut()],
        );
        assert!(err.is_err(), "negative position must return an error");
    }

    #[test]
    fn rope_bad_position_ids_shape_errors() {
        // position_ids must have batch*seq = 2 elements; supplying 1 is invalid.
        let x = Owned::f32(&[1, 2, 4], &[1., 2., 3., 4., 5., 6., 7., 8.]);
        let cos = Owned::f32(&[2, 1], &[1.0, 0.0]);
        let sin = Owned::f32(&[2, 1], &[0.0, 1.0]);
        let pos = Owned::i64(&[1, 1], &[0]);
        let mut out = Owned::zeros_f32(&[1, 2, 4]);
        let err = RotaryEmbeddingKernel {
            interleaved: false,
            num_heads: 2,
            rotary_embedding_dim: 0,
            contrib: false,
        }
        .execute(
            &[x.view(), cos.view(), sin.view(), pos.view()],
            &mut [out.view_mut()],
        );
        assert!(err.is_err(), "malformed position_ids must return an error");
    }

    #[test]
    fn contrib_input_order_matches_standard() {
        // The com.microsoft op orders inputs (X, position_ids, cos, sin); the
        // standard op uses (X, cos, sin, position_ids). Both must produce the
        // same rotation. Use position_ids to gather from a 2D cache.
        let x = Owned::f32(&[1, 1, 2, 4], &[1., 2., 3., 4., 5., 6., 7., 8.]);
        // cache [max_pos=2, half=2].
        let cos = Owned::f32(&[2, 2], &[0.5, 0.8, 0.6, 0.9]);
        let sin = Owned::f32(&[2, 2], &[0.6, 0.5, 0.7, 0.4]);
        let pos = Owned::i64(&[1, 2], &[0, 1]);

        let mut out_std = Owned::zeros_f32(&[1, 1, 2, 4]);
        RotaryEmbeddingKernel {
            interleaved: false,
            num_heads: 0,
            rotary_embedding_dim: 0,
            contrib: false,
        }
        .execute(
            &[x.view(), cos.view(), sin.view(), pos.view()],
            &mut [out_std.view_mut()],
        )
        .unwrap();

        let mut out_contrib = Owned::zeros_f32(&[1, 1, 2, 4]);
        RotaryEmbeddingKernel {
            interleaved: false,
            num_heads: 0,
            rotary_embedding_dim: 0,
            contrib: true,
        }
        .execute(
            &[x.view(), pos.view(), cos.view(), sin.view()],
            &mut [out_contrib.view_mut()],
        )
        .unwrap();

        for (a, b) in out_std.to_f32().iter().zip(out_contrib.to_f32().iter()) {
            assert!((a - b).abs() < 1e-6, "contrib {b} != standard {a}");
        }
    }
}
