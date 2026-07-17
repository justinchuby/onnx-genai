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

use std::ffi::c_void;
use std::sync::Arc;

use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::Node;

use crate::runtime::{CudaRuntime, cuptr};

fn check_arity(
    name: &str,
    inputs: &[TensorView],
    outputs: &[TensorMut],
    min: usize,
    max: usize,
    expected_outputs: usize,
) -> Result<()> {
    if !(min..=max).contains(&inputs.len()) || outputs.len() != expected_outputs {
        return Err(EpError::KernelFailed(format!(
            "{name}: invalid input/output arity"
        )));
    }
    Ok(())
}
fn read_bytes(runtime: &CudaRuntime, view: &TensorView) -> Result<Vec<u8>> {
    if !view.is_contiguous() {
        return Err(EpError::KernelFailed(
            "RotaryEmbedding: non-contiguous input".into(),
        ));
    }
    let mut bytes = vec![0; view.dtype.storage_bytes(view.numel())];
    unsafe {
        runtime.dtoh(&mut bytes, cuptr(view.data_ptr::<u8>() as *const c_void))?;
    }
    Ok(bytes)
}
fn read_f32(runtime: &CudaRuntime, view: &TensorView) -> Result<Vec<f32>> {
    if view.dtype != onnx_runtime_ir::DataType::Float32 {
        return Err(EpError::KernelFailed("RotaryEmbedding: f32 only".into()));
    }
    Ok(read_bytes(runtime, view)?
        .chunks_exact(4)
        .map(|b| f32::from_ne_bytes(b.try_into().unwrap()))
        .collect())
}
fn read_i64(runtime: &CudaRuntime, view: &TensorView) -> Result<Vec<i64>> {
    if view.dtype != onnx_runtime_ir::DataType::Int64 {
        return Err(EpError::KernelFailed(
            "RotaryEmbedding: position_ids must be int64".into(),
        ));
    }
    Ok(read_bytes(runtime, view)?
        .chunks_exact(8)
        .map(|b| i64::from_ne_bytes(b.try_into().unwrap()))
        .collect())
}
fn write_f32(runtime: &CudaRuntime, output: &mut TensorMut, values: &[f32]) -> Result<()> {
    if output.dtype != onnx_runtime_ir::DataType::Float32
        || !output.is_contiguous()
        || output.numel() != values.len()
    {
        return Err(EpError::KernelFailed(
            "RotaryEmbedding: invalid f32 output".into(),
        ));
    }
    let bytes = unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
    };
    let ptr = cuptr(output.data_ptr_mut::<u8>() as *const c_void);
    unsafe {
        runtime.htod(bytes, ptr)?;
    }
    Ok(())
}

/// f32 RotaryEmbedding kernel carrying the resolved attributes.
pub struct RotaryEmbeddingKernel {
    runtime: Arc<CudaRuntime>,
    interleaved: bool,
    num_heads: usize,
    rotary_embedding_dim: usize,
}

/// Factory reading `interleaved` (0), `num_heads` (0), `rotary_embedding_dim` (0).
pub struct RotaryEmbeddingFactory {
    pub runtime: Arc<CudaRuntime>,
}

impl KernelFactory for RotaryEmbeddingFactory {
    fn create(&self, node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
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
            runtime: self.runtime.clone(),
            interleaved,
            num_heads,
            rotary_embedding_dim,
        }))
    }
}

impl Kernel for RotaryEmbeddingKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        check_arity("RotaryEmbedding", inputs, outputs, 3, 4, 1)?;
        // Inputs may have been uploaded asynchronously on the EP stream.
        self.runtime.synchronize()?;
        let x = read_f32(&self.runtime, &inputs[0])?;
        let cos_cache = read_f32(&self.runtime, &inputs[1])?;
        let sin_cache = read_f32(&self.runtime, &inputs[2])?;
        let position_ids = if inputs.len() == 4 {
            Some(read_i64(&self.runtime, &inputs[3])?)
        } else {
            None
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
            return write_f32(&self.runtime, &mut outputs[0], &[]);
        }

        // With `position_ids` present, validate its shape matches [batch, seq].
        if let Some(pos) = &position_ids {
            let pos_shape = inputs[3].shape;
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

        write_f32(&self.runtime, &mut outputs[0], &y)?;
        self.runtime.synchronize()
    }

    fn supports_strided_input(&self, _input_idx: usize) -> bool {
        true
    }
}
