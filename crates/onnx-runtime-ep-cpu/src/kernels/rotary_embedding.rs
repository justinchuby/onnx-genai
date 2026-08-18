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

use super::{check_arity, to_dense_i64};
use crate::dtype::{
    output_direct_write_eligible, slice_byte_range, to_dense_f32_widen, write_dense_f32_narrow,
};

/// Elements below which the rotation stays on one thread. RoPE is two multiplies
/// and an add per element - firmly memory bound - so the crossover is set by the
/// pool wake-up cost, not by arithmetic. A single llama3 decode step (32 heads x
/// 128 dims = 4096 elements) stays serial; a 128-token prefill does not.
const MIN_PARALLEL_ROTARY_ELEMENTS: usize = 64 * 1024;

/// Smallest output run, in f32 elements, worth handing to its own Rayon task.
///
/// RoPE moves 8 bytes per element (one read, one write) for three flops, so a
/// task's runtime is set by bandwidth: 16 Ki elements is 128 KiB of traffic,
/// roughly 2 us on one core here. Rayon's split-and-join costs ~0.5 us per
/// task, so this keeps scheduling under ~25% of the work even in the worst
/// case, while still leaving enough tasks to balance across the pool.
///
/// Chunking per `(b, h)` plane or per `(b, s)` row - the natural layout unit -
/// was measured at one task per 4096 elements, which made the parallel path
/// *slower* than serial at every thread count (t=1 162 us vs 126 us, t=16
/// 195 us vs 127 us on `rope_llama3_s128`). Tasks must aggregate whole units
/// until they clear this bar.
const MIN_ROTARY_TASK_ELEMENTS: usize = 16 * 1024;

/// How many layout units (`[B,H,S,D]` planes or `[B,S,H*D]` rows) one Rayon
/// task should own, given `total_units` of `unit_elems` each.
///
/// Aims for ~4 tasks per worker so a straggler costs a quarter of a worker's
/// share, then raises the chunk until each task clears
/// [`MIN_ROTARY_TASK_ELEMENTS`]. Returns `total_units` (a single task) when the
/// pool has one worker, so the serial shape is reached without paying for a
/// fan-out that cannot overlap with anything.
fn rotary_units_per_task(total_units: usize, unit_elems: usize, workers: usize) -> usize {
    if workers <= 1 || total_units <= 1 {
        return total_units.max(1);
    }
    let by_balance = total_units.div_ceil(workers.saturating_mul(4)).max(1);
    let by_size = MIN_ROTARY_TASK_ELEMENTS.div_ceil(unit_elems.max(1)).max(1);
    by_balance.max(by_size).min(total_units)
}

// ─── Rotation inner loops ───────────────────────────────────────────────────
//
// Both arms compute the same expression the ONNX reference does:
//
//     real = cos*x1 - sin*x2      imag = sin*x1 + cos*x2
//
// The AVX2 arms deliberately use separate `_mm256_mul_ps` + `_mm256_sub_ps` /
// `_mm256_add_ps` rather than FMA. FMA would drop the intermediate rounding of
// `cos*x1`, which makes the vector and scalar tails disagree in the last ulp on
// the same tensor - a parity failure that is a nuisance to chase and buys
// nothing here, because this kernel is bandwidth-bound (8 bytes moved per 3
// flops), not issue-bound. Vectorising is worth ~1.8x; fusing the multiply is
// not measurable.

/// `interleaved=0` (GPT-NeoX): rotate the two contiguous halves of the rotary
/// sub-vector against each other.
#[inline]
fn rotate_split_half(
    lo_in: &[f32],
    hi_in: &[f32],
    lo_out: &mut [f32],
    hi_out: &mut [f32],
    cos_row: &[f32],
    sin_row: &[f32],
) {
    let half = lo_out.len();
    debug_assert_eq!(hi_out.len(), half);
    debug_assert!(lo_in.len() >= half && hi_in.len() >= half);
    debug_assert!(cos_row.len() >= half && sin_row.len() >= half);

    let mut k = 0;
    #[cfg(target_arch = "x86_64")]
    if vector_rotation_available() {
        // SAFETY: guarded by the runtime AVX2+FMA probe; every slice is
        // asserted at least `half` long and the loop reads 8 lanes at a time
        // only while `k + 8 <= half`.
        k = unsafe { avx2::split_half(lo_in, hi_in, lo_out, hi_out, cos_row, sin_row, half) };
    }
    while k < half {
        let (cos, sin) = (cos_row[k], sin_row[k]);
        let (x1, x2) = (lo_in[k], hi_in[k]);
        lo_out[k] = cos * x1 - sin * x2;
        hi_out[k] = sin * x1 + cos * x2;
        k += 1;
    }
}

/// `interleaved=1` (GPT-J): rotate adjacent even/odd channel pairs.
#[inline]
fn rotate_interleaved(src: &[f32], dst: &mut [f32], cos_row: &[f32], sin_row: &[f32]) {
    let half = dst.len() / 2;
    debug_assert!(src.len() >= 2 * half);
    debug_assert!(cos_row.len() >= half && sin_row.len() >= half);

    let mut k = 0;
    #[cfg(target_arch = "x86_64")]
    if vector_rotation_available() {
        // SAFETY: guarded by the runtime AVX2+FMA probe; the vector loop reads
        // and writes `2 * 8` lanes only while `k + 8 <= half`, and `src`/`dst`
        // are asserted at least `2 * half` long.
        k = unsafe { avx2::interleaved(src, dst, cos_row, sin_row, half) };
    }
    while k < half {
        let (cos, sin) = (cos_row[k], sin_row[k]);
        let (x1, x2) = (src[2 * k], src[2 * k + 1]);
        dst[2 * k] = cos * x1 - sin * x2;
        dst[2 * k + 1] = sin * x1 + cos * x2;
        k += 1;
    }
}

/// Whether the AVX2 rotation arms are live on this machine.
#[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
#[inline]
pub(crate) fn vector_rotation_available() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::arch::is_x86_feature_detected!("avx2")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

#[cfg(target_arch = "x86_64")]
mod avx2 {
    use std::arch::x86_64::*;

    /// Eight split-half pairs per iteration. Returns the number of `k` values
    /// consumed, leaving the tail to the scalar loop.
    ///
    /// # Safety
    /// Caller must have verified AVX2 support and that every slice holds at
    /// least `half` elements.
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn split_half(
        lo_in: &[f32],
        hi_in: &[f32],
        lo_out: &mut [f32],
        hi_out: &mut [f32],
        cos_row: &[f32],
        sin_row: &[f32],
        half: usize,
    ) -> usize {
        unsafe {
            let mut k = 0;
            while k + 8 <= half {
                let c = _mm256_loadu_ps(cos_row.as_ptr().add(k));
                let s = _mm256_loadu_ps(sin_row.as_ptr().add(k));
                let x1 = _mm256_loadu_ps(lo_in.as_ptr().add(k));
                let x2 = _mm256_loadu_ps(hi_in.as_ptr().add(k));
                let real = _mm256_sub_ps(_mm256_mul_ps(c, x1), _mm256_mul_ps(s, x2));
                let imag = _mm256_add_ps(_mm256_mul_ps(s, x1), _mm256_mul_ps(c, x2));
                _mm256_storeu_ps(lo_out.as_mut_ptr().add(k), real);
                _mm256_storeu_ps(hi_out.as_mut_ptr().add(k), imag);
                k += 8;
            }
            k
        }
    }

    /// Eight interleaved pairs per iteration: de-interleave 16 lanes into
    /// even/odd registers, rotate, re-interleave.
    ///
    /// # Safety
    /// Caller must have verified AVX2 support and that `src`/`dst` hold at
    /// least `2 * half` elements and `cos_row`/`sin_row` at least `half`.
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn interleaved(
        src: &[f32],
        dst: &mut [f32],
        cos_row: &[f32],
        sin_row: &[f32],
        half: usize,
    ) -> usize {
        unsafe {
            // `_mm256_shuffle_ps` works within each 128-bit lane, so gathering
            // the even (or odd) channels of 16 consecutive floats leaves them
            // in the order [0,1,4,5,2,3,6,7]. That permutation is its own
            // inverse, so the same index vector both straightens the operands
            // and pre-scrambles the results for `unpack`.
            let lanes = _mm256_setr_epi32(0, 1, 4, 5, 2, 3, 6, 7);
            let mut k = 0;
            while k + 8 <= half {
                let a = _mm256_loadu_ps(src.as_ptr().add(2 * k));
                let b = _mm256_loadu_ps(src.as_ptr().add(2 * k + 8));
                let x1 = _mm256_permutevar8x32_ps(_mm256_shuffle_ps::<0b1000_1000>(a, b), lanes);
                let x2 = _mm256_permutevar8x32_ps(_mm256_shuffle_ps::<0b1101_1101>(a, b), lanes);
                let c = _mm256_loadu_ps(cos_row.as_ptr().add(k));
                let s = _mm256_loadu_ps(sin_row.as_ptr().add(k));
                let real = _mm256_sub_ps(_mm256_mul_ps(c, x1), _mm256_mul_ps(s, x2));
                let imag = _mm256_add_ps(_mm256_mul_ps(s, x1), _mm256_mul_ps(c, x2));
                let rp = _mm256_permutevar8x32_ps(real, lanes);
                let ip = _mm256_permutevar8x32_ps(imag, lanes);
                _mm256_storeu_ps(dst.as_mut_ptr().add(2 * k), _mm256_unpacklo_ps(rp, ip));
                _mm256_storeu_ps(dst.as_mut_ptr().add(2 * k + 8), _mm256_unpackhi_ps(rp, ip));
                k += 8;
            }
            k
        }
    }
}

/// Floating-point RotaryEmbedding kernel carrying the resolved attributes.
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
        let x = to_dense_f32_widen("RotaryEmbedding", &inputs[0])?;
        let cos_cache = to_dense_f32_widen("RotaryEmbedding", &inputs[cos_i])?;
        let sin_cache = to_dense_f32_widen("RotaryEmbedding", &inputs[sin_i])?;
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
            return write_dense_f32_narrow("RotaryEmbedding", &mut outputs[0], &[]);
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

        // The cos/sin row for every (b, s) up front. Resolving it inside the
        // rotation loop meant the fallible `position_ids` lookup could not be
        // hoisted out of a parallel region, and it was re-done once per (b, s)
        // per head anyway.
        let mut cache_rows = vec![0usize; batch * seq];
        for b in 0..batch {
            for s in 0..seq {
                cache_rows[b * seq + s] = cache_row(b, s)?;
            }
        }

        let interleaved = self.interleaved;
        // One head's rotation, given its `head_size`-long input and output
        // runs and the (cos, sin) row for its position. Layout-independent:
        // both callers below hand it contiguous per-head slices.
        let rotate_head = |src: &[f32], dst: &mut [f32], crow: usize| {
            let cos_row = &cos_cache[crow..crow + half];
            let sin_row = &sin_cache[crow..crow + half];
            if interleaved {
                rotate_interleaved(&src[..rotary_dim], &mut dst[..rotary_dim], cos_row, sin_row);
            } else {
                // Split-half: the two operand runs and the two result runs are
                // each contiguous, so this is four linear streams with no
                // index arithmetic in the loop body.
                let (lo_out, hi_out) = dst[..rotary_dim].split_at_mut(half);
                rotate_split_half(
                    &src[..half],
                    &src[half..rotary_dim],
                    lo_out,
                    hi_out,
                    cos_row,
                    sin_row,
                );
            }
            // Pass-through channels beyond the rotary sub-vector.
            dst[rotary_dim..head_size].copy_from_slice(&src[rotary_dim..head_size]);
        };

        // Every input element is read exactly once, into the output element at
        // the same flat position, so a contiguous f32 output that does not
        // alias any input we read can be written straight through. That saves
        // zeroing a full-tensor scratch buffer *and* copying it out - together
        // the dominant cost of this kernel, which is otherwise two multiplies
        // and an add per element.
        let read_ranges: Vec<_> = [&*x, &*cos_cache, &*sin_cache]
            .into_iter()
            .map(slice_byte_range)
            .collect();
        let direct = output_direct_write_eligible(&mut outputs[0], x.len(), &read_ranges);
        let mut owned;
        let y: &mut [f32] = if direct {
            // SAFETY: `output_direct_write_eligible` proved the buffer is a
            // host-accessible contiguous Float32 tensor of exactly `x.len()`
            // elements whose byte range is disjoint from every input read here.
            unsafe { std::slice::from_raw_parts_mut(outputs[0].data_ptr_mut::<f32>(), x.len()) }
        } else {
            owned = vec![0.0f32; x.len()];
            &mut owned
        };
        // Chunk the output so each task owns a disjoint, contiguous run in the
        // tensor's *native* layout - `[B, H, S, D]` splits naturally per
        // (b, h) plane, `[B, S, H·D]` per (b, s) row. That removes the
        // per-element layout branch the old flat-index closure carried, and
        // lets the outer loop fan out across the shared Rayon pool. ORT
        // parallelizes the same transform.
        let workers = rayon::current_num_threads().max(1);
        let parallel = workers > 1 && y.len() >= MIN_PARALLEL_ROTARY_ELEMENTS;
        if is_4d {
            // [B, H, S, D]: plane (b, h) is `seq * head_size` contiguous.
            let plane = seq * head_size;
            let fill = |bh: usize, out: &mut [f32]| {
                let b = bh / heads;
                let base = bh * plane;
                for s in 0..seq {
                    let off = s * head_size;
                    rotate_head(
                        &x[base + off..base + off + head_size],
                        &mut out[off..off + head_size],
                        cache_rows[b * seq + s],
                    );
                }
            };
            let per_task = rotary_units_per_task(batch * heads, plane, workers);
            if parallel && per_task < batch * heads {
                use rayon::prelude::*;
                y.par_chunks_mut(plane * per_task)
                    .enumerate()
                    .for_each(|(task, slab)| {
                        for (i, out) in slab.chunks_mut(plane).enumerate() {
                            fill(task * per_task + i, out);
                        }
                    });
            } else {
                for (bh, out) in y.chunks_mut(plane).enumerate() {
                    fill(bh, out);
                }
            }
        } else {
            // [B, S, H·D]: row (b, s) is `heads * head_size` contiguous.
            let row_len = heads * head_size;
            let fill = |bs: usize, out: &mut [f32]| {
                let base = bs * row_len;
                let crow = cache_rows[bs];
                for h in 0..heads {
                    let off = h * head_size;
                    rotate_head(
                        &x[base + off..base + off + head_size],
                        &mut out[off..off + head_size],
                        crow,
                    );
                }
            };
            let per_task = rotary_units_per_task(batch * seq, row_len, workers);
            if parallel && per_task < batch * seq {
                use rayon::prelude::*;
                y.par_chunks_mut(row_len * per_task)
                    .enumerate()
                    .for_each(|(task, slab)| {
                        for (i, out) in slab.chunks_mut(row_len).enumerate() {
                            fill(task * per_task + i, out);
                        }
                    });
            } else {
                for (bs, out) in y.chunks_mut(row_len).enumerate() {
                    fill(bs, out);
                }
            }
        }

        if direct {
            Ok(())
        } else {
            write_dense_f32_narrow("RotaryEmbedding", &mut outputs[0], y)
        }
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
    fn rope_f16_matches_f32_reference_after_narrowing() {
        let x_values = [1.25, -0.75, 2.5, -3.0];
        let cos_values = [0.5, 0.875];
        let sin_values = [0.75, -0.25];
        let x_f32 = Owned::f32(&[1, 1, 1, 4], &x_values);
        let cos_f32 = Owned::f32(&[1, 1, 2], &cos_values);
        let sin_f32 = Owned::f32(&[1, 1, 2], &sin_values);
        let mut out_f32 = Owned::zeros_f32(&[1, 1, 1, 4]);
        let kernel = RotaryEmbeddingKernel {
            interleaved: false,
            num_heads: 0,
            rotary_embedding_dim: 0,
            contrib: false,
        };
        kernel
            .execute(
                &[x_f32.view(), cos_f32.view(), sin_f32.view()],
                &mut [out_f32.view_mut()],
            )
            .unwrap();

        let x_f16 = Owned::f16(&[1, 1, 1, 4], &x_values);
        let cos_f16 = Owned::f16(&[1, 1, 2], &cos_values);
        let sin_f16 = Owned::f16(&[1, 1, 2], &sin_values);
        let mut out_f16 = Owned::f16(&[1, 1, 1, 4], &[0.0; 4]);
        kernel
            .execute(
                &[x_f16.view(), cos_f16.view(), sin_f16.view()],
                &mut [out_f16.view_mut()],
            )
            .unwrap();

        let expected = Owned::f16(&[1, 1, 1, 4], &out_f32.to_f32()).to_f16_as_f32();
        assert_eq!(out_f16.to_f16_as_f32(), expected);
    }

    #[test]
    fn rope_bfloat16_decode_and_prefill_match_widened_reference() {
        let kernel = RotaryEmbeddingKernel {
            interleaved: false,
            num_heads: 0,
            rotary_embedding_dim: 0,
            contrib: false,
        };
        for (batch, sequence) in [(1, 1), (2, 4)] {
            let heads = 2;
            let head_size = 8;
            let half = head_size / 2;
            let input_values: Vec<f32> = (0..batch * heads * sequence * head_size)
                .map(|index| ((index * 11 % 37) as f32 - 18.0) * 0.125)
                .collect();
            let angle_values: Vec<f32> = (0..batch * sequence * half)
                .map(|index| index as f32 * 0.03125)
                .collect();
            let cosine_values: Vec<f32> = angle_values.iter().map(|angle| angle.cos()).collect();
            let sine_values: Vec<f32> = angle_values.iter().map(|angle| angle.sin()).collect();
            let input = Owned::bf16(&[batch, heads, sequence, head_size], &input_values);
            let cosine = Owned::bf16(&[batch, sequence, half], &cosine_values);
            let sine = Owned::bf16(&[batch, sequence, half], &sine_values);
            let mut output = Owned::zeros(
                onnx_runtime_ir::DataType::BFloat16,
                &[batch, heads, sequence, head_size],
            );
            kernel
                .execute(
                    &[input.view(), cosine.view(), sine.view()],
                    &mut [output.view_mut()],
                )
                .unwrap();

            let widened_input = input.to_bf16_as_f32();
            let widened_cosine = cosine.to_bf16_as_f32();
            let widened_sine = sine.to_bf16_as_f32();
            let mut expected = vec![0.0; widened_input.len()];
            for batch_index in 0..batch {
                for head_index in 0..heads {
                    for sequence_index in 0..sequence {
                        let input_base = ((batch_index * heads + head_index) * sequence
                            + sequence_index)
                            * head_size;
                        let cache_base = (batch_index * sequence + sequence_index) * half;
                        for channel in 0..half {
                            let first = widened_input[input_base + channel];
                            let second = widened_input[input_base + half + channel];
                            let cosine = widened_cosine[cache_base + channel];
                            let sine = widened_sine[cache_base + channel];
                            expected[input_base + channel] = cosine * first - sine * second;
                            expected[input_base + half + channel] = sine * first + cosine * second;
                        }
                    }
                }
            }
            for (index, (actual, expected)) in output
                .to_bf16_as_f32()
                .into_iter()
                .zip(expected)
                .enumerate()
            {
                let tolerance = 2e-3 + 1e-2 * expected.abs();
                assert!(
                    (actual - expected).abs() <= tolerance,
                    "batch {batch}, sequence {sequence}, element {index}: {actual} != {expected}"
                );
            }
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

#[cfg(test)]
mod parallel_tests {
    use super::*;
    use crate::kernels::testutil::Owned;

    /// Independent scalar oracle written in the original flat-index style, so
    /// it cross-checks the layout-specialised chunked loops rather than
    /// restating them.
    #[allow(clippy::too_many_arguments)]
    fn reference(
        x: &[f32],
        cos: &[f32],
        sin: &[f32],
        batch: usize,
        seq: usize,
        heads: usize,
        head_size: usize,
        rotary_dim: usize,
        interleaved: bool,
        is_4d: bool,
    ) -> Vec<f32> {
        let half = rotary_dim / 2;
        let idx = |b: usize, h: usize, s: usize, d: usize| -> usize {
            if is_4d {
                ((b * heads + h) * seq + s) * head_size + d
            } else {
                (b * seq + s) * (heads * head_size) + h * head_size + d
            }
        };
        let mut y = vec![0.0f32; x.len()];
        for b in 0..batch {
            for s in 0..seq {
                let crow = (b * seq + s) * half;
                for h in 0..heads {
                    for k in 0..half {
                        let (c, sn) = (cos[crow + k], sin[crow + k]);
                        let (d1, d2) = if interleaved {
                            (2 * k, 2 * k + 1)
                        } else {
                            (k, k + half)
                        };
                        let x1 = x[idx(b, h, s, d1)];
                        let x2 = x[idx(b, h, s, d2)];
                        y[idx(b, h, s, d1)] = c * x1 - sn * x2;
                        y[idx(b, h, s, d2)] = sn * x1 + c * x2;
                    }
                    for d in rotary_dim..head_size {
                        y[idx(b, h, s, d)] = x[idx(b, h, s, d)];
                    }
                }
            }
        }
        y
    }

    fn synthetic(n: usize, seed: u32) -> Vec<f32> {
        let mut state = seed.wrapping_mul(2654435761).wrapping_add(1);
        (0..n)
            .map(|_| {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                ((state >> 8) as f32 / (1u32 << 24) as f32) * 4.0 - 2.0
            })
            .collect()
    }

    /// Sweep both layouts, both rotation modes, full and partial rotary dims,
    /// and sizes that straddle `MIN_PARALLEL_ROTARY_ELEMENTS` in both
    /// directions. The fan-out chunks disjoint runs, so it must be exactly
    /// bit-identical to the scalar oracle - not merely close.
    #[test]
    fn every_layout_and_size_matches_the_scalar_reference() {
        for &(batch, seq, heads, head_size, rotary_dim) in &[
            (1usize, 1usize, 4usize, 8usize, 8usize), // tiny, serial
            (1, 1, 32, 128, 128),                     // llama3 decode step, serial
            (1, 128, 32, 128, 128),                   // prefill, parallel
            (2, 40, 6, 16, 8),                        // partial rotary, parallel-ish
            (3, 7, 5, 12, 12),                        // nothing divides evenly
        ] {
            let half = rotary_dim / 2;
            let n = batch * seq * heads * head_size;
            let x = synthetic(n, (batch * 7 + seq) as u32);
            let cos = synthetic(batch * seq * half, 11);
            let sin = synthetic(batch * seq * half, 13);

            for interleaved in [false, true] {
                for is_4d in [false, true] {
                    let shape: Vec<usize> = if is_4d {
                        vec![batch, heads, seq, head_size]
                    } else {
                        vec![batch, seq, heads * head_size]
                    };
                    let xt = Owned::f32(&shape, &x);
                    let pos: Vec<i64> = (0..(batch * seq) as i64).collect();
                    let post = Owned::i64(&[batch, seq], &pos);
                    let cost = Owned::f32(&[batch * seq, half], &cos);
                    let sint = Owned::f32(&[batch * seq, half], &sin);
                    let mut out = Owned::zeros_f32(&shape);

                    let kernel = RotaryEmbeddingKernel {
                        interleaved,
                        num_heads: if is_4d { 0 } else { heads },
                        rotary_embedding_dim: if rotary_dim == head_size {
                            0
                        } else {
                            rotary_dim
                        },
                        contrib: true,
                    };
                    kernel
                        .execute(
                            &[xt.view(), post.view(), cost.view(), sint.view()],
                            &mut [out.view_mut()],
                        )
                        .unwrap();

                    let expect = reference(
                        &x,
                        &cos,
                        &sin,
                        batch,
                        seq,
                        heads,
                        head_size,
                        rotary_dim,
                        interleaved,
                        is_4d,
                    );
                    assert_eq!(
                        out.to_f32(),
                        expect,
                        "b={batch} s={seq} h={heads} d={head_size} rd={rotary_dim} \
                         interleaved={interleaved} is_4d={is_4d}"
                    );
                }
            }
        }
    }

    /// RoPE's output binding may alias its `X` input. Every rotated pair reads
    /// both halves before writing either, so an aliased output would corrupt
    /// the second half; the direct-write guard must decline here.
    #[test]
    fn an_output_aliasing_its_input_matches_the_disjoint_result() {
        use onnx_runtime_ep_api::{DevicePtr, DevicePtrMut};
        use onnx_runtime_ir::{DataType, DeviceId, compute_contiguous_strides};

        let (batch, seq, heads, head_size) = (1usize, 64usize, 8usize, 64usize);
        let n = batch * seq * heads * head_size;
        let x = synthetic(n, 5);
        let pos: Vec<i64> = (0..(batch * seq) as i64).collect();
        let shape = [batch, seq, heads * head_size];
        let f32c = DataType::Float32;
        let cpu = DeviceId::cpu();
        let strides = compute_contiguous_strides(&shape);

        // Full rotation is elementwise alias-safe on its own; partial rotary
        // adds a pass-through `copy_from_slice`, which is only sound when the
        // guard has declined. Both are covered.
        for rotary_dim in [head_size, head_size / 2] {
            let half = rotary_dim / 2;
            let cos = synthetic(batch * seq * half, 6);
            let sin = synthetic(batch * seq * half, 7);
            let post = Owned::i64(&[batch, seq], &pos);
            let cost = Owned::f32(&[batch * seq, half], &cos);
            let sint = Owned::f32(&[batch * seq, half], &sin);
            let kernel = || RotaryEmbeddingKernel {
                interleaved: false,
                num_heads: heads,
                rotary_embedding_dim: if rotary_dim == head_size {
                    0
                } else {
                    rotary_dim
                },
                contrib: true,
            };

            let disjoint = {
                let xt = Owned::f32(&shape, &x);
                let mut out = Owned::zeros_f32(&shape);
                kernel()
                    .execute(
                        &[xt.view(), post.view(), cost.view(), sint.view()],
                        &mut [out.view_mut()],
                    )
                    .unwrap();
                out.to_f32()
            };

            let mut shared = x.clone();
            let in_ptr = shared.as_ptr() as *const std::ffi::c_void;
            let out_ptr = shared.as_mut_ptr() as *mut std::ffi::c_void;
            let xv = TensorView::new(DevicePtr(in_ptr), f32c, &shape, &strides, cpu);
            let outv = TensorMut::new(DevicePtrMut(out_ptr), f32c, &shape, &strides, cpu);
            kernel()
                .execute(&[xv, post.view(), cost.view(), sint.view()], &mut [outv])
                .unwrap();

            assert_eq!(shared, disjoint, "rotary_dim={rotary_dim}");
        }
    }
}

#[cfg(test)]
mod vector_arm_tests {
    use super::*;

    /// The AVX2 arms must agree with the scalar arms **bit for bit**, not just
    /// to a tolerance: a run that splits a tensor differently across tasks (a
    /// different thread count, a different chunk size) would otherwise produce
    /// a different answer for the same graph, and the A/B parity check would
    /// flip on scheduling alone. This is why the vector arms avoid FMA.
    fn scalar_split_half(
        lo_in: &[f32],
        hi_in: &[f32],
        cos_row: &[f32],
        sin_row: &[f32],
        half: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let mut lo = vec![0.0f32; half];
        let mut hi = vec![0.0f32; half];
        for k in 0..half {
            let (c, s) = (cos_row[k], sin_row[k]);
            let (x1, x2) = (lo_in[k], hi_in[k]);
            lo[k] = c * x1 - s * x2;
            hi[k] = s * x1 + c * x2;
        }
        (lo, hi)
    }

    fn scalar_interleaved(src: &[f32], cos_row: &[f32], sin_row: &[f32], half: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; 2 * half];
        for k in 0..half {
            let (c, s) = (cos_row[k], sin_row[k]);
            let (x1, x2) = (src[2 * k], src[2 * k + 1]);
            out[2 * k] = c * x1 - s * x2;
            out[2 * k + 1] = s * x1 + c * x2;
        }
        out
    }

    fn ramp(n: usize, seed: f32) -> Vec<f32> {
        (0..n)
            .map(|i| ((i as f32) * 0.017 + seed).sin() * 3.0 - 0.5)
            .collect()
    }

    #[test]
    fn avx2_split_half_is_bit_identical_to_the_scalar_loop() {
        // Sweep `half` across, below, and either side of the 8-lane vector
        // width so the tail loop is exercised at every residue.
        for half in [1usize, 3, 7, 8, 9, 15, 16, 31, 32, 63, 64, 65, 127, 128] {
            let lo_in = ramp(half, 0.3);
            let hi_in = ramp(half, 1.7);
            let cos_row = ramp(half, 2.9);
            let sin_row = ramp(half, 4.1);
            let (want_lo, want_hi) = scalar_split_half(&lo_in, &hi_in, &cos_row, &sin_row, half);
            let mut got_lo = vec![0.0f32; half];
            let mut got_hi = vec![0.0f32; half];
            rotate_split_half(&lo_in, &hi_in, &mut got_lo, &mut got_hi, &cos_row, &sin_row);
            assert_eq!(
                got_lo.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                want_lo.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "split-half real part diverged at half={half}"
            );
            assert_eq!(
                got_hi.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                want_hi.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "split-half imag part diverged at half={half}"
            );
        }
    }

    #[test]
    fn avx2_interleaved_is_bit_identical_to_the_scalar_loop() {
        // The interleaved arm de-interleaves with a lane permutation that is
        // its own inverse; a wrong index vector still produces plausible
        // magnitudes, so this compares every element rather than a norm.
        for half in [1usize, 3, 7, 8, 9, 15, 16, 31, 32, 63, 64, 65] {
            let src = ramp(2 * half, 0.11);
            let cos_row = ramp(half, 5.3);
            let sin_row = ramp(half, 6.7);
            let want = scalar_interleaved(&src, &cos_row, &sin_row, half);
            let mut got = vec![0.0f32; 2 * half];
            rotate_interleaved(&src, &mut got, &cos_row, &sin_row);
            assert_eq!(
                got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                want.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "interleaved rotation diverged at half={half} (vector arm live: {})",
                vector_rotation_available()
            );
        }
    }

    #[test]
    fn task_chunking_covers_every_unit_exactly_once() {
        // `rotary_units_per_task` feeds `par_chunks_mut`, so an off-by-one
        // silently drops or double-writes a row. Check the derived tiling is a
        // partition for a wide range of shapes and pool sizes.
        for total in [1usize, 2, 3, 5, 8, 31, 32, 128, 512, 4096] {
            for unit in [64usize, 512, 4096, 32768] {
                for workers in [1usize, 2, 8, 16, 32] {
                    let per = rotary_units_per_task(total, unit, workers);
                    assert!(per >= 1 && per <= total.max(1), "per={per} total={total}");
                    let tasks = total.div_ceil(per);
                    assert_eq!(
                        (0..tasks)
                            .map(|t| (total - t * per).min(per))
                            .sum::<usize>(),
                        total,
                        "tiling total={total} unit={unit} workers={workers} per={per}"
                    );
                    if workers == 1 {
                        assert_eq!(per, total.max(1), "one worker must stay serial");
                    }
                }
            }
        }
    }

    #[test]
    fn a_single_worker_pool_never_splits_the_rotation() {
        // Measured: fanning 128 tasks out over a one-worker pool cost 36us of
        // pure Rayon bookkeeping on `rope_llama3_s128` (162us vs 126us). The
        // serial shape must be reachable.
        assert_eq!(rotary_units_per_task(128, 4096, 1), 128);
        assert_eq!(rotary_units_per_task(4096, 64, 1), 4096);
    }

    #[test]
    fn tasks_clear_the_minimum_work_bar_when_the_pool_is_wide() {
        // 128 rows of 4096 elements over 16 workers must not become 128 tasks.
        let per = rotary_units_per_task(128, 4096, 16);
        assert!(
            per * 4096 >= MIN_ROTARY_TASK_ELEMENTS,
            "task of {} elements is below the {MIN_ROTARY_TASK_ELEMENTS}-element bar",
            per * 4096
        );
        assert!(128 / per >= 16, "only {} tasks for 16 workers", 128 / per);
        // Tiny units must be aggregated hard rather than shipped one per task.
        assert!(rotary_units_per_task(65536, 8, 16) * 8 >= MIN_ROTARY_TASK_ELEMENTS);
    }
}
