//! Shape-collapsing, blocked and parallel `Transpose`.
//!
//! # What the naive kernel costs
//!
//! A permutation is defined element-wise, so the obvious implementation is an
//! odometer over the output index plus a rank-length dot product to find the
//! matching input element. That is what this kernel used to do, and it costs a
//! carry chain, `rank` multiply-adds and a one-element `copy_from_slice` for
//! **every element**, no matter how much of the tensor is actually contiguous.
//! Measured against ONNX Runtime on attention-shaped tensors it was 13.8x to
//! 120.8x slower.
//!
//! Almost none of that work is inherent. A permutation on a contiguous tensor is
//! a *block* move: whichever trailing axes the permutation leaves in place form
//! runs of consecutive bytes that move together, and adjacent axes that stay
//! adjacent can be merged into one longer axis. The element loop only has to run
//! for the axes that genuinely interleave.
//!
//! # The four paths, in the order they are tried
//!
//! 1. **Identity** — after collapsing, the permutation is the identity, so the
//!    whole tensor is one `memcpy`. `perm=[0,1,2,3]` written explicitly, and any
//!    permutation that only moves size-1 axes, land here.
//! 2. **Rank-1 after collapsing** — same thing: one `memcpy`.
//! 3. **Block move** — the permutation fixes a non-empty suffix of axes, so each
//!    output position takes a run of `inner` contiguous elements. This is the
//!    attention case: `(B,S,H,D) -> (B,H,S,D)` with `perm=[0,2,1,3]` leaves `D`
//!    innermost, so it copies `D` elements at a time instead of one. The
//!    remaining outer loop is an odometer over the *collapsed* outer axes only.
//! 4. **Blocked 2-D scatter** — the last axis moves, so elements genuinely
//!    interleave. Collapsing reduces almost every real permutation to a 2-D
//!    transpose (or a batch of them), which is then done in cache-sized tiles so
//!    each tile's source and destination both stay resident, instead of striding
//!    the full row pitch on every element.
//!
//! All four write **straight into the output tensor** when it is dense host
//! memory that does not alias the input. The old code allocated a zeroed
//! `Vec<u8>` the size of the output, filled it, and then copied it into the
//! output — three passes over the data (zero, fill, copy) where one suffices,
//! plus an allocation whose size scales with the tensor.
//!
//! # Why collapsing is safe
//!
//! Two output axes `i` and `i+1` can be merged exactly when their input axes are
//! also adjacent and in the same order (`perm[i+1] == perm[i] + 1`), because
//! then the elements they address are consecutive in *both* tensors, so the pair
//! addresses one contiguous run of length `shape[i] * shape[i+1]`. Size-1 axes
//! address one element regardless of where they sit, so they are dropped first.
//! Collapsing therefore preserves the exact element mapping; it only removes
//! loop levels. `collapse_matches_the_element_walk` checks that against the
//! original odometer for every permutation of ranks 1-4.

use onnx_runtime_ep_api::{
    EpError, Kernel, KernelFactory, Result, TensorMut, TensorView, ViewOutput,
};
use onnx_runtime_ir::{Node, compute_contiguous_strides};

use super::{check_arity, elem_size, to_dense_bytes, write_dense_bytes};
use crate::strided::{next_index, numel};

/// Tile edge, in elements, for the blocked 2-D scatter.
///
/// A 64x64 tile of 4-byte elements is 16 KiB on each side, so a source tile, a
/// destination tile and their TLB entries all sit in a 32 KiB L1 without
/// evicting each other. The point of the tile is that a column walk of the
/// source touches 64 rows and then reuses all of them, instead of touching 64
/// rows once each and returning to the first after the row pitch has evicted it.
const TILE: usize = 64;

/// Minimum output bytes before the parallel fan-out is worth its latency.
///
/// A `rayon` fan-out costs a few microseconds of scheduling; below roughly this
/// size the copy finishes in less than that, so splitting it makes the operation
/// slower. Chosen at 256 KiB because that is also where the copy stops fitting
/// in a single core's L2 and starts being bandwidth-bound, which is exactly when
/// more cores start helping.
const MIN_PARALLEL_BYTES: usize = 256 * 1024;

/// Dtype-generic Transpose kernel carrying the resolved `perm`.
pub struct TransposeKernel {
    /// Axis permutation; `None` means reverse all axes.
    perm: Option<Vec<usize>>,
}

/// Factory reading the `perm` attribute from the node.
pub struct TransposeFactory;

impl KernelFactory for TransposeFactory {
    fn create(&self, node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let perm = node
            .attr("perm")
            .and_then(|a| a.as_ints())
            .map(|ints| ints.iter().map(|&v| v as usize).collect::<Vec<_>>());
        Ok(Box::new(TransposeKernel { perm }))
    }
}

/// A permutation reduced to the axes that actually interleave.
///
/// `shape` and `perm` describe the same element mapping as the originals, with
/// size-1 axes dropped and mergeable neighbours fused.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Collapsed {
    /// Input shape after collapsing.
    shape: Vec<usize>,
    /// Permutation over the collapsed axes: output axis `i` is input axis
    /// `perm[i]`.
    perm: Vec<usize>,
}

/// Drop size-1 axes and fuse adjacent axes that stay adjacent.
///
/// Returns a rank-0 result for an empty or single-element tensor, which the
/// callers treat as "one contiguous block".
fn collapse(in_shape: &[usize], perm: &[usize]) -> Collapsed {
    // Output-order view of the walk: `out_axes[i]` is the input axis that
    // supplies output axis `i`, with its extent.
    let mut out_axes: Vec<(usize, usize)> = perm
        .iter()
        .map(|&p| (p, in_shape[p]))
        .filter(|&(_, dim)| dim != 1)
        .collect();

    // Renumber the surviving input axes densely, preserving their relative
    // order, so "adjacent in the input" stays a `+1` test after size-1 axes have
    // been removed. Without this, dropping axis 1 from `perm=[0,2]` would leave
    // `2` looking non-adjacent to `0` when in fact nothing separates them any
    // more.
    let mut surviving: Vec<usize> = out_axes.iter().map(|&(axis, _)| axis).collect();
    surviving.sort_unstable();
    for (axis, _) in out_axes.iter_mut() {
        *axis = surviving.binary_search(axis).expect("axis is in the set");
    }

    // Fuse: output axes `i`, `i+1` whose input axes are consecutive address one
    // contiguous run in both tensors.
    let mut fused: Vec<(usize, usize)> = Vec::with_capacity(out_axes.len());
    for (axis, dim) in out_axes {
        match fused.last_mut() {
            Some((prev_axis, prev_dim)) if *prev_axis + 1 == axis => {
                *prev_dim *= dim;
                // The fused pair keeps the *lower* input axis as its identity so
                // later neighbours can fuse onto it in turn; its extent now
                // covers both, so a following axis is adjacent iff it equals
                // `axis + 1`.
                *prev_axis = axis;
            }
            _ => fused.push((axis, dim)),
        }
    }

    // Renumber again: fusion removed axes, so the identities are sparse.
    let mut kept: Vec<usize> = fused.iter().map(|&(axis, _)| axis).collect();
    kept.sort_unstable();
    let rank = kept.len();
    let mut shape = vec![0usize; rank];
    let mut collapsed_perm = Vec::with_capacity(rank);
    for (axis, dim) in fused {
        let dense = kept.binary_search(&axis).expect("axis is in the set");
        shape[dense] = dim;
        collapsed_perm.push(dense);
    }
    Collapsed {
        shape,
        perm: collapsed_perm,
    }
}

impl Kernel for TransposeKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        check_arity("Transpose", inputs, outputs, 1, 1, 1)?;
        let in_shape = inputs[0].shape.to_vec();
        let rank = in_shape.len();
        let perm = match &self.perm {
            Some(p) => {
                if p.len() != rank {
                    return Err(EpError::KernelFailed(format!(
                        "Transpose: perm rank {} != input rank {rank}",
                        p.len()
                    )));
                }
                p.clone()
            }
            None => (0..rank).rev().collect(),
        };
        if perm.iter().any(|&axis| axis >= rank) || {
            let mut sorted = perm.clone();
            sorted.sort_unstable();
            sorted != (0..rank).collect::<Vec<_>>()
        } {
            return Err(EpError::KernelFailed(format!(
                "Transpose: perm {perm:?} is not a permutation of 0..{rank}"
            )));
        }

        if outputs[0].dtype != inputs[0].dtype {
            return Err(EpError::KernelFailed(format!(
                "Transpose: output dtype {:?} must match input dtype {:?}",
                outputs[0].dtype, inputs[0].dtype
            )));
        }
        let esize = elem_size(inputs[0].dtype)?;
        let out_shape: Vec<usize> = perm.iter().map(|&p| in_shape[p]).collect();
        let total = numel(&out_shape);
        if total == 0 {
            // Nothing to move, but the output view still has to be valid: an
            // empty tensor with a bad descriptor is a bug the caller wants to
            // hear about here rather than at the next consumer.
            outputs[0].validate()?;
            return Ok(());
        }

        // Fast path: read the input in place and write the output in place.
        // `to_dense_bytes` on a contiguous input would copy the whole tensor
        // just to hand back what is already there.
        if inputs[0].is_contiguous()
            && inputs[0].device.is_host_accessible()
            && dense_disjoint_output(&inputs[0], &mut outputs[0], total * esize)
        {
            let bytes = total * esize;
            // SAFETY: `inputs[0]` is a validated, contiguous, host-accessible
            // view of exactly `bytes` readable bytes from its origin (ep-api
            // safety invariant #1), and `dense_disjoint_output` established that
            // `outputs[0]` is a contiguous host tensor of the same dtype and
            // element count whose byte range does not overlap the input's. So
            // the two slices are valid, non-overlapping, and `u8` has no invalid
            // bit patterns.
            let (src, dst) = unsafe {
                (
                    std::slice::from_raw_parts(inputs[0].data_ptr::<u8>(), bytes),
                    std::slice::from_raw_parts_mut(outputs[0].data_ptr_mut::<u8>(), bytes),
                )
            };
            permute_bytes(src, dst, &in_shape, &perm, esize);
            return Ok(());
        }

        // Slow path: a strided or non-host input, or an output that aliases the
        // input or is not dense. Materialize both ends as the old kernel did.
        let din = to_dense_bytes(&inputs[0])?;
        let mut out = vec![0u8; total * esize];
        permute_bytes(&din, &mut out, &in_shape, &perm, esize);
        write_dense_bytes(&mut outputs[0], &out)
    }

    fn supports_strided_input(&self, _input_idx: usize) -> bool {
        true
    }

    fn view_outputs(&self, inputs: &[TensorView], num_outputs: usize) -> Option<Vec<ViewOutput>> {
        if num_outputs != 1 || inputs.len() != 1 || inputs[0].dtype.byte_size() == 0 {
            return None;
        }
        let input = &inputs[0];
        let rank = input.shape.len();
        let perm = self
            .perm
            .clone()
            .unwrap_or_else(|| (0..rank).rev().collect());
        if perm.len() != rank || perm.iter().any(|&axis| axis >= rank) || {
            let mut sorted = perm.clone();
            sorted.sort_unstable();
            sorted != (0..rank).collect::<Vec<_>>()
        } {
            return None;
        }
        Some(vec![ViewOutput {
            input_index: 0,
            shape: perm.iter().map(|&axis| input.shape[axis]).collect(),
            strides: perm.iter().map(|&axis| input.strides[axis]).collect(),
            byte_offset: input.byte_offset,
        }])
    }
}

/// Whether `output` is a dense host tensor of `bytes` bytes that does not
/// overlap `input`.
///
/// Writing through a raw slice into a tensor that aliases the input would
/// corrupt elements the permutation has not read yet, and `copy_from_slice` on
/// overlapping regions is undefined behaviour outright. The check is a handful
/// of pointer comparisons, so the disjoint case — every graph-allocated output —
/// keeps the direct-write speed.
fn dense_disjoint_output(input: &TensorView, output: &mut TensorMut, bytes: usize) -> bool {
    if !output.is_contiguous()
        || !output.device.is_host_accessible()
        || output.dtype != input.dtype
        || output.numel() * output.dtype.byte_size() != bytes
    {
        return false;
    }
    let out_start = output.data_ptr_mut::<u8>() as usize;
    let in_start = input.data_ptr::<u8>() as usize;
    let out_end = out_start.saturating_add(bytes);
    let in_end = in_start.saturating_add(bytes);
    out_start >= in_end || in_start >= out_end
}

/// Permute `src` into `dst`, both dense row-major, choosing the cheapest path.
fn permute_bytes(src: &[u8], dst: &mut [u8], in_shape: &[usize], perm: &[usize], esize: usize) {
    debug_assert_eq!(src.len(), dst.len());
    let collapsed = collapse(in_shape, perm);
    let rank = collapsed.shape.len();

    // Paths 1 and 2: nothing interleaves, so the tensors are byte-identical.
    if rank <= 1 || collapsed.perm.iter().copied().eq(0..rank) {
        dst.copy_from_slice(src);
        return;
    }

    // Path 3: the permutation fixes a trailing run of axes, so whole blocks of
    // `inner` elements move together.
    let fixed_suffix = collapsed
        .perm
        .iter()
        .enumerate()
        .rev()
        .take_while(|&(i, &p)| i == p)
        .count();
    if fixed_suffix > 0 {
        let split = rank - fixed_suffix;
        let inner: usize = collapsed.shape[split..].iter().product();
        block_move(
            src,
            dst,
            &collapsed.shape[..split],
            &collapsed.perm[..split],
            inner * esize,
        );
        return;
    }

    // Path 4: the last axis moves. Reduce to batched 2-D transposes when the
    // permutation is a pure swap of the two innermost axes under any batch
    // prefix, which is what every real attention and matmul layout collapses to.
    if rank == 2 {
        blocked_2d(src, dst, collapsed.shape[0], collapsed.shape[1], esize);
        return;
    }
    if collapsed.perm[..rank - 2].iter().copied().eq(0..rank - 2)
        && collapsed.perm[rank - 2] == rank - 1
        && collapsed.perm[rank - 1] == rank - 2
    {
        let batch: usize = collapsed.shape[..rank - 2].iter().product();
        let rows = collapsed.shape[rank - 2];
        let cols = collapsed.shape[rank - 1];
        let plane = rows * cols * esize;
        batched_blocked_2d(src, dst, batch, rows, cols, esize, plane);
        return;
    }

    // General fallback: an odometer, but over the collapsed axes only, and
    // writing straight into `dst`.
    scatter_general(src, dst, &collapsed.shape, &collapsed.perm, esize);
}

/// Move `block` contiguous bytes per output position, permuting the positions.
///
/// `shape`/`perm` describe the outer axes only; the trailing axes the
/// permutation fixed have already been folded into `block`.
fn block_move(src: &[u8], dst: &mut [u8], shape: &[usize], perm: &[usize], block: usize) {
    let out_shape: Vec<usize> = perm.iter().map(|&p| shape[p]).collect();
    let in_strides = compute_contiguous_strides(shape);
    let blocks = numel(&out_shape);
    debug_assert_eq!(blocks * block, dst.len());

    // Strides of the *output* walk expressed in input blocks, so the source
    // offset for output block `n` is a dot product with `n`'s mixed-radix
    // digits — no odometer state to carry between tasks, which is what lets the
    // parallel split below hand each task an arbitrary block range.
    let src_block_strides: Vec<usize> = perm.iter().map(|&p| in_strides[p] as usize).collect();

    let body = |dst_chunk: &mut [u8], first_block: usize| {
        let mut idx = unflatten(first_block, &out_shape);
        let mut written = 0usize;
        loop {
            let src_block: usize = idx
                .iter()
                .zip(&src_block_strides)
                .map(|(&i, &s)| i * s)
                .sum();
            let src_at = src_block * block;
            dst_chunk[written..written + block].copy_from_slice(&src[src_at..src_at + block]);
            written += block;
            if written == dst_chunk.len() || !next_index(&out_shape, &mut idx) {
                break;
            }
        }
    };

    match parallel_blocks_per_task(blocks, block) {
        Some(per_task) => {
            use rayon::prelude::*;
            dst.par_chunks_mut(per_task * block)
                .enumerate()
                .for_each(|(task, chunk)| body(chunk, task * per_task));
        }
        None => body(dst, 0),
    }
}

/// Cache-blocked 2-D transpose of a `rows x cols` matrix of `esize`-byte
/// elements, split across workers by destination row bands.
fn blocked_2d(src: &[u8], dst: &mut [u8], rows: usize, cols: usize, esize: usize) {
    // The destination is `cols x rows`, so a band of destination rows is a
    // contiguous, disjoint range — which is what lets `par_chunks_mut` hand each
    // worker a `&mut [u8]` without any unsafe pointer splitting.
    let dst_row = rows * esize;
    match parallel_blocks_per_task(cols, dst_row) {
        Some(per_task) => {
            use rayon::prelude::*;
            dst.par_chunks_mut(per_task * dst_row)
                .enumerate()
                .for_each(|(task, chunk)| {
                    blocked_2d_band(src, chunk, rows, cols, esize, task * per_task);
                });
        }
        None => blocked_2d_band(src, dst, rows, cols, esize, 0),
    }
}

/// One band of destination rows `[first_dst_row, first_dst_row + band)`.
///
/// `band` is inferred from `dst.len()`, so the final chunk's short tail needs no
/// special case.
fn blocked_2d_band(
    src: &[u8],
    dst: &mut [u8],
    rows: usize,
    cols: usize,
    esize: usize,
    first_dst_row: usize,
) {
    let dst_row = rows * esize;
    let band = dst.len() / dst_row;
    let mut c0 = first_dst_row;
    while c0 < first_dst_row + band {
        let cn = TILE.min(first_dst_row + band - c0);
        let mut r0 = 0;
        while r0 < rows {
            let rn = TILE.min(rows - r0);
            for c in c0..c0 + cn {
                let base = (c - first_dst_row) * dst_row + r0 * esize;
                let out_run = &mut dst[base..base + rn * esize];
                for r in 0..rn {
                    let src_at = ((r0 + r) * cols + c) * esize;
                    out_run[r * esize..(r + 1) * esize]
                        .copy_from_slice(&src[src_at..src_at + esize]);
                }
            }
            r0 += TILE;
        }
        c0 += cn;
    }
}

/// `batch` independent `rows x cols` transposes laid out back to back.
///
/// Splitting at the batch level when there are enough planes keeps each worker
/// on whole planes; `blocked_2d_band` rather than `blocked_2d` is called inside
/// so the two levels cannot both fan out for the same bytes.
fn batched_blocked_2d(
    src: &[u8],
    dst: &mut [u8],
    batch: usize,
    rows: usize,
    cols: usize,
    esize: usize,
    plane: usize,
) {
    match parallel_blocks_per_task(batch, plane) {
        Some(per_task) => {
            use rayon::prelude::*;
            dst.par_chunks_mut(per_task * plane)
                .enumerate()
                .for_each(|(task, chunk)| {
                    for (n, dst_plane) in chunk.chunks_mut(plane).enumerate() {
                        let base = (task * per_task + n) * plane;
                        blocked_2d_band(&src[base..base + plane], dst_plane, rows, cols, esize, 0);
                    }
                });
        }
        None => {
            // Too few planes to split, so let each plane decide for itself
            // whether it is large enough to fan out.
            for (n, dst_plane) in dst.chunks_mut(plane).enumerate() {
                let base = n * plane;
                blocked_2d(&src[base..base + plane], dst_plane, rows, cols, esize);
            }
        }
    }
}

/// Element-wise odometer over collapsed axes, writing straight into `dst`.
///
/// Reached only by permutations that move the last axis and are not a swap of
/// the innermost pair — rank >= 3 after collapsing with a rotation, which real
/// models produce rarely. It is still strictly better than the original: the
/// odometer runs over the collapsed rank, and there is no intermediate buffer.
fn scatter_general(src: &[u8], dst: &mut [u8], shape: &[usize], perm: &[usize], esize: usize) {
    let out_shape: Vec<usize> = perm.iter().map(|&p| shape[p]).collect();
    let in_strides = compute_contiguous_strides(shape);
    let src_strides: Vec<usize> = perm.iter().map(|&p| in_strides[p] as usize).collect();
    let mut idx = vec![0usize; shape.len()];
    let mut written = 0usize;
    loop {
        let src_elem: usize = idx.iter().zip(&src_strides).map(|(&i, &s)| i * s).sum();
        let src_at = src_elem * esize;
        dst[written..written + esize].copy_from_slice(&src[src_at..src_at + esize]);
        written += esize;
        if !next_index(&out_shape, &mut idx) {
            break;
        }
    }
}

/// Mixed-radix digits of `flat` under `shape`.
fn unflatten(mut flat: usize, shape: &[usize]) -> Vec<usize> {
    let mut idx = vec![0usize; shape.len()];
    for axis in (0..shape.len()).rev() {
        idx[axis] = flat % shape[axis];
        flat /= shape[axis];
    }
    idx
}

/// `Some(units_per_task)` when splitting `units` of `unit_bytes` each pays off.
fn parallel_blocks_per_task(units: usize, unit_bytes: usize) -> Option<usize> {
    if units.saturating_mul(unit_bytes) < MIN_PARALLEL_BYTES {
        return None;
    }
    let workers = rayon::current_num_threads().max(1);
    if workers < 2 || units < 2 {
        return None;
    }
    Some(units.div_ceil(workers).max(1))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::testutil::Owned;
    use onnx_runtime_ir::{DataType, DeviceId};

    fn run(perm: Option<Vec<usize>>, input: &Owned, out: &mut Owned) {
        let k = TransposeKernel { perm };
        k.execute(&[input.view()], &mut [out.view_mut()]).unwrap();
    }

    /// An f32-shaped buffer carrying arbitrary bytes, so a test can compare raw
    /// element movement without going through float formatting.
    fn raw_f32(shape: &[usize], bytes: &[u8]) -> Owned {
        let mut owned = Owned::zeros(DataType::Float32, shape);
        assert_eq!(owned.bytes.len(), bytes.len());
        owned.bytes.copy_from_slice(bytes);
        owned
    }

    /// The element-at-a-time walk the kernel used to perform, kept verbatim as
    /// the reference every fast path is checked against.
    fn reference(src: &[u8], in_shape: &[usize], perm: &[usize], esize: usize) -> Vec<u8> {
        let out_shape: Vec<usize> = perm.iter().map(|&p| in_shape[p]).collect();
        let in_strides = compute_contiguous_strides(in_shape);
        let mut out = vec![0u8; numel(&out_shape) * esize];
        if out.is_empty() {
            return out;
        }
        let mut oidx = vec![0usize; in_shape.len()];
        let mut flat = 0usize;
        loop {
            let mut in_flat = 0i64;
            for (i, &p) in perm.iter().enumerate() {
                in_flat += in_strides[p] * oidx[i] as i64;
            }
            let src_at = in_flat as usize * esize;
            out[flat * esize..flat * esize + esize].copy_from_slice(&src[src_at..src_at + esize]);
            flat += 1;
            if !next_index(&out_shape, &mut oidx) {
                break;
            }
        }
        out
    }

    fn permutations(rank: usize) -> Vec<Vec<usize>> {
        fn go(current: &mut Vec<usize>, k: usize, out: &mut Vec<Vec<usize>>) {
            if k == current.len() {
                out.push(current.clone());
                return;
            }
            for i in k..current.len() {
                current.swap(k, i);
                go(current, k + 1, out);
                current.swap(k, i);
            }
        }
        let mut result = Vec::new();
        let mut current: Vec<usize> = (0..rank).collect();
        go(&mut current, 0, &mut result);
        result
    }

    #[test]
    fn transpose_2d_default_reverses() {
        let a = Owned::f32(&[2, 3], &[1., 2., 3., 4., 5., 6.]);
        let mut out = Owned::zeros_f32(&[3, 2]);
        run(None, &a, &mut out);
        assert_eq!(out.to_f32(), vec![1., 4., 2., 5., 3., 6.]);
    }

    #[test]
    fn transpose_3d_perm() {
        let a = Owned::f32(&[2, 1, 3], &[1., 2., 3., 4., 5., 6.]);
        let mut out = Owned::zeros_f32(&[1, 2, 3]);
        run(Some(vec![1, 0, 2]), &a, &mut out);
        assert_eq!(out.to_f32(), vec![1., 2., 3., 4., 5., 6.]);
    }

    #[test]
    fn transpose_3d_swap_last_two() {
        let a = Owned::f32(&[1, 2, 3], &[1., 2., 3., 4., 5., 6.]);
        let mut out = Owned::zeros_f32(&[1, 3, 2]);
        run(Some(vec![0, 2, 1]), &a, &mut out);
        assert_eq!(out.to_f32(), vec![1., 4., 2., 5., 3., 6.]);
    }

    #[test]
    fn transpose_is_a_zero_copy_strided_view() {
        let a = Owned::f32(&[2, 3], &[1., 2., 3., 4., 5., 6.]);
        let view = TransposeKernel {
            perm: Some(vec![1, 0]),
        }
        .view_outputs(&[a.view()], 1)
        .unwrap()
        .pop()
        .unwrap();
        assert_eq!(view.shape, [3, 2]);
        assert_eq!(view.strides, [1, 3]);
        assert_eq!(view.byte_offset, 0);
    }

    #[test]
    fn transpose_bf16_preserves_element_bits() {
        let x = Owned::bf16(&[2, 2], &[1., -2., 3., 4.]);
        let mut out = Owned::zeros(DataType::BFloat16, &[2, 2]);
        run(None, &x, &mut out);
        assert_eq!(
            out.to_u16_bits(),
            vec![
                x.to_u16_bits()[0],
                x.to_u16_bits()[2],
                x.to_u16_bits()[1],
                x.to_u16_bits()[3]
            ]
        );
    }

    /// The correctness contract for the whole rewrite: every fast path must
    /// reproduce the original element-at-a-time walk exactly, for every
    /// permutation of every rank up to 4, including the size-1 axes that drive
    /// collapsing and the odd extents that leave tile remainders.
    #[test]
    fn every_path_matches_the_reference_walk() {
        let shapes: &[&[usize]] = &[
            &[7],
            &[5, 3],
            &[1, 8],
            &[8, 1],
            &[64, 64],
            &[65, 63],
            &[2, 3, 4],
            &[1, 6, 5],
            &[6, 1, 5],
            &[6, 5, 1],
            &[3, 70, 66],
            &[2, 3, 4, 5],
            &[2, 1, 4, 5],
            &[1, 3, 1, 5],
            &[2, 8, 3, 16],
            &[1, 2, 65, 33],
        ];
        for shape in shapes {
            let n = numel(shape);
            let src: Vec<u8> = (0..n * 4).map(|i| (i * 31 % 251) as u8).collect();
            let input = raw_f32(shape, &src);
            for perm in permutations(shape.len()) {
                let out_shape: Vec<usize> = perm.iter().map(|&p| shape[p]).collect();
                let expect = reference(&src, shape, &perm, 4);
                let mut out = Owned::zeros(DataType::Float32, &out_shape);
                run(Some(perm.clone()), &input, &mut out);
                assert_eq!(out.bytes, expect, "shape {shape:?} perm {perm:?}");
            }
        }
    }

    /// Collapsing is the step every fast path depends on, so check it against
    /// the element walk directly rather than only through the kernel.
    #[test]
    fn collapse_matches_the_element_walk() {
        let shapes: &[&[usize]] = &[&[4], &[3, 5], &[1, 4, 1], &[2, 3, 4], &[2, 1, 3, 4]];
        for shape in shapes {
            let n = numel(shape);
            let src: Vec<u8> = (0..n).map(|i| (i % 253) as u8).collect();
            for perm in permutations(shape.len()) {
                let collapsed = collapse(shape, &perm);
                assert_eq!(
                    numel(&collapsed.shape),
                    n,
                    "collapse changed the element count for {shape:?} {perm:?}"
                );
                assert_eq!(
                    reference(&src, &collapsed.shape, &collapsed.perm, 1),
                    reference(&src, shape, &perm, 1),
                    "shape {shape:?} perm {perm:?}"
                );
            }
        }
    }

    /// A permutation that only moves size-1 axes, or that is written out as the
    /// identity, must reach the single-`memcpy` path rather than the scatter.
    #[test]
    fn identity_shaped_permutations_collapse_to_a_single_block() {
        assert_eq!(collapse(&[2, 3, 4], &[0, 1, 2]).shape, vec![24]);
        assert_eq!(collapse(&[1, 6, 1], &[2, 1, 0]).shape, vec![6]);
        assert_eq!(collapse(&[4, 1, 5], &[1, 0, 2]).shape, vec![20]);
        // A genuine interleave must survive collapsing.
        assert_eq!(
            collapse(&[2, 3, 4], &[1, 0, 2]),
            Collapsed {
                shape: vec![2, 3, 4],
                perm: vec![1, 0, 2]
            }
        );
        // The attention layout: (B,S,H,D) -> (B,H,S,D) keeps D innermost, so it
        // stays a block move over three axes rather than becoming a scatter.
        assert_eq!(
            collapse(&[2, 8, 4, 64], &[0, 2, 1, 3]),
            Collapsed {
                shape: vec![2, 8, 4, 64],
                perm: vec![0, 2, 1, 3]
            }
        );
    }

    /// The direct-write path must not be taken when the output aliases the
    /// input, and the fallback it takes instead must still be correct. Without
    /// the guard this is a silent wrong-answer bug, not a crash.
    #[test]
    fn an_aliasing_output_falls_back_and_stays_correct() {
        use onnx_runtime_ep_api::{DevicePtr, DevicePtrMut, TensorMut, TensorView};

        let shape = [4usize, 4];
        let strides = [4i64, 1];
        let mut buffer: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let in_ptr = buffer.as_ptr() as *const std::ffi::c_void;
        let out_ptr = buffer.as_mut_ptr() as *mut std::ffi::c_void;
        let cpu = DeviceId::cpu();
        let view = TensorView::new(DevicePtr(in_ptr), DataType::Float32, &shape, &strides, cpu);
        let mut out = TensorMut::new(
            DevicePtrMut(out_ptr),
            DataType::Float32,
            &shape,
            &strides,
            cpu,
        );
        assert!(
            !dense_disjoint_output(&view, &mut out, 64),
            "an exactly-aliasing output must be rejected"
        );
        TransposeKernel { perm: None }
            .execute(&[view], &mut [out])
            .unwrap();
        let expect: Vec<f32> = (0..16).map(|i| ((i % 4) * 4 + i / 4) as f32).collect();
        assert_eq!(buffer, expect, "the aliasing fallback produced wrong data");
    }

    /// A zero-element tensor must not touch memory or fail; the odometer in the
    /// old kernel was guarded by an emptiness check and the new paths need the
    /// same one.
    #[test]
    fn an_empty_tensor_is_a_no_op() {
        let input = Owned::zeros(DataType::Float32, &[0, 4]);
        let mut out = Owned::zeros(DataType::Float32, &[4, 0]);
        run(None, &input, &mut out);
        assert!(out.bytes.is_empty());
    }

    /// A permutation that is not a permutation must be rejected rather than
    /// indexing out of bounds.
    #[test]
    fn a_malformed_perm_is_rejected() {
        let input = Owned::f32(&[2, 2], &[1., 2., 3., 4.]);
        let mut out = Owned::zeros_f32(&[2, 2]);
        let error = TransposeKernel {
            perm: Some(vec![0, 0]),
        }
        .execute(&[input.view()], &mut [out.view_mut()])
        .expect_err("a repeated axis is not a permutation");
        assert!(format!("{error}").contains("not a permutation"), "{error}");
    }

    /// The parallel fan-out must be bit-identical to the serial path, and must
    /// actually be exercised: tensors above the threshold with several workers,
    /// one per path.
    #[test]
    fn the_parallel_path_is_bit_identical_to_the_serial_path() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        let cases: &[(&[usize], &[usize])] = &[
            (&[4, 32, 8, 64], &[0, 2, 1, 3]),
            (&[512, 256], &[1, 0]),
            (&[8, 128, 96], &[0, 2, 1]),
        ];
        for (shape, perm) in cases {
            let n = numel(shape);
            let src: Vec<u8> = (0..n * 4).map(|i| (i * 17 % 251) as u8).collect();
            let expect = reference(&src, shape, perm, 4);
            let out_shape: Vec<usize> = perm.iter().map(|&p| shape[p]).collect();
            let input = raw_f32(shape, &src);
            let mut out = Owned::zeros(DataType::Float32, &out_shape);
            pool.install(|| run(Some(perm.to_vec()), &input, &mut out));
            assert_eq!(out.bytes, expect, "{shape:?} {perm:?}");
            // And serially, from the same inputs, to pin that the split is not
            // what makes it right.
            let mut serial = Owned::zeros(DataType::Float32, &out_shape);
            rayon::ThreadPoolBuilder::new()
                .num_threads(1)
                .build()
                .unwrap()
                .install(|| run(Some(perm.to_vec()), &input, &mut serial));
            assert_eq!(serial.bytes, out.bytes, "{shape:?} {perm:?}");
        }
    }

    /// Every output byte must be written exactly once. Extents deliberately not
    /// multiples of `TILE` so every remainder branch runs; the output starts
    /// poisoned so a skipped tile is a value mismatch rather than a lucky zero.
    #[test]
    fn no_output_byte_is_left_unwritten() {
        let cases: &[(&[usize], &[usize])] = &[
            (&[131, 67], &[1, 0]),
            (&[3, 129, 65], &[0, 2, 1]),
            (&[2, 5, 3, 33], &[0, 2, 1, 3]),
            (&[7, 9, 11], &[2, 0, 1]),
        ];
        for (shape, perm) in cases {
            let n = numel(shape);
            let src: Vec<u8> = (0..n * 4).map(|i| ((i * 13 + 7) % 251) as u8).collect();
            let out_shape: Vec<usize> = perm.iter().map(|&p| shape[p]).collect();
            let input = raw_f32(shape, &src);
            let mut out = raw_f32(&out_shape, &vec![0xA5u8; src.len()]);
            run(Some(perm.to_vec()), &input, &mut out);
            assert_eq!(
                out.bytes,
                reference(&src, shape, perm, 4),
                "{shape:?} {perm:?}"
            );
        }
    }
}
