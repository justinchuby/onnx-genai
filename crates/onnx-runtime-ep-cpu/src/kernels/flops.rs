//! Shared *structural* FLOP estimators for CPU kernels (issue #995).
//!
//! These count arithmetic work as a machine-INDEPENDENT property of the op and
//! its input shapes. They contain **no** machine rates — a FLOP/s figure is a
//! host property supplied by `onnx-runtime-cost-model`, never baked in here.
//!
//! Every estimator returns `Option<u64>` and yields `None` when the shape is not
//! statically known (issue #995 constraint: "Unknown must be representable" —
//! we never substitute a plausible-looking default for a value we do not know).

/// Element count of a fully-static shape. A rank-0 shape (scalar) has one
/// element, matching ONNX broadcasting.
pub(crate) fn numel(shape: &[usize]) -> u64 {
    shape.iter().map(|&d| d as u64).product::<u64>().max(1)
}

/// FLOPs for an elementwise op over `input_shapes`: one arithmetic op per output
/// element. The output is the numpy-broadcast of the inputs, whose element count
/// is the max of the input element counts. Returns `None` if there are no inputs.
pub(crate) fn elementwise_flops(input_shapes: &[Vec<usize>]) -> Option<u64> {
    input_shapes.iter().map(|s| numel(s)).max()
}

/// FLOPs for a dense `MatMul(A, B)` where `a`/`b` are the full input shapes.
///
/// Uses the ONNX MatMul convention: the last two dims of each operand are the
/// matrix `[M, K]` and `[K, N]`; any leading dims are batch and broadcast. The
/// multiply-add over the shared `K` costs `2*M*N*K` per batch element. 1-D
/// operands are promoted per ONNX (prepend/append a 1) then the added dim is
/// removed from the batch. Returns `None` if either operand is rank-0 or the
/// inner dimensions disagree.
pub(crate) fn matmul_flops(a: &[usize], b: &[usize]) -> Option<u64> {
    if a.is_empty() || b.is_empty() {
        return None;
    }
    // Promote 1-D operands to 2-D per ONNX MatMul semantics.
    let (a2, a_prepended) = if a.len() == 1 {
        (vec![1usize, a[0]], true)
    } else {
        (a.to_vec(), false)
    };
    let (b2, b_appended) = if b.len() == 1 {
        (vec![b[0], 1usize], true)
    } else {
        (b.to_vec(), false)
    };
    let m = a2[a2.len() - 2] as u64;
    let ka = a2[a2.len() - 1] as u64;
    let kb = b2[b2.len() - 2] as u64;
    let n = b2[b2.len() - 1] as u64;
    if ka != kb {
        return None;
    }
    // Broadcast the batch dims (everything but the trailing matrix dims).
    let a_batch = &a2[..a2.len() - 2];
    let b_batch = &b2[..b2.len() - 2];
    let batch = broadcast_numel(a_batch, b_batch)?;
    let _ = (a_prepended, b_appended); // promoted dims fold into M=1 / N=1.
    Some(
        2u64.saturating_mul(batch)
            .saturating_mul(m)
            .saturating_mul(n)
            .saturating_mul(ka),
    )
}

/// Element count of the numpy-broadcast of two batch-dim slices, or `None` if
/// they are not broadcast-compatible.
fn broadcast_numel(a: &[usize], b: &[usize]) -> Option<u64> {
    let rank = a.len().max(b.len());
    let mut acc: u64 = 1;
    for i in 1..=rank {
        // Align from the right: the i-th dim from the end of each operand.
        let da = if i <= a.len() { a[a.len() - i] } else { 1 };
        let db = if i <= b.len() { b[b.len() - i] } else { 1 };
        let dim = if da == db {
            da
        } else if da == 1 {
            db
        } else if db == 1 {
            da
        } else {
            return None;
        };
        acc = acc.saturating_mul(dim as u64);
    }
    Some(acc.max(1))
}

/// FLOPs for a `MatMulNBits` GEMM: `A[rows, k] x Wᵀ[k, n]` after dequant. The
/// dominant arithmetic is the `2*rows*n*k` multiply-add; dequant of the packed
/// weights is `O(n*k)` and lower-order, so it is omitted (kept structural, not
/// padded). Returns `None` if `rows` is unknown.
pub(crate) fn matmul_nbits_flops(rows: Option<u64>, n: u64, k: u64) -> Option<u64> {
    let rows = rows?;
    Some(
        2u64.saturating_mul(rows)
            .saturating_mul(n)
            .saturating_mul(k),
    )
}

/// Leading "row" count of a GEMM activation `A`: the product of every dim except
/// the last (the reduction dim `K`). For `[batch, seq, k]` this is `batch*seq`.
/// Returns `None` for a rank-0 shape.
pub(crate) fn leading_rows(a: &[usize]) -> Option<u64> {
    if a.is_empty() {
        return None;
    }
    Some(
        a[..a.len() - 1]
            .iter()
            .map(|&d| d as u64)
            .product::<u64>()
            .max(1),
    )
}

/// FLOPs for a single group-query-attention call, given fully-resolved geometry.
///
/// The two GEMMs — scores `Q·Kᵀ` and context `P·V` — each cost
/// `2*head_size` multiply-adds per (query, key) pair, so the total is
/// `2 * (2*head_size) * batch * num_heads * seq_q * seq_k`. Softmax and rotary
/// are `O(batch*num_heads*seq_q*seq_k)` and `O(batch*num_heads*seq_q*head_size)`
/// respectively — lower order, so they are omitted to keep the estimate
/// structural rather than padded.
///
/// NOTE: `seq_k` (the KV-cache occupancy) is **not** a static shape at graph
/// build time — in the GQA op it is carried by the runtime `seqlens_k` /
/// `total_sequence_length` value inputs. This function therefore exists for the
/// cost model to call once those values are known; the kernel's
/// `estimated_flops()` returns `None` because it cannot see them (issue #995
/// constraint 2: unknown must be representable, never fabricated).
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn gqa_flops(batch: u64, num_heads: u64, head_size: u64, seq_q: u64, seq_k: u64) -> u64 {
    let pairs = batch
        .saturating_mul(num_heads)
        .saturating_mul(seq_q)
        .saturating_mul(seq_k);
    // 2 GEMMs × (2·head_size) MACs per (query,key) pair.
    4u64.saturating_mul(head_size).saturating_mul(pairs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numel_scalar_is_one() {
        assert_eq!(numel(&[]), 1);
        assert_eq!(numel(&[2, 3, 4]), 24);
    }

    #[test]
    fn elementwise_broadcasts_to_max() {
        assert_eq!(elementwise_flops(&[vec![4, 1], vec![4, 8]]), Some(32));
        assert_eq!(elementwise_flops(&[]), None);
    }

    #[test]
    fn matmul_2d() {
        // [2,3] x [3,5] => 2*2*5*3 = 60
        assert_eq!(matmul_flops(&[2, 3], &[3, 5]), Some(60));
    }

    #[test]
    fn matmul_batched() {
        // [4,2,3] x [4,3,5] => batch 4 * 2*2*5*3 = 240
        assert_eq!(matmul_flops(&[4, 2, 3], &[4, 3, 5]), Some(240));
    }

    #[test]
    fn matmul_broadcast_batch() {
        // [4,2,3] x [3,5] => batch 4 => 240
        assert_eq!(matmul_flops(&[4, 2, 3], &[3, 5]), Some(240));
    }

    #[test]
    fn matmul_inner_mismatch_is_none() {
        assert_eq!(matmul_flops(&[2, 3], &[4, 5]), None);
    }

    #[test]
    fn nbits_needs_rows() {
        assert_eq!(matmul_nbits_flops(Some(2), 5, 3), Some(60));
        assert_eq!(matmul_nbits_flops(None, 5, 3), None);
    }

    #[test]
    fn leading_rows_products() {
        assert_eq!(leading_rows(&[2, 4, 8]), Some(8));
        assert_eq!(leading_rows(&[8]), Some(1));
        assert_eq!(leading_rows(&[]), None);
    }

    #[test]
    fn gqa_two_gemms() {
        // batch 1, 2 heads, head_size 4, seq_q 3, seq_k 5:
        // 4 * 4 * (1*2*3*5) = 16 * 30 = 480
        assert_eq!(gqa_flops(1, 2, 4, 3, 5), 480);
    }
}
