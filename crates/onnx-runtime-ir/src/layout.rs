//! Physical strided layout on tensor values (see `docs/architecture/ORT2.md` §5).
//!
//! Unlike upstream ONNX / `onnx-ir`, every [`crate::Value`] carries a
//! [`TensorLayout`]. This lets optimization passes track non-contiguous
//! (transposed / broadcast) layouts and eliminate copies at EP boundaries.

use crate::dtype::DataType;
use crate::error::IrError;

/// Compute row-major (C-order) contiguous strides, in **elements**, for a shape.
pub fn compute_contiguous_strides(shape: &[usize]) -> Vec<i64> {
    let n = shape.len();
    let mut strides = vec![1i64; n];
    for i in (0..n.saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1] as i64;
    }
    strides
}

/// Whether `strides` describe a row-major contiguous layout for `shape`.
pub fn is_contiguous(shape: &[usize], strides: &[i64]) -> bool {
    strides == compute_contiguous_strides(shape).as_slice()
}

/// Whether a tensor with `shape` and `strides` is **dense**: it occupies a
/// contiguous block of memory (no holes, no overlaps) even though the logical
/// axis order may differ from row-major. This is exactly the condition under
/// which a per-element unary op can process the backing buffer wholesale —
/// every element lives at a unique offset in `[0, numel)` and the operation
/// is order-independent.
///
/// Formally: when dimensions are sorted by ascending absolute stride, each
/// stride must equal the product of all preceding dimensions' sizes. Dimensions
/// of size 0 or 1 are ignored (their stride is unconstrained because they
/// contribute no extent).
///
/// This is strictly weaker than [`is_contiguous`]: every contiguous tensor is
/// dense, but a column-major or NHWC-permuted tensor is dense without being
/// row-major contiguous.
pub fn is_dense(shape: &[usize], strides: &[i64]) -> bool {
    if shape.len() != strides.len() {
        return false;
    }
    // `is_dense` runs once per operand per node on the dispatch path. Collecting
    // into a `Vec` to inspect a handful of numbers put a heap allocation there:
    // perf sampling of a 100-node elementwise chain attributed 5.25% of this
    // EP's dispatch time to this function and the `Vec` it built, against 3%
    // for the arithmetic the whole graph exists to do.
    //
    // Rank <= 8 covers every tensor ONNX produces in practice, so those pairs
    // live on the stack. Higher ranks keep the heap path rather than impose a
    // limit the type system does not have. Both paths hand the same slice to
    // the same routine, so there is one implementation of the predicate.
    const INLINE_RANK: usize = 8;
    let nontrivial = |(&d, &s): (&usize, &i64)| (s.unsigned_abs() as i64, d);
    if shape.len() <= INLINE_RANK {
        let mut pairs = [(0i64, 0usize); INLINE_RANK];
        let mut len = 0;
        for pair in shape
            .iter()
            .zip(strides)
            .filter(|&(&d, _)| d > 1)
            .map(nontrivial)
        {
            pairs[len] = pair;
            len += 1;
        }
        dense_extents(&mut pairs[..len])
    } else {
        let mut pairs: Vec<(i64, usize)> = shape
            .iter()
            .zip(strides)
            .filter(|&(&d, _)| d > 1)
            .map(nontrivial)
            .collect();
        dense_extents(&mut pairs)
    }
}

/// The density predicate over the non-trivial `(abs_stride, size)` extents.
///
/// Sorts `pairs` in place by ascending stride, so the caller owns the storage
/// and `is_dense` can keep it on the stack for ordinary ranks.
fn dense_extents(pairs: &mut [(i64, usize)]) -> bool {
    if pairs.is_empty() {
        return true; // scalar or all-ones shape
    }
    // Sort by stride ascending.
    pairs.sort_unstable_by_key(|&(s, _)| s);
    // The smallest stride must be 1 (element-adjacent).
    if pairs[0].0 != 1 {
        return false;
    }
    // Each subsequent stride must equal the product of all preceding sizes.
    let mut expected_stride: i64 = 1;
    for &(stride, size) in &*pairs {
        if stride != expected_stride {
            return false;
        }
        expected_stride *= size as i64;
    }
    true
}

/// Compute the output shape of a numpy-style broadcast of `a` and `b`.
pub fn broadcast_shapes(a: &[usize], b: &[usize]) -> Result<Vec<usize>, IrError> {
    let max_ndim = a.len().max(b.len());
    let mut result = Vec::with_capacity(max_ndim);
    for i in 0..max_ndim {
        let da = if i < a.len() { a[a.len() - 1 - i] } else { 1 };
        let db = if i < b.len() { b[b.len() - 1 - i] } else { 1 };
        if da == db || db == 1 {
            result.push(da);
        } else if da == 1 {
            result.push(db);
        } else {
            return Err(IrError::BroadcastIncompatible {
                a: a.to_vec(),
                b: b.to_vec(),
            });
        }
    }
    result.reverse();
    Ok(result)
}

/// Memory-format hint used to pick vectorized kernels.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub enum MemoryFormat {
    /// Standard row-major.
    #[default]
    Contiguous,
    /// NHWC channels-last.
    ChannelsLast,
    /// Blocked/tiled format with the given block width (e.g. 16 for VNNI/AMX).
    Blocked(usize),
    /// An arbitrary strided layout that matches none of the named formats.
    Custom,
}

/// First-class strided layout for a value.
///
/// `strides == None` means "contiguous row-major for the value's shape"; this
/// is the common case and avoids materializing strides for every value.
#[derive(Clone, Debug, PartialEq)]
pub struct TensorLayout {
    /// Physical strides in **elements**. `None` == contiguous row-major.
    pub strides: Option<Vec<i64>>,
    /// Memory-format hint.
    pub format: MemoryFormat,
    /// Required alignment in bytes for the backing allocation.
    pub alignment: usize,
}

/// Default alignment (bytes) — 64 covers AVX-512 / cache-line requirements.
pub const DEFAULT_ALIGNMENT: usize = 64;

impl Default for TensorLayout {
    fn default() -> Self {
        Self {
            strides: None,
            format: MemoryFormat::Contiguous,
            alignment: DEFAULT_ALIGNMENT,
        }
    }
}

impl TensorLayout {
    /// A contiguous row-major layout (strides implied by shape).
    pub fn contiguous() -> Self {
        Self::default()
    }

    /// A layout with explicit strides (marked [`MemoryFormat::Custom`]).
    pub fn strided(strides: Vec<i64>) -> Self {
        Self {
            strides: Some(strides),
            format: MemoryFormat::Custom,
            alignment: DEFAULT_ALIGNMENT,
        }
    }

    /// Whether this layout is contiguous row-major for `shape`.
    pub fn is_contiguous(&self, shape: &[usize]) -> bool {
        match &self.strides {
            None => true,
            Some(s) => is_contiguous(shape, s),
        }
    }

    /// The strides for `shape` under this layout, materializing the implied
    /// contiguous strides when `strides == None`.
    pub fn resolved_strides(&self, shape: &[usize]) -> Vec<i64> {
        self.strides
            .clone()
            .unwrap_or_else(|| compute_contiguous_strides(shape))
    }

    /// Reorder axes without copying data (a lazy transpose).
    pub fn transpose(&self, shape: &[usize], perm: &[usize]) -> Self {
        let base = self.resolved_strides(shape);
        let strides = perm.iter().map(|&p| base[p]).collect();
        Self {
            strides: Some(strides),
            format: MemoryFormat::Custom,
            alignment: self.alignment,
        }
    }

    /// Total backing storage size in bytes: the largest byte offset reachable
    /// via the strides, plus one element. Handles negative strides.
    pub fn storage_size(&self, shape: &[usize], dtype: DataType) -> usize {
        let elem = dtype.byte_size().max(1);
        match &self.strides {
            None => shape.iter().product::<usize>() * elem,
            Some(strides) => {
                let max_offset: i64 = shape
                    .iter()
                    .zip(strides.iter())
                    .map(|(&dim, &stride)| dim.saturating_sub(1) as i64 * stride.abs())
                    .sum();
                (max_offset as usize + 1) * elem
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The pre-optimisation `is_dense`, verbatim, as a differential oracle.
    ///
    /// Kept deliberately naive and heap-based: its whole value is that it was
    /// not written by the same edit as the version under test, so a mistake in
    /// the inline-storage path cannot hide behind a matching mistake here.
    fn is_dense_reference(shape: &[usize], strides: &[i64]) -> bool {
        if shape.len() != strides.len() {
            return false;
        }
        let mut pairs: Vec<(i64, usize)> = shape
            .iter()
            .zip(strides)
            .filter(|&(&d, _)| d > 1)
            .map(|(&d, &s)| (s.unsigned_abs() as i64, d))
            .collect();
        if pairs.is_empty() {
            return true;
        }
        pairs.sort_unstable_by_key(|&(s, _)| s);
        if pairs[0].0 != 1 {
            return false;
        }
        let mut expected_stride: i64 = 1;
        for &(stride, size) in &pairs {
            if stride != expected_stride {
                return false;
            }
            expected_stride *= size as i64;
        }
        true
    }

    /// Every rank the inline path serves, plus the ranks that spill to the
    /// heap, must agree with the original implementation on every case --
    /// dense, non-dense, permuted, zero-sized, negative-strided and
    /// mismatched-length. Falsifier — change `INLINE_RANK`, drop the `d > 1`
    /// filter, or forget to truncate the inline array to `len`, and the two
    /// implementations disagree here.
    #[test]
    fn inline_storage_agrees_with_the_original_implementation() {
        // Deterministic pseudo-random cases: a fixed LCG so a failure is
        // reproducible from the printed inputs alone.
        let mut state: u64 = 0x2545_F491_4F6C_DD1D;
        let mut next = |bound: u64| -> u64 {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 33) % bound
        };

        let mut checked_dense = 0usize;
        // Ranks 0..=10 straddle INLINE_RANK (8) in both directions.
        for rank in 0..=10usize {
            for _ in 0..400 {
                let shape: Vec<usize> = (0..rank).map(|_| next(4) as usize).collect();
                // Mix genuinely contiguous layouts (so the dense arm is
                // actually exercised, not just the early rejections) with
                // arbitrary ones.
                let strides: Vec<i64> = if next(2) == 0 {
                    compute_contiguous_strides(&shape)
                } else {
                    (0..rank)
                        .map(|_| next(9) as i64 - 4) // includes 0 and negatives
                        .collect()
                };
                let got = is_dense(&shape, &strides);
                if got {
                    checked_dense += 1;
                }
                assert_eq!(
                    got,
                    is_dense_reference(&shape, &strides),
                    "disagreement for shape {shape:?} strides {strides:?}"
                );

                // Length mismatch must stay a rejection on both paths.
                let mut short = strides.clone();
                short.pop();
                assert_eq!(
                    is_dense(&shape, &short),
                    is_dense_reference(&shape, &short),
                    "disagreement for shape {shape:?} strides {short:?}"
                );
            }
        }
        assert!(
            checked_dense > 100,
            "the corpus degenerated into rejections only; it proved nothing about \
             the dense arm (only {checked_dense} dense cases)"
        );
    }

    /// The heap fallback must still be reachable and correct: a rank above
    /// `INLINE_RANK` cannot fit the stack array.
    #[test]
    fn ranks_above_the_inline_bound_use_the_heap_path_correctly() {
        let shape = [2usize, 2, 2, 2, 2, 2, 2, 2, 2];
        let strides = compute_contiguous_strides(&shape);
        assert!(shape.len() > 8, "this test must exercise the fallback");
        assert!(is_dense(&shape, &strides));
        assert!(is_dense_reference(&shape, &strides));

        let mut broken = strides.clone();
        broken[0] += 1;
        assert!(!is_dense(&shape, &broken));
        assert!(!is_dense_reference(&shape, &broken));
    }

    #[test]
    fn contiguous_strides_row_major() {
        assert_eq!(compute_contiguous_strides(&[2, 3, 4]), vec![12, 4, 1]);
        assert_eq!(compute_contiguous_strides(&[5]), vec![1]);
        assert_eq!(compute_contiguous_strides(&[]), Vec::<i64>::new());
    }

    #[test]
    fn is_contiguous_check() {
        assert!(is_contiguous(&[2, 3], &[3, 1]));
        assert!(!is_contiguous(&[2, 3], &[1, 2]));
    }

    #[test]
    fn broadcast_basic() {
        assert_eq!(broadcast_shapes(&[3, 1], &[1, 4]).unwrap(), vec![3, 4]);
        assert_eq!(broadcast_shapes(&[5], &[3, 5]).unwrap(), vec![3, 5]);
        assert_eq!(broadcast_shapes(&[], &[2, 2]).unwrap(), vec![2, 2]);
    }

    #[test]
    fn broadcast_incompatible() {
        assert!(matches!(
            broadcast_shapes(&[3], &[4]),
            Err(IrError::BroadcastIncompatible { .. })
        ));
    }

    #[test]
    fn transpose_swaps_strides() {
        let l = TensorLayout::contiguous();
        let t = l.transpose(&[2, 3], &[1, 0]);
        // contiguous [2,3] -> strides [3,1]; transposed -> [1,3]
        assert_eq!(t.strides, Some(vec![1, 3]));
        assert!(!t.is_contiguous(&[3, 2]));
    }

    #[test]
    fn storage_size_contiguous_and_strided() {
        let l = TensorLayout::contiguous();
        assert_eq!(l.storage_size(&[2, 3], DataType::Float32), 24);
        // transposed view still covers the same 6 elements
        let t = l.transpose(&[2, 3], &[1, 0]);
        assert_eq!(t.storage_size(&[3, 2], DataType::Float32), 24);
    }
}
