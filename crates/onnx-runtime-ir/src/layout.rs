//! Physical strided layout on tensor values (see `docs/ORT2.md` §5).
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
