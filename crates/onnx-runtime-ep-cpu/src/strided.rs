//! Strided-view access helpers shared by the CPU kernels (`docs/ORT2.md` §5).
//!
//! Every kernel input arrives as a [`TensorView`](onnx_runtime_ep_api::TensorView)
//! that may be **non-contiguous** — a transposed weight, a broadcast bias, a
//! sliced activation — carrying DLPack-style element strides (possibly negative)
//! and a byte offset to the element origin. These helpers turn that description
//! into safe row-major iteration and centralize the two unsafe operations the
//! CPU EP needs: reading and writing an element at a computed offset.
//!
//! ## Enforcing Holden's view-bounds invariant (ep-api safety review #1)
//!
//! A `TensorView` does not know the size of its backing allocation, so
//! [`TensorView::validate`](onnx_runtime_ep_api::TensorView::validate) cannot
//! check storage bounds. [`view_in_bounds`] closes that gap: given the backing
//! allocation length it verifies that **both** the minimum and maximum byte
//! addressed by the (shape, strides, byte_offset) triple — accounting for
//! negative strides — lie inside `[0, buffer_len)`. Callers that own the buffer
//! (the session, Track D) MUST call it before handing a view to a kernel.

use onnx_runtime_ep_api::{EpError, Result};

/// Advance a row-major multi-dimensional index in place.
///
/// Returns `true` if `index` now points at the next element, or `false` when
/// the iteration space is exhausted (the index wrapped back to all-zero). A
/// rank-0 (scalar) shape has exactly one element, so the first call returns
/// `false`.
pub fn next_index(shape: &[usize], index: &mut [usize]) -> bool {
    for axis in (0..shape.len()).rev() {
        index[axis] += 1;
        if index[axis] < shape[axis] {
            return true;
        }
        index[axis] = 0;
    }
    false
}

/// Element offset (in elements, may be negative) of `index` from the element
/// origin, given `strides` in elements.
pub fn elem_offset(strides: &[i64], index: &[usize]) -> isize {
    let mut offset = 0i64;
    for (stride, &i) in strides.iter().zip(index) {
        offset += stride * i as i64;
    }
    offset as isize
}

/// The number of elements described by `shape`.
pub fn numel(shape: &[usize]) -> usize {
    shape.iter().product()
}

/// The inclusive element-offset range `[min, max]` addressed by a view of
/// `shape` with `strides`, relative to the element origin. Accounts for
/// negative strides (which reach *below* the origin).
pub fn addressed_elem_range(shape: &[usize], strides: &[i64]) -> (i64, i64) {
    let mut min = 0i64;
    let mut max = 0i64;
    for (&dim, &stride) in shape.iter().zip(strides) {
        if dim == 0 {
            continue;
        }
        let extent = (dim as i64 - 1) * stride;
        if extent < 0 {
            min += extent;
        } else {
            max += extent;
        }
    }
    (min, max)
}

/// Verify that every byte a view of `shape`/`strides`/`byte_offset` can address
/// falls within `[0, buffer_len)` for an element size of `esize` bytes.
///
/// This is the storage-bounds check that
/// [`TensorView::validate`](onnx_runtime_ep_api::TensorView::validate)
/// intentionally omits (it cannot see the allocation size). It upholds ep-api
/// safety-review invariant #1 and MUST run before any kernel dereferences a
/// view whose backing length is known to the caller.
pub fn view_in_bounds(
    shape: &[usize],
    strides: &[i64],
    byte_offset: usize,
    esize: usize,
    buffer_len: usize,
) -> Result<()> {
    if shape.len() != strides.len() {
        return Err(EpError::InvalidTensorView {
            reason: format!(
                "rank mismatch: shape {} dims, strides {}",
                shape.len(),
                strides.len()
            ),
        });
    }
    // An empty tensor addresses no bytes; the origin itself need not be valid.
    if numel(shape) == 0 {
        return Ok(());
    }
    let (min_elem, max_elem) = addressed_elem_range(shape, strides);
    // Compute the addressed byte range in i128 so the element→byte multiply and
    // the origin offset can never wrap: a huge static shape whose
    // `max_elem * esize` overflows i64 would otherwise wrap *below* buffer_len
    // and silently pass this gate, then read/write out of bounds. Any overflow
    // of even the i128 math is turned into a rejection, never a silent pass.
    let esize = esize as i128;
    let origin = byte_offset as i128;
    let lo = (min_elem as i128)
        .checked_mul(esize)
        .and_then(|m| origin.checked_add(m));
    let hi = (max_elem as i128)
        .checked_mul(esize)
        .and_then(|m| origin.checked_add(m))
        .and_then(|h| h.checked_add(esize)); // exclusive upper bound
    let (lo, hi) = match (lo, hi) {
        (Some(lo), Some(hi)) => (lo, hi),
        _ => {
            return Err(EpError::InvalidTensorView {
                reason: format!(
                    "view address computation overflowed (shape {shape:?}, strides {strides:?}, \
                     byte_offset {byte_offset}, esize {esize})"
                ),
            });
        }
    };
    if lo < 0 || hi > buffer_len as i128 {
        return Err(EpError::InvalidTensorView {
            reason: format!(
                "view addresses bytes [{lo}, {hi}) outside backing allocation [0, {buffer_len})"
            ),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn next_index_walks_row_major() {
        let shape = [2usize, 3];
        let mut idx = [0usize, 0];
        let mut seen = vec![idx];
        while next_index(&shape, &mut idx) {
            seen.push(idx);
        }
        assert_eq!(seen.len(), 6);
        assert_eq!(seen[0], [0, 0]);
        assert_eq!(seen[1], [0, 1]);
        assert_eq!(seen[3], [1, 0]);
        assert_eq!(seen[5], [1, 2]);
    }

    #[test]
    fn scalar_has_single_element() {
        let shape: [usize; 0] = [];
        let mut idx: [usize; 0] = [];
        assert!(!next_index(&shape, &mut idx));
    }

    #[test]
    fn offset_uses_strides() {
        // transposed [3,2] over [2,3] contiguous: strides [1,3]
        assert_eq!(elem_offset(&[1, 3], &[0, 0]), 0);
        assert_eq!(elem_offset(&[1, 3], &[2, 1]), 2 + 3);
    }

    #[test]
    fn addressed_range_handles_negative_strides() {
        // reversed axis of length 4: stride -1 reaches -3 below origin
        assert_eq!(addressed_elem_range(&[4], &[-1]), (-3, 0));
        assert_eq!(addressed_elem_range(&[2, 3], &[3, 1]), (0, 5));
    }

    #[test]
    fn bounds_accept_contiguous_and_reject_overrun() {
        // [2,3] f32 contiguous needs 24 bytes
        assert!(view_in_bounds(&[2, 3], &[3, 1], 0, 4, 24).is_ok());
        assert!(view_in_bounds(&[2, 3], &[3, 1], 0, 4, 23).is_err());
    }

    #[test]
    fn bounds_reject_negative_stride_underrun() {
        // stride -1 with origin at byte 0 would read before the buffer
        assert!(view_in_bounds(&[4], &[-1], 0, 4, 16).is_err());
        // ...but with the origin at the last element it is in bounds
        assert!(view_in_bounds(&[4], &[-1], 12, 4, 16).is_ok());
    }

    #[test]
    fn empty_tensor_is_in_bounds() {
        assert!(view_in_bounds(&[0, 5], &[5, 1], 0, 4, 0).is_ok());
    }

    /// H-D1 (bounds layer): a view whose `max_elem * esize` would overflow i64
    /// must be REJECTED. The old i64 gate wrapped this product below
    /// `buffer_len` and silently passed, enabling an out-of-bounds access on an
    /// under-sized buffer. The i128 computation makes the wrap impossible.
    #[test]
    fn bounds_reject_overflowing_address_math() {
        // max_elem = (2-1) * i64::MAX = i64::MAX; * esize (8) overflows i64.
        let shape = [2usize];
        let strides = [i64::MAX];
        let err = view_in_bounds(&shape, &strides, 0, 8, 1024);
        assert!(
            err.is_err(),
            "an address computation that overflows must be rejected, never wrap-passed"
        );
    }
}
