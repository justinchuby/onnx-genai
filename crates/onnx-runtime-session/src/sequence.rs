//! Runtime **sequence-of-tensors** value type and the ONNX `Sequence*` op
//! semantics, implemented **copy-free** and **race-free**.
//!
//! ## Why this exists
//!
//! ONNX models can carry a `Sequence` value: an ordered, homogeneously-typed
//! list of tensors (`docs/ORT2.md` §3.2, `TypeProto::Sequence`). The stock
//! ONNX Runtime implementation of the sequence ops is *costly*: `SequenceInsert`
//! / `SequenceErase` rebuild the vector **and** deep-copy element tensor data,
//! and `SequenceAt` copies the selected element out. For a long sequence that is
//! O(total bytes) of memcpy per mutation.
//!
//! ## The no-copy invariant
//!
//! A sequence op here is **value-semantic over shared, immutable elements**:
//!
//! * Each element is an [`Arc`]-shared [`SeqTensor`]. A [`SeqTensor`] is
//!   **immutable once constructed** — no method ever mutates its bytes.
//! * A mutating op ([`SequenceValue::insert`], [`SequenceValue::erase`], …)
//!   returns a **new** [`SequenceValue`] whose `items` vector **shares the same
//!   element `Arc`s** as the input (a persistent-data-structure style update).
//!   Only `Arc` handles (pointers + a refcount bump) are cloned — **never the
//!   element bytes**.
//! * [`SequenceValue::at`] returns a *clone of the element `Arc`* — a shared
//!   handle to the exact same allocation that was inserted. No deep copy. The
//!   unit test `at_returns_shared_handle_no_copy` proves this with
//!   [`Arc::ptr_eq`] and a data-pointer equality assertion.
//!
//! ## The no-race guarantee
//!
//! Because a [`SeqTensor`] is immutable after construction and is only ever
//! *shared read-only* through [`Arc`], concurrent readers of the same element
//! (or the same [`SequenceValue`], which is [`Clone`] by `Arc`-sharing) observe
//! a stable, never-mutated view. There is **no interior mutability** anywhere in
//! this module, so no data race is possible: the only cross-thread interaction
//! is `Arc`'s atomic refcount, which is itself race-free. `SeqTensor: Send +
//! Sync` and therefore `SequenceValue: Send + Sync` (verified by
//! `sequence_value_is_send_sync`).

use std::sync::Arc;

use onnx_runtime_ir::DataType;

/// One immutable, `Arc`-shared tensor element of a runtime [`SequenceValue`].
///
/// Elements are stored as contiguous row-major little-endian bytes plus their
/// dtype and shape. **A `SeqTensor` is never mutated after construction** — this
/// is the invariant that makes sharing it across sequences and threads sound
/// (see the module docs). Always hold it behind an [`Arc`] (see
/// [`SeqTensor::shared`]) so a sequence op shares the handle instead of copying
/// the bytes.
#[derive(Debug)]
pub(crate) struct SeqTensor {
    pub dtype: DataType,
    pub shape: Vec<usize>,
    /// Contiguous row-major little-endian element bytes. Immutable.
    pub data: Vec<u8>,
}

impl SeqTensor {
    /// Wrap an owned tensor's bytes in a shared, immutable element handle. The
    /// bytes are moved in (no copy); every later sequence op shares this `Arc`.
    pub(crate) fn shared(dtype: DataType, shape: Vec<usize>, data: Vec<u8>) -> Arc<Self> {
        Arc::new(Self { dtype, shape, data })
    }

    /// Base address of the element bytes — used by the executor to hand a
    /// zero-copy [`TensorView`](onnx_runtime_ep_api::TensorView) over this
    /// element to a downstream kernel, and by tests to prove no deep copy
    /// occurred (the pointer is stable across every sequence op).
    pub(crate) fn as_ptr(&self) -> *const u8 {
        self.data.as_ptr()
    }
}

/// An ordered, immutable-element runtime **sequence** value.
///
/// Cloning is cheap: it clones the `items` vector, i.e. bumps each element's
/// `Arc` refcount — **no element bytes are copied**. Every mutating op returns a
/// fresh `SequenceValue` that shares surviving elements with the input.
#[derive(Clone, Debug)]
pub(crate) struct SequenceValue {
    /// The tensor element type every item shares (ONNX requires homogeneity).
    pub elem_dtype: DataType,
    /// Ordered elements, each an `Arc`-shared immutable tensor.
    pub items: Vec<Arc<SeqTensor>>,
}

/// A sequence-op failure carrying the actionable what/why/how (see `RULES.md`
/// §1). The executor maps this into `SessionError::SequenceOp`.
#[derive(Debug)]
pub(crate) struct SeqOpError {
    pub op: &'static str,
    pub reason: String,
}

impl SeqOpError {
    fn new(op: &'static str, reason: impl Into<String>) -> Self {
        Self {
            op,
            reason: reason.into(),
        }
    }
}

type SeqResult<T> = std::result::Result<T, SeqOpError>;

/// Resolve a possibly-negative ONNX access index against a sequence of length
/// `len`, returning the non-negative position or an out-of-bounds error whose
/// message states the valid range and the offending value.
fn resolve_index(op: &'static str, pos: i64, len: usize) -> SeqResult<usize> {
    let n = len as i64;
    let idx = if pos < 0 { n + pos } else { pos };
    if idx < 0 || idx >= n {
        return Err(SeqOpError::new(
            op,
            format!(
                "position {pos} is out of bounds for a sequence of length {len} \
                 (valid range is [{}, {}]; negative values count from the end). \
                 To fix: pass an index within range, or check the producer that \
                 computed this position",
                -n,
                n - 1
            ),
        ));
    }
    Ok(idx as usize)
}

impl SequenceValue {
    /// `SequenceEmpty`: an empty sequence with declared element dtype.
    pub(crate) fn empty(elem_dtype: DataType) -> Self {
        Self {
            elem_dtype,
            items: Vec::new(),
        }
    }

    /// `SequenceConstruct`: a sequence from N (≥1) element handles, which are
    /// **shared** (the `Arc`s are moved in, no bytes copied). Every element must
    /// share the sequence's dtype.
    pub(crate) fn construct(items: Vec<Arc<SeqTensor>>) -> SeqResult<Self> {
        let elem_dtype = items
            .first()
            .map(|t| t.dtype)
            .ok_or_else(|| {
                SeqOpError::new(
                    "SequenceConstruct",
                    "requires at least one input tensor, but none were supplied. \
                     To fix: pass ≥1 tensor, or use SequenceEmpty for an empty sequence",
                )
            })?;
        for (i, t) in items.iter().enumerate() {
            if t.dtype != elem_dtype {
                return Err(SeqOpError::new(
                    "SequenceConstruct",
                    format!(
                        "element {i} has dtype {:?} but the sequence element type is {:?} \
                         (a sequence is homogeneous). To fix: Cast the mismatched input",
                        t.dtype, elem_dtype
                    ),
                ));
            }
        }
        Ok(Self { elem_dtype, items })
    }

    /// Number of elements (`SequenceLength`).
    pub(crate) fn len(&self) -> usize {
        self.items.len()
    }

    /// `SequenceInsert`: a **new** sequence with `tensor` inserted at `position`
    /// (default = append). The returned sequence **shares** every existing
    /// element `Arc` plus the new one — no element bytes are copied.
    pub(crate) fn insert(
        &self,
        tensor: Arc<SeqTensor>,
        position: Option<i64>,
    ) -> SeqResult<SequenceValue> {
        if tensor.dtype != self.elem_dtype {
            return Err(SeqOpError::new(
                "SequenceInsert",
                format!(
                    "tensor dtype {:?} does not match the sequence element type {:?} \
                     (a sequence is homogeneous). To fix: Cast the tensor to {:?}",
                    tensor.dtype, self.elem_dtype, self.elem_dtype
                ),
            ));
        }
        let len = self.len();
        // Insertion admits one more slot than access: the back (index == len).
        let idx = match position {
            None => len,
            Some(p) => {
                let n = len as i64;
                let i = if p < 0 { n + p } else { p };
                if i < 0 || i > n {
                    return Err(SeqOpError::new(
                        "SequenceInsert",
                        format!(
                            "position {p} is out of bounds for inserting into a sequence \
                             of length {len} (valid range is [{}, {}]; negative values \
                             count from the end). To fix: pass an in-range index or omit \
                             it to append",
                            -n, n
                        ),
                    ));
                }
                i as usize
            }
        };
        let mut items = self.items.clone(); // Arc clones only — no bytes copied.
        items.insert(idx, tensor);
        Ok(SequenceValue {
            elem_dtype: self.elem_dtype,
            items,
        })
    }

    /// `SequenceErase`: a **new** sequence with the element at `position`
    /// (default = last) removed. The returned sequence **shares** every
    /// surviving element `Arc` — no bytes copied.
    pub(crate) fn erase(&self, position: Option<i64>) -> SeqResult<SequenceValue> {
        if self.is_empty() {
            return Err(SeqOpError::new(
                "SequenceErase",
                "cannot erase from an empty sequence. To fix: guard with \
                 SequenceLength before erasing",
            ));
        }
        let idx = match position {
            None => self.len() - 1,
            Some(p) => resolve_index("SequenceErase", p, self.len())?,
        };
        let mut items = self.items.clone(); // Arc clones only — no bytes copied.
        items.remove(idx);
        Ok(SequenceValue {
            elem_dtype: self.elem_dtype,
            items,
        })
    }

    /// `SequenceAt`: a **shared handle** to the element at `position` (negative
    /// allowed). Returns a clone of the element `Arc` — the exact same
    /// allocation that was inserted, with **no deep copy**.
    pub(crate) fn at(&self, position: i64) -> SeqResult<Arc<SeqTensor>> {
        let idx = resolve_index("SequenceAt", position, self.len())?;
        Ok(Arc::clone(&self.items[idx]))
    }

    /// Whether the sequence has no elements.
    pub(crate) fn is_empty(&self) -> bool {
        self.items.is_empty()
    }
}

/// Split a contiguous row-major tensor's bytes into `chunks` sub-tensors along
/// `axis`, one output byte buffer per chunk. `sizes` gives each chunk's extent
/// along `axis` (they sum to `shape[axis]`). Element bytes are **copied once**
/// into each freshly-allocated output (a single-alloc slice — the memory the
/// element owns as a shared sequence item). Returns `(out_shape, out_bytes)`
/// per chunk. Callers exclude sub-byte dtypes (esize ≥ 1).
pub(crate) fn split_axis(
    data: &[u8],
    shape: &[usize],
    axis: usize,
    sizes: &[usize],
    esize: usize,
) -> Vec<(Vec<usize>, Vec<u8>)> {
    let outer: usize = shape[..axis].iter().product();
    let inner: usize = shape[axis + 1..].iter().product::<usize>() * esize;
    let axis_dim = shape[axis];
    let mut out = Vec::with_capacity(sizes.len());
    let mut start = 0usize;
    for &k in sizes {
        let mut buf = vec![0u8; outer * k * inner];
        for o in 0..outer {
            let src_off = (o * axis_dim + start) * inner;
            let dst_off = o * k * inner;
            buf[dst_off..dst_off + k * inner]
                .copy_from_slice(&data[src_off..src_off + k * inner]);
        }
        let mut oshape = shape.to_vec();
        oshape[axis] = k;
        out.push((oshape, buf));
        start += k;
    }
    out
}

/// `ConcatFromSequence` with `new_axis = 0`: concatenate `elements` (each a
/// contiguous row-major byte buffer with the matching `shapes[i]`) along an
/// existing `axis` into one freshly-allocated output. This necessarily
/// allocates the output once and memcpies each element in exactly once (no
/// redundant copies — the single-alloc Concat pattern). Returns
/// `(out_shape, out_bytes)`.
pub(crate) fn concat_axis(
    elements: &[&[u8]],
    shapes: &[Vec<usize>],
    axis: usize,
    esize: usize,
) -> (Vec<usize>, Vec<u8>) {
    let base = &shapes[0];
    let outer: usize = base[..axis].iter().product();
    let inner: usize = base[axis + 1..].iter().product::<usize>() * esize;
    let total_axis: usize = shapes.iter().map(|s| s[axis]).sum();
    let mut oshape = base.clone();
    oshape[axis] = total_axis;
    let mut buf = vec![0u8; outer * total_axis * inner];
    for o in 0..outer {
        let mut axis_cursor = 0usize;
        for (e, src) in elements.iter().enumerate() {
            let k = shapes[e][axis];
            let src_off = o * k * inner;
            let dst_off = (o * total_axis + axis_cursor) * inner;
            buf[dst_off..dst_off + k * inner]
                .copy_from_slice(&src[src_off..src_off + k * inner]);
            axis_cursor += k;
        }
    }
    (oshape, buf)
}

/// `ConcatFromSequence` with `new_axis = 1`: **stack** `elements` (all sharing
/// `elem_shape`) along a brand-new axis inserted at `axis`, into one freshly
/// allocated output. Single alloc, one memcpy per element. Returns
/// `(out_shape, out_bytes)`.
pub(crate) fn stack_new_axis(
    elements: &[&[u8]],
    elem_shape: &[usize],
    axis: usize,
    esize: usize,
) -> (Vec<usize>, Vec<u8>) {
    let n = elements.len();
    let outer: usize = elem_shape[..axis].iter().product();
    let inner: usize = elem_shape[axis..].iter().product::<usize>() * esize;
    let mut oshape = Vec::with_capacity(elem_shape.len() + 1);
    oshape.extend_from_slice(&elem_shape[..axis]);
    oshape.push(n);
    oshape.extend_from_slice(&elem_shape[axis..]);
    let mut buf = vec![0u8; n * outer * inner];
    for (j, src) in elements.iter().enumerate() {
        for o in 0..outer {
            let src_off = o * inner;
            let dst_off = (o * n + j) * inner;
            buf[dst_off..dst_off + inner].copy_from_slice(&src[src_off..src_off + inner]);
        }
    }
    (oshape, buf)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn elem(dtype: DataType, shape: &[usize], data: &[u8]) -> Arc<SeqTensor> {
        SeqTensor::shared(dtype, shape.to_vec(), data.to_vec())
    }

    #[test]
    fn construct_len_and_dtype() {
        let s = SequenceValue::construct(vec![
            elem(DataType::Float32, &[2], &[0; 8]),
            elem(DataType::Float32, &[2], &[1; 8]),
        ])
        .unwrap();
        assert_eq!(s.len(), 2);
        assert_eq!(s.elem_dtype, DataType::Float32);
    }

    #[test]
    fn construct_rejects_mixed_dtype_actionably() {
        let err = SequenceValue::construct(vec![
            elem(DataType::Float32, &[1], &[0; 4]),
            elem(DataType::Int64, &[1], &[0; 8]),
        ])
        .unwrap_err();
        assert_eq!(err.op, "SequenceConstruct");
        assert!(err.reason.contains("homogeneous"));
        assert!(err.reason.contains("To fix"));
    }

    /// The core no-copy proof: `at` hands back the *same allocation* that was
    /// inserted — `Arc::ptr_eq` holds and the data pointer is identical. No
    /// intervening sequence op (construct → insert → erase) copied the bytes.
    #[test]
    fn at_returns_shared_handle_no_copy() {
        let a = elem(DataType::Float32, &[3], &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
        let a_ptr = a.as_ptr();
        let s0 = SequenceValue::construct(vec![Arc::clone(&a)]).unwrap();
        // Insert another element in front; `a` is now at index 1.
        let b = elem(DataType::Float32, &[3], &[0; 12]);
        let s1 = s0.insert(b, Some(0)).unwrap();
        assert_eq!(s1.len(), 2);
        let got = s1.at(1).unwrap();
        // Same Arc allocation and same byte address → zero deep copy.
        assert!(Arc::ptr_eq(&a, &got));
        assert_eq!(got.as_ptr(), a_ptr);
        // Erasing front still shares `a` (now index 0), pointer unchanged.
        let s2 = s1.erase(Some(0)).unwrap();
        let got2 = s2.at(-1).unwrap();
        assert!(Arc::ptr_eq(&a, &got2));
        assert_eq!(got2.as_ptr(), a_ptr);
    }

    /// Mutating ops share element `Arc`s with the source: the strong count rises
    /// by exactly the number of sequences referencing an element — proof that
    /// `insert`/`erase` clone handles, not bytes.
    #[test]
    fn mutations_share_arcs_not_bytes() {
        let a = elem(DataType::Float32, &[1], &[0; 4]);
        assert_eq!(Arc::strong_count(&a), 1);
        let s0 = SequenceValue::construct(vec![Arc::clone(&a)]).unwrap();
        assert_eq!(Arc::strong_count(&a), 2); // a + s0
        let s1 = s0.insert(elem(DataType::Float32, &[1], &[9; 4]), None).unwrap();
        assert_eq!(Arc::strong_count(&a), 3); // a + s0 + s1 (shared, not copied)
        drop(s0);
        assert_eq!(Arc::strong_count(&a), 2); // a + s1
        let _ = s1;
    }

    #[test]
    fn insert_positions_and_default_append() {
        let mk = |v: u8| elem(DataType::Uint8, &[1], &[v]);
        let s = SequenceValue::empty(DataType::Uint8);
        let s = s.insert(mk(1), None).unwrap(); // [1]
        let s = s.insert(mk(2), None).unwrap(); // [1,2]
        let s = s.insert(mk(0), Some(0)).unwrap(); // [0,1,2]
        let s = s.insert(mk(9), Some(-1)).unwrap(); // insert before last -> [0,1,9,2]
        let vals: Vec<u8> = (0..s.len() as i64)
            .map(|i| s.at(i).unwrap().data[0])
            .collect();
        assert_eq!(vals, vec![0, 1, 9, 2]);
    }

    #[test]
    fn erase_default_last_and_indexed() {
        let mk = |v: u8| elem(DataType::Uint8, &[1], &[v]);
        let s = SequenceValue::construct(vec![mk(1), mk(2), mk(3)]).unwrap();
        let s = s.erase(None).unwrap(); // remove last -> [1,2]
        assert_eq!(s.len(), 2);
        let s = s.erase(Some(0)).unwrap(); // remove first -> [2]
        assert_eq!(s.at(0).unwrap().data[0], 2);
    }

    #[test]
    fn at_out_of_bounds_is_actionable() {
        let s = SequenceValue::construct(vec![elem(DataType::Uint8, &[1], &[7])]).unwrap();
        let err = s.at(5).unwrap_err();
        assert_eq!(err.op, "SequenceAt");
        assert!(err.reason.contains("out of bounds"));
        assert!(err.reason.contains("valid range"));
    }

    #[test]
    fn insert_dtype_mismatch_is_actionable() {
        let s = SequenceValue::construct(vec![elem(DataType::Float32, &[1], &[0; 4])]).unwrap();
        let err = s
            .insert(elem(DataType::Int64, &[1], &[0; 8]), None)
            .unwrap_err();
        assert_eq!(err.op, "SequenceInsert");
        assert!(err.reason.contains("does not match"));
    }

    #[test]
    fn empty_sequence_insert_dtype_mismatch_is_actionable() {
        let s = SequenceValue::empty(DataType::Float32);
        let err = s
            .insert(elem(DataType::Int64, &[1], &[0; 8]), None)
            .unwrap_err();
        assert_eq!(err.op, "SequenceInsert");
        assert!(err.reason.contains("does not match"));
        assert!(err.reason.contains("To fix"));
    }

    #[test]
    fn split_even_along_axis0() {
        // shape [4,2] f32-as-u8 (esize 4): split into 4 rows of size 1.
        let data: Vec<u8> = (0..(4 * 2 * 4) as u8).collect();
        let parts = split_axis(&data, &[4, 2], 0, &[1, 1, 1, 1], 4);
        assert_eq!(parts.len(), 4);
        assert_eq!(parts[0].0, vec![1, 2]);
        assert_eq!(parts[0].1, data[0..8]);
        assert_eq!(parts[3].1, data[24..32]);
    }

    #[test]
    fn split_uneven_along_axis1() {
        // shape [2,3] esize 1: split axis 1 into sizes [1,2].
        let data: Vec<u8> = vec![0, 1, 2, 3, 4, 5];
        let parts = split_axis(&data, &[2, 3], 1, &[1, 2], 1);
        assert_eq!(parts[0].0, vec![2, 1]);
        assert_eq!(parts[0].1, vec![0, 3]); // column 0
        assert_eq!(parts[1].0, vec![2, 2]);
        assert_eq!(parts[1].1, vec![1, 2, 4, 5]); // columns 1,2
    }

    #[test]
    fn concat_existing_axis_roundtrips_split() {
        let data: Vec<u8> = vec![0, 1, 2, 3, 4, 5];
        let parts = split_axis(&data, &[2, 3], 1, &[1, 2], 1);
        let refs: Vec<&[u8]> = parts.iter().map(|(_, b)| b.as_slice()).collect();
        let shapes: Vec<Vec<usize>> = parts.iter().map(|(s, _)| s.clone()).collect();
        let (oshape, out) = concat_axis(&refs, &shapes, 1, 1);
        assert_eq!(oshape, vec![2, 3]);
        assert_eq!(out, data);
    }

    #[test]
    fn stack_new_axis_front() {
        // two [2] elements stacked at new axis 0 -> [2,2].
        let a: Vec<u8> = vec![1, 2];
        let b: Vec<u8> = vec![3, 4];
        let (oshape, out) = stack_new_axis(&[&a, &b], &[2], 0, 1);
        assert_eq!(oshape, vec![2, 2]);
        assert_eq!(out, vec![1, 2, 3, 4]);
    }

    #[test]
    fn stack_new_axis_back_interleaves() {
        // two [2] elements stacked at new axis 1 -> [2,2] interleaved.
        let a: Vec<u8> = vec![1, 2];
        let b: Vec<u8> = vec![3, 4];
        let (oshape, out) = stack_new_axis(&[&a, &b], &[2], 1, 1);
        assert_eq!(oshape, vec![2, 2]);
        assert_eq!(out, vec![1, 3, 2, 4]);
    }

    /// Concurrency smoke test: many threads read the same shared sequence and
    /// its elements at once. Immutable `Arc` elements → no data race; correct
    /// reads under contention prove the shared-read-only design.
    #[test]
    fn concurrent_readers_no_race() {
        use std::thread;
        let s = SequenceValue::construct(vec![
            elem(DataType::Int32, &[1], &10i32.to_le_bytes()),
            elem(DataType::Int32, &[1], &20i32.to_le_bytes()),
            elem(DataType::Int32, &[1], &30i32.to_le_bytes()),
        ])
        .unwrap();
        let shared = Arc::new(s);
        let mut handles = Vec::new();
        for _ in 0..8 {
            let seq = Arc::clone(&shared);
            handles.push(thread::spawn(move || {
                let mut acc = 0i32;
                for _ in 0..1000 {
                    for i in 0..seq.len() as i64 {
                        let e = seq.at(i).unwrap();
                        acc += i32::from_le_bytes(e.data[..4].try_into().unwrap());
                    }
                }
                acc
            }));
        }
        for h in handles {
            assert_eq!(h.join().unwrap(), 60 * 1000);
        }
    }

    #[test]
    fn sequence_value_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<SequenceValue>();
        assert_send_sync::<Arc<SeqTensor>>();
    }
}
