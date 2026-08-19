//! A sequence that keeps a short run of elements in place.
//!
//! Every tensor crossing the kernel boundary carries a shape and a stride
//! vector, and both are rebuilt on **every** `Compute` call: ORT hands the
//! plugin raw `i64` dimensions per input per `Run`, and the output shape is
//! copied back out the same way. As `Vec`s that is four heap allocations for a
//! one-in/one-out elementwise node — two per input, two per output — to hold a
//! handful of integers whose entire lifetime is the call that made them.
//!
//! Real tensor ranks are tiny. ONNX permits any rank, but the shapes this EP
//! actually sees are 0-5 dimensions; rank 8 is already beyond anything in the
//! model zoo. So [`INLINE_RANK`] elements live in the value itself and only a
//! taller tensor reaches for the heap. The tall case is not a fallback in the
//! "shouldn't happen" sense — it is exercised by tests at ranks 9 and 12 and
//! behaves identically, just with an allocation.
//!
//! There is **no `unsafe` here**. `T: Copy + Default` means the inline buffer
//! can be created filled and the slots past `len` are ordinary initialised
//! values, never `MaybeUninit`. That is why this is a hand-rolled type rather
//! than a dependency: the useful part is 60 lines of safe code, and a shape
//! vector does not need a general-purpose small-vector crate's `unsafe`.

use std::ops::Deref;

/// Rank at or below which a shape or stride list stays off the heap.
///
/// Eight, because it is past every rank this EP has been handed (0-5 in
/// practice) while keeping [`InlineVec<usize, INLINE_RANK>`] at 80 bytes: the
/// 64-byte buffer, `len`, and a discriminant that cannot borrow the `Vec`
/// pointer's null niche because the array payload has no niche to share. That
/// buys a wider `OwnedInput` copied by value against four `malloc`/`free`
/// pairs removed per `Run`, which at ~3.5 us per small node is the better
/// trade.
pub(crate) const INLINE_RANK: usize = 8;

/// A `Vec`-like sequence that stores up to `N` elements inline.
///
/// Derefs to `[T]`, so anything taking `&[T]` — `TensorView::new`, ORT's
/// `KernelContext_GetOutput`, the shape-inference helpers — takes one of these
/// unchanged.
#[derive(Clone, Debug)]
pub(crate) enum InlineVec<T: Copy + Default, const N: usize> {
    /// Elements `[0, len)` of `buf` are live; the rest are `T::default()`.
    Inline { buf: [T; N], len: usize },
    /// Used when more than `N` elements were asked for or pushed.
    Heap(Vec<T>),
}

impl<T: Copy + Default, const N: usize> InlineVec<T, N> {
    /// An empty sequence, inline.
    pub(crate) fn new() -> Self {
        Self::Inline {
            buf: [T::default(); N],
            len: 0,
        }
    }

    /// An empty sequence that will hold `cap` elements without reallocating.
    ///
    /// Deciding once, here, is what keeps [`push`](Self::push) from having to
    /// spill in the hot path: every caller in this crate knows the rank before
    /// it starts.
    pub(crate) fn with_capacity(cap: usize) -> Self {
        if cap <= N {
            Self::new()
        } else {
            Self::Heap(Vec::with_capacity(cap))
        }
    }

    /// Append `value`, moving to the heap if the inline buffer is full.
    pub(crate) fn push(&mut self, value: T) {
        match self {
            Self::Inline { buf, len } => {
                if *len < N {
                    buf[*len] = value;
                    *len += 1;
                } else {
                    let mut heap = Vec::with_capacity(N * 2);
                    heap.extend_from_slice(&buf[..*len]);
                    heap.push(value);
                    *self = Self::Heap(heap);
                }
            }
            Self::Heap(heap) => heap.push(value),
        }
    }

    /// Copy `src` into a new sequence.
    pub(crate) fn from_slice(src: &[T]) -> Self {
        if src.len() <= N {
            let mut buf = [T::default(); N];
            buf[..src.len()].copy_from_slice(src);
            Self::Inline {
                buf,
                len: src.len(),
            }
        } else {
            Self::Heap(src.to_vec())
        }
    }

    /// The live elements.
    pub(crate) fn as_slice(&self) -> &[T] {
        match self {
            Self::Inline { buf, len } => &buf[..*len],
            Self::Heap(heap) => heap.as_slice(),
        }
    }

    /// Whether the elements are stored in the value rather than on the heap.
    ///
    /// Only tests consult this — production code goes through the `Deref`.
    /// It exists so a test can prove *which* representation it exercised,
    /// rather than passing on both and pinning neither.
    #[cfg(test)]
    pub(crate) fn is_inline(&self) -> bool {
        matches!(self, Self::Inline { .. })
    }
}

impl<T: Copy + Default, const N: usize> Deref for InlineVec<T, N> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T: Copy + Default + PartialEq, const N: usize> PartialEq for InlineVec<T, N> {
    /// Compares contents, not representation: the same dimensions inline and
    /// on the heap are the same shape.
    ///
    /// Deliberately not derived, and `Eq`/`Hash` are deliberately absent. If
    /// either is ever added it must hash and compare [`Self::as_slice`] for
    /// the same reason this does — a derived one would split the two
    /// representations of an equal value apart and break any map keyed on a
    /// shape.
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}

/// So a caller — in practice a test — can compare against a literal without
/// caring which representation the value happens to be in.
impl<T: Copy + Default + PartialEq, const N: usize> PartialEq<Vec<T>> for InlineVec<T, N> {
    fn eq(&self, other: &Vec<T>) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl<T: Copy + Default + PartialEq, const N: usize, const M: usize> PartialEq<[T; M]>
    for InlineVec<T, N>
{
    fn eq(&self, other: &[T; M]) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl<T: Copy + Default + PartialEq, const N: usize> PartialEq<&[T]> for InlineVec<T, N> {
    fn eq(&self, other: &&[T]) -> bool {
        self.as_slice() == *other
    }
}

/// Row-major strides for `shape`, without touching the heap at small ranks.
///
/// Same result as `onnx_runtime_ir::compute_contiguous_strides`, which this
/// replaces at the kernel boundary; `strides_match_the_ir_helper_at_every_rank`
/// below holds the two together.
pub(crate) fn contiguous_strides(shape: &[usize]) -> InlineVec<i64, INLINE_RANK> {
    let n = shape.len();
    let mut strides = InlineVec::<i64, INLINE_RANK>::with_capacity(n);
    for _ in 0..n {
        strides.push(1i64);
    }
    match &mut strides {
        InlineVec::Inline { buf, .. } => {
            for i in (0..n.saturating_sub(1)).rev() {
                buf[i] = buf[i + 1] * shape[i + 1] as i64;
            }
        }
        InlineVec::Heap(heap) => {
            for i in (0..n.saturating_sub(1)).rev() {
                heap[i] = heap[i + 1] * shape[i + 1] as i64;
            }
        }
    }
    strides
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pushing_past_the_inline_capacity_spills_and_keeps_every_element() {
        let mut v = InlineVec::<usize, 4>::new();
        for i in 0..4 {
            v.push(i);
            assert!(v.is_inline(), "{} elements should still be inline", i + 1);
        }
        v.push(4);
        assert!(!v.is_inline(), "the fifth element must move to the heap");
        // The spill must not lose or reorder what was already there. Dropping
        // the `extend_from_slice` in `push` leaves `[4]` here.
        assert_eq!(&*v, &[0, 1, 2, 3, 4]);
        v.push(5);
        assert_eq!(&*v, &[0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn with_capacity_decides_the_representation_up_front() {
        assert!(InlineVec::<usize, 8>::with_capacity(8).is_inline());
        assert!(!InlineVec::<usize, 8>::with_capacity(9).is_inline());
        // ...and a heap-backed one still behaves like a sequence.
        let mut v = InlineVec::<usize, 8>::with_capacity(9);
        for i in 0..9 {
            v.push(i);
        }
        assert_eq!(v.len(), 9);
        assert_eq!(v[8], 8);
    }

    #[test]
    fn from_slice_round_trips_on_both_sides_of_the_boundary() {
        for rank in [0usize, 1, 7, 8, 9, 12, 33] {
            let src: Vec<usize> = (0..rank).map(|i| i * 3 + 1).collect();
            let v = InlineVec::<usize, INLINE_RANK>::from_slice(&src);
            assert_eq!(&*v, &src[..], "rank {rank} content");
            assert_eq!(
                v.is_inline(),
                rank <= INLINE_RANK,
                "rank {rank} representation"
            );
        }
    }

    #[test]
    fn equality_ignores_where_the_elements_live() {
        let inline = InlineVec::<usize, 2>::from_slice(&[1, 2]);
        let mut heap = InlineVec::<usize, 2>::with_capacity(3);
        heap.push(1);
        heap.push(2);
        assert!(inline.is_inline() && !heap.is_inline());
        assert_eq!(inline, heap);
    }

    #[test]
    fn strides_match_the_ir_helper_at_every_rank() {
        // Includes the ranks either side of the inline/heap boundary, so a
        // divergence in the heap arm of `contiguous_strides` cannot hide.
        let shapes: &[&[usize]] = &[
            &[],
            &[5],
            &[2, 3],
            &[2, 3, 4],
            &[7, 1, 1, 3],
            &[2, 2, 2, 2, 2, 2, 2, 2],
            &[2, 2, 2, 2, 2, 2, 2, 2, 3],
            &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            &[4, 0, 3],
        ];
        for shape in shapes {
            let ours = contiguous_strides(shape);
            let theirs = onnx_runtime_ir::compute_contiguous_strides(shape);
            assert_eq!(&*ours, &theirs[..], "strides for {shape:?}");
            assert_eq!(
                ours.is_inline(),
                shape.len() <= INLINE_RANK,
                "representation for {shape:?}"
            );
        }
    }
}
