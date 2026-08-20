//! A shape/stride vector that does not call the allocator for ordinary ranks.
//!
//! Every input of every `Run` gets a shape and a stride list built for it, and
//! both were `Vec`s. On the dispatch grid that is three allocations per input
//! per `Run` — `dims`, `shape`, `strides` — and at depth 1, where the fixed
//! per-`Run` cost is the whole story, the C allocator was measured at **9,089
//! instructions per `Run` against ORT's 5,500**. Our excess was almost exactly
//! the Rust side of that number. These lists are tiny and short-lived: a rank-4
//! shape is 32 bytes that lives for one kernel call, and `malloc` charges more
//! to hand it over than the kernel spends reading it.
//!
//! [`DimVec`] keeps up to [`INLINE_RANK`] elements in the value itself and
//! moves to a `Vec` beyond that, so ordinary tensors cost nothing and unusual
//! ones keep working. Rank 8 covers ONNX in practice — NCHW is 4, attention
//! layouts reach 5 or 6 — but nothing here *assumes* it: the spilled
//! representation is a plain `Vec` with no length ceiling, and the type is
//! written so the two representations are indistinguishable to every reader.
//!
//! There is no `unsafe` in this module. `T` is `Copy + Default`, so the inline
//! array can simply be zero-filled and the unused tail is a normal value that
//! is never read rather than uninitialised memory that must not be.
//!
//! # Invariants
//!
//! * `Inline` holds `len <= INLINE_RANK`; elements past `len` are `T::default()`
//!   and are not part of the slice.
//! * `Heap` may hold *any* length, including one that would fit inline. A
//!   `DimVec` that spilled and then shrank stays spilled — the representation
//!   is not canonical, so no reader may depend on it. Every comparison,
//!   iteration and hash below goes through the slice for that reason.

use std::ops::{Deref, DerefMut};

/// Ranks up to this are free; beyond it a `DimVec` spills to the heap.
///
/// Sized from what ONNX actually produces rather than from a round number:
/// image models are rank 4, batched attention is rank 5 or 6, and the tail
/// beyond that is rare enough that paying `malloc` for it is the right trade.
pub(crate) const INLINE_RANK: usize = 8;

/// A small vector for shapes and strides.
///
/// See the module docs for why this exists and what it does not promise.
#[derive(Clone)]
pub(crate) enum DimVec<T: Copy + Default> {
    /// Up to [`INLINE_RANK`] elements, stored in place.
    Inline { buf: [T; INLINE_RANK], len: usize },
    /// Any number of elements, on the heap.
    Heap(Vec<T>),
}

impl<T: Copy + Default> DimVec<T> {
    /// An empty vector, allocation-free.
    #[inline]
    pub(crate) fn new() -> Self {
        Self::Inline {
            buf: [T::default(); INLINE_RANK],
            len: 0,
        }
    }

    /// An empty vector that will not need to spill for `cap` elements.
    ///
    /// Allocation-free when `cap` fits inline, which is the point: callers
    /// that know the rank up front get the fast representation directly
    /// instead of discovering it one `push` at a time.
    #[inline]
    pub(crate) fn with_capacity(cap: usize) -> Self {
        if cap <= INLINE_RANK {
            Self::new()
        } else {
            Self::Heap(Vec::with_capacity(cap))
        }
    }

    /// Copies a slice, inline when it fits.
    #[inline]
    pub(crate) fn from_slice(src: &[T]) -> Self {
        if src.len() <= INLINE_RANK {
            let mut buf = [T::default(); INLINE_RANK];
            buf[..src.len()].copy_from_slice(src);
            Self::Inline {
                buf,
                len: src.len(),
            }
        } else {
            Self::Heap(src.to_vec())
        }
    }

    /// `len` copies of `T::default()` — zero for the integer dims and strides
    /// this holds.
    ///
    /// This is the bulk path for builders that immediately overwrite every
    /// element. It exists instead of a `filled(len, value)` because the inline
    /// array is already `T::default()`-initialised by construction, so seeding
    /// it with a caller-chosen value costs a second pass over the buffer that
    /// those builders then throw away. Building one `push` at a time is worse
    /// still: it re-matches the representation on every element for a length
    /// that is known up front.
    #[inline]
    pub(crate) fn zeroed(len: usize) -> Self {
        if len <= INLINE_RANK {
            Self::Inline {
                buf: [T::default(); INLINE_RANK],
                len,
            }
        } else {
            Self::Heap(vec![T::default(); len])
        }
    }

    /// Appends one element, spilling to the heap if the inline space is full.
    #[inline]
    pub(crate) fn push(&mut self, value: T) {
        match self {
            Self::Inline { buf, len } if *len < INLINE_RANK => {
                buf[*len] = value;
                *len += 1;
            }
            Self::Inline { buf, len } => {
                // Full: move to the heap, keeping order. Reserve one past the
                // current need so a rank-9 tensor does not immediately grow
                // again on the next push.
                let mut heap = Vec::with_capacity(INLINE_RANK * 2);
                heap.extend_from_slice(&buf[..*len]);
                heap.push(value);
                *self = Self::Heap(heap);
            }
            Self::Heap(v) => v.push(value),
        }
    }

    /// The elements, as a slice. The single point where the two
    /// representations become one.
    #[inline]
    pub(crate) fn as_slice(&self) -> &[T] {
        match self {
            Self::Inline { buf, len } => &buf[..*len],
            Self::Heap(v) => v.as_slice(),
        }
    }

    #[inline]
    fn as_mut_slice(&mut self) -> &mut [T] {
        match self {
            Self::Inline { buf, len } => &mut buf[..*len],
            Self::Heap(v) => v.as_mut_slice(),
        }
    }

    /// Whether this value is currently using the heap.
    ///
    /// For tests and allocation accounting only. No behaviour may branch on
    /// it: see the representation invariant in the module docs.
    #[cfg(test)]
    pub(crate) const fn is_spilled(&self) -> bool {
        matches!(self, Self::Heap(_))
    }
}

impl<T: Copy + Default> Default for DimVec<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Copy + Default> Deref for DimVec<T> {
    type Target = [T];
    #[inline]
    fn deref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T: Copy + Default> DerefMut for DimVec<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

impl<T: Copy + Default> FromIterator<T> for DimVec<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let it = iter.into_iter();
        // `size_hint` is a hint, not a promise: `push` handles the case where
        // it was wrong in either direction.
        let mut out = Self::with_capacity(it.size_hint().0);
        for v in it {
            out.push(v);
        }
        out
    }
}

impl<T: Copy + Default> From<Vec<T>> for DimVec<T> {
    /// Takes the `Vec` as-is rather than copying it back inline: the caller
    /// already paid for the allocation, and re-inlining would only add work.
    fn from(v: Vec<T>) -> Self {
        Self::Heap(v)
    }
}

impl<T: Copy + Default> From<&[T]> for DimVec<T> {
    fn from(v: &[T]) -> Self {
        Self::from_slice(v)
    }
}

impl<T: Copy + Default, const N: usize> From<[T; N]> for DimVec<T> {
    fn from(v: [T; N]) -> Self {
        Self::from_slice(&v)
    }
}

// Comparison, hashing and formatting all go through the slice, so a spilled
// value and an inline value with the same elements are indistinguishable.
impl<T: Copy + Default + PartialEq> PartialEq for DimVec<T> {
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl<T: Copy + Default + Eq> Eq for DimVec<T> {}

impl<T: Copy + Default + PartialEq> PartialEq<[T]> for DimVec<T> {
    fn eq(&self, other: &[T]) -> bool {
        self.as_slice() == other
    }
}

impl<T: Copy + Default + PartialEq> PartialEq<&[T]> for DimVec<T> {
    fn eq(&self, other: &&[T]) -> bool {
        self.as_slice() == *other
    }
}

impl<T: Copy + Default + PartialEq> PartialEq<Vec<T>> for DimVec<T> {
    fn eq(&self, other: &Vec<T>) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl<T: Copy + Default + std::hash::Hash> std::hash::Hash for DimVec<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.as_slice().hash(state);
    }
}

impl<T: Copy + Default + std::fmt::Debug> std::fmt::Debug for DimVec<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Prints as a list, not as an enum: error messages that embed a shape
        // must not leak which representation it happened to be in.
        std::fmt::Debug::fmt(self.as_slice(), f)
    }
}

impl<'a, T: Copy + Default> IntoIterator for &'a DimVec<T> {
    type Item = &'a T;
    type IntoIter = std::slice::Iter<'a, T>;
    fn into_iter(self) -> Self::IntoIter {
        self.as_slice().iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_empty_vec_is_inline_and_empty() {
        let v = DimVec::<usize>::new();
        assert!(!v.is_spilled());
        assert!(v.is_empty());
        assert_eq!(v.as_slice(), &[] as &[usize]);
    }

    /// The whole point of the type: an ordinary rank must not reach the heap.
    #[test]
    fn ordinary_ranks_stay_inline() {
        for rank in 0..=INLINE_RANK {
            let v: DimVec<usize> = (0..rank).collect();
            assert!(!v.is_spilled(), "rank {rank} spilled but should fit inline");
            assert_eq!(v.len(), rank);
            assert!(v.iter().copied().eq(0..rank), "rank {rank} lost its order");
        }
    }

    /// Rank > 8 is not a supported-ranks question, it is a representation
    /// question: it must keep working, in order, with every element intact.
    #[test]
    fn ranks_past_the_inline_limit_spill_and_keep_every_element() {
        for rank in [INLINE_RANK + 1, INLINE_RANK + 2, 64, 1000] {
            let v: DimVec<usize> = (0..rank).collect();
            assert!(v.is_spilled(), "rank {rank} should have spilled");
            assert_eq!(v.len(), rank);
            assert!(v.iter().copied().eq(0..rank), "rank {rank} lost its order");
        }
    }

    /// The spill happens mid-`push`, which is where an off-by-one would live.
    #[test]
    fn pushing_across_the_boundary_preserves_order() {
        let mut v = DimVec::<i64>::new();
        for i in 0..(INLINE_RANK as i64 + 3) {
            v.push(i * 7);
            assert_eq!(
                v.len(),
                i as usize + 1,
                "length wrong right after pushing {i}"
            );
            assert!(
                v.iter().copied().eq((0..=i).map(|k| k * 7)),
                "contents wrong right after pushing {i}"
            );
        }
        assert!(v.is_spilled());
    }

    /// `zeroed` is the bulk path the stride and dims builders use; it must
    /// agree with the `push` loop it replaced at every length, including across
    /// the spill boundary and at zero.
    #[test]
    fn zeroed_agrees_with_pushing_one_at_a_time() {
        for len in 0..(INLINE_RANK + 4) {
            let bulk = DimVec::<i64>::zeroed(len);
            let mut one_at_a_time = DimVec::<i64>::with_capacity(len);
            for _ in 0..len {
                one_at_a_time.push(0i64);
            }
            assert_eq!(bulk, one_at_a_time, "len {len}");
            assert_eq!(bulk.len(), len, "len {len}");
            assert_eq!(
                bulk.is_spilled(),
                len > INLINE_RANK,
                "len {len} took the wrong representation"
            );
        }
    }

    /// Zero-sized dimensions are ordinary values here, not a special case, and
    /// a shape of all zeros must not be confused with an empty shape.
    #[test]
    fn zero_dims_are_values_not_absence() {
        let zeros = DimVec::from_slice(&[0usize, 0, 0]);
        assert_eq!(zeros.len(), 3);
        assert!(!zeros.is_empty());
        assert_ne!(zeros, DimVec::<usize>::new());
    }

    /// Equality is by contents, so the representation can never leak into
    /// behaviour. A cache keyed on a shape must not miss because one side
    /// happened to spill.
    #[test]
    fn a_spilled_value_equals_an_inline_value_with_the_same_contents() {
        let inline = DimVec::from_slice(&[1usize, 2, 3]);
        let spilled = DimVec::from(vec![1usize, 2, 3]);
        assert!(!inline.is_spilled());
        assert!(spilled.is_spilled(), "From<Vec> must keep the caller's Vec");
        assert_eq!(inline, spilled);
        assert_eq!(spilled, inline);
        assert_eq!(format!("{inline:?}"), format!("{spilled:?}"));

        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let h = |v: &DimVec<usize>| {
            let mut s = DefaultHasher::new();
            v.hash(&mut s);
            s.finish()
        };
        assert_eq!(h(&inline), h(&spilled), "hash must follow equality");
    }

    /// `with_capacity` is the hint path; getting it wrong in either direction
    /// must stay correct, only slower.
    #[test]
    fn with_capacity_is_only_a_hint() {
        let mut small = DimVec::<usize>::with_capacity(0);
        for i in 0..20 {
            small.push(i);
        }
        assert!(small.iter().copied().eq(0..20));

        let mut large = DimVec::<usize>::with_capacity(100);
        assert!(large.is_spilled());
        large.push(1);
        assert_eq!(large.as_slice(), &[1]);
    }

    /// Mutation through `DerefMut` must reach the real storage in both
    /// representations, not a copy of it.
    #[test]
    fn mutation_reaches_both_representations() {
        let mut inline = DimVec::from_slice(&[1usize, 2, 3]);
        inline[1] = 99;
        assert_eq!(inline.as_slice(), &[1, 99, 3]);

        let mut spilled: DimVec<usize> = (0..12).collect();
        spilled[11] = 42;
        assert_eq!(spilled[11], 42);
    }

    /// Elements past `len` exist but must never be observable.
    #[test]
    fn the_inline_tail_is_not_part_of_the_value() {
        let mut v = DimVec::<usize>::new();
        v.push(5);
        v.push(6);
        assert_eq!(v.as_slice(), &[5, 6]);
        assert_eq!(v.iter().count(), 2);
        assert_eq!(v.last(), Some(&6));
    }

    /// A `DimVec` is passed by value through the dispatch path, so its size
    /// matters: it must be small enough to move cheaply, and it must not have
    /// silently grown past what a `Vec` plus a small array needs.
    #[test]
    fn the_value_stays_small_enough_to_move() {
        assert!(
            size_of::<DimVec<usize>>() <= (INLINE_RANK + 2) * size_of::<usize>(),
            "DimVec<usize> is {} bytes",
            size_of::<DimVec<usize>>()
        );
    }
}
