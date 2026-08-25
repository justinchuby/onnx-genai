//! # `onnx-runtime-virtual-memory`
//!
//! Virtually contiguous, physically scattered memory.
//!
//! ## The problem this solves
//!
//! Paged KV storage and attention operators want opposite things. Paging wants
//! small, individually reclaimable, individually migratable blocks. A
//! `GroupQueryAttention` kernel wants one flat buffer per layer, because that is
//! what the ONNX graph declares its `past_key`/`past_value` inputs to be.
//!
//! The usual reconciliations are to copy the pages into a contiguous staging
//! buffer every step, or to change the model graph so the operator understands
//! block tables. The first costs a full KV copy per decode step; the second only
//! works for models we control the export of.
//!
//! There is a third option: **reserve a contiguous range of virtual addresses
//! and map physically separate blocks into it**. The operator sees one flat
//! buffer and runs unmodified. The blocks stay individually reclaimable, and
//! growing the range costs a mapping call rather than a copy.
//!
//! ## What decides whether this is cheap
//!
//! Mapping granularity, which is a platform and device property rather than
//! something this crate chooses. Measured on a Windows host with an RTX 4060:
//!
//! | mapping | granularity | tokens per granule, 8B GQA (2048 B/token) |
//! |---|---|---|
//! | Windows host | 64 KiB | 32 |
//! | Linux / macOS host | page size, 4 KiB or 16 KiB | 2 to 8 |
//! | CUDA VMM | 2 MiB | 1024 |
//!
//! On the host that is as fine as a KV page, so virtual contiguity costs nothing
//! in wasted memory. On CUDA a sequence rounds up to `num_layers * 2 * 2 MiB`
//! whatever its length, which only matters with many concurrent short
//! sequences. See #596 for why that trade was accepted.
//!
//! ## Apple Silicon
//!
//! macOS uses the same `mmap` path as Linux. `MAP_NORESERVE` is defined there
//! but is effectively ignored; it is an accounting hint, not a correctness
//! requirement, so the reservation behaves the same. Note that Apple Silicon
//! pages are **16 KiB**, not 4 KiB, so [`granularity`] must be queried rather
//! than assumed — a hard-coded 4096 would misalign every offset.
//!
//! The bigger Apple consequence is not in this crate: CPU and GPU share one
//! physical pool, so "device memory" and "host memory" are the same bytes.
//! Anything holding separate per-tier budgets will over-commit there unless it
//! knows the tiers alias.
//!
//! ## What this crate is not
//!
//! It does not decide *whether* memory may be held — that is
//! `onnx-runtime-memory-governor`. A [`VirtualRange`] is a mapping mechanism; a
//! lease is the permission to use one.

#![allow(unsafe_code)]

pub mod backing;
pub mod buffer;

pub use backing::{HostBacking, PhysicalMemoryAccounting, VirtualBacking};
pub use buffer::{VirtualBuffer, VirtualBufferError};

use std::ptr::NonNull;

mod sys;

/// Why a virtual range could not be reserved, mapped, or unmapped.
///
/// Not `Clone`/`PartialEq`: a refusal that carries the cause underneath it
/// cannot be meaningfully duplicated or compared, and keeping the cause is
/// worth more than either.
#[derive(Debug, thiserror::Error)]
pub enum VirtualMemoryError {
    /// The requested size or offset is not a multiple of the mapping
    /// granularity.
    ///
    /// Reported rather than rounded, because silently rounding an offset would
    /// place a block somewhere the caller did not ask for and every subsequent
    /// read would be of the wrong data.
    #[error(
        "{what} of {value} bytes is not a multiple of the {granularity} byte mapping \
         granularity; round it up to {rounded} or ask this platform for its granularity via \
         `granularity()` before splitting a buffer"
    )]
    Misaligned {
        /// Which quantity was misaligned.
        what: &'static str,
        /// The value supplied.
        value: usize,
        /// This platform's granularity.
        granularity: usize,
        /// The next legal value at or above `value`.
        rounded: usize,
    },
    /// A layer this call delegated to refused the request.
    ///
    /// Distinct from [`VirtualMemoryError::Os`], which means the *kernel*
    /// refused it. Keeping the two apart matters because they call for
    /// opposite responses: a governor that declines a reservation is telling
    /// the caller to ask for less or free something first, while a driver that
    /// fails a mapping is telling it that the request cannot be served at all.
    /// Flattening a decision into an OS error also has to invent an `errno`,
    /// and `os error 0` in a log sends the next reader hunting for a kernel
    /// fault that never happened.
    ///
    /// The refusal is kept whole rather than stringified so callers can still
    /// match on it after it has crossed this boundary.
    #[error("{operation} failed: {source}")]
    Delegated {
        /// Which call was being made when the lower layer refused.
        operation: &'static str,
        /// The refusal itself, kept whole so it survives `downcast_ref`.
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },
    /// The operating system refused the request.
    #[error("{operation} failed: {reason} (os error {code})")]
    Os {
        /// Which call failed.
        operation: &'static str,
        /// What the OS reported.
        reason: String,
        /// The raw error code.
        code: i32,
    },
    /// A mapping would fall outside the reserved range.
    #[error(
        "cannot map {length} bytes at offset {offset} of a {reserved} byte range; the mapping \
         would run {overrun} bytes past the end. Reserve a larger range, or map at a lower offset"
    )]
    OutOfRange {
        /// Where the caller asked to map.
        offset: usize,
        /// How much.
        length: usize,
        /// The reservation's size.
        reserved: usize,
        /// How far past the end it would reach.
        overrun: usize,
    },
    /// The range is already mapped at that offset.
    ///
    /// Replacing a live mapping silently would leave the previous block
    /// allocated but unreachable, so it is refused.
    #[error("offset {offset} of this range is already mapped; unmap it before mapping again")]
    AlreadyMapped {
        /// The offset in question.
        offset: usize,
    },
    /// A caller-supplied `PhysicalLocation` (or platform-specific stand-in for
    /// it) did not match the physical backing a pool/handle actually holds.
    ///
    /// Rejected before any lease charge, handle acquisition, mapping, or
    /// accounting mutation happens: this is a caller-programming-error check,
    /// not a driver refusal, and must not have any side effect on the pool it
    /// was refused against.
    #[error(
        "requested location {requested} does not match this pool's backing location {actual}; \
         a mismatched location must never be silently accepted, ask the pool for its own \
         `location()` instead of asserting one"
    )]
    LocationMismatch {
        /// What the caller asked to commit against.
        requested: String,
        /// What the pool is actually backed by.
        actual: String,
    },
}

/// This platform's minimum mapping granularity, in bytes.
///
/// Every offset and length handed to [`VirtualRange`] must be a multiple of it.
pub fn granularity() -> usize {
    sys::granularity()
}

/// A reserved range of virtual addresses with nothing behind it yet.
///
/// Reserving costs address space, not memory. Blocks are mapped in afterwards
/// with [`VirtualRange::map`], and the range is readable only where something
/// has been mapped — touching an unmapped offset faults rather than reading
/// zeroes, which is deliberate: a silent zero would look like KV that was
/// written but empty.
#[derive(Debug)]
pub struct VirtualRange {
    base: NonNull<u8>,
    len: usize,
    /// Offsets currently backed by a block, each with its mapped length.
    mapped: Vec<(usize, usize)>,
}

// The pointer is an owned reservation; nothing in it is thread-affine.
unsafe impl Send for VirtualRange {}
unsafe impl Sync for VirtualRange {}

impl VirtualRange {
    /// Reserve `len` bytes of address space.
    ///
    /// `len` must be a multiple of [`granularity`].
    pub fn reserve(len: usize) -> Result<Self, VirtualMemoryError> {
        check_aligned("reservation length", len)?;
        if len == 0 {
            return Err(VirtualMemoryError::Misaligned {
                what: "reservation length",
                value: 0,
                granularity: granularity(),
                rounded: granularity(),
            });
        }
        let base = sys::reserve(len)?;
        Ok(Self {
            base,
            len,
            mapped: Vec::new(),
        })
    }

    /// Bytes of address space reserved.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the reservation covers no bytes.
    ///
    /// Always false: a zero-length reservation is refused at construction.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Bytes currently backed by a mapped block.
    pub fn mapped_bytes(&self) -> usize {
        self.mapped.iter().map(|&(_, len)| len).sum()
    }

    /// The base address. Only offsets that have been mapped may be read.
    pub fn as_ptr(&self) -> *const u8 {
        self.base.as_ptr()
    }

    /// The base address, mutable.
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.base.as_ptr()
    }

    /// Back `offset..offset + len` with freshly committed memory.
    ///
    /// Both `offset` and `len` must be multiples of [`granularity`].
    pub fn map(&mut self, offset: usize, len: usize) -> Result<(), VirtualMemoryError> {
        check_aligned("mapping offset", offset)?;
        check_aligned("mapping length", len)?;
        let end = offset.saturating_add(len);
        if end > self.len {
            return Err(VirtualMemoryError::OutOfRange {
                offset,
                length: len,
                reserved: self.len,
                overrun: end - self.len,
            });
        }
        if self.mapped.iter().any(|&(at, mapped_len)| {
            // Any overlap counts: partially replacing a mapping would strand the
            // rest of the block it belonged to.
            offset < at + mapped_len && at < end
        }) {
            return Err(VirtualMemoryError::AlreadyMapped { offset });
        }
        // SAFETY: the offset and length are aligned and within the reservation,
        // and the overlap check above proves nothing is mapped there yet.
        //
        // Windows needs to know whether this span is a strict subset of the
        // placeholder that contains it, which depends on the surrounding *free*
        // gap rather than on the whole reservation. Computing it here keeps the
        // platform layer free of the range's bookkeeping.
        let gap = self.free_gap_containing(offset);
        // SAFETY: as above.
        unsafe { sys::map(self.base, gap, offset, len)? };
        self.mapped.push((offset, len));
        Ok(())
    }

    /// The free span surrounding `offset`, as `(start, len)`.
    ///
    /// A reservation starts as one placeholder and is carved as blocks are
    /// mapped, so the span a new block must be split out of is bounded by its
    /// mapped neighbours, not by the reservation.
    fn free_gap_containing(&self, offset: usize) -> (usize, usize) {
        let mut start = 0;
        let mut end = self.len;
        for &(at, len) in &self.mapped {
            let block_end = at + len;
            if block_end <= offset {
                start = start.max(block_end);
            } else if at > offset {
                end = end.min(at);
            }
        }
        (start, end - start)
    }

    /// Release the block backing `offset`, leaving the address space reserved.
    ///
    /// The offset must be one previously passed to [`Self::map`]; releasing a
    /// sub-range would strand the remainder of the block.
    pub fn unmap(&mut self, offset: usize) -> Result<(), VirtualMemoryError> {
        let Some(index) = self.mapped.iter().position(|&(at, _)| at == offset) else {
            return Ok(());
        };
        let (_, len) = self.mapped[index];
        // SAFETY: this offset and length came from a successful `map`.
        unsafe { sys::unmap(self.base, offset, len)? };
        self.mapped.remove(index);
        Ok(())
    }

    /// A read-only view of a mapped span.
    ///
    /// # Safety
    ///
    /// The caller must ensure `offset..offset + len` is entirely mapped.
    /// Reading unmapped address space faults.
    pub unsafe fn slice_unchecked(&self, offset: usize, len: usize) -> &[u8] {
        // SAFETY: the caller guarantees the span is mapped, and the base pointer
        // is valid for the life of `self`.
        unsafe { std::slice::from_raw_parts(self.base.as_ptr().add(offset), len) }
    }

    /// A mutable view of a mapped span.
    ///
    /// # Safety
    ///
    /// The caller must ensure `offset..offset + len` is entirely mapped.
    pub unsafe fn slice_unchecked_mut(&mut self, offset: usize, len: usize) -> &mut [u8] {
        // SAFETY: as above, plus `&mut self` rules out aliasing views.
        unsafe { std::slice::from_raw_parts_mut(self.base.as_ptr().add(offset), len) }
    }
}

impl Drop for VirtualRange {
    fn drop(&mut self) {
        for &(offset, len) in &self.mapped {
            // SAFETY: every entry came from a successful `map`.
            let _ = unsafe { sys::unmap(self.base, offset, len) };
        }
        // SAFETY: `base` came from `sys::reserve` and is released exactly once.
        unsafe { sys::release(self.base, self.len) };
    }
}

fn check_aligned(what: &'static str, value: usize) -> Result<(), VirtualMemoryError> {
    let granularity = granularity();
    if value.is_multiple_of(granularity) {
        return Ok(());
    }
    Err(VirtualMemoryError::Misaligned {
        what,
        value,
        granularity,
        rounded: value.div_ceil(granularity) * granularity,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `Delegated` and `Os` are kept apart because they call for opposite
    /// responses, and flattening the first into the second has to invent an
    /// `errno` -- an `os error 0` in a log sends the next reader hunting for a
    /// kernel fault that never happened.
    #[test]
    fn a_delegated_refusal_is_reported_as_itself_not_as_an_os_error() {
        #[derive(Debug, thiserror::Error)]
        #[error("the tier is full")]
        struct TierFull;

        let error = VirtualMemoryError::Delegated {
            operation: "growing physical handle pool lease",
            source: Box::new(TierFull),
        };

        let rendered = error.to_string();
        assert_eq!(
            rendered,
            "growing physical handle pool lease failed: the tier is full"
        );
        assert!(
            !rendered.contains("os error"),
            "a refusal from a lower layer must not be dressed up as a kernel failure: {rendered}"
        );

        let cause = std::error::Error::source(&error).expect("the refusal must be reachable");
        assert!(
            cause.downcast_ref::<TierFull>().is_some(),
            "the cause must arrive as itself, not as a box around itself: {cause}"
        );
    }

    /// Two separately mapped blocks must read back as one flat buffer.
    ///
    /// This is the entire point of the crate: an operator that requires
    /// contiguity gets it, while the memory behind it stays in independently
    /// mappable pieces.
    #[test]
    fn separately_mapped_blocks_read_back_as_one_contiguous_buffer() {
        let g = granularity();
        let mut range = VirtualRange::reserve(g * 2).expect("two granules of address space");
        range.map(0, g).expect("first block maps");
        range.map(g, g).expect("second block maps");

        // SAFETY: both granules were just mapped.
        unsafe {
            let buffer = range.slice_unchecked_mut(0, g * 2);
            for (index, byte) in buffer.iter_mut().enumerate() {
                *byte = (index % 251) as u8;
            }
        }

        // SAFETY: still mapped.
        let read = unsafe { range.slice_unchecked(0, g * 2) };
        for (index, &byte) in read.iter().enumerate() {
            assert_eq!(
                byte,
                (index % 251) as u8,
                "byte {index} read back wrong across the block boundary"
            );
        }
        assert_eq!(range.mapped_bytes(), g * 2);
    }

    /// A write through the virtual range must survive unmapping its neighbour.
    ///
    /// If the two blocks were secretly one allocation, releasing one would
    /// disturb the other -- which would mean the pieces are not independently
    /// reclaimable and the whole premise fails.
    #[test]
    fn unmapping_one_block_leaves_its_neighbour_intact() {
        let g = granularity();
        let mut range = VirtualRange::reserve(g * 2).expect("address space");
        range.map(0, g).expect("first block");
        range.map(g, g).expect("second block");

        // SAFETY: mapped above.
        unsafe {
            range.slice_unchecked_mut(0, g)[0] = 0xAB;
            range.slice_unchecked_mut(g, g)[0] = 0xCD;
        }
        range.unmap(g).expect("second block releases");

        // SAFETY: the first block is still mapped.
        assert_eq!(unsafe { range.slice_unchecked(0, g) }[0], 0xAB);
        assert_eq!(range.mapped_bytes(), g);
    }

    /// Reserving costs address space, not memory.
    ///
    /// A reservation far larger than RAM must succeed, or growth would have to
    /// re-reserve and copy -- exactly what this crate exists to avoid.
    #[test]
    fn reserving_far_more_than_ram_succeeds_because_nothing_is_committed() {
        let g = granularity();
        let huge = g * 1024 * 64;
        let range = VirtualRange::reserve(huge).expect("address space is not memory");
        assert_eq!(range.len(), huge);
        assert_eq!(range.mapped_bytes(), 0, "reserving committed memory");
    }

    /// A misaligned request is refused with the value that would work.
    #[test]
    fn a_misaligned_offset_is_refused_and_names_the_next_legal_value() {
        let g = granularity();
        let mut range = VirtualRange::reserve(g * 2).expect("address space");
        let error = range
            .map(1, g)
            .expect_err("an offset of 1 cannot be a multiple of the granularity");
        match error {
            VirtualMemoryError::Misaligned { value, rounded, .. } => {
                assert_eq!(value, 1);
                assert_eq!(rounded, g, "the suggested value must itself be legal");
            }
            other => panic!("expected a misalignment error, got {other}"),
        }
    }

    /// Mapping past the end is refused rather than silently truncated.
    #[test]
    fn mapping_past_the_end_reports_how_far_it_overruns() {
        let g = granularity();
        let mut range = VirtualRange::reserve(g).expect("address space");
        let error = range.map(g, g).expect_err("offset g is already the end");
        assert!(
            matches!(error, VirtualMemoryError::OutOfRange { overrun, .. } if overrun == g),
            "expected a range error naming the overrun, got {error}"
        );
    }

    /// Mapping over a live block is refused.
    ///
    /// Silently replacing it would leave the previous block allocated but
    /// unreachable through this range.
    #[test]
    fn mapping_over_a_live_block_is_refused() {
        let g = granularity();
        let mut range = VirtualRange::reserve(g * 2).expect("address space");
        range
            .map(0, g * 2)
            .expect("one block covering both granules");
        let error = range
            .map(g, g)
            .expect_err("the second granule is inside the live block");
        assert!(
            matches!(error, VirtualMemoryError::AlreadyMapped { .. }),
            "expected an already-mapped error, got {error}"
        );
    }

    /// Unmapping an offset that was never mapped is not an error.
    ///
    /// Cleanup paths run without knowing what succeeded, and making them check
    /// first would just move the race.
    #[test]
    fn unmapping_an_unmapped_offset_is_a_no_op() {
        let g = granularity();
        let mut range = VirtualRange::reserve(g).expect("address space");
        range.unmap(0).expect("unmapping nothing is fine");
    }
    /// Granularity is queried, never assumed.
    ///
    /// Apple Silicon pages are 16 KiB and Windows carves at 64 KiB, so a
    /// hard-coded 4096 would misalign every offset on two of the three hosts
    /// this project targets. Asserting the property rather than a number is the
    /// only form of this test that can run everywhere.
    #[test]
    fn granularity_is_a_power_of_two_that_every_legal_offset_is_a_multiple_of() {
        let g = granularity();
        assert!(g > 0, "a zero granularity would divide by zero");
        assert!(
            g.is_power_of_two(),
            "granularity {g} is not a power of two, so alignment rounding is wrong"
        );
        // Whatever it is, a range of it must be reservable and mappable.
        let mut range = VirtualRange::reserve(g).expect("one granule");
        range.map(0, g).expect("one granule maps");
    }

    /// Rounding suggested by a misalignment error must itself be accepted.
    ///
    /// An error that names an unusable value is worse than one that names
    /// nothing, because a caller will follow it. Note the suggestion is about
    /// *alignment* only — the reservation still has to be large enough, which
    /// is why this reserves three granules to map one at the rounded offset.
    #[test]
    fn the_rounded_value_a_misalignment_error_suggests_is_itself_legal() {
        let g = granularity();
        let mut range = VirtualRange::reserve(g * 3).expect("address space");
        let VirtualMemoryError::Misaligned { rounded, .. } = range
            .map(g + 1, g)
            .expect_err("an offset one past a granule boundary is misaligned")
        else {
            panic!("expected a misalignment error");
        };
        range
            .map(rounded, g)
            .expect("the value the error suggested must be usable");
    }
    /// Blocks mapped out of order, with gaps, must each land correctly.
    ///
    /// This is where the platform split logic actually gets exercised: a block
    /// in the middle of free space has to be carved out of the placeholder
    /// containing it, and that placeholder is bounded by whatever was mapped
    /// before -- not by the reservation. Getting that wrong fails only for
    /// interior blocks, so the simple adjacent-blocks test does not catch it.
    #[test]
    fn blocks_mapped_out_of_order_with_gaps_each_land_at_their_own_offset() {
        let g = granularity();
        let mut range = VirtualRange::reserve(g * 5).expect("five granules");

        // Deliberately not in address order, and leaving holes.
        for (index, &slot) in [3usize, 0, 4].iter().enumerate() {
            range.map(slot * g, g).unwrap_or_else(|error| {
                panic!("mapping granule {slot} (step {index}) failed: {error}")
            });
            // SAFETY: just mapped.
            unsafe {
                range.slice_unchecked_mut(slot * g, g)[0] = 0x10 + slot as u8;
            }
        }

        for &slot in &[3usize, 0, 4] {
            // SAFETY: mapped above and never unmapped.
            let seen = unsafe { range.slice_unchecked(slot * g, g) }[0];
            assert_eq!(
                seen,
                0x10 + slot as u8,
                "granule {slot} read back another block's data"
            );
        }
        assert_eq!(range.mapped_bytes(), g * 3, "the holes were backed too");
    }
}
