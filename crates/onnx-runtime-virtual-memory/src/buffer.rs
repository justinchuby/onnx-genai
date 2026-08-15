//! A buffer that grows without moving.
//!
//! # The problem this solves
//!
//! A KV cache grows a token at a time. The obvious implementation reallocates
//! and copies, and that costs three things beyond the copy itself:
//!
//! * **The address changes.** Anything holding the old pointer is wrong. A
//!   captured device graph recorded the old address, so the capture is dead and
//!   has to be retaken.
//! * **Peak memory doubles at the seam.** Old and new are live at once, so a
//!   grow can fail at a tier that is merely full rather than over-subscribed.
//! * **The copy is O(everything so far)**, paid on every growth step.
//!
//! Reserving address space is free — it costs no memory, only address bits,
//! and 64-bit processes have plenty. So reserve for the largest the buffer
//! could ever be, and commit physical pages behind it as it actually grows. The
//! base address is fixed at reservation and never moves again.
//!
//! # What it costs instead
//!
//! Growth rounds up to the platform's mapping granularity — 64 KiB on Windows,
//! a page on unix — so a buffer that grows by a hundred bytes commits a whole
//! granule. That is a real overhead on small buffers and irrelevant on the
//! large ones this exists for.
//!
//! # Leasing
//!
//! Only **committed** bytes are leased. Reserving is not an allocation and
//! charging for it would make a governor refuse a buffer that will never use
//! the address space it reserved — which is precisely the arrangement that
//! makes reserving generously safe.

use std::sync::Arc;

use onnx_runtime_memory_governor::{
    HolderId, MemoryError, MemoryGovernor, MemoryLease, MemoryRole, Tier,
};

use crate::VirtualMemoryError;
use crate::backing::{HostBacking, PhysicalMemoryAccounting, VirtualBacking};

/// A growable region whose base address never changes.
///
/// Created with a *capacity* — an upper bound on address space — and a length
/// that starts at zero. [`VirtualBuffer::grow_to`] commits pages;
/// [`VirtualBuffer::shrink_to`] gives them back. The pointer returned by
/// [`VirtualBuffer::as_ptr`] is the same for the buffer's whole life.
pub struct VirtualBuffer<B: VirtualBacking = HostBacking> {
    backing: B,
    reservation: B::Reservation,
    /// Address space reserved at construction. Recorded here because a
    /// reservation is opaque to this type -- only the backing knows its shape.
    capacity_bytes: usize,
    /// Bytes the caller has asked for, which may be less than what is committed
    /// because commitment rounds up to a granule.
    len: usize,
    /// Bytes actually backed by physical memory. Always a multiple of the
    /// granularity and always at least `len`.
    committed: usize,
    governor: Arc<dyn MemoryGovernor + Send + Sync>,
    tier: Tier,
    role: MemoryRole,
    holder: HolderId,
    physical_memory_accounting: PhysicalMemoryAccounting,
    /// Covers exactly `committed` bytes. `None` while nothing is committed,
    /// because a zero-byte lease is not a thing to hold.
    lease: Option<MemoryLease>,
}

/// What went wrong growing or shrinking a [`VirtualBuffer`].
#[derive(Debug, thiserror::Error)]
pub enum VirtualBufferError {
    /// The address space could not be reserved or mapped.
    #[error(transparent)]
    Memory(#[from] VirtualMemoryError),
    /// The governor refused to lease the pages the growth needs.
    #[error(transparent)]
    Budget(#[from] MemoryError),
    /// A backing and buffer were connected to different accounting books.
    #[error(
        "virtual backing charges physical memory to {backing}, but the buffer governor uses \
         {governor}; both must use the same memory authority"
    )]
    AuthorityMismatch {
        /// Authority that owns the backing's physical allocations.
        backing: onnx_runtime_memory_governor::MemoryAuthorityId,
        /// Authority supplied to the buffer.
        governor: onnx_runtime_memory_governor::MemoryAuthorityId,
    },
    /// The request exceeds the address space reserved at construction.
    #[error(
        "cannot grow to {requested} bytes: this buffer reserved {capacity} bytes of address \
         space and the reservation cannot be extended in place; construct it with a larger \
         capacity"
    )]
    OverCapacity {
        /// What was asked for.
        requested: usize,
        /// What was reserved.
        capacity: usize,
    },
}

impl VirtualBuffer<HostBacking> {
    /// Reserve capacity bytes of the process's own address space.
    ///
    /// The device equivalent takes a backing; see [VirtualBuffer::with_backing].
    pub fn with_capacity(
        capacity: usize,
        governor: Arc<dyn MemoryGovernor + Send + Sync>,
        tier: Tier,
        role: MemoryRole,
        holder: HolderId,
    ) -> Result<Self, VirtualBufferError> {
        Self::with_backing(HostBacking, capacity, governor, tier, role, holder)
    }
}

impl<B: VirtualBacking> VirtualBuffer<B> {
    /// Reserve `capacity` bytes of address space from `backing`, committing
    /// nothing.
    ///
    /// `capacity` is rounded up to the backing's granularity. Reserve for the
    /// largest the buffer could ever be: it costs address space, not memory,
    /// and it is the only bound that cannot be raised later.
    pub fn with_backing(
        backing: B,
        capacity: usize,
        governor: Arc<dyn MemoryGovernor + Send + Sync>,
        tier: Tier,
        role: MemoryRole,
        holder: HolderId,
    ) -> Result<Self, VirtualBufferError> {
        let physical_memory_accounting = backing.physical_memory_accounting();
        if let PhysicalMemoryAccounting::Backing { authority } = physical_memory_accounting {
            let governor_authority = governor.authority_id();
            if authority != governor_authority {
                return Err(VirtualBufferError::AuthorityMismatch {
                    backing: authority,
                    governor: governor_authority,
                });
            }
        }
        let capacity = round_up(backing.granularity(), capacity.max(1));
        let reservation = backing.reserve(capacity)?;
        Ok(Self {
            backing,
            reservation,
            capacity_bytes: capacity,
            len: 0,
            committed: 0,
            governor,
            tier,
            role,
            holder,
            physical_memory_accounting,
            lease: None,
        })
    }

    /// Bytes the caller has asked for.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the buffer holds nothing yet.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Bytes backed by physical memory, and therefore leased.
    ///
    /// At least [`VirtualBuffer::len`] and rounded to a granule, so the two
    /// differ by up to one granule after any growth.
    pub fn committed(&self) -> usize {
        self.committed
    }

    /// Address space reserved at construction. Cannot be raised.
    pub fn capacity(&self) -> usize {
        self.capacity_bytes
    }

    /// The base address, fixed for this buffer's whole life.
    ///
    /// Only the first [`VirtualBuffer::len`] bytes may be read or written.
    pub fn as_ptr(&self) -> *const u8 {
        B::base(&self.reservation) as *const u8
    }

    /// The base address, mutable.
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        B::base(&self.reservation) as *mut u8
    }

    /// The committed prefix, as bytes.
    ///
    /// # Safety
    ///
    /// Every byte of `..len` must have been initialised by the caller. Growth
    /// commits pages but does not promise their contents beyond what the
    /// platform guarantees for fresh mappings.
    pub unsafe fn as_slice(&self) -> &[u8] {
        // SAFETY: `..len` is committed, and the caller states it is initialised.
        unsafe { std::slice::from_raw_parts(self.as_ptr(), self.len) }
    }

    /// Grow to `bytes`, committing and leasing whatever pages that needs.
    ///
    /// A no-op when the buffer is already at least that long. On failure the
    /// buffer is exactly as it was: the lease is taken before the mapping, and
    /// released again if the mapping fails.
    pub fn grow_to(&mut self, bytes: usize) -> Result<(), VirtualBufferError> {
        if bytes <= self.len {
            return Ok(());
        }
        if bytes > self.capacity() {
            return Err(VirtualBufferError::OverCapacity {
                requested: bytes,
                capacity: self.capacity(),
            });
        }

        let needed = round_up(self.backing.granularity(), bytes);
        if needed > self.committed {
            let extra = needed - self.committed;
            let backing_accounts = matches!(
                self.physical_memory_accounting,
                PhysicalMemoryAccounting::Backing { .. }
            );
            if !backing_accounts {
                // Lease before mapping. A refusal must not commit memory, or
                // the budget is decorative.
                match self.lease.as_mut() {
                    Some(lease) => lease.grow(extra as u64)?,
                    None => {
                        self.lease = Some(self.governor.reserve(
                            self.tier,
                            extra as u64,
                            self.role,
                            self.holder,
                        )?);
                    }
                }
            }
            if let Err(error) = self
                .backing
                .commit(&mut self.reservation, self.committed, extra)
            {
                // Give the pages back rather than leaving the governor
                // believing they are held.
                if !backing_accounts {
                    self.release(extra);
                }
                return Err(error.into());
            }
            self.committed = needed;
        }
        self.len = bytes;
        Ok(())
    }

    /// Shrink to `bytes`, returning whole granules that fall entirely above it.
    ///
    /// A no-op when the buffer is already that short. The granule containing
    /// `bytes` stays committed, because part of it is still in use — so a small
    /// shrink can return nothing, which is honest rather than a failure.
    pub fn shrink_to(&mut self, bytes: usize) -> Result<(), VirtualBufferError> {
        if bytes >= self.len {
            return Ok(());
        }
        let needed = round_up(self.backing.granularity(), bytes);
        let mut offset = self.committed;
        while offset > needed {
            let granule = self.backing.granularity();
            offset -= granule;
            self.backing
                .release(&mut self.reservation, offset, granule)?;
            if matches!(
                self.physical_memory_accounting,
                PhysicalMemoryAccounting::Buffer
            ) {
                self.release(granule);
            }
            self.committed = offset;
        }
        self.len = bytes;
        Ok(())
    }

    /// Return `bytes` of budget, dropping the lease entirely when it empties.
    fn release(&mut self, bytes: usize) {
        let Some(lease) = self.lease.as_mut() else {
            return;
        };
        lease.shrink(bytes as u64);
        if lease.bytes() == 0 {
            self.lease = None;
        }
    }
}

fn round_up(granule: usize, bytes: usize) -> usize {
    bytes.div_ceil(granule) * granule
}

// `MemoryGovernor` is not `Debug` — it is a trait a third party implements, and
// requiring `Debug` of them to make this struct derivable would be the wrong
// way round. Report what the buffer itself knows.
impl<B: VirtualBacking> std::fmt::Debug for VirtualBuffer<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VirtualBuffer")
            .field("base", &self.as_ptr())
            .field("len", &self.len)
            .field("committed", &self.committed)
            .field("capacity", &self.capacity())
            .field("tier", &self.tier)
            .field("role", &self.role)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::granularity;
    use onnx_runtime_memory_governor::{DeviceKey, LeaseLedger, LedgerGovernor, MemoryAuthorityId};
    use std::sync::atomic::{AtomicUsize, Ordering};

    const HOLDER: HolderId = HolderId::new(4);

    fn buffer(capacity: usize, budget: u64) -> (VirtualBuffer, LedgerGovernor) {
        let governor = LedgerGovernor::new(LeaseLedger::new(0, budget, 0));
        let buffer = VirtualBuffer::with_capacity(
            capacity,
            Arc::new(governor.clone()),
            Tier::Host,
            MemoryRole::KvCache,
            HOLDER,
        )
        .expect("address space");
        (buffer, governor)
    }

    #[derive(Debug, Clone)]
    struct AuthorityBacking {
        authority: MemoryAuthorityId,
        reserves: Arc<AtomicUsize>,
        commits: Arc<AtomicUsize>,
    }

    // SAFETY: this test backing returns an inert address, never exposes slices,
    // and records mapping calls without touching memory.
    unsafe impl VirtualBacking for AuthorityBacking {
        type Reservation = usize;

        fn granularity(&self) -> usize {
            4096
        }

        fn physical_memory_accounting(&self) -> PhysicalMemoryAccounting {
            PhysicalMemoryAccounting::Backing {
                authority: self.authority,
            }
        }

        fn reserve(&self, _len: usize) -> Result<Self::Reservation, VirtualMemoryError> {
            self.reserves.fetch_add(1, Ordering::Relaxed);
            Ok(0x1000)
        }

        fn base(reservation: &Self::Reservation) -> usize {
            *reservation
        }

        fn commit(
            &self,
            _reservation: &mut Self::Reservation,
            _offset: usize,
            _len: usize,
        ) -> Result<(), VirtualMemoryError> {
            self.commits.fetch_add(1, Ordering::Relaxed);
            Ok(())
        }

        fn release(
            &self,
            _reservation: &mut Self::Reservation,
            _offset: usize,
            _len: usize,
        ) -> Result<(), VirtualMemoryError> {
            Ok(())
        }
    }

    #[test]
    fn backing_accounting_accepts_the_same_authority_without_double_charge() {
        let governor = LedgerGovernor::new(LeaseLedger::new_for_device(
            DeviceKey::device(2),
            8192,
            0,
            0,
        ));
        let commits = Arc::new(AtomicUsize::new(0));
        let backing = AuthorityBacking {
            authority: governor.authority_id(),
            reserves: Arc::new(AtomicUsize::new(0)),
            commits: Arc::clone(&commits),
        };
        let mut buffer = VirtualBuffer::with_backing(
            backing,
            8192,
            Arc::new(governor.clone()),
            Tier::Device,
            MemoryRole::KvCache,
            HOLDER,
        )
        .expect("matching authority");

        buffer.grow_to(4096).expect("backing owns the charge");

        assert_eq!(commits.load(Ordering::Relaxed), 1);
        assert_eq!(
            governor.used(Tier::Device),
            0,
            "mapped attribution must not add a second physical charge"
        );
    }

    #[test]
    fn backing_accounting_rejects_a_different_authority_before_reservation() {
        let backing_governor = LedgerGovernor::new(LeaseLedger::new_for_device(
            DeviceKey::device(2),
            8192,
            0,
            0,
        ));
        let buffer_governor = LedgerGovernor::new(LeaseLedger::new_for_device(
            DeviceKey::device(2),
            8192,
            0,
            0,
        ));
        let reserves = Arc::new(AtomicUsize::new(0));
        let commits = Arc::new(AtomicUsize::new(0));
        let backing = AuthorityBacking {
            authority: backing_governor.authority_id(),
            reserves: Arc::clone(&reserves),
            commits: Arc::clone(&commits),
        };

        let error = VirtualBuffer::with_backing(
            backing,
            8192,
            Arc::new(buffer_governor.clone()),
            Tier::Device,
            MemoryRole::KvCache,
            HOLDER,
        )
        .expect_err("different accounting authorities must be rejected");

        assert!(matches!(
            error,
            VirtualBufferError::AuthorityMismatch { backing, governor }
                if backing == backing_governor.authority_id()
                    && governor == buffer_governor.authority_id()
        ));
        assert_eq!(reserves.load(Ordering::Relaxed), 0);
        assert_eq!(commits.load(Ordering::Relaxed), 0);
        assert_eq!(buffer_governor.used(Tier::Device), 0);
    }

    /// The property the whole type exists for: growth does not move the buffer.
    ///
    /// A reallocating buffer would pass every other test here. This is the one
    /// that distinguishes them, and it is why a captured device graph survives
    /// growth.
    #[test]
    fn the_address_does_not_change_as_the_buffer_grows() {
        let (mut buffer, _) = buffer(64 << 20, 128 << 20);
        let base = buffer.as_ptr();
        for target in [1usize, 4096, 1 << 20, 8 << 20, 32 << 20] {
            buffer.grow_to(target).expect("within capacity and budget");
            assert_eq!(
                buffer.as_ptr(),
                base,
                "growing to {target} moved the buffer, which is the one thing it must not do"
            );
            assert_eq!(buffer.len(), target);
        }
    }

    /// Reserving address space costs no budget. Charging for it would make a
    /// governor refuse a buffer that never uses what it reserved.
    #[test]
    fn reserving_address_space_leases_nothing() {
        let (buffer, governor) = buffer(1 << 30, 1 << 20);
        assert_eq!(
            governor.available(Tier::Host),
            1 << 20,
            "a 1 GiB reservation must not consume a 1 MiB budget"
        );
        assert_eq!(buffer.committed(), 0);
        assert!(buffer.is_empty());
    }

    /// Committed bytes are leased, and the governor sees them.
    #[test]
    fn growth_leases_exactly_what_it_commits() {
        let (mut buffer, governor) = buffer(16 << 20, 32 << 20);
        buffer.grow_to(1).expect("granted");
        assert_eq!(
            buffer.committed(),
            granularity(),
            "growth commits whole granules"
        );
        assert_eq!(
            (32u64 << 20) - governor.available(Tier::Host),
            buffer.committed() as u64,
            "the governor must be charged the committed bytes, not the requested ones"
        );

        let before = governor.available(Tier::Host);
        buffer.grow_to(2).expect("granted");
        assert_eq!(
            governor.available(Tier::Host),
            before,
            "growing within an already-committed granule must not lease again"
        );
    }

    /// The written bytes survive growth. A buffer that remapped underneath the
    /// caller would lose them.
    #[test]
    fn contents_survive_growth() {
        let (mut buffer, _) = buffer(8 << 20, 16 << 20);
        buffer.grow_to(4096).expect("granted");
        // SAFETY: the first 4096 bytes are committed.
        unsafe { std::ptr::write_bytes(buffer.as_mut_ptr(), 0xC7, 4096) };

        buffer.grow_to(4 << 20).expect("granted");
        // SAFETY: still committed, and just written.
        let head = unsafe { std::slice::from_raw_parts(buffer.as_ptr(), 4096) };
        assert!(
            head.iter().all(|&byte| byte == 0xC7),
            "growth lost the bytes that were already there"
        );
    }

    /// Shrinking gives whole granules back to the governor.
    #[test]
    fn shrinking_returns_committed_pages() {
        let (mut buffer, governor) = buffer(16 << 20, 32 << 20);
        buffer.grow_to(4 << 20).expect("granted");
        let held = buffer.committed();
        assert!(held >= 4 << 20);

        buffer.shrink_to(0).expect("shrunk");
        assert_eq!(buffer.committed(), 0, "everything must come back");
        assert_eq!(
            governor.available(Tier::Host),
            32 << 20,
            "the governor must see the pages returned"
        );
        assert_eq!(buffer.len(), 0);
    }

    /// A shrink inside one granule returns nothing, because part of that
    /// granule is still in use. Reporting that honestly beats unmapping memory
    /// the caller still reads.
    #[test]
    fn a_shrink_within_a_granule_keeps_the_page() {
        let (mut buffer, _) = buffer(16 << 20, 32 << 20);
        buffer.grow_to(granularity()).expect("granted");
        let committed = buffer.committed();

        buffer.shrink_to(granularity() - 1).expect("shrunk");
        assert_eq!(
            buffer.committed(),
            committed,
            "the granule containing the new end must stay mapped"
        );
        assert_eq!(buffer.len(), granularity() - 1);
    }

    /// Growth past the reservation fails and says why, rather than silently
    /// reallocating and moving the address.
    #[test]
    fn growing_past_the_reservation_is_refused_and_names_the_capacity() {
        let (mut buffer, _) = buffer(1 << 20, 64 << 20);
        let capacity = buffer.capacity();
        let error = buffer
            .grow_to(capacity + 1)
            .expect_err("the reservation cannot be extended");
        let message = error.to_string();
        assert!(
            message.contains("larger capacity"),
            "the error must say what to do, got: {message}"
        );
        assert_eq!(buffer.len(), 0, "a refused growth must change nothing");
    }

    /// A refused lease commits nothing, and leaves the buffer usable.
    #[test]
    fn a_refused_lease_leaves_the_buffer_untouched() {
        let (mut buffer, governor) = buffer(64 << 20, granularity() as u64);
        buffer
            .grow_to(1)
            .expect("the first granule fits the budget");
        let committed = buffer.committed();

        let error = buffer.grow_to(32 << 20);
        assert!(error.is_err(), "a 32 MiB growth cannot fit one granule");
        assert_eq!(
            buffer.committed(),
            committed,
            "a refused growth must not commit pages"
        );
        assert_eq!(governor.available(Tier::Host), 0);

        // Still usable afterwards.
        // SAFETY: the first granule is committed.
        unsafe { std::ptr::write_bytes(buffer.as_mut_ptr(), 0x11, committed) };
    }

    /// Dropping the buffer returns everything, without an explicit release.
    #[test]
    fn dropping_the_buffer_returns_its_budget() {
        let (mut buffer, governor) = buffer(16 << 20, 32 << 20);
        buffer.grow_to(8 << 20).expect("granted");
        assert!(governor.available(Tier::Host) < 32 << 20);

        drop(buffer);
        assert_eq!(
            governor.available(Tier::Host),
            32 << 20,
            "the lease must be released when the buffer goes"
        );
    }
}
