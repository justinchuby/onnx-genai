//! Plain-data records that cross the `nxmem` boundary.
//!
//! Every type here is `#[repr(C)]`, holds no Rust trait object, no `Arc`, no
//! Rust enum layout, and no heap allocation owned by the other module. Enum-
//! shaped values travel as raw `u32` wire codes with checked accessors.

use crate::status::{NxmemStatus, NxmemStatusCode};

/// Wire values for the memory tier an object lives in.
///
/// Mirrors the host-side `Tier`, but travels as a raw `u32`: see
/// [`NxmemDeviceId::tier_code`].
pub const NXMEM_TIER_DEVICE: u32 = 0;
/// Host RAM tier wire value.
pub const NXMEM_TIER_HOST: u32 = 1;
/// Spill-to-disk tier wire value.
pub const NXMEM_TIER_DISK: u32 = 2;

/// Which physical device an object belongs to.
///
/// Carried by every allocation, backing, and shared-mapping call so a
/// cross-device request is rejected before it can free or map anything.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NxmemDeviceId {
    /// Tier wire value: one of [`NXMEM_TIER_DEVICE`], [`NXMEM_TIER_HOST`],
    /// [`NXMEM_TIER_DISK`]. Unknown values must be rejected, not guessed.
    pub tier: u32,
    /// Device ordinal within the tier.
    pub index: u32,
}

impl NxmemDeviceId {
    /// The single host-RAM device.
    pub const HOST: Self = Self {
        tier: NXMEM_TIER_HOST,
        index: 0,
    };

    /// An accelerator device by ordinal.
    pub const fn device(index: u32) -> Self {
        Self {
            tier: NXMEM_TIER_DEVICE,
            index,
        }
    }

    /// The tier wire value, or `None` when this build does not know it.
    pub const fn tier_code(self) -> Option<u32> {
        match self.tier {
            NXMEM_TIER_DEVICE | NXMEM_TIER_HOST | NXMEM_TIER_DISK => Some(self.tier),
            _ => None,
        }
    }
}

/// The identity of one allocation as it crosses the boundary.
///
/// `allocation_id` comes from a monotonic counter on the side that created the
/// allocation. It is deliberately **never** derived from the address: a
/// virtual address can be reused, an identity cannot. A plugin must reject a
/// call whose `allocation_id` it does not currently hold, even when `ptr`
/// happens to name live memory.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct NxmemAllocation {
    /// `size_of` this struct as the sender defines it.
    pub struct_size: u32,
    /// Reserved; must be zero.
    pub reserved: u32,
    /// The mechanism instance that created the allocation.
    pub mechanism_id: u64,
    /// Monotonic, never address-derived allocation identity.
    pub allocation_id: u64,
    /// The device the allocation lives on.
    pub device: NxmemDeviceId,
    /// The allocation's base address.
    pub ptr: *mut u8,
    /// The exact byte count the allocation was created with.
    pub bytes: u64,
    /// The exact alignment the allocation was created with.
    pub align: u64,
}

impl NxmemAllocation {
    /// An allocation record with the correct `struct_size`.
    pub fn new(
        mechanism_id: u64,
        allocation_id: u64,
        device: NxmemDeviceId,
        ptr: *mut u8,
        bytes: u64,
        align: u64,
    ) -> Self {
        Self {
            struct_size: core::mem::size_of::<Self>() as u32,
            reserved: 0,
            mechanism_id,
            allocation_id,
            device,
            ptr,
            bytes,
            align,
        }
    }
}

/// A request to create an allocation.
///
/// The same record serves ordinary allocation and lazy reserve-with-commit.
/// `committed_ranges` is ignored by the ordinary `allocate` slot; the virtual
/// backing slot honours it.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct NxmemAllocRequest {
    /// `size_of` this struct as the sender defines it.
    pub struct_size: u32,
    /// Reserved; must be zero.
    pub reserved: u32,
    /// The mechanism instance being addressed. A plugin must reject a value
    /// that is not its own.
    pub mechanism_id: u64,
    /// The identity the host assigns to the allocation being created.
    pub allocation_id: u64,
    /// The device being addressed. A plugin must reject a foreign device.
    pub device: NxmemDeviceId,
    /// Requested byte count.
    pub bytes: u64,
    /// Requested alignment; must be a power of two.
    pub align: u64,
    /// Ranges to commit immediately. Borrowed for the call only; the plugin
    /// must not retain the pointer.
    pub committed_ranges: *const NxmemByteRange,
    /// Number of entries in `committed_ranges`.
    pub committed_range_count: u64,
}

impl NxmemAllocRequest {
    /// An ordinary allocation request with no pre-committed ranges.
    pub fn new(
        mechanism_id: u64,
        allocation_id: u64,
        device: NxmemDeviceId,
        bytes: u64,
        align: u64,
    ) -> Self {
        Self {
            struct_size: core::mem::size_of::<Self>() as u32,
            reserved: 0,
            mechanism_id,
            allocation_id,
            device,
            bytes,
            align,
            committed_ranges: core::ptr::null(),
            committed_range_count: 0,
        }
    }
}

/// A half-open `[offset, offset + bytes)` span inside one allocation.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NxmemByteRange {
    /// Byte offset from the allocation base.
    pub offset: u64,
    /// Span length in bytes.
    pub bytes: u64,
}

/// What an allocation call produced.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct NxmemAllocResult {
    /// `size_of` this struct as the sender defines it.
    pub struct_size: u32,
    /// Reserved; must be zero.
    pub reserved: u32,
    /// The allocation base address. Non-null on success.
    pub ptr: *mut u8,
    /// Physical bytes whose ownership was newly created by this call. A
    /// mapping that reuses already-owned physical memory reports zero.
    pub owned_bytes: u64,
    /// Bytes whose mapped attribution newly became mapped.
    pub mapped_bytes: u64,
}

impl NxmemAllocResult {
    /// A zeroed result for the caller to pass in.
    pub const fn zeroed() -> Self {
        Self {
            struct_size: core::mem::size_of::<Self>() as u32,
            reserved: 0,
            ptr: core::ptr::null_mut(),
            owned_bytes: 0,
            mapped_bytes: 0,
        }
    }
}

/// A commit/decommit request against a span of an existing allocation.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct NxmemRangeRequest {
    /// `size_of` this struct as the sender defines it.
    pub struct_size: u32,
    /// Reserved; must be zero.
    pub reserved: u32,
    /// The allocation being addressed, carrying mechanism and device identity.
    pub allocation: NxmemAllocation,
    /// The span within that allocation.
    pub range: NxmemByteRange,
}

impl NxmemRangeRequest {
    /// A range request with the correct `struct_size`.
    pub fn new(allocation: NxmemAllocation, offset: u64, bytes: u64) -> Self {
        Self {
            struct_size: core::mem::size_of::<Self>() as u32,
            reserved: 0,
            allocation,
            range: NxmemByteRange { offset, bytes },
        }
    }
}

// ─── Release outcome ────────────────────────────────────────────────────────

/// Release completed: the whole allocation is gone.
pub const NXMEM_RELEASE_COMPLETE: u32 = 0;
/// Release mutated state but could not finish. Residual ownership stays with
/// the plugin and the virtual span must not be reused.
pub const NXMEM_RELEASE_QUARANTINED: u32 = 1;
/// Release changed nothing. This is the only shape that means "unchanged".
pub const NXMEM_RELEASE_FAILED: u32 = 2;

/// The structured result of a terminal release.
///
/// Zero unmapped bytes is a valid complete result and never signals failure —
/// the state field is the only failure channel.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct NxmemReleaseOutcome {
    /// `size_of` this struct as the sender defines it.
    pub struct_size: u32,
    /// One of [`NXMEM_RELEASE_COMPLETE`], [`NXMEM_RELEASE_QUARANTINED`],
    /// [`NXMEM_RELEASE_FAILED`]. Unknown values must be treated as failure.
    pub state: u32,
    /// The allocation's byte count, echoed for accounting.
    pub allocation_bytes: u64,
    /// Bytes whose global mapping reference transitioned to unmapped.
    pub unmapped_bytes: u64,
    /// Physical bytes the plugin still owns after a quarantined release.
    pub residual_owned_bytes: u64,
    /// Why the release did not complete. Meaningful when `state` is not
    /// [`NXMEM_RELEASE_COMPLETE`].
    pub failure: NxmemStatus,
}

impl NxmemReleaseOutcome {
    /// A zeroed outcome for the caller to pass in.
    ///
    /// The zero value is `Complete` with zero bytes, which is only ever
    /// meaningful once a slot has actually written to it; callers must not
    /// interpret an unwritten outcome.
    pub const fn zeroed() -> Self {
        Self {
            struct_size: core::mem::size_of::<Self>() as u32,
            state: NXMEM_RELEASE_COMPLETE,
            allocation_bytes: 0,
            unmapped_bytes: 0,
            residual_owned_bytes: 0,
            failure: NxmemStatus::ok(),
        }
    }

    /// A complete release.
    pub const fn complete(allocation_bytes: u64, unmapped_bytes: u64) -> Self {
        Self {
            struct_size: core::mem::size_of::<Self>() as u32,
            state: NXMEM_RELEASE_COMPLETE,
            allocation_bytes,
            unmapped_bytes,
            residual_owned_bytes: 0,
            failure: NxmemStatus::ok(),
        }
    }

    /// A release that mutated state and left residual ownership behind.
    pub fn quarantined(
        allocation_bytes: u64,
        unmapped_bytes: u64,
        residual_owned_bytes: u64,
        failure: NxmemStatus,
    ) -> Self {
        Self {
            struct_size: core::mem::size_of::<Self>() as u32,
            state: NXMEM_RELEASE_QUARANTINED,
            allocation_bytes,
            unmapped_bytes,
            residual_owned_bytes,
            failure,
        }
    }

    /// A release that changed nothing.
    pub fn failed(allocation_bytes: u64, failure: NxmemStatus) -> Self {
        Self {
            struct_size: core::mem::size_of::<Self>() as u32,
            state: NXMEM_RELEASE_FAILED,
            allocation_bytes,
            unmapped_bytes: 0,
            residual_owned_bytes: 0,
            failure,
        }
    }

    /// Whether the allocation is fully gone.
    pub const fn is_complete(&self) -> bool {
        self.state == NXMEM_RELEASE_COMPLETE
    }
}

/// One retired deferred release, reported to the host.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct NxmemReleaseCompletion {
    /// `size_of` this struct as the sender defines it.
    pub struct_size: u32,
    /// Reserved; must be zero.
    pub reserved: u32,
    /// The ticket the enqueue call returned.
    pub ticket: u64,
    /// The mechanism instance that owned the allocation.
    pub mechanism_id: u64,
    /// The allocation identity that retired.
    pub allocation_id: u64,
    /// The structured outcome of the physical release.
    pub outcome: NxmemReleaseOutcome,
}

impl NxmemReleaseCompletion {
    /// A completion record with the correct `struct_size`.
    pub fn new(
        ticket: u64,
        mechanism_id: u64,
        allocation_id: u64,
        outcome: NxmemReleaseOutcome,
    ) -> Self {
        Self {
            struct_size: core::mem::size_of::<Self>() as u32,
            reserved: 0,
            ticket,
            mechanism_id,
            allocation_id,
            outcome,
        }
    }
}

// ─── Host callbacks ─────────────────────────────────────────────────────────

/// A plugin's request that the host free cached memory.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct NxmemReclaimRequest {
    /// `size_of` this struct as the sender defines it.
    pub struct_size: u32,
    /// Reserved; must be zero.
    pub reserved: u32,
    /// The mechanism instance asking.
    pub mechanism_id: u64,
    /// The device under pressure.
    pub device: NxmemDeviceId,
    /// Bytes the plugin needs.
    pub bytes: u64,
}

impl NxmemReclaimRequest {
    /// A reclaim request with the correct `struct_size`.
    pub fn new(mechanism_id: u64, device: NxmemDeviceId, bytes: u64) -> Self {
        Self {
            struct_size: core::mem::size_of::<Self>() as u32,
            reserved: 0,
            mechanism_id,
            device,
            bytes,
        }
    }
}

/// Callbacks the host offers to a plugin allocator.
///
/// # Ownership
///
/// The table and `host_ctx` are **borrowed for the lifetime of the opened
/// allocator**. The host guarantees they outlive the allocator's final
/// `release` and every queued release that names it. The plugin must not free
/// or retain either beyond that point.
///
/// # Threading and reentrancy
///
/// * The host calls into the plugin **without holding any governance lock**.
/// * A plugin may therefore call back into these slots from inside a plugin
///   call the host is currently making (allocate → `request_reclaim` is the
///   expected pattern), and from a plugin-owned worker thread.
/// * Host callbacks must not block indefinitely and must not re-enter the same
///   allocator instance.
/// * A callback returning a non-`Ok` status is a normal, expected outcome. The
///   plugin must handle it — typically by failing the operation with
///   [`NxmemStatusCode::CallbackFailed`] or `OutOfMemory` — and must not
///   abort, panic, or leak.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct NxmemHostCallbacks {
    /// `size_of` this struct as the host defines it.
    pub struct_size: u32,
    /// The minor version the host negotiated.
    pub abi_minor: u32,
    /// Opaque host state passed back to every slot. Never dereferenced by the
    /// plugin.
    pub host_ctx: *mut core::ffi::c_void,
    /// Ask the host to release cached memory. Writes the bytes actually
    /// reclaimed. Null when the host offers no reclaim path.
    pub request_reclaim: Option<
        unsafe extern "C" fn(
            host_ctx: *mut core::ffi::c_void,
            request: *const NxmemReclaimRequest,
            reclaimed_out: *mut u64,
        ) -> NxmemStatus,
    >,
    /// Report one retired deferred release so the host can settle accounting.
    /// Null when the host does not consume deferred release.
    pub release_completed: Option<
        unsafe extern "C" fn(
            host_ctx: *mut core::ffi::c_void,
            completion: *const NxmemReleaseCompletion,
        ) -> NxmemStatus,
    >,
}

impl NxmemHostCallbacks {
    /// A table with no callbacks offered.
    pub const fn empty(abi_minor: u32) -> Self {
        Self {
            struct_size: core::mem::size_of::<Self>() as u32,
            abi_minor,
            host_ctx: core::ptr::null_mut(),
            request_reclaim: None,
            release_completed: None,
        }
    }
}

/// How much a plugin still owns, used to gate unload.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct NxmemUnloadReport {
    /// `size_of` this struct as the plugin defines it.
    pub struct_size: u32,
    /// Reserved; must be zero.
    pub reserved: u32,
    /// Allocator instances the host has opened and not released.
    pub live_allocators: u64,
    /// Allocations not yet terminally released.
    pub live_allocations: u64,
    /// Borrowed views over live allocations.
    pub live_views: u64,
    /// Capability objects (shared prefixes, backing handles) still held.
    pub live_capabilities: u64,
    /// Releases enqueued and not yet retired.
    pub queued_releases: u64,
}

impl NxmemUnloadReport {
    /// A zeroed report for the caller to pass in.
    pub const fn zeroed() -> Self {
        Self {
            struct_size: core::mem::size_of::<Self>() as u32,
            reserved: 0,
            live_allocators: 0,
            live_allocations: 0,
            live_views: 0,
            live_capabilities: 0,
            queued_releases: 0,
        }
    }

    /// Total live objects. Unload is refused or deferred while this is
    /// non-zero.
    pub const fn total(&self) -> u64 {
        self.live_allocators
            .saturating_add(self.live_allocations)
            .saturating_add(self.live_views)
            .saturating_add(self.live_capabilities)
            .saturating_add(self.queued_releases)
    }
}

// ─── Shared physical prefixes ───────────────────────────────────────────────

/// A plugin-owned shared physical prefix.
///
/// # Ownership
///
/// Created by `create_shared_prefix` and destroyed by `release_shared_prefix`,
/// exactly once, on the same mechanism instance. `handle` is opaque to the
/// host: it is never dereferenced, only echoed back.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct NxmemSharedPrefixHandle {
    /// `size_of` this struct as the plugin defines it.
    pub struct_size: u32,
    /// Reserved; must be zero.
    pub reserved: u32,
    /// The mechanism instance that owns the prefix.
    pub mechanism_id: u64,
    /// Opaque plugin-side identity. Non-zero when valid.
    pub handle: u64,
    /// The device the prefix lives on.
    pub device: NxmemDeviceId,
    /// Device address of the prefix.
    pub device_ptr: u64,
    /// Physical bytes owned by the prefix.
    pub committed_physical_bytes: u64,
    /// Bytes currently mapped.
    pub mapped_bytes: u64,
    /// Bytes originally requested.
    pub requested_bytes: u64,
}

impl NxmemSharedPrefixHandle {
    /// A zeroed handle for the caller to pass in.
    pub const fn zeroed() -> Self {
        Self {
            struct_size: core::mem::size_of::<Self>() as u32,
            reserved: 0,
            mechanism_id: 0,
            handle: 0,
            device: NxmemDeviceId { tier: 0, index: 0 },
            device_ptr: 0,
            committed_physical_bytes: 0,
            mapped_bytes: 0,
            requested_bytes: 0,
        }
    }
}

/// A request to map a shared prefix into an allocation.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct NxmemSharedPrefixCommitRequest {
    /// `size_of` this struct as the sender defines it.
    pub struct_size: u32,
    /// Reserved; must be zero.
    pub reserved: u32,
    /// The prefix to map. Borrowed for the call only.
    pub prefix: NxmemSharedPrefixHandle,
    /// The destination allocation.
    pub allocation: NxmemAllocation,
    /// Byte offset within the allocation to map at.
    pub byte_offset: u64,
}

impl NxmemSharedPrefixCommitRequest {
    /// A commit request with the correct `struct_size`.
    pub fn new(
        prefix: NxmemSharedPrefixHandle,
        allocation: NxmemAllocation,
        byte_offset: u64,
    ) -> Self {
        Self {
            struct_size: core::mem::size_of::<Self>() as u32,
            reserved: 0,
            prefix,
            allocation,
            byte_offset,
        }
    }
}

/// The accounting result of mapping a shared prefix.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct NxmemSharedPrefixCommitInfo {
    /// `size_of` this struct as the sender defines it.
    pub struct_size: u32,
    /// Reserved; must be zero.
    pub reserved: u32,
    /// Newly owned physical bytes. A valid additional mapping of an
    /// already-owned prefix reports zero.
    pub additional_owned_bytes: u64,
    /// Newly mapped bytes on the mapped-attribution axis.
    pub newly_mapped_bytes: u64,
    /// Physical granules touched.
    pub granules: u64,
}

impl NxmemSharedPrefixCommitInfo {
    /// A zeroed info record for the caller to pass in.
    pub const fn zeroed() -> Self {
        Self {
            struct_size: core::mem::size_of::<Self>() as u32,
            reserved: 0,
            additional_owned_bytes: 0,
            newly_mapped_bytes: 0,
            granules: 0,
        }
    }
}

/// Reject a call whose mechanism or device identity does not match the
/// receiver's own.
///
/// Both sides use this: the host checks before calling, the plugin checks on
/// entry. Two independent checks are deliberate — the host protects itself
/// from its own bookkeeping bugs, and the plugin protects itself from a
/// hostile or confused host.
pub fn check_identity(
    expected_mechanism: u64,
    expected_device: NxmemDeviceId,
    actual_mechanism: u64,
    actual_device: NxmemDeviceId,
) -> Result<(), NxmemStatus> {
    if actual_mechanism != expected_mechanism {
        return Err(NxmemStatus::with_message(
            NxmemStatusCode::WrongMechanism,
            &format!(
                "nxmem: object belongs to mechanism {actual_mechanism} but mechanism \
                 {expected_mechanism} was addressed; an allocation may only be used and released \
                 by the mechanism that created it"
            ),
        ));
    }
    if actual_device != expected_device {
        return Err(NxmemStatus::with_message(
            NxmemStatusCode::WrongDevice,
            &format!(
                "nxmem: object belongs to device (tier {}, index {}) but device (tier {}, index \
                 {}) was addressed",
                actual_device.tier, actual_device.index, expected_device.tier, expected_device.index
            ),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constructors_stamp_the_real_struct_size() {
        let allocation =
            NxmemAllocation::new(7, 9, NxmemDeviceId::HOST, core::ptr::null_mut(), 64, 16);
        assert_eq!(
            allocation.struct_size as usize,
            core::mem::size_of::<NxmemAllocation>()
        );
        let request = NxmemAllocRequest::new(7, 9, NxmemDeviceId::HOST, 64, 16);
        assert_eq!(
            request.struct_size as usize,
            core::mem::size_of::<NxmemAllocRequest>()
        );
        let range = NxmemRangeRequest::new(allocation, 0, 32);
        assert_eq!(
            range.struct_size as usize,
            core::mem::size_of::<NxmemRangeRequest>()
        );
    }

    #[test]
    fn a_wrong_mechanism_is_rejected_before_a_wrong_device() {
        let status = check_identity(1, NxmemDeviceId::HOST, 2, NxmemDeviceId::device(3))
            .expect_err("both identities differ");
        assert_eq!(status.status_code(), Some(NxmemStatusCode::WrongMechanism));
        assert!(status.describe().contains("mechanism that created it"));
    }

    #[test]
    fn a_wrong_device_on_the_right_mechanism_is_rejected() {
        let status = check_identity(1, NxmemDeviceId::HOST, 1, NxmemDeviceId::device(0))
            .expect_err("device differs");
        assert_eq!(status.status_code(), Some(NxmemStatusCode::WrongDevice));
    }

    #[test]
    fn matching_identity_is_accepted() {
        assert!(check_identity(4, NxmemDeviceId::device(1), 4, NxmemDeviceId::device(1)).is_ok());
    }

    #[test]
    fn zero_unmapped_bytes_is_a_valid_complete_release() {
        let outcome = NxmemReleaseOutcome::complete(4096, 0);
        assert!(outcome.is_complete());
        assert_eq!(outcome.unmapped_bytes, 0);
        assert!(outcome.failure.is_ok());
    }

    #[test]
    fn a_quarantined_release_keeps_residual_ownership() {
        let outcome = NxmemReleaseOutcome::quarantined(
            4096,
            2048,
            2048,
            NxmemStatus::with_message(NxmemStatusCode::DeviceError, "unmap failed midway"),
        );
        assert!(!outcome.is_complete());
        assert_eq!(outcome.state, NXMEM_RELEASE_QUARANTINED);
        assert_eq!(outcome.residual_owned_bytes, 2048);
    }

    #[test]
    fn a_failed_release_reports_no_mutation() {
        let outcome = NxmemReleaseOutcome::failed(
            4096,
            NxmemStatus::from_code(NxmemStatusCode::UnknownAllocation),
        );
        assert_eq!(outcome.state, NXMEM_RELEASE_FAILED);
        assert_eq!(outcome.unmapped_bytes, 0);
        assert_eq!(outcome.residual_owned_bytes, 0);
    }

    #[test]
    fn unload_report_totals_every_axis() {
        let report = NxmemUnloadReport {
            live_allocators: 1,
            live_allocations: 2,
            live_views: 3,
            live_capabilities: 4,
            queued_releases: 5,
            ..NxmemUnloadReport::zeroed()
        };
        assert_eq!(report.total(), 15);
        assert_eq!(NxmemUnloadReport::zeroed().total(), 0);
    }

    #[test]
    fn unknown_tier_codes_are_not_guessed() {
        assert_eq!(NxmemDeviceId::HOST.tier_code(), Some(NXMEM_TIER_HOST));
        assert_eq!(
            NxmemDeviceId {
                tier: 77,
                index: 0
            }
            .tier_code(),
            None
        );
    }
}
