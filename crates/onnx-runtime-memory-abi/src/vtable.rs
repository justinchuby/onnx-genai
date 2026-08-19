//! Vtables and the prefix-negotiation rules that make them safe to read.
//!
//! # Reading an untrusted vtable
//!
//! A participant may hand over a struct that is **smaller** than this build's
//! definition (it was compiled at an older minor) or **larger** (newer minor,
//! same major). Dereferencing the whole struct in either case is wrong: the
//! small case reads past the end of the allocation.
//!
//! The only supported way to read one is [`NxmemAllocatorVtable::read_prefix`]
//! and its siblings. They:
//!
//! 1. reject a null or misaligned pointer;
//! 2. read `struct_size` — always the first field, at offset 0, in every
//!    version;
//! 3. reject a `struct_size` below the required prefix for the level the
//!    vtable declares ([`crate::NxmemStatusCode::ShortStruct`]);
//! 4. copy `min(struct_size, size_of::<Self>())` bytes into a zero-initialised
//!    local, so trailing slots this build knows about but the sender does not
//!    read back as `None`/null, and trailing bytes this build does not know
//!    about are ignored.
//!
//! Every optional slot is a nullable function pointer. Null means the
//! capability is absent and must surface as
//! [`crate::NxmemStatusCode::UnsupportedCapability`] — never as a successful
//! no-op.
//!
//! # Per-vtable levels
//!
//! Negotiation fixes the **ceiling**. Each vtable independently declares the
//! level it actually implements in `abi_minor`, so one plugin may ship a
//! current mechanism next to one still built at the baseline prefix. A vtable
//! declaring a level *above* the negotiated ceiling is **clamped**, not
//! rejected: its struct is a strict superset, so the host reads only the
//! prefix it agreed to and refuses to call any slot introduced above the
//! ceiling. `read_prefix` rewrites `abi_minor` to that effective level, so a
//! caller need only consult the value it gets back. A vtable whose
//! `struct_size` is smaller than the level it *claims* is still rejected: that
//! is a sender that contradicts itself, not an old one.

use core::ffi::c_void;

use crate::status::{NxmemStatus, NxmemStatusCode};
use crate::types::{
    NxmemAllocRequest, NxmemAllocResult, NxmemAllocation, NxmemDeviceId, NxmemHostCallbacks,
    NxmemRangeRequest, NxmemReleaseOutcome, NxmemSharedPrefixCommitInfo,
    NxmemSharedPrefixCommitRequest, NxmemSharedPrefixHandle,
};

/// The request a host passes to `open_allocator`.
///
/// # Ownership
///
/// `callbacks` is borrowed for the **lifetime of the opened allocator**, not
/// just for the call. See [`NxmemHostCallbacks`].
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct NxmemOpenRequest {
    /// `size_of` this struct as the host defines it.
    pub struct_size: u32,
    /// The minor version negotiated for this module. Acts as the ceiling for
    /// every vtable the plugin returns.
    pub abi_minor: u32,
    /// The device the host wants an allocator for.
    pub device: NxmemDeviceId,
    /// Capabilities the host requires. The plugin must fail the open rather
    /// than return an allocator missing one of them.
    pub required_capability_flags: u64,
    /// Host callbacks, borrowed for the allocator's lifetime. May be null when
    /// the host offers none.
    pub callbacks: *const NxmemHostCallbacks,
}

impl NxmemOpenRequest {
    /// An open request with the correct `struct_size`.
    pub fn new(
        abi_minor: u32,
        device: NxmemDeviceId,
        required_capability_flags: u64,
        callbacks: *const NxmemHostCallbacks,
    ) -> Self {
        Self {
            struct_size: core::mem::size_of::<Self>() as u32,
            abi_minor,
            device,
            required_capability_flags,
            callbacks,
        }
    }
}

/// Reject a vtable pointer that is null or misaligned for `T`.
fn check_vtable_ptr<T>(ptr: *const T, what: &str) -> Result<(), NxmemStatus> {
    if ptr.is_null() {
        return Err(NxmemStatus::with_message(
            NxmemStatusCode::InvalidArgument,
            &format!("nxmem: {what} vtable pointer is null"),
        ));
    }
    if !(ptr as usize).is_multiple_of(core::mem::align_of::<T>()) {
        return Err(NxmemStatus::with_message(
            NxmemStatusCode::InvalidArgument,
            &format!(
                "nxmem: {what} vtable pointer {ptr:p} is not aligned to {} bytes",
                core::mem::align_of::<T>()
            ),
        ));
    }
    Ok(())
}

/// Copy the readable prefix of `ptr` into a zero-initialised `T`.
///
/// # Safety
///
/// `ptr` must be non-null, aligned for `T`, and point to at least
/// `min(declared_size, size_of::<T>())` readable bytes. Callers reach this
/// through the `read_prefix` helpers, which enforce those conditions.
unsafe fn copy_prefix<T: Copy>(ptr: *const T, declared_size: usize) -> T {
    let readable = declared_size.min(core::mem::size_of::<T>());
    let mut out = core::mem::MaybeUninit::<T>::zeroed();
    // SAFETY: the destination is a fresh `T`-sized allocation; the source is
    // valid for `readable` bytes by this function's contract; the regions
    // cannot overlap because `out` is a local.
    unsafe {
        core::ptr::copy_nonoverlapping(ptr as *const u8, out.as_mut_ptr() as *mut u8, readable);
        out.assume_init()
    }
}

/// Read the leading `struct_size` field, which lives at offset 0 in every
/// version of every `nxmem` struct.
///
/// # Safety
///
/// `ptr` must be non-null, aligned, and point to at least four readable bytes.
unsafe fn read_struct_size<T>(ptr: *const T) -> u32 {
    // SAFETY: `struct_size` is the first field of every nxmem struct, so four
    // readable bytes at `ptr` are guaranteed by this function's contract.
    unsafe { core::ptr::read_unaligned(ptr as *const u32) }
}

fn short_struct(what: &str, declared: usize, required: usize, minor: u32) -> NxmemStatus {
    NxmemStatus::with_message(
        NxmemStatusCode::ShortStruct,
        &format!(
            "nxmem: {what} declares struct_size {declared} but nxmem minor {minor} requires at \
             least {required} bytes; rebuild the plugin against a matching nxmem ABI header"
        ),
    )
}

// ─── Allocator vtable ───────────────────────────────────────────────────────

/// The ordinary allocator mechanism plus its optional capability slots.
///
/// # Ownership
///
/// The vtable and its `ctx` are owned by the plugin. The host takes one
/// reference on open and gives it back with exactly one `release`. `retain`
/// takes an extra reference; every `retain` must be paired with a `release`.
/// The plugin must keep the vtable, `ctx`, and `name` alive until the final
/// `release` returns.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct NxmemAllocatorVtable {
    /// `size_of` this struct as the plugin defines it.
    pub struct_size: u32,
    /// The nxmem minor level this vtable implements. Must not exceed the
    /// negotiated ceiling.
    pub abi_minor: u32,
    /// Non-zero identity of this mechanism instance. Echoed by the host in
    /// every call and re-checked by the plugin.
    pub mechanism_id: u64,
    /// The device this mechanism serves.
    pub device: NxmemDeviceId,
    /// Capabilities this mechanism actually provides.
    pub capability_flags: u64,
    /// NUL-terminated UTF-8 name, valid until the final `release` returns.
    pub name: *const u8,
    /// Opaque plugin state passed to every slot.
    pub ctx: *mut c_void,

    // ─── required at minor 0 ───
    /// Ordinary allocation. Required.
    pub allocate: Option<
        unsafe extern "C" fn(
            ctx: *mut c_void,
            request: *const NxmemAllocRequest,
            result_out: *mut NxmemAllocResult,
        ) -> NxmemStatus,
    >,
    /// Terminal release reporting only newly unmapped bytes. Required; it is
    /// the fallback when structured release is absent.
    pub deallocate: Option<
        unsafe extern "C" fn(
            ctx: *mut c_void,
            allocation: *const NxmemAllocation,
            unmapped_bytes_out: *mut u64,
        ) -> NxmemStatus,
    >,
    /// Take an extra reference on the mechanism. Required.
    pub retain: Option<unsafe extern "C" fn(ctx: *mut c_void)>,
    /// Drop one reference. Required. The plugin destroys the mechanism when
    /// the count reaches zero **and** no queued release still names it.
    pub release: Option<unsafe extern "C" fn(ctx: *mut c_void)>,

    // ─── optional at minor 0 ───
    /// Lazy backing capability, or null when unsupported.
    pub virtual_backing: *const NxmemVirtualBackingVtable,
    /// Shared physical mapping capability, or null when unsupported.
    pub shared_mapping: *const NxmemSharedMappingVtable,
    /// Queue a stream-ordered release instead of freeing now. Writes a ticket.
    pub enqueue_release: Option<
        unsafe extern "C" fn(
            ctx: *mut c_void,
            allocation: *const NxmemAllocation,
            ticket_out: *mut u64,
        ) -> NxmemStatus,
    >,
    /// Retire up to `max` queued releases, invoking the host's
    /// `release_completed` callback for each, in enqueue order.
    pub drain_releases: Option<
        unsafe extern "C" fn(ctx: *mut c_void, max: u64, retired_out: *mut u64) -> NxmemStatus,
    >,
    /// How many releases are queued and not yet retired.
    pub pending_release_count:
        Option<unsafe extern "C" fn(ctx: *mut c_void, count_out: *mut u64) -> NxmemStatus>,

    // ─── added at minor 1 ───
    /// Terminal release reporting a structured outcome that can distinguish
    /// complete, quarantined, and failed. Null below minor 1.
    pub release_allocation: Option<
        unsafe extern "C" fn(
            ctx: *mut c_void,
            allocation: *const NxmemAllocation,
            outcome_out: *mut NxmemReleaseOutcome,
        ) -> NxmemStatus,
    >,
}

impl NxmemAllocatorVtable {
    /// Bytes a minor-0 allocator vtable must provide.
    pub const MIN_STRUCT_SIZE_MINOR_0: usize =
        core::mem::offset_of!(Self, pending_release_count) + core::mem::size_of::<usize>();

    /// Bytes a minor-1 allocator vtable must provide.
    pub const MIN_STRUCT_SIZE_MINOR_1: usize =
        core::mem::offset_of!(Self, release_allocation) + core::mem::size_of::<usize>();

    /// The prefix required at `minor`.
    pub const fn required_struct_size(minor: u32) -> usize {
        if minor == 0 {
            Self::MIN_STRUCT_SIZE_MINOR_0
        } else {
            Self::MIN_STRUCT_SIZE_MINOR_1
        }
    }

    /// A zeroed vtable, for a plugin to fill in.
    pub const fn zeroed() -> Self {
        Self {
            struct_size: core::mem::size_of::<Self>() as u32,
            abi_minor: 0,
            mechanism_id: 0,
            device: NxmemDeviceId { tier: 0, index: 0 },
            capability_flags: 0,
            name: core::ptr::null(),
            ctx: core::ptr::null_mut(),
            allocate: None,
            deallocate: None,
            retain: None,
            release: None,
            virtual_backing: core::ptr::null(),
            shared_mapping: core::ptr::null(),
            enqueue_release: None,
            drain_releases: None,
            pending_release_count: None,
            release_allocation: None,
        }
    }

    /// Safely read the readable prefix of an allocator vtable.
    ///
    /// `negotiated_minor` is the ceiling agreed at load time.
    ///
    /// # Safety
    ///
    /// `ptr` must either be null (rejected) or point to a live vtable whose
    /// first `struct_size` bytes are readable for the duration of the call.
    pub unsafe fn read_prefix(
        ptr: *const Self,
        negotiated_minor: u32,
    ) -> Result<Self, NxmemStatus> {
        check_vtable_ptr(ptr, "allocator")?;
        // SAFETY: `ptr` was proved non-null and aligned; `struct_size` is the
        // first field of every version of this struct.
        let declared_size = unsafe { read_struct_size(ptr) } as usize;
        if declared_size < core::mem::offset_of!(Self, allocate) {
            // Too small even to name its own level.
            return Err(short_struct(
                "allocator vtable",
                declared_size,
                Self::required_struct_size(0),
                0,
            ));
        }
        // SAFETY: the size check above proved `abi_minor` (offset 4) is inside
        // the readable prefix.
        let declared_minor = unsafe { core::ptr::read_unaligned(ptr.cast::<u32>().add(1)) };
        // The sender's own claim must be self-consistent: a vtable that says
        // "minor 1" must be at least as big as minor 1 requires, whatever the
        // host negotiated.
        let required = Self::required_struct_size(declared_minor);
        if declared_size < required {
            return Err(short_struct(
                "allocator vtable",
                declared_size,
                required,
                declared_minor,
            ));
        }
        // A sender ahead of the negotiated ceiling is not an error: its struct
        // is a strict superset, so the host reads the prefix it agreed to and
        // clamps the level it will act on. This is what lets a baseline host
        // drive a current plugin.
        let effective_minor = declared_minor.min(negotiated_minor);
        // SAFETY: `declared_size` bytes are readable and we copy no more.
        let mut vtable = unsafe { copy_prefix(ptr, declared_size) };
        vtable.abi_minor = effective_minor;
        // A slot neither side agreed to must never be called, even if both
        // builds happen to know about it.
        if effective_minor < 1 {
            vtable.release_allocation = None;
        }
        vtable.validate_required()?;
        Ok(vtable)
    }

    /// Reject a vtable whose required slots are null.
    ///
    /// A plugin that memsets its vtable to zero must fail here rather than
    /// have the host call through a null pointer.
    pub fn validate_required(&self) -> Result<(), NxmemStatus> {
        for (slot, present) in [
            ("allocate", self.allocate.is_some()),
            ("deallocate", self.deallocate.is_some()),
            ("retain", self.retain.is_some()),
            ("release", self.release.is_some()),
        ] {
            if !present {
                return Err(NxmemStatus::with_message(
                    NxmemStatusCode::ShortStruct,
                    &format!(
                        "nxmem: allocator vtable '{slot}' slot is null but it is required at every \
                         nxmem version"
                    ),
                ));
            }
        }
        if self.mechanism_id == 0 {
            return Err(NxmemStatus::with_message(
                NxmemStatusCode::InvalidArgument,
                "nxmem: allocator vtable mechanism_id must be non-zero so cross-mechanism misuse \
                 can be rejected",
            ));
        }
        Ok(())
    }
}

// ─── Virtual backing vtable ─────────────────────────────────────────────────

/// Lazy reserve/commit/decommit. Optional.
///
/// Terminal release never happens here: it always goes back through the
/// owning [`NxmemAllocatorVtable`].
///
/// # Ownership
///
/// Owned by the plugin and kept alive by the allocator that exposed it. It has
/// no independent reference count; releasing the allocator invalidates it.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct NxmemVirtualBackingVtable {
    /// `size_of` this struct as the plugin defines it.
    pub struct_size: u32,
    /// The nxmem minor level this vtable implements.
    pub abi_minor: u32,
    /// Must equal the owning allocator's `mechanism_id`.
    pub mechanism_id: u64,
    /// Opaque plugin state passed to every slot.
    pub ctx: *mut c_void,
    /// Reserve an allocation, committing only the requested ranges.
    pub allocate_committed: Option<
        unsafe extern "C" fn(
            ctx: *mut c_void,
            request: *const NxmemAllocRequest,
            result_out: *mut NxmemAllocResult,
        ) -> NxmemStatus,
    >,
    /// Commit one span of an existing allocation.
    pub commit_range: Option<
        unsafe extern "C" fn(ctx: *mut c_void, request: *const NxmemRangeRequest) -> NxmemStatus,
    >,
    /// Decommit one span, writing the bytes actually unmapped.
    pub decommit_range: Option<
        unsafe extern "C" fn(
            ctx: *mut c_void,
            request: *const NxmemRangeRequest,
            unmapped_out: *mut u64,
        ) -> NxmemStatus,
    >,
    /// Conservative mapped-byte cost of committing a batch of spans. Only the
    /// mechanism knows its granularity and which spans share granules.
    pub mapped_bytes_for_ranges: Option<
        unsafe extern "C" fn(
            ctx: *mut c_void,
            requests: *const NxmemRangeRequest,
            count: u64,
            mapped_out: *mut u64,
        ) -> NxmemStatus,
    >,
    /// Conservative mapped-byte cost of a whole allocation.
    pub mapped_bytes_for_allocation: Option<
        unsafe extern "C" fn(
            ctx: *mut c_void,
            request: *const NxmemAllocRequest,
            mapped_out: *mut u64,
        ) -> NxmemStatus,
    >,
    /// Bytes currently committed inside one allocation.
    pub committed_bytes: Option<
        unsafe extern "C" fn(
            ctx: *mut c_void,
            allocation: *const NxmemAllocation,
            committed_out: *mut u64,
        ) -> NxmemStatus,
    >,
}

impl NxmemVirtualBackingVtable {
    /// Bytes a minor-0 virtual-backing vtable must provide.
    pub const MIN_STRUCT_SIZE_MINOR_0: usize =
        core::mem::offset_of!(Self, committed_bytes) + core::mem::size_of::<usize>();

    /// A zeroed vtable, for a plugin to fill in.
    pub const fn zeroed() -> Self {
        Self {
            struct_size: core::mem::size_of::<Self>() as u32,
            abi_minor: 0,
            mechanism_id: 0,
            ctx: core::ptr::null_mut(),
            allocate_committed: None,
            commit_range: None,
            decommit_range: None,
            mapped_bytes_for_ranges: None,
            mapped_bytes_for_allocation: None,
            committed_bytes: None,
        }
    }

    /// Safely read the readable prefix of a virtual-backing vtable.
    ///
    /// # Safety
    ///
    /// Same contract as [`NxmemAllocatorVtable::read_prefix`].
    pub unsafe fn read_prefix(
        ptr: *const Self,
        negotiated_minor: u32,
        owner_mechanism_id: u64,
    ) -> Result<Self, NxmemStatus> {
        check_vtable_ptr(ptr, "virtual-backing")?;
        // SAFETY: `ptr` was proved non-null and aligned.
        let declared_size = unsafe { read_struct_size(ptr) } as usize;
        if declared_size < Self::MIN_STRUCT_SIZE_MINOR_0 {
            return Err(short_struct(
                "virtual-backing vtable",
                declared_size,
                Self::MIN_STRUCT_SIZE_MINOR_0,
                0,
            ));
        }
        // SAFETY: the size check proved `abi_minor` is readable.
        let declared_minor = unsafe { core::ptr::read_unaligned(ptr.cast::<u32>().add(1)) };
        // A sender ahead of the negotiated ceiling is not an error: its struct
        // is a strict superset, so the host simply reads the prefix it agreed
        // to and clamps the level it will act on.
        let effective_minor = declared_minor.min(negotiated_minor);
        // SAFETY: `declared_size` bytes are readable and we copy no more.
        let mut vtable = unsafe { copy_prefix(ptr, declared_size) };
        vtable.abi_minor = effective_minor;
        if vtable.mechanism_id != owner_mechanism_id {
            return Err(NxmemStatus::with_message(
                NxmemStatusCode::WrongMechanism,
                &format!(
                    "nxmem: virtual-backing vtable reports mechanism {} but was discovered from \
                     mechanism {owner_mechanism_id}; a capability must belong to the allocator \
                     that exposed it",
                    vtable.mechanism_id
                ),
            ));
        }
        Ok(vtable)
    }
}

// ─── Shared mapping vtable ──────────────────────────────────────────────────

/// Reusable shared physical handles and prefix mapping. Optional and
/// independent of virtual backing.
///
/// # Ownership
///
/// Each prefix produced by `create_shared_prefix` is plugin-owned and must be
/// given back with exactly one `release_shared_prefix` on the same mechanism.
/// `retain_shared_prefix` adds a reference; every retain needs a release.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct NxmemSharedMappingVtable {
    /// `size_of` this struct as the plugin defines it.
    pub struct_size: u32,
    /// The nxmem minor level this vtable implements.
    pub abi_minor: u32,
    /// Must equal the owning allocator's `mechanism_id`.
    pub mechanism_id: u64,
    /// Opaque plugin state passed to every slot.
    pub ctx: *mut c_void,
    /// Create a shared physical prefix.
    pub create_shared_prefix: Option<
        unsafe extern "C" fn(
            ctx: *mut c_void,
            mechanism_id: u64,
            bytes: u64,
            handle_out: *mut NxmemSharedPrefixHandle,
        ) -> NxmemStatus,
    >,
    /// Add a reference to a prefix.
    pub retain_shared_prefix: Option<
        unsafe extern "C" fn(
            ctx: *mut c_void,
            handle: *const NxmemSharedPrefixHandle,
        ) -> NxmemStatus,
    >,
    /// Drop a reference to a prefix. The plugin frees the physical bytes when
    /// the final reference and the final mapping retire.
    pub release_shared_prefix: Option<
        unsafe extern "C" fn(
            ctx: *mut c_void,
            handle: *const NxmemSharedPrefixHandle,
        ) -> NxmemStatus,
    >,
    /// Incremental owned physical cost of admitting one more mapping. A
    /// foreign, wrong-device, or wrong-mechanism prefix must be rejected here
    /// rather than reported as costing zero.
    pub incremental_owned_bytes: Option<
        unsafe extern "C" fn(
            ctx: *mut c_void,
            handle: *const NxmemSharedPrefixHandle,
            bytes_out: *mut u64,
        ) -> NxmemStatus,
    >,
    /// Map a prefix into an allocation.
    pub commit_shared_prefix: Option<
        unsafe extern "C" fn(
            ctx: *mut c_void,
            request: *const NxmemSharedPrefixCommitRequest,
            info_out: *mut NxmemSharedPrefixCommitInfo,
        ) -> NxmemStatus,
    >,
}

impl NxmemSharedMappingVtable {
    /// Bytes a minor-0 shared-mapping vtable must provide.
    pub const MIN_STRUCT_SIZE_MINOR_0: usize =
        core::mem::offset_of!(Self, commit_shared_prefix) + core::mem::size_of::<usize>();

    /// A zeroed vtable, for a plugin to fill in.
    pub const fn zeroed() -> Self {
        Self {
            struct_size: core::mem::size_of::<Self>() as u32,
            abi_minor: 0,
            mechanism_id: 0,
            ctx: core::ptr::null_mut(),
            create_shared_prefix: None,
            retain_shared_prefix: None,
            release_shared_prefix: None,
            incremental_owned_bytes: None,
            commit_shared_prefix: None,
        }
    }

    /// Safely read the readable prefix of a shared-mapping vtable.
    ///
    /// # Safety
    ///
    /// Same contract as [`NxmemAllocatorVtable::read_prefix`].
    pub unsafe fn read_prefix(
        ptr: *const Self,
        negotiated_minor: u32,
        owner_mechanism_id: u64,
    ) -> Result<Self, NxmemStatus> {
        check_vtable_ptr(ptr, "shared-mapping")?;
        // SAFETY: `ptr` was proved non-null and aligned.
        let declared_size = unsafe { read_struct_size(ptr) } as usize;
        if declared_size < Self::MIN_STRUCT_SIZE_MINOR_0 {
            return Err(short_struct(
                "shared-mapping vtable",
                declared_size,
                Self::MIN_STRUCT_SIZE_MINOR_0,
                0,
            ));
        }
        // SAFETY: the size check proved `abi_minor` is readable.
        let declared_minor = unsafe { core::ptr::read_unaligned(ptr.cast::<u32>().add(1)) };
        // A sender ahead of the negotiated ceiling is not an error: its struct
        // is a strict superset, so the host simply reads the prefix it agreed
        // to and clamps the level it will act on.
        let effective_minor = declared_minor.min(negotiated_minor);
        // SAFETY: `declared_size` bytes are readable and we copy no more.
        let mut vtable = unsafe { copy_prefix(ptr, declared_size) };
        vtable.abi_minor = effective_minor;
        if vtable.mechanism_id != owner_mechanism_id {
            return Err(NxmemStatus::with_message(
                NxmemStatusCode::WrongMechanism,
                &format!(
                    "nxmem: shared-mapping vtable reports mechanism {} but was discovered from \
                     mechanism {owner_mechanism_id}; a capability must belong to the allocator \
                     that exposed it",
                    vtable.mechanism_id
                ),
            ));
        }
        Ok(vtable)
    }
}

// ─── Factory vtable ─────────────────────────────────────────────────────────

/// One named allocator mechanism a plugin can open.
///
/// # Ownership
///
/// The host owns every factory returned by `NxmemCreateAllocatorFactories` and
/// must call `release` on each exactly once. A factory must stay usable until
/// its own `release`, and releasing a factory must not invalidate allocators
/// already opened from it.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct NxmemAllocatorFactoryVtable {
    /// `size_of` this struct as the plugin defines it.
    pub struct_size: u32,
    /// The nxmem minor level this vtable implements.
    pub abi_minor: u32,
    /// NUL-terminated UTF-8 name, valid until `release` returns. Names the
    /// mechanism, and is how a host selects one of several.
    pub name: *const u8,
    /// The device this factory serves.
    pub device: NxmemDeviceId,
    /// Capabilities the opened allocator will provide.
    pub capability_flags: u64,
    /// Opaque plugin state passed to every slot.
    pub ctx: *mut c_void,
    /// Open an allocator instance. Required.
    pub open_allocator: Option<
        unsafe extern "C" fn(
            ctx: *mut c_void,
            request: *const NxmemOpenRequest,
            allocator_out: *mut *const NxmemAllocatorVtable,
        ) -> NxmemStatus,
    >,
    /// Give the factory back. Required.
    pub release: Option<unsafe extern "C" fn(ctx: *mut c_void)>,
}

impl NxmemAllocatorFactoryVtable {
    /// Bytes a minor-0 factory vtable must provide.
    pub const MIN_STRUCT_SIZE_MINOR_0: usize =
        core::mem::offset_of!(Self, release) + core::mem::size_of::<usize>();

    /// A zeroed vtable, for a plugin to fill in.
    pub const fn zeroed() -> Self {
        Self {
            struct_size: core::mem::size_of::<Self>() as u32,
            abi_minor: 0,
            name: core::ptr::null(),
            device: NxmemDeviceId { tier: 0, index: 0 },
            capability_flags: 0,
            ctx: core::ptr::null_mut(),
            open_allocator: None,
            release: None,
        }
    }

    /// Safely read the readable prefix of a factory vtable.
    ///
    /// # Safety
    ///
    /// Same contract as [`NxmemAllocatorVtable::read_prefix`].
    pub unsafe fn read_prefix(
        ptr: *const Self,
        negotiated_minor: u32,
    ) -> Result<Self, NxmemStatus> {
        check_vtable_ptr(ptr, "allocator-factory")?;
        // SAFETY: `ptr` was proved non-null and aligned.
        let declared_size = unsafe { read_struct_size(ptr) } as usize;
        if declared_size < Self::MIN_STRUCT_SIZE_MINOR_0 {
            return Err(short_struct(
                "allocator-factory vtable",
                declared_size,
                Self::MIN_STRUCT_SIZE_MINOR_0,
                0,
            ));
        }
        // SAFETY: the size check proved `abi_minor` is readable.
        let declared_minor = unsafe { core::ptr::read_unaligned(ptr.cast::<u32>().add(1)) };
        // A sender ahead of the negotiated ceiling is not an error: its struct
        // is a strict superset, so the host simply reads the prefix it agreed
        // to and clamps the level it will act on.
        let effective_minor = declared_minor.min(negotiated_minor);
        // SAFETY: `declared_size` bytes are readable and we copy no more.
        let mut vtable = unsafe { copy_prefix(ptr, declared_size) };
        vtable.abi_minor = effective_minor;
        if vtable.open_allocator.is_none() || vtable.release.is_none() {
            return Err(NxmemStatus::with_message(
                NxmemStatusCode::ShortStruct,
                "nxmem: allocator-factory vtable must provide both 'open_allocator' and 'release'",
            ));
        }
        Ok(vtable)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    unsafe extern "C" fn stub_allocate(
        _ctx: *mut c_void,
        _request: *const NxmemAllocRequest,
        _result: *mut NxmemAllocResult,
    ) -> NxmemStatus {
        NxmemStatus::ok()
    }

    unsafe extern "C" fn stub_deallocate(
        _ctx: *mut c_void,
        _allocation: *const NxmemAllocation,
        _unmapped: *mut u64,
    ) -> NxmemStatus {
        NxmemStatus::ok()
    }

    unsafe extern "C" fn stub_void(_ctx: *mut c_void) {}

    unsafe extern "C" fn stub_release_allocation(
        _ctx: *mut c_void,
        _allocation: *const NxmemAllocation,
        _outcome: *mut NxmemReleaseOutcome,
    ) -> NxmemStatus {
        NxmemStatus::ok()
    }

    fn conforming_allocator(minor: u32) -> NxmemAllocatorVtable {
        NxmemAllocatorVtable {
            struct_size: NxmemAllocatorVtable::required_struct_size(minor) as u32,
            abi_minor: minor,
            mechanism_id: 42,
            device: NxmemDeviceId::HOST,
            capability_flags: 1,
            allocate: Some(stub_allocate),
            deallocate: Some(stub_deallocate),
            retain: Some(stub_void),
            release: Some(stub_void),
            release_allocation: Some(stub_release_allocation),
            ..NxmemAllocatorVtable::zeroed()
        }
    }

    #[test]
    fn the_minor_1_prefix_is_larger_than_the_minor_0_prefix() {
        // Compile-time, because a regression here is a layout bug that must
        // not be allowed to link, let alone run.
        const _: () = assert!(
            NxmemAllocatorVtable::MIN_STRUCT_SIZE_MINOR_1
                > NxmemAllocatorVtable::MIN_STRUCT_SIZE_MINOR_0,
            "minor 1 must require a longer prefix than minor 0"
        );
        assert_eq!(
            NxmemAllocatorVtable::MIN_STRUCT_SIZE_MINOR_1,
            core::mem::size_of::<NxmemAllocatorVtable>()
        );
    }

    #[test]
    fn a_current_vtable_reads_back_intact() {
        let vtable = conforming_allocator(1);
        // SAFETY: `vtable` is a live, aligned local of exactly this type.
        let read = unsafe { NxmemAllocatorVtable::read_prefix(&vtable, 1) }.expect("valid vtable");
        assert_eq!(read.mechanism_id, 42);
        assert!(read.release_allocation.is_some());
    }

    #[test]
    fn a_minor_0_vtable_reads_back_without_the_minor_1_slot() {
        let mut vtable = conforming_allocator(0);
        // A baseline plugin does not even define the trailing slot; simulate
        // that by declaring the smaller size while leaving stale bytes behind.
        vtable.struct_size = NxmemAllocatorVtable::MIN_STRUCT_SIZE_MINOR_0 as u32;
        // SAFETY: `vtable` is a live, aligned local of exactly this type, and
        // we only ask to read the smaller declared prefix.
        let read = unsafe { NxmemAllocatorVtable::read_prefix(&vtable, 1) }.expect("valid vtable");
        assert_eq!(read.abi_minor, 0);
        assert!(
            read.release_allocation.is_none(),
            "a slot the sender does not define must never be called"
        );
        assert!(read.allocate.is_some(), "the baseline prefix survives");
    }

    #[test]
    fn a_short_vtable_is_refused() {
        let mut vtable = conforming_allocator(0);
        vtable.struct_size = (NxmemAllocatorVtable::MIN_STRUCT_SIZE_MINOR_0 - 1) as u32;
        // SAFETY: `vtable` is a live, aligned local; `read_prefix` reads only
        // `struct_size` and `abi_minor` before rejecting.
        let status =
            unsafe { NxmemAllocatorVtable::read_prefix(&vtable, 1) }.expect_err("short struct");
        assert_eq!(status.status_code(), Some(NxmemStatusCode::ShortStruct));
        assert!(status.describe().contains("rebuild the plugin"));
    }

    #[test]
    fn a_vtable_above_the_negotiated_ceiling_is_clamped_not_refused() {
        let vtable = conforming_allocator(1);
        assert!(
            vtable.release_allocation.is_some(),
            "the sender really does implement the minor-1 slot"
        );
        // SAFETY: `vtable` is a live, aligned local.
        let read = unsafe { NxmemAllocatorVtable::read_prefix(&vtable, 0) }
            .expect("a newer sender is a strict superset, so a baseline host can still read it");
        assert_eq!(
            read.abi_minor, 0,
            "the effective level is clamped to what both sides agreed"
        );
        assert!(
            read.release_allocation.is_none(),
            "a slot neither side agreed to must never be callable"
        );
        assert!(
            read.allocate.is_some() && read.deallocate.is_some(),
            "the baseline prefix still reads back intact"
        );
    }

    #[test]
    fn a_vtable_smaller_than_the_level_it_claims_is_refused() {
        let mut vtable = conforming_allocator(1);
        // The sender contradicts itself: it says "minor 1" but sizes itself
        // for the baseline. That is not an old peer, it is a broken one.
        vtable.struct_size = NxmemAllocatorVtable::MIN_STRUCT_SIZE_MINOR_0 as u32;
        // SAFETY: `vtable` is a live, aligned local.
        let status = unsafe { NxmemAllocatorVtable::read_prefix(&vtable, 1) }
            .expect_err("a self-contradicting struct_size must be refused");
        assert_eq!(status.status_code(), Some(NxmemStatusCode::ShortStruct));
        assert!(
            status.describe().contains("minor 1"),
            "the refusal must name the level the sender claimed: {}",
            status.describe()
        );
    }

    #[test]
    fn a_zeroed_vtable_never_produces_a_null_call() {
        let mut vtable = NxmemAllocatorVtable::zeroed();
        vtable.mechanism_id = 1;
        // SAFETY: `vtable` is a live, aligned local.
        let status = unsafe { NxmemAllocatorVtable::read_prefix(&vtable, 1) }
            .expect_err("required slots are null");
        assert_eq!(status.status_code(), Some(NxmemStatusCode::ShortStruct));
        assert!(status.describe().contains("'allocate' slot is null"));
    }

    #[test]
    fn a_zero_mechanism_id_is_refused() {
        let mut vtable = conforming_allocator(1);
        vtable.mechanism_id = 0;
        // SAFETY: `vtable` is a live, aligned local.
        let status = unsafe { NxmemAllocatorVtable::read_prefix(&vtable, 1) }
            .expect_err("mechanism_id must be non-zero");
        assert_eq!(status.status_code(), Some(NxmemStatusCode::InvalidArgument));
    }

    #[test]
    fn a_null_vtable_pointer_is_refused() {
        // SAFETY: a null pointer is exactly what this test exercises.
        let status = unsafe { NxmemAllocatorVtable::read_prefix(core::ptr::null(), 1) }
            .expect_err("null pointer");
        assert_eq!(status.status_code(), Some(NxmemStatusCode::InvalidArgument));
    }

    #[test]
    fn a_misaligned_vtable_pointer_is_refused() {
        let vtable = conforming_allocator(1);
        let misaligned =
            (&vtable as *const NxmemAllocatorVtable as usize + 1) as *const NxmemAllocatorVtable;
        // SAFETY: `read_prefix` rejects on alignment before dereferencing.
        let status =
            unsafe { NxmemAllocatorVtable::read_prefix(misaligned, 1) }.expect_err("misaligned");
        assert_eq!(status.status_code(), Some(NxmemStatusCode::InvalidArgument));
    }

    #[test]
    fn a_capability_vtable_from_a_foreign_mechanism_is_refused() {
        let backing = NxmemVirtualBackingVtable {
            mechanism_id: 999,
            ..NxmemVirtualBackingVtable::zeroed()
        };
        // SAFETY: `backing` is a live, aligned local.
        let status = unsafe { NxmemVirtualBackingVtable::read_prefix(&backing, 1, 42) }
            .expect_err("mechanism mismatch");
        assert_eq!(status.status_code(), Some(NxmemStatusCode::WrongMechanism));

        let shared = NxmemSharedMappingVtable {
            mechanism_id: 999,
            ..NxmemSharedMappingVtable::zeroed()
        };
        // SAFETY: `shared` is a live, aligned local.
        let status = unsafe { NxmemSharedMappingVtable::read_prefix(&shared, 1, 42) }
            .expect_err("mechanism mismatch");
        assert_eq!(status.status_code(), Some(NxmemStatusCode::WrongMechanism));
    }

    #[test]
    fn a_factory_missing_a_required_slot_is_refused() {
        let factory = NxmemAllocatorFactoryVtable::zeroed();
        // SAFETY: `factory` is a live, aligned local.
        let status = unsafe { NxmemAllocatorFactoryVtable::read_prefix(&factory, 1) }
            .expect_err("required slots are null");
        assert_eq!(status.status_code(), Some(NxmemStatusCode::ShortStruct));
    }

    /// A newer plugin sends a larger struct. This build must ignore the extra
    /// bytes rather than reject, and must not read them.
    #[test]
    fn a_larger_struct_from_a_newer_peer_is_tolerated() {
        #[repr(C)]
        struct Extended {
            base: NxmemAllocatorVtable,
            future_slot: usize,
        }
        let extended = Extended {
            base: NxmemAllocatorVtable {
                struct_size: core::mem::size_of::<Extended>() as u32,
                ..conforming_allocator(1)
            },
            future_slot: 0xdead_beef,
        };
        // SAFETY: the pointer names a live `Extended` whose leading bytes are
        // a valid `NxmemAllocatorVtable`; `read_prefix` copies at most
        // `size_of::<NxmemAllocatorVtable>()` bytes.
        let read = unsafe {
            NxmemAllocatorVtable::read_prefix(
                (&extended as *const Extended).cast::<NxmemAllocatorVtable>(),
                1,
            )
        }
        .expect("a larger struct at the same major is compatible");
        assert_eq!(read.mechanism_id, 42);
        assert!(read.release_allocation.is_some());
    }
}
