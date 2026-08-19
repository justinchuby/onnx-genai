//! ABI version negotiation and capability flags.
//!
//! # Version semantics
//!
//! * **Major** — breaking change. Host and plugin must agree on one major
//!   version. Any mismatch is a hard rejection.
//! * **Minor** — additive change only: new trailing vtable slots and new
//!   capability bits. The agreed minor is `min(host_minor_max,
//!   plugin_minor_max)`. Both sides then behave as if only the slots defined
//!   at or below the agreed minor exist.
//!
//! # Prefix negotiation
//!
//! Every ABI struct starts with `struct_size: u32`. A participant fills in
//! `size_of` its own definition. The reader compares that against the minimum
//! prefix it requires:
//!
//! * `struct_size` **smaller than the required prefix** for the negotiated
//!   minor → [`crate::NxmemStatusCode::ShortStruct`], fail closed. Never read
//!   the missing bytes.
//! * `struct_size` **larger** than the reader knows about → the reader ignores
//!   the trailing bytes. That is how an older host tolerates a newer plugin at
//!   the same major version.
//!
//! Optional slots are nullable function pointers. Null means "not supported"
//! and must be surfaced as [`crate::NxmemStatusCode::UnsupportedCapability`],
//! never as a successful no-op.
//!
//! # Baseline
//!
//! Minor `0` is the baseline prefix. Minor `1` appends the structured-release
//! slot to the allocator vtable. A minor-0 plugin therefore reports a smaller
//! `struct_size` and the host falls back to the unstructured release path;
//! that is the supported-older-participant case, not an error.

use crate::status::{NxmemStatus, NxmemStatusCode};

/// Current major version of the `nxmem` ABI.
pub const NXMEM_ABI_VERSION_MAJOR: u32 = 1;

/// Current minor version of the `nxmem` ABI.
pub const NXMEM_ABI_VERSION_MINOR: u32 = 1;

/// The oldest minor version this build still supports as a peer.
///
/// A participant reporting this minor uses only the baseline prefix of every
/// struct and none of the slots added after it.
pub const NXMEM_ABI_VERSION_MINOR_BASELINE: u32 = 0;

/// An inclusive version range a participant supports.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NxmemVersionRange {
    /// Lowest supported major version (inclusive).
    pub major_min: u32,
    /// Highest supported major version (inclusive).
    pub major_max: u32,
    /// Lowest supported minor version at `major_max` (inclusive).
    pub minor_min: u32,
    /// Highest supported minor version at `major_max` (inclusive).
    pub minor_max: u32,
}

impl NxmemVersionRange {
    /// The range this build supports.
    pub const fn current() -> Self {
        Self {
            major_min: NXMEM_ABI_VERSION_MAJOR,
            major_max: NXMEM_ABI_VERSION_MAJOR,
            minor_min: NXMEM_ABI_VERSION_MINOR_BASELINE,
            minor_max: NXMEM_ABI_VERSION_MINOR,
        }
    }

    /// A range pinned to exactly one `major.minor`, for tests and for hosts
    /// that deliberately restrict themselves.
    pub const fn exact(major: u32, minor: u32) -> Self {
        Self {
            major_min: major,
            major_max: major,
            minor_min: minor,
            minor_max: minor,
        }
    }
}

/// What the host sends to `NxmemNegotiate`.
///
/// # Ownership
///
/// Borrowed by the plugin for the duration of the call only. The plugin must
/// not retain the pointer.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct NxmemNegotiateRequest {
    /// `size_of` this struct as the host defines it.
    pub struct_size: u32,
    /// Reserved; must be zero.
    pub reserved: u32,
    /// The versions the host can speak.
    pub host_range: NxmemVersionRange,
    /// Capabilities the host is prepared to consume. A plugin must not
    /// advertise a capability the host did not offer.
    pub host_capability_flags: u64,
}

impl NxmemNegotiateRequest {
    /// A request advertising this build's range and every known capability.
    pub const fn current() -> Self {
        Self {
            struct_size: core::mem::size_of::<Self>() as u32,
            reserved: 0,
            host_range: NxmemVersionRange::current(),
            host_capability_flags: NXMEM_CAP_KNOWN_MASK,
        }
    }

    /// A request advertising a specific range, used by hosts that pin an older
    /// contract and by negotiation tests.
    pub const fn with_range(host_range: NxmemVersionRange) -> Self {
        Self {
            struct_size: core::mem::size_of::<Self>() as u32,
            reserved: 0,
            host_range,
            host_capability_flags: NXMEM_CAP_KNOWN_MASK,
        }
    }
}

/// What the plugin writes back from `NxmemNegotiate`.
///
/// # Ownership
///
/// The host owns the buffer and it is valid only for the call.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct NxmemNegotiateResponse {
    /// `size_of` this struct as the plugin defines it.
    pub struct_size: u32,
    /// Agreed major version. Meaningful only when the status is `Ok`.
    pub agreed_major: u32,
    /// Agreed minor version. Meaningful only when the status is `Ok`.
    pub agreed_minor: u32,
    /// Reserved; must be zero.
    pub reserved: u32,
    /// The plugin's own range. Always filled, including on failure, so the
    /// host can produce an actionable diagnostic.
    pub plugin_range: NxmemVersionRange,
    /// Capabilities the plugin offers at the agreed version. Always a subset
    /// of the host's offered flags and of [`NXMEM_CAP_KNOWN_MASK`].
    pub capability_flags: u64,
}

impl NxmemNegotiateResponse {
    /// A zeroed response for the host to pass in.
    pub const fn zeroed() -> Self {
        Self {
            struct_size: core::mem::size_of::<Self>() as u32,
            agreed_major: 0,
            agreed_minor: 0,
            reserved: 0,
            plugin_range: NxmemVersionRange {
                major_min: 0,
                major_max: 0,
                minor_min: 0,
                minor_max: 0,
            },
            capability_flags: 0,
        }
    }
}

// ─── Capability flags ───────────────────────────────────────────────────────

/// Ordinary allocation and terminal release. Every plugin must set this bit;
/// it is the minimum contract, not an option.
pub const NXMEM_CAP_ALLOCATOR: u64 = 1 << 0;

/// Optional lazy reserve/commit/decommit (`VirtualBacking`).
pub const NXMEM_CAP_VIRTUAL_BACKING: u64 = 1 << 1;

/// Optional shared physical handles and prefix mapping (`SharedMapping`).
pub const NXMEM_CAP_SHARED_MAPPING: u64 = 1 << 2;

/// Optional stream-ordered deferred release with host completion callbacks.
pub const NXMEM_CAP_DEFERRED_RELEASE: u64 = 1 << 3;

/// Optional structured release outcome (added at minor 1).
pub const NXMEM_CAP_STRUCTURED_RELEASE: u64 = 1 << 4;

/// Every capability bit known at this ABI version.
///
/// A participant that sets a bit outside this mask is rejected: the host
/// cannot reason about a capability it has never heard of.
pub const NXMEM_CAP_KNOWN_MASK: u64 = NXMEM_CAP_ALLOCATOR
    | NXMEM_CAP_VIRTUAL_BACKING
    | NXMEM_CAP_SHARED_MAPPING
    | NXMEM_CAP_DEFERRED_RELEASE
    | NXMEM_CAP_STRUCTURED_RELEASE;

/// The minor version at which each capability bit was introduced.
///
/// A capability may only be advertised once the agreed minor is at least this
/// value, otherwise the vtable prefix carrying it does not exist.
pub const fn capability_min_minor(flag: u64) -> u32 {
    match flag {
        NXMEM_CAP_STRUCTURED_RELEASE => 1,
        _ => 0,
    }
}

/// Human-facing names for capability bits, used in diagnostics.
pub fn describe_capabilities(flags: u64) -> String {
    const NAMES: [(u64, &str); 5] = [
        (NXMEM_CAP_ALLOCATOR, "allocator"),
        (NXMEM_CAP_VIRTUAL_BACKING, "virtual-backing"),
        (NXMEM_CAP_SHARED_MAPPING, "shared-mapping"),
        (NXMEM_CAP_DEFERRED_RELEASE, "deferred-release"),
        (NXMEM_CAP_STRUCTURED_RELEASE, "structured-release"),
    ];
    let mut parts: Vec<&str> = NAMES
        .iter()
        .filter(|(bit, _)| flags & bit != 0)
        .map(|(_, name)| *name)
        .collect();
    let unknown = flags & !NXMEM_CAP_KNOWN_MASK;
    let unknown_text;
    if unknown != 0 {
        unknown_text = format!("unknown({unknown:#x})");
        parts.push(&unknown_text);
    }
    if parts.is_empty() {
        return String::from("none");
    }
    parts.join(", ")
}

// ─── Plugin-side negotiation ────────────────────────────────────────────────

/// The plugin half of the handshake.
///
/// A plugin's `NxmemNegotiate` export can delegate here when it speaks the
/// version this crate was compiled for. A plugin that deliberately speaks an
/// older minor calls [`negotiate_as`] instead.
///
/// # Safety
///
/// `request` and `response_out` must be valid, non-null, and correctly aligned
/// for the duration of the call.
pub unsafe fn negotiate(
    request: *const NxmemNegotiateRequest,
    response_out: *mut NxmemNegotiateResponse,
) -> NxmemStatus {
    // SAFETY: delegated unchanged to this function's contract.
    unsafe {
        negotiate_as(
            request,
            response_out,
            NxmemVersionRange::current(),
            NXMEM_CAP_KNOWN_MASK,
        )
    }
}

/// The plugin half of the handshake for a participant that advertises
/// `plugin_range` and `plugin_capabilities` rather than this build's own.
///
/// This is what an intentionally-older plugin (or a compatibility fixture)
/// uses. It is the same production logic; only the advertised range differs.
///
/// # Safety
///
/// `request` and `response_out` must be valid, non-null, and correctly aligned
/// for the duration of the call.
pub unsafe fn negotiate_as(
    request: *const NxmemNegotiateRequest,
    response_out: *mut NxmemNegotiateResponse,
    plugin_range: NxmemVersionRange,
    plugin_capabilities: u64,
) -> NxmemStatus {
    if request.is_null() || response_out.is_null() {
        return NxmemStatus::with_message(
            NxmemStatusCode::InvalidArgument,
            "NxmemNegotiate: request and response pointers must be non-null",
        );
    }

    // SAFETY: both pointers were checked for null and the caller guarantees
    // validity and alignment.
    let response = unsafe { &mut *response_out };
    *response = NxmemNegotiateResponse::zeroed();
    response.plugin_range = plugin_range;

    // The request must be at least large enough to contain the fields we read.
    // SAFETY: `request` is non-null and valid per this function's contract;
    // `struct_size` is the first field of the baseline prefix, which every
    // version of this struct has.
    let request_size = unsafe { (*request).struct_size } as usize;
    if request_size < core::mem::size_of::<NxmemNegotiateRequest>() {
        return NxmemStatus::with_message(
            NxmemStatusCode::ShortStruct,
            &format!(
                "NxmemNegotiate: host request struct_size {request_size} is smaller than the \
                 required {} bytes; rebuild the host against a matching nxmem ABI header",
                core::mem::size_of::<NxmemNegotiateRequest>()
            ),
        );
    }
    // SAFETY: the size check above proved the struct contains every field we
    // are about to read.
    let request = unsafe { &*request };

    let major = plugin_range.major_max;
    if major < request.host_range.major_min || major > request.host_range.major_max {
        return NxmemStatus::with_message(
            NxmemStatusCode::VersionMismatch,
            &format!(
                "NxmemNegotiate: plugin speaks nxmem major {major} but the host accepts only \
                 [{}, {}]; rebuild the plugin against the host's nxmem ABI major version",
                request.host_range.major_min, request.host_range.major_max
            ),
        );
    }

    let agreed_minor = plugin_range.minor_max.min(request.host_range.minor_max);
    if agreed_minor < plugin_range.minor_min || agreed_minor < request.host_range.minor_min {
        return NxmemStatus::with_message(
            NxmemStatusCode::VersionMismatch,
            &format!(
                "NxmemNegotiate: no common nxmem minor version; plugin supports [{}, {}] and the \
                 host supports [{}, {}]",
                plugin_range.minor_min,
                plugin_range.minor_max,
                request.host_range.minor_min,
                request.host_range.minor_max
            ),
        );
    }

    // Only offer what the host asked for, what this build knows, and what the
    // agreed minor actually defines a prefix for.
    let mut offered = plugin_capabilities & request.host_capability_flags & NXMEM_CAP_KNOWN_MASK;
    for flag in [
        NXMEM_CAP_ALLOCATOR,
        NXMEM_CAP_VIRTUAL_BACKING,
        NXMEM_CAP_SHARED_MAPPING,
        NXMEM_CAP_DEFERRED_RELEASE,
        NXMEM_CAP_STRUCTURED_RELEASE,
    ] {
        if agreed_minor < capability_min_minor(flag) {
            offered &= !flag;
        }
    }

    if offered & NXMEM_CAP_ALLOCATOR == 0 {
        return NxmemStatus::with_message(
            NxmemStatusCode::UnsupportedCapability,
            "NxmemNegotiate: the ordinary allocator capability is mandatory and was not agreed",
        );
    }

    response.agreed_major = major;
    response.agreed_minor = agreed_minor;
    response.capability_flags = offered;
    NxmemStatus::ok()
}

// ─── Host-side validation ───────────────────────────────────────────────────

/// Why a host refused a negotiation response.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NegotiationRejection {
    /// A stable, actionable explanation.
    pub reason: String,
}

impl core::fmt::Display for NegotiationRejection {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(&self.reason)
    }
}

/// Host-side validation of a plugin's negotiation response.
///
/// Fails closed on: a major outside the host range, a minor above what the
/// host can speak, a minor below the host's floor, unknown capability bits, a
/// missing mandatory allocator capability, and a capability advertised at a
/// minor that does not define it.
pub fn validate_negotiation(
    host_range: &NxmemVersionRange,
    response: &NxmemNegotiateResponse,
) -> Result<(), NegotiationRejection> {
    let reject = |reason: String| Err(NegotiationRejection { reason });

    if response.agreed_major < host_range.major_min || response.agreed_major > host_range.major_max
    {
        return reject(format!(
            "plugin agreed nxmem major {} but this host accepts only [{}, {}]",
            response.agreed_major, host_range.major_min, host_range.major_max
        ));
    }
    if response.agreed_minor > host_range.minor_max {
        return reject(format!(
            "plugin agreed nxmem minor {} but this host speaks at most minor {}; the host cannot \
             call vtable slots it does not know",
            response.agreed_minor, host_range.minor_max
        ));
    }
    if response.agreed_minor < host_range.minor_min {
        return reject(format!(
            "plugin agreed nxmem minor {} but this host no longer supports anything below minor \
             {}",
            response.agreed_minor, host_range.minor_min
        ));
    }
    let unknown = response.capability_flags & !NXMEM_CAP_KNOWN_MASK;
    if unknown != 0 {
        return reject(format!(
            "plugin advertises unknown nxmem capability bits {unknown:#x}; refusing to load \
             (fail closed)"
        ));
    }
    if response.capability_flags & NXMEM_CAP_ALLOCATOR == 0 {
        return reject(String::from(
            "plugin does not advertise the mandatory nxmem allocator capability",
        ));
    }
    for flag in [
        NXMEM_CAP_VIRTUAL_BACKING,
        NXMEM_CAP_SHARED_MAPPING,
        NXMEM_CAP_DEFERRED_RELEASE,
        NXMEM_CAP_STRUCTURED_RELEASE,
    ] {
        if response.capability_flags & flag != 0 && response.agreed_minor < capability_min_minor(flag)
        {
            return reject(format!(
                "plugin advertises capability {} at nxmem minor {} but that capability is only \
                 defined from minor {}",
                describe_capabilities(flag),
                response.agreed_minor,
                capability_min_minor(flag)
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn negotiate_current(request: &NxmemNegotiateRequest) -> (NxmemStatus, NxmemNegotiateResponse) {
        let mut response = NxmemNegotiateResponse::zeroed();
        // SAFETY: both pointers reference live, aligned locals.
        let status = unsafe { negotiate(request, &mut response) };
        (status, response)
    }

    #[test]
    fn matching_versions_agree_on_the_current_minor() {
        let (status, response) = negotiate_current(&NxmemNegotiateRequest::current());
        assert!(status.is_ok(), "{}", status.describe());
        assert_eq!(response.agreed_major, NXMEM_ABI_VERSION_MAJOR);
        assert_eq!(response.agreed_minor, NXMEM_ABI_VERSION_MINOR);
        assert!(response.capability_flags & NXMEM_CAP_ALLOCATOR != 0);
        assert!(validate_negotiation(&NxmemVersionRange::current(), &response).is_ok());
    }

    #[test]
    fn a_major_outside_the_host_range_is_rejected_with_a_fix() {
        let request = NxmemNegotiateRequest::with_range(NxmemVersionRange::exact(99, 0));
        let (status, response) = negotiate_current(&request);
        assert_eq!(status.status_code(), Some(NxmemStatusCode::VersionMismatch));
        assert!(status.describe().contains("rebuild the plugin"));
        // The plugin's own range is reported even on failure, for diagnostics.
        assert_eq!(response.plugin_range, NxmemVersionRange::current());
    }

    #[test]
    fn an_older_host_pins_the_agreed_minor_down() {
        let request = NxmemNegotiateRequest::with_range(NxmemVersionRange {
            major_min: NXMEM_ABI_VERSION_MAJOR,
            major_max: NXMEM_ABI_VERSION_MAJOR,
            minor_min: 0,
            minor_max: 0,
        });
        let (status, response) = negotiate_current(&request);
        assert!(status.is_ok(), "{}", status.describe());
        assert_eq!(response.agreed_minor, 0);
        assert_eq!(
            response.capability_flags & NXMEM_CAP_STRUCTURED_RELEASE,
            0,
            "a capability defined at minor 1 must not be offered at minor 0"
        );
    }

    #[test]
    fn an_older_plugin_is_accepted_by_the_current_host() {
        let request = NxmemNegotiateRequest::current();
        let mut response = NxmemNegotiateResponse::zeroed();
        // SAFETY: both pointers reference live, aligned locals.
        let status = unsafe {
            negotiate_as(
                &request,
                &mut response,
                NxmemVersionRange::exact(NXMEM_ABI_VERSION_MAJOR, 0),
                NXMEM_CAP_ALLOCATOR | NXMEM_CAP_VIRTUAL_BACKING,
            )
        };
        assert!(status.is_ok(), "{}", status.describe());
        assert_eq!(response.agreed_minor, 0);
        assert!(validate_negotiation(&NxmemVersionRange::current(), &response).is_ok());
    }

    #[test]
    fn a_short_request_struct_is_refused_before_any_field_is_read() {
        let mut request = NxmemNegotiateRequest::current();
        request.struct_size = 4;
        let (status, _) = negotiate_current(&request);
        assert_eq!(status.status_code(), Some(NxmemStatusCode::ShortStruct));
    }

    #[test]
    fn null_pointers_fail_closed() {
        let mut response = NxmemNegotiateResponse::zeroed();
        // SAFETY: a null request is exactly what this test exercises; the
        // response pointer is a live local.
        let status = unsafe { negotiate(core::ptr::null(), &mut response) };
        assert_eq!(status.status_code(), Some(NxmemStatusCode::InvalidArgument));
    }

    #[test]
    fn a_plugin_may_not_advertise_a_capability_the_host_did_not_offer() {
        let mut request = NxmemNegotiateRequest::current();
        request.host_capability_flags = NXMEM_CAP_ALLOCATOR;
        let (status, response) = negotiate_current(&request);
        assert!(status.is_ok(), "{}", status.describe());
        assert_eq!(response.capability_flags, NXMEM_CAP_ALLOCATOR);
    }

    #[test]
    fn unknown_capability_bits_are_rejected_by_the_host() {
        let mut response = NxmemNegotiateResponse::zeroed();
        response.agreed_major = NXMEM_ABI_VERSION_MAJOR;
        response.agreed_minor = NXMEM_ABI_VERSION_MINOR;
        response.capability_flags = NXMEM_CAP_ALLOCATOR | (1 << 62);
        let rejection = validate_negotiation(&NxmemVersionRange::current(), &response)
            .expect_err("unknown bits must fail closed");
        assert!(rejection.reason.contains("unknown nxmem capability bits"));
    }

    #[test]
    fn a_missing_allocator_capability_is_rejected_by_the_host() {
        let mut response = NxmemNegotiateResponse::zeroed();
        response.agreed_major = NXMEM_ABI_VERSION_MAJOR;
        response.agreed_minor = NXMEM_ABI_VERSION_MINOR;
        response.capability_flags = NXMEM_CAP_VIRTUAL_BACKING;
        let rejection = validate_negotiation(&NxmemVersionRange::current(), &response)
            .expect_err("the allocator capability is mandatory");
        assert!(rejection.reason.contains("mandatory"));
    }

    #[test]
    fn a_capability_advertised_below_its_minor_is_rejected() {
        let mut response = NxmemNegotiateResponse::zeroed();
        response.agreed_major = NXMEM_ABI_VERSION_MAJOR;
        response.agreed_minor = 0;
        response.capability_flags = NXMEM_CAP_ALLOCATOR | NXMEM_CAP_STRUCTURED_RELEASE;
        let rejection = validate_negotiation(&NxmemVersionRange::current(), &response)
            .expect_err("structured release does not exist at minor 0");
        assert!(
            rejection.reason.contains("structured-release"),
            "the rejection must name the offending capability, got {:?}",
            rejection.reason
        );
        assert!(
            rejection.reason.contains("defined from minor"),
            "the rejection must explain the minor floor, got {:?}",
            rejection.reason
        );
    }

    #[test]
    fn a_newer_plugin_minor_is_rejected_by_an_older_host() {
        let mut response = NxmemNegotiateResponse::zeroed();
        response.agreed_major = NXMEM_ABI_VERSION_MAJOR;
        response.agreed_minor = NXMEM_ABI_VERSION_MINOR + 5;
        response.capability_flags = NXMEM_CAP_ALLOCATOR;
        let rejection = validate_negotiation(&NxmemVersionRange::current(), &response)
            .expect_err("a host cannot call slots it does not know");
        assert!(rejection.reason.contains("does not know"));
    }

    #[test]
    fn capability_names_are_stable_and_report_unknown_bits() {
        assert_eq!(describe_capabilities(0), "none");
        assert_eq!(
            describe_capabilities(NXMEM_CAP_ALLOCATOR | NXMEM_CAP_SHARED_MAPPING),
            "allocator, shared-mapping"
        );
        assert!(describe_capabilities(1 << 62).contains("unknown(0x4000000000000000)"));
    }
}
