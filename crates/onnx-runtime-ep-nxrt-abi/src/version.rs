//! ABI version negotiation.
//!
//! # Semantics
//!
//! - **Major version**: breaking changes. Host and plugin must share the same
//!   major version. A mismatch is a hard rejection (fail closed).
//! - **Minor version**: additive changes (new vtable fields at the end, new
//!   capability flags). A host with minor N can load a plugin with minor M
//!   where M ≤ N (the host is at least as new). If the plugin is newer
//!   (M > N), the host rejects — it cannot safely call unknown vtable slots.
//!
//! # Forward compatibility mechanism
//!
//! Every vtable struct carries a `struct_size: u32` field as its first member.
//! A host only reads fields up to min(its_known_size, reported_size). New
//! fields are appended at the end and guarded by a minor version bump. An
//! older host seeing a larger struct ignores trailing bytes. A newer host
//! seeing a smaller struct treats absent fields as zero/null (the default for
//! every field must be a safe no-op — fail closed, not silent success).
//!
//! # Protocol
//!
//! 1. Host fills [`NxrtNegotiateRequest`] with its supported version range.
//! 2. Plugin's `NxrtNegotiate` checks if it can satisfy the request.
//! 3. Plugin fills [`NxrtNegotiateResponse`] with the agreed version (or its
//!    own range on failure for diagnostics).
//! 4. If status is Ok, the host proceeds with `NxrtCreateEpFactories`.

use crate::status::{NxrtStatus, NxrtStatusCode};

/// Current nxrt ABI major version. Bump on breaking changes.
pub const NXRT_ABI_VERSION_MAJOR: u32 = 1;

/// Current nxrt ABI minor version. Bump on additive changes.
pub const NXRT_ABI_VERSION_MINOR: u32 = 1;

/// A version range a side (host or plugin) supports.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NxrtVersionRange {
    /// Minimum major version supported (inclusive).
    pub major_min: u32,
    /// Maximum major version supported (inclusive).
    pub major_max: u32,
    /// Maximum minor version supported at `major_max`.
    pub minor_max: u32,
}

impl NxrtVersionRange {
    /// A range covering exactly the current compiled version.
    pub const fn current() -> Self {
        Self {
            major_min: NXRT_ABI_VERSION_MAJOR,
            major_max: NXRT_ABI_VERSION_MAJOR,
            minor_max: NXRT_ABI_VERSION_MINOR,
        }
    }
}

/// Request the host sends to the plugin during negotiation.
///
/// # Ownership
///
/// Borrowed by the plugin for the duration of the `NxrtNegotiate` call.
/// The host owns the allocation and it is valid only within the call frame.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct NxrtNegotiateRequest {
    /// Size of this struct in bytes (for forward compat).
    pub struct_size: u32,
    /// The host's supported version range.
    pub host_range: NxrtVersionRange,
}

impl NxrtNegotiateRequest {
    /// Construct a request advertising the current compiled version.
    pub const fn current() -> Self {
        Self {
            struct_size: std::mem::size_of::<Self>() as u32,
            host_range: NxrtVersionRange::current(),
        }
    }
}

/// Response the plugin writes during negotiation.
///
/// # Ownership
///
/// The host provides the buffer; the plugin fills it. Valid only within the
/// call frame.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct NxrtNegotiateResponse {
    /// Size of this struct in bytes.
    pub struct_size: u32,
    /// The negotiated major version (valid only on success).
    pub agreed_major: u32,
    /// The negotiated minor version (valid only on success).
    pub agreed_minor: u32,
    /// The plugin's supported range (always filled, for diagnostics on failure).
    pub plugin_range: NxrtVersionRange,
    /// Capability flags the plugin advertises at the agreed version.
    pub capability_flags: u64,
}

impl NxrtNegotiateResponse {
    /// Zero-initialized response (for the host to allocate before the call).
    pub const fn zeroed() -> Self {
        Self {
            struct_size: std::mem::size_of::<Self>() as u32,
            agreed_major: 0,
            agreed_minor: 0,
            plugin_range: NxrtVersionRange {
                major_min: 0,
                major_max: 0,
                minor_max: 0,
            },
            capability_flags: 0,
        }
    }
}

// ─── Capability flags ───────────────────────────────────────────────────────

/// The plugin supports device enumeration via the factory vtable.
pub const NXRT_CAP_DEVICE_ENUMERATION: u64 = 1 << 0;
/// The plugin supports custom allocators.
pub const NXRT_CAP_ALLOCATOR: u64 = 1 << 1;
/// The plugin supports stream/sync primitives.
pub const NXRT_CAP_STREAM_SYNC: u64 = 1 << 2;
/// The plugin supports compiled kernel caching (EpContext).
pub const NXRT_CAP_EP_CONTEXT: u64 = 1 << 3;

/// All flags known at this ABI version. Used for fail-closed validation:
/// if a plugin sets bits outside this mask, the host rejects.
pub const NXRT_CAP_KNOWN_MASK: u64 =
    NXRT_CAP_DEVICE_ENUMERATION | NXRT_CAP_ALLOCATOR | NXRT_CAP_STREAM_SYNC | NXRT_CAP_EP_CONTEXT;

// ─── Negotiation logic ──────────────────────────────────────────────────────

/// The plugin-side negotiation implementation.
///
/// # Safety
///
/// Both pointers must be valid and non-null.
pub unsafe fn negotiate(
    request: *const NxrtNegotiateRequest,
    response_out: *mut NxrtNegotiateResponse,
) -> NxrtStatus {
    if request.is_null() || response_out.is_null() {
        return NxrtStatus::from_code_with_message(
            NxrtStatusCode::InvalidArgument,
            "NxrtNegotiate: null request or response pointer",
        );
    }

    // SAFETY: pointers validated above.
    let req = unsafe { &*request };
    let resp = unsafe { &mut *response_out };

    // Always report our range for diagnostics.
    resp.struct_size = std::mem::size_of::<NxrtNegotiateResponse>() as u32;
    resp.plugin_range = NxrtVersionRange::current();

    // Check major version compatibility.
    let plugin_major = NXRT_ABI_VERSION_MAJOR;
    if plugin_major < req.host_range.major_min || plugin_major > req.host_range.major_max {
        return NxrtStatus::from_code_with_message(
            NxrtStatusCode::VersionMismatch,
            &format!(
                "NxrtNegotiate: plugin major version {plugin_major} outside host range [{}, {}]",
                req.host_range.major_min, req.host_range.major_max
            ),
        );
    }

    // Major matches. Agree on the minimum minor of both sides.
    let plugin_minor = NXRT_ABI_VERSION_MINOR;
    let agreed_minor = plugin_minor.min(req.host_range.minor_max);

    resp.agreed_major = plugin_major;
    resp.agreed_minor = agreed_minor;
    resp.capability_flags = NXRT_CAP_DEVICE_ENUMERATION;

    NxrtStatus::ok()
}

/// Host-side validation of a negotiation response. Returns Ok(()) or an
/// error message. Utility for the host crate (`onnx-runtime-ep-nxrt-host`).
pub fn validate_negotiation(
    host_range: &NxrtVersionRange,
    response: &NxrtNegotiateResponse,
) -> Result<(), String> {
    if response.agreed_major < host_range.major_min || response.agreed_major > host_range.major_max
    {
        return Err(format!(
            "agreed major {} outside host range [{}, {}]",
            response.agreed_major, host_range.major_min, host_range.major_max
        ));
    }
    if response.agreed_minor > host_range.minor_max {
        return Err(format!(
            "agreed minor {} exceeds host minor_max {}",
            response.agreed_minor, host_range.minor_max
        ));
    }
    // Fail closed on unknown capability flags.
    let unknown = response.capability_flags & !NXRT_CAP_KNOWN_MASK;
    if unknown != 0 {
        return Err(format!(
            "plugin advertises unknown capability flags: {unknown:#x} — refusing (fail closed)"
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ptr;

    #[test]
    fn negotiate_success_same_version() {
        let req = NxrtNegotiateRequest::current();
        let mut resp = NxrtNegotiateResponse::zeroed();
        let status = unsafe { negotiate(&req, &mut resp) };
        assert!(status.is_ok());
        assert_eq!(resp.agreed_major, NXRT_ABI_VERSION_MAJOR);
        assert_eq!(resp.agreed_minor, NXRT_ABI_VERSION_MINOR);
    }

    #[test]
    fn negotiate_rejects_incompatible_major() {
        let req = NxrtNegotiateRequest {
            struct_size: std::mem::size_of::<NxrtNegotiateRequest>() as u32,
            host_range: NxrtVersionRange {
                major_min: 99,
                major_max: 99,
                minor_max: 0,
            },
        };
        let mut resp = NxrtNegotiateResponse::zeroed();
        let status = unsafe { negotiate(&req, &mut resp) };
        assert_eq!(status.status_code(), Some(NxrtStatusCode::VersionMismatch));
        let msg = status.message_str().unwrap();
        assert!(msg.contains("plugin major version"));
    }

    #[test]
    fn negotiate_agrees_on_min_minor() {
        let req = NxrtNegotiateRequest {
            struct_size: std::mem::size_of::<NxrtNegotiateRequest>() as u32,
            host_range: NxrtVersionRange {
                major_min: 1,
                major_max: 1,
                minor_max: 5,
            },
        };
        let mut resp = NxrtNegotiateResponse::zeroed();
        let status = unsafe { negotiate(&req, &mut resp) };
        assert!(status.is_ok());
        assert_eq!(resp.agreed_minor, NXRT_ABI_VERSION_MINOR);
    }

    #[test]
    fn negotiate_null_pointers_fail_closed() {
        let mut resp = NxrtNegotiateResponse::zeroed();
        let status = unsafe { negotiate(ptr::null(), &mut resp) };
        assert_eq!(status.status_code(), Some(NxrtStatusCode::InvalidArgument));
    }

    #[test]
    fn validate_negotiation_rejects_unknown_caps() {
        let host_range = NxrtVersionRange::current();
        let resp = NxrtNegotiateResponse {
            struct_size: std::mem::size_of::<NxrtNegotiateResponse>() as u32,
            agreed_major: NXRT_ABI_VERSION_MAJOR,
            agreed_minor: NXRT_ABI_VERSION_MINOR,
            plugin_range: NxrtVersionRange::current(),
            capability_flags: 1 << 63,
        };
        let result = validate_negotiation(&host_range, &resp);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("unknown capability flags"));
    }

    #[test]
    fn validate_negotiation_accepts_known_caps() {
        let host_range = NxrtVersionRange::current();
        let resp = NxrtNegotiateResponse {
            struct_size: std::mem::size_of::<NxrtNegotiateResponse>() as u32,
            agreed_major: NXRT_ABI_VERSION_MAJOR,
            agreed_minor: NXRT_ABI_VERSION_MINOR,
            plugin_range: NxrtVersionRange::current(),
            capability_flags: NXRT_CAP_DEVICE_ENUMERATION | NXRT_CAP_ALLOCATOR,
        };
        let result = validate_negotiation(&host_range, &resp);
        assert!(result.is_ok());
    }

    #[test]
    fn version_range_current_matches_constants() {
        let r = NxrtVersionRange::current();
        assert_eq!(r.major_min, NXRT_ABI_VERSION_MAJOR);
        assert_eq!(r.major_max, NXRT_ABI_VERSION_MAJOR);
        assert_eq!(r.minor_max, NXRT_ABI_VERSION_MINOR);
    }
}
