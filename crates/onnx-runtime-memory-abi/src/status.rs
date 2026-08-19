//! Stable status/error values for the `nxmem` memory plugin ABI.
//!
//! # Why an inline message buffer
//!
//! The plugin and the host are separate dynamic modules that may be linked
//! against different C runtimes. Memory allocated by one module and freed by
//! the other is undefined behaviour, so [`NxmemStatus`] carries its message in
//! a fixed inline array. There is nothing to free, nothing to retain, and no
//! Rust allocator ownership crossing the boundary.
//!
//! # Wire-code safety
//!
//! [`NxmemStatus::code`] is a raw `u32`, never a Rust enum. A newer participant
//! may send a discriminant this build does not know; transmuting it into
//! [`NxmemStatusCode`] would be undefined behaviour. Use
//! [`NxmemStatus::status_code`], which returns `None` for unknown codes, and
//! treat `None` as a hard failure (fail closed).

/// Maximum message bytes (excluding the NUL terminator) in an [`NxmemStatus`].
///
/// Longer messages are truncated on a UTF-8 character boundary.
pub const NXMEM_STATUS_MESSAGE_MAX: usize = 255;

/// Total inline buffer length, including the NUL terminator.
pub const NXMEM_STATUS_MESSAGE_BUF: usize = NXMEM_STATUS_MESSAGE_MAX + 1;

/// Stable status codes.
///
/// Discriminants are append-only and never reordered: a value that shipped
/// keeps its meaning forever. Unknown discriminants must be treated as fatal.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NxmemStatusCode {
    /// The call succeeded.
    Ok = 0,
    /// The host and plugin could not agree on an ABI version.
    VersionMismatch = 1,
    /// A struct's `struct_size` is smaller than the required prefix for the
    /// negotiated ABI version, or a required function slot is null.
    ShortStruct = 2,
    /// The requested capability is not supported by this participant. This is
    /// the explicit representation of absence — never a successful no-op.
    UnsupportedCapability = 3,
    /// A pointer was null, a size was nonsensical, or a field was out of range.
    InvalidArgument = 4,
    /// An unexpected internal failure, including a caught panic.
    InternalError = 5,
    /// The slot exists in the vtable but this participant does not implement it.
    NotImplemented = 6,
    /// The backing device or driver reported a failure.
    DeviceError = 7,
    /// The allocation could not be satisfied.
    OutOfMemory = 8,
    /// The object belongs to a different device than the one addressed.
    WrongDevice = 9,
    /// The object belongs to a different mechanism instance than the one
    /// addressed — this rejects cross-provider misuse before any free.
    WrongMechanism = 10,
    /// The allocation identity is unknown, already retired, or from an earlier
    /// generation that reused the same address.
    UnknownAllocation = 11,
    /// Release mutated state but could not complete; residual ownership stays
    /// with the plugin. See the accompanying [`crate::NxmemReleaseOutcome`].
    ReleaseQuarantined = 12,
    /// A host callback returned a failure and the plugin could not proceed.
    CallbackFailed = 13,
    /// The plugin still owns live objects and cannot satisfy the request now.
    Busy = 14,
}

impl NxmemStatusCode {
    /// Whether this code means success.
    pub const fn is_ok(self) -> bool {
        matches!(self, Self::Ok)
    }

    /// Checked conversion from a raw wire value.
    ///
    /// Returns `None` for values this build does not recognise. Never
    /// transmute — an untrusted participant may send anything.
    pub const fn from_u32(value: u32) -> Option<Self> {
        match value {
            0 => Some(Self::Ok),
            1 => Some(Self::VersionMismatch),
            2 => Some(Self::ShortStruct),
            3 => Some(Self::UnsupportedCapability),
            4 => Some(Self::InvalidArgument),
            5 => Some(Self::InternalError),
            6 => Some(Self::NotImplemented),
            7 => Some(Self::DeviceError),
            8 => Some(Self::OutOfMemory),
            9 => Some(Self::WrongDevice),
            10 => Some(Self::WrongMechanism),
            11 => Some(Self::UnknownAllocation),
            12 => Some(Self::ReleaseQuarantined),
            13 => Some(Self::CallbackFailed),
            14 => Some(Self::Busy),
            _ => None,
        }
    }

    /// Stable machine-readable name. These strings are part of the contract
    /// and are what diagnostics quote, so they do not change with locale or
    /// with the Rust `Debug` formatting of the enum.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Ok => "OK",
            Self::VersionMismatch => "VERSION_MISMATCH",
            Self::ShortStruct => "SHORT_STRUCT",
            Self::UnsupportedCapability => "UNSUPPORTED_CAPABILITY",
            Self::InvalidArgument => "INVALID_ARGUMENT",
            Self::InternalError => "INTERNAL_ERROR",
            Self::NotImplemented => "NOT_IMPLEMENTED",
            Self::DeviceError => "DEVICE_ERROR",
            Self::OutOfMemory => "OUT_OF_MEMORY",
            Self::WrongDevice => "WRONG_DEVICE",
            Self::WrongMechanism => "WRONG_MECHANISM",
            Self::UnknownAllocation => "UNKNOWN_ALLOCATION",
            Self::ReleaseQuarantined => "RELEASE_QUARANTINED",
            Self::CallbackFailed => "CALLBACK_FAILED",
            Self::Busy => "BUSY",
        }
    }

    /// Every code known at this ABI version, in wire order.
    pub const ALL: [Self; 15] = [
        Self::Ok,
        Self::VersionMismatch,
        Self::ShortStruct,
        Self::UnsupportedCapability,
        Self::InvalidArgument,
        Self::InternalError,
        Self::NotImplemented,
        Self::DeviceError,
        Self::OutOfMemory,
        Self::WrongDevice,
        Self::WrongMechanism,
        Self::UnknownAllocation,
        Self::ReleaseQuarantined,
        Self::CallbackFailed,
        Self::Busy,
    ];
}

/// The value every `nxmem` entry point returns.
///
/// # Layout
///
/// ```text
/// struct NxmemStatus {
///     uint32_t code;         // NxmemStatusCode wire value
///     uint32_t message_len;  // bytes of `message` that are meaningful
///     uint8_t  message[256]; // NUL-terminated UTF-8
/// };
/// ```
///
/// # Ownership
///
/// Pure value type. It is returned by value, copied freely, and never freed by
/// the other side.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct NxmemStatus {
    /// Raw wire status code. Use [`Self::status_code`] for checked conversion.
    pub code: u32,
    /// Meaningful bytes in `message`, excluding the NUL terminator.
    pub message_len: u32,
    /// Inline NUL-terminated UTF-8 message. Only the first `message_len` bytes
    /// are meaningful.
    pub message: [u8; NXMEM_STATUS_MESSAGE_BUF],
}

// SAFETY: `NxmemStatus` is a plain value type holding no pointers and no
// interior mutability, so it is trivially safe to move between threads.
unsafe impl Send for NxmemStatus {}
// SAFETY: see the `Send` justification; shared references expose only reads.
unsafe impl Sync for NxmemStatus {}

impl core::fmt::Debug for NxmemStatus {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("NxmemStatus")
            .field("code", &self.status_code())
            .field("code_raw", &self.code)
            .field("message", &self.message_str())
            .finish()
    }
}

impl NxmemStatus {
    /// A success status with no message.
    pub const fn ok() -> Self {
        Self {
            code: NxmemStatusCode::Ok as u32,
            message_len: 0,
            message: [0u8; NXMEM_STATUS_MESSAGE_BUF],
        }
    }

    /// A status carrying `code` and no message.
    pub const fn from_code(code: NxmemStatusCode) -> Self {
        Self {
            code: code as u32,
            message_len: 0,
            message: [0u8; NXMEM_STATUS_MESSAGE_BUF],
        }
    }

    /// A status carrying `code` and a message.
    ///
    /// The message is truncated at the first interior NUL and at
    /// [`NXMEM_STATUS_MESSAGE_MAX`] bytes. Truncation lands on a UTF-8
    /// character boundary so [`Self::message_str`] never fails on our own
    /// output.
    pub fn with_message(code: NxmemStatusCode, message: &str) -> Self {
        let mut status = Self::from_code(code);
        let bytes = message.as_bytes();
        let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
        let mut len = end.min(NXMEM_STATUS_MESSAGE_MAX);
        // Back off to a character boundary so truncation cannot produce
        // invalid UTF-8.
        while len > 0 && !message.is_char_boundary(len) {
            len -= 1;
        }
        status.message[..len].copy_from_slice(&bytes[..len]);
        status.message[len] = 0;
        status.message_len = len as u32;
        status
    }

    /// Whether the wire code is success.
    pub const fn is_ok(&self) -> bool {
        self.code == NxmemStatusCode::Ok as u32
    }

    /// Checked conversion of the wire code.
    ///
    /// `None` means "a participant sent a code this build does not know"; the
    /// caller must treat that as a fatal error.
    pub const fn status_code(&self) -> Option<NxmemStatusCode> {
        NxmemStatusCode::from_u32(self.code)
    }

    /// The message, if any, as UTF-8.
    ///
    /// Returns `None` when there is no message or when the bytes are not valid
    /// UTF-8 (defensive against a corrupt or hostile participant).
    pub fn message_str(&self) -> Option<&str> {
        let len = (self.message_len as usize).min(NXMEM_STATUS_MESSAGE_MAX);
        if len == 0 {
            return None;
        }
        core::str::from_utf8(&self.message[..len]).ok()
    }

    /// A stable rendering of code and message for diagnostics.
    ///
    /// Unknown codes render as `UNKNOWN(<n>)` rather than being dropped, so a
    /// newer plugin's failure is still traceable.
    pub fn describe(&self) -> String {
        let name = match self.status_code() {
            Some(code) => code.name().to_string(),
            None => format!("UNKNOWN({})", self.code),
        };
        match self.message_str() {
            Some(message) => format!("{name}: {message}"),
            None => name,
        }
    }
}

impl Default for NxmemStatus {
    fn default() -> Self {
        Self::ok()
    }
}

impl core::fmt::Display for NxmemStatus {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(&self.describe())
    }
}

/// Run `f`, converting a Rust panic into [`NxmemStatusCode::InternalError`].
///
/// Every `extern "C"` function on either side of the boundary must funnel
/// through this (or [`catch_void_panic`]). Unwinding across a `cdylib`
/// boundary is undefined behaviour.
pub fn catch_status_panic<F: FnOnce() -> NxmemStatus>(f: F) -> NxmemStatus {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)) {
        Ok(status) => status,
        Err(_) => NxmemStatus::with_message(
            NxmemStatusCode::InternalError,
            "nxmem: panic caught at the ABI boundary (fail closed)",
        ),
    }
}

/// Run `f`, swallowing a Rust panic.
///
/// Used for `void`-returning slots such as `retain`/`release`, which have no
/// channel to report a failure.
pub fn catch_void_panic<F: FnOnce()>(f: F) {
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ok_status_has_no_message() {
        let status = NxmemStatus::ok();
        assert!(status.is_ok());
        assert_eq!(status.message_str(), None);
        assert_eq!(status.describe(), "OK");
    }

    #[test]
    fn every_known_code_round_trips_and_has_a_stable_name() {
        let mut seen = Vec::new();
        for code in NxmemStatusCode::ALL {
            assert_eq!(NxmemStatusCode::from_u32(code as u32), Some(code));
            assert!(!code.name().is_empty());
            assert!(
                !seen.contains(&code.name()),
                "duplicate name {}",
                code.name()
            );
            seen.push(code.name());
        }
        assert_eq!(seen.len(), NxmemStatusCode::ALL.len());
    }

    #[test]
    fn unknown_wire_code_is_not_ok_and_does_not_transmute() {
        let mut status = NxmemStatus::ok();
        status.code = 9_999;
        assert!(!status.is_ok());
        assert_eq!(status.status_code(), None);
        assert_eq!(status.describe(), "UNKNOWN(9999)");
    }

    #[test]
    fn long_message_truncates_on_a_character_boundary() {
        // Three-byte characters do not divide 255 evenly, so a naive
        // truncation would split one and make `message_str` return None.
        let message = "\u{4e2d}".repeat(200);
        let status = NxmemStatus::with_message(NxmemStatusCode::DeviceError, &message);
        assert_eq!(status.message_len, NXMEM_STATUS_MESSAGE_MAX as u32);
        let recovered = status.message_str().expect("valid utf-8 after truncation");
        assert!(recovered.chars().all(|c| c == '\u{4e2d}'));
    }

    #[test]
    fn interior_nul_truncates_the_message() {
        let status = NxmemStatus::with_message(NxmemStatusCode::InvalidArgument, "head\0tail");
        assert_eq!(status.message_str(), Some("head"));
    }

    #[test]
    fn status_layout_is_the_documented_264_bytes() {
        assert_eq!(std::mem::size_of::<NxmemStatus>(), 264);
        assert_eq!(std::mem::align_of::<NxmemStatus>(), 4);
    }

    #[test]
    fn panic_in_a_status_slot_becomes_an_internal_error() {
        let status = catch_status_panic(|| panic!("deliberate"));
        assert_eq!(status.status_code(), Some(NxmemStatusCode::InternalError));
        assert!(status.describe().contains("panic"));
    }

    #[test]
    fn panic_in_a_void_slot_does_not_unwind() {
        catch_void_panic(|| panic!("deliberate"));
    }
}
