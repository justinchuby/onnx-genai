//! Stable error/status type for the nxrt ABI.
//!
//! # Ownership
//!
//! `NxrtStatus` is a **pure value type** with a fixed inline message buffer.
//! No heap allocation, no cross-module free, no CRT coupling. The struct can
//! be returned by value from `extern "C"` functions and memcpy'd freely.
//!
//! ## Why inline?
//!
//! The nxrt ABI is a stable `cdylib` boundary. The plugin and host may be
//! linked against different C runtimes (common on Windows). Memory allocated
//! by one side and freed by the other is undefined behaviour. An inline
//! buffer eliminates that class of bug entirely — there is nothing to free.
//!
//! The maximum message length is [`NXRT_STATUS_MESSAGE_MAX`] bytes (excluding
//! the NUL terminator). Messages longer than that are silently truncated.

/// Maximum number of message bytes (excluding NUL terminator) in an
/// [`NxrtStatus`]. Messages longer than this are silently truncated.
pub const NXRT_STATUS_MESSAGE_MAX: usize = 255;

/// Total size of the inline message buffer (including NUL terminator).
const MESSAGE_BUF_LEN: usize = NXRT_STATUS_MESSAGE_MAX + 1;

/// Status codes for the nxrt ABI. Exhaustive and stable — new codes are
/// appended, never reordered. Unknown codes must be treated as fatal errors
/// (fail closed).
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NxrtStatusCode {
    /// Success.
    Ok = 0,
    /// ABI major version incompatible.
    VersionMismatch = 1,
    /// A required capability is not supported.
    UnsupportedCapability = 2,
    /// An argument was null or invalid.
    InvalidArgument = 3,
    /// An internal error (including caught panics).
    InternalError = 4,
    /// The operation is not implemented by this plugin.
    NotImplemented = 5,
    /// A device/resource error.
    DeviceError = 6,
    /// Out of memory.
    OutOfMemory = 7,
}

impl NxrtStatusCode {
    /// Returns `true` for success.
    pub fn is_ok(self) -> bool {
        matches!(self, Self::Ok)
    }

    /// Checked conversion from a raw `u32` wire code.
    ///
    /// Returns `None` for unrecognised discriminants. This is the safe path
    /// for handling values from an untrusted plugin — **never transmute**.
    pub fn from_u32(value: u32) -> Option<Self> {
        match value {
            0 => Some(Self::Ok),
            1 => Some(Self::VersionMismatch),
            2 => Some(Self::UnsupportedCapability),
            3 => Some(Self::InvalidArgument),
            4 => Some(Self::InternalError),
            5 => Some(Self::NotImplemented),
            6 => Some(Self::DeviceError),
            7 => Some(Self::OutOfMemory),
            _ => None,
        }
    }
}

/// ABI-stable status value returned by all nxrt entry points.
///
/// # Layout
///
/// ```text
/// struct NxrtStatus {
///     code:        u32,       // NxrtStatusCode discriminant (wire value)
///     message_len: u32,       // length of message in bytes (0 = no message)
///     message:     [u8; 256], // inline NUL-terminated UTF-8 message buffer
/// }
/// ```
///
/// Size: 264 bytes. The struct is a **pure value type** — no heap allocation,
/// no pointers, no cross-module free. This makes it safe to return by value
/// across a `cdylib` boundary regardless of CRT configuration.
///
/// # Wire-code safety
///
/// The `code` field is stored as a raw `u32` — **not** as `NxrtStatusCode`.
/// This is deliberate: the nxrt ABI is a plugin boundary, and the other side
/// may be a newer plugin version that sends status codes unknown to this host.
/// Transmuting an unrecognised discriminant into a Rust enum is undefined
/// behaviour. The safe accessor [`NxrtStatus::status_code()`] performs a
/// checked conversion, mapping unknown values to `None`.
///
/// # Cross-module safety
///
/// Because the message is inline, there is no allocator coupling between
/// plugin and host. The plugin writes the message into the buffer; the host
/// reads it. Neither side frees anything the other allocated.
#[repr(C)]
#[derive(Clone)]
pub struct NxrtStatus {
    /// Wire status code as a raw `u32`. Use [`Self::status_code()`] for
    /// checked conversion to `NxrtStatusCode`. Do **not** transmute — an
    /// untrusted plugin may send an unrecognised discriminant.
    pub code: u32,
    /// Number of valid UTF-8 bytes in `message` (excluding NUL terminator).
    /// Zero means no message. Always ≤ [`NXRT_STATUS_MESSAGE_MAX`].
    pub message_len: u32,
    /// Inline NUL-terminated UTF-8 message buffer. Only the first
    /// `message_len` bytes are meaningful; `message[message_len]` is `0`.
    /// For success statuses this is typically all zeros.
    pub message: [u8; MESSAGE_BUF_LEN],
}

// SAFETY: NxrtStatus is a plain value type with no pointers.
unsafe impl Send for NxrtStatus {}
unsafe impl Sync for NxrtStatus {}

impl std::fmt::Debug for NxrtStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NxrtStatus")
            .field("code", &self.status_code())
            .field("code_raw", &self.code)
            .field("message", &self.message_str())
            .finish()
    }
}

impl NxrtStatus {
    /// Create a success status.
    pub const fn ok() -> Self {
        Self {
            code: NxrtStatusCode::Ok as u32,
            message_len: 0,
            message: [0u8; MESSAGE_BUF_LEN],
        }
    }

    /// Create a status from a code with no message.
    pub const fn from_code(code: NxrtStatusCode) -> Self {
        Self {
            code: code as u32,
            message_len: 0,
            message: [0u8; MESSAGE_BUF_LEN],
        }
    }

    /// Create a status from a code and a message string.
    ///
    /// Messages longer than [`NXRT_STATUS_MESSAGE_MAX`] bytes are silently
    /// truncated. Interior NUL bytes cause truncation at the first NUL.
    pub fn from_code_with_message(code: NxrtStatusCode, msg: &str) -> Self {
        let mut status = Self::from_code(code);
        // Truncate at first NUL (if any) and at max length.
        let bytes = msg.as_bytes();
        let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
        let len = end.min(NXRT_STATUS_MESSAGE_MAX);
        status.message[..len].copy_from_slice(&bytes[..len]);
        status.message[len] = 0; // NUL terminator
        status.message_len = len as u32;
        status
    }

    /// Whether this status is success (wire code == 0).
    pub fn is_ok(&self) -> bool {
        self.code == NxrtStatusCode::Ok as u32
    }

    /// Checked conversion from the raw wire code to a known `NxrtStatusCode`.
    ///
    /// Returns `None` if the discriminant is not recognised — this is the
    /// **expected** case when a newer plugin sends a code unknown to this host.
    /// Callers must treat `None` as a fatal error (fail closed).
    pub fn status_code(&self) -> Option<NxrtStatusCode> {
        NxrtStatusCode::from_u32(self.code)
    }

    /// Get the message as a string slice, if present.
    ///
    /// Returns `None` if `message_len` is zero. Returns `None` if the
    /// message bytes are not valid UTF-8 (defensive against corrupt data
    /// from an untrusted plugin).
    pub fn message_str(&self) -> Option<&str> {
        let len = (self.message_len as usize).min(NXRT_STATUS_MESSAGE_MAX);
        if len == 0 {
            return None;
        }
        std::str::from_utf8(&self.message[..len]).ok()
    }
}

/// Helper for `extern "C"` void-returning callbacks: catches panics silently.
pub fn catch_void_panic<F: FnOnce()>(f: F) {
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f));
}

/// Helper for `extern "C"` status-returning callbacks: catches panics and
/// converts them to [`NxrtStatusCode::InternalError`].
pub fn catch_status_panic<F: FnOnce() -> NxrtStatus>(f: F) -> NxrtStatus {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)) {
        Ok(status) => status,
        Err(_) => NxrtStatus::from_code_with_message(
            NxrtStatusCode::InternalError,
            "panic caught at ABI boundary (fail-closed)",
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ok_status_is_success() {
        let s = NxrtStatus::ok();
        assert!(s.is_ok());
        assert_eq!(s.message_len, 0);
        assert!(s.message_str().is_none());
    }

    #[test]
    fn from_code_with_message_roundtrips() {
        let s = NxrtStatus::from_code_with_message(
            NxrtStatusCode::VersionMismatch,
            "major version 99 not supported",
        );
        assert!(!s.is_ok());
        assert_eq!(s.status_code(), Some(NxrtStatusCode::VersionMismatch));
        let msg = s.message_str().unwrap();
        assert!(msg.contains("major version 99"));
    }

    #[test]
    fn drop_is_trivial_no_heap() {
        // Pure value type — drop is a no-op. Just verify no panic/crash.
        let _s = NxrtStatus::from_code_with_message(NxrtStatusCode::InternalError, "test drop");
    }

    #[test]
    fn catch_status_panic_converts_to_error() {
        let s = catch_status_panic(|| {
            panic!("deliberate test panic");
        });
        assert_eq!(s.status_code(), Some(NxrtStatusCode::InternalError));
        let msg = s.message_str().unwrap();
        assert!(msg.contains("panic"));
    }

    #[test]
    fn catch_void_panic_does_not_unwind() {
        // Must not panic the test thread.
        catch_void_panic(|| {
            panic!("void panic test");
        });
    }

    #[test]
    fn message_truncation_at_max_length() {
        let long_msg = "x".repeat(500);
        let s = NxrtStatus::from_code_with_message(NxrtStatusCode::InternalError, &long_msg);
        assert_eq!(s.message_len as usize, NXRT_STATUS_MESSAGE_MAX);
        let msg = s.message_str().unwrap();
        assert_eq!(msg.len(), NXRT_STATUS_MESSAGE_MAX);
    }

    #[test]
    fn unknown_code_is_not_ok() {
        // Simulating a code from a newer plugin — must fail closed.
        let s = NxrtStatus::from_code(NxrtStatusCode::OutOfMemory);
        assert!(!s.is_ok());
    }

    #[test]
    fn clone_produces_independent_copy() {
        let s1 = NxrtStatus::from_code_with_message(NxrtStatusCode::DeviceError, "dev err");
        let s2 = s1.clone();
        assert_eq!(s1.message_str(), s2.message_str());
        assert_eq!(s1.code, s2.code);
    }

    #[test]
    fn struct_is_repr_c_fixed_size() {
        // Verify the struct size is stable (important for ABI).
        let size = std::mem::size_of::<NxrtStatus>();
        // code(4) + message_len(4) + message(256) = 264
        assert_eq!(size, 264);
    }

    #[test]
    fn unknown_discriminant_does_not_cause_ub() {
        // Simulate a newer plugin sending an unknown status code.
        // The raw u32 wire format means no transmute, no UB.
        let mut s = NxrtStatus::ok();
        s.code = 255; // unknown discriminant
        // Must not be Ok
        assert!(!s.is_ok());
        // Checked conversion returns None (fail closed)
        assert_eq!(s.status_code(), None);
    }

    #[test]
    fn all_known_codes_roundtrip_through_from_u32() {
        let codes = [
            NxrtStatusCode::Ok,
            NxrtStatusCode::VersionMismatch,
            NxrtStatusCode::UnsupportedCapability,
            NxrtStatusCode::InvalidArgument,
            NxrtStatusCode::InternalError,
            NxrtStatusCode::NotImplemented,
            NxrtStatusCode::DeviceError,
            NxrtStatusCode::OutOfMemory,
        ];
        for code in codes {
            let s = NxrtStatus::from_code(code);
            assert_eq!(s.status_code(), Some(code));
        }
    }
}
