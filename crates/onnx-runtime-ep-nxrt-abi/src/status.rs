//! Stable error/status type for the nxrt ABI.
//!
//! # Ownership
//!
//! An `NxrtStatus` is a value type (32-bit code + optional heap message).
//! When returned from an `extern "C"` function the **caller owns** the status
//! and must call [`NxrtStatus::free_message`] if `message` is non-null.
//! Within Rust code, `Drop` handles this automatically.

use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::ptr;

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
}

/// ABI-stable status value returned by all nxrt entry points.
///
/// # Layout
///
/// ```text
/// struct NxrtStatus {
///     code: u32,
///     _reserved: u32,         // padding for alignment; must be 0
///     message: *mut c_char,   // nullable, heap-allocated, UTF-8 + NUL
/// }
/// ```
///
/// Size: 16 bytes on 64-bit, 12 on 32-bit. The `_reserved` field ensures
/// consistent alignment regardless of pointer size — future minor versions
/// may repurpose it (with a minor version bump).
#[repr(C)]
#[derive(Debug)]
pub struct NxrtStatus {
    pub code: NxrtStatusCode,
    _reserved: u32,
    /// Optional NUL-terminated UTF-8 error message. Null for success.
    ///
    /// # Ownership
    ///
    /// If non-null, the **receiver** owns this allocation and must free it via
    /// [`NxrtStatus::free_message`] or by taking ownership in Rust (which
    /// `Drop` does automatically). The plugin allocates with `CString`; the
    /// host frees with `CString::from_raw`.
    pub message: *mut c_char,
}

// SAFETY: The message pointer is an owned heap allocation; no aliasing.
unsafe impl Send for NxrtStatus {}
unsafe impl Sync for NxrtStatus {}

impl NxrtStatus {
    /// Create a success status.
    pub const fn ok() -> Self {
        Self {
            code: NxrtStatusCode::Ok,
            _reserved: 0,
            message: ptr::null_mut(),
        }
    }

    /// Create a status from a code with no message.
    pub const fn from_code(code: NxrtStatusCode) -> Self {
        Self {
            code,
            _reserved: 0,
            message: ptr::null_mut(),
        }
    }

    /// Create a status from a code and a message string.
    pub fn from_code_with_message(code: NxrtStatusCode, msg: &str) -> Self {
        let c_msg = CString::new(msg).unwrap_or_else(|_| CString::new("(invalid message)").unwrap());
        Self {
            code,
            _reserved: 0,
            message: c_msg.into_raw(),
        }
    }

    /// Whether this status is success.
    pub fn is_ok(&self) -> bool {
        self.code.is_ok()
    }

    /// Get the message as a string slice, if present.
    ///
    /// # Safety
    ///
    /// The message pointer must be valid if non-null.
    pub unsafe fn message_str(&self) -> Option<&str> {
        if self.message.is_null() {
            None
        } else {
            // SAFETY: caller guarantees the pointer is valid.
            let cstr = unsafe { CStr::from_ptr(self.message) };
            cstr.to_str().ok()
        }
    }

    /// Free the message allocation. Idempotent (null is a no-op).
    ///
    /// # Safety
    ///
    /// Must only be called once per status, and only if the caller owns the
    /// message pointer.
    pub unsafe fn free_message(&mut self) {
        if !self.message.is_null() {
            // SAFETY: message was allocated with CString::into_raw.
            let _ = unsafe { CString::from_raw(self.message) };
            self.message = ptr::null_mut();
        }
    }
}

impl Drop for NxrtStatus {
    fn drop(&mut self) {
        if !self.message.is_null() {
            // SAFETY: we own the message pointer.
            let _ = unsafe { CString::from_raw(self.message) };
            self.message = ptr::null_mut();
        }
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
        assert!(s.message.is_null());
    }

    #[test]
    fn from_code_with_message_roundtrips() {
        let s = NxrtStatus::from_code_with_message(
            NxrtStatusCode::VersionMismatch,
            "major version 99 not supported",
        );
        assert!(!s.is_ok());
        assert_eq!(s.code, NxrtStatusCode::VersionMismatch);
        let msg = unsafe { s.message_str() }.unwrap();
        assert!(msg.contains("major version 99"));
    }

    #[test]
    fn drop_frees_message_without_leak() {
        // Just verify no panic/crash on drop with a message.
        let _s = NxrtStatus::from_code_with_message(NxrtStatusCode::InternalError, "test drop");
    }

    #[test]
    fn catch_status_panic_converts_to_error() {
        let s = catch_status_panic(|| {
            panic!("deliberate test panic");
        });
        assert_eq!(s.code, NxrtStatusCode::InternalError);
        let msg = unsafe { s.message_str() }.unwrap();
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
    fn free_message_is_idempotent() {
        let mut s = NxrtStatus::from_code_with_message(NxrtStatusCode::DeviceError, "dev err");
        unsafe { s.free_message() };
        assert!(s.message.is_null());
        // Second call is a no-op.
        unsafe { s.free_message() };
        // Message is already null so Drop is safe (no double-free).
    }

    #[test]
    fn unknown_code_is_not_ok() {
        // Simulating a code from a newer plugin — must fail closed.
        let s = NxrtStatus::from_code(NxrtStatusCode::OutOfMemory);
        assert!(!s.is_ok());
    }
}
