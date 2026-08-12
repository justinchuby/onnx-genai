#![allow(dead_code)]
//! Cross-platform helper for encoding `Path` into the ORT `ORTCHAR_T*` type.
//!
//! On Windows, ORT path-taking APIs (`CreateSession`, `RegisterExecutionProviderLibrary`, …)
//! expect `*const u16` (NUL-terminated UTF-16, matching `wchar_t`).
//! On Unix they expect `*const c_char` (NUL-terminated UTF-8).
//!
//! This module provides [`OrtPathBuf`] which encodes a `&Path` into the
//! correct representation and exposes `.as_ptr()` with the matching pointer type.
//! The encoded buffer owns the data, so the caller just needs to keep the
//! `OrtPathBuf` alive for the duration of the FFI call.

use std::path::Path;

/// Platform-correct, NUL-terminated encoding of a filesystem path for ORT APIs.
///
/// # Lifetime
///
/// The `.as_ptr()` return borrows `self` — bind the `OrtPathBuf` to a local
/// variable that outlives every FFI call that uses the pointer.
pub struct OrtPathBuf {
    #[cfg(windows)]
    buf: Vec<u16>,
    #[cfg(not(windows))]
    buf: std::ffi::CString,
}

impl OrtPathBuf {
    /// Encode `path` into the platform-correct ORT representation.
    ///
    /// # Panics
    ///
    /// Panics if the path contains an interior NUL byte (which would be
    /// invalid for any OS path anyway).
    pub fn new(path: impl AsRef<Path>) -> Self {
        let path = path.as_ref();
        #[cfg(windows)]
        {
            use std::os::windows::ffi::OsStrExt;
            let mut wide: Vec<u16> = path.as_os_str().encode_wide().collect();
            assert!(
                !wide.contains(&0),
                "ORT path contains interior NUL: {path:?}"
            );
            wide.push(0); // NUL terminator
            Self { buf: wide }
        }
        #[cfg(not(windows))]
        {
            let s = path
                .to_str()
                .unwrap_or_else(|| panic!("ORT path is not valid UTF-8: {path:?}"));
            Self {
                buf: std::ffi::CString::new(s)
                    .unwrap_or_else(|_| panic!("ORT path contains interior NUL: {path:?}")),
            }
        }
    }

    /// Return a pointer suitable for passing to ORT `ORTCHAR_T*` parameters.
    ///
    /// On Windows this is `*const u16`; on Unix `*const c_char`.
    #[cfg(windows)]
    pub fn as_ptr(&self) -> *const u16 {
        self.buf.as_ptr()
    }

    #[cfg(not(windows))]
    pub fn as_ptr(&self) -> *const std::os::raw::c_char {
        self.buf.as_ptr()
    }
}
