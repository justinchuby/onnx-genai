//! # `onnx-runtime-ep-nxrt-abi`
//!
//! Stable versioned C ABI for native nxrt execution-provider plugins (§524).
//!
//! This crate defines the binary interface that an nxrt EP plugin exposes as a
//! `cdylib`. The host (`onnx-runtime-ep-nxrt-host`) `dlopen`s the library,
//! negotiates the ABI version, and obtains vtable-based handles for factory,
//! EP, and kernel objects.
//!
//! # Design principles (improvements over ORT plugin ABI)
//!
//! 1. **Explicit ownership**: every pointer-returning function documents who
//!    owns the result. Borrowed pointers are valid only within the callback
//!    frame unless stated otherwise. Owned pointers have a paired release fn.
//! 2. **Fail closed**: incompatible major version → hard rejection with an
//!    actionable error message. Unknown capability bits → rejected.
//! 3. **Panic containment**: every `extern "C"` entry catches panics. Status-
//!    returning functions produce an error code; void-returning ones swallow.
//! 4. **Forward compatibility**: struct-size-based versioning + capability
//!    flags. Adding fields never corrupts older consumers.
//!
//! # Exported symbols
//!
//! A conforming nxrt plugin `cdylib` exports exactly two symbols:
//! - [`NXRT_SYMBOL_NEGOTIATE`] (`NxrtNegotiate`) — version handshake
//! - [`NXRT_SYMBOL_CREATE_EP_FACTORIES`] (`NxrtCreateEpFactories`) — factory creation
//!
//! The host calls `NxrtNegotiate` first. If it succeeds, the host calls
//! `NxrtCreateEpFactories`. Factories and EPs are released through vtable
//! function pointers — no separate release symbol needed.

#![deny(clippy::mem_forget)]

pub mod status;
pub mod version;
pub mod vtable;

pub use status::{NxrtStatus, NxrtStatusCode};
pub use version::{
    NXRT_ABI_VERSION_MAJOR, NXRT_ABI_VERSION_MINOR, NxrtNegotiateRequest, NxrtNegotiateResponse,
    NxrtVersionRange,
};
pub use vtable::{
    NxrtAllocatorVtable, NxrtEpFactoryVtable, NxrtEpVtable, NxrtKernelVtable,
    NxrtNodeCapability, NxrtTensorDesc,
};

// ─── Exported symbol name constants ─────────────────────────────────────────

/// Symbol for version negotiation: `NxrtNegotiate`.
pub const NXRT_SYMBOL_NEGOTIATE: &[u8] = b"NxrtNegotiate";

/// Symbol for factory creation: `NxrtCreateEpFactories`.
pub const NXRT_SYMBOL_CREATE_EP_FACTORIES: &[u8] = b"NxrtCreateEpFactories";

// ─── Function pointer types for the two exported symbols ────────────────────

/// Signature of `NxrtNegotiate`.
///
/// # Ownership
///
/// `request` is borrowed for the duration of the call (caller owns).
/// `response_out` is a caller-provided buffer the plugin writes into.
///
/// # Contract
///
/// Returns [`NxrtStatusCode::Ok`] if negotiation succeeds. On failure the
/// plugin fills `response_out` with the range it supports and returns
/// [`NxrtStatusCode::VersionMismatch`].
pub type NxrtNegotiateFn = unsafe extern "C" fn(
    request: *const NxrtNegotiateRequest,
    response_out: *mut NxrtNegotiateResponse,
) -> NxrtStatus;

/// Signature of `NxrtCreateEpFactories`.
///
/// # Ownership
///
/// Each `*mut NxrtEpFactoryVtable` written to `out_factories` is **owned by
/// the caller** (the host) — the host must eventually call the factory's
/// `release` vtable function to free it.
///
/// # Contract
///
/// Writes up to `max_factories` factory pointers into `out_factories` and
/// sets `*out_num` to the count written. Returns a status.
pub type NxrtCreateEpFactoriesFn = unsafe extern "C" fn(
    out_factories: *mut *mut NxrtEpFactoryVtable,
    max_factories: usize,
    out_num: *mut usize,
) -> NxrtStatus;

// ─── Export macro ───────────────────────────────────────────────────────────

/// Generates the two required nxrt plugin entry points for a `cdylib`.
///
/// # Usage
///
/// ```rust,ignore
/// use onnx_runtime_ep_nxrt_abi::export_nxrt_ep_factories;
/// use my_ep::MyExecutionProvider;
///
/// export_nxrt_ep_factories!(|| MyExecutionProvider::new());
/// ```
///
/// This exports `NxrtNegotiate` and `NxrtCreateEpFactories` with full panic
/// containment. The closure is called once per `NxrtCreateEpFactories` call.
#[macro_export]
macro_rules! export_nxrt_ep_factories {
    ($constructor:expr) => {
        /// nxrt ABI version negotiation entry point.
        ///
        /// # Safety
        ///
        /// Pointers must be valid and non-null.
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn NxrtNegotiate(
            request: *const $crate::NxrtNegotiateRequest,
            response_out: *mut $crate::NxrtNegotiateResponse,
        ) -> $crate::NxrtStatus {
            let result = ::std::panic::catch_unwind(::std::panic::AssertUnwindSafe(|| {
                // SAFETY: caller guarantees pointer validity.
                unsafe { $crate::version::negotiate(request, response_out) }
            }));
            match result {
                ::std::result::Result::Ok(status) => status,
                ::std::result::Result::Err(_) => {
                    $crate::NxrtStatus::from_code($crate::NxrtStatusCode::InternalError)
                }
            }
        }

        /// nxrt EP factory creation entry point.
        ///
        /// # Safety
        ///
        /// All pointers must be valid per the nxrt ABI contract.
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn NxrtCreateEpFactories(
            out_factories: *mut *mut $crate::NxrtEpFactoryVtable,
            max_factories: usize,
            out_num: *mut usize,
        ) -> $crate::NxrtStatus {
            let out_factories_raw = out_factories;
            let out_num_raw = out_num;
            let result = ::std::panic::catch_unwind(::std::panic::AssertUnwindSafe(|| {
                unsafe {
                    $crate::vtable::create_ep_factories(
                        out_factories_raw,
                        max_factories,
                        out_num_raw,
                        $constructor,
                    )
                }
            }));
            match result {
                ::std::result::Result::Ok(status) => status,
                ::std::result::Result::Err(_) => {
                    if !out_num_raw.is_null() {
                        // SAFETY: pointer validity is caller's responsibility.
                        unsafe { *out_num_raw = 0 };
                    }
                    $crate::NxrtStatus::from_code_with_message(
                        $crate::NxrtStatusCode::InternalError,
                        "NxrtCreateEpFactories: constructor panicked (fail-closed)",
                    )
                }
            }
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn symbol_names_are_non_empty() {
        assert!(!NXRT_SYMBOL_NEGOTIATE.is_empty());
        assert!(!NXRT_SYMBOL_CREATE_EP_FACTORIES.is_empty());
    }
}
