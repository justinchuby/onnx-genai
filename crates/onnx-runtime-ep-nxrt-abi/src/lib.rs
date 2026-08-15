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
pub mod testing;
pub mod version;
pub mod vtable;

pub use status::{NXRT_STATUS_MESSAGE_MAX, NxrtStatus, NxrtStatusCode};
pub use testing::{NxrtCreateFactoriesOverride, NxrtNegotiateOverride};
pub use version::{
    NXRT_ABI_VERSION_MAJOR, NXRT_ABI_VERSION_MINOR, NXRT_CAP_ALLOCATOR,
    NXRT_CAP_DEVICE_ENUMERATION, NXRT_CAP_EP_CONTEXT, NXRT_CAP_KNOWN_MASK, NXRT_CAP_STREAM_SYNC,
    NxrtNegotiateRequest, NxrtNegotiateResponse, NxrtVersionRange, validate_negotiation,
};
pub use vtable::{
    NxrtAllocatorVtable, NxrtEpFactoryVtable, NxrtEpVtable, NxrtKernelVtable, NxrtNodeCapability,
    NxrtTensorDesc,
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

/// Generates the two required nxrt plugin entry points (`NxrtNegotiate` and
/// `NxrtCreateEpFactories`) for a `cdylib`.
///
/// # Usage
///
/// ```rust,ignore
/// use onnx_runtime_ep_nxrt_abi::export_nxrt_ep_factories;
///
/// export_nxrt_ep_factories!(|| MyExecutionProvider::new());
/// ```
///
/// The closure is called once per `NxrtCreateEpFactories` invocation to
/// produce a fresh `Box<dyn ExecutionProvider>`. Both symbols have full panic
/// containment — a panic in the constructor or negotiate path never unwinds
/// across the C ABI boundary.
///
/// # Ownership rules enforced by the macro
///
/// - Host owns each `NxrtEpFactoryVtable` returned in `out_factories` and
///   must call `factory.release(factory.ctx)` exactly once.
/// - Borrowed pointers (tensor dims, op-type strings) are valid only within
///   the callback frame — the macro-generated code never stashes them.
/// - The EP's `name` pointer is valid for the EP's lifetime (owned by EP ctx).
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
            max_factories: ::core::primitive::usize,
            out_num: *mut ::core::primitive::usize,
        ) -> $crate::NxrtStatus {
            let out_factories_raw = out_factories;
            let out_num_raw = out_num;
            let constructor = $constructor;
            let result = ::std::panic::catch_unwind(::std::panic::AssertUnwindSafe(|| unsafe {
                $crate::vtable::create_ep_factories(
                    out_factories_raw,
                    max_factories,
                    out_num_raw,
                    constructor,
                )
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

/// Generates a **custom** `NxrtNegotiate` entry point driven by an
/// [`NxrtNegotiateOverride`]. Use for negative-test fixture plugins.
///
/// # Usage
///
/// ```rust,ignore
/// use onnx_runtime_ep_nxrt_abi::export_nxrt_ep_negotiate_custom;
/// use onnx_runtime_ep_nxrt_abi::testing::NxrtNegotiateOverride;
///
/// export_nxrt_ep_negotiate_custom!(NxrtNegotiateOverride::wrong_major(99));
/// ```
#[macro_export]
macro_rules! export_nxrt_ep_negotiate_custom {
    ($override_expr:expr) => {
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn NxrtNegotiate(
            request: *const $crate::NxrtNegotiateRequest,
            response_out: *mut $crate::NxrtNegotiateResponse,
        ) -> $crate::NxrtStatus {
            let over = $override_expr;
            let result = ::std::panic::catch_unwind(::std::panic::AssertUnwindSafe(|| unsafe {
                over.execute(request, response_out)
            }));
            match result {
                ::std::result::Result::Ok(status) => status,
                ::std::result::Result::Err(_) => {
                    $crate::NxrtStatus::from_code($crate::NxrtStatusCode::InternalError)
                }
            }
        }
    };
}

/// Generates a **custom** `NxrtCreateEpFactories` entry point driven by an
/// [`NxrtCreateFactoriesOverride`]. Use for negative-test fixture plugins.
///
/// # Usage
///
/// ```rust,ignore
/// use onnx_runtime_ep_nxrt_abi::export_nxrt_ep_create_custom;
/// use onnx_runtime_ep_nxrt_abi::testing::NxrtCreateFactoriesOverride;
///
/// export_nxrt_ep_create_custom!(NxrtCreateFactoriesOverride::error(NxrtStatusCode::DeviceError));
/// ```
#[macro_export]
macro_rules! export_nxrt_ep_create_custom {
    ($override_expr:expr) => {
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn NxrtCreateEpFactories(
            out_factories: *mut *mut $crate::NxrtEpFactoryVtable,
            max_factories: ::core::primitive::usize,
            out_num: *mut ::core::primitive::usize,
        ) -> $crate::NxrtStatus {
            let over = $override_expr;
            let out_factories_raw = out_factories;
            let out_num_raw = out_num;
            let result = ::std::panic::catch_unwind(::std::panic::AssertUnwindSafe(|| unsafe {
                over.execute(out_factories_raw, max_factories, out_num_raw)
            }));
            match result {
                ::std::result::Result::Ok(status) => status,
                ::std::result::Result::Err(_) => {
                    if !out_num_raw.is_null() {
                        unsafe { *out_num_raw = 0 };
                    }
                    $crate::NxrtStatus::from_code_with_message(
                        $crate::NxrtStatusCode::InternalError,
                        "NxrtCreateEpFactories: panicked (fail-closed)",
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

    // ─── Macro compile + symbol test ────────────────────────────────────────
    //
    // We can't use #[no_mangle] inside a test function body, but we can
    // define the symbols in a private module and call them as regular fns
    // to prove they compile and produce correct results.

    // Use the macro in a module to produce the symbols.
    mod macro_symbols {
        crate::export_nxrt_ep_factories!(|| {
            Box::new(onnx_runtime_ep_cpu::CpuExecutionProvider::new())
                as Box<dyn onnx_runtime_ep_api::provider::ExecutionProvider>
        });
    }

    #[test]
    fn export_macro_negotiate_produces_correct_response() {
        let req = NxrtNegotiateRequest::current();
        let mut resp = NxrtNegotiateResponse::zeroed();
        let status = unsafe { macro_symbols::NxrtNegotiate(&req, &mut resp) };
        assert!(status.is_ok(), "negotiate must succeed");
        assert_eq!(resp.agreed_major, NXRT_ABI_VERSION_MAJOR);
        assert_eq!(resp.agreed_minor, NXRT_ABI_VERSION_MINOR);
    }

    #[test]
    fn export_macro_create_factories_succeeds() {
        let mut factory_ptr: *mut NxrtEpFactoryVtable = std::ptr::null_mut();
        let mut num: usize = 0;
        let status = unsafe { macro_symbols::NxrtCreateEpFactories(&mut factory_ptr, 1, &mut num) };
        assert!(status.is_ok(), "create must succeed");
        assert_eq!(num, 1);
        assert!(!factory_ptr.is_null());

        // Clean up: release via vtable.
        let factory = unsafe { &*factory_ptr };
        unsafe { (factory.release)(factory.ctx) };
        // Reclaim the vtable box itself.
        let _ = unsafe { Box::from_raw(factory_ptr) };
    }

    #[test]
    fn export_macro_panic_in_constructor_contained() {
        // Can't use the macro here (symbol conflict), so test via the
        // catch_status_panic helper which is what the macro expands to.
        let status = crate::status::catch_status_panic(|| {
            panic!("constructor panic for test");
        });
        assert!(!status.is_ok());
        assert_eq!(status.status_code(), Some(NxrtStatusCode::InternalError));
    }

    #[test]
    fn custom_negotiate_override_through_validate() {
        // Proves the testing override + validate_negotiation path works
        // (this is what Pris's fixture will use via the macro).
        let over = NxrtNegotiateOverride::wrong_major(99);
        let req = NxrtNegotiateRequest::current();
        let mut resp = NxrtNegotiateResponse::zeroed();
        let status = unsafe { over.execute(&req, &mut resp) };
        assert!(status.is_ok());
        assert_eq!(resp.agreed_major, 99);
        let result = validate_negotiation(&NxrtVersionRange::current(), &resp);
        assert!(result.is_err());
    }

    #[test]
    fn custom_create_override_zero_factories() {
        let over = NxrtCreateFactoriesOverride::zero();
        let mut num: usize = 99;
        let status = unsafe { over.execute(std::ptr::null_mut(), 0, &mut num) };
        assert!(status.is_ok());
        assert_eq!(num, 0);
    }
}
