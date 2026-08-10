//! Outbound ORT plugin-EP adapter.
//!
//! Projects any nxrt [`ExecutionProvider`] through the ORT plugin-EP C ABI so
//! upstream ONNX Runtime can `dlopen` a cdylib and load it as a real EP.
//!
//! # Architecture
//!
//! All `unsafe` FFI lives here. Per-EP shim crates are a few lines: construct
//! the Rust EP and invoke [`export_ep_factories!`].
//!
//! # Export symbol name
//!
//! **ASSUMED: `CreateEpFactories`** — this matches the symbol name the inbound
//! loader in `onnx-runtime-ep-api/src/abi/runtime.rs` resolves. Pris's test plan
//! says `CreateEpApiFactories`; Challenger is verifying against the real ORT
//! 1.27.0 header. The name is behind [`EXPORT_SYMBOL_CREATE`] /
//! [`EXPORT_SYMBOL_RELEASE`] constants so a one-line change fixes it.
//!
//! # Deferred (fail-closed, not faked)
//!
//! - Allocator callbacks (CPU EP uses host malloc — not needed)
//! - Data transfer / MemCpy callbacks
//! - EPContext save/load
//! - Custom ops

#![allow(clippy::missing_safety_doc)] // FFI callbacks documented at call sites

use onnx_genai_ort_sys as ort;

pub mod factory;
pub mod ep;
pub mod compute;
pub mod graph_reader;
pub mod kernel_ctx;
pub mod status;

pub use factory::ExportedFactory;
pub use ep::ExportedEp;

/// Re-export of `onnx_genai_ort_sys` for use by the `export_ep_factories!` macro.
#[doc(hidden)]
pub use onnx_genai_ort_sys;

// ─── Export symbol name constants ────────────────────────────────────────────
// If Challenger's verdict says the real ORT 1.27 header uses a different name,
// change ONLY these two constants.

/// The C symbol name ORT looks up via `dlsym` to create EP factories.
///
/// **Assumed:** `CreateEpFactories` (matches inbound loader in
/// `onnx-runtime-ep-api/src/abi/runtime.rs:57`). May need to be
/// `CreateEpApiFactories` per Pris — pending Challenger verification.
pub const EXPORT_SYMBOL_CREATE: &[u8] = b"CreateEpFactories";

/// The C symbol name ORT looks up to release an EP factory.
pub const EXPORT_SYMBOL_RELEASE: &[u8] = b"ReleaseEpFactory";

/// ORT API version we were built against. Used for version negotiation.
pub const ORT_API_VERSION_SUPPORTED: u32 = ort::ORT_API_VERSION;

/// Generates `#[unsafe(no_mangle)] pub extern "C"` entry points for the ORT
/// plugin-EP C ABI.
///
/// # Usage
///
/// ```rust,ignore
/// use onnx_runtime_ep_cpu::CpuExecutionProvider;
/// use onnx_runtime_ep_plugin::export_ep_factories;
///
/// export_ep_factories!(|| CpuExecutionProvider::new());
/// ```
///
/// This expands to `CreateEpFactories` and `ReleaseEpFactory` symbols. The
/// closure is called once per `CreateEpFactories` invocation to produce a fresh
/// EP instance.
#[macro_export]
macro_rules! export_ep_factories {
    ($constructor:expr) => {
        /// ORT plugin-EP entry point: create EP factories.
        ///
        /// # Symbol name
        ///
        /// Exported as `CreateEpFactories`. If ORT 1.27+ uses a different name,
        /// update [`onnx_runtime_ep_plugin::EXPORT_SYMBOL_CREATE`].
        ///
        /// # Safety
        ///
        /// Called by ORT's plugin loader. All pointer arguments must be valid per
        /// the ORT plugin-EP C ABI contract.
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn CreateEpFactories(
            _registration_name: *const ::std::ffi::c_char,
            api_base: *const $crate::onnx_genai_ort_sys::OrtApiBase,
            _logger: *const $crate::onnx_genai_ort_sys::OrtLogger,
            out_factories: *mut *mut $crate::onnx_genai_ort_sys::OrtEpFactory,
            max_factories: usize,
            out_num: *mut usize,
        ) -> *mut $crate::onnx_genai_ort_sys::OrtStatus {
            $crate::factory::create_ep_factories(
                api_base,
                out_factories,
                max_factories,
                out_num,
                $constructor,
            )
        }

        /// ORT plugin-EP entry point: release an EP factory.
        ///
        /// # Safety
        ///
        /// `factory` must be a pointer returned by `CreateEpFactories` from this
        /// library, and must not be used after this call.
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn ReleaseEpFactory(
            factory: *mut $crate::onnx_genai_ort_sys::OrtEpFactory,
        ) -> *mut $crate::onnx_genai_ort_sys::OrtStatus {
            $crate::factory::release_ep_factory(factory)
        }
    };
}
