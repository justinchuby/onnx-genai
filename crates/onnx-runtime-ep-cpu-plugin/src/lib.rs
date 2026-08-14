//! CPU execution provider exported as an ORT plugin-EP cdylib.
//!
//! This crate produces `libonnx_runtime_ep_cpu_plugin.so` (or platform
//! equivalent) that upstream ONNX Runtime can load via `dlopen` and use as a
//! real execution provider.
//!
//! The crate is intentionally thin: construct the EP, derive kernel-registry
//! entries from the real CPU registry, and export via the C ABI.

use onnx_runtime_ep_cpu::{CpuExecutionProvider, build_cpu_registry_with_descriptors};
use onnx_runtime_ep_plugin::ep::KernelRegistryEntry;

/// Build `KernelRegistryEntry` slices from the CPU EP's real registry.
///
/// Each entry's `supported_dtypes` is derived from the kernel's actual dispatch
/// implementation via `supported_dtypes_for_op` — fail closed (f32-only for
/// unknown ops). f16/bf16 are advertised only for ops whose kernels genuinely
/// handle them (Add, Sub, Mul, MatMul, etc.).
fn build_kernel_registry_entries() -> Vec<KernelRegistryEntry> {
    let (_registry, descriptors) = build_cpu_registry_with_descriptors();
    descriptors
        .into_iter()
        .map(|d| {
            // Clamp since_version to i32 range (always fits for ONNX opsets).
            let since = d.since_version as i32;
            KernelRegistryEntry {
                op_type: leak_str(&d.op_type),
                domain: leak_str(&d.domain),
                since_version: since,
                // Cover all future opset versions so ORT matches our kernel
                // regardless of the model's declared opset (e.g. Add@7 must
                // still match a model at opset 21).
                end_version: i32::MAX,
                supported_dtypes: d.supported_dtypes,
            }
        })
        .collect()
}

/// Leak a string to get a `&'static str` (the entries must live for the EP lifetime).
fn leak_str(s: &str) -> &'static str {
    Box::leak(s.to_owned().into_boxed_str())
}

/// ORT plugin-EP entry point: create EP factories with kernel-registry type
/// constraints for f16/bf16 routing.
///
/// # Why hand-written (not `export_ep_factories!`)
///
/// The macro calls `factory::create_ep_factories`, which does not accept a
/// kernel-registry slice. This shim needs
/// `factory::create_ep_factories_with_registry` to advertise typed dtype
/// constraints to ORT. There is no macro variant for that path yet, so the
/// function is written explicitly here. The signature MUST remain identical
/// to the `CreateEpFactories` arm of `export_ep_factories!` in
/// `onnx-runtime-ep-plugin/src/lib.rs`; if that macro arm changes, update
/// this function in lockstep.
///
/// # Safety
///
/// Called by ORT's plugin loader. All pointer arguments must be valid per the
/// ORT plugin-EP C ABI contract.
///
/// # Panic safety
///
/// Any panic is caught; on panic, `*out_num` is set to `0` and an error
/// `OrtStatus` is returned.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn CreateEpFactories(
    _registration_name: *const ::std::ffi::c_char,
    api_base: *const onnx_runtime_ep_plugin::onnx_genai_ort_sys::OrtApiBase,
    _logger: *const onnx_runtime_ep_plugin::onnx_genai_ort_sys::OrtLogger,
    out_factories: *mut *mut onnx_runtime_ep_plugin::onnx_genai_ort_sys::OrtEpFactory,
    max_factories: usize,
    out_num: *mut usize,
) -> *mut onnx_runtime_ep_plugin::onnx_genai_ort_sys::OrtStatus {
    let out_factories_raw = out_factories;
    let out_num_raw = out_num;
    let result = ::std::panic::catch_unwind(::std::panic::AssertUnwindSafe(|| {
        let entries = build_kernel_registry_entries();
        unsafe {
            onnx_runtime_ep_plugin::factory::create_ep_factories_with_registry(
                api_base,
                out_factories_raw,
                max_factories,
                out_num_raw,
                || Box::new(CpuExecutionProvider::new()),
                entries,
            )
        }
    }));
    match result {
        Ok(status) => status,
        Err(_panic_payload) => {
            if !out_num_raw.is_null() {
                unsafe { *out_num_raw = 0 };
            }
            onnx_runtime_ep_plugin::panic_to_fail_status(
                "CreateEpFactories: constructor panicked; plugin not loaded (fail-closed)",
            )
        }
    }
}

/// ORT plugin-EP entry point: release an EP factory.
///
/// # ABI reference
///
/// `onnxruntime_ep_c_api.h:2669`:
/// ```c
/// typedef OrtStatus* (*ReleaseEpApiFactoryFn)(_In_ OrtEpFactory* factory);
/// ```
/// Returns `nullptr` on success or a non-null `OrtStatus*` on error/panic.
///
/// # Safety
///
/// `factory` must be a pointer returned by `CreateEpFactories` from this
/// library, and must not be used after this call.
///
/// # Panic safety
///
/// Any panic inside the release path is caught and surfaced as a failure
/// `OrtStatus`. Unwinding into ORT would be undefined behaviour.
///
/// # Kept in sync with `export_ep_factories!`
///
/// This shim stays hand-written because `CreateEpFactories` above calls
/// `create_ep_factories_with_registry` (not available in the macro). The body
/// below is intentionally identical to the `ReleaseEpFactory` arm of the
/// `export_ep_factories!` macro in `onnx-runtime-ep-plugin/src/lib.rs`.
/// If that macro arm changes, this must change in lockstep.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ReleaseEpFactory(
    factory: *mut onnx_runtime_ep_plugin::onnx_genai_ort_sys::OrtEpFactory,
) -> *mut onnx_runtime_ep_plugin::onnx_genai_ort_sys::OrtStatus {
    let result = ::std::panic::catch_unwind(::std::panic::AssertUnwindSafe(|| {
        // SAFETY: caller guarantees the pointer was returned by
        // CreateEpFactories from this library.
        unsafe { onnx_runtime_ep_plugin::factory::release_ep_factory(factory) }
    }));
    match result {
        ::std::result::Result::Ok(status) => status,
        ::std::result::Result::Err(_panic_payload) => onnx_runtime_ep_plugin::panic_to_fail_status(
            "ReleaseEpFactory: panic during factory release (fail-closed)",
        ),
    }
}

// ─── Test observability: compiled-node counter ───────────────────────────────

/// Returns the number of nodes compiled by our EP since last reset.
/// Exported as a C symbol so integration tests (which load us via dlopen)
/// can assert that our EP actually claimed and compiled nodes.
#[unsafe(no_mangle)]
pub extern "C" fn nxrt_ep_compiled_node_count() -> usize {
    onnx_runtime_ep_plugin::ep::compiled_node_count()
}

/// Resets the compiled-node counter to zero.
#[unsafe(no_mangle)]
pub extern "C" fn nxrt_ep_reset_compiled_node_count() {
    onnx_runtime_ep_plugin::ep::reset_compiled_node_count()
}

/// Number of workspace **placement resolutions** since the last reset.
///
/// Mirrors the CUDA plugin's accessor so the same validation harness runs
/// against either cdylib: on a GPU-less host it can be pointed at this library
/// to exercise its own plumbing before the GPU host runs it for real.
#[unsafe(no_mangle)]
pub extern "C" fn nxrt_ep_workspace_placement_queries() -> usize {
    onnx_runtime_ep_plugin::compute::workspace_placement_queries()
}

/// Resets the workspace placement-resolution counter to zero.
#[unsafe(no_mangle)]
pub extern "C" fn nxrt_ep_reset_workspace_placement_queries() {
    onnx_runtime_ep_plugin::compute::reset_workspace_placement_queries()
}
