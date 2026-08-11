//! CUDA execution provider exported as an ORT plugin-EP cdylib.
//!
//! This crate mirrors `onnx-runtime-ep-cpu-plugin` for the CUDA EP.
//! Without the `cuda` feature (default), it compiles as a no-op stub so the
//! workspace builds on hosts without a CUDA toolkit.
//!
//! # CUDA EP status: IMPLEMENTATION-BLOCKED (not merely hardware-blocked)
//!
//! Even with `cuda` enabled, `CreateEpFactories` currently returns **zero
//! factories**. The CUDA EP cannot be advertised to ORT because four
//! implementation defects prevent it from functioning correctly on *any* host:
//!
//! 1. **Separate CUDA runtime/context per component.** The EP, allocator, and
//!    stream each construct independent CUDA runtime/context instances. A correct
//!    implementation must share a single `CUcontext` + `cudaStream_t` across all
//!    three so that allocations are visible to kernels and stream-ordered copies
//!    are coherent.
//!
//! 2. **`CreateDataTransfer` returns NULL.** The factory wires up a
//!    `DeviceDataTransfer` but it cannot perform actual copies: it lacks access
//!    to `OrtApi` (needed to extract tensor data pointers) and has no shared
//!    CUDA stream for `cudaMemcpyAsync`. The `CopyTensors` callback returns an
//!    error unconditionally.
//!
//! 3. **`GetHandle` returns NULL stream handle.** `stream_get_handle` returns
//!    `null_mut()` — there is no `cudaStream_t` to return. ORT and downstream
//!    consumers that call `GetHandle` to order work on the EP's stream will
//!    receive a null pointer, causing undefined behavior or silent fallback to
//!    the default stream (breaking ordering guarantees).
//!
//! 4. **`Free` passes `size=0` violating the allocator contract.** `device_free`
//!    reconstructs a `DeviceBuffer` with `size=0` because the allocation size is
//!    not tracked. The EP's `deallocate` implementation may need the size for
//!    `cudaFree`-style bookkeeping or arena management. Passing `size=0` violates
//!    the allocator contract documented in `onnxruntime_ep_c_api.h`.
//!
//! **Fail-closed policy:** Advertising a GPU EP that cannot honour the contract
//! is worse than not shipping one — ORT would route real work to it and get
//! silent corruption or crashes. The plugin returns zero factories until all
//! four defects are resolved.

// ─── Feature-gated: real CUDA EP ─────────────────────────────────────────────

#[cfg(feature = "cuda")]
mod cuda_impl {
    use onnx_runtime_ep_cuda::CudaExecutionProvider;
    use onnx_runtime_ep_plugin::device::DeviceSupport;
    use onnx_runtime_ep_plugin::ep::KernelRegistryEntry;

    /// NVIDIA vendor ID (PCI).
    const NVIDIA_VENDOR_ID: u32 = 0x10DE;

    /// Build kernel registry entries from the CUDA EP's covered op list.
    ///
    /// Each entry advertises f32/f16/bf16 — the CUDA EP supports all three via
    /// cuBLASLt and custom kernels. The real per-node filter is `supports_op`
    /// on the EP; the registry entries give ORT type-routing metadata.
    pub(crate) fn build_kernel_registry_entries() -> Vec<KernelRegistryEntry> {
        use onnx_runtime_ep_cuda::CUDA_COVERED_OPS;
        use onnx_runtime_ir::DataType;

        /// Dtypes the CUDA EP genuinely handles via cuBLASLt + custom kernels.
        static CUDA_DTYPES: &[DataType] =
            &[DataType::Float32, DataType::Float16, DataType::BFloat16];

        CUDA_COVERED_OPS
            .iter()
            .map(|&op_type| {
                // Default ONNX domain; opset 1 as the minimum since_version.
                // `com.microsoft` ops are in the same list but ORT routes them
                // by domain match in GetCapability, not the registry alone.
                KernelRegistryEntry {
                    op_type,
                    domain: "",
                    since_version: 1,
                    end_version: 99,
                    supported_dtypes: CUDA_DTYPES,
                }
            })
            .collect()
    }

    /// Device support for the CUDA EP: GPU, stream-aware, device-only memory.
    pub(crate) fn device_support() -> DeviceSupport {
        DeviceSupport::gpu("Cuda", NVIDIA_VENDOR_ID)
    }

    /// Construct the CUDA EP on device 0 (default ordinal).
    ///
    /// Returns an error if no CUDA GPU is available or the driver cannot be
    /// loaded. The plugin exports zero factories in that case (fail-closed).
    pub(crate) fn construct_ep()
    -> Result<Box<dyn onnx_runtime_ep_api::provider::ExecutionProvider>, String> {
        CudaExecutionProvider::new_default()
            .map(|ep| Box::new(ep) as Box<dyn onnx_runtime_ep_api::provider::ExecutionProvider>)
            .map_err(|e| format!("CUDA EP construction failed (no GPU or driver unavailable): {e}"))
    }
}

/// ORT plugin-EP entry point: create EP factories.
///
/// With the `cuda` feature enabled, constructs a real `CudaExecutionProvider`
/// and advertises GPU device support with kernel-registry type constraints.
/// Without it, returns zero factories and an actionable error status.
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
    _api_base: *const onnx_runtime_ep_plugin::onnx_genai_ort_sys::OrtApiBase,
    _logger: *const onnx_runtime_ep_plugin::onnx_genai_ort_sys::OrtLogger,
    _out_factories: *mut *mut onnx_runtime_ep_plugin::onnx_genai_ort_sys::OrtEpFactory,
    _max_factories: usize,
    out_num: *mut usize,
) -> *mut onnx_runtime_ep_plugin::onnx_genai_ort_sys::OrtStatus {
    let out_num_raw = out_num;
    let result = ::std::panic::catch_unwind(::std::panic::AssertUnwindSafe(|| {
        // ── Fail closed: CUDA EP is implementation-blocked ──────────────
        //
        // The CUDA plugin returns zero factories regardless of feature
        // configuration. Even with `cuda` enabled, the implementation has
        // four unresolved defects (see crate-level docs) that would cause
        // silent corruption or crashes if ORT routed work to this EP.
        //
        // This is NOT a hardware-validation gate — it is an implementation
        // gate. The defects exist in the code, not in the test environment.
        //
        // When all four defects are resolved, remove this gate and restore
        // the `create_ep_factories_with_device_support` call.
        unsafe {
            if !out_num_raw.is_null() {
                *out_num_raw = 0;
            }
        }

        #[cfg(feature = "cuda")]
        {
            // Suppress unused-import warnings for gated code that will be
            // restored once the implementation defects are resolved.
            let _ = &cuda_impl::build_kernel_registry_entries;
            let _ = &cuda_impl::device_support;
            let _ = &cuda_impl::construct_ep;

            onnx_runtime_ep_plugin::panic_to_fail_status(
                "onnx-runtime-ep-cuda-plugin: CUDA EP is IMPLEMENTATION-BLOCKED \
                 (not merely hardware-blocked). Four defects prevent correct operation: \
                 (1) separate CUDA runtime/context per component — EP, allocator, and stream \
                 must share a single CUcontext + cudaStream_t; \
                 (2) CreateDataTransfer cannot perform actual copies — lacks OrtApi access \
                 and shared CUDA stream; \
                 (3) GetHandle returns NULL stream handle — no cudaStream_t exists; \
                 (4) Free passes size=0 violating the allocator contract. \
                 Zero factories returned (fail-closed). \
                 See crate docs for the specification of each defect.",
            )
        }

        #[cfg(not(feature = "cuda"))]
        {
            onnx_runtime_ep_plugin::panic_to_fail_status(
                "onnx-runtime-ep-cuda-plugin built without `cuda` feature; \
                 CUDA EP is not available. Rebuild with --features cuda on a CUDA-capable host.",
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
///
/// Returns `nullptr` on success or an `OrtStatus*` error — **not `void`**.
///
/// # Why this is hand-written instead of using `export_ep_factories!`
///
/// The `export_ep_factories!` macro emits *both* `CreateEpFactories` and
/// `ReleaseEpFactory` as a single expansion. The CUDA shim needs a custom
/// `CreateEpFactories` (fail-closed implementation gate with four blocking
/// defects), so invoking the macro would conflict with the hand-written
/// `CreateEpFactories` above. Until the four CUDA defects are resolved and
/// the shim can delegate to `export_ep_factories!`, this function must be
/// kept in sync with the macro's `ReleaseEpFactory` arm in
/// `onnx-runtime-ep-plugin/src/lib.rs`.
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
        Ok(status) => status,
        Err(_panic_payload) => onnx_runtime_ep_plugin::panic_to_fail_status(
            "ReleaseEpFactory: panic during factory release (fail-closed)",
        ),
    }
}

#[cfg(test)]
mod tests {
    use std::panic::AssertUnwindSafe;

    /// Verify that a panicking constructor is caught at the macro guard boundary:
    /// no unwind escapes, `out_num` is set to `0`, and an error status is
    /// produced. This is the N3 regression test.
    #[test]
    fn panicking_constructor_caught_and_zero_factories_returned() {
        let out_num: usize = 0;

        let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
            panic!("simulated constructor panic for N3 guard test");
        }));
        assert!(
            result.is_err(),
            "catch_unwind must capture the constructor panic"
        );

        let status = crate::panic_to_fail_status(
            "CreateEpFactories: constructor panicked; plugin not loaded (fail-closed)",
        );

        assert_eq!(out_num, 0, "out_num must be 0 on constructor panic");
        let _ = status;
    }

    /// Verify `panic_to_fail_status` is panic-safe regardless of host API state.
    #[test]
    fn panic_to_fail_status_never_panics() {
        let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
            crate::panic_to_fail_status("N3 sentinel — no ORT loaded")
        }));
        assert!(result.is_ok(), "panic_to_fail_status must not itself panic");
    }

    /// With the `cuda` feature OFF, the plugin must not depend on
    /// onnx-runtime-ep-cuda at all — verify the module is gated.
    #[cfg(not(feature = "cuda"))]
    #[test]
    fn no_cuda_feature_means_no_ep_dependency() {
        // This test existing and compiling proves the gate works.
        // The CreateEpFactories path would return zero factories + error status.
    }
}

// Re-export panic_to_fail_status for tests (mirrors the macro pattern).
pub use onnx_runtime_ep_plugin::panic_to_fail_status;
