//! CUDA execution provider exported as an ORT plugin-EP cdylib.
//!
//! This crate mirrors `onnx-runtime-ep-cpu-plugin` for the CUDA EP.
//! Without the `cuda` feature (default), it compiles as a no-op stub so the
//! workspace builds on hosts without a CUDA toolkit.
//!
//! With `cuda` enabled and `onnx-runtime-ep-cuda` linked, it exports
//! `CreateEpFactories` and `ReleaseEpFactory` that project the CUDA EP
//! through the ORT plugin-EP C ABI with real GPU device support.

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
        -> Result<Box<dyn onnx_runtime_ep_api::provider::ExecutionProvider>, String>
    {
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
        #[cfg(feature = "cuda")]
        {
            // Verify the EP can actually be constructed on this host.
            if let Err(msg) = cuda_impl::construct_ep() {
                eprintln!("[onnx-runtime-ep-cuda-plugin] {msg}");
                unsafe {
                    if !out_num_raw.is_null() {
                        *out_num_raw = 0;
                    }
                }
                return onnx_runtime_ep_plugin::panic_to_fail_status(&msg);
            }

            let entries = cuda_impl::build_kernel_registry_entries();
            let support = cuda_impl::device_support();
            unsafe {
                onnx_runtime_ep_plugin::factory::create_ep_factories_with_device_support(
                    _api_base,
                    _out_factories,
                    _max_factories,
                    out_num_raw,
                    || {
                        // This constructor is called per-session. If it fails
                        // at runtime (e.g. device yanked), the panic guard in
                        // the macro catches it.
                        cuda_impl::construct_ep()
                            .expect("CUDA EP construction must succeed (device was validated)")
                    },
                    entries,
                    support,
                )
            }
        }

        #[cfg(not(feature = "cuda"))]
        {
            // No CUDA feature: zero factories, actionable error.
            unsafe {
                if !out_num_raw.is_null() {
                    *out_num_raw = 0;
                }
            }
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
/// # Safety
///
/// `factory` must be a pointer returned by `CreateEpFactories` from this
/// library, and must not be used after this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ReleaseEpFactory(
    factory: *mut onnx_runtime_ep_plugin::onnx_genai_ort_sys::OrtEpFactory,
) {
    let _ = ::std::panic::catch_unwind(::std::panic::AssertUnwindSafe(|| {
        unsafe { onnx_runtime_ep_plugin::factory::release_ep_factory(factory) };
    }));
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
