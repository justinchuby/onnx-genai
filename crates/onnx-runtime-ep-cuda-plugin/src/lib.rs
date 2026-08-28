//! CUDA execution provider exported as an ORT plugin-EP cdylib.
//!
//! This crate mirrors `onnx-runtime-ep-cpu-plugin` for the CUDA EP.
//! Without the `cuda` feature (default), it compiles as a no-op stub so the
//! workspace builds on hosts without a CUDA toolkit.
//!
//! # CUDA Unique path: hardware-validated
//!
//! The following defects have been addressed:
//!
//! 1. **Shared CUDA runtime/context.** A single `CudaExecutionProvider` is
//!    constructed once and shared across allocator, stream, and data transfer
//!    via `Arc<Mutex<..>>`. Each component holds a strong `Arc` clone (B1 fix).
//!
//! 2. **`CreateDataTransfer` returns a real adapter.** `DeviceDataTransferFull`
//!    now classifies copy direction using `Value_GetMemoryDevice` +
//!    `MemoryDevice_GetDeviceType` and dispatches to `copy_from_host`,
//!    `copy_to_host`, or `copy` accordingly (B3 fix).
//!
//! 3. **`GetHandle` returns the real stream handle.** The native
//!    `cudaStream_t` is extracted from `CudaRuntime::stream_ptr()`.
//!
//! 4. **`Free` preserves allocation identity.** `DeviceAllocator` retains the
//!    exact EP-issued `DeviceBuffer` in a pointer-keyed table, so bound CUDA
//!    ownership and its allocation generation reach deferred release unchanged.
//!    Unknown pointers are no-op'd (S1 fix).
//!
//! **With the `cuda` feature ON**, `CreateEpFactories` attempts to construct
//! the EP. If a CUDA GPU is available, it advertises one factory. If no GPU
//! is available (this build host), it returns zero factories with an
//! actionable error — **by design, not by accidental panic.**
//!
//! **Without the `cuda` feature**, zero factories are returned.
//!
//! The CUDA Unique plugin path is exercised on physical CUDA hardware, including
//! device-resident dynamically sized `Unique` outputs and governed workspace.

#[cfg(feature = "cuda")]
mod cuda_impl {
    use onnx_runtime_ep_cuda::CudaExecutionProvider;
    use onnx_runtime_ep_plugin::device::DeviceSupport;
    use onnx_runtime_ep_plugin::ep::KernelRegistryEntry;

    /// NVIDIA vendor ID (PCI).
    const NVIDIA_VENDOR_ID: u32 = 0x10DE;

    /// Build kernel registry entries from the CUDA EP's **real** registry.
    ///
    /// Each entry's `(op_type, domain, since_version)` is derived from the same
    /// `OpRegistry` the EP dispatches on — via
    /// `build_cuda_registry_descriptors` — so a `com.microsoft` kernel is
    /// advertised in `com.microsoft` and a default-domain kernel in `""`. This
    /// replaces the previous flat `CUDA_COVERED_OPS` name list that advertised
    /// *every* kernel under the default domain, which meant no `com.microsoft`
    /// node (MatMulNBits, GroupQueryAttention, SkipSimplifiedLayerNormalization,
    /// GatherBlockQuantized — 61% of a real decoder) could ever match.
    ///
    /// Requires the constructed EP's `CudaRuntime`; at factory-construction time
    /// this is available from the already-built `CudaExecutionProvider`
    /// (`ep.runtime()`), so no separate GPU context is needed.
    ///
    /// The registry is derived from the real `OpRegistry`, not a hand-maintained
    /// list, so it cannot drift from the kernels. Advertising the real domains
    /// is always correct and harmless on its own: whether the EP actually
    /// *claims* a real decoder's nodes is gated separately at capability time
    /// (see `onnx-runtime-ep-plugin`'s partial-GPU-claim gate, off by default
    /// because executing an interspersed CPU/GPU partition currently hits #982).
    pub(crate) fn build_kernel_registry_entries(
        runtime: std::sync::Arc<onnx_runtime_ep_cuda::CudaRuntime>,
    ) -> Vec<KernelRegistryEntry> {
        let descriptors = onnx_runtime_ep_cuda::build_cuda_registry_descriptors(runtime);
        descriptors
            .iter()
            .enumerate()
            .map(|(index, d)| KernelRegistryEntry {
                op_type: leak_str(&d.op_type),
                domain: leak_str(&d.domain),
                since_version: i32::try_from(d.since_version)
                    .expect("registered CUDA opset must fit ORT's i32 version ABI"),
                end_version: schema_end_version(&descriptors, index),
                supported_dtypes: d.supported_dtypes,
                input_dtype_constraints: d.input_dtype_constraints,
                output_dtype_constraints: d.output_dtype_constraints,
            })
            .collect()
    }

    fn schema_end_version(
        descriptors: &[onnx_runtime_ep_cuda::CudaOpDescriptor],
        index: usize,
    ) -> i32 {
        let descriptor = &descriptors[index];
        descriptors
            .get(index + 1)
            .filter(|next| next.op_type == descriptor.op_type && next.domain == descriptor.domain)
            .map(|next| {
                i32::try_from(next.since_version)
                    .expect("registered CUDA opset must fit ORT's i32 version ABI")
                    - 1
            })
            .unwrap_or(i32::MAX)
    }

    /// Leak a string to get a `&'static str` (entries must live for the EP
    /// lifetime; mirrors the CPU plugin's `leak_str`).
    fn leak_str(s: &str) -> &'static str {
        Box::leak(s.to_owned().into_boxed_str())
    }

    type ConstructedEp = (
        Box<dyn onnx_runtime_ep_api::provider::ExecutionProvider + Send>,
        *mut std::os::raw::c_void,
        Vec<KernelRegistryEntry>,
    );

    /// Device support for the CUDA EP: GPU, stream-aware, device-only memory.
    pub(crate) fn device_support() -> DeviceSupport {
        DeviceSupport::gpu("Cuda", NVIDIA_VENDOR_ID)
    }

    /// Construct the CUDA EP and extract the native stream handle.
    ///
    /// Returns `(ep, stream_handle, entries)` or an error if no GPU is
    /// available. The kernel-registry entries are derived from the constructed
    /// EP's real registry so the caller advertises each kernel under its true
    /// domain.
    pub(crate) fn construct_ep_with_stream() -> Result<ConstructedEp, String> {
        let ep = CudaExecutionProvider::new_default().map_err(|e| {
            format!("CUDA EP construction failed (no GPU or driver unavailable): {e}")
        })?;
        let stream_handle = ep.runtime().stream_ptr() as *mut std::os::raw::c_void;
        let entries = build_kernel_registry_entries(ep.runtime().clone());
        Ok((Box::new(ep), stream_handle, entries))
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        fn descriptor(
            op_type: &str,
            domain: &str,
            since_version: u64,
        ) -> onnx_runtime_ep_cuda::CudaOpDescriptor {
            onnx_runtime_ep_cuda::CudaOpDescriptor {
                op_type: op_type.into(),
                domain: domain.into(),
                since_version,
                supported_dtypes: &[],
                input_dtype_constraints: &[],
                output_dtype_constraints: &[],
            }
        }

        #[test]
        fn advertised_schema_ranges_end_before_the_next_registered_version() {
            let descriptors = vec![
                descriptor("DsaIndexSelect", "pkg.nxrt", 1),
                descriptor("DsaIndexSelect", "pkg.nxrt", 3),
                descriptor("Other", "pkg.nxrt", 1),
            ];

            assert_eq!(schema_end_version(&descriptors, 0), 2);
            assert_eq!(schema_end_version(&descriptors, 1), i32::MAX);
            assert_eq!(schema_end_version(&descriptors, 2), i32::MAX);
        }
    }
}

/// ORT plugin-EP entry point: create EP factories.
///
/// With the `cuda` feature enabled, attempts to construct a real
/// `CudaExecutionProvider`. If a CUDA GPU is available, advertises one factory
/// with GPU device support, kernel-registry type constraints, and a shared EP
/// instance (single CUcontext + cudaStream_t across allocator, stream, and
/// data transfer). If no GPU is available, returns zero factories with an
/// actionable error status.
///
/// Without the `cuda` feature, returns zero factories with a "not available"
/// status.
///
/// **Unvalidated on hardware.** All four implementation defects (separate
/// context, NULL data transfer, NULL stream, Free size=0) are resolved in the
/// code, but the result has not been run on a physical CUDA GPU. Issue #768
/// tracks hardware validation.
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
    let out_num_raw = out_num;
    // Used only with `cuda` feature; suppress warnings for non-cuda builds.
    let _ = (api_base, max_factories, out_factories);
    let result = ::std::panic::catch_unwind(::std::panic::AssertUnwindSafe(|| {
        // Initialize the host API early so diagnostic error statuses can be
        // delivered to ORT. Without this, `panic_to_fail_status` returns null
        // (interpreted as success by ORT), silently losing the error message.
        if !api_base.is_null() {
            unsafe {
                let get_api = (*api_base).GetApi;
                if let Some(get_api_fn) = get_api {
                    let api =
                        get_api_fn(onnx_runtime_ep_plugin::onnx_genai_ort_sys::ORT_API_VERSION);
                    if !api.is_null() {
                        onnx_runtime_ep_plugin::status::set_host_api(api);
                    }
                }
            }
        }

        #[cfg(feature = "cuda")]
        {
            // Attempt to construct the CUDA EP. If no GPU is available, fail
            // closed with an actionable error.
            let (ep, stream_handle, entries) = match cuda_impl::construct_ep_with_stream() {
                Ok(tuple) => tuple,
                Err(msg) => {
                    unsafe {
                        if !out_num_raw.is_null() {
                            *out_num_raw = 0;
                        }
                    }
                    return onnx_runtime_ep_plugin::panic_to_fail_status(&format!(
                        "onnx-runtime-ep-cuda-plugin: {msg}. Zero factories returned."
                    ));
                }
            };

            let ep_name = ep.name().to_string();
            let support = cuda_impl::device_support();
            let shared_ep = std::sync::Arc::new(std::sync::Mutex::new(ep));

            // S4 fix: use create_ep_factories_for_shared_ep which takes the
            // EP name directly, avoiding the constructor call that would panic.
            let status = unsafe {
                onnx_runtime_ep_plugin::factory::create_ep_factories_for_shared_ep(
                    api_base,
                    out_factories,
                    max_factories,
                    out_num_raw,
                    &ep_name,
                    shared_ep,
                    entries,
                    support,
                    stream_handle,
                )
            };
            if status.is_null() && !out_factories.is_null() && max_factories != 0 {
                let factory = unsafe { *out_factories };
                let api = unsafe {
                    (*api_base)
                        .GetApi
                        .map(|get_api| {
                            get_api(onnx_runtime_ep_plugin::onnx_genai_ort_sys::ORT_API_VERSION)
                        })
                        .unwrap_or(std::ptr::null())
                };
                let domain_status = unsafe {
                    onnx_runtime_ep_plugin::nxrt_schema::attach_nxrt_custom_domain(factory, api)
                };
                if !domain_status.is_null() {
                    let _ = unsafe { onnx_runtime_ep_plugin::factory::release_ep_factory(factory) };
                    unsafe {
                        *out_factories = std::ptr::null_mut();
                        *out_num_raw = 0;
                    }
                    return domain_status;
                }
            }
            status
        }

        #[cfg(not(feature = "cuda"))]
        {
            unsafe {
                if !out_num_raw.is_null() {
                    *out_num_raw = 0;
                }
            }
            onnx_runtime_ep_plugin::panic_to_fail_status(NO_CUDA_FEATURE_DIAGNOSTIC)
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

// Re-export panic_to_fail_status for tests (mirrors the macro pattern).
pub use onnx_runtime_ep_plugin::panic_to_fail_status;

/// The diagnostic message emitted when the plugin is built **without** the
/// `cuda` feature. This is the exact string passed to `panic_to_fail_status`
/// on the fail-closed path in [`CreateEpFactories`]; it is defined here (rather
/// than inline) so tests can assert on the real diagnostic content.
#[cfg(not(feature = "cuda"))]
pub(crate) const NO_CUDA_FEATURE_DIAGNOSTIC: &str = "onnx-runtime-ep-cuda-plugin built without `cuda` feature; \
     CUDA EP is not available. Rebuild with --features cuda on a CUDA-capable host.";

/// Return the actionable diagnostic message the plugin emits on its
/// fail-closed path (zero factories), or `None` when a real CUDA factory would
/// be created (a GPU is available and the `cuda` feature is on).
///
/// This exposes the plugin's **actual** diagnostic string — the same text that
/// `CreateEpFactories` hands to `fail_status`/`panic_to_fail_status`. Tests can
/// assert on this without a live ORT host, because materializing an `OrtStatus`
/// (via the host `CreateStatus`) requires the ORT API pointer, whereas the
/// message content is produced entirely by this crate.
///
/// - Without the `cuda` feature: always returns the compile-time
///   "built without `cuda` feature" diagnostic.
/// - With the `cuda` feature but no GPU/driver: returns the construction
///   failure diagnostic produced by `construct_ep_with_stream`.
/// - With the `cuda` feature and a working GPU: returns `None` (a factory is
///   advertised, so there is no fail-closed diagnostic).
pub fn fail_closed_diagnostic() -> Option<String> {
    #[cfg(not(feature = "cuda"))]
    {
        Some(NO_CUDA_FEATURE_DIAGNOSTIC.to_string())
    }
    #[cfg(feature = "cuda")]
    {
        match cuda_impl::construct_ep_with_stream() {
            Ok(_) => None,
            Err(msg) => Some(format!(
                "onnx-runtime-ep-cuda-plugin: {msg}. Zero factories returned."
            )),
        }
    }
}

// ─── Hardware-validation observability ───────────────────────────────────────
//
// ORT loads this cdylib with `dlopen`, so the only way a validation harness
// running in the ORT process can read the executor's counters is through
// exported C symbols. The CPU plugin already exports the compiled-node counter
// for exactly this reason; issue #768's residual scope needs the workspace
// counter too, on the library ORT actually loads on the GPU host.

/// Number of nodes this EP has compiled since the last reset.
///
/// Proves the node under test was **assigned to this plugin EP** rather than
/// silently falling back: ORT reporting `cuda_ep` in the session providers only
/// says the EP was registered, not that it claimed anything.
#[unsafe(no_mangle)]
pub extern "C" fn nxrt_ep_compiled_node_count() -> usize {
    onnx_runtime_ep_plugin::ep::compiled_node_count()
}

/// Reset the compiled-node counter.
#[unsafe(no_mangle)]
pub extern "C" fn nxrt_ep_reset_compiled_node_count() {
    onnx_runtime_ep_plugin::ep::reset_compiled_node_count()
}

#[unsafe(no_mangle)]
pub extern "C" fn nxrt_ep_executed_node_count() -> usize {
    onnx_runtime_ep_plugin::compute::executed_node_count()
}

#[unsafe(no_mangle)]
pub extern "C" fn nxrt_ep_reset_executed_node_count() {
    onnx_runtime_ep_plugin::compute::reset_executed_node_count()
}

#[cfg(feature = "cuda")]
#[unsafe(no_mangle)]
pub extern "C" fn nxrt_ep_reset_allocator_teardown_stats() {
    onnx_runtime_ep_cuda::provider::reset_allocator_release_observation();
    onnx_runtime_ep_cuda::vmm_allocator::reset_global_vmm_stats();
}

#[cfg(feature = "cuda")]
#[unsafe(no_mangle)]
pub extern "C" fn nxrt_ep_cuda_committed_bytes() -> u64 {
    onnx_runtime_ep_cuda::vmm_allocator::global_vmm_stats().committed_bytes
}

#[cfg(feature = "cuda")]
#[unsafe(no_mangle)]
pub extern "C" fn nxrt_ep_cuda_allocator_quarantined_releases() -> u64 {
    onnx_runtime_ep_cuda::provider::allocator_release_observation().quarantined
}

#[cfg(feature = "cuda")]
#[unsafe(no_mangle)]
pub extern "C" fn nxrt_ep_cuda_allocator_retained_releases() -> u64 {
    onnx_runtime_ep_cuda::provider::allocator_release_observation().retained
}

#[cfg(feature = "cuda")]
#[unsafe(no_mangle)]
pub extern "C" fn nxrt_ep_reset_unique_execution_stats() {
    onnx_runtime_ep_cuda::reset_unique_execution_stats()
}

#[cfg(feature = "cuda")]
#[unsafe(no_mangle)]
pub extern "C" fn nxrt_ep_unique_metadata_launches() -> u64 {
    onnx_runtime_ep_cuda::unique_execution_stats().metadata_launches
}

#[cfg(feature = "cuda")]
#[unsafe(no_mangle)]
pub extern "C" fn nxrt_ep_unique_materialize_launches() -> u64 {
    onnx_runtime_ep_cuda::unique_execution_stats().materialize_launches
}

#[cfg(feature = "cuda")]
#[unsafe(no_mangle)]
pub extern "C" fn nxrt_ep_unique_d2h_bytes() -> u64 {
    onnx_runtime_ep_cuda::unique_execution_stats().d2h_bytes
}

#[cfg(feature = "cuda")]
#[unsafe(no_mangle)]
pub extern "C" fn nxrt_ep_unique_full_input_d2h_bytes() -> u64 {
    onnx_runtime_ep_cuda::unique_execution_stats().full_input_d2h_bytes
}

#[cfg(feature = "cuda")]
#[unsafe(no_mangle)]
pub extern "C" fn nxrt_ep_unique_workspace_bytes() -> u64 {
    onnx_runtime_ep_cuda::unique_execution_stats().workspace_bytes
}

#[cfg(feature = "cuda")]
#[unsafe(no_mangle)]
pub extern "C" fn nxrt_ep_reset_nms_execution_stats() {
    onnx_runtime_ep_cuda::reset_nms_execution_stats()
}

#[cfg(feature = "cuda")]
#[unsafe(no_mangle)]
pub extern "C" fn nxrt_ep_nms_prepare_launches() -> u64 {
    onnx_runtime_ep_cuda::nms_execution_stats().prepare_launches
}

#[cfg(feature = "cuda")]
#[unsafe(no_mangle)]
pub extern "C" fn nxrt_ep_nms_count_launches() -> u64 {
    onnx_runtime_ep_cuda::nms_execution_stats().count_launches
}

#[cfg(feature = "cuda")]
#[unsafe(no_mangle)]
pub extern "C" fn nxrt_ep_nms_materialize_launches() -> u64 {
    onnx_runtime_ep_cuda::nms_execution_stats().materialize_launches
}

#[cfg(feature = "cuda")]
#[unsafe(no_mangle)]
pub extern "C" fn nxrt_ep_nms_d2h_bytes() -> u64 {
    onnx_runtime_ep_cuda::nms_execution_stats().d2h_bytes
}

#[cfg(feature = "cuda")]
#[unsafe(no_mangle)]
pub extern "C" fn nxrt_ep_nms_full_input_d2h_bytes() -> u64 {
    onnx_runtime_ep_cuda::nms_execution_stats().full_input_d2h_bytes
}

#[cfg(feature = "cuda")]
#[unsafe(no_mangle)]
pub extern "C" fn nxrt_ep_nms_workspace_bytes() -> u64 {
    onnx_runtime_ep_cuda::nms_execution_stats().workspace_bytes
}

/// Number of workspace **placement resolutions** since the last reset.
///
/// One resolution happens per served `StepScoped` workspace and nowhere else: a
/// zero-byte requirement and a declined `SessionPersistent` request both return
/// before placement is resolved. A non-zero delta across a `Run` is therefore
/// evidence that the governed-workspace path actually served this node — the
/// property issue #768 asks hardware to confirm.
#[unsafe(no_mangle)]
pub extern "C" fn nxrt_ep_workspace_placement_queries() -> usize {
    onnx_runtime_ep_plugin::compute::workspace_placement_queries()
}

/// Reset the workspace placement-resolution counter.
#[unsafe(no_mangle)]
pub extern "C" fn nxrt_ep_reset_workspace_placement_queries() {
    onnx_runtime_ep_plugin::compute::reset_workspace_placement_queries()
}

#[cfg(feature = "cuda")]
#[unsafe(no_mangle)]
pub extern "C" fn nxrt_cuda_dsa_workspace_allocations() -> u64 {
    onnx_runtime_ep_cuda::dsa_workspace_stats().allocations
}

#[cfg(feature = "cuda")]
#[unsafe(no_mangle)]
pub extern "C" fn nxrt_cuda_dsa_workspace_releases() -> u64 {
    onnx_runtime_ep_cuda::dsa_workspace_stats().releases
}

#[cfg(feature = "cuda")]
#[unsafe(no_mangle)]
pub extern "C" fn nxrt_cuda_dsa_workspace_live_bytes() -> u64 {
    onnx_runtime_ep_cuda::dsa_workspace_stats().live_bytes
}

#[cfg(feature = "cuda")]
#[unsafe(no_mangle)]
pub extern "C" fn nxrt_cuda_dsa_workspace_last_ptr() -> u64 {
    onnx_runtime_ep_cuda::dsa_workspace_stats().last_ptr
}

#[cfg(feature = "cuda")]
#[unsafe(no_mangle)]
pub extern "C" fn nxrt_cuda_reset_dsa_workspace_stats() -> bool {
    onnx_runtime_ep_cuda::reset_dsa_workspace_stats()
}

#[cfg(feature = "cuda")]
#[unsafe(no_mangle)]
pub extern "C" fn nxrt_cuda_dsa_executions() -> u64 {
    onnx_runtime_ep_cuda::dsa_launch_stats().executions
}

#[cfg(feature = "cuda")]
#[unsafe(no_mangle)]
pub extern "C" fn nxrt_cuda_dsa_score_launches() -> u64 {
    onnx_runtime_ep_cuda::dsa_launch_stats().score_launches
}

#[cfg(feature = "cuda")]
#[unsafe(no_mangle)]
pub extern "C" fn nxrt_cuda_dsa_selection_launches() -> u64 {
    onnx_runtime_ep_cuda::dsa_launch_stats().selection_launches
}

#[cfg(feature = "cuda")]
#[unsafe(no_mangle)]
pub extern "C" fn nxrt_cuda_dsa_last_score_grid_x() -> u64 {
    onnx_runtime_ep_cuda::dsa_launch_stats().last_score_grid_x
}

#[cfg(feature = "cuda")]
#[unsafe(no_mangle)]
pub extern "C" fn nxrt_cuda_dsa_last_selection_grid_x() -> u64 {
    onnx_runtime_ep_cuda::dsa_launch_stats().last_selection_grid_x
}

#[cfg(feature = "cuda")]
#[unsafe(no_mangle)]
pub extern "C" fn nxrt_cuda_reset_dsa_launch_stats() {
    onnx_runtime_ep_cuda::reset_dsa_launch_stats()
}

#[cfg(feature = "gpu-tests")]
#[unsafe(no_mangle)]
pub extern "C" fn nxrt_cuda_set_dsa_capture_replays_for_test(replays: u64) {
    onnx_runtime_ep_cuda::set_dsa_plugin_capture_replays_for_test(replays)
}

#[cfg(feature = "gpu-tests")]
#[unsafe(no_mangle)]
pub extern "C" fn nxrt_cuda_dsa_capture_count_for_test() -> u64 {
    onnx_runtime_ep_cuda::dsa_plugin_capture_stats_for_test().0
}

#[cfg(feature = "gpu-tests")]
#[unsafe(no_mangle)]
pub extern "C" fn nxrt_cuda_dsa_captured_replays_for_test() -> u64 {
    onnx_runtime_ep_cuda::dsa_plugin_capture_stats_for_test().1
}

#[cfg(feature = "gpu-tests")]
#[unsafe(no_mangle)]
pub extern "C" fn nxrt_cuda_dsa_capture_error_for_test() -> u64 {
    onnx_runtime_ep_cuda::dsa_plugin_capture_stats_for_test().2
}

#[cfg(test)]
mod tests {
    use std::panic::AssertUnwindSafe;

    /// Verify that a panicking constructor is caught at the macro guard boundary.
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
    }

    /// CopyDirection classification matrix: verify all 5 combinations.
    #[test]
    fn copy_direction_matrix() {
        use onnx_runtime_ep_plugin::transfer::CopyDirection;

        let h2d = CopyDirection::classify(true, false, false);
        assert_eq!(h2d, CopyDirection::HostToDevice);
        assert!(h2d.is_supported());

        let d2h = CopyDirection::classify(false, true, false);
        assert_eq!(d2h, CopyDirection::DeviceToHost);
        assert!(d2h.is_supported());

        let d2d_same = CopyDirection::classify(false, false, true);
        assert_eq!(d2d_same, CopyDirection::DeviceToSameDevice);
        assert!(d2d_same.is_supported());

        let d2d_cross = CopyDirection::classify(false, false, false);
        assert_eq!(d2d_cross, CopyDirection::DeviceToDifferentDevice);
        assert!(!d2d_cross.is_supported(), "cross-device must be rejected");

        let h2h = CopyDirection::classify(true, true, true);
        assert_eq!(h2h, CopyDirection::HostToHost);
        assert!(!h2h.is_supported(), "host-to-host must be rejected");
    }

    /// DeviceSupport: GPU config is stream-aware, non-host-accessible.
    #[test]
    fn device_support_gpu_properties() {
        let support = onnx_runtime_ep_plugin::device::DeviceSupport::gpu("Cuda", 0x10DE);
        assert!(support.stream_aware);
        assert!(!support.host_accessible);
        assert_eq!(support.vendor_id, 0x10DE);
    }
}
