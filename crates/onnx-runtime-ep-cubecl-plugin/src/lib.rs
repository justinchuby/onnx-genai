//! CubeCL execution providers exported as both ORT plugin-EP and nxrt ABIs.
//!
//! The cdylib advertises one factory for each CubeCL backend that is compiled
//! in, supported on this platform, and able to open its device. If no backend is
//! usable it fails closed with an actionable diagnostic rather than advertising
//! a factory that would fail on first use.

use std::ptr;

use onnx_runtime_ep_cubecl::backend::CubeclBackend;
use onnx_runtime_ep_plugin::device::DeviceSupport;
use onnx_runtime_ep_plugin::ep::KernelRegistryEntry;

const GENERIC_GPU_VENDOR_ID: u32 = 0;

type DynEp = Box<dyn onnx_runtime_ep_api::ExecutionProvider + Send>;

struct PreparedFactory {
    backend: CubeclBackend,
    ep: DynEp,
    entries: Vec<KernelRegistryEntry>,
    support: DeviceSupport,
}

/// Backends compiled into this plugin and supported by the target platform.
pub fn compile_available_provider_names() -> Vec<&'static str> {
    CubeclBackend::ALL
        .into_iter()
        .filter(|backend| backend.unavailable_message().is_none())
        .map(CubeclBackend::provider_name)
        .collect()
}

/// The diagnostic used when the plugin returns zero factories.
pub fn zero_factory_diagnostic(reasons: &[String]) -> String {
    let mut message = "onnx-runtime-ep-cubecl-plugin: zero CubeCL factories returned; no execution provider was advertised. ".to_string();
    if reasons.is_empty() {
        message.push_str(
            "No CubeCL backend could be opened. Check the requested ONNX_GENAI_EP value, \
             install a compatible GPU driver, or use ONNX_GENAI_EP=cpu.",
        );
    } else {
        message.push_str(&reasons.join(" "));
    }
    if !message.contains("ONNX_GENAI_EP=cpu") {
        message.push_str(" To run without CubeCL, use ONNX_GENAI_EP=cpu.");
    }
    message
}

/// Return the actual fail-closed diagnostic, or `None` when at least one backend
/// can be opened and a factory would be advertised.
pub fn fail_closed_diagnostic() -> Option<String> {
    let (factories, diagnostics) = prepare_factories();
    if factories.is_empty() {
        Some(zero_factory_diagnostic(&diagnostics))
    } else {
        None
    }
}

fn kernel_registry_entries() -> Vec<KernelRegistryEntry> {
    #[cfg(feature = "webgpu")]
    {
        onnx_runtime_ep_cubecl::provider::build_cubecl_registry_descriptors()
            .iter()
            .map(|descriptor| KernelRegistryEntry {
                op_type: descriptor.op_type,
                domain: descriptor.domain,
                since_version: descriptor.since_version,
                end_version: i32::MAX,
                supported_dtypes: descriptor.supported_dtypes,
                input_dtype_constraints: &[],
            })
            .collect()
    }
    #[cfg(not(feature = "webgpu"))]
    {
        Vec::new()
    }
}

fn device_support(backend: CubeclBackend) -> DeviceSupport {
    DeviceSupport::gpu(backend.provider_name(), GENERIC_GPU_VENDOR_ID)
}

fn construct_backend_ep(backend: CubeclBackend) -> Result<DynEp, String> {
    if let Some(message) = backend.unavailable_message() {
        return Err(message);
    }

    match backend {
        CubeclBackend::WebGpu => construct_webgpu_ep(backend),
        CubeclBackend::Vulkan => construct_vulkan_ep(backend),
    }
}

#[cfg(feature = "webgpu")]
fn construct_webgpu_ep(backend: CubeclBackend) -> Result<DynEp, String> {
    use onnx_runtime_ep_cubecl::provider::CubeclExecutionProvider;
    use onnx_runtime_ep_cubecl::runtime::WebGpuRuntime;

    CubeclExecutionProvider::<WebGpuRuntime>::new(backend, 0)
        .map(|ep| Box::new(ep) as DynEp)
        .map_err(|error| format!("execution provider '{}' device open failed: {error}. Check that a compatible WebGPU/wgpu adapter is visible to this process, or use ONNX_GENAI_EP=cpu.", backend.provider_name()))
}

#[cfg(not(feature = "webgpu"))]
fn construct_webgpu_ep(backend: CubeclBackend) -> Result<DynEp, String> {
    Err(backend.unavailable_message().unwrap_or_else(|| {
        format!(
            "execution provider '{}' is unavailable because the plugin was built without \
             --features webgpu. Rebuild with --features webgpu, or use ONNX_GENAI_EP=cpu.",
            backend.provider_name()
        )
    }))
}

#[cfg(all(feature = "vulkan", not(target_os = "macos")))]
fn construct_vulkan_ep(backend: CubeclBackend) -> Result<DynEp, String> {
    use onnx_runtime_ep_cubecl::provider::CubeclExecutionProvider;
    use onnx_runtime_ep_cubecl::runtime::VulkanRuntime;

    CubeclExecutionProvider::<VulkanRuntime>::new(backend, 0)
        .map(|ep| Box::new(ep) as DynEp)
        .map_err(|error| format!("execution provider '{}' device open failed: {error}. Check that a compatible Vulkan adapter and driver are visible to this process, or use ONNX_GENAI_EP=cpu.", backend.provider_name()))
}

#[cfg(any(not(feature = "vulkan"), target_os = "macos"))]
fn construct_vulkan_ep(backend: CubeclBackend) -> Result<DynEp, String> {
    Err(backend.unavailable_message().unwrap_or_else(|| {
        format!(
            "execution provider '{}' is unavailable in this build. Rebuild with \
             --features vulkan on a non-macOS host, or use ONNX_GENAI_EP=cpu.",
            backend.provider_name()
        )
    }))
}

fn prepare_factories() -> (Vec<PreparedFactory>, Vec<String>) {
    let mut factories = Vec::new();
    let mut diagnostics = Vec::new();

    for backend in CubeclBackend::ALL {
        if let Some(message) = backend.unavailable_message() {
            diagnostics.push(message);
            continue;
        }
        match construct_backend_ep(backend) {
            Ok(ep) => factories.push(PreparedFactory {
                backend,
                ep,
                entries: kernel_registry_entries(),
                support: device_support(backend),
            }),
            Err(message) => diagnostics.push(message),
        }
    }

    (factories, diagnostics)
}

unsafe fn initialise_ort_api_for_status(
    api_base: *const onnx_runtime_ep_plugin::onnx_genai_ort_sys::OrtApiBase,
) {
    if api_base.is_null() {
        return;
    }
    let get_api = unsafe { (*api_base).GetApi };
    if let Some(get_api_fn) = get_api {
        let api =
            unsafe { get_api_fn(onnx_runtime_ep_plugin::onnx_genai_ort_sys::ORT_API_VERSION) };
        if !api.is_null() {
            unsafe { onnx_runtime_ep_plugin::status::set_host_api(api) };
        }
    }
}

/// ORT plugin-EP entry point: create one factory per usable CubeCL backend.
///
/// # Safety
///
/// Called by ORT's plugin loader. Pointer arguments must satisfy the ORT
/// plugin-EP C ABI.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn CreateEpFactories(
    _registration_name: *const std::ffi::c_char,
    api_base: *const onnx_runtime_ep_plugin::onnx_genai_ort_sys::OrtApiBase,
    _logger: *const onnx_runtime_ep_plugin::onnx_genai_ort_sys::OrtLogger,
    out_factories: *mut *mut onnx_runtime_ep_plugin::onnx_genai_ort_sys::OrtEpFactory,
    max_factories: usize,
    out_num: *mut usize,
) -> *mut onnx_runtime_ep_plugin::onnx_genai_ort_sys::OrtStatus {
    let out_factories_raw = out_factories;
    let out_num_raw = out_num;
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        unsafe { initialise_ort_api_for_status(api_base) };

        if max_factories == 0 || out_factories_raw.is_null() || out_num_raw.is_null() {
            if !out_num_raw.is_null() {
                unsafe { *out_num_raw = 0 };
            }
            return onnx_runtime_ep_plugin::panic_to_fail_status(
                "CreateEpFactories: out_factories is null, out_num is null, or max_factories is 0; \
                 CubeCL plugin cannot return factories. Pass a non-null output buffer with capacity \
                 for at least one factory, or use ONNX_GENAI_EP=cpu.",
            );
        }

        let (factories, diagnostics) = prepare_factories();
        if factories.is_empty() {
            unsafe { *out_num_raw = 0 };
            return onnx_runtime_ep_plugin::panic_to_fail_status(&zero_factory_diagnostic(
                &diagnostics,
            ));
        }
        if max_factories < factories.len() {
            unsafe { *out_num_raw = 0 };
            return onnx_runtime_ep_plugin::panic_to_fail_status(&format!(
                "CreateEpFactories: output capacity max_factories={max_factories} is too small \
                 for {} CubeCL backend(s). Pass capacity for every available backend, or select \
                 a single provider explicitly with ONNX_GENAI_EP.",
                factories.len()
            ));
        }

        let mut written = 0usize;
        for prepared in factories {
            let registration_name = prepared.backend.registration_name();
            let shared_ep = std::sync::Arc::new(std::sync::Mutex::new(prepared.ep));
            let mut local_num = 0usize;
            let status = unsafe {
                onnx_runtime_ep_plugin::factory::create_ep_factories_for_shared_ep(
                    api_base,
                    out_factories_raw.add(written),
                    max_factories - written,
                    &mut local_num,
                    registration_name,
                    shared_ep,
                    prepared.entries,
                    prepared.support,
                    ptr::null_mut(),
                )
            };
            if !status.is_null() {
                if !out_num_raw.is_null() {
                    unsafe { *out_num_raw = written };
                }
                return status;
            }
            written += local_num;
        }

        if !out_num_raw.is_null() {
            unsafe { *out_num_raw = written };
        }
        ptr::null_mut()
    }));

    match result {
        Ok(status) => status,
        Err(_) => {
            if !out_num_raw.is_null() {
                unsafe { *out_num_raw = 0 };
            }
            onnx_runtime_ep_plugin::panic_to_fail_status(
                "CreateEpFactories: CubeCL factory creation panicked; zero factories returned \
                 (fail-closed). Use ONNX_GENAI_EP=cpu or inspect nxrt_ep diagnostics.",
            )
        }
    }
}

/// ORT plugin-EP entry point: release an EP factory.
///
/// # Safety
///
/// `factory` must be a pointer returned by this library's `CreateEpFactories`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ReleaseEpFactory(
    factory: *mut onnx_runtime_ep_plugin::onnx_genai_ort_sys::OrtEpFactory,
) -> *mut onnx_runtime_ep_plugin::onnx_genai_ort_sys::OrtStatus {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
        onnx_runtime_ep_plugin::factory::release_ep_factory(factory)
    }));
    match result {
        Ok(status) => status,
        Err(_) => onnx_runtime_ep_plugin::panic_to_fail_status(
            "ReleaseEpFactory: panic during CubeCL factory release (fail-closed)",
        ),
    }
}

/// nxrt ABI version negotiation entry point.
///
/// # Safety
///
/// Pointers must be valid and non-null per the nxrt ABI.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn NxrtNegotiate(
    request: *const onnx_runtime_ep_nxrt_abi::NxrtNegotiateRequest,
    response_out: *mut onnx_runtime_ep_nxrt_abi::NxrtNegotiateResponse,
) -> onnx_runtime_ep_nxrt_abi::NxrtStatus {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
        onnx_runtime_ep_nxrt_abi::version::negotiate(request, response_out)
    }));
    result.unwrap_or_else(|_| {
        onnx_runtime_ep_nxrt_abi::NxrtStatus::from_code_with_message(
            onnx_runtime_ep_nxrt_abi::NxrtStatusCode::InternalError,
            "NxrtNegotiate: CubeCL plugin negotiation panicked (fail-closed)",
        )
    })
}

/// nxrt ABI entry point: create one factory per usable CubeCL backend.
///
/// # Safety
///
/// All pointers must be valid per the nxrt ABI.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn NxrtCreateEpFactories(
    out_factories: *mut *mut onnx_runtime_ep_nxrt_abi::NxrtEpFactoryVtable,
    max_factories: usize,
    out_num: *mut usize,
) -> onnx_runtime_ep_nxrt_abi::NxrtStatus {
    let out_factories_raw = out_factories;
    let out_num_raw = out_num;
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        if max_factories == 0 || out_factories_raw.is_null() || out_num_raw.is_null() {
            return onnx_runtime_ep_nxrt_abi::NxrtStatus::from_code_with_message(
                onnx_runtime_ep_nxrt_abi::NxrtStatusCode::InvalidArgument,
                "NxrtCreateEpFactories: out_factories is null, out_num is null, or max_factories is 0",
            );
        }

        let (factories, diagnostics) = prepare_factories();
        if factories.is_empty() {
            unsafe { *out_num_raw = 0 };
            return onnx_runtime_ep_nxrt_abi::NxrtStatus::from_code_with_message(
                onnx_runtime_ep_nxrt_abi::NxrtStatusCode::DeviceError,
                &zero_factory_diagnostic(&diagnostics),
            );
        }
        if max_factories < factories.len() {
            unsafe { *out_num_raw = 0 };
            return onnx_runtime_ep_nxrt_abi::NxrtStatus::from_code_with_message(
                onnx_runtime_ep_nxrt_abi::NxrtStatusCode::InvalidArgument,
                &format!(
                    "NxrtCreateEpFactories: output capacity max_factories={max_factories} is too \
                     small for {} CubeCL backend(s). Pass capacity for every available backend, \
                     or select a single provider explicitly with ONNX_GENAI_EP.",
                    factories.len()
                ),
            );
        }

        let mut written = 0usize;
        for prepared in factories {
            let backend = prepared.backend;
            drop(prepared);
            let mut local_num = 0usize;
            let status = unsafe {
                onnx_runtime_ep_nxrt_abi::vtable::create_ep_factories(
                    out_factories_raw.add(written),
                    max_factories - written,
                    &mut local_num,
                    move || match construct_backend_ep(backend) {
                        Ok(ep) => ep as Box<dyn onnx_runtime_ep_api::ExecutionProvider>,
                        Err(message) => panic!("{message}"),
                    },
                )
            };
            if !status.is_ok() {
                unsafe { *out_num_raw = written };
                return status;
            }
            written += local_num;
        }

        unsafe { *out_num_raw = written };
        onnx_runtime_ep_nxrt_abi::NxrtStatus::ok()
    }));

    result.unwrap_or_else(|_| {
        if !out_num_raw.is_null() {
            unsafe { *out_num_raw = 0 };
        }
        onnx_runtime_ep_nxrt_abi::NxrtStatus::from_code_with_message(
            onnx_runtime_ep_nxrt_abi::NxrtStatusCode::InternalError,
            "NxrtCreateEpFactories: CubeCL factory creation panicked; zero factories returned (fail-closed). Use ONNX_GENAI_EP=cpu.",
        )
    })
}

/// Number of nodes this EP has compiled since the last reset.
#[unsafe(no_mangle)]
pub extern "C" fn nxrt_ep_compiled_node_count() -> usize {
    onnx_runtime_ep_plugin::ep::compiled_node_count()
}

/// Reset the compiled-node counter.
#[unsafe(no_mangle)]
pub extern "C" fn nxrt_ep_reset_compiled_node_count() {
    onnx_runtime_ep_plugin::ep::reset_compiled_node_count()
}

/// Number of workspace placement resolutions since the last reset.
#[unsafe(no_mangle)]
pub extern "C" fn nxrt_ep_workspace_placement_queries() -> usize {
    onnx_runtime_ep_plugin::compute::workspace_placement_queries()
}

/// Reset the workspace placement-resolution counter.
#[unsafe(no_mangle)]
pub extern "C" fn nxrt_ep_reset_workspace_placement_queries() {
    onnx_runtime_ep_plugin::compute::reset_workspace_placement_queries()
}
