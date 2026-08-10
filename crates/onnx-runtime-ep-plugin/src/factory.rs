//! `ExportedFactory` — the heap object behind an opaque `OrtEpFactory*`.
//!
//! Owns an EP constructor and fills the `OrtEpFactory` vtable so ORT can
//! discover devices, create EPs, and release them.

use std::ffi::{CString, c_char};
use std::panic::AssertUnwindSafe;
use std::ptr;

use onnx_genai_ort_sys as ort;
use onnx_runtime_ep_api::provider::ExecutionProvider;

use crate::ep::ExportedEp;
use crate::status::{fail_status, host_api, ok_status, set_host_api};

/// A heap-allocated factory whose raw pointer is returned as `OrtEpFactory*`.
///
/// The first field is the `OrtEpFactory` vtable struct so the pointer can be
/// cast directly to `*mut OrtEpFactory` (ORT dereferences the vtable fields
/// at fixed offsets from the factory pointer).
#[repr(C)]
pub struct ExportedFactory {
    /// The vtable ORT reads through the `OrtEpFactory*` pointer.
    pub vtable: ort::OrtEpFactory,
    /// The EP name as a C string, kept alive for `GetName`.
    pub name_cstr: CString,
    /// Vendor name as a C string, kept alive for `GetVendor`.
    pub vendor_cstr: CString,
    /// Version string as a C string, kept alive for `GetVersion`.
    pub version_cstr: CString,
    /// Constructor that produces a fresh EP instance.
    pub constructor: Box<dyn Fn() -> Box<dyn ExecutionProvider> + Send + Sync>,
    /// Optional kernel registry entries for type-constraint advertisement.
    /// When non-empty, `create_ep` builds an ORT kernel registry so that ORT
    /// routes f16/bf16 (and other typed) nodes to this EP.
    pub kernel_registry_entries: Vec<crate::ep::KernelRegistryEntry>,
}

/// Implementation of `CreateEpFactories` — called by the macro-generated export.
///
/// # Safety
///
/// All pointer arguments must be valid per the ORT plugin-EP C ABI.
pub unsafe fn create_ep_factories<F>(
    api_base: *const ort::OrtApiBase,
    out_factories: *mut *mut ort::OrtEpFactory,
    max_factories: usize,
    out_num: *mut usize,
    constructor: F,
) -> *mut ort::OrtStatus
where
    F: Fn() -> Box<dyn ExecutionProvider> + Send + Sync + 'static,
{
    // Version negotiation: obtain the OrtApi from the host.
    if api_base.is_null() {
        // Cannot call fail_status yet — no API. Return null (success) and
        // write zero factories.
        unsafe {
            if !out_num.is_null() {
                *out_num = 0;
            }
        }
        return ptr::null_mut();
    }

    let get_api = unsafe { (*api_base).GetApi };
    let api = match get_api {
        Some(get_api) => unsafe { get_api(ort::ORT_API_VERSION) },
        None => {
            unsafe {
                if !out_num.is_null() {
                    *out_num = 0;
                }
            }
            return ptr::null_mut();
        }
    };

    if api.is_null() {
        // Fail closed: host ORT does not support our API version. We cannot
        // proceed because the vtable may lack functions we depend on.
        // Try to create an error status using an older API version for error
        // reporting. If that also fails, return null + 0 factories.
        let fallback_api = unsafe { (get_api.unwrap())(1) }; // v1 always has CreateStatus
        if let Some(create_status) = (!fallback_api.is_null())
            .then(|| unsafe { (*fallback_api).CreateStatus })
            .flatten()
        {
            let msg = c"EP plugin requires ORT API version 27 but host does not support it. \
                       Plugin will not load (fail-closed).";
            unsafe {
                if !out_num.is_null() {
                    *out_num = 0;
                }
            }
            return unsafe { create_status(ort::ORT_FAIL, msg.as_ptr()) };
        }
        unsafe {
            if !out_num.is_null() {
                *out_num = 0;
            }
        }
        return ptr::null_mut();
    }

    // Store the host API for later status creation, graph reading, etc.
    unsafe { set_host_api(api) };

    if max_factories == 0 || out_factories.is_null() || out_num.is_null() {
        return fail_status("CreateEpFactories: out_factories is null or max_factories is 0");
    }

    // Create a temporary EP to get the name.
    let ep = constructor();
    let name = ep.name().to_string();
    drop(ep);

    let name_cstr =
        CString::new(name.as_str()).unwrap_or_else(|_| CString::new("nxrt_ep").unwrap());
    let vendor_cstr = CString::new("nxrt").unwrap();
    let version_cstr = CString::new("0.1.0").unwrap();

    let factory = Box::new(ExportedFactory {
        vtable: ort::OrtEpFactory {
            ort_version_supported: ort::ORT_API_VERSION,
            GetName: Some(factory_get_name),
            GetVendor: Some(factory_get_vendor),
            GetSupportedDevices: Some(factory_get_supported_devices),
            CreateEp: Some(factory_create_ep),
            ReleaseEp: Some(factory_release_ep),
            GetVendorId: Some(factory_get_vendor_id),
            GetVersion: Some(factory_get_version),
            ValidateCompiledModelCompatibilityInfo: Some(factory_validate_compiled_model),
            CreateAllocator: Some(factory_create_allocator),
            ReleaseAllocator: Some(factory_release_allocator),
            CreateDataTransfer: Some(factory_create_data_transfer),
            IsStreamAware: Some(factory_is_stream_aware),
            CreateSyncStreamForDevice: Some(factory_create_sync_stream),
            GetHardwareDeviceIncompatibilityDetails: Some(factory_get_hw_incompatibility),
            CreateExternalResourceImporterForDevice: Some(factory_create_resource_importer),
            GetNumCustomOpDomains: Some(factory_get_num_custom_op_domains),
            GetCustomOpDomains: Some(factory_get_custom_op_domains),
            InitGraphicsInterop: Some(factory_init_graphics_interop),
            DeinitGraphicsInterop: Some(factory_deinit_graphics_interop),
        },
        name_cstr,
        vendor_cstr,
        version_cstr,
        constructor: Box::new(move || constructor()),
        kernel_registry_entries: Vec::new(),
    });

    let factory_ptr = Box::into_raw(factory);
    // SAFETY: factory_ptr points to an ExportedFactory whose first field is
    // OrtEpFactory, so the cast is valid.
    unsafe {
        // Zero the entire output array so ORT doesn't read stale pointers.
        for i in 0..max_factories {
            *out_factories.add(i) = ptr::null_mut();
        }
        *out_factories = factory_ptr.cast::<ort::OrtEpFactory>();
        *out_num = 1;
    }
    ok_status()
}

/// Like [`create_ep_factories`] but also registers kernel-registry entries for
/// type-constraint metadata. This enables ORT to route typed nodes (e.g.
/// f16/bf16) to the EP.
///
/// # Safety
///
/// All pointer arguments must be valid per the ORT plugin-EP C ABI.
pub unsafe fn create_ep_factories_with_registry<F>(
    api_base: *const ort::OrtApiBase,
    out_factories: *mut *mut ort::OrtEpFactory,
    max_factories: usize,
    out_num: *mut usize,
    constructor: F,
    entries: Vec<crate::ep::KernelRegistryEntry>,
) -> *mut ort::OrtStatus
where
    F: Fn() -> Box<dyn ExecutionProvider> + Send + Sync + 'static,
{
    // Delegate to the standard path — it builds the factory.
    let status = unsafe {
        create_ep_factories(api_base, out_factories, max_factories, out_num, constructor)
    };
    if !status.is_null() {
        return status;
    }
    // Patch the factory's kernel_registry_entries field.
    if !out_factories.is_null() {
        let factory_ptr = unsafe { *out_factories };
        if !factory_ptr.is_null() {
            let exported = unsafe { &mut *(factory_ptr.cast::<ExportedFactory>()) };
            exported.kernel_registry_entries = entries;
        }
    }
    ok_status()
}

/// Implementation of `ReleaseEpFactory`.
///
/// # Safety
///
/// `factory` must be a pointer returned by `create_ep_factories`.
pub unsafe fn release_ep_factory(factory: *mut ort::OrtEpFactory) -> *mut ort::OrtStatus {
    if factory.is_null() {
        return ok_status();
    }
    // SAFETY: The pointer was created by Box::into_raw in create_ep_factories.
    unsafe {
        drop(Box::from_raw(factory.cast::<ExportedFactory>()));
    }
    ok_status()
}

// ─── OrtEpFactory callbacks ─────────────────────────────────────────────────

unsafe extern "C" fn factory_get_name(factory: *const ort::OrtEpFactory) -> *const c_char {
    let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
        if factory.is_null() {
            return c"unknown".as_ptr();
        }
        let exported = unsafe { &*(factory.cast::<ExportedFactory>()) };
        exported.name_cstr.as_ptr()
    }));
    result.unwrap_or(c"unknown".as_ptr())
}

unsafe extern "C" fn factory_get_vendor(factory: *const ort::OrtEpFactory) -> *const c_char {
    let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
        if factory.is_null() {
            return c"unknown".as_ptr();
        }
        let exported = unsafe { &*(factory.cast::<ExportedFactory>()) };
        exported.vendor_cstr.as_ptr()
    }));
    result.unwrap_or(c"unknown".as_ptr())
}

unsafe extern "C" fn factory_get_vendor_id(_factory: *const ort::OrtEpFactory) -> u32 {
    // No PCI vendor ID for CPU EP; return 0.
    0
}

unsafe extern "C" fn factory_get_version(factory: *const ort::OrtEpFactory) -> *const c_char {
    let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
        if factory.is_null() {
            return c"0.0.0".as_ptr();
        }
        let exported = unsafe { &*(factory.cast::<ExportedFactory>()) };
        exported.version_cstr.as_ptr()
    }));
    result.unwrap_or(c"0.0.0".as_ptr())
}

unsafe extern "C" fn factory_get_supported_devices(
    factory: *mut ort::OrtEpFactory,
    in_devices: *const *const ort::OrtHardwareDevice,
    num_in: usize,
    out_devices: *mut *mut ort::OrtEpDevice,
    max_out: usize,
    out_num: *mut usize,
) -> *mut ort::OrtStatus {
    let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
        // Null/bounds checks.
        if out_num.is_null() {
            return fail_status("GetSupportedDevices: out_num is null");
        }
        // Default to zero so early returns are safe.
        unsafe { *out_num = 0 };

        if factory.is_null() || out_devices.is_null() || max_out == 0 {
            return fail_status("GetSupportedDevices: invalid arguments");
        }
        if in_devices.is_null() || num_in == 0 {
            // No hardware devices from ORT — nothing to match. Clean return.
            return ok_status();
        }

        // Get the ORT API and EP API.
        let api = host_api();
        if api.is_null() {
            return fail_status("GetSupportedDevices: host ORT API not initialized");
        }
        let hw_type_fn = unsafe { (*api).HardwareDevice_Type };
        let hw_type_fn = match hw_type_fn {
            Some(f) => f,
            None => return fail_status("GetSupportedDevices: HardwareDevice_Type not available"),
        };
        let get_ep_api = match unsafe { (*api).GetEpApi } {
            Some(f) => f,
            None => return fail_status("GetSupportedDevices: GetEpApi not available"),
        };
        let ep_api = unsafe { get_ep_api() };
        if ep_api.is_null() {
            return fail_status("GetSupportedDevices: GetEpApi returned null");
        }
        let create_ep_device = match unsafe { (*ep_api).CreateEpDevice } {
            Some(f) => f,
            None => return fail_status("GetSupportedDevices: CreateEpDevice not available"),
        };

        // Iterate input hardware devices, filter for CPU type, create an
        // OrtEpDevice for each one (up to max_out).
        let mut count: usize = 0;
        for i in 0..num_in {
            if count >= max_out {
                break;
            }
            let hw_device = unsafe { *in_devices.add(i) };
            if hw_device.is_null() {
                continue;
            }
            let dev_type = unsafe { hw_type_fn(hw_device) };
            if dev_type != ort::OrtHardwareDeviceType_CPU {
                continue;
            }

            // Create an OrtEpDevice. ORT takes ownership of the returned pointer.
            // Create empty metadata and options KVPs — ORT may dereference these internally.
            let create_kvp = unsafe { (*api).CreateKeyValuePairs };
            let release_kvp = unsafe { (*api).ReleaseKeyValuePairs };
            let mut ep_metadata: *mut ort::OrtKeyValuePairs = ptr::null_mut();
            let mut ep_options: *mut ort::OrtKeyValuePairs = ptr::null_mut();
            if let Some(create_kvp) = create_kvp {
                unsafe { create_kvp(&mut ep_metadata) };
                unsafe { create_kvp(&mut ep_options) };
            }

            let mut ep_device: *mut ort::OrtEpDevice = ptr::null_mut();
            let status = unsafe {
                create_ep_device(factory, hw_device, ep_metadata, ep_options, &mut ep_device)
            };

            // Release KVPs (CreateEpDevice copies them per the doc).
            if let Some(release) = release_kvp {
                if !ep_metadata.is_null() {
                    unsafe { release(ep_metadata) };
                }
                if !ep_options.is_null() {
                    unsafe { release(ep_options) };
                }
            }

            if !status.is_null() {
                return status;
            }
            if ep_device.is_null() {
                return fail_status("GetSupportedDevices: CreateEpDevice returned null device");
            }

            // Register CPU memory info so ORT knows how to allocate for this device.
            // Required: ORT internally accesses the device's allocator info.
            let add_alloc_info = match unsafe { (*ep_api).EpDevice_AddAllocatorInfo } {
                Some(f) => f,
                None => {
                    return fail_status(
                        "GetSupportedDevices: EpDevice_AddAllocatorInfo not available",
                    );
                }
            };

            // Use CreateMemoryInfo_V2 to produce a properly-typed OrtMemoryInfo
            // with OrtMemoryInfoDeviceType and OrtDeviceMemoryType fields that the
            // EP device system requires. The legacy CreateCpuMemoryInfo creates
            // old-format memory info whose device-type/memory-type fields are
            // uninitialized in the new EP ABI, producing garbage values.
            let create_mem_info_v2 = match unsafe { (*api).CreateMemoryInfo_V2 } {
                Some(f) => f,
                None => {
                    return fail_status("GetSupportedDevices: CreateMemoryInfo_V2 not available");
                }
            };

            // CPU device allocator with OrtDeviceMemoryType_DEFAULT.
            let mut mem_info: *mut ort::OrtMemoryInfo = ptr::null_mut();
            let status = unsafe {
                create_mem_info_v2(
                    c"Cpu".as_ptr(),
                    ort::OrtMemoryInfoDeviceType_CPU, // device_type
                    0,                                // vendor_id (generic)
                    0,                                // device_id
                    ort::OrtDeviceMemoryType_DEFAULT, // mem_type
                    0,                                // alignment (default)
                    ort::OrtDeviceAllocator,          // allocator_type
                    &mut mem_info,
                )
            };
            if !status.is_null() {
                return status;
            }
            if mem_info.is_null() {
                return fail_status("GetSupportedDevices: CreateMemoryInfo_V2 returned null");
            }

            // EpDevice_AddAllocatorInfo stores the OrtMemoryInfo pointer inside
            // the OrtEpDevice. ORT accesses it later (e.g. CreateAllocator,
            // EpDevice_MemoryInfo). We must NOT release the OrtMemoryInfo here —
            // ORT will release it when the OrtEpDevice is released via
            // ReleaseEpDevice. Releasing here causes a use-after-free that
            // manifests as garbage DeviceType/MemoryType after repeated
            // register/unregister cycles.
            let status = unsafe { add_alloc_info(ep_device, mem_info) };
            if !status.is_null() {
                // Release mem_info only on failure since it was not consumed.
                if let Some(release) = unsafe { (*api).ReleaseMemoryInfo } {
                    unsafe { release(mem_info) };
                }
                return status;
            }

            unsafe { *out_devices.add(count) = ep_device };
            count += 1;
        }

        unsafe { *out_num = count };
        ok_status()
    }));
    result.unwrap_or_else(|_| fail_status("GetSupportedDevices: internal panic"))
}

unsafe extern "C" fn factory_create_ep(
    factory: *mut ort::OrtEpFactory,
    _hardware: *const *const ort::OrtHardwareDevice,
    _metadata: *const *const ort::OrtKeyValuePairs,
    _num_devices: usize,
    _session_options: *const ort::OrtSessionOptions,
    _logger: *const ort::OrtLogger,
    out_ep: *mut *mut ort::OrtEp,
) -> *mut ort::OrtStatus {
    let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
        if factory.is_null() || out_ep.is_null() {
            return fail_status("CreateEp: invalid arguments");
        }

        let exported = unsafe { &*(factory.cast::<ExportedFactory>()) };
        let mut ep = (exported.constructor)();

        // Initialize the EP with default config.
        let config = onnx_runtime_ep_api::provider::EpConfig::default();
        if let Err(e) = ep.initialize(&config) {
            return fail_status(&format!("CreateEp: EP initialization failed: {e}"));
        }

        let exported_ep = Box::new(ExportedEp::new_with_registry(
            ep,
            crate::ep::build_ort_kernel_registry(
                &exported.kernel_registry_entries,
                exported.name_cstr.to_str().unwrap_or("nxrt_ep"),
            ),
        ));
        let ep_ptr = Box::into_raw(exported_ep);
        // SAFETY: ExportedEp's first field is OrtEp vtable.
        unsafe { *out_ep = ep_ptr.cast::<ort::OrtEp>() };
        ok_status()
    }));
    result.unwrap_or_else(|_| fail_status("CreateEp: internal panic"))
}

unsafe extern "C" fn factory_release_ep(_factory: *mut ort::OrtEpFactory, ep: *mut ort::OrtEp) {
    let _ = std::panic::catch_unwind(AssertUnwindSafe(|| {
        if ep.is_null() {
            return;
        }
        // SAFETY: ep was created by factory_create_ep via Box::into_raw.
        unsafe {
            let mut exported_ep = Box::from_raw(ep.cast::<ExportedEp>());
            // Best-effort shutdown.
            let _ = exported_ep.ep.shutdown();
        }
    }));
}

/// CPU EP uses ORT's default CPU allocator.
///
/// ORT 1.27 has a bug: after a successful CreateAllocator (null status return),
/// it dereferences the output allocator without null-checking. We MUST provide
/// a valid allocator even though the spec says "Set to nullptr for default."
unsafe extern "C" fn factory_create_allocator(
    _factory: *mut ort::OrtEpFactory,
    _memory_info: *const ort::OrtMemoryInfo,
    _allocator_options: *const ort::OrtKeyValuePairs,
    allocator: *mut *mut ort::OrtAllocator,
) -> *mut ort::OrtStatus {
    // Minimal panic-safe path. Avoid complex closures to reduce chance of panic.
    if allocator.is_null() {
        return fail_status("CreateAllocator: allocator output pointer is null");
    }
    let api = host_api();
    if api.is_null() {
        return fail_status("CreateAllocator: host API not available");
    }
    let get_default = unsafe { (*api).GetAllocatorWithDefaultOptions };
    match get_default {
        Some(f) => unsafe { f(allocator) },
        None => fail_status("CreateAllocator: GetAllocatorWithDefaultOptions unavailable"),
    }
}

/// No-op: we never create allocators, so nothing to release.
unsafe extern "C" fn factory_release_allocator(
    _factory: *mut ort::OrtEpFactory,
    _allocator: *mut ort::OrtAllocator,
) {
}

/// CPU EP is not stream-aware.
unsafe extern "C" fn factory_is_stream_aware(_factory: *const ort::OrtEpFactory) -> bool {
    false
}

/// No data transfer needed for CPU EP — set output to null.
unsafe extern "C" fn factory_create_data_transfer(
    _factory: *mut ort::OrtEpFactory,
    data_transfer: *mut *mut ort::OrtDataTransferImpl,
) -> *mut ort::OrtStatus {
    if !data_transfer.is_null() {
        unsafe { *data_transfer = ptr::null_mut() };
    }
    ok_status()
}

/// CPU EP does not create sync streams.
unsafe extern "C" fn factory_create_sync_stream(
    _factory: *mut ort::OrtEpFactory,
    _memory_device: *const ort::OrtMemoryDevice,
    _stream_options: *const ort::OrtKeyValuePairs,
    stream: *mut *mut ort::OrtSyncStreamImpl,
) -> *mut ort::OrtStatus {
    if !stream.is_null() {
        unsafe { *stream = ptr::null_mut() };
    }
    ok_status()
}

/// Always compatible — no compiled model validation.
unsafe extern "C" fn factory_validate_compiled_model(
    _factory: *mut ort::OrtEpFactory,
    _devices: *const *const ort::OrtHardwareDevice,
    _num_devices: usize,
    _compatibility_info: *const c_char,
    model_compatibility: *mut ort::OrtCompiledModelCompatibility,
) -> *mut ort::OrtStatus {
    if !model_compatibility.is_null() {
        unsafe { *model_compatibility = 0 }; // Compatible
    }
    ok_status()
}

/// No hardware incompatibility details to report.
unsafe extern "C" fn factory_get_hw_incompatibility(
    _factory: *mut ort::OrtEpFactory,
    _hw: *const ort::OrtHardwareDevice,
    _details: *mut ort::OrtDeviceEpIncompatibilityDetails,
) -> *mut ort::OrtStatus {
    ok_status()
}

/// No external resource importer for CPU EP.
unsafe extern "C" fn factory_create_resource_importer(
    _factory: *mut ort::OrtEpFactory,
    _ep_device: *const ort::OrtEpDevice,
    out_importer: *mut *mut ort::OrtExternalResourceImporterImpl,
) -> *mut ort::OrtStatus {
    if !out_importer.is_null() {
        unsafe { *out_importer = ptr::null_mut() };
    }
    ok_status()
}

/// No custom op domains.
unsafe extern "C" fn factory_get_num_custom_op_domains(
    _factory: *mut ort::OrtEpFactory,
    num_domains: *mut usize,
) -> *mut ort::OrtStatus {
    if !num_domains.is_null() {
        unsafe { *num_domains = 0 };
    }
    ok_status()
}

/// No custom op domains — nothing to fill.
unsafe extern "C" fn factory_get_custom_op_domains(
    _factory: *mut ort::OrtEpFactory,
    _domains: *mut *mut ort::OrtCustomOpDomain,
    _num_domains: usize,
) -> *mut ort::OrtStatus {
    ok_status()
}

/// Graphics interop not supported for CPU EP.
unsafe extern "C" fn factory_init_graphics_interop(
    _factory: *mut ort::OrtEpFactory,
    _ep_device: *const ort::OrtEpDevice,
    _config: *const ort::OrtGraphicsInteropConfig,
) -> *mut ort::OrtStatus {
    ok_status()
}

/// Graphics interop not supported for CPU EP.
unsafe extern "C" fn factory_deinit_graphics_interop(
    _factory: *mut ort::OrtEpFactory,
    _ep_device: *const ort::OrtEpDevice,
) -> *mut ort::OrtStatus {
    ok_status()
}
