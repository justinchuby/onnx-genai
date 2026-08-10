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
use crate::status::{fail_status, ok_status, set_host_api};

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
        if let Some(create_status) =
            (!fallback_api.is_null()).then(|| unsafe { (*fallback_api).CreateStatus }).flatten()
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
        return fail_status(
            "CreateEpFactories: out_factories is null or max_factories is 0",
        );
    }

    // Create a temporary EP to get the name.
    let ep = constructor();
    let name = ep.name().to_string();
    drop(ep);

    let name_cstr = CString::new(name.as_str()).unwrap_or_else(|_| CString::new("nxrt_ep").unwrap());
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
            ValidateCompiledModelCompatibilityInfo: None,
            CreateAllocator: None,
            ReleaseAllocator: None,
            CreateDataTransfer: None,
            IsStreamAware: None,
            CreateSyncStreamForDevice: None,
            GetHardwareDeviceIncompatibilityDetails: None,
            CreateExternalResourceImporterForDevice: None,
            GetNumCustomOpDomains: None,
            GetCustomOpDomains: None,
            InitGraphicsInterop: None,
            DeinitGraphicsInterop: None,
        },
        name_cstr,
        vendor_cstr,
        version_cstr,
        constructor: Box::new(move || constructor()),
    });

    let factory_ptr = Box::into_raw(factory);
    // SAFETY: factory_ptr points to an ExportedFactory whose first field is
    // OrtEpFactory, so the cast is valid.
    unsafe {
        *out_factories = factory_ptr.cast::<ort::OrtEpFactory>();
        *out_num = 1;
    }
    ok_status()
}

/// Implementation of `ReleaseEpFactory`.
///
/// # Safety
///
/// `factory` must be a pointer returned by `create_ep_factories`.
pub unsafe fn release_ep_factory(
    factory: *mut ort::OrtEpFactory,
) -> *mut ort::OrtStatus {
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

unsafe extern "C" fn factory_get_name(
    factory: *const ort::OrtEpFactory,
) -> *const c_char {
    let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
        if factory.is_null() {
            return c"unknown".as_ptr();
        }
        let exported = unsafe { &*(factory.cast::<ExportedFactory>()) };
        exported.name_cstr.as_ptr()
    }));
    result.unwrap_or(c"unknown".as_ptr())
}

unsafe extern "C" fn factory_get_vendor(
    factory: *const ort::OrtEpFactory,
) -> *const c_char {
    let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
        if factory.is_null() {
            return c"unknown".as_ptr();
        }
        let exported = unsafe { &*(factory.cast::<ExportedFactory>()) };
        exported.vendor_cstr.as_ptr()
    }));
    result.unwrap_or(c"unknown".as_ptr())
}

unsafe extern "C" fn factory_get_vendor_id(
    _factory: *const ort::OrtEpFactory,
) -> u32 {
    // No PCI vendor ID for CPU EP; return 0.
    0
}

unsafe extern "C" fn factory_get_version(
    factory: *const ort::OrtEpFactory,
) -> *const c_char {
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
    _in_devices: *const *const ort::OrtHardwareDevice,
    _num_in: usize,
    out_devices: *mut *mut ort::OrtEpDevice,
    max_out: usize,
    out_num: *mut usize,
) -> *mut ort::OrtStatus {
    let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
        if factory.is_null() || out_devices.is_null() || out_num.is_null() || max_out == 0 {
            return fail_status("GetSupportedDevices: invalid arguments");
        }

        // For CPU EP we return zero EP devices — the EP does not own hardware that
        // ORT needs to enumerate.
        unsafe { *out_num = 0 };
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

        let exported_ep = Box::new(ExportedEp::new(ep));
        let ep_ptr = Box::into_raw(exported_ep);
        // SAFETY: ExportedEp's first field is OrtEp vtable.
        unsafe { *out_ep = ep_ptr.cast::<ort::OrtEp>() };
        ok_status()
    }));
    result.unwrap_or_else(|_| fail_status("CreateEp: internal panic"))
}

unsafe extern "C" fn factory_release_ep(
    _factory: *mut ort::OrtEpFactory,
    ep: *mut ort::OrtEp,
) {
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
