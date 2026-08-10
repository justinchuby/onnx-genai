//! `EpError` → `*mut OrtStatus` projection and helpers.

use std::ffi::CString;
use std::ptr;
use std::sync::atomic::{AtomicPtr, Ordering};

use onnx_genai_ort_sys as ort;

/// The `OrtApi` pointer obtained from the host's `OrtApiBase::GetApi`.
///
/// Set once during `CreateEpFactories` and valid for the process lifetime.
/// Uses `AtomicPtr` to avoid data races if ORT calls us from multiple threads.
static HOST_ORT_API: AtomicPtr<ort::OrtApi> = AtomicPtr::new(ptr::null_mut());

/// Store the host-provided `OrtApi` for later use.
///
/// # Safety
///
/// The pointer must remain valid for the process lifetime (ORT guarantees this
/// for `GetApi` results).
pub unsafe fn set_host_api(api: *const ort::OrtApi) {
    HOST_ORT_API.store(api as *mut ort::OrtApi, Ordering::Release);
}

/// Get the host-provided `OrtApi`.
///
/// Returns null if [`set_host_api`] has not been called yet.
pub(crate) fn host_api() -> *const ort::OrtApi {
    HOST_ORT_API.load(Ordering::Acquire)
}

/// Create an `OrtStatus` with `ORT_FAIL` using the host's `CreateStatus`.
///
/// If the host API is not available (should never happen after init), returns
/// a null pointer which ORT interprets as success — but we document this path
/// cannot be reached.
pub(crate) fn fail_status(message: &str) -> *mut ort::OrtStatus {
    let c_msg = CString::new(message).unwrap_or_else(|_| CString::new("unknown error").unwrap());
    let api = host_api();
    if api.is_null() {
        return ptr::null_mut();
    }
    // SAFETY: api was set during CreateEpFactories and is process-lifetime valid.
    unsafe {
        match (*api).CreateStatus {
            Some(create_status) => create_status(ort::ORT_FAIL, c_msg.as_ptr()),
            None => ptr::null_mut(),
        }
    }
}


/// Success: null status pointer.
pub(crate) fn ok_status() -> *mut ort::OrtStatus {
    ptr::null_mut()
}
