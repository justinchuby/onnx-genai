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
/// If the host API is not available (before `CreateEpFactories` completes
/// init), returns a null pointer — which ORT interprets as success. Callers in
/// the pre-init window (e.g. null `api_base`) must handle errors without
/// `fail_status`. After init, the host API is always set.
pub fn fail_status(message: &str) -> *mut ort::OrtStatus {
    status_with_code(ort::ORT_FAIL, message)
}

/// Create an `OrtStatus` with `ORT_INVALID_ARGUMENT`. Use for null or
/// out-of-range pointer arguments coming from ORT.
pub(crate) fn invalid_arg_status(message: &str) -> *mut ort::OrtStatus {
    status_with_code(ort::ORT_INVALID_ARGUMENT, message)
}

/// Create an `OrtStatus` with an explicit error code.
pub(crate) fn status_with_code(code: ort::OrtErrorCode, message: &str) -> *mut ort::OrtStatus {
    let c_msg = CString::new(message).unwrap_or_else(|_| CString::new("unknown error").unwrap());
    let api = host_api();
    if api.is_null() {
        return ptr::null_mut();
    }
    // SAFETY: api was set during CreateEpFactories and is process-lifetime valid.
    unsafe {
        match (*api).CreateStatus {
            Some(create_status) => create_status(code, c_msg.as_ptr()),
            None => ptr::null_mut(),
        }
    }
}

/// Success: null status pointer.
pub fn ok_status() -> *mut ort::OrtStatus {
    ptr::null_mut()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ptr;

    #[test]
    fn host_api_starts_null() {
        // In a test binary with no ORT loaded, host_api() returns null.
        // We can't guarantee this if another test sets it, but the AtomicPtr
        // default is null.
        let _ = host_api(); // must not panic or crash
    }

    #[test]
    fn fail_status_returns_null_when_no_api() {
        // When the API is not initialized (test environment without ORT),
        // fail_status returns null_mut (fail-safe: ORT would interpret as success,
        // but we document this path is unreachable in production).
        // We zero the atomic so this test is deterministic.
        HOST_ORT_API.store(ptr::null_mut(), Ordering::Release);
        let status = fail_status("test error");
        assert!(status.is_null(), "expected null when no ORT API is set");
    }

    #[test]
    fn invalid_arg_status_returns_null_when_no_api() {
        HOST_ORT_API.store(ptr::null_mut(), Ordering::Release);
        let status = invalid_arg_status("null pointer");
        assert!(status.is_null());
    }

    #[test]
    fn ok_status_is_null() {
        assert!(ok_status().is_null());
    }

    #[test]
    fn catch_unwind_prevents_panic_from_escaping() {
        // Verify that std::panic::catch_unwind absorbs a panic — this is the
        // same pattern every extern "C" callback uses.
        HOST_ORT_API.store(ptr::null_mut(), Ordering::Release);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            panic!("test panic inside extern C");
        }));
        // Panic was caught; the test thread is still alive.
        assert!(result.is_err(), "catch_unwind should return Err on panic");
    }

    #[test]
    fn set_host_api_stores_and_loads() {
        // Verify the AtomicPtr round-trips a non-null value.
        let dummy: u8 = 42;
        let ptr = &dummy as *const u8 as *const ort::OrtApi;
        unsafe { set_host_api(ptr) };
        assert_eq!(host_api(), ptr);
        // Restore to null so other tests are not affected.
        HOST_ORT_API.store(std::ptr::null_mut(), Ordering::Release);
    }
}
