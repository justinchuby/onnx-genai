#[test]
fn pinned_runtime_loads_and_its_c_api_status_roundtrips() {
    use onnx_genai_ort_sys as ort;

    // SAFETY: the loader retains the library for process lifetime. Every
    // pointer is checked before use, and the created status is released once.
    unsafe {
        let base = ort::OrtGetApiBase();
        assert!(
            !base.is_null(),
            "{}",
            ort::ort_load_error().unwrap_or_default()
        );
        assert_eq!(ort::loaded_ort_version().as_deref(), Some(ort::ORT_VERSION));
        assert_eq!(ort::loaded_ort_api_version(), Some(ort::ORT_API_VERSION));
        let api = (*base).GetApi.expect("ORT GetApi")(ort::ORT_API_VERSION);
        assert!(!api.is_null());
        let status = (*api).CreateStatus.expect("ORT CreateStatus")(
            ort::ORT_INVALID_ARGUMENT,
            c"nxrt API status roundtrip".as_ptr(),
        );
        assert!(!status.is_null());
        let code = (*api).GetErrorCode.expect("ORT GetErrorCode")(status);
        let message =
            std::ffi::CStr::from_ptr((*api).GetErrorMessage.expect("ORT GetErrorMessage")(status))
                .to_owned();
        (*api).ReleaseStatus.expect("ORT ReleaseStatus")(status);
        assert_eq!(code, ort::ORT_INVALID_ARGUMENT);
        assert_eq!(message.as_bytes(), b"nxrt API status roundtrip");
    }
}
