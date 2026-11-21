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

#[test]
fn string_attribute_read_preserves_embedded_and_trailing_nuls() {
    use onnx_genai_ort_sys as ort;
    // SAFETY: process-lifetime ORT API; the attribute is owned and released.
    unsafe {
        let base = ort::OrtGetApiBase();
        assert!(!base.is_null());
        let api = &*((*base).GetApi.unwrap()(ort::ORT_API_VERSION));
        for data in [b"nearest".as_slice(), b"a\0b\0".as_slice()] {
            let mut attribute = std::ptr::null_mut();
            assert!(
                api.CreateOpAttr.unwrap()(
                    c"mode".as_ptr(),
                    data.as_ptr().cast(),
                    data.len() as i32,
                    ort::ORT_OP_ATTR_STRING,
                    &mut attribute
                )
                .is_null()
            );
            let mut required = 0;
            let status = api.ReadOpAttr.unwrap()(
                attribute,
                ort::ORT_OP_ATTR_STRING,
                std::ptr::null_mut(),
                0,
                &mut required,
            );
            if !status.is_null() {
                api.ReleaseStatus.unwrap()(status);
            }
            let mut bytes = vec![0; required];
            let mut written = 0;
            let status = api.ReadOpAttr.unwrap()(
                attribute,
                ort::ORT_OP_ATTR_STRING,
                bytes.as_mut_ptr().cast(),
                bytes.len(),
                &mut written,
            );
            api.ReleaseOpAttr.unwrap()(attribute);
            assert!(status.is_null());
            bytes.truncate(written);
            assert_eq!(bytes, data);
        }
    }
}
