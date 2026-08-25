#![cfg(feature = "cuda")]

use std::ffi::{CStr, CString, c_void};
use std::path::PathBuf;
use std::ptr;
use std::sync::{Mutex, MutexGuard};

use onnx_genai_ort_sys as ort;

static ORT_LOCK: Mutex<()> = Mutex::new(());

fn lock_ort() -> MutexGuard<'static, ()> {
    ORT_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

unsafe fn ort_api(library: &libloading::Library) -> *const ort::OrtApi {
    type GetApiBase = unsafe extern "C" fn() -> *const ort::OrtApiBase;
    let get_api_base: libloading::Symbol<'_, GetApiBase> =
        unsafe { library.get(b"OrtGetApiBase") }.expect("OrtGetApiBase");
    let base = unsafe { get_api_base() };
    let get_api = unsafe { (*base).GetApi }.expect("GetApi");
    let api = unsafe { get_api(ort::ORT_API_VERSION) };
    assert!(!api.is_null());
    api
}

unsafe fn check(api: *const ort::OrtApi, status: *mut ort::OrtStatus, stage: &str) {
    if status.is_null() {
        return;
    }
    let message = unsafe {
        CStr::from_ptr(((*api).GetErrorMessage.unwrap())(status))
            .to_string_lossy()
            .into_owned()
    };
    unsafe { ((*api).ReleaseStatus.unwrap())(status) };
    panic!("{stage}: {message}");
}

struct Session {
    _ort: libloading::Library,
    plugin: libloading::Library,
    api: *const ort::OrtApi,
    env: *mut ort::OrtEnv,
    options: *mut ort::OrtSessionOptions,
    session: *mut ort::OrtSession,
    registration: CString,
}

impl Drop for Session {
    fn drop(&mut self) {
        unsafe {
            ((*self.api).ReleaseSession.unwrap())(self.session);
            ((*self.api).ReleaseSessionOptions.unwrap())(self.options);
            let status = ((*self.api).UnregisterExecutionProviderLibrary.unwrap())(
                self.env,
                self.registration.as_ptr(),
            );
            check(self.api, status, "UnregisterExecutionProviderLibrary");
            ((*self.api).ReleaseEnv.unwrap())(self.env);
        }
    }
}

unsafe fn session(model: &str, registration: &str) -> Option<Session> {
    let ort_dir = onnx_runtime_ort_testkit::find_ort_lib_dir()?;
    let plugin_path = onnx_runtime_ort_testkit::find_plugin_cdylib_with_features(
        "onnx-runtime-ep-cuda-plugin",
        &["cuda"],
    )?;
    let ort_library =
        unsafe { libloading::Library::new(ort_dir.join(onnx_runtime_ort_testkit::ort_lib_name())) }
            .ok()?;
    let plugin = unsafe { libloading::Library::new(&plugin_path) }.ok()?;
    let reset_compiled: libloading::Symbol<'_, unsafe extern "C" fn()> =
        unsafe { plugin.get(b"nxrt_ep_reset_compiled_node_count") }.ok()?;
    unsafe { reset_compiled() };
    let api = unsafe { ort_api(&ort_library) };

    let mut env = ptr::null_mut();
    let log_id = CString::new(format!("cuda_unique_{registration}")).unwrap();
    unsafe {
        check(
            api,
            ((*api).CreateEnv.unwrap())(ort::ORT_LOGGING_LEVEL_WARNING, log_id.as_ptr(), &mut env),
            "CreateEnv",
        )
    };
    let registration = CString::new(registration).unwrap();
    let plugin_path = onnx_runtime_ort_testkit::OrtPathBuf::new(&plugin_path);
    unsafe {
        check(
            api,
            ((*api).RegisterExecutionProviderLibrary.unwrap())(
                env,
                registration.as_ptr(),
                plugin_path.as_ptr(),
            ),
            "RegisterExecutionProviderLibrary",
        )
    };

    let mut devices: *const *const ort::OrtEpDevice = ptr::null();
    let mut device_count = 0usize;
    unsafe {
        check(
            api,
            ((*api).GetEpDevices.unwrap())(env, &mut devices, &mut device_count),
            "GetEpDevices",
        )
    };
    let ep_name = unsafe { (*api).EpDevice_EpName.unwrap() };
    let mut cuda_device = ptr::null();
    for index in 0..device_count {
        let device = unsafe { *devices.add(index) };
        let name = unsafe { CStr::from_ptr(ep_name(device)) }.to_string_lossy();
        if name == "cuda_ep" {
            cuda_device = device;
            break;
        }
    }
    assert!(!cuda_device.is_null(), "cuda_ep device was not registered");

    let mut options = ptr::null_mut();
    unsafe {
        check(
            api,
            ((*api).CreateSessionOptions.unwrap())(&mut options),
            "CreateSessionOptions",
        )
    };
    let add_config = unsafe { (*api).AddSessionConfigEntry.unwrap() };
    for (key, value) in [
        ("session.disable_cpu_ep_fallback", "1"),
        ("session.record_ep_graph_assignment_info", "1"),
    ] {
        let key = CString::new(key).unwrap();
        let value = CString::new(value).unwrap();
        unsafe {
            check(
                api,
                add_config(options, key.as_ptr(), value.as_ptr()),
                "AddSessionConfigEntry",
            )
        };
    }
    unsafe {
        check(
            api,
            ((*api).SessionOptionsAppendExecutionProvider_V2.unwrap())(
                options,
                env,
                [cuda_device].as_ptr(),
                1,
                ptr::null(),
                ptr::null(),
                0,
            ),
            "SessionOptionsAppendExecutionProvider_V2",
        )
    };

    let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join(model)
        .join("model.onnx");
    let bytes = std::fs::read(&fixture).unwrap();
    let mut session = ptr::null_mut();
    unsafe {
        check(
            api,
            ((*api).CreateSessionFromArray.unwrap())(
                env,
                bytes.as_ptr().cast(),
                bytes.len(),
                options,
                &mut session,
            ),
            "CreateSessionFromArray",
        )
    };
    Some(Session {
        _ort: ort_library,
        plugin,
        api,
        env,
        options,
        session,
        registration,
    })
}

unsafe fn input(api: *const ort::OrtApi, values: &mut [f32]) -> *mut ort::OrtValue {
    let mut memory_info = ptr::null_mut();
    unsafe {
        check(
            api,
            ((*api).CreateCpuMemoryInfo.unwrap())(
                ort::OrtDeviceAllocator,
                ort::OrtMemTypeDefault,
                &mut memory_info,
            ),
            "CreateCpuMemoryInfo",
        )
    };
    let shape = [values.len() as i64];
    let mut value = ptr::null_mut();
    unsafe {
        check(
            api,
            ((*api).CreateTensorWithDataAsOrtValue.unwrap())(
                memory_info,
                values.as_mut_ptr().cast(),
                std::mem::size_of_val(values),
                shape.as_ptr(),
                1,
                ort::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                &mut value,
            ),
            "CreateTensorWithDataAsOrtValue",
        );
        ((*api).ReleaseMemoryInfo.unwrap())(memory_info);
    }
    value
}

unsafe fn output_bytes(
    api: *const ort::OrtApi,
    output: *mut ort::OrtValue,
) -> (Vec<i64>, ort::ONNXTensorElementDataType, Vec<u8>) {
    let mut info = ptr::null_mut();
    unsafe {
        check(
            api,
            ((*api).GetTensorTypeAndShape.unwrap())(output, &mut info),
            "GetTensorTypeAndShape",
        )
    };
    let mut rank = 0usize;
    unsafe {
        check(
            api,
            ((*api).GetDimensionsCount.unwrap())(info, &mut rank),
            "GetDimensionsCount",
        )
    };
    let mut shape = vec![0i64; rank];
    unsafe {
        check(
            api,
            ((*api).GetDimensions.unwrap())(info, shape.as_mut_ptr(), rank),
            "GetDimensions",
        )
    };
    let mut dtype = 0;
    unsafe {
        check(
            api,
            ((*api).GetTensorElementType.unwrap())(info, &mut dtype),
            "GetTensorElementType",
        );
        ((*api).ReleaseTensorTypeAndShapeInfo.unwrap())(info);
    }
    let elements = shape.iter().try_fold(1usize, |product, &extent| {
        product.checked_mul(usize::try_from(extent).ok()?)
    });
    let elements = elements.unwrap();
    let element_size = if dtype == ort::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT {
        4
    } else {
        8
    };
    let mut data: *mut c_void = ptr::null_mut();
    unsafe {
        check(
            api,
            ((*api).GetTensorMutableData.unwrap())(output, &mut data),
            "GetTensorMutableData",
        )
    };
    let bytes = if elements == 0 {
        Vec::new()
    } else {
        unsafe { std::slice::from_raw_parts(data.cast::<u8>(), elements * element_size) }.to_vec()
    };
    (shape, dtype, bytes)
}

fn f32s(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_ne_bytes(chunk.try_into().unwrap()))
        .collect()
}

fn i64s(bytes: &[u8]) -> Vec<i64> {
    bytes
        .chunks_exact(8)
        .map(|chunk| i64::from_ne_bytes(chunk.try_into().unwrap()))
        .collect()
}

#[test]
fn cuda_unique_is_claimed_and_materialized_through_real_ort_plugin() {
    let _lock = lock_ort();
    let Some(session) = (unsafe { session("unique_all_outputs", "cuda_unique_all") }) else {
        if std::env::var("NXRT_REQUIRE_ORT_TESTS").as_deref() == Ok("1") {
            panic!("CUDA plugin or ORT unavailable");
        }
        return;
    };
    unsafe {
        let compiled: libloading::Symbol<'_, unsafe extern "C" fn() -> usize> =
            session.plugin.get(b"nxrt_ep_compiled_node_count").unwrap();
        let reset_executed: libloading::Symbol<'_, unsafe extern "C" fn()> = session
            .plugin
            .get(b"nxrt_ep_reset_executed_node_count")
            .unwrap();
        let executed: libloading::Symbol<'_, unsafe extern "C" fn() -> usize> =
            session.plugin.get(b"nxrt_ep_executed_node_count").unwrap();
        let reset_stats: libloading::Symbol<'_, unsafe extern "C" fn()> = session
            .plugin
            .get(b"nxrt_ep_reset_unique_execution_stats")
            .unwrap();
        let metadata_launches: libloading::Symbol<'_, unsafe extern "C" fn() -> u64> = session
            .plugin
            .get(b"nxrt_ep_unique_metadata_launches")
            .unwrap();
        let materialize_launches: libloading::Symbol<'_, unsafe extern "C" fn() -> u64> = session
            .plugin
            .get(b"nxrt_ep_unique_materialize_launches")
            .unwrap();
        let d2h_bytes: libloading::Symbol<'_, unsafe extern "C" fn() -> u64> =
            session.plugin.get(b"nxrt_ep_unique_d2h_bytes").unwrap();
        let full_d2h: libloading::Symbol<'_, unsafe extern "C" fn() -> u64> = session
            .plugin
            .get(b"nxrt_ep_unique_full_input_d2h_bytes")
            .unwrap();
        let reset_workspace: libloading::Symbol<'_, unsafe extern "C" fn()> = session
            .plugin
            .get(b"nxrt_ep_reset_workspace_placement_queries")
            .unwrap();
        let workspace_queries: libloading::Symbol<'_, unsafe extern "C" fn() -> usize> = session
            .plugin
            .get(b"nxrt_ep_workspace_placement_queries")
            .unwrap();
        let workspace_bytes: libloading::Symbol<'_, unsafe extern "C" fn() -> u64> = session
            .plugin
            .get(b"nxrt_ep_unique_workspace_bytes")
            .unwrap();
        reset_executed();
        reset_stats();
        reset_workspace();

        let mut values = [2., 1., 1., 3., 4., 3.];
        let input = input(session.api, &mut values);
        let input_name = CString::new("X").unwrap();
        let output_names: Vec<CString> = ["Y", "indices", "inverse_indices", "counts"]
            .into_iter()
            .map(|name| CString::new(name).unwrap())
            .collect();
        let output_name_ptrs: Vec<_> = output_names.iter().map(|name| name.as_ptr()).collect();
        let mut outputs = [ptr::null_mut(); 4];
        check(
            session.api,
            ((*session.api).Run.unwrap())(
                session.session,
                ptr::null(),
                [input_name.as_ptr()].as_ptr(),
                [input as *const ort::OrtValue].as_ptr(),
                1,
                output_name_ptrs.as_ptr(),
                4,
                outputs.as_mut_ptr(),
            ),
            "Run",
        );
        assert_eq!(compiled(), 1);
        assert_eq!(executed(), 1);
        let got: Vec<_> = outputs
            .iter()
            .map(|output| output_bytes(session.api, *output))
            .collect();
        assert_eq!(got[0].0, [4]);
        assert_eq!(f32s(&got[0].2), [2., 1., 3., 4.]);
        assert_eq!(i64s(&got[1].2), [0, 1, 3, 4]);
        assert_eq!(i64s(&got[2].2), [0, 1, 1, 2, 3, 2]);
        assert_eq!(i64s(&got[3].2), [1, 2, 2, 1]);
        assert_eq!(metadata_launches(), 1);
        assert_eq!(materialize_launches(), 1);
        assert_eq!(d2h_bytes(), 8);
        assert_eq!(full_d2h(), 0);
        assert_eq!(workspace_queries(), 1);
        assert_eq!(workspace_bytes(), 80);

        for output in outputs {
            ((*session.api).ReleaseValue.unwrap())(output);
        }
        ((*session.api).ReleaseValue.unwrap())(input);
    }
}

#[test]
fn cuda_unique_optional_outputs_stay_positional_through_real_ort_plugin() {
    let _lock = lock_ort();
    let Some(session) = (unsafe { session("unique_optional_subset", "cuda_unique_optional") })
    else {
        if std::env::var("NXRT_REQUIRE_ORT_TESTS").as_deref() == Ok("1") {
            panic!("CUDA plugin or ORT unavailable");
        }
        return;
    };
    unsafe {
        let reset_executed: libloading::Symbol<'_, unsafe extern "C" fn()> = session
            .plugin
            .get(b"nxrt_ep_reset_executed_node_count")
            .unwrap();
        let executed: libloading::Symbol<'_, unsafe extern "C" fn() -> usize> =
            session.plugin.get(b"nxrt_ep_executed_node_count").unwrap();
        let reset_stats: libloading::Symbol<'_, unsafe extern "C" fn()> = session
            .plugin
            .get(b"nxrt_ep_reset_unique_execution_stats")
            .unwrap();
        let metadata_launches: libloading::Symbol<'_, unsafe extern "C" fn() -> u64> = session
            .plugin
            .get(b"nxrt_ep_unique_metadata_launches")
            .unwrap();
        let materialize_launches: libloading::Symbol<'_, unsafe extern "C" fn() -> u64> = session
            .plugin
            .get(b"nxrt_ep_unique_materialize_launches")
            .unwrap();
        let d2h_bytes: libloading::Symbol<'_, unsafe extern "C" fn() -> u64> =
            session.plugin.get(b"nxrt_ep_unique_d2h_bytes").unwrap();
        reset_executed();
        reset_stats();

        let mut values = [3., 1., 3., 2., 1.];
        let input = input(session.api, &mut values);
        let input_name = CString::new("X").unwrap();
        let output_names: Vec<CString> = ["Y", "inverse_indices"]
            .into_iter()
            .map(|name| CString::new(name).unwrap())
            .collect();
        let output_name_ptrs: Vec<_> = output_names.iter().map(|name| name.as_ptr()).collect();
        let mut outputs = [ptr::null_mut(); 2];
        check(
            session.api,
            ((*session.api).Run.unwrap())(
                session.session,
                ptr::null(),
                [input_name.as_ptr()].as_ptr(),
                [input as *const ort::OrtValue].as_ptr(),
                1,
                output_name_ptrs.as_ptr(),
                2,
                outputs.as_mut_ptr(),
            ),
            "Run optional subset",
        );
        assert_eq!(executed(), 1);
        let y = output_bytes(session.api, outputs[0]);
        let inverse = output_bytes(session.api, outputs[1]);
        assert_eq!(y.0, [3]);
        assert_eq!(f32s(&y.2), [1., 2., 3.]);
        assert_eq!(inverse.0, [5]);
        assert_eq!(i64s(&inverse.2), [2, 0, 2, 1, 0]);
        assert_eq!(metadata_launches(), 1);
        assert_eq!(materialize_launches(), 1);
        assert_eq!(d2h_bytes(), 8);
        for output in outputs {
            ((*session.api).ReleaseValue.unwrap())(output);
        }
        ((*session.api).ReleaseValue.unwrap())(input);
    }
}
