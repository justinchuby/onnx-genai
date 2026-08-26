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

unsafe fn assignments(
    api: *const ort::OrtApi,
    session: *mut ort::OrtSession,
) -> Vec<(String, String)> {
    let get_info = unsafe { (*api).Session_GetEpGraphAssignmentInfo.unwrap() };
    let get_ep_name = unsafe { (*api).EpAssignedSubgraph_GetEpName.unwrap() };
    let get_nodes = unsafe { (*api).EpAssignedSubgraph_GetNodes.unwrap() };
    let get_op_type = unsafe { (*api).EpAssignedNode_GetOperatorType.unwrap() };
    let mut subgraphs: *const *const ort::OrtEpAssignedSubgraph = ptr::null();
    let mut subgraph_count = 0usize;
    unsafe {
        check(
            api,
            get_info(session, &mut subgraphs, &mut subgraph_count),
            "Session_GetEpGraphAssignmentInfo",
        )
    };
    let mut result = Vec::new();
    for subgraph_index in 0..subgraph_count {
        let subgraph = unsafe { *subgraphs.add(subgraph_index) };
        let mut ep_name = ptr::null();
        unsafe {
            check(
                api,
                get_ep_name(subgraph, &mut ep_name),
                "EpAssignedSubgraph_GetEpName",
            )
        };
        let ep_name = unsafe { CStr::from_ptr(ep_name) }
            .to_string_lossy()
            .into_owned();
        let mut nodes: *const *const ort::OrtEpAssignedNode = ptr::null();
        let mut node_count = 0usize;
        unsafe {
            check(
                api,
                get_nodes(subgraph, &mut nodes, &mut node_count),
                "EpAssignedSubgraph_GetNodes",
            )
        };
        for node_index in 0..node_count {
            let mut op_type = ptr::null();
            unsafe {
                check(
                    api,
                    get_op_type(*nodes.add(node_index), &mut op_type),
                    "EpAssignedNode_GetOperatorType",
                )
            };
            result.push((
                ep_name.clone(),
                unsafe { CStr::from_ptr(op_type) }
                    .to_string_lossy()
                    .into_owned(),
            ));
        }
    }
    result
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
    unsafe { session_with_fallback(model, registration, true) }
}

unsafe fn session_with_fallback(
    model: &str,
    registration: &str,
    disable_cpu_fallback: bool,
) -> Option<Session> {
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
    let mut config = vec![("session.record_ep_graph_assignment_info", "1")];
    if disable_cpu_fallback {
        config.push(("session.disable_cpu_ep_fallback", "1"));
    }
    for (key, value) in config {
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
    if !disable_cpu_fallback {
        unsafe {
            check(
                api,
                ((*api).SetSessionGraphOptimizationLevel.unwrap())(options, ort::ORT_DISABLE_ALL),
                "SetSessionGraphOptimizationLevel",
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
        .join("model.onnx.textproto");
    let text = std::fs::read_to_string(&fixture).unwrap();
    let parsed = onnx_std::textproto::from_textproto(&text)
        .unwrap_or_else(|error| panic!("{model}: parse textproto: {error}"));
    if matches!(model, "dft_device_shape" | "stft_device_shape") {
        verify_signal_fixture(model, &parsed);
    } else if model == "squeeze_device_axes" {
        verify_squeeze_fixture(&parsed);
    } else if model == "reduce_sum_device_axes" {
        verify_reduce_sum_fixture(&parsed);
    } else {
        verify_fixture(model, &parsed);
    }
    let bytes = ort_fixture_bytes(&text);
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

fn verify_signal_fixture(name: &str, model: &onnx_std::Model) {
    use onnx_runtime_ir::DataType;

    assert_eq!(model.metadata.ir_version, 11);
    assert_eq!(model.graph.opset_imports.get("").copied(), Some(24));
    assert!(model.graph.initializers.is_empty());
    assert_eq!(
        model.graph.inputs.len(),
        3,
        "{name}: signal plus two device-resident scalar operands"
    );
    assert_eq!(model.graph.outputs.len(), 1);
    let input = model.graph.value(model.graph.inputs[0]);
    let output = model.graph.value(model.graph.outputs[0]);
    assert_eq!(input.name.as_deref(), Some("X"));
    assert_eq!(input.dtype, DataType::Float32);
    assert_eq!(output.name.as_deref(), Some("Y"));
    assert_eq!(output.dtype, DataType::Float32);
    let nodes: Vec<_> = model.graph.nodes.iter().map(|(_, node)| node).collect();
    let signal_op = if name == "dft_device_shape" {
        "DFT"
    } else {
        "STFT"
    };
    assert!(nodes.iter().any(|node| node.op_type == signal_op));
    assert!(
        nodes.iter().any(|node| node.op_type == "Identity"),
        "{name}: scalar metadata must pass through a device-capable producer"
    );
}

fn verify_squeeze_fixture(model: &onnx_std::Model) {
    use onnx_runtime_ir::DataType;

    assert_eq!(model.metadata.ir_version, 11);
    assert_eq!(model.graph.opset_imports.get("").copied(), Some(24));
    assert!(model.graph.initializers.is_empty());
    assert_eq!(model.graph.inputs.len(), 2);
    assert_eq!(model.graph.outputs.len(), 1);
    let input = model.graph.value(model.graph.inputs[0]);
    let axes = model.graph.value(model.graph.inputs[1]);
    let output = model.graph.value(model.graph.outputs[0]);
    assert_eq!(input.name.as_deref(), Some("X"));
    assert_eq!(input.dtype, DataType::Float32);
    assert_eq!(axes.name.as_deref(), Some("axes_input"));
    assert_eq!(axes.dtype, DataType::Int64);
    assert_eq!(output.name.as_deref(), Some("Y"));
    assert_eq!(output.dtype, DataType::Float32);
    let nodes: Vec<_> = model.graph.nodes.iter().map(|(_, node)| node).collect();
    assert!(nodes.iter().any(|node| node.op_type == "Identity"));
    assert!(nodes.iter().any(|node| node.op_type == "Squeeze"));
}

fn verify_reduce_sum_fixture(model: &onnx_std::Model) {
    use onnx_runtime_ir::DataType;

    assert_eq!(model.metadata.ir_version, 11);
    assert_eq!(model.graph.opset_imports.get("").copied(), Some(24));
    assert!(model.graph.initializers.is_empty());
    assert_eq!(model.graph.inputs.len(), 2);
    assert_eq!(model.graph.outputs.len(), 1);
    let input = model.graph.value(model.graph.inputs[0]);
    let axes = model.graph.value(model.graph.inputs[1]);
    let output = model.graph.value(model.graph.outputs[0]);
    assert_eq!(input.name.as_deref(), Some("X"));
    assert_eq!(input.dtype, DataType::Float32);
    assert_eq!(axes.name.as_deref(), Some("axes_input"));
    assert_eq!(axes.dtype, DataType::Int64);
    assert_eq!(output.name.as_deref(), Some("Y"));
    assert_eq!(output.dtype, DataType::Float32);
    let nodes: Vec<_> = model.graph.nodes.iter().map(|(_, node)| node).collect();
    assert!(nodes.iter().any(|node| node.op_type == "Identity"));
    let reduction = nodes
        .iter()
        .find(|node| node.op_type == "ReduceSum")
        .expect("ReduceSum fixture node");
    assert_eq!(
        reduction
            .attr("keepdims")
            .and_then(onnx_runtime_ir::Attribute::as_int),
        Some(0)
    );
}

fn ort_fixture_bytes(text: &str) -> Vec<u8> {
    // ONNX's proto3 AttributeProto uses `type` as the presence discriminator,
    // but a generic proto3 encoder elides an explicitly written integer zero.
    // ORT then drops that valueless attribute and applies Unique's sorted=1
    // schema default. Restore the legal wire-level `i = 0` field for INT
    // attributes before handing the in-memory fixture to ORT.
    let bytes = onnx_std::textproto::to_binary(text).unwrap();
    rewrite_length_delimited(&bytes, 7, |graph| {
        rewrite_length_delimited(graph, 1, |node| {
            rewrite_length_delimited(node, 5, preserve_zero_int_attribute)
        })
    })
}

fn preserve_zero_int_attribute(attribute: &[u8]) -> Vec<u8> {
    const ATTRIBUTE_TYPE_INT: u64 = 2;
    if field_varint(attribute, 20) == Some(ATTRIBUTE_TYPE_INT)
        && field_varint(attribute, 3).is_none()
    {
        let mut output = attribute.to_vec();
        encode_varint((3 << 3) as u64, &mut output);
        encode_varint(0, &mut output);
        output
    } else {
        attribute.to_vec()
    }
}

fn field_varint(message: &[u8], wanted_field: u64) -> Option<u64> {
    let mut cursor = 0usize;
    while cursor < message.len() {
        let key = decode_varint(message, &mut cursor);
        let field = key >> 3;
        let wire_type = key & 7;
        match wire_type {
            0 => {
                let value = decode_varint(message, &mut cursor);
                if field == wanted_field {
                    return Some(value);
                }
            }
            1 => cursor = cursor.checked_add(8).expect("fixed64 offset overflow"),
            2 => {
                let length = decode_varint(message, &mut cursor) as usize;
                cursor = cursor
                    .checked_add(length)
                    .expect("length-delimited offset overflow");
            }
            5 => cursor = cursor.checked_add(4).expect("fixed32 offset overflow"),
            other => panic!("unsupported protobuf wire type {other}"),
        }
        assert!(cursor <= message.len(), "malformed protobuf field");
    }
    None
}

fn rewrite_length_delimited(
    message: &[u8],
    wanted_field: u64,
    mut rewrite: impl FnMut(&[u8]) -> Vec<u8>,
) -> Vec<u8> {
    let mut cursor = 0usize;
    let mut output = Vec::with_capacity(message.len());
    while cursor < message.len() {
        let field_start = cursor;
        let key = decode_varint(message, &mut cursor);
        let field = key >> 3;
        let wire_type = key & 7;
        match wire_type {
            0 => {
                decode_varint(message, &mut cursor);
                output.extend_from_slice(&message[field_start..cursor]);
            }
            1 => {
                cursor = cursor.checked_add(8).expect("fixed64 offset overflow");
                assert!(cursor <= message.len(), "malformed fixed64 field");
                output.extend_from_slice(&message[field_start..cursor]);
            }
            2 => {
                let length = decode_varint(message, &mut cursor) as usize;
                let payload_end = cursor
                    .checked_add(length)
                    .expect("length-delimited offset overflow");
                assert!(payload_end <= message.len(), "malformed protobuf field");
                if field == wanted_field {
                    let rewritten = rewrite(&message[cursor..payload_end]);
                    encode_varint(key, &mut output);
                    encode_varint(rewritten.len() as u64, &mut output);
                    output.extend_from_slice(&rewritten);
                } else {
                    output.extend_from_slice(&message[field_start..payload_end]);
                }
                cursor = payload_end;
            }
            5 => {
                cursor = cursor.checked_add(4).expect("fixed32 offset overflow");
                assert!(cursor <= message.len(), "malformed fixed32 field");
                output.extend_from_slice(&message[field_start..cursor]);
            }
            other => panic!("unsupported protobuf wire type {other}"),
        }
    }
    output
}

fn decode_varint(bytes: &[u8], cursor: &mut usize) -> u64 {
    let mut value = 0u64;
    for shift in (0..70).step_by(7) {
        let byte = *bytes.get(*cursor).expect("truncated protobuf varint");
        *cursor += 1;
        value |= u64::from(byte & 0x7f) << shift;
        if byte & 0x80 == 0 {
            return value;
        }
    }
    panic!("protobuf varint exceeds u64");
}

fn encode_varint(mut value: u64, output: &mut Vec<u8>) {
    while value >= 0x80 {
        output.push((value as u8 & 0x7f) | 0x80);
        value >>= 7;
    }
    output.push(value as u8);
}

fn verify_fixture(name: &str, model: &onnx_std::Model) {
    use onnx_runtime_ir::{Attribute, DataType};

    assert_eq!(model.metadata.ir_version, 11, "{name}: IR version");
    assert_eq!(
        model.graph.opset_imports.get("").copied(),
        Some(24),
        "{name}: ai.onnx opset"
    );
    if name == "nms_device_workspace" {
        assert_eq!(model.graph.inputs.len(), 5);
        let nodes: Vec<_> = model.graph.nodes.iter().map(|(_, node)| node).collect();
        assert_eq!(nodes.len(), 1);
        assert_eq!(nodes[0].op_type, "NonMaxSuppression");
        assert_eq!(model.graph.outputs.len(), 1);
        assert_eq!(
            model.graph.value(model.graph.outputs[0]).dtype,
            DataType::Int64
        );
        return;
    }
    assert!(model.graph.initializers.is_empty(), "{name}: initializers");
    assert_eq!(model.graph.inputs.len(), 1, "{name}: graph inputs");
    let input = model.graph.value(model.graph.inputs[0]);
    assert_eq!(input.name.as_deref(), Some("X"), "{name}: input name");
    assert_eq!(input.dtype, DataType::Float32, "{name}: input dtype");

    let nodes: Vec<_> = model.graph.nodes.iter().map(|(_, node)| node).collect();
    assert_eq!(nodes.len(), 1, "{name}: node count");
    let node = nodes[0];
    assert_eq!(node.op_type, "Unique", "{name}: op type");
    assert_eq!(node.inputs.len(), 1, "{name}: node input slots");
    assert_eq!(node.outputs.len(), 4, "{name}: positional output slots");
    let sorted = match node.attr("sorted") {
        Some(Attribute::Int(value)) => *value,
        other => panic!("{name}: expected integer sorted attribute, got {other:?}"),
    };

    let output_names: Vec<_> = node
        .outputs
        .iter()
        .map(|&output| model.graph.value(output).name.as_deref())
        .collect();
    let graph_outputs: Vec<_> = model
        .graph
        .outputs
        .iter()
        .map(|&output| {
            let value = model.graph.value(output);
            (value.name.as_deref(), value.dtype)
        })
        .collect();
    match name {
        "unique_all_outputs" => {
            assert_eq!(sorted, 0);
            assert_eq!(
                output_names,
                [
                    Some("Y"),
                    Some("indices"),
                    Some("inverse_indices"),
                    Some("counts")
                ]
            );
            assert_eq!(
                graph_outputs,
                [
                    (Some("Y"), DataType::Float32),
                    (Some("indices"), DataType::Int64),
                    (Some("inverse_indices"), DataType::Int64),
                    (Some("counts"), DataType::Int64),
                ]
            );
        }
        "unique_optional_subset" => {
            assert_eq!(sorted, 1);
            assert_eq!(
                output_names,
                [Some("Y"), None, Some("inverse_indices"), None],
                "{name}: optional middle/trailing slots"
            );
            assert_eq!(
                graph_outputs,
                [
                    (Some("Y"), DataType::Float32),
                    (Some("inverse_indices"), DataType::Int64),
                ]
            );
        }
        other => panic!("unexpected CUDA Unique fixture {other}"),
    }
}

unsafe fn tensor<T>(
    api: *const ort::OrtApi,
    values: &mut [T],
    shape: &[i64],
    dtype: ort::ONNXTensorElementDataType,
) -> *mut ort::OrtValue {
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
    let mut value = ptr::null_mut();
    unsafe {
        check(
            api,
            ((*api).CreateTensorWithDataAsOrtValue.unwrap())(
                memory_info,
                values.as_mut_ptr().cast(),
                std::mem::size_of_val(values),
                shape.as_ptr(),
                shape.len(),
                dtype,
                &mut value,
            ),
            "CreateTensorWithDataAsOrtValue",
        );
        ((*api).ReleaseMemoryInfo.unwrap())(memory_info);
    }
    value
}

unsafe fn input(api: *const ort::OrtApi, values: &mut [f32]) -> *mut ort::OrtValue {
    let shape = [values.len() as i64];
    unsafe {
        tensor(
            api,
            values,
            &shape,
            ort::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
        )
    }
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
    let Some(plugin_path) = onnx_runtime_ort_testkit::find_plugin_cdylib_with_features(
        "onnx-runtime-ep-cuda-plugin",
        &["cuda"],
    ) else {
        if std::env::var("NXRT_REQUIRE_ORT_TESTS").as_deref() == Ok("1") {
            panic!("CUDA plugin unavailable");
        }
        return;
    };
    let observer = unsafe { libloading::Library::new(plugin_path) }.unwrap();
    let reset_teardown: libloading::Symbol<'_, unsafe extern "C" fn()> = unsafe {
        observer
            .get(b"nxrt_ep_reset_allocator_teardown_stats")
            .unwrap()
    };
    unsafe { reset_teardown() };

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

    // ReleaseSession -> ReleaseAllocator -> UnregisterExecutionProviderLibrary
    // is the production teardown order. Keep a second module handle open so
    // the post-unregister counters remain callable without depending on unload
    // timing.
    drop(session);
    unsafe {
        let committed: libloading::Symbol<'_, unsafe extern "C" fn() -> u64> =
            observer.get(b"nxrt_ep_cuda_committed_bytes").unwrap();
        let quarantined: libloading::Symbol<'_, unsafe extern "C" fn() -> u64> = observer
            .get(b"nxrt_ep_cuda_allocator_quarantined_releases")
            .unwrap();
        let retained: libloading::Symbol<'_, unsafe extern "C" fn() -> u64> = observer
            .get(b"nxrt_ep_cuda_allocator_retained_releases")
            .unwrap();
        assert_eq!(
            committed(),
            0,
            "allocator teardown left CUDA memory committed"
        );
        assert_eq!(
            quarantined(),
            0,
            "allocator teardown quarantined a deferred release"
        );
        assert_eq!(
            retained(),
            0,
            "allocator teardown retained device ownership"
        );
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

#[test]
fn cuda_nms_dynamic_output_runs_through_real_ort_plugin() {
    let _lock = lock_ort();
    let Some(session) = (unsafe { session("nms_device_workspace", "cuda_nms") }) else {
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
            .get(b"nxrt_ep_reset_nms_execution_stats")
            .unwrap();
        let prepare: libloading::Symbol<'_, unsafe extern "C" fn() -> u64> =
            session.plugin.get(b"nxrt_ep_nms_prepare_launches").unwrap();
        let count: libloading::Symbol<'_, unsafe extern "C" fn() -> u64> =
            session.plugin.get(b"nxrt_ep_nms_count_launches").unwrap();
        let materialize: libloading::Symbol<'_, unsafe extern "C" fn() -> u64> = session
            .plugin
            .get(b"nxrt_ep_nms_materialize_launches")
            .unwrap();
        let d2h: libloading::Symbol<'_, unsafe extern "C" fn() -> u64> =
            session.plugin.get(b"nxrt_ep_nms_d2h_bytes").unwrap();
        let full_d2h: libloading::Symbol<'_, unsafe extern "C" fn() -> u64> = session
            .plugin
            .get(b"nxrt_ep_nms_full_input_d2h_bytes")
            .unwrap();
        reset_executed();
        reset_stats();

        let mut boxes: [f32; 12] = [
            0., 0., 1., 1., //
            0., 0., 0.9, 0.9, //
            2., 2., 3., 3.,
        ];
        let mut scores: [f32; 6] = [0.9, 0.8, 0.7, 0.1, 0.95, 0.2];
        let mut max_output = [2_i64];
        let mut iou = [0.5_f32];
        let mut score = [0.15_f32];
        let values = [
            tensor(
                session.api,
                &mut boxes,
                &[1, 3, 4],
                ort::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            ),
            tensor(
                session.api,
                &mut scores,
                &[1, 2, 3],
                ort::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            ),
            tensor(
                session.api,
                &mut max_output,
                &[],
                ort::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
            ),
            tensor(
                session.api,
                &mut iou,
                &[],
                ort::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            ),
            tensor(
                session.api,
                &mut score,
                &[],
                ort::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            ),
        ];
        let input_names: Vec<CString> = [
            "boxes",
            "scores",
            "max_output",
            "iou_threshold",
            "score_threshold",
        ]
        .into_iter()
        .map(|name| CString::new(name).unwrap())
        .collect();
        let input_ptrs: Vec<_> = input_names.iter().map(|name| name.as_ptr()).collect();
        let value_ptrs: Vec<_> = values
            .iter()
            .map(|value| *value as *const ort::OrtValue)
            .collect();
        let output_name = CString::new("selected_indices").unwrap();
        let mut output = ptr::null_mut();
        check(
            session.api,
            ((*session.api).Run.unwrap())(
                session.session,
                ptr::null(),
                input_ptrs.as_ptr(),
                value_ptrs.as_ptr(),
                values.len(),
                [output_name.as_ptr()].as_ptr(),
                1,
                &mut output,
            ),
            "Run CUDA NMS",
        );
        assert_eq!(compiled(), 1);
        assert_eq!(executed(), 1);
        let got = output_bytes(session.api, output);
        assert_eq!(got.0, [4, 3]);
        assert_eq!(got.1, ort::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
        assert_eq!(i64s(&got.2), [0, 0, 0, 0, 0, 2, 0, 1, 1, 0, 1, 2]);
        assert_eq!(prepare(), 1);
        assert_eq!(count(), 1);
        assert_eq!(materialize(), 1);
        assert_eq!(d2h(), 8);
        assert_eq!(full_d2h(), 0);
        ((*session.api).ReleaseValue.unwrap())(output);
        for value in values {
            ((*session.api).ReleaseValue.unwrap())(value);
        }
    }
}

#[test]
fn cuda_shape_value_ops_decline_before_device_scalar_host_reads() {
    struct RestorePartialClaim(Option<std::ffi::OsString>);
    impl Drop for RestorePartialClaim {
        fn drop(&mut self) {
            match self.0.take() {
                Some(value) => unsafe {
                    std::env::set_var("ONNX_GENAI_PLUGIN_PARTIAL_GPU_CLAIM", value)
                },
                None => unsafe { std::env::remove_var("ONNX_GENAI_PLUGIN_PARTIAL_GPU_CLAIM") },
            }
        }
    }

    let _lock = lock_ort();
    let _restore = RestorePartialClaim(std::env::var_os("ONNX_GENAI_PLUGIN_PARTIAL_GPU_CLAIM"));
    unsafe { std::env::set_var("ONNX_GENAI_PLUGIN_PARTIAL_GPU_CLAIM", "1") };
    for (fixture, op_type, input_shape, output_shape) in [
        (
            "dft_device_shape",
            "DFT",
            vec![1i64, 8, 6, 1],
            vec![1i64, 5, 6, 2],
        ),
        (
            "stft_device_shape",
            "STFT",
            vec![1i64, 8, 1],
            vec![1i64, 3, 3, 2],
        ),
        (
            "squeeze_device_axes",
            "Squeeze",
            vec![1i64, 1, 3],
            vec![1i64, 3],
        ),
        (
            "reduce_sum_device_axes",
            "ReduceSum",
            vec![2i64, 3, 4],
            vec![2i64, 4],
        ),
    ] {
        let registration = format!("cuda_shape_{op_type}");
        let Some(session) = (unsafe { session_with_fallback(fixture, &registration, false) })
        else {
            if std::env::var("NXRT_REQUIRE_ORT_TESTS").as_deref() == Ok("1") {
                panic!("CUDA plugin or ORT unavailable for {fixture}");
            }
            return;
        };
        unsafe {
            let assigned = assignments(session.api, session.session);
            assert!(
                assigned
                    .iter()
                    .any(|(ep, op)| ep == "cuda_ep" && op == "Identity"),
                "{fixture}: scalar Identity must execute on CUDA so its output is device-resident; \
                     assignments={assigned:?}"
            );
            assert!(
                !assigned
                    .iter()
                    .any(|(ep, op)| ep == "cuda_ep" && op == op_type),
                "{fixture}: {op_type} must decline before plugin Compute can host-read its device \
                     scalar; assignments={assigned:?}"
            );

            let elements = input_shape.iter().product::<i64>() as usize;
            let mut values = vec![0.0f32; elements];
            let input_value = tensor(
                session.api,
                &mut values,
                &input_shape,
                ort::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            );
            let mut scalar_storage = match op_type {
                "DFT" => vec![8i64, 1i64],
                "STFT" => vec![2i64, 4],
                "Squeeze" => vec![0i64],
                "ReduceSum" => vec![1i64],
                other => panic!("unhandled device-shape fixture op {other}"),
            };
            let mut input_values = vec![input_value];
            for scalar in &mut scalar_storage {
                let metadata_shape: &[i64] = if matches!(op_type, "Squeeze" | "ReduceSum") {
                    &[1]
                } else {
                    &[]
                };
                input_values.push(tensor(
                    session.api,
                    std::slice::from_mut(scalar),
                    metadata_shape,
                    ort::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
                ));
            }
            let input_names: Vec<CString> = match op_type {
                "DFT" => vec!["X", "dft_length_input", "axis_input"],
                "STFT" => vec!["X", "frame_step_input", "frame_length_input"],
                "Squeeze" => vec!["X", "axes_input"],
                "ReduceSum" => vec!["X", "axes_input"],
                other => panic!("unhandled device-shape fixture op {other}"),
            }
            .into_iter()
            .map(|name| CString::new(name).unwrap())
            .collect();
            let input_name_ptrs: Vec<_> = input_names.iter().map(|name| name.as_ptr()).collect();
            let input_value_ptrs: Vec<_> = input_values
                .iter()
                .map(|value| *value as *const ort::OrtValue)
                .collect();
            let output_name = CString::new("Y").unwrap();
            let mut output = ptr::null_mut();
            check(
                session.api,
                ((*session.api).Run.unwrap())(
                    session.session,
                    ptr::null(),
                    input_name_ptrs.as_ptr(),
                    input_value_ptrs.as_ptr(),
                    input_values.len(),
                    [output_name.as_ptr()].as_ptr(),
                    1,
                    &mut output,
                ),
                "Run device-shape fallback",
            );
            let (actual_shape, dtype, bytes) = output_bytes(session.api, output);
            assert_eq!(actual_shape, output_shape);
            assert_eq!(dtype, ort::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
            assert!(
                bytes
                    .chunks_exact(4)
                    .all(|value| f32::from_ne_bytes(value.try_into().unwrap()) == 0.0),
                "{fixture}: zero input must produce zero output"
            );
            ((*session.api).ReleaseValue.unwrap())(output);
            for value in input_values {
                ((*session.api).ReleaseValue.unwrap())(value);
            }
        }
    }
}
