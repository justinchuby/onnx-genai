use super::env_config::{
    default_cpu_ort_intra_op_threads_for_available_on, effective_intra_op_threads,
};
use super::*;

#[test]
fn recognizes_cuda_provider_names() {
    let available = vec!["CUDAExecutionProvider".to_string()];
    assert!(provider_is_available("CUDAExecutionProvider", &available));
    assert!(provider_is_available("CUDA", &available));
}

#[test]
fn fixed_capacity_present_binding_uses_capabilities_or_opt_in() {
    let resolve = |name: &str| resolve_execution_provider(&ep_selection(name));
    assert!(fixed_capacity_present_binding_supported(
        &[resolve("cpu")],
        false
    ));
    assert!(fixed_capacity_present_binding_supported(
        &[resolve("cuda")],
        false
    ));
    assert!(fixed_capacity_present_binding_supported(
        &[resolve("webgpu")],
        false
    ));
    assert!(fixed_capacity_present_binding_supported(
        &[resolve("metal")],
        false
    ));
    assert!(!fixed_capacity_present_binding_supported(
        &[resolve("coreml")],
        false
    ));
    assert!(!fixed_capacity_present_binding_supported(
        &[resolve("qnn")],
        false
    ));
    assert!(!fixed_capacity_present_binding_supported(
        &[resolve("some-unknown-ep")],
        false
    ));
    // The operator opt-in overrides the conservative default.
    assert!(fixed_capacity_present_binding_supported(
        &[resolve("some-unknown-ep")],
        true
    ));
}

#[test]
fn resolves_cpu_to_host_defaults() {
    let resolved = resolve_execution_provider(&ep_selection("cpu"));
    assert!(resolved.caps.is_host());
    assert!(
        resolved
            .caps
            .has(capability::FIXED_CAPACITY_PRESENT_BINDING)
    );
    assert!(!resolved.is_strict());
    assert!(matches!(
        resolved.strategy,
        ep_compat::AppendStrategy::HostDefault
    ));
}

#[test]
fn resolves_qnn_to_conservative_plugin_npu() {
    let resolved = resolve_execution_provider(&ep_selection("qnn"));
    assert_eq!(resolved.caps.name, "qnn");
    assert_eq!(resolved.caps.hardware, HardwareKind::Npu);
    assert!(
        !resolved
            .caps
            .has(capability::FIXED_CAPACITY_PRESENT_BINDING)
    );
    assert!(!resolved.caps.has(capability::DEVICE_KV));
    assert!(!resolved.caps.has(capability::GRAPH_CAPTURE));
    assert!(resolved.is_strict());
    match &resolved.strategy {
        ep_compat::AppendStrategy::PluginLibrary {
            lib,
            registration_name,
            options,
            device,
        } => {
            let expected_plugin = if cfg!(windows) {
                "onnxruntime_providers_qnn.dll"
            } else if cfg!(target_os = "macos") {
                "libonnxruntime_providers_qnn.dylib"
            } else {
                "libonnxruntime_providers_qnn.so"
            };
            let expected_backend = if cfg!(windows) {
                "QnnHtp.dll"
            } else if cfg!(target_os = "macos") {
                "libQnnHtp.dylib"
            } else {
                "libQnnHtp.so"
            };
            assert_eq!(
                lib.file_name().and_then(|name| name.to_str()),
                Some(expected_plugin)
            );
            assert_eq!(registration_name, "onnxruntime_qnn_ep");
            assert_eq!(device.as_deref(), Some("NPU"));
            assert_eq!(
                options
                    .iter()
                    .find(|(key, _)| key == "backend_path")
                    .map(|(_, value)| value.as_str()),
                Some(expected_backend)
            );
        }
        other => panic!("expected QNN PluginLibrary, got {other:?}"),
    }
}

#[test]
fn resolves_cuda_to_nvidia_gpu_capabilities() {
    let resolved = resolve_execution_provider(&ep_selection("cuda"));
    assert!(resolved.caps.is_gpu());
    assert!(resolved.caps.is_nvidia());
    assert!(resolved.caps.device_id().is_some());
    for flag in [
        capability::FIXED_CAPACITY_PRESENT_BINDING,
        capability::GRAPH_CAPTURE,
        capability::DEVICE_KV,
        capability::DEVICE_SAMPLING,
    ] {
        assert!(resolved.caps.has(flag), "cuda should advertise {flag}");
    }
    #[cfg(feature = "cuda")]
    assert!(matches!(
        resolved.strategy,
        ep_compat::AppendStrategy::CudaTyped { .. }
    ));
    #[cfg(not(feature = "cuda"))]
    assert!(matches!(
        resolved.strategy,
        ep_compat::AppendStrategy::CudaUnavailable
    ));
}

#[test]
fn cuda_selection_device_id_option_overrides_environment_default() {
    let mut selection = ep_selection("cuda");
    selection
        .options
        .insert("device_id".to_string(), "3".to_string());

    let resolved = resolve_execution_provider(&selection);

    assert_eq!(resolved.caps.device_id(), Some(3));
    #[cfg(feature = "cuda")]
    assert!(matches!(
        resolved.strategy,
        ep_compat::AppendStrategy::CudaTyped { device_id: 3 }
    ));
}

#[test]
fn convenience_selection_uses_env_name_normalization() {
    let cuda = ep_selection("CUDA");
    assert_eq!(cuda, EpSelection::new("CUDA"));
    assert!(resolve_execution_provider(&cuda).caps.is_nvidia());

    let cpu = ep_selection(" cpu ");
    assert_eq!(cpu, EpSelection::new(" cpu "));
    assert!(resolve_execution_provider(&cpu).caps.is_host());
}

#[test]
fn resolves_unknown_ep_to_named_generic_other_hardware() {
    let resolved = resolve_execution_provider(&ep_selection("openvino"));
    assert_eq!(resolved.caps.hardware, HardwareKind::Other);
    assert!(!resolved.caps.is_gpu());
    assert!(!resolved.caps.is_host());
    assert!(
        !resolved
            .caps
            .has(capability::FIXED_CAPACITY_PRESENT_BINDING)
    );
    assert!(!resolved.is_strict());
    match &resolved.strategy {
        ep_compat::AppendStrategy::NamedGeneric {
            ort_name,
            provider_name,
        } => {
            assert_eq!(ort_name, "openvino");
            assert_eq!(provider_name, "openvinoExecutionProvider");
        }
        other => panic!("expected NamedGeneric, got {other:?}"),
    }
}

#[test]
fn named_generic_forwards_opaque_provider_options() {
    let config = onnx_genai_runtime_config::RuntimeConfig::from_fn(|name| match name {
        "ONNX_GENAI_EP" => Some("openvino".to_owned()),
        "ONNX_GENAI_EP_OPTIONS" => Some("device_type=GPU,precision=FP16".to_owned()),
        _ => None,
    });
    let ExecutionProviderEntry::Builtin(selection) = &config.execution_providers[0] else {
        panic!("expected named provider selection");
    };
    let resolved = resolve_execution_provider(selection);
    assert_eq!(
        named_provider_options(&resolved),
        vec![("device_type", "GPU"), ("precision", "FP16")]
    );
}

#[test]
fn resolves_webgpu_and_coreml_separator_aliases() {
    for name in ["webgpu", "web-gpu", "web_gpu"] {
        let resolved = resolve_execution_provider(&ep_selection(name));
        assert!(
            resolved.caps.is_gpu(),
            "{name} should resolve to WebGPU GPU caps"
        );
        assert!(
            resolved.transitional_webgpu,
            "{name} should be the WebGPU transitional EP"
        );
        assert!(
            resolved.caps.has(capability::DEVICE_KV),
            "{name} should keep WebGPU device-KV"
        );
        assert_eq!(resolved.caps.name, "webgpu");
    }
    for name in ["coreml", "core-ml", "core_ml"] {
        let resolved = resolve_execution_provider(&ep_selection(name));
        assert_eq!(
            resolved.caps.hardware,
            HardwareKind::Npu,
            "{name} should resolve to CoreML"
        );
        assert_eq!(resolved.caps.name, "coreml");
        assert!(matches!(
            resolved.strategy,
            ep_compat::AppendStrategy::NamedGeneric { .. }
        ));
    }
}

#[test]
fn strict_providers_include_cuda_and_plugins() {
    // CUDA and Metal (a plugin library) are strict: load failure must not
    // silently fall back to CPU. Named-generic providers are non-strict.
    let cuda = SessionOptions::with_execution_provider(ep_selection("cuda"));
    assert!(requested_non_cpu_provider(&cuda));
    assert!(requested_strict_provider(&cuda));

    let metal = SessionOptions::with_execution_provider(ep_selection("metal"));
    assert!(requested_non_cpu_provider(&metal));
    assert!(requested_strict_provider(&metal));

    let qnn = SessionOptions::with_execution_provider(ep_selection("qnn"));
    assert!(requested_non_cpu_provider(&qnn));
    assert!(requested_strict_provider(&qnn));

    let webgpu = SessionOptions::with_execution_provider(ep_selection("webgpu"));
    assert!(requested_non_cpu_provider(&webgpu));
    assert!(!requested_strict_provider(&webgpu));

    let cpu = SessionOptions::cpu();
    assert!(!requested_non_cpu_provider(&cpu));
    assert!(!requested_strict_provider(&cpu));
}

#[test]
fn explicit_intra_op_threads_override_auto_default() {
    let options = SessionOptions::cpu().with_intra_op_threads(3);
    assert_eq!(effective_intra_op_threads(&options), 3);
}

#[test]
fn windows_arm64_cpu_ort_default_uses_half_concurrency_capped() {
    assert_eq!(
        default_cpu_ort_intra_op_threads_for_available_on(12, true),
        Some(6)
    );
    assert_eq!(
        default_cpu_ort_intra_op_threads_for_available_on(8, true),
        Some(4)
    );
    assert_eq!(
        default_cpu_ort_intra_op_threads_for_available_on(2, true),
        Some(1)
    );
    assert_eq!(
        default_cpu_ort_intra_op_threads_for_available_on(64, true),
        Some(16)
    );
}

#[test]
fn cpu_ort_auto_default_is_disabled_off_windows_arm64() {
    assert_eq!(
        default_cpu_ort_intra_op_threads_for_available_on(12, false),
        None
    );
}

#[cfg(feature = "cuda")]
#[test]
fn unfused_cuda_attention_uses_math_provider_option() {
    assert_eq!(
        cuda_provider_options("3".to_string(), true, &CudaAttentionMode::Unfused, None),
        vec![
            ("device_id".to_string(), "3".to_string()),
            ("enable_cuda_graph".to_string(), "1".to_string()),
            ("sdpa_kernel".to_string(), "16".to_string()),
        ]
    );
    assert_eq!(
        cuda_provider_options("0".to_string(), false, &CudaAttentionMode::Auto, None),
        vec![("device_id".to_string(), "0".to_string())]
    );
    assert_eq!(
        cuda_provider_options("0".to_string(), false, &CudaAttentionMode::Fused, None),
        vec![("device_id".to_string(), "0".to_string())]
    );
}

#[cfg(feature = "cuda")]
#[test]
fn unavailable_cuda_error_is_actionable() {
    let error = cuda_provider_unavailable_error(&["CPUExecutionProvider".to_string()]);
    let message = error.to_string();
    assert!(message.contains("CUDAExecutionProvider was requested"));
    assert!(message.contains(cuda_provider_library_name()));
    assert!(message.contains(cuda_library_search_path()));
    assert!(message.contains("ONNX_GENAI_EP=cpu"));
}

#[cfg(not(target_os = "macos"))]
#[test]
fn auto_default_providers_are_macos_only() {
    // MLX/Metal auto-selection is gated to macOS; every other platform keeps
    // the plain CPU default regardless of environment.
    assert!(super::auto_default_execution_providers().is_none());
}

#[cfg(not(feature = "cuda"))]
#[test]
fn cuda_request_requires_compile_time_feature() {
    let resolved = resolve_execution_provider(&ep_selection("cuda"));
    let error = append_execution_provider(
        &Environment::new("cuda-feature-test").expect("environment"),
        std::ptr::null_mut(),
        &resolved,
        false,
        &CudaAttentionMode::Auto,
        None,
        &[],
    )
    .expect_err("CUDA must be rejected without the cargo feature");
    assert!(
        error
            .to_string()
            .contains("CUDA support not compiled in; rebuild with --features cuda")
    );
}

/// The stream `Session::user_compute_stream` reports must be the stream the
/// CUDA EP actually computes on.
///
/// The getter exists so a caller can order its own device work against the
/// session's runs without a device-wide barrier. A session that reported a
/// stream it was not computing on would turn every such ordering into a silent
/// race, so this test proves agreement by *executing* rather than by inspecting
/// fields:
///
/// 1. The input buffer is filled with `0xFF` bytes - `f32::NAN` - on the
///    reported stream, then a long chain of device-to-device copies is queued
///    behind it to push the real input far back in that stream's timeline.
/// 2. The real input is queued on the same stream, and the session runs with
///    **no** device synchronization in between.
/// 3. The output is read back with a stream-ordered copy on the reported
///    stream, synchronizing only that stream.
///
/// If the EP ran on a stream of its own, step 2 would read the poison that step
/// 1 left behind while the real input was still queued, and the encoder would
/// emit NaN. Ordering is the only thing that makes this deterministic, so the
/// assertion on finite output is an assertion about which stream ran the model.
///
/// Graph capture is on because that is the configuration in which a stream the
/// session did not adopt fails at run time rather than merely running slowly.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires a CUDA device"]
fn cuda_session_computes_on_the_stream_it_reports() {
    use crate::binding::IoBinding;
    use crate::cuda_rt::CudaComputeStream;
    use crate::value::{DataType, Value};

    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/tiny-whisper/encoder.onnx.textproto");
    assert!(path.exists(), "fixture missing: {}", path.display());

    // Two skips, and only two. Both are proven by asking the system directly
    // before anything is built, never by pattern-matching a later error: the
    // driver must give us a stream, and the linked ONNX Runtime must report the
    // CUDA provider. Every failure past this point is a real failure.
    let Ok(probe) = CudaComputeStream::new(0) else {
        eprintln!("cuda_session_computes_on_the_stream_it_reports: no CUDA device, skipping");
        return;
    };
    drop(probe);
    let providers = super::options::available_execution_providers().expect("provider list");
    if !provider_is_available("CUDAExecutionProvider", &providers) {
        eprintln!(
            "cuda_session_computes_on_the_stream_it_reports: linked ONNX Runtime reports \
             {providers:?}, no CUDA provider, skipping"
        );
        return;
    }

    let environment = Environment::new("shared-stream-execution-test").expect("environment");
    let mut options = SessionOptions::with_execution_provider(ep_selection("cuda"));
    options.graph_capture = true;
    options.share_cuda_compute_stream();
    let stream = options
        .cuda_user_compute_stream
        .clone()
        .expect("a CUDA device is present, so the pipeline stream must have been created");
    let handle = stream.handle();

    // Session creation fails loudly if ONNX Runtime did not record the stream,
    // so no error here is a skip.
    let session =
        Session::new(&environment, &path, options).expect("CUDA session with shared stream");
    assert_eq!(
        session.user_compute_stream(),
        Some(handle),
        "the session must report exactly the stream its provider adopted"
    );

    // Device-resident I/O: CUDA graph capture requires it, and it is what lets
    // the test address the input buffer directly.
    let allocator = session
        .device_kv_allocator()
        .expect("device allocator query")
        .expect("the CUDA EP is attached, so a device allocator must exist");

    const FRAMES: usize = 8;
    const MELS: usize = 80;
    let input = Value::empty_in(
        &[1, MELS as i64, FRAMES as i64],
        DataType::Float32,
        &allocator,
    )
    .expect("device input");
    let input_ptr = input.data_ptr_addr().expect("device input address");
    let input_bytes = MELS * FRAMES * std::mem::size_of::<f32>();

    let output = Value::empty_in(&[1, 4, 4], DataType::Float32, &allocator).expect("device output");
    let output_ptr = output.data_ptr_addr().expect("device output address");
    let output_len = 4 * 4;

    let mut binding = IoBinding::new(&session).expect("binding");
    binding
        .bind_input("input_features", &input)
        .expect("bind input");
    binding
        .bind_output("encoder_hidden_states", &output)
        .expect("bind output");

    // Scratch for the delay chain. 16 MiB copied 64 times is ~1 GiB of traffic,
    // which is hundreds of microseconds of stream time - long enough that a run
    // on any other stream would reach the input first.
    const SCRATCH: usize = 16 * 1024 * 1024;
    let scratch =
        Value::empty_in(&[SCRATCH as i64], DataType::Uint8, &allocator).expect("scratch a");
    let scratch_b =
        Value::empty_in(&[SCRATCH as i64], DataType::Uint8, &allocator).expect("scratch b");
    let scratch_ptr = scratch.data_ptr_addr().expect("scratch a address");
    let scratch_b_ptr = scratch_b.data_ptr_addr().expect("scratch b address");
    stream
        .memset_async(scratch_ptr, 0, SCRATCH)
        .expect("prime scratch");

    let host_input = vec![0u8; input_bytes];
    let mut host_output = vec![0u8; output_len * std::mem::size_of::<f32>()];

    // Three runs: the first captures the graph for annotation 0 and the rest
    // replay it. The replays are the discriminating ones - capture itself
    // involves enough extra synchronization to hide a wrong stream, whereas a
    // replay is lean and races. Verified by handing ONNX Runtime a decoy stream
    // while the session still reported this one: pass 0 passed and pass 1 came
    // back all-NaN.
    for pass in 0..3 {
        // Poison, delay, then the real input - all on the reported stream, with
        // no device-wide barrier anywhere in this block.
        stream
            .memset_async(input_ptr, 0xFF, input_bytes)
            .expect("poison the input");
        for _ in 0..64 {
            stream
                .memcpy_device_to_device_async(scratch_b_ptr, scratch_ptr, SCRATCH)
                .expect("delay chain");
        }
        stream
            .memcpy_host_to_device_async(input_ptr, &host_input)
            .expect("queue the real input");

        session
            .run_with_binding_graph(&binding, 0)
            .unwrap_or_else(|error| panic!("run {pass} on the shared stream: {error}"));

        stream
            .memcpy_device_to_host_async(&mut host_output, output_ptr)
            .expect("stream-ordered readback");
        stream
            .synchronize()
            .expect("only this stream is synchronized");

        let values: Vec<f32> = host_output
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
            .collect();
        assert_eq!(values.len(), output_len);
        assert!(
            values.iter().all(|value| value.is_finite()),
            "run {pass} produced {values:?}; a non-finite value means the model read the \
             poison this test queued ahead of the real input, so the CUDA EP did not run on \
             the stream the session reports"
        );
    }
}
