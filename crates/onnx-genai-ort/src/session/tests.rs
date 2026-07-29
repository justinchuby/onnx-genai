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
        cuda_provider_options("3".to_string(), true, &CudaAttentionMode::Unfused),
        vec![
            ("device_id".to_string(), "3".to_string()),
            ("enable_cuda_graph".to_string(), "1".to_string()),
            ("sdpa_kernel".to_string(), "16".to_string()),
        ]
    );
    assert_eq!(
        cuda_provider_options("0".to_string(), false, &CudaAttentionMode::Auto),
        vec![("device_id".to_string(), "0".to_string())]
    );
    assert_eq!(
        cuda_provider_options("0".to_string(), false, &CudaAttentionMode::Fused),
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
        &[],
    )
    .expect_err("CUDA must be rejected without the cargo feature");
    assert!(
        error
            .to_string()
            .contains("CUDA support not compiled in; rebuild with --features cuda")
    );
}
