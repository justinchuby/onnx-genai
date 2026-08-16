use super::env_config::{
    default_cpu_ort_intra_op_threads_for_available_on, effective_intra_op_threads,
    resolve_execution_provider_entry,
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
    // The operator opt-in overrides the conservative default.
    assert!(fixed_capacity_present_binding_supported(
        &[resolve("coreml")],
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
fn rejects_unrecognized_ep_names_without_conservative_append() {
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
        ep_compat::AppendStrategy::UnsupportedName { name } => {
            assert_eq!(name, "openvino");
        }
        other => panic!("expected UnsupportedName, got {other:?}"),
    }

    let error = append_execution_provider(
        &Environment::new("unsupported-ep-test").expect("environment"),
        std::ptr::null_mut(),
        &resolved,
        false,
        &CudaAttentionMode::Auto,
        &["CPUExecutionProvider".to_string()],
    )
    .expect_err("unrecognized provider names must be rejected");
    let message = error.to_string();
    assert!(message.contains("Unrecognized ONNX_GENAI_EP value 'openvino'"));
    assert!(message.contains("Known values are"));
    assert!(message.contains("plugin:<library>"));
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
    assert!(resolved.is_unsupported_name());
}

#[test]
fn invalid_plugin_requests_are_preserved_as_session_creation_errors() {
    let bare = onnx_genai_runtime_config::RuntimeConfig::from_fn(|name| match name {
        "ONNX_GENAI_EP" => Some("plugin".to_owned()),
        _ => None,
    });
    let resolved = resolve_execution_provider_entry(&bare.execution_providers[0], &bare);
    let error = append_execution_provider(
        &Environment::new("bare-plugin-test").expect("environment"),
        std::ptr::null_mut(),
        &resolved,
        false,
        &CudaAttentionMode::Auto,
        &["CPUExecutionProvider".to_string()],
    )
    .expect_err("bare plugin without a library must fail instead of disappearing");
    assert!(
        error.to_string().contains("ONNX_GENAI_EP_LIBRARY"),
        "{error}"
    );

    let inline = onnx_genai_runtime_config::RuntimeConfig::from_fn(|name| match name {
        "ONNX_GENAI_EP" => Some("plugin:".to_owned()),
        _ => None,
    });
    let resolved = resolve_execution_provider_entry(&inline.execution_providers[0], &inline);
    let error = append_execution_provider(
        &Environment::new("empty-inline-plugin-test").expect("environment"),
        std::ptr::null_mut(),
        &resolved,
        false,
        &CudaAttentionMode::Auto,
        &["CPUExecutionProvider".to_string()],
    )
    .expect_err("inline plugin with an empty library must fail instead of disappearing");
    assert!(error.to_string().contains("plugin:<library>"), "{error}");
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
    // CUDA and plugin libraries are strict. Other recognized built-ins are not
    // strict, but they still cannot fall back to CPU unless explicitly opted in.
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
fn cpu_fallback_requires_explicit_opt_in_and_supported_provider_name() {
    let default_webgpu = SessionOptions::with_execution_provider(ep_selection("webgpu"));
    assert!(!cpu_fallback_allowed(&default_webgpu));

    let fallback_webgpu =
        SessionOptions::with_execution_provider(ep_selection("webgpu")).with_cpu_fallback(true);
    assert!(cpu_fallback_allowed(&fallback_webgpu));

    let unsupported =
        SessionOptions::with_execution_provider(ep_selection("openvino")).with_cpu_fallback(true);
    assert!(
        !cpu_fallback_allowed(&unsupported),
        "unknown provider names must error instead of falling back"
    );
}

#[test]
fn unavailable_named_provider_errors_instead_of_falling_back() {
    let resolved = resolve_execution_provider(&ep_selection("webgpu"));
    let error = append_execution_provider(
        &Environment::new("webgpu-unavailable-test").expect("environment"),
        std::ptr::null_mut(),
        &resolved,
        false,
        &CudaAttentionMode::Auto,
        &["CPUExecutionProvider".to_string()],
    )
    .expect_err("unavailable requested provider must fail clearly");
    let message = error.to_string();
    assert!(message.contains("WebGpuExecutionProvider"));
    assert!(message.contains("was requested"));
    assert!(message.contains("Available providers"));
    assert!(message.contains("ONNX_GENAI_EP=cpu"));
    assert!(message.contains("ONNX_GENAI_EP_FALLBACK=1"));
}

#[test]
fn ordered_cpu_alternative_preserves_non_ep_options() {
    let mut options = SessionOptions::with_execution_provider(ep_selection("coreml"));
    options
        .execution_providers
        .push(resolve_execution_provider(&ep_selection("cpu")));
    options.optimization_level = 0;
    options.intra_op_num_threads = 3;
    options.inter_op_num_threads = 2;
    options
        .session_config_entries
        .push(("custom.entry".to_string(), "kept".to_string()));

    let cpu_candidate = options.for_execution_providers(vec![cpu_provider()], false);

    assert_eq!(cpu_candidate.optimization_level, 0);
    assert_eq!(cpu_candidate.intra_op_num_threads, 3);
    assert_eq!(cpu_candidate.inter_op_num_threads, 2);
    assert!(cpu_candidate.has_session_config("custom.entry"));
    assert!(!cpu_candidate.graph_capture);
    assert!(!cpu_candidate.webgpu_disable_validation);
}

#[test]
fn provider_owned_session_config_is_recomputed_for_candidates() {
    let mut qnn = SessionOptions::with_execution_provider(ep_selection("qnn"));
    qnn.session_config_entries
        .push(("ep.context_enable".to_string(), "1".to_string()));
    qnn.session_config_entries
        .push(("ep.context_file_path".to_string(), "qnn.ctx".to_string()));
    qnn.session_config_entries.push((
        "session.disable_cpu_ep_fallback".to_string(),
        "1".to_string(),
    ));
    qnn.session_config_entries
        .push(("custom.entry".to_string(), "kept".to_string()));

    let cpu_candidate = qnn.for_execution_providers(vec![cpu_provider()], false);

    assert!(cpu_candidate.has_session_config("custom.entry"));
    assert!(!cpu_candidate.has_session_config("ep.context_enable"));
    assert!(!cpu_candidate.has_session_config("ep.context_file_path"));
    assert!(!cpu_candidate.has_session_config("session.disable_cpu_ep_fallback"));
}

#[test]
fn same_provider_candidate_preserves_qnn_context_session_config() {
    let mut qnn = SessionOptions::with_execution_provider(ep_selection("qnn"));
    qnn.session_config_entries
        .push(("ep.context_enable".to_string(), "1".to_string()));
    qnn.session_config_entries
        .push(("ep.context_file_path".to_string(), "qnn.ctx".to_string()));
    qnn.session_config_entries.push((
        "session.disable_cpu_ep_fallback".to_string(),
        "1".to_string(),
    ));
    qnn.session_config_entries
        .push(("custom.entry".to_string(), "kept".to_string()));

    let qnn_candidate = qnn.for_execution_providers(
        vec![resolve_execution_provider(&ep_selection("qnn"))],
        false,
    );

    assert!(qnn_candidate.has_session_config("custom.entry"));
    assert!(qnn_candidate.has_session_config("ep.context_enable"));
    assert!(qnn_candidate.has_session_config("ep.context_file_path"));
    assert!(qnn_candidate.has_session_config("session.disable_cpu_ep_fallback"));
}

#[test]
fn ordered_candidates_keep_accelerator_chain_before_cpu() {
    let mut options = SessionOptions::with_execution_provider(ep_selection("webgpu"));
    options
        .execution_providers
        .push(resolve_execution_provider(&ep_selection("coreml")));
    options
        .execution_providers
        .push(resolve_execution_provider(&ep_selection("cpu")));

    let candidates = execution_provider_candidates(&options);
    let names = candidates
        .iter()
        .map(|candidate| provider_names(&candidate.providers))
        .collect::<Vec<_>>();

    assert_eq!(
        names,
        vec![
            "webgpu, coreml",
            "webgpu, coreml",
            "coreml",
            "coreml",
            "cpu"
        ]
    );
    assert!(
        !candidates[0].allow_cpu_nodes,
        "CPU node fallback must not be allowed while accelerator alternatives remain"
    );
    assert!(
        candidates[1].allow_cpu_nodes,
        "CPU node fallback is only considered after the full accelerator chain failed without CPU"
    );
    assert!(
        candidates[4].whole_session_cpu_fallback,
        "CPU is only the final whole-session alternative"
    );
}

#[test]
fn mid_chain_append_failure_prunes_only_the_failing_provider() {
    let mut options = SessionOptions::with_execution_provider(ep_selection("webgpu"));
    options
        .execution_providers
        .push(resolve_execution_provider(&ep_selection("coreml")));
    options
        .execution_providers
        .push(resolve_execution_provider(&ep_selection("cpu")));
    let mut candidates = execution_provider_candidates(&options);

    prune_failed_provider_from_candidates(&mut candidates, "coreml");

    let surviving = candidates
        .iter()
        .filter(|candidate| !candidate.providers.is_empty())
        .map(|candidate| {
            (
                provider_names(&candidate.providers),
                candidate.allow_cpu_nodes,
                candidate.whole_session_cpu_fallback,
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(
        surviving,
        vec![
            ("webgpu".to_string(), false, false),
            ("webgpu".to_string(), true, false),
            ("cpu".to_string(), false, true),
        ],
        "a CoreML append failure after WebGPU must keep WebGPU viable before CPU"
    );
}

#[test]
fn implicit_node_cpu_fallback_is_disabled_unless_explicitly_allowed() {
    let webgpu = SessionOptions::with_execution_provider(ep_selection("webgpu"));
    assert!(implicit_cpu_ep_fallback_disabled(&webgpu));

    let allowed =
        SessionOptions::with_execution_provider(ep_selection("webgpu")).with_cpu_fallback(true);
    assert!(!implicit_cpu_ep_fallback_disabled(&allowed));

    let cpu = SessionOptions::cpu();
    assert!(!implicit_cpu_ep_fallback_disabled(&cpu));
}

#[test]
fn append_chain_error_reports_the_mid_chain_provider_that_failed() {
    let mut options = SessionOptions::with_execution_provider(ep_selection("coreml"));
    options.execution_providers[0] = ResolvedEp {
        selection: ep_selection("webgpu"),
        caps: EpCapabilities::new("webgpu", HardwareKind::Gpu, None, None, &[]),
        strategy: ep_compat::AppendStrategy::HostDefault,
        graph_capture_env: false,
        transitional_webgpu: false,
    };
    options
        .execution_providers
        .push(resolve_execution_provider(&ep_selection("coreml")));

    let error = append_execution_providers(
        &Environment::new("mid-chain-provider-error-test").expect("environment"),
        std::ptr::null_mut(),
        &options,
    )
    .expect_err("the second provider append must fail on this host");

    assert_eq!(error.provider_index, 1);
    assert_eq!(error.provider_name, "coreml");
    assert!(
        error.source.to_string().contains("CoreMLExecutionProvider"),
        "failure must be attributed to CoreML, not the working WebGPU prefix: {error:?}"
    );
}

#[test]
fn execution_provider_status_surfaces_cpu_and_skipped_providers() {
    let status = ExecutionProviderStatus {
        active: vec!["cpu".to_string()],
        skipped: vec![SkippedExecutionProvider {
            name: "webgpu".to_string(),
            reason: "unavailable".to_string(),
        }],
        whole_session_cpu_fallback: true,
        node_cpu_fallback_allowed: false,
        node_cpu_fallback_used: None,
    };

    assert_eq!(
        status.summary(),
        "cpu (CPU session fallback); skipped: webgpu"
    );

    let mixed = ExecutionProviderStatus {
        active: vec!["webgpu".to_string()],
        skipped: Vec::new(),
        whole_session_cpu_fallback: false,
        node_cpu_fallback_allowed: true,
        node_cpu_fallback_used: None,
    };
    assert_eq!(
        mixed.summary(),
        "webgpu (CPU node fallback allowed; actual placement unavailable)"
    );
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
