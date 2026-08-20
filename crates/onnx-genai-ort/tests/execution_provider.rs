use std::path::{Path, PathBuf};

use onnx_genai_ort::{
    Environment, Session, SessionOptions, available_execution_providers, capability, ep_selection,
    resolve_execution_provider,
};

fn tiny_llm() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-llm/model.onnx.textproto")
}

#[test]
fn reports_available_execution_providers() {
    let providers = available_execution_providers().expect("available providers");
    println!("available execution providers: {providers:?}");
    assert!(
        providers
            .iter()
            .any(|provider| provider == "CPUExecutionProvider"),
        "available providers: {providers:?}"
    );
}

#[cfg(not(target_os = "macos"))]
#[test]
fn requested_unavailable_execution_provider_errors_by_default() {
    let env = Environment::new("execution-provider-error-test").expect("env");
    let options = SessionOptions::with_execution_provider(ep_selection("coreml"));
    let error = match Session::new(&env, &tiny_llm(), options) {
        Ok(_) => panic!("requested unavailable EP must not silently fall back"),
        Err(error) => error,
    };
    let message = error.to_string();
    assert!(message.contains("CoreMLExecutionProvider"), "{message}");
    assert!(message.contains("was requested"), "{message}");
    assert!(message.contains("ONNX_GENAI_EP=cpu"), "{message}");
    assert!(message.contains("ONNX_GENAI_EP_FALLBACK=1"), "{message}");
}

#[test]
fn unknown_execution_provider_error_lists_cubecl_providers() {
    let env = Environment::new("unknown-execution-provider-error-test").expect("env");
    let options = SessionOptions::with_execution_provider(ep_selection("openvino"));
    let error = match Session::new(&env, &tiny_llm(), options) {
        Ok(_) => panic!("unknown EP must not silently fall back"),
        Err(error) => error,
    };
    let message = error.to_string();
    assert!(message.contains("Unrecognized ONNX_GENAI_EP value 'openvino'"));
    assert!(message.contains("cubecl-webgpu"));
    assert!(message.contains("cubecl-vulkan"));
    assert!(message.contains("plugin:<library>"));
}

#[test]
fn cubecl_providers_resolve_to_plugin_bridges_without_advanced_caps() {
    for (name, registration_name) in [
        ("cubecl-webgpu", "onnxruntime_cubecl_webgpu_ep"),
        ("cubecl_webgpu", "onnxruntime_cubecl_webgpu_ep"),
        ("cubecl-vulkan", "onnxruntime_cubecl_vulkan_ep"),
        ("cubecl_vulkan", "onnxruntime_cubecl_vulkan_ep"),
    ] {
        let resolved = resolve_execution_provider(&ep_selection(name));
        assert!(resolved.caps.is_gpu(), "{name} should resolve to GPU caps");
        for flag in [
            capability::FIXED_CAPACITY_PRESENT_BINDING,
            capability::GRAPH_CAPTURE,
            capability::DEVICE_KV,
            capability::DEVICE_SAMPLING,
        ] {
            assert!(
                !resolved.caps.has(flag),
                "{name} should not advertise {flag}"
            );
        }

        let bridge = resolved
            .native_plugin_bridge()
            .expect("CubeCL EPs should use the plugin-library append path");
        assert_eq!(bridge.registration_name, registration_name);
        assert!(matches!(
            bridge.provider_name.as_str(),
            "cubecl-webgpu" | "cubecl-vulkan"
        ));
        assert!(
            bridge
                .lib
                .file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.contains("onnx_runtime_ep_cubecl_plugin")),
            "unexpected CubeCL plugin path: {}",
            bridge.lib.display()
        );
    }
}

#[cfg(not(target_os = "macos"))]
#[test]
fn explicit_cpu_fallback_is_visible_in_session_status() {
    let env = Environment::new("execution-provider-visible-fallback-test").expect("env");
    let options =
        SessionOptions::with_execution_provider(ep_selection("coreml")).with_cpu_fallback(true);
    let session =
        Session::new(&env, &tiny_llm(), options).expect("explicit fallback should retry on CPU");

    assert!(session.cpu_fallback_used());
    assert!(
        session
            .execution_providers()
            .iter()
            .any(|provider| provider.caps.is_host()),
        "effective providers should report CPU after fallback"
    );
    assert!(!session.graph_capture());
}

#[cfg(not(target_os = "macos"))]
#[test]
fn explicit_cpu_alternative_is_tried_after_unavailable_provider() {
    let env = Environment::new("execution-provider-ordered-cpu-test").expect("env");
    let mut options = SessionOptions::with_execution_provider(ep_selection("coreml"));
    options
        .execution_providers
        .push(resolve_execution_provider(&ep_selection("cpu")));
    let session = Session::new(&env, &tiny_llm(), options)
        .expect("explicit CPU alternative should load after CoreML is unavailable");

    assert!(session.cpu_fallback_used());
    assert_eq!(session.skipped_execution_providers().len(), 1);
    assert_eq!(session.skipped_execution_providers()[0].name, "coreml");
    assert!(
        session
            .execution_providers()
            .iter()
            .any(|provider| provider.caps.is_host())
    );
    assert!(
        session
            .execution_provider_status()
            .summary()
            .contains("skipped: coreml")
    );
}

#[cfg(not(target_os = "macos"))]
#[test]
fn fallback_flag_tries_ordered_alternatives_before_cpu_retry() {
    let env = Environment::new("execution-provider-ordered-fallback-test").expect("env");
    let mut options =
        SessionOptions::with_execution_provider(ep_selection("webgpu")).with_cpu_fallback(true);
    options
        .execution_providers
        .push(resolve_execution_provider(&ep_selection("coreml")));
    let session = Session::new(&env, &tiny_llm(), options)
        .expect("fallback flag should retry CPU after trying ordered EPs");

    assert!(session.cpu_fallback_used());
    let skipped = session
        .skipped_execution_providers()
        .iter()
        .map(|provider| provider.name.as_str())
        .collect::<Vec<_>>();
    assert_eq!(skipped, vec!["webgpu", "coreml"]);
    assert!(
        session
            .execution_provider_status()
            .summary()
            .contains("skipped: webgpu, coreml")
    );
}
