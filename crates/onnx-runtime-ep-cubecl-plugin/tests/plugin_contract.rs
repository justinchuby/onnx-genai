use onnx_runtime_ep_cubecl::backend::CubeclBackend;

#[test]
fn compiled_factory_candidates_follow_backend_availability() {
    let expected: Vec<_> = CubeclBackend::ALL
        .into_iter()
        .filter(|backend| backend.unavailable_message().is_none())
        .map(CubeclBackend::provider_name)
        .collect();

    assert_eq!(
        onnx_runtime_ep_cubecl_plugin::compile_available_provider_names(),
        expected,
        "plugin factory candidates must be CubeclBackend::ALL filtered by availability()"
    );
}

#[test]
fn zero_factory_diagnostic_is_actionable() {
    let reasons = vec![
        "execution provider 'cubecl-webgpu' could not open a GPU device (ordinal 0). Use ONNX_GENAI_EP=cpu.".to_string(),
        "execution provider 'cubecl-vulkan' is not compiled into this build. Rebuild with --features vulkan.".to_string(),
    ];
    let message = onnx_runtime_ep_cubecl_plugin::zero_factory_diagnostic(&reasons);

    assert!(message.contains("zero CubeCL factories returned"));
    assert!(message.contains("cubecl-webgpu"));
    assert!(message.contains("cubecl-vulkan"));
    assert!(message.contains("--features vulkan"));
    assert!(message.contains("ONNX_GENAI_EP=cpu"));
}
