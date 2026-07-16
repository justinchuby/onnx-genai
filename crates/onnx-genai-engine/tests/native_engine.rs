#![cfg(feature = "native-backend")]

use onnx_genai_engine::{
    Engine, EngineConfig, EngineDecodeBackend, GeneratePrompt, GenerateRequest, NativeDecodeDevice,
    SpeculativeMode,
};
use onnx_genai_ort::{ExecutionProvider, SessionOptions};
use std::path::Path;

#[test]
fn engine_generates_through_explicit_native_backend() -> anyhow::Result<()> {
    let fixture =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-native-engine");
    let mut engine = Engine::from_dir(
        &fixture,
        EngineConfig {
            decode_backend: EngineDecodeBackend::Native,
            ..EngineConfig::default()
        },
    )?;
    assert_eq!(engine.decode_backend(), EngineDecodeBackend::Native);

    let mut request = GenerateRequest::new(GeneratePrompt::TokenIds(vec![0]));
    request.options.max_new_tokens = 3;
    request.options.temperature = 0.0;
    request.options.stop_on_eos = false;
    let mut streamed = Vec::new();
    let mut callback = |token: onnx_genai_engine::GenerateToken| -> anyhow::Result<()> {
        streamed.push(token.token_id);
        Ok(())
    };
    let result = engine.generate_with_callback(request, Some(&mut callback))?;

    assert_eq!(result.token_ids, vec![1, 1, 1]);
    assert_eq!(streamed, result.token_ids);
    assert!(engine.create_session().is_err());
    Ok(())
}

#[test]
fn native_backend_rejects_request_level_speculation() -> anyhow::Result<()> {
    let fixture =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-native-engine");
    let mut engine = Engine::from_dir(
        &fixture,
        EngineConfig {
            decode_backend: EngineDecodeBackend::Native,
            ..EngineConfig::default()
        },
    )?;
    let mut request = GenerateRequest::new(GeneratePrompt::TokenIds(vec![0]));
    request.options.speculative_mode = Some(SpeculativeMode::PromptLookup {
        ngram: 2,
        max_tokens: 2,
    });

    let error = engine
        .generate(request)
        .expect_err("native backend must reject request-level speculation");
    assert!(
        error
            .to_string()
            .contains("does not support per-request prompt-lookup speculative decoding")
    );

    let mut request = GenerateRequest::new(GeneratePrompt::TokenIds(vec![0]));
    request.options.num_speculative_tokens = Some(2);
    let error = engine
        .generate(request)
        .expect_err("native backend must reject request-level speculative width");
    assert!(
        error
            .to_string()
            .contains("does not support the per-request num_speculative_tokens option")
    );
    Ok(())
}

#[test]
fn native_backend_rejects_unsupported_session_device() {
    let fixture =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-native-engine");
    let error = Engine::from_dir_with_session_options(
        &fixture,
        EngineConfig {
            decode_backend: EngineDecodeBackend::Native,
            ..EngineConfig::default()
        },
        SessionOptions::with_execution_provider(ExecutionProvider::WebGpu),
    )
    .err()
    .expect("native backend must reject unsupported session options");
    assert!(
        error
            .to_string()
            .contains("does not support execution provider WebGpu")
    );
}

#[cfg(not(feature = "cuda"))]
#[test]
fn native_backend_rejects_cuda_without_cuda_feature() {
    let fixture =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-native-engine");
    let error = Engine::from_dir(
        &fixture,
        EngineConfig {
            decode_backend: EngineDecodeBackend::Native,
            native_device: Some(NativeDecodeDevice::Cuda { index: Some(0) }),
            ..EngineConfig::default()
        },
    )
    .err()
    .expect("native CUDA must require the CUDA feature");
    assert!(error.to_string().contains(
        "requires building onnx-genai-engine with both the 'native-backend' and 'cuda' features"
    ));
}

#[cfg(feature = "cuda")]
#[test]
fn engine_native_cuda_matches_cpu_tokens() -> anyhow::Result<()> {
    if let Err(error) = onnx_runtime_ep_cuda::CudaExecutionProvider::new(0) {
        eprintln!("skipping native engine CUDA parity; CUDA is unavailable: {error}");
        return Ok(());
    }

    let fixture =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-native-cuda-engine");
    let config = |native_device| EngineConfig {
        decode_backend: EngineDecodeBackend::Native,
        native_device: Some(native_device),
        ..EngineConfig::default()
    };
    let generate = |engine: &mut Engine| -> anyhow::Result<Vec<u32>> {
        let mut request = GenerateRequest::new(GeneratePrompt::Text("Hello".to_string()));
        request.options.max_new_tokens = 16;
        request.options.temperature = 0.0;
        request.options.stop_on_eos = false;
        Ok(engine.generate(request)?.token_ids)
    };

    let mut cpu = Engine::from_dir(&fixture, config(NativeDecodeDevice::Cpu))?;
    let mut cuda = Engine::from_dir(
        &fixture,
        config(NativeDecodeDevice::Cuda { index: Some(0) }),
    )?;
    let mut cuda_from_session_options = Engine::from_dir_with_session_options(
        &fixture,
        EngineConfig {
            decode_backend: EngineDecodeBackend::Native,
            ..EngineConfig::default()
        },
        SessionOptions::with_execution_provider(ExecutionProvider::Cuda { device_id: 0 }),
    )?;
    let cpu_tokens = generate(&mut cpu)?;
    let cuda_tokens = generate(&mut cuda)?;
    let session_options_tokens = generate(&mut cuda_from_session_options)?;

    assert_eq!(cpu_tokens.len(), 16);
    assert_eq!(cuda_tokens, cpu_tokens);
    assert_eq!(session_options_tokens, cpu_tokens);
    assert!(cuda_tokens.iter().all(|&token| token == 1));
    Ok(())
}
