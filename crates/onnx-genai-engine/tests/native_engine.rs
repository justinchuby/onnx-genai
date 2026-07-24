#![cfg(feature = "native-backend")]

use onnx_genai_engine::{
    Engine, EngineConfig, EngineDecodeBackend, GeneratePrompt, GenerateRequest, NativeDecodeDevice,
    SpeculativeMode,
};
use onnx_genai_ort::{SessionOptions, ep_selection};
use std::path::Path;
#[cfg(feature = "cuda")]
use std::path::PathBuf;

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
fn native_backend_rejects_unimplemented_speculation_but_allows_prompt_lookup() -> anyhow::Result<()>
{
    let fixture =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-native-engine");
    let mut engine = Engine::from_dir(
        &fixture,
        EngineConfig {
            decode_backend: EngineDecodeBackend::Native,
            ..EngineConfig::default()
        },
    )?;

    // Prompt-lookup is implemented on the native path (WP2): it must NOT be
    // rejected, and it must produce the same greedy stream as the plain path.
    let mut request = GenerateRequest::new(GeneratePrompt::TokenIds(vec![0]));
    request.options.max_new_tokens = 3;
    request.options.temperature = 0.0;
    request.options.stop_on_eos = false;
    request.options.speculative_mode = Some(SpeculativeMode::PromptLookup {
        ngram: 2,
        max_tokens: 2,
    });
    let result = engine.generate(request)?;
    assert_eq!(result.token_ids, vec![1, 1, 1]);

    // Draft-model speculation is not yet ported to native and must be rejected.
    let mut request = GenerateRequest::new(GeneratePrompt::TokenIds(vec![0]));
    request.options.speculative_mode = Some(SpeculativeMode::DraftModel);
    let error = engine
        .generate(request)
        .expect_err("native backend must reject draft-model speculation");
    assert!(
        error
            .to_string()
            .contains("does not yet support per-request draft-model speculative decoding"),
        "unexpected error: {error}"
    );

    // A bare speculative width without a native speculative mode is meaningless.
    let mut request = GenerateRequest::new(GeneratePrompt::TokenIds(vec![0]));
    request.options.num_speculative_tokens = Some(2);
    let error = engine
        .generate(request)
        .expect_err("native backend must reject request-level speculative width");
    assert!(
        error
            .to_string()
            .contains("does not support the per-request num_speculative_tokens option"),
        "unexpected error: {error}"
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
        SessionOptions::with_execution_provider(ep_selection("webgpu")),
    )
    .err()
    .expect("native backend must reject unsupported session options");
    let message = format!("{error:#}");
    assert!(
        message.contains("does not support execution provider")
            && message.contains("WebGpuExecutionProvider"),
        "{message}"
    );
}

#[test]
fn native_sub4_cpu_generates_from_multi_token_prompt() -> anyhow::Result<()> {
    let fixture =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-native-sub4-engine");
    let mut engine = Engine::from_dir(
        &fixture,
        EngineConfig {
            decode_backend: EngineDecodeBackend::Native,
            native_device: Some(NativeDecodeDevice::Cpu),
            ..EngineConfig::default()
        },
    )?;
    let mut request = GenerateRequest::new(GeneratePrompt::TokenIds(vec![0, 0]));
    request.options.max_new_tokens = 3;
    request.options.temperature = 0.0;
    request.options.stop_on_eos = false;

    let result = engine.generate(request)?;
    assert_eq!(result.token_ids, vec![1, 1, 1]);
    Ok(())
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
    let message = format!("{error:#}");
    assert!(
        message.contains(
            "requires building onnx-genai-engine with both the 'native-backend' and 'cuda' features"
        ),
        "{message}"
    );
}

#[cfg(feature = "cuda")]
fn native_cuda_engine(model_dir: &Path) -> anyhow::Result<Engine> {
    Engine::from_dir(
        model_dir,
        EngineConfig {
            decode_backend: EngineDecodeBackend::Native,
            native_device: Some(NativeDecodeDevice::Cuda { index: Some(0) }),
            ..EngineConfig::default()
        },
    )
}

#[cfg(feature = "cuda")]
fn greedy_request(prompt: GeneratePrompt, max_new_tokens: usize) -> GenerateRequest {
    let mut request = GenerateRequest::new(prompt);
    request.options.max_new_tokens = max_new_tokens;
    request.options.temperature = 0.0;
    request.options.greedy = true;
    request.options.stop_on_eos = false;
    request
}

#[cfg(feature = "cuda")]
#[test]
fn native_sub4_cuda_fallback_generates_coherent_decode() -> anyhow::Result<()> {
    if let Err(error) = onnx_runtime_ep_cuda::CudaExecutionProvider::new(0) {
        eprintln!("skipping native sub-4-bit CUDA fallback test; CUDA is unavailable: {error}");
        return Ok(());
    }

    let fixture =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-native-sub4-engine");
    let mut explicit = native_cuda_engine(&fixture)?;
    let mut routed = Engine::from_dir_with_session_options(
        &fixture,
        EngineConfig {
            decode_backend: EngineDecodeBackend::Native,
            ..EngineConfig::default()
        },
        SessionOptions::with_execution_provider(ep_selection("cuda")),
    )?;

    for engine in [&mut explicit, &mut routed] {
        assert_eq!(
            engine
                .generate(greedy_request(GeneratePrompt::TokenIds(vec![0, 0]), 3))?
                .token_ids,
            vec![1, 1, 1]
        );
    }
    Ok(())
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
        Ok(engine
            .generate(greedy_request(
                GeneratePrompt::Text("Hello".to_string()),
                16,
            ))?
            .token_ids)
    };

    let mut cpu = Engine::from_dir(&fixture, config(NativeDecodeDevice::Cpu))?;
    let mut cuda = native_cuda_engine(&fixture)?;
    let mut cuda_from_session_options = Engine::from_dir_with_session_options(
        &fixture,
        EngineConfig {
            decode_backend: EngineDecodeBackend::Native,
            ..EngineConfig::default()
        },
        SessionOptions::with_execution_provider(ep_selection("cuda")),
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

#[cfg(feature = "cuda")]
#[test]
fn qwen15b_native_decode_locks_accurate_near_tie_token() -> anyhow::Result<()> {
    let Some(model_dir) = std::env::var_os("ONNX_GENAI_QWEN15B_CUDA_DIR").map(PathBuf::from) else {
        eprintln!(
            "skipping Qwen2.5-1.5B native CUDA near-tie regression; \
             set ONNX_GENAI_QWEN15B_CUDA_DIR"
        );
        return Ok(());
    };
    if !model_dir.is_dir() {
        eprintln!(
            "skipping Qwen2.5-1.5B native CUDA near-tie regression; model is not installed at {}",
            model_dir.display()
        );
        return Ok(());
    }
    if let Err(error) = onnx_runtime_ep_cuda::CudaExecutionProvider::new(0) {
        eprintln!(
            "skipping Qwen2.5-1.5B native CUDA near-tie regression; CUDA is unavailable: {error}"
        );
        return Ok(());
    }

    let mut engine = native_cuda_engine(&model_dir)?;
    let generated_token_ids = engine
        .generate(greedy_request(
            GeneratePrompt::Text("Hello".to_string()),
            32,
        ))?
        .token_ids;

    // Native FP32 accumulation correctly resolves this <=1-ULP int4 near-tie;
    // ORT CUDA's FP16 accumulation chooses 821 here, so do not match it.
    assert_eq!(generated_token_ids[26], 1909, "{generated_token_ids:?}");
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn engine_native_scalar_gqa_runs_without_metadata_permission() -> anyhow::Result<()> {
    if let Err(error) = onnx_runtime_ep_cuda::CudaExecutionProvider::new(0) {
        eprintln!("skipping native scalar GQA CUDA parity; CUDA is unavailable: {error}");
        return Ok(());
    }

    let fixture =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-native-scalar-gqa");
    let config = |native_device| EngineConfig {
        decode_backend: EngineDecodeBackend::Native,
        native_device: Some(native_device),
        ..EngineConfig::default()
    };
    let generate = |engine: &mut Engine| -> anyhow::Result<Vec<u32>> {
        let mut request = GenerateRequest::new(GeneratePrompt::TokenIds(vec![0]));
        request.options.max_new_tokens = 4;
        request.options.temperature = 0.0;
        request.options.stop_on_eos = false;
        Ok(engine.generate(request)?.token_ids)
    };

    let mut cpu = Engine::from_dir(&fixture, config(NativeDecodeDevice::Cpu))?;
    let mut cuda = Engine::from_dir(
        &fixture,
        config(NativeDecodeDevice::Cuda { index: Some(0) }),
    )?;
    let cpu_tokens = generate(&mut cpu)?;
    let cuda_tokens = generate(&mut cuda)?;
    assert_eq!(cpu_tokens, vec![1, 1, 1, 1]);
    assert_eq!(cuda_tokens, cpu_tokens);

    Ok(())
}
