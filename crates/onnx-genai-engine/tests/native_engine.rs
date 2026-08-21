#![cfg(feature = "native-backend")]

use onnx_genai_engine::{
    Engine, EngineConfig, EngineDecodeBackend, GeneratePrompt, GenerateRequest,
    NATIVE_SESSION_INCREMENTAL_PREFILL_TEST_HITS, NativeDecodeDevice, RewindTokenCount,
    SpeculativeMode,
};
use onnx_genai_ort::{SessionOptions, ep_selection};
use std::path::Path;
#[cfg(feature = "native-cuda")]
use std::path::PathBuf;
use std::sync::atomic::Ordering;

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
    let session_id = engine.create_session()?;
    engine.close_session(session_id)?;
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
            && message.contains("webgpu")
            && message.contains("neither host, CUDA, nor an ORT plugin"),
        "{message}"
    );
}

#[test]
fn native_sub4_cpu_generates_from_multi_token_prompt() -> anyhow::Result<()> {
    let fixture =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-native-engine");
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

#[cfg(not(feature = "native-cuda"))]
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

#[cfg(feature = "native-cuda")]
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

#[cfg(feature = "native-cuda")]
fn greedy_request(prompt: GeneratePrompt, max_new_tokens: usize) -> GenerateRequest {
    let mut request = GenerateRequest::new(prompt);
    request.options.max_new_tokens = max_new_tokens;
    request.options.temperature = 0.0;
    request.options.greedy = true;
    request.options.stop_on_eos = false;
    request
}

#[cfg(feature = "native-cuda")]
#[test]
fn native_sub4_cuda_fallback_generates_coherent_decode() -> anyhow::Result<()> {
    if let Err(error) = onnx_runtime_ep_cuda::CudaExecutionProvider::new(0) {
        eprintln!("skipping native sub-4-bit CUDA fallback test; CUDA is unavailable: {error}");
        return Ok(());
    }

    let fixture =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-native-engine");
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

#[cfg(feature = "native-cuda")]
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

#[cfg(feature = "native-cuda")]
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

#[cfg(feature = "native-cuda")]
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

// ─── Session KV Phase 1 tests ────────────────────────────────────────────────

/// Multi-turn equivalence: incremental-prefill session produces token-identical
/// output vs. stateless full-reset path.
#[test]
fn native_session_incremental_matches_stateless() -> anyhow::Result<()> {
    let fixture =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-native-sub4-engine");
    let mut engine = Engine::from_dir(
        &fixture,
        EngineConfig {
            decode_backend: EngineDecodeBackend::Native,
            ..EngineConfig::default()
        },
    )?;

    // Stateless (full-reset) generation: 3 turns accumulating context.
    let stateless_results = {
        let mut stateless_engine = Engine::from_dir(
            &fixture,
            EngineConfig {
                decode_backend: EngineDecodeBackend::Native,
                ..EngineConfig::default()
            },
        )?;
        let mut results = Vec::new();
        // Simulate multi-turn: each turn adds a new "user token" then generates.
        let mut context = vec![0u32];
        for turn in 0..3 {
            // Add a "new user token" for turns after the first.
            if turn > 0 {
                context.push(0);
            }
            let mut request = GenerateRequest::new(GeneratePrompt::TokenIds(context.clone()));
            request.options.max_new_tokens = 2;
            request.options.temperature = 0.0;
            request.options.stop_on_eos = false;
            let result = stateless_engine.generate(request)?;
            context.extend_from_slice(&result.token_ids);
            results.push(result.token_ids.clone());
        }
        results
    };

    // Session (incremental) generation: same turns through session API.
    let session_id = engine.create_session()?;
    let hits_before = NATIVE_SESSION_INCREMENTAL_PREFILL_TEST_HITS.load(Ordering::Relaxed);
    let mut context = vec![0u32];
    let mut session_results = Vec::new();
    for turn in 0..3 {
        if turn > 0 {
            context.push(0);
        }
        let mut request = GenerateRequest::new(GeneratePrompt::TokenIds(context.clone()));
        request.options.max_new_tokens = 2;
        request.options.temperature = 0.0;
        request.options.stop_on_eos = false;
        let result = engine.generate_in_session(session_id, request)?;
        context.extend_from_slice(&result.token_ids);
        session_results.push(result.token_ids.clone());
    }
    let hits_after = NATIVE_SESSION_INCREMENTAL_PREFILL_TEST_HITS.load(Ordering::Relaxed);

    // Verify token-identical output.
    assert_eq!(session_results, stateless_results);
    // Verify incremental path fired for turns 2 and 3 (turn 1 has no prefix to reuse).
    assert!(
        hits_after - hits_before >= 2,
        "incremental prefill counter did not fire: before={hits_before}, after={hits_after}"
    );
    engine.close_session(session_id)?;
    Ok(())
}

/// Divergence/rewind test: an edited prefix produces the same output as a fresh
/// session with that prefix.
#[test]
fn native_session_rewind_produces_correct_output() -> anyhow::Result<()> {
    let fixture =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-native-sub4-engine");
    let mut engine = Engine::from_dir(
        &fixture,
        EngineConfig {
            decode_backend: EngineDecodeBackend::Native,
            ..EngineConfig::default()
        },
    )?;

    // Build 3 turns.
    let session_id = engine.create_session()?;
    let mut context = vec![0u32];
    for _ in 0..3 {
        let mut request = GenerateRequest::new(GeneratePrompt::TokenIds(context.clone()));
        request.options.max_new_tokens = 2;
        request.options.temperature = 0.0;
        request.options.stop_on_eos = false;
        let result = engine.generate_in_session(session_id, request)?;
        context.extend_from_slice(&result.token_ids);
    }

    // Now "edit history": pass a different prefix that diverges at token 2.
    let edited_context = vec![0u32, 0, 0];
    let mut request = GenerateRequest::new(GeneratePrompt::TokenIds(edited_context.clone()));
    request.options.max_new_tokens = 2;
    request.options.temperature = 0.0;
    request.options.stop_on_eos = false;
    let session_result = engine.generate_in_session(session_id, request)?;

    // Fresh stateless with same edited context must produce identical tokens.
    let mut fresh_engine = Engine::from_dir(
        &fixture,
        EngineConfig {
            decode_backend: EngineDecodeBackend::Native,
            ..EngineConfig::default()
        },
    )?;
    let mut fresh_request = GenerateRequest::new(GeneratePrompt::TokenIds(edited_context));
    fresh_request.options.max_new_tokens = 2;
    fresh_request.options.temperature = 0.0;
    fresh_request.options.stop_on_eos = false;
    let fresh_result = fresh_engine.generate(fresh_request)?;

    assert_eq!(session_result.token_ids, fresh_result.token_ids);
    engine.close_session(session_id)?;
    Ok(())
}

#[test]
fn native_session_switching_matches_cold_start() -> anyhow::Result<()> {
    let fixture =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-native-sub4-engine");
    let mut engine = Engine::from_dir(
        &fixture,
        EngineConfig {
            decode_backend: EngineDecodeBackend::Native,
            ..EngineConfig::default()
        },
    )?;
    let session_a = engine.create_session()?;
    let session_b = engine.create_session()?;

    let mut context_a = vec![0u32];
    let mut context_b = vec![0u32, 0];
    for (session, context) in [(session_a, &mut context_a), (session_b, &mut context_b)] {
        let mut request = GenerateRequest::new(GeneratePrompt::TokenIds(context.clone()));
        request.options.max_new_tokens = 2;
        request.options.temperature = 0.0;
        request.options.stop_on_eos = false;
        let result = engine.generate_in_session(session, request)?;
        context.extend_from_slice(&result.token_ids);
    }

    context_a.push(0);
    let mut request = GenerateRequest::new(GeneratePrompt::TokenIds(context_a.clone()));
    request.options.max_new_tokens = 2;
    request.options.temperature = 0.0;
    request.options.stop_on_eos = false;
    let switched_result = engine.generate_in_session(session_a, request)?;

    let mut fresh = Engine::from_dir(
        &fixture,
        EngineConfig {
            decode_backend: EngineDecodeBackend::Native,
            ..EngineConfig::default()
        },
    )?;
    let mut cold_request = GenerateRequest::new(GeneratePrompt::TokenIds(context_a));
    cold_request.options.max_new_tokens = 2;
    cold_request.options.temperature = 0.0;
    cold_request.options.stop_on_eos = false;
    cold_request.options.cold_start = true;
    let cold_result = fresh.generate(cold_request)?;
    assert_eq!(switched_result.token_ids, cold_result.token_ids);
    Ok(())
}

/// Native sessions use the unified session API and allow multiple logical sessions.
/// A valid explicit rewind through the shared `session_state` policy truncates
/// the native session's logical length. This drives the same shared bound check
/// that the ORT `failed_rewind_of_*` tests reach, so inverting the check in
/// `session_state::rewind_to` turns *both* backends red — the proof that the two
/// arms genuinely share one policy rather than two look-alike copies.
#[test]
fn native_session_rewind_by_truncates_logical_length() -> anyhow::Result<()> {
    let fixture =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-native-sub4-engine");
    let mut engine = Engine::from_dir(
        &fixture,
        EngineConfig {
            decode_backend: EngineDecodeBackend::Native,
            ..EngineConfig::default()
        },
    )?;

    let session = engine.create_session()?;
    let mut request = GenerateRequest::new(GeneratePrompt::TokenIds(vec![0u32, 0, 0]));
    request.options.max_new_tokens = 2;
    request.options.temperature = 0.0;
    request.options.stop_on_eos = false;
    engine.generate_in_session(session, request)?;

    let before = engine.session_token_count(session)?;
    assert!(
        before >= 2,
        "expected at least two logical tokens before rewind, got {before}"
    );

    let new_position = engine.rewind_session_by(session, RewindTokenCount::new(2))?;
    assert_eq!(new_position.get(), before - 2);
    assert_eq!(engine.session_token_count(session)?, before - 2);

    // Rewinding past the start is rejected by the same shared bound check.
    let error = engine
        .rewind_session_by(session, RewindTokenCount::new(before))
        .expect_err("rewinding past the start must fail");
    assert!(
        error.to_string().contains("cannot rewind session"),
        "unexpected error: {error:#}"
    );

    engine.close_session(session)?;
    Ok(())
}

#[test]
fn native_session_creation_uses_unified_multi_session_api() -> anyhow::Result<()> {
    let fixture =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-native-sub4-engine");
    let mut engine = Engine::from_dir(
        &fixture,
        EngineConfig {
            decode_backend: EngineDecodeBackend::Native,
            ..EngineConfig::default()
        },
    )?;

    let first = engine.create_session()?;
    let second = engine.create_session()?;
    assert_ne!(first, second);
    assert_eq!(engine.session_token_count(first)?, 0);
    assert_eq!(engine.session_token_count(second)?, 0);
    engine.close_session(first)?;
    engine.close_session(second)?;
    Ok(())
}

#[test]
fn native_stateless_generate_reuses_default_session_by_default() -> anyhow::Result<()> {
    let fixture =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-native-sub4-engine");
    let mut reused = Engine::from_dir(
        &fixture,
        EngineConfig {
            decode_backend: EngineDecodeBackend::Native,
            ..EngineConfig::default()
        },
    )?;
    let mut cold = Engine::from_dir(
        &fixture,
        EngineConfig {
            decode_backend: EngineDecodeBackend::Native,
            ..EngineConfig::default()
        },
    )?;

    let hits_before = NATIVE_SESSION_INCREMENTAL_PREFILL_TEST_HITS.load(Ordering::Relaxed);
    let mut reused_context = vec![0u32];
    let mut cold_context = vec![0u32];
    for turn in 0..3 {
        if turn > 0 {
            reused_context.push(0);
            cold_context.push(0);
        }
        let mut reused_request =
            GenerateRequest::new(GeneratePrompt::TokenIds(reused_context.clone()));
        reused_request.options.max_new_tokens = 2;
        reused_request.options.temperature = 0.0;
        reused_request.options.stop_on_eos = false;
        let reused_result = reused.generate(reused_request)?;

        let mut cold_request = GenerateRequest::new(GeneratePrompt::TokenIds(cold_context.clone()));
        cold_request.options.max_new_tokens = 2;
        cold_request.options.temperature = 0.0;
        cold_request.options.stop_on_eos = false;
        cold_request.options.cold_start = true;
        let cold_result = cold.generate(cold_request)?;

        assert_eq!(reused_result.token_ids, cold_result.token_ids);
        reused_context.extend_from_slice(&reused_result.token_ids);
        cold_context.extend_from_slice(&cold_result.token_ids);
    }
    let hits_after = NATIVE_SESSION_INCREMENTAL_PREFILL_TEST_HITS.load(Ordering::Relaxed);
    assert!(
        hits_after - hits_before >= 2,
        "default session did not reuse KV: before={hits_before}, after={hits_after}"
    );
    Ok(())
}

/// Reuse must not leak state between unrelated prompts.
///
/// The growing-context test only exercises the case where each prompt extends
/// the last, which is the case reuse is designed for. The risk of making reuse
/// the default is the opposite case: two consecutive `generate()` calls that
/// share no prefix must behave exactly as if each ran on a fresh engine, or
/// stateless callers silently get output conditioned on a previous request.
#[test]
fn stateless_generate_with_unrelated_prompts_matches_a_cold_engine() -> anyhow::Result<()> {
    let fixture =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-native-sub4-engine");
    let native_config = || EngineConfig {
        decode_backend: EngineDecodeBackend::Native,
        ..EngineConfig::default()
    };
    let mut reusing = Engine::from_dir(&fixture, native_config())?;

    // Deliberately divergent: no shared prefix beyond nothing at all.
    let prompts = vec![
        vec![0u32, 0, 0],
        vec![1u32],
        vec![0u32, 1],
        vec![1u32, 1, 1],
    ];
    for prompt in prompts {
        let mut request = GenerateRequest::new(GeneratePrompt::TokenIds(prompt.clone()));
        request.options.max_new_tokens = 2;
        request.options.temperature = 0.0;
        request.options.stop_on_eos = false;
        let reused_result = reusing.generate(request)?;

        // A brand-new engine is the ground truth for "no carried state".
        let mut fresh = Engine::from_dir(&fixture, native_config())?;
        let mut cold_request = GenerateRequest::new(GeneratePrompt::TokenIds(prompt.clone()));
        cold_request.options.max_new_tokens = 2;
        cold_request.options.temperature = 0.0;
        cold_request.options.stop_on_eos = false;
        cold_request.options.cold_start = true;
        let cold_result = fresh.generate(cold_request)?;

        assert_eq!(
            reused_result.token_ids, cold_result.token_ids,
            "default-on reuse changed the output for prompt {prompt:?}"
        );
    }
    Ok(())
}

#[test]
fn native_session_lru_eviction_keeps_remaining_session_correct() -> anyhow::Result<()> {
    let fixture =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-native-sub4-engine");
    let mut engine = Engine::from_dir(
        &fixture,
        EngineConfig {
            decode_backend: EngineDecodeBackend::Native,
            native_max_sessions: 1,
            ..EngineConfig::default()
        },
    )?;
    let evicted = engine.create_session()?;
    let retained = engine.create_session()?;
    assert!(engine.session_token_count(evicted).is_err());

    let prompt = vec![0u32, 0];
    let mut request = GenerateRequest::new(GeneratePrompt::TokenIds(prompt.clone()));
    request.options.max_new_tokens = 2;
    request.options.temperature = 0.0;
    request.options.stop_on_eos = false;
    let retained_result = engine.generate_in_session(retained, request)?;

    let mut fresh = Engine::from_dir(
        &fixture,
        EngineConfig {
            decode_backend: EngineDecodeBackend::Native,
            ..EngineConfig::default()
        },
    )?;
    let mut cold_request = GenerateRequest::new(GeneratePrompt::TokenIds(prompt));
    cold_request.options.max_new_tokens = 2;
    cold_request.options.temperature = 0.0;
    cold_request.options.stop_on_eos = false;
    cold_request.options.cold_start = true;
    let cold_result = fresh.generate(cold_request)?;
    assert_eq!(retained_result.token_ids, cold_result.token_ids);
    Ok(())
}

/// End-to-end proof that a CPU KV cache which is *reallocated* mid-decode is
/// still aliased present==past afterwards.
///
/// This is the property the whole in-place path rests on and the one the unit
/// tests cannot reach: `GroupQueryAttention` decides to append in place by
/// comparing the past-input and present-output pointers at execution time, so
/// a replacement buffer that failed to re-alias would silently stop appending.
///
/// Token equality alone is not evidence here — this fixture's q/k/v are
/// `Constant` zeros, so its output does not depend on KV contents at all and
/// would stay `[1, 1, 1, 1]` even if growth dropped the entire history
/// (verified by deliberately breaking the copy). The load-bearing assertion is
/// therefore the in-place append *counter*: a cache forced to grow repeatedly
/// must take the in-place path exactly as often as one that never grows. If
/// aliasing were lost at the first realloc, the post-growth steps would fall
/// through to the copy path and the counts would differ.
#[test]
fn a_regrown_cpu_kv_cache_is_still_appended_in_place() -> anyhow::Result<()> {
    let fixture =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-native-scalar-gqa");

    let generate = |max_len: &str| -> anyhow::Result<(Vec<u32>, usize)> {
        // SAFETY: both arms set the variable explicitly and it is removed at
        // the end, so no unset-vs-set race exists between them.
        unsafe { std::env::set_var("ONNX_GENAI_CPU_KV_MAX_LEN", max_len) };
        let before = onnx_runtime_ep_cpu::present_inplace_count();
        let mut engine = Engine::from_dir(
            &fixture,
            EngineConfig {
                decode_backend: EngineDecodeBackend::Native,
                native_device: Some(NativeDecodeDevice::Cpu),
                ..EngineConfig::default()
            },
        )?;
        let mut request = GenerateRequest::new(GeneratePrompt::TokenIds(vec![0]));
        request.options.max_new_tokens = 4;
        request.options.temperature = 0.0;
        request.options.stop_on_eos = false;
        let tokens = engine.generate(request)?.token_ids;
        Ok((
            tokens,
            onnx_runtime_ep_cpu::present_inplace_count() - before,
        ))
    };

    // Capacity 2 cannot hold the 1-token prompt plus 4 generated tokens, so the
    // cache must grow mid-decode; 4096 never grows at all.
    let (grown_tokens, grown_appends) = generate("2")?;
    let (roomy_tokens, roomy_appends) = generate("4096")?;
    unsafe { std::env::remove_var("ONNX_GENAI_CPU_KV_MAX_LEN") };

    assert!(
        roomy_appends > 0,
        "the fixture must reach the in-place append path at all, else this \
         test proves nothing about aliasing"
    );
    assert_eq!(
        grown_appends, roomy_appends,
        "a cache that grew mid-decode must keep appending in place; losing the \
         present==past aliasing at realloc would drop these counts"
    );
    assert_eq!(
        grown_tokens, roomy_tokens,
        "growth must not change the decode"
    );
    assert_eq!(roomy_tokens, vec![1, 1, 1, 1]);
    Ok(())
}
