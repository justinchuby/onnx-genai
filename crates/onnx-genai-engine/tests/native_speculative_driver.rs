//! WP2 exit criterion: the native speculative (prompt-lookup) driver produces a
//! token stream **identical** to native greedy, and engages (accepts > 0) on a
//! repetitive context.
//!
//! Two tiers, mirroring the WP1 convention:
//!   * A hermetic CPU test on the `tiny-native-engine` fixture that always runs
//!     offline (token-identity + acceptance on a highly repetitive stream).
//!   * An env-gated H200 test on the real Qwen2.5-0.5B int4 package, enabled with
//!     `ONNX_GENAI_RUN_CUDA_SMOKE=1` exactly like
//!     `native_decode::tests::native_cuda_verify_rewind_no_kv_corruption`.

#![cfg(feature = "native-backend")]

use onnx_genai_engine::{
    Engine, EngineConfig, EngineDecodeBackend, GeneratePrompt, GenerateRequest, NativeDecodeDevice,
    SpeculativeMode,
};
use std::path::{Path, PathBuf};

fn native_engine(model_dir: &Path, device: Option<NativeDecodeDevice>) -> anyhow::Result<Engine> {
    Engine::from_dir(
        model_dir,
        EngineConfig {
            decode_backend: EngineDecodeBackend::Native,
            native_device: device,
            ..EngineConfig::default()
        },
    )
}

fn greedy_request(prompt: GeneratePrompt, max_new_tokens: usize) -> GenerateRequest {
    let mut request = GenerateRequest::new(prompt);
    request.options.max_new_tokens = max_new_tokens;
    request.options.temperature = 0.0;
    request.options.greedy = true;
    request.options.stop_on_eos = false;
    request
}

fn with_prompt_lookup(
    mut request: GenerateRequest,
    ngram: usize,
    max_tokens: usize,
) -> GenerateRequest {
    request.options.speculative_mode = Some(SpeculativeMode::PromptLookup { ngram, max_tokens });
    request
}

#[test]
fn native_prompt_lookup_matches_plain_greedy_cpu() -> anyhow::Result<()> {
    let fixture =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-native-engine");

    let mut baseline = native_engine(&fixture, None)?;
    let mut speculative = native_engine(&fixture, None)?;

    let request = greedy_request(GeneratePrompt::TokenIds(vec![0]), 8);

    let expected = baseline.generate(request.clone())?;
    let actual = speculative.generate(with_prompt_lookup(request, 1, 4))?;
    let stats = speculative.last_speculative_stats();

    // Exit criterion: byte-identical token stream vs plain native greedy.
    assert_eq!(actual.token_ids, expected.token_ids, "spec stream diverged");
    assert_eq!(actual.finish_reason, expected.finish_reason);
    assert_eq!(actual.text, expected.text);

    // The repetitive stream must engage the loop and accept > 0 tokens.
    assert!(
        stats.proposed_tokens > 0,
        "driver never proposed: {stats:?}"
    );
    assert!(
        stats.accepted_tokens > 0,
        "driver accepted nothing on a repetitive stream: {stats:?}"
    );
    Ok(())
}

#[test]
fn native_prompt_lookup_respects_context_limit_cpu() -> anyhow::Result<()> {
    let fixture =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-native-engine");

    let mut baseline = native_engine(&fixture, None)?;
    let mut speculative = native_engine(&fixture, None)?;

    // A tight context limit must stop the speculative run at the same place as
    // plain greedy, even when a stop lands mid-accepted-run.
    let mut request = greedy_request(GeneratePrompt::TokenIds(vec![0, 0]), 32);
    request.options.max_context = Some(5);

    let expected = baseline.generate(request.clone())?;
    let actual = speculative.generate(with_prompt_lookup(request, 1, 4))?;

    assert_eq!(actual.token_ids, expected.token_ids);
    assert_eq!(actual.finish_reason, expected.finish_reason);
    Ok(())
}

/// Env-gated H200 exit criterion on the real Qwen2.5-0.5B int4 package.
///
/// Enabled with `ONNX_GENAI_RUN_CUDA_SMOKE=1`, mirroring WP1's GPU test
/// (`native_decode::tests::native_cuda_verify_rewind_no_kv_corruption`).
///
/// IMPORTANT — bounded generation length. This model's ONNX Runtime CUDA
/// attention exhibits a *prefill/decode numerical split*: the eager multi-token
/// forward (used by both prefill and `decode_verify`) is bit-identical to the
/// M=1 captured-graph decode only up to a fixed absolute sequence position
/// (~30 tokens on this H200 + package), after which the two kernels' logits
/// diverge by O(1) and can flip a greedy argmax. That divergence is a property
/// of the runtime's attention kernels — it is reproducible with a plain long
/// eager prefill, with no speculation involved — and it is therefore *outside*
/// the WP2 driver and the WP1 `decode_verify` primitive. See
/// `.squad/decisions/inbox/ripley-wp2-native-driver.md` (§"Top risk"). To keep
/// this an exit-criterion identity test rather than a runtime-divergence probe,
/// the prompt + `max_new_tokens` are sized to stay inside the numerically
/// coherent window, where host-argmax acceptance is provably greedy-identical.
/// Strict identity across arbitrarily long generations is blocked on that
/// runtime issue, which must be resolved (or accepted) before native
/// speculation ships for production-length decodes.
#[test]
fn native_prompt_lookup_matches_plain_greedy_cuda() -> anyhow::Result<()> {
    if std::env::var_os("ONNX_GENAI_RUN_CUDA_SMOKE").is_none() {
        eprintln!("skipping CUDA smoke; set ONNX_GENAI_RUN_CUDA_SMOKE=1 to run");
        return Ok(());
    }
    let model_dir = std::env::var_os("ONNX_GENAI_NATIVE_SPEC_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/home/justinchu/qwen2.5-0.5b-int4-onnx"));
    if !model_dir.join("model.onnx").is_file() {
        eprintln!(
            "skipping CUDA smoke; native model is not installed at {}",
            model_dir.display()
        );
        return Ok(());
    }

    let device = Some(NativeDecodeDevice::Cuda { index: Some(0) });
    let mut baseline = native_engine(&model_dir, device)?;
    let mut speculative = native_engine(&model_dir, device)?;

    // A deliberately repetitive prompt so prompt-lookup finds a matching n-gram,
    // sized (~16 prompt tokens + 6 generated ≈ position 22) to stay inside the
    // runtime's numerically coherent window (see the doc comment above).
    let prompt = "The cat sat on the mat. The cat sat on the mat.";
    let request = greedy_request(GeneratePrompt::Text(prompt.to_string()), 6);

    let expected = baseline.generate(request.clone())?;
    let actual = speculative.generate(with_prompt_lookup(request, 2, 4))?;
    let stats = speculative.last_speculative_stats();

    // Exit criterion: byte-identical token stream vs plain native greedy.
    assert_eq!(
        actual.token_ids, expected.token_ids,
        "native prompt-lookup diverged from greedy: stats={stats:?}"
    );
    assert_eq!(actual.finish_reason, expected.finish_reason);
    assert_eq!(actual.text, expected.text);

    // The repetitive prompt must engage the loop and accept > 0 tokens, proving
    // the propose -> verify -> host-argmax-accept -> rewind -> commit path runs
    // end-to-end on CUDA (not a degenerate single-step fallback).
    assert!(
        stats.accepted_tokens > 0,
        "prompt-lookup accepted nothing on a repetitive prompt: {stats:?}"
    );
    assert!(
        stats.verification_steps > 0,
        "driver never verified: {stats:?}"
    );
    eprintln!(
        "native prompt-lookup CUDA: proposed={} accepted={} multi_accepts={} steps={}",
        stats.proposed_tokens,
        stats.accepted_tokens,
        stats.multi_token_accepts,
        stats.verification_steps
    );
    Ok(())
}
