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
    // This smoke test exercises the captured-verify ENGAGE path on a short (6
    // token) repetitive fixture. Disable the adaptive hit-density gate so it
    // engages immediately instead of waiting for the density window to fill;
    // the gate only chooses between two byte-identical paths, so disabling it
    // does not weaken the token-identity assertion below.
    // SAFETY: set before the driver is constructed; the smoke test is invoked
    // single-threaded (guarded behind ONNX_GENAI_RUN_CUDA_SMOKE).
    unsafe {
        std::env::set_var("ONNX_GENAI_SPEC_GATE", "0");
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
    let mut baseline = native_engine(&model_dir, device.clone())?;
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

/// BUG1 regression (#984 re-review, Gaff): the M=1 base-decode graph and the
/// M=width captured-verify graph share the session's single device-graph slot.
/// An engage→miss→re-engage alternation must NEVER let an M=1 decode replay a
/// stale width-W verify graph (the "invalidated graph replay" / illegal-address
/// hazard). This drives the captured-verify path with the adaptive gate ON over
/// a mixed prompt so hit density fluctuates and the driver crosses the
/// engage↔miss boundary repeatedly, asserting (a) no error is returned and
/// (b) the stream stays byte-identical to plain greedy across every transition.
///
/// Enabled with `ONNX_GENAI_RUN_CUDA_SMOKE=1`; requires a real int4 CUDA
/// package. Sized to stay inside the runtime's numerically coherent window (see
/// the note on `native_prompt_lookup_matches_plain_greedy_cuda`).
#[test]
fn native_captured_verify_engage_miss_reengage_no_stale_replay_cuda() -> anyhow::Result<()> {
    if std::env::var_os("ONNX_GENAI_RUN_CUDA_SMOKE").is_none() {
        eprintln!("skipping CUDA smoke; set ONNX_GENAI_RUN_CUDA_SMOKE=1 to run");
        return Ok(());
    }
    // SAFETY: set before any engine/driver construction; the CUDA smoke suite is
    // invoked single-threaded (guarded behind ONNX_GENAI_RUN_CUDA_SMOKE).
    unsafe {
        std::env::set_var("ONNX_GENAI_SPEC_CAPTURED_VERIFY", "1");
        std::env::set_var("ONNX_GENAI_MARLIN_M_GT_1", "1");
        // Keep the adaptive hit-density gate ON (default) with a small window so
        // the engage↔miss boundary is crossed several times over a short run —
        // this is exactly the mode-transition the BUG1 fix guards.
        std::env::remove_var("ONNX_GENAI_SPEC_GATE");
        std::env::set_var("ONNX_GENAI_SPEC_GATE_WINDOW", "4");
        std::env::set_var("ONNX_GENAI_SPEC_GATE_MIN_HITS", "2");
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
    let mut baseline = native_engine(&model_dir, device.clone())?;
    let mut speculative = native_engine(&model_dir, device)?;

    // A repetitive prompt so prompt-lookup reliably ENGAGES (proven by
    // `native_prompt_lookup_matches_plain_greedy_cuda`), with a small gate window
    // so the adaptive gate warms up MISSED (density below threshold) before it
    // ENGAGES once hits cluster — that miss→engage→… boundary is exactly the
    // shared graph-slot transition BUG1 guards. Sized (~16 prompt + 12 generated
    // ≈ position 28) to stay inside the runtime's numerically coherent window.
    let prompt = "The cat sat on the mat. The cat sat on the mat.";
    let request = greedy_request(GeneratePrompt::Text(prompt.to_string()), 12);

    let expected = baseline.generate(request.clone())?;
    // If BUG1 regressed, this call fails with an invalidated-replay /
    // CUDA_ERROR_ILLEGAL_ADDRESS error instead of returning.
    let actual = speculative.generate(with_prompt_lookup(request, 2, 4))?;
    let stats = speculative.last_speculative_stats();

    assert_eq!(
        actual.token_ids, expected.token_ids,
        "captured-verify engage/miss alternation diverged from plain greedy: stats={stats:?}"
    );
    assert_eq!(actual.finish_reason, expected.finish_reason);
    // The gate must have crossed into the engaged (captured verify) mode at least
    // once — otherwise the miss↔engage transition (and its slot invalidation) is
    // never exercised.
    assert!(
        stats.verification_steps > 0,
        "engage/miss regression never engaged the captured verify: {stats:?}"
    );
    eprintln!(
        "engage/miss/re-engage CUDA: proposed={} accepted={} steps={}",
        stats.proposed_tokens, stats.accepted_tokens, stats.verification_steps
    );
    Ok(())
}

/// BUG2 regression (#984 re-review, Chew): the persistent GroupQueryAttention
/// workspace is sized by the query width and prepared once at prefill for
/// `q_seq = prompt_len`. When the prompt is SHORTER than the captured verify
/// width `W`, the width-W verify needs a larger workspace than prefill reserved;
/// before the fix the executor rejected it ("GroupQueryAttention workspace
/// invariant mismatch: requires N, prepared M") and the run crashed. With the
/// fix `run_verify_captured_cuda` reserves the workspace for `q_seq = W` before
/// warming, so a short repetitive prompt must run to completion under both gates
/// ON. The critical assertion is *no crash* over a multi-step run.
///
/// Enabled with `ONNX_GENAI_RUN_CUDA_SMOKE=1`; requires a real int4 CUDA package.
#[test]
fn native_captured_verify_short_prompt_grows_gqa_workspace_cuda() -> anyhow::Result<()> {
    if std::env::var_os("ONNX_GENAI_RUN_CUDA_SMOKE").is_none() {
        eprintln!("skipping CUDA smoke; set ONNX_GENAI_RUN_CUDA_SMOKE=1 to run");
        return Ok(());
    }
    // SAFETY: set before any engine/driver construction; single-threaded suite.
    unsafe {
        std::env::set_var("ONNX_GENAI_SPEC_CAPTURED_VERIFY", "1");
        std::env::set_var("ONNX_GENAI_MARLIN_M_GT_1", "1");
        // Force immediate engage so the width-W verify (and its larger workspace)
        // is exercised from the first steady-state step.
        std::env::set_var("ONNX_GENAI_SPEC_GATE", "0");
    }
    let model_dir = std::env::var_os("ONNX_GENAI_NATIVE_SPEC_QWEN_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from("/home/justinchu/shared-models/qwen2.5-14b-instruct-int4-zp-onnx")
        });
    if !model_dir.join("model.onnx").is_file() {
        eprintln!(
            "skipping CUDA smoke; qwen model is not installed at {}",
            model_dir.display()
        );
        return Ok(());
    }

    let device = Some(NativeDecodeDevice::Cuda { index: Some(0) });
    let mut speculative = native_engine(&model_dir, device)?;

    // A SHORT, highly repetitive prompt (degenerate-repetition class): its token
    // count is below the verify width W = 1 + draft_width, so prefill reserves a
    // smaller GQA workspace than the width-W verify requires. ngram=1 so the
    // repeated token is always proposed and the verify engages every step.
    let prompt = "哈哈哈哈";
    // ≥160 generated tokens: the crash surfaces on the first engaged verify, but
    // a long run also proves the reserved workspace stays valid across replays.
    let request = greedy_request(GeneratePrompt::Text(prompt.to_string()), 160);

    // The assertion IS that this returns Ok — before the fix it errors with the
    // GQA workspace invariant mismatch on the first engaged verify.
    let actual = speculative.generate(with_prompt_lookup(request, 1, 4))?;
    let stats = speculative.last_speculative_stats();
    assert!(
        stats.verification_steps > 0,
        "workspace-growth test never engaged the captured verify: {stats:?}"
    );
    assert!(
        !actual.token_ids.is_empty(),
        "workspace-growth test produced no tokens: {stats:?}"
    );
    eprintln!(
        "short-prompt GQA-workspace CUDA: generated={} verify_steps={} accepted={}",
        actual.token_ids.len(),
        stats.verification_steps,
        stats.accepted_tokens
    );
    Ok(())
}

/// Regression for the qwen width-dependent workspace contract failure found in
/// #988 re-review: pre-reserving the cold generation workspace at W leaked a
/// width-dependent GQA layout into later M=1 fallback decode. W=7 was the
/// smallest failing width and W=9 was the original escaped case. Both must stay
/// byte-identical to plain greedy even on this degenerate prompt.
#[test]
fn native_captured_verify_qwen_w7_w9_match_plain_greedy_cuda() -> anyhow::Result<()> {
    if std::env::var_os("ONNX_GENAI_RUN_CUDA_SMOKE").is_none() {
        eprintln!("skipping CUDA smoke; set ONNX_GENAI_RUN_CUDA_SMOKE=1 to run");
        return Ok(());
    }
    unsafe {
        std::env::set_var("ONNX_GENAI_SPEC_CAPTURED_VERIFY", "1");
        std::env::set_var("ONNX_GENAI_MARLIN_M_GT_1", "1");
        std::env::set_var("ONNX_GENAI_SPEC_GATE", "0");
    }
    let model_dir = std::env::var_os("ONNX_GENAI_NATIVE_SPEC_QWEN_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from("/home/justinchu/shared-models/qwen2.5-14b-instruct-int4-zp-onnx")
        });
    if !model_dir.join("model.onnx").is_file() {
        eprintln!(
            "skipping CUDA smoke; qwen model is not installed at {}",
            model_dir.display()
        );
        return Ok(());
    }

    let prompt = "哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈";
    let device = Some(NativeDecodeDevice::Cuda { index: Some(0) });
    let mut baseline = native_engine(&model_dir, device.clone())?;
    let expected = baseline.generate(greedy_request(
        GeneratePrompt::Text(prompt.to_string()),
        160,
    ))?;

    for spec_tokens in [6, 8] {
        let mut speculative = native_engine(&model_dir, device.clone())?;
        let actual = speculative.generate(with_prompt_lookup(
            greedy_request(GeneratePrompt::Text(prompt.to_string()), 160),
            1,
            spec_tokens,
        ))?;
        let stats = speculative.last_speculative_stats();
        assert_eq!(
            actual.token_ids,
            expected.token_ids,
            "qwen captured-spec W={} diverged from plain greedy: stats={stats:?}",
            spec_tokens + 1
        );
        assert_eq!(actual.finish_reason, expected.finish_reason);
        eprintln!(
            "qwen W={} captured-spec: proposed={} accepted={} steps={} near_tie_rejections={}",
            spec_tokens + 1,
            stats.proposed_tokens,
            stats.accepted_tokens,
            stats.verification_steps,
            stats.near_tie_rejections
        );
    }
    Ok(())
}

#[test]
fn native_captured_verify_qwen_w5_to_w9_accept_path_matches_plain_greedy_cuda() -> anyhow::Result<()>
{
    if std::env::var_os("ONNX_GENAI_RUN_CUDA_SMOKE").is_none() {
        eprintln!("skipping CUDA smoke; set ONNX_GENAI_RUN_CUDA_SMOKE=1 to run");
        return Ok(());
    }
    unsafe {
        std::env::set_var("ONNX_GENAI_SPEC_CAPTURED_VERIFY", "1");
        std::env::set_var("ONNX_GENAI_MARLIN_M_GT_1", "1");
        std::env::set_var("ONNX_GENAI_SPEC_GATE", "0");
    }
    let model_dir = std::env::var_os("ONNX_GENAI_NATIVE_SPEC_QWEN_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from("/home/justinchu/shared-models/qwen2.5-14b-instruct-int4-zp-onnx")
        });
    if !model_dir.join("model.onnx").is_file() {
        eprintln!(
            "skipping CUDA smoke; qwen model is not installed at {}",
            model_dir.display()
        );
        return Ok(());
    }

    let prompts = [
        ("degenerate", "哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈"),
        (
            "normal",
            "The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox",
        ),
    ];
    let device = Some(NativeDecodeDevice::Cuda { index: Some(0) });
    for (label, prompt) in prompts {
        let mut baseline = native_engine(&model_dir, device.clone())?;
        let expected = baseline.generate(greedy_request(
            GeneratePrompt::Text(prompt.to_string()),
            160,
        ))?;
        for spec_tokens in [4, 5, 6, 7, 8] {
            let mut speculative = native_engine(&model_dir, device.clone())?;
            let actual = speculative.generate(with_prompt_lookup(
                greedy_request(GeneratePrompt::Text(prompt.to_string()), 160),
                1,
                spec_tokens,
            ))?;
            let stats = speculative.last_speculative_stats();
            assert!(
                stats.accepted_tokens > 0,
                "qwen {label} W={} did not exercise accept path: {stats:?}",
                spec_tokens + 1
            );
            assert_eq!(
                actual.token_ids,
                expected.token_ids,
                "qwen {label} captured-spec W={} diverged from plain greedy: stats={stats:?}",
                spec_tokens + 1
            );
            eprintln!(
                "qwen {label} W={} captured-spec: proposed={} accepted={} steps={}",
                spec_tokens + 1,
                stats.proposed_tokens,
                stats.accepted_tokens,
                stats.verification_steps
            );
        }
    }
    Ok(())
}

#[test]
fn native_captured_verify_glm_w6_accept_path_matches_plain_greedy_cuda() -> anyhow::Result<()> {
    if std::env::var_os("ONNX_GENAI_RUN_CUDA_SMOKE").is_none() {
        eprintln!("skipping CUDA smoke; set ONNX_GENAI_RUN_CUDA_SMOKE=1 to run");
        return Ok(());
    }
    unsafe {
        std::env::set_var("ONNX_GENAI_SPEC_CAPTURED_VERIFY", "1");
        std::env::set_var("ONNX_GENAI_MARLIN_M_GT_1", "1");
        std::env::set_var("ONNX_GENAI_SPEC_GATE", "0");
    }
    let model_dir = std::env::var_os("ONNX_GENAI_NATIVE_SPEC_GLM_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/home/justinchu/glm-e2e-artifacts/glm-4-9b-int4-cuda"));
    if !model_dir.join("model.onnx").is_file() {
        eprintln!(
            "skipping CUDA smoke; glm model is not installed at {}",
            model_dir.display()
        );
        return Ok(());
    }

    let prompts = [
        (
            "generic",
            "Write a short note about deterministic GPU inference. Deterministic GPU inference needs careful kernels and tests.",
        ),
        (
            "repetitive",
            "red blue green red blue green red blue green red blue green",
        ),
    ];
    let device = Some(NativeDecodeDevice::Cuda { index: Some(0) });
    for (label, prompt) in prompts {
        let mut baseline = native_engine(&model_dir, device.clone())?;
        let expected = baseline.generate(greedy_request(
            GeneratePrompt::Text(prompt.to_string()),
            128,
        ))?;
        let mut speculative = native_engine(&model_dir, device.clone())?;
        let actual = speculative.generate(with_prompt_lookup(
            greedy_request(GeneratePrompt::Text(prompt.to_string()), 128),
            1,
            5,
        ))?;
        let stats = speculative.last_speculative_stats();
        assert!(
            stats.accepted_tokens > 0,
            "glm {label} W=6 did not exercise accept path: {stats:?}"
        );
        assert_eq!(
            actual.token_ids, expected.token_ids,
            "glm {label} captured-spec W=6 diverged from plain greedy: stats={stats:?}"
        );
        eprintln!(
            "glm {label} W=6 captured-spec: proposed={} accepted={} steps={}",
            stats.proposed_tokens, stats.accepted_tokens, stats.verification_steps
        );
    }
    Ok(())
}
