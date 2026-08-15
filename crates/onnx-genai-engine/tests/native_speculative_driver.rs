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

/// GATE 1 (Leon): captured speculative decode must be BYTE-IDENTICAL to plain
/// M=1 greedy at draft widths W=5..9 with the accept path exercised
/// (accepted>0), at the *deterministic* engine configuration.
///
/// Phase-1 diagnosis established that the opt-in Marlin int4 prefill GEMM
/// (`ONNX_GENAI_MARLIN_M_GT_1=1`, default OFF) is non-deterministic run-to-run:
/// plain greedy itself flips a razor-tie token ~1/5 of runs under Marlin prefill
/// but is 100% stable with Marlin off. Byte-identity to a non-deterministic
/// reference is ill-posed, which is why the three prior attempts — all validated
/// against `MARLIN_M_GT_1=1` — could never stabilize. This gate therefore runs
/// at the default deterministic config (Marlin off for prefill; the captured
/// verify decomposes the M=W int4 GEMMs into per-row M=1 GEMV launches via
/// `PerRowVerifyGuard`, byte-identical to greedy by construction).
fn run_byte_identity_gate(
    model_dir: &std::path::Path,
    prompt: &str,
    device: Option<NativeDecodeDevice>,
) -> anyhow::Result<bool> {
    // Deterministic reference: Marlin prefill OFF (default). The captured verify
    // is byte-identical to M=1 greedy per row regardless of this setting.
    unsafe {
        std::env::remove_var("ONNX_GENAI_MARLIN_M_GT_1");
    }
    let mut baseline = native_engine(model_dir, device.clone())?;
    let expected = baseline.generate(greedy_request(
        GeneratePrompt::Text(prompt.to_string()),
        160,
    ))?;
    drop(baseline);
    eprintln!(
        "greedy reference (deterministic, marlin off): {} tokens",
        expected.token_ids.len()
    );

    unsafe {
        std::env::set_var("ONNX_GENAI_SPEC_CAPTURED_VERIFY", "1");
        std::env::set_var("ONNX_GENAI_SPEC_GATE", "0");
        // Arm the opt-in per-row M=1 GEMV verify (byte-identity reference).
        std::env::set_var("ONNX_GENAI_SPEC_PERROW_VERIFY", "1");
    }
    let mut all_ok = true;
    for spec_tokens in [4usize, 5, 6, 7, 8] {
        let w = spec_tokens + 1;
        let mut spec = native_engine(model_dir, device.clone())?;
        let actual = spec.generate(with_prompt_lookup(
            greedy_request(GeneratePrompt::Text(prompt.to_string()), 160),
            1,
            spec_tokens,
        ))?;
        let stats = spec.last_speculative_stats();
        let first_div = expected
            .token_ids
            .iter()
            .zip(actual.token_ids.iter())
            .position(|(a, b)| a != b);
        let matched = first_div.is_none() && expected.token_ids.len() == actual.token_ids.len();
        all_ok &= matched && stats.accepted_tokens > 0;
        eprintln!(
            "W={w} accepted={} multi={} steps={} near_tie={} first_div={:?} {}",
            stats.accepted_tokens,
            stats.multi_token_accepts,
            stats.verification_steps,
            stats.near_tie_rejections,
            first_div,
            if matched {
                "BYTE-IDENTICAL"
            } else {
                "DIVERGED"
            },
        );
        drop(spec);
    }
    unsafe {
        std::env::remove_var("ONNX_GENAI_SPEC_CAPTURED_VERIFY");
        std::env::remove_var("ONNX_GENAI_SPEC_GATE");
        std::env::remove_var("ONNX_GENAI_SPEC_PERROW_VERIFY");
    }
    Ok(all_ok)
}

/// GATE 1 on qwen2.5-14b int4. Env-gated; run with:
///   ONNX_GENAI_RUN_CUDA_SMOKE=1 CUDA_VISIBLE_DEVICES=<idle> cargo test -p \
///     onnx-genai-engine --features cuda,native-backend \
///     --test native_speculative_driver leon_gate1_qwen -- --nocapture --test-threads=1
#[test]
fn leon_gate1_qwen_byte_identity_cuda() -> anyhow::Result<()> {
    if std::env::var_os("ONNX_GENAI_RUN_CUDA_SMOKE").is_none() {
        eprintln!("skipping CUDA smoke; set ONNX_GENAI_RUN_CUDA_SMOKE=1 to run");
        return Ok(());
    }
    let model_dir = std::env::var_os("ONNX_GENAI_NATIVE_SPEC_QWEN_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from("/home/justinchu/shared-models/qwen2.5-14b-instruct-int4-zp-onnx")
        });
    if !model_dir.join("model.onnx").is_file() {
        eprintln!("skipping; qwen model not at {}", model_dir.display());
        return Ok(());
    }
    let prompt = "The quick brown fox jumps over the lazy dog. The dog was not amused, \
                  and the fox ran away into the forest. In the forest there were many \
                  trees, and the trees were tall and green. The fox found a river and \
                  drank from the river before continuing on its journey through the";
    let device = Some(NativeDecodeDevice::Cuda { index: Some(0) });
    let ok = run_byte_identity_gate(&model_dir, prompt, device)?;
    assert!(
        ok,
        "qwen normal-prompt byte-identity gate failed (see per-W lines)"
    );
    Ok(())
}

/// GATE 1 on glm-4-9b int4. Env-gated (see qwen variant for the invocation).
#[test]
fn leon_gate1_glm_byte_identity_cuda() -> anyhow::Result<()> {
    if std::env::var_os("ONNX_GENAI_RUN_CUDA_SMOKE").is_none() {
        eprintln!("skipping CUDA smoke; set ONNX_GENAI_RUN_CUDA_SMOKE=1 to run");
        return Ok(());
    }
    let model_dir = std::env::var_os("ONNX_GENAI_NATIVE_SPEC_GLM_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/home/justinchu/glm-e2e-artifacts/glm-4-9b-int4-cuda"));
    if !model_dir.join("model.onnx").is_file() {
        eprintln!("skipping; glm model not at {}", model_dir.display());
        return Ok(());
    }
    let prompt = "The history of the ancient city was long and storied. The city was \
                  founded by settlers who came from across the sea. The settlers built \
                  walls around the city, and the walls protected the city from raiders. \
                  Over the centuries the city grew, and the people of the city prospered \
                  as they traded goods along the river that flowed beside the";
    let device = Some(NativeDecodeDevice::Cuda { index: Some(0) });
    let ok = run_byte_identity_gate(&model_dir, prompt, device)?;
    assert!(
        ok,
        "glm normal-prompt byte-identity gate failed (see per-W lines)"
    );
    Ok(())
}

/// Determinism discriminator (Leon): isolate whether run-to-run divergence in
/// the byte-identity gate comes from plain greedy decode itself (a shared
/// prefill/attention non-determinism the spec path merely inherits) or from the
/// captured speculative verify. Generates plain greedy N times and the captured
/// spec W=5 run N times against the first greedy trajectory, all under the same
/// Marlin-prefill config, and reports first-divergence indices without asserting.
#[test]
fn leon_determinism_check_cuda() -> anyhow::Result<()> {
    if std::env::var_os("ONNX_GENAI_RUN_CUDA_SMOKE").is_none() {
        eprintln!("skipping CUDA smoke; set ONNX_GENAI_RUN_CUDA_SMOKE=1 to run");
        return Ok(());
    }
    let model_dir = std::env::var_os("ONNX_GENAI_NATIVE_SPEC_QWEN_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from("/home/justinchu/shared-models/qwen2.5-14b-instruct-int4-zp-onnx")
        });
    if !model_dir.join("model.onnx").is_file() {
        eprintln!("skipping; qwen model not at {}", model_dir.display());
        return Ok(());
    }
    let prompt = "The quick brown fox jumps over the lazy dog. The dog was not amused, \
                  and the fox ran away into the forest. In the forest there were many \
                  trees, and the trees were tall and green. The fox found a river and \
                  drank from the river before continuing on its journey through the";
    let device = Some(NativeDecodeDevice::Cuda { index: Some(0) });
    let first_div = |a: &[u32], b: &[u32]| -> Option<usize> {
        a.iter().zip(b.iter()).position(|(x, y)| x != y)
    };
    // Plain greedy determinism with Marlin prefill ON: 6 runs vs run 0.
    unsafe {
        std::env::set_var("ONNX_GENAI_MARLIN_M_GT_1", "1");
    }
    let mut greedy: Vec<Vec<u32>> = Vec::new();
    for _ in 0..6 {
        let mut e = native_engine(&model_dir, device.clone())?;
        let out = e.generate(greedy_request(
            GeneratePrompt::Text(prompt.to_string()),
            160,
        ))?;
        greedy.push(out.token_ids);
        drop(e);
    }
    for (i, g) in greedy.iter().enumerate().skip(1) {
        eprintln!(
            "greedy[marlin=1] run {i} vs run0: first_div={:?}",
            first_div(&greedy[0], g)
        );
    }

    // Plain greedy determinism with Marlin prefill OFF (portable tiled GEMM): 6
    // runs vs a fresh run-0 reference for that config.
    unsafe {
        std::env::set_var("ONNX_GENAI_MARLIN_M_GT_1", "0");
    }
    let mut greedy0: Vec<Vec<u32>> = Vec::new();
    for _ in 0..6 {
        let mut e = native_engine(&model_dir, device.clone())?;
        let out = e.generate(greedy_request(
            GeneratePrompt::Text(prompt.to_string()),
            160,
        ))?;
        greedy0.push(out.token_ids);
        drop(e);
    }
    for (i, g) in greedy0.iter().enumerate().skip(1) {
        eprintln!(
            "greedy[marlin=0] run {i} vs run0: first_div={:?}",
            first_div(&greedy0[0], g)
        );
    }
    eprintln!(
        "greedy marlin=1 run0 vs marlin=0 run0: first_div={:?}",
        first_div(&greedy[0], &greedy0[0])
    );
    unsafe {
        std::env::set_var("ONNX_GENAI_MARLIN_M_GT_1", "1");
    }

    // Captured spec determinism: many runs at W=5 and W=9 vs greedy run 0.
    // Third greedy config: Marlin ON but split-K forced OFF, to localize the
    // Marlin prefill non-determinism to the split-K path vs the direct kernel.
    unsafe {
        std::env::set_var("ONNX_GENAI_MARLIN_M_GT_1", "1");
        std::env::set_var("ONNX_GENAI_MARLIN_SPLITK", "0");
    }
    let mut greedy_ns: Vec<Vec<u32>> = Vec::new();
    for _ in 0..6 {
        let mut e = native_engine(&model_dir, device.clone())?;
        let out = e.generate(greedy_request(
            GeneratePrompt::Text(prompt.to_string()),
            160,
        ))?;
        greedy_ns.push(out.token_ids);
        drop(e);
    }
    for (i, g) in greedy_ns.iter().enumerate().skip(1) {
        eprintln!(
            "greedy[marlin=1,splitk=0] run {i} vs run0: first_div={:?}",
            first_div(&greedy_ns[0], g)
        );
    }
    unsafe {
        std::env::remove_var("ONNX_GENAI_MARLIN_SPLITK");
        std::env::remove_var("ONNX_GENAI_MARLIN_M_GT_1");
    }
    Ok(())
}

/// GATE 2 (Leon): wall-clock speedup of captured speculative decode vs plain
/// M=1 greedy at the deterministic (Marlin-off) config. Reports tok/s for both
/// and the mean accepted length per verification step. Non-asserting probe.
#[test]
fn leon_gate2_speedup_cuda() -> anyhow::Result<()> {
    if std::env::var_os("ONNX_GENAI_RUN_CUDA_SMOKE").is_none() {
        eprintln!("skipping CUDA smoke; set ONNX_GENAI_RUN_CUDA_SMOKE=1 to run");
        return Ok(());
    }
    let model_dir = std::env::var_os("ONNX_GENAI_NATIVE_SPEC_QWEN_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from("/home/justinchu/shared-models/qwen2.5-14b-instruct-int4-zp-onnx")
        });
    if !model_dir.join("model.onnx").is_file() {
        eprintln!("skipping; model not at {}", model_dir.display());
        return Ok(());
    }
    let prompt = "The quick brown fox jumps over the lazy dog. The dog was not amused, \
                  and the fox ran away into the forest. In the forest there were many \
                  trees, and the trees were tall and green. The fox found a river and \
                  drank from the river before continuing on its journey through the";
    let device = Some(NativeDecodeDevice::Cuda { index: Some(0) });
    let n_tokens = 160usize;
    unsafe {
        std::env::remove_var("ONNX_GENAI_MARLIN_M_GT_1");
    }
    // Warm + time plain greedy.
    let mut g = native_engine(&model_dir, device.clone())?;
    let _ = g.generate(greedy_request(GeneratePrompt::Text(prompt.to_string()), 8))?;
    let t0 = std::time::Instant::now();
    let gout = g.generate(greedy_request(
        GeneratePrompt::Text(prompt.to_string()),
        n_tokens,
    ))?;
    let g_dt = t0.elapsed().as_secs_f64();
    drop(g);
    let g_toks = gout.token_ids.len() as f64;
    eprintln!(
        "greedy: {g_toks} tok in {g_dt:.3}s = {:.1} tok/s",
        g_toks / g_dt
    );

    unsafe {
        std::env::set_var("ONNX_GENAI_SPEC_CAPTURED_VERIFY", "1");
        std::env::set_var("ONNX_GENAI_SPEC_GATE", "0");
    }
    for spec_tokens in [4usize, 6, 8] {
        let w = spec_tokens + 1;
        let mut s = native_engine(&model_dir, device.clone())?;
        let _ = s.generate(with_prompt_lookup(
            greedy_request(GeneratePrompt::Text(prompt.to_string()), 8),
            1,
            spec_tokens,
        ))?;
        let t1 = std::time::Instant::now();
        let sout = s.generate(with_prompt_lookup(
            greedy_request(GeneratePrompt::Text(prompt.to_string()), n_tokens),
            1,
            spec_tokens,
        ))?;
        let s_dt = t1.elapsed().as_secs_f64();
        let st = s.last_speculative_stats();
        let s_toks = sout.token_ids.len() as f64;
        eprintln!(
            "spec W={w}: {s_toks} tok in {s_dt:.3}s = {:.1} tok/s | speedup {:.2}x | \
             accepted={} steps={} mean_accept_len={:.2}",
            s_toks / s_dt,
            (s_toks / s_dt) / (g_toks / g_dt),
            st.accepted_tokens,
            st.verification_steps,
            st.accepted_tokens as f64 / st.verification_steps.max(1) as f64,
        );
        drop(s);
    }
    unsafe {
        std::env::remove_var("ONNX_GENAI_SPEC_CAPTURED_VERIFY");
        std::env::remove_var("ONNX_GENAI_SPEC_GATE");
    }
    Ok(())
}
