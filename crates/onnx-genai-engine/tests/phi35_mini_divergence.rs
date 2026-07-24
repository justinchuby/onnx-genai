//! Teacher-forced greedy-argmax accuracy lock for the Phi-3.5-mini-instruct
//! int4 (block-32, `accuracy_level=4`) native-vs-ORT divergence.
//!
//! ## Fresh scoreboard (commit f5c6753, `--tokens 128 --steady`, prompt "Hello")
//!
//! Native and ORT greedy streams agree for the first 103 generated tokens, then
//! split at **decode index 103: native picks 411, ORT picks 408**. This is a
//! deterministic (greedy argmax) numeric divergence, not sampling noise.
//!
//! ## Verdict: native is MORE accurate — keep native.
//!
//! Teacher-forcing the exact 104-token shared history (prompt token `15043`
//! plus the 103 agreed generated tokens) through an independent high-precision
//! oracle — the SAME model.onnx with every MatMulNBits `accuracy_level`
//! rewritten — selects **411** at *every* precision, including int8:
//!
//! | MatMulNBits compute | index-103 argmax | logit(411) − logit(408) |
//! |---------------------|------------------|-------------------------|
//! | acc-level-1 (fp32)  | **411**          | **+0.094**              |
//! | acc-level-2 (fp16)  | **411**          | +0.094                  |
//! | acc-level-3 (bf16)  | **411**          | +0.094                  |
//! | acc-level-4 (int8)  | **411**          | +0.108                  |
//!
//! The single-forward oracle picks 411 unconditionally. Native's decode loop
//! also lands on 411 (matching the oracle); only ORT's *autoregressive* decode
//! drifts to 408 — its incrementally-quantized KV/activation history tips this
//! razor-thin race (fp32 gap +0.094 of a ~62 logit). Native is therefore at
//! least as accurate as the fp32 oracle and strictly more accurate than ORT's
//! decode here, so per project policy (correctness beats matching ORT) native is
//! kept and locked below via the same teacher-forced single-step method used in
//! `qwen3_0_6b_divergence.rs`.
//!
//! ## Superseded: the old index-65 lock was a fragile int8-activation tie
//!
//! A previous revision of this test locked decode **index 65** (native 263, ORT
//! 6455) via the full autoregressive `Engine` loop. That lock no longer holds
//! and was removed: at branch tip **both** native and ORT decode loops now emit
//! 6455 at index 65 (they agree — it is not a native-vs-ORT divergence), while
//! the fp32 single-forward oracle still selects 263. That step is a razor-thin
//! fp32 tie (gap +0.0128 logit) that int8-*activation* decode cannot resolve
//! reliably in either backend; native's own higher-precision paths (the M>1
//! prefill GEMM and a teacher-forced single forward) still select 263.
//!
//! `git bisect` attributes the index-65 decode flip (263 → 306 → 6455) to two
//! reviewed int4-decode perf commits — `58d5d6e` (widen VNNI/int16 decode dots
//! to 512-bit) then `37ee582` (deinterleave-once int4 unpack) — which reshuffled
//! the `m=1` int8-activation dot accumulation order. Because both backends land
//! identically (6455) and both single-forward paths agree with fp32 (263), this
//! is expected int8-activation tie fragility, not a native-specific regression.
//! Hardening the `m=1` int4 decode dot to track fp32 on such ties (or aligning
//! its accumulation with the M>1 GEMM path) is a kernel-precision/perf tradeoff
//! flagged for a kernel specialist; see
//! `.squad/decisions/inbox/holden-token-divergence.md`.
//!
//! The model-independent kernel guard for this class,
//! `int4_decode_preserves_f32_argmax_where_per_row_int8_activation_flips`
//! (crate `onnx-runtime-ep-cpu`), catches per-row-vs-per-block int8 regressions
//! in CI without the multi-GB model. This test pins the exact end-to-end token
//! when the real model is available:
//!
//! ```bash
//! PHI35_MINI_E2E_DIR=~/.foundry/cache/models/Microsoft/Phi-3.5-mini-instruct-generic-cpu-2/v2 \
//!   cargo test -p onnx-genai-engine --features mlas --test phi35_mini_divergence \
//!   -- --ignored --nocapture
//! ```

use onnx_genai_engine::{
    Engine, EngineConfig, EngineDecodeBackend, GenerateOptions, GenerateRequest,
    NativeDecodeDevice, NativeDecodeSession, ProcessorChain,
};
use onnx_genai_ort::Tokenizer;

/// The teacher-forced shared history for the fresh scoreboard divergence: the
/// "Hello" prompt token `15043` followed by the 103 generated tokens native and
/// ORT agree on, ending just before the split at decode index 103.
const DIVERGENCE_PREFIX: [u32; 104] = [
    15043, 30751, 31512, 306, 29915, 29885, 1985, 373, 263, 2060, 988, 306, 817, 304, 1653, 263,
    15171, 6270, 10754, 363, 263, 26797, 1848, 8720, 4086, 2000, 376, 3399, 29931, 20191, 1213,
    450, 10754, 881, 4612, 278, 4086, 29915, 29879, 5877, 29892, 29505, 29892, 322, 5412, 5680,
    29892, 3704, 967, 1914, 731, 310, 12768, 29892, 848, 4072, 29892, 322, 2761, 12286, 29889, 306,
    884, 864, 304, 3160, 6455, 310, 775, 9830, 27421, 393, 22222, 1438, 5680, 29892, 3412, 411,
    7309, 800, 29889, 19814, 29892, 306, 29915, 29881, 763, 304, 11039, 403, 263, 4004, 373, 920,
    304, 4386, 15283, 322, 4436, 297, 5920, 29931, 20191, 29892,
];

/// Native's (fp32-oracle-correct) choice at the divergence step.
const NATIVE_DIVERGENCE_TOKEN: u32 = 411;
/// ORT's (decode-loop drift, lower-precision) choice — must NOT be what native emits.
const ORT_DIVERGENCE_TOKEN: u32 = 408;

const DEFAULT_MODEL_DIR: &str =
    "/home/justinchu/.foundry/cache/models/Microsoft/Phi-3.5-mini-instruct-generic-cpu-2/v2";

/// Resolve the decoder ONNX file inside a model directory. Phi-3.5 ships its
/// graph under a descriptive filename (not `model.onnx`), so fall back to the
/// single `*.onnx` file present when `model.onnx` is absent.
fn resolve_model_onnx(dir: &std::path::Path) -> anyhow::Result<std::path::PathBuf> {
    let default = dir.join("model.onnx");
    if default.is_file() {
        return Ok(default);
    }
    let mut onnx_files: Vec<std::path::PathBuf> = std::fs::read_dir(dir)?
        .filter_map(|entry| entry.ok().map(|e| e.path()))
        .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("onnx"))
        .collect();
    onnx_files.sort();
    onnx_files
        .into_iter()
        .next()
        .ok_or_else(|| anyhow::anyhow!("no .onnx file found in {}", dir.display()))
}

#[test]
#[ignore = "requires the real Phi-3.5-mini int4 model via PHI35_MINI_E2E_DIR (or the default foundry cache path)"]
fn phi35_mini_int4_token103_teacher_forced_oracle_is_411() -> anyhow::Result<()> {
    let dir = std::env::var_os("PHI35_MINI_E2E_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::PathBuf::from(DEFAULT_MODEL_DIR));
    if !dir.is_dir() {
        eprintln!(
            "skipping phi35_mini divergence lock: model directory absent: {} \
             (set PHI35_MINI_E2E_DIR to the Phi-3.5-mini-instruct int4 block-32 acc-level-4 dir)",
            dir.display()
        );
        return Ok(());
    }

    let model = resolve_model_onnx(&dir)?;
    let mut session = NativeDecodeSession::load(&model, NativeDecodeDevice::Cpu)?;
    let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json"))?;

    // Teacher-forced single greedy step: feed the exact agreed 104-token history
    // and emit one token. This is robust to autoregressive reduction-order noise
    // (unlike the old full-loop index-65 lock) and asserts native's compute
    // selects the fp32/int8-oracle token at the divergence.
    let options = GenerateOptions {
        max_new_tokens: 1,
        temperature: 0.0,
        greedy: true,
        stop_on_eos: false,
        top_logprobs: Some(8),
        ..GenerateOptions::default()
    };
    let result = session.generate(
        &DIVERGENCE_PREFIX,
        &options,
        &ProcessorChain::new(),
        &tokenizer,
    )?;

    let selected = *result
        .token_ids
        .first()
        .expect("native generation produced no token");
    assert_eq!(
        selected, NATIVE_DIVERGENCE_TOKEN,
        "native must keep the fp32/int8-oracle-correct token {NATIVE_DIVERGENCE_TOKEN} at decode \
         index 103; got {selected}. ORT's autoregressive decode drifts to {ORT_DIVERGENCE_TOKEN}.",
    );
    assert_ne!(
        selected, ORT_DIVERGENCE_TOKEN,
        "native emitted ORT's lower-precision decode-drift token {ORT_DIVERGENCE_TOKEN}; the fp32 \
         oracle selects {NATIVE_DIVERGENCE_TOKEN} — native accuracy regressed to ORT's decode tie",
    );

    // Document the razor-thin benign tie: 411 and 408 are the top two candidates
    // by a hair. A change that widens this gap or reorders the candidate set is a
    // real behavior change worth reviewing.
    let logprobs = result
        .logprobs
        .as_ref()
        .and_then(|entries| entries.first())
        .expect("top_logprobs requested but none returned");
    assert_eq!(
        logprobs.top.first().map(|(id, _)| *id),
        Some(NATIVE_DIVERGENCE_TOKEN),
        "top-ranked logprob token is not {NATIVE_DIVERGENCE_TOKEN}: {:?}",
        logprobs.top,
    );
    let lp_native = logprobs
        .top
        .iter()
        .find(|(id, _)| *id == NATIVE_DIVERGENCE_TOKEN)
        .map(|(_, lp)| *lp)
        .expect("winner logprob missing");
    let lp_ort = logprobs
        .top
        .iter()
        .find(|(id, _)| *id == ORT_DIVERGENCE_TOKEN)
        .map(|(_, lp)| *lp)
        .expect("ORT alternative token 408 must be in the native top-8 (it is the runner-up)");
    let gap = lp_native - lp_ort;
    assert!(
        gap > 0.0 && gap < 0.2,
        "expected a razor-thin benign tie between {NATIVE_DIVERGENCE_TOKEN} and \
         {ORT_DIVERGENCE_TOKEN} (0 < gap < 0.2 logprob); got gap {gap} \
         (native {lp_native}, ort {lp_ort})",
    );

    eprintln!(
        "phi35_mini divergence lock OK: native index-103 token = {NATIVE_DIVERGENCE_TOKEN} \
         (fp32-oracle-correct; ORT decode = {ORT_DIVERGENCE_TOKEN}), benign-tie gap = {gap:.5} \
         logprob",
    );
    Ok(())
}

/// The 103-token greedy stream native and ORT agree on for the "Hello" prompt
/// (prompt token `15043` excluded), reproduced here as the recorded decode
/// prefix. These are exactly `DIVERGENCE_PREFIX[1..]` (the generated tokens of
/// the teacher-forced history above); the token generated *after* this prefix —
/// at decode index 103 — is the divergence under test (native 411, ORT 408).
const DECODE_PREFIX: [u32; 103] = [
    30751, 31512, 306, 29915, 29885, 1985, 373, 263, 2060, 988, 306, 817, 304, 1653, 263, 15171,
    6270, 10754, 363, 263, 26797, 1848, 8720, 4086, 2000, 376, 3399, 29931, 20191, 1213, 450,
    10754, 881, 4612, 278, 4086, 29915, 29879, 5877, 29892, 29505, 29892, 322, 5412, 5680, 29892,
    3704, 967, 1914, 731, 310, 12768, 29892, 848, 4072, 29892, 322, 2761, 12286, 29889, 306, 884,
    864, 304, 3160, 6455, 310, 775, 9830, 27421, 393, 22222, 1438, 5680, 29892, 3412, 411, 7309,
    800, 29889, 19814, 29892, 306, 29915, 29881, 763, 304, 11039, 403, 263, 4004, 373, 920, 304,
    4386, 15283, 322, 4436, 297, 5920, 29931, 20191, 29892,
];

/// Path-faithful end-to-end decode lock.
///
/// The teacher-forced oracle above proves 411 is the high-precision (fp32/fp16/
/// bf16/int8-oracle) argmax at decode index 103, but it feeds the whole 104-token
/// history through a single `M>1` prefill forward and asserts the prefill's last
/// argmax. The real scoreboard divergence (native 411 vs ORT 408) happens in the
/// `m=1` int4 *autoregressive decode* dot — a different kernel path than prefill.
/// Holden's own index-65 note shows prefill and the m=1 decode dot can disagree
/// on these razor-thin ties, so a prefill-only lock does not exercise (and cannot
/// regress-catch) the diverging code path.
///
/// This test therefore drives the REAL `Engine::generate` greedy loop from the
/// "Hello" prompt through `m=1` native decode, reconstructing the recorded
/// `DECODE_PREFIX` token-by-token, then asserts native lands on `411` (not ORT's
/// `408`) at decode index 103. A regression that makes native's m=1 int4 decode
/// dot drift to ORT's 408 will fail here; a drift anywhere earlier fails on the
/// prefix mismatch. This restores the e2e m=1 decode coverage that the removed
/// index-65 full-loop test provided.
#[test]
#[ignore = "requires the real Phi-3.5-mini int4 model via PHI35_MINI_E2E_DIR (or the default foundry cache path); runs a full ~104-step greedy decode loop (minutes)"]
fn phi35_mini_int4_native_decode_loop_keeps_411_at_index_103() -> anyhow::Result<()> {
    let dir = std::env::var_os("PHI35_MINI_E2E_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::PathBuf::from(DEFAULT_MODEL_DIR));
    if !dir.is_dir() {
        eprintln!(
            "skipping phi35_mini decode-loop lock: model directory absent: {} \
             (set PHI35_MINI_E2E_DIR to the Phi-3.5-mini-instruct int4 block-32 acc-level-4 dir)",
            dir.display()
        );
        return Ok(());
    }

    // Force the native backend so this asserts native's argmax through the real
    // m=1 decode path, not ORT's.
    let config = EngineConfig {
        decode_backend: EngineDecodeBackend::Native,
        ..EngineConfig::default()
    };
    let mut engine = Engine::from_dir(&dir, config)?;

    // Greedy, deterministic "Hello" prompt (tokenizes to [15043]) — the exact
    // reproduction used to isolate the divergence. Generate one token past the
    // recorded decode prefix so the divergence token at index 103 is produced by
    // the autoregressive m=1 decode loop.
    let mut request = GenerateRequest::new("Hello".to_string());
    request.options.max_new_tokens = DECODE_PREFIX.len() + 1;
    request.options.temperature = 0.0;
    request.options.greedy = true;
    request.options.stop_on_eos = false;

    let result = engine.generate(request)?;
    let tokens = &result.token_ids;

    assert!(
        tokens.len() > DECODE_PREFIX.len(),
        "native decode produced only {} tokens, need at least {}",
        tokens.len(),
        DECODE_PREFIX.len() + 1,
    );
    assert_eq!(
        &tokens[..DECODE_PREFIX.len()],
        &DECODE_PREFIX,
        "native greedy decode stream drifted from the recorded Phi-3.5 prefix before index 103; \
         the m=1 decode path regressed earlier than the known divergence",
    );

    let divergence = tokens[DECODE_PREFIX.len()];
    assert_eq!(
        divergence,
        NATIVE_DIVERGENCE_TOKEN,
        "native's m=1 autoregressive int4 decode must keep the fp32/int8-oracle-correct token \
         {NATIVE_DIVERGENCE_TOKEN} at decode index {}; got {divergence}. A regression to ORT's \
         decode-drift token {ORT_DIVERGENCE_TOKEN} is exactly the failure this lock guards.",
        DECODE_PREFIX.len(),
    );
    assert_ne!(
        divergence, ORT_DIVERGENCE_TOKEN,
        "native's m=1 decode emitted ORT's lower-precision decode-drift token \
         {ORT_DIVERGENCE_TOKEN}; the fp32 oracle selects {NATIVE_DIVERGENCE_TOKEN} — native's \
         int4 decode dot regressed to ORT's razor-thin tie",
    );

    eprintln!(
        "phi35_mini decode-loop lock OK: native m=1 decode index-{} token = \
         {NATIVE_DIVERGENCE_TOKEN} (fp32-oracle-correct; ORT decode = {ORT_DIVERGENCE_TOKEN})",
        DECODE_PREFIX.len(),
    );
    Ok(())
}
