//! Greedy-decode accuracy lock for the qwen3-0.6b int4/int8 (block-128,
//! `accuracy_level=4`) native-vs-ORT divergence.
//!
//! ## Reproduction
//!
//! Greedy generation from the prompt *"Write a short story about a brave knight
//! who explores an ancient forest."* (raw tokens
//! `[7985, 264, 2805, 3364, 911, 264, 33200, 46609, 879, 40324, 458, 13833,
//! 13638, 13]`) agrees between native and ORT for the first four generated
//! tokens `[576, 3364, 1265, 2924]`, then splits: **native picks token 518, ORT
//! picks token 264**. This is a deterministic (greedy argmax) numeric
//! divergence, not sampling noise, and matches the ~token-20 class the user
//! reported (prompt-dependent split index).
//!
//! ## Verdict: native is (at least) as accurate — KEEP native.
//!
//! An independent high-precision oracle was built by running the *same*
//! `model.onnx` through ONNX Runtime with every `MatMulNBits` `accuracy_level`
//! rewritten to `1` (fp32 compute, no int8 *activation* quantization). At the
//! divergence step the oracle selects **518**:
//!
//! | MatMulNBits compute | argmax | logit(518) − logit(264) |
//! |---------------------|--------|-------------------------|
//! | acc-level-1 (fp32)  | **518**| **+0.0438**             |
//! | native (acc-4 int8) | **518**| +0.0516                 |
//! | ORT   (acc-4 int8)  | 264    | −0.0527                 |
//!
//! Across 30 teacher-forced positions the native argmax matches the fp32 oracle
//! in **29/30** positions; ORT matches in **28/30**. Every disagreement is a
//! razor-thin tie (oracle top1−top2 gap ≤ 0.044 logits; one is 0.0011). ONLY
//! int8 *activation* quantization (`accuracy_level=4`, both backends' default
//! here) can tip these near-ties, and native's per-BLOCK int8 activation scale
//! tracks the fp32 oracle more faithfully than ORT's looser per-ROW scale. So
//! native is not less accurate than ORT — it is marginally more accurate — and
//! per project policy (correctness beats matching ORT) native is KEPT.
//!
//! This is the same failure class locked for Phi-3.5-mini in
//! `phi35_mini_divergence.rs`, whose model-independent kernel guard
//! `int4_decode_preserves_f32_argmax_where_per_row_int8_activation_flips`
//! (crate `onnx-runtime-ep-cpu`) catches the regression class in CI without a
//! multi-GB model. This test additionally pins qwen3-0.6b's exact divergence
//! step against the real model when it is available:
//!
//! ```bash
//! QWEN3_0_6B_E2E_DIR=~/.foundry/cache/models/Microsoft/qwen3-0.6b-generic-cpu-4/v4 \
//!   cargo test -p onnx-genai-engine --features mlas --test qwen3_0_6b_divergence \
//!   -- --ignored --nocapture
//! ```

use onnx_genai_engine::{GenerateOptions, NativeDecodeDevice, NativeDecodeSession, ProcessorChain};
use onnx_genai_ort::Tokenizer;

/// Raw teacher-forced prefix: the 14 prompt tokens plus the four generated
/// tokens native and ORT agree on, ending just before the divergence step.
const DIVERGENCE_PREFIX: [u32; 18] = [
    7985, 264, 2805, 3364, 911, 264, 33200, 46609, 879, 40324, 458, 13833, 13638, 13, 576, 3364,
    1265, 2924,
];

/// Native's (fp32-oracle-correct) choice at the divergence step.
const NATIVE_DIVERGENCE_TOKEN: u32 = 518;
/// ORT's (int8-activation, lower-precision) choice — must NOT be what native emits.
const ORT_DIVERGENCE_TOKEN: u32 = 264;

/// Default model directory on the numerical-correctness host. Overridable via
/// `QWEN3_0_6B_E2E_DIR` so other hosts can point at their own copy.
const DEFAULT_MODEL_DIR: &str =
    "/home/justinchu/.foundry/cache/models/Microsoft/qwen3-0.6b-generic-cpu-4/v4";

#[test]
#[ignore = "requires the real qwen3-0.6b int4 model via QWEN3_0_6B_E2E_DIR (or the default foundry cache path)"]
fn qwen3_0_6b_int4_native_decode_keeps_high_precision_argmax() -> anyhow::Result<()> {
    let dir = std::env::var_os("QWEN3_0_6B_E2E_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::PathBuf::from(DEFAULT_MODEL_DIR));
    if !dir.is_dir() {
        eprintln!(
            "skipping qwen3-0.6b divergence lock: model directory absent: {} \
             (set QWEN3_0_6B_E2E_DIR to a directory containing model.onnx, genai_config.json, \
             and tokenizer.json)",
            dir.display()
        );
        return Ok(());
    }

    let mut session = NativeDecodeSession::load(dir.join("model.onnx"), NativeDecodeDevice::Cpu)?;
    let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json"))?;

    // Teacher-forced single greedy step: feed the exact agreed prefix, emit one
    // token, and capture the top log-probabilities so we can assert the argmax
    // *and* document the benign near-tie against ORT's alternative.
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
        "native must keep the fp32-oracle-correct token {NATIVE_DIVERGENCE_TOKEN} at the \
         divergence step; got {selected}. ORT's lower-precision int8-activation path selects \
         {ORT_DIVERGENCE_TOKEN} — a change here means native accuracy regressed to (or below) \
         ORT's int8 tie-break.",
    );
    assert_ne!(
        selected, ORT_DIVERGENCE_TOKEN,
        "native emitted ORT's lower-precision int8-activation token {ORT_DIVERGENCE_TOKEN}; the \
         fp32 oracle selects {NATIVE_DIVERGENCE_TOKEN}",
    );

    // Document the benign-tie structure: 518 and 264 are the top two candidates
    // and their margin is razor-thin. A future change that widens this gap or
    // changes the candidate set is a real behavior change worth reviewing.
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
        .expect("ORT alternative token 264 must be in the native top-8 (it is the runner-up)");
    let gap = lp_native - lp_ort;
    assert!(
        gap > 0.0 && gap < 0.2,
        "expected a razor-thin benign tie between {NATIVE_DIVERGENCE_TOKEN} and \
         {ORT_DIVERGENCE_TOKEN} (0 < gap < 0.2 logprob); got gap {gap} \
         (native {lp_native}, ort {lp_ort})",
    );

    eprintln!(
        "qwen3-0.6b divergence lock OK: native token = {NATIVE_DIVERGENCE_TOKEN} \
         (fp32-oracle-correct; ORT = {ORT_DIVERGENCE_TOKEN}), benign-tie gap = {gap:.5} logprob",
    );
    Ok(())
}
