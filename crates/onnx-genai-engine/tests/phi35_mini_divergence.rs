//! End-to-end greedy-decode regression lock for the Phi-3.5-mini-instruct int4
//! (block-32, `accuracy_level=4`) native-vs-ORT divergence.
//!
//! Native and ORT greedy streams share the first 65 tokens for the "Hello"
//! prompt, then split at decode index 65: **native picks token 263, ORT picks
//! 6455**. This is a deterministic (greedy argmax) numeric divergence, not
//! sampling noise.
//!
//! ## Verdict: native is MORE accurate — keep native.
//!
//! An independent high-precision oracle was built by running the *same*
//! `model.onnx` through ONNX Runtime with every `MatMulNBits` `accuracy_level`
//! rewritten:
//!
//! | MatMulNBits compute | index-65 argmax | logit gap (263 − 6455) |
//! |---------------------|-----------------|------------------------|
//! | acc-level-1 (fp32)  | **263**         | **+0.01281**           |
//! | acc-level-2 (fp16)  | **263**         | **+0.01281**           |
//! | acc-level-3 (bf16)  | **263**         | **+0.01281**           |
//! | acc-level-4 (int8)  | 6455            | −0.02850               |
//!
//! Every non-int8 (higher-precision) compute type selects **263** by a +0.0128
//! margin (~0.02 % of the ~59.7 logit). ONLY int8 *activation* quantization
//! (`accuracy_level=4`, ORT's default here) flips the winner to 6455. Native
//! matches the fp32/fp16/bf16 oracle; ORT does not. Per project policy
//! (correctness beats matching ORT), native is kept and locked here. This is the
//! same failure class Ridley proved on qwen3-0.6b: int8 activation quantization
//! tipping a razor-thin greedy logit race.
//!
//! The kernel-level guard for this class lives in
//! `onnx-runtime-ep-cpu` as
//! `int4_decode_preserves_f32_argmax_where_per_row_int8_activation_flips` and
//! does NOT need the multi-GB Phi-3.5 model, so CI catches the regression class
//! without this fixture. This test additionally pins the exact end-to-end token
//! stream when the real model is available:
//!
//! ```bash
//! PHI35_MINI_E2E_DIR=~/.foundry/cache/models/Microsoft/Phi-3.5-mini-instruct-generic-cpu-2/v2 \
//!   cargo test -p onnx-genai-engine --features mlas --test phi35_mini_divergence \
//!   -- --ignored --nocapture
//! ```

use onnx_genai_engine::{Engine, EngineConfig, EngineDecodeBackend, GenerateRequest};

/// The 65-token greedy prefix native and ORT agree on for the "Hello" prompt,
/// ending at token 3160 just before the divergence.
const SHARED_PREFIX: [u32; 65] = [
    30751, 31512, 306, 29915, 29885, 1985, 373, 263, 2060, 988, 306, 817, 304, 1653, 263, 15171,
    6270, 10754, 363, 263, 26797, 1848, 8720, 4086, 2000, 376, 3399, 29931, 20191, 1213, 450,
    10754, 881, 4612, 278, 4086, 29915, 29879, 5877, 29892, 29505, 29892, 322, 5412, 5680, 29892,
    3704, 967, 1914, 731, 310, 12768, 29892, 848, 4072, 29892, 322, 2761, 12286, 29889, 306, 884,
    864, 304, 3160,
];

/// Native's (fp32-oracle-correct) choice at the divergence step.
const NATIVE_DIVERGENCE_TOKEN: u32 = 263;
/// ORT's (int8-activation, lower-precision) choice — must NOT be what native emits.
const ORT_DIVERGENCE_TOKEN: u32 = 6455;

#[test]
#[ignore = "requires the real Phi-3.5-mini int4 model via PHI35_MINI_E2E_DIR"]
fn phi35_mini_int4_native_decode_keeps_high_precision_argmax() -> anyhow::Result<()> {
    let Some(dir) = std::env::var_os("PHI35_MINI_E2E_DIR") else {
        eprintln!(
            "skipping phi35_mini divergence lock: set PHI35_MINI_E2E_DIR to a directory \
             containing the Phi-3.5-mini-instruct int4 (block-32, acc-level-4) model, its \
             genai_config.json, and tokenizer.json"
        );
        return Ok(());
    };
    let dir = std::path::PathBuf::from(dir);
    if !dir.is_dir() {
        eprintln!(
            "skipping phi35_mini divergence lock: PHI35_MINI_E2E_DIR is absent: {}",
            dir.display()
        );
        return Ok(());
    }

    // Force the native backend so this asserts native's argmax, not ORT's.
    let config = EngineConfig {
        decode_backend: EngineDecodeBackend::Native,
        ..EngineConfig::default()
    };
    let mut engine = Engine::from_dir(&dir, config)?;

    // Greedy, deterministic, "Hello" prompt (tokenizes to [15043]) — the exact
    // reproduction used to isolate the divergence.
    let mut request = GenerateRequest::new("Hello".to_string());
    request.options.max_new_tokens = SHARED_PREFIX.len() + 1;
    request.options.temperature = 0.0;
    request.options.greedy = true;
    request.options.stop_on_eos = false;

    let result = engine.generate(request)?;
    let tokens = &result.token_ids;

    assert!(
        tokens.len() >= SHARED_PREFIX.len() + 1,
        "native decode produced only {} tokens, need at least {}",
        tokens.len(),
        SHARED_PREFIX.len() + 1
    );
    assert_eq!(
        &tokens[..SHARED_PREFIX.len()],
        &SHARED_PREFIX,
        "native greedy prefix drifted from the recorded Phi-3.5 stream",
    );
    assert_eq!(
        tokens[SHARED_PREFIX.len()],
        NATIVE_DIVERGENCE_TOKEN,
        "native must keep the fp32/fp16/bf16-oracle-correct token {NATIVE_DIVERGENCE_TOKEN} at \
         decode index {}; got {}",
        SHARED_PREFIX.len(),
        tokens[SHARED_PREFIX.len()],
    );
    assert_ne!(
        tokens[SHARED_PREFIX.len()],
        ORT_DIVERGENCE_TOKEN,
        "native emitted ORT's lower-precision int8-activation token {ORT_DIVERGENCE_TOKEN}; the \
         higher-precision oracle selects {NATIVE_DIVERGENCE_TOKEN} — native accuracy regressed",
    );

    eprintln!(
        "phi35_mini divergence lock OK: native index-{} token = {} (oracle-correct; ORT = {})",
        SHARED_PREFIX.len(),
        NATIVE_DIVERGENCE_TOKEN,
        ORT_DIVERGENCE_TOKEN,
    );
    Ok(())
}
