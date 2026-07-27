//! Greedy-decode **golden** lock for the shipped Foundry Qwen3-0.6B int4/int8
//! artifact (`qwen3-0.6b-generic-cpu-4/v4`), run on the native CUDA execution
//! provider.
//!
//! This is deliberately a *different* artifact from `qwen3_0_6b_native_cuda_e2e`
//! (which locks a re-exported "postfix" export). The Foundry artifact ships two
//! op configurations the CUDA EP historically declined at load, which — because
//! native CUDA placement is all-or-nothing — pushed the whole decoder onto the
//! CPU EP (measured ~60x slower than ORT):
//!   1. `com.microsoft::GatherBlockQuantized(bits=8, block_size=128)` on
//!      `/model/embed_tokens` (no CUDA handler existed at all), and
//!   2. 105 `MatMulNBits(bits=8, block_size=128, accuracy_level=4)` nodes (the
//!      CUDA factory only claimed int8 at `block_size=32`).
//!
//! With both configurations now claimed and served by capture-safe CUDA
//! kernels, the whole decoder runs on the GPU (verified: one CUDA-graph capture,
//! zero fallbacks, and `ONNX_GENAI_REQUIRE_CUDA=1` no longer reports a CPU-EP
//! reassignment).
//!
//! ## Correctness horizon — fp32 oracle golden stream
//! [`EXPECTED_TOKENS`] is the coherent completion produced by a true fp32
//! reference: an independent oracle rewrote every `MatMulNBits` in the *same*
//! model from `accuracy_level=4` to `accuracy_level=1` (fp32 compute, **no int8
//! activation quantization**) and ran it through onnxruntime_genai. That stream
//! reads " Paris. The capital of Italy is Rome. The capital of Spain is Madrid.
//! The capital of Portugal is Lisbon. The" — a coherent enumeration of
//! capitals.
//!
//! Native CUDA must reproduce that fp32-oracle golden stream **token-exact**.
//! This artifact's MatMulNBits are int4/int8 at **block_size=128**, and int8
//! *activation* quantization for `accuracy_level=4` is only calibrated at
//! `block_size=32`. Quantizing the activations at block-32 against block-128
//! weights (regression #123, generalized by #163) discarded enough precision to
//! flip a razor-thin decode logit tie at generated-token index 5 — native
//! picked token `9625` (" France") and looped
//! (" Paris. The capital of France is Paris. …") instead of the oracle's `15344`
//! (" Italy"). The fix routes int4 `accuracy_level=4` at `block_size != 32`
//! (decode *and* prefill) through the fp32-activation path (the same precedent
//! the int8/block-128 decode specialization set), restoring the coherent stream.
//!
//! This is a **golden** lock: it asserts native CUDA == [`EXPECTED_TOKENS`]
//! directly and does **not** require the ORT CUDA provider to load (which often
//! cannot in this environment). The CPU EP is *not* a usable oracle here — it
//! decodes the same degenerate looping stream on this artifact — so the golden
//! reference is the fp32 `accuracy_level=1` oracle, not ORT or CPU.
//!
//! ```bash
//! ONNX_GENAI_QWEN3_0_6B_FOUNDRY_DIR=\
//!   ~/.foundry/cache/models/Microsoft/qwen3-0.6b-generic-cpu-4/v4 \
//! CUDA_VISIBLE_DEVICES=3 cargo test -p onnx-genai-engine \
//!   --features cuda,native-backend --test qwen3_0_6b_foundry_native_cuda_lock \
//!   -- --ignored --nocapture
//! ```
#![cfg(all(feature = "cuda", feature = "native-backend"))]

#[path = "common/decode_lock.rs"]
mod decode_lock;

const PROMPT: &str = "The capital of France is";

/// The first 24 greedy tokens of the fp32-oracle (`accuracy_level=1`) golden
/// stream for this artifact: " Paris. The capital of Italy is Rome. The capital
/// of Spain is Madrid. The capital of Portugal is Lisbon. The". Native CUDA must
/// reproduce this exactly. Index 5 (`15344`, " Italy") is the tie that regression
/// #123 flipped to `9625` (" France") — the degenerate loop this lock guards.
const EXPECTED_TOKENS: &[u32] = &[
    12095, 13, 576, 6722, 315, 15344, 374, 21718, 13, 576, 6722, 315, 17689, 374, 24081, 13, 576,
    6722, 315, 33311, 374, 80701, 13, 576,
];

#[test]
#[ignore = "requires the shipped Foundry Qwen3-0.6B int4 artifact and a CUDA device"]
fn qwen3_0_6b_foundry_native_matches_golden() -> anyhow::Result<()> {
    decode_lock::assert_native_matches_golden(
        "ONNX_GENAI_QWEN3_0_6B_FOUNDRY_DIR",
        PROMPT,
        EXPECTED_TOKENS,
    )
}
