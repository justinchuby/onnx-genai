//! Greedy-decode parity lock for the **shipped Foundry** Qwen3-0.6B int4/int8
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
//! With both configurations now claimed and served by capture-safe CUDA
//! kernels, the whole decoder runs on the GPU (verified: one CUDA-graph capture,
//! zero fallbacks, and `ONNX_GENAI_REQUIRE_CUDA=1` no longer reports a CPU-EP
//! reassignment).
//!
//! ## Correctness horizon
//! Native CUDA and ORT CUDA agree **token-exact** for the first 24 greedy tokens
//! on this prompt (they actually agree through token 25). Both decode paths are
//! `accuracy_level=4`, but ORT quantizes the fp32 activations to int8 before the
//! block-quantized matmul while the native GEMV accumulates in fp32, so the two
//! numerically-distinct implementations eventually reach a near-tie in the
//! logits and pick different tokens at token 26. Two independent EP
//! implementations agreeing exactly for 24 greedy steps is the regression lock;
//! both continue to produce coherent completions past the horizon. (The CPU EP
//! is *not* a usable higher-precision oracle here: it decodes a different,
//! degenerate looping stream on this artifact.)
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

/// The first 24 greedy tokens, where native CUDA and ORT CUDA agree exactly on
/// the Foundry Qwen3-0.6B int4/int8 artifact.
const EXPECTED_TOKENS: &[u32] = &[
    12095, 13, 576, 6722, 315, 15344, 374, 21718, 13, 576, 6722, 315, 17689, 374, 24081, 13, 576,
    6722, 315, 33311, 374, 80701, 13, 576,
];

#[test]
#[ignore = "requires the shipped Foundry Qwen3-0.6B int4 artifact and a CUDA device"]
fn qwen3_0_6b_foundry_native_matches_ort_greedy() -> anyhow::Result<()> {
    decode_lock::assert_native_matches_ort_greedy(
        "ONNX_GENAI_QWEN3_0_6B_FOUNDRY_DIR",
        PROMPT,
        EXPECTED_TOKENS,
    )
}
