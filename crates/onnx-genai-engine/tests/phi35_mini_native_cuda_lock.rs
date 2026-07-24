//! Greedy-decode parity lock for the shipped Foundry **Phi-3.5-mini-instruct**
//! int4 (block-32, `accuracy_level=4`) artifact, run on the native CUDA
//! execution provider.
//!
//! ## The bug this locks (native fp32-activation acc-4 CUDA incoherence)
//!
//! On the fp32-activation decode path (the default for this int4/block-32
//! `accuracy_level=4` Olive/Foundry artifact) the native CUDA `MatMulNBits`
//! `accuracy_level=4` kernels quantized the fp32 activations to int8 with a
//! **single per-row scale over the whole K**. That per-row scale is dominated by
//! activation outliers and rounds the small in-block magnitudes to zero, so
//! native decode diverged from ORT at the **third** generated token and spiralled
//! into garbage (`"…nınızın dışında dışınızın…"`), while ORT CUDA produced
//! coherent English on the identical, unmodified graph with no experimental
//! flags.
//!
//! Root cause was isolated with an fp32 oracle: rewriting every `MatMulNBits`
//! `accuracy_level` `4 → 1` (fp32 activation compute) made native CUDA select the
//! coherent token `306` by a wide margin (the garbage token `29876` was not even
//! in the fp32 top-6), proving the defect lived purely in the acc-4
//! int8-activation quantization, not attention/rotary/norm. The fix quantizes the
//! activation **per K-block** (block-32), matching ORT/MLAS CompInt8 and the CPU
//! native path, which restores token-exact agreement with ORT CUDA below.
//!
//! The fast int4/block-32 **fp16** decode path used by every other model is a
//! different kernel family (`matmul_nbits_gemv_int8_f16*`) and is byte-identical
//! (source unchanged); its GPU parity tests and a Qwen2.5-0.5b native-vs-ORT
//! sanity are unaffected.
//!
//! ## Correctness horizon
//! After the fix native CUDA and ORT CUDA agree **token-exact** for the greedy
//! stream locked below on the `"Hello"` prompt. Both paths are `accuracy_level=4`
//! with per-block int8 activation quantization; two numerically-distinct EP
//! implementations agreeing exactly is the regression lock.
//!
//! ```bash
//! PHI35_MINI_E2E_DIR=\
//!   ~/.foundry/cache/models/Microsoft/Phi-3.5-mini-instruct-generic-cpu-2/v2 \
//! CUDA_VISIBLE_DEVICES=0 cargo test -p onnx-genai-engine \
//!   --features cuda,native-backend --test phi35_mini_native_cuda_lock \
//!   -- --ignored --nocapture
//! ```
#![cfg(all(feature = "cuda", feature = "native-backend"))]

#[path = "common/decode_lock.rs"]
mod decode_lock;

const PROMPT: &str = "Hello";

/// The greedy stream where native CUDA and ORT CUDA agree exactly on the Foundry
/// Phi-3.5-mini int4/block-32 `accuracy_level=4` artifact after the per-K-block
/// activation-quantization fix. Before the fix native diverged at index 2
/// (native `29876`, ORT `306`) and produced incoherent output.
const EXPECTED_TOKENS: &[u32] = &[
    30751, 31512, 306, 29915, 29885, 1985, 373, 263, 2060, 988, 306, 817, 304, 1653, 263, 15171,
    6270, 10754, 363, 263, 26797, 1848, 8720, 4086,
];

#[test]
#[ignore = "requires the shipped Foundry Phi-3.5-mini int4 artifact and a CUDA device"]
fn phi35_mini_native_matches_ort_greedy() -> anyhow::Result<()> {
    decode_lock::assert_native_matches_ort_greedy("PHI35_MINI_E2E_DIR", PROMPT, EXPECTED_TOKENS)
}
