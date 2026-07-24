//! Greedy-decode parity lock for the opt-in **fp16-fused decode** path on the
//! shipped Foundry **Phi-3.5-mini-instruct** int4 (block-32, `accuracy_level=4`)
//! artifact, run on the native CUDA execution provider.
//!
//! ## What this locks
//!
//! Phi-3.5-mini is an fp32-activation quantized export: its `MatMulNBits` carry
//! fp32 scales, so by default the native CUDA backend runs the (fp32-activation)
//! `accuracy_level=4` decode path. Selecting
//! [`DecodePrecision::Fp16`](onnx_genai_engine::DecodePrecision::Fp16) opts the
//! decoder into a whole-graph fp32→fp16 rewrite at session-build time, landing
//! it on the fast fp16-fused decode kernels (half2 `MatMulNBits` GEMV, fused
//! gate/up SwiGLU, skip-RMSNorm, fp16 GQA) that the fp16-activation models use.
//!
//! The gating question was empirical: fp16 is *not* more precise than fp32, so
//! routing an fp32-activation model through fp16-fused decode is only acceptable
//! if it stays token-exact vs the fp32 reference over the decode horizon. This
//! lock answers it: native fp16-fused decode must reproduce the trusted **ORT
//! fp32 CUDA** greedy stream token-for-token for 64 tokens on the `"Hello"`
//! prompt. Two numerically-distinct implementations (native fp16-fused kernels
//! vs ORT fp32) agreeing exactly is the regression lock; any divergence fails.
//!
//! The 64-token horizon sits below the known `accuracy_level=4` int8 activation
//! near-tie at index 103 (see `phi35_mini_divergence.rs`), so the streams agree
//! exactly here with no benign-tie flip to reason about.
//!
//! ```bash
//! PHI35_MINI_E2E_DIR=\
//!   ~/.foundry/cache/models/Microsoft/Phi-3.5-mini-instruct-generic-cpu-2/v2 \
//! CUDA_VISIBLE_DEVICES=0 cargo test -p onnx-genai-engine \
//!   --features cuda,native-backend --test phi35_mini_fp16_decode_lock \
//!   -- --ignored --nocapture
//! ```
#![cfg(all(feature = "cuda", feature = "native-backend"))]

#[path = "common/decode_lock.rs"]
mod decode_lock;

const PROMPT: &str = "Hello";

/// The 64-token greedy stream where native CUDA fp16-fused decode
/// ([`DecodePrecision::Fp16`]) and ORT fp32 CUDA agree exactly on the Foundry
/// Phi-3.5-mini int4/block-32 `accuracy_level=4` artifact for the `"Hello"`
/// prompt. Its first 24 tokens match the fp32-path lock in
/// `phi35_mini_native_cuda_lock.rs`.
const EXPECTED_TOKENS: &[u32] = &[
    30751, 31512, 306, 29915, 29885, 1985, 373, 263, 2060, 988, 306, 817, 304, 1653, 263, 15171,
    6270, 10754, 363, 263, 26797, 1848, 8720, 4086, 2000, 376, 3399, 29931, 20191, 1213, 450,
    10754, 881, 4612, 278, 4086, 29915, 29879, 5877, 29892, 29505, 29892, 322, 5412, 5680, 29892,
    3704, 967, 1914, 731, 310, 12768, 29892, 848, 4072, 29892, 322, 2761, 12286, 29889, 306, 884,
    864, 304,
];

#[test]
#[ignore = "requires the shipped Foundry Phi-3.5-mini int4 artifact and a CUDA device"]
fn phi35_mini_native_fp16_matches_ort_greedy() -> anyhow::Result<()> {
    decode_lock::assert_native_fp16_matches_ort_greedy("PHI35_MINI_E2E_DIR", PROMPT, EXPECTED_TOKENS)
}
