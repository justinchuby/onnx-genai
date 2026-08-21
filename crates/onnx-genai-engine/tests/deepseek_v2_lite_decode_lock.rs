//! Native CUDA greedy-decode coherence lock for real DeepSeek-V2-Lite int4.
//!
//! This locks a regenerated post-#434 Mobius export with explicit `model.io`
//! roles. The older `deepseek_e2e` test is a tiny-vocab structural smoke; this
//! file protects the real MLA+MoE native decode stream.
//!
//! ```bash
//! DEEPSEEK_V2_LITE_CUDA_DIR=/path/to/deepseek-v2-lite-real-int4-post434 \
//! CUDA_VISIBLE_DEVICES=0 cargo test -p onnx-genai-engine \
//!   --features native-backend,cuda --test deepseek_v2_lite_decode_lock \
//!   -- --ignored --nocapture
//! ```
#![cfg(feature = "native-cuda")]

#[path = "common/decode_lock.rs"]
mod decode_lock;

const PROMPT: &str = "Hello";

// Golden = the native-CUDA greedy stream (NOT the CPU stream). Rebased from the
// former CPU golden per Rachael's oracle ruling: for int4 decode the CPU EP is
// *not* a bit-identity oracle. CPU folds each MatMulNBits/QMoE K-reduction
// sequentially left-to-right; native CUDA reduces 256 lanes in a log-depth tree.
// Both are valid fp32 orders, but the tree reduction is provably closer to f64
// truth -- see `matmul_nbits_gpu.rs::run_int8_f64_reference_parity` (dense) and
// `qmoe_gpu.rs::qmoe_int4_identity_expert_gemv_within_f64_roundoff` (expert GEMV,
// CUDA/f64 4.6e-6 vs CPU/f64 7.0e-6 at K=512).
//
// This sequence differs from the old CPU golden only at generated indices 5..
// (CPU: ...207, 17, 15, 1012...  ->  CUDA: ...207, 16, 24, 1012...). Root cause,
// evidenced by the token-5 router-selection dump (ONNX_GENAI_QMOE_ROUTE_DUMP, see
// .squad/decisions/inbox/luv-v2lite-cpu-cuda-divergence.md): a single top-k
// *boundary* swap in the forward that emits token 5. At layer 25 the 6th of 6
// selected experts is a near-tie -- CPU keeps expert 61 (logit 2.94246e-2), CUDA
// keeps expert 1 (2.94382e-2); the first 5 experts and their order are identical
// on both. The two experts' router logits differ by ~5e-5, which is *smaller*
// than the ~4.7-6.4e-5 CPU-vs-CUDA reassociation drift on that same logit -- i.e.
// the ordering is below fp32 numerical resolution. Forwards 0..4 have byte-
// identical expert sets across all 130 routing decisions. The selection logic is
// bit-identical given identical logits (both: total_order desc, index asc), so
// this is benign reassociation tipping a genuine near-tie, not a router bug.
//
// Reproduced identically on GPU 0 (graphs off) and GPU 1 (graphs on): the stream
// is deterministic and graph-independent. Regenerate with:
//   deepseek_v2_lite_native_cuda_matches_golden_greedy_sequence (this file).
const EXPECTED_TOKENS: &[u32] = &[
    11, 304, 608, 245, 207, 16, 24, 1012, 1712, 5075, 13, 304, 608, 245, 1079, 37844, 1491, 13,
    304, 608, 245, 1079, 2074, 18891,
];

#[test]
#[ignore = "requires the real DeepSeek-V2-Lite int4 export and a CUDA device"]
fn deepseek_v2_lite_native_cuda_matches_golden_greedy_sequence() -> anyhow::Result<()> {
    decode_lock::assert_native_matches_golden("DEEPSEEK_V2_LITE_CUDA_DIR", PROMPT, EXPECTED_TOKENS)
}

#[test]
#[ignore = "requires the real DeepSeek-V2-Lite int4 export, a CUDA device, and ~340 generated tokens"]
fn deepseek_v2_lite_engine_long_context_workspace_survives_capacity_growth() -> anyhow::Result<()> {
    decode_lock::assert_native_long_context_eager_and_capture_match_prefix(
        "DEEPSEEK_V2_LITE_CUDA_DIR",
        PROMPT,
        340,
        EXPECTED_TOKENS,
    )
}
