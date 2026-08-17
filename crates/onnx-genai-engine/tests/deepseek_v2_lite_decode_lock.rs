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
#![cfg(all(feature = "native-backend", feature = "cuda"))]

#[path = "common/decode_lock.rs"]
mod decode_lock;

const PROMPT: &str = "Hello";
const EXPECTED_TOKENS: &[u32] = &[
    11, 304, 608, 245, 207, 16, 24, 1012, 1712, 5075, 13, 304, 608, 245, 1079, 37844, 1491, 13,
    304, 608, 245, 1079, 2074, 18891,
];

#[test]
#[ignore = "requires the real DeepSeek-V2-Lite int4 export and a CUDA device"]
fn deepseek_v2_lite_native_cuda_matches_golden_greedy_sequence() -> anyhow::Result<()> {
    decode_lock::assert_native_matches_golden("DEEPSEEK_V2_LITE_CUDA_DIR", PROMPT, EXPECTED_TOKENS)
}
