//! Greedy-decode parity lock for Phi-4-mini int4.
//!
//! Native CUDA and ORT CUDA must produce the same fixed greedy stream for this
//! deterministic prompt. The full expected sequence makes this a regression
//! lock rather than a merely self-consistent backend comparison.
//!
//! ```bash
//! ONNX_GENAI_PHI4_MINI_CUDA_DIR=/path/to/model CUDA_VISIBLE_DEVICES=0 \
//! cargo test -p onnx-genai-engine --features cuda,native-backend \
//!   --test phi4_mini_divergence -- --ignored --nocapture
//! ```
#![cfg(all(feature = "cuda", feature = "native-backend"))]

#[path = "common/decode_lock.rs"]
mod decode_lock;

const PROMPT: &str = "The capital of France is";
const EXPECTED_TOKENS: &[u32] = &[
    12650, 13, 4614, 382, 290, 9029, 328, 10128, 30, 12650, 13, 199999, 198, 27, 956, 2518, 1904,
    29, 15, 198, 3575, 553, 261, 10297, 326, 44363, 20837, 29186, 13, 1608, 738, 6052, 5359, 4122,
    402, 290, 3992, 21179, 11, 1118, 382, 261, 77177, 22311, 328, 261, 53556, 885, 8866, 326, 3100,
    364, 56949, 290, 53556, 8866, 326, 3100, 316, 6052, 290, 3992, 4928, 25,
];

#[test]
#[ignore = "requires the real Phi-4-mini int4 export and a CUDA device"]
fn phi4_mini_native_matches_ort_greedy() -> anyhow::Result<()> {
    decode_lock::assert_native_matches_ort_greedy(
        "ONNX_GENAI_PHI4_MINI_CUDA_DIR",
        PROMPT,
        EXPECTED_TOKENS,
    )
}
