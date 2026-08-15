//! Native CUDA greedy-decode coherence lock for GLM-4-9B int4.
//!
//! GLM-4's partial-RoPE GQA schema cannot be loaded by the available ONNX
//! Runtime build, so this lock compares native CUDA against an exact golden
//! sequence rather than against ORT CUDA.
//!
//! ```bash
//! ONNX_GENAI_GLM4_9B_CUDA_DIR=/path/to/glm-4-9b-int4-cuda \
//! CUDA_VISIBLE_DEVICES=0 cargo test -p onnx-genai-engine \
//!   --features native-backend,cuda --test glm4_9b_decode_lock \
//!   -- --ignored --nocapture
//! ```
#![cfg(all(feature = "native-backend", feature = "cuda"))]

#[path = "common/decode_lock.rs"]
mod decode_lock;

const PROMPT: &str = "Hello";
const EXPECTED_TOKENS: &[u32] = &[
    11, 358, 1079, 264, 220, 18, 6498, 1042, 5458, 518, 279, 3822, 315, 75767, 13, 358, 1079, 5023,
    4633, 264, 3308, 304, 28166, 75835, 323, 358, 1079, 3432, 12258, 448, 279, 2701, 3491, 1447,
    14076, 59, 7265, 90, 65, 18082, 92, 220, 16, 609, 220, 15, 609, 220, 15, 24908, 220, 15, 609,
    220, 16, 609, 220, 15, 24908, 220, 15, 609, 220, 15,
];

#[test]
#[ignore = "requires the real GLM-4-9B int4 export and a CUDA device"]
fn glm4_9b_native_cuda_matches_golden_greedy_sequence() -> anyhow::Result<()> {
    decode_lock::assert_native_matches_golden(
        "ONNX_GENAI_GLM4_9B_CUDA_DIR",
        PROMPT,
        EXPECTED_TOKENS,
    )
}
