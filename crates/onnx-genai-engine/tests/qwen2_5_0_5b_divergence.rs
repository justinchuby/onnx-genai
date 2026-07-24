//! Qwen2.5-0.5B int4 native-vs-ORT CUDA greedy-decode lock.
//!
//! For the raw prompt `"The capital of France is"`, native and ORT CUDA produce
//! the same 64-token sequence. Run the real-model lock with:
//!
//! ```bash
//! ONNX_GENAI_QWEN05B_CUDA_DIR=/path/to/model \
//! CUDA_VISIBLE_DEVICES=0 cargo test -p onnx-genai-engine \
//!   --features native-backend,cuda --test qwen2_5_0_5b_divergence \
//!   -- --ignored --nocapture
//! ```
#![cfg(all(feature = "native-backend", feature = "cuda"))]

#[path = "common/qwen_decode_lock.rs"]
mod qwen_decode_lock;

const EXPECTED_TOKENS: &[u32] = &[
    12095, 13, 1084, 374, 279, 7772, 3283, 304, 279, 1879, 323, 279, 1429, 94451, 3283, 304, 279,
    1879, 13, 1084, 374, 1083, 279, 6722, 315, 279, 9292, 315, 279, 1852, 829, 13, 576, 6722, 315,
    279, 9292, 315, 279, 1852, 829, 374, 1083, 12095, 13, 3555, 374, 279, 6722, 315, 279, 9292,
    315, 279, 1852, 829, 30, 576, 4226, 374, 3070, 59604, 334, 13,
];

#[test]
#[ignore = "requires the deployed Qwen2.5-0.5B int4 model and a CUDA device"]
fn qwen2_5_0_5b_native_cuda_matches_ort_cuda() -> anyhow::Result<()> {
    qwen_decode_lock::assert_native_cuda_matches_ort_cuda(
        "Qwen2.5-0.5B",
        "ONNX_GENAI_QWEN05B_CUDA_DIR",
        EXPECTED_TOKENS,
    )
}
