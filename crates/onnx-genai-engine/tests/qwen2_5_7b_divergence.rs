//! Qwen2.5-7B int4 native-vs-ORT CUDA greedy-decode lock.
//!
//! For the raw prompt `"The capital of France is"`, native and ORT CUDA produce
//! the same 64-token sequence. Run the real-model lock with:
//!
//! ```bash
//! ONNX_GENAI_QWEN7B_CUDA_DIR=/path/to/model \
//! CUDA_VISIBLE_DEVICES=0 cargo test -p onnx-genai-engine \
//!   --features native-backend,cuda --test qwen2_5_7b_divergence \
//!   -- --ignored --nocapture
//! ```
#![cfg(all(feature = "native-backend", feature = "cuda"))]

#[path = "common/qwen_decode_lock.rs"]
mod qwen_decode_lock;

const EXPECTED_TOKENS: &[u32] = &[
    12095, 13, 1084, 374, 7407, 304, 279, 18172, 8622, 949, 315, 279, 3146, 13, 576, 3283, 374,
    279, 7772, 304, 9625, 323, 374, 279, 4746, 315, 279, 59108, 273, 6810, 7276, 34106, 5537, 13,
    12095, 374, 279, 1429, 94451, 3283, 304, 279, 7513, 9145, 448, 458, 15662, 7042, 315, 220, 16,
    15, 13, 20, 3526, 13, 1084, 374, 279, 12752, 11, 8353, 11, 323,
];

#[test]
#[ignore = "requires the deployed Qwen2.5-7B int4 model and a CUDA device"]
fn qwen2_5_7b_native_cuda_matches_ort_cuda() -> anyhow::Result<()> {
    qwen_decode_lock::assert_native_cuda_matches_ort_cuda(
        "Qwen2.5-7B",
        "ONNX_GENAI_QWEN7B_CUDA_DIR",
        EXPECTED_TOKENS,
    )
}
