//! Native CUDA greedy-decode byte-identity lock for the real **Qwen3.5-2B-text
//! hybrid** (linear-attention / SSM + gated attention).
//!
//! Qwen3.5-2B-text is a mandate moat model of a NEW class: a *graph-block* moat
//! (distinct from gpt-oss's CPU-fallback moat). The export is
//! `Qwen3_5ForConditionalGeneration` with 24 layers = 18× `LinearAttention` +
//! 18× `CausalConvWithState` (Mamba-style, constant-size recurrent state) + only
//! 6× `GroupQueryAttention` (full attention at layers 3,7,11,15,19,23). ORT-CUDA
//! *does* place ~1027/1037 nodes on its EP, but the hybrid forces 25 `Memcpy`
//! nodes into the graph, so ORT reports "unable to run CUDA graph" and runs
//! eager-only. Native captures the whole hybrid graph (fallbacks=0), so its
//! decode throughput is context-FLAT while ORT's collapses with context — the
//! native advantage GROWS with context (matched fair A/B, GPU-pinned,
//! medians-of-5):
//!
//!   ctx ~5 tok   : native g1 174.9  vs ORT 164.6  = 1.06x
//!   ctx ~362 tok : native g1 169.0  vs ORT 112.9  = 1.50x
//!   ctx ~1729 tok: native g1 168.6  vs ORT  55.6  = 3.03x  (and still climbing)
//!
//! Golden = the native-CUDA greedy stream (same oracle rationale as
//! `gpt_oss_20b_decode_lock` / `deepseek_v2_lite_decode_lock`: for this hybrid
//! int4 export the ORT path runs eager-only and is not a byte-identity oracle —
//! native's captured decode reduces in a different but equally valid fp32 order).
//! Native's stream is coherent and deterministic across graph off/on; this test
//! locks it so the hybrid moat enablement cannot silently regress.
//!
//! ```bash
//! QWEN35_2B_TEXT_DIR=/home/justinchu/.foundry/cache/models/Microsoft/qwen3.5-2b-text-generic-cpu-1/v1 \
//! CUDA_VISIBLE_DEVICES=2 cargo test -p onnx-genai-engine \
//!   --features native-backend,cuda --test qwen35_2b_text_decode_lock \
//!   -- --ignored --test-threads=1 --nocapture
//! ```
#![cfg(all(feature = "native-backend", feature = "cuda"))]

#[path = "common/decode_lock.rs"]
mod decode_lock;

/// Env override for the model dir; falls back to the on-box Foundry cache export
/// (mirrors the `QWEN35_0_8B_HYBRID_DIR` / `DEFAULT_MODEL_DIR` pattern).
const MODEL_DIR_ENV: &str = "QWEN35_2B_TEXT_DIR";
const DEFAULT_MODEL_DIR: &str =
    "/home/justinchu/.foundry/cache/models/Microsoft/qwen3.5-2b-text-generic-cpu-1/v1";

const PROMPT: &str = "The capital of France is";

// Native-CUDA greedy stream (24 tokens). Captured GPU-pinned, --test-threads=1,
// graph capture on (production default). Token 0 = 11751 (" Paris"); the full
// continuation is coherent English: " Paris. The capital of the United States
// is Washington, D.C. The capital of the United Kingdom is London. The". The
// stream is deterministic run-to-run and across graph capture off/on.
const EXPECTED_TOKENS: &[u32] = &[
    11751, 13, 561, 6511, 314, 279, 3516, 4042, 369, 6312, 11, 414, 707, 13, 561, 6511, 314, 279,
    3516, 14634, 369, 6924, 13, 561,
];

#[test]
#[ignore = "requires the real qwen3.5-2b-text int4 hybrid export and a CUDA device"]
fn qwen35_2b_text_native_cuda_matches_golden_greedy_sequence() -> anyhow::Result<()> {
    if std::env::var_os(MODEL_DIR_ENV).is_none() {
        // Default to the on-box Foundry cache export when the override is unset.
        unsafe {
            std::env::set_var(MODEL_DIR_ENV, DEFAULT_MODEL_DIR);
        }
    }
    decode_lock::assert_native_matches_golden(MODEL_DIR_ENV, PROMPT, EXPECTED_TOKENS)
}
