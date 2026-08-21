//! Native CUDA greedy-decode coherence lock for real gpt-oss-20b int4 (MoE).
//!
//! gpt-oss-20b is a mandate moat model: ORT-CUDA has no QMoE kernel and forces
//! all 24 QMoE + 24 GQA nodes to CPU (~0.52 tok/s, graph disabled). Native runs
//! it fully on GPU. Enabling native required three op-coverage fixes landed on
//! this branch:
//!   1. `GatherBlockQuantized` `bits` attribute default (4) when omitted.
//!   2. `GatherBlockQuantized` signed native `Int4` packed-dtype support
//!      (components=1, symmetric zero-point 0, sign-extended nibble) for the
//!      `model.embed_tokens.weight_Q4` [201088, 2880] signed-int4 embedding.
//!   3. `GroupQueryAttention` learned attention sink (`head_sink`, input 11):
//!      gpt-oss adds a per-head sink logit to the softmax denominator. Wired
//!      through the f32 reference (prefill) and f32 split-K decode/merge paths;
//!      fused flash is routed around when a sink is present.
//!
//! Golden = the native-CUDA greedy stream (same oracle rationale as
//! `deepseek_v2_lite_decode_lock`: for int4 MoE decode the CPU EP is not a
//! bit-identity oracle -- CPU folds each K-reduction sequentially while native
//! reduces in a log-depth tree; both are valid fp32 orders). Wallace's ORT-CPU
//! reference `[1072, 290, 29082, ...]` was captured on a *different* (16-token)
//! tokenization of the same prompt and is prompt-tail repetition, so it is not a
//! matched byte-identity target for native's 11-token tokenization. Native's
//! stream is coherent ("... and then runs ...") and deterministic across graphs
//! off/on; this test locks it so the enablement cannot silently regress.
//!
//! ```bash
//! GPT_OSS_20B_CUDA_DIR=/home/justinchu/.foundry/cache/models/Microsoft/gpt-oss-20b-generic-cpu-1/v1 \
//! CUDA_VISIBLE_DEVICES=2 cargo test -p onnx-genai-engine \
//!   --features native-backend,cuda --test gpt_oss_20b_decode_lock \
//!   -- --ignored --test-threads=1 --nocapture
//! ```
#![cfg(feature = "native-cuda")]

#[path = "common/decode_lock.rs"]
mod decode_lock;

const PROMPT: &str = "The quick brown fox jumps over the lazy dog and then";

// Native-CUDA greedy stream (24 tokens). Captured GPU-pinned, --test-threads=1.
// Token 0 = 13719 (" runs"); the full continuation is coherent English:
// "... and then runs into the forest, and the fox is the fastest animal in the
// forest. The fox is the fastest animal in the".
const EXPECTED_TOKENS: &[u32] = &[
    13719, 1511, 290, 19458, 11, 326, 290, 68347, 382, 290, 32840, 13983, 306, 290, 19458, 13, 623,
    68347, 382, 290, 32840, 13983, 306, 290,
];

#[test]
#[ignore = "requires the real gpt-oss-20b int4 export and a CUDA device"]
fn gpt_oss_20b_native_cuda_matches_golden_greedy_sequence() -> anyhow::Result<()> {
    decode_lock::assert_native_matches_golden("GPT_OSS_20B_CUDA_DIR", PROMPT, EXPECTED_TOKENS)
}
