//! Native CUDA greedy-decode coherence lock for Gemma-2-2b (`head_dim = 256`).
//!
//! Gemma-2 uses a non-standard attention head size of **256** (standard decoders
//! use 64/128). This test is the durable regression gate for that head-size
//! breadth: the native CUDA attention/GEMV path takes `head_size` as a runtime
//! parameter (no compile-time `head_size == 128` assumption), so `head_dim = 256`
//! is supported by construction -- this lock proves it end-to-end and stops any
//! future kernel change from silently regressing the wide-head path.
//!
//! The export is a standalone, text-only Gemma-2-2b-it decoder (`input_ids` +
//! `attention_mask` -> `logits` + per-layer KV) produced by Mobius. The golden
//! stream below was captured from the **f32** reference export and is
//! **byte-identical** to the CPU EP and ORT streams for the same export (all
//! three emit the same 24 token ids), and is graph-independent (identical with
//! `ONNX_GENAI_CUDA_GRAPH=0` and `=1`).
//!
//! Canonical export (persisted, reproduces this golden):
//!   `GEMMA2_2B_CUDA_DIR=/home/justinchu/gemma2-2b-it-mobius-cpu-f32`
//!
//! IMPORTANT: Gemma degenerates into single-token repetition ("Hel" = id 2405)
//! without a leading `<bos>`. The export's `tokenizer.json` MUST carry the BOS
//! `TemplateProcessing` post-processor (the native tokenizer honors it) --
//! early GGUF-derived exports dropped it (fixed in Mobius #518). Only the f32
//! reference reproduces this exact stream: int4 exports decode coherently but
//! their argmax stream drifts from the f32 golden by quantization, so this
//! byte-identical gate is pinned to the f32 reference.
//!
//! ```bash
//! GEMMA2_2B_CUDA_DIR=/home/justinchu/gemma2-2b-it-mobius-cpu-f32 \
//! CUDA_VISIBLE_DEVICES=0 cargo test -p onnx-genai-engine \
//!   --features native-backend,cuda --test gemma2_2b_head_dim_256_decode_lock \
//!   -- --ignored --nocapture
//! ```
#![cfg(feature = "native-cuda")]

#[path = "common/decode_lock.rs"]
mod decode_lock;

const PROMPT: &str = "Hello";

// Golden = the native-CUDA greedy stream for the standalone Gemma-2-2b-it export.
// Byte-identical to the CPU EP and ORT streams for the same export (head_dim=256,
// GQA 8/4, sliding_window=4096), and reproduced identically with graphs off and
// on. Regenerate with `profile_native --backend native --ep cuda --tokens 24
// --prompt "Hello"` against the export dir.
const EXPECTED_TOKENS: &[u32] = &[
    235248, 109, 235285, 1144, 5326, 577, 3104, 476, 3890, 4451, 2177, 19319, 235269, 26862,
    235269, 578, 22978, 235265, 590, 1144, 3372, 10779, 675, 573,
];

#[test]
#[ignore = "requires the standalone Gemma-2-2b (head_dim=256) export and a CUDA device"]
fn gemma2_2b_head_dim_256_native_cuda_matches_golden_greedy_sequence() -> anyhow::Result<()> {
    decode_lock::assert_native_matches_golden("GEMMA2_2B_CUDA_DIR", PROMPT, EXPECTED_TOKENS)
}
