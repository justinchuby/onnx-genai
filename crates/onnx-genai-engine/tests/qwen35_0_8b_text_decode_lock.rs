//! Native CUDA greedy-decode byte-identity lock for **Qwen3.5-0.8B (text)** —
//! the small-end anchor of the qwen3.5 hybrid context-scaling moat family
//! (0.8B + 2B + 9B).
//!
//! Qwen3.5-0.8B is a **hybrid** decoder: 24 layers = **6 periodic
//! full-attention `GroupQueryAttention` layers** (indices 3, 7, 11, 15, 19, 23;
//! `head_dim = 256`, 8 q-heads / 2 kv-heads) interleaved with **18 gated
//! linear-attention / short-conv** recurrent layers (`conv_state` +
//! `recurrent_state` instead of dense KV). Native captures the whole hybrid
//! graph into one CUDA graph (context-flat), while ORT-CUDA cannot place the
//! recurrent ops on GPU and runs eager with per-step Memcpy fallbacks — the same
//! 25-Memcpy graph-block Wallace confirmed at the arch level on 0.8B (1037 CUDA
//! / 56 CPU nodes). This lock is the durable byte-identity gate for that moat at
//! the smallest family scale.
//!
//! ## Export (persisted, self-contained, reproduces this golden)
//!   `QWEN35_0_8B_TEXT_DIR=/home/justinchu/qwen35-0.8b-text-cuda`
//!
//! Foundry ships qwen3.5-0.8b as a multimodal **split** package
//! (`embedding.onnx` + `text.onnx` + `vision.onnx` + `genai_config.json`) that
//! the native single-model loader rejects (and that segfaults `onnxruntime-genai`
//! on load). This export is a **standalone, text-only** single graph
//! (`input_ids` + `attention_mask` + `position_ids` + per-layer past KV /
//! conv+recurrent state -> `logits` + present state) composed by graph surgery:
//! the `embedding` subgraph is **pruned to its `GatherBlockQuantized` text
//! gather** (dropping the image-token `Equal`/`NonZero`/`ScatterND` merge, which
//! is a no-op with no image tokens and whose dynamic `NonZero` native
//! shape-inference cannot bound) and fused into the `text` decoder via
//! `inputs_embeds`. Both `embedding.onnx.data` (0.13 GB) and `text.onnx.data`
//! (0.73 GB) are real copies inside the dir, so the export is stable and
//! self-contained. This is the identical playbook used for qwen3.5-9b (#1449)
//! and gemma4-e2b (#1442); reproduced by
//! `qwen35-0.8b-text-cuda/export_qwen35_0_8b_text.py`.
//!
//! Per-op head sizes / KV-vs-recurrent layer roles are resolved structurally by
//! the loader from the graph's own port inventory (attribute/shape-driven, never
//! model-name-gated; RULES.md §2). Like the qwen3.5-9b export (and unlike the
//! composed gemma4-e2b export) this one **captures cleanly** into a single CUDA
//! graph — so the lock runs on the default capture path.
//!
//! ```bash
//! QWEN35_0_8B_TEXT_DIR=/home/justinchu/qwen35-0.8b-text-cuda \
//! CUDA_VISIBLE_DEVICES=1 cargo test -p onnx-genai-engine \
//!   --features native-backend,cuda --test qwen35_0_8b_text_decode_lock \
//!   -- --ignored --test-threads=1 --nocapture
//! ```
#![cfg(feature = "native-cuda")]

#[path = "common/decode_lock.rs"]
mod decode_lock;

const MODEL_DIR_ENV: &str = "QWEN35_0_8B_TEXT_DIR";
const PROMPT: &str = "The capital of France is";

// Golden = the native-CUDA (whole-graph capture) greedy stream for the composed
// qwen3.5-0.8b text export. Byte-identical (all 16 tokens) to the independently
// validated ORT-driven reference stream for this model — the split-package
// `qwen35_0_8b_hybrid_text_decode_e2e` lock (ORT places the standard-attention
// layers on its EP and CPU-falls-back the com.microsoft hybrid ops) decodes the
// exact same `" Paris, and the capital of Germany is Berlin.\nThe capital of
// France is"`. Every step drives all 24 hybrid layers, including the 6
// head_dim=256 full-attention layers and the 18 linear-attn / short-conv
// recurrent layers.
const EXPECTED_TOKENS: &[u32] = &[
    11751, 11, 321, 279, 6511, 314, 9564, 369, 19241, 13, 198, 760, 6511, 314, 9338, 369,
];

#[test]
#[ignore = "requires the standalone qwen3.5-0.8b text export and a CUDA device"]
fn qwen35_0_8b_text_native_cuda_matches_golden_greedy_sequence() -> anyhow::Result<()> {
    decode_lock::assert_native_matches_golden(MODEL_DIR_ENV, PROMPT, EXPECTED_TOKENS)
}
