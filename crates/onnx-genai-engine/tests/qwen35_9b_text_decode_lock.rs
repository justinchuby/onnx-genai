//! Native CUDA greedy-decode byte-identity lock for **Qwen3.5-9B (text)** — the
//! scale-up of the qwen3.5 hybrid context-scaling moat.
//!
//! Qwen3.5-9B is a **hybrid** decoder: 32 layers = **8 periodic full-attention
//! `GroupQueryAttention` layers** (indices 3, 7, 11, 15, 19, 23, 27, 31;
//! `head_dim = 256`, 16 q-heads / 4 kv-heads) interleaved with **24 gated
//! linear-attention / short-conv layers** carrying `conv_state` +
//! `recurrent_state` instead of dense KV. Native captures the whole hybrid graph
//! into one CUDA graph (context-flat), while ORT-CUDA cannot place the recurrent
//! ops and runs eager with per-step Memcpy fallbacks — so the native lead grows
//! with context. This lock is the durable byte-identity gate for that moat at
//! the 9B scale.
//!
//! ## Export (persisted, self-contained, reproduces this golden)
//!   `QWEN35_9B_TEXT_DIR=/home/justinchu/qwen35-9b-text-cuda`
//!
//! Foundry ships qwen3.5-9b as a multimodal **split** package
//! (`embedding.onnx` + `text.onnx` + `vision.onnx` + `genai_config.json`) that
//! the native single-model loader rejects. This export is a **standalone,
//! text-only** single graph (`input_ids` + `attention_mask` + `position_ids` +
//! per-layer past KV / conv+recurrent state -> `logits` + present state)
//! composed by graph surgery: the `embedding` subgraph is **pruned to its
//! `GatherBlockQuantized` text-embedding gather** (dropping the image-token
//! `Equal`/`NonZero`/`ScatterND` merge, which is a no-op with no image tokens
//! and whose dynamic `NonZero` native shape-inference cannot bound) and fused
//! into the `text` decoder via `inputs_embeds`. Both `embedding.onnx.data`
//! (0.54 GB) and `text.onnx.data` (7.65 GB) are real copies inside the dir, so
//! the export is stable and self-contained.
//!
//! Per-op head sizes / KV-vs-recurrent layer roles are resolved structurally by
//! the loader from the graph's own port inventory
//! (`engine/load.rs::maybe_fill_hybrid_io_from_graph`, attribute/shape-driven,
//! never model-name-gated; RULES.md §2). Unlike the composed gemma4-e2b export,
//! this one **captures cleanly** into a single CUDA graph — so the lock runs on
//! the default capture path.
//!
//! ```bash
//! QWEN35_9B_TEXT_DIR=/home/justinchu/qwen35-9b-text-cuda \
//! CUDA_VISIBLE_DEVICES=1 cargo test -p onnx-genai-engine \
//!   --features native-backend,cuda --test qwen35_9b_text_decode_lock \
//!   -- --ignored --test-threads=1 --nocapture
//! ```
#![cfg(all(feature = "native-backend", feature = "cuda"))]

#[path = "common/decode_lock.rs"]
mod decode_lock;

const PROMPT: &str = "The capital of France is";

// Golden = the native-CUDA (whole-graph capture) greedy stream for the composed
// qwen3.5-9b text export. Byte-identical to the ORT CPU-EP greedy stream for the
// same export over all 16 tokens. The first 8 decode " Paris.\nThe capital of
// France is"; greedy then repeats that clause (stop_on_eos disabled, so the
// fixed-length stream is fully pinned). Every step drives all 32 hybrid layers,
// including the 8 head_dim=256 full-attention layers and the 24 linear-attn /
// short-conv recurrent layers.
const EXPECTED_TOKENS: &[u32] = &[
    11751, 13, 198, 760, 6511, 314, 9338, 369, 11751, 13, 198, 760, 6511, 314, 9338, 369,
];

#[test]
#[ignore = "requires the standalone qwen3.5-9b text export and a CUDA device"]
fn qwen35_9b_text_native_cuda_matches_golden_greedy_sequence() -> anyhow::Result<()> {
    decode_lock::assert_native_matches_golden("QWEN35_9B_TEXT_DIR", PROMPT, EXPECTED_TOKENS)
}
