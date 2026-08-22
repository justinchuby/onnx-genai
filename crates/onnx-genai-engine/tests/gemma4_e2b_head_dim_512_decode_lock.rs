//! Native CUDA greedy-decode coherence lock for **Gemma-3n / gemma4-e2b**, the
//! dual-head-size ("gemma4 双head size") text decoder.
//!
//! Gemma-3n interleaves two attention head sizes across its 35 decoder layers:
//! **28 sliding layers at `head_dim = 256`** and **7 full-attention layers at
//! `head_dim = 512`** (KV layers 4, 9, 14, 19, 24, 29, 34). The exact per-layer
//! head sizes are read structurally by the engine's KV bridge from each
//! `present.N.key` output shape — never from a model name or a fixed value
//! (`onnx-genai-engine/src/kv_bridge.rs::layer_configs_from_key_outputs`,
//! RULES.md §2: head size is a fully runtime per-attention-op parameter). This
//! lock proves the whole mixed-head decode end-to-end, and in particular that
//! the `head_dim = 512` full-attention layers run on the **fused split-K decode
//! kernel** whose ceiling was raised 256 -> 512 in #1438 (`gqa_decode{,_fp16,
//! _bf16}.rs MAX_HEAD_DIM = 512`). Without #1438 those layers would silently
//! fall back to the serial `gqa_attention_reference` path; this test is the
//! durable e2e gate that they stay on the fast path and stay coherent.
//!
//! ## Export (persisted, self-contained, reproduces this golden)
//!   `GEMMA4_E2B_512_DIR=/home/justinchu/gemma4-e2b-it-text-cuda`
//!
//! It is a **standalone, text-only** single graph
//! (`input_ids` + `attention_mask` + per-layer past KV -> `logits` + present KV)
//! composed from the official Gemma-3n E2B ONNX pipeline
//! (`/home/justinchu/gemma4-e2b-onnx`) by fusing the `embedding` subgraph
//! (which produces `inputs_embeds` + the routed `per_layer_inputs`) into the
//! `decoder` subgraph and baking the multimodal `image_features`/`audio_features`
//! inputs to empty constants, so the standard `input_ids` decode loop drives it
//! exactly like the Gemma-2 head_dim=256 lock. Both `embedding.onnx.data` and
//! `decoder.onnx.data` are real copies inside the dir (no dangling symlinks, no
//! external escape), so the export is stable and self-contained.
//!
//! IMPORTANT: like Gemma-2, Gemma-3n degenerates into single-token repetition
//! without a leading `<bos>`. The pipeline's stock `tokenizer.json` ships a
//! no-op `TemplateProcessing` post-processor (it does NOT add BOS); this export
//! patches it to prepend `<bos>` (id 2), which the native tokenizer honors, so
//! `generate("Hello")` tokenizes to `[2, 9259]` and decodes coherently.
//!
//! CAPTURE: this lock runs with CUDA-graph capture OFF. The composed graph's
//! merged present-KV sequence axis is an opaque symbol that the prefill
//! workspace planner cannot yet upper-bound for capture (it trips on a
//! head_dim=256 *sliding* layer, not on the 512 layers), so capture is deferred
//! (see `.squad/decisions/inbox/deckard-gemma4-512-e2e.md`). The eager path
//! exercises the identical fused decode kernels, so the head_dim=512 fast-path
//! coherence guarantee is fully locked here; the greedy stream is
//! capture-independent (same kernels, same reduction order).
//!
//! ```bash
//! GEMMA4_E2B_512_DIR=/home/justinchu/gemma4-e2b-it-text-cuda \
//! CUDA_VISIBLE_DEVICES=0 cargo test -p onnx-genai-engine \
//!   --features native-backend,cuda --test gemma4_e2b_head_dim_512_decode_lock \
//!   -- --ignored --test-threads=1 --nocapture
//! ```
#![cfg(feature = "native-cuda")]

#[path = "common/decode_lock.rs"]
mod decode_lock;

const PROMPT: &str = "Hello";

// Golden = the native-CUDA (eager) greedy stream for the composed gemma4-e2b
// text export. Byte-identical to the ORT CPU-EP greedy stream for the same
// export over all 16 tokens (both emit these exact ids), decoding to
// "! How can I help you today?" followed by `<end_of_turn>`/`<eos>` cycling
// (stop_on_eos is disabled so the fixed-length stream is fully pinned). Every
// decode step attends through all 35 layers, including the 7 head_dim=512
// full-attention layers on the fused split-K kernel.
const EXPECTED_TOKENS: &[u32] = &[
    236888, 2088, 740, 564, 1601, 611, 3124, 236881, 106, 106, 107, 1, 106, 107, 1, 106,
];

#[test]
#[ignore = "requires the standalone gemma4-e2b (dual head_dim 256/512) text export and a CUDA device"]
fn gemma4_e2b_head_dim_512_native_cuda_matches_golden_greedy_sequence() -> anyhow::Result<()> {
    decode_lock::assert_native_matches_golden_eager("GEMMA4_E2B_512_DIR", PROMPT, EXPECTED_TOKENS)
}
