# Sapper — History

## 2026-07-12: Joined
Hired as Systems Dev to add capacity alongside Deckard on model building and preprocessing. Project: onnx-genai, a Rust ONNX Runtime generative-AI inference runtime. Context: `onnx-genai-preprocess` is its own crate (image + audio); Mobius (`../mobius`) builds models — `build-gguf` (Q4 MatMulNBits), `--ep webgpu` (GQA), `--static-cache`; we emit our own `InferenceMetadata` (`inference_metadata.yaml`) not ORT-GenAI genai_config. Python builders use onnxscript/onnx-ir. Mobius PRs must pass `lintrunner` (RUFF + RUFF-FORMAT).

## 2026-07-13: Landed multi-tile VLM prompt expansion
Added the preprocessing-side prompt token-expansion library for multi-tile VLM inputs so vision token blocks can be expanded before generation. Landed as commit `9610b34`.

## 2026-07-13T20:55:00Z — Mobius emitter aligned to shared_kv + Gemma4 VLM scope
- Updated the Mobius onnx-genai emitter to emit canonical proposal_type: shared_kv (was gemma4_assistant). Tests 17/17 + 41/41 passing. Commit 498ecf0 on branch feat/gemma4-assistant-onnx-genai.
- Recorded Gemma4 multimodal (VLM) export as a major deferred effort: requires rank-3 pre-patchified vision ingestion, embedding→decoder orchestration (Gemma4 feeds inputs_embeds, not token IDs), and extended Mobius PR #398 pipeline topology. Adding two metadata fields alone cannot make the package load. Concrete values: image token id 258880, tokens_per_tile=280 (E2B).

## 2026-07-14T00-49-37Z — Gemma4 E2B real-run batch (W1 + Milestone A)

**W1 — Gemma-4 E2B merged export** (Mobius commit 8c77d78, feat/gemma4-assistant-onnx-genai)
- Package: `~/gemma4-e2b-onnx/` — target 10.3 GB f16 + assistant 359 MB + merged metadata
- TARGET: `input_ids → logits + projected_state(f32,1536) + present.{0..14}` (hd256 sliding / hd512 full at layers 4,9,14)
- Merged `inference_metadata.yaml` with target-folded `shared_kv` groups (sliding→[0,1,2,3,5,6,7,8,10,11,12,13], full→[4,9,14])
- Mobius: `projected_state` f32 output, text-only registry, `write_merged_inference_metadata`, `_folded_shared_kv_groups`
- Tests: 20/20 schema + 162/162 integration passing

**Milestone A — CUDA greedy on H200** (commit abd0b7a)
- Prompt `"<bos>The capital of France is"` → `"Paris."` ✅; ~166 tok/s, 19 GB VRAM, 83% GPU util
- Root cause of initial garbage: missing BOS — tokenizer has no `add_bos_token: true`
- Code change: `crates/onnx-genai/Cargo.toml` `cuda = ["onnx-genai-ort/cuda"]` feature
- `scripts/run_target_greedy_cuda.sh` added

**Follow-up needed:** fix E2B package tokenizer to auto-prepend BOS
