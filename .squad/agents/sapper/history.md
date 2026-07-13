# Sapper — History

## 2026-07-12: Joined
Hired as Systems Dev to add capacity alongside Deckard on model building and preprocessing. Project: onnx-genai, a Rust ONNX Runtime generative-AI inference runtime. Context: `onnx-genai-preprocess` is its own crate (image + audio); Mobius (`../mobius`) builds models — `build-gguf` (Q4 MatMulNBits), `--ep webgpu` (GQA), `--static-cache`; we emit our own `InferenceMetadata` (`inference_metadata.yaml`) not ORT-GenAI genai_config. Python builders use onnxscript/onnx-ir. Mobius PRs must pass `lintrunner` (RUFF + RUFF-FORMAT).

## 2026-07-13: Landed multi-tile VLM prompt expansion
Added the preprocessing-side prompt token-expansion library for multi-tile VLM inputs so vision token blocks can be expanded before generation. Landed as commit `9610b34`.

## 2026-07-13T20:55:00Z — Mobius emitter aligned to shared_kv + Gemma4 VLM scope
- Updated the Mobius onnx-genai emitter to emit canonical proposal_type: shared_kv (was gemma4_assistant). Tests 17/17 + 41/41 passing. Commit 498ecf0 on branch feat/gemma4-assistant-onnx-genai.
- Recorded Gemma4 multimodal (VLM) export as a major deferred effort: requires rank-3 pre-patchified vision ingestion, embedding→decoder orchestration (Gemma4 feeds inputs_embeds, not token IDs), and extended Mobius PR #398 pipeline topology. Adding two metadata fields alone cannot make the package load. Concrete values: image token id 258880, tokens_per_tile=280 (E2B).
