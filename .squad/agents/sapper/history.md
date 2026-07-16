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

## 2026-07-14T02:37:00Z — Mobius input_embedding durable export
- **Commit:** 2fed4f7 (mobius repo @ feat/gemma4-assistant-onnx-genai)
- Implemented `_find_scaled_token_embedding` (reads scale from graph's post-embed `Mul`, not hardcoded), `write_input_embedding_artifact` (raw f32 [vocab, hidden], 1.6 GB for E2B).
- `speculative.input_embedding` now emitted in YAML by default when target_model is supplied.
- Scale: graph f16 `39.1875` vs Leon's manual `sqrt(1536)=39.1918`; 1.1e-4 difference, negligible.
- 23 integration tests pass. Regenerated `~/gemma4-e2b-onnx/input_embedding.f32`.

- 2026-07-14T19:05:00Z — ITT tracer collector review by Joshi recorded GREEN for commit `977a50b`; unsafe prohibition, nesting, bounded domain lifetime, graceful degradation, feature hygiene, and all gates verified.

- 2026-07-15 — Added the Range Int64 addressability guard (merged `29f0772`).

## 2026-07-15T00:00:00Z — Cross-agent session update

- Closed RoPE checked-overflow and Range f32 parity fixes; canonical default-domain import merging also landed with loader validation.

## 2026-07-16T18:11:48+0000 — CUDA RMS FMA parity correction

- Merged `de3c556`: CUDA RMSNorm and SkipRMSNorm use separately rounded f32 multiplication and addition to match CPU serial reductions.
- Wallace 🟢 verified H200 coverage; exact native decode parity now reaches token 11, with token-12 MatMulNBits reduction order still open.

## 2026-07-16T19:05:18+0000 — CUDA SiLU and acc4 drift closure

- Merged `5c7dcc9`: matching CPU's fused-SiLU operation order and explicitly rounded acc4 scale boundaries eliminates token-12 drift; greedy CPU/CUDA parity now reaches token 15.
- The K=4864 `1.9073486e-5` reduction-order difference first diverges at token 16 and is accepted because exact GPU reduction emulation costs 8.4%. Wallace reviewed 🟢.

## 2026-07-16T19-27-57+0000 — Scribe session update

- Merged `67c1e3b`: shape inference for `BlockQuantizedMatMul` and `MatMulNBits` now returns `A.shape[..-1] + [N]`, unblocking unmodified real-model native E2E and the HTTP-server path.
