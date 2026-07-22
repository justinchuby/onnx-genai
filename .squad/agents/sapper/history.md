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

## 2026-07-16T23:30:00+0000 — GAFF control-flow loader foundation

- Merged `2a9e5b1`: graph attributes retain ordered typed formal I/O and scoped inline initializers, including UNDEFINED graph attributes and nested subgraphs.
- Leon 🟢 cleared the isolated scopes and Loop regression; child-executor and If/Loop/Scan execution remain next.

## 2026-07-16T23:58:29+0000 — GAFF ChildExecutor foundation

- Landed the recursive `ChildExecutor` foundation: lazy compile/cache by signature, lexical captures, scoped inline initializers, and nested child scopes; 114 session tests passed.
- Next: implement `If` branches keyed by `(node_id, branch)`; expand cache storage beyond the current last-signature plan.

## 2026-07-17T00:19:41+0000 — GAFF If execution

- Merged `7a369ef`: ONNX `If` selects validated BOOL branches through separate `(node_id, branch)` cached ChildExecutors with fresh lexical captures and positional output binding.
- Holden 🟢 verified alternating branches/capture freshness; 117 session tests passed. Loader → ChildExecutor → If is complete; Loop/Scan and multi-signature caching remain.


## 2026-07-17T00:58:13Z — GAFF Loop review handoff

- Initial Loop implementation `8052891` was 🔴 rejected by Holden: scan accumulation eagerly reserved from untrusted `M`, enabling an early-exit `i64::MAX` capacity-overflow DoS, and carried shapes were not validated.
- Sapper was locked out of the revision; Leon owned the remediation. The final Loop revision was cleared and merged as `f6e8ba6`; `Scan` remains the final control-flow op.

## 2026-07-14T00:00:00Z — GAFF Scan complete

- Implemented ONNX Scan through ChildExecutor; Leon’s checked stack-arithmetic repair and Holden’s approval closed the final control-flow-op gate. If + Loop + Scan are now complete.

## 2026-07-17T07:19:39Z — onnx-rs multi-device/sharding proto landing

- Delivered `be68145`: seven IR13 device/sharding protobuf messages with Model/Node wiring, checker and codec round trips, and `docs/ONNX_RS_SPEC_COVERAGE.md`.
- Deckard's `b5ccd3c` correction made `SimpleShardedDimProto.dim` optional; Bryant 🟢 approved. Remaining parity gaps are in flight.
- 2026-07-19: Landed BQMoE v1 CPU parity oracle and frozen ABI (`7f31162`).
- 2026-07-19T07:55:00Z: CSA Phase B B0 device-state/stage-dispatch scaffolding merged at `9c56d9c` after numerical correction.

## 2026-07-19T07:42:20Z — CSA Phase B B3 landed

- Landed device ratio-4 FP4 index-key compression with device-resident index cache/carry in `3ae3244`. Chew approved; 15/15 H200 GPU parity tests are bit-exact.

## 2026-07-19T07:42:20Z — CSA B7 nit follow-up

- Assigned Chew's non-blocking B7 nits: completed-block rollback boundary coverage and the five-output ratio-4 host metrics mode label. Sapper is closing both as a follow-up.


- **2026-07-19T16:15:00Z — Activation precision fix:** Added true f64 Selu/ThresholdedRelu computation and precision-sensitive tests; activations landed as `39edb76` after Luv approval.


## 2026-07-19T18:20:00Z — CPU-EP op coverage 936→975

- Fixed OneHot out-of-range indices to all-off and required BitShift direction (`49d8827`); Gaff approved the correction.


## 2026-07-19T20:10Z — CPU-EP op coverage Batch 4

- Revised Deckard's rejected GridSample artifact: opset 16 now rejects rank-5 input while opset 20 retains 2-D/3-D support (`61d5e63`, landed in `9c250c6`); Gaff approved.

- 2026-07-19: Kernel Criterion microbenchmarks, ten numeric regressions, and optional ORT baseline landed as d89a47e; thread-matched follow-up established a ~17–21× medium-f32 MatMul gap. Removed unsafe Unique String execution so the safe numeric/bool/bf16 kernel could land.


### 2026-07-20 — Vendored MLAS CPU-GEMM parity

Recorded MLAS ep-cpu integration (`d696b7a`) and the provenance README correction (`ee7a6cd`).


## 2026-07-20T07:15Z — Bounded M=1 decode pool

- Landed `d7a0819`: defaulted M=1 MatMulNBits decode to a bounded 8-thread Rayon pool, improving 18.75→23.46 tok/s (+26%); luv-33 reviewed 🟢.


## 2026-07-20T13:35:00Z — Multistream performance and issue #40

- Landed persistent M=1 CPU decode-pool residency (`cbacb75`), avoiding repeated pool installs for a bit-identical +3–6% decode gain; Luv approved.

## 2026-07-21T03:15:00Z — CUDA graph M4 validated
- Made supported unary/binary elementwise decode paths capture-safe with persistent broadcast metadata; Deckard's exact-signature hardening completed the landed artifact (`85b6f4e`).

- 2026-07-21: Scribe reconciled the perf campaign inbox; key decisions are now consolidated in `.squad/decisions.md` under the 2026-07-21 perf campaign section.


## 2026-07-21 — Wave-2 and CI milestone
CI now covers all 27 offline crates with warnings-as-errors and native Windows ARM64. Capture-safe native fp16 CUDA decode wave 2 stacked GQA prep fusion, warp-shuffle RMSNorm, and specialized down-projection GEMV on wave 1, reaching 663–672 tok/s on H200 versus ORT GenAI at 657, with zero fallbacks. All CUDA EP kernel work must remain correct and fast across supported SM architectures, not only sm_90.
- 2026-07-21T23:55Z — VLM WP0 metadata, WP3 generic every_step executor, and WP2 correction path were captured; WP2 landed after Pris approval.
## 2026-07-22T00:00:00Z — BLOCKER #3 explicit int4 zero-points merged

- Sapper fixed native CUDA fp16 int4 GEMV explicit `zero_points` support for asymmetric Phi-4-mini-style MatMulNBits, preserving symmetric zp=8 fast paths and M==1 capture safety.
- Holden completed the non-author review 🟢 (SM-portability, capture-safety, symmetric no-regress, genericity, correctness); merged to main as `48de993`.

### 2026-07-22T14:59:36+0000 — WP-B landed
WP-B landed: Sapper's WP-B3 v3 admission fix landed at `3d84b9b`, making raw `GraphProto.input` authoritative for optional fallback validation.
