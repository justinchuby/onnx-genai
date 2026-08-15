# Sapper — History (compacted 2026-08-11T23-30-00Z)

**Role:** Systems/model-building implementer for onnx-genai and Mobius export/preprocess work. Owns native CUDA/CPU EP correctness and model-package metadata details; must preserve real-model parity, capture safety, Mobius lintrunner hygiene, and reviewer lockouts.

## Durable lessons
- onnx-genai uses its own `InferenceMetadata` (`inference_metadata.yaml`), not ORT-GenAI `genai_config`; Mobius PRs must pass lintrunner (RUFF + RUFF-FORMAT).
- CUDA RMSNorm/SkipRMSNorm parity requires separately rounded f32 multiply/add; CUDA SiLU and acc4 scale boundaries need CPU-matching operation order/rounding.
- Reviewer lockouts remain binding; do not revise an artifact you authored after reviewer rejection.
- CUDA graph/kernel work must stay capture-safe and portable across supported SM architectures.
- WP-B optional fallback validation treats raw `GraphProto.input` as authoritative.
- A self-report of "all defects fixed" is not evidence: Gaff found three blockers (UAF, pointer equality, direction classification) after Sapper's claim of completion on B4.

## Historical context

Pre-2026-08-11 entries archived in `history-archive.md`. Covers: Mobius PR triage, rewind policy split (PR #291, RewindRequest), Wave 5 CUDA ops (PR #331), B2 ReleaseEpFactory ABI fix, CUDA B4 implementation defects (REJECTED by Gaff — UAF, pointer equality, direction classification), B2 follow-up CPU shim fix, B2/B3 docs/OperatorKernels.md + leakage sweep (#31974), CUDA B4 REJECTED.

## 2026-08-11 — Rebase PR #31974

Rebased `nxrt/mlas-bf16-layernorm` onto `86d38813a8` (no conflicts). Build+test green: 17 BF16 tests, 96 LayerNorm suite, `-Werror` clean. Force-pushed to `5755a8a129`. PR remains draft.

## 2026-08-11 — Rebase PR #31974 (semantic conflict with #31676)

Upstream landed `a29da16687` (Validate SkipLayerNorm prepacked lengths) which conflicted in `skip_layer_norm.cc`. Two hunks: (1) competing include additions — kept both; (2) upstream's `tensor_size > 0` guard vs our bf16 branch — took both, extended guard to bf16 path. Upstream's shape validation covers bf16 because `ConvertMLFloat16ToFloatIfNeeded` now handles bf16 and sets `is_packed`. All 5 properties preserved. Tests: 17 bf16, 103 LayerNorm suite, 6 upstream prepacked-validation tests — all green. Force-pushed to `71bc68a41b`. PR remains draft.

## 2026-08-11 (upstream CI correction wave) — Session append

Both upstream PRs converted back to draft per user instruction. Rebase outcomes confirmed stable: no code changes needed. Lessons: "not caused by us" ≠ "safe to mark ready"; draft until CI board is green.

## 2026-08-12 — PR #31974 PrePack A/B

Converted `LayerNorm17_PrePack_ScaleBiasInitializers` and `SkipLayerNorm_PrePack_GammaBetaInitializers` from single-config (`is_initializer=true`) to A/B loop (`{false, true}`). Both configs assert against the same reference output, proving PrePack does not change results. 20 BF16 / 106 LayerNorm tests green. Head: `e053afd77e`.

## 2026-08-12 — PR #31974 PrePack A/B + PR body correction

**PrePack A/B:** Converted `LayerNorm17_PrePack_ScaleBiasInitializers` and `SkipLayerNorm_PrePack_GammaBetaInitializers` to A/B loop (`is_initializer ∈ {false, true}`). Both configs assert against the same `LayerNormRef` reference. 20 BF16 / 106 LayerNorm-suite tests green. Head: `e053afd77e`.

**PR body correction (Coordinator):** PR #31974 also changes pre-existing MLFloat16 behavior: (1) `REGISTER_CONTRIB_KERNELS(MLFloat16)` → `REGISTER_CONTRIB_KERNELS(MLFloat16, float)`, (2) MLFloat16 `ComputeJob` overload uses `WriteStat<U=float>` instead of `MLFloat16(mean)`. PR body rewritten to disclose both explicitly; stale "45 MLAS tests / 10 operator tests" table removed. Awaiting CI.
