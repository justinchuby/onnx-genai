# Gaff — History (compacted 2026-08-12)

## Project context
- Review specialist for onnx-genai correctness, runtime/loader boundaries, transactional semantics, and validation quality.
- Joined 2026-07-12 after phases 1-4, tool use/grammar/chat-template, Qwen2.5-0.5B, Hermes E2E, and static-cache KV work were established.

## Condensed prior record through 2026-08-11
- Reviewed and approved multiple ORT2 loader, shape-inference, fused-domain, EPContext, C-ABI, external-data, and conformance changes after checking byte fidelity, dispatch invariants, path confinement, FFI behavior, and model-backed tests.
- Used reviewer lockout discipline on real blockers: unauthenticated debug exposure, MatMul+Add fusion shape guard gaps, duplicate EPContext primary identity over-rejection, unsupported-op user-facing opset leakage, and thread/benchmark provenance issues.
- Reviewed env var verifier false-positive, #31973 AVX2 LayerNorm, multiple rounds of PR #762 CUDA EP.
- Helped consolidate performance and CUDA/native guidance; benchmark comparisons must be matched and reproducible.
- CLI is a development/maintainer harness; `docs/research/cli/00-backlog.md` is the backlog source of truth.
- PR #762 lockout chain (Sapper rejection → Nabil B1/B3/S4 fix → conditional pass → final ready) verified all four B items. Verify API absence before deferring — MemoryDevice_GetDeviceId existed in ORT 1.27 bindings.

_Pre-2026-08-11 detailed dated entries archived to `history-archive.md`._

## 2026-08-11 — PR #762 final scoped delta review

**Task:** Final delta review of commits `c1d2556b5` through `bb280c0ea`.

**Verdict:** No blockers. Ready.

- EP name `"cpu_ep"` originates from `provider.rs:120`, distinct from ORT's `"CPUExecutionProvider"`.
- `disable_cpu_ep_fallback=1` set before session creation in both test files.
- `Session_GetEpGraphAssignmentInfo` + subgraph/node iteration logic correct in both copies.
- BL1 shape assertions intact: mean/inv_std `[2, 3, 1]`, values `[2.5, 6.5, 10.5, 14.5, 18.5, 22.5]`.
- No CUDA hardware claims in docs.
- Substantive finding: helper duplication in test files — tech debt for follow-up.

## 2026-08-11 — Review of microsoft/onnxruntime #31988 (initial)

**Commit:** a4aa076657 | **Verdict:** SUBSTANTIVE — keep as draft

- Bit-identicality: confirmed by tracing kernel warp assignment and reduction tree.
- Routing invariance: confirmed — guard at line ~763 preserves n%8 requirement; pinning test non-vacuous.
- Wide-n invariance: safe — threshold arithmetic correct, no overflow risk.
- All 3 instantiations reachable; failsafe adequate; no persona leaks or perf claims.
- One nit: "fills the target" comment misleading at very small n.
- Recommendation: keep as draft until GPU benchmarks available.

## 2026-08-12 — Fresh review of microsoft/onnxruntime #31988 (post-Chew guard, a4aa076657)

**Verdict:** SUBSTANTIVE — keep as draft. No blockers.

- Bit-identicality, routing invariance, wide-n invariance: all confirmed.
- All 3 instantiations reachable; failsafe adequate.
- Recommendation: keep draft until GPU benchmarks on ≥2 GPU generations.
