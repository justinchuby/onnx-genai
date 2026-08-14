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

## 2026-08-12 — Delta review PR #762 (commits 2106ac0..3826e11 + 8b3197e)

**Scope:** Focused delta on 7 test-integrity items closed by Rachael/Coco/Isidore.  
**Verdict:** Ready to leave draft. No blockers.

- Gate coverage: genuinely closed (all 4 panic conditions verified).
- `scratch_alloc_bytes`: single source of truth, canaries prove correctness.
- `validate_write_dtype`: dead in production (tests-only) — flagged as SUBSTANTIVE, not blocking.
- `find_ort_lib_dir`: 3 copies, one already drifted — flagged as SUBSTANTIVE follow-up.
- CUDA `i32::MAX`: safe for current op list; over-claim risk only on future additions.
- Vtable negative test: non-vacuous (both arms of the atomic flag exercised).
- Compile-time routing rejection: correct and conservative.

## 2026-08-12 — PR #762 delta review (focused, no blockers)

Focused delta review of 5 commits. Verdict: ready to leave draft. Two substantive (non-blocking): `validate_write_dtype` dead in production; `find_ort_lib_dir` had one drifted copy in `layernorm_dynamic_axis.rs`. Both addressed by Freysa. Gate coverage, scratch_alloc_bytes, CUDA i32::MAX, vtable test, compile-time routing: all genuinely closed.

## 2026-08-12 — PR #31973 evidence-accuracy review (focused delta)

Reviewed HEAD `fbf322f76b` — the evidence-accuracy fix commit. Built from clean, ran all three validation commands. Accuracy figures (B1 regression, sweep) reproduced exactly. Benchmark `nullptr` MeanOut fix confirmed against production code. One nit: RMSNorm ~3.3x at NormSize 256 is optimistic (measured ~2.84x). NormSize 15 RMSNorm "~0.83x regression" measured as 1.00x — body is conservative, adequate. No persona leaks. 41/2/43 test counts confirmed. Verdict: ready to leave draft, no blockers.

## 2026-08-12 — PR #31973 evidence-accuracy review (focused delta)

Focused evidence review of Mariette's B1+B2 fixes (HEAD `fbf322f76b`). Reproduced all four accuracy figures to 4 significant figures. Confirmed `nullptr` MeanOut matches production. Dispatch assertion fires. No stale claims found. Test counts 41/2/43 confirmed on fresh build. One nit: RMSNorm ~3.3x at NormSize 256 is ~14% above measured ~2.84x; body says ~0.83x at NormSize 15, measured 1.00x (body is conservative). Coordinator widened variance disclosure to ~15%. Verdict: ready to leave draft, no blockers.

## 2026-08-14 — PR #960 Marlin int4 M>1 GEMM quality/portability review 🟢 APPROVE
Reviewed Deckard's Marlin M>1 GEMM for quality / Rule 11 portability / capture-safety / build hygiene (numerics
= Chew; the B\*=2.16 plateau is an accepted honest plateau, not a defect). Frozen head `a11facbc` (code
`3735d57e`). **Zero blocking defects.** Rule 11 PASS: genuinely opt-in (`ONNX_GENAI_MARLIN_M_GT_1` default OFF,
checked first at all three M>1 dispatch seams), SM80 guard correct (`device_supports_marlin` ANDed at every call
site — even a force-set env var on <SM80 falls through to the portable tiled GEMM), and Marlin-OFF is provably
byte-identical to the prior tiled path; any runtime ineligibility/launch error also falls through. Env-var honesty
PASS: both knobs read & wired (`verify_documented_env_vars.py` EXIT 0). Capture-safety valve family sound: four
caches share one contract — cache-hit → warm (no alloc/sync ⇒ capture-safe); cold miss while `is_capturing()` →
return Err so caller falls back, never alloc inside capture; pre-warm forward populates them. GQA M>1 Fused flash
path + SkipLN latch relaxation both verified capture-safe (on-device totals, no host read-back, shape-keyed cache
rejects mid-capture shape change). `cargo fmt` + the exact CUDA and engine `-D warnings` clippy gates clean; all
`unsafe` blocks carry SAFETY justifications. One trivial `cfg(test)`-only `clippy::unusual_byte_groupings` note
(`0xcafef00d_1234_5678u64`) — outside the CUDA CI gate (no `--all-targets`), does not break CI, trivially fixable.
Rule reinforced: env-var honesty + a byte-identical default-OFF fallback are the portability contract for any
tier-scoped kernel.
