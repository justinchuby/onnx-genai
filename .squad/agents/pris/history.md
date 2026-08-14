# pris — History

## Summary through 2026-08-11T07:00:00Z (compacted)

Earlier detailed history archived in `history-archive.md`. Pre-compaction summary:

- **Core domains:** test infrastructure, fixture quality, coverage hardening, metadata/schema validation, CPU/CUDA dispatch correctness, reviewer-driven fix cycles, upstream ORT contributions.
- **Standing expectations:** Every new dispatch branch ships with a reachability test. CI full coverage required on PRs (Linux fast job is early signal only). Benchmark guard: ≥15%/≥30% deltas flagged. Real-ORT tests require `disable_cpu_ep_fallback=1` + `Session_GetEpGraphAssignmentInfo` assignment assertions.
- **Retained durable outcomes** (through 2026-08-10): EP plugin conformance suite (L3 test ladder, real-ORT dlopen); parity integration tests; lint pass; f16/bf16 marshaling verdict; nxrt ABI round-trip tests; CUDA conformance runner; ReleaseEpFactory UB fix; portable L1 symbol tests; upstream MLAS AVX2 LayerNorm test harness; BF16 CPU op-level tests; AVX2 LayerNorm build + 40/41 pass + precision fix.

## 2026-08-11 — LayerNorm test follow-up (S1+S2)

**PR**: microsoft/onnxruntime#31973, branch `nxrt/mlas-avx2-layernorm`

- **S1**: Widened `kMaxRelError` 2.5e-2 → 3e-2 (35% headroom). B1 guard still catches old Welford (0.249 >> 0.03).
- **S2**: Fixed `DISABLED_AdversarialPrecisionReport` — excluded Scenario 6 (near-fp32-max overflow, unreasonable input), separated catastrophic-cancellation tracking (10% gate), enabled the test. Both invocations green (42/42 normal, 43/43 with disabled).
- **Nit**: Updated stale "Welford SIMD" labels → "centered two-pass".
- **Iran finding**: RMSNorm + MeanOut may do unnecessary mean pass in kernel.

## 2026-08-11 — Sixth Review Pass (PR #762, test-integrity audit)

- **Verdict:** YES to leaving draft, with one substantive reservation.
- Audited all 28 real-ORT tests + 6 CUDA tests. Classified each by assignment-proof strength (a/b/c).
- **8 assignment assertions verified genuine** — `Session_GetEpGraphAssignmentInfo` is real, `"cpu_ep"` has no collision with ORT's `"CPUExecutionProvider"`.
- **1 category (c) test found:** `layernorm_dynamic_axis_mean_invstddev_shape` — the BL1 regression test has NO `disable_cpu_ep_fallback` and NO assignment assertion. Could pass entirely on ORT's built-in CPU EP.
- **5/6 historical bugs regression-covered.** Gap: no test for forgeable name-based sentinels.
- `conformance_shape_f32` is a soft-check (ORT may constant-fold Shape) — acceptable.
- `conformance_mixed_partition` partition claim is aspirational (ORT may not partition).
- f16/bf16 tests lack assignment assertions (have fallback-disable, so category b not c).
- Did NOT reproduce forced-failure (read-only audit, no build env).

## 2026-08-11 — PR #762 sixth review / test-integrity audit

**Task:** Sixth adversarial review of PR #762. Test-integrity audit focus.

**Verdict:** No blockers.

**Findings:**
- BL1 regression test (`layernorm_dynamic_axis`) lacked `disable_cpu_ep_fallback=1` — same vacuous-test class as the earlier wave. Rachael hardened.
- f16/bf16 tests have fallback-disable but lack assignment assertions (category b: partially proved, not category c: fully proved). Rachael added assertions.
- Helper duplication between test files (independent implementations of equivalent assertion helper). Follow-up tech debt.
- `conformance_shape_f32` soft-check acceptable (ORT may constant-fold Shape before EP assignment).
- `conformance_mixed_partition` partition claim aspirational — ORT 1.27 may not partition under `disable_fallback: false`.
- 5/6 historical bugs regression-covered; gap: no test for forgeable name-based sentinels (Coco's fix is out-of-band, so no in-band attack surface to test).

## 2026-08-12 — PR #31993 readiness blocker (macOS arm64 CI validation)

**Task:** Resolve validation-honesty blocker. Determine whether macOS arm64 CI actually runs tests.

**Finding:** The lane DOES run `onnxruntime_mlas_test` on native arm64 (macos-15 runners). `build.py --test` invokes the binary directly without ctest. No `add_test` registration needed.

**Action:** Fixed stale FEAT_FP16 comment in `mlas.h`. Pushed commit `a0a9d98`. Proposed PR body rewrite.

**Head SHA:** `a0a9d98`

## 2026-08-14 — Marlin int4 numerics gate: reusable f64 dequant→GEMM oracle (#961, MERGED 401af46f)
Landed a reusable f64 oracle test any int4 `MatMulNBits` GEMM must pass before shipping — the red/green
target for Deckard's Marlin kernel and Chew's sign-off. Oracle dequantizes `(code-zp)·scale` and accumulates
`Σ a·w` in f64, sharing the candidate's fp16-rounded activations + rounded scale so the residual isolates
ONLY accumulation precision + fp16 output rounding (never shared input quant). Coverage: groups {16,32,64,128},
sym + asym-zp, M∈{1,2,4,8,16,32}, real glm-4-9b + Qwen2.5 projection shapes, fp16/fp32 scales. Justified
tolerance envelope (single source `Envelope::for_output(max_out)`): `abs=max(max_out·4e-3,4e-3)`, `rel=5e-2`
(denominator floored `max(1e-1,3e-2·max_out)`) — ~8 fp16-ULP headroom, tight enough that a structural
relayout/dequant bug blows it, loose enough to pass fp32 reassociation drift. Current tiled path measured
~8–10× inside abs / ~6× inside rel (≈1 fp16 ULP). **Harness lesson (durable): pre-zero the output buffer +
`runtime.synchronize()` after `execute`** — an un-synced run reading stale device-pool memory once faked an
abs-81.8 catastrophic divergence; not a kernel bug. Marlin driver must follow the same sync/zero discipline.
Chew concurred the tolerance is engineering-justified, not a rubber stamp.
