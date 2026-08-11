# Chew — History (compacted 2026-07-29)

**Role:** Numerics/precision reviewer. Require reference-backed coherent outputs rather than mere execution success, and guard dtype/layout symmetry, silent coercions, opset semantics, broadcast behavior, stable reductions/softmax, and realistic parity tests.

## Durable lessons
- Review standard: coherent reference-backed outputs beat successful execution; dtype/layout symmetry, opset semantics, broadcast, stable reductions/softmax, and realistic parity tests are mandatory.
- Connector/KV work must preserve cache separation, byte-layout symmetry, prefix-dependent hashing, fetch/recompute boundaries, per-layer heterogeneous geometry, and graceful recompute fallback.
- Original contrib FusedMatMul shape rule ignored transpose attributes; Chew rejected it and Deckard's corrected rule is canonical.
- ONNX dtype decoding must fail closed; never silently fall back to Float32.
- Fusion tolerances are distinct from conformance tolerances and must not be loosened; LayerNorm needs axis-as-input, epsilon-type, and operand-order decline guards.
- EPContext cannot fall through to CPU execution; payloads remain byte-exact, FFI is null/UTF-8/panic guarded, and disabled export must be side-effect-free.
- CSA B5's five-output ratio-4 dispatch bug was a real misroute to ratio-128; Roy's ratio-keyed fix/regression is canonical.
- CPU reduction axes semantics distinguish omitted axes from present-empty axes; Deckard's fix after Chew rejection is canonical.
- Fused QMoE must not clobber `_group_topk_selection`; grouped routing requires original signature and group-mask behavior.
- CUDA graph capture reviews require real replay coverage, exact signatures, detect-before-consume poisoning, and guard-break proofs, not just smoke success.
- WP-B loader-IR shape authority rejection directly informed Sapper's final WP-B3 v3 fix.
- PR #227 SiLU polynomial measured ~28 ULP, not ~1 ULP; the docstring claim was wrong though numerics were acceptable for inference. NEON SDPA dispatch had zero path coverage until follow-up.
- PR #334 formatting failures are review-blocking even when numerics are sound; Iran was the revision agent after rejection.
- BNNS grouped/depthwise convolution via deprecated API is genuinely broken for groups > 1; guard is justified, but im2col is only a correct intermediate step and direct NEON depthwise should target 2–3× ORT.
- Documentation rationales are correctness artifacts: wrong L1-cache premises and derived-looking fitted constants must be corrected before merge.

## Recent work (current wave, ~2026-07-28/29)
## 2026-07-27T01:30:00-07:00 — PR #227 CPU EP NEON numerics review

- **APPROVE with concerns** for Iran's 4-commit CPU EP optimization branch (`squad/mac-cpu-ep-roofline`): NEON SiLU, SDPA, GEMV, Accelerate sgemm, dtype fast path.
- SiLU polynomial: measured ~28 ULP in practical range (claimed ~1 ULP — docstring incorrect). Acceptable for inference.
- Swish→SiLU canonicalization: exact f32 equality correct, no silent misroute.
- SDPA NEON: numerics sound (softmax max-subtraction stability inherited), but zero test coverage for the NEON dispatch path — all tests call scalar reference directly.
- GEMV: transpose correct, tail handling correct, f32 accumulation throughout. Guard-break test passed.
- dtype.rs f32 memcpy: contiguity guard is sound.
- matmul_nbits.rs: visibility change only, safe.
- All NEON intrinsics are ARMv8 baseline. No hardcoded cache/thread counts.
- 7 dead code items from removed Accelerate sgemv path.
- Filed to `.squad/decisions/inbox/chew-pr227-numerics-review.md`.

## 2026-07-27T02:00:00-07:00 — PR #227 FP16 Path Review (Second Pass)

**Scope:** Commits `75311827` (FP16 storage GEMV + NEON bulk f16↔f32) and `3a88ba8c` (SPMD pool for FP32 GEMV + cleanup).

**Verdict: APPROVE** — numerics are sound.

### Key findings
- **Inline asm `fcvtl`:** Constraints, clobbers, and options are correct. Bit-exact against scalar `half::f16::to_f32()` across all edge cases (denorm, inf, NaN, ±0). Using asm to avoid nightly `f16` type is justified today. Recommend TODO for intrinsic replacement.
- **FP32 accumulation verified:** Measured max relative error 2.38e-7 vs f64 reference across model-scale shapes (gate/down/q/kv projections). FP16 vs F32 GEMV same-weight discrepancy: 1.73e-6 — confirms accumulation is genuinely f32.
- **Bulk conversion:** `fcvtn` narrow matches `half::f16::from_f32()` bit-for-bit. Round-to-nearest-even confirmed. Overflow → inf. Denormal/NaN preserved. Asm annotations correct (`nostack` only for write path, `readonly,pure` for read path).
- **Tail handling:** K=67, N=9 correct. K=1/N=1 correct.
- **Transpose cache:** `OnceLock` provides thread-safe lazy init. Rayon `par_chunks_mut` writes to disjoint slices.
- **SPMD pool:** `perf_cores.saturating_sub(1).max(1).min(available)` guarantees ≥1 worker. `None` fallback on Intel/VM is correct.
- **Tests:** 922 passing (906 lib + extras). 3 new FP16 GEMV tests + 1 updated cache test.
- **Non-blocking concerns:** C1 = add TODO for intrinsic migration; C2 = tighten test error thresholds (2% → 1e-4).
- Filed to `.squad/decisions/inbox/chew-pr227-fp16-review.md`.

## 2026-07-28T00:40:00-07:00 — PR #334 Grouped/Depthwise Conv Review

- **REJECT** (formatting) for Deckard's depthwise conv im2col+GEMM PR.
- `cargo fmt --all -- --check` fails with 3 violations — same class as #324.
- **Numerics: SOUND.** Grouped im2col indexing is correct across all 8 parity tests (true depthwise, grouped-not-depthwise, channel multiplier, non-SIMD-width channels, stride>1, dilation, asymmetric padding). Guard-break test detects off-by-one immediately.
- **BNNS claim independently verified:** Probed `BNNSFilterCreateLayerConvolution` directly via FFI. With `groups > 1`, BNNS either returns NULL (oc_per_group in descriptor) or accepts but only writes group 0's output (full oc mode). The deprecated API is genuinely broken for groups > 1. Guard is justified.
- **Fall-through:** No #275 pattern. Both paths produce fully-populated output vectors.
- **Non-grouped path untouched** (byte-identical except defensive n==0 guard).
- **Reachability:** Counter `CONV_IM2COL_GEMM_TEST_HITS` covers both branches, manifest claim present, test genuinely forces grouped path.
- **12× gap judgement:** im2col is structurally wrong for depthwise (memory-bound, K=9, M=1). Direct NEON kernel would be 4–8× faster (eliminates im2col buffer entirely). This PR is a correct intermediate step. Schedule NEON depthwise follow-up targeting 2–3× ORT.
- **Revision agent:** Iran.
- Filed to `.squad/decisions/inbox/chew-pr334-review.md`.
- 2026-07-28: Reviews of PR #347 and #349 approved after verifying numerical bounds and real decode firing. Documentation rationales are reviewable correctness artifacts: wrong L1-cache premises and derived-looking fitted constants must be corrected before merge.

Full pre-compaction history in `history-archive.md`.
- 2026-08-11: **AVX2 LayerNorm two-pass vs Welford precision audit** (PR #31973 on microsoft/onnxruntime). Built adversarial numeric tests in `test_layernorm.cpp`. Key finding: two-pass `E[x²] - mean²` suffers catastrophic cancellation when mean is large and variance is small (inv_std_dev error = 100% at base=1e7). Realistic LLM activations are fine (≤3e-05 rel error). Verdict: **two-pass is not acceptable for full LayerNorm; recommend Welford-preserving SIMD**. RMSNorm unaffected. 3 passing tests added + 1 DISABLED comparison report. All 39 tests green. Filed to `.squad/decisions/inbox/chew-layernorm-numerics.md`.
- 2026-08-11: **Test contract update for Welford SIMD + NormSize<8 threshold** (same PR). Resch replaced two-pass with Welford-preserving SIMD and added dispatch threshold. Fixed 9 failing tests by encoding conditional dispatch contract: N≥8 → ASSERT_TRUE(used), N<8 → ASSERT_FALSE(used) with scalar fallback verification. Skip mean check for RMSNorm (not part of contract). Added committed `CatastrophicCancellationPasses` test asserting no NaN/Inf and exact parity with scalar Welford. Re-measured: Welford SIMD is 0.1–0.8× the error of scalar Welford (ratio < 1.0 everywhere). Worst output error 1.34e-05 at N=16384. Updated labels from "two-pass" to "Welford SIMD" throughout. All 40 tests green, 0 failures.
- 2026-08-11: **BF16 LayerNorm/RMSNorm precision oracle** (branch `nxrt/mlas-bf16-layernorm`). Created `test_layernorm_bf16.cpp` (45 tests). Validated ORT's BFloat16 RNE rounding (336/336 tie-to-even, 672/672 directional). Measured bf16 representation-error floor: max_abs ≈ 3.9e-3 (0.5 ULP), max_ulp=0 across all N up to 65536. Simulated widen→f32-accumulate→narrow: ≤1 bf16 ULP above floor even at N=65536. BF16/FP16 ratio = 8.0× as expected (8 vs 11 mantissa bits). Catastrophic cancellation shape from #31973 does NOT apply to bf16 (coarse quantization makes variance relatively large). Recommended kernel tolerance: ≤2 bf16 ULP. Verdict: **ACCEPT** widen-accumulate-narrow approach. Kernel hook deferred until Resch's API lands. Report in `.squad/decisions/inbox/chew-bf16-numerics.md`.

## 2026-08-11 — PR #31973 clarity pass (nxrt/mlas-avx2-layernorm)

**S2 — ReferenceLayerNorm two-pass comment** (`test_layernorm.cpp` lines ~44–76):

Added an explanatory block comment above `ReferenceLayerNorm` covering two points:

1. *Why two-pass is safe in fp64.* At float32 magnitudes the subtracted terms `E[x²] - mean²` differ by at most ~2⁵³ ULPs in fp64, so cancellation error stays inside fp64's dynamic range and the result is accurate to single precision. The catastrophic cancellation that produced NaN / 100% error in fp32 does not occur here.

2. *Why two-pass is deliberately kept rather than switched to Welford.* An independent algorithm in the reference cross-checks the kernel rather than repeating its logic. If both used Welford, a shared conceptual mistake (wrong initialisation, off-by-one in the running count) could produce identical wrong answers without being caught. The algebraic equivalence of the two formulas, combined with fp64 safety, makes two-pass the right oracle here. I left the reference on two-pass and noted explicitly that it should not be "fixed" to Welford.

**N1 — dead `(void)0;` at line ~695:** Removed.

**clang-format:** Run on `onnxruntime/test/mlas/unittest/test_layernorm.cpp`.

**Validation:** 40/40 LayerNorm tests pass.
```
[==========] 40 tests from 3 test suites ran. (4 ms total)
[  PASSED  ] 40 tests.
```

## 2026-08-11 — #762 Test Repair (Corrective Wave)

All four blocker test repairs completed:
- B1: Fixed compile errors (output_dtype→output_dtypes), added 4 dtype conformance tests (Cast, Where, Shape, LayerNorm). LayerNorm `#[ignore]`d — found real shape inference bug (owner: Batty).
- B2: Restored correct `ReleaseEpFactory -> *mut OrtStatus` signature + null assertion.
- B3: Fixed `*const i8` → `*const c_char` portability, added message_str() cdylib boundary assertion.
- B4: Created `cuda_fail_closed.rs` with 3 tests asserting zero factories and error status.

All suites green:
- `onnx-runtime-ep-plugin`: 154 passed, 0 failed
- `onnx-runtime-ep-cpu-plugin`: 20 passed, 1 ignored (LayerNorm shape bug), 0 failed
- `onnx-runtime-ep-nxrt-abi`: 32 passed, 0 failed
- `onnx-runtime-ep-nxrt-host`: 10 passed, 0 failed
- `onnx-runtime-ep-cuda-plugin`: 6 passed (3 unit + 3 integration), 0 failed
- clippy: clean (warnings as errors)
- fmt: clean

---

## 2026-08-11 — PR #762: LayerNorm test hardening

**Task:** Remove `#[ignore]`, strengthen `conformance_layer_norm_multi_output`, add new coverage.

**Files changed (tests/ only):**
- `crates/onnx-runtime-ep-cpu-plugin/tests/plugin_ort_e2e.rs`
- `crates/onnx-runtime-ep-cpu-plugin/tests/fixtures/generate_fixtures.py`
- `.gitignore` (two new `!` negations)
- New fixture dirs: `layer_norm_neg_axis_f32/`, `simplified_layer_norm_f32/`

**Actions:**
1. Removed `#[ignore]` and stale bug comment from `conformance_layer_norm_multi_output`.
2. Added `assert_output_shape` unsafe helper (ORT GetDimensions API).
3. Strengthened existing test: explicit shape assertions Mean=[2,1], InvStdDev=[2,1]; invstd value check.
4. New test `conformance_layer_norm_neg_axis`: 3D [2,3,4] input, axis=-1, Mean/InvStdDev [2,3,1] asserted with values.
5. New test `conformance_rms_norm`: RMSNormalization (opset 23) single-output, shape [2,4], rms(Y_row)≈1.0.
6. Fail-closed decline path: cannot be tested from a valid ONNX fixture. Noted in decision doc.

**Results:**
```
cargo test -p onnx-runtime-ep-cpu-plugin → 23 passed; 0 failed; 0 ignored
cargo test -p onnx-runtime-ep-plugin     → 9 passed; 0 failed; 0 ignored
clippy -D warnings                       → clean
cargo fmt --check                        → clean
```

### 2026-08-11 — PR #31974 CI fix: unused functions

**Problem:** 46 CI jobs failing on `-Werror=unused-function` in `test_layernorm_bf16.cpp`.
**Fix:** Removed two dead static functions: `BF16Ulp` (scaffolding, not needed — tolerances use `BF16UlpDistance`) and `ReportErrors` (uncalled private method).
**Sweep:** All 8 PR-touched files checked — no other `-Werror` issues.
**Validation:** `build.py --config Debug --target onnxruntime_mlas_test` (no `--compile_no_warning_as_error`) → clean build. 45/45 BF16 MLAS tests pass.
**Lesson:** Never validate with `--compile_no_warning_as_error` — it masks the exact class of failure that blocks upstream CI.
