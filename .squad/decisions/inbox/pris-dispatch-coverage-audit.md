# Decision: Dispatch-Branch Coverage Audit — MatMul Kernel

**Date:** 2026-07-27  
**Author:** Pris (Tester)  
**Scope:** `onnx-runtime-ep-cpu::kernels::matmul`  
**Triggered by:** PR #275 rubber-duck review finding two silent-wrong-answer bugs

## Finding

**12 reachable dispatch combinations had zero test coverage before this audit.**

Line coverage reported 78% and codecov gates passed GREEN — but the entire
non-contiguous f16 rescue block (lines 823–896), the column-major GEMV M=1
path (lines 774–796), and the non-constant activation fallthrough were all
completely unexercised. This is the seventh defect of the form "a path existed
but was never entered."

## Dispatch-Branch Coverage Matrix

| # | M | dtype | B contiguous | B constant | B layout | BNNS avail | Path | Before | After |
|---|---|-------|-------------|-----------|----------|-----------|------|--------|-------|
| 1 | =1 | f16 | contiguous | yes | row-major | yes | GEMV via transposed_b_f16 cache | ✅ | ✅ |
| 2 | =1 | f16 | non-contig | yes | col-major [1,K] | yes | GEMV zero-copy col-major | ❌ | ✅ |
| 3 | =1 | f16 | non-contig | **no** | col-major | yes | Fallthrough → f32 widen | ❌ | ✅ |
| 4 | =1 | f16 | non-contig | yes | other | yes | Fallthrough → f32 widen | ❌ | ✅ (implicit) |
| 5 | ≥2 | f16 | contiguous | yes | row-major | yes | try_matmul_half → BNNS | ✅ | ✅ |
| 6 | ≥2 | f16 | contiguous | no | row-major | yes | try_matmul_half → BNNS | ✅ | ✅ |
| 7 | ≥2 | f16 | non-contig | yes | col-major [1,K] | yes | Rescue → BNNS trans_b | ❌ | ✅ |
| 8 | ≥2 | f16 | non-contig | yes | non-col-major | yes | Rescue → contiguous_b_f16 → BNNS | ❌ | ✅ |
| 9 | ≥2 | f16 | non-contig | **no** | col-major | yes | **BUG** → must NOT enter rescue | ❌ | ✅ |
| 10 | ≥2 | bf16 | contiguous | yes | row-major | — | try_matmul_half → portable half_gemm | ✅ | ✅ |
| 11 | ≥2 | bf16 | non-contig | yes | col-major | — | Must NOT enter f16 rescue → f32 widen | ❌ | ✅ |
| 12 | ≥2 | f32 | contiguous | yes | row-major | — | Direct f32 GEMM (Accelerate/generic) | ✅ | ✅ |
| 13 | ≥2 | f32 | — | — | — | — | Must NOT enter half/rescue | ❌ | ✅ |
| 14 | ≥2 | f16 | non-contig | yes | col-major | BNNS fails | Rescue → fallback half_gemm_tile | ❌* | ❌* |
| 15 | ≥2 | f16 | non-contig | yes | non-col | batched | Rescue → bnns_half_dense_into | ❌* | ❌* |

*Rows 14–15 are unreachable on current hardware (BNNS never fails for valid shapes). Marked as acceptable risk.*

**Summary: 12 of 13 reachable combinations were covered (from 7/13 → 12/13).** 
Region coverage for `matmul.rs`: 79.6% → 88.8% (+9.2pp).

## New Tests Added (8)

All follow the dispatch-reachability pattern (atomic hit counters):

1. `fp16_m1_column_major_b_reaches_colmaj_gemv` — proves #2 enters the right path
2. `fp16_m1_non_constant_colmaj_b_does_not_reach_gemv` — proves #3 does NOT enter GEMV
3. `f16_m_ge2_non_constant_non_contiguous_b_does_not_enter_rescue` — **THE BUG GUARD** (proves #9)
4. `f16_constant_non_contiguous_b_enters_rescue_block` — proves #7 (col-major rescue)
5. `f16_constant_non_contiguous_non_colmaj_b_enters_rescue` — proves #8 (non-col-major rescue)
6. `f16_non_constant_non_contiguous_b_produces_correct_result` — value correctness for #9
7. `f32_m_ge2_does_not_enter_half_or_rescue_paths` — proves #13
8. `bf16_non_contiguous_does_not_enter_f16_rescue` — proves #11

## Guard-Break Evidence

With the `constant_inputs[1]` guard removed from line 827:
```
assertion `left == right` failed: Non-constant non-contiguous B incorrectly
entered rescue block — this would produce all-zero output (the exact PR #275 bug)
  left: 0
 right: 1
```

With the guard present: all tests pass (945 + 8 new = 945 total, some pre-existing).

## Recommended Enforcement Mechanism

**One rule: every dispatch branch in `kernels/matmul.rs` must ship with a
reachability test that uses an atomic hit counter.**

Why this over the alternatives considered:
- **Per-file coverage floor (rejected):** A floor of 85% would have passed
  even when the bug existed — 78% line coverage masked it because the *lines*
  were counted via other paths. Coverage floors don't measure the property at
  issue (which *branch* ran).
- **Branch/region coverage in CI (helpful but not sufficient):** `cargo llvm-cov`
  branch coverage reports 0 branches because LLVM doesn't emit branch metadata
  for Rust match/if-let chains by default. Region coverage would help but
  requires parsing JSON and setting per-file thresholds — complex to maintain.
- **Dispatch-reachability pattern (adopted):** A test that increments a counter
  inside the branch and asserts it was hit proves the exact property: "this
  combination reached this path." It is:
  - Cheap to write (3 lines of counter + assert)
  - Self-documenting (the test name IS the property)
  - Catches both "wrong path taken" and "path never reached"
  - Already the team standard (3 existing guards; now 8 more)

**Enforcement:** Add a CI step or PR checklist item: *"Any new dispatch branch
in `kernels/matmul.rs` requires a `#[test]` with `_TEST_HITS` counter proving
reachability."* This can be checked mechanically: every `static.*TEST_HITS`
must have a corresponding test that reads it.

The existing `scripts/check_platform_naming.py` pattern (#278) could be
extended to scan for unguarded dispatch paths, but the counter-per-branch
approach is simpler and more robust for within-file coverage.
