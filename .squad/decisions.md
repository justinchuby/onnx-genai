# Decisions

Canonical, append-only record of accepted team decisions. Only the Coordinator (via Scribe merge) writes here. Agents drop proposals in `decisions/inbox/`.

---

# CUDA standard Attention / RoPE review — 2026-07-17

## 🔴 REJECT

Host staging is an acceptable correctness-first slice: the f32 compute loops have a fixed sequential reduction order, use no floating-point atomics, and inherit the `Kernel` default `cuda_graph_compatible() == false`. The CUDA integration test also asserts byte-identical repeated results.

Two contract defects block this claim:

1. `StandardAttentionKernel` and `RotaryEmbeddingKernel` return `true` from `supports_strided_input()` while their staging readers (`dense_bytes` / `read_bytes`) explicitly reject every non-contiguous input. The executor may therefore pass a strided tensor instead of inserting a contiguous conversion, and the newly claimed CUDA op fails at runtime. Return `false` for all inputs, or implement stride-aware host materialization. Add an integration test with a strided Q/mask/cache and RoPE input.
2. CUDA `supports_op()` claims these registrations solely from the op registry; it has no newly added deny path for f16/bf16. Those tensors are claimed and subsequently fail in `execute()` with `KernelFailed`, not `KernelMatch::Unsupported { reason }` as required. Add dtype-aware claim validation (or the appropriate input-value metadata plumbing) that cleanly declines f16/bf16 with an actionable reason, plus a coverage-diagnostic test.

Suggested different fixer: **Deckard** (Systems Dev), with Pris adding the negative/strided coverage.

## Verification

`cargo test --locked -p onnx-runtime-ep-cuda` passed: all CUDA EP tests green, including the two new deterministic integration tests.

## Non-blocking follow-ups

- Route standard Attention/RoPE through a device SDPA/NVRTC path and add f16/bf16 support; host staging has substantial transfer/performance cost.
- Broaden GPU parity tests to additive masks, past/present decode, explicit scale, softcap, qk-output modes, and both RoPE layouts without position IDs.
- `git diff 712b1b1..ea6c61e` also contains an unrelated deletion of `docs/GRAPHVIEW_LENS_DESIGN.md`; restore or separately account for it before integration.

<!-- Source: apone-custom-shape-handlers.md -->
### 2026-07-17: Custom-op shape handlers for GLM/DeepSeek exports
**By:** Apone
**What:** Added shape inference for:
- `com.microsoft::MoE` and `QMoE`: one output, identical dtype/shape to activation input 0. Matched `crates/onnx-runtime-ep-cpu/src/kernels/moe.rs` and `qmoe.rs`.
- `com.microsoft::GatherBlockQuantized`: Gather shape `data[:gather_axis] + indices + data[gather_axis+1:]`, output dtype from scales; packed `uint8` expands the quantize-axis extent by `8 / bits`. Matched upstream ORT CPU `onnxruntime/contrib_ops/cpu/quantization/gather_block_quantized.cc` because this repository does not yet contain its CPU kernel.
- `pkg.nxrt::SparseKvGather`: `[B,G,C,D]` cache plus `[B,G,Q,K]` indices produces `[B,G,Q,K,D]`. Matched `crates/onnx-runtime-ep-cpu/src/kernels/sparse_kv_gather.rs`.
- `pkg.nxrt::CompressedSparseAttention` and the requested `com.microsoft` shape alias: Y preserves `[B,S,H,D]`; present cache is `[B,floor(total/compression_ratio),stored_width]`; ratio-128 carry is `[B,128,2,D]`; ratio-4 carry is `[B,8,2,2D]`, index cache is `[B,records,fp4_width(index_dim)]`, index carry is `[B,8,2,2*index_dim]`, and optional selected indices are `[B,index_heads,S,min(index_topk,records)]`. Runtime-dependent record counts become deterministic fresh symbolic dimensions. Matched `crates/onnx-runtime-ep-cpu/src/kernels/compressed_sparse_attention.rs`.

**Why:** The loader/compiler needs every custom-op output typed and ranked, especially CSA's frozen present-state tensors. No requested operator was left unregistered.

<!-- Source: gorman-shape-handlers-review.md -->
# 🔴 REJECT — custom shape handlers review

Reviewed `csa-shapes` commit `82456ed` against `origin/main` (`712b1b1`).

## Blocking defect

`GatherBlockQuantized` is registered with a guessed shape contract
(`custom_ops.rs`, `gather_block_quantized`): it assumes packed `Uint8` input
data expands the quantization axis by `8 / bits` and that output dtype is
input 2's dtype.  There is no `GatherBlockQuantized` CPU kernel or prior
implementation anywhere in this repository to establish either assumption
(`git log -S GatherBlockQuantized` finds this change only).  Its only test
asserts that same assumption.  This violates the no-wrong-shapes gate: an
unverified custom operator must remain unregistered (or have its exact
runtime/schema contract implemented and tested), rather than emit a
potentially incorrect rank/extent/type.

**Fixer: Coco** (the routed owner for GatherBlockQuantized/quantized data
kernels; not Apone).  Remove this registration/handler pending a verified
contract, or implement the authoritative kernel/schema and parity tests for
all supported axes, packed widths, and dtypes.

## Verified non-blocking observations

- CSA ratio-4 supplies all 5 required state outputs and optional selected
  indices with the CPU kernel's rank/extent formulas; ratio-128 supplies its
  3 outputs.  Its fresh symbols are allocated in fixed call order, not from
  hash iteration.
- MoE/QMoE and SparseKvGather output ranks match their CPU kernels.
- Change scope is limited to `onnx-runtime-shape-inference`.
- `cargo test --locked -p onnx-runtime-shape-inference` passed: 160 unit tests
  and 1 doctest.

<!-- Source: coco-remove-gbq-handler.md -->
### 2026-07-17: Remove guessed GatherBlockQuantized shape inference
**By:** Coco
**What:** Removed the `GatherBlockQuantized` shape-inference handler, registration, and self-referential unit test.
**Why:** Its packed-width expansion and output dtype lacked an authoritative in-repo contract. Shape inference is deferred until an authoritative `com.microsoft` schema or CPU-kernel contract exists; leaving the op unregistered safely leaves its output shapes unknown.

<!-- Source: gorman-shape-handlers-rereview.md -->
### 2026-07-17: Shape-handler re-review
**By:** Gorman
**What:** 🟢 APPROVE `d2976b3` relative to `82456ed`.
**Why:** `GatherBlockQuantized`, its registration, GBQ-only `axis_attr` helper/import, and its guessed-shape test are removed. The module documentation explicitly states that GBQ is deliberately unregistered pending an authoritative `com.microsoft` schema/CPU-kernel contract and that unregistered operations leave output shapes uninferred. The retained `MoE`, `QMoE`, `SparseKvGather`, and both-domain `CompressedSparseAttention` registrations and tests are unchanged; CSA coverage still resolves every present-state output. The diff is limited to `onnx-runtime-shape-inference`.
**Validation:** `STATE_BACKEND=local CARGO_TARGET_DIR=/home/justinchu/target-gorman-recheck cargo test --locked -p onnx-runtime-shape-inference` passed: 159 unit/integration tests and 1 doctest, 0 failures. A package no-run compilation reported no warnings.

## 2026-07-17 — Scribe inbox merge (22:55Z)

<!-- merged from bishop-codecov-review.md -->

### 2026-07-17: Approve Codecov coverage CI
**By:** Bishop
**What:** 🟢 APPROVE commit `0c0e674`. The `coverage` job is valid YAML, is a sibling of the existing jobs, leaves `rust-quality`, `rust-portable`, and `cuda-compile` structurally unchanged, and uploads the generated `codecov.json` with `if: always()` and `fail_ci_if_error: false`.
**Why:** Dependency-tree checks found no `onnx-genai-ort`, `ort-sys`, or `onnx-runtime-ep-cuda` dependency across all 15 selected crates with non-dev edges; an additional all-edge check of the three `onnx-genai-*` crates was also clean. `codecov.yml` makes project and patch statuses informational and uses sensible ignore globs. Human follow-up to set `CODECOV_TOKEN` and enable the repository on codecov.io is non-blocking. The branch is one commit behind the latest `origin/main`, but the reviewed commit itself changes only the workflow and `codecov.yml`; update the branch when landing.

<!-- merged from bryant-onnxrs-r8-rereview.md -->

### 2026-07-18: 🔴 REJECT — Squeeze validation still hides invalid axes

Leon’s revision correctly uses `checked_axis` and `ShapeInferError::Invalid`,
preserves unresolved output for a selected dynamic extent, and the requested
gate passes. However, `squeeze_common` validates an axis’s extent before
validating every static axis in the list. For an input such as `[symbolic, 1]`
with static axes `[0, 0]`, the first axis returns `Ok(())` because its extent
is dynamic, so the duplicate is never rejected. Likewise, `[0, 2]` returns
unresolved before reporting static out-of-range axis `2`. Both are malformed
static-axis graphs and violate the required error behavior.

Fix by first normalizing and validating the entire axes list (including
duplicates), then inspecting selected extents and returning unresolved only if
the validated list selects an unknown extent. Add regression tests for
`[symbolic, 1]` with `[0, 0]` and `[0, 2]`.

**Revision owner:** Deckard (Systems) — distinct from locked-out Sapper and
Leon.

**Gate:** `cargo test --locked -p onnx-rs -p onnx-runtime-shape-inference`
passed: all reported test suites passed (including 162 shape-inference tests);
0 failures.

<!-- merged from bryant-onnxrs-r8-rereview2.md -->

### 2026-07-17: ONNX-RS Squeeze round-8 third-cycle re-review approved
**By:** Bryant
**What:** 🟢 APPROVE the two-pass static-axis validation in `squeeze_common`.
**Why:** The first pass normalizes, range-checks, and duplicate-checks every static axis before the second pass reads any extent. The new symbolic-dimension regressions use `unwrap_err()` and assert the duplicate and out-of-range `Invalid` errors. Valid dynamic extents and runtime axes still leave inference unresolved, while valid static size-1 axes and no-axes static inputs retain concrete squeezing. `checked_axis` is fallible and all validated indexing is safe. The required locked test gate passed.

<!-- merged from bryant-onnxrs-r8-review.md -->

### 2026-07-18: 🔴 REJECT onnx-rs round 8 schema review
**By:** Bryant
**What:** Reject round 8. `Squeeze` shape inference fabricates a successful output shape for a static but invalid selected dimension.
**Why:** `squeeze_common` in `crates/onnx-runtime-shape-inference/src/handlers/movement.rs` removes every selected axis without checking that a known extent is `1`, and normalizes out-of-range axes by clamping (`norm_axis`). For example, Squeeze `[2, 3]` with a static axes initializer `[1]` infers `[2]`, although ONNX requires the selected dimension to be 1 and shape inference must reject it; an axis `[9]` is likewise silently treated as the final axis. This makes the checker/inference authority report fabricated shapes for malformed graphs. The new runtime-input guard is correct, but the new round-8 tests cover no invalid static axis or non-singleton selected-dimension case. Leon should revise; Sapper is locked out.

<!-- merged from coordinator-ci-must-report-test-coverage-to-codecov-add-regres.md -->

### 2026-07-17T21-20-01: CI must report test coverage to Codecov; add regression tests for reviewer-found bug classes
**By:** coordinator
**What:** CI must report test coverage to Codecov; add regression tests for reviewer-found bug classes
**References:** scripts/coverage.sh, .github/workflows/ci.yml, docs/GLM_READINESS_GAPS.md
**Why:** User directive (2026-07-17): raise test coverage and ensure defects like the round-8 Squeeze (static non-singleton axis fabrication) and CUDA TopK (non-final-axis layout) bugs are caught by tests. CI currently collects NO coverage — scripts/coverage.sh exists (cargo-llvm-cov, supports --codecov) but ci.yml has no coverage job or Codecov upload, no codecov.yml. Action: add a CI coverage job over the offline-safe crate set that uploads to Codecov, add codecov.yml, and treat regression tests for reviewer-found bug classes as expected going forward. Note the CI portable test job's crate list is narrow (ir/ep-api/ep-cpu/loader/session/tracer) — it omits onnx-rs and shape-inference where the Squeeze bug lived; coverage should surface this.

<!-- merged from dallas-ci-codecov.md -->

### 2026-07-17: Measure portable Rust coverage with Codecov
**By:** Dallas
**What:** The CI coverage lane explicitly covers `onnx-runtime-ir`, `onnx-runtime-ep-api`, `onnx-runtime-ep-cpu`, `onnx-runtime-loader`, `onnx-runtime-session`, `onnx-runtime-tracer`, `onnx-rs`, `onnx-runtime-shape-inference`, `onnx-runtime-optimizer`, `onnx-runtime-quantization`, `onnx-runtime-memory`, `onnx-runtime-cpuinfo`, `onnx-genai-genai-config`, `onnx-genai-metadata`, and `onnx-genai-runtime-config`. No crate from this validated pure/offline-safe candidate set was excluded. The rest of the workspace remains excluded because ORT-backed crates can trigger native ORT downloads and CUDA crates require unavailable GPU support.
**Why:** The complete explicit crate set passed offline tests and produced a Codecov JSON report with `cargo llvm-cov`; avoiding `--workspace` preserves the existing CI constraints. A human must enable the repository on codecov.io and may need to configure the `CODECOV_TOKEN` repository secret for authenticated uploads.

<!-- merged from deckard-onnxrs-squeeze-fix2.md -->

### 2026-07-17: Squeeze validates static axes before extents
**By:** Deckard
**What:** Split static-axis Squeeze validation into a complete structural pass followed by extent inspection.
**Why:** Every axis is now normalized and checked for range and duplication before an unknown extent can make the result unresolved. This preserves invalid-graph errors for duplicate or out-of-range static axes while retaining unresolved inference when a structurally valid selected extent is dynamic.

<!-- merged from gorman-topk-fix-review.md -->

### 2026-07-17: TopK and analogous axis-validation review
**By:** Gorman  
**Verdict:** 🟢 APPROVE

**Findings:**
- No blocking correctness findings.
- `crates/onnx-runtime-shape-inference/src/handlers/selection.rs:13-42,66-93` replaces clamping with `checked_axis` for ArgMax/ArgMin and TopK, returning `ShapeInferError::Invalid` before indexing. Defaults remain unchanged, and dynamic TopK `K` still produces a fresh symbolic extent.
- `crates/onnx-runtime-shape-inference/src/handlers/mod.rs:31-42` preserves the valid negative-axis boundary: `-rank` normalizes to `0`, while `rank` and values below `-rank` are rejected.
- `crates/onnx-runtime-shape-inference/src/handlers/movement.rs:59-95` validates every explicit Transpose entry, rejects duplicates after normalization, and preserves the missing-`perm` reverse default.
- `crates/onnx-runtime-shape-inference/src/handlers/movement.rs:291-328` validates Unsqueeze axes against output rank, rejects normalized duplicates, and leaves dynamic axes unresolved.
- `crates/onnx-runtime-shape-inference/src/handlers/movement.rs:1162-1195` validates Gather axis against data rank and retains the correct `data[..axis] + indices.shape + data[axis+1..]` construction.
- Unknown input rank/type still returns unresolved before axis validation in the affected handlers; symbolic extents remain supported. No random iteration, concurrency, or other nondeterministic behavior was introduced.
- `crates/onnx-runtime-shape-inference/tests/op_rules.rs:714-827,1008-1040,1259-1275,1387-1407,1635-1663` covers dynamic K/axes, valid negative/default/middle axes, correct output values, and the new out-of-range/duplicate rejection branches.

**Gate observed:**
- `STATE_BACKEND=local CARGO_TARGET_DIR=/home/justinchu/target-gorman-shapetests cargo test --locked -p onnx-runtime-shape-inference`: green. Reported suites: 14, 13, 168, and 1 tests passed; zero failures, panics, or compiler errors.

Plain-text summary: 🟢 APPROVE — the clamp-to-valid-shape defects are correctly replaced by explicit validation without regressing valid axes, unresolved dynamic cases, Gather construction, Transpose defaults, or determinism; the required crate gate is green.

<!-- merged from holden-cuda-cmp-review.md -->

### 2026-07-17: CUDA comparison dtype-generic review
**By:** Holden
**Verdict:** 🟢 APPROVE

**Findings:**
- No blocking correctness or contract findings.
- `crates/onnx-runtime-ep-cuda/src/kernels/pointwise.rs:446-461` generates direct signed `int`/`long long` comparisons for i32/i64, with no integer-to-float coercion; every op uses the correct operator and writes canonical one-byte Bool values.
- `crates/onnx-runtime-ep-cuda/src/kernels/pointwise.rs:477-503,650-690` dispatches by operand dtype, permits Bool only for `Equal`, rejects unsupported or mixed operands before launch, validates Bool output, dense inputs, and the broadcast output shape.
- `crates/onnx-runtime-ep-cuda/src/provider.rs:180-188` applies the positional `input_dtypes` contract to exactly the registered comparison family. `crates/onnx-runtime-ep-cuda/src/kernels/pointwise.rs:518-533` claims only same-type f32/i32/i64 operands plus Bool `Equal`, with actionable denials otherwise.
- `crates/onnx-runtime-ep-cuda/src/kernels/pointwise.rs:684-730` reuses right-aligned zero-stride broadcast metadata and introduces no atomics, reductions, shared writes, or races. Per-element output ownership is deterministic.
- `crates/onnx-runtime-ep-cuda/src/kernels/pointwise.rs:739-745` correctly declines strided inputs and CUDA graph capture because execution allocates/copies/frees per-call metadata and synchronizes.
- `crates/onnx-runtime-ep-cuda/tests/pointwise_gpu.rs:462-575` checks all five i32/i64 operators against a CPU broadcast oracle, GLM-like i64 position masks, repeated deterministic results, i32 masks, Bool equality, supported claims, and an actionable unsupported-dtype denial.

**Gate observed:**
- `cargo test --locked -p onnx-runtime-ep-cuda` with the requested CUDA environment and `CARGO_TARGET_DIR=/home/justinchu/target-holden-cudacmp`: green; all reported test binaries passed with zero failures.
- Targeted GPU confirmation: `integer_comparisons_broadcast_match_cpu_oracle`, `integer_comparisons_cover_glm_like_masks_and_are_deterministic`, and `comparison_claims_integer_dtypes_and_rejects_unsupported_dtype` all executed and passed.

Plain-text summary: 🟢 APPROVE — CUDA comparison dtype dispatch, claim gating, broadcasting, determinism, and value assertions are correct; the required CUDA test gate is green.

<!-- merged from hudson-topk-fix-coverage.md -->

# TopK axis validation and shape-inference coverage uplift

- **Date:** 2026-07-17
- **Owner:** Hudson

## Correctness fixes

`TopK` previously normalized axes through `norm_axis`, which clamps invalid values. A rank-3 input and `axis=3`, for example, incorrectly inferred a valid last-axis result. It now uses `checked_axis` and returns `ShapeInferError::Invalid` with the invalid axis and rank. Both values and indices continue to share the validated output shape.

The focused neighboring-axis review found and fixed the same contract violation in `ArgMax`/`ArgMin`, `Transpose`, `Unsqueeze`, and `Gather`. These now reject out-of-range axes; `Transpose` and `Unsqueeze` additionally reject duplicate axes. Valid negative axes and valid normal behavior are unchanged.

## Coverage tests added

- TopK middle-axis and out-of-range regression; axis-validation coverage for ArgReduce, Transpose, Unsqueeze, and Gather.
- Custom-op arity/rank/compression-ratio contract failures, alongside existing all-output CSA tests.
- Shape-data source propagation and invalid Cast metadata paths.
- Pooling missing/malformed kernel, rank validation, and dynamic spatial extent behavior.

## Line coverage (before → after)

| Handler | Before | After |
|---|---:|---:|
| `custom_ops.rs` | 71.21% | 79.17% |
| `data_ops.rs` | 62.83% | 71.39% |
| `movement.rs` | 65.60% | 66.92% |
| `pooling.rs` | 71.57% | 77.82% |

The remaining unexercised branches are primarily broader operator paths already outside this focused regression/edge-case set; no unreachable branch was claimed as covered.

<!-- merged from leon-onnxrs-squeeze-fix.md -->

### 2026-07-18: Squeeze static-axis validation

Leon fixed the round-8 `Squeeze` shape-inference defect in
`onnx-runtime-shape-inference`. Statically known axes now use `checked_axis`,
reject duplicate axes, reject statically known non-singleton extents, and leave
the output unresolved when a selected extent is dynamic. No-axes inference also
leaves dynamic input extents unresolved.

The implementation matches sibling handlers' invalid-axis convention:
`ShapeInferError::Invalid { op, detail }`, returning `Err` rather than clamping
or fabricating an output shape.

<!-- merged from ripley-bqmoe-design.md -->

# Decision: `pkg.nxrt::BlockQuantizedMoE` v1 ABI proposal (design note)

- **Author:** Ripley (architect)
- **Date:** 2026-07-18
- **Branch:** `bqmoe-design` (worktree `/home/justinchu/wt-bqmoe`), commit `2a7e45a`
- **Deliverable:** `docs/BLOCKQUANTIZEDMOE_DESIGN.md` (docs-only, no kernels)
- **Status:** AWAITING JUSTIN SIGN-OFF on the op contract before any kernel work

## Summary

Design note for P1 #6 — the largest practical GLM blocker. `BlockQuantizedMoE`
= QMoE's expert structure, but with native IQ/MXFP4 codebook block formats
(reusing the existing `BlockQuantizedMatMul` decoders) and **selected-expert
dispatch** (only routed experts evaluated), wired to the lazy weight-lease seam
at `weight.rs:94-104`.

## Proposed ABI (v1)

- Domain `pkg.nxrt`, op `BlockQuantizedMoE`, version 1. Single output = input
  shape/dtype.
- **Key deviation from QMoE:** IQ/MXFP4 blocks are self-describing, so v1 has
  **no `*_scales`, no `*_zero_points`, no codebook inputs** (codebooks are
  compile-time grids). Each expert weight is one packed `uint8`
  `[E, N, blocks, block_bytes]` tensor.
- Inputs (dense 0–8): `input`, `router_probs` (logits), `fc1_experts_weights`,
  `fc1_experts_bias?`, `fc2_experts_weights`, `fc2_experts_bias?`,
  `fc3_experts_weights?`, `fc3_experts_bias?`, `router_weights?`.
- Attributes reuse `MoeAttributes` (`k`, `activation_type`,
  `normalize_routing_weights`, `swiglu_fusion`, `activation_alpha/beta`,
  `swiglu_limit`, `use_sparse_mixer`) + `format`/`block_layout_version` from
  `BlockQuantizedMatMul`. Dropped: `expert_weight_bits`, `block_size`,
  `quant_type`. `inter`/`E` inferred from weight shapes.
- Dispatch: route-first grouping (always-on), decode each selected expert once,
  resident fallback for non-paging EPs. Selected-expert device leasing requires
  a lazy-seam extension: the current binder receives only a whole
  `LazyWeight`, not an expert/slice selector
  (`crates/onnx-runtime-ep-api/src/weight.rs:121-161,205-227`).
- Determinism: f32 accumulate; fixed ascending-expert reduction is a new v1
  rule, not a QMoE precedent (QMoE reduces in top-k route-slot order); tie-break
  = lower expert index (existing `total_cmp().then(a.cmp(b))` rule;
  `crates/onnx-runtime-ep-cpu/src/kernels/qmoe.rs:312-335,356,404-420` and
  `crates/onnx-runtime-ep-cpu/src/kernels/moe.rs:400-402`).
- Staging: Phase 1 CPU parity oracle → Phase 2 CUDA bitwise parity → Phase 3b
  live device paging (out of v1 scope).

## Open decisions needing Justin's sign-off (defaults recommended)

1. Weight-input ordering — dense 0–8 (default) vs. QMoE-index-preserving.
2. Router input — logits + optional aggregation weights (default) vs. pre-norm.
3. Block-format enum — string `format` (default) vs. int enum.
4. sparse_mixer placement — normalize OUTSIDE the op, error if set (default).
5. Uniform single `format` for all experts/projections (default) vs. per-proj.
6. `inter`/`expert_dim` inferred from shapes (default) vs. declared attributes.
7. v1 offload — resident-first, with an expert-indexed lazy-binder/
   `LazyWeight` slice-descriptor extension required before Phase 3b paging
   (default) vs. require paging now
   (`crates/onnx-runtime-ep-api/src/weight.rs:121-161,205-227`).
8. Freeze as version 1 now (default: yes).

Full rationale and file:line citations in `docs/BLOCKQUANTIZEDMOE_DESIGN.md`.

<!-- merged from sapper-onnxrs-r8.md -->

# onnx-rs schema round 8

Added standard-domain schemas for Attention, CastLike, CumSum, GreaterOrEqual,
LessOrEqual, Min, Max, RotaryEmbedding, Softplus, Squeeze, TopK, and
Unsqueeze. Unsqueeze includes the target fixture signatures for opsets 11, 13,
and 24; the other target schemas resolve at opset 24.

Skipped Constant because it was already registered. None of the requested
round-8 operators was registered before this change.

Shape inference already covered the requested elementwise, attention, rotary,
CastLike, CumSum, and TopK rules. This round makes runtime axes for Squeeze
(like Unsqueeze) deliberately unresolved because they can change output rank;
runtime TopK K remains an unresolved axis extent.

<!-- merged from vasquez-cuda-int-compare.md -->

### 2026-07-17: CUDA comparison operators support GLM integer indices
**By:** Vasquez
**What:** CUDA EP `Equal`, `Greater`, `Less`, `GreaterOrEqual`, and `LessOrEqual` now support same-dtype f32, int32, and int64 operands with a bool output and existing NumPy broadcasting. `Equal` additionally supports bool operands. `Where`, `Not`, and logical ops were not changed.
**Why:** GLM position/token IDs and mask construction compare int64/int32 tensors; selecting a dtype-specific NVRTC entry keeps these comparisons on-device and makes claim-time coverage accurately reject unsupported dtypes.

<!-- merged from wallace-shapeinf-coverage.md -->

### 2026-07-17: Shape-inference coverage finding — TopK invalid axis is clamped
**By:** Wallace
**What:** Added a regression test for TopK on a middle axis and an out-of-range axis. The middle-axis shape is correct (`[2, 8, 4]`, axis `1`, K `3` → `[2, 3, 4]` for both outputs), but axis `3` for rank `3` is accepted instead of rejected.
**Why:** ONNX requires TopK's axis to be in `[-rank, rank)`. `selection::top_k` calls `norm_axis`, which clamps `3` to the final axis, fabricating a valid-looking output. The new test fails at `op_rules.rs:758`; per the test-only scope, handler logic was not changed and coverage expansion stopped for follow-up.

Before coverage (line): custom_ops.rs 71.21% (188/264), data_ops.rs 62.83% (235/374), movement.rs 66.27% (937/1414), pooling.rs 71.57% (355/496), selection.rs 84.26% (166/197), elementwise.rs 66.48% (117/176).
After coverage: not measured; the newly added correctness regression intentionally fails against the current handler.
Tests added: 1 (one regression test with middle-axis and invalid-axis assertions).

