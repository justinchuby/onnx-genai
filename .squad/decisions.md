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

## 2026-07-18 Scribe merge: Reshape/Split, CUDA RoPE, and GLM audit

- **Reshape/Split:** Ferro's regressions exposed multiple `-1` inference acceptance; Leon added static target, count, and Split validation. Bryant rejected the first revision for zero-product Reshape and non-positive `num_outputs`; Deckard fixed both, added exact `Invalid` assertions, and Bryant approved. Shape-inference tests and coverage gates passed.
- **GPU-native RoPE:** Drake removed f32 host staging with a cached CUDA kernel covering 3D/4D layouts, rotation modes, cache addressing, broadcast, and tails. Holden found invalid `position_ids` were silently accepted and rejected; Deckard added device validation, error propagation, boundary tests, and `B=2,H=2` parity coverage. Final review approved commit `74a891b`; graph capture remains disabled due to host flag synchronization.
- **GLM CUDA audit:** Newt refreshed the standard-op readiness audit, distinguishing registration/loading from smooth execution and documenting the all-f32 constraints of denied operators. Parker corrected overclaims: casts can satisfy constrained inputs in mixed graphs; host-staged Attention/RoPE and custom BlockQuantizedMoE, IndexShare, and MTP boundaries remain throughput/execution blockers.

## 2026-07-18 — Scribe inbox merge (PR triage, Attention, CI, and test coverage)

<!-- merged from ash-cuda-attention-native.md -->

# Decision: GPU-native standard `ai.onnx::Attention` kernel

**Author:** Ash (CUDA kernel engineer)
**Branch:** `cuda-attention-native` (worktree `/home/justinchu/wt-attn`, off origin/main 74a891b)
**File:** `crates/onnx-runtime-ep-cuda/src/kernels/standard_attention.rs`

## Summary

Converted the host-staged standard `Attention` kernel to a GPU-native
implementation. The previous impl did `dtoh` on Q/K/V/mask into `Vec<f32>`, ran
the entire SDPA on the CPU, then `htod` the result — a D2H→CPU→H2D round trip
that was the GLM-decode perf blocker. Bulk tensors now stay resident on the
device end to end.

## What moved to device

Two NVRTC kernels (module `standard_attention_f32_v1`), following the crate's
RoPE/where_op NVRTC pattern (runtime module cache, `LaunchConfig` +
`PushKernelArg`, `cuptr` device pointers, `self.runtime.stream()`):

- **`build_kv`** — gathers each K/V input into a contiguous
  `[batch, kv_heads, total_seq, dim]` present buffer, doing the 3D→4D head
  reshape **and** the `past ⧺ current` cache concat on device (stride-aware
  indexing, handles 3D or 4D current/past). Writes directly into the
  `present_key`/`present_value` output slots when those outputs are requested,
  else into scratch (still the attention kernel's K/V source).
- **`attention_row`** — one CUDA block per `(batch, q_head, query)` row. Computes
  scaled QK scores (√scale folded into each operand), softcap, the composed
  causal + padding + attn-mask frontiers, numerically-stable softmax, and the
  probs·V accumulation. Writes `Y` (in 4D or 3D output layout, so no host
  reshape) and the optional `qk_matmul_output` directly to device buffers. Q is
  read in place with stride-aware indexing (no Q materialization); the mask input
  is read in place on device (bool bytes / f32), broadcast in-kernel exactly like
  the CPU reference (right-aligned dims, short-last-dim → −inf).

No Q/K/V/mask/score tensor ever leaves the device in bulk.

## What stayed on host (and why)

- **Attribute/shape/validation resolution** and **all error messages** (arity,
  dtype deny, non-contiguous, past-together, past-vs-nonpad exclusivity, dim
  mismatches, qk mode range) — pure control logic, no bulk data.
- **Per-batch causal `offset` and padding-frontier arrays** — built on host from
  scalars and uploaded as tiny `int64[batch]` device arrays (as the task
  suggested).
- **`nonpad_kv_seqlen`** (7th input, opset 24+) — a per-batch scalar count read
  back via a small `dtoh` to compute the offsets/pad limits. It is tiny control
  state (length = batch), not bulk data.

No `qk_matmul_output` host recompute was needed: all four modes (0 raw, 1 after
softcap, 2 after mask, 3 after softmax) are produced in-kernel at the correct
stage.

## Determinism

Fixed per-row reduction order, byte-identical run to run:
- QK dot products and probs·V sums each run in ascending index order **within a
  single thread** → bit-identical to the CPU reference for those stages.
- Softmax max/exp/sum are performed sequentially by the block's lead thread
  (thread 0), then a parallel normalize. No atomics feed a shared accumulator.
- The only float-rounding divergence from the CPU reference is transcendental
  ulp (`expf`/`tanhf` vs Rust `f32::exp`/`tanh`); covered by the 1e-4 test
  tolerance.

## cuda_graph_compatible

`false` (explicit override, matching the prior default). Setup performs
synchronous small `htod` uploads of the per-batch control arrays and a `dtoh` of
`nonpad_kv_seqlen`, which are not capture-safe. A capturable design would need
device-side offset construction; deferred as a follow-up. The bulk-data win is
independent of graph capture.

## Semantic parity preserved

√scale-folded scores; softcap `s·tanh(s/softcap)` only when nonzero; composed
(intersected) causal/pad/attn masks; GQA/MQA head sharing `kvh = qh/group`;
3D↔4D reshape; in-op KV cache concat + `present_key`/`present_value` outputs;
`qk_matmul_output` modes 0–3 (softcap_mode=1, mask_mode=2); numerically-stable
softmax with fully-masked-row → zero-output guard; f32-only claim-time deny.

## Test coverage added (`tests/standard_attention_gpu.rs`)

Kept all existing tests green; added a general inline CPU reference (`sdpa_ref`)
and GPU-vs-reference parity tests: basic MHA; GQA with `kv_heads<q_heads` and
`batch>1, heads>1`; 3D-input reshape path; in-op past cache (`past_seq>0`, plus
`present_key`/`present_value` correctness); float mask add; bool mask; softcap≠0;
fully-masked row → zero output; and `qk_matmul_output` modes 0–3. Extended the
test harness with `run_opt` to model omitted optional inputs (absent
`TensorView` for the empty-string mask slot when past KV is supplied).

## Reviewer scrutiny

- **Softmax denominator rounding**: block lead-thread sequential sum matches CPU
  order; other stages are bit-identical. Confirm the 1e-4 tolerance is
  acceptable for the intended models (it is for GLM f32).
- **Scores scratch memory**: a `[batch, q_heads, q_seq, total_seq]` f32 scratch
  is allocated (same magnitude as `qk_matmul_output`). Fine for decode; large
  prefills allocate the full attention matrix. Correctness-first; a
  shared-memory/online-softmax variant is a perf follow-up.
- **value/key head-count assumption**: present buffers are built with
  `kv_heads = key.heads`; `value.heads` is assumed equal (standard GQA), matching
  the prior impl's implicit assumption.


<!-- merged from bryant-child-lru-review.md -->

# Bryant review: ChildExecutor multi-signature LRU

**Verdict: 🟢 APPROVE — land as-is.**

Reviewed commit `caf2dba` on `child-lru`.

- Scope is clean: only `crates/onnx-runtime-session/src/executor.rs` changed; no `.squad/` or unrelated crate changes.
- The capacity-4 `Vec` LRU is bounded and deterministic. Hits remove and append the matching plan without incrementing `builds`; misses compile, evict index 0 only when full, append the new MRU plan, and increment `builds`.
- The selected last index is borrowed mutably only after cache mutation completes. `runs` increments per successful locate/compile before dispatch; public API and stats propagation remain unchanged.
- A→B→A correctly asserts 2 builds/3 runs; the prior single-slot implementation would build three times.
- Eviction coverage exceeds capacity, confirms the oldest signature rebuilds, and confirms a retained recent signature remains a hit.
- Capture reuse changes tensor values, checks the expected result, and compares against a fresh-compile reference, ruling out stale captured state.

Gate: `cargo test -p onnx-runtime-session` passed. Targeted ChildExecutor tests also passed (4/4).


<!-- merged from deckard-attention-fix.md -->

### 2026-07-18: Attention K/V cache concatenation uses per-operand geometry
**By:** Deckard
**What:** The GPU-native standard Attention kernel now launches `build_kv` for key with `past_key.seq + K.seq` and for value with `past_value.seq + V.seq`, while retaining the CPU reference's validation that the two resulting present sequence lengths agree. Added GPU regressions for differing key/value cache splits, opset-24 `nonpad_kv_seqlen`, and explicit non-default scale.
**Why:** The value launch incorrectly reused the key cache's past and total sequence lengths, shifting current value tokens and corrupting both `present_value` and Y when key/value concat geometry differed. The new present-value test fails on the original code and directly checks both present output buffers; the other tests cover padding/causal offset semantics, opset-23 rejection, and sqrt-scale folding.


<!-- merged from dietrich-pr23.md -->

### 2026-07-18: Rebase PR #23 benchmarks and use the complete scatter fixture
**By:** Dietrich
**What:** Rebased `bench/serving-scenarios` onto current `origin/main`, retained main's continuous-batch admission/eviction and prefix-cache cold/warm scenarios, and changed the end-to-end tokens/second benchmark plus README to use `tiny-llm-scatter`.
**Why:** Main independently contained evolved versions of both new serving scenarios, so retaining them preserved the intended benchmark coverage without duplicate functions. `tiny-llm-scatter` is semantically suitable, has the same tokenizer, and includes both `model.onnx` and `model.onnx.data`, unlike the incomplete `tiny-llm` fixture.


<!-- merged from ferro-ci-python-deps.md -->

### 2026-07-18: Install projection-fusion Python dependencies in Rust CI lanes
**By:** Ferro
**What:** The `rust-portable` and `coverage` jobs set up Python 3.12 and install current `numpy` and `onnxscript` before running crates that include `onnx-runtime-session`.
**Why:** The projection-fusion integration test generates ONNX fixtures through Python, and both CI lanes otherwise fail with `ModuleNotFoundError`, leaving main red.


<!-- merged from gorman-session-cov-review.md -->

### 2026-07-18: Session edge-case regression review
**By:** Gorman
**What:** 🟢 APPROVE commit `71b859d` for landing as-is. The commit changes only `crates/onnx-runtime-session/tests/executor.rs` (88 test lines), with no production or `.squad/` changes. The reverse-insertion test repeatedly computes topology and runs the session twice with identical outputs; the cycle test builds a real two-node cycle and matches `SessionError::Graph(GraphError::CycleDetected)`; the initializer test creates a producer-backed initializer and checks the actionable tensor, node, and initializer error details. Construction uses the existing IR fixture style (`Graph`, `Node`, `ValueId`, and helpers) used throughout the file.
**Why:** Although line coverage remains 77.55%, these tests add meaningful behavioral regression guards for dependency ordering/determinism and two malformed-graph rejection contracts. Reworking solely to increase line coverage would trade away useful semantic protection; land them and pursue uncovered branches separately. Gate passed: `STATE_BACKEND=local cargo test -p onnx-runtime-session` completed with zero failures.


<!-- merged from gorman-test-quality-review.md -->

### 2026-07-18: Test-quality regression review
**By:** Gorman
**What:** 🟢 APPROVE commit `cffbcb6`.
**Why:** Scope is test-only and limited to the three requested pure crates. BatchNorm uses `training_mode=1` with three outputs and verifies Y preserves X while training statistics remain unresolved. BF16 Mod independently derives f32 remainder results from BF16-rounded inputs, rounds back to BF16, and covers a negative dividend. Loader builds a genuine `a ↔ b` dependency cycle that protobuf validation accepts and the public load path rejects as `LoaderError::GraphBuild` containing `CycleDetected`, from `Graph::validate()`.

**Gates:** `cargo test -p onnx-runtime-shape-inference`, `cargo test -p onnx-runtime-ep-cpu`, and `cargo test -p onnx-runtime-loader` all passed. Each named regression also passed directly.


<!-- merged from hicks-child-lru.md -->

### 2026-07-18: Bounded deterministic LRU for child executors
**By:** Hicks
**What:** Replaced `ChildExecutor`'s single compiled-plan slot with a four-entry, signature-keyed LRU. Cache hits move the plan to the most-recently-used end; misses compile once, increment `builds`, and evict the oldest entry at capacity.
**Why:** Control-flow bodies can alternate among stable dtype/shape signatures (for example A→B→A). Four entries cover a small working set without unbounded executor retention. A `Vec` provides explicit deterministic ordering with no hash iteration dependence. Regression tests cover A→B→A reuse, oldest-entry eviction with a recent-entry hit, and capture rebinding parity against a freshly compiled plan.


<!-- merged from hicks-pr27.md -->

### 2026-07-18: Harden CUDA decode shared-KV bucket growth
**By:** Hicks
**What:** Rebased PR #27 (`fix/cuda-decode-kv-capacity`) cleanly onto `origin/main` at `53ef68c`. Shared-KV growth now rejects required capacity above the model-declared `max_length`, and the bucket helper never returns an over-limit allocation. KV replacement buffers and the fallible captured attention-mask replacement are fully prepared before the old captured graph is released and the session state is committed.
**Why:** This preserves power-of-two grow-on-boundary behavior while preventing allocations beyond the model ceiling and preventing mask-allocation failures from leaving KV/capture state partially updated. Validation passed after the final rebase: `cargo check -p onnx-genai-ort`, 22/22 crate library tests, and the four KV bucket tests. Commit `deea1ab` was force-with-lease pushed only to `fix/cuda-decode-kv-capacity`.


<!-- merged from mariette-attention-rereview.md -->

### 2026-07-18: Approve CUDA Attention concat-geometry revision
**By:** Mariette
**What:** 🟢 APPROVE commit `f57e35` for landing as-is.
**Why:** `present_key` uses key past/current/total geometry, while `present_value` independently uses value past/current/total geometry and `v_head_size`. Equal final sequence lengths remain required, matching the CPU `concat_cache` behavior while allowing different key/value past-current splits. Output buffers remain correctly sized, and the masking, GQA/layout, and deterministic softmax logic is unchanged. The regression directly checks `present_value` contents and failed against the pre-fix kernel (`[100, 200, 0, 300, 400]` vs `[100, 200, 300, 400, 500]`). Opset-24 nonpad masking/causal-offset/v23 rejection and non-default scale parity are meaningfully covered. Required gate: **211 passed, 0 failed, 0 ignored**.


<!-- merged from mariette-attention-review.md -->

### 2026-07-18: Reject GPU-native standard Attention
**By:** Mariette
**What:** 🔴 REJECT commit `ffd231d`. Ash is locked out; **Deckard** should revise.
**Why:** `standard_attention.rs:621` derives `past_seq` only from `past_key`, then `:895` passes that key length into the `present_value` `build_kv` launch even though `:682` separately permits a different past/current V split with the same total length. The prior CPU code concatenated K and V using their own past lengths. A focused GPU probe with K split 2+1 and V split 1+2 produced present V `[100, 0, 200]` instead of `[100, 200, 300]`.

The new tests also leave real semantic paths uncovered: there is no opset-24 `nonpad_kv_seqlen` test (per-batch negative causal offsets, unconditional pad frontier, v23 rejection, or past-cache mutual exclusion), and no explicit `scale` test. These omissions fail the requested semantic gate.

Mask composition, GQA mapping, ordinary 3D/4D layout, equal-split past/present, qk modes, fully-masked rows, and deterministic fixed-order reductions otherwise match the prior reference in the reviewed paths.

**Gate:** `STATE_BACKEND=local CARGO_TARGET_DIR=/home/justinchu/target-mariette-attn cargo test -p onnx-runtime-ep-cuda` passed, including all 16 `standard_attention_gpu` tests. The focused cache-split parity probe failed as described above; its temporary test file was removed.


<!-- merged from newt-pr25.md -->

### 2026-07-18: ORT plugin registration state is process-global
**By:** Newt
**What:** Track plugin registration paths and discovered provider names in a process-global registry, and serialize registration with EP-device diff discovery across `Environment` instances.
**Why:** ORT plugin-library registration is process-global. Per-environment state could re-register the same handle or lose the provider name when another environment performed the first registration.


<!-- merged from ripley-session-coverage.md -->

# Executor edge-case coverage

- **Commit:** `71b859d0174fad6b1e7c10cf1d2cc8038fdea0ad`
- **Coverage:** executor source lines were **77.55% (3310/4268)** before and **77.55% (3310/4268)** after; the new public API regressions exercise already-covered execution seams, so the source-line percentage did not move. Total source coverage remains **79.06%**.
- **Tests added:**
  - reverse-inserted dependency DAG executes in deterministic topological order across repeated planning/runs;
  - cyclic graph fails with `GraphError::CycleDetected` rather than constructing a partial plan;
  - initializer reused as a node output is rejected, protecting immutable weight storage.
- **Validation:** `cargo test -p onnx-runtime-session` passed; targeted `--test executor` passed (23 tests).
- **Bug found:** none.


<!-- merged from spunkmeyer-pr20.md -->

### 2026-07-18: PyO3 0.29 migration for onnx-runtime-python
**By:** Spunkmeyer
**What:** Migrated GIL acquisition/release to `Python::attach`/`Python::detach`, replaced deprecated downcasts with `Bound::cast`/`cast_into`, updated interpreter initialization and owned-pointer construction, and enabled `pyo3/extension-module` only in maturin wheel builds.
**Why:** PyO3 0.29 removed the old APIs. Keeping `extension-module` out of ordinary Cargo builds preserves wheel behavior while allowing the Rust unit-test harness to link libpython successfully.


<!-- merged from vasquez-pr25-review.md -->

# Vasquez review — PR #25 global ORT plugin registration

## Verdict

🔴 **REJECT**

### Blocking: the static cache outlives ORT's registration state

`registered_ep_libraries()` is process-static (`env.rs:16-20`), but ORT 1.27's
global `OrtEnv` is reference-counted and destroyed when the last `Environment`
calls `ReleaseEnv`. Destruction also destroys the ORT environment-owned plugin
factories/devices. The Rust map is never cleared.

Consequently:

1. Environment A registers a plugin and caches its provider.
2. A is dropped as the last live environment; ORT unloads that registration.
3. Environment B is created later with a fresh ORT environment.
4. `register_execution_provider_library` returns `Ok(false)` from stale Rust
   state, so the plugin is not registered in B's ORT environment.
5. B reads the stale provider name, but `GetEpDevices` has no corresponding
   devices, and session setup fails.

This is a normal sequential-engine lifecycle, not merely a shutdown edge case.
It also conflicts with the repository's established invariant that the ORT
environment owns the plugin factory (`crates/onnx-genai-engine/src/engine.rs:
333-338`).

**Required fix:** Deckard must tie the registration/cache lifetime to the live
ORT environment generation. For example, serialize Rust `Environment`
creation/drop with an active-environment count and clear registration state
when the final wrapper releases ORT, or retain a canonical process-lifetime ORT
environment reference. Add a drop-last-environment → create-new-environment →
register-same-plugin regression test.

### Other review results

- The discovery mutex prevents the checked append path's registration TOCTOU;
  provider names are visible across concurrently live `Environment` wrappers.
- No concrete lock-order inversion was found. Both mutex poison cases return an
  error rather than deadlocking, although plugin DLL loading occurs while the
  Rust registration mutex is held.
- Windows `ORTCHAR_T` UTF-16 handling is preserved.
- The added test manually inserts into the Rust map. It does not perform real
  ORT registration/device-diff discovery, exercise concurrent callers, or test
  environment destruction/recreation. The test itself passes.

Revision owner: **Deckard** (Newt is locked out for this revision).


<!-- merged from vasquez-test-quality.md -->

### 2026-07-18: Test-quality regressions for BatchNorm, Mod, and loader cycles
**By:** Vasquez
**What:** Added tests that (1) run opset-15 BatchNormalization with `training_mode=1` and three declared outputs, asserting Y retains X's shape/dtype while unresolved training statistics are not fabricated; (2) run bf16 `Mod` with `fmod=1`, comparing f32-computed and bf16-rounded remainders and checking a negative dividend preserves its sign; and (3) load a protobuf graph with a two-node data-dependency cycle through the public bytes-loading API, asserting its `GraphBuild` error reports `CycleDetected`.
**Why:** These paths previously lacked direct regression coverage for multi-output graceful degradation, bf16 promotion semantics, and structural validation that occurs only after IR graph construction.

No real bug was surfaced: all three regression tests pass against the current implementation.


<!-- merged from wierzbowski-pr20-review.md -->

### 2026-07-18: PyO3 0.23 → 0.29 API migration review
**By:** Wierzbowski
**What:** 🟢 APPROVE PR #20 (`5ee76ea`, author Spunkmeyer).
**Why:** The migration preserves the required GIL, checked-cast, and FFI ownership semantics.

## Safety review

- All 3 `Python::with_gil` replacements are correct. PyO3 0.29 explicitly renamed this API to `Python::attach`; it attaches the thread and acquires/re-borrows the GIL. The two DLPack guard drops and the streaming callback therefore still execute Python-sensitive work while attached.
- All 4 `py.allow_threads` replacements are correct. `Python::detach` explicitly releases the GIL for the closure and reacquires it afterward. The streaming path correctly reattaches only around the Python callback.
- All 13 `downcast` / `downcast_into` replacements remain checked and fallible. `Bound::cast` and `cast_into` call `PyTypeCheck::type_check` and return `Result`; only the separately named `cast_unchecked` APIs are unchecked.
- `Python::initialize()` is the direct replacement for `prepare_freethreaded_python()`. Both implementations call `Py_InitializeEx(0)` when needed and then `PyEval_SaveThread()`.
- Both pointer conversions are ownership-correct. Each pointer comes directly from successful `PyCapsule_New`, which returns a new/owned reference. `Bound::from_owned_ptr` consumes that owned reference without incrementing it, and `.unbind()` transfers the same ownership into `Py<PyAny>` without an extra incref/decref. The null paths release only the separately allocated DLPack managed tensor and fetch the Python exception. No borrowed pointer is passed to `from_owned_ptr`, and refcount balance is unchanged from the former `Py::<PyAny>::from_owned_ptr`.

## Verification

- `cargo test --locked -p onnx-runtime-python --lib`: 19 passed.
- Built the abi3 wheel with maturin successfully, confirming the `extension-module` feature placement.
- Installed the wheel into an isolated review venv and ran `test_dlpack.py`: 41 passed, including off-thread DLPack deleter/GIL coverage.
- Focused API/eager/genai tests: 26 passed, 2 deselected.

## Non-blocking observations

- PyO3 0.29 `Python::attach` deliberately panics during interpreter shutdown; this is safer than attempting an invalid late attachment, but background-owned tensors must still not outlive the interpreter.
- A broader API test run exposed one pre-existing stale expectation: Float16 `Cast` now succeeds although the test expects an unsupported-kernel error. It is unrelated to this PyO3-only migration.



<!-- merged from deckard-pr25-fix.md (late arrival) -->

### 2026-07-18: Clear plugin registration state with the last ORT environment
**By:** Deckard
**What:** Track live `onnx_genai_ort::Environment` wrappers under a process-global lifecycle mutex. The last `ReleaseEnv` clears the plugin registration/provider-name cache before another `CreateEnv` can proceed.
**Why:** ORT 1.27 destroys environment-owned plugin factories and devices when its final environment reference is released. Generation-scoped cache state prevents a later fresh ORT environment from incorrectly reusing stale registration metadata, while preserving sharing between concurrently live environments.


<!-- merged from lambert-kimi-k3-readiness.md (late arrival) -->

### 2026-07-18: Treat KDA, MLA, and CSA as separate runtime state kinds
**By:** Lambert
**What:** Reuse CSA's versioned state/operator lifecycle, standard Attention/RoPE as fallback oracles, and QMoE/block-dequant internals, but implement KDA and Gated MLA as distinct semantic operators. Keep K3 MTP conditional until the released package verifies it.
**Why:** Current CPU CSA is DeepSeek-specific ratio-4/128 temporal compression, while public KDA uses gated recurrent matrix state and MLA uses learned low-rank latent KV. Conflating them would freeze an incorrect cache ABI before K3 weights and the technical report arrive on or before 2026-07-27.


<!-- merged from bishop-cuda-sparse-kv-gather.md -->

### 2026-07-18: Device-native `pkg.nxrt::SparseKvGather` on the CUDA EP
**By:** Bishop
**What:** Added and registered a device-native CUDA `SparseKvGather` kernel for `pkg.nxrt::SparseKvGather` v1, with raw-byte support for f32/f16/bf16 cache values, host-side deterministic index validation, and 9 parity/edge tests. The kernel preserves order and duplicates and keeps cache/index data device-resident during the copy.
**Why:** DeepSeek/GLM compressed sparse-attention was CPU-only; this closes the GPU execution gap while preserving the authoritative CPU contract. Host validation means `cuda_graph_compatible()` remains false.

<!-- merged from gorman-cuda-sparse-kv-gather-review.md -->

### 2026-07-18: CUDA SparseKvGather correctness review
**By:** Gorman
**What:** 🔴 REJECT commit `751a387`; reassign to Leon and lock Bishop out for this revision cycle.
**Why:** The `output_bytes == 0` early return skipped index validation for non-empty records with `D == 0`, allowing CUDA to accept negative or out-of-range indices that the CPU implementation rejects. Validation must run before the zero-output return whenever records are nonzero; add valid, negative, and upper-bound `D == 0` parity tests. Other registration, indexing, dtype, and graph-compatibility mechanics were correct.

<!-- merged from pris-pr25-test.md -->

### 2026-07-18: Exercise the real ORT Environment lifecycle in the plugin-cache regression
**By:** Pris
**What:** Replaced the simulated parallel regression with an isolated child-process test using real `Environment` values, the production lifecycle counter, and registration cache. It verifies live sharing, last-drop clearing, and that a fresh environment attempts registration instead of returning a stale-cache hit.
**Why:** A child process guarantees the real process-local 1 → 0 transition despite concurrent test harness activity. A missing plugin path provides evidence of a fresh registration attempt without requiring a shared-library fixture.

<!-- merged from vasquez-pr25-rereview.md -->

### 2026-07-18: PR #25 lifecycle fix re-review
**By:** Vasquez
**What:** 🔴 REJECT commit `8c96fba`; the production lifecycle fix is sound, but the regression test only exercises `SimulatedEnvironment` and a local registration map.
**Why:** It never calls production `Environment::new`, `Drop`, or `register_execution_provider_library`, so it would pass if either production lifecycle hook were removed. Pris must drive the actual create/drop/recreate/registration path; Newt and Deckard are locked out for this artifact revision.

<!-- merged from gorman-cuda-sparse-kv-gather-rereview.md -->

### 2026-07-18T01:20:34Z: CUDA SparseKvGather D==0 re-review
**By:** Gorman
**What:** 🟢 APPROVE commit `c2180c9`.
**Why:** For nonzero records, CUDA now copies and validates every index before the zero-byte-output return, so `D == 0` enforces negative-index and upper-bound errors through the same path as non-empty output. Zero-record cases safely skip validation; valid-length mapping and normal execution remain correct. The focused integration gate passed 12/12 tests.

<!-- merged from leon-cuda-sparse-kv-gather-fix.md -->

### 2026-07-18T01:20:34Z: Validate CUDA SparseKvGather indices for zero-width records
**By:** Leon
**What:** Moved host-side D2H index validation before the zero-output return and gated it on `records > 0`, preserving `valid_lengths` handling while allowing `D == 0` to return only after all indices are validated. Added valid, negative, equal-to-`C`, and greater-than-`C` CUDA/CPU parity tests.
**Why:** CUDA previously skipped validation whenever output bytes were zero, diverging from the CPU `out_of_range="error"` contract. The targeted suite passed 12/12 tests.

<!-- merged from vasquez-pr25-rereview-2.md -->

### 2026-07-18T01:20:34Z: PR #25 plugin lifecycle regression re-review
**By:** Vasquez
**What:** 🟢 APPROVE commit `dbff29c`.
**Why:** The rewritten test uses real `Environment::new` values and their real `Drop`, proving cache sharing while environments are live, retention after the first drop, clearing on final drop, and a fresh registration attempt after recreation. The isolated child process and PID-specific key prevent global-state leakage; Linux, Windows, and macOS checks are green.
