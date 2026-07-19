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

## 2026-07-18 — CUDA CSA Phase A and MTP Phase 1 landing decisions

The following reviewer chains record the two landed features in chronological order.

<!-- merged from ash-cuda-csa-phaseA.md -->
# Decision: CUDA host-staged CompressedSparseAttention (CSA) — Phase A

- Author: Ash (ep-cuda)
- Date: 2026-07-18T01:20:34Z
- Branch: `feat/cuda-csa-hoststaged` (worktree `/home/justinchu/wt-csa`, based on origin/main b09a8e8)
- Scope: correctness-first, host-staged CUDA execution of `pkg.nxrt::CompressedSparseAttention` v1.

## Reuse strategy — delegate to the CPU oracle (no math re-derivation)

The CUDA kernel does **not** re-derive any CSA math. It delegates to the fully-implemented CPU
kernel, which is the authoritative numerical oracle:

- `CompressedSparseAttentionFactory` (CUDA) builds the CPU kernel by calling the CPU
  `CompressedSparseAttentionFactory` (`onnx_runtime_ep_cpu::kernels::compressed_sparse_attention`)
  from the *same* `Node`, so the CPU kernel carries the identical, fully-validated frozen-v1 attribute
  configuration. The CUDA kernel wraps that `Box<dyn Kernel>`.
- At execute time the CUDA kernel: (1) D2H-copies every present device input into host buffers,
  (2) builds host-resident `TensorView`/`TensorMut` (DeviceId::cpu) over those buffers reusing each
  tensor's contiguous shape/strides, (3) runs the CPU oracle kernel verbatim, (4) H2D-uploads every
  host output back to its device buffer. This guarantees bit-parity by construction.

### ep-cpu visibility change: NONE

No ep-cpu change was required. `pub mod compressed_sparse_attention` and
`pub struct CompressedSparseAttentionFactory` were already public, and `KernelFactory`/`Kernel` are
public `onnx_runtime_ep_api` traits. The only build change was promoting `onnx-runtime-ep-cpu` from a
`[dev-dependencies]` to a `[dependencies]` entry in `crates/onnx-runtime-ep-cuda/Cargo.toml` so the
library (not just tests) can link the oracle. `git diff --stat -- crates/onnx-runtime-ep-cpu/` is empty.

## State threading (prefill → decode → decode)

CSA statefulness is expressed through the graph as ordinary `past_* → present_*` I/O tensors (the ONNX
KV-cache pattern), **not** as internal kernel state — the CPU kernel struct holds only config. The
host-staged CUDA kernel therefore reproduces the entire lifecycle for free: each step's `present_*`
device outputs are fed back as the next step's `past_*` device inputs by the session/caller. Because we
delegate to the CPU kernel, the compressed-cache, compression-carry (and, for ratio-4, index-key /
index-carry) evolution — including block-boundary emission of a fresh FP8-quantized record and carry
reset — is identical to the CPU oracle. Host-resident state (via the round trip) is accepted for this
correctness-first phase; device-resident state is Phase B.

## `supports_op` contract — accepted vs rejected (doc §4.8)

`crate::kernels::compressed_sparse_attention::unsupported_reason(node, input_dtypes)` gates claims,
wired into `provider.rs` alongside the other pkg.nxrt/attention gates:

- **Rejected at claim time** (never reaches `execute`):
  - Any attribute combination the CPU factory rejects, obtained by dry-running the CPU factory
    (`CpuCsaFactory.create(node, &[])`): `compression_ratio` other than 4 or 128; unknown
    `cache_format` (only `f32`, `fp8_e4m3_block64`, `fp4_e2m1_block32`); `sink_mode` other than
    `logit_only`; ratio-4 requiring positive `index_num_heads`/`index_head_dim`/`index_topk` while
    ratio-128 requires them zero; missing `num_heads`/`head_dim`; `qk_rope_head_dim > head_dim`;
    non-conforming `causal`/`cache_layout_version`/`index_layout_version`; input arity outside 11..=20
    or output arity outside 3..=6; any omitted required frozen-v1 input by name.
  - dtype mismatches on the dtype-fixed inputs: query(0)=f32, seqlens_k(8)=i32,
    total_sequence_length(9)=i64, head_sink(10)=f32, and past_compressed_kv(6) = uint8 for the
    block-quantized formats / f32 for `cache_format=f32`.
- **Accepted (claimed) then executed host-staged**: ratio-128 and ratio-4 frozen-v1 configs the CPU
  oracle supports (D=512, RD=64; ratio-4 additionally ID=128), with `cache_format` `f32` /
  `fp8_e4m3_block64` (and `fp4_e2m1_block32` for the ratio-4 index stream). Remaining shape checks that
  depend on runtime shapes are enforced identically by the delegated CPU kernel at execute time.

`cuda_graph_compatible()` returns **false** (host round trip + per-copy stream syncs), and
`supports_strided_input()` returns false (the host blit is dense; strided inputs are rejected in
execute with an actionable error).

## Phase-B TODO markers (device-resident kernel)

`crates/onnx-runtime-ep-cuda/src/kernels/compressed_sparse_attention.rs` carries a top-of-file
`// TODO(csa-cuda phase B): ...` referencing `docs/DEEPSEEK_CSA_MTP_RUNTIME.md §4.8`, calling out the
device-resident replacement: device-resident compressed cache/carry, fused
selection/score/sink-softmax/value-reduction, CUDA-graph capture, and elimination of the host round
trip. `cuda_graph_compatible()` also documents the Phase-B goal inline.

## Tests added (`tests/compressed_sparse_attention_gpu.rs`)

Uses the Rust `onnx_runtime_ir` graph builders (no Python `ir.to_proto`), mirroring the CPU kernel's
own ratio-128 test value generators so the oracle comparison is apples-to-apples:

- `ratio128_prefill_then_two_decodes_matches_cpu`: prefill (S=126, dense-window + sink / dense-fallback
  path, 0 compressed records) → decode@126 → decode@127 (crosses the 128-block boundary, emitting the
  first FP8-quantized compressed record and resetting the carry). Runs the SAME inputs through the CPU
  and CUDA kernels at every step and asserts bit-parity on `Y`, `present_compressed_kv` (exact bytes),
  and `present_compression_carry`, threading CPU `present_*` outputs into the next step's `past_*`.
- `supports_op_rejects_unsupported_configs`: a valid ratio-128 config is claimed; unsupported
  `compression_ratio=8`, unknown `cache_format`, and non-f32 query dtype are each rejected at claim time.

## Results

- `cargo test -p onnx-runtime-ep-cuda --test compressed_sparse_attention_gpu`: **2 passed, 0 failed**.
- `cargo test -p onnx-runtime-ep-cuda --lib`: **101 passed, 0 failed** (updated the
  `covered_ops_have_no_duplicates` count 84 → 85 for the new registered op).
- `cargo test -p onnx-runtime-ep-cuda --test sparse_kv_gather_gpu`: 12 passed (no registration
  regression).
- ep-cpu unchanged (no CPU behavior/signature change).

<!-- merged from mariette-cuda-csa-phaseA-review.md -->
### 2026-07-18T01:20:34Z: Reject CUDA CSA Phase A claim gating
**By:** Mariette
**What:** 🔴 REJECT commit `f1bd482`. Reassign the revision to **Leon**; Ash is locked out.
**Why:** The host-staging implementation itself faithfully copies every present input to host, invokes the CPU kernel built from the same `Node`, uploads every output, threads state through graph I/O, and explicitly disables CUDA graph capture. The ratio-128 test crosses the 128-record boundary and compares all three outputs at every step. However, `supports_op` is not a true gate. `unsupported_reason` checks only inputs 0, 6, 8, 9, and 10, while the CPU runtime requires many additional f32/u8 inputs. For example, a Float16 `current_kv` at input 1 is claimed and then fails in `execute`. The CPU factory dry-run also accepts configurations rejected only at runtime: ratio-4 with `index_head_dim != 128`, missing index inputs 11..18, fewer than five outputs, or a non-FP8 cache format; ratio-128 similarly over-claims FP4 cache format and ratio-4-only inputs/outputs. ORT can therefore place unsupported CSA nodes on CUDA and fail during execution.
**How:** Make claim validation mirror the CPU runtime's ratio-specific structural and dtype contract, using `shapes` where needed. Add a valid ratio-4 claim test plus negative claim tests for a non-query dtype (for example `current_kv`), invalid ratio-4 index dimensions, missing ratio-4 index state, and ratio-specific cache-format/output-count mismatches. The existing negative test only proves ratio, unknown-format, and query-dtype rejection, so it does not catch these over-claims.

**Verification:** CPU source diff is empty; registration is unique; `CUDA_COVERED_OPS` contains 85 unique entries; ep-cpu has no ep-cuda dependency. Focused CUDA CSA tests passed 2/2, and CUDA EP library tests passed 101/101.

<!-- merged from leon-cuda-csa-supports-op-fix.md -->
### 2026-07-18T01:20:34Z: CUDA CSA claim gate mirrors CPU ratio contracts
**By:** Leon
**What:** `CompressedSparseAttention` CUDA claim-time validation now passes static shapes into the CPU factory dry-run, requires complete positional shape/dtype metadata, and applies the CPU runtime's ratio-specific contract before claiming:
- Ratio 4: 19 or 20 inputs with all index inputs 11..18 present; 5 or 6 outputs; `head_dim=512`, `qk_rope_head_dim=64`, `index_head_dim=128`; FP8/BF16 attention cache only; f32 inputs 0..5, 7, 10..16, 18; u8 inputs 6 and 17; i32 input 8; i64 input 9; optional input 19 is additive f32 bias. Static input ranks/extents mirror the CPU ratio-4 path, including 8-slot 2D/index carries and packed widths 583/68.
- Ratio 128: index inputs 11..18 must be absent; exactly 3 outputs; `head_dim=512`, `qk_rope_head_dim=64`; f32 inputs 0..5, 7, 10; i32 input 8; i64 input 9; input 6 matches the cache format. FP4 is rejected. CPU-supported f32 and FP8/BF16 caches remain claimable, with packed widths 512/583 respectively. Static input ranks/extents mirror the CPU ratio-128 path.
**Why:** The prior gate checked only inputs 0, 6, 8, 9, and 10, so CUDA placement could claim nodes that the delegated CPU oracle rejected during `execute`. The CPU kernel remains the runtime source of truth; the host-staged compute and `cuda_graph_compatible() == false` are unchanged.

The documentation describes the production format as FP8/BF16 and uses logical/cache placeholders in §4.3, while the CPU runtime explicitly also accepts an f32 ratio-128 cache and stores packed caches as `[B, records, stored_width]`. The gate follows CPU behavior and therefore rejects ratio-128 FP4 without over-rejecting CPU-supported f32.

Tests added: valid ratio-4 claim plus CPU/CUDA parity across all six outputs; claim rejection for Float16 `current_kv`, ratio-4 `index_head_dim != 128`, missing inputs 11..18, fewer than five ratio-4 outputs, ratio-128 FP4, and a ratio-4-only input under ratio-128. Results: CSA integration 9 passed/0 failed; CUDA library 101 passed/0 failed.

<!-- merged from mariette-cuda-csa-phaseA-rereview.md -->
### 2026-07-18T01:20:34Z: 🔴 REJECT CUDA CSA Phase-A re-review
**By:** Mariette

**What:** Reject commit `e4442bf`. The six requested negative claim tests are load-bearing, valid ratio-4 and ratio-128 configurations remain claimed, ratio-4 CPU/CUDA parity passes, and the previously named failures are fixed. However, ratio-128 still over-claims when optional input 19 (`attention_bias`) is present.

**Why:** `validate_ratio128_claim` validates only inputs 0–10 and never checks input 19. The CPU runtime calls `AttentionBias::new`, which rejects non-`Float32` bias and rank greater than four. Therefore a ratio-128 node with a present Float16 (or rank-5) `attention_bias` is reported `Supported` and then fails inside `execute`, violating the required CPU-contract claim gate. Ratio-4 already performs these dtype/rank checks, demonstrating the missing ratio-128 branch.

**How:** Deckard should add shared optional-`attention_bias` claim validation for both ratios, including Float32 dtype, rank ≤ 4, and statically checkable broadcast dimensions, plus a negative ratio-128 claim test that appends absent slots 11–18 and a bad input 19. Ash and Leon are locked out.

**Verification:** CPU EP diff is empty; registration is unique; `CUDA_COVERED_OPS` remains 85 with one CSA entry; `cuda_graph_compatible()` remains false; the host-staging kernel body is byte-identical to the prior revision; determinism/state threading are untouched. CUDA CSA tests passed 9/9 and CUDA library tests passed 101/101.

<!-- merged from deckard-cuda-csa-bias-claim-fix.md -->
### 2026-07-18T01:20:34Z: Share CSA optional attention-bias claim validation
**By:** Deckard
**What:** Added `validate_attention_bias_claim`, invoked after either ratio-4 or ratio-128 structural validation. When optional input 19 is absent it accepts the node. When present it requires Float32, rank <= 4, a statically safe dense f32 layout, and right-aligned broadcasting to the attention-score shape `[batch, num_heads, sequence, candidates]`: static non-1 bias dimensions must equal the corresponding statically known target dimension; symbolic dimensions and the runtime-dependent candidate dimension remain claimable. Ratio-128 keeps inputs 11-18 absent and checks the bias at its actual positional slot 19.
**Why:** CPU `AttentionBias::new` applies the same dtype, rank, layout, and broadcast contract at execution. Sharing the claim helper prevents ratio-128 from being over-claimed while retaining ratio-4 behavior and avoiding rejection where broadcast compatibility cannot be disproved statically. Added one ratio-128 claim test covering Float16, rank-5, statically incompatible broadcast, and valid broadcastable f32 bias at input 19. CUDA CSA integration tests passed 10/10; CUDA EP library tests passed 101/101 (two pre-existing warnings).

<!-- merged from mariette-cuda-csa-phaseA-rereview2.md -->
### 2026-07-18T01:20:34Z: CUDA CSA phase-A third review approved
**By:** Mariette
**What:** 🟢 APPROVE commit `d23cac5` for the CUDA host-staged `CompressedSparseAttention` kernel.
**Why:** The shared input-19 validator now mirrors CPU `AttentionBias::new` for both ratio-4 and ratio-128: optional absence is accepted; present bias requires Float32, rank <= 4, safe static byte layout, and every statically knowable broadcast axis matches `[B, N, S, Candidate]`. Ratio-128 correctly preserves absent slots 11–18 before placing bias at index 19. The valid f32 broadcast case remains claimed, while dtype/rank/broadcast negatives are load-bearing because ratio-128 would otherwise claim them. No other optional input is ignored: 11–18 are fully validated for ratio-4 and forbidden for ratio-128, and 19 is now validated. Host-staging execution is byte-identical to `e4442bf`; ep-cpu and registration/count surfaces are unchanged. Verified 10/10 CSA integration tests and 101/101 CUDA library tests pass, including the 85-op count assertion.

<!-- merged from hudson-mtp-phase1.md -->
### 2026-07-18T01:20:34Z: DeepSeek/GLM MTP Phase 1 metadata and HC adapter
**By:** Hudson
**What:** Implemented native `proposal_type: mtp` resolution into `MtpConfig`, package-referenced target embedding/LM-head adapters, rank-4 Hyper-Connection state extraction/binding/threading, and one persistent MTP proposer per generation. Manual raw-f32 `MtpConfig` remains supported through file weight sources and the legacy `BSH`, `hc_mult=1`, no-`mtp_state` path.

The §6.7 metadata fields are:
- `proposal_type` string enum (`mtp`);
- required `model` string/path;
- `num_speculative_tokens` positive integer, default `4`;
- `target_hidden_output` string, default `hidden_states`;
- `target_hidden_layout` enum `BSH`/`BSHC`, default `BSHC`;
- required positive integers `target_hidden_size` and `hc_mult`;
- `mtp_hidden_output` string, default `mtp_hidden`;
- `mtp_state_output` string, default `mtp_state`;
- `kv_mode` enum `proposal_local`/`accepted_prefix`, default `proposal_local`;
- required `embedding` and `lm_head` objects, each with `source: target_initializer` and a non-empty exact initializer `name`.

Metadata carries exact initializer identity rather than a guessed filename or tensor name. The runtime inspects ONNX dtypes and currently borrows Float32, Float16, or BFloat16 initializer bytes from `WeightStore`; sidecar activations support Float32, Float16, and BFloat16. This follows §6.7 and the frozen configuration constants (§“Configuration constants that pin this contract”).

HC handling follows §2.4 and §6.3: target extraction preserves the final `[B,S,hc_mult,H]` row as `hc_mult*H`; `MtpDecodeSession` binds `hidden_states` as BSHC, returns separate `[B,S,H] mtp_hidden` and `[B,S,hc_mult,H] mtp_state`, and the proposer feeds only `mtp_hidden` to the LM head while threading `mtp_state` to the next draft. Absolute MTP positions use target length plus draft index. The proposer/session is constructed once per generation and reused across verification iterations (§6.1-§6.2).

**Blocked:** The released §2.4 sidecar described in the design exports only `mtp_hidden`; `hc_mult>1` iterative execution therefore still requires Mobius to export the explicit `mtp_state` required by §6.3. `accepted_prefix` parses but runtime reuse is rejected because the frozen numerical contract explicitly does not define correction-token/cache lifetime alignment. FP8/block-quantized target initializers remain blocked on wiring the runtime embedding/quantized matmul components; no quantization semantics were invented.

**Tests:** Added metadata-to-`MtpConfig`, malformed descriptor, validation, package-initializer parity, rank-4 `hc_mult=2` recurrent threading, and legacy `hc_mult=1` coverage. `cargo test -p onnx-genai-engine`: 147 passed, 0 failed, 10 ignored. `cargo check -p onnx-genai-engine`: passed. Metadata tests: 23 passed. Targeted ORT MTP test: passed.

**Why:** Phase 1 must make the approved package contract executable without hand-built raw-weight configuration while preserving the existing speculative verify/correction state machine and refusing to guess recurrent-state or cache semantics absent from the frozen contract.

<!-- merged from ripley-mtp-phase1-review.md -->
### 2026-07-18T01:20:34Z: Reject MTP Phase 1 backward-compatibility break
**By:** Ripley
**What:** 🔴 REJECT commit `2243968`. Reassign the revision to **Batty**; Hudson is locked out.
**Why:** The metadata-to-runtime mapping, rank-4 `[B,S,hc_mult,H]` threading, `hc_mult=2` recurrence test, proposer lifetime, and proposal-local reset behavior are sound. The blocked `mtp_state`, `accepted_prefix`, and quantized-adapter work is legitimately unfrozen by the pinned contract. However, the public manual `MtpConfig` API is source-breaking: five required fields were added, and `embedding_weights`/`lm_head_weights` changed from `PathBuf` to `MtpWeightSource`. Every existing external `SpeculativeMode::Mtp(MtpConfig { ... })` consumer now fails to compile. The repository test was updated with new fields and `.into()`, so it does not prove backward compatibility.
**How:** Restore the original public `MtpConfig` struct-literal contract unchanged. Resolve metadata-only layout, HC, output-name, cache-scope, and initializer-reference data into a separate internal configuration path (or an additive API that does not alter `SpeculativeMode::Mtp(MtpConfig)`). Add a compile-time compatibility fixture using the pre-`2243968` struct literal verbatim, while retaining the metadata and `hc_mult > 1` tests.

**Verification:** Targeted metadata, engine MTP, and ORT MTP tests passed; the ignored MTP greedy-equivalence test also passed when run explicitly. The rejection is specifically for the public source-compatibility regression.

<!-- merged from batty-mtp-config-compat-fix.md -->
### 2026-07-18T01:20:34Z: Preserve the public MTP configuration contract
**By:** Batty
**What:** Restored `MtpConfig` to its pre-`2243968` eight-field public struct, including `PathBuf` embedding and LM-head fields. Metadata-only layout, Hyper-Connection, sidecar output/cache, and initializer-source settings now live in crate-private `ResolvedMtpConfig`; manual configs resolve to legacy BSH, `hc_mult = 1`, `mtp_hidden`, no recurrent state, proposal-local cache, and file-backed weights. Added and exercised `pre_phase1_mtp_config_literal_remains_source_compatible` using the original struct literal.
**Why:** Existing external `SpeculativeMode::Mtp(MtpConfig { ... })` consumers must remain source-compatible while metadata-driven DeepSeek/GLM MTP retains Phase-1 behavior. Validation passed: engine tests 148 passed, 0 failed, 10 ignored; compatibility fixture 1 passed; `cargo check` passed for `onnx-genai-engine` and `onnx-genai-ort`.

<!-- merged from ripley-mtp-phase1-rereview.md -->
# Ripley MTP Phase 1 Re-review

- **CURRENT_DATETIME:** 2026-07-18T01:20:34Z
- **Commit:** `ea92bf5`
- **Verdict:** 🟢 APPROVE

The public eight-field `MtpConfig` contract exactly matches its pre-`2243968` definition, including `PathBuf` weight fields. The compatibility test uses the original struct literal without `Default` masking and passes. `ResolvedMtpConfig` preserves legacy manual defaults while retaining metadata resolution, rank-4 HC threading, hc_mult=2 recurrence coverage, persistent per-generation proposer lifetime, proposal-local reset, and malformed-descriptor rejection.

Validation: `cargo test -p onnx-genai-engine` passed all 148 non-ignored tests; `cargo check -p onnx-genai-ort` passed.

## 2026-07-18 — Scribe inbox merge (03:50Z)

<!-- merged from frost-csa-phase2-audit.md -->

### 2026-07-18T01:20:34Z: Phase 2 CPU learned-sink and sparse-gather audit
**By:** Frost
**What:** Audited every Phase 2 bullet and added only missing edge coverage plus explicit sink taxonomy errors. Code commit: `83ec096cf13695f6b6bf71f8a9154e857af4704d`.
**Why:** Most Phase 2 implementation had already landed in `c5bdafd`; the remaining gaps were explicit negative/cache-end bounds coverage, an explicitly named empty-compressed-prefix case, multi-axis deterministic-layout coverage, and an actionable error distinguishing learned logit sinks from retained sink tokens.

## Per-bullet verdict

1. **`head_sink` in the CPU dense attention reference — already satisfied; no `attention.rs` change.**
   - `attention.rs` is specifically the standard `ai.onnx::Attention` operator (`crates/onnx-runtime-ep-cpu/src/kernels/attention.rs:1-13`) and enforces that standard schema's 3..=7 inputs (`attention.rs:324-326`). It has no schema slot for a private DeepSeek `head_sink`; extending it would change standard ONNX operator semantics.
   - The Phase 2 dense reference meant by the CSA document is the assembled-cache/decomposed reference path: `CompressedSparseAttentionKernel` (`compressed_sparse_attention.rs:58-66`, factory seam at `:269-275`, execution at `:2474-2650`). It gathers selected records, computes explicit scores, and adds the learned per-head sink only to the denominator at `:2613-2643`.
   - The independent scalar-oracle test already existed at `compressed_sparse_attention.rs:3414-3471`.

2. **CPU `SparseKvGather` v1 — already implemented and registered.**
   - Factory and v1 attribute validation: `sparse_kv_gather.rs:39-69`.
   - Kernel execution: `sparse_kv_gather.rs:72-105`.
   - `pkg.nxrt::SparseKvGather` v1 registration: `crates/onnx-runtime-ep-cpu/src/kernels/mod.rs:224-227`.
   - Existing tests already covered duplicates/order (`sparse_kv_gather.rs:607-623`), valid-length bounds with exact coordinates (`:638-650`), `-1` masks (`:669-680`), contiguous empty selection (`:683-687`), and frozen candidate ordering (`:738-805`).
   - Missing explicit cases added: negative and `index == C` bounds (`:653-666`), deterministic `[B,G,Q,K,D]` layout (`:690-717`), and empty compressed prefix masking (`:720-735`).

3. **Learned logit sink vs. `sink_tokens` taxonomy — metadata was distinct; error text was incomplete and is now fixed.**
   - Inference metadata defines `sink_tokens` only as StreamingLLM retained leading tokens (`crates/onnx-genai-metadata/src/schema.rs:341-346`); engine parsing repeats that retained-token meaning (`crates/onnx-genai-engine/src/decode.rs:1010-1018`).
   - CSA node metadata parses `sink_mode` independently. Its rejection now explicitly says `head_sink` is a learned per-head logit input while metadata `sink_tokens` configures unrelated retained prefix tokens (`compressed_sparse_attention.rs:197-212`).
   - Added a metadata/error regression test at `compressed_sparse_attention.rs:3475-3494`.

## Frozen sink formula

Used the exact contract from `docs/DEEPSEEK_CSA_MTP_RUNTIME.md` §4.5 (`:533-543`) and the frozen online formula (`:1360-1388`):

```text
m = max(real_scores)
Z = sum_j exp(real_score_j - m) + exp(head_sink[h] - m)
O = sum_j exp(real_score_j - m) * V_j / Z
```

The sink is not included in `m`, contributes only the extra denominator term, has no value vector, and therefore contributes nothing to the numerator/output.

## Validation

- Focused sparse-gather tests: 11 passed, 0 failed.
- Learned-sink taxonomy test: 1 passed, 0 failed.
- Existing independent scalar sink oracle: 1 passed, 0 failed.
- Full `cargo test -p onnx-runtime-ep-cpu`: 503 passed, 0 failed, 1 ignored; doctests 0 passed, 0 failed, 1 ignored.
- No engine or metadata source changed, so the conditional engine check was not required.
- Blocked sub-pieces: none.

<!-- merged from gorman-frost-phase2-review.md -->

### 2026-07-18: Frost CSA Phase 2 CPU review — APPROVED
**By:** Gorman
**What:** 🟢 APPROVE of Frost's commit `83ec096` (CSA Phase 2 CPU). Only runtime change is the clarified `sink_mode != "logit_only"` error string distinguishing the learned per-head logit input `head_sink` from metadata `sink_tokens` — no semantics changed. Hand-checked the new `gather_uses_deterministic_b_g_q_k_d_layout` expected array against the B/G/Q/K/D layout + index table, and the ratio-4 `-1`/cache-end (`4`) empty-prefix expectations against the frozen formula. Deterministic ordering preserved (no HashMap iteration in gather path). Gate: 503 passed, 0 failed, 1 ignored (`-p onnx-runtime-ep-cpu`).
**Why:** Reviewer verdict for the strict-lockout protocol; landed to main as `83ec096`.

<!-- merged from keaton-cuda-csa-phaseb-plan.md -->

# Decision: CUDA CSA Phase B — phased implementation plan

- **Date:** 2026-07-18T01:20:34Z
- **Author:** Keaton (CUDA architect)
- **Artifact:** `docs/CUDA_CSA_PHASE_B_PLAN.md` (branch `docs/cuda-csa-phaseb-plan`, based on origin/main 73629cd)

## Summary

Phase B (the device-resident fused CUDA `CompressedSparseAttention` kernel replacing the Phase A host-staged path) is decomposed into eight independently-landable, CPU-parity-gated sub-phases: **B0** device-execution scaffolding + per-stage Host/Device dispatch + FP8/FP4 quant round-trip primitives (no numeric change); **B1** device sparse sink-softmax attention core for ratio-128 (state still host-staged); **B2** device ratio-128 compression + device-resident FP8/f32 cache & carry (ratio-128 fully device-resident); **B3** device ratio-4 FP4 index-key compression; **B4** device ratio-4 index scoring + deterministic top-k selection; **B5** device ratio-4 fused selection→attention (ratio-4 fully device-resident); **B6** CUDA-graph capture compatibility (stable addresses, device cursors, graph-safe top-k, flip `cuda_graph_compatible()`→true); **B7** stream-ordered checkpoint/restore for speculative decode + switch the default off the host-staged fallback (retained behind a debug flag). Each slice keeps the Phase A host-staged path as a correctness fallback via a stage dispatch flag, lands with existing GPU parity tests green, and has an explicit pass bar + rollback (flip the stage back to Host). Hardest slices: B4 (top-k tie determinism) and B1 (attention numerics). 

## Decisions for Justin

- **D1** Parity target: CPU-f32 oracle (current gate) vs. official BF16 kernel.py numerics — they differ (CPU oracle accumulates attention in f32; kernel.py casts `p_j` to BF16). Recommend targeting the CPU oracle.
- **D2** FP8/FP4 device compute: extract/reuse `block_quantized_matmul` decode helpers + add device *quantize*, vs. self-contained CSA quant module. Recommend a shared, graph-safe quant/dequant NVRTC snippet.
- **D3** Device cache: fixed-capacity (required for stable addresses / graph capture) sized from `max_seq_len`; confirm `max_seq_len`/window `W` budget and fail-closed cap.
- **D4** Confirm equal-length per-batch cursors for v1 (ragged deferred, per §10-Q10).
- **D5** Device top-k determinism & graph-capturability: accept an index-only host readback until B6, then fully device-resident/capturable.
- **D6** Checkpoint/restore ownership: kernel owns device cursors, engine drives `checkpoint()`/`restore(base_len+accepted)`, restore clears tails without recompress.
- **D7** Retire host-staged path in B7 or keep behind a `--csa-oracle` debug flag (recommend keep for triage; never default).

<!-- merged from hudson-mtp-phase1-remaining.md -->

### 2026-07-18: MTP Phase 1 remaining-bullet audit — Phase 1 complete
**By:** Hudson
**What:** Audited the remaining MTP Phase 1 bullets in `docs/DEEPSEEK_CSA_MTP_RUNTIME.md` against the engine implementation. Classification:
- Metadata resolution — already implemented
- Package embedding / LM-head references — already implemented
- Rank-4 Hyper-Connection extraction — already implemented
- BSHC sidecar hidden_states binding — already implemented
- Persistent per-generation proposer — already implemented
- Greedy draft/verify/correction reuse — already implemented
- Explicit `mtp_state` — legitimately blocked (unfrozen for released Mobius packages)

Implementable-now: **none**. No source changes, no commit. Tests: engine 148 passed / 10 ignored; ORT MTP 2 passed; MTP greedy-equivalence 1 passed.
**Why:** Confirms MTP Phase 1 is functionally complete; the only outstanding item (`mtp_state`) remains user/Mobius-contract-blocked. No further Phase 1 engine work until the contract is frozen.


## 2026-07-18 — Scribe inbox merge (EP omitted-optional contract and IndexShare audit)

<!-- merged from wallace-session-coverage.md -->

### 2026-07-18: Preserve omitted optional-input dtype during EP claims
**By:** Wallace
**What:** Fixed `onnx-runtime-session` planning to pass `DataType::Undefined` (not a silent `Float32` fallback) for each interior omitted optional input in `supports_op` calls. Added a regression EP that refuses the former fake Float32 signature and accepts `[Float32, Undefined, Bool]`.
**Why:** Claim-time dtype validation must distinguish an ONNX omitted optional input from a supplied Float32 tensor. The old fallback could make an EP accept or reject a node on false dtype metadata, masking a provider contract violation until compilation or execution. Coverage rose from 84.64% to 84.72% regions and 79.41% to 79.49% lines; `cargo test -p onnx-runtime-session` passed 148 tests.

<!-- merged from mariette-wallace-optional-dtype-review.md -->

### 2026-07-18: REJECT Wallace's omitted-optional dtype claim fix
**By:** Mariette
**What:** The session change correctly makes `NodePlan::input_dtypes` positional and maps an omitted `None` input to `DataType::Undefined`; its revised documentation is accurate. The regression test proves that contract. CSA is safe: both CPU execution paths determine `attention_bias` presence from the bound optional slot (`get(19)` plus `!is_absent()`), and CUDA claim validation first checks `node.inputs[19].is_some()` before reading dtype 19. CPU `Attention` does not inspect claim dtypes. CUDA `RotaryEmbedding` checks only required inputs 0–2; optional `position_ids` is slot 3 and is not read by its claim-time dtype gate.

CUDA standard `Attention` is not safe. Its `unsupported_reason` iterates dtype indices `[0, 1, 2, 4, 5]` without checking `node.inputs`. Inputs 4 and 5 (`past_key`, `past_value`) are optional and execution itself correctly detects them with `!inputs[index].is_absent()`. After this change, an omitted past-KV pair reaches the gate as `Undefined`, which is rejected as non-f32. Before this commit, the accidental Float32 placeholder let the same valid no-past Attention node be claimed. This is a claim regression.

**Required revision owner:** Nabil (not Wallace or Mariette).

**Required change:** Change CUDA standard-Attention claim validation to use node-slot presence for optional positions. Pass `&Node` to `standard_attention::unsupported_reason` from `CudaExecutionProvider::supports_op`, always validate required Q/K/V slots 0–2 as f32, and validate slots 4 and 5 as f32 only when `node.inputs.get(index).is_some_and(Option::is_some)`. Add a regression claim test for an Attention node with f32 Q/K/V and omitted mask/past slots whose positional dtypes are `Undefined`; it must be supported. Preserve rejection for a present non-f32 past input.

**Validation:** `cargo test -p onnx-runtime-session -p onnx-runtime-ep-cpu` passed 649 / failed 0. `cargo test -p onnx-runtime-ep-cuda` passed 233 / failed 0. The CUDA suite includes deterministic standard-Attention and RotaryEmbedding tests, so existing execution determinism remains covered; no cuDNN-related failures occurred.

<!-- merged from nabil-cuda-attention-optional-fix.md -->

### 2026-07-18: CUDA Attention claims omitted optional inputs correctly
**By:** Nabil
**What:** Updated standard `ai.onnx::Attention` CUDA claim validation for all optional schema positions: `attn_mask` (input 3), `past_key` (4), `past_value` (5), and opset-24 `nonpad_kv_seqlen` (6). `DataType::Undefined` now means the slot is absent; supplied masks must be bool/f32, supplied past caches must be f32 and paired, and supplied nonpad lengths must be int64, opset-24+, and mutually exclusive with in-op past caches. Added a graph-builder regression proving omitted past KV is claimed while a present non-f32 `past_key` is rejected.
**Why:** Session planning now preserves omitted optional slots as `DataType::Undefined` instead of the old f32 placeholder. CUDA Attention's f32-only loop treated that absence marker as a real wrong-typed cache, blocking valid GLM/Mobius prefill. The revised gate mirrors CPU presence semantics without weakening supplied-tensor validation. No other CUDA claim gate compares an optional input slot against f32; RotaryEmbedding's claim check covers only its three required floating inputs.

<!-- merged from mariette-nabil-rereview.md -->

### 2026-07-18: APPROVE Nabil's CUDA Attention omitted-optional claim fix (`8eb23f1`)
**By:** Mariette
**Verdict:** 🟢 APPROVE

**What:** `standard_attention::unsupported_reason(opset, input_dtypes)` now treats `DataType::Undefined` as an absent optional input for every optional standard-Attention position: `attn_mask` (3), `past_key` (4), `past_value` (5), and `nonpad_kv_seqlen` (6). Required Q/K/V (0–2) remain f32-only. A supplied mask remains bool/f32-only, supplied past KV remains f32-only and paired, and supplied nonpad length remains int64-only, opset-24+, and mutually exclusive with in-op past KV.

**Why:** This precisely resolves the prior claim regression: an Attention node with omitted mask/past slots and positional `Undefined` dtypes is now claimed. The added regression constructs that exact node and calls `supports_op`, which is the previously failing claim path under `848ad87`; before this change its omitted slot 4 was rejected by the f32-only loop. It also constructs a node with an actual `Int64` `past_key` plus f32 `past_value` and verifies rejection, preserving claim-then-fail protection.

**Contract review:** The CUDA claim semantics mirror CPU execution's input-slot contract: CPU treats optional mask, past KV, and nonpad inputs as provided only when their binding is non-absent, and enforces the same mask/past/nonpad type and compatibility rules at execution. Under the session's positional dtype contract, `Undefined` represents that absent binding. No required or supplied dtype validation was weakened. `cuda_graph_compatible()` is unchanged and remains `false`.

**Validation:** `cargo test -p onnx-runtime-ep-cuda` passed 234 / failed 0 (including the new omitted-vs-wrong-typed past-cache claim regression). `cargo test -p onnx-runtime-session -p onnx-runtime-ep-cpu` passed 649 / failed 0. No cuDNN failures occurred.

<!-- merged from ferro-indexshare-selected-token.md -->

### 2026-07-18: IndexShare selected-token attention is contract-blocked
**By:** Ferro
**What:** Audited the GLM IndexShare fallback and the DeepSeek CSA sparse path. No runtime source was changed because the GLM production-op contract is not frozen.

The current GLM fallback is emitted by Mobius PR #404. `GlmMoeDsaIndexer.select` computes IndexShare scores and `TopK` indices; `_sparse_bias` expands a dense FLOAT-min tensor over the complete key length, scatters zero at selected indices, adds the ordinary causal/padding attention bias, and feeds that dense mask plus full K/V (and past K/V) to standard `ai.onnx::Attention`. The CPU Attention kernel materializes the full present K/V cache and computes/scans scores and value accumulation across `total_seq`, so it is correct but dense over the cache even though unselected logits underflow to zero probability.

DeepSeek CSA ratio-4 already implements selected-record attention. Inputs 11–18 are mandatory index-query/weight/compressor/state inputs; they build the FP4 index-key stream, select top-k compressed records, and `ratio4_attention` scores only the dense 128-token window plus those selected compressed records. The stateful hot path indexes selected records directly. `SparseKvGather` is the reusable checked gather and is used by CSA's assembled-cache/decomposed reference path; it is not evidence that GLM IndexShare has a production handler. The approved shared-boundary decision explicitly says DeepSeek CSA and GLM IndexShare require separate fused ops because selection semantics differ.

For a fixed, valid selected set, the additive-mask standard-Attention result is a clean correctness oracle: a sparse implementation can gather the same K/V rows and match the dense fallback, including empty/all-selected cases. However, an in-tree implementation cannot currently be wired without inventing the GLM private-op ABI and numerical ordering contract.

**Decisions for Justin:**
1. Freeze the `pkg.nxrt` GLM op name/version and boundary: does it consume exporter-computed top-k indices, or own full/shared IndexShare selection plus index-key cache/state?
2. Freeze index semantics: ordered list versus set, duplicate policy, `-1` sentinel behavior, out-of-range behavior, and empty selection.
3. Freeze deterministic/numerical parity: preserve incoming `TopK(sorted=0)` order, or canonicalize selected cache indices into dense-cache order to reproduce the additive-mask accumulation order; specify exact f32 equality versus tolerance.
4. Freeze mask/cache ABI: composition with causal/padding bias, past/present K/V outputs, supported layouts/head sharing, and whether shared-layer indices are explicit inputs/outputs.

**Why:** Adding a helper alone would not make exported GLM graphs use selected-token attention, while extending standard ONNX Attention with private index inputs would violate its schema. The additive-mask path supplies a strong parity oracle after the GLM boundary is frozen, but choosing that boundary and its ordering/cache semantics is a user-visible FROZEN-contract decision.



## 2026-07-18 — Scribe inbox merge (GLM readiness and CSA CPU claim validation)

<!-- merged from newt-glm-readiness-refresh.md -->

### 2026-07-18: Refresh GLM readiness and consolidate owner decisions
**By:** Newt
**What:** Refreshed `docs/GLM_READINESS_GAPS.md` at `main` `8d9c958` to record GPU-native CUDA Attention, RotaryEmbedding, and SparseKvGather; CUDA CSA Phase A; the optional-input `DataType::Undefined` claim fix; and functionally complete MTP Phase 1. Added `docs/DECISIONS_FOR_JUSTIN.md` with 32 open decision points: CSA 7, BlockQuantizedMoE 8, Kimi K3 5, IndexShare 4, GraphView 5, MLA 1, and Mobius exports 2. Committed as `22841f1`.
**Why:** The previous readiness audit still described landed CUDA paths as missing or host-staged, and owner-blocking roadmap decisions were scattered across team state and design documents. The refresh gives Justin one evidence-linked morning checklist without changing runtime source.

<!-- merged from bishop-csa-cpu-coverage.md -->

### 2026-07-18: Harden CSA CPU claim gate and sparse planner validation
**By:** Bishop
**What:** Fixed the CSA factory to reject ratio-4 graphs missing positional index inputs 11..=18 or index-state outputs, and ratio-128 graphs carrying ratio-4 inputs or extra outputs, at claim time. Sparse ratio-4 prefill/decode planners now reject an empty learned top-k rather than constructing a zero-width selection. Added regression coverage for these claims, attention-bias dtype/rank/broadcast rejection, FP8/FP4 block widths, masked-gather sentinel/bounds checks, and empty top-k planners.
**Why:** The previous factory could claim invalid ratio-specific graphs and fail only when executed; empty learned top-k also silently produced a degenerate ratio-4 plan. Full CPU tests pass (508 passed, 1 ignored). llvm-cov line coverage improved from 85.27% to 85.95% for compressed_sparse_attention.rs and from 78.63% to 79.39% for sparse_kv_gather.rs.

<!-- merged from deckard-bishop-csa-review.md -->

### 2026-07-18: Review of `af0725c` CSA ratio-specific validation
**By:** Deckard
**Verdict:** 🔴 REJECT
**Finding:** CUDA is consistent: its `unsupported_reason` first dry-runs `CpuCsaFactory.create` and also independently enforces ratio-4 inputs 11..=18 plus 5–6 outputs, and ratio-128 absent inputs 11..=18 plus exactly 3 outputs. The CPU EP's actual `supports_op`, however, only checks registry membership and unconditionally claims every registered CSA node. It therefore accepts each invalid ratio-specific schema that the new CPU factory rejects, retaining a CPU claim-then-fail path and violating the requested cross-EP claim contract.
**Required change (Leon):** Add CSA-specific validation to `CpuExecutionProvider::supports_op` (or an equivalent per-op claim hook) that invokes the same frozen/ratio-specific schema validation as the factory, and add provider-level tests proving the three newly rejected malformed schemas are declined before `get_kernel`.
**Other review results:** The new `unreachable!` is safe because `create_impl` validates `compression_ratio ∈ {4,128}` immediately before calling the helper; the assembled-cache path bypasses the helper. Empty learned top-k produces clean errors before zero-width plan construction and preserves deterministic ordering. CPU tests: 508 passed, 0 failed, 1 ignored (doctests: 0 passed, 0 failed, 1 ignored). CUDA tests: 234 passed, 0 failed; including `compressed_sparse_attention_gpu`: 10 passed, 0 failed.

<!-- merged from leon-csa-cpu-claim-validation.md -->

### 2026-07-18: CPU CSA claim-time contract validation
**By:** Leon
**What:** Added a `pkg.nxrt::CompressedSparseAttention` claim hook to `CpuExecutionProvider::supports_op`. The hook rejects malformed ratio-4/ratio-128 positional arity through a dry-run of `CompressedSparseAttentionFactory`, then applies the CUDA-equivalent fixed ratio, input dtype, input shape, and optional `attention_bias` checks. Added provider-level denials for ratio-4 missing-index and wrong-output cases plus ratio-128 index-present and wrong-output cases.
**Why:** CPU previously claimed every registry-known CSA node and deferred malformed ratio-specific schemas until factory creation. Dry-running the same factory from the claim hook makes the factory's frozen-v1 and ratio-specific validation the single source of truth, preventing claim-then-fail drift while preserving the existing assembled-cache reference path.

<!-- merged from deckard-leon-rereview.md -->

### 2026-07-18: Re-review of Leon's CPU CSA claim validation (`6c9cfd1`)
**By:** Deckard
**Verdict:** 🟢 APPROVE
**What:** The CPU provider now calls CSA `unsupported_reason` from `supports_op`. It dry-runs the same `CompressedSparseAttentionFactory.create` used by `get_kernel`, so frozen-v1 attributes and ratio-specific positional-input/output arity are denied before placement. The provider-level test invokes `ep.supports_op` directly and denies ratio-4 missing-index and wrong-output nodes plus ratio-128 index-present and wrong-output nodes. The remaining dtype, static-shape, and optional-bias checks mirror the runtime CSA paths; valid/dynamic metadata is not rejected merely for being dynamic.
**Why:** The prior CPU claim-then-fail gap is closed without loosening Bishop's factory checks or changing the assembled-cache-reference bypass. The dry-run only parses/validates metadata and boxes a scalar kernel descriptor; it allocates no device or tensor buffers and performs no copies, cache decoding, or compute, so it is cheap for repeated placement. Deterministic execution and `assembled_cache_reference` semantics remain intact. Validation passed: CPU 509 passed, 0 failed, 1 ignored (plus 0/0/1 doctest); CUDA 234 passed, 0 failed, 0 ignored.

## 2026-07-18 — Scribe inbox merge (CUDA GLM claim-gate hardening)

### 2026-07-18: Harden CUDA GLM standard-op claim contracts
**By:** Holden
**What:** Audited CUDA claim gates against their runtime contracts for the GLM standard-op path. Findings:

| Op | Finding |
|---|---|
| `RMSNormalization` | Bug: it claimed non-f32 X/scale although the CUDA kernel is f32-only; it also silently accepted unsupported `stash_type`. |
| `RotaryEmbedding` | Bug: required f32 inputs were checked, but a present optional `position_ids` with a non-int64 dtype was claimed. Explicit omitted slot 3 (`Undefined`) is correctly treated absent. Negative dimensions/non-boolean `interleaved` were silently coerced by the factory. |
| `TopK` | Bug: it claimed non-f32 values/non-int64 K, then failed at execution; non-boolean `largest`/`sorted` were silently coerced. |
| `CumSum` | Bug: it claimed unsupported data or non-int64 axis, then failed at execution; non-boolean flags were silently coerced. |
| `Gather` | Bug: it claimed non-integer indices and packed/variable-width data, then failed at execution. |
| `GatherElements` | Bug: it claimed non-int64 indices and packed/variable-width data, then failed at execution. |
| `ScatterElements` | Bug: it claimed non-int64 indices, unsupported data/updates, then failed at execution; malformed reduction attributes were deferred. |
| `Where` | Bug: it claimed a non-bool condition or packed/variable-width branches, then failed at execution. |
| `Expand` | Bug: it claimed a non-int64 ONNX shape input, despite the schema contract. |
| `CompressedSparseAttention` | OK: ratio-specific factory dry-run plus dtype/shape validation correctly rejects invalid present `attention_bias`; an explicit omitted input 19 with `Undefined` is absent and claims. |

Also rechecked standard `Attention`: the landed omitted-optional gate remains correct.

**Why:** Added a shared standard-op CUDA claim validator that rejects each runtime-unsupported required input before placement, while preserving the `Undefined` omitted-optional contract for RoPE. Factories now reject attributes that previously silently coerced values. Added `claim_gates_gpu` coverage for all repaired standard-op dtype gates, RoPE omitted-vs-present optional behavior, and invalid attribute gates; added CSA input-19 omission coverage. Full CUDA EP suite passed 238 tests, 0 failed.

### 2026-07-18: Approve CUDA standard claim-gate hardening
**By:** Mariette
**What:** 🟢 APPROVE Holden's `030faa1` (`fix(cuda): harden GLM standard claim gates`).

| Op | New claim requirement | GLM / CPU / CUDA parity conclusion |
|---|---|---|
| RMSNormalization | f32 X/scale; `stash_type=1` | GLM's portable profile is f32; CUDA is f32-only and CPU accepts this subset. |
| RotaryEmbedding | f32 X/cos/sin; present positions int64; valid boolean/non-negative attrs | Matches GLM's f32 and int64-position contract; an omitted `Undefined` slot remains claimed. |
| TopK | f32 X, int64 K, valid boolean attrs | Matches documented GLM f32 values and int64 scalar K; CUDA execution has the same limits. |
| CumSum | f32 or int64 X, int64 axis, valid boolean attrs | Matches the GLM contract and CPU/CUDA supported subset. |
| Gather | fixed-width data; int32/int64 indices | Matches the CUDA byte-copy kernel and GLM integer indexing. |
| GatherElements | fixed-width data; int64 indices | Matches CUDA's int64-only index kernel and GLM usage. |
| ScatterElements | f32/int64 matched data/updates; int64 indices; valid reduction | Matches the constrained CUDA kernel and GLM contract. |
| Where | bool condition; matched fixed-width branches | Matches CUDA execution and GLM use. |
| Expand | fixed-width input; int64 shape | Matches the ONNX shape-input contract and CUDA movement kernel. |

**Why:** The shared helper is correctly limited to standard domains, is called only after registry lookup, checks metadata arity before indexing, and preserves RoPE's omitted optional input contract. New factory validation gives actionable errors instead of coercing the audited attributes; CUDA-graph compatibility methods were unchanged. No GLM over-rejection or remaining audited dtype/attribute claim-then-fail path was found. The CUDA EP suite passed 238 tests and failed 0 (missing-cuDNN failures were not present).

---

### 2026-07-18: Reserve the nxrt PyPI name with an sdist-only publish
**By:** Deckard
**What:** Added an independent `publish-pypi-sdist` job to `publish.yml` that publishes only the `nxrt` source distribution through the `pypi` environment using PyPI Trusted Publishing. The initial burned version is `0.1.0.dev2`. Workflow dispatch now has opt-in `publish_crates` and `publish_pypi` inputs.
**Why:** A reliable sdist-only release reserves the PyPI name without coupling it to crates.io publication or prematurely publishing platform wheels; wheels will ship later through `wheels.yml`.

<!-- merged from deckard-onnx-genai-pypi.md -->

### 2026-07-18: Reserve the `onnx-genai` PyPI name
**By:** Deckard
**What:** Reserved the `onnx-genai` PyPI name with a pure-Python placeholder at `python/onnx-genai/`, fixed at version `0.0.0`. Added the dispatch-only `publish-onnx-genai-sdist` job in `publish.yml`, publishing through PyPI Trusted Publishing with the `pypi` environment.
**Why:** The placeholder and opt-in sdist publication reserve the package name without coupling it to the future native implementation or platform wheel release path.


## 2026-07-19 — Scribe inbox merge (BQMoE v1, PR #30, and PR #34)

<!-- merged from sapper-bqmoe-v1.md -->

### 2026-07-19: BlockQuantizedMoE v1 CPU reference and frozen ABI
**By:** Sapper
**What:** Landed the CPU parity-oracle implementation for `pkg.nxrt::BlockQuantizedMoE` v1. Frozen inputs are: 0 `input` f32; 1 `router_logits` f32; 2 `fc1_experts_weights` u8; 3 optional `fc1_experts_bias` f32; 4 `fc2_experts_weights` u8; 5 optional `fc2_experts_bias` f32; 6 optional `fc3_experts_weights` u8; 7 optional `fc3_experts_bias` f32; 8 optional `router_weights` f32. Frozen attributes are `format`, `block_layout_version`, `k`, `activation_type`, `normalize_routing_weights`, `swiglu_fusion`, `activation_alpha`, `activation_beta`, and `swiglu_limit`.
**Why:** The CPU kernel is the deterministic numerical oracle for the verified GLM/IQ profile. It shares `BlockFormat` and `dequantize_weight_kn` with `BlockQuantizedMatMul`, materializes resident packed tensors before selected-expert decoding, and defers expert-slice/device paging to D7. The CPU claim gate dry-runs attribute parsing and validates positional arity, omitted-input `Undefined` dtypes, concrete dtypes, and available shape metadata so unsupported nodes are declined before execution.

<!-- merged from chew-bqmoe-rereview.md -->

### 2026-07-19: BQMoE v1 re-review
**By:** Chew
**What:** 🔴 REJECT. The symbolic-shape claim-then-fail gap is closed: the partial validator covers all statically knowable execution shape relationships, optional `Undefined` slots are accepted as absent, present input dtypes are checked, the selected-expert and activation tests discriminate, and the frozen ABI matches. However, the claim path still violates the hardened zero-allocation requirement.
**Why:** `unsupported_reason` calls `BlockQuantizedMoEFactory.create` at `crates/onnx-runtime-ep-cpu/src/kernels/block_quantized_moe.rs:60`; a valid claim reaches `Ok(Box::new(BlockQuantizedMoEKernel { ... }))` at line 51, allocating a heap object solely to validate attributes. Claiming is required to be a cheap metadata-only dry run that allocates nothing. Batty should extract a non-allocating shared attribute/config validator used by both `create` and `unsupported_reason`, leaving `Box::new` only in actual kernel construction, then add a focused allocation-free claim regression if the test harness supports it.

<!-- merged from deckard-bqmoe-claimfix.md -->

2026-07-19 — Unified BlockQuantizedMoE claim-shape validation in `validate_partial_claim_shapes`: every independently static axis is now checked for packed FC1/FC2/FC3 tensors, optional biases, router weights, and flattened router rows; the former fully-static-only second pass was removed. Added dense-reference tests for discriminating top-k expert selection, ReLU/GELU, SwiGLU alpha/beta/limit attributes, and symbolic claim rejection for invalid FC1 bias, FC3, and router-weight axes.

<!-- merged from batty-bqmoe-zeroalloc.md -->

### 2026-07-19: BQMoE claim zero-alloc fix
**By:** Batty
**What:** Routed BQMoE claim-time attribute, dtype, and symbolic-shape checks through a metadata validator that returns validated stack-owned configuration without constructing a kernel; factory creation reuses it before allocating the kernel box.
**Why:** The hardened claim contract requires successful support checks to remain metadata-only and allocation-free while preserving exact agreement with construction and execution validation.

<!-- merged from chew-bqmoe-rereview3.md -->

### 2026-07-19: BQMoE v1 zero-alloc re-review
**By:** Chew
**What:** 🟢 APPROVE commit `67abdb5`. `unsupported_reason` now invokes metadata-only `validate_metadata(..., Some(...))`; its successful claim path parses stack-only `MoeAttributes`/`BlockFormat` and checks shape/dtype metadata without constructing a `BlockQuantizedMoEKernel`, `Box`, or weight buffer. `Factory::create` invokes the same validation with no claim metadata and then performs the sole `Box::new(BlockQuantizedMoEKernel { ... })` construction.
**Why:** The prior successful-claim allocation is removed while ABI, symbolic-dimension deferral, static mismatch rejection, and `Undefined` omitted-option handling remain intact. Discriminating top-k and activation reference tests are present. `cargo build -p onnx-runtime-ep-cpu` and `cargo test -p onnx-runtime-ep-cpu` passed (520 passed, 1 ignored).

<!-- merged from leon-pr30-device-sampler.md -->

# PR #30 device sampler — fix note (Leon, Engine Dev / KV & Buffers)

Branch: `perf/cuda-on-device-argmax` (worktree `pr30-fix`), pushed commit **9b062f9**.
File: `crates/onnx-genai-ort/src/device_sampler.rs` (+ 1 call-site line in `decode.rs`).

## Host sampling pipeline — exact semantics mirrored

Source of truth: `onnx-genai-engine` `build_processor_chain` (order) +
`logits.rs` processors + `sampling.rs::sample_categorical`.

Order (each stage operates on the running logit array and masks pruned entries
to `-inf`; every stage AFTER temperature recomputes softmax over the CURRENT,
already-masked logits, i.e. renormalizes over survivors):

1. **Temperature** — `logit /= temperature` when `is_finite && >0 && !=1`.
2. **TopK** — sort non-NaN logits desc, `threshold = sorted[k-1]` (ties at the
   threshold all kept → count-with-multiplicity, not distinct rank); mask
   `logit < threshold` to `-inf`. Applied only when `top_k>0 && top_k<len`.
3. **TopP** — softmax over current logits (already restricted to the top-k
   survivors → renormalized), sort probs desc, keep the smallest prefix whose
   cumulative mass `>= top_p`, mask the rest. Applied when `top_p < 1.0`.
4. **MinP** — softmax over current logits, `top_prob = 1/exp_sum`, mask
   `prob < min(min_p,1)*top_prob`. Applied when `min_p > 0`.
5. **Final draw** — `sample_categorical`: fresh softmax over survivors, walk in
   index order, return first token with running `cumulative > rng`. A non-finite
   max (e.g. `+inf`) falls back to greedy = lowest-index max. Greedy path uses
   argmax directly (lowest-index max, NaN ignored).

The key parity insight: top-p/min-p must be computed on the **post-top-k
renormalized** distribution, NOT as independent thresholds over the full vocab.

## The four fixes

1. **(HIGH, parity) Sequential filters.** Rewrote the `finish_row` CUDA kernel to
   apply top-k → top-p(renorm) → min-p as sequential `-inf` masks, then a fresh
   softmax inverse-CDF draw — exactly the host order. Previously it computed
   three independent thresholds over the full-vocab softmax and combined them
   with `max`, which kept a different nucleus. Reviewer counterexample
   `[.505,.061,.040,10×.039]`, `top_k=3, top_p=0.9`: host keeps `{0,1}`, old
   device kept `{0,1,2}`; fixed device now keeps `{0,1}`.
   `device_sampler.rs:294-457` (finish_row).

2. **(HIGH, correctness) `+inf` logits.** `expf(+inf - +inf) = NaN` poisoned the
   softmax and forced token 0. Added an explicit `m == +inf` branch that does a
   block-wide min-index reduction over the `+inf` entries and returns the
   lowest-index `+inf` token — matching the host greedy fallback.
   `device_sampler.rs:315-333`.

3. **(HIGH, memory safety) Scratch growth.** `OutScratch::ensure` /
   `WorkScratch::ensure` freed the old pointer BEFORE the fallible `malloc`, so an
   alloc failure left a dangling `self.ptr` (double-free on retry/Drop). Reordered
   to **allocate-new-first, then free-old-and-swap only after malloc succeeds**;
   on failure the existing buffer is untouched and the error propagates, so `Drop`
   can never double-free. `device_sampler.rs:~955-1005`.

4. **(MED, hot-path perf) Per-decode allocations.** Added an allocation-free
   single-row path: `argmax_into`/`sample_into` fill a caller slice, and a new
   `sample_one` (trait method) reads the winner into a stack `[i32;1]` and returns
   a scalar `u32` — removing the per-token `i32` vec + `u32` vec on the captured
   decode path. `ctx_sync_enabled()` now caches the env read in a `OnceLock`
   (no per-token `String` alloc). `decode.rs` captured path calls `sample_one`.

## Tests (all pass; GPU present, so GPU tests really executed)

`cargo test -p onnx-genai-ort --features cuda --lib` → 24 passed, 0 failed.
Host-only `--lib` → 15 passed. New tests:

- `device_algo_matches_host_oracle_cpu_sweep` — CPU port of the device kernel vs
  a faithful port of the host processors + `sample_categorical`; identical tokens
  over a grid of (temperature, top_k, top_p, min_p) × 7 seeds on 5 well-separated
  distributions (3220 combos).
- `counterexample_keeps_nucleus_zero_one` / `counterexample_matches_host_on_gpu`
  — Gaff's case; token 2 never selectable, device==host for every seed (CPU+GPU).
- `plus_inf_selects_lowest_index_cpu` / `_gpu` — single and multiple `+inf`.
- Updated `categorical_matches_host_oracle_f32` and the multi-row test to assert
  against the new faithful `host_oracle`.

### Note on exact ties
Token-for-token host parity is only well-defined for distinct distributions: the
host TopP uses `sort_unstable` + keep-count, which breaks EXACT probability ties
non-reproducibly. The device is deterministic (threshold keeps all tied tokens).
The parity sweep therefore uses well-separated rows; the reviewer counterexample's
nucleus boundary is distinct (`.040` vs `.039`) and passes exactly.

Skipped: none required — a GPU was available. (Any `conv_gpu`/cuDNN-missing
failures elsewhere are unrelated and not touched here.)

<!-- merged from gaff-pr30-rereview3.md -->

### 2026-07-19: PR #30 review cycle 3
**By:** Gaff
**What:** 🔴 REJECT. Batty correctly prevents post-run extraction/sampling failures from falling back to `step_standard`, and the `ONNX_GENAI_DEVICE_ARGMAX` lookup is correctly cached in a `OnceLock`. Two blockers remain: (1) `decode.rs:751-753` classifies every `run_with_binding_graph` error as `RunInvoked`, but `session.rs:355-381` contains fallible API lookup, run-options creation, and config insertion before the actual `RunWithBinding` call at `session.rs:383-385`; those pre-run failures are therefore propagated instead of safely retrying through the standard path. (2) `decode.rs:114-126` is tautological: it initializes the run count to one, manually constructs `RunInvoked`, and tests only the retry helper, never invoking `step_dispatch`, a model runner, or a failing sampler.
**Why:** The requested phase split is incomplete for failures inside the session wrapper, and the regression test cannot catch either a future call-site misclassification or an accidental model replay. Roy should revise this artifact (Batty and Leon are locked out): expose phase-aware graph-run errors or split setup from invocation so only errors at/after the actual ORT run are `RunInvoked`, then add a non-tautological injected runner/sampler test proving a post-run sampling failure produces exactly one model invocation and no standard fallback. CUDA build and full CUDA-feature test suite passed; the named new test also passed, but only exercises the helper.

<!-- merged from batty-pr30-decode-retry.md -->

### 2026-07-19: Captured decode retry safety
**By:** Batty
**What:** Captured-step failures are classified at the ORT graph-run boundary. Setup/binding failures before the run are retryable through `step_standard`; any run invocation, output extraction, or sampling failure propagates without replaying the model. `ONNX_GENAI_DEVICE_ARGMAX` is resolved once through `OnceLock`.
**Why:** A completed or potentially partially executed graph run may already have mutated shared KV buffers, so retrying would double-advance decode state. Caching the environment flag removes an environment lookup from every generated token.

<!-- merged from roy-pr30-errclass-fix.md -->

### 2026-07-19: PR #30 pre/post-run error classification fix
**By:** Roy
**What:** Split the CUDA captured-decode run entrypoint into phases. Added `Session::run_with_binding_graph_phased` returning a discriminated `RunPhaseError` (`Setup` vs `Invoked`): everything before the ORT `Run` call (API lookup, run-option creation, `gpu_graph_id` config entry) is `Setup`; only the `Run` call itself is `Invoked`. `run_with_binding_graph` now delegates and flattens via `RunPhaseError::into_inner`. In `decode.rs`, the captured-step call site maps `Setup -> CapturedStepError::PreRun` (retryable via `step_standard`) and `Invoked -> RunInvoked` (propagate, no replay) through a new `classify_run_phase` helper. Replaced the tautological `post_run_sampling_failure_does_not_rerun_model` test with a `FakeRunner` harness that mirrors the setup->invoke->sample phase ordering and asserts exact model-invocation counts: post-run sampler failure and run-call failure each invoke the model exactly once with NO `step_standard` fallback; a pre-run setup failure retries and invokes the model exactly once via the standard step. Added a mapping test for `classify_run_phase`.
**Why:** Gaff (cycle 3) found that setup/binding failures originating inside `run_with_binding_graph` were mislabeled `RunInvoked` (propagated) when they are genuinely PRE-run and safe to retry, and that the existing no-rerun test proved nothing because it never ran a model or sampler. The KV cache double-advance invariant requires structural (not heuristic) knowledge of whether the model was invoked, so the run helper now reports that fact directly. Verified with `cargo build`/`cargo test -p onnx-genai-ort --features cuda` on a real GPU; all 4 new unit tests pass and the suite is green.

<!-- merged from gaff-pr30-rereview4.md -->

### 2026-07-19: PR #30 review cycle 4
**By:** Gaff
**What:** 🟢 APPROVE commit `b99d4ca`; the coordinator may run `gh pr merge 30 --rebase`.
**Why:** `run_with_binding_graph_phased` classifies every failure before `RunWithBinding` as `Setup` and the run call itself as `Invoked`; the captured decode path structurally maps those to `PreRun` and `RunInvoked`, and only `PreRun` reaches `step_standard`. The replacement `FakeRunner` tests count actual simulated model invocations and exercise post-run propagation plus the pre-run retry closure. The phased wrapper preserves the tag for the captured caller while the legacy wrapper intentionally flattens it for unchanged callers. The CUDA build, full CUDA-feature crate test suite, and all four targeted retry tests passed. No regression to the earlier sampler or no-double-run fixes was found.

<!-- merged from deckard-pr30-rebase.md -->

### 2026-07-19: PR #30 rebase onto main
**By:** Deckard
**What:** Resolved conflicts in `crates/onnx-genai-ort/Cargo.toml` (twice), `crates/onnx-genai-ort/src/lib.rs` (twice), and `crates/onnx-genai-bench/src/bin/profile_decode.rs`; preserved main's CUDA runtime/shared-KV and chat-template profiling support alongside PR #30's device sampler, configurable CUDA API features, and sampling profiler options.
**Why:** Rebase `perf/cuda-on-device-argmax` cleanly onto main while retaining both independently reviewed feature sets.

<!-- merged from gaff-pr30-rebase-verify.md -->

### 2026-07-19: PR #30 rebase integration verified
**By:** Gaff
**What:** 🟢 APPROVE PR #30 at `87baba8`; the coordinator may merge it with `gh pr merge 30 --rebase`.
**Why:** The branch is FF-ready on `origin/main` (`67abdb5`), has no conflict markers, and preserves both CUDA integrations: `cuda_rt.rs`/`libloading` for shared-KV growth and the previously approved on-device sampler in `device_sampler.rs`/`cudarc`. (`cuda_argmax.rs` was not present at approved tip `b99d4ca`; `device_sampler.rs` is the actual module name.) Cargo enables both dependencies, both modules are CUDA-gated and wired from `decode.rs`, profile-decode retains main's chat-template path plus PR sampling options, and Roy's phased-error symbols/tests remain. CUDA build and full crate tests pass; targeted sampler and phased-error tests also pass. GPU sampler cases self-skipped because this host's CUDA 13.3 NVRTC emits PTX unsupported by driver 580, an environment mismatch rather than a rebase regression.

<!-- merged from batty-pr34-controlflow.md -->

# PR #34 control-flow capture safety fix

CUDA graph capture is now conservatively disabled whenever control-flow detection cannot cheaply inspect a model, including models larger than 512 MiB, unreadable models, and unparseable models. The invariant is: when in doubt, disable capture; capture is optional and must not risk ORT's control-flow slow path.

Detection now records each ONNX node domain and treats `If`, `Loop`, and `Scan` as control flow only in the standard ONNX domains (`""` or `"ai.onnx"`). Custom-domain operators with those names do not disable CUDA graph capture.

## 2026-07-19 — PR #32, IndexShare v1, and CSA B0 landing wave (07:55Z)

<!-- merged from leon-pr32-rebase-fix.md -->

### 2026-07-19: PR #32 rebase + build/review fixes
**By:** Leon
**What:** Rebased the EP-capabilities refactor onto `main`, retained the newer device-sampler, CUDA KV-copy, phased-run, BQMoE, multi-EP, and generic plugin behavior, and fixed the three Copilot review findings.
**Why:** PR #32 was stacked on pre-#30 history; integrating both lines preserves EP-agnostic capability decisions without regressing current runtime, CUDA, plugin, or failure-classification behavior.

<!-- merged from luv-pr32-review.md -->

### 2026-07-19: PR #32 review
**By:** Luv
**What:** Approved PR #32 at `473f32f` after verifying the post-rebase ancestry, capability-based EP selection, half-precision argmax edge cases, opaque NamedGeneric options forwarding, retained CUDA/device-sampler/cuda_rt and BlockQuantizedMoE integration, and green CI.
**Why:** `argmax_half_bits` now has the same NaN, all-`-inf`, tie, and first-element behavior as the f32 reference, including tested `[NaN, -inf] -> 1`; `EpSelection::new` shares `normalize_ep_name` with environment parsing; and NamedGeneric options reach the ORT append call. Workspace and CUDA builds passed; library tests passed 39/39 and 48/48. The broader ORT integration test command could not run fixture-dependent decode tests because `tests/fixtures/tiny-llm/model.onnx` is absent locally, while PR CI is green.

<!-- merged from batty-indexshare-v1.md -->

### 2026-07-19: IndexShare v1 CPU kernel + frozen ABI
**By:** Batty
**What:** Froze `pkg.nxrt::IndexShare` v1 as exporter-selected, deterministic f32 selected-token attention with explicit past/present KV, additive bias, strict dense-order indices, GQA/shared indices, and a CPU reference kernel.
**Why:** The ratified D1-D4 defaults require a stable private-op boundary and an exact dense additive-mask oracle before Mobius emission and production EP implementations can replace the full-cache fallback.

<!-- merged from chew-indexshare-review.md -->

### 2026-07-19: IndexShare v1 numerical review
**By:** Chew
**What:** 🟡 APPROVE-WITH-NITS for `feat-indexshare-v1` at `b61fb81`. The CPU kernel validates ordered selected indices at execution, implements GQA and explicit past/present KV I/O, and exactly matches an independent full-cache dense additive-mask oracle in its selected-subset, GQA/shared-index, and causal/padding-bias tests.
**Why:** The oracle independently builds a full `-inf`-masked score vector and performs softmax/value reduction in ascending dense-cache order; parity assertions use `assert_eq!`. Claim metadata validation has no tensor reads or allocations on the supported path and accepts Undefined optionals while rejecting present wrong dtypes. The sole coverage nit is that `rejects_invalid_index_rows_at_execution` omits the required `[-1, valid, ...]` non-trailing-sentinel case. Build and test pass (525 passed, 1 ignored); IndexShare has no Clippy diagnostics, although the crate retains 19 pre-existing Clippy warnings elsewhere.

<!-- merged from sapper-csa-b0.md -->

### 2026-07-19: CSA Phase B B0 scaffolding
**By:** Sapper
**What:** Added fixed-capacity CUDA CSA buffer reservation, all-Host per-stage dispatch and golden-capture seams, plus shared NVRTC block quant/dequant scaffolding with CPU-dequant round-trip tests.
**Why:** Establish stable device-state and stage-parity seams without changing Phase A host-oracle numerics; D2/D3 are implemented as the official defaults.

<!-- merged from chew-csa-b0-review.md -->

### 2026-07-19: CSA Phase B B0 review
**By:** Chew
**What:** Rejected B0 (`fad07aa`) pending a replacement quant round-trip test and completion of the shared NVRTC quantization scaffold. `block_quant.rs:157-177` only asserts that CPU-dequantized data is finite and checks one scale exponent. It does not compare reconstructed values or packed codes to independently derived expected results, so an incorrect E4M3/E2M1 rounding implementation still passes. Additionally, `BLOCK_QUANT_CUH` at `block_quant.rs:41-53` contains only FP4 code selection; it explicitly defers FP8 E4M3FN encoding, scale derivation, clamping, and subnormal handling to B2, contrary to B0’s required shared FP8/FP4 quantizer scaffold.
**Why:** B0’s numeric safety gate requires tests that fail on incorrect scale or rounding, and future device stages need the complete common NVRTC quant/dequant contract now. The all-Host path remains the CPU oracle, original CSA GPU test file is byte-unchanged, `cuda_graph_compatible()` remains false, no CSA dependency on `block_quantized_matmul` was found, and build/full CUDA EP tests passed (including the unchanged 11 CSA GPU tests). The test scaffolding nevertheless cannot establish the stated quantization correctness requirement.

<!-- merged from deckard-csa-b0-fix.md -->

### 2026-07-19: CSA B0 FP8 quant + real round-trip tests
**By:** Deckard
**What:** Added shared NVRTC FP8 E4M3 block-64 quantization with E8M0 power-of-two scaling, round-to-nearest-even encoding, saturation, and subnormal handling. Replaced tautological FP8/FP4 tests with hand-computed packed-code and CPU-dequantized-value assertions.
**Why:** B0 needs a self-contained shared quant/dequant scaffold with a numerically meaningful oracle gate before later CSA device stages consume it.

<!-- merged from chew-csa-b0-rereview.md -->

### 2026-07-19: CSA B0 re-review
**By:** Chew
**What:** 🟢 APPROVE commit `5c308f7`. FP8 E4M3 and FP4 E2M1 quantization use the required scales, round-to-nearest-even behavior, saturation, and subnormal handling; independent hand-computed tests cover scale selection, ties, saturation, packed codes, and reconstructed values.
**Why:** CUDA build and full EP tests passed, including 8 block-quant tests and all 11 unchanged CSA GPU tests. CSA remains graph-incompatible, and the coordinator may merge B0.
