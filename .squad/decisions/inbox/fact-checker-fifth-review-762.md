# Fact Checker — Fifth Review of PR #762

**Date:** 2026-08-11
**Commit:** `38625fb38`
**Reviewer:** Fact Checker (Verification + Devil's Advocate)

---

## Claim Verdicts

### 1. "Unforgeable" — absent_outputs via ValueId ✅ Verified

**Evidence:** `graph_reader.rs:158-200` — `absent_outputs` is populated only when
`node_output_names[i]` contains an empty string (the ONNX representation of an absent
optional output). The ValueId is created by `ir_graph.create_named_value(format!("_absent_{i}_{slot}"), ...)`.
Model content cannot cause an empty output *name* — ORT's graph API reports empty names
only for genuinely absent optional outputs. Attacker-controlled tensor names go through
the non-empty path and land in `value_map`, never in `absent_outputs`.

The forgery-rejection test at `ep.rs:1438` proves that a value named `"__absent_output_0_1"`
is NOT exempt unless its ValueId is explicitly in the HashSet. The old string path is fully removed.

### 2. "Rank is preserved" ⚠️ Unverified (acceptable risk)

**Evidence:** Both sites (`ep.rs:257`, `ep.rs:505`) now use `map(|d| d.as_static())` producing
`Vec<Option<usize>>` — vector length equals rank. ✅ Correct.

**Concern:** At `ep.rs:517`, `unwrap_or(0)` converts `Option<usize>` → `usize` for the
`get_kernel` trait interface. A dim of `0` is passed to kernels. This is documented:
> "substitute 0 for unknown dims. This preserves rank while signalling 'dynamic' — valid
> static dims are always ≥1 for non-empty tensors, and the kernel receives actual shapes
> from OrtKernelContext at runtime."

The **shape inference** step (prior to compile) operates on `Vec<Option<usize>>` and fails
closed (Conv example verified: `conv_declines_with_symbolic_spatial_dims` test passes).
At compile-time the `0` dims are used only for *buffer pre-allocation hints*, not loop bounds
or strides — actual shapes come from `OrtKernelContext` at runtime.

**Residual risk:** If any kernel uses compile-time shapes for *allocation* and allocates
a zero-byte buffer, a runtime write would overflow. Not observed in current tests, but
worth an assertion `debug_assert!(dim > 0)` in allocation paths.

### 3. "Fails closed" ✅ Verified

**Evidence:** `build_conv` returns `None` when spatial dims are symbolic (test:
`conv_declines_with_symbolic_spatial_dims`). The `for_node` method uses `?` propagation
extensively. `LayerNormalization` etc. are `Declined` from `for_op` and resolved only
when attributes are available in `for_node`. The claim filter at `ep.rs:265` skips any
node where `ShapeInference::Declined` is returned.

Shape-dependent ops (Conv, Resize, GridSample in cuda-plugin) all return `None` or
`Declined` when required dimensions are unknown.

### 4. "Absent optional outputs claimed with fallback disabled" ✅ Verified

**Evidence:** I ran both test suites:
- `optional_slots` (4/4 pass) — sets `disable_cpu_ep_fallback=1` at line 161
- `plugin_ort_e2e` (23/23 pass) — `conformance_setup` takes `disable_fallback: bool`;
  all tests except `conformance_mixed_partition` pass `true`

The `clip_omitted_min_with_max` and `skip_layer_norm_output_sum_position` tests that
were vacuous in round 3 now pass with fallback disabled — our EP claims and executes
these nodes.

### 5. "ORT 1.27 has no per-node provider attribution" ❌ Contradicted

**Evidence:** ORT 1.27 prebuilt header `onnxruntime_c_api.h:7246` contains:
```c
/** \brief Get information about the subgraphs assigned to each execution provider...
 * \since Version 1.24.
 */
ORT_API2_STATUS(Session_GetEpGraphAssignmentInfo, ...)
```

Additionally, `Node_GetEpName` exists since ORT 1.23 (line 6556). The claim in
`plugin_ort_e2e.rs:1155` that "ORT 1.27 has no per-node provider attribution query"
is **false**. `Session_GetEpGraphAssignmentInfo` (available since 1.24, with config
key `session.record_ep_graph_assignment_info`) provides exactly this capability.

The `conformance_mixed_partition` test should use this API rather than the workaround
`nxrt_ep_compiled_node_count` symbol.

### 6. CUDA honesty ✅ Verified (honest)

**Evidence:** `docs/CUDA_EP_STATUS.md` is exemplary in honesty:
- Title: "Compiles, Unvalidated on Hardware"
- Every row says "✅ Fixed (unvalidated on hardware)"
- §4 "What Remains Unknowable Without a GPU" lists 5 concrete unknowns
- §6 "scripts/cuda_conformance_runner.sh exits 2 (UNVALIDATED)"
- Issue #768 is cited for hardware validation

No overstatement found. Nothing implies hardware validation or presents mocks as evidence.

### 7. Remaining compaction/default patterns ⚠️ One minor residual

**Residual:** `compute.rs:1650` uses `filter_map` in ReduceMean/Sum shape inference.
This is **correct** — when `keepdims=false`, dimensions are intentionally removed per
ONNX spec. NOT a bug.

**`unwrap_or(0)` audit:** Most `unwrap_or(0)` in `compute.rs` are for ONNX attributes
with default values (e.g., `transA=0`, `axis=0`, `batch_dims=0`, `allowzero=0`). These
are correct per ONNX operator specifications. The `graph_reader.rs:287` `unwrap_or(0)`
is for index lookup (non-critical path).

No remaining instances of the two recurring patterns (compaction losing rank; defaults
substituted for absent info) in shape-critical paths.

---

## BLOCKING / SUBSTANTIVE / NITS

### BLOCKING

**(none)**

### SUBSTANTIVE

| # | File:Line | Issue | Owner |
|---|-----------|-------|-------|
| S1 | `plugin_ort_e2e.rs:1155` | Claim "ORT 1.27 has no per-node provider attribution" is false. `Session_GetEpGraphAssignmentInfo` exists since 1.24. Should use it for `conformance_mixed_partition` or correct the comment. | Freysa (test infra) |

### NITS

| # | File:Line | Issue | Owner |
|---|-----------|-------|-------|
| N1 | `ep.rs:517` | Consider `debug_assert!(dim > 0)` in kernel allocation paths to catch zero-dim at dev time | Coco |

---

## Is the CPU EP genuinely proven end-to-end with fallback disabled?

**YES.** Verified by running:
- `cargo test -p onnx-runtime-ep-cpu-plugin --test plugin_ort_e2e --no-fail-fast` → 23/23 pass
- `cargo test -p onnx-runtime-ep-cpu-plugin --test optional_slots --no-fail-fast` → 4/4 pass

Both set `disable_cpu_ep_fallback=1`. Tests load the cdylib, register with real ORT 1.27,
create sessions, and run inference. This is not mock-based.

---

## False or Overstated Claims

| Location | Claim | Verdict |
|----------|-------|---------|
| `plugin_ort_e2e.rs:1155` | "ORT 1.27 has no per-node provider attribution query" | **False.** `Session_GetEpGraphAssignmentInfo` since 1.24; `Node_GetEpName` since 1.23. |

---

## Should #762 leave draft?

**Not yet.** The shortest path to yes:
1. Fix S1: Either use `Session_GetEpGraphAssignmentInfo` to add a hard EP-assignment
   assertion in `conformance_mixed_partition`, OR correct the comment to accurately state
   why it isn't used (e.g., "requires `session.record_ep_graph_assignment_info` config key
   which we haven't wired up yet" — a deferral, not a claim of impossibility). ~1 hour.

After S1, there are no blocking or substantive issues remaining in the EP crates.

---

## Devil's Advocate Brief

**Strongest argument against merging as-is:**

The `unwrap_or(0)` at compile-time (`ep.rs:517`) is a load-bearing convention: it assumes
every kernel implementation treats `0` dims as "will be resolved at runtime from
OrtKernelContext" and never uses them for allocation sizes, loop bounds, or stride
calculations at compile time. This convention is documented in a *comment* but not
enforced by the type system or a runtime check.

**Load-bearing assumption:** No kernel ever pre-allocates based on compile-time shapes.

**30-day failure scenario:** A new contributor adds a kernel (e.g., ConvTranspose) that
pre-computes output buffer size from the shapes passed to `get_kernel`. For a model with
a dynamic batch dim, `batch=0` yields a zero-byte allocation. At runtime, ORT provides
the real shape and writes to the zero-byte buffer → heap corruption, silent data corruption,
or segfault in production. The CI passes because existing test models have static shapes.

**Mitigation:** Add `debug_assert!(d != 0, "compile-time dim must not be used as allocation size")`
at the entry of any kernel that uses shapes from `get_kernel` for memory allocation.
Alternatively, make the compile-time shapes `Vec<Option<usize>>` all the way into the
kernel trait so the type system prevents accidental use.

---

## What I Verified vs. Took on Trust

| Verified myself | Took on trust |
|-----------------|---------------|
| `absent_outputs` populated only from empty output names (read code) | ORT only reports empty names for genuinely absent outputs |
| `filter_map` removed from shape paths (grep, no hits) | nxrt-abi round-trip tests (did not re-run; established fact: 32/32) |
| `disable_cpu_ep_fallback=1` set in both test files (read code) | ORT respects this config key (documented ORT behaviour) |
| All 27 EP tests pass (ran them) | CUDA code correctness on hardware (explicitly unvalidated) |
| `Session_GetEpGraphAssignmentInfo` exists in ORT 1.27 headers | That the Rust bindings expose it (did not check `ort-sys` bindings.rs) |
| `build_conv` returns None for symbolic dims (test exists) | All other shape-dependent ops decline similarly (spot-checked, not exhaustive) |
| No `filter_map(as_static)` remaining (grep clean) | `unwrap_or(0)` safety at runtime (relies on OrtKernelContext providing real shapes) |
