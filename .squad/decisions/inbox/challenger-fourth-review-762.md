# Challenger Fourth Review — PR #762

**Date:** 2026-08-11  
**Reviewer:** Challenger (first review of this PR)  
**Commit range:** `fbd565160`..`4757e25b6`

---

## BLOCKING

### B1. String sentinel is forgeable — in-band signalling with no namespace isolation
**File:** `crates/onnx-runtime-ep-plugin/src/graph_reader.rs:187`, `ep.rs:283`, `ep.rs:401`  
**Problem:** `__absent_output_{i}_{slot}` is a plain string prefix-matched with `starts_with("__absent_output_")`. Any ONNX model that names a tensor `__absent_output_0_2` will be misidentified as an absent slot, causing the EP to skip dtype validation and allocate a scratch buffer instead of a real output. This is not merely adversarial — model export tools can produce arbitrary internal names. The sentinel is constructed at exactly one site (`graph_reader.rs:187`) but matched at two (`ep.rs:283`, `ep.rs:401`), and the convention will inevitably leak to tests and docs.

**Verdict on sentinel:** **Replace before merge.** The correct representation is `Option<ValueId>` in the node's output list (matching the input list which already uses `Option<ValueIndex>`). The IR already distinguishes present from absent on the *input* side (`node_inputs` returns `&[Option<ValueIndex>]`); outputs should do the same. The string sentinel is a cheap workaround that converts a type-safe absence into a stringly-typed one.

**Minimum acceptable alternative if refactoring `outputs: Vec<ValueId>` to `Vec<Option<ValueId>>` is too invasive for this PR:** add a `HashSet<ValueId>` of absent output IDs to the `IrGraph` and check membership there — no in-band name needed. The sentinel name then becomes a debug-only label with no semantic load.

**Owner:** Mariette (but locked out — needs Sebastian or Leon)

---

### B2. `filter_map` on shape dims discards dynamic dimensions silently
**File:** `ep.rs:493`, `ep.rs:238`  
```rust
.filter_map(|d| d.as_static())
```
This compacts the shape: a `[batch, 4, seq]` with `batch` and `seq` dynamic becomes `[4]` — a rank-1 shape where rank was 3. Any kernel that indexes by position (every matmul, every layer-norm axis computation) will misinterpret the result. The code only uses this for `get_kernel` shape hints and compile-time shape inference, but `ShapeInference::for_node` reads `shapes[0].len()` as rank. If any dynamic dim precedes a static dim, the rank is wrong.

**Concrete scenario:** `LayerNorm(input=[batch,seq,768], axis=-1)`. Reported shape: `[768]` (rank 1). `normalise_axis(-1, 1) = 0`. Kernel reduces dim 0 of a rank-3 tensor — wrong axis.

**This is the same class of bug as the original `filter_map` compaction bug on outputs.** Position is load-bearing; compaction destroys it.

**Fix:** Replace `filter_map(|d| d.as_static())` with `map(|d| d.as_static().unwrap_or(0))` or an explicit `Vec<Option<usize>>` so rank is preserved. Kernels must distinguish "unknown extent" from "extent removed."

**Owner:** Sebastian

---

## SUBSTANTIVE

### S1. `conformance_mixed_partition` passes `disable_fallback: false`
**File:** `plugin_ort_e2e.rs:1146`  
The exemption is legitimate — the model contains `NonZero` which is deliberately unsupported, so fallback is required. However, the test **does not verify which nodes our EP claimed vs which fell back**. It only checks final output correctness. If our EP declines *all* nodes (e.g., due to a bug in the claim filter), the test still passes — ORT's CPU EP handles the full graph. This is exactly the "tests pass via fallback" failure mode.

**Fix:** After session creation, query the EP assignment (ORT's `GetSessionProfilingInfo` or a partitioning log) to assert our EP claimed at least one subgraph (the Add node). Alternatively, inject a tracing hook in `ep_get_capability` that records claimed node count and assert > 0.

**Owner:** Freysa

### S2. Scratch buffer fallback of 8 bytes — fail-quiet, not fail-closed
**File:** `compute.rs:932-940`  
When `output_dtypes[0]` is also `Undefined` (all outputs absent — unlikely but representable), `unwrap_or(8)` allocates 8 bytes per element. If shape inference returns a non-trivial shape, this silently produces a large allocation that is never read by ORT — hiding the bug. The design principle for this PR is "fail closed when information is absent." A `return fail_status(...)` would be consistent.

**However:** In practice, slot 0 is always the primary output and always has a known dtype (enforced by the claim filter at `ep.rs:279` which only skips Undefined for *absent* slots). So this path is dead. **Acceptable as defensive code** provided it is annotated as unreachable.

**Owner:** Mariette

### S3. `input_slots` correctness for all-present case
**File:** `ep.rs:524-537`  
When all inputs are present, `input_slots` = `[Some(0), Some(1), ..., Some(n-1)]` — identity mapping. The fast path then does `inputs[*ort_idx].view()` which is correct. For interior absent inputs (e.g., `Clip(input, _, max)`), ORT does not deliver absent inputs in the `inputs` array at all — they are simply not in the `KernelContext` input list. The code correctly skips them (`None` → `TensorView::absent()`), and `ort_input_idx` only increments for `Some`. **Verified correct for trailing, interior, and all-present cases.** No off-by-one.

### S4. Remaining `filter_map` / `flatten` uses
**File:** `factory.rs:99` — `.flatten()` on an `Option<Vec<...>>` — safe, not positional.  
**File:** `compute.rs:1644` — `filter_map` in `ReduceMean` shape inference with `keepdims=false` — **this is correct**, it intentionally removes reduced dimensions from the output shape per ONNX spec.

---

## NITS

### N1. Sentinel string not behind a constant
**File:** `graph_reader.rs:187`, `ep.rs:283`, `ep.rs:401`  
The prefix `"__absent_output_"` appears as a raw literal in three places. Extract to `const ABSENT_OUTPUT_PREFIX: &str`.

### N2. `c_char` signedness on aarch64
**File:** `graph_reader.rs:396,412,440,459,567,778`  
All uses are `*const c_char` returned by ORT's C API and consumed via `CStr::from_ptr` — correct regardless of signedness. No issue found.

---

## VERIFICATION SUMMARY

| What | Verified by me | Method |
|------|---------------|--------|
| `disable_cpu_ep_fallback=1` in `optional_slots.rs` | ✅ | Read `optional_slots.rs:161-170` — key set, status checked |
| `disable_cpu_ep_fallback=1` in `plugin_ort_e2e.rs` (all except mixed_partition) | ✅ | Read `conformance_setup` signature + all 16 call sites: only `conformance_mixed_partition` passes `false` |
| `conformance_mixed_partition` exemption legitimacy | ⚠️ Partial | Contains `NonZero` (unsupported), so fallback needed — but no assertion that our EP claimed *anything* |
| `input_slots` off-by-one | ✅ | Traced logic for interior-absent, trailing-absent, all-present |
| Panics across FFI | ✅ | All 8 `extern "C"` fn files wrap in `catch_unwind` (55 occurrences) |
| Cross-module allocation | Not verified | Did not trace allocator boundaries — taken on trust from prior reviews |
| CUDA honesty | ✅ | No hardware on this host; no mock pretending to be hardware; #768 tracks |
| `filter_map` positional bugs | **FOUND B2** | `ep.rs:493` compacts dynamic dims, losing rank |

**Is the CPU EP genuinely proven end-to-end with fallback disabled?** **YES** for the 21 conformance tests in `plugin_ort_e2e.rs` (all pass `disable_fallback: true`). **YES** for the 2 optional-slot tests. **PARTIAL** for `conformance_mixed_partition` (fallback enabled, no EP-assignment assertion).

---

## SHOULD #762 LEAVE DRAFT?

**No.** Two blockers:

1. **B1 (sentinel):** Must be replaced with an out-of-band representation, or at minimum a `HashSet<ValueId>` on `IrGraph`. The current approach is a name-collision vulnerability.
2. **B2 (`filter_map` shape compaction):** Positional rank is destroyed for shapes with dynamic dims. This is the same class of bug that motivated this PR's existence.

**Shortest path to yes:**
- B1: Add `absent_outputs: HashSet<ValueId>` to `IrGraph`; check membership instead of name prefix. (~30 min)
- B2: Replace `filter_map(|d| d.as_static())` with `map(|d| d.as_static().unwrap_or(1))` (using 1 as "unknown extent, rank preserved") in `ep.rs:493` and `ep.rs:238`. Add a test with a dynamic-batch model. (~1 hour)
- S1: Add EP-assignment assertion to `conformance_mixed_partition`. (~15 min)

After those three, this is mergeable.

---

## WHAT I VERIFIED MYSELF vs TOOK ON TRUST

**Verified (read code, traced logic):**
- Sentinel construction/match sites (exactly 1 construction, 2 match)
- `disable_cpu_ep_fallback` presence in all test paths
- `input_slots` mapping correctness
- `catch_unwind` coverage of all `extern "C"` functions
- `filter_map` / `flatten` usage audit (found B2)
- Scratch buffer fallback reachability analysis
- `c_char` / aarch64 safety

**Taken on trust (from prior reviews):**
- Cross-module allocation boundaries (Luv R2 verified)
- ORT 1.27 binary compatibility (Holden R1 verified against prebuilt)
- 266-pass / 0-fail EP crate test count (established fact from prior run)
- Base branch pre-existing failures (20 failures documented)
