# pris — History

## Summary through 2026-07-28T21:15:00+0000
- Earlier detailed history was compacted by Scribe after exceeding the 15KB history gate. Durable team-wide decisions remain in `.squad/decisions.md` and `.squad/decisions-archive/`.
- Pris's sustained domains: test infrastructure, fixture quality, coverage hardening, metadata/schema validation, CPU/CUDA dispatch correctness, and reviewer-driven fix cycles.
- Delivered/maintained tiny-LLM and Mobius fixtures; KV/session and CPU-EP op coverage; macOS benchmark harness guards; CLI ORT CI; and dispatch reachability auditing.

## Retained durable outcomes
- **Dispatch correctness (2026-07-27):** Added reachability tests and `_TEST_HITS` counters for matmul/SDPA dispatch, with guard-break proof. Established the durable expectation that every new dispatch branch ships with a reachability test; this later became linted in CI.
- **CI hardening (2026-07-27–28):** Replaced personal setup actions with direct `rustup` and GitHub cache actions; adopted architecture-aware cache keys and workflow concurrency cancellation. CLI ORT coverage became a dedicated lane, and Windows path-safety remains a targeted fast-coverage exception.
- **CI shape (2026-07-28):** Full coverage remains required on PRs. A parallel uninstrumented Linux fast job offers early feedback but never substitutes for the full gate. Windows ARM64 retains tests/clippy but not llvm-cov reporting due to rust-lang/rust#150123; coverage signal remains supplied by Linux, Windows x86_64, and macOS.
- **Benchmark discipline (2026-07-28):** Per-PR benchmarks compare merge-base first and PR second, remain informational, and flag ≥15% / ≥30% deltas. Real regression gates remain profile throughput floors plus dispatch-reachability tests on representative hardware.
- **CLI test contracts:** Fixed `--decode-skip 0` accounting and stdout/stderr JSON routing; hardened REPL e2e tests against nondeterministic stderr and preserved stdout-specific behavioral assertions.
- **Scope update (2026-07-28):** #54 model-package and #299 LoRA are owned by another squad and remain out of scope.

## 2026-07-29T12:30:00Z — tiny-reasoning-fixture round 1 (PR #410)

Reviewed round 1 of the tiny reasoning fixture. Locked out after round 1; rounds 2 and 3 proceeded under Gaff (PR #410 REJECT) and Luv (PR #411 rounds 2 and 3). Inbox drop `pris-tiny-reasoning-fixture.md` was lost when the worktree was deleted before Scribe ran; content reconstructed into `.squad/decisions.md` (reconstructed rules section, 2026-07-29).

---

## 2026-08-10 — EP Plugin Export: Validation Strategy + Environment Feasibility

**Requested by:** @justinchuby  
**Task:** Recon/plan pass for outbound EP plugin export (our EPs as ORT-loadable dylibs).

### Environment findings

- Installed ORT 1.28.0 via pip to `.ort-probe/` venv; confirmed `libonnxruntime.so.1.28.0`
  is present at `.ort-probe/lib/.../onnxruntime/capi/`.
- ORT 1.28 dynamic symbol table has **exactly 2 exports**: `OrtGetApiBase` and
  `OrtSessionOptionsAppendExecutionProvider_CPU`. All internal symbols are stripped.
- ORT 1.28 is backward-compatible with API v27 (our ort-sys target version).
- ORT 1.28's public C API **does not expose a function to load a plugin EP from a file path**.
  `RegisterCustomOpsLibrary_V2` exists only for custom ops. `add_provider(name, config)`
  requires the EP to already be registered by name in the OrtEnv. L3 is blocked.

### Output

- Created `docs/EP_PLUGIN_EXPORT_TEST_PLAN.md` with:
  - Full environment feasibility report (ORT 1.28 symbol evidence, API version compatibility)
  - Three-level test ladder: L1 (symbol export assertion), L2 (in-process dlopen ABI driver),
    L3 (real ORT e2e, marked not achievable with current wheel + rationale)
  - Existing test surface mapped to reuse paths
  - Fixture inventory (existing .onnx models and how to synthesize a new one)
  - CI integration plan

### Decisions

None recorded (plan pass; no source changes).

---

## 2026-08-10 — L3 Test Built and Executed

### Context
Challenger proved that ORT's `RegisterExecutionProviderLibrary` EXISTS in the vtable
(available since ORT 1.22). My prior recon using `nm -D` was the wrong instrument —
the ORT C API is entirely a vtable behind `OrtGetApiBase`.

### What was done
1. Built L3 end-to-end test: `crates/onnx-runtime-ep-cpu-plugin/tests/plugin_ort_e2e.rs`
2. Created fixture: `crates/onnx-runtime-ep-cpu-plugin/tests/fixtures/add_1x4/model.onnx`
   (float32 Add, opset 17, shape [1,4])
3. Test drives full ORT 1.27.0 registration path against our cdylib
4. Added pre-flight vtable check to catch missing entries without segfaulting

### Real result (ORT 1.27.0)
- ✅ CreateEpFactories succeeds (factory pointer returned)
- ✅ Pre-flight confirms: GetName, GetSupportedDevices, CreateEp, ReleaseEp populated
- ❌ **GetVendor, GetVendorId, GetVersion are None** in the factory vtable
- ❌ ORT segfaults at `RegisterExecutionProviderLibrary` calling `GetVendor` (offset 16)
- Stages NOT reached: registration, device enumeration, session creation, Run

### What blocks full green
Fix required in `crates/onnx-runtime-ep-plugin/src/factory.rs`:
Add `GetVendor: Some(...)`, `GetVendorId: Some(...)`, `GetVersion: Some(...)` to the
`OrtEpFactory` vtable initialization (currently `..Default::default()` leaves them None).

### Corrections to test plan
- §2.4: "ORT cannot load plugins" → WRONG; corrected to YES via vtable API
- §L3: "Not achievable" → NOW achievable; test written and running
- Symbol name: `CreateEpFactories` (not `CreateEpApiFactories`)

---

## 2026-08-10 — EP Plugin Conformance Test Harness (squad/ep-plugin-export)

### ⚠️ RETRACTION: Prior "e2e impossible" claim

In the previous session I claimed end-to-end testing was **impossible** because
`nm -D libonnxruntime.so` showed only 2 exported symbols. **This was wrong.**

The ORT C API is a vtable — `RegisterExecutionProviderLibrary`, `GetEpDevices`, and
`SessionOptionsAppendExecutionProvider_V2` are slots inside the `OrtApi` struct
obtained via `OrtGetApiBase()->GetApi(ORT_API_VERSION)`, not exported as free
functions. They are invisible to `nm -D` by design. Challenger investigated and
formally overturned my claim; the decision record is at
`.squad/decisions/inbox/challenger-ort-plugin-abi-truth.md`. I accept and acknowledge
that retraction.

### Work done this session

**Bug fixed in ep.rs (Nabil's file — strictly minimal):**
- `OrtEp` struct initializer had `ValidateCompiledModelCompatibilityInfo: None`
  which belongs to `OrtEpFactory`, not `OrtEp`. Removed the stray field; added
  `..Default::default()` for forward-compatibility with future optional fields.
  This was the known compile-blocking error.

**Tests built:**

L1 — ABI surface (`plugin_export_abi.rs`):
- `l1_nm_exported_symbols`: `nm --dynamic` confirms exactly 2 T symbols, no leakage
- `l1_readelf_dyn_syms`: `readelf --dyn-syms` confirms FUNC GLOBAL in .dynsym

L2 — dlopen direct drive (`plugin_export_abi.rs`):
- `dlopen_and_create_factory` (pre-existing, verified)
- `compute_add_end_to_end` (pre-existing, verified)
- `compute_add_broadcast` (pre-existing, verified)
- `l2_fail_closed_unsupported_api_version`: NEW — verifies plugin rejects null-API host

L3 — Real ORT 1.27.0 (`plugin_ort_e2e.rs`, rewritten):
- `ort_api_sanity`: PASSES — all 18 plugin-EP vtable slots non-null in ORT 1.27
- `diag_ort_ep_api_nullcheck`: PASSES — full audit printed
- `ort_register_ep_library`: IGNORED (see blocker below)
- `ort_loads_our_ep_and_runs_model`: IGNORED (see blocker below)
- `ort_unsupported_op_declines_not_crashes`: IGNORED (see blocker below)

**Fixture added:** `tests/fixtures/nonzero_1x4/model.onnx` (NonZero op, not supported by our EP)

### Bug found and documented (not fixed — Nabil's scope)

`factory.rs::factory_get_supported_devices` returns `*out_num = 0` (zero EP devices).
ORT 1.27.0 calls `GetSupportedDevices` inside `RegisterExecutionProviderLibrary` and
segfaults (SIGSEGV) when the factory reports zero devices. Confirmed via isolation:
ORT API functions are all non-null; crash is in `RegisterExecutionProviderLibrary`
calling our callback with no return path, then ORT crashing on 0 devices.

Fix needed: call `OrtEpApi::CreateEpDevice` per CPU hardware device in
`factory_get_supported_devices`. File: `crates/onnx-runtime-ep-plugin/src/factory.rs`.

### Final test counts
```
plugin_export_abi: 6 passed, 0 failed, 0 ignored
plugin_ort_e2e:    2 passed, 0 failed, 3 ignored
```
Zero crashes, zero false passes. All failures are genuine, precisely documented.

---

## 2026-08-10 — EP Conformance Suite (squad/ep-plugin-export)

**Objective:** Convert the single "Add 1×4" proof point into a real conformance suite.

### Summary

Nabil fixed `GetSupportedDevices` (factory.rs) so ORT no longer segfaults on
registration. The original `ort_loads_our_ep_and_runs_model` test previously
verified a full end-to-end pass (Run returned [6,8,10,12]).

This session broadened coverage by adding 6 new ONNX fixtures and 8 new L3
conformance tests, serialising all EP tests via `ORT_EP_LOCK: Mutex<()>` to
prevent non-deterministic parallel ORT state corruption.

### Tests un-ignored

| Test | Status |
|------|--------|
| `ort_register_ep_library` | ✅ Un-ignored (passes) |
| `ort_unsupported_op_declines_not_crashes` | ✅ Un-ignored (passes — NonZero declined, ORT default CPU EP handles Run correctly) |
| `conformance_add_broadcast` | ✅ Un-ignored (got [[11,22,33],[14,25,36]]) |
| `conformance_add_dynamic_dim` | ✅ Un-ignored (got [6,8,10,12] with batch=1 at runtime) |
| `conformance_add_int32` | ✅ Un-ignored (got [11,22,33,44]) |
| `conformance_chain_add_mul` | ✅ Un-ignored (got [4,6,8,10] — topological intermediates proven) |
| `conformance_matmul_2d` | ✅ Un-ignored (got [[4,2],[10,5]]) |
| `conformance_mixed_partition` | ✅ Un-ignored (ORT partitions Add→our EP, NonZero→default EP; final output correct) |

### Bugs found

**BUG 1 (factory.rs / Nabil):** `OrtEpDevice` descriptor becomes corrupt after
≥6 `RegisterExecutionProviderLibrary`+Run+Unregister cycles in the same process.
Values observed: `DeviceType:-112 MemoryType:-85 VendorId:31163 DeviceId:3139`
(uninitialized/dangling pointer). Root cause: likely stack allocation or
incorrect lifetime in `GetSupportedDevices`. Blocks `conformance_multiple_run_calls`,
`conformance_two_sessions`, and `ort_loads_our_ep_and_runs_model` (when run as 7th+
cycle). Not observed until cycle 7; default suite uses ≤6 cycles.

### Final test counts

```
Default run (cargo test -p onnx-runtime-ep-cpu-plugin):
  unittests:         0 passed, 0 failed, 0 ignored
  plugin_export_abi: 6 passed, 0 failed, 0 ignored
  plugin_ort_e2e:   10 passed, 0 failed, 3 ignored
  TOTAL:            16 passed, 0 failed, 3 ignored

With --include-ignored:
  plugin_ort_e2e:    8 passed, 5 failed (BUG 1 manifests at cycle 7+)
```

### Fixtures added

`add_broadcast`, `chain_add_mul`, `matmul_2d`, `mixed_partition`,
`add_int32`, `add_dynamic_dim` — all committed under `tests/fixtures/`.
Generator: `tests/fixtures/generate_fixtures.py`.

---

## 2026-08-10 — EP plugin conformance final pass (branch: squad/ep-plugin-export)

**Context:** Deckard fixed the use-after-free in commit c92838dba (`OrtMemoryInfo` released while ORT held the raw pointer; `CreateCpuMemoryInfo` replaced with correct pre-1.22 API `CreateMemoryInfo_V2`).

**Actions taken:**

1. **Un-ignored `ort_loads_our_ep_and_runs_model`** — allocator bug confirmed fixed. Test passes.

2. **Un-ignored `conformance_multiple_run_calls`** — use-after-free fix confirmed. Test passes (7th+ cycle in suite, clean).

3. **Fixed and un-ignored `conformance_two_sessions`** — test bug: `EpDevice_EpName` returns factory's `GetName` ("cpu_ep"), not the registration key ("cpu_ep_2sess"). Corrected the device-search comparison and assertion message. Test passes.

4. **Added `conformance_matmul_batched_nd`** — new fixture `tests/fixtures/matmul_batched_nd/model.onnx` (MatMul [2,3,4]×[2,4,2]→[2,3,2]). Proves batched-ND MatMul dispatch through ORT. All 12 output values independently verified.

5. **Added `stress_register_run_unregister_cycles`** — 25 complete cycles (≥4× the failure threshold of the fixed bug). Every cycle verifies Run output. Regression gate for use-after-free.

6. **f16/bf16** — CPU kernels support both dtypes; ORT plugin cannot route f16/bf16 nodes without `GetKernelRegistry` type-constraint metadata. Documented as coverage gap, no fake test written.

**Results (two runs):**
- Run 1: `test result: ok. 15 passed; 0 failed; 0 ignored`
- Run 2: `test result: ok. 15 passed; 0 failed; 0 ignored`
- Suite is order-independent (stress test alone exceeds the former corruption threshold).

**Decision filed:** `.squad/decisions/inbox/pris-ep-conformance-final.md`

## 2026-08-10 — Trait ↔ C-ABI parity integration tests

**Branch:** `squad/ep-plugin-parity-cuda`
**File:** `crates/onnx-runtime-ep-plugin/tests/trait_cabi_parity.rs`

Added 9 integration tests proving the capability-parity rule between the Rust
`ExecutionProvider` trait path and the ORT plugin C ABI path:
- Capability parity: supported ops claimed by both, Declined ops excluded by C ABI only
- Numerical parity: memory roundtrip and device copy are bit-exact
- Error parity: unsupported ops declined by both, shape-Declined is C-ABI-only filter

**Encoded parity rule:** `C_ABI_claims = trait_claims ∩ { nodes where ShapeInference::for_node ≠ Declined }`

**Status:** Tests written and formatted; lib does not compile due to in-flight changes
from Deckard (`ep.rs:114,406`) and Nabil (`device.rs:121,221,466,556`). These are
transient teammate edits, not bugs in the parity logic.

**Decision filed:** `.squad/decisions/inbox/pris-trait-cabi-parity.md`

## 2026-08-10 — Lint fix, parity test strengthened, f16/bf16 verdict (branch: squad/ep-plugin-parity-cuda)

**Task 1 — Two lints cleared in `trait_cabi_parity.rs`**

1. **PI approximation (`3.14`)** at line 235: replaced with `3.5` — this test exercises
   memory roundtrip, not PI. The literal was incidentally close to π and triggered
   clippy's `approx_constant` lint.

2. **Unused `ep` variable** at line ~302 in `error_parity_declined_shape_inference_is_cabi_only`:
   The `ep` was unused — a real weakness, not just a lint. The test only checked
   shape inference twice but never verified the actual trait/C-ABI divergence. Fixed by
   rewriting the test to exercise all three steps of the parity rule:
   - Step 1: `OrtGraphView::query_capabilities(&ep)` returns a claim (trait says yes).
   - Step 2: `ShapeInference::for_node` returns Declined for the same node.
   - Step 3: Simulating the C ABI filter (the `∩` from `ep_get_capability`) produces
     empty — divergence confirmed end-to-end. The `ep` is now genuinely used.

`cargo clippy -p onnx-runtime-ep-plugin --all-targets -- -D warnings`: **clean**.

**Task 2 — f16/bf16 verdict: OUTCOME 1 (our EP accelerates f16/bf16)**

Ran both previously-ignored tests against commit `577047a74`:
```
test conformance_add_float16  ... ok
test conformance_add_bfloat16 ... ok
```

Bit-exact output confirmed:
- Float16: `[0x4000, 0x4400, 0x4600, 0x4800]` (2.0, 4.0, 6.0, 8.0)
- BFloat16: `[0x4000, 0x4080, 0x40C0, 0x4100]` (2.0, 4.0, 6.0, 8.0)

**Why this is outcome 1 and not outcome 2:**
`build_cpu_registry_with_descriptors()` (Deckard's landing) derives Float16/BFloat16
type-constraint metadata; `crates/onnx-runtime-ep-cpu-plugin/src/lib.rs` wires it
through `GetKernelRegistry` via `build_kernel_registry_entries()`. ORT uses this
metadata to route f16/bf16 nodes to our EP rather than falling back to its built-in CPU EP.

`#[ignore]` removed from both `conformance_add_float16` and `conformance_add_bfloat16`.
Doc comments updated to remove the "blocked" framing.

**Final test counts:**
```
cargo clippy -p onnx-runtime-ep-plugin --all-targets -- -D warnings: clean
cargo test -p onnx-runtime-ep-plugin:
  unittests: 132 passed; 0 failed
  trait_cabi_parity: 9 passed; 0 failed; 0 ignored
cargo test -p onnx-runtime-ep-cpu-plugin -- --include-ignored:
  plugin_export_abi: 6 passed; 0 failed; 0 ignored
  plugin_ort_e2e:   17 passed; 0 failed; 0 ignored
  TOTAL:            23 passed; 0 failed; 0 ignored
```

**Decision updated:** `.squad/decisions/inbox/pris-trait-cabi-parity.md`


Two compile errors blocked the tests:
1. `DataType::Float` → corrected to `DataType::Float32` (6 occurrences, test file only)
2. `KernelMatch` missing `#[derive(Debug)]` → added to `crates/onnx-runtime-ep-api/src/kernel.rs`

Running the tests revealed **the parity rule holds, but the initial test assumptions were wrong**:
- `ShapeInference::for_node` does NOT decline Squeeze/ReduceMean/Conv without attributes —
  it is smarter than `for_op`: it reads input shapes and uses defaults, successfully inferring
  Squeeze with empty axes, ReduceMean with no axes, and Conv from weight dimensions.
- The **confirmed Declined case**: `Unsqueeze` at opset ≥ 13 where axes come from `input[1]`
  (a runtime tensor, not an attribute). Tests updated to use this real example.
- Graphs built without opset imports get `effective_opset() = 0`, which causes `supports_op`
  to decline everything. Fixed by adding `graph.opset_imports.insert(String::new(), 13)` in the
  test graph builder.
- `OrtGraphView::query_capabilities` is the trait-only first half of the C ABI path; it does
  NOT apply the shape-inference filter. The filter lives in `ep.rs:ep_get_capability`. Tests
  that asserted "C ABI claims = 0" via `query_capabilities` were wrong. Fixed to test the
  filter predicate directly.

**Final counts:**
- `cargo test -p onnx-runtime-ep-plugin`: **127 passed; 0 failed** (9 integration parity tests all pass)
- `cargo test -p onnx-runtime-ep-cpu-plugin`: **15 passed; 0 failed; 2 ignored**

**Task 2 — f16/bf16 end-to-end: honest ignored tests**

`registry_entries()` on `CpuExecutionProvider` has not landed (Deckard, owner file:
`crates/onnx-runtime-ep-cpu/src/provider.rs`). Without it ORT does not route Float16/BFloat16
nodes to our EP via `GetKernelRegistry`. Tests written with real fixtures and exact bit-pattern
assertions; `#[ignore]`-d with precise reason naming the blocking file and owner.

New ignored tests:
- `conformance_add_float16` — `tests/fixtures/add_float16/model.onnx` created
- `conformance_add_bfloat16` — `tests/fixtures/add_bfloat16/model.onnx` created
- New helpers `make_float16_tensor` / `make_bfloat16_tensor` added to the test file

**Decision updated:** `.squad/decisions/inbox/pris-trait-cabi-parity.md`
