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

## 2026-08-11 — nxrt ABI round-trip tests + CUDA conformance runner

**Branch:** `squad/ep-plugin-parity-cuda` (PR #762)

**Delivered:**
- 10 nxrt ABI integration tests (`crates/onnx-runtime-ep-nxrt-abi/tests/nxrt_roundtrip.rs`)
  - Round-trip happy path, ownership/lifetime (Drop counter → 0), library-outlives-EP
  - Negative: version mismatch, missing symbol, missing library, factory error, zero devices, panic containment, null handle
- Test fixture plugin (`crates/onnx-runtime-ep-nxrt-testplugin/`) — standalone cdylib with env-var-controlled failure modes
- CUDA conformance runner (`scripts/cuda_conformance_runner.sh`) — single command, exit 0/1/2 (validated/failed/unvalidated)
- Created nxrt-abi submodules (`version.rs`, `vtable.rs`) so the workspace compiles — Nabil will replace with full implementations

**Bugs found:**
- `onnx-runtime-ep-nxrt-host/src/provider_adapter.rs:90` — missing `sync` method on `impl ExecutionProvider for NxrtExecutionProvider` (owner: Isidore)
- nxrt ABI contract mismatch: abi crate exports `NxrtNegotiate`/`NxrtCreateEpFactories` while host expects `nxrt_abi_version`/`nxrt_create_ep`/etc (owners: Nabil + Isidore)

**Test counts:** ep-plugin 154+9, nxrt-abi 19+10 = 192 pass, 0 fail. CUDA UNVALIDATED (no GPU).

**Decision:** `.squad/decisions/inbox/pris-nxrt-and-cuda-runner.md`

---

## 2026-08-11T00:40:00Z — nxrt ABI testplugin rebuilt on real shipped ABI

**Commit:** 99560c876  
**Branch:** `squad/ep-plugin-parity-cuda` (PR #762)

### What changed
- **Rebuilt testplugin on Nabil's real ABI:** `crates/onnx-runtime-ep-nxrt-testplugin` now
  depends on `onnx-runtime-ep-nxrt-abi` and uses the `export_nxrt_ep_factories!` macro.
  Old private duplicate symbols (`nxrt_abi_version`, `nxrt_create_ep`, etc.) removed.
- **Added to workspace:** removed crate-local `[workspace]` table, registered as member
  (not default-member) in root `Cargo.toml`. `cargo check --workspace` now builds it.
- **10 integration tests** in `crates/onnx-runtime-ep-nxrt-host/tests/nxrt_abi_roundtrip.rs`
  exercise the real cdylib via `libloading`:
  - Full lifecycle (negotiate → create factories → create EP → query → release)
  - Ownership: drop counter returns to zero
  - Negative: incompatible major, minor > host, unknown cap bits, missing lib,
    missing symbol, panic containment, factory error
- **CUDA runner updated** with libcublasLt.so.13 precondition check, GPU count check,
  and weight-offload phase. Exits 2 (UNVALIDATED) on this GPU-less host.

### Bug found
- `crates/onnx-runtime-ep-nxrt-host/src/abi_contract.rs` is a stale private duplicate
  of the ABI — the host loader still expects the old symbols. **Owner: Isidore.**

### Validation
- `cargo check --workspace`: PASS (includes testplugin)
- `cargo test -p onnx-runtime-ep-nxrt-host`: 4 lib + 10 integration = 14 passed
- `cargo test -p onnx-runtime-ep-plugin`: 154 lib + 9 parity = 163 passed
- `cargo test -p onnx-runtime-ep-cpu-plugin`: 6 + 17 = 23 passed
- CUDA: UNVALIDATED (exit 2)

## 2026-08-11 — nxrt fixture resolution fix & suite de-duplication

- Fixed `testplugin_path()` to resolve workspace-level `target/` with CARGO_TARGET_DIR and PROFILE support
- Added auto-build fallback so tests self-heal when cdylib is missing
- Deleted duplicate `crates/onnx-runtime-ep-nxrt-abi/tests/nxrt_roundtrip.rs` — authoritative suite is in host crate
- Added ENV_MUTEX to serialize env-var-dependent tests (fixed parallel flakiness)
- Added `onnx-runtime-ep-nxrt-testplugin` as dev-dep of host (ensures rlib is built)
- All suites green: ABI 30, host 14, plugin 163, cpu-plugin 23
- Clean-state verified: auto-build triggers correctly
- CUDA remains UNVALIDATED (no GPU)

## 2026-08-11 — Clippy fix + doc-test `no_run` analysis

### Task 1 — Clippy error in testplugin
Added `Default` impl for `TestNxrtEp` (delegates to `::new()`).
`cargo clippy -p onnx-runtime-ep-nxrt-host --all-targets -- -D warnings` → clean.

### Task 2 — `ignore`d doc-tests in onnx-runtime-ep-nxrt-abi
**Verdict: keep `ignore` for all four; `no_run` is not viable.** Rationale:

1. `export_nxrt_ep_factories!` (lib.rs ~103): example calls `MyExecutionProvider::new()`
   which is undefined — fails to compile regardless of `ignore`/`no_run`.

2. All three macro examples (`export_nxrt_ep_factories`, `export_nxrt_ep_negotiate_custom`,
   `export_nxrt_ep_create_custom`) and the `testing.rs` module example invoke macros that
   emit `#[unsafe(no_mangle)] pub unsafe extern "C" fn ...` items. Without an explicit
   `fn main()`, cargo wraps the doc-test body in `fn main() {}`. `pub` on an inner function
   generates a dead_code/unreachable-pub lint that fails under `-D warnings`, and
   `#[unsafe(no_mangle)]` on a nested fn is semantically meaningless. Both issues prevent
   `no_run` from working cleanly. The `#[no_mangle]` symbol-clash concern that may have
   originally motivated `ignore` is *not* the real blocker (each doc-test is its own binary);
   the item-inside-main-function issue is.

**Recommendation to Nabil:** Keep `ignore` on all four. Add a brief comment above each
doc-test block explaining why, e.g.:
  ```
  // `ignore` rather than `no_run`: macro emits `#[unsafe(no_mangle)] pub extern "C"` items
  // which are invalid inside the `fn main()` wrapper cargo generates for doc-tests.
  ```

**Macro coverage adequacy:** Sufficient. Nabil's `lib.rs` tests cover:
- `export_macro_negotiate_produces_correct_response` — symbol fires, returns correct ABI version
- `export_macro_create_factories_succeeds` — factory vtable allocated, released cleanly
- `export_macro_panic_in_constructor_contained` — panic → InternalError status (via catch_status_panic)
- `custom_negotiate_override_through_validate` — NxrtNegotiateOverride wrong_major path
- `custom_create_override_zero_factories` — NxrtCreateFactoriesOverride::zero path
All five pass. The doc-tests are purely illustrative usage examples, not the test surface.

### Validation (final)
- `cargo clippy -p onnx-runtime-ep-nxrt-host --all-targets -- -D warnings`: clean
- `cargo test -p onnx-runtime-ep-nxrt-abi`: 30 passed, 4 ignored (expected)
- `cargo test -p onnx-runtime-ep-nxrt-host`: 4 unit + 10 round-trip = 14 passed
- `cargo test -p onnx-runtime-ep-plugin`: 154 lib + 9 parity = 163 passed
- `cargo test -p onnx-runtime-ep-cpu-plugin`: 6 + 17 = 23 passed

---

## 2026-08-11T01:03:00Z — Last lint gate: NULL_API + collapsible-if (PR #762)

**Branch:** `squad/ep-plugin-parity-cuda`

### NULL_API verdict: genuinely redundant

`static NULL_API: OnceLock<ort::OrtApi>` in `tests/plugin_export_abi.rs:631` was scaffolding that was never wired up. The `l2_fail_closed_unsupported_api_version` test correctly uses a raw function pointer (`returns_null_api`) returned via `NULL_API_BASE.GetApi`, not a stored `OrtApi` instance. The fail-closed null-API behavior is fully covered by that test — `NULL_API` was an unused placeholder. **Removed** the dead static.

### Collapsed-if fix

`tests/plugin_ort_e2e.rs:66-67` — two nested `if` blocks collapsed to `if build_dir.exists() && let Ok(entries) = std::fs::read_dir(&build_dir)` as clippy suggested. The surrounding logic is a real directory scan returning `Some(lib_dir)` — no silent skip, just a None return if the dir doesn't exist or can't be read (correct behavior for a build-artifact finder).

### Validation

```
cargo clippy -p onnx-runtime-ep-cpu-plugin --all-targets -- -D warnings
→ Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.38s  (clean)

cargo test -p onnx-runtime-ep-cpu-plugin --test plugin_export_abi
→ test result: ok. 6 passed; 0 failed

cargo test -p onnx-runtime-ep-cpu-plugin
→ test result: ok. 17 passed; 0 failed (integration suite)

cargo test -p onnx-runtime-ep-plugin -p onnx-runtime-ep-nxrt-abi -p onnx-runtime-ep-nxrt-host
→ 30 passed (abi) · 4+10 passed (host) · 154+9 passed (plugin)

cargo fmt --all -- --check → EXIT:0
```

---

## 2026-08-11T01:40:00Z — Stale-artifact false-pass fix for cpu-plugin tests

### Context
Five CI lanes failed (`Fast (Linux)`, `Rust (Windows ARM64)`, `Rust coverage` ×3) at
`l1_nm_exported_symbols` because `plugin_export_abi.rs` and `plugin_ort_e2e.rs`
hardcoded `target/debug/libonnx_runtime_ep_cpu_plugin.so` without CARGO_TARGET_DIR
support, auto-build, or platform-aware extensions. Local "23 tests passing" was a
false positive from a stale artifact.

### Changes
- Created `crates/onnx-runtime-ep-cpu-plugin/tests/cdylib_resolve.rs` — shared helper
  with env override (`NXRT_CPU_PLUGIN_PATH`), `CARGO_TARGET_DIR`/`PROFILE` support,
  platform lib names (`.so`/`.dylib`/`.dll`), and auto-build fallback.
- Updated `plugin_export_abi.rs` and `plugin_ort_e2e.rs` to use `mod cdylib_resolve;`.

### Audit
- nxrt tests (`nxrt_abi_roundtrip.rs`): already correct (testplugin_path() pattern).
- `trait_cabi_parity.rs`: no artifact loading (uses Rust types directly).
- `onnx-genai-bench/tests/profile_native.rs`: has env override + graceful skip. OK.
- No other tests under `tests/` reference `target/` for built artifacts.

### Validation (clean state)
- `rm -f target/debug/libonnx_runtime_ep_cpu_plugin.so` → `cargo test -p onnx-runtime-ep-cpu-plugin` → 23 passed (auto-build triggered).
- `cargo test -p onnx-runtime-ep-plugin` → 154+9; `-p onnx-runtime-ep-nxrt-abi` → 30; `-p onnx-runtime-ep-nxrt-host` → 4+10.
- `RUSTFLAGS="-D warnings" cargo clippy --locked --all-targets -p onnx-runtime-ep-cpu-plugin` → clean.
- `cargo fmt --all -- --check` → clean.

---

## 2026-08-11T01:55:00Z — ORTCHAR_T portability fix for Windows

**Branch:** `squad/ep-plugin-parity-cuda` (draft PR #762)

### Problem
ORT e2e tests used `CString` for all path arguments, producing `*const i8`.
On Windows, ORT expects `*const u16` (UTF-16 `wchar_t`) for `ORTCHAR_T*` parameters.
12 `E0308` type errors on `Rust (Windows ARM64)` CI lane.

### Fix
Created `crates/onnx-runtime-ep-cpu-plugin/tests/ort_path.rs`:
- `OrtPathBuf::new(path)` — `#[cfg(windows)]` encodes to NUL-terminated UTF-16 `Vec<u16>`,
  `#[cfg(not(windows))]` wraps in `CString`.
- `as_ptr()` returns the platform-correct pointer type.
- Accepts `impl AsRef<Path>`, owns the buffer (no dangling pointer risk).

### Call sites changed (12 total in `plugin_ort_e2e.rs`)
- `RegisterExecutionProviderLibrary` path arg: lines 224, 278, 467, 675, 1474, 1755
- `CreateSession` model_path arg: lines 332, 512, 728, 1516, 1536, 1816

### Audit
- No other test files pass paths into ORT APIs via CString.
- nxrt tests use `libloading` which takes `OsStr` — already portable.
- `plugin_export_abi.rs` loads via `libloading::Library::new` — already portable.

### Verification
- Windows cross-check (`cargo check --target x86_64-pc-windows-msvc --tests`) failed at
  `bindgen` (missing Windows SDK `stdlib.h` on Linux host). **Windows compilation unverified
  locally — CI must confirm.**
- Code is type-correct by construction: `#[cfg(windows)]` branch returns `*const u16`,
  `#[cfg(not(windows))]` returns `*const c_char`.
- Linux: `cargo test -p onnx-runtime-ep-cpu-plugin` → 6 unit + 17 integration = 23 passed.
- Clippy: clean (`RUSTFLAGS="-D warnings"`).
- `cargo fmt --all -- --check`: clean.
- No regressions: ep-plugin 154+9, nxrt-abi 30, nxrt-host 4+10.

## 2026-08-11 — Committed gitignored ONNX fixtures; fixed mutex poisoning cascade

**Problem:** 11 `.onnx` test fixtures in `crates/onnx-runtime-ep-cpu-plugin/tests/fixtures/`
were never committed because `*.onnx` in `.gitignore` silently excluded them. The ORT e2e
suite (`ort_loads_our_ep_and_runs_model`, `stress_register_run_unregister_cycles`) failed
in CI with missing-file panics, and the shared `ORT_EP_LOCK` mutex got poisoned, causing
cascading `PoisonError` failures in unrelated tests.

**Fixes:**
1. Added 11 explicit `!` negation entries in `.gitignore` for the cpu-plugin fixtures.
2. Added `lock_ort_ep()` helper that recovers from `PoisonError` via `into_inner()` with
   a warning, preventing one test's panic from masquerading as failures in others.
3. Extended `generate_fixtures.py` with 4 missing generators (`add_1x4`, `add_float16`,
   `add_bfloat16`, `nonzero_1x4`) so the script is the complete source of truth.
4. Audited all crates for untracked test assets — none found beyond the 11 fixed.

**Validation:** 17/17 ORT e2e, 6/6 ABI, 30 nxrt-abi, 4+10 nxrt-host — all green.
Clippy clean, fmt clean.

## 2026-08-11 — ReleaseEpFactory UB fix + portable L1 symbol tests

**Context:** Three non-Linux CI lanes (Windows ARM64, Windows, macOS arm64)
all failed `dlopen_and_create_factory` in `plugin_export_abi.rs`. Root cause:
the test declared `ReleaseEpFactory` as returning `*mut OrtStatus` but the real
export returns `void`. Calling a void function through a status-returning
pointer is UB — on x86-64 Linux the garbage in RAX happened to be null (pass),
on arm64 it was non-null (fail).

**Changes:**
1. Fixed `ReleaseEpFactory` type alias to `fn(*mut OrtEpFactory)` (no return).
   Removed the status assertion — there is no status to check.
2. Audited all other `extern "C"` fn-pointer types in test files:
   - `CreateEpFactories` (line 62): correct (`-> *mut OrtStatus`), matches export.
   - `CreateFn` (line 624): correct, same as above.
   - `GetApiBaseFn` in `plugin_ort_e2e.rs` (line 120): correct (`-> *const OrtApiBase`).
   No other mismatches found.
3. Replaced ELF-only `l1_nm_exported_symbols` and `l1_readelf_dyn_syms` with:
   - `l1_required_symbols_resolve`: portable (libloading/dlsym), runs on all platforms.
   - `l1_no_symbol_leakage`: Linux-only `nm --dynamic` check, skips with message elsewhere.

**Verified on Linux:** 6/6 ABI tests green, 17/17 ORT e2e green, ep-plugin 154+9,
clippy clean, fmt clean.
**CI must confirm:** macOS arm64, Windows, Windows ARM64 — these could not be
tested locally.
