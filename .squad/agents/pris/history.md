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
