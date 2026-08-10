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
