# Isidore — History Archive

Pre-2026-08-11 entries archived during Scribe compaction 2026-08-12T02:00:00Z.

## 2026-07-26 — Joined the team
Cast into the CPU & Edge pod. Standing directive: optimizations must be portable (consumer/edge hardware, not just H200); every perf claim backed by a benchmark; SIMD/NPU paths must match the scalar/f64 reference within a justified tolerance and be locked with regression tests.

## 2026-07-27 — CLI native CUDA feature plumbing
Added a first-class `native-cuda` CLI feature for the native backend plus hand-written CUDA EP, clarified ORT-vs-native CUDA feature comments/docs, and recorded the remaining `--backend` CLI gap for one-shot native decode.

## 2026-07-27 — REPL e2e cross-platform gates
Un-gated the reasoning REPL e2e helper and two reasoning tests that only use portable piped stdin/stdout process driving. Left the idle Ctrl-C test Unix-only with an explicit rationale: it sends SIGINT with `kill`, while Windows needs reliable `GenerateConsoleCtrlEvent` console-process-group plumbing. Recorded the team rule that REPL e2e tests gate only on genuine platform dependencies, never by default.

## 2026-08-10 — N3 fix: panic guards on CreateEpFactories / ReleaseEpFactory (lib.rs)

**Finding:** N3 (MEDIUM, ship-blocking) from Holden's re-audit. The
`export_ep_factories!` macro generated `CreateEpFactories` and `ReleaseEpFactory`
without `catch_unwind`, leaving the user-supplied constructor and `ep.name()`
free to panic across the C ABI boundary into ORT's dlopen path — undefined
behaviour.

**Fix (lib.rs only):**
- `CreateEpFactories`: wrapped in `catch_unwind(AssertUnwindSafe(...))`. On panic, `*out_num = 0` and `panic_to_fail_status(...)` returns an `ORT_FAIL` status.
- `ReleaseEpFactory`: return type corrected to `void` (per ORT ABI). Body wrapped in `let _ = catch_unwind(...)`.
- Added `#[doc(hidden)] pub fn panic_to_fail_status` in `lib.rs`.
- Macro is fully hygienic.

**Validation:** `cargo build -p onnx-runtime-ep-cpu-plugin` → Finished. Clippy clean. 66/66 baseline passed; 2 new N3 regression tests added.

## 2026-08-10 — Clippy lint cleanup (lib.rs test body)

Removed `mut` from unused-mut binding; replaced typed diverging `let` with bare `panic!()`. Macro body untouched. Tests: 82 passed / 0 failed (ep-plugin), 21 passed (ep-cpu-plugin).

---

## ARCHIVED 2026-08-12 (compaction wave)

### 2026-08-11 — nxrt Host Loader (§524)
Implemented `crates/onnx-runtime-ep-nxrt-host/`. dlopen via `libloading`, version negotiation, `NxrtExecutionProvider` implementing `ExecutionProvider`, panic containment. 4 passed, 0 failed.

### 2026-08-11 — nxrt Host Loader ABI Reconciliation
Deleted `src/abi_contract.rs`. Rewrote to real protocol: `NxrtNegotiate` → `NxrtCreateEpFactories`. `provider_adapter.rs` holds `*mut NxrtEpVtable`. Borrowed-pointer rules: EP `name` copied to owned String. 14 passed (4 unit + 10 integration).

### 2026-08-11 — ABI correctness: enum UB, struct_size, CUDA status loss
Commit `94bbbe545`. `NxrtStatus.code` → raw `u32`. struct_size checked before vtable access. CUDA `set_host_api(api)` before diagnostics. Vacuous CUDA tests replaced. 264 tests, 0 failures.

### 2026-08-11 — PR #762 third corrective wave: ABI safety
Commits `94bbbe545`, `24ba2fe31`. All three ABI issues genuinely fixed. Verified by Luv. Struct size 264 bytes unchanged.

### 2026-08-12 — iOS_CI_on_Mac failure triage
INFRA FLAKE. SSL cert failure (libcurl status 60) downloading FXdiv. No code change. Re-run recommended.

### 2026-08-12 — PR #32001 lockout revision (S1/S2/S3)
warn-and-disable (not FATAL_ERROR); `--use_apple_accelerate` added to build_args.py/build.py; removed dangling `MLAS_USE_APPLE_ACCELERATE` define. Head `d16a108252`.

### 2026-08-12 — PR #32003: Complete strict-aliasing fix
Fixed 4 remaining `reinterpret_cast<half2*>(&vec_a.member)` sites in `__CUDA_ARCH__ < 530` fallback. 0 member-punning sites remain. Commit `23dcfddaaf`.
