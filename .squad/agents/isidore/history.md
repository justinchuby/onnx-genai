# Isidore — History

## Project Context (joined day)
- **Project:** onnx-genai — Rust inference runtime for generative AI on ONNX Runtime.
- **State when joined:** Native CUDA EP beats/parity ORT on several Foundry models; correctness suite green (int8/block32 f64-adjudicated in #190). Team reorganized into pods; CPU & Edge pod formed to broaden hardware coverage beyond CUDA/Metal.
- **Role:** Mobile & Bindings Engineer — C ABI, Python (PyO3), Swift/Kotlin bindings, mobile/edge cross-compilation and packaging.
- **Requested by:** Justin Chu
- **Joined:** 2026-07-26

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
- `CreateEpFactories`: wrapped in `catch_unwind(AssertUnwindSafe(...))`. On
  panic, `*out_num = 0` (null-checked) and `panic_to_fail_status(...)` returns
  an `ORT_FAIL` status. Output factory array is left untouched.
- `ReleaseEpFactory`: return type corrected from `*mut OrtStatus` to `void`
  (per ORT ABI). Body wrapped in `let _ = catch_unwind(...)`. Panic silently
  swallowed; leaking is preferable to UB.
- Added `#[doc(hidden)] pub fn panic_to_fail_status` in `lib.rs` so the macro
  can call `status::fail_status` (pub(crate)) from consumer crates.
- Macro is fully hygienic: all paths `::std::...` / `$crate::...`, no
  reliance on caller-shadowed names.

**Validation:**
- `cargo build -p onnx-runtime-ep-cpu-plugin` → Finished (macro consumer works).
- `cargo clippy -p onnx-runtime-ep-plugin --lib -- -D warnings` → clean.
- `cargo test -p onnx-runtime-ep-plugin --lib` blocked by concurrent
  compile errors in `compute.rs` (Leon) and `kernel_ctx.rs` (Nabil). Baseline
  66/66 passed; 2 new N3 regression tests added that will pass once those files
  are resolved.
- Decision doc written to `.squad/decisions/inbox/isidore-ep-export-guards.md`.


## 2026-08-10 — Clippy lint cleanup (lib.rs test body)

Two clippy errors remained in the `export_ep_factories!` macro test block after
the N3 panic-guard fix landed:

**Error 1** `unused-mut` at `lib.rs:184`
- `let mut out_num: usize = 0;` — `out_num` is never mutated in the test (it
  is read via `assert_eq!` but never assigned after construction).
- Fix: removed `mut`.

**Error 2** `diverging_sub_expression` at `lib.rs:189`
- The original code used `let _: Box<dyn ExecutionProvider> = panic!(...)` to
  produce a typed diverging expression inside `catch_unwind`. Clippy flags a
  `panic!` (which diverges) used as the RHS of a `let`-binding as a
  "sub-expression diverges". The type ascription was unnecessary — the closure
  body only needs to panic, not return a value.
- Fix: replaced with a bare `panic!(...)` statement. No safety check weakened;
  `catch_unwind` still absorbs the panic and the `result.is_err()` assertion
  still validates the guard behaviour.
- The macro body itself (`CreateEpFactories`, `ReleaseEpFactory`) was NOT
  touched; both panic guards, null checks, and the ORT API version check are
  intact.

**Clippy result** (post-fix):
```
Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.78s
```
No errors from `lib.rs`. `ep.rs:499` (Deckard's `manual_dangling_ptr`) was
already resolved concurrently and does not appear.

**Test results**:
- `cargo test -p onnx-runtime-ep-plugin --lib` → 82 passed, 0 failed
- `cargo test -p onnx-runtime-ep-cpu-plugin` → 21 passed (6+15), 0 failed

---

## 2026-08-11 — nxrt Host Loader (§524)

Implemented `crates/onnx-runtime-ep-nxrt-host/` — the inbound dynamic-loading half of the nxrt plugin ABI.

**Delivered:**
- Cross-platform `dlopen` via `libloading` with `Arc<Library>` lifetime safety
- Version negotiation (fail closed on major mismatch)
- `NxrtExecutionProvider` implementing full `ExecutionProvider` trait
- Panic containment at C boundary via `catch_unwind`
- 6 negative-path error variants, all actionable

**Test results:**
- `cargo test -p onnx-runtime-ep-nxrt-host` → 4 passed, 0 failed
- `cargo clippy -p onnx-runtime-ep-nxrt-host --all-targets -- -D warnings` → clean
- Existing tests unaffected (`onnx-runtime-ep-plugin`, `onnx-runtime-ep-cpu-plugin`)

**Needs from Nabil:** ABI type exports in `onnx-runtime-ep-nxrt-abi`; once landed, replace local `abi_contract` module.

## 2026-08-11 — nxrt Host Loader ABI Reconciliation (§524 critical path)

**Problem:** The host loader carried a duplicate ABI in `src/abi_contract.rs` (opaque-handle model: `nxrt_abi_version`/`nxrt_create_ep`) that was incompatible with Nabil's shipped ABI (vtable model: `NxrtNegotiate`/`NxrtCreateEpFactories`). The two halves compiled independently but never linked — green build, nothing connected.

**Fix:**
1. Deleted `src/abi_contract.rs` entirely.
2. Added `onnx-runtime-ep-nxrt-abi = { workspace = true }` dependency.
3. Rewrote `loader.rs` to the real protocol: `NxrtNegotiate` → `validate_negotiation()` → `NxrtCreateEpFactories`.
4. Rewrote `provider_adapter.rs` to hold `*mut NxrtEpVtable` (owned, released on Drop) instead of opaque handle.
5. Factory vtables owned by `FactorySet` (in `Arc`), released on Drop with panic containment.
6. Borrowed-pointer rules honored: EP `name` pointer copied to owned `String` at creation time.

**Validation:**
```
cargo clippy -p onnx-runtime-ep-nxrt-host --all-targets -- -D warnings → clean
cargo test -p onnx-runtime-ep-nxrt-host → 14 passed (4 unit + 10 integration)
cargo check --workspace → ok
grep for duplicate ABI symbols → none found
```

**Nothing missing from Nabil's ABI.** All types and functions the host needs are exported.
