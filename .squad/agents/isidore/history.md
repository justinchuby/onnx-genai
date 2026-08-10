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

