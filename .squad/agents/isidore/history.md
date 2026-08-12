# Isidore — History

## Role summary
Mobile & Bindings Engineer — C ABI, Python (PyO3), Swift/Kotlin bindings, mobile/edge cross-compilation and packaging. CPU & Edge pod. Joined 2026-07-26.

## Historical context
Joined during CUDA parity wave. Plumbed native-cuda CLI feature and cross-platform REPL e2e gates. Implemented N3 panic-guard fix (`CreateEpFactories`/`ReleaseEpFactory` wrapped in `catch_unwind`) and clippy cleanup for ep-plugin crate. Implemented nxrt host loader (`crates/onnx-runtime-ep-nxrt-host/`) with real ABI reconciliation to vtable model. Full detail in `history-archive.md`.

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


## 2026-08-11 — ABI correctness: enum UB, struct_size, CUDA status loss

**Task:** Fix four ABI-correctness findings from third Opus review of PR #762.

**Changes (commit 94bbbe545):**

1. **Unknown enum discriminant UB removed.** `NxrtStatus.code` changed from
   `NxrtStatusCode` (enum) to `u32` (wire type). Safe accessor
   `status_code() -> Option<NxrtStatusCode>` added via `from_u32()`.
   Test `unknown_discriminant_does_not_cause_ub` proves no UB: a status with
   code=255 returns `None` from `status_code()`.

2. **struct_size validated before vtable access.** `provider_adapter.rs` now
   checks `struct_size >= offset_of(create_ep)+size` before dereferencing
   factory.create_ep, and validates EP struct_size after creation.

3. **CUDA diagnostic status loss fixed.** `CreateEpFactories` now calls
   `set_host_api(api)` from `api_base` BEFORE `construct_ep_with_stream()`.
   Previously, the error status was null (ORT interprets as success).

4. **Vacuous CUDA tests replaced.** `cuda_plugin_error_status_or_null_without_ort`
   and `cuda_plugin_diagnostic_message_contract` replaced with tests that assert
   specific properties (out_num always zeroed, diagnostic message is actionable).

**Pre-fix failure evidence:**
- Fix 1: Before the change, code like `s.code = 255u32` wouldn't compile because
  `code` was typed as enum. With the old type, receiving 255 from a plugin would
  be UB (invalid enum discriminant). Now it compiles and `status_code()` returns None.
- Fix 2: No struct_size check existed; reading past a smaller struct would occur.
- Fix 3: `fail_status()` returns null when host API not set (documented in
  `status.rs:31`). New code sets API first.
- Fix 4: Old tests only called `eprintln!` — no assert could fail.

**Validation:**
- `cargo test --no-fail-fast` across 5 EP crates: 264 tests, 0 failures.
- `cargo check --features cuda`: compiles.
- `cargo clippy` on owned crates: clean.
- `cargo fmt --check`: clean.
- Workspace clippy has pre-existing failures in `onnx-genai-engine` (not mine).

## 2026-08-11 — PR #762 third corrective wave: ABI safety

**Task:** Fix ABI issues: `NxrtStatus` UB from unknown discriminant, `struct_size` not checked before vtable access, CUDA diagnostic status loss.

**Commits:** `94bbbe545`, `24ba2fe31`

- `NxrtStatus.code` is now raw `u32` (wire type). Safe accessor `status_code() -> Option<NxrtStatusCode>` uses `from_u32()`. Unknown codes → `None` → fail closed. No transmute UB.
- Host validates `struct_size` before calling through any vtable slot.
- CUDA plugin initialises ORT host API before running diagnostics.

**Outcome:** All three ABI issues genuinely fixed. Verified by Luv. Struct size unchanged (264 bytes).

---

## 2026-08-12 — iOS_CI_on_Mac failure triage (PRs #31993, #31988)

**Task:** Diagnose `iOS_CI_on_Mac` (reported as "Objective-C-StaticAnalysis") failure on two unrelated PRs.

**Finding:** INFRA FLAKE. SSL certificate verification failure (libcurl status 60) downloading FXdiv dependency archive from github.com. Failure occurred before compilation — no source code involvement. PR #31988's identical job on a different runner passed the download stage normally.

**Action:** No code change. Re-run recommended.

## 2026-08-12 — PR #32001 review fixes

- Fixed S1 (warn-and-disable instead of FATAL_ERROR), S2 (build.py plumbing), S3 (remove dangling define).
- Verified: default Linux configure shows zero Accelerate references; `--use_apple_accelerate` ON on Linux warns and disables; `build.py --help` shows the new flag.
- Head: d16a108252. PR remains draft.

## 2026-08-12 — PR #32001 lockout revision (S1/S2/S3)

**Task:** Fix all three substantive review issues under reviewer lockout (Luba = author, Luv = reviewer, both barred).

**Changes (head d16a108252):**
1. **S1:** Replaced `FATAL_ERROR` with `message(WARNING ...) + set(onnxruntime_USE_APPLE_ACCELERATE OFF)` on non-Apple. Evidence: `onnxruntime_USE_SVE` (cmake/CMakeLists.txt:581) and `onnxruntime_USE_KLEIDIAI` (line 611) both use this warn+disable pattern.
2. **S2:** Added `--use_apple_accelerate` to `build_args.py` and forwarding as `-Donnxruntime_USE_APPLE_ACCELERATE=ON` in `build.py`. Option now reachable from standard upstream tooling.
3. **S3:** Removed `target_compile_definitions(onnxruntime_mlas PRIVATE MLAS_USE_APPLE_ACCELERATE=1)` — no consumer exists yet; avoids static-analysis noise.

**Verified on Linux x86-64:** default configure shows zero Accelerate references; ON warns and disables on Linux; `build.py --help` shows new flag.

**What cannot be confirmed here:** `find_library(Accelerate)` resolution and actual linking on Apple SDKs — requires Apple CI.

---

### 2026-08-12 — PR #32003: Complete strict-aliasing fix in matmul_4bits_common.cuh

**Session:** S4 — Complete partially-applied `memcpy` fix for strict-aliasing UB.

**What was done:**
- Fixed 4 remaining `reinterpret_cast<half2*>(&vec_a.member)` sites in the `__CUDA_ARCH__ < 530` fallback overload, replacing with `memcpy` into named locals matching the existing style.
- Swept the file: 0 member-punning reinterpret_cast sites remain.
- Left `half2* sums_half2 = reinterpret_cast<half2*>(sums)` alone (array reinterpretation, not member-punning; idiomatic CUDA; not flagged by compiler).
- Passed `clang-format --dry-run --Werror`.
- Full `nvcc` compile blocked by missing ORT deps (expected on this host).

**Commit:** `23dcfddaaf` on `nxrt/cuda-matmul4bits-strict-aliasing`. PR #32003 remains draft.

## 2026-08-12 — PR #32003 complete aliasing fix (4 vec_a sites)

Under lockout revision (Deckard locked out as author). Fixed 4 remaining `reinterpret_cast<half2*>(&vec_a.{x,y,z,w})` dereferences in `__CUDA_ARCH__ < 530` fallback — identical UB to `vec_permuted` sites Deckard fixed. Replaced with `memcpy` into named `half2` locals. Left `reinterpret_cast<half2*>(sums)` alone: canonical CUDA vectorised-access idiom, not the flagged pattern. 0 member-punning sites remain. Head `23dcfddaaf`.
