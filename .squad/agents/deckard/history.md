# Deckard — History (compacted 2026-08-11T12-05-00Z)

**Role:** Systems developer on the Rust runtime, ORT2 loader/shape/IR/EPContext tracks, and CPU/CUDA execution performance. Preserve model-agnostic dispatch, fail-closed claims, checked arithmetic, byte-exact serialization, precision-sensitive tests, and reviewer-lockout ownership transfers.

## Durable lessons
- Repeated invariants: model-agnostic dispatch, fail closed at claim time, checked arithmetic, byte-exact serialization, and precision-sensitive tests.
- Parallel commit-producing work requires separate worktrees; reviewer rejection transfers ownership and must be recorded.
- **ORT plugin-EP ABI:** `OrtMemoryInfo` passed to `EpDevice_AddAllocatorInfo` must outlive the `OrtEpDevice` — ORT stores the raw pointer; do not call `ReleaseMemoryInfo` on success. Use `CreateMemoryInfo_V2`. Release only on failure.
- **Shape inference fail-closed:** `Declined` is the correct return for any unmodelled op; never fall back to `SameAsInput(0)`.
- **`validate_dims` must be wired** in the actual read path, not just implemented.
- `OrtGraph*` / `OrtNode*` handles must NOT be stored beyond callback return.

## Historical context

Pre-2026-08-10 entries in `history-archive.md` (shape inference overhaul, EP device lifetime UAF, clippy lint cleanup).

2026-08-10 ep-plugin-export wave archived in `history-archive.md` under "Archive batch 2026-08-10": GetKernelRegistry + NEW-2, f16/bf16 registry, dtype-aware GetCapability, needless_borrow fix.

## Current entries (wave: ep-plugin-parity-cuda, 2026-08-11)

### 2026-08-11 — CUDA plugin wiring: real EP behind feature gates

Rewrote `onnx-runtime-ep-cuda-plugin/src/lib.rs` from panic-stub to real plugin. `cuda` OFF → zero factories + error status; `cuda` ON → EP construction attempted with GPU DeviceSupport. Made `fail_status` pub in `status.rs`. `prefetch_lazy_weight` left as `Ok(false)` stub (no `try_without_eviction` API; implementing with eviction violates standing directive). Both `cargo check` variants succeed on CUDA-less host. 3 cuda-plugin + 23 cpu-plugin tests pass.

### 2026-08-11 — PR #762 CI unblock: clippy Range::contains lint + registry consolidation

`kernels/mod.rs:2293`: `delta >= 0 && delta < 50` → `(0..50).contains(&delta)` (clippy `manual_range_contains`, fatal under `-D warnings`). Assertion bounds a static structural count difference; cannot flake.

`OnceLock`-based `shared_registry_with_descriptors()` fixture — registry+descriptors built once across four tests. On Linux: negligible timing/memory difference; Windows ARM64 flake pre-existing (SPMD subprocess OOM). All tests pass; clippy clean; fmt clean.

### 2026-08-11 — BFloat16 contrib U type constraint fix (#31974)

Contrib macro changed from `(T)` to `(T, U)`. Narrow types registered with `U=float`. Consistent with CUDA contrib and schema. No runtime correctness impact (SrcDispatcher always uses `ComputeImpl<T, float>`). 10/10 `LayerNormBFloat16*` tests pass.

### 2026-08-11 — macOS CI failure investigation (PRs #31973, #31974)

All failures are INFRA FLAKES: Gradle CDN timeout (#31973 coreml Debug), pytorch_cpuinfo FetchContent failure (#31974 webgpu Debug), DNS error in quantization test (#31974 coreml Release). Cross-PR evidence: same jobs pass on #31969–#31972. Both PRs safe to mark ready-for-review. No code changes needed.

### 2026-08-11 — Session update (Scribe append)
Both upstream PRs marked ready-for-review. `.squad/` git history purge complete.

## 2026-08-11 — Fix stale MRotaryEmbedding doc upstream

- **PR:** https://github.com/microsoft/onnxruntime/pull/31985 (draft)
- **Branch:** `nxrt/fix-mrope-contrib-doc` on `justinchuby/onnxruntime`
- **Fix:** Removed inaccurate `(or omitting it)` from `docs/ContribOperators.md` to match schema in `bert_defs.cc`. `mrope_section` is required so the phrase was wrong.
- **Origin:** #31728 introduced the mismatch; CI has been red since.
- **Scope:** Single 1-line change, hand-edited to match generator output.

## 2026-08-11 (upstream CI correction wave) — PR #31985 (MRotaryEmbedding doc fix)

Traced `Windows GPU Kernel Documentation Validation` CI failure to `docs/ContribOperators.md` stale text from upstream PR #31728 (`e415ef9afd`). Confirmed `mrope_section` is a required attribute (no default in `bert_defs.cc`); the phrase "(or omitting it)" was factually wrong, not just stale. Opened PR #31985 as a one-line hand-edit fix. PR reached 86/86 CI green and was marked ready for review.

## 2026-08-12 — PR #31988 build fix (CUDA 13.0 -Werror)

Fixed `-Werror=strict-aliasing` and `-Werror=unused-parameter` in `matmul_4bits_common.cuh`.
Replaced `reinterpret_cast` type-punning with `memcpy`; added `(void)` casts for bf16
params guarded by `__CUDA_ARCH__ >= 800`. Pushed `0ba804b7f7`.
Could not compile locally (no nvcc). iPhone failure is a dep-download flake, left alone.

## 2026-08-12 — PR #31988 TensorRT CI triage (initial assumption disproved)

- Diagnosed CUDA 13.0 `-Werror=strict-aliasing` and `-Werror=unused-parameter` failures in `matmul_4bits_common.cuh`. Pushed `memcpy`-based punning and `(void)` casts (commit `0ba804b7f7`).
- Initial assessment of `blockIdx`/`__threadfence` TensorRT errors: "CUDA-13 base-codebase incompatibilities, not ours." **This was disproved** by Leon's cross-PR comparison showing #31678 green / #31988 red. The errors were caused by our test's inclusion chain (`matmul_4bits_common.cuh` → CUB device headers from host `.cc`). Leon fixed by extracting a host-only header.

## 2026-08-12 — PR #32003 strict-aliasing standalone split

- Split strict-aliasing / unused-parameter fix out of #31988 into standalone draft PR #32003.
- Scope: single file `matmul_4bits_common.cuh` — `memcpy` replacements, `(void)` casts, `#include <cstring>`.
- Worktree: `/workspace/upstream/ort-aliasing`, branch `nxrt/cuda-matmul4bits-strict-aliasing`.
- Leak check clean. clang-format passes. nvcc parse-check OK (full compile blocked by missing gsl deps).
- PR URL: https://github.com/microsoft/onnxruntime/pull/32003

## 2026-08-12 — PR #32003 draft (strict-aliasing split from #31988)

Split strict-aliasing/`-Werror` `memcpy` fixes from #31988 into standalone draft PR #32003. Fixed `vec_permuted` overload and bf16 overload. Coordinator found 4 missed identical sites in `vec_a` (`__CUDA_ARCH__ < 530` fallback, lines 117–120). Isidore completed those under lockout. Lesson: grep for the full pattern when fixing aliasing, not just the first overload.

## 2026-08-12 — PR #31973 wording fix (x86-64 → x86)

Fixed inaccurate "x86-64" in comments to "x86 (32-bit and 64-bit)" / "x86" across
`mlas.h`, `layernorm_kernel_avx2.cpp`, and `test_layernorm.cpp`. The compile gate
is `MLAS_TARGET_AMD64 || MLAS_TARGET_IX86` so the narrower term understated scope.
Build: 41 passed, 2 disabled. Pushed as `4a16925a88`.

## 2026-08-12 — PR #31973 x86 wording correction (lockout from Challenger nit)

Challenger's delta review flagged `mlas.h:1699` saying "x86-64" while the `#if` covers
`MLAS_TARGET_AMD64 || MLAS_TARGET_IX86` (both widths). Changed to "x86 (32-bit and 64-bit)"
in `mlas.h` doxygen comment; "x86" in shorter inline comments and six `GTEST_SKIP` messages
in `test_layernorm.cpp` and `layernorm_kernel_avx2.cpp`. Comment-only; build verified 41/2.
Commit `4a16925a88`. Irony: a prior readability fix ("AMD64/IX86" → "x86-64") made the
comment less accurate.

## 2026-08-12 — CUDA-graph capture arc: PR #848 graph-truth SWA detection (MERGED)

Link 1 (**CLASSIFY**) of the 5-blocker capture chain for Muse-Glimmer-30B native decode.
Replaced the vestigial `sliding_window` (SWA) signal with graph-truth SWA detection, routing
Muse-Glimmer to shared-buffer / fixed-capacity KV (capture-stable). 463 tests. Merged first in
dependency order (#848 → #850 → #852 → #855 → #854). Shared arc result: native CUDA decode
**11.4 → 23.13 tok/s**, CUDA-graph capture fully engaged (1 captured segment, 0 eager seams).
