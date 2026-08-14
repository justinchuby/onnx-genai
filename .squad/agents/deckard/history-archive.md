# Deckard — History Archive

## Pre-2026-07-29 (archived by compaction)

Full pre-compaction history in original `history.md` before 2026-07-29 compaction.

## 2026-07-27T20:15:00Z — Kernel pre-binding (Stage 3)

- Implemented per-plan-node kernel pre-binding to eliminate the 2.15 µs/op dispatch tax.
- Added `kernel_bindings: Vec<Option<KernelKey>>` on Executor, `get_prebound` zero-alloc fast path on KernelCache.
- Static-shape graphs pre-populate bindings at build; symbolic graphs populate on first dispatch.
- Shape changes detected via `matches_shapes` (no alloc) and fall through to `get_or_create`.
- PREBIND_FAST_PATH_TEST_HITS / PREBIND_FALLBACK_TEST_HITS counters with paired tests.
- All session tests pass (211+), both clippy targets clean.

## 2026-08-02T10:05:00+0000 — #594 lockout revisions

- Took over #594 after Harry's reviewer lockout, ran `cargo fmt --all`, produced formatting-only fix.
- Fixed pinned shape-inference registry counts: operator_count 217→218, entry_count 262→263.

## 2026-08-02T11:40:00+0000 — #595 profile_native bench fix

- Authored and merged #595, restoring `reset_exec_phase_profile` so `profile_native --steady` bench binaries compile on main.

## 2026-08-02T19:00:00+0000 — PR #602 lockout revision

- Took over #602 after Harry's rejection. Added conservative `function_has_attribute_parameters` fail-closed behavior; preserved `ModelFunction.attributes`; added `ParamLeakyRelu` regression coverage.

## 2026-08-02T19:50:00+0000 — PR #604 phase-profile flake fix

- Authored and merged #604: replaced racing global assertions with phase-scoped snapshot checks. Validated 30/30 full-parallel lib runs.

## 2026-08-03T03:10:00+0000 — mobius PR #449 PackedMHA bias slot

- Added `bias` as 4th formal to `PackedMultiHeadAttention` fallback while keeping it inert. 30/30 `ep_optimization_test` regression passed.

## 2026-08-10T20:16:00+0000 — EP plugin-export inventory

- Produced `docs/ep-plugin/EP_PLUGIN_EXPORT_INVENTORY.md`: 2 production EPs (CPU NEAR, CUDA BLOCKED). Named 6 trait/ABI gaps.

## 2026-08-10T21:09:00+0000 — EP Plugin Compute Path

- Real `Compute` callback in `compute.rs`: reads inputs from OrtKernelContext, infers output shapes, allocates ORT outputs, executes kernels in topological order. 14 unit tests added.

## 2026-08-10 — EP Plugin Shape Inference + Fail-Closed Policy

- 22 ShapeInference variants. Fail-closed `Declined` replaces silent `SameAsInput(0)`. `SubgraphRouting` for multi-node fused subgraphs. 66 tests pass.

## 2026-08-10 — EP Device Lifetime Fix (BUG 1 + BUG 2)

- Root cause: use-after-free on `OrtMemoryInfo` + wrong legacy API. Fix: `CreateMemoryInfo_V2` + do not release on success. `conformance_multiple_run_calls` passes.

## 2026-08-10 — Clippy lint ep.rs:499 (manual_dangling_ptr)

- Replaced `1usize as *mut ort::OrtEp` with `std::ptr::dangling_mut()`. 82 unit tests, 21 conformance tests pass.

---

## Archive batch 2026-08-10 (ep-plugin-export/parity-cuda wave)

### 2026-08-10 — GetKernelRegistry + NEW-2 Compile cleanup
NEW-2: `ep_compile_inner` frees/nulls `out_infos[0..i]` on mid-loop failure. `GetKernelRegistry` infrastructure: full ORT 1.24+ kernel-registry machinery. `ExportedEp` holds optional `OrtKernelRegistry*` from `KernelRegistryEntry` slices. f16/bf16 blocker for Pris: `ExecutionProvider` trait lacks `op_entries()` iterator. 120 lib tests pass.

### 2026-08-10 — f16/bf16 kernel-registry entries wired end-to-end
Blocker resolved: CPU EP plugin passes `KernelRegistryEntry` slices to `create_ep_factories_with_registry`. `build_cpu_registry_with_descriptors` as inherent function (not trait method) avoids circular dep. f16/bf16 advertised for standard ML ops. 127 ep-plugin tests, 21 cpu-plugin tests all pass.

### 2026-08-10 — Dtype-aware GetCapability claim predicate
`ep_get_capability_inner` now applies `node_passes_dtype_filter()`. Single source of truth (same descriptors as GetKernelRegistry). Fail-closed: reject if no registry entry, Undefined dtype, or dtype not in set. 5 new unit tests. 132 ep-plugin, 23 cpu-plugin tests pass. Clippy clean.

### 2026-08-10 — needless_borrow clippy fixes in ep.rs test helper
Two `&` removed from `graph_with_node` test helper (`ep.rs:1041,1047`). Assertion sanity check: all five `node_passes_dtype_filter` sites verified non-vacuous. 132 passed. `conformance_add_float16` and `conformance_add_bfloat16` pass.


## Archive batch 2026-08-14 (Scribe decode-levers) — 2026-08-11 ep-plugin-parity-cuda wave + 2026-08-11/12 upstream-CI correction chronicle

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

## 2026-08-12 — Assigned Blocker 2 (CLASSIFY) of the CUDA-capture escalation
Branch `squad/decode-path-swa-classify`. Remove/correct the vestigial SWA
(sliding-window attention) classification on the native decode path so the decode
graph classifies correctly — precondition (with Batty's LOAD fix) for CUDA-graph
capture engaging. Part of Sebastian's 3-blocker escalation. Shared team goal:
**beat ORT 40 tok/s via CUDA-graph capture**. In progress.
