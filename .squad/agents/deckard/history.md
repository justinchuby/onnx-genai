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
