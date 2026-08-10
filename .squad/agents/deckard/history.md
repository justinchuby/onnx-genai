# Deckard — History (compacted 2026-08-10T21:09:11Z)

**Role:** Systems developer on the Rust runtime, ORT2 loader/shape/IR/EPContext tracks, and CPU/CUDA execution performance. Preserve model-agnostic dispatch, fail-closed claims, checked arithmetic, byte-exact serialization, precision-sensitive tests, and reviewer-lockout ownership transfers.

## Durable lessons
- Repeated invariants: model-agnostic dispatch, fail closed at claim time, checked arithmetic, byte-exact serialization, and precision-sensitive tests.
- Parallel commit-producing work requires separate worktrees; reviewer rejection transfers ownership and must be recorded.
- Deckard owns canonical revisions after lockouts for shape inference, IR dtype, EPContext writer, and 2026-07-19 CPU reduction/activation dtype waves.
- CSA B5 initial five-output ratio-4 assembly misrouted to the ratio-128 kernel; Roy's ratio-keyed fix is canonical.
- CUDA token-index-10 drift root cause was SkipSimplifiedLayerNorm RMS FMA contraction; fix landed in `de3c556`/`ccf994c`.
- `cudarc` CUDA feature unification: ORT keeps CUDA 12.6 weak default, engine disables ORT defaults and selects CUDA 13.0 with `onnx-runtime-ep-cuda`.
- GridSample opset-16 rank-5 acceptance was rejected; Sapper's correction is canonical.
- Replay binding metadata caching gained only +0.23%; do not reattempt raw-address correctness-sensitive hot-path caching without stronger evidence.
- CUDA graph capture fixes require exact warmed signatures, persisted GQA scratch, handle ownership correctness, and replay metadata bounds.
- Fitted performance constants are acceptable only when labelled as fitted and bracketed by measured data.
- Public rewind/checkpoint APIs may use existing speculative helpers; public `fork_session` stays capability-gated.
- Runtime ORT selection order is machine-independent: explicit env vars, active conda/venv, target-cache fallback.
- **ORT plugin-EP ABI:** `OrtMemoryInfo` passed to `EpDevice_AddAllocatorInfo` must outlive the `OrtEpDevice` — ORT stores the raw pointer; do not call `ReleaseMemoryInfo` on success. Use `CreateMemoryInfo_V2`, not the legacy `CreateCpuMemoryInfo` (latter leaves `OrtMemoryInfoDeviceType`/`OrtDeviceMemoryType` uninitialized). Release only on failure.
- **Shape inference fail-closed:** `Declined` is the correct return for any unmodelled op; never fall back to `SameAsInput(0)`.
- **`validate_dims` must be wired** in the actual read path, not just implemented.

## Historical context

Pre-2026-08-10 entries moved to `history-archive.md`. Covers: kernel pre-binding (#Stage3), #594/#595/#602/#604 lockout revisions, mobius PR #449, EP plugin-export inventory, EP compute path implementation, shape inference overhaul (22 variants), EP device lifetime UAF fix, clippy lint cleanup.

## Current entries (wave: ep-plugin-export, 2026-08-10)

- Compute path implemented; 14 unit tests.
- 22-variant `ShapeInference` + `SubgraphRouting` for multi-node subgraphs; 66 tests.
- Device lifetime UAF fix in `factory.rs` (`CreateMemoryInfo_V2` + no release on success). Clippy `manual_dangling_ptr` lint fixed.
- Post-merge advisory: `ep_compile_inner` partial-output cleanup on mid-loop failure (not yet resolved).

## 2026-08-10 — GetKernelRegistry + NEW-2 Compile cleanup

- **NEW-2 resolved**: `ep_compile_inner` now frees and nulls `out_infos[0..i]` on mid-loop failure. Safe under both "ORT frees on failure" (null → no-op) and "ORT doesn't free" (we freed → no leak). Header lines 2179/2203–2207 do not specify failure-path ownership.
- **GetKernelRegistry infrastructure**: Implemented full ORT 1.24+ kernel-registry-based type-constraint machinery. `ExportedEp` now optionally holds an `OrtKernelRegistry*` built from `KernelRegistryEntry` slices. GetKernelRegistry callback wired. Coexists with Compile path (not mutually exclusive per header line 1522).
- **Blocker for f16/bf16**: `ExecutionProvider` trait lacks `op_entries()` iterator; CPU EP plugin must pass entries to `create_ep_factories_with_registry`. Pris: once entries are wired, re-test f16/bf16 routing.
- 120 lib tests pass (4 new: cleanup, dtype mapping, entry construction, no-host-api guard). 15 conformance tests pass. No regression.
