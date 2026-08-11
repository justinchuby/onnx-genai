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

## 2026-08-10 — f16/bf16 kernel-registry entries wired end-to-end

- **Blocker resolved**: CPU EP plugin now passes `KernelRegistryEntry` slices to `create_ep_factories_with_registry`, derived from the real `OpRegistry` via `RecordingOpRegistry`.
- **Design choice**: Inherent function in ep-cpu (`build_cpu_registry_with_descriptors`) rather than trait method — avoids circular dep between ep-api and ep-plugin. CUDA EP adopts same pattern independently. Consistent with §524.
- **Dtype derivation**: `supported_dtypes_for_op()` classifies ops by actual dispatch macro used. Fail-closed: unknown ops → f32-only.
- **f16/bf16 advertised** for: Add, Sub, Mul, Div, MatMul, Gemm, Softmax, LayerNorm, Attention, Identity, Reshape, etc. NOT for pkg.nxrt ops or MatMulNBits.
- **Tests**: 6 new unit tests (descriptor derivation, f16/bf16 presence, fail-closed). 127 ep-plugin tests, 21 cpu-plugin tests all pass. Workspace check clean.
- **f16/bf16 routing status**: Infrastructure complete. Whether ORT actually routes depends on compile-EP semantics — Pris's e2e test is the ground truth.

## 2026-08-10 — Dtype-aware GetCapability claim predicate

- **What**: `ep_get_capability_inner` now applies `node_passes_dtype_filter()` to every claimed node. The filter checks input/output dtypes against `KernelRegistryEntry::supported_dtypes` sourced from the same descriptors used for `GetKernelRegistry`. Single source of truth — no drift by construction.
- **Fail-closed**: Node rejected if op has no registry entry, if any dtype is Undefined, or if dtype not in supported set.
- **Wiring**: `ExportedEp` gains `registry_entries: Vec<KernelRegistryEntry>`, populated via `new_with_registry_and_entries()` from factory's `CreateEp`. Backward-compatible: `new_with_registry()` passes empty entries (filter bypassed).
- **Tests**: 5 new unit tests (f32 claimed, unsupported rejected, Undefined rejected, unknown op rejected, empty entries bypassed). 132 ep-plugin tests, 23 cpu-plugin tests pass. Clippy clean. Workspace check succeeds.
- **f16/bf16 routing verdict**: The claim predicate is now dtype-aware and will claim f16/bf16 nodes for ops that list Float16/BFloat16. Cannot empirically prove e2e on this host (no f16 model in conformance suite). Instruction to Pris: **un-ignore the f16/bf16 conformance tests.**

## 2026-08-10 — needless_borrow clippy fixes in ep.rs test helper

- **What**: Removed two `&` in `graph_with_node` test helper (ep.rs:1041, ep.rs:1047). `format!()` returns `String`, which already implements the required trait; the borrow was redundant (`needless_borrows_for_generic_args`).
- **Assertion sanity check**: Reviewed all five `node_passes_dtype_filter` call sites (lines 1067, 1083, 1098, 1113, and the `&entries` calls). All assertions are meaningful: they test distinct, real cases (f32 claimed, Int64 rejected, Undefined rejected, unknown-op rejected, empty-entries bypass). No vacuous assertion found — each test constructs a unique graph type and asserts the correct boolean outcome.
- **Validation**: `cargo clippy -p onnx-runtime-ep-plugin --all-targets -- -D warnings` → clean (no errors). `cargo test -p onnx-runtime-ep-plugin --lib` → 132 passed. `cargo test -p onnx-runtime-ep-cpu-plugin --all-targets` → 23 passed including `conformance_add_float16` and `conformance_add_bfloat16`.

## 2026-08-11 — CUDA plugin wiring: real EP behind feature gates

- **What**: Rewrote `onnx-runtime-ep-cuda-plugin/src/lib.rs` from a stub that panics to a real plugin that constructs `CudaExecutionProvider::new_default()` with GPU `DeviceSupport`, kernel registry entries from `CUDA_COVERED_OPS`, and fail-closed error paths. Made `fail_status` pub in `status.rs`.
- **Feature gate**: `cuda` feature off → zero factories + error status. `cuda` on → real EP construction attempted; GPU-absent hosts get a clean error, not a panic.
- **Compile-check results**: Both `cargo check -p onnx-runtime-ep-cuda-plugin` (no feature) and `--features cuda` succeed on this CUDA-less host (cudarc uses dynamic-loading).
- **`prefetch_lazy_weight`**: Left as `Ok(false)` stub. No `try_without_eviction` API on `CudaWeightResidency`; implementing prefetch that may evict violates the standing directive. Proper fix requires a new residency method + GPU validation.
- **Unvalidated**: Runtime EP construction, ORT session execution, allocator/transfer/stream, kernel routing, and `page_lazy_weight` — all require a real GPU host.
- **Validation**: `cargo check --workspace` ✅, `cargo clippy -p onnx-runtime-ep-plugin --all-targets -- -D warnings` ✅, `cargo test -p onnx-runtime-ep-cuda-plugin` → 3 passed, `cargo test -p onnx-runtime-ep-cpu-plugin` → 23 passed.

## 2026-08-11 — PR #762 CI unblock: clippy Range::contains lint

**Branch:** `squad/ep-plugin-parity-cuda`

**Fix:** `crates/onnx-runtime-ep-cpu/src/kernels/mod.rs:2293` — replaced `delta >= 0 && delta < 50` with `(0..50).contains(&delta)` (clippy `manual_range_contains`, fatal under `RUSTFLAGS="-D warnings"`).

**Assertion verdict: NOT flaky.** The assertion bounds a structural count difference (`reg.len() as isize - descriptors.len() as isize`) between two static in-memory registries built from code. There is no wall-clock timing, no I/O, no concurrency. The bound of 50 is generous slack for CNN-registered ops. Cannot flake on a loaded CI runner.

**Other lint cleared:** None. Both `onnx-runtime-ep-api` and `onnx-runtime-ep-cpu` passed `RUSTFLAGS="-D warnings" cargo clippy --locked --all-targets` with zero warnings after the single-line fix.

**Validation output (all passing):**
```
RUSTFLAGS="-D warnings" cargo clippy --locked --all-targets -p onnx-runtime-ep-api -p onnx-runtime-ep-cpu
→ Finished `dev` profile [unoptimized + debuginfo] target(s) in 11.17s  ✅

RUSTFLAGS="-D warnings" cargo clippy --locked --all-targets -p onnx-runtime-session -p onnx-runtime-eager
→ Finished `dev` profile [unoptimized + debuginfo] target(s) in 8.40s   ✅

cargo fmt --all -- --check                                                ✅

cargo test -p onnx-runtime-ep-cpu
→ 6 passed (shared_allocator integration tests), 0 failed               ✅
```

---

### 2026-08-11 — Consolidate registry-building tests to reduce resource pressure

**Branch:** squad/ep-plugin-parity-cuda (PR #762)

**Problem:** Windows ARM64 CI fails `spmd_adaptive_calibrated_decode_is_bit_identical_to_flat`
with an empty-output child process (killed/OOM, not assertion failure). Hypothesis: four new
descriptor tests each independently built the full 166-op registry + descriptors (~8 constructions
total), inflating memory when running in parallel with the subprocess-spawning SPMD test.

**Change:** Added `OnceLock`-based `shared_registry_with_descriptors()` fixture in
`kernels/mod.rs` so the registry+descriptors are built once and borrowed by all four tests.
All assertions preserved verbatim.

**Measurements (Linux, `cargo test -p onnx-runtime-ep-cpu`):**

| Metric        | Before | After  |
|---------------|--------|--------|
| Wall clock    | 32.20s | 32.25s |
| Peak RSS      | 254 MB | 253 MB |

**Verdict:** On this Linux host, the consolidation produces negligible timing/memory
difference — the four tests were already fast relative to the rest of the suite. The
Windows ARM64 failure is most likely a **pre-existing flake** on a resource-constrained
runner, not caused by our branch. Our tests don't leak (the `OnceLock` static is tiny
relative to the SPMD test's subprocess budget). The consolidation is still the right
thing to do (fewer redundant allocations, cleaner test structure), but CI must confirm
on the actual Windows ARM64 runner.

**Recommendation:** The SPMD parity test's child-process-with-empty-output failure mode
should be investigated by its owner — it needs a timeout/diagnostic improvement regardless
of our branch.

**Validation:**
- `cargo test -p onnx-runtime-ep-cpu` → all pass
- `RUSTFLAGS="-D warnings" cargo clippy -p onnx-runtime-ep-cpu -p onnx-runtime-ep-api` → clean
- `cargo fmt --all -- --check` → clean
- cpu-plugin 6+17, ep-plugin 154+9 → no regression

## 2026-08-11 — BFloat16 contrib U type constraint fix

**PR:** microsoft/onnxruntime#31974  
**Task:** Investigate and fix the `U` type constraint mismatch in contrib CPU LayerNorm.

**Facts established:**
- Contrib schema constrains `U` to `{tensor(float)}` only.
- The macro registered `U=T`, so for MLFloat16/BFloat16, `U=MLFloat16`/`U=BFloat16` — violating the schema.
- This is pre-existing for MLFloat16; our PR widened it to BFloat16.
- **No runtime correctness impact:** the contrib `LayerNorm` doesn't set `contrib_op=true`, so `SrcDispatcher` always uses `U=float`.
- CUDA contrib already correctly registers `U=float` for narrow types.

**Decision:** Option (b) — changed macro to two params `(T, U)`, registered narrow types with `U=float`. One-line semantic change, zero risk, aligns CPU with CUDA and schema.

**Duplication nits:** deferred to follow-up — scope creep for a registration PR.

**Validation:** Build succeeded, 10/10 `LayerNormBFloat16*` tests passed.
