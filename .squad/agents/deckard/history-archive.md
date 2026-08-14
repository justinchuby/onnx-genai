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
