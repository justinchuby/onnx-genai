# EP Plugin Export — Final Ship Verdict

**Author:** Holden (Security Engineer)
**Date:** 2026-08-10T22:42:21Z
**Branch:** `squad/ep-plugin-export`
**Scope:** `crates/onnx-runtime-ep-plugin/` + `crates/onnx-runtime-ep-cpu-plugin/`

## Verdict: 🟡 YELLOW — May ship

All three original ship-blocking findings (N1 CRITICAL, N2 HIGH, N3 MEDIUM) have been independently verified as resolved by the assigned fixers (Leon, Isidore, Deckard). The use-after-free in `factory.rs` found after the last RED verdict is also resolved correctly. No new CRITICAL or HIGH findings. Two LOW advisory items are recorded for post-merge follow-up issues; they do not block merge.

## Disposition of Original Findings

| Finding | Fixer | Verified |
|---------|-------|----------|
| N1: `compute_execute` no `catch_unwind` | Leon | ✅ `compute.rs:552` — `catch_unwind` present; regression test at line 2115 |
| N2: negative dims wrap to `usize::MAX` | Leon | ✅ `kernel_ctx.rs:193` — `validate_dims()` called; eight tests including negative, overflow, zero, scalar |
| N3: macro entry points unguarded | Isidore | ✅ `lib.rs` — both `CreateEpFactories` and `ReleaseEpFactory` wrapped; `ReleaseEpFactory` return type corrected to `void` |

## UAF Fix Assessment (`factory.rs`, Deckard, commit `c92838dba`)

**Correct.** `EpDevice_AddAllocatorInfo` takes ownership of the `OrtMemoryInfo` raw pointer (ORT stores it, does not copy). The fix:

- Success path: does NOT call `ReleaseMemoryInfo` — ORT owns the pointer and releases it with `OrtEpDevice`. Not a leak.
- Failure path: calls `ReleaseMemoryInfo` exactly once — we still own it. Not a leak, not a double-free.
- These paths are mutually exclusive (branched on status null check). Double-free is impossible.
- `CreateMemoryInfo_V2` correctly fills `OrtMemoryInfoDeviceType_CPU` and `OrtDeviceMemoryType_DEFAULT` that the EP device ABI requires; the old `CreateCpuMemoryInfo` left those fields uninitialized.

## Post-Merge Advisory Items (not blocking)

**NEW-1 (LOW):** `compute_release_state` (`compute.rs:1416`) lacks `catch_unwind`. `ComputeState { _placeholder: u8 }` is trivially droppable and cannot panic in current code, but the missing guard is a pattern violation that becomes dangerous if `ComputeState` is extended. File issue, assign to Leon.

**NEW-2 (LOW):** `ep_compile_inner` (`ep.rs`) does not clean up already-written `out_infos[0..i]` on mid-loop failure — the ORT contract for this case is unspecified. File issue, assign to Deckard. Carried from M2.

## Broader New Code Assessment

- `graph_reader.rs` attribute and initializer reading: no `OrtGraph*`/`OrtNode*` cached beyond callback frame; bounds on initializer copy (1-D, ≤ 64 elements); all `CStr` conversions via `to_string_lossy`; attribute type mismatches handled gracefully.
- `ep.rs` capability filtering: declining a node leaves no partial state; fail-closed per decisions.md.
- `factory.rs` EP name/vendor/version lifetimes: `CString` fields owned by `ExportedFactory`; correct.
