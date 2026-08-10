# Holden — EP Plugin Milestone 2 Security Audit

**Date:** 2026-08-10T23:09:23Z
**Branch:** `squad/ep-plugin-parity-cuda`
**Commits:** `2da0c4e7f`, `577047a74`

## Verdict: 🟡 YELLOW — May ship

Milestone 2 adds ~600 lines of new device-side FFI (`device.rs`), generalized device enumeration (`factory.rs`), kernel registry (`ep.rs`), and recording registry (`kernels/mod.rs`). One MEDIUM resource-leak finding; no memory-safety or host-corruption issues.

## Findings

| ID | Severity | Finding | Owner | Fix |
|----|----------|---------|-------|-----|
| M2-1 | MEDIUM | EP instance leaked in `stream_release` — `Box::into_raw`'d EP never freed | Nabil (author) → **Leon** (fixer) | Add `Box::from_raw(stream.ep)` in `stream_release`, matching `factory_release_allocator` pattern |
| M2-2 | LOW | Misleading doc comment on `DeviceAllocator::memory_info` says "owned; freed on drop" but it's borrowed from ORT | Nabil → **Leon** | Correct doc comment |

## Prior Advisory Disposition

| Advisory | Status |
|----------|--------|
| NEW-1 (`compute_release_state` no `catch_unwind`) | **RESOLVED** by Leon — `compute.rs:1567` |
| NEW-2 (partial info leak in `ep_compile_inner`) | **RESOLVED** by Deckard — `cleanup_partial_infos` free-and-null |

## Key Audit Conclusions

- **`#[repr(C)]` layout:** Correct for `DeviceAllocator`, `DeviceSyncStream`. First-field vtable at offset 0.
- **Panic safety:** All 14 new `extern "C"` callbacks verified guarded or trivially non-panicking.
- **`mem::forget` sites:** Confirmed no-ops (DeviceBuffer has no Drop). Test-only code.
- **RecordingOpRegistry:** Cannot over-advertise; unknown ops default to f32-only (fail-closed).
- **Allocator arithmetic:** No adapter-layer overflow; size passed through to EP.
- **Vtable ownership:** Allocator path correct (EP freed in release). Stream path leaks (M2-1).
