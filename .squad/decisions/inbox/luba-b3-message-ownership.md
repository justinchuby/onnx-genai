# B3 — NxrtStatus Message Ownership: Inline Buffer

**Date**: 2026-08-11
**Author**: Luba (ARM/edge, cross-platform ABI)
**Blocker**: B3 — cross-module allocator violation

## Decision

**Option 3 — Fixed inline buffer** was chosen.

`NxrtStatus.message` changed from `*mut c_char` (heap-allocated) to `[u8; 256]` with a `message_len: u32` field. The struct is now a pure value type — no heap allocation, no pointers, no `Drop` logic.

## Rationale

The nxrt ABI is a stable `cdylib` boundary. Plugin and host may be linked against different CRTs (especially on Windows). The original design allocated message memory in the plugin (`CString::into_raw`) and freed it in the host (`CString::from_raw` / `Drop`). This is undefined behaviour when the two sides use different allocators.

Option 1 (free callback) adds API surface and complexity. Option 2 (shared C allocator) relies on a fragile assumption. Option 3 eliminates the entire class of bug: there is nothing to allocate, nothing to free, nothing that can go wrong. The cost is struct size (264 bytes vs 16 bytes), which is negligible for a status return value.

Messages are truncated at 255 bytes. This is sufficient for diagnostic error messages (the only use case).

## ABI impact

- **`NxrtStatus` layout changed** — this is a major-version-level change within the current dev cycle. Since ABI major is still 1.0-dev and no external consumers exist, this is acceptable.
- **`struct_size` / version negotiation**: unchanged. The status struct is not versioned via `struct_size` (it's a return value, not a vtable).
- **`NXRT_CAP_KNOWN_MASK`**: unchanged.
- **`message_str()` is no longer `unsafe`** — a strict improvement.
- **`free_message()` removed** — no longer needed (nothing to free).
- **`Drop` impl removed** — pure value type.

## Ownership rule (for future implementers)

**`NxrtStatus` is a value type. It owns no heap memory. It can be copied, moved, or dropped without any cleanup.** This rule is documented in the module-level doc comment of `status.rs`.

## Also fixed

- **`c_char` portability**: `as *const i8` casts in `loader.rs` and `provider_adapter.rs` replaced with `as *const std::os::raw::c_char`. The original casts compile on x86-64 (where `c_char = i8`) but fail on aarch64 (where `c_char = u8`). Two instances remain in `tests/nxrt_abi_roundtrip.rs` — Chew owns those files.

## Unverified on Windows

This fix eliminates the cross-CRT allocator bug by design, but the actual Windows build has not been tested (this is a Linux environment). The struct layout and alignment should be verified on MSVC.
