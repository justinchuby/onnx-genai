# PR #762 — S1/S2/S3 resolution (Sebastian, 2026-08-12)

## S1: Canary tests now mirror production allocation

**Problem:** Canary tests allocated `numel × byte_size` (16 bytes for 8 f16 elems) while production uses `numel × max(byte_size, 8)` (64 bytes). A wrong-dtype write (e.g. Float32 into f16 slot) would be absorbed by production's padding and go undetected by the old canaries.

**Fix:** Extracted `production_scratch_alloc(numel, dtype)` helper mirroring compute.rs exactly. All canary tests use it. Added two new tests:
- `scratch_buffer_wider_write_absorbed_by_padding`: proves 4-byte writes into f16 slots are absorbed (by design).
- `scratch_buffer_detects_oversized_write`: proves 16-byte writes overflow the production allocation — the canary catches it.

**Can the canaries still detect the pre-fix Float32-hardcoded regression?** No — because production now over-allocates to `max(byte_size, 8)`, a Float32 (4 bytes) write fits. This is intentional: the fix was never "allocate exactly byte_size" but rather "allocate generously and carry the correct dtype". The canaries now test the actual production contract, not the old bug's narrow failure mode. The `scratch_dtype_matches_absent_slot_dtype` test still directly verifies the dtype is never hardcoded Float32.

## S2: `mark_absent()` — documented + enforceable, not restructured

**Problem:** `absent` flag was advisory; nothing validated write dtype against declared dtype.

**Resolution:** Added `TensorMut::validate_write_dtype(write_dtype)` that:
- For present outputs: requires exact dtype match.
- For absent outputs: rejects writes exceeding `max(declared_byte_size, 8)` bytes/elem.

Full automatic enforcement on every raw-pointer write would require restructuring the kernel API (kernels write through `data_ptr_mut<T>()`). Instead: the invariant is loudly documented on `mark_absent()`, and `validate_write_dtype()` is available for kernel harnesses and test assertions. This is "make violation loud rather than silent" — the method exists, the contract is documented, and test code can call it.

## S3: `NodeOutputSink::Absent` — landed

Added `NodeOutputSink::Absent` variant mirroring `NodeInputSource::Absent`. `build_subgraph_routing` now checks `absent_outputs` and assigns `Absent` instead of allocating phantom `Buffer` slots. `num_intermediate_buffers` no longer inflated. The compute path's match on `NodeOutputSink` handles `Absent` defensively (returns error if reached, since absent slots are handled earlier via `absent_output_slots`).

## Nits

- Removed 4 no-op `transmute::<&[T], &[T]>` calls (identity lifetime coercions) from both compute paths.
- `memoffset_of_create_ep` — not found in current code; skipped.

## Validation

- 280 tests passed, 0 failed (278 baseline + 2 new canary tests).
- Clippy clean (`-D warnings`).
- `cargo fmt --check` clean.
- Miri: 4 canary tests pass clean. Pre-existing Miri failure in ep-api (`load_legacy_rejects_an_incompatible_plugin_abi`) from filesystem ops in test fixture — unrelated.
- ASAN: not attempted (requires instrumented build with nightly + `-Zsanitizer=address`; the Miri coverage on pure-Rust paths is stronger for these canaries).
