# Challenger Review v4 — PR #762 (EP Plugin Parity CUDA)

**Date:** 2026-08-12  
**Reviewer:** Challenger (4th independent Opus review)  
**Commits reviewed:** af45043fd (Nabil B1+B2), b906ab2bb (Batty completion)  
**Status:** Conditional PASS — ready to leave draft

---

## BLOCKING

None found.

## SUBSTANTIVE

### S1 — Canary tests simulate a different allocation size than the real code (compute.rs:2801-2862)

The canary tests allocate `numel × byte_size` (e.g. 8 × 2 = 16 bytes for f16). The real code allocates `numel × max(byte_size, 8)` (e.g. 8 × 8 = 64 bytes). The canaries prove the *minimum* sizing is safe, but the real code over-allocates 4×. A regression that replaces `max(byte_size, 8)` with just `byte_size` would still be safe (the canaries prove it), but the canaries can never catch a wrong-dtype write at the 8-byte level because the over-allocation absorbs it. **The canaries are honest but test a tighter constraint than the code provides — they're regression guards for the old bug, not for new wrong-dtype writes.**

**Recommended action:** Add a comment in the canary tests acknowledging they test minimum byte_size, not the max(byte_size, 8) the runtime uses.

### S2 — `mark_absent()` is advisory, not enforced (tensor.rs:287-294)

`TensorMut::mark_absent()` sets a flag; nothing prevents a kernel from writing the wrong dtype to a scratch output. The over-allocation (`max(byte_size, 8)`) provides a safety net up to 8 bytes/element — sufficient for f64 — but if a future dtype > 8 bytes is added, or a kernel writes wider-than-element data, the over-allocation won't save it. This is a known design tradeoff, not a blocker, but should be documented.

### S3 — Phantom intermediate buffers in routed path (ep.rs:711-726)

`build_subgraph_routing` assigns `NodeOutputSink::Buffer(buf_idx)` to absent outputs (they're not in `output_index`). These phantom buffer indices inflate `num_intermediate_buffers` and allocate `Option<IntermediateBuf>` slots that are never written. The runtime correctly skips them via `absent_output_slots.contains()`, so this is harmless but wasteful. A `NodeOutputSink::Absent` variant would make the intent explicit and avoid wasted allocations.

## NITS

### N1 — `memoffset_of_create_ep()` manually computes layout (provider_adapter.rs:151-157)

Uses `size_of::<u32>() * 2 + size_of::<*const u8>() + size_of::<fn()>()`. This is correct for the current `#[repr(C)]` layout but fragile if fields are reordered. Use `std::mem::offset_of!(NxrtEpFactoryVtable, create_ep) + std::mem::size_of::<T>()` (stabilized Rust 1.77).

### N2 — Unnecessary transmutes in routed path (compute.rs:930-935)

`std::mem::transmute::<&[usize], &[usize]>` is a no-op transmute. These were likely lifetime-extension attempts, but `absent_shapes` and `absent_strides_storage` already outlive the `all_output_views` collection.

---

## DETAILED FINDINGS

### Is the heap overflow provably gone?

**Yes.** Traced both paths (single-kernel fast path compute.rs:995-1016 and routed multi-node path compute.rs:825-842):
- Dtype is derived from `entry.output_dtypes[out_slot]`, never hardcoded
- `Undefined` dtype → fail closed with error return
- Buffer sized `numel × max(byte_size, 8)` → at least 8 bytes per element
- `TensorMut` constructed with `scratch_dtype`, not `Float32`

For f16/bf16 (byte_size=2): buffer is 8 bytes/element, kernel writes 2 bytes/element. For f32 (4 bytes): buffer is 8 bytes. For f64 (8 bytes): buffer is 8 bytes exactly. No overflow possible for any current dtype.

### Would the canaries catch a regression?

Partially. They would catch a regression to the old hardcoded-Float32 dtype. They would NOT catch a wrong-dtype write masked by the over-allocation (see S1).

### Is RoutedSlotKind positionally sound?

**Yes.** Verified:
- `slot_kinds` is built in `output_shapes.iter().enumerate()` order — one entry per slot
- Absent slots → `RoutedSlotKind::Absent(idx)`, non-absent → `Ort` or `Buffer`
- `sinks` has entries for ALL positions (built from `node_outputs` in ep.rs:712-726, which includes absent outputs)
- Reconstruction loop iterates `slot_kinds` by index, pairing `ort_iter`/`buf_iter` drains correctly
- Interior absent (slots 1,2 of SkipLayerNorm's 4 outputs), trailing, and multiple-absent all handled

### Does `end_version = i32::MAX` over-claim?

**No.** The ep.rs test entries use `end_version: 21` (test fixtures only). The real cpu-plugin (lib.rs:33) uses `i32::MAX`, matching ORT's own pattern for ops like Add/Sub/Mul whose schema changes are backward-compatible. The old bug (`end_version: since`) under-claimed; `i32::MAX` is the standard fix.

### Is the assignment assertion non-vacuous?

**Yes.** `optional_slots.rs:1169` asserts `["Add", "SkipLayerNormalization", "Mul"]` are assigned to `"cpu_ep"` (our plugin EP), not ORT's built-in CPUExecutionProvider. `disable_cpu_ep_fallback=1` ensures ORT errors if our EP declines a node. If our EP declined any of these, the session creation would fail.

### Did I find a fifth absent-slot defect?

**No standalone fifth defect found.** The closest candidate (S3 — phantom buffer indices) is wasteful but not incorrect. The `absent_output_slots` → runtime skip → `sinks` indexing chain is sound for all observed cases.

The absent-slot machinery now has four distinct defense layers:
1. Out-of-band `HashSet<ValueId>` (not name-based sentinels)
2. Dtype from `output_dtypes[slot]` (not hardcoded Float32)
3. `RoutedSlotKind`/`SlotKind` enum preserving positional 1:1 mapping
4. `TensorMut::mark_absent()` flag for kernel-side detection

### Sanitizer coverage honesty

**Honest.** Canary tests cover pure-Rust buffer sizing logic. Miri covers Rust-only memory safety. Neither crosses FFI into ORT, which is acknowledged. The ORT e2e tests (optional_slots.rs) provide functional coverage of the real integration path without sanitizers. ASAN was not attempted and not claimed.

---

## VERDICT

**Ready to leave draft.** No blocking defects. The heap overflow is provably gone. The three substantive issues are design documentation improvements, not safety issues.

## Verified myself vs. taken on trust

**Verified myself:**
- Traced scratch allocation → TensorMut construction for both compute paths (single-kernel and routed)
- Verified `RoutedSlotKind` construction and consumption positional correctness
- Verified `end_version` values in ep.rs tests (21) vs cpu-plugin production code (i32::MAX)
- Verified struct_size guards cover all field accesses in loader.rs and provider_adapter.rs
- Verified assignment assertion targets "cpu_ep" and includes Add+SkipLayerNormalization+Mul
- Verified canary test allocation size mismatch (byte_size vs max(byte_size,8))
- Ran `cargo test --no-fail-fast` for all 5 EP crates: 86 passed, 0 failed
- Ran `cargo clippy -D warnings`: clean

**Taken on trust:**
- Miri pass (not re-run; claims are plausible given the tests are pure-Rust)
- ORT e2e tests (`optional_slots.rs`, `plugin_ort_e2e.rs`) require a real ORT library; results from prior reviewer's run accepted
- The 16-directory fail-loud gate test (prior reviewer verified)
