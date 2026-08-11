# Decision: S1–S3 Optional-Slot Liveness Proof

**Date:** 2026-08-11
**Author:** Mariette
**PR:** #762

## S1: The EP Was Declining Optional-Slot Nodes

**Finding:** With `session.disable_cpu_ep_fallback=1`, both `skip_layer_norm_output_sum_position` and `clip_omitted_min_with_max` failed at `CreateSession` — confirming our EP never executed these nodes. ORT's built-in CPU EP was handling them via fallback.

**Root causes (three separate issues):**

1. **Claim filter** (`ep.rs:275`): rejected any node whose outputs include `DataType::Undefined`, but absent optional outputs are created with that dtype by graph_reader. Fix: recognize `__absent_output_*` sentinel names as intentional absence.

2. **Dtype filter** (`node_passes_dtype_filter`): same Undefined rejection for outputs. Same fix applied.

3. **Shape inference gap**: `Clip` was missing from the shape inference op lists, causing the shape-inference filter to decline Clip nodes even though a kernel existed. Fix: added Clip to `SameAsInput(0)`.

4. **Single-kernel fast path** (`compute.rs:888`): passed ORT inputs directly to kernels without injecting absent sentinels. For `Clip(X, "", max)`, ORT provides [X, max] but the kernel expects [X, absent, max]. Fix: use `input_slots` mapping to reconstruct positional inputs.

## S2: Off-by-one in axis bounds

`resolved > rank` → `resolved >= rank`. Valid axis indices are `0..rank-1`; axis == rank is OOB.

## S3: Scratch buffer sizing

Replaced hardcoded `numel * 4` with `numel * primary_output.byte_size()`, falling back to 8 bytes. Prevents under-allocation for f64/i64 kernels and over-allocation for f16/bf16.

## Evidence

| Test | Pre-fix (no fallback guard) | Pre-fix + fallback=1 | Post-fix + fallback=1 |
|------|---------------------------|---------------------|---------------------|
| skip_layer_norm_output_sum_position | PASS (vacuous) | FAIL (EP declines) | PASS (EP executes) |
| clip_omitted_min_with_max | PASS (vacuous) | FAIL (EP declines) | PASS (EP executes) |
| layer_norm_axis_eq_rank | no test existed | N/A | PASS (rejects axis==rank) |
| layer_norm_axis_eq_rank_minus_one | no test existed | N/A | PASS (accepts axis==rank-1) |
