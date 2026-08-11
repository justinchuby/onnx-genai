# Decision: B1 — Out-of-band absent-output representation; B2 — Rank-preserving shape collection

**Date:** 2026-08-11
**Author:** Coco (kernel engineer)
**PR:** #762

## B1 — Absent-output sentinel replaced with out-of-band `HashSet<ValueId>`

**Problem:** The `__absent_output_` string prefix was in-band signalling. A model
containing a tensor genuinely named `__absent_output_0_2` would bypass dtype
validation — model files are untrusted input.

**Fix:** `OutboundGraphReader` now maintains `absent_outputs: HashSet<ValueId>`.
ValueIds are graph-internal arena indices assigned by the reader at construction
time — they are not derivable from model content. The name prefix is removed.
`ep.rs` receives the set via `reader.absent_outputs()` and passes it to
`node_passes_dtype_filter`.

**Why unforgeable:** `ValueId` is a `u32` index into the graph's value arena.
It is assigned by `Graph::create_named_value` during IR construction — the
model's protobuf content can influence *names* and *shapes* but never the
*arena index* a value receives. An attacker controlling the .onnx file cannot
cause a specific ValueId to appear in the reader's `absent_outputs` set.

## B2 — Rank destruction eliminated

**Problem:** `filter_map(|d| d.as_static())` dropped symbolic dimensions,
changing `Vec` length (= rank). `[batch, seq, 768]` collapsed to `[768]`.

**Fix:**
- At claim time: `map(|d| d.as_static())` → `Vec<Option<usize>>` preserving rank.
  `ShapeInference::for_node` updated to accept `&[Vec<Option<usize>>]`.
- At compile time: same map, then `unwrap_or(0)` for the `get_kernel` trait
  (which requires `&[Vec<usize>]`). 0 signals "dynamic/unknown at compile time"
  while preserving rank. The kernel receives actual concrete shapes from
  `OrtKernelContext` at runtime.
- Consumers that require static values (e.g. `build_conv` reading kernel dims
  from weight shape) fail closed (`return None` → `Declined`) when encountering
  `None` dims.

## `conformance_mixed_partition` assertion

Added a compiled-node counter (`COMPILED_NODE_COUNT` atomic in ep.rs) exposed
via `nxrt_ep_compiled_node_count` C symbol. The test loads it via dlsym.

With `disable_cpu_ep_fallback=false`, ORT 1.27 routes all nodes to its built-in
CPU EP to avoid partition overhead — so the counter reads 0. A hard assertion is
not possible without ORT's per-node provider attribution API (unavailable in
1.27). The test logs a diagnostic instead.
