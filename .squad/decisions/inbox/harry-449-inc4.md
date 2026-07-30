# #449 increment 4 — closeout STATUS: SequenceMap + cross-subgraph capture + Scan state

Branch `squad/449-container-inc4`, stacked on **#527** (inc3a If/Loop container
threading). refs #449. CPU-only, pure `onnx-runtime-shape-inference` crate work.

This is the **closeout slice** of the #449 container-type arc
(foundation → seq ops → seq↔tensor → If/Loop control flow → **this**).

## What landed

1. **SequenceMap (ONNX opset 17)** — maps a subgraph body over sequence
   element(s). Handled in `infer_control_flow` alongside `If`/`Loop`/`Scan`
   (subgraph-bearing ops are NOT registry rules — the registry `InferenceFn` has
   no body access). Body formal input `i` is seeded from the per-element type of
   SequenceMap operand `i` (sequence → element type; extra tensor input → whole
   tensor each iteration). Each body output `j` is wrapped back as
   `Sequence<body_output_j>`. Multiple input sequences **zip** (falls out of
   positional per-operand seeding); additional non-sequence inputs are broadcast.

2. **Cross-subgraph container capture** — fixed `remap_node_io`, which previously
   set `value_type: None`, dropping the `ValueType` of any container value
   captured from an outer scope into a subgraph body. New `remap_container_type`
   helper threads the captured value's full `ValueType` (recursing, remapping the
   tensor-leaf element-shape dims via `remap_dim_expr` — mirrors the tensor path).
   `bind_captures` / `extend_visible_scope` gained a `containers` channel so a body
   that references an outer-scope sequence now sees its element type. This is what
   makes real nested-control-flow-with-sequences correct.

3. **Scan container state (folded in — previously deferred as inc3b)** — only
   Scan **state** vars (first `num_state` body inputs) can be containers;
   scan-inputs/scan-outputs stack/slice tensors and are never containers. Reuses
   the identical `seed_container_input` helper + `body_container_seeds` (now a
   `match` with `Loop`/`Scan`/`SequenceMap` arms). `infer_scan_outputs` checks
   `report.containers` for state slots first. ~a dozen lines given the inc3a
   machinery → folded in for completeness so **all four** subgraph ops
   (`If`/`Loop`/`Scan`/`SequenceMap`) are container-complete.

## DRY proof (composed from inc3a, not forked)

Every part re-points the inc3a body-threading driver:
- element **seeding**: `set_body_input` (tensor leaf) + `body_container_seeds` /
  `seed_container_input` (container leaf) — the same two seed channels Loop uses;
- output **read-back**: `read_body_output` + `map_container_to_parent` +
  `CfOutput::Container` + `apply_cf_outputs` — the same output-mapping Loop/If use;
- element **unification**: the foundation's `unify_value_type` / `unify_tensor_type`
  / `merge_element_shape` helpers (no per-op copy-paste).

## Byte-identical tensor path — PRESERVED

Every new path is gated on a container actually being present:
`body_container_seeds` returns empty for tensor graphs, container read-backs miss,
and `remap_container_type` is only invoked when `value_type.is_some()`. No extra
`fresh_dim` on tensor graphs → symbol numbering unchanged. The regression test
`tensor_only_path_is_byte_identical_after_container_type_model` stays **GREEN**.

## Catalog count — HONEST note (stays 217 ops / 262 entries, NOT 218)

SequenceMap is a **subgraph op handled in `infer_control_flow`**, not a registry
rule (like `If`/`Loop`/`Scan`, none of which are registry entries). There is no
218th registry handler, so bumping the pinned count
(`expanded_registry_catalog_count_is_pinned`) would be a **phantom declaration**
and would fail the pin. The count is correctly **left at 217/262**. (This
contradicts the task's anticipated "+1", but the +1 assumed a registry rule; the
honest/correct outcome is no bump.)

## Tests (all GREEN)

`graph_inference.rs` 33 → **39** (6 new):
- `sequence_map_identity_preserves_element_type`
- `sequence_map_body_transforms_element_type` (body `Shape`: `[2,3]` → int64 `[2]`)
- `sequence_map_zips_two_input_sequences` (body `Add`)
- `sequence_map_additional_tensor_input_is_broadcast`
- `if_branches_capture_outer_scope_sequence` (proves the `remap_node_io` fix — an
  outer-scope sequence resolves, with element type, inside both `If` branches)
- `scan_carries_sequence_state_variable` (body `SequenceErase`; element type
  preserved across the state carry)

Full verification: `cargo test -p onnx-runtime-shape-inference` all GREEN
(16 op_rules + 39 graph_inference + 275 unit + 1 doctest); `cargo fmt --all
--check` clean; `cargo clippy -p onnx-runtime-shape-inference --all-targets --
-D warnings` clean; `cargo build -p onnx-runtime-session -p onnx-runtime-eager`
GREEN.

## Recommendation: **CLOSE #449**

The **sequence** container-type arc — the container that actually appears in real
hybrid/seq models — is complete end to end:
foundation (ValueType layer) → all Sequence ops → seq↔tensor conversion
(`SplitToSequence`/`ConcatFromSequence`) → container-aware control flow
(`If`/`Loop`/`Scan`/`SequenceMap`) → cross-subgraph container capture.

The only remaining container surface is **Optional/Map op handlers**
(`Optional`/`OptionalHasElement`/`OptionalGetElement`/`ZipMap`). The
`ValueType::Optional`/`Map` representation already exists, but these ops are **not
implemented and not used by any in-tree target model** (only referenced in a
`handlers/ml.rs` comment excluding `ZipMap`, a `lib.rs` doc comment, and `.squad`
notes). This is a separate, smaller, lower-priority surface with **no current
real-model demand** — not a load-bearing gap for the sequence models #449 targets.

→ **Close #449.** Optional/Map op handlers = optional small follow-up, to be
opened only if/when a target model requires them.
