# #449 increment 3a — container-aware control flow (STATUS)

Branch `squad/449-container-inc3`, stacked on **#486** (inc2 seq mutations +
conversions, not yet merged). PR base = `squad/449-container-inc2`. refs #449.
CPU-only, pure shape-inference crate work.

## What landed (inc3a)
Container (`ValueType::Sequence/Optional/Map`) types now thread through **If**
branches and **Loop** carried dependencies:

- **If**: when both branches produce a container output, the If output type =
  `unify_value_type(then_out, else_out)` (recursive: Tensor⨝Tensor via the
  foundation's `unify_tensor_type`; Sequence/Optional recurse; Map requires an
  equal key dtype). Element extent disagreement degrades to a fresh symbol;
  element **dtype** disagreement is an error; one branch container + other
  branch tensor is an error ("one branch produces a container output while the
  other produces a tensor").
- **Loop**: a loop-carried operand that is a container **seeds** the body's
  formal input (via `body_container_seeds`), the body threads/mutates it
  (`SequenceInsert`, `Identity`, or passthrough), and the carried body output
  container type flows to the Loop output (remapped to the parent symbol space).
  Loop **scan** outputs stack tensors and are never containers — skipped by
  construction.

## Mechanism
- `InferenceReport` gains a `containers: HashMap<ValueId, ValueType>` field —
  **double duty**: it is the observation channel for tests AND the channel that
  threads a body's inferred container outputs to the parent CF node
  (`result.report.containers`).
- `infer_graph_scoped` gains a `seed_containers` param (injects body-input
  container types for Loop carried seqs) and raises the child interner floor
  above any symbol appearing only in a seeded container's element shape
  (`next_container_symbol`), so a body-local fresh symbol can't alias a
  parent symbol carried in via a container seed.
- New `CfOutput::Container(ValueType)` variant; `apply_cf_outputs` routes it
  into the parallel `containers` map (and removes any stale tensor entry).

## DRY confirmation
- The foundation's `merge_tensor`/`merge_shape` were **promoted** from
  `handlers/container.rs` to `context.rs` as `unify_tensor_type` /
  `merge_element_shape` (taking `&mut SymbolInterner`), so both the container
  handlers (`ctx.interner_mut()`) and `infer.rs` (its own `interner`) share one
  implementation. New recursive `unify_value_type` reuses `unify_tensor_type`
  at the leaves. Container symbol remapping (`map_container_to_parent`) mirrors
  the exact per-dim rule the tensor CF path uses (`map_body_shape` / the If
  merge): const→const, parent-origin symbol→pass through, else→fresh.

## Byte-identical tensor path
All container work is gated on a non-empty container map: `seed_containers`
and `report.containers` are empty for pure-tensor graphs, the container
read-backs miss, and **no extra `fresh_dim` is minted** on the tensor path, so
symbol numbering is unchanged. `tensor_only_path_is_byte_identical_after_
container_type_model` stays GREEN, as do all existing If/Loop/Scan tensor tests.

## Honest limitation (documented in design doc)
A **seeded** symbolic element dim crossing a Loop body is not in the body's
`child_to_parent` map (container seeds never touch a body IR `Value`), so it
degrades to a fresh parent symbol on the way out — SOUND but loses symbolic
identity. dtype/structure/concrete extents are preserved exactly. Covered by
`loop_passthrough_preserves_seeded_sequence_dtype`.

## Scoped OUT
- **inc3b — Scan container state vars**: mechanically identical to Loop carried
  seeding but rare in real models and with no scan-output container payoff.
- **inc4 — SequenceMap** (body maps over sequence elements, distinct element-
  typed body signature) **and cross-subgraph container capture** (If branches /
  Loop bodies reading an *outer-scope* container by name; `remap_node_io`
  currently sets `value_type: None`). Tests build sequences INSIDE the branch/
  body to avoid needing capture until inc4 lands it.

## Verification
- `cargo test -p onnx-runtime-shape-inference`: graph_inference **27 → 33**
  (+6 container CF tests); op_rules 275, container 16, doctest 1 — all GREEN;
  byte-identical regression GREEN.
- `cargo fmt --all --check` clean; `cargo clippy -p onnx-runtime-shape-inference
  --all-targets -D warnings` clean.
- `cargo build -p onnx-runtime-session -p onnx-runtime-eager` GREEN.
- Catalog counts **UNCHANGED at 217 ops / 262 entries** — inc3a registers no new
  ops (control-flow threading, not new operators).

## New tests
`if_unifies_matching_sequence_branch_outputs`,
`if_sequence_branch_extent_disagreement_degrades_to_symbol`,
`if_sequence_branch_dtype_disagreement_errors`,
`if_container_versus_tensor_branch_disagreement_errors`,
`loop_carries_sequence_accumulator_preserving_element_type`,
`loop_passthrough_preserves_seeded_sequence_dtype`.

## Roadmap
- inc1 (foundation, #477, MERGED): ValueType layer + SequenceEmpty/Construct/
  Length/At.
- inc2 (#486, in review): SequenceInsert/Erase, SplitToSequence,
  ConcatFromSequence (seq↔tensor conversion).
- **inc3a (this PR): If + Loop container threading.**
- inc3b (next): Scan container state vars.
- inc4: SequenceMap + cross-subgraph container capture.
