# #449 increment 3 — container-aware control flow (DESIGN)

Branch `squad/449-container-inc3`, stacked on **#486** (inc2). refs #449. CPU-only.

## Problem statement
The existing control-flow shape inference (If/Loop/Scan, the tensor-only slice
from #362/#448) threads **tensor** types into/out of subgraph bodies by
mutating the body's IR `Value` (dtype+shape). Container types (`ValueType::
Sequence/Optional/Map`) live only in a **transient `containers: HashMap<ValueId,
ValueType>`** map inside `infer_graph_scoped`, which is dropped when the scope
returns (`ScopedInference` carries only `report` + `parent_symbols`). The IR
`Value` has no container representation (`crates/onnx-runtime-ir/src/value.rs`
is dtype+shape only). So today a sequence cannot cross a subgraph boundary — see
the documented follow-up at `infer.rs` `remap_node_io` (`value_type: None`).

## Where a subgraph i/o carries a CONTAINER type (enumeration)
- **If**: `then_branch` / `else_branch` **outputs** may be sequences (a branch
  builds a sequence internally, e.g. `SequenceConstruct`, and returns it). The If
  output type = **unify(then_out, else_out)**. Branches take no formal inputs, so
  there is no container *input* seeding — only output reconciliation. This is the
  smallest, cleanest slice.
- **Loop**: loop-**carried** dependencies can be sequences (`v_initial → body →
  v_final`): the carried operand seeds the body formal input as a container, the
  body threads/mutates it (`SequenceInsert`/`Identity`), and the carried body
  output flows to the Loop output. Requires container **input seeding** *and*
  container **output read-back**. Loop *scan outputs* stack per-iteration tensors
  along a new axis and therefore **cannot** be sequences — out of scope by
  construction.
- **Scan**: scan-inputs are per-iteration tensor slices; scan-outputs are stacked
  tensors — neither can be a container. Only **state** variables could be
  containers, mechanically identical to Loop carried-dep seeding but rare in real
  models and with no scan-output container payoff. **Deferred (inc3b).**
- **SequenceMap**: maps a subgraph *body* over each sequence element; output is a
  sequence whose element type = the body's per-element output type. Distinct body
  signature (element-typed formal input, not a generic CF reconciliation).
  **Deferred (inc4)** — lands naturally once the CF container plumbing here exists.

## Honest split
- **inc3a (this PR): If + Loop container threading** — the two headline cases the
  roadmap calls out (an If returning a sequence from both branches with unify; a
  Loop carrying a sequence accumulator, element type preserved across the body).
- **inc3b: Scan container state variables** (same seeding machinery, limited payoff).
- **inc4: SequenceMap** + cross-subgraph container **capture** (the `remap_node_io`
  `value_type: None` follow-up, i.e. a branch returning an *outer* captured
  sequence). Not needed here because test branches build the sequence internally.

## Mechanism (inc3a)
1. **Expose + thread body containers.** Add `containers: HashMap<ValueId,
   ValueType>` to `InferenceReport` (additive; empty for tensor graphs, so the
   report is unchanged for the tensor path). `infer_graph_scoped` fills it from
   its final `containers` map. This does double duty: (a) whole-graph tests can
   observe container outputs; (b) a parent CF node reads a body output's
   `ValueType` via `result.report.containers`.
2. **Seed body-input containers.** `infer_graph_scoped` gains a `seed_containers:
   HashMap<ValueId, ValueType>` param (empty at top level → byte-identical). It
   pre-populates the child `containers` map before body inference so body ops
   (e.g. `SequenceInsert` on the carried input) see the sequence element type.
   `infer_child_subgraphs` builds the seed from the node's **container operands**
   using the same positional logic as `seed_loop_body` (carried operand `2+i` →
   body input `2+i`). Requires threading the outer `containers` map into
   `infer_child_subgraphs`.
3. **Symbol-collision safety.** Seeded container element-shape symbols are
   parent-space `SymbolId`s that never touch a body `Value`, so
   `seed_next_symbol(body)` would not see them and the child interner could mint a
   colliding fresh symbol. Fix: raise the child interner floor above every symbol
   appearing in `seed_containers` element shapes.
4. **Read-back + symbol mapping.** In `infer_if_outputs` / `infer_loop_outputs`,
   *before* the tensor path, check `result.report.containers.get(&body_output)`.
   If present, remap the element-shape `DimExpr`s to the parent space with the
   **same per-dim rule the tensor path already uses**: constant → constant,
   parent-origin symbol (`result.parent_symbols` / `child_to_parent`) → pass
   through, otherwise → fresh parent symbol. Emit a new `CfOutput::Container`.
   *Honest limitation:* a **seeded** symbolic element dim that crosses a Loop body
   is not in `child_to_parent`, so it degrades to a fresh parent symbol on the way
   out — **sound** (an opaque symbol), and dtype/structure/**concrete** extents are
   preserved exactly. Symbolic element-dim *identity* across a Loop boundary is a
   deferred refinement.
5. **If unify.** Reuse the foundation's element unification: promote
   `merge_tensor`/`merge_shape` from `handlers/container.rs` to shared
   `pub(crate)` helpers (`unify_tensor_type`, `merge_element_shape`) taking `&mut
   SymbolInterner`, plus a new **recursive** `unify_value_type` (Tensor⨝Tensor via
   `unify_tensor_type`; Sequence/Optional recurse; Map requires equal key dtype;
   mismatched variants → error). `handlers/container.rs` keeps calling the shared
   helpers (DRY — no fork). `infer_if_outputs` calls `unify_value_type(then, else)`
   → branch container-type disagreement is an **error**, matching the tensor
   branch's dtype-mismatch error and the foundation's merge semantics.
6. **Apply.** Add `CfOutput::Container(ValueType)`; `apply_cf_outputs` writes it to
   the outer `containers` map (threaded in). Container CF outputs are not written
   to graph `Value`/`types` (no IR container repr) — consistent with how
   top-level `SequenceConstruct` outputs are handled today.

## Byte-identical guarantee
All new work is gated on a **non-empty** container map: `seed_containers` is empty
and `report.containers` is empty for pure-tensor graphs, the container read-back
check misses, and **no extra `fresh_dim` is minted** on the tensor path — so
symbol numbering and every resolved dtype/shape are unchanged. Regression test
`tensor_only_path_is_byte_identical_after_container_type_model` plus the existing
If/Loop/Scan tensor tests must stay green untouched.

## Tests (meaningful)
- **If** returns `Sequence<f32 [2,3]>` from both branches → outer output is
  `Sequence<f32 [2,3]>` (element dtype+shape asserted).
- **If** branch container **disagreement** (Sequence vs tensor; f32 vs i64
  element) → error.
- **If** branch element **extent** disagreement → element dim degrades to symbol,
  rank kept (foundation semantics).
- **Loop** carrying a sequence accumulator (body `SequenceInsert`s onto the
  carried input) → Loop output is a sequence with the element dtype preserved;
  carried-through concrete element dims preserved.
- **Loop** passthrough of a seeded sequence (Identity body) → element type
  preserved out.
- **Byte-identical** tensor control-flow regression stays green.

## Catalog counts
No new *ops* are registered (If/Loop/Scan already exist; SequenceMap deferred), so
the pinned `operator_count`/`entry_count` (217/262 after inc2) are **unchanged**.
If SequenceMap were added it would bump them; it is not in this slice.
