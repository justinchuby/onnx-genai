# #449 increment 4 — closeout: SequenceMap + cross-subgraph container capture (DESIGN)

Branch `squad/449-container-inc4`, stacked on **#527** (inc3a If/Loop container
threading). refs #449. CPU-only, pure shape-inference crate.

## Goal
Bring #449 to a natural DONE point by closing the two remaining container gaps
in control flow, both of which **compose from the inc3a body-threading driver**
(`seed_containers` → `infer_graph_scoped` → `InferenceReport.containers` →
`CfOutput::Container`), plus a fold-in of the previously-deferred Scan state case.

## 1. SequenceMap (ONNX opset 17)
`SequenceMap(input_sequence, additional_inputs...) { body }` applies `body` to
each element of `input_sequence`, zipped with the additional inputs; each body
output becomes an output **sequence**.

Type rules:
- **Body formal input `i`** = the per-element type of SequenceMap input `i`:
  - input `i` is a **sequence** → body sees its **element** type (`Sequence<E>`
    ⇒ `E`). If `E` is a tensor, seed the body input `Value` (tensor path); if `E`
    is itself a container (seq-of-seq), seed via the container map.
  - input `i` is a **tensor** → body sees the **whole tensor** each iteration
    (broadcast), seeded on the body input `Value`.
- **Body output `j`** (a tensor, or a container for seq-of-seq bodies) → wrapped
  as **`Sequence<body_output_j>`** = SequenceMap output `j`.

Composition (DRY): this is exactly the inc3a machinery re-pointed:
- element seeding reuses `set_body_input` (tensor leaf) and `body_container_seeds`
  (container leaf) — the same two seed channels Loop uses;
- output read-back reuses `read_body_output` + `map_container_to_parent` (the same
  helpers Loop/Scan/If use), then wraps each in `ValueType::sequence(...)`;
- dispatched from `infer_control_flow` alongside If/Loop/Scan (a subgraph-bearing
  op — it needs body inference, so like If/Loop/Scan it lives in the CF driver,
  **not** the registry, which has no subgraph access).

Multiple input sequences (zip) + additional tensor inputs fall out for free: the
seeding walks body formal inputs positionally against SequenceMap operands and
picks tensor-vs-container per operand.

## 2. Cross-subgraph container capture (`remap_node_io` fix)
Today `remap_node_io` (and `extend_visible_scope`) set `value_type: None`, so a
**sequence captured by name from an outer scope** into a subgraph body loses its
container type — the body sees an untyped value. Fix:
- **`remap_container_type`** (new): parent→child remap of a `ValueType`, recursing
  and remapping each tensor-leaf element-shape symbol through the same
  `remap_dim_expr` the tensor capture path uses (parent symbol → interned child
  symbol, recorded in `child_to_parent`). Mirrors the tensor branch of
  `remap_node_io` exactly.
- **`remap_node_io`**: thread `io.value_type` through `remap_container_type`.
- **`extend_visible_scope`**: publish an outer-produced value's container type
  into the scope binding (build a binding when the value has a tensor type **or**
  a container type, not only a tensor type).
- **`bind_captures`**: gains the `containers` map and writes a captured binding's
  `value_type` into the body's container map, so a body value that references an
  outer sequence resolves to its element type.

This is what makes real nested-control-flow-with-sequences correct: e.g. a Loop
whose body's `If` returns a sequence built from an outer-scope sequence.

## 3. Scan container state (inc3b) — FOLD IN
Earlier deferred as "mechanically identical to Loop seeding, rare, no scan-output
payoff." Re-assessed against the inc3a machinery: it is genuinely ~a dozen lines
and reuses the identical seed/read-back helpers, so folding it in lets #449 close
with **all four** subgraph ops (If/Loop/Scan/SequenceMap) container-complete
rather than leaving a dangling inc3b:
- **`body_container_seeds`** Scan arm: seed the first `num_state` body inputs from
  the node's container **state** operands (operands `0..num_state`).
- **`infer_scan_outputs`**: for a **state** slot (`slot < num_state`), read the
  body output's container type first and emit `CfOutput::Container` remapped to
  parent; scan-output slots stack tensors and are never containers (unchanged).

Byte-identical: gated on a non-empty container operand, exactly like Loop — a
tensor Scan seeds/read-backs nothing new.

## Byte-identical tensor path (the gate stays real)
Every new path is gated on a container actually being present: `body_container_
seeds` returns empty, the container read-backs miss, and `remap_container_type` is
only invoked when `io.value_type.is_some()`. Pure-tensor graphs mint no extra
`fresh_dim` and touch no container map, so symbol numbering is unchanged and the
`tensor_only_path_is_byte_identical_after_container_type_model` regression holds.

## Catalog counts — HONEST note
SequenceMap is a **subgraph-bearing op handled in `infer_control_flow`**, exactly
like If/Loop/Scan — none of which are registry rules (the registry `InferenceFn`
has no access to subgraph bodies). So the pinned registry catalog **stays 217 ops
/ 262 entries**: there is no new *registry* rule to count. (The task anticipated
"+1"; that assumed a registry handler, which is structurally impossible for a
body-inferring op. Documented here to avoid a phantom declaration — requirement 4
is "no phantom declarations", which this honours.)

## Test plan (meaningful, not non-crash)
- SequenceMap: body transforms the element (e.g. `Shape`/`Identity`/`Cast`);
  assert the OUTPUT sequence's element dtype/shape inferred from the body output.
- SequenceMap zip: two input sequences + the body consuming both; output element
  type reflects the body.
- SequenceMap with an additional **tensor** input (whole-tensor broadcast).
- Cross-subgraph capture: an outer-scope sequence referenced inside an `If`
  branch body resolves to its element type (the `remap_node_io` fix), proven via
  `report.containers` on the outer node.
- Scan state container: a Scan carrying a sequence state var → state output is a
  sequence (element preserved).
- Byte-identical regression stays GREEN; existing If/Loop/Scan tensor tests stay
  GREEN.

## After inc4 — is #449 DONE?
With foundation (ValueType) + seq ops (Empty/Construct/Length/At/Insert/Erase) +
seq↔tensor conversion (SplitToSequence/ConcatFromSequence) + container control
flow (If/Loop/Scan/SequenceMap) + cross-subgraph capture, the container-type
shape-inference surface a real model exercises is covered. Recommendation drafted
in the status note after implementation.
