### 2026-07-30: #449 IR container-type model + Sequence/Optional/Map shape inference — design
**By:** Harry (shape-inference / IR)
**What:** Additive `ValueType` container-element model for `onnx-runtime-shape-inference`,
plus the first proven op family (a Sequence subset). Foundation + simplest slice only;
the rest of #449 is a documented multi-PR roadmap.

#### Blast radius (today's tensor-only type model)

`crates/onnx-runtime-shape-inference/`:

- `src/context.rs`
  - `TypeInfo { dtype: DataType, shape: TypedShape }` — the only value-type today; no
    container representation.
  - `NodeIo { type_info: Option<TypeInfo>, shape_data: Option<ShapeData> }` — the per-slot
    value handed to/from handlers.
  - `InferenceContext` — handler-facing API: inputs via `input_type/input_shape/input_dtype`,
    outputs via `set_output/set_output_type`. Handlers only ever see/produce `TypeInfo`.
- `src/registry.rs`
  - `InferenceFn = fn(&mut InferenceContext) -> Result<(), ShapeInferError>`; every handler
    has this signature. `infer_node` returns `Vec<NodeIo>` (one per output slot).
- `src/infer.rs` (whole-graph driver)
  - `seed_sources` seeds `types: HashMap<ValueId, TypeInfo>` from IR `Value` dtype+shape.
  - `gather_inputs` assembles per-input `NodeIo` from `types`/`shape_data`.
  - `infer_node_outputs` writes each output `NodeIo` back into `types`/`shape_data`.
  - `write_back_types` lowers `TypeInfo` into the IR `Value` (dtype + `Vec<Dim>`).
  - Control flow: `seed_control_flow_body`/`set_body_input` seed Loop/Scan body inputs;
    `infer_if/loop/scan_outputs` + `CfOutput` + `apply_cf_outputs` map body outputs back;
    all typed by `TypeInfo`. Scope capture across subgraphs: `remap_node_io`,
    `extend_visible_scope` (both copy `TypeInfo`).
- `src/lib.rs` — crate docs explicitly defer Sequence/Optional/Map "until `TypeInfo` gains
  a container element type" (issue #355 note).
- IR crate (`onnx-runtime-ir`): `Value` carries only tensor dtype + `Vec<Dim>`. It has **no**
  container representation and is **not** modified by this increment (see below).

#### Least-invasive additive representation (chosen)

Add a **parallel** container-type layer; do not disturb the tensor `TypeInfo` path.

- New sum type in `context.rs` (shape-inference crate, *not* the IR crate):
  ```rust
  enum ValueType {
      Tensor(TensorType),
      Sequence(Box<ValueType>),
      Optional(Box<ValueType>),
      Map(DataType, Box<ValueType>),
  }
  struct TensorType { dtype: DataType, shape: Option<TypedShape> }
  ```
  Deliberate refinement of the issue's `Tensor(TypeInfo)` sketch: the tensor leaf carries an
  **optional** shape, because dtype-only container producers (`SequenceEmpty` knows the
  element dtype but not its rank) must be representable without fabricating a bogus shape.
  A `TensorType` with a known shape converts to/from `TypeInfo` losslessly.
- `NodeIo` gains `value_type: Option<ValueType>` (defaults `None` via `#[derive(Default)]`).
- The driver gains a **parallel** `containers: HashMap<ValueId, ValueType>` alongside the
  existing `types` map. A value is a plain tensor unless it has a `containers` entry.
- `InferenceContext` gains `input_value_type(i)` / `set_output_value_type(i, ValueType)`,
  used *only* by container handlers.
- Container types live only in-inference this increment; they are **not** written back to the
  IR `Value` (the IR cannot represent them yet — that is a documented follow-up). Sequence
  edges therefore stay `unresolved` at the IR level, exactly as today, but are now typed
  *inside* inference so consumers like `SequenceAt` recover the element tensor type.

#### How the tensor-only path stays byte-identical

1. `TypeInfo`, the `types` map, `seed_sources`, per-op inference, and `write_back_types` are
   **unchanged** — no field added to `TypeInfo`, no change to any existing method.
2. The new `NodeIo.value_type` field defaults `None`; the new `containers` map starts empty
   and is only ever populated by the four new container handlers. No existing handler calls
   the new setters, so for a pure-tensor graph every existing code path executes identically.
3. A regression test (`tensor_only_path_is_byte_identical`) infers a real multi-op tensor
   graph twice — with the container layer compiled in — and asserts the resulting `TypeInfo`
   set is byte-identical to a captured baseline.

#### Implemented this increment (simplest Sequence subset)

- `SequenceEmpty` (opset 11): `seq(tensor(dtype))`, dtype from the `dtype` attr (default
  Float32), element shape unknown.
- `SequenceConstruct` (opset 11): `seq(elem)` where `elem` is the common element type of the
  tensor inputs. Homogeneous-dtype is required by the ONNX spec → mismatched input dtypes
  raise `ShapeInferError::Invalid` (documented rule). Element shape = per-dim agreement of the
  inputs (equal dims kept, incl. symbolic; disagreements degrade to a fresh symbol; differing
  ranks → unknown element shape).
- `SequenceLength` (opset 11): `i64` scalar tensor (pure tensor output).
- `SequenceAt` (opset 11): `seq(elem)` + index → the element tensor type. Recovers the full
  `TypeInfo` when the element shape is known (preserving symbolic dims); dtype-only elements
  stay unresolved at the tensor level pending the unknown-rank follow-up.

Shared element-type helpers on `ValueType`/`TensorType` (no per-op copy-paste): constructors,
`as_sequence_element`, `as_tensor`, and `TypeInfo`↔`TensorType` conversion.

#### #449 follow-up roadmap (multi-PR)

1. **(this PR)** Foundation `ValueType` + Sequence subset: Empty/Construct/Length/At.
2. **Sequence mutations & tensor⇄sequence:** `SequenceInsert`, `SequenceErase` (return seq),
   `ConcatFromSequence` (seq→tensor, new concat/stack axis), `SplitToSequence` (tensor→seq),
   plus dtype-only `SequenceAt` recovery via a known-dtype/unknown-rank tensor output.
3. **Container-aware control flow:** carry `ValueType` through Loop/Scan body seeding and
   `If` branch reconciliation, and through cross-subgraph scope capture
   (`remap_node_io`/`extend_visible_scope`).
4. **Optional family:** `Optional`, `OptionalHasElement`, `OptionalGetElement`.
5. **Map + ONNX-ML:** the `Map` variant, `ZipMap`, `DictVectorizer`, `CategoryMapper`
   sequences.
6. **IR persistence:** extend the IR `Value`/type-proto so container edges survive write-back
   and serialization (today they live only within an inference pass).

**Why:** Container ops cannot be inferred while a value type is only `(dtype, shape)`. Wrapping
rather than replacing keeps 100% of tensor handlers and the hot tensor path untouched, so the
99% tensor case has zero behavior/perf change, while unblocking the container families one
proven slice at a time.
