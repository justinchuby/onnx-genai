### 2026-07-30: #355 scope split — control-flow / recurrent shape inference is landed; container types deferred
**By:** Harry (shape-inference / IR)
**What:** Audited `crates/onnx-runtime-shape-inference/` against issue #355 and split it into a
tractable tensor-only half (done) and a container-type half (deferred, needs a new IR/type-model
extension). Recommend closing the tensor-only slice against #355 and opening a **new follow-up
issue** for the Sequence/Optional/Map container-type model.

---

#### Current coverage (what already exists on `main`)

The tensor-only slice of #355 is **already implemented and tested** — it does not need to be
re-implemented:

- **Control flow** (`src/infer.rs`, landed by PR #362 "control-flow (If/Loop/Scan) shape
  inference (#355)"):
  - `If` — reconciles the two branch outputs positionally; equal concrete dims stay concrete,
    matching parent-origin symbols pass through, otherwise a fresh symbol; rank mismatch →
    known-dtype/unknown-rank; dtype mismatch → error. (`infer_if_outputs`)
  - `Loop` — seeds the body formal inputs `(iter_num:i64[], cond_in:bool[], v_1..v_N)` from the
    node operands (`seed_loop_body`), infers the body, then maps body outputs
    `(cond_out, v_1..v_N, scan_1..scan_K)` back: loop-carried finals keep their body shape, scan
    outputs gain a **symbolic** leading trip-count axis (static `M` is only an upper bound because
    `cond` may exit early). (`infer_loop_outputs`)
  - `Scan` (opset ≥ 9 only; opset-8 form deliberately left unresolved) — seeds N state inputs
    unchanged and M = `num_scan_inputs` scan slices with their `scan_input_axes[j]` (default 0)
    stripped (`seed_scan_body`); maps outputs back re-inserting the scan axis
    (`scan_output_axes[k]`, default 0) sized to the first scan input's sequence length.
    (`infer_scan_outputs`)
  - Shared machinery: `seed_control_flow_body`, `read_body_output`, `map_body_shape`/
    `body_parent_symbols` (parent-symbol pass-through vs fresh-symbol remap), `cf_typed`
    (degrades an overflowing all-static shape to unknown-rank so eager buffer sizing can't reject
    the graph), `apply_cf_outputs`. Recurses into nested subgraphs (`infer_child_subgraphs`).
- **Recurrent** (`src/handlers/recurrent.rs`, landed by PR #386 "RNN/GRU/LSTM shape inference
  (#355)"):
  - `RNN`/`GRU`/`LSTM` registered at their opset boundaries: v1 (pre-layout) and v14 (`layout`
    attr). `Y=[seq, num_dir, batch, hidden]`, `Y_h=[num_dir, batch, hidden]`, and for LSTM
    `Y_c=[num_dir, batch, hidden]`; `layout=1` (opset ≥ 14) swaps to batch-major. `num_directions`
    = 2 for `bidirectional`, else 1. Reads only `X` + `hidden_size`; missing optional inputs
    (`B`, `sequence_lens`, `initial_h`/`initial_c`) are irrelevant to output shape. Permissive on
    missing/invalid `hidden_size`/`direction` and on non-rank-3 `X`.

**Naming caution for future readers:** `src/handlers/sequence.rs` is misleadingly named — it
handles `Tile`/`Range`/`CumSum` (pure tensor ops), **not** the ONNX `Sequence*` container family.
The container family is genuinely unhandled.

#### Tensor-only tractable set — ONNX rules (for reference / regression intent)

- **Loop**: outputs = N loop-carried finals (body carried-output shape) + K scan outputs
  (per-iteration body output stacked along a new leading axis = trip count; symbolic).
- **Scan** (opset 9+): `num_scan_inputs`, `scan_input_axes` (slice axis stripped per iteration),
  `scan_output_axes` (stack axis re-inserted), sequence length = scan input extent along its scan
  axis. N final-state outputs keep body shape; K scan outputs re-insert the sequence axis.
- **GRU/LSTM/RNN**: `Y [seq, num_dir, batch, hidden]`, `Y_h [num_dir, batch, hidden]`, LSTM adds
  `Y_c [num_dir, batch, hidden]`; `hidden_size` attr; `num_directions = 2` iff bidirectional;
  `layout=1` (opset ≥ 14) → batch-major axis order.

#### This task's delta

Implementation was already complete, so this PR **hardens the tensor-only slice with edge-case
tests** the existing suites did not cover (reverse/unidirectional direction, unknown/absent/
non-rank-3 `X`, invalid `direction`, multi-scan-input `Scan` with per-input axes, multi-carried
`Loop` with multiple scan outputs, opset v1/v14 boundary). No behavior change was required; no bug
was found.

#### DEFERRED — container-type model (Sequence / Optional / Map)

`SequenceEmpty/Construct/Insert/Erase/At/Length/ConcatFromSequence/SplitToSequence`, `Optional/
OptionalHasElement/OptionalGetElement`, and the Map ops **cannot** be inferred today: IR `Value`
and inference `TypeInfo` carry only `(dtype, shape)` on SSA edges — there is no way to express
"sequence of tensor(float)", "optional(tensor)", or "map(int64, tensor(float))". This is an
architectural type-model change, out of scope here.

**What the follow-up needs (sketch):** a container element-type variant on the type model, e.g.
```
enum ValueType {
    Tensor(TypeInfo),                 // today's dtype+shape
    Sequence(Box<ValueType>),         // seq(element)
    Optional(Box<ValueType>),         // optional(element)
    Map(DataType, Box<ValueType>),    // map(key_dtype, value)
}
```
threaded through `TypeInfo`/`NodeIo`, the IR `Value`, `seed_sources`, control-flow body
seeding/read-back, and every handler that produces or consumes containers. Then the Sequence/
Optional/Map handlers become straightforward element-type transforms.

**Recommendation:** the container-type model is a large, self-contained architectural change with
its own test surface. **Split it into a new issue** ("IR container-type model + Sequence/Optional/
Map shape inference") rather than keeping #355 open indefinitely. #355's tensor-only intent
(control flow + recurrent) is satisfied.
**Why:** Keeps #355 closable on delivered value, gives the container work a clean, well-specified
home, and prevents a large type-model refactor from blocking the shipped recurrent/control-flow
inference.
