# #449 container types — increment 2 (Harry)

Branch: `squad/449-container-inc2` · base `main` · refs #449 (foundation PR #477 merged).

## What I added (all `""` domain, opset 11, in `handlers/container.rs`)

Sequence **mutation** ops:
- **SequenceInsert** `(seq, tensor, [position]) → seq`: element type unifies the
  existing element type with the inserted tensor via the shared `merge_tensor`
  helper (dtype-mismatch → error; extent disagreement → fresh symbol; rank
  mismatch / dtype-only side → unknown shape). `position` never affects the type.
- **SequenceErase** `(seq, [position]) → seq`: element type preserved verbatim
  (length is not tracked in `ValueType`, so nothing else changes).

Tensor⇔sequence **conversion** ops:
- **SplitToSequence** `(tensor, [split], axis, keepdims) → seq`: element dtype =
  input dtype. Explicit `split` input present → split-axis extent is symbolic
  (chunk sizes vary; a single element type can't pin it). No split → chunk size
  1, `keepdims` decides keep-at-1 vs remove the axis. Matches the runtime
  `sequence/split.rs` semantics (keepdims only affects the no-split `Each` case).
- **ConcatFromSequence** `(seq, axis, new_axis) → tensor` — the **sequence→tensor**
  direction of the seam. Recovers a tensor `TypeInfo` from the sequence element
  tensor type: dtype = element dtype; `new_axis=0` makes the concat axis symbolic
  (unknown total across the sequence); `new_axis=1` inserts a fresh symbolic stack
  dim at `axis` (rank +1). `axis` mandatory (errors if absent, like the runtime).
  A dtype-only element (unknown rank) leaves the output unresolved — same honest
  degradation as `SequenceAt`.

## DRY (reused foundation helpers — no per-op special-casing)
- Generalised the foundation's `merge_element` → **`merge_tensor(ctx, op, acc, other)`**
  (dtype homogeneity + per-dim `merge_shape` agreement, degrade-to-symbol). Now
  shared by `SequenceConstruct` **and** `SequenceInsert`. `merge_shape` (fresh-dim
  on disagreement, `None` on rank mismatch) unchanged and reused.
- New `sequence_element_tensor(ctx, i)` helper reads a sequence input's element
  tensor leaf; reused by SequenceInsert / ConcatFromSequence.
- Axis handling reuses `handlers::checked_axis`; `output_rank = rank + new_axis`
  makes one axis check cover both concat and stack (no bespoke range math).

## Hard requirements
- **Tensor path byte-identical**: `tensor_only_path_is_byte_identical_after_container_type_model`
  still GREEN, untouched. No tensor handler touches the container layer.
- **Catalog count**: 213→**217** ops / 258→**262** entries (+4 ops, one opset-11
  entry each). Pinned test updated; no phantom declarations.
- **Tests**: +21 tests (op_rules.rs 254→275): real inferred dtype/shape asserts,
  symbolic-dim preservation, dtype/rank-mismatch → error/degrade, and the two
  round-trips that matter — SequenceConstruct→Insert→At recovers the element type,
  and **SplitToSequence→ConcatFromSequence recovers a rank-2 f32 tensor** (concat
  axis honestly symbolic). Split/concat dtype cases parameterised.
- fmt + `clippy -D warnings` clean; `onnx-runtime-session`/`-eager` still build.

## Scoped OUT to increment 3+
- **SequenceMap**: needs subgraph/body type-threading (map a body graph over each
  element) — that is container-aware control flow, i.e. increment 3 territory
  alongside Loop/Scan/If container carries. Deferred deliberately.
- **Optional** ops (OptionalGetElement / OptionalHasElement / Optional) and
  **Map** producers: foundation `ValueType` already models them, but no op rules
  yet — candidate increment 3 once a consuming model needs them.
- ConcatFromSequence with a dtype-only element stays unresolved because the
  tensor `TypeInfo` path has no dtype-only representation; revisit if/when an
  unknown-rank tensor type lands.

## Roadmap (updated)
- inc1 ✅ foundation (PR #477, merged): ValueType layer + Empty/Construct/Length/At.
- inc2 ✅ this PR: sequence mutation + tensor⇔sequence conversion (seam proven both directions).
- inc3 → container-aware control flow (SequenceMap, If/Loop/Scan container carries) + Optional/Map op rules.
