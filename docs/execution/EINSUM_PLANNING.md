# Native Einsum planning contract

`onnx-runtime-ir::EinsumPlan` is the typed execution-provider-neutral contract
for ONNX `Einsum` opset 12 and preserves the exact validated input/output dtype.
`EinsumShapePlan` carries the same structural contract for kernel factories
that receive shapes but no dtype; it cannot claim a fabricated dtype. Shape
inference and native EPs consume the appropriate representation and must not
reparse `equation` or classify an equation by matching its string.

## Guarantees

- ASCII `A-Z` and `a-z` are case-sensitive labels (`A` and `a` are distinct).
  U+0020 ASCII spaces are stripped from the equation; every other whitespace or
  non-syntax character is rejected. The normalized equation is parsed once and
  validated against every input dtype, rank, and dimension.
- Every explicit ellipsis expands to the same fixed number of dimensions, as
  required by opset 12. Terms without ellipsis do not acquire synthetic axes.
  Input axes map to canonical named or ellipsis axes. Repeated
  labels are represented as one operand axis with all contributing physical
  axes, so a diagonal view uses the sum of their strides.
- Each logical axis records all physical occurrences, its equality or broadcast
  rule, its best static extent, whether runtime validation remains necessary,
  and whether it is retained or reduced.
- Output axes and dimensions are in exact requested order. Implicit outputs put
  ellipsis axes first, then labels occurring exactly once in ASCII order.
- Fully static input, output, and flattened GEMM products are checked for
  `usize` overflow. Contraction-tree costs use checked `u64` counters. Dynamic
  symbols remain unknown with an infinite upper bound until a caller supplies
  concrete runtime shapes; the planner never invents a finite symbolic bound.

## Structural classifications

`EinsumClassification` is non-exhaustive and currently has six outcomes:

1. `ViewOnlyPermutation`: one input, no diagonal, no reduction. The output map
   is `output axis -> source unique axis`.
2. `DiagonalView`: diagonal extraction and optional permutation, with no
   arithmetic reduction.
3. `ReductionOrElementwise`: aligned elementwise product and/or uncoupled
   reductions. The iteration order is output axes followed by reduction axes.
4. `Gemm`: a binary contraction directly lowerable to GEMM/BMM.
5. `ContractionTree`: all ordered binary-tree candidates for a coupled
   two-input contraction with leaf-local reductions or a coupled three-input
   contraction.
6. `Unsupported`: legal N-way/general contraction semantics with a structured
   reason. Native EPs fail clearly or decline the node; they do not guess.

Downstream EP matches include a defensive wildcard refusal. Adding a future
classification therefore cannot silently enter an existing execution path.

## Coupled contraction trees

For arity two the tree planner enumerates both operand orientations when a
leaf-local reduction prevents the existing flat `Gemm` route. For arity three
it enumerates all 12 ordered candidates: every ordered first pair, with the
result on either side of the root. Candidate IDs such as `((0,1),2)` are stable
and lexicographically ordered.

Each candidate records:

- leaf, unary-result, binary-intermediate, and final value IDs;
- leaf-local reductions at the lowest node containing all occurrences;
- each binary node's batch/M/K/N axes, value-axis mappings, virtual singleton
  ellipsis axes, checked geometry, and output permutation;
- a linear-scan temporary-slot schedule with birth/last-use steps;
- a structured refusal when checked geometry or cost accounting overflows.

A reduced label is eliminated only at the lowest tree node whose leaf set
contains every input occurrence and only when the label is absent from the
output. Three-way shared reduced axes are refused because pairwise evaluation
would first require a Hadamard/outer intermediate that cannot eliminate the
axis. Reduced ellipses and coupled contractions with more than three inputs are
also refused. Retained shared axes, repeated-label diagonal leaves, zero
extents, terms without ellipsis, and case-sensitive explicit/implicit outputs
remain part of the canonical equation plan.

The deterministic EP-neutral score compares, in order: scalar FLOPs;
leaf-unary/K-free product work; intermediate elements (and their dtype-scaled
bytes); peak live temporary bytes; total temporary traffic; layout/packing
traffic; broadcast amplification; slot count; and stable candidate ID.
`EinsumContractionCost` exposes symbolic/static bounds. Typed
`resolve_concrete_contraction_tree` uses the plan's real dtype width;
`EinsumShapePlan` requires the caller to supply that width, preserving the
shape-only contract. Concrete resolution revalidates shapes, re-scores every
candidate, and returns exact per-node B/M/K/N geometry.

Every unary or binary step stores its result in the Einsum node's common ONNX
dtype. The plan describes these storage/rounding boundaries and operation
counts; the EP owns accumulation precision, GEMM library choice, packing
strategy, and kernel policy.

## GEMM/BMM layout contract

A `Gemm` plan exposes canonical groups and mappings:

- left: `[batch..., M..., K...]`
- right: `[batch..., K..., N...]`
- result: `[batch..., M..., N...]`

Each operand order entry indexes its post-diagonal `unique_axes`; `None` inserts
a singleton for an ellipsis batch axis absent from a term that did not contain
ellipsis. Named batch labels require equality, while ellipsis batch axes use
broadcast constraints.
`output_permutation` maps the requested output to the canonical result.
`EinsumGemmGeometry` carries the full batch shape and checked flattened
`batch/M/K/N` products when static; dynamic values remain explicit.

CPU/CUDA implementations may form diagonal/transpose views, insert singleton
batch axes, flatten the declared groups, invoke their GEMM/BMM primitive, reshape
the canonical result, and apply `output_permutation`. They must re-check every
logical axis marked `requires_runtime_check` against concrete runtime shapes.
`resolve_concrete_output_shape` performs those equality/broadcast checks without
reparsing, and `resolve_concrete_gemm_geometry` returns overflow-checked concrete
batch/M/K/N geometry for `Gemm` plans.

The native CPU EP implements the four pre-existing executable classes for
`Float32`/`Float16`: zero-copy view/diagonal outputs where the executor permits
aliases, canonical reduction/elementwise loops, and binary GEMM/BMM lowering
through the existing MatMul kernel. BFloat16 is not in the canonical ONNX
Einsum opset-12 type constraint and is declined before kernel creation; this
implementation does not expand the schema. CPU and CUDA deliberately decline
`ContractionTree` until their separate execution PRs consume the recorded
temporary schedule and concrete re-score. `Unsupported` remains a claim-time
refusal with the plan's structured reason. Other EPs may implement different
schema-valid subsets without changing this shared contract.
