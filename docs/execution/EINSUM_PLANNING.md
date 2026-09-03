# Native Einsum semantic and planning contract

`onnx-runtime-ir::EinsumPlan` is the typed, execution-provider-neutral
contract for ONNX `Einsum`. `EinsumShapePlan` is the corresponding shape-only
contract for factories that do not receive dtypes. Both carry an explicit
`EinsumSchema` proof; model-facing paths resolve that proof from the effective
imported `ai.onnx` opset and never infer the schema from a dtype.

## Authoritative schema resolution

| effective `ai.onnx` opset | resolved schema | legal homogeneous dtypes |
|---|---|---|
| `< 12` | invalid: `Einsum` does not exist | none |
| `12..=27` | `Einsum-12` | uint8/16/32/64, int8/16/32/64, float16/32/64 |
| `>= 28` | `Einsum-28` | all `Einsum-12` types plus bfloat16 |

Evidence is pinned to ONNX main commit
[`5732eb5`](https://github.com/onnx/onnx/tree/5732eb5de3e6b353e1a5aa49fe5d577f81bb58e0):

- [`Einsum-12` uses `OpSchema::all_numeric_types()`](https://github.com/onnx/onnx/blob/5732eb5de3e6b353e1a5aa49fe5d577f81bb58e0/onnx/defs/math/old.cc#L3642-L3645).
- [`Einsum-28` uses `OpSchema::all_numeric_types_ir4()`](https://github.com/onnx/onnx/blob/5732eb5de3e6b353e1a5aa49fe5d577f81bb58e0/onnx/defs/math/defs.cc#L2005-L2008).
- [`all_numeric_types_ir4()` adds BFLOAT16 while
  `all_numeric_types()` does not](https://github.com/onnx/onnx/blob/5732eb5de3e6b353e1a5aa49fe5d577f81bb58e0/onnx/defs/schema.cc#L1092-L1124).
- The upstream change is
  [`2bcd4d5d218fc6fa713835833bba106629ca3167`](https://github.com/onnx/onnx/commit/2bcd4d5d218fc6fa713835833bba106629ca3167),
  “Add Einsum-28 with bfloat16 support (#8313).”

`EinsumPlan::build` and `EinsumShapePlan::build` remain explicitly
`Einsum-12`-compatible source entry points. Imported-model paths use
`build_for_opset` or resolve `EinsumSchema` and call `build_for_schema`.

Loader validation distinguishes declared metadata from graph-construction
placeholders through `Graph::value_type_is_known` and
`Graph::value_shape_is_known`. A placeholder `Float32`/empty shape is never
treated as a declared scalar. Known dtypes (including a declared output dtype)
and every provable known-shape constraint are still validated when other
metadata is absent; only facts that require the missing rank, dimensions, or
dtype are deferred to shape inference and runtime validation.

## Universal semantics

Every legal equation produces an `EinsumSemanticPlan` and a mandatory
`EinsumGenericNativePlan`; legality is never rejected because an expression is
not GEMM-shaped.

The generic index program covers:

- one through arbitrary-N operands, including scalar and zero-sized tensors;
- case-sensitive ASCII `A-Z`/`a-z`;
- fixed-rank ellipses, terms without ellipsis, and explicitly reduced
  ellipsis axes;
- repeated-label diagonals;
- local reductions and labels shared by three or more operands;
- outer, Hadamard, product, contraction, and broadcast combinations;
- explicit and implicit output ordering.

Only U+0020 ASCII spaces are removed. Tabs, other Unicode whitespace, Unicode
labels, malformed ellipses/arrows, duplicate or missing output labels, rank
mismatches, diagonal mismatches, non-broadcastable ellipses, mixed dtypes, and
schema-illegal dtypes are errors.

### Classification compatibility and migration

`EinsumClassification` retains its original exhaustive public variant set:
`ViewOnlyPermutation`, `DiagonalView`, `ReductionOrElementwise`, `Gemm`, and
`Unsupported(EinsumUnsupportedReason)`. The type is not `#[non_exhaustive]`, so
existing downstream exhaustive matches continue to compile.

`Unsupported` is deprecated and compatibility-only. Canonical planning never
constructs it for a legal equation. A general legal contraction that does not
fit one of the original fast-path variants is represented in the legacy
classification as its universal `ReductionOrElementwise` product/reduction
program. Invalid equations and schema-illegal dtypes still return errors rather
than a classification.

New code that needs to distinguish bounded tree planning uses
`EinsumPlan::planning_classification()` or
`EinsumShapePlan::planning_classification()`. Their new
`EinsumPlanningClassification` result contains `ContractionTree` and is
`#[non_exhaustive]` from its first release. CPU and CUDA placement/execution use
this planning classification, so the legacy compatibility mapping cannot cause
an unimplemented general contraction to be assigned to an old fast path.

The pre-existing exhaustive `EinsumPlanErrorKind` variant set is also
unchanged. New opset-resolution failures use `EinsumOpsetPlanError`, and the
shape-only zero-element-width check uses `EinsumConcretePlanError`; this avoids
retroactively adding variants or fields to the established error enum.
GEMM/BMM remains one optimization subtype over the same semantic/index program.

## General contraction trees

Every multi-operand semantic plan publishes either a bounded binary contraction
candidate set or an explicit `GenericNativeFallback` planner result, including
outer/Hadamard products even when a simpler execution class is available. The
fallback is an optimization decision, never a semantic rejection: the universal
index program remains complete for arbitrary operand counts.

Each intermediate records:

- its source leaf set;
- live logical axes and their canonical global index map;
- accumulator/intermediate storage policy;
- birth/last-use interval and reusable temporary slot;
- checked element and byte costs.

A reduced axis is eliminated exactly at the lowest node where:

1. it is absent from the requested output; and
2. every operand occurrence is contained in the merged subtree.

Thus a label shared by three or more operands remains live through earlier
Hadamard/product intermediates and is reduced only after the final required
occurrence enters the subtree.

## Bounded deterministic planning

The default `EinsumPlannerBudget` is explicit:

- exact subset DP through arity 5;
- at most 64 DP states;
- at most 4096 exact candidates;
- at most 64 logical axes on the exact path;
- at most 4096 total greedy work units, including a conservative reservation
  for tree nodes, source-leaf membership, stable IDs, lowering metadata, and
  pair scoring.

If any exact-path bound is exceeded, planning switches to the bounded stable
greedy heuristic. Exact enumeration has an additional hard structural ceiling
of `EinsumPlannerBudget::MAX_EXACT_TREE_OPERANDS` (eight), exact retained
metadata is bounded by
`max_candidates * EinsumPlannerBudget::EXACT_METADATA_UNITS_PER_CANDIDATE`,
and contraction depth is capped by
`EinsumPlannerBudget::MAX_CONTRACTION_TREE_DEPTH` (64). If the greedy work,
metadata, or depth bound cannot represent a candidate, the plan reports
`GenericNativeFallback`, zero tree candidates, and zero tree metadata while
retaining arbitrary-N semantics through `EinsumGenericNativePlan`.

`EinsumPlannerQuality` and `EinsumPlannerUsage` publish the selected mode,
state/candidate/axis use, total work, metadata units, maximum depth, and public
candidate-ID bytes. `fallback_reason()` distinguishes a work/metadata budget
decision, the depth ceiling, and checked bookkeeping overflow. Candidate IDs
are constructed incrementally once, remain stable lexicographic tree
expressions for materialized candidates, and are the final tie-break.
Equal-cost greedy merges prefer shallower trees before the ID tie-break, so tied
unit-extent products remain balanced rather than producing a skew chain.
Lowering walks the bounded tree in iterative post-order; it does not recurse
with operand count.

Static and concrete costs use checked `u128`: FLOPs, unary/product work,
intermediate elements/bytes, peak live bytes, temporary traffic,
layout/packing traffic, and broadcast amplification. A symbolic dimension has
an unbounded unknown upper bound; no finite size is invented. Concrete runtime
shapes trigger exact re-scoring and deterministic greedy re-planning.

`preferred_candidate_with_memory_ceiling` chooses the lowest-cost candidate
that fits the ceiling. If none fits, that is not semantic rejection: execution
selects the mandatory generic/tiled index program.

## Precision policy

Precision is IR data, not a backend fast-path choice:

- float16/bfloat16 load, multiply, accumulate, and materialize intermediates in
  float32, then narrow exactly once at the final output;
- float32 accumulates/stores intermediates in float32;
- float64 accumulates/stores intermediates in float64;
- fixed-width integers retain their input width, with arithmetic defined as
  wrapping modulo `2^width` so signed overflow never relies on host-language
  undefined behavior.

## Staged CPU/CUDA execution

The final native CPU/CUDA goal is exhaustive execution of every expression and
dtype legal under the resolved schema.

This PR establishes the semantic/index/planning API only; it does **not** add
new execution kernels. Current CPU continues executing its existing
float32/float16 view, diagonal, generic reduction/elementwise, and flat
GEMM/BMM paths. Current CUDA continues executing its existing float32/float16
view/diagonal and flat GEMM/BMM paths. They may temporarily decline
`GenericNative`, general contraction-tree execution, and bfloat16 execution
with an actionable staged-implementation message. Those declines describe
current kernel coverage, not a permanent semantic limitation.
