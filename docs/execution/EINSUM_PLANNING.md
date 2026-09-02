# Native Einsum planning contract

`onnx-runtime-ir::EinsumPlan` is the execution-provider-neutral contract for
ONNX `Einsum` opset 12. Shape inference and every native EP consume this plan;
they must not reparse `equation` or classify an equation by matching its string.

## Guarantees

- Only lowercase ASCII letters are labels. U+0020 ASCII spaces are stripped
  from the equation; every other whitespace or non-syntax character is
  rejected. The normalized equation is parsed once and validated against every
  input dtype, rank, and dimension.
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
  `usize` overflow. Dynamic plans retain constraints for concrete admission.

## Structural classifications

`EinsumClassification` has five outcomes:

1. `ViewOnlyPermutation`: one input, no diagonal, no reduction. The output map
   is `output axis -> source unique axis`.
2. `DiagonalView`: diagonal extraction and optional permutation, with no
   arithmetic reduction.
3. `ReductionOrElementwise`: aligned elementwise product and/or uncoupled
   reductions. The iteration order is output axes followed by reduction axes.
4. `Gemm`: a binary contraction directly lowerable to GEMM/BMM.
5. `Unsupported`: legal N-way/general contraction semantics with a structured
   reason. Native EPs fail clearly or decline the node; they do not guess.

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

No execution kernel is defined by this contract.
