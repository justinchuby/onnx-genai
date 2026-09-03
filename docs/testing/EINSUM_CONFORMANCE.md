# Einsum conformance corpus

`onnx-runtime-einsum-conformance` is the shared correctness boundary for native
CPU and CUDA Einsum work. It intentionally does **not** depend on
`onnx-runtime-ir`, `EinsumPlan`, contraction trees, cost models, or an execution
provider.

## Authority

`fixtures/onnx-einsum-schema-authority.txt` vendors the relevant source excerpts
from ONNX commit
[`5732eb5de3e6b353e1a5aa49fe5d577f81bb58e0`](https://github.com/onnx/onnx/commit/5732eb5de3e6b353e1a5aa49fe5d577f81bb58e0).
The harness verifies the complete fixture SHA-256 and the source facts:

- Einsum-12 uses `all_numeric_types()` and excludes BF16.
- Einsum-28 uses `all_numeric_types_ir4()` and includes BF16.

An installed ONNX Python package is an optional evaluator adapter, never schema
authority. In particular, ONNX 1.22 exposes only Einsum-12; the adapter reports
that limitation and refuses to reinterpret an opset-28 BF16 case as v12.

## Corpus

The checked-in `corpus-v1.json` stores only the generator configuration, case
counts, and canonical JSON digest. Tensor payloads are regenerated from per-case
seeds. The corpus contains:

- 31 named cases and 128 seeded generated cases;
- operand arities 1, 2, 3, 4, 8, and 16;
- all 52 case-sensitive ASCII labels, diagonals, scalar terms, explicit and
  implicit output, fixed-rank retained/reduced ellipsis, and terms without
  ellipsis;
- local and shared multi-axis reductions, outer/Hadamard products, zero/one
  extents, and fixed-ellipsis broadcasting;
- mandatory f32/f16 opset-12 and BF16 opset-28 lanes, plus exact integer
  wrapping cases;
- 24 malformed records covering grammar, arrows, output labels, ellipses,
  whitespace/Unicode, node arity, rank/dimensions, dtypes, and BF16 before
  opset 28.
- 346 forced-route probes split evenly between CPU and CUDA handoff records.

Each tensor is capped at 8 MiB, aggregate CPU working set at 32 MiB, and GPU
case/workspace at 64 MiB. The generator caps materialized elements and direct
oracle work, not semantic operand arity.

## Oracle and tolerance

The direct evaluator parses the equation independently. It multiplies factors
in input order and advances reduction coordinates lexicographically. F16 and
BF16 inputs promote to f32 and narrow once at final output. Integers multiply
and add modulo their fixed width. Floating comparisons use each output's sum of
absolute products and operation count; NaN class, infinity sign, and special
signed-zero cases are checked separately.

The Python adapter runs ONNX `ReferenceEvaluator` and ONNX Runtime CPU when
installed. It is a second oracle for finite portable lanes, not a replacement
for the direct evaluator.

## CPU/CUDA handoff

Every `CaseRecord` carries `RouteProbe` records for applicable forced routes:
GenericNative, exact subset-DP, deterministic heuristic, native MatMul, and
CUDA cuBLAS. A real backend adapter must return `BackendObservation` with the
route actually taken, planner quality, measured workspace, capture result, and
output tensor. Capture assertions require nonzero captures and replays with zero
fallbacks. `verify_observation` checks all fields against one oracle evaluation.

The harness deliberately ships no CPU or CUDA adapter and no placeholder route.
Backend owners must wire genuine forcing/telemetry before a route test can pass.
