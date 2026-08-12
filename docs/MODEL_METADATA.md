# Inference metadata

Inference metadata is a declarative package IR. A bare single-model decoder may use
`model.io`. Every composite package uses exactly one `pipeline.workflow`; composite
metadata must not declare top-level `model.io`.

## Workflow boundary

`workflow.inputs` and `workflow.outputs` are named, typed package values. Every value
has a `TensorContract` (`dtype`, `rank`, optional symbolic shape). Inputs identify a
versioned runtime role or remain opaque application values. Outputs identify their
role and whether they are pre- or post-adapter.

## Components and graph

`workflow.components` declares ONNX artifacts or manifest-pinned adapter ABIs with
typed ports. The graph is recursive SSA control flow:

- `invoke` binds named values to component ports;
- `sequence` orders nodes;
- `loop` declares setup, body, condition, maximum iterations, induction value, and
  carried state;
- `branch` selects one case and exports only declared phi results/effect joins;
- `transfer` makes placement changes explicit;
- `emit` publishes streaming or final package outputs.

There are no strategies or phases. Control-flow location defines lifecycle. Policy
math—sampling, termination, scheduler/solver steps, masked updates, speculative
acceptance, and state updates—is supplied as ONNX components.

## Effects, state, and serving

State cells have typed contracts, invocation/session scope, an initializer ValueRef,
and explicit loop recurrence. Reads, writes, RNG, and emits thread linear effect
tokens. Session state is lease-protected. Serving contracts declare generic admission,
compaction, row activity, accepted length, and KV slot values without model-family
logic.

## Manifest and validation

The required manifest pins IR, ONNX opset, adapter ABI, custom-op, and capability
versions. Capabilities are derived from used features and checked at load. Unknown,
unsupported, unresolved, unordered, or ill-typed documents fail before execution.

Policy artifact port contracts and minimal workflows are specified in
[WORKFLOW_POLICY_COMPONENTS.md](WORKFLOW_POLICY_COMPONENTS.md).
