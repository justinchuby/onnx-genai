# Inference metadata

Inference metadata is a declarative package IR. Every package — including one that
ships a single ONNX file — expresses its executable graph ABI in exactly one place:
`pipeline.workflow`. Component ports and their roles, `invoke` bindings, state cells,
and `state_service` groups are the sole serialized truth about which port carries
what. `model:` holds package-wide baked facts (attention geometry, vocabulary and
length limits, MoE structure, sharding), not port names.

The flattened decode ABI a single-graph fast path wants is *derived* from that one
representation by `InferenceMetadata::decoder_io()`, never serialized separately. The
legacy top-level `model.io` block is import-only: it is read only when a document
carries no workflow (the `genai_config.json` import path) and is rejected beside one.

## Workflow boundary

`workflow.inputs` and `workflow.outputs` are named, typed package values. Every value
has a `TensorContract` (`dtype`, `rank`, optional symbolic shape). Inputs identify a
versioned runtime role or remain opaque application values. Outputs identify their
role and whether they are pre- or post-adapter.

## Components and graph

`workflow.components` declares ONNX artifacts, native implementations, runtime bindings,
or manifest-pinned adapter ABIs with typed ports. Each contract declares an equivalence
class (`bitwise | distribution_preserving | semantic`, defaulting to `semantic`) that
decides whether the runtime may substitute an equivalent implementation on its own. The graph is recursive SSA control flow:

- `invoke` binds named values to component ports;
- `sequence` orders nodes;
- `loop` declares setup, body, condition, maximum iterations, induction value, and
  carried state;
- `branch` selects one case and exports only declared phi results/effect joins;
- `emit` publishes streaming or final package outputs.

`transfer` is internal lowered IR only: the planner introduces it when it assigns
placement, and metadata must not serialize one.

There are no strategies or phases. Control-flow location defines lifecycle. Policy
math—sampling, termination, scheduler/solver steps, masked updates, speculative
acceptance, and state updates—is supplied as ONNX components.

## Effects, state, and serving

State cells have typed contracts, invocation/session scope, an initializer ValueRef,
and explicit loop recurrence. A cell bound to a runtime state service group declares
`management: runtime` and a `release_boundary`; ordinary tensors use SSA liveness.
Reads, writes, RNG, and emits thread linear effect tokens through declared effect
domains, each carrying an independent `retry` class and `speculation_safety` bound.
Session state is lease-protected. Serving contracts declare generic admission,
compaction, row activity, and accepted length without model-family logic and without
any serialized row identity — state groups declare semantic kind, geometry, graph ABI
`aliasing`, reuse, and rollback/snapshot/fork bounds, never a storage mode.

Top-level `adapters` migrates the durable `InferenceMetadata.adapters` and
`LoraTargetManifest` contracts from LoRA PRs #318/#374 into
`onnx-genai.adapters@1`. It supports PEFT+safetensors and ORT `.onnx_adapter`
sources, an authoritative architecture-neutral target manifest, Phase-1 optional
graph inputs, Phase-2 segment routing generalized to ordered composition, exact
SHA-256 verification, and identity-free request-aligned selection that compacts with
every other row-scoped value. The
normative ABI and reuse/adapt/retire matrix are specified in
[WORKFLOW_POLICY_COMPONENTS.md](WORKFLOW_POLICY_COMPONENTS.md#parameter-adapters-lora).

## Manifest and validation

The required manifest pins IR, ONNX opset, adapter ABI, and capability
versions. Capabilities are derived from used features and checked at load. Unknown,
unsupported, unresolved, unordered, or ill-typed documents fail before execution.

Policy artifact port contracts and minimal workflows are specified in
[WORKFLOW_POLICY_COMPONENTS.md](WORKFLOW_POLICY_COMPONENTS.md). The normative contract
is [INFERENCE_METADATA_DECISIONS.md](INFERENCE_METADATA_DECISIONS.md).
