# Inference metadata

Inference metadata is a declarative package IR. Every package — including one that
ships a single ONNX file — expresses its executable graph ABI in exactly one place:
`pipeline.workflow`. Component ports and their roles, `invoke` bindings, state cells,
and `state_service` groups are the sole serialized truth about which port carries
what. `model:` holds package-wide baked facts (attention geometry, vocabulary and
length limits, MoE structure, sharding), not port names.

The flattened decode ABI a single-graph fast path wants is *derived* from that one
representation by `InferenceMetadata::decoder_io()`, never serialized separately.
The retired top-level `model.io` block no longer exists: a package declaring it is
refused at load with an error naming `migrate_model_io`, the offline conversion.
A single decoder is a workflow with one ONNX component, not a second package shape.

## Workflow boundary

`workflow.inputs` and `workflow.outputs` are named, typed package values. Every value
has a `TensorContract` with `dtype` and a required complete `shape`; the shape length
is the tensor rank. `shape: []` is a scalar. Fixed extents and named symbols constrain
dimensions, while each `Any` occurrence is independently unconstrained and never
implies equality with another `Any`. Serialized `rank` is rejected rather than
inferred or migrated. Inputs identify a versioned runtime role or remain opaque
application values. Outputs identify their role and whether they are pre- or
post-adapter.

## Components and graph

Schema `v1.17` adds `graph_weights`, an explicit mapping from package-relative
ONNX graph paths to immutable GGUF or SafeTensors containers:

```yaml
schema_version: v1.17
graph_weights:
  decoder.onnx:
    - path: weights/decoder.gguf
      format: gguf
      renames:
        container.parameter: decoder.parameter
```

Use `format: safetensors` for SafeTensors. Container keys retain their names
unless `renames` maps them to ONNX initializer names. The ONNX graph declares
external initializers with an omitted or empty `external_data.location`; their
dtype and shape must match the container exactly. An optional external length
must also match. Loading does not cast, transpose, reshape, or infer an architecture.
Packed quantized tensors require the graph's declared storage representation and
supported operators. A container alone does not supply an executable ONNX graph.

These bindings require the Native backend; explicit ORT selection is rejected.
All declared paths must resolve to regular files beneath the package root.
Bindings apply to decoder graphs, workflow components and MTP heads. Native
portable session state includes the declared containers' contents in its model
identity: relocating an unchanged package preserves identity, while changing a
container invalidates saved state.

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

## Non-generative decision profiles (v1.10)

`profiles.<name>.decision` declares the prompt, option bound, variant ensemble,
confidence formulas and score policy for a `kind: decision`, `version: '1'`
profile. This core field requires schema v1.10; unknown required profiles and
unsupported versions fail before execution. The workflow supplies calibrated
floating probabilities, independent of model architecture or execution provider.
See [the SystemOne contract](../SYSTEMONE.md) and the executable
`tests/fixtures/onnx_genai_workflows/decision` package for precise input/output
roles, shape/resource admission, and remaining multimodal/entry-point work.

### Explicit infill context (core schema v1.18)

`package.infill` declares exact tokenizer control spellings and ordering.
`extra_context_template` is optional; omit it when extra file context is not
part of the model's prompt format. Budgets remain server configuration.

```yaml
schema_version: "1.18"
package:
  infill:
    prefix_token: "<model-prefix>"
    middle_token: "<model-middle>"
    suffix_token: "<model-suffix>"
    order: psm
    additional_stop_tokens: []
    extra_context_template: "{% for file in input_extra %}{{ file.filename }}\n{{ file.text }}\n{% endfor %}"
```

The example spellings are placeholders for exact registered package tokens,
not defaults. The authored template receives ordered filename/text strings and
renders only text; filenames never authorize filesystem access. Rendered context
is prepended before either PSM or SPM framing on native `/infill`. A declared
template renders the ordered file-context list even when that list is empty.
Without a declared template, an absent or empty list preserves ordinary framing;
a nonempty list requires an authored template.

Explicit infill control roles terminate generated FIM output. Extra terminators
come only from `additional_stop_tokens` and the package EOS facts; token spellings
such as file separators have no implicit stopping meaning in this declaration.

`extra_context_template` serializes the native `/infill` file-context list,
including an empty list. OpenAI completion `suffix` and engine FIM convenience
methods use the declared control roles without this wire-specific template.
Prepared-token FIM execution can carry the already rendered context.
