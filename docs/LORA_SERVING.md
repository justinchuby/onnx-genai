# Declared LoRA serving

The server implements llama.cpp's `GET /lora-adapters`, `POST /lora-adapters`
and per-request `lora: [{"id": 0, "scale": 0.5}]` selection. Adapter IDs and paths
come from the loaded package's `adapters.artifacts` contract. IDs select
package-declared files; returned catalog objects can be posted back unchanged. This is not a GGUF adapter loader.

GET returns every declared adapter as `{id, path, scale}`. POST replaces the
model's default composition; omitted IDs and zero scales disable an adapter.
Requests capture immutable defaults during preparation. A request's `lora`
replaces those defaults, including `[]` to select the base model. Invalid IDs,
non-finite scales and compositions exceeding the declared
selection/cache capacities fail before updating defaults. Scales are not
restricted to an arbitrary numeric interval. As in the pinned upstream parser,
repeated IDs use their last scale and missing/null/non-numeric scales default
to zero. Compositions are normalized into ID order; display fields are ignored.

All adapters initially have scale 1. Use `--lora-init-without-apply` (or
`NXRT_LORA_INIT_WITHOUT_APPLY`) to start with every scale zero, including when
the catalog is larger than the maximum active composition. Defaults belong to
the admitted model instance; changing them does not modify running requests or
other models. Model-qualified discovery uses `?model=ID&autoload=false`.
Unloading/reloading a model creates fresh defaults from the startup policy.
A model without an admitted adapter service returns an empty catalog and
rejects nonempty selections.

## Executed contract

The current portable executor reads declared float32 JSON adapter bundles.
Each target maps to explicit workflow component/initializer metadata. JSON
`targets.<weight_key>.a` is row-major `[rank, input_features]`; `b` is row-major
`[output_features, rank]` (or `[output_slice.width, rank]` for a declared fused slice). The static factor follows the declared
`alpha_over_rank` or `baked` encoding. Alternative HF PEFT, safetensors and ORT
GenAI source loaders and device-fused application remain unfinished. Selection applies to generation routes, including Responses and Messages, and embedding routes backed by an embedding v1 workflow profile.

`onnx-genai.parameter-overlay@1` requires separate `input`, `base`, and `output`
contract bindings. `input` is the source activation; `base` is the actual base
projection output. The executor computes:

`output = base + sum(scale * static_factor * B * A * input)`

The workflow executes the base graph itself, so non-identity and non-square
projections retain their declared semantics. Sparse target bindings leave other
projections unchanged. Multiple declared fused-output slices add only into their
explicit ranges, preserving untouched columns. Base weights are immutable.
A bare decoder without an adapter-aware workflow refuses a nonempty selection.
Native host execution, including tensor ownership and views, does not load ORT.
ORT remains an explicitly selected reference backend.

Parsed weights retain host-memory leases until eviction or model destruction.
JSON reading/parsing reserves bounded staging from the same model governor;
source growth is rejected. Selection tensors, projection staging and output
buffers also reserve host credit. No-copy views retain the output's credit
until the last view is released. Weight entries and remembered composition
variants are bounded by the declared cache capacity.

## Sessions

A workflow can switch adapters across named turns when its only retained state
is declared token history and it rebuilds dependent tensors by replaying that
history. This preserves the conversation and recomputes it under the new
composition. Workflows retaining other session-scoped tensors currently require
stateless full-conversation requests; they cannot reuse an unqualified cache.
`/props` reports `nxrt_capabilities.lora` and `lora_sessions` from the admitted
executor, plus current default scales in generation settings.

Tests execute real Native ONNX projections and generated tokens, including a
missing-ORT-library subprocess, non-square/non-identity math, default/override
isolation, model discovery, replayed sessions, invalid updates, host-budget
refusal and no-copy lease lifetime.

## Embedding workflows

An embedding package declares its executable tensor explicitly:

```yaml
profiles:
  vectors:
    kind: embedding
    version: '1'
    outputs: {embedding: vector}
    # For a [batch, sequence, features] tensor; omit for a final vector.
    pooling: {kind: mean, axis: 1, normalize: true}
```

`vector` names a floating workflow output. Without `pooling` it is a final
`[features]` or `[batch, features]` vector. With pooling it is
`[sequence, features]` or `[batch, sequence, features]`, and the declared axis
is the sequence axis. Mean, max, first-token (`cls`) and last-token reductions
are executed; no output, axis or model architecture is guessed. One invocation
processes one input sequence. Unsupported required versions, ambiguous profiles,
non-floating outputs and invalid axes fail at package validation.

`/v1/embeddings` and `/embedding`/`/embeddings` snapshot the model's current LoRA
composition once for the whole request. An explicit `lora: []` uses base weights;
a nonempty override replaces defaults. Adapter-aware embeddings execute the
complete admitted workflow before pooling. A legacy decoder hidden-state path
cannot apply arbitrary LoRA selections and refuses them with guidance to declare
an embedding profile. `/props` advertises `lora_embeddings` separately from named
session support. Host float32 tensors are borrowed for pooling; lower-precision
conversion and the reduced vector reserve model-governed host staging first.
