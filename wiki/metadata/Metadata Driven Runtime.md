---
title: Metadata Driven Runtime
aliases:
  - Inference Metadata
  - Model Agnostic Runtime
tags:
  - metadata
  - models
  - contracts
status: maintained
created: 2026-08-17
updated: 2026-08-17
---

# Metadata Driven Runtime

> [!summary] Question answered
> How does the runtime support different model architectures without hardcoding model names and dimensions?

The runtime treats model behavior as data. Inference metadata, graph structure,
ONNX semantics, capabilities and explicit user configuration replace
model-family dispatch.

## What metadata describes

Depending on the package/model, metadata can declare:

- model components and pipeline topology;
- token/embedding input and logits/hidden outputs;
- KV and recurrent state ownership;
- attention/KV geometry and axes;
- tokenizer/chat-template behavior;
- image/audio preprocessing programs;
- generation defaults and constraints;
- speculative proposer contracts;
- runtime-required capabilities;
- execution/placement hints;
- quantization and hardware requirements.

## Capability negotiation

Models list stable capability identifiers. The runtime advertises its supported
set and fails load with the missing capabilities instead of guessing from model
identity.

```text
model requires: [loop_carried_state, multi_axis_positions]
runtime offers: [loop_carried_state]
result: fail clearly; missing multi_axis_positions
```

Capability strings are data and should not become hidden branches by model name.

## Resolution priority

For execution hints, stronger/more local user intent overrides embedded defaults:

1. programmatic builder API;
2. user `execution_hints.json`;
3. inference metadata execution hints;
4. ONNX `onnx_runtime.*` metadata properties.

Conflicting forced constraints should fail rather than choose silently.

## Names versus roles

Tensor port names are data, not semantics. Resolution should prefer:

1. exact declared role/port;
2. a unique structural/shape signal;
3. failure naming the required metadata when ambiguous.

Conventional-name fallback may exist for a narrow compatibility case, but a
declared contract always wins and the fallback must not guess among ambiguous
ports.

## State ownership

Metadata distinguishes:

- token IDs versus input embeddings as sequence source;
- owned versus shared KV;
- positional KV inputs/outputs;
- hidden/recurrent outputs;
- proposer-to-target shared-state mappings.

These declarations let ORT and native backends implement the same generation
semantics without copying model-family assumptions into engine code.

## Schema and forward evolution

`onnx-genai-metadata` provides typed Rust structures, validation and deterministic
JSON Schema generation. Readers should:

- validate required capabilities;
- retain actionable paths to invalid fields;
- define schema-version behavior;
- tolerate explicitly allowed additive evolution;
- reject ambiguity that would change semantics.

## Metadata is not a performance wish list

Hints do not override correctness, physical capability or user policy. A request
for CUDA graph capture, a kernel or a device still passes capability and
compatibility checks.

## Formal sources

- [`onnx-genai-metadata`](../../crates/onnx-genai-metadata/src/lib.rs)
- [`MODEL_METADATA.md`](../../docs/genai/MODEL_METADATA.md)
- [`MODEL_PACKAGE.md`](../../docs/genai/MODEL_PACKAGE.md)
- [`RULES.md`](../../RULES.md)

## Related notes

- [[metadata/Model Packages and Variants]]
- [[contracts/Runtime Contracts]]
- [[api/API Design Principles]]
- [[architecture/Inference Request Lifecycle]]
