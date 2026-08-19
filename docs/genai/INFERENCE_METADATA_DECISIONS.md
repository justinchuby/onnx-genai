# Inference metadata decisions

Status: **Normative design direction**

This document records the approved boundary between portable model metadata,
runtime behavior, caller configuration, and deployment artifacts. It supersedes
contrary design statements elsewhere in this branch. Schema and runtime changes
must converge on these decisions.

## Scope and compatibility

1. The typed workflow IR is the sole executable metadata design. Other pipeline
   proposals may inform missing scenarios but are not compatibility targets.
2. The workflow uses structural `sequence`, `invoke`, `loop`, `branch`, and
   `emit` control flow. Phases and strategies have no execution semantics.
3. Existing `genai_config.json` packages may be imported into the new metadata
   when their semantics can be represented. Exporting new metadata back to
   `genai_config.json` and preserving backward compatibility are not goals.
4. The schema has a strict core. Unknown core fields fail validation. Explicit
   extension containers may preserve optional extensions.
5. Model facts, generation workflow, preprocessing, pooling profiles, and
   optional plan references remain in one metadata document for now. Their
   internal version boundaries must permit a future lossless split.
6. Non-generative tasks such as embedding, reranking, classification, and
   reward scoring use task-specific executable profiles that share the common
   model/package facts. They do not extend the generation workflow vocabulary.

## Component implementations

7. Metadata declares component semantics, not one universal implementation
   language. ONNX, native code, and runtime bindings are all valid.
8. ONNX is preferred when it enables portable composition and efficient
   graph-level optimization. Grammar engines, parsers, and other algorithms
   with mature high-performance native implementations need not be encoded as
   ONNX.
9. The runtime may freely choose an equivalent implementation of declared
   semantics. Explicit capability negotiation and standardized implementation
   diagnostics are not metadata requirements.
10. Preprocessing and generated-input behavior use typed semantic contracts.
    Implementations may be ONNX or native. Runtimes must not infer behavior
    from a model-family name.
11. Application policy inputs such as grammar or JSON Schema are request data.
    Metadata carries the tokenizer, vocabulary, special-token, and parser facts
    required to interpret those requests correctly.

## Runtime-owned execution

12. KV allocation, paging, shared buffers, compaction, adapter lifecycle,
    placement, transfers, execution islands, graph capture, memory planning,
    cache identity, and distributed state transfer are runtime mechanisms.
13. Metadata declares semantic state facts and real graph ABI constraints. It
    does not select `paged`, `shared_buffer`, or `separate` storage, a slot
    allocation algorithm, a device, or an execution provider.
14. Device transfers are planner-lowered internal operations, not serialized
    workflow nodes.
15. Execution-island, capture, and memory plans may be persisted in separate,
    platform-specific, disposable deployment files. They are not portable model
    metadata.
16. Memory budget, concurrency, cache capacity, quality-of-service, tiering,
    deadlines, and other deployment policy are caller or runtime configuration.
17. Metadata exposes static model and state geometry, not benchmark-derived cost
    or admission predictions.
18. Artifact integrity, signatures, provenance, and trust policy belong to the
    distribution layer. Existing format-specific checksums may remain part of a
    loader ABI, but there is no global metadata digest requirement.
19. Determinism levels and execution diagnostics are runtime/API concerns and
    are not standardized by inference metadata.

## Batch and request identity

20. Physical batch rows, scheduler slots, epochs, block tables, and paged
    attention sequence handles are runtime-private.
21. Metadata does not contain `slot_ids`, `request_epochs`, or another row
    identity representation. The runtime is responsible for moving every
    per-row tensor and state consistently during compaction or slot reuse.
22. Workflow contracts describe per-row tensor semantics without exposing the
    scheduler's identity mechanism. Ragged output association is provided by
    the runtime output API, not by serialized row IDs.
23. Cache-key construction, request salting, tenant isolation, and cross-process
    cache identity are runtime/security responsibilities.
24. Prefill/decode and encoder/decoder state interchange are private distributed
    runtime protocols, not portable metadata contracts.
25. Metadata may declare that a typed encoder or multimodal value can be
    supplied externally and must describe its placeholder/splice semantics.
    Remote caching, identity, and transport remain runtime-owned.

## State

26. Metadata declares per-layer or per-group semantic state kinds and their
    mapping. Full attention, sliding attention, MLA, recurrent/SSM, cross
    attention, encoder state, and future kinds must not be conflated with a
    physical storage mode.
27. State contracts declare legal rollback, snapshot, and fork capabilities,
    bounds, and dependency/cascade semantics. Runtime policy determines when to
    use them.
28. Runtime-managed and external state declares semantic lifetime and a logical
    release boundary. Ordinary tensors use SSA liveness and require no manual
    release annotation.
29. Session metadata declares only the typed state and mutation semantics needed
    for correctness. Session IDs, storage, TTL, locking implementation,
    migration, and retention policy are runtime/server concerns. This boundary
    should remain minimal and may be reduced if a runtime-private implementation
    proves sufficient.
30. Internal state is private by default. Portable state is exported only
    through an explicit versioned checkpoint adapter; an internal state cell is
    not directly a public package output.
31. Each external effect declares the minimum retry-relevant class, such as
    pure, idempotent, transactional, or non-retryable. Timeout, cancellation,
    retry, and recovery orchestration belong to the runtime/server.
32. Interactive world-model, robotics, and streaming observations enter as
    separate workflow invocations with session state between invocations.
    Portable workflow IR does not add a network-aware `receive` or `await`
    operation.

## Generation and speculative execution

33. Packages carry authoritative generation defaults and model token facts.
    Callers may override supported generation defaults per request.
34. Speculative metadata declares proposer/target compatibility facts, typed
    ports, shared-state or shared-weight bindings, vocabulary compatibility,
    and rollback requirements. Proposal width, tree shape, scheduling, kernels,
    and whether speculation is enabled are runtime decisions.
35. LoRA metadata contains the authoritative target manifest, artifact bindings,
    and request-selection contract. Runtime and caller code own loading,
    caching, budgeting, eviction, and the concrete application implementation.
    Runtime execution must not discover targets from model-family conventions.
36. KV and other model-state quantization are fixed by the published package.
    A runtime does not select a different state-quantization mode from a caller
    memory budget. Different modes require distinct packages or profiles.

## Distributed execution

37. Metadata declares legal tensor, pipeline, and expert sharding facts,
    including shard axes, replication requirements, expert identity, and
    cross-stage state ports.
38. The caller/runtime chooses TP, PP, and EP degree, device mapping, placement,
    and collective backend.
39. Portable metadata does not standardize a cross-runtime KV/cache wire format.

