# Workflow runtime

Composite inference packages execute through `pipeline.workflow`. The runtime is a
generic interpreter for typed SSA values and recursive `invoke`, `sequence`, `loop`,
`branch`, and `emit` nodes. It contains no model-family planner or scheduler/sampler
implementation. `transfer` is internal lowered IR the planner introduces when it
assigns placement; it is never serialized in metadata.

The normative contract is
[INFERENCE_METADATA_DECISIONS.md](INFERENCE_METADATA_DECISIONS.md).

Components perform tensor and policy math. A component may be ONNX, native, or a
runtime binding; each contract declares an equivalence class
(`bitwise | distribution_preserving | semantic`, defaulting to `semantic`) that decides
whether the runtime may substitute an equivalent implementation without the caller
asking. Stable adapters are selected through a versioned ABI registry. Package inputs bind request roles or opaque application
tensors; package outputs are emitted according to their boundary contracts.

State and side effects are explicit. Loop-carried cells declare recurrence, session
cells use per-session leases, RNG is counter state, and every state mutation or emit
threads a linear effect token. Branches join mutually exclusive SSA/effect successors
through declared phi mappings.

Generic serving services—admission, continuous batching, compaction, state paging and
slot allocation, resource governance, and device placement—remain runtime mechanisms
and are driven by typed workflow values rather than model-family dispatch. Metadata
declares only semantic state kinds, geometry, graph ABI facts, and reuse/rollback
bounds; it never selects a storage mode, an allocator, or a device.

## Encoder and media batching

The v1.1 workflow schema can describe generalized image, video, audio, and text
encoder batches. `batch_capacity` is the component's sole authored claim that a
valid group is equivalent to per-item execution. Dense rows use
`request_aligned`; right-padded dimensions name `valid_lengths`; ragged item
counts use one axis-0 `token_packed` run with one or two ownership levels.
Uniformity and static footprint bounds are keyed by shape symbol.

Compatibility and request-local spans are derived from those contracts.
Whether to group, the target size, device-memory limits, and backend readiness
are runtime policy/evidence. The old profile/model/capability batching hints are
retired rather than kept as competing sources of truth.

Current status is metadata-only: parsing and semantic validation are shipped,
but the interpreter, preprocessors, scheduler, and backends do not yet assemble
or execute generalized encoder groups. See
[ENCODER_BATCHING.md](ENCODER_BATCHING.md) for the acceptance matrix and
explicit runtime gaps.

## Batched policy islands

Sampler and termination components may enter an execution island only through their
version 2 per-row ABI. A sampler must bind `logits`, `active`, `temperature`, `top_k`,
`done`, `top_p`, `min_p`, `seed`, `counter`, `token`, and `next_counter`, with contract
parameters `batching: per_row` and `inactive_rows: preserve`. Termination additionally
binds per-row EOS lengths and iteration limits; state updates bind `active` and `done` masks.

These are semantic requirements, not scalar-broadcast conveniences. Every row owns
its sampling parameters, ragged logical lengths, deterministic RNG seed/counter,
termination state, and KV/state update decision. Inactive rows consume no RNG,
preserve semantic state, and produce no compacted emit event. Stable max-batch device
buffers may contain inactive capacity, but capture and super-island plans must remain
valid as the active-row set changes without specializing the ONNX graph to batch one.

Ragged emission is derived, not declared: an output is row-wise when it is ragged — when
a `valid_length` or a `when` guard is present — and row-wise-ness belongs to the output,
so every emit of that output publishes rows. Guards and valid lengths are zipped with
tensors before compaction. A row-wise output must be `request_aligned`.

Metadata serializes no row identity. Compaction permutes every request-aligned value and
every row-scoped component together; each row-scoped native or stateful component must
implement the mandatory `compact(permutation)` / `release(row)` ABI
(`onnx_genai_engine::RowScopedState`). Beam and speculative row expansion uses a
runtime-minted `row_selection` gather that carries no scheduler identity. Consumers
enumerate structured rows through the output API; compatibility keys such as
`tokens.row.3` are positional serialization details and are not parsed for behavior.
