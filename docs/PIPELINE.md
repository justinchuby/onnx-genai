# Workflow runtime

Composite inference packages execute through `pipeline.workflow`. The runtime is a
generic interpreter for typed SSA values and recursive `invoke`, `sequence`, `loop`,
`branch`, `transfer`, and `emit` nodes. It contains no model-family planner or
scheduler/sampler implementation.

ONNX components perform tensor and policy math. Stable adapters are selected through
a versioned ABI registry. Package inputs bind request roles or opaque application
tensors; package outputs are emitted according to their boundary contracts.

State and side effects are explicit. Loop-carried cells declare recurrence, session
cells use per-session leases, RNG is counter state, and every state mutation or emit
threads a linear effect token. Branches join mutually exclusive SSA/effect successors
through declared phi mappings.

Generic serving services—admission, continuous batching, compaction, KV paging/slot
allocation, resource governance, and device placement—remain runtime mechanisms and
are driven by typed workflow values rather than model-family dispatch.

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

Ragged emission is explicit: `emit.row_ids` binds physical rows to semantic `int64[B]`
identities. Guards and valid lengths are zipped with tensors and IDs before compaction.
Consumers enumerate structured rows by output declaration or semantic role; compatibility
keys such as `tokens.row.17` are serialization details and are not parsed for behavior.
