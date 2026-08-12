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
