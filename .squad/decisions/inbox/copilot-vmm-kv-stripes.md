### 2026-08-06: VMM KV reserves full context but exposes bucketed strides
**By:** Copilot
**What:** Native CUDA KV bindings can reserve the full-context address range under the VMM allocator while exposing only the current bucket as their physical shape; growth commits the next bucket and repacks valid prefixes in place.
**Why:** Full-context strides committed one granule per head and made the arena worse. Bucketed strides preserve CUDA graph capture; floor claims must wait for #694 because ledger-refusal passes made the sweep non-monotonic.
