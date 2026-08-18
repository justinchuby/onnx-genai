---
title: Runtime Contracts
aliases:
  - Contract Map
  - Runtime Invariants
tags:
  - contracts
  - architecture
  - invariants
status: maintained
created: 2026-08-17
updated: 2026-08-17
---

# Runtime Contracts

> [!summary] Question answered
> Which cross-layer promises prevent individually reasonable components from composing into an incorrect runtime?

A contract states what one layer may assume about another. It is stronger than a
code comment when encoded in types, tests, capability tokens or ABI structure.

## Contract families

| Contract | Producer | Consumer |
|---|---|---|
| Model semantics | metadata/package/exporter | engine, loader, scheduler, backend |
| Graph semantics | loader/IR/shape inference | optimizer, placement, EP |
| EP claim | EP capability/compile | session executor |
| Tensor ownership | allocator/session/EP | kernels and bindings |
| Async ordering | EP streams/fences | executor and release path |
| Memory capacity | authority/Governor | holders and mechanisms |
| Persistent state | model/engine/backend | checkpoint, fork, migrate, resume |
| Plugin ABI | ABI crate/host | dynamic provider |
| Public API | engine | CLI/server/Python/C clients |

## Design rules

### Make invalid states unrepresentable

Use:

- newtypes for IDs, lengths, offsets and token/page counts;
- owned handles for ownership;
- borrows for temporary views;
- capability values for resolved optional behavior;
- enums for explicit state transitions.

A test should validate behavior, not compensate for an API that allows arbitrary
cross-device or cross-owner combinations.

### Reservation before effect

For state-visible memory changes:

```text
plan → reserve → provisional execute → commit
```

Failure before commit restores old state and returns provisional capacity.
Waiting must not occur while holding partial scarce resources or governance locks.

### Holder chooses victim

The authority can request bytes back; it cannot safely identify a weight, KV page
or in-flight buffer to delete. Policy lives with the holder that understands
pinning, recompute cost and execution state.

### Claims are honest

An EP or backend claim is a promise that the compiled path supports the resolved
opset, shape, dtype, layout and required capability. Unsupported is a valid,
diagnostic result; silent semantic fallback is not.

### State commits together

Generation state can include KV, recurrent/conv state, sampler/search state and
request progress. A migration/checkpoint/step must not expose a mixture of old
and new components.

### ABI owns lifetime explicitly

Every cross-module pointer needs:

- one owner;
- a release operation in the correct module;
- a validity interval;
- version/layout negotiation;
- panic and error containment;
- unload pinning while callbacks or objects remain.

## How contracts are enforced

1. Rust type system and borrowing.
2. Constructor validation.
3. Capability negotiation.
4. Focused invariant tests.
5. End-to-end conformance/parity tests.
6. Runtime counters that expose underflow/unaccounted bytes.
7. Versioned ABI records.
8. Actionable failure at the earliest boundary.
9. Model checking plus trace refinement for concurrency protocols.

> [!warning] A green build is not integration evidence
> Two components can compile against duplicated or disconnected contracts while
> never exercising each other. Workspace membership, round-trip tests and one
> canonical contract definition are part of correctness.

## Formal sources

- [`RULES.md`](../../RULES.md)
- [`MEMORY_MANAGEMENT_MODEL_DESIGN.md`](../../docs/memory/MEMORY_MANAGEMENT_MODEL_DESIGN.md)
- [`NXRT_ABI.md`](../../docs/architecture/NXRT_ABI.md)
- [`EP_CONFORMANCE.md`](../../docs/execution/EP_CONFORMANCE.md)
- [`MODEL_METADATA.md`](../../docs/genai/MODEL_METADATA.md)
- [TLA+ model index](../../specs/tla/README.md)

## Related notes

- [[execution/Execution Provider Contract]]
- [[memory/Memory Management for Beginners]]
- [[metadata/Metadata Driven Runtime]]
- [[execution/Plugin Execution Providers]]
- [[contracts/Formal Verification with TLA+]]
