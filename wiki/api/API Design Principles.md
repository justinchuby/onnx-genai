---
title: API Design Principles
aliases:
  - Public API Design
  - Binding Design
tags:
  - api
  - architecture
  - ffi
  - compatibility
status: maintained
created: 2026-08-17
updated: 2026-08-17
---

# API Design Principles

> [!summary] Question answered
> How should one behavior remain coherent across Rust, CLI, HTTP, Python and C without forcing every surface to expose identical mechanics?

## Layered surfaces

| Surface | Primary audience | Design emphasis |
|---|---|---|
| Rust facade | Rust applications | Typed ownership and composability |
| Engine API | Advanced integrations | Full generation policy/control |
| CLI/REPL | Humans and scripts | Discoverability, stable stdout, actionable errors |
| HTTP server | OpenAI-compatible clients | Wire compatibility, streaming, sessions |
| Python | Python/NumPy users | Familiar session API, DLPack where explicit |
| C ABI | Cross-language hosts | Opaque handles, value/vtable ABI, explicit ownership |

The surfaces share semantics, not necessarily identical signatures.

## One semantic core

Prompt processing, generation options, finish reasons, stop behavior and sampling
should converge on engine types. Bindings marshal into that core instead of
reimplementing generation policy.

For example, a foreign sampler replaces terminal token selection while the engine
still applies the configured processor/constraint chain.

## Actionable failure

Every surface should preserve:

- what operation failed;
- which argument/node/shape/dtype/path/device was involved;
- why it was rejected;
- how the caller can fix it;
- the underlying cause where safe.

Rust uses typed crate errors; orchestration may add `anyhow::Context`; C uses
machine-readable status plus a rich message; Python/HTTP map without erasing
diagnostic detail.

## Ownership by language

### Rust

Use ownership, borrowing and newtypes to make invalid states difficult to express.
Fallible device/runtime work returns `Result`.

### C

- opaque handles have matching create/release functions;
- null is checked before dereference;
- every panic is caught at the ABI boundary;
- status/message ownership is explicit;
- caller and library never free each other's unspecified heap allocations.

### Python

- copied NumPy output is the safe default;
- DLPack/zero-copy paths are explicit;
- thread/reentrancy behavior is stated;
- iterator/callback completion semantics must distinguish token events from final
  result state.

### HTTP

- request/response shapes follow the intended OpenAI-compatible contract;
- SSE streaming preserves completion/error semantics;
- debug/admin endpoints are opt-in;
- payloads and secrets are not logged by default.

## Compatibility policy

The repository is pre-release for its own Rust/product APIs: reshape them cleanly
and update all callers instead of accumulating aliases and deprecations.

That freedom does not apply to:

- ONNX semantics/opsets;
- documented model metadata;
- stable C/plugin ABI versions;
- OpenAI-compatible wire behavior promised to users;
- supported Python wheel ABI.

## Explicit behavior beats hidden convenience

- no silent CPU fallback unless explicitly enabled;
- no implicit cross-device transfer in eager APIs;
- capability negotiation before using optional behavior;
- no model-name dispatch where metadata/graph structure is required;
- report a constrained fallback instead of pretending the requested mode ran.

## Formal sources

- [`RULES.md`](../../RULES.md)
- [`onnx-genai` facade](../../crates/onnx-genai/src/lib.rs)
- [`onnx-genai-server`](../../crates/onnx-genai-server/src/lib.rs)
- [`PYTHON.md`](../../docs/architecture/PYTHON.md)
- [`onnx-runtime-capi`](../../crates/onnx-runtime-capi/src/lib.rs)
- [`onnx-genai-capi`](../../crates/onnx-genai-capi/src/lib.rs)
- [`ERROR_AND_LOGGING_CONVENTIONS.md`](../../docs/architecture/ERROR_AND_LOGGING_CONVENTIONS.md)

## Related notes

- [[contracts/Runtime Contracts]]
- [[metadata/Metadata Driven Runtime]]
- [[execution/Plugin Execution Providers]]
