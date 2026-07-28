# Decision: MLX EP logging — `log` facade, not `tracing`

**Date:** 2026-07-27
**Author:** Nabil (ORT Plugin EP Engineer — Metal)
**PR:** https://github.com/justinchuby/onnxruntime-mlx/pull/9
**Status:** Proposed (PR open, not merged)

## Context

The `onnxruntime-mlx` plugin EP (a Rust `cdylib` loaded into ORT) had 12 raw
`eprintln!` sites with no structured logging. A stale build was polluting
profile captures with per-subgraph prints on every decode step.

## Decision

Use the **`log` crate** (facade only) with a minimal 90-line in-crate logger,
rather than the `tracing` ecosystem used by onnx-genai.

## Rationale

| Concern | `log` | `tracing` |
|---------|-------|-----------|
| Plugin isolation | ✅ facade-only; dylib owns its statics | ⚠️ subscriber model adds complexity in a shared-process context |
| Host conflict risk | None — `set_boxed_logger` is private to the dylib's copy | The subscriber global could interfere if the host also uses tracing |
| Dependency weight | ~0 transitive deps | 5–10 transitive deps (tracing-core, tracing-subscriber, etc.) |
| Benefit of subscriber | N/A — host cannot install a subscriber into the plugin anyway | None — the plugin is the only consumer of its own log output |

The sibling onnx-genai repo uses `tracing` because it's an *application* that
benefits from span-based structured logging and async instrumentation. A plugin
dylib has fundamentally different constraints: it must be minimal, must not
conflict with the host, and has no subscriber to route events to.

## Levels

- Default: **Warn** (panics + user-visible failures only)
- `ONNXRUNTIME_EP_MLX_VERBOSE=1`: Info
- `ONNXRUNTIME_EP_MLX_TRACE=<path>`: Debug
- `RUST_LOG=onnxruntime_ep_mlx=<level>`: explicit override

## Implications

- Future `eprintln!` additions to the MLX EP crate should use `log::*!` macros
  at the appropriate level.
- The `[rust-mlx-ep]` prefix is now added by the logger, not by each call site.
- The session summary and slowest-ops table are **deliberate user-facing features**
  (not stray debugging) — gated at `info!` behind `VERBOSE=1` or `TRACE=`.
