# Error and logging conventions

This guide records patterns already present in the Rust workspace. It is a
contributor reference, not a replacement for [`RULES.md`](../../RULES.md), whose
error-message requirements apply to every failure path.

## Error handling

### Define errors at the crate boundary

Most library crates expose a crate-specific error enum derived with
[`thiserror::Error`](https://docs.rs/thiserror):

- `onnx-genai-ort` defines `OrtError` and `pub type Result<T> =
  std::result::Result<T, OrtError>` in
  [`crates/onnx-genai-ort/src/error.rs`](../../crates/onnx-genai-ort/src/error.rs).
- `onnx-runtime-session` keeps `SessionError` and its `Result<T>` alias in its
  private `error` module in
  [`crates/onnx-runtime-session/src/lib.rs`](../../crates/onnx-runtime-session/src/lib.rs).
- `onnx-runtime-loader` defines a focused `LoaderError` in
  [`crates/onnx-runtime-loader/src/error.rs`](../../crates/onnx-runtime-loader/src/error.rs).

Follow that shape when a crate owns a coherent public/fallible boundary:
`#[derive(Debug, thiserror::Error)]`, a named `*Error` enum, and a local
`Result<T>` alias where it improves signatures. Model failures with variants
and fields rather than opaque strings when callers or users need to distinguish
them.

This is a predominant pattern, not a universal rule. Some code uses
`anyhow::Result` for orchestration and application-style paths; for example,
the native decode shared-KV helper adds context with `with_context` in
[`crates/onnx-genai-engine/src/native_decode/mod.rs`](../../crates/onnx-genai-engine/src/native_decode/mod.rs).
Do not introduce a public enum merely to wrap an internal one-off operation.

### Preserve causes and useful context

- Use `#[from]` (often with `#[error(transparent)]`) when the enclosing error
  should retain a dependency error unchanged. `SessionError` does this for
  loader, EP, IR, optimizer, and shape-inference errors in
  [`crates/onnx-runtime-session/src/lib.rs`](../../crates/onnx-runtime-session/src/lib.rs).
  `OrtError::Io(#[from] std::io::Error)` is a smaller example in
  [`crates/onnx-genai-ort/src/error.rs`](../../crates/onnx-genai-ort/src/error.rs).
- Add a variant or `map_err` context when crossing an operation boundary. The
  ORT shared-batch path maps failures to messages such as
  `bind input_ids '{name}'` in
  [`crates/onnx-genai-ort/src/decode/shared_batch.rs`](../../crates/onnx-genai-ort/src/decode/shared_batch.rs).
- Make displays actionable. Loader errors commonly state what failed, why, and
  how to fix it; see `NodeVersionNotRepresentable` and model-validation
  variants in
  [`crates/onnx-runtime-loader/src/error.rs`](../../crates/onnx-runtime-loader/src/error.rs).
  Include relevant names, shapes, dtypes, paths, options, and rejected values
  when that information helps diagnose the failure.
- At `anyhow` boundaries, use `context`/`with_context` for the operation and
  resource. Do not erase the underlying source with a generic "failed".

### Return errors for recoverable conditions

Library code returns `Result` for data-dependent or recoverable failure:
missing model inputs, malformed configuration, conversion/overflow, allocation,
and device/runtime failures. Propagate existing failures with `?`; convert
`Option` or narrow conversion failures with `ok_or_else`/`map_err` and a
specific error.

Recent decode and tensor paths illustrate the rule:

- Decode validation turns absent cache state and capture graph IDs into
  `OrtError`/`CapturedStepError` with `ok_or_else`, rather than panicking, in
  [`crates/onnx-genai-ort/src/decode/dynamic.rs`](../../crates/onnx-genai-ort/src/decode/dynamic.rs).
- Decode byte-slice conversions use `map_err(|_| anyhow::anyhow!(...))?` in
  [`crates/onnx-genai-engine/src/native_decode/cuda.rs`](../../crates/onnx-genai-engine/src/native_decode/cuda.rs).
- `Tensor::try_clone` reports shape, allocation, and tensor-creation failures;
  callers in fallible control flow should choose it over `Clone`. See
  [`crates/onnx-runtime-session/src/tensor.rs`](../../crates/onnx-runtime-session/src/tensor.rs).

### Panics are for proven invariants only

Do not use `unwrap()` or `expect()` for recoverable, input-dependent, or
runtime/device-dependent failures in library paths. The codebase retains
`expect()` for ownership/lifetime invariants that cannot fail in the operation's
contract, and its messages say why. For example,
`SharedTensorBuffer::buffer()` documents that its buffer is taken only during
`Drop`, and `Tensor::clone()` explicitly delegates to the fallible clone before
asserting re-allocation of identical bytes in
[`crates/onnx-runtime-session/src/tensor.rs`](../../crates/onnx-runtime-session/src/tensor.rs).

Before adding `expect()`, be able to explain both the invariant and why external
model data, configuration, allocation, or device state cannot violate it. If
that proof is not durable, return a contextual error instead.

## Logging

### Framework and message shape

The workspace uses [`tracing`](https://docs.rs/tracing), not the `log` macros:
production logging uses `tracing::{info, debug, warn, error}!`; the workspace
pins `tracing` and `tracing-subscriber` in `Cargo.toml`. The server, router, and
CLI install `tracing_subscriber::fmt()` with an `EnvFilter` whose default is
`info`; see:

- [`crates/onnx-genai-server/src/main.rs`](../../crates/onnx-genai-server/src/main.rs)
- [`crates/onnx-genai-router/src/main.rs`](../../crates/onnx-genai-router/src/main.rs)
- [`crates/onnx-genai-cli/src/lib.rs`](../../crates/onnx-genai-cli/src/lib.rs)

Use a stable human-readable event message plus structured fields. Existing
calls use fields such as `error = %err`, `node = %id`, `chunk_hits = ...`, and
`model = ...`; see the server's registry failure in
[`crates/onnx-genai-server/src/routes/mod.rs`](../../crates/onnx-genai-server/src/routes/mod.rs)
and connector diagnostics in
[`crates/onnx-genai-engine/src/connector_bridge.rs`](../../crates/onnx-genai-engine/src/connector_bridge.rs).
The inspected calls rely on the default module target; no project-wide custom
`target:` naming convention is established.

Use a span for a request or operation that has correlated fields over time. The
server wraps each request in `tracing::info_span!("http.request", ...)`, then
records `status` and `latency_ms` on completion in
[`crates/onnx-genai-server/src/lib.rs`](../../crates/onnx-genai-server/src/lib.rs).

### Levels seen in the code

| Level | Existing use | Use it for |
| --- | --- | --- |
| `error!` | Server registry operation failed | A failed operation that is being surfaced/handled as an error. |
| `warn!` | Unhealthy nodes, unavailable optional metadata/providers | Degraded, unexpected, or fallback behavior that continues. |
| `info!` | Server start, model load/unload, enabled provider | Lifecycle and significant operator-visible state changes. |
| `debug!` | Missed polls and KV connector fallback details | Recoverable diagnostic detail useful while troubleshooting. |
| `trace!` | No `tracing::trace!` call was found in the inspected workspace logging sites | There is no local example to copy; keep any new high-volume detail deliberate and review its cost/data. |

Do not log secrets, credentials, prompts, PII, token streams, or full tensor
contents. The existing events favor operational identifiers, dimensions/counts,
paths, statuses, and formatted errors rather than payload dumps. No separate
repository-wide sensitive-logging policy was found during this audit, so this is
an observation of the current restraint rather than a claim that payloads are
safe to emit.

## Checklist for new code

- Pick or reuse the crate's error type; add a `Result` return for recoverable
  library failures.
- Preserve source errors with `?`, `#[from]`, `map_err`, or `Context`.
- Include the operation and the useful resource/value context in the error.
- Convert absent/invalid runtime state with `ok_or_else` or `map_err`; do not
  panic because model data or a device call is unexpected.
- Use `expect()` only for a documented, true invariant, with a clear message.
- Emit structured `tracing` events with a concise message and fields.
- Choose `warn!` for continued degradation and `debug!` for recoverable detail;
  reserve `error!` for failures that matter to operators.
- Avoid payload and sensitive-data logging; prefer identifiers, counts, shapes,
  statuses, and safe error text.
