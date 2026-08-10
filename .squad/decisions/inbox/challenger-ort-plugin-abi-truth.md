# Decision: ORT Plugin EP Export ABI — Ground Truth

**Date:** 2026-08-10T20:12:35.793+00:00
**Status:** accepted
**Decider:** Challenger (挑战者), verified from ORT 1.27.0 C headers
**Requested by:** @justinchuby

## Context

Three contradictory claims were made about the ORT plugin EP ABI:
- Claim A (Nabil): export symbol is `CreateEpFactories`
- Claim B (Pris): export symbol is `CreateEpApiFactories`
- Claim C (Pris): end-to-end test impossible because `nm -D` shows only 2 exported symbols

## Decision

1. **Correct export symbols:** `CreateEpFactories` and `ReleaseEpFactory` (both required).
   Source: `onnxruntime_c_api.h:5579` and `onnxruntime_ep_c_api.h:2637,2661`.

2. **End-to-end test IS possible.** `RegisterExecutionProviderLibrary`, `GetEpDevices`,
   and `SessionOptionsAppendExecutionProvider_V2` are members of the `OrtApi` struct
   (since v1.22). They are invisible to `nm -D` by design — the ORT C API is a vtable.

3. **`ort_version_supported` is forward-compatible, not fail-closed.** For Justin's
   fail-closed requirement, add an explicit check in `CreateEpFactories`.

## Consequences

- Nabil's implementation uses the correct symbol name.
- Pris's claim that e2e tests are impossible must be retracted; the full call sequence is documented in `docs/EP_PLUGIN_EXPORT_ABI_TRUTH.md`.
- Tests should use `RegisterExecutionProviderLibrary` → `GetEpDevices` → `SessionOptionsAppendExecutionProvider_V2` → `CreateSession` → `Run`.
