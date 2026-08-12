# Freysa — History Archive

## 2026-07-12: Joined
Hired as MPS Perf & Testing engineer for the new Apple Metal EP for ONNX Runtime. Owns per-kernel correctness (vs ORT CPU reference), benchmarking, Metal profiling, and E2E testing through the onnx-genai runtime. Targets: beat llama.cpp Metal / LM Studio / Foundry Local on Apple Silicon. Pairs with Sebastian.

## 2026-07-14T19:05:00Z — Pipeline API seams
`ChatTemplate::builtin_default`, `Engine::tokenize`, `embed_text*` recorded in decisions; Holden review GREEN for commit `ecba2c1`.

## 2026-07-15T00:00:00Z — Cross-agent session update
Added missing C1 shape handlers and initial DLPack import support; consolidated in the July 15 coverage/interoperability work.

## 2026-07-16T00:00:00Z — Performance-and-design wave
Reviewed the unified string-serde surface as approve-with-notes.

## 2026-07-16T00:00:00Z — onnx-rs Python binding review cycle
Rejected Batty's initial `onnx_rs` binding for lossy paths, an `exists()` preflight, and swallowed `__fspath__` exceptions. Cleared Deckard's `5b348b5` revision after targeted Rust tests and six Python regressions verified lossless paths and native filesystem errors.

## 2026-07-21 — Scribe reconciled
Perf campaign inbox consolidated; key decisions now in `.squad/decisions.md`.

## 2026-07-22T14:59:36+0000 — WP-B landed
WP-B: Freysa's raw-protobuf admission rejection resolved in the final WP-B3 path.

## 2026-07-28T09-10-28+00-00 — PR #338 review
Approved Luv's #67 CUDA `Pad`/`Range` batch after H200 GPU 2 passed 174/174 parity cases, coverage gate passed, content-corrupting mutation probe failed as expected, default-target warnings-denied Clippy clean. PR #338 merged as `c59383db`.

## 2026-08-11 — EP assignment proof + PR #762 third corrective wave

Added `session.disable_cpu_ep_fallback=1` to `conformance_setup()` in `plugin_ort_e2e.rs`. All 22 conformance tests + stress test now prove our EP actually claimed nodes. `conformance_mixed_partition` exempted (intentionally tests partition with fallback). `mixed_partition` correctly fails with fallback disabled — non-vacuity proved.

**Lesson:** Verify an API's absence before deferring on it. Check the generated bindings.
