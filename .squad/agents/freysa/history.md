# Freysa — History

## 2026-07-12: Joined
Hired as MPS Perf & Testing engineer for the new Apple Metal EP for ONNX Runtime (`../onnxruntime-mps`). Owns per-kernel correctness (vs ORT CPU reference), benchmarking, Metal profiling, and E2E testing through the onnx-genai runtime (`ONNX_GENAI_EP=metal`). Targets: beat llama.cpp Metal / LM Studio / Foundry Local on Apple Silicon. Reuses the onnx-genai benchmark harness (`scripts/compare_runtimes.sh`, `compare.rs`). Pairs with Sebastian. Correctness (coherent output) gates every perf claim.

- 2026-07-14T19:05:00Z — Pipeline API seams (`ChatTemplate::builtin_default`, `Engine::tokenize`, `embed_text*`) recorded in decisions; Holden review GREEN for commit `ecba2c1`.

## 2026-07-15T00:00:00Z — Cross-agent session update

- Added missing C1 shape handlers and initial DLPack import support; both were consolidated in the July 15 coverage/interoperability work.

### 2026-07-16T00:00:00Z — Performance-and-design wave
Reviewed the unified string-serde surface as approve-with-notes.

### 2026-07-16T00:00:00Z — onnx-rs Python binding review cycle
Rejected Batty's initial `onnx_rs` binding for lossy paths, an `exists()` preflight, and swallowed `__fspath__` exceptions. Cleared Deckard's `5b348b5` revision after targeted Rust tests and six Python regressions verified lossless paths and native filesystem errors.

- 2026-07-21: Scribe reconciled the perf campaign inbox; key decisions are now consolidated in `.squad/decisions.md` under the 2026-07-21 perf campaign section.

### 2026-07-22T14:59:36+0000 — WP-B landed
WP-B landed: Freysa's raw-protobuf admission rejection was resolved in the final WP-B3 path.
