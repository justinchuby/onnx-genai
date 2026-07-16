# Freysa — History

## 2026-07-12: Joined
Hired as MPS Perf & Testing engineer for the new Apple Metal EP for ONNX Runtime (`../onnxruntime-mps`). Owns per-kernel correctness (vs ORT CPU reference), benchmarking, Metal profiling, and E2E testing through the onnx-genai runtime (`ONNX_GENAI_EP=metal`). Targets: beat llama.cpp Metal / LM Studio / Foundry Local on Apple Silicon. Reuses the onnx-genai benchmark harness (`scripts/compare_runtimes.sh`, `compare.rs`). Pairs with Sebastian. Correctness (coherent output) gates every perf claim.

- 2026-07-14T19:05:00Z — Pipeline API seams (`ChatTemplate::builtin_default`, `Engine::tokenize`, `embed_text*`) recorded in decisions; Holden review GREEN for commit `ecba2c1`.

## 2026-07-15T00:00:00Z — Cross-agent session update

- Added missing C1 shape handlers and initial DLPack import support; both were consolidated in the July 15 coverage/interoperability work.

### 2026-07-16T00:00:00Z — Performance-and-design wave
Reviewed the unified string-serde surface as approve-with-notes.
