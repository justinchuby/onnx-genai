# Freysa — History

## 2026-07-12: Joined
Hired as MPS Perf & Testing engineer for the new Apple Metal EP for ONNX Runtime (`../onnxruntime-mps`). Owns per-kernel correctness (vs ORT CPU reference), benchmarking, Metal profiling, and E2E testing through the onnx-genai runtime (`ONNX_GENAI_EP=metal`). Targets: beat llama.cpp Metal / LM Studio / Foundry Local on Apple Silicon. Reuses the onnx-genai benchmark harness (`scripts/compare_runtimes.sh`, `compare.rs`). Pairs with Sebastian. Correctness (coherent output) gates every perf claim.

- 2026-07-14T19:05:00Z — Pipeline API seams (`ChatTemplate::builtin_default`, `Engine::tokenize`, `embed_text*`) recorded in decisions; Holden review GREEN for commit `ecba2c1`.
