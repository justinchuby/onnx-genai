# Routing

Maps work domains to team members. The coordinator uses this to dispatch.

## Pods (route by pod first, then to the owning member)

| Pod | Members |
|-----|---------|
| 🚀 CUDA & Perf | Deckard, Batty, Leon, Sebastian |
| 🧩 Models & Export | Sapper (+ Deckard/Batty) |
| 🍎 Metal / MPS | Nabil, Mariette, Coco, Freysa |
| 🧮 CPU & Edge | Resch (Intel), Iran (Mac), Luba (ARM/QNN), Isidore (Bindings) |
| 🌐 Server / API | Rachael, Zhora |
| 🔎 Quality & Safety | Pris, Gaff, Luv, Chew, Holden, Rai, Fact Checker, Challenger |

## Domains

| Domain / Signal | Owner |
|-----------------|-------|
| Architecture, design decisions, metadata standard, scope, code review | Roy (Lead) |
| KV cache, paged memory, prefix trie, tiered storage, ORT sessions, backend, tokenizers, Rust internals | Deckard (Systems) |
| Preprocessing (image/audio/tiling), Mobius model build/GGUF/EP, InferenceMetadata emission | Sapper (Systems) |
| Generation engine, scheduler, continuous batching, speculative decoding, logit chain, sampling | Batty (Engine) |
| KV bridge, static-cache/GQA share-buffer, device-resident buffers, IoBinding, fp16/Q4 tensors | Leon (Engine) |
| HTTP server, OpenAI-compatible API, request/response handling, streaming | Rachael (Server) |
| OpenAI endpoints (embeddings/audio/logprobs), streaming, sessions, API fidelity | Zhora (Server) |
| Tests, benchmarks, correctness, edge cases, fixtures | Pris (Tester) |
| Code review, quality, clarity, maintainability, extensibility verdicts | Gaff (Reviewer) |
| Code review — correctness/safety/API-contract gates | Luv (Reviewer) |
| Code review — numerics/precision (fp16/Q4/sampling), model-conversion fidelity | Chew (Reviewer) |
| Metal EP integration (ORT plugin-EP C ABI, kernel registration, partitioning) — `../onnxruntime-mlx` | Nabil (Metal EP) |
| Metal/MPS compute kernels (MatMulNBits, GQA/attention, softmax, RoPE, RMSNorm) | Mariette (Metal) |
| Metal/MPS data/quant kernels (GatherBlockQuantized, quantize/dequant, KV, elementwise) | Coco (Metal) |
| Metal EP correctness + benchmarking + E2E via onnx-genai | Freysa (Metal) |
| Performance, throughput, latency, batched serving, KV efficiency, benchmarks | Sebastian (Perf) |
| Intel/AMD x86-64 CPU EP perf (AVX2/AVX-512/VNNI), MLAS interplay, int8 DP4A GEMV | Resch (CPU & Edge) |
| Apple Silicon / macOS CPU EP perf (NEON, Accelerate/AMX), aarch64-apple-darwin hot paths | Iran (CPU & Edge) |
| ARM64 CPU perf (NEON/SVE), Qualcomm QNN NPU execution provider, edge/Windows-on-ARM | Luba (CPU & Edge) |
| Language bindings (C ABI/PyO3/Swift/Kotlin), mobile/edge cross-compilation & packaging | Isidore (CPU & Edge) |
| Security, FFI/unsafe audit, server hardening, supply-chain, untrusted-input safety | Holden (Security) |
| Memory, decisions, session logs | Scribe |
| Work queue, backlog, keep-alive | Ralph |
| RAI review, content safety | Rai |
| Claim verification, devil's advocate, pre-mortem | Fact Checker |
| Counter-intuitive measurements, direction-setting results, "was the feature actually on?" | Challenger |

## Notes

- Cross-cutting changes (touching multiple crates) → Roy triages, then fans out.
- Metadata parsing/validation lives in `onnx-genai-metadata` → Deckard, with Roy on standard conformance.
- Performance-critical hot paths → the owning dev pairs with Pris for benchmarks.
