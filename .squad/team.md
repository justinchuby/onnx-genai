# Team

## Project Context

- **Project:** onnx-genai — a Rust inference runtime for generative AI models, built on ONNX Runtime.
- **Description:** Reference implementation of the ONNX Inference Metadata Standard. Standard-driven behavior, agent-first (prefix caching, multi-session, CoW fork, KV rewind), speculative decoding, continuous batching. OpenAI-compatible HTTP + Rust library.
- **Stack:** Rust (edition 2024), Cargo workspace, ONNX Runtime (ORT), HF tokenizers.
- **Crates:** onnx-genai, onnx-genai-metadata, onnx-genai-kv, onnx-genai-scheduler, onnx-genai-engine, onnx-genai-ort (+ort-sys), onnx-genai-server, onnx-genai-bench, onnx-genai-preprocess.
- **Sibling repos:** `../mobius` (ONNX model builder), `../onnxruntime-mlx` (custom Apple Metal/MPS execution provider for ONNX Runtime — new).
- **Requested by:** Justin Chu
- **Created:** 2026-07-12

## Members

| Name | Role | Charter | Badge |
|------|------|---------|-------|
| Roy | Lead | .squad/agents/roy/charter.md | 🏗️ Lead |
| Deckard | Systems Dev | .squad/agents/deckard/charter.md | 🦀 Systems |
| Sapper | Systems Dev (Models & Preprocess) | .squad/agents/sapper/charter.md | 🦀 Systems |
| Batty | Engine Dev | .squad/agents/batty/charter.md | ⚡ Engine |
| Leon | Engine Dev (KV & Buffers) | .squad/agents/leon/charter.md | ⚡ Engine |
| Rachael | Server Dev | .squad/agents/rachael/charter.md | 🌐 Server |
| Zhora | Server Dev (API) | .squad/agents/zhora/charter.md | 🌐 Server |
| Pris | Tester | .squad/agents/pris/charter.md | 🧪 Test |
| Gaff | Code Reviewer / Quality | .squad/agents/gaff/charter.md | 🔎 Review |
| Luv | Code Reviewer | .squad/agents/luv/charter.md | 🔎 Review |
| Chew | Code Reviewer (Numerics) | .squad/agents/chew/charter.md | 🔎 Review |
| Nabil | ORT Plugin EP Engineer (Metal) | .squad/agents/nabil/charter.md | 🍎 Metal EP |
| Mariette | Metal/MPS Kernel Engineer | .squad/agents/mariette/charter.md | 🍎 Metal |
| Coco | Metal/MPS Kernel Engineer | .squad/agents/coco/charter.md | 🍎 Metal |
| Freysa | MPS Perf & Testing | .squad/agents/freysa/charter.md | 🍎 Metal |
| Sebastian | Performance Engineer | .squad/agents/sebastian/charter.md | ⚙️ Perf |
| Isidore | Mobile & Bindings Engineer | .squad/agents/isidore/charter.md | 🔌 Bindings |
| Iran | Mac CPU Optimization Engineer | .squad/agents/iran/charter.md | 🧮 Mac CPU |
| Resch | Intel CPU Optimization Engineer | .squad/agents/resch/charter.md | 🧮 Intel CPU |
| Luba | ARM CPU / QNN EP Engineer | .squad/agents/luba/charter.md | 📱 ARM/QNN |
| Holden | Security Engineer | .squad/agents/holden/charter.md | 🔒 Security |
| Scribe | Session Logger | .squad/agents/scribe/charter.md | 📋 Scribe |
| Ralph | Work Monitor | .squad/agents/ralph/charter.md | 🔄 Monitor |
| Rai | RAI Reviewer | .squad/agents/rai/charter.md | 🛡️ RAI |
| Challenger | Claim Challenger (挑战者) | .squad/agents/challenger/charter.md | 🎯 Challenge |
| Fact Checker | Fact Checker | .squad/agents/fact-checker/charter.md | 🔍 Verifier |

## Sub-Teams (Pods)

The team is organized into specialized pods. The coordinator routes by pod first,
then to the owning member. Roy (Lead) spans all pods; Scribe/Ralph/Rai/Fact Checker
are cross-pod built-ins. Members may act as liaisons into another pod (noted below).

| Pod | Focus | Members |
|-----|-------|---------|
| 🚀 **CUDA & Perf** | CUDA EP kernels, decode engine, KV/buffers, throughput vs ORT — *primary front* | Deckard, Batty, Leon, Sebastian |
| 🧩 **Models & Export** | GLM/DeepSeek/Gemma4/Phi native enablement, Mobius export, preprocessing, metadata | Sapper (+ Deckard/Batty collaborate) |
| 🍎 **Metal / MPS** | Metal EP integration + MPS compute/data kernels + E2E/bench (`../onnxruntime-mlx`) | Nabil, Mariette, Coco, Freysa |
| 🧮 **CPU & Edge** | CPU EP perf across Intel/ARM/Apple Silicon, QNN NPU EP, language bindings & mobile packaging | Resch (Intel), Iran (Mac), Luba (ARM/QNN), Isidore (Bindings) |
| 🌐 **Server / API** | HTTP server, OpenAI-compatible API, streaming | Rachael, Zhora |
| 🔎 **Quality & Safety** | Tests, code review, numerics, security, RAI, verification | Pris, Gaff, Luv, Chew, Holden, Rai, Fact Checker |

**Cross-pod liaisons**
- **Chew** (Quality/Numerics) is the standing precision gate for 🚀 CUDA & Perf and 🧮 CPU & Edge quant work.
- **Pris** (Quality) pairs with the owning dev on any hot-path change to produce the benchmark.
- **Isidore** (CPU & Edge/Bindings) pairs with **Luba** on ARM/Windows-on-ARM cross-compilation and with **Rachael/Zhora** on server-side binding surfaces.
- **Sebastian** (CUDA & Perf) advises 🧮 CPU & Edge on benchmark methodology and portability gates.

**Portability rule (all perf pods):** optimizations must help consumer/edge hardware, not just H200 — every perf claim is backed by a benchmark, and SIMD/NPU/kernel paths must match the scalar/f64 reference within a justified tolerance and be locked with a regression test.
