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
| Holden | Security Engineer | .squad/agents/holden/charter.md | 🔒 Security |
| Scribe | Session Logger | .squad/agents/scribe/charter.md | 📋 Scribe |
| Ralph | Work Monitor | .squad/agents/ralph/charter.md | 🔄 Monitor |
| Rai | RAI Reviewer | .squad/agents/rai/charter.md | 🛡️ RAI |
| Fact Checker | Fact Checker | .squad/agents/fact-checker/charter.md | 🔍 Verifier |
