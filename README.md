# onnx-genai

A Rust inference runtime for generative AI models, built on ONNX Runtime.

**Reference implementation** of the [ONNX Inference Metadata Standard](https://github.com/onnx/onnx/issues/8184).

## Features

- 🦀 **Rust-native** — memory safety for KV cache management, fearless concurrency for batching
- 📦 **ORT backend** — leverages ONNX Runtime's execution providers (CUDA, DirectML, CoreML, etc.)
- 🧠 **Agent-first** — prefix caching, multi-session, CoW fork, KV rewind
- 📋 **Standard-driven** — behavior from inference metadata declarations, not hardcoded model-type dispatch
- ⚡ **Speculative decoding** — draft/verify loop with configurable acceptance rules
- 🔄 **Continuous batching** — preemptive scheduler for concurrent requests

## Architecture

```
Public API (OpenAI-compatible HTTP + Rust library)
         ↓
Generation Engine (scheduler + speculative + logit chain)
         ↓
Memory Management (paged KV cache + prefix trie + tiered storage)
         ↓
Backend (ORT sessions + HF tokenizers)
```

## Quick Start

```bash
# Build
cargo build --release

# Run server
cargo run --release -p onnx-genai-server

# Use as library
# [dependencies]
# onnx-genai = { git = "https://github.com/justinchuby/onnx-genai" }
```

## Project Structure

```
crates/
├── onnx-genai/            # Main library (re-exports)
├── onnx-genai-metadata/   # Inference metadata parser + validation
├── onnx-genai-kv/         # Paged KV cache manager
├── onnx-genai-scheduler/  # Continuous batching scheduler
├── onnx-genai-engine/     # Generation engine
└── onnx-genai-server/     # OpenAI-compatible HTTP server
```

## Design

See [docs/DESIGN.md](docs/DESIGN.md) for the full design document.

## Status

**Phase 1: Foundation** — in progress

- [x] Workspace scaffold
- [x] Inference metadata parser
- [x] Paged KV cache (page table, CoW fork, rewind)
- [x] Prefix cache (radix trie)
- [x] Continuous batching scheduler
- [x] Logit processor chain
- [ ] ORT session integration
- [ ] Tokenizer integration
- [ ] End-to-end generation loop
- [ ] CLI tool

## License

MIT
