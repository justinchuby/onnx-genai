# CLI Improvement Research

Research track for improvements to the `onnx-genai` CLI (`crates/onnx-genai-cli`).

This is a **research-only** directory — no CLI source changes land on this branch.

| Doc | Author | Scope |
|-----|--------|-------|
| `01-architecture-triage.md` | Roy (Lead) | Command-surface coherence, module structure, ranked improvement backlog |
| `02-ux-and-server-surface.md` | Rachael (Server) | Interactive REPL, output/streaming, `serve` ergonomics, runtime capabilities unreachable from the CLI |
| `03-competitive-and-devils-advocate.md` | Fact Checker | Verified feature matrix vs. ollama / llama.cpp / vLLM / mlx_lm, plus the case against investing here |

Findings feed into a prioritized backlog; implementation happens on follow-up branches.
