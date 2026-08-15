# CLI Improvement Research

Research track for improvements to the `onnx-genai` CLI (`crates/onnx-genai-cli`).

**Entry point:** start with [`00-backlog.md`](00-backlog.md). The CLI charter is now settled: `onnx-genai` is a **development / maintainer tool**, not a consumer local-inference product and not an Ollama competitor. Prioritize work that shortens maintainer debug/iterate loops or exposes engine behavior that is otherwise hard to observe.

This is a **research-only** directory — no CLI source changes land on this branch.

| Doc | Author | Scope |
|-----|--------|-------|
| `00-backlog.md` | Roy (Lead) | Consolidated backlog under the dev-tool charter |
| `01-architecture-triage.md` | Roy (Lead) | Command-surface coherence, module structure, ranked improvement backlog |
| `02-ux-and-server-surface.md` | Rachael (Server) | Interactive REPL, output/streaming, `serve` ergonomics, runtime capabilities unreachable from the CLI |
| `03-competitive-and-devils-advocate.md` | Fact Checker | Verified feature matrix vs. ollama / llama.cpp / vLLM / mlx_lm, plus the case against investing here |

Findings feed into `00-backlog.md`; implementation happens on follow-up branches.
