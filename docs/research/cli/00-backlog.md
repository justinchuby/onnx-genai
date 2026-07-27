# CLI backlog

**Charter:** `onnx-genai` CLI is a development and maintainer tool, not a consumer local-inference product and not an Ollama competitor. Rank work by whether it shortens a maintainer's debug/iterate loop or exposes engine/server behavior that is currently hard to observe. Competitive parity, remote-client UX, registry workflows, and onboarding polish are out unless they directly accelerate development.

## P0 — do next

| ID | Item | WHY, in dev-loop terms | Effort | Proposed owner | Source doc |
|---|---|---|---|---|---|
| P0.1 | Make live stats reachable: add `run --stats` / `--live-stats`, mention it in clap help and startup text, and keep `/stats` as the in-REPL toggle. | The ratatui live viewport already exists but is effectively hidden: it requires TTY stdout, `show_stats`, and an undocumented in-REPL `/stats` toggle. This is exactly the kind of existing diagnostic Justin needs during local profiling. | S | Roy + Gaff; Pris tests | New directive; `02-ux-and-server-surface.md`; `live_turn.rs:91`, `interactive.rs:614`, `interactive.rs:976` |
| P0.2 | Add structured maintainer output: `--json` for `show`, `list`, `version`, final `generate` metadata, and `--jsonl` for generation events where useful; fix `--profile-json -` collisions. | Maintainers need scriptable experiment loops and reproducible logs. Today only transcripts and profiles have JSON, and profile JSON can collide with generated text on stdout. | M | Zhora + Roy; Pris contract tests | `01-architecture-triage.md`, `02-ux-and-server-surface.md` |
| P0.3 | Add `bench` / `batch` harness for local prompt files: JSONL prompts, repeat count, warmup, profile summary, and per-run JSON output. | This directly shortens perf/debug loops and keeps Fact Checker's only confirmed competitive gap that matters under the dev-tool charter. | M/L | Sebastian + Pris; Batty for scheduler semantics | `03-competitive-and-devils-advocate.md`, `02-ux-and-server-surface.md` |
| P0.4 | Expose unreachable engine behavior behind explicit dev flags/subcommands: speculative decoding controls first, then FIM and embeddings. | These features exist in runtime/server paths but are not locally exercisable from the CLI, forcing maintainers into ad hoc tests or code edits. | M | Batty for speculative/FIM; Zhora for embeddings; Roy for CLI shape | `02-ux-and-server-surface.md` |
| P0.5 | Add top-level help snapshots and a generated REPL command help table. | The CLI is a maintainer harness, so help must reveal diagnostics and advanced controls. Current REPL help is hard-coded and drift-prone. | S | Gaff + Pris | `01-architecture-triage.md`, `02-ux-and-server-surface.md` |

## P1

| ID | Item | WHY, in dev-loop terms | Effort | Proposed owner | Source doc |
|---|---|---|---|---|---|
| P1.1 | Add `doctor` / `env` diagnostics for ORT location, EP availability, compiled features, selectable providers, tokenizer/model files, and metadata capabilities. | Reduces time lost to local setup, wrong EP selection, missing tokenizer, or package shape bugs. | M | Deckard + Sapper + Isidore | `01-architecture-triage.md`; dev-tool directive |
| P1.2 | Add explicit verbosity/color controls: `-v`, `--quiet`, `--color=auto|always|never`, and `--no-live`. | Keeps scripted runs clean while still allowing rich local diagnostics. | S | Zhora + Gaff | `01-architecture-triage.md`, `02-ux-and-server-surface.md` |
| P1.3 | Add local session/KV debug controls: `/session` subcommands, `/rewind`, `/fork` only where local engine APIs support it, `kv`/`pages` snapshots in JSON. | Makes prefix cache, page pressure, and session behavior observable without writing one-off tests. | M/L | Deckard + Leon; Roy for UX; Pris tests | `02-ux-and-server-surface.md` |
| P1.4 | Improve REPL maintainer ergonomics: multiline paste, literal slash escape, prompt history file, save/load transcript. | Speeds repeated repro prompts and model-behavior investigations without pretending to be a consumer chat product. | M | Roy + Gaff | `02-ux-and-server-surface.md` |
| P1.5 | Improve local `serve` maintainer ergonomics: ready JSON on stderr/stdout by flag, `--host`/`--port` aliases for `--addr`, and visible admin/debug endpoint posture. | Helps integration tests and server debugging without building a separate remote client. | M | Rachael + Zhora | `01-architecture-triage.md`, `02-ux-and-server-surface.md` |
| P1.6 | Add narrow local model/package utilities: `show --json`, `verify`, `where`, and metadata/schema validation for local paths only. | Useful for model-package authors and test fixtures; not a registry or consumer model lifecycle. | M | Sapper + Roy; Pris tests | `01-architecture-triage.md`, dev-tool directive |

## P2

| ID | Item | WHY, in dev-loop terms | Effort | Proposed owner | Source doc |
|---|---|---|---|---|---|
| P2.1 | Further split `lib.rs` only opportunistically: extract `args.rs` or output types when a P0/P1 feature already touches them. | Wierzbowski already split `lib.rs` from 3,559 lines into focused modules. A standalone refactor now is churn unless it unblocks JSON/runtime surfaces. | S/M if piggybacked | Roy + Gaff | `01-architecture-triage.md`; `wierzbowski-split-cli-lib.md` |
| P2.2 | Add shell completions. | Helpful for advanced flags, but it does not itself expose new behavior or shorten profiling loops as much as P0/P1. | S | Isidore + Pris | `01-architecture-triage.md` |
| P2.3 | Add prompt input conveniences: `generate --input FILE`, stdin prompt conventions, and text output file options. | Useful for repro scripts, but covered partly by the higher-priority batch harness. | S | Roy + Gaff | `01-architecture-triage.md`, `02-ux-and-server-surface.md` |
| P2.4 | Add profile comparison sugar over existing profile JSON. | Nice once `bench` emits stable JSON; not needed before the batch harness exists. | M | Sebastian + Pris | `01-architecture-triage.md`, `03-competitive-and-devils-advocate.md` |

## Explicitly Rejected

| ID | Item | WHY rejected | Effort | Proposed owner | Source doc |
|---|---|---|---|---|---|
| R.1 | Remote-client mode for OpenAI-compatible servers. | Authoritatively out of scope. Justin will use third-party CLIs for remote server clients; this would duplicate non-core tooling and overrides Rachael's original top finding. | N/A | None | `coordinator-cli-is-a-dev-tool.md`, `02-ux-and-server-surface.md` |
| R.2 | Model registry / pull / consumer model lifecycle. | Not competing with Ollama. Only local package validation/inspection remains in scope when it speeds development. | N/A | None | `coordinator-cli-is-a-dev-tool.md`, `03-competitive-and-devils-advocate.md` |
| R.3 | Conversion, quantization, and fine-tune loops as CLI product features. | Deprioritized unless a specific maintainer workflow needs a thin wrapper. These belong to model-builder/export tooling, not broad CLI parity. | N/A | None | `coordinator-cli-is-a-dev-tool.md`, `03-competitive-and-devils-advocate.md` |
| R.4 | Standalone "split lib.rs because it is large" project. | Partially stale: the major split already happened. Further extraction must be tied to a concrete P0/P1 feature. | N/A | None | `wierzbowski-split-cli-lib.md`, `01-architecture-triage.md` |
