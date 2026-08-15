# CLI architecture triage

**Author:** Roy (Lead)  
**Date:** 2026-07-27  
**Scope:** `crates/onnx-genai-cli` command surface, module boundaries, and improvement backlog for the `onnx-genai` binary.

## Executive read

The CLI is useful and mostly coherent as a thin local-inference front end: `serve`, `generate`, `run`, `show`, `list`/`ls`, `transcribe`, and `version` cover server startup, one-shot generation, interactive chat, inspection, discovery, speech transcription, and runtime identity (`crates/onnx-genai-cli/src/lib.rs:71-104`). `main.rs` is correctly trivial and delegates all behavior to the library entry point shared with the Python wheel (`crates/onnx-genai-cli/src/main.rs:1-10`; `crates/onnx-genai-cli/Cargo.toml:1-16`).

The weak point is not the basic feature set; it is growth pressure. `lib.rs` is a 1,134-line god module containing top-level clap definitions, shared argument structs, request conversion helpers, profiling side effects, dispatch, path resolution, and a large embedded test suite (`crates/onnx-genai-cli/src/lib.rs:71-630`, `crates/onnx-genai-cli/src/lib.rs:632-1134`). That structure is already expensive to extend, especially for model-management, config/profile, JSON output, and completion work.

## Current command surface

| Command | Current shape | Notes |
|---|---|---|
| `serve` | Reuses `onnx_genai_server::ServeArgs` (`crates/onnx-genai-cli/src/lib.rs:54`, `crates/onnx-genai-cli/src/lib.rs:89-90`). | Good reuse; server args live once in `onnx-genai-server` (`crates/onnx-genai-server/src/cli.rs:1-5`). |
| `generate` | Positional model + `--prompt/-p`, shared sampling, attachments, engine, CPU, output image/audio flags (`crates/onnx-genai-cli/src/lib.rs:448-478`). | Overloaded across text, text-to-image, and text-to-speech. |
| `run` | Positional model + shared sampling/attachments/engine/CPU (`crates/onnx-genai-cli/src/lib.rs:480-496`). | Interactive REPL has rich slash commands. |
| `transcribe` | Positional model + audio paths/stdin + segmenting and output format flags (`crates/onnx-genai-cli/src/lib.rs:509-564`). | Has `--format json`; other commands do not. |
| `show` | Positional model inspection (`crates/onnx-genai-cli/src/lib.rs:566-570`). | Human-only output. |
| `list` / `ls` | `--models-dir` / `ONNX_GENAI_MODELS_DIR` only (`crates/onnx-genai-cli/src/lib.rs:572-578`). | No cache/default model-store concept. |
| `version` | Version and execution providers (`crates/onnx-genai-cli/src/lib.rs:102-103`, `crates/onnx-genai-cli/src/model_inspection.rs:63-70`). | Provider list is compile-feature based and may not match selectable providers as closely as REPL `/ep`. |

## Structural assessment

### What is working

- **Shared server surface is in the right crate.** `ServeArgs` is defined once and shared between the standalone server and unified CLI (`crates/onnx-genai-server/src/cli.rs:1-5`, `crates/onnx-genai-server/src/cli.rs:19-107`).
- **Common local generation flags exist.** Sampling, CPU, engine memory limits, and attachment args are factored as clap `Args` groups (`crates/onnx-genai-cli/src/lib.rs:106-147`, `crates/onnx-genai-cli/src/lib.rs:185-225`, `crates/onnx-genai-cli/src/lib.rs:330-342`).
- **Runtime behavior is mostly metadata-driven.** `generate`/`run` route through `Backend`, `PipelineEngine`, and declared multimodal specs rather than model-name switches (`crates/onnx-genai-cli/src/interactive.rs:327-364`, `crates/onnx-genai-cli/src/interactive.rs:469-507`).
- **Some errors are excellent.** Many errors use the project’s `What/Why/How` style, especially memory limits, profile commands, attachments, and pipeline loading (`crates/onnx-genai-cli/src/lib.rs:216-225`, `crates/onnx-genai-cli/src/commands.rs:126-168`, `crates/onnx-genai-cli/src/interactive.rs:1037-1061`).
- **Stdout/stderr separation is mostly intentional.** Tracing defaults to stderr and generated content/transcripts go to stdout (`crates/onnx-genai-cli/src/lib.rs:593-604`, `crates/onnx-genai-cli/src/generate.rs:67-77`, `crates/onnx-genai-cli/src/transcribe.rs:34-58`).

### Architecture problems

1. **`lib.rs` is doing too much.** It owns clap definitions, conversion to engine requests, profile environment mutation, dispatch, model path normalization, and tests (`crates/onnx-genai-cli/src/lib.rs:71-630`, `crates/onnx-genai-cli/src/lib.rs:632-1134`). The file should become an entry/dispatch shell over dedicated `args`, `dispatch`, and `format` modules.

2. **`commands.rs` is misnamed.** It is not top-level command handling; it is REPL slash-command parsing and session helpers (`crates/onnx-genai-cli/src/commands.rs:8-35`, `crates/onnx-genai-cli/src/commands.rs:191-237`). This confuses the module map once real command-management subcommands are added.

3. **Argument parsing, orchestration, and rendering are only partially separated.** `generate.rs` orchestrates model loading, validates incompatible output modes, renders stdout messages, and emits profiles in one path (`crates/onnx-genai-cli/src/generate.rs:19-86`, `crates/onnx-genai-cli/src/generate.rs:89-158`, `crates/onnx-genai-cli/src/generate.rs:161-219`). `interactive.rs` combines session state, slash-command dispatch, rendering, and error policy in a single 1,009-line file (`crates/onnx-genai-cli/src/interactive.rs:596-1030`).

4. **Output formatting is fragmented.** `show`/`list` print ad hoc text (`crates/onnx-genai-cli/src/model_inspection.rs:9-60`), `transcribe --format json` manually prints JSON lines (`crates/onnx-genai-cli/src/transcribe.rs:34-79`), and profiling manually builds JSON strings (`crates/onnx-genai-cli/src/profile.rs:640-753`). There is no CLI-wide output contract.

5. **Flag semantics are inconsistent across commands.** Local `generate`, `run`, `show`, and `transcribe` use positional models (`crates/onnx-genai-cli/src/lib.rs:448-451`, `crates/onnx-genai-cli/src/lib.rs:480-483`, `crates/onnx-genai-cli/src/lib.rs:510-518`, `crates/onnx-genai-cli/src/lib.rs:566-570`), while `serve` uses `--model` / `--models-dir` / `--models-config` (`crates/onnx-genai-server/src/cli.rs:21-50`). Tests explicitly reject `--model` for local generation and REPL (`crates/onnx-genai-cli/src/lib.rs:714-734`).

6. **Profiling flags are global in shape but command-specific in behavior.** `ProfileArgs` is flattened onto the root CLI (`crates/onnx-genai-cli/src/lib.rs:77-83`, `crates/onnx-genai-cli/src/lib.rs:227-327`) and installed before all dispatch (`crates/onnx-genai-cli/src/lib.rs:610-620`), but only generation/transcription paths emit reports. That makes `onnx-genai --profile serve ...` syntactically plausible yet semantically unclear.

7. **Help quality will not scale.** The top-level help has only a short `about` and subcommand doc comments (`crates/onnx-genai-cli/src/lib.rs:71-104`). The REPL help is a hard-coded newline string (`crates/onnx-genai-cli/src/interactive.rs:652-655`). There are no top-level help snapshot tests; only REPL help coverage was found in e2e tests (`crates/onnx-genai-cli/tests/repl_e2e.rs:108-149`, `crates/onnx-genai-cli/tests/repl_e2e.rs:796-800`).

## Capability gaps versus a modern local-inference CLI

- **Model management:** no `pull`, `install`, `list` from cache, `rm`, `where`, `verify`, or model aliases. `list` only scans immediate children of one directory (`crates/onnx-genai-cli/src/model_inspection.rs:51-60`). Server admin endpoints can load/unload models at runtime (`crates/onnx-genai-server/src/cli.rs:79-83`), but the local CLI has no equivalent model-store UX.
- **Config and profiles:** no user config file, named profiles, default model, default EP/backend, or reusable serve/generate presets. Current state is positional args plus env vars such as `ONNX_GENAI_MODELS_DIR`, `ONNX_GENAI_MODEL`, and profile env mutation (`crates/onnx-genai-cli/src/lib.rs:256-269`, `crates/onnx-genai-server/src/cli.rs:30-105`).
- **Machine-readable output:** only `transcribe --format json` and `--profile-json` exist (`crates/onnx-genai-cli/src/lib.rs:235-242`, `crates/onnx-genai-cli/src/lib.rs:524-526`). `show`, `list`, `version`, errors, and generation metadata have no stable JSON schema.
- **Shell completions:** no `completions` subcommand and no `clap_complete` dependency in the CLI crate (`crates/onnx-genai-cli/Cargo.toml:51-61`).
- **Serve ergonomics:** `serve` takes a single `--addr` defaulting to `127.0.0.1:8080` (`crates/onnx-genai-server/src/cli.rs:58-60`). Users commonly expect `--host`, `--port`, `--cors`, `--open-browser`, `--print-ready-json`, and visible admin/debug warnings.
- **Verbosity/logging:** tracing is controlled by ambient env filter, defaulting to `info` (`crates/onnx-genai-cli/src/lib.rs:593-604`). There is no `-v/--verbose`, `-q/--quiet`, `--log-format json`, or command-local diagnostics mode.
- **Exit-code contract:** Ctrl-C maps to 130 in generation/REPL paths (`crates/onnx-genai-cli/src/interactive.rs:29-31`, `crates/onnx-genai-cli/src/generate.rs:79-82`), but user errors, unsupported model capabilities, OOM/resource exhaustion, and runtime failures are not classified into documented exit codes.

## Ranked backlog

### P0 — correctness / UX blockers

| Item | What | Why it matters | Effort | Owner |
|---|---|---|---|---|
| P0.1 | Define a CLI output contract: `--json` / `--format json` for `show`, `list`, `version`, generation metadata, and structured error envelopes; keep generated tokens/transcripts on stdout and diagnostics on stderr. | Automation cannot reliably consume ad hoc text from `show`/`list`/`version`, and only profile/transcribe have JSON today (`model_inspection.rs:9-60`, `transcribe.rs:34-58`, `profile.rs:640-753`). | M | Roy + Zhora; Pris for contract tests |
| P0.2 | Split `lib.rs` into `args.rs`, `dispatch.rs`, `profile_flags.rs`, and CLI parser tests/help snapshots. Leave `lib.rs` as `run()` plus module wiring. | The current 1,134-line command/parser/test module is the choke point for every follow-up feature (`lib.rs:71-630`, `lib.rs:632-1134`). Refactoring first reduces regression risk. | M | Roy + Gaff; Pris for snapshots |
| P0.3 | Normalize model-source grammar across commands: support both positional model and `--model` where appropriate, document precedence, and align `serve`/local commands without breaking existing usage. | Tests intentionally reject `--model` for local commands (`lib.rs:714-734`) while `serve` requires flags (`server/src/cli.rs:21-50`). This is a high-friction inconsistency for users moving from generation to serving. | M | Roy + Rachael/Zhora |
| P0.4 | Document and implement exit-code taxonomy for usage, missing model, unsupported capability, interrupted, resource exhaustion, and internal errors. | `130` is handled for interrupts (`interactive.rs:29-31`, `generate.rs:79-82`), but scripts cannot distinguish common failure modes. | S/M | Roy + Luv; Pris for e2e assertions |

### P1 — high-value improvements

| Item | What | Why it matters | Effort | Owner |
|---|---|---|---|---|
| P1.1 | Add local model-store commands: `models list`, `models show`, `models add`, `models rm`, `models verify`, and later `models pull`. | Current `list` is only a directory scan (`model_inspection.rs:51-60`); modern local inference CLIs center on model lifecycle. Start with local paths before remote pulls. | L | Roy architecture; Isidore packaging; Sapper for metadata validation |
| P1.2 | Add config/profiles: default model directory, default model, EP/backend, memory limits, serve presets, and named profiles. | Users should not have to repeat `--cpu-cores`, `--vram-limit`, `ONNX_GENAI_EP`, or model paths. Current config is spread across CLI flags and env vars (`lib.rs:135-147`, `lib.rs:185-225`, `server/src/cli.rs:30-105`). | L | Roy + Isidore; Rachael for serve profile shape |
| P1.3 | Improve `serve` ergonomics: add `--host`, `--port`, ready output, admin/debug warnings, and JSON startup summary. | `--addr` is precise but not friendly (`server/src/cli.rs:58-60`), and admin/debug endpoints are powerful enough to deserve visible startup posture (`server/src/cli.rs:74-83`). | M | Rachael + Zhora |
| P1.4 | Add shell completions and a discoverable `completions <shell>` command. | There is no completions support or dependency today (`Cargo.toml:51-61`). This is low-risk and high-value for a growing command tree. | S | Isidore + Pris |
| P1.5 | Replace manual JSON construction with typed `serde::Serialize` output structs. | Both transcript and profile JSON are manually escaped/assembled (`transcribe.rs:62-79`, `profile.rs:640-760`). Typed structs reduce schema drift and make `--json` expansion safer. | M | Zhora + Gaff; Pris tests |
| P1.6 | Move REPL slash command definitions into a table-driven registry with generated help. | Slash parsing is hand-coded in `commands.rs` and help is a hard-coded string in `interactive.rs` (`commands.rs:191-237`, `interactive.rs:652-655`). A registry avoids drift as `/model`, `/ep`, `/backend`, `/profile`, `/image`, and future commands grow. | M | Roy + Gaff |

### P2 — nice-to-have / polish

| Item | What | Why it matters | Effort | Owner |
|---|---|---|---|---|
| P2.1 | Add `-v/--verbose`, `-q/--quiet`, and `--log-format text|json`. | Logging currently relies on ambient env filter defaulting to `info` (`lib.rs:593-604`). Explicit flags are easier for CLI users and CI. | S | Zhora |
| P2.2 | Add `doctor` / `env` diagnostics for ORT location, EP availability, tokenizer/model file checks, and model metadata capabilities. | `show` exposes some model details (`model_inspection.rs:9-48`), but there is no environment-level health check. | M | Deckard + Sapper + Isidore |
| P2.3 | Add `benchmark` / `profile compare` wrappers over existing profile JSON. | Profiling is strong (`profile.rs:1-12`, `profile.rs:393-637`), but users still need external scripts to compare runs. | M | Sebastian + Pris |
| P2.4 | Add `generate --input @file`, prompt stdin conventions, and explicit output-file options for text. | `generate` requires `--prompt/-p` text today (`lib.rs:471-477`), while other modalities already have file flags. | S | Roy + Gaff |
| P2.5 | Improve top-level and subcommand `long_about` with examples. | Current clap surface uses short comments only (`lib.rs:71-104`); examples reduce support burden once model/config commands land. | S | Scribe + Roy |

## Recommended sequence

1. Land P0.2 first: split `lib.rs` and add parser/help snapshot tests without changing behavior.
2. Land P0.1 and P1.5 together: typed output schemas make JSON a product feature rather than another set of hand-built strings.
3. Land P0.3 before adding model-store commands, so the model-source grammar is stable.
4. Then build P1.1/P1.2 as the user-facing CLI expansion.
5. Let Rachael/Zhora improve `serve` in parallel once the shared output/config conventions are agreed.

## Significant architecture call

Treat the next CLI wave as an **interface contract project**, not a bag of subcommands. The durable seam should be:

- clap parser structs in a dedicated args module;
- command handlers that return typed output events/results;
- renderers that choose text, JSON, or streaming output;
- documented exit codes and help snapshots.

That preserves the current thin-binary/wheel-sharing design while making the CLI safe to grow.
