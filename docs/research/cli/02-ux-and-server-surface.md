# CLI UX and server-surface audit

Author: Rachael (Server Dev)  
Date: 2026-07-27  
Scope: end-user CLI experience, output behavior, and the relationship between `onnx-genai` and `onnx-genai-server`.

## Executive findings

1. **Highest-value gap: the CLI is not a server client.** `onnx-genai serve` starts an in-process OpenAI-compatible server, while `generate`, `run`, and `transcribe` all load local models directly. There is no `--base-url`, `--api-key`, or OpenAI-compatible client mode, despite the server exposing `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, `/v1/audio/transcriptions`, `/v1/images/generations`, and `/v1/audio/speech` (`crates\onnx-genai-server\src\lib.rs:60-76`; CLI dispatch calls `run_serve` directly at `crates\onnx-genai-cli\src\lib.rs:615-622`).
2. **The REPL is functional but very line-oriented.** It uses `stdin.lock().read_line`, prints a bare `>>> ` prompt, and treats an empty line or Ctrl-D as exit (`crates\onnx-genai-cli\src\interactive.rs:637-650`). That means no command history/search, no line editing beyond the host terminal, no first-class multi-line input, no pasted blocks, and no way to submit an intentionally empty prompt.
3. **The interactive session stores chat history, not engine sessions.** Each turn renders the full in-memory `history` through the chat template (`crates\onnx-genai-cli\src\interactive.rs:629-633`, `crates\onnx-genai-cli\src\interactive.rs:939-956`). The engine has persistent sessions (`create_session`, `generate_in_session`, `close_session`) and the server exposes create/delete plus `X-Session-Id`, but the CLI REPL does not expose those IDs or lifecycle controls (`crates\onnx-genai-engine\src\engine\runtime.rs:244-279`, `crates\onnx-genai-engine\src\engine\runtime.rs:500-612`, `crates\onnx-genai-server\src\routes\sessions.rs:3-60`, `crates\onnx-genai-server\src\routes\completions.rs:323-356`).
4. **Streaming output is carefully terminal-aware, but scripting output is under-designed.** Live rendering is disabled unless stdout is a terminal and uses ratatui inline viewports instead of the alternate screen (`crates\onnx-genai-cli\src\live_turn.rs:24-32`, `crates\onnx-genai-cli\src\live_turn.rs:84-163`). `generate` has `--stream`, and profiling can emit JSON, but the text generation commands lack `--json`, `--quiet`, `--no-color`, or NDJSON event output (`crates\onnx-genai-cli\src\lib.rs:448-496`, `crates\onnx-genai-cli\src\lib.rs:227-243`).
5. **Runtime/server capabilities are ahead of the CLI.** The engine exposes speculative modes, prefix-cache stats, persistent sessions, static and continuous batching, FIM, embeddings, page/KV introspection, and resource controls; the server exposes much of that over HTTP. The CLI exposes only a subset: local generate/run/transcribe/show/list/serve/version, `--vram-limit`, `--host-ram-limit`, `--cpu-cores`, `/pages`, `/stats`, `/profile`, `/ep`, and `/backend` (`crates\onnx-genai-cli\src\commands.rs:220-235`, `crates\onnx-genai-cli\src\lib.rs:185-201`, `crates\onnx-genai-cli\src\lib.rs:615-629`).

## 1. Interactive REPL walkthrough

### What works well

- Startup tells the user accepted input modalities, exit mechanics, Ctrl-C behavior, and `/help` (`crates\onnx-genai-cli\src\interactive.rs:623-627`).
- Slash commands cover core local-session operations: `/help`, `/reset`, `/raw`, `/stats`, `/pages`, `/profile`, `/model`, `/session`, `/ep`, `/backend`, `/system`, `/image`, and `/audio` (`crates\onnx-genai-cli\src\interactive.rs:652-655`; parser in `crates\onnx-genai-cli\src\commands.rs:220-235`).
- Model/provider/backend switches are fail-soft. Replacement backends are loaded before the current session is replaced, so a bad `/ep` or `/backend` does not kill the REPL (`crates\onnx-genai-cli\src\commands.rs:45-52`, `crates\onnx-genai-cli\src\interactive.rs:793-847`).
- Ctrl-C handling is intentionally two-stage: during generation, first press soft-cancels and second press exits; while idle, first press warns and second exits with 130 (`crates\onnx-genai-cli\src\interactive.rs:82-128`). Interrupted turns are removed from history (`crates\onnx-genai-cli\src\interactive.rs:1015-1020`).
- `/session` is privacy-preserving: it reports counts, settings, and usage, not message content (`crates\onnx-genai-cli\src\interactive.rs:249-314`).
- Reasoning model output is handled thoughtfully: thinking can be dimmed while streaming, and only the final answer is retained in history (`crates\onnx-genai-cli\src\output.rs:155-200`, `crates\onnx-genai-cli\src\interactive.rs:982-1005`).

### Rough or surprising UX

- **No multi-line input mode.** A single `read_line` drives the prompt loop, and an empty line exits (`crates\onnx-genai-cli\src\interactive.rs:637-650`). Suggested design:
  - `/paste` starts a block terminated by `.\n` or Ctrl-D.
  - `"""` heredoc mode for prompt blocks.
  - `Alt+Enter` only if a future rustyline/reedline layer is adopted.
- **No persistent input history.** The REPL maintains model conversation history, but not command/prompt recall across the process or across invocations (`crates\onnx-genai-cli\src\interactive.rs:629-633`). Suggested flags:
  - `onnx-genai run MODEL --history-file ~/.onnx-genai/history`
  - `/history`, `/save transcript.md`, `/load transcript.jsonl`
- **Prompts beginning with `/` are command-looking.** Unknown slash input reports `unknown command` rather than being sent to the model (`crates\onnx-genai-cli\src\interactive.rs:930-932`). Suggested design: `//text` escapes a literal leading slash, and `/raw once <text>` sends one literal prompt.
- **`/backend` with no argument prints the whole session summary.** The parser has a dedicated decode-backend command, but the no-arg branch prints `SessionSummary` instead of just `decode backend: ...` (`crates\onnx-genai-cli\src\interactive.rs:830-859`). Suggested design: `/backend` prints current backend and choices; `/session` remains the full summary.
- **Empty prompt cannot be sent.** Empty line exits (`crates\onnx-genai-cli\src\interactive.rs:649-650`). Suggested design: `/send-empty` or `/submit` in multi-line mode.
- **No visible progress before first token.** Stats and live rendering start when tokens arrive (`crates\onnx-genai-cli\src\output.rs:167-200`). Long model load, tokenizer load, image preprocessing, or prefill can look hung. Suggested design: stderr phases such as `[loading model]`, `[prefill 4,096 tokens]`, or a spinner gated on TTY and disabled by `--quiet`.
- **Conversation state is local and implicit.** `/reset` clears local history and attachments (`crates\onnx-genai-cli\src\interactive.rs:658-664`), but there are no user-visible server-style session IDs, no fork, no rewind, no export/import, and no "continue this session from another process".

## 2. Output, streaming, and scripting

### Strengths

- Diagnostics and tracing initialize on stderr so stdout can remain parseable output (`crates\onnx-genai-cli\src\lib.rs:593-604`).
- Terminal detection is explicit. The live renderer stays inert when stdout is not a terminal, preserving plain text for pipes and tests (`crates\onnx-genai-cli\src\live_turn.rs:24-32`, `crates\onnx-genai-cli\src\live_turn.rs:377-385`).
- Live rendering uses an inline viewport and keeps scrollback selectable instead of taking over an alternate screen (`crates\onnx-genai-cli\src\live_turn.rs:15-22`).
- The status line shows token count, decode tok/s, and TTFT mid-turn; the post-turn stats line can include prompt/output tokens, reuse, encoder cache hits, and RSS (`crates\onnx-genai-cli\src\profile.rs:185-256`).
- Profiling has both human text and machine JSON/timeline output (`crates\onnx-genai-cli\src\lib.rs:227-243`, `crates\onnx-genai-cli\src\lib.rs:272-325`).
- `transcribe` has text, JSON, and SRT output formats and flushes per segment for live streams (`crates\onnx-genai-cli\src\lib.rs:498-507`, `crates\onnx-genai-cli\src\transcribe.rs:26-59`).

### Gaps

- **No generation JSON mode.** `generate` prints either streamed tokens or a final string (`crates\onnx-genai-cli\src\generate.rs:59-85`). There is no structured `GenerateResult` output with finish reason, token IDs, usage, prefix hit length, logprobs, or timings.
- **No quiet mode.** Scripts cannot suppress banners, progress, warnings, or stats uniformly. Suggested convention:
  - `--quiet`: no banner/progress; errors only on stderr.
  - `--json`: one final JSON object on stdout.
  - `--jsonl`: streaming events (`token`, `stats`, `done`, `error`) on stdout.
  - `--no-color`: force no ANSI even on TTY.
- **`--profile-json -` can collide with command output.** Profile JSON writes to stdout when path is `-` (`crates\onnx-genai-cli\src\lib.rs:292-297`), while `generate` also prints generated text to stdout (`crates\onnx-genai-cli\src\generate.rs:67-72`). Suggested design: reject `--profile-json -` unless `--json` wraps output and profile together, or require `--output - --profile-json profile.json`.
- **TTY color policy is minimal.** Reasoning dimming uses ANSI reset in the plain streaming path only when stdout is a terminal (`crates\onnx-genai-cli\src\output.rs:155-200`), and ratatui styles the live path (`crates\onnx-genai-cli\src\live_turn.rs:276-298`). There is no `NO_COLOR`, `CLICOLOR`, or explicit `--color=auto|always|never` surface.
- **Windows/POSIX rendering looks intentionally portable, but cursor-query activation is risky by design.** The renderer delays terminal activation because inline viewport setup asks the terminal for cursor position and reads stdin (`crates\onnx-genai-cli\src\live_turn.rs:48-57`). This is a good mitigation, but Windows terminals vary; keep the pipe fallback and add manual `--no-live` / `--live=never|auto|always`.

## 3. CLI and server relationship

### What exists

- `onnx-genai serve` is a subcommand on the unified CLI (`crates\onnx-genai-cli\src\lib.rs:87-104`) and parses the shared server `ServeArgs` (`crates\onnx-genai-server\src\cli.rs:1-5`).
- Server mode is ergonomic for local serving: single model, models directory, or models config are mutually exclusive and required; model source env vars are supported; `--addr` defaults to `127.0.0.1:8080`; max output tokens, max sessions, queue depth, debug/admin endpoints, max loaded models, and KV cache dtype have flags/env vars (`crates\onnx-genai-server\src\cli.rs:19-107`).
- The CLI crate's `_onnx_genai_server` library target exists for packaging, not because the CLI is a network client. Cargo explains that the unified binary and PyO3 extension share one codebase, and the wheel entry point preloads ONNX Runtime from the `onnxruntime` wheel before calling into the module (`crates\onnx-genai-cli\Cargo.toml:1-16`, `crates\onnx-genai-cli\Cargo.toml:26-31`).

### Notable gap: no remote-client mode

`generate`, `run`, and `transcribe` all require a model path and construct local engines (`crates\onnx-genai-cli\src\lib.rs:448-496`, `crates\onnx-genai-cli\src\generate.rs:19-52`, `crates\onnx-genai-cli\src\transcribe.rs:180-188`). None can target `http://host:port/v1`.

Suggested command designs:

```text
onnx-genai chat --base-url http://127.0.0.1:8080/v1 --model qwen --stream
onnx-genai generate --remote http://127.0.0.1:8080/v1 --model qwen -p "hello" --json
onnx-genai run --remote http://127.0.0.1:8080/v1 --model qwen --session auto
onnx-genai transcribe --remote http://127.0.0.1:8080/v1 --model whisper audio.wav --format json
onnx-genai models --remote http://127.0.0.1:8080/v1
```

Recommended behavior:

- `--base-url` defaults from `OPENAI_BASE_URL` or `ONNX_GENAI_BASE_URL`.
- `--api-key` / `--api-key-env` defaults from `OPENAI_API_KEY`, even though local `serve` has no auth, so the same client can hit proxied deployments.
- `run --remote --session auto` creates a server session and sends `X-Session-Id`; `/fork`, `/rewind`, and `/reset` can map to future server lifecycle endpoints.
- `serve --print-openai-env` could print:
  ```text
  export OPENAI_BASE_URL=http://127.0.0.1:8080/v1
  export OPENAI_API_KEY=unused
  ```
- `serve --wait-ready` is not needed for the foreground server, but tests and scripts would benefit from a companion `onnx-genai health --base-url ...`.

## 4. Runtime/server feature surface vs CLI reachability

| Capability | Runtime/server evidence | CLI reachability | Gap |
|---|---|---|---|
| OpenAI chat/completions streaming | Routes exist for chat/completions and SSE streaming (`crates\onnx-genai-server\src\lib.rs:68-76`, `crates\onnx-genai-server\src\routes\completions.rs:200-321`, `crates\onnx-genai-server\src\routes\completions.rs:514-740`). | Local `generate --stream` and `run`; no HTTP client. | Add remote chat/generate client. |
| Server persistent sessions | `POST /v1/sessions`, `DELETE /v1/sessions/{id}`, and `X-Session-Id` generation paths exist (`crates\onnx-genai-server\src\routes\sessions.rs:3-60`, `crates\onnx-genai-server\src\routes\completions.rs:323-356`). | REPL has only local prompt history. | Add `/session create|use|delete`, `--session`, remote session mode. |
| Prefix cache | Engine result reports `prefix_cache_hit_len` (`crates\onnx-genai-engine\src\config.rs:1246-1258`), CLI profile records it (`crates\onnx-genai-cli\src\output.rs:214-220`), server metrics expose hit counters (`crates\onnx-genai-server\src\metrics.rs:285-306`). | Visible via `/stats`, `--profile`, `/pages`; not configurable. | Add JSON fields and cache controls/clear. |
| KV pages/introspection | CLI `/pages` renders page usage (`crates\onnx-genai-cli\src\interactive.rs:863-870`, `crates\onnx-genai-cli\src\pages.rs:20-84`). Server debug KV says engine page stats are unavailable there (`crates\onnx-genai-server\src\routes\admin.rs:118-141`). | Local REPL only. | Add `onnx-genai kv` and server debug parity. |
| Multi-session / CoW-like sharing | Engine sessions retain tokens/KV (`crates\onnx-genai-engine\src\session.rs:11-24`); decode state supports truncation for forked/edited conversations (`crates\onnx-genai-engine\src\decode\state.rs:301-360`, `crates\onnx-genai-engine\src\decode\state.rs:623-625`). | No CLI fork/rewind/session tree. | Add session tree UX after server/library API is explicit. |
| KV rewind | Decode runners can rewind static, past/present, and native state (`crates\onnx-genai-engine\src\decode\state.rs:362-380`). | Not exposed. | Add `/rewind <turn|tokens>` and transcript checkpoints. |
| Speculative decoding | Engine config supports draft model, prompt lookup, MTP, Eagle3, and shared-KV speculative modes (`crates\onnx-genai-engine\src\config.rs:379-400`, `crates\onnx-genai-engine\src\config.rs:468-522`); speculative loop uses KV rewind (`crates\onnx-genai-engine\src\speculative.rs:1-6`). | No CLI flags for `--draft-model`, `--speculative-mode`, `--num-speculative-tokens`, or stats. | Add advanced `generate/run/serve` flags plus profile stats. |
| Static and continuous batching | Engine exposes fixed static batching and continuous batching (`crates\onnx-genai-engine\src\batched.rs:420-455`, `crates\onnx-genai-engine\src\batched.rs:639-660`); server driver continuously batches non-session requests (`crates\onnx-genai-server\src\driver.rs:108-125`, `crates\onnx-genai-server\src\driver.rs:520-571`). | CLI one-shot cannot submit batches except image batch size; REPL is single-user. | Add `generate --batch prompts.jsonl`, `bench batch`, and client load-test helpers. |
| FIM | Server `/v1/completions` supports suffix/FIM and rejects sessions for FIM (`crates\onnx-genai-server\src\routes\completions.rs:14-25`); engine has `generate_fim` (`crates\onnx-genai-engine\src\engine\runtime.rs:197-223`). | No direct CLI `fim` command or `generate --suffix`. | Add `onnx-genai fim MODEL --prefix file --suffix file`. |
| Embeddings | Server exposes `/v1/embeddings` (`crates\onnx-genai-server\src\routes\completions.rs:36-92`). | No CLI `embed`. | Add `onnx-genai embed MODEL --input ... --format json|base64`. |
| Image/audio HTTP endpoints | Server exposes image generation, audio speech, and audio transcription (`crates\onnx-genai-server\src\routes\multimodal.rs:3-117`, `crates\onnx-genai-server\src\routes\multimodal.rs:119-240`). | Local `generate --output-image`, `generate --output-audio`, and `transcribe`; no remote client. | Reuse OpenAI-compatible client surface. |
| Resource controls | CLI local engine exposes `--vram-limit` and `--host-ram-limit` (`crates\onnx-genai-cli\src\lib.rs:185-213`); server exposes max sessions/queue/KV dtype/admin resource endpoint (`crates\onnx-genai-server\src\cli.rs:62-107`, `crates\onnx-genai-server\src\routes\admin.rs:144-150`). Engine config has more knobs: page count/size, scheduler, draft model, KV connector, pipeline cache (`crates\onnx-genai-engine\src\config.rs:489-522`). | Partial. | Decide safe stable knobs and expose under `--advanced` or config file. |

## Prioritized backlog proposal

1. **Remote client mode:** `--base-url`, `--api-key-env`, `models`, `health`, `generate`, `run`, `transcribe`, `embed`, and image/audio client paths.
2. **Structured output:** `--json`, `--jsonl`, `--quiet`, `--color`, and collision-free profile output.
3. **REPL input layer:** command history, multi-line paste, literal-slash escape, save/load transcript, and explicit empty prompt.
4. **Session UX:** local and remote session IDs, `/session create|use|reset|delete`, transcript checkpoints, then `/fork` and `/rewind` when server APIs exist.
5. **Advanced runtime knobs:** speculative decoding flags, FIM, embeddings, batch prompt files, KV/cache controls, and server/CLI parity for resource introspection.
