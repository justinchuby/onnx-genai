# Rachael — History (compacted 2026-07-29)

**Role:** CLI/server/API implementer for onnx-genai. Owns OpenAI-compatible server behavior, REPL/maintainer-tool UX, endpoint routing, streaming/session invariants, and user-visible runtime controls while preserving non-TTY byte stability.

## Durable lessons
- Server surface includes `/health`, `/v1/models`, `/v1/chat/completions`, SSE streaming, `X-Session-Id`, session lifecycle routes, tools/tool_choice/tool-role handling, JSON response constraints, FIM, image parts, and audio routing.
- Server DoS/session hardening is canonical: `max_output_tokens=4096`, `max_sessions=256` LRU, 128-bit CSPRNG session ids, context-token caps, loopback/no-auth deployment notes.
- Static-cache HTTP concurrency uses a single engine driver thread and channels; do not reintroduce shared Engine locking. Future tool pause/resume should extend the driver protocol.
- Batched-driver admission is bounded by `max_pending` with HTTP 429; output delivery must be non-blocking so slow/closed clients cannot stall other rows.
- Vision/audio quality is gated on production Mobius model packages and complete processor metadata; routing alone does not prove real quality.
- `/v1/debug/*` must be default-off and redact session identifiers.
- Embeddings empty `model` must fall back to the default, matching other inference endpoints; unknown model returns 404.
- Unsupported-op diagnostics use explicit `OpsetVersion::{Known, Undeclared}` plus graceful unnamed nodes; normal loader validation makes undeclared opsets unreachable.
- Zero-copy mmap initializer borrowing landed, but producer-aliasing soundness restrictions were later added by Zhora.
- Zhora's full-spec onnx-rs serde claim was rejected: vendored ONNX v1.16.2/IR10 proto was stale versus v1.22.0/IR13, and base64 retained-proto native text is non-authoritative; Zhora is locked out and Batty owns revision.
- Qwen Sigmoid fusion must recognize `Mul(x, Sigmoid(x))`, be allocation-free, and include multi-consumer negative coverage.
- The initial Python genai Engine wrapper had a cross-thread PyO3 panic; Sebastian's cleared revision is canonical.
- MLAS SQNBit regressed single-sequence decode but retained prefill potential, informing hybrid M-routing.
- Flash-attention/GQA work must preserve causal-origin parity; CUDA graph decode relies on persistent external-shape seeding.
- Per-op trace sites should emit logical bytes and documented FLOP estimates without overhead regressions.
- Native fp16 CUDA performance work must remain correct and fast across supported SM architectures, not only sm_90.
- QKV-bias folding and paired gate/up+SwiGLU fusion require guarded patterns and two-op-exact fp16 rounding.
- WP-B optional-modality metadata schema landed; Rachael's WP-B design note remains active for WP-B2/WP-B3.
- CLI is a maintainer/development harness; backlog source of truth is `docs/research/cli/00-backlog.md`.
- REPL Phase 1 invariants: non-TTY stdin/stdout stays byte-stable; true TTY uses `reedline`; slash parser/help/completion come from a declarative registry; `/fork` and `/rewind` remain out of Phase 1.
- TTY output owns exactly one separator newline when generated text lacks a trailing newline; piped REPL output keeps legacy separators.
- Compact TTY stats are allowed to be a two-line block; non-TTY opt-in stats keep the single-line formatter for byte stability.
- Reviewer lockouts remain canonical, including PR #300 after author lockout and PR #346 after Bryant lockout.

## Recent work (current wave, ~2026-07-28/29)

## 2026-07-27T09:15:14-07:00 — CLI UX/server-surface research

- Audited `onnx-genai` interactive REPL, streaming output, `serve` ergonomics, and CLI reachability of server/runtime features.
- Wrote findings to `docs\research\cli\02-ux-and-server-surface.md`; top gaps are remote OpenAI-compatible client mode, structured/quiet output, stronger REPL input/history, explicit session/fork/rewind UX, and advanced runtime controls.

## 2026-07-27T09:30:56-07:00 — REPL redesign research

- Wrote `docs\research\cli\05-repl-redesign.md` after Justin clarified the CLI is a maintainer tool and the REPL is the primary CLI investment.
- Recommended preserving ratatui inline viewport/native scrollback, adding a TTY-only `reedline` editor, generating slash help/completions from a declarative registry, and keeping non-TTY e2e output byte-stable.

### 2026-07-27T13:10:00-07:00 — CLI backlog now on main
Scribe note: the CLI dev-tool charter and prioritized backlog from the merged CLI improvement track are now on main at `docs/research/cli/00-backlog.md`. Use that file as the source of truth before picking up queued CLI backlog work.

## 2026-07-27T14:15:00-07:00 — REPL Phase 1 implementation

- Implemented the Phase 1 TTY/plain split for `onnx-genai run`: piped stdin/stdout stays on the byte-stable `read_line` loop, while true TTY sessions use a rich `reedline` editor.
- Chose `reedline` after verifying it shares `crossterm v0.29.0` with ratatui 0.30.2; it provides multiline Alt+Enter input, cursor movement, persistent file-backed history, bracketed paste, and slash completion.
- Replaced the hand slash-command parser/help with a declarative registry that also drives command and argument completion; `/fork` and `/rewind` remain out of Phase 1.
- Made compact stats default-on only for interactive TTY sessions and added `run --no-stats`; non-TTY scripts keep stats opt-in via `/stats`.

## 2026-07-27T02:00:00Z — Roadmap wave update
- Fixed PR #300 / #76 after author lockout: capability projection now rejects non-convex merges with deterministic union-find + Kahn check; merged after Leon approval.

## 2026-07-28T11:20:06+0000 — Independent #75 lockout revision
- Took ownership of PR #346's revision after Holden requested changes and Bryant was locked out.
- `c20ec211` corrected StringNormalizer/TfIdfVectorizer default-domain registration and LabelEncoder-1 default-attribute dtype selection; Holden re-approved.

## 2026-07-28T10:10:00-07:00 — REPL newline and stats polish

- Fixed TTY turn finalization so generated text that lacks its own trailing newline is followed by exactly one CLI-owned separator before the next prompt/status/error path; piped REPL output keeps the legacy byte-stable separator behavior.
- Expanded compact stats/profile reporting with explicit prefix-cache hit tokens plus percent of prompt, KV page activity on the compact line, and matching JSON/profile fields.

## 2026-07-28T10:33:00-07:00 — REPL stats block two-line follow-up

- Revised TTY stats rendering from a squeezed one-line string to a deliberate two-line block: performance/termination first, cache/context/scheduler/pages/RSS second.
- Restored compact finish reason and end-to-end throughput to the always-on TTY block now that Justin allowed two lines; non-TTY opt-in stats keep the single-line formatter for byte stability.

Full pre-compaction history in `history-archive.md`.

## 2026-08-11 — Test hardening: EP assignment assertions

**Task:** Add `disable_cpu_ep_fallback` + `Session_GetEpGraphAssignmentInfo` assertions to layernorm_dynamic_axis and f16/bf16/LN/RMS tests.

**Result:** 6 tests now assert EP ownership. 269 passed, 0 failed. Non-vacuity proven (Relu assertion → immediate panics). Commit `ecfaeeec5`.

**Files modified:**
- `crates/onnx-runtime-ep-cpu-plugin/tests/layernorm_dynamic_axis.rs`
- `crates/onnx-runtime-ep-cpu-plugin/tests/plugin_ort_e2e.rs`
