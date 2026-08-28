# REPL redesign: terminal interaction design

Author: Rachael (Server Dev)  
Date: 2026-07-27  
Scope: `onnx-genai run` terminal layout, input editing, slash-command completion, rendering, stats, and migration.

## Context and constraints

- The CLI is now explicitly a **developer/maintainer tool**, not a consumer local-inference product; remote-client mode and consumer parity are out of scope (`.squad\decisions.md:5652-5672`, `.squad\decisions.md:5691-5693`, `docs\research\cli\00-backlog.md:1-4`, `docs\research\cli\00-backlog.md:35-42`).
- The REPL is now the primary CLI investment. Required capabilities are Copilot-CLI-style layout, real editing, fork/runtime controls, stats by default, and slash-command completion (`.squad\decisions.md:5698-5720`).
- Existing REPL is a plain line loop: banner on stderr, `>>> ` prompt on stdout, `read_line`, empty line exits, then hard-coded command handling (`crates\onnx-genai-cli\src\interactive.rs:623-650`, `crates\onnx-genai-cli\src\interactive.rs:652-655`).
- Existing live output is valuable and must not be discarded: ratatui inline viewport, TTY-gated, scrollback-preserving, no raw mode, plain fallback for pipes/tests (`crates\onnx-genai-cli\src\live_turn.rs:15-32`, `crates\onnx-genai-cli\src\live_turn.rs:48-57`, `crates\onnx-genai-cli\src\live_turn.rs:84-111`).
- Existing e2e tests drive the real binary through piped stdin/stdout/stderr and assert plain text, including `>>> ` prompt counts and stats opt-in behavior (`crates\onnx-genai-cli\tests\repl_e2e.rs:1-6`, `crates\onnx-genai-cli\tests\repl_e2e.rs:41-74`, `crates\onnx-genai-cli\tests\repl_e2e.rs:260-272`, `crates\onnx-genai-cli\tests\repl_e2e.rs:462-496`).
- Deckard's `docs\research\cli\04-runtime-capability-inventory.md` is the source of truth for *which* runtime capabilities become commands. This design only describes how those commands are presented, edited, completed, and rendered.

## Copilot CLI grounding

GitHub's public changelog says the redesigned Copilot CLI terminal UI has tabs at the top, mouse-switchable tab navigation, in-session configuration dialogs, theme-aware semantic colors, responsive components, screen-reader support, and settings/theme commands inside the session.[^copilot-changelog]

DeepWiki's Copilot CLI docs describe interactive mode as a conversational interface with a continuously updating timeline, a status bar, a multi-line input area, completion popups for mentions/paths/commands, prompt stashing, and alt-screen mode enabled by default in version 1.0.8.[^copilot-basics] Its UI/accessibility notes describe the alt-screen as a full-screen persistent interface with header, scrollable viewports, dialogs, scrollbar, mouse support, tmux/WSL fixes, and screen-reader/color-mode affordances.[^copilot-ui]

Adaptation for `onnx-genai`: copy the **stable bottom input + streaming timeline above + status/stats context** pattern, but do not copy GitHub-specific tabs as P0. The `onnx-genai` REPL is a maintainer harness; top-level tabs for Issues/PRs/Gists would violate the dev-tool scope unless a future local debugging pane proves useful.

[^copilot-changelog]: https://github.blog/changelog/2026-06-23-copilot-cli-new-terminal-interface-is-generally-available/
[^copilot-basics]: https://deepwiki.com/github/copilot-cli/3.1-interactive-session-basics
[^copilot-ui]: https://deepwiki.com/github/copilot-cli/3.9-ui-modes-and-accessibility

## 1. Layout design

### Recommended screen model

Use a chat-like terminal layout while preserving native scrollback:

- **Transcript / streaming output:** output appears above the input area. Completed turns are ordinary terminal text in native scrollback. During generation, the current assistant reply occupies an inline ratatui viewport, as today, and spills completed lines upward into terminal scrollback (`crates\onnx-genai-cli\src\live_turn.rs:198-219`).
- **Input box:** a persistent bottom editor drawn after each completed frame. It should support one or more visual rows, not a single `>>> ` prompt. It contains placeholder/help text when empty and exposes completion menus just above the input frame.
- **Status/stats line:** one compact status row immediately above or inside the input frame. During generation it shows live token count, decode rate, and TTFT using the existing live summary (`crates\onnx-genai-cli\src\profile.rs:185-202`). After each turn it shows the compact stats line by default (`crates\onnx-genai-cli\src\profile.rs:205-256`).
- **Reasoning vs answer:** preserve the current distinction. Reasoning segments remain dimmed while streaming and are not retained in chat history (`crates\onnx-genai-cli\src\output.rs:155-200`, `crates\onnx-genai-cli\src\interactive.rs:982-1005`). In the richer layout, prefix reasoning blocks with a muted `thinking` label and answer blocks with `assistant`; do not print raw reasoning markers.
- **Scrollback:** native terminal scrollback remains the historical transcript. The active viewport should only reserve enough rows for the live reply + status + input editor, then release completed content into scrollback.
- **Dialogs:** avoid modal full-screen dialogs in phase 1. Use inline help/completion popups. If later commands need pickers (`/model`, `/ep`), make them narrow inline overlays first.

### ASCII mockup

```text
onnx-genai run C:\models\qwen2.5-int4  ep=cuda  backend=native  raw=off
──────────────────────────────────────────────────────────────────────────────
user  > Explain why prefix reuse changed after /fork.

thinking  (dim)
  The session tree has two branches; branch B reused the shared prefix...

assistant
  Prefix reuse changed because the fork kept the common KV prefix but diverged
  after turn 4. The next decode reused 3,840 prompt tokens and allocated 12
  new pages for the branch-local suffix.

stats  4,112 in · 173 out · 64.8 tok/s · ttft 91 ms · 3,840 reused · rss 7.2 GiB
──────────────────────────────────────────────────────────────────────────────
> /rewind <TAB>
  /rewind turn <n>     rewind conversation to a prior turn
  /rewind tokens <n>   drop generated tokens from the active branch
  /fork <name>         create a branch from the current session state
──────────────────────────────────────────────────────────────────────────────
Enter submits · Alt+Enter newline · Tab complete · Ctrl+R history · /help commands
```

Narrow terminals should stack the status row instead of truncating important values, following Copilot CLI's responsive status-line approach.[^copilot-ui]

## 2. Architectural decision: full-screen alternate screen vs inline viewport

This is the key user-facing tradeoff and should be decided explicitly.

| Dimension | Full-screen ratatui alternate-screen app | Inline ratatui viewport + rich line editor |
|---|---|---|
| Scrollback preservation | Poor by default. The transcript lives in the alternate buffer and disappears from the terminal after exit unless separately exported. This contradicts the current live renderer's design goal that the conversation remains selectable, copyable, and present after session end (`crates\onnx-genai-cli\src\live_turn.rs:15-22`). | Strong. Completed output remains ordinary terminal scrollback. This matches the existing module contract (`crates\onnx-genai-cli\src\live_turn.rs:15-22`). |
| Copy/paste | Can support mouse selection inside app, but terminal-native copy of long transcripts is worse and depends on app scrollback. | Best for maintainers copying prompts, answers, stats, and error text into issues or perf notes. |
| Piping / non-TTY fallback | Requires a separate non-TTY code path anyway. Easy to accidentally regress with raw-mode/global renderer assumptions. | Natural fit: keep existing `stdout().is_terminal()` gate and plain fallback (`crates\onnx-genai-cli\src\live_turn.rs:24-32`, `crates\onnx-genai-cli\src\live_turn.rs:84-95`). |
| Existing e2e tests | Higher risk. Current tests rely on piped scripts and plain merged stdout/stderr; full-screen code must be bypassed perfectly (`crates\onnx-genai-cli\tests\repl_e2e.rs:41-74`). | Lower risk. Non-TTY remains byte-stable and can keep the current `read_line` loop until TTY-only editor path is introduced. |
| Windows Terminal / conhost | Modern Windows Terminal handles alt-screen and VT sequences well; older conhost behavior is less predictable. Full-screen redraw, mouse, suspend/resume, and resize handling expand the test matrix. | Fewer moving parts. Inline ratatui already works cross-platform enough for current live rendering. Rich editing still needs raw mode, but only while reading input. |
| Input/editor integration | Coherent if the entire app owns raw mode, key events, rendering, scrolling, mouse, and popups. Best long-term if `onnx-genai` wants an IDE-like shell. | More incremental but trickier at the seam: the editor must coexist with the live viewport and avoid ratatui cursor queries while the user is typing. The current live renderer already defers activation because inline viewport setup reads stdin (`crates\onnx-genai-cli\src\live_turn.rs:48-57`). |
| Accessibility | Potentially richer: labelled regions, themes, scrollbars, dialogs, screen-reader mode like Copilot CLI. | Simpler and native-terminal-friendly. Screen readers see real output lines; input editor accessibility depends on the chosen crate. |
| Implementation cost | High. Requires app state machine, transcript model, viewport scrolling, resize handling, paste, selection tradeoffs, non-TTY bypass, and test harness redesign. | Medium. Reuse `LiveTurn`, add an editor for TTY input, add declarative command registry/completion, and keep non-TTY path unchanged. |

**Recommendation: keep the inline viewport for phase 1 and layer a rich TTY-only line editor on top.** Copilot CLI can afford the alternate screen because it is a broad agent shell with tabs, dialogs, resource browsing, and app-local history. `onnx-genai` is a maintainer harness whose outputs need to be copied into issues, benchmarks, and traces. Native scrollback is a feature, not a limitation. Revisit alternate screen only after phase 1 proves that inline editing cannot handle completion popups, multiline paste, and live rendering cleanly.

## 3. Input editor crate evaluation

The CLI already depends on ratatui 0.30.2 with the crossterm backend (`crates\onnx-genai-cli\Cargo.toml:61`). Any editor crate must be checked with `cargo tree -p crossterm` before landing to avoid multiple incompatible crossterm versions in one process. Ratatui/crossterm version skew is a known class of problems in TUI stacks; prefer one crossterm version and avoid owning terminal modes from two places at once.

| Option | Multiline editing | Persistent history | Slash completion / hints | Bracketed paste | Windows support | Ratatui/crossterm risk | Assessment |
|---|---|---|---|---|---|---|---|
| `reedline` | Strong. Designed for shell-style multiline editing; supports Emacs/Vi modes and rich prompts.[^reedline] | Strong. File-backed history and history-based hints/completions are core features.[^reedline] | Strong. Custom completers, completion menus, inline hints, and syntax highlighting fit `/` commands well.[^reedline-completion] | Supported; designed to keep pasted multiline blocks from executing accidentally.[^reedline] | Cross-platform and uses crossterm.[^reedline] | Medium. It brings its own crossterm dependency; must align with ratatui 0.30.2 or accept duplicate crossterm only if terminal ownership is isolated. | **Recommended**. Best feature match for a Copilot-like prompt without hand-rolling. |
| `rustyline` | Good. Supports multiline with continuation prompts.[^rustyline] | Good. File-backed history via feature flags.[^rustyline] | Good but lower-level. Custom `Completer`, `Hinter`, `Highlighter` traits are enough for slash commands.[^rustyline] | Supported in modern terminals.[^rustyline] | Mature Windows support for cmd.exe/PowerShell; mintty caveats.[^rustyline] | Low/medium. Historically less tied to crossterm rendering than reedline, but still owns terminal input behavior. | Solid fallback if reedline crossterm versions conflict or its rendering fights inline ratatui. |
| Hand-rolled on `crossterm` | Whatever we build. Multiline, Unicode graphemes, history search, kill/yank, undo, paste, resize, and completions are all project work. | Must build. | Must build registry UI, menu navigation, hints, path completion. | Crossterm has `EnableBracketedPaste`, but Windows behavior has historically varied.[^crossterm-paste] | Possible, but conhost/Windows Terminal details become our burden. | Can align exactly with ratatui if using the same crossterm version. | Do **not** start here. Only hand-roll the thin glue around a crate. |

**Recommendation: prototype `reedline` first.** It is the closest to the required editing surface: multiline, history file, completion menus, inline hints, bracketed paste, and Windows via crossterm. The first spike should verify that `reedline` can:

1. render a prompt after `LiveTurn::finish()`;
2. yield cleanly before generation so `LiveTurn` can own the inline viewport;
3. leave non-TTY input on the current `read_line` path;
4. avoid duplicate incompatible crossterm versions with ratatui 0.30.2.

If the dependency tree or rendering ownership is bad, use `rustyline` as the lower-risk editor and implement a smaller custom completion menu.

[^reedline]: https://docs.rs/reedline/latest/reedline/ and https://www.nushell.sh/book/line_editor.html
[^reedline-completion]: https://deepwiki.com/nushell/reedline/8-completion-system
[^rustyline]: https://docs.rs/crate/rustyline/latest and https://github.com/kkawakam/rustyline/blob/master/Features.md
[^crossterm-paste]: https://docs.rs/crossterm/latest/crossterm/event/struct.EnableBracketedPaste.html and https://github.com/crossterm-rs/crossterm/issues/737

## 4. Slash-command autocomplete design

Today's command surface is an enum plus hand parser (`crates\onnx-genai-cli\src\commands.rs:8-35`, `crates\onnx-genai-cli\src\commands.rs:191-237`) and hard-coded `/help` text (`crates\onnx-genai-cli\src\interactive.rs:652-655`). A completion system wants a declarative registry instead:

```rust
struct ReplCommandSpec {
    name: &'static str,
    aliases: &'static [&'static str],
    summary: &'static str,
    usage: &'static str,
    category: CommandCategory,
    args: &'static [ArgSpec],
    availability: Availability,
    handler: ReplCommandHandler,
}

enum CompletionSource {
    Fixed(&'static [&'static str]),
    Files { kind: FileKind },
    ExecutionProviders,
    DecodeBackends,
    Models,
    SessionNames,
    RuntimeExtensions, // populated from the generated metadata extension registry
}
```

Design rules:

- `parse_repl_line`, completer, inline hints, and `/help` all read the same registry. No hard-coded help string.
- Registry remains local-code data, not generated from docs. Deckard's inventory doc decides command candidates; implementation registers accepted commands.
- Slash commands are categorized: session/tree (`/fork`, `/rewind`, `/session`), model/runtime (`/model`, `/ep`, `/backend`, future runtime controls), diagnostics (`/stats`, `/pages`, `/profile`), input/attachments (`/system`, `/image`, `/audio`, raw/literal slash), help/config.
- Unknown slash input should keep today's safety behavior by default (`unknown command`, session continues; `crates\onnx-genai-cli\src\interactive.rs:930-932`), but add `//text` as an escape for prompts beginning with `/`.

### Completion behavior

- **Tab at empty input:** show top commands grouped by category.
- **Tab after `/`:** complete command names; repeated Tab cycles or opens menu depending on editor crate default.
- **Tab after command + space:** complete arguments from the command's `CompletionSource`.
  - `/model <path>`: filesystem directories/model package paths.
  - `/ep <name>`: `available_execution_providers()` (`crates\onnx-genai-cli\src\commands.rs:70-80`).
  - `/backend <name>`: `auto`, `ort`, `native` (`crates\onnx-genai-cli\src\commands.rs:82-93`).
  - `/profile verbosity <level>`: `TraceVerbosity::ALL` (`crates\onnx-genai-cli\src\commands.rs:139-153`).
  - `/image` and `/audio`: filesystem paths.
  - `/fork`, `/rewind`, and future runtime commands: names/ids/states from the active session/runtime once Deckard's inventory maps capabilities to APIs.
- **Inline hints:** show command usage ghost text after a unique command prefix, e.g. typing `/prof` hints `ile [on|off|trace <path>|verbosity <level>]`.
- **Descriptions:** completion menu rows show `usage` and `summary`, not just names.
- **Generated `/help`:** `/help` prints the same registry in a stable order; `/help /profile` prints long help, arguments, examples, and availability.
- **Non-TTY:** no interactive completion; `/help` output remains plain text from the same registry.

## 5. Stats by default

Current default is `show_stats = false`, with `/stats` toggling it and live rendering following that flag (`crates\onnx-genai-cli\src\interactive.rs:612-619`, `crates\onnx-genai-cli\src\interactive.rs:873-879`, `crates\onnx-genai-cli\src\interactive.rs:973-976`). The new default should invert this for TTY interactive sessions.

### What to show per turn

Use the existing compact stats fields:

- prompt tokens (`N in`);
- output tokens (`N out`);
- decode throughput (`tok/s`);
- TTFT (`ttft N ms`);
- prefix/multimodal reuse (`N reused`);
- encoder cache hit ratio when relevant (`encoder H/T`);
- peak RSS (`rss N`);
- optionally finish reason if it fits without wrapping; otherwise keep finish reason in `/profile`.

These are already available in `RunProfile::to_stats_line()` (`crates\onnx-genai-cli\src\profile.rs:214-256`) and populated after generation (`crates\onnx-genai-cli\src\output.rs:213-220`, `crates\onnx-genai-cli\src\interactive.rs:958-1013`).

### Rendering

- **TTY + inline live:** live status line updates mid-turn with tokens/rate/TTFT; final stats row stays just above the next input box.
- **TTY + no live/editor failure:** print one compact stats line to stderr after the answer, as today.
- **Non-TTY:** keep byte-stable default for now to protect e2e tests and scripts. Do not start printing stats by default when stdin or stdout is not a TTY. This means "stats by default" is an interactive-TTY default, not a pipe default.
- **With `--profile`:** keep suppressing compact stats when the full profile report is on to avoid duplicate reports (`crates\onnx-genai-cli\src\output.rs:100-110`).

### Opt-out

- Add startup flag: `run --no-stats` (or `--stats=off` if a tri-state is preferred later).
- Keep `/stats` as a runtime toggle, but update text to `per-turn stats disabled/enabled`.
- Add `/stats compact|full|off` later if the profile UX wants levels.

## 6. Migration and compatibility plan

Non-TTY must stay byte-stable for `crates\onnx-genai-cli\tests\repl_e2e.rs`. Migration rule: **if stdin or stdout is not a TTY, run the existing line-loop semantics.**

1. Split the REPL into two paths:
   - `run_repl_plain`: current `read_line` loop, `>>> ` prompt, empty-line exit, no rich editor, plain help. Used when stdin or stdout is not a terminal and by a `--plain`/`--no-live` escape hatch.
   - `run_repl_tty`: rich editor, inline live viewport, stats default.
2. Keep parser behavior shared through the registry so tests still cover command handling.
3. Preserve current prompt and banner bytes in `run_repl_plain`; do not update `repl_e2e.rs` until a specific test opts into TTY behavior through a pseudo-terminal harness.
4. Add unit tests for the command registry/completion independent of TTY.
5. Add a small pseudo-terminal integration test later for rich editing only if the project already accepts that test dependency; do not block phase 1 on it.
6. Keep live rendering disabled on stdout pipes exactly as today (`crates\onnx-genai-cli\src\live_turn.rs:24-32`, `crates\onnx-genai-cli\src\live_turn.rs:84-95`).

## 7. Phased delivery plan

### Phase 1 — useful fastest

- Add TTY/plain path split without changing non-TTY bytes.
- In TTY path, enable stats by default and add `--no-stats`.
- Replace `>>> ` TTY prompt with a rich editor using `reedline`.
- Add persistent history file, multiline input, bracketed paste, cursor movement, and basic history search.
- Add declarative command registry for existing commands only; generate `/help` from it.
- Add slash-command completion for existing commands plus `/model` filesystem, `/ep`, `/backend`, `/profile verbosity`, `/image`, and `/audio`.

### Phase 2 — session/runtime interaction shell

- Add command registry entries for runtime capabilities accepted from `docs\research\cli\04-runtime-capability-inventory.md`.
- Add `/fork`, `/rewind`, richer `/session` subcommands, and command-specific argument completion from live runtime/session state.
- Add inline command result cards for session tree, KV pages, and profile summaries.

### Phase 3 — richer visual mode if needed

- Add optional `--fullscreen` alternate-screen experiment only if inline popups become limiting.
- Add mouse/picker UI for model/session/runtime selections.
- Add theme/color/accessibility modes if actual maintainer use shows a need.

## Final recommendation

Keep ratatui's inline viewport and add a TTY-only rich editor. Choose `reedline` first, with a `rustyline` fallback if crossterm/version ownership conflicts with ratatui 0.30.2. Phase 1 should deliver stats-by-default for TTY, persistent multiline editing/history, slash-command completion from a declarative registry, generated `/help`, and zero non-TTY behavior changes.
