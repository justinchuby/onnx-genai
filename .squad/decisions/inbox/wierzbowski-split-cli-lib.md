### 2026-07-27: Split CLI orchestration from presentation and REPL parsing
**By:** Wierzbowski
**What:** Split `crates/onnx-genai-cli/src/lib.rs` (3,559 lines before; 1,233 after) into `generate.rs` (219 LOC), `interactive.rs` (953), `commands.rs` (234), `output.rs` (232), `model_inspection.rs` (71), and `transcribe.rs` (709), retaining the existing `profile.rs`. `lib.rs` remains the CLI argument/type and dispatch facade.
**Why:** Cohesive private modules make generation, interactive orchestration, command parsing, presentation, model inspection, and transcription independently navigable without changing the crate's public surface, CLI shapes, or output text.

Ctrl-C wiring was moved intact into `interactive.rs`: the `Once`-guarded `ctrlc::set_handler` body retains its registration sites and order, the same `GENERATING`, `INTERRUPT_REQUESTED`, and `EXIT_ARMED` atomics with `SeqCst`, and the REPL still clears `EXIT_ARMED` immediately after a submitted line before parsing it. One-shot generation and transcription install the same handler at their original points.

Gates: `cargo build -p onnx-genai-cli` passed; `cargo test -p onnx-genai-cli` passed (127 tests total across targets); strict `cargo clippy -p onnx-genai-cli --all-targets -- -D warnings` is blocked by pre-existing unchanged `crates/onnx-genai-cli/src/pages.rs:129` (`clippy::manual_checked_ops`); clippy passes with only that lint allowed. `cargo fmt -p onnx-genai-cli -- --check` and `git diff --check` passed. Non-author code review found no significant issues.
