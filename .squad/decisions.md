# Decisions

Canonical, append-only record of accepted team decisions. Only the Coordinator (via Scribe merge) writes here. Agents drop proposals in `decisions/inbox/`.

---

### 2026-07-12T08:56:27-07:00: Expand Rust and Python gitignore coverage
**By:** Deckard
**What:** Added standard Rust backup/debug-symbol patterns and comprehensive Python cache, build, packaging, virtual environment, coverage, test, type-checking, lint, notebook checkpoint, and native extension ignore patterns to `.gitignore`.
**Why:** The repository is a Rust Cargo workspace that may include Python tooling, scripts, model conversion, or tests, so ignoring common generated artifacts keeps source control focused on intentional source files while preserving existing `/target` and `Cargo.lock` behavior.
