# Decisions

Canonical, append-only record of accepted team decisions. Only the Coordinator (via Scribe merge) writes here. Agents drop proposals in `decisions/inbox/`.

---

### 2026-07-12T14:50:00-07:00: Decisions archive rollover
**By:** Scribe
**What:** Archived the fully merged canonical decision file to `decisions/archive/2026-07-12T14-50-00Z-decisions.md` after the publish/audit merge pushed `decisions.md` above 20480 bytes.
**Why:** Keep the hot decisions file small while retaining the complete merged inbox record, including publish, CI, audit, security, and speculative-runtime notes, under `.squad/decisions/archive/`.

---

### 2026-07-12T14:50:00-07:00: PUBLISHED to crates.io: onnx-genai v0.1.0 and seven sub-crates
**By:** Scribe
**What:** Published `onnx-genai v0.1.0` plus all seven sub-crates to crates.io: `onnx-genai-metadata`, `onnx-genai-kv`, `onnx-genai-scheduler`, `onnx-genai-ort`, `onnx-genai-ort-sys`, `onnx-genai-engine`, and `onnx-genai-server`. The release contract is `.github/workflows/publish.yml` using the protected `crates` environment and `CARGO_REGISTRY_TOKEN`; it publishes leaves-first, checks crates.io before each publish, skips versions already present, uses a User-Agent header for registry API calls, and is safe to re-run idempotently. Future releases are performed by bumping the workspace version and re-running the workflow.
**CI/Audit:** `.github/workflows/ci.yml` runs fmt/build/test on push and PR; clippy remains non-blocking until warning cleanup makes `-D warnings` viable. `.github/workflows/audit.yml` runs weekly cargo-audit and on dependency changes; fresh `cargo audit` found 0 vulnerabilities. Audit code regularly via scheduled cargo-audit plus periodic review passes.
**Security:** Batched-driver DoS findings were fixed with bounded active+pending admission (`max_pending`, HTTP 429 + `Retry-After`) and non-blocking bounded delivery that drops slow/closed clients instead of stalling the shared driver.
**Speculative runtime:** §27/§28 includes prompt-lookup n-gram speculation (`NgramProposer`, greedy-identical), `MtpProposer` and `SpeculativeMode::Mtp`, a full ignored tiny-MTP package/e2e fixture (`tiny-mtp-full`) that matches greedy, an EAGLE-3 fixture with proposer TBD, and speculator config auto-discovery for vLLM-style metadata. Remaining optional speculative work: EAGLE-3 proposer and optimized full-MTP hidden-output decode.
**Model policy:** Agents use task-appropriate models with `gpt-5.5` as the floor.
**Why:** The repository is now published, continuously validated, routinely audited, and has the complete runtime milestone set recorded in the canonical team log.

---

### 2026-07-12T14:50:00-07:00: Release and audit conventions
**By:** Scribe
**What:** Publishing is via `.github/workflows/publish.yml` using the protected `crates` environment; bump the workspace version and re-run the workflow for future releases. Audit code regularly through scheduled cargo-audit plus periodic review passes.
**Why:** These contracts should be durable and visible to future agents before they change release or security workflows.
