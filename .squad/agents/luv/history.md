# Luv — History (compacted 2026-07-29)

**Role:** Code reviewer for correctness/safety gates across decode, sampling, KV, concurrency, CPU/CUDA EP coverage, and API contracts. Validate with real exit codes and mutation/guard-break evidence; rejection triggers strict reviewer lockout.

## Durable lessons
- Never approve on style alone: verify builds/tests/fmt/clippy or equivalent real exit codes, and prefer mutation/guard-break proof for regression tests.
- Reviewer lockout facts remain canonical: Batty locked out on #14 vision token expansion; Pris locked out on Bitwise/Hardmax until Deckard revision; Batty locked out on PR #287 backend flag until Deckard revision; Batty locked out on PR #411 round-2 fixture until Leon revision.
- MatMulNBits direct-int4 four/eight-column GEMV tiling regressed from register pressure, spills, and non-contiguous packed-weight streams; one-column GEMV remains production.
- Unique kernel history: unsafe String execution/UB was rejected; final support is numeric/bool/bf16 with String reported unsupported.
- CPU EP f64 matters: f32-only activations and f64 narrowing were rejected; true-f64 correction was required.
- All CUDA EP kernel work must remain correct and fast across supported SM architectures, not only sm_90.
- CUDA coverage #338 moved Pad/Range parity forward, but #67 remains open; ScatterND, quantization, and cuDNN work were deferred.
- CLI sampling/context review invariants: explicit `--max-new-tokens` preserved, per-turn REPL budget recomputed, finite unknown-context fallback, correct context arithmetic, sampling/greedy semantics, non-TTY stability.
- Backend flag contract: reporting must show resolved backend, not requested `auto`; clarify/test whether shared `--backend` on `transcribe` is intentional.
- Run a new test in isolation before believing it; a full parallel-suite green can be a fluke.
- A near-deterministic token stream cannot witness sampling through its tokens; PR #411 token-stream assertion was ~95% false-fail, and stderr timestamps did not prove stdout diversity.
- For sampling-policy regressions, instrument the generation boundary: `--stats`/`--profile` must reflect the policy actually resolved for the turn, not a separate or stale `/session` view.

## Recent work (current wave, ~2026-07-28/29)

### 2026-07-28T09-10-28+00-00 — CUDA coverage batch 7 merged
- PR #338 (`c59383db`) added CUDA `Pad` and `Range`, moving CUDA coverage 134→136 and standard CPU parity 105→107/141. Freysa approved after 174/174 H200 GPU 2 parity cases, coverage validation, content-corrupting mutation proof, and clean default-target Clippy. #67 remains open; ScatterND, quantization, and cuDNN work are deferred.

### 2026-07-29T12:30:00Z — tiny-reasoning-fixture rounds 2 and 3 (PR #411)

#### Round 2 REJECT
Ran Batty's statistical token-stream test alone in isolation: 15/15 failures with the fix intact. One green in full parallel suite was a fluke. Supporting evidence ("8/8 distinct outputs") was a stderr-timestamp artifact; test compared stdout only. Issued REJECT; Batty locked out. Also diagnosed: at `temperature 0.6, top_k 20` decode is near-greedy — 80/80 no-flag runs byte-identical to the greedy stream. The token-stream assertion is ~95% false-fail, not false-pass; raising the run count or picking a seed does not rescue it.

#### Round 3 APPROVE (commit `f8ed4fb4`)
Verified by building, running, and mutating — not by reading the report. Isolation: 10/10 PASS both new policy tests. Full suite: 44/44. Mutation (per-turn resolution disabled): both tests FAIL 3/3 deterministically; `/session`-keyed tests stayed GREEN. Mutated stats line: `greedy=true temperature=1 top_k=0` — the #385/#392 regression. Running the mutated binary: `--stats` reported `greedy=true` while `/session` reported `greedy=false temperature=0.6 top_k=20` — visible divergence proves stats reads the generation path. Issued APPROVE.

#### Delta APPROVE (commit `88fa86b5`)
Capture moved inside `run_generation_turn`. Mutation still bites 3/3. `turn` bound immutably; moved into `backend.generate(turn, …)` with no reassignment between capture and move. Divergence now impossible by construction. Full suite: 44/44. Issued APPROVE.

Durable rules recorded:
- "Run a new test in isolation before believing it."
- "A near-deterministic fixture cannot witness sampling through its tokens."
Full review detail in `.squad/decisions.md` ("Luv round-3 review" section, 2026-07-29). Inbox drops `luv-round3-verdict.md` and `luv-round3-delta-verdict.md` survived (written to both TEAM ROOT and worktree) and merged into decisions.

Full pre-compaction history in `history-archive.md`.
