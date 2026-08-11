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

## 2026-08-11 — Review PR #31974 (BFloat16 LayerNorm/RMSNorm CPU EP)

Reviewed for @justinchuby. Verdict: CONDITIONAL APPROVE. One substantive finding (contrib U=BFloat16 schema mismatch — pre-existing pattern from MLFloat16). No blockers. 10 tests, clean anti-fallback design, correct rounding (RNE via BFloat16 constructor). Noted code duplication (NarrowToFloat/FloatToNarrow in two files, BFloat16 ComputeJob/BFloat16Math are near-clones of MLFloat16 versions). Full review at `.squad/decisions/inbox/luv-review-pr31974.md`.

## 2026-08-11 — Re-Review PR #31974 (S1 fix: U=float for narrow-float contrib kernels)

Re-reviewed commit `142cb563c5` for @justinchuby. Verdict: APPROVE. The fix correctly changes contrib macro to `(T, U)` and registers `MLFloat16,float` / `BFloat16,float`. Verified: (1) declaration-only, no runtime change — contrib LayerNorm constructor sets `contrib_op=false`, so `SrcDispatcher` always calls `ComputeImpl<T, float>`; (2) kernel matching improves for schema-compliant models, no breakage for existing valid models; (3) CUDA parity confirmed; (4) all 4 macro expansions correct; (5) recommended keeping MLFloat16 fix combined. 10 bf16 tests pass. Full re-review appended to `.squad/decisions/inbox/luv-review-pr31974.md`.

## 2026-08-11 — B4 + B6 Test Rework for #31974

**Requested by**: @justinchuby (reviewer rejection protocol)

### B4: Deleted `test/mlas/unittest/test_layernorm_bf16.cpp` (1037 lines)
- File called zero MLAS APIs despite living in the MLAS test directory
- 45 registered tests tested BFloat16 rounding/oracle arithmetic, not PR code
- The "45 MLAS kernel tests" claim was false — retracted

### B6: Rewrote `test/contrib_ops/layer_norm_bf16_cpu_test.cc` (10 → 17 tests)
- Added SkipLayerNormalization (3 tests)
- Added contrib LayerNormalization opset 1–16 (2 tests)
- Added Mean/InvStdDev float stat assertions at 1e-5 tolerance (2 tests)
- Dual tolerance: bf16 Y at 0.016 abs, float stats at 1e-5 abs
- Removed persona comments ("Chew", "Resch")
- All 96 LayerNorm tests pass, no regressions

### Honest test count: 17 BF16 CPU EP operator tests

## 2026-08-11 — PR #762 third review (head 034876d30)

**Task:** Third adversarial review of PR #762.

**Verdict:** No blockers. Substantive findings:

- **S1 (HIGH):** Optional-slot conformance tests may be vacuous — claim filter at ep.rs:275 rejects `DataType::Undefined` outputs; absent optional outputs use that dtype; EP likely declines these nodes; ORT falls back silently. BL2 compute-path fix is dead code in ORT plugin path.
- **S2 (LOW):** LayerNorm axis bounds allows `resolved == rank` (off-by-one vs ONNX spec).
- **S3 (LOW):** Scratch buffer hardcodes 4 bytes/element; unsafe for f64/i64.
- Confirmed Isidore's ABI fixes, Freysa's fallback disable, Sebastian's runtime axis genuine.

**Outcome:** S1 confirmed by Mariette. Fixes landed via Mariette, Challenger, Coco, Resch, Rachael chain. Reviewer lockout held.

## 2026-08-11 — Review PR #31985 (ort-docfix, MRotaryEmbedding doc)

**Task:** Adversarial review of one-line doc fix removing "(or omitting it)" from ContribOperators.md.

**Findings:**
- `mrope_section` confirmed required (`AttributeProto::INTS`, no default, description says "Required.")
- Hand edit is byte-exact to schema text at bert_defs.cc:2008
- No leaks, no scope creep, single-file single-line commit
- PR body accurate: names #31728, explains hand-edit reasoning

**Verdict:** NITS only. Ready to leave draft.

## 2026-08-11 (upstream CI correction wave) — Review PR #31985

Adversarial review of one-line `docs/ContribOperators.md` fix. Confirmed `mrope_section` is required (no default in `bert_defs.cc:2046–2051`). Hand edit byte-exact to schema text. Single-file, single-line commit; no leaks, no scope creep. **Verdict: NITS only — ready to leave draft.** PR subsequently reached 86/86 CI green.
