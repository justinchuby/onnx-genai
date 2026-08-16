# Luv — History Archive

## Archived 2026-07-29 (full pre-compaction snapshot)

# Luv — History

## 2026-07-12: Joined
Hired as an additional Code Reviewer (alongside Gaff) as the codebase grew to 9 crates with many concurrent workstreams. Project: onnx-genai, a Rust ONNX Runtime generative-AI inference runtime. Focus: correctness/safety gates on decode, sampling, KV, concurrency, and API contracts. Strict reviewer-lockout semantics apply on rejection. Validate green with real exit codes; never approve on style alone.


## 2026-07-13T18:30:00Z — Review/fix batch
- Reviewed Batty's issue #14 vision token-expansion wiring and rejected multi-image over-count plus missing `tokens_per_tile` guards; Batty was locked out and Leon owned the fix.

## 2026-07-16T00:00:02Z — MatMulNBits GEMV tiling result
- Evaluated four- and eight-column direct-int4 GEMV tiling; both regressed at 24 and 96 threads because of register pressure, spills, and non-contiguous packed-weight streams.
- Reverted the experiments and documented the negative result in `79c52a6`; the one-column GEMV remains the production path.

## 2026-07-16T00:00:00Z — CUDA M2 op-coverage delivery
- Landed `16c1e92`: f32 `com.microsoft::Silu` and standard-domain `ai.onnx::SimplifiedLayerNormalization` CUDA registrations, matching CPU EP coverage.
- Holden cleared independent parity checks; the CUDA suite passed 114/114.
- 2026-07-19T07:55:00Z: Approved PR #32 after capability, half-argmax, options-forwarding, retained-integration, and CI verification.

- 2026-07-19T12:40Z: Re-verified Bryant's conformance refresh counts (875 pass / 890 fail / 1,765 CUDA skip) and approved the measurement-only update.
## 2026-07-19T14:10Z — Bitwise/Hardmax review cycle
- 🔴 Rejected Pris's `43df6c0`, locking Pris out, then 🟢 approved Deckard's `7fe8961` revision for fp16/bf16 Hardmax and genuine bitwise broadcast/rejection coverage.


- **2026-07-19T16:15:00Z — CPU-EP review:** Rejected activation f32-only and f64-narrowing implementations, then approved Sapper’s true-f64 correction; activations landed as `39edb76`.


## 2026-07-19T18:20:00Z — CPU-EP op coverage 936→975

- Approved AffineGrid/Col2Im/CenterCropPad (`8e49948`) with a non-blocking Col2Im dilation-test nit.

- 2026-07-19: Drove Unique through three rejection cycles: O(n²)/NaN/dtype shortcomings, unreachable String execution, then runtime-layer String UB. Approved after unsafe String handling was removed; final kernel supports safe numeric/bool/bf16 and reports String unsupported.

## 2026-07-19T21:30:00Z — oneDNN removal review
- 🟢 Approved Bryant's `453d280` oneDNN CPU GEMM removal after verifying clean references/submodule removal, 620 CPU-EP library tests, 28 tracer tests, and registry-count integrity.
- 25 clippy lints observed remain pre-existing.


### 2026-07-20 — Vendored MLAS CPU-GEMM parity

Cross-agent update: vendored MLAS is now the opt-in CPU-GEMM parity path; follow-ups include buffer reuse, prepacked B, dtype coverage, int4, default flip, and Windows MASM.


## 2026-07-20T13:35:00Z — Multistream performance and issue #40

- Approved Sapper’s decode-pool residency and Roy’s guarded GQA parallelism after concurrency, numerical-order, opt-out, feature-gate, and E2E parity checks.

- 2026-07-21: Scribe reconciled the perf campaign inbox; key decisions are now consolidated in `.squad/decisions.md` under the 2026-07-21 perf campaign section.


## 2026-07-21 — Wave-2 and CI milestone
CI now covers all 27 offline crates with warnings-as-errors and native Windows ARM64. Capture-safe native fp16 CUDA decode wave 2 stacked GQA prep fusion, warp-shuffle RMSNorm, and specialized down-projection GEMV on wave 1, reaching 663–672 tok/s on H200 versus ORT GenAI at 657, with zero fallbacks. All CUDA EP kernel work must remain correct and fast across supported SM architectures, not only sm_90.

## 2026-07-21T13:15:00Z — GQA metadata fold landed
- Folded batch-1 GQA metadata into fused prep, removing 24 launches/token while preserving bounds/latch semantics and exact tokens. Holden approved; merged as `bd30e6c`, moving the stack from ~691 to ~710 tok/s at 256.
- 2026-07-21T23:55Z — Reviewed/approved VLM WP0 metadata and WP3 generic every_step executor; segment decisions now record both landings.
## 2026-07-22T12:00:00Z — Partial CUDA-graph Phase 0 landed
- Landed Phase 0 capture path-kind diagnostics on `main` as `3c94a57`, adding structural `CapturePathKind`/`SeamReason` metadata and seam labels without changing partitioning behavior. Deckard reviewed 🟢 GREEN.

## 2026-07-26T20:00:00Z — Scribe update

- 2026-07-26T20:00:03Z — Reviewed PR #203: reproduced +1.54% A/B and split-K numeric parity, requested changes for the `(896,97)` test routing to DownProjection instead of split-K.

## 2026-07-27T10:05:00-07:00 — PR #277 CLI sampling/context review
- 🟢 Approved Batty's CLI sampling/context fix after checking explicit `--max-new-tokens` preservation, per-turn REPL budget recomputation, finite unknown-context fallback, context arithmetic, sampling/greedy flag semantics, non-TTY stability, and the engine accessor-only API addition.
- Validation: `cargo build -q -p onnx-genai-cli`, `cargo test -q -p onnx-genai-cli --lib` (72 passed), and `cargo fmt -p onnx-genai-cli -- --check` passed; `cargo clippy -q -p onnx-genai-cli --all-targets -- -D warnings` hit only the known pre-existing `pages.rs:129` manual_checked_ops lint.

## 2026-07-27T14:49:32-07:00 — PR #287 CLI backend flag review
- 🔴 Rejected Batty's `--backend auto|ort|native` CLI flag because reporting paths use the requested backend (`auto`) rather than the resolved backend actually in use; Deckard should revise under reviewer lockout.
- Verified CLI build, CLI lib tests (79 passed), fmt, clippy, server build, invalid-value parser error, and text `--backend native` fail-loud behavior on a non-native build.
- Noted unresolved API-contract question: `--backend` now appears on `transcribe`; document/test it if intentional or split shared args if not.

## 2026-07-28T09-10-28+00-00 — CUDA coverage batch 7 merged
- PR #338 (`c59383db`) added CUDA `Pad` and `Range`, moving CUDA coverage 134→136 and standard CPU parity 105→107/141. Freysa approved after 174/174 H200 GPU 2 parity cases, coverage validation, content-corrupting mutation proof, and clean default-target Clippy. #67 remains open; ScatterND, quantization, and cuDNN work are deferred.

## 2026-07-29T12:30:00Z — tiny-reasoning-fixture rounds 2 and 3 (PR #411)

### Round 2 REJECT
Ran Batty's statistical token-stream test alone in isolation: 15/15 failures with the
fix intact. One green in full parallel suite was a fluke. Supporting evidence ("8/8
distinct outputs") was a stderr-timestamp artifact; test compared stdout only. Issued
REJECT; Batty locked out. Also diagnosed: at `temperature 0.6, top_k 20` decode is
near-greedy — 80/80 no-flag runs byte-identical to the greedy stream. The token-stream
assertion is ~95% false-fail, not false-pass; raising the run count or picking a seed
does not rescue it.

### Round 3 APPROVE (commit `f8ed4fb4`)
Verified by building, running, and mutating — not by reading the report. Isolation:
10/10 PASS both new policy tests. Full suite: 44/44. Mutation (per-turn resolution
disabled): both tests FAIL 3/3 deterministically; `/session`-keyed tests stayed GREEN.
Mutated stats line: `greedy=true temperature=1 top_k=0` — the #385/#392 regression.
Running the mutated binary: `--stats` reported `greedy=true` while `/session` reported
`greedy=false temperature=0.6 top_k=20` — visible divergence proves stats reads the
generation path. Issued APPROVE.

### Delta APPROVE (commit `88fa86b5`)
Capture moved inside `run_generation_turn`. Mutation still bites 3/3. `turn` bound
immutably; moved into `backend.generate(turn, …)` with no reassignment between capture
and move. Divergence now impossible by construction. Full suite: 44/44. Issued APPROVE.

Durable rules recorded:
- "Run a new test in isolation before believing it."
- "A near-deterministic fixture cannot witness sampling through its tokens."
Full review detail in `.squad/decisions.md` ("Luv round-3 review" section, 2026-07-29).
Inbox drops `luv-round3-verdict.md` and `luv-round3-delta-verdict.md` survived (written
to both TEAM ROOT and worktree) and merged into decisions.

## Archived 2026-08-12 (wave 2026-08-11 entries)

### 2026-08-11 — Review PR #31974 (BFloat16 LayerNorm/RMSNorm CPU EP)

Verdict: CONDITIONAL APPROVE. One substantive finding (contrib U=BFloat16 schema mismatch — pre-existing pattern from MLFloat16). No blockers. 10 tests, clean anti-fallback design, correct rounding (RNE via BFloat16 constructor). Noted code duplication. Full review at `.squad/decisions/inbox/luv-review-pr31974.md`.

### 2026-08-11 — Re-Review PR #31974 (S1 fix: U=float for narrow-float contrib kernels)

Verdict: APPROVE. Commit `142cb563c5`. Fix correctly changes contrib macro to `(T, U)` and registers `MLFloat16,float` / `BFloat16,float`. Verified declaration-only (no runtime change), contrib constructor sets `contrib_op=false`, CUDA parity confirmed, all 4 macro expansions correct. 10 bf16 tests pass.

### 2026-08-11 — B4 + B6 Test Rework for #31974

Deleted `test/mlas/unittest/test_layernorm_bf16.cpp` (zero MLAS API calls; 45 tests were testing BFloat16 rounding arithmetic, not PR code). Rewrote `test/contrib_ops/layer_norm_bf16_cpu_test.cc` (10 → 17 tests: added SkipLayerNorm, contrib opset 1–16, mean/InvStdDev float assertions). All 96 LayerNorm tests pass.

### 2026-08-11 — PR #762 third review (head 034876d30)

Verdict: No blockers. S1 (HIGH): optional-slot conformance tests may be vacuous — absent optional outputs use DataType::Undefined, EP likely declines these nodes. S2: LayerNorm axis allows `resolved == rank` (off-by-one). S3: Scratch buffer hardcodes 4 bytes/element. S1 confirmed by Mariette; fixes landed via Mariette/Challenger/Coco/Resch/Rachael chain.

### 2026-08-11 — Review PR #31985 (ort-docfix, MRotaryEmbedding doc)

Confirmed `mrope_section` required. Hand edit byte-exact to schema. Single-file single-line; no leaks. Verdict: NITS only. PR reached 86/86 CI green. (Duplicate entry existed; merged here.)

### 2026-08-12 — Review PR #32001 (Apple Accelerate CMake option)

Three substantive: S1 `FATAL_ERROR` wrong (upstream idiom is warn+disable); S2 no `build.py` argument; S3 `MLAS_USE_APPLE_ACCELERATE=1` defined but no consumer. No blockers. Lockout held: Luba/Luv barred, Isidore fixed.
