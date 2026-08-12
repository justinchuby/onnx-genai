# Luv — History (compacted 2026-08-12)

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
- A reviewer's "SAFE" is not proof; verify the load-bearing claim independently.
- A reviewer's blocker can be a false positive; verify reviewer claims the same way we verify author claims.

## Historical context (pre-2026-08-12)

Wave 2026-07-28/29: CUDA coverage batch 7 merged (PR #338); PR #411 tiny-reasoning fixture rounds 2 and 3; reviewer lockout held throughout. Full detail in `history-archive.md`.

Wave 2026-08-11: Reviews of PR #31974 (BFloat16 LayerNorm/RMSNorm CPU EP, conditional approve → approve after S1 fix), B4+B6 test rework, PR #762 third review (no blockers, S1 optional-slot vacuity confirmed), PR #31985 (ort-docfix, NITS only), PR #32001 (Apple Accelerate CMake, three substantive, no blockers). Full detail in `history-archive.md`.

## Archive pointer

Older entries in `history-archive.md`.

## 2026-08-12 — Review #31973 v3 (AVX2 LayerNorm blocker fix)

Fresh adversarial review of `nxrt/mlas-avx2-layernorm` @ `72e02cd92c` after the architecture-specific dispatch threshold fix. Key findings:
- **Threshold fix is genuine**, not cosmetic — old test would not have caught the original RVV bug because both production and test shared the same wrong universal-8 assumption. New architecture-specific constant correctly encodes the real contract.
- **Three stale Welford comments** (test_layernorm.cpp:275,646,1055) describe the wrong algorithm after the Welford→centered-two-pass rewrite. Substantive — fix before leaving draft.
- **CatastrophicCancellation honestly fixed** — two new scenarios with condition < 1e7 make the accuracy branch actually execute.
- All five non-blocking fixes landed honestly. No leaks, no performance overclaims.
- **Not ready for draft exit** until stale Welford comments fixed.
- Head reviewed: `72e02cd92c`. Wrote `.squad/decisions/inbox/luv-review-31973-v3.md`.
