# pris — History

## Summary through 2026-07-27T19:35:00Z
- Earlier history was compacted by Scribe because the file exceeded 15KB. Durable project decisions remain in `.squad/decisions.md` and archives.
- Pris has repeatedly owned test infrastructure, fixture quality, coverage hardening, metadata/schema validation, CPU/CUDA dispatch correctness, and reviewer-driven fix cycles.
- Notable prior outcomes include tiny-LLM fixtures, KV/session tests, Mobius export coverage, CPU-EP op coverage fixes, CLI ORT CI lane, Mac CPU bench harness guards, and dispatch reachability auditing.

## Recent retained entries

## 2026-07-27T12:11:15-07:00 — CLI CI coverage lane
- Added a dedicated `onnx-genai-cli` CI lane so ORT-linked CLI build/test/clippy coverage is no longer hidden behind the offline crate allowlist. The lane is isolated because `onnx-genai-ort-sys` downloads pinned ONNX Runtime 1.27.0 prebuilt archives when no local ORT is configured.
- Verified the Linux lane ran `a_turn_that_stops_inside_the_reasoning_says_it_has_no_answer`; observed green run 30298789423 cost 1m13s on Linux and 6m48s on Windows.

## 2026-07-26T22:38:02+00:00 — ORT2 remaining-work audit

- Recorded that ORT2 Phase 1 is complete, full ORT2 runtime vision is roughly 65–70% complete, and core GenAI functionality is roughly 70% complete; remaining work is breadth, compatibility, heterogeneous placement, packages, CI, and productization.

### 2026-07-27 — CLI maintainer-tool backlog queued
Justin confirmed the onnx-genai CLI is a development/maintainer harness, not a consumer product. P0 CLI work in docs/research/cli/00-backlog.md is queued under that charter: live stats discoverability, structured maintainer output, batch/bench harnesses, explicit dev flags for engine behavior, and help snapshots/REPL help. Remote-client mode is out of scope.

## 2026-07-27T07:35:00-07:00 — PR #227 reviewer-comment fixes
- Fixed `--decode-skip 0` inflating decode tok/s by subtracting TTFT instead of `Duration::ZERO`; extracted `decode_throughput()` helper.
- Fixed `--profile-json -` in non-direct mode emitting invalid JSON (markdown + JSON mixed on stdout); mirrored direct-mode stderr routing.
- Added `decode_throughput_skip_0_1_2` test with guard-break proof; all 9 `compare` tests pass.
- Published figures unaffected: profile README used `--decode-skip 2`.

## 2026-07-27T08:11:00-07:00 — SDPA test helpers cfg-gated for x86_64 CI
- Gated `deterministic_values`, `PatternBias`, `PatternMask`, `sdpa_f64_reference` with `#[cfg(target_arch = "aarch64")]` to match their consuming tests.
- Without gating, these helpers compiled as dead code on x86_64 and x86, causing `-D warnings` CI failure.
- Chose precise `cfg` gating over `#[allow(dead_code)]` to avoid silencing future genuine dead-code findings in this module.
- Verified: `cargo clippy --all-targets --target x86_64-apple-darwin -- -D warnings` passes; native aarch64 clippy and 13 SDPA tests pass.

## 2026-07-27T08:40:00-07:00 — Regression guard hardening: dispatch test + raised floors
- Added `fp16_m1_decode_reaches_neon_gemv_not_half_gemm` dispatch-reachability test with `GEMV_F16_TEST_HITS` atomic counter in `matmul.rs`. Uses f16×f16 M=1 tensors matching real model dtype.
- Guard-break verified: test fails on current HEAD (before Iran's M=1 gate in `try_matmul_half`); passes with gate applied locally.
- Raised FP32 absolute floor from 3.50 → 18.0 tok/s, roofline fraction from 0.30 → 0.35.
- Added new FP16 floor test: absolute 28.0 tok/s, roofline fraction 0.25. Would have caught the 4.5× regression (13.37 < 28).
- All machines check roofline fraction; measurement rig additionally checks absolute floor.
- x86_64 cross-compile clean; aarch64 clippy clean; 132/133 matmul tests pass (1 expected failure: dispatch test correctly fails until Iran's fix lands).
- Added aarch64-only `sdpa_f32_neon` parity coverage against scalar and f64 references on Qwen-style decode, odd/tail dimensions, masks/`-inf`, causal/softcap, and large-score softmax stability cases.
- Added a dispatcher reach test proving `sdpa_f32(...)` executes the NEON path on Apple Silicon when MLAS is not selected.
- Guard-break probe skipped the `dot_neon` scalar tail and the new parity test failed (`max_abs=9.221658e-4`, `max_rel=2.034264e0`); restored code passes.
- Tightened model-scale GEMV max-relative tolerance from 2.0% to 1.8%, based on Chew's 1.57% measured worst legitimate f32 accumulation-order drift.

## 2026-07-27T15:25:00-07:00 — REPL e2e output assertion hardening
- Fixed the flaky `piped_help_with_an_argument_still_prints_full_help` regression test by comparing stdout-only REPL help listings instead of merged stdout+stderr, which can contain ONNX Runtime/tracing timestamps.
- Audited `crates/onnx-genai-cli/tests/repl_e2e.rs` and split several command/error assertions so stdout help/session-continuation checks are not coupled to stderr logs.
- Guard reasoning: a pre-fix plain/piped `/help anything` that prints command-specific help would still differ from bare `/help` on stdout and fail the equality check.
## 2026-07-27T14:35:00-07:00 — Dispatch-branch coverage audit (PR #275 blocking bugs)

- **Finding: 12 of 13 reachable dispatch combinations in `matmul.rs` had zero test coverage** while codecov reported PASS. Line coverage (78%) masked the gap.
- Added 8 new dispatch-reachability tests with atomic hit counters:
  - `fp16_m1_column_major_b_reaches_colmaj_gemv`
  - `fp16_m1_non_constant_colmaj_b_does_not_reach_gemv`
  - `f16_m_ge2_non_constant_non_contiguous_b_does_not_enter_rescue` ← **THE BUG GUARD**
  - `f16_constant_non_contiguous_b_enters_rescue_block`
  - `f16_constant_non_contiguous_non_colmaj_b_enters_rescue`
  - `f16_non_constant_non_contiguous_b_produces_correct_result`
  - `f32_m_ge2_does_not_enter_half_or_rescue_paths`
  - `bf16_non_contiguous_does_not_enter_f16_rescue`
- Added two new static counters: `GEMV_F16_COLMAJ_TEST_HITS`, `NONCONTIG_RESCUE_TEST_HITS`.
- **Guard-break evidence:** removing `constant_inputs[1]` guard → test fails with exact message proving wrong dispatch.
- Region coverage improved: 79.6% → 88.8% (+9.2pp).
- Recommended enforcement: every new dispatch branch ships with a `_TEST_HITS` reachability test.
- Decision filed: `.squad/decisions/inbox/pris-dispatch-coverage-audit.md`.
- Commit: `17be7087` (coordinated with Iran's fix in same commit).

## 2026-07-27T15:17:00-07:00 — Dispatch-reachability CI lint

- Implemented `scripts/check_dispatch_reachability.py`: enforces that every `static ...TEST_HITS` counter has a corresponding `#[test]` reading it.
- Wired into `.github/workflows/ci.yml` alongside `check_platform_naming.py`.
- Guard-break proof: commenting out `GEMV_F16_TEST_HITS.load(...)` → lint fails with instructive message citing PR #275 history.
- False-positive analysis: no matches on non-dispatch statics (requires `TEST_HITS` suffix), strips `//` comments before matching.
- Documented known gap: lint cannot catch a missing counter on a new branch (review-time responsibility).
- BNNS-fail fallback (13th combination) confirmed unreachable on current hardware; documented as acceptable risk.
- 91 files scanned, 5 counters all paired with tests on main.
- Decision filed: `.squad/decisions/inbox/pris-dispatch-reachability-lint.md`.

## 2026-07-27T19:35:00Z — Roadmap wave update
- Fixed PR #293 Unique data-dependent-extent coverage and recorded durable dispatch coverage/reachability audit lessons.
