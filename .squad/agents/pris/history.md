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

## 2026-07-27T14:08:06-07:00 — CI supply-chain hardening and coverage
- Replaced personally-owned Rust setup/cache/install actions in `ci.yml` and `audit.yml` with direct `rustup`, GitHub-owned `actions/cache@v4`, and direct pinned `cargo install cargo-llvm-cov 0.8.7`.
- Converted coverage-capable test lanes to upload Codecov flags: `offline`, `mlas`, `cli-ort-linux`, and `cli-ort-windows`; verified final CI green at https://github.com/justinchuby/onnx-genai/actions/runs/30309892830.
- Confirmed CLI ORT Linux still executes `a_turn_that_stops_inside_the_reasoning_says_it_has_no_answer`; Windows CLI ORT stages the DLL into cargo-llvm-cov target paths with `--no-clean`.
- Documented Windows ARM64 coverage blocker (rust-lang/rust#150123) and release workflow debt in `.squad/decisions/inbox/pris-ci-supply-chain-and-coverage.md`.
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

## 2026-07-27T16:31:27-07:00 — CI concurrency cancellation
- Added top-level concurrency cancellation to `ci.yml` using PR-number grouping for pull requests and SHA grouping for push/main runs, preserving post-merge `main` signal while cancelling stale PR coverage runs.
- Added top-level concurrency cancellation to `audit.yml`, grouped by ref because audit has no pull request trigger.
- Deliberately exempted release/state-mutating workflows: `publish.yml`, `wheels.yml`, and the squad issue/label workflows.
- Verified by rapid pushes: run 30314651548 cancelled and run 30314662714 completed successfully.
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

## 2026-07-28T00:35:00Z — Per-PR benchmark CI workflow (PR #306)

- Implemented `.github/workflows/benchmark.yml`: separate benchmark workflow running kernel micro-benchmarks and hot-path benchmarks on every PR, posting a comparison comment.
- Design: benchmarks merge-base FIRST (cold-start), PR SECOND (warm runner). Systematic bias toward PR appearing faster reduces false positives.
- Workload: ep-cpu kernels (MatMul at M=1 decode + prefill shapes, Add, Gather, ReduceMean × f32/f16/bf16) + genai-bench no_model (tokenization, sampling, KV cache, logit processing, grammar).
- **Informational only** (per Justin's direction): does NOT block CI. Visual flags ⚠️ ≥15%, 🔴 ≥30% calibrated against measured runner noise (~27% worst-case).
- Real regression gates stay in `profile_native.rs` (throughput floors + dispatch-reachability tests on real hardware).
- Regression-detection proof: simulated 4.5× M=1 slowdown → six 🔴 lines at +350%, header "🔴 Benchmark Regression Detected". Unmissable for reviewer.
- CI proof: runs green on PR (22m on macOS arm64 M1 Virtual 3-core). Comment updated in place on re-push.
- Decision filed: `.squad/decisions/inbox/pris-benchmark-ci.md`.
## 2026-07-27T17:59:12-07:00 — PR #296 review fixes
- Fixed cargo cache key correctness: keys now include OS, runner architecture, actual target triple, rustc release, cached cargo tool version, and `Cargo.lock`; this prevents `windows-latest` and `windows-11-arm` from sharing `target/` or `~/.cargo/bin` artifacts.
- Confirmed `audit.yml` does not use clippy or rustfmt and removed those rustup components from the audit toolchain install.

## 2026-07-27T20:00:00-07:00 — CI fast/slow tier split
- Split CI so every PR push runs fast uninstrumented Linux x86_64 + Windows x86_64 Rust tests and CLI ORT e2e, while coverage, Windows ARM64, macOS arm64, CUDA compile, and audit move to slow/full triggers.
- Added `pull_request` `labeled` handling and created the `ci:full` label so labeling an open PR starts slow/full CI without a new push.
- Preserved #296 supply-chain/concurrency constraints: direct `rustup`, GitHub-owned `actions/cache`, arch-aware keys, and SHA-keyed non-PR concurrency.
- Measured fast dispatch run 30324179873 at 7m57s wall-clock; final slow dispatch run 30324584315 passed after rerunning a flaky Windows ARM64 failure, earlier equivalent slow dispatch 30322967863 passed at 15m30s, and audit run 30322969130 passed at 3m08s. Fast remains above the ~4m target because Windows x86_64 Rust tests are the critical path, but Windows correctness stayed on PRs.

## 2026-07-27T22:56:00-07:00 — CI tier refinement: Windows CLI-only fast path
- Refined fast CI so the broad offline-crate suite runs per PR only on Linux x86_64; Windows x86_64 stays per PR only for the CLI ORT lane (build/test/e2e/clippy), where Windows platform differences have actually bitten: CLI contracts, ORT DLL loading, filesystem/terminal/dynamic-linking boundaries.
- Kept the full Windows x86_64 offline-crate suite in slow CI via the Windows `Rust coverage` job.
- Audited offline crates for Windows-specific cfg tests before moving broad Windows tests out of fast PR CI; found `onnx-runtime-loader::pathsafe::tests::rejects_rooted_path` as the meaningful Windows-only test that now relies on slow CI, plus Windows-specific implementation code in tracer and CPU decode affinity.
- Remeasured final-SHA fast dispatch run 30334227247 at 4m33s wall-clock; critical path was `CLI ORT (Windows x86_64)` at 4m21s. The preceding same-workflow run 30333847821 was 3m59s with `Rust (Linux x86_64)` at 3m55s and Windows CLI at 2m51s; the first refined run was 8m09s due a Windows cache-save post step. Conclusion: the split reaches the target with warm caches, but Windows CLI variance can still exceed 4m.

## 2026-07-28T01:18:00-07:00 — Windows path-safety exception in fast CI
- Added `cargo test --locked -p onnx-runtime-loader pathsafe` to the fast Windows CLI lane so `rejects_rooted_path` continues to run before merge. This is the narrow exception to the pure-Rust-offline-on-Linux rule because Windows path semantics are the behavior under test.
- Rechecked the Windows cfg audit: no other meaningful `#[cfg(windows)]` tests in the offline crate suite; remaining hits are Windows-specific implementation code in tracer/decode affinity or ORT/CUDA code outside the offline crate set.
- Measured repeat fast run 30342667705 at 3m45s wall-clock; critical path `Rust (Linux x86_64)` at 3m40s, Windows CLI at 3m26s, loader path-safety step 12s. First run on the commit was 7m31s from Windows cache/build variance, with loader step 34s.
## 2026-07-28T04-08-08+0000 — Wave 2 regression/roadmap update
- CI supply-chain and coverage hardening note was merged into decisions.

## 2026-07-28T02:18:00-07:00 — Stable names for skipped slow CI jobs
- Fixed unexpanded skipped check names caused by job-level `if:` conditions on matrix jobs: GitHub skips before matrix expansion, producing raw `${{ matrix.name }}` in fast-tier skipped checks.
- Replaced the slow-tier matrices in `ci.yml` with explicit per-platform jobs so skipped checks are named `Rust coverage (Linux x86_64)`, `Rust coverage (Windows x86_64)`, `Rust slow platform (...)`, and `CUDA compile (...)`.
- Confirmed `miri.yml` has no matrix/gated job-name issue. Repo-wide scan found analogous conditional matrices only in release/manual workflows (`publish.yml`, `wheels.yml`), not PR CI; left them outside this PR.
- Recorded Justin's live `ci:full` verification on PR #340: CI run 30345861354 and audit run 30345861425 started via label.
- Added CLI-specific cache keys after the main merge exposed a shared-cache collision/incomplete-restore pattern in the Windows CLI lane. Remeasured final-head fast run 30349042835 at 4m33s wall-clock; critical path `Rust (Linux x86_64)` at 4m22s, with Windows CLI at 3m14s. The preceding workflow-code run 30348596658 was 3m51s; current variance is in Linux Rust/quality after the main merge, not the skipped-name fix.

## 2026-07-28T17:40:00+0000
#54 model-package and #299 LoRA were confirmed out of scope for this squad; no artifacts retained.
