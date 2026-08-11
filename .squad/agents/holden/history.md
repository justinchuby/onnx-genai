# Holden — History (compacted 2026-08-11T23-30-00Z)

**Role:** Security engineer for onnx-genai. Focuses on unsafe/resource/supply-chain, path confinement, FFI, allocation bounds, and adversarial tests.

## Durable lessons
- Enforce reviewer lockout end-to-end: no author revises their own rejected artifact.
- Require `catch_unwind` on every `extern "C"` callback — panic across FFI is UB.
- `OrtGraph*`/`OrtNode*` must NOT be stored beyond callback return.
- `OrtMemoryInfo` passed to `EpDevice_AddAllocatorInfo` must outlive the `OrtEpDevice` — ORT stores the raw pointer; do not call `ReleaseMemoryInfo` on success.
- Leak scans must grep source content in the diff, not only `.squad/` paths and commit messages.
- "Not caused by us" is not the same as "safe to mark ready" — draft until CI board is green.

## Historical context

Pre-2026-08-11 entries archived in `history-archive.md`. Covers: Wave 3 path confinement, EP plugin FFI audit (CRITICAL catch_unwind, HIGH static mut race), EP plugin re-audit (C1 partial — compute_execute unguarded), EP plugin final ship verdict (🟡 YELLOW), EP milestone 2 audit (M2-1 stream_release leak), #31973/#31974 initial re-reviews (all blockers fixed; detected .squad/ leakage), PR #762 final sign-off (APPROVE).

## 2026-08-11 — PR #31974 re-review + PR #762 final sign-off

**PR #31974 re-review** (after B1-B6 fixes on `nxrt/mlas-bf16-layernorm`):
- Verdict: READY FOR REVIEW. B5 stat-narrowing fixed (WriteStat<U=float>); B4 test file deletion correct (zero MLAS calls); B6 17/17 BFloat16 tests pass, all 5 op families covered. Anti-fallback `ConfigEp(DefaultCpuExecutionProvider())` confirmed.
- Flagged `.squad/` leakage in git history of both upstream branches (content reachable after delete commit).

**PR #762 final sign-off** at `fb9d757b3`:
- Verdict: APPROVE. CPU EP E2E: 23/23 ORT conformance tests. CUDA: zero factories + actionable status, `catch_unwind` at 18+ sites. nxrt ABI: 10/10 roundtrip. Honesty sweep: clean.
- 211+ tests, 0 failed, 7 ignored. Clippy clean. Cross-platform c_char verified.

## 2026-08-11 — Re-review PR #31973 (AVX2 LayerNorm)

Fresh adversarial pass after rebase onto upstream/main. All 6 original blockers remain fixed. Found 2 substantive issues: internal name leaks ("Iran" at test_layernorm.cpp:1191, "Pris" at :492). Kernel is reachable, tests are non-vacuous, no upstream conflict. Verdict: NOT ready to leave draft until name leaks are scrubbed; otherwise clear. Triggered lockout-clean fix by Chew.

## 2026-08-11 (upstream CI correction wave) — Lessons enforced

Persona name leaks found only after two prior diff sweeps. Rule confirmed: grep the actual source content in the diff, not just `.squad/` paths. Two PRs converted back to draft — "infra flake" reasoning does not justify marking ready while CI is red.
