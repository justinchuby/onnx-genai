## 2026-07-28T05-49-08+0000 — Wave 3 update
Fixed PR #322 security blockers with HostTrust, open_with_trust, and symlink-resolving canonicalize_confined confinement. Nine adversarial tests were mutation-proven; manifest-only packages can no longer escape package root.

## 2026-07-28T11:20:06+0000 — #75 strict-lockout review cycle
- Requested changes on PR #346 for incorrect default-domain registration and LabelEncoder-1 dtype selection.
- Re-approved after Rachael, not the locked-out original author Bryant, independently corrected both defects in `c20ec211`.

## 2026-08-10T20:12:35+0000 — Outbound EP plugin FFI audit
- Full security audit of `crates/onnx-runtime-ep-plugin/` (~1,300 LOC of raw FFI).
- 1 CRITICAL: no `catch_unwind` on any `extern "C"` callback — panic across FFI is UB.
- 3 HIGH: `static mut` data race on HOST_ORT_API, missing null-check on `graphs` in `ep_compile`, unsound blanket `Send+Sync` on `OutboundGraphReader`.
- 2 MEDIUM, 1 LOW (hardening, non-blocking).
- Decision record filed: `.squad/decisions/inbox/holden-ep-plugin-ffi-audit.md`.
- Full report: `docs/ep-plugin/EP_PLUGIN_EXPORT_SECURITY_AUDIT.md`.
- `compute.rs` and `kernel_ctx.rs` flagged for re-audit when Phase 2 Compute lands.

## 2026-08-10T22:42:21+0000 — EP plugin final ship verdict (squad/ep-plugin-export)
- Verified N1 (CRITICAL): `compute_execute` at `compute.rs:552` is wrapped in `catch_unwind`; regression test at line 2115 confirmed. RESOLVED.
- Verified N2 (HIGH): `validate_dims()` called from `read_inputs()` at `kernel_ctx.rs:193`; rejects negative dims, `checked_mul` for overflow, eight tests. RESOLVED.
- Verified N3 (MEDIUM): both `CreateEpFactories` and `ReleaseEpFactory` wrapped in `catch_unwind` in `lib.rs`; `ReleaseEpFactory` return type corrected to `void`; tests present. RESOLVED.
- Verified Deckard's `factory.rs` UAF fix (commit `c92838dba`): `EpDevice_AddAllocatorInfo` ownership transfer correct; success path does not release `mem_info`; failure path releases exactly once; `CreateMemoryInfo_V2` used. CORRECT.
- New LOW advisory: `compute_release_state` (`compute.rs:1416`) missing `catch_unwind` — trivially safe now but pattern violation. Filed for Leon post-merge.
- New LOW advisory: `ep_compile_inner` partial info leak on error path — carry-forward M2. Filed for Deckard post-merge.
- Broader new-code audit: graph_reader.rs attribute/initializer reading, ep.rs capability filter, factory.rs lifetime — all sound.
- Verdict: 🟡 YELLOW — may ship. No blockers. Decision record: `.squad/decisions/inbox/holden-ep-plugin-final-verdict.md`.

## 2026-08-10T21:30:26+0000 — EP plugin re-audit (commit 526a883c4)
- Re-audited Nabil's remediation commit on `squad/ep-plugin-export`.
- H1 (AtomicPtr), H2 (null graphs guard), H3 (Send/Sync removed): all RESOLVED.
- C1 (catch_unwind): PARTIAL — 12 of 13 callbacks guarded; `compute_execute` (compute.rs:119) left unguarded. Reinstates CRITICAL.
- New CRITICAL (N1): `compute_execute` has no `catch_unwind`; `kernel.execute()` and slice indexing can panic across C ABI. Owner: Deckard.
- New HIGH (N2): negative dims wrap to `usize::MAX` in `kernel_ctx.rs:154`; compounds N1. Owner: Deckard.
- New MEDIUM (N3): macro-generated `CreateEpFactories`/`ReleaseEpFactory` lack `catch_unwind`. Owner: Nabil.
- Verdict: 🔴 RED. Decision record: `.squad/decisions/inbox/holden-ep-plugin-reaudit.md`.

## 2026-08-10T23:09:23+0000 — EP plugin milestone 2 audit (squad/ep-plugin-parity-cuda)
- Audited new device-side FFI surface: `device.rs` (~600 lines), `factory.rs` generalized device/allocator/stream, `ep.rs` kernel registry, `kernels/mod.rs` RecordingOpRegistry.
- M2-1 (MEDIUM): EP leaked in `stream_release` — `Box::into_raw`'d EP never freed. Assign Leon.
- M2-2 (LOW): Misleading doc comment on `DeviceAllocator::memory_info` ownership. Advisory.
- Verified NEW-1 (RESOLVED): `compute_release_state` now wrapped in `catch_unwind`.
- Verified NEW-2 (RESOLVED): `cleanup_partial_infos` frees and nulls partial `out_infos`.
- `#[repr(C)]` layout verified correct for `DeviceAllocator` and `DeviceSyncStream`.
- All 14 new `extern "C"` callbacks: panic-guarded or trivially non-panicking.
- `mem::forget` sites independently verified safe (DeviceBuffer has no Drop impl).
- RecordingOpRegistry fail-closed by construction (under-advertises, never over-advertises).
- Verdict: 🟡 YELLOW — may ship. No memory-safety or corruption issues. Decision: `.squad/decisions/inbox/holden-ep-milestone2-audit.md`.

## 2026-08-11 — Independent Re-Review: PR #31973 (AVX2 LayerNorm)

- Adversarial re-review of AVX2 LayerNorm/RMSNorm kernel after two prior accuracy failures.
- Independently verified: 41/41 tests pass, double-precision first pass is genuine, B1 failure mode eliminated.
- Found: DISABLED_AdversarialPrecisionReport fails when enabled (Scenario 6 near-fp32-max overflow); main sweep tolerance has 12% headroom (fragile across CPU microarchitectures).
- Verdict: **READY FOR REVIEW** — no blockers. 2 substantive findings (S1: widen sweep tolerance, S2: fix DISABLED test assertion). Both owned by Pris.
- Decision: `.squad/decisions/inbox/holden-rereview-pr31973.md`.

## 2026-08-11 — Adversarial re-review of PR #31974 (BF16 LayerNorm CPU EP)
- Independent review of B2/B4/B5/B6/N1 fixes after 3 prior public corrections.
- Built and ran tests: 17/17 BF16-specific, 96/96 broader LayerNorm — all pass.
- B5 stat-narrowing bug genuinely fixed; stat tests would fail 78–1558× tolerance against pre-B5 code.
- B4 deleted file tested nothing in this PR (zero MLAS calls). B2 docs match all 5 registrations.
- N1 (MLFloat16 U=float) is a correct pre-existing bug fix, not just declaration-only.
- Found: git history leakage (commit 58b5d23246 exposes .squad/ path and persona name "pris"). Needs squash before merge.
- Verdict: **READY FOR REVIEW** — no blockers. 1 substantive (squash leaky commit), 1 nit (NarrowToFloat duplication).
- Decision: `.squad/decisions/inbox/holden-rereview-pr31974.md`.

## 2026-08-11 — Final sign-off: PR #762 (EP plugin parity CUDA)

- Ran all EP crate tests: 211+ pass, 0 fail. CPU E2E with real ORT: 23 pass. nxrt ABI roundtrip: 10 pass. CUDA plugin: 12 pass.
- Clippy clean on all EP crates.
- CPU EP genuinely usable by real ORT — confirmed end-to-end.
- CUDA is fail-closed by design, not by accident.
- nxrt ABI is real across a cdylib boundary (version negotiation, panic containment, inline status — no cross-module free).
- Docs are honest: all CUDA claims say "unvalidated on hardware", reference #768.
- No blocking issues found.
- Verdict: **APPROVE — leave draft**.
- Decision: `.squad/decisions/inbox/holden-final-762.md`.

## 2026-08-11 — PR #31974 re-review + PR #762 final sign-off

**PR #31974 re-review** (after B1-B6 fixes on `nxrt/mlas-bf16-layernorm`):
- Verdict: READY FOR REVIEW. B5 stat-narrowing fixed (WriteStat<U=float>); B4 test file deletion correct (zero MLAS calls); B6 17/17 BFloat16 tests pass, all 5 op families covered. Anti-fallback `ConfigEp(DefaultCpuExecutionProvider())` confirmed.
- Flagged `.squad/` leakage in git history of both upstream branches (content reachable after delete commit).

**PR #762 final sign-off** at `fb9d757b3`:
- Verdict: APPROVE. CPU EP E2E: 23/23 ORT conformance tests. CUDA: zero factories + actionable status, `catch_unwind` at 18+ sites. nxrt ABI: 10/10 roundtrip. Honesty sweep: clean.
- 211+ tests, 0 failed, 7 ignored. Clippy clean. Cross-platform c_char verified.

