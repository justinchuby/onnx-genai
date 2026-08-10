# Holden — History

## Project context
- Security engineer for onnx-genai. Focuses on unsafe/resource/supply-chain, path confinement, FFI, allocation bounds, and adversarial tests.
- Joined 2026-07-12 after phases 1-4, tool use/grammar/chat-template, Qwen2.5-0.5B, Hermes E2E, and static-cache KV work were established.

## Condensed prior record through 2026-07-27
- Audited unsafe/resource/supply-chain posture; cargo-audit had no active vulnerabilities and recurring audit workflow was established.
- Repeatedly reviewed ORT2/EP safety surfaces: DeviceBuffer ownership, strided bounds, C API FFI guard behavior, checked storage/shape overflow, dtype fail-close, loader validation, control-flow allocation bounds, and CUDA executor host/device safety.
- Enforced reviewer lockout patterns on security rejects: unchecked symbolic-dim arithmetic, thread-cap parsing, PyO3 unsendable engine cross-thread panics, CUDA SequenceAt/Scan host-pointer misuse, and other safety blockers were fixed by different agents before approval.
- Helped harden CUDA/native claims and package/loader boundaries by requiring fail-closed dtype/attribute handling, bounded allocations, and explicit validation rather than silent fallbacks.
- Recent consolidated decisions remain authoritative in `.squad/decisions.md`; this history was summarized by Scribe because it exceeded the 15KB threshold.

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
- Full report: `docs/EP_PLUGIN_EXPORT_SECURITY_AUDIT.md`.
- `compute.rs` and `kernel_ctx.rs` flagged for re-audit when Phase 2 Compute lands.

## 2026-08-10T21:30:26+0000 — EP plugin re-audit (commit 526a883c4)
- Re-audited Nabil's remediation commit on `squad/ep-plugin-export`.
- H1 (AtomicPtr), H2 (null graphs guard), H3 (Send/Sync removed): all RESOLVED.
- C1 (catch_unwind): PARTIAL — 12 of 13 callbacks guarded; `compute_execute` (compute.rs:119) left unguarded. Reinstates CRITICAL.
- New CRITICAL (N1): `compute_execute` has no `catch_unwind`; `kernel.execute()` and slice indexing can panic across C ABI. Owner: Deckard.
- New HIGH (N2): negative dims wrap to `usize::MAX` in `kernel_ctx.rs:154`; compounds N1. Owner: Deckard.
- New MEDIUM (N3): macro-generated `CreateEpFactories`/`ReleaseEpFactory` lack `catch_unwind`. Owner: Nabil.
- Verdict: 🔴 RED. Decision record: `.squad/decisions/inbox/holden-ep-plugin-reaudit.md`.
