
# Ferro — History

## 2026-07-18T04-55-00Z — Scribe session update

- Audited GLM IndexShare selected-token attention versus DeepSeek CSA; made no code changes and blocked implementation pending a frozen GLM private-op ABI, index semantics, parity/order, and mask/cache contract.

- 2026-07-27T10:09:19Z: Roadmap wave landed: #239/#246/#249/#248/#256/#263/#259 plus fmt gate #264; reviewer-lockout protocol enforced where required.

## 2026-07-27T13:12:20+00:00 — Roadmap wave-5

- Reviewed PR #266: rejected the naive ReduceLogSumExp overflow path, then approved Deckard's stable two-pass CUDA reduction and large-value parity regression.
# 2026-07-27 — Roadmap wave-6

Approved PR #269 after confirming IsInf, IsNaN, PRelu, CPU IsNaN parity, claim gates, and targeted CPU/GPU validation.

## 2026-07-27T16:44:54Z — Wave 8 update
- Reviewed Keaton PR #276 for #87 and requested changes: GPU test literal build break plus WAR-racy `drive_double_buffer` path; Deckard owns fixes.

## 2026-07-27T16:44:54Z — Wave 9 update
Requested changes on PR #276, then approved Deckard's fix cycle for #87 async prefetch overlap.

## 2026-07-27T19:35:00Z — Roadmap wave update
- Reviewed PR #294, requested changes, then approved after the aarch64 test-build break was fixed.
