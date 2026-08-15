## 2026-07-17 — CUDA integer comparisons

- Landed `11da40d`: CUDA Equal/Greater/Less/GreaterOrEqual/LessOrEqual support same-dtype f32, int32, and int64 operands with Bool output and broadcasting; Equal also supports Bool.

- 2026-07-18: PR #25 lifecycle review rejected stale registration cache; Deckard owns revision.

## 2026-07-18T01:20:34Z — PR #25 lifecycle regression approved
- Approved `dbff29c`: real Environment lifecycle, last-drop cache clearing, and fresh registration attempt are covered; PR #25 merged.

- 2026-07-27T10:09:19Z: Roadmap wave landed: #239/#246/#249/#248/#256/#263/#259 plus fmt gate #264; reviewer-lockout protocol enforced where required.

## 2026-07-27T16:44:54Z — Wave 8 update
- Approved Dallas PR #272 for #47 DDPM + flow-matching schedulers; merged as 229c1401.

## 2026-07-27T19:35:00Z — Roadmap wave update
- PR #285 merged (`d889e85b`), closing #74 with CPU standard Conv without MLAS; fixed eager-dispatch cross-crate test breakage before merge.
