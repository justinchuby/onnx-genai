## 2026-07-17 — CUDA integer comparisons

- Landed `11da40d`: CUDA Equal/Greater/Less/GreaterOrEqual/LessOrEqual support same-dtype f32, int32, and int64 operands with Bool output and broadcasting; Equal also supports Bool.

- 2026-07-18: PR #25 lifecycle review rejected stale registration cache; Deckard owns revision.

## 2026-07-18T01:20:34Z — PR #25 lifecycle regression approved
- Approved `dbff29c`: real Environment lifecycle, last-drop cache clearing, and fresh registration attempt are covered; PR #25 merged.
