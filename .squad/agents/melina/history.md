# melina — History

## 2026-07-28T21:15:00+0000 — Declarative I/O contracts shipped
- PR #373 for #231 merged as `61d3bdac` after Richter's approval.
- Established declaration-first, name-agnostic model I/O resolution; shared-buffer KV is operator-agnostic and attention sequence lengths are validated strictly.

## 2026-07-29T00:45:00+0000 — Name-agnostic core decode path landed
- PR #380 for #377 merged as `47c3331d` after Cohaagen approved the fix-delta re-review.
- Removed core decoder/proposer I/O-name guessing: roles require explicit metadata or one unique shape candidate; ambiguous fixtures now declare component decoder I/O.

## 2026-07-30T13:36:00Z — CUDA TopK parity review
- Independent review of PR #445 (fp16/bf16 CUDA TopK operations); approved and merged.
- Work advances CUDA standard-domain parity from 159 to 161 ops as part of the tractable wave.

## 2026-07-31T00:25:00Z — PR #534 server-contracts review (approved, held)

- Reviewed and APPROVED Harry's PR #534 for server contracts #481/#482 (`build_dirty` Option<bool> present-as-null; truncated predicate uses actual scan size). HELD from main because it targets Justin's active branch `feat/genai-demo-dashboard` (PR #476); the described code exists only there.

## 2026-07-31T03:03:15Z — Reviewed #535 + #543 (both merged)

- Reviewed #535 (hybrid loader / text-only decode synthesis) and #543 (rank-3 mrope native positions). For #543 ran the e2e parity suite: 1 passed / 0 failed, 309 conformance cases; rank-2 byte-identical; native-CUDA hybrid decode == ORT token-for-token on real qwen3.5-0.8b. Both APPROVED and MERGED this wave.

## 2026-07-31T08:48:28Z — REQUEST-CHANGES then re-APPROVE #544

- Initial REQUEST-CHANGES on #544: the negative poison-control arm of `async_pagein_fence_orders_weight_page_in_consumer` was racy (wall-clock race). Harry fixed by event-ordering the negative transfer after consumer.
- Re-reviewed and APPROVED after Harry's fix (bf345904): green 5/5 parallel, non-vacuous fail 3/3. PR #544 MERGED.
