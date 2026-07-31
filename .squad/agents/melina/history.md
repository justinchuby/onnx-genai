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
