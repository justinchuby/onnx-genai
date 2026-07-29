# melina — History

## 2026-07-28T21:15:00+0000 — Declarative I/O contracts shipped
- PR #373 for #231 merged as `61d3bdac` after Richter's approval.
- Established declaration-first, name-agnostic model I/O resolution; shared-buffer KV is operator-agnostic and attention sequence lengths are validated strictly.

## 2026-07-29T00:45:00+0000 — Name-agnostic core decode path landed
- PR #380 for #377 merged as `47c3331d` after Cohaagen approved the fix-delta re-review.
- Removed core decoder/proposer I/O-name guessing: roles require explicit metadata or one unique shape candidate; ambiguous fixtures now declare component decoder I/O.
