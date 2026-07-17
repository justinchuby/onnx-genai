# Brion — History

## 2026-07-15T17:52:18Z — Wave 2 CPU reductions
- Merged `b87aa27`: f32 `ReduceL1`, `ReduceLogSum`, and max-stabilized `ReduceLogSumExp`, including shape inference.
- ONNX backend CPU passes rose 720 → 735 (+15). Roy’s review found two blockers; Leon supplied the locked-out corrective fix.

- 2026-07-15 — Wave 4: Shipped CPU Clip numeric dtype coverage (`49778eb`); backend CPU coverage reached 741 and Roy reviewed 🟢.

## 2026-07-15 — Wave 6
- Corrected Resource-Governor wiring after Tyrell’s 🔴: checked byte parsing, YAML float fractions, provider-capacity clamp, and per-tier snapshots. Final `1eebf5d`; Tyrell 🟢.
