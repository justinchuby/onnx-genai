# Hauser — History

## 2026-07-29T04:55:00+0000 — Recurrent shape inference landed

- Merged PR #386 (`39c28b44`), the #355 RNN/GRU/LSTM slice.
- Shared `recurrent()` covers symbolic dimension propagation, direction/hidden-size geometry, LSTM Y_c, declared-output handling, and permissive fallbacks.
- Opset 14 is the explicit layout boundary; pre-14 remains sequence-major.
