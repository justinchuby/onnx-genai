# quaid — History

## 2026-08-04T00:40:00Z — PR #625 initializer-exclusion revision

- Took over #625 under Cohaagen lockout and fixed the native loader metadata path to exclude initializer inputs, matching `graph_builder.rs`.
- Added metadata==Session KV-geometry parity coverage; HEAD `3b615953` was mergeable after Harry approved.

## 2026-08-06T12:30:27Z — PR #692 fresh-engine oracle fix

- Fixed the #676 oracle test in merged PR #692 by using a fresh engine for teacher-forced logits after autoregressive decode.
- Root cause: reused-engine teacher forcing restored attention KV but not Mamba conv/recurrent state, causing wrong argmax 279; fresh-engine teacher forcing correctly returns oracle token 33803.
- Flagged the underlying hybrid-Mamba prefix-cache-reuse engine bug, now filed as issue #695.

## 2026-08-11T03:25:00Z — Megakernel Phase 0 dispatched

- Dispatched for persistent single-op QMoE decode kernel work: counter-synchronized FC1→FC2 pipeline, remove four scratch DRAM round-trips, preserve fp32 accumulation order.
- Current status is running; continuation gates are oracle margin 0.09375 and >=3% wall-clock improvement.
