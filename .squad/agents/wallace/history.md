# Wallace — History

## 2026-07-16T00:00:02Z — MatMulNBits direct-int4 GEMV review

- 🟢 Approved Roy's direct-int4 VNNI M=1 GEMV after checking nibble order, zero point, scaling, padded tails, CPU gates, and disjoint N partitioning.
- Added scalar-fallback and direct-int4 partial-K serial/parallel coverage in `2d7c974`; CPU EP tests passed 411/411.
- Independent results showed about +9.1% decode throughput at 24 threads and +81.2% at 96 threads.


## 2026-07-16T00:00:00Z — Fused RMSNorm direct-output review

- 🟢 Cleared Roy's `de62f76` after verifying operation order, distinct output buffers, strict contiguous-f32/same-shape guards, and retained fallback behavior.
- Independently reproduced identical tokens, RMSNorm 1.091→0.734 ms (-32.7%), decode gains of 6–16%, and 413 passing CPU EP tests.
