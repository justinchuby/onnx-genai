# Wallace — History

## 2026-07-16T00:00:02Z — MatMulNBits direct-int4 GEMV review

- 🟢 Approved Roy's direct-int4 VNNI M=1 GEMV after checking nibble order, zero point, scaling, padded tails, CPU gates, and disjoint N partitioning.
- Added scalar-fallback and direct-int4 partial-K serial/parallel coverage in `2d7c974`; CPU EP tests passed 411/411.
- Independent results showed about +9.1% decode throughput at 24 threads and +81.2% at 96 threads.


## 2026-07-16T00:00:00Z — Fused RMSNorm direct-output review

- 🟢 Cleared Roy's `de62f76` after verifying operation order, distinct output buffers, strict contiguous-f32/same-shape guards, and retained fallback behavior.
- Independently reproduced identical tokens, RMSNorm 1.091→0.734 ms (-32.7%), decode gains of 6–16%, and 413 passing CPU EP tests.

## 2026-07-16T00:00:00Z — CUDA M2 GQA/PTX repair
- Replaced the host-seeded GQA regression with real packed prefill plus pointer-aliased decode appends and an independent oracle.
- Added global native `sm_90` CUBIN fallback after CUDA 13.3 PTX ISA 9.3 is rejected by driver 580.105.08; merged in `4a34c66` after Sebastian's clear review.

## 2026-07-16T14:20:00Z — SM-general CUDA NVRTC
- Merged `b56c5cb`: CUDA architecture strings now derive from the live selected device capability (SM60–SM120), retaining the unsupported-PTX native-CUBIN fallback. Holden cleared 117 CUDA tests and 6/6 GQA tests.

## 2026-07-16T18:11:48+0000 — CUDA sub-4-bit and FMA-drift reviews

- 🟢 Cleared Roy's MXFP4/IQ4_NL M=1 CUDA GEMV on H200, including exact decoded-weight checks and 124 CUDA tests.
- 🟢 Cleared Sapper's RMS reduction anti-FMA fix; parity reaches token 11, while token-12 MatMulNBits reduction order remains a follow-up.
