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

## 2026-07-16T19:05:18+0000 — CUDA SiLU and acc4 drift review

- 🟢 Cleared Sapper's `5c7dcc9`: CUDA now matches CPU's branch-stable SiLU operation order and explicitly rounds acc4 f32 scale/accumulation boundaries without serializing warp reduction.
- H200 validation passed all 128 CUDA EP tests and parity through token 15. The token-16 `1.9073486e-5` reduction-order drift is accepted because exact emulation costs 8.4%.

## 2026-07-16T19:27:57+0000 — CUDA IQ super-block GEMV wave

- 🟢 Cleared Roy's CUDA IQ super-block GEMV on H200: IQ4_XS, IQ2_XXS, IQ3_XXS, IQ2_XS, IQ2_S, and IQ3_S are bit-exact against CPU; IQ1_S/IQ1_M and M>1 fall back correctly. Full CUDA validation passed 128/128 without SM90 hardcoding.

## 2026-07-16T19:27:57+0000 — CUDA IQ1 GEMV review

- 🟢 Cleared Roy's merged `06c4c06` on H200: IQ1_S/IQ1_M M=1 CUDA decoding is bit-exact versus CPU, including both known traces; shared `IQ1S_GRID` hash is `0x6703ed863501ae2e`.
- Full CUDA validation passed 129 tests across 15 groups and the CPU gate passed 15 (one ignored); M>1/unknown fallback and SM-general NVRTC behavior remain correct.


## 2026-07-16T23:06:37+0000 — Native CUDA serving safety re-review

- 🔴 Rejected Roy's CUDA-only `559c46f` because the real 144-BQMM model failed mid-serving without CPU fallback; Roy is locked out. 🟢 Cleared Deckard's `fa30410`: real unsupported CUDA models now fail at startup with heterogeneous-placement guidance, while CPU sub-4 generation and CUDA-positive smoke coverage remain valid.

## 2026-07-16T23:30:00+0000 — CUDA MatMul stale-test review

- 🟢 Cleared Roy's `3d19b72`: the test asserts the real Int64 unsupported error and cannot pass if that dtype becomes accepted.
- Exact target coverage and the full CUDA suite passed 129/129 with cuDNN available.
