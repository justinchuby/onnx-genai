# Decision — Dense-decode megakernel P1.5 (fused int4 GEMV + grid.sync-capture gate)

**Author:** Sebastian (Performance/Systems) · **Date:** 2026-08-13 · **Branch:** `squad/dense-megakernel-p15`

## Context
Follow-up to P1.5: build the *real* fused int4 GEMV one-layer megakernel and answer
the P2 architecture question the coordinator flagged — **does cooperative-groups
`grid.sync` survive CUDA-graph capture?** Two throwaway `#[ignore]` GPU probes added
to `crates/onnx-runtime-ep-cuda/tests/megakernel_headroom_gpu.rs` (not wired into any
dispatch path). Measured on this H200 (`CUDA_VISIBLE_DEVICES=0`).

## Measurements
- **Single-CTA fused int4 MLP** (gate/up/SiLU·Mul/down, 19968-wide intermediate
  resident in 104 KiB shared, zero activation DRAM round-trips) vs per-op baseline
  (4 launches, `grid=N`, full-device parallel weight reads):
  **0.664 ms → 615.3 ms/layer-MLP = 926× SLOWER. Byte-exact (0 ulp).**
  → Residency-only fusion is a dead end on the weight-heavy GEMVs; one SM ≈ 1/132 of
    device weight-read bandwidth. **The megakernel MUST be multi-CTA.**
  (Absolute ms is my reference f32 GEMV, not the production f16 split-K dp4a path —
   the *ratio* + byte-exactness are the findings, not the absolute time.)
- **grid.sync / cooperative-launch under CUDA-graph capture** (the P2 gate):
  (A) cooperative launch outside capture = OK; (B) cooperative launch **during**
  thread-local capture = **launch Ok, capture status ACTIVE, graph instantiated**.
  → **grid.sync IS capturable on this H200/driver.** The most-feared P2 blocker does
    NOT fire here.

## Decision / Recommendation
**P2 = GO (staged prototype), architecture now pinned:** a **persistent multi-CTA
cooperative megakernel** — grid sized to occupancy (`cuOccupancyMaxActiveBlocksPerMultiprocessor`
× SM count) for co-residency, `grid.sync` barriers between resident sub-GEMVs,
activations + norm/RoPE/SiLU state in shared/registers, weights + KV streamed from
DRAM with full-device parallelism. Single-CTA residency shortcut is ruled out.

**Gate answer for the coordinator: YES, grid.sync survives capture** (driver/CTK
dependent — keep a runtime capability check + graph-break fallback, since older
drivers returned `CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED`).

## Risks to budget for P2
1. Occupancy/co-residency grid sizing at H=6656 (register/smem pressure).
2. One fixed cooperative grid hosting GEMVs of different N (19968 vs 6656).
3. Numerics: any fused RMSNorm / GQA-softmax / GEMV-K reduction reorder → **Chew**
   gate + f64 oracle. (Pure structural residency proven byte-exact here.)
4. Driver-version portability of captured cooperative launch → graph-break fallback.
5. Capture-safety: no internal alloc/free/dynamic-parallelism (#854/#867 rules).

## Ownership
- Kernel side (persistent cooperative megakernel, `kernels/*.rs` + bench): **Sebastian**.
- Graph/decode-loop side (emitting one fused op instead of ~49 nodes/layer,
  `optimizer.rs`): coordinate with **Batty**.
- Numerics gate: **Chew** (f64 oracle) for any reduction reorder in the full kernel.

Docs: `docs/research/dense-decode-megakernel-feasibility.md` §6.
