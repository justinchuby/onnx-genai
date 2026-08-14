# Roper: Tensor Parallelism for native CUDA decode — go/no-go (with precondition)

Date: 2026-08-14
Author: Roper (feasibility scoping; design-only, no code/kernels/build)
Requested by: Justin (@justinchuby)
Full design + cost model: `docs/research/tensor-parallelism-feasibility.md`

## Verdict — GATED on Sebastian's achieved-DRAM-%

**tok/s, H200/datacenter tier: 🟥 NO-GO as the next lever.** TP scales aggregate HBM
bandwidth ~N×, but only helps if decode is **bandwidth-bound**. Muse-Glimmer-30B
decode is **not**: 47 tok/s reads 15.3 GB/tok at **~724 GB/s = ~15% of the 4.8 TB/s
roofline**, and the byte-fold probe (−75% weight bytes → **+2.8%**) proves the
binding constraint is the **serial ~2568-node launch/latency chain (~21 ms/tok)**,
not bandwidth. TP shards the wrong axis and **adds 104 all-reduces/token** to the
critical path → **net-negative (−3% to −7%)**.

**Precondition to flip to GO (state explicitly):** *TP pays off IFF single-GPU
decode GEMV is at/near bandwidth-bound (>~55% peak). Until then, fix single-GPU
kernel efficiency (node-collapse / decode-megakernel) first.* Sebastian's in-flight
achieved-DRAM-% measurement is the gate; the existing byte-fold probe already implies
~15%, so the expected outcome is NO-GO-for-now unless that number is surprisingly
high.

**Separately: 🟢 GO for fit/capacity.** TP splits weights (15.3 GB → 7.65 GB/GPU @
N=2) + KV across GPUs → run models/contexts that don't fit one H200. This axis is
independent of the tok/s roofline and may be the stronger reason to build TP.

## Expected tok/s vs N
- **Regime A (today, ~15% util):** N=2 ~45.6 (−3%), N=4 ~44.6 (−5%), N=8 ~43.7 (−7%). Net-negative.
- **Regime B (hypothetical, >55% util after megakernel):** N=2 ~1.5–1.7×, N=4 ~2.5–3×, N=8 ~4–4.5× (near-linear minus comm). Does not exist today.

## Sharding scheme (Megatron 1-D)
Col-parallel QKV + row-parallel O (all-reduce #1); col-parallel gate/up + row-parallel
down (all-reduce #2). 2 all-reduces/layer × 52 = 104/token, ~13 KB each (hidden 6656 ×
bf16). **Divisibility:** Q heads 32 split cleanly to N=8; MLP 19968 splits cleanly to
N=8; **but kv_heads=2 splits cleanly only at N=2 — N≥4 requires KV replication.**

## Integration difficulty
- `onnx-runtime-comm` has the full `Communicator` trait + in-process reference +
  TLA+-checked ownership, **but no NCCL backend and zero inbound edges** (unwired). L→multi-week to build/wire.
- Sharded int4 weight loading (packed layout + block_size=32 scales/zp): M–L.
- **NCCL × CUDA-graph capture: viable** (NCCL ≥2.9/CUDA ≥11.3; cooperative-launch-in-
  capture already CLEARED on this box) **but forces a multi-process, one-rank-per-GPU
  decode driver with synchronized capture/replay** — the hardest structural change.
- KV shards per-head (clean @ N=2; replicated @ N≥4).

## Effort & phasing
- **Phase 0 (S):** 2-GPU 13 KB all-reduce microbench — measure real small-message
  NVLink latency, eager + in-graph replay; confirm the 0.5–1.6 ms/token comm tax and
  NCCL-in-graph on this H200. Cheapest, highest-info de-risk. (Designed, not run.)
- **Phase 1 (L/multi-week):** N=2 TP, weights+KV sharded, NcclCommunicator, collective
  insertion, multi-process capture. Start only for the fit axis or after Regime-B.
- **Phase 2 (multi-week):** N=4/8 with KV replication + vocab-parallel head.

## Portability stance (Rule 11)
Datacenter-only; **optional, NVLink-gated**; default N=1 byte-identical to today;
graceful single-GPU fallback; never regress single-GPU/CPU-EP; tier-scoped claims.
Same device-conditioned discipline as the lowbit datacenter NO-GO.

## Recommendation
Do **not** start TP for tok/s now — it's net-negative until decode is made
bandwidth-bound (that's the megakernel/node-collapse lever, already GO in
`dense-decode-megakernel-feasibility.md`). **If** the motivation is fit/capacity,
TP is justified — scope it there. Regardless, run the **Phase-0 microbench** first;
it's S-sized and settles the cost model + NCCL-in-graph question empirically.
