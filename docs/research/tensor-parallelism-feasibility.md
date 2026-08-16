# Tensor Parallelism for native CUDA decode — feasibility (DESIGN-ONLY)

**Author:** Roper (feasibility scoping; read-only codebase analysis + web research; NO code/kernels/build)
**Date:** 2026-08-14
**Model:** Muse-Glimmer-30B, `cuda/int4` Olive package (52 layers, hidden 6656,
intermediate 19968, heads 32, kv_heads 2, head_dim 128, vocab 202048,
`tie_word_embeddings=false`). Source config: `docs/research/lowbit-quant-feasibility.md:13`.
**HW:** H200 SXM (HBM3e ≈ 4.8 TB/s, NVLink4/NVSwitch). **Regime:** M=1 captured decode, `ONNX_GENAI_CUDA_GRAPH=1`.
**Baseline:** ~47 tok/s single-GPU (native CUDA, capture 1 seg / 0 seams).

> **Headline (read first).** TP shards weights across N GPUs so aggregate HBM
> bandwidth scales ~N× — it **only helps if decode is bandwidth-bound.** Our decode
> is **NOT**: at 47 tok/s it reads 15.3 GB/token at only **~724 GB/s = ~15% of the
> H200's 4.8 TB/s roofline**, and a direct **byte-fold probe cutting weight DRAM to
> ¼ raised decode only +2.8%** (`docs/research/lowbit-quant-feasibility.md:7,215`).
> The binding constraint is the **serial ~2568-node launch/latency chain (~8.2
> µs/node ≈ 21 ms/token)** — exactly the axis TP does **not** shard. Worse, TP
> **adds 104 latency-bound all-reduces/token** (52 layers × 2) onto that critical
> path. **Verdict for tok/s on the H200/datacenter tier: 🟥 NO-GO as the next lever,
> conditioned on Sebastian's fresh achieved-DRAM-% — which the existing byte-fold
> probe already implies is ~15%, far below the ~55% break-even.** TP becomes a GO
> for tok/s **only after** single-GPU kernel efficiency is first pushed into the
> bandwidth-bound regime (the node-collapse / decode-megakernel lever that
> `docs/research/dense-decode-megakernel-feasibility.md` already identifies as the
> real lever). **SEPARATELY, TP is a legitimate GO for _fit/capacity_** (weights +
> KV split across GPUs → run models/contexts that don't fit one H200) — an axis
> independent of the tok/s roofline, mirroring the lowbit "fit-ability" finding.

---

## 1. The precondition, stated precisely

TP shards each weight matrix across N GPUs. The per-GPU GEMV reads 1/N of the
weights, so **aggregate weight-read bandwidth scales ~N×**. This accelerates decode
**iff the weight-read (bandwidth) is the binding path.** Formally, let a token's
wall time split into two largely-overlapping components (per
`lowbit-quant-feasibility.md:154`):

- `T_bw` = weight/KV DRAM read time (shardable by TP)
- `T_lat` = serial node launch/latency chain + fixed per-op latency (NOT shardable;
  TP keeps the same layer/op count per GPU)

TP's benefit ceiling is governed by the shardable fraction `T_bw / (T_bw + T_lat)`.
**Measured on this box:** the byte-fold probe (−75% bytes → +2.8%) means
`T_bw`-dominated work is **≈3–4% of the token**; `T_lat` is ≈96%. So even N=∞ TP
(weight read → 0) removes ≤~3–4% — **before** paying the all-reduce tax. This is the
same structural result as the lowbit NO-GO, reached from the bandwidth side.

**Break-even framing (matches the task's gate):** TP pays off **iff single-GPU
decode is at/near bandwidth-bound (>~55% of peak DRAM).** We are at ~15%. Until the
single-GPU kernel efficiency is fixed first, TP is net-negative for tok/s.

---

## 2. Sharding scheme for this architecture

Standard Megatron-style 1-D tensor parallelism. Residual stream (hidden=6656) stays
**replicated**; each of the two sub-blocks reduces once.

**Attention (per layer):**
- **QKV projection — column-parallel.** Split output heads across GPUs. Q has 32
  heads → N=2:16, N=4:8, N=8:4 per GPU (clean). Each GPU computes its heads' Q,K,V
  locally; no comm before attention.
- **Attention compute** runs locally per GPU on its head shard (GQA decode kernel,
  `gqa_decode.rs`), reading only its shard of the KV cache.
- **O projection — row-parallel.** Each GPU produces a partial O over hidden=6656;
  **all-reduce #1** sums partials → full attention output on every rank.

**SwiGLU MLP (per layer):**
- **gate/up — column-parallel.** intermediate=19968 → N=2:9984, N=4:4992, N=8:2496
  (all clean, incl. N=8). Each GPU computes its slice of gate & up, applies
  SiLU·Mul locally (elementwise, no cross-GPU dependency).
- **down — row-parallel.** Each GPU produces a partial over hidden=6656;
  **all-reduce #2** sums partials → full MLP output.

**⇒ exactly 2 all-reduces/layer × 52 = 104 all-reduces/token.** Both land at the
**end** of each sub-block (after O-proj, after down-proj), on the critical path.

**Embedding / lm_head:** vocab=202048, `tie_word_embeddings=false` ⇒ a separate
202048-wide lm_head (a large GEMV). Best sharded **vocab-parallel** (column-parallel
lm_head → all-gather logits, or keep the argmax device-side and all-reduce the
partial max). Input embedding: vocab-parallel gather + all-reduce, or replicate
(embedding table is a lookup, cheap to replicate). This is one extra collective at
the head, not per-layer.

**GQA head-group divisibility — the sharp edge (kv_heads = 2):**

| N | Q heads/GPU (32) | KV heads/GPU (2) | Clean? |
|---|---|---|---|
| 2 | 16 | 1 | ✅ perfect — KV splits 1/GPU |
| 4 | 8 | 0.5 | ❌ **2 KV heads cannot split 4 ways** |
| 8 | 4 | 0.25 | ❌ cannot split 8 ways |

With only **2 KV heads**, clean per-head KV sharding exists **only at N=2.** For
N=4/8 the KV heads must be **replicated** across GPU sub-groups (Megatron does this
when `num_kv_heads < TP`): e.g. N=4 → two groups of 2 GPUs each replicate one KV
head; each GPU still owns 8 Q heads but the KV cache is **duplicated within the
group**. Consequence: the **KV-memory saving from TP is fully realized only at N=2**;
at N≥4 KV is duplicated (Q-proj and MLP still shard cleanly, so compute/weight TP
still works — only the KV *capacity* win degrades). head_dim=128 is never split.

---

## 3. Batch-1 all-reduce cost model

**Message size at M=1:** one all-reduce carries the residual-stream vector =
hidden × dtype = 6656 × 2 B (bf16) ≈ **13.3 KB**. This is a **tiny, latency-bound**
message — far below the bandwidth regime, so NVLink *bandwidth* (~900 GB/s) is
irrelevant; the cost is the collective's **latency floor** (ring/tree steps + kernel
overhead).

**Per-all-reduce latency (NVLink4/NVSwitch, intra-node, small message):**
- Optimistic (in-graph replay, NVSwitch, LL/LL128 protocol): **~5 µs**
- Conservative (eager, ring, protocol overhead): **~15 µs**
- Ring all-reduce latency grows ~2(N−1) steps, so N=4/8 sit toward the high end.

**Per-token comm overhead = 104 × latency:**

| N | per-AR (est.) | 104 all-reduces | as % of 21.3 ms token |
|---|---|---|---|
| 2 | ~5–8 µs | **0.52–0.83 ms** | +2.4–3.9% |
| 4 | ~8–12 µs | **0.83–1.25 ms** | +3.9–5.9% |
| 8 | ~12–18 µs | **1.25–1.87 ms** | +5.9–8.8% |

*(These are the numbers Phase-0 microbench must confirm on this exact box — §6.)*

### Expected tok/s vs N — BOTH regimes

**Regime A — current low-DRAM-util (~15%, MEASURED, latency-bound).**
Binding path = ~21.3 ms serial node chain, **unchanged by weight sharding** (each
GPU runs the same 52-layer, ~2568-node chain; narrower GEMVs are latency-bound so
they don't shrink in wall time). TP only **adds** the all-reduce tax:

| N | token time | tok/s | Δ vs 47 |
|---|---|---|---|
| 1 (today) | 21.3 ms | ~47 | — |
| 2 | 21.3 + ~0.7 | ~21.9 ms | **~45.6 (−3%)** |
| 4 | 21.3 + ~1.0 + KV-replication | ~22.4 ms | **~44.6 (−5%)** |
| 8 | 21.3 + ~1.6 | ~22.9 ms | **~43.7 (−7%)** |

**⇒ net-negative for tok/s at every N.** TP shards the wrong axis and taxes the
critical path.

**Regime B — bandwidth-bound (HYPOTHETICAL: after single-GPU efficiency fixed to
>~55% peak, e.g. via node-collapse/megakernel).** Now binding path ≈ weight read =
15.3 GB / 4.8 TB/s ≈ 3.19 ms → ~313 tok/s single-GPU ceiling. TP shards this N×:

| N | weight read /N | + comm | approx tok/s | vs 313 |
|---|---|---|---|---|
| 1 | 3.19 ms | — | ~313 | 1.0× |
| 2 | 1.60 ms | +0.7 | ~2.3 ms → **~430–530** | ~1.5–1.7× |
| 4 | 0.80 ms | +1.0 | **~700–900** | ~2.5–3.0× |
| 8 | 0.40 ms | +1.6 | **~1200–1400** | ~4–4.5× |

**⇒ near-linear minus comm** — the classic TP win. **This regime does not exist
today** and is the entire content of the precondition.

---

## 4. Integration difficulty in THIS codebase (honest assessment)

**4.1 Comm layer — abstraction exists, backend + wiring do NOT.**
`crates/onnx-runtime-comm/` ships a complete `Communicator` trait with every
collective (`all_reduce`, `all_gather`, `reduce_scatter`, `all_to_all`, p2p,
barrier — `communicator.rs:52-171`), an `InProcessCommunicator` reference backend +
deterministic reduction + collective ordering + TLA+-checked ownership registry
(`lib.rs:1-49`). **But:** (a) there is **no `NcclCommunicator`** — it's Phase 2 in
`docs/DISTRIBUTED_RUNTIME.md:1697-1708`, unbuilt; (b) the crate has **zero inbound
workspace edges by design** (`lib.rs:19-28`) — it is validated in isolation and is
**completely unwired** from the session/EP/decode path. So the *contract and
correctness scaffolding* are a real head start; the *NCCL backend + all sharded
execution wiring* is greenfield. Effort: **L → multi-week.**

**4.2 Sharded weight loading.** Weights load from the Olive `cuda/int4` package
(`decoder/model.onnx`, 417 `MatMulNBits` nodes, `bits=4 block_size=32`, asymmetric
zero-points, bf16 scales — `lowbit-quant-feasibility.md:13`). TP needs each
MatMulNBits split column- or row-parallel **at the packed-int4 layer**, keeping
block_size=32 alignment and slicing scales/zero-points on the matching axis. Doable
either offline (repackage N shards) or at load-time slicing, but the packed layout
makes it fiddly. Effort: **M–L.**

**4.3 Inserting all-reduce into the executor/graph.** Collective nodes must be
inserted after O-proj and down-proj. The executor already has a capture/seam model
(`onnx-runtime-session/src/executor/capture.rs`) and whole-step capture gating
(`onnx-genai-engine/src/native_decode/cuda.rs:30-116`). A collective becomes a new
node kind dispatched through the `Communicator`. Effort: **M.**

**4.4 CRITICAL — CUDA-graph capture × NCCL collectives.** ✅ **Viable, but with a
process-model cost.** NCCL supports capturing collectives into CUDA graphs since
**NCCL 2.9 / CUDA 11.3** (https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/cuda-graphs.html).
Adjacent local evidence: `dense-decode-megakernel-feasibility.md` empirically
**CLEARED** that even a *cooperative* launch records into this H200/driver's active
decode capture and instantiates — so exotic launches capture fine here. **The
catch:** NCCL-in-graph requires **identical capture/replay across all ranks** and
NVIDIA recommends **one GPU per process** (multi-GPU-per-process risks deadlock).
That means TP decode must move from today's **single-process** decode loop to a
**multi-process, one-rank-per-GPU** launch with synchronized per-rank graph
capture/replay — a significant structural change to the decode driver, not just a
kernel insertion. This is the single hardest integration point.

**4.5 KV-cache sharding.** KV shards naturally per-head with TP (each GPU keeps its
KV heads, feeding its local GQA decode). Clean at **N=2** (1 KV head/GPU). At **N≥4
KV must be replicated** (only 2 KV heads) — standard for GQA, but the KV *capacity*
saving then applies only at N=2. Integrates with the existing per-head KV layout
(`gqa_decode.rs`, `native_decode/paged_gqa.rs`, `onnx-runtime-cuda-memory` VMM KV).

---

## 5. Portability (Rule 11) — datacenter-only, must be optional

TP requires multi-GPU + NVLink → **datacenter-only**, exactly the device-conditioned
posture of the lowbit datacenter NO-GO (`RULES.md:107-117`,
`lowbit-quant-feasibility.md:196-201`). Mandatory stance:

- **Default N=1:** no communicator, no all-reduce nodes, **byte-identical** to
  today's single-GPU decode. TP must never regress the single-GPU or CPU-EP path.
- **Opt-in only:** TP activates solely when explicitly configured **and** ≥2
  NVLinked GPUs are detected at runtime (query device properties, don't bake in
  constants — Rule 11). On a single GPU, non-NVLink, or CPU EP → **graceful
  single-GPU fallback**, never crash, never silently OOM.
- **Tier-scoped claims:** any tok/s number is H200/NVSwitch-scoped; do not
  generalize. On PCIe-only multi-GPU the all-reduce latency is far worse and TP is
  even less attractive for tok/s.

---

## 6. Bonus value beyond tok/s — the fit/capacity axis (independent GO)

TP splits **weights (15.3 GB → 7.65 GB/GPU at N=2)** and **KV** across GPUs. This is
a **capacity** win independent of the tok/s roofline — directly parallel to the
lowbit "fit-ability" axis (`lowbit-quant-feasibility.md:176-179`):

- **Run bigger models:** a 60–70B int4 (~30–35 GB) that won't fit one H200's 141 GB…
  actually fits — but a model >141 GB (e.g. a future 200B+ int4, or fp16 variants)
  needs weight sharding to load at all. TP is the mechanism.
- **Longer context / bigger KV:** at N=2, KV splits per-head, doubling the
  context/batch that fits in aggregate VRAM.
- This value **does not care that decode is latency-bound** — it's about *runs vs
  doesn't run*, and is plausibly the **stronger** justification for building TP.

**⇒ If TP is built, build it for fit/capacity, and gate the tok/s justification on
the Regime-B precondition.**

---

## 7. Effort & phasing

- **Phase 0 — cost-model validation microbench (size S; design below, DO NOT run).**
  A 2-GPU all-reduce microbench that measures the **real 13 KB bf16 small-message
  NVLink all-reduce latency on this exact H200 box**, both eager and **captured into
  a CUDA graph and replayed** (to confirm §3's 5–15 µs band and §4.4's capturability
  end-to-end). This single cheap experiment de-risks the entire cost model and the
  NCCL-in-graph question before any L investment.
  - *Design:* init NCCL on 2 ranks (1 proc/GPU); allocate a 6656-elem bf16 device
    buffer; loop `ncclAllReduce(sum)` on an NVLink stream; measure median latency
    over ~10k iters. Then `cudaStreamBeginCapture` → one `ncclAllReduce` → end/
    instantiate; time graph **replay** latency. Report both, and the 104× projection
    (0.5–1.6 ms/token). Also record ring vs tree/LL protocol pick. ~S effort.
- **Phase 1 — N=2 TP, single-node (size L / multi-week).** `NcclCommunicator`
  (DISTRIBUTED_RUNTIME §Phase 2), sharded int4 weight loader, collective insertion
  after O-proj/down-proj, **multi-process one-rank-per-GPU** decode driver with
  synchronized capture/replay, KV per-head split, oracle-grade all-reduce
  determinism. **Only worth starting for the fit axis, or after Regime-B holds.**
- **Phase 2 — N=4/8 (multi-week on top).** KV-head replication, vocab-parallel
  embedding/lm_head, ring-latency mitigation.

---

## 8. Recommendation (with precondition)

1. **tok/s, H200/datacenter tier — 🟥 NO-GO as the next lever, GATED on Sebastian's
   achieved-DRAM-%.** The existing byte-fold probe (−75% bytes → +2.8%) already
   places us at ~15% of roofline, far below the ~55% break-even. In this regime TP
   is **net-negative** (Regime A: −3% to −7%) because it shards the weight-read
   axis, which is not binding, and adds 104 latency-bound all-reduces to the serial
   critical path. **Precondition to flip to GO:** single-GPU decode must first be
   made bandwidth-bound (>~55% peak) — i.e. the **node-collapse / decode-megakernel**
   lever (`dense-decode-megakernel-feasibility.md`) must land first. Only then does
   TP deliver the Regime-B near-linear-minus-comm scaling (N=2 ~1.5–1.7×, N=4
   ~2.5–3×, N=8 ~4–4.5×).
2. **fit/capacity — 🟢 GO justification, independent of tok/s.** If the goal is
   running models/contexts that don't fit one H200, TP is the right tool and should
   be scoped for that, not for speed.
3. **Cheapest next step regardless:** run the **Phase-0 microbench** to nail the real
   comm tax and confirm NCCL-in-graph on this box — S effort, high information.
4. **Portability:** keep TP optional, NVLink-gated, single-GPU fallback byte-
   identical to today (Rule 11).

## Sources
- Hazy/roofline & measured probes (internal): `docs/research/lowbit-quant-feasibility.md`
  (byte-fold −75%→+2.8%, 15% roofline, Muse config), `docs/research/dense-decode-megakernel-feasibility.md`
  (2568-node ~8.2 µs/node latency chain; cooperative-launch-in-capture CLEARED).
- Comm design: `docs/DISTRIBUTED_RUNTIME.md` (Communicator, NCCL Phase 2, NCCL-in-graph plan).
- `crates/onnx-runtime-comm/` (trait + in-process backend, no NCCL, zero inbound edges).
- NCCL + CUDA Graphs: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/cuda-graphs.html (NCCL ≥2.9, CUDA ≥11.3; identical capture/replay per rank; 1 GPU/process recommended).
- Rule 11 portability: `RULES.md:107-117`.
