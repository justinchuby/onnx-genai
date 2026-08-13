# Dense-decode megakernel feasibility (Muse-Glimmer-30B, native CUDA, M=1)

**Author:** Sebastian (Performance/Systems). **Date:** 2026-08-13. **Branch:** `squad/dense-decode-megakernel`.
**Status:** Phase A gate **PASSED** → Phase B micro-benchmark run → **GO to prototype P2** (staged; see §5).

> **Headline.** Native decode of Muse-Glimmer-30B is confirmed **latency-bound on a
> ~2568-node serial launch chain** (~21.4 ms/token, 46.7 tok/s on H200 at
> `ONNX_GENAI_CUDA_GRAPH=1`), **not** weight-bandwidth-bound (prior byte-fold probe:
> −75% weight bytes → +2.8%). Phase A shows the megakernel-recoverable overhead is
> **large** — pure per-launch floor alone is ~20% of the token, and the essential
> weight+KV DRAM floor is only ~3.2 ms (a **~6.7× theoretical ceiling**, ~313 tok/s).
> Phase B directly measured the fusion mechanism: collapsing the elementwise/norm
> "glue" chain (70% of the node count, ~44% of eager decode) into one resident
> kernel **recovers 85.6% of that chain's GPU time**, byte-exact. **Verdict: GO** —
> a dense one-layer megakernel is a real lever worth prototyping (P2), projected
> **~62 tok/s conservative (+33%) to ~100+ tok/s (2×+)**. This is a *different* op
> chain than the #769 P0 QMoE-MoE megakernel (Amdahl-capped ~13%, NO-SHIP); that
> result does not bind the dense path.

---

## 1. Baseline & op mix (measured, this H200, `CUDA_VISIBLE_DEVICES=0`)

`profile_native --model <muse-glimmer> --pipeline --ep cuda --backend native --steady --warmups 1 --runs 3 --tokens 128`

| config | ms/token | tok/s | note |
|---|---:|---:|---|
| captured (`ONNX_GENAI_CUDA_GRAPH=1`) | **21.40** | **46.72** | baseline; 1 capture segment / 0 seams |
| eager (`ONNX_GENAI_CUDA_GRAPH=0`) | 27.65 | 36.17 | capture already recovered ~6.1 ms/token of **CPU** launch overhead |

Eager per-op decode mix (`ONNX_GENAI_PROFILE_OPS=1`, one forward, 27.64 ms total):

| op | total ms | % eager | calls | per-call µs |
|---|---:|---:|---:|---:|
| MatMulNBits | 7.80 | 28.2 | 417 | 18.7 |
| GroupQueryAttention | 7.26 | 26.3 | 52 | 139.5 |
| Mul | 6.12 | 22.1 | 210 | 29.1 |
| Add | 2.44 | 8.8 | 311 | 7.9 |
| SimplifiedLayerNormalization | 1.33 | 4.8 | 312 | 4.3 |
| Sigmoid | 1.32 | 4.8 | 104 | 12.7 |
| Reshape | 0.95 | 3.4 | 208 | 4.5 |
| (tail: ReduceSum/Skip/Gather/Cast/…) | <0.2 | <1 | ~11 | — |

Executor plan ≈ **1626 op-nodes/token**, which the capture pass expands to **2568
device-graph nodes/token** (memsets, broadcast expansions, view materializations).
The **elementwise/norm "glue"** — `Mul`+`Add`+`Norm`+`Sigmoid`+`Reshape` = **1145
op-nodes (70% of the count), ~44% of eager time**. Each is a tiny launch that reads
and rewrites the 6656-element bf16 hidden vector (~13 KiB) through global memory —
exactly the inter-op round-trip a megakernel keeps in registers/shared. The 469
"heavy" nodes (417 GEMV + 52 GQA, 54% of eager time) carry the real weight/KV DRAM
+ MAC work.

---

## 2. Phase A — headroom gate (the three megakernel-recoverable costs)

Decomposing the **21.4 ms captured floor** (CPU launch already ≈0 under graph replay,
so this is the GPU-side serial floor). Real per-node cost = 21.4 ms / 2568 = **8.33 µs/node**.

**(a) Per-node launch/schedule floor.** Measured directly (Phase B micro-bench, trivial
1-thread kernel, 2568 sequential launches on one stream): **~1.5–2.0 µs/launch**. So the
pure launch floor is 2568 × ~1.7 µs ≈ **4.4 ms ≈ 20% of the token** — recoverable by
node collapse alone, and this is only the *floor* (excludes the kernel's own work).

**(b) Inter-op activation DRAM round-trips.** The hidden vector bounces to global memory
between ~1145 glue ops (read+write ~26 KiB each ≈ 30 MB/token — trivial *bandwidth*, but
each round-trip is **serialized**: op N+1 cannot start its global load until op N's store
retires). The cost is the *latency* of that read-after-write chain, folded into the 8.33 µs.

**(c) Occupancy fill/drain.** Every M=1 decode kernel launches ~1 block on a 132-SM H200
(<1% of the machine) and finishes before steady state — the global-load latency
(Long-Scoreboard) is never hidden. A resident, pipelined megakernel overlaps the next
layer's weight loads with the current layer's compute.

**Essential floor a megakernel must still pay.** Weight + KV DRAM = 15.325 GB/token; at
H200 HBM3 4.8 TB/s that is **3.19 ms/token = ~313 tok/s** if perfectly bandwidth-bound.
The byte-fold probe (#885) showed only ~3–4% of that (~0.7 ms) is *currently exposed* —
i.e. today the weights are latency-hidden behind the launch chain, not throughput-bound.

**Headroom = 21.4 ms − 3.2 ms essential ≈ 18.2 ms ≈ 85% of the token is recoverable
overhead** (launch + round-trip latency + fill/drain). **This is ≫ the 25% gate.**

> **GATE: PASS.** Recoverable overhead ~85% of the token; theoretical ceiling ~6.7×
> (313 tok/s). Proceed to Phase B.

---

## 3. Phase B — fusion-mechanism micro-benchmark (measured)

A **throwaway** CUDA-event micro-bench (`crates/onnx-runtime-ep-cuda/tests/megakernel_headroom_gpu.rs`,
`#[ignore]`, **not wired into any pipeline** — run with `--ignored --nocapture`) isolates
the recoverable costs on a faithful model of one layer's glue chain: G=22 glue ops per
layer, each reading+rewriting the H=6656 bf16 vector, vs one fused kernel that loads H
once, applies all 22 ops in registers (fp32), stores once. Median of 200 iters, H200:

| measurement | result |
|---|---|
| per-launch GPU floor (trivial kernel × 2568) | **1.5–2.0 µs/launch** |
| realistic glue op (H=6656 bf16 round-trip) | **~2.1 µs/op** |
| **22 unfused glue launches** | **0.044–0.046 ms** |
| **1 fused launch (22 ops in registers)** | **0.006–0.007 ms** |
| **glue-chain time recovered by fusion** | **85.6%** (2 runs: 85.5 / 85.7) |
| numerics: fused (fp32-chained) vs unfused (bf16 per-op) | **max_abs = 0.0** (byte-identical here) |

Two facts fall out: (1) a glue op costs ~2.1 µs ≈ the ~1.7 µs launch floor + ~0.4 µs
memory — glue ops are **launch/latency-bound, not compute-bound**, so collapsing them
recovers essentially all their time (85.6% measured). (2) The fused fp32 chain matched the
per-op bf16 round-trip to 0 ulp on this input (silu with stable inputs) — **but this does
NOT clear the numerics gate for the real megakernel**, whose RMSNorm tree-reduction and
GQA softmax reorder fp32 reductions; those must keep per-thread fp32 accumulation and go
through **Chew** (as in #855/#860).

---

## 4. Whole-model projection (measured-anchored)

- **Conservative (glue-only fusion, GEMV/GQA left as separate launches).** The glue is
  ~30% of the *captured* 21.4 ms (less than the 44% eager share, since capture already
  removed the CPU-launch half) ≈ 6.4 ms. Recover 85.6% → save ~5.5 ms → **15.9 ms/token ≈
  63 tok/s (+35%)**. This is the *floor* of the win and needs only glue fusion.
- **Optimistic (full one-layer persistent megakernel).** Collapse each layer's ~49
  device nodes to ~2–3 launches (a GQA softmax dependency forces one chunk boundary,
  Hazy-style), keep activations resident, and pipeline the next layer's int4 weight loads
  to hide Long-Scoreboard latency. Recovering the launch floor (~4.4 ms) plus most of the
  round-trip/fill-drain latency plausibly reaches **2–3× → ~95–140 tok/s**.
- **Absolute ceiling** (weight+KV bandwidth-bound, perfectly pipelined): **~313 tok/s**.

Projected range: **~63 tok/s conservative → ~100+ tok/s realistic**, ceiling ~313.
All above 47.25; the lever is real.

---

## 5. Go/no-go for P2 (whole-step integration)

**GO — but staged, because P2 is a multi-week capture-safe kernel effort, not a quick win.**

- **Why the #769 P0 no-go does not apply:** P0 was a per-op persistent **QMoE** kernel on
  the 35B-A3B **MoE** path (Amdahl-capped ~13% of decode, regressed). The dense
  Muse-Glimmer path is a completely different op chain (417 dense int4 GEMV + 52 GQA +
  1145 glue) that P0 never touched, and Phase A/B show its recoverable overhead is ~85%.
- **Recommended next step (P1.5, before full P2):** build the *real* one-layer dense
  megakernel with actual int4 weights — RMSNorm → QKV MatMulNBits → RoPE → GQA → O-proj →
  residual → RMSNorm → gate/up MatMulNBits → SiLU·Mul → down → residual — keeping
  intermediates in registers/shared (chunked producer/consumer at the GQA softmax
  boundary), behind an env flag, measured vs the current per-op layer. This closes the one
  gap in this brief: Phase B measured the **glue** recovery (the 70%-of-nodes component)
  and the launch floor directly, but did **not** yet build the fused int4 GEMV path — that
  per-layer number is what should gate funding the full P2 pipeline integration.
- **P2 risks to budget:** (1) capture-safety — the megakernel must do no alloc/free/sync
  internally (all scratch pre-staged, per the #854/#867 capture rules); (2) numerics —
  fused RMSNorm/softmax reductions reorder fp32 sums → **Chew gate** + f64 oracle
  mandatory; (3) it needs decode-loop/graph-structure changes to emit one fused op instead
  of ~49 nodes/layer → coordinate with **Batty** (graph/optimizer side). Keep byte-exact
  greedy parity target (first-16 ref ids `[24,372,1045,10016,328,2885,262,5091,8811,511,917,4921,768,328,2885,262]`).

---

## Appendix — reproducibility

- Env: `source /home/justinchu/onnx-genai/.cudaenv.sh`; `CUDA_VISIBLE_DEVICES=0 ONNX_GENAI_CUDA_DEVICE=0`.
- Baseline/eager/op-mix: `profile_native` as in §1 (`ONNX_GENAI_CUDA_GRAPH=0/1`, `ONNX_GENAI_PROFILE_OPS=1`).
- Phase B micro-bench (throwaway, `#[ignore]`, never shipped):
  `cargo test --release -p onnx-runtime-ep-cuda --features cuda --test megakernel_headroom_gpu -- --ignored --nocapture`.
  Knobs: `NXRT_MK_HIDDEN` (6656), `NXRT_MK_GLUE_OPS` (22), `NXRT_MK_ITERS` (200), `NXRT_MK_LAYERS` (52).
- Model dir: `.../olive-recipes/meta-models-Muse-Glimmer-30B/cuda/int4/models` (`--pipeline`; never write through the symlink).
- Profilers (ncu/nsys) are blocked in-sandbox (`RmProfilingAdminOnly=1`); all numbers are built-in op timer + CUDA-event + wall-clock, per the profiling skill.
