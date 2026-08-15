# Lower-bit quantization + structured sparsity feasibility for sub-47 tok/s decode

**Author:** Sebastian (Performance/Systems) · **Date:** 2026-08-13 · **Status:** feasibility / decision-support (NO kernel work in this brief)
**Model:** Muse-Glimmer-30B, `cuda/int4` Olive package · **HW:** H200 SXM (HBM3e ≈ 4.8 TB/s) · **Regime:** M=1 captured decode, `ONNX_GENAI_CUDA_GRAPH=1`
**Baseline:** 47.25 tok/s median (native CUDA, capture 1 seg / 0 seams), the current ceiling after the #855/#854/#860/#867/#870 arc.

> **Headline (read this first).** At 47.25 tok/s the decoder reads **15.3 GB of weights/token at only 724 GB/s — 15% of the H200's 4.8 TB/s roofline.** Decode is **NOT weight-bandwidth-bound** — and this is now **confirmed by direct measurement** (§Bandwidth Probe): a kernel-level A/B that cuts the packed-weight DRAM footprint to **¼** raises decode only **47.29 → 48.62 tok/s (+2.8%)**. The binding constraint is the **serial ~2568-node dependency chain (~8.2 µs/node)**. Therefore the naive "halve the bytes → double tok/s" projection is **empirically false**, and even *int2-everywhere* would gain **≈+2%, not +80%**. Combined with a 🔴 accuracy cliff and an **L-sized tooling+kernel dependency we do not have** (only int4 weights exist; our GEMV supports only bits∈{4,8}; sub-4-bit needs re-quant from the fp16 source), **lower-bit quant is a measured NO-GO as the next lever — *on the H200/datacenter tier.*** The evidence-backed next move *there* is **drastic node-count collapse (decode megakernel)**. **⚠️ This verdict is device-dependent (§6):** on bandwidth-starved consumer/edge GPUs the same weight read stops being latency-hidden and lower-bit quant regains real value — as a **speed** lever below the ~0.7 TB/s crossover, and as a **fit** lever below ~12 GB VRAM (a 30B int4 ≈ 15 GB won't even load). Keep lowbit on the roadmap for that tier, gated on re-running the byte-fold probe on a representative consumer GPU.

---

## 1. Baseline byte budget (measured from the real ONNX graph)

Source: `decoder/model.onnx` (`meta-models-Muse-Glimmer-30B/cuda/int4/models`), 417 `MatMulNBits` nodes, **all** `bits=4, block_size=32`, **asymmetric (zero-points present)**, scales `bf16`. Config: 52 layers, hidden 6656, intermediate 19968, heads 32, kv_heads 2, head_dim 128, vocab 202048, `tie_word_embeddings=false` (⇒ separate 202048-wide lm_head).

Per `MatMulNBits(K, N, bits=4, bs=32)` the weight-related bytes read once per decode token are:

| tensor | formula | bytes |
|---|---|---|
| packed weight | `N · ceil(K/bs) · (bs·bits/8)` | `N · (K/32) · 16` |
| scales (bf16) | `N · ceil(K/bs) · 2` | one bf16 per (col, block) |
| zero-points (4-bit) | `N · ceil(K/bs) · 0.5` | one nibble per (col, block) |

Enumerating all 417 GEMVs (exact, summed programmatically from the initializer dims):

| shape (K×N) | count | role | packed wt each |
|---|---:|---|---:|
| 6656 × 4096 | 104 | q_proj, gate_proj-attn | 13.63 MB |
| 6656 × 256 | 104 | k_proj, v_proj | 0.85 MB |
| 6656 × 19968 | 104 | MLP gate, MLP up | 66.45 MB |
| 4096 × 6656 | 52 | o_proj | 13.63 MB |
| 19968 × 6656 | 52 | MLP down | 66.45 MB |
| 6656 × 202048 | 1 | **lm_head** | 672.42 MB |

**Totals per token:**

| component | MB/token | GiB/token |
|---|---:|---:|
| packed int4 weights | 13 254.3 | 12.344 |
| bf16 scales | 1 656.8 | 1.617 |
| int4 zero-points | 414.2 | 0.404 |
| **grand total (weight traffic)** | **15 325.3** | **14.273** |

**Cross-check:** computed 15 325 MB ≈ the on-disk `model.onnx.data` = 15.367 GB (the small delta is the handful of non-quantized tensors — norms, embed table, biases). The byte budget is trustworthy. MLP is **~78%** of weight bytes; attention ~12%; lm_head ~5%; scales+zp metadata **~13.5%** (and metadata does **not** shrink with fewer weight bits — see §2).

### Roofline cross-check (the crux)

```
weight traffic / token      = 15.325 GB
achieved bandwidth @47.25   = 15.325 GB × 47.25 /s = 724 GB/s
H200 HBM3e peak             ≈ 4.8 TB/s
utilization                 = 724 / 4800 = 15.1%
pure-bandwidth roofline     = 4.8 TB/s / 15.325 GB = 313 tok/s
```

We are **6.6× slower than the memory roofline.** If decode were weight-bandwidth-bound we'd already be at ~313 tok/s. We are at 47. **The binding constraint is dispatch/kernel-issue latency** (2568 captured nodes/token × ~8.2 µs/node ≈ 21.0 ms/token ≈ 47.6 tok/s), exactly the #870 finding. This single fact governs every projection below: **reducing weight bytes only speeds up the fraction of the 21 ms that is actually spent moving weight bytes.**

**Estimated weight-bound fraction `W`.** If the large GEMVs ran at 100% BW, weight reads = 15.325/4800 = 3.19 ms = **15%** of 21 ms. At a more realistic M=1 GEMV efficiency of 40–70% of peak, weight reads ≈ 4.5–8 ms = **~21–38%**. Take **W ≈ 0.28** as the central estimate; the remaining ~72% is fixed per-node dispatch/latency across the 2151 non-GEMV nodes (834 Cast, 312 RMSNorm, 311 Add, 210 Mul, 208 Reshape, 104 Sigmoid, 52 GQA, …) plus GEMV launch overhead. **This W must be measured, not assumed — see §5 experiment.**

---

## 2. Candidate formats — byte budget & projected tok/s

**Metadata floor.** bf16 scales (1 657 MB) are per-(col,block) and **independent of weight bits** — they do not shrink. Zero-points scale with bits. So total bytes shrink **sub-proportionally** to the bit-width:

| format | packed wt | scales(bf16) | zp | **total MB** | **× base** |
|---|---:|---:|---:|---:|---:|
| int4 (current) | 13 254 | 1 657 | 414 | **15 325** | 1.000 |
| int3 | 9 941 | 1 657 | 310 | **11 908** | **0.777** |
| int2 | 6 627 | 1 657 | 207 | **8 491** | **0.554** |
| mixed (int4 attn+lm / int2 MLP) | 7 358 | 1 657 | ~300 | **~9 315** | **~0.608** |
| 2:4 sparse on int4 (nonzeros+idx) | ~6 627+idx | 1 657 | 414 | **~9 100** | **~0.594** |
| NF4/AF4 (still 4-bit) | 13 254 | 1 657 | 414 | **15 325** | **1.000** |

Note int2 lands at **0.554×**, not 0.5×, because of the bf16-scale floor — so even the *naive* int2 ceiling is 47.25/0.554 = **85 tok/s**, not the 94 in the ask. (Shrinking scales to fp8/int8 could recover part of the floor but adds its own accuracy risk and a second kernel change.)

**Projected decode tok/s — naive (assume bandwidth-bound) vs pre-probe Amdahl (`W≈0.28` from the 15% roofline).** ⚠️ **Both columns are now superseded by the *measured* §5 result** (`W ≈ 0.035`), which drops even the "realistic" int2 figure from ~54 to ~48 tok/s. They are retained to show the modeling before measurement:
`realistic speedup = 1 / ((1−W) + x·W)`, `x = ×base`.

| format | ×base bytes | naive tok/s (W=1) | **realistic tok/s (W≈0.28)** | accuracy risk | tooling dep |
|---|---:|---:|---:|:--:|:--:|
| int3 | 0.777 | 60.8 | **~50.0** | 🟡 (GPTQ/AWQ ok; K-quant Q3_K ok) | re-quant + new kernel |
| int2 | 0.554 | 85.3 | **~54.1** | 🔴 (cliff without QuIP#/AQLM-class) | re-quant + new kernel |
| mixed int4/int2 | 0.608 | 77.7 | **~53.0** | 🟡/🔴 (MLP int2 is the risk) | re-quant + mixed-bit kernel |
| 2:4 sparse int4 | 0.594 | 79.5 | **~53.4** | 🟡 (needs sparse-aware fine-tune) | sparse GEMV from scratch |
| NF4/AF4 | 1.000 | 47.25 | **47.25** | 🟢 (accuracy *enabler*, not byte saver) | LUT-dequant kernel |

**The gap between the naive and realistic columns is the whole story:** because we sit at 15% of the memory roofline, the biggest theoretically-available byte cut (int2, −45%) buys only **+14% tok/s** even in this Amdahl model — and the **measured probe (§5) collapses that further to ~+1.6%.** NF4/AF4 saves **zero** bytes (still 4-bit) — its only value is improving accuracy-per-bit so that int3/int2 becomes *tolerable*; it is an enabler for the rows above, not a lever by itself.

> All tok/s figures are **projections** from the measured byte budget + an estimated `W`; none are measured on a real sub-4-bit build (we have no such build — §4). No accuracy numbers are quoted because we have not run perplexity; risk flags cite the **method class**, not measured deltas.

---

## 3. Kernel feasibility (our NVRTC `MatMulNBits` GEMV)

Current kernel (`crates/onnx-runtime-ep-cuda/src/kernels/matmul_nbits.rs`): dispatches **only `bits==4` and `bits==8`** (e.g. `mask = bits==8 ? 255 : (1<<bits)-1` but the surrounding load/unpack is hard-coded to nibble/byte layout). Packed weights load as **128-bit `uint4` vectors**; int4 is a clean nibble unpack; block_size 32; fp32 accumulate; an `accuracy_level=4` **dp4a int8-activation** path already exists (`matmul_nbits_gemv_accuracy4_*`, `__dp4a`).

| format | packing at bs=32 | unpack cost | kernel effort |
|---|---|---|---|
| **int3** | 12 B/block, **crosses byte boundaries** (values straddle bytes) | needs `bfe`/funnel-shift extraction; **no clean `uint4` nibble path**; awkward warp-load alignment (12 B not a power of two) | **M–L** |
| **int2** | 8 B/block, **4 values/byte, clean** | simple mask+shift, structurally close to int4; `uint4` load fine; dp4a-friendly | **S–M** |
| **mixed** | per-node bits | per-node dispatch already keyed on `bits`, but needs the 2-/3-bit kernels to exist first + graph must carry mixed-bit nodes | **M** (on top of int2/int3) |
| **2:4 sparse** | nonzeros + 2-bit index | **new sparse GEMV**; NVIDIA Sparse Tensor Cores (`mma.sp`) need M≫1 to pay off — **useless at M=1 decode**; benefit would come only from *reading fewer bytes*, needing a bespoke sparse-index GEMV, not cuSPARSELt (which targets GEMM) | **L** |
| **NF4/AF4** | still 4-bit, + 16-entry LUT | add a shared-mem LUT lookup in dequant; ALU is ~free at M=1 (compute headroom) but **no byte win** | **S** (but pointless for bytes) |

At M=1 there is compute headroom, so the extra dequant ALU (bitfield extract, LUT) is **effectively free until it isn't** — the loads dominate. So the kernel side of int2 is genuinely tractable (S–M). **int3's non-byte-alignment makes it the hardest of the "simple" bit-reductions.** 2:4 sparsity gets **no tensor-core benefit at M=1** and is 🔴 for this workload.

---

## 4. THE REAL BLOCKER — where do sub-4-bit weights come from?

We **only have int4.** The package was produced by (README + `cuda/int4/config.json`):

```
HfModel meta-models/Muse-Glimmer-30B  (bf16, ~60 GB source, public HF)
  → MobiusBuilder(precision=bf16)          [736 s]
  → OnnxKQuantQuantization(bits=4, bs=32)  [718 s]   # Q4_K_M-style K-quant
  → cuda/int4/models                       (int4 only; no int3/int2/mixed variant exists)
```

To get any sub-4-bit weights we must **re-quantize from the bf16 source** with an accuracy-preserving method — you cannot losslessly re-quantize *from the existing int4* (that compounds error). Concretely this requires, in order:

1. **Download the ~60 GB bf16 HF checkpoint** + enough host RAM for source + intermediate ONNX (README warns on this). *We do not have it staged.*
2. **A quantizer that emits sub-4-bit.** Olive's `OnnxKQuantQuantization` exposes a `bits` param and K-quant *does* have a Q3_K/Q2_K family, **but it is unverified that this pass emits valid `MatMulNBits(bits=2|3)` ONNX** — and K-quant alone (round-to-nearest-ish) at 2-bit is a known 🔴 accuracy cliff. Safe sub-4-bit needs **GPTQ / AWQ / QuIP# / HQQ / AQLM-class** calibration, which is **not in this recipe** and may not be wired into Olive for this custom `muse_glimmer` arch at all.
3. **Runtime support:** even if the graph carries `bits=2|3`, **neither ORT's MatMulNBits nor our kernel executes bits∉{4,8}** — new kernels (§3) are mandatory before a single token decodes.

**This dependency chain — source download + a calibrated sub-4-bit recipe for a custom multimodal arch + new op/kernel support — dominates the go/no-go.** It is weeks of cross-team work (tooling + numerics + kernel) *before* the +14% (int2) / +6% (int3) payoff can even be measured, and the payoff is capped low by the dispatch bound (§1).

---

## 5. Bandwidth Probe — measured result (the arbiter)

**Motivation.** Two facts were in tension: the fusion arc (#870/#872/#873) found node-count reduction flat/regressive (⇒ *not* naively launch-dispatch-bound), while §1's roofline shows only 15% HBM utilization (⇒ *not* aggregate-bandwidth-bound). Both negatives can't leave the system unexplained. So I ran the probe directly.

**Method (throwaway, reverted — never shipped).** Added a private env flag `ONNX_GENAI_WEIGHT_FOLD=D` that folds the *weight-read column* of the dominant decode GEMV (`matmul_nbits_gemv_f16_scales_f16_zp_splitk`, which handles **all** int4 decode projections + lm_head on this model — the fused gate/up path does not fire here) so every output column aliases into the first `N/D` weight rows. This shrinks the packed-weight + scale + zero-point **DRAM footprint to 1/D while keeping the loop-trip count, instruction stream, launch grid, and captured node count byte-identical.** Crucially this isolates the *memory-throughput* axis — exactly what lower-bit quant reduces (lower-bit keeps the same K-block count, only fewer bytes/weight; so a K-loop-shortening probe would be *unfaithful*, but address-folding is faithful). Output is numerically garbage; only timing is trusted.

**Measured (H200, `ONNX_GENAI_CUDA_GRAPH=1`, `--pipeline`, staged Muse-Glimmer, capture 1 seg/0 seams, median of 3×128-tok runs):**

| probe | weight DRAM footprint | decode tok/s | Δ vs full |
|---|---:|---:|---:|
| **full (D=1, real)** | 100% | **47.29** | — |
| **half (D=2)** | 50% | **47.98** | **+1.5%** |
| **quarter (D=4)** | 25% | **48.62** | **+2.8%** |

**Interpretation — the "flat" branch fired.** Removing **75% of weight DRAM traffic buys only +2.8% tok/s.** Extrapolating, a hypothetical *zero-cost* weight read (bytes→0) would land ≈48.5–49 tok/s — an Amdahl ceiling implying the **weight-DRAM-bound fraction is only ~3–4%**, far below even §1's 15% roofline estimate (the roofline counts bytes moved; the probe additionally shows those bytes are *latency-hidden*, not throughput-limiting, because folding keeps the same per-load Long-Scoreboard stalls). **Decode is empirically NOT weight-bandwidth-bound.** Mapping onto §2: int2-everywhere (−45% bytes) would gain ≈ 0.45 × 3.5% ≈ **+1.6% (~48 tok/s), not +14% and certainly not +80%.** This is the measured kill-shot for lower-bit quant.

**What the system *is* bound by.** The two negatives now reconcile: it is **neither** bandwidth-bound (byte-fold flat) **nor** sensitive to *marginal* node changes (#872/#873: ±4–8% nodes → flat/regressive). It is bound by the **serial critical-path latency of the ~2568-node dependency chain — ~8.2 µs/node fixed floor × 2568 ≈ 21 ms/token.** Under graph replay the per-node CPU launch cost is ~0, so this floor is GPU-side per-kernel latency (grid launch + block scheduling + minimum memory-latency tail), which neither cheaper bytes nor merging a handful of nodes removes. **Only a *drastic* collapse of the node count — a decode megakernel that keeps activations in registers/shared and does many ops per launch — attacks this floor.**

---

## 6. Machine-class sensitivity — the byte axis is device-dependent

**The NO-GO above is H200-specific, not universal.** The probe measured *this* box (H200, ~4.8 TB/s HBM3). The reason the byte axis is dead here is that on the H200 the weight read is almost entirely **hidden behind** the serial launch-latency chain — it is not the binding constraint. On a **bandwidth-starved** device the *same* 15.3 GB/token weight read stops being hidden and becomes the bottleneck, and lower-bit quant regains real value. So the conclusion must be device-tiered.

**Two-component model (per token).** Treat wall time as two largely-overlapping components:

- `T_latency` — the dispatch/launch-latency chain of the ~2568-node graph. Under graph replay this is dominated by GPU-side per-kernel latency (grid launch + block scheduling + minimum memory-latency tail), ~device-independent in the sense that it does **not** scale with HBM bandwidth. On the H200 we measured this floor at ~21 ms/token (~8.2 µs/node × 2568).
- `T_weightread` = 15.3 GB / `B_device` — the time to stream int4 weights from VRAM each token.

Per-token time ≈ `max(T_latency, T_weightread)` in the fully-overlapped limit (the truth is between `max` and the sum; the probe shows H200 is near the fully-hidden `max`-dominated regime). On the H200:

- `T_weightread` = 15.3 GB / 4.8 TB/s = **3.19 ms** — naively ~15% of the 21.15 ms/token roofline, yet the probe shows only **~3–4%** is *exposed*, i.e. almost all of it overlaps `T_latency`. **`T_latency` dominates → latency-bound → byte cuts are ~free-riding on hidden time → lower-bit useless for speed.**

**The crossover.** As `B_device` falls, `T_weightread` grows (it's inversely proportional to bandwidth) while `T_latency` stays roughly fixed. Once `T_weightread` exceeds the hidden headroom under `T_latency`, weight reads stop being hidden, the byte-fold slope steepens, and lower-bit quant's payoff climbs from the H200's ~3% toward the naive roofline (up to ~1/bits-ratio). Rough crossover: weight read stops being hidden once `15.3 GB / B_device ≳ T_latency` — i.e. below **B_device ≈ 15.3 GB / 21 ms ≈ 0.73 TB/s** the weight read alone exceeds the H200 latency floor and the device tips bandwidth-bound. **⚠️ This is an EXTRAPOLATION from a single-device measurement; the crossover band is a model, not a measurement. Projections below are marked as projections — no fabricated cross-device tok/s.**

**Device-tier table** (bandwidth + VRAM are widely-published spec-sheet *ranges*, not benchmarks I ran; 15.3 GB = the int4 Muse-Glimmer-30B weight footprint from §1):

| tier | example GPUs | mem BW (spec range) | VRAM | fits 15.3 GB int4? | regime (projected) | lowbit value (projected) |
|---|---|---|---|---|---|---|
| Datacenter | H200 / H100 | ~3.3–4.8 TB/s | 80–141 GB | ✅ fits | **latency-bound** (measured on H200) | 🟥 ~useless for speed |
| High-end consumer | RTX 4090 / 5090 | ~1.0–1.8 TB/s | 24–32 GB | ✅ fits | mixed / near crossover | 🟡 modest speed help |
| Mid consumer | RTX 4060 / 4070 | ~270–500 GB/s | 8–12 GB | ⚠️ often won't fit | **bandwidth-bound** + fit pressure | 🟢 speed **and** fit |
| Laptop / iGPU / Jetson / edge | mobile dGPU, Orin, iGPU | ~100–270 GB/s | ≤8 GB | ❌ can't fit 15 GB | **strongly bandwidth-bound** | 🟢 **required to run at all** |

At B_device ≈ 300 GB/s a full int4 weight read is `15.3/0.3 ≈ 51 ms/token` — **~2.4× the H200 latency floor**, so it is now the dominant term and halving the bytes (int2) would roughly halve *that* term. The byte axis that is dead on H200 is the primary lever here (projection).

**The two distinct values of lowbit, separated.** These are independent and must not be conflated:

1. **Speed** — only pays off in the bandwidth-bound regime (mid-consumer and below). On H200 it does not; on a 300 GB/s device it plausibly does (projection).
2. **Fit-ability / footprint** — a 30B model at int4 ≈ **15 GB won't load** on ≤12 GB devices at all. int3 (~11.5 GB) / int2 (~7.7 GB) shrinks it enough to *run*. **This is a portability win entirely independent of the speed roofline** and may be the stronger motivation: on an 8 GB laptop GPU the choice is not "faster vs slower" but "runs vs doesn't run." Fit-ability does not care that H200 is latency-bound.

---

## 7. Recommendation & the reopened lever

**Ranking** *(H200/datacenter tier — see §6 for the consumer/edge picture, where rows 1–4 move up)* by (measured tok/s gain × accuracy safety × impl+tooling cost) — gains use the **measured** ~3.5% weight-bound fraction, not the pre-probe Amdahl estimate:

| rank | option | measured-grounded gain | accuracy | cost | verdict |
|---|---|---|---|---|---|
| — | **decode megakernel / drastic node-count collapse** | the only axis the probe leaves open (attacks the ~8.2 µs × 2568-node floor) | 🟢 (structural) | **L** (new effort) | **REOPENED — the real lever** |
| 1 | int3 via GPTQ/AWQ from bf16 source | **~+1–2% (~48 tok/s)** | 🟡 | **L** (source+recipe+kernel) | 🟥 **NO-GO** (measured) |
| 2 | mixed int4-attn / int2-MLP | ~+1–2% | 🟡/🔴 | **L** | 🟥 NO-GO |
| 3 | int2-everywhere | **~+1.6% (~48 tok/s)** | 🔴 | **L** | 🟥 NO-GO (measured; accuracy cliff too) |
| 4 | 2:4 sparse int4 | ~+1–2% | 🟡 | **L** (no M=1 HW benefit) | 🟥 NO-GO for decode |
| 5 | NF4/AF4 | 0% bytes | 🟢 | S | ➖ enabler only, irrelevant given the probe |

**Go/no-go — MEASURED, and now DEVICE-CONDITIONED (see §6):** the blanket NO-GO applies **only to the H200/datacenter tier we measured**:

- **H200 / datacenter (measured):** 🟥 **NO-GO for speed.** The bandwidth probe empirically caps the byte-reduction direction at **~+3% best case (bytes→0)**; int2 (−45%) buys ~+1.6%, behind a 🔴 accuracy cliff and an L-sized source-weights+recipe+kernel dependency. On this tier the lever is the **megakernel / node-collapse** below.
- **Consumer / edge (projected, §6):** 🟢/🟡 **KEEP ON ROADMAP.** As `B_device` drops below the ~0.7 TB/s crossover the weight read stops being hidden and lower-bit quant becomes a real **speed** lever, and below ~12 GB VRAM it is the only way to **fit** a 30B model at all (int4 ≈ 15 GB won't load). The fit-ability win is independent of the speed roofline and may be the stronger motivation.
  - **Concrete gate before investing:** we only have an H200 here and therefore **cannot measure** the consumer/edge regime — the next validation step is to run **this same `ONNX_GENAI_WEIGHT_FOLD` byte-fold probe on a representative consumer/edge GPU** (e.g. RTX 4070 ~500 GB/s and a ≤8 GB laptop dGPU). If the probe slope there is steep (byte cut → large tok/s gain), lowbit is a GO for that tier.
  - The **accuracy path is device-independent** (Fact Checker: int3 / ~3.5 bpw imatrix / SpQR 🟢; int2 needs codebook/trellis methods 🟡; scalar int2 🔴; **all** require re-quant from the fp16 source — which we do not have staged — plus new sub-4-bit kernels).

The largest safe byte cut still matters *only* where the device is bandwidth-bound; on H200 it would only matter after the node-count floor is broken (i.e. once decode is bandwidth-bound again — which it currently is not, by measurement).

**Reopened lever — decode megakernel.** The probe leaves exactly one axis open: the serial ~2568-node latency chain. The fusion arc (#872/#873) only disproved *marginal* fusion (removing 4–8% of nodes → flat/regressive); it did **not** test *drastic* collapse. A persistent/megakernel that fuses a whole decoder layer's elementwise+norm+GEMV epilogue chain into few launches — keeping activations resident and paying per-kernel latency a handful of times per layer instead of ~49 — is the only direction consistent with all three measurements (byte-fold flat, marginal-fusion flat, 15% HBM util). It is a large effort and itself needs a prototype/probe (fuse one layer, measure the per-layer node count and tok/s), but it is the **true next lever** and should replace lower-bit quant on the roadmap.

**PROGRESS.md correction call (explicit).** The 2026-08-13 entry *"Native CUDA decode ceiling: bandwidth-bound at ~47 tok/s (#870/#872/#873)"* wording — **"weight-bandwidth/compute-floor bound at ~47.25 tok/s … the roofline is bytes-moved, not launches"** — is **WRONG / misleading and should be corrected.** It inferred "bandwidth-bound" from a *negative* (fusion flat) without ever measuring the bandwidth axis; the direct byte-fold probe now refutes it (−75% weight bytes → +2.8%). Suggested replacement wording: *"latency-bound on the serial ~2568-node dependency chain (~8.2 µs/node); **neither** weight-bandwidth-bound (a 4× weight-byte cut yields only +2.8%, HBM util ~15%) **nor** removable by marginal node fusion — only drastic node-count collapse (a decode megakernel) can move it."* The entry's *conclusions* (small fusion is a dead end; 47.25 is the ceiling for the current kernel/graph structure) stand; only the **mechanism attribution** ("bandwidth") is wrong. The "Real levers to beat 47" bullet should drop *"reduce weight bytes/token (lower-bit quant, sparsity)"* and keep only *"decode megakernel."* (Left to the coordinator/Scribe to edit that dated entry, per source-of-truth ownership.)

### Appendix — reproducibility

- Byte budget: enumerated from `decoder/model.onnx` initializer dims (417 `MatMulNBits`, bits=4/bs=32/asymmetric/bf16-scales), summed in Python; cross-checked vs `model.onnx.data` = 15.367 GB.
- Roofline: 15.325 GB × 47.25 tok/s = 724 GB/s; H200 HBM3e peak ≈ 4.8 TB/s (per task spec / NVIDIA H200 SXM datasheet class).
- Baseline 47.25 tok/s: prior merged measurement (#867), `CUDA_VISIBLE_DEVICES=0 ONNX_GENAI_CUDA_GRAPH=1 profile_native --pipeline --ep cuda --backend native --steady --warmups 1 --runs 3 --tokens 128`, capture 1 seg/0 seams.
- Node count 2568/token, ~8.2 µs/node: prior finding #870 (dispatch-bound).
- **Bandwidth probe (this task):** throwaway `ONNX_GENAI_WEIGHT_FOLD=D` flag folds the weight-read column of `matmul_nbits_gemv_f16_scales_f16_zp_splitk` (the sole int4 decode GEMV entry that fires for Muse-Glimmer, incl. lm_head; verified via one-shot NVRTC entry logging) so the packed-weight/scale/zp DRAM footprint shrinks to 1/D with loop-trip/instruction/launch/node-count held byte-identical. Measured median (3×128-tok, `ONNX_GENAI_CUDA_GRAPH=1`, `--pipeline`): D=1 → 47.29, D=2 → 47.98, D=4 → 48.62 tok/s. **Probe code was reverted and is NOT shipped.**
- Kernel bits support: `crates/onnx-runtime-ep-cuda/src/kernels/matmul_nbits.rs` dispatches only `bits∈{4,8}`; `uint4` (128-bit) weight loads; `accuracy_level=4` dp4a int8-activation path present.
- Tooling: `meta-models-Muse-Glimmer-30B/cuda/int4/config.json` (`OnnxKQuantQuantization bits=4 block_size=32`); only `cuda/int4` exists under the model root; source `meta-models/Muse-Glimmer-30B` (bf16, HF).
- **All tok/s beyond 47.25 are projections; no accuracy/perplexity numbers were measured — risk flags cite method class only.**
