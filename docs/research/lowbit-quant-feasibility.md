# Lower-bit quantization + structured sparsity feasibility for sub-47 tok/s decode

**Author:** Sebastian (Performance/Systems) · **Date:** 2026-08-13 · **Status:** feasibility / decision-support (NO kernel work in this brief)
**Model:** Muse-Glimmer-30B, `cuda/int4` Olive package · **HW:** H200 SXM (HBM3e ≈ 4.8 TB/s) · **Regime:** M=1 captured decode, `ONNX_GENAI_CUDA_GRAPH=1`
**Baseline:** 47.25 tok/s median (native CUDA, capture 1 seg / 0 seams), the current ceiling after the #855/#854/#860/#867/#870 arc.

> **Headline (read this first).** At 47.25 tok/s the decoder reads **15.3 GB of weights/token at only 724 GB/s — 15% of the H200's 4.8 TB/s roofline.** Decode is **dispatch-bound (2568 captured nodes/token, ~8.2 µs/node), NOT aggregate-weight-bandwidth-bound.** Therefore the naive "halve the bytes → double tok/s" projection is **wrong by ~2×**: an Amdahl analysis puts even *int2-everywhere* at ~**54 tok/s (+14%)**, not 94, and that is gated behind a 🔴 accuracy cliff and an **L-sized tooling+kernel dependency we do not currently have** (only int4 weights exist; our GEMV supports only bits∈{4,8}). **Recommendation: lower-bit quant is a NO-GO as the next lever.** The evidence-backed next move is (a) a ~1-day *bandwidth probe* micro-experiment to empirically confirm the payoff ceiling before any quant tooling is funded, and (b) if a sub-4-bit path is still wanted, **int3 (never int2-everywhere) via GPTQ/AWQ from the bf16 source**, scoped as a large multi-team effort.

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

**Projected decode tok/s — naive (assume bandwidth-bound) vs realistic (Amdahl, `W≈0.28`):**
`realistic speedup = 1 / ((1−W) + x·W)`, `x = ×base`.

| format | ×base bytes | naive tok/s (W=1) | **realistic tok/s (W≈0.28)** | accuracy risk | tooling dep |
|---|---:|---:|---:|:--:|:--:|
| int3 | 0.777 | 60.8 | **~50.0** | 🟡 (GPTQ/AWQ ok; K-quant Q3_K ok) | re-quant + new kernel |
| int2 | 0.554 | 85.3 | **~54.1** | 🔴 (cliff without QuIP#/AQLM-class) | re-quant + new kernel |
| mixed int4/int2 | 0.608 | 77.7 | **~53.0** | 🟡/🔴 (MLP int2 is the risk) | re-quant + mixed-bit kernel |
| 2:4 sparse int4 | 0.594 | 79.5 | **~53.4** | 🟡 (needs sparse-aware fine-tune) | sparse GEMV from scratch |
| NF4/AF4 | 1.000 | 47.25 | **47.25** | 🟢 (accuracy *enabler*, not byte saver) | LUT-dequant kernel |

**The gap between the naive and realistic columns is the whole story:** because we sit at 15% of the memory roofline, the biggest theoretically-available byte cut (int2, −45%) buys only **+14% tok/s** in practice, and only if the accuracy cliff and tooling are solved. NF4/AF4 saves **zero** bytes (still 4-bit) — its only value is improving accuracy-per-bit so that int3/int2 becomes *tolerable*; it is an enabler for the rows above, not a lever by itself.

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

## 5. Recommendation & smallest de-risking experiment

**Ranking** by (expected tok/s gain × accuracy safety × impl+tooling cost):

| rank | option | realistic gain | accuracy | cost | verdict |
|---|---|---|---|---|---|
| — | **(not lower-bit) more node-count fusion** | attacks the *actual* bound (2568 nodes) | 🟢 (structural) | already in-flight (Batty) | **the real lever** |
| 1 | **int3 via GPTQ/AWQ from bf16 source** | ~50 (+6%) | 🟡 | **L** (source+recipe+kernel) | 🟥 **NO-GO now**, revisit only if fusion tapped out |
| 2 | mixed int4-attn / int3-MLP | ~51 (+8%) | 🟡 | **L** | 🟥 NO-GO now |
| 3 | int2-everywhere | ~54 (+14%) | 🔴 | **L** | 🟥 NO-GO (accuracy cliff) |
| 4 | 2:4 sparse int4 | ~53 | 🟡 | **L** (no M=1 HW benefit) | 🟥 NO-GO for decode |
| 5 | NF4/AF4 | 0% bytes | 🟢 | S | ➖ only as an *enabler* for #1–#3 |

**Go/no-go:** **NO-GO on lower-bit quantization as the next perf lever.** It is the wrong tool for a dispatch-bound workload: the maximum realistic prize is ~+14% (int2), it sits behind a 🔴 accuracy cliff and an L-sized source-weights + recipe + new-kernel dependency we don't have, while the measured bottleneck (2568 nodes/token) is untouched by it. Byte-reduction only becomes the right lever **after** node-count fusion has pushed `W` (the weight-bound fraction) back up toward 1 — i.e. once decode is actually bandwidth-bound again.

### The single smallest experiment that de-risks the biggest uncertainty

Two cheap, **kernel-only, no-tooling** micro-experiments — do the first before funding *anything*:

1. **Bandwidth probe (≈1 day, decisive).** In a throwaway build, modify the int4 GEMV to **read only half the packed-weight bytes** (e.g. skip alternate `uint4` loads and double-count, or memset-alias the upper nibbles) — deliberately wrong numerically, but it makes the kernel move ~0.5× weight bytes. Run the standard `profile_native --pipeline --tokens 128` steady loop and read tok/s. **This empirically measures `W` and the true tok/s ceiling of *any* byte reduction — for free, with no re-quantization, no new format, no accuracy work.** If tok/s barely moves (predicted: 47→~52–54), that is the quantitative kill-shot for the whole lower-bit direction; if it jumps toward the naive ~85, the roofline model is wrong and lower-bit becomes worth funding. **Either way we learn the answer in a day instead of weeks.**

2. **Accuracy probe (only if #1 is favorable).** Before writing production kernels, quantize **just the MLP `down_proj` of a few layers to int3 and int2** from the bf16 source (GPTQ, offline, Python/numpy — no CUDA kernel needed) and measure **perplexity delta** on a small eval set. This isolates the accuracy cliff (the true 🔴) at near-zero engineering cost and tells us whether int3 (🟡) or nothing (int2 🔴) is the viable floor.

**Bottom line for the coordinator:** don't fund lower-bit quant yet. Run the 1-day bandwidth probe (#5.1). Expect it to confirm decode is dispatch-bound and cap the lower-bit prize at ~+14%, redirecting effort back to node-count fusion — the lever that actually matches the bottleneck.

---

### Appendix — reproducibility

- Byte budget: enumerated from `decoder/model.onnx` initializer dims (417 `MatMulNBits`, bits=4/bs=32/asymmetric/bf16-scales), summed in Python; cross-checked vs `model.onnx.data` = 15.367 GB.
- Roofline: 15.325 GB × 47.25 tok/s = 724 GB/s; H200 HBM3e peak ≈ 4.8 TB/s (per task spec / NVIDIA H200 SXM datasheet class).
- Baseline 47.25 tok/s: prior merged measurement (#867), `CUDA_VISIBLE_DEVICES=0 ONNX_GENAI_CUDA_GRAPH=1 profile_native --pipeline --ep cuda --backend native --steady --warmups 1 --runs 3 --tokens 128`, capture 1 seg/0 seams.
- Node count 2568/token, ~8.2 µs/node: prior finding #870 (dispatch-bound).
- Kernel bits support: `crates/onnx-runtime-ep-cuda/src/kernels/matmul_nbits.rs` dispatches only `bits∈{4,8}`; `uint4` (128-bit) weight loads; `accuracy_level=4` dp4a int8-activation path present.
- Tooling: `meta-models-Muse-Glimmer-30B/cuda/int4/config.json` (`OnnxKQuantQuantization bits=4 block_size=32`); only `cuda/int4` exists under the model root; source `meta-models/Muse-Glimmer-30B` (bf16, HF).
- **All tok/s beyond 47.25 are projections; no accuracy/perplexity numbers were measured — risk flags cite method class only.**
