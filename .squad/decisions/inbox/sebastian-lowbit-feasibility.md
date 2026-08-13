### 2026-08-13: Lower-bit quant + sparsity feasibility — NO-GO as next perf lever (decode is dispatch-bound, not weight-bandwidth-bound)

**By:** Sebastian (Performance/Systems). Full brief: `docs/research/lowbit-quant-feasibility.md` (PR: `docs(research): lower-bit quant + sparsity feasibility for sub-47 tok/s decode`).

**The one number that decides it:** at 47.25 tok/s the decoder reads **15.325 GB weights/token at 724 GB/s = 15% of the H200's 4.8 TB/s roofline.** Decode is **dispatch-bound (2568 captured nodes/token, ~8.2 µs/node)**, NOT aggregate-weight-bandwidth-bound (confirms #870). If it were bandwidth-bound we'd be at ~313 tok/s. **So byte reduction only speeds up the ~28% of the token that is actually weight-read time (Amdahl W≈0.28).**

**Byte budget (measured from the real ONNX, 417 MatMulNBits, bits=4/bs=32/asymmetric/bf16-scales):** packed weights 13 254 MB + bf16 scales 1 657 MB + int4 zero-points 414 MB = **15 325 MB/token** (cross-checks vs 15.37 GB `model.onnx.data`). MLP = 78% of bytes; scales/zp metadata = 13.5% and **the bf16 scale floor does NOT shrink with fewer bits**, so int2 total = **0.554×**, not 0.5×.

**Realistic tok/s projections (Amdahl W≈0.28) vs naive:**
| format | ×bytes | naive | realistic | risk |
|---|---|---|---|---|
| int3 | 0.777 | 60.8 | **~50 (+6%)** | 🟡 |
| int2 | 0.554 | 85.3 | **~54 (+14%)** | 🔴 cliff |
| mixed int4/int2 | 0.608 | 77.7 | **~53** | 🟡/🔴 |
| 2:4 sparse int4 | 0.594 | 79.5 | **~53** | 🟡 (no M=1 HW benefit) |
| NF4/AF4 | 1.000 | 47.25 | 47.25 | 🟢 enabler only, 0 bytes saved |

**The dominant blocker (likely fatal near-term):** we only HAVE int4. Sub-4-bit requires re-quantizing from the ~60 GB bf16 HF source (not staged) with a calibrated method (GPTQ/AWQ/QuIP#) — the current Olive recipe is `OnnxKQuantQuantization(bits=4)` only. AND neither ORT nor our kernel runs bits∉{4,8} (our GEMV: int4/int8 only; int3 doesn't byte-align → bfe/funnel-shift, M–L kernel; int2 is clean, S–M). So it's an **L-sized source+recipe+kernel+numerics chain** for a capped ~+14% best case.

**Recommendation — NO-GO on lower-bit quant as the next lever.** It's the wrong tool for a dispatch-bound workload. The real lever remains **node-count fusion** (Batty's in-flight work) — it attacks the actual 2568-node bottleneck. Byte reduction only pays off *after* fusion pushes W back toward 1 (decode becomes bandwidth-bound again).

**Smallest experiment to de-risk before funding anything (≈1 day, kernel-only, NO tooling):** a **bandwidth probe** — hack the int4 GEMV to read only half the packed bytes (numerically wrong on purpose) and measure tok/s on the standard steady profile. This empirically measures W and the true ceiling of ANY byte reduction for free. Predicted 47→~52–54 (kill-shot for lower-bit); if it jumps toward ~85 the roofline model is wrong and lower-bit becomes worth funding. Only if favorable: an offline GPTQ int3/int2 perplexity probe on a few MLP down_proj layers (Python, no kernel) to test the accuracy cliff.

**Go/no-go per format:** int3 🟥 (defer), int2 🟥 (accuracy cliff), mixed 🟥 (defer), 2:4 sparse 🟥 (no M=1 tensor-core benefit), NF4/AF4 ➖ (enabler only). All tok/s are projections; no accuracy numbers measured — risk flags cite method class. Chew is the numerics gate if any sub-4-bit path is later funded.

---

### 2026-08-13 (UPDATE): Bandwidth probe RESOLVES the tension — latency-chain bound, NOT bandwidth-bound. Lower-bit is a MEASURED no-go; megakernel REOPENED.

I ran the arbiter probe (env `ONNX_GENAI_WEIGHT_FOLD=D`, throwaway/reverted/never shipped): fold the weight-read column of the sole int4 decode GEMV that fires for Muse-Glimmer (`matmul_nbits_gemv_f16_scales_f16_zp_splitk`, incl. lm_head) so all output columns alias into the first N/D weight rows → packed+scale+zp **DRAM footprint → 1/D** with loop-trip/instruction/launch/node-count byte-identical. This faithfully isolates memory-throughput (lower-bit keeps the same K-block count, only fewer bytes/weight — a K-shortening probe would be unfaithful).

**Measured (H200, CUDA_GRAPH=1, --pipeline, 3×128-tok median, 1seg/0seams):** full(D=1) **47.29** → half(D=2) **47.98** (+1.5%) → quarter(D=4) **48.62** (+2.8%).

**Verdict — "flat" branch fired.** −75% weight DRAM → +2.8% ⇒ weight-DRAM-bound fraction ≈ **3–4%**. Decode is **empirically NOT weight-bandwidth-bound**. int2-everywhere (−45% bytes) → **≈+1.6% (~48 tok/s)**, not +14%/+80%. **Lower-bit quant (all variants) = MEASURED 🟥 NO-GO** as next lever (on top of 🔴 accuracy cliff + the "only-int4-exists, sub-4-bit needs re-quant from fp16 source + new kernels" L-blocker).

**Bound by:** serial critical-path latency of the ~2568-node chain (~8.2 µs/node × 2568 ≈ 21 ms/token). Reconciles both negatives: not bandwidth (byte-fold flat) AND not marginal-node-sensitive (#872/#873 flat/regressive).

**Megakernel REOPENED as the true lever** — #872/#873 only disproved *marginal* fusion, never *drastic* collapse. A layer-level persistent/megakernel (activations resident, few launches/layer) is the only direction consistent with all three measurements. Large effort; needs a one-layer prototype. Should replace lower-bit on the roadmap.

**PROGRESS.md correction call (EXPLICIT):** the 2026-08-13 "bandwidth-bound at ~47 tok/s (#870/#872/#873)" entry wording "weight-bandwidth/compute-floor bound … roofline is bytes-moved, not launches" is **WRONG/misleading — correct it.** Suggested: "latency-bound on the serial ~2568-node dependency chain (~8.2 µs/node); neither weight-bandwidth-bound (4× byte cut → +2.8%, HBM util ~15%) nor removable by marginal node fusion — only drastic node-count collapse (decode megakernel) can move it." Entry's *conclusions* (marginal fusion dead, 47.25 current-structure ceiling, QKV-fusion opt-in) stand; only the **mechanism attribution** ("bandwidth") + the "reduce weight bytes/token (lower-bit, sparsity)" future-lever bullet are wrong. Left to coordinator/Scribe to edit that dated entry.
