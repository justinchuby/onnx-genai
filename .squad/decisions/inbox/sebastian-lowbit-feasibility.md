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
