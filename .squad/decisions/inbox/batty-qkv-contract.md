### 2026-08-13: QKV-projection fusion — graph-only, byte-exact, NO new kernel needed (heads-up + decisive dispatch-vs-bandwidth test)

**By:** Batty (Decode-graph / pipeline)
**For:** Sebastian (kernels), coordinator

**What / good news:** Testing the one untested lever from #872 — reducing the count of *expensive* nodes (not cheap ones). The 3 per-layer attention projections (`q_proj` N=4096, `k_proj` N=256, `v_proj` N=256) are separate `MatMulNBits` GEMVs all reading the *same* `input_layernorm` activation (K=6656, block_size=32, bits=4, **bf16** activation/scales/output, uint4 zero-points). I'm fusing them **graph-side only** by physically column-concatenating their int4 weight / scale / zero-point initializers into one `[N=4608]` `MatMulNBits`, then a single `Split(axis=-1, split=[4096,256,256])` demuxes the output back to the existing q→qk_norm, k→qk_norm, v→GQA consumers.

**No new kernel required** — this reuses the existing `MatMulNBits` kernel (just wider N) and the existing capture-safe `Split` kernel. So there is **no dependency on you** for the measurement. (Contrast with the gate/up paired kernel, which keeps weights separate; that fp16-only pass does not fire here because Muse-Glimmer is bf16.)

**Byte-exactness:** each output column's math (dequant `(code-zp)*scale`, K-reduction) is untouched by concatenating rows → output `[0:4096]`≡q, `[4096:4352]`≡k, `[4352:4608]`≡v, bit-for-bit. Split is a pure copy. Expect byte-identical greedy ids (ref first-16 `[24, 372, 1045, 10016, 328, 2885, 262, 5091, 8811, 511, 917, 4921, 768, 328, 2885, 262]`). If the wider-N `MatMulNBits` kernel changes per-column K-accumulation order and ids diverge → I flag **Chew** and stop.

**Node effect/token:** MatMulNBits **417→313** (−104 GEMV launches, 2/layer×52); +52 `Split` nodes; net −52 nodes. **The decisive test:** if tok/s improves → decode is dispatch-bound and expensive-launch count is the lever (o_proj / mlp down/gate/up may fuse similarly next). If flat/worse → decode is int4-weight-bandwidth/compute-floor bound and 47.25 is the architectural ceiling (fusing disjoint-weight GEMVs cannot cut bytes). Either way I report honestly like #872 and do **not** ship a regression.

**Possible follow-up for you (only if this wins):** a fused-launch QKV kernel that writes Q/K/V to 3 destinations from one launch would drop even the 52 `Split` copies. Not needed for the measurement; flagging so you can pre-plan the epilogue variant.

**PR:** `perf(cuda-ep): fuse QKV projections into one MatMulNBits (47→NN tok/s)` off main 64c138fa.

---

### 2026-08-13: RESULT — decisive negative finding (flat), fusion shipped OPT-IN / disabled-by-default

**Measured — Muse-Glimmer-30B int4, CUDA graph, GPU0 idle, warmups 2 / runs 5 / tokens 64, 3 interleaved trials on ONE release binary:**

| Config | tok/s (trials) | median |
|---|---|---|
| Baseline (fusion off, default) | 47.23 / 47.33 / 47.38 | **47.33** |
| Fused QKV (`ENABLE_QKV_FUSION=1`) | 47.21 / 47.26 / 47.28 | **47.26** |

- **Node effect (empirical):** `MatMulNBits` **417 → 313** (−104 GEMV launches/token, 2/layer × 52), `Split` **0 → 52**, GQA 52 unchanged. Capture **1 segment / 0 eager seams** both configs.
- **Parity: BYTE-EXACT.** All 64 `generated_token_ids` identical between configs; first-16 match reference `[24, 372, 1045, 10016, 328, 2885, 262, 5091, 8811, 511, 917, 4921, 768, 328, 2885, 262]`. **No Chew flag needed** — pure structural fusion, same math.
- **Throughput: FLAT / marginally worse** (−0.07 tok/s, within run-to-run noise).

**Conclusion:** Removing 104 *expensive* GEMV launches yields **zero** throughput gain — the definitive counterpart to #870 (GQA) and #872 (cheap Add fold). Decode of this int4 model is **weight-bandwidth / compute-floor bound**, NOT dispatch-bound on either the cheap OR the expensive path. The three projections read disjoint int4 weights, so fusing cannot reduce bytes moved; the fused wider GEMV reads the same total bytes as the three separate ones. **47.25 tok/s is the architectural ceiling** for native CUDA decode of Muse-Glimmer-30B. **We stop here and bank the clean 47.25 win** (already beats ORT).

**Disposition (no regression shipped):** the pass is correct, tested (4 unit tests), and byte-exact, so it is **retained but disabled by default**, opt-in via `ONNX_GENAI_CUDA_ENABLE_QKV_FUSION=1`. Default binary keeps the 3 separate GEMVs (baseline 47.33). Preserved for future dispatch-bound architectures (e.g. fp16 activations, or shapes with a higher launch-latency-to-bandwidth ratio) where expensive-launch reduction could pay off.

**Sebastian:** the fused-launch QKV epilogue kernel (Q/K/V to 3 destinations from one launch, dropping the 52 Splits) is **NOT worth building** — even at −104 launches / −52 Splits the ceiling is bandwidth, so a 3-destination kernel would also be flat. Don't spend cycles there.
