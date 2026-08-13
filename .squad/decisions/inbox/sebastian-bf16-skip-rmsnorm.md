### 2026-08-13: bf16 skip-RMSNorm — byte-exact KERNEL ships; standalone FOLD is a measured no-ship (opt-in, default OFF)

**By:** Sebastian
**Refs:** builds on Batty's #900 (§8.5 blocker); PR references #900. Doc: `docs/research/dense-decode-megakernel-feasibility.md` §8.6.

**What:** Closed the §8.5 blocker — the missing **byte-exact bf16 skip-RMSNorm kernel** for
Muse-Glimmer-30B (Gemma3 sandwich-norm, 6 norms/layer × 52 = 312 norm nodes, seam
`Add(residual, sublayer_out) → SimplifiedLayerNormalization`).

1. **`skip_rmsnorm_bf16` NVRTC kernel** (`crates/onnx-runtime-ep-cuda/src/kernels/normalization.rs`):
   `sum = __float2bfloat16_rn(f32(residual)+f32(x))` (bit-for-bit a standalone bf16 `Add`), stores the
   bf16-rounded sum as the next layer's residual, then the **identical** `rmsnorm_bf16` block-tree
   reduction (fp32 accumulate over the *rounded* sum, same `NORM_BLOCK=256`). Guarded native dispatch
   with `run_bf16_via_f32` fallback for non-dense/bias/header-unavailable (Rule 11 portability).
2. **`CudaSkipRmsNormFusion`** optimizer fold (`optimizer.rs`): collapses the seam into one
   `com.microsoft::SkipSimplifiedLayerNormalization`, deleting the standalone `Add`+norm. bf16 (or
   f32-gamma-over-bf16) only; fp16 left to `CudaSkipRmsNormMatMulFusion`.

**Numeric fidelity (Chew gate): BYTE-EXACT — 0-ulp.**
- GPU unit tests: `bf16_native_skip_rmsnorm_is_byte_exact_with_{bf16,f32}_gamma` pass (bit-identical vs
  standalone `Add(bf16)`→`rmsnorm_bf16` at H=6656).
- Real-model greedy stream fold OFF vs ON: **bit-identical** (48/48 and 128/128 tokens),
  `[24, 372, 1045, 10016, 328, 2885, 262, 5091, 8811, 511, 917, 4921, 768, …]`. No reduction reordered.

**Perf A/B (H200, `CUDA_VISIBLE_DEVICES=0`, `ONNX_GENAI_CUDA_GRAPH=1`, `--steady --warmups 1 --runs 3
--tokens 128`, interleaved, one release binary):**
- fold OFF (default): **47.77 tok/s** (20.93 ms/token).
- fold ON: **47.06 tok/s** (21.25 ms/token) = **−1.5% REGRESSION**. 104 seams folded (2/layer × 52).
- Eager per-op timer (mechanism confirm): fused glue *is* faster eagerly (Add+norm+skip 355.6 → 349.0 ms)
  — the launch saving is real, but **graph replay already amortizes launches** (§8.1 ~0.9 µs/node floor),
  so it does not surface in wall-clock.

**Why the fold regresses (mechanism, honest):** at M=1 the RMS reduction is **single-CTA** (all H=6656
in one block). Folding the residual add into it **serializes** work that the standalone `Add` had spread
across all 132 SMs (whole-GPU, multi-CTA). Under replay the launch saving is already banked, so the
fused single-CTA skip kernel is strictly heavier than `multi-CTA Add + single-CTA norm`. Same structural
reason the multi-CTA GEMV megakernel was NO-GO (§7/#898) and glue collapse paid only +0.9% (§8.5/#900):
**decode is at its launch-amortized latency floor.**

**Verdict:**
- **KERNEL: SHIP** — proven byte-exact (0-ulp), portability-gated, no production behavior change. It is
  the prerequisite for the only bf16 path that could win: folding the norm into the neighbouring
  **multi-CTA int4 GEMV** prologue/epilogue (bf16 analogue of `CudaSkipRmsNormMatMulFusion`, keeps the
  reduction distributed across the GEMV's CTAs) — a larger GEMV-kernel job, NOT this fold.
- **FOLD: NO-SHIP as default** — retained behind `ONNX_GENAI_CUDA_ENABLE_SKIP_RMSNORM_FUSION` (**default
  OFF**) for A/B and for a future bandwidth-bound device where the multi-CTA-Add-vs-single-CTA-skip trade
  may flip. Default binary unchanged (47.77 tok/s == baseline; no regression).

**Validation:** `cargo fmt --all -- --check` clean; clippy clean on the ep-cuda crate (pre-existing
optimizer.rs:3347 PI + standard_attention.rs warnings are not mine); 61 optimizer unit tests + 20
normalization (incl. 2 new byte-exact) tests pass. CUDA EP cdylib `ldd`-confirmed **not** linking
`libonnxruntime`.

**Do NOT self-merge** — Chew gates numerics (byte-exact result above is the artifact). Even though the
kernel is 0-ulp, the fold-as-default is a no-ship on perf grounds regardless.
