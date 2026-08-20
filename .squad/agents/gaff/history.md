# Gaff — History (compacted 2026-08-19T16:40Z)

## Project context
- Review specialist for onnx-genai correctness, runtime/loader boundaries, transactional semantics, and validation quality.
- Joined 2026-07-12. In gaff-13 wave, role expanded to kernel fusion engineering (CUDA, capture-safe).

## Condensed prior record through 2026-08-14

**Review work (2026-08-11 to 2026-08-14):**
- PR #762 (CUDA EP correctness): four rounds of focused delta review across commits c1d2556b5…bb280c0ea; substantive findings (test helper duplication, `validate_write_dtype` dead in production, `find_ort_lib_dir` drift) flagged non-blocking; ready-to-leave-draft verdict delivered.
- microsoft/onnxruntime #31988 (LayerNorm perf): two review passes (pre/post Chew guard, commit a4aa076657); bit-identicality + routing invariance confirmed; kept draft pending GPU benchmarks on ≥2 GPU generations.
- PR #31973 (evidence-accuracy fix): accuracy figures reproduced to 4 sig figs; nullptr MeanOut fix confirmed; test counts 41/2/43 verified; one nit (RMSNorm 3.3× figure optimistic vs measured 2.84×).
- PR #960 (Marlin int4 M>1 GEMM, Deckard): 🟢 APPROVE. Rule 11 PASS (genuinely opt-in, SM80-gated, byte-identical fallback). Capture-safety valve family sound (4 caches, one contract). Env-var honesty PASS. Zero blocking defects.

**Durable rules reinforced:**
- Env-var honesty + byte-identical default-OFF fallback = portability contract for any tier-scoped kernel.
- Reviewer lockout discipline: real blockers transfer revision ownership; do not revise own rejected artifacts.
- Benchmark comparisons must be matched and reproducible; `validate_write_dtype` dead-in-production pattern must be flagged even if non-blocking.

_Detailed dated entries for 2026-08-11 through 2026-08-14 moved to history-archive.md._

## 2026-08-19T16:40Z — SSM-reduce fusion + CudaRsqrtFusion (gaff-13, PR #1486)

- **SSM-reduce fusion:** Routed f16 ReduceSum/ReduceMean to NVRTC block-reduction (cuDNN gate narrowed to fp32 only). Eliminates 3-kernel fp32 round-trip (20.7% of decoded GPU time per Deckard's per-op profile). Result: 214.4 → 231.6 tok/s (+7.4%), byte-identical, captures=4. Reclaim 0.346 ms = exactly Deckard's 0.3–0.4 ms prediction.
- **CudaRsqrtFusion (glue-tail):** Collapsed `Sqrt+Reciprocal` → `rsqrt` (clean algebraic fusion, bit-identical, capture-safe, general). Largest glue buckets (mul 9%, transpose/split ~8%) rejected as layout-invasive or graph-structure-dependent. Result: 231.6 → 233.6 tok/s (+0.8%).
- **Combined PR #1486:** 214.4 → 233.6 tok/s (**+8.9%**), byte-identical greedy tokens, captures=4 preserved.
- **Lesson reinforced:** Profiling-guided fusion (Deckard's per-op profile → fix hypothesis → implementation) beats speculative fusion. Validate with a faithful captured-graph profile (nsys `--cuda-graph-trace=node`) before building any fusion.

## 2026-08-19T18:20Z — LinearAttention decode fusion assigned (Deckard attribution)

Deckard's kernel attribution (deckard-8, 2026-08-19) identified the **gated-delta LinearAttention decode path as the active #1 lever** for closing the ~24% native-vs-ORT gap on qwen3.5-0.8b-hybrid fp16io. The decomposed native path (fp32 cuDNN ReduceSum + transpose/split/concat data-shuffle + reciprocal/sqrt/sigmoid/softplus/mul gating chain) costs ~700–900 µs/step vs ORT's fused `LinearAttentionDecodeColKernel` at ~250 µs/step. Task: fuse native's path into one fp16 kernel to recover ~650–900 µs/step → est. ~250–256 tok/s, closing most of the gap to ORT ~284–296. Int4 GEMV latency-hiding is #2 after this. Coordinator has dispatched Gaff to implement.

## 2026-08-19T19:00Z — LinearAttention fp32 ReduceSum fusion (gaff-14, PR #1495)

- **ReduceSum fusion (fp32):** Routed the fp32 cuDNN ReduceSum in the gated-delta LinearAttention decode path onto the NVRTC block reduction (fp32 analogue of #1486). Result: 200.95 → 209.71 tok/s (**+4.36%**), byte-identical, golden lock PASS. Coordinator independently re-validated on GPU0 (22.77 s). Admin-merged (squash); worktree swept.

## 2026-08-19T19:45Z — LinearAttention gating chain fusion (gaff-15, PR #1496)

- **CudaLinearAttentionGatingFusion (structural pass):** Folds beta (Sigmoid) and decay (exp·Softplus) gate chains into the LinearAttention kernel epilogue. Drops ~5 elementwise nodes/layer × 18 layers from captured decode graph. Result: 229.4 → 237.4 tok/s (**+3.4%**), byte-identical, golden lock PASS. Coordinator independently re-validated on GPU0 (32.62 s). Admin-merged (squash). 3 new unit tests. Opt-out: `ONNX_GENAI_CUDA_DISABLE_LINATTN_GATING_FUSION=1`.
- **Sub-lever #2 (layout / L2-norm addressing) — DEFERRED:** Needs a dedicated byte-lock harness before attempting: (1) ORT ReduceSum exact summation order inside LA loop (byte-divergence risk); (2) strided-input plumbing (`supports_strided_input=false` — large blast radius). Out of clean scope; should be tackled in isolation.

## 2026-08-19T20:00Z — int4 MatMulNBits GEMV occupancy gate (gaff-16, PR #1501)

- **Occupancy-gated GEMV entry selection:** Routes well-occupied launches (`ceil(N/cols) >= mp_count*32`, e.g. LM head N=248320) to low-register plain entry (−14%: 98.8→85.0 µs/call); keeps prefetch-pipelined entry for grid-starved projections (already at byte-identical floor ~4.3–4.7 µs/call). Gate keys on N/launch-width/SM-count — capture-safe. Byte-identical golden lock PASS. Coordinator re-validated GPU0 (33.06 s). Admin-merged (squash, 2026-08-19). Opt-out: `ONNX_GENAI_GEMV_PIPE_WELLOCC=0`.
- **End-to-end result:** +1.3–1.6 tok/s (~242→243.5 tok/s). Grid-starved projections at floor; split-K barred (fp32 reduction reorder → not byte-identical).
- **4-win session arc:** #1486 (+8.9%) → #1495 (+4.36%) → #1496 (+3.4%) → #1501 (+1.3–1.6 tok/s). Native fp16io decode now ~243.5 tok/s vs ORT eager ~290 (−16% gap; ORT ~1.19× ahead).

## 2026-08-19T20:30Z — LinearAttention warp-cooperative rewrite (gaff-16, PR #1503)

- **Warp-cooperative kernel rewrite:** Replaced per-thread `sc[MAX_D_K=256]` local-memory state array with warp-cooperative layout (one warp per state column, d_k rows in lane registers). Replaced serial 128-iter d_k loops with `__shfl_xor` warp reductions. Local-load sectors 163,840 → 0, local-store 98,304 → 0. Kernel −42% (21,664 → 12,510 ns), SM util 1.31% → 13.3%, occupancy 12.0% → 21.2%.
- **End-to-end:** +7.4–10.6 tok/s (~240 → ~249 tok/s, +3.2%). Native ~249 vs ORT ~290 (~1.16× ORT-ahead). Coordinator re-validated golden lock GPU0 (35.01 s). Admin-merged (squash, 2026-08-19). Opt-out: `ONNX_GENAI_CUDA_DISABLE_LINATTN_WARP_COOP=1`. Worktrees swept.
- **CRITICAL FINDING — lock tests assert token-IDs, NOT bit-exact bytes:** `qwen35_*_text_decode_lock` assert `Vec<u32>` argmax. Warp-shuffle reduction reorder is ULP-divergent but argmax-stable. qwen3.5-0.8b + qwen3.5-2b PASS, oracle-parity suite PASS (all update rules, d_k=128/96/130). **De-risk: future ULP-divergent kernel rewrites (warp reductions, reduction-order changes) are not barred by the lock gate — validate ≥2 models for argmax stability, not bit-identity.**
- **5-win session arc complete (#1486/#1495/#1496/#1501/#1503):** ~214 → ~249 tok/s (+16.4%). Remaining gap to ORT is per-layer launch/fusion structure — diminishing-returns territory.

## 2026-08-19T21:15Z — Shuffle-fusion honest floor (gaff-16, PR #1511 closed)

Attempted CUDA no-op Transpose zero-copy views for the data-shuffle lever; no code shipped. The view path fired but mutates buffer lifetimes mid-capture, aborting graph recording and fragmenting decode capture 16→70 segments. `movement.rs` was reverted and docs-only PR #1511 was closed. Reusable finding: captured decode already amortizes tiny movement kernels; attribute with captured-replay profiling, not eager kernel counts.


## 2026-08-20T00:15Z — Symmetric int8 GEMV split-K grid-fill (PR #1516)

Landed the final GEMV lever: symmetric int8 `MatMulNBits` now uses split-K for grid-starved decode projections, general and capture-safe, opt-out `ONNX_GENAI_CUDA_DISABLE_INT8_SYMMETRIC_SPLITK=1`. Bias-fusion was honestly a no-op (already on main; Qwen3.5 exports bias-free). Result: qwen3.5-0.8b-hybrid fp16io **254.0 → 263.8 tok/s** (+3.9%, +9.8 tok/s); ncu N=1024 **11424→8049 ns** and N=2048 **6065→4877 ns** with occupancy roughly doubled. Golden token-ID locks passed on 0.8b and 2b; PR #1516 merged at `ec695d897`.
