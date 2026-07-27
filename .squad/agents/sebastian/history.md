# Sebastian — History

## Project Context (joined day)
- **Project:** onnx-genai — Rust inference runtime for generative AI on ONNX Runtime.
- **State when joined:** Phases 1-4 done; tool use/grammar/chat-template; Qwen2.5-0.5B runs; Hermes agent E2E; long-context O(1)/token via static-cache in-place KV. Working on DESIGN §26 batched serving + reviews.
- **Requested by:** Justin Chu
- **Joined:** 2026-07-12

## 2026-07-12T13:14:00-07:00 — Performance review merged
Sebastian's perf review is now in decisions. §26 should prioritize active-row compaction, ORT KV as hot source of truth, fewer per-step allocations, direct/borrowed logits access, and explicit snapshot/import/export for paged KV.

## 2026-07-12T13:52:00-07:00 — §26 Stage A/B complete
- Sebastian delivered `Engine::generate_batched_static` and `ContinuousBatchManager`; fixed batched static-cache generation matches individual runs and measured 6.2x throughput on the tiny fixture.
- Future scheduler/perf work should preserve the `submit`/`step`/`poll` contract and use Deckard's active-row compaction when rows finish or new requests are admitted.

## 2026-07-12T14:28:00-07:00 — Batched-test ORT determinism fixed
- Sebastian added `SessionOptions::with_intra_op_threads` and `Engine::from_dir_with_session_options` so correctness tests can force single-thread ORT execution.
- Batched static-cache exact-equality tests now use `intra_op_threads=1`, eliminating reduction-order FP tie flakes while production defaults remain unchanged.
- Preserve this convention for future real-model exact-equality tests.

## 2026-07-12T16:14:00-07:00 — Benchmark and observability contracts logged
- `onnx-genai-bench` and `scripts/run_benchmarks.sh` are canonical for device-comparable Criterion runs; preserve stable scenario names and machine metadata.
- Observability core is canonical: atomic metrics, `/metrics`, `/v1/status`, request spans, trace IDs, driver/session/token/TTFT/latency/cache-hit/429 counters.
- Perfetto, OTLP, and full debug endpoints remain future work.

## 2026-07-12T17:30:00-07:00 — Audio DSP and cross-runtime benchmarks logged
- Native Whisper log-mel preprocessing and the OpenAI HTTP cross-runtime benchmark harness are canonical.
- True 1:1 GGUF benchmarking remains in progress and was intentionally not logged as complete.

## 2026-07-12T21:35:00-07:00 — H200 runbook + CPU decode profile
- Wrote `docs/benchmarks/H200-CUDA-runbook.md`: full build/run/benchmark procedure for the CUDA path on H200, assembled from Leon's CUDA-EP flags and Sapper's stacked CUDA model, with a coherence gate (Hopper/ORT garbled-token caveat), checklist, and troubleshooting.
- Profiled CPU decode: **98.9% of per-token time is ORT `session.run`**; orchestration ~1%. Gap is ORT-kernel-bound, not ours. CPU-vs-CPU (same GGUF): ours 43.6 vs LM Studio CPU 157 tok/s (~3.6x).
- Biggest addressable lever: fixed model ships a **544 MB fp32 `lm_head` MatMul** every token (~23% of per-token cost) — quantize embedding+head in Mobius (GatherBlockQuantized) like the CUDA stacked model.
- Added env-gated profiler (`ONNX_GENAI_PROFILE`) + `profile_decode` harness; added `ONNX_GENAI_INTRA_OP_THREADS` override (M1 Max: 6-8 perf cores optimal, 10 threads ~2x slower). Decision in `.squad/decisions/inbox/sebastian-cpu-profile.md`. Did NOT commit.

## 2026-07-13T07:12:00-07:00 — Foundry Local model-vs-runtime isolation (DECISIVE: parity, not FL win)
- Downloaded FL's exact CPU model `qwen2.5-0.5b-instruct-generic-cpu:4` (SHA `997228…cd21`, byte-identical to the 07-12 bench) and ran it through OUR CPU runtime.
- **Decisive result: decode PARITY.** OURS-on-FL-model ~215 tok/s ≈ OURS-on-our-model ~206 ≈ FL-on-FL-model ~200-212. Warm HTTP: short 211.8 (ours) vs 212.1 (FL); long **175.0 (ours) vs 159.8 (FL) — we lead** after the fp32-GQA shared-KV fix. The 07-12 "FL leads 202.7/165.8" gap was pre-KV-fix + thermal/under-warmed sampling (machine variance 85-216 tok/s unwarmed).
- **Graph diff:** FL fuses Q/K/V into one MatMulNBits (N=1152) → 121 MatMulNBits / 299 nodes vs our 169 / 394 (48 fewer dispatches/token). But decode is bandwidth-bound (M=1), so fused QKV is **decode-neutral** — measured neutral. Low priority for CPU decode; prefill-only.
- **Task B (FL C++):** FL sets **zero custom ORT SessionOptions** — delegates to onnxruntime-genai (`genai_model_instance.cc:29-58`); IO binding + `past_present_share_buffer` are inside that lib. Our runtime already matches (ORT_ENABLE_ALL, IO binding, shared-KV). No missing session option.
- **No code change** (none warranted). Follow-ups: warmup discipline + server startup priming (Leon/Seb), TTFT/prefill ~2-4% residual (Leon), fused-QKV low-prio (Sapper). Doc: `docs/benchmarks/2026-07-13-foundry-local-analysis.md`; decision inbox `sebastian-foundry-analysis.md`. Did NOT commit.

- 2026-07-14T19:05:00Z — DESIGN.md §26.11 Resource Governor merged in `d6736e1`, specifying live byte-denominated VRAM/RAM limits, transactional lowering, and actionable over-budget errors.

## 2026-07-15T00:00:00Z — Cross-agent session update

- Validated non-empty IR>=3 opset imports while preserving custom-only models; merged in the loader legality stack.

### 2026-07-16T00:00:00Z — Performance-and-design wave
Marked Gather non-capturable and fixed thread-count-aware MatMulNBits partitioning.

- 2026-07-16T00:00:01Z — 🟢 Approved Rachael's exact single-consumer `x * Sigmoid(x)`→SiLU fusion (`682c93d`); added multi-consumer non-fusion coverage in `d116a96`. Independent interleaved benchmark: 44.45→47.64 tok/s (+7.2%) with unchanged tokens.

### 2026-07-16T00:00:03Z — Safe decode-thread configuration fix
Revised the rejected decode-only Rayon pool with a pure `resolve_decode_threads(raw, available)` helper (`feea8e5`). Empty, invalid, zero, negative, and overflowing settings now retain default behavior; positive values cap at available parallelism. Holden cleared the change after 413 tests.

### 2026-07-16T00:00:00Z — nxrt Python Engine threading revision
Replaced Rachael's rejected `RefCell`/unsendable Python genai Engine with a sendable `Mutex<RustEngine>` wrapper (`41d8c31`). Engine work releases the GIL and `try_lock` makes concurrent or callback-reentrant access an actionable `RuntimeError`; Holden cleared the fix.

### 2026-07-16T00:00:00Z — GQA decode direct-write review
🟢 Cleared Leon's M=1 contiguous-f32 GQA writer (`1fdd1ec`): prefill, strided, and non-f32 outputs retain the generic writer; BSH/BNSH layouts, RoPE, KV behavior, and grouping are preserved. Independent profiling measured GQA 0.883→0.457 ms/step and throughput 51.58→59.42 tok/s with exact eight-token output; 413 CPU EP tests passed.

## 2026-07-16T00:00:00Z — CUDA M2 packed-GQA review cycle
- 🔴 Rejected Roy's initial packed-GQA artifact for bypassing real packed prefill and failing unsupported-PTX validation; strict lockout enforced.
- 🟢 Cleared Wallace's repaired `4a34c66`: real packed-prefill→aliased-decode coverage, shared SM90 CUBIN fallback, 6/6 GQA and 114/114 CUDA tests passing.

## 2026-07-16T14:20:00Z — M3 device-resident CUDA KV review
- 🟢 Cleared Roy's `398c536`: 48 persistent aliased K/V buffers remain stable and make no KV host transfers. M2 and M3 CUDA streams are byte-identical; the CPU mismatch starts at index 10 and is a pre-existing numerical-drift follow-up.

## 2026-07-16T15:39:27Z — Scribe session update

- Fixed `onnx-runtime-python` `onnx_type_string` exhaustiveness for Undefined/Complex64/Complex128 (`f058594`); this main commit includes the completed onnx-rs full-spec merge.

- 2026-07-21: Scribe reconciled the perf campaign inbox; key decisions are now consolidated in `.squad/decisions.md` under the 2026-07-21 perf campaign section.

## 2026-07-21T05:40:00Z — fp16 decode and cross-platform reconciliation

- Integrated the end-to-end fp16 native CUDA decode path (`c8741ba`): coherent H200 Qwen output at about 344 tok/s with zero CUDA-graph fallbacks; f32 remained near 200 tok/s. Holden approved.


## 2026-07-21 — Wave-2 and CI milestone
CI now covers all 27 offline crates with warnings-as-errors and native Windows ARM64. Capture-safe native fp16 CUDA decode wave 2 stacked GQA prep fusion, warp-shuffle RMSNorm, and specialized down-projection GEMV on wave 1, reaching 663–672 tok/s on H200 versus ORT GenAI at 657, with zero fallbacks. All CUDA EP kernel work must remain correct and fast across supported SM architectures, not only sm_90.

## 2026-07-21T11:15:00Z — Wave-3 long-context GQA
- Raised capture-safe fp16 GQA `MAX_SPLITS` 8→16; Holden approved and `3b972bf` merged. Independent H200 review measured about 647→693 tok/s at 1024 tokens (+7.1%), flat at 256, with identical tokens and zero fallbacks.

- 2026-07-22T23:20:00Z — Revised the rejected persistent SPMD pool under lockout; `cee3c20` added real 31-worker parity, precedence diagnostics, and panic-safe poisoning, then merged after approval.

## 2026-07-26T20:00:00Z — Scribe update

- 2026-07-26T20:00:04Z — Repaired PR #203 coverage under lockout by changing the split-K numeric test to `n=1152`, exercising `matmul_nbits_gemv_f16_scales_f16_splitk`.

## 2026-07-27T07:55:00-07:00 — MLAS vs Native CPU EP Strategy Analysis

Delivered decision-grade brief (`sebastian-mlas-vs-native-strategy.md`) answering Justin's question: **can the native CPU EP stand alone on Apple Silicon?**

**Key findings (all measured on M1 Max, dual-corroborated):**

1. **Q5 (strategy): YES, stand alone.** Decode → our NEON f16 GEMV (88–100 GB/s, 1.42× over ORT). Prefill → Accelerate/AMX (900–2100 GFLOPS, 50–100× over NEON-only). ORT cannot reach AMX (no Accelerate linkage). Under this split we never need MLAS on Mac.

2. **Q1 (fp16 fragility): FRAGILE.** Our 1.42× fp16 decode lead rests on ORT's `FuseFp16InitializerToFp32NodeTransformer` widening fp16→fp32 before MLAS hgemm sees it. MLAS has a fully-wired hgemm path (`MlasHGemmDispatchNeon`, `MLAS_HGEMM_DATA_PARAMS`). Upstream fix is one graph-optimizer config change. If fixed, our residual advantage shrinks to ~10–20%. **Real moat is AMX for prefill, not fp16 for decode.**

3. **Q2 (AMX threshold): No lower threshold.** AMX pays off at M=10 (170+ GFLOPS vs NEON's 21 GFLOPS). Crossover is exactly M=1 (decode) vs M≥2 (prefill). No hybrid needed. Predicted TTFT with Accelerate: ~47 ms at M=40 vs ORT's 107 ms.

4. **Q3 (MLAS porting): Skip.** Accelerate makes MLAS's NEON GEMM/packing/tiling moot for compute-bound work. Our NEON GEMV suffices for bandwidth-bound decode. KleidiAI GEMV offers ~5–15% marginal improvement at high FFI maintenance cost. Only potential future interest: quantized GEMM for int4 models.

5. **Q4 (8% f32 gap): Concede.** Gap is dispatch overhead (435 vs ~300 ops/token) + kernel micro-optimization. Irrelevant when Mac ships fp16 as default (we lead 42% there).

**No code changes made** (per rules — Iran, Deckard, and Pris are pushing to the active branch).

## 2026-07-27T08:09:00-07:00 — Q6: Vendoring ARM MLAS verdict (NOT WORTH IT)

Added Q6 to the strategy brief responding to Justin's proposal to vendor MLAS's ARM kernels and default-enable the `mlas` feature.

**Decisive finding: MLAS's ARM GEMV kernel is tied with ours.** Head-to-head microbenchmark (isolated kernel, no graph fusion confound) at Qwen2.5-0.5B decode shapes:
- Run 1 (load 24.9): MLAS/ours = 1.05×
- Run 2 (load 7.0): MLAS/ours = 1.00×

The 8% ORT gap is ~60% graph-fusion (dispatch overhead) and ~40% kernel quality. Vendoring MLAS addresses only the smaller half. Accelerate makes MLAS's GEMM irrelevant for prefill (8–15× faster).

**KleidiAI is load-bearing** — ORT's ARM perf comes from KleidiAI microkernels, not just MLAS assembly. Vendoring MLAS without KleidiAI would not reproduce ORT's speed. KleidiAI headers are NOT in our vendor snapshot — would need separate vendor drop. Both MIT-licensed.

**Strategy ranking:**
1. Ours + Accelerate (already implemented) — best
2. Option 1 + graph-level op fusion — second best, addresses actual gap source
3. Vendor ARM MLAS (Justin's proposal) — not worth it; 0–5% gain vs high cost

## 2026-07-27T13:12:20+00:00 — Roadmap wave-5

- PR #265 for #58 merged after Hicks approved runtime-dispatched AVX2/F16C and NEON f16/bf16 GEMM SIMD, including scalar/tail fallbacks and parity coverage.

## 2026-07-27T15:55:00+00:00 — half_gemm.rs analysis (Q6/Q1 update)

After main merge (e104664b), analyzed the new `half_gemm.rs` kernel (898 lines).

Key findings:
- **Architecture**: Blocked GEBP GEMM with MR=4, NR=8, KC=128, NC=64. Packs f16→f32 during panel creation. NEON microkernel uses separate vmulq+vaddq (not fmla — 45% headroom left). Rayon-parallelized over row blocks.
- **Estimated GFLOPS**: ~18–20/core single-threaded, ~100–160 multi-threaded (8 P-cores). Accelerate/AMX is 6–15× faster at all prefill shapes. MLAS hgemm would be ~1.3–1.7× faster on NEON (native fp16 = 2× element width), but both are irrelevant vs Accelerate.
- **⚠️ Dispatch bug found**: `try_matmul_half` (matmul.rs:488) fires for fully-fp16 models at ALL M values, including M=1. It intercepts the optimized `neon_gemv_f16_col_parallel` path. At M=1, half_gemm is single-threaded (1 row block) and packs f16→f32→f32 GEMM, whereas the GEMV reads f16 directly with multi-threaded column parallelism. Estimated 4–8× slower for M=1 decode. Flagged to Iran.
- **Strategic impact**: Strengthens anti-vendoring argument (we now have portable f16 GEMM for non-Mac ARM). Q1 FP16 moat sharpened: even if ORT fixes routing, our prefill moat via AMX grows. Q5 unchanged except dispatch fix needed.
- Updated decision brief with these findings.
