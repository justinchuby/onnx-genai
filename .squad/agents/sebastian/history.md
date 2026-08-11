# Sebastian — History (compacted 2026-07-29)

**Role:** Owns DESIGN §26 batched serving, runtime/server performance, and cross-runtime benchmark analysis for `onnx-genai`. Preserve `submit`/`step`/`poll` batching semantics, force single-thread ORT for exact-equality real-model tests, and use canonical benchmark/observability harnesses for runtime comparisons.

## Durable lessons
- §26 Stage A/B delivered `Engine::generate_batched_static` and `ContinuousBatchManager`; later governor notes require byte-denominated VRAM/RAM limits and transactional lowering.
- Benchmark/observability contracts include `onnx-genai-bench`, `scripts/run_benchmarks.sh`, atomic metrics, `/metrics`, `/v1/status`, spans, trace IDs, token/TTFT/latency/cache-hit/429 counters.
- CPU decode profiling showed ORT `session.run` dominates (~98.9%); fp32 `lm_head` quantization and op fusion are major levers.
- Foundry Local isolation proved decode parity with FL's exact CPU model; QKV fusion was decode-neutral/low priority and no missing FL session option was found.
- PR #203 lockout repair changed the split-K numeric test to `n=1152` so it exercised `matmul_nbits_gemv_f16_scales_f16_splitk`.
- Native CPU EP can stand alone on Apple Silicon; the moat is AMX/Accelerate prefill, not fp16 decode. MLAS hgemm can erode decode-only advantage, while KleidiAI/MLAS vendoring is lower value than graph fusion and Accelerate.
- `half_gemm.rs` is portable for non-Mac ARM but 15–25× slower than BNNS/AMX on Mac prefill; `try_matmul_half` catching M=1 fp16 can bypass optimized GEMV.
- CLI is a development/maintainer harness, not a consumer product; use `docs/research/cli/00-backlog.md` as source of truth and keep remote-client mode out of scope.
- BNNS `BNNSMatMul` f16→f32 measured 2000–2450 GFLOPS; M=1 should use GEMV and M≥2 on macOS should use BNNS/sgemm/Accelerate. BNNS is deprecated in macOS 15 but still works.
- Retract batch-decode 15× claims unless same-load ORT confirms them; current cautious estimate was ~4–5× pending remeasurement.
- Convert pointwise/Conv per-layer speedups through Amdahl/model-level measurement before making campaign claims.

## Recent work (current wave, ~2026-07-28/29)
- 2026-07-28: Pointwise Conv microbench diagnosis was useful, but the initial 5.7–9.8× BNNS headline overstated real impact.
- 2026-07-28T17:40:00+0000: PR #362 merged (`5a079029`): If/Loop/Scan inference landed; #355 container typing remains deferred.

- 2026-08-11T03:47:00+0000: AVX2 LayerNorm/RMSNorm benchmark — measured against true scalar fp32 baseline (Welford's fallback from `layer_norm_impl.cc`). Found Pris's original fp64-reference numbers were conservative, not inflated. True speedup: LayerNorm 15–22× (algorithmic + SIMD), RMSNorm 3–4× (pure SIMD). Updated benchmark in `test_layernorm.cpp` to use correct baseline. Report: `.squad/decisions/inbox/sebastian-layernorm-benchmark.md`. Upstream PR #31973.

Full pre-compaction history in `history-archive.md`.

- 2026-08-11T13:30:00+0000: BL1/BL3 fixes for PR #762 (ep-plugin-parity-cuda).
  - **BL1:** LayerNorm axis no longer pre-resolved against truncated static shape. `ShapeInference::LayerNorm` stores `raw_axis: i64`, resolved at runtime in `infer_shapes` against actual input rank.
  - **BL3 carry-over:** `build_subgraph_routing` now emits `NodeInputSource::Absent` for `None` inputs instead of `Ort(0)`.
  - **Registry:** `build_ort_kernel_registry` validates `end_version >= since_version > 0`, collects per-entry failures in `RegistryBuildOutcome`, and surfaces them as actionable status in factory CreateEp.
  - **Test:** `layernorm_dynamic_axis.rs` — real ORT, dynamic `[B, S, H]`, axis=-1, asserts Mean/InvStdDev shape `[2, 3, 1]`. Fails pre-fix (axis resolves against `[4]` rank-1).
  - Test counts: 216 passed / 0 failed (baseline was 215).
