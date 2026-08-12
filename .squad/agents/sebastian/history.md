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

## 2026-08-11 — PR #762 third corrective wave: BL1 runtime axis resolution

**Task:** BL1 fix — `raw_axis` preserved, resolved at runtime against actual input rank; `build_subgraph_routing` emits `NodeInputSource::Absent`; `end_version` validation.

**Commit:** `168e40c3e`

- `ShapeInference::LayerNorm` stores `raw_axis: i64`; resolved per-invocation in `infer_shapes()` against actual input rank.
- Eliminates false resolution of `axis=-1` to index 0 on `[B, S, H]` inputs where B and S are symbolic (filter_map collapsed them).
- `ep.rs` emits `NodeInputSource::Absent` for None inputs in `build_subgraph_routing`.

**Outcome:** BL1 fix genuine. Challenger's review found residual `filter_map(|d| d.as_static())` at claim time in ep.rs still destroyed rank at two sites. Coco fixed.

**Lesson reinforced:** filter_map is wrong wherever position or rank is load-bearing; use map → Vec<Option<usize>>.

---

### 2026-08-11 — Adversarial review of onnxruntime #31988 (CUDA MatMulNBits SM-adaptive cols)

**Task:** Read-only rubber-duck review of draft PR adding `SelectColsPerBlock(n, sm_count)` to M=1 GEMV path.

**Findings:**
- Bit-identicality: **confirmed** by kernel trace — per-column reduction invariant to CTA width.
- Wide-n invariance: **confirmed** — threshold arithmetic sound, no boundary regression.
- No OOB risk (divisibility gate guards all paths).
- 3× template instantiation cost — flagged as BLOCKING for PR description.
- No perf data, no leaked internals — clean.
- Recommended PR exist as draft with methodology section.

**Output:** `.squad/decisions/inbox/sebastian-review-31988.md`

## 2026-08-12 — PR #31988 review (adversarial, read-only)

- Confirmed bit-identicality by kernel trace.
- Confirmed wide-n invariance.
- Flagged template instantiation cost (24 → 72) as BLOCKING for PR description — this was a correct finding.
- Declared `n % 8 != 0` path "SAFE" — **this was wrong.** `n=12` with `SelectColsPerBlock` returning 4 would have been newly accepted by the M=1 GEMV, changing shape routing.
- Also claimed benchmark methodology was missing — this was incorrect; it was present in the PR body.
- Barred from revision under reviewer lockout. Chew revised and confirmed the routing concern was real.
- Lesson: a reviewer's "SAFE" is not proof; verify the load-bearing claim independently.

### 2026-08-12 — PR #31973: Fix architecture-specific dispatch threshold in LayerNorm tests

- **Blocker fixed:** `kAvx2DispatchThreshold = 8` was applied universally; RISC-V RVV dispatches for N < 8. Renamed to `kKernelDispatchThreshold` with `#if` guard matching production `layernorm.cpp`. Test and production share the same preprocessor condition.
- **CatastrophicCancellationPasses:** Added scenarios with condition < 1e7 so the accuracy body is exercised (both prior scenarios had condition = 1e9, making accuracy unreachable).
- **AdversarialPrecisionReport:** Marked `DISABLED_` to match its comment — it's a measurement tool, not a gate.
- **Benchmark:** Removed N=7 (below x86 threshold; was timing the fallback, not the kernel).
- **Denormals/LargeMagnitudes:** Clarified these are finiteness-only checks, not accuracy checks.
- **MlasLayerNormF32 doc:** Updated to describe the x86 dispatch threshold.
- **Optional perf (8-float mean pattern):** Deferred — risks complicating review of the blocker fix.
- Tests: 41 pass + 2 disabled (baseline was 42 + 1; AdversarialPrecisionReport moved to DISABLED). 43 pass with `--gtest_also_run_disabled_tests`.
- Head: `72e02cd92c`. PR stays draft; needs Opus re-approval.

## 2026-08-12 — PR #31973 v2: architecture-specific dispatch threshold fix (blocker)

- Renamed `kAvx2DispatchThreshold` → `kKernelDispatchThreshold`; made it `8` under `#if defined(MLAS_TARGET_AMD64) || defined(MLAS_TARGET_IX86)`, else `1`, matching production `layernorm.cpp` exactly.
- Fixed `CatastrophicCancellation`: added scenarios with condition 1e4 and 1e5 so accuracy branch is reachable (prior scenarios had condition 1e9 — unreachable).
- Renamed `AdversarialPrecisionReport` → `DISABLED_AdversarialPrecisionReport`.
- Removed N=7 benchmark (below x86 threshold; was timing the scalar fallback).
- Clarified `TestDenormals`/`TestLargeMagnitudes` as finiteness-only.
- Updated `MlasLayerNormF32` docstring to describe x86 threshold.
- Deferred 8-float mean-pass optimisation to follow-up PR.
- Head `72e02cd92c`. Tests: 41 passed + 2 disabled; 43 with disabled.
