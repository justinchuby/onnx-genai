# Sebastian — History

## Project context
- Joined 2026-07-12 to cover DESIGN §26 batched serving, runtime/server performance, and cross-runtime benchmark analysis for `onnx-genai`.
- Durable conventions: preserve `submit`/`step`/`poll` for batching; force single-thread ORT for exact-equality real-model tests; use canonical benchmark/observability harnesses when comparing runtimes.

## Consolidated 2026-07-12 to 2026-07-23
- Delivered §26 Stage A/B (`Engine::generate_batched_static`, `ContinuousBatchManager`) and measured 6.2x tiny-fixture throughput; later design/resource-governor notes established byte-denominated VRAM/RAM limits and transactional lowering.
- Established benchmark and observability contracts: `onnx-genai-bench`, `scripts/run_benchmarks.sh`, atomic metrics, `/metrics`, `/v1/status`, spans, trace IDs, token/TTFT/latency/cache-hit/429 counters.
- CPU decode profiling showed ORT `session.run` dominates (~98.9%); fp32 `lm_head` quantization and op fusion were flagged as major levers.
- Foundry Local isolation proved decode parity with FL's exact CPU model; QKV fusion was decode-neutral/low priority, and no missing FL session option was found.
- Reviewed/cleared or recorded fixes for SiLU fusion, safe decode-thread config, GQA direct-write, CUDA packed-GQA repair, device-resident CUDA KV, Python Engine threading, full-spec onnx-rs, fp16 native CUDA decode, CI/warnings-as-errors, and wave-3 long-context GQA.
- Perf-campaign decisions through 2026-07-23 were consolidated in `.squad/decisions.md`.

## 2026-07-26T20:00:00Z — Scribe update
- Repaired PR #203 coverage under lockout by changing the split-K numeric test to `n=1152`, exercising `matmul_nbits_gemv_f16_scales_f16_splitk`.

## 2026-07-27T07:55:00-07:00 — MLAS vs Native CPU EP Strategy Analysis
- Delivered `sebastian-mlas-vs-native-strategy.md`: native CPU EP can stand alone on Apple Silicon; real moat is AMX/Accelerate prefill, not fp16 decode. MLAS hgemm could erode decode-only advantage, but Accelerate/AMX dominates prefill. KleidiAI/MLAS vendoring is low value relative to graph fusion and Accelerate.

## 2026-07-27T15:55:00+00:00 — half_gemm.rs analysis
- Found `half_gemm.rs` is a portable blocked f16 GEMM for non-Mac ARM, but on Mac prefill it is 15–25x slower than BNNS/AMX. Flagged a decode dispatch bug: `try_matmul_half` catches M=1 fp16 and can bypass the optimized GEMV path.

- PR #265 for #58 merged after Hicks approved runtime-dispatched AVX2/F16C and NEON f16/bf16 GEMM SIMD, including scalar/tail fallbacks and parity coverage.

### 2026-07-27 — CLI maintainer-tool backlog queued
Justin confirmed the onnx-genai CLI is a development/maintainer harness, not a consumer product. P0 CLI work in docs/research/cli/00-backlog.md is queued under that charter: live stats discoverability, structured maintainer output, batch/bench harnesses, explicit dev flags for engine behavior, and help snapshots/REPL help. Remote-client mode is out of scope.
## 2026-07-27T16:28:00+00:00 — BNNS fp16 AMX discovery
- Measured BNNS `BNNSMatMul` f16→f32 at 2000–2450 GFLOPS and established a simple threshold: M=1 uses GEMV; M≥2 on macOS should use BNNS/sgemm/Accelerate. BNNS is deprecated in macOS 15 but still works; BNNSGraph migration is future maintenance.

## 2026-07-27T16:48:00+00:00 — Batch decode correction
- Batch decode favors the native path at B≥2, but the earlier 15x-vs-ORT claim was retracted after measuring ORT at B=32 (~345 tok/s under high load). Current cautious estimate is ~4–5x pending same-load remeasurement. Commit `ad920725` title is retracted.

### 2026-07-27T13:10:00-07:00 — CLI backlog now on main
Scribe note: the CLI dev-tool charter and prioritized backlog from the merged CLI improvement track are now on main at `docs/research/cli/00-backlog.md`. Use that file as the source of truth before picking up queued CLI backlog work.

## 2026-07-27T02:00:00Z — Roadmap wave update
- Reviewed PR #303 / #59 and approved after byte-identical scheduled-vs-greedy/sequential parity plus mutation probes.
- 2026-07-28: Pointwise Conv microbench diagnosis was useful, but the initial 5.7-9.8x BNNS headline overstated real impact. Always convert per-layer speedups through Amdahl/model-level measurement before making campaign claims.
