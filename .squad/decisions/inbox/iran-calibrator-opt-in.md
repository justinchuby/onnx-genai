### 2026-07-27: Made load-adaptive path selection opt-in (Iran)

**By:** Iran (Mac CPU Optimization Engineer)
**Directive from:** Justin Chu (via coordinator, `coordinator-calibrator-opt-in.md`)
**PR:** #227 (`squad/mac-cpu-ep-roofline`)

**What changed:**

The `ONNX_GENAI_CPU_DECODE_PERSISTENT_POOL` env var semantics changed:

| Value | Before (old) | After (new) |
|---|---|---|
| unset (default) | `Auto` — calibrator probes both paths, defaults to flat | `On` — persistent SPMD pool, deterministic, no probing |
| `=1` | `Forced` — always pool | `On` — same as default (explicit) |
| `=0` | `Off` — always flat | `Off` — always flat (unchanged) |
| `=auto` | *(not recognized)* | `Adaptive` — opt-in calibrator, same logic as old `Auto` |

**Why:**

1. The default was unpredictable: under host load, the calibrator silently selected the flat path, halving throughput with no user indication.
2. Different paths use different FP reduction orders, making greedy decode non-reproducible across load conditions.
3. The calibrator itself was bitten during this campaign — it mis-sampled under agent load and produced a false Fact Checker verdict.

A library should be predictable by default, adaptive on request.

**Measurements (M1 Max, FP16 Qwen2.5-0.5B, post GEMV dispatch fix):**

| Condition | Default (pool) | Adaptive (`=auto`) | Flat (`=0`) | ORT |
|---|---|---|---|---|
| Quiet (load ~4-5) | 53.35 tok/s | 56.10 tok/s | 42.84 tok/s | 42.19 tok/s |
| 4×`yes` load (~10) | 18.96 tok/s | 31.95 tok/s | 31.57 tok/s | 37.76 tok/s |

Under moderate load (4 contending cores), the pool degrades ~2× more than flat because its pinned workers compete with load processes. This is the accepted tradeoff for predictability and reproducibility. Users who need adaptation set `=auto`.

**Observability:** The selected path is queryable via `decode_path_label()` → `"spmd-pool"`, `"adaptive"`, `"flat"`, or `"unresolved"`. With the `tracing` feature enabled, path selection is emitted as `tracing::debug!(path = "spmd-pool", workers = 9, "cpu decode path selected")` per `docs/ERROR_AND_LOGGING_CONVENTIONS.md`. Without the feature, `NXRT_CALIB_DEBUG` gated eprintln serves as fallback.

**half_gemm.rs overlap analysis (2026-07-27):**

Sebastian's `half_gemm.rs` (GEMM, M>1) and my FP16 GEMV (M=1 decode) are complementary:
- **Conversion helpers:** Not duplicated. `half_gemm.rs::widen_f16_neon` uses `vcvt_f32_f16` intrinsic (requires FEAT_FP16 runtime detection), bulk-widening into pre-packed panels. My `load_f16x4_to_f32x4` uses inline asm `fcvtl` (ARMv8 base, no FEAT_FP16 needed), widening within the FMA inner loop. Different APIs, different feature requirements, different use patterns.
- **Dispatch:** Fixed in `ed7a65e3` — GEMV check runs before `try_matmul_half` so M=1 f16 goes to the bandwidth-optimal GEMV, M>1 to half_gemm. This is now deliberate.
- **Superseding:** Neither supersedes the other. GEMV is bandwidth-optimal for M=1 decode; GEMM with panel packing is compute-optimal for M>1. The `ExecutionPath` runtime dispatch pattern is cleaner than compile-time `#[cfg]` but adds overhead to the hottest decode path for no benefit at M=1.
- **Consolidation recommendation:** Defer to a separate PR. Unifying the two widening approaches would need careful handling of the FEAT_FP16 vs ARMv8-base distinction and is a refactor, not a correctness issue.

**Fallback:** Single-core hosts (cpuset=1) and `THREADS=0` fall back to the flat path with a diagnostic. The `P-1` worker formula produces `max(1,0)=1` on a 1-P-core host.
