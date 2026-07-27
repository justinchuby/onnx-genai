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

**Observability:** The selected path is queryable via `decode_path_label()` → `"spmd-pool"`, `"adaptive"`, `"flat"`, or `"unresolved"`. Diagnostic prints are gated behind `NXRT_CALIB_DEBUG` env var (no unconditional stderr output from the library).

**Fallback:** Single-core hosts (cpuset=1) and `THREADS=0` fall back to the flat path with a diagnostic. The `P-1` worker formula produces `max(1,0)=1` on a 1-P-core host.
