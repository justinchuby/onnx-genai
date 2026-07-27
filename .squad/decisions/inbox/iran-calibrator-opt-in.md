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

**Measurements (M1 Max, FP16 Qwen2.5-0.5B):**

| Condition | Default (pool) | Old default (would have picked) |
|---|---|---|
| Quiet host | 43.75 tok/s | 43.75 tok/s (pool) |
| 8x load | 3.09 tok/s | ~13 tok/s (flat, via calibrator) |

The pool is worse under heavy load — this is the accepted tradeoff for predictability. Users who need adaptation set `=auto`.

**Observability:** The selected path is logged once at pool build time via `eprintln!`, e.g.:
```
onnx-genai: decode path = persistent SPMD pool (default). Set ...=auto for load-adaptive selection, =0 for the flat legacy path
```

**Fallback:** Single-core hosts (cpuset=1) and `THREADS=0` fall back to the flat path with a diagnostic. The `P-1` worker formula produces `max(1,0)=1` on a 1-P-core host.
