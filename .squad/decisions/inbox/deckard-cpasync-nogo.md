# Decision drop — cp.async weight prefetch on the M=1 int4 GEMV: measured NO-GO (fresh data)

**Author:** Deckard (Systems Dev, CUDA/decode-perf)
**Branch:** `squad/int4-gemv-cpasync-v2` (off main `3f826936`)
**Date:** 2026-08-15
**Status:** DRAFT PR — reproducible NEGATIVE result, default OFF, byte-identical
**Refs:** TRT-LLM port plan PORT-3; prior no-go `2131046e` (narrow kernel); #986/#992/#996

## TL;DR
The sota/TRT-LLM roadmap flagged **cp.async double-buffered weight prefetch +
`L2::evict_first`** as the #1 lever (15–30%, byte-identical). I built it on the
current multicol NC=4 GEMV and **measured it: NO-GO, −4.2% glm decode**. It is
byte-identical and it *does* hide load latency as designed — but it is
occupancy-negative, which dominates at M=1. **Recommend dropping PORT-3 (and
PORT-5 PDL, same class).** Fresh data + mechanism below.

## Why I re-tested (the prior no-go was on a different kernel)
cp.async was already a measured no-go once (`2131046e`, follow-on to #978) — but
on the **narrow** pre-multicol kernel, which was register/warp-bound at ~80%
occupancy and 8–12% DRAM. The current kernel is very different, so I re-profiled
first. Fresh ncu on the **current default fp32 multicol** gate_up GEMV (glm,
`--graph-profiling node`, launch-skip 240):

| stall reason | warps stalled / issue-active |
|---|---|
| **Long Scoreboard (global-load latency)** | **2.35** ← cp.async's target, now #1 |
| Wait (math/FMA latency) | 1.80 |
| Short Scoreboard (smem/L1) | 0.20 |
| No instruction | 0.15 |

Long-Scoreboard IS now the top stall (precondition for cp.async **met**, unlike
the prior no-go) — so a genuine re-test was warranted.

## What I built
`matmul_nbits_gemv_f16_general_bs_wide_multicol_cpasync`: double-buffers each
lane's WIDE_NC=4 128-bit weight words through shared memory with
`cp.async.cg.shared.global.L2::evict_first` (the Marlin `cp_async4_stream`
idiom, 16-byte), prefetching K-tile k+1 while tile k computes. Exact per-lane
depth striding + per-column fp32 accumulation order preserved ⇒ byte-identical.
SM80+ with a byte-identical direct-load `#else` fallback (Rule 11). Gate
`use_gemv_cpasync(cc)`: SM80+, int4/block≠32 multicol only, **DEFAULT OFF**,
opt-in `ONNX_GENAI_GEMV_CPASYNC=1`.

## Result — NO-GO
**glm decode: 199.9 → 191.6 tok/s (−4.2%)**, greedy tokens BYTE-IDENTICAL
(0 diffs / 160 tokens, generic + repetitive). qwen unaffected (block-32 never
routes here). f64 numerics 7/7 (default path unchanged). fmt + clippy clean.

### ncu — cp.async worked, but is occupancy-negative
| metric | direct-load (default) | **cp.async** | delta |
|---|---|---|---|
| kernel time | 33.9 µs | **40.2 µs** | **+18%** |
| DRAM | 1.78 TB/s | 1.50 TB/s | −16% |
| **Long Scoreboard** | 2.35 | **1.88** | **−20% (hid load latency ✓)** |
| Wait (math latency) | 1.80 | 2.07 | +15% |
| registers/thread | 64 | **71** | +7 |
| **achieved occupancy** | 41.6% | **33.3%** | **−8 pp** |

**Mechanism:** cp.async did exactly what it promises — it cut the global-load
stall 20%. But the `cvta`-to-shared address + pipeline bookkeeping raised
registers 64→71, dropping occupancy 41.6%→33.3%. At M=1 this GEMV hides load
latency **primarily via warp-level parallelism (occupancy)**, not per-warp
prefetch depth; losing 8 pp of occupancy removes more latency-hiding than the
prefetch adds. Net: kernel +18%, DRAM −16%, e2e −4.2%. The short (~4-trip) K-loop
and the kernel's existing 4-way weight ILP (all WIDE_NC loads issued up front)
leave little for a software pipeline to add.

This is the **third convergent prefetch negative**: (1) register depth-4 prefetch
(occupancy loss), (2) prior cp.async on the narrow kernel, (3) this multicol
cp.async. **The M=1 decode GEMV is occupancy-bound, not prefetch-bound.**

## Recommendation on the roadmap
- **PORT-3 (cp.async prefetch): DROP.** Disproven with fresh, mechanistically
  explained data on the current kernel.
- **PORT-5 (PDL / programmatic launch): deprioritize.** Same latency-hiding class;
  the kernel isn't launch-bubble bound (issue-active 69.5%).
- **The real lever is OCCUPANCY**, which is register-bound at 64 (WIDE_NC=4).
  Anything that cuts the kernel's register/instruction footprint *without* losing
  the multicol activation reuse would raise occupancy → more warp-parallel latency
  hiding. That points at **PORT-1** (offline bias+interleave → cheaper runtime
  dequant, fewer instrs/regs) as the better byte-identical next lever — but PORT-1
  needs an **offline weight-layout repack + a versioned layout flag** (loader-side,
  model-artifact-compat risk), a much larger surface than a pure kernel change.
- **Strategic honesty:** per #996's measurements, this GEMV is already at ~95% of
  ORT's DRAM (2.29 vs 2.42 TB/s) and near ORT's kernel time (fp16 26.4 vs 24.9 µs).
  The remaining e2e gap to ORT ~250 base is dominated by **other ops (lm_head)**,
  owned separately by deckard-4. The banked honest wins to beat ORT base are the
  **fp16 GEMV (#996, +6.7%)** + **lm_head cuBLASLt**, not further GEMV prefetch.

## Ask
- Coordinator: accept the NO-GO; decide whether to **merge this as a documented
  negative** (default-off, like `2131046e`) or take the decision drop and close
  the PR. Either is fine — no regression ships by default.
- Next: your call between (a) me scoping **PORT-1** (offline repack + layout flag,
  the occupancy lever) vs (b) me landing #996 fp16 default-on after Chew's gate and
  handing the remaining gap to the lm_head work.
